"""
第二节：文本到元数据过滤器（Text-to-Metadata Filter）

本文件演示了如何使用 LangChain 的 SelfQueryRetriever（自查询检索器）
实现"用自然语言查询元数据"的功能。

核心思路：
  用户输入自然语言查询（如"时长大于600秒的视频"）
    → LLM 将其解析为结构化的元数据过滤条件（如 length > 600）
    → 向量数据库同时执行语义搜索 + 元数据过滤
    → 返回符合条件的文档

为什么需要这个技术？
  向量搜索擅长语义匹配（"找相关文档"），但不擅长精确过滤（"找2022年的文档"）。
  SelfQueryRetriever 把两者结合起来，让 RAG 系统既能理解语义，又能做精确筛选。
"""

import os
from dotenv import load_dotenv

load_dotenv()  # 加载项目根目录的 .env 文件

# 设置 HuggingFace 离线模式，避免每次加载都联网检查更新
os.environ["HF_HUB_OFFLINE"] = "1"

import re
import requests
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import logging

# 配置日志，打印 INFO 级别以上的信息，便于调试和观察内部执行过程
logging.basicConfig(level=logging.INFO)

# ============================================================
# 步骤1：加载B站视频数据并预处理元数据
# ============================================================

# 定义要加载的B站视频URL列表
video_urls = [
    "https://www.bilibili.com/video/BV1Bo4y1A7FU",
    "https://www.bilibili.com/video/BV1ug4y157xA",
    "https://www.bilibili.com/video/BV1yh411V7ge",
]

bili = []

# 【为什么不用 BiliBiliLoader？】
# LangChain 自带的 BiliBiliLoader 底层依赖 bilibili-api-python 库，
# 该库在处理 B站返回的 Brotli 压缩响应时会报错。
# 因此这里改用 requests 直接调 B站开放 API，通过设置 Accept-Encoding: gzip, deflate
# 来避开 Brotli 编码问题。
#
# B站获取视频信息的接口：
#   GET https://api.bilibili.com/x/web-interface/view?bvid=视频ID
# 返回 JSON 数据，包含标题、作者、观看数、时长等字段。
#
# 提取BV号的正则表达式
BV_PATTERN = re.compile(r"BV\w+")

for url in video_urls:
    bvid_match = BV_PATTERN.search(url)
    if not bvid_match:
        print(f"无法从URL中提取BV号: {url}")
        continue

    bvid = bvid_match.group()
    try:
        resp = requests.get(
            "https://api.bilibili.com/x/web-interface/view",
            params={"bvid": bvid},
            headers={
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
                # 关键：只接受 gzip/deflate 编码，避免 B站返回 Brotli 导致解码失败
                "Accept-Encoding": "gzip, deflate",
            },
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()

        if data.get("code") != 0:
            print(f"获取视频信息失败 [{bvid}]: {data.get('message')}")
            continue

        info = data["data"]

        # 构建 LangChain Document 对象
        #   page_content="" 为空：因为本节主题是元数据过滤，查询"时长大于600秒"
        #   纯粹是数值比较，跟文本语义完全无关，所以不需要填充视频字幕或简介。
        #   如果 page_content 填了字幕内容，嵌入模型会把它转成向量存入 Chroma，
        #   但在这个场景下，语义搜索那条通道不会被用到。
        #   metadata 存储扁平化的字段，供 SelfQueryRetriever 使用。
        doc = Document(
            page_content="",  # 【注意】这里为空 → 嵌入模型生成的向量是空文本的向量，实际不参与检索
            metadata={
                "title": info.get("title", "未知标题"),
                "author": info.get("owner", {}).get("name", "未知作者"),
                "source": info.get("bvid", "未知ID"),
                "view_count": info.get("stat", {}).get("view", 0),
                "length": info.get("duration", 0),
            },
        )
        bili.append(doc)
        print(f"已加载视频: {doc.metadata['title']}")

    except Exception as e:
        print(f"加载视频 {bvid} 时出错: {str(e)}")

# 如果没有任何视频加载成功，直接退出程序
if not bili:
    print("没有成功加载任何视频，程序退出")
    exit()

# ============================================================
# 步骤2：创建向量存储（Vector Store）
# ============================================================

# HuggingFaceEmbeddings 用于将文本转换为向量（数值表示）
# BAAI/bge-small-zh-v1.5 是智源研究院开源的中文向量模型，体积小、速度快，适合语义检索
#
# 【易混淆点】：这里的嵌入模型 和 上面的 LLM 是两个完全独立的通道！
#   - LLM（qwen3.6-plus）：负责"翻译"——把用户的自然语言（如"时长"）映射到元数据字段（length）
#   - 嵌入模型（bge-small-zh）：负责"语义搜索"——把 page_content 文本转成向量做相似度匹配
#   在本例中，因为 page_content=""（空文本），所以嵌入模型实际上没有参与核心检索过程，
#   真正起作用的是 LLM 的查询构造能力。
#
# 【三种检索方式对比】：
#   稀疏检索（Sparse）：基于精确词匹配（BM25/TF-IDF），不需要嵌入模型，擅长找"字面匹配的词"
#   密集检索（Dense）：基于语义向量相似度，需要嵌入模型，擅长找"意思相近的内容"
#   混合检索（Hybrid）：稀疏 + 密集加权融合，同时需要精确匹配和语义理解
embed_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")

# Chroma 是一个轻量级的本地向量数据库
# 【类比理解】：传统数据库（如MySQL）存表格按行查，Chroma 存向量按"语义相似度"查
#   核心能力：1) 语义搜索（找意思相近的内容） 2) 元数据过滤（按字段精确筛选） 3) 两者结合
#   适合本地开发和小规模数据，生产级（千万级以上向量）通常换 Milvus/Pinecone/Qdrant
# from_documents 做了两件事：
#   1. 用 embed_model 把每个视频文档的内容（page_content）转换成向量
#   2. 将这些向量及其对应的元数据（metadata）存入 Chroma，供后续检索使用
vectorstore = Chroma.from_documents(bili, embed_model)

# ============================================================
# 步骤3：配置元数据字段信息（告诉 LLM 有哪些字段可用）
# ============================================================

# 【这一步至关重要！】
# SelfQueryRetriever 需要知道文档有哪些元数据字段、每个字段是什么类型、含义是什么。
# AttributeInfo 就是这个"字段说明书"，LLM 会参考它来把用户的话翻译成过滤条件。
#
# 比如用户说"时长大于600秒"，LLM 看到 length 字段的 description 是"视频长度（整数）"，
# 就会理解到：应该把"时长"映射到 length 字段，用 > 比较，值是 600。
#
# 【易混淆点】：这个匹配是 LLM 做的，不是嵌入模型/语义检索做的！
#   LLM 天生理解中文里"时长" ≈ "长度" ≈ "视频有多长"，它靠 description 的文字描述
#   来建立映射关系。如果 description 只写了一个干巴巴的 "length"，
#   LLM 大概率就匹配不上。写清楚"视频长度（整数），单位为秒"才是让LLM正确映射的关键。
metadata_field_info = [
    AttributeInfo(
        name="title",
        description="视频标题（字符串）",
        type="string",
    ),
    AttributeInfo(
        name="author",
        description="视频作者（字符串）",
        type="string",
    ),
    AttributeInfo(
        name="view_count",
        description="视频观看次数（整数）",
        type="integer",
    ),
    AttributeInfo(
        name="length",
        description="视频长度（整数），单位为秒",
        type="integer"
    )
]

# ============================================================
# 步骤4：创建自查询检索器（SelfQueryRetriever）
# ============================================================

# 使用阿里云百炼的大语言模型作为"翻译器"（DashScope 接口兼容 OpenAI SDK）
# temperature=0 表示让模型输出完全确定、可复现的结果
#   → 在自查询场景中，同样的自然语言应该始终翻译成相同的过滤条件
#   → 如果 temperature 较高，可能导致同一句话每次翻译出的过滤条件不一样
llm = ChatOpenAI(
    model=os.getenv("MODEL", "qwen3.6-plus"),
    temperature=0,
    openai_api_key=os.getenv("DASHSCOPE_API_KEY"),  # 从环境变量读取API密钥
    openai_api_base=os.getenv("DASHSCOPE_BASE_URL")  # 阿里云百炼的兼容接口地址
)

# SelfQueryRetriever.from_llm 内部做了两件关键的事：
#   1. 创建一个"查询构造链"：利用 llm + document_contents + metadata_field_info，
#      将用户的自然语言查询转换为通用的结构化查询对象（Query）。
#      例如："时长大于600秒的视频" → Query(query='', filter=Comparison('length', 'gt', 600))
#   2. 匹配向量数据库的"翻译器"：根据使用的向量数据库类型（这里是 Chroma），
#      自动选择一个对应的翻译器，把通用查询对象翻译成 Chroma 能理解的原生过滤语法。

# ============================================================
# 【错误写法】只开启 enable_limit，未开启 enable_sort
# ============================================================
# 问题：当查询是"时间最短的视频"时，LLM 能解析出 limit=1（只返回1条），
# 但由于 enable_sort=False，LLM 无法生成排序条件（order=asc on length），
# 最终生成的查询是：query=' ' filter=None limit=1
# 结果：只是随机返回任意一条文档，而非按时长升序排序后取最短的那条。
#
# 例如：视频01=390秒、视频02=1063秒、视频03=806秒
# 正确结果应该是视频01（390秒），但实际返回了视频03（806秒），
# 因为它只是随机取了第一条，没有做升序排序。
#
# ============================================================
# # retriever = SelfQueryRetriever.from_llm(
# #     llm=llm,
# #     vectorstore=vectorstore,
# #     document_contents="记录视频标题、作者、观看次数等信息的视频元数据",
# #     metadata_field_info=metadata_field_info,
# #     enable_limit=True,     # ✅ 能解析数量限制
# #     enable_sort=False,     # ❌ 未开启排序 → "最短的视频"无法被解析为排序指令
# #     verbose=True
# # )
# ============================================================

# ============================================================
# 【正确写法】同时开启 enable_limit 和 enable_sort
# ============================================================
# 原因："时间最短的视频"包含两层含义：
#   1. 排序：按时长（length）升序排列（从小到大）  → 需要 enable_sort=True
#   2. 限制：只返回第1条（最短的那条）               → 需要 enable_limit=True
# 两者缺一不可。
#
# 开启后，LLM 生成的查询变为：query=' ' filter=None sort=[('length', 'asc')] limit=1
# 这样 Chroma 会先按 length 升序排序，再取第1条，正确返回视频01（390秒）。
# ============================================================
retriever = SelfQueryRetriever.from_llm(
    llm=llm,
    vectorstore=vectorstore,
    # document_contents 是对文档库内容的整体描述
    # LLM 会参考这段描述来理解文档库的主题，从而更准确地解析查询
    document_contents="记录视频标题、作者、观看次数等信息的视频元数据",
    metadata_field_info=metadata_field_info,
    # enable_limit=True 允许 LLM 解析查询中的数量限制，
    #   例如"最短的视频"会被理解为限制返回 1 条
    enable_limit=True,
    # enable_sort=True 允许 LLM 解析查询中的排序指令，
    #   例如"最短的视频"会被理解为按 length 字段升序排序
    enable_sort=True,
    # verbose=True 会在控制台打印 LLM 生成的查询结构，便于调试观察
    verbose=True
)

# ============================================================
# 步骤5：执行查询示例
# ============================================================

# 准备两个不同类型的查询来演示 SelfQueryRetriever 的能力：
#   - "时间最短的视频"：会被解析为按 length 升序排序 + 限制返回 1 条（用到 enable_limit）
#   - "时长大于600秒的视频"：会被解析为 length > 600 的比较过滤条件
queries = [
    "时间最短的视频",
    "时长大于600秒的视频"
]

for query in queries:
    print(f"\n--- 查询: '{query}' ---")
    # invoke 是 SelfQueryRetriever 的入口方法，内部完整链路如下：
    #
    # 第1步【LLM翻译】：将自然语言 → 结构化查询对象
    #   输入: "时长大于600秒的视频"
    #   LLM prompt 中包含了 metadata_field_info 的描述信息
    #   输出: Query(query='', filter=Comparison('length', 'gt', 600), sort=None, limit=None)
    #
    # 第2步【翻译器转换】：通用 Query 对象 → Chroma 原生过滤语法
    #   SelfQueryRetriever 自动匹配 Chroma 的内置翻译器
    #   将 Comparison('length', 'gt', 600) 转为 where={"length": {"$gt": 600}}
    #
    # 第3步【Chroma执行】：按过滤条件逐条判断
    #   视频01: length=390 → 390>600? ❌ 排除
    #   视频02: length=1063 → 1063>600? ✅ 保留
    #   视频03: length=806 → 806>600? ✅ 保留
    #
    # 第4步【返回结果】：返回 [doc02, doc03]，代码遍历打印输出
    #
    # 注意：嵌入模型（bge-small-zh）在这条链路中不参与，因为 page_content 为空，
    # 整个查询纯粹是 LLM 翻译 + Chroma 元数据过滤。
    results = retriever.invoke(query)
    if results:
        for doc in results:
            # 从文档的元数据中提取各字段，使用 .get 避免字段缺失报错
            title = doc.metadata.get('title', '未知标题')
            author = doc.metadata.get('author', '未知作者')
            view_count = doc.metadata.get('view_count', '未知')
            length = doc.metadata.get('length', '未知')
            print(f"标题: {title}")
            print(f"作者: {author}")
            print(f"观看次数: {view_count}")
            print(f"时长: {length}秒")
            print("="*50)
    else:
        print("未找到匹配的视频")