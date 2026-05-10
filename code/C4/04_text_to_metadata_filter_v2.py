import os
import json
from dotenv import load_dotenv
import re
import requests
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain.chains.query_constructor.base import AttributeInfo
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import logging

load_dotenv()
os.environ["HF_HUB_OFFLINE"] = "1"

logging.basicConfig(level=logging.INFO)

# 1. 初始化视频数据
video_urls = [
    "https://www.bilibili.com/video/BV1Bo4y1A7FU",
    "https://www.bilibili.com/video/BV1ug4y157xA",
    "https://www.bilibili.com/video/BV1yh411V7ge",
]

bili = []

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
        doc = Document(
            page_content="",
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

if not bili:
    print("没有成功加载任何视频，程序退出")
    exit()

# 2. 创建向量存储
embed_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
vectorstore = Chroma.from_documents(bili, embed_model)

# 3. 配置元数据字段信息
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
        description="视频长度（整数）",
        type="integer"
    )
]

# 4. 初始化LLM客户端（使用阿里云百炼 qwen3.6-plus）
llm = ChatOpenAI(
    model=os.getenv("MODEL", "qwen3.6-plus"),
    temperature=0,
    openai_api_key=os.getenv("DASHSCOPE_API_KEY"),
    openai_api_base=os.getenv("DASHSCOPE_BASE_URL"),
)

# 5. 获取所有文档用于排序
all_documents = vectorstore.similarity_search("", k=len(bili))

# 6. 执行查询示例
queries = [
    "时间最短的视频",
    "播放量最高的视频"
]

for query in queries:
    print(f"\n--- 原始查询: '{query}' ---")

    # 使用大模型将自然语言转换为排序指令
    prompt = f"""你是一个智能助手，请将用户的问题转换成一个用于排序视频的JSON指令。

你需要识别用户想要排序的字段和排序方向。
- 排序字段必须是 'view_count' (观看次数) 或 'length' (时长) 之一。
- 排序方向必须是 'asc' (升序) 或 'desc' (降序) 之一。

例如:
- '时间最短的视频' 或 '哪个视频时间最短' 应转换为 {{"sort_by": "length", "order": "asc"}}
- '播放量最高的视频' 或 '哪个视频最火' 应转换为 {{"sort_by": "view_count", "order": "desc"}}

请根据以下问题生成JSON指令:
原始问题: "{query}"

JSON指令:"""

    response = llm.invoke(
        [{"role": "user", "content": prompt}]
    )

    try:
        instruction_str = response.content
        instruction = json.loads(instruction_str)
        print(f"--- 生成的排序指令: {instruction} ---")

        sort_by = instruction.get('sort_by')
        order = instruction.get('order')

        if sort_by in ['length', 'view_count'] and order in ['asc', 'desc']:
            # 在代码中执行排序
            reverse_order = (order == 'desc')
            sorted_docs = sorted(all_documents, key=lambda doc: doc.metadata.get(sort_by, 0), reverse=reverse_order)

            # 获取排序后的第一个结果
            if sorted_docs:
                doc = sorted_docs[0]
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
                print("没有找到任何视频")
        else:
            print("生成的指令无效，无法执行排序")

    except (json.JSONDecodeError, KeyError) as e:
        print(f"解析或执行指令失败: {e}")
