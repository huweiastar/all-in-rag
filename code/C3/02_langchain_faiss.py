"""
基于 LangChain + FAISS 的向量存储与检索示例

FAISS（Facebook AI Similarity Search）是 Meta 开源的高性能向量检索库，
专为大规模密集向量设计。相比内存型的 InMemoryVectorStore，FAISS 的优势：
  - 支持上亿级向量，内存占用更小（使用量化、压缩等技术）
  - 支持多种索引类型（Flat、IVF、HNSW、PQ 等）
  - 可以持久化到磁盘，避免每次运行都重新编码

本示例完整演示了 RAG 中向量存储的核心生命周期：
  1. 创建嵌入模型
  2. 构建向量索引
  3. 保存到磁盘（持久化）
  4. 从磁盘加载
  5. 执行相似性搜索

HuggingFace 镜像设置：如果国内环境无法访问 HuggingFace 启用该设置
"""
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

# ── 第一步：准备文档和嵌入模型 ─────────────────────────────────────────────────

# 示例文本列表（实际项目中这里会是文档加载器加载的内容）
texts = [
    "张三是法外狂徒",
    "FAISS是一个用于高效相似性搜索和密集向量聚类的库。",
    "LangChain是一个用于开发由语言模型驱动的应用程序的框架。"
]

# 将纯文本列表转换为 LangChain 的 Document 对象列表
# Document 是 LangChain 中文档的标准表示，包含 page_content（文本）和 metadata（元数据）
# 这里只传了 page_content，metadata 为空，实际可以附加来源、页码、作者等信息
docs = [Document(page_content=t) for t in texts]

# 初始化嵌入模型
# HuggingFaceEmbeddings 会在首次使用时自动从 HuggingFace 下载模型（约 100MB）
# 下载后会缓存在 ~/.cache/huggingface/ 目录，后续运行直接使用缓存
# BGE-small-zh 输出 512 维向量，适合中文
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")

# ── 第二步：创建 FAISS 向量存储 ───────────────────────────────────────────────
# from_documents 内部执行流程：
#   1. 遍历每个 Document，调用 embeddings.embed_query() 将文本转为 512 维向量
#   2. 将所有向量存入 FAISS 的 Flat 索引（暴力搜索，精确但慢于近似搜索）
#   3. 记录向量与 Document 的映射关系（用于检索时回查原始文本）
#
# FAISS 支持的索引类型（通过 index_factory 参数指定）：
#   "IDFlat"  — 默认，精确最近邻搜索
#   "IVF256,Flat" — 倒排文件索引，适合大规模数据，速度更快但略牺牲精度
#   "HNSW32"  — 分层可导航小世界图，兼顾速度和精度
vectorstore = FAISS.from_documents(docs, embeddings)

# ── 第三步：持久化到磁盘 ──────────────────────────────────────────────────────
# 保存后会在目录下生成两个文件：
#   faiss_index Store/index.faiss — FAISS 索引二进制文件（包含所有向量）
#   faiss_index Store/index.pkl   — Pickle 序列化的元数据（Document 对象、映射关系等）
local_faiss_path = "./faiss_index_store"
vectorstore.save_local(local_faiss_path)

print(f"FAISS index has been saved to {local_faiss_path}")

# ── 第四步：从磁盘加载索引 ────────────────────────────────────────────────────
# 注意：
#   1. 加载时必须使用与创建时相同的嵌入模型（向量维度必须一致）：embeddings
#   2. allow_dangerous_deserialization=True 是必须的，因为 index.pkl 使用 Pickle 反序列化
#      该参数名为 "dangerous" 是因为 Pickle 可以执行任意代码，
#      但如果数据是你自己生成的（如本例），可以安全开启
loaded_vectorstore = FAISS.load_local(
    local_faiss_path,
    embeddings,
    allow_dangerous_deserialization=True
)

# ── 第五步：执行相似性搜索 ────────────────────────────────────────────────────
# similarity_search 的内部流程：
#   1. 将查询文本 "FAISS是做什么的？" 用相同的嵌入模型转为向量
#   2. 在 FAISS 索引中计算该向量与所有已存向量之间的余弦距离（或 L2 距离）
#   3. 按距离从近到远排序，返回前 k 个结果
#   4. 将对应的 Document 对象列表返回
query = "FAISS是做什么的？"
results = loaded_vectorstore.similarity_search(query, k=1)

print(f"\n查询: '{query}'")
print("相似度最高的文档:")
for doc in results:
    print(f"- {doc.page_content}")

# ── 补充：FAISS 的其他检索方式 ─────────────────────────────────────────────────
#
# 1. similarity_search_with_score(query, k) — 同时返回文本和距离分数
#    for doc, score in vectorstore.similarity_search_with_score(query, k=2):
#        print(f"距离: {score:.4f}, 内容: {doc.page_content}")
#
# 2. max_marginal_relevance_search(query, k, fetch_k, lambda_mult) — MMR 去重检索
#    在返回结果多样性时使用，避免 top-k 内容高度重复
#    results = vectorstore.max_marginal_relevance_search(query, k=3, fetch_k=20, lambda_mult=0.5)
#
# 3. 添加/删除向量 — 支持动态更新
#    vectorstore.add_documents([Document(page_content="新内容")])
#    vectorstore.delete(ids=["向量ID"])