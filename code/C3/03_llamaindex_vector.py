"""
基于 LlamaIndex 的向量存储与检索示例（对比 LangChain + FAISS）

与 02_langchain_faiss.py 功能完全相同，但使用 LlamaIndex 实现。
对比要点：
  - LlamaIndex 用 Settings.embed_model 全局配置嵌入模型
  - 索引创建和持久化只需两行代码（from_documents + persist）
  - LlamaIndex 的 Document 使用 text 属性，LangChain 使用 page_content

HuggingFace 镜像设置：如果国内环境无法访问 HuggingFace 启用该设置
"""
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# ── 第一步：配置全局嵌入模型 ────────────────────────────────────────────────────
# Settings 是 LlamaIndex 的全局配置中心
# 设置 embed_model 后，索引构建和查询引擎都会自动使用这个嵌入模型
# 首次运行时会自动从 HuggingFace（或镜像）下载模型并缓存
Settings.embed_model = HuggingFaceEmbedding("BAAI/bge-small-zh-v1.5")

# ── 第二步：准备示例文档 ────────────────────────────────────────────────────────
texts = [
    "张三是法外狂徒",
    "LlamaIndex是一个用于构建和查询私有或领域特定数据的框架。",
    "它提供了数据连接、索引和查询接口等工具。"
]

# LlamaIndex 的 Document 使用 text 参数（LangChain 用 page_content）
# 两个框架的 Document 对象结构不同，但功能等价
docs = [Document(text=t) for t in texts]

# ── 第三步：构建向量索引 ────────────────────────────────────────────────────────
# from_documents 内部自动完成：
#   1. 使用 Settings.embed_model 将每个 Document 编码为向量
#   2. 构建内存中的向量索引（默认 SimpleVectorStore）
#   3. 建立向量与 Document 的映射关系
index = VectorStoreIndex.from_documents(docs)

# ── 第四步：持久化到磁盘 ────────────────────────────────────────────────────────
# persist 会将索引、嵌入向量和 Document 元数据保存到本地
# 生成的文件结构：
#   llamaindex_index_store/
#     ├── docstore.json    — Document 对象的 JSON 序列化（原始文本 + 元数据）
#     ├── index_store.json  — 向量索引的元数据
#     ├── graph_store.json  — 图存储（本例为空，因为没使用知识图谱）
#     ├── image_store.json  — 图片向量存储（本例为空）
#     └── vector_store.json — 向量存储（所有嵌入向量 + 相似度数据）
persist_path = "./llamaindex_index_store"
index.storage_context.persist(persist_dir=persist_path)
print(f"LlamaIndex 索引已保存至: {persist_path}")

# ── 第五步：从磁盘加载索引 ──────────────────────────────────────────────────────
# 加载流程：
#   1. 用 StorageContext.from_defaults 读取持久化目录中的所有 JSON 文件
#      它会反序列化 docstore（文档）、vector_store（向量）、index_store（索引元数据）
#   2. 用 load_index_from_storage 从 StorageContext 中重建索引对象
#      这一步会恢复向量与 Document 的映射关系
storage_context = StorageContext.from_defaults(persist_dir=persist_path)
loaded_index = load_index_from_storage(storage_context)

print(f"LlamaIndex 索引已从 {persist_path} 加载")

# ── 第六步：执行相似性搜索 ──────────────────────────────────────────────────────
# 使用 retriever 进行向量检索（底层等价于 FAISS 的 similarity_search）
# 内部流程：
#   1. 将查询文本用 Settings.embed_model 编码为向量
#   2. 在 vector_store 中计算与所有已存向量的余弦相似度
#   3. 返回最相似的 top_k 个 Node（LlamaIndex 中叫 Node，等价于 LangChain 的 Document）
retriever = loaded_index.as_retriever(similarity_top_k=1)

query = "LlamaIndex是做什么的？"
results = retriever.retrieve(query)

print(f"\n查询: '{query}'")
print("相似度最高的文档:")
for node in results:
    # node.text 是 LlamaIndex 中 Node 的文本属性
    # node.score 是相似度分数（余弦相似度，越接近 1 越相似）
    print(f"- 内容: {node.text}")
    print(f"  相似度: {node.score:.4f}")