"""
SemanticChunker — 语义分块器

与前三种分块器（按固定长度/分隔符切分）完全不同，
SemanticChunker 基于语义相似度来决定在哪里切分。

工作原理：
  1. 先将文本按句子切分成小段
  2. 用嵌入模型将每句话转为向量
  3. 计算相邻句子之间的语义距离（向量差异）
  4. 根据设定的阈值策略，在语义突变处切分

关键参数 breakpoint_threshold_type 的四种策略：
  - "percentile"（百分位）：将语义距离排序，取前 N% 最大的距离处切分
    例如 percentile=95 表示只在最大的 5% 语义差异处切分
    值越高 → 块越少越大；值越低 → 块越多越小
  - "standard_deviation"（标准差）：当某处语义距离超过均值 + N 倍标准差时切分
  - "interquartile"（四分位距）：使用 IQR 统计方法确定切分点
  - "gradient"（梯度）：基于语义距离变化率的二阶导数找突变点

优点：
  - 自动保持语义连贯性，同一个主题的内容不会被切断
  - 不需要手动调 chunk_size，块大小由内容本身决定
  - 最适合长篇文章、论文、技术文档等结构化程度不高的文本

缺点：
  - 需要调用嵌入模型，速度慢、资源消耗大
  - 不适合代码、日志、表格等结构规整的文本
"""
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader

# ── 初始化嵌入模型 ────────────────────────────────────────────────────────────
# SemanticChunker 依赖嵌入模型计算句子间的语义距离
# 与 C1 中使用的完全相同的 BGE-small-zh 模型
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

# ── 初始化 SemanticChunker ───────────────────────────────────────────────────
# 第一个参数是嵌入模型，必须传入
# breakpoint_threshold_type 控制切分策略，percentile 是最常用的模式
# 还可以通过 breakpoint_threshold_amount 参数微调阈值（默认值视策略而定）
text_splitter = SemanticChunker(
    embeddings,
    breakpoint_threshold_type="percentile"  # 百分位策略
    # 其他可选值:
    #   "standard_deviation" — 标准差
    #   "interquartile"      — 四分位距
    #   "gradient"           — 梯度
)

# ── 加载文档 ──────────────────────────────────────────────────────────────────
loader = TextLoader("../../data/C2/txt/蜂医.txt", encoding="utf-8")
documents = loader.load()

# ── 执行语义分块 ──────────────────────────────────────────────────────────────
# 注意：这里没有 chunk_size 参数！块大小由语义相似度自动决定
docs = text_splitter.split_documents(documents)

# ── 打印结果 ──────────────────────────────────────────────────────────────────
print(f"文本被切分为 {len(docs)} 个块。\n")
print("--- 前2个块内容示例 ---")
for i, chunk in enumerate(docs[:2]):
    print("=" * 60)
    print(f'块 {i+1} (长度: {len(chunk.page_content)}):\n"{chunk.page_content}"')