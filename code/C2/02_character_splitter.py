"""
CharacterTextSplitter — 固定字符分块器

最基础的分块策略：按指定的分隔符（默认是换行符 "\n\n"）切分文本，
然后将切分后的片段拼接成固定大小的块。

工作原理：
  1. 先用分隔符把整篇文档切成"片段"（splits）
  2. 逐个片段累积，直到总长度接近 chunk_size
  3. 形成一个块，保留 chunk_overlap 长度的尾部作为下一个块的开头
  4. 重复直到所有片段处理完毕

适用场景：结构规整的文本（如日志、代码、每段长度均匀的文章）
缺点：对于自然语言，分隔符可能出现在句子中间，导致语义被截断
"""
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader

# ── 第一步：加载文档 ──────────────────────────────────────────────────────────
# TextLoader 读取纯文本文件，encoding 指定编码防止中文乱码
loader = TextLoader("../../data/C2/txt/蜂医.txt", encoding="utf-8")
docs = loader.load()

# ── 第二步：初始化固定大小分块器 ──────────────────────────────────────────────
# separator: 切分的分隔符，默认是 "\n\n"（双换行，即段落边界）
#   如果文本中没有双换行，会回退到单换行、空格、空字符串依次尝试
# chunk_size: 每个块的目标字符数上限
# chunk_overlap: 相邻两个块之间保留多少字符的重叠
#   重叠的作用：防止关键信息恰好被切在两块交界处而被遗漏
text_splitter = CharacterTextSplitter(
    chunk_size=200,    # 每个块的大小（字符数）
    chunk_overlap=10   # 块之间的重叠（字符数）
)

# ── 第三步：执行分块 ──────────────────────────────────────────────────────────
# 输入是 Document 对象列表，输出也是 Document 对象列表
# 每个输出 Document 的 page_content 属性包含切分后的文本片段
chunks = text_splitter.split_documents(docs)

# ── 第四步：打印结果 ──────────────────────────────────────────────────────────
print(f"文本被切分为 {len(chunks)} 个块。\n")
print("--- 前5个块内容示例 ---")
for i, chunk in enumerate(chunks[:5]):
    print("=" * 60)
    # chunk 是一个 Document 对象，需要访问它的 .page_content 属性来获取文本
    print(f'块 {i+1} (长度: {len(chunk.page_content)}): "{chunk.page_content}"')