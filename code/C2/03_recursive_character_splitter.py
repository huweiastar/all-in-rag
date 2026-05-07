"""
RecursiveCharacterTextSplitter — 递归字符分块器

LangChain 中最常用、也是上一章节 C1 中使用的分块器。

与 CharacterTextSplitter 的区别：
  - CharacterTextSplitter 只有一个分隔符，找不到合适切点时会直接截断
  - RecursiveCharacterTextSplitter 有一组分隔符列表，按顺序递归尝试

工作原理（递归切分）：
  1. 先用第一个分隔符（如 "\n\n" 段落边界）切分
  2. 如果某一块仍然超过 chunk_size，就用下一个分隔符（如 "\n" 换行）继续切
  3. 如果还超，就用 "。"（句号）、"，"（逗号）、" "（空格）继续切
  4. 最后才用空字符串 "" 强制截断（保底策略）

这种"先粗后细"的策略保证了：
  - 优先在自然边界（段落、句子）处切分，保持语义完整
  - 只有万不得已才在单词/字符级别硬切

适用场景：中英文混合的自然语言文本，RAG 项目的默认选择
"""
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader

# ── 加载文档 ──────────────────────────────────────────────────────────────────
loader = TextLoader("../../data/C2/txt/蜂医.txt", encoding="utf-8")
docs = loader.load()

# ── 初始化递归字符分块器 ──────────────────────────────────────────────────────
# separators 列表的顺序很重要：
#   "\n\n" → 段落边界（最优先）
#   "\n"   → 换行
#   "。"   → 中文句号
#   "，"   → 中文逗号
#   " "    → 英文空格
#   ""     → 空字符串，最后的保底，确保不会出现超长块
#
# 注意：默认的分隔符列表是 ["\n\n", "\n", " ", ""]，不包含中文标点。
# 这里手动添加了 "。" 和 "，"，对中文文本的切分质量提升显著。
text_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", "。", "，", " ", ""],  # 按顺序尝试分割
    chunk_size=200,
    chunk_overlap=10
)

# ── 执行分块 ──────────────────────────────────────────────────────────────────
chunks = text_splitter.split_documents(docs)

# ── 打印结果 ──────────────────────────────────────────────────────────────────
print(f"文本被切分为 {len(chunks)} 个块。\n")
print("--- 前5个块内容示例 ---")
for i, chunk in enumerate(chunks[:5]):
    print("=" * 60)
    print(f'块 {i+1} (长度: {len(chunk.page_content)}): "{chunk.page_content}"')