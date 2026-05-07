"""
基于 LangChain 的 RAG（检索增强生成）基础示例 — chunk 参数对比

用途：修改 chunk_size 和 chunk_overlap，观察分块数量和回答结果的变化。

完整流程：
1. 加载 Markdown 文档
2. 用不同参数将文档切分为文本块
3. 使用嵌入模型将文本块转为向量并存储
4. 根据用户问题检索相关文本块
5. 将检索到的上下文 + 问题一起发送给大语言模型生成回答

依赖环境变量:
    DASHSCOPE_API_KEY - 大语言模型的 API 密钥
    DASHSCOPE_BASE_URL - API 端点
"""
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from dotenv import load_dotenv
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()

# ── 第一步：加载文档 ──────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
markdown_path = os.path.join(BASE_DIR, "data/C1/markdown/easy-rl-chapter1.md")
loader = UnstructuredMarkdownLoader(markdown_path)
docs = loader.load()
print(f"文档总字符数: {sum(len(d.page_content) for d in docs):,}")

# ── 第二步 & 第三步：不同参数分块 + 向量化 + 存储 ─────────────────────────────
# 要对比的参数组合: (chunk_size, chunk_overlap)
param_groups = [
    (4000, 200),   # 默认
    (500, 50),     # 小块，小重叠
    (500, 0),      # 小块，无重叠
    (2000, 400),   # 中等块，较大重叠
]

question = "文中举了哪些例子？"

prompt = ChatPromptTemplate.from_template("""请根据下面提供的上下文信息来回答问题。
请确保你的回答完全基于这些上下文。
如果上下文中没有足够的信息来回答问题，请直接告知："抱歉，我无法根据提供的上下文找到相关信息来回答此问题。"

上下文:
{context}

问题: {question}

回答:""")

llm = ChatOpenAI(
    model="qwen3.6-plus",
    temperature=0.7,
    max_tokens=4096,
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url=os.getenv("DASHSCOPE_BASE_URL")
)

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

for chunk_size, chunk_overlap in param_groups:
    print(f"\n{'='*60}")
    print(f"参数: chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")
    print(f"{'='*60}")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = text_splitter.split_documents(docs)

    # 统计信息
    sizes = [len(c.page_content) for c in chunks]
    print(f"  分块数量: {len(chunks)}")
    print(f"  最小块字符数: {min(sizes)}")
    print(f"  最大块字符数: {max(sizes)}")
    print(f"  平均块字符数: {sum(sizes) // len(sizes)}")

    vectorstore = InMemoryVectorStore.from_documents(chunks, embeddings)

    retrieved_docs = vectorstore.similarity_search(question, k=3)
    docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)

    answer = llm.invoke(prompt.format(question=question, context=docs_content))
    print(f"\n  回答:\n  {answer.content}")

# ── chunk_size 与 chunk_overlap 参数对比总结 ─────────────────────────────────
#
# 运行环境: qwen3.6-plus 模型, k=3 检索, 问题: "文中举了哪些例子？"
#
# ┌─────────────────────────┬────┬──────┬─────────────────────────────────────┐
# │ 参数组合                │块数│均值  │ 回答质量                            │
# ├─────────────────────────┼────┼──────┼─────────────────────────────────────┤
# │ 4000, 200 (默认)        │  6 │ 3804 │ 最全面，5 大类 + 十几个具体例子     │
# │ 2000, 400               │ 15 │ 1771 │ 较全面，补充奖励机制、状态表示      │
# │  500,  50               │ 61 │  374 │ 信息丢失，只抓到 3 类               │
# │  500,   0               │ 60 │  371 │ 最差，只有探索/利用和动作空间       │
# └─────────────────────────┴────┴──────┴─────────────────────────────────────┘
#
# 核心结论：
# 1. chunk_size 越大，每个块的语义越完整，top-k 检索能覆盖更多内容
# 2. chunk_size 过小（如 500）会将同一个概念切碎分散到不同块中，
#    向量检索只取 top-3，大量相关内容不在上下文里，模型回答不完整
# 3. chunk_overlap 的作用：在块大小相同的情况下，
#    有重叠保留了切分处的语义连续性，效果略优于无重叠
# 4. 经验法则：chunk_size 应与文档平均段落长度匹配，过小比过大更危险