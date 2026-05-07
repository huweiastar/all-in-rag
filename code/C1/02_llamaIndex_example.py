"""
基于 LlamaIndex 的 RAG（检索增强生成）基础示例

与 01_langchain_example.py 功能相同，但使用 LlamaIndex 框架实现。
LlamaIndex 相比 LangChain 提供了更高层的抽象，用更少的代码完成相同的流程：

  LangChain: 需要手动串联 加载 → 分块 → 向量化 → 存储 → 检索 → 提示词 → 调用
  LlamaIndex: 只需设置好全局配置，调用 from_documents + as_query_engine 即可

核心区别：
  - LangChain 显式暴露每个步骤，适合需要精细控制的场景
  - LlamaIndex 封装度更高，默认分块器、默认提示词、默认检索策略都已配好

依赖环境变量:
    DASHSCOPE_API_KEY - 大语言模型的 API 密钥
    DASHSCOPE_BASE_URL - API 端点
"""
import os
# 国内环境无法访问 HuggingFace 时启用该镜像设置
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# 从 .env 文件中加载环境变量
load_dotenv()

# ── 配置全局大语言模型（LLM） ─────────────────────────────────────────────────
# Settings 是 LlamaIndex 的全局配置对象，类似"注册表"，设置一次后所有组件默认使用
# OpenAILike 用于连接任何兼容 OpenAI 接口的第三方服务（智谱、DeepSeek 等）
Settings.llm = OpenAILike(
    model="qwen3.6-plus",
    api_key=os.getenv("DASHSCOPE_API_KEY"),  # API 密钥从环境变量读取
    api_base=os.getenv("DASHSCOPE_BASE_URL"),  # 兼容 OpenAI 格式的代理端点
    is_chat_model=True  # 声明这是对话型模型（使用 Chat 接口而非 Completion 接口）
)

# ── 配置全局嵌入模型（Embedding） ─────────────────────────────────────────────
# BGE-small-zh 是专为中文优化的开源嵌入模型
# LlamaIndex 在构建索引和检索时会自动使用该模型将文本转为向量
Settings.embed_model = HuggingFaceEmbedding("BAAI/bge-small-zh-v1.5")

# ── 第一步：加载文档 ──────────────────────────────────────────────────────────
# SimpleDirectoryReader 是 LlamaIndex 内置的文档加载器，支持多种格式：
#   .txt, .pdf, .docx, .md, .csv, .html, .json 等
# input_files 指定具体文件列表；也可以用 input_dir 指定整个目录
docs = SimpleDirectoryReader(input_files=["../../data/C1/markdown/easy-rl-chapter1.md"]).load_data()

# ── 第二步 & 第三步：构建向量索引 ─────────────────────────────────────────────
# from_documents 一行代码完成了 LangChain 中需要多步才能完成的事情：
#
#   1. 分块：使用默认的 SentenceSplitter（chunk_size=1024, chunk_overlap=20）
#      按句子边界切分，比 LangChain 默认的 RecursiveCharacterTextSplitter 更温和
#
#   2. 向量化：遍历所有文本块，调用 Settings.embed_model 生成向量
#      自动批量处理，无需手动调用 encode
#
#   3. 存储：将 (文本, 向量, 元数据) 存入内存中的 SimpleVectorStore
#      使用余弦相似度作为默认距离度量
#
# 返回值 index 包含了完整的索引结构，后续可以基于它做检索、查询、过滤等操作
index = VectorStoreIndex.from_documents(docs)

# ── 第四步：创建查询引擎 ──────────────────────────────────────────────────────
# as_query_engine() 返回一个开箱即用的问答引擎（RetrieverQueryEngine 实例）
# 内部执行链路：
#
#   用户问题
#       ↓
#   VectorIndexRetriever（从索引中按相似度检索 top-k 个文本块）
#       ↓
#   ResponseSynthesizer（将检索结果 + 问题组装成提示词，发送给 LLM，合成最终回答）
#       ↓
#   返回回答
#
# 可选参数示例：
#   index.as_query_engine(similarity_top_k=5)        # 默认检索 2 个，改为 5 个
#   index.as_query_engine(response_mode="compact")   # 紧凑模式，减少 token 消耗
#   index.as_query_engine(response_mode="tree_summarize")  # 多块时分层总结
query_engine = index.as_query_engine()

# ── 调试：打印查询引擎的提示词模板 ────────────────────────────────────────────
# get_prompts() 返回当前查询引擎使用的所有 Prompt 模板
# LlamaIndex 默认使用以下模板（中英文混合场景可能需要自定义）：
#
#   default_refine_prompt_template: 用于多块文档的分步精炼
#   default_text_qa_prompt_template: 主问答模板，包含上下文占位符 {context_str} 和问题占位符 {query_str}
#   default_tree_summarize_prompt: 树形总结模式下的提示词
#
# 如果默认提示词效果不好，可以用 query_engine.update_prompts() 替换
print("=== 查询引擎使用的提示词模板 ===")
print(query_engine.get_prompts())

# ── 第五步：执行查询 ──────────────────────────────────────────────────────────
# query() 方法接收自然语言问题，内部自动完成检索 + 生成
# 返回值是一个 Response 对象，其 .response 属性包含生成的文本内容
# 这与 LangChain 返回的 AIMessage 不同，LlamaIndex 的 Response 只包含回答内容，
# 没有 token 统计、模型 ID 等元数据（更干净）
response = query_engine.query("文中举了哪些例子?")
print(f"\n=== 回答 ===")
print(response)

# ── LlamaIndex vs LangChain 对照表 ────────────────────────────────────────────
#
# 功能              | LangChain                              | LlamaIndex
# ──────────────────┼────────────────────────────────────────┼──────────────────────────
# 文档加载          | UnstructuredMarkdownLoader             | SimpleDirectoryReader
# 文本分块          | RecursiveCharacterTextSplitter         | SentenceSplitter（内置）
# 嵌入模型          | HuggingFaceEmbeddings                  | HuggingFaceEmbedding
# 向量存储          | InMemoryVectorStore                    | SimpleVectorStore（内置）
# 构建索引          | 手动 add_documents                     | VectorStoreIndex.from_documents
# 查询              | similarity_search + llm.invoke         | query_engine.query()
# 返回类型          | AIMessage（含大量元数据）               | Response（纯文本内容）
# 提示词            | ChatPromptTemplate.from_template       | 内置默认 + get_prompts() 查看
# 全局配置          | 无，每次手动传递                        | Settings（llm / embed_model）
#
# 经验法则：
# - 快速原型 / 简单 RAG → LlamaIndex（代码更少）
# - 需要精细控制每个环节 → LangChain（更灵活）