"""
RAG系统主程序

本模块是食谱RAG系统（"尝尝咸淡"）的入口文件，包含：
1. RecipeRAGSystem: 系统主类，整合所有模块实现完整的RAG问答流程
2. main(): 命令行交互入口

系统架构流程：
  用户查询 → 查询路由(list/detail/general) → 查询重写(可选)
  → 元数据过滤条件提取 → 混合检索(向量+BM25+RRF)
  → 子块还原为完整文档 → 根据路由类型生成回答 → 返回结果
"""

import os
import sys
import logging
from pathlib import Path
from typing import List

# 将当前目录加入sys.path，确保可以导入同级的config和rag_modules包
# Path(__file__).parent获取当前文件所在目录的绝对路径
sys.path.append(str(Path(__file__).parent))

from dotenv import load_dotenv
from config import DEFAULT_CONFIG, RAGConfig
from rag_modules import (
    DataPreparationModule,
    IndexConstructionModule,
    RetrievalOptimizationModule,
    GenerationIntegrationModule
)

# 加载.env文件中的环境变量（如DASHSCOPE_API_KEY）
# load_dotenv会搜索当前目录及父目录中的.env文件
load_dotenv()

# 配置全局日志格式
# 格式: 时间 - 模块名 - 日志级别 - 消息内容
# level=INFO 过滤掉DEBUG级别的日志，避免嵌入模型的详细输出刷屏
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RecipeRAGSystem:
    """
    食谱RAG系统主类 - "尝尝咸淡"的核心编排层

    职责：
    1. 系统初始化：按顺序初始化数据、索引、检索、生成四个模块
    2. 知识库构建：加载文档→分块→构建向量索引→保存→初始化检索器
    3. 智能问答：完整的RAG流程编排（路由→重写→检索→生成）
    4. 交互式对话：命令行交互界面

    初始化的四个模块在系统中的角色：
    - DataPreparationModule: 数据层，负责原始数据的加载和处理
    - IndexConstructionModule: 向量层，负责将文本转为可搜索的向量
    - RetrievalOptimizationModule: 检索层，负责多路召回和融合排序
    - GenerationIntegrationModule: 生成层，负责调用LLM生成最终回答

    索引缓存策略：
    - 首次运行：构建FAISS索引并保存到本地磁盘
    - 后续运行：直接从磁盘加载已保存的索引（跳过嵌入计算）
    - 这大大加快了系统启动速度，因为嵌入模型推理是启动过程中最耗时的步骤
    """

    def __init__(self, config: RAGConfig = None):
        """
        初始化RAG系统

        构造函数只做基本的配置验证和环境检查，不加载任何模型或数据。
        实际的模块初始化在initialize_system()中进行，实现延迟加载。

        初始化检查：
        1. 数据路径存在性：避免后续加载失败才发现
        2. API密钥存在性：LLM调用必须的环境变量

        Args:
            config: RAG系统配置对象，不传则使用DEFAULT_CONFIG

        Raises:
            FileNotFoundError: 当data_path不存在时抛出
            ValueError: 当DASHSCOPE_API_KEY未设置时抛出
        """
        # 使用传入的配置或默认配置
        self.config = config or DEFAULT_CONFIG

        # 四个核心模块，初始化为None，在initialize_system()中创建
        self.data_module = None
        self.index_module = None
        self.retrieval_module = None
        self.generation_module = None

        # 检查数据路径是否存在（快速失败原则）
        if not Path(self.config.data_path).exists():
            raise FileNotFoundError(f"数据路径不存在: {self.config.data_path}")

        # 检查API密钥是否已设置
        if not os.getenv("DASHSCOPE_API_KEY"):
            raise ValueError("请设置 DASHSCOPE_API_KEY 环境变量")

    def initialize_system(self):
        """
        按依赖顺序初始化所有模块

        初始化顺序非常重要：
        1. 数据准备模块（DataPreparationModule）
           - 最先初始化，因为后续模块都依赖它加载的数据
        2. 索引构建模块（IndexConstructionModule）
           - 需要嵌入模型加载，独立于数据
        3. 生成集成模块（GenerationIntegrationModule）
           - 需要LLM连接，独立于数据和索引

        检索优化模块（RetrievalOptimizationModule）不在此处初始化，
        而是在build_knowledge_base()中创建，因为它依赖已构建好的
        vectorstore和chunks数据。

        打印emoji前缀的进度信息，改善命令行交互体验：
        🚀 = 启动, 🤖 = AI/LLM相关, ✅ = 完成
        """
        print("🚀 正在初始化RAG系统...")

        # 1. 初始化数据准备模块
        #    传入数据路径，此时只记录路径不实际加载文件
        print("初始化数据准备模块...")
        self.data_module = DataPreparationModule(self.config.data_path)

        # 2. 初始化索引构建模块
        #    传入嵌入模型名称和索引保存路径
        #    包含嵌入模型的下载和加载（首次运行需要下载）
        print("初始化索引构建模块...")
        self.index_module = IndexConstructionModule(
            model_name=self.config.embedding_model,
            index_save_path=self.config.index_save_path
        )

        # 3. 初始化生成集成模块
        #    建立与LLM API的连接
        #    需要有效的DASHSCOPE_API_KEY
        print("🤖 初始化生成集成模块...")
        self.generation_module = GenerationIntegrationModule(
            model_name=self.config.llm_model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens
        )

        print("✅ 系统初始化完成！")

    def build_knowledge_base(self):
        """
        构建或加载知识库（RAG系统的核心数据工程步骤）

        完整流程：
        1. 尝试加载已保存的FAISS索引（快速路径）
           - 如果存在：直接加载，避免重新计算嵌入向量
           - 然后仍需加载文档和分块（用于检索模块和元数据过滤）
        2. 如果索引不存在，走完整的构建流程（首次运行路径）：
           a. 加载所有食谱Markdown文件
           b. 按Markdown标题结构分块
           c. 计算每个块的嵌入向量
           d. 构建FAISS向量索引
           e. 将索引持久化到磁盘
        3. 初始化检索优化模块（依赖vectorstore和chunks）
        4. 展示知识库统计信息

        为什么索引加载后还要加载文档和分块？
        - FAISS索引存储的是向量和与向量关联的Document，
          但检索模块的BM25Retriever需要chunks列表来构建BM25索引
        - 元数据过滤也需要访问chunks的metadata
        - 此外，get_parent_documents需要完整文档列表
        """
        print("\n正在构建知识库...")

        # 1. 尝试加载已保存的向量索引
        vectorstore = self.index_module.load_index()

        if vectorstore is not None:
            # 快速路径：索引已存在，直接使用
            print("✅ 成功加载已保存的向量索引！")
            # 仍需加载文档和分块用于检索模块
            print("加载食谱文档...")
            self.data_module.load_documents()
            print("进行文本分块...")
            chunks = self.data_module.chunk_documents()
        else:
            # 首次运行路径：完整构建
            print("未找到已保存的索引，开始构建新索引...")

            # 2. 加载文档
            print("加载食谱文档...")
            self.data_module.load_documents()

            # 3. 文本分块
            print("进行文本分块...")
            chunks = self.data_module.chunk_documents()

            # 4. 构建向量索引（最耗时的步骤）
            #    嵌入模型需要将每个chunk的文本编码为向量
            print("构建向量索引...")
            vectorstore = self.index_module.build_vector_index(chunks)

            # 5. 保存索引到磁盘，下次启动可直接加载
            print("保存向量索引...")
            self.index_module.save_index()

        # 6. 初始化检索优化模块
        #    需要vectorstore（向量搜索）和chunks（BM25搜索）
        print("初始化检索优化...")
        self.retrieval_module = RetrievalOptimizationModule(vectorstore, chunks)

        # 7. 显示知识库统计信息
        #    帮助用户了解当前知识库的规模和分布
        stats = self.data_module.get_statistics()
        print(f"\n📊 知识库统计:")
        print(f"   文档总数: {stats['total_documents']}")
        print(f"   文本块数: {stats['total_chunks']}")
        print(f"   菜品分类: {list(stats['categories'].keys())}")
        print(f"   难度分布: {stats['difficulties']}")

        print("✅ 知识库构建完成！")

    def ask_question(self, question: str, stream: bool = False):
        """
        核心问答方法 - 完整的RAG流程编排

        处理流程（6步）：
        ┌─────────────────────────────────────────────────────┐
        │ 1. 查询路由 (query_router)                           │
        │    将用户问题分类为: list / detail / general         │
        ├─────────────────────────────────────────────────────┤
        │ 2. 查询重写 (query_rewrite)                          │
        │    list类型保持原样 | detail/general类型智能重写    │
        ├─────────────────────────────────────────────────────┤
        │ 3. 元数据过滤条件提取 (_extract_filters_from_query)  │
        │    从查询中识别分类和难度关键词作为过滤条件          │
        ├─────────────────────────────────────────────────────┤
        │ 4. 检索 (hybrid_search / metadata_filtered_search)   │
        │    有过滤条件→元数据过滤检索 | 无过滤→纯混合检索    │
        ├─────────────────────────────────────────────────────┤
        │ 5. 子块还原为完整文档 (get_parent_documents)         │
        │    从检索到的小块通过parent_id映射找回完整食谱       │
        ├─────────────────────────────────────────────────────┤
        │ 6. 答案生成 (根据路由类型选择生成策略)               │
        │    list → generate_list_answer (不调LLM)             │
        │    detail → generate_step_by_step_answer (结构化)    │
        │    general → generate_basic_answer (通用)            │
        └─────────────────────────────────────────────────────┘

        Args:
            question: 用户输入的问题
            stream: 是否启用流式输出（打字机效果）

        Returns:
            - stream=False: 返回完整的回答字符串
            - stream=True: 返回生成器，逐token yield文本片段

        Raises:
            ValueError: 当知识库尚未构建时抛出
        """
        # 前置检查：确保检索和生成模块已初始化
        if not all([self.retrieval_module, self.generation_module]):
            raise ValueError("请先构建知识库")

        print(f"\n❓ 用户问题: {question}")

        # ==== 第1步：查询路由 ====
        # 通过LLM判断用户意图是哪一类
        route_type = self.generation_module.query_router(question)
        print(f"🎯 查询类型: {route_type}")

        # ==== 第2步：智能查询重写 ====
        # list类型的查询不需要重写（"有哪些川菜"保留原样即可）
        if route_type == 'list':
            rewritten_query = question
            print(f"📝 列表查询保持原样: {question}")
        else:
            # detail和general类型的查询通过LLM进行智能重写
            # 模糊查询会被扩充，具体查询保持不变
            print("🤖 智能分析查询...")
            rewritten_query = self.generation_module.query_rewrite(question)

        # ==== 第3步：检索相关子块 ====
        # 首先尝试从查询中提取元数据过滤条件
        print("🔍 检索相关文档...")
        filters = self._extract_filters_from_query(question)

        # 根据是否有过滤条件选择不同的检索策略
        if filters:
            # 有过滤条件：先混合检索，再按元数据筛选
            print(f"应用过滤条件: {filters}")
            relevant_chunks = self.retrieval_module.metadata_filtered_search(
                rewritten_query, filters, top_k=self.config.top_k
            )
        else:
            # 无过滤条件：直接混合检索（向量+BM25+RRF）
            relevant_chunks = self.retrieval_module.hybrid_search(
                rewritten_query, top_k=self.config.top_k
            )

        # ==== 显示检索到的子块信息 ====
        # 友好地展示检索结果，帮助用户理解系统找到了什么
        if relevant_chunks:
            chunk_info = []
            for chunk in relevant_chunks:
                dish_name = chunk.metadata.get('dish_name', '未知菜品')
                # 尝试从内容开头提取章节标题，方便用户了解chunk定位
                content_preview = chunk.page_content[:100].strip()
                if content_preview.startswith('#'):
                    # 如果chunk以Markdown标题开头，提取标题文本
                    title_end = (
                        content_preview.find('\n')
                        if '\n' in content_preview
                        else len(content_preview)
                    )
                    section_title = content_preview[:title_end].replace('#', '').strip()
                    chunk_info.append(f"{dish_name}({section_title})")
                else:
                    chunk_info.append(f"{dish_name}(内容片段)")

            print(
                f"找到 {len(relevant_chunks)} 个相关文档块: "
                f"{', '.join(chunk_info)}"
            )
        else:
            print(f"找到 {len(relevant_chunks)} 个相关文档块")

        # ==== 第4步：未找到相关内容的处理 ====
        # 提前返回友好提示，避免将空上下文传给LLM
        if not relevant_chunks:
            return "抱歉，没有找到相关的食谱信息。请尝试其他菜品名称或关键词。"

        # ==== 第5步：根据路由类型选择回答策略 ====
        if route_type == 'list':
            # 列表查询：还原为完整文档后，直接用generate_list_answer生成列表
            # 不调用LLM，直接提取菜名并格式化
            print("📋 生成菜品列表...")
            relevant_docs = self.data_module.get_parent_documents(relevant_chunks)

            # 显示找到的文档名称
            doc_names = []
            for doc in relevant_docs:
                dish_name = doc.metadata.get('dish_name', '未知菜品')
                doc_names.append(dish_name)
            if doc_names:
                print(f"找到文档: {', '.join(doc_names)}")

            return self.generation_module.generate_list_answer(question, relevant_docs)
        else:
            # detail和general查询：还原为完整文档后用LLM生成详细回答
            print("获取完整文档...")
            relevant_docs = self.data_module.get_parent_documents(relevant_chunks)

            # 显示找到的文档名称
            doc_names = []
            for doc in relevant_docs:
                dish_name = doc.metadata.get('dish_name', '未知菜品')
                doc_names.append(dish_name)
            if doc_names:
                print(f"找到文档: {', '.join(doc_names)}")
            else:
                print(f"对应 {len(relevant_docs)} 个完整文档")

            print("✍️ 生成详细回答...")

            # ==== 第6步：根据具体路由类型选择生成方法 ====
            if route_type == "detail":
                # 详细查询：使用分步骤指导模式
                # 结构化输出：菜品介绍 → 食材 → 步骤 → 技巧
                if stream:
                    return self.generation_module.generate_step_by_step_answer_stream(
                        question, relevant_docs
                    )
                else:
                    return self.generation_module.generate_step_by_step_answer(
                        question, relevant_docs
                    )
            else:
                # 一般查询：使用基础回答模式
                # 非结构化输出，更灵活
                if stream:
                    return self.generation_module.generate_basic_answer_stream(
                        question, relevant_docs
                    )
                else:
                    return self.generation_module.generate_basic_answer(
                        question, relevant_docs
                    )

    def _extract_filters_from_query(self, query: str) -> dict:
        """
        从用户查询文本中智能提取元数据过滤条件

        提取策略：
        1. 分类关键词：遍历所有支持的分类标签（如"荤菜"、"素菜"），
           检查是否出现在查询中。第一个匹配的分类即为过滤条件。
        2. 难度关键词：遍历所有支持的难度等级，按字符串长度降序排列，
           优先匹配较长的关键词（避免"简单"匹配到"非常简单"）。

        为什么要按长度降序匹配：
        - "非常简单"包含"简单"二字，如果先匹配"简单"，则"非常简单"会被识别为"简单"
        - 按长度降序保证优先匹配更具体的关键词

        使用示例：
        - "推荐几个简单的素菜" → {'category': '素菜', 'difficulty': '简单'}
        - "有什么荤菜" → {'category': '荤菜'}
        - "宫保鸡丁怎么做" → {}（无可提取的过滤条件）

        Args:
            query: 用户查询文本

        Returns:
            过滤条件字典，如 {'category': '荤菜', 'difficulty': '简单'}
            如果无法提取则返回空字典 {}
        """
        filters = {}

        # 提取分类关键词
        # 使用DataPreparationModule的类方法获取支持的分类列表
        category_keywords = DataPreparationModule.get_supported_categories()
        for cat in category_keywords:
            if cat in query:
                filters['category'] = cat
                break  # 只取第一个匹配的分类

        # 提取难度关键词
        # 按字符串长度降序排列，优先匹配长关键词
        difficulty_keywords = DataPreparationModule.get_supported_difficulties()
        for diff in sorted(difficulty_keywords, key=len, reverse=True):
            if diff in query:
                filters['difficulty'] = diff
                break  # 只取第一个匹配的难度

        return filters

    def search_by_category(self, category: str, query: str = "") -> List[str]:
        """
        按分类搜索菜品

        使用元数据过滤检索，仅返回匹配特定分类的菜品名称。
        先做混合检索再做分类过滤，保证结果既与查询相关又符合分类要求。

        Args:
            category: 菜品分类，如 "荤菜"、"素菜"
            query: 可选的额外查询条件，为空时使用分类名作为查询

        Returns:
            去重后的菜品名称列表
        """
        if not self.retrieval_module:
            raise ValueError("请先构建知识库")

        # 使用分类名作为基础查询（如果用户没有提供额外查询）
        search_query = query if query else category
        filters = {"category": category}

        # 检索top_k=10个结果（多于默认的3个，因为可能有些结果不相关）
        docs = self.retrieval_module.metadata_filtered_search(
            search_query, filters, top_k=10
        )

        # 提取并去重菜品名称
        dish_names = []
        for doc in docs:
            dish_name = doc.metadata.get('dish_name', '未知菜品')
            if dish_name not in dish_names:
                dish_names.append(dish_name)

        return dish_names

    def get_ingredients_list(self, dish_name: str) -> str:
        """
        获取指定菜品的食材信息

        便捷方法：封装了检索→生成两个步骤，一次调用获取食材信息。

        Args:
            dish_name: 菜品名称，如 "宫保鸡丁"

        Returns:
            包含食材信息的回答文本
        """
        if not all([self.retrieval_module, self.generation_module]):
            raise ValueError("请先构建知识库")

        # 用菜品名称进行混合检索
        docs = self.retrieval_module.hybrid_search(dish_name, top_k=3)

        # 构造食材查询语句并生成回答
        # 显式拼接"需要什么食材"，引导LLM重点回答食材部分
        answer = self.generation_module.generate_basic_answer(
            f"{dish_name}需要什么食材？", docs
        )

        return answer

    def run_interactive(self):
        """
        运行交互式命令行问答会话

        交互式循环流程：
        1. 初始化系统（加载模型和连接）
        2. 构建/加载知识库
        3. 进入无限循环：
           a. 读取用户输入
           b. 检查退出条件（"退出"/"quit"/"exit"/空输入）
           c. 询问是否使用流式输出
           d. 调用ask_question获取回答
           e. 显示结果

        交互设计要点：
        - 支持Ctrl+C（KeyboardInterrupt）优雅退出
        - 异常不中断循环，打印错误后继续接受下一个问题
        - 空输入视为退出意图
        - 默认启用流式输出（体验更好）
        """
        # 打印欢迎横幅
        print("=" * 60)
        print("🍽️  尝尝咸淡RAG系统 - 交互式问答  🍽️")
        print("=" * 60)
        print("💡 解决您的选择困难症，告别'今天吃什么'的世纪难题！")

        # 初始化所有模块
        self.initialize_system()

        # 构建或加载知识库
        self.build_knowledge_base()

        print("\n交互式问答 (输入'退出'结束):")

        # 主交互循环
        while True:
            try:
                # 读取用户问题
                user_input = input("\n您的问题: ").strip()

                # 退出条件检测
                # 支持中文"退出"、英文"quit"/"exit"、以及空输入
                if user_input.lower() in ['退出', 'quit', 'exit', '']:
                    break

                # 询问流式输出偏好
                # 默认为y（使用流式输出），体验如打字机效果的实时文本生成
                stream_choice = input("是否使用流式输出? (y/n, 默认y): ").strip().lower()
                use_stream = stream_choice != 'n'

                print("\n回答:")
                if use_stream:
                    # 流式输出模式：逐token打印
                    # ask_question(stream=True)返回生成器
                    # end="" 避免print自动换行
                    # flush=True 强制刷新缓冲区，确保用户能看到实时输出
                    for chunk in self.ask_question(user_input, stream=True):
                        print(chunk, end="", flush=True)
                    print("\n")
                else:
                    # 普通模式：等待完整回答后一次性打印
                    answer = self.ask_question(user_input, stream=False)
                    print(f"{answer}\n")

            except KeyboardInterrupt:
                # 用户按下Ctrl+C，优雅退出循环
                break
            except Exception as e:
                # 捕获其他异常，打印错误但不中断会话
                # 用户可以在排查问题后继续提问
                print(f"处理问题时出错: {e}")

        print("\n感谢使用尝尝咸淡RAG系统！")


def main():
    """
    命令行入口函数

    执行流程：
    1. 使用默认配置创建RecipeRAGSystem实例
    2. 运行交互式问答循环

    异常处理：捕获并打印所有未处理的异常，避免程序静默崩溃
    """
    try:
        # 创建RAG系统实例，使用默认配置
        rag_system = RecipeRAGSystem()

        # 启动交互式问答会话
        rag_system.run_interactive()

    except Exception as e:
        # 记录到日志文件和打印到控制台
        logger.error(f"系统运行出错: {e}")
        print(f"系统错误: {e}")


# Python脚本入口：当直接运行 python main.py 时执行main()
if __name__ == "__main__":
    main()