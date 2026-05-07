"""
生成集成模块

本模块负责与LLM的集成，实现RAG系统的"生成"环节（Retrieve-And-Generate中的Generate）：
1. LLM初始化和配置
2. 查询分析：智能路由（list/detail/general）和查询重写
3. 答案生成：基础回答、分步骤回答、列表式回答
4. 支持普通模式和流式输出模式

本系统使用Moonshot API（兼容OpenAI接口），调用Kimi-K2模型，
通过LangChain的MoonshotChat封装进行对话生成。
"""

import os
import logging
from typing import List

from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

logger = logging.getLogger(__name__)

class GenerationIntegrationModule:
    """
    生成集成模块 - 负责LLM集成和回答生成

    本模块承担RAG系统中所有涉及LLM调用的逻辑：

    一、查询分析（在检索之前执行）：
    - query_router: 将用户查询分类为 list/detail/general 三种类型
    - query_rewrite: 对模糊查询进行智能重写，提高检索命中率

    二、答案生成（在检索之后执行）：
    - generate_basic_answer: 通用烹饪问题回答
    - generate_step_by_step_answer: 详细分步骤指导
    - generate_list_answer: 菜品推荐列表（不调用LLM，直接格式化）

    三、流式输出（支持打字机效果的实时文本生成）：
    - generate_basic_answer_stream
    - generate_step_by_step_answer_stream

    LCEL（LangChain Expression Language）的使用：
    本模块使用LCEL的管道操作符 | 构建处理链:
    {"question": ..., "context": ...} | prompt | llm | StrOutputParser()
    这种声明式风格让数据流清晰可读，且天然支持stream()方法。
    """

    def __init__(self, model_name: str = "kimi-k2-0711-preview", temperature: float = 0.1, max_tokens: int = 2048):
        """
        初始化生成集成模块

        构造时立即初始化LLM连接，因为LLM是所有方法的前置依赖。
        初始化失败（如API Key缺失）会在构造阶段暴露，遵循"快速失败"原则。

        Args:
            model_name: Moonshot平台上的模型名称。Kimi-K2系列擅长中文理解和生成
            temperature: 采样温度（0-1），0.1接近确定性输出，
                         适合需要精确食谱信息的场景，减少幻觉
            max_tokens: 单次生成的最大token数，2048足以覆盖大多数食谱回答
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.llm = None
        self.setup_llm()

    def setup_llm(self):
        """
        初始化大语言模型连接

        使用MoonshotChat封装，它是LangChain Community中为Moonshot（月之暗面）
        定制的ChatModel实现，兼容OpenAI的API格式。

        API密钥通过环境变量传递而非代码硬编码，符合安全最佳实践：
        - 避免密钥泄露到版本控制系统
        - 方便在不同环境（开发/测试/生产）间切换

        Raises:
            ValueError: 当MOONSHOT_API_KEY环境变量未设置时抛出
        """
        logger.info(f"正在初始化LLM: {self.model_name}")

        # 从环境变量获取API密钥
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            raise ValueError("请设置 DASHSCOPE_API_KEY 环境变量")

        # self.llm = MoonshotChat(
        #     model=self.model_name,
        #     temperature=self.temperature,
        #     max_tokens=self.max_tokens,
        #     moonshot_api_key=api_key
        # )
        self.llm = ChatOpenAI(
            model="qwen3.6-plus",
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url=os.getenv("DASHSCOPE_BASE_URL")
        )

        logger.info("LLM初始化完成")

    # ==================== 答案生成方法 ====================

    def generate_basic_answer(self, query: str, context_docs: List[Document]) -> str:
        """
        生成通用烹饪问题的基础回答

        适用场景：
        - 食材信息查询（"XX需要什么食材"）
        - 烹饪技巧咨询
        - 一般性的烹饪知识问答

        使用LCEL链实现：合并question和context → 填充prompt模板 →
        调用LLM → 解析输出字符串

        LCEL管道分析：
        - {"question": RunnablePassthrough(), "context": lambda _: context}
          → 构建输入字典，question直接透传原始查询，context由lambda注入
        - | prompt → 将字典中的值填充到模板中
        - | self.llm → 调用LLM生成回答
        - | StrOutputParser() → 从LLM返回的AIMessage中提取纯文本内容

        Args:
            query: 用户原始查询
            context_docs: 检索到的相关食谱文档列表

        Returns:
            LLM生成的纯文本回答
        """
        # 将文档列表格式化为上下文文本块
        context = self._build_context(context_docs)

        # 定义基础回答的prompt模板
        # ChatPromptTemplate.from_template自动将{question}和{context}识别为占位符
        prompt = ChatPromptTemplate.from_template("""
你是一位专业的烹饪助手。请根据以下食谱信息回答用户的问题。

用户问题: {question}

相关食谱信息:
{context}

请提供详细、实用的回答。如果信息不足，请诚实说明。

回答:""")

        # 构建LCEL处理链
        # RunnablePassthrough(): 一个简单的Runnable，直接透传输入值
        # 在这里用于将query字符串原样绑定到"question"键
        # lambda _: context: 忽略输入，始终返回预先构建好的context字符串
        chain = (
            {"question": RunnablePassthrough(), "context": lambda _: context}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        # invoke是同步调用，等待LLM返回完整结果
        response = chain.invoke(query)
        return response

    def generate_step_by_step_answer(self, query: str, context_docs: List[Document]) -> str:
        """
        生成带分步骤指导的详细回答

        与generate_basic_answer的区别：
        - 使用更结构化的prompt模板，要求LLM输出菜品介绍、食材、步骤、技巧等
        - 适合detail类型的查询，即用户想知道"怎么做"的场景
        - 强调实用性和可操作性，鼓励LLM灵活组织而非死板套用模板

        Prompt设计考量：
        - 模板用中文撰写，因为食谱内容和用户查询都是中文，
          LLM在中文prompt下的输出更自然
        - 建议了四个部分但允许LLM灵活调整（"可根据实际内容调整"），
          避免没有技巧内容时LLM强行编造
        - 特别提示"不要强行填充无关内容"，减少LLM的过度发挥

        Args:
            query: 用户原始查询
            context_docs: 检索到的相关食谱文档列表

        Returns:
            结构化的分步骤回答
        """
        context = self._build_context(context_docs)

        prompt = ChatPromptTemplate.from_template("""
你是一位专业的烹饪导师。请根据食谱信息，为用户提供详细的分步骤指导。

用户问题: {question}

相关食谱信息:
{context}

请灵活组织回答，建议包含以下部分（可根据实际内容调整）：

## 🥘 菜品介绍
[简要介绍菜品特点和难度]

## 🛒 所需食材
[列出主要食材和用量]

## 👨‍🍳 制作步骤
[详细的分步骤说明，每步包含具体操作和大概所需时间]

## 💡 制作技巧
[仅在有实用技巧时包含。优先使用原文中的实用技巧，如果原文的"附加内容"与烹饪无关或为空，可以基于制作步骤总结关键要点，或者完全省略此部分]

注意：
- 根据实际内容灵活调整结构
- 不要强行填充无关内容或重复制作步骤中的信息
- 重点突出实用性和可操作性
- 如果没有额外的技巧要分享，可以省略制作技巧部分

回答:""")

        chain = (
            {"question": RunnablePassthrough(), "context": lambda _: context}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        response = chain.invoke(query)
        return response

    def generate_list_answer(self, query: str, context_docs: List[Document]) -> str:
        """
        生成列表式推荐回答 - 适用于"推荐几个XX菜"类查询

        这是唯一不调用LLM的回答生成方法，因为：
        - 列表推荐的逻辑简单明确：提取菜名、去重、格式化
        - 不需要LLM的语言理解和生成能力
        - 直接格式化更快、更稳定，避免了LLM可能的无关发挥

        格式化策略：
        - 1个菜品：简单推荐语句
        - 2-3个菜品：带序号的列表
        - 4个以上菜品：只显示前3个 + "还有X道可供选择"
          限制展示数量避免信息过载

        Args:
            query: 用户原始查询（当前未在方法内使用，保留为接口一致性）
            context_docs: 检索到的相关文档列表

        Returns:
            格式化的菜品推荐列表文本
        """
        if not context_docs:
            return "抱歉，没有找到相关的菜品信息。"

        # 提取并去重菜品名称（保持原始顺序）
        dish_names = []
        for doc in context_docs:
            dish_name = doc.metadata.get('dish_name', '未知菜品')
            if dish_name not in dish_names:
                dish_names.append(dish_name)

        # 根据数量选择不同的展示格式
        if len(dish_names) == 1:
            return f"为您推荐：{dish_names[0]}"
        elif len(dish_names) <= 3:
            return f"为您推荐以下菜品：\n" + "\n".join(
                [f"{i+1}. {name}" for i, name in enumerate(dish_names)]
            )
        else:
            # 展示前3个，提示还有更多
            return (
                f"为您推荐以下菜品：\n" +
                "\n".join([f"{i+1}. {name}" for i, name in enumerate(dish_names[:3])]) +
                f"\n\n还有其他 {len(dish_names)-3} 道菜品可供选择。"
            )

    # ==================== 流式输出方法 ====================

    def generate_basic_answer_stream(self, query: str, context_docs: List[Document]):
        """
        生成基础回答的流式版本（打字机效果）

        与generate_basic_answer的区别：
        - 使用chain.stream()代替chain.invoke()
        - 返回生成器（generator），逐token yield文本片段
        - 调用方使用 for chunk in stream: print(chunk, end="", flush=True) 实现打字效果

        流式输出的意义：
        - 用户体验：逐字显示让等待感大大降低
        - 首token延迟可见：用户能看到系统"在思考"而非"卡住了"

        Args:
            query: 用户查询
            context_docs: 上下文文档列表

        Yields:
            生成的文本片段（每次yield一个token或少量tokens）
        """
        context = self._build_context(context_docs)

        # 使用与普通版本相同的prompt模板
        prompt = ChatPromptTemplate.from_template("""
你是一位专业的烹饪助手。请根据以下食谱信息回答用户的问题。

用户问题: {question}

相关食谱信息:
{context}

请提供详细、实用的回答。如果信息不足，请诚实说明。

回答:""")

        chain = (
            {"question": RunnablePassthrough(), "context": lambda _: context}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        # stream()返回一个生成器，每次yield一个文本片段
        # 调用方迭代这个生成器即可获得流式输出
        for chunk in chain.stream(query):
            yield chunk

    def generate_step_by_step_answer_stream(self, query: str, context_docs: List[Document]):
        """
        生成分步骤回答的流式版本

        与generate_step_by_step_answer功能相同，但以流式方式输出。
        适用于detail类型的查询，让用户逐段阅读分步骤指导。

        Args:
            query: 用户查询
            context_docs: 上下文文档列表

        Yields:
            分步骤回答的文本片段
        """
        context = self._build_context(context_docs)

        prompt = ChatPromptTemplate.from_template("""
你是一位专业的烹饪导师。请根据食谱信息，为用户提供详细的分步骤指导。

用户问题: {question}

相关食谱信息:
{context}

请灵活组织回答，建议包含以下部分（可根据实际内容调整）：

## 🥘 菜品介绍
[简要介绍菜品特点和难度]

## 🛒 所需食材
[列出主要食材和用量]

## 👨‍🍳 制作步骤
[详细的分步骤说明，每步包含具体操作和大概所需时间]

## 💡 制作技巧
[仅在有实用技巧时包含。如果原文的"附加内容"与烹饪无关或为空，可以基于制作步骤总结关键要点，或者完全省略此部分]

注意：
- 根据实际内容灵活调整结构
- 不要强行填充无关内容
- 重点突出实用性和可操作性

回答:""")

        chain = (
            {"question": RunnablePassthrough(), "context": lambda _: context}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        # 流式输出，每次yield一个片段
        for chunk in chain.stream(query):
            yield chunk

    # ==================== 查询分析方法 ====================

    def query_rewrite(self, query: str) -> str:
        """
        智能查询重写 - 通过LLM分析并优化模糊的用户查询

        设计动机：
        用户的自然语言查询往往不够精确，例如：
        - "做菜" → 太宽泛，不知道用户想要什么
        - "川菜" → 是想要菜谱列表还是川菜介绍？
        - "推荐个菜" → 没有偏好信息

        查询重写通过LLM对模糊查询进行扩充和优化，添加关键烹饪术语，
        使其更适合向量检索和BM25检索。

        重写策略：
        1. 具体查询（包含菜名、具体问法）→ 保持原样，避免过度改写
        2. 模糊查询（过于宽泛、缺乏信息）→ 添加烹饪术语使其更具体

        使用PromptTemplate而非ChatPromptTemplate的原因：
        查询重写是简单的文本→文本转换，不需要聊天格式的消息结构

        Args:
            query: 用户原始查询

        Returns:
            重写后的查询（或原查询，如果不需要重写）
        """
        prompt = PromptTemplate(
            template="""
你是一个智能查询分析助手。请分析用户的查询，判断是否需要重写以提高食谱搜索效果。

原始查询: {query}

分析规则：
1. **具体明确的查询**（直接返回原查询）：
   - 包含具体菜品名称：如"宫保鸡丁怎么做"、"红烧肉的制作方法"
   - 明确的制作询问：如"蛋炒饭需要什么食材"、"糖醋排骨的步骤"
   - 具体的烹饪技巧：如"如何炒菜不粘锅"、"怎样调制糖醋汁"

2. **模糊不清的查询**（需要重写）：
   - 过于宽泛：如"做菜"、"有什么好吃的"、"推荐个菜"
   - 缺乏具体信息：如"川菜"、"素菜"、"简单的"
   - 口语化表达：如"想吃点什么"、"有饮品推荐吗"

重写原则：
- 保持原意不变
- 增加相关烹饪术语
- 优先推荐简单易做的
- 保持简洁性

示例：
- "做菜" → "简单易做的家常菜谱"
- "有饮品推荐吗" → "简单饮品制作方法"
- "推荐个菜" → "简单家常菜推荐"
- "川菜" → "经典川菜菜谱"
- "宫保鸡丁怎么做" → "宫保鸡丁怎么做"（保持原查询）
- "红烧肉需要什么食材" → "红烧肉需要什么食材"（保持原查询）

请输出最终查询（如果不需要重写就返回原查询）:""",
            input_variables=["query"]
        )

        chain = (
            {"query": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        # strip()去除LLM输出首尾的空白字符
        response = chain.invoke(query).strip()

        # 记录重写结果，便于分析查询重写的效果
        if response != query:
            logger.info(f"查询已重写: '{query}' → '{response}'")
        else:
            logger.info(f"查询无需重写: '{query}'")

        return response

    def query_router(self, query: str) -> str:
        """
        查询路由 - 将用户查询分类为三种处理类型

        路由类型说明：
        - 'list': 用户想要菜品列表或推荐，后续走generate_list_answer
                  不需要LLM生成，直接格式化结果列表即可
        - 'detail': 用户想要具体制作方法，后续走generate_step_by_step_answer
                    使用结构化prompt，要求分步骤指导
        - 'general': 其他一般性问题，后续走generate_basic_answer
                     使用通用prompt灵活回答

        为什么需要路由：
        不同类型的查询对回答的格式和深度要求不同：
        - "有哪些川菜"只需要列菜名，让LLM自由发挥反而啰嗦
        - "宫保鸡丁怎么做"需要详细步骤，基础回答可能不够结构化
        - 路由让系统为每种查询选择最优的回答策略

        容错处理：
        如果LLM返回的不是三种有效类型之一，默认回退为'general'

        Args:
            query: 用户查询

        Returns:
            路由类型字符串: 'list', 'detail' 或 'general'
        """
        prompt = ChatPromptTemplate.from_template("""
根据用户的问题，将其分类为以下三种类型之一：

1. 'list' - 用户想要获取菜品列表或推荐，只需要菜名
   例如：推荐几个素菜、有什么川菜、给我3个简单的菜

2. 'detail' - 用户想要具体的制作方法或详细信息
   例如：宫保鸡丁怎么做、制作步骤、需要什么食材

3. 'general' - 其他一般性问题
   例如：什么是川菜、制作技巧、营养价值

请只返回分类结果：list、detail 或 general

用户问题: {query}

分类结果:""")

        chain = (
            {"query": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        # 转为小写并去除空白，防止LLM返回 "List" 或 " list " 等变体
        result = chain.invoke(query).strip().lower()

        # 验证分类结果有效性
        if result in ['list', 'detail', 'general']:
            return result
        else:
            # LLM返回了非预期值，默认使用general类型作为安全回退
            return 'general'

    # ==================== 内部辅助方法 ====================

    def _build_context(self, docs: List[Document], max_length: int = 2000) -> str:
        """
        将检索到的文档列表格式化为LLM可读的上下文字符串

        格式化策略：
        1. 为每个文档添加元数据头信息（编号、菜名、分类、难度）
        2. 拼接文档的完整内容
        3. 控制总长度不超过max_length，超出部分截断

        max_length=2000的原因：
        - LLM的上下文窗口有限，需要控制prompt长度
        - 2000字符足以容纳2-3个完整食谱的精华部分
        - 同时也给用户的query和系统指令留出空间
        - 实际调用中，Moonshot API的上下文窗口远大于此，
          这里限制是出于成本控制和响应速度的考虑

        Args:
            docs: 检索到的文档列表
            max_length: 上下文最大字符数限制

        Returns:
            格式化的上下文字符串，使用"======"分隔不同文档
        """
        if not docs:
            return "暂无相关食谱信息。"

        context_parts = []
        current_length = 0

        # 逐个添加文档，直到达到长度上限
        for i, doc in enumerate(docs, 1):
            # 构建文档的元数据头
            metadata_info = f"【食谱 {i}】"
            if 'dish_name' in doc.metadata:
                metadata_info += f" {doc.metadata['dish_name']}"
            if 'category' in doc.metadata:
                metadata_info += f" | 分类: {doc.metadata['category']}"
            if 'difficulty' in doc.metadata:
                metadata_info += f" | 难度: {doc.metadata['difficulty']}"

            # 组装完整文档块：元数据头 + 换行 + 正文内容
            doc_text = f"{metadata_info}\n{doc.page_content}\n"

            # 长度检查：加上当前文档后是否超出限制
            if current_length + len(doc_text) > max_length:
                # 超出限制则停止添加，优先保证已添加文档的完整性
                break

            context_parts.append(doc_text)
            current_length += len(doc_text)

        # 用等号分隔线连接各文档块，使上下文结构清晰
        return "\n" + "="*50 + "\n".join(context_parts)