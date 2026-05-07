"""
索引构建模块

本模块负责将文档块转换为向量表示，并构建FAISS向量索引，
支持索引的持久化保存与加载，避免每次启动都重新计算嵌入。
"""

import logging
from typing import List
from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

class IndexConstructionModule:
    """
    索引构建模块 - 负责向量化和索引构建

    核心职责：
    1. 初始化嵌入模型（HuggingFace BGE系列）
    2. 将文档块编码为稠密向量
    3. 构建FAISS向量索引以支持高效相似度搜索
    4. 索引的本地持久化（保存/加载）

    嵌入模型选择BGE-small-zh-v1.5的原因：
    - 专为中文优化的双塔模型，在中文语义理解benchmark上表现优异
    - small版本（~100MB）比large版本（~1.3GB）轻量很多，适合本地部署
    - 支持normalize_embeddings，归一化后可直接用内积代替余弦相似度
    - 通过HuggingFaceEmbeddings包装，与LangChain生态无缝集成
    """

    def __init__(self, model_name: str = "BAAI/bge-small-zh-v1.5", index_save_path: str = "./vector_index"):
        """
        初始化索引构建模块

        构造函数中会立即初始化嵌入模型，因为模型加载是必须的前置步骤。
        FAISS索引则在build_vector_index或load_index时才创建。

        Args:
            model_name: HuggingFace模型标识符，默认为BGE中文小模型
            index_save_path: FAISS索引文件的本地保存目录路径
        """
        self.model_name = model_name
        self.index_save_path = index_save_path
        # 嵌入模型实例，后续FAISS操作依赖此对象
        self.embeddings = None
        # FAISS向量存储实例，是检索操作的核心数据结构
        self.vectorstore = None
        # 构造时立即初始化嵌入模型
        self.setup_embeddings()

    def setup_embeddings(self):
        """
        初始化嵌入模型

        配置说明：
        - device='cpu': 在CPU上运行推理。BGE-small模型轻量，CPU推理速度足够，
          且避免了GPU显存占用和CUDA依赖问题
        - normalize_embeddings=True: 输出L2归一化后的向量，使得向量内积等价于余弦相似度，
          FAISS的IndexFlatIP（内积索引）可以直接使用，比IndexFlatL2更高效
        """
        logger.info(f"正在初始化嵌入模型: {self.model_name}")

        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.model_name,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

        logger.info("嵌入模型初始化完成")

    def build_vector_index(self, chunks: List[Document]) -> FAISS:
        """
        构建FAISS向量索引

        工作流程：
        1. 将每个文档块的page_content通过嵌入模型编码为向量
        2. 在内存中构建FAISS索引结构（默认使用IndexFlatIP做精确搜索）
        3. 将所有向量加入索引，同时保留文档的元数据和文本内容

        FAISS.from_documents内部会：
        - 调用self.embeddings.embed_documents()批量编码所有文本
        - 建立向量到文档的映射关系
        - 构建搜索索引结构

        Args:
            chunks: 经过Markdown分块后的文档块列表

        Returns:
            FAISS向量存储对象，可直接用于相似度搜索

        Raises:
            ValueError: 当chunks为空时抛出
        """
        logger.info("正在构建FAISS向量索引...")

        if not chunks:
            raise ValueError("文档块列表不能为空")

        # FAISS.from_documents 是 LangChain 提供的高层API
        # 内部封装了：文本→向量编码 → 构建FAISS索引 → 建立文档映射 的完整流程
        self.vectorstore = FAISS.from_documents(
            documents=chunks,
            embedding=self.embeddings
        )

        logger.info(f"向量索引构建完成，包含 {len(chunks)} 个向量")
        return self.vectorstore

    def add_documents(self, new_chunks: List[Document]):
        """
        向现有索引增量添加新文档

        适用场景：知识库更新时，无需重建整个索引，只需将新文档追加进去。
        注意：FAISS的add操作不会自动去重，如果重复添加同一文档会导致检索结果中出现重复。

        Args:
            new_chunks: 需要新增的文档块列表

        Raises:
            ValueError: 当索引尚未构建时抛出
        """
        if not self.vectorstore:
            raise ValueError("请先构建向量索引")

        logger.info(f"正在添加 {len(new_chunks)} 个新文档到索引...")
        # add_documents内部会自动对新文档进行向量编码并追加到索引
        self.vectorstore.add_documents(new_chunks)
        logger.info("新文档添加完成")

    def save_index(self):
        """
        将FAISS向量索引持久化到磁盘

        保存的内容包括：
        - FAISS索引二进制文件（index.faiss）
        - 文档元数据和文本内容的序列化文件（index.pkl）

        这是构建索引后必须调用的方法，否则下次启动需要重新计算所有嵌入。
        使用Path.mkdir确保父目录存在，避免因目录缺失导致保存失败。

        Raises:
            ValueError: 当索引尚未构建时抛出
        """
        if not self.vectorstore:
            raise ValueError("请先构建向量索引")

        # 确保保存目录存在，parents=True递归创建中间目录
        Path(self.index_save_path).mkdir(parents=True, exist_ok=True)

        # save_local是LangChain FAISS包装器提供的便捷方法
        self.vectorstore.save_local(self.index_save_path)
        logger.info(f"向量索引已保存到: {self.index_save_path}")

    def load_index(self):
        """
        从磁盘加载已保存的FAISS向量索引

        加载逻辑：
        1. 确保嵌入模型已初始化（因为反序列化时需要用到）
        2. 检查索引目录是否存在，不存在则返回None
        3. 尝试加载，失败时返回None而非抛异常，让调用方决定降级策略

        注意allow_dangerous_deserialization=True：
        FAISS索引的pickle反序列化可能执行任意代码，此处设为True是因为
        索引文件由本系统自己生成，可信任，不会加载外部来源的索引。

        Returns:
            成功返回FAISS向量存储对象，失败或不存在返回None
        """
        # 确保嵌入模型已加载，反序列化时需要它来进行查询向量的编码
        if not self.embeddings:
            self.setup_embeddings()

        # 索引目录不存在，说明是首次运行，需要从头构建
        if not Path(self.index_save_path).exists():
            logger.info(f"索引路径不存在: {self.index_save_path}，将构建新索引")
            return None

        try:
            # FAISS.load_local从本地文件恢复完整的向量存储
            # 需要传入embeddings对象以便查询时将文本转为向量
            self.vectorstore = FAISS.load_local(
                self.index_save_path,
                self.embeddings,
                allow_dangerous_deserialization=True  # 信任本地索引文件
            )
            logger.info(f"向量索引已从 {self.index_save_path} 加载")
            return self.vectorstore
        except Exception as e:
            # 加载失败的原因可能包括：文件损坏、版本不兼容、嵌入模型变更等
            # 返回None让上层降级为重新构建索引
            logger.warning(f"加载向量索引失败: {e}，将构建新索引")
            return None

    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        """
        基础相似度搜索

        使用FAISS进行纯粹的向量相似度搜索，不做任何后处理。
        此方法主要供简单场景或调试使用，生产检索推荐使用
        RetrievalOptimizationModule的混合检索以获得更好的效果。

        搜索流程：
        1. 用嵌入模型将query编码为向量
        2. 在FAISS索引中查找与该向量最相似的k个文档
        3. 返回对应的文档对象（包含文本和元数据）

        Args:
            query: 用户查询文本
            k: 返回的最相似文档数量

        Returns:
            按相似度降序排列的文档列表

        Raises:
            ValueError: 当索引尚未构建或加载时抛出
        """
        if not self.vectorstore:
            raise ValueError("请先构建或加载向量索引")

        # similarity_search封装了 query→向量→FAISS搜索→文档映射 的完整流程
        return self.vectorstore.similarity_search(query, k=k)