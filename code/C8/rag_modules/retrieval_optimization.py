"""
检索优化模块

本模块实现了混合检索策略，将向量检索（语义匹配）与BM25检索（关键词匹配）结合，
通过RRF（Reciprocal Rank Fusion）算法融合两者的排序结果，兼顾语义理解和精确匹配。

同时提供元数据过滤功能，支持按菜品分类、难度等维度缩小检索范围。
"""

import logging
from typing import List, Dict, Any

from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

class RetrievalOptimizationModule:
    """
    检索优化模块 - 负责混合检索和元数据过滤

    设计思路：
    - 向量检索（FAISS）：擅长语义匹配，能理解"好吃的"与"美味的"含义相近，
      但可能漏掉精确的关键词匹配，比如"糖醋排骨"中的"糖醋"
    - BM25检索：基于词频-逆文档频率的经典算法，擅长精确关键词匹配，
      但无法处理同义词和语义相关性
    - RRF融合：不依赖分数绝对值，仅根据排名位置融合，避免了两种不同
      量纲的分数难以直接比较的问题
    - 元数据过滤：在检索结果上叠加分类/难度等结构化条件

    混合检索 = 向量检索 + BM25检索 → RRF重排 → 元数据过滤 → 最终结果
    """

    def __init__(self, vectorstore: FAISS, chunks: List[Document]):
        """
        初始化检索优化模块

        需要同时传入FAISS向量存储和文档块列表，因为：
        - FAISS用于向量相似度检索
        - 文档块列表用于构建BM25索引（BM25需要对全文建倒排索引）

        Args:
            vectorstore: 已构建或加载的FAISS向量存储实例
            chunks: 全部文档块列表，用于初始化BM25检索器
        """
        self.vectorstore = vectorstore
        self.chunks = chunks
        # 构造时立即初始化两个检索器
        self.setup_retrievers()

    def setup_retrievers(self):
        """
        初始化向量检索器和BM25检索器

        向量检索器：
        - search_type="similarity": 使用标准余弦相似度（因为嵌入已做了L2归一化）
        - k=5: 每种检索器各召回5个，共10个候选，RRF重排后取top_k个

        BM25检索器：
        - 基于scikit-learn风格实现的Okapi BM25算法
        - 需要传入全部文档块来构建倒排索引和计算IDF值
        - k=5: 与向量检索器保持一致，确保两种检索方式贡献相当

        两种检索器返回相同数量的候选，避免某一种检索方式主导最终结果。
        """
        logger.info("正在设置检索器...")

        # 向量检索器：基于FAISS索引的语义相似度搜索
        # as_retriever()将底层FAISS的相似度搜索封装为LangChain标准的Retriever接口
        self.vector_retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )

        # BM25检索器：基于词频统计的关键词匹配
        # from_documents在初始化时会：
        # 1. 对全部文档进行分词
        # 2. 计算每个词的IDF（逆文档频率）值
        # 3. 构建倒排索引以支持快速检索
        self.bm25_retriever = BM25Retriever.from_documents(
            self.chunks,
            k=5
        )

        logger.info("检索器设置完成")

    def hybrid_search(self, query: str, top_k: int = 3) -> List[Document]:
        """
        混合检索 - 结合向量检索和BM25检索，使用RRF重排融合结果

        流程：
        1. 分别用向量检索器和BM25检索器搜索
        2. 对两组结果进行RRF重排（Reciprocal Rank Fusion）
        3. 返回融合后排名最高的top_k个文档

        为什么使用RRF而不是简单的分数加权：
        - 向量相似度的范围是[-1,1]（余弦），BM25的分数没有上界
        - 两种分数的分布和量纲完全不同，直接加权需要大量调参
        - RRF只关心"相对排名"而非"绝对分数"，天然解决了量纲问题
        - RRF公式: score(d) = Σ 1/(k + rank_i(d))，其中k是平滑参数

        Args:
            query: 用户查询文本
            top_k: 最终返回的文档数量

        Returns:
            经过RRF重排后的top_k个文档列表
        """
        # 第一步：分别获取两组检索结果
        # invoke()是LangChain Retriever的标准调用接口
        vector_docs = self.vector_retriever.invoke(query)
        bm25_docs = self.bm25_retriever.invoke(query)

        # 第二步：使用RRF算法融合排序
        reranked_docs = self._rrf_rerank(vector_docs, bm25_docs)

        # 第三步：截取前top_k个
        return reranked_docs[:top_k]

    def metadata_filtered_search(self, query: str, filters: Dict[str, Any], top_k: int = 5) -> List[Document]:
        """
        带元数据过滤的检索

        实现策略：先检索后过滤
        - 先通过混合检索获取 top_k * 3 个候选文档（扩大候选池以免过滤后结果不足）
        - 再逐文档检查元数据是否满足过滤条件
        - 收集到足够数量后提前终止

        为什么不使用FAISS自带的过滤？
        - FAISS的过滤功能需要额外的索引结构和内存开销
        - 对于本系统的数据规模（数百到数千篇食谱），后过滤的性能完全够用
        - 后过滤实现更简单、更灵活，支持复杂的复合条件

        Args:
            query: 用户查询文本
            filters: 元数据过滤条件字典，如 {"category": "荤菜", "difficulty": "简单"}
                     值可以是单个字符串或字符串列表（支持多选）
            top_k: 期望返回的文档数量

        Returns:
            满足过滤条件的文档列表，长度不超过top_k
        """
        # 先进行混合检索，获取top_k * 3个候选
        # 乘3是为了应对过滤后数量不足的情况：
        # 如果只检索top_k个，过滤条件严格时可能返回0个结果
        docs = self.hybrid_search(query, top_k * 3)

        # 逐文档检查过滤条件
        filtered_docs = []
        for doc in docs:
            match = True
            for key, value in filters.items():
                # 检查元数据中是否存在该键
                if key in doc.metadata:
                    # 支持列表类型的值：doc的元数据值只需在列表中即可
                    if isinstance(value, list):
                        if doc.metadata[key] not in value:
                            match = False
                            break
                    # 标量值：要求精确匹配
                    else:
                        if doc.metadata[key] != value:
                            match = False
                            break
                else:
                    # 文档元数据中没有该键，视为不匹配
                    match = False
                    break

            if match:
                filtered_docs.append(doc)
                # 提前终止：已收集到足够的文档
                if len(filtered_docs) >= top_k:
                    break

        return filtered_docs

    def _rrf_rerank(self, vector_docs: List[Document], bm25_docs: List[Document], k: int = 60) -> List[Document]:
        """
        使用RRF（Reciprocal Rank Fusion）算法融合两组检索结果

        RRF算法原理：
        - 对于每个文档，其在每个排序列表中的贡献为 1/(k + rank)
        - rank从1开始计数（不是从0开始）
        - k是平滑参数，用于减轻高位排名（rank=1,2）的权重差距
        - 最终分数 = 向量检索的RRF分数 + BM25检索的RRF分数

        k值的选取：
        - k=60是经验值，源自RRF原始论文的实验结论
        - 较小的k（如10）会让高位排名的权重差异更大
        - 较大的k（如100）会让排名趋于均匀
        - 60在大多数场景下是一个稳健的默认值

        以文档内容哈希值作为唯一标识的原因：
        - Document对象可能在两次检索中是不同的实例（即使内容相同）
        - 哈希值确保了内容相同的文档被正确识别为同一文档

        Args:
            vector_docs: 向量检索结果列表（已按相似度排序）
            bm25_docs: BM25检索结果列表（已按BM25分数排序）
            k: RRF平滑参数，默认60

        Returns:
            按融合分数降序排列的文档列表
        """
        # doc_scores: 存储每个文档的累计RRF分数
        # doc_objects: 存储文档对象引用，用于最终返回
        doc_scores = {}
        doc_objects = {}

        # 处理向量检索结果
        # enumerate从0开始，rank = index + 1
        for rank, doc in enumerate(vector_docs):
            # 使用page_content的哈希值作为文档唯一标识
            # 注意：Python的hash()在不同进程间可能不同，但在同一进程内是一致的
            doc_id = hash(doc.page_content)
            doc_objects[doc_id] = doc

            # RRF核心公式: score = 1 / (k + rank)
            # rank从1开始（rank + 1），rank=0 → 1/(60+1)=1/61 ≈ 0.0164
            rrf_score = 1.0 / (k + rank + 1)
            # 如果同一文档同时出现在两组结果中，分数累加
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + rrf_score

            logger.debug(f"向量检索 - 文档{rank+1}: RRF分数 = {rrf_score:.4f}")

        # 处理BM25检索结果（逻辑与向量检索相同）
        for rank, doc in enumerate(bm25_docs):
            doc_id = hash(doc.page_content)
            doc_objects[doc_id] = doc

            rrf_score = 1.0 / (k + rank + 1)
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + rrf_score

            logger.debug(f"BM25检索 - 文档{rank+1}: RRF分数 = {rrf_score:.4f}")

        # 按最终RRF分数从高到低排序
        # sorted返回 (doc_id, final_score) 元组列表
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)

        # 构建按RRF分数排序的最终文档列表
        reranked_docs = []
        for doc_id, final_score in sorted_docs:
            if doc_id in doc_objects:
                doc = doc_objects[doc_id]
                # 将RRF融合分数存入元数据，便于调试和分析检索质量
                doc.metadata['rrf_score'] = final_score
                reranked_docs.append(doc)
                logger.debug(f"最终排序 - 文档: {doc.page_content[:50]}... 最终RRF分数: {final_score:.4f}")

        logger.info(
            f"RRF重排完成: 向量检索{len(vector_docs)}个文档, "
            f"BM25检索{len(bm25_docs)}个文档, 合并后{len(reranked_docs)}个文档"
        )

        return reranked_docs