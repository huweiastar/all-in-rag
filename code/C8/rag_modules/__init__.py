"""
RAG系统模块包

本包包含食谱RAG系统的四个核心模块：
- DataPreparationModule: 数据加载、清洗、分块
- IndexConstructionModule: 向量嵌入与FAISS索引构建
- RetrievalOptimizationModule: 混合检索（向量+BM25）与RRF重排
- GenerationIntegrationModule: LLM集成、查询分析与回答生成
"""

# 从各子模块导入核心类，使用户可以直接通过包名访问
# 例如: from rag_modules import DataPreparationModule
from .data_preparation import DataPreparationModule
from .index_construction import IndexConstructionModule
from .retrieval_optimization import RetrievalOptimizationModule
from .generation_integration import GenerationIntegrationModule

# __all__控制 from rag_modules import * 的行为
# 明确列出公开API，避免意外导出内部实现细节
__all__ = [
    'DataPreparationModule',
    'IndexConstructionModule',
    'RetrievalOptimizationModule',
    'GenerationIntegrationModule'
]

# 包版本号，遵循语义化版本规范
__version__ = "1.0.0"