"""
RAG系统配置文件

本模块定义了食谱RAG系统的所有可配置参数，使用dataclass实现类型安全和默认值管理。
配置项涵盖：数据路径、模型选择、检索和生成参数四大类。
"""

from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class RAGConfig:
    """
    RAG系统配置类

    使用Python dataclass装饰器自动生成__init__、__repr__等方法，
    提供类型提示和默认值，同时支持从字典创建实例和序列化到字典。
    """

    # ==================== 路径配置 ====================
    # 食谱Markdown数据的存放目录，使用相对路径指向data目录
    data_path: str = "../../data/C8/cook"
    # FAISS向量索引的本地保存路径，避免每次重新构建索引
    index_save_path: str = "./vector_index"

    # ==================== 模型配置 ====================
    # 嵌入模型：BGE-small中文版v1.5，由BAAI开源，在中文语义理解上表现优秀
    # 选择small版本而非large版本，在保证效果的同时减少内存占用和推理时间
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    # 大语言模型：Moonshot的Kimi-K2系列，用于生成回答和查询分析
    # 通过DASHSCOPE_API_KEY环境变量进行鉴权
    llm_model: str = "deepseek-v4-pro"

    # ==================== 检索配置 ====================
    # 检索时返回的最相关文档块数量，3是一个平衡值：
    # 太大会引入噪声，太小可能遗漏关键信息
    top_k: int = 3

    # ==================== 生成配置 ====================
    # 温度参数控制生成文本的随机性：
    # 0.1接近确定性输出，适合需要精确食谱信息的场景，减少幻觉
    temperature: float = 0.1
    # 最大生成token数，2048足以覆盖大多数食谱回答的长度需求
    max_tokens: int = 2048

    def __post_init__(self):
        """
        dataclass初始化后自动调用的钩子方法

        当前为空实现，预留用于：
        - 验证配置参数的合法性（如top_k > 0, temperature在0-1之间）
        - 自动创建必要的目录
        - 根据环境变量动态调整配置
        """
        pass

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'RAGConfig':
        """
        从字典创建配置对象（类方法）

        用于从JSON/YAML配置文件加载配置，或从环境变量解析后传入。
        使用**解包将字典键值对直接传给构造函数。

        Args:
            config_dict: 包含配置项的字典，键名需与类属性名一致

        Returns:
            新的RAGConfig实例
        """
        return cls(**config_dict)

    def to_dict(self) -> Dict[str, Any]:
        """
        将配置序列化为字典

        用于保存配置到JSON文件、日志记录或传递给其他需要字典参数的组件。
        手动列举字段而非使用dataclasses.asdict()，避免意外暴露内部属性。

        Returns:
            包含所有配置项的字典
        """
        return {
            'data_path': self.data_path,
            'index_save_path': self.index_save_path,
            'embedding_model': self.embedding_model,
            'llm_model': self.llm_model,
            'top_k': self.top_k,
            'temperature': self.temperature,
            'max_tokens': self.max_tokens
        }

# 默认配置实例，作为全局单例供其他模块导入使用
# 使用者也可以基于RAGConfig创建自定义配置后传入RecipeRAGSystem
DEFAULT_CONFIG = RAGConfig()