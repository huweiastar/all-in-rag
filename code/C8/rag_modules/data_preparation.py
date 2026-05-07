"""
数据准备模块

本模块负责RAG系统的数据预处理全流程：
1. 从Markdown文件加载食谱文档
2. 自动提取和增强元数据（分类、菜品名称、难度等级）
3. 基于Markdown标题结构进行语义感知的文档分块
4. 维护父子文档映射关系，支持从小块回溯到完整文档
5. 提供文档过滤、统计和元数据导出功能
"""

import logging
import hashlib
from typing import List, Dict, Any

from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_core.documents import Document
from pathlib import Path
import uuid

logger = logging.getLogger(__name__)

class DataPreparationModule:
    """
    数据准备模块 - 负责数据加载、清洗和预处理

    数据组织结构：
    - 父文档（parent document）: 完整的食谱Markdown文件
    - 子文档（child document/chunk）: 按Markdown标题拆分后的文档块
    - 父子映射（parent_child_map）: 记录每个子块属于哪个父文档，用于检索后还原完整内容

    元数据层级：
    - source: 原始文件路径
    - parent_id: 父文档的MD5哈希（基于相对路径，确定性）
    - doc_type: "parent"（完整文档）或 "child"（分块）
    - category: 菜品分类（荤菜、素菜、汤品等），从目录结构提取
    - dish_name: 菜品名称，从文件名提取
    - difficulty: 难度星级（★），从文本内容分析得出
    - chunk_id: 子块的UUID（随机生成）
    - chunk_index: 子块在父文档中的位置序号
    """

    # 统一维护的分类映射字典，key为目录名（英文），value为显示用中文名
    # 定义为类属性而非实例属性，方便外部通过类方法直接访问
    CATEGORY_MAPPING = {
        'meat_dish': '荤菜',
        'vegetable_dish': '素菜',
        'soup': '汤品',
        'dessert': '甜品',
        'breakfast': '早餐',
        'staple': '主食',
        'aquatic': '水产',
        'condiment': '调料',
        'drink': '饮品'
    }
    # 使用set去重得到唯一的分类标签列表
    CATEGORY_LABELS = list(set(CATEGORY_MAPPING.values()))
    # 难度等级从低到高排列，用于UI展示和过滤条件
    DIFFICULTY_LABELS = ['非常简单', '简单', '中等', '困难', '非常困难']

    def __init__(self, data_path: str):
        """
        初始化数据准备模块

        Args:
            data_path: 食谱Markdown文件的根目录路径。
                       目录结构应为：data_path/{category}/{dish_name}.md
                       例如：data/C8/cook/meat_dish/宫保鸡丁.md
        """
        self.data_path = data_path
        # 父文档列表：存储完整的食谱Markdown文件内容
        self.documents: List[Document] = []
        # 子文档列表：存储按标题分块后的小块
        self.chunks: List[Document] = []
        # 父文档ID -> 子块ID列表 的映射，当前使用子块ID -> 父文档ID 的单向映射
        # 用于从检索到的子块还原对应的完整父文档
        self.parent_child_map: Dict[str, str] = {}

    def load_documents(self) -> List[Document]:
        """
        从文件系统加载所有食谱Markdown文档

        加载流程：
        1. 递归扫描data_path下所有.md文件
        2. 读取每个文件的内容（保持原始Markdown格式）
        3. 为每个文档生成确定性的parent_id（基于相对路径的MD5哈希）
        4. 为每个文档自动增强元数据（分类、菜名、难度）

        使用MD5哈希作为parent_id的优势：
        - 确定性：同一文件每次运行生成相同ID，索引加载后父子关系仍然一致
        - 唯一性：不同文件路径的哈希碰撞概率极低

        Returns:
            加载的Document对象列表，每个Document包含完整食谱内容和元数据
        """
        logger.info(f"正在从 {self.data_path} 加载文档...")

        documents = []
        data_path_obj = Path(self.data_path)

        # rglob("*.md") 递归匹配所有Markdown文件
        for md_file in data_path_obj.rglob("*.md"):
            try:
                # 直接以UTF-8编码读取原始Markdown内容
                # 保留所有Markdown格式（标题、列表、加粗等），不进行HTML转换
                with open(md_file, 'r', encoding='utf-8') as f:
                    content = f.read()

                # 生成确定性的父文档ID
                # 使用相对路径而非绝对路径，确保在不同机器上的一致性
                try:
                    data_root = Path(self.data_path).resolve()
                    relative_path = Path(md_file).resolve().relative_to(data_root).as_posix()
                except Exception:
                    # 如果计算相对路径失败（异常场景），回退使用完整路径
                    relative_path = Path(md_file).as_posix()
                # MD5生成32字符的十六进制字符串作为唯一标识
                parent_id = hashlib.md5(relative_path.encode("utf-8")).hexdigest()

                # 创建LangChain Document对象
                # page_content存储文档文本，metadata存储结构化属性
                doc = Document(
                    page_content=content,
                    metadata={
                        "source": str(md_file),     # 完整文件路径
                        "parent_id": parent_id,      # 确定性唯一标识
                        "doc_type": "parent"         # 标记为父文档类型
                    }
                )
                documents.append(doc)

            except Exception as e:
                # 单个文件读取失败不应中断整个加载流程
                logger.warning(f"读取文件 {md_file} 失败: {e}")

        # 批量为所有文档增强元数据（分类、菜名、难度）
        for doc in documents:
            self._enhance_metadata(doc)

        self.documents = documents
        logger.info(f"成功加载 {len(documents)} 个文档")
        return documents

    def _enhance_metadata(self, doc: Document):
        """
        从文件路径和文档内容中自动提取和增强元数据

        提取逻辑：
        1. 分类（category）：从文件路径的目录名匹配CATEGORY_MAPPING
           例如 .../meat_dish/宫保鸡丁.md → category = "荤菜"
        2. 菜品名称（dish_name）：取文件名的stem部分（去掉.md后缀）
           例如 宫保鸡丁.md → dish_name = "宫保鸡丁"
        3. 难度（difficulty）：从文档内容中统计★符号的数量
           ★ = 非常简单, ★★ = 简单, ..., ★★★★★ = 非常困难

        这些元数据后续用于：
        - 检索时的元数据过滤（按分类/难度筛选）
        - 回答生成时提供菜品背景信息
        - 统计分析和UI展示

        Args:
            doc: 需要增强元数据的Document对象（原地修改）
        """
        file_path = Path(doc.metadata.get('source', ''))
        path_parts = file_path.parts

        # 提取分类：遍历目录路径的每一层，匹配CATEGORY_MAPPING
        # 例如 path_parts包含 ['data', 'C8', 'cook', 'meat_dish', '宫保鸡丁.md']
        # 匹配到 'meat_dish' → 映射为 '荤菜'
        doc.metadata['category'] = '其他'  # 默认值，防止空分类
        for key, value in self.CATEGORY_MAPPING.items():
            if key in path_parts:
                doc.metadata['category'] = value
                break

        # 提取菜品名称：文件名去掉.md后缀
        # Path.stem 自动去除扩展名
        doc.metadata['dish_name'] = file_path.stem

        # 分析难度等级：统计文档内容中★符号的数量
        # 使用if-elif链按数量从多到少匹配，避免 ★ 匹配到 ★★★ 的子串
        content = doc.page_content
        if '★★★★★' in content:
            doc.metadata['difficulty'] = '非常困难'
        elif '★★★★' in content:
            doc.metadata['difficulty'] = '困难'
        elif '★★★' in content:
            doc.metadata['difficulty'] = '中等'
        elif '★★' in content:
            doc.metadata['difficulty'] = '简单'
        elif '★' in content:
            doc.metadata['difficulty'] = '非常简单'
        else:
            doc.metadata['difficulty'] = '未知'

    @classmethod
    def get_supported_categories(cls) -> List[str]:
        """
        对外提供支持的分类标签列表

        定义为类方法而非实例方法，方便在未创建实例时获取，
        例如在main.py的_extract_filters_from_query中直接通过类名调用。

        Returns:
            分类标签列表，如 ['荤菜', '素菜', '汤品', ...]
        """
        return cls.CATEGORY_LABELS

    @classmethod
    def get_supported_difficulties(cls) -> List[str]:
        """
        对外提供支持的难度标签列表

        Returns:
            难度标签列表，按星级从低到高排列
        """
        return cls.DIFFICULTY_LABELS

    def chunk_documents(self) -> List[Document]:
        """
        对已加载的文档执行Markdown结构感知分块

        分块策略：
        - 使用Markdown标题层级（#, ##, ###）作为分割边界
        - 保留标题文本而非剥离，确保每个chunk的上下文完整
        - 每个chunk继承父文档的所有元数据并追加自己的chunk_id

        为什么用Markdown结构分块而非固定长度分块：
        - 食谱文档有清晰的标题结构（菜名→原料→步骤），
          在标题边界切分保持了语义完整性
        - 固定长度切分可能在句子中间切断，破坏语义连贯性
        - 结构化分块后每个chunk对应一个明确的主题段落

        Returns:
            分块后的Document列表，每个都带有父文档关联信息

        Raises:
            ValueError: 当文档尚未加载时抛出
        """
        logger.info("正在进行Markdown结构感知分块...")

        if not self.documents:
            raise ValueError("请先加载文档")

        # 调用Markdown标题分割器进行实际分块
        chunks = self._markdown_header_split()

        # 为每个chunk补充基础索引信息
        for i, chunk in enumerate(chunks):
            if 'chunk_id' not in chunk.metadata:
                # 如果是分割失败回退的完整文档，补充生成chunk_id
                chunk.metadata['chunk_id'] = str(uuid.uuid4())
            # batch_index记录在整个chunks列表中的位置
            chunk.metadata['batch_index'] = i
            # chunk_size用于统计分析和日志展示
            chunk.metadata['chunk_size'] = len(chunk.page_content)

        self.chunks = chunks
        logger.info(f"Markdown分块完成，共生成 {len(chunks)} 个chunk")
        return chunks

    def _markdown_header_split(self) -> List[Document]:
        """
        使用Markdown标题分割器对每个父文档进行结构化分割

        分割层级配置：
        - # (H1): 主标题，通常是菜品名称
        - ## (H2): 二级标题，如"必备原料"、"计算"、"操作"
        - ### (H3): 三级标题，如"简易版本"、"复杂版本"

        strip_headers=False的原因：
        保留标题行在chunk内容中，这样检索时chunk自身就具备
        完整的上下文信息，不需要额外回溯到父文档获取章节标题。

        特别处理：
        - 无标题文档：日志warn提示，但不会中断流程
        - 只有一个chunk：说明文档没有匹配的标题结构，标记warn
        - 分割异常：降级为将整个文档作为一个chunk，保证数据不丢失

        Returns:
            所有文档的结构化分块列表
        """
        # 定义要识别和切分的标题层级及其显示名称
        # headers_to_split_on的格式: [(标题符号, 元数据键名), ...]
        headers_to_split_on = [
            ("#", "主标题"),      # 菜品名称级别
            ("##", "二级标题"),   # 原料、计算、操作等大段标题
            ("###", "三级标题")   # 子版本变体标题
        ]

        # 创建Markdown分割器实例
        # strip_headers=False: 保留标题行在chunk内容中，确保上下文完整
        markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on,
            strip_headers=False
        )

        all_chunks = []

        for doc in self.documents:
            try:
                # 预检查：快速扫描文档前200字符是否包含Markdown标题
                # 这对性能影响极小，但可以帮助提前发现格式问题
                content_preview = doc.page_content[:200]
                has_headers = any(line.strip().startswith('#')
                                  for line in content_preview.split('\n'))

                if not has_headers:
                    logger.warning(
                        f"文档 {doc.metadata.get('dish_name', '未知')} "
                        f"内容中没有发现Markdown标题"
                    )
                    logger.debug(f"内容预览: {content_preview}")

                # 调用LangChain的MarkdownHeaderTextSplitter进行分割
                # split_text会返回一个Document列表，每个Document对应一个标题区域
                md_chunks = markdown_splitter.split_text(doc.page_content)

                logger.debug(
                    f"文档 {doc.metadata.get('dish_name', '未知')} "
                    f"分割成 {len(md_chunks)} 个chunk"
                )

                # 仅1个chunk意味着分割器没有找到有效的标题边界
                if len(md_chunks) <= 1:
                    logger.warning(
                        f"文档 {doc.metadata.get('dish_name', '未知')} "
                        f"未能按标题分割，可能缺少标题结构"
                    )

                # 获取父文档ID，用于建立父子关联
                parent_id = doc.metadata["parent_id"]

                for i, chunk in enumerate(md_chunks):
                    # 每个子块分配一个随机UUID作为唯一标识
                    # 使用UUID而非MD5的原因：同一父文档的多个子块内容不同，
                    # 不能用基于内容的哈希（太长的内容hash性能差），UUID更方便
                    child_id = str(uuid.uuid4())

                    # 关键步骤：继承父文档元数据 + 追加子块特有元数据
                    # update会覆盖同名键，所以父文档的元数据优先保留
                    chunk.metadata.update(doc.metadata)
                    chunk.metadata.update({
                        "chunk_id": child_id,       # 子块唯一标识
                        "parent_id": parent_id,     # 所属父文档标识（关键！）
                        "doc_type": "child",        # 标记为子文档类型
                        "chunk_index": i            # 在父文档中的位置序号
                    })

                    # 建立子块→父文档的映射关系
                    # 这是检索后还原完整文档的关键桥梁
                    self.parent_child_map[child_id] = parent_id

                all_chunks.extend(md_chunks)

            except Exception as e:
                # 分割失败时的降级策略：将整个父文档作为一个chunk
                # 这保证了即使格式异常，该食谱的信息仍然可以被检索到
                logger.warning(
                    f"文档 {doc.metadata.get('source', '未知')} "
                    f"Markdown分割失败: {e}"
                )
                all_chunks.append(doc)

        logger.info(
            f"Markdown结构分割完成，生成 {len(all_chunks)} 个结构化块"
        )
        return all_chunks

    def filter_documents_by_category(self, category: str) -> List[Document]:
        """
        按菜品分类过滤父文档

        适用场景：
        - 用户询问"有哪些荤菜"
        - 按分类展示食谱列表
        - 统计分析各分类的食谱数量

        Args:
            category: 菜品分类标签，如 "荤菜", "素菜", 需与CATEGORY_MAPPING的值一致

        Returns:
            匹配分类的父文档列表（列表推导式实现，简洁高效）
        """
        return [
            doc for doc in self.documents
            if doc.metadata.get('category') == category
        ]

    def filter_documents_by_difficulty(self, difficulty: str) -> List[Document]:
        """
        按难度等级过滤父文档

        适用场景：
        - 用户询问"有哪些简单易做的菜"
        - 难度筛选辅助

        Args:
            difficulty: 难度等级，如 "简单", "中等"

        Returns:
            匹配难度的父文档列表
        """
        return [
            doc for doc in self.documents
            if doc.metadata.get('difficulty') == difficulty
        ]

    def get_statistics(self) -> Dict[str, Any]:
        """
        统计已加载数据的整体概况

        统计维度：
        - 文档总数和分块总数
        - 各分类的食谱数量分布
        - 各难度等级的食谱数量分布
        - 平均chunk大小（可用于评估分块策略是否合理）

        使用场景：
        - 知识库构建完成后打印统计信息
        - 前端展示数据看板
        - 监控数据质量和规模

        Returns:
            包含多维度统计数据的字典，文档未加载时返回空字典
        """
        if not self.documents:
            return {}

        categories = {}
        difficulties = {}

        for doc in self.documents:
            # 统计各分类的文档数量
            category = doc.metadata.get('category', '未知')
            categories[category] = categories.get(category, 0) + 1

            # 统计各难度等级的文档数量
            difficulty = doc.metadata.get('difficulty', '未知')
            difficulties[difficulty] = difficulties.get(difficulty, 0) + 1

        # 计算平均chunk大小：总字符数 / chunk数量
        # 避免除零错误
        avg_chunk_size = (
            sum(chunk.metadata.get('chunk_size', 0) for chunk in self.chunks)
            / len(self.chunks)
        ) if self.chunks else 0

        return {
            'total_documents': len(self.documents),
            'total_chunks': len(self.chunks),
            'categories': categories,
            'difficulties': difficulties,
            'avg_chunk_size': avg_chunk_size
        }

    def export_metadata(self, output_path: str):
        """
        将所有文档的元数据导出为JSON文件

        用途：
        - 数据审计：检查元数据提取的准确性
        - 数据分析：对食谱库进行统计分析
        - 调试：快速查看所有文档的分类和难度标注情况

        导出的字段：文件路径、菜品名称、分类、难度、内容长度

        Args:
            output_path: JSON文件的输出路径
        """
        import json

        metadata_list = []
        for doc in self.documents:
            metadata_list.append({
                'source': doc.metadata.get('source'),
                'dish_name': doc.metadata.get('dish_name'),
                'category': doc.metadata.get('category'),
                'difficulty': doc.metadata.get('difficulty'),
                'content_length': len(doc.page_content)
            })

        # ensure_ascii=False 保证中文正确显示而非被转义为\\uXXXX
        # indent=2 使JSON格式可读
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(metadata_list, f, ensure_ascii=False, indent=2)

        logger.info(f"元数据已导出到: {output_path}")

    def get_parent_documents(self, child_chunks: List[Document]) -> List[Document]:
        """
        根据检索到的子块还原对应的完整父文档（带智能去重和排序）

        核心逻辑：
        1. 统计每个父文档被匹配的次数（作为相关性指标）
           - 匹配次数多 = 该父文档的多个子块与查询相关 = 高度相关
        2. 按相关性降序排列父文档
        3. 自动去重：同一父文档的多个子块只返回一个父文档

        为什么需要这个映射：
        - 检索在小块（chunks）上进行，获得精细的匹配粒度
        - 但回答生成需要完整食谱内容，而非片段
        - 通过parent_id反向查找，还原完整文档

        相关性排序的意义：
        - 有3个子块匹配的菜谱比只有1个匹配的更可能满足用户需求
        - 这使得最终回答优先展示最相关的完整文档

        Args:
            child_chunks: 检索阶段返回的子块列表

        Returns:
            去重并按相关性降序排列的完整父文档列表
        """
        # parent_relevance: 统计每个父文档被匹配的子块数量
        # parent_docs_map: 缓存父文档对象，避免对同一文档重复查找
        parent_relevance = {}
        parent_docs_map = {}

        for chunk in child_chunks:
            parent_id = chunk.metadata.get("parent_id")
            if parent_id:
                # 累加该父文档的相关性计数
                parent_relevance[parent_id] = parent_relevance.get(parent_id, 0) + 1

                # 首次遇到该父文档时，从self.documents中找到完整文档并缓存
                if parent_id not in parent_docs_map:
                    for doc in self.documents:
                        if doc.metadata.get("parent_id") == parent_id:
                            parent_docs_map[parent_id] = doc
                            break

        # 按匹配次数降序排列parent_id
        # 匹配次数越多说明该文档与查询越相关
        sorted_parent_ids = sorted(
            parent_relevance.keys(),
            key=lambda x: parent_relevance[x],
            reverse=True
        )

        # 构建去重后按相关性排序的父文档列表
        parent_docs = []
        for parent_id in sorted_parent_ids:
            if parent_id in parent_docs_map:
                parent_docs.append(parent_docs_map[parent_id])

        # 记录日志：展示每个父文档的相关性信息
        parent_info = []
        for doc in parent_docs:
            dish_name = doc.metadata.get('dish_name', '未知菜品')
            parent_id = doc.metadata.get('parent_id')
            relevance_count = parent_relevance.get(parent_id, 0)
            parent_info.append(f"{dish_name}({relevance_count}块)")

        logger.info(
            f"从 {len(child_chunks)} 个子块中找到 {len(parent_docs)} "
            f"个去重父文档: {', '.join(parent_info)}"
        )
        return parent_docs
