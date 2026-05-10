import json
import os
import numpy as np
# pymilvus 核心组件：
# - connections/MilvusClient: 连接管理与客户端
# - FieldSchema/CollectionSchema/DataType/Collection: Schema 定义与集合操作
# - AnnSearchRequest: 封装单路向量搜索请求
# - RRFRanker: 倒数排序融合（Reciprocal Rank Fusion）排序器
from pymilvus import connections, MilvusClient, FieldSchema, CollectionSchema, DataType, Collection, AnnSearchRequest, RRFRanker
# BGEM3EmbeddingFunction: BGE-M3 模型，能同时生成稀疏向量（词法）和密集向量（语义）
from pymilvus.model.hybrid import BGEM3EmbeddingFunction

# 1. 初始化设置
# 混合检索（Hybrid Search）结合稀疏向量的关键词精确匹配能力与密集向量的语义理解能力，
# 以克服单一向量检索的局限性，在各类搜索场景下提供更准确、更鲁棒的检索结果。
COLLECTION_NAME = "dragon_hybrid_demo"
MILVUS_URI = "http://localhost:19530"  # 服务器模式（非本地文件模式）
DATA_PATH = "../../data/C4/metadata/dragon.json"  # 相对路径，包含龙类图片的元数据（title、description、category 等）
BATCH_SIZE = 50

# 2. 连接 Milvus 并初始化嵌入模型
print(f"--> 正在连接到 Milvus: {MILVUS_URI}")
connections.connect(uri=MILVUS_URI)

print("--> 正在初始化 BGE-M3 嵌入模型...")
# BGE-M3 是 BAAI 推出的多语言嵌入模型，核心优势是能同时输出两种向量：
# - 密集向量（dense）: 1024 维浮点数向量，用于语义相似度匹配
# - 稀疏向量（sparse）: 基于词权的稀疏表示，每个维度对应词汇表中的一个词，用于关键词精确匹配
# use_fp16=False 使用全精度；device="cpu" 在 CPU 上运行（有 GPU 可改为 "cuda"）
ef = BGEM3EmbeddingFunction(use_fp16=False, device="cpu")
print(f"--> 嵌入模型初始化完成。密集向量维度: {ef.dim['dense']}")

# 3. 创建 Collection
# Collection 是 Milvus 中的数据容器，类比关系型数据库中的"表"。
# 每条数据称为一个 Entity，包含标量字段（元数据）和向量字段。
milvus_client = MilvusClient(uri=MILVUS_URI)
if milvus_client.has_collection(COLLECTION_NAME):
    print(f"--> 正在删除已存在的 Collection '{COLLECTION_NAME}'...")
    milvus_client.drop_collection(COLLECTION_NAME)

# 定义 Schema：混合检索需要同时存储稀疏向量和密集向量两种向量字段
fields = [
    # 主键：auto_id=True 让 Milvus 自动生成唯一 ID，避免手动管理主键冲突
    FieldSchema(name="pk", dtype=DataType.VARCHAR, is_primary=True, auto_id=True, max_length=100),
    # ---- 以下为标量字段（元数据），用于存储搜索结果所需的展示信息 ----
    FieldSchema(name="img_id", dtype=DataType.VARCHAR, max_length=100),
    FieldSchema(name="path", dtype=DataType.VARCHAR, max_length=256),
    FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=256),
    FieldSchema(name="description", dtype=DataType.VARCHAR, max_length=4096),
    FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=64),
    FieldSchema(name="location", dtype=DataType.VARCHAR, max_length=128),
    FieldSchema(name="environment", dtype=DataType.VARCHAR, max_length=64),
    # ---- 向量字段 ----
    # 稀疏向量：SPARSE_FLOAT_VECTOR 类型，维度极高（可达词汇表大小，如 250002），
    # 但绝大多数元素为零。采用倒排索引存储，只保存非零值，极大节省空间。
    # 用于关键词级别的精确匹配（如产品型号、专有名词等）。
    FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
    # 密集向量：FLOAT_VECTOR 类型，固定维度（BGE-M3 为 1024 维），所有位置都有值。
    # 用于捕捉语义相似性（如同义词、上下文理解等）。
    FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=ef.dim["dense"])
]

# 如果集合不存在，则创建它及索引
if not milvus_client.has_collection(COLLECTION_NAME):
    print(f"--> 正在创建 Collection '{COLLECTION_NAME}'...")
    schema = CollectionSchema(fields, description="关于龙的混合检索示例")
    # consistency_level="Strong" 确保写入后立即可见，适用于数据量较小的演示场景
    collection = Collection(name=COLLECTION_NAME, schema=schema, consistency_level="Strong")
    print("--> Collection 创建成功。")

    # 4. 创建索引
    # 索引是向量搜索性能的核心，不同类型向量需要不同的索引策略
    print("--> 正在为新集合创建索引...")
    # 稀疏向量使用 SPARSE_INVERTED_INDEX（倒排索引）：
    # 类似于搜索引擎中的倒排表，将每个词映射到包含该词的文档列表，
    # metric_type="IP"（内积）用于计算查询词权重与文档词权重的匹配程度
    sparse_index = {"index_type": "SPARSE_INVERTED_INDEX", "metric_type": "IP"}
    collection.create_index("sparse_vector", sparse_index)
    print("稀疏向量索引创建成功。")

    # 密集向量使用 AUTOINDEX：
    # Milvus 自动选择最适合当前数据规模和硬件的最优索引类型（如 HNSW、IVF 等），
    # metric_type="IP"（内积）度量向量之间的语义相似度
    dense_index = {"index_type": "AUTOINDEX", "metric_type": "IP"}
    collection.create_index("dense_vector", dense_index)
    print("密集向量索引创建成功。")

# 重新获取 Collection 对象（如果上面创建了新的，确保拿到最新引用）
collection = Collection(COLLECTION_NAME)

# 5. 加载数据并插入
# 将 Collection 加载到内存中，后续的搜索操作均在内存中执行以获得低延迟
collection.load()
print(f"--> Collection '{COLLECTION_NAME}' 已加载到内存。")

if collection.is_empty:
    print(f"--> Collection 为空，开始插入数据...")
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"数据文件未找到: {DATA_PATH}")
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    # 多字段文本合并策略：将 title、description、location、environment 拼接为一段文本，
    # 这样 BGE-M3 在生成向量时能同时考虑多个维度的信息，提升检索质量。
    # 注释掉的 combat_details 和 scene_info 字段可用于扩展更多上下文信息。
    # docs 和 metadata 各有不同用途：
    # - docs:  拼接后的纯文本字符串列表，传给 BGE-M3 生成向量 ef(docs)
    # - metadata: 原始 JSON 对象（dict）列表，用于后续提取标量字段插入 Milvus
    docs, metadata = [], []
    for item in dataset:
        parts = [
            item.get('title', ''),
            item.get('description', ''),
            item.get('location', ''),
            item.get('environment', ''),
            # *item.get('combat_details', {}).get('combat_style', []),
            # *item.get('combat_details', {}).get('abilities_used', []),
            # item.get('scene_info', {}).get('time_of_day', '')
        ]
        # filter(None, parts) 过滤掉空字符串，避免产生多余空格
        docs.append(' '.join(filter(None, parts)))
        metadata.append(item)
    print(f"--> 数据加载完成，共 {len(docs)} 条。")

    print("--> 正在生成向量嵌入...")
    # BGE-M3 的 __call__ 方法同时生成两种向量：
    # embeddings["dense"]: 语义向量（1024维浮点数数组），用于理解查询意图
    # embeddings["sparse"]: 词法向量（稀疏格式），用于关键词级别的精确匹配
    embeddings = ef(docs)
    print("--> 向量生成完成。")

    print("--> 正在分批插入数据...")
    # 为每个字段准备批量数据
    # 字段顺序必须严格遵循 Schema 中定义字段的顺序（7个标量 + 2个向量）
    img_ids = [doc["img_id"] for doc in metadata]
    paths = [doc["path"] for doc in metadata]
    titles = [doc["title"] for doc in metadata]
    descriptions = [doc["description"] for doc in metadata]
    categories = [doc["category"] for doc in metadata]
    locations = [doc["location"] for doc in metadata]
    environments = [doc["environment"] for doc in metadata]

    # 获取向量
    sparse_vectors = embeddings["sparse"]   # 稀疏向量：词法/关键词权重（类比 BM25 的索引:权重映射）
    dense_vectors = embeddings["dense"]     # 密集向量：语义编码（类比 BERT 输出的 embedding）

    # 插入数据：将所有字段按 Schema 顺序一次性批量插入
    # 参数顺序必须与 FieldSchema 的定义顺序一致
    collection.insert([
        img_ids,
        paths,
        titles,
        descriptions,
        categories,
        locations,
        environments,
        sparse_vectors,
        dense_vectors
    ])

    # flush() 将内存缓冲区中的数据强制持久化到磁盘，并构建索引，使数据立即可被搜索
    collection.flush()
    print(f"--> 数据插入完成，总数: {collection.num_entities}")
else:
    print(f"--> Collection 中已有 {collection.num_entities} 条数据，跳过插入。")

# 6. 执行搜索
# 混合检索的核心流程：
# 1) 对查询文本同时生成密集向量和稀疏向量
# 2) 分别执行两路独立的 ANN 搜索（近似最近邻搜索）
# 3) 使用 RRF（Reciprocal Rank Fusion）算法融合两路结果
search_query = "悬崖上的巨龙"
# 标量过滤：仅搜索指定类别的数据，减少无关联数据的干扰
search_filter = 'category in ["western_dragon", "chinese_dragon", "movie_character"]'
top_k = 5  # 返回最相似的 top_k 条结果

print(f"\n{'='*20} 开始混合搜索 {'='*20}")
print(f"查询: '{search_query}'")
print(f"过滤器: '{search_filter}'")

# 对查询文本生成向量：调用 ef([query]) 同时获得密集和稀疏两种向量表示
query_embeddings = ef([search_query])
# dense_vec: 形状为 (1024,) 的 numpy 数组，用于语义搜索
dense_vec = query_embeddings["dense"][0]
# sparse_vec: 稀疏向量对象，._getrow(0) 获取第1行（即唯一的查询行）
# 稀疏向量内部结构类似 {词ID: 权重} 的映射，非零元素极少（本例中仅6个非零值 / 250002 维）
sparse_vec = query_embeddings["sparse"]._getrow(0)

# 打印向量信息
# 以下输出有助于理解两种向量的结构差异：
# - 密集向量：1024个浮点数全部有值，范数为1（已归一化）
# - 稀疏向量：250002维但仅6个非零元素，密度约0.0024%，每个非零值对应一个具体的关键词权重
print("\n=== 向量信息 ===")
print(f"密集向量维度: {len(dense_vec)}")
print(f"密集向量前5个元素: {dense_vec[:5]}")
print(f"密集向量范数: {np.linalg.norm(dense_vec):.4f}")
# 密集向量范数接近1.0，表示向量已经过归一化，可直接用于内积/IP 相似度计算

print(f"\n稀疏向量维度: {sparse_vec.shape[1]}")
print(f"稀疏向量非零元素数量: {sparse_vec.nnz}")
print("稀疏向量前5个非零元素:")
# 打印稀疏向量的非零索引及对应权重，每个索引对应 BGE-M3 词表中的一个具体 token
for i in range(min(5, sparse_vec.nnz)):
    print(f"  - 索引: {sparse_vec.indices[i]}, 值: {sparse_vec.data[i]:.4f}")
# 稀疏密度 = 非零元素数 / 总维度，反映向量的稀疏程度
density = (sparse_vec.nnz / sparse_vec.shape[1] * 100)
print(f"\n稀疏向量密度: {density:.8f}%")
# 极低的密度说明稀疏向量只激活了极少数的关键词维度，存储效率极高

# 定义搜索参数
# metric_type="IP": 使用内积（Inner Product）作为相似度度量
# 对于已归一化的向量，内积等价于余弦相似度
# params={} 表示使用默认的搜索超参数
search_params = {"metric_type": "IP", "params": {}}

# 先执行单独的搜索
# ---- 密集向量搜索 ----
# 基于语义相似度：找到与查询"悬崖上的巨龙"在语义空间中最接近的图片描述。
# 即使描述中没有出现"悬崖"这个词，只要语义相近（如"山崖"、"峭壁"）也可能被匹配。
print("\n--- [单独] 密集向量搜索结果 ---")
dense_results = collection.search(
    [dense_vec],             # 查询向量（列表形式，支持批量搜索）
    anns_field="dense_vector",  # 指定搜索的向量字段
    param=search_params,
    limit=top_k,
    expr=search_filter,      # 标量过滤表达式，缩小搜索范围
    output_fields=["title", "path", "description", "category", "location", "environment"]  # 返回字段
)[0]  # [0] 取第一个（也是唯一一个）查询的结果

for i, hit in enumerate(dense_results):
    # hit.distance 是内积得分，分数越高表示语义越相似
    print(f"{i+1}. {hit.entity.get('title')} (Score: {hit.distance:.4f})")
    print(f"    路径: {hit.entity.get('path')}")
    print(f"    描述: {hit.entity.get('description')[:100]}...")

# ---- 稀疏向量搜索 ----
# 基于关键词精确匹配：稀疏向量搜索相当于对文档中的关键词进行加权匹配。
# 如果查询中的关键词（如"悬崖"、"龙"）在文档中出现且权重较高，则该文档得分高。
# 与密集搜索的区别：它能精确匹配专有名词和特定术语，但无法理解同义词。
print("\n--- [单独] 稀疏向量搜索结果 ---")
sparse_results = collection.search(
    [sparse_vec],
    anns_field="sparse_vector",
    param=search_params,
    limit=top_k,
    expr=search_filter,
    output_fields=["title", "path", "description", "category", "location", "environment"]
)[0]

for i, hit in enumerate(sparse_results):
    print(f"{i+1}. {hit.entity.get('title')} (Score: {hit.distance:.4f})")
    print(f"    路径: {hit.entity.get('path')}")
    print(f"    描述: {hit.entity.get('description')[:100]}...")

# ---- 混合搜索（密集 + 稀疏） ----
# RRF（Reciprocal Rank Fusion，倒数排序融合）核心原理：
# 不关心各路检索系统的原始得分，只关心每个文档在各自结果集中的排名 rank。
# 公式: RRF_score(d) = Σ 1 / (rank_i(d) + k)，其中 k 是常数，i 遍历各检索系统。
# k=60 是经验默认值：k 越大，排名靠前的结果权重差距越小、排序越平滑；
# k 越小，高排名结果的权重越突出。
# 这种方法的优势是无需对不同检索系统的得分进行归一化，直接基于排名融合。
print("\n--- [混合] 稀疏+密集向量搜索结果 ---")
# 创建 RRF 融合器，k=60 控制排序平滑程度
rerank = RRFRanker(k=60)

# 创建搜索请求：将两路独立的搜索封装为 AnnSearchRequest 对象
dense_req = AnnSearchRequest([dense_vec], "dense_vector", search_params, limit=top_k)
sparse_req = AnnSearchRequest([sparse_vec], "sparse_vector", search_params, limit=top_k)

# 执行混合搜索
# hybrid_search 内部流程:
# 1) 并行执行 sparse_req 和 dense_req 两路搜索
# 2) 用 RRFRanker 对两路结果进行排名融合
# 3) 返回融合后的 top_k 条结果
# 注意: 混合搜索的 score 是 RRF 融合得分（非原始内积得分），
# 因此数值范围与单独搜索不同（通常远小于1），不宜跨搜索类型直接比较得分。
results = collection.hybrid_search(
    [sparse_req, dense_req],
    rerank=rerank,
    limit=top_k,
    output_fields=["title", "path", "description", "category", "location", "environment"]
)[0]

# 打印最终结果
for i, hit in enumerate(results):
    print(f"{i+1}. {hit.entity.get('title')} (Score: {hit.distance:.4f})")
    print(f"    路径: {hit.entity.get('path')}")
    print(f"    描述: {hit.entity.get('description')[:100]}...")
# 混合检索结果解读：
# 与单独搜索相比，混合检索往往能发现仅靠单一路径难以排到前列的结果。
# 例如"霸王龙的怒吼"可能在密集搜索中排名靠后（缺少"悬崖"语义），
# 但在稀疏搜索中因为精确匹配了"龙"关键词而得分较高，
# 经过 RRF 融合后，其在最终结果中的排名得以提升。

# 7. 清理资源
# release_collection: 从内存中卸载 Collection，释放内存资源（Collection 元数据和索引保留）
milvus_client.release_collection(collection_name=COLLECTION_NAME)
print(f"已从内存中释放 Collection: '{COLLECTION_NAME}'")
# drop_collection: 彻底删除 Collection 及其所有数据和索引（不可恢复）
milvus_client.drop_collection(COLLECTION_NAME)
print(f"已删除 Collection: '{COLLECTION_NAME}'")