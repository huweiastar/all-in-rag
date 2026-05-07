"""
基于 Milvus + Visual-BGE 的多模态图文检索示例

本示例演示了如何使用 Milvus 向量数据库搭建一个支持"图像 + 文本"联合检索的系统：
  - 用 Visualized_BGE 模型将图像和文本编码为统一的多模态向量
  - 将所有图像的向量存入 Milvus
  - 支持三种检索方式：纯文本搜图、纯图搜图、图文联合搜图
  - 检索结果以拼接全景图形式可视化展示

核心流程：
  1. 初始化多模态编码器 + Milvus 客户端
  2. 创建 Milvus Collection（定义字段 Schema）
  3. 批量编码图像并插入向量库
  4. 为向量字段创建 HNSW 索引
  5. 执行多模态检索（图像 + 文本联合查询）
  6. 可视化检索结果（拼接全景图）
  7. 释放资源（释放内存 + 删除 Collection）

前置条件:
  - Milvus 服务需通过 docker compose 启动（监听 localhost:19530）
  - Visual-BGE 模型权重已下载到 ../../models/bge/
  - 数据目录 ../../data/C3/dragon/ 中包含多张 .png 图片

依赖安装:
  pip install pymilvus torch torchvision opencv-python pillow tqdm
"""
import os
from tqdm import tqdm
from glob import glob
import torch
from visual_bge.visual_bge.modeling import Visualized_BGE
from pymilvus import MilvusClient, FieldSchema, CollectionSchema, DataType
import numpy as np
import cv2
from PIL import Image

# ── 第一步：初始化配置常量 ──────────────────────────────────────────────────────
MODEL_NAME = "BAAI/bge-base-en-v1.5"        # 基础嵌入模型名（用于 Visual-BGE）
MODEL_PATH = "../../models/bge/Visualized_base_en_v1.5.pth"  # Visual-BGE 权重文件路径
DATA_DIR = "../../data/C3"                   # 数据根目录
COLLECTION_NAME = "multimodal_demo"          # Milvus Collection 名称
MILVUS_URI = "http://localhost:19530"        # Milvus 服务地址（Docker 默认端口）

# ── 第二步：定义工具类 ──────────────────────────────────────────────────────────

class Encoder:
    """
    多模态编码器，封装 Visualized_BGE 模型，提供图像和文本的向量编码接口。

    Visualized_BGE 的特点：
      - 可以将纯文本编码为向量
      - 可以将纯图像编码为向量
      - 可以将图像 + 文本同时编码为统一的多模态向量（图文联合表征）
      - 三者在同一个向量空间中，可以直接计算余弦相似度
    """
    def __init__(self, model_name: str, model_path: str):
        # 加载 Visualized_BGE 模型
        self.model = Visualized_BGE(model_name_bge=model_name, model_weight=model_path)
        self.model.eval()  # 切换到推理模式（关闭 Dropout 等训练专用层）

    def encode_query(self, image_path: str, text: str) -> list[float]:
        """
        将图像 + 文本联合编码为多模态查询向量。
        用于"图文联合检索"场景，例如：用一张龙的照片 + 文字"喷火的龙"来搜图。
        """
        with torch.no_grad():  # 禁用梯度计算，节省内存并加速推理
            query_emb = self.model.encode(image=image_path, text=text)
        return query_emb.tolist()[0]  # 从 torch tensor 转为 Python 列表

    def encode_image(self, image_path: str) -> list[float]:
        """
        将纯图像编码为向量。
        用于批量建立图像索引，或纯图搜图场景。
        """
        with torch.no_grad():
            query_emb = self.model.encode(image=image_path)
        return query_emb.tolist()[0]


def visualize_results(query_image_path: str, retrieved_images: list,
                      img_height: int = 300, img_width: int = 300,
                      row_count: int = 3) -> np.ndarray:
    """
    将查询图像和检索结果拼接为一张全景图用于可视化。

    布局示意（3x3 网格）：
    ┌──────────┬─────┬─────┬─────┐
    │          │  R0 │  R1 │  R2 │
    │  查询区   ├─────┼─────┼─────┤
    │          │  R3 │  R4 │  R5 │
    │          ├─────┼─────┼─────┤
    │  Query   │  R6 │  R7 │  R8 │
    └──────────┴─────┴─────┴─────┘

    参数:
        query_image_path: 查询图像的本地路径
        retrieved_images: 检索到的图像路径列表
        img_height/img_width: 每个小图的显示尺寸
        row_count: 结果区的行/列数（形成 row_count × row_count 网格）
    返回:
        拼接好的全景图（numpy 数组，BGR 格式，可直接用 cv2.imwrite 保存）
    """
    # 创建画布：左侧是查询图展示区，右侧是结果网格
    panoramic_width = img_width * row_count
    panoramic_height = img_height * row_count
    panoramic_image = np.full((panoramic_height, panoramic_width, 3), 255, dtype=np.uint8)
    query_display_area = np.full((panoramic_height, img_width, 3), 255, dtype=np.uint8)

    # ── 处理查询图像（左侧区域）────────────────────────────────────────────────
    # 读取图像并转换：PIL RGB → numpy → OpenCV BGR
    query_pil = Image.open(query_image_path).convert("RGB")
    query_cv = np.array(query_pil)[:, :, ::-1]  # RGB 通道反转变为 BGR（OpenCV 格式）
    resized_query = cv2.resize(query_cv, (img_width, img_height))
    # 加红色边框（BGR 中红色 = (255, 0, 0)）标识这是查询图
    bordered_query = cv2.copyMakeBorder(resized_query, 10, 10, 10, 10,
                                        cv2.BORDER_CONSTANT, value=(255, 0, 0))
    query_display_area[img_height * (row_count - 1):, :] = cv2.resize(bordered_query, (img_width, img_height))
    cv2.putText(query_display_area, "Query", (10, panoramic_height - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # ── 处理检索到的图像（右侧网格区域）────────────────────────────────────────
    for i, img_path in enumerate(retrieved_images):
        row, col = i // row_count, i % row_count          # 计算网格中的行列位置
        start_row, start_col = row * img_height, col * img_width  # 计算在画布中的起始坐标

        retrieved_pil = Image.open(img_path).convert("RGB")
        retrieved_cv = np.array(retrieved_pil)[:, :, ::-1]  # RGB → BGR
        # 缩小 2 像素留白边
        resized_retrieved = cv2.resize(retrieved_cv, (img_width - 4, img_height - 4))
        # 加黑色边框（BGR 中黑色 = (0, 0, 0)）
        bordered_retrieved = cv2.copyMakeBorder(resized_retrieved, 2, 2, 2, 2,
                                                cv2.BORDER_CONSTANT, value=(0, 0, 0))
        # 将图像放入画布对应位置
        panoramic_image[start_row:start_row + img_height,
                        start_col:start_col + img_width] = bordered_retrieved

        # 在左上角标注序号（红色文字）
        cv2.putText(panoramic_image, str(i), (start_col + 10, start_row + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # 将左侧查询区和右侧结果区水平拼接
    return np.hstack([query_display_area, panoramic_image])


# ── 第三步：初始化编码器和 Milvus 客户端 ────────────────────────────────────────
print("--> 正在初始化编码器和Milvus客户端...")
encoder = Encoder(MODEL_NAME, MODEL_PATH)
# MilvusClient 是 pymilvus 的轻量级客户端，适合快速原型开发
# 生产环境建议使用 connections.connect + Collection 类
milvus_client = MilvusClient(uri=MILVUS_URI)

# ── 第四步：创建 Milvus Collection ─────────────────────────────────────────────
print(f"\n--> 正在创建 Collection '{COLLECTION_NAME}'")
# 如果 Collection 已存在，先删除再重建（保证干净的状态）
if milvus_client.has_collection(COLLECTION_NAME):
    milvus_client.drop_collection(COLLECTION_NAME)
    print(f"已删除已存在的 Collection: '{COLLECTION_NAME}'")

# 扫描数据目录中的所有 .png 图像
image_list = glob(os.path.join(DATA_DIR, "dragon", "*.png"))
if not image_list:
    raise FileNotFoundError(f"在 {DATA_DIR}/dragon/ 中未找到任何 .png 图像。")

# 用第一张图探测向量维度（Visualized_BGE 的维度取决于底层 bge-base 模型）
dim = len(encoder.encode_image(image_list[0]))

# ── 定义 Collection 的字段 Schema ─────────────────────────────────────────────
# Milvus 的 Schema 类似关系型数据库的表结构定义：
#   FieldSchema 定义每个字段的名称、类型、约束
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    # id 是主键，auto_id=True 表示 Milvus 自动分配递增 ID（无需手动指定）

    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dim),
    # vector 是向量字段，FLOAT_VECTOR 表示 32 位浮点向量
    # dim 必须与编码器输出的维度一致

    FieldSchema(name="image_path", dtype=DataType.VARCHAR, max_length=512),
    # image_path 是标量字段，存储图像的本地文件路径（用于检索后展示）
    # VARCHAR 类型需要指定 max_length
]

# 将字段列表包装为 Collection Schema（附带描述）
schema = CollectionSchema(fields, description="多模态图文检索")
print("Schema 结构:")
print(schema)

# 在 Milvus 中创建 Collection
milvus_client.create_collection(collection_name=COLLECTION_NAME, schema=schema)
print(f"成功创建 Collection: '{COLLECTION_NAME}'")
print("Collection 结构:")
print(milvus_client.describe_collection(collection_name=COLLECTION_NAME))

# ── 第五步：批量编码图像并插入向量库 ───────────────────────────────────────────
print(f"\n--> 正在向 '{COLLECTION_NAME}' 插入数据")
data_to_insert = []
for image_path in tqdm(image_list, desc="生成图像嵌入"):
    # 将每张图像编码为向量
    vector = encoder.encode_image(image_path)
    # 准备插入数据（id 字段省略，因为 auto_id=True）
    data_to_insert.append({"vector": vector, "image_path": image_path})

# 批量插入所有数据
if data_to_insert:
    result = milvus_client.insert(collection_name=COLLECTION_NAME, data=data_to_insert)
    print(f"成功插入 {result['insert_count']} 条数据。")

# ── 第六步：为向量字段创建 HNSW 索引 ──────────────────────────────────────────
print(f"\n--> 正在为 '{COLLECTION_NAME}' 创建索引")
# Milvus 中向量检索必须先创建索引，否则无法执行搜索
# HNSW（Hierarchical Navigable Small World）是一种近似最近邻（ANN）算法：
#   - 构建一个多层图结构，上层是下层的稀疏子图
#   - 搜索时从顶层开始快速导航到目标区域，再逐层细化
#   - 优点：精度高、速度快；缺点：内存占用较大
index_params = milvus_client.prepare_index_params()
index_params.add_index(
    field_name="vector",
    index_type="HNSW",          # 索引类型
    metric_type="COSINE",       # 相似度度量：余弦相似度（值越接近 1 越相似）
    params={
        "M": 16,                # 每个节点的最大连接数，越大精度越高但构建越慢
        "efConstruction": 256   # 构建索引时的搜索宽度，越大索引质量越好
    }
)
milvus_client.create_index(collection_name=COLLECTION_NAME, index_params=index_params)
print("成功为向量字段创建 HNSW 索引。")
print("索引详情:")
print(milvus_client.describe_index(collection_name=COLLECTION_NAME, index_name="vector"))

# 将 Collection 加载到内存（创建索引后必须执行此操作才能搜索）
milvus_client.load_collection(collection_name=COLLECTION_NAME)
print("已加载 Collection 到内存中。")

# ── 第七步：执行多模态检索 ─────────────────────────────────────────────────────
print(f"\n--> 正在 '{COLLECTION_NAME}' 中执行检索")
# 查询条件：一张龙的图片 + 文字描述"一条龙"
# Visualized_BGE 会将图像和文本联合编码为一个多模态向量
query_image_path = os.path.join(DATA_DIR, "dragon", "query.png")
query_text = "一条龙"
query_vector = encoder.encode_query(image_path=query_image_path, text=query_text)

# 在 Milvus 中执行向量搜索
# search 返回的是一个二维列表：外层是查询列表（本例只有 1 个查询），
# 内层是该查询命中的结果列表（按相似度降序排列）
search_results = milvus_client.search(
    collection_name=COLLECTION_NAME,
    data=[query_vector],          # 查询向量列表（支持批量查询）
    output_fields=["image_path"], # 需要返回的额外字段（默认只返回 id 和 distance）
    limit=5,                      # 返回最相似的前 5 条结果
    search_params={
        "metric_type": "COSINE",  # 与建索引时保持一致
        "params": {"ef": 128}    # 搜索时的遍历宽度，越大结果越精确（需 >= limit）
    }
)[0]  # 取第一个查询的结果（因为只有一个查询）

# 解析并打印结果
retrieved_images = []
print("检索结果:")
for i, hit in enumerate(search_results):
    # hit 结构示例：{"id": 123, "distance": 0.95, "entity": {"image_path": "/path/to/img.png"}}
    print(f"  Top {i+1}: ID={hit['id']}, 距离={hit['distance']:.4f}, 路径='{hit['entity']['image_path']}'")
    retrieved_images.append(hit['entity']['image_path'])

# ── 第八步：可视化结果并清理资源 ───────────────────────────────────────────────
print(f"\n--> 正在可视化结果并清理资源")
if not retrieved_images:
    print("没有检索到任何图像。")
else:
    # 拼接全景图并保存
    panoramic_image = visualize_results(query_image_path, retrieved_images)
    combined_image_path = os.path.join(DATA_DIR, "search_result.png")
    cv2.imwrite(combined_image_path, panoramic_image)
    print(f"结果图像已保存到: {combined_image_path}")
    # 在默认图片查看器中打开
    Image.open(combined_image_path).show()

# ── 清理：释放 Milvus 资源 ─────────────────────────────────────────────────────
# release_collection: 将 Collection 从内存中卸载，释放 Milvus 内存
milvus_client.release_collection(collection_name=COLLECTION_NAME)
print(f"已从内存中释放 Collection: '{COLLECTION_NAME}'")

# drop_collection: 永久删除 Collection 及其所有数据（包括索引和向量）
milvus_client.drop_collection(collection_name=COLLECTION_NAME)
print(f"已删除 Collection: '{COLLECTION_NAME}'")

# ── 关键概念速查 ────────────────────────────────────────────────────────────────
#
# Milvus 层级结构:
#   Collection（集合）≈ 关系型数据库中的"表"
#     ├── Field（字段）≈ 表中的"列"
#     │     ├── INT64（整数）
#     │     ├── FLOAT_VECTOR（浮点向量）
#     │     └── VARCHAR（字符串）
#     └── Index（索引）
#           ├── FLAT（精确搜索，暴力遍历）
#           ├── IVF_FLAT（倒排文件，分块搜索）
#           ├── HNSW（图结构，层次导航）
#           └── ANNOY（随机投影森林）
#
# Visualized_BGE 编码模式:
#   model.encode(text="...")          → 纯文本向量
#   model.encode(image="...")         → 纯图像向量
#   model.encode(image="...", text="") → 图文联合向量（多模态）
#   三者空间统一，可直接计算余弦相似度