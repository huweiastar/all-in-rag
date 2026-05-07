"""
Visual-BGE 多模态嵌入模型演示

本示例展示 Visualized_BGE 模型的核心能力 —— 将图像和文本编码到统一的向量空间中，
使得图像、文本、图文组合三者之间可以直接计算相似度。

Visualized_BGE 是 BAAI 提出的多模态扩展版本，基于 BGE 文本嵌入模型，
通过额外的视觉编码器将图像信息融合到文本嵌入空间中。

核心概念：
  - 纯文本向量：model.encode(text="...") —— 只编码文本语义
  - 纯图像向量：model.encode(image="...") —— 只编码图像视觉特征
  - 多模态向量：model.encode(image="...", text="...") —— 同时编码图文信息

三种向量在同一个空间中，可以用向量内积（@ 运算符）直接计算余弦相似度（因为向量已归一化）。

前置条件：
  - Visual-BGE 模型权重已下载（运行 download_model.py）
  - 测试图片已放入 ../../data/C3/imgs/ 目录
"""
import os
# 国内环境使用 HuggingFace 镜像加速模型下载
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import torch
from visual_bge.visual_bge.modeling import Visualized_BGE

# ── 第一步：加载 Visualized_BGE 多模态模型 ─────────────────────────────────────
# model_name_bge: 底层文本嵌入模型（BGE-base-en-v1.5 输出 768 维向量）
# model_weight: 预训练的视觉投影层权重（将图像特征映射到文本嵌入空间）
# .eval() 将模型切换到推理模式，关闭 Dropout 等训练专用层
model = Visualized_BGE(model_name_bge="BAAI/bge-base-en-v1.5",
                      model_weight="../../models/bge/Visualized_base_en_v1.5.pth")
model.eval()

# ── 第二步：生成多种类型的嵌入向量 ─────────────────────────────────────────────
# torch.no_grad() 上下文禁用梯度计算，节省内存并加速推理
with torch.no_grad():
    # 纯文本编码：将文本描述转为向量
    text_emb = model.encode(text="datawhale开源组织的logo")

    # 纯图像编码：将图片转为向量（不依赖任何文字描述）
    img_emb_1 = model.encode(image="../../data/C3/imgs/datawhale01.png")

    # 图文联合编码：同时输入图片和文字，得到一个融合了视觉+语义的多模态向量
    # 这种方式通常比单独的图像或文本向量更具判别力
    multi_emb_1 = model.encode(image="../../data/C3/imgs/datawhale01.png", text="datawhale开源组织的logo")

    # 另一张图片的纯图像和多模态向量（用于对比）
    img_emb_2 = model.encode(image="../../data/C3/imgs/datawhale02.png")
    multi_emb_2 = model.encode(image="../../data/C3/imgs/datawhale02.png", text="datawhale开源组织的logo")

# ── 第三步：计算跨模态相似度 ─────────────────────────────────────────────────
# 由于模型输出的向量已经过 L2 归一化，向量内积等价于余弦相似度
# 值域为 [-1, 1]，越接近 1 表示越相似
sim_1 = img_emb_1 @ img_emb_2.T       # 纯图像 vs 纯图像：两张图的视觉相似度
sim_2 = img_emb_1 @ multi_emb_1.T     # 图文结合 vs 纯图像：加入文本后与纯图像的差异
sim_3 = text_emb @ multi_emb_1.T      # 图文结合 vs 纯文本：多模态向量与纯文本的语义对齐程度
sim_4 = multi_emb_1 @ multi_emb_2.T   # 图文结合 vs 图文结合：两张图的全面相似度

print("=== 相似度计算结果 ===")
print(f"纯图像 vs 纯图像: {sim_1}")
print(f"图文结合1 vs 纯图像: {sim_2}")
print(f"图文结合1 vs 纯文本: {sim_3}")
print(f"图文结合1 vs 图文结合2: {sim_4}")

# ── 第四步：分析向量维度信息 ─────────────────────────────────────────────────
print("\n=== 嵌入向量信息 ===")
# Visualized_BGE 输出的向量维度 = 底层 BGE 文本模型的维度（base 版本为 768）
print(f"多模态向量维度: {multi_emb_1.shape}")
print(f"图像向量维度: {img_emb_1.shape}")
# 查看前 10 个元素，可以观察到不同编码方式的向量值分布差异
print(f"多模态向量示例 (前10个元素): {multi_emb_1[0][:10]}")
print(f"图像向量示例 (前10个元素):   {img_emb_1[0][:10]}")
