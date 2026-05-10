"""
Unstructured PDF 解析策略对比

对比 partition_pdf 的两种处理策略：
  - hi_res: 高分辨率模式，使用布局分析模型识别文档结构 + OCR 提取文本
  - ocr_only: 纯 OCR 模式，将 PDF 每页渲染为图片后直接做文字识别

hi_res 的工作原理：
  1. 先用 OCR 提取页面中的所有文本
  2. 再用 Detectron2/YOLOX 等布局分析模型识别每个文本块的角色
     （标题、段落、表格、列表、页眉/页脚等）
  3. 将文本与布局信息结合，返回结构化 Element 列表
  4. 结果最精确，但速度慢，依赖 detectron2 等额外依赖

ocr_only 的工作原理：
  1. 将 PDF 每页渲染为图片
  2. 直接用 Tesseract OCR 识别图片中的文字
  3. 不做布局分析，所有文本统一归类为 NarrativeText 或 Title
  4. 速度中等，适合扫描件 / 无法提取嵌入文本的 PDF

fast（对照）的工作原理：
  1. 直接读取 PDF 中嵌入的文本流（PyMuPDF）
  2. 不做 OCR，不做布局分析
  3. 速度最快，但对扫描件或图片型 PDF 无效

依赖安装:
  pip install unstructured[all-docs]
  hi_res 策略还需要: pip install unstructured-inference
  ocr_only 策略还需要: brew install tesseract（macOS）
"""
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from unstructured.partition.pdf import partition_pdf
from collections import Counter

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
pdf_path = os.path.join(BASE_DIR, "data/C2/pdf/rag.pdf")

# ── 对比三种策略 ──────────────────────────────────────────────────────────────
strategies = ["fast", "hi_res", "ocr_only"]

for strategy in strategies:
    print(f"\n{'='*60}")
    print(f"策略: {strategy}")
    print(f"{'='*60}")

    elements = partition_pdf(
        filename=pdf_path,
        strategy=strategy,
        languages=["chi_sim", "eng"]  # 中英文混合
    )

    # 统计信息


    # 展示所有元素
    print(f"\n  所有元素:")
    for i, element in enumerate(elements, 1):
        print(f"  Element {i} ({element.category}):")
        print(f"  {element}")
        print("  " + "-" * 56)

# ── 策略对比总结（基于 rag.pdf 实际运行结果） ─────────────────────────────────
#
# 策略        | 元素数 | 字符数 | 元素类型种类              | 特点
# ────────────┼───────┼───────┼──────────────────────────┼────────────────────────────────
# fast        |  279  | 7500  | 6 种 (Title 195 为主)     | 直接读嵌入文本，百度页面噪声多
# hi_res      |  227  | 8248  | 8 种 (含 Image 21,        | 用布局分析+OCR，能识别图片/表格，
#             |       |       |     Table 4, Figure 4,    | NarrativeText 从 3 提升到 68，
#             |       |       |     NarrativeText 68)     | 结构识别最精准但速度最慢
# ocr_only    |   87  | 2469  | 3 种 (Title/Narrative/    | 纯 OCR 渲染图片，丢失大量文本，
#             |       |       |     UncategorizedText)    | 仅 87 个元素，信息丢失最严重
#
# 关键观察：
# 1. fast 对于这个 HTML 转 PDF 的文件反而"太准确"了——把百度页面导航栏都识别了出来
# 2. hi_res 是唯一能识别 Table（表格）、Image（图片）、FigureCaption（图片说明）的策略
#    同时 NarrativeText 从 fast 的 3 个提升到 68 个，段落识别显著改善
# 3. ocr_only 只提取了 2469 字符（远少于 fast 的 7500），大量文本被遗漏
#    因为 OCR 渲染时对排版复杂的页面容易出现漏行和错位
#
# 经验法则：
# 1. 原生 PDF（文本可复制）→ 优先用 fast（快且够准）
# 2. 扫描件 / 图片 PDF → 必须用 hi_res 或 ocr_only
# 3. 需要精细结构（表格、图片、标题层级）→ hi_res（唯一选择）
# 4. 中文 PDF 建议显式指定 languages=["chi_sim", "eng"]
# 5. 实际项目中，更常用 PaddleOCR / MinerU 等专业工具处理 PDF
