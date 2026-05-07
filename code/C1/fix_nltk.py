"""
NLTK 数据下载脚本

UnstructuredMarkdownLoader（LangChain 的文档加载器）内部依赖 NLTK 库进行文本处理，
特别是以下两个数据包：

1. punkt       - 句子分词器（Sentence Tokenizer），用于将一段文本切分为独立的句子
2. averaged_perceptron_tagger - 词性标注器（Part-of-Speech Tagger），用于识别每个词的词性

首次运行时会从网络下载数据包到本地 NLTK data 目录。
如果遇到网络问题导致下载失败，可尝试以下方法：

    # 设置 NLTK 数据目录
    import nltk
    nltk.data.path.append('/path/to/nltk_data')

    # 或者手动下载数据包后放到 nltk_data 目录
    # https://www.nltk.org/nltk_data/
"""
import nltk

# force=True 表示强制重新下载，覆盖已有数据
nltk.download('punkt', force=True)
nltk.download('averaged_perceptron_tagger', force=True)