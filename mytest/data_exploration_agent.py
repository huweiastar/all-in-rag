#!/usr/bin/env python3
"""
Titanic Data Exploration Agent — OpenAI client + DashScope (Qwen 3.6 Plus)

Three capabilities via natural-language conversation:
  1. Data summary  — mean, variance, max, min, distribution
  2. Visualization — plot any column (count/hist/box/bar)
  3. ML training   — sklearn RandomForest to predict Survived

Uses the OpenAI Python client directly to avoid langchain version conflicts.
"""

import os
import sys
import json
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

# ── matplotlib non-interactive backend ──
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# Chinese-friendly font fallback
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

from dotenv import load_dotenv
from openai import OpenAI

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# ═══════════════════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════════════════
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

DATA_PATH = os.path.join(os.path.dirname(__file__), "titanic_cleaned.csv")
OUTPUT_DIR = os.path.dirname(__file__)

MODEL = os.getenv("MODEL", "qwen-plus")
BASE_URL = os.getenv("DASHSCOPE_BASE_URL")
API_KEY = os.getenv("DASHSCOPE_API_KEY")

# ═══════════════════════════════════════════════════════════════════════
#  LOAD DATA
# ═══════════════════════════════════════════════════════════════════════
df = pd.read_csv(DATA_PATH)
print(f"[INFO] Loaded {len(df)} rows, {len(df.columns)} columns")


# ═══════════════════════════════════════════════════════════════════════
#  TOOLS  (plain Python functions, each returns a string result)
# ═══════════════════════════════════════════════════════════════════════

def tool_data_summary(dummy: str = "") -> str:
    """Get comprehensive statistics: mean, variance, max, min, quartiles, missing values, Survived distribution."""
    lines = []

    lines.append("=" * 60)
    lines.append("📊 数据集基本信息")
    lines.append("=" * 60)
    lines.append(f"行数: {len(df)}")
    lines.append(f"列数: {len(df.columns)}")
    lines.append(f"列名: {list(df.columns)}")

    missing = df.isnull().sum()
    missing = missing[missing > 0]
    lines.append(f"\n缺失值统计:")
    lines.append(missing.to_string() if len(missing) > 0 else "无缺失值")

    lines.append("\n" + "=" * 60)
    lines.append("📈 描述性统计 (count / mean / std / min / 25% / 50% / 75% / max)")
    lines.append("=" * 60)
    lines.append(df.describe().to_string())

    lines.append("\n" + "=" * 60)
    lines.append("📐 数值列方差 (Variance)")
    lines.append("=" * 60)
    num_cols = df.select_dtypes(include=[np.number]).columns
    lines.append(df[num_cols].var().to_string())

    lines.append("\n" + "=" * 60)
    lines.append("🎯 目标变量 Survived 分布")
    lines.append("=" * 60)
    vc = df["Survived"].value_counts()
    vc_pct = df["Survived"].value_counts(normalize=True) * 100
    lines.append(f"  0 (未存活): {vc.get(0, 0):5d}  ({vc_pct.get(0, 0):.2f}%)")
    lines.append(f"  1 (存活):   {vc.get(1, 0):5d}  ({vc_pct.get(1, 0):.2f}%)")

    cat_cols = df.select_dtypes(include=["object"]).columns
    if len(cat_cols):
        lines.append("\n" + "=" * 60)
        lines.append("📋 分类列概览")
        lines.append("=" * 60)
        for c in cat_cols:
            top5 = df[c].value_counts().head(5)
            lines.append(f"\n  {c} ({df[c].nunique()} 个唯一值):")
            lines.append(f"    Top 5: {dict(top5)}")

    return "\n".join(lines)


def tool_create_plot(column_name: str, plot_type: str = "count") -> str:
    """Create and save a statistical chart (count/hist/box/bar) for a given column."""
    if column_name not in df.columns:
        return f"❌ 列 '{column_name}' 不存在。可用列: {list(df.columns)}"

    fig, ax = plt.subplots(figsize=(10, 6))

    if plot_type == "count":
        if df[column_name].dtype == object or df[column_name].nunique() < 12:
            sns.countplot(data=df, x=column_name, hue=column_name,
                          palette="Set2", ax=ax, legend=False)
        else:
            sns.histplot(data=df, x=column_name, bins=30, kde=True, ax=ax)
            plot_type = "hist"
        ax.set_title(f"Distribution of {column_name}", fontsize=14, fontweight="bold")

    elif plot_type == "hist":
        sns.histplot(data=df, x=column_name, bins=30, kde=True, ax=ax)
        ax.set_title(f"Histogram of {column_name} (with KDE)", fontsize=14, fontweight="bold")

    elif plot_type == "box":
        sns.boxplot(data=df, y=column_name, ax=ax)
        ax.set_title(f"Box Plot of {column_name}", fontsize=14, fontweight="bold")

    elif plot_type == "bar":
        vc = df[column_name].value_counts()
        bars = ax.bar(range(len(vc)), vc.values,
                      color=sns.color_palette("Set2", len(vc)))
        ax.set_xticks(range(len(vc)))
        ax.set_xticklabels(vc.index, rotation=45)
        ax.set_title(f"Bar Chart of {column_name}", fontsize=14, fontweight="bold")
        for bar, val in zip(bars, vc.values):
            ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.5,
                    str(val), ha="center", va="bottom")
    else:
        return f"❌ 不支持的图表类型: {plot_type}。支持: count, hist, box, bar"

    ax.set_xlabel(column_name)
    ax.set_ylabel("Count" if plot_type in ("count", "bar") else "Frequency")
    plt.tight_layout()

    filepath = os.path.join(OUTPUT_DIR, f"plot_{column_name}_{plot_type}.png")
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return f"✅ 图表已保存: {filepath}  (类型={plot_type}, 列={column_name})"


def tool_train_model(target_column: str = "Survived") -> str:
    """Train a RandomForest classifier with sklearn to predict the target column.
    Returns accuracy, classification report, confusion matrix, and feature importance."""
    if target_column not in df.columns:
        return f"❌ 目标列 '{target_column}' 不存在。可用列: {list(df.columns)}"

    lines = []
    lines.append("=" * 60)
    lines.append("🤖 机器学习模型训练 — Random Forest")
    lines.append("=" * 60)

    data = df.copy()

    drop_cols = ["PassengerId", "Name", "Ticket", "Cabin"]
    keep_drop = [c for c in drop_cols if c in data.columns and c != target_column]
    X = data.drop(columns=[target_column] + keep_drop)
    y = data[target_column]

    lines.append(f"\n特征列 ({len(X.columns)}): {list(X.columns)}")
    lines.append(f"目标列: {target_column}")

    cat_cols = X.select_dtypes(include=["object", "category"]).columns
    for col in cat_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        lines.append(f"  编码: {col} → {list(le.classes_)}")

    if X.isnull().sum().sum() > 0:
        X = X.fillna(X.median(numeric_only=True))
        for col in X.columns:
            if X[col].isnull().any():
                X[col] = X[col].fillna(X[col].mode()[0] if len(X[col].mode()) > 0 else 0)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    lines.append(f"\n训练集: {len(X_train)} | 测试集: {len(X_test)}")

    clf = RandomForestClassifier(
        n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    lines.append("\n" + "=" * 60)
    lines.append("📊 模型评估")
    lines.append("=" * 60)
    lines.append(f"\n准确率 (Accuracy): {acc:.4f}  ({acc * 100:.2f}%)")

    lines.append("\n分类报告 (Classification Report):")
    lines.append(classification_report(
        y_test, y_pred, target_names=["未存活(0)", "存活(1)"], zero_division=0
    ))

    cm = confusion_matrix(y_test, y_pred)
    lines.append(f"混淆矩阵 (Confusion Matrix):")
    lines.append(f"  TN={cm[0][0]:3d}  FP={cm[0][1]:3d}")
    lines.append(f"  FN={cm[1][0]:3d}  TP={cm[1][1]:3d}")

    lines.append("\n" + "=" * 60)
    lines.append("⭐ 特征重要性 Top-10")
    lines.append("=" * 60)
    imp = pd.DataFrame({
        "feature": X.columns,
        "importance": clf.feature_importances_,
    }).sort_values("importance", ascending=False)
    for _, row in imp.head(10).iterrows():
        lines.append(f"  {row['feature']:20s}: {row['importance']:.4f}")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════
#  TOOL DEFINITIONS  (OpenAI function-calling format)
# ═══════════════════════════════════════════════════════════════════════

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "data_summary",
            "description": "获取 Titanic 数据集的完整统计摘要。返回均值、方差、标准差、最大值、最小值、四分位数、缺失值统计、Survived 分布。当用户询问数据摘要、数据统计、数据概览时调用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "dummy": {"type": "string", "description": "占位参数，传空字符串即可"}
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "create_plot",
            "description": "为指定列创建统计图表并保存为PNG文件。当用户要画图、绘制分布图、可视化数据时调用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "column_name": {
                        "type": "string",
                        "description": "要绘图的列名，例如 Survived, Age, Pclass, Fare, Sex, Embarked"
                    },
                    "plot_type": {
                        "type": "string",
                        "enum": ["count", "hist", "box", "bar"],
                        "description": "图表类型: count=分布图, hist=直方图, box=箱线图, bar=柱状图"
                    }
                },
                "required": ["column_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "train_prediction_model",
            "description": "使用 sklearn RandomForest 训练分类模型预测目标列。返回准确率、分类报告、混淆矩阵、特征重要性。当用户要训练模型、预测Survived、做分类任务时调用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "target_column": {
                        "type": "string",
                        "description": "要预测的目标列名，默认 Survived"
                    }
                }
            }
        }
    }
]

TOOL_MAP = {
    "data_summary": tool_data_summary,
    "create_plot": tool_create_plot,
    "train_prediction_model": tool_train_model,
}


# ═══════════════════════════════════════════════════════════════════════
#  AGENT  (custom loop using OpenAI client directly)
# ═══════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are a professional data analysis assistant working with a Titanic passenger dataset (891 rows, 12 columns: PassengerId, Survived, Pclass, Name, Sex, Age, SibSp, Parch, Ticket, Fare, Cabin, Embarked).

You have three tools:
  - data_summary: comprehensive statistics (mean, variance, max, min, distribution)
  - create_plot: draw and save a chart for a given column
  - train_prediction_model: train sklearn RandomForest to predict a target column

RULES:
1. User asks for 数据摘要 / 统计 / 概览 → call data_summary()
2. User asks for 画图 / 绘制 / 分布图 → call create_plot(column_name="Survived", plot_type="count"), default to "Survived" if no column specified
3. User asks for 训练模型 / 预测 / 分类 → call train_prediction_model(target_column="Survived")
4. ALWAYS respond in Chinese. The tool returns raw data — you MUST summarize the key findings in natural, helpful Chinese.
5. Only call ONE tool per response. If user asks multiple things, pick the most relevant one.
"""


class DataExplorationAgent:
    def __init__(self, client: OpenAI, model: str):
        self.client = client
        self.model = model

    def run(self, query: str, verbose: bool = True) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ]

        for _turn in range(5):
            kwargs = {
                "model": self.model,
                "messages": messages,
                "temperature": 0.1,
            }

            # First call: ask LLM with tools available, but use tool_choice="auto"
            # so it can also respond without tools when appropriate
            kwargs["tools"] = TOOLS
            kwargs["tool_choice"] = "auto"

            response = self.client.chat.completions.create(**kwargs)
            msg = response.choices[0].message

            # If no tool call, return the text response
            if not msg.tool_calls:
                return msg.content or "（模型未返回内容）"

            # Execute tool calls
            for tc in msg.tool_calls:
                tool_name = tc.function.name
                try:
                    tool_args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    tool_args = {}

                if verbose:
                    print(f"  🔧 Calling: {tool_name}({tool_args})")

                tool_fn = TOOL_MAP.get(tool_name)
                if tool_fn:
                    try:
                        result = tool_fn(**tool_args)
                    except Exception as e:
                        result = f"工具执行错误: {e}"
                else:
                    result = f"未知工具: {tool_name}"

                # Append assistant message (with tool_calls) and tool result
                messages.append({
                    "role": "assistant",
                    "content": msg.content or "",
                    "tool_calls": [{
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "arguments": tc.function.arguments,
                        }
                    }]
                })
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })

        return "已达到最大对话轮次（5轮），请重新提问。"


# ═══════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
agent = DataExplorationAgent(client, MODEL)


def run_query(query: str) -> str:
    return agent.run(query)


def run_demo():
    """Run the three required demo tasks."""
    tasks = [
        "请帮我做一个数据摘要，统计数据的均值、方差、最大值、最小值等",
        "请画出Survived列的分布图",
        "请用sklearn训练一个模型来预测Survived列，给出评估结果",
    ]
    for i, task in enumerate(tasks, 1):
        print(f"\n{'=' * 60}")
        print(f"  Task {i}: {task}")
        print("=" * 60)
        try:
            result = run_query(task)
            print(f"\n{result}")
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"\n❌ Task {i} 失败: {e}")


def interactive():
    """Interactive conversation loop."""
    print("\n" + "=" * 60)
    print("  🚢 Titanic 数据探索 Agent")
    print("  可以问我: 数据统计 / 画图 / 训练模型")
    print("  输入 'quit' 或 'exit' 退出")
    print("=" * 60)

    while True:
        try:
            user_input = input("\n🧑 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("再见!")
            break

        try:
            result = run_query(user_input)
            print(f"\n🤖 {result}")
        except Exception as e:
            print(f"\n❌ 出错: {e}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        run_demo()
    else:
        interactive()