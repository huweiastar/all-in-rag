"""
第三节：文本到SQL（Text-to-SQL）

本文件演示了 Text2SQL 的完整流程：用户用自然语言提问 → LLM 翻译成 SQL 语句 → 执行查询 → 返回结果。

核心思路：
  用户输入自然语言问题（如"年龄大于30的用户有哪些"）
    → Text2SQLAgent 从知识库检索相关的表结构和示例
    → LLM 根据表结构信息将问题翻译为 SQL（如 SELECT name FROM users WHERE age > 30）
    → 在 SQLite 数据库中执行该 SQL
    → 如果执行失败，LLM 还能根据错误信息自动修复 SQL 并重试

为什么需要这个技术？
  非技术人员不懂 SQL 语法，但能用自然语言描述自己想要什么数据。
  Text2SQL 让这些人也能直接查询数据库，不需要依赖开发人员写 SQL。

三大业务挑战及应对策略：
  1. "幻觉"问题 → LLM 可能捏造不存在的表/字段。应对：提供精确的 DDL（表结构定义）作为上下文。
  2. 结构理解不足 → LLM 需要理解表关联关系才能正确生成 JOIN。应对：提供外键关系和字段描述。
  3. 用户输入模糊 → 问题可能不规范或有歧义。应对：利用 RAG 增强上下文，提供相似问答示例。

优化策略（在知识库中积累三类知识）：
  - DDL 知识：表的 CREATE TABLE 语句，给 LLM 一张"数据库地图"
  - Q-SQL 示例：历史问答对，让 LLM 照猫画虎
  - 描述知识：表和字段的业务含义，帮助 LLM 理解语义

框架三个核心模块的协作流程：
  +------------------+     +-------------------+     +------------------+
  |  knowledge_base  |---->|  sql_generator    |---->|  text2sql_agent  |
  |  （知识库）       |     |  （SQL生成器）     |     |  （代理/控制器）  |
  +------------------+     +-------------------+     +------------------+
  存储表结构、示例、描述    将自然语言翻译为 SQL       协调整个流程，执行 SQL
  通过语义检索提供上下文    具备错误修复能力           带重试和安全限制
"""

import os
import sys
import sqlite3
from dotenv import load_dotenv

# 加载项目根目录的 .env 文件，使 DASHSCOPE_API_KEY 等变量生效
load_dotenv()

# 添加text2sql模块路径
# 【小白理解】：Python 默认只从标准路径找包。text2sql 是当前目录下的子文件夹，
# 不是 pip 安装的包，所以需要手动把它的路径加到 sys.path 中，Python 才能 import 到它。
sys.path.append(os.path.join(os.path.dirname(__file__), 'text2sql'))

from text2sql.text2sql_agent import SimpleText2SQLAgent


def setup_demo():
    """设置演示环境

    这是整个演示的"开机启动"流程，做了4件事：
    1. 检查 API 密钥是否配置
    2. 创建临时演示用 SQLite 数据库（内含用户表、产品表、订单表）
    3. 初始化 Text2SQL 代理（内部会创建 LLM 连接）
    4. 加载知识库（表结构 DDL、查询示例等，帮助 LLM 更准确地生成 SQL）
    """
    print("=== Text2SQL框架演示 ===\n")

    # 检查API密钥
    # 【小白理解】：LLM 需要调用 API 来翻译自然语言 → SQL，这里使用阿里云百炼的 API 密钥
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        print("请先设置DASHSCOPE_API_KEY环境变量")
        return None

    # 创建演示数据库
    print("创建演示数据库...")
    db_path = create_demo_database()

    # 初始化Text2SQL代理
    # 【小白理解】：Agent（代理）是整个 Text2SQL 流程的"大脑"，它负责：
    #   1. 接收用户问题
    #   2. 从知识库查找相关表信息
    #   3. 让 LLM 生成 SQL
    #   4. 执行 SQL
    #   5. 如果失败，自动修复 SQL 并重试
    print("初始化Text2SQL代理...")
    agent = SimpleText2SQLAgent(
        api_key=api_key,
        base_url=os.getenv("DASHSCOPE_BASE_URL"),
        model=os.getenv("MODEL", "qwen3.6-plus")
    )
    
    # 连接数据库
    print("连接数据库...")
    if not agent.connect_database(db_path):
        print("数据库连接失败!")
        return None
    
    # 加载知识库
    print("加载知识库...")
    try:
        agent.load_knowledge_base()
        print("知识库加载成功!")
    except Exception as e:
        print(f"知识库加载失败: {str(e)}")
        return None
    
    return agent, db_path


def create_demo_database():
    """创建演示数据库

    创建了一个模拟电商场景的 SQLite 数据库，包含3张表：
      users（用户表）    —— 存储用户基本信息
      products（产品表） —— 存储商品及库存信息
      orders（订单表）   —— 存储用户购买的记录，通过外键关联用户和产品

    【小白理解】：这3张表之间形成"用户 → 购买 → 产品"的关系链，
    演示问题中会涉及单表查询、条件过滤、JOIN关联查询、聚合统计等多种 SQL 操作。
    """
    db_path = "text2sql_demo.db"
    
    if os.path.exists(db_path):
        os.remove(db_path)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # 创建用户表
    # users 表：每个用户有唯一 id，email 设置了 UNIQUE 不能重复
    cursor.execute("""
        CREATE TABLE users (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            email TEXT UNIQUE,
            age INTEGER,
            city TEXT
        )
    """)

    # 创建产品表
    # products 表：category 是分类，price 是价格，stock 是库存数量
    cursor.execute("""
        CREATE TABLE products (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            category TEXT,
            price REAL,
            stock INTEGER
        )
    """)

    # 创建订单表
    # orders 表：通过外键（FOREIGN KEY）关联 users 和 products 表
    #   user_id     → 指向 users.id（谁买的）
    #   product_id  → 指向 products.id（买了什么）
    #   quantity    → 买了几个
    #   total_price → 这笔订单的总金额
    # 【小白理解】：外键就像"引用指针"，确保订单里的 user_id 一定是 users 表中真实存在的 id
    cursor.execute("""
        CREATE TABLE orders (
            id INTEGER PRIMARY KEY,
            user_id INTEGER,
            product_id INTEGER,
            quantity INTEGER,
            order_date TEXT,
            total_price REAL,
            FOREIGN KEY (user_id) REFERENCES users(id),
            FOREIGN KEY (product_id) REFERENCES products(id)
        )
    """)
    
    # 插入示例数据
    # 【注意】：每条数据的顺序和外键关系要对应好
    #   比如订单 (1, 1, 1, ...) 表示：用户id=1（张三）买了产品id=1（iPhone 15）1件，总价7999
    users_data = [
        (1, '张三', 'zhangsan@email.com', 25, '北京'),
        (2, '李四', 'lisi@email.com', 32, '上海'),
        (3, '王五', 'wangwu@email.com', 28, '广州'),
        (4, '赵六', 'zhaoliu@email.com', 35, '深圳'),
        (5, '陈七', 'chenqi@email.com', 29, '杭州'),
    ]
    
    products_data = [
        (1, 'iPhone 15', '电子产品', 7999.0, 50),
        (2, 'MacBook Pro', '电子产品', 12999.0, 20),
        (3, 'Nike运动鞋', '服装', 599.0, 100),
        (4, '办公椅', '家具', 899.0, 30),
        (5, '台灯', '家具', 199.0, 80),
        (6, 'iPad', '电子产品', 3999.0, 40),
        (7, 'Adidas外套', '服装', 399.0, 60),
    ]
    
    orders_data = [
        (1, 1, 1, 1, '2024-01-15', 7999.0),
        (2, 2, 3, 2, '2024-01-16', 1198.0),
        (3, 3, 5, 1, '2024-01-17', 199.0),
        (4, 1, 2, 1, '2024-01-18', 12999.0),
        (5, 4, 4, 1, '2024-01-19', 899.0),
        (6, 5, 6, 1, '2024-01-20', 3999.0),
        (7, 2, 7, 1, '2024-01-21', 399.0),
    ]
    
    cursor.executemany("INSERT INTO users VALUES (?, ?, ?, ?, ?)", users_data)
    cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?)", products_data)
    cursor.executemany("INSERT INTO orders VALUES (?, ?, ?, ?, ?, ?)", orders_data)
    
    conn.commit()
    conn.close()
    
    print(f"演示数据库已创建: {db_path}")
    return db_path


def run_demo_queries(agent):
    """运行演示查询

    准备了6个不同难度的自然语言问题，覆盖以下 SQL 知识点：
      1. "查询所有用户的姓名和邮箱"        → 基本 SELECT（单表查）
      2. "年龄大于30的用户有哪些"          → WHERE 条件过滤
      3. "哪些产品的库存少于50"            → 数值比较
      4. "查询来自北京的用户的所有订单"    → JOIN 多表关联 + WHERE
      5. "统计每个城市的用户数量"          → GROUP BY 分组聚合
      6. "查询价格在500-8000之间的产品"    → BETWEEN 范围查询

    每个问题的处理流程（agent.query 内部）：
      第1步：知识库检索 → 从 Milvus 向量库中找出最相关的表结构、示例、描述
      第2步：SQL 生成    → LLM 根据上下文将自然语言翻译为 SQL
      第3步：SQL 执行    → 在 SQLite 中执行，自动添加 LIMIT 100 防止返回过多数据
      第4步：错误修复    → 如果执行报错，LLM 根据错误信息修改 SQL 并重试（最多3次）
    """
    demo_questions = [
        "查询所有用户的姓名和邮箱",
        "年龄大于30的用户有哪些",
        "哪些产品的库存少于50",
        "查询来自北京的用户的所有订单",
        "统计每个城市的用户数量",
        "查询价格在500-8000之间的产品"
    ]

    print("\n开始运行演示查询...\n")

    success_count = 0  # 统计成功执行的查询数量

    for i, question in enumerate(demo_questions, 1):
        print(f"问题 {i}: {question}")
        print("-" * 60)

        try:
            # 【小白理解】：agent.query 是整个 Text2SQL 的核心入口
            # 它内部会经历"检索 → 生成 → 执行 → (失败时)修复"的完整链路
            result = agent.query(question)

            if result["success"]:
                print(f"成功! SQL: {result['sql']}")

                if isinstance(result["results"], dict) and "rows" in result["results"]:
                    count = result["results"]["count"]
                    print(f"返回 {count} 行数据")

                    # 显示前2行数据
                    # 【小白理解】：只展示前2行是为了控制台输出简洁，
                    # 实际数据可能有更多（被 LIMIT 100 限制），用"...还有 N 行"提示
                    if count > 0:
                        for j, row in enumerate(result["results"]["rows"][:2]):
                            row_str = " | ".join(f"{k}: {v}" for k, v in row.items())
                            print(f"  {j+1}. {row_str}")

                        if count > 2:
                            print(f"  ... 还有 {count - 2} 行")
                else:
                    print(f"结果: {result['results']}")

                success_count += 1

            else:
                # SQL 生成或执行失败（超过重试次数）
                print(f"失败: {result['error']}")
                print(f"SQL: {result['sql']}")

        except Exception as e:
            print(f"执行错误: {str(e)}")

        print()

    # 输出统计
    total_count = len(demo_questions)
    # （此处可扩展：print(f"成功率: {success_count}/{total_count}")）


def cleanup(agent, db_path):
    """清理资源

    演示结束后做两件事：
    1. 关闭数据库连接、清理知识库（agent.cleanup）
    2. 删除临时创建的 text2sql_demo.db 文件，不留垃圾
    """
    print("\n清理资源...")
    
    if agent:
        agent.cleanup()
    
    if os.path.exists(db_path):
        os.remove(db_path)
        print(f"已删除演示数据库: {db_path}")


def main():
    """主函数

    程序入口，执行顺序：
    1. setup_demo()      → 创建数据库、初始化代理、加载知识库
    2. run_demo_queries() → 逐个执行预设的自然语言问题
    3. cleanup()          → 清理资源（无论成功还是失败都会执行）

    【小白理解】：用 try/finally 确保即使查询过程中出错，也能正常关闭数据库连接和删除临时文件。
    """
    # 设置演示环境
    setup_result = setup_demo()
    
    if setup_result is None:
        return
    
    agent, db_path = setup_result
    
    try:
        # 运行演示查询
        run_demo_queries(agent)
        
    finally:
        # 清理资源
        cleanup(agent, db_path)


if __name__ == "__main__":
    main() 