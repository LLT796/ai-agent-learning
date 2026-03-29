"""
Day 1 - 练习：添加自定义工具 + 多步推理测试（通义千问版）
目标：自己定义工具，给 Agent 更多能力，观察它如何在多工具间做选择

在 01_first_agent.py 的基础上，新增：
1. compare_products 工具 —— 对比两个商品
2. search_products 工具 —— 按条件搜索
3. 更复杂的导购场景测试

运行方式：uv run python day1-basic-agent/02_custom_tools.py
"""
import os
from dotenv import load_dotenv

load_dotenv("day1-basic-agent/.env")

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

# ============================================================
# 初始化通义千问
# ============================================================
llm = ChatOpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    model="qwen-plus",
    temperature=0,
)

# ============================================================
# 模拟商品数据库
# ============================================================
PRODUCT_DB = {
    "耐克跑鞋": {
        "name": "Nike Air Zoom Pegasus 41",
        "price": 899,
        "original_price": 1099,
        "category": "运动鞋",
        "stock": 156,
        "rating": 4.7,
        "features": ["透气网面", "Zoom Air 气垫", "适合日常跑步训练"],
    },
    "阿迪跑鞋": {
        "name": "Adidas Ultraboost Light",
        "price": 1199,
        "original_price": 1499,
        "category": "运动鞋",
        "stock": 78,
        "rating": 4.5,
        "features": ["Boost 中底", "Primeknit 鞋面", "适合长距离跑步"],
    },
    "蓝牙耳机": {
        "name": "Sony WF-1000XM5",
        "price": 1699,
        "original_price": 1999,
        "category": "耳机",
        "stock": 89,
        "rating": 4.6,
        "features": ["主动降噪", "LDAC 高清编码", "8小时续航"],
    },
    "华为耳机": {
        "name": "HUAWEI FreeBuds Pro 3",
        "price": 1099,
        "original_price": 1299,
        "category": "耳机",
        "stock": 234,
        "rating": 4.5,
        "features": ["智能降噪", "HWA 高清编码", "6.5小时续航"],
    },
}

def _find_product(name: str) -> dict | None:
    """内部函数：模糊匹配商品"""
    for key, info in PRODUCT_DB.items():
        if key in name or name in key:
            return {**info, "key": key}
    return None

# ============================================================
# 工具 1: 商品查询
# ============================================================

@tool
def get_product_info(product_name: str) -> str:
    """查询单个商品的详细信息。
    :param product_name: 商品名称关键词，如'耐克跑鞋'、'蓝牙耳机'
    """
    product = _find_product(product_name)
    if not product:
        available = "、".join(PRODUCT_DB.keys())
        return f"未找到 '{product_name}'。当前可查询的商品有：{available}"

    discount = round(1 - product["price"] / product["original_price"]) * 100
    return (
        f"商品：{product['name']}\n"
        f"分类：{product['category']}\n"
        f"现价：¥{product['price']}（原价 ¥{product['original_price']}，优惠 {discount}%）\n"
        f"库存：{product['stock']} 件\n"
        f"评分：{product['rating']}/5.0\n"
        f"卖点：{'、'.join(product['features'])}"
    )

# ============================================================
# 工具 2: 计算器
# ============================================================
@tool
def calculate(expression: str) -> str:
    """计算数学表达式。用于价格计算、折扣计算、对比差价等。
    Args:
        expression: 数学表达式，如 '1699 - 1099' 或 '899 / 1099 * 100'
    """
    try:
        result = eval(expression)
        if isinstance(result, float):
            result = round(result, 2)
        return f"{expression} = {result}"
    except Exception as e:
        return f"计算错误：{e}"

# ============================================================
# 工具 3: 商品对比
# ============================================================
@tool
def compare_products(product_a: str, product_b: str) -> str:
    """对比两个商品的详细参数。当用户想要比较两个商品时使用。
    Args:
        product_a: 第一个商品名称
        product_b: 第二个商品名称
    """
    a = _find_product(product_a)
    b = _find_product(product_b)

    if not a:
        return f"未找到商品 '{product_a}'"
    if not b:
        return f"未找到商品 '{product_b}'"

    price_diff = abs(a["price"] - b["price"])
    cheaper = a["name"] if a["price"] < b["price"] else b["name"]
    higher_rated = a["name"] if a["rating"] > b["rating"] else b["name"]
    return (
        f"=== 商品对比 ===\n"
        f"商品A：{a['name']} | 现价 ¥{a['price']} | 评分 {a['rating']}/5.0\n"
        f"商品B：{b['name']} | 现价 ¥{b['price']} | 评分 {b['rating']}/5.0\n"
        f"价格差：¥{price_diff}（{cheaper} 更便宜）\n"
        f"评分更高：{higher_rated}\n"
        f"A 卖点：{'、'.join(a['features'])}\n"
        f"B 卖点：{'、'.join(b['features'])}"
    )

# ============================================================
# 工具 4: 按条件搜索商品
# ============================================================
@tool
def search_products(
        category: str = "",
        max_price: float = 0,
        min_rating: float = 0.
) -> str:
    """按条件搜索商品。可按分类、价格上限、最低评分筛选。
        Args:
            category: 商品分类，如 '运动鞋'、'耳机'。留空则搜索全部
            max_price: 价格上限（元）。0 表示不限
            min_rating: 最低评分。0 表示不限
        """
    results = []
    for key, info in PRODUCT_DB.items():
        if category and info["category"] != category:
            continue
        if max_price > 0 and info["price"] > max_price:
            continue
        if min_rating > 0 and info["rating"] < min_rating:
            continue
        results.append(info)

    if not results:
        return f"没有找到符合条件的商品（分类={category}, 最高价={max_price}, 最低评分={min_rating}"

    output = f"找到 {len(results)} 个商品：\n"
    for i, p in enumerate(results, 1):
        output += f" {i}. {p['name']} - ￥{p['price']} - 评分{p['rating']}\n"
    return output

# ============================================================
# 组装 Agent（4 个工具）
# ============================================================
system_prompt = """你是淘天平台的智能导购助手。你的职责是帮用户找到最合适的商品。

工作流程：
1. 先理解用户需求（想买什么、预算多少、有什么偏好）
2. 使用 search_products 搜索符合条件的商品
3. 使用 get_product_info 获取具体商品详情
4. 如果用户在纠结两个商品，使用 compare_products 做对比
5. 如果需要计算折扣或差价，使用 calculate

注意：
- 一定要基于工具返回的真实数据回答，不要编造信息
- 推荐时要给出理由，不能只说"推荐这个"
- 如果用户的需求模糊，先追问细化需求
"""
agent = create_react_agent(
    model=llm,
    tools=[get_product_info, calculate, compare_products, search_products],
    prompt=system_prompt,
)

# ============================================================
# 详细日志版对话函数
# ============================================================
def chat(user_input: str):
    """与Agent 对话, 打印完整推理链"""
    print(f"\n{'=' * 60}")
    print(f"🧑 用户: {user_input}")
    print(f"{'=' * 60}")

    result = agent.invoke(
        {"messages": [{"role": "user", "content": user_input}]}
    )
    step = 0
    for msg in result["messages"]:
        msg_type = msg.__class__.__name__

        if msg_type == "HumanMessage":
            continue
        elif msg_type == "AIMessage":
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    step += 1
                    print(f"\n  Step {step} 🤖 Thought → Action:")
                    print(f"    工具: {tc['name']}")
                    print(f"    参数: {tc['args']}")
            elif msg.content:
                print(f"\n  ✅ Final Answer:")
                print(f"    {msg.content}")
        elif msg_type == "ToolMessage":
            print(f"Observation ({msg.name}):")
            for line in msg.content.split("\n"):
                print(f"{line}")
    print(f"\n{'=' * 60}\n")

# ============================================================
# 测试场景
# ============================================================\
if __name__ == "__main__":
    # 场景 1: 精确查询
    print("\n>>> 场景 1: 精确查询")
    chat("华为耳机多少钱？")

    # 场景 2: 多步推理 —— 搜索 → 查详情 → 计算
    print("\n>>> 场景 2: 多步推理")
    chat("有什么1000块以内的运动鞋？帮我算算最便宜的那个打了几折")

    # 场景 3: 商品对比
    print("\n>>> 场景 3: 商品对比")
    chat("耐克跑鞋和阿迪跑鞋哪个更值得买？")

    # 场景 4: 模糊需求
    print("\n>>> 场景 4: 模糊需求")
    chat("我想买个降噪耳机，预算 1500 以内，有推荐吗？")

    # 场景 5: 故意给一个模糊指令
    print("\n>>> 场景 5: 模糊指令")
    chat("帮我挑个好东西")