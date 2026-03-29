"""
Day1 - 第一个 ReAct Agent
目标: 理解 Agent = LLM + 工具调用 + 循环推理

这个 Agent 有两个工具:
1. calculate - 数学计算
2. get_product_info - 模拟查询商品信息（为导购场景预热）

运行方式: uv run python day1-basic-agent/01_first_agent.py
"""

import os
from dotenv import load_dotenv

# ========================================
# Step 1: 加载环境变量
# ========================================
load_dotenv("day1-basic-agent/.env")

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

# ========================================
# Step 2: 初始化 LLM
# ========================================
# temperature=0 让输出更稳定, 方便调试
llm = ChatOpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    model="qwen-plus",
    temperature=0,
)

# ========================================
# Step 3: 定义工具(Tool)
# 这是 Agent 的手和脚--LLM 只能思考, 工具让他能行动
# ========================================

@tool
def calculate(expression: str) -> str:
    """ 计算数学表达式, 当用户需要做数学运算时使用此工具
    Args:
        expression: 数学表达式, 如 ‘100 * 0.85’ 或 '(1299 - 999) / 1299'
    """
    try:
        # 生产环境不要使用 eval
        result = eval(expression)
        return f"计算结果：{expression} = {result}"
    except Exception as e:
        return f"计算错误：{e}"

@tool
def get_product_info(product_name: str) -> str:
    """ 查询商品信息, 当用户询问某个商品的价格、库存或者详情时使用此工具

    Args:
         product_name: 商品名称, 如 '耐克跑鞋' 或 'iphone 16'
    """
    # 模拟商品数据库
    products = {
        "耐克跑鞋": {
            "name": "Nike Air Zoom Pegasus 41",
            "price": 899,
            "original_price": 1099,
            "stock": 156,
            "rating": 4.7,
            "features": ["透气网面", "Zoom Air 气垫", "适合日常跑步训练"],
        },
        "iPhone 16": {
            "name": "Apple iPhone 16 128GB",
            "price": 5999,
            "original_price": 5999,
            "stock": 2340,
            "rating": 4.8,
            "features": ["A18 芯片", "4800万像素", "Action Button"],
        },
        "蓝牙耳机": {
            "name": "Sony WF-1000XM5",
            "price": 1699,
            "original_price": 1999,
            "stock": 89,
            "rating": 4.6,
            "features": ["主动降噪", "LDAC 高清编码", "8小时续航"],
        },
    }

    # 模糊匹配: 遍历所有商品, 看用户输入是否包含关键词
    for key, info in products.items():
        if key in product_name or product_name in key:
            discount = round(1 - info["price"] / info["original_price"]) * 100
            return (
                f"商品: {info['name']}\n"
                f"现价: ￥{info['price']} (原价 ￥{info['original_price']})"
                f"折扣: {discount}% \n"
                f"库存: {info['stock']} 件\n"
                f"评分: {info['rating']}/5.0\n"
                f"卖点: {'、'.join(info['features'])}"
            )
    return f"未找到商品 '{product_name}', 请尝试搜索: 耐克跑鞋、iPhone 16、蓝牙耳机"

# ========================================
# Step 4: 创建 Agent
# create_react_agent 是 LangGraph 提供的预置 ReAct Agent
# 它自动处理了 Thought → Action → Observation 的循环
# ========================================

# 定义 Agent 的 System Prompt -- 告诉它"你是谁"
system_prompt = """你是一个电商导购助手, 你的职责是:
1. 帮用户查询商品信息
2. 帮用户做价格计算（折扣、对比等）
3. 根据用户需求推荐商品

规则:
- 先理解用户的真实需求，再决定是否需要使用工具
- 如果需要查商品信息，使用 get_product_info 工具
- 如果需要做计算，使用 calculate 工具
- 回答要简洁、有用，像一个专业的导购员
"""

# 把 LLM、工具、System Prompt 组装成 Prompt
agent = create_react_agent(
    model=llm,
    tools=[calculate, get_product_info],
    prompt=system_prompt,
)

# ========================================
# Step 5: 运行 Agent, 观察它的推理过程
# ========================================

def chat(user_input: str):
    """与 Agent 对话，并打印详细的推理过程"""
    print(f"\n{'='*60}")
    print(f"用户: {user_input}")
    print(f"{'='*60}")

    # 调用 Agent
    result = agent.invoke(
        {"messages": [{"role": "user", "content": user_input}]}
    )

    # 打印 Agent 的每一步推理过程
    for msg in result["messages"]:
        msg_type = msg.__class__.__name__

        if msg_type == "HumanMessage":
            # 用户消息，跳过（已经打印过了）
            continue
        elif msg_type == "AIMessage":
            # LLM 的思考和决策
            if msg.tool_calls:
                # LLM 决定调用工具
                for tc in msg.tool_calls:
                    print(f"\n Thought → 我需要调用工具")
                    print(f"\n Action: {tc['name']}({tc['args']})")
            else:
                # LLM 给出最终答案
                print(f"\n Final Answer: {msg.content}")
        elif msg_type == "toolMessage":
            # 工具返回的结果
            print(f"\n Observation ({msg.name}):")
            print(f"    {msg.content}")
    print(f"\n{'=' * 60}\n")

# ========================================
# Step 6: 测试！跑几个典型场景
# ========================================
if __name__ == "__main__":
    # 测试1: 简单商品查询（需要调用 1 个工具）
    chat("帮我看看耐克跑鞋多少钱")

    # 测试2: 需要多步推理（查询 + 计算）
    chat("蓝牙耳机现在打几折？帮我算算比原价便宜了多少钱")

    # 测试3: 不需要工具的简单问题
    chat("你好，你能帮我做什么？")







