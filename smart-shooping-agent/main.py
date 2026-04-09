"""
Day 4 - Smart Shopping Agent 入口
功能：
1. 单轮测试模式：跑预设的测试场景，观察 Agent 推理过程
2. 交互对话模式：手动输入问题，实时对话
运行方式：
  测试模式：uv run python smart-shopping-agent/main.py
  对话模式：uv run python smart-shopping-agent/main.py --chat
"""

import sys
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from graph import agent

# ============================================================
# 详细日志打印（观察 Agent 每一步的推理过程）
# ============================================================

def run_with_logging(user_input: str):
    """执行一次查询, 打印完整的推理链"""
    print(f"\n{'=' * 65}")
    print(f"🧑 用户: {user_input}")
    print(f"{'=' * 65}")

    # 调用 Agent
    result = agent.invoke({
        "messages": [HumanMessage(content=user_input)],
        "step_count": 0,
    })

    # 遍历消息历史, 打印每一步
    step = 0
    for msg in result["messages"]:
        if isinstance(msg, SystemMessage):
            continue    # 跳过 System Prompt
        elif isinstance(msg, HumanMessage):
            continue    # 已经打印过了
        elif isinstance(msg, AIMessage):
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    step += 1
                    print(f"\n  Step {step} 🤖 LLM 决策 → 调用工具")
                    print(f"    工具: {tc['name']}")
                    print(f"    参数: {tc['args']}")
            elif msg.content:
                print(f"\n  ✅ 最终回答:")
                # 缩进打印多行内容
                for line in msg.content.split("\n"):
                    print(f"    {line}")
        elif isinstance(msg, ToolMessage):
            print(f"  🔧 工具返回 ({msg.name}):")
            for line in msg.content.split("]n")[:8]:    # 最多打印8行
                print(f"    {line}")
            if len(msg.content.split("\n")) > 8:
                print(f"    ... (省略)")
    total_steps = result.get("step_count", 0)
    print(f"\n  📊 共执行 {total_steps} 轮推理，调用 {step} 次工具")
    print(f"{'=' * 65}\n")

    return result

# ============================================================
# 测试场景
# ============================================================
def run_test_scenarios():
    """运行预设的测试场景"""
    print("🚀 Smart Shopping Agent - Day 4 测试\n")

    scenarios = [
        {
            "name": "场景 1: 简单查询（1步）",
            "input": "华为耳机多少钱？",
            "expected_tools": ["search_products 或 get_product_detail"],
            "expected_steps": "1-2步",
        },
        {
            "name": "场景 2: 带条件搜索（2步）",
            "input": "有什么1000块以内的运动鞋？",
            "expected_tools": ["search_products → get_product_detail"],
            "expected_steps": "2步",
        },
        {
            "name": "场景 3: 对比 + 计算（3-4步）",
            "input": "帮我对比一下索尼和华为的降噪耳机，哪个更划算？便宜多少钱？",
            "expected_tools": ["search_products → compare_products → calculate_price"],
            "expected_steps": "3-4步",
        },
        {
            "name": "场景 4: 完整导购流程（4-5步）",
            "input": "我想买一个跑步用的鞋，预算1000以内，最好是知名品牌的，帮我推荐一下",
            "expected_tools": ["search_products → get_product_detail → get_recommendation_reason"],
            "expected_steps": "3-5步",
        },
        {
            "name": "场景 5: 模糊需求（测试追问能力）",
            "input": "帮我挑个礼物",
            "expected_tools": ["可能不调工具，直接追问"],
            "expected_steps": "0步（追问）",
        },
        {
            "name": "场景 6: 超出范围（测试拒绝能力）",
            "input": "帮我写一首诗",
            "expected_tools": ["不应调用任何工具"],
            "expected_steps": "0步（拒绝）",
        },
    ]

    for scenario in scenarios:
        print(f"\n>>> {scenario['name']}")
        print(f"    预期工具链: {scenario['expected_tools']}")
        print(f"    预期步数: {scenario['expected_steps']}")
        run_with_logging(scenario["input"])

# ============================================================
# 交互对话模式
# ============================================================

def run_chat_mode():
    """交互式对话（单轮，每次独立）"""
    print("💬 Smart Shopping Agent - 交互模式")
    print("   输入购物相关的问题，输入 'quit' 退出\n")

    while True:
        user_input = input("🧑 你: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("👋 再见！")
            break
        run_with_logging(user_input)


# ============================================================
# 程序入口
# ============================================================

if __name__ == "__main__":
    if "--chat" in sys.argv:
        run_chat_mode()
    else:
        run_test_scenarios()