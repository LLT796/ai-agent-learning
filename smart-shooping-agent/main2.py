"""
Day 5 - 记忆模块测试

测试内容：
1. 无记忆 vs 有记忆的多轮对话对比
2. 短期记忆三种策略效果
3. 工作记忆注入效果
4. 长期记忆的跨会话召回

运行方式：uv run python smart-shopping-agent/main_v2.py
"""

import sys
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from graph2 import agent, working_mem, long_term, short_term
from memory import ShortTermMemory


# ============================================================
# 日志打印
# ============================================================

def print_result(result):
    """打印 Agent 的推理链"""
    step = 0
    for msg in result["messages"]:
        if isinstance(msg, SystemMessage) or isinstance(msg, HumanMessage):
            continue
        elif isinstance(msg, AIMessage):
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    step += 1
                    print(f"    Step {step} 🤖 → {tc['name']}({tc['args']})")
            elif msg.content:
                print(f"    ✅ 回答: {msg.content[:200]}")
                if len(msg.content) > 200:
                    print(f"       ...({len(msg.content)}字)")
        elif isinstance(msg, ToolMessage):
            preview = msg.content[:80].replace("\n", " ")
            print(f"    🔧 {msg.name}: {preview}...")


# ============================================================
# 实验 1: 无记忆 vs 有记忆的多轮对话
# ============================================================

def experiment_memory_comparison():
    print("=" * 65)
    print("📊 实验 1: 无记忆 vs 有记忆 —— 多轮对话对比")
    print("=" * 65)

    # 模拟一段 5 轮对话
    conversation = [
        "我想买个降噪耳机",
        "预算 1500 以内的有哪些？",
        "帮我看看索尼那款的详情",
        "华为那款呢？",
        "我刚才看的那两款耳机，帮我对比一下哪个更值",  # 依赖前面的上下文
    ]

    # --- 无记忆：每轮都是独立的 ---
    print("\n🚫 无记忆模式（每轮独立，不传历史）")
    print("-" * 50)
    for i, user_input in enumerate(conversation):
        print(f"\n  第 {i+1} 轮 🧑: {user_input}")
        result = agent.invoke({
            "messages": [HumanMessage(content=user_input)],  # 每次只传当前消息
            "step_count": 0,
            "working_memory_context": "",
        })
        print_result(result)

    # --- 有记忆：累积传递历史 ---
    print(f"\n{'='*50}")
    print("✅ 有记忆模式（累积历史消息）")
    print("-" * 50)
    history = []
    for i, user_input in enumerate(conversation):
        print(f"\n  第 {i+1} 轮 🧑: {user_input}")
        history.append(HumanMessage(content=user_input))

        result = agent.invoke({
            "messages": history,
            "step_count": 0,
            "working_memory_context": "",
        })

        # 把 Agent 的响应加入历史
        for msg in result["messages"]:
            if msg not in history:
                history.append(msg)

        print_result(result)

    print(f"\n💡 关键对比：第5轮'那两款耳机'")
    print(f"   无记忆：Agent 不知道'那两款'是什么，只能瞎猜或追问")
    print(f"   有记忆：Agent 从历史中知道是索尼 XM5 和华为 FreeBuds Pro 3")
    print(f"   消息历史长度：{len(history)} 条")


# ============================================================
# 实验 2: 短期记忆三种策略对比
# ============================================================

def experiment_short_term_strategies():
    print(f"\n{'='*65}")
    print("📊 实验 2: 短期记忆三种策略对比")
    print("=" * 65)

    # 构造一段长对话（20条消息）
    long_conversation = []
    pairs = [
        ("帮我找运动鞋", "好的，为您搜索运动鞋..."),
        ("要耐克的", "找到了 Nike Pegasus 41，¥899"),
        ("还有别的吗", "还有 Adidas Ultraboost，¥1199"),
        ("耐克那个评分多少", "Nike Pegasus 41 评分 4.7/5.0"),
        ("阿迪那个呢", "Adidas Ultraboost 评分 4.5/5.0"),
        ("哪个更轻", "Nike 280g，Adidas 275g，Adidas更轻"),
        ("价格差多少", "差¥300，Nike更便宜"),
        ("有什么优惠吗", "Nike优惠200元，Adidas优惠300元"),
        ("我再看看耳机", "好的，为您搜索耳机..."),
        ("有降噪的吗", "有Sony XM5和华为FreeBuds Pro 3"),
    ]
    for user_msg, ai_msg in pairs:
        long_conversation.append(HumanMessage(content=user_msg))
        long_conversation.append(AIMessage(content=ai_msg))

    print(f"\n  原始对话: {len(long_conversation)} 条消息")

    # 策略 1: 滑动窗口
    result1 = ShortTermMemory.sliding_window(long_conversation, max_pairs=3)
    print(f"\n  📌 滑动窗口 (保留最近3轮): {len(result1)} 条消息")
    for msg in result1:
        role = "🧑" if isinstance(msg, HumanMessage) else "🤖"
        print(f"    {role} {msg.content[:60]}")

    # 策略 2: Token 截断
    result2 = ShortTermMemory.token_trim(long_conversation, max_tokens=500)
    print(f"\n  📌 Token截断 (500字符): {len(result2)} 条消息")
    for msg in result2:
        role = "🧑" if isinstance(msg, HumanMessage) else "🤖"
        print(f"    {role} {msg.content[:60]}")

    # 策略 3: 摘要压缩
    sys_msg = SystemMessage(content="你是导购助手")
    result3 = ShortTermMemory.summarize_and_trim(
        [sys_msg] + long_conversation, max_pairs=3, summary_threshold=8
    )
    print(f"\n  📌 摘要压缩 (保留最近3轮+摘要): {len(result3)} 条消息")
    for msg in result3:
        if isinstance(msg, SystemMessage) and "摘要" in msg.content:
            print(f"    📝 {msg.content[:120]}...")
        elif isinstance(msg, SystemMessage):
            print(f"    ⚙️ [System Prompt]")
        else:
            role = "🧑" if isinstance(msg, HumanMessage) else "🤖"
            print(f"    {role} {msg.content[:60]}")

    print(f"\n💡 策略对比:")
    print(f"   滑动窗口: 最简单，但丢失所有早期信息")
    print(f"   Token截断: 精确控制长度，但可能切断完整对话")
    print(f"   摘要压缩: 保留早期信息的摘要，最佳但多一次 LLM 调用")


# ============================================================
# 实验 3: 工作记忆效果
# ============================================================

def experiment_working_memory():
    print(f"\n{'='*65}")
    print("📊 实验 3: 工作记忆（Scratchpad）效果")
    print("=" * 65)

    # 模拟用户提供了偏好信息
    working_mem.clear()
    working_mem.set_preference("预算", "1500元以内")
    working_mem.set_preference("用途", "日常通勤跑步")
    working_mem.set_preference("品牌偏好", "国际品牌")
    working_mem.add_product("SKU001")
    working_mem.add_product("SKU003")
    working_mem.add_note("已对比", "Nike和Sony两款已对比过")

    wm_text = working_mem.to_context_string()
    print(f"\n  工作记忆内容:")
    for line in wm_text.split("\n"):
        print(f"    {line}")

    # 无工作记忆
    print(f"\n  🚫 无工作记忆 → 问'还有什么推荐'")
    result1 = agent.invoke({
        "messages": [HumanMessage(content="根据我的需求还有什么推荐吗？")],
        "step_count": 0,
        "working_memory_context": "",
    })
    print_result(result1)

    # 有工作记忆
    print(f"\n  ✅ 有工作记忆 → 同样问'还有什么推荐'")
    result2 = agent.invoke({
        "messages": [HumanMessage(content="根据我的需求还有什么推荐吗？")],
        "step_count": 0,
        "working_memory_context": wm_text,
    })
    print_result(result2)

    print(f"\n💡 对比:")
    print(f"   无工作记忆: Agent 不知道用户偏好，只能追问或泛泛推荐")
    print(f"   有工作记忆: Agent 知道预算1500、用途跑步、偏好国际品牌，精准推荐")

    working_mem.clear()


# ============================================================
# 实验 4: 长期记忆跨会话召回
# ============================================================

def experiment_long_term_memory():
    print(f"\n{'='*65}")
    print("📊 实验 4: 长期记忆 —— 跨会话召回")
    print("=" * 65)

    # 模拟保存两个历史会话
    print("\n  💾 保存历史会话到长期记忆...")

    # 会话 1: 咨询过跑鞋
    session1_msgs = [
        HumanMessage(content="我想买跑鞋，预算1000以内"),
        AIMessage(content="为您找到 Nike Pegasus 41，¥899，评分4.7，推荐日常慢跑训练。"),
    ]
    long_term.save_conversation("session_001", session1_msgs)
    print("    保存会话1: 咨询跑鞋，推荐了 Nike Pegasus")

    # 会话 2: 咨询过降噪耳机
    session2_msgs = [
        HumanMessage(content="有什么好的降噪耳机？"),
        AIMessage(content="推荐 Sony XM5，¥1699，降噪35dB，续航8小时。也可以看看华为 FreeBuds Pro 3，¥1099，降噪47dB。"),
    ]
    long_term.save_conversation("session_002", session2_msgs)
    print("    保存会话2: 咨询降噪耳机，推荐了 Sony 和 华为")

    # 新会话：用户再次来咨询
    print(f"\n  🔍 新会话开始，用户问：'上次看的那个鞋子还在卖吗'")
    query = "上次看的那个鞋子还在卖吗"
    memories = long_term.recall(query, k=2)

    print(f"  📚 召回的历史记忆:")
    for i, mem in enumerate(memories):
        print(f"    [{i+1}] {mem[:100]}...")

    print(f"\n💡 长期记忆让 Agent 知道用户'上次'看的是什么")
    print(f"   可以把召回的记忆注入到 System Prompt 或工作记忆中")


# ============================================================
# 主流程
# ============================================================

if __name__ == "__main__":
    print("🧠 Day 5 - 记忆模块测试\n")

    experiment_memory_comparison()
    experiment_short_term_strategies()
    experiment_working_memory()
    experiment_long_term_memory()

    print(f"\n{'='*65}")
    print("📝 三种记忆总结")
    print("=" * 65)
    print("""
    ┌──────────┬──────────────────┬───────────────┬──────────────┐
    │ 记忆类型 │ 作用             │ 存储位置      │ 生命周期     │
    ├──────────┼──────────────────┼───────────────┼──────────────┤
    │ 短期记忆 │ 管理当前对话长度 │ messages 列表 │ 当前会话     │
    │ 长期记忆 │ 跨会话记住用户   │ ChromaDB      │ 永久         │
    │ 工作记忆 │ 推理中的草稿纸   │ State 字段    │ 单次推理任务 │
    └──────────┴──────────────────┴───────────────┴──────────────┘
    """)