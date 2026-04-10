"""
Agent 核心 v2：集成记忆模块

相比 Day 4 的 graph.py，新增：
1. State 中加入 working_memory 字段
2. agent_node 在调用 LLM 前注入工作记忆和长期记忆上下文
3. 每轮推理后用短期记忆策略管理消息长度
4. 支持多轮对话（传入历史 messages）

运行方式：被 main_v2.py 导入
"""

from typing import Annotated
from typing_extensions import TypedDict

from langchain_core.messages import AnyMessage, AIMessage, ToolMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from config import get_llm
from tools import ALL_TOOLS
from prompts import SHOPPING_AGENT_PROMPT
from memory import ShortTermMemory, LongTermMemory, WorkingMemory

# ============================================================
# Step 1: 定义 State（Agent 的记忆）
# ============================================================
# State 是 LangGraph 的核心概念：图中所有节点共享的数据结构
# 每个节点读取 State、处理、返回更新后的 State
class AgentState(TypedDict):
    """Agent 的状态
    messages: 完整的消息历史（用户消息 + AI消息 + 工具消息）
                Annotated[..., add_messages] 表示新消息会追加到列表末尾
    step_count: 当前已执行的步数（用于防无限循环）
    """
    messages: Annotated[list[AnyMessage], add_messages]
    step_count: int
    working_memory_context: str # 工作记忆的文本表示, 注入到 Prompt

# ============================================================
# Step 2: 定义节点（Node）
# ============================================================
# 节点就是函数，接收 State，返回更新后的 State

# 最大推理步数: 防止 Agent 无限循环调用工具
MAX_STEPS = 10

# 初始化 LLM 并绑定工具
llm = get_llm()
# bind_tools 告诉 LLM "你这些工具可以用"
# LLM 会在合适的时候输出 tool_calls
llm_with_tools = llm.bind_tools(ALL_TOOLS)

# 初始化记忆组件
short_term = ShortTermMemory()
long_term = LongTermMemory()
working_mem = WorkingMemory()


def agent_node(state: AgentState) -> dict:
    """Agent 节点：调用 LLM 进行推理(带记忆)
    这个节点做的事情：
        1. 把当前的消息历史发给 LLM
        2. LLM 返回一个 AIMessage（可能包含 tool_calls，也可能是最终回答）
        3. 把 AIMessage 追加到消息历史
    如果达到最大步数，强制 LLM 给出最终回答
    """
    messages = state["messages"]
    step_count = state.get("step_count", 0)

    # 安全检查: 如果步数超限，强制终止
    if step_count >= MAX_STEPS:
        return {
            "messages": [AIMessage(
                content="抱歉，我在处理您的请求时遇到了一些困难。让我直接基于已有信息为您回答。请问您还需要什么帮助？")],
            "step_count": step_count,
        }

    # --- 构建增强版 System Prompt ---
    enhanced_prompt = SHOPPING_AGENT_PROMPT

    wm_context = state.get("working_memory_context", "")
    if wm_context:
        enhanced_prompt += f"\n\n## 当前工作记忆\n{wm_context}"

    # 确保第一条消息是 System Prompt
    if messages and isinstance(messages[0], SystemMessage):
        messages[0] = SystemMessage(content=enhanced_prompt)
    else:
        messages = [SystemMessage(content=enhanced_prompt)] + messages

    # --- 短期记忆：控制消息长度 ---
    messages = short_term.summarize_and_trim(messages, max_pairs=8, summary_threshold=20)

    # 调用 LLM
    response = llm_with_tools.invoke(messages)

    return {
        "messages": [response],
        "step_count": step_count + 1,
    }

# 工具执行节点: 使用 LangGraph 预置的 ToolNode
# 它会自动执行 AIMessage 中的 tool_calls, 返回 ToolMessage
tool_executor = ToolNode(ALL_TOOLS)

def safe_tool_node(state: AgentState) -> dict:
    """带错误处理的工具执行节点
        如果工具执行出错，不会让整个 Agent 崩溃，
        而是返回一个错误信息给 LLM，让它决定下一步怎么办
    """
    from langchain_core.messages import ToolMessage
    try:
        return tool_executor.invoke(state)
    except Exception as e:
        # 获取最后一条 AI 消息中的 tool_calls
        last_ai_msg = state["messages"][-1]
        error_messages = []
        if hasattr(last_ai_msg, "tool_calls"):
            for tc in last_ai_msg.tool_calls:
                error_messages.append(
                    ToolMessage(
                        content=f"❌ 工具出错: {e}",
                        tool_call_id=tc["id"],
                    )
                )
        return {"messages": error_messages}

# ============================================================
# Step 3: 定义边（Edge）—— 决定下一步走哪个节点
# ============================================================

def should_continue(state: AgentState) -> str:
    """条件边：判断 Agent 应该继续调工具还是结束
    逻辑：
        - 如果 LLM 最后一条消息包含 tool_calls → 去 tools 节点
        - 否则 → 结束（LLM 给出了最终回答）
    """
    last_message = state["messages"][-1]

    # 检查是否有工具调用
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"
    else:
        return END

# ============================================================
# Step 4: 组装状态图
# ============================================================

def build_agent_graph():
    """构建 Agent 的状态图

    图结构：
        START
          ↓
        agent（LLM推理）
          ↓
        should_continue?
         /          \\
       tools        END
      (执行工具)   (输出答案)
         ↓
       agent（带着工具结果再次推理）
    """
    # 创建状态图，指定 State 类型
    graph = StateGraph(AgentState)

    # 添加节点
    graph.add_node("agent", agent_node)
    graph.add_node("tools", safe_tool_node)

    # 设置入口: 从 Agent 节点开始
    graph.set_entry_point("agent")

    # 添加条件边: agent 执行后, 根据结果决定下一步
    graph.add_conditional_edges(
        "agent",            # 从 agent 节点出发
        should_continue,    # 用这个函数判断
        {
            "tools": "tools", # 如果返回 "tools" → 去 tools 节点
            END: END,         # 如果返回 END → 结束
        }
    )

    # 工具执行完后, 回到 agent 节点（LLM 看到结果后继续推理）
    graph.add_edge("tools", "agent")

    # 编译图（变成可执行的 Runnable）
    return graph.compile()

# 创建全局 Agent 实例
agent = build_agent_graph()