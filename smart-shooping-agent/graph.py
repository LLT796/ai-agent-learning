"""
Agent 核心：用 LangGraph StateGraph 手搓 ReAct Agent
Day 1 用的 create_react_agent 是一个封装好的黑盒。
今天我们拆开这个黑盒，自己用 StateGraph 实现同样的逻辑。
这样做的好处：
1. 完全理解 Agent 内部的状态流转
2. 可以自定义每个节点的行为（错误处理、日志、限制等）
3. 为 Day 6 的 Workflow 改造打下基础
状态图结构：
  START → agent（LLM推理）→ 有tool_calls? → 是 → tools（执行工具）→ 回到 agent
                                           → 否 → END
运行方式：被 main.py 导入使用
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