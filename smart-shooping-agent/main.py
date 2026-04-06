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