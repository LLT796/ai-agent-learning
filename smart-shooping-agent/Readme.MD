# Smart Shopping Agent

> 项目：淘天智能导购 Agent

## 项目结构

```
smart-shopping-agent/
├── config.py          # 配置：LLM、环境变量
├── tools.py           # 所有工具定义
├── graph.py           # LangGraph 状态图（Agent 核心）
├── prompts.py         # System Prompt（独立管理，不硬编码）
├── main.py            # 入口：运行 Agent
└── README.md          # 本文件
```

## 能力演进

| Day | 新增能力 |
|-----|----------|
| Day 4 | 多工具 + LangGraph 手搓 Agent + 错误处理 |
| Day 5 | 记忆模块（短期 / 长期 / 工作记忆） |
| Day 6 | LangGraph Workflow 改造（状态流转） |
| Day 7 | Prompt 工程化 + 代码整理 |

## 运行

```bash
uv run python smart-shopping-agent/main.py
```