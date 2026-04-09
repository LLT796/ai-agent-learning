# Day 4 开发日志：LangGraph 手搓多步推理 Agent

> 日期：2026-04-02
> 目标：用 LangGraph StateGraph 手动构建多工具、多步推理的导购 Agent

---

## 一、Day 4 做了什么

把 Day 1 的"一个脚本跑通 Agent"升级为**模块化、可控、生产级**的 Agent 项目：

```
smart-shopping-agent/
├── config.py     # LLM 配置集中管理
├── tools.py      # 5 个工具，有调用层次关系
├── prompts.py    # System Prompt 独立管理
├── graph.py      # LangGraph 状态图（Agent 核心）
└── main.py       # 入口 + 测试场景 + 交互模式
```

核心变化：不再用 `create_react_agent` 黑盒，而是用 `StateGraph` 手动构建 Agent 的推理循环。

---

## 二、LangGraph StateGraph 核心概念

### 2.1 三个构建块

LangGraph 的 Agent 由三样东西组成：

| 构建块 | 是什么 | 类比 |
|--------|--------|------|
| **State** | Agent 的共享记忆，所有节点都能读写 | 全局变量 |
| **Node** | 处理函数，接收 State、返回更新 | 流水线上的工位 |
| **Edge** | 节点之间的连接，决定执行顺序 | 流水线的传送带 |

### 2.2 State 的设计

```python
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]  # 消息历史
    step_count: int                                       # 已执行步数
```

两个字段各有用途：

- `messages`：完整的对话和推理历史。`add_messages` 注解意味着新消息是追加，不是覆盖。这保证了 LLM 每轮推理都能看到之前所有的 tool_calls 和 Observation。
- `step_count`：防御性字段，用于限制最大推理步数，防止 Agent 无限循环调工具。

### 2.3 两个 Node

**agent_node（LLM 推理节点）**：
- 把当前所有 messages 发给 LLM
- LLM 返回 AIMessage（可能包含 tool_calls，也可能是最终文本回答）
- 如果 step_count 超过 MAX_STEPS，强制返回兜底回答

**safe_tool_node（工具执行节点）**：
- 读取 AIMessage 中的 tool_calls，执行对应的工具函数
- 把工具返回值包装成 ToolMessage 追加到 messages
- 关键：用 try/except 包裹，工具报错不会让 Agent 崩溃，而是返回错误信息让 LLM 自己决定下一步

### 2.4 条件边（Conditional Edge）

```python
def should_continue(state: AgentState) -> str:
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"    # 还需要调工具 → 去 tools 节点
    else:
        return END         # 不需要了 → 结束
```

这个函数是 Agent 循环的"开关"：LLM 输出了 tool_calls 就继续，没有就停。和 Day 1 学的 ReAct 循环终止条件完全一致，只是现在自己写出来了。

### 2.5 完整的图结构

```
START
  ↓
agent_node（LLM 推理）
  ↓
should_continue?
  ├── 有 tool_calls → tools_node（执行工具）→ 回到 agent_node
  └── 无 tool_calls → END（输出最终回答）
```

这个循环会一直转，直到 LLM 认为信息够了不再调工具，或者达到 MAX_STEPS 上限。

---

## 三、手搓 Agent vs create_react_agent

| 维度 | Day 1: create_react_agent | Day 4: StateGraph 手搓 |
|------|---------------------------|----------------------|
| 代码量 | 1 行 | ~80 行 |
| 透明度 | 黑盒，看不到内部逻辑 | 完全透明，每个节点可控 |
| 错误处理 | 工具报错直接崩溃 | safe_tool_node 捕获异常，返回错误信息 |
| 步数限制 | 依赖框架默认值 | MAX_STEPS 自定义控制 |
| 自定义节点 | 不支持 | 随时加新 Node（如日志节点、审核节点） |
| 状态扩展 | 只有 messages | 可加任意字段（step_count、user_profile 等） |
| 适用场景 | 快速原型验证 | 生产环境 |

核心认识：`create_react_agent` 就是把今天写的这些代码封装了一层。理解了内部原理后，遇到它满足不了的需求，就知道怎么自己改。

---

## 四、工具设计的层次关系

Day 1 的工具是平级的，Agent 随便调哪个都行。Day 4 的 5 个工具有明确的**调用层次**：

```
search_products（搜索，入口）
  ↓ 返回商品 ID 列表
get_product_detail（查详情，需要 ID）
  ↓ 返回完整信息
compare_products（对比，需要两个 ID）
  ↓ 返回对比结果
calculate_price（计算，需要具体数字）
  ↓ 返回计算结果
get_recommendation_reason（推荐，需要 ID + 用户需求）
  ↓ 返回推荐话术
```

LLM 要自己规划调用顺序。它不能直接调 `compare_products`，因为它还不知道商品 ID——必须先搜索。这就是**多步推理**的核心：LLM 通过多轮 Thought → Action → Observation 循环，逐步收集信息，最终给出完整答案。

工具的 docstring 里也暗示了这个顺序，比如：

```python
"""获取某个商品的完整详情。需要先通过 search_products 获取商品ID。"""
```

这句话不是给人看的，是**引导 LLM 按正确顺序调工具**。

---

## 五、防御性编程：不信任 LLM 的输出

### 5.1 参数清洗

```python
product = PRODUCTS.get(product_id.upper().strip())
```

`.upper()` 防止 LLM 传小写 `"sku003"`，`.strip()` 防止带空格 `" SKU003 "`。LLM 生成的参数经常有这类格式瑕疵，不做容错就会 KeyError。

### 5.2 安全计算

```python
allowed = set("0123456789.+-*/() ")
if not all(c in allowed for c in expression):
    return "❌ 表达式包含非法字符"
```

`calculate_price` 工具用了 `eval()`，如果 LLM 传了恶意代码会很危险。白名单限制只允许数字和运算符，生产环境应该用专门的数学解析库。

### 5.3 最大步数限制

```python
MAX_STEPS = 10

if step_count >= MAX_STEPS:
    return {"messages": [AIMessage(content="抱歉，处理遇到困难...")]}
```

防止 LLM 陷入"调工具 → 结果不满意 → 换参数再调 → 还不满意 → 继续调..."的死循环。超过 10 步强制终止，给用户一个兜底回答。

### 5.4 工具异常捕获

```python
def safe_tool_node(state):
    try:
        return tool_node.invoke(state)
    except Exception as e:
        return {"messages": [ToolMessage(content=f"❌ 工具执行出错: {e}")]}
```

工具可能因为各种原因失败（网络超时、参数错误、数据库异常）。捕获异常后返回错误信息给 LLM，让它自己决定重试还是换策略，而不是让整个 Agent 崩溃。

---

## 六、Prompt 工程化

### 6.1 独立管理

Prompt 放在 `prompts.py` 中，不硬编码在 `graph.py` 或 `main.py` 里。好处：

- 改 Prompt 不需要动业务逻辑代码
- 可以维护多个版本（v1、v2），方便 A/B 测试
- 团队协作时非技术人员也可以调 Prompt

### 6.2 结构化设计

```
角色定义：你是导购助手「小淘」
  ↓
工具说明：5 个工具的用途和调用时机
  ↓
工作流程：理解需求 → 搜索 → 详情 → 对比 → 推荐
  ↓
回答规范：简洁、基于真实数据、给出理由
  ↓
限制边界：只推荐自营商品、不能下单、不回答无关问题
```

这个结构是通用的——几乎所有 Agent 的 System Prompt 都可以按这五个模块组织。

### 6.3 Prompt 调优观察

实验发现的几个规律：
- 在工具说明中加"通常第一步"这样的顺序暗示，Agent 的调用顺序更合理
- "不要编造价格和参数"这条规则非常有效，显著减少幻觉
- "涉及价格计算时使用 calculate_price 工具，不要心算"也很关键，否则 LLM 经常自己算然后算错
- 删掉"不要回答与购物无关的问题"后，Agent 会开始回答任何问题，边界约束必须显式声明

---

## 七、测试场景观察记录

| 场景 | 实际步数 | Agent 行为 | 是否符合预期 |
|------|----------|------------|-------------|
| 简单查询"华为耳机多少钱" | 1-2 步 | 搜索 → 回答 | ✅ |
| 带条件"1000以内运动鞋" | 2 步 | 搜索(category+max_price) → 回答 | ✅ |
| 对比+计算"索尼vs华为哪个划算" | 3-4 步 | 搜索 → 对比 → 计算差价 → 回答 | ✅ |
| 完整导购"推荐跑步鞋" | 3-5 步 | 搜索 → 详情 → 推荐理由 → 回答 | ✅ |
| 模糊需求"帮我挑礼物" | 0 步 | 追问预算、送谁、场景 | ✅ |
| 超出范围"帮我写诗" | 0 步 | 礼貌拒绝 | ✅ |

关键发现：Agent 在对比场景中有时会跳过 `search_products` 直接调 `compare_products`，如果 LLM 从上下文猜到了 ID 的话。这不算错误，但说明 LLM 的规划路径不完全可预测。

---

## 八、今日踩过的坑

1. **模块导入路径**：从项目根目录运行 `uv run python smart-shopping-agent/main.py` 时，Python 不会自动把 `smart-shopping-agent/` 加入模块搜索路径。如果遇到 `ModuleNotFoundError`，需要在运行命令前加 `cd smart-shopping-agent` 或者用相对导入。

2. **LangGraph 版本兼容**：`langgraph` v1.x 的 API 和一些教程中的 v0.x 写法不同，比如 `add_conditional_edges` 的参数格式。以官方最新文档为准。

3. **LLM 心算问题**：不加"不要心算"的限制时，qwen-plus 在简单计算（如 1999-1699）上经常直接给出答案而不调 `calculate_price`，偶尔会算错。显式要求"用工具算"后改善明显。

---

## 九、Day 4 与前三天的关系

```
Day 1: @tool + create_react_agent
  → 学会了 Agent 的基本概念（工具、ReAct循环）
  → 但用的是黑盒封装

Day 2: RAG Pipeline
  → 学会了从文档检索知识
  → 但只是固定管线，没有 Agent 决策

Day 3: MCP Server
  → 学会了把能力标准化暴露
  → 但只是协议层，没有推理逻辑

Day 4: LangGraph StateGraph ← 今天
  → 手搓了 Agent 的内部机制
  → 理解了 State、Node、Edge 三件套
  → 为 Day 5（记忆）和 Day 6（Workflow）打下基础
```

明天 Day 5 会在这个项目上加记忆模块——让 Agent 能在多轮对话中记住之前聊过什么。

---

## 十、Day 4 自检

- [x] 能解释 StateGraph 的三个构建块：State、Node、Edge
- [x] 能说清楚 agent_node 和 tool_node 各自做什么
- [x] 理解 should_continue 条件边的判断逻辑
- [x] 理解 MAX_STEPS 和 safe_tool_node 的防御作用
- [x] 能说出 5 个工具之间的调用层次关系
- [x] 理解 Prompt 独立管理的好处
- [x] Agent 能稳定完成 3 步以上的导购任务
