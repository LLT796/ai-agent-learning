# Day 1 开发日志：Tool 定义方式 & Agent 运行机制

> 日期：2026-03-30
> 目标：理解 Agent = LLM + 工具调用 + 循环推理

---

## 一、Tool 的定义方式

### 1.1 核心概念

Tool 是 Agent 的"手和脚"。LLM 本身只能思考和生成文本，Tool 让它能够与外部世界交互（查数据库、调 API、做计算等）。

LLM 不会直接执行 Tool 的代码，它做的是：**根据 Tool 的描述（docstring）决定是否调用、传什么参数**。所以 Tool 的描述写得好不好，直接决定 Agent 选工具的准确率。

### 1.2 使用 @tool 装饰器定义

这是最常用的方式，用 LangChain 的 `@tool` 装饰器把一个普通 Python 函数变成 Agent 可调用的工具。

```python
from langchain_core.tools import tool

@tool
def get_product_info(product_name: str) -> str:
    """查询商品信息。当用户询问某个商品的价格、库存或详情时使用此工具。
    
    Args:
        product_name: 商品名称，如 '耐克跑鞋' 或 'iPhone 16'
    """
    # 函数体：实际的业务逻辑
    return "商品信息..."
```

### 1.3 @tool 装饰器背后做了什么

装饰器会从函数中提取三样东西，组装成一个标准的 Tool 对象：

| 提取项 | 来源 | 作用 |
|--------|------|------|
| **工具名称** | 函数名 `get_product_info` | LLM 在决定调用时引用这个名字 |
| **工具描述** | 函数的 docstring | LLM 根据这段文字判断"什么时候该用这个工具" |
| **参数定义** | 函数签名 + Args 描述 | LLM 知道要传什么参数、什么类型 |

这三样东西最终会被序列化成 JSON Schema，随 System Prompt 一起发给 LLM。
LLM 看到的大概是这样的：

```json
{
  "name": "get_product_info",
  "description": "查询商品信息。当用户询问某个商品的价格、库存或详情时使用此工具。",
  "parameters": {
    "type": "object",
    "properties": {
      "product_name": {
        "type": "string",
        "description": "商品名称，如 '耐克跑鞋' 或 'iPhone 16'"
      }
    },
    "required": ["product_name"]
  }
}
```

### 1.4 写好 Tool 的关键原则

1. **docstring 是给 LLM 看的，不是给人看的**
   - 要用自然语言清楚描述"什么场景下该用这个工具"
   - 要写清楚每个参数的含义和示例值
   - 不需要写实现细节，LLM 不关心你内部怎么实现

2. **函数名要有语义**
   - `get_product_info` 比 `query` 好
   - `compare_products` 比 `compare` 好
   - LLM 会把函数名也作为理解工具用途的线索

3. **返回值要是字符串**
   - Tool 的返回值会作为 Observation 喂回给 LLM
   - 返回格式化的、LLM 容易理解的字符串
   - 包含关键数据即可，不要返回过长的内容

4. **一个工具做一件事**
   - `get_product_info` 只查询，不做推荐
   - `compare_products` 只对比，不做决策
   - 让 LLM 来组合多个工具完成复杂任务

### 1.5 多工具时 LLM 如何选择

当 Agent 有多个工具时（如我们的 4 工具版本），LLM 收到的 prompt 里包含所有工具的描述。
它会根据用户输入，匹配最合适的工具：

- "华为耳机多少钱" → 匹配 `get_product_info`（精确查单个商品）
- "有什么1000以内的运动鞋" → 匹配 `search_products`（按条件筛选）
- "耐克和阿迪哪个好" → 匹配 `compare_products`（两个商品对比）
- "打了几折" → 匹配 `calculate`（需要数学计算）

如果一个任务需要多步，LLM 会**依次调用多个工具**，每次拿到结果后再决定下一步。

---

## 二、Agent 的运行机制（ReAct 循环）

### 2.1 整体架构

```
用户输入
   ↓
┌─────────────────────────────────────────────┐
│              ReAct 循环                       │
│                                               │
│  LLM 思考（Thought）                          │
│    ↓                                          │
│  需要工具？──── 否 ───→ 输出最终答案（Final）   │
│    │ 是                                       │
│    ↓                                          │
│  调用工具（Action）                            │
│    ↓                                          │
│  获得结果（Observation）                       │
│    ↓                                          │
│  回到 LLM 思考（带上新信息）                    │
│    ↓                                          │
│  ......循环直到 LLM 认为信息足够               │
│                                               │
└─────────────────────────────────────────────┘
   ↓
最终回答返回给用户
```

### 2.2 create_react_agent 做了什么

```python
from langgraph.prebuilt import create_react_agent

agent = create_react_agent(
    model=llm,           # LLM 实例（通义千问）
    tools=[tool1, tool2], # 工具列表
    prompt=system_prompt,  # System Prompt
)
```

这一行代码内部构建了一个 LangGraph 状态图（StateGraph），包含：

- **agent 节点**：调用 LLM，让它决定下一步做什么
- **tools 节点**：执行 LLM 选择的工具，拿到结果
- **条件边**：如果 LLM 返回了 tool_calls → 走 tools 节点；否则 → 结束

本质上就是一个循环：agent → tools → agent → tools → ... → 结束

### 2.3 一次完整调用的详细流程

以 "蓝牙耳机打几折？帮我算算便宜了多少" 为例：

**第 1 轮：**
```
输入：用户消息 "蓝牙耳机打几折？帮我算算便宜了多少"
   ↓
LLM 思考：用户想知道折扣信息，我需要先查商品信息
LLM 输出：tool_calls=[{name: "get_product_info", args: {product_name: "蓝牙耳机"}}]
   ↓
执行工具：get_product_info("蓝牙耳机")
返回结果："商品：Sony WF-1000XM5，现价：¥1699，原价：¥1999..."
```

**第 2 轮：**
```
输入：之前的对话 + 工具返回的商品信息
   ↓
LLM 思考：我知道原价1999、现价1699了，用户还想知道便宜了多少，我来算
LLM 输出：tool_calls=[{name: "calculate", args: {expression: "1999 - 1699"}}]
   ↓
执行工具：calculate("1999 - 1699")
返回结果："1999 - 1699 = 300"
```

**第 3 轮：**
```
输入：之前的对话 + 商品信息 + 计算结果
   ↓
LLM 思考：信息够了，我可以回答了
LLM 输出：content="蓝牙耳机目前约8.5折，比原价便宜了300元..."（无 tool_calls）
   ↓
没有 tool_calls → 循环结束 → 返回最终答案给用户
```

### 2.4 关键机制总结

| 机制 | 说明 |
|------|------|
| **循环的驱动者** | 是 LLM 自己。它每次看到 Observation 后决定是继续调工具还是直接回答 |
| **循环的终止条件** | LLM 的输出里没有 tool_calls，只有 content（最终文本） |
| **上下文累积** | 每一轮的 tool_calls 和 Observation 都会追加到消息列表中，LLM 能看到完整的推理历史 |
| **错误处理** | 如果工具返回错误信息，LLM 可以看到错误并决定换一种方式（比如换个参数重试） |
| **最大步数** | 默认有步数限制，防止 LLM 无限循环调工具 |

### 2.5 messages 列表的真实结构

调用 `agent.invoke()` 后返回的 `result["messages"]` 就是整个推理过程的记录：

```python
[
    HumanMessage("蓝牙耳机打几折？"),           # 用户输入
    AIMessage(tool_calls=[...]),                # LLM 第 1 次决策：调工具
    ToolMessage("商品信息..."),                  # 工具返回结果
    AIMessage(tool_calls=[...]),                # LLM 第 2 次决策：再调工具
    ToolMessage("1999 - 1699 = 300"),           # 工具返回结果
    AIMessage(content="蓝牙耳机约8.5折..."),     # LLM 最终回答
]
```

三种消息类型：
- **HumanMessage**：用户说的话
- **AIMessage**：LLM 的输出（可能包含 tool_calls 或 content）
- **ToolMessage**：工具执行后的返回值

---

## 三、今日踩过的坑

1. **load_dotenv 路径问题**：在 Windows 上用相对路径 `"day1-basic-agent/.env"` 可能找不到文件。
   解决：改用 `Path(__file__).parent / ".env"` 基于脚本自身定位。

2. **阿里云账户欠费**：API 返回 `Arrearage` 错误，需要先充值几块钱。

3. **LangGraph 版本警告**：`create_react_agent` 的 import 路径在 LangGraph V1.0 中已迁移，
   从 `langgraph.prebuilt` 移到了 `langchain.agents`，当前代码仍可运行但有 DeprecationWarning。

---

## 四、Day 1 自检

- [x] 能口述 ReAct 循环的工作流程：Thought → Action → Observation → 循环
- [x] 能解释 @tool 装饰器提取了哪三样东西：名称、描述、参数
- [x] 理解 LLM 是根据 docstring 选工具，不是根据函数内部实现
- [x] 理解循环终止条件：LLM 输出中没有 tool_calls
- [x] 能区分 HumanMessage / AIMessage / ToolMessage 三种消息类型
