# Day 3 开发日志：MCP 协议实战

> 日期：2026-04-01
> 目标：理解 MCP 协议，能写一个 MCP Server 并连上 Client

---

## 一、MCP 是什么

MCP（Model Context Protocol）是一个**为 LLM 设计的标准化协议**，定义了 LLM 应用（Host）和外部能力（Server）之间的通信方式。

最精准的类比是 **USB-C**：不管你是什么设备（Claude Desktop、Cursor、自己的 Agent），不管你要连什么能力（商品目录、数据库、文件系统），只要双方都实现了 MCP 协议，插上就能用。

MCP 由 Anthropic 于 2024 年底发布，2026 年已成为 AI Agent 生态的事实标准，OpenAI、Vercel 等主要平台均已支持。

---

## 二、MCP 的架构

```
┌─────────────────────┐
│      MCP Host        │    Claude Desktop / Cursor / 你的 App
│  ┌───────────────┐   │
│  │  MCP Client   │───┼──── MCP 协议 ──→ MCP Server A (商品目录) ──→ 数据库
│  │  (内置)       │───┼──── MCP 协议 ──→ MCP Server B (订单系统) ──→ API
│  └───────────────┘   │
└─────────────────────┘
```

三个角色：
- **Host**：运行 LLM 的应用（Claude Desktop、你的 Agent 程序）
- **Client**：Host 内部的 MCP 客户端模块，负责与 Server 通信
- **Server**：暴露具体能力的服务（你今天写的就是这个）

一个 Host 可以同时连接多个 Server，每个 Server 专注做一件事。

---

## 三、MCP Server 的三类能力

这是 MCP 协议最核心的设计——把 Server 能暴露的能力分为三类，每类有不同的控制者和使用方式。

### 3.1 Tools（工具）—— LLM 控制

**类比**：REST API 的 POST 接口
**谁决定调用**：LLM 根据用户意图自行判断
**用途**：执行操作、查询数据、产生副作用

```python
@mcp.tool()
def search_products(category: str = "", max_price: float = 0) -> str:
    """搜索商品。可按分类、最高价格筛选。"""
    # 业务逻辑...
    return "搜索结果..."
```

和 Day 1 的 `@tool` 装饰器非常相似——函数名是工具名，docstring 是工具描述，参数签名定义输入格式。区别在于 Day 1 的 `@tool` 是 LangChain 的私有格式，而 `@mcp.tool()` 遵循 MCP 标准协议，任何兼容 Client 都能识别。

### 3.2 Resources（资源）—— 应用程序控制

**类比**：REST API 的 GET 接口
**谁决定读取**：应用程序（不是 LLM）
**用途**：暴露只读数据，如配置、目录、文档

```python
@mcp.resource("product://catalog")
def get_product_catalog() -> str:
    """获取完整的商品目录列表。"""
    return "商品目录内容..."
```

Resource 和 Tool 的关键区别：
- Tool 由 LLM 在推理过程中决定是否调用（动态的）
- Resource 由应用程序在初始化或特定时机主动读取（静态的）
- Resource 适合加载"背景信息"到 LLM 上下文，Tool 适合"按需查询"

Resource 还支持参数化 URI（模板化）：

```python
@mcp.resource("product://{product_id}/detail")
def get_product_resource(product_id: str) -> str:
    """获取指定商品的详情"""
    return f"商品 {product_id} 的详情..."
```

### 3.3 Prompts（提示模板）—— 用户控制

**类比**：预定义的交互模板
**谁决定使用**：用户在 Client 界面中选择
**用途**：提供标准化的 LLM 交互方式

```python
@mcp.prompt()
def shopping_guide(user_need: str) -> str:
    """导购助手话术模板。"""
    return f"你是导购助手，用户需求是：{user_need}，请按以下流程帮助..."
```

Prompt 不是 LLM 自己选的，而是用户显式触发的（比如在 Claude Desktop 中选择"使用导购模式"）。它的作用是标准化常见的交互场景，让用户不需要每次手动输入复杂的指令。

### 3.4 三类能力的对比总结

| 维度 | Tools | Resources | Prompts |
|------|-------|-----------|---------|
| 类比 | POST 接口 | GET 接口 | 模板文件 |
| 控制者 | LLM 决定调用 | 应用程序读取 | 用户选择 |
| 时机 | 推理过程中按需 | 启动时或特定时机 | 用户主动触发 |
| 用途 | 执行操作/查询 | 加载上下文数据 | 引导 LLM 行为 |
| 发现 | list_tools() | list_resources() | list_prompts() |
| 调用 | call_tool() | read_resource() | get_prompt() |
| 示例 | 搜索商品、下单 | 商品目录、政策 | 导购话术、对比模板 |

---

## 四、Transport 模式（通信方式）

MCP 协议不绑定特定的传输方式，目前支持两种主要模式：

### 4.1 stdio（标准输入输出）

```
Host 进程 ──→ 启动 Server 子进程 ──→ 通过 stdin/stdout 通信
```

- Server 作为 Host 的子进程运行
- 通过标准输入输出传递 JSON-RPC 消息
- 适合本地开发、Claude Desktop 集成
- 不需要网络，安全性高

启动方式：
```python
mcp.run()  # 默认就是 stdio
```

### 4.2 Streamable HTTP

```
Client ──→ HTTP 请求 ──→ Server（独立运行的 HTTP 服务）
```

- Server 作为独立的 HTTP 服务运行
- Client 通过网络连接
- 适合远程部署、多 Client 共享一个 Server
- 需要处理网络安全（认证、加密等）

启动方式：
```python
mcp.run(transport="streamable-http", host="127.0.0.1", port=8000)
```

### 4.3 如何选择

| 场景 | 推荐 Transport |
|------|---------------|
| 本地开发和测试 | stdio |
| Claude Desktop 集成 | stdio |
| 部署到服务器供多人使用 | Streamable HTTP |
| 需要跨网络访问 | Streamable HTTP |

---

## 五、FastMCP 的设计哲学

FastMCP 是 MCP Python SDK 提供的高级 API，设计理念和 Day 1 的 `@tool` 装饰器一脉相承：

1. **装饰器定义能力**：用 `@mcp.tool()` / `@mcp.resource()` / `@mcp.prompt()`
2. **类型提示即参数定义**：从函数签名自动生成 JSON Schema
3. **docstring 即描述**：给 LLM 或用户看的说明
4. **零模板代码**：不需要手动写 JSON Schema 或注册函数

```python
# 这个函数的签名和 docstring 会被自动转成 MCP 协议格式
@mcp.tool()
def search_products(category: str = "", max_price: float = 0) -> str:
    """搜索商品。可按分类、最高价格筛选。"""
```

FastMCP 自动把这些信息序列化为标准 MCP 消息，Client 调 `list_tools()` 就能看到。

---

## 六、MCP Client 的工作流程

通过 02_test_client.py 验证的完整流程：

```
1. 建立连接
   Client ──→ Server: initialize（握手）
   Server ──→ Client: 返回 Server 信息和支持的能力

2. 发现能力
   Client ──→ Server: list_tools / list_resources / list_prompts
   Server ──→ Client: 返回所有能力的描述和参数定义

3. 使用能力
   Client ──→ Server: call_tool("search_products", {category: "耳机"})
   Server ──→ Client: 返回搜索结果
   
   Client ──→ Server: read_resource("product://catalog")
   Server ──→ Client: 返回商品目录
   
   Client ──→ Server: get_prompt("shopping_guide", {user_need: "买耳机"})
   Server ──→ Client: 返回填充好的 Prompt 模板
```

关键点：**Client 不需要提前知道 Server 有什么能力**。它通过 `list_*` 方法动态发现，这就是 MCP "即插即用" 的核心。

---

## 七、MCP vs 普通 REST API

| 维度 | 普通 REST API | MCP Server |
|------|--------------|------------|
| 协议 | HTTP + 自定义接口设计 | MCP 标准协议（JSON-RPC） |
| 发现机制 | 需要读 Swagger/文档 | Client 自动发现（list_tools 等） |
| 描述方式 | OpenAPI Spec | JSON Schema + 自然语言 docstring |
| 消费者 | 开发者写代码调用 | LLM 根据描述自主调用 |
| 互操作性 | 每个 API 接口不同 | 所有 Server 遵循同一协议 |
| 能力分类 | 无标准分类 | Tool / Resource / Prompt 三类 |
| 连接方式 | HTTP only | stdio / HTTP / SSE 多种 |

一句话总结：REST API 是给开发者用的，MCP 是给 LLM 用的。

---

## 八、MCP 与 Day 1、Day 2 的关系

三天的内容构成了一条清晰的能力演进链：

```
Day 1: Agent + @tool 装饰器
  → LLM 能调用工具了，但工具是硬编码在 Agent 代码里的
  → 问题：换一个 Agent 框架，工具要重写

Day 2: RAG Pipeline
  → LLM 能从文档中检索知识了，但只是一条固定管线
  → 问题：知识检索逻辑和 Agent 逻辑耦合

Day 3: MCP Server
  → 把工具和数据源封装为标准化服务
  → 任何 MCP Client（Claude/Cursor/自己的Agent）都能即插即用
  → 解耦了"能力提供方"和"能力消费方"
```

在后续 Day 4-7 的整合项目中，可以：
- 把 Day 2 的 RAG 知识库包装成一个 MCP Resource + Tool
- Agent 通过 MCP 协议连接，实现松耦合的架构
- 换 Agent 框架不影响 MCP Server，换数据源不影响 Agent

---

## 九、今日踩过的坑

1. **mcp[cli] 安装**：`uv add "mcp[cli]"` 中的 `[cli]` 是额外依赖组，包含 MCP Inspector 和命令行工具。不加 `[cli]` 的话 `mcp dev` 命令不可用。

2. **Windows stdio 兼容性**：Windows 上 stdio transport 可能有进程管理的兼容性问题。如果 Client 脚本报错，优先用 MCP Inspector 浏览器工具测试 Server。

3. **async 编程**：MCP Client 是异步的（`async with`），需要用 `asyncio.run()` 包裹。如果之前不熟悉 Python 异步编程，这里需要适应 `async/await` 的写法。

---

## 十、Day 3 自检

- [x] 能解释 MCP 是什么：为 LLM 设计的标准化协议，类似 USB-C
- [x] 能区分 Tool / Resource / Prompt 三类能力的控制者和用途
- [x] 能用 FastMCP 写一个包含三类能力的 Server
- [x] 理解 stdio 和 HTTP 两种 transport 模式的区别和适用场景
- [x] 能向别人解释 MCP 和普通 REST API 的核心区别
- [x] 理解 Client 的 "发现 → 调用" 两步工作流程
- [x] 能说清楚 Day 1-3 三天内容的演进关系
