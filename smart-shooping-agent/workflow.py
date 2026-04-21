"""
Day 6 - 导购 Workflow（有明确状态流转的 Agent）

和 Day 4 ReAct Agent 的核心区别：
  ReAct: LLM 自由决定调什么工具、什么顺序 → 灵活但不可控
  Workflow: 预定义好流程节点和分支条件 → 可预测、可调试、可审计

节点设计：
  classify_intent  → 判断用户意图（购物/闲聊/模糊）
  clarify_needs    → 追问用户需求
  search           → 搜索商品
  analyze_results  → 分析搜索结果数量，决定下一步
  get_detail       → 查看单个商品详情
  compare          → 对比多个商品
  recommend        → 生成推荐理由
  respond          → 输出最终回答
  reject           → 拒绝非购物请求

运行方式：被 main_v3.py 导入
"""

from typing import Annotated, Literal
from typing_extensions import TypedDict

from langchain_core.messages import AnyMessage, AIMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

from config import get_llm
from tools import PRODUCTS


# ============================================================
# State：比 Day 4 丰富得多，每个字段对应一个业务含义
# ============================================================

class WorkflowState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    # --- 流程控制字段 ---
    intent: str  # 用户意图：shopping / chitchat / vague
    search_results: list[dict]  # 搜索到的商品列表
    selected_products: list[str]  # 选中要详细查看的商品 ID
    comparison_done: bool  # 是否已完成对比
    recommendation: str  # 最终推荐文本
    final_response: str  # 输出给用户的回答


llm = get_llm()

# ============================================================
# 节点 1: 意图分类
# ============================================================
def classify_intent(state: WorkflowState) -> dict:
    """判断用户的意图：明确购物需求 / 模糊需求 / 非购物话题

    这是 Workflow 的入口节点，决定后续走哪条分支。
    用 LLM 做分类，但限制输出格式，确保可解析。
    """
    messages = state["messages"]
    last_user_msg = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            last_user_msg = msg.content
            break

    classify_prompt = f"""请判断以下用户输入的意图类别，只输出类别名称，不要其他内容。

类别：
- shopping: 用户有明确的购物需求（想买某类商品、问价格、要推荐等）
- vague: 用户想买东西但需求不明确（帮我挑个礼物、有什么好东西等）
- chitchat: 非购物话题（闲聊、写作、翻译等）

用户输入："{last_user_msg}"

类别："""

    response = llm.invoke([HumanMessage(content=classify_prompt)])
    intent = response.content.strip().lower()

    # 容错处理：如果 LLM 输出了多余内容，提取关键词
    if "shopping" in intent:
        intent = "shopping"
    elif "vague" in intent:
        intent = "vague"
    else:
        intent = "chitchat"

    print(f"    🏷️ 意图分类: {intent}")
    return {"intent": intent}


# ============================================================
# 节点 2: 需求澄清（模糊意图时触发）
# ============================================================

def clarify_needs(state: WorkflowState) -> dict:
    """当用户需求模糊时，生成追问话术"""
    messages = state["messages"]
    last_user_msg = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            last_user_msg = msg.content
            break

    clarify_prompt = f"""用户说了"{last_user_msg}"，但需求不够明确。
请生成一段简短的追问（不超过3个问题），帮助明确：
1. 想买什么品类的商品
2. 预算大概多少
3. 主要用途或使用场景

语气要友好自然，像一个热情的导购员。"""

    response = llm.invoke([HumanMessage(content=clarify_prompt)])
    print(f"    ❓ 追问: {response.content[:80]}...")
    return {
        "final_response": response.content,
    }


# ============================================================
# 节点 3: 拒绝非购物请求
# ============================================================

def reject(state: WorkflowState) -> dict:
    """礼貌拒绝非购物话题"""
    return {
        "final_response": "不好意思，我是淘天自营的导购助手，主要帮您挑选和推荐商品。有什么购物方面的需求我可以帮您吗？😊"
    }


# ============================================================
# 节点 4: 搜索商品
# ============================================================

def search(state: WorkflowState) -> dict:
    """根据用户需求搜索商品

    不直接调 tool，而是在节点内部用 LLM 提取搜索条件，然后查数据库。
    Workflow 模式下工具调用是显式的——你决定在哪个节点调什么。
    """
    messages = state["messages"]
    last_user_msg = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            last_user_msg = msg.content
            break

    # 用 LLM 提取搜索条件
    extract_prompt = f"""从以下用户输入中提取商品搜索条件，用 JSON 格式输出：
{{"category": "商品分类（运动鞋/耳机/背包/手表/键盘/空字符串）", "max_price": 最高价格数字或0, "keyword": "关键词或空字符串"}}

只输出 JSON，不要其他内容。

用户输入："{last_user_msg}" """

    response = llm.invoke([HumanMessage(content=extract_prompt)])

    # 解析 LLM 输出的搜索条件
    import json
    try:
        raw = response.content.strip()
        # 去掉可能的 markdown 代码块标记
        raw = raw.replace("```json", "").replace("```", "").strip()
        conditions = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        conditions = {"category": "", "max_price": 0, "keyword": ""}

    category = conditions.get("category", "")
    max_price = conditions.get("max_price", 0)
    keyword = conditions.get("keyword", "")

    print(f"    🔍 搜索条件: 分类={category}, 最高价={max_price}, 关键词={keyword}")

    # 在数据库中搜索
    results = []
    for product in PRODUCTS.values():
        if category and product["category"] != category:
            continue
        if max_price and product["price"] > max_price:
            continue
        if keyword and keyword not in product["name"] and keyword not in product.get("description", ""):
            continue
        results.append(product)

    print(f"    📦 找到 {len(results)} 个商品")
    return {
        "search_results": results,
        "selected_products": [p["id"] for p in results[:3]],  # 最多选前3个
    }