"""
记忆模块：三种记忆机制

1. 短期记忆（Short-term Memory）
   - 对话历史管理，防止上下文超限
   - 三种策略：滑动窗口 / Token 截断 / 摘要压缩

2. 长期记忆（Long-term Memory）
   - 用向量数据库存储历史对话摘要
   - 新对话开始时，按相关性召回过去的交互

3. 工作记忆（Working Memory / Scratchpad）
   - Agent 推理过程中的临时笔记本
   - 记录已收集的信息、待完成的子任务

设计理念：
  短期记忆 = 人的"当前对话记忆"（聊到一半记得前面说了啥）
  长期记忆 = 人的"经验记忆"（上次这个客户买过跑鞋）
  工作记忆 = 人的"草稿纸"（边聊边记笔记，整理思路）
"""
from pathlib import Path
from langchain_core.messages import (
    AnyMessage, AIMessage, HumanMessage, SystemMessage, ToolMessage,
    trim_messages,
)
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

from config import get_llm, DASHSCOPE_API_KEY, DASHSCOPE_BASE_URL

# ============================================================
# 一、短期记忆（Short-term Memory）
# ============================================================

class ShortTermMemory:
    """短期记忆：管理当前对话的消息历史
    问题：多轮对话越来越长，LLM 上下文窗口会爆
    解决：用不同策略控制消息数量
    """

    @staticmethod
    def sliding_window(messages: list[AnyMessage], max_pairs: int = 10) -> list[AnyMessage]:
        """策略一：滑动窗口 —— 只保留最近 N 轮对话

            优点：简单粗暴，速度快
            缺点：早期对话直接丢失，用户说"我最开始问的那个"就找不到了

            Args:
                messages: 完整消息历史
                max_pairs: 保留最近多少轮（一轮 = 一次用户消息 + Agent 完整响应链）
        """
        # 保留 SystemMessage
        system_msgs = [m for m in messages if isinstance(m, SystemMessage)]
        non_system = [m for m in messages if not isinstance(m, SystemMessage)]

        if len(non_system) <= max_pairs * 4:
            # 还没超限，不需要截断
            # (一轮对话约 4 条信息: Human + AI(tool_calls) + Tool + AI(answer))
            return messages

        # 保留最近的 max_pairs * 4 条非系统消息
        trimmed = non_system[-(max_pairs * 4):]
        return system_msgs + trimmed

    @staticmethod
    def token_trim(messages: list[AnyMessage], max_tokens: int = 4000) -> list[AnyMessage]:
        """策略二：Token 截断 —— 按 Token 数限制
            优点：精确控制上下文长度
            缺点：可能在消息中间截断，破坏完整性

            使用 LangChain 内置的 trim_messages 工具
        """
        return trim_messages(
            messages,
            max_tokens=max_tokens,
            strategy="last",            # 保留最后的消息（最新消息）
            token_counter=len,          # 简化：用字符数估算（生产环境用 tiktoken）
            start_on="human",           # 确保截断后第一条是用户消息
            include_system=True,        # 始终保留 SystemMessage
            allow_partial=False,        # 不允许截断消息内容
        )

    @staticmethod
    def summarize_and_trim(
            message: list[AnyMessage],
            max_pairs: int = 6,
            summary_threshold: int = 12,
    ) -> list[AnyMessage]:
        """策略三：摘要压缩 —— 早期对话压缩为摘要，近期对话保留原文
            优点：不会完全丢失早期信息
            缺点：摘要会损失细节，且需要额外调一次 LLM（多花 Token 和时间）

            工作方式：
                如果消息超过 summary_threshold 条：
                  1. 把早期消息用 LLM 压缩成一段摘要
                  2. 摘要作为一条 SystemMessage 插入
                  3. 只保留最近 max_pairs 轮的原始消息
        """
        system_msgs = [m for m in message if isinstance(m, SystemMessage)]
        non_system = [m for m in message if not isinstance(m, SystemMessage)]

        if len(non_system) <= summary_threshold:
            return message

        # 分成两部分：要压缩的和要保留的
        to_summarize = non_system[:-(max_pairs * 4)]
        to_keep = non_system[-(max_pairs * 4):]

        # 构建摘要文本（不调 LLM，用简单提取代替，节省 Token）
        summary_parts = []
        for msg in to_summarize:
            if isinstance(msg, HumanMessage):
                summary_parts.append(f"用户问了: {msg.content[:100]}")
            elif isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
                summary_parts.append(f"助手回答了: {msg.content[:100]}")

        if summary_parts:
            summary_text = "【早期对话摘要】\n" + "\n".join(summary_parts)
            summary_msg = SystemMessage(content=summary_text)
            return system_msgs + [summary_msg] + to_keep
        else:
            return system_msgs + to_keep

# ============================================================
# 二、长期记忆（Long-term Memory）
# ============================================================

class LongTermMemory:
    """长期记忆：跨会话的记忆，用向量数据库存储
    场景：用户昨天咨询过跑鞋，今天再来时 Agent 还记得
    实现：每次对话结束后，把对话摘要存入 ChromaDB
            新对话开始时，用用户的第一个问题检索相关的历史交互
    """

    def __init__(self, persist_dir: str = None):
        if persist_dir is None:
            persist_dir = str(Path(__file__).parent / "memory_store")

        self.embeddings = OpenAIEmbeddings(
            api_key=DASHSCOPE_API_KEY,
            base_url=DASHSCOPE_BASE_URL,
            model="text-embedding-v3",
        )
        self.vectorstore = Chroma(
            collection_name="conversation_memory",
            embedding_function=self.embeddings,
            persist_directory=persist_dir,
        )

    def save_conversation(self, session_id: str, messages: list[AIMessage]):
        """保存一次对话的摘要到长期记忆
            Args:
                session_id: 会话ID，用于区分不同对话
                messages: 对话消息列表
        """
        # 提取对话中的关键信息
        user_questions = []
        agent_answers = []

        for msg in messages:
            if isinstance(msg, HumanMessage):
                user_questions.append(msg.content)
            elif isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
                agent_answers.append(msg.content[:200])

        if not user_questions:
            return

        # 构建摘要文档
        summary = (
            f"用户问题: {'; '.join(user_questions)}\n"
            f"助手回答: {'; '.join(agent_answers[:3])}"
        )

        self.vectorstore.add_texts(
            texts=[summary],
            metadatas=[{"session_id": session_id}],
        )

    def recall(self, query: str, k: int = 3) -> list[str]:
        """根据当前问题，召回相关的历史对话
            Args:
                query: 当前用户的问题
                k: 召回几条历史记录
        """
        results = self.vectorstore.similarity_search(query, k=k)
        return [doc.page_content for doc in results]

# ============================================================
# 三、工作记忆（Working Memory / Scratchpad）
# ============================================================
class WorkingMemory:
    """工作记忆：Agent 推理过程中的临时笔记本

    用途：
    1. 记录已经从工具获取的信息（避免重复调用）
    2. 记录用户的偏好（提取的结构化需求）
    3. 记录待完成的子任务

    实现：一个简单的字典，存在 AgentState 中
    每轮推理时作为额外上下文注入到 Prompt
    """
    def __init__(self):
        self.notes: dict[str, str] = {}
        self.user_preferences: dict[str, str] = {}
        self.collected_products: list[str] = []

    def add_note(self, key: str, value: str):
        """添加笔记"""
        self.notes[key] = value

    def set_preference(self, key: str, value: str):
        """记录用户偏好"""
        self.user_preferences[key] = value

    def add_product(self, product_id: str):
        """记录已查看的商品"""
        if product_id not in self.collected_products:
            self.collected_products.append(product_id)

    def to_context_string(self) -> str:
        """转成文本，注入到 Prompt中"""
        parts = []

        if self.user_preferences:
            prefs = "，".join(f"{k}: {v}" for k, v in self.user_preferences.items())
            parts.append(f"【用户偏好】{parts}")

        if self.collected_products:
            parts.append(f"【已查看商品】{', '.join(self.collected_products)}")

        if self.notes:
            notes = "; ".join(f"{k}: {v}" for k, v in self.notes.items())
            parts.append(f"【笔记】{notes}")
        return "\n".join(parts) if parts else ""

    def clear(self):
        """清空工作记忆（新会话时调用）"""
        self.notes.clear()
        self.user_preferences.clear()
        self.collected_products.clear()

