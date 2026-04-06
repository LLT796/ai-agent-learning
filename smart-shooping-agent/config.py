"""
配置模块：LLM 初始化、环境变量加载
所有需要 API Key 的地方都从这里导入，避免分散管理
"""
import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# 加载 .env（兼容从项目根目录或子目录运行）
_env_path = Path(__file__).parent.parent / "day1-basic-agent" / ".env"
if _env_path.exists():
    load_dotenv(_env_path)

DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

def get_llm(temperature: float = 0, model: str = "qwen-plus") -> ChatOpenAI:
    """获取 LLM 实例, 统一管理"""
    return ChatOpenAI(
        api_key=DASHSCOPE_API_KEY,
        base_url=DASHSCOPE_BASE_URL,
        model=model,
        temperature=temperature,
    )