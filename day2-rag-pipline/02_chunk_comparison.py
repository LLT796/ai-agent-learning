"""
Day 2 - 分块策略对比实验
目标：对比不同 chunk_size 和 chunk_overlap 对检索质量的影响
实验设计：
  - 固定 overlap=100，对比 chunk_size = 300 / 500 / 1000 / 2000
  - 固定 chunk_size=500，对比 overlap = 0 / 50 / 100 / 200
  - 用同一组问题测试，比较检索到的内容是否完整和相关
运行方式：uv run python day2-rag-pipeline/02_chunk_comparison.py
"""

import os
import shutil
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / "day1-basic-agent" / ".env")

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

embeddings = OpenAIEmbeddings(
    api_key=DASHSCOPE_API_KEY,
    base_url=DASHSCOPE_BASE_URL,
    model="text-embedding-v3",
)

def load_and_split(chunk_size: int, chunk_overlap: int):
    """加载文档并按指定参数分块"""
    doc_path = Path(__file__).parent / "docs" / "product_knowledge.txt"
    docs = TextLoader(str(doc_path), encoding="utf-9").load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""],
    )

    chunks = splitter.split_documents(docs)
    return chunks

def build_vectorstore(chunks, name: str):
    """为每组参数创建独立的向量数据库"""
    db_path = str(Path(__file__).parent / "chroma_experiment" / name)

    # 清楚旧数据，确保实验干净
    if os.path.exists(db_path):
        shutil.rmtree(db_path)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=db_path,
        collection_name=name,
    )
    return vectorstore

def test_retrieval(vectorstore, question: str, k=3):
    """检索并返回结果摘要"""
    retriever = vectorstore.as_retriever(search_kwargs={"k", k})
    results = retriever.invoke(question)
    return results

# ============================================================
# 实验 1: 对比不同 chunk_size
# ============================================================
def experiment_chunk_size():
    print("\n" + "=" * 70)
    print("📊 实验 1: chunk_size 对检索质量的影响")
    print("   固定 overlap=100, 对比 size = 300 / 500 / 1000 / 2000")
    print("=" * 70)

    sizes = [300, 500, 1000, 2000]
    question = "Sony 降噪耳机的降噪效果怎么样？续航多久？"

    print(f"\n🔍 测试问题: {question}\n")

    for size in sizes:
        chunks = load_and_split(chunk_size=size, chunk_overlap=100)
        vs = build_vectorstore(chunks, f"size_{size}")
        results = test_retrieval(vs, question)
        print(f"\n--- chunk_size={size} | 共 {len(chunks)} 块 ---")

        for i, doc in enumerate(results):
            content = doc.page_content
            # 检查是否包含关键信息
            has_noise_reduction = "降噪" in content or "dB" in content
            has_battery = "续航" in content or "小时" in content
            markers = []

            if has_noise_reduction:
                markers.append("降噪")
            if has_battery:
                markers.append("续航")
            if not markers:
                markers.append("无关键信息")

            preview = content[:80].replace("\n", " ")
            print(f"  [{i + 1}] ({len(content)}字) {' '.join(markers)}")
            print(f"      {preview}...")

    print("\n💡 分析要点:")
    print("  - chunk_size=300: 块太小，一个完整的商品描述被切成多块，")
    print("    信息分散，可能检索到降噪信息但丢失续航信息")
    print("  - chunk_size=500: 适中，一个段落基本完整，检索精度高")
    print("  - chunk_size=1000: 块较大，一个块可能包含完整商品信息，")
    print("    但也可能混入无关内容，降低检索精度")
    print("  - chunk_size=2000: 块太大，接近全文检索，失去了RAG")
    print("    '精确定位'的优势")


# ============================================================
# 实验 2: 对比不同 chunk_overlap
# ============================================================


# ============================================================
# 实验 3: 不同类型问题的检索质量对比
# ============================================================


# ============================================================
# 运行所有实验
# ============================================================






