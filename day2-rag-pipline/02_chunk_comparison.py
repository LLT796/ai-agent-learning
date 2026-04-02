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
def experiment_chunk_overlap():
    print("\n" + "=" * 70)
    print("📊 实验 2: chunk_overlap 对检索质量的影响")
    print("   固定 size=500, 对比 overlap = 0 / 50 / 100 / 200")
    print("=" * 70)

    overlaps = [0, 50, 100, 200]
    # 这个问题的答案跨越两个段落（降噪性能在一段，佩戴感受在下一段）
    question = "Sony 耳机的降噪深度是多少？佩戴舒适吗？"
    print(f"\n🔍 测试问题: {question}")
    print(f"   (这个问题的答案横跨两个段落，测试 overlap 能否保留边界信息)\n")

    for overlap in overlaps:
        chunks = load_and_split(chunk_size=500, chunk_overlap=overlap)
        vs = build_vectorstore(chunks, f"overlap_{overlap}")
        results = test_retrieval(vs, question)

        print(f"\n--- chunk_overlap={overlap} | 共 {len(chunks)} 块 ---")
        for i, doc in enumerate(results):
            content = doc.page_content
            has_db = "dB" in content or "降噪" in content
            has_comfort = "佩戴" in content or "耳塞" in content or "重量" in content
            markers = []
            if has_db:
                markers.append("降噪参数")
            if has_comfort:
                markers.append("佩戴信息")
            if not markers:
                markers.append("无关键信息")

            preview = content[:80].replace("\n", " ")
            print(f"  [{i + 1}] ({len(content)}字) {' '.join(markers)}")
            print(f"      {preview}...")

    print("\n💡 分析要点:")
    print("  - overlap=0: 块之间完全不重叠。如果关键信息正好在分割点附近，")
    print("    会被切成两半，两个块各拿到一半，都不完整")
    print("  - overlap=50: 少量重叠，能缓解部分边界切割问题")
    print("  - overlap=100: 推荐值。重叠的20%内容提供了较好的边界保护，")
    print("    同时不会导致过多重复内容占用向量数据库空间")
    print("  - overlap=200: 重叠太多（40%），大量重复内容存入数据库，")
    print("    检索时可能返回高度相似的重复块，浪费 top-K 的名额")

# ============================================================
# 实验 3: 不同类型问题的检索质量对比
# ============================================================
def experiment_question_types():
    print("\n" + "=" * 70)
    print("📊 实验 3: 不同类型问题的检索质量")
    print("   使用推荐参数 (size=500, overlap=100)")
    print("=" * 70)

    chunks = load_and_split(chunk_size=500, chunk_overlap=100)
    vs = build_vectorstore(chunks, "question_types")
    retriever = vs.as_retriever(search_kwargs={"k": 3})

    questions = {
        "精确参数查询": "XM5耳机的电池续航是多少小时？",
        "对比型查询": "索尼和华为的耳机哪个降噪更好？",
        "模糊意图查询": "跑步用什么鞋好？",
        "政策查询": "退货运费谁出？",
        "超出范围": "苹果手机多少钱？",
    }

    for qtype, question in questions.items():
        results = retriever.invoke(question)
        print(f"\n--- {qtype}: '{question}' ---")
        for i, doc in enumerate(results):
            preview = doc.page_content[:100].replace("\n", " ")
            print(f"  [{i + 1}] ({len(doc.page_content)}字) {preview}...")


# ============================================================
# 运行所有实验
# ============================================================
if __name__ == "__main__":
    print("🧪 Day 2 - 分块策略对比实验")
    print("注意：本实验会多次调用 Embedding API，请确保账户有余额\n")

    experiment_chunk_size()
    experiment_chunk_overlap()
    experiment_question_types()

    print("\n" + "=" * 70)
    print("📝 实验结论")
    print("=" * 70)
    print("""
    推荐配置（中文电商场景）：
    ├── chunk_size: 400-600 字符
    ├── chunk_overlap: 80-120 字符（约 chunk_size 的 20%）
    └── separators: 优先按段落和句子分割

    核心权衡：
    ├── chunk 太小 → 信息碎片化，上下文丢失
    ├── chunk 太大 → 检索精度下降，混入无关内容
    ├── overlap 太小 → 边界信息被切断
    └── overlap 太大 → 重复内容过多，浪费检索名额

    最佳实践：
    ├── 根据你的文档特点调整，不存在万能参数
    ├── 用真实问题测试，不要只看 chunk 数量
    └── 优先保证"一个 chunk 包含一个完整的知识点"
    """)





