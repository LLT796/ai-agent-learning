"""
Day 2 - 完整 RAG Pipeline（通义千问版）
目标：跑通 文档加载 → 分块 → Embedding → 向量存储 → 检索 → 生成

数据流：
  原始文档 → 文本块 → 向量 → ChromaDB → 检索相关块 → LLM 生成答案

运行方式：uv run python day2-rag-pipeline/01_rag_pipeline.py
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / "day1-basic-agent" / ".env")

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_chroma import Chroma

# ============================================================
# Step 1: 配置 LLM 和 Embedding 模型
# ============================================================
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# LLM: 用于最后根据检索到的内容生成答案
llm = ChatOpenAI(
    api_key=DASHSCOPE_API_KEY,
    base_url=DASHSCOPE_BASE_URL,
    model="qwen-plus",
    temperature=0,
)

# Embedding 模型: 把文本转成向量（数字数组）
# 通义千问的 Embedding 模型也走 OpenAI 兼容接口
embeddings = OpenAIEmbeddings(
    api_key=DASHSCOPE_API_KEY,
    base_url=DASHSCOPE_BASE_URL,
    model="text-embedding-v3",  # 通义千问的 Embedding 模型
)

# ============================================================
# Step 2: 加载文档
# ============================================================
def load_documents(doc_path: str):
    """加载文档。支持 .txt 和 .pdf 格式"""
    path = Path(doc_path)

    if path.suffix == ".pdf":
        loader = PyPDFLoader(str(path))
        print(f"使用 pyPDFLoader 加载：{path.name}")
    elif path.suffix == ".txt":
        loader = TextLoader(str(path), encodings="utf-8")
        print(f"使用 TextLoader 加载：{path.name}")
    else:
        raise ValueError(f"不支持的文件格式：{path.suffix}")

    docs = loader.load()
    total_chars = sum(len(d.page_content) for d in docs)
    print(f"    加载完成：{len(docs)} 个文档，共 {total_chars} 字符")
    return docs

# ============================================================
# Step 3: 文本分块（Chunking）
# ============================================================
def split_documents(docs, chunk_size=500, chunk_overlap=100):
    """把长文档切成小块
    为什么要分块？
        1. LLM 有上下文长度限制，不能把整个文档塞进去
        2. 检索时需要找到"最相关的段落"，而不是"最相关的整篇文档"
        3. 小块的向量表示更精确，检索质量更高
    关键参数：
        - chunk_size: 每个块的最大字符数
        - chunk_overlap: 相邻块之间的重叠字符数（防止关键信息被切断）
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        # 分割优先级：先按段落分，段落太长就按句子分，再按词分
        separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""],
        length_function=len,
    )

    chunks = splitter.split_documents(docs)
    print(f"  分块完成: {len(chunks)} 个块 (chunk_size={chunk_size}, overlap={chunk_overlap})")

    # 打印前 3 个块的预览
    for i, chunk in enumerate(chunks[:3]):
        preview = chunk.page_content[:80].replace("\n", " ")
        print(f"   块 {i}: [{len(chunk.page_content)} 字符] {preview}...")

    return chunks

# ============================================================
# Step 4: 向量化 + 存入 ChromaDB
# ============================================================
def create_vector_store(chunks, collection_name="product_knowledge"):
    """把文本块转成向量并存入 ChromaDB
    过程：
        1. 对每个 chunk 调用 Embedding 模型，得到一个向量（如 1024 维的浮点数数组）
        2. 把向量和原始文本一起存入 ChromaDB
        3. ChromaDB 会建立向量索引，加速后续的相似度搜索
    """
    # persist_directory: 向量数据库的存储位置（持久化到磁盘）
    db_path = str(Path(__file__).parent / "chroma_db")

    print(f" 正在向量化 {len(chunks)} 个文本块并存入 ChromaDB...")
    print(f"   存储路径: {db_path}")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=db_path,
        collection_name=collection_name,
    )
    print(f"   向量数据库创建完成，共 {vectorstore._collection.count()} 条记录")
    return vectorstore

# ============================================================
# Step 5: 构建检索器（Retriever）
# ============================================================
def create_retriever(vectorstore, k=3):
    """创建检索器
    检索过程：
        1. 把用户的问题也转成向量（用同一个 Embedding 模型）
        2. 在 ChromaDB 中找到和问题向量最相似的 K 个文本块
        3. 返回这 K 个块的原始文本
    相似度计算：使用余弦相似度（Cosine Similarity）
        - 两个向量方向越接近，相似度越高（值越接近 1）
        - "蓝牙耳机降噪效果" 的向量会和 "降噪性能...35dB" 的块向量很接近
    """
    retriever = vectorstore.as_retriever(
        search_type="similarity",   # 基于余弦相似度
        search_kwargs={"k", k},     # 返回最相似的 K 个块
    )
    print(f"    检索器创建完成 (top-{k}")
    return retriever

# ============================================================
# Step 6: 构建 RAG Chain（检索 + 生成）
# ============================================================

# RAG 的 Prompt 模板: 把检索到的上下文和用户问题组合起来
RAG_PROMPT = ChatPromptTemplate.from_template("""你是淘天自营平台的智能客服助手。
请根据以下商品知识库中的信息来回答用户的问题。
 
要求：
1. 只基于提供的上下文信息回答，不要编造数据
2. 如果上下文中没有相关信息，诚实地说"根据现有资料无法回答这个问题"
3. 回答要具体，包含价格、参数等关键数据
4. 如果涉及多个商品，可以做对比
 
===== 知识库相关内容 =====
{context}
===== 知识库结束 =====
 
用户问题：{question}
 
请回答：""")

def rag_query(question: str, retriever, verbose=True):
    """执行一次完整的 RAG 查询
    完整流程：
        1. 用 retriever 检索相关文本块
        2. 把检索到的块拼接为上下文
        3. 把上下文 + 用户问题填入 Prompt
        4. 调用 LLM 生成答案
    """
    retrieved_docs = retriever.invoke(question)

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"🧑 问题: {question}")
        print(f"{'=' * 60}")
        print(f"\n📚 检索到 {len(retrieved_docs)} 个相关文本块:")
        for i, doc in enumerate(retrieved_docs):
            preview = doc.page_content[:100].replace("\n", " ")
            print(f"    [{i+1}] {preview}...")\

    # 拼接上下文
    context = "\n\n---\n\n".join(doc.page_content for doc in retrieved_docs)

    # 生成答案
    chain = RAG_PROMPT | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": question})

    if verbose:
        print(f"\n🤖 回答:\n{answer}")
        print(f"\n{'=' * 60}\n")
    return answer

# ============================================================
# Step 7: 主流程 —— 把所有步骤串起来
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("🚀 Day 2 - RAG Pipeline 完整演示")
    print("=" * 60)

    # 加载文档
    doc_path = Path(__file__).parent / "docs" / "product_knowledge.txt"
    docs = load_documents(str(doc_path))

    # 分块
    chunks = split_documents(docs, chunk_size=500, chunk_overlap=100)

    # 向量化 + 存储
    vectorstore = create_vector_store(chunks)

    # --- 在线查询阶段 ---
    print("\n 阶段二：在线查询（检索 → 生成）\n")

    # 创建检索器
    retriever = create_retriever(vectorstore, k=3)

    # 测试问题
    test_questions = [
        # 精确问题：答案在某个特定段落里
        "Sony WF-1000XM5 的降噪效果怎么样？地铁里能用吗？",

        # 对比问题：需要检索多个商品的信息
        "耐克飞马和阿迪Ultraboost哪个更适合日常跑步？",

        # 政策问题：答案在退换货政策段落
        "买了商品降价了怎么办？可以退差价吗？",

        # 模糊问题：需要理解用户意图
        "预算1500以内想买个好耳机，有什么推荐？",

        # 超出知识库范围的问题
        "你们卖笔记本电脑吗？MacBook Pro 多少钱？",
    ]

    for q in test_questions:
        rag_query(q, retriever)









