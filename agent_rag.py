import os
import torch
from typing import Annotated, List, Dict, TypedDict
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# --- 社区组件 ---
from langchain_community.chat_models import ChatOllama
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
# BM25 必须从 community 导入
from langchain_community.retrievers import BM25Retriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

# --- 核心组件 (检索器) ---
try:
    # 尝试从主包导入 (标准做法)
    from langchain.retrievers import EnsembleRetriever
    from langchain.retrievers import ContextualCompressionRetriever
    from langchain.retrievers.document_compressors import CrossEncoderReranker
except ImportError:
    # 备选方案：尝试从 community 导入 (兼容旧版/错版)
    print("⚠️ 正在尝试从 community 导入检索器...")
    from langchain_community.retrievers import EnsembleRetriever
    from langchain_community.retrievers import ContextualCompressionRetriever
    from langchain.retrievers.document_compressors import CrossEncoderReranker

# --- 图逻辑 ---
from langgraph.graph import END, StateGraph
# ================= 配置 =================
DB_PATH = "./vector_db"
EMBEDDING_MODEL = "moka-ai/m3e-base"
RERANK_MODEL = "BAAI/bge-reranker-base"
LLM_MODEL = "qwen2.5:7b"

print("⚙️  1. 初始化 Embeddings (CPU)...")
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={'device': 'cpu'}
)

print(f"🔗 2. 连接向量库 {DB_PATH}...")
if not os.path.exists(DB_PATH):
    print("❌ 错误：找不到向量库，请先运行 build.py！")
    exit()

vectordb = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)

print("⏳ 3. 构建混合检索 (BM25 + Vector)...")
try:
    # 获取所有文档用于构建 BM25 索引
    db_data = vectordb.get()
    all_docs = db_data["documents"]
    metadatas = db_data["metadatas"]

    if not all_docs:
        print("❌ 向量库是空的！请检查 data 文件夹并重新运行 build.py")
        exit()

    doc_objects = [Document(page_content=t, metadata=m) for t, m in zip(all_docs, metadatas)]

    # BM25 检索器
    bm25_retriever = BM25Retriever.from_documents(doc_objects)
    bm25_retriever.k = 5

    # 向量检索器
    vector_retriever = vectordb.as_retriever(search_kwargs={"k": 5})

    # 混合检索器
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.4, 0.6]
    )
except Exception as e:
    print(f"❌ 构建检索器失败: {e}")
    exit()

print("⚖️  4. 加载重排序模型 (Reranker)...")
try:
    rerank_model = HuggingFaceCrossEncoder(model_name=RERANK_MODEL, model_kwargs={'device': 'cpu'})
    compressor = CrossEncoderReranker(model=rerank_model, top_n=3)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=ensemble_retriever
    )
except Exception as e:
    print(f"⚠️  重排序模型加载失败 (可能是网络问题): {e}")
    print("⚠️  将降级使用普通混合检索...")
    compression_retriever = ensemble_retriever

print(f"🤖 5. 连接 Ollama ({LLM_MODEL})...")
llm = ChatOllama(model=LLM_MODEL, temperature=0)


# === 定义状态 (State) ===
class GraphState(TypedDict):
    question: str
    generation: str
    documents: List[str]
    retry_count: int


# === 定义节点 (Nodes) ===
def retrieve(state):
    print(f"\n🔍 [Step 1: 检索] 问题: {state['question']}")
    question = state["question"]
    docs = compression_retriever.invoke(question)
    doc_texts = [d.page_content for d in docs]
    print(f"   - 检索到 {len(doc_texts)} 条相关片段")
    return {"documents": doc_texts, "question": question}


def generate(state):
    print("✍️  [Step 2: 生成] 模型正在撰写回答...")
    question = state["question"]
    documents = state["documents"]

    context = "\n\n".join(documents) if documents else "无相关资料"

    prompt = f"""
    你是专业的雅思助教。请结合背景资料回答问题。如果资料不足，请用你的专业知识补充。

    [背景资料]:
    {context}

    [问题]:
    {question}
    """
    response = llm.invoke(prompt)
    return {"generation": response.content, "retry_count": state.get("retry_count", 0)}


def transform_query(state):
    print("🔄 [Step 3: 改写] 发现回答质量不高，正在尝试改写问题...")
    question = state["question"]
    retry_count = state.get("retry_count", 0) + 1

    prompt = f"""
    用户的问题可能检索不到结果。请将其改写为一个更好的雅思搜索查询。只输出新问题。
    原问题: {question}
    """
    better_question = llm.invoke(prompt).content.strip()
    print(f"   ✨ 新问题: {better_question}")
    return {"question": better_question, "retry_count": retry_count}


def hallucination_check(state):
    print("🧠 [Step 4: 反思] 检查是否有幻觉...")
    generation = state["generation"]
    retry_count = state["retry_count"]

    if retry_count > 1:  # 降低一点阈值，防止太慢
        print("   ⚠️ 重试次数耗尽，强制输出。")
        return "useful"

    # 这里的 Prompt 可以根据实际情况调整
    prompt = f"""
    Review the answer. Does it answer the question helpfuly? Answer 'yes' or 'no'.
    Question: {state['question']}
    Answer: {generation}
    """
    grade = llm.invoke(prompt).content.lower()

    if "yes" in grade:
        print("   ✅ 通过检查")
        return "useful"
    else:
        print("   ❌ 未通过，需要重试")
        return "not supported"


# === 构建图 (Graph) ===
workflow = StateGraph(GraphState)

workflow.add_node("retrieve", retrieve)
workflow.add_node("generate", generate)
workflow.add_node("transform_query", transform_query)

workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_conditional_edges(
    "generate",
    hallucination_check,
    {
        "useful": END,
        "not supported": "transform_query"
    }
)
workflow.add_edge("transform_query", "retrieve")

app = workflow.compile()

# === 运行 ===
if __name__ == "__main__":
    print("\n✅ 系统启动完毕！这是带有【反思能力】的雅思高级 Agent。")
    while True:
        try:
            q = input("\n🙋 请提问 (q退出): ")
            if q.lower() in ['q', 'exit']: break

            inputs = {"question": q, "retry_count": 0}

            # 使用 invoke 直接获取最终结果
            result = app.invoke(inputs)
            print(f"\n🤖 最终回答:\n{result['generation']}")
            print("-" * 50)

        except Exception as e:
            print(f"❌ 运行出错: {e}")