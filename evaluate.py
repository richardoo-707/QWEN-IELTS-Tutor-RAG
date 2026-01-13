import os
import json
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import HuggingFaceEmbeddings

# 导入你的 RAG 系统 (需要稍微修改 agent_rag.py 以便调用，或者在这里重新定义一遍简单的链)
# 为了演示方便，我们这里直接实例化一个简单的 RAG 链，复用你之前的逻辑
from langchain_community.vectorstores import Chroma
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# ================= 配置 =================
DB_PATH = "./vector_db"
EMBEDDING_MODEL = "moka-ai/m3e-base"
LLM_MODEL = "qwen2.5:7b"

print("⚙️  1. 初始化评估环境...")

# 1. 准备 Embeddings
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

# 2. 准备 LLM (作为 RAG 的生成器)
generator_llm = ChatOllama(model=LLM_MODEL, temperature=0)

# 3. 准备 LLM (作为 Ragas 的裁判/Critic)
# Ragas 通常推荐用 GPT-4 当裁判，但为了纯本地，我们强制用 Qwen 当裁判
critic_llm = ChatOllama(model=LLM_MODEL, temperature=0)

# 4. 连接向量库
vectordb = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
retriever = vectordb.as_retriever(search_kwargs={"k": 3})

# 5. 定义简单的 RAG 链 (用于生成测试结果)
template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)


def format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])


rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | generator_llm
        | StrOutputParser()
)

# ================= 开始评估 =================
print("📂 2. 加载测试集...")
with open("test_data.json", "r", encoding="utf-8") as f:
    test_data = json.load(f)

questions = [item["question"] for item in test_data]
ground_truths = [[item["ground_truth"]] for item in test_data]  # Ragas 需要二维列表

answers = []
contexts = []

print("🤖 3. 正在生成回答 (这可能需要几分钟)...")
for i, q in enumerate(questions):
    print(f"   [{i + 1}/{len(questions)}] 处理问题: {q}")

    # 1. 获取回答
    ans = rag_chain.invoke(q)
    answers.append(ans)

    # 2. 获取检索到的上下文 (为了计算 Context Metrics)
    docs = retriever.invoke(q)
    ctx = [d.page_content for d in docs]
    contexts.append(ctx)

# 构建 Ragas 数据集
data_dict = {
    "question": questions,
    "answer": answers,
    "contexts": contexts,
    "ground_truth": ground_truths
}
dataset = Dataset.from_dict(data_dict)

print("⚖️  4. 开始打分 (使用 Ragas)...")
# 这里的 metrics 就是你想展示的核心指标
results = evaluate(
    dataset=dataset,
    metrics=[
        faithfulness,  # 忠实度：回答是否由上下文支撑 (防幻觉)
        answer_relevancy,  # 相关性：回答是否切题
        # context_precision, # 上下文精确度：检索到的内容是否相关 (可选)
    ],
    llm=critic_llm,  # 这里的 llm 是用来当裁判的
    embeddings=embeddings  # 这里的 embeddings 用来计算相似度
)

# ================= 输出结果 =================
print("\n📊 评估报告:")
df = results.to_pandas()
print(df[["question", "faithfulness", "answer_relevancy"]])

print("\n🏆 平均分:")
print(results)

# 保存为 CSV 方便放到 GitHub
df.to_csv("evaluation_results.csv", index=False)
print("✅ 结果已保存至 evaluation_results.csv")