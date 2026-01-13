from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import PromptTemplate

# ================= 配置区域 =================
DB_PATH = "./vector_db"
EMBEDDING_MODEL = "moka-ai/m3e-base"
LLM_MODEL = "qwen2.5:7b"


def rag_chat_system():
    print("⏳ 1. 连接向量数据库...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectordb = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)

    print(f"🔗 2. 连接本地 Ollama ({LLM_MODEL})...")
    llm = ChatOllama(model=LLM_MODEL, temperature=0.7)  # 温度调高一点(0.7)，让它更灵活

    print("✅ 雅思全能助手已就绪！(既懂知识库，又懂通用知识)")

    while True:
        query = input("\n🙋 请提问 (exit退出): ")
        if query.lower() in ["exit", "quit", "q"]:
            break

        # --- 检索环节 ---
        # 找3个相关片段
        docs = vectordb.similarity_search(query, k=3)

        # 即使没找到相关文档，也不要让 context 为空，给它一个占位符
        if not docs:
            context = "（未检索到相关资料）"
        else:
            context = "\n".join([f"- {doc.page_content}" for doc in docs])

        # --- 生成环节 (核心修改：混合模式 Prompt) ---
        # 这里的 Prompt 赋予了模型“自主裁决权”
        prompt_template = f"""
        你是一位专业的雅思助教和英语专家。

        我为你提供了一些【参考资料】，请按照以下策略回答用户的问题：

        1. **优先参考**：如果【参考资料】与用户的问题**直接相关**，请基于资料回答，确保准确性。
        2. **自主回答**：如果【参考资料】是无关的、乱码、或者完全没提到答案，请**忽略资料**，直接使用你自己的专业知识来回答。
        3. **不要死板**：不要在回答中说“根据参考资料...”或者“资料里没提到...”，直接给出最佳答案即可。

        【参考资料】：
        {context}

        【用户问题】：
        {query}

        【你的回答】：
        """

        print("🤖 思考中...")
        response = llm.invoke(prompt_template)

        print(f"\n{response.content}")


if __name__ == "__main__":
    rag_chat_system()