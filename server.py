import time
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# 1. 导入你修好的 RAG Agent
# 这行代码执行时，会加载模型 (Ollama, Embeddings)，可能需要几秒钟
print("正在初始化 RAG Agent，请稍候...")
from agent_rag import app as rag_agent

print("RAG Agent 初始化完成！")

# 2. 初始化 FastAPI 应用
app = FastAPI(
    title="IELTS Tutor RAG Service",
    description="基于 LangGraph 的雅思助教 RAG 接口",
    version="1.0"
)


# 3. 定义请求和响应的数据格式 (Schema)
class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    answer: str
    latency: float


# 4. 核心接口
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    start_time = time.time()

    try:
        # 构造 LangGraph 需要的初始状态 inputs
        # 注意：你的 agent_rag.py 里定义的状态是 {"question": ..., "retry_count": ...}
        inputs = {"question": request.question, "retry_count": 0}

        # 调用 Agent (invoke 是同步阻塞的，这里直接用即可)
        # 你的 agent_rag.py 最后返回的是一个字典 (State)
        result = rag_agent.invoke(inputs)

        # 计算耗时
        end_time = time.time()
        process_time = round(end_time - start_time, 3)

        # 从结果中提取生成的回答 (你的代码里 key 叫 'generation')
        final_answer = result.get("generation", "Error: No generation found")

        return {
            "answer": final_answer,
            "latency": process_time
        }

    except Exception as e:
        print(f"❌ 内部错误: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    # 启动服务，监听 8000 端口
    uvicorn.run(app, host="0.0.0.0", port=8000)