import time
import uvicorn
import redis
import json
import hashlib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from agent_rag import app as rag_agent

# 1. 连接 Redis (这就好比雇了一个记性很好的秘书)
# host='localhost' 表示 Redis 就在本机
# port=6379 是 Redis 的默认端口
redis_client = redis.Redis(host='localhost', port=6380, db=0, decode_responses=True)

app = FastAPI(title="Cached RAG Service")


class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    answer: str
    latency: float
    source: str  # 新增字段：告诉你答案是 "Model" 算的，还是 "Cache" 查的


def get_cache_key(text):
    """把问题变成一个唯一的指纹(MD5)，方便存储"""
    return hashlib.md5(text.encode()).hexdigest()


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    start_time = time.time()

    # --- 核心 Infra 逻辑：先查缓存 ---
    cache_key = get_cache_key(request.question)
    cached_result = redis_client.get(cache_key)

    if cached_result:
        # 命中缓存！直接返回，不打扰显卡
        end_time = time.time()
        print(f"⚡ 命中缓存: {request.question[:10]}...")
        return {
            "answer": cached_result,
            "latency": round(end_time - start_time, 3),
            "source": "Redis Cache ⚡"  # 标记来源
        }

    # --- 缓存没命中，只能辛苦显卡了 ---
    try:
        print(f"🐢 显卡计算中: {request.question[:10]}...")
        inputs = {"question": request.question, "retry_count": 0}
        result = rag_agent.invoke(inputs)
        final_answer = result.get("generation", "Error")

        # --- 算完后，马上记到小本本(Redis)上 ---
        # ex=3600 表示这条记录只存 1 小时，过期自动删除
        redis_client.set(cache_key, final_answer, ex=3600)

        end_time = time.time()
        return {
            "answer": final_answer,
            "latency": round(end_time - start_time, 3),
            "source": "LLM Inference 🐢"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)