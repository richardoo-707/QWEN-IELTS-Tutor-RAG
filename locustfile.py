from locust import HttpUser, task, between
import random

# 准备一些雅思相关的问题，模拟真实场景
TEST_QUESTIONS = [
    "How to improve my listening skills?",
    "Tell me about IELTS writing task 2",
    "What is the difference between academic and general training?",
    "Give me some tips for speaking test part 2",
    "List common vocabulary for environment topic",
    "How is the reading section scored?"
]


class IELTSUser(HttpUser):
    # 模拟用户思考时间：每发完一个请求，休息 5 到 10 秒
    # 因为 RAG 推理比较慢，设置太短会让队列积压太严重
    wait_time = between(5, 10)

    @task
    def ask_agent(self):
        # 1. 随机选一个问题
        question = random.choice(TEST_QUESTIONS)

        payload = {
            "question": question
        }

        # 2. 发送 POST 请求到 /chat 接口
        # catch_response=True 允许我们自定义什么是成功，什么是失败
        with self.client.post("/chat", json=payload, catch_response=True) as response:

            if response.status_code == 200:
                data = response.json()
                latency = data.get("latency", 0)

                # 设定一个 Infra 指标：如果回答时间超过 30秒，就算“严重超时”
                if latency > 30.0:
                    response.failure(f"Too slow! Took {latency}s")
                else:
                    response.success()
            else:
                # 如果返回 500 或 422，标记为失败
                response.failure(f"Status code: {response.status_code}")
