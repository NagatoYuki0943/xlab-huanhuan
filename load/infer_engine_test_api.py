from infer_engine import InferEngine, ApiConfig
import os


"""
设置临时变量

linux:
    export API_KEY="your token"

powershell:
    $env:API_KEY = "your token"
"""
api_config = ApiConfig(
    base_url = "https://api.moonshot.cn/v1",
    api_key = os.getenv("API_KEY", "sk-SS08e69T7J6dvUjlD8U4tJDNRlfUKIk841H7PHUmUzxdgFlZ"),
    model = "moonshot-v1-8k",
)


infer_engine = InferEngine(backend='api', api_config=api_config)
print(infer_engine.get_available_models())
# ['moonshot-v1-8k', 'moonshot-v1-32k', 'moonshot-v1-128k']


history = [] # [['What is the capital of France?', 'The capital of France is Paris.'], ['Thanks', 'You are Welcome']]
query = "猫和老鼠的作者是谁?"

response, history = infer_engine.chat(
    query = query,
    history = history,
    max_new_tokens = 1024,
    temperature = 0.8,
    top_p = 0.8,
    top_k = 40,
    model = "moonshot-v1-8k",
)
print("回答:", response)


query = [
    {'role': 'user', 'content': query},
    {'role': 'assistant', 'content': response},
    {'role': 'user', 'content': "讲一个猫和老鼠的小故事"},
]

response, history = infer_engine.chat(
    query = query,
    history = None,
    max_new_tokens = 1024,
    temperature = 0.8,
    top_p = 0.8,
    top_k = 40,
    model = "moonshot-v1-8k",
)
print("回答:", response)
print("*" * 100)


query = [{'role': 'user', 'content': "猫和老鼠的作者是谁?"}]

response, history = infer_engine.chat(
    query = query,
    history = None,
    max_new_tokens = 1024,
    temperature = 0.8,
    top_p = 0.8,
    top_k = 40,
    model = "moonshot-v1-8k",
)
print("回答:", response)

query = "讲一个猫和老鼠的小故事"

response, history = infer_engine.chat(
    query = query,
    history = history,
    max_new_tokens = 1024,
    temperature = 0.8,
    top_p = 0.8,
    top_k = 40,
    model = "moonshot-v1-8k",
)
print("回答:", response)
