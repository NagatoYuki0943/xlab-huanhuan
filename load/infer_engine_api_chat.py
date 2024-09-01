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
    api_key = os.getenv("API_KEY", "I AM AN API_KEY"),
    model = "moonshot-v1-8k",
)


infer_engine = InferEngine(backend='api', api_config=api_config)
print(infer_engine.get_available_models())
# ['moonshot-v1-8k', 'moonshot-v1-32k', 'moonshot-v1-128k']


history = [] # [['What is the capital of France?', 'The capital of France is Paris.'], ['Thanks', 'You are Welcome']]
while True:
    query = input("请输入提示: ")
    query = query.strip()
    if query == None or len(query) < 1:
        continue
    if query.lower() == "exit":
        break

    response = infer_engine.chat(
        query = query,
        history = history,
        max_new_tokens = 1024,
        temperature = 0.8,
        top_p = 0.8,
        top_k = 40,
        model = "moonshot-v1-8k",
    )
    history.append([query, response])
    print("回答:", response)
