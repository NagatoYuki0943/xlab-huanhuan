from infer_engine import InferEngine, ApiConfig
import os


"""
设置临时变量

linux:
    export API_KEY="your token"

powershell:
    $env:API_KEY = "your token"
"""
# URL = "http://localhost:8000/v1/chat/completions"
# URL = "https://api.moonshot.cn/v1/chat/completions"
URL = "https://api.siliconflow.cn/v1/chat/completions"

API_KEY = os.getenv("API_KEY", "I AM AN API_KEY")
print(f"API_KEY: {API_KEY}")

api_config = ApiConfig(
    base_url=URL,
    api_key=API_KEY,
    model="internlm/internlm2_5-7b-chat",
)


infer_engine = InferEngine(backend="api", api_config=api_config)
print(infer_engine.get_available_models())
# ['moonshot-v1-8k', 'moonshot-v1-32k', 'moonshot-v1-128k']


history = []  # [['What is the capital of France?', 'The capital of France is Paris.'], ['Thanks', 'You are Welcome']]
while True:
    query = input("请输入提示: ")
    query = query.strip()
    if query is None or len(query) < 1:
        continue
    if query.lower() == "exit":
        break

    print("回答: ", end="", flush=True)
    responses = []
    for response in infer_engine.chat_stream(
        query=query,
        history=history,
        max_new_tokens=1024,
        temperature=0.8,
        top_p=0.8,
        top_k=40,
    ):
        responses.append(response)
        print(response, flush=True, end="")
    _response = "".join(responses)
    history.append([query, _response])
    print("\n回答: ", _response)
