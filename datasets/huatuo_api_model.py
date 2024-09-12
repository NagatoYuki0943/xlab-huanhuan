# https://internlm.intern-ai.org.cn/api/document
# https://platform.moonshot.cn/docs/api/chat
import os
from openai import OpenAI


"""
设置临时变量

linux:
    export API_KEY="your token"

powershell:
    $env:API_KEY = "your token"
"""
api_key = os.getenv("API_KEY", "I AM AN API_KEY")
print(f"{ api_key = }")


client = OpenAI(
    api_key=api_key,  # 此处传token，不带Bearer
    # base_url = "https://internlm-chat.intern-ai.org.cn/puyu/api/v1/",
    # base_url = "https://api.moonshot.cn/v1",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)


models = client.models.list()
for model in models:
    print(model.id)
