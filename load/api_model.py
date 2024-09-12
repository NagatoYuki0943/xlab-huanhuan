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

lmdeploy
    lmdeploy serve api_server \
        ./models/internlm2-chat-7b \
        --backend turbomind \
        --model-format hf \
        --tp 1 \
        --cache-max-entry-count 0.8 \
        --quant-policy 0 \
        --model-name internlm2_7b_chat \
        --server-name 0.0.0.0 \
        --server-port 23333
"""
api_key = os.getenv("API_KEY", "I AM AN API_KEY")


client = OpenAI(
    api_key=api_key,  # 此处传token，不带Bearer
    # base_url = "https://internlm-chat.intern-ai.org.cn/puyu/api/v1/",
    # base_url = "https://api.moonshot.cn/v1",
    base_url="http://0.0.0.0:23333/v1",  # lmdeploy 本地端口
)


models = client.models.list()
for model in models:
    print(model)
    print(model.id)
# Model(id='internlm2-latest', created=1715253376, object='model', owned_by='SH-AILab')
# internlm2-latest
