import os
from infer_engine import InferEngine, LmdeployConfig



# clone 模型
MODEL_PATH = '../models/internlm2-chat-1_8b'
# os.system(f'git clone https://code.openxlab.org.cn/OpenLMLab/internlm2-chat-1.8b {MODEL_PATH}')
# os.system(f'cd {MODEL_PATH} && git lfs pull')

SYSTEM_PROMPT = """You are an AI assistant whose name is InternLM (书生·浦语).
- InternLM (书生·浦语) is a conversational language model that is developed by Shanghai AI Laboratory (上海人工智能实验室). It is designed to be helpful, honest, and harmless.
- InternLM (书生·浦语) can understand and communicate fluently in the language chosen by the user such as English and 中文.
"""

LMDEPLOY_CONFIG = LmdeployConfig(
    model_path = MODEL_PATH,
    backend = 'turbomind',
    model_format = 'hf',
    model_name = 'internlm2',
    custom_model_name = 'internlm2_chat_1_8b',
    system_prompt = SYSTEM_PROMPT
)

# 载入模型
infer_engine = InferEngine(
    backend = 'lmdeploy', # transformers, lmdeploy
    lmdeploy_config = LMDEPLOY_CONFIG
)

history = [] # [['What is the capital of France?', 'The capital of France is Paris.'], ['Thanks', 'You are Welcome']]
while True:
    query = input("请输入提示: ")
    query = query.strip()
    if query == None or len(query) < 1:
        continue
    if query.lower() == "exit":
        break

    response, history = infer_engine.chat(
        query = query,
        history = history,
        max_new_tokens = 1024,
        top_p = 0.8,
        top_k = 40,
        temperature = 0.8,
    )
    print("回答:", response)
