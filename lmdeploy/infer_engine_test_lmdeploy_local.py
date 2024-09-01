from infer_engine import InferEngine, LmdeployConfig
import os


MODEL_PATH = '../models/internlm2_5-1_8b-chat'

SYSTEM_PROMPT = """You are an AI assistant whose name is InternLM (书生·浦语).
- InternLM (书生·浦语) is a conversational language model that is developed by Shanghai AI Laboratory (上海人工智能实验室). It is designed to be helpful, honest, and harmless.
- InternLM (书生·浦语) can understand and communicate fluently in the language chosen by the user such as English and 中文.
"""

LMDEPLOY_CONFIG = LmdeployConfig(
    model_path = MODEL_PATH,
    backend = 'turbomind',
    model_name = 'internlm2',
    model_format = 'hf',
    tp = 1,                         # Tensor Parallelism.
    max_batch_size = 128,
    cache_max_entry_count = 0.8,    # 调整 KV Cache 的占用比例为0.8
    quant_policy = 0,               # KV Cache 量化, 0 代表禁用, 4 代表 4bit 量化, 8 代表 8bit 量化
    system_prompt = SYSTEM_PROMPT,
    deploy_method = 'local'
)

# 载入模型
infer_engine = InferEngine(
    backend = 'lmdeploy', # transformers, lmdeploy, api
    lmdeploy_config = LMDEPLOY_CONFIG
)


history = [] # [['What is the capital of France?', 'The capital of France is Paris.'], ['Thanks', 'You are Welcome']]
query = "猫和老鼠的作者是谁?"

response, history = infer_engine.chat(
    query = query,
    history = history,
    max_new_tokens = 1024,
    temperature = 0.8,
    top_p = 0.8,
    top_k = 40,
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
)
print("回答:", response)
