import os
from infer_engine import InferEngine, TransformersConfig


PRETRAINED_MODEL_NAME_OR_PATH = "../models/internlm2_5-1_8b-chat"
ADAPTER_PATH = None
# 量化
LOAD_IN_8BIT = False
LOAD_IN_4BIT = False

SYSTEM_PROMPT = """You are an AI assistant whose name is InternLM (书生·浦语).
- InternLM (书生·浦语) is a conversational language model that is developed by Shanghai AI Laboratory (上海人工智能实验室). It is designed to be helpful, honest, and harmless.
- InternLM (书生·浦语) can understand and communicate fluently in the language chosen by the user such as English and 中文.
"""

TRANSFORMERS_CONFIG = TransformersConfig(
    pretrained_model_name_or_path=PRETRAINED_MODEL_NAME_OR_PATH,
    adapter_path=ADAPTER_PATH,
    load_in_8bit=LOAD_IN_8BIT,
    load_in_4bit=LOAD_IN_4BIT,
    model_name="internlm2",
    system_prompt=SYSTEM_PROMPT,
)

# 载入模型
infer_engine = InferEngine(
    backend="transformers",  # transformers, lmdeploy
    transformers_config=TRANSFORMERS_CONFIG,
)


# history = [] # [['What is the capital of France?', 'The capital of France is Paris.'], ['Thanks', 'You are Welcome']]
# query = "猫和老鼠的作者是谁?"

# response = infer_engine.chat(
#     query = query,
#     history = history,
#     max_new_tokens = 1024,
#     temperature = 0.8,
#     top_p = 0.8,
#     top_k = 40,
# )
# print("回答:", response)


# query = [
#     {'role': 'user', 'content': query},
#     {'role': 'assistant', 'content': response},
#     {'role': 'user', 'content': "讲一个猫和老鼠的小故事"},
# ]

# response = infer_engine.chat(
#     query = query,
#     history = None,
#     max_new_tokens = 1024,
#     temperature = 0.8,
#     top_p = 0.8,
#     top_k = 40,
# )
# print("回答:", response)
# print("*" * 100)


query = [{"role": "user", "content": "猫和老鼠的作者是谁?"}]

response = infer_engine.chat(
    query=query,
    history=None,
    max_new_tokens=1024,
    temperature=0.8,
    top_p=0.8,
    top_k=40,
)

history = [[query[0]["content"], response]]
print("回答:", response)


query = "讲一个猫和老鼠的小故事"

response = infer_engine.chat(
    query=query,
    history=history,
    max_new_tokens=1024,
    temperature=0.8,
    top_p=0.8,
    top_k=40,
)
print("回答:", response)
