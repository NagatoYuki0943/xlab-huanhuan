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

history = []  # [['What is the capital of France?', 'The capital of France is Paris.'], ['Thanks', 'You are Welcome']]
while True:
    query = input("请输入提示: ")
    query = query.strip()
    if query is None or len(query) < 1:
        continue
    if query.lower() == "exit":
        break

    response = infer_engine.chat(
        query=query,
        history=history,
        max_new_tokens=1024,
        temperature=0.8,
        top_p=0.8,
        top_k=40,
    )
    history.append([query, response])
    print("回答:", response)
