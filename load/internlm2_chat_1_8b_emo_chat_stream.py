import os
from infer_engine import InferEngine, TransformersConfig


# clone 模型
PRETRAINED_MODEL_NAME_OR_PATH = '../models/internlm2-chat-1_8b'
# os.system(f'git clone https://code.openxlab.org.cn/OpenLMLab/internlm2-chat-1.8b {PRETRAINED_MODEL_NAME_OR_PATH}')
# os.system(f'cd {PRETRAINED_MODEL_NAME_OR_PATH} && git lfs pull')
ADAPTER_DIR = "../work_dirs/internlm2_chat_1_8b_qlora_emo_e3_hf/checkpoint-3000"
# 量化
LOAD_IN_8BIT= False
LOAD_IN_4BIT = False

SYSTEM_PROMPT = "现在你是一个心理专家，我有一些心理问题，请你用专业的知识帮我解决。"

TRANSFORMERS_CONFIG = TransformersConfig(
    pretrained_model_name_or_path = PRETRAINED_MODEL_NAME_OR_PATH,
    adapter_dir = ADAPTER_DIR,
    load_in_8bit = LOAD_IN_8BIT,
    load_in_4bit = LOAD_IN_4BIT,
    system_prompt = SYSTEM_PROMPT
)

# 载入模型
infer_engine = InferEngine(
    backend = 'transformers', # transformers, lmdeploy
    transformers_config = TRANSFORMERS_CONFIG,
)

history = [] # [['What is the capital of France?', 'The capital of France is Paris.'], ['Thanks', 'You are Welcome']]
while True:
    query = input("请输入提示: ")
    query = query.strip()
    if query == None or len(query) < 1:
        continue
    if query.lower() == "exit":
        break

    print(f"回答: ", end="", flush=True)
    length = 0
    for response, history in infer_engine.chat_stream(
        query = query,
        history = history,
        max_new_tokens = 1024,
        top_p = 0.8,
        top_k = 40,
        temperature = 0.8,
    ):
        print(response[length:], flush=True, end="")
        length = len(response)
    print("\n")
