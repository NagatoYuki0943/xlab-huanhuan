import os
from infer_engine import InferEngine, TransformersConfig


# clone 模型
PRETRAINED_MODEL_NAME_OR_PATH = '../models/internlm2-chat-1_8b'
# os.system(f'git clone https://code.openxlab.org.cn/OpenLMLab/internlm2-chat-1.8b {PRETRAINED_MODEL_NAME_OR_PATH}')
# os.system(f'cd {PRETRAINED_MODEL_NAME_OR_PATH} && git lfs pull')
ADAPTER_PATH = "./internlm2_chat_1_8b_qlora_huatuo_e3/epoch_3_hf"
# 量化
LOAD_IN_8BIT= False
LOAD_IN_4BIT = False

SYSTEM_PROMPT = "你现在是一名医生，具备丰富的医学知识和临床经验。你擅长诊断和治疗各种疾病，能为病人提供专业的医疗建议。你有良好的沟通技巧，能与病人和他们的家人建立信任关系。请在这个角色下为我解答以下问题。"

TRANSFORMERS_CONFIG = TransformersConfig(
    pretrained_model_name_or_path = PRETRAINED_MODEL_NAME_OR_PATH,
    adapter_path = ADAPTER_PATH,
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
