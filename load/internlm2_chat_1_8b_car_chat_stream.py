import os
from infer_engine import InferEngine, TransformersConfig


# clone 模型
PRETRAINED_MODEL_NAME_OR_PATH = '../models/internlm2-chat-1_8b'
# os.system(f'git clone https://code.openxlab.org.cn/OpenLMLab/internlm2-chat-1.8b {PRETRAINED_MODEL_NAME_OR_PATH}')
# os.system(f'cd {PRETRAINED_MODEL_NAME_OR_PATH} && git lfs pull')
ADAPTER_DIR = "../work_dirs/internlm2_chat_1_8b_qlora_car_e20/hf"
# 量化
LOAD_IN_8BIT= False
LOAD_IN_4BIT = False

SYSTEM_PROMPT = "你现在是评论总结小助手，负责总结用户对汽车的评论，要按照以下顺序输出评论的各个内容\n" + \
    "1、首先输出这个车的整体评价，要求言简意赅。\n" + \
    "2、然后输出这个车的好的评价，要求言简意赅。\n" + \
    "3、之后输出这个车的不好的评价，要求言简意赅。\n" + \
    "4、最后输出这个车的每个部分的评价，有多少提取多少。\n" + \
    "注意，只总结用户的输入的信息，不要自己编造用户没说的信息，以下是用户的评论，请进行总结\n"

TRANSFORMERS_CONFIG = TransformersConfig(
    pretrained_model_name_or_path=PRETRAINED_MODEL_NAME_OR_PATH,
    adapter_dir=ADAPTER_DIR,
    load_in_8bit=LOAD_IN_8BIT,
    load_in_4bit=LOAD_IN_4BIT,
    system_prompt=SYSTEM_PROMPT
)

# 载入模型
infer_engine = InferEngine(
    backend='transformers', # transformers, lmdeploy
    transformers_config=TRANSFORMERS_CONFIG,
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
