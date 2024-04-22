from load_model import load_model
import os


# clone 模型
PRETRAINED_MODEL_NAME_OR_PATH = "../models/internlm2-chat-1_8b"
# os.system(f'git clone https://code.openxlab.org.cn/OpenLMLab/internlm2-chat-1.8b {PRETRAINED_MODEL_NAME_OR_PATH}')
# os.system(f'cd {PRETRAINED_MODEL_NAME_OR_PATH} && git lfs pull')
ADAPTER_DIR = "../work_dirs/internlm2_chat_1_8b_qlora_car_e20/hf"
# 量化
LOAD_IN_8BIT= False
LOAD_IN_4BIT = False
tokenizer, model = load_model(PRETRAINED_MODEL_NAME_OR_PATH, ADAPTER_DIR, LOAD_IN_8BIT, LOAD_IN_4BIT)

SYSTEM_PROMPT = "你现在是评论总结小助手，负责总结用户对汽车的评论，要按照以下顺序输出评论的各个内容\n" + \
    "1、首先输出这个车的整体评价，要求言简意赅。\n" + \
    "2、然后输出这个车的好的评价，要求言简意赅。\n" + \
    "3、之后输出这个车的不好的评价，要求言简意赅。\n" + \
    "4、最后输出这个车的每个部分的评价，有多少提取多少。\n" + \
    "注意，只总结用户的输入的信息，不要自己编造用户没说的信息，以下是用户的评论，请进行总结\n"
print(SYSTEM_PROMPT)


while True:
    query = input("请输入提示: ")
    query = query.strip()
    if query == None or len(query) < 1:
        continue
    if query.lower() == "exit":
        break

    print("回答: ", end="")
    # https://huggingface.co/internlm/internlm2-chat-1_8b/blob/main/modeling_internlm2.py#L1185
    # stream_chat 返回的句子长度是逐渐边长的,length的作用是记录之前的输出长度,用来截断之前的输出
    length = 0
    history = []
    for response, history in model.stream_chat(
            tokenizer = tokenizer,
            query = query,
            history = history,
            max_new_tokens = 1024,
            do_sample = True,
            temperature = 0.1,
            top_p = 0.75,
            top_k = 40,
            meta_instruction = SYSTEM_PROMPT,
        ):
        if response is not None:
            print(response[length:], flush=True, end="")
            length = len(response)
    print("\n")
