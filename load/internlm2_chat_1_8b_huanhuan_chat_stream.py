from load_model import load_model


# clone 模型
pretrained_model_name_or_path = '../models/internlm2-chat-1_8b'
# os.system(f'git clone https://code.openxlab.org.cn/OpenLMLab/internlm2-chat-1.8b {pretrained_model_name_or_path}')
# os.system(f'cd {pretrained_model_name_or_path} && git lfs pull')
adapter_dir = "../work_dirs/internlm2_chat_1_8b_qlora_huanhuan_e3_hf/checkpoint-699"

# 量化
load_in_8bit = False
load_in_4bit = False

tokenizer, model = load_model(pretrained_model_name_or_path, adapter_dir, load_in_8bit, load_in_4bit)

system_prompt = "现在你要扮演皇帝身边的女人--甄嬛"
print(system_prompt)


history = []
while True:
    query = input("请输入提示: ")
    query = query.replace(' ', '')
    if query == None or len(query) < 1:
        continue
    if query.lower() == "exit":
        break

    print("回答: ", end="")
    # https://huggingface.co/internlm/internlm2-chat-1_8b/blob/main/modeling_internlm2.py#L1185
    # stream_chat 返回的句子长度是逐渐边长的,length的作用是记录之前的输出长度,用来截断之前的输出
    length = 0
    for response, history in model.stream_chat(
            tokenizer = tokenizer,
            query = query,
            history = history,
            max_new_tokens = 1024,
            do_sample = True,
            temperature = 0.8,
            top_p = 0.8,
            top_k = 40,
            meta_instruction = system_prompt,
        ):
        if response is not None:
            print(response[length:], flush=True, end="")
            length = len(response)
    print("\n")
