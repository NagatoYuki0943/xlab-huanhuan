from load_model import load_model


# clone 模型
pretrained_model_name_or_path = '../models/internlm2-chat-1_8b'
# os.system(f'git clone https://code.openxlab.org.cn/OpenLMLab/internlm2-chat-1.8b {pretrained_model_name_or_path}')
# os.system(f'cd {pretrained_model_name_or_path} && git lfs pull')
adapter_dir = None

# 量化
load_in_8bit = False
load_in_4bit = False

tokenizer, model = load_model(pretrained_model_name_or_path, adapter_dir, load_in_8bit, load_in_4bit)

system_prompt = """You are an AI assistant whose name is InternLM (书生·浦语).
- InternLM (书生·浦语) is a conversational language model that is developed by Shanghai AI Laboratory (上海人工智能实验室). It is designed to be helpful, honest, and harmless.
- InternLM (书生·浦语) can understand and communicate fluently in the language chosen by the user such as English and 中文.
"""
print("system_prompt: ", system_prompt)


# history: [('What is the capital of France?', 'The capital of France is Paris.'), ('Thanks', 'You are Welcome')]
history = []
while True:
    query = input("请输入提示: ")
    query = query.replace(' ', '')
    if query == None or len(query) < 1:
        continue
    if query.lower() == "exit":
        break

    print("回答: ", end="")
    # https://huggingface.co/internlm/internlm2-chat-1_8b/blob/main/modeling_internlm2.py#L1149
    # chat 调用的 generate
    response, history = model.chat(
        tokenizer = tokenizer,
        query = query,
        history = history,
        streamer = None,
        max_new_tokens = 1024,
        do_sample = True,
        temperature = 0.8,
        top_p = 0.8,
        top_k = 40,
        meta_instruction = system_prompt,
    )
    print(response)
    # print("history:", history)
