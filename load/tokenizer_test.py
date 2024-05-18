from load_model import load_model


# clone 模型
PRETRAINED_MODEL_NAME_OR_PATH = '../models/internlm2-chat-1_8b'
# os.system(f'git clone https://code.openxlab.org.cn/OpenLMLab/internlm2-chat-1.8b {PRETRAINED_MODEL_NAME_OR_PATH}')
# os.system(f'cd {PRETRAINED_MODEL_NAME_OR_PATH} && git lfs pull')
ADAPTER_PATH = None
# 量化
LOAD_IN_8BIT= False
LOAD_IN_4BIT = False
tokenizer, model = load_model(PRETRAINED_MODEL_NAME_OR_PATH, ADAPTER_PATH, LOAD_IN_8BIT, LOAD_IN_4BIT)


print(tokenizer.all_special_tokens) # ['<s>', '</s>', '<unk>', '<|im_start|>', '<|im_end|>', '<|action_start|>', '<|action_end|>', '<|interpreter|>', '<|plugin|>']
print(tokenizer.all_special_ids)    # [1, 2, 0, 92543, 92542, 92541, 92540, 92539, 92538]


token = tokenizer.decode(tokenizer.all_special_ids, skip_special_tokens=False)
print(f"{token = }")    # token = '<s></s><unk><|im_start|><|im_end|><|action_start|><|action_end|><|interpreter|><|plugin|>'

token = tokenizer.decode(tokenizer.all_special_ids, skip_special_tokens=True)
print(f"{token = }")    # token = ''
