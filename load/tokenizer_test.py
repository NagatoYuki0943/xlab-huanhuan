from load_tokenizer_processor_and_model import load_tokenizer_processor_and_model, TransformersConfig


# clone 模型
PRETRAINED_MODEL_NAME_OR_PATH = '../models/internlm2-chat-1_8b'
# os.system(f'git clone https://code.openxlab.org.cn/OpenLMLab/internlm2-chat-1.8b {PRETRAINED_MODEL_NAME_OR_PATH}')
# os.system(f'cd {PRETRAINED_MODEL_NAME_OR_PATH} && git lfs pull')
ADAPTER_PATH = None
# 量化
LOAD_IN_8BIT= False
LOAD_IN_4BIT = False

SYSTEM_PROMPT = """You are an AI assistant whose name is InternLM (书生·浦语).
- InternLM (书生·浦语) is a conversational language model that is developed by Shanghai AI Laboratory (上海人工智能实验室). It is designed to be helpful, honest, and harmless.
- InternLM (书生·浦语) can understand and communicate fluently in the language chosen by the user such as English and 中文.
"""

TRANSFORMERS_CONFIG = TransformersConfig(
    pretrained_model_name_or_path = PRETRAINED_MODEL_NAME_OR_PATH,
    adapter_path = ADAPTER_PATH,
    load_in_8bit = LOAD_IN_8BIT,
    load_in_4bit = LOAD_IN_4BIT,
    model_name = 'internlm2',
    system_prompt = SYSTEM_PROMPT
)

tokenizer, processor, model = load_tokenizer_processor_and_model(config=TRANSFORMERS_CONFIG)


print(tokenizer.all_special_tokens) # ['<s>', '</s>', '<unk>', '<|im_start|>', '<|im_end|>', '<|action_start|>', '<|action_end|>', '<|interpreter|>', '<|plugin|>']
print(tokenizer.all_special_ids)    # [1, 2, 0, 92543, 92542, 92541, 92540, 92539, 92538]


token = tokenizer.decode(tokenizer.all_special_ids, skip_special_tokens=False)
print(f"{token = }")    # token = '<s></s><unk><|im_start|><|im_end|><|action_start|><|action_end|><|interpreter|><|plugin|>'

token = tokenizer.decode(tokenizer.all_special_ids, skip_special_tokens=True)
print(f"{token = }")    # token = ''
