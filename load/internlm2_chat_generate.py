import torch
from transformers import GenerationConfig
from load_tokenizer_processor_and_model import load_tokenizer_processor_and_model, TransformersConfig


# clone 模型
PRETRAINED_MODEL_NAME_OR_PATH = '../models/internlm2_5-1_8b-chat'
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


# https://huggingface.co/internlm/internlm2-chat-1_8b/blob/main/modeling_internlm2.py#L1136-L1146
def build_inputs(
    tokenizer,
    query: str,
    history: list[tuple[str, str]] | None = None,
    meta_instruction = ""
) -> tuple[str, list]:
    history = [] if history is None else list(history)
    if tokenizer.add_bos_token:
        prompt = ""
    else:
        prompt = tokenizer.bos_token
    if meta_instruction:
        prompt += f"""<|im_start|>system\n{meta_instruction}<|im_end|>\n"""
    for record in history:
        prompt += f"""<|im_start|>user\n{record[0]}<|im_end|>\n<|im_start|>assistant\n{record[1]}<|im_end|>\n"""
    prompt += f"""<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n"""
    return prompt, tokenizer([prompt], return_tensors="pt")


prompt, inputs = build_inputs(tokenizer, "给我讲一个猫和老鼠的小故事", history=[], meta_instruction=SYSTEM_PROMPT)
print(prompt)
inputs = inputs.to(model.device)
print("input_ids: ", inputs["input_ids"])
print("attention_mask: ", inputs["attention_mask"])


generation_config = GenerationConfig(
    max_new_tokens = 1024,
    do_sample = True,
    num_beams = 1,
    temperature = 0.8,
    top_k = 40,
    top_p = 0.8,
    eos_token_id = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids(["<|im_end|>"])[0]]
)

model.eval()
with torch.inference_mode():
    outputs = model.generate(
        input_ids = inputs["input_ids"],
        attention_mask = inputs["attention_mask"],
        generation_config = generation_config,
    )

print(outputs)
# 取出第一条数据
ids = outputs[0].cpu()[len(inputs["input_ids"][0]) :]
# decode 处理一维数据
response = tokenizer.decode(ids, skip_special_tokens=True)
print(response)
# batch_decode 处理二维数据
# print(tokenizer.batch_decode([ids], skip_special_tokens=True)[0])
