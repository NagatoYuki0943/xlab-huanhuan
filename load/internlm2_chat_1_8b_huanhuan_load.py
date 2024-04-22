import torch
from transformers import GenerationConfig
from load_model import load_model
from typing import List, Tuple
import os


# clone 模型
PRETRAINED_MODEL_NAME_OR_PATH = '../models/internlm2-chat-1_8b'
# os.system(f'git clone https://code.openxlab.org.cn/OpenLMLab/internlm2-chat-1.8b {PRETRAINED_MODEL_NAME_OR_PATH}')
# os.system(f'cd {PRETRAINED_MODEL_NAME_OR_PATH} && git lfs pull')
ADAPTER_DIR = "../work_dirs/internlm2_chat_1_8b_qlora_huanhuan_e3_hf/checkpoint-699"
# 量化
LOAD_IN_8BIT= False
LOAD_IN_4BIT = False
tokenizer, model = load_model(PRETRAINED_MODEL_NAME_OR_PATH, ADAPTER_DIR, LOAD_IN_8BIT, LOAD_IN_4BIT)

SYSTEM_PROMPT = "现在你要扮演皇帝身边的女人--甄嬛"


def build_inputs(query: str, history: List[Tuple[str, str]] = [], meta_instruction="我是系统"):
    prompt = ""
    if meta_instruction:
        # <s> tokenizer会默认添加,不过这里使用手动添加的方式
        prompt += f"""<s><|im_start|>system\n{meta_instruction}<|im_end|>\n"""
    else:
        prompt += "<s>"
    for record in history:
        prompt += f"""<|im_start|>user\n{record[0]}<|im_end|>\n<|im_start|>assistant\n{record[1]}<|im_end|>\n"""
    prompt += f"""<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n"""
    return prompt


inputs = build_inputs("小主，敬事房传来消息，说皇上晚上去华妃那儿", history=[], meta_instruction=SYSTEM_PROMPT)
print(inputs)
inputs = tokenizer(inputs, add_special_tokens=False, return_tensors="pt").to(model.device)
print(inputs["input_ids"])
print(inputs["attention_mask"])

generation_config = GenerationConfig(
    max_length=50,
    do_sample=True,
    num_beams=2,
    temperature=0.7,
    top_k=50,
    top_p=0.9
)

model.eval()
with torch.inference_mode():
    outputs = model.generate(
        input_ids = inputs["input_ids"],
        attention_mask = inputs["attention_mask"],
        generation_config = generation_config,
    )
print(outputs)
result_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
print(result_text)
