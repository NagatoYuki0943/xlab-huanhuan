import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, BitsAndBytesConfig
from peft import PeftModel
import torch
from typing import List, Tuple


print("torch version: ", torch.__version__)
print("transformers version: ", transformers.__version__)

model_dir = "./models/internlm2-chat-1_8b-sft"
adapter_dir = "./work_dirs/internlm2_lora_huanhuan/checkpoint-650"

# tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False, trust_remote_code=True)

# 创建模型
model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    torch_dtype=torch.float16,
    trust_remote_code=True,
    device_map='auto',
)

# 2种加载adapter的方式
# 1. load adapter https://huggingface.co/docs/transformers/main/zh/peft
# model.load_adapter(adapter_dir)

# 2. https://huggingface.co/docs/peft/v0.9.0/en/package_reference/peft_model#peft.PeftModel.from_pretrained
model = PeftModel.from_pretrained(model, adapter_dir)

print(f"model.device: {model.device}, model.dtype: {model.dtype}")


system_prompt = "现在你要扮演皇帝身边的女人--甄嬛"

def build_inputs(query: str, history: List[Tuple[str, str]] = [], meta_instruction="我是系统"):
    prompt = ""
    if meta_instruction:
        prompt += f"""<s>[UNUSED_TOKEN_146]system\n{meta_instruction}[UNUSED_TOKEN_145]\n"""
    else:
        prompt += "<s>"
    for record in history:
        prompt += f"""[UNUSED_TOKEN_146]user\n{record[0]}[UNUSED_TOKEN_145]\n[UNUSED_TOKEN_146]assistant\n{record[1]}[UNUSED_TOKEN_145]\n"""
    prompt += f"""[UNUSED_TOKEN_146]user\n{query}[UNUSED_TOKEN_145]\n[UNUSED_TOKEN_146]assistant\n"""
    return prompt


inputs = build_inputs("小主，敬事房传来消息，说皇上晚上去华妃那儿", history=[], meta_instruction=system_prompt)
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
