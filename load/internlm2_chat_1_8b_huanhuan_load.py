import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, BitsAndBytesConfig
from peft import PeftModel
import torch
from typing import List, Tuple


print("torch version: ", torch.__version__)
print("transformers version: ", transformers.__version__)

model_path = "./models/internlm2-chat-1_8b"
adapter_dir = "./work_dirs/internlm2_chat_1_8b_qlora_huanhuan_e3_hf/checkpoint-699"
# adapter_dir = "./work_dirs/internlm2_chat_1_8b_qlora_huanhuan_e3/hf"
quantization = False

# tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)

# 量化
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,                      # 是否在4位精度下加载模型。如果设置为True，则在4位精度下加载模型。
    load_in_8bit=False,
    llm_int8_threshold=6.0,
    llm_int8_has_fp16_weight=False,
    bnb_4bit_compute_dtype=torch.float16,   # 4位精度计算的数据类型。这里设置为torch.float16，表示使用半精度浮点数。
    bnb_4bit_quant_type='nf4',              # 4位精度量化的类型。这里设置为"nf4"，表示使用nf4量化类型。 nf4: 4bit-NormalFloat
    bnb_4bit_use_double_quant=True,         # 是否使用双精度量化。如果设置为True，则使用双精度量化。
)

# 创建模型
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    trust_remote_code=True,
    device_map='auto',
    low_cpu_mem_usage=True, # 是否使用低CPU内存,使用 device_map 参数必须为 True
    quantization_config=quantization_config if quantization else None,
)
model.eval()

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
        # <s> tokenizer会默认添加,不过这里使用手动添加的方式
        prompt += f"""<s><|im_start|>system\n{meta_instruction}<|im_end|>\n"""
    else:
        prompt += "<s>"
    for record in history:
        prompt += f"""<|im_start|>user\n{record[0]}<|im_end|>\n<|im_start|>assistant\n{record[1]}<|im_end|>\n"""
    prompt += f"""<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n"""
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
