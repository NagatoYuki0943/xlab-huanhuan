import requests
from PIL import Image
import torch
from transformers import GenerationConfig
from load_tokenizer_processor_and_model import (
    load_tokenizer_processor_and_model,
    TransformersConfig,
)


PRETRAINED_MODEL_NAME_OR_PATH = "../models/fuyu-8b"
ADAPTER_PATH = None
# 量化
LOAD_IN_8BIT = False
LOAD_IN_4BIT = False

SYSTEM_PROMPT = """You are a healthy, intelligent, and helpful AI assistant."""

TRANSFORMERS_CONFIG = TransformersConfig(
    pretrained_model_name_or_path=PRETRAINED_MODEL_NAME_OR_PATH,
    adapter_path=ADAPTER_PATH,
    load_in_8bit=LOAD_IN_8BIT,
    load_in_4bit=LOAD_IN_4BIT,
    model_name="fuyu-8b_chat",  # useless
    system_prompt=SYSTEM_PROMPT,  # useless
)

tokenizer, processor, model = load_tokenizer_processor_and_model(
    config=TRANSFORMERS_CONFIG
)


url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

prompt = "What is shown in this image?"

inputs = processor(text=prompt, images=image, return_tensors="pt").to("cuda:0")
print(inputs.keys())
# dict_keys(['input_ids', 'attention_mask', 'pixel_values', 'image_sizes'])

generation_config = GenerationConfig(
    max_new_tokens=1024,
    do_sample=True,
    num_beams=1,
    temperature=0.01,
    top_k=40,
    top_p=0.8,
    eos_token_id=[tokenizer.eos_token_id],
)

model.eval()
with torch.inference_mode():
    outputs = model.generate(
        **inputs,
        generation_config=generation_config,
    )

print(outputs)

# 取出第一条数据
ids = outputs[0].cpu()[len(inputs["input_ids"][0]) :]

# decode 处理一维数据
response = tokenizer.decode(ids, skip_special_tokens=True)
print(response)
# Two cats are lying on a pink blanket, with one cat sleeping and the other cat lying next to it.

# batch_decode 处理二维数据
# print(tokenizer.batch_decode([ids], skip_special_tokens=True)[0])
