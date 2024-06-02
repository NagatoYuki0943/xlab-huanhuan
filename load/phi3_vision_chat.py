import requests
from PIL import Image
import torch
from transformers import GenerationConfig
from load_tokenizer_processor_and_model import load_tokenizer_processor_and_model, TransformersConfig


# clone 模型
PRETRAINED_MODEL_NAME_OR_PATH = '../models/Phi-3-vision-128k-instruct'
# os.system(f'git clone https://code.openxlab.org.cn/OpenLMLab/internlm2-chat-1.8b {PRETRAINED_MODEL_NAME_OR_PATH}')
# os.system(f'cd {PRETRAINED_MODEL_NAME_OR_PATH} && git lfs pull')
ADAPTER_PATH = None
# 量化
LOAD_IN_8BIT= False
LOAD_IN_4BIT = False

SYSTEM_PROMPT = """You are a healthy, intelligent, and helpful AI assistant."""

TRANSFORMERS_CONFIG = TransformersConfig(
    pretrained_model_name_or_path = PRETRAINED_MODEL_NAME_OR_PATH,
    adapter_path = ADAPTER_PATH,
    load_in_8bit = LOAD_IN_8BIT,
    load_in_4bit = LOAD_IN_4BIT,
    model_name = 'phi3_chat',
    system_prompt = SYSTEM_PROMPT
)

tokenizer, processor, model = load_tokenizer_processor_and_model(config=TRANSFORMERS_CONFIG)


messages = [
    {"role": "user", "content": "<|image_1|>\nWhat is shown in this image?"},
    {"role": "assistant", "content": "The chart displays the percentage of respondents who agree with various statements about their preparedness for meetings. It shows five categories: 'Having clear and pre-defined goals for meetings', 'Knowing where to find the information I need for a meeting', 'Understanding my exact role and responsibilities when I'm invited', 'Having tools to manage admin tasks like note-taking or summarization', and 'Having more focus time to sufficiently prepare for meetings'. Each category has an associated bar indicating the level of agreement, measured on a scale from 0% to 100%."},
    {"role": "user", "content": "Provide insightful questions to spark discussion."}
]

url = "https://assets-c4akfrf5b4d3f4b7.z01.azurefd.net/assets/2024/04/BMDataViz_661fb89f3845e.png"
image = Image.open(requests.get(url, stream=True).raw)

prompt = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
print(prompt)
# <|user|>
# <|image_1|>
# What is shown in this image?<|end|>
# <|assistant|>
# The chart displays the percentage of respondents who agree with various statements about their preparedness for meetings. It shows five categories: 'Having clear and pre-defined goals for meetings', 'Knowing where to find the information I need for a meeting', 'Understanding my exact role and responsibilities when I'm invited', 'Having tools to manage admin tasks like note-taking or summarization', and 'Having more focus time to sufficiently prepare for meetings'. Each category has an associated bar indicating the level of agreement, measured on a scale from 0% to 100%.<|end|>
# <|user|>
# Provide insightful questions to spark discussion.<|end|>
# <|assistant|>
#

inputs = processor(text=prompt, images=image, return_tensors="pt").to("cuda:0")
print(inputs.keys())
# dict_keys(['input_ids', 'attention_mask', 'pixel_values', 'image_sizes'])

generation_config = GenerationConfig(
    max_new_tokens = 1024,
    do_sample = True,
    num_beams = 1,
    temperature = 0.01,
    top_k = 40,
    top_p = 0.8,
    eos_token_id = [tokenizer.eos_token_id]
)

model.eval()
with torch.inference_mode():
    outputs = model.generate(
        **inputs,
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
