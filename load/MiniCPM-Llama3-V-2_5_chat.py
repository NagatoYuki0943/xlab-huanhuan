# https://huggingface.co/openbmb/MiniCPM-Llama3-V-2_5

import requests
from PIL import Image
from load_tokenizer_processor_and_model import load_tokenizer_processor_and_model, TransformersConfig
from typing import Generator


PRETRAINED_MODEL_NAME_OR_PATH = '../models/MiniCPM-Llama3-V-2_5'
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
    model_name = 'llama3',          # useless
    system_prompt = SYSTEM_PROMPT   # useless
)

tokenizer, processor, model = load_tokenizer_processor_and_model(config=TRANSFORMERS_CONFIG)


url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)
question = '图片中有什么？'
msgs = [{'role': 'user', 'content': question}]


generation_config = dict(
    max_new_tokens = 1024,
    do_sample = True,
    num_beams = 1,
    temperature = 0.8,
    top_k = 40,
    top_p = 0.8,
    eos_token_id = [tokenizer.eos_token_id]
)


response: str = model.chat(
    image = image,
    msgs = msgs,
    tokenizer = tokenizer,
    sampling = True,
    **generation_config,
    system_prompt = TRANSFORMERS_CONFIG.system_prompt,
)
print(f"question: {question}")
print(f"response: {response}")
print("\n\n")
# question: 图片中有什么？
# response: 在图片中，有两只猫躺在粉红色的毯子上。这些猫是主要的焦点，周围没有其他可辨认的物体或人物。


## if you want to use streaming, please make sure sampling=True and stream=True
## the model.chat will return a generator
response: Generator
response = model.chat(
    image = image,
    msgs = msgs,
    tokenizer = tokenizer,
    sampling = True,
    **generation_config,
    system_prompt = TRANSFORMERS_CONFIG.system_prompt,
    stream = True
)

print(f"question: {question}")
print(f"response: ", end="", flush=True)
generated_text: str = ""
for new_text in response:
    generated_text += new_text
    print(new_text, flush=True, end='')
print('\n')
# question: 图片中有什么？
# response: 在这张图片中，有两只猫躺在一条粉色的毯子上。


# 多轮对话
question = '请根据图片写一首诗'
msgs += [{'role': 'assistant', 'content': generated_text}, {'role': 'user', 'content': question}]
response = model.chat(
    image = None,
    msgs = msgs,
    tokenizer = tokenizer,
    sampling = True,
    **generation_config,
    system_prompt = TRANSFORMERS_CONFIG.system_prompt,
    stream = True
)

print(f"question: {question}")
print(f"response: ", end="", flush=True)
generated_text: str = ""
for new_text in response:
    generated_text += new_text
    print(new_text, flush=True, end='')
print('\n')
# question: 请根据图片写一首诗
# response: 粉色毯子上躺着两只猫，
# 柔软的毛发在阳光下闪闪发亮。
# 它们伸展着身体，打个盹儿，
# 一个小小的鼻子，一个睁大了眼睛。
# 轻轻地呼吸，放松而满足，
# 这是一幅宁静的画面。
# 在这个世界上，有些事情是必要的，
# 像爱和温暖，这些东西永远不会过时。
