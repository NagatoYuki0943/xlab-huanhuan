from load_pipe import load_pipe, LmdeployConfig
from lmdeploy import GenerationConfig
from lmdeploy.vl import load_image
from lmdeploy.vl.constants import IMAGE_TOKEN


MODEL_PATH = '../models/InternVL2-2B'

SYSTEM_PROMPT = '我是书生·万象，英文名是InternVL，是由上海人工智能实验室及多家合作单位联合开发的多模态基础模型。人工智能实验室致力于原始技术创新，开源开放，共享共创，推动科技进步和产业发展。'

LMDEPLOY_CONFIG = LmdeployConfig(
    model_path = MODEL_PATH,
    backend = 'turbomind',
    model_name = 'internvl-internlm2',
    model_format = 'hf',
    tp = 1,                         # Tensor Parallelism.
    max_batch_size = 128,
    cache_max_entry_count= 0.8,     # 调整 KV Cache 的占用比例为0.8
    quant_policy = 0,               # KV Cache 量化, 0 代表禁用, 4 代表 4bit 量化, 8 代表 8bit 量化
    system_prompt = SYSTEM_PROMPT,
)

pipe = load_pipe(config = LMDEPLOY_CONFIG)

# https://lmdeploy.readthedocs.io/zh-cn/latest/api/pipeline.html#generationconfig
gen_config = GenerationConfig(
    n = 1,
    max_new_tokens = 1024,
    top_p = 0.8,
    top_k = 40,
    temperature = 0.8,
    repetition_penalty = 1.0,
    ignore_eos = False,
    random_seed = None,
    stop_words = None,
    bad_words = None,
    min_new_tokens = None,
    skip_special_tokens = True,
)

image = load_image('https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/tests/data/tiger.jpeg')
response = pipe(('describe this image', image))
print(response.text)


image_urls=[
    'https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/demo/resources/human-pose.jpg',
    'https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/demo/resources/det.jpg'
]

images = [load_image(img_url) for img_url in image_urls]
response = pipe((f'Image-1: {IMAGE_TOKEN}\nImage-2: {IMAGE_TOKEN}\ndescribe these two images', images))
print(response.text)
