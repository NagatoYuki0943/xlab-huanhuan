from lmdeploy import GenerationConfig
from lmdeploy.serve.async_engine import AsyncEngine
from lmdeploy.serve.vl_async_engine import VLAsyncEngine
from lmdeploy.vl import load_image
from lmdeploy.vl.constants import IMAGE_TOKEN

from load_pipe import load_pipe, LmdeployConfig
from infer_utils import encode_image_base64


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

pipe: AsyncEngine | VLAsyncEngine = load_pipe(config = LMDEPLOY_CONFIG)

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
    logprobs = None,
)

image = load_image('https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/tests/data/tiger.jpeg')
response = pipe(
    prompts = ('描述一下这张图片', image),
    # gen_config = gen_config
)
print(response.text)
print("\n\n")
# 这张图片展示了一只老虎。老虎正躺在绿色的草地上，它的身体上有典型的黑色条纹和棕色的皮毛。
# 老虎的目光坚定，似乎在注视着镜头。背景中可以看到一些树木的影子，环境显得非常宁静和自然。


# openai 格式 PIL.Image.Image
messages = [
    {
        "role": "user",
        "content": [
            {
                'type': 'text',
                'text': '描述一下这张图片',
            },
            {
                'type': 'image_data',
                'image_data': {
                    'data': image
                }
            },
        ]
    }
]
response = pipe(
    prompts = messages,
    # gen_config = gen_config
)
print(response.text)
print("\n\n")
# 这是一张老虎的照片。老虎正躺在绿色的草地上，显得非常宁静和安详。它的身体呈现出典型的橙色和黑色条纹，面部有尖锐的白色斑纹，眼神锐利，似乎在注视着前方。
# 阳光照射在老虎身上，使它的毛发显得格外光亮。背景是绿色的草地，给人一种宁静自然的感觉。


image_urls=[
    'https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/demo/resources/human-pose.jpg',
    'https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/demo/resources/det.jpg'
]

images = [load_image(img_url) for img_url in image_urls]
response = pipe(
    prompts = (f'Image-1: {IMAGE_TOKEN}\nImage-2: {IMAGE_TOKEN}\n描述一下这两张图片', images),
    # gen_config = gen_config
)
print(response.text)
print("\n\n")
# 这两张图片展示了不同的场景：
# 1. **第一张图片**：
#    - 图中是一位穿着红色夹克和黑色裤子的滑雪者，戴着黑白条纹的帽子，穿着滑雪鞋，站在雪地上。
#    - 背景中可以看到雪地和滑雪者的滑雪杖。
#    - 天气看起来比较寒冷，可能是一个滑雪场。
# 2. **第二张图片**：
#    - 图中是一个公园或街道的户外场景。
#    - 中心位置有一张金属框架的木制长椅，长椅表面有垂直的条纹。
#    - 长椅旁边是草地和几棵树。
#    - 背景中可以看到停放的汽车、树木和街道，天气晴朗，阳光明媚。
#    - 整体环境看起来非常宁静和舒适。
# 这两张图片分别展示了冬季运动和休闲户外活动的场景，以及一个安静、整洁的公园或街道环境。


# openai 格式 PIL.Image.Image
messages = [
    {
        "role": "user",
        "content": [
            {
                'type': 'text',
                'text': f'Image-1: {IMAGE_TOKEN}\nImage-2: {IMAGE_TOKEN}\n描述一下这两张图片',
            },
            {
                'type': 'image_data',
                'image_data': {
                    'data': images[0]
                }
            },
            {
                'type': 'image_data',
                'image_data': {
                    'data': images[1]
                }
            },
        ]
    }
]
response = pipe(
    prompts = messages,
    # gen_config = gen_config
)
print(response.text)
print("\n\n")
# 这两张图片展示了不同的场景：
# 1. **第一张图片**：
#    - 图片中显示的是一个正在滑雪的人。这个人穿着红色的滑雪夹克，戴着黑白条纹的毛线帽，戴着太阳镜，穿着黑色的滑雪裤和蓝色的滑雪靴。她手持滑雪杖，站在雪地上，看起来像是在滑雪道上。背景是雪地，地面上有滑雪痕迹。
# 2. **第二张图片**：
#    - 图片展示的是一个公园场景。前景中有一张长椅，长椅是黑色的金属结构，带有垂直的条纹。背景中可以看到几辆停放的汽车和一些树木，还有一条人行道。长椅周围是绿色的草地，环境显得非常宁静和舒适。
# 这两张图片分别展示了冬季运动（滑雪）和休闲户外活动的场景，分别强调了滑雪运动的活力和宁静的公园环境。


images_base64_data = [encode_image_base64(img_url) for img_url in image_urls]
# openai 格式 image_base64
messages = [
    {
        "role": "user",
        "content": [
            {
                'type': 'text',
                'text': f'Image-1: {IMAGE_TOKEN}\nImage-2: {IMAGE_TOKEN}\n描述一下这两张图片',
            },
            {
                'type': 'image_url',
                'image_url': {
                    'url':
                    f'data:image/jpeg;base64,{images_base64_data[0]}'
                }
            },
            {
                'type': 'image_url',
                'image_url': {
                    'url':
                    f'data:image/jpeg;base64,{images_base64_data[1]}'
                }
            },
        ]
    }
]
response = pipe(
    prompts = messages,
    # gen_config = gen_config
)
print(response.text)
# 这两张图片展示了一个滑雪场景和公园场景。
# 第一张图片展示了一位滑雪者，她穿着红色和黑色的滑雪服，戴着墨镜和帽子，正在滑雪。背景是一个滑雪场地，地面是雪地。滑雪者手持滑雪杖，似乎正在滑行。
# 第二张图片展示了一个公园场景，公园内有几张长椅和树木。长椅上覆盖着铁皮，表面有竖直的条纹。背景中可以看到一些停放的汽车和更多的树木，还有一条铺好的步道。整体氛围显得非常宁静和休闲。
