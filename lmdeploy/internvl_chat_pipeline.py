from lmdeploy import GenerationConfig
from lmdeploy.serve.async_engine import AsyncEngine
from lmdeploy.serve.vl_async_engine import VLAsyncEngine
from lmdeploy.vl import load_image
from lmdeploy.vl.constants import IMAGE_TOKEN

from load_pipe import load_pipe, LmdeployConfig
from infer_utils import encode_image_base64


MODEL_PATH = "../models/InternVL2-2B"

SYSTEM_PROMPT = "我是书生·万象，英文名是InternVL，是由上海人工智能实验室及多家合作单位联合开发的多模态基础模型。人工智能实验室致力于原始技术创新，开源开放，共享共创，推动科技进步和产业发展。"

LMDEPLOY_CONFIG = LmdeployConfig(
    model_path=MODEL_PATH,
    backend="turbomind",
    model_name="internvl-internlm2",
    model_format="hf",
    tp=1,  # Tensor Parallelism.
    max_batch_size=128,
    cache_max_entry_count=0.8,  # 调整 KV Cache 的占用比例为0.8
    quant_policy=0,  # KV Cache 量化, 0 代表禁用, 4 代表 4bit 量化, 8 代表 8bit 量化
    system_prompt=SYSTEM_PROMPT,
)

pipe: AsyncEngine | VLAsyncEngine = load_pipe(config=LMDEPLOY_CONFIG)

# https://lmdeploy.readthedocs.io/zh-cn/latest/api/pipeline.html#generationconfig
gen_config = GenerationConfig(
    n=1,
    max_new_tokens=1024,
    top_p=0.8,
    top_k=40,
    temperature=0.8,
    repetition_penalty=1.0,
    ignore_eos=False,
    random_seed=None,
    stop_words=None,
    bad_words=None,
    min_new_tokens=None,
    skip_special_tokens=True,
    logprobs=None,
)

image = load_image(
    "https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/tests/data/tiger.jpeg"
)
response = pipe(
    prompts=("描述一下这张图片", image),
    # gen_config = gen_config
)
print(response.text)
print("*" * 100)
# 这张图片展示了一只老虎，它正躺在草地上。老虎的毛色主要是橙色和黑色相间的条纹，它有着锐利的眼神和强壮的体态。
# 背景是一片绿色的草地，阳光明媚，环境显得非常自然和宁静。老虎的姿态显得非常放松，似乎在享受这片宁静的环境。


# openai 格式 PIL.Image.Image
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "描述一下这张图片",
            },
            {"type": "image_data", "image_data": {"data": image}},
        ],
    }
]
response = pipe(
    prompts=messages,
    # gen_config = gen_config
)
print(response.text)
print("*" * 100)
# 这张图片展示了一只老虎。老虎正躺在绿色的草地上，四肢舒展，显得非常放松。老虎的毛色是典型的虎斑纹，有黑色、白色和棕色的条纹。
# 老虎的面部表情显得非常平静和威严，它的眼睛锐利而专注，耳朵竖立，似乎在观察周围的环境。背景是一些模糊的绿色植物，
# 可能是树木或其他植被，给人一种宁静而自然的感觉。


# 尝试把一条消息转换为多条消息,不太好用,回答结果不太好
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": f"Image: {IMAGE_TOKEN}\n",
            },
            {"type": "image_data", "image_data": {"data": image}},
        ],
    },
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "描述一下这张图片",
            },
        ],
    },
]
response = pipe(
    prompts=messages,
    # gen_config = gen_config
)
print(response.text)
print("*" * 100)
print("*" * 100)
# 这张图片展示了一只老虎，它正悠闲地躺在绿色的草地上。老虎的毛色主要是橙色和白色，身上有黑色和棕色的条纹。
# 它的眼睛锐利，表情看起来非常警觉。背景是模糊的，但可以看出是自然环境，有阳光照射在老虎身上，使其显得更加生动和真实。


image_urls = [
    "https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/demo/resources/human-pose.jpg",
    "https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/demo/resources/det.jpg",
]

images = [load_image(img_url) for img_url in image_urls]
response = pipe(
    prompts=(
        f"Image-1: {IMAGE_TOKEN}\nImage-2: {IMAGE_TOKEN}\n描述一下这两张图片",
        images,
    ),
    # gen_config = gen_config
)
print(response.text)
print("*" * 100)
# 这两张图片展示了一个冬季的场景。
# 第一张图片：
# - 图片中有一名穿着红色夹克和黑色裤子的滑雪者，戴着滑雪帽和太阳镜，正准备滑下山坡。
# - 滑雪者手持滑雪杖，脚踏滑雪板，看起来正在准备滑行。
# - 背景是一片白雪覆盖的山坡，看起来像是在滑雪场。
# 第二张图片：
# - 图片展示了一个公园的景象。
# - 前景中有一张金属长椅，长椅上覆盖着一些叶子，显得有些陈旧。
# - 长椅周围是绿色的草地，长椅后面有一些树木。
# - 长椅旁边有一个小路，可以看到远处有几辆停放的汽车。
# - 背景中可以看到一些行人，似乎是在一个城市的公园中。
# 两张图片分别展示了滑雪和城市公园的场景，一个是滑雪活动，另一个是城市公园中的休闲时光。


# openai 格式 PIL.Image.Image
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": f"Image-1: {IMAGE_TOKEN}\nImage-2: {IMAGE_TOKEN}\n描述一下这两张图片",
            },
            {"type": "image_data", "image_data": {"data": images[0]}},
            {"type": "image_data", "image_data": {"data": images[1]}},
        ],
    }
]
response = pipe(
    prompts=messages,
    # gen_config = gen_config
)
print(response.text)
print("*" * 100)
# 这两张图片展示的是不同的场景。
# 第一张图片展示了一位穿着红色夹克和黑色裤子的滑雪者，戴着帽子，穿着滑雪靴，正站在雪地上，似乎准备开始滑雪。
# 背景中可以看到一些雪地和远处的树木。
# 第二张图片展示的是一个公园或街道场景。公园内有一张长椅，长椅是金属材质，上面有纵向的条纹。
# 长椅旁边是绿色的草地，长椅后面可以看到一些树木和汽车，还有一条人行道。整体氛围显得非常宁静和自然。


images_base64_data = [encode_image_base64(img_url) for img_url in image_urls]
# openai 格式 image_base64
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": f"Image-1: {IMAGE_TOKEN}\nImage-2: {IMAGE_TOKEN}\n描述一下这两张图片",
            },
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{images_base64_data[0]}"},
            },
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{images_base64_data[1]}"},
            },
        ],
    }
]
response = pipe(
    prompts=messages,
    # gen_config = gen_config
)
print(response.text)
print("*" * 100)
# 这两张图片展示了不同场景下的户外活动和休闲活动。
# 第一张图片：
# - 图中是一位正在滑雪的女性。她穿着红色夹克和黑色裤子，戴着黑白条纹的帽子，太阳镜，并持有滑雪杖。她站在雪坡上，看起来正在进行滑雪活动。
# 第二张图片：
# - 这是一张公园场景的照片。图中有一张长椅，长椅旁边是草地，背景中可以看到一些树木和街道。长椅的设计比较现代，有竖直的条纹。远处有一些停放的汽车和行人。长椅周围有一些落叶，显示这是一个公园或类似的休闲场所。
# 这两张图片展示了人们在不同环境下进行户外活动的情景，从滑雪到休闲散步，都展现了人们享受自然和户外活动的场景。


# 删除 IMAGE_TOKEN
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": f"描述一下这两张图片",
            },
            {"type": "image_data", "image_data": {"data": images[0]}},
            {"type": "image_data", "image_data": {"data": images[1]}},
        ],
    }
]
response = pipe(
    prompts=messages,
    # gen_config = gen_config
)
print(response.text)
print("*" * 100)
# 这两张图片分别展示了不同的场景。
# 1. **第一张图片**：
#    - 场景：一个雪地
#    - 人物：一个戴着帽子和墨镜的人
#    - 活动：滑雪或单板滑雪
#    - 其他细节：地面上有滑雪痕迹，背景中可以看到一些汽车和远处的建筑物
# 2. **第二张图片**：
#    - 场景：一个公园或广场
#    - 人物：一个坐在长椅上的人
#    - 活动：坐在长椅上
#    - 其他细节：长椅上没有人，背景中可以看到一些树木和停放的汽车
# 这两张图片分别展示了冬季户外活动（滑雪或单板滑雪）和休闲活动（坐在公园长椅上）的场景。


# 尝试把一条消息转换为多条消息,不太好用,回答结果不太好
messages = [
    {
        "role": "user",
        "content": [
            # text占位符
            {
                "type": "text",
                "text": f"Image: {IMAGE_TOKEN}\n",
            },
            {"type": "image_data", "image_data": {"data": images[0]}},
        ],
    },
    {
        "role": "user",
        "content": [
            # text占位符
            {
                "type": "text",
                "text": f"Image: {IMAGE_TOKEN}\n",
            },
            {"type": "image_data", "image_data": {"data": images[1]}},
        ],
    },
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": f"描述一下上面这两张图片",
            },
        ],
    },
]
response = pipe(
    prompts=messages,
    # gen_config = gen_config
)
print(response.text)
print("*" * 100)
# 这张图片展示了一位穿着红色夹克和黑色裤子的滑雪者，站在雪坡上。她戴着黑白条纹的帽子，戴着墨镜，显得非常时尚。
# 滑雪者双手握着滑雪杖，看起来正在准备滑行。背景是一片冬季的雪地，远处可以看到一些树木和汽车。
# 另一张图片展示了一个公园的景象。前景中有一张长椅，长椅的扶手和靠背上有垂直的条纹。
# 长椅位于一片绿色的草地上，旁边是一条铺有水泥的小路。远处可以看到一些树木和停放的汽车，还有一条铺有砖块的步道。
