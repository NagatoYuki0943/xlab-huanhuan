# https://huggingface.co/OpenGVLab/InternVL-Chat-V1-5

import requests
from PIL import Image
import torch
from torch import Tensor
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import GenerationConfig
from load_tokenizer_processor_and_model import load_tokenizer_processor_and_model, TransformersConfig


PRETRAINED_MODEL_NAME_OR_PATH = '../models/InternVL-Chat-V1-5'
ADAPTER_PATH = None
# 量化
LOAD_IN_8BIT= False
LOAD_IN_4BIT = False

SYSTEM_PROMPT = """You are an AI assistant whose name is InternLM (书生·浦语)."""

TRANSFORMERS_CONFIG = TransformersConfig(
    pretrained_model_name_or_path = PRETRAINED_MODEL_NAME_OR_PATH,
    adapter_path = ADAPTER_PATH,
    load_in_8bit = LOAD_IN_8BIT,
    load_in_4bit = LOAD_IN_4BIT,
    model_name = 'internlm2-chat',  # useless
    system_prompt = SYSTEM_PROMPT   # useless
)

tokenizer, processor, model = load_tokenizer_processor_and_model(config=TRANSFORMERS_CONFIG)



IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size: int):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform


def find_closest_aspect_ratio(
    aspect_ratio: float,
    target_ratios: list,
    width: int,
    height: int,
    image_size: int
):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=6, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_image(image_file, input_size=448, max_num=6):
    image: Image.Image = Image.open(image_file).convert('RGB')
    transform: T.Compose= build_transform(input_size=input_size)
    images: list = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values: list = [transform(image) for image in images]
    pixel_values: Tensor = torch.stack(pixel_values)
    return pixel_values



url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image_raw = requests.get(url, stream=True).raw
# set the max number of tiles in `max_num`
pixel_values: Tensor = load_image(image_raw, max_num=6).to(torch.bfloat16).cuda()


# generation_config = GenerationConfig(
#     max_new_tokens = 1024,
#     do_sample = True,
#     num_beams = 1,
#     temperature = 0.01,
#     top_k = 40,
#     top_p = 0.8,
#     eos_token_id = [tokenizer.eos_token_id]
# )

generation_config = dict(
    max_new_tokens = 1024,
    do_sample = True,
    num_beams = 1,
    temperature = 0.01,
    top_k = 40,
    top_p = 0.8,
    eos_token_id = [tokenizer.eos_token_id]
)

# tokenizer, pixel_values, question, generation_config, history=None, return_history=False,
# https://huggingface.co/OpenGVLab/InternVL-Chat-V1-5/blob/main/modeling_internvl_chat.py#L271

# single-round single-image conversation
question = "请详细描述图片" # Please describe the picture in detail
response: str = model.chat(
    tokenizer = tokenizer,
    pixel_values = pixel_values,
    question = question,
    generation_config = generation_config,
)
print(f"question: {question}")
print(f"response: {response}")
print('\n\n')
# question: 请详细描述图片
# response: 在这张图片的宁静场景中，两只虎斑猫正在一张鲜艳的粉色沙发上安静地睡觉。左边的猫，毛色是灰色和黑色的混合，侧躺着，头舒适地靠在沙发扶手上。它的同伴在右边，毛色是橙色和黑色的混合，也侧躺着，头靠在沙发扶手上。
# 这两只猫似乎在沙发上找到了完美的休息点，它们放松的姿势表明它们正在深度睡眠中。它们的位置和姿势表明它们彼此之间感到舒适和熟悉。
# 在它们之间，有两个遥控器，一个在左边，一个在右边。它们的存在为这个本来自然的场景增添了一丝现代感。遥控器和睡觉的猫一起创造了一个有趣的对比，将科技与自然融为一体。
# 总的来说，这张图片捕捉了两只虎斑猫生命中宁静的瞬间，它们在粉色沙发上享受着平静的睡眠。


# multi-round single-image conversation
question = "请详细描述图片" # Please describe the picture in detail
history: list # [['What is the capital of France?', 'The capital of France is Paris.'], ['Thanks', 'You are Welcome']]
response, history = model.chat(
    tokenizer = tokenizer,
    pixel_values = pixel_values,
    question = question,
    generation_config = generation_config,
    history = None,
    return_history = True
)
print(f"question: {question}")
print(f"response: {response}")
print('\n\n')
# question: 请详细描述图片
# response: 在这张图片的宁静场景中，两只虎斑猫正在一张鲜艳的粉色沙发上安静地睡觉。左边的猫，毛色是灰色和黑色的混合，侧躺着，头舒适地靠在沙发扶手上。它的同伴在右边，毛色是橙色和黑色的混合，也侧躺着，头靠在沙发扶手上。
# 这两只猫似乎在沙发上找到了完美的休息点，它们放松的姿势表明它们正在深度睡眠中。它们睡觉的沙发不仅颜色鲜艳，而且还有两个遥控器放在上面，暗示着这个空间用于放松和娱乐。
# 这张图片捕捉到了这两只猫生命中宁静的瞬间，它们在粉色沙发的舒适环境中享受着平静的睡眠。

question = "请根据图片写一首诗" # Please write a poem according to the picture
response, history = model.chat(
    tokenizer = tokenizer,
    pixel_values = pixel_values,
    question = question,
    generation_config = generation_config,
    history = history,
    return_history = True
)
print(f"question: {question}")
print(f"response: {response}")
print('\n\n')
# question: 请根据图片写一首诗
# response: 两只虎斑猫，
# 粉色沙发上安睡。
# 左边一只灰色黑，
# 右边一只橙色黑。

# 头枕扶手舒心怀，
# 遥控器旁随意摆。
# 宁静画面入眼帘，
# 猫儿安睡真可爱。


# multi-round multi-image conversation
url1 = "http://images.cocodataset.org/val2017/000000039769.jpg"
image_raw1 = requests.get(url1, stream=True).raw
url2 = "http://images.cocodataset.org/train2017/000000000650.jpg"
image_raw2 = requests.get(url1, stream=True).raw
pixel_values1: Tensor = load_image(image_raw1, max_num=6).to(torch.bfloat16).cuda()
pixel_values2: Tensor = load_image(image_raw2, max_num=6).to(torch.bfloat16).cuda()
pixel_values = torch.cat((pixel_values1, pixel_values2), dim=0)

question = "详细描述这两张图片" # Describe the two pictures in detail
response, history = model.chat(
    tokenizer = tokenizer,
    pixel_values = pixel_values,
    question = question,
    generation_config = generation_config,
    history = None,
    return_history = True
)
print(f"question: {question}")
print(f"response: {response}")
print('\n\n')
# question: 详细描述这两张图片
# response: 这张图片展示了两只猫在一张粉色的沙发上休息。左边的猫是灰色的，它侧躺着，头枕在沙发上，身体伸展在沙发上。右边的猫是棕色的，它也侧躺着，头枕在沙发上，身体伸展在沙发上。两只猫都看起来很放松，似乎在享受一个宁静的下午。


question = "这两张图片的相同点和区别分别是什么" # What are the similarities and differences between these two pictures
response, history = model.chat(
    tokenizer = tokenizer,
    pixel_values = pixel_values,
    question = question,
    generation_config = generation_config,
    history = history,
    return_history = True
)
print(f"question: {question}")
print(f"response: {response}")
# question: 这两张图片的相同点和区别分别是什么
# response: 这两张图片的相同点是它们都展示了两只猫在一张粉色的沙发上休息。不同点在于，左边的猫是灰色的，而右边的猫是棕色的。此外，左边的猫侧躺着，头枕在沙发上，身体伸展在沙发上，而右边的猫也侧躺着，头枕在沙发上，身体伸展在沙发上。
