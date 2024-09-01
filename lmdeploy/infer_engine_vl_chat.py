from infer_engine import InferEngine, LmdeployConfig
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
    deploy_method = 'local',
)

# 载入模型
infer_engine = InferEngine(
    backend = 'lmdeploy', # transformers, lmdeploy, api
    lmdeploy_config = LMDEPLOY_CONFIG
)


image = load_image('https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/tests/data/tiger.jpeg')
history = [] # [['What is the capital of France?', 'The capital of France is Paris.'], ['Thanks', 'You are Welcome']]
query = ('描述一下这幅图片', image)

response, history = infer_engine.chat(
    query = query,
    history = history,
    max_new_tokens = 1024,
    temperature = 0.8,
    top_p = 0.8,
    top_k = 40,
)
print("response:", response)
print("history:", history)
print("\n\n")
# response: 这是一张老虎的图片。老虎正躺在绿色的草地上，它的身体呈现出典型的条纹图案。老虎的面部表情显得十分警觉和威严，它的眼睛注视着前方。背景是自然环境，有绿色的草地和一些阴影，整体给人一种宁静而庄严的感觉。
# history: [
#   [('描述一下这幅图片', <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=278x182 at 0x7F71384E0730>),
#    '这是一张老虎的图片。老虎正躺在绿色的草地上，它的身体呈现出典型的条纹图案。老虎的面部表情显得十分警觉和威严，它的眼睛注视着前方。背景是自然环境，有绿色的草地和一些阴影，整体给人一种宁静而庄严的感觉。']
# ]


query = '根据这一张图片写一首诗'
response, history = infer_engine.chat(
    query = query,
    history = history,
    max_new_tokens = 1024,
    temperature = 0.8,
    top_p = 0.8,
    top_k = 40,
)
print("response:", response)
print("history:", history)
print("\n\n")
# response: 老虎卧绿草，威严目光高。
# 条纹如云霞，威武姿态扬。
# 自然背景美，静谧似梦境。
# 雄伟姿态显，威严不可挡。
# history: [
#   [('描述一下这幅图片', <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=278x182 at 0x7F71384E0730>),
#    '这是一张老虎的图片。老虎正躺在绿色的草地上，它的身体呈现出典型的条纹图案。老虎的面部表情显得十分警觉和威严，它的眼睛注视着前方。背景是自然环境，有绿色的草地和一些阴影，整体给人一种宁静而庄严的感觉。'],
#   ['根据这一张图片写一首诗', '老虎卧绿草，威严目光高。\n条纹如云霞，威武姿态扬。\n自然背景美，静谧似梦境。\n雄伟姿态显，威严不可挡。']
# ]


image_urls=[
    'https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/demo/resources/human-pose.jpg',
    'https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/demo/resources/det.jpg'
]

images = [load_image(img_url) for img_url in image_urls]
history = [] # [['What is the capital of France?', 'The capital of France is Paris.'], ['Thanks', 'You are Welcome']]
query = (f'Image-1: {IMAGE_TOKEN}\nImage-2: {IMAGE_TOKEN}\n描述一下这两张图片', images)

response, history = infer_engine.chat(
    query = query,
    history = history,
    max_new_tokens = 1024,
    temperature = 0.8,
    top_p = 0.8,
    top_k = 40,
)
print("response:", response)
print("history:", history)
print("\n\n")
# response: 这两张图片展示了一个冬季场景。
# 1. 第一张图片：
#    - 图中是一位穿着红色夹克和黑色裤子的滑雪者，戴着帽子，戴着太阳镜，手持滑雪杖。
#    - 滑雪者正在雪地上滑雪，脚下是一块滑雪板。
#    - 背景中可以看到雪覆盖的山坡和一些树木。
# 2. 第二张图片：
#    - 图中是一张长椅，长椅是金属材质的，表面有垂直的条纹。
#    - 长椅位于一个公园的草地上，周围有树木和草地。
#    - 背景中可以看到一些停放的汽车和街道，以及一些人在远处活动。
# 这两张图片分别展示了一个人在雪地中滑雪，以及一个公园中的长椅场景。
# history: [
#     [('Image-1: <IMAGE_TOKEN>\nImage-2: <IMAGE_TOKEN>\n描述一下这两张图片', [<PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=218x346 at 0x7F9C64950F70>, <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=640x427 at 0x7F9C64951480>]),
#      '这两张图片展示了一个冬季场景。\n\n1. 第一张图片：\n   - 图中是一位穿着红色夹克和黑色裤子的滑雪者，戴着帽子，戴着太阳镜，手持滑雪杖。\n   - 滑雪者正在雪地上滑雪，脚下是一块滑雪板。\n   - 背景中可以看到雪覆盖的山坡和一些树木。\n\n2. 第二张图片：\n   - 图中是一张长椅，长椅是金属材质的，表面有垂直的条纹。\n   - 长椅位于一个公园的草地上，周围有树木和草地。\n   - 背景中可以看到一些停放的汽车和街道，以及一些人在远处活动。\n\n这两张图片分别展示了一个人在雪地中滑雪，以及一个公园中的长椅场景。']
# ]


query = '联想这2张图片讲一个关联的小故事'
response, history = infer_engine.chat(
    query = query,
    history = history,
    max_new_tokens = 1024,
    temperature = 0.8,
    top_p = 0.8,
    top_k = 40,
)
print("response:", response)
print("history:", history)
# response: 想象一下，这两个人物，一个是在雪山上滑雪的滑雪者，另一个是在公园长椅上的行人。
# 在一个寒冷的冬日早晨，滑雪者穿着厚重的滑雪服，戴着手套和帽子，熟练地操控着滑雪杖，沿着雪坡滑下。滑雪时，她感受到滑雪杖的重量和风带来的寒冷，但她依然保持着冷静和专注。
# 而坐在公园长椅上的行人，则悠闲地享受着阳光和空气。她穿着舒适的休闲服，戴着太阳镜，在树荫下小憩。阳光透过树叶的缝隙洒在她的身上，让她感受到温暖和宁静。
# 这两个场景虽然不同，但都体现了人在面对大自然时的不同态度和心情。滑雪者专注于滑雪的挑战和乐趣，而行人则享受着悠闲和放松的时刻。这反映了人与自然和谐相处的重要性，以及在不同的情境下，人们如何应对和享受生活。
# 这就是这两张图片背后的故事，展示了滑雪者与公园长椅上行人的不同心态和体验，以及他们各自在自然中的不同角色和位置。
# history: [
#     [('Image-1: <IMAGE_TOKEN>\nImage-2: <IMAGE_TOKEN>\n描述一下这两张图片', [<PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=218x346 at 0x7F9C64950F70>, <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=640x427 at 0x7F9C64951480>]),
#      '这两张图片展示了一个冬季场景。\n\n1. 第一张图片：\n   - 图中是一位穿着红色夹克和黑色裤子的滑雪者，戴着帽子，戴着太阳镜，手持滑雪杖。\n   - 滑雪者正在雪地上滑雪，脚下是一块滑雪板。\n   - 背景中可以看到雪覆盖的山坡和一些树木。\n\n2. 第二张图片：\n   - 图中是一张长椅，长椅是金属材质的，表面有垂直的条纹。\n   - 长椅位于一个公园的草地上，周围有树木和草地。\n   - 背景中可以看到一些停放的汽车和街道，以及一些人在远处活动。\n\n这两张图片分别展示了一个人在雪地中滑雪，以及一个公园中的长椅场景。'],
#     ['联想这2张图片讲一个关联的小故事',
#      '想象一下，这两个人物，一个是在雪山上滑雪的滑雪者，另一个是在公园长椅上的行人。\n\n在一个寒冷的冬日早晨，滑雪者穿着厚重的滑雪服，戴着手套和帽子，熟练地操控着滑雪杖，沿着雪坡滑下。滑雪时，她感受到滑雪杖的重量和风带来的寒冷，但她依然保持着冷静和专注。\n\n而坐在公园长椅上的行人，则悠闲地享受着阳光和空气。她穿着舒适的休闲服，戴着太阳镜，在树荫下小憩。阳光透过树叶的缝隙洒在她的身上，让她感受到温暖和宁静。\n\n这两个场景虽然不同，但都体现了人在面对大自然时的不同态度和心情。滑雪者专注于滑雪的挑战和乐趣，而行人则享受着悠闲和放松的时刻。这反映了人与自然和谐相处的重要性，以及在不同的情境下，人们如何应对和享受生活。\n\n这就是这两张图片背后的故事，展示了滑雪者与公园长椅上行人的不同心态和体验，以及他们各自在自然中的不同角色和位置。']
# ]

