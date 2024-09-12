# https://huggingface.co/datasets/FreedomIntelligence/Huatuo26M-Lite
# https://hf-mirror.com/datasets/FreedomIntelligence/Huatuo26M-Lite


from datasets import load_dataset
import os
import numpy as np
from pathlib import Path
from tqdm import tqdm
from copy import deepcopy
import json
from collections import defaultdict


# 清洗数据

# 数据集
path = "FreedomIntelligence/Huatuo26M-Lite"  # hugginface 地址
path = "./Huatuo26M-Lite"  # 本地路径

dataset = load_dataset(path)

# 拆分数据

# 每种病的类别数量
# 每种病的类别数量
disease_categories_number = defaultdict(lambda: 0)
disease_categories = defaultdict(list)

for data in tqdm(dataset["train"]):
    disease_categories_number[data["label"]] += 1
    disease_categories[data["label"]].append(data)

# 将每种疾病都放到同一个文件中
# subdir = os.path.join(path, "category")
# os.makedirs(subdir, exist_ok=True)
# for category_name, category_data in disease_categories.items():
#     save_path = os.path.join(subdir, f"{category_name}.jsonl")
#     print(save_path)
#     with open(save_path, "w", encoding="utf-8") as f:
#         for data in tqdm(category_data):
#             f.write(json.dumps(data, ensure_ascii=False) + "\n")

# 每个文件1000条数据
split_size = 1000

# 将每种疾病按照 split_size 切片,每个疾病都放到一个文件夹中，每个切片都放到这个文件夹中
for category_name, category_data in disease_categories.items():
    print(category_name)
    category_dir = os.path.join(path, "category", f"{category_name}")
    os.makedirs(category_dir, exist_ok=True)

    category_group_data = defaultdict(list)
    for i, data in enumerate(category_data):
        group_id = i // split_size
        category_group_data[group_id].append(data)

    for id, group_data in tqdm(category_group_data.items()):
        file_path = os.path.join(category_dir, f"{id}.jsonl")
        with open(file_path, "w", encoding="utf-8") as f:
            for data in group_data:
                f.write(json.dumps(data, ensure_ascii=False) + "\n")
