import os
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
from copy import deepcopy
from collections import defaultdict


# 保存路径
file_paths = [
    "Huatuo26M-Lite/Huatuo26M-Lite-markdown.jsonl",
    "Huatuo26M-Lite/Huatuo26M-Lite-old.jsonl",
]
save_paths = [
    "Huatuo26M-Lite/Huatuo26M-Lite-markdown-xtuner.json",
    "Huatuo26M-Lite/Huatuo26M-Lite-old-xtuner.json",
]


system = """
你是医疗保健智能体，名字叫做 "HeathcareAgent"。
    - ”HeathcareAgent“ 可以根据自己丰富的医疗知识来回答问题。
    - ”HeathcareAgent“ 的回答应该是有益的、诚实的和无害的。
    - ”HeathcareAgent“ 可以使用用户选择的语言（如英语和中文）进行理解和交流。
"""


xtuner_format = {
    "conversation": [
        {
            "system": "",
            "input": "",
            "output": ""
        }
    ]
}


for file_path, save_path in zip(file_paths, save_paths):
    # 读取文件
    xtuner_dataset = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            if data['answer'].strip() == "" or data["question"].strip() == "":
                print("empty data")
                continue
            xtuner_format_c = deepcopy(xtuner_format)
            xtuner_format_c["conversation"][0]["system"] = system
            xtuner_format_c["conversation"][0]["input"] = data["question"]
            xtuner_format_c["conversation"][0]["output"] = data["answer"]
            xtuner_dataset.append(xtuner_format_c)

    # 写入文件
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(xtuner_dataset, f, ensure_ascii=False, indent=4)
