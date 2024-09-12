import json
from pathlib import Path

file_path = Path("Huatuo26M-Lite/category_format/其他/1.jsonl")


with open(file_path, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        data = json.loads(line)
        print(data["answer"])
        print("\n", "*" * 100, "\n")
        if i > 10:
            break
