import numpy as np
from pathlib import Path
import json
from huatuo_ai_format import Formater
from tqdm import tqdm
import time
from loguru import logger


dummy = False
if dummy:
    # 线程
    from multiprocessing.dummy import Process, Pool, Queue, Pipe, Lock
else:
    # 进程
    from multiprocessing import Process, Pool, Queue, Pipe, Lock


log_file = logger.add("./logs/runtime_{time}.log")


base_path = Path("category_old/")
save_path = Path("category_markdown/")
save_path.mkdir(parents=True, exist_ok=True)


jsonl_paths = (
    list(base_path.glob("**/5.jsonl"))
    + list(base_path.glob("**/6.jsonl"))
    + list(base_path.glob("**/7.jsonl"))
    + list(base_path.glob("**/8.jsonl"))
)
jsonl_paths_num = len(jsonl_paths)
logger.info(jsonl_paths)
logger.info(f"{jsonl_paths_num = }")
# print(jsonl_paths)


def format_answers(jsonl_path: Path) -> None:
    parent_name = jsonl_path.parent.name
    # 存储路径
    formated_jsonl_path = save_path / parent_name / jsonl_path.name
    formated_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    # 记录路径
    record_path = save_path / parent_name / f"{jsonl_path.stem}_record.txt"
    logger.info("format", jsonl_path, "to", formated_jsonl_path)

    # 创建格式化工具
    formater = Formater()

    with open(jsonl_path, "r", encoding="utf-8") as jsonl_file, open(
        formated_jsonl_path, "a", encoding="utf-8"
    ) as formated_jsonl_file, open(record_path, "a", encoding="utf-8") as record_file:
        i = 1
        fails = 0
        lines = jsonl_file.read().splitlines()
        lines_num = len(lines)
        for line in tqdm(lines, desc=parent_name):
            # 跳过空行
            if len(line) < 1:
                i += 1
                continue
            # 读取
            data = json.loads(line)
            # 格式化
            res, answer = formater.format_answer(data["answer"])
            # 更新回答
            data["answer"] = answer
            # 保存
            formated_jsonl_file.write(json.dumps(data, ensure_ascii=False) + "\n")
            # 刷新缓冲区
            formated_jsonl_file.flush()
            # 写入日志
            record_file.write(f"{res}\n")
            # 刷新缓冲区
            record_file.flush()
            # 成功还是失败
            if res:
                logger.success(
                    f"{parent_name}: num = {i} / {lines_num}, data id = {data['id']} formated"
                )
            else:
                fails += 1
                logger.error(
                    f"{parent_name}: num = {i} / {lines_num}, data id = {data['id']} format failed"
                )
            i += 1
            # if fails > 10:
            #     logger.critical(f"发生大量错误，请排查")
            #     exit()
    print(f"fails number: {fails}")


if __name__ == "__main__":
    with Pool(processes=10) as pool:
        ...
        # 将多组参数传递给一个函数,生成多个进程,使进程阻塞直到结果返回
        # context 要配合 map 使用,否则也需要使用 close 和 join 方法
        pool.map(func=format_answers, iterable=jsonl_paths)

        # 将多组参数传递给一个函数,生成多个进程,非阻塞
        # pool.map_async(func=sing, iterable=argss)
        # join之前有close
        # pool.close()
        # pool.join()

    print("done")
