import os
from huggingface_hub import hf_hub_download, snapshot_download


endpoint = "https://hf-mirror.com"
proxies = {"https": "http://localhost:7897"}


# 设置环境变量
# os.environ['HF_ENDPOINT'] = endpoint
# 下载整个模型库
# os.system('huggingface-cli download --resume-download internlm/internlm2-chat-1_8b --local-dir models/internlm2-chat-1_8b')


# 下载整个模型库
if False:
    snapshot_download(
        repo_id="internlm/internlm2-chat-1_8b",
        local_dir="models/internlm2-chat-1_8b",
        # proxies=proxies,
        max_workers=8,
        endpoint=endpoint,
    )


# 下载模型部分文件
hf_hub_download(
    repo_id="internlm/internlm2-chat-1_8b",
    filename="config.json",
    local_dir="models/internlm2-chat-1_8b",
    # proxies=proxies,
    endpoint=endpoint,
)
