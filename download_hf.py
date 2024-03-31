import os
from huggingface_hub import hf_hub_download, snapshot_download


# 设置环境变量
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# 下载模型库
# os.system('huggingface-cli download --resume-download internlm/internlm2-chat-1_8b --local-dir models/internlm2-chat-1_8b')


# 下载模型库
snapshot_download(
    repo_id="internlm/internlm2-chat-1_8b",
    local_dir="models/internlm2-chat-1_8b",
    resume_download=True,
    # proxies={"https": "http://localhost:7897"},
    max_workers=8,
    endpoint="https://hf-mirror.com",
)


# 下载模型部分文件
if False:
    hf_hub_download(
        repo_id="internlm/internlm2-chat-1_8b",
        filename="config.json",
        local_dir="models/internlm2-chat-1_8b",
        resume_download=True,
        # proxies={"https": "http://localhost:7897"},
        endpoint="https://hf-mirror.com",
    )
