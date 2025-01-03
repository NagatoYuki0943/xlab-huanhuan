import lmdeploy
from lmdeploy import serve, GenerationConfig, TurbomindEngineConfig, ChatTemplateConfig
import os


print("lmdeploy version: ", lmdeploy.__version__)


MODEL_PATH = "../models/internlm2_5-1_8b-chat"


if __name__ == "__main__":
    # 可以直接使用transformers的模型,会自动转换格式
    # https://lmdeploy.readthedocs.io/zh-cn/latest/api/pipeline.html#turbomindengineconfig
    backend_config = TurbomindEngineConfig(
        model_name="internlm2",
        model_format="hf",  # The format of input model. `hf` meaning `hf_llama`, `llama` meaning `meta_llama`, `awq` meaning the quantized model by awq. Default: None. Type: str
        tp=1,
        session_len=8192,
        max_batch_size=128,
        cache_max_entry_count=0.5,  # 调整KV Cache的占用比例为0.5
        cache_block_seq_len=64,
        enable_prefix_caching=False,
        quant_policy=0,  # KV Cache 量化, 0 代表禁用, 4 代表 4bit 量化, 8 代表 8bit 量化
        rope_scaling_factor=0.0,
        use_logn_attn=False,
        download_dir=None,
        revision=None,
        max_prefill_token_num=8192,
        num_tokens_per_iter=0,
        max_prefill_iters=1,
    )

    system_prompt = """You are an AI assistant whose name is InternLM (书生·浦语).
    - InternLM (书生·浦语) is a conversational language model that is developed by Shanghai AI Laboratory (上海人工智能实验室). It is designed to be helpful, honest, and harmless.
    - InternLM (书生·浦语) can understand and communicate fluently in the language chosen by the user such as English and 中文.
    """

    # https://lmdeploy.readthedocs.io/zh-cn/latest/_modules/lmdeploy/model.html#ChatTemplateConfig
    chat_template_config = ChatTemplateConfig(
        model_name="internlm2",  # All the chat template names: `lmdeploy list`
        system=None,
        meta_instruction=system_prompt,
    )

    # https://lmdeploy.readthedocs.io/zh-cn/latest/api/pipeline.html
    # https://github.com/InternLM/lmdeploy/blob/main/lmdeploy/api.py
    # https://github.com/InternLM/lmdeploy/blob/main/lmdeploy/serve/openai/api_server.py
    client = serve(
        model_path=MODEL_PATH,
        model_name=None,
        backend="turbomind",
        backend_config=backend_config,
        chat_template_config=chat_template_config,
        server_name="0.0.0.0",
        server_port=23333,
        log_level="ERROR",
        api_keys=None,
        ssl=False,
    )
    # 防止进程退出
    while True:
        client
