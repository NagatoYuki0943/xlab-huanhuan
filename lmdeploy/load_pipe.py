import lmdeploy
from lmdeploy import pipeline, PytorchEngineConfig, TurbomindEngineConfig, ChatTemplateConfig
from typing import Literal
from lmdeploy.serve.async_engine import AsyncEngine
from lmdeploy.serve.vl_async_engine import VLAsyncEngine
from loguru import logger
from dataclasses import dataclass


@dataclass
class LmdeployConfig:
    model_path: str
    backend: Literal['turbomind', 'pytorch'] = 'turbomind'
    model_name: str = 'internlm2'
    model_format: Literal['hf', 'llama', 'awq'] = 'hf'
    tp: int = 1                         # Tensor Parallelism.
    max_batch_size: int = 128
    cache_max_entry_count: float = 0.8  # 调整 KV Cache 的占用比例为0.8
    quant_policy: int = 0               # KV Cache 量化, 0 代表禁用, 4 代表 4bit 量化, 8 代表 8bit 量化
    system_prompt: str = """You are an AI assistant whose name is InternLM (书生·浦语).
    - InternLM (书生·浦语) is a conversational language model that is developed by Shanghai AI Laboratory (上海人工智能实验室). It is designed to be helpful, honest, and harmless.
    - InternLM (书生·浦语) can understand and communicate fluently in the language chosen by the user such as English and 中文.
    """
    log_level: Literal['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG'] = 'ERROR'
    deploy_method: Literal['local', 'serve'] = 'local'
    # for server
    server_name: str = '0.0.0.0'
    server_port: int = 23333
    api_keys: list[str] | str | None = None
    ssl: bool = False


def load_pipe(
    config: LmdeployConfig
) -> AsyncEngine | VLAsyncEngine:
    logger.info(f"lmdeploy version: {lmdeploy.__version__}")
    logger.info(f"lmdeploy config: {config}")

    assert config.backend in ['turbomind', 'pytorch'], \
        f"backend must be 'turbomind' or 'pytorch', but got {config.backend}"
    assert config.model_format in ['hf', 'llama', 'awq'], \
        f"model_format must be 'hf' or 'llama' or 'awq', but got {config.model_format}"
    assert config.cache_max_entry_count >= 0.0 and config.cache_max_entry_count <= 1.0, \
        f"cache_max_entry_count must be >= 0.0 and <= 1.0, but got {config.cache_max_entry_count}"
    assert config.quant_policy in [0, 4, 8], f"quant_policy must be 0, 4 or 8, but got {config.quant_policy}"

    if config.backend == 'turbomind':
        # 可以直接使用transformers的模型,会自动转换格式
        # https://lmdeploy.readthedocs.io/zh-cn/latest/api/pipeline.html#turbomindengineconfig
        # https://github.com/InternLM/lmdeploy/blob/main/lmdeploy/messages.py
        backend_config = TurbomindEngineConfig(
            model_name = config.model_name,
            model_format = config.model_format, # The format of input model. `hf` meaning `hf_llama`, `llama` meaning `meta_llama`, `awq` meaning the quantized model by awq. Default: None. Type: str
            tp = config.tp,                     # Tensor Parallelism.
            session_len = None,                 # the max session length of a sequence, default to None
            max_batch_size = config.max_batch_size,
            cache_max_entry_count = config.cache_max_entry_count,
            cache_block_seq_len = 64,
            enable_prefix_caching = False,
            quant_policy = config.quant_policy, # KV Cache 量化, 0 代表禁用, 4 代表 4bit 量化, 8 代表 8bit 量化
            rope_scaling_factor = 0.0,
            use_logn_attn = False,
            download_dir = None,
            revision = None,
            max_prefill_token_num = 8192,
            num_tokens_per_iter = 0,
            max_prefill_iters = 1,
        )
    else:
        # https://lmdeploy.readthedocs.io/zh-cn/latest/api/pipeline.html#pytorchengineconfig
        # https://github.com/InternLM/lmdeploy/blob/main/lmdeploy/messages.py
        backend_config = PytorchEngineConfig(
            model_name = config.model_name,
            tp = config.tp,                     # Tensor Parallelism.
            session_len = None,                 # the max session length of a sequence, default to None
            max_batch_size = config.max_batch_size,
            cache_max_entry_count = config.cache_max_entry_count,
            eviction_type = 'recompute',
            prefill_interval = 16,
            block_size = 64,
            num_cpu_blocks = 0,
            num_gpu_blocks = 0,
            adapters = None,
            max_prefill_token_num = 4096,
            thread_safe = False,
            enable_prefix_caching = False,
            download_dir = None,
            revision = None,
        )
    logger.info(f"lmdeploy backend_config: {backend_config}")

    # https://lmdeploy.readthedocs.io/zh-cn/latest/_modules/lmdeploy/model.html#ChatTemplateConfig
    chat_template_config = ChatTemplateConfig(
        model_name = config.model_name, # All the chat template names: `lmdeploy list`
        system = None,
        meta_instruction = config.system_prompt,
        eosys = None,
        user = None,
        eoh = None,
        assistant = None,
        eoa = None,
        separator = None,
        capability = None,
        stop_words = None,
    )
    logger.info(f"lmdeploy chat_template_config: {chat_template_config}")

    # https://lmdeploy.readthedocs.io/zh-cn/latest/api/pipeline.html
    # https://github.com/InternLM/lmdeploy/blob/main/lmdeploy/api.py
    # https://github.com/InternLM/lmdeploy/blob/main/lmdeploy/serve/async_engine.py
    pipe = pipeline(
        model_path = config.model_path,
        model_name = None,
        backend_config = backend_config,
        chat_template_config = chat_template_config,
        log_level = config.log_level
    )

    return pipe


if __name__ == '__main__':
# clone 模型
    MODEL_PATH = '../models/internlm2_5-1_8b-chat'
    # os.system(f'git clone https://code.openxlab.org.cn/OpenLMLab/internlm2-chat-1.8b {MODEL_PATH}')
    # os.system(f'cd {MODEL_PATH} && git lfs pull')

    SYSTEM_PROMPT = """You are an AI assistant whose name is InternLM (书生·浦语).
    - InternLM (书生·浦语) is a conversational language model that is developed by Shanghai AI Laboratory (上海人工智能实验室). It is designed to be helpful, honest, and harmless.
    - InternLM (书生·浦语) can understand and communicate fluently in the language chosen by the user such as English and 中文.
    """

    LMDEPLOY_CONFIG = LmdeployConfig(
        model_path = MODEL_PATH,
        backend = 'turbomind',
        model_name = 'internlm2',
        model_format = 'hf',
        cache_max_entry_count = 0.8,    # 调整 KV Cache 的占用比例为0.8
        quant_policy = 0,               # KV Cache 量化, 0 代表禁用, 4 代表 4bit 量化, 8 代表 8bit 量化
        system_prompt = SYSTEM_PROMPT,
        deploy_method = 'local'
    )

    pipe = load_pipe(config=LMDEPLOY_CONFIG)
    print(pipe)
