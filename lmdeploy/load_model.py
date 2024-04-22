import lmdeploy
from lmdeploy import pipeline, PytorchEngineConfig, TurbomindEngineConfig, ChatTemplateConfig


def load_model(
    model_path: str,
    backend: str = 'turbomind', # turbomind, pytorch
    model_format: str = 'hf',
    model_name: str = 'internlm2',
    custom_model_name: str = 'internlm2_chat_1_8b',
    system_prompt: str = """You are an AI assistant whose name is InternLM (书生·浦语).
    - InternLM (书生·浦语) is a conversational language model that is developed by Shanghai AI Laboratory (上海人工智能实验室). It is designed to be helpful, honest, and harmless.
    - InternLM (书生·浦语) can understand and communicate fluently in the language chosen by the user such as English and 中文.
    """,
):
    print("lmdeploy version: ", lmdeploy.__version__)

    assert backend in ['turbomind', 'pytorch'], f"backend must be 'turbomind' or 'pytorch', but got {backend}"

    if backend == 'turbomind':
        # 可以直接使用transformers的模型,会自动转换格式
        # https://lmdeploy.readthedocs.io/zh-cn/latest/api/pipeline.html#turbomindengineconfig
        backend_config = TurbomindEngineConfig(
            model_name = model_name,
            model_format = model_format, # The format of input model. `hf` meaning `hf_llama`, `llama` meaning `meta_llama`, `awq` meaning the quantized model by awq. Default: None. Type: str
            tp = 1,
            session_len = 2048,
            max_batch_size = 128,
            cache_max_entry_count = 0.8, # 调整KV Cache的占用比例为0.8
            cache_block_seq_len = 64,
            quant_policy = 0, # 默认为0, 4为开启kvcache int8 量化
            rope_scaling_factor = 0.0,
            use_logn_attn = False,
            download_dir = None,
            revision = None,
            max_prefill_token_num = 8192,
        )
    else:
        # https://lmdeploy.readthedocs.io/zh-cn/latest/api/pipeline.html#pytorchengineconfig
        backend_config = PytorchEngineConfig(
            model_name = model_name,
            tp = 1,
            session_len = 2048,
            max_batch_size = 128,
            cache_max_entry_count = 0.8, # 调整KV Cache的占用比例为0.8
            eviction_type = 'recompute',
            prefill_interval = 16,
            block_size = 64,
            num_cpu_blocks = 0,
            num_gpu_blocks = 0,
            adapters = None,
            max_prefill_token_num = 4096,
            thread_safe = False,
            download_dir = None,
            revision = None,
        )

    # https://lmdeploy.readthedocs.io/zh-cn/latest/_modules/lmdeploy/model.html#ChatTemplateConfig
    chat_template_config = ChatTemplateConfig(
        model_name = model_name,
        system = None,
        meta_instruction = system_prompt,
    )

    # https://lmdeploy.readthedocs.io/zh-cn/latest/api/pipeline.html
    # https://github.com/InternLM/lmdeploy/blob/main/lmdeploy/api.py
    # https://github.com/InternLM/lmdeploy/blob/main/lmdeploy/serve/async_engine.py
    pipe = pipeline(
        model_path = model_path,
        model_name = custom_model_name,
        backend_config = backend_config,
        chat_template_config = chat_template_config,
    )

    return pipe


if __name__ == '__main__':
    # clone 模型
    model_path = '../models/internlm2-chat-1_8b'
    # os.system(f'git clone https://code.openxlab.org.cn/OpenLMLab/internlm2-chat-1.8b {model_path}')
    # os.system(f'cd {model_path} && git lfs pull')

    pipe = load_model(model_path)
