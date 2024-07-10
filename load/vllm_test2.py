import asyncio
from transformers import AutoTokenizer
from vllm import AsyncLLMEngine, EngineArgs, AsyncEngineArgs, SamplingParams
from vllm.outputs import RequestOutput
# https://docs.vllm.ai/en/latest/models/lora.html
from vllm.lora.request import LoRARequest
from infer_utils import random_uuid



PRETRAINED_MODEL_NAME_OR_PATH = '../models/internlm2-chat-1_8b'
ADAPTER_PATH = None


tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL_NAME_OR_PATH, trust_remote_code = True)
stop_words = [tokenizer.eos_token]
print(stop_words)
stop_token_ids = [tokenizer.eos_token_id]
print(tokenizer.decode(stop_token_ids))


# https://github.com/vllm-project/vllm/blob/main/vllm/engine/arg_utils.py
engine_args = AsyncEngineArgs(
    # Arguments for vLLM engine.
    model = PRETRAINED_MODEL_NAME_OR_PATH,
    served_model_name = None,
    tokenizer = None,
    skip_tokenizer_init = False,
    tokenizer_mode = 'auto',
    trust_remote_code = True,
    download_dir = None,
    load_format = 'auto',
    dtype = 'auto',
    kv_cache_dtype = 'auto',
    quantization_param_path = None,
    seed = 0,
    max_model_len = None,
    worker_use_ray = False,
    pipeline_parallel_size = 1,
    tensor_parallel_size = 1,
    max_parallel_loading_workers = None,
    block_size = 16,
    enable_prefix_caching = False,
    use_v2_block_manager = False,
    swap_space = 4,  # GiB
    gpu_memory_utilization = 0.90,
    max_num_batched_tokens = None,
    max_num_seqs = 256,
    max_logprobs = 5,  # OpenAI default value
    disable_log_stats = False,
    revision = None,
    code_revision = None,
    tokenizer_revision = None,
    quantization = None,
    enforce_eager = False,
    max_context_len_to_capture = None,
    max_seq_len_to_capture = 8192,
    disable_custom_all_reduce = False,
    tokenizer_pool_size = 0,
    tokenizer_pool_type = "ray",
    tokenizer_pool_extra_config = None,
    enable_lora = False, # enable_lora
    max_loras = 1,
    max_lora_rank = 16,
    fully_sharded_loras = False,
    lora_extra_vocab_size = 256,
    # lora_dtype = 'auto',
    max_cpu_loras = None,
    device = 'auto',
    ray_workers_use_nsight = False,
    num_gpu_blocks_override = None,
    num_lookahead_slots = 0,
    model_loader_extra_config = None,

    # Related to Vision-language models such as llava
    image_input_type = None,
    image_token_id = None,
    image_input_shape = None,
    image_feature_size = None,
    scheduler_delay_factor = 0.0,
    enable_chunked_prefill = False,

    guided_decoding_backend = 'outlines',
    # Speculative decoding configuration.
    speculative_model = None,
    num_speculative_tokens = None,
    speculative_max_model_len = None,
    speculative_disable_by_batch_size = None,
    ngram_prompt_lookup_max = None,
    ngram_prompt_lookup_min = None,

    # Arguments for asynchronous vLLM engine.
    engine_use_ray = False,
    disable_log_requests = False,
    max_log_len = None,
)
# https://docs.vllm.ai/en/latest/dev/engine/async_llm_engine.html
# https://github.com/vllm-project/vllm/blob/main/vllm/engine/async_llm_engine.py
# https://github.com/vllm-project/vllm/blob/main/vllm/entrypoints/api_server.py
engine: AsyncLLMEngine = AsyncLLMEngine.from_engine_args(engine_args)


# https://github.com/vllm-project/vllm/blob/main/vllm/sampling_params.py
sampling_config: SamplingParams = SamplingParams(
    n = 1,
    max_tokens = 1024,
    temperature = 0.8,
    top_p = 0.8,
    top_k = 0.8,
    presence_penalty = 0.0,     # 存在惩罚，介于-2.0到2.0之间的数字。正值会根据新生成的词汇是否出现在文本中来进行惩罚，增加模型讨论新话题的可能性
    frequency_penalty = 0.0,    # 频率惩罚，介于-2.0到2.0之间的数字。正值会根据新生成的词汇在文本中现有的频率来进行惩罚，减少模型一字不差重复同样话语的可能性
    repetition_penalty = 1.0,   # 用于控制模型在生成文本时对新词和重复词的偏好，通过调整这个参数，可以影响生成文本的多样性和重复性。
                                # 如果设置的值 > 1，那么模型会更倾向于生成新的词，因为对重复词的惩罚增加，从而鼓励使用不同的词汇。
                                # 如果设置的值 < 1，那么模型会更倾向于重复已经生成的词，因为对新词的惩罚减少，从而鼓励重复使用词汇。
    length_penalty = 1.0,
    skip_special_tokens = True,
    stop = stop_words,
    stop_token_ids = stop_token_ids
)

# Sample prompts.
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]


async def main() -> None:
    # stream
    response: RequestOutput
    async for response in engine.generate(
        inputs = prompts[0],
        sampling_params = sampling_config,
        request_id = random_uuid('str'),  # unique id
        # lora_request = LoRARequest("emo", 1, ADAPTER_PATH)
    ):
        print(response)


asyncio.run(main())
