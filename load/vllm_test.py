from transformers import AutoTokenizer
from vllm import LLM, LLMEngine, AsyncLLMEngine, EngineArgs, AsyncEngineArgs, SamplingParams


PRETRAINED_MODEL_NAME_OR_PATH = '../models/internlm2-chat-1_8b'


tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL_NAME_OR_PATH, trust_remote_code = True)
stop_words = [tokenizer.eos_token]
print(stop_words)
stop_token_ids = [tokenizer.eos_token_id]
print(tokenizer.decode(stop_token_ids))


# https://docs.vllm.ai/en/latest/getting_started/examples/offline_inference.html
# https://github.com/vllm-project/vllm/blob/main/vllm/entrypoints/llm.py
# https://github.com/vllm-project/vllm/blob/main/vllm/engine/llm_engine.py
model = LLM(
    model = PRETRAINED_MODEL_NAME_OR_PATH,
    tokenizer = tokenizer, # 可以为 None,会自动从模型路径载入,不能传递实例化好的 tokenizer
    tokenizer_mode = "auto",
    skip_tokenizer_init = False,
    trust_remote_code = True,
    tensor_parallel_size = 1, # The number of GPUs to use for distributed execution with tensor parallelism.
    dtype = "auto",
    quantization = None,      # support "awq", "gptq", "squeezellm", and "fp8" (experimental). If None, we first check the `quantization_config` attribute in the model config file.
    revision = None,
    tokenizer_revision = None,
    seed = 0,
    gpu_memory_utilization = 0.9, # The ratio (between 0 and 1) of GPU memory to reserve for the model weights, activations, and KV cache.
    swap_space = 4,
    enforce_eager = False,
    max_context_len_to_capture = None,
    max_seq_len_to_capture = 8192, # Maximum context len covered by CUDA graphs.
    disable_custom_all_reduce= False,
)


# https://github.com/vllm-project/vllm/blob/main/vllm/engine/arg_utils.py
engine_args = AsyncEngineArgs(
    # Arguments for vLLM engine.
    model = PRETRAINED_MODEL_NAME_OR_PATH,
    served_model_name = None,
    tokenizer = tokenizer,
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
    enable_lora = False,
    max_loras = 1,
    max_lora_rank = 16,
    fully_sharded_loras = False,
    lora_extra_vocab_size = 256,
    lora_dtype = 'auto',
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
engine = AsyncLLMEngine.from_engine_args(engine_args)


sampling_config = SamplingParams(
    n = 1,
    max_tokens = 1024,
    temperature = 0.8,
    top_p = 0.8,
    top_k = 0.8,
    presence_penalty = 0.0,
    frequency_penalty = 0.0,
    repetition_penalty = 1.0,
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

# batch
for response in model.generate(prompts = prompts, sampling_params = sampling_config):
    print(response)

# stream
for response in engine.generate(prompt = prompts[0], sampling_params = sampling_config):
    print(response)
