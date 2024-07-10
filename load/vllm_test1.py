from transformers import AutoTokenizer
from vllm import LLM, LLMEngine, SamplingParams
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


# https://docs.vllm.ai/en/latest/getting_started/examples/offline_inference.html
# https://github.com/vllm-project/vllm/blob/main/vllm/entrypoints/llm.py
# https://github.com/vllm-project/vllm/blob/main/vllm/engine/llm_engine.py
model: LLM = LLM(
    model = PRETRAINED_MODEL_NAME_OR_PATH,
    tokenizer = None, # 可以为 None,会自动从模型路径载入,不能传递实例化好的 tokenizer
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
    enable_lora = False,    # enable_lora
)


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


# batch
response: RequestOutput
for response in model.generate(
    prompts = prompts,
    sampling_params = sampling_config,
    use_tqdm = True,
    # lora_request = LoRARequest("emo", 1, ADAPTER_PATH)
):
    print(response)
