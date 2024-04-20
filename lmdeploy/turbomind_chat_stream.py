import os
import lmdeploy
from lmdeploy import pipeline, GenerationConfig, TurbomindEngineConfig, ChatTemplateConfig


print("lmdeploy version: ", lmdeploy.__version__)


# clone 模型
model_path = '../models/internlm2-chat-1_8b'
# os.system(f'git clone https://code.openxlab.org.cn/OpenLMLab/internlm2-chat-1.8b {model_path}')
# os.system(f'cd {model_path} && git lfs pull')


system_prompt = """You are an AI assistant whose name is InternLM (书生·浦语).
- InternLM (书生·浦语) is a conversational language model that is developed by Shanghai AI Laboratory (上海人工智能实验室). It is designed to be helpful, honest, and harmless.
- InternLM (书生·浦语) can understand and communicate fluently in the language chosen by the user such as English and 中文.
"""
print("system_prompt: ", system_prompt)


def get_model(model_path: str):
    # 可以直接使用transformers的模型,会自动转换格式
    # https://lmdeploy.readthedocs.io/zh-cn/latest/api/pipeline.html#turbomindengineconfig
    backend_config = TurbomindEngineConfig(
        model_name = 'internlm2',
        model_format = 'hf', # The format of input model. `hf` meaning `hf_llama`, `llama` meaning `meta_llama`, `awq` meaning the quantized model by awq. Default: None. Type: str
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

    # https://lmdeploy.readthedocs.io/zh-cn/latest/_modules/lmdeploy/model.html#ChatTemplateConfig
    chat_template_config = ChatTemplateConfig(
        model_name = 'internlm2',
        system = None,
        meta_instruction = system_prompt,
    )

    # https://lmdeploy.readthedocs.io/zh-cn/latest/api/pipeline.html
    # https://github.com/InternLM/lmdeploy/blob/main/lmdeploy/api.py
    # https://github.com/InternLM/lmdeploy/blob/main/lmdeploy/serve/async_engine.py
    pipe = pipeline(
        model_path = model_path,
        model_name = 'internlm2_chat_1_8b',
        backend_config = backend_config,
        chat_template_config = chat_template_config,
    )

    return pipe


pipe = get_model(model_path)

# https://lmdeploy.readthedocs.io/zh-cn/latest/api/pipeline.html#generationconfig
gen_config = GenerationConfig(
    n = 1,
    max_new_tokens = 1024,
    top_p = 0.8,
    top_k = 40,
    temperature = 0.8,
    repetition_penalty = 1.0,
    ignore_eos = False,
    random_seed = None,
    stop_words = None,
    bad_words = None,
    min_new_tokens = None,
    skip_special_tokens = True,
)

#----------------------------------------------------------------------#
# prompts (List[str] | str | List[Dict] | List[Dict]): a batch of
#     prompts. It accepts: string prompt, a list of string prompts,
#     a chat history in OpenAI format or a list of chat history.
# [
#     {
#         "role": "system",
#         "content": "You are a helpful assistant."
#     },
#     {
#         "role": "user",
#         "content": "What is the capital of France?"
#     },
#     {
#         "role": "assistant",
#         "content": "The capital of France is Paris."
#     },
#     {
#         "role": "user",
#         "content": "Thanks!"
#     },
#     {
#         "role": "assistant",
#         "content": "You are welcome."
#     }
# ]
#----------------------------------------------------------------------#
prompts = []

while True:
    query = input("请输入提示: ")
    query = query.replace(' ', '')
    if query == None or len(query) < 1:
        continue
    if query.lower() == "exit":
        break

    prompts.append(
        {
            'role': 'user',
            'content': query
        }
    )

    print(f"query: {query}; response: ", end="", flush=True)
    response = ""
    # 放入 [{},{}] 格式返回一个response
    # 放入 [] 或者 [[{},{}]] 格式返回一个response列表
    for response_ in pipe.stream_infer(
        prompts = prompts,
        gen_config = gen_config,
        do_preprocess = True,
        adapter_name = None
    ):
        # print(response)
        # Response(text='很高兴', generate_token_len=10, input_token_len=111, session_id=0, finish_reason=None)
        # Response(text='认识', generate_token_len=11, input_token_len=111, session_id=0, finish_reason=None)
        # Response(text='你', generate_token_len=12, input_token_len=111, session_id=0, finish_reason=None)
        print(response_.text, flush=True, end="")
        response += response_.text
    print("\n")
    prompts.append(
        {
            'role': 'assistant',
            'content': response
        }
    )
    print(f"history: {prompts}")
    print("\n")
