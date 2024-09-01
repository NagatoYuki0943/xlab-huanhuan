from load_pipe import load_pipe, LmdeployConfig
from lmdeploy import GenerationConfig
import os


MODEL_PATH = '../models/internlm2_5-1_8b-chat'

SYSTEM_PROMPT = """You are an AI assistant whose name is InternLM (书生·浦语).
- InternLM (书生·浦语) is a conversational language model that is developed by Shanghai AI Laboratory (上海人工智能实验室). It is designed to be helpful, honest, and harmless.
- InternLM (书生·浦语) can understand and communicate fluently in the language chosen by the user such as English and 中文.
"""

LMDEPLOY_CONFIG = LmdeployConfig(
    model_path = MODEL_PATH,
    backend = 'pytorch',
    model_name = 'internlm2',
    model_format = 'hf',
    cache_max_entry_count = 0.8,    # 调整 KV Cache 的占用比例为0.8
    quant_policy = 0,               # KV Cache 量化, 0 代表禁用, 4 代表 4bit 量化, 8 代表 8bit 量化
    system_prompt = SYSTEM_PROMPT,
    deploy_method = 'local'
)

pipe = load_pipe(config=LMDEPLOY_CONFIG)


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
prompts = [[{
    'role': 'user',
    'content': 'Hi, pls intro yourself'
}], [{
    'role': 'user',
    'content': 'Shanghai is'
}]]

# https://github.com/InternLM/lmdeploy/blob/main/lmdeploy/serve/async_engine.py#L274
responses = pipe(prompts=prompts, gen_config=gen_config)
# 放入 [{},{}] 格式返回一个response
# 放入 [] 或者 [[{},{}]] 格式返回一个response列表
for response in responses:
    print(response)
    print('text:', response.text)
    print('generate_token_len:', response.generate_token_len)
    print('input_token_len:', response.input_token_len)
    print('session_id:', response.session_id)
    print('finish_reason:', response.finish_reason)
    print()
# Response(text='你好，我是一款语言模型，我的名字是书生·浦语。我是由上海人工智能实验室开发的，我的设计理念是有用、诚实并且无害。我可以理解并回应英文 和中文的问题，但我不能看、听、尝、触摸、闻、移动、与物理世界交互，也无法感受情感或体验感官输入，但我可以用我自己的方式来帮助人类。', generate_token_len=77, input_token_len=108, session_id=0, finish_reason='stop')
# text: 你好，我是一款语言模型，我的名字是书生·浦语。我是由上海人工智能实验室开发的，我的设计理念是有用、诚实并且无害。我可以理解并回应英文和中文的问题，但我不能看、听、尝、触摸、闻、移动、与物理世界交互，也无法感受情感或体验感官输入，但我可以用我自己的方式来帮助人类。
# generate_token_len: 77
# input_token_len: 108
# session_id: 0
# finish_reason: stop

# Response(text='上海是中国的一座城市，位于中国东部沿海地区，是中国的经济、文化和交通中心之一。它是中国最大的城市之一，拥有许多重要的旅游景点、商业区和文化设施。', generate_token_len=35, input_token_len=105, session_id=1, finish_reason='stop')
# text: 上海是中国的一座城市，位于中国东部沿海地区，是中国的经济、文化和交通中心之一。它是中国最大的城市之一，拥有许多重要的旅游景点、商业区和文化设 施。
# generate_token_len: 35
# input_token_len: 105
# session_id: 1
# finish_reason: stop

# 流式返回处理结果
# for item in pipe.stream_infer(prompts, gen_config=gen_config):
#     print(item)
    # Response(text=' assist', generate_token_len=32, input_token_len=108, session_id=0, finish_reason=None)
    # Response(text='', generate_token_len=38, input_token_len=108, session_id=0, finish_reason='stop')
    # Response(text=' heritage', generate_token_len=49, input_token_len=105, session_id=1, finish_reason=None)
    # Response(text='', generate_token_len=54, input_token_len=105, session_id=1, finish_reason='stop')

    # print(item.text, end='')
    # if item.finish_reason == 'stop':
    #     print()
