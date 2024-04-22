from load_model import load_model
from lmdeploy import GenerationConfig
import os


# clone 模型
MODEL_PATH = '../models/internlm2-chat-1_8b'
# os.system(f'git clone https://code.openxlab.org.cn/OpenLMLab/internlm2-chat-1.8b {MODEL_PATH}')
# os.system(f'cd {MODEL_PATH} && git lfs pull')

SYSTEM_PROMPT = """You are an AI assistant whose name is InternLM (书生·浦语).
- InternLM (书生·浦语) is a conversational language model that is developed by Shanghai AI Laboratory (上海人工智能实验室). It is designed to be helpful, honest, and harmless.
- InternLM (书生·浦语) can understand and communicate fluently in the language chosen by the user such as English and 中文.
"""
print("system_prompt: ", SYSTEM_PROMPT)

pipe = load_model(MODEL_PATH, backend='turbomind', system_prompt=SYSTEM_PROMPT)


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
    query = query.strip()
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
