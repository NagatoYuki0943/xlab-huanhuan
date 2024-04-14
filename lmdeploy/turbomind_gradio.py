# https://github.com/InternLM/lmdeploy/blob/main/lmdeploy/serve/gradio/vl.py
import os
import gradio as gr
from lmdeploy import pipeline, GenerationConfig, TurbomindEngineConfig, ChatTemplateConfig


model_path = './models/internlm2-chat-1_8b'
# os.system(f'git clone https://code.openxlab.org.cn/OpenLMLab/internlm2-chat-1.8b {model_path}')
# os.system(f'cd {model_path} && git lfs pull')

# 可以直接使用transformers的模型,会自动转换格式
# https://lmdeploy.readthedocs.io/zh-cn/latest/api/pipeline.html#turbomindengineconfig
backend_config = TurbomindEngineConfig(
    model_name = 'internlm2',
    model_format = 'hf', # The format of input model. `hf` meaning `hf_llama`, `llama` meaning `meta_llama`, `awq` meaning the quantized model by awq. Default: None. Type: str
    tp = 1,
    session_len = 2048,
    max_batch_size = 128,
    cache_max_entry_count = 0.4, # 调整KV Cache的占用比例为0.4
    cache_block_seq_len = 64,
    quant_policy = 0, # 默认为0, 4为开启kvcache int8 量化
    rope_scaling_factor = 0.0,
    use_logn_attn = False,
    download_dir = None,
    revision = None,
    max_prefill_token_num = 8192,
)

system_prompt = """You are an AI assistant whose name is InternLM (书生·浦语).
- InternLM (书生·浦语) is a conversational language model that is developed by Shanghai AI Laboratory (上海人工智能实验室). It is designed to be helpful, honest, and harmless.
- InternLM (书生·浦语) can understand and communicate fluently in the language chosen by the user such as English and 中文.
"""

# https://lmdeploy.readthedocs.io/zh-cn/latest/_modules/lmdeploy/model.html#ChatTemplateConfig
chat_template_config = ChatTemplateConfig(
    model_name = 'internlm2',
    system = None,
    meta_instruction = system_prompt,
)

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

# https://lmdeploy.readthedocs.io/zh-cn/latest/api/pipeline.html
# https://github.com/InternLM/lmdeploy/blob/main/lmdeploy/api.py
pipe = pipeline(
    model_path = model_path,
    model_name = 'internlm2_chat_1_8b',
    backend_config = backend_config,
    chat_template_config = chat_template_config,
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


def chat(
    query: str,
    history: list,  # [['What is the capital of France?', 'The capital of France is Paris.'], ['Thanks', 'You are Welcome']]
    max_new_tokens: int = 1024,
    top_p: float = 0.8,
    top_k: int = 40,
    temperature: float = 0.8
) -> tuple[str, list]:
    global gen_config

    query = query.replace(' ', '')
    if query == None or len(query) < 1:
        return "", history

    # 将历史记录转换为openai格式
    history_t = []
    for user, assistant in history:
        history_t.append(
            {
                "role": "user",
                "content": user
            }
        )
        history_t.append(
            {
                "role": "assistant",
                "content": assistant
            })
    # 需要添加当前的query
    history_t.append(
        {
            "role": "user",
            "content": query
        }
    )

    # 修改生成参数
    gen_config.max_new_tokens = max_new_tokens
    gen_config.top_p = top_p
    gen_config.top_k = top_k
    gen_config.temperature = temperature
    print("gen_config: ", gen_config)

    # 放入 [{},{}] 格式返回一个response
    # 放入 [] 或者 [[{},{}]] 格式返回一个response列表
    response = pipe(history_t, gen_config=gen_config).text

    print("chat: ", query, response)

    history.append([query, response])
    return "", history


block = gr.Blocks()
with block as demo:
    with gr.Row(equal_height=True):
        with gr.Column(scale=15):
            gr.Markdown("""<h1><center>InternLM</center></h1>
                <center>InternLM2</center>
                """)
        # gr.Image(value=LOGO_PATH, scale=1, min_width=10,show_label=False, show_download_button=False)

    with gr.Row():
        with gr.Column(scale=4):
            # 创建聊天框
            chatbot = gr.Chatbot(height=450, show_copy_button=True)
            # 创建一个文本框组件，用于输入 prompt。
            query = gr.Textbox(label="Prompt/问题")

            with gr.Row():
                max_new_tokens = gr.Slider(
                    minimum=1,
                    maximum=2048,
                    value=1024,
                    step=1,
                    label='Maximum new tokens'
                )
                top_p = gr.Slider(
                    minimum=0.01,
                    maximum=1,
                    value=0.8,
                    step=0.01,
                    label='Top_p'
                )
                top_k = gr.Slider(
                    minimum=1,
                    maximum=100,
                    value=40,
                    step=1,
                    label='Top_k'
                )
                temperature = gr.Slider(
                    minimum=0.01,
                    maximum=1.5,
                    value=0.8,
                    step=0.01,
                    label='Temperature'
                )

            with gr.Row():
                # 创建提交按钮。
                btn = gr.Button("Chat")

            with gr.Row():
                # 创建一个清除按钮，用于清除聊天机器人组件的内容。
                clear = gr.ClearButton(components=[chatbot], value="Clear console")

        btn.click(
            chat,
            inputs=[query, chatbot, max_new_tokens, top_p, top_k, temperature],
            outputs=[query, chatbot]
        )

# threads to consume the request
gr.close_all()

# 启动新的 Gradio 应用，设置分享功能为 True，并使用环境变量 PORT1 指定服务器端口。
# demo.launch(share=True, server_port=int(os.environ['PORT1']))
# 直接启动
demo.launch()
