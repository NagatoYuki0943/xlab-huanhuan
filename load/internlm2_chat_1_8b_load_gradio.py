import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import torch
import os
import gradio as gr


print("torch version: ", torch.__version__)
print("transformers version: ", transformers.__version__)
print("gradio version: ", gr.__version__)


# clone 模型
model_path = './models/internlm2-chat-1_8b'
# os.system(f'git clone https://code.openxlab.org.cn/OpenLMLab/internlm2-chat-1.8b {model_path}')
# os.system(f'cd {model_path} && git lfs pull')
quantization = False

# tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)

# 量化
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,                      # 是否在4位精度下加载模型。如果设置为True，则在4位精度下加载模型。
    load_in_8bit=False,
    llm_int8_threshold=6.0,
    llm_int8_has_fp16_weight=False,
    bnb_4bit_compute_dtype=torch.float16,   # 4位精度计算的数据类型。这里设置为torch.float16，表示使用半精度浮点数。
    bnb_4bit_quant_type='nf4',              # 4位精度量化的类型。这里设置为"nf4"，表示使用nf4量化类型。 nf4: 4bit-NormalFloat
    bnb_4bit_use_double_quant=True,         # 是否使用双精度量化。如果设置为True，则使用双精度量化。
)

# 创建模型
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    trust_remote_code=True,
    device_map='auto',
    low_cpu_mem_usage=True, # 是否使用低CPU内存,使用 device_map 参数必须为 True
    quantization_config=quantization_config if quantization else None,
)
model.eval()

# print(model.__class__.__name__) # InternLM2ForCausalLM

print(f"model.device: {model.device}, model.dtype: {model.dtype}")

system_prompt = """You are an AI assistant whose name is InternLM (书生·浦语).
- InternLM (书生·浦语) is a conversational language model that is developed by Shanghai AI Laboratory (上海人工智能实验室). It is designed to be helpful, honest, and harmless.
- InternLM (书生·浦语) can understand and communicate fluently in the language chosen by the user such as English and 中文.
"""
print("system_prompt: ", system_prompt)


def chat(
    query: str,
    history: list | None,  # [['What is the capital of France?', 'The capital of France is Paris.'], ['Thanks', 'You are Welcome']]
    max_new_tokens: int = 1024,
    top_p: float = 0.8,
    temperature: float = 0.8,
    regenerate: bool = False
) -> list:
    history = [] if history is None else history
    # 重新生成时要把最后的query和response弹出,重用query
    if regenerate:
        # 有历史就重新生成,没有历史就返回空
        if len(history) > 0:
            query, _ = history.pop(-1)
        else:
            return history
    else:
        query = query.replace(' ', '')
        if query == None or len(query) < 1:
            return history

    print({"max_new_tokens":  max_new_tokens, "top_p": top_p, "temperature": temperature})

    # https://huggingface.co/internlm/internlm2-chat-1_8b/blob/main/modeling_internlm2.py#L1149
    # chat 调用的 generate
    response, history = model.chat(
        tokenizer = tokenizer,
        query = query,
        history = history,
        streamer = None,
        max_new_tokens = max_new_tokens,
        do_sample = True,
        temperature = temperature,
        top_p = top_p,
        meta_instruction = system_prompt,
    )
    print("chat: ", query, response)

    return history


def regenerate(
    history: list | None,
    max_new_tokens: int = 1024,
    top_p: float = 0.8,
    temperature: float = 0.8,
) -> list:
    """重新生成最后一次对话的内容"""
    return chat("", history, max_new_tokens, top_p, temperature, regenerate=True)


def revocery(history: list | None) -> list:
    """恢复到上一轮对话"""
    history = [] if history is None else history
    if len(history) > 0:
        history.pop(-1)
    return history


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
            chatbot = gr.Chatbot(height=800, show_copy_button=True)

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
                temperature = gr.Slider(
                    minimum=0.01,
                    maximum=1.5,
                    value=0.8,
                    step=0.01,
                    label='Temperature'
                )

            with gr.Row():
                # 创建一个文本框组件，用于输入 prompt。
                query = gr.Textbox(label="Prompt/问题")
                # 创建提交按钮。
                # variant https://www.gradio.app/docs/button
                # scale https://www.gradio.app/guides/controlling-layout
                submit = gr.Button("💬 Chat", variant="primary", scale=0)

            with gr.Row():
                # 创建一个重新生成按钮，用于重新生成当前对话内容。
                regen = gr.Button("🔄 Retry", variant="secondary")
                undo = gr.Button("↩️ Undo", variant="secondary")
                # 创建一个清除按钮，用于清除聊天机器人组件的内容。
                clear = gr.ClearButton(components=[chatbot], value="🗑️ Clear", variant="stop")

        # 回车提交
        query.submit(
            chat,
            inputs=[query, chatbot, max_new_tokens, top_p, temperature],
            outputs=[chatbot]
        )

        # 清空query
        query.submit(
            lambda: gr.Textbox(value=""),
            [],
            [query],
        )

        # 按钮提交
        submit.click(
            chat,
            inputs=[query, chatbot, max_new_tokens, top_p, temperature],
            outputs=[chatbot]
        )

        # 清空query
        submit.click(
            lambda: gr.Textbox(value=""),
            [],
            [query],
        )

        # 重新生成
        regen.click(
            regenerate,
            inputs=[chatbot, max_new_tokens, top_p, temperature],
            outputs=[chatbot]
        )

        # 撤销
        undo.click(
            revocery,
            inputs=[chatbot],
            outputs=[chatbot]
        )

    gr.Markdown("""提醒：<br>
    1. 使用中如果出现异常，将会在文本输入框进行展示，请不要惊慌。<br>
    """)

# threads to consume the request
gr.close_all()

# 设置队列启动，队列最大长度为 100
demo.queue(max_size=100)

# 启动新的 Gradio 应用，设置分享功能为 True，并使用环境变量 PORT1 指定服务器端口。
# demo.launch(share=True, server_port=int(os.environ['PORT1']))
# 直接启动
# demo.launch(server_name="127.0.0.1", server_port=7860)
demo.launch()
