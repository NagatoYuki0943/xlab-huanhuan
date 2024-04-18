# https://github.com/dataprofessor/llama2/blob/master/streamlit_app_v2.py
# cmd: streamlit run ./load/internlm2_chat_1_8b_streamlit.py

import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import torch
import streamlit as st
import time


print("torch version: ", torch.__version__)
print("transformers version: ", transformers.__version__)
print("streamlit version: ", st.__version__)


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


# clone 模型
model_path = './models/internlm2-chat-1_8b'
# os.system(f'git clone https://code.openxlab.org.cn/OpenLMLab/internlm2-chat-1.8b {model_path}')
# os.system(f'cd {model_path} && git lfs pull')


# 量化
quantization = False


system_prompt = """You are an AI assistant whose name is InternLM (书生·浦语).
- InternLM (书生·浦语) is a conversational language model that is developed by Shanghai AI Laboratory (上海人工智能实验室). It is designed to be helpful, honest, and harmless.
- InternLM (书生·浦语) can understand and communicate fluently in the language chosen by the user such as English and 中文.
"""
print("system_prompt: ", system_prompt)


@st.cache_resource
def get_model(model_path: str):
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
    return tokenizer, model


tokenizer, model = get_model(model_path)

# App title
st.set_page_config(page_title="🦙💬 Llama 2 Chatbot")

# Store LLM generated responses
if "messages" not in st.session_state.keys():
    print("streamlit init messages")
    st.session_state.messages = []

# chat
def chat(
    prompts: list,
    max_new_tokens: int = 1024,
    top_p: float = 0.8,
    temperature: float = 0.8,
    regenerate: bool = False
) -> str:
    """聊天"""
    # 重新生成时要把最后的query和response弹出,重用query
    if regenerate:
        # 有历史就重新生成,没有历史就返回空
        if len(prompts) > 1:
            prompts.pop(-1)
        else:
            return ""

    print({"max_new_tokens":  max_new_tokens, "top_p": top_p, "temperature": temperature})

    history = []
    if len(history) > 2:
        for i in range(0, len(prompts)-1, 2):
            history.append([prompts[i]["content"], prompts[i+1]["content"]])
    query = prompts[-1]["content"]
    print("history:", history)
    print("query:", query)

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
    print(f"query: {query}; response: {response}\n")

    return response


def regenerate():
    """重新生成"""
    if len(st.session_state.messages) > 0:
        st.session_state.messages.pop(-1)


def undo():
    """恢复到上一轮对话"""
    if len(st.session_state.messages) > 1:
        for i in range(2):
            st.session_state.messages.pop(-1)


# clearn chat history
def clear_chat_history():
    st.session_state.messages = []


def main():
    # Replicate Credentials
    with st.sidebar:
        st.title('🦙💬 Llama 2 Chatbot')
        st.write('This chatbot is created using the open-source Llama 2 LLM model from Meta.')

        st.subheader('Models and parameters')
        selected_model = st.sidebar.selectbox('Choose a Llama2 model', ['Llama2-7B', 'Llama2-13B', 'Llama2-70B'], key='selected_model')

        max_new_tokens = st.sidebar.slider(label='max_new_tokens', min_value=1, max_value=2048, value=1024, step=1)
        top_p = st.sidebar.slider(label='top_p', min_value=0.01, max_value=1.0, value=0.8, step=0.01)
        temperature = st.sidebar.slider(label='temperature', min_value=0.01, max_value=1.5, value=0.8, step=0.01)

        st.subheader('Chat functions')
        st.sidebar.button('🔄 Retry', on_click=regenerate)
        st.sidebar.button('↩️ Undo', on_click=undo)
        st.sidebar.button('🗑️ Clear', on_click=clear_chat_history)

    # Display or clear chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # User-provided prompt
    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

    # Generate a new response if last message is not from assistant
    if len(st.session_state.messages) > 0 and st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            # with st.spinner("Thinking..."):
                response = chat(st.session_state.messages, max_new_tokens, top_p, temperature)
                placeholder = st.empty()
                full_response = ''
                for item in response:
                    full_response += item
                    placeholder.markdown(full_response)
                placeholder.markdown(full_response)

        message = {"role": "assistant", "content": full_response}
        st.session_state.messages.append(message)


if __name__ == "__main__":
    main()
