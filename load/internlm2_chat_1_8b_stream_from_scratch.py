import torch
from transformers import GenerationConfig
from transformers.generation.streamers import BaseStreamer
from load_model import load_model
import queue
import threading
from typing import Generator, Literal, Sequence, Any


# clone 模型
PRETRAINED_MODEL_NAME_OR_PATH = '../models/internlm2-chat-1_8b'
# os.system(f'git clone https://code.openxlab.org.cn/OpenLMLab/internlm2-chat-1.8b {PRETRAINED_MODEL_NAME_OR_PATH}')
# os.system(f'cd {PRETRAINED_MODEL_NAME_OR_PATH} && git lfs pull')
ADAPTER_PATH = None
# 量化
LOAD_IN_8BIT= False
LOAD_IN_4BIT = False
tokenizer, model = load_model(PRETRAINED_MODEL_NAME_OR_PATH, ADAPTER_PATH, LOAD_IN_8BIT, LOAD_IN_4BIT)

SYSTEM_PROMPT = """You are an AI assistant whose name is InternLM (书生·浦语).
    - InternLM (书生·浦语) is a conversational language model that is developed by Shanghai AI Laboratory (上海人工智能实验室). It is designed to be helpful, honest, and harmless.
    - InternLM (书生·浦语) can understand and communicate fluently in the language chosen by the user such as English and 中文.
    """


# https://huggingface.co/internlm/internlm2-chat-1_8b/blob/main/modeling_internlm2.py#L1136-L1146
def build_inputs(
    tokenizer,
    query: str,
    history: list[tuple[str, str]] | None = None,
    meta_instruction = ""
) -> tuple[str, Sequence]:
    history = [] if history is None else list(history)

    if tokenizer.add_bos_token:
        prompt = ""
    else:
        prompt = tokenizer.bos_token
    if meta_instruction:
        prompt += f"""<|im_start|>system\n{meta_instruction}<|im_end|>\n"""
    for record in history:
        prompt += f"""<|im_start|>user\n{record[0]}<|im_end|>\n<|im_start|>assistant\n{record[1]}<|im_end|>\n"""
    prompt += f"""<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n"""
    return prompt, tokenizer([prompt], return_tensors="pt")


prompt, inputs = build_inputs(tokenizer, "给我讲一个猫和老鼠的小故事", history=[], meta_instruction=SYSTEM_PROMPT)
print(prompt)
inputs = inputs.to(model.device)
print("input_ids: ", inputs["input_ids"])
print("attention_mask: ", inputs["attention_mask"])


# https://huggingface.co/internlm/internlm2-chat-1_8b/blob/main/modeling_internlm2.py#L1148-L1182
@torch.no_grad()
def chat(
    tokenizer,
    query: str,
    history: Sequence | None = None,
    streamer: BaseStreamer | None = None,
    max_new_tokens: int = 1024,
    do_sample: bool = True,
    temperature: float = 0.8,
    top_p: float = 0.8,
    meta_instruction: str = "You are an AI assistant whose name is InternLM (书生·浦语).\n"
    "- InternLM (书生·浦语) is a conversational language model that is developed by Shanghai AI Laboratory (上海人工智能实验室). It is designed to be helpful, honest, and harmless.\n"
    "- InternLM (书生·浦语) can understand and communicate fluently in the language chosen by the user such as English and 中文.",
    **kwargs,
) -> tuple[str, Sequence]:
    history = [] if history is None else list(history)
    _, inputs = build_inputs(tokenizer, query, history, meta_instruction)
    inputs = {k: v.to(model.device) for k, v in inputs.items() if torch.is_tensor(v)}
    # also add end-of-assistant token in eos token id to avoid unnecessary generation
    eos_token_id = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids(["<|im_end|>"])[0]]
    outputs = model.generate(
        **inputs,
        streamer=streamer,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        eos_token_id=eos_token_id,
        **kwargs,
    )
    outputs = outputs[0].cpu().tolist()[len(inputs["input_ids"][0]) :]
    response = tokenizer.decode(outputs, skip_special_tokens=True)
    response = response.split("<|im_end|>")[0]
    history = history + [(query, response)]
    return response, history


# https://huggingface.co/internlm/internlm2-chat-1_8b/blob/main/modeling_internlm2.py#L1184-L1268
@torch.no_grad()
def stream_chat(
    tokenizer,
    query: str,
    history: Sequence | None = None,
    max_new_tokens: int = 1024,
    do_sample: bool = True,
    temperature: float = 0.8,
    top_p: float = 0.8,
    **kwargs,
) -> Generator[tuple[str, Sequence], None, None]:
    """
    Return a generator in format: (response, history)
    Eg.
    ('你好，有什么可以帮助您的吗', [('你好', '你好，有什么可以帮助您的吗')])
    ('你好，有什么可以帮助您的吗？', [('你好', '你好，有什么可以帮助您的吗？')])
    """
    history = [] if history is None else list(history)

    response_queue = queue.Queue(maxsize=20)

    class ChatStreamer(BaseStreamer):
        def __init__(self, tokenizer) -> None:
            super().__init__()
            self.tokenizer = tokenizer
            self.queue = response_queue
            self.query = query
            self.history = history
            self.response = ""
            self.cache = []
            self.received_inputs = False
            self.queue.put((self.response, history + [(self.query, self.response)]))

        def put(self, value):
            if len(value.shape) > 1 and value.shape[0] > 1:
                raise ValueError("ChatStreamer only supports batch size 1")
            elif len(value.shape) > 1:
                value = value[0]

            if not self.received_inputs:
                # The first received value is input_ids, ignore here
                self.received_inputs = True
                return

            self.cache.extend(value.tolist())
            token = self.tokenizer.decode(self.cache, skip_special_tokens=True)
            if token.strip() != "<|im_end|>":
                self.response = self.response + token
                history = self.history + [(self.query, self.response)]
                self.queue.put((self.response, history))
                self.cache = []
            else:
                self.end()

        def end(self):
            self.queue.put(None)

    def stream_producer():
        return chat(
            tokenizer=tokenizer,
            query=query,
            streamer=ChatStreamer(tokenizer=tokenizer),
            history=history,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            **kwargs,
        )

    def consumer():
        producer = threading.Thread(target=stream_producer)
        producer.start()
        while True:
            res = response_queue.get()
            if res is None:
                return
            yield res

    return consumer()


if __name__ == "__main__":
    query = "给我讲一个猫和老鼠的小故事"

    print(f"query: {query}")
    response, history = chat(
        tokenizer = tokenizer,
        query = query,
        history = [],
        streamer = None,
        max_new_tokens = 1024,
        do_sample = True,
        temperature = 0.8,
        top_p = 0.8,
        top_k = 40,
        meta_instruction = SYSTEM_PROMPT
    )
    print(f"response: {response}")
    print("history: ", history)
    print("#" * 100)

    print(f"query: {query}")
    print(f"response: ", end="", flush=True)
    length = 0
    for response, history in stream_chat(
        tokenizer = tokenizer,
        query = query,
        history = [],
        max_new_tokens = 1024,
        top_p = 0.8,
        top_k = 40,
        temperature = 0.8,
    ):
        print(response[length:], flush=True, end="")
        length = len(response)
    print("\n")
    print("history: ", history)
