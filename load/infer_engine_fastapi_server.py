# https://github.com/vllm-project/vllm/blob/main/vllm/entrypoints/api_server.py
# https://github.com/NagatoYuki0943/fastapi-learn/blob/main/34-stream/openai_server.py

import os
import time
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import random

from infer_engine import InferEngine, ApiConfig
from infer_utils import random_uuid_int


PRETRAINED_MODEL_NAME_OR_PATH = "../models/internlm2_5-1_8b-chat"
ADAPTER_PATH = None
# 量化
LOAD_IN_8BIT = False
LOAD_IN_4BIT = False

SYSTEM_PROMPT = """You are an AI assistant whose name is InternLM (书生·浦语).
- InternLM (书生·浦语) is a conversational language model that is developed by Shanghai AI Laboratory (上海人工智能实验室). It is designed to be helpful, honest, and harmless.
- InternLM (书生·浦语) can understand and communicate fluently in the language chosen by the user such as English and 中文.
"""


"""
设置临时变量

linux:
    export API_KEY="your token"

powershell:
    $env:API_KEY = "your token"
"""
# URL = "http://localhost:8000/v1/"
# URL = "https://api.moonshot.cn/v1/"
URL = "https://api.siliconflow.cn/v1/"

API_KEY = os.getenv("API_KEY", "I AM AN API_KEY")
print(f"API_KEY: {API_KEY}")

api_config = ApiConfig(
    base_url=URL,
    api_key=API_KEY,
    model="internlm/internlm2_5-7b-chat",
)


# 载入模型
infer_engine: InferEngine = None


def init_engine():
    global infer_engine
    infer_engine = infer_engine or InferEngine(
        backend="api",  # transformers, lmdeploy, api
        api_config=api_config,
    )


init_engine()


app = FastAPI()


# 与声明查询参数一样，包含默认值的模型属性是可选的，否则就是必选的。默认值为 None 的模型属性也是可选的。
class ChatRequest(BaseModel):
    model: str | None = Field(
        None,
        description="The model used for generating the response",
        examples=["gpt4o", "gpt4"],
    )
    messages: list[dict[str, str | list]] = Field(
        None,
        description="List of dictionaries containing the input text and the corresponding user id",
        examples=[
            [{"role": "user", "content": "你是谁?"}],
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "图片中有什么内容?"},
                        {
                            "type": "image_url",
                            "image_url": {"url": "https://example.com/image.jpg"},
                        },
                    ],
                }
            ],
        ],
    )
    max_tokens: int = Field(
        1024, ge=1, le=2048, description="Maximum number of new tokens to generate"
    )
    n: int = Field(
        1,
        ge=1,
        le=10,
        description="Number of completions to generate for each prompt",
    )
    temperature: float = Field(
        0.8,
        ge=0.1,
        le=2.0,
        description="Sampling temperature (lower temperature results in less random completions",
    )
    top_p: float = Field(
        0.8,
        ge=0.0,
        le=1.0,
        description="Nucleus sampling top-p (top-p sampling chooses from the smallest possible set of tokens whose cumulative probability mass exceeds the probability top_p)",
    )
    top_k: int = Field(
        50,
        ge=0,
        le=100,
        description="Top-k sampling chooses from the top k tokens with highest probability",
    )
    stream: bool = Field(
        False,
        description="Whether to stream the output or wait for the whole response before returning it",
    )


# -------------------- 非流式响应模型 --------------------#
class ChatCompletionMessage(BaseModel):
    content: str | None = Field(
        None,
        description="The input text of the user or assistant",
        examples=["你是谁?"],
    )
    # 允许添加额外字段
    references: list[str] | None = Field(
        None,
        description="The references text(s) used for generating the response",
        examples=[["book1", "book2"]],
    )
    role: str = Field(
        None,
        description="The role of the user or assistant",
        examples=["system", "user", "assistant"],
    )
    refusal: bool = Field(
        False,
        description="Whether the user or assistant refused to provide a response",
        examples=[False, True],
    )
    function_call: str | None = Field(
        None,
        description="The function call that the user or assistant made",
        examples=["ask_name", "ask_age", "ask_location"],
    )
    tool_calls: str | None = Field(
        None,
        description="The tool calls that the user or assistant made",
        examples=["weather", "calendar", "news"],
    )

    def __repr__(self) -> str:
        return self.model_dump_json()


class ChatCompletionChoice(BaseModel):
    index: int = Field(
        None,
        description="The index of the choice",
        examples=[0, 1, 2],
    )
    finish_reason: str | None = Field(
        None,
        description="The reason for finishing the conversation",
        examples=[None, "stop"],
    )
    logprobs: list[float] | None = Field(
        None,
        description="The log probabilities of the choices",
        examples=[-1.3862943611198906, -1.3862943611198906, -1.3862943611198906],
    )
    message: ChatCompletionMessage | None = Field(
        None,
        description="The message generated by the model",
    )

    def __repr__(self) -> str:
        return self.model_dump_json()


class CompletionUsage(BaseModel):
    prompt_tokens: int = Field(
        0,
        description="The number of tokens in the prompt",
        examples=[10],
    )
    completion_tokens: int = Field(
        0,
        description="The number of tokens in the completion",
        examples=[10],
    )
    total_tokens: int = Field(
        0,
        description="The total number of tokens generated",
        examples=[10],
    )

    def __repr__(self) -> str:
        return self.model_dump_json()


class ChatCompletion(BaseModel):
    id: str | int | None = Field(
        None,
        description="The id of the conversation",
        examples=[123456, "abc123"],
    )
    choices: list[ChatCompletionChoice] = Field(
        [],
        description="The choices generated by the model",
    )
    created: int | float | None = Field(
        None,
        description="The timestamp when the conversation was created",
    )
    model: str | None = Field(
        None,
        description="The model used for generating the response",
        examples=["gpt4o", "gpt4"],
    )
    object: str = Field(
        "chat.completion",
        description="The object of the conversation",
        examples=["chat.completion"],
    )
    service_tier: str | None = Field(
        None,
        description="The service tier of the conversation",
        examples=["basic", "premium"],
    )
    system_fingerprint: str | None = Field(
        None,
        description="The system fingerprint of the conversation",
        examples=["1234567890abcdef"],
    )
    usage: CompletionUsage = Field(
        CompletionUsage(),
        description="The usage of the completion",
    )

    def __repr__(self) -> str:
        return self.model_dump_json()


# -------------------- 非流式响应模型 --------------------#


# -------------------- 流式响应模型 --------------------#
class ChoiceDelta(ChatCompletionMessage): ...


class ChatCompletionChunkChoice(BaseModel):
    index: int = Field(
        None,
        description="The index of the choice",
        examples=[0, 1, 2],
    )
    finish_reason: str | None = Field(
        None,
        description="The reason for finishing the conversation",
        examples=[None, "stop"],
    )
    logprobs: list[float] | None = Field(
        None,
        description="The log probabilities of the choices",
        examples=[-1.3862943611198906, -1.3862943611198906, -1.3862943611198906],
    )
    delta: ChoiceDelta | None = Field(
        None,
        description="The message generated by the model",
    )

    def __repr__(self) -> str:
        return self.model_dump_json()


class ChatCompletionChunk(BaseModel):
    id: str | int | None = Field(
        None,
        description="The id of the conversation",
        examples=[123456, "abc123"],
    )
    choices: list[ChatCompletionChunkChoice] = Field(
        [],
        description="The choices generated by the model",
    )
    created: int | float | None = Field(
        None,
        description="The timestamp when the conversation was created",
    )
    model: str | None = Field(
        None,
        description="The model used for generating the response",
        examples=["gpt4o", "gpt4"],
    )
    object: str = Field(
        "chat.completion.chunk",
        description="The object of the conversation",
        examples=["chat.completion.chunk"],
    )
    service_tier: str | None = Field(
        None,
        description="The service tier of the conversation",
        examples=["basic", "premium"],
    )
    system_fingerprint: str | None = Field(
        None,
        description="The system fingerprint of the conversation",
        examples=["1234567890abcdef"],
    )
    usage: CompletionUsage = Field(
        None,
        description="The usage of the completion",
    )

    def __repr__(self) -> str:
        return self.model_dump_json()


# -------------------- 流式响应模型 --------------------#


# 将请求体作为 JSON 读取
# 在函数内部，你可以直接访问模型对象的所有属性
# http://127.0.0.1:8000/docs
@app.post("/v1/chat/completions", response_model=ChatCompletion)
async def chat(request: ChatRequest):
    print("request: ", request)

    messages = request.messages
    print("messages: ", messages)

    if not messages or len(messages) == 0:
        raise HTTPException(status_code=400, detail="No messages provided")

    role = messages[-1].get("role", "")
    if role not in ["user", "assistant"]:
        raise HTTPException(status_code=400, detail="Invalid role")

    content = messages[-1].get("content", "")
    if not content:
        raise HTTPException(status_code=400, detail="content is empty")
    content_len = len(content)

    session_id = random.getrandbits(64)

    # 流式响应
    if request.stream:

        async def generate():
            response_lens = 0
            for response in infer_engine.chat_stream(
                request.messages,
                None,
                request.max_tokens,
                request.temperature,
                request.top_p,
                request.top_k,
                random_uuid_int(),
            ):
                response_lens += len(response)
                chat_completion_chunk = ChatCompletionChunk(
                    id=session_id,
                    choices=[
                        ChatCompletionChunkChoice(
                            index=0,
                            finish_reason=None,
                            delta=ChoiceDelta(
                                content=response,
                                role="assistant",
                            ),
                        )
                    ],
                    created=time.time(),
                    usage=CompletionUsage(
                        prompt_tokens=content_len,
                        completion_tokens=response_lens,
                        total_tokens=content_len + response_lens,
                    ),
                )
                print(chat_completion_chunk)
                # openai api returns \n\n as a delimiter for messages
                yield f"data: {chat_completion_chunk.model_dump_json()}\n\n"

            chat_completion_chunk = ChatCompletionChunk(
                id=session_id,
                choices=[
                    ChatCompletionChunkChoice(
                        index=0,
                        finish_reason="stop",
                        delta=ChoiceDelta(),
                    )
                ],
                created=time.time(),
                usage=CompletionUsage(
                    prompt_tokens=content_len,
                    completion_tokens=response_lens,
                    total_tokens=content_len + response_lens,
                ),
            )
            print(chat_completion_chunk)
            # openai api returns \n\n as a delimiter for messages
            yield f"data: {chat_completion_chunk.model_dump_json()}\n\n"

            yield "data: [DONE]\n\n"

        return StreamingResponse(generate())

    response = infer_engine.chat(
        request.messages,
        None,
        request.max_tokens,
        request.temperature,
        request.top_p,
        request.top_k,
        random_uuid_int(),
    )

    # 非流式响应
    chat_completion = ChatCompletion(
        id=session_id,
        choices=[
            ChatCompletionChoice(
                index=0,
                finish_reason="stop",
                message=ChatCompletionMessage(
                    content=response,
                    role="assistant",
                ),
            ),
        ],
        created=time.time(),
        usage=CompletionUsage(
            prompt_tokens=content_len,
            completion_tokens=len(response),
            total_tokens=content_len + len(response),
        ),
    )
    print(chat_completion)
    return chat_completion


# uvicorn infer_engine_fastapi_server:app --reload --port=8000
# uvicorn main:app --reload --port=8000
#   main: main.py 文件(一个 Python「模块」)。
#   app: 在 main.py 文件中通过 app = FastAPI() 创建的对象。
#   --reload: 让服务器在更新代码后重新启动。仅在开发时使用该选项。
