# https://github.com/vllm-project/vllm/blob/main/vllm/entrypoints/api_server.py
# https://github.com/NagatoYuki0943/fastapi-learn/blob/main/xx_stream/server.py

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from infer_engine import InferEngine, TransformersConfig
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

TRANSFORMERS_CONFIG = TransformersConfig(
    pretrained_model_name_or_path=PRETRAINED_MODEL_NAME_OR_PATH,
    adapter_path=ADAPTER_PATH,
    load_in_8bit=LOAD_IN_8BIT,
    load_in_4bit=LOAD_IN_4BIT,
    model_name="internlm2",
    system_prompt=SYSTEM_PROMPT,
)


# 载入模型
infer_engine: InferEngine = None


def init_engine():
    global infer_engine
    infer_engine = infer_engine or InferEngine(
        backend="transformers",  # transformers, lmdeploy
        transformers_config=TRANSFORMERS_CONFIG,
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
        examples=[[{"role": "user", "content": "你是谁?"}]],
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


class Response(BaseModel):
    response: str = Field(
        None,
        description="Generated text response",
        examples=[
            "InternLM (书生·浦语) is a conversational language model that is developed by Shanghai AI Laboratory (上海人工智能实验室)."
        ],
    )

    def __repr__(self) -> str:
        return self.model_dump_json()


# 将请求体作为 JSON 读取
# 在函数内部，你可以直接访问模型对象的所有属性
# http://127.0.0.1:8000/docs
@app.post("/chat", response_model=Response)
async def chat(request: ChatRequest):
    print(request)

    if not request.messages or len(request.messages) == 0:
        raise HTTPException(status_code=400, detail="No messages provided")

    role = request.messages[-1].get("role", "")
    if role not in ["user", "assistant"]:
        raise HTTPException(status_code=400, detail="Invalid role")

    content = request.messages[-1].get("content", "")
    if not content:
        raise HTTPException(status_code=400, detail="content is empty")

    if request.stream:

        async def generate():
            for response in infer_engine.chat_stream(
                request.messages,
                None,
                request.max_tokens,
                request.temperature,
                request.top_p,
                request.top_k,
                random_uuid_int(),
            ):
                # openai api returns \n\n as a delimiter for messages
                yield Response(response=response).model_dump_json() + "\n\n"

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

    return Response(response=response)


# uvicorn infer_engine_fastapi_server:app --reload --port=8000
# uvicorn main:app --reload --port=8000
#   main: main.py 文件(一个 Python「模块」)。
#   app: 在 main.py 文件中通过 app = FastAPI() 创建的对象。
#   --reload: 让服务器在更新代码后重新启动。仅在开发时使用该选项。
