# copy from https://github.com/NagatoYuki0943/fastapi-learn/blob/main/34-stream/openai_client.py

import os
import requests
import httpx
import aiohttp
import json


URL = "http://localhost:8000/v1/chat/completions"
# URL = "https://api.moonshot.cn/v1/chat/completions"
# URL = "https://api.siliconflow.cn/v1/chat/completions"


"""
设置临时变量

linux:
    export API_KEY="your token"

powershell:
    $env:API_KEY = "your token"
"""
api_key = os.getenv("API_KEY", "I AM AN API_KEY")

# https://www.perplexity.ai/search/xia-mian-shi-yi-ge-http-qing-q-dxXvXy_8TbaGcy3n_EJ6gA
headers = {
    "accept": "application/json",  # Accept头部用于指定客户端能够接受的响应内容类型
    "content-type": "application/json",  # Content-Type头部指定了请求体的媒体类型
    "Authorization": f"Bearer {api_key}",  # Authorization头部用于发送客户端的身份验证凭据, 使用了Bearer令牌认证方式,将API密钥作为访问令牌发送
}


# https://github.com/InternLM/lmdeploy/blob/main/lmdeploy/serve/openai/api_client.py
def requests_chat(data: dict):
    stream: bool = data["stream"]

    response: requests.Response = requests.post(
        URL, json=data, headers=headers, timeout=60, stream=stream
    )
    response.raise_for_status()

    if not stream:
        yield response.json()
    else:
        chunk: bytes
        for chunk in response.iter_lines(
            chunk_size=8192, decode_unicode=False, delimiter=b"\n\n"
        ):
            if chunk:
                decoded: str = chunk.decode("utf-8")
                if decoded.startswith("data: "):
                    decoded = decoded[6:]
                    if decoded.strip() == "[DONE]":
                        continue
                    yield json.loads(decoded)


# help: https://www.perplexity.ai/search/wo-shi-yong-requests-shi-xian-q_g712n3SBObB5xH_2fnMQ
def httpx_sync_chat(data: dict):
    stream: bool = data["stream"]

    with httpx.Client() as client:
        if not stream:
            response: httpx.Response = client.post(
                URL, json=data, headers=headers, timeout=60
            )
            response.raise_for_status()

            yield response.json()
        else:
            with client.stream(
                "POST", URL, json=data, headers=headers, timeout=60
            ) as response:
                response.raise_for_status()

                chunk: str
                for chunk in response.iter_lines():
                    if chunk and chunk.startswith("data: "):
                        chunk = chunk[6:]
                        if chunk.strip() == "[DONE]":
                            continue
                        yield json.loads(chunk)


async def httpx_async_chat(data: dict):
    stream: bool = data["stream"]

    async with httpx.AsyncClient() as client:
        if not stream:
            response: httpx.Response = await client.post(
                URL, json=data, headers=headers, timeout=60
            )
            response.raise_for_status()

            yield response.json()
        else:
            async with client.stream(
                "POST", URL, json=data, headers=headers, timeout=60
            ) as response:
                response.raise_for_status()

                chunk: str
                async for chunk in response.aiter_lines():
                    if chunk and chunk.startswith("data: "):
                        chunk = chunk[6:]
                        if chunk.strip() == "[DONE]":
                            continue
                        yield json.loads(chunk)


# https://www.perplexity.ai/search/wo-shi-yong-aiohttpshi-xian-mo-6J27VL0aQsGNCykznLPlMw
async def aiohttp_async_chat(data: dict):
    stream: bool = data["stream"]

    async with aiohttp.ClientSession() as session:

        async with session.post(
            URL, json=data, headers=headers, timeout=60
        ) as response:
            response.raise_for_status()

            if not stream:
                data: str = await response.text("utf-8")
                yield json.loads(data)
            else:
                chunk: bytes
                buffer = ""
                async for chunk in response.content.iter_any():
                    if chunk:
                        buffer += chunk.decode("utf-8")
                        # openai api returns \n\n as a delimiter for messages
                        while "\n\n" in buffer:
                            message, buffer = buffer.split("\n\n", 1)
                            if message.startswith("data: "):
                                message = message[6:]
                                if message.strip() == "[DONE]":
                                    continue
                                yield json.loads(message)


async def async_chat(data: dict, func: callable):
    async for output in func(data):
        print(output)


if __name__ == "__main__":
    data = {
        # "model": "moonshot-v1-8k",
        "model": "internlm/internlm2_5-7b-chat",
        "messages": [
            {"role": "user", "content": "你是谁"},
            {
                "role": "assistant",
                "content": "我是你的小助手",
                "reference": ["book1", "book3"],
            },
            {"role": "user", "content": "猫和老鼠的作者是谁?"},
        ],
        "max_tokens": 1024,
        "n": 1,
        "temperature": 0.8,
        "top_p": 0.8,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "stream": False,
    }
    data_stream = data.copy()
    data_stream["stream"] = True

    for output in requests_chat(data):
        print(output)

    for output in requests_chat(data_stream):
        print(output)

    print("\n")

    for output in httpx_sync_chat(data):
        print(output)

    for output in httpx_sync_chat(data_stream):
        print(output)

    print("\n")

    import asyncio

    asyncio.run(async_chat(data, httpx_async_chat))
    asyncio.run(async_chat(data_stream, httpx_async_chat))

    print("\n")

    asyncio.run(async_chat(data, aiohttp_async_chat))
    asyncio.run(async_chat(data_stream, aiohttp_async_chat))
