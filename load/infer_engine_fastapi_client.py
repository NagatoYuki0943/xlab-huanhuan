# copy from https://github.com/NagatoYuki0943/fastapi-learn/blob/main/xx_stream/client.py

import requests
import httpx
import aiohttp
import json


URL = "http://localhost:8000/chat"

api_key = "I AM AN API_KEY"

headers = {
    "accept": "application/json",
    "content-type": "application/json",
    "Authorization": f"Bearer {api_key}",
}


# https://github.com/InternLM/lmdeploy/blob/main/lmdeploy/serve/openai/api_client.py
def requests_chat(data: dict):
    stream: bool = data["stream"]

    response: requests.Response = requests.post(
        URL, json=data, headers=headers, timeout=60, stream=stream
    )
    if not stream:
        yield response.json()
    else:
        chunk: bytes
        for chunk in response.iter_lines(
            chunk_size=8192, decode_unicode=False, delimiter=b"\n\n"
        ):
            if chunk:
                decoded: str = chunk.decode("utf-8")
                yield json.loads(decoded)


# help: https://www.perplexity.ai/search/wo-shi-yong-requests-shi-xian-q_g712n3SBObB5xH_2fnMQ
def httpx_sync_chat(data: dict):
    stream: bool = data["stream"]

    with httpx.Client() as client:
        if not stream:
            response: httpx.Response = client.post(
                URL, json=data, headers=headers, timeout=60
            )
            yield response.json()
        else:
            with client.stream(
                "POST", URL, json=data, headers=headers, timeout=60
            ) as response:
                chunk: str
                for chunk in response.iter_lines():
                    if chunk:
                        yield json.loads(chunk)


async def httpx_async_chat(data: dict):
    stream: bool = data["stream"]

    async with httpx.AsyncClient() as client:
        if not stream:
            response: httpx.Response = await client.post(
                URL, json=data, headers=headers, timeout=60
            )
            yield response.json()
        else:
            async with client.stream(
                "POST", URL, json=data, headers=headers, timeout=60
            ) as response:
                chunk: str
                async for chunk in response.aiter_lines():
                    if chunk:
                        yield json.loads(chunk)


# https://www.perplexity.ai/search/wo-shi-yong-aiohttpshi-xian-mo-6J27VL0aQsGNCykznLPlMw
async def aiohttp_async_chat(data: dict):
    stream: bool = data["stream"]

    async with aiohttp.ClientSession() as session:
        async with session.post(
            URL, json=data, headers=headers, timeout=60
        ) as response:
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
                            yield json.loads(message)


async def async_chat(data: dict, func: callable):
    async for output in func(data):
        print(output)


if __name__ == "__main__":
    data = {
        "messages": [{"role": "user", "content": "讲一个猫和老鼠的故事"}],
        "max_tokens": 1024,
        "temperature": 0.8,
        "top_p": 0.8,
        "top_k": 50,
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
