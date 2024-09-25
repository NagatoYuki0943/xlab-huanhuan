# copy from https://github.com/NagatoYuki0943/fastapi-learn/blob/main/xx_stream/client.py

import requests
import httpx
import aiohttp
import json


URL = "http://localhost:8000/chat"


def requests_chat(data: dict):
    stream = data["stream"]
    response: requests.Response = requests.post(
        URL, json=data, timeout=60, stream=stream
    )
    if not stream:
        yield response.json()
    else:
        for chunk in response.iter_lines(
            chunk_size=8192, decode_unicode=False, delimiter=b"\n"
        ):
            if chunk:
                decoded = chunk.decode("utf-8")
                output = json.loads(decoded)
                yield output


# help: https://www.perplexity.ai/search/wo-shi-yong-requests-shi-xian-q_g712n3SBObB5xH_2fnMQ
def httpx_sync_chat(data: dict):
    stream = data["stream"]
    with httpx.Client() as client:
        if not stream:
            response: httpx.Response = client.post(URL, json=data, timeout=60)
            yield response.json()
        else:
            with client.stream("POST", URL, json=data, timeout=60) as response:
                chunk: str
                for chunk in response.iter_lines():
                    if chunk:
                        output: dict = json.loads(chunk)
                        yield output


async def httpx_async_chat(data: dict):
    stream = data["stream"]
    async with httpx.AsyncClient() as client:
        if not stream:
            response: httpx.Response = await client.post(URL, json=data, timeout=60)
            yield response.json()
        else:
            async with client.stream("POST", URL, json=data, timeout=60) as response:
                chunk: str
                async for chunk in response.aiter_lines():
                    if chunk:
                        output: dict = json.loads(chunk)
                        yield output


async def aiohttp_async_chat(data: dict):
    stream = data["stream"]

    async with aiohttp.ClientSession() as session:
        async with session.post(URL, json=data, timeout=60) as response:
            if not stream:
                data = await response.text('utf-8')
                yield json.loads(data)
            else:
                # 使用 content.iter_any() 逐块读取响应体
                async for chunk in response.content.iter_any():
                    # 处理每个数据块
                    decoded = chunk.decode('utf-8')
                    output = json.loads(decoded)
                    yield output


async def async_chat(data: dict, func: callable):
    async for output in func(data):
        print(output)


if __name__ == "__main__":
    data = {
        "messages": [{"content": "讲一个猫和老鼠的故事", "role": "user"}],
        "max_new_tokens": 1024,
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
