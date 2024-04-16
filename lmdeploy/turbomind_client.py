import os
import lmdeploy
from lmdeploy import client, GenerationConfig, TurbomindEngineConfig, ChatTemplateConfig


if __name__ == "__main__":
    # https://github.com/InternLM/lmdeploy/blob/main/lmdeploy/api.py
    # https://github.com/InternLM/lmdeploy/blob/main/lmdeploy/serve/openai/api_client.py
    clientt = client(
        api_server_url = 'http://0.0.0.0:23333',
        api_key = None,
    )

    prompt = "你是谁?"

    text, tokens, finish_reason = clientt.chat(
        prompt,
        session_id = None,
        request_output_len = 512,
        stream = False,
        top_p = 0.8,
        top_k = 40,
        temperature = 0.8,
        repetition_penalty = 1.0,
        ignore_eos = False
    )
    print(text)
    print(tokens)
    print(finish_reason)
