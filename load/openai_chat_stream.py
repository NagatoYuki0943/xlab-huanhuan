# https://internlm.intern-ai.org.cn/api/document
# https://platform.moonshot.cn/docs/api/chat
import os
import openai
from openai import OpenAI, Stream


"""
设置临时变量

linux:
    export API_KEY="your token"

powershell:
    $env:API_KEY = "your token"
"""
API_KEY = os.getenv("API_KEY", "I AM AN API_KEY")
print(f"API_KEY: {API_KEY}")

client = OpenAI(
    api_key=API_KEY,  # 此处传token，不带Bearer
    # base_url="https://api.moonshot.cn/v1",
    base_url="https://api.siliconflow.cn/v1/",
)


messages = [{"role": "user", "content": "猫和老鼠的作者是谁?"}]


try:
    response: Stream = client.chat.completions.create(
        messages=messages,
        # model="moonshot-v1-8k",
        model="internlm/internlm2_5-7b-chat",
        max_tokens=1024,
        n=1,  # 为每条输入消息生成多少个结果，默认为 1
        presence_penalty=0.0,  # 存在惩罚，介于-2.0到2.0之间的数字。正值会根据新生成的词汇是否出现在文本中来进行惩罚，增加模型讨论新话题的可能性
        frequency_penalty=0.0,  # 频率惩罚，介于-2.0到2.0之间的数字。正值会根据新生成的词汇在文本中现有的频率来进行惩罚，减少模型一字不差重复同样话语的可能性
        stream=True,
        temperature=0.8,
        top_p=0.8,
    )
except openai.APIError as e:
    print(f"OpenAI API返回错误: {e}")
except Exception as e:
    print(f"发生其他错误: {e}")
else:
    print(response)
    # <openai.Stream object at 0x000002294D483AF0>

    responses = []
    # print("response: ", end="", flush=True)
    for idx, chunk in enumerate(response):
        print(chunk)
        # ChatCompletionChunk(
        #     id='chatcmpl-66f77c34b90ececb38615406',
        #     choices=[
        #         Choice(
        #             delta=ChoiceDelta(content='.', function_call=None, refusal=None, role=None, tool_calls=None),
        #             finish_reason=None,
        #             index=0,
        #             logprobs=None
        #         )
        #     ],
        #     created=1727495220,
        #     model='moonshot-v1-8k',
        #     object='chat.completion.chunk',
        #     service_tier=None,
        #     system_fingerprint='fpv0_e61aef71',
        #     usage=None
        # )
        # ChatCompletionChunk(
        #     id='chatcmpl-66f77c34b90ececb38615406',
        #     choices=[
        #         Choice(
        #             delta=ChoiceDelta(content=None, function_call=None, refusal=None, role=None, tool_calls=None),
        #             finish_reason='stop',
        #             index=0,
        #             logprobs=None,
        #             usage={'prompt_tokens': 8, 'completion_tokens': 24, 'total_tokens': 32}
        #         )
        #     ],
        #     created=1727495220,
        #     model='moonshot-v1-8k',
        #     object='chat.completion.chunk',
        #     service_tier=None,
        #     system_fingerprint='fpv0_e61aef71',
        #     usage=None
        # )

        chunk_message = chunk.choices[0].delta
        if not chunk_message.content:
            continue
        content = chunk_message.content

        # print(content, end="", flush=True)
        responses.append(content)

    print("\ncomplete response: ", "".join(responses))
