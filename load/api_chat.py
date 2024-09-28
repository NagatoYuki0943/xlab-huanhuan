# https://internlm.intern-ai.org.cn/api/document
# https://platform.moonshot.cn/docs/api/chat
import os
from openai import OpenAI
from openai.types.chat.chat_completion import ChatCompletion


"""
设置临时变量

linux:
    export API_KEY="your token"

powershell:
    $env:API_KEY = "your token"
"""
api_key = os.getenv("API_KEY", "I AM AN API_KEY")


client = OpenAI(
    api_key=api_key,  # 此处传token，不带Bearer
    base_url="https://api.moonshot.cn/v1",
    # base_url="https://api.siliconflow.cn/v1/",
)


messages = [{"role": "user", "content": "hello"}]


response: ChatCompletion = client.chat.completions.create(
    messages=messages,
    model="moonshot-v1-8k",
    # model="internlm/internlm2_5-7b-chat",
    max_tokens=1024,
    n=1,  # 为每条输入消息生成多少个结果，默认为 1
    presence_penalty=0.0,  # 存在惩罚，介于-2.0到2.0之间的数字。正值会根据新生成的词汇是否出现在文本中来进行惩罚，增加模型讨论新话题的可能性
    frequency_penalty=0.0,  # 频率惩罚，介于-2.0到2.0之间的数字。正值会根据新生成的词汇在文本中现有的频率来进行惩罚，减少模型一字不差重复同样话语的可能性
    stream=False,
    temperature=0.8,
    top_p=0.8,
)
print(response)
# ChatCompletion(
#     id='chatcmpl-66f77d3d260d34a3a5c2af50',
#     choices=[
#         Choice(
#             finish_reason='stop',
#             index=0,
#             logprobs=None,
#             message=ChatCompletionMessage(
#                 content='Hello! How can I help you today? If you have any questions or need assistance, feel free to ask.',
#                 refusal=None,
#                 role='assistant',
#                 function_call=None,
#                 tool_calls=None
#             )
#         )
#     ],
#     created=1727495485,
#     model='moonshot-v1-8k',
#     object='chat.completion',
#     service_tier=None,
#     system_fingerprint=None,
#     usage=CompletionUsage(completion_tokens=24, prompt_tokens=8, total_tokens=32, completion_tokens_details=None)
# )


for choice in response.choices:
    print(choice)
    # Choice(
    #     finish_reason='stop',
    #     index=0,
    #     logprobs=None,
    #     message=ChatCompletionMessage(
    #         content='Hello! How can I help you today? If you have any questions or need assistance, feel free to ask.',
    #         refusal=None,
    #         role='assistant',
    #         function_call=None,
    #         tool_calls=None
    #     )
    # )
    print(choice.message.content)
    # Hello! How can I help you today? If you have any questions or need assistance, feel free to ask.
