# https://internlm.intern-ai.org.cn/api/document
# https://platform.moonshot.cn/docs/api/chat
import os
from openai import OpenAI
from openai.types.chat.chat_completion import ChatCompletion
import random
from loguru import logger

"""
设置临时变量

linux:
    export API_KEY="your token"

powershell:
    $env:API_KEY = "your token"
"""
api_key = os.getenv("API_KEY", "I AM AN API_KEY")
print(f"{ api_key = }")


class Formater:
    def __init__(self) -> None:

        self.client = OpenAI(
            api_key = api_key,  # 此处传token，不带Bearer
            # base_url = "https://internlm-chat.intern-ai.org.cn/puyu/api/v1/",
            # base_url = "https://api.moonshot.cn/v1",
            base_url = "http://0.0.0.0:23333/v1" # lmdeploy 本地端口
        )

    def format_answer(self, answer: str) -> str:

        messages = [
            {"role": "system", "content": "你是一个文档格式化小助手，你很擅长将文档转换为markdown格式"},
            {"role": "user", "content": "请将以下医学问题的答案格式化为Markdown格式，保持内容的专业性不变，同时使格式更加美观，易于阅读。回答时直接返回格式化的句子，不需要其他说明。\n{answer}".format(answer=answer)},
        ]

        try:
            response: ChatCompletion = self.client.chat.completions.create(
                messages = messages,
                model = "internlm2_20b_chat",
                max_tokens = 2000,
                n = 1,  # 为每条输入消息生成多少个结果，默认为 1
                presence_penalty = 0.0,     # 存在惩罚，介于-2.0到2.0之间的数字。正值会根据新生成的词汇是否出现在文本中来进行惩罚，增加模型讨论新话题的可能性
                frequency_penalty = 0.0,    # 频率惩罚，介于-2.0到2.0之间的数字。正值会根据新生成的词汇在文本中现有的频率来进行惩罚，减少模型一字不差重复同样话语的可能性
                stream = False,
                temperature = 0.8,
                top_p = 0.8,
                seed = random.randint(1, 1e9),
            )

            choice = response.choices[0]
            return 1, choice.message.content
        except:
            return 0, answer


if __name__ == "__main__":
    answer = "肝腹水的症状表现会随着病情的发展而不同，早期症状可能比较轻微，而晚期症状则比较严重。腹水出现前常有腹胀，大量水使腹部膨隆、腹壁绷紧发高亮，状如蛙腹，患者行走困难，有时膈显著抬高，出现呼吸和脐疝。部分患者伴有胸水，多见于右侧。晚期症状包括面色多而且黝黑，出现黄疸，腹壁静脉怒张，全身发热，内分泌功能失调，经常性的行走困难，腹水不易消退。肝硬化患者要引起注意，及时治疗非常重要。"

    formater = Formater()
    formated_answer = formater.format_answer(answer)
    print(formated_answer[1])
