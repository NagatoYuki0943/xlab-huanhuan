import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import torch


print("torch version: ", torch.__version__)
print("transformers version: ", transformers.__version__)


model_dir = "./models/internlm2-chat-1_8b-sft"
adapter_dir = "./work_dirs/internlm2_1_8b_qlora_emo_e3/hf"


# tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False, trust_remote_code=True)

# 创建模型
model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    torch_dtype=torch.float16,
    trust_remote_code=True,
    device_map='auto',
)

# 2种加载adapter的方式
# 1. load adapter https://huggingface.co/docs/transformers/main/zh/peft
# model.load_adapter(adapter_dir)

# 2. https://huggingface.co/docs/peft/v0.9.0/en/package_reference/peft_model#peft.PeftModel.from_pretrained
model = PeftModel.from_pretrained(model, adapter_dir)

# print(model.__class__.__name__) # InternLM2ForCausalLM

print(f"model.device: {model.device}, model.dtype: {model.dtype}")


system_prompt = "现在你是一个心理专家，我有一些心理问题，请你用专业的知识帮我解决。"
print(system_prompt)

history = []
while True:
    query = input("请输入提示: ")
    if query.lower() == "exit":
        break
    # https://github.com/InternLM/lmdeploy/blob/main/lmdeploy/pytorch/modeling/modeling_internlm2.py#L1132
    # chat 调用的 generate
    response, history = model.chat(
        tokenizer = tokenizer,
        query = query,
        meta_instruction = system_prompt,
        history = history,
        max_new_tokens = 2048,
        do_sample = True,
        temperature = 0.8,
        top_p = 0.8,
    )
    # print("history:", history)
    print("回应: ", response)
