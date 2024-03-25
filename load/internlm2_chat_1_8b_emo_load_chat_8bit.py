import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import torch


print("torch version: ", torch.__version__)
print("transformers version: ", transformers.__version__)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model_dir = "./models/internlm2-chat-1_8b"
adapter_dir = "./work_dirs/internlm2_chat_1_8b_qlora_emo_e3_hf/checkpoint-3000"
adapter_dir = "./work_dirs/internlm2_chat_1_8b_qlora_emo_e3/hf"


# tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False, trust_remote_code=True)

# 创建模型
quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
)

model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    torch_dtype=torch.float16,
    trust_remote_code=True,
    device_map='auto',
    low_cpu_mem_usage=True,             # 是否使用低CPU内存,使用 device_map 参数必须为 True
    quantization_config=quantization_config,
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
    query = input("请输入提示:")
    if query.lower() == "exit":
        break
    # https://huggingface.co/internlm/internlm2-chat-1_8b/blob/main/modeling_internlm2.py#L1149
    # chat 调用的 generate
    response, history = model.chat(
        tokenizer = tokenizer,
        query = query,
        history = history,
        streamer = None,
        max_new_tokens = 1024,
        do_sample = True,
        temperature = 0.8,
        top_p = 0.8,
        meta_instruction = system_prompt,
    )
    # print("history:", history)
    print(response)
