from datasets import Dataset
import pandas as pd
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer, BitsAndBytesConfig, GenerationConfig
import torch
from typing import List, Tuple

print("torch version: ", torch.__version__)
print("transformers version: ", transformers.__version__)


MAX_LENGTH = 512    # 分词器会将一个中文字切分为多个token，因此需要放开一些最大长度，保证数据的完整性
data_path = "./data/huanhuan.json"
model_dir = "./model/internlm2-chat-1_8b-sft"
work_dir = "./work_dirs/internlm2_lora_huanhuan"
system_prompt = "现在你要扮演皇帝身边的女人--甄嬛"


## 载入数据
df = pd.read_json(data_path)
ds = Dataset.from_pandas(df)


## 处理数据集
tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False, trust_remote_code=True)

def build_inputs(query: str, history: List[Tuple[str, str]] = [], meta_instruction="我是系统"):
    prompt = ""
    if meta_instruction:
        prompt += f"""<s>[UNUSED_TOKEN_146]system\n{meta_instruction}[UNUSED_TOKEN_145]\n"""
    else:
        prompt += "<s>"
    for record in history:
        prompt += f"""[UNUSED_TOKEN_146]user\n{record[0]}[UNUSED_TOKEN_145]\n[UNUSED_TOKEN_146]assistant\n{record[1]}[UNUSED_TOKEN_145]\n"""
    prompt += f"""[UNUSED_TOKEN_146]user\n{query}[UNUSED_TOKEN_145]\n[UNUSED_TOKEN_146]assistant\n"""
    return prompt

def process_func(example):

    input_ids, attention_mask, labels = [], [], []
    instruction = tokenizer(f"<s>[UNUSED_TOKEN_146]system{system_prompt}[UNUSED_TOKEN_145]\n[UNUSED_TOKEN_146]user{example['instruction']}[UNUSED_TOKEN_145]\n[UNUSED_TOKEN_146]assistant", add_special_tokens=False)  # add_special_tokens 不在开头加 special_tokens
    response = tokenizer(f"{example['output']}[UNUSED_TOKEN_145]<s>", add_special_tokens=False)
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]  # 因为eos token咱们也是要关注的所以 补充为1
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]
    if len(input_ids) > MAX_LENGTH:  # 做一个截断
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

tokenized_id = ds.map(process_func, remove_columns=ds.column_names)


## 创建模型
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,                      # 是否在4位精度下加载模型。如果设置为True，则在4位精度下加载模型。
    load_in_8bit=False,
    llm_int8_threshold=6.0,
    llm_int8_has_fp16_weight=False,
    bnb_4bit_compute_dtype=torch.float16,   # 4位精度计算的数据类型。这里设置为torch.float16，表示使用半精度浮点数。
    bnb_4bit_quant_type='nf4',              # 4位精度量化的类型。这里设置为"nf4"，表示使用nf4量化类型。 nf4: 4bit-NormalFloat
    bnb_4bit_use_double_quant=True,         # 是否使用双精度量化。如果设置为True，则使用双精度量化。
)

model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    torch_dtype=torch.float16,
    trust_remote_code=True,
    device_map='auto',
    low_cpu_mem_usage=True,             # 是否使用低CPU内存，使用 device_map 参数必须为 True
    quantization_config=quantization_config,
)
model.enable_input_require_grads()      # 开启梯度检查点时，要执行该方法

print(f"model.device: {model.device}, model.dtype: {model.dtype}")


## Lora
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training, load_peft_weights

# https://huggingface.co/docs/peft/developer_guides/quantization
model = prepare_model_for_kbit_training(model)

config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,   # 训练模式
    r=64,                   # Lora 秩
    target_modules=['wqkv', 'wo', 'w1', 'w2', 'w3'],
    lora_alpha=16,          # Lora alaph，具体作用参见 Lora 原理
    lora_dropout=0.1,       # Dropout 比例
    bias='none'
)

model = get_peft_model(model, config)

model.print_trainable_parameters()

## 配置训练参数
args = TrainingArguments(
    output_dir=work_dir,
    optim="paged_adamw_32bit",
    learning_rate=1e-5,
    gradient_checkpointing=True,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,  # 4*4=16
    logging_steps=10,
    num_train_epochs=3,
    save_steps=50,
    save_total_limit=3,
    save_on_each_node=True,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_id,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
)

trainer.train()
