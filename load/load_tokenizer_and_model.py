import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import torch
from dataclasses import dataclass
from loguru import logger


@dataclass
class TransformersConfig:
    pretrained_model_name_or_path: str
    adapter_path: str = None
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    model_name: str = 'internlm2'  # 用于查找对应的对话模板
    system_prompt: str = """You are an AI assistant whose name is InternLM (书生·浦语).
    - InternLM (书生·浦语) is a conversational language model that is developed by Shanghai AI Laboratory (上海人工智能实验室). It is designed to be helpful, honest, and harmless.
    - InternLM (书生·浦语) can understand and communicate fluently in the language chosen by the user such as English and 中文.
    """


def load_tokenizer_and_model(
    config: TransformersConfig,
) -> tuple[AutoTokenizer, AutoModelForCausalLM]:
    logger.info(f"torch version: {torch.__version__}")
    logger.info(f"transformers version: {transformers.__version__}")
    logger.info(f"transformers config: {config}")

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.pretrained_model_name_or_path, trust_remote_code = True)

    # 量化
    quantization_config = BitsAndBytesConfig(
        load_in_4bit = config.load_in_4bit,                # 是否在4位精度下加载模型。如果设置为True，则在4位精度下加载模型。
        load_in_8bit = False if config.load_in_4bit else config.load_in_8bit,
        llm_int8_threshold = 6.0,
        llm_int8_has_fp16_weight = False,
        bnb_4bit_compute_dtype = torch.bfloat16,    # 4位精度计算的数据类型。这里设置为torch.bfloat16，表示使用半精度浮点数。
        bnb_4bit_quant_type = 'nf4',                # 4位精度量化的类型。这里设置为"nf4"，表示使用nf4量化类型。 nf4: 4bit-NormalFloat
        bnb_4bit_use_double_quant = True,           # 是否使用双精度量化。如果设置为True，则使用双精度量化。
    )

    # 创建模型
    model = AutoModelForCausalLM.from_pretrained(
        config.pretrained_model_name_or_path,
        torch_dtype = torch.bfloat16,
        trust_remote_code = True,
        device_map = 'auto',
        low_cpu_mem_usage = True,   # 是否使用低CPU内存,使用 device_map 参数必须为 True
        quantization_config = quantization_config if config.load_in_8bit or config.load_in_4bit else None,
    )

    if config.adapter_path:
        logger.info(f"load adapter: {config.adapter_path}")
        # 2种加载adapter的方式
        # 1. load adapter https://huggingface.co/docs/transformers/main/zh/peft
        # model.load_adapter(adapter_path)
        # 2. https://huggingface.co/docs/peft/main/en/package_reference/peft_model#peft.PeftModel.from_pretrained
        model = PeftModel.from_pretrained(model, config.adapter_path)

    model.eval()

    logger.info(f"model.device: {model.device}, model.dtype: {model.dtype}")


    print(f"model.device: {model.device}, model.dtype: {model.dtype}")
    return tokenizer, model


if __name__ == '__main__':
    # clone 模型
    PRETRAINED_MODEL_NAME_OR_PATH = '../models/internlm2-chat-1_8b'
    # os.system(f'git clone https://code.openxlab.org.cn/OpenLMLab/internlm2-chat-1.8b {PRETRAINED_MODEL_NAME_OR_PATH}')
    # os.system(f'cd {PRETRAINED_MODEL_NAME_OR_PATH} && git lfs pull')
    ADAPTER_PATH = None
    # 量化
    LOAD_IN_8BIT= False
    LOAD_IN_4BIT = False

    SYSTEM_PROMPT = """You are an AI assistant whose name is InternLM (书生·浦语).
    - InternLM (书生·浦语) is a conversational language model that is developed by Shanghai AI Laboratory (上海人工智能实验室). It is designed to be helpful, honest, and harmless.
    - InternLM (书生·浦语) can understand and communicate fluently in the language chosen by the user such as English and 中文.
    """

    TRANSFORMERS_CONFIG = TransformersConfig(
        pretrained_model_name_or_path = PRETRAINED_MODEL_NAME_OR_PATH,
        adapter_path = ADAPTER_PATH,
        load_in_8bit = LOAD_IN_8BIT,
        load_in_4bit = LOAD_IN_4BIT,
        model_name = 'internlm2',
        system_prompt = SYSTEM_PROMPT
    )

    tokenizer, model = load_tokenizer_and_model(config=TRANSFORMERS_CONFIG)
    print(tokenizer)
    print(model)
