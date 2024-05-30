"""
train:
    xtuner train $CONFIG [other_config]
    ex:
        xtuner train train/llama3_8b_instruct_qlora_huatuo_e3.py --deepspeed deepspeed_zero2

convert:
    xtuner convert pth_to_hf $CONFIG $PATH_TO_PTH_MODEL $SAVE_PATH_TO_HF_MODEL --max-shard-size 2GB

    ex:
        xtuner convert pth_to_hf \
            train/llama3_8b_instruct_qlora_huatuo_e3.py \
            work_dirs/llama3_8b_instruct_qlora_huatuo_e3/epoch_3.pth \
            work_dirs/llama3_8b_instruct_qlora_huatuo_e3/epoch_3_hf \
            --max-shard-size 2GB

merge adapter:
    xtuner convert merge $LLM $ADAPTER $SAVE_PATH --max-shard-size 2GB

    ex:
        xtuner convert merge \
            models/models/Meta-Llama-3-8B-Instruct \
            work_dirs/llama3_8b_instruct_qlora_huatuo_e3/hf \
            work_dirs/llama3_8b_instruct_qlora_huatuo_e3/merged \
            --max-shard-size 2GB

chat:
    xtuner chat $LLM --adapter $ADAPTER --bits $BITS --temperature $TEMPERATURE --top-k $TOP_K --top-p $TOP_P --system $SYSTEM_TEXT

    ex:
        xtuner chat \
            models/models/Meta-Llama-3-8B-Instruct \
            --adapter work_dirs/llama3_8b_instruct_qlora_huatuo_e3/hf \
            --bits 8 --temperature 0.7 --top-k 50 --top-p 0.9 \
            --system '你是医疗保健智能体，名字叫做 "HeathcareAgent"。\n    - "HeathcareAgent" 可以根据自己丰富的医疗知识来回答问题。\n    - "HeathcareAgent" 的回答应该是有益的、诚实的和无害的。\n    - "HeathcareAgent" 可以使用用户选择的语言（如英语和中文）进行理解和交流。'

验证数据集是否正确构建:
    xtuner check-custom-dataset $CONFIG

    ex:
        xtuner check-custom-dataset train/llama3_8b_instruct_qlora_huatuo_e3.py
"""


import torch
from datasets import load_dataset
from mmengine.dataset import DefaultSampler
from mmengine.hooks import (CheckpointHook, DistSamplerSeedHook, IterTimerHook,
                            LoggerHook, ParamSchedulerHook)
from mmengine.visualization import Visualizer, LocalVisBackend, TensorboardVisBackend
from mmengine.optim import AmpOptimWrapper, CosineAnnealingLR, LinearLR
from peft import LoraConfig, TaskType
from torch.optim import AdamW
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig)

from xtuner.dataset import ConcatDataset, process_hf_dataset
from xtuner.dataset.collate_fns import default_collate_fn
from xtuner.dataset.map_fns import openai_map_fn, template_map_fn_factory
from xtuner.engine.hooks import (DatasetInfoHook, EvaluateChatHook,
                                 VarlenAttnArgsToMessageHubHook, ThroughputHook)
from xtuner.engine.runner import TrainLoop
from xtuner.model import SupervisedFinetune
from xtuner.parallel.sequence import SequenceParallelSampler
from xtuner.utils import PROMPT_TEMPLATE, SYSTEM_TEMPLATE

#######################################################################
#                          PART 1  Settings                           #
#######################################################################
# Model
pretrained_model_name_or_path = './models/Meta-Llama-3-8B-Instruct'
use_varlen_attn = False

# Data
data_path1 = './data/Huatuo26M-Lite/Huatuo26M-Lite-markdown-xtuner.json' # 61222
data_path2 = './data/Huatuo26M-Lite/Huatuo26M-Lite-old-xtuner.json'      # 116481
data_path3 = './data/Huatuo26M-Lite/healthcare_format_add_system.jsonl'  # 788
prompt_template = PROMPT_TEMPLATE.llama3_chat
max_length = 2048
pack_to_max_length = True

# parallel
# https://xtuner.readthedocs.io/zh-cn/latest/acceleration/hyper_parameters.html#sequence-parallel-size-accumulative-counts
# https://xtuner.readthedocs.io/zh-cn/latest/acceleration/train_extreme_long_sequence.html
sequence_parallel_size = 1

# Scheduler & Optimizer
batch_size = 1  # per_device
accumulative_counts = 16
accumulative_counts *= sequence_parallel_size
dataloader_num_workers = 0
max_epochs = 3
optim_type = AdamW
lr = 2e-4
betas = (0.9, 0.999)
weight_decay = 0
max_norm = 1  # grad clip
warmup_ratio = 0.03

# Save
by_epoch = True    # save and log by epoch or by iteration
save_steps = 1
save_total_limit = 3  # Maximum checkpoints to keep (-1 means unlimited)

# Evaluate the generation performance during the training
evaluation_freq = 500
SYSTEM = """
你是医疗保健智能体，名字叫做 "HeathcareAgent"。
    - "HeathcareAgent" 可以根据自己丰富的医疗知识来回答问题。
    - "HeathcareAgent" 的回答应该是有益的、诚实的和无害的。
    - "HeathcareAgent" 可以使用用户选择的语言（如英语和中文）进行理解和交流。
"""
evaluation_inputs = [
    '我自去年春天双手起了一些对称性水泡，奇痒还脱皮，一直用药至今不见好。。有些医生说是汗疱疹，有些说是湿疹，用了一些地奈德乳膏，尿素软膏，还有一些中药泡手应该怎样治疗？',
    '一年前我不幸患上肺结核病，经过打针，吃药治疗，已满一年了，近期去医院复查，做了CT,验血生化检测，已经算是没什么了，但怎样才算康复呢，要不要停止吃药？'
]

#######################################################################
#                      PART 2  Model & Tokenizer                      #
#######################################################################
tokenizer = dict(
    type=AutoTokenizer.from_pretrained,
    pretrained_model_name_or_path=pretrained_model_name_or_path,
    trust_remote_code=True,
    padding_side='right')

model = dict(
    type=SupervisedFinetune,
    use_varlen_attn=use_varlen_attn,
    llm=dict(
        type=AutoModelForCausalLM.from_pretrained,
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        # device_map='auto',
        # low_cpu_mem_usage=True,                   # 是否使用低CPU内存，使用 device_map 参数必须为 True
        quantization_config=dict(
            type=BitsAndBytesConfig,
            load_in_4bit=True,                      # 是否在4位精度下加载模型。如果设置为True，则在4位精度下加载模型。
            load_in_8bit=False,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=torch.bfloat16,  # 4位精度计算的数据类型。这里设置为torch.bfloat16，表示使用半精度浮点数。
            bnb_4bit_use_double_quant=True,         # 是否使用双精度量化。如果设置为True，则使用双精度量化。
            bnb_4bit_quant_type='nf4')),            # 4位精度量化的类型。这里设置为"nf4"，表示使用nf4量化类型。 nf4: 4bit-NormalFloat
    lora=dict(
        type=LoraConfig,
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,   # 训练模式
        r=64,                   # Lora 秩
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
        lora_alpha=16,          # Lora alaph，具体作用参见 Lora 原理
        lora_dropout=0.1,       # Dropout 比例
        bias='none'))

#######################################################################
#                      PART 3  Dataset & Dataloader                   #
#######################################################################
train_dataset1 = dict(
    type=process_hf_dataset,
    dataset=dict(
        type=load_dataset, path='json', data_files=dict(train=data_path1)),
    tokenizer=tokenizer,
    max_length=max_length,
    dataset_map_fn=None,
    template_map_fn=dict(
        type=template_map_fn_factory, template=prompt_template),
    remove_unused_columns=True,
    shuffle_before_pack=True,
    pack_to_max_length=pack_to_max_length,
    use_varlen_attn=use_varlen_attn)

train_dataset2 = dict(
    type=process_hf_dataset,
    dataset=dict(
        type=load_dataset, path='json', data_files=dict(train=data_path2)),
    tokenizer=tokenizer,
    max_length=max_length,
    dataset_map_fn=None,
    template_map_fn=dict(
        type=template_map_fn_factory, template=prompt_template),
    remove_unused_columns=True,
    shuffle_before_pack=True,
    pack_to_max_length=pack_to_max_length,
    use_varlen_attn=use_varlen_attn)

train_dataset3 = dict(
    type=process_hf_dataset,
    dataset=dict(
        type=load_dataset, path='json', data_files=dict(train=data_path3)),
    tokenizer=tokenizer,
    max_length=max_length,
    dataset_map_fn=openai_map_fn,
    template_map_fn=dict(
        type=template_map_fn_factory, template=prompt_template),
    remove_unused_columns=True,
    shuffle_before_pack=True,
    pack_to_max_length=pack_to_max_length,
    use_varlen_attn=use_varlen_attn)

train_dataset = dict(type=ConcatDataset, datasets=[train_dataset1, train_dataset2, train_dataset3])

sampler = SequenceParallelSampler \
    if sequence_parallel_size > 1 else DefaultSampler
train_dataloader = dict(
    batch_size=batch_size,
    num_workers=dataloader_num_workers,
    dataset=train_dataset,
    sampler=dict(type=sampler, shuffle=True),
    collate_fn=dict(type=default_collate_fn, use_varlen_attn=use_varlen_attn))

#######################################################################
#                    PART 4  Optimizer & Scheduler                    #
#######################################################################
# optimizer
optim_wrapper = dict(
    type=AmpOptimWrapper,
    optimizer=dict(
        type=optim_type, lr=lr, betas=betas, weight_decay=weight_decay),
    clip_grad=dict(max_norm=max_norm, error_if_nonfinite=False),
    accumulative_counts=accumulative_counts,
    loss_scale='dynamic',
    dtype='float16')

# learning policy
# More information: https://github.com/open-mmlab/mmengine/blob/main/docs/en/tutorials/param_scheduler.md  # noqa: E501
param_scheduler = [
    dict(
        type=LinearLR,
        start_factor=1e-5,
        by_epoch=True,
        begin=0,
        end=warmup_ratio * max_epochs,
        convert_to_iter_based=True),
    dict(
        type=CosineAnnealingLR,
        eta_min=0.0,
        by_epoch=True,
        begin=warmup_ratio * max_epochs,
        end=max_epochs,
        convert_to_iter_based=True)
]

# train, val, test setting
# 在 EpochBased 模式下，val_interval 的默认值为 1，表示训练一个 Epoch，验证一次
# 在 IterBased 模式下，val_interval 的默认值为 1000，表示训练迭代 1000 次，验证一次
# train_cfg = dict(type=TrainLoop, max_epochs=max_epochs)
train_cfg = dict(by_epoch=True, max_epochs=max_epochs, val_interval=1)

#######################################################################
#                           PART 5  Runtime                           #
#######################################################################
# Log the dialogue periodically during the training process, optional
custom_hooks = [
    dict(type=DatasetInfoHook, tokenizer=tokenizer),
    dict(
        type=EvaluateChatHook,
        tokenizer=tokenizer,
        every_n_iters=evaluation_freq,
        evaluation_inputs=evaluation_inputs,
        system=SYSTEM,
        prompt_template=prompt_template),
    dict(type=ThroughputHook)
]

if use_varlen_attn:
    custom_hooks += [dict(type=VarlenAttnArgsToMessageHubHook)]

# configure default hooks
default_hooks = dict(
    # record the time of every iteration.
    timer=dict(type=IterTimerHook),
    # print log every 10 iterations.
    logger=dict(type=LoggerHook, log_metric_by_epoch=False, interval=10),
    # enable the parameter scheduler.
    param_scheduler=dict(type=ParamSchedulerHook),
    # save checkpoint per `save_steps`.
    checkpoint=dict(
        type=CheckpointHook,
        by_epoch=by_epoch,
        interval=save_steps,
        max_keep_ckpts=save_total_limit),
    # set sampler seed in distributed evrionment.
    sampler_seed=dict(type=DistSamplerSeedHook)
)

# configure environment
env_cfg = dict(
    # whether to enable cudnn benchmark
    cudnn_benchmark=False,
    # set multi process parameters
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    # set distributed parameters
    dist_cfg=dict(backend='nccl')
)

# set visualizer
visualizer = dict(
    type=Visualizer,
    vis_backends=[dict(type=LocalVisBackend), dict(type=TensorboardVisBackend)]
)

# set log level
log_level = 'INFO'

# load from which checkpoint
load_from = None

# whether to resume training from the loaded checkpoint
resume = False

# Defaults to use random seed and disable `deterministic`
randomness = dict(seed=None, deterministic=False)

# set log processor
# log_processor = dict(by_epoch=False)
log_processor = dict(by_epoch=by_epoch)
