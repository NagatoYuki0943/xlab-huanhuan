"""
https://zhuanlan.zhihu.com/p/677761706

train:
    xtuner train $CONFIG [other_config]
    ex:
        xtuner train train/internlm2_chat_1_8b_qlora_car_e100.py --deepspeed deepspeed_zero2

convert:
    xtuner convert pth_to_hf $CONFIG $PATH_TO_PTH_MODEL $SAVE_PATH_TO_HF_MODEL --max-shard-size 2GB

    ex:
        xtuner convert pth_to_hf \
            train/internlm2_chat_1_8b_qlora_car_e100.py \
            work_dirs/internlm2_chat_1_8b_qlora_car_e100/epoch_100.pth \
            work_dirs/internlm2_chat_1_8b_qlora_car_e100/epoch_100_hf \
            --max-shard-size 2GB

merge adapter:
    xtuner convert merge $LLM $ADAPTER $SAVE_PATH --max-shard-size 2GB

    ex:
        xtuner convert merge \
            models/internlm2-chat-1_8b \
            work_dirs/internlm2_chat_1_8b_qlora_car_e100/hf \
            work_dirs/internlm2_chat_1_8b_qlora_car_e100/merged \
            --max-shard-size 2GB

chat:
    xtuner chat $LLM --adapter $ADAPTER --bits $BITS --temperature $TEMPERATURE --top-k $TOP_K --top-p $TOP_P --system $SYSTEM_TEXT

    ex:
        xtuner chat \
            models/internlm2-chat-1_8b \
            --adapter work_dirs/internlm2_chat_1_8b_qlora_car_e100/hf \
            --bits 8 --temperature 0.7 --top-k 50 --top-p 0.9 \
            --system "你现在是评论总结小助手，负责总结用户对汽车的评论，要按照以下顺序输出评论的各个内容\n1、首先输出这个车的整体评价，要求言简意赅。\n2、然后输出这个车的好的评价，要求言简意赅。\n3、之后输出这个车的不好的评价，要求言简意赅。\n4、最后输出这个车的每个部分的评价，有多少提取多少。\n注意，只总结用户的输入的信息，不要自己编造用户没说的信息，以下是用户的评论，请进行总结\n"

验证数据集是否正确构建:
    xtuner check-custom-dataset $CONFIG

    ex:
        xtuner check-custom-dataset train/internlm2_chat_1_8b_qlora_car_e100.py
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

from xtuner.dataset import process_hf_dataset
from xtuner.dataset.collate_fns import default_collate_fn
from xtuner.dataset.map_fns import template_map_fn_factory
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
pretrained_model_name_or_path = './models/internlm2-chat-1_8b'
use_varlen_attn = False

# Data
data_path = './data/car.json'
prompt_template = PROMPT_TEMPLATE.internlm2_chat
max_length = 2048
pack_to_max_length = True

# parallel
sequence_parallel_size = 1

# Scheduler & Optimizer
batch_size = 1  # per_device
accumulative_counts = 16
accumulative_counts *= sequence_parallel_size
dataloader_num_workers = 0
max_epochs = 100
optim_type = AdamW
lr = 2e-5
betas = (0.9, 0.999)
weight_decay = 0
max_norm = 1  # grad clip
warmup_ratio = 0.03

# Save
by_epoch = True    # save and log by epoch or by iteration
save_steps = 10
save_total_limit = 10  # Maximum checkpoints to keep (-1 means unlimited)

# Evaluate the generation performance during the training
evaluation_freq = 500
SYSTEM = "你现在是评论总结小助手，负责总结用户对汽车的评论，要按照以下顺序输出评论的各个内容\n" + \
    "1、首先输出这个车的整体评价，要求言简意赅。\n" + \
    "2、然后输出这个车的好的评价，要求言简意赅。\n" + \
    "3、之后输出这个车的不好的评价，要求言简意赅。\n" + \
    "4、最后输出这个车的每个部分的评价，有多少提取多少。\n" + \
    "注意，只总结用户的输入的信息，不要自己编造用户没说的信息，以下是用户的评论，请进行总结\n"
evaluation_inputs = [
    '这个车看起来尺寸应该不算太小，就是这个中网太大了，这个前灯还是挺喜欢的，有三条竖线，还WEYlogo这个搭配还挺统一的，前灯和下边这个雾灯分体式的设计挺有意思;侧面一看挺修长的，还是溜背的，但侧窗看起来挺窄，感觉后排的空间挺小，虽然这样设计视觉上挺酷的，但乘坐起来应该挺有封闭感，这个轱辘倒是挺大，给人感觉跑起来应该动力十足;一看后边最吸引我的是拳头大小的排气筒，感觉动力应该非常足，也看到和前灯类似的也是三竖线设计，感觉设计语言保持了一致，这点我很满意。',
    '坐在前排与越野车没什么区别，配置丰富，四轮碟刹 车身稳定 半坡辅助 定速巡航 语音控制 主驾驶电动座椅 座椅加热 ，国产皮卡中配置算丰富的，4D20D发动机，最大功率105 最大扭矩315，转速只要不掉2000那是相当可以！3000斤，2600转，动力那是相当到位！横向空间略窄，再宽个10公分就完美了，后排储物空间几乎没有前排空间舒适，后排空间略窄，横向空间略窄！！电子油门有延迟，超车，上坡不能低于2000转！2.0T柴油机都不用算油耗，它能高到哪里去！多功能方向盘，悬浮式中控屏，主驾电动座椅，主副驾座椅加热，这是加分项，左敲敲，右敲敲，尽然全是硬塑料，硬塑料就硬塑料吧！多用途货车，最重要的就是它的多用途！拉货，坐人，越野都有了，还要什么自行车！！！'
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
        torch_dtype=torch.float16,
        # device_map='auto',
        # low_cpu_mem_usage=True,                   # 是否使用低CPU内存，使用 device_map 参数必须为 True
        quantization_config=dict(
            type=BitsAndBytesConfig,
            load_in_4bit=True,                      # 是否在4位精度下加载模型。如果设置为True，则在4位精度下加载模型。
            load_in_8bit=False,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=torch.float16,   # 4位精度计算的数据类型。这里设置为torch.float16，表示使用半精度浮点数。
            bnb_4bit_use_double_quant=True,         # 是否使用双精度量化。如果设置为True，则使用双精度量化。
            bnb_4bit_quant_type='nf4')),            # 4位精度量化的类型。这里设置为"nf4"，表示使用nf4量化类型。 nf4: 4bit-NormalFloat
    lora=dict(
        type=LoraConfig,
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,   # 训练模式
        r=64,                   # Lora 秩
        target_modules=['wqkv', 'wo', 'w1', 'w2', 'w3'],
        lora_alpha=16,          # Lora alaph，具体作用参见 Lora 原理
        lora_dropout=0.1,       # Dropout 比例
        bias='none'))

#######################################################################
#                      PART 3  Dataset & Dataloader                   #
#######################################################################
train_dataset = dict(
    type=process_hf_dataset,
    dataset=dict(
        type=load_dataset, path='json', data_files=dict(train=data_path)),
    tokenizer=tokenizer,
    max_length=max_length,
    dataset_map_fn=None,
    template_map_fn=dict(
        type=template_map_fn_factory, template=prompt_template),
    remove_unused_columns=True,
    shuffle_before_pack=True,
    pack_to_max_length=pack_to_max_length,
    use_varlen_attn=use_varlen_attn)

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
