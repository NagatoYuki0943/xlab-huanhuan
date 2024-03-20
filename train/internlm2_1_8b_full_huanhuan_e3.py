"""
https://zhuanlan.zhihu.com/p/677761706

train:
    xtuner train $CONFIG [other_config]
    ex:
        xtuner train train/internlm2_1_8b_full_huanhuan_e3.py --deepspeed deepspeed_zero2

convert:
    xtuner convert pth_to_hf \
        $CONFIG \
        $PATH_TO_PTH_MODEL \
        $SAVE_PATH_TO_HF_MODEL
    ex:
        xtuner convert pth_to_hf \
            train/internlm2_1_8b_full_huanhuan_e3.py \
            work_dirs/internlm2_1_8b_full_huanhuan_e3/epoch_3.pth \
            work_dirs/internlm2_1_8b_full_huanhuan_e3/hf

merge adapter
    xtuner convert merge \
        $LLM \
        $ADAPTER \
        $SAVE_PATH \
        --max-shard-size 2GB
    ex:
        xtuner convert merge \
            model/internlm2-chat-1_8b-sft \
            work_dirs/internlm2_1_8b_full_huanhuan_e3/hf \
            work_dirs/internlm2_1_8b_full_huanhuan_e3/merged \
            --max-shard-size 2GB
"""


import torch
from datasets import load_dataset
from mmengine.dataset import DefaultSampler
from mmengine.hooks import (CheckpointHook, DistSamplerSeedHook, IterTimerHook,
                            LoggerHook, ParamSchedulerHook)
from mmengine.visualization import Visualizer, LocalVisBackend, TensorboardVisBackend
from mmengine.optim import AmpOptimWrapper, CosineAnnealingLR, LinearLR
from peft import LoraConfig
from torch.optim import AdamW
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig)

from xtuner.dataset import process_hf_dataset
from xtuner.dataset.collate_fns import default_collate_fn
from xtuner.dataset.map_fns import template_map_fn_factory
from xtuner.engine.hooks import (DatasetInfoHook, EvaluateChatHook,
                                 ThroughputHook)
from xtuner.model import SupervisedFinetune
from xtuner.utils import PROMPT_TEMPLATE

#######################################################################
#                          PART 1  Settings                           #
#######################################################################
data_path = './data/huanhuan_xtuner.json'
model_dir = './models/internlm2-chat-1_8b-sft'
work_dir = './work_dirs/internlm2_1_8b_full_huanhuan_e3'

launcher = 'none'

max_epochs = 3
by_epoch = True    # save and log by epoch or by iteration
save_steps = 1
save_total_limit = 5

optim_type = AdamW
lr = 2e-05
betas = (
    0.9,
    0.999,
)
weight_decay = 0
max_norm = 1  # grad clip
accumulative_counts = 16
warmup_ratio = 0.03

batch_size = 1
num_workers = 0
max_length = 512
pack_to_max_length = True

SYSTEM = '现在你要扮演皇帝身边的女人--甄嬛'
evaluation_freq = 500
evaluation_inputs = [
    '你好',
    '小主，敬事房传来消息，说皇上晚上去华妃那儿。',
]
prompt_template = PROMPT_TEMPLATE.internlm2_chat

#######################################################################
#                      PART 2  Model & Tokenizer                      #
#######################################################################
tokenizer = dict(
    type=AutoTokenizer.from_pretrained,
    pretrained_model_name_or_path=model_dir,
    trust_remote_code=True,
    use_fast=False,
    padding_side='right'
)

model = dict(
    type=SupervisedFinetune,
    llm=dict(
        type=AutoModelForCausalLM.from_pretrained,
        pretrained_model_name_or_path=model_dir,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map='auto',
        low_cpu_mem_usage=True,     # 是否使用低CPU内存，使用 device_map 参数必须为 True
    )
)

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
    pack_to_max_length=True
)

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    dataset=train_dataset,
    sampler=dict(type=DefaultSampler, shuffle=True),
    collate_fn=dict(type=default_collate_fn)
)

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
    dtype='float16'
)

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
    dict(type=ThroughputHook),
]

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
resume = None

# Defaults to use random seed and disable `deterministic`
randomness = dict(seed=None, deterministic=False)

# set log processor
log_processor = dict(by_epoch=by_epoch)
