https://github.com/InternLM/xtuner

# help

```sh
> xtuner --help
04/12 17:34:13 - mmengine - INFO -
    Arguments received: ['xtuner', '--help']. xtuner commands use the following syntax:

        xtuner MODE MODE_ARGS ARGS

        Where   MODE (required) is one of ('list-cfg', 'copy-cfg', 'log-dataset', 'check-custom-dataset', 'train', 'test', 'chat', 'convert', 'preprocess', 'mmbench', 'eval_refcoco')
                MODE_ARG (optional) is the argument for specific mode
                ARGS (optional) are the arguments for specific command

    Some usages for xtuner commands: (See more by using -h for specific command!)

        1. List all predefined configs:
            xtuner list-cfg
        2. Copy a predefined config to a given path:
            xtuner copy-cfg $CONFIG $SAVE_FILE
        3-1. Fine-tune LLMs by a single GPU:
            xtuner train $CONFIG
        3-2. Fine-tune LLMs by multiple GPUs:
            NPROC_PER_NODE=$NGPUS NNODES=$NNODES NODE_RANK=$NODE_RANK PORT=$PORT ADDR=$ADDR xtuner dist_train $CONFIG $GPUS
        4-1. Convert the pth model to HuggingFace's model:
            xtuner convert pth_to_hf $CONFIG $PATH_TO_PTH_MODEL $SAVE_PATH_TO_HF_MODEL
        4-2. Merge the HuggingFace's adapter to the pretrained base model:
            xtuner convert merge $LLM $ADAPTER $SAVE_PATH
            xtuner convert merge $CLIP $ADAPTER $SAVE_PATH --is-clip
        4-3. Split HuggingFace's LLM to the smallest sharded one:
            xtuner convert split $LLM $SAVE_PATH
        5-1. Chat with LLMs with HuggingFace's model and adapter:
            xtuner chat $LLM --adapter $ADAPTER --prompt-template $PROMPT_TEMPLATE --system-template $SYSTEM_TEMPLATE
        5-2. Chat with VLMs with HuggingFace's model and LLaVA:
            xtuner chat $LLM --llava $LLAVA --visual-encoder $VISUAL_ENCODER --image $IMAGE --prompt-template $PROMPT_TEMPLATE --system-template $SYSTEM_TEMPLATE
        6-1. Preprocess arxiv dataset:
            xtuner preprocess arxiv $SRC_FILE $DST_FILE --start-date $START_DATE --categories $CATEGORIES
        6-2. Preprocess refcoco dataset:
            xtuner preprocess refcoco --ann-path $RefCOCO_ANN_PATH --image-path $COCO_IMAGE_PATH --save-path $SAVE_PATH
        7-1. Log processed dataset:
            xtuner log-dataset $CONFIG
        7-2. Verify the correctness of the config file for the custom dataset:
            xtuner check-custom-dataset $CONFIG
        8. MMBench evaluation:
            xtuner mmbench $LLM --llava $LLAVA --visual-encoder $VISUAL_ENCODER --prompt-template $PROMPT_TEMPLATE --data-path $MMBENCH_DATA_PATH
        9. Refcoco evaluation:
            xtuner eval_refcoco $LLM --llava $LLAVA --visual-encoder $VISUAL_ENCODER --prompt-template $PROMPT_TEMPLATE --data-path $REFCOCO_DATA_PATH
        10. List all dataset formats which are supported in XTuner

    Run special commands:

        xtuner help
        xtuner version

    GitHub: https://github.com/InternLM/xtuner
```

# xtuner list-cfg

```sh
> xtuner list-cfg --help
usage: list_cfg.py [-h] [-p PATTERN]

options:
  -h, --help            show this help message and exit
  -p PATTERN, --pattern PATTERN
                        Pattern for fuzzy matching
```

```sh
> xtuner list-cfg -p internlm2_chat_1_8b
==========================CONFIGS===========================
PATTERN: internlm2_chat_1_8b
-------------------------------
internlm2_chat_1_8b_full_alpaca_e3
internlm2_chat_1_8b_qlora_alpaca_e3
llava_internlm2_chat_1_8b_clip_vit_large_p14_336_e1_gpu8_pretrain
llava_internlm2_chat_1_8b_qlora_clip_vit_large_p14_336_lora_e1_gpu8_finetune
=============================================================
```

# copy-cfg

```sh
> xtuner copy-cfg --help
usage: copy_cfg.py [-h] config_name save_dir

positional arguments:
  config_name  config name
  save_dir     save directory for copied config

options:
  -h, --help   show this help message and exit
```

```sh
> xtuner copy-cfg internlm2_chat_1_8b_qlora_alpaca_e3 ./
Copy to ./internlm2_chat_1_8b_qlora_alpaca_e3_copy.py
```

# 配置文件修改

配置文件介绍

打开配置文件后，我们可以看到整体的配置文件分为五部分：

1. **PART 1 Settings**：涵盖了模型基本设置，如预训练模型的选择、数据集信息和训练过程中的一些基本参数（如批大小、学习率等）。

2. **PART 2 Model & Tokenizer**：指定了用于训练的模型和分词器的具体类型及其配置，包括预训练模型的路径和是否启用特定功能（如可变长度注意力），这是模型训练的核心组成部分。

3. **PART 3 Dataset & Dataloader**：描述了数据处理的细节，包括如何加载数据集、预处理步骤、批处理大小等，确保了模型能够接收到正确格式和质量的数据。

4. **PART 4 Scheduler & Optimizer**：配置了优化过程中的关键参数，如学习率调度策略和优化器的选择，这些是影响模型训练效果和速度的重要因素。

5. **PART 5 Runtime**：定义了训练过程中的额外设置，如日志记录、模型保存策略和自定义钩子等，以支持训练流程的监控、调试和结果的保存。

一般来说我们需要更改的部分其实只包括前三部分，而且修改的主要原因是我们修改了配置文件中规定的模型、数据集。后两部分都是 XTuner 官方帮我们优化好的东西，一般而言只有在魔改的情况下才需要进行修改。

**常用超参**

| 参数名                     | 解释                                                         |
| -------------------------- | ------------------------------------------------------------ |
| **data_path**              | 数据路径或 HuggingFace 仓库名                                |
| **max_length**             | 单条数据最大 Token 数，超过则截断                            |
| **pack_to_max_length**     | 是否将多条短数据拼接到 max_length，提高 GPU 利用率           |
| **accumulative_counts**    | 梯度累积，每多少次 backward 更新一次参数                     |
| **sequence_parallel_size** | 并行序列处理的大小，用于模型训练时的序列并行                 |
| **batch_size**             | 每个设备上的批量大小                                         |
| **dataloader_num_workers** | 数据加载器中工作进程的数量                                   |
| **max_epochs**             | 训练的最大轮数                                               |
| **optim_type**             | 优化器类型，例如 AdamW                                       |
| **lr**                     | 学习率                                                       |
| **betas**                  | 优化器中的 beta 参数，控制动量和平方梯度的移动平均           |
| **weight_decay**           | 权重衰减系数，用于正则化和避免过拟合                         |
| **max_norm**               | 梯度裁剪的最大范数，用于防止梯度爆炸                         |
| **warmup_ratio**           | 预热的比例，学习率在这个比例的训练过程中线性增加到初始学习率 |
| **save_steps**             | 保存模型的步数间隔                                           |
| **save_total_limit**       | 保存的模型总数限制，超过限制时删除旧的模型文件               |
| **prompt_template**        | 模板提示，用于定义生成文本的格式或结构                       |
| ......                     | ......                                                       |

> 如果想把显卡的现存吃满，充分利用显卡资源，可以将 `max_length` 和 `batch_size` 这两个参数调大。

```python
"""
https://zhuanlan.zhihu.com/p/677761706

train:
    xtuner train $CONFIG [other_config]
    ex:
        xtuner train train/internlm2_chat_1_8b_qlora_emo_e3.py --deepspeed deepspeed_zero2

convert:
    xtuner convert pth_to_hf $CONFIG $PATH_TO_PTH_MODEL $SAVE_PATH_TO_HF_MODEL --max-shard-size 2GB

    ex:
        xtuner convert pth_to_hf \
            train/internlm2_chat_1_8b_qlora_emo_e3.py \
            work_dirs/internlm2_chat_1_8b_qlora_emo_e3/epoch_3.pth \
            work_dirs/internlm2_chat_1_8b_qlora_emo_e3/hf \
            --max-shard-size 2GB

merge adapter:
    xtuner convert merge $LLM $ADAPTER $SAVE_PATH --max-shard-size 2GB

    ex:
        xtuner convert merge \
            models/internlm2-chat-1_8b \
            work_dirs/internlm2_chat_1_8b_qlora_emo_e3/hf \
            work_dirs/internlm2_chat_1_8b_qlora_emo_e3/merged \
            --max-shard-size 2GB

chat:
    xtuner chat $LLM --adapter $ADAPTER --bits $BITS --temperature $TEMPERATURE --top-k $TOP_K --top-p $TOP_P --system $SYSTEM_TEXT

    ex:
        xtuner chat \
            models/internlm2-chat-1_8b \
            --adapter work_dirs/internlm2_chat_1_8b_qlora_emo_e3/hf \
            --bits 8 --temperature 0.7 --top-k 50 --top-p 0.9 \
            --system 现在你是一个心理专家，我有一些心理问题，请你用专业的知识帮我解决。

验证数据集是否正确构建:
    xtuner check-custom-dataset $CONFIG

    ex:
        xtuner check-custom-dataset train/internlm2_chat_1_8b_qlora_emo_e3.py
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
from xtuner.utils import PROMPT_TEMPLATE, SYSTEM_TEMPLATE

#######################################################################
#                          PART 1  Settings                           #
#######################################################################
# Model
pretrained_model_name_or_path = './models/internlm2-chat-1_8b'
use_varlen_attn = False

# Data
data_path = './data/emo_1234_xtuner.json'
prompt_template = PROMPT_TEMPLATE.internlm2_chat
max_length = 2048
pack_to_max_length = True

# Scheduler & Optimizer
batch_size = 1  # per_device
accumulative_counts = 16
dataloader_num_workers = 0
max_epochs = 3
optim_type = AdamW
lr = 2e-5
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
SYSTEM = '现在你是一个心理专家，我有一些心理问题，请你用专业的知识帮我解决。'
evaluation_inputs = [
    '医生，我最近对学习完全提不起兴趣，尤其是数学课，一想到要上课或者做作业我就感到无比厌恶和烦躁。',
    '医生，我最近总觉得自己对喜欢的事物失去了兴趣，比如以前我热爱画画，现在拿起画笔就感到焦虑和压力。'
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
        device_map='auto',
        low_cpu_mem_usage=True,                     # 是否使用低CPU内存，使用 device_map 参数必须为 True
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

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=dataloader_num_workers,
    dataset=train_dataset,
    sampler=dict(type=DefaultSampler, shuffle=True),
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
log_processor = dict(by_epoch=by_epoch)
```

# train

```sh
> xtuner train --help
usage: train.py [-h] [--work-dir WORK_DIR] [--deepspeed DEEPSPEED] [--resume RESUME] [--seed SEED] [--cfg-options CFG_OPTIONS [CFG_OPTIONS ...]] [--launcher {none,pytorch,slurm,mpi}] [--local_rank LOCAL_RANK] config

Train LLM

positional arguments:
  config                config file name or path.

options:
  -h, --help            show this help message and exit
  --work-dir WORK_DIR   the dir to save logs and models
  --deepspeed DEEPSPEED
                        the path to the .json file for deepspeed
  --resume RESUME       specify checkpoint path to be resumed from.
  --seed SEED           Random seed for the training
  --cfg-options CFG_OPTIONS [CFG_OPTIONS ...]
                        override some settings in the used config, the key-value pair in xxx=yyy format will be merged into config file. If the value to be overwritten is a list, it should be like key="[a,b]" or key=a,b It also allows nested
                        list/tuple values, e.g. key="[(a,b),(c,d)]" Note that the quotation marks are necessary and that no white space is allowed.
  --launcher {none,pytorch,slurm,mpi}
                        job launcher
  --local_rank LOCAL_RANK, --local-rank LOCAL_RANK
```

> example

```sh
# Fine-tune LLMs by a single GPU:
xtuner train $CONFIG [other_config]

xtuner train train/internlm2_1_8b_qlora_emo_e3.py --deepspeed deepspeed_zero2

# Fine-tune LLMs by multiple GPUs:
NPROC_PER_NODE=$NGPUS NNODES=$NNODES NODE_RANK=$NODE_RANK PORT=$PORT ADDR=$ADDR xtuner dist_train $CONFIG $GPUS
```

## 常规训练

当我们准备好了配置文件好，我们只需要将使用 `xtuner train` 指令即可开始训练。

我们可以通过添加 `--work-dir` 指定特定的文件保存位置，比如说就保存在 `/root/ft/train` 路径下。假如不添加的话模型训练的过程文件将默认保存在 `./work_dirs/internlm2_1_8b_qlora_alpaca_e3_copy` 的位置，就比如说我是在 `/root/ft/train` 的路径下输入该指令，那么我的文件保存的位置就是在 `/root/ft/train/work_dirs/internlm2_1_8b_qlora_alpaca_e3_copy` 的位置下。

```bash
# 指定保存路径
xtuner train /root/ft/config/internlm2_1_8b_qlora_alpaca_e3_copy.py --work-dir /root/ft/train
```

在输入训练完后的文件如下所示：

```
|-- train/
    |-- internlm2_1_8b_qlora_alpaca_e3_copy.py
    |-- iter_600.pth
    |-- last_checkpoint
    |-- iter_768.pth
    |-- iter_300.pth
    |-- 20240406_203957/
        |-- 20240406_203957.log
        |-- vis_data/
            |-- 20240406_203957.json
            |-- eval_outputs_iter_599.txt
            |-- eval_outputs_iter_767.txt
            |-- scalars.json
            |-- eval_outputs_iter_299.txt
            |-- config.py
```

## 使用 deepspeed 来加速训练

https://github.com/InternLM/xtuner/tree/main/xtuner/configs/deepspeed

除此之外，我们也可以结合 XTuner 内置的 `deepspeed` 来加速整体的训练过程，共有三种不同的 `deepspeed` 类型可进行选择，分别是 `deepspeed_zero1`, `deepspeed_zero2` 和 `deepspeed_zero3`（详细的介绍可看下拉框）。

DeepSpeed优化器及其选择方法


DeepSpeed是一个深度学习优化库，由微软开发，旨在提高大规模模型训练的效率和速度。它通过几种关键技术来优化训练过程，包括模型分割、梯度累积、以及内存和带宽优化等。DeepSpeed特别适用于需要巨大计算资源的大型模型和数据集。

在DeepSpeed中，`zero` 代表“ZeRO”（Zero Redundancy Optimizer），是一种旨在降低训练大型模型所需内存占用的优化器。ZeRO 通过优化数据并行训练过程中的内存使用，允许更大的模型和更快的训练速度。ZeRO 分为几个不同的级别，主要包括：

- **deepspeed_zero1**：这是ZeRO的基本版本，它优化了模型参数的存储，使得每个GPU只存储一部分参数，从而减少内存的使用。

- **deepspeed_zero2**：在deepspeed_zero1的基础上，deepspeed_zero2进一步优化了梯度和优化器状态的存储。它将这些信息也分散到不同的GPU上，进一步降低了单个GPU的内存需求。

- **deepspeed_zero3**：这是目前最高级的优化等级，它不仅包括了deepspeed_zero1和deepspeed_zero2的优化，还进一步减少了激活函数的内存占用。这通过在需要时重新计算激活（而不是存储它们）来实现，从而实现了对大型模型极其内存效率的训练。

选择哪种deepspeed类型主要取决于你的具体需求，包括模型的大小、可用的硬件资源（特别是GPU内存）以及训练的效率需求。一般来说：

- 如果你的模型较小，或者内存资源充足，可能不需要使用最高级别的优化。
- 如果你正在尝试训练非常大的模型，或者你的硬件资源有限，使用deepspeed_zero2或deepspeed_zero3可能更合适，因为它们可以显著降低内存占用，允许更大模型的训练。
- 选择时也要考虑到实现的复杂性和运行时的开销，更高级的优化可能需要更复杂的设置，并可能增加一些计算开销。

```bash
# 使用 deepspeed 来加速训练
xtuner train /root/ft/config/internlm2_1_8b_qlora_alpaca_e3_copy.py --work-dir /root/ft/train_deepspeed --deepspeed deepspeed_zero2
```

可以看到，通过 `deepspeed` 来训练后得到的权重文件和原本的权重文件是有所差别的，原本的仅仅是一个 .pth 的文件，而使用了 `deepspeed` 则是一个名字带有 .pth 的文件夹，在该文件夹里保存了两个 .pt 文件。当然这两者在具体的使用上并没有太大的差别，都是可以进行转化并整合。

```
|-- train_deepspeed/
    |-- internlm2_1_8b_qlora_alpaca_e3_copy.py
    |-- zero_to_fp32.py
    |-- last_checkpoint
    |-- iter_600.pth/
        |-- bf16_zero_pp_rank_0_mp_rank_00_optim_states.pt
        |-- mp_rank_00_model_states.pt
    |-- 20240406_220727/
        |-- 20240406_220727.log
        |-- vis_data/
            |-- 20240406_220727.json
            |-- eval_outputs_iter_599.txt
            |-- eval_outputs_iter_767.txt
            |-- scalars.json
            |-- eval_outputs_iter_299.txt
            |-- config.py
    |-- iter_768.pth/
        |-- bf16_zero_pp_rank_0_mp_rank_00_optim_states.pt
        |-- mp_rank_00_model_states.pt
    |-- iter_300.pth/
        |-- bf16_zero_pp_rank_0_mp_rank_00_optim_states.pt
        |-- mp_rank_00_model_states.pt
```

## 模型续训指南

假如我们的模型训练过程中突然被中断了，我们也可以通过在原有指令的基础上加上 `--resume {checkpoint_path}` 来实现模型的继续训练。需要注意的是，这个继续训练得到的权重文件和中断前的完全一致，并不会有任何区别。下面我将用训练了500轮的例子来进行演示。



```bash
# 模型续训
xtuner train /root/ft/config/internlm2_1_8b_qlora_alpaca_e3_copy.py --work-dir /root/ft/train --resume /root/ft/train/iter_600.pth
```

在实测过程中，虽然权重文件并没有发生改变，但是会多一个以时间戳为名的训练过程文件夹保存训练的过程数据。

```
|-- train/
    |-- internlm2_1_8b_qlora_alpaca_e3_copy.py
    |-- iter_600.pth
    |-- last_checkpoint
    |-- iter_768.pth
    |-- iter_300.pth
    |-- 20240406_203957/
        |-- 20240406_203957.log
        |-- vis_data/
            |-- 20240406_203957.json
            |-- eval_outputs_iter_599.txt
            |-- eval_outputs_iter_767.txt
            |-- scalars.json
            |-- eval_outputs_iter_299.txt
            |-- config.py
    |-- 20240406_225723/
        |-- 20240406_225723.log
        |-- vis_data/
            |-- 20240406_225723.json
            |-- eval_outputs_iter_767.txt
            |-- scalars.json
            |-- config.py
```

# convert

## 模型转换 pth_to_hf

```sh
> xtuner convert pth_to_hf --help
usage: pth_to_hf.py [-h] [--fp32] [--max-shard-size MAX_SHARD_SIZE] [--cfg-options CFG_OPTIONS [CFG_OPTIONS ...]] config pth_model save_dir

Convert the pth model to HuggingFace model

positional arguments:
  config                config file name or path.
  pth_model             pth model file
  save_dir              the directory to save HuggingFace model

options:
  -h, --help            show this help message and exit
  --fp32                Save LLM in fp32. If not set, fp16 will be used by default.
  --max-shard-size MAX_SHARD_SIZE
                        Only applicable for LLM. The maximum size for each sharded checkpoint.
  --cfg-options CFG_OPTIONS [CFG_OPTIONS ...]
                        override some settings in the used config, the key-value pair in xxx=yyy format will be merged into config file. If the value to be overwritten is a list, it should be like key="[a,b]" or key=a,b It also allows nested
                        list/tuple values, e.g. key="[(a,b),(c,d)]" Note that the quotation marks are necessary and that no white space is allowed.
```

> example

```sh
xtuner convert pth_to_hf ${配置文件地址} ${权重文件地址} ${转换后模型保存地址} --max-shard-size 2GB

xtuner convert pth_to_hf \
    train/internlm2_1_8b_qlora_emo_e3.py \
    work_dirs/internlm2_1_8b_qlora_emo_e3/epoch_3.pth \
    work_dirs/internlm2_1_8b_qlora_emo_e3/hf \
    --max-shard-size 2GB
```

模型转换的本质其实就是将原本使用 Pytorch 训练出来的模型权重文件转换为目前通用的 Huggingface 格式文件，那么我们可以通过以下指令来实现一键转换。

```sh
# 模型转换
# xtuner convert pth_to_hf ${配置文件地址} ${权重文件地址} ${转换后模型保存地址}
xtuner convert pth_to_hf /root/ft/train/internlm2_1_8b_qlora_alpaca_e3_copy.py /root/ft/train/iter_768.pth /root/ft/huggingface
```

转换完成后，可以看到模型被转换为 Huggingface 中常用的 .bin 格式文件，这就代表着文件成功被转化为 Huggingface 格式了。

```sh
|-- huggingface/
    |-- adapter_config.json
    |-- xtuner_config.py
    |-- adapter_model.bin
    |-- README.md
```

<span style="color: red;">**此时，huggingface 文件夹即为我们平时所理解的所谓 “LoRA 模型文件”**</span>

> 可以简单理解：LoRA 模型文件 = Adapter

除此之外，我们其实还可以在转换的指令中添加几个额外的参数，包括以下两个：

| 参数名                | 解释                                         |
| --------------------- | -------------------------------------------- |
| --fp32                | 代表以fp32的精度开启，假如不输入则默认为fp16 |
| --max-shard-size {GB} | 代表每个权重文件最大的大小（默认为2GB）      |

假如有特定的需要，我们可以在上面的转换指令后进行添加。由于本次测试的模型文件较小，并且已经验证过拟合，故没有添加。假如加上的话应该是这样的：

```sh
xtuner convert pth_to_hf /root/ft/train/internlm2_1_8b_qlora_alpaca_e3_copy.py /root/ft/train/iter_768.pth /root/ft/huggingface --fp32 --max-shard-size 2GB
```

## 模型整合 merge

```sh
> xtuner convert merge --help
usage: merge.py [-h] [--max-shard-size MAX_SHARD_SIZE] [--is-clip] [--device {cuda,cpu,auto}] model_name_or_path adapter_name_or_path save_dir

Merge a HuggingFace adapter to base model

positional arguments:
  model_name_or_path    model name or path
  adapter_name_or_path  adapter name or path
  save_dir              the directory to save the merged model

options:
  -h, --help            show this help message and exit
  --max-shard-size MAX_SHARD_SIZE
                        Only applicable for LLM. The maximum size for each sharded checkpoint.
  --is-clip             Indicate if the model is a clip model
  --device {cuda,cpu,auto}
                        Indicate the device
```

> example

```sh
xtuner convert merge ${NAME_OR_PATH_TO_LLM} ${NAME_OR_PATH_TO_ADAPTER} ${SAVE_PATH} --max-shard-size 2GB

xtuner convert merge \
    models/internlm2-chat-1_8b \
    work_dirs/internlm2_1_8b_qlora_emo_e3/hf \
    work_dirs/internlm2_1_8b_qlora_emo_e3/merged \
    --max-shard-size 2GB
```

对于 LoRA 或者 QLoRA 微调出来的模型其实并不是一个完整的模型，而是一个额外的层（adapter）。那么训练完的这个层最终还是要与原模型进行组合才能被正常的使用。

而对于全量微调的模型（full）其实是不需要进行整合这一步的，因为全量微调修改的是原模型的权重而非微调一个新的 adapter ，因此是不需要进行模型整合的。

<img src="https://github.com/InternLM/Tutorial/assets/108343727/dbb82ca8-e0ef-41db-a8a9-7d6958be6a96" width="300" height="300">

在 XTuner 中也是提供了一键整合的指令，但是在使用前我们需要准备好三个地址，包括原模型的地址、训练好的 adapter 层的地址（转为 Huggingface 格式后保存的部分）以及最终保存的地址。

```sh
# 解决一下线程冲突的 Bug 
export MKL_SERVICE_FORCE_INTEL=1

# 进行模型整合
# xtuner convert merge  ${NAME_OR_PATH_TO_LLM} ${NAME_OR_PATH_TO_ADAPTER} ${SAVE_PATH} 
xtuner convert merge /root/ft/model /root/ft/huggingface /root/ft/final_model
```

那除了以上的三个基本参数以外，其实在模型整合这一步还是其他很多的可选参数，包括：

| 参数名                 | 解释                                                         |
| ---------------------- | ------------------------------------------------------------ |
| --max-shard-size {GB}  | 代表每个权重文件最大的大小（默认为2GB）                      |
| --device {device_name} | 这里指的就是device的名称，可选择的有cuda、cpu和auto，默认为cuda即使用gpu进行运算 |
| --is-clip              | 这个参数主要用于确定模型是不是CLIP模型，假如是的话就要加上，不是就不需要添加 |

> CLIP（Contrastive Language–Image Pre-training）模型是 OpenAI 开发的一种预训练模型，它能够理解图像和描述它们的文本之间的关系。CLIP 通过在大规模数据集上学习图像和对应文本之间的对应关系，从而实现了对图像内容的理解和分类，甚至能够根据文本提示生成图像。
> 在模型整合完成后，我们就可以看到 final_model 文件夹里生成了和原模型文件夹非常近似的内容，包括了分词器、权重文件、配置信息等等。当我们整合完成后，我们就能够正常的调用这个模型进行对话测试了。

整合完成后可以查看在 final_model 文件夹下的内容。

```
|-- final_model/
    |-- tokenizer.model
    |-- config.json
    |-- pytorch_model.bin.index.json
    |-- pytorch_model-00001-of-00002.bin
    |-- tokenization_internlm2.py
    |-- tokenizer_config.json
    |-- special_tokens_map.json
    |-- pytorch_model-00002-of-00002.bin
    |-- modeling_internlm2.py
    |-- configuration_internlm2.py
    |-- tokenizer.json
    |-- generation_config.json
    |-- tokenization_internlm2_fast.py
```

## split

```sh
> xtuner convert split --help 
usage: split.py [-h] src_dir dst_dir

Split a HuggingFace model to the smallest sharded one

positional arguments:
  src_dir     the directory of the model
  dst_dir     the directory to save the new model

options:
  -h, --help  show this help message and exit
```

> example

```sh
# Split HuggingFace's LLM to the smallest sharded one:
xtuner convert split $LLM $SAVE_PATH

xtuner convert split \
    models/internlm2-chat-1_8b \
    work_dirs/internlm2-chat-1_8b-sft-split
```

# chat

```sh
> xtuner chat --help
usage: chat.py [-h] [--adapter ADAPTER | --llava LLAVA] [--visual-encoder VISUAL_ENCODER] [--visual-select-layer VISUAL_SELECT_LAYER] [--image IMAGE] [--torch-dtype {fp16,bf16,fp32,auto}]
               [--prompt-template {default,zephyr,internlm_chat,internlm2_chat,moss_sft,llama2_chat,code_llama_chat,chatglm2,chatglm3,qwen_chat,baichuan_chat,baichuan2_chat,wizardlm,wizardcoder,vicuna,deepseek_coder,deepseekcoder,deepseek_moe,mistral,mixtral,gemma}]
               [--system SYSTEM | --system-template {moss_sft,alpaca,arxiv_gentile,colorist,coder,lawyer,medical,sql}] [--bits {4,8,None}] [--bot-name BOT_NAME] [--with-plugins {calculate,solve,search} [{calculate,solve,search} ...]] [--no-streamer]
               [--lagent] [--stop-words STOP_WORDS [STOP_WORDS ...]] [--offload-folder OFFLOAD_FOLDER] [--max-new-tokens MAX_NEW_TOKENS] [--temperature TEMPERATURE] [--top-k TOP_K] [--top-p TOP_P] [--repetition-penalty REPETITION_PENALTY]
               [--seed SEED]
               model_name_or_path

Chat with a HF model

positional arguments:
  model_name_or_path    Hugging Face model name or path

options:
  -h, --help            show this help message and exit
  --adapter ADAPTER     adapter name or path
  --llava LLAVA         llava name or path
  --visual-encoder VISUAL_ENCODER
                        visual encoder name or path
  --visual-select-layer VISUAL_SELECT_LAYER
                        visual select layer
  --image IMAGE         image
  --torch-dtype {fp16,bf16,fp32,auto}
                        Override the default `torch.dtype` and load the model under a specific `dtype`.
  --prompt-template {default,zephyr,internlm_chat,internlm2_chat,moss_sft,llama2_chat,code_llama_chat,chatglm2,chatglm3,qwen_chat,baichuan_chat,baichuan2_chat,wizardlm,wizardcoder,vicuna,deepseek_coder,deepseekcoder,deepseek_moe,mistral,mixtral,gemma}
                        Specify a prompt template
  --system SYSTEM       Specify the system text
  --system-template {moss_sft,alpaca,arxiv_gentile,colorist,coder,lawyer,medical,sql}
                        Specify a system template
  --bits {4,8,None}     LLM bits
  --bot-name BOT_NAME   Name for Bot
  --with-plugins {calculate,solve,search} [{calculate,solve,search} ...]
                        Specify plugins to use
  --no-streamer         Whether to with streamer
  --lagent              Whether to use lagent
  --stop-words STOP_WORDS [STOP_WORDS ...]
                        Stop words
  --offload-folder OFFLOAD_FOLDER
                        The folder in which to offload the model weights (or where the model weights are already offloaded).
  --max-new-tokens MAX_NEW_TOKENS
                        Maximum number of new tokens allowed in generated text
  --temperature TEMPERATURE
                        The value used to modulate the next token probabilities.
  --top-k TOP_K         The number of highest probability vocabulary tokens to keep for top-k-filtering.
  --top-p TOP_P         If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.
  --repetition-penalty REPETITION_PENALTY
                        The parameter for repetition penalty. 1.0 means no penalty.
  --seed SEED           Random seed for reproducible text generation
```

> example

```sh
# Chat with LLMs with HuggingFace's model and adapter:
xtuner chat $LLM --adapter $ADAPTER --bits $BITS --temperature $TEMPERATURE --top-k $TOP_K --top-p $TOP_P --system $SYSTEM_TEXT

xtuner chat \
    models/internlm2-chat-1_8b \
    --adapter work_dirs/internlm2_1_8b_qlora_emo_e3/hf \
    --bits 8 --temperature 0.7 --top-k 50 --top-p 0.9 \
    --prompt-template internlm2_chat \
    --system "现在你是一个心理专家，我有一些心理问题，请你用专业的知识帮我解决。"
```

```sh
# Chat with VLMs with HuggingFace's model and LLaVA:
xtuner chat $LLM --llava $LLAVA --visual-encoder $VISUAL_ENCODER --image $IMAGE --prompt-template $PROMPT_TEMPLATE --system-template $SYSTEM_TEMPLATE
```

在 XTuner 中也直接的提供了一套基于 transformers 的对话代码，让我们可以直接在终端与 Huggingface 格式的模型进行对话操作。我们只需要准备我们刚刚转换好的模型路径并选择对应的提示词模版（prompt-template）即可进行对话。假如 prompt-template 选择有误，很有可能导致模型无法正确的进行回复。

> 想要了解具体模型的 prompt-template 或者 XTuner 里支持的 prompt-tempolate，可以到 XTuner 源码中的 `xtuner/utils/templates.py` 这个文件中进行查找。

```Bash
# 与模型进行对话
xtuner chat /root/ft/final_model --prompt-template internlm2_chat
```

那对于 `xtuner chat` 这个指令而言，还有很多其他的参数可以进行设置的，包括：

| 启动参数            | 解释                                                         |
| ------------------- | ------------------------------------------------------------ |
| --system            | 指定SYSTEM文本，用于在对话中插入特定的系统级信息             |
| --system-template   | 指定SYSTEM模板，用于自定义系统信息的模板                     |
| **--bits**          | 指定LLM运行时使用的位数，决定了处理数据时的精度              |
| --bot-name          | 设置bot的名称，用于在对话或其他交互中识别bot                 |
| --with-plugins      | 指定在运行时要使用的插件列表，用于扩展或增强功能             |
| **--no-streamer**   | 关闭流式传输模式，对于需要一次性处理全部数据的场景           |
| **--lagent**        | 启用lagent，用于特定的运行时环境或优化                       |
| --command-stop-word | 设置命令的停止词，当遇到这些词时停止解析命令                 |
| --answer-stop-word  | 设置回答的停止词，当生成回答时遇到这些词则停止               |
| --offload-folder    | 指定存放模型权重的文件夹，用于加载或卸载模型权重             |
| --max-new-tokens    | 设置生成文本时允许的最大token数量，控制输出长度              |
| **--temperature**   | 设置生成文本的温度值，较高的值会使生成的文本更多样，较低的值会使文本更确定 |
| --top-k             | 设置保留用于顶k筛选的最高概率词汇标记数，影响生成文本的多样性 |
| --top-p             | 设置累计概率阈值，仅保留概率累加高于top-p的最小标记集，影响生成文本的连贯性 |
| --seed              | 设置随机种子，用于生成可重现的文本内容                       |

除了这些参数以外其实还有一个非常重要的参数就是 `--adapter` ，这个参数主要的作用就是可以在转化后的 adapter 层与原模型整合之前来对该层进行测试。使用这个额外的参数对话的模型和整合后的模型几乎没有什么太多的区别，因此我们可以通过测试不同的权重文件生成的 adapter 来找到最优的 adapter 进行最终的模型整合工作。

```bash
# 使用 --adapter 参数与完整的模型进行对话
xtuner chat /root/ft/model --adapter /root/ft/huggingface --prompt-template internlm2_chat
```

# chcek-custom-dataset

在修改配置文件后，可以运行`xtuner/tools/check_custom_dataset.py`脚本验证数据集是否正确构建。

```sh
xtuner check-custom-dataset $CONFIG
```

其中 `$CONFIG` 是 config 的文件路径。

