https://github.com/InternLM/xtuner



# help

```sh
> xtuner --help
03/20 13:57:38 - mmengine - INFO -
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

# train

```sh
> xtuner train --help
usage: train.py [-h] [--work-dir WORK_DIR] [--deepspeed DEEPSPEED] [--resume RESUME] [--seed SEED] [--cfg-options CFG_OPTIONS [CFG_OPTIONS ...]]
                [--launcher {none,pytorch,slurm,mpi}] [--local_rank LOCAL_RANK]
                config

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
                        override some settings in the used config, the key-value pair in xxx=yyy format will be merged into config file. If the
                        value to be overwritten is a list, it should be like key="[a,b]" or key=a,b It also allows nested list/tuple values, e.g.
                        key="[(a,b),(c,d)]" Note that the quotation marks are necessary and that no white space is allowed.
  --launcher {none,pytorch,slurm,mpi}
                        job launcher
  --local_rank LOCAL_RANK, --local-rank LOCAL_RANK
```

> example

```sh
# Fine-tune LLMs by a single GPU:
xtuner train $CONFIG [other_config]

xtuner train train/internlm2_1_8b_qlora_huanhuan_e3.py --deepspeed deepspeed_zero2

# Fine-tune LLMs by multiple GPUs:
NPROC_PER_NODE=$NGPUS NNODES=$NNODES NODE_RANK=$NODE_RANK PORT=$PORT ADDR=$ADDR xtuner dist_train $CONFIG $GPUS
```

> deepspeed support
>
> https://github.com/InternLM/xtuner/tree/main/xtuner/configs/deepspeed

```
deepspeed_zero1 deepspeed_zero2 deepspeed_zero2_offload deepspeed_zero3 deepspeed_zero3_offload
```

# convert

## pth_to_hf

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
                        override some settings in the used config, the key-value pair in xxx=yyy format will be merged into config file. If the
                        value to be overwritten is a list, it should be like key="[a,b]" or key=a,b It also allows nested list/tuple values, e.g.
                        key="[(a,b),(c,d)]" Note that the quotation marks are necessary and that no white space is allowed.
```

> example

```sh
xtuner convert pth_to_hf $CONFIG $PATH_TO_PTH_MODEL $SAVE_PATH_TO_HF_MODEL --max-shard-size 2GB

xtuner convert pth_to_hf \
    train/internlm2_1_8b_qlora_huanhuan_e3.py \
    work_dirs/internlm2_1_8b_qlora_huanhuan_e3/epoch_3.pth \
    work_dirs/internlm2_1_8b_qlora_huanhuan_e3/hf \
    --max-shard-size 2GB
```

## merge adapter

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
xtuner convert merge $LLM $ADAPTER $SAVE_PATH --max-shard-size 2GB

xtuner convert merge \
    models/internlm2-chat-1_8b \
    work_dirs/internlm2_1_8b_qlora_emo_e3/hf \
    work_dirs/internlm2_1_8b_qlora_emo_e3/merged \
    --max-shard-size 2GB
```

## split

```sh
 xtuner convert split --help 
usage: split.py [-h] src_dir dst_dir

Split a HuggingFace model to the smallest sharded one

positional arguments:
  src_dir     the directory of the model
  dst_dir     the directory to save the new model
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
> xtuner chat --help                                                                                                              (mm) 0 (7.751s)
usage: chat.py [-h] [--adapter ADAPTER | --llava LLAVA] [--visual-encoder VISUAL_ENCODER] [--visual-select-layer VISUAL_SELECT_LAYER]
               [--image IMAGE] [--torch-dtype {fp16,bf16,fp32,auto}]
               [--prompt-template {default,zephyr,internlm_chat,internlm2_chat,moss_sft,llama2_chat,code_llama_chat,chatglm2,chatglm3,qwen_chat,baichuan_chat,baichuan2_chat,wizardlm,wizardcoder,vicuna,deepseek_coder,deepseekcoder,deepseek_moe,mistral,mixtral,gemma}]
               [--system SYSTEM | --system-template {moss_sft,alpaca,arxiv_gentile,colorist,coder,lawyer,medical,sql}] [--bits {4,8,None}]
               [--bot-name BOT_NAME] [--with-plugins {calculate,solve,search} [{calculate,solve,search} ...]] [--no-streamer] [--lagent]
               [--stop-words STOP_WORDS [STOP_WORDS ...]] [--offload-folder OFFLOAD_FOLDER] [--max-new-tokens MAX_NEW_TOKENS]
               [--temperature TEMPERATURE] [--top-k TOP_K] [--top-p TOP_P] [--repetition-penalty REPETITION_PENALTY] [--seed SEED]
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
  --top-p TOP_P         If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are
                        kept for generation.
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
    --adapter work_dirs/internlm2_1_8b_qlora_huanhuan_e3/hf \
    --bits 8 --temperature 0.7 --top-k 50 --top-p 0.9 \
    --system 现在你要扮演皇帝身边的女人--甄嬛
```

```sh
# Chat with VLMs with HuggingFace's model and LLaVA:
xtuner chat $LLM --llava $LLAVA --visual-encoder $VISUAL_ENCODER --image $IMAGE --prompt-template $PROMPT_TEMPLATE --system-template $SYSTEM_TEMPLATE
```

# check-custom-dataset

在修改配置文件后，可以运行`xtuner/tools/check_custom_dataset.py`脚本验证数据集是否正确构建。

```sh
xtuner check-custom-dataset $CONFIG
```

其中 `$CONFIG` 是 config 的文件路径。

