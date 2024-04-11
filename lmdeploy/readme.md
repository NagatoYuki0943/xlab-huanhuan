https://github.com/InternLM/Tutorial/tree/camp2/lmdeploy

https://github.com/InternLM/lmdeploy

https://lmdeploy.readthedocs.io/zh-cn/latest/

https://github.com/InternLM/lmdeploy/blob/main/lmdeploy/cli

https://github.com/InternLM/lmdeploy/blob/main/lmdeploy/api.py

# help

```sh
> lmdeploy --help
usage: lmdeploy [-h] [-v] {chat,lite,serve,convert,list,check_env} ...

The CLI provides a unified API for converting, compressing and deploying large language models.

options:
  -h, --help            show this help message and exit
  -v, --version         show program's version number and exit

Commands:
  lmdeploy has following commands:

  {chat,lite,serve,convert,list,check_env}
    chat                Chat with pytorch or turbomind engine.
    lite                Compressing and accelerating LLMs with lmdeploy.lite module
    serve               Serve LLMs with gradio, openai API or triton server.
    convert             Convert LLMs to turbomind format.
    list                List the supported model names.
    check_env           Check the environmental information.
```

## chat 聊天

```sh
> lmdeploy chat --help
usage: lmdeploy chat [-h] [--backend {pytorch,turbomind}] [--trust-remote-code] [--meta-instruction META_INSTRUCTION]
                     [--cap {completion,infilling,chat,python}] [--adapters [ADAPTERS ...]] [--tp TP] [--model-name MODEL_NAME]
                     [--session-len SESSION_LEN] [--max-batch-size MAX_BATCH_SIZE] [--cache-max-entry-count CACHE_MAX_ENTRY_COUNT]
                     [--model-format {hf,llama,awq}] [--quant-policy QUANT_POLICY] [--rope-scaling-factor ROPE_SCALING_FACTOR]
                     model_path

Chat with pytorch or turbomind engine.

positional arguments:
  model_path            The path of a model. it could be one of the following options: - i) a local directory path of a turbomind model which is
                        converted by `lmdeploy convert` command or download from ii) and iii). - ii) the model_id of a lmdeploy-quantized model
                        hosted inside a model repo on huggingface.co, such as "internlm/internlm-chat-20b-4bit", "lmdeploy/llama2-chat-70b-4bit",
                        etc. - iii) the model_id of a model hosted inside a model repo on huggingface.co, such as "internlm/internlm-chat-7b",
                        "qwen/qwen-7b-chat ", "baichuan-inc/baichuan2-7b-chat" and so on. Type: str

options:
  -h, --help            show this help message and exit
  --backend {pytorch,turbomind}
                        Set the inference backend. Default: turbomind. Type: str
  --trust-remote-code   Trust remote code for loading hf models. Default: True
  --meta-instruction META_INSTRUCTION
                        System prompt for ChatTemplateConfig. Deprecated. Please use --chat-template instead. Default: None. Type: str
  --cap {completion,infilling,chat,python}
                        The capability of a model. Deprecated. Please use --chat-template instead. Default: chat. Type: str

PyTorch engine arguments:
  --adapters [ADAPTERS ...]
                        Used to set path(s) of lora adapter(s). One can input key-value pairs in xxx=yyy format for multiple lora adapters. If
                        only have one adapter, one can only input the path of the adapter.. Default: None. Type: str
  --tp TP               GPU number used in tensor parallelism. Should be 2^n. Default: 1. Type: int
  --model-name MODEL_NAME
                        The name of the to-be-deployed model, such as llama-7b, llama-13b, vicuna-7b and etc. You can run `lmdeploy list` to get
                        the supported model names. Default: None. Type: str
  --session-len SESSION_LEN
                        The max session length of a sequence. Default: None. Type: int
  --max-batch-size MAX_BATCH_SIZE
                        Maximum batch size. Default: 128. Type: int
  --cache-max-entry-count CACHE_MAX_ENTRY_COUNT
                        The percentage of gpu memory occupied by the k/v cache. Default: 0.8. Type: float

TurboMind engine arguments:
  --tp TP               GPU number used in tensor parallelism. Should be 2^n. Default: 1. Type: int
  --model-name MODEL_NAME
                        The name of the to-be-deployed model, such as llama-7b, llama-13b, vicuna-7b and etc. You can run `lmdeploy list` to get
                        the supported model names. Default: None. Type: str
  --session-len SESSION_LEN
                        The max session length of a sequence. Default: None. Type: int
  --max-batch-size MAX_BATCH_SIZE
                        Maximum batch size. Default: 128. Type: int
  --cache-max-entry-count CACHE_MAX_ENTRY_COUNT
                        The percentage of gpu memory occupied by the k/v cache. Default: 0.8. Type: float
  --model-format {hf,llama,awq}
                        The format of input model. `hf` meaning `hf_llama`, `llama` meaning `meta_llama`, `awq` meaning the quantized model by
                        awq. Default: None. Type: str
  --quant-policy QUANT_POLICY
                        Whether to use kv int8. Default: 0. Type: int
  --rope-scaling-factor ROPE_SCALING_FACTOR
                        Rope scaling factor. Default: 0.0. Type: float
```

```sh
lmdeploy chat ./models/internlm2-chat-1_8b

# 使用pytorch后端
lmdeploy chat \
    ./models/internlm2-chat-1_8b \
    --backend pytorch \
    --tp 1 \
    --cache-max-entry-count 0.8

# 使用turbomind后端
lmdeploy chat \
    ./models/internlm2-chat-1_8b \
    --backend turbomind \
    --tp 1 \
    --cache-max-entry-count 0.8 \
    --quant-policy 0
```

### [chat torch/turbomind 支持的参数](https://github.com/InternLM/lmdeploy/blob/main/lmdeploy/cli/chat.py)

```python
    @staticmethod
    def add_parser_torch():
        """Add parser for torch command."""
        parser = SubCliChat.subparsers.add_parser(
            'torch',
            formatter_class=DefaultsAndTypesHelpFormatter,
            help=SubCliChat.torch.__doc__,
            description=SubCliChat.torch.__doc__,
        )
        parser.set_defaults(run=SubCliChat.torch)
        parser.add_argument('model_path',
                            type=str,
                            help='The huggingface model path')
        # engine args
        engine_group = parser.add_argument_group('Engine arguments')
        ArgumentHelper.model_name(engine_group)
        ArgumentHelper.tp(engine_group)
        ArgumentHelper.session_len(engine_group)
        ArgumentHelper.adapters(engine_group)
        ArgumentHelper.cache_max_entry_count(engine_group)
        ArgumentHelper.cache_block_seq_len(engine_group)

        # other args
        parser.add_argument('--trust-remote-code',
                            action='store_false',
                            default=True,
                            help='Trust remote code')

    @staticmethod
    def add_parser_turbomind():
        """Add parser for turbomind command."""
        parser = SubCliChat.subparsers.add_parser(
            'turbomind',
            formatter_class=DefaultsAndTypesHelpFormatter,
            help=SubCliChat.turbomind.__doc__,
            description=SubCliChat.turbomind.__doc__,
        )
        parser.set_defaults(run=SubCliChat.turbomind)
        parser.add_argument(
            'model_path',
            type=str,
            help='The path of the deployed model. '
            'It can be in format of huggingface or turbomind. '
            'When it is turbomind model, all arguments for engine'
            'config would be ignored, so you need to change the `config.ini`')
        # engine arguments
        engine_group = parser.add_argument_group('Engine arguments')
        ArgumentHelper.tp(engine_group)
        ArgumentHelper.model_format(engine_group)
        ArgumentHelper.quant_policy(engine_group)
        ArgumentHelper.model_name(engine_group)
        ArgumentHelper.cache_max_entry_count(engine_group)
        ArgumentHelper.cache_block_seq_len(engine_group)
        ArgumentHelper.rope_scaling_factor(engine_group)
        ArgumentHelper.session_len(engine_group)
        # other arguments
        ArgumentHelper.cap(parser)
        ArgumentHelper.meta_instruction(parser)  # TODO remove
        ArgumentHelper.chat_template(parser)
```

## lite 量化

本部分内容主要介绍如何对模型进行量化。主要包括 KV8量化和W4A16量化。总的来说，量化是一种以参数或计算中间结果精度下降换空间节省（以及同时带来的性能提升）的策略。

正式介绍 LMDeploy 量化方案前，需要先介绍两个概念：

* 计算密集（compute-bound）: 指推理过程中，绝大部分时间消耗在数值计算上；针对计算密集型场景，可以通过使用更快的硬件计算单元来提升计算速。
* 访存密集（memory-bound）: 指推理过程中，绝大部分时间消耗在数据读取上；针对访存密集型场景，一般通过减少访存次数、提高计算访存比或降低访存量来优化。

常见的 LLM 模型由于 Decoder Only 架构的特性，实际推理时大多数的时间都消耗在了逐 Token 生成阶段（Decoding 阶段），是典型的访存密集型场景。

那么，如何优化 LLM 模型推理中的访存密集问题呢？ 我们可以使用**KV8量化**和**W4A16**量化。KV8量化是指将逐 Token（Decoding）生成过程中的上下文 K 和 V 中间结果进行 INT8 量化（计算时再反量化），以降低生成过程中的显存占用。W4A16 量化，将 FP16 的模型权重量化为 INT4，Kernel 计算时，访存量直接降为 FP16 模型的 1/4，大幅降低了访存成本。Weight Only 是指仅量化权重，数值计算依然采用 FP16（需要将 INT4 权重反量化）。

```sh
> lmdeploy lite --help
usage: lmdeploy lite [-h] {auto_awq,calibrate,kv_qparams,smooth_quant} ...

Compressing and accelerating LLMs with lmdeploy.lite module

options:
  -h, --help            show this help message and exit

Commands:
  This group has the following commands:

  {auto_awq,calibrate,kv_qparams,smooth_quant}
    auto_awq            Perform weight quantization using AWQ algorithm.
    calibrate           Perform calibration on a given dataset.
    kv_qparams          Export key and value stats.
    smooth_quant        Perform w8a8 quantization using SmoothQuant.
```

https://github.com/InternLM/lmdeploy/tree/main/docs/zh_cn/quantization

### KV Cache

KV Cache是一种缓存技术，通过存储键值对的形式来复用计算结果，以达到提高性能和降低内存消耗的目的。在大规模训练和推理中，KV Cache可以显著减少重复计算量，从而提升模型的推理速度。理想情况下，KV Cache全部存储于显存，以加快访存速度。当显存空间不足时，也可以将KV Cache放在内存，通过缓存管理器控制将当前需要使用的数据放入显存。

模型在运行时，占用的显存可大致分为三部分：模型参数本身占用的显存、KV Cache占用的显存，以及中间运算结果占用的显存。LMDeploy的KV Cache管理器可以通过设置`--cache-max-entry-count`参数，控制KV缓存**占用剩余显存**的最大比例。默认的比例为0.8。

下面通过几个例子，来看一下调整`--cache-max-entry-count`参数的效果。首先保持不加该参数（默认0.8），运行1.8B模型。

```sh
# 改变--cache-max-entry-count参数，设为0.5
lmdeploy chat \
    ./models/internlm2-chat-1_8b-4bit \ 
    --backend turbomind \
    --tp 1 \
    --cache-max-entry-count 0.8 \
    --quant-policy 0
```

#### [KV Cache int8](https://github.com/InternLM/lmdeploy/blob/main/docs/zh_cn/quantization/kv_int8.md)

```sh
> lmdeploy lite calibrate --help
usage: lmdeploy lite calibrate [-h] [--work-dir WORK_DIR] [--calib-dataset CALIB_DATASET] [--calib-samples CALIB_SAMPLES]
                               [--calib-seqlen CALIB_SEQLEN] [--device {cuda,cpu}]
                               model

Perform calibration on a given dataset.

positional arguments:
  model                 The name or path of the model to be loaded. Type: str

options:
  -h, --help            show this help message and exit
  --work-dir WORK_DIR   The working directory to save results. Default: ./work_dir. Type: str
  --calib-dataset CALIB_DATASET
                        The calibration dataset name. Default: ptb. Type: str
  --calib-samples CALIB_SAMPLES
                        The number of samples for calibration. Default: 128. Type: int
  --calib-seqlen CALIB_SEQLEN
                        The sequence length for calibration. Default: 2048. Type: int
  --device {cuda,cpu}   Device type of running. Default: cuda. Type: str
```

```sh
lmdeploy chat \
    ./models/internlm2-chat-1_8b-4bit \ 
    --backend turbomind \
    --tp 1 \
    --cache-max-entry-count 0.8 \
    --quant-policy 4 # 启用 kv int8 量化, W4A16量化的模型才能使用 kv int8 量化
```

### [W4A16](https://github.com/InternLM/lmdeploy/blob/main/docs/zh_cn/quantization/w4a16.md)

使用 AWQ 算法，实现模型 4bit 权重量化。

```sh
> lmdeploy lite auto_awq --help
usage: lmdeploy lite auto_awq [-h] [--work-dir WORK_DIR] [--calib-dataset CALIB_DATASET] [--calib-samples CALIB_SAMPLES]
                              [--calib-seqlen CALIB_SEQLEN] [--device {cuda,cpu}] [--w-bits W_BITS] [--w-sym] [--w-group-size W_GROUP_SIZE]
                              model

Perform weight quantization using AWQ algorithm.

positional arguments:
  model                 The path of model in hf format. Type: str

options:
  -h, --help            show this help message and exit
  --work-dir WORK_DIR   The working directory to save results. Default: ./work_dir. Type: str
  --calib-dataset CALIB_DATASET
                        The calibration dataset name. Default: ptb. Type: str
  --calib-samples CALIB_SAMPLES
                        The number of samples for calibration. Default: 128. Type: int
  --calib-seqlen CALIB_SEQLEN
                        The sequence length for calibration. Default: 2048. Type: int
  --device {cuda,cpu}   Device type of running. Default: cuda. Type: str
  --w-bits W_BITS       Bit number for weight quantization. Default: 4. Type: int
  --w-sym               Whether to do symmetric quantization. Default: False
  --w-group-size W_GROUP_SIZE
                        Group size for weight quantization statistics. Default: 128. Type: int
```

#### 模型量化

仅需执行一条命令，就可以完成模型量化工作。量化结束后，权重文件存放在 `$WORK_DIR` 下。

```sh
export HF_MODEL=internlm/internlm-chat-7b
export WORK_DIR=internlm/internlm-chat-7b-4bit

lmdeploy lite auto_awq \
    $HF_MODEL \
    --calib-dataset 'ptb' \
    --calib-samples 128 \
    --calib-seqlen 2048 \
    --w-bits 4 \
    --w-group-size 128 \
    --work-dir $WORK_DIR

lmdeploy lite auto_awq \
    ./models/internlm2-chat-1_8b \
    --calib-dataset 'ptb' \
    --calib-samples 128 \
    --calib-seqlen 2048 \
    --w-bits 4 \
    --w-group-size 128 \
    --work-dir ./models/internlm2-chat-1_8b-4bit
```

量化后的模型，可以用一些工具快速验证对话效果。

比如，直接在控制台和模型对话，

```sh
lmdeploy chat \
    ./models/internlm2-chat-1_8b-4bit \
    --backend turbomind \
    --model-format awq \
    --tp 1 \
    --cache-max-entry-count 0.8 \
    --quant-policy 0

# 不支持 torch
```

#### 模型推理

量化后的模型，通过以下几行简单的代码，可以实现离线推理：

```sh
from lmdeploy import pipeline, TurbomindEngineConfig

engine_config = TurbomindEngineConfig(model_format='awq', cache_max_entry_count=0.5)
pipe = pipeline("./models/internlm2-chat-1_8b-4bit", backend_config=engine_config)
response = pipe(["Hi, pls intro yourself", "Shanghai is"])

print(response)
```

关于 pipeline 的详细介绍，请参考[这里](https://github.com/InternLM/lmdeploy/blob/main/docs/zh_cn/inference/pipeline.md)

除了推理本地量化模型外，LMDeploy 还支持直接推理 huggingface hub 上的通过 AWQ 量化的 4bit 权重模型，比如 [lmdeploy 空间](https://huggingface.co/lmdeploy) 和 [TheBloke 空间](https://huggingface.co/TheBloke) 下的模型。

```sh
# 推理 lmdeploy 空间下的模型
from lmdeploy import pipeline, TurbomindEngineConfig
pipe = pipeline("lmdeploy/llama2-chat-70b-4bit",
                backend_config=TurbomindEngineConfig(model_format='awq', tp=4, cache_max_entry_count=0.5))
response = pipe(["Hi, pls intro yourself", "Shanghai is"])
print(response)

# 推理 TheBloke 空间下的模型（试试codellama行不行）
from lmdeploy import pipeline, TurbomindEngineConfig, ChatTemplateConfig
pipe = pipeline("TheBloke/LLaMA2-13B-Tiefighter-AWQ",
                backend_config=TurbomindEngineConfig(model_format='awq', cache_max_entry_count=0.5),
                chat_template_config=ChatTemplateConfig(model_name='llama2')
                )
response = pipe(["Hi, pls intro yourself", "Shanghai is"])
print(response)
```

### [W8A8](https://github.com/InternLM/lmdeploy/blob/main/docs/zh_cn/quantization/w8a8.md)

```sh
> lmdeploy lite smooth_quant --help
usage: lmdeploy lite smooth_quant [-h] [--work-dir WORK_DIR] [--calib-dataset CALIB_DATASET] [--calib-samples CALIB_SAMPLES]
                                  [--calib-seqlen CALIB_SEQLEN] [--device {cuda,cpu}]
                                  model

Perform w8a8 quantization using SmoothQuant.

positional arguments:
  model                 The name or path of the model to be loaded. Type: str

options:
  -h, --help            show this help message and exit
  --work-dir WORK_DIR   The working directory for outputs. defaults to "./work_dir". Type: str
  --calib-dataset CALIB_DATASET
                        The calibration dataset name. Default: ptb. Type: str
  --calib-samples CALIB_SAMPLES
                        The number of samples for calibration. Default: 128. Type: int
  --calib-seqlen CALIB_SEQLEN
                        The sequence length for calibration. Default: 2048. Type: int
  --device {cuda,cpu}   Device type of running. Default: cuda. Type: str
```

使用 8 bit 整数对神经网络模型进行量化和推理的功能。

在开始推理前，需要确保已经正确安装了 lmdeploy 和 openai/triton。

将原 16bit 权重量化为 8bit，并保存至 `internlm-chat-7b-w8` 目录下，操作命令如下：

```sh
lmdeploy lite smooth_quant ./models/internlm2-chat-1_8b --work-dir ./models/internlm2-chat-1_8b-w8
```

然后，执行以下命令，即可在终端与模型对话：

```sh
lmdeploy chat \
    ./models/internlm2-chat-1_8b \
    --backend pytorch \
    --tp 1 \
    --cache-max-entry-count 0.8 \

# 不支持 turbomind
```

## serve

```sh
> lmdeploy serve --help
usage: lmdeploy serve [-h] {gradio,api_server,api_client,triton_client} ...

Serve LLMs with gradio, openai API or triton server.

options:
  -h, --help            show this help message and exit

Commands:
  This group has the following commands:

  {gradio,api_server,api_client,triton_client}
    gradio              Serve LLMs with web UI using gradio.
    api_server          Serve LLMs with restful api using fastapi.
    api_client          Interact with restful api server in terminal.
    triton_client       Interact with Triton Server using gRPC protocol.
```

### api_server

```sh
> lmdeploy serve api_server -h
usage: lmdeploy serve api_server [-h] [--server-name SERVER_NAME] [--server-port SERVER_PORT] [--allow-origins ALLOW_ORIGINS [ALLOW_ORIGINS ...]]
                                 [--allow-credentials] [--allow-methods ALLOW_METHODS [ALLOW_METHODS ...]]
                                 [--allow-headers ALLOW_HEADERS [ALLOW_HEADERS ...]] [--qos-config-path QOS_CONFIG_PATH]
                                 [--backend {pytorch,turbomind}] [--log-level {CRITICAL,FATAL,ERROR,WARN,WARNING,INFO,DEBUG,NOTSET}]
                                 [--api-keys [API_KEYS ...]] [--ssl] [--meta-instruction META_INSTRUCTION] [--chat-template CHAT_TEMPLATE]
                                 [--cap {completion,infilling,chat,python}] [--adapters [ADAPTERS ...]] [--tp TP] [--model-name MODEL_NAME]
                                 [--session-len SESSION_LEN] [--max-batch-size MAX_BATCH_SIZE] [--cache-max-entry-count CACHE_MAX_ENTRY_COUNT]
                                 [--cache-block-seq-len CACHE_BLOCK_SEQ_LEN] [--model-format {hf,llama,awq}] [--quant-policy QUANT_POLICY]
                                 [--rope-scaling-factor ROPE_SCALING_FACTOR]
                                 model_path

Serve LLMs with restful api using fastapi.

positional arguments:
  model_path            The path of a model. it could be one of the following options: - i) a local directory path of a turbomind model which is
                        converted by `lmdeploy convert` command or download from ii) and iii). - ii) the model_id of a lmdeploy-quantized model
                        hosted inside a model repo on huggingface.co, such as "internlm/internlm-chat-20b-4bit", "lmdeploy/llama2-chat-70b-4bit",
                        etc. - iii) the model_id of a model hosted inside a model repo on huggingface.co, such as "internlm/internlm-chat-7b",
                        "qwen/qwen-7b-chat ", "baichuan-inc/baichuan2-7b-chat" and so on. Type: str

options:
  -h, --help            show this help message and exit
  --server-name SERVER_NAME
                        Host ip for serving. Default: 0.0.0.0. Type: str
  --server-port SERVER_PORT
                        Server port. Default: 23333. Type: int
  --allow-origins ALLOW_ORIGINS [ALLOW_ORIGINS ...]
                        A list of allowed origins for cors. Default: ['*']. Type: str
  --allow-credentials   Whether to allow credentials for cors. Default: False
  --allow-methods ALLOW_METHODS [ALLOW_METHODS ...]
                        A list of allowed http methods for cors. Default: ['*']. Type: str
  --allow-headers ALLOW_HEADERS [ALLOW_HEADERS ...]
                        A list of allowed http headers for cors. Default: ['*']. Type: str
  --qos-config-path QOS_CONFIG_PATH
                        Qos policy config path. Default: . Type: str
  --backend {pytorch,turbomind}
                        Set the inference backend. Default: turbomind. Type: str
  --log-level {CRITICAL,FATAL,ERROR,WARN,WARNING,INFO,DEBUG,NOTSET}
                        Set the log level. Default: ERROR. Type: str
  --api-keys [API_KEYS ...]
                        Optional list of space separated API keys. Default: None. Type: str
  --ssl                 Enable SSL. Requires OS Environment variables 'SSL_KEYFILE' and 'SSL_CERTFILE'. Default: False
  --meta-instruction META_INSTRUCTION
                        System prompt for ChatTemplateConfig. Deprecated. Please use --chat-template instead. Default: None. Type: str
  --chat-template CHAT_TEMPLATE
                        A JSON file or string that specifies the chat template configuration. Please refer to
                        https://lmdeploy.readthedocs.io/en/latest/advance/chat_template.html for the specification. Default: None. Type: str
  --cap {completion,infilling,chat,python}
                        The capability of a model. Deprecated. Please use --chat-template instead. Default: chat. Type: str

PyTorch engine arguments:
  --adapters [ADAPTERS ...]
                        Used to set path(s) of lora adapter(s). One can input key-value pairs in xxx=yyy format for multiple lora adapters. If
                        only have one adapter, one can only input the path of the adapter.. Default: None. Type: str
  --tp TP               GPU number used in tensor parallelism. Should be 2^n. Default: 1. Type: int
  --model-name MODEL_NAME
                        The name of the to-be-deployed model, such as llama-7b, llama-13b, vicuna-7b and etc. You can run `lmdeploy list` to get
                        the supported model names. Default: None. Type: str
  --session-len SESSION_LEN
                        The max session length of a sequence. Default: None. Type: int
  --max-batch-size MAX_BATCH_SIZE
                        Maximum batch size. Default: 128. Type: int
  --cache-max-entry-count CACHE_MAX_ENTRY_COUNT
                        The percentage of gpu memory occupied by the k/v cache. Default: 0.8. Type: float
  --cache-block-seq-len CACHE_BLOCK_SEQ_LEN
                        The length of the token sequence in a k/v block. For Turbomind Engine, if the GPU compute capability is >= 8.0, it should
                        be a multiple of 32, otherwise it should be a multiple of 64. For Pytorch Engine, if Lora Adapter is specified, this
                        parameter will be ignored. Default: 64. Type: int

TurboMind engine arguments:
  --tp TP               GPU number used in tensor parallelism. Should be 2^n. Default: 1. Type: int
  --model-name MODEL_NAME
                        The name of the to-be-deployed model, such as llama-7b, llama-13b, vicuna-7b and etc. You can run `lmdeploy list` to get
                        the supported model names. Default: None. Type: str
  --session-len SESSION_LEN
                        The max session length of a sequence. Default: None. Type: int
  --max-batch-size MAX_BATCH_SIZE
                        Maximum batch size. Default: 128. Type: int
  --cache-max-entry-count CACHE_MAX_ENTRY_COUNT
                        The percentage of gpu memory occupied by the k/v cache. Default: 0.8. Type: float
  --cache-block-seq-len CACHE_BLOCK_SEQ_LEN
                        The length of the token sequence in a k/v block. For Turbomind Engine, if the GPU compute capability is >= 8.0, it should
                        be a multiple of 32, otherwise it should be a multiple of 64. For Pytorch Engine, if Lora Adapter is specified, this
                        parameter will be ignored. Default: 64. Type: int
  --model-format {hf,llama,awq}
                        The format of input model. `hf` meaning `hf_llama`, `llama` meaning `meta_llama`, `awq` meaning the quantized model by
                        awq. Default: None. Type: str
  --quant-policy QUANT_POLICY
                        Whether to use kv int8. Default: 0. Type: int
  --rope-scaling-factor ROPE_SCALING_FACTOR
                        Rope scaling factor. Default: 0.0. Type: float
```

```sh
lmdeploy serve api_server \
    ./models/internlm2-chat-1_8b \
    --backend pytorch \
    --tp 1 \
    --cache-max-entry-count 0.8 \
    --model-name internlm2_1_8b_chat \
    --server-name {ip_addr} \
    --server-port {port}

lmdeploy serve api_server \
    ./models/internlm2-chat-1_8b \
    --backend turbomind \
    --tp 1 \
    --cache-max-entry-count 0.8 \
    --quant-policy 0 \
    --model-name internlm2_1_8b_chat \
    --server-name {ip_addr} \
    --server-port {port}

lmdeploy serve api_server \
    ./models/internlm2-chat-1_8b \
    --backend turbomind \
    --tp 1 \
    --cache-max-entry-count 0.8 \
    --quant-policy 0 \
    --model-name internlm2_1_8b_chat \
    --server-name 0.0.0.0 \
    --server-port 23333
```

server访问

```sh
> lmdeploy serve api_client --help                                                                                                (mm) 0 (2.716s)
usage: lmdeploy serve api_client [-h] [--api-key API_KEY] [--session-id SESSION_ID] api_server_url

Interact with restful api server in terminal.

positional arguments:
  api_server_url        The URL of api server. Type: str

options:
  -h, --help            show this help message and exit
  --api-key API_KEY     api key. Default to None, which means no api key will be used. Type: str
  --session-id SESSION_ID
                        The identical id of a session. Default: 1. Type: int
```

```sh
lmdeploy serve api_client http://localhost:23333
```

使用的架构是这样的：

![](imgs/4.2_4.jpg)

### gradio

```sh
> lmdeploy serve gradio -h
usage: lmdeploy serve gradio [-h] [--server-name SERVER_NAME] [--server-port SERVER_PORT] [--backend {pytorch,turbomind}]
                             [--meta-instruction META_INSTRUCTION] [--chat-template CHAT_TEMPLATE] [--cap {completion,infilling,chat,python}]
                             [--tp TP] [--model-name MODEL_NAME] [--session-len SESSION_LEN] [--max-batch-size MAX_BATCH_SIZE]
                             [--cache-max-entry-count CACHE_MAX_ENTRY_COUNT] [--cache-block-seq-len CACHE_BLOCK_SEQ_LEN]
                             [--model-format {hf,llama,awq}] [--quant-policy QUANT_POLICY] [--rope-scaling-factor ROPE_SCALING_FACTOR]
                             model_path_or_server

Serve LLMs with web UI using gradio.

positional arguments:
  model_path_or_server  The path of the deployed model or the tritonserver url or restful api url. for example: - ./workspace - 0.0.0.0:23333 -
                        http://0.0.0.0:23333. Type: str

options:
  -h, --help            show this help message and exit
  --server-name SERVER_NAME
                        The ip address of gradio server. Default: 0.0.0.0. Type: str
  --server-port SERVER_PORT
                        The port of gradio server. Default: 6006. Type: int
  --backend {pytorch,turbomind}
                        Set the inference backend. Default: turbomind. Type: str
  --meta-instruction META_INSTRUCTION
                        System prompt for ChatTemplateConfig. Deprecated. Please use --chat-template instead. Default: None. Type: str
  --chat-template CHAT_TEMPLATE
                        A JSON file or string that specifies the chat template configuration. Please refer to
                        https://lmdeploy.readthedocs.io/en/latest/advance/chat_template.html for the specification. Default: None. Type: str
  --cap {completion,infilling,chat,python}
                        The capability of a model. Deprecated. Please use --chat-template instead. Default: chat. Type: str

PyTorch engine arguments:
  --tp TP               GPU number used in tensor parallelism. Should be 2^n. Default: 1. Type: int
  --model-name MODEL_NAME
                        The name of the to-be-deployed model, such as llama-7b, llama-13b, vicuna-7b and etc. You can run `lmdeploy list` to get
                        the supported model names. Default: None. Type: str
  --session-len SESSION_LEN
                        The max session length of a sequence. Default: None. Type: int
  --max-batch-size MAX_BATCH_SIZE
                        Maximum batch size. Default: 128. Type: int
  --cache-max-entry-count CACHE_MAX_ENTRY_COUNT
                        The percentage of gpu memory occupied by the k/v cache. Default: 0.8. Type: float
  --cache-block-seq-len CACHE_BLOCK_SEQ_LEN
                        The length of the token sequence in a k/v block. For Turbomind Engine, if the GPU compute capability is >= 8.0, it should
                        be a multiple of 32, otherwise it should be a multiple of 64. For Pytorch Engine, if Lora Adapter is specified, this
                        parameter will be ignored. Default: 64. Type: int

TurboMind engine arguments:
  --tp TP               GPU number used in tensor parallelism. Should be 2^n. Default: 1. Type: int
  --model-name MODEL_NAME
                        The name of the to-be-deployed model, such as llama-7b, llama-13b, vicuna-7b and etc. You can run `lmdeploy list` to get
                        the supported model names. Default: None. Type: str
  --session-len SESSION_LEN
                        The max session length of a sequence. Default: None. Type: int
  --max-batch-size MAX_BATCH_SIZE
                        Maximum batch size. Default: 128. Type: int
  --cache-max-entry-count CACHE_MAX_ENTRY_COUNT
                        The percentage of gpu memory occupied by the k/v cache. Default: 0.8. Type: float
  --cache-block-seq-len CACHE_BLOCK_SEQ_LEN
                        The length of the token sequence in a k/v block. For Turbomind Engine, if the GPU compute capability is >= 8.0, it should
                        be a multiple of 32, otherwise it should be a multiple of 64. For Pytorch Engine, if Lora Adapter is specified, this
                        parameter will be ignored. Default: 64. Type: int
  --model-format {hf,llama,awq}
                        The format of input model. `hf` meaning `hf_llama`, `llama` meaning `meta_llama`, `awq` meaning the quantized model by
                        awq. Default: None. Type: str
  --quant-policy QUANT_POLICY
                        Whether to use kv int8. Default: 0. Type: int
  --rope-scaling-factor ROPE_SCALING_FACTOR
                        Rope scaling factor. Default: 0.0. Type: float
```

> 启动gradio

```sh
lmdeploy serve gradio \
    ./models/internlm2-chat-1_8b \
    --backend pytorch \
    --tp 1 \
    --cache-max-entry-count 0.8 \
    --server-name {ip_addr} \
    --server-port {port}

lmdeploy serve gradio \
    ./models/internlm2-chat-1_8b \
    --backend turbomind \
    --tp 1 \
    --cache-max-entry-count 0.8 \
    --quant-policy 0 \
    --server-name {ip_addr} \
    --server-port {port}

lmdeploy serve gradio \
    ./models/internlm2-chat-1_8b \
    --backend turbomind \
    --tp 1 \
    --cache-max-entry-count 0.8 \
    --quant-policy 0 \
    --server-name 0.0.0.0 \
    --server-port 6006
```

> 先启动server作为后端，再启动gradio作为前端

```sh
lmdeploy serve api_server \
    ./models/internlm2-chat-1_8b \
    --backend turbomind \
    --tp 1 \
    --cache-max-entry-count 0.8 \
    --quant-policy 0 \
    --model-name internlm2_1_8b_chat \
    --server-name 0.0.0.0 \
    --server-port 23333

lmdeploy serve gradio http://localhost:23333 \
    --server-name 0.0.0.0 \
    --server-port 6006
```

现在使用的架构是这样的：

![4.3_3](imgs/4.3_3.jpg)

## convert

```sh
> lmdeploy convert --help
usage: lmdeploy convert [-h] [--model-format {hf,llama,awq}] [--tp TP] [--tokenizer-path TOKENIZER_PATH] [--dst-path DST_PATH]
                        [--quant-path QUANT_PATH] [--group-size GROUP_SIZE]
                        model_name model_path

Convert LLMs to turbomind format.

positional arguments:
  model_name            The name of the to-be-deployed model, such as llama-7b, llama-13b, vicuna-7b and etc. You can run `lmdeploy list` to get
                        the supported model names. Type: str
  model_path            The directory path of the model. Type: str

options:
  -h, --help            show this help message and exit
  --model-format {hf,llama,awq}
                        The format of input model. `hf` meaning `hf_llama`, `llama` meaning `meta_llama`, `awq` meaning the quantized model by
                        awq. Default: None. Type: str
  --tp TP               GPU number used in tensor parallelism. Should be 2^n. Default: 1. Type: int
  --tokenizer-path TOKENIZER_PATH
                        The path of tokenizer model. Default: None. Type: str
  --dst-path DST_PATH   The destination path that saves outputs. Default: workspace. Type: str
  --quant-path QUANT_PATH
                        Path of the quantized model, which can be none. Default: None. Type: str
  --group-size GROUP_SIZE
                        A parameter used in awq to quantize fp16 weights to 4 bits. Default: 0. Type: int
```

```sh
lmdeploy convert internlm2 \
    ./models/internlm2-chat-1_8b \
    --dst-path ./models/internlm2-chat-1_8b-turbomind

# 转化量化后的W4A16默认,需要设置 group-size
lmdeploy convert internlm2 \
    ./models/internlm2-chat-1_8b-4bit \
    --model-format awq \
    --group-size 128 \ # 需要这个
    --dst-path ./models/internlm2-chat-1_8b-4bit-turbomind

lmdeploy chat \
    ./models/internlm2-chat-1_8b-turbomind \
    --backend turbomind \
    --tp 1 \
    --cache-max-entry-count 0.8 \
    --quant-policy 0 # 0/4 turbomind格式可以直接使用 kv int8 量化
```

## list

```sh
> lmdeploy list --help
usage: lmdeploy list [-h]

List the supported model names.

options:
  -h, --help  show this help message and exit
```

```sh
> lmdeploy list
The older chat template name like "internlm2-7b", "qwen-7b" and so on are deprecated and will be removed in the future. The supported chat template names are:
baichuan2
chatglm
codellama
dbrx
deepseek
deepseek-coder
deepseek-vl
falcon
gemma
internlm
internlm2
llama
llama2
mistral
mixtral
puyu
qwen
solar
ultracm
ultralm
vicuna
wizardlm
yi
yi-vl
```

## check_env

```sh
> lmdeploy check_env --help
usage: lmdeploy check_env [-h] [--dump-file DUMP_FILE]

Check the environmental information.

options:
  -h, --help            show this help message and exit
  --dump-file DUMP_FILE
                        The file path to save env info. Only support file format in `json`, `yml`, `pkl`. Default: None. Type: str
```

```sh
> lmdeploy check_env                                                                                                     )
sys.platform: linux
Python: 3.10.13 (main, Sep 11 2023, 13:44:35) [GCC 11.2.0]
CUDA available: True
MUSA available: False
numpy_random_seed: 2147483648
GPU 0: Tesla V100-SXM2-16GB
CUDA_HOME: /usr/local/cuda
NVCC: Cuda compilation tools, release 12.1, V12.1.105
GCC: gcc (Ubuntu 9.4.0-1ubuntu1~20.04.1) 9.4.0
PyTorch: 2.1.2+cu121
PyTorch compiling details: PyTorch built with:
  - GCC 9.3
  - C++ Version: 201703
  - Intel(R) oneAPI Math Kernel Library Version 2022.2-Product Build 20220804 for Intel(R) 64 architecture applications
  - Intel(R) MKL-DNN v3.1.1 (Git Hash 64f6bcbcbab628e96f33a62c3e975f8535a7bde4)
  - OpenMP 201511 (a.k.a. OpenMP 4.5)
  - LAPACK is enabled (usually provided by MKL)
  - NNPACK is enabled
  - CPU capability usage: AVX512
  - CUDA Runtime 12.1
  - NVCC architecture flags: -gencode;arch=compute_50,code=sm_50;-gencode;arch=compute_60,code=sm_60;-gencode;arch=compute_70,code=sm_70;-gencode;arch=compute_75,code=sm_75;-gencode;arch=compute_80,code=sm_80;-gencode;arch=compute_86,code=sm_86;-gencode;arch=compute_90,code=sm_90
  - CuDNN 8.9.2
  - Magma 2.6.1
  - Build settings: BLAS_INFO=mkl, BUILD_TYPE=Release, CUDA_VERSION=12.1, CUDNN_VERSION=8.9.2, CXX_COMPILER=/opt/rh/devtoolset-9/root/usr/bin/c++, CXX_FLAGS= -D_GLIBCXX_USE_CXX11_ABI=0 -fabi-version=11 -fvisibility-inlines-hidden -DUSE_PTHREADPOOL -DNDEBUG -DUSE_KINETO -DLIBKINETO_NOROCTRACER -DUSE_FBGEMM -DUSE_QNNPACK -DUSE_PYTORCH_QNNPACK -DUSE_XNNPACK -DSYMBOLICATE_MOBILE_DEBUG_HANDLE -O2 -fPIC -Wall -Wextra -Werror=return-type -Werror=non-virtual-dtor -Werror=bool-operation -Wnarrowing -Wno-missing-field-initializers -Wno-type-limits -Wno-array-bounds -Wno-unknown-pragmas -Wno-unused-parameter -Wno-unused-function -Wno-unused-result -Wno-strict-overflow -Wno-strict-aliasing -Wno-stringop-overflow -Wno-psabi -Wno-error=pedantic -Wno-error=old-style-cast -Wno-invalid-partial-specialization -Wno-unused-private-field -Wno-aligned-allocation-unavailable -Wno-missing-braces -fdiagnostics-color=always -faligned-new -Wno-unused-but-set-variable -Wno-maybe-uninitialized -fno-math-errno -fno-trapping-math -Werror=format -Werror=cast-function-type -Wno-stringop-overflow, LAPACK_INFO=mkl, PERF_WITH_AVX=1, PERF_WITH_AVX2=1, PERF_WITH_AVX512=1, TORCH_DISABLE_GPU_ASSERTS=ON, TORCH_VERSION=2.1.2, USE_CUDA=ON, USE_CUDNN=ON, USE_EXCEPTION_PTR=1, USE_GFLAGS=OFF, USE_GLOG=OFF, USE_MKL=ON, USE_MKLDNN=ON, USE_MPI=OFF, USE_NCCL=1, USE_NNPACK=ON, USE_OPENMP=ON, USE_ROCM=OFF,

TorchVision: 0.16.2+cu121
LMDeploy: 0.3.0+
transformers: 4.37.2
gradio: 4.25.0
fastapi: 0.110.1
pydantic: 2.6.4
triton: 2.1.0
```

