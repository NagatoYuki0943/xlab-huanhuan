https://lmdeploy.readthedocs.io/zh-cn/latest/

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
usage: lmdeploy chat [-h] {torch,turbomind} ...

Chat with pytorch or turbomind engine.

options:
  -h, --help         show this help message and exit

Commands:
  This group has the following commands:

  {torch,turbomind}
    torch            Chat with PyTorch inference engine through terminal.
    turbomind        Chat with TurboMind inference engine through terminal.
```

```sh
lmdeploy chat torch ./models/internlm2-chat-1_8b-sft

lmdeploy chat turbomind ./models/internlm2-chat-1_8b-sft
```

## lite 量化

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

### [w8a8](https://github.com/InternLM/lmdeploy/blob/main/docs/zh_cn/quantization/w8a8.md)

使用 8 bit 整数对神经网络模型进行量化和推理的功能。

在开始推理前，需要确保已经正确安装了 lmdeploy 和 openai/triton。

将原 16bit 权重量化为 8bit，并保存至 `internlm-chat-7b-w8` 目录下，操作命令如下：

```sh
lmdeploy lite smooth_quant internlm/internlm-chat-7b --work-dir ./internlm-chat-7b-w8

lmdeploy lite smooth_quant ./models/internlm2-chat-1_8b-sft --work-dir ./models/internlm2-chat-1_8b-sft-w8
```

然后，执行以下命令，即可在终端与模型对话：

```sh
lmdeploy chat torch ./internlm-chat-7b-w8

lmdeploy chat torch ./models/internlm2-chat-1_8b-sft-w8

# 不支持 turbomind
```

### [w4a16](https://github.com/InternLM/lmdeploy/blob/main/docs/zh_cn/quantization/w4a16.md)

使用 AWQ 算法，实现模型 4bit 权重量化。

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
```

绝大多数情况下，在执行上述命令时，可选参数可不用填写，使用默认的即可。比如量化 [internlm/internlm-chat-7b](https://huggingface.co/internlm/internlm-chat-7b) 模型，命令可以简化为：

```sh
lmdeploy lite auto_awq internlm/ianternlm-chat-7b --work-dir internlm-chat-7b-4bit

lmdeploy lite auto_awq ./models/internlm2-chat-1_8b-sft --work-dir ./models/internlm2-chat-1_8b-sft-4bit
```

量化后的模型，可以用一些工具快速验证对话效果。

比如，直接在控制台和模型对话，

```sh
lmdeploy chat turbomind ./internlm-chat-7b-4bit --model-format awq

lmdeploy chat turbomind ./models/internlm2-chat-1_8b-sft-4bit --model-format awq

# 不支持 pytorch
```

或者，启动gradio服务，

```sh
lmdeploy serve gradio ./internlm-chat-7b-4bit --server-name {ip_addr} --server-port {port} --model-format awq
```

然后，在浏览器中打开 http://{ip_addr}:{port}，即可在线对话

#### 模型推理

量化后的模型，通过以下几行简单的代码，可以实现离线推理：

```sh
from lmdeploy import pipeline, TurbomindEngineConfig
engine_config = TurbomindEngineConfig(model_format='awq')
pipe = pipeline("./internlm-chat-7b-4bit", backend_config=engine_config)
response = pipe(["Hi, pls intro yourself", "Shanghai is"])
print(response)
```

关于 pipeline 的详细介绍，请参考[这里](https://github.com/InternLM/lmdeploy/blob/main/docs/zh_cn/inference/pipeline.md)

除了推理本地量化模型外，LMDeploy 还支持直接推理 huggingface hub 上的通过 AWQ 量化的 4bit 权重模型，比如 [lmdeploy 空间](https://huggingface.co/lmdeploy) 和 [TheBloke 空间](https://huggingface.co/TheBloke) 下的模型。

```sh
# 推理 lmdeploy 空间下的模型
from lmdeploy import pipeline, TurbomindEngineConfig
pipe = pipeline("lmdeploy/llama2-chat-70b-4bit",
                backend_config=TurbomindEngineConfig(model_format='awq', tp=4))
response = pipe(["Hi, pls intro yourself", "Shanghai is"])
print(response)

# 推理 TheBloke 空间下的模型（试试codellama行不行）
from lmdeploy import pipeline, TurbomindEngineConfig, ChatTemplateConfig
pipe = pipeline("TheBloke/LLaMA2-13B-Tiefighter-AWQ",
                backend_config=TurbomindEngineConfig(model_format='awq'),
                chat_template_config=ChatTemplateConfig(model_name='llama2')
                )
response = pipe(["Hi, pls intro yourself", "Shanghai is"])
print(response)
```

#### 推理服务

LMDeploy `api_server` 支持把模型一键封装为服务，对外提供的 RESTful API 兼容 openai 的接口。以下为服务启动的示例：

```
lmdeploy serve api_server internlm/internlm-chat-7b --backend turbomind --model-format awq
```

服务默认端口是23333。在 server 启动后，你可以在终端通过`api_client`与server进行对话：

```
lmdeploy serve api_client http://0.0.0.0:23333
```

还可以通过 Swagger UI `http://0.0.0.0:23333` 在线阅读和试用 `api_server` 的各接口，也可直接查阅[文档](https://github.com/InternLM/lmdeploy/blob/main/docs/zh_cn/serving/api_server.md)，了解各接口的定义和使用方法。

### kv_int8

**第一步**

通过以下命令，获取量化参数，并保存至原HF模型目录

```sh
# get minmax
export HF_MODEL=internlm/internlm-chat-7b

lmdeploy lite calibrate \
  $HF_MODEL \
  --calib-dataset 'ptb' \
  --calib-samples 128 \
  --calib-seqlen 2048 \
  --work-dir $HF_MODEL

lmdeploy lite calibrate \
  ./models/internlm2-chat-1_8b-sft \
  --calib-dataset 'ptb' \
  --calib-samples 128 \
  --calib-seqlen 2048 \
  --work-dir ./models/internlm2-chat-1_8b-sft
```

**第二步**



测试聊天效果。注意需要添加参数`--quant-policy 4`以开启KV Cache int8模式。

```sh
lmdeploy chat turbomind $HF_MODEL --model-format hf --quant-policy 4

lmdeploy chat turbomind ./models/internlm2-chat-1_8b-sft --quant-policy 4
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

```sh
lmdeploy serve gradio ./internlm-chat-7b-4bit --server-name {ip_addr} --server-port {port}

lmdeploy serve gradio ./models/internlm2-chat-1_8b-sft --server-name 127.0.0.1 --server-port 12345
```

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
lmdeploy convert internlm2 ./models/internlm2-chat-1_8b-sft --dst-path ./models/internlm2-chat-1_8b-sft-turbomind

lmdeploy chat turbomind ./models/internlm2-chat-1_8b-sft-turbomind
```

## list

```sh
> lmdeploy list --help
usage: lmdeploy list [-h] [--engine {pytorch,turbomind}]

List the supported model names.

options:
  -h, --help            show this help message and exit
  --engine {pytorch,turbomind}
                        Set the inference backend. Default: turbomind. Type: str
```

```sh
> lmdeploy list --engine pytorch
The older chat template name like "internlm2-7b", "qwen-7b" and so on are deprecated and will be removed in the future. The supported chat template names are:
baichuan2
chatglm
codellama
deepseek
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
> lmdeploy check_env                                                                                                     (mm) bash-0 | 0 (0.001s)
sys.platform: linux
Python: 3.10.13 (main, Sep 11 2023, 13:44:35) [GCC 11.2.0]
CUDA available: True
MUSA available: False
numpy_random_seed: 2147483648
GPU 0: Tesla V100-SXM2-16GB
CUDA_HOME: /usr/local/cuda:/usr/local/cuda
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
LMDeploy: 0.2.6+7541def
transformers: 4.38.1
gradio: 4.21.0
fastapi: 0.110.0
pydantic: 2.6.4
```

