from lmdeploy import pipeline, GenerationConfig, TurbomindEngineConfig, ChatTemplateConfig


if __name__ == '__main__':
    # 可以直接使用transformers的模型,会自动转换格式
    # https://lmdeploy.readthedocs.io/zh-cn/latest/api/pipeline.html#turbomindengineconfig
    backend_config = TurbomindEngineConfig(
        model_name = 'internlm2',
        model_format = 'hf', # The format of input model. `hf` meaning `hf_llama`, `llama` meaning `meta_llama`, `awq` meaning the quantized model by awq. Default: None. Type: str
        tp = 1,
        session_len = 2048,
        max_batch_size = 128,
        cache_max_entry_count = 0.4, # 调整KV Cache的占用比例为0.4
        cache_block_seq_len = 64,
        quant_policy = 0, # 默认为0, 4为开启kvcache int8 量化
        rope_scaling_factor = 0.0,
        use_logn_attn = False,
        download_dir = None,
        revision = None,
        max_prefill_token_num = 8192,
    )
    # https://lmdeploy.readthedocs.io/zh-cn/latest/_modules/lmdeploy/model.html#ChatTemplateConfig
    chat_template_config = ChatTemplateConfig(
        model_name = 'internlm2',
        system = None,
        meta_instruction = None,
    )
    # https://lmdeploy.readthedocs.io/zh-cn/latest/api/pipeline.html#generationconfig
    gen_config = GenerationConfig(
        n = 1,
        max_new_tokens = 1024,
        top_p = 0.8,
        top_k = 40,
        temperature = 0.8,
        repetition_penalty = 1.0,
        ignore_eos = False,
        random_seed = None,
        stop_words = None,
        bad_words = None,
        min_new_tokens = None,
        skip_special_tokens = True,
    )

    # https://lmdeploy.readthedocs.io/zh-cn/latest/api/pipeline.html
    # https://github.com/InternLM/lmdeploy/blob/main/lmdeploy/api.py
    pipe = pipeline(
        model_path = './models/internlm2-chat-1_8b',
        model_name = 'internlm2_chat_1_8b',
        backend_config = backend_config,
        chat_template_config = chat_template_config,
    )

    #----------------------------------------------------------------------#
    # prompts (List[str] | str | List[Dict] | List[Dict]): a batch of
    #     prompts. It accepts: string prompt, a list of string prompts,
    #     a chat history in OpenAI format or a list of chat history.
    #----------------------------------------------------------------------#
    prompts = [[{
        'role': 'user',
        'content': 'Hi, pls intro yourself'
    }], [{
        'role': 'user',
        'content': 'Shanghai is'
    }]]

    # https://github.com/InternLM/lmdeploy/blob/main/lmdeploy/serve/async_engine.py#L274
    responses = pipe(prompts, gen_config=gen_config)
    for response in responses:
        print(response)
        print('text:', response.text)
        print('generate_token_len:', response.generate_token_len)
        print('input_token_len:', response.input_token_len)
        print('session_id:', response.session_id)
        print('finish_reason:', response.finish_reason)
        print()
    # Response(text='你好，我是一款语言模型，我的名字是书生·浦语。我是由上海人工智能实验室开发的，我的设计理念是有用、诚实并且无害。我可以理解并回应英文 和中文的问题，但我不能看、听、尝、触摸、闻、移动、与物理世界交互，也无法感受情感或体验感官输入，但我可以用我自己的方式来帮助人类。', generate_token_len=77, input_token_len=108, session_id=0, finish_reason='stop')
    # text: 你好，我是一款语言模型，我的名字是书生·浦语。我是由上海人工智能实验室开发的，我的设计理念是有用、诚实并且无害。我可以理解并回应英文和中文的问题，但我不能看、听、尝、触摸、闻、移动、与物理世界交互，也无法感受情感或体验感官输入，但我可以用我自己的方式来帮助人类。
    # generate_token_len: 77
    # input_token_len: 108
    # session_id: 0
    # finish_reason: stop

    # Response(text='上海是中国的一座城市，位于中国东部沿海地区，是中国的经济、文化和交通中心之一。它是中国最大的城市之一，拥有许多重要的旅游景点、商业区和文化设施。', generate_token_len=35, input_token_len=105, session_id=1, finish_reason='stop')
    # text: 上海是中国的一座城市，位于中国东部沿海地区，是中国的经济、文化和交通中心之一。它是中国最大的城市之一，拥有许多重要的旅游景点、商业区和文化设 施。
    # generate_token_len: 35
    # input_token_len: 105
    # session_id: 1
    # finish_reason: stop

    # 流式返回处理结果
    # for item in pipe.stream_infer(prompts, gen_config=gen_config):
    #     print(item)
        # Response(text=' assist', generate_token_len=32, input_token_len=108, session_id=0, finish_reason=None)
        # Response(text='', generate_token_len=38, input_token_len=108, session_id=0, finish_reason='stop')
        # Response(text=' heritage', generate_token_len=49, input_token_len=105, session_id=1, finish_reason=None)
        # Response(text='', generate_token_len=54, input_token_len=105, session_id=1, finish_reason='stop')

        # print(item.text, end='')
        # if item.finish_reason == 'stop':
        #     print()