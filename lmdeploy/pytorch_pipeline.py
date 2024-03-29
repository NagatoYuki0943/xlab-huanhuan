from lmdeploy import pipeline, GenerationConfig, PytorchEngineConfig, ChatTemplateConfig


if __name__ == '__main__':
    # https://lmdeploy.readthedocs.io/zh-cn/latest/api/pipeline.html#pytorchengineconfig
    backend_config = PytorchEngineConfig(session_len=2048, cache_max_entry_count=0.5)
    # https://lmdeploy.readthedocs.io/zh-cn/latest/_modules/lmdeploy/model.html#ChatTemplateConfig
    chat_template_config = ChatTemplateConfig(model_name='internlm2', system=None)
    # https://lmdeploy.readthedocs.io/zh-cn/latest/api/pipeline.html#generationconfig
    gen_config = GenerationConfig(
        top_p=0.8,
        top_k=40,
        temperature=0.8,
        max_new_tokens=1024
    )

    # https://github.com/InternLM/lmdeploy/blob/main/lmdeploy/api.py
    pipe = pipeline(
        './models/internlm2-chat-1_8b',
        backend_config=backend_config,
        chat_template_config=chat_template_config,
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

    # https://lmdeploy.readthedocs.io/zh-cn/latest/api/pipeline.html
    # https://github.com/InternLM/lmdeploy/blob/main/lmdeploy/serve/async_engine.py#L126
    responses = pipe(prompts, gen_config=gen_config)
    for response in responses:
        print(response)
        print('text:', response.text)
        print('generate_token_len:', response.generate_token_len)
        print('input_token_len:', response.input_token_len)
        print('session_id:', response.session_id)
        print('finish_reason:', response.finish_reason)
        print()
    # Response(text='Hello! I am InternLM, a language model developed by Shanghai AI Laboratory. I am designed to help people by providing helpful, honest, and harmless responses. If you have any questions or need assistance with anything, feel free to ask!', generate_token_len=48, input_token_len=108, session_id=0, finish_reason='stop')
    # text: Hello! I am InternLM, a language model developed by Shanghai AI Laboratory. I am designed to help people by providing helpful, honest, and harmless responses. If you have any questions or need assistance with anything, feel free to ask!
    # generate_token_len: 48
    # input_token_len: 108
    # session_id: 0
    # finish_reason: stop

    # Response(text='Shanghai is the capital city of China, located in the eastern part of the country. It is the most populous city in China and is known for its rich history, cultural heritage, and modern skyline.', generate_token_len=43, input_token_len=105, session_id=1, finish_reason='stop')
    # text: Shanghai is the capital city of China, located in the eastern part of the country. It is the most populous city in China and is known for its rich history, cultural heritage, and modern skyline.
    # generate_token_len: 43
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
