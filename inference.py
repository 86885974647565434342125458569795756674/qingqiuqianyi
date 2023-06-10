from transformers import pipeline, set_seed
import types

import my_pipeline
from my_pipeline import data, my_preprocess, my_sample, my_forward, my__forward, text2id
set_seed(42)

batch_size = 2

generator = pipeline('text-generation', model='gpt2')
generator.tokenizer.pad_token_id = generator.model.config.eos_token_id

# for out in generator(data(generator.tokenizer,generator.framework,batch_size), batch_size=batch_size,max_length=100):
#     print(out)

generator.my_preprocess = types.MethodType(my_preprocess, generator)
generator.model.sample = types.MethodType(my_sample, generator.model)
generator.model._forward = types.MethodType(my__forward, generator)
generator.model.forward = types.MethodType(my_forward, generator)


if __name__=='__main__':
    ids=text2id(generator.tokenizer,generator.framework,batch_size)
    # 加入显存
    forward_params=generator.my_preprocess(max_length=100)
    forward_params["process_list"]=[]

    for _ in range(1):
        x=data(generator.tokenizer,generator.framework,batch_size)

        process_list = generator.forward(x, **forward_params)

        forward_params["process_list"] = process_list

    for p0 in forward_params["process_list"]:
        p0.join()