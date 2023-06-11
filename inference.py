from transformers import pipeline, set_seed
import types
from my_pipeline import data, my_preprocess, my_sample, my_forward, my__forward, text2id
from multiprocessing import Process, Queue
from transformers.pipelines.base import no_collate_fn, pad_collate_fn

if __name__=='__main__':
    set_seed(42)
    text_size = 8
    max_length = 10
    batch_size = 2
    epoch=4
    generator = pipeline('text-generation', model='gpt2')
    generator.tokenizer.pad_token_id = generator.model.config.eos_token_id

    # for out in generator(data(generator.tokenizer,generator.framework,batch_size), batch_size=batch_size,max_length=100):
    #     print(out)

    generator.my_preprocess = types.MethodType(my_preprocess, generator)
    generator.model.sample = types.MethodType(my_sample, generator.model)
    generator._forward = types.MethodType(my__forward, generator)
    generator.forward = types.MethodType(my_forward, generator)

    q=Queue()
    q.put(None, block=True)

    feature_extractor = generator.feature_extractor if generator.feature_extractor is not None else generator.image_processor
    collate_fn = no_collate_fn if batch_size == 1 else pad_collate_fn(generator.tokenizer, feature_extractor)
    text2id(q,generator.tokenizer,generator.framework,collate_fn,text_size)

    forward_params,postprocess_params=generator.my_preprocess(max_length=max_length)
    forward_params["postprocess_params"]=postprocess_params
    forward_params["process_list"]=[]
    forward_params["q"]=q
    forward_params["collate_fn"]=collate_fn

    for _ in range(epoch):
        x=data(q,batch_size)
        process_list = generator.forward(x, **forward_params)
        forward_params["process_list"] = process_list

    for p0 in forward_params["process_list"]:
        p0.join()

    ids_tensor = q.get(block=True)
    for i in ids_tensor["prompt_text"]:
        print(i)