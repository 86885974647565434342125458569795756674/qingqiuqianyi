from torch.multiprocessing import Process, Queue, Event
import torch
import time
from transformers import pipeline, set_seed
import types
from my_pipeline1 import my_preprocess, my_sample, my_forward, my__forward, my_pad_collate_fn, my_inner, my_no_collate_fn
import numpy


def prompt_to_id(done_event, Qid_to_1, **kwargs):
    tokenizer = kwargs["tokenizer"]
    framework = kwargs["framework"]
    text_size = kwargs["text_size"]
    if text_size <= 0:
        raise ValueError("text_size<=0")
    seq_list = ["Hello, I'm a language model,", "Who is Jack?", "I want some rice,"]
    while True:
        if done_event.is_set():
            break
        prompt_texts = numpy.random.choice(seq_list, text_size)
        for prompt_text in prompt_texts:
            inputs = tokenizer(prompt_text, padding=False, add_special_tokens=False, return_tensors=framework)
            inputs["prompt_text"] = prompt_text
            Qid_to_1.put(inputs)
            del inputs
        break

def id_queue(done_event, q, Qid_to_1):
    while True:
        try:
            xid = Qid_to_1.get()
        except Exception:
            assert done_event.is_set()
            break
        if xid is None:
            assert done_event.is_set()
            break
        q.put(xid)
        del xid


def send_id_from_id_queue(done_event, q, Qid_req, id_to_gpu_list, **kwargs):
    collate_fn_params=kwargs["collate_fn_params"]
    collate_fn=kwargs["collate_fn"]
    while True:
        try:
            req_id_batch_size = Qid_req.get()
        except Exception:
            assert done_event.is_set()
            break
        if req_id_batch_size is None:
            assert done_event.is_set()
            break
        req_id, batch_size = req_id_batch_size
        items=[]
        for _ in range(batch_size):
            try:
                xid = q.get()
            except Exception:
                assert done_event.is_set()
                return
            if xid is None:
                assert done_event.is_set()
                return
            items.append(xid)
        xid = collate_fn(items, **collate_fn_params)
        id_to_gpu_list[req_id].put(xid)
        del req_id, xid



def one_iter(done_event, mid, batch_size, forward, Qid_req, Qid_to_2, Q2_0_to_1, Q2_0_to_2, forward_params):
    if batch_size <= 0:
        raise ValueError("batch size<=0")

    while True:
        Qid_req.put((mid, batch_size))
        try:
            xid = Qid_to_2.get()
        except Exception:
            assert done_event.is_set()
            break
        if xid is None:
            assert done_event.is_set()
            break
        generated_sequence, attention_mask, unfinished_sequences, prompt_text, input_ids, self = forward(xid, **forward_params)

        for index, seq in enumerate(unfinished_sequences):
            if seq == 1:
                Q2_0_to_2.put((generated_sequence[index], attention_mask[index], prompt_text[index],))

            else:
                Q2_0_to_1.put((self, {"generated_sequence": [generated_sequence[index]], "input_ids": input_ids[index], "prompt_text": prompt_text[index]},))
        del xid, generated_sequence, attention_mask, unfinished_sequences, prompt_text, input_ids, self


def finished_seq(done_event, Q2_0_to_1, postprocess_params):
    while True:
        try:
            xid = Q2_0_to_1.get()
        except Exception:
            assert done_event.is_set()
            break
        if xid is None:
            assert done_event.is_set()
            break
        self, model_outputs = xid
        with self.device_placement():
            if self.framework == "tf":
                pass
            elif self.framework == "pt":
                inference_context = self.get_inference_context()
                with inference_context():
                    model_outputs = self._ensure_tensor_on_device(model_outputs, device=torch.device("cpu"))
            else:
                raise ValueError(f"Framework {self.framework} is not supported")
        output = self.postprocess(model_outputs, **postprocess_params)
        print(output)
        del self, model_outputs, output


def unfinished_seq(done_event, Qid_to_1, Q2_0_to_2):
    while True:
        try:
            xid = Q2_0_to_2.get()
        except Exception:
            assert done_event.is_set()
            break
        if xid is None:
            assert done_event.is_set()
            break
        input_ids, attention_mask, prompt_text = xid
        if len(input_ids.shape) == 1:
            input_ids = torch.unsqueeze(input_ids, 0)
            attention_mask = torch.unsqueeze(attention_mask, 0)
        inputs = {"input_ids": input_ids, "attention_mask": attention_mask, "prompt_text":prompt_text}
        Qid_to_1.put(inputs)
        del xid, inputs

if __name__ == "__main__":
    set_seed(42)
    max_length = 10
    batch_size = 2
    text_size = batch_size * 4

    RUN_TIME = 10.0
    GPU_NUM = 2

    # on cpu for now
    generator = pipeline('text-generation', model='gpt2')
    generator.tokenizer.pad_token_id = generator.model.config.eos_token_id

    # 重载函数
    generator.my_preprocess = types.MethodType(my_preprocess, generator)
    generator.model.sample = types.MethodType(my_sample, generator.model)
    generator._forward = types.MethodType(my__forward, generator)
    generator.forward = types.MethodType(my_forward, generator)

    feature_extractor = generator.feature_extractor if generator.feature_extractor is not None else generator.image_processor
    f_padding_value, t_padding_value, padding_side = my_pad_collate_fn(generator.tokenizer, feature_extractor)
    collate_fn_params = {"tokenizer": generator.tokenizer, "feature_extractor": feature_extractor,
                         "f_padding_value": f_padding_value, "t_padding_value": t_padding_value,
                         "padding_side": padding_side}
    collate_fn = my_no_collate_fn if batch_size == 1 else my_inner

    # 结束所有进程的信号
    done_event = Event()

    q_list=[]
    # 请求池
    q = Queue()
    q_list.append(q)

    # 从prompt到请求队列
    Qid_to_1 = Queue()
    q_list.append(Qid_to_1)

    # 申请id进行forward
    Qid_req = Queue()
    q_list.append(Qid_req)

    process_list = []
    prompt_to_id_params = {"tokenizer": generator.tokenizer, "framework": generator.framework, "text_size": text_size}
    p0_0 = Process(target=prompt_to_id, args=(done_event, Qid_to_1, prompt_to_id_params))
    process_list.append(p0_0)
    p1_0 = Process(target=id_queue, args=(done_event, q, Qid_to_1,))
    process_list.append(p1_0)


    forward_params, postprocess_params = generator.my_preprocess(max_length=max_length)

    id_to_gpu_list = []
    # 3元组
    for mid in range(GPU_NUM):
        # 发送id到mid
        Qid_to_gpu = Queue()
        q_list.append(Qid_to_gpu)
        id_to_gpu_list.append(Qid_to_gpu)

        # 发送已完成id
        Qgpu_0_to_1 = Queue()
        q_list.append(Qgpu_0_to_1)

        # 发送未完成id
        Qgpu_0_to_2 = Queue()
        q_list.append(Qgpu_0_to_2)

        pgpu_0 = Process(target=one_iter, args=(done_event, mid, batch_size, generator.forward, Qid_req, Qid_to_gpu, Qgpu_0_to_1, Qgpu_0_to_2, forward_params))
        process_list.append(pgpu_0)
        pgpu_1 = Process(target=finished_seq, args=(done_event, Qgpu_0_to_1, postprocess_params,))
        process_list.append(pgpu_1)
        pgpu_2 = Process(target=unfinished_seq, args=(done_event, Qid_to_1, Qgpu_0_to_2,))
        process_list.append(pgpu_2)

    send_id_from_id_queue_param={"collate_fn_params":collate_fn_params,"collate_fn":collate_fn}
    p1_1 = Process(target=send_id_from_id_queue, args=(done_event, q, Qid_req, id_to_gpu_list,send_id_from_id_queue_param))
    process_list.append(p1_1)

    # 执行
    for p in process_list:
        p.start()
    time.sleep(RUN_TIME)

    # 结束
    done_event.set()
    for q in q_list:
        q.put(None)
    for p in process_list:
        p.join()
