from torch.multiprocessing import Process, Queue, Event
import torch
import time
from transformers import pipeline, set_seed
import types
from my_pipeline1 import my_preprocess, my_sample, my_forward, my__forward, my_pad_collate_fn, my_inner, my_no_collate_fn
import numpy

def one_iter():
    generator = pipeline('text-generation', model='gpt2')

    attention_mask=torch.ones((2,8))
    input_ids=torch.ones((2,8))
    input_ids = input_ids.repeat_interleave(1, dim=0)
    print(input_ids)

if __name__=='__main__':

    p = Process(target=one_iter)

    p.start()
    while True:
        pass
