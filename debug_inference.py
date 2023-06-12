from transformers import pipeline, set_seed
import types
from debug_my_pipeline import data, forward, text2id

from torch.multiprocessing import Process, Queue

if __name__ == '__main__':

    epoch = 2
    q = Queue()
    text2id(q)
    forward_params = {"q": q, "process_list": []}
    for _ in range(epoch):
        x = data(q)
        process_list = forward(x, **forward_params)
        forward_params["process_list"] = process_list

    for p0 in forward_params["process_list"]:
        p0.join()

    ids_tensor = q.get(block=True)
    for i in ids_tensor["attention_mask"]:
        print(i)
