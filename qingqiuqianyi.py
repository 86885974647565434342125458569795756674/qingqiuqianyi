from multiprocessing import Process, Queue
from queue import Empty, Full
import numpy as np
seq_list = ["Hello, I'm a language model,", "Who is Jack?", "I want some rice,"]
data_generator_batch_size=10

def data_generator(batch_size=128):
    return np.random.choice(seq_list, batch_size)

def request_queue(q):
    while True:
        batch_data=data_generator(batch_size=data_generator_batch_size)
        for data in batch_data:
            try:
                q.put(data,block=False)
            except Full:
                print("queue is full", flush=True)
            except Exception:
                exit()
        break

def gpu1(q):
    while True:
        try:
            data=q.get(block=False)
            print(data)
        except Empty:
            # print("queue is empty", flush=True)
            pass
        except Exception:
            exit()


def gpu2(q):
    while True:
        try:
            data=q.get(block=False)
            print(data)
        except Empty:
            # print("queue is empty", flush=True)
            pass
        except Exception:
            exit()

if __name__=='__main__':
    q = Queue()
    p0 = Process(target=request_queue,args=(q,))
    p1 = Process(target=gpu1,args=(q,))
    p2 = Process(target=gpu2, args=(q,))
    p0.start()
    p1.start()
    p2.start()
    p0.join()
    p1.join()
    p2.join()
