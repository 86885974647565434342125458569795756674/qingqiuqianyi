from torch.multiprocessing import Process, Queue, Event
import torch
import queue
import time

def f0(q):
    x=torch.zeros((2,2))
    time.sleep(3)
    q.put(x)
    print("aaaaa")

def f1(q):
    x=q.get()
    print("dd")
    print(x)

if __name__=="__main__":
    q=Queue()
    p0=Process(target=f0,args=(q,))
    p1=Process(target=f1,args=(q,))
    p0.start()
    p1.start()
    # time.sleep(102)
    p0.join()
    p1.join()