from torch.multiprocessing import Process, Queue, Event
import torch
import queue
import time

def f0(q):
    x=torch.zeros((2,2))
    q.put(4676)
    q.put(x)
    q.put(46)
    q.close()
    print("aaaaa")
    # time.sleep(6)

def f1(q):
    time.sleep(2)
    x=torch.ones((2,2))
    q.put(x)
    x=q.get()
    print(x)
    try:
        x=q.get()
        print(x)
    except Exception as e:
        print(e)
    # x = q.get()
    x=q.get()
    print(x)
    x = q.get()
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