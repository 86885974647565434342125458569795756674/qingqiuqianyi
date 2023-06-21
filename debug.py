from torch.multiprocessing import Process, Queue
import torch


def unfinished_seq(q):
    x = q.get()
    x = x + 1
    print(x)
    q.put(x)
    del x
    q.close()


if __name__ == "__main__":

    process_list = []
    q = Queue()
    x = torch.zeros((2, 2))
    q.put(x, block=True)
    del x

    for _ in range(2):
        p = Process(target=unfinished_seq, args=(q,))
        p.start()
        process_list.append(p)

    # x = q.get()
    # print(x,0)
    # del x

    for p0 in process_list:
        p0.join(timeout=5.0)

    q.close()

    for p in process_list:
        p.terminate()

