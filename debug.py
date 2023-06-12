from torch.multiprocessing import Process, Queue
import torch


def unfinished_seq(q):
    x = q.get(block=True)
    x = x + 1
    print(x)
    q.put(x, block=True)


if __name__ == "__main__":

    process_list = []
    q = Queue()
    x = torch.zeros((2, 2))
    q.put(x, block=True)

    for _ in range(2):
        p0 = Process(target=unfinished_seq, args=(q,))
        p0.start()
        # p0.join()
        process_list.append(p0)

    for p0 in process_list:
        p0.join()