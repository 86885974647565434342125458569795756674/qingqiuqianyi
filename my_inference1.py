from torch.multiprocessing import Process, Queue, Event
import torch
import queue
import time

MP_STATUS_CHECK_INTERVAL = 2.0
RUN_TIME = 4.0
GPU_NUM = 2


def prompt_to_id(done_event, Qid_to_1):
    while True:
        if done_event.is_set():
            break
        xid = torch.zeros((2, 2))
        Qid_to_1.put(xid)
        del xid
        time.sleep(1)


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


def send_id_from_id_queue(done_event, q, Qid_req, id_to_gpu_list):
    while True:
        try:
            req_id = Qid_req.get()
        except Exception:
            assert done_event.is_set()
            break
        if req_id is None:
            assert done_event.is_set()
            break
        try:
            xid = q.get()
        except Exception:
            assert done_event.is_set()
            break
        if xid is None:
            assert done_event.is_set()
            break
        id_to_gpu_list[req_id].put(xid)
        del req_id
        del xid



def one_iter(done_event, mid, forward, Qid_req, Qid_to_2, Q2_0_to_1, Q2_0_to_2):
    while True:
        Qid_req.put(mid)
        try:
            xid = Qid_to_2.get()
        except Exception:
            assert done_event.is_set()
            break
        if xid is None:
            assert done_event.is_set()
            break
        xid = forward(xid)
        if True:
            Q2_0_to_1.put(xid)
        else:
            Q2_0_to_2.put(xid)
        del xid


def finished_seq(done_event, Q2_0_to_1):
    while True:
        try:
            xid = Q2_0_to_1.get()
        except Exception:
            assert done_event.is_set()
            break
        if xid is None:
            assert done_event.is_set()
            break
        del xid


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
        Qid_to_1.put(xid)
        del xid


def forward(x):
    x += 1
    return x


if __name__ == "__main__":
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
    p0_0 = Process(target=prompt_to_id, args=(done_event, Qid_to_1,))
    process_list.append(p0_0)
    p1_0 = Process(target=id_queue, args=(done_event, q, Qid_to_1,))
    process_list.append(p1_0)

    id_to_gpu_list=[]
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

        pgpu_0 = Process(target=one_iter, args=(done_event, mid, forward, Qid_req, Qid_to_gpu, Qgpu_0_to_1, Qgpu_0_to_2))
        process_list.append(pgpu_0)
        pgpu_1 = Process(target=finished_seq, args=(done_event, Qgpu_0_to_1,))
        process_list.append(pgpu_1)
        pgpu_2 = Process(target=unfinished_seq, args=(done_event, Qid_to_1, Qgpu_0_to_2,))
        process_list.append(pgpu_2)

    p1_1 = Process(target=send_id_from_id_queue, args=(done_event, q, Qid_req, id_to_gpu_list))
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
