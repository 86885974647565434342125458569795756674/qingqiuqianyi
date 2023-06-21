import torch.multiprocessing as multiprocessing
import torch
import queue

MP_STATUS_CHECK_INTERVAL = 5.0


def _worker_loop(index_queue, data_queue, done_event):

    while True:
        try:
            r = index_queue.get(timeout=MP_STATUS_CHECK_INTERVAL)
        except queue.Empty:
            if done_event.is_set():
                data_queue.cancel_join_thread()
                data_queue.close()
                return
            continue
        print(r)
        data = torch.ones((2, 2))
        data_queue.put(data)
        del data, r


if __name__ == '__main__':
    multiprocessing_context = multiprocessing
    _worker_result_queue = multiprocessing_context.Queue()
    _workers_done_event = multiprocessing_context.Event()
    index_queue = multiprocessing_context.Queue()
    w = multiprocessing_context.Process(
        target=_worker_loop,
        args=(index_queue, _worker_result_queue, _workers_done_event,))
    w.daemon = True
    w.start()
    _data_queue = _worker_result_queue
    index = torch.zeros((2, 2))
    index_queue.put(index)
    del index
    while True:
        try:
            data = _data_queue.get(timeout=MP_STATUS_CHECK_INTERVAL)
        except queue.Empty:
            continue
        print(data)
        del data
        break

    _workers_done_event.set()
    w.join(timeout=MP_STATUS_CHECK_INTERVAL)

    index_queue.cancel_join_thread()
    index_queue.close()

    if w.is_alive():
        w.terminate()
