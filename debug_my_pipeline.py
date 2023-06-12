import random
import numpy as np
import copy
import inspect
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

from torch import nn

import os

from transformers.generation.stopping_criteria import (
    MaxLengthCriteria,
    MaxTimeCriteria,
    StoppingCriteria,
    StoppingCriteriaList,
    validate_stopping_criteria,
)

from torch.multiprocessing import Process, Queue



import torch

import numpy as np


def text2id(q):
    attention_mask = torch.zeros((2, 2))
    # attention_mask.share_memory_()
    # attention_mask = np.zeros(
    #     (2, 2))
    x = {'attention_mask': attention_mask}
    q.put(x, block=True)


def data(q):
    ids_tensor = q.get(block=True)
    x = {}
    q.put(ids_tensor, block=True)
    return x


def forward(model_inputs, **generate_kwargs):
    process_list = generate_kwargs.pop("process_list")
    q = generate_kwargs.pop("q")
    for _ in range(2):
        p0 = Process(target=unfinished_seq, args=(q,))
        p0.start()
        # p0.join()
        process_list.append(p0)
    return process_list


def unfinished_seq(q):
    x = q.get(block=True)
    x["attention_mask"] = x["attention_mask"] + 1
    q.put(x, block=True)
