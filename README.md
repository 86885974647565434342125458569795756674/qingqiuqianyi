# -1

```text
ssh cyy@172.18.217.119
pip install virtualenv
export PATH=$PATH:/home/cyy/.local/bin
mkdir /data/test
cd /data/test
virtualenv env
source env/bin/activate
deactivate
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
export TRANSFORMERS_CACHE=/data/cyy/huggingface
git config --global https.proxy http://172.18.216.:7890   
git config --global http.proxy http://172.18.216.:7890   
scp D:/qingqiuqianyi/qingqiuqianyi/debug.py cyy@172.18.217.119:/data/cyy
```

直接从2开始看，对应的代码在src，在qingqiuqianyi目录下有相同文件，在qingqiuqianyi目录可以运行my_inference1.py

# 0

```text
from transformers import pipeline
def data():
	seq_list = ["Hello, I'm a language model,", "Who is Jack?", "I want some rice,"]
    while True:
		yield random.sample(seq_list, 1)[0]

generator = pipeline('text-generation', model='gpt2')
generator.tokenizer.pad_token_id = generator.model.config.eos_token_id
    
for out in generator(data(), batch_size=batch_size, max_length=100):
    print(out)
```

pipelines.\__init__.py.pipeline:

pipeline_class = targeted_task["impl"]

framework, model = infer_framework_load_model(...)

tokenizer = AutoTokenizer.from_pretrained(
    tokenizer_identifier, use_fast=use_fast, _from_pipeline=task, \**hub_kwargs, **tokenizer_kwargs
)

return pipeline_class(model=model, framework=framework, task=task, **kwargs)



pipelines.text_generation.py.TextGenerationPipeline.\__call__:

pipelines.base.py.Pipeline.\__call__:

pipelines.base.py.Pipeline.get_iterator:

dataset = PipelineIterator(inputs, self.preprocess, preprocess_params)

collate_fn = no_collate_fn if batch_size == 1 else pad_collate_fn(self.tokenizer, feature_extractor)
dataloader = DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, collate_fn=collate_fn)
model_iterator = PipelineIterator(dataloader, self.forward, forward_params, loader_batch_size=batch_size)
final_iterator = PipelineIterator(model_iterator, self.postprocess, postprocess_params)



pipelines.base.py.Pipeline.forward:

model_inputs = self.\_ensure_tensor_on_device(model_inputs, device=self.device)
model_outputs = self._forward(model_inputs, **forward_params)



pipelines.text_generation.py.TextGenerationPipeline.\_forward:

generated_sequence = self.model.generate(input_ids=input_ids, attention_mask=attention_mask, **generate_kwargs)



models.gpt2.modeling_gpt2.py.GPT2LMHeadModel-GPT2PreTrainedModel-PreTrainedModel-generation.utils.py.GenerationMixin.generate:

generation.utils.py.GenerationMixin.sample:

unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)

while True:

	outputs = self(
	**model_inputs,
	return_dict=True,
	output_attentions=output_attentions,
	output_hidden_states=output_hidden_states,
	)
	next_token_logits = outputs.logits[:, -1, :]
	next_token_scores = logits_warper(input_ids, next_token_scores)
	probs = nn.functional.softmax(next_token_scores, dim=-1)
	next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
	next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)
	input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
	unfinished_sequences = unfinished_sequences.mul( next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0))

return input_ids

# 1

```text
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

# Download test data from open datasets.
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

batch_size = 64

# Create data loaders.
test_dataloader = DataLoader(test_data, batch_size=batch_size num_workers=4)

iterator = iter(test_dataloader)
X, y = next(iterator)
print(f"Shape of X [N, C, H, W]: {X.shape}")
print(f"Shape of y: {y.shape} {y.dtype}")
```

DataLoader.\_\_init__:

sampler = SequentialSampler(dataset)

batch_sampler = BatchSampler(sampler, batch_size, drop_last)

DataLoader.\_\_iter__:

return self._get_iterator()

DataLoader._get_iterator:

return _MultiProcessingDataLoaderIter(self)

DataLoader._index_sampler:

return self.batch_sampler

_MultiProcessingDataLoaderIter.\_\_init__:

self._dataset = loader.dataset

self.\_index_sampler = loader._index_sampler

self.\_sampler_iter = iter(self._index_sampler)

multiprocessing_context = multiprocessing

self._worker_result_queue = multiprocessing_context.Queue()

self._workers_done_event = multiprocessing_context.Event()

self.\_index_queues = []

self._workers = []

index_queue = multiprocessing_context.Queue()

index_queue.cancel_join_thread()

target=\_utils.worker._worker_loop

w.daemon = True

w.start()

self.\_data_queue = self._worker_result_queue

self._reset(loader, first_iter=True)

\_MultiProcessingDataLoaderIter._reset:

self.\_sampler_iter = iter(self._index_sampler)

for _ in range(self.\_prefetch_factor * self.\_num_workers):
    self._try_put_index()

\_BaseDataLoaderIter._next_index:

return next(self._sampler_iter)

\_MultiProcessingDataLoaderIter._try_put_index:

index = self._next_index()

self.\_index_queues[worker_queue_idx].put((self._send_idx, index))

\_BaseDataLoaderIter.\__next__:

data = self._next_data()

return data

_MultiProcessingDataLoaderIter.\_next_data:

if not self.\_persistent_workers:
                    self._shutdown_workers()
                raise StopIteration

idx, data = self._get_data()

return self._process_data(data)

\_MultiProcessingDataLoaderIter._get_data:

while True:

​    success, data = self._try_get_data()

​    if success:

​        return data

_MultiProcessingDataLoaderIter.\_try_get_data:

data = self._data_queue.get(timeout=timeout)
return (True, data)

\_MultiProcessingDataLoaderIter._process_data:

self._try_put_index()

return data

utils.data.\_utils.worker._worker_loop:

while watchdog.is_alive():

​	r = index_queue.get(timeout=MP_STATUS_CHECK_INTERVAL)

​	elif r is None:
​                	assert done_event.is_set() or iteration_end
​               	 break

​	data = fetcher.fetch(index)

​	data_queue.put((idx, data))

​	del data, idx, index, r

if done_event.is_set():
    data_queue.cancel_join_thread()
    data_queue.close()

\_MultiProcessingDataLoaderIter._shutdown_workers:

self._workers_done_event.set()

for worker_id in range(len(self._workers)):

​	self.\_mark_worker_as_unavailable(worker_id, shutdown=True)

for w in self._workers:

​    w.join(timeout=\_utils.MP_STATUS_CHECK_INTERVAL)

for q in self._index_queues:

​    q.cancel_join_thread()

​    q.close()

for w in self._workers:

​	if w.is_alive():

​        w.terminate()

\_MultiProcessingDataLoaderIter.\_mark_worker_as_unavailable:

q = self._index_queues[worker_id]

q.put(None)

self._workers_status[worker_id] = False

# 2

preprocess:text-id,mask(tokenizer)

batch and padding

forward

postprocess



text2id:text-id,mask-padding

my_preprocess:\__call__+get_iterator

data:取数据进行前向

my_forward:去掉to(cpu)

my__forward:前向，分别处理完成和未完成seq

my_sample:一次迭代

# 3

只考虑内存

机器0：

进程0：发送prompt_id给机器1（将text转成id，发送prompt_id到Qid_to_1）



机器1：

进程0：维护id队列，从Qid_to_1得到id，将id加入id队列

进程1：从Qid_req得到req_id, batch\_size，从id队列取id，发送batching id到Qid_to_?



机器2：

进程0：发送2, batch_size到Qid_req，从Qid_to_2获取id，forward，已完成的seq发送Q2_0_to_1，没完成的seq发送Q2_0_to_2

进程1：从Q2_0_to_1得到id，将id转成text，打印

进程2：从Q2_0_to_2得到id，通过Qid_to_1发送id



机器3：

进程0：发送3, batch_size到Qid_req，从Qid_to_3获取id，forward，已完成的seq发送Q3_0_to_1，没完成的seq发送Q3_0_to_2

进程1：将id转成text，打印

进程2：通过Qid_to_1发送id

# 4

哪些放到显存上，怎么放到显存上，缺少全局kv_cache管理器

监控显存大小：一个新进程轮询显存大小
