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

你们直接从2开始看，对应的代码在src，在qingqiuqianyi目录下有相同文件，在qingqiuqianyi目录可以运行my_inference1.py

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

# 2

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

# 3

哪些放到显存上，怎么放到显存上，缺少全局kv_cache管理器

监控显存大小：一个新进程轮询显存大小



目前先看看pytorch源码，再看下一步做什么
