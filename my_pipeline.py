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

from multiprocessing import Process, Queue

import torch


def my_pad(items, key, padding_value, padding_side):
    batch_size = len(items)
    if isinstance(items[0][key], torch.Tensor):
        # Others include `attention_mask` etc...
        shape = items[0][key].shape
        dim = len(shape)
        if key in ["pixel_values", "image"]:
            # This is probable image so padding shouldn't be necessary
            # B, C, H, W
            return torch.cat([item[key] for item in items], dim=0)
        elif dim == 4 and key == "input_features":
            # this is probably a mel spectrogram batched
            return torch.cat([item[key] for item in items], dim=0)
        max_length = max(item[key].shape[1] for item in items)
        min_length = min(item[key].shape[1] for item in items)
        dtype = items[0][key].dtype

        if dim == 2:
            if max_length == min_length:
                # Bypass for `ImageGPT` which doesn't provide a padding value, yet
                # we can consistently pad since the size should be matching
                return torch.cat([item[key] for item in items], dim=0)
            tensor = torch.zeros((batch_size, max_length), dtype=dtype) + padding_value
        elif dim == 3:
            tensor = torch.zeros((batch_size, max_length, shape[-1]), dtype=dtype) + padding_value
        elif dim == 4:
            tensor = torch.zeros((batch_size, max_length, shape[-2], shape[-1]), dtype=dtype) + padding_value

        for i, item in enumerate(items):
            if dim == 2:
                if padding_side == "left":
                    tensor[i, -len(item[key][0]):] = item[key][0].clone()
                else:
                    tensor[i, : len(item[key][0])] = item[key][0].clone()
            elif dim == 3:
                if padding_side == "left":
                    tensor[i, -len(item[key][0]):, :] = item[key][0].clone()
                else:
                    tensor[i, : len(item[key][0]), :] = item[key][0].clone()
            elif dim == 4:
                if padding_side == "left":
                    tensor[i, -len(item[key][0]):, :, :] = item[key][0].clone()
                else:
                    tensor[i, : len(item[key][0]), :, :] = item[key][0].clone()

        return tensor
    else:
        return [item[key] for item in items]


def my_no_collate_fn(items, tokenizer, feature_extractor, f_padding_value, t_padding_value, padding_side):
    if len(items) != 1:
        raise ValueError("This collate_fn is meant to be used with batch_size=1")
    return items[0]


def my_pad_collate_fn(tokenizer, feature_extractor):
    f_padding_value = None
    # Tokenizer
    t_padding_side = None
    # Feature extractor
    f_padding_side = None
    if tokenizer is None and feature_extractor is None:
        raise ValueError("Pipeline without tokenizer or feature_extractor cannot do batching")
    if tokenizer is not None:
        if tokenizer.pad_token_id is None:
            raise ValueError(
                "Pipeline with tokenizer without pad_token cannot do batching. You can try to set it with "
                "`pipe.tokenizer.pad_token_id = model.config.eos_token_id`."
            )
        else:
            t_padding_value = tokenizer.pad_token_id
            t_padding_side = tokenizer.padding_side
    if feature_extractor is not None:
        # Feature extractor can be images, where no padding is expected
        f_padding_value = getattr(feature_extractor, "padding_value", None)
        f_padding_side = getattr(feature_extractor, "padding_side", None)

    if t_padding_side is not None and f_padding_side is not None and t_padding_side != f_padding_side:
        raise ValueError(
            f"The feature extractor, and tokenizer don't agree on padding side {t_padding_side} != {f_padding_side}"
        )
    padding_side = "right"
    if t_padding_side is not None:
        padding_side = t_padding_side
    if f_padding_side is not None:
        padding_side = f_padding_side
    return f_padding_value, t_padding_value, padding_side


def my_inner(items, tokenizer, feature_extractor, f_padding_value, t_padding_value, padding_side):
    keys = set(items[0].keys())
    for item in items:
        if set(item.keys()) != keys:
            raise ValueError(
                f"The elements of the batch contain different keys. Cannot batch them ({set(item.keys())} !="
                f" {keys})"
            )
    # input_values, input_pixels, input_ids, ...
    padded = {}
    no_padded = {}
    for key in keys:
        if key in {"input_ids"}:
            # ImageGPT uses a feature extractor
            if tokenizer is None and feature_extractor is not None:
                _padding_value = f_padding_value
            else:
                _padding_value = t_padding_value
        elif key in {"input_values", "pixel_values", "input_features"}:
            _padding_value = f_padding_value
        elif key in {"p_mask", "special_tokens_mask"}:
            _padding_value = 1
        elif key in {"attention_mask", "token_type_ids"}:
            _padding_value = 0
        else:
            # This is likely another random key maybe even user provided
            _padding_value = 0
        padded[key] = my_pad(items, key, _padding_value, padding_side)
        # no_padded[key]=_no_pad(items, key, _padding_value, padding_side)
    return padded
    # return no_padded


def text2id(q, tokenizer, framework, collate_fn, collate_fn_params, text_size=8):
    if text_size <= 0:
        raise ValueError("text_size<=0")
    seq_list = ["Hello, I'm a language model,", "Who is Jack?", "I want some rice,"]
    # while True:
    #     yield random.sample(seq_list, 1)[0]
    prompt_texts = np.random.choice(seq_list, text_size)
    items = []
    for prompt_text in prompt_texts:
        inputs = tokenizer(prompt_text, padding=False, add_special_tokens=False, return_tensors=framework)
        inputs["prompt_text"] = prompt_text
        items.append(inputs)
    # items:[{'input_ids': tensor([[   40,   765,   617, 11464,    11]]), 'attention_mask': tensor([[1, 1, 1, 1, 1]]), 'prompt_text': 'I want some rice,'}, {'input_ids': tensor([[15496,    11,   314,  1101,   257,  3303,  2746,    11]]), 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1]]), 'prompt_text': "Hello, I'm a language model,"}, {'input_ids': tensor([[   40,   765,   617, 11464,    11]]), 'attention_mask': tensor([[1, 1, 1, 1, 1]]), 'prompt_text': 'I want some rice,'}, {'input_ids': tensor([[   40,   765,   617, 11464,    11]]), 'attention_mask': tensor([[1, 1, 1, 1, 1]]), 'prompt_text': 'I want some rice,'}, {'input_ids': tensor([[15496,    11,   314,  1101,   257,  3303,  2746,    11]]), 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1]]), 'prompt_text': "Hello, I'm a language model,"}, {'input_ids': tensor([[15496,    11,   314,  1101,   257,  3303,  2746,    11]]), 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1]]), 'prompt_text': "Hello, I'm a language model,"}, {'input_ids': tensor([[   40,   765,   617, 11464,    11]]), 'attention_mask': tensor([[1, 1, 1, 1, 1]]), 'prompt_text': 'I want some rice,'}, {'input_ids': tensor([[8241,  318, 3619,   30]]), 'attention_mask': tensor([[1, 1, 1, 1]]), 'prompt_text': 'Who is Jack?'}]
    # ids_tensor:{'attention_mask': tensor([[1, 1, 1, 1, 1, 0, 0, 0],
    #                            [1, 1, 1, 1, 1, 1, 1, 1],
    #                            [1, 1, 1, 1, 1, 0, 0, 0],
    #                            [1, 1, 1, 1, 1, 0, 0, 0],
    #                            [1, 1, 1, 1, 1, 1, 1, 1],
    #                            [1, 1, 1, 1, 1, 1, 1, 1],
    #                            [1, 1, 1, 1, 1, 0, 0, 0],
    #                            [1, 1, 1, 1, 0, 0, 0, 0]]),
    #  'prompt_text': ['I want some rice,', "Hello, I'm a language model,", 'I want some rice,', 'I want some rice,',
    #                  "Hello, I'm a language model,", "Hello, I'm a language model,", 'I want some rice,',
    #                  'Who is Jack?'], 'input_ids': tensor([[40, 765, 617, 11464, 11, 50256, 50256, 50256],
    #                                                        [15496, 11, 314, 1101, 257, 3303, 2746, 11],
    #                                                        [40, 765, 617, 11464, 11, 50256, 50256, 50256],
    #                                                        [40, 765, 617, 11464, 11, 50256, 50256, 50256],
    #                                                        [15496, 11, 314, 1101, 257, 3303, 2746, 11],
    #                                                        [15496, 11, 314, 1101, 257, 3303, 2746, 11],
    #                                                        [40, 765, 617, 11464, 11, 50256, 50256, 50256],
    #                                                        [8241, 318, 3619, 30, 50256, 50256, 50256, 50256]])}
    ids_tensor = q.get(block=True)
    if ids_tensor is not None:
        items = dict_list2list_dict(ids_tensor, items)
    x = collate_fn(items, **collate_fn_params)
    if len(x["input_ids"].shape) == 1:
        x["input_ids"] = torch.unsqueeze(x["input_ids"], 0)
        x["attention_mask"] = torch.unsqueeze(x["attention_mask"], 0)
    if isinstance(x['prompt_text'], str):
        x['prompt_text'] = [x['prompt_text']]
    q.put(x, block=True)


def dict_list2list_dict(ids_tensor, items):
    # {'input_ids': tensor([[   40,   765,   617, 11464,    11]]), 'attention_mask': tensor([[1, 1, 1, 1, 1]]), 'prompt_text': 'I want some rice,'}
    keys = ids_tensor.keys()
    length = len(ids_tensor[list(ids_tensor)[0]])
    for i in range(length):
        new_item = {}
        for key in keys:
            if isinstance(ids_tensor[key][i],torch.Tensor):
                new_item[key] = torch.unsqueeze(ids_tensor[key][i], 0)
            else:
                new_item[key]=ids_tensor[key][i]
        items.append(new_item)
    return items


def data(q, batch_size=2):
    if batch_size <= 0:
        raise ValueError("batch size<=0")
    while True:
        ids_tensor = q.get(block=True)
        if len(ids_tensor[list(ids_tensor)[0]]) >= batch_size:
            break
        q.put(ids_tensor, block=True)
    x = {}
    for key in ids_tensor.keys():
        x[key] = ids_tensor[key][:batch_size]
        ids_tensor[key] = ids_tensor[key][batch_size:]
    q.put(ids_tensor, block=True)
    return x


def my_preprocess(self, **kwargs):
    preprocess_params, forward_params, postprocess_params = self._sanitize_parameters(**kwargs)

    # Fuse __init__ params and __call__ params without modifying the __init__ ones.
    forward_params = {**self._forward_params, **forward_params}
    postprocess_params = {**self._postprocess_params, **postprocess_params}

    self.call_count += 1
    if self.call_count > 10 and self.framework == "pt" and self.device.type == "cuda":
        warnings.warn(
            "You seem to be using the pipelines sequentially on GPU. In order to maximize efficiency please use a"
            " dataset",
            UserWarning,
        )

    if "TOKENIZERS_PARALLELISM" not in os.environ:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
    return forward_params, postprocess_params


def my_sample(
        self,
        input_ids: torch.LongTensor,
        logits_processor=None,
        stopping_criteria=None,
        logits_warper=None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        synced_gpus: bool = False,
        streamer=None,
        **model_kwargs,
):
    # logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
    # stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
    if max_length is not None:
        warnings.warn(
            "`max_length` is deprecated in this function, use"
            " `stopping_criteria=StoppingCriteriaList(MaxLengthCriteria(max_length=max_length))` instead.",
            UserWarning,
        )
        stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
    # logits_warper = logits_warper if logits_warper is not None else LogitsProcessorList()
    pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
    eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id
    if isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]
    eos_token_id_tensor = torch.tensor(eos_token_id).to(input_ids.device) if eos_token_id is not None else None
    output_scores = output_scores if output_scores is not None else self.generation_config.output_scores
    output_attentions = (
        output_attentions if output_attentions is not None else self.generation_config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
    )
    return_dict_in_generate = (
        return_dict_in_generate
        if return_dict_in_generate is not None
        else self.generation_config.return_dict_in_generate
    )

    # init attention / hidden states / scores tuples
    scores = () if (return_dict_in_generate and output_scores) else None
    decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
    cross_attentions = () if (return_dict_in_generate and output_attentions) else None
    decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

    # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
    if return_dict_in_generate and self.config.is_encoder_decoder:
        encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
        encoder_hidden_states = (
            model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
        )

    # keep track of which sequences are already finished
    unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)

    # auto-regressive generation
    # while True:

    # tensor([[40, 765, 617, 11464, 11, 50256, 50256, 50256],
    #         [15496, 11, 314, 1101, 257, 3303, 2746, 11]])
    # {'attention_mask': tensor([[1, 1, 1, 1, 1, 0, 0, 0],
    #                            [1, 1, 1, 1, 1, 1, 1, 1]]), 'use_cache': True}

    # {'input_ids': tensor([[40, 765, 617, 11464, 11, 50256, 50256, 50256],
    #                       [15496, 11, 314, 1101, 257, 3303, 2746, 11]]), 'past_key_values': None, 'use_cache': True,
    #  'position_ids': tensor([[0, 1, 2, 3, 4, 1, 1, 1],
    #                          [0, 1, 2, 3, 4, 5, 6, 7]]), 'attention_mask': tensor([[1, 1, 1, 1, 1, 0, 0, 0],
    #                                                                                [1, 1, 1, 1, 1, 1, 1, 1]]),
    #  'token_type_ids': None}

    # prepare model inputs
    model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

    # logits, past_key_values

    # forward pass to get next token
    outputs = self(
        **model_inputs,
        return_dict=True,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
    )

    next_token_logits = outputs.logits[:, -1, :]

    # pre-process distribution
    next_token_scores = logits_processor(input_ids, next_token_logits)
    next_token_scores = logits_warper(input_ids, next_token_scores)

    # Store scores, attentions and hidden_states when required
    if return_dict_in_generate:
        if output_scores:
            scores += (next_token_scores,)
        if output_attentions:
            decoder_attentions += (
                (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
            )
            if self.config.is_encoder_decoder:
                cross_attentions += (outputs.cross_attentions,)

        if output_hidden_states:
            decoder_hidden_states += (
                (outputs.decoder_hidden_states,)
                if self.config.is_encoder_decoder
                else (outputs.hidden_states,)
            )

    # sample
    probs = nn.functional.softmax(next_token_scores, dim=-1)
    next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)

    # finished sentences should have their next token be a padding token
    if eos_token_id is not None:
        if pad_token_id is None:
            raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
        next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

    # tensor([[   40,   765,   617, 11464,    11, 50256, 50256, 50256,     1],
    #         [15496,    11,   314,  1101,   257,  3303,  2746,    11,   290]])

    # update generated ids, model inputs, and length for next step
    input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
    if streamer is not None:
        streamer.put(next_tokens.cpu())

    model_kwargs = self._update_model_kwargs_for_generation(
        outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
    )

    # if eos_token was found in one sentence, set sentence to finished
    if eos_token_id_tensor is not None:
        unfinished_sequences = unfinished_sequences.mul(
            next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
        )

        # stop when each sentence is finished
        # if unfinished_sequences.max() == 0:
        #     this_peer_finished = True

    # stop if we exceed the maximum length
    if stopping_criteria(input_ids, scores):
        unfinished_sequences = torch.zeros(input_ids.shape[0], dtype=torch.long, device=input_ids.device)

    if streamer is not None:
        streamer.end()

    # if return_dict_in_generate:
    #     if self.config.is_encoder_decoder:
    #         return SampleEncoderDecoderOutput(
    #             sequences=input_ids,
    #             scores=scores,
    #             encoder_attentions=encoder_attentions,
    #             encoder_hidden_states=encoder_hidden_states,
    #             decoder_attentions=decoder_attentions,
    #             cross_attentions=cross_attentions,
    #             decoder_hidden_states=decoder_hidden_states,
    #         )
    #     else:
    #         return SampleDecoderOnlyOutput(
    #             sequences=input_ids,
    #             scores=scores,
    #             attentions=decoder_attentions,
    #             hidden_states=decoder_hidden_states,
    #         )
    # else:
    #     return input_ids
    return input_ids, model_kwargs['attention_mask'], unfinished_sequences


def my__forward(self, model_inputs, **generate_kwargs):
    input_ids = model_inputs["input_ids"]
    attention_mask = model_inputs.get("attention_mask", None)
    # Allow empty prompts
    if input_ids.shape[1] == 0:
        input_ids = None
        attention_mask = None
    #     in_b = 1
    # else:
    #     in_b = input_ids.shape[0]
    prompt_text = model_inputs.pop("prompt_text")

    postprocess_params = generate_kwargs.pop("postprocess_params")
    process_list = generate_kwargs.pop("process_list")
    q = generate_kwargs.pop("q")
    collate_fn = generate_kwargs.pop("collate_fn")
    collate_fn_params = generate_kwargs.pop("collate_fn_params")

    # If there is a prefix, we may need to adjust the generation length. Do so without permanently modifying
    # generate_kwargs, as some of the parameterization may come from the initialization of the pipeline.
    generate_kwargs = copy.deepcopy(generate_kwargs)
    prefix_length = generate_kwargs.pop("prefix_length", 0)
    if prefix_length > 0:
        has_max_new_tokens = "max_new_tokens" in generate_kwargs or (
                "generation_config" in generate_kwargs
                and generate_kwargs["generation_config"].max_new_tokens is not None
        )
        if not has_max_new_tokens:
            generate_kwargs["max_length"] = generate_kwargs.get("max_length") or self.model.config.max_length
            generate_kwargs["max_length"] += prefix_length
        has_min_new_tokens = "min_new_tokens" in generate_kwargs or (
                "generation_config" in generate_kwargs
                and generate_kwargs["generation_config"].min_new_tokens is not None
        )
        if not has_min_new_tokens and "min_length" in generate_kwargs:
            generate_kwargs["min_length"] += prefix_length

    # BS x SL
    generated_sequence, attention_mask, unfinished_sequences = self.model.generate(input_ids=input_ids,
                                                                                   attention_mask=attention_mask,
                                                                                   **generate_kwargs)
    for index, seq in enumerate(unfinished_sequences):
        if seq == 1:
            p0 = Process(target=unfinished_seq,
                         args=(q, generated_sequence[index], attention_mask[index], prompt_text[index], collate_fn,
                               collate_fn_params,))
            p0.start()
            # p0.join()
            process_list.append(p0)
        else:
            p0 = Process(target=finished_seq, args=(self, {"generated_sequence": [generated_sequence[index]],
                                                           "input_ids": input_ids[index],
                                                           "prompt_text": prompt_text[index]}, postprocess_params,))
            p0.start()
            # p0.join()
            process_list.append(p0)
    return process_list


def finished_seq(self, model_outputs, postprocess_params):
    with self.device_placement():
        if self.framework == "tf":
            pass
        elif self.framework == "pt":
            inference_context = self.get_inference_context()
            with inference_context():
                model_outputs = self._ensure_tensor_on_device(model_outputs, device=torch.device("cpu"))
        else:
            raise ValueError(f"Framework {self.framework} is not supported")
    output = self.postprocess(model_outputs, **postprocess_params)
    print(output)


def unfinished_seq(q, input_ids, attention_mask, prompt_text, collate_fn, collate_fn_params):
    # 维度
    if len(input_ids.shape) == 1:
        input_ids = torch.unsqueeze(input_ids, 0)
        attention_mask = torch.unsqueeze(attention_mask, 0)
    if isinstance(prompt_text, str):
        prompt_text = [prompt_text]
    new_ids_tensor = {"input_ids": input_ids, "attention_mask": attention_mask, "prompt_text": prompt_text}

    # padding and batch
    items = []
    ids_tensor = q.get(block=True)
    if ids_tensor is not None:
        items = dict_list2list_dict(ids_tensor, items)
    items = dict_list2list_dict(new_ids_tensor, items)
    x = collate_fn(items, **collate_fn_params)
    q.put(x, block=True)


def my_forward(self, model_inputs, **forward_params):
    with self.device_placement():
        if self.framework == "tf":
            model_inputs["training"] = False
            model_outputs = self._forward(model_inputs, **forward_params)
        elif self.framework == "pt":
            inference_context = self.get_inference_context()
            with inference_context():
                model_inputs = self._ensure_tensor_on_device(model_inputs, device=self.device)
                model_outputs = self._forward(model_inputs, **forward_params)
                # model_outputs = self._ensure_tensor_on_device(model_outputs, device=torch.device("cpu"))
        else:
            raise ValueError(f"Framework {self.framework} is not supported")
    return model_outputs
