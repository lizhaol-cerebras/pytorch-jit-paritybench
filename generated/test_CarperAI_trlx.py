
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, matplotlib, numbers, numpy, pandas, queue, random, re, scipy, sklearn, string, tensorflow, time, torch, torchvision, types, typing, uuid, warnings
import operator as op
from dataclasses import dataclass
import numpy as np
from torch import Tensor
__version__ = '1.0.0'
xrange = range
wraps = functools.wraps


import math


from itertools import islice


import numpy as np


import torch


from torch import nn


from string import Template


from math import floor


from typing import List


from typing import Dict


from typing import Callable


from typing import Optional


from typing import Tuple


import random


from torch.utils.data import Dataset


import pandas as pd


from torch.utils.data import DataLoader


import copy


from functools import lru_cache


from typing import Iterable


import inspect


from typing import Any


from typing import Union


import torch.nn as nn


from copy import deepcopy


from functools import reduce


import torch.nn.functional as F


from functools import partial


from math import sqrt


from typing import Mapping


import torch.distributed


from math import ceil


from typing import Sequence


import logging


import re


from abc import abstractmethod


from abc import abstractstaticmethod


from torch.nn.utils.rnn import pad_sequence


import time


from time import time


from typing import cast


import uuid


import itertools


from collections import defaultdict


from logging import getLogger


from typing import Iterator


from torch.utils.data import DistributedSampler


from enum import Enum


from itertools import repeat


from numbers import Number


from torch.optim.lr_scheduler import CosineAnnealingLR


from torch.optim.lr_scheduler import LinearLR


from logging import CRITICAL


from logging import DEBUG


from logging import ERROR


from logging import FATAL


from logging import INFO


from logging import NOTSET


from logging import WARN


from logging import WARNING


import functools


from typing import MutableMapping


import torch.distributed as dist


class RewardModel(nn.Module):

    def __init__(self, checkpoint_path, eos_token_id):
        super().__init__()
        model = AutoModelForCausalLM.from_pretrained(checkpoint_path)
        self.transformer = model.transformer
        self.v_head = nn.Linear(model.config.n_embd, 1, bias=False)
        self.eos_token_id = eos_token_id

    def forward(self, input_ids):
        states = self.transformer(input_ids)[0]
        rewards = self.v_head(states).squeeze(-1)
        ends = torch.argmax((input_ids == self.eos_token_id).float(), dim=1).view(-1, 1)
        returns = torch.gather(rewards, 1, ends).squeeze(-1)
        return returns


class GPTRewardModel(nn.Module):

    def __init__(self, model_path):
        super().__init__()
        model = AutoModelForCausalLM.from_pretrained(model_path)
        self.config = model.config
        self.config.n_embd = self.config.hidden_size if hasattr(self.config, 'hidden_size') else self.config.n_embd
        self.transformer = model.transformer
        self.v_head = nn.Linear(self.config.n_embd, 1, bias=False)
        self.tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-j-6B')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.PAD_ID = self.tokenizer(self.tokenizer.pad_token)['input_ids'][0]

    def forward(self, input_ids=None, past_key_values=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None, mc_token_ids=None, labels=None, return_dict=False, output_attentions=False, output_hidden_states=False):
        loss = None
        transformer_outputs = self.transformer(input_ids, past_key_values=past_key_values, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds)
        hidden_states = transformer_outputs[0]
        rewards = self.v_head(hidden_states).squeeze(-1)
        chosen_end_scores = []
        rejected_end_scores = []
        assert len(input_ids.shape) == 2
        bs = input_ids.shape[0] // 2
        chosen = input_ids[:bs]
        rejected = input_ids[bs:]
        chosen_rewards = rewards[:bs]
        rejected_rewards = rewards[bs:]
        loss = 0
        inference = False
        for i in range(bs):
            if torch.all(torch.eq(chosen[i], rejected[i])).item():
                c_inds = (chosen[i] == self.PAD_ID).nonzero()
                c_ind = c_inds[0].item() if len(c_inds) > 0 else chosen.shape[1]
                chosen_end_scores.append(chosen_rewards[i, c_ind - 1])
                inference = True
                continue
            c_inds = (chosen[i] == self.PAD_ID).nonzero()
            c_ind = c_inds[0].item() if len(c_inds) > 0 else chosen.shape[1]
            r_inds = (rejected[i] == self.PAD_ID).nonzero()
            r_ind = r_inds[0].item() if len(r_inds) > 0 else rejected.shape[1]
            end_ind = max(c_ind, r_ind)
            divergence_ind = (chosen[i] != rejected[i]).nonzero()[0]
            assert divergence_ind > 0
            c_truncated_reward = chosen_rewards[i][divergence_ind:end_ind]
            r_truncated_reward = rejected_rewards[i][divergence_ind:end_ind]
            chosen_end_scores.append(c_truncated_reward[-1])
            rejected_end_scores.append(r_truncated_reward[-1])
            loss += -torch.log(torch.sigmoid(c_truncated_reward - r_truncated_reward)).mean()
        loss = loss / bs
        if not inference:
            chosen_end_scores = torch.stack(chosen_end_scores)
            rejected_end_scores = torch.stack(rejected_end_scores)
        if inference:
            chosen_end_scores = torch.stack(chosen_end_scores)
            return {'chosen_end_scores': chosen_end_scores}
        return {'loss': loss, 'chosen_end_scores': chosen_end_scores, 'rejected_end_scores': rejected_end_scores}


def is_peft_available():
    return importlib.util.find_spec('peft') is not None


def make_head(n_embd: 'int', out: 'int', dtype: 'type'=torch.float32) ->nn.Sequential:
    """Returns a generic sequential MLP head."""
    return nn.Sequential(nn.Linear(n_embd, n_embd * 2, dtype=dtype), nn.ReLU(), nn.Linear(n_embd * 2, out, dtype=dtype))


def rgetattr(obj, attr: 'str', *args) ->object:
    """A chain-able attribute version of getattr. For example, to get the
    attribute `foo.bar.baz` from `obj`, you can use:
        `rgetattr(obj, "foo.bar.baz")`
    Reference: https://stackoverflow.com/a/31174427
    """

    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))


def rhasattr(obj, attr):
    """A chain-able attribute version of hasattr. For example, to check if
    `obj` has the attribute `foo.bar.baz`, you can use:
        `rhasattr(obj, "foo.bar.baz")`
    Reference: https://stackoverflow.com/a/67303315
    """
    _nested_attrs = attr.split('.')
    _curr_obj = obj
    for _a in _nested_attrs[:-1]:
        if hasattr(_curr_obj, _a):
            _curr_obj = getattr(_curr_obj, _a)
        else:
            return False
    return hasattr(_curr_obj, _nested_attrs[-1])


def findattr(obj, attrs: 'Tuple[str]') ->Union[object, None]:
    for attr in attrs:
        if rhasattr(obj, attr):
            return rgetattr(obj, attr)
    raise ValueError(f'Could not find an attribute from `{attrs}` in `{obj}`')


def hf_get_hidden_size(config: 'transformers.PretrainedConfig') ->int:
    """Returns the hidden layer dimensionality of the model architecture specified
    by the HuggingFace transformers config.
    NOTE: Different model configurations have different hidden size attribute names.
        - hidden_size: (OPTConfig, BloomConfig)
        - n_embd: (GPT2Config, GPTJConfig)
        - d_model: (PegasusConfig, XLNetConfig)
    """
    hidden_size_attrs = 'hidden_size', 'n_embd', 'd_model'
    return findattr(config, hidden_size_attrs)


def hf_get_lm_head(model: 'nn.Module') ->nn.Module:
    """Returns the language modeling (lm) head of the specified HuggingFace
    transformers model.
    NOTE: Different model configurations have different `lm_head` attribute names.
        - lm_head: (GPT2LMHeadModel, BloomForCausalLM)
        - embed_out: (GPTNeoXForCausalLM)
    """
    return model.get_output_embeddings()


def topk_mask(xs: 'torch.FloatTensor', k: 'int'):
    if k > xs.shape[-1]:
        return xs
    mintop = torch.topk(xs, k)[0][:, -1].unsqueeze(-1)
    return torch.where(xs < mintop, -np.inf * torch.ones_like(xs, dtype=xs.dtype), xs)


class ParallelLinear(nn.Module):
    """Linear layer parallelized over the longer dimension."""

    def __init__(self, in_size: 'int', out_size: 'int', init_method=None, use_cpu_initialization=False, bias=True, sequence_parallel=False, gradient_accumulation_fusion=False, gather_output=True, input_is_parallel=False, dtype=torch.bfloat16):
        super().__init__()
        if init_method is None:
            init_method = partial(nn.init.uniform_, a=-sqrt(1.0 / in_size), b=sqrt(1.0 / in_size))
        no_async_tensor_model_parallel_allreduce = parallel_state.get_tensor_model_parallel_world_size() == 1 or sequence_parallel
        with tensor_parallel.random.get_cuda_rng_tracker().fork():
            if in_size < out_size:
                self.layer = tensor_parallel.ColumnParallelLinear(in_size, out_size, gather_output=gather_output, init_method=init_method, skip_bias_add=False, use_cpu_initialization=use_cpu_initialization, bias=bias, sequence_parallel_enabled=sequence_parallel, no_async_tensor_model_parallel_allreduce=no_async_tensor_model_parallel_allreduce, gradient_accumulation_fusion=gradient_accumulation_fusion, params_dtype=dtype)
            else:
                self.layer = tensor_parallel.RowParallelLinear(in_size, out_size, input_is_parallel=input_is_parallel, init_method=init_method, skip_bias_add=False, use_cpu_initialization=use_cpu_initialization, bias=bias, sequence_parallel_enabled=sequence_parallel, gradient_accumulation_fusion=gradient_accumulation_fusion, params_dtype=dtype)
            self.layer.bias.data.uniform_(-sqrt(1.0 / out_size), sqrt(1.0 / out_size))

    def forward(self, x):
        output, bias = self.layer(x)
        if bias is not None:
            return output + bias
        return output


def make_parallel_head(n_embd: 'int', out: 'int', sequence_parallel=False, dtype=torch.bfloat16) ->nn.Sequential:
    """Returns a generic sequential model parallel MLP head."""
    parallel_intermediate = out < n_embd * 2
    return nn.Sequential(ParallelLinear(n_embd, n_embd * 2, sequence_parallel=sequence_parallel, gather_output=not parallel_intermediate, dtype=dtype), nn.ReLU(), ParallelLinear(n_embd * 2, out, sequence_parallel=sequence_parallel, input_is_parallel=parallel_intermediate, dtype=dtype))


def tree_map(f, tree: 'Any') ->Any:
    """
    Apply function f to all leaves in tree
    """
    if is_dataclass(tree):
        return tree.__class__(**{k: tree_map(f, v) for k, v in tree.__dict__.items()})
    elif isinstance(tree, dict):
        return {k: tree_map(f, v) for k, v in tree.items()}
    elif isinstance(tree, (list, tuple)):
        return tree.__class__(tree_map(f, v) for v in tree)
    else:
        return f(tree)


class ParallelILQLHeads(nn.Module):

    def __init__(self, config: 'ILQLConfig', hidden_size: 'int', vocab_size: 'int', sequence_parallel=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.v_head = make_parallel_head(hidden_size, 1, sequence_parallel=sequence_parallel)
        self.config = config
        n_qs = 2 if self.config.two_qs else 1
        self.q_heads = nn.ModuleList(make_parallel_head(self.hidden_size, self.vocab_size) for _ in range(n_qs))
        self.target_q_heads = nn.ModuleList(deepcopy(q_head) for q_head in self.q_heads)
        self.target_q_heads.requires_grad_(False)

    def forward(self, hidden_states):
        qs = tuple(q_head(hidden_states) for q_head in self.q_heads)
        target_qs = tuple(q_head(hidden_states) for q_head in self.target_q_heads)
        vs = self.v_head(hidden_states)
        qs, target_qs, vs = tree_map(lambda t: rearrange(t, 'T N ... -> N T ...'), (qs, target_qs, vs))
        return qs, target_qs, vs

    def _sync_target_q_heads(self, alpha: 'float'):
        for target_q_head, q_head in zip(self.target_q_heads, self.q_heads):
            for target_param, copy_param in zip(target_q_head.parameters(), q_head.parameters()):
                target_param.data.copy_(alpha * copy_param.data + (1.0 - alpha) * target_param.data)

    def sync_target_q_heads(self):
        self._sync_target_q_heads(self.config.alpha)


class ValueHead(nn.Module):

    def __init__(self, hidden_size: 'int', sequence_parallel=False, dtype=torch.bfloat16):
        super().__init__()
        self.hidden_size = hidden_size
        self.v_head = make_parallel_head(hidden_size, 1, sequence_parallel=sequence_parallel, dtype=dtype)
        self.sequence_parallel = sequence_parallel

    def forward(self, x):
        vs = self.v_head(x)
        if self.sequence_parallel:
            vs = gather_from_sequence_parallel_region(vs, to_model_parallel=False)
        return rearrange(vs, 'T N 1 -> N T')


def hf_get_decoder_blocks(model: 'nn.Module') ->Tuple[nn.Module]:
    """Returns the decoder hidden layers of the specified model.
    NOTE: Different model configurations have different hidden layer attribute names.
        - transformer.h: (BloomForCausalLM, GPT2LMHeadModel, GPTJForCausalLM)
        - model.decoder.layers: (OPTForCausalLM)
        - gpt_neox.layers: (GPTNeoXForCausalLM)
        - decoder.block: (T5ForConditionalGeneration)
    """
    hidden_layers_attrs = 'h', 'layers', 'model.layers', 'decoder.layers', 'transformer.h', 'transformer.blocks', 'model.decoder.layers', 'gpt_neox.layers', 'decoder.block'
    return findattr(model, hidden_layers_attrs)


def hf_get_decoder_final_norm(model: 'nn.Module') ->float:
    """Returns the final (layer) norm of the specified decoder.
    NOTE: Different model configurations have different final norm attribute names.
        - transformer.ln_f: (GPT2LMHeadModel, GPTJForCausalLM)
        - model.decoder.final_layer_norm: (OPTForCausalLM)
        - gpt_neox.layers.final_layer_norm: (GPTNeoXForCausalLM)
    """
    norm_attrs = 'transformer.ln_f', 'model.decoder.final_layer_norm', 'model.norm', 'decoder.final_layer_norm', 'gpt_neox.final_layer_norm'
    return findattr(model, norm_attrs)


def hf_get_num_hidden_layers(config: 'transformers.PretrainedConfig') ->int:
    """Returns the number of hidden layers in the model architecture specified
    by the HuggingFace transformers config.
    NOTE: Different model configurations have different number-of-layers attribute
    names.
        - num_hidden_layers: (GPTNeoXConfig, OPTConfig)
        - n_layer: (GPT2Config, GPTJConfig, BloomConfig)
    """
    num_hidden_layers_attrs = 'num_hidden_layers', 'n_layer'
    return findattr(config, num_hidden_layers_attrs)


def hf_get_branch_class(config: 'transformers.PretrainedConfig') ->'ModelBranch':
    """Returns the model branch class for the given config."""
    gpt_branch_supported_archs = ['GPTJForCausalLM', 'GPT2LMHeadModel', 'GPTNeoForCausalLM', 'GPTNeoXForCausalLM']
    opt_branch_supported_archs = ['OPTForCausalLM']
    bloom_branch_supported_archs = ['BloomModel', 'BloomForCausalLM']
    llama_branch_supported_archs = ['LlamaModel', 'LlamaForCausalLM']
    bigcode_branch_supported_archs = ['GPTBigCodeModel', 'GPTBigCodeForCausalLM']
    arch = config.architectures[0]
    if arch in gpt_branch_supported_archs:
        return GPTModelBranch
    elif arch in opt_branch_supported_archs:
        return OPTModelBranch
    elif arch in bloom_branch_supported_archs:
        return BloomModelBranch
    elif arch in llama_branch_supported_archs:
        return LlamaModelBranch
    elif arch in bigcode_branch_supported_archs:
        return GPTBigCodeModelBranch
    else:
        all_supported_archs = sum([gpt_branch_supported_archs, opt_branch_supported_archs, bloom_branch_supported_archs, llama_branch_supported_archs, bigcode_branch_supported_archs], [])
        raise ValueError(f'Unsupported architecture: `{arch}`. The following architectures are available for model branching:\n{all_supported_archs}')


def make_value_branch(base_model, num_value_layers_unfrozen):
    value_head = make_head(hf_get_hidden_size(base_model.config), 1)
    if num_value_layers_unfrozen == 0:
        return value_head
    config = base_model.config
    branch_class = hf_get_branch_class(config)
    value_branch = branch_class(base_model, num_layers_unfrozen=num_value_layers_unfrozen, frozen=False)
    value_branch.lm_head = value_head
    return value_branch


def hf_get_decoder(model: 'nn.Module') ->nn.Module:
    """Returns the causal decoder backbone of the specified HuggingFace transformers
    model.
    NOTE: Different model configurations have different causal decoder attribute
    names.
        - transformer: (GPT2LMHeadModel, GPTJConfig)
        - model.decoder: (OPTConfig, BloomConfig)
        - gpt_neox: (GPTNeoXConfig)
    """
    decoder_attrs = 'transformer', 'model.decoder', 'gpt_neox', 'decoder'
    return findattr(model, decoder_attrs)

