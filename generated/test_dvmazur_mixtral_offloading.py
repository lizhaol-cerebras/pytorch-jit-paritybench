
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


from functools import cache


import typing as tp


import torch


from torch import nn


import copy


import functools


from typing import Dict


from typing import Any


from torch.nn import functional as F


from typing import Optional


from typing import Iterator


from typing import Tuple


from typing import List


from collections import deque


from collections import defaultdict


from collections import OrderedDict


def triton_matmul2_transpose(groupsize: 'int', a: 'torch.FloatTensor', qweight: 'torch.IntTensor', scales: 'torch.FloatTensor', zeros: 'torch.FloatTensor', bias: 'Optional[torch.FloatTensor]'=None) ->torch.FloatTensor:
    """
    Compute the matrix multiplication C = A x B + bias.
    Where B is quantized using GPTQ and groupsize = -1 into 4-bit values.

    A is of shape (M, K) float16
    qweight is of shape (N // 4, K) int32
    scales is of shape (G, K) float16
    zeros is of shape (G, K) float16
    bias is of shape (1, N) float16

    groupsize is the number of infeatures in each group.
    G = N // groupsize
    
    C = A @ qweight.T
    Returns C of shape (..., N) float16
    """
    assert a.shape[-1] == qweight.shape[1]
    assert a.is_contiguous(), 'A must be contiguous'
    assert scales.shape[1] == zeros.shape[1]
    assert scales.shape[1] == qweight.shape[1]
    x = a.view(-1, a.shape[-1])
    M, K = x.shape
    N = qweight.shape[0] * 4
    c = torch.empty((M, N), device='cuda', dtype=torch.float16)
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),)
    matmul2_kernel_transpose[grid](x, qweight, c, scales, zeros, M, N, K, x.stride(0), x.stride(1), qweight.stride(0), qweight.stride(1), c.stride(0), c.stride(1), scales.stride(0), scales.stride(1), zeros.stride(0), zeros.stride(1), groupsize, groupsize == N)
    c = c.view(a.shape[:-1] + (N,))
    if bias is not None:
        c = c + bias
    return c


def triton_matmul3_transpose(groupsize: 'int', a: 'torch.FloatTensor', qweight: 'torch.IntTensor', scales: 'torch.FloatTensor', zeros: 'torch.FloatTensor', N: 'int', bias: 'Optional[torch.FloatTensor]'=None) ->torch.FloatTensor:
    """
    Compute the matrix multiplication C = A x B + bias.
    Where B is quantized using GPTQ and groupsize = -1 into 4-bit values.

    A is of shape (M, K) float16
    qweight is of shape (ceil(N / 10), K) int32
    scales is of shape (G, K) float16
    zeros is of shape (G, K) float16
    bias is of shape (1, N) float16

    groupsize is the number of infeatures in each group.
    G = N // groupsize
    
    C = A @ qweight.T
    Returns C of shape (..., N) float16
    """
    assert a.shape[-1] == qweight.shape[1]
    assert a.is_contiguous(), 'A must be contiguous'
    assert scales.shape[1] == zeros.shape[1]
    assert scales.shape[1] == qweight.shape[1]
    x = a.view(-1, a.shape[-1])
    M, K = x.shape
    assert 0 <= qweight.shape[0] * 10 - N < 10
    c = torch.empty((M, N), device='cuda', dtype=torch.float16)
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),)
    matmul3_kernel_transpose[grid](x, qweight, c, scales, zeros, M, N, K, x.stride(0), x.stride(1), qweight.stride(0), qweight.stride(1), c.stride(0), c.stride(1), scales.stride(0), scales.stride(1), zeros.stride(0), zeros.stride(1), groupsize, groupsize == N)
    c = c.view(a.shape[:-1] + (N,))
    if bias is not None:
        c = c + bias
    return c


def triton_matmul4_transpose(groupsize: 'int', a: 'torch.FloatTensor', qweight: 'torch.IntTensor', scales: 'torch.FloatTensor', zeros: 'torch.FloatTensor', bias: 'Optional[torch.FloatTensor]'=None) ->torch.FloatTensor:
    """
    Compute the matrix multiplication C = A x B + bias.
    Where B is quantized using GPTQ and groupsize = -1 into 4-bit values.

    A is of shape (M, K) float16
    qweight is of shape (N//2, K) int32
    scales is of shape (G, K) float16
    zeros is of shape (G, K) float16
    bias is of shape (1, N) float16

    groupsize is the number of infeatures in each group.
    G = N // groupsize
    
    C = A @ qweight.T
    Returns C of shape (..., N) float16
    """
    assert a.shape[-1] == qweight.shape[1]
    assert a.is_contiguous(), 'A must be contiguous'
    assert scales.shape[1] == zeros.shape[1]
    assert scales.shape[1] == qweight.shape[1]
    x = a.view(-1, a.shape[-1])
    M, K = x.shape
    N = qweight.shape[0] * 2
    c = torch.empty((M, N), device='cuda', dtype=torch.float16)
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),)
    matmul4_kernel_transpose[grid](x, qweight, c, scales, zeros, M, N, K, x.stride(0), x.stride(1), qweight.stride(0), qweight.stride(1), c.stride(0), c.stride(1), scales.stride(0), scales.stride(1), zeros.stride(0), zeros.stride(1), groupsize, groupsize == N)
    c = c.view(a.shape[:-1] + (N,))
    if bias is not None:
        c = c + bias
    return c


class MixtralBLockSparseTop2MLP_HQQ(nn.Module):

    def __init__(self, config: 'MixtralConfig', quant_config: 'Dict[str, Any]', meta1, meta2):
        super().__init__()
        self.w1 = HQQLinearTritonSavable(None, quant_config, meta1)
        self.w2 = HQQLinearTritonSavable(None, quant_config, meta2)
        self.w3 = HQQLinearTritonSavable(None, quant_config, meta1)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states):
        current_hidden_states = self.act_fn(self.w1(hidden_states)) * self.w3(hidden_states)
        current_hidden_states = self.w2(current_hidden_states)
        return current_hidden_states


class SparseMoeWrapper(nn.Module):

    def __init__(self, config, layer_id, gate, expert_cache):
        super().__init__()
        self.hidden_dim = config.hidden_size
        self.ffn_dim = config.intermediate_size
        self.num_experts = config.num_local_experts
        self.top_k = config.num_experts_per_tok
        self.layer_id = layer_id
        self.gate = gate
        self.experts = expert_cache

    def forward(self, hidden_states: 'torch.Tensor') ->torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        router_logits = self.gate(hidden_states)
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights
        final_hidden_states = torch.zeros((batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device)
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)
        active_experts = selected_experts.flatten().unique().tolist()
        for (_layer_index, expert_idx), expert_layer in self.experts.load_experts(*((self.layer_id, expert_idx) for expert_idx in active_experts), unordered=True):
            idx, top_x = torch.where(expert_mask[expert_idx])
            assert top_x.shape[0] > 0
            top_x_list = top_x.tolist()
            idx_list = idx.tolist()
            current_state = hidden_states[None, top_x_list].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x_list, idx_list, None]
            final_hidden_states.index_add_(0, top_x, current_hidden_states)
        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states, router_logits


def nested_flatten(t):
    """
    Turn nested list/tuple/dict into a flat iterator.
    """
    if isinstance(t, (list, tuple)):
        for x in t:
            yield from nested_flatten(x)
    elif isinstance(t, dict):
        for k, v in sorted(t.items()):
            yield from nested_flatten(v)
    else:
        yield t


def is_namedtuple(x):
    """Checks if x is a namedtuple instance. Taken from https://stackoverflow.com/a/2166841 ."""
    t = type(x)
    b = t.__bases__
    if len(b) != 1 or b[0] != tuple:
        return False
    f = getattr(t, '_fields', None)
    if not isinstance(f, tuple):
        return False
    return all(type(n) == str for n in f)


def _nested_pack(flat_iter, structure):
    if is_namedtuple(structure):
        return type(structure)(*[_nested_pack(flat_iter, x) for x in structure])
    elif isinstance(structure, (list, tuple)):
        return type(structure)(_nested_pack(flat_iter, x) for x in structure)
    elif isinstance(structure, dict):
        return {k: _nested_pack(flat_iter, v) for k, v in sorted(structure.items())}
    else:
        return next(flat_iter)


def nested_pack(flat, structure):
    """
    Restore nested structure from flattened state
    :param flat: result of nested_flatten
    :param structure: used as example when recovering structure
    :returns: nested structure like :structure: filled with elements of :flat:
    """
    return _nested_pack(iter(flat), structure)


class MixtralExpertWrapper(nn.Module):

    def __init__(self, expert_module: 'tp.Any', device: 'torch.device'):
        super().__init__()
        expert_module, self.storage = self.replace_layer_storage(expert_module, device)
        self.expert_module = lambda *args, **kwargs: expert_module(*args, **kwargs)
        self._register_state_dict_hook(self._add_storage_to_state_dict_hook)
        self._register_load_state_dict_pre_hook(self._load_storage_from_state_dict_hook)

    @staticmethod
    def _add_storage_to_state_dict_hook(self, state_dict, prefix, local_metadata):
        state_dict[prefix + 'storage'] = torch.as_tensor(self.storage, dtype=torch.uint8)
        return state_dict

    def _load_storage_from_state_dict_hook(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        self.storage.copy_(state_dict[prefix + 'storage'].storage().untyped())
        del state_dict[prefix + 'storage']

    def forward(self, *args, **kwargs):
        return self.expert_module(*args, **kwargs)

    @staticmethod
    def replace_layer_storage(layer: 'tp.Any', device: 'torch.device'):
        state_dict = {f'w{i}': {'W_q': getattr(layer, f'w{i}').W_q, 'meta': getattr(layer, f'w{i}').meta, 'bias': getattr(layer, f'w{i}').bias} for i in range(1, 4)}
        storage_size = 0
        offsets = [0]
        for x in nested_flatten(state_dict):
            if not isinstance(x, torch.Tensor):
                continue
            storage_size += x.nbytes
            offsets.append(storage_size)
        storage = torch.UntypedStorage(storage_size, device=device)
        i = 0
        new_flattened_states = list()
        for x in nested_flatten(state_dict):
            if not isinstance(x, torch.Tensor):
                new_flattened_states.append(x)
                continue
            start = offsets[i]
            end = offsets[i + 1]
            a_view = torch.as_tensor(storage[start:end], dtype=x.dtype, device=device).view(x.shape)
            a_view[...] = x
            assert a_view.data_ptr() == storage.data_ptr() + start
            i += 1
            new_flattened_states.append(a_view)
        state_dict = nested_pack(new_flattened_states, state_dict)
        for layer_id, states in state_dict.items():
            patched = getattr(layer, layer_id)
            patched.W_q = states['W_q']
            patched.meta = states['meta']
            patched.bias = states['bias']
            setattr(layer, layer_id, patched)
        return layer, storage

