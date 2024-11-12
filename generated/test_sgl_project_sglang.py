
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


import itertools


import logging


import time


from typing import Tuple


import numpy as np


import pandas as pd


import torch


import torch.distributed as dist


from collections import OrderedDict


from collections import defaultdict


from typing import List


from typing import Optional


from typing import Union


import torch.nn as nn


import torch.nn.functional as F


from abc import ABC


from abc import abstractmethod


from torch import nn


from typing import TYPE_CHECKING


from enum import Enum


from enum import auto


import functools


from typing import Any


from typing import Dict


from torch.nn import Module


from torch.nn import functional as F


from torch.nn.parameter import Parameter


from torch.nn.parameter import UninitializedParameter


from enum import IntEnum


from typing import Type


import inspect


from typing import Set


from typing import Sequence


import re


import warnings


from collections import deque


from types import SimpleNamespace


from queue import Queue


from typing import Callable


import math


from functools import lru_cache


from typing import Iterable


from torch.nn import LayerNorm


import torch.utils.checkpoint


from functools import partial


from typing import TypedDict


import types


import abc


import typing


import random


from torch.profiler import ProfilerActivity


from torch.profiler import profile


from torch.profiler import record_function


import enum


def set_weight_attrs(weight: 'torch.Tensor', weight_attrs: 'Optional[Dict[str, Any]]'):
    """Set attributes on a weight tensor.

    This method is used to set attributes on a weight tensor. This method
    will not overwrite existing attributes.

    Args:
        weight: The weight tensor.
        weight_attrs: A dictionary of attributes to set on the weight tensor.
    """
    if weight_attrs is None:
        return
    for key, value in weight_attrs.items():
        assert not hasattr(weight, key), f'Overwriting existing tensor attribute: {key}'
        setattr(weight, key, value)


class ScaledActivation(nn.Module):
    """An activation function with post-scale parameters.

    This is used for some quantization methods like AWQ.
    """

    def __init__(self, act_module: 'nn.Module', intermediate_size: 'int', input_is_parallel: 'bool'=True, params_dtype: 'Optional[torch.dtype]'=None):
        super().__init__()
        self.act = act_module
        self.input_is_parallel = input_is_parallel
        if input_is_parallel:
            tp_size = get_tensor_model_parallel_world_size()
            intermediate_size_per_partition = divide(intermediate_size, tp_size)
        else:
            intermediate_size_per_partition = intermediate_size
        if params_dtype is None:
            params_dtype = torch.get_default_dtype()
        self.scales = nn.Parameter(torch.empty(intermediate_size_per_partition, dtype=params_dtype))
        set_weight_attrs(self.scales, {'weight_loader': self.weight_loader})

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        return self.act(x) / self.scales

    def weight_loader(self, param: 'nn.Parameter', loaded_weight: 'torch.Tensor'):
        param_data = param.data
        if self.input_is_parallel:
            tp_rank = get_tensor_model_parallel_rank()
            shard_size = param_data.shape[0]
            start_idx = tp_rank * shard_size
            loaded_weight = loaded_weight.narrow(0, start_idx, shard_size)
        assert param_data.shape == loaded_weight.shape
        param_data.copy_(loaded_weight)


class QuantizeMethodBase(ABC):
    """Base class for different quantized methods."""

    @abstractmethod
    def create_weights(self, layer: 'torch.nn.Module', *weight_args, **extra_weight_attrs):
        """Create weights for a layer.

        The weights will be set as attributes of the layer."""
        raise NotImplementedError

    @abstractmethod
    def apply(self, layer: 'torch.nn.Module', *args, **kwargs) ->torch.Tensor:
        """Apply the weights in layer to the input tensor.

        Expects create_weights to have been called before on the layer."""
        raise NotImplementedError

    def process_weights_after_loading(self, layer: 'nn.Module') ->None:
        """Process the weight after loading.

        This can be used for example, to transpose weights for computation.
        """
        return


class FusedMoEMethodBase(QuantizeMethodBase):

    @abstractmethod
    def create_weights(self, layer: 'torch.nn.Module', num_experts: 'int', hidden_size: 'int', intermediate_size: 'int', params_dtype: 'torch.dtype', **extra_weight_attrs):
        raise NotImplementedError

    @abstractmethod
    def apply(self, layer: 'torch.nn.Module', x: 'torch.Tensor', router_logits: 'torch.Tensor', top_k: 'int', renormalize: 'bool'=True, use_grouped_topk: 'bool'=False, num_expert_group: 'Optional[int]'=None, topk_group: 'Optional[int]'=None) ->torch.Tensor:
        raise NotImplementedError


def invoke_fused_moe_kernel(A: 'torch.Tensor', B: 'torch.Tensor', C: 'torch.Tensor', A_scale: 'Optional[torch.Tensor]', B_scale: 'Optional[torch.Tensor]', topk_weights: 'torch.Tensor', topk_ids: 'torch.Tensor', sorted_token_ids: 'torch.Tensor', expert_ids: 'torch.Tensor', num_tokens_post_padded: 'torch.Tensor', mul_routed_weight: 'bool', top_k: 'int', config: 'Dict[str, Any]', compute_type: 'tl.dtype', use_fp8: 'bool') ->None:
    assert topk_weights.stride(1) == 1
    assert sorted_token_ids.stride(0) == 1
    if not use_fp8:
        assert A_scale is None
        assert B_scale is None
    else:
        A, A_scale = ops.scaled_fp8_quant(A, A_scale)
        assert B_scale is not None
    grid = lambda META: (triton.cdiv(sorted_token_ids.shape[0], META['BLOCK_SIZE_M']) * triton.cdiv(B.shape[1], META['BLOCK_SIZE_N']),)
    fused_moe_kernel[grid](A, B, C, A_scale, B_scale, topk_weights, sorted_token_ids, expert_ids, num_tokens_post_padded, B.shape[1], B.shape[2] - padding_size, sorted_token_ids.shape[0], topk_ids.numel(), A.stride(0), A.stride(1), B.stride(0), B.stride(2), B.stride(1), C.stride(1), C.stride(2), MUL_ROUTED_WEIGHT=mul_routed_weight, top_k=top_k, compute_type=compute_type, use_fp8=use_fp8, **config)


def moe_align_block_size(topk_ids: 'torch.Tensor', block_size: 'int', num_experts: 'int') ->Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Aligns the token distribution across experts to be compatible with block
    size for matrix multiplication.

    Parameters:
    - topk_ids: A tensor of shape [total_tokens, top_k] representing the
        top-k expert indices for each token.
    - block_size: The block size used in block matrix multiplication.
    - num_experts: The total number of experts.

    Returns:
    - sorted_token_ids: A tensor containing the sorted token indices according
        to their allocated expert.
    - expert_ids: A tensor indicating the assigned expert index for each block.
    - num_tokens_post_padded: The total number of tokens after padding,
        ensuring divisibility by block_size.

    This function pads the number of tokens that each expert needs to process
    so that it is divisible by block_size.
    Padding ensures that during block matrix multiplication, the dimensions
    align correctly.

    Example:
    Given topk_ids = [[2, 3, 4], [1, 2, 4], [1, 3, 4], [1, 2, 3]],
    block_size = 4, and num_experts = 4:
    - We initially have 12 tokens (after repeating 'top_k' times) and 4 experts,
        with each expert needing to process 3 tokens.
    - As block_size is 4, we pad 1 token for each expert.
    - First, flatten topk_ids to [2, 3, 4, 1, 2, 4, 1, 3, 4, 1, 2, 3].
    - Then append padding tokens [12, 12, 12, 12] for each block.
    - After sorting by expert index, we obtain token_ids
        [3, 6, 9, 12, 0, 4, 10, 12, 1, 7, 11, 12, 2, 5, 8, 12].
        Tokens 12 are non-existent (padding) and are ignored in
        the subsequent matrix multiplication.
    - The padding ensures that the total number of tokens is now divisible
        by block_size for proper block matrix operations.
    """
    max_num_tokens_padded = topk_ids.numel() + num_experts * (block_size - 1)
    sorted_ids = torch.empty((max_num_tokens_padded,), dtype=torch.int32, device=topk_ids.device)
    sorted_ids.fill_(topk_ids.numel())
    max_num_m_blocks = triton.cdiv(max_num_tokens_padded, block_size)
    expert_ids = torch.empty((max_num_m_blocks,), dtype=torch.int32, device=topk_ids.device)
    num_tokens_post_pad = torch.empty(1, dtype=torch.int32, device=topk_ids.device)
    ops.moe_align_block_size(topk_ids, num_experts, block_size, sorted_ids, expert_ids, num_tokens_post_pad)
    return sorted_ids, expert_ids, num_tokens_post_pad


def get_default_config(M: 'int', E: 'int', N: 'int', K: 'int', topk: 'int', dtype: 'Optional[str]') ->Dict[str, int]:
    if dtype == 'float8':
        config = {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 32, 'num_warps': 8, 'num_stages': 4}
        if M <= E:
            config = {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4}
    else:
        config = {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}
        if M <= E:
            config = {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1}
    return config


def get_config_file_name(E: 'int', N: 'int', dtype: 'Optional[str]') ->str:
    device_name = torch.cuda.get_device_name().replace(' ', '_')
    dtype_selector = '' if not dtype else f',dtype={dtype}'
    return f'E={E},N={N},device_name={device_name}{dtype_selector}.json'


logger = logging.getLogger(__name__)


@functools.lru_cache
def get_moe_configs(E: 'int', N: 'int', dtype: 'Optional[str]') ->Optional[Dict[int, Any]]:
    """
    Return optimized configurations for the fused MoE kernel.

    The return value will be a dictionary that maps an irregular grid of
    batch sizes to configurations of the fused_moe kernel. To evaluate the
    kernel on a given batch size bs, the closest batch size in the grid should
    be picked and the associated configuration chosen to invoke the kernel.
    """
    json_file_name = get_config_file_name(E, N, dtype)
    config_file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'configs', json_file_name)
    if os.path.exists(config_file_path):
        with open(config_file_path) as f:
            logger.info('Using configuration from %s for MoE layer.', config_file_path)
            return {int(key): val for key, val in json.load(f).items()}
    return None


def try_get_optimal_moe_config(w1_shape: 'Tuple[int, ...]', w2_shape: 'Tuple[int, ...]', top_k: 'int', dtype: 'Optional[str]', M: 'int', override_config: 'Optional[Dict[str, Any]]'=None):
    if override_config:
        config = override_config
    else:
        E, _, N = w2_shape
        configs = get_moe_configs(E, N, dtype)
        if configs:
            config = configs[min(configs.keys(), key=lambda x: abs(x - M))]
        else:
            config = get_default_config(M, E, N, w1_shape[2], top_k, dtype)
    return config


def fused_experts(hidden_states: 'torch.Tensor', w1: 'torch.Tensor', w2: 'torch.Tensor', topk_weights: 'torch.Tensor', topk_ids: 'torch.Tensor', inplace: 'bool'=False, override_config: 'Optional[Dict[str, Any]]'=None, use_fp8: 'bool'=False, w1_scale: 'Optional[torch.Tensor]'=None, w2_scale: 'Optional[torch.Tensor]'=None, a1_scale: 'Optional[torch.Tensor]'=None, a2_scale: 'Optional[torch.Tensor]'=None):
    assert hidden_states.shape[1] == w1.shape[2] - padding_size, 'Hidden size mismatch'
    assert topk_weights.shape == topk_ids.shape, 'topk shape mismatch'
    assert hidden_states.is_contiguous(), 'Hidden_states must be contiguous'
    assert w1.is_contiguous(), 'Expert weights1 must be contiguous'
    assert w2.is_contiguous(), 'Expert weights2 must be contiguous'
    assert hidden_states.dtype in [torch.float32, torch.float16, torch.bfloat16]
    num_tokens, _ = hidden_states.shape
    E, N, _ = w1.shape
    CHUNK_SIZE = envs.VLLM_FUSED_MOE_CHUNK_SIZE
    M = min(num_tokens, CHUNK_SIZE)
    get_config_func = functools.partial(try_get_optimal_moe_config, w1.shape, (w2.shape[0], w2.shape[1], w2.shape[2] - padding_size), topk_ids.shape[1], 'float8' if use_fp8 else None, override_config=override_config)
    config = get_config_func(M)
    intermediate_cache1 = torch.empty((M, topk_ids.shape[1], N), device=hidden_states.device, dtype=hidden_states.dtype)
    intermediate_cache2 = torch.empty((M * topk_ids.shape[1], N // 2), device=hidden_states.device, dtype=hidden_states.dtype)
    intermediate_cache3 = torch.empty((M, topk_ids.shape[1], w2.shape[1]), device=hidden_states.device, dtype=hidden_states.dtype)
    compute_type = tl.bfloat16 if hidden_states.dtype == torch.bfloat16 else tl.float16
    if inplace:
        out_hidden_states = hidden_states
    else:
        out_hidden_states = torch.empty_like(hidden_states)
    for chunk in range(num_tokens // CHUNK_SIZE + 1):
        begin_chunk_idx, end_chunk_idx = chunk * CHUNK_SIZE, min((chunk + 1) * CHUNK_SIZE, num_tokens)
        curr_hidden_states = hidden_states[begin_chunk_idx:end_chunk_idx]
        tokens_in_chunk, _ = curr_hidden_states.shape
        if tokens_in_chunk == 0:
            break
        if tokens_in_chunk < CHUNK_SIZE and chunk > 0:
            intermediate_cache1 = intermediate_cache1[:tokens_in_chunk]
            intermediate_cache2 = intermediate_cache2[:tokens_in_chunk]
            intermediate_cache3 = intermediate_cache3[:tokens_in_chunk]
            config = get_config_func(tokens_in_chunk)
        curr_topk_ids = topk_ids[begin_chunk_idx:end_chunk_idx]
        curr_topk_weights = topk_weights[begin_chunk_idx:end_chunk_idx]
        sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(curr_topk_ids, config['BLOCK_SIZE_M'], E)
        invoke_fused_moe_kernel(curr_hidden_states, w1, intermediate_cache1, a1_scale, w1_scale, curr_topk_weights, curr_topk_ids, sorted_token_ids, expert_ids, num_tokens_post_padded, False, topk_ids.shape[1], config, compute_type=compute_type, use_fp8=use_fp8)
        ops.gelu_and_mul(intermediate_cache2, intermediate_cache1.view(-1, N))
        invoke_fused_moe_kernel(intermediate_cache2, w2, intermediate_cache3, a2_scale, w2_scale, curr_topk_weights, curr_topk_ids, sorted_token_ids, expert_ids, num_tokens_post_padded, True, 1, config, compute_type=compute_type, use_fp8=use_fp8)
        torch.sum(intermediate_cache3.view(*intermediate_cache3.shape), dim=1, out=out_hidden_states[begin_chunk_idx:end_chunk_idx])
    return out_hidden_states


def fused_topk(hidden_states: 'torch.Tensor', gating_output: 'torch.Tensor', topk: 'int', renormalize: 'bool'):
    assert hidden_states.shape[0] == gating_output.shape[0], 'Number of tokens mismatch'
    M, _ = hidden_states.shape
    topk_weights = torch.empty(M, topk, dtype=torch.float32, device=hidden_states.device)
    topk_ids = torch.empty(M, topk, dtype=torch.int32, device=hidden_states.device)
    token_expert_indicies = torch.empty(M, topk, dtype=torch.int32, device=hidden_states.device)
    ops.topk_softmax(topk_weights, topk_ids, token_expert_indicies, gating_output.float())
    del token_expert_indicies
    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    return topk_weights, topk_ids


def grouped_topk(hidden_states: 'torch.Tensor', gating_output: 'torch.Tensor', topk: 'int', renormalize: 'bool', num_expert_group: 'int'=0, topk_group: 'int'=0):
    assert hidden_states.shape[0] == gating_output.shape[0], 'Number of tokens mismatch'
    scores = torch.softmax(gating_output, dim=-1)
    num_token = scores.shape[0]
    group_scores = scores.view(num_token, num_expert_group, -1).max(dim=-1).values
    group_idx = torch.topk(group_scores, k=topk_group, dim=-1, sorted=False)[1]
    group_mask = torch.zeros_like(group_scores)
    group_mask.scatter_(1, group_idx, 1)
    score_mask = group_mask.unsqueeze(-1).expand(num_token, num_expert_group, scores.shape[-1] // num_expert_group).reshape(num_token, -1)
    tmp_scores = scores.masked_fill(~score_mask.bool(), 0.0)
    topk_weights, topk_ids = torch.topk(tmp_scores, k=topk, dim=-1, sorted=False)
    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    return topk_weights, topk_ids


def fused_moe(hidden_states: 'torch.Tensor', w1: 'torch.Tensor', w2: 'torch.Tensor', gating_output: 'torch.Tensor', topk: 'int', renormalize: 'bool', inplace: 'bool'=False, override_config: 'Optional[Dict[str, Any]]'=None, use_grouped_topk: 'bool'=False, num_expert_group: 'Optional[int]'=None, topk_group: 'Optional[int]'=None, use_fp8: 'bool'=False, w1_scale: 'Optional[torch.Tensor]'=None, w2_scale: 'Optional[torch.Tensor]'=None, a1_scale: 'Optional[torch.Tensor]'=None, a2_scale: 'Optional[torch.Tensor]'=None) ->torch.Tensor:
    """
    This function computes a Mixture of Experts (MoE) layer using two sets of
    weights, w1 and w2, and top-k gating mechanism.

    Parameters:
    - hidden_states (torch.Tensor): The input tensor to the MoE layer.
    - w1 (torch.Tensor): The first set of expert weights.
    - w2 (torch.Tensor): The second set of expert weights.
    - gating_output (torch.Tensor): The output of the gating operation
        (before softmax).
    - topk (int): The number of top-k experts to select.
    - renormalize (bool): If True, renormalize the top-k weights to sum to 1.
    - inplace (bool): If True, perform the operation in-place.
        Defaults to False.
    - override_config (Optional[Dict[str, Any]]): Optional override
        for the kernel configuration.
    - num_expert_group: Optional[int]: additional parameter for grouped_topk
    - topk_group: Optional[int]: additional parameter for grouped_topk
    - use_grouped_topk: If True, use grouped_topk instead of fused_topk
        note: Deepseekv2 model uses grouped_topk
    - use_fp8 (bool): If True, use fp8 arithmetic to compute the inner
        products for w1 and w2. Defaults to False.
    - w1_scale (Optional[torch.Tensor]): Optional scale to be used for
        w1.
    - w2_scale (Optional[torch.Tensor]): Optional scale to be used for
        w2.

    Returns:
    - torch.Tensor: The output tensor after applying the MoE layer.
    """
    assert gating_output.shape[1] == w1.shape[0], 'Number of experts mismatch'
    if use_grouped_topk:
        assert num_expert_group is not None and topk_group is not None
        topk_weights, topk_ids = grouped_topk(hidden_states, gating_output, topk, renormalize, num_expert_group, topk_group)
    else:
        topk_weights, topk_ids = fused_topk(hidden_states, gating_output, topk, renormalize)
    return fused_experts(hidden_states, w1, w2, topk_weights, topk_ids, inplace=inplace, override_config=override_config, use_fp8=use_fp8, w1_scale=w1_scale, w2_scale=w2_scale, a1_scale=a1_scale, a2_scale=a2_scale)


def is_hip() ->bool:
    """Return whether it is HIP on the AMD ROCm platform."""
    return torch.version.hip is not None


class Fp8MoEMethod(FusedMoEMethodBase):
    """MoE method for FP8.
    Supports loading FP8 checkpoints with static weight scale and
    dynamic/static activation scale.

    Also supports loading quantized FP16/BF16 model checkpoints with dynamic
    activation scaling. The weight scaling factor will be initialized after
    the model weights are loaded.

    Args:
        quant_config: The quantization config.
    """

    def __init__(self, quant_config: 'Fp8Config'):
        self.quant_config = quant_config

    def create_weights(self, layer: 'Module', num_experts: 'int', hidden_size: 'int', intermediate_size: 'int', params_dtype: 'torch.dtype', **extra_weight_attrs):
        if self.quant_config.is_checkpoint_fp8_serialized:
            params_dtype = torch.float8_e4m3fn
        w13_weight = torch.nn.Parameter(torch.empty(num_experts, 2 * intermediate_size, hidden_size, dtype=params_dtype), requires_grad=False)
        layer.register_parameter('w13_weight', w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)
        w2_weight = torch.nn.Parameter(torch.empty(num_experts, hidden_size, intermediate_size, dtype=params_dtype), requires_grad=False)
        layer.register_parameter('w2_weight', w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)
        w13_scale = torch.nn.Parameter(torch.ones(num_experts, 2, dtype=torch.float32), requires_grad=False)
        layer.register_parameter('w13_scale', w13_scale)
        w2_scale = torch.nn.Parameter(torch.ones(num_experts, dtype=torch.float32), requires_grad=False)
        layer.register_parameter('w2_scale', w2_scale)
        if self.quant_config.is_checkpoint_fp8_serialized:
            set_weight_attrs(w13_scale, extra_weight_attrs)
            set_weight_attrs(w2_scale, extra_weight_attrs)
        if self.quant_config.activation_scheme == 'static':
            if not self.quant_config.is_checkpoint_fp8_serialized:
                raise ValueError('Found static activation scheme for checkpoint that was not serialized fp8.')
            a13_scale = torch.nn.Parameter(torch.ones(num_experts, dtype=torch.float32), requires_grad=False)
            layer.register_parameter('a13_scale', a13_scale)
            set_weight_attrs(a13_scale, extra_weight_attrs)
            a2_scale = torch.nn.Parameter(torch.ones(num_experts, dtype=torch.float32), requires_grad=False)
            layer.register_parameter('a2_scale', a2_scale)
            set_weight_attrs(a2_scale, extra_weight_attrs)
        else:
            layer.a13_scale = None
            layer.a2_scale = None

    def process_weights_after_loading(self, layer: 'Module') ->None:
        if not self.quant_config.is_checkpoint_fp8_serialized:
            fp8_dtype = torch.float8_e4m3fnuz if is_hip() else torch.float8_e4m3fn
            w13_weight = torch.empty_like(layer.w13_weight.data, dtype=fp8_dtype)
            w2_weight = torch.empty_like(layer.w2_weight.data, dtype=fp8_dtype)
            layer.w13_scale = torch.nn.Parameter(torch.ones(layer.num_experts, dtype=torch.float32, device=w13_weight.device), requires_grad=False)
            for expert in range(layer.num_experts):
                w13_weight[expert, :, :], layer.w13_scale[expert] = ops.scaled_fp8_quant(layer.w13_weight.data[expert, :, :])
                w2_weight[expert, :, :], layer.w2_scale[expert] = ops.scaled_fp8_quant(layer.w2_weight.data[expert, :, :])
            layer.w13_weight = torch.nn.Parameter(w13_weight, requires_grad=False)
            layer.w2_weight = torch.nn.Parameter(w2_weight, requires_grad=False)
            if is_hip() and bool(int(os.getenv('MOE_PADDING', '0'))):
                layer.w13_weight = torch.nn.Parameter(F.pad(layer.w13_weight.data, (0, padding_size), 'constant', 0), requires_grad=False)
                torch.cuda.empty_cache()
                layer.w2_weight = torch.nn.Parameter(F.pad(layer.w2_weight.data, (0, padding_size), 'constant', 0), requires_grad=False)
                torch.cuda.empty_cache()
            return
        else:
            if self.quant_config.activation_scheme == 'static':
                if layer.a13_scale is None or layer.a2_scale is None:
                    raise ValueError('QuantConfig has static quantization, but found activation scales are None.')
                if not all_close_1d(layer.a13_scale) or not all_close_1d(layer.a2_scale):
                    print_warning_once('Found input_scales that are not equal for fp8 MoE layer. Using the maximum across experts for each layer. ')
                layer.a13_scale = torch.nn.Parameter(layer.a13_scale.max(), requires_grad=False)
                layer.a2_scale = torch.nn.Parameter(layer.a2_scale.max(), requires_grad=False)
            if is_hip():
                w13_weight, w13_scale, a13_scale = normalize_e4m3fn_to_e4m3fnuz(layer.w13_weight, layer.w13_scale, layer.a13_scale)
                w2_weight, w2_scale, a2_scale = normalize_e4m3fn_to_e4m3fnuz(layer.w2_weight, layer.w2_scale, layer.a2_scale)
                layer.w13_weight = torch.nn.Parameter(w13_weight, requires_grad=False)
                layer.w13_scale = torch.nn.Parameter(w13_scale, requires_grad=False)
                if a13_scale is not None:
                    layer.a13_scale = torch.nn.Parameter(a13_scale, requires_grad=False)
                layer.w2_weight = torch.nn.Parameter(w2_weight, requires_grad=False)
                layer.w2_scale = torch.nn.Parameter(w2_scale, requires_grad=False)
                if a2_scale is not None:
                    layer.a2_scale = torch.nn.Parameter(a2_scale, requires_grad=False)
            assert layer.w13_scale is not None
            shard_size = layer.intermediate_size_per_partition
            max_w13_scales = layer.w13_scale.max(dim=1).values
            for expert_id in range(layer.num_experts):
                start = 0
                for shard_id in range(2):
                    dq_weight = per_tensor_dequantize(layer.w13_weight[expert_id][start:start + shard_size, :], layer.w13_scale[expert_id][shard_id])
                    layer.w13_weight[expert_id][start:start + shard_size, :], _ = ops.scaled_fp8_quant(dq_weight, max_w13_scales[expert_id])
                    start += shard_size
            layer.w13_scale = torch.nn.Parameter(max_w13_scales, requires_grad=False)
            if is_hip() and bool(int(os.getenv('MOE_PADDING', '0'))):
                layer.w13_weight = torch.nn.Parameter(F.pad(layer.w13_weight.data, (0, padding_size), 'constant', 0), requires_grad=False)
                torch.cuda.empty_cache()
                layer.w2_weight = torch.nn.Parameter(F.pad(layer.w2_weight.data, (0, padding_size), 'constant', 0), requires_grad=False)
                torch.cuda.empty_cache()
            return

    def apply(self, layer: 'torch.nn.Module', x: 'torch.Tensor', router_logits: 'torch.Tensor', top_k: 'int', renormalize: 'bool'=True, use_grouped_topk: 'bool'=False, num_expert_group: 'Optional[int]'=None, topk_group: 'Optional[int]'=None) ->torch.Tensor:
        return fused_moe(x, layer.w13_weight, layer.w2_weight, router_logits, top_k, renormalize=renormalize, inplace=True, use_fp8=True, w1_scale=layer.w13_scale, w2_scale=layer.w2_scale, a1_scale=layer.a13_scale, a2_scale=layer.a2_scale, use_grouped_topk=use_grouped_topk, num_expert_group=num_expert_group, topk_group=topk_group)


class FusedMoE(torch.nn.Module):
    """FusedMoE layer for MoE models.

    This layer contains both MergedColumnParallel weights (gate_up_proj /
    w13) and RowParallelLinear weights (down_proj/ w2).

    Note: Mixtral uses w1, w2, and w3 for gate, up, and down_proj. We
    copy that naming convention here and handle any remapping in the
    load_weights function in each model implementation.

    Args:
        num_experts: Number of experts in the model
        top_k: Number of experts selected for each token
        hidden_size: Input hidden state size of the transformer
        intermediate_size: Intermediate size of the experts
        params_dtype: Data type for the parameters.
        reduce_results: Whether to all all_reduce on the output of the layer
        renomalize: Whether to renormalize the logits in the fused_moe kernel
        quant_config: Quantization configure.
    """

    def __init__(self, num_experts: 'int', top_k: 'int', hidden_size: 'int', intermediate_size: 'int', params_dtype: 'Optional[torch.dtype]'=None, reduce_results: 'bool'=False, renormalize: 'bool'=True, use_grouped_topk: 'bool'=False, num_expert_group: 'Optional[int]'=None, topk_group: 'Optional[int]'=None, quant_config: 'Optional[QuantizationConfig]'=None, tp_size: 'Optional[int]'=None, prefix: 'str'=''):
        super().__init__()
        if params_dtype is None:
            params_dtype = torch.get_default_dtype()
        self.tp_size = tp_size if tp_size is not None else get_tensor_model_parallel_world_size()
        self.top_k = top_k
        self.num_experts = num_experts
        self.intermediate_size_per_partition = intermediate_size // self.tp_size
        self.reduce_results = reduce_results
        self.renormalize = renormalize
        self.use_grouped_topk = use_grouped_topk
        if self.use_grouped_topk:
            assert num_expert_group is not None and topk_group is not None
        self.num_expert_group = num_expert_group
        self.topk_group = topk_group
        if quant_config is None:
            self.quant_method: 'Optional[QuantizeMethodBase]' = UnquantizedFusedMoEMethod()
        elif isinstance(quant_config, Fp8Config):
            self.quant_method = Fp8MoEMethod(quant_config)
        else:
            self.quant_method = quant_config.get_quant_method(self, prefix)
        assert self.quant_method is not None
        self.quant_method.create_weights(layer=self, num_experts=num_experts, hidden_size=hidden_size, intermediate_size=self.intermediate_size_per_partition, params_dtype=params_dtype, weight_loader=self.weight_loader)

    def weight_loader(self, param: 'torch.nn.Parameter', loaded_weight: 'torch.Tensor', weight_name: 'str', shard_id: 'int', expert_id: 'int', use_presharded_weights: 'bool'=False):
        param_data = param.data
        if 'input_scale' in weight_name:
            if param_data[expert_id] != 1 and (param_data[expert_id] - loaded_weight).abs() > 1e-05:
                raise ValueError(f'input_scales of w1 and w3 of a layer must be equal. But got {param_data[expert_id]} vs. {loaded_weight}')
            param_data[expert_id] = loaded_weight
        elif 'weight_scale' in weight_name:
            if shard_id == 0 or shard_id == 2:
                idx = 0 if shard_id == 0 else 1
                param_data[expert_id][idx] = loaded_weight
            else:
                param_data[expert_id] = loaded_weight
        else:
            tp_rank = get_tensor_model_parallel_rank()
            shard_size = self.intermediate_size_per_partition
            if use_presharded_weights:
                shard = slice(None)
            else:
                shard = slice(tp_rank * shard_size, (tp_rank + 1) * shard_size)
            if shard_id == 0:
                param_data[expert_id, 0:shard_size, :] = loaded_weight[shard, :]
            elif shard_id == 2:
                param_data[expert_id, shard_size:2 * shard_size, :] = loaded_weight[shard, :]
            elif shard_id == 1:
                param_data[expert_id, :, :] = loaded_weight[:, shard]
            else:
                raise ValueError(f'Shard id must be in [0,1,2] but got {shard_id}')

    def forward(self, hidden_states: 'torch.Tensor', router_logits: 'torch.Tensor'):
        assert self.quant_method is not None
        final_hidden_states = self.quant_method.apply(self, x=hidden_states, router_logits=router_logits, top_k=self.top_k, renormalize=self.renormalize, use_grouped_topk=self.use_grouped_topk, num_expert_group=self.num_expert_group, topk_group=self.topk_group)
        if self.reduce_results and self.tp_size > 1:
            final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states)
        return final_hidden_states

    @classmethod
    def make_expert_params_mapping(cls, ckpt_gate_proj_name: 'str', ckpt_down_proj_name: 'str', ckpt_up_proj_name: 'str', num_experts: 'int') ->List[Tuple[str, str, int, int]]:
        gate_up = [ckpt_gate_proj_name, ckpt_up_proj_name]
        gate_down_up = [ckpt_gate_proj_name, ckpt_down_proj_name, ckpt_up_proj_name]
        return [('experts.w13_scale' if weight_name in gate_up else 'experts.w2_scale', f'experts.{expert_id}.{weight_name}.weight_scale', expert_id, shard_id) for expert_id in range(num_experts) for shard_id, weight_name in enumerate(gate_down_up)] + [('experts.w13_weight' if weight_name in gate_up else 'experts.w2_weight', f'experts.{expert_id}.{weight_name}.weight', expert_id, shard_id) for expert_id in range(num_experts) for shard_id, weight_name in enumerate(gate_down_up)] + [('experts.a13_scale' if weight_name in gate_up else 'experts.a2_scale', f'experts.{expert_id}.{weight_name}.input_scale', expert_id, shard_id) for expert_id in range(num_experts) for shard_id, weight_name in enumerate(gate_down_up)]


class MRotaryEmbedding:
    """Rotary Embedding with Multimodal Sections."""

    @staticmethod
    def get_input_positions(input_tokens: 'torch.Tensor', image_grid_thw: 'Union[List[List[int]], torch.Tensor]', vision_start_token_id: 'int', spatial_merge_size: 'int', context_len: 'int'=0) ->Tuple[List[List[int]], int]:
        """Get mrope input positions and delta value."""
        if isinstance(image_grid_thw, torch.Tensor):
            image_grid_thw = image_grid_thw.tolist()
        vision_start_indices = torch.argwhere(input_tokens == vision_start_token_id).squeeze(1)
        image_indices = vision_start_indices + 1
        image_nums = image_indices.shape[0]
        llm_pos_ids_list: 'list' = []
        st = 0
        input_tokens_len = input_tokens.shape[0]
        for image_index in range(image_nums):
            ed = image_indices[image_index].item()
            t, h, w = image_grid_thw[image_index][0], image_grid_thw[image_index][1], image_grid_thw[image_index][2]
            llm_grid_t, llm_grid_h, llm_grid_w = t, h // spatial_merge_size, w // spatial_merge_size
            text_len = ed - st
            st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
            llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)
            t_index = torch.arange(llm_grid_t).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten()
            h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
            w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
            llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + text_len + st_idx)
            st = ed + llm_grid_t * llm_grid_h * llm_grid_w
        if st < input_tokens_len:
            st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
            text_len = input_tokens_len - st
            llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)
        llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
        llm_positions = llm_positions[:, context_len:]
        mrope_position_delta = (llm_positions.max() + 1 - input_tokens_len).item()
        return llm_positions.tolist(), mrope_position_delta

    @staticmethod
    def get_next_input_positions(mrope_position_delta: 'int', context_len: 'int', seq_len: 'int') ->List[List[int]]:
        return [list(range(context_len + mrope_position_delta, seq_len + mrope_position_delta)) for _ in range(3)]


class LogitsProcessor(nn.Module):

    def __init__(self, config, skip_all_gather: 'bool'=False):
        super().__init__()
        self.config = config
        self.do_tensor_parallel_all_gather = not skip_all_gather and get_tensor_model_parallel_world_size() > 1

    def _get_normalized_prompt_logprobs(self, input_token_logprobs: 'torch.Tensor', logits_metadata: 'LogitsMetadata'):
        logprobs_cumsum = torch.cumsum(input_token_logprobs, dim=0, dtype=torch.float32)
        pruned_lens = torch.tensor(logits_metadata.extend_logprob_pruned_lens_cpu, device='cuda')
        start = torch.zeros_like(pruned_lens)
        start[1:] = torch.cumsum(pruned_lens[:-1], dim=0)
        end = torch.clamp(start + pruned_lens - 2, min=0, max=logprobs_cumsum.shape[0] - 1)
        sum_logp = logprobs_cumsum[end] - logprobs_cumsum[start] + input_token_logprobs[start]
        normalized_prompt_logprobs = sum_logp / (pruned_lens - 1).clamp(min=1)
        return normalized_prompt_logprobs

    @staticmethod
    def get_top_logprobs(all_logprobs: 'torch.Tensor', logits_metadata: 'LogitsMetadata'):
        max_k = max(logits_metadata.top_logprobs_nums)
        ret = all_logprobs.topk(max_k, dim=1)
        values = ret.values.tolist()
        indices = ret.indices.tolist()
        if logits_metadata.forward_mode.is_decode():
            output_top_logprobs = []
            for i, k in enumerate(logits_metadata.top_logprobs_nums):
                output_top_logprobs.append(list(zip(values[i][:k], indices[i][:k])))
            return None, output_top_logprobs
        else:
            input_top_logprobs, output_top_logprobs = [], []
            pt = 0
            for k, pruned_len in zip(logits_metadata.top_logprobs_nums, logits_metadata.extend_logprob_pruned_lens_cpu):
                if pruned_len <= 0:
                    input_top_logprobs.append([])
                    output_top_logprobs.append([])
                    continue
                input_top_logprobs.append([list(zip(values[pt + j][:k], indices[pt + j][:k])) for j in range(pruned_len - 1)])
                output_top_logprobs.append(list(zip(values[pt + pruned_len - 1][:k], indices[pt + pruned_len - 1][:k])))
                pt += pruned_len
            return input_top_logprobs, output_top_logprobs

    def forward(self, input_ids, hidden_states, weight, logits_metadata: 'Union[LogitsMetadata, ForwardBatch]'):
        if isinstance(logits_metadata, ForwardBatch):
            logits_metadata = LogitsMetadata.from_forward_batch(logits_metadata)
        assert isinstance(logits_metadata, LogitsMetadata)
        if logits_metadata.forward_mode.is_decode():
            last_index = None
            last_hidden = hidden_states
        else:
            last_index = torch.cumsum(logits_metadata.extend_seq_lens, dim=0) - 1
            last_hidden = hidden_states[last_index]
        last_logits = torch.matmul(last_hidden, weight.T)
        if self.do_tensor_parallel_all_gather:
            last_logits = tensor_model_parallel_all_gather(last_logits)
        last_logits = last_logits[:, :self.config.vocab_size].float()
        if hasattr(self.config, 'final_logit_softcapping'):
            last_logits.div_(self.config.final_logit_softcapping)
            torch.tanh(last_logits, out=last_logits)
            last_logits.mul_(self.config.final_logit_softcapping)
        if not logits_metadata.return_logprob:
            return LogitsProcessorOutput(next_token_logits=last_logits, next_token_logprobs=None, normalized_prompt_logprobs=None, input_token_logprobs=None, input_top_logprobs=None, output_top_logprobs=None)
        else:
            last_logprobs = torch.nn.functional.log_softmax(last_logits, dim=-1)
            if logits_metadata.forward_mode.is_decode():
                if logits_metadata.return_top_logprob:
                    output_top_logprobs = self.get_top_logprobs(last_logprobs, logits_metadata)[1]
                else:
                    output_top_logprobs = None
                return LogitsProcessorOutput(next_token_logits=last_logits, next_token_logprobs=last_logprobs, normalized_prompt_logprobs=None, input_token_logprobs=None, input_top_logprobs=None, output_top_logprobs=output_top_logprobs)
            else:
                pt, states, pruned_input_ids = 0, [], []
                for start_len, extend_len in zip(logits_metadata.extend_logprob_start_lens_cpu, logits_metadata.extend_seq_lens_cpu):
                    states.append(hidden_states[pt + start_len:pt + extend_len])
                    pruned_input_ids.append(input_ids[pt + start_len:pt + extend_len])
                    pt += extend_len
                states = torch.cat(states, dim=0)
                all_logits = torch.matmul(states, weight.T)
                if self.do_tensor_parallel_all_gather:
                    all_logits = tensor_model_parallel_all_gather(all_logits)
                all_logits = all_logits[:, :self.config.vocab_size].float()
                if hasattr(self.config, 'final_logit_softcapping'):
                    all_logits.div_(self.config.final_logit_softcapping)
                    torch.tanh(all_logits, out=all_logits)
                    all_logits.mul_(self.config.final_logit_softcapping)
                all_logprobs = all_logits
                del all_logits, hidden_states
                all_logprobs[:] = torch.nn.functional.log_softmax(all_logprobs, dim=-1)
                if logits_metadata.return_top_logprob:
                    input_top_logprobs, output_top_logprobs = self.get_top_logprobs(all_logprobs, logits_metadata)
                else:
                    input_top_logprobs = output_top_logprobs = None
                input_token_logprobs = all_logprobs[torch.arange(all_logprobs.shape[0], device='cuda'), torch.cat([torch.cat(pruned_input_ids)[1:], torch.tensor([0], device='cuda')])]
                normalized_prompt_logprobs = self._get_normalized_prompt_logprobs(input_token_logprobs, logits_metadata)
                return LogitsProcessorOutput(next_token_logits=last_logits, next_token_logprobs=last_logprobs, normalized_prompt_logprobs=normalized_prompt_logprobs, input_token_logprobs=input_token_logprobs, input_top_logprobs=input_top_logprobs, output_top_logprobs=output_top_logprobs)


@dataclass
class EmbeddingPoolerOutput:
    embeddings: 'torch.Tensor'


class PoolingType(IntEnum):
    LAST = 0


class Pooler(nn.Module):
    """A layer that pools specific information from hidden states.
    This layer does the following:
    1. Extracts specific tokens or aggregates data based on pooling method.
    2. Normalizes output if specified.
    3. Returns structured results as `PoolerOutput`.
    Attributes:
        pooling_type: The type of pooling to use (LAST, AVERAGE, MAX).
        normalize: Whether to normalize the pooled data.
    """

    def __init__(self, pooling_type: 'PoolingType', normalize: 'bool'):
        super().__init__()
        self.pooling_type = pooling_type
        self.normalize = normalize

    def forward(self, hidden_states: 'torch.Tensor', forward_batch: 'ForwardBatch') ->EmbeddingPoolerOutput:
        if self.pooling_type == PoolingType.LAST:
            last_token_indices = torch.cumsum(forward_batch.extend_seq_lens, dim=0) - 1
            pooled_data = hidden_states[last_token_indices]
        else:
            raise ValueError(f'Invalid pooling type: {self.pooling_type}')
        if self.normalize:
            pooled_data = nn.functional.normalize(pooled_data, p=2, dim=1)
        return EmbeddingPoolerOutput(embeddings=pooled_data)


class RadixAttention(nn.Module):
    """
    The attention layer implementation.
    """

    def __init__(self, num_heads: 'int', head_dim: 'int', scaling: 'float', num_kv_heads: 'int', layer_id: 'int', logit_cap: 'float'=0.0, v_head_dim: 'int'=-1, sliding_window_size: 'int'=-1, is_cross_attention: 'bool'=False):
        super().__init__()
        self.tp_q_head_num = num_heads
        self.tp_k_head_num = num_kv_heads
        self.tp_v_head_num = num_kv_heads
        self.head_dim = head_dim
        self.qk_head_dim = head_dim
        self.v_head_dim = v_head_dim if v_head_dim != -1 else head_dim
        self.scaling = scaling
        self.layer_id = layer_id
        self.logit_cap = logit_cap
        self.sliding_window_size = sliding_window_size or -1
        self.is_cross_attention = is_cross_attention

    def forward(self, q, k, v, forward_batch: 'ForwardBatch'):
        if k is not None:
            assert v is not None
            k = k.view(-1, self.tp_k_head_num, self.qk_head_dim)
            v = v.view(-1, self.tp_v_head_num, self.v_head_dim)
        return forward_batch.attn_backend.forward(q, k, v, self, forward_batch)


def is_flashinfer_available():
    """
    Check whether flashinfer is available.
    As of Oct. 6, 2024, it is only available on NVIDIA GPUs.
    """
    return torch.cuda.is_available() and not is_hip()


def is_ipv6(address):
    try:
        ipaddress.IPv6Address(address)
        return True
    except ipaddress.AddressValueError:
        return False


def top_k_top_p_min_p_sampling_from_probs_torch(probs: 'torch.Tensor', top_ks: 'torch.Tensor', top_ps: 'torch.Tensor', min_ps: 'torch.Tensor'):
    """A top-k, top-p and min-p sampling implementation with native pytorch operations."""
    probs_sort, probs_idx = probs.sort(dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    min_p_thresholds = probs_sort[:, 0] * min_ps
    probs_sort[probs_sum - probs_sort > top_ps.view(-1, 1)] = 0.0
    probs_sort[torch.arange(0, probs.shape[-1], device=probs.device).view(1, -1) >= top_ks.view(-1, 1)] = 0.0
    probs_sort[probs_sort < min_p_thresholds.view(-1, 1)] = 0.0
    probs_sort.div_(probs_sort.max(dim=-1, keepdim=True)[0])
    sampled_index = torch.multinomial(probs_sort, num_samples=1)
    batch_next_token_ids = torch.gather(probs_idx, dim=1, index=sampled_index).view(-1)
    return batch_next_token_ids


class Sampler(nn.Module):

    def __init__(self):
        super().__init__()
        self.use_nan_detectioin = not global_server_args_dict['disable_nan_detection']

    def forward(self, logits: 'Union[torch.Tensor, LogitsProcessorOutput]', sampling_info: 'SamplingBatchInfo'):
        if isinstance(logits, LogitsProcessorOutput):
            logits = logits.next_token_logits
        logits = logits.contiguous()
        if self.use_nan_detectioin and torch.any(torch.isnan(logits)):
            logger.warning('Detected errors during sampling! NaN in the logits.')
            logits = torch.where(torch.isnan(logits), torch.full_like(logits, -100000.0), logits)
            exit(1) if crash_on_warning else None
        if sampling_info.is_all_greedy:
            batch_next_token_ids = torch.argmax(logits, -1)
        else:
            logits.div_(sampling_info.temperatures)
            probs = torch.softmax(logits, dim=-1)
            logits = None
            del logits
            if global_server_args_dict['sampling_backend'] == 'flashinfer':
                max_top_k_round, batch_size = 32, probs.shape[0]
                uniform_samples = torch.rand((max_top_k_round, batch_size), device=probs.device)
                if sampling_info.need_min_p_sampling:
                    probs = top_k_renorm_prob(probs, sampling_info.top_ks)
                    probs = top_p_renorm_prob(probs, sampling_info.top_ps)
                    batch_next_token_ids, success = min_p_sampling_from_probs(probs, uniform_samples, sampling_info.min_ps)
                else:
                    batch_next_token_ids, success = top_k_top_p_sampling_from_probs(probs, uniform_samples, sampling_info.top_ks, sampling_info.top_ps, filter_apply_order='joint')
                if not torch.all(success):
                    logger.warning('Detected errors during sampling!')
                    batch_next_token_ids = torch.zeros_like(batch_next_token_ids)
            elif global_server_args_dict['sampling_backend'] == 'pytorch':
                batch_next_token_ids = top_k_top_p_min_p_sampling_from_probs_torch(probs, sampling_info.top_ks, sampling_info.top_ps, sampling_info.min_ps)
            else:
                raise ValueError(f"Invalid sampling backend: {global_server_args_dict['sampling_backend']}")
        return batch_next_token_ids


DEFAULT_VOCAB_PADDING_SIZE = 64


class UnquantizedEmbeddingMethod(QuantizeMethodBase):
    """Unquantized method for embeddings."""

    def create_weights(self, layer: 'torch.nn.Module', input_size_per_partition: 'int', output_partition_sizes: 'List[int]', input_size: 'int', output_size: 'int', params_dtype: 'torch.dtype', **extra_weight_attrs):
        """Create weights for embedding layer."""
        weight = Parameter(torch.empty(sum(output_partition_sizes), input_size_per_partition, dtype=params_dtype), requires_grad=False)
        set_weight_attrs(weight, {'input_dim': 1, 'output_dim': 0})
        layer.register_parameter('weight', weight)
        set_weight_attrs(weight, extra_weight_attrs)

    def apply(self, layer: 'torch.nn.Module', x: 'torch.Tensor', bias: 'Optional[torch.Tensor]'=None) ->torch.Tensor:
        return F.linear(x, layer.weight, bias)

    def embedding(self, layer: 'torch.nn.Module', input_: 'torch.Tensor') ->torch.Tensor:
        return F.embedding(input_, layer.weight)


@torch.jit.script
def get_masked_input_and_mask(input_: 'torch.Tensor', org_vocab_start_index: 'int', org_vocab_end_index: 'int', num_org_vocab_padding: 'int', added_vocab_start_index: 'int', added_vocab_end_index: 'int') ->Tuple[torch.Tensor, torch.Tensor]:
    org_vocab_mask = (input_ >= org_vocab_start_index) & (input_ < org_vocab_end_index)
    added_vocab_mask = (input_ >= added_vocab_start_index) & (input_ < added_vocab_end_index)
    added_offset = added_vocab_start_index - (org_vocab_end_index - org_vocab_start_index) - num_org_vocab_padding
    valid_offset = org_vocab_start_index * org_vocab_mask + added_offset * added_vocab_mask
    vocab_mask = org_vocab_mask | added_vocab_mask
    input_ = vocab_mask * (input_ - valid_offset)
    return input_, ~vocab_mask


def method_has_implemented_embedding(method_class: 'Type[QuantizeMethodBase]') ->bool:
    """
    Not all quant methods have embedding implemented, so we need to check that
    it exists for our given method. We check this by making sure the function
    has been changed from the base implementation.
    """
    base_embedding = inspect.getattr_static(QuantizeMethodBase, 'embedding', None)
    class_embedding = inspect.getattr_static(method_class, 'embedding', None)
    return class_embedding is not None and class_embedding is not base_embedding


def pad_vocab_size(vocab_size: 'int', pad_to: 'int'=DEFAULT_VOCAB_PADDING_SIZE) ->int:
    """Pad the vocab size to the given value."""
    return (vocab_size + pad_to - 1) // pad_to * pad_to


def vocab_range_from_per_partition_vocab_size(per_partition_vocab_size: 'int', rank: 'int', offset: 'int'=0) ->Sequence[int]:
    index_f = rank * per_partition_vocab_size
    index_l = index_f + per_partition_vocab_size
    return index_f + offset, index_l + offset


def vocab_range_from_global_vocab_size(global_vocab_size: 'int', rank: 'int', world_size: 'int', offset: 'int'=0) ->Sequence[int]:
    per_partition_vocab_size = divide(global_vocab_size, world_size)
    return vocab_range_from_per_partition_vocab_size(per_partition_vocab_size, rank, offset=offset)


class BaseLayerWithLoRA(nn.Module):

    def __init__(self, base_layer, segment_gemm, lora_rank, scaling):
        super().__init__()
        self.base_layer = base_layer
        self.segment_gemm = segment_gemm
        self.lora_rank = lora_rank
        self.scaling = scaling
        self.set_lora = False

    def forward(self, x: 'torch.Tensor'):
        return self.base_layer.forward(x)

    def set_lora_info(self, *args):
        pass


class VocabParallelEmbeddingWithLoRA(BaseLayerWithLoRA):

    def __init__(self, base_layer: 'VocabParallelEmbedding', segment_gemm, lora_rank, scaling) ->None:
        super().__init__(base_layer, segment_gemm, lora_rank, scaling)
        self.weight = base_layer.weight


class ColumnParallelLinearWithLoRA(BaseLayerWithLoRA):

    def __init__(self, base_layer: 'ColumnParallelLinear', segment_gemm, lora_rank, scaling) ->None:
        super().__init__(base_layer, segment_gemm, lora_rank, scaling)

    def apply_lora(self, output: 'torch.Tensor', x: 'torch.Tensor') ->torch.Tensor:
        return output

    def forward(self, input_: 'torch.Tensor'):
        bias = self.base_layer.bias if not self.base_layer.skip_bias_add else None
        output_parallel = self.base_layer.quant_method.apply(self.base_layer, input_, bias)
        if self.set_lora:
            output_parallel = self.apply_lora(output_parallel, input_)
        if self.base_layer.gather_output:
            output = tensor_model_parallel_all_gather(output_parallel)
        else:
            output = output_parallel
        output_bias = self.base_layer.bias if self.base_layer.skip_bias_add else None
        return output, output_bias


class MergedColumnParallelLinearWithLoRA(ColumnParallelLinearWithLoRA):

    def __init__(self, base_layer: 'MergedColumnParallelLinear', segment_gemm, lora_rank, scaling) ->None:
        super().__init__(base_layer, segment_gemm, lora_rank, scaling)

    def set_lora_info(self, A_buffer, B_buffer, bs, seg_indptr, weight_indices):
        self.set_lora = True
        self.A_buffer = A_buffer
        self.B_buffer = B_buffer
        self.bs = bs
        self.seg_indptr = seg_indptr
        self.weight_indices = weight_indices

    def apply_lora(self, base_output: 'torch.Tensor', x: 'torch.Tensor') ->torch.Tensor:
        lora_a_output = self.segment_gemm.run(x=x, weights=self.A_buffer, batch_size=self.bs, weight_column_major=True, seg_indptr=self.seg_indptr, weight_indices=self.weight_indices)
        lora_output = torch.empty_like(base_output)
        output_dim = lora_output.shape[-1] // 2
        for i in range(2):
            left = output_dim * i
            right = left + output_dim
            lora_output[:, left:right] = self.segment_gemm.run(x=lora_a_output[:, self.lora_rank * i:self.lora_rank * (i + 1)].contiguous(), weights=self.B_buffer[:, left:right, :].contiguous(), batch_size=self.bs, weight_column_major=True, seg_indptr=self.seg_indptr, weight_indices=self.weight_indices)
        return base_output + lora_output * self.scaling


class QKVParallelLinearWithLoRA(ColumnParallelLinearWithLoRA):

    def __init__(self, base_layer: 'QKVParallelLinear', segment_gemm, lora_rank, scaling) ->None:
        super().__init__(base_layer, segment_gemm, lora_rank, scaling)

    def set_lora_info(self, A_buffer_qkv, B_buffer_q, B_buffer_kv, bs, seg_indptr, weight_indices):
        self.set_lora = True
        self.A_buffer_qkv = A_buffer_qkv
        self.B_buffer_q = B_buffer_q
        self.B_buffer_kv = B_buffer_kv
        self.bs = bs
        self.seg_indptr = seg_indptr
        self.weight_indices = weight_indices

    def apply_lora(self, base_output: 'torch.Tensor', x: 'torch.Tensor') ->torch.Tensor:
        lora_a_output = self.segment_gemm.run(x=x, weights=self.A_buffer_qkv, batch_size=self.bs, weight_column_major=True, seg_indptr=self.seg_indptr, weight_indices=self.weight_indices)
        lora_output = torch.empty_like(base_output)
        output_dim_q = self.B_buffer_q.shape[-2]
        lora_output[:, :output_dim_q] = self.segment_gemm.run(x=lora_a_output[:, :self.lora_rank].contiguous(), weights=self.B_buffer_q, batch_size=self.bs, weight_column_major=True, seg_indptr=self.seg_indptr, weight_indices=self.weight_indices)
        output_dim_kv = self.B_buffer_kv.shape[-2] // 2
        for i in range(2):
            left = output_dim_kv * i
            right = left + output_dim_kv
            lora_output[:, output_dim_q + left:output_dim_q + right] = self.segment_gemm.run(x=lora_a_output[:, self.lora_rank * (i + 1):self.lora_rank * (i + 2)].contiguous(), weights=self.B_buffer_kv[:, left:right, :].contiguous(), batch_size=self.bs, weight_column_major=True, seg_indptr=self.seg_indptr, weight_indices=self.weight_indices)
        return base_output + lora_output * self.scaling


class RowParallelLinearWithLoRA(BaseLayerWithLoRA):

    def __init__(self, base_layer: 'RowParallelLinear', segment_gemm, lora_rank, scaling) ->None:
        super().__init__(base_layer, segment_gemm, lora_rank, scaling)

    def set_lora_info(self, A_buffer, B_buffer, bs, seg_indptr, weight_indices):
        self.set_lora = True
        self.A_buffer = A_buffer
        self.B_buffer = B_buffer
        self.bs = bs
        self.seg_indptr = seg_indptr
        self.weight_indices = weight_indices

    def apply_lora(self, base_output: 'torch.Tensor', x: 'torch.Tensor') ->torch.Tensor:
        lora_output = self.segment_gemm.run(x=x, weights=self.A_buffer, batch_size=self.bs, weight_column_major=True, seg_indptr=self.seg_indptr, weight_indices=self.weight_indices)
        lora_output = self.segment_gemm.run(x=lora_output, weights=self.B_buffer, batch_size=self.bs, weight_column_major=True, seg_indptr=self.seg_indptr, weight_indices=self.weight_indices)
        return base_output + lora_output * self.scaling

    def forward(self, input_):
        if self.base_layer.input_is_parallel:
            input_parallel = input_
        else:
            tp_rank = get_tensor_model_parallel_rank()
            splitted_input = split_tensor_along_last_dim(input_, num_partitions=self.base_layer.tp_size)
            input_parallel = splitted_input[tp_rank].contiguous()
        output_parallel = self.base_layer.quant_method.apply(self.base_layer, input_parallel)
        if self.set_lora:
            output_parallel = self.apply_lora(output_parallel, input_parallel)
        if self.base_layer.reduce_results and self.base_layer.tp_size > 1:
            output_ = tensor_model_parallel_all_reduce(output_parallel)
        else:
            output_ = output_parallel
        if not self.base_layer.skip_bias_add:
            output = output_ + self.base_layer.bias if self.base_layer.bias is not None else output_
            output_bias = None
        else:
            output = output_
            output_bias = self.base_layer.bias
        return output, output_bias


class LoRALayer(nn.Module):

    def __init__(self, config, base_hf_config):
        super().__init__()
        self.config = config
        self.base_hf_config = base_hf_config
        self.weights = {}
        self.weight_gpu = {}

    def load_to_gpu(self):
        for name, weight in self.weights.items():
            self.weight_gpu[name] = weight.to(torch.float16)

    def offload_from_gpu(self):
        for name, weight in self.weights.items():
            self.weight_gpu[name] = None


class LoRAAdapter(nn.Module):

    def __init__(self, uid, config, base_hf_config, load_config):
        super().__init__()
        self.uid = uid
        self.config = config
        assert self.config.hf_config['peft_type'].lower() == 'lora'
        self.base_hf_config = base_hf_config
        self.load_config = load_config
        self.scaling = self.config.lora_alpha / self.config.r
        self.layers = nn.ModuleList([LoRALayer(config, base_hf_config) for i in range(base_hf_config.num_hidden_layers)])
        self.weights = {}
        self.weights_gpu = {}

    def get_stacked_multiply(self, module_name):
        stacked_rank = {'qkv_proj': 3, 'kv_proj': 2, 'gate_up_proj': 2}
        return stacked_rank[module_name] if module_name in stacked_rank else 1

    def load_to_gpu(self):
        for name, weight in self.weights.items():
            self.weights_gpu[name] = weight.to(torch.float16)
        for layer in self.layers:
            layer.load_to_gpu()

    def offload_from_gpu(self):
        for name, weight in self.weights.items():
            self.weights_gpu[name] = None
        for layer in self.layers:
            layer.offload_from_gpu()

    def initialize_weights(self):
        model_path = self.config.path
        loader = DefaultModelLoader(self.load_config)
        revision = getattr(self.config.hf_config, 'revision', None)
        for name, loaded_weight in loader._get_weights_iterator(DefaultModelLoader.Source(model_path, revision=revision, fall_back_to_pt=True)):
            match = re.search('layers\\.(\\d+)\\.', name)
            if match is not None:
                layer_id = int(match.group(1))
                self.layers[layer_id].weights[name] = loaded_weight.cpu()
            else:
                self.weights[name] = loaded_weight.cpu()
        for i in range(self.base_hf_config.num_hidden_layers):
            layer = self.layers[i]
            weight_names = [name for name, _ in layer.weights.items()]
            for weight_name in weight_names:
                if 'k_proj' in weight_name:
                    q_name = weight_name.replace('k_proj', 'q_proj')
                    v_name = weight_name.replace('k_proj', 'v_proj')
                    kv_name = weight_name.replace('k_proj', 'kv_proj')
                    qkv_name = weight_name.replace('k_proj', 'qkv_proj')
                    if 'lora_A' in weight_name:
                        layer.weights[qkv_name] = torch.cat((layer.weights[q_name], layer.weights[weight_name], layer.weights[v_name]), 0)
                        layer.weights.pop(q_name)
                        layer.weights.pop(weight_name)
                        layer.weights.pop(v_name)
                    else:
                        layer.weights[kv_name] = torch.cat((layer.weights[weight_name], layer.weights[v_name]), 0)
                        layer.weights.pop(weight_name)
                        layer.weights.pop(v_name)
                elif 'gate_proj' in weight_name:
                    up_name = weight_name.replace('gate_proj', 'up_proj')
                    gate_up_name = weight_name.replace('gate_proj', 'gate_up_proj')
                    layer.weights[gate_up_name] = torch.cat((layer.weights[weight_name], layer.weights[up_name]), 0)
                    layer.weights.pop(weight_name)
                    layer.weights.pop(up_name)


WEIGHT_LOADER_V2_SUPPORTED = ['CompressedTensorsLinearMethod', 'AWQMarlinLinearMethod', 'AWQLinearMethod', 'GPTQMarlinLinearMethod', 'Fp8LinearMethod', 'MarlinLinearMethod', 'GPTQLinearMethod']


def adjust_marlin_shard(param, shard_size, shard_offset):
    marlin_tile_size = getattr(param, 'marlin_tile_size', None)
    if marlin_tile_size is None:
        return shard_size, shard_offset
    return shard_size * marlin_tile_size, shard_offset * marlin_tile_size


def adjust_scalar_to_fused_array(param, loaded_weight, shard_id):
    """For fused modules (QKV and MLP) we have an array of length
    N that holds 1 scale for each "logical" matrix. So the param
    is an array of length N. The loaded_weight corresponds to
    one of the shards on disk. Here, we slice the param based on
    the shard_id for loading.
    """
    qkv_idxs = {'q': 0, 'k': 1, 'v': 2}
    if isinstance(shard_id, str):
        shard_id = qkv_idxs[shard_id]
    elif not isinstance(shard_id, int):
        raise ValueError(f'Unknown Shard Id {shard_id}')
    if len(loaded_weight.shape) != 0:
        assert loaded_weight.shape[0] == 1
        loaded_weight = loaded_weight[0]
    return param[shard_id], loaded_weight


class BaiChuanMLP(nn.Module):

    def __init__(self, hidden_size: 'int', intermediate_size: 'int', hidden_act: 'str', quant_config: 'Optional[QuantizationConfig]'=None):
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(hidden_size, [intermediate_size] * 2, bias=False, quant_config=quant_config)
        self.down_proj = RowParallelLinear(intermediate_size, hidden_size, bias=False, quant_config=quant_config)
        if hidden_act != 'silu':
            raise ValueError(f'Unsupported activation: {hidden_act}. Only silu is supported for now.')
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


def adjust_bitsandbytes_4bit_shard(param: 'Parameter', qkv_offsets: 'Dict[str, Tuple[int, int]]', loaded_shard_id: 'str') ->Tuple[int, int]:
    """Adjust the quantization offsets and sizes for BitsAndBytes sharding."""
    total, _ = qkv_offsets['total']
    orig_offset, orig_size = qkv_offsets[loaded_shard_id]
    quantized_total = param.data.shape[0]
    quantized_offset = orig_offset * quantized_total // total
    quantized_size = orig_size * quantized_total // total
    return quantized_size, quantized_offset


def _get_alibi_slopes(total_num_heads: 'int') ->torch.Tensor:
    closest_power_of_2 = 2 ** math.floor(math.log2(total_num_heads))
    base = torch.tensor(2 ** -2 ** -(math.log2(closest_power_of_2) - 3), dtype=torch.float32)
    powers = torch.arange(1, 1 + closest_power_of_2, dtype=torch.int32)
    slopes = torch.pow(base, powers)
    if closest_power_of_2 != total_num_heads:
        extra_base = torch.tensor(2 ** -2 ** -(math.log2(2 * closest_power_of_2) - 3), dtype=torch.float32)
        num_remaining_heads = min(closest_power_of_2, total_num_heads - closest_power_of_2)
        extra_powers = torch.arange(start=1, end=1 + 2 * num_remaining_heads, step=2, dtype=torch.int32)
        slopes = torch.cat([slopes, torch.pow(extra_base, extra_powers)], dim=0)
    return slopes


class BaiChuanAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, hidden_size: 'int', num_heads: 'int', position_embedding: 'str', rope_theta: 'float'=10000, max_position_embeddings: 'int'=8192, quant_config: 'Optional[QuantizationConfig]'=None, layer_id: 'int'=0):
        super().__init__()
        self.hidden_size = hidden_size
        tensor_model_parallel_world_size = get_tensor_model_parallel_world_size()
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tensor_model_parallel_world_size == 0
        self.num_heads = self.total_num_heads // tensor_model_parallel_world_size
        self.head_dim = hidden_size // self.total_num_heads
        self.postion_embedding = position_embedding
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
        self.total_num_kv_heads = self.num_heads
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.W_pack = QKVParallelLinear(hidden_size, self.head_dim, self.total_num_heads, self.total_num_heads, bias=False, quant_config=quant_config)
        self.o_proj = RowParallelLinear(self.total_num_heads * self.head_dim, hidden_size, bias=False, quant_config=quant_config)
        if self.postion_embedding == 'ALIBI':
            tp_rank = get_tensor_model_parallel_rank()
            head_start = tp_rank * self.num_heads
            head_end = (tp_rank + 1) * self.num_heads
            alibi_slopes = _get_alibi_slopes(self.total_num_heads)
            alibi_slopes = alibi_slopes[head_start:head_end].tolist()
            scaling = self.head_dim ** -0.5
            self.attn = RadixAttention(self.num_heads, self.head_dim, scaling, num_kv_heads=self.num_kv_heads, layer_id=layer_id)
        else:
            self.rotary_emb = get_rope(self.head_dim, rotary_dim=self.head_dim, max_position=self.max_position_embeddings, base=self.rope_theta)
            self.scaling = self.head_dim ** -0.5
            self.attn = RadixAttention(self.num_heads, self.head_dim, self.scaling, num_kv_heads=self.num_kv_heads, layer_id=layer_id)

    def forward(self, positions: 'torch.Tensor', hidden_states: 'torch.Tensor', forward_batch: 'ForwardBatch') ->torch.Tensor:
        qkv, _ = self.W_pack(hidden_states)
        q, k, v = qkv.chunk(chunks=3, dim=-1)
        if self.postion_embedding != 'ALIBI':
            q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v, forward_batch)
        output, _ = self.o_proj(attn_output)
        return output


class BaiChuanDecoderLayer(nn.Module):

    def __init__(self, config: 'PretrainedConfig', position_embedding: 'str', layer_id: 'int'=0, quant_config: 'Optional[QuantizationConfig]'=None):
        super().__init__()
        self.hidden_size = config.hidden_size
        rope_theta = getattr(config, 'rope_theta', 10000)
        max_position_embeddings = getattr(config, 'max_position_embeddings', 8192)
        self.self_attn = BaiChuanAttention(hidden_size=self.hidden_size, num_heads=config.num_attention_heads, position_embedding=position_embedding, rope_theta=rope_theta, layer_id=layer_id, max_position_embeddings=max_position_embeddings, quant_config=quant_config)
        self.mlp = BaiChuanMLP(hidden_size=self.hidden_size, intermediate_size=config.intermediate_size, hidden_act=config.hidden_act, quant_config=quant_config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, positions: 'torch.Tensor', hidden_states: 'torch.Tensor', forward_batch: 'ForwardBatch', residual: 'Optional[torch.Tensor]') ->Tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions=positions, hidden_states=hidden_states, forward_batch=forward_batch)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class BaiChuanModel(nn.Module):

    def __init__(self, config: 'PretrainedConfig', position_embedding: 'str', quant_config: 'Optional[QuantizationConfig]'=None):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([BaiChuanDecoderLayer(config, layer_id=i, position_embedding=position_embedding, quant_config=quant_config) for i in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids: 'torch.Tensor', positions: 'torch.Tensor', forward_batch: 'ForwardBatch') ->torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for i in range(len(self.layers)):
            layer = self.layers[i]
            hidden_states, residual = layer(positions, hidden_states, forward_batch, residual)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class BaiChuanBaseForCausalLM(nn.Module):
    packed_modules_mapping = {'W_pack': ['W_pack'], 'gate_up_proj': ['gate_proj', 'up_proj']}
    supported_lora_modules = ['W_pack', 'o_proj', 'gate_up_proj', 'down_proj']
    embedding_modules = {}
    embedding_padding_modules = []

    def __init__(self, config: 'PretrainedConfig', position_embedding: 'str', cache_config=None, quant_config: 'Optional[QuantizationConfig]'=None):
        super().__init__()
        self.config = config
        self.quant_config = quant_config
        self.model = BaiChuanModel(config, position_embedding, quant_config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size, quant_config=quant_config)
        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
        self.logits_processor = LogitsProcessor(config)

    def forward(self, input_ids: 'torch.Tensor', positions: 'torch.Tensor', forward_batch: 'ForwardBatch') ->torch.Tensor:
        hidden_states = self.model(input_ids, positions, forward_batch)
        return self.logits_processor(input_ids, hidden_states, self.lm_head.weight, forward_batch)

    def load_weights(self, weights: 'Iterable[Tuple[str, torch.Tensor]]'):
        stacked_params_mapping = [('gate_up_proj', 'gate_proj', 0), ('gate_up_proj', 'up_proj', 1)]
        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            if 'rotary_emb.inv_freq' in name:
                continue
            if name == 'lm_head.weight':
                is_baichuan2 = self.config.vocab_size == 125696
                if is_baichuan2:
                    loaded_weight = torch.nn.functional.normalize(loaded_weight)
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                if name.endswith('.bias') and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                if name.endswith('.bias') and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, 'weight_loader', default_weight_loader)
                weight_loader(param, loaded_weight)


class BaichuanForCausalLM(BaiChuanBaseForCausalLM):
    """Baichuan 13B and Baichuan2 7B/13B."""

    def __init__(self, config, cache_config=None, quant_config: 'Optional[QuantizationConfig]'=None):
        if config.hidden_size == 4096:
            super().__init__(config, 'ROPE', cache_config, quant_config)
        else:
            super().__init__(config, 'ALIBI', cache_config, quant_config)


class GLMAttention(nn.Module):

    def __init__(self, config, layer_id: 'int'=0, cache_config=None, quant_config: 'Optional[QuantizationConfig]'=None):
        super().__init__()
        self.hidden_size = config.hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = config.num_attention_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.multi_query_attention = config.multi_query_attention
        self.total_num_kv_heads = config.multi_query_group_num if config.multi_query_attention else config.num_attention_heads
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = config.hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim ** -0.5
        self.query_key_value = QKVParallelLinear(self.hidden_size, self.head_dim, self.total_num_heads, self.total_num_kv_heads, bias=config.add_bias_linear or config.add_qkv_bias, quant_config=quant_config)
        self.dense = RowParallelLinear(self.total_num_heads * self.head_dim, config.hidden_size, bias=config.add_bias_linear, quant_config=quant_config)
        rope_ratio = getattr(config, 'rope_ratio', 1.0)
        max_positions = getattr(config, 'seq_length', 8192)
        self.rotary_emb = get_rope(self.head_dim, rotary_dim=self.head_dim // 2, max_position=max_positions, base=10000 * rope_ratio, is_neox_style=False)
        self.attn = RadixAttention(self.num_heads, self.head_dim, self.scaling, num_kv_heads=self.num_kv_heads, layer_id=layer_id)

    def forward(self, hidden_states: 'torch.Tensor', position_ids: 'torch.Tensor', forward_batch: 'ForwardBatch') ->torch.Tensor:
        qkv, _ = self.query_key_value(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(position_ids, q, k)
        context_layer = self.attn(q, k, v, forward_batch)
        attn_output, _ = self.dense(context_layer)
        return attn_output


class GLMMLP(nn.Module):
    """MLP.

    MLP will take the input with h hidden state, project it to 4*h
    hidden dimension, perform nonlinear transformation, and project the
    state back into h hidden dimension.
    """

    def __init__(self, config, quant_config: 'Optional[QuantizationConfig]'=None):
        super().__init__()
        self.add_bias = config.add_bias_linear
        self.dense_h_to_4h = MergedColumnParallelLinear(config.hidden_size, [config.ffn_hidden_size] * 2, bias=config.add_bias_linear, quant_config=quant_config)
        self.activation_func = SiluAndMul()
        self.dense_4h_to_h = RowParallelLinear(config.ffn_hidden_size, config.hidden_size, bias=config.add_bias_linear, quant_config=quant_config)

    def forward(self, hidden_states):
        intermediate_parallel, _ = self.dense_h_to_4h(hidden_states)
        intermediate_parallel = self.activation_func(intermediate_parallel)
        output, _ = self.dense_4h_to_h(intermediate_parallel)
        return output


class GLMBlock(nn.Module):
    """A single transformer layer.

    Transformer layer takes input with size [s, b, h] and returns an
    output of the same size.
    """

    def __init__(self, config, layer_id: 'int', cache_config=None, quant_config: 'Optional[QuantizationConfig]'=None):
        super().__init__()
        self.apply_residual_connection_post_layernorm = config.apply_residual_connection_post_layernorm
        self.fp32_residual_connection = config.fp32_residual_connection
        layer_norm_func = RMSNorm if config.rmsnorm else LayerNorm
        self.input_layernorm = layer_norm_func(config.hidden_size, eps=config.layernorm_epsilon)
        self.self_attention = GLMAttention(config, layer_id, cache_config, quant_config)
        self.hidden_dropout = config.hidden_dropout
        self.post_attention_layernorm = layer_norm_func(config.hidden_size, eps=config.layernorm_epsilon)
        self.mlp = GLMMLP(config, quant_config)

    def forward(self, hidden_states: 'torch.Tensor', position_ids: 'torch.Tensor', forward_batch: 'ForwardBatch') ->torch.Tensor:
        layernorm_output = self.input_layernorm(hidden_states)
        attention_output = self.self_attention(hidden_states=layernorm_output, position_ids=position_ids, forward_batch=forward_batch)
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = hidden_states
        layernorm_input = residual + attention_output
        layernorm_output = self.post_attention_layernorm(layernorm_input)
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = layernorm_input
        output = self.mlp(layernorm_output) + residual
        return output


class GLMTransformer(nn.Module):
    """Transformer class."""

    def __init__(self, config, cache_config=None, quant_config: 'Optional[QuantizationConfig]'=None):
        super().__init__()
        self.post_layer_norm = config.post_layer_norm
        self.num_layers = config.num_layers
        self.layers = nn.ModuleList([GLMBlock(config, i, cache_config, quant_config) for i in range(self.num_layers)])
        if self.post_layer_norm:
            layer_norm_func = RMSNorm if config.rmsnorm else LayerNorm
            self.final_layernorm = layer_norm_func(config.hidden_size, eps=config.layernorm_epsilon)

    def forward(self, hidden_states: 'torch.Tensor', position_ids: 'torch.Tensor', forward_batch: 'ForwardBatch') ->torch.Tensor:
        for i in range(self.num_layers):
            layer = self.layers[i]
            hidden_states = layer(hidden_states=hidden_states, position_ids=position_ids, forward_batch=forward_batch)
        if self.post_layer_norm:
            hidden_states = self.final_layernorm(hidden_states)
        return hidden_states


class ChatGLMM(nn.Module):

    def __init__(self, config, cache_config=None, quant_config: 'Optional[QuantizationConfig]'=None):
        super().__init__()
        self.embedding = VocabParallelEmbedding(config.padded_vocab_size, config.hidden_size)
        self.num_layers = config.num_layers
        self.multi_query_group_num = config.multi_query_group_num
        self.kv_channels = config.kv_channels
        self.encoder = GLMTransformer(config, cache_config, quant_config)
        self.output_layer = ParallelLMHead(config.padded_vocab_size, config.hidden_size)

    def forward(self, input_ids: 'torch.Tensor', position_ids: 'torch.Tensor', forward_batch: 'ForwardBatch') ->torch.Tensor:
        inputs_embeds = self.embedding(input_ids)
        hidden_states = self.encoder(hidden_states=inputs_embeds, position_ids=position_ids, forward_batch=forward_batch)
        return hidden_states


class ChatGLMForCausalLM(nn.Module):
    packed_modules_mapping = {'query_key_value': ['query_key_value'], 'dense_h_to_4h': ['dense_h_to_4h']}
    supported_lora_modules = ['query_key_value', 'dense', 'dense_h_to_4h', 'dense_4h_to_h']
    embedding_modules = {}
    embedding_padding_modules = []

    def __init__(self, config: 'ChatGLMConfig', cache_config=None, quant_config: 'Optional[QuantizationConfig]'=None, lora_config: 'Optional[LoraConfig]'=None):
        super().__init__()
        self.config: 'ChatGLMConfig' = config
        self.quant_config = quant_config
        self.max_position_embeddings = getattr(config, 'max_sequence_length', 8192)
        self.transformer = ChatGLMM(config, cache_config, quant_config)
        self.lm_head = self.transformer.output_layer
        self.logits_processor = LogitsProcessor(config)

    @torch.no_grad()
    def forward(self, input_ids: 'torch.Tensor', positions: 'torch.Tensor', forward_batch: 'ForwardBatch') ->torch.Tensor:
        hidden_states = self.transformer(input_ids, positions, forward_batch)
        return self.logits_processor(input_ids, hidden_states, self.lm_head.weight, forward_batch)

    def load_weights(self, weights: 'Iterable[Tuple[str, torch.Tensor]]'):
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        for name, loaded_weight in weights:
            if 'rotary_pos_emb.inv_freq' in name:
                continue
            if 'word_embeddings' in name:
                name = name.replace('.word_embeddings', '')
            if name.endswith('.bias') and name not in params_dict:
                continue
            param = params_dict[name]
            weight_loader = getattr(param, 'weight_loader', default_weight_loader)
            weight_loader(param, loaded_weight)


class ChatGLMModel(ChatGLMForCausalLM):
    pass


@torch.compile
def layer_norm_func(hidden_states, weight, variance_epsilon):
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states
    mean = hidden_states.mean(-1, keepdim=True)
    variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    hidden_states = (hidden_states - mean) * torch.rsqrt(variance + variance_epsilon)
    hidden_states = weight * hidden_states
    return hidden_states


class LayerNorm(nn.Module):

    def __init__(self, param_shape=None, eps=1e-05):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(param_shape))
        self.variance_epsilon = eps
        set_weight_attrs(self.weight, {'weight_loader': self.weight_loader})

    def forward(self, hidden_states, residuals=None):
        hidden_states = layer_norm_func(hidden_states, self.weight, self.variance_epsilon)
        return hidden_states, residuals

    def weight_loader(self, param: 'Parameter', loaded_weight: 'torch.Tensor'):
        tp_rank = get_tensor_model_parallel_rank()
        shard_dim = 0 if param.dim() != 1 else None
        param_data = param.data
        if shard_dim is not None:
            shard_size = param_data.shape[shard_dim]
            start_idx = tp_rank * shard_size
            loaded_weight = loaded_weight.narrow(shard_dim, start_idx, shard_size)
        assert param_data.shape == loaded_weight.shape
        param_data.copy_(loaded_weight)


class CohereMLP(nn.Module):

    def __init__(self, config, quant_config: 'Optional[QuantizationConfig]'=None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_up_proj = MergedColumnParallelLinear(self.hidden_size, [self.intermediate_size] * 2, bias=False, quant_config=quant_config)
        self.down_proj = RowParallelLinear(self.intermediate_size, self.hidden_size, bias=False, quant_config=quant_config)
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class CohereAttention(nn.Module):

    def __init__(self, config: 'PretrainedConfig', layer_id: 'int'=0, quant_config: 'Optional[QuantizationConfig]'=None):
        super().__init__()
        tp_size = get_tensor_model_parallel_world_size()
        self.config = config
        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.total_num_heads = config.num_attention_heads
        self.num_heads = self.total_num_heads // tp_size
        self.head_dim = self.hidden_size // self.total_num_heads
        self.total_num_kv_heads = config.num_key_value_heads
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim ** -0.5
        self.max_position_embeddings = getattr(config, 'model_max_length', None) or getattr(config, 'max_position_embeddings', 8192)
        self.rope_theta = config.rope_theta
        self.rope_scaling = getattr(config, 'rope_scaling', None)
        self.use_qk_norm = getattr(config, 'use_qk_norm', False)
        self.qkv_proj = QKVParallelLinear(self.hidden_size, self.head_dim, self.total_num_heads, self.total_num_kv_heads, bias=False, quant_config=quant_config)
        self.o_proj = RowParallelLinear(self.total_num_heads * self.head_dim, self.hidden_size, bias=False, quant_config=quant_config)
        self.rotary_emb = get_rope(self.head_dim, rotary_dim=self.head_dim, max_position=self.max_position_embeddings, base=self.rope_theta, rope_scaling=self.rope_scaling, is_neox_style=False)
        self.attn = RadixAttention(self.num_heads, self.head_dim, self.scaling, num_kv_heads=self.num_kv_heads, layer_id=layer_id)
        if self.use_qk_norm:
            self.q_norm = LayerNorm(param_shape=(self.num_heads, self.head_dim), eps=config.layer_norm_eps)
            self.k_norm = LayerNorm(param_shape=(self.num_kv_heads, self.head_dim), eps=config.layer_norm_eps)

    def _apply_qk_norm(self, q, k):
        q = q.view(*q.shape[:-1], -1, self.head_dim)
        k = k.view(*k.shape[:-1], -1, self.head_dim)
        q, _ = self.q_norm(q)
        k, _ = self.k_norm(k)
        q = q.view(*q.shape[:-2], -1)
        k = k.view(*k.shape[:-2], -1)
        return q, k

    def forward(self, positions: 'torch.Tensor', hidden_states: 'torch.Tensor', forward_batch: 'ForwardBatch') ->torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        if self.use_qk_norm:
            q, k = self._apply_qk_norm(q, k)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v, forward_batch)
        output, _ = self.o_proj(attn_output)
        return output


class CohereDecoderLayer(nn.Module):

    def __init__(self, config: 'PretrainedConfig', layer_id: 'int'=0, quant_config: 'Optional[QuantizationConfig]'=None):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = CohereAttention(config, layer_id=layer_id, quant_config=quant_config)
        self.mlp = CohereMLP(config, quant_config=quant_config)
        self.input_layernorm = LayerNorm(param_shape=config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, positions: 'torch.Tensor', hidden_states: 'torch.Tensor', forward_batch: 'ForwardBatch', residual: 'Optional[torch.Tensor]') ->Tuple[torch.Tensor, torch.Tensor]:
        residual = hidden_states
        hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states_attention = self.self_attn(positions=positions, hidden_states=hidden_states, forward_batch=forward_batch)
        hidden_states_mlp = self.mlp(hidden_states)
        hidden_states = residual + hidden_states_attention + hidden_states_mlp
        return hidden_states, residual


class CohereModel(nn.Module):

    def __init__(self, config: 'PretrainedConfig', quant_config: 'Optional[QuantizationConfig]'=None):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([CohereDecoderLayer(config, i, quant_config=quant_config) for i in range(config.num_hidden_layers)])
        self.norm = LayerNorm(param_shape=config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, input_ids: 'torch.Tensor', positions: 'torch.Tensor', forward_batch: 'ForwardBatch') ->torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for i in range(len(self.layers)):
            layer = self.layers[i]
            hidden_states, residual = layer(positions, hidden_states, forward_batch, residual)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class CohereForCausalLM(nn.Module):

    def __init__(self, config: 'PretrainedConfig', quant_config: 'Optional[QuantizationConfig]'=None, cache_config=None) ->None:
        super().__init__()
        self.config = config
        self.quant_config = quant_config
        self.logits_processor = LogitsProcessor(config)
        self.model = CohereModel(config, quant_config)

    @torch.no_grad()
    def forward(self, input_ids: 'torch.Tensor', positions: 'torch.Tensor', forward_batch: 'ForwardBatch') ->torch.Tensor:
        hidden_states = self.model(input_ids, positions, forward_batch)
        return self.logits_processor(input_ids, hidden_states, self.model.embed_tokens.weight, forward_batch)

    def load_weights(self, weights: 'Iterable[Tuple[str, torch.Tensor]]'):
        stacked_params_mapping = [('qkv_proj', 'q_proj', 'q'), ('qkv_proj', 'k_proj', 'k'), ('qkv_proj', 'v_proj', 'v'), ('gate_up_proj', 'gate_proj', 0), ('gate_up_proj', 'up_proj', 1)]
        params_dict = dict(self.named_parameters())
        loaded_params = set()
        for name, loaded_weight in weights:
            for param_name, shard_name, shard_id in stacked_params_mapping:
                if shard_name not in name:
                    continue
                name = name.replace(shard_name, param_name)
                if name.endswith('.bias') and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                if 'lm_head.weight' in name:
                    continue
                if name.endswith('.bias') and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, 'weight_loader', default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)


class DbrxRouter(nn.Module):
    """A Router implementation for DBRX that returns logits for each expert
    per token.
    """

    def __init__(self, config: 'DbrxConfig', params_dtype: 'Optional[torch.dtype]'=None):
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.num_total_experts = config.ffn_config.moe_num_experts
        self.d_model = config.d_model
        self.layer = ReplicatedLinear(self.d_model, self.num_total_experts, bias=False, params_dtype=params_dtype, quant_config=None)

    def forward(self, hidden_states: 'torch.Tensor') ->torch.Tensor:
        router_logits, _ = self.layer(hidden_states)
        return router_logits


class DbrxExperts(nn.Module):
    """A tensor-parallel MoE implementation for DBRX.

    Each expert's weights are sharded across all ranks and a fused MoE
    kernel is used for the forward pass, and finally we reduce the outputs
    across ranks.
    """

    def __init__(self, config: 'DbrxConfig', quant_config: 'Optional[QuantizationConfig]'=None, params_dtype: 'Optional[torch.dtype]'=None):
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.num_total_experts = config.ffn_config.moe_num_experts
        self.top_k = config.ffn_config.moe_top_k
        self.d_model = config.d_model
        self.intermediate_size = config.ffn_config.ffn_hidden_size // self.tp_size
        if params_dtype is None:
            params_dtype = torch.get_default_dtype()
        self.params_dtype = params_dtype
        self.router = DbrxRouter(config, self.params_dtype)
        self.ws = nn.Parameter(torch.empty(self.num_total_experts, 2 * self.intermediate_size, self.d_model, device='cuda', dtype=self.params_dtype))
        self.w2s = nn.Parameter(torch.empty(self.num_total_experts, self.d_model, self.intermediate_size, device='cuda', dtype=self.params_dtype))
        set_weight_attrs(self.ws, {'weight_loader': self.weight_loader})
        set_weight_attrs(self.w2s, {'weight_loader': self.weight_loader})

    def weight_loader(self, param: 'nn.Parameter', loaded_weight: 'torch.Tensor', weight_name: 'str'):
        tp_rank = get_tensor_model_parallel_rank()
        param_data = param.data
        shard_size = self.intermediate_size
        shard = slice(tp_rank * shard_size, (tp_rank + 1) * shard_size)
        if weight_name.endswith('w1'):
            loaded_weight = torch.reshape(loaded_weight, [-1, self.intermediate_size * self.tp_size, self.d_model])
            param_data[:, 0:shard_size, :] = loaded_weight[:, shard, :]
        if weight_name.endswith('v1'):
            loaded_weight = torch.reshape(loaded_weight, [-1, self.intermediate_size * self.tp_size, self.d_model])
            param_data[:, shard_size:2 * shard_size, :] = loaded_weight[:, shard, :]
        if weight_name.endswith('w2'):
            loaded_weight = torch.reshape(loaded_weight, [-1, self.intermediate_size * self.tp_size, self.d_model]).transpose(1, 2)
            param_data[:] = loaded_weight[:, :, shard]

    def forward(self, hidden_states: 'torch.Tensor') ->torch.Tensor:
        num_tokens, hidden_size = hidden_states.shape
        hidden_states = hidden_states.view(-1, self.d_model)
        router_logits = self.router(hidden_states)
        final_hidden_states = fused_moe(hidden_states, self.ws, self.w2s, router_logits, self.top_k, renormalize=True, inplace=True)
        if self.tp_size > 1:
            final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states)
        return final_hidden_states.view(num_tokens, hidden_size)


class DbrxAttention(nn.Module):

    def __init__(self, config: 'DbrxConfig', layer_id: 'int'=0, quant_config: 'Optional[QuantizationConfig]'=None):
        super().__init__()
        self.d_model = config.d_model
        self.total_num_heads = config.n_heads
        self.head_dim = self.d_model // self.total_num_heads
        self.total_num_kv_heads = config.attn_config.kv_n_heads
        self.clip_qkv = config.attn_config.clip_qkv
        self.rope_theta = config.attn_config.rope_theta
        self.max_position = config.max_seq_len
        self.Wqkv = QKVParallelLinear(self.d_model, self.head_dim, self.total_num_heads, self.total_num_kv_heads, bias=False, quant_config=quant_config)
        self.out_proj = RowParallelLinear(self.d_model, self.d_model, bias=False, quant_config=quant_config)
        self.rotary_emb = get_rope(self.head_dim, rotary_dim=self.head_dim, max_position=self.max_position, base=int(self.rope_theta), is_neox_style=True)
        tp_world_size = get_tensor_model_parallel_world_size()
        self.tp_size = tp_world_size
        assert self.total_num_heads % tp_world_size == 0
        self.num_heads = self.total_num_heads // tp_world_size
        if self.total_num_kv_heads >= tp_world_size:
            assert self.total_num_kv_heads % tp_world_size == 0
        else:
            assert tp_world_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_world_size)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim ** -0.5
        self.attn = RadixAttention(self.num_heads, self.head_dim, self.scaling, num_kv_heads=self.num_kv_heads, layer_id=layer_id)

    def forward(self, position_ids: 'torch.Tensor', hidden_states: 'torch.Tensor', forward_batch: 'ForwardBatch') ->torch.Tensor:
        qkv, _ = self.Wqkv(hidden_states)
        if self.clip_qkv is not None:
            qkv.clamp_(min=-self.clip_qkv, max=self.clip_qkv)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(position_ids, q, k)
        attn_output = self.attn(q, k, v, forward_batch)
        hidden_states, _ = self.out_proj(attn_output)
        return hidden_states


class DbrxFusedNormAttention(nn.Module):

    def __init__(self, config: 'DbrxConfig', layer_id: 'int'=0, quant_config: 'Optional[QuantizationConfig]'=None):
        super().__init__()
        self.d_model = config.d_model
        self.attn = DbrxAttention(config, layer_id, quant_config=quant_config)
        self.norm_1 = nn.LayerNorm(self.d_model)
        self.norm_2 = nn.LayerNorm(self.d_model)

    def forward(self, position_ids: 'torch.Tensor', hidden_states: 'torch.Tensor', forward_batch: 'ForwardBatch') ->torch.Tensor:
        residual = hidden_states
        hidden_states = self.norm_1(hidden_states)
        x = self.attn(position_ids=position_ids, hidden_states=hidden_states, forward_batch=forward_batch)
        hidden_states = residual + x
        residual = hidden_states
        hidden_states = self.norm_2(hidden_states)
        return hidden_states, residual


class DbrxBlock(nn.Module):

    def __init__(self, config: 'DbrxConfig', layer_id: 'int'=0, quant_config: 'Optional[QuantizationConfig]'=None):
        super().__init__()
        self.norm_attn_norm = DbrxFusedNormAttention(config, layer_id, quant_config=quant_config)
        self.ffn = DbrxExperts(config, quant_config=quant_config)

    def forward(self, position_ids: 'torch.Tensor', hidden_states: 'torch.Tensor', forward_batch: 'ForwardBatch') ->torch.Tensor:
        hidden_states, residual = self.norm_attn_norm(position_ids=position_ids, hidden_states=hidden_states, forward_batch=forward_batch)
        hidden_states = self.ffn(hidden_states)
        hidden_states = hidden_states + residual
        return hidden_states


class DbrxModel(nn.Module):

    def __init__(self, config: 'DbrxConfig', quant_config: 'Optional[QuantizationConfig]'=None):
        super().__init__()
        self.wte = VocabParallelEmbedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList([DbrxBlock(config, i, quant_config=quant_config) for i in range(config.n_layers)])
        self.norm_f = nn.LayerNorm(config.d_model, eps=1e-05)
        for module in self.modules():
            if hasattr(module, 'bias') and isinstance(module.bias, nn.Parameter):
                module.register_parameter('bias', None)

    def forward(self, input_ids: 'torch.Tensor', position_ids: 'torch.Tensor', forward_batch: 'ForwardBatch', input_embeds: 'torch.Tensor'=None) ->torch.Tensor:
        if input_embeds is None:
            hidden_states = self.wte(input_ids)
        else:
            hidden_states = input_embeds
        for i in range(len(self.blocks)):
            block = self.blocks[i]
            hidden_states = block(position_ids, hidden_states, forward_batch)
        hidden_states = self.norm_f(hidden_states)
        return hidden_states


class DbrxForCausalLM(nn.Module):

    def __init__(self, config: 'DbrxConfig', quant_config: 'Optional[QuantizationConfig]'=None, cache_config=None):
        super().__init__()
        self.config = config
        self.quant_config = quant_config
        self.unpadded_vocab_size = config.vocab_size
        self.transformer = DbrxModel(config, quant_config=quant_config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.d_model, org_num_embeddings=config.vocab_size, padding_size=DEFAULT_VOCAB_PADDING_SIZE)
        self.logits_processor = LogitsProcessor(config)

    @torch.no_grad()
    def forward(self, input_ids: 'torch.Tensor', positions: 'torch.Tensor', forward_batch: 'ForwardBatch') ->torch.Tensor:
        hidden_states = self.transformer(input_ids, positions, forward_batch)
        return self.logits_processor(input_ids, hidden_states, self.lm_head.weight, forward_batch)

    def load_weights(self, weights: 'Iterable[Tuple[str, torch.Tensor]]'):
        expert_params_mapping = [('ws' if weight_name in ['w1', 'v1'] else 'w2s', f'experts.mlp.{weight_name}') for weight_name in ['w1', 'v1', 'w2']]
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        for name, loaded_weight in weights:
            for param_name, weight_name in expert_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, weight_name)
                break
            else:
                param = params_dict[name]
                weight_loader = getattr(param, 'weight_loader', default_weight_loader)
                weight_loader(param, loaded_weight)


class DeepseekMLP(nn.Module):

    def __init__(self, hidden_size: 'int', intermediate_size: 'int', hidden_act: 'str', quant_config: 'Optional[QuantizationConfig]'=None, reduce_results: 'bool'=True) ->None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(hidden_size, [intermediate_size] * 2, bias=False, quant_config=quant_config)
        self.down_proj = RowParallelLinear(intermediate_size, hidden_size, bias=False, quant_config=quant_config, reduce_results=reduce_results)
        if hidden_act != 'silu':
            raise ValueError(f'Unsupported activation: {hidden_act}. Only silu is supported for now.')
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class DeepseekMoE(nn.Module):

    def __init__(self, config: 'PretrainedConfig', quant_config: 'Optional[QuantizationConfig]'=None):
        super().__init__()
        self.config = config
        self.rank = get_tensor_model_parallel_rank()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.n_routed_experts = config.n_routed_experts
        self.top_k = config.num_experts_per_tok
        if self.tp_size > self.n_routed_experts:
            raise ValueError(f'Tensor parallel size {self.tp_size} is greater than the number of experts {self.n_routed_experts}.')
        self.experts = nn.ModuleList([DeepseekMLP(hidden_size=config.hidden_size, intermediate_size=config.moe_intermediate_size, hidden_act=config.hidden_act, quant_config=quant_config, reduce_results=False) for idx in range(self.n_routed_experts)])
        self.pack_params()
        self.gate = ReplicatedLinear(config.hidden_size, self.n_routed_experts, bias=False, quant_config=None)
        if config.n_shared_experts is not None:
            intermediate_size = config.moe_intermediate_size * config.n_shared_experts
            self.shared_experts = DeepseekMLP(hidden_size=config.hidden_size, intermediate_size=intermediate_size, hidden_act=config.hidden_act, quant_config=quant_config, reduce_results=False)

    def pack_params(self):
        w1 = []
        w2 = []
        for expert in self.experts:
            w1.append(expert.gate_up_proj.weight)
            w2.append(expert.down_proj.weight)
        self.w1 = torch._utils._flatten_dense_tensors(w1)
        w1s = torch._utils._unflatten_dense_tensors(self.w1, w1)
        for data, param in zip(w1s, w1):
            param.data = data
        self.w1 = self.w1.view(len(w1), *w1s[0].shape)
        self.w2 = torch._utils._flatten_dense_tensors(w2)
        w2s = torch._utils._unflatten_dense_tensors(self.w2, w2)
        for data, param in zip(w2s, w2):
            param.data = data
        self.w2 = self.w2.view(len(w2), *w2s[0].shape)

    def forward(self, hidden_states: 'torch.Tensor') ->torch.Tensor:
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        if self.config.n_shared_experts is not None:
            shared_output = self.shared_experts(hidden_states)
        router_logits, _ = self.gate(hidden_states)
        final_hidden_states = fused_moe(hidden_states, self.w1, self.w2, router_logits, self.top_k, renormalize=self.config.norm_topk_prob, inplace=True)
        if self.config.n_shared_experts is not None:
            final_hidden_states = final_hidden_states + shared_output
        final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states)
        return final_hidden_states.view(num_tokens, hidden_dim)


class DeepseekAttention(nn.Module):

    def __init__(self, hidden_size: 'int', num_heads: 'int', num_kv_heads: 'int', layer_id: 'int'=0, rope_theta: 'float'=10000, rope_scaling: 'Optional[Dict[str, Any]]'=None, max_position_embeddings: 'int'=8192, cache_config=None, quant_config: 'Optional[QuantizationConfig]'=None) ->None:
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim ** -0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
        self.qkv_proj = QKVParallelLinear(hidden_size, self.head_dim, self.total_num_heads, self.total_num_kv_heads, bias=False, quant_config=quant_config)
        self.o_proj = RowParallelLinear(self.total_num_heads * self.head_dim, hidden_size, bias=False, quant_config=quant_config)
        self.rotary_emb = get_rope(self.head_dim, rotary_dim=self.head_dim, max_position=max_position_embeddings, base=rope_theta, rope_scaling=rope_scaling)
        self.attn = RadixAttention(self.num_heads, self.head_dim, self.scaling, num_kv_heads=self.num_kv_heads, layer_id=layer_id)

    def forward(self, positions: 'torch.Tensor', hidden_states: 'torch.Tensor', forward_batch: 'ForwardBatch') ->torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v, forward_batch)
        output, _ = self.o_proj(attn_output)
        return output


class DeepseekDecoderLayer(nn.Module):

    def __init__(self, config: 'PretrainedConfig', layer_id: 'int', cache_config=None, quant_config: 'Optional[QuantizationConfig]'=None) ->None:
        super().__init__()
        self.hidden_size = config.hidden_size
        rope_theta = getattr(config, 'rope_theta', 10000)
        rope_scaling = getattr(config, 'rope_scaling', None)
        max_position_embeddings = getattr(config, 'max_position_embeddings', 8192)
        self.self_attn = DeepseekAttention(hidden_size=self.hidden_size, num_heads=config.num_attention_heads, num_kv_heads=config.num_key_value_heads, layer_id=layer_id, rope_theta=rope_theta, rope_scaling=rope_scaling, max_position_embeddings=max_position_embeddings, cache_config=cache_config, quant_config=quant_config)
        if config.n_routed_experts is not None and layer_id >= config.first_k_dense_replace and layer_id % config.moe_layer_freq == 0:
            self.mlp = DeepseekMoE(config=config, quant_config=quant_config)
        else:
            self.mlp = DeepseekMLP(hidden_size=config.hidden_size, intermediate_size=config.intermediate_size, hidden_act=config.hidden_act, quant_config=quant_config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, positions: 'torch.Tensor', hidden_states: 'torch.Tensor', forward_batch: 'ForwardBatch', residual: 'Optional[torch.Tensor]') ->torch.Tensor:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions=positions, hidden_states=hidden_states, forward_batch=forward_batch)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class DeepseekModel(nn.Module):
    fall_back_to_pt_during_load = False

    def __init__(self, config: 'PretrainedConfig', cache_config=None, quant_config: 'Optional[QuantizationConfig]'=None) ->None:
        super().__init__()
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([DeepseekDecoderLayer(config, layer_id, cache_config, quant_config=quant_config) for layer_id in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids: 'torch.Tensor', positions: 'torch.Tensor', forward_batch: 'ForwardBatch') ->torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for i in range(len(self.layers)):
            layer = self.layers[i]
            hidden_states, residual = layer(positions, hidden_states, forward_batch, residual)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class DeepseekForCausalLM(nn.Module):

    def __init__(self, config: 'PretrainedConfig', cache_config=None, quant_config: 'Optional[QuantizationConfig]'=None) ->None:
        super().__init__()
        self.config = config
        self.quant_config = quant_config
        self.model = DeepseekModel(config, cache_config, quant_config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size, quant_config=quant_config)
        self.logits_processor = LogitsProcessor(config)

    @torch.no_grad()
    def forward(self, input_ids: 'torch.Tensor', positions: 'torch.Tensor', forward_batch: 'ForwardBatch') ->torch.Tensor:
        hidden_states = self.model(input_ids, positions, forward_batch)
        return self.logits_processor(input_ids, hidden_states, self.lm_head.weight, forward_batch)

    def load_weights(self, weights: 'Iterable[Tuple[str, torch.Tensor]]'):
        stacked_params_mapping = [('qkv_proj', 'q_proj', 'q'), ('qkv_proj', 'k_proj', 'k'), ('qkv_proj', 'v_proj', 'v'), ('gate_up_proj', 'gate_proj', 0), ('gate_up_proj', 'up_proj', 1)]
        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            if 'rotary_emb.inv_freq' in name:
                continue
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                if name.endswith('.bias') and name not in params_dict:
                    continue
                if ('mlp.experts.' in name or 'mlp.shared_experts.' in name) and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                if name.endswith('.bias') and name not in params_dict:
                    continue
                if ('mlp.experts.' in name or 'mlp.shared_experts.' in name) and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, 'weight_loader', default_weight_loader)
                weight_loader(param, loaded_weight)


class DeepseekV2MLP(nn.Module):

    def __init__(self, hidden_size: 'int', intermediate_size: 'int', hidden_act: 'str', quant_config: 'Optional[QuantizationConfig]'=None, reduce_results: 'bool'=True) ->None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(hidden_size, [intermediate_size] * 2, bias=False, quant_config=quant_config)
        self.down_proj = RowParallelLinear(intermediate_size, hidden_size, bias=False, quant_config=quant_config, reduce_results=reduce_results)
        if hidden_act != 'silu':
            raise ValueError(f'Unsupported activation: {hidden_act}. Only silu is supported for now.')
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class DeepseekV2MoE(nn.Module):

    def __init__(self, config: 'PretrainedConfig', quant_config: 'Optional[QuantizationConfig]'=None):
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.routed_scaling_factor = config.routed_scaling_factor
        self.n_shared_experts = config.n_shared_experts
        self.routed_scaling_factor = config.routed_scaling_factor
        if self.tp_size > config.n_routed_experts:
            raise ValueError(f'Tensor parallel size {self.tp_size} is greater than the number of experts {config.n_routed_experts}.')
        if config.hidden_act != 'silu':
            raise ValueError(f'Unsupported activation: {config.hidden_act}. Only silu is supported for now.')
        self.experts = FusedMoE(num_experts=config.n_routed_experts, top_k=config.num_experts_per_tok, hidden_size=config.hidden_size, intermediate_size=config.moe_intermediate_size, reduce_results=False, renormalize=config.norm_topk_prob, quant_config=quant_config, use_grouped_topk=True, num_expert_group=config.n_group, topk_group=config.topk_group)
        self.gate = ReplicatedLinear(config.hidden_size, config.n_routed_experts, bias=False, quant_config=None)
        if config.n_shared_experts is not None:
            intermediate_size = config.moe_intermediate_size * config.n_shared_experts
            self.shared_experts = DeepseekV2MLP(hidden_size=config.hidden_size, intermediate_size=intermediate_size, hidden_act=config.hidden_act, quant_config=quant_config, reduce_results=False)

    def forward(self, hidden_states: 'torch.Tensor') ->torch.Tensor:
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        if self.n_shared_experts is not None:
            shared_output = self.shared_experts(hidden_states)
        router_logits, _ = self.gate(hidden_states)
        final_hidden_states = self.experts(hidden_states=hidden_states, router_logits=router_logits) * self.routed_scaling_factor
        if shared_output is not None:
            final_hidden_states = final_hidden_states + shared_output
        if self.tp_size > 1:
            final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states)
        return final_hidden_states.view(num_tokens, hidden_dim)


def yarn_get_mscale(scale: 'float'=1, mscale: 'float'=1) ->float:
    import math
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


class DeepseekV2Attention(nn.Module):

    def __init__(self, config: 'PretrainedConfig', hidden_size: 'int', num_heads: 'int', qk_nope_head_dim: 'int', qk_rope_head_dim: 'int', v_head_dim: 'int', q_lora_rank: 'int', kv_lora_rank: 'int', rope_theta: 'float'=10000, rope_scaling: 'Optional[Dict[str, Any]]'=None, max_position_embeddings: 'int'=8192, cache_config=None, quant_config: 'Optional[QuantizationConfig]'=None, layer_id=None) ->None:
        super().__init__()
        self.layer_id = layer_id
        self.hidden_size = hidden_size
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.num_heads = num_heads
        tp_size = get_tensor_model_parallel_world_size()
        assert num_heads % tp_size == 0
        self.num_local_heads = num_heads // tp_size
        self.scaling = self.qk_head_dim ** -0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
        if self.q_lora_rank is not None:
            self.q_a_proj = ReplicatedLinear(self.hidden_size, self.q_lora_rank, bias=False, quant_config=quant_config)
            self.q_a_layernorm = RMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
            self.q_b_proj = ColumnParallelLinear(q_lora_rank, self.num_heads * self.qk_head_dim, bias=False, quant_config=quant_config)
        else:
            self.q_proj = ColumnParallelLinear(self.hidden_size, self.num_heads * self.qk_head_dim, bias=False, quant_config=quant_config)
        self.kv_a_proj_with_mqa = ReplicatedLinear(self.hidden_size, self.kv_lora_rank + self.qk_rope_head_dim, bias=False, quant_config=quant_config)
        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps)
        self.kv_b_proj = ColumnParallelLinear(self.kv_lora_rank, self.num_heads * (self.qk_nope_head_dim + self.v_head_dim), bias=False, quant_config=quant_config)
        self.o_proj = RowParallelLinear(self.num_heads * self.v_head_dim, self.hidden_size, bias=False, quant_config=quant_config)
        rope_scaling['rope_type'] = 'deepseek_yarn'
        self.rotary_emb = get_rope(qk_rope_head_dim, rotary_dim=qk_rope_head_dim, max_position=max_position_embeddings, base=rope_theta, rope_scaling=rope_scaling, is_neox_style=False)
        if rope_scaling:
            mscale_all_dim = rope_scaling.get('mscale_all_dim', False)
            scaling_factor = rope_scaling['factor']
            mscale = yarn_get_mscale(scaling_factor, float(mscale_all_dim))
            self.scaling = self.scaling * mscale * mscale
        self.attn = RadixAttention(self.num_local_heads, 256, self.scaling, num_kv_heads=self.num_local_heads, layer_id=layer_id)

    def forward(self, positions: 'torch.Tensor', hidden_states: 'torch.Tensor', forward_batch: 'ForwardBatch') ->torch.Tensor:
        if self.q_lora_rank is not None:
            q = self.q_a_proj(hidden_states)[0]
            q = self.q_a_layernorm(q)
            q = self.q_b_proj(q)[0].view(-1, self.num_local_heads, self.qk_head_dim)
        else:
            q = self.q_proj(hidden_states)[0].view(-1, self.num_local_heads, self.qk_head_dim)
        _, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        latent_cache = self.kv_a_proj_with_mqa(hidden_states)[0]
        kv_a, _ = latent_cache.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        latent_cache = latent_cache.unsqueeze(1)
        kv_a = self.kv_a_layernorm(kv_a.contiguous())
        kv = self.kv_b_proj(kv_a)[0]
        kv = kv.view(-1, self.num_local_heads, self.qk_nope_head_dim + self.v_head_dim)
        k_nope, v = kv.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        k_pe = latent_cache[:, :, self.kv_lora_rank:]
        q_pe, k_pe = self.rotary_emb(positions, q_pe, k_pe)
        q[..., self.qk_nope_head_dim:] = q_pe
        k = torch.empty_like(q)
        k[..., :self.qk_nope_head_dim] = k_nope
        k[..., self.qk_nope_head_dim:] = k_pe
        q = torch.nn.functional.pad(q, [0, 256 - self.qk_head_dim], value=0).view(-1, self.num_local_heads * 256)
        k = torch.nn.functional.pad(k, [0, 256 - self.qk_head_dim], value=0).view(-1, self.num_local_heads * 256)
        v = torch.nn.functional.pad(v, [0, 256 - self.v_head_dim], value=0).view(-1, self.num_local_heads * 256)
        attn_output = self.attn(q, k, v, forward_batch)
        attn_output = attn_output.view(-1, self.num_local_heads, 256)[..., :self.v_head_dim].reshape(-1, self.num_local_heads * self.v_head_dim)
        output, _ = self.o_proj(attn_output)
        return output


def input_to_float8(x, dtype=torch.float8_e4m3fn):
    finfo = torch.finfo(dtype)
    min_val, max_val = x.aminmax()
    amax = torch.maximum(min_val.abs(), max_val.abs()).clamp(min=1e-12)
    scale = finfo.max / amax
    x_scl_sat = (x * scale).clamp(min=finfo.min, max=finfo.max)
    return x_scl_sat.contiguous(), scale.float().reciprocal()


class DeepseekV2AttentionMLA(nn.Module):

    def __init__(self, config: 'PretrainedConfig', hidden_size: 'int', num_heads: 'int', qk_nope_head_dim: 'int', qk_rope_head_dim: 'int', v_head_dim: 'int', q_lora_rank: 'int', kv_lora_rank: 'int', rope_theta: 'float'=10000, rope_scaling: 'Optional[Dict[str, Any]]'=None, max_position_embeddings: 'int'=8192, cache_config=None, quant_config: 'Optional[QuantizationConfig]'=None, layer_id=None) ->None:
        super().__init__()
        self.layer_id = layer_id
        self.hidden_size = hidden_size
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.num_heads = num_heads
        tp_size = get_tensor_model_parallel_world_size()
        assert num_heads % tp_size == 0
        self.num_local_heads = num_heads // tp_size
        self.scaling = self.qk_head_dim ** -0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
        if self.q_lora_rank is not None:
            self.q_a_proj = ReplicatedLinear(self.hidden_size, self.q_lora_rank, bias=False, quant_config=quant_config)
            self.q_a_layernorm = RMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
            self.q_b_proj = ColumnParallelLinear(q_lora_rank, self.num_heads * self.qk_head_dim, bias=False, quant_config=quant_config)
        else:
            self.q_proj = ColumnParallelLinear(self.hidden_size, self.num_heads * self.qk_head_dim, bias=False, quant_config=quant_config)
        self.kv_a_proj_with_mqa = ReplicatedLinear(self.hidden_size, self.kv_lora_rank + self.qk_rope_head_dim, bias=False, quant_config=quant_config)
        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps)
        self.kv_b_proj = ColumnParallelLinear(self.kv_lora_rank, self.num_heads * (self.qk_nope_head_dim + self.v_head_dim), bias=False, quant_config=quant_config)
        self.o_proj = RowParallelLinear(self.num_heads * self.v_head_dim, self.hidden_size, bias=False, quant_config=quant_config)
        rope_scaling['rope_type'] = 'deepseek_yarn'
        self.rotary_emb = get_rope(qk_rope_head_dim, rotary_dim=qk_rope_head_dim, max_position=max_position_embeddings, base=rope_theta, rope_scaling=rope_scaling, is_neox_style=False)
        if rope_scaling:
            mscale_all_dim = rope_scaling.get('mscale_all_dim', False)
            scaling_factor = rope_scaling['factor']
            mscale = yarn_get_mscale(scaling_factor, float(mscale_all_dim))
            self.scaling = self.scaling * mscale * mscale
        self.attn = RadixAttention(self.num_local_heads, self.kv_lora_rank + self.qk_rope_head_dim, self.scaling, num_kv_heads=1, layer_id=layer_id, v_head_dim=self.kv_lora_rank)
        self.w_kc = None
        self.w_vc = None
        self.w_scale = None

    def forward(self, positions: 'torch.Tensor', hidden_states: 'torch.Tensor', forward_batch: 'ForwardBatch') ->torch.Tensor:
        q_len = hidden_states.shape[0]
        q_input = hidden_states.new_empty(q_len, self.num_local_heads, self.kv_lora_rank + self.qk_rope_head_dim)
        if self.q_lora_rank is not None:
            q = self.q_a_proj(hidden_states)[0]
            q = self.q_a_layernorm(q)
            q = self.q_b_proj(q)[0].view(-1, self.num_local_heads, self.qk_head_dim)
        else:
            q = self.q_proj(hidden_states)[0].view(-1, self.num_local_heads, self.qk_head_dim)
        q_nope, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        if self.w_kc.dtype == torch.float8_e4m3fn:
            q_nope_val, q_nope_scale = input_to_float8(q_nope.transpose(0, 1), torch.float8_e4m3fn)
            q_nope_out = bmm_fp8(q_nope_val, self.w_kc, q_nope_scale, self.w_scale, torch.bfloat16)
        else:
            q_nope_out = torch.bmm(q_nope.transpose(0, 1), self.w_kc)
        q_input[..., :self.kv_lora_rank] = q_nope_out.transpose(0, 1)
        latent_cache = self.kv_a_proj_with_mqa(hidden_states)[0]
        v_input = latent_cache[..., :self.kv_lora_rank]
        v_input = self.kv_a_layernorm(v_input.contiguous()).unsqueeze(1)
        k_input = latent_cache.unsqueeze(1)
        k_input[..., :self.kv_lora_rank] = v_input
        k_pe = k_input[..., self.kv_lora_rank:]
        q_pe, k_pe = self.rotary_emb(positions, q_pe, k_pe)
        q_input[..., self.kv_lora_rank:] = q_pe
        k_input[..., self.kv_lora_rank:] = k_pe
        attn_output = self.attn(q_input, k_input, v_input, forward_batch)
        attn_output = attn_output.view(-1, self.num_local_heads, self.kv_lora_rank)
        if self.w_vc.dtype == torch.float8_e4m3fn:
            attn_output_val, attn_output_scale = input_to_float8(attn_output.transpose(0, 1), torch.float8_e4m3fn)
            attn_bmm_output = bmm_fp8(attn_output_val, self.w_vc, attn_output_scale, self.w_scale, torch.bfloat16)
        else:
            attn_bmm_output = torch.bmm(attn_output.transpose(0, 1), self.w_vc)
        attn_output = attn_bmm_output.transpose(0, 1).flatten(1, 2)
        output, _ = self.o_proj(attn_output)
        return output


class DeepseekV2DecoderLayer(nn.Module):

    def __init__(self, config: 'PretrainedConfig', layer_id: 'int', cache_config=None, quant_config: 'Optional[QuantizationConfig]'=None) ->None:
        super().__init__()
        self.hidden_size = config.hidden_size
        rope_theta = getattr(config, 'rope_theta', 10000)
        rope_scaling = getattr(config, 'rope_scaling', None)
        max_position_embeddings = getattr(config, 'max_position_embeddings', 8192)
        if not global_server_args_dict['disable_mla']:
            self.self_attn = DeepseekV2AttentionMLA(config=config, hidden_size=self.hidden_size, num_heads=config.num_attention_heads, qk_nope_head_dim=config.qk_nope_head_dim, qk_rope_head_dim=config.qk_rope_head_dim, v_head_dim=config.v_head_dim, q_lora_rank=config.q_lora_rank if hasattr(config, 'q_lora_rank') else None, kv_lora_rank=config.kv_lora_rank, rope_theta=rope_theta, rope_scaling=rope_scaling, max_position_embeddings=max_position_embeddings, cache_config=cache_config, quant_config=quant_config, layer_id=layer_id)
        else:
            self.self_attn = DeepseekV2Attention(config=config, hidden_size=self.hidden_size, num_heads=config.num_attention_heads, qk_nope_head_dim=config.qk_nope_head_dim, qk_rope_head_dim=config.qk_rope_head_dim, v_head_dim=config.v_head_dim, q_lora_rank=config.q_lora_rank if hasattr(config, 'q_lora_rank') else None, kv_lora_rank=config.kv_lora_rank, rope_theta=rope_theta, rope_scaling=rope_scaling, max_position_embeddings=max_position_embeddings, cache_config=cache_config, quant_config=quant_config, layer_id=layer_id)
        if config.n_routed_experts is not None and layer_id >= config.first_k_dense_replace and layer_id % config.moe_layer_freq == 0:
            self.mlp = DeepseekV2MoE(config=config, quant_config=quant_config)
        else:
            self.mlp = DeepseekV2MLP(hidden_size=config.hidden_size, intermediate_size=config.intermediate_size, hidden_act=config.hidden_act, quant_config=quant_config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, positions: 'torch.Tensor', hidden_states: 'torch.Tensor', forward_batch: 'ForwardBatch', residual: 'Optional[torch.Tensor]') ->torch.Tensor:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions=positions, hidden_states=hidden_states, forward_batch=forward_batch)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class DeepseekV2Model(nn.Module):
    fall_back_to_pt_during_load = False

    def __init__(self, config: 'PretrainedConfig', cache_config=None, quant_config: 'Optional[QuantizationConfig]'=None) ->None:
        super().__init__()
        self.padding_id = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([DeepseekV2DecoderLayer(config, layer_id, cache_config=cache_config, quant_config=quant_config) for layer_id in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids: 'torch.Tensor', positions: 'torch.Tensor', forward_batch: 'ForwardBatch') ->torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for i in range(len(self.layers)):
            layer = self.layers[i]
            hidden_states, residual = layer(positions, hidden_states, forward_batch, residual)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class DeepseekV2ForCausalLM(nn.Module):

    def __init__(self, config: 'PretrainedConfig', cache_config=None, quant_config: 'Optional[QuantizationConfig]'=None) ->None:
        super().__init__()
        self.config = config
        self.quant_config = quant_config
        self.model = DeepseekV2Model(config, cache_config, quant_config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size, quant_config=quant_config)
        self.logits_processor = LogitsProcessor(config)

    @torch.no_grad()
    def forward(self, input_ids: 'torch.Tensor', positions: 'torch.Tensor', forward_batch: 'ForwardBatch') ->torch.Tensor:
        hidden_states = self.model(input_ids, positions, forward_batch)
        return self.logits_processor(input_ids, hidden_states, self.lm_head.weight, forward_batch)

    def load_weights(self, weights: 'Iterable[Tuple[str, torch.Tensor]]'):
        stacked_params_mapping = [('gate_up_proj', 'gate_proj', 0), ('gate_up_proj', 'up_proj', 1)]
        expert_params_mapping = FusedMoE.make_expert_params_mapping(ckpt_gate_proj_name='gate_proj', ckpt_down_proj_name='down_proj', ckpt_up_proj_name='up_proj', num_experts=self.config.n_routed_experts)
        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            if 'rotary_emb.inv_freq' in name:
                continue
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                if 'mlp.experts.' in name and name not in params_dict:
                    continue
                name = name.replace(weight_name, param_name)
                if name.endswith('.bias') and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)
                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    weight_loader(param, loaded_weight, name, shard_id=shard_id, expert_id=expert_id)
                    break
                else:
                    if name.endswith('.bias') and name not in params_dict:
                        continue
                    param = params_dict[name]
                    weight_loader = getattr(param, 'weight_loader', default_weight_loader)
                    weight_loader(param, loaded_weight)
        if not global_server_args_dict['disable_mla']:
            for layer_id in range(self.config.num_hidden_layers):
                self_attn = self.model.layers[layer_id].self_attn
                w_kc, w_vc = self_attn.kv_b_proj.weight.unflatten(0, (-1, self_attn.qk_nope_head_dim + self_attn.v_head_dim)).split([self_attn.qk_nope_head_dim, self_attn.v_head_dim], dim=1)
                self_attn.w_kc = w_kc.transpose(1, 2).contiguous().transpose(1, 2)
                self_attn.w_vc = w_vc.contiguous().transpose(1, 2)
                if hasattr(self_attn.kv_b_proj, 'weight_scale'):
                    self_attn.w_scale = self_attn.kv_b_proj.weight_scale
                del self_attn.kv_b_proj


class ExaoneGatedMLP(nn.Module):

    def __init__(self, hidden_size: 'int', intermediate_size: 'int', hidden_act: 'str', quant_config: 'Optional[QuantizationConfig]'=None, prefix: 'str'='') ->None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(hidden_size, [intermediate_size] * 2, bias=False, quant_config=quant_config, prefix=f'{prefix}.gate_up_proj')
        self.c_proj = RowParallelLinear(intermediate_size, hidden_size, bias=False, quant_config=quant_config, prefix=f'{prefix}.c_proj')
        if hidden_act != 'silu':
            raise ValueError(f'Unsupported activation: {hidden_act}. Only silu is supported for now.')
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.c_proj(x)
        return x


class ExaoneAttention(nn.Module):

    def __init__(self, config, hidden_size: 'int', num_heads: 'int', num_kv_heads: 'int', layer_id: 'int'=0, rope_theta: 'float'=500000, rope_scaling: 'Optional[Dict[str, Any]]'=None, rope_is_neox_style: 'bool'=True, max_position_embeddings: 'int'=4096, quant_config: 'Optional[QuantizationConfig]'=None, prefix: 'str'='') ->None:
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = getattr(config, 'head_dim', self.hidden_size // self.total_num_heads)
        self.rotary_dim = int(self.head_dim * getattr(config, 'partial_rotary_factor', 1))
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim ** -0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
        self.qkv_proj = QKVParallelLinear(hidden_size, self.head_dim, self.total_num_heads, self.total_num_kv_heads, bias=False, quant_config=quant_config, prefix=f'{prefix}.qkv_proj')
        self.out_proj = RowParallelLinear(self.total_num_heads * self.head_dim, hidden_size, bias=False, quant_config=quant_config, prefix=f'{prefix}.out_proj')
        self.rotary_emb = get_rope(self.head_dim, rotary_dim=self.rotary_dim, max_position=max_position_embeddings, base=rope_theta, rope_scaling=rope_scaling, is_neox_style=rope_is_neox_style)
        self.attn = RadixAttention(self.num_heads, self.head_dim, self.scaling, num_kv_heads=self.num_kv_heads, layer_id=layer_id)

    def forward(self, positions: 'torch.Tensor', hidden_states: 'torch.Tensor', forward_batch: 'ForwardBatch') ->torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v, forward_batch)
        output, _ = self.out_proj(attn_output)
        return output


class ExaoneDecoderLayer(nn.Module):

    def __init__(self, config, layer_id: 'int'=0, quant_config: 'Optional[QuantizationConfig]'=None, prefix: 'str'='') ->None:
        super().__init__()
        self.hidden_size = config.hidden_size
        rope_theta = getattr(config, 'rope_theta', 500000)
        rope_scaling = getattr(config, 'rope_scaling', None)
        if rope_scaling is not None and getattr(config, 'original_max_position_embeddings', None):
            rope_scaling['original_max_position_embeddings'] = config.original_max_position_embeddings
        rope_is_neox_style = getattr(config, 'rope_is_neox_style', True)
        max_position_embeddings = getattr(config, 'max_position_embeddings', 4096)
        self.self_attn = ExaoneAttention(config=config, hidden_size=self.hidden_size, num_heads=config.num_attention_heads, num_kv_heads=config.num_key_value_heads, layer_id=layer_id, rope_theta=rope_theta, rope_scaling=rope_scaling, rope_is_neox_style=rope_is_neox_style, max_position_embeddings=max_position_embeddings, quant_config=quant_config, prefix=f'{prefix}.self_attn')
        self.mlp = ExaoneGatedMLP(hidden_size=self.hidden_size, intermediate_size=config.intermediate_size, hidden_act=config.activation_function, quant_config=quant_config, prefix=f'{prefix}.mlp')
        rms_norm_eps = config.layer_norm_epsilon
        self.ln_1 = RMSNorm(config.hidden_size, eps=rms_norm_eps)
        self.ln_2 = RMSNorm(config.hidden_size, eps=rms_norm_eps)

    def forward(self, positions: 'torch.Tensor', hidden_states: 'torch.Tensor', forward_batch: 'ForwardBatch', residual: 'Optional[torch.Tensor]') ->Tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.ln_1(hidden_states)
        else:
            hidden_states, residual = self.ln_1(hidden_states, residual)
        hidden_states = self.self_attn(positions=positions, hidden_states=hidden_states, forward_batch=forward_batch)
        hidden_states, residual = self.ln_2(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class ExaoneModel(nn.Module):

    def __init__(self, config, quant_config: 'Optional[QuantizationConfig]'=None) ->None:
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.wte = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.h = nn.ModuleList([ExaoneDecoderLayer(config, i, quant_config=quant_config, prefix=f'model.h.{i}') for i in range(config.num_hidden_layers)])
        rms_norm_eps = config.layer_norm_epsilon
        self.ln_f = RMSNorm(config.hidden_size, eps=rms_norm_eps)

    def forward(self, input_ids: 'torch.Tensor', positions: 'torch.Tensor', forward_batch: 'ForwardBatch', input_embeds: 'torch.Tensor'=None) ->torch.Tensor:
        if input_embeds is None:
            hidden_states = self.wte(input_ids)
        else:
            hidden_states = input_embeds
        residual = None
        for i in range(len(self.h)):
            layer = self.h[i]
            hidden_states, residual = layer(positions, hidden_states, forward_batch, residual)
        hidden_states, _ = self.ln_f(hidden_states, residual)
        return hidden_states


class GemmaMLP(nn.Module):

    def __init__(self, hidden_size: 'int', intermediate_size: 'int', quant_config: 'Optional[QuantizationConfig]'=None) ->None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(hidden_size, [intermediate_size] * 2, bias=False, quant_config=quant_config)
        self.down_proj = RowParallelLinear(intermediate_size, hidden_size, bias=False, quant_config=quant_config)
        self.act_fn = GeluAndMul('none')

    def forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class GemmaAttention(nn.Module):

    def __init__(self, hidden_size: 'int', num_heads: 'int', num_kv_heads: 'int', head_dim: 'int', layer_id: 'int'=0, max_position_embeddings: 'int'=8192, rope_theta: 'float'=10000, quant_config: 'Optional[QuantizationConfig]'=None) ->None:
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim ** -0.5
        self.rope_theta = rope_theta
        self.qkv_proj = QKVParallelLinear(hidden_size, self.head_dim, self.total_num_heads, self.total_num_kv_heads, bias=False, quant_config=quant_config)
        self.o_proj = RowParallelLinear(self.total_num_heads * self.head_dim, hidden_size, bias=False, quant_config=quant_config)
        self.rotary_emb = get_rope(self.head_dim, rotary_dim=self.head_dim, max_position=max_position_embeddings, base=self.rope_theta, is_neox_style=True)
        self.attn = RadixAttention(self.num_heads, self.head_dim, self.scaling, num_kv_heads=self.num_kv_heads, layer_id=layer_id)

    def forward(self, positions: 'torch.Tensor', hidden_states: 'torch.Tensor', forward_batch: 'ForwardBatch') ->torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v, forward_batch)
        output, _ = self.o_proj(attn_output)
        return output


class GemmaDecoderLayer(nn.Module):

    def __init__(self, config: 'PretrainedConfig', layer_id: 'int'=0, quant_config: 'Optional[QuantizationConfig]'=None) ->None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = GemmaAttention(hidden_size=self.hidden_size, num_heads=config.num_attention_heads, num_kv_heads=config.num_key_value_heads, head_dim=config.head_dim, layer_id=layer_id, max_position_embeddings=config.max_position_embeddings, rope_theta=config.rope_theta, quant_config=quant_config)
        self.mlp = GemmaMLP(hidden_size=self.hidden_size, intermediate_size=config.intermediate_size, quant_config=quant_config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, positions: 'torch.Tensor', hidden_states: 'torch.Tensor', forward_batch: 'ForwardBatch', residual: 'Optional[torch.Tensor]') ->Tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions=positions, hidden_states=hidden_states, forward_batch=forward_batch)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class GemmaModel(nn.Module):

    def __init__(self, config: 'PretrainedConfig', quant_config: 'Optional[QuantizationConfig]'=None) ->None:
        super().__init__()
        self.config = config
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([GemmaDecoderLayer(config, i, quant_config=quant_config) for i in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids: 'torch.Tensor', positions: 'torch.Tensor', forward_batch: 'ForwardBatch', input_embeds: 'torch.Tensor'=None) ->torch.Tensor:
        if input_embeds is None:
            hidden_states = self.embed_tokens(input_ids)
        else:
            hidden_states = input_embeds
        hidden_states *= self.config.hidden_size ** 0.5
        residual = None
        for i in range(len(self.layers)):
            layer = self.layers[i]
            hidden_states, residual = layer(positions, hidden_states, forward_batch, residual)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class GemmaForCausalLM(nn.Module):
    packed_modules_mapping = {'qkv_proj': ['q_proj', 'k_proj', 'v_proj'], 'gate_up_proj': ['gate_proj', 'up_proj']}
    supported_lora_modules = ['qkv_proj', 'o_proj', 'gate_up_proj', 'down_proj']
    embedding_modules = {}
    embedding_padding_modules = []

    def __init__(self, config: 'PretrainedConfig', quant_config: 'Optional[QuantizationConfig]'=None, lora_config: 'Optional[LoRAConfig]'=None, cache_config=None) ->None:
        del lora_config
        super().__init__()
        self.config = config
        self.quant_config = quant_config
        self.model = GemmaModel(config, quant_config=quant_config)
        self.logits_processor = LogitsProcessor(config)

    @torch.no_grad()
    def forward(self, input_ids: 'torch.Tensor', positions: 'torch.Tensor', forward_batch: 'ForwardBatch', input_embeds: 'torch.Tensor'=None) ->torch.Tensor:
        hidden_states = self.model(input_ids, positions, forward_batch, input_embeds)
        return self.logits_processor(input_ids, hidden_states, self.model.embed_tokens.weight, forward_batch)

    def load_weights(self, weights: 'Iterable[Tuple[str, torch.Tensor]]'):
        stacked_params_mapping = [('qkv_proj', 'q_proj', 'q'), ('qkv_proj', 'k_proj', 'k'), ('qkv_proj', 'v_proj', 'v'), ('gate_up_proj', 'gate_proj', 0), ('gate_up_proj', 'up_proj', 1)]
        params_dict = dict(self.named_parameters())
        loaded_params = set()
        for name, loaded_weight in weights:
            for param_name, shard_name, shard_id in stacked_params_mapping:
                if shard_name not in name:
                    continue
                name = name.replace(shard_name, param_name)
                if name.endswith('.bias') and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                if 'lm_head.weight' in name:
                    continue
                if name.endswith('.bias') and name not in params_dict:
                    continue
                if 'norm.weight' in name:
                    loaded_weight += 1.0
                param = params_dict[name]
                weight_loader = getattr(param, 'weight_loader', default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
        unloaded_params = params_dict.keys() - loaded_params
        if unloaded_params:
            raise RuntimeError(f'Some weights are not initialized from checkpoints: {unloaded_params}')


class Gemma2MLP(nn.Module):

    def __init__(self, hidden_size: 'int', intermediate_size: 'int', hidden_act: 'str', hidden_activation: 'str', quant_config: 'Optional[QuantizationConfig]'=None) ->None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(hidden_size, [intermediate_size] * 2, bias=False, quant_config=quant_config)
        self.down_proj = RowParallelLinear(intermediate_size, hidden_size, bias=False, quant_config=quant_config)
        if not hidden_act == hidden_activation == 'gelu_pytorch_tanh':
            raise ValueError('Gemma2 uses `gelu_pytorch_tanh` as the hidden activation function. Please set `hidden_act` and `hidden_activation` to `gelu_pytorch_tanh`.')
        self.act_fn = GeluAndMul()

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


def get_attention_sliding_window_size(config):
    return config.sliding_window - 1


class Gemma2Attention(nn.Module):

    def __init__(self, layer_idx: 'int', config: 'PretrainedConfig', hidden_size: 'int', num_heads: 'int', num_kv_heads: 'int', head_dim: 'int', max_position_embeddings: 'int', rope_theta: 'float', cache_config=None, quant_config: 'Optional[QuantizationConfig]'=None) ->None:
        super().__init__()
        self.layer_idx = layer_idx
        self.config = config
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = config.query_pre_attn_scalar ** -0.5
        self.rope_theta = rope_theta
        self.qkv_proj = QKVParallelLinear(hidden_size, self.head_dim, self.total_num_heads, self.total_num_kv_heads, bias=config.attention_bias, quant_config=quant_config)
        self.o_proj = RowParallelLinear(self.total_num_heads * self.head_dim, hidden_size, bias=config.attention_bias, quant_config=quant_config)
        self.rotary_emb = GemmaRotaryEmbedding(self.head_dim, self.head_dim, max_position_embeddings, base=self.rope_theta, is_neox_style=True, dtype=torch.get_default_dtype())
        use_sliding_window = layer_idx % 2 == 0 and hasattr(config, 'sliding_window')
        self.attn = RadixAttention(self.num_heads, self.head_dim, self.scaling, num_kv_heads=self.num_kv_heads, layer_id=layer_idx, logit_cap=self.config.attn_logit_softcapping, sliding_window_size=get_attention_sliding_window_size(config) if use_sliding_window else None)

    def forward(self, positions: 'torch.Tensor', hidden_states: 'torch.Tensor', forward_batch: 'ForwardBatch') ->torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v, forward_batch)
        output, _ = self.o_proj(attn_output)
        return output


class Gemma2DecoderLayer(nn.Module):

    def __init__(self, layer_idx: 'int', config: 'PretrainedConfig', cache_config=None, quant_config: 'Optional[QuantizationConfig]'=None) ->None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Gemma2Attention(layer_idx=layer_idx, config=config, hidden_size=self.hidden_size, num_heads=config.num_attention_heads, num_kv_heads=config.num_key_value_heads, head_dim=config.head_dim, max_position_embeddings=config.max_position_embeddings, rope_theta=config.rope_theta, cache_config=cache_config, quant_config=quant_config)
        self.hidden_size = config.hidden_size
        self.mlp = Gemma2MLP(hidden_size=self.hidden_size, intermediate_size=config.intermediate_size, hidden_act=config.hidden_act, hidden_activation=config.hidden_activation, quant_config=quant_config)
        self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_feedforward_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_feedforward_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, positions: 'torch.Tensor', hidden_states: 'torch.Tensor', forward_batch: 'ForwardBatch', residual: 'Optional[torch.Tensor]') ->Tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions=positions, hidden_states=hidden_states, forward_batch=forward_batch)
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states, residual = self.pre_feedforward_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        return hidden_states, residual


class Gemma2Model(nn.Module):

    def __init__(self, config: 'PretrainedConfig', cache_config=None, quant_config: 'Optional[QuantizationConfig]'=None) ->None:
        super().__init__()
        self.config = config
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([Gemma2DecoderLayer(layer_idx, config, cache_config, quant_config) for layer_idx in range(config.num_hidden_layers)])
        self.norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        normalizer = self.config.hidden_size ** 0.5
        self.register_buffer('normalizer', torch.tensor(normalizer))

    def forward(self, input_ids: 'torch.Tensor', positions: 'torch.Tensor', forward_batch: 'ForwardBatch', input_embeds: 'torch.Tensor'=None) ->torch.Tensor:
        if input_embeds is None:
            hidden_states = self.embed_tokens(input_ids)
        else:
            hidden_states = input_embeds
        normalizer = torch.tensor(self.config.hidden_size ** 0.5, dtype=torch.float16)
        hidden_states *= normalizer
        residual = None
        for i in range(len(self.layers)):
            layer = self.layers[i]
            hidden_states, residual = layer(positions, hidden_states, forward_batch, residual)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class Gemma2ForCausalLM(nn.Module):
    packed_modules_mapping = {'qkv_proj': ['q_proj', 'k_proj', 'v_proj'], 'gate_up_proj': ['gate_proj', 'up_proj']}
    supported_lora_modules = ['qkv_proj', 'o_proj', 'gate_up_proj', 'down_proj']
    embedding_modules = {}
    embedding_padding_modules = []

    def __init__(self, config: 'PretrainedConfig', cache_config=None, quant_config: 'Optional[QuantizationConfig]'=None, lora_config: 'Optional[LoRAConfig]'=None) ->None:
        del lora_config
        super().__init__()
        self.config = config
        self.quant_config = quant_config
        self.model = Gemma2Model(config, cache_config, quant_config)
        self.logits_processor = LogitsProcessor(config)

    @torch.no_grad()
    def forward(self, input_ids: 'torch.Tensor', positions: 'torch.Tensor', forward_batch: 'ForwardBatch', input_embeds: 'torch.Tensor'=None) ->torch.Tensor:
        hidden_states = self.model(input_ids, positions, forward_batch, input_embeds)
        return self.logits_processor(input_ids, hidden_states, self.model.embed_tokens.weight, forward_batch)

    def get_attention_sliding_window_size(self):
        return get_attention_sliding_window_size(self.config)

    def load_weights(self, weights: 'Iterable[Tuple[str, torch.Tensor]]'):
        stacked_params_mapping = [('qkv_proj', 'q_proj', 'q'), ('qkv_proj', 'k_proj', 'k'), ('qkv_proj', 'v_proj', 'v'), ('gate_up_proj', 'gate_proj', 0), ('gate_up_proj', 'up_proj', 1)]
        params_dict = dict(self.named_parameters())
        loaded_params: 'Set[str]' = set()
        for name, loaded_weight in weights:
            for param_name, shard_name, shard_id in stacked_params_mapping:
                if shard_name not in name:
                    continue
                name = name.replace(shard_name, param_name)
                if name.endswith('.bias') and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                if 'lm_head.weight' in name:
                    continue
                if name.endswith('.bias') and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, 'weight_loader', default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
        unloaded_params = params_dict.keys() - loaded_params
        if unloaded_params:
            raise RuntimeError(f'Some weights are not initialized from checkpoints: {unloaded_params}')


class Gemma2ForSequenceClassification(nn.Module):

    def __init__(self, config: 'Gemma2Config', quant_config: 'Optional[QuantizationConfig]'=None, cache_config=None) ->None:
        super().__init__()
        self.config = config
        self.torchao_config = None
        self.quant_config = quant_config
        self.num_labels = config.num_labels
        self.model = Gemma2Model(config, quant_config=quant_config)
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)
        self.pooler = Pooler(pooling_type=PoolingType.LAST, normalize=False)
        self.eos_token_id = config.eos_token_id

    @torch.no_grad()
    def forward(self, input_ids: 'torch.Tensor', positions: 'torch.Tensor', forward_batch: 'ForwardBatch', input_embeds: 'torch.Tensor'=None, get_embedding: 'bool'=True) ->EmbeddingPoolerOutput:
        assert get_embedding, 'Gemma2ForSequenceClassification is only used for embedding'
        hidden_states = self.model(input_ids, positions, forward_batch, input_embeds)
        last_token_hidden = self.pooler(hidden_states, forward_batch).embeddings
        scores = self.score(last_token_hidden)
        return EmbeddingPoolerOutput(scores)

    def load_weights(self, weights: 'Iterable[Tuple[str, torch.Tensor]]'):
        Gemma2ForCausalLM.load_weights(self, weights)


class GPT2Attention(nn.Module):

    def __init__(self, layer_id: 'int', config: 'GPT2Config', cache_config=None, quant_config: 'Optional[QuantizationConfig]'=None, prefix: 'str'=''):
        super().__init__()
        self.hidden_size = config.hidden_size
        total_num_heads = config.num_attention_heads
        tensor_model_parallel_world_size = get_tensor_model_parallel_world_size()
        assert total_num_heads % tensor_model_parallel_world_size == 0
        self.num_heads = total_num_heads // tensor_model_parallel_world_size
        self.head_dim = self.hidden_size // total_num_heads
        self.scale = self.head_dim ** -0.5
        self.c_attn = QKVParallelLinear(self.hidden_size, self.head_dim, total_num_heads, bias=True, quant_config=quant_config, prefix=f'{prefix}.c_attn')
        self.c_proj = RowParallelLinear(self.hidden_size, self.hidden_size, bias=True, quant_config=quant_config, prefix=f'{prefix}.c_proj')
        self.attn = RadixAttention(self.num_heads, self.head_dim, scaling=self.scale, num_kv_heads=total_num_heads, layer_id=layer_id)

    def forward(self, hidden_states: 'torch.Tensor', forward_batch: 'ForwardBatch') ->torch.Tensor:
        qkv, _ = self.c_attn(hidden_states)
        q, k, v = qkv.chunk(chunks=3, dim=-1)
        attn_output = self.attn(q, k, v, forward_batch)
        attn_output, _ = self.c_proj(attn_output)
        return attn_output


_ACTIVATION_REGISTRY = {'gelu': nn.GELU(), 'gelu_pytorch_tanh': nn.GELU(approximate='tanh')}


def get_act_fn(act_fn_name: 'str', quant_config: 'Optional[QuantizationConfig]'=None, intermediate_size: 'Optional[int]'=None, input_is_parallel: 'bool'=True, params_dtype: 'Optional[torch.dtype]'=None) ->nn.Module:
    """Get an activation function by name."""
    act_fn_name = act_fn_name.lower()
    if act_fn_name not in _ACTIVATION_REGISTRY:
        raise ValueError(f'Activation function {act_fn_name!r} is not supported.')
    act_fn = _ACTIVATION_REGISTRY[act_fn_name]
    if quant_config is not None and act_fn_name in quant_config.get_scaled_act_names():
        if intermediate_size is None:
            raise ValueError('intermediate_size must be specified for scaled activation functions.')
        return ScaledActivation(act_fn, intermediate_size, input_is_parallel, params_dtype)
    return act_fn


class GPT2MLP(nn.Module):

    def __init__(self, intermediate_size: 'int', config: 'GPT2Config', quant_config: 'Optional[QuantizationConfig]'=None, prefix: 'str'=''):
        super().__init__()
        hidden_size = config.hidden_size
        self.c_fc = ColumnParallelLinear(hidden_size, intermediate_size, bias=True, quant_config=quant_config, prefix=f'{prefix}.c_fc')
        self.c_proj = RowParallelLinear(intermediate_size, hidden_size, bias=True, quant_config=quant_config, prefix=f'{prefix}.c_proj')
        self.act = get_act_fn(config.activation_function, quant_config, intermediate_size)

    def forward(self, hidden_states: 'torch.Tensor') ->torch.Tensor:
        hidden_states, _ = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states, _ = self.c_proj(hidden_states)
        return hidden_states


class GPT2Block(nn.Module):

    def __init__(self, layer_id: 'int', config: 'GPT2Config', cache_config=None, quant_config: 'Optional[QuantizationConfig]'=None, prefix: 'str'=''):
        super().__init__()
        hidden_size = config.hidden_size
        inner_dim = config.n_inner if config.n_inner is not None else 4 * hidden_size
        self.ln_1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.attn = GPT2Attention(layer_id, config, cache_config, quant_config, prefix=f'{prefix}.attn')
        self.ln_2 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.mlp = GPT2MLP(inner_dim, config, quant_config, prefix=f'{prefix}.mlp')

    def forward(self, hidden_states: 'torch.Tensor', forward_batch: 'ForwardBatch') ->torch.Tensor:
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_output = self.attn(hidden_states=hidden_states, forward_batch=forward_batch)
        hidden_states = attn_output + residual
        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states)
        hidden_states = residual + feed_forward_hidden_states
        return hidden_states


class GPT2Model(nn.Module):

    def __init__(self, config: 'GPT2Config', cache_config=None, quant_config: 'Optional[QuantizationConfig]'=None, prefix: 'str'=''):
        super().__init__()
        self.config = config
        assert not config.add_cross_attention
        assert not config.scale_attn_by_inverse_layer_idx
        assert not config.reorder_and_upcast_attn
        self.embed_dim = config.hidden_size
        self.wte = VocabParallelEmbedding(config.vocab_size, self.embed_dim)
        self.wpe = nn.Embedding(config.max_position_embeddings, self.embed_dim)
        self.h = nn.ModuleList([GPT2Block(i, config, cache_config, quant_config) for i in range(config.num_hidden_layers)])
        self.ln_f = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)

    def forward(self, input_ids: 'torch.Tensor', position_ids: 'torch.Tensor', forward_batch: 'ForwardBatch') ->torch.Tensor:
        inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds
        for i in range(len(self.h)):
            layer = self.h[i]
            hidden_states = layer(hidden_states, forward_batch)
        hidden_states = self.ln_f(hidden_states)
        return hidden_states


class GPT2LMHeadModel(nn.Module):

    def __init__(self, config: 'GPT2Config', cache_config=None, quant_config: 'Optional[QuantizationConfig]'=None):
        super().__init__()
        self.config = config
        self.quant_config = quant_config
        self.transformer = GPT2Model(config, cache_config, quant_config, prefix='transformer')
        self.lm_head = self.transformer.wte
        self.logits_processor = LogitsProcessor(config)

    def forward(self, input_ids: 'torch.Tensor', positions: 'torch.Tensor', forward_batch: 'ForwardBatch') ->torch.Tensor:
        hidden_states = self.transformer(input_ids, positions, forward_batch)
        return self.logits_processor(input_ids, hidden_states, self.lm_head.weight, forward_batch)

    def load_weights(self, weights: 'Iterable[Tuple[str, torch.Tensor]]'):
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        for name, loaded_weight in weights:
            if 'lm_head.weight' in name:
                continue
            if '.attn.bias' in name or '.attn.masked_bias' in name:
                continue
            if not name.startswith('transformer.'):
                name = 'transformer.' + name
            param = params_dict[name]
            for conv1d_weight_name in ['c_attn', 'c_proj', 'c_fc']:
                if conv1d_weight_name not in name:
                    continue
                if not name.endswith('.weight'):
                    continue
                loaded_weight = loaded_weight.t()
            weight_loader = getattr(param, 'weight_loader', default_weight_loader)
            weight_loader(param, loaded_weight)


class GPTBigCodeAttention(nn.Module):

    def __init__(self, layer_id: 'int', config: 'GPTBigCodeConfig', cache_config=None, quant_config: 'Optional[QuantizationConfig]'=None):
        super().__init__()
        self.hidden_size = config.hidden_size
        total_num_heads = config.num_attention_heads
        self.tensor_model_parallel_world_size = get_tensor_model_parallel_world_size()
        assert total_num_heads % self.tensor_model_parallel_world_size == 0
        self.num_heads = total_num_heads // self.tensor_model_parallel_world_size
        self.head_dim = self.hidden_size // total_num_heads
        self.scale = self.head_dim ** -0.5
        self.multi_query = config.multi_query
        if self.multi_query:
            total_num_kv_heads = 1
            self.num_kv_heads = 1
        else:
            total_num_kv_heads = total_num_heads
            self.num_kv_heads = self.num_heads
        self.kv_dim = self.head_dim * self.num_kv_heads
        self.c_attn = QKVParallelLinear(self.hidden_size, self.head_dim, total_num_heads, total_num_kv_heads, bias=True, quant_config=quant_config)
        self.c_proj = RowParallelLinear(self.hidden_size, self.hidden_size, bias=True, quant_config=quant_config)
        self.attn = RadixAttention(self.num_heads, self.head_dim, scaling=self.scale, num_kv_heads=self.num_kv_heads, layer_id=layer_id)

    def forward(self, hidden_states: 'torch.Tensor', forward_batch: 'ForwardBatch') ->torch.Tensor:
        qkv, _ = self.c_attn(hidden_states)
        q, k, v = qkv.split([self.hidden_size // self.tensor_model_parallel_world_size, self.kv_dim, self.kv_dim], dim=-1)
        attn_output = self.attn(q, k, v, forward_batch)
        attn_output, _ = self.c_proj(attn_output)
        return attn_output


class GPTBigMLP(nn.Module):

    def __init__(self, intermediate_size: 'int', config: 'GPTBigCodeConfig', quant_config: 'Optional[QuantizationConfig]'=None):
        super().__init__()
        hidden_size = config.hidden_size
        self.c_fc = ColumnParallelLinear(hidden_size, intermediate_size, bias=True, quant_config=quant_config)
        self.c_proj = RowParallelLinear(intermediate_size, hidden_size, bias=True, quant_config=quant_config)
        self.act = get_act_fn(config.activation_function, quant_config, intermediate_size)

    def forward(self, hidden_states: 'torch.Tensor') ->torch.Tensor:
        hidden_states, _ = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states, _ = self.c_proj(hidden_states)
        return hidden_states


class GPTBigCodeBlock(nn.Module):

    def __init__(self, layer_id: 'int', config: 'GPTBigCodeConfig', cache_config=None, quant_config: 'Optional[QuantizationConfig]'=None):
        super().__init__()
        hidden_size = config.hidden_size
        inner_dim = config.n_inner if config.n_inner is not None else 4 * hidden_size
        self.ln_1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.attn = GPTBigCodeAttention(layer_id, config, cache_config, quant_config)
        self.ln_2 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.mlp = GPTBigMLP(inner_dim, config, quant_config)

    def forward(self, hidden_states: 'torch.Tensor', forward_batch: 'ForwardBatch') ->torch.Tensor:
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_output = self.attn(hidden_states=hidden_states, forward_batch=forward_batch)
        hidden_states = attn_output + residual
        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states)
        hidden_states = residual + feed_forward_hidden_states
        return hidden_states


class GPTBigCodeModel(nn.Module):

    def __init__(self, config: 'GPTBigCodeConfig', cache_config=None, quant_config: 'Optional[QuantizationConfig]'=None, lora_config: 'Optional[LoRAConfig]'=None):
        super().__init__()
        self.config = config
        assert not config.add_cross_attention
        self.embed_dim = config.hidden_size
        lora_vocab = lora_config.lora_extra_vocab_size * (lora_config.max_loras or 1) if lora_config else 0
        self.vocab_size = config.vocab_size + lora_vocab
        self.wte = VocabParallelEmbedding(self.vocab_size, self.embed_dim, org_num_embeddings=config.vocab_size)
        self.wpe = nn.Embedding(config.max_position_embeddings, self.embed_dim)
        self.h = nn.ModuleList([GPTBigCodeBlock(i, config, cache_config, quant_config) for i in range(config.num_hidden_layers)])
        self.ln_f = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)

    def forward(self, input_ids: 'torch.Tensor', position_ids: 'torch.Tensor', forward_batch: 'ForwardBatch') ->torch.Tensor:
        inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds
        for i in range(len(self.h)):
            layer = self.h[i]
            hidden_states = layer(hidden_states, forward_batch)
        hidden_states = self.ln_f(hidden_states)
        return hidden_states


class GPTBigCodeForCausalLM(nn.Module):
    packed_modules_mapping = {'c_attn': ['c_attn']}
    supported_lora_modules = ['c_fc', 'c_proj', 'wte', 'c_attn']
    embedding_modules = {'wte': 'input_embeddings', 'lm_head': 'output_embeddings'}
    embedding_padding_modules = []

    def __init__(self, config: 'GPTBigCodeConfig', cache_config=None, quant_config: 'Optional[QuantizationConfig]'=None, lora_config: 'Optional[LoRAConfig]'=None):
        super().__init__()
        self.config = config
        self.lora_config = lora_config
        self.quant_config = quant_config
        self.transformer = GPTBigCodeModel(config, cache_config, quant_config, lora_config)
        self.lm_head = self.transformer.wte
        self.unpadded_vocab_size = config.vocab_size
        if lora_config:
            self.unpadded_vocab_size += lora_config.lora_extra_vocab_size
        self.logits_processor = LogitsProcessor(config)

    @torch.no_grad()
    def forward(self, input_ids: 'torch.Tensor', positions: 'torch.Tensor', forward_batch: 'ForwardBatch') ->torch.Tensor:
        hidden_states = self.transformer(input_ids, positions, forward_batch)
        return self.logits_processor(input_ids, hidden_states, self.lm_head.weight, forward_batch)

    def load_weights(self, weights: 'Iterable[Tuple[str, torch.Tensor]]'):
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        for name, loaded_weight in weights:
            if 'lm_head.weight' in name:
                continue
            if '.attn.bias' in name:
                continue
            param = params_dict[name]
            weight_loader = getattr(param, 'weight_loader', default_weight_loader)
            if 'c_attn.input_scale' in name or 'c_attn.weight_scale' in name:
                weight_loader(param, loaded_weight, 'q')
                weight_loader(param, loaded_weight, 'k')
                weight_loader(param, loaded_weight, 'v')
            else:
                weight_loader(param, loaded_weight)


class Grok1MoE(nn.Module):
    """A tensor-parallel MoE implementation for Grok1 that shards each expert
    across all ranks.

    Each expert's weights are sharded across all ranks and a fused MoE
    kernel is used for the forward pass, and finally we reduce the outputs
    across ranks.
    """

    def __init__(self, num_experts: 'int', top_k: 'int', hidden_size: 'int', intermediate_size: 'int', params_dtype: 'Optional[torch.dtype]'=None, quant_config: 'Optional[QuantizationConfig]'=None, tp_size: 'Optional[int]'=None):
        super().__init__()
        self.hidden_size = hidden_size
        self.gate = ReplicatedLinear(hidden_size, num_experts, bias=False, params_dtype=params_dtype, quant_config=None)
        self.experts = FusedMoE(num_experts=num_experts, top_k=top_k, hidden_size=hidden_size, intermediate_size=intermediate_size, params_dtype=params_dtype, reduce_results=True, renormalize=False, quant_config=quant_config, tp_size=tp_size)

    def forward(self, hidden_states: 'torch.Tensor') ->torch.Tensor:
        orig_shape = hidden_states.shape
        hidden_states = hidden_states.view(-1, self.hidden_size)
        router_logits, _ = self.gate(hidden_states)
        router_logits = 30.0 * F.tanh(router_logits / 30.0)
        final_hidden_states = self.experts(hidden_states, router_logits)
        return final_hidden_states.view(orig_shape)


class Grok1Attention(nn.Module):

    def __init__(self, hidden_size: 'int', num_heads: 'int', num_kv_heads: 'int', layer_id: 'int'=0, max_position: 'int'=4096 * 32, rope_theta: 'float'=10000, logit_cap: 'float'=30, quant_config: 'Optional[QuantizationConfig]'=None) ->None:
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = 128
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim ** -0.5
        self.rope_theta = rope_theta
        self.qkv_proj = QKVParallelLinear(hidden_size, self.head_dim, self.total_num_heads, self.total_num_kv_heads, bias=False, quant_config=quant_config)
        self.o_proj = RowParallelLinear(self.total_num_heads * self.head_dim, hidden_size, bias=False, quant_config=quant_config)
        self.rotary_emb = get_rope(self.head_dim, rotary_dim=self.head_dim, max_position=max_position, base=int(self.rope_theta), is_neox_style=True)
        self.attn = RadixAttention(self.num_heads, self.head_dim, self.scaling, num_kv_heads=self.num_kv_heads, layer_id=layer_id, logit_cap=logit_cap)

    def forward(self, positions: 'torch.Tensor', hidden_states: 'torch.Tensor', forward_batch: 'ForwardBatch') ->torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v, forward_batch)
        output, _ = self.o_proj(attn_output)
        return output


class Grok1DecoderLayer(nn.Module):

    def __init__(self, config: 'PretrainedConfig', layer_id: 'int'=0, quant_config: 'Optional[QuantizationConfig]'=None) ->None:
        super().__init__()
        self.hidden_size = config.hidden_size
        rope_theta = getattr(config, 'rope_theta', 10000)
        self.self_attn = Grok1Attention(hidden_size=self.hidden_size, num_heads=config.num_attention_heads, max_position=config.max_position_embeddings, num_kv_heads=config.num_key_value_heads, layer_id=layer_id, rope_theta=rope_theta, quant_config=quant_config)
        self.block_sparse_moe = Grok1MoE(num_experts=config.num_local_experts, top_k=config.num_experts_per_tok, hidden_size=config.hidden_size, intermediate_size=config.intermediate_size, quant_config=quant_config)
        self.pre_attn_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attn_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_moe_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_moe_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, positions: 'torch.Tensor', hidden_states: 'torch.Tensor', forward_batch: 'ForwardBatch') ->torch.Tensor:
        hidden_states = self.post_attn_norm(self.self_attn(positions=positions, hidden_states=self.pre_attn_norm(hidden_states), forward_batch=forward_batch)) + hidden_states
        hidden_states = self.post_moe_norm(self.block_sparse_moe(self.pre_moe_norm(hidden_states))) + hidden_states
        return hidden_states


class Grok1Model(nn.Module):

    def __init__(self, config: 'PretrainedConfig', quant_config: 'Optional[QuantizationConfig]'=None) ->None:
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([Grok1DecoderLayer(config, i, quant_config=quant_config) for i in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids: 'torch.Tensor', positions: 'torch.Tensor', forward_batch: 'ForwardBatch', input_embeds: 'torch.Tensor'=None) ->torch.Tensor:
        if input_embeds is None:
            hidden_states = self.embed_tokens(input_ids)
            hidden_states.mul_(self.config.embedding_multiplier_scale)
        else:
            hidden_states = input_embeds
        for i in range(len(self.layers)):
            hidden_states = self.layers[i](positions, hidden_states, forward_batch)
        hidden_states = self.norm(hidden_states)
        hidden_states.mul_(self.config.output_multiplier_scale)
        return hidden_states


def _prepare_presharded_weights(self, model_name_or_path: 'str', revision: 'Optional[str]', fall_back_to_pt: 'bool') ->Tuple[str, List[str], bool]:
    if get_tensor_model_parallel_world_size() == 1:
        return old_prepare_weights(self, model_name_or_path, revision, fall_back_to_pt)
    tp_rank = get_tensor_model_parallel_rank()
    allow_patterns = [f'*-{tp_rank:03d}.bin']
    hf_folder = model_name_or_path
    hf_weights_files: 'List[str]' = []
    for pattern in allow_patterns:
        hf_weights_files += glob.glob(os.path.join(hf_folder, pattern))
    use_safetensors = False
    return hf_folder, hf_weights_files, use_safetensors


class Grok1ForCausalLM(nn.Module):

    def __init__(self, config: 'PretrainedConfig', quant_config: 'Optional[QuantizationConfig]'=None, cache_config=None) ->None:
        super().__init__()
        self.config = config
        self.quant_config = quant_config
        self.model = Grok1Model(config, quant_config=quant_config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        self.logits_processor = LogitsProcessor(config)
        setattr(DefaultModelLoader, '_prepare_weights', _prepare_presharded_weights)
        self.use_presharded_weights = True
        warnings.filterwarnings('ignore', category=FutureWarning)

    def forward(self, input_ids: 'torch.Tensor', positions: 'torch.Tensor', forward_batch: 'ForwardBatch', input_embeds: 'torch.Tensor'=None) ->torch.Tensor:
        hidden_states = self.model(input_ids, positions, forward_batch, input_embeds)
        return self.logits_processor(input_ids, hidden_states, self.lm_head.weight, forward_batch)

    def load_weights(self, weights: 'Iterable[Tuple[str, torch.Tensor]]'):
        stacked_params_mapping = [('qkv_proj', 'q_proj', 'q'), ('qkv_proj', 'k_proj', 'k'), ('qkv_proj', 'v_proj', 'v')]
        expert_params_mapping = FusedMoE.make_expert_params_mapping(ckpt_gate_proj_name='w1', ckpt_down_proj_name='w2', ckpt_up_proj_name='w3', num_experts=self.config.num_local_experts)
        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            if 'rotary_emb.inv_freq' in name:
                continue
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                if name.endswith('.bias') and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)
                    if self.use_presharded_weights:
                        extra_kwargs = {'use_presharded_weights': self.use_presharded_weights}
                    else:
                        extra_kwargs = {}
                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    weight_loader(param, loaded_weight, weight_name, shard_id=shard_id, expert_id=expert_id, **extra_kwargs)
                    break
                else:
                    if name.endswith('.bias') and name not in params_dict:
                        continue
                    if name is None:
                        continue
                    param = params_dict[name]
                    weight_loader = getattr(param, 'weight_loader', default_weight_loader)
                    weight_loader(param, loaded_weight)


class Grok1ModelForCausalLM(Grok1ForCausalLM):
    """An alias for backward-compatbility."""
    pass


class InternLM2MLP(nn.Module):

    def __init__(self, hidden_size: 'int', intermediate_size: 'int', hidden_act: 'str', quant_config: 'Optional[QuantizationConfig]'=None) ->None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(hidden_size, [intermediate_size] * 2, bias=False, quant_config=quant_config)
        self.w2 = RowParallelLinear(intermediate_size, hidden_size, bias=False, quant_config=quant_config)
        if hidden_act != 'silu':
            raise ValueError(f'Unsupported activation: {hidden_act}. Only silu is supported for now.')
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.w2(x)
        return x


class InternLM2Attention(nn.Module):

    def __init__(self, hidden_size: 'int', num_heads: 'int', num_kv_heads: 'int', rope_theta: 'float'=10000, rope_scaling: 'Optional[Dict[str, Any]]'=None, max_position_embeddings: 'int'=8192, layer_id: 'int'=0, quant_config: 'Optional[QuantizationConfig]'=None) ->None:
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim ** -0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
        self.wqkv = QKVParallelLinear(hidden_size, self.head_dim, self.total_num_heads, self.total_num_kv_heads, bias=False, quant_config=quant_config)
        self.wo = RowParallelLinear(self.total_num_heads * self.head_dim, hidden_size, bias=False, quant_config=quant_config)
        self.rotary_emb = get_rope(self.head_dim, rotary_dim=self.head_dim, max_position=max_position_embeddings, base=rope_theta, rope_scaling=rope_scaling)
        self.attn = RadixAttention(self.num_heads, self.head_dim, self.scaling, self.num_kv_heads, layer_id)

    def forward(self, positions: 'torch.Tensor', hidden_states: 'torch.Tensor', forward_batch: 'ForwardBatch') ->torch.Tensor:
        qkv, _ = self.wqkv(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v, forward_batch)
        output, _ = self.wo(attn_output)
        return output


class InternLMDecoderLayer(nn.Module):

    def __init__(self, config: 'PretrainedConfig', layer_id: 'int'=0, quant_config: 'Optional[QuantizationConfig]'=None) ->None:
        super().__init__()
        self.hidden_size = config.hidden_size
        rope_theta = getattr(config, 'rope_theta', 10000)
        rope_scaling = getattr(config, 'rope_scaling', None)
        max_position_embeddings = getattr(config, 'max_position_embeddings', 8192)
        self.attention = InternLM2Attention(hidden_size=self.hidden_size, num_heads=config.num_attention_heads, num_kv_heads=config.num_key_value_heads, rope_theta=rope_theta, rope_scaling=rope_scaling, max_position_embeddings=max_position_embeddings, layer_id=layer_id, quant_config=quant_config)
        self.feed_forward = InternLM2MLP(hidden_size=self.hidden_size, intermediate_size=config.intermediate_size, hidden_act=config.hidden_act, quant_config=quant_config)
        self.attention_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.ffn_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, positions: 'torch.Tensor', hidden_states: 'torch.Tensor', forward_batch: 'ForwardBatch', residual: 'Optional[torch.Tensor]') ->Tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.attention_norm(hidden_states)
        else:
            hidden_states, residual = self.attention_norm(hidden_states, residual)
        hidden_states = self.attention(positions=positions, hidden_states=hidden_states, forward_batch=forward_batch)
        hidden_states, residual = self.ffn_norm(hidden_states, residual)
        hidden_states = self.feed_forward(hidden_states)
        return hidden_states, residual


class InternLM2Model(nn.Module):

    def __init__(self, config: 'PretrainedConfig', quant_config: 'Optional[QuantizationConfig]'=None) ->None:
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.tok_embeddings = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([InternLMDecoderLayer(config, i, quant_config) for i in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids: 'torch.Tensor', positions: 'torch.Tensor', forward_batch: 'ForwardBatch', input_embeds: 'torch.Tensor'=None) ->torch.Tensor:
        if input_embeds is None:
            hidden_states = self.tok_embeddings(input_ids)
        else:
            hidden_states = input_embeds
        residual = None
        for i in range(len(self.layers)):
            layer = self.layers[i]
            hidden_states, residual = layer(positions, hidden_states, forward_batch, residual)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class InternLM2ForCausalLM(nn.Module):

    def __init__(self, config: 'PretrainedConfig', quant_config: 'Optional[QuantizationConfig]'=None, cache_config=None) ->None:
        super().__init__()
        self.config = config
        self.quant_config = quant_config
        self.model = InternLM2Model(config, quant_config)
        self.output = ParallelLMHead(config.vocab_size, config.hidden_size)
        self.logits_processor = LogitsProcessor(config)

    @torch.no_grad()
    def forward(self, input_ids: 'torch.Tensor', positions: 'torch.Tensor', forward_batch: 'ForwardBatch', input_embeds: 'torch.Tensor'=None) ->torch.Tensor:
        hidden_states = self.model(input_ids, positions, forward_batch, input_embeds)
        return self.logits_processor(input_ids, hidden_states, self.output.weight, forward_batch)

    def load_weights(self, weights: 'Iterable[Tuple[str, torch.Tensor]]'):
        stacked_params_mapping = [('gate_up_proj', 'w1', 0), ('gate_up_proj', 'w3', 1)]
        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            if 'rotary_emb.inv_freq' in name:
                continue
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                if name.endswith('.bias') and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                if name.endswith('.bias') and name not in params_dict:
                    continue
                param = params_dict[name]
                if 'wqkv' in name:
                    config = self.config
                    kv_groups = config.num_attention_heads // config.num_key_value_heads
                    head_dim = config.hidden_size // config.num_attention_heads
                    loaded_weight = loaded_weight.view(-1, 2 + kv_groups, head_dim, loaded_weight.shape[-1])
                    wq, wk, wv = torch.split(loaded_weight, [kv_groups, 1, 1], dim=1)
                    wq = wq.reshape(-1, wq.shape[-1])
                    wk = wk.reshape(-1, wk.shape[-1])
                    wv = wv.reshape(-1, wv.shape[-1])
                    weight_loader = param.weight_loader
                    weight_loader(param, wq, 'q')
                    weight_loader(param, wk, 'k')
                    weight_loader(param, wv, 'v')
                else:
                    weight_loader = getattr(param, 'weight_loader', default_weight_loader)
                    weight_loader(param, loaded_weight)


def gate_up_proj_weight_loader(self, param: 'Parameter', loaded_weight: 'torch.Tensor', loaded_shard_id: 'Optional[int]'=None):
    if loaded_shard_id is None:
        shard_offsets: 'List[Tuple[int, int, int]]' = []
        for i, output_size in enumerate(self.output_sizes):
            shard_offsets.append((i, current_shard_offset, output_size))
            current_shard_offset += output_size
        for shard_id, shard_offset, shard_size in shard_offsets:
            loaded_weight_shard = loaded_weight.narrow(output_dim, shard_offset, shard_size)
            self.weight_loader(param, loaded_weight_shard, shard_id)
    else:
        assert loaded_shard_id < len(self.output_sizes)
        param_data = param.data
        shard_size = loaded_weight.shape[0]
        shard_offset = loaded_shard_id * shard_size
        param_data = param_data.narrow(0, shard_offset, shard_size)
        assert param_data.shape == loaded_weight.shape
        param_data.copy_(loaded_weight)
    return


class LlamaMLP(nn.Module):

    def __init__(self, hidden_size: 'int', intermediate_size: 'int', hidden_act: 'str', quant_config: 'Optional[QuantizationConfig]'=None, prefix: 'str'='') ->None:
        super().__init__()
        self.gate_up_proj = torch.nn.Linear(hidden_size, intermediate_size * 2, bias=False)
        self.gate_up_proj.output_sizes = [intermediate_size] * 2
        self.gate_up_proj.weight_loader = types.MethodType(gate_up_proj_weight_loader, self.gate_up_proj)
        self.gate_up_proj.weight.weight_loader = self.gate_up_proj.weight_loader
        self.down_proj = torch.nn.Linear(intermediate_size, hidden_size, bias=False)
        if hidden_act != 'silu':
            raise ValueError(f'Unsupported activation: {hidden_act}. Only silu is supported for now.')
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x = self.down_proj(x)
        return x


def _get_shard_offset_mapping(self, loaded_shard_id: 'str'):
    shard_offset_mapping = {'q': 0, 'k': self.num_heads * self.head_size, 'v': (self.num_heads + self.num_kv_heads) * self.head_size, 'total': (self.num_heads + 2 * self.num_kv_heads) * self.head_size}
    return shard_offset_mapping.get(loaded_shard_id)


def _get_shard_size_mapping(self, loaded_shard_id: 'str'):
    shard_size_mapping = {'q': self.num_heads * self.head_size, 'k': self.num_kv_heads * self.head_size, 'v': self.num_kv_heads * self.head_size}
    return shard_size_mapping.get(loaded_shard_id)


def qkv_proj_weight_loader(self, param: 'Parameter', loaded_weight: 'torch.Tensor', loaded_shard_id: 'Optional[str]'=None):
    if loaded_shard_id is None:
        shard_offsets = [('q', 0, self.total_num_heads * self.head_size), ('k', self.total_num_heads * self.head_size, self.total_num_kv_heads * self.head_size), ('v', (self.total_num_heads + self.total_num_kv_heads) * self.head_size, self.total_num_kv_heads * self.head_size)]
        for shard_id, shard_offset, shard_size in shard_offsets:
            loaded_weight_shard = loaded_weight.narrow(param.output_dim, shard_offset, shard_size)
            self.weight_loader(param, loaded_weight_shard, shard_id)
    else:
        shard_offset = self._get_shard_offset_mapping(loaded_shard_id)
        shard_size = self._get_shard_size_mapping(loaded_shard_id)
        param_data = param.data
        param_data = param_data.narrow(0, shard_offset, shard_size)
        assert param_data.shape == loaded_weight.shape
        param_data.copy_(loaded_weight)
    return


class LlamaAttention(nn.Module):

    def __init__(self, config: 'LlamaConfig', hidden_size: 'int', num_heads: 'int', num_kv_heads: 'int', layer_id: 'int'=0, rope_theta: 'float'=10000, rope_scaling: 'Optional[Dict[str, Any]]'=None, rope_is_neox_style: 'bool'=True, max_position_embeddings: 'int'=8192, quant_config: 'Optional[QuantizationConfig]'=None, prefix: 'str'='') ->None:
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = getattr(config, 'head_dim', self.hidden_size // self.total_num_heads)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim ** -0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
        self.qkv_proj = torch.nn.Linear(hidden_size, (self.total_num_heads + 2 * self.total_num_kv_heads) * self.head_dim, bias=False)
        self.qkv_proj.total_num_heads = self.total_num_heads
        self.qkv_proj.head_size = self.head_dim
        self.qkv_proj.total_num_kv_heads = self.total_num_kv_heads
        self.qkv_proj.num_heads = self.total_num_heads
        self.qkv_proj.num_kv_heads = self.total_num_kv_heads
        self.qkv_proj.weight_loader = types.MethodType(qkv_proj_weight_loader, self.qkv_proj)
        self.qkv_proj._get_shard_offset_mapping = types.MethodType(_get_shard_offset_mapping, self.qkv_proj)
        self.qkv_proj._get_shard_size_mapping = types.MethodType(_get_shard_size_mapping, self.qkv_proj)
        self.qkv_proj.weight.weight_loader = self.qkv_proj.weight_loader
        self.qkv_proj.weight.output_dim = 0
        self.o_proj = torch.nn.Linear(self.total_num_heads * self.head_dim, hidden_size, bias=False)
        self.rotary_emb = get_rope(self.head_dim, rotary_dim=self.head_dim, max_position=max_position_embeddings, base=rope_theta, rope_scaling=rope_scaling, is_neox_style=rope_is_neox_style)
        self.attn = RadixAttention(self.num_heads, self.head_dim, self.scaling, num_kv_heads=self.num_kv_heads, layer_id=layer_id)

    def forward(self, positions: 'torch.Tensor', hidden_states: 'torch.Tensor', forward_batch: 'ForwardBatch') ->torch.Tensor:
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v, forward_batch)
        output = self.o_proj(attn_output)
        return output


class LlamaDecoderLayer(nn.Module):

    def __init__(self, config: 'LlamaConfig', layer_id: 'int'=0, quant_config: 'Optional[QuantizationConfig]'=None, prefix: 'str'='') ->None:
        super().__init__()
        self.hidden_size = config.hidden_size
        rope_theta = getattr(config, 'rope_theta', 10000)
        rope_scaling = getattr(config, 'rope_scaling', None)
        if rope_scaling is not None and getattr(config, 'original_max_position_embeddings', None):
            rope_scaling['original_max_position_embeddings'] = config.original_max_position_embeddings
        rope_is_neox_style = getattr(config, 'rope_is_neox_style', True)
        max_position_embeddings = getattr(config, 'max_position_embeddings', 8192)
        self.self_attn = LlamaAttention(config=config, hidden_size=self.hidden_size, num_heads=config.num_attention_heads, num_kv_heads=config.num_key_value_heads, layer_id=layer_id, rope_theta=rope_theta, rope_scaling=rope_scaling, rope_is_neox_style=rope_is_neox_style, max_position_embeddings=max_position_embeddings, quant_config=quant_config, prefix=f'{prefix}.self_attn')
        self.mlp = LlamaMLP(hidden_size=self.hidden_size, intermediate_size=config.intermediate_size, hidden_act=config.hidden_act, quant_config=quant_config, prefix=f'{prefix}.mlp')
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, positions: 'torch.Tensor', hidden_states: 'torch.Tensor', forward_batch: 'ForwardBatch', residual: 'Optional[torch.Tensor]') ->Tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions=positions, hidden_states=hidden_states, forward_batch=forward_batch)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class LlamaModel(nn.Module):

    def __init__(self, config: 'LlamaConfig', quant_config: 'Optional[QuantizationConfig]'=None) ->None:
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([LlamaDecoderLayer(config, i, quant_config=quant_config, prefix=f'model.layers.{i}') for i in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids: 'torch.Tensor', positions: 'torch.Tensor', forward_batch: 'ForwardBatch', input_embeds: 'torch.Tensor'=None) ->torch.Tensor:
        if input_embeds is None:
            hidden_states = self.embed_tokens(input_ids)
        else:
            hidden_states = input_embeds
        residual = None
        for i in range(len(self.layers)):
            layer = self.layers[i]
            hidden_states, residual = layer(positions, hidden_states, forward_batch, residual)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


def torchao_quantize_param_data(param: 'torch.Tensor', torchao_config: 'str'):
    """Quantize a Tensor with torchao quantization specified by torchao_config

    Args:
       `param`: weight parameter of the linear module
       `torchao_config`: type of quantization and their arguments we want to use to
        quantize the Tensor, e.g. int4wo-128 means int4 weight only quantization with group_size
        128
    """
    dummy_linear = torch.nn.Linear(param.shape[1], param.shape[0], bias=False)
    dummy_linear.weight = param
    if 'int8wo' in torchao_config:
        quantize_(dummy_linear, int8_weight_only())
    elif 'int8dq' in torchao_config:
        quantize_(dummy_linear, int8_dynamic_activation_int8_weight())
    elif 'int4wo' in torchao_config:
        group_size = int(torchao_config.split('-')[-1])
        assert group_size in [32, 64, 128, 256], f'int4wo groupsize needs to be one of [32, 64, 128, 256] but got {group_size}'
        quantize_(dummy_linear, int4_weight_only(group_size=group_size))
    elif 'fp8wo' in torchao_config:
        quantize_(dummy_linear, float8_weight_only())
    elif 'fp8dq' in torchao_config:
        granularity = torchao_config.split('-')[-1]
        GRANULARITY_MAP = {'per_row': PerRow(), 'per_tensor': PerTensor()}
        assert granularity in GRANULARITY_MAP, f'Supported granularity are: {GRANULARITY_MAP.keys()}, got {granularity}'
        quantize_(dummy_linear, float8_dynamic_activation_float8_weight(granularity=GRANULARITY_MAP[granularity]))
    return dummy_linear.weight


def apply_torchao_config_(self: 'torch.nn.Module', params_dict: 'Dict[str, torch.Tensor]', param_suffixes: 'Set[str]') ->None:
    """A util function used for quantizing the weight parameters after they are loaded if
       self.torchao_config is specified

    Args:
      `self`: the model we want to quantize
      `params_dict`: dictionary mapping from param_name to the parameter Tensor
      `param_suffixes`: a set of suffixes, we'll quantize the Tensor matching these suffixes

    Returns:
       None, the `params_dict` is modified inplace and the weights of `self` model are quantized
    """
    if self.torchao_config:
        for param_suffix in param_suffixes:
            for name in params_dict:
                param = params_dict[name]
                if param_suffix in name and param.ndim == 2:
                    params_dict[name] = torchao_quantize_param_data(param, self.torchao_config)
        self.load_state_dict(params_dict, assign=True)


class LlamaForClassification(nn.Module):

    def __init__(self, config: 'LlamaConfig', quant_config: 'Optional[QuantizationConfig]'=None, cache_config=None) ->None:
        super().__init__()
        self.config = config
        self.torchao_config = None
        self.quant_config = quant_config
        self.model = LlamaModel(config, quant_config=quant_config)
        self.classification_head = nn.Linear(config.hidden_size, config.classification_out_size, bias=False)
        self.eos_token_id = config.eos_token_id

    @torch.no_grad()
    def forward(self, input_ids: 'torch.Tensor', positions: 'torch.Tensor', forward_batch: 'ForwardBatch', input_embeds: 'torch.Tensor'=None) ->torch.Tensor:
        hidden_states = self.model(input_ids, positions, forward_batch, input_embeds)
        is_eos_token = input_ids == self.eos_token_id
        hidden_states = hidden_states[is_eos_token]
        scores = self.classification_head(hidden_states)
        if scores.shape[0] != forward_batch.batch_size:
            None
            scores = torch.ones((forward_batch.batch_size, self.config.classification_out_size))
        logits_output = LogitsProcessorOutput(next_token_logits=scores, next_token_logprobs=scores, normalized_prompt_logprobs=scores, input_token_logprobs=torch.ones_like(input_ids), input_top_logprobs=None, output_top_logprobs=None)
        return logits_output

    def load_weights(self, weights: 'Iterable[Tuple[str, torch.Tensor]]'):
        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            if 'classification_head' in name:
                param = params_dict[name]
                weight_loader = getattr(param, 'weight_loader', default_weight_loader)
                weight_loader(param, loaded_weight)
            elif 'lm_head' in name:
                continue
            else:
                LlamaForCausalLM.load_weights(self, [(name, loaded_weight)])


class LlamaEmbeddingModel(nn.Module):

    def __init__(self, config: 'LlamaConfig', quant_config=None, cache_config=None) ->None:
        super().__init__()
        self.model = LlamaModel(config, quant_config=quant_config)
        self.pooler = Pooler(pooling_type=PoolingType.LAST, normalize=True)

    @torch.no_grad()
    def forward(self, input_ids: 'torch.Tensor', positions: 'torch.Tensor', forward_batch: 'ForwardBatch', input_embeds: 'torch.Tensor'=None, get_embedding: 'bool'=True) ->EmbeddingPoolerOutput:
        assert get_embedding, 'LlamaEmbeddingModel / MistralModel is only used for embedding'
        hidden_states = self.model(input_ids, positions, forward_batch, input_embeds)
        return self.pooler(hidden_states, forward_batch)

    def load_weights(self, weights: 'Iterable[Tuple[str, torch.Tensor]]'):
        stacked_params_mapping = [('qkv_proj', 'q_proj', 'q'), ('qkv_proj', 'k_proj', 'k'), ('qkv_proj', 'v_proj', 'v'), ('gate_up_proj', 'gate_proj', 0), ('gate_up_proj', 'up_proj', 1)]
        params_dict = dict(self.model.named_parameters())
        for name, loaded_weight in weights:
            if 'rotary_emb.inv_freq' in name or 'projector' in name:
                return
            if 'rotary_emb.cos_cached' in name or 'rotary_emb.sin_cached' in name:
                return
            if name.startswith('model.vision_tower') and name not in params_dict:
                return
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                if name.endswith('.bias') and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                if name.endswith('.bias') and name not in params_dict:
                    return
                param = params_dict[name]
                weight_loader = getattr(param, 'weight_loader', default_weight_loader)
                weight_loader(param, loaded_weight)


class MistralModel(LlamaEmbeddingModel):
    pass


class LlamaForSequenceClassification(nn.Module):

    def __init__(self, config: 'LlamaConfig', quant_config: 'Optional[QuantizationConfig]'=None, cache_config=None) ->None:
        super().__init__()
        self.config = config
        self.torchao_config = None
        self.quant_config = quant_config
        self.num_labels = config.num_labels
        self.model = LlamaModel(config, quant_config=quant_config)
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)
        self.pooler = Pooler(pooling_type=PoolingType.LAST, normalize=False)
        self.eos_token_id = config.eos_token_id

    @torch.no_grad()
    def forward(self, input_ids: 'torch.Tensor', positions: 'torch.Tensor', forward_batch: 'ForwardBatch', input_embeds: 'torch.Tensor'=None, get_embedding: 'bool'=True) ->EmbeddingPoolerOutput:
        assert get_embedding, 'LlamaForSequenceClassification is only used for embedding'
        hidden_states = self.model(input_ids, positions, forward_batch, input_embeds)
        last_token_hidden = self.pooler(hidden_states, forward_batch).embeddings
        scores = self.score(last_token_hidden)
        return EmbeddingPoolerOutput(scores)

    def load_weights(self, weights: 'Iterable[Tuple[str, torch.Tensor]]'):
        return LlamaForCausalLM.load_weights(self, weights)


class LlamaForSequenceClassificationWithNormal_Weights(LlamaForSequenceClassification):


    class Weights(torch.nn.Module):

        def __init__(self, hidden_size, num_label):
            super().__init__()
            self.fc = torch.nn.Sequential(torch.nn.Linear(hidden_size, hidden_size, dtype=torch.float16), torch.nn.SELU(), torch.nn.Linear(hidden_size, hidden_size, dtype=torch.float16), torch.nn.SELU(), torch.nn.Linear(hidden_size, num_label // 2, dtype=torch.float16))

        def forward(self, x):
            return self.fc(x)

    def __init__(self, config: 'LlamaConfig', quant_config: 'Optional[QuantizationConfig]'=None, cache_config=None) ->None:
        super().__init__(config, quant_config, cache_config)
        self.weights = self.Weights(config.hidden_size, self.num_labels)

    @torch.no_grad()
    def forward(self, input_ids: 'torch.Tensor', positions: 'torch.Tensor', forward_batch: 'ForwardBatch', input_embeds: 'torch.Tensor'=None, get_embedding: 'bool'=True) ->EmbeddingPoolerOutput:
        assert get_embedding, 'LlamaForSequenceClassification is only used for embedding'
        hidden_states = self.model(input_ids, positions, forward_batch, input_embeds)
        logits = self.score(hidden_states)
        weights = self.weights(hidden_states)
        pooled_logits = self.pooler(logits, forward_batch).embeddings
        pooled_weights = self.pooler(weights, forward_batch).embeddings
        rews = pooled_logits.view(-1, self.num_labels // 2, 2)[:, :, 0].view(-1, self.num_labels // 2)
        scores = (rews * pooled_weights).sum(dim=-1).view(-1, 1)
        return EmbeddingPoolerOutput(scores)

    def load_weights(self, weights: 'Iterable[Tuple[str, torch.Tensor]]'):
        return super().load_weights(weights)


def select_best_resolution(original_size, possible_resolutions):
    """
    Selects the best resolution from a list of possible resolutions based on the original size.

    Args:
        original_size (tuple): The original size of the image in the format (width, height).
        possible_resolutions (list): A list of possible resolutions in the format [(width1, height1), (width2, height2), ...].

    Returns:
        tuple: The best fit resolution in the format (width, height).
    """
    original_width, original_height = original_size
    best_fit = None
    max_effective_resolution = 0
    min_wasted_resolution = float('inf')
    for width, height in possible_resolutions:
        scale = min(width / original_width, height / original_height)
        downscaled_width, downscaled_height = int(original_width * scale), int(original_height * scale)
        effective_resolution = min(downscaled_width * downscaled_height, original_width * original_height)
        wasted_resolution = width * height - effective_resolution
        if effective_resolution > max_effective_resolution or effective_resolution == max_effective_resolution and wasted_resolution < min_wasted_resolution:
            max_effective_resolution = effective_resolution
            min_wasted_resolution = wasted_resolution
            best_fit = width, height
    return best_fit


def get_anyres_image_grid_shape(image_size, grid_pinpoints, patch_size):
    """
    Calculate the shape of the image patch grid after the preprocessing for images of any resolution.

    Args:
        image_size (tuple): The size of the input image in the format (width, height).
        grid_pinpoints (str): A string representation of a list of possible resolutions.
        patch_size (int): The size of each image patch.

    Returns:
        tuple: The shape of the image patch grid in the format (width, height).
    """
    if isinstance(grid_pinpoints, str) and 'x' in grid_pinpoints:
        assert patch_size in [224, 336, 384, 448, 512], 'patch_size should be in [224, 336, 384, 448, 512]'
        matches = re.findall('\\((\\d+)x(\\d+)\\)', grid_pinpoints)
        range_start = tuple(map(int, matches[0]))
        range_end = tuple(map(int, matches[-1]))
        grid_pinpoints = [(i, j) for i in range(range_start[0], range_end[0] + 1) for j in range(range_start[1], range_end[1] + 1)]
        grid_pinpoints = [[(dim * patch_size) for dim in pair] for pair in grid_pinpoints]
    if type(grid_pinpoints) is list:
        possible_resolutions = grid_pinpoints
    else:
        possible_resolutions = ast.literal_eval(grid_pinpoints)
    width, height = select_best_resolution(image_size, possible_resolutions)
    return width // patch_size, height // patch_size


def unpad_image(tensor, original_size):
    """
    Unpads a PyTorch tensor of a padded and resized image.

    Args:
    tensor (torch.Tensor): The image tensor, assumed to be in CxHxW format.
    original_size (tuple): The original size of the image (height, width).

    Returns:
    torch.Tensor: The unpadded image tensor.
    """
    original_width, original_height = original_size
    current_height, current_width = tensor.shape[1:]
    original_aspect_ratio = original_width / original_height
    current_aspect_ratio = current_width / current_height
    if original_aspect_ratio > current_aspect_ratio:
        scale_factor = current_width / original_width
        new_height = int(original_height * scale_factor)
        padding = (current_height - new_height) // 2
        unpadded_tensor = tensor[:, padding:current_height - padding, :]
    else:
        scale_factor = current_height / original_height
        new_width = int(original_width * scale_factor)
        padding = (current_width - new_width) // 2
        unpadded_tensor = tensor[:, :, padding:current_width - padding]
    return unpadded_tensor


def unpad_image_shape(current_height, current_width, original_size):
    """
    Unpads a PyTorch tensor of a padded and resized image
    and returns the new shape.
    """
    original_width, original_height = original_size
    original_aspect_ratio = original_width / original_height
    current_aspect_ratio = current_width / current_height
    if original_aspect_ratio > current_aspect_ratio:
        scale_factor = current_width / original_width
        new_height = int(original_height * scale_factor)
        padding = (current_height - new_height) // 2
        new_shape = current_height - 2 * padding, current_width
    else:
        scale_factor = current_height / original_height
        new_width = int(original_width * scale_factor)
        padding = (current_width - new_width) // 2
        new_shape = current_height, current_width - 2 * padding
    return new_shape


class LlavaBaseForCausalLM(nn.Module):

    def pad_input_ids(self, input_ids: 'List[int]', image_inputs: 'ImageInputs'):
        image_sizes, pad_values = image_inputs.image_sizes, image_inputs.pad_values
        image_aspect_ratio = 'anyres' if len(image_sizes) == 1 else 'pad'
        offset_list = []
        for image_s in image_sizes:
            if len(image_sizes) > 16:
                new_image_feature_len = math.ceil(self.image_size / self.patch_size / 2) ** 2
            else:
                new_image_feature_len = self.image_feature_len
            height = width = self.num_patches_per_side
            if 'anyres' in image_aspect_ratio:
                num_patch_width, num_patch_height = get_anyres_image_grid_shape(image_s, self.image_grid_pinpoints, self.vision_tower.config.image_size)
                h = num_patch_height * height
                w = num_patch_width * width
                new_h, new_w = unpad_image_shape(h, w, image_s)
                if 'anyres_max' in self.config.image_aspect_ratio:
                    matched_anyres_max_num_patches = re.match('anyres_max_(\\d+)', self.config.image_aspect_ratio)
                    if matched_anyres_max_num_patches:
                        max_num_patches = int(matched_anyres_max_num_patches.group(1))
                    times = math.sqrt(new_h * new_w / (max_num_patches * self.image_feature_len))
                    if times > 1.1:
                        new_h = int(new_h // times)
                        new_w = int(new_w // times)
                new_image_feature_len += new_h * (new_w + 1)
            pad_ids = pad_values * ((new_image_feature_len + len(pad_values)) // len(pad_values))
            try:
                offset = input_ids.index(self.config.image_token_index)
            except ValueError:
                offset = 0
            input_ids = input_ids[:offset] + pad_ids[:new_image_feature_len] + input_ids[offset + 1:]
            offset_list.append(offset)
        image_inputs.image_offsets = offset_list
        return input_ids

    def encode_images(self, pixel_values: 'torch.Tensor') ->torch.Tensor:
        image_outputs = self.vision_tower(pixel_values, output_hidden_states=True)
        selected_image_feature = image_outputs.hidden_states[self.vision_feature_layer]
        if self.vision_feature_select_strategy in ['default', 'patch']:
            selected_image_feature = selected_image_feature[:, 1:]
        elif self.vision_feature_select_strategy == 'full':
            selected_image_feature = selected_image_feature
        else:
            raise ValueError(f'Unexpected select feature strategy: {self.config.vision_feature_select_strategy}')
        image_features = self.multi_modal_projector(selected_image_feature)
        return image_features

    @torch.no_grad()
    def forward(self, input_ids: 'torch.LongTensor', positions: 'torch.Tensor', forward_batch: 'ForwardBatch') ->torch.Tensor:
        image_inputs = forward_batch.image_inputs
        if forward_batch.forward_mode.is_extend():
            bs = forward_batch.batch_size
            modalities_list = []
            max_image_offset = []
            for im in image_inputs:
                if im and im.modalities is not None:
                    modalities_list.extend(im.modalities)
                if im and im.image_offsets is not None:
                    max_image_offset.append(max(im.image_offsets))
                else:
                    max_image_offset.append(-1)
            input_embeds = self.language_model.model.embed_tokens(input_ids)
            start_positions = positions[forward_batch.extend_start_loc].cpu().numpy()
            need_vision = start_positions <= np.array(max_image_offset)
            if need_vision.any():
                pixel_values = [image_inputs[i].pixel_values for i in range(bs) if need_vision[i]]
                image_sizes = [image_inputs[i].image_sizes for i in range(bs) if need_vision[i]]
                if pixel_values[0].ndim == 4:
                    np.concatenate(pixel_values, axis=0)
                    concat_images = torch.tensor(np.concatenate(pixel_values, axis=0), device=self.vision_tower.device)
                    image_features = self.encode_images(concat_images)
                    split_sizes = [image.shape[0] for image in pixel_values]
                    image_features = torch.split(image_features, split_sizes, dim=0)
                else:
                    pixel_values = torch.tensor(np.array(pixel_values), device=self.vision_tower.device)
                    image_features = self.encode_images(pixel_values)
                if self.mm_patch_merge_type.startswith('spatial'):
                    new_image_features = []
                    height = width = self.num_patches_per_side
                    for image_idx, image_feature in enumerate(image_features):
                        if modalities_list[image_idx] == 'image':
                            image_aspect_ratio = self.config.image_aspect_ratio
                        elif modalities_list[image_idx] == 'multi-images' or modalities_list[image_idx] == 'video':
                            image_aspect_ratio = 'pad'
                        if image_feature.shape[0] > 1 and 'anyres' in image_aspect_ratio and modalities_list[image_idx] == 'image':
                            base_image_feature = image_feature[0]
                            image_feature = image_feature[1:]
                            assert height * width == base_image_feature.shape[0]
                            if 'anyres_max' in image_aspect_ratio:
                                matched_anyres_max_num_patches = re.match('anyres_max_(\\d+)', image_aspect_ratio)
                                if matched_anyres_max_num_patches:
                                    max_num_patches = int(matched_anyres_max_num_patches.group(1))
                            if image_aspect_ratio == 'anyres' or 'anyres_max' in image_aspect_ratio:
                                vision_tower_image_size = self.image_size
                                try:
                                    num_patch_width, num_patch_height = get_anyres_image_grid_shape(image_sizes[image_idx][0], self.config.image_grid_pinpoints, vision_tower_image_size)
                                except Exception as e:
                                    None
                                    num_patch_width, num_patch_height = 2, 2
                                image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
                            else:
                                image_feature = image_feature.view(2, 2, height, width, -1)
                            if 'unpad' in self.mm_patch_merge_type:
                                unit = image_feature.shape[2]
                                image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                                image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                                image_feature = unpad_image(image_feature, image_sizes[image_idx][0])
                                if 'anyres_max' in image_aspect_ratio and matched_anyres_max_num_patches:
                                    c, h, w = image_feature.shape
                                    times = math.sqrt(h * w / (max_num_patches * unit ** 2))
                                    if times > 1.1:
                                        image_feature = image_feature[None]
                                        image_feature = nn.functional.interpolate(image_feature, [int(h // times), int(w // times)], mode='bilinear')[0]
                                image_feature = torch.cat((image_feature, self.language_model.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1)), dim=-1)
                                image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                            else:
                                image_feature = image_feature.permute(0, 2, 1, 3, 4).contiguous()
                                image_feature = image_feature.flatten(0, 3)
                            image_feature = torch.cat((base_image_feature, image_feature), dim=0)
                            image_feature = image_feature.unsqueeze(0)
                        else:
                            if modalities_list[image_idx] == 'video':
                                num_of_frames = image_feature.shape[0]
                                image_feature = image_feature.view(num_of_frames, height, width, -1)
                                image_feature = image_feature.permute(0, 3, 1, 2).contiguous()
                                height, weight = image_feature.shape[2:]
                                scaled_shape = [math.ceil(height / 2), math.ceil(weight / 2)]
                                image_feature = nn.functional.interpolate(image_feature, size=scaled_shape, mode='bilinear')
                                image_feature = image_feature.flatten(2).transpose(1, 2).contiguous()
                            if 'unpad' in self.mm_patch_merge_type:
                                image_feature = torch.cat((image_feature, self.language_model.model.image_newline[None, None].expand(image_feature.shape[0], 1, image_feature.shape[-1])), dim=1)
                        new_image_features.append(image_feature)
                    image_features = new_image_features
                extend_start_loc_cpu = forward_batch.extend_start_loc.cpu().numpy()
                prefix_lens_cpu = forward_batch.extend_prefix_lens.cpu().numpy()
                pt = 0
                for i in range(bs):
                    if not need_vision[i]:
                        continue
                    start_idx = extend_start_loc_cpu[i]
                    prefix_len = prefix_lens_cpu[i]
                    for j, image_offset in enumerate(image_inputs[i].image_offsets):
                        if image_offset < prefix_len:
                            continue
                        tmp_image_feature = image_features[pt][j]
                        pad_len = tmp_image_feature.shape[0]
                        left_idx = start_idx + (image_offset - prefix_len)
                        right_idx = start_idx + (image_offset - prefix_len) + pad_len
                        try:
                            input_embeds[left_idx:right_idx] = tmp_image_feature
                        except RuntimeError as e:
                            None
                            None
                            None
                    pt += 1
            return self.language_model(input_ids, positions, forward_batch, input_embeds=input_embeds)
        elif forward_batch.forward_mode.is_decode():
            return self.language_model(input_ids, positions, forward_batch)

    def load_weights(self, weights: 'Iterable[Tuple[str, torch.Tensor]]'):
        vision_path = self.config.mm_vision_tower
        if 'clip' in vision_path:
            self.vision_tower = CLIPVisionModel.from_pretrained(vision_path, torch_dtype=torch.float16)
        elif 'siglip' in vision_path:
            self.vision_tower = SiglipVisionModel.from_pretrained(vision_path, torch_dtype=torch.float16)
            self.config.mm_vision_select_feature = 'full'
        self.vision_tower.eval()
        self.vision_feature_layer = self.config.mm_vision_select_layer
        self.vision_feature_select_strategy = self.config.mm_vision_select_feature
        self.image_size = self.vision_tower.config.image_size
        self.patch_size = self.vision_tower.config.patch_size
        self.mm_patch_merge_type = getattr(self.config, 'mm_patch_merge_type', 'flat')
        self.image_aspect_ratio = getattr(self.config, 'image_aspect_ratio', 'square')
        self.image_grid_pinpoints = getattr(self.config, 'image_grid_pinpoints', None)
        self.image_feature_len = int((self.image_size // self.patch_size) ** 2)
        if self.vision_feature_select_strategy == 'patch' or self.vision_feature_select_strategy == 'full':
            pass
        elif self.vision_feature_select_strategy == 'cls_patch':
            self.image_feature_len += 1
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        projector_weights = {'model.mm_projector.0': 'multi_modal_projector.linear_1', 'model.mm_projector.2': 'multi_modal_projector.linear_2', 'model.vision_tower.vision_tower': 'vision_tower', 'model.image_newline': 'language_model.model.image_newline'}
        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            if 'projector' in name or 'vision_tower' in name or 'image_newline' in name:
                for weight_name, param_name in projector_weights.items():
                    if weight_name in name:
                        name = name.replace(weight_name, param_name)
                param = params_dict[name]
                weight_loader = getattr(param, 'weight_loader', default_weight_loader)
                weight_loader(param, loaded_weight)
            else:
                self.language_model.load_weights([(name, loaded_weight)])

    @property
    def num_patches_per_side(self):
        return self.image_size // self.patch_size


class LlavaLlamaForCausalLM(LlavaBaseForCausalLM):

    def __init__(self, config: 'LlavaConfig', quant_config: 'Optional[QuantizationConfig]'=None, cache_config=None) ->None:
        super().__init__()
        self.config = config
        self.vision_tower = None
        self.config.vision_config.hidden_size = config.mm_hidden_size
        self.config.text_config.hidden_size = config.hidden_size
        self.multi_modal_projector = LlavaMultiModalProjector(config)
        self.language_model = LlamaForCausalLM(config, quant_config=quant_config)
        if 'unpad' in getattr(config, 'mm_patch_merge_type', ''):
            self.language_model.model.image_newline = nn.Parameter(torch.empty(config.text_config.hidden_size, dtype=torch.float16))


Qwen2Config = None


class Qwen2Attention(nn.Module):

    def __init__(self, hidden_size: 'int', num_heads: 'int', num_kv_heads: 'int', layer_id: 'int'=0, rope_theta: 'float'=1000000, rope_scaling: 'Optional[Dict[str, Any]]'=None, max_position_embeddings: 'int'=32768, quant_config: 'Optional[QuantizationConfig]'=None) ->None:
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim ** -0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
        self.qkv_proj = QKVParallelLinear(hidden_size, self.head_dim, self.total_num_heads, self.total_num_kv_heads, bias=True, quant_config=quant_config)
        self.o_proj = RowParallelLinear(self.total_num_heads * self.head_dim, hidden_size, bias=False, quant_config=quant_config)
        self.rotary_emb = get_rope(self.head_dim, rotary_dim=self.head_dim, max_position=max_position_embeddings, base=rope_theta, rope_scaling=rope_scaling)
        self.attn = RadixAttention(self.num_heads, self.head_dim, self.scaling, num_kv_heads=self.num_kv_heads, layer_id=layer_id)

    def forward(self, positions: 'torch.Tensor', hidden_states: 'torch.Tensor', forward_batch: 'ForwardBatch') ->torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v, forward_batch)
        output, _ = self.o_proj(attn_output)
        return output


class Qwen2MLP(nn.Module):

    def __init__(self, hidden_size: 'int', intermediate_size: 'int', hidden_act: 'str', quant_config: 'Optional[QuantizationConfig]'=None) ->None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(hidden_size, [intermediate_size] * 2, bias=False, quant_config=quant_config)
        self.down_proj = RowParallelLinear(intermediate_size, hidden_size, bias=False, quant_config=quant_config)
        if hidden_act != 'silu':
            raise ValueError(f'Unsupported activation: {hidden_act}. Only silu is supported for now.')
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class Qwen2DecoderLayer(nn.Module):

    def __init__(self, config: 'Qwen2Config', layer_id: 'int'=0, quant_config: 'Optional[QuantizationConfig]'=None) ->None:
        super().__init__()
        self.hidden_size = config.hidden_size
        rope_theta = getattr(config, 'rope_theta', 1000000)
        rope_scaling = getattr(config, 'rope_scaling', None)
        max_position_embeddings = getattr(config, 'max_position_embeddings', 32768)
        self.self_attn = Qwen2Attention(hidden_size=self.hidden_size, num_heads=config.num_attention_heads, num_kv_heads=config.num_key_value_heads, layer_id=layer_id, rope_theta=rope_theta, rope_scaling=rope_scaling, max_position_embeddings=max_position_embeddings, quant_config=quant_config)
        self.mlp = Qwen2MLP(hidden_size=self.hidden_size, intermediate_size=config.intermediate_size, hidden_act=config.hidden_act, quant_config=quant_config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, positions: 'torch.Tensor', hidden_states: 'torch.Tensor', forward_batch: 'ForwardBatch', residual: 'Optional[torch.Tensor]') ->Tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions=positions, hidden_states=hidden_states, forward_batch=forward_batch)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class Qwen2Model(nn.Module):

    def __init__(self, config: 'Qwen2Config', quant_config: 'Optional[QuantizationConfig]'=None) ->None:
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([Qwen2DecoderLayer(config, i, quant_config=quant_config) for i in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids: 'torch.Tensor', positions: 'torch.Tensor', forward_batch: 'ForwardBatch', input_embeds: 'torch.Tensor'=None) ->torch.Tensor:
        if input_embeds is None:
            hidden_states = self.embed_tokens(input_ids)
        else:
            hidden_states = input_embeds
        residual = None
        for i in range(len(self.layers)):
            layer = self.layers[i]
            hidden_states, residual = layer(positions, hidden_states, forward_batch, residual)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class Qwen2ForCausalLM(nn.Module):

    def __init__(self, config: 'Qwen2Config', quant_config: 'Optional[QuantizationConfig]'=None, cache_config=None) ->None:
        super().__init__()
        self.config = config
        self.quant_config = quant_config
        self.model = Qwen2Model(config, quant_config=quant_config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        self.logits_processor = LogitsProcessor(config)
        self.pooler = Pooler(pooling_type=PoolingType.LAST, normalize=True)

    @torch.no_grad()
    def forward(self, input_ids: 'torch.Tensor', positions: 'torch.Tensor', forward_batch: 'ForwardBatch', input_embeds: 'torch.Tensor'=None, get_embedding: 'bool'=False) ->torch.Tensor:
        hidden_states = self.model(input_ids, positions, forward_batch, input_embeds)
        if not get_embedding:
            return self.logits_processor(input_ids, hidden_states, self.lm_head.weight, forward_batch)
        else:
            return self.pooler(hidden_states, forward_batch)

    def load_weights(self, weights: 'Iterable[Tuple[str, torch.Tensor]]'):
        stacked_params_mapping = [('qkv_proj', 'q_proj', 'q'), ('qkv_proj', 'k_proj', 'k'), ('qkv_proj', 'v_proj', 'v'), ('gate_up_proj', 'gate_proj', 0), ('gate_up_proj', 'up_proj', 1)]
        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            if 'rotary_emb.inv_freq' in name or 'projector' in name:
                continue
            if 'rotary_emb.cos_cached' in name or 'rotary_emb.sin_cached' in name:
                continue
            if name.startswith('model.vision_tower') and name not in params_dict:
                continue
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                if name.endswith('.bias') and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                if name.endswith('.bias') and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, 'weight_loader', default_weight_loader)
                weight_loader(param, loaded_weight)
                if self.config.tie_word_embeddings and name == 'model.embed_tokens.weight':
                    weight_loader(params_dict['lm_head.weight'], loaded_weight)


class LlavaQwenForCausalLM(LlavaBaseForCausalLM):

    def __init__(self, config: 'LlavaConfig', quant_config: 'Optional[QuantizationConfig]'=None, cache_config=None) ->None:
        super().__init__()
        self.config = config
        self.vision_tower = None
        if getattr(self.config, 'vision_config', None) is None:
            self.config.vision_config = CLIPVisionConfig(self.config.mm_vision_tower)
        if getattr(self.config, 'text_config', None) is None:
            self.config.text_config = Qwen2Config(self.config._name_or_path)
        self.config.vision_config.hidden_size = config.mm_hidden_size
        self.config.text_config.hidden_size = config.hidden_size
        if getattr(self.config, 'projector_hidden_act', None) is None:
            self.config.projector_hidden_act = 'gelu'
        if getattr(self.config, 'image_token_index', None) is None:
            self.config.image_token_index = 151646
        self.multi_modal_projector = LlavaMultiModalProjector(config)
        self.language_model = Qwen2ForCausalLM(config, quant_config=quant_config)
        if 'unpad' in getattr(config, 'mm_patch_merge_type', ''):
            self.language_model.model.image_newline = nn.Parameter(torch.empty(config.text_config.hidden_size, dtype=torch.float16))


class LlavaMistralForCausalLM(LlavaBaseForCausalLM):

    def __init__(self, config: 'LlavaConfig', quant_config: 'Optional[QuantizationConfig]'=None, cache_config=None) ->None:
        super().__init__()
        self.config = config
        self.vision_tower = None
        if getattr(self.config, 'vision_config', None) is None:
            self.config.vision_config = CLIPVisionConfig(self.config.mm_vision_tower)
        if getattr(self.config, 'text_config', None) is None:
            self.config.text_config = MistralConfig(self.config._name_or_path)
        self.config.vision_config.hidden_size = config.mm_hidden_size
        self.config.text_config.hidden_size = config.hidden_size
        if getattr(self.config, 'projector_hidden_act', None) is None:
            self.config.projector_hidden_act = 'gelu'
        if getattr(self.config, 'image_token_index', None) is None:
            self.config.image_token_index = 32000
        self.multi_modal_projector = LlavaMultiModalProjector(config)
        self.language_model = MistralForCausalLM(config, quant_config=quant_config)
        if 'unpad' in getattr(config, 'mm_patch_merge_type', ''):
            self.language_model.model.image_newline = nn.Parameter(torch.empty(config.text_config.hidden_size, dtype=torch.float16))


class LlavaVidForCausalLM(nn.Module):

    def __init__(self, config: 'LlavaConfig', quant_config: 'Optional[QuantizationConfig]'=None, cache_config=None) ->None:
        super().__init__()
        self.config = config
        self.vision_tower = None
        self.config.vision_config.hidden_size = config.mm_hidden_size
        self.config.text_config.hidden_size = config.hidden_size
        self.multi_modal_projector = LlavaMultiModalProjector(config)
        self.mm_spatial_pool_stride = getattr(self.config, 'mm_spatial_pool_stride', 2)
        self.resampler = nn.AvgPool2d(kernel_size=self.mm_spatial_pool_stride, stride=self.mm_spatial_pool_stride)
        self.language_model = LlamaForCausalLM(config, quant_config=quant_config)
        self.num_frames = getattr(self.config, 'num_frames', 16)
        if 'unpad' in getattr(config, 'mm_patch_merge_type', ''):
            self.language_model.model.image_newline = nn.Parameter(torch.empty(config.text_config.hidden_size, dtype=torch.float16))

    def pad_input_ids(self, input_ids: 'List[int]', image_inputs: 'ImageInputs'):
        pad_values = image_inputs.pad_values
        new_image_feature_len = self.image_feature_len
        pad_ids = pad_values * ((new_image_feature_len + len(pad_values)) // len(pad_values))
        offset = input_ids.index(self.config.image_token_index)
        new_input_ids = input_ids[:offset] + pad_ids[:new_image_feature_len] + input_ids[offset + 1:]
        image_inputs.image_offsets = [offset]
        return new_input_ids

    def encode_images(self, pixel_values: 'torch.Tensor') ->torch.Tensor:
        image_outputs = self.vision_tower(pixel_values, output_hidden_states=True)
        selected_image_feature = image_outputs.hidden_states[self.vision_feature_layer]
        if self.vision_feature_select_strategy in ['default', 'patch']:
            selected_image_feature = selected_image_feature[:, 1:]
        elif self.vision_feature_select_strategy == 'full':
            selected_image_feature = selected_image_feature
        else:
            raise ValueError(f'Unexpected select feature strategy: {self.config.vision_feature_select_strategy}')
        height = width = self.num_patches_per_side
        num_of_frames = selected_image_feature.shape[0]
        selected_image_feature = selected_image_feature.view(num_of_frames, height, width, -1)
        selected_image_feature = selected_image_feature.permute(0, 3, 1, 2).contiguous()
        selected_image_feature = self.resampler(selected_image_feature).flatten(2).transpose(1, 2).contiguous()
        image_features = self.multi_modal_projector(selected_image_feature)
        return image_features

    @torch.no_grad()
    def forward(self, input_ids: 'torch.LongTensor', positions: 'torch.Tensor', forward_batch: 'ForwardBatch') ->torch.Tensor:
        image_inputs = forward_batch.image_inputs
        if forward_batch.forward_mode.is_extend():
            bs = forward_batch.batch_size
            input_embeds = self.language_model.model.embed_tokens(input_ids)
            max_image_offset = []
            for im in image_inputs:
                if im and im.image_offsets:
                    max_image_offset.append(max(im.image_offsets))
                else:
                    max_image_offset.append(-1)
            start_positions = positions[forward_batch.extend_start_loc].cpu().numpy()
            need_vision = start_positions <= np.array(max_image_offset)
            if need_vision.any():
                pixel_values = [image_inputs[i].pixel_values for i in range(bs) if need_vision[i]]
                image_offsets = [image_inputs[i].image_offsets for i in range(bs) if need_vision[i]]
                if pixel_values[0].ndim == 4:
                    np.concatenate(pixel_values, axis=0)
                    concat_images = torch.tensor(np.concatenate(pixel_values, axis=0), device=self.vision_tower.device)
                    image_features = self.encode_images(concat_images)
                    split_sizes = [image.shape[0] for image in pixel_values]
                    image_features = torch.split(image_features, split_sizes, dim=0)
                else:
                    pixel_values = torch.tensor(np.array(pixel_values), device=self.vision_tower.device)
                    image_features = self.encode_images(pixel_values)
                new_image_features = []
                for image_idx, image_feature in enumerate(image_features):
                    new_image_features.append(image_feature.flatten(0, 1))
                image_features = new_image_features
                extend_start_loc_cpu = forward_batch.extend_start_loc.cpu().numpy()
                prefix_lens_cpu = forward_batch.extend_prefix_lens.cpu().numpy()
                pt = 0
                for i in range(bs):
                    if not need_vision[i]:
                        continue
                    start_idx = extend_start_loc_cpu[i]
                    prefix_len = prefix_lens_cpu[i]
                    for image_offset in image_offsets[i]:
                        if image_offset < prefix_len:
                            continue
                        tmp_image_feature = image_features[pt]
                        pad_len = tmp_image_feature.shape[0]
                        left_idx = start_idx + (image_offset - prefix_len)
                        right_idx = start_idx + (image_offset - prefix_len) + pad_len
                        try:
                            input_embeds[left_idx:right_idx] = tmp_image_feature
                        except RuntimeError as e:
                            None
                            None
                            None
                        pt += 1
            return self.language_model(input_ids, positions, forward_batch, input_embeds=input_embeds)
        elif forward_batch.forward_mode.is_decode():
            return self.language_model(input_ids, positions, forward_batch)

    def load_weights(self, weights: 'Iterable[Tuple[str, torch.Tensor]]'):
        vision_path = self.config.mm_vision_tower
        self.vision_tower = CLIPVisionModel.from_pretrained(vision_path, torch_dtype=torch.float16)
        self.vision_tower.eval()
        self.vision_feature_layer = self.config.mm_vision_select_layer
        self.vision_feature_select_strategy = self.config.mm_vision_select_feature
        self.image_size = self.vision_tower.config.image_size
        self.patch_size = self.vision_tower.config.patch_size
        self.mm_patch_merge_type = getattr(self.config, 'mm_patch_merge_type', 'flat')
        self.image_aspect_ratio = getattr(self.config, 'image_aspect_ratio', 'square')
        self.image_grid_pinpoints = getattr(self.config, 'image_grid_pinpoints', None)
        None
        self.image_feature_len = self.num_frames * int((self.image_size / self.patch_size / self.mm_spatial_pool_stride) ** 2)
        if self.vision_feature_select_strategy == 'patch':
            pass
        elif self.vision_feature_select_strategy == 'cls_patch':
            self.image_feature_len += 1
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        projector_weights = {'model.mm_projector.0': 'multi_modal_projector.linear_1', 'model.mm_projector.2': 'multi_modal_projector.linear_2', 'model.vision_resampler.mm_projector.0': 'multi_modal_projector.linear_1', 'model.vision_resampler.mm_projector.2': 'multi_modal_projector.linear_2', 'model.vision_tower.vision_tower': 'vision_tower', 'model.image_newline': 'language_model.model.image_newline'}
        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            if 'projector' in name or 'vision_tower' in name or 'image_newline' in name:
                for weight_name, param_name in projector_weights.items():
                    if weight_name in name:
                        name = name.replace(weight_name, param_name)
                if name in params_dict:
                    param = params_dict[name]
                else:
                    None
                    continue
                weight_loader = getattr(param, 'weight_loader', default_weight_loader)
                weight_loader(param, loaded_weight)
            else:
                self.language_model.load_weights([(name, loaded_weight)])

    @property
    def num_patches_per_side(self):
        return self.image_size // self.patch_size


class MiniCPMMLP(nn.Module):

    def __init__(self, hidden_size: 'int', intermediate_size: 'int', hidden_act: 'str', quant_config: 'Optional[QuantizationConfig]'=None) ->None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(hidden_size, [intermediate_size] * 2, bias=False, quant_config=quant_config)
        self.down_proj = RowParallelLinear(intermediate_size, hidden_size, bias=False, quant_config=quant_config)
        if hidden_act != 'silu':
            raise ValueError(f'Unsupported activation: {hidden_act}. Only silu is supported for now.')
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class MiniCPMAttention(nn.Module):

    def __init__(self, hidden_size: 'int', num_heads: 'int', num_kv_heads: 'int', layer_id: 'int'=0, rope_theta: 'float'=10000, rope_scaling: 'Optional[Dict[str, Any]]'=None, max_position_embeddings: 'int'=8192, quant_config: 'Optional[QuantizationConfig]'=None) ->None:
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim ** -0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
        self.qkv_proj = QKVParallelLinear(hidden_size, self.head_dim, self.total_num_heads, self.total_num_kv_heads, bias=False, quant_config=quant_config)
        self.o_proj = RowParallelLinear(self.total_num_heads * self.head_dim, hidden_size, bias=False, quant_config=quant_config)
        self.rotary_emb = get_rope(self.head_dim, rotary_dim=self.head_dim, max_position=max_position_embeddings, base=rope_theta, rope_scaling=rope_scaling)
        self.rotary_emb.cos_sin_cache = self.rotary_emb._compute_cos_sin_cache()
        self.attn = RadixAttention(self.num_heads, self.head_dim, self.scaling, num_kv_heads=self.num_kv_heads, layer_id=layer_id)

    def forward(self, positions: 'torch.Tensor', hidden_states: 'torch.Tensor', forward_batch: 'ForwardBatch') ->torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        orig_dtype = q.dtype
        q, k = q.float(), k.float()
        q, k = self.rotary_emb(positions, q, k)
        q, k = q, k
        attn_output = self.attn(q, k, v, forward_batch)
        output, _ = self.o_proj(attn_output)
        return output


class MiniCPMDecoderLayer(nn.Module):

    def __init__(self, config, layer_id: 'int'=0, quant_config: 'Optional[QuantizationConfig]'=None) ->None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        rope_theta = getattr(config, 'rope_theta', 10000)
        rope_scaling = getattr(config, 'rope_scaling', None)
        max_position_embeddings = getattr(config, 'max_position_embeddings', 8192)
        self.self_attn = MiniCPMAttention(hidden_size=self.hidden_size, num_heads=config.num_attention_heads, num_kv_heads=config.num_key_value_heads, layer_id=layer_id, rope_theta=rope_theta, rope_scaling=rope_scaling, max_position_embeddings=max_position_embeddings, quant_config=quant_config)
        self.mlp = MiniCPMMLP(hidden_size=self.hidden_size, intermediate_size=config.intermediate_size, hidden_act=config.hidden_act, quant_config=quant_config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, positions: 'torch.Tensor', hidden_states: 'torch.Tensor', forward_batch: 'ForwardBatch', residual: 'Optional[torch.Tensor]') ->Tuple[torch.Tensor, torch.Tensor]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(positions=positions, hidden_states=hidden_states, forward_batch=forward_batch)
        hidden_states = residual + hidden_states * (self.config.scale_depth / math.sqrt(self.config.num_hidden_layers))
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states * (self.config.scale_depth / math.sqrt(self.config.num_hidden_layers))
        return hidden_states, None


class MiniCPMModel(nn.Module):

    def __init__(self, config, quant_config: 'Optional[QuantizationConfig]'=None) ->None:
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = VocabParallelEmbedding(self.vocab_size, config.hidden_size, org_num_embeddings=config.vocab_size)
        self.layers = nn.ModuleList([MiniCPMDecoderLayer(config, i, quant_config=quant_config) for i in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids: 'torch.Tensor', positions: 'torch.Tensor', forward_batch: 'ForwardBatch', input_embeds: 'torch.Tensor'=None) ->torch.Tensor:
        if input_embeds is None:
            hidden_states = self.embed_tokens(input_ids) * self.config.scale_emb
        else:
            hidden_states = input_embeds
        residual = None
        for i in range(len(self.layers)):
            layer = self.layers[i]
            hidden_states, residual = layer(positions, hidden_states, forward_batch, residual)
        hidden_states = self.norm(hidden_states)
        return hidden_states


class MiniCPMForCausalLM(nn.Module):

    def __init__(self, config, quant_config: 'Optional[QuantizationConfig]'=None, cache_config=None) ->None:
        super().__init__()
        self.config = config
        self.num_experts = getattr(self.config, 'num_experts', 0)
        self.quant_config = quant_config
        self.model = MiniCPMModel(config, quant_config=quant_config)
        if not self.config.tie_word_embeddings:
            self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size, org_num_embeddings=config.vocab_size)
        self.scale_width = self.config.hidden_size / self.config.dim_model_base
        self.logits_processor = LogitsProcessor(config)

    @torch.no_grad()
    def forward(self, input_ids: 'torch.Tensor', positions: 'torch.Tensor', forward_batch: 'ForwardBatch', input_embeds: 'torch.Tensor'=None) ->torch.Tensor:
        if input_embeds is not None:
            input_embeds = input_embeds * self.config.scale_emb
        hidden_states = self.model(input_ids, positions, forward_batch, input_embeds)
        hidden_states = hidden_states / self.scale_width
        if self.config.tie_word_embeddings:
            lm_head_weight = self.model.embed_tokens.weight
        else:
            lm_head_weight = self.lm_head.weight
        return self.logits_processor(input_ids, hidden_states, lm_head_weight, forward_batch)

    def load_weights(self, weights: 'Iterable[Tuple[str, torch.Tensor]]'):
        stacked_params_mapping = [('qkv_proj', 'q_proj', 'q'), ('qkv_proj', 'k_proj', 'k'), ('qkv_proj', 'v_proj', 'v'), ('gate_up_proj', 'gate_proj', 0), ('gate_up_proj', 'up_proj', 1)]
        expert_params_mapping = [('ws' if weight_name in ['w1', 'w3'] else 'w2s', f'experts.{expert_id}.{weight_name}.weight', expert_id) for expert_id in range(self.num_experts) for weight_name in ['w1', 'w2', 'w3']]
        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            if 'rotary_emb.inv_freq' in name:
                continue
            if 'rotary_emb.cos_cached' in name or 'rotary_emb.sin_cached' in name:
                continue
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                if name.endswith('.bias') and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                for param_name, weight_name, expert_id in expert_params_mapping:
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)
                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    weight_loader(param, loaded_weight, weight_name, expert_id=expert_id)
                    break
                else:
                    if name.endswith('.bias') and name not in params_dict:
                        continue
                    param = params_dict[name]
                    weight_loader = getattr(param, 'weight_loader', default_weight_loader)
                    weight_loader(param, loaded_weight)


class MiniCPM3MLP(nn.Module):

    def __init__(self, hidden_size: 'int', intermediate_size: 'int', hidden_act: 'str', quant_config: 'Optional[QuantizationConfig]'=None) ->None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(hidden_size, [intermediate_size] * 2, bias=False, quant_config=quant_config)
        self.down_proj = RowParallelLinear(intermediate_size, hidden_size, bias=False, quant_config=quant_config)
        if hidden_act != 'silu':
            raise ValueError(f'Unsupported activation: {hidden_act}. Only silu is supported for now.')
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class MiniCPM3Attention(nn.Module):

    def __init__(self, config: 'PretrainedConfig', hidden_size: 'int', num_heads: 'int', qk_nope_head_dim: 'int', qk_rope_head_dim: 'int', v_head_dim: 'int', q_lora_rank: 'int', kv_lora_rank: 'int', rope_theta: 'float'=10000, rope_scaling: 'Optional[Dict[str, Any]]'=None, max_position_embeddings: 'int'=8192, cache_config=None, quant_config: 'Optional[QuantizationConfig]'=None, layer_id=None) ->None:
        super().__init__()
        self.layer_id = layer_id
        self.hidden_size = hidden_size
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.num_heads = num_heads
        tp_size = get_tensor_model_parallel_world_size()
        assert num_heads % tp_size == 0
        self.num_local_heads = num_heads // tp_size
        self.scaling = self.qk_head_dim ** -0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
        if self.q_lora_rank is not None:
            self.q_a_proj = ReplicatedLinear(self.hidden_size, self.q_lora_rank, bias=False, quant_config=quant_config)
            self.q_a_layernorm = RMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
            self.q_b_proj = ColumnParallelLinear(q_lora_rank, self.num_heads * self.qk_head_dim, bias=False, quant_config=quant_config)
        else:
            self.q_proj = ColumnParallelLinear(self.hidden_size, self.num_heads * self.qk_head_dim, bias=False, quant_config=quant_config)
        self.kv_a_proj_with_mqa = ReplicatedLinear(self.hidden_size, self.kv_lora_rank + self.qk_rope_head_dim, bias=False, quant_config=quant_config)
        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps)
        self.kv_b_proj = ColumnParallelLinear(self.kv_lora_rank, self.num_heads * (self.qk_nope_head_dim + self.v_head_dim), bias=False, quant_config=quant_config)
        self.o_proj = RowParallelLinear(self.num_heads * self.v_head_dim, self.hidden_size, bias=False, quant_config=quant_config)
        self.rotary_emb = get_rope(qk_rope_head_dim, rotary_dim=qk_rope_head_dim, max_position=max_position_embeddings, base=rope_theta, rope_scaling=rope_scaling)
        self.attn = RadixAttention(self.num_local_heads, 128, self.scaling, num_kv_heads=self.num_local_heads, layer_id=layer_id)

    def forward(self, positions: 'torch.Tensor', hidden_states: 'torch.Tensor', forward_batch: 'ForwardBatch') ->torch.Tensor:
        if self.q_lora_rank is not None:
            q = self.q_a_proj(hidden_states)[0]
            q = self.q_a_layernorm(q)
            q = self.q_b_proj(q)[0].view(-1, self.num_local_heads, self.qk_head_dim)
        else:
            q = self.q_proj(hidden_states)[0].view(-1, self.num_local_heads, self.qk_head_dim)
        _, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        latent_cache = self.kv_a_proj_with_mqa(hidden_states)[0]
        kv_a, _ = latent_cache.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        latent_cache = latent_cache.unsqueeze(1)
        kv_a = self.kv_a_layernorm(kv_a.contiguous())
        kv = self.kv_b_proj(kv_a)[0]
        kv = kv.view(-1, self.num_local_heads, self.qk_nope_head_dim + self.v_head_dim)
        k_nope, v = kv.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        k_pe = latent_cache[:, :, self.kv_lora_rank:]
        original_shapes = [q_pe.shape, k_pe.shape]
        q_pe, k_pe = self.rotary_emb(positions, q_pe.reshape(q_pe.shape[0], -1), k_pe.reshape(k_pe.shape[0], -1))
        q_pe, k_pe = q_pe.view(original_shapes[0]), k_pe.view(original_shapes[1])
        q[..., self.qk_nope_head_dim:] = q_pe
        k = torch.empty_like(q)
        k[..., :self.qk_nope_head_dim] = k_nope
        k[..., self.qk_nope_head_dim:] = k_pe
        q = torch.nn.functional.pad(q, [0, 128 - self.qk_head_dim], value=0).view(-1, self.num_local_heads * 128)
        k = torch.nn.functional.pad(k, [0, 128 - self.qk_head_dim], value=0).view(-1, self.num_local_heads * 128)
        v = torch.nn.functional.pad(v, [0, 128 - self.v_head_dim], value=0).view(-1, self.num_local_heads * 128)
        attn_output = self.attn(q, k, v, forward_batch)
        attn_output = attn_output.view(-1, self.num_local_heads, 128)[..., :self.v_head_dim].reshape(-1, self.num_local_heads * self.v_head_dim)
        output, _ = self.o_proj(attn_output)
        return output


class MiniCPM3AttentionMLA(nn.Module):

    def __init__(self, config: 'PretrainedConfig', hidden_size: 'int', num_heads: 'int', qk_nope_head_dim: 'int', qk_rope_head_dim: 'int', v_head_dim: 'int', q_lora_rank: 'int', kv_lora_rank: 'int', rope_theta: 'float'=10000, rope_scaling: 'Optional[Dict[str, Any]]'=None, max_position_embeddings: 'int'=8192, cache_config=None, quant_config: 'Optional[QuantizationConfig]'=None, layer_id=None) ->None:
        super().__init__()
        self.layer_id = layer_id
        self.hidden_size = hidden_size
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.num_heads = num_heads
        tp_size = get_tensor_model_parallel_world_size()
        assert num_heads % tp_size == 0
        self.num_local_heads = num_heads // tp_size
        self.scaling = self.qk_head_dim ** -0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
        if self.q_lora_rank is not None:
            self.q_a_proj = ReplicatedLinear(self.hidden_size, self.q_lora_rank, bias=False, quant_config=quant_config)
            self.q_a_layernorm = RMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
            self.q_b_proj = ColumnParallelLinear(q_lora_rank, self.num_heads * self.qk_head_dim, bias=False, quant_config=quant_config)
        else:
            self.q_proj = ColumnParallelLinear(self.hidden_size, self.num_heads * self.qk_head_dim, bias=False, quant_config=quant_config)
        self.kv_a_proj_with_mqa = ReplicatedLinear(self.hidden_size, self.kv_lora_rank + self.qk_rope_head_dim, bias=False, quant_config=quant_config)
        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps)
        self.kv_b_proj = ColumnParallelLinear(self.kv_lora_rank, self.num_heads * (self.qk_nope_head_dim + self.v_head_dim), bias=False, quant_config=quant_config)
        self.o_proj = RowParallelLinear(self.num_heads * self.v_head_dim, self.hidden_size, bias=False, quant_config=quant_config)
        self.rotary_emb = get_rope(qk_rope_head_dim, rotary_dim=qk_rope_head_dim, max_position=max_position_embeddings, base=rope_theta, rope_scaling=rope_scaling)
        self.attn = RadixAttention(self.num_local_heads, self.kv_lora_rank + self.qk_rope_head_dim, self.scaling, num_kv_heads=1, layer_id=layer_id, v_head_dim=self.kv_lora_rank)
        self.w_kc = None
        self.w_vc = None
        self.w_scale = None

    def forward(self, positions: 'torch.Tensor', hidden_states: 'torch.Tensor', forward_batch: 'ForwardBatch') ->torch.Tensor:
        q_len = hidden_states.shape[0]
        q_input = hidden_states.new_empty(q_len, self.num_local_heads, self.kv_lora_rank + self.qk_rope_head_dim)
        if self.q_lora_rank is not None:
            q = self.q_a_proj(hidden_states)[0]
            q = self.q_a_layernorm(q)
            q = self.q_b_proj(q)[0].view(-1, self.num_local_heads, self.qk_head_dim)
        else:
            q = self.q_proj(hidden_states)[0].view(-1, self.num_local_heads, self.qk_head_dim)
        q_nope, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        if self.w_kc.dtype == torch.float8_e4m3fn:
            q_nope_val, q_nope_scale = input_to_float8(q_nope.transpose(0, 1), torch.float8_e4m3fn)
            q_nope_out = bmm_fp8(q_nope_val, self.w_kc, q_nope_scale, self.w_scale, torch.bfloat16)
        else:
            q_nope_out = torch.bmm(q_nope.transpose(0, 1), self.w_kc)
        q_input[..., :self.kv_lora_rank] = q_nope_out.transpose(0, 1)
        latent_cache = self.kv_a_proj_with_mqa(hidden_states)[0]
        v_input = latent_cache[..., :self.kv_lora_rank]
        v_input = self.kv_a_layernorm(v_input.contiguous()).unsqueeze(1)
        k_input = latent_cache.unsqueeze(1)
        k_input[..., :self.kv_lora_rank] = v_input
        k_pe = k_input[..., self.kv_lora_rank:]
        original_shapes = [q_pe.shape, k_pe.shape]
        q_pe, k_pe = self.rotary_emb(positions, q_pe.reshape(q_pe.shape[0], -1), k_pe.reshape(k_pe.shape[0], -1))
        q_pe, k_pe = q_pe.view(original_shapes[0]), k_pe.view(original_shapes[1])
        q_input[..., self.kv_lora_rank:] = q_pe
        k_input[..., self.kv_lora_rank:] = k_pe
        attn_output = self.attn(q_input, k_input, v_input, forward_batch)
        attn_output = attn_output.view(-1, self.num_local_heads, self.kv_lora_rank)
        if self.w_vc.dtype == torch.float8_e4m3fn:
            attn_output_val, attn_output_scale = input_to_float8(attn_output.transpose(0, 1), torch.float8_e4m3fn)
            attn_bmm_output = bmm_fp8(attn_output_val, self.w_vc, attn_output_scale, self.w_scale, torch.bfloat16)
        else:
            attn_bmm_output = torch.bmm(attn_output.transpose(0, 1), self.w_vc)
        attn_output = attn_bmm_output.transpose(0, 1).flatten(1, 2)
        output, _ = self.o_proj(attn_output)
        return output


class MiniCPM3DecoderLayer(nn.Module):

    def __init__(self, config: 'PretrainedConfig', layer_id: 'int', cache_config=None, quant_config: 'Optional[QuantizationConfig]'=None) ->None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        rope_theta = getattr(config, 'rope_theta', 10000)
        rope_scaling = getattr(config, 'rope_scaling', None)
        max_position_embeddings = getattr(config, 'max_position_embeddings', 8192)
        if not global_server_args_dict['disable_mla']:
            self.self_attn = MiniCPM3AttentionMLA(config=config, hidden_size=self.hidden_size, num_heads=config.num_attention_heads, qk_nope_head_dim=config.qk_nope_head_dim, qk_rope_head_dim=config.qk_rope_head_dim, v_head_dim=self.hidden_size // config.num_attention_heads, q_lora_rank=config.q_lora_rank if hasattr(config, 'q_lora_rank') else None, kv_lora_rank=config.kv_lora_rank, rope_theta=rope_theta, rope_scaling=rope_scaling, max_position_embeddings=max_position_embeddings, cache_config=cache_config, quant_config=quant_config, layer_id=layer_id)
        else:
            self.self_attn = MiniCPM3Attention(config=config, hidden_size=self.hidden_size, num_heads=config.num_attention_heads, qk_nope_head_dim=config.qk_nope_head_dim, qk_rope_head_dim=config.qk_rope_head_dim, v_head_dim=self.hidden_size // config.num_attention_heads, q_lora_rank=config.q_lora_rank if hasattr(config, 'q_lora_rank') else None, kv_lora_rank=config.kv_lora_rank, rope_theta=rope_theta, rope_scaling=rope_scaling, max_position_embeddings=max_position_embeddings, cache_config=cache_config, quant_config=quant_config, layer_id=layer_id)
        self.mlp = MiniCPM3MLP(hidden_size=self.hidden_size, intermediate_size=config.intermediate_size, hidden_act=config.hidden_act, quant_config=quant_config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, positions: 'torch.Tensor', hidden_states: 'torch.Tensor', forward_batch: 'ForwardBatch', residual: 'Optional[torch.Tensor]') ->Tuple[torch.Tensor, torch.Tensor]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(positions=positions, hidden_states=hidden_states, forward_batch=forward_batch)
        hidden_states = residual + hidden_states * (self.config.scale_depth / math.sqrt(self.config.num_hidden_layers))
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states * (self.config.scale_depth / math.sqrt(self.config.num_hidden_layers))
        return hidden_states, None


class MiniCPM3Model(nn.Module):

    def __init__(self, config: 'PretrainedConfig', cache_config=None, quant_config: 'Optional[QuantizationConfig]'=None) ->None:
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = VocabParallelEmbedding(self.vocab_size, config.hidden_size, org_num_embeddings=config.vocab_size)
        self.layers = nn.ModuleList([MiniCPM3DecoderLayer(config, i, cache_config=cache_config, quant_config=quant_config) for i in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids: 'torch.Tensor', positions: 'torch.Tensor', forward_batch: 'ForwardBatch', input_embeds: 'torch.Tensor'=None) ->torch.Tensor:
        if input_embeds is None:
            hidden_states = self.embed_tokens(input_ids) * self.config.scale_emb
        else:
            hidden_states = input_embeds
        residual = None
        for i in range(len(self.layers)):
            layer = self.layers[i]
            hidden_states, residual = layer(positions, hidden_states, forward_batch, residual)
        hidden_states = self.norm(hidden_states)
        return hidden_states


class MiniCPM3ForCausalLM(nn.Module):

    def __init__(self, config: 'PretrainedConfig', cache_config=None, quant_config: 'Optional[QuantizationConfig]'=None) ->None:
        super().__init__()
        self.config = config
        self.num_experts = getattr(self.config, 'num_experts', 0)
        self.quant_config = quant_config
        self.model = MiniCPM3Model(config, cache_config=cache_config, quant_config=quant_config)
        if not self.config.tie_word_embeddings:
            self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size, org_num_embeddings=config.vocab_size)
        self.scale_width = self.config.hidden_size / self.config.dim_model_base
        self.logits_processor = LogitsProcessor(config)

    @torch.no_grad()
    def forward(self, input_ids: 'torch.Tensor', positions: 'torch.Tensor', forward_batch: 'ForwardBatch', input_embeds: 'torch.Tensor'=None) ->torch.Tensor:
        if input_embeds is not None:
            input_embeds = input_embeds * self.config.scale_emb
        hidden_states = self.model(input_ids, positions, forward_batch, input_embeds)
        hidden_states = hidden_states / self.scale_width
        if self.config.tie_word_embeddings:
            lm_head_weight = self.model.embed_tokens.weight
        else:
            lm_head_weight = self.lm_head.weight
        return self.logits_processor(input_ids, hidden_states, lm_head_weight, forward_batch)

    def load_weights(self, weights: 'Iterable[Tuple[str, torch.Tensor]]'):
        stacked_params_mapping = [('gate_up_proj', 'gate_proj', 0), ('gate_up_proj', 'up_proj', 1)]
        expert_params_mapping = [('ws' if weight_name in ['w1', 'w3'] else 'w2s', f'experts.{expert_id}.{weight_name}.weight', expert_id) for expert_id in range(self.num_experts) for weight_name in ['w1', 'w2', 'w3']]
        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            if 'rotary_emb.inv_freq' in name:
                continue
            if 'rotary_emb.cos_cached' in name or 'rotary_emb.sin_cached' in name:
                continue
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                if name.endswith('.bias') and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                for param_name, weight_name, expert_id in expert_params_mapping:
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)
                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    weight_loader(param, loaded_weight, weight_name, expert_id=expert_id)
                    break
                else:
                    if name.endswith('.bias') and name not in params_dict:
                        continue
                    param = params_dict[name]
                    weight_loader = getattr(param, 'weight_loader', default_weight_loader)
                    weight_loader(param, loaded_weight)
        if not global_server_args_dict['disable_mla']:
            for layer_id in range(self.config.num_hidden_layers):
                self_attn = self.model.layers[layer_id].self_attn
                w_kc, w_vc = self_attn.kv_b_proj.weight.unflatten(0, (-1, self_attn.qk_nope_head_dim + self_attn.v_head_dim)).split([self_attn.qk_nope_head_dim, self_attn.v_head_dim], dim=1)
                self_attn.w_kc = w_kc.transpose(1, 2).contiguous().transpose(1, 2)
                self_attn.w_vc = w_vc.contiguous().transpose(1, 2)
                if hasattr(self_attn.kv_b_proj, 'weight_scale'):
                    self_attn.w_scale = self_attn.kv_b_proj.weight_scale
                del self_attn.kv_b_proj


class MixtralMLP(nn.Module):

    def __init__(self, num_experts: 'int', hidden_size: 'int', intermediate_size: 'int', quant_config: 'Optional[QuantizationConfig]'=None) ->None:
        super().__init__()
        self.num_experts = num_experts
        self.ffn_dim = intermediate_size
        self.hidden_dim = hidden_size
        self.w1 = ReplicatedLinear(self.hidden_dim, self.ffn_dim, bias=False, quant_config=quant_config)
        self.w2 = ReplicatedLinear(self.ffn_dim, self.hidden_dim, bias=False, quant_config=quant_config)
        self.w3 = ReplicatedLinear(self.hidden_dim, self.ffn_dim, bias=False, quant_config=quant_config)
        self.act_fn = nn.SiLU()

    def forward(self, hidden_states: 'torch.Tensor') ->torch.Tensor:
        w1_out, _ = self.w1(hidden_states)
        w1_out = self.act_fn(w1_out)
        w3_out, _ = self.w3(hidden_states)
        current_hidden_states = w1_out * w3_out
        current_hidden_states, _ = self.w2(current_hidden_states)
        return current_hidden_states


class MixtralMoE(nn.Module):

    def __init__(self, config: 'MixtralConfig', quant_config: 'Optional[QuantizationConfig]'=None):
        super().__init__()
        self.config = config
        self.rank = get_tensor_model_parallel_rank()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.num_total_experts = config.num_local_experts
        self.top_k = config.num_experts_per_tok
        if self.tp_size > self.num_total_experts:
            raise ValueError(f'Tensor parallel size {self.tp_size} is greater than the number of experts {self.num_total_experts}.')
        self.expert_indicies = np.array_split(range(self.num_total_experts), self.tp_size)[self.rank].tolist()
        if not self.expert_indicies:
            raise ValueError(f'Rank {self.rank} has no experts assigned to it.')
        self.experts = nn.ModuleList([(MixtralMLP(self.num_total_experts, config.hidden_size, config.intermediate_size, quant_config=quant_config) if idx in self.expert_indicies else None) for idx in range(self.num_total_experts)])
        self.gate = ReplicatedLinear(config.hidden_size, self.num_total_experts, bias=False, quant_config=None)

    def forward(self, hidden_states: 'torch.Tensor') ->torch.Tensor:
        router_logits, _ = self.gate(hidden_states)
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        final_hidden_states = None
        for expert_idx in self.expert_indicies:
            expert_layer = self.experts[expert_idx]
            expert_mask = selected_experts == expert_idx
            expert_weights = (routing_weights * expert_mask).sum(dim=-1, keepdim=True)
            current_hidden_states = expert_layer(hidden_states).mul_(expert_weights)
            if final_hidden_states is None:
                final_hidden_states = current_hidden_states
            else:
                final_hidden_states.add_(current_hidden_states)
        return tensor_model_parallel_all_reduce(final_hidden_states)


class MixtralAttention(nn.Module):

    def __init__(self, hidden_size: 'int', num_heads: 'int', num_kv_heads: 'int', layer_id: 'int'=0, max_position: 'int'=4096 * 32, rope_theta: 'float'=10000, quant_config: 'Optional[QuantizationConfig]'=None) ->None:
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim ** -0.5
        self.rope_theta = rope_theta
        self.qkv_proj = QKVParallelLinear(hidden_size, self.head_dim, self.total_num_heads, self.total_num_kv_heads, bias=False, quant_config=quant_config)
        self.o_proj = RowParallelLinear(self.total_num_heads * self.head_dim, hidden_size, bias=False, quant_config=quant_config)
        self.rotary_emb = get_rope(self.head_dim, rotary_dim=self.head_dim, max_position=max_position, base=int(self.rope_theta), is_neox_style=True)
        self.attn = RadixAttention(self.num_heads, self.head_dim, self.scaling, num_kv_heads=self.num_kv_heads, layer_id=layer_id)

    def forward(self, positions: 'torch.Tensor', hidden_states: 'torch.Tensor', forward_batch: 'ForwardBatch') ->torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v, forward_batch)
        output, _ = self.o_proj(attn_output)
        return output


class MixtralDecoderLayer(nn.Module):

    def __init__(self, config: 'MixtralConfig', layer_id: 'int'=0, quant_config: 'Optional[QuantizationConfig]'=None) ->None:
        super().__init__()
        self.hidden_size = config.hidden_size
        rope_theta = getattr(config, 'rope_theta', 10000)
        self.self_attn = MixtralAttention(hidden_size=self.hidden_size, num_heads=config.num_attention_heads, max_position=config.max_position_embeddings, num_kv_heads=config.num_key_value_heads, layer_id=layer_id, rope_theta=rope_theta, quant_config=quant_config)
        self.block_sparse_moe = MixtralMoE(config=config, quant_config=quant_config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, positions: 'torch.Tensor', hidden_states: 'torch.Tensor', forward_batch: 'ForwardBatch', residual: 'Optional[torch.Tensor]') ->torch.Tensor:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions=positions, hidden_states=hidden_states, forward_batch=forward_batch)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.block_sparse_moe(hidden_states)
        return hidden_states, residual


class MixtralModel(nn.Module):

    def __init__(self, config: 'MixtralConfig', quant_config: 'Optional[QuantizationConfig]'=None) ->None:
        super().__init__()
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([MixtralDecoderLayer(config, i, quant_config=quant_config) for i in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids: 'torch.Tensor', positions: 'torch.Tensor', forward_batch: 'ForwardBatch', input_embeds: 'torch.Tensor'=None) ->torch.Tensor:
        if input_embeds is None:
            hidden_states = self.embed_tokens(input_ids)
        else:
            hidden_states = input_embeds
        residual = None
        for i in range(len(self.layers)):
            layer = self.layers[i]
            hidden_states, residual = layer(positions, hidden_states, forward_batch, residual)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class MixtralForCausalLM(nn.Module):

    def __init__(self, config: 'MixtralConfig', quant_config: 'Optional[QuantizationConfig]'=None, cache_config=None) ->None:
        super().__init__()
        self.config = config
        self.quant_config = quant_config
        self.torchao_config = global_server_args_dict['torchao_config']
        self.model = MixtralModel(config, quant_config=quant_config, prefix='model')
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        self.logits_processor = LogitsProcessor(config)

    def forward(self, input_ids: 'torch.Tensor', positions: 'torch.Tensor', forward_batch: 'ForwardBatch', input_embeds: 'torch.Tensor'=None) ->torch.Tensor:
        hidden_states = self.model(input_ids, positions, forward_batch, input_embeds)
        return self.logits_processor(input_ids, hidden_states, self.lm_head.weight, forward_batch)

    def load_weights(self, weights: 'Iterable[Tuple[str, torch.Tensor]]'):
        stacked_params_mapping = [('qkv_proj', 'q_proj', 'q'), ('qkv_proj', 'k_proj', 'k'), ('qkv_proj', 'v_proj', 'v')]
        expert_params_mapping = FusedMoE.make_expert_params_mapping(ckpt_gate_proj_name='w1', ckpt_down_proj_name='w2', ckpt_up_proj_name='w3', num_experts=self.config.num_local_experts)
        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            if 'rotary_emb.inv_freq' in name:
                continue
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                if name.endswith('.bias') and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)
                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    weight_loader(param, loaded_weight, name, shard_id=shard_id, expert_id=expert_id)
                    break
                else:
                    if name.endswith('.bias') and name not in params_dict:
                        continue
                    if name.endswith('.kv_scale') and name not in params_dict:
                        continue
                    if name is None:
                        continue
                    param = params_dict[name]
                    weight_loader = getattr(param, 'weight_loader', default_weight_loader)
                    weight_loader(param, loaded_weight)
        apply_torchao_config_(self, params_dict, set(['proj.weight']))


class QuantMixtralForCausalLM(nn.Module):

    def __init__(self, config: 'MixtralConfig', quant_config: 'Optional[QuantizationConfig]'=None, cache_config=None) ->None:
        super().__init__()
        self.config = config
        self.quant_config = quant_config
        self.model = MixtralModel(config, quant_config=quant_config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        self.logits_processor = LogitsProcessor(config)

    @torch.no_grad()
    def forward(self, input_ids: 'torch.Tensor', positions: 'torch.Tensor', forward_batch: 'ForwardBatch', input_embeds: 'torch.Tensor'=None) ->torch.Tensor:
        hidden_states = self.model(input_ids, positions, forward_batch, input_embeds)
        return self.logits_processor(input_ids, hidden_states, self.lm_head.weight, forward_batch)

    def load_weights(self, weights: 'Iterable[Tuple[str, torch.Tensor]]'):
        stacked_params_mapping = [('qkv_proj', 'q_proj', 'q'), ('qkv_proj', 'k_proj', 'k'), ('qkv_proj', 'v_proj', 'v')]
        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            if 'rotary_emb.inv_freq' in name:
                continue
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                if name.endswith('.bias') and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                if name.endswith('.bias') and name not in params_dict:
                    continue
                if 'block_sparse_moe.experts.' in name and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, 'weight_loader', default_weight_loader)
                weight_loader(param, loaded_weight)


class ColumnParallelConv2dPatch(torch.nn.Module):
    """Conv2D Patching layer with model parallelism.
    Column parallel over unfolded input.
    Arguments:
        in_channels: Input channels.
        out_channels: Output channels.
        kernel_size: Size of convolution kernel.
        stride (default 1): Stride for convolution.
        bias (default False): Use bias in Conv2d.
    Input: (bsz, in_channels, width, height)
    Output: (bsz, num_tokens, out_channels)
    """

    def __init__(self, in_channels: 'int', out_channels: 'int', kernel_size: 'Union[int, Tuple[int, int]]', stride: 'Union[int, Tuple[int, int]]', bias: 'bool'=False) ->None:
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = kernel_size, kernel_size
        self._unfold = torch.nn.Unfold(kernel_size=kernel_size, stride=stride)
        self._linear = ColumnParallelLinear(in_channels * kernel_size[0] * kernel_size[1], out_channels, bias=bias)

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        x = self._unfold(x)
        x = x.permute(0, 2, 1)
        x, _ = self._linear(x)
        return x


class MllamaPrecomputedAspectRatioEmbedding(nn.Module):

    def __init__(self, config: 'config_mllama.MllamaVisionConfig', is_gated: 'bool'=True):
        super().__init__()
        self.max_num_tiles = config.max_num_tiles
        self.hidden_size = config.hidden_size
        self.max_aspect_ratio_id = config.max_aspect_ratio_id
        self.is_gated = is_gated
        self.embedding = nn.Embedding(self.max_aspect_ratio_id + 1, self.max_num_tiles * self.hidden_size)
        if is_gated:
            self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, hidden_state: 'torch.Tensor', aspect_ratio_ids: 'torch.Tensor') ->torch.Tensor:
        embeddings = self.embedding(aspect_ratio_ids)
        embeddings = embeddings.reshape(-1, self.max_num_tiles, 1, self.hidden_size)
        if self.is_gated:
            embeddings = embeddings * self.gate.tanh()
        hidden_state = hidden_state + embeddings
        return hidden_state


class MllamaPrecomputedPositionEmbedding(nn.Module):

    def __init__(self, config: 'config_mllama.MllamaVisionConfig'):
        super().__init__()
        self.max_num_tiles = config.max_num_tiles
        self.max_aspect_ratio_id = config.max_aspect_ratio_id
        self.num_patches = (config.image_size // config.patch_size) ** 2 + 1
        self.hidden_size = config.hidden_size
        self.scale = config.hidden_size ** -0.5
        self.gate = nn.Parameter(torch.zeros(1))
        position_embedding = torch.randn(self.num_patches, self.hidden_size)
        self.embedding = nn.Parameter(self.scale * position_embedding)
        self.tile_embedding = nn.Embedding(self.max_aspect_ratio_id + 1, self.max_num_tiles * self.num_patches * self.hidden_size)

    def forward(self, hidden_state: 'torch.Tensor', aspect_ratio_ids: 'torch.Tensor') ->torch.Tensor:
        gated_position_embedding = (1 - self.gate.tanh()) * self.embedding
        hidden_state = hidden_state + gated_position_embedding.view(1, 1, self.num_patches, self.hidden_size)
        tile_position_embedding = self.tile_embedding(aspect_ratio_ids)
        batch_size = hidden_state.shape[0]
        tile_position_embedding = tile_position_embedding.reshape(batch_size, self.max_num_tiles, self.num_patches, self.hidden_size)
        gated_tile_position_embedding = self.gate.tanh() * tile_position_embedding
        hidden_state = hidden_state + gated_tile_position_embedding
        return hidden_state


class MllamaVisionSdpaAttention(nn.Module):

    def __init__(self, config: 'config_mllama.MllamaVisionConfig'):
        super().__init__()
        model_parallel_size = get_tensor_model_parallel_world_size()
        self.embed_dim = config.hidden_size
        self.num_heads = config.attention_heads
        self.head_dim = config.hidden_size // config.attention_heads
        self.num_local_heads = self.num_heads // model_parallel_size
        self.q_size = self.num_local_heads * self.head_dim
        self.kv_size = self.num_local_heads * self.head_dim
        self.qkv_proj = QKVParallelLinear(self.embed_dim, self.head_dim, self.num_heads, bias=False)
        self.o_proj = RowParallelLinear(self.num_heads * self.head_dim, self.embed_dim, bias=False, input_is_parallel=True)

    def forward(self, hidden_state: 'torch.Tensor', attention_mask: 'Optional[torch.Tensor]'=None) ->torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_state)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = q.view(q.shape[0], q.shape[1], self.num_local_heads, self.head_dim).transpose(1, 2)
        k = k.view(k.shape[0], k.shape[1], self.num_local_heads, self.head_dim).transpose(1, 2)
        v = v.view(v.shape[0], v.shape[1], self.num_local_heads, self.head_dim).transpose(1, 2)
        attn_output = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask, dropout_p=0.0)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(attn_output.shape[0], attn_output.shape[1], -1)
        output, _ = self.o_proj(attn_output)
        return output


class MllamaVisionMLP(nn.Module):

    def __init__(self, config, quant_config: 'Optional[QuantizationConfig]'=None):
        super().__init__()
        self.config = config
        self.activation_fn = get_act_fn(config.hidden_act)
        self.fc1 = ColumnParallelLinear(config.hidden_size, config.intermediate_size, bias=True, quant_config=quant_config)
        self.fc2 = RowParallelLinear(config.intermediate_size, config.hidden_size, bias=True, quant_config=quant_config)

    def forward(self, hidden_states: 'torch.Tensor') ->torch.Tensor:
        hidden_states, _ = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states, _ = self.fc2(hidden_states)
        return hidden_states


class MllamaVisionEncoderLayer(nn.Module):

    def __init__(self, config: 'config_mllama.MllamaVisionConfig', is_gated: 'bool'=False):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.attention_heads
        self.is_gated = is_gated
        self.intermediate_size = config.intermediate_size
        self.self_attn = MllamaVisionSdpaAttention(config)
        self.mlp = MllamaVisionMLP(config)
        self.input_layernorm = nn.LayerNorm(self.hidden_size, eps=config.norm_eps)
        self.post_attention_layernorm = nn.LayerNorm(self.hidden_size, eps=config.norm_eps)
        if is_gated:
            self.gate_attn = nn.Parameter(torch.ones(1) * math.pi / 4)
            self.gate_ffn = nn.Parameter(torch.ones(1) * math.pi / 4)

    def forward(self, hidden_state: 'torch.Tensor', attention_mask: 'Optional[torch.Tensor]'=None):
        residual = hidden_state
        hidden_state = self.input_layernorm(hidden_state)
        hidden_state = self.self_attn(hidden_state, attention_mask=attention_mask)
        gate_attn = 1 if not self.is_gated else self.gate_attn.tanh()
        hidden_state = residual + gate_attn * hidden_state
        residual = hidden_state
        hidden_state = self.post_attention_layernorm(hidden_state)
        hidden_state = self.mlp(hidden_state)
        gate_ffn = 1 if not self.is_gated else self.gate_ffn.tanh()
        hidden_state = residual + gate_ffn * hidden_state
        return hidden_state


class MllamaVisionModel(nn.Module):

    def __init__(self, config: 'config_mllama.MllamaVisionConfig'):
        super().__init__()
        self.image_size = config.image_size
        self.patch_size = config.patch_size
        self.max_num_tiles = config.max_num_tiles
        self.hidden_size = config.hidden_size
        self.in_channels = config.num_channels
        self.intermediate_layers_indices = config.intermediate_layers_indices
        self.num_patches = (self.image_size // self.patch_size) ** 2 + 1
        self.scale = config.hidden_size ** -0.5
        self.patch_embedding = ColumnParallelConv2dPatch(in_channels=config.num_channels, out_channels=self.hidden_size, kernel_size=self.patch_size, stride=self.patch_size, bias=False)
        self.class_embedding = nn.Parameter(self.scale * torch.randn(self.hidden_size))
        self.gated_positional_embedding = MllamaPrecomputedPositionEmbedding(config)
        self.pre_tile_positional_embedding = MllamaPrecomputedAspectRatioEmbedding(config, is_gated=True)
        self.post_tile_positional_embedding = MllamaPrecomputedAspectRatioEmbedding(config, is_gated=True)
        self.layernorm_pre = nn.LayerNorm(self.hidden_size)
        self.layernorm_post = nn.LayerNorm(self.hidden_size)
        self.transformer = MllamaVisionEncoder(config, config.num_hidden_layers, is_gated=False, output_hidden_states=config.intermediate_layers_indices)
        self.global_transformer = MllamaVisionEncoder(config, config.num_global_layers, is_gated=True)

    def apply_class_embedding(self, hidden_state: 'torch.Tensor') ->torch.Tensor:
        batch_size, _, hidden_size = hidden_state.shape
        class_embedding = self.class_embedding.expand(batch_size, 1, hidden_size)
        hidden_state = torch.cat([class_embedding, hidden_state], dim=1)
        return hidden_state

    def forward(self, pixel_values: 'torch.Tensor', aspect_ratio_ids: 'torch.Tensor', aspect_ratio_mask: 'torch.Tensor') ->torch.Tensor:
        batch_size, num_concurrent_media, num_tiles, num_channels, height, width = pixel_values.shape
        pixel_values = pixel_values.reshape(batch_size * num_concurrent_media * num_tiles, num_channels, height, width)
        aspect_ratio_ids = aspect_ratio_ids.reshape(batch_size * num_concurrent_media, -1)
        patch_embeds = self.patch_embedding(pixel_values)
        hidden_state = patch_embeds
        hidden_state = ps.get_tp_group().all_gather(hidden_state)
        _, num_patches, dim = hidden_state.shape
        hidden_state = hidden_state.reshape(batch_size * num_concurrent_media, num_tiles, -1, dim)
        hidden_state = self.pre_tile_positional_embedding(hidden_state, aspect_ratio_ids)
        hidden_state = hidden_state.reshape(batch_size * num_concurrent_media * num_tiles, num_patches, dim)
        hidden_state = self.apply_class_embedding(hidden_state)
        num_patches += 1
        hidden_state = hidden_state.reshape(batch_size * num_concurrent_media, num_tiles, num_patches, dim)
        hidden_state = self.gated_positional_embedding(hidden_state, aspect_ratio_ids)
        hidden_state = self.layernorm_pre(hidden_state)
        num_padding_patches = (8 - hidden_state.shape[-2] % 8) % 8
        padding = 0, 0, 0, num_padding_patches
        hidden_state = F.pad(hidden_state, padding, mode='constant', value=0)
        slice_index = -num_padding_patches if num_padding_patches > 0 else None
        attention_mask = aspect_ratio_mask.reshape(batch_size * num_concurrent_media, -1)
        attention_mask = _prepare_aspect_ratio_attention_mask(aspect_ratio_mask=attention_mask, num_patches=self.num_patches, target_length=hidden_state.shape[2], dtype=self.layernorm_pre.weight.dtype)
        hidden_state = hidden_state.view(batch_size * num_concurrent_media, -1, dim)
        output = self.transformer(hidden_state, attention_mask=attention_mask)
        hidden_state, intermediate_hidden_states = output[0], output[1]
        intermediate_hidden_states = torch.stack(intermediate_hidden_states, dim=-1)
        hidden_state = self.layernorm_post(hidden_state)
        hidden_state = hidden_state.reshape(batch_size * num_concurrent_media, num_tiles, num_patches + num_padding_patches, dim)
        hidden_state = self.post_tile_positional_embedding(hidden_state, aspect_ratio_ids)
        hidden_state = hidden_state.reshape(batch_size * num_concurrent_media, num_tiles * (num_patches + num_padding_patches), dim)
        hidden_state = self.global_transformer(hidden_state, attention_mask=attention_mask)[0]
        hidden_state = hidden_state.reshape(batch_size * num_concurrent_media, num_tiles, num_patches + num_padding_patches, dim)
        hidden_state = hidden_state[:, :, :slice_index]
        hidden_state = hidden_state.reshape(batch_size, num_concurrent_media, num_tiles, num_patches, dim)
        intermediate_hidden_states = intermediate_hidden_states.reshape(batch_size * num_concurrent_media, num_tiles, num_patches + num_padding_patches, -1)
        intermediate_hidden_states = intermediate_hidden_states[:, :, :slice_index]
        intermediate_hidden_states = intermediate_hidden_states.reshape(batch_size, num_concurrent_media, num_tiles, num_patches, -1)
        hidden_state = torch.cat([hidden_state, intermediate_hidden_states], dim=-1)
        return hidden_state


class MllamaTextRMSNorm(nn.Module):

    def __init__(self, hidden_size, eps=1e-06):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states

    def extra_repr(self):
        return f'{tuple(self.weight.shape)}, eps={self.variance_epsilon}'


class MllamaTextCrossAttention(nn.Module):

    def __init__(self, config: 'Optional[config_mllama.MllamaTextConfig]'=None, layer_id: 'Optional[int]'=None, quant_config: 'Optional[QuantizationConfig]'=None):
        super().__init__()
        self.config = config
        self.model_parallel_size = get_tensor_model_parallel_world_size()
        self.num_heads = self.config.num_attention_heads
        self.num_local_heads = self.num_heads // self.model_parallel_size
        self.num_key_value_heads = self.config.num_key_value_heads
        self.num_local_key_value_heads = self.num_key_value_heads // self.model_parallel_size
        self.dropout = config.dropout
        self.hidden_size = config.hidden_size
        self.head_dim = config.hidden_size // self.num_heads
        self.layer_id = layer_id
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.q_local_size = self.num_local_heads * self.head_dim
        self.kv_local_size = self.num_local_key_value_heads * self.head_dim
        self.qkv_proj = QKVParallelLinear(self.hidden_size, self.head_dim, self.num_heads, self.num_key_value_heads, bias=False, quant_config=quant_config)
        self.o_proj = RowParallelLinear(self.num_heads * self.head_dim, self.hidden_size, bias=False, input_is_parallel=True, quant_config=quant_config)
        self.q_norm = MllamaTextRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = MllamaTextRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.scaling = self.head_dim ** -0.5
        self.attn = RadixAttention(self.num_local_heads, self.head_dim, self.scaling, self.num_local_key_value_heads, layer_id=layer_id, is_cross_attention=True)

    def forward(self, hidden_states: 'torch.Tensor', attention_mask: 'Optional[torch.Tensor]', cross_attention_states: 'Optional[torch.Tensor]', forward_batch: 'ForwardBatch') ->torch.Tensor:
        qkv_dec, _ = self.qkv_proj(hidden_states)
        q, _, _ = qkv_dec.split([self.q_local_size, self.kv_local_size, self.kv_local_size], dim=-1)
        if cross_attention_states is None:
            k = None
            v = None
        else:
            qkv_enc, _ = self.qkv_proj(cross_attention_states)
            _, k, v = qkv_enc.split([self.q_local_size, self.kv_local_size, self.kv_local_size], dim=-1)
            k = k.view(-1, self.num_local_key_value_heads, self.head_dim)
            v = v.view(-1, self.num_local_key_value_heads, self.head_dim)
            k = self.k_norm(k)
        q = q.view(-1, self.num_local_heads, self.head_dim)
        q = self.q_norm(q)
        output = self.attn(q, k, v, forward_batch)
        out, _ = self.o_proj(output)
        return out


class MllamaCrossAttentionDecoderLayer(torch.nn.Module):
    """Cross-attention transformer block with tanh-gated attention
    and feedforward."""

    def __init__(self, config: 'config_mllama.MllamaTextConfig', layer_id: 'int', quant_config: 'Optional[QuantizationConfig]') ->None:
        super().__init__()
        self.layer_id = layer_id
        self.cross_attn = MllamaTextCrossAttention(config=config, layer_id=layer_id, quant_config=quant_config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.cross_attn_attn_gate = torch.nn.Parameter(torch.zeros(1))
        self.mlp = LlamaMLP(hidden_size=config.hidden_size, intermediate_size=config.intermediate_size, hidden_act=config.hidden_act, quant_config=quant_config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.cross_attn_mlp_gate = torch.nn.Parameter(torch.zeros(1))

    def forward(self, hidden_states: 'torch.Tensor', cross_attention_states: 'torch.Tensor', cross_attention_mask: 'torch.Tensor', full_text_row_masked_out_mask: 'torch.Tensor', forward_batch: 'ForwardBatch') ->torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.cross_attn(hidden_states=hidden_states, attention_mask=cross_attention_mask, cross_attention_states=cross_attention_states, forward_batch=forward_batch)
        hidden_states = full_text_row_masked_out_mask * hidden_states
        hidden_states = residual + self.cross_attn_attn_gate.tanh() * hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = full_text_row_masked_out_mask * hidden_states
        hidden_states = residual + self.cross_attn_mlp_gate.tanh() * hidden_states
        return hidden_states


class OlmoAttention(nn.Module):
    """
    This is the attention block where the output is computed as
    ``Attention(LN(x))`` in ``MLP(LN(x + Attention(LN(x))))``
    (plus another skip connection).
    """

    def __init__(self, config: 'OlmoConfig', layer_id: 'int'=0, quant_config: 'Optional[QuantizationConfig]'=None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        tensor_model_parallel_world_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = config.num_attention_heads
        assert self.hidden_size % self.total_num_heads == 0
        assert self.total_num_heads % tensor_model_parallel_world_size == 0
        self.num_heads = self.total_num_heads // tensor_model_parallel_world_size
        self.head_dim = self.hidden_size // self.total_num_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.clip_qkv = config.clip_qkv
        self.qkv_proj = QKVParallelLinear(self.hidden_size, self.head_dim, self.total_num_heads, bias=config.attention_bias)
        self.rotary_emb = get_rope(self.head_dim, rotary_dim=self.head_dim, max_position=self.max_position_embeddings, base=self.rope_theta)
        self.scaling = self.head_dim ** -0.5
        self.attn = RadixAttention(self.num_heads, self.head_dim, self.scaling, num_kv_heads=self.num_heads, layer_id=layer_id)
        self.o_proj = RowParallelLinear(self.hidden_size, self.hidden_size, bias=config.attention_bias)

    def forward(self, positions: 'torch.Tensor', hidden_states: 'torch.Tensor', forward_batch: 'ForwardBatch') ->torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        if self.clip_qkv is not None:
            qkv.clamp_(min=-self.clip_qkv, max=self.clip_qkv)
        q, k, v = qkv.chunk(chunks=3, dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v, forward_batch)
        output, _ = self.o_proj(attn_output)
        return output


class OlmoMLP(nn.Module):
    """
    This is the MLP block where the output is computed as
    ``MLP(LN(x))`` in ``MLP(LN(x + Attention(LN(x))))``
    (plus another skip connection).
    """

    def __init__(self, config: 'OlmoConfig', quant_config: 'Optional[QuantizationConfig]'=None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_up_proj = MergedColumnParallelLinear(self.hidden_size, [self.intermediate_size] * 2, bias=False, quant_config=quant_config)
        self.act_fn = SiluAndMul()
        self.down_proj = RowParallelLinear(self.intermediate_size, self.hidden_size, bias=False, quant_config=quant_config)

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class OlmoDecoderLayer(nn.Module):
    """
    This is a typical transformer block where the output is
    computed as ``MLP(LN(x + Attention(LN(x))))``
    (plus another skip connection).
    """

    def __init__(self, config: 'OlmoConfig', layer_id: 'int'=0, quant_config: 'Optional[QuantizationConfig]'=None):
        super().__init__()
        self.self_attn = OlmoAttention(config, layer_id, quant_config)
        self.mlp = OlmoMLP(config, quant_config)
        self.input_layernorm = nn.LayerNorm(config.hidden_size, elementwise_affine=False, bias=False)
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size, elementwise_affine=False, bias=False)

    def forward(self, positions: 'torch.Tensor', hidden_states: 'torch.Tensor', forward_batch: 'ForwardBatch') ->Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(positions, hidden_states, forward_batch)
        hidden_states = hidden_states + residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class OlmoModel(nn.Module):

    def __init__(self, config: 'OlmoConfig', quant_config: 'Optional[QuantizationConfig]'=None):
        super().__init__()
        self.config = config
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([OlmoDecoderLayer(config, layer_idx, quant_config) for layer_idx in range(config.num_hidden_layers)])
        self.norm = nn.LayerNorm(config.hidden_size, elementwise_affine=False, bias=False)

    def forward(self, input_ids: 'torch.Tensor', positions: 'torch.Tensor', forward_batch: 'ForwardBatch', input_embeds: 'torch.Tensor'=None) ->torch.Tensor:
        """
        :param input_ids: A tensor of shape `(batch_size, seq_len)`.
        """
        if input_embeds is None:
            hidden_states = self.embed_tokens(input_ids)
        else:
            hidden_states = input_embeds
        for layer_idx, decoder_layer in enumerate(self.layers):
            hidden_states = decoder_layer(positions, hidden_states, forward_batch)
        hidden_states = self.norm(hidden_states)
        return hidden_states


class OlmoForCausalLM(nn.Module):
    """
    Extremely barebones HF model wrapper.
    """

    def __init__(self, config: 'OlmoConfig', cache_config=None, quant_config: 'Optional[QuantizationConfig]'=None):
        super().__init__()
        self.config = config
        self.model = OlmoModel(config, quant_config)
        if config.tie_word_embeddings:
            self.lm_head = self.model.embed_tokens
        else:
            self.unpadded_vocab_size = config.vocab_size
            self.lm_head = ParallelLMHead(self.unpadded_vocab_size, config.hidden_size, org_num_embeddings=config.vocab_size, quant_config=quant_config)
        self.logits_processor = LogitsProcessor(config)

    def forward(self, input_ids: 'torch.Tensor', positions: 'torch.Tensor', forward_batch: 'ForwardBatch', input_embeds: 'torch.Tensor'=None) ->torch.Tensor:
        hidden_states = self.model(input_ids=input_ids, positions=positions, forward_batch=forward_batch, input_embeds=input_embeds)
        return self.logits_processor(input_ids, hidden_states, self.lm_head.weight, forward_batch)

    def load_weights(self, weights: 'Iterable[Tuple[str, torch.Tensor]]'):
        stacked_params_mapping = [('qkv_proj', 'q_proj', 'q'), ('qkv_proj', 'k_proj', 'k'), ('qkv_proj', 'v_proj', 'v'), ('gate_up_proj', 'gate_proj', 0), ('gate_up_proj', 'up_proj', 1)]
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        for name, loaded_weight in weights:
            if 'rotary_emb.inv_freq' in name:
                continue
            if 'rotary_emb.cos_cached' in name or 'rotary_emb.sin_cached' in name:
                continue
            if self.config.tie_word_embeddings and 'lm_head.weight' in name:
                continue
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                if name.endswith('.bias') and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                if name.endswith('.bias') and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, 'weight_loader', default_weight_loader)
                weight_loader(param, loaded_weight)


class OlmoeMoE(nn.Module):
    """A tensor-parallel MoE implementation for Olmoe that shards each expert
    across all ranks.

    Each expert's weights are sharded across all ranks and a fused MoE
    kernel is used for the forward pass, and finally we reduce the outputs
    across ranks.
    """

    def __init__(self, num_experts: 'int', top_k: 'int', hidden_size: 'int', intermediate_size: 'int', params_dtype: 'Optional[torch.dtype]'=None, quant_config: 'Optional[QuantizationConfig]'=None, tp_size: 'Optional[int]'=None, prefix: 'str'=''):
        super().__init__()
        self.hidden_size = hidden_size
        self.gate = ReplicatedLinear(hidden_size, num_experts, bias=False, quant_config=None)
        self.experts = FusedMoE(num_experts=num_experts, top_k=top_k, hidden_size=hidden_size, intermediate_size=intermediate_size, reduce_results=True, renormalize=False, quant_config=quant_config, tp_size=tp_size)

    def forward(self, hidden_states: 'torch.Tensor') ->torch.Tensor:
        orig_shape = hidden_states.shape
        hidden_states = hidden_states.view(-1, self.hidden_size)
        router_logits, _ = self.gate(hidden_states)
        final_hidden_states = self.experts(hidden_states=hidden_states, router_logits=router_logits)
        return final_hidden_states.view(orig_shape)


class OlmoeAttention(nn.Module):

    def __init__(self, layer_id: 'int', hidden_size: 'int', num_heads: 'int', num_kv_heads: 'int', rope_theta: 'float'=10000, rope_scaling: 'Optional[Dict[str, Any]]'=None, max_position_embeddings: 'int'=4096, quant_config: 'Optional[QuantizationConfig]'=None) ->None:
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim ** -0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
        self.qkv_proj = QKVParallelLinear(hidden_size, self.head_dim, self.total_num_heads, self.total_num_kv_heads, bias=False, quant_config=quant_config)
        self.q_norm = RMSNorm(hidden_size, eps=1e-05)
        self.k_norm = RMSNorm(hidden_size, eps=1e-05)
        self.o_proj = RowParallelLinear(self.total_num_heads * self.head_dim, hidden_size, bias=False, quant_config=quant_config)
        self.rotary_emb = get_rope(self.head_dim, rotary_dim=self.head_dim, max_position=max_position_embeddings, base=rope_theta, rope_scaling=rope_scaling, is_neox_style=True)
        self.attn = RadixAttention(self.num_heads, self.head_dim, self.scaling, layer_id=layer_id, num_kv_heads=self.num_kv_heads)

    def forward(self, positions: 'torch.Tensor', hidden_states: 'torch.Tensor', forward_batch: 'ForwardBatch') ->torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.q_norm(q.contiguous()), self.k_norm(k.contiguous())
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v, forward_batch)
        output, _ = self.o_proj(attn_output)
        return output


class OlmoeDecoderLayer(nn.Module):

    def __init__(self, config: 'PretrainedConfig', layer_id: 'int'=0, quant_config: 'Optional[QuantizationConfig]'=None) ->None:
        super().__init__()
        self.hidden_size = config.hidden_size
        rope_theta = getattr(config, 'rope_theta', 10000)
        rope_scaling = getattr(config, 'rope_scaling', None)
        max_position_embeddings = getattr(config, 'max_position_embeddings', 4096)
        self.self_attn = OlmoeAttention(layer_id, hidden_size=self.hidden_size, num_heads=config.num_attention_heads, num_kv_heads=config.num_key_value_heads, rope_theta=rope_theta, rope_scaling=rope_scaling, max_position_embeddings=max_position_embeddings, quant_config=quant_config)
        self.mlp = OlmoeMoE(num_experts=config.num_experts, top_k=config.num_experts_per_tok, hidden_size=config.hidden_size, intermediate_size=config.intermediate_size, quant_config=quant_config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=1e-05)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=1e-05)

    def forward(self, positions: 'torch.Tensor', hidden_states: 'torch.Tensor', forward_batch: 'ForwardBatch', residual: 'Optional[torch.Tensor]') ->torch.Tensor:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions=positions, hidden_states=hidden_states, forward_batch=forward_batch)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class OlmoeModel(nn.Module):

    def __init__(self, config: 'PretrainedConfig', quant_config: 'Optional[QuantizationConfig]'=None) ->None:
        super().__init__()
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([OlmoeDecoderLayer(config, layer_id, quant_config=quant_config) for layer_id in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=1e-05)

    def forward(self, input_ids: 'torch.Tensor', positions: 'torch.Tensor', forward_batch: 'ForwardBatch', input_embeds: 'torch.Tensor'=None) ->torch.Tensor:
        if input_embeds is None:
            hidden_states = self.embed_tokens(input_ids)
        else:
            hidden_states = input_embeds
        residual = None
        for i in range(len(self.layers)):
            layer = self.layers[i]
            hidden_states, residual = layer(positions, hidden_states, forward_batch, residual)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class OlmoeForCausalLM(nn.Module):
    fall_back_to_pt_during_load = False

    def __init__(self, config: 'PretrainedConfig', cache_config=None, quant_config: 'Optional[QuantizationConfig]'=None) ->None:
        super().__init__()
        self.config = config
        self.quant_config = quant_config
        self.model = OlmoeModel(config, quant_config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size, quant_config=quant_config)
        self.logits_processor = LogitsProcessor(config)

    def forward(self, input_ids: 'torch.Tensor', positions: 'torch.Tensor', forward_batch: 'ForwardBatch', input_embeds: 'torch.Tensor'=None) ->torch.Tensor:
        hidden_states = self.model(input_ids, positions, forward_batch, input_embeds)
        return self.logits_processor(input_ids, hidden_states, self.lm_head.weight, forward_batch)

    def load_weights(self, weights: 'Iterable[Tuple[str, torch.Tensor]]'):
        stacked_params_mapping = [('qkv_proj', 'q_proj', 'q'), ('qkv_proj', 'k_proj', 'k'), ('qkv_proj', 'v_proj', 'v'), ('gate_up_proj', 'gate_proj', 0), ('gate_up_proj', 'up_proj', 1)]
        expert_params_mapping = FusedMoE.make_expert_params_mapping(ckpt_gate_proj_name='gate_proj', ckpt_down_proj_name='down_proj', ckpt_up_proj_name='up_proj', num_experts=self.config.num_experts)
        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            if 'rotary_emb.inv_freq' in name:
                continue
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                if 'mlp.experts' in name:
                    continue
                name = name.replace(weight_name, param_name)
                if name.endswith('.bias') and name not in params_dict:
                    continue
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)
                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    weight_loader(param, loaded_weight, name, shard_id=shard_id, expert_id=expert_id)
                    break
                else:
                    if name.endswith('.bias') and name not in params_dict:
                        continue
                    if name.endswith('kv_scale'):
                        remapped_kv_scale_name = name.replace('.kv_scale', '.attn.kv_scale')
                        if remapped_kv_scale_name not in params_dict:
                            print_warning_once(f'Found kv scale in the checkpoint (e.g. {name}), but not found the expected name in the model (e.g. {remapped_kv_scale_name}). kv-scale is not loaded.')
                            continue
                        else:
                            name = remapped_kv_scale_name
                    param = params_dict[name]
                    weight_loader = getattr(param, 'weight_loader', default_weight_loader)
                    weight_loader(param, loaded_weight)


class QWenMLP(nn.Module):

    def __init__(self, hidden_size: 'int', intermediate_size: 'int', hidden_act: 'str'='silu', quant_config: 'Optional[QuantizationConfig]'=None):
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(hidden_size, 2 * [intermediate_size], bias=False, gather_output=False, quant_config=quant_config)
        self.c_proj = RowParallelLinear(intermediate_size, hidden_size, bias=False, input_is_parallel=True, quant_config=quant_config)
        if hidden_act != 'silu':
            raise ValueError(f'Unsupported activation: {hidden_act}. Only silu is supported for now.')
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.c_proj(x)
        return x


class QWenAttention(nn.Module):

    def __init__(self, hidden_size: 'int', num_heads: 'int', max_position_embeddings: 'int', layer_id: 'int'=0, rope_theta: 'float'=10000, rope_scaling: 'Optional[Dict[str, Any]]'=None, quant_config: 'Optional[QuantizationConfig]'=None):
        super().__init__()
        self.hidden_size = hidden_size
        tensor_model_parallel_world_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tensor_model_parallel_world_size == 0
        self.num_heads = self.total_num_heads // tensor_model_parallel_world_size
        self.head_dim = hidden_size // self.total_num_heads
        self.c_attn = QKVParallelLinear(hidden_size, self.head_dim, self.total_num_heads, bias=True, quant_config=quant_config)
        self.c_proj = RowParallelLinear(self.total_num_heads * self.head_dim, hidden_size, bias=False, input_is_parallel=True, quant_config=quant_config)
        self.rotary_emb = get_rope(self.head_dim, rotary_dim=self.head_dim, max_position=max_position_embeddings, base=rope_theta, rope_scaling=rope_scaling)
        self.scaling = self.head_dim ** -0.5
        self.attn = RadixAttention(self.num_heads, self.head_dim, self.scaling, num_kv_heads=self.num_heads, layer_id=layer_id)

    def forward(self, positions: 'torch.Tensor', hidden_states: 'torch.Tensor', forward_batch: 'ForwardBatch') ->torch.Tensor:
        qkv, _ = self.c_attn(hidden_states)
        q, k, v = qkv.chunk(chunks=3, dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v, forward_batch)
        output, _ = self.c_proj(attn_output)
        return output


class QWenBlock(nn.Module):

    def __init__(self, config: 'PretrainedConfig', layer_id, quant_config: 'Optional[QuantizationConfig]'=None):
        super().__init__()
        self.ln_1 = RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        rope_theta = getattr(config, 'rope_theta', 10000)
        rope_scaling = getattr(config, 'rope_scaling', None)
        self.attn = QWenAttention(config.hidden_size, config.num_attention_heads, config.max_position_embeddings, rope_theta=rope_theta, rope_scaling=rope_scaling, layer_id=layer_id, quant_config=quant_config)
        self.ln_2 = RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.mlp = QWenMLP(config.hidden_size, config.intermediate_size // 2, quant_config=quant_config)

    def forward(self, positions: 'torch.Tensor', hidden_states: 'torch.Tensor', forward_batch: 'ForwardBatch') ->torch.Tensor:
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        hidden_states = self.attn(positions=positions, hidden_states=hidden_states, forward_batch=forward_batch)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class QWenModel(nn.Module):

    def __init__(self, config: 'PretrainedConfig', quant_config: 'Optional[QuantizationConfig]'=None):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        vocab_size = (config.vocab_size + 63) // 64 * 64
        self.wte = VocabParallelEmbedding(vocab_size, config.hidden_size)
        self.h = nn.ModuleList([QWenBlock(config, i, quant_config=quant_config) for i in range(config.num_hidden_layers)])
        self.ln_f = RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)

    def forward(self, input_ids: 'torch.Tensor', positions: 'torch.Tensor', forward_batch: 'ForwardBatch') ->torch.Tensor:
        hidden_states = self.wte(input_ids)
        for i in range(len(self.h)):
            layer = self.h[i]
            hidden_states = layer(positions, hidden_states, forward_batch)
        hidden_states = self.ln_f(hidden_states)
        return hidden_states


class QWenLMHeadModel(nn.Module):

    def __init__(self, config: 'PretrainedConfig', quant_config: 'Optional[QuantizationConfig]'=None, cache_config=None):
        super().__init__()
        self.config = config
        self.transformer = QWenModel(config, quant_config=quant_config)
        vocab_size = (config.vocab_size + 63) // 64 * 64
        self.lm_head = ParallelLMHead(vocab_size, config.hidden_size)
        self.logits_processor = LogitsProcessor(config)

    @torch.no_grad()
    def forward(self, input_ids: 'torch.Tensor', positions: 'torch.Tensor', forward_batch: 'ForwardBatch'):
        hidden_states = self.transformer(input_ids, positions, forward_batch)
        return self.logits_processor(input_ids, hidden_states, self.lm_head.weight, forward_batch)

    def load_weights(self, weights: 'Iterable[Tuple[str, torch.Tensor]]'):
        stacked_params_mapping = [('gate_up_proj', 'w2', 0), ('gate_up_proj', 'w1', 1)]
        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            if 'rotary_emb.inv_freq' in name:
                continue
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                if name.endswith('.bias') and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                if name.endswith('.bias') and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, 'weight_loader', default_weight_loader)
                weight_loader(param, loaded_weight)


class Qwen2MoeMLP(nn.Module):

    def __init__(self, hidden_size: 'int', intermediate_size: 'int', hidden_act: 'str', quant_config: 'Optional[QuantizationConfig]'=None, reduce_results: 'bool'=True) ->None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(hidden_size, [intermediate_size] * 2, bias=False, quant_config=quant_config)
        self.down_proj = RowParallelLinear(intermediate_size, hidden_size, bias=False, quant_config=quant_config, reduce_results=reduce_results)
        if hidden_act != 'silu':
            raise ValueError(f'Unsupported activation: {hidden_act}. Only silu is supported for now.')
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class Qwen2MoeSparseMoeBlock(nn.Module):

    def __init__(self, config: 'PretrainedConfig', quant_config: 'Optional[QuantizationConfig]'=None):
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        if self.tp_size > config.num_experts:
            raise ValueError(f'Tensor parallel size {self.tp_size} is greater than the number of experts {config.num_experts}.')
        self.experts = FusedMoE(num_experts=config.num_experts, top_k=config.num_experts_per_tok, hidden_size=config.hidden_size, intermediate_size=config.moe_intermediate_size, reduce_results=False, renormalize=config.norm_topk_prob, quant_config=quant_config)
        self.gate = ReplicatedLinear(config.hidden_size, config.num_experts, bias=False, quant_config=None)
        if config.shared_expert_intermediate_size > 0:
            self.shared_expert = Qwen2MoeMLP(hidden_size=config.hidden_size, intermediate_size=config.shared_expert_intermediate_size, hidden_act=config.hidden_act, quant_config=quant_config, reduce_results=False)
        else:
            self.shared_expert = None
        self.shared_expert_gate = torch.nn.Linear(config.hidden_size, 1, bias=False)

    def forward(self, hidden_states: 'torch.Tensor') ->torch.Tensor:
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        shared_output = None
        if self.shared_expert is not None:
            shared_output = self.shared_expert(hidden_states)
            if self.shared_expert_gate is not None:
                shared_output = F.sigmoid(self.shared_expert_gate(hidden_states)) * shared_output
        router_logits, _ = self.gate(hidden_states)
        final_hidden_states = self.experts(hidden_states=hidden_states, router_logits=router_logits)
        if shared_output is not None:
            final_hidden_states = final_hidden_states + shared_output
        if self.tp_size > 1:
            final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states)
        return final_hidden_states.view(num_tokens, hidden_dim)


class Qwen2MoeAttention(nn.Module):

    def __init__(self, hidden_size: 'int', num_heads: 'int', num_kv_heads: 'int', layer_id: 'int'=0, rope_theta: 'float'=10000, rope_scaling: 'Optional[Dict[str, Any]]'=None, max_position_embeddings: 'int'=8192, cache_config=None, quant_config: 'Optional[QuantizationConfig]'=None) ->None:
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim ** -0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
        self.qkv_proj = QKVParallelLinear(hidden_size, self.head_dim, self.total_num_heads, self.total_num_kv_heads, bias=True, quant_config=quant_config)
        self.o_proj = RowParallelLinear(self.total_num_heads * self.head_dim, hidden_size, bias=False, quant_config=quant_config)
        self.rotary_emb = get_rope(self.head_dim, rotary_dim=self.head_dim, max_position=max_position_embeddings, base=rope_theta, rope_scaling=rope_scaling)
        self.attn = RadixAttention(self.num_heads, self.head_dim, self.scaling, num_kv_heads=self.num_kv_heads, layer_id=layer_id)

    def forward(self, positions: 'torch.Tensor', hidden_states: 'torch.Tensor', forward_batch: 'ForwardBatch') ->torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v, forward_batch)
        output, _ = self.o_proj(attn_output)
        return output


class Qwen2MoeDecoderLayer(nn.Module):

    def __init__(self, config: 'PretrainedConfig', layer_id: 'int', cache_config=None, quant_config: 'Optional[QuantizationConfig]'=None) ->None:
        super().__init__()
        self.hidden_size = config.hidden_size
        rope_theta = getattr(config, 'rope_theta', 10000)
        rope_scaling = getattr(config, 'rope_scaling', None)
        max_position_embeddings = getattr(config, 'max_position_embeddings', 8192)
        self.self_attn = Qwen2MoeAttention(hidden_size=self.hidden_size, num_heads=config.num_attention_heads, num_kv_heads=config.num_key_value_heads, layer_id=layer_id, rope_theta=rope_theta, rope_scaling=rope_scaling, max_position_embeddings=max_position_embeddings, cache_config=cache_config, quant_config=quant_config)
        mlp_only_layers = [] if not hasattr(config, 'mlp_only_layers') else config.mlp_only_layers
        if layer_id not in mlp_only_layers and (config.num_experts > 0 and (layer_id + 1) % config.decoder_sparse_step == 0):
            self.mlp = Qwen2MoeSparseMoeBlock(config=config, quant_config=quant_config)
        else:
            self.mlp = Qwen2MoeMLP(hidden_size=config.hidden_size, intermediate_size=config.intermediate_size, hidden_act=config.hidden_act, quant_config=quant_config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, positions: 'torch.Tensor', hidden_states: 'torch.Tensor', forward_batch: 'ForwardBatch', residual: 'Optional[torch.Tensor]') ->torch.Tensor:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions=positions, hidden_states=hidden_states, forward_batch=forward_batch)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class Qwen2MoeModel(nn.Module):

    def __init__(self, config: 'PretrainedConfig', cache_config=None, quant_config: 'Optional[QuantizationConfig]'=None) ->None:
        super().__init__()
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([Qwen2MoeDecoderLayer(config, layer_id, cache_config, quant_config=quant_config) for layer_id in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids: 'torch.Tensor', positions: 'torch.Tensor', forward_batch: 'ForwardBatch', input_embeds: 'torch.Tensor'=None) ->torch.Tensor:
        if input_embeds is None:
            hidden_states = self.embed_tokens(input_ids)
        else:
            hidden_states = input_embeds
        residual = None
        for i in range(len(self.layers)):
            layer = self.layers[i]
            hidden_states, residual = layer(positions, hidden_states, forward_batch, residual)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class Qwen2MoeForCausalLM(nn.Module):
    fall_back_to_pt_during_load = False

    def __init__(self, config: 'PretrainedConfig', cache_config=None, quant_config: 'Optional[QuantizationConfig]'=None) ->None:
        super().__init__()
        self.config = config
        self.quant_config = quant_config
        self.torchao_config = global_server_args_dict['torchao_config']
        self.model = Qwen2MoeModel(config, cache_config, quant_config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size, quant_config=quant_config)
        self.logits_processor = LogitsProcessor(config)

    @torch.no_grad()
    def forward(self, input_ids: 'torch.Tensor', positions: 'torch.Tensor', forward_batch: 'ForwardBatch', input_embeds: 'torch.Tensor'=None) ->torch.Tensor:
        hidden_states = self.model(input_ids, positions, forward_batch, input_embeds)
        return self.logits_processor(input_ids, hidden_states, self.lm_head.weight, forward_batch)

    def load_weights(self, weights: 'Iterable[Tuple[str, torch.Tensor]]'):
        stacked_params_mapping = [('qkv_proj', 'q_proj', 'q'), ('qkv_proj', 'k_proj', 'k'), ('qkv_proj', 'v_proj', 'v'), ('gate_up_proj', 'gate_proj', 0), ('gate_up_proj', 'up_proj', 1)]
        expert_params_mapping = FusedMoE.make_expert_params_mapping(ckpt_gate_proj_name='gate_proj', ckpt_down_proj_name='down_proj', ckpt_up_proj_name='up_proj', num_experts=self.config.num_experts)
        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            if 'rotary_emb.inv_freq' in name:
                continue
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                if 'mlp.experts' in name:
                    continue
                name = name.replace(weight_name, param_name)
                if name.endswith('.bias') and name not in params_dict:
                    continue
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)
                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    weight_loader(param, loaded_weight, name, shard_id=shard_id, expert_id=expert_id)
                    break
                else:
                    if name.endswith('.bias') and name not in params_dict:
                        continue
                    if name not in params_dict:
                        continue
                    param = params_dict[name]
                    weight_loader = getattr(param, 'weight_loader', default_weight_loader)
                    weight_loader(param, loaded_weight)
        apply_torchao_config_(self, params_dict, set(['proj.weight']))


def rotate_half(x: 'torch.Tensor', interleaved: 'bool'=False) ->torch.Tensor:
    if not interleaved:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    else:
        x1, x2 = x[..., ::2], x[..., 1::2]
        return rearrange(torch.stack((-x2, x1), dim=-1), '... d two -> ... (d two)', two=2)


def apply_rotary_emb_torch(x: 'torch.Tensor', cos: 'torch.Tensor', sin: 'torch.Tensor', interleaved: 'bool'=False) ->torch.Tensor:
    """
    x: (batch_size, seqlen, nheads, headdim)
    cos, sin: (seqlen, rotary_dim / 2) or (batch_size, seqlen, rotary_dim / 2)
    """
    ro_dim = cos.shape[-1] * 2
    assert ro_dim <= x.shape[-1]
    cos = repeat(cos, '... d -> ... 1 (2 d)' if not interleaved else '... d -> ... 1 (d 2)')
    sin = repeat(sin, '... d -> ... 1 (2 d)' if not interleaved else '... d -> ... 1 (d 2)')
    return torch.cat([x[..., :ro_dim] * cos + rotate_half(x[..., :ro_dim], interleaved) * sin, x[..., ro_dim:]], dim=-1)


def apply_rotary_pos_emb_vision(t: 'torch.Tensor', freqs: 'torch.Tensor') ->torch.Tensor:
    t_ = t.float()
    cos = freqs.cos()
    sin = freqs.sin()
    output = apply_rotary_emb_torch(t_, cos, sin).type_as(t)
    return output


is_cuda_available = torch.cuda.is_available()


def context_attention_fwd(q, k, v, o, b_start_loc, b_seq_len, max_input_len, is_causal=True):
    if is_cuda_available and CUDA_CAPABILITY[0] > 8:
        BLOCK = 128
    else:
        BLOCK = 64
    Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
    sm_scale = 1.0 / Lq ** 0.5
    batch, head = b_seq_len.shape[0], q.shape[1]
    kv_group_num = q.shape[1] // k.shape[1]
    grid = batch, head, triton.cdiv(max_input_len, BLOCK)
    num_warps = 4 if Lk <= 64 else 8
    _fwd_kernel[grid](q, k, v, sm_scale, b_start_loc, b_seq_len, o, q.stride(0), q.stride(1), k.stride(0), k.stride(1), v.stride(0), v.stride(1), o.stride(0), o.stride(1), kv_group_num=kv_group_num, BLOCK_M=BLOCK, BLOCK_DMODEL=triton.next_power_of_2(Lk), BLOCK_N=BLOCK, IS_CAUSAL=is_causal, num_warps=num_warps, num_stages=1, Lk=Lk)


class Qwen2VisionAttention(nn.Module):

    def __init__(self, embed_dim: 'Optional[int]'=None, num_heads: 'Optional[int]'=None, projection_size: 'Optional[int]'=None, quant_config: 'Optional[QuantizationConfig]'=None) ->None:
        super().__init__()
        world_size = parallel_state.get_tensor_model_parallel_world_size()
        self.hidden_size_per_attention_head = dist_utils.divide(projection_size, num_heads)
        self.num_attention_heads_per_partition = dist_utils.divide(num_heads, world_size)
        self.qkv = ColumnParallelLinear(input_size=embed_dim, output_size=3 * projection_size, quant_config=quant_config)
        self.proj = RowParallelLinear(input_size=projection_size, output_size=embed_dim, quant_config=quant_config)

    def forward(self, x: 'torch.Tensor', cu_seqlens: 'torch.Tensor', rotary_pos_emb: 'torch.Tensor'=None) ->torch.Tensor:
        x, _ = self.qkv(x)
        new_x_shape = x.size()[:-1] + (self.num_attention_heads_per_partition, 3 * self.hidden_size_per_attention_head)
        x = x.view(*new_x_shape)
        q, k, v = dist_utils.split_tensor_along_last_dim(x, 3)
        batch_size = q.shape[1]
        q, k, v = [rearrange(x, 's b ... -> b s ...').contiguous() for x in (q, k, v)]
        if rotary_pos_emb is not None:
            q = apply_rotary_pos_emb_vision(q, rotary_pos_emb)
            k = apply_rotary_pos_emb_vision(k, rotary_pos_emb)
        seq_lens = cu_seqlens[1:] - cu_seqlens[:-1]
        max_seqlen = seq_lens.max().item()
        q, k, v = [rearrange(x, 'b s ... -> (b s) ...') for x in [q, k, v]]
        output = torch.empty_like(q)
        context_attention_fwd(q, k, v, output, cu_seqlens, seq_lens, max_seqlen, is_causal=False)
        context_layer = rearrange(output, '(b s) ... -> b s ...', b=batch_size)
        context_layer = rearrange(context_layer, 'b s h d -> s b (h d)').contiguous()
        output, _ = self.proj(context_layer)
        return output


class Qwen2VisionPatchEmbed(nn.Module):

    def __init__(self, patch_size: 'int'=14, temporal_patch_size: 'int'=2, in_chans: 'int'=3, embed_dim: 'int'=1152) ->None:
        super().__init__()
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.embed_dim = embed_dim
        kernel_size = [temporal_patch_size, patch_size, patch_size]
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=kernel_size, stride=kernel_size, bias=False)

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        L, C = x.shape
        x = x.view(L, -1, self.temporal_patch_size, self.patch_size, self.patch_size)
        x = self.proj(x).view(L, self.embed_dim)
        return x


class Qwen2VisionPatchMerger(nn.Module):

    def __init__(self, d_model: 'int', context_dim: 'int', norm_layer: 'Type[nn.Module]'=None, spatial_merge_size: 'int'=2, quant_config: 'Optional[QuantizationConfig]'=None) ->None:
        super().__init__()
        self.hidden_size = context_dim * spatial_merge_size ** 2
        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-06)
        self.ln_q = norm_layer(context_dim)
        self.mlp = nn.ModuleList([ColumnParallelLinear(self.hidden_size, self.hidden_size, bias=True, quant_config=quant_config), nn.GELU(), RowParallelLinear(self.hidden_size, d_model, bias=True, quant_config=quant_config)])

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        x = self.ln_q(x)
        x = x.view(-1, self.hidden_size)
        mlp_fc1, mlp_act, mlp_fc2 = self.mlp
        x_parallel, _ = mlp_fc1(x)
        x_parallel = mlp_act(x_parallel)
        out, _ = mlp_fc2(x_parallel)
        return out


class Qwen2VisionRotaryEmbedding(nn.Module):

    def __init__(self, dim: 'int', theta: 'float'=10000.0) ->None:
        super().__init__()
        self.dim = dim
        self.theta = theta
        inv_freq = 1.0 / theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim)
        self.register_buffer('inv_freq', inv_freq, persistent=False)
        self._seq_len_cached = 0
        self._freqs_cached = None

    def update_freqs_cache(self, seqlen: 'int') ->None:
        if seqlen > self._seq_len_cached:
            seqlen *= 2
            self._seq_len_cached = seqlen
            self.inv_freq = 1.0 / self.theta ** (torch.arange(0, self.dim, 2, dtype=torch.float, device=self.inv_freq.device) / self.dim)
            seq = torch.arange(seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(seq, self.inv_freq)
            self._freqs_cached = freqs

    def forward(self, seqlen: 'int') ->torch.Tensor:
        self.update_freqs_cache(seqlen)
        return self._freqs_cached[:seqlen]


class Qwen2VisionTransformer(nn.Module):

    def __init__(self, vision_config: 'Qwen2VLVisionConfig', norm_eps: 'float'=1e-06, quant_config: 'Optional[QuantizationConfig]'=None) ->None:
        super().__init__()
        patch_size: 'int' = vision_config.patch_size
        temporal_patch_size: 'int' = vision_config.temporal_patch_size
        spatial_merge_size: 'int' = vision_config.spatial_merge_size
        in_chans: 'int' = vision_config.in_chans
        hidden_size: 'int' = vision_config.hidden_size
        embed_dim: 'int' = vision_config.embed_dim
        depth: 'int' = vision_config.depth
        num_heads: 'int' = vision_config.num_heads
        mlp_ratio: 'float' = vision_config.mlp_ratio
        self.spatial_merge_size = spatial_merge_size
        self.patch_embed = Qwen2VisionPatchEmbed(patch_size=patch_size, temporal_patch_size=temporal_patch_size, in_chans=in_chans, embed_dim=embed_dim)
        norm_layer = partial(nn.LayerNorm, eps=norm_eps)
        head_dim = embed_dim // num_heads
        self.rotary_pos_emb = Qwen2VisionRotaryEmbedding(head_dim // 2)
        self.blocks = nn.ModuleList([Qwen2VisionBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, norm_layer=norm_layer, quant_config=quant_config) for _ in range(depth)])
        self.merger = Qwen2VisionPatchMerger(d_model=hidden_size, context_dim=embed_dim, norm_layer=norm_layer, quant_config=quant_config)

    @property
    def dtype(self) ->torch.dtype:
        return self.blocks[0].mlp.fc2.weight.dtype

    @property
    def device(self) ->torch.device:
        return self.blocks[0].mlp.fc2.weight.device

    def rot_pos_emb(self, grid_thw: 'torch.Tensor') ->torch.Tensor:
        pos_ids = []
        for t, h, w in grid_thw:
            hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
            wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
            hpos_ids = hpos_ids.reshape(h // self.spatial_merge_size, self.spatial_merge_size, w // self.spatial_merge_size, self.spatial_merge_size).permute(0, 2, 1, 3).flatten()
            wpos_ids = wpos_ids.reshape(h // self.spatial_merge_size, self.spatial_merge_size, w // self.spatial_merge_size, self.spatial_merge_size).permute(0, 2, 1, 3).flatten()
            pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))
        pos_ids = torch.cat(pos_ids, dim=0)
        max_grid_size = grid_thw[:, 1:].max()
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
        rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(1)
        return rotary_pos_emb

    def forward(self, x: 'torch.Tensor', grid_thw: 'torch.Tensor') ->torch.Tensor:
        x = x
        x = self.patch_embed(x)
        rotary_pos_emb = self.rot_pos_emb(grid_thw)
        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(dim=0, dtype=torch.int32)
        cu_seqlens = F.pad(cu_seqlens, (1, 0), 'constant', 0)
        x = x.unsqueeze(1)
        for blk in self.blocks:
            x = blk(x, cu_seqlens=cu_seqlens, rotary_pos_emb=rotary_pos_emb)
        x = self.merger(x)
        return x


class Qwen2VLImageInputs(TypedDict):
    pixel_values: 'torch.Tensor'
    """Shape:
    `(num_patches, num_channels * patch_size * patch_size)`
    """
    image_grid_thw: 'torch.Tensor'
    """Shape: `(num_images, 3)`

    This should be in `(grid_t, grid_h, grid_w)` format.
    """


def attach_additional_stop_token_ids(tokenizer):
    if '<|eom_id|>' in tokenizer.get_added_vocab():
        tokenizer.additional_stop_token_ids = set([tokenizer.get_added_vocab()['<|eom_id|>']])
    else:
        tokenizer.additional_stop_token_ids = None


def get_processor(tokenizer_name: 'str', *args, tokenizer_mode: str='auto', trust_remote_code: bool=False, tokenizer_revision: Optional[str]=None, **kwargs):
    processor = AutoProcessor.from_pretrained(tokenizer_name, *args, trust_remote_code=trust_remote_code, tokenizer_revision=tokenizer_revision, **kwargs)
    attach_additional_stop_token_ids(processor.tokenizer)
    return processor


cached_get_processor = lru_cache(get_processor)


class Qwen2VLForConditionalGeneration(nn.Module):

    def calculate_num_image_tokens(self, image_grid_thw: 'Tuple[int, int, int]'):
        processor = cached_get_processor(self.config._name_or_path)
        grid_t, grid_h, grid_w = image_grid_thw
        num_image_tokens = grid_t * grid_h * grid_w // processor.image_processor.merge_size // processor.image_processor.merge_size
        return num_image_tokens

    def pad_input_ids(self, input_ids: 'List[int]', image_inputs: 'ImageInputs'):
        image_grid_thws = image_inputs.image_grid_thws
        pad_values = image_inputs.pad_values
        image_indices = [idx for idx, token in enumerate(input_ids) if token == self.config.image_token_id]
        image_inputs.image_offsets = []
        input_ids_with_image = []
        for image_cnt, _ in enumerate(image_grid_thws):
            num_image_tokens = self.calculate_num_image_tokens(image_grid_thws[image_cnt])
            if image_cnt == 0:
                non_image_tokens = input_ids[:image_indices[image_cnt]]
            else:
                non_image_tokens = input_ids[image_indices[image_cnt - 1] + 1:image_indices[image_cnt]]
            input_ids_with_image.extend(non_image_tokens)
            image_inputs.image_offsets.append(len(input_ids_with_image))
            pad_ids = pad_values * ((num_image_tokens + len(pad_values)) // len(pad_values))
            input_ids_with_image.extend(pad_ids[:num_image_tokens])
        input_ids_with_image.extend(input_ids[image_indices[-1] + 1:])
        return input_ids_with_image

    def __init__(self, config: 'Qwen2VLConfig', cache_config: 'Optional[CacheConfig]'=None, quant_config: 'Optional[QuantizationConfig]'=None) ->None:
        super().__init__()
        self.config = config
        self.visual = Qwen2VisionTransformer(config.vision_config, norm_eps=getattr(config, 'rms_norm_eps', 1e-06), quant_config=None)
        self.model = Qwen2Model(config, quant_config)
        if config.tie_word_embeddings:
            self.lm_head = self.model.embed_tokens
        else:
            self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size, quant_config=quant_config)
        self.logits_processor = LogitsProcessor(config)

    def _process_image_input(self, image_input: 'Qwen2VLImageInputs') ->torch.Tensor:
        pixel_values = image_input['pixel_values'].type(self.visual.dtype)
        image_embeds = self.visual(pixel_values, grid_thw=image_input['image_grid_thw'])
        return image_embeds

    def _process_video_input(self, video_input: 'Qwen2VLVideoInputs') ->torch.Tensor:
        pixel_values_videos = video_input['pixel_values_videos'].type(self.visual.dtype)
        video_embeds = self.visual(pixel_values_videos, grid_thw=video_input['video_grid_thw'])
        return video_embeds

    def forward(self, input_ids: 'torch.Tensor', positions: 'torch.Tensor', forward_batch: 'ForwardBatch'):
        """Run forward pass for Qwen2-VL.

        Args:
            input_ids: Flattened (concatenated) input_ids corresponding to a
                batch.
            positions: Flattened (concatenated) position ids corresponding to a
                batch.
                **NOTE**: If mrope is enabled (default setting for Qwen2-VL
                opensource models), the shape will be `(3, seq_len)`,
                otherwise it will be `(seq_len,).
                (Use input_metadata.mrope_positions to replace it)
            pixel_values: Pixel values to be fed to a model.
                `None` if no images are passed.
            image_grid_thw: Tensor `(n_images, 3)` of image 3D grid in LLM.
                `None` if no images are passed.
        """
        image_inputs = None
        if forward_batch.image_inputs is not None:
            image_inputs = [img for img in forward_batch.image_inputs if img is not None]
        positions = forward_batch.mrope_positions
        if forward_batch.forward_mode.is_decode() or image_inputs is None or len(image_inputs) == 0:
            inputs_embeds = self.model.embed_tokens(input_ids)
        else:
            if getattr(self.config, 'rope_scaling', {}).get('type', None) == 'mrope':
                assert positions.ndim == 2 and positions.size(0) == 3, f'multimodal section rotary embedding requires (3, seq_len) positions, but got {positions.size()}'
            inputs_embeds = self.model.embed_tokens(input_ids)
            extend_start_loc_cpu = forward_batch.extend_start_loc.cpu().numpy()
            prefix_lens_cpu = forward_batch.extend_prefix_lens.cpu().numpy()
            for i, image in enumerate(forward_batch.image_inputs):
                if image is None:
                    continue
                start_idx = extend_start_loc_cpu[i]
                prefix_len = prefix_lens_cpu[i]
                pixel_values = torch.tensor(image.pixel_values, device='cuda')
                image_grid_thws = torch.tensor(np.array(image.image_grid_thws), device='cuda')
                image_offsets = image.image_offsets
                image_input = Qwen2VLImageInputs(pixel_values=pixel_values, image_grid_thw=image_grid_thws)
                image_embeds = self._process_image_input(image_input)
                image_embeds_offset = 0
                for idx, image_offset in enumerate(image_offsets):
                    if image_offset < prefix_len:
                        continue
                    num_image_tokens = self.calculate_num_image_tokens(image_grid_thws[idx])
                    left_idx = start_idx + (image_offset - prefix_len)
                    right_idx = start_idx + (image_offset - prefix_len) + num_image_tokens
                    inputs_embeds[left_idx:right_idx] = image_embeds[image_embeds_offset:image_embeds_offset + num_image_tokens]
                    image_embeds_offset += num_image_tokens
            input_ids = None
        hidden_states = self.model(input_ids=input_ids, positions=positions, forward_batch=forward_batch, input_embeds=inputs_embeds)
        return self.logits_processor(input_ids, hidden_states, self.lm_head.weight, forward_batch)

    def load_weights(self, weights: 'Iterable[Tuple[str, torch.Tensor]]'):
        stacked_params_mapping = [('qkv_proj', 'q_proj', 'q'), ('qkv_proj', 'k_proj', 'k'), ('qkv_proj', 'v_proj', 'v'), ('gate_up_proj', 'up_proj', 1), ('gate_up_proj', 'gate_proj', 0)]
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        for name, loaded_weight in weights:
            if 'rotary_emb.inv_freq' in name:
                continue
            if self.config.tie_word_embeddings and 'lm_head.weight' in name:
                continue
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                if name.endswith('.bias') and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                if 'visual' in name and 'qkv.weight' in name:
                    visual_num_heads = self.config.vision_config.num_heads
                    visual_embed_dim = self.config.vision_config.embed_dim
                    head_size = visual_embed_dim // visual_num_heads
                    loaded_weight = loaded_weight.view(3, visual_num_heads, head_size, visual_embed_dim)
                    loaded_weight = loaded_weight.transpose(0, 1)
                    loaded_weight = loaded_weight.reshape(-1, visual_embed_dim)
                elif 'visual' in name and 'qkv.bias' in name:
                    visual_num_heads = self.config.vision_config.num_heads
                    visual_embed_dim = self.config.vision_config.embed_dim
                    head_size = visual_embed_dim // visual_num_heads
                    loaded_weight = loaded_weight.view(3, visual_num_heads, head_size)
                    loaded_weight = loaded_weight.transpose(0, 1)
                    loaded_weight = loaded_weight.reshape(-1)
                try:
                    if name.endswith('.bias') and name not in params_dict:
                        continue
                    param = params_dict[name]
                except KeyError:
                    None
                    raise
                weight_loader = getattr(param, 'weight_loader', default_weight_loader)
                weight_loader(param, loaded_weight)


class StablelmMLP(nn.Module):

    def __init__(self, config: 'PretrainedConfig', quant_config: 'Optional[QuantizationConfig]'=None) ->None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_up_proj = MergedColumnParallelLinear(config.hidden_size, [config.intermediate_size] * 2, bias=False, quant_config=quant_config)
        self.down_proj = RowParallelLinear(config.intermediate_size, config.hidden_size, bias=False, quant_config=quant_config)
        self.act_fn = SiluAndMul()

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class StablelmAttention(nn.Module):

    def __init__(self, config: 'PretrainedConfig', layer_id: 'int'=0, quant_config: 'Optional[QuantizationConfig]'=None) ->None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = config.num_attention_heads
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_key_value_heads = config.num_key_value_heads
        if self.total_num_key_value_heads >= tp_size:
            assert self.total_num_key_value_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_key_value_heads == 0
        self.num_key_value_heads = max(1, self.total_num_key_value_heads // tp_size)
        self.head_dim = self.hidden_size // self.total_num_heads
        self.max_position_embeddings = config.max_position_embeddings
        rope_pct = getattr(config, 'rope_pct', getattr(config, 'partial_rotary_factor', 1))
        self.rotary_ndims = int(self.head_dim * rope_pct)
        self.scaling = self.head_dim ** -0.5
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_key_value_heads * self.head_dim
        self.qkv_bias = getattr(config, 'use_qkv_bias', False)
        if self.head_dim * self.num_heads * tp_size != self.hidden_size:
            raise ValueError(f'hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size} and `num_heads`: {self.num_heads}).')
        self.qkv_proj = QKVParallelLinear(self.hidden_size, self.head_dim, self.total_num_heads, self.total_num_key_value_heads, self.qkv_bias)
        self.o_proj = RowParallelLinear(self.total_num_heads * self.head_dim, self.hidden_size, bias=False)
        self.rotary_emb = get_rope(self.head_dim, rotary_dim=self.rotary_ndims, max_position=self.config.max_position_embeddings, base=self.config.rope_theta)
        self.attn = RadixAttention(self.num_heads, self.head_dim, self.scaling, num_kv_heads=self.num_key_value_heads, layer_id=layer_id)

    def forward(self, positions: 'torch.Tensor', hidden_states: 'torch.Tensor', forward_batch: 'ForwardBatch') ->torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v, forward_batch)
        output, _ = self.o_proj(attn_output)
        return output


class StablelmDecoderLayer(nn.Module):

    def __init__(self, config: 'PretrainedConfig', layer_id: 'int'=0, quant_config: 'Optional[QuantizationConfig]'=None) ->None:
        super().__init__()
        self.self_attn = StablelmAttention(config, layer_id=layer_id)
        self.mlp = StablelmMLP(config, quant_config=quant_config)
        norm_eps = getattr(config, 'norm_eps', getattr(config, 'layer_norm_eps', 1e-05))
        self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=norm_eps)
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size, eps=norm_eps)

    def forward(self, positions: 'torch.Tensor', hidden_states: 'torch.Tensor', forward_batch: 'ForwardBatch') ->Tuple[torch.Tensor, torch.Tensor]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(positions=positions, hidden_states=hidden_states, forward_batch=forward_batch)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states, residual


class StableLMEpochModel(nn.Module):

    def __init__(self, config: 'PretrainedConfig', quant_config: 'Optional[QuantizationConfig]'=None) ->None:
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([StablelmDecoderLayer(config, i, quant_config=quant_config) for i in range(config.num_hidden_layers)])
        norm_eps = getattr(config, 'norm_eps', getattr(config, 'layer_norm_eps', 1e-05))
        self.norm = nn.LayerNorm(config.hidden_size, eps=norm_eps)

    def forward(self, input_ids: 'torch.Tensor', positions: 'torch.Tensor', forward_batch: 'ForwardBatch', input_embeds: 'torch.Tensor'=None) ->torch.Tensor:
        if input_embeds is None:
            hidden_states = self.embed_tokens(input_ids)
        else:
            hidden_states = input_embeds
        for i in range(len(self.layers)):
            layer = self.layers[i]
            hidden_states, residual = layer(positions, hidden_states, forward_batch)
        hidden_states = self.norm(hidden_states)
        return hidden_states


class StableLmForCausalLM(nn.Module):

    def __init__(self, config: 'PretrainedConfig', quant_config: 'Optional[QuantizationConfig]'=None, cache_config=None) ->None:
        super().__init__()
        self.config = config
        self.quant_config = quant_config
        self.model = StableLMEpochModel(config, quant_config=quant_config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        self.logits_processor = LogitsProcessor(config)

    @torch.no_grad()
    def forward(self, input_ids: 'torch.Tensor', positions: 'torch.Tensor', forward_batch: 'ForwardBatch', input_embeds: 'torch.Tensor'=None) ->torch.Tensor:
        hidden_states = self.model(input_ids, positions, forward_batch, input_embeds)
        return self.logits_processor(input_ids, hidden_states, self.lm_head.weight, forward_batch)

    def load_weights(self, weights: 'Iterable[Tuple[str, torch.Tensor]]'):
        stacked_params_mapping = [('qkv_proj', 'q_proj', 'q'), ('qkv_proj', 'k_proj', 'k'), ('qkv_proj', 'v_proj', 'v'), ('gate_up_proj', 'gate_proj', 0), ('gate_up_proj', 'up_proj', 1)]
        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            if 'rotary_emb.inv_freq' in name:
                continue
            if 'rotary_emb.cos_cached' in name or 'rotary_emb.sin_cached' in name:
                continue
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                if name.endswith('.bias') and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                if name.endswith('.bias') and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, 'weight_loader', default_weight_loader)
                weight_loader(param, loaded_weight)


class XverseMLP(nn.Module):

    def __init__(self, hidden_size: 'int', intermediate_size: 'int', hidden_act: 'str', quant_config: 'Optional[QuantizationConfig]'=None, reduce_results: 'bool'=True) ->None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(hidden_size, [intermediate_size] * 2, bias=False, quant_config=quant_config)
        self.down_proj = RowParallelLinear(intermediate_size, hidden_size, bias=False, quant_config=quant_config, reduce_results=reduce_results)
        if hidden_act != 'silu':
            raise ValueError(f'Unsupported activation: {hidden_act}. Only silu is supported for now.')
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class XverseAttention(nn.Module):

    def __init__(self, hidden_size: 'int', num_heads: 'int', num_kv_heads: 'int', layer_id: 'int'=0, rope_theta: 'float'=10000, rope_scaling: 'Optional[Dict[str, Any]]'=None, max_position_embeddings: 'int'=8192, cache_config=None, quant_config: 'Optional[QuantizationConfig]'=None) ->None:
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim ** -0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
        self.qkv_proj = QKVParallelLinear(hidden_size, self.head_dim, self.total_num_heads, self.total_num_kv_heads, bias=False, quant_config=quant_config)
        self.o_proj = RowParallelLinear(self.total_num_heads * self.head_dim, hidden_size, bias=False, quant_config=quant_config)
        self.rotary_emb = get_rope(self.head_dim, rotary_dim=self.head_dim, max_position=max_position_embeddings, base=rope_theta, rope_scaling=rope_scaling)
        self.attn = RadixAttention(self.num_heads, self.head_dim, self.scaling, num_kv_heads=self.num_kv_heads, layer_id=layer_id)

    def forward(self, positions: 'torch.Tensor', hidden_states: 'torch.Tensor', forward_batch: 'ForwardBatch') ->torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v, forward_batch)
        output, _ = self.o_proj(attn_output)
        return output


class XverseMoE(nn.Module):

    def __init__(self, config: 'PretrainedConfig', quant_config: 'Optional[QuantizationConfig]'=None):
        super().__init__()
        self.config = config
        self.rank = get_tensor_model_parallel_rank()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.n_routed_experts = config.num_experts
        self.top_k = config.moe_top_k
        if self.tp_size > self.n_routed_experts:
            raise ValueError(f'Tensor parallel size {self.tp_size} is greater than the number of experts {self.n_routed_experts}.')
        self.experts = nn.ModuleList([XverseMLP(hidden_size=config.hidden_size, intermediate_size=config.intermediate_size, hidden_act=config.hidden_act, quant_config=quant_config, reduce_results=False) for _ in range(self.n_routed_experts)])
        self.pack_params()
        self.router = ReplicatedLinear(config.hidden_size, self.n_routed_experts, bias=False, quant_config=None)
        if config.num_shared_experts is not None:
            intermediate_size = config.intermediate_size * config.num_shared_experts
            self.shared_experts = XverseMLP(hidden_size=config.hidden_size, intermediate_size=intermediate_size, hidden_act=config.hidden_act, quant_config=quant_config, reduce_results=False)

    def pack_params(self):
        w1 = []
        w2 = []
        for expert in self.experts:
            w1.append(expert.gate_up_proj.weight)
            w2.append(expert.down_proj.weight)
        self.w1 = torch._utils._flatten_dense_tensors(w1)
        w1s = torch._utils._unflatten_dense_tensors(self.w1, w1)
        for data, param in zip(w1s, w1):
            param.data = data
        self.w1 = self.w1.view(len(w1), *w1s[0].shape)
        self.w2 = torch._utils._flatten_dense_tensors(w2)
        w2s = torch._utils._unflatten_dense_tensors(self.w2, w2)
        for data, param in zip(w2s, w2):
            param.data = data
        self.w2 = self.w2.view(len(w2), *w2s[0].shape)

    def forward(self, hidden_states: 'torch.Tensor') ->torch.Tensor:
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        if self.config.num_shared_experts is not None:
            shared_output = self.shared_experts(hidden_states)
        router_logits, _ = self.router(hidden_states)
        final_hidden_states = fused_moe(hidden_states, self.w1, self.w2, router_logits, self.top_k, renormalize=getattr(self.config, 'norm_topk_prob', False), inplace=True)
        if self.config.num_shared_experts is not None:
            final_hidden_states = final_hidden_states + shared_output
        final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states)
        return final_hidden_states.view(num_tokens, hidden_dim)


class XverseDecoderLayer(nn.Module):

    def __init__(self, config: 'PretrainedConfig', layer_id: 'int', cache_config=None, quant_config: 'Optional[QuantizationConfig]'=None) ->None:
        super().__init__()
        self.hidden_size = config.hidden_size
        rope_theta = getattr(config, 'rope_theta', 10000)
        rope_scaling = getattr(config, 'rope_scaling', None)
        max_position_embeddings = getattr(config, 'max_position_embeddings', 8192)
        num_key_value_heads = getattr(config, 'num_key_value_heads', config.num_attention_heads)
        self.self_attn = XverseAttention(hidden_size=self.hidden_size, num_heads=config.num_attention_heads, num_kv_heads=num_key_value_heads, layer_id=layer_id, rope_theta=rope_theta, rope_scaling=rope_scaling, max_position_embeddings=max_position_embeddings, cache_config=cache_config, quant_config=quant_config)
        if config.num_experts is not None:
            self.mlp = XverseMoE(config=config, quant_config=quant_config)
        else:
            self.mlp = XverseMLP(hidden_size=config.hidden_size, intermediate_size=config.intermediate_size, hidden_act=config.hidden_act, quant_config=quant_config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, positions: 'torch.Tensor', hidden_states: 'torch.Tensor', forward_batch: 'ForwardBatch', residual: 'Optional[torch.Tensor]') ->torch.Tensor:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions=positions, hidden_states=hidden_states, forward_batch=forward_batch)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class XverseModel(nn.Module):
    fall_back_to_pt_during_load = False

    def __init__(self, config: 'PretrainedConfig', cache_config=None, quant_config: 'Optional[QuantizationConfig]'=None) ->None:
        super().__init__()
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([XverseDecoderLayer(config, layer_id, cache_config, quant_config=quant_config) for layer_id in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids: 'torch.Tensor', positions: 'torch.Tensor', forward_batch: 'ForwardBatch') ->torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for i in range(len(self.layers)):
            layer = self.layers[i]
            hidden_states, residual = layer(positions, hidden_states, forward_batch, residual)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class XverseForCausalLM(nn.Module):

    def __init__(self, config: 'LlamaConfig', quant_config: 'Optional[QuantizationConfig]'=None, cache_config=None, efficient_weight_load=False) ->None:
        super().__init__()
        self.config = config
        self.quant_config = quant_config
        self.model = XverseModel(config, quant_config=quant_config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        self.logits_processor = LogitsProcessor(config)

    @torch.no_grad()
    def forward(self, input_ids: 'torch.Tensor', positions: 'torch.Tensor', forward_batch: 'ForwardBatch', input_embeds: 'torch.Tensor'=None) ->torch.Tensor:
        hidden_states = self.model(input_ids, positions, forward_batch, input_embeds)
        return self.logits_processor(input_ids, hidden_states, self.lm_head.weight, forward_batch)

    def load_weights(self, weights: 'Iterable[Tuple[str, torch.Tensor]]', name=None, loaded_weight=None):
        stacked_params_mapping = [('qkv_proj', 'q_proj', 'q'), ('qkv_proj', 'k_proj', 'k'), ('qkv_proj', 'v_proj', 'v'), ('gate_up_proj', 'gate_proj', 0), ('gate_up_proj', 'up_proj', 1)]
        params_dict = dict(self.named_parameters())

        def load_weights_per_param(name, loaded_weight):
            if 'rotary_emb.inv_freq' in name or 'projector' in name:
                return
            if 'rotary_emb.cos_cached' in name or 'rotary_emb.sin_cached' in name:
                return
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                if name.endswith('.bias') and name not in params_dict:
                    continue
                if name.startswith('model.vision_tower') and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                if name.endswith('.bias') and name not in params_dict:
                    return
                if name.startswith('model.vision_tower') and name not in params_dict:
                    return
                param = params_dict[name]
                weight_loader = getattr(param, 'weight_loader', default_weight_loader)
                weight_loader(param, loaded_weight)
        if name is None or loaded_weight is None:
            for name, loaded_weight in weights:
                load_weights_per_param(name, loaded_weight)
        else:
            load_weights_per_param(name, loaded_weight)


class XverseMoeForCausalLM(nn.Module):

    def __init__(self, config: 'PretrainedConfig', cache_config=None, quant_config: 'Optional[QuantizationConfig]'=None) ->None:
        super().__init__()
        self.config = config
        self.quant_config = quant_config
        self.model = XverseModel(config, cache_config, quant_config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size, quant_config=quant_config)
        self.logits_processor = LogitsProcessor(config)

    @torch.no_grad()
    def forward(self, input_ids: 'torch.Tensor', positions: 'torch.Tensor', forward_batch: 'ForwardBatch') ->torch.Tensor:
        hidden_states = self.model(input_ids, positions, forward_batch)
        return self.logits_processor(input_ids, hidden_states, self.lm_head.weight, forward_batch)

    def load_weights(self, weights: 'Iterable[Tuple[str, torch.Tensor]]'):
        stacked_params_mapping = [('qkv_proj', 'q_proj', 'q'), ('qkv_proj', 'k_proj', 'k'), ('qkv_proj', 'v_proj', 'v'), ('gate_up_proj', 'gate_proj', 0), ('gate_up_proj', 'up_proj', 1)]
        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            if 'rotary_emb.inv_freq' in name:
                continue
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                if name.endswith('.bias') and name not in params_dict:
                    continue
                if ('mlp.experts.' in name or 'mlp.shared_experts.' in name) and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                if name.endswith('.bias') and name not in params_dict:
                    continue
                if ('mlp.experts.' in name or 'mlp.shared_experts.' in name) and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, 'weight_loader', default_weight_loader)
                weight_loader(param, loaded_weight)


class YiVLMultiModalProjector(nn.Module):

    def __init__(self, config: 'LlavaConfig'):
        super().__init__()
        self.linear_1 = nn.Linear(config.vision_config.hidden_size, config.text_config.hidden_size)
        self.ln_1 = nn.LayerNorm(config.text_config.hidden_size)
        self.act = nn.GELU()
        self.linear_2 = nn.Linear(config.text_config.hidden_size, config.text_config.hidden_size)
        self.ln_2 = nn.LayerNorm(config.text_config.hidden_size)

    def forward(self, image_features):
        hidden_states = self.linear_1(image_features)
        hidden_states = self.ln_1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        hidden_states = self.ln_2(hidden_states)
        return hidden_states


class YiVLForCausalLM(LlavaLlamaForCausalLM):

    def __init__(self, config: 'LlavaConfig', quant_config: 'Optional[QuantizationConfig]'=None, cache_config=None) ->None:
        super().__init__(config, quant_config, cache_config)
        self.multi_modal_projector = YiVLMultiModalProjector(self.config)
        self.vision_tower_subfolder = self.config.mm_vision_tower.replace('./', '')

    def load_weights(self, weights: 'Iterable[Tuple[str, torch.Tensor]]'):
        self.vision_tower = CLIPVisionModel.from_pretrained(self.config._name_or_path, torch_dtype=torch.float16, subfolder=self.vision_tower_subfolder)
        self.vision_tower.eval()
        self.vision_feature_layer = self.config.mm_vision_select_layer
        self.vision_feature_select_strategy = self.config.mm_vision_select_feature
        self.image_size = self.vision_tower.config.image_size
        self.patch_size = self.vision_tower.config.patch_size
        self.mm_patch_merge_type = getattr(self.config, 'mm_patch_merge_type', 'flat')
        self.image_aspect_ratio = getattr(self.config, 'image_aspect_ratio', 'square')
        self.image_grid_pinpoints = getattr(self.config, 'image_grid_pinpoints', None)
        self.image_feature_len = int((self.image_size / self.patch_size) ** 2)
        if self.vision_feature_select_strategy == 'patch':
            pass
        elif self.vision_feature_select_strategy == 'cls_patch':
            self.image_feature_len += 1
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        projector_weights = {'model.mm_projector.0': 'multi_modal_projector.linear_1', 'model.mm_projector.1': 'multi_modal_projector.ln_1', 'model.mm_projector.3': 'multi_modal_projector.linear_2', 'model.mm_projector.4': 'multi_modal_projector.ln_2', 'model.vision_tower.vision_tower': 'vision_tower'}
        params_dict = dict(self.named_parameters())
        weights = list(weights)
        for name, loaded_weight in weights:
            if 'projector' in name or 'vision_tower' in name:
                for weight_name, param_name in projector_weights.items():
                    if weight_name in name:
                        name = name.replace(weight_name, param_name)
                param = params_dict[name]
                weight_loader = getattr(param, 'weight_loader', default_weight_loader)
                weight_loader(param, loaded_weight)
        self.language_model.load_weights(weights)


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (MllamaTextRMSNorm,
     lambda: ([], {'hidden_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (YiVLMultiModalProjector,
     lambda: ([], {'config': SimpleNamespace(vision_config=SimpleNamespace(hidden_size=4), text_config=SimpleNamespace(hidden_size=4))}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
]

