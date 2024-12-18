
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


from enum import auto


from enum import Enum


from types import SimpleNamespace


from typing import Any


from typing import Dict


from typing import List


from typing import Optional


from typing import Tuple


import torch


import torch.distributed as dist


from torch.distributed.pipelining import PipelineStage


from torch.distributed.pipelining import ScheduleGPipe


from typing import Union


import torch._dynamo.config


import torch._inductor.config


import torch.nn as nn


from torch.distributed.device_mesh import DeviceMesh


from torch.distributed.elastic.multiprocessing.errors import record


from torch.distributed.elastic.utils.distributed import get_free_port


import logging


import re


from typing import Mapping


import torch.distributed.checkpoint as dist_cp


from torch.distributed._tensor import DTensor


from torch.distributed._tensor import Replicate


from torch.distributed._tensor import Shard


from torch.nn import Module


from typing import Set


from collections import defaultdict


from types import MethodType


from torch.distributed import DeviceMesh


from torch.distributed._tensor import Placement


from torch.distributed.tensor._utils import compute_local_shape_and_global_offset


from typing import Sequence


from abc import abstractmethod


from collections import deque


from functools import partial


from uuid import uuid4


import torch.multiprocessing as mp


from torch.distributed.device_mesh import init_device_mesh


from torch.distributed.tensor.parallel import ColwiseParallel


from torch.distributed.tensor.parallel import RowwiseParallel


from torch.distributed.tensor.parallel import parallelize_module


import itertools


import time


from collections import OrderedDict


from torch._subclasses import FakeTensor


import numpy as np


from torch.export import Dim


import torch._inductor


from abc import ABC


from typing import Callable


import uuid


import copy


import torch.nn.functional as F


from torch.ao.quantization.fx._decomposed import quantized_decomposed_lib


import inspect


def find_multiple(n: 'int', k: 'int') ->int:
    if n % k == 0:
        return n
    return n + k - n % k


def get_precision():
    """get_precision() is a torchchat-internal API that returns the dtype we're building the model for, as specified by the `--dtype` CLI option+,
    or the precision quantizer.
    """
    global precision
    if precision is None:
        precision = torch.float32
    return precision


def get_group_qparams(w, n_bit=4, groupsize=128, *, scales_dtype=torch.float):
    if groupsize > w.shape[-1]:
        groupsize = w.shape[-1]
    assert groupsize > 1
    assert w.shape[-1] % groupsize == 0
    assert w.dim() == 2
    to_quant = w.reshape(-1, groupsize)
    assert torch.isnan(to_quant).sum() == 0
    max_val = to_quant.amax(dim=1, keepdim=True)
    min_val = to_quant.amin(dim=1, keepdim=True)
    max_int = 2 ** n_bit - 1
    scales = (max_val - min_val).clamp(min=1e-06) / max_int
    zeros = min_val + scales * 2 ** (n_bit - 1)
    return scales.reshape(w.shape[0], -1), zeros.reshape(w.shape[0], -1)


def group_quantize_tensor_from_qparams(w, scales, zeros, n_bit=4, groupsize=128):
    assert groupsize > 1
    if groupsize > w.shape[-1] and scales.shape[-1] == 1:
        groupsize = w.shape[-1]
    assert w.shape[-1] % groupsize == 0
    assert w.dim() == 2
    to_quant = w.reshape(-1, groupsize)
    assert torch.isnan(to_quant).sum() == 0
    scales = scales.reshape(-1, 1)
    zeros = zeros.reshape(-1, 1)
    min_val = zeros - scales * 2 ** (n_bit - 1)
    max_int = 2 ** n_bit - 1
    min_int = 0
    w_int32 = to_quant.sub(min_val).div(scales).round().clamp_(min_int, max_int).reshape_as(w)
    return w_int32


def pack_scales_and_zeros(scales, zeros, *, scales_dtype=torch.float):
    assert scales.shape == zeros.shape
    assert scales.dtype == scales_dtype
    assert zeros.dtype == scales_dtype
    return torch.cat([scales.reshape(scales.size(0), scales.size(1), 1), zeros.reshape(zeros.size(0), zeros.size(1), 1)], 2).transpose(0, 1).contiguous()


def group_quantize_tensor(w, n_bit=4, groupsize=128):
    scales, zeros = get_group_qparams(w, n_bit, groupsize)
    w_int32 = group_quantize_tensor_from_qparams(w, scales, zeros, n_bit, groupsize)
    scales_and_zeros = pack_scales_and_zeros(scales, zeros)
    return w_int32, scales_and_zeros


def linear_int4(input, weight_int4pack, scales_and_zeros, out_features, groupsize):
    origin_input_size = input.size()
    input = input.reshape(-1, origin_input_size[-1])
    if 'cuda' in str(input.device):
        c = torch.ops.aten._weight_int4pack_mm(input.to(torch.bfloat16), weight_int4pack, groupsize, scales_and_zeros.to(torch.bfloat16))
    else:
        c = torch.ops.aten._weight_int4pack_mm(input, weight_int4pack, groupsize, scales_and_zeros)
    new_shape = origin_input_size[:-1] + (out_features,)
    c = c.reshape(new_shape)
    return c


class WeightOnlyInt4Linear(torch.nn.Module):
    __constants__ = ['in_features', 'out_features']
    in_features: 'int'
    out_features: 'int'
    weight: 'torch.Tensor'
    scales_and_zeros: 'torch.Tensor'

    def __init__(self, in_features: 'int', out_features: 'int', bias=True, device=None, dtype=None, *, groupsize: int=128, inner_k_tiles: int=8, weight: Optional[torch.Tensor]=None, scales_and_zeros: Optional[torch.Tensor]=None) ->None:
        super().__init__()
        self.padding = not self._check_k(k=in_features, groupsize=groupsize, inner_k_tiles=inner_k_tiles)
        if self.padding:
            self.origin_in_features = in_features
            in_features = find_multiple(in_features, 1024)
        self.in_features = in_features
        self.out_features = out_features
        assert not bias, 'require bias=False'
        self.groupsize = groupsize
        self.inner_k_tiles = inner_k_tiles
        assert out_features % 8 == 0, 'require out_features % 8 == 0'
        assert in_features % (inner_k_tiles * 16) == 0, 'require in_features % (innerKTiles * 16) == 0'
        assert (weight is None) == bool(scales_and_zeros is None), 'must specify both weights and scales_and_zeros, or neither'
        if weight is None:
            weight = torch.empty((out_features // 8, in_features // (inner_k_tiles * 16), 32, inner_k_tiles // 2), dtype=torch.int32, device=device)
            scales_and_zeros = torch.empty((in_features // groupsize, out_features, 2), dtype=get_precision(), device=device)
        self.register_buffer('weight', weight)
        self.register_buffer('scales_and_zeros', scales_and_zeros)

    def forward(self, input: 'torch.Tensor') ->torch.Tensor:
        if self.padding:
            input = F.pad(input, pad=(0, self.in_features - self.origin_in_features))
        return linear_int4(input, self.weight, self.scales_and_zeros, self.out_features, self.groupsize)

    @classmethod
    def _check_k(cls, *, k, groupsize=1, inner_k_tiles=1):
        return k % groupsize == 0 and k % (inner_k_tiles * 16) == 0

    @classmethod
    def _prepare_weight_and_scales_and_zeros(cls, weight_bf16, groupsize, inner_k_tiles):
        weight_int32, scales_and_zeros = group_quantize_tensor(weight_bf16, n_bit=4, groupsize=groupsize)
        weight_uint8 = weight_int32[:, ::2] << 4 | weight_int32[:, 1::2]
        weight_int4pack = torch.ops.aten._convert_weight_to_int4pack(weight_uint8, inner_k_tiles)
        return weight_int4pack, scales_and_zeros

    @classmethod
    def _calc_padded_size(cls, *, k, groupsize=1, innner_k_tiles=1):
        return find_multiple(k, 1024)


def linear_int8_aoti(input, weight, scales):
    n_groups = scales.numel() // scales.shape[0]
    if n_groups == 1:
        scales = scales.view(-1)
        if torch.compiler.is_compiling() or input.device.type not in ['cpu', 'mps'] or not hasattr(torch.ops.aten, '_weight_int8pack_mm'):
            lin = F.linear(input, weight)
            return lin * scales
        return torch.ops.aten._weight_int8pack_mm(input.reshape(-1, input.shape[-1]), weight, scales).reshape(input.shape[:-1] + (weight.shape[0],))
    return F.linear(input, (weight.view(weight.shape[0], n_groups, -1) * scales.view(weight.shape[0], n_groups, -1)).view(weight.shape[0], -1))


def _qdq_dynamic_quantized_linear(x_fp32, x_quant_min, x_quant_max, x_eps, weight_i8, weight_scale, weight_zero_point, weight_quant_min, weight_quant_max, bias_fp32):
    x_scale, x_zero_point = torch.ops.quantized_decomposed.choose_qparams(x_fp32, x_quant_min, x_quant_max, x_eps, torch.int8)
    x_i8 = torch.ops.quantized_decomposed.quantize_per_tensor(x_fp32, x_scale, x_zero_point, x_quant_min, x_quant_max, torch.int8)
    x_fp32 = torch.ops.quantized_decomposed.dequantize_per_tensor(x_i8, x_scale, x_zero_point, x_quant_min, x_quant_max, torch.int8)
    weight_fp32 = torch.ops.quantized_decomposed.dequantize_per_tensor(weight_i8, weight_scale, weight_zero_point, weight_quant_min, weight_quant_max, torch.int8)
    out_fp32 = torch.ops.aten.linear.default(x_fp32, weight_fp32, bias_fp32)
    return out_fp32


def linear_int8_et(input, weight, scales):
    n_groups = scales.numel() // scales.shape[0]
    if n_groups == 1:
        scales = scales.view(-1)
        if True:
            lin = F.linear(input, weight)
            return lin * scales
        return _qdq_dynamic_quantized_linear(x_fp32=input.float(), x_quant_min=-128, x_quant_max=127, x_eps=torch.finfo(input.dtype).eps, weight_i8=weight, weight_scale=scales.float(), weight_zero_point=0, weight_quant_min=-128, weight_quant_max=127, bias_fp32=None)
    return F.linear(input, (weight.view(weight.shape[0], n_groups, -1) * scales.view(weight.shape[0], n_groups, -1)).view(weight.shape[0], -1))


class _Backend(Enum):
    AOTI = 0
    EXECUTORCH = 1


def _active_backend() ->Optional[_Backend]:
    global active_builder_args_dso
    global active_builder_args_aoti_package
    global active_builder_args_pte
    args = active_builder_args_dso, active_builder_args_pte, active_builder_args_aoti_package
    if not any(args):
        return None
    if sum(map(bool, args)) > 1:
        raise RuntimeError('Code generation needs to choose different implementations.  Please only use one export option, and call export twice if necessary!')
    return _Backend.EXECUTORCH if active_builder_args_pte else _Backend.AOTI


def use_et_backend() ->bool:
    return _active_backend() == _Backend.EXECUTORCH


class WeightOnlyInt8Linear(nn.Module):
    __constants__ = ['in_features', 'out_features']
    in_features: 'int'
    out_features: 'int'
    weight: 'torch.Tensor'
    scales: 'torch.Tensor'

    def __init__(self, in_features, out_features, bias=None, device=None, dtype=None, *, weight: Optional[torch.Tensor]=None, scales: Optional[torch.Tensor]=None, groupsize: Optional[int]=None):
        super().__init__()
        if dtype is None:
            dtype = torch.get_default_dtype()
        if device is None:
            device = 'cpu'
        assert not bias, 'Bias is not supported by LinearInt8'
        self.in_features = in_features
        self.out_features = out_features
        assert (weight is None) == bool(scales is None), 'must specify both weights and scales, or neither'
        if weight is None:
            weight = torch.empty((out_features, in_features), dtype=torch.int8, device=device)
            if groupsize is None or groupsize == 0:
                scales = torch.empty(out_features, dtype=dtype, device=device)
            else:
                n_groups = (in_features + groupsize - 1) // groupsize
                scales = torch.empty(out_features, n_groups, dtype=dtype, device=device)
        self.register_buffer('weight', weight)
        self.register_buffer('scales', scales)
        if use_et_backend():
            self.forward = self.et_forward
        else:
            self.forward = self.aoti_forward

    def aoti_forward(self, input: 'torch.Tensor') ->torch.Tensor:
        return linear_int8_aoti(input, self.weight, self.scales)

    def et_forward(self, input: 'torch.Tensor') ->torch.Tensor:
        return linear_int8_et(input, self.weight, self.scales)


class QuantizedEmbedding(torch.nn.Module):

    def __init__(self, num_embeddings: 'int', embedding_dim: 'int', device=None, dtype=None, *, bitwidth: int, groupsize: Optional[int]=None, weight: Optional[torch.Tensor]=None, scales: Optional[torch.Tensor]=None) ->None:
        super().__init__()
        if dtype is None:
            dtype = get_precision()
        if groupsize is None or groupsize == 0:
            groupsize = embedding_dim
        self.groupsize = groupsize
        self.dtype = dtype
        self.bitwidth = bitwidth
        assert (weight is None) == bool(scales is None), 'must specify both weights and scales, or neither'
        if bitwidth not in [4, 8]:
            raise RuntimeError(f'QUantized embedding does not support bitwidth={bitwidth}')
        if weight is None:
            groups_per_row = (embedding_dim + groupsize - 1) // groupsize
            weight = torch.empty((num_embeddings, embedding_dim * bitwidth // 8), dtype=torch.int8, device=device)
            scales = torch.empty((num_embeddings, groups_per_row), dtype=dtype, device=device).squeeze(dim=-1)
        self.register_buffer('weight', weight)
        self.register_buffer('scales', scales)
        if use_et_backend():
            self.forward = self.et_forward
        else:
            self.forward = self.aoti_forward

    @torch.no_grad()
    def et_forward(self, indices: 'torch.Tensor') ->torch.Tensor:
        if self.bitwidth == 8:
            return torch.ops.quantized_decomposed.embedding_byte.dtype(self.weight, self.scales, None, 0, 0, indices, dtype=self.dtype)
        else:
            return torch.ops.quantized_decomposed.embedding_4bit.dtype(self.weight, self.scales, None, 0, 0, indices, dtype=self.dtype)

    @torch.no_grad()
    def aoti_forward(self, indices: 'torch.Tensor') ->torch.Tensor:
        if self.bitwidth == 4:
            weight_even = self.weight.div(16, rounding_mode='trunc')
            weight_odd = self.weight.remainder(16)
            weight_unpacked = torch.stack((weight_even, weight_odd), dim=-1)
            weight = weight_unpacked.view(self.weight.shape[0], -1)
            weight = weight.add(-8)
        else:
            weight = self.weight
        scales = self.scales.view(weight.shape[0], -1)
        result_weights = F.embedding(indices, weight)
        result_scales = F.embedding(indices, scales)
        rw_view = result_weights.view(tuple(result_weights.shape[:-1] + (scales.shape[1], -1)))
        rs_view = result_scales.view(tuple(result_scales.shape[:-1]) + (scales.shape[1], 1))
        r = rw_view * rs_view
        return r.view(indices.size() + (-1,))

