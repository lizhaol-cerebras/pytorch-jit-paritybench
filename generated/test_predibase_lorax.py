
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


from torch.utils.cpp_extension import BuildExtension


from torch.utils.cpp_extension import CUDAExtension


import torch


from abc import ABC


from abc import abstractmethod


from typing import TYPE_CHECKING


from typing import Dict


from typing import Optional


from typing import Set


from typing import Tuple


from collections import defaultdict


from typing import List


from typing import Type


from typing import Union


from torch.distributed import ProcessGroup


import torch.distributed


from abc import abstractclassmethod


from typing import TypeVar


from typing import Any


from typing import Callable


import torch.nn as nn


from functools import lru_cache


import math


import numpy as np


from torch.cuda.amp import custom_fwd


from torch import nn


from torch.nn import functional as F


import warnings


import torch.utils.checkpoint


from torch.nn import LayerNorm


import torch.nn.functional as F


from torch.nn import CrossEntropyLoss


import random


from torch.nn.init import _calculate_fan_in_and_fan_out


import copy


from typing import ContextManager


import torch.profiler


import re


from torch.utils._triton import has_triton as has_triton_torch


from typing import Iterable


import inspect


import time


from logging import getLogger


from typing import Literal


import itertools


import torch.utils.cpp_extension as torch_cpp_ext


from copy import copy


class FastLinear(torch.nn.Module):

    def __init__(self, weight, bias) ->None:
        super().__init__()
        self.weight = torch.nn.Parameter(weight, requires_grad=False)
        if bias is not None:
            self.bias = torch.nn.Parameter(bias, requires_grad=False)
        else:
            self.bias = None

    @classmethod
    def load(cls, config, prefix: 'str', weights, bias: 'bool'):
        weight = weights.get_tensor(f'{prefix}.weight')
        if bias:
            bias = weights.get_tensor(f'{prefix}.bias')
        else:
            bias = None
        return cls(weight, bias)

    def forward(self, input: 'torch.Tensor') ->torch.Tensor:
        return F.linear(input, self.weight, self.bias)


class ResBlock(torch.nn.Module):

    def __init__(self, config: 'MedusaConfig', prefix: 'str', weights: 'AbstractWeights'):
        super().__init__()
        self.linear = FastLinear.load(config, prefix=f'{prefix}.linear', weights=weights, bias=True)
        self.act = torch.nn.SiLU()
        self.scaling = 1

    def forward(self, x):
        return x + self.act(self.linear(x))


class MedusaHead(torch.nn.Module):

    def __init__(self, config: 'MedusaConfig', prefix: 'str', weights: 'AbstractWeights'):
        super().__init__()
        self.blocks = torch.nn.ModuleList([ResBlock(config, prefix=f'{prefix}.{i}', weights=weights) for i in range(config.medusa_num_layers)])
        n = len(self.blocks)
        self.out = FastLinear.load(config, prefix=f'{prefix}.{n}', weights=weights, bias=False)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        x = self.out(x)
        return x


class MedusaV1(torch.nn.Module):

    def __init__(self, config: 'MedusaConfig', weights: 'AbstractWeights'):
        super().__init__()
        self.heads = torch.nn.ModuleList([MedusaHead(config, prefix=f'{i}', weights=weights) for i in range(config.medusa_num_heads)])

    def forward(self, x, lm_head, segments: 'Optional[MedusaSegments]'=None):
        logits = lm_head(x)
        speculative_logits = torch.stack([head(x) for head in self.heads], dim=1)
        return logits, speculative_logits


class SuperLayer(torch.nn.Module):

    def __init__(self, linear):
        super().__init__()
        self.linear = linear

    def forward(self, x):
        return self.linear.forward(x)


class AWQLinear(nn.Module):

    def __init__(self, w_bit, group_size, qweight, qzeros, scales, bias):
        super().__init__()
        if w_bit != 4:
            raise NotImplementedError('Only 4-bit are supported for now.')
        self.in_features = qweight.shape[0]
        self.out_features = qweight.shape[1] * 32 // w_bit
        self.split_k_iters = 8
        self.w_bit = w_bit
        self.group_size = group_size if group_size != -1 else self.in_features
        assert self.in_features % self.group_size == 0, 'in_features must be divisible by group_size'
        assert self.out_features % (32 // self.w_bit) == 0, 'out_features must be divisible by 32 // w_bit'
        self.qweight = qweight
        self.qzeros = qzeros
        self.scales = scales
        self.bias = bias

    @torch.no_grad()
    def forward(self, x):
        out_shape = x.shape[:-1] + (self.out_features,)
        input_dtype = x.dtype
        if input_dtype != torch.float16:
            x = x.half()
        out = awq_inference_engine.gemm_forward_cuda(x.reshape(-1, x.shape[-1]), self.qweight, self.scales, self.qzeros, 8)
        if input_dtype != torch.float16:
            out = out
        out = out + self.bias if self.bias is not None else out
        return out.reshape(out_shape)

    @property
    def weight(self) ->torch.Tensor:
        return self.qweight


class EETQLinear(torch.nn.Module):

    def __init__(self, weight, bias) ->None:
        super().__init__()
        device = weight.device
        if weight.dtype != torch.float16:
            weight = weight
        weight = torch.t(weight).contiguous().cpu()
        weight, scale = quant_weights(weight, torch.int8, False)
        self.weight = weight
        self.scale = scale
        self.bias = bias if bias is not None else None

    def forward(self, input: 'torch.Tensor') ->torch.Tensor:
        output = w8_a16_gemm(input, self.weight, self.scale)
        output = output + self.bias if self.bias is not None else output
        return output


def apply_fp8_linear(input: 'torch.Tensor', qweight: 'torch.Tensor', weight_scale: 'torch.Tensor', input_scale: 'Optional[torch.Tensor]'=None, input_scale_ub: 'Optional[torch.Tensor]'=None, qbias: 'Optional[torch.Tensor]'=None) ->torch.Tensor:
    qinput, x_scale = ops.scaled_fp8_quant(input, input_scale, scale_ub=input_scale_ub, use_per_token_if_dynamic=False)
    output = ops.cutlass_scaled_mm(qinput, qweight, out_dtype=input.dtype, scale_a=x_scale, scale_b=weight_scale, bias=qbias)
    return output


class Fp8Linear(torch.nn.Module):

    def __init__(self, weight, bias, weight_scale, input_scale) ->None:
        super().__init__()
        self.dtype = weight.dtype
        self.qweight = weight.t()
        self.weight_scale = weight_scale.view(1, -1).contiguous().float()
        self.qbias = bias if bias is not None else None
        self.input_scale = input_scale.float() if input_scale is not None else None

    def forward(self, input: 'torch.Tensor') ->torch.Tensor:
        return apply_fp8_linear(input=input, qweight=self.qweight, weight_scale=self.weight_scale, input_scale=self.input_scale, qbias=self.qbias)

    @property
    def weight(self) ->torch.Tensor:
        return self.qweight


class Linear4bit(torch.nn.Module):

    def __init__(self, weight, bias, quant_type):
        super().__init__()
        self.weight = Params4bit(weight.data, requires_grad=False, compress_statistics=True, quant_type=quant_type)
        self.compute_dtype = None
        self.weight
        self.bias = bias

    def forward(self, x: 'torch.Tensor'):
        if self.bias is not None and self.bias.dtype != x.dtype:
            self.bias.data = self.bias.data
        if getattr(self.weight, 'quant_state', None) is None:
            None
        inp_dtype = x.dtype
        if self.compute_dtype is not None:
            x = x
        bias = None if self.bias is None else self.bias
        out = bnb.matmul_4bit(x, self.weight.t(), bias=bias, quant_state=self.weight.quant_state)
        out = out
        return out


class Linear8bitLt(torch.nn.Module):

    def __init__(self, weight, bias, has_fp16_weights=True, memory_efficient_backward=False, threshold=0.0, index=None):
        super().__init__()
        assert not memory_efficient_backward, 'memory_efficient_backward is no longer required and the argument is deprecated in 0.37.0 and will be removed in 0.39.0'
        self.state = bnb.MatmulLtState()
        self.index = index
        self.state.threshold = threshold
        self.state.has_fp16_weights = has_fp16_weights
        self.state.memory_efficient_backward = memory_efficient_backward
        if threshold > 0.0 and not has_fp16_weights:
            self.state.use_pool = True
        self.weight = Int8Params(weight.data, has_fp16_weights=has_fp16_weights, requires_grad=has_fp16_weights)
        self.weight
        self.bias = bias

    def init_8bit_state(self):
        self.state.CB = self.weight.CB
        self.state.SCB = self.weight.SCB
        self.weight.CB = None
        self.weight.SCB = None

    def forward(self, x: 'torch.Tensor'):
        self.state.is_training = self.training
        if self.weight.CB is not None:
            self.init_8bit_state()
        if self.bias is not None and self.bias.dtype != x.dtype:
            self.bias.data = self.bias.data
        out = bnb.matmul(x, self.weight, bias=self.bias, state=self.state)
        if not self.state.has_fp16_weights:
            if self.state.CB is not None and self.state.CxB is not None:
                del self.state.CB
                self.weight.data = self.state.CxB
        return out


def matmul248(input, qweight, scales, qzeros, g_idx, bits, maxq):
    with torch.device(input.device):
        output = torch.empty((input.shape[0], qweight.shape[1]), device=input.device, dtype=torch.float16)
        grid = lambda META: (triton.cdiv(input.shape[0], META['BLOCK_SIZE_M']) * triton.cdiv(qweight.shape[1], META['BLOCK_SIZE_N']),)
        matmul_248_kernel[grid](input, qweight, output, scales, qzeros, g_idx, input.shape[0], qweight.shape[1], input.shape[1], bits, maxq, input.stride(0), input.stride(1), qweight.stride(0), qweight.stride(1), output.stride(0), output.stride(1), scales.stride(0), qzeros.stride(0))
        return output


class QuantLinearFunction(torch.autograd.Function):

    @staticmethod
    @custom_fwd(cast_inputs=torch.float16)
    def forward(ctx, input, qweight, scales, qzeros, g_idx, bits, maxq):
        output = matmul248(input, qweight, scales, qzeros, g_idx, bits, maxq)
        return output


class QuantLinear(nn.Module):

    def __init__(self, qweight, qzeros, scales, g_idx, bias, bits, groupsize):
        super().__init__()
        self.register_buffer('qweight', qweight)
        self.register_buffer('qzeros', qzeros)
        self.register_buffer('scales', scales)
        self.register_buffer('g_idx', g_idx)
        if bias is not None:
            self.register_buffer('bias', bias)
        else:
            self.bias = None
        if bits not in [2, 4, 8]:
            raise NotImplementedError('Only 2,4,8 bits are supported.')
        self.bits = bits
        self.maxq = 2 ** self.bits - 1
        self.groupsize = groupsize
        self.outfeatures = qweight.shape[1]
        self.infeatures = qweight.shape[0] * 32 // 4

    @classmethod
    def new(cls, bits, groupsize, infeatures, outfeatures, bias):
        if bits not in [2, 4, 8]:
            raise NotImplementedError('Only 2,4,8 bits are supported.')
        qweight = torch.zeros((infeatures // 32 * bits, outfeatures), dtype=torch.int32)
        qzeros = torch.zeros((math.ceil(infeatures / groupsize), outfeatures // 32 * bits), dtype=torch.int32)
        scales = torch.zeros((math.ceil(infeatures / groupsize), outfeatures), dtype=torch.float16)
        g_idx = torch.tensor([(i // groupsize) for i in range(infeatures)], dtype=torch.int32)
        if bias:
            bias = torch.zeros(outfeatures, dtype=torch.float16)
        else:
            bias = None
        return cls(qweight, qzeros, scales, g_idx, bias, bits, groupsize)

    def pack(self, linear, scales, zeros, g_idx=None):
        self.g_idx = g_idx.clone() if g_idx is not None else self.g_idx
        scales = scales.t().contiguous()
        zeros = zeros.t().contiguous()
        scale_zeros = zeros * scales
        self.scales = scales.clone().half()
        if linear.bias is not None:
            self.bias = linear.bias.clone().half()
        intweight = []
        for idx in range(self.infeatures):
            intweight.append(torch.round((linear.weight.data[:, idx] + scale_zeros[self.g_idx[idx]]) / self.scales[self.g_idx[idx]])[:, None])
        intweight = torch.cat(intweight, dim=1)
        intweight = intweight.t().contiguous()
        intweight = intweight.numpy().astype(np.uint32)
        qweight = np.zeros((intweight.shape[0] // 32 * self.bits, intweight.shape[1]), dtype=np.uint32)
        i = 0
        row = 0
        while row < qweight.shape[0]:
            if self.bits in [2, 4, 8]:
                for j in range(i, i + 32 // self.bits):
                    qweight[row] |= intweight[j] << self.bits * (j - i)
                i += 32 // self.bits
                row += 1
            else:
                raise NotImplementedError('Only 2,4,8 bits are supported.')
        qweight = qweight.astype(np.int32)
        self.qweight = torch.from_numpy(qweight)
        zeros -= 1
        zeros = zeros.numpy().astype(np.uint32)
        qzeros = np.zeros((zeros.shape[0], zeros.shape[1] // 32 * self.bits), dtype=np.uint32)
        i = 0
        col = 0
        while col < qzeros.shape[1]:
            if self.bits in [2, 4, 8]:
                for j in range(i, i + 32 // self.bits):
                    qzeros[:, col] |= zeros[:, j] << self.bits * (j - i)
                i += 32 // self.bits
                col += 1
            else:
                raise NotImplementedError('Only 2,4,8 bits are supported.')
        qzeros = qzeros.astype(np.int32)
        self.qzeros = torch.from_numpy(qzeros)

    def forward(self, x):
        out_shape = x.shape[:-1] + (self.outfeatures,)
        out = QuantLinearFunction.apply(x.reshape(-1, x.shape[-1]), self.qweight, self.scales, self.qzeros, self.g_idx, self.bits, self.maxq)
        out = out + self.bias if self.bias is not None else out
        return out.reshape(out_shape)

    @property
    def weight(self) ->torch.Tensor:
        return self.qweight


def is_fp8(quantize):
    return quantize and quantize.startswith('fp8')


def get_linear(weight, bias, quantize, fan_in_fan_out=False, weight_scale=None, input_scale=None):
    if fan_in_fan_out:
        weight = weight.T.contiguous()
    if quantize is None:
        linear = FastLinear(weight, bias)
    elif is_fp8(quantize):
        linear = Fp8Linear(weight, bias, weight_scale=weight_scale, input_scale=input_scale)
    elif quantize == 'bitsandbytes':
        linear = Linear8bitLt(weight, bias, has_fp16_weights=False, threshold=6.0)
        if bias is not None:
            linear.bias = nn.Parameter(bias)
    elif quantize == 'bitsandbytes-nf4':
        linear = Linear4bit(weight, bias, quant_type='nf4')
    elif quantize == 'bitsandbytes-fp4':
        linear = Linear4bit(weight, bias, quant_type='fp4')
    elif quantize == 'eetq':
        linear = EETQLinear(weight, bias)
    elif quantize == 'gptq':
        try:
            qweight, qzeros, scales, g_idx, bits, groupsize, use_exllama = weight
        except Exception:
            raise NotImplementedError('The passed weight is not `gptq` compatible, loader needs to be updated.')
        if use_exllama:
            linear = exllamav2QuantLinear(qweight, qzeros, scales, g_idx, bias, bits, groupsize)
        else:
            linear = QuantLinear(qweight, qzeros, scales, g_idx, bias, bits, groupsize)
    elif quantize == 'awq':
        try:
            qweight, qzeros, scales, _, bits, groupsize, _ = weight
        except Exception:
            raise NotImplementedError('The passed weight is not compatible with `awq`')
        linear = AWQLinear(w_bit=bits, group_size=groupsize, qweight=qweight, qzeros=qzeros, scales=scales, bias=bias)
    elif 'hqq-' in quantize:
        linear = get_hqq_linear(quantize, weight, bias)
    else:
        raise NotImplementedError(f'Quantization `{quantize}` is not implemented yet.')
    return linear


class TensorParallelColumnLinear(SuperLayer):

    @classmethod
    def load_gate_up(cls, config, prefix: 'str', weights, bias: 'bool'):
        """Specific method when the QKV was joined after the fact"""
        weight = weights.get_weights_col_packed_gate_up(prefix, quantize=config.quantize)
        if bias:
            raise NotImplementedError('packed_gate_up only implemented without bias')
        else:
            bias = None
        linear = get_linear(weight, bias, config.quantize)
        return cls(linear)

    @classmethod
    def load_qkv(cls, config, prefix: 'str', weights, bias: 'bool'):
        """Specific method when the QKV was joined after the fact"""
        weight = weights.get_weights_col_packed_qkv(prefix, quantize=config.quantize)
        if bias:
            raise NotImplementedError('packed_qkv only implemented for baichuan')
        else:
            bias = None
        linear = get_linear(weight, bias, config.quantize)
        return cls(linear)

    @classmethod
    def load(cls, config, prefix: 'str', weights, bias: 'bool'):
        return cls.load_multi(config, [prefix], weights, bias, dim=0)

    @classmethod
    def load_multi(cls, config, prefixes: 'List[str]', weights, bias: 'bool', dim: 'int'):
        weight = weights.get_multi_weights_col(prefixes, quantize=config.quantize, dim=dim)
        input_scale, weight_scale = None, None
        if type(weight) is tuple:
            weight, input_scale, weight_scale = weight
        if bias:
            b = [weights.get_sharded(f'{p}.bias', dim=0) for p in prefixes]
            bias = torch.cat(b, dim=dim)
        else:
            bias = None
        linear = get_linear(weight, bias, config.quantize, weight_scale=weight_scale, input_scale=input_scale)
        return cls(linear)


logger = getLogger(__name__)


def segmented_matmul(y: 'torch.Tensor', x: 'torch.Tensor', w: 'List[torch.Tensor]', b: 'List[torch.Tensor]', s_start: 'torch.IntTensor', s_end: 'torch.IntTensor'):
    for i in range(len(w)):
        if s_end[i] - s_start[i] <= 0:
            continue
        xi = x[s_start[i]:s_end[i]]
        wi = w[i]
        bi = b[i]
        y[s_start[i]:s_end[i]] = F.linear(xi, wi, bi)


class MedusaV2(torch.nn.Module):

    def __init__(self, config: 'MedusaConfig', weights: 'AbstractWeights'):
        super().__init__()
        self.n_medusa_heads = config.medusa_num_heads
        assert config.medusa_num_layers == 1
        self.linear = TensorParallelColumnLinear.load_multi(config, prefixes=[f'{i}.0.linear' for i in range(self.n_medusa_heads)], dim=0, weights=weights, bias=True)
        self.process_group = weights.process_group
        self.world_size = self.process_group.size()
        self.rank = self.process_group.rank()
        self.act = torch.nn.SiLU()

    def forward(self, x, lm_head, segments: 'Optional[MedusaSegments]'=None):
        if x.shape[0] > LORAX_SPECULATION_MAX_BATCH_SIZE:
            logger.info(f'Skipping speculation at batch size = {x.shape[0]}')
            logits = lm_head(x)
            return logits, None
        size = x.shape[-1]
        block_size = (size + self.world_size - 1) // self.world_size
        start = self.rank * block_size
        stop = (self.rank + 1) * block_size
        x_block = x[:, start:stop]
        if segments is not None:
            y = torch.empty((x.shape[0], self.n_medusa_heads * x_block.shape[-1]), device=x.device, dtype=x.dtype)
            segmented_matmul(y, x, segments.w, segments.b, segments.s_start, segments.s_end)
        else:
            y = self.linear(x)
        medusa_res = self.act(y).reshape(*x_block.shape[:-1], self.n_medusa_heads, x_block.shape[-1])
        output = x[:, start:stop].unsqueeze(-2) + medusa_res
        world_output = [torch.empty_like(output) for _ in range(self.process_group.size())]
        torch.distributed.all_gather(world_output, output, group=self.process_group)
        world_output = torch.cat(world_output, dim=-1)
        stacked_x = torch.cat([x.unsqueeze(-2), world_output], dim=-2)
        logits = lm_head(stacked_x)
        logits, speculative_logits = torch.split(logits, [1, self.n_medusa_heads], dim=-2)
        logits = logits.squeeze(-2)
        return logits, speculative_logits


class MedusaModel(torch.nn.Module):

    def __init__(self, config: 'MedusaConfig', weights: 'AbstractWeights'):
        super().__init__()
        if config.medusa_num_layers > 1 or weights.has_tensor(f'0.{config.medusa_num_layers}.weight'):
            self.medusa = MedusaV1(config, weights)
        else:
            self.medusa = MedusaV2(config, weights)

    def forward(self, x, lm_head, segments: 'Optional[MedusaSegments]'=None):
        return self.medusa(x, lm_head, segments)


class WQLinear(nn.Module):

    def __init__(self, w_bit, group_size, qweight, qzeros, scales, bias):
        super().__init__()
        if w_bit not in [4]:
            raise NotImplementedError('Only 4-bit are supported for now.')
        self.in_features = qweight.shape[0]
        self.out_features = qweight.shape[1] * 32 // w_bit
        self.w_bit = w_bit
        self.group_size = group_size if group_size != -1 else self.in_features
        assert self.in_features % self.group_size == 0
        assert self.out_features % (32 // self.w_bit) == 0
        self.qweight = qweight
        self.qzeros = qzeros
        self.scales = scales
        if bias:
            self.bias = bias
        else:
            self.bias = None

    @torch.no_grad()
    def forward(self, x):
        out_shape = x.shape[:-1] + (self.out_features,)
        out = awq_inference_engine.gemm_forward_cuda(x.reshape(-1, x.shape[-1]), self.qweight, self.scales, self.qzeros, 8)
        out = out + self.bias if self.bias is not None else out
        return out.reshape(out_shape)

    @property
    def weight(self) ->torch.Tensor:
        return self.qweight


none_tensor = torch.empty((1, 1), device='meta')


def ext_make_q4(qweight, qzeros, scales, g_idx, device):
    """Construct Q4Matrix, return handle"""
    return make_q4(qweight, qzeros, scales, g_idx if g_idx is not None else none_tensor, device)


def ext_q4_matmul(x, q4, q4_width):
    """Matrix multiplication, returns x @ q4"""
    outshape = x.shape[:-1] + (q4_width,)
    x = x.view(-1, x.shape[-1])
    output = torch.empty((x.shape[0], q4_width), dtype=torch.float16, device=x.device)
    q4_matmul(x, q4, output)
    return output.view(outshape)


class Ex4bitLinear(torch.nn.Module):
    """Linear layer implementation with per-group 4-bit quantization of the weights"""

    def __init__(self, qweight, qzeros, scales, g_idx, bias, bits, groupsize):
        super().__init__()
        global MAX_DQ, MAX_INNER, ACT_ORDER, DEVICE
        assert bits == 4
        self.device = qweight.device
        self.qweight = qweight
        self.qzeros = qzeros
        self.scales = scales
        self.g_idx = g_idx.cpu() if g_idx is not None else None
        self.bias = bias if bias is not None else None
        if self.g_idx is not None and ((self.g_idx == 0).all() or torch.equal(g_idx.cpu(), torch.tensor([(i // groupsize) for i in range(g_idx.shape[0])], dtype=torch.int32))):
            self.empty_g_idx = True
            self.g_idx = None
        assert self.device.type == 'cuda'
        assert self.device.index is not None
        self.q4 = ext_make_q4(self.qweight, self.qzeros, self.scales, self.g_idx, self.device.index)
        self.height = qweight.shape[0] * 8
        self.width = qweight.shape[1]
        self.groupsize = None
        if self.qzeros.shape[0] > 1:
            self.groupsize = self.qweight.shape[0] * 8 // self.qzeros.shape[0]
        if self.groupsize is not None:
            assert groupsize == self.groupsize
        if self.g_idx is not None:
            if self.groupsize is None:
                raise ValueError('Found group index but no groupsize. What do?')
            self.act_order = True
        else:
            self.act_order = False
        DEVICE = self.qweight.device
        MAX_DQ = max(MAX_DQ, self.qweight.numel() * 8)
        if self.act_order:
            MAX_INNER = max(MAX_INNER, self.height, self.width)
            ACT_ORDER = True

    def forward(self, x):
        out = ext_q4_matmul(x, self.q4, self.width)
        if self.bias is not None:
            out.add_(self.bias)
        return out


SYSTEM = None


class FastRMSNorm(nn.Module):

    def __init__(self, weight: 'torch.Tensor', eps: 'float'):
        super().__init__()
        self.weight = nn.Parameter(weight)
        self.variance_epsilon = eps

    @classmethod
    def load(cls, prefix, weights, eps=1e-06):
        weight = weights.get_tensor(f'{prefix}.weight')
        return cls(weight, eps)

    def forward(self, hidden_states, residual=None):
        if SYSTEM == 'xpu':
            residual_out = hidden_states
            out = ipex.llm.functional.add_rms_norm(residual, hidden_states, self.weight, None, self.variance_epsilon, True)
            if residual is not None:
                residual_out = residual
            return out, residual_out
        elif hidden_states.shape[-1] > 8192:
            if residual is not None:
                hidden_states += residual
            residual = hidden_states
            hidden_states = hidden_states
            variance = hidden_states.pow(2).mean(-1, keepdim=True)
            hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
            if self.weight.dtype in [torch.float16, torch.bfloat16]:
                hidden_states = hidden_states
            return self.weight * hidden_states, residual
        elif SYSTEM == 'cuda':
            normed_hidden_states, res, *rest = dropout_layer_norm.dropout_add_ln_fwd(hidden_states, residual, self.weight, None, None, None, None, None, 0.0, self.variance_epsilon, 1.0, 0, None, False, True)
            if res is None:
                res = hidden_states
            return normed_hidden_states, res
        elif SYSTEM == 'rocm':
            if residual is not None:
                hidden_states += residual
            residual = hidden_states
            out = torch.empty_like(hidden_states)
            ops.rms_norm(out, hidden_states, self.weight.data, self.variance_epsilon)
            return out, residual
        else:
            raise ValueError('Your system seem to be not supported. Please check your install or open an issue at https://github.com/huggingface/text-generation-inference/issues with a clear reproduction.')


class FastLinearROCm(torch.nn.Module):

    def __init__(self, weight, bias) ->None:
        super().__init__()
        self.weight = torch.nn.Parameter(weight)
        if bias is not None:
            self.bias = torch.nn.Parameter(bias)
        else:
            self.bias = None

    @classmethod
    def load(cls, config, prefix: 'str', weights, bias: 'bool'):
        weight = weights.get_tensor(f'{prefix}.weight')
        if bias:
            bias = weights.get_tensor(f'{prefix}.bias')
        else:
            bias = None
        return cls(weight, bias)

    def forward(self, inp: 'torch.Tensor') ->torch.Tensor:
        weight = self.weight
        bias = self.bias
        if SYSTEM == 'rocm' and inp.numel() // inp.shape[-1] == 1:
            batched = False
            inp_shape = inp.shape
            if inp.dim() == 3:
                inp = inp.view(-1, inp_shape[-1])
                batched = True
            m, k = weight.shape[0], inp_shape[1]
            out = torch.empty(inp_shape[0], weight.shape[0], dtype=inp.dtype, device='cuda')
            if k == 8192 and (m == 1280 or m == 7168) or k == 3584 and m == 8192:
                _custom_C.LLMM1(weight, inp, out, 8)
            elif k <= 8192 and k % 8 == 0 and m % 4 == 0:
                _custom_C.LLMM1(weight, inp, out, 4)
            else:
                out = F.linear(inp, weight)
            if batched:
                out.view(*inp_shape[:-1], out.shape[-1])
            if bias is not None:
                out = out + bias
            return out
        return F.linear(inp, self.weight, self.bias)

