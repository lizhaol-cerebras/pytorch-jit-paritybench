import sys
_module = sys.modules[__name__]
del sys
conftest = _module
test_client = _module
test_errors = _module
test_inference_api = _module
test_types = _module
text_generation = _module
client = _module
errors = _module
inference_api = _module
types = _module
test_bloom_560m = _module
test_bloom_560m_sharded = _module
test_chat_llama = _module
test_completion_prompts = _module
test_flash_awq = _module
test_flash_awq_sharded = _module
test_flash_deepseek_v2 = _module
test_flash_falcon = _module
test_flash_gemma = _module
test_flash_gemma2 = _module
test_flash_gemma_gptq = _module
test_flash_gpt2 = _module
test_flash_grammar_llama = _module
test_flash_llama = _module
test_flash_llama_exl2 = _module
test_flash_llama_fp8 = _module
test_flash_llama_fp8_kv_cache = _module
test_flash_llama_gptq = _module
test_flash_llama_marlin = _module
test_flash_llama_marlin_24 = _module
test_flash_llama_prefix = _module
test_flash_llama_prefix_flashdecoding = _module
test_flash_medusa = _module
test_flash_mistral = _module
test_flash_mixtral = _module
test_flash_mixtral_awq = _module
test_flash_mixtral_gptq = _module
test_flash_neox = _module
test_flash_neox_sharded = _module
test_flash_pali_gemma = _module
test_flash_phi = _module
test_flash_phi35_moe = _module
test_flash_qwen2 = _module
test_flash_qwen2_vl = _module
test_flash_santacoder = _module
test_flash_starcoder = _module
test_flash_starcoder2 = _module
test_flash_starcoder_gptq = _module
test_grammar_llama = _module
test_grammar_response_format_llama = _module
test_idefics = _module
test_idefics2 = _module
test_llava_next = _module
test_lora_mistral = _module
test_mamba = _module
test_mllama = _module
test_mpt = _module
test_mt0_base = _module
test_neox = _module
test_neox_sharded = _module
test_opt = _module
test_t5_sharded = _module
test_tools_llama = _module
filter = _module
orca = _module
setup = _module
setup = _module
setup = _module
test_bloom = _module
test_causal_lm = _module
test_model = _module
test_santacoder = _module
test_seq2seq_lm = _module
test_adapter = _module
test_convert = _module
test_hub = _module
test_layers = _module
test_tokens = _module
test_watermark = _module
test_weights = _module
text_generation_server = _module
adapters = _module
config = _module
lora = _module
weights = _module
cache = _module
cli = _module
interceptor = _module
layers = _module
attention = _module
common = _module
cuda = _module
flash_attn_triton = _module
flashinfer = _module
ipex = _module
kv_cache = _module
rocm = _module
conversion_utils = _module
quantize = _module
cuda = _module
ipex = _module
bnb = _module
conv = _module
eetq = _module
exl2 = _module
fp8 = _module
gptq = _module
custom_autotune = _module
exllama = _module
exllamav2 = _module
ipex = _module
quantize = _module
triton = _module
utils = _module
layernorm = _module
linear = _module
lora = _module
marlin = _module
fp8 = _module
gptq = _module
marlin = _module
util = _module
medusa = _module
mlp = _module
moe = _module
fused_moe_rocm = _module
gptq_marlin = _module
unquantized = _module
rotary = _module
speculative = _module
tensor_parallel = _module
models = _module
bloom = _module
causal_lm = _module
custom_modeling = _module
bloom_modeling = _module
clip = _module
flash_cohere_modeling = _module
flash_dbrx_modeling = _module
flash_deepseek_v2_modeling = _module
flash_gemma2_modeling = _module
flash_gemma_modeling = _module
flash_gpt2_modeling = _module
flash_gptj_modeling = _module
flash_llama_modeling = _module
flash_mistral_modeling = _module
flash_mixtral_modeling = _module
flash_neox_modeling = _module
flash_pali_gemma_modeling = _module
flash_phi_modeling = _module
flash_phi_moe_modeling = _module
flash_qwen2_modeling = _module
flash_rw_modeling = _module
flash_santacoder_modeling = _module
flash_starcoder2_modeling = _module
idefics2 = _module
idefics_config = _module
idefics_image_processing = _module
idefics_modeling = _module
idefics_perceiver = _module
idefics_processing = _module
idefics_vision = _module
llava_next = _module
mamba_modeling = _module
mllama = _module
mpt_modeling = _module
neox_modeling = _module
opt_modeling = _module
phi_modeling = _module
qwen2_vl = _module
siglip = _module
t5_modeling = _module
vlm = _module
flash_causal_lm = _module
galactica = _module
globals = _module
idefics_causal_lm = _module
mamba = _module
metadata_kernels = _module
mllama_causal_lm = _module
model = _module
pali_gemma = _module
seq2seq_lm = _module
types = _module
vlm_causal_lm = _module
server = _module
tracing = _module
adapter = _module
chunks = _module
convert = _module
dist = _module
hub = _module
import_utils = _module
log = _module
logits_process = _module
strategies = _module
utils = _module
peft = _module
prefill_chunking = _module
quantization = _module
segments = _module
sgmv = _module
speculate = _module
tokens = _module
watermark = _module
weights = _module
update_doc = _module

from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, matplotlib, numbers, numpy, pandas, queue, random, re, scipy, sklearn, string, tensorflow, time, torch, torchaudio, torchvision, types, typing, uuid, warnings
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


from copy import copy


import numpy as np


from types import SimpleNamespace


from typing import List


from typing import Optional


from typing import Dict


from typing import Union


from abc import ABC


from abc import abstractmethod


from typing import Set


from typing import Tuple


from collections import defaultdict


from typing import Type


from torch.distributed import ProcessGroup


from abc import abstractclassmethod


from typing import TypeVar


from typing import Callable


from typing import Any


import torch.nn as nn


import math


import time


from torch.cuda.amp import custom_fwd


from torch import nn


from torch.nn import functional as F


from typing import TYPE_CHECKING


import torch.distributed


import numpy


import functools


from typing import Protocol


from typing import runtime_checkable


from typing import Iterable


import enum


import warnings


import torch.utils.checkpoint


from torch.nn import LayerNorm


import torch.nn.functional as F


from torch.nn import CrossEntropyLoss


import random


from torch.nn.init import _calculate_fan_in_and_fan_out


import copy


from typing import ContextManager


import re


from torch.utils._triton import has_triton as has_triton_torch


import inspect


from functools import lru_cache


from typing import DefaultDict


from typing import Literal


class WQLinear(nn.Module):

    def __init__(self, w_bit, group_size, qweight, qzeros, scales, bias: 'Optional[torch.Tensor]'):
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
        self.bias = bias
        self.woq_linear = ipex.llm.quantization.IPEXWeightOnlyQuantizedLinear.from_weight(self.qweight, self.scales, self.qzeros, self.in_features, self.out_features, bias=self.bias, group_size=self.group_size, quant_method=ipex.llm.quantization.QuantMethod.AWQ_GEMM, dtype=ipex.llm.quantization.QuantDtype.INT4)

    @torch.no_grad()
    def forward(self, x):
        out_shape = x.shape[:-1] + (self.out_features,)
        out = self.woq_linear(x.reshape(-1, x.shape[-1]))
        return out.reshape(out_shape)


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


SYSTEM = None


def normalize_e4m3fn_to_e4m3fnuz(weight: 'torch.Tensor', weight_scale: 'torch.Tensor', input_scale: 'Optional[torch.Tensor]'=None) ->Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    assert weight.dtype == torch.float8_e4m3fn
    weight_as_int8 = weight.view(torch.int8)
    ROCM_FP8_NAN_AS_INT = -128
    weight_as_int8[weight_as_int8 == ROCM_FP8_NAN_AS_INT] = 0
    weight = weight_as_int8.view(torch.float8_e4m3fnuz)
    weight_scale = weight_scale * 2.0
    if input_scale is not None:
        input_scale = input_scale * 2.0
    return weight, weight_scale, input_scale


def fp8_quantize(weight: 'torch.Tensor', scale: 'Optional[torch.Tensor]'=None, scale_upper_bound: 'Optional[torch.Tensor]'=None, qdtype: 'torch.dtype'=torch.float8_e4m3fn, scalar: 'bool'=False):
    """
    This function returns a reciprocal of the scale, so that a tensor can be unscaled
    by multiplying it with the returned scale. If a scale is given through the `scale`
    argument, it must also be a reciprocal (so that scales from an FP8 checkpoint can
    be used without modification).
    """
    if marlin_kernels is not None:
        shape = weight.shape
        qweight, scale = marlin_kernels.scaled_fp8_quant(weight.reshape(-1, shape[-1]), dtype=qdtype, scale=scale, scale_ub=scale_upper_bound, use_per_token_if_dynamic=not scalar)
        return qweight.reshape(shape), scale
    finfo = torch.finfo(qdtype)
    if scale is None:
        scale = finfo.max / weight.abs().max().clamp(min=1e-12, max=scale_upper_bound)
        qweight = (weight * scale).clamp(min=finfo.min, max=finfo.max)
        scale = scale.float().reciprocal()
    else:
        qweight = (weight * scale.reciprocal()).clamp(min=finfo.min, max=finfo.max)
    qweight = qweight
    if SYSTEM == 'rocm':
        qweight, scale, _ = normalize_e4m3fn_to_e4m3fnuz(qweight, scale)
    return qweight, scale


def log_master(log, msg: 'str'):
    if RANK == 0:
        log(msg)


@lru_cache(10)
def log_once(log, msg: 'str', master=True):
    if master:
        log_master(log, msg)
    else:
        log(msg)


class Fp8Linear(torch.nn.Module):
    _device_identity_cache = {}

    def __init__(self, qweight: 'torch.Tensor', scale: 'torch.Tensor', dtype: 'torch.dtype', bias: 'Optional[torch.Tensor]'=None, input_scale: 'Optional[torch.Tensor]'=None, scale_upper_bound: 'Optional[float]'=None) ->None:
        super().__init__()
        if CUTLASS_FP8_AVAILABLE:
            log_once(logger.info, 'Using cutlass w8a8 kernels')
        if SYSTEM == 'rocm' and qweight.dtype == torch.float8_e4m3fn:
            qweight, scale, _ = normalize_e4m3fn_to_e4m3fnuz(weight=qweight, weight_scale=scale)
        self.dtype = dtype
        self.qweight = qweight
        self.scale = scale.float()
        self.input_scale = input_scale.float() if input_scale is not None else None
        if CUTLASS_FP8_AVAILABLE and scale_upper_bound is not None:
            self.scale_upper_bound = torch.tensor(scale_upper_bound, dtype=torch.float32, device=qweight.device)
        else:
            self.scale_upper_bound = scale_upper_bound
        self.bias = bias if bias is not None else None

    @classmethod
    def from_unquant(cls, weight, bias, dtype):
        qweight, scale = fp8_quantize(weight, scalar=not CUTLASS_FP8_AVAILABLE)
        return cls(qweight=qweight, scale=scale, dtype=dtype, bias=bias, input_scale=None, scale_upper_bound=None)

    @classmethod
    def from_fp8(cls, weight: 'torch.Tensor', scale: 'torch.Tensor', dtype: 'torch.dtype', bias: 'Optional[torch.Tensor]'=None, **kwargs) ->'Fp8Linear':
        input_scale = kwargs.get('input_scale', None)
        scale_upper_bound = kwargs.get('scale_upper_bound', None)
        return cls(qweight=weight, scale=scale, input_scale=input_scale, scale_upper_bound=scale_upper_bound, bias=bias, dtype=dtype)

    @classmethod
    def get_shared_device_identity(cls, device):
        if device not in cls._device_identity_cache:
            cls._device_identity_cache[device] = torch.ones(1, device=device)
        return cls._device_identity_cache[device]

    def forward(self, input: 'torch.Tensor') ->torch.Tensor:
        if CUTLASS_FP8_AVAILABLE:
            qinput, scale = fp8_quantize(input, scale_upper_bound=self.scale_upper_bound, scalar=False)
            return marlin_kernels.cutlass_scaled_mm(qinput, self.qweight.t(), scale, self.scale, input.dtype, self.bias)
        qinput, scale = fp8_quantize(input, self.input_scale, scale_upper_bound=self.scale_upper_bound, scalar=True)
        per_tensor_weights = self.scale.numel() == 1
        per_tensor_activations = scale.numel() == 1
        if SYSTEM != 'rocm' or per_tensor_weights and per_tensor_activations:
            output = torch._scaled_mm(qinput, self.qweight.t(), out_dtype=self.dtype, scale_a=scale, scale_b=self.scale, bias=self.bias)
            if isinstance(output, tuple) and len(output) == 2:
                output = output[0]
        else:
            device_identity = None
            if SYSTEM == 'rocm':
                device_identity = self.get_shared_device_identity(self.qweight.device)
            output = torch._scaled_mm(qinput, self.qweight.t(), scale_a=device_identity, scale_b=device_identity, out_dtype=torch.float32)
            if isinstance(output, tuple) and len(output) == 2:
                output = output[0]
            output = output * scale * self.scale.t()
            if self.bias is not None:
                output = output + self.bias
            output = output
        return output


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

    def __init__(self, weight: 'GPTQWeight', bias):
        super().__init__()
        global MAX_DQ, MAX_INNER, ACT_ORDER, DEVICE
        assert weight.bits == 4
        self.device = weight.qweight.device
        self.qweight = weight.qweight
        self.qzeros = weight.qzeros
        self.scales = weight.scales
        self.g_idx = weight.g_idx.cpu() if weight.g_idx is not None else None
        self.bias = bias if bias is not None else None
        if self.g_idx is not None and ((self.g_idx == 0).all() or torch.equal(weight.g_idx.cpu(), torch.tensor([(i // weight.groupsize) for i in range(weight.g_idx.shape[0])], dtype=torch.int32))):
            self.empty_g_idx = True
            self.g_idx = None
        assert self.device.type == 'cuda'
        assert self.device.index is not None
        self.q4 = ext_make_q4(self.qweight, self.qzeros, self.scales, self.g_idx, self.device.index)
        self.height = weight.qweight.shape[0] * 8
        self.width = weight.qweight.shape[1]
        self.groupsize = None
        if self.qzeros.shape[0] > 1:
            self.groupsize = self.qweight.shape[0] * 8 // self.qzeros.shape[0]
        if self.groupsize is not None:
            assert weight.groupsize == self.groupsize
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


def matmul248(input, qweight, scales, qzeros, g_idx, bits, maxq):
    with torch.device(input.device):
        output = torch.empty((input.shape[0], qweight.shape[1]), device=input.device, dtype=torch.float16)

        def grid(META):
            return triton.cdiv(input.shape[0], META['BLOCK_SIZE_M']) * triton.cdiv(qweight.shape[1], META['BLOCK_SIZE_N']),
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
        self.infeatures = qweight.shape[0] * 32 // bits

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


class Quantizer(nn.Module):

    def __init__(self, shape=1):
        super(Quantizer, self).__init__()
        self.register_buffer('maxq', torch.tensor(0))
        self.register_buffer('scale', torch.zeros(shape))
        self.register_buffer('zero', torch.zeros(shape))

    def configure(self, bits, perchannel=False, sym=True, mse=False, norm=2.4, grid=100, maxshrink=0.8, trits=False):
        self.maxq = torch.tensor(2 ** bits - 1)
        self.perchannel = perchannel
        self.sym = sym
        self.mse = mse
        self.norm = norm
        self.grid = grid
        self.maxshrink = maxshrink
        if trits:
            self.maxq = torch.tensor(-1)
        self.scale = torch.zeros_like(self.scale)

    def _quantize(self, x, scale, zero, maxq):
        if maxq < 0:
            return (x > scale / 2).float() * scale + (x < zero / 2).float() * zero
        q = torch.clamp(torch.round(x / scale) + zero, 0, maxq)
        return scale * (q - zero)

    def find_params(self, x, weight=False):
        dev = x.device
        self.maxq = self.maxq
        shape = x.shape
        if self.perchannel:
            if weight:
                x = x.flatten(1)
            else:
                if len(shape) == 4:
                    x = x.permute([1, 0, 2, 3])
                    x = x.flatten(1)
                if len(shape) == 3:
                    x = x.reshape((-1, shape[-1])).t()
                if len(shape) == 2:
                    x = x.t()
        else:
            x = x.flatten().unsqueeze(0)
        tmp = torch.zeros(x.shape[0], device=dev)
        xmin = torch.minimum(x.min(1)[0], tmp)
        xmax = torch.maximum(x.max(1)[0], tmp)
        if self.sym:
            xmax = torch.maximum(torch.abs(xmin), xmax)
            tmp = xmin < 0
            if torch.any(tmp):
                xmin[tmp] = -xmax[tmp]
        tmp = (xmin == 0) & (xmax == 0)
        xmin[tmp] = -1
        xmax[tmp] = +1
        if self.maxq < 0:
            self.scale = xmax
            self.zero = xmin
        else:
            self.scale = (xmax - xmin) / self.maxq
            if self.sym:
                self.zero = torch.full_like(self.scale, (self.maxq + 1) / 2)
            else:
                self.zero = torch.round(-xmin / self.scale)
        if self.mse:
            best = torch.full([x.shape[0]], float('inf'), device=dev)
            for i in range(int(self.maxshrink * self.grid)):
                p = 1 - i / self.grid
                xmin1 = p * xmin
                xmax1 = p * xmax
                scale1 = (xmax1 - xmin1) / self.maxq
                zero1 = torch.round(-xmin1 / scale1) if not self.sym else self.zero
                q = self._quantize(x, scale1.unsqueeze(1), zero1.unsqueeze(1), self.maxq)
                q -= x
                q.abs_()
                q.pow_(self.norm)
                err = torch.sum(q, 1)
                tmp = err < best
                if torch.any(tmp):
                    best[tmp] = err[tmp]
                    self.scale[tmp] = scale1[tmp]
                    self.zero[tmp] = zero1[tmp]
        if not self.perchannel:
            if weight:
                tmp = shape[0]
            else:
                tmp = shape[1] if len(shape) != 3 else shape[2]
            self.scale = self.scale.repeat(tmp)
            self.zero = self.zero.repeat(tmp)
        if weight:
            shape = [-1] + [1] * (len(shape) - 1)
            self.scale = self.scale.reshape(shape)
            self.zero = self.zero.reshape(shape)
            return
        if len(shape) == 4:
            self.scale = self.scale.reshape((1, -1, 1, 1))
            self.zero = self.zero.reshape((1, -1, 1, 1))
        if len(shape) == 3:
            self.scale = self.scale.reshape((1, 1, -1))
            self.zero = self.zero.reshape((1, 1, -1))
        if len(shape) == 2:
            self.scale = self.scale.unsqueeze(0)
            self.zero = self.zero.unsqueeze(0)

    def quantize(self, x):
        if self.ready():
            return self._quantize(x, self.scale, self.zero, self.maxq)
        return x

    def enabled(self):
        return self.maxq > 0

    def ready(self):
        return torch.all(self.scale != 0)


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
        if SYSTEM == 'ipex':
            out = ipex.llm.functional.add_rms_norm(residual, hidden_states, self.weight, None, self.variance_epsilon, residual is not None)
            return out, residual if residual is not None else hidden_states
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


class FastLinearROCm(torch.nn.Module):

    def __init__(self, weight, bias) ->None:
        super().__init__()
        self.weight = torch.nn.Parameter(weight)
        if bias is not None:
            self.bias = torch.nn.Parameter(bias)
        else:
            self.bias = None
        self.cu_count = torch.cuda.get_device_properties(device='cuda').multi_processor_count
        self.use_skinny_gemm = ROCM_USE_SKINNY_GEMM and 'gfx1' not in torch.cuda.get_device_properties('cuda').gcnArchName

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
        if self.use_skinny_gemm and inp.dtype == torch.float16 and inp.shape[-1] % 8 == 0:
            batched = False
            inp_shape = inp.shape
            if inp.dim() == 3:
                inp = inp.view(-1, inp_shape[-1])
                batched = True
            m, n, k = weight.shape[0], inp_shape[0], inp_shape[1]
            if m > 8 and n <= 4:
                out = torch.empty(inp_shape[0], weight.shape[0], dtype=inp.dtype, device=weight.device)
                _custom_C.wvSpltK(weight, inp, out, n, self.cu_count)
            elif m % 4 == 0 and n == 1 and k <= 8192:
                out = torch.empty(inp_shape[0], weight.shape[0], dtype=inp.dtype, device=weight.device)
                _custom_C.LLMM1(weight, inp, out, 4)
            else:
                out = F.linear(inp, weight)
            if batched:
                out.view(*inp_shape[:-1], out.shape[-1])
            if bias is not None:
                out = out + bias
            return out
        return F.linear(inp, self.weight, self.bias)


def add_lora_a_bgmv(v: 'torch.Tensor', x: 'torch.Tensor', wa_T_all: 'torch.Tensor', indicies: 'torch.LongTensor', layer_idx: 'int'):
    _kernels.dispatch_bgmv(v, x, wa_T_all, indicies, layer_idx, 1.0)


def add_lora_b_bgmv(y: 'torch.Tensor', v: 'torch.Tensor', wb_T_all: 'torch.Tensor', indicies: 'torch.LongTensor', layer_idx: 'int'):
    _kernels.dispatch_bgmv(y, v, wb_T_all, indicies, layer_idx, 1.0)


def has_sgmv() ->bool:
    return HAS_SGMV


MAX_RANK_CUSTOM = 128


MIN_RANK_CUSTOM = 16


def lora_a_sgmv_cutlass(x: 'torch.Tensor', tmp: 'torch.Tensor', wa_ptr: 'torch.Tensor', s_start: 'torch.IntTensor', s_end: 'torch.IntTensor', layer_idx: 'int', lora_rank: 'int') ->torch.Tensor:
    v = torch.zeros((x.size(0), lora_rank), dtype=x.dtype, device=x.device)
    if MIN_RANK_CUSTOM <= lora_rank <= MAX_RANK_CUSTOM:
        _kernels.sgmv_shrink(v, x, wa_ptr, s_start, s_end, tmp, layer_idx)
    else:
        _kernels.sgmv_cutlass(v, x, wa_ptr, s_start, s_end, tmp, layer_idx)
    return v


def lora_b_sgmv_cutlass(y: 'torch.Tensor', v: 'torch.Tensor', tmp: 'torch.Tensor', wb_ptr: 'torch.Tensor', s_start: 'torch.IntTensor', s_end: 'torch.IntTensor', layer_idx: 'int'):
    _kernels.sgmv_cutlass(y, v, wb_ptr, s_start, s_end, tmp, layer_idx)


def orient_for_rank(t: 'torch.Tensor', rank: 'int') ->torch.Tensor:
    if MIN_RANK_CUSTOM <= rank <= MAX_RANK_CUSTOM:
        return t.transpose(0, 1)
    return t


class LoraLinear(nn.Module):

    def __init__(self, base_layer: 'nn.Module', layer_id: 'int', process_group: 'ProcessGroup'):
        super().__init__()
        self.base_layer = base_layer
        self.layer_id = layer_id
        self.process_group = process_group

    def forward_layer_type(self, result: 'torch.Tensor', input: 'torch.Tensor', adapter_data: "'AdapterBatchData'", layer_type: 'str', start_idx: 'int', end_idx: 'int') ->torch.Tensor:
        if adapter_data is None:
            return result
        data: "Optional['BatchLoraWeights']" = adapter_data.data.get(layer_type)
        if has_sgmv() and data is not None and data.can_vectorize(self.process_group):
            if end_idx - start_idx != result.shape[1]:
                proj = torch.zeros_like(result[:, start_idx:end_idx])
            else:
                proj = result
            for r, rank_segments in data.rank_data.items():
                lora_a_ptr = rank_segments.lora_a_ptr
                lora_b_ptr = rank_segments.lora_b_ptr
                if lora_a_ptr is None or lora_b_ptr is None:
                    raise ValueError('LoRA data is missing')
                if data.use_sgmv:
                    v = lora_a_sgmv_cutlass(input, rank_segments.tmp_shrink, lora_a_ptr, rank_segments.segment_starts, rank_segments.segment_ends, self.layer_id, r)
                    if self.process_group.size() > 1:
                        v = self.collect_lora_a(v)
                    lora_b_sgmv_cutlass(proj, v, rank_segments.tmp_expand, lora_b_ptr, rank_segments.segment_starts, rank_segments.segment_ends, self.layer_id)
                else:
                    v = torch.zeros((input.size(0), r), dtype=input.dtype, device=input.device)
                    add_lora_a_bgmv(v, input, lora_a_ptr, rank_segments.indices, self.layer_id)
                    if self.process_group.size() > 1:
                        v = self.collect_lora_a(v)
                    add_lora_b_bgmv(proj, v, lora_b_ptr, rank_segments.indices, self.layer_id)
            if end_idx - start_idx != result.shape[1]:
                result[:, start_idx:end_idx] += proj
        else:
            for adapter_index in adapter_data.meta.adapter_set:
                if data is not None and data.has_adapter(adapter_index):
                    adapter_mask = (adapter_data.meta.adapter_indices == adapter_index).view(-1, 1)
                    layer_result = self.forward_lora(input, data, adapter_index, adapter_mask)
                    result[:, start_idx:end_idx] += layer_result
        return result

    def forward_lora(self, input: 'torch.Tensor', data: "'BatchLoraWeights'", adapter_index: 'int', adapter_mask: 'torch.Tensor') ->torch.Tensor:
        lora_a = data.lora_a[adapter_index][self.layer_id, :, :]
        lora_b = data.lora_b[adapter_index][self.layer_id, :, :]
        lora_a = orient_for_rank(lora_a, lora_b.size(0))
        a_out = input @ lora_a
        if self.process_group.size() > 1:
            a_out = self.collect_lora_a(a_out)
        result = a_out @ lora_b * adapter_mask
        return result

    def collect_lora_a(self, a_out: 'torch.Tensor') ->torch.Tensor:
        raise NotImplementedError('Implemented in subclasses')


class TensorParallelMultiAdapterLinear(LoraLinear):

    def __init__(self, base_layer: 'nn.Module', layer_id: 'int', layer_names: 'List[str]', sizes: 'List[int]', process_group: 'ProcessGroup'):
        super().__init__(base_layer, layer_id, process_group)
        self.layer_names = layer_names
        self.sizes = sizes

    @classmethod
    def load(cls, base_layer: 'nn.Module', layer_id: 'int', layer_names: 'List[str]', sizes: 'List[int]', process_group: 'ProcessGroup'):
        return TensorParallelMultiAdapterLinear(base_layer, layer_id, layer_names, sizes, process_group)

    def forward(self, input: 'torch.Tensor', adapter_data: "'AdapterBatchData'") ->torch.Tensor:
        result = self.base_layer(input)
        if self.layer_names is None:
            return result
        prev_shape = result.shape
        is_3d = len(input.shape) >= 3
        if is_3d:
            input = input.reshape(-1, input.shape[-1])
            result = result.reshape(-1, result.shape[-1])
        offset = 0
        for i, layer_name in enumerate(self.layer_names):
            start_idx = offset // self.process_group.size()
            if self.sizes is not None:
                offset += self.sizes[i]
                end_idx = offset // self.process_group.size()
            else:
                end_idx = result.shape[1]
            result = self.forward_layer_type(result, input, adapter_data, layer_name, start_idx, end_idx)
        if is_3d:
            result = result.reshape(prev_shape)
        return result

    def collect_lora_a(self, a_out: 'torch.Tensor') ->torch.Tensor:
        gathered_tensors = [torch.empty_like(a_out) for _ in range(self.process_group.size())]
        torch.distributed.all_gather(gathered_tensors, a_out)
        return torch.cat(gathered_tensors, dim=1)


class TensorParallelAdapterRowLinear(LoraLinear):

    def __init__(self, base_layer, layer_id, layer_name, process_group):
        super().__init__(base_layer, layer_id, process_group)
        self.layer_name = layer_name

    @classmethod
    def load(cls, base_layer, layer_id, layer_name, process_group):
        return cls(base_layer, layer_id, layer_name, process_group)

    def forward(self, input: 'torch.Tensor', adapter_data: "'AdapterBatchData'") ->torch.Tensor:
        result = self.base_layer(input)
        if self.layer_name is None:
            return result
        stride = result.shape[-1] // self.process_group.size()
        start_idx = self.process_group.rank() * stride
        end_idx = (self.process_group.rank() + 1) * stride
        self.forward_layer_type(result, input, adapter_data, self.layer_name, start_idx, end_idx)
        return result

    def collect_lora_a(self, a_out: 'torch.Tensor') ->torch.Tensor:
        torch.distributed.all_reduce(a_out, group=self.process_group)
        return a_out


MARLIN_TILE_SIZE = 16


def _check_marlin_kernels():
    if not (SYSTEM == 'cuda' and has_sm_8_0):
        raise NotImplementedError('Using quantized Marlin models requires a GPU with CUDA capability 8.0 or later.')
    if marlin_kernels is None:
        raise NotImplementedError('marlin is not installed, install it with: pip install server/marlin')


def _check_valid_shape(in_features: 'int', out_features: 'int'):
    if (in_features % 128 != 0 or out_features % 64 != 0) and (in_features % 64 != 0 or out_features % 128 != 0):
        raise ValueError(f'The GPTQ Marlin kernel does not have a valid thread configuration for weight matrix with shape ({out_features}, {in_features}). The shape elements must be divisible by (128, 64) or (64, 128).')


def pack_fp8_as_int32(fp8_tensor: 'torch.Tensor') ->torch.Tensor:
    """
    Repack FP8 weights to gptq format (packed int32 elements).
    """
    assert fp8_tensor.dtype == torch.float8_e4m3fn
    if fp8_tensor.shape[0] % 4 != 0:
        raise ValueError(f'Leading tensor dimension is not divisable by 4: {fp8_tensor.shape[0]}')
    reshaped = fp8_tensor.reshape(-1, 4, *fp8_tensor.shape[1:])
    byte_tensor = reshaped.view(torch.uint8)
    packed = torch.zeros(fp8_tensor.shape[0] // 4, fp8_tensor.shape[1], dtype=torch.int32, device=fp8_tensor.device)
    for i in range(4):
        packed.bitwise_or_(byte_tensor[:, i] << i * 8)
    return packed


@functools.cache
def get_perms() ->Tuple[List[int], List[int]]:
    scale_perm = []
    for i in range(8):
        scale_perm.extend([(i + 8 * j) for j in range(8)])
    scale_perm_single = []
    for i in range(4):
        scale_perm_single.extend([(2 * i + j) for j in [0, 1, 8, 9, 16, 17, 24, 25]])
    return scale_perm, scale_perm_single


def permute_scales(scales: 'torch.Tensor'):
    scale_perm, scale_perm_single = get_perms()
    out_features = scales.shape[1]
    if scales.shape[0] == 1:
        scales = scales.reshape((-1, len(scale_perm_single)))[:, scale_perm_single]
    else:
        scales = scales.reshape((-1, len(scale_perm)))[:, scale_perm]
    return scales.reshape((-1, out_features)).contiguous()


def repack_fp8_for_marlin(weight: 'torch.Tensor', scales: 'torch.Tensor'):
    """
    Repack FP8 tensor for GPTQ-Marlin.
    """
    out_features, in_features = weight.shape
    qweight = pack_fp8_as_int32(weight.t())
    perm = torch.empty(0, dtype=torch.int, device=qweight.device)
    repacked = marlin_kernels.gptq_marlin_repack(qweight, perm, in_features, out_features, 8)
    scales = permute_scales(scales)
    return repacked, scales


class GPTQMarlinFP8Linear(nn.Module):
    """
    FP8 GPTQ-Marlin linear layer.
    """

    def __init__(self, qweight: 'torch.Tensor', scales: 'torch.Tensor', bias: 'Optional[torch.Tensor]') ->None:
        super().__init__()
        _check_marlin_kernels()
        assert marlin_kernels is not None
        log_once(logger.info, 'GPU does not support FP8, using Marlin FP8 kernel')
        scales = scales.unsqueeze(0)
        if scales.shape[1] == 1:
            out_features, in_features = qweight.shape
            scales = scales.repeat(1, out_features)
        qweight, scales = repack_fp8_for_marlin(qweight, scales)
        in_features = qweight.shape[0] * MARLIN_TILE_SIZE
        out_features = scales.shape[1]
        _check_valid_shape(in_features=in_features, out_features=out_features)
        self.qweight = qweight
        self.scales = scales
        self.bias = bias if bias is not None else None
        self.workspace = torch.zeros(out_features // 64 * 16, dtype=torch.int, device=qweight.device)

    @classmethod
    def from_unquant(cls, weight, bias, dtype):
        qweight, scales = fp8_quantize(weight)
        return cls(qweight=qweight, scales=scales, bias=bias)

    @classmethod
    def from_fp8(cls, weight: 'torch.Tensor', scale: 'torch.Tensor', bias: 'torch.Tensor', dtype: 'torch.dtype', **kwargs):
        return cls(qweight=weight, scales=scale, bias=bias)

    def forward(self, A: 'torch.Tensor') ->torch.Tensor:
        assert marlin_kernels is not None
        A_flat = A.view(-1, A.shape[-1])
        C = marlin_kernels.fp8_marlin_gemm(A_flat, self.qweight, self.scales, self.workspace, 8, A_flat.shape[0], self.scales.shape[1], A_flat.shape[1])
        C = C.reshape(A.shape[:-1] + (self.scales.shape[1],))
        if self.bias is not None:
            C += self.bias
        return C

