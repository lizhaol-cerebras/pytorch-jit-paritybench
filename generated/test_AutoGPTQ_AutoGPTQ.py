
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


from abc import abstractmethod


from typing import Any


from typing import Dict


from typing import List


from typing import Optional


from typing import Union


import torch


from torch import LongTensor


import math


from collections import Counter


import numpy as np


import copy


import logging


import torch.nn as nn


from torch import device


from logging import getLogger


from typing import Tuple


from torch.nn import functional as F


from torch import nn


import itertools


from torch.cuda.amp import custom_bwd


from torch.cuda.amp import custom_fwd


import time


import random


from functools import partial


from typing import Callable


from torch.utils.data import DataLoader


import warnings


import pandas as pd


from itertools import chain


import torch.cuda.amp


from torch.optim import Adam


import torch.cuda


import torch.utils.benchmark as benchmark


class CHECKPOINT_FORMAT:
    GPTQ = 'gptq'
    MARLIN = 'marlin'
    AWQ_GEMM = 'gemm'


CHECKPOINT_FORMAT_FIELD = 'checkpoint_format'


CHECKPOINT_FORMAT_FIELD_COMPAT_MARLIN = 'is_marlin_format'


QUANT_CONFIG_ARG_SYNONYMS = {'w_bit': 'bits', 'q_group_size': 'group_size'}


QUANT_CONFIG_FILENAME = 'quantize_config.json'


class QUANT_METHOD:
    GPTQ = 'gptq'
    AWQ = 'awq'


QUANT_METHOD_FIELD = 'quant_method'


QUANT_METHOD_FORMAT_MAPPING = {QUANT_METHOD.GPTQ: {CHECKPOINT_FORMAT.GPTQ, CHECKPOINT_FORMAT.MARLIN}, QUANT_METHOD.AWQ: {CHECKPOINT_FORMAT.AWQ_GEMM}}


logger = logging.getLogger(__name__)


CPU = device('cpu')


CUDA_0 = device('cuda:0')


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
                q = quantize(x, scale1.unsqueeze(1), zero1.unsqueeze(1), self.maxq)
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
            return quantize(x, self.scale, self.zero, self.maxq)
        return x

    def enabled(self):
        return self.maxq > 0

    def ready(self):
        return torch.all(self.scale != 0)


class GPTQ:

    def __init__(self, layer):
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.pytorch_utils.Conv1D):
            W = W.t()
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0
        self.quantizer = Quantizer()

    def add_batch(self, inp, out):
        if os.environ.get('DEBUG'):
            self.inp1 = inp
            self.out1 = out
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear) or isinstance(self.layer, transformers.Conv1D):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
        if isinstance(self.layer, nn.Conv2d):
            unfold = nn.Unfold(self.layer.kernel_size, dilation=self.layer.dilation, padding=self.layer.padding, stride=self.layer.stride)
            inp = unfold(inp)
            inp = inp.permute([1, 0, 2])
            inp = inp.flatten(1)
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        self.H += inp.matmul(inp.t())

    def fasterquant(self, blocksize=128, percdamp=0.01, group_size=-1, actorder=False, static_groups=False):
        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()
        tick = time.time()
        if not self.quantizer.ready():
            self.quantizer.find_params(W, weight=True)
        H = self.H
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0
        g_idx = []
        scale = []
        zero = []
        now_idx = 1
        if static_groups:
            import copy
            groups = []
            for i in range(0, self.columns, group_size):
                quantizer = copy.deepcopy(self.quantizer)
                quantizer.find_params(W[:, i:i + group_size], weight=True)
                scale.append(quantizer.scale)
                zero.append(quantizer.zero)
                groups.append(quantizer)
        if actorder:
            perm = torch.argsort(torch.diag(H), descending=True)
            W = W[:, perm]
            H = H[perm][:, perm]
            invperm = torch.argsort(perm)
        Losses = torch.zeros_like(W)
        Q = torch.zeros_like(W)
        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H
        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1
            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]
            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]
                if group_size != -1:
                    if not static_groups:
                        if (i1 + i) % group_size == 0:
                            self.quantizer.find_params(W[:, i1 + i:i1 + i + group_size], weight=True)
                        if (i1 + i) // group_size - now_idx == -1:
                            scale.append(self.quantizer.scale)
                            zero.append(self.quantizer.zero)
                            now_idx += 1
                    else:
                        idx = i1 + i
                        if actorder:
                            idx = perm[idx]
                        self.quantizer = groups[idx // group_size]
                q = self.quantizer.quantize(w.unsqueeze(1)).flatten()
                Q1[:, i] = q
                Losses1[:, i] = (w - q) ** 2 / d ** 2
                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1
            Q[:, i1:i2] = Q1
            Losses[:, i1:i2] = Losses1 / 2
            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])
            if os.environ.get('DEBUG'):
                self.layer.weight.data[:, :i2] = Q[:, :i2]
                self.layer.weight.data[:, i2:] = W[:, i2:]
                logger.debug(torch.sum((self.layer(self.inp1) - self.out1) ** 2))
                logger.debug(torch.sum(Losses))
        torch.cuda.synchronize()
        logger.info(f'duration: {time.time() - tick}')
        logger.info(f'avg loss: {torch.sum(Losses).item() / self.nsamples}')
        group_size = group_size if group_size != -1 else self.columns
        if static_groups and actorder:
            g_idx = [(perm[i] // group_size) for i in range(self.columns)]
        else:
            g_idx = [(i // group_size) for i in range(self.columns)]
        g_idx = torch.tensor(g_idx, dtype=torch.int32, device=Q.device)
        if actorder:
            Q = Q[:, invperm]
            g_idx = g_idx[invperm]
        if isinstance(self.layer, transformers.Conv1D):
            Q = Q.t()
        self.layer.weight.data = Q.reshape(self.layer.weight.shape).type_as(self.layer.weight.data)
        if os.environ.get('DEBUG'):
            logger.debug(torch.sum((self.layer(self.inp1) - self.out1) ** 2))
        if scale == []:
            scale.append(self.quantizer.scale)
            zero.append(self.quantizer.zero)
        scale = torch.cat(scale, dim=1)
        zero = torch.cat(zero, dim=1)
        return scale, zero, g_idx

    def free(self):
        if os.environ.get('DEBUG'):
            self.inp1 = None
            self.out1 = None
        self.H = None
        self.Losses = None
        self.Trace = None
        torch.cuda.empty_cache()


class GeneralQuantLinear(nn.Linear):

    def __init__(self, quant_linear_module):
        super().__init__(in_features=quant_linear_module.infeatures, out_features=quant_linear_module.outfeatures, bias=True)
        self.infeatures = quant_linear_module.infeatures
        self.outfeatures = quant_linear_module.outfeatures
        self.bits = quant_linear_module.bits
        self.group_size = quant_linear_module.group_size
        self.maxq = quant_linear_module.maxq
        self.weight.requires_grad = False
        self.weight.data = quant_linear_module.qweight
        self.register_buffer('qweight', quant_linear_module.qweight)
        self.bias.data = quant_linear_module.bias
        self.qweight.requires_grad = False
        self.bias.requires_grad = False
        self.register_buffer('qzeros', quant_linear_module.qzeros)
        self.register_buffer('scales', quant_linear_module.scales)
        self.register_buffer('g_idx', quant_linear_module.g_idx)
        if hasattr(quant_linear_module, 'wf'):
            self.wf = quant_linear_module.wf
        if hasattr(quant_linear_module, 'kernel_switch_threshold'):
            self.kernel_switch_threshold = quant_linear_module.kernel_switch_threshold
        if hasattr(quant_linear_module, 'autogptq_cuda_available'):
            self.autogptq_cuda_available = quant_linear_module.autogptq_cuda_available
        self.trainable = quant_linear_module.trainable
        self.forward = quant_linear_module.forward

    @classmethod
    def inject_to_model(cls, model, target_module_type):
        for name, m in model.named_modules():
            if not isinstance(m, target_module_type):
                continue
            new_m = cls(m)
            if '.' in name:
                parent_name = name.rsplit('.', 1)[0]
                child_name = name[len(parent_name) + 1:]
                parent = model.get_submodule(parent_name)
            else:
                parent_name = ''
                parent = model
                child_name = name
            setattr(parent, child_name, new_m)


QUANTIZE_BLACK_LIST = {QUANT_METHOD.AWQ}


def quant_matmul_248(input, qweight, scales, qzeros, g_idx, bits, maxq):
    with torch.device(input.device):
        output = torch.empty((input.shape[0], qweight.shape[1]), device=input.device, dtype=input.dtype)
        grid = lambda META: (triton.cdiv(input.shape[0], META['BLOCK_SIZE_M']) * triton.cdiv(qweight.shape[1], META['BLOCK_SIZE_N']),)
        quant_matmul_248_kernel[grid](input, qweight, output, scales, qzeros, g_idx, input.shape[0], qweight.shape[1], input.shape[1], bits, maxq, input.stride(0), input.stride(1), qweight.stride(0), qweight.stride(1), output.stride(0), output.stride(1), scales.stride(0), qzeros.stride(0))
        return output


def transpose_quant_matmul_248(input, qweight, scales, qzeros, g_idx, bits, maxq):
    with torch.device(input.device):
        output_dim = qweight.shape[0] * 32 // bits
        output = torch.empty((input.shape[0], output_dim), device=input.device, dtype=input.dtype)
        grid = lambda META: (triton.cdiv(input.shape[0], META['BLOCK_SIZE_M']) * triton.cdiv(output_dim, META['BLOCK_SIZE_K']),)
        transpose_quant_matmul_248_kernel[grid](input, qweight, output, scales, qzeros, g_idx, input.shape[0], qweight.shape[1], output_dim, bits, maxq, input.stride(0), input.stride(1), qweight.stride(0), qweight.stride(1), output.stride(0), output.stride(1), scales.stride(0), qzeros.stride(0))
        return output


class QuantLinearFunction(torch.autograd.Function):

    @staticmethod
    @custom_fwd
    def forward(ctx, input, qweight, scales, qzeros, g_idx, bits, maxq):
        output = quant_matmul_248(input, qweight, scales, qzeros, g_idx, bits, maxq)
        ctx.save_for_backward(qweight, scales, qzeros, g_idx)
        ctx.bits, ctx.maxq = bits, maxq
        return output

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        qweight, scales, qzeros, g_idx = ctx.saved_tensors
        bits, maxq = ctx.bits, ctx.maxq
        grad_input = None
        if ctx.needs_input_grad[0]:
            grad_input = transpose_quant_matmul_248(grad_output, qweight, scales, qzeros, g_idx, bits, maxq)
        return grad_input, None, None, None, None, None, None


class TritonModuleMixin:

    @classmethod
    def warmup(cls, model, transpose=False, seqlen=2048):
        pass


class QuantLinear(nn.Module, TritonModuleMixin):
    """
    Triton v2 quantized linear layer.

    Calls dequant kernel (see triton_utils/dequant) to dequantize the weights then uses
    torch.matmul to compute the output whereas original `triton` quantized linear layer fused
    dequant and matmul into single kernel.add()
    """
    QUANT_TYPE = 'tritonv2'

    def __init__(self, bits, group_size, infeatures, outfeatures, bias, trainable=False, **kwargs):
        super().__init__()
        if bits not in [2, 4, 8]:
            raise NotImplementedError('Only 2,4,8 bits are supported.')
        if infeatures % 32 != 0 or outfeatures % 32 != 0:
            raise NotImplementedError('in_feature and out_feature must be divisible by 32.')
        self.infeatures = infeatures
        self.outfeatures = outfeatures
        self.bits = bits
        self.group_size = group_size if group_size != -1 else infeatures
        self.maxq = 2 ** self.bits - 1
        self.register_buffer('qweight', torch.zeros((infeatures // 32 * self.bits, outfeatures), dtype=torch.int32))
        self.register_buffer('qzeros', torch.zeros((math.ceil(infeatures / self.group_size), outfeatures // 32 * self.bits), dtype=torch.int32))
        self.register_buffer('scales', torch.zeros((math.ceil(infeatures / self.group_size), outfeatures), dtype=torch.float16))
        self.register_buffer('g_idx', torch.tensor([(i // self.group_size) for i in range(infeatures)], dtype=torch.int32))
        if bias:
            self.register_buffer('bias', torch.zeros(outfeatures, dtype=torch.float16))
        else:
            self.bias = None
        self.trainable = trainable

    def post_init(self):
        pass

    def pack(self, linear, scales, zeros, g_idx=None):
        W = linear.weight.data.clone()
        if isinstance(linear, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(linear, transformers.pytorch_utils.Conv1D):
            W = W.t()
        self.g_idx = g_idx.clone() if g_idx is not None else self.g_idx
        scales = scales.t().contiguous()
        zeros = zeros.t().contiguous()
        scale_zeros = zeros * scales
        self.scales = scales.clone().half()
        if linear.bias is not None:
            self.bias = linear.bias.clone().half()
        intweight = []
        for idx in range(self.infeatures):
            intweight.append(torch.round((W[:, idx] + scale_zeros[self.g_idx[idx]]) / self.scales[self.g_idx[idx]])[:, None])
        intweight = torch.cat(intweight, dim=1)
        intweight = intweight.t().contiguous()
        intweight = intweight.numpy().astype(np.uint32)
        i = 0
        row = 0
        qweight = np.zeros((intweight.shape[0] // 32 * self.bits, intweight.shape[1]), dtype=np.uint32)
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
        quant_linear_fn = QuantLinearFunction
        out = quant_linear_fn.apply(x.reshape(-1, x.shape[-1]), self.qweight, self.scales, self.qzeros, self.g_idx, self.bits, self.maxq)
        out = out.half().reshape(out_shape)
        out = out + self.bias if self.bias is not None else out
        return out

    @classmethod
    def warmup(cls, model, transpose=False, seqlen=2048):
        """
        Pre-tunes the quantized kernel
        """
        kn_values = {}
        for _, m in model.named_modules():
            if not isinstance(m, cls):
                continue
            k = m.infeatures
            n = m.outfeatures
            if (k, n) not in kn_values:
                kn_values[k, n] = m.qweight, m.scales, m.qzeros, m.g_idx, m.bits, m.maxq
        logger.info(f'Found {len(kn_values)} unique KN Linear values.')
        logger.info('Warming up autotune cache ...')
        with torch.no_grad():
            for m in tqdm(range(0, math.ceil(math.log2(seqlen)) + 1)):
                m = 2 ** m
                for (k, n), (qweight, scales, qzeros, g_idx, bits, maxq) in kn_values.items():
                    a = torch.randn(m, k, dtype=torch.float16, device=model.device)
                    quant_matmul_248(a, qweight, scales, qzeros, g_idx, bits, maxq)
        del kn_values


SUPPORTED_MODELS = ['bloom', 'gptj', 'gpt2', 'gpt_neox', 'opt', 'moss', 'gpt_bigcode', 'codegen', 'RefinedWebModel', 'RefinedWeb', 'baichuan', 'internlm', 'qwen', 'xverse', 'deci', 'stablelm_epoch', 'mpt', 'cohere', 'minicpm3']


def _validate_marlin_compatibility(cfg: 'BaseQuantizeConfig'):
    if not MARLIN_AVAILABLE:
        return f'AutoGPTQ is not compiled with the Marlin kernel, with the following error: {MARLIN_EXCEPTION}'
    if cfg.bits != 4:
        return f'The quantized model uses a bitwidth different than 4 (found {cfg.bits})'
    if cfg.group_size != 128 and cfg.group_size != -1:
        return 'The quantized model uses a group size that is not 128 or -1 (found quantization_config.group_size)'
    if not cfg.sym:
        return 'The quantized model uses asymmetric quantization'
    if cfg.desc_act:
        return 'The quantized model uses act-order (also called desc-act) scheme'
    if cfg.quant_method == QUANT_METHOD.AWQ:
        return 'awq_gemm format is currently not compatible with marlin'
    return None


def _validate_marlin_device_support() ->bool:
    """
        Validates if the current device is compatible for Marlin.
        ref: https://github.com/IST-DASLab/marlin?tab=readme-ov-file#requirements

        Returns:
            bool: indicates if CUDA device is compatible for Marlin
        """
    return torch.cuda.get_device_capability()[0] >= 8


EXLLAMA_DEFAULT_MAX_INPUT_LENGTH = 2048


def _torch_device(idx):
    if idx == -1:
        return 'cpu'
    return f'cuda:{idx}'


class ExLlamaV2DeviceTensors:
    device_idx: 'int'
    scratch_bytes: 'int'
    scratch_idx: 'int'
    scratch: 'torch.tensor' = None

    def __init__(self, device_idx, scratch_bytes):
        self.device_idx = device_idx
        self.scratch_bytes = scratch_bytes

    def prepare(self):
        self.scratch = torch.empty((self.scratch_bytes // 2,), dtype=torch.half, device=_torch_device(self.device_idx))

    def get_scratch_slice(self, size_bytes):
        if self.scratch is None:
            self.prepare()
        size_bytes = (size_bytes + 127) // 128 * 128
        size_half = size_bytes // 2
        scratch_slice = self.scratch.narrow(0, 0, size_half)
        return scratch_slice


def collate_data(blocks: 'List[Dict[str, List[List[int]]]]', pad_token_id: 'int') ->Dict[str, LongTensor]:

    def pad_block(block, pads):
        return torch.cat((pads, block), dim=-1)
    input_ids_blocks = [LongTensor(block['input_ids']) for block in blocks]
    attention_mask_blocks = [LongTensor(block['attention_mask']) for block in blocks]
    label_blocks = [LongTensor(block['labels']) for block in blocks]
    bsz = len(blocks)
    inp_max_len = max([block.size(-1) for block in input_ids_blocks])
    label_max_len = max([block.size(-1) for block in label_blocks])
    for i in range(bsz):
        block_bsz, block_inp_len = input_ids_blocks[i].shape
        block_label_len = label_blocks[i].shape[-1]
        pad_num = inp_max_len - block_inp_len
        if pad_num > 0:
            input_ids_blocks[i] = pad_block(input_ids_blocks[i], torch.ones((block_bsz, pad_num)) * pad_token_id)
            attention_mask_blocks[i] = pad_block(attention_mask_blocks[i], torch.zeros((block_bsz, pad_num)))
        label_pad_num = label_max_len - block_label_len
        if label_pad_num > 0:
            label_blocks[i] = pad_block(label_blocks[i], torch.ones((block_bsz, label_pad_num)) * -100)
    return {'input_ids': torch.cat(input_ids_blocks, dim=0).long(), 'attention_mask': torch.cat(attention_mask_blocks, dim=0).long(), 'labels': torch.cat(label_blocks, dim=0).long()}


def find_layers(module, layers=None, name=''):
    if not layers:
        layers = [transformers.pytorch_utils.Conv1D, nn.Conv2d, nn.Linear]
    for layer in layers:
        if isinstance(module, layer):
            return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(child, layers=layers, name=name + '.' + name1 if name != '' else name1))
    return res


def get_checkpoints(model_name_or_path: 'str', extensions: 'List[str]', possible_model_basenames: 'List[str]', **cached_file_kwargs):
    """
    Retrives (and if necessary downloads from Hugging Face Hub) the model checkpoint. Sharding is supported. All the `possible_model_basenames` (e.g. `["model", "model-4bit-gptq"]`) will be explored over all `extensions` (e.g. `[".bin", ".safetensors"]`).
    """
    searched_files = []
    resolved_archive_file = None
    true_model_basename = None
    if os.path.isdir(model_name_or_path):
        for ext in extensions:
            for possible_model_basename in possible_model_basenames:
                shard_index_name = possible_model_basename + ext + '.index.json'
                searched_files.append(shard_index_name)
                possible_index_file = os.path.join(model_name_or_path, shard_index_name)
                if os.path.isfile(possible_index_file):
                    possible_model_basename = possible_index_file.replace(ext + '.index.json', '')
                    return True, possible_index_file, possible_model_basename
                else:
                    model_save_name = os.path.join(model_name_or_path, possible_model_basename)
                    searched_files.append(possible_model_basename + ext)
                    if os.path.isfile(model_save_name + ext):
                        resolved_archive_file = model_save_name + ext
                        return False, resolved_archive_file, possible_model_basename
    else:
        temp = None
        for ext in extensions:
            for possible_model_basename in possible_model_basenames:
                shard_index_name = possible_model_basename + ext + '.index.json'
                shard_index = cached_file(model_name_or_path, shard_index_name, **cached_file_kwargs)
                searched_files.append(shard_index_name)
                if shard_index is not None:
                    with open(str(shard_index)) as f:
                        index_json = json.load(f)
                        shards = list(set(index_json['weight_map'].values()))
                        for shard in shards:
                            resolved_archive_file = cached_file(model_name_or_path, shard, **cached_file_kwargs)
                        return True, shard_index, possible_model_basename
                else:
                    resolved_archive_file = cached_file(model_name_or_path, possible_model_basename + ext, **cached_file_kwargs)
                    if resolved_archive_file is None:
                        resolved_archive_file = temp
                    searched_files.append(possible_model_basename + ext)
                    if resolved_archive_file is not None:
                        temp = resolved_archive_file
                        return False, resolved_archive_file, possible_model_basename
    if resolved_archive_file is None:
        raise FileNotFoundError(f"Could not find a model in {model_name_or_path} with a name in {', '.join(searched_files)}. Please specify the argument model_basename to use a custom file name.")
    return False, resolved_archive_file, true_model_basename


def get_device(obj: 'Union[torch.Tensor, nn.Module]'):
    if isinstance(obj, torch.Tensor):
        return obj.device
    return next(obj.parameters()).device


def get_module_by_name_prefix(model, module_name: 'str'):
    for name, module in model.named_modules():
        if name.startswith(module_name):
            return module


def get_module_by_name_suffix(model, module_name: 'str'):
    for name, module in model.named_modules():
        if name.endswith(module_name):
            return module


def recurse_setattr(module, name, value):
    """A function to recursively set attributes to a module."""
    if '.' not in name:
        setattr(module, name, value)
    else:
        name, rest = name.split('.', 1)
        recurse_setattr(getattr(module, name), rest, value)


def make_quant(module, names, bits, group_size, name='', use_triton: 'bool'=False, use_marlin: 'bool'=False, disable_exllama: 'Optional[bool]'=None, disable_exllamav2: 'bool'=False, use_qigen: 'bool'=False, use_cuda_fp16: 'bool'=True, desc_act: 'bool'=False, trainable: 'bool'=False, use_tritonv2: 'bool'=False):
    if disable_exllama is None:
        if disable_exllamav2:
            disable_exllama = False
        else:
            disable_exllama = True
    QuantLinear = dynamically_import_QuantLinear(use_triton=use_triton, desc_act=desc_act, group_size=group_size, bits=bits, use_marlin=use_marlin, disable_exllama=disable_exllama, disable_exllamav2=disable_exllamav2, use_qigen=use_qigen, use_tritonv2=use_tritonv2)
    if isinstance(module, QuantLinear):
        return
    for name, submodule in module.named_modules():
        if name in names:
            ori_layer_device = next(submodule.parameters()).device
            if isinstance(submodule, nn.Linear):
                in_features = submodule.in_features
                out_features = submodule.out_features
            elif isinstance(submodule, nn.Conv2d):
                in_features = submodule.in_channels
                out_features = submodule.out_channels
            elif isinstance(submodule, transformers.pytorch_utils.Conv1D):
                in_features = submodule.weight.shape[0]
                out_features = submodule.weight.shape[1]
            bias = submodule.bias is not None
            if (not desc_act or group_size == -1) and not use_triton and not use_qigen and not use_tritonv2:
                new_layer = QuantLinear(bits, group_size, in_features, out_features, bias, use_cuda_fp16=use_cuda_fp16, trainable=trainable, weight_dtype=submodule.weight.dtype)
            else:
                new_layer = QuantLinear(bits, group_size, in_features, out_features, bias, trainable=trainable, weight_dtype=submodule.weight.dtype)
            new_layer.device = ori_layer_device
            recurse_setattr(module, name, new_layer)


def make_sure_no_tensor_in_meta_device(model, use_triton: 'bool', desc_act: 'bool', group_size: 'int', bits: 'int', disable_exllama: 'bool', disable_exllamav2: 'bool', use_marlin: 'bool'=False, use_tritonv2: 'bool'=False):
    QuantLinear = dynamically_import_QuantLinear(use_triton, desc_act, group_size, bits=bits, disable_exllama=disable_exllama, disable_exllamav2=disable_exllamav2, use_marlin=use_marlin, use_tritonv2=use_tritonv2)
    for n, m in model.named_modules():
        if isinstance(m, QuantLinear) and m.bias is not None and m.bias.device == torch.device('meta'):
            m.register_buffer('bias', torch.zeros(m.outfeatures, dtype=torch.float16, device='cpu'))


def move_to_device(obj: 'Optional[Union[torch.Tensor, nn.Module]]', device: 'torch.device'):
    if obj is None:
        return obj
    else:
        if get_device(obj) != device:
            obj = obj
        return obj


def nested_move_to_device(v, device):
    if isinstance(v, torch.Tensor):
        return move_to_device(v, device)
    elif isinstance(v, (list, tuple)):
        return type(v)([nested_move_to_device(e, device) for e in v])
    else:
        return v


def pack_from_tensors(unpacked_qweight: 'torch.Tensor', unpacked_qzeros: 'torch.Tensor', awq_scales: 'torch.Tensor', bits: 'int', group_size: 'int'):
    """
    Args:
        unpacked_qweight (`torch.LongTensor`):
            Expected shape: (in_features, out_features)
        unpacked_qzeros (`torch.LongTensor`):
            Expected shape: (in_features // group_size, out_features)
        awq_scales (`torch.LongTensor`):
            Expected shape: (in_features // group_size, out_features)

    Returns:
        qweight (`torch.LongTensor`):
            With shape (in_features // (32 // bits), out_features)
        qzeros (`torch.LongTensor`):
            With shape (in_features // group_size, out_features // (32 // bits))
    """
    assert bits == 4
    W = unpacked_qweight.clone().cpu()
    awq_scales = awq_scales.t().contiguous()
    unpacked_qzeros = unpacked_qzeros.contiguous()
    unpacked_qzeros = unpacked_qzeros.cpu()
    awq_scales = awq_scales.cpu()
    scale_zeros = unpacked_qzeros.t() * awq_scales
    scales = awq_scales.clone()
    infeatures = unpacked_qweight.shape[1]
    intweight = []
    for idx in range(infeatures):
        g_idx = idx // group_size
        intweight.append(torch.round((W[:, idx] + scale_zeros[:, g_idx]) / scales[:, g_idx])[:, None])
    intweight = torch.cat(intweight, dim=1)
    intweight = intweight.t().contiguous()
    intweight = intweight.numpy().astype(np.uint32)
    i = 0
    row = 0
    qweight = np.zeros((intweight.shape[0] // 32 * bits, intweight.shape[1]), dtype=np.uint32)
    while row < qweight.shape[0]:
        for j in range(i, i + 32 // bits):
            qweight[row] |= intweight[j] << bits * (j - i)
        i += 32 // bits
        row += 1
    qweight = qweight.astype(np.int32)
    qweight = torch.from_numpy(qweight)
    unpacked_qzeros = unpacked_qzeros - 1
    torch.bitwise_and(unpacked_qzeros, 2 ** bits - 1, out=unpacked_qzeros)
    unpacked_qzeros = unpacked_qzeros.numpy().astype(np.uint32)
    qzeros = np.zeros((unpacked_qzeros.shape[0], unpacked_qzeros.shape[1] // 32 * bits), dtype=np.uint32)
    i = 0
    col = 0
    while col < qzeros.shape[1]:
        for j in range(i, i + 32 // bits):
            qzeros[:, col] |= unpacked_qzeros[:, j] << bits * (j - i)
        i += 32 // bits
        col += 1
    qzeros = qzeros.astype(np.int32)
    qzeros = torch.from_numpy(qzeros)
    return qweight, qzeros


def pack_model(model, quantizers, bits, group_size, use_triton=False, use_cuda_fp16=True, desc_act=False, warmup_triton: 'bool'=False, force_layer_back_to_cpu: 'bool'=False, use_marlin: 'bool'=False, use_tritonv2: 'bool'=False):
    QuantLinear = dynamically_import_QuantLinear(use_triton=use_triton, desc_act=desc_act, group_size=group_size, bits=bits, disable_exllama=False, disable_exllamav2=True, use_marlin=use_marlin, use_tritonv2=use_tritonv2)
    if force_layer_back_to_cpu:
        model
    logger.info('Packing model...')
    layers = find_layers(model)
    layers = {n: layers[n] for n in quantizers}
    make_quant(model, quantizers, bits, group_size, use_triton=use_triton, use_cuda_fp16=use_cuda_fp16, desc_act=desc_act, disable_exllama=False, disable_exllamav2=True, use_marlin=use_marlin)
    qlayers = find_layers(model, [QuantLinear])
    with tctl.threadpool_limits(limits=1):
        pbar = tqdm(qlayers.keys(), leave=True)
        for name in pbar:
            pbar.set_description(f'Packing {name}...', refresh=True)
            quantizers[name], scale, zero, g_idx = quantizers[name]
            layer_device = qlayers[name].device
            qlayers[name]
            layers[name], scale, zero, g_idx = layers[name], scale, zero, g_idx
            if QuantLinear.QUANT_TYPE == 'marlin':
                qlayers[name].pack(layers[name], scale)
            else:
                qlayers[name].pack(layers[name], scale, zero, g_idx)
            qlayers[name]
    logger.info('Model packed.')
    if use_triton and warmup_triton:
        logger.warning('using autotune_warmup will move model to GPU, make sure you have enough VRAM to load the whole model.')
        QuantLinear.warmup(model, seqlen=model.seqlen)


def _get_perms():
    perm = []
    for i in range(32):
        perm1 = []
        col = i // 4
        for block in [0, 1]:
            for row in [2 * (i % 4), 2 * (i % 4) + 1, 2 * (i % 4 + 4), 2 * (i % 4 + 4) + 1]:
                perm1.append(16 * row + col + 8 * block)
        for j in range(4):
            perm.extend([(p + 256 * j) for p in perm1])
    perm = np.array(perm)
    interleave = np.array([0, 2, 4, 6, 1, 3, 5, 7])
    perm = perm.reshape((-1, 8))[:, interleave].ravel()
    perm = torch.from_numpy(perm)
    scale_perm = []
    for i in range(8):
        scale_perm.extend([(i + 8 * j) for j in range(8)])
    scale_perm_single = []
    for i in range(4):
        scale_perm_single.extend([(2 * i + j) for j in [0, 1, 8, 9, 16, 17, 24, 25]])
    return perm, scale_perm, scale_perm_single


def unpack_qzeros(qzeros):
    unpacked_zeros = torch.zeros((qzeros.shape[0], qzeros.shape[1] * 8), dtype=torch.int8, device=qzeros.device, requires_grad=False)
    for col in range(unpacked_zeros.shape[1]):
        i = col % 8
        unpacked_zeros[:, col] = qzeros[:, col // 8] >> 4 * i & 15
    return unpacked_zeros + 1


@torch.no_grad()
def convert_to_marlin(model, model_quantlinear, quantization_config: 'BaseQuantizeConfig', repack: 'bool', strict: 'bool'=False):
    """
    Converts GPTQ-packed weights to the Marlin format. This assumes that the model already meets Marlin kernel constraints.

    Arguments:
        repack (`bool`):
            Whether to repack the qweights from `model` into the Marlin's QuantLinear layers.
    """
    if repack:
        message = 'Repacking weights to be compatible with Marlin kernel...'
    else:
        message = "Overriding QuantLinear layers to use Marlin's QuantLinear..."
    for name, module in tqdm(model.named_modules(), desc=message, total=len(list(model.named_modules()))):
        if not isinstance(module, model_quantlinear):
            continue
        parent_name = '.'.join(name.split('.')[:-1])
        layer_name = name[len(parent_name) + 1:]
        with torch.device('meta'):
            new_module = MarlinQuantLinear(bits=4, group_size=module.group_size, infeatures=module.infeatures, outfeatures=module.outfeatures, bias=module.bias is not None, trainable=False)
        new_module.workspace = torch.zeros(module.outfeatures // 128 * 16, dtype=torch.int, device=module.device)
        if repack:
            marlin_repacked_weight = autogptq_marlin_cuda.gptq_repack(module.qweight)
            if strict:
                dequantized_qzeros = unpack_qzeros(module.qzeros)
                if not torch.all(dequantized_qzeros == 8):
                    raise ValueError('Marlin kernel is compatible only with checkpoints using symmetric quantization.Found non-symmetric quantization for the weight {name}.')
            _, _scale_perm, _scale_perm_single = _get_perms()
            s = module.scales.data.clone()
            if module.group_size != module.infeatures:
                s = s.reshape((1, -1))
                s = s.reshape((-1, len(_scale_perm)))[:, _scale_perm]
            else:
                s = s.reshape((-1, len(_scale_perm_single)))[:, _scale_perm_single]
            s = s.reshape((-1, module.outfeatures)).contiguous()
            new_module.B = marlin_repacked_weight
            new_module.s = s
            new_module.bias = module.bias
            new_module = new_module
        parent_module = model.get_submodule(parent_name)
        setattr(parent_module, layer_name, new_module)
        del module
        if repack:
            del marlin_repacked_weight
        gc.collect()
    quantization_config.checkpoint_format = CHECKPOINT_FORMAT.MARLIN
    return model


def recurse_getattr(obj, attr: 'str'):
    """
    Recursive `getattr`.

    Args:
        obj:
            A class instance holding the attribute.
        attr (`str`):
            The attribute that is to be retrieved, e.g. 'attribute1.attribute2'.
    """

    def _getattr(obj, attr):
        return getattr(obj, attr)
    return functools.reduce(_getattr, [obj] + attr.split('.'))


def prepare_model_for_marlin_load(model, quantize_config: 'BaseQuantizeConfig', quant_linear_class, torch_dtype, current_model_save_name, device_map):
    if quantize_config.checkpoint_format == CHECKPOINT_FORMAT.MARLIN:
        model_save_name = current_model_save_name
        logger.info(f'Loading a GPTQ model, detected Marlin serialized format at {model_save_name}.')
        model = convert_to_marlin(model, quant_linear_class, quantize_config, repack=False)
    else:
        model_save_name, is_cached = quantize_config.get_cache_file_path(quant_method=QUANT_METHOD.GPTQ, checkpoint_format=CHECKPOINT_FORMAT.MARLIN)
        if is_cached:
            logger.info(f'Loading a GPTQ model, detected a cached repacked weight for Marlin kernel at {model_save_name}.')
            model = convert_to_marlin(model, quant_linear_class, quantize_config, repack=False)
        else:
            load_checkpoint_in_model(model, dtype=torch_dtype, checkpoint=current_model_save_name, device_map=device_map, offload_state_dict=True, offload_buffers=True)
            model = convert_to_marlin(model, quant_linear_class, quantize_config, repack=True)
            tied_params = find_tied_parameters(model)
            for weight_group in tied_params:
                for param_name in weight_group:
                    if isinstance(recurse_getattr(model, param_name), torch.nn.Parameter):
                        recurse_setattr(model, param_name, torch.nn.Parameter(recurse_getattr(model, param_name).clone()))
                    else:
                        recurse_setattr(model, param_name, recurse_getattr(model, param_name).clone())
            safe_save(model.state_dict(), model_save_name)
    return model, model_save_name


def simple_dispatch_model(model, device_map):
    if '' in device_map:
        d = device_map['']
        model = model
        model.hf_device_map = device_map
        return model
    tied_params = accelerate.utils.modeling.find_tied_parameters(model)
    if set(device_map.values()) == {'cpu'} or set(device_map.values()) == {'cpu', 'disk'}:
        main_device = 'cpu'
    else:
        main_device = [d for d in device_map.values() if d not in ['cpu', 'disk']][0]
    cpu_offload_group = [(n, d) for n, d in device_map.items() if d == 'cpu']
    prev_hook = None
    for idx, (n, d) in enumerate(cpu_offload_group):
        m = get_module_by_name_suffix(model, n)
        _, prev_hook = accelerate.cpu_offload_with_hook(m, execution_device=main_device, prev_module_hook=prev_hook)
    if len(cpu_offload_group) > 1:
        get_module_by_name_suffix(model, cpu_offload_group[0][0])._hf_hook.prev_module_hook = prev_hook
    for n, d in device_map.items():
        m = get_module_by_name_suffix(model, n)
        if d != 'cpu':
            d = torch.device(d)
            hook = AlignDevicesHook(d, io_same_device=True, place_submodules=True)
            add_hook_to_module(m, hook)
    accelerate.utils.modeling.retie_parameters(model, tied_params)
    model.hf_device_map = device_map
    return model


def awq_reverse_reorder_int_tensor(int_tensor, bits: 'int'):
    assert bits == 4
    int_tensor = int_tensor.T.contiguous()
    compress_ratio = 32 // bits
    assert int_tensor.shape[-1] % compress_ratio == 0
    order_map = [0, 2, 4, 6, 1, 3, 5, 7]
    order_tensor = torch.tensor(order_map, dtype=torch.int32, device=int_tensor.device).reshape(1, -1)
    order_tensor = order_tensor.repeat(int_tensor.shape[1] // compress_ratio, 1)
    order_tensor = order_tensor + torch.arange(0, int_tensor.shape[1], compress_ratio, dtype=torch.int32, device=int_tensor.device).reshape(-1, 1)
    order_tensor = order_tensor.reshape(-1)
    reverse_order_tensor = torch.arange(order_tensor.shape[0])[order_tensor]
    reverse_order_tensor = reverse_order_tensor[order_tensor]
    int_tensor = int_tensor[:, reverse_order_tensor]
    return int_tensor


def unpack_awq(awq_qweight: 'torch.Tensor', awq_qzeros: 'torch.Tensor', awq_scales: 'torch.Tensor', bits: 'int', group_size: 'int'):
    """
    Args:
        awq_qweight (`torch.LongTensor`):
            Expected shape: (in_features, out_features // (32 // bits))
        awq_qzeros (`torch.LongTensor`):
            Expected shape: (in_features // group_size, out_features // (32 // bits))
        awq_scales (`torch.LongTensor`):
            Expected shape: (in_features // group_size, out_features)

    Returns:
        fp16_weight (`torch.LongTensor`):
            With shape (in_features, out_features).
        zeros (`torch.LongTensor`):
            With shape (in_features // group_size, out_features).
    """
    assert bits == 4
    qzeros = awq_qzeros
    qweight = awq_qweight
    qweight = qweight.T.contiguous()
    infeatures = awq_qweight.shape[0]
    wf = torch.tensor(list(range(0, 32, bits)), dtype=torch.int32, device=qzeros.device).unsqueeze(0)
    zeros = torch.bitwise_right_shift(torch.unsqueeze(qzeros, 2), wf.unsqueeze(0))
    torch.bitwise_and(zeros, 2 ** bits - 1, out=zeros)
    zeros = zeros.reshape(-1, 1, zeros.shape[1] * zeros.shape[2])
    weight = torch.bitwise_right_shift(torch.unsqueeze(qweight, 1), wf.unsqueeze(-1))
    torch.bitwise_and(weight, 2 ** bits - 1, out=weight)
    weight = weight.reshape(-1, group_size, weight.shape[2])
    weight = weight.view(-1, weight.shape[-1])
    zeros = zeros.view(-1, zeros.shape[-1])
    zeros = zeros.T.contiguous()
    zeros = awq_reverse_reorder_int_tensor(zeros, bits)
    weight = awq_reverse_reorder_int_tensor(weight, bits)
    scales = awq_scales
    zeros = zeros.contiguous()
    scale_zeros = zeros * scales
    g_idx = torch.tensor([(i // group_size) for i in range(infeatures)], dtype=torch.int32)
    scale_mat = scales[g_idx]
    scale_zeros_mat = scale_zeros[g_idx].half()
    qdq_weight_T = weight * scale_mat - scale_zeros_mat.half()
    fp16_weight = qdq_weight_T.T
    return fp16_weight, zeros


class FusedBaseModule(nn.Module, TritonModuleMixin):

    @classmethod
    @abstractmethod
    def inject_to_model(cls, *args, **kwargs):
        raise NotImplementedError()


class FusedBaseAttentionModule(FusedBaseModule):

    @classmethod
    @abstractmethod
    def inject_to_model(cls, model, use_triton=False, group_size=-1, use_cuda_fp16=True, desc_act=False, trainable=False, **kwargs):
        raise NotImplementedError()

    @classmethod
    def warmup(cls, model, transpose=False, seqlen=2048):
        pass


class FusedBaseMLPModule(FusedBaseModule):

    @classmethod
    @abstractmethod
    def inject_to_model(cls, model, use_triton=False, **kwargs):
        raise NotImplementedError()


def duplicate_interleave(m):
    """
    A simple version of `torch.repeat_interleave` for duplicating a matrix while interleaving the copy.
    """
    dim0 = m.shape[0]
    m = m.view(-1, 1)
    m = m.repeat(1, 2)
    m = m.view(dim0, -1)
    return m


def rotate_every_two(x):
    x1 = x[:, :, :, ::2]
    x2 = x[:, :, :, 1::2]
    x = torch.stack((-x2, x1), dim=-1)
    return x.flatten(-2)


def apply_rotary_pos_emb(x, sincos, offset=0):
    sin, cos = (duplicate_interleave(t)[None, offset:x.shape[1] + offset, None, :] for t in sincos)
    return x * cos + rotate_every_two(x) * sin


def compare_pytorch_version(version: 'str'='v2.0.0', op: 'str'='eq'):
    assert op in ['eq', 'lt', 'le', 'gt', 'ge']
    from torch import __version__
    return getattr(parse_version(__version__), f'__{op}__')(parse_version(version))


def fixed_pos_embedding(x, seq_dim=1, seq_len=None):
    dim = x.shape[-1]
    if seq_len is None:
        seq_len = x.shape[seq_dim]
    inv_freq = 1.0 / 10000 ** (torch.arange(0, dim, 2) / dim)
    sinusoid_inp = torch.einsum('i , j -> i j', torch.arange(seq_len, dtype=torch.float), inv_freq).float()
    return torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)


class FusedGPTJAttentionForQuantizedModel(FusedBaseAttentionModule):

    def __init__(self, config):
        super().__init__()
        max_positions = config.max_position_embeddings
        self.register_buffer('bias', torch.tril(torch.ones((max_positions, max_positions), dtype=torch.bool)).view(1, 1, max_positions, max_positions))
        self.register_buffer('masked_bias', torch.tensor(-1000000000.0))
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.attn_dropout_p = config.attn_pdrop
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        self.embed_dim = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_attention_heads
        if self.head_dim * self.num_attention_heads != self.embed_dim:
            raise ValueError(f'embed_dim must be divisible by num_attention_heads (got `embed_dim`: {self.embed_dim} and `num_attention_heads`: {self.num_attention_heads}).')
        self.scale_attn = torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        self.qkv_proj = nn.Linear(self.embed_dim, self.embed_dim * 3, bias=False)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.rotary_dim = config.rotary_dim

    def _split_heads(self, qkv):
        """
        Splits hidden dim into attn_head_size and num_attention_heads
        """
        new_shape = qkv.size()[:-1] + (3, self.num_attention_heads, self.head_dim)
        qkv = qkv.view(new_shape)
        query = qkv[:, :, 0]
        key = qkv[:, :, 1]
        value = qkv[:, :, 2]
        return query, key, value

    def _merge_heads(self, tensor, num_attention_heads, attn_head_size):
        """
        Merges attn_head_size dim and num_attn_heads dim into hidden dim
        """
        if len(tensor.shape) == 5:
            tensor = tensor.permute(0, 1, 3, 2, 4).contiguous()
        elif len(tensor.shape) == 4:
            tensor = tensor.permute(0, 2, 1, 3).contiguous()
        else:
            raise ValueError(f'Input tensor rank should be one of [4, 5], but is: {len(tensor.shape)}')
        new_shape = tensor.size()[:-2] + (num_attention_heads * attn_head_size,)
        return tensor.view(new_shape)

    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
        query_length, key_length = query.size(-2), key.size(-2)
        causal_mask = self.bias[:, :, key_length - query_length:key_length, :key_length]
        query = query
        key = key
        attn_weights = torch.matmul(query, key.transpose(-1, -2))
        mask_value = torch.finfo(attn_weights.dtype).min
        mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype)
        attn_weights = torch.where(causal_mask, attn_weights, mask_value)
        attn_weights = attn_weights / self.scale_attn
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attn_weights = attn_weights
        attn_weights = self.attn_dropout(attn_weights)
        if head_mask is not None:
            attn_weights = attn_weights * head_mask
        attn_output = torch.matmul(attn_weights, value)
        return attn_output, attn_weights

    def forward(self, hidden_states: 'torch.FloatTensor', layer_past: 'Optional[Tuple[torch.Tensor]]'=None, attention_mask: 'Optional[torch.FloatTensor]'=None, position_ids: 'Optional[torch.LongTensor]'=None, head_mask: 'Optional[torch.FloatTensor]'=None, use_cache: 'Optional[bool]'=False, output_attentions: 'Optional[bool]'=False) ->Union[Tuple[torch.Tensor, Tuple[torch.Tensor]], Optional[Tuple[torch.Tensor, Tuple[torch.Tensor], Tuple[torch.Tensor, ...]]]]:
        query, key, value = self._split_heads(self.qkv_proj(hidden_states))
        seq_len = key.shape[1]
        offset = 0
        if layer_past is not None:
            offset = layer_past[0].shape[-2]
            seq_len += offset
        if self.rotary_dim is not None:
            k_rot = key[:, :, :, :self.rotary_dim]
            k_pass = key[:, :, :, self.rotary_dim:]
            q_rot = query[:, :, :, :self.rotary_dim]
            q_pass = query[:, :, :, self.rotary_dim:]
            sincos = fixed_pos_embedding(k_rot, 1, seq_len=seq_len)
            k_rot = apply_rotary_pos_emb(k_rot, sincos, offset=offset)
            q_rot = apply_rotary_pos_emb(q_rot, sincos, offset=offset)
            key = torch.cat([k_rot, k_pass], dim=-1)
            query = torch.cat([q_rot, q_pass], dim=-1)
        else:
            sincos = fixed_pos_embedding(key, 1, seq_len=seq_len)
            key = apply_rotary_pos_emb(key, sincos, offset=offset)
            query = apply_rotary_pos_emb(query, sincos, offset=offset)
        key = key.permute(0, 2, 1, 3)
        query = query.permute(0, 2, 1, 3)
        value = value.permute(0, 2, 1, 3)
        is_causal = layer_past is None
        if layer_past is not None:
            past_key = layer_past[0]
            past_value = layer_past[1]
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)
        if use_cache is True:
            query = query.contiguous()
            key = key.contiguous()
            value = value.contiguous()
            present = key, value
        else:
            present = None
        if compare_pytorch_version('v2.0.0', op='ge'):
            attn_output = F.scaled_dot_product_attention(query, key, value, attn_mask=None if is_causal else attention_mask, dropout_p=self.attn_dropout_p, is_causal=is_causal)
            attn_weights = None
        else:
            attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)
        attn_output = self._merge_heads(attn_output, self.num_attention_heads, self.head_dim)
        attn_output = self.out_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)
        outputs = attn_output, present
        if output_attentions:
            outputs += attn_weights,
        return outputs

    @classmethod
    def inject_to_model(cls, model, use_triton=False, group_size=-1, use_cuda_fp16=True, desc_act=False, trainable=False, bits: 'int'=4, disable_exllama=True, disable_exllamav2=False, **kwargs):
        config = model.config
        QuantLinear = dynamically_import_QuantLinear(use_triton=use_triton, desc_act=desc_act, group_size=group_size, bits=bits, disable_exllama=disable_exllama, disable_exllamav2=disable_exllamav2)
        for name, m in model.named_modules():
            if not isinstance(m, GPTJAttention):
                continue
            attn = cls(config)
            q_proj = m.q_proj
            k_proj = m.k_proj
            v_proj = m.v_proj
            qweights = torch.cat([q_proj.qweight, k_proj.qweight, v_proj.qweight], dim=1)
            qzeros = torch.cat([q_proj.qzeros, k_proj.qzeros, v_proj.qzeros], dim=1)
            scales = torch.cat([q_proj.scales, k_proj.scales, v_proj.scales], dim=1)
            if QuantLinear.QUANT_TYPE == 'exllama':
                if desc_act:
                    raise ValueError('Exllama kernel does not support query/key/value fusion with act-order. Please either use inject_fused_attention=False or disable_exllama=True.')
                else:
                    g_idx = None
            else:
                g_idx = torch.cat([q_proj.g_idx, k_proj.g_idx, v_proj.g_idx], dim=0)
            bias = torch.cat([q_proj.bias, k_proj.bias, v_proj.bias], dim=0) if q_proj.bias is not None else None
            qlinear_args = q_proj.bits, q_proj.group_size, q_proj.infeatures, q_proj.outfeatures + k_proj.outfeatures + v_proj.outfeatures, True if q_proj.bias is not None else False
            qlinear_kwargs = {'trainable': trainable}
            if (not desc_act or group_size == -1) and not use_triton:
                qlinear_kwargs['use_cuda_fp16'] = use_cuda_fp16
            qlinear_kwargs['weight_dtype'] = q_proj.scales.dtype
            qkv_proj = QuantLinear(*qlinear_args, **qlinear_kwargs)
            qkv_proj.qweight = qweights
            qkv_proj.qzeros = qzeros
            qkv_proj.scales = scales
            qkv_proj.g_idx = g_idx
            qkv_proj.bias = bias
            if '.' in name:
                parent_name = name.rsplit('.', 1)[0]
                child_name = name[len(parent_name) + 1:]
                parent = model.get_submodule(parent_name)
            else:
                parent_name = ''
                parent = model
                child_name = name
            attn.qkv_proj = qkv_proj
            attn.out_proj = m.out_proj
            setattr(parent, child_name, attn)
            del m


class FusedLlamaAttentionForQuantizedModel(FusedBaseAttentionModule):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, hidden_size, num_heads, qkv_proj, o_proj, rotary_emb, layer_idx):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.layer_idx = layer_idx
        if self.head_dim * num_heads != self.hidden_size:
            raise ValueError(f'hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size} and `num_heads`: {num_heads}).')
        self.qkv_proj = qkv_proj
        self.o_proj = o_proj
        self.rotary_emb = rotary_emb

    def _shape(self, tensor, seq_len, bsz):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(self, hidden_states, past_key_value=None, attention_mask=None, position_ids=None, output_attentions=False, use_cache=False, **kwargs):
        """Input shape: Batch x Time x Channel"""
        bsz, q_len, _ = hidden_states.size()
        qkv_states = self.qkv_proj(hidden_states)
        query_states, key_states, value_states = torch.split(qkv_states, self.hidden_size, dim=2)
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(f'The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} for auto-regressive decoding with k/v caching, please make sure to initialize the attention class with a layer index. Please open an issue in AutoGPTQ if you hit this.')
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        if past_key_value is not None:
            cache_kwargs = {'sin': sin, 'cos': cos}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
        if use_cache:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()
        if compare_pytorch_version('v2.0.0', op='ge'):
            attn_output = F.scaled_dot_product_attention(query_states, key_states, value_states, attn_mask=attention_mask, is_causal=attention_mask is None and q_len > 1)
            attn_weights = None
        else:
            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
            if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
                raise ValueError(f'Attention weights should be of size {bsz * self.num_heads, q_len, kv_seq_len}, but is {attn_weights.size()}')
            if attention_mask is not None:
                if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                    raise ValueError(f'Attention mask should be of size {bsz, 1, q_len, kv_seq_len}, but is {attention_mask.size()}')
                attn_weights = attn_weights + attention_mask
                attn_weights = torch.max(attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min))
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32)
            attn_output = torch.matmul(attn_weights, value_states)
        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(f'`attn_output` should be of size {bsz, self.num_heads, q_len, self.head_dim}, but is {attn_output.size()}')
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        if not output_attentions:
            attn_weights = None
        return attn_output, attn_weights, past_key_value

    @classmethod
    def inject_to_model(cls, model, use_triton=False, group_size=-1, use_cuda_fp16=True, desc_act=False, trainable=False, bits: 'int'=4, disable_exllama=True, disable_exllamav2=False, **kwargs):
        """
        Replace all LlamaAttention modules with QuantLlamaAttention modules, fusing the q, k, v projections.
        """
        QuantLinear = dynamically_import_QuantLinear(use_triton=use_triton, desc_act=desc_act, group_size=group_size, bits=bits, disable_exllama=disable_exllama, disable_exllamav2=disable_exllamav2)
        for name, m in model.named_modules():
            if not isinstance(m, LlamaAttention):
                continue
            q_proj = m.q_proj
            k_proj = m.k_proj
            v_proj = m.v_proj
            qweights = torch.cat([q_proj.qweight, k_proj.qweight, v_proj.qweight], dim=1)
            qzeros = torch.cat([q_proj.qzeros, k_proj.qzeros, v_proj.qzeros], dim=1)
            scales = torch.cat([q_proj.scales, k_proj.scales, v_proj.scales], dim=1)
            if QuantLinear.QUANT_TYPE == 'exllama':
                if desc_act:
                    raise ValueError('Exllama kernel does not support query/key/value fusion with act-order. Please either use inject_fused_attention=False or disable_exllama=True.')
                else:
                    g_idx = None
            else:
                g_idx = torch.cat([q_proj.g_idx, k_proj.g_idx, v_proj.g_idx], dim=0)
            bias = torch.cat([q_proj.bias, k_proj.bias, v_proj.bias], dim=0) if q_proj.bias is not None else None
            qlinear_args = q_proj.bits, q_proj.group_size, q_proj.infeatures, q_proj.outfeatures + k_proj.outfeatures + v_proj.outfeatures, True if q_proj.bias is not None else False
            qlinear_kwargs = {'trainable': trainable}
            if (not desc_act or group_size == -1) and not use_triton:
                qlinear_kwargs['use_cuda_fp16'] = use_cuda_fp16
            qlinear_kwargs['weight_dtype'] = q_proj.scales.dtype
            qkv_layer = QuantLinear(*qlinear_args, **qlinear_kwargs)
            qkv_layer.qweight = qweights
            qkv_layer.qzeros = qzeros
            qkv_layer.scales = scales
            qkv_layer.g_idx = g_idx
            qkv_layer.bias = bias
            layer_idx = None
            if hasattr(m, 'layer_idx'):
                layer_idx = m.layer_idx
            attn = cls(m.hidden_size, m.num_heads, qkv_layer, m.o_proj, m.rotary_emb, layer_idx=layer_idx)
            if '.' in name:
                parent_name = name.rsplit('.', 1)[0]
                child_name = name[len(parent_name) + 1:]
                parent = model.get_submodule(parent_name)
            else:
                parent_name = ''
                parent = model
                child_name = name
            setattr(parent, child_name, attn)


class FusedLlamaMLPForQuantizedModel(FusedBaseMLPModule):

    def __init__(self, gate_proj, down_proj, up_proj):
        super().__init__()
        self.infeatures = gate_proj.infeatures
        self.intermediate_size = gate_proj.outfeatures
        self.outfeatures = down_proj.outfeatures
        self.bits = gate_proj.bits
        self.maxq = gate_proj.maxq
        self.gate_proj = gate_proj
        self.up_proj = up_proj
        self.down_proj = down_proj

    def forward(self, x):
        return self.down_proj(self.triton_llama_mlp(x))

    def triton_llama_mlp(self, x):
        with torch.device(x.device):
            out_shape = x.shape[:-1] + (self.intermediate_size,)
            x = x.reshape(-1, x.shape[-1])
            M, K = x.shape
            N = self.intermediate_size
            c = torch.empty((M, N), device=x.device, dtype=torch.float16)
            grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),)
            quant_fused_matmul_248_kernel[grid](x, c, self.gate_proj.qweight, self.gate_proj.scales, self.gate_proj.qzeros, self.gate_proj.g_idx, self.up_proj.qweight, self.up_proj.scales, self.up_proj.qzeros, self.up_proj.g_idx, M, N, K, self.bits, self.maxq, x.stride(0), x.stride(1), self.gate_proj.qweight.stride(0), self.gate_proj.qweight.stride(1), c.stride(0), c.stride(1), self.gate_proj.scales.stride(0), self.gate_proj.qzeros.stride(0))
            c = c.reshape(out_shape)
            return c

    @classmethod
    def inject_to_model(cls, model, use_triton=False, **kwargs):
        if not use_triton:
            logger.warning(f'Skipping module injection for {cls.__name__} as currently not supported with use_triton=False.')
            return
        elif not TRITON_AVAILABLE:
            logger.warning(f'Skipping module injection for {cls.__name__} as Triton is not available. Please check your installation.')
            return
        for name, m in model.named_modules():
            if not isinstance(m, LlamaMLP):
                continue
            mlp = cls(m.gate_proj, m.down_proj, m.up_proj)
            if '.' in name:
                parent_name = name.rsplit('.', 1)[0]
                child_name = name[len(parent_name) + 1:]
                parent = model.get_submodule(parent_name)
            else:
                parent_name = ''
                parent = model
                child_name = name
            setattr(parent, child_name, mlp)

    @classmethod
    def warmup(cls, model, transpose=False, seqlen=2048):
        kn_values = {}
        for _, m in model.named_modules():
            if not isinstance(m, cls):
                continue
            k = m.infeatures
            n = m.intermediate_size
            if (k, n) not in kn_values:
                kn_values[k, n] = m
        logger.info(f'Found {len(kn_values)} unique fused mlp KN values.')
        logger.info('Warming up autotune cache ...')
        with torch.no_grad():
            for m in tqdm(range(0, math.ceil(math.log2(seqlen)) + 1)):
                m = 2 ** m
                for (k, n), modules in kn_values.items():
                    a = torch.randn(m, k, dtype=torch.float16, device=model.device)
                    modules.triton_llama_mlp(a)
        del kn_values


def _get_linear_feature_count(linear_layer: 'LinearLayer') ->Tuple[int, int]:
    in_features = getattr(linear_layer, 'in_features', getattr(linear_layer, 'infeatures'))
    out_features = getattr(linear_layer, 'out_features', getattr(linear_layer, 'outfeatures'))
    return in_features, out_features


def _get_weight(linear_layer: 'LinearLayer') ->torch.Tensor:
    return getattr(linear_layer, 'weight', getattr(linear_layer, 'qweight'))

