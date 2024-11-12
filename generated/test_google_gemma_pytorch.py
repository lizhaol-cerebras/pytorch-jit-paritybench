
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


import enum


import torch


from typing import Optional


from typing import Sequence


from torch import nn


import torch.nn.functional as F


from typing import Any


from typing import List


from typing import Tuple


from typing import Union


import re


from typing import Mapping


from copy import deepcopy


from typing import Callable


import torch.ao.quantization.fx._decomposed


import torch.distributed as dist


import torch.distributed._functional_collectives as fc


import torch.distributed.distributed_c10d as c10d


import torch.nn.init as init


from torch.nn.parameter import Parameter


import random


import numpy as np


import torch.multiprocessing


GROUP_SIZE = None


RANKSET = None


TAG = None


def my_reduce(input_: 'torch.Tensor', groups, world_size, rank) ->torch.Tensor:
    """All-reduce the the input tensor across model parallel group."""
    if world_size == 1:
        return input_
    if USE_CUDA:
        input_ = torch.ops.c10d_functional.all_reduce(input_, 'sum', TAG, RANKSET, GROUP_SIZE)
    else:
        input_ = xm.all_reduce(xm.REDUCE_SUM, input_, groups=groups)
    return input_


class _ReduceFromModelParallelRegion(torch.autograd.Function):
    """All-redcue the input from the model parallel region."""

    @staticmethod
    def forward(ctx, input_, groups, world_size, rank):
        return my_reduce(input_, groups, world_size, rank)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


def reduce_from_model_parallel_region(input_: 'torch.Tensor', groups, world_size, rank) ->torch.Tensor:
    return _ReduceFromModelParallelRegion.apply(input_, groups, world_size, rank)


def my_gather(input_: 'torch.Tensor', groups, world_size, rank) ->torch.Tensor:
    """Gather tensors and concatinate along the last dimension."""
    if world_size == 1:
        return input_
    if USE_CUDA:
        last_dim = input_.dim() - 1
        size = input_.size(last_dim)
        padding = [0] * (2 * input_.dim())
        ordinal = rank
        left, right = ordinal, world_size - 1 - ordinal
        idx = input_.dim() - 1 - last_dim
        padding[2 * idx] = left * size
        padding[2 * idx + 1] = right * size
        output = torch.ops.c10d_functional.all_reduce(F.pad(input_, padding), 'sum', TAG, RANKSET, GROUP_SIZE)
    else:
        output = xm.all_gather(input_, dim=-1, groups=groups)
    return output


def ensure_divisibility(numerator: 'int', denominator: 'int') ->None:
    """Ensure that numerator is divisible by the denominator."""
    assert numerator % denominator == 0, '{} is not divisible by {}'.format(numerator, denominator)


def divide_and_check_no_remainder(numerator: 'int', denominator: 'int') ->int:
    """Ensure that numerator is divisible by the denominator and return
    the division value."""
    ensure_divisibility(numerator, denominator)
    return numerator // denominator


def split_tensor_along_last_dim(tensor: 'torch.Tensor', num_partitions: 'int', contiguous_split_chunks: 'bool'=False) ->Tuple[torch.Tensor, ...]:
    """Split a tensor along its last dimension.
    Arguments:
        tensor: input tensor.
        num_partitions: number of partitions to split the tensor
        contiguous_split_chunks: If True, make each chunk contiguous
                                in memory.
    """
    last_dim = tensor.dim() - 1
    last_dim_size = divide_and_check_no_remainder(tensor.size()[last_dim], num_partitions)
    tensor_list = torch.split(tensor, last_dim_size, dim=last_dim)
    if contiguous_split_chunks:
        return tuple(chunk.contiguous() for chunk in tensor_list)
    return tensor_list


def my_split(input_: 'torch.Tensor', groups, world_size, rank) ->torch.Tensor:
    """Split the tensor along its last dimension and keep the

    corresponding slice.
    """
    if world_size == 1:
        return input_
    input_list = split_tensor_along_last_dim(input_, world_size)
    output = input_list[rank].contiguous()
    return output


class _ScatterToModelParallelRegion(torch.autograd.Function):
    """Split the input and keep only the corresponding chuck to the rank."""

    @staticmethod
    def forward(ctx, input_, groups, world_size, rank):
        ctx.groups, ctx.world_size, ctx.rank = groups, world_size, rank
        return my_split(input_, groups, world_size, rank)

    @staticmethod
    def backward(ctx, grad_output):
        groups, world_size, rank = ctx.groups, ctx.world_size, ctx.rank
        return my_gather(grad_output, groups, world_size, rank)


def scatter_to_model_parallel_region(input_: 'torch.Tensor', groups, world_size, rank) ->torch.Tensor:
    return _ScatterToModelParallelRegion.apply(input_, groups, world_size, rank)


class Sampler(nn.Module):

    def __init__(self, vocab_size: 'int', world_size: 'int', rank: 'int', config: 'gemma_config.GemmaConfig') ->None:
        super().__init__()
        self.vocab_size = vocab_size
        self.world_size = world_size
        self.rank = rank
        self.config = config

    @torch.no_grad()
    def forward(self, embedding: 'torch.Tensor', hidden_states: 'torch.Tensor', output_positions: 'torch.Tensor', temperatures: 'Union[torch.Tensor, None]', top_ps: 'torch.Tensor', top_ks: 'torch.Tensor', embedding_bias: 'Optional[torch.Tensor]'=None) ->Tuple[torch.Tensor, torch.Tensor]:
        hidden_states = hidden_states.index_select(1, output_positions).squeeze(dim=1)
        hidden_states_parallel = scatter_to_model_parallel_region(hidden_states, groups=None, world_size=self.world_size, rank=self.rank)
        hidden_states_parallel = torch.matmul(hidden_states_parallel, embedding.t())
        logits = reduce_from_model_parallel_region(hidden_states_parallel, groups=None, world_size=self.world_size, rank=self.rank)
        if embedding_bias is not None:
            logits += embedding_bias
        if self.config.final_logit_softcapping is not None:
            logits = logits / self.config.final_logit_softcapping
            logits = torch.tanh(logits)
            logits = logits * self.config.final_logit_softcapping
        if temperatures is None:
            return torch.argmax(logits, dim=-1).squeeze(dim=-1), logits
        logits.div_(temperatures.unsqueeze(dim=1))
        probs = torch.softmax(logits, dim=-1, dtype=torch.float)
        probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
        probs_sum = torch.cumsum(probs_sort, dim=-1)
        top_ps_mask = probs_sum - probs_sort > top_ps.unsqueeze(dim=1)
        probs_sort = torch.where(top_ps_mask, 0, probs_sort)
        top_ks_mask = torch.arange(probs_idx.shape[-1], device=probs_idx.device)
        top_ks_mask = top_ks_mask.expand(probs_idx.shape[0], -1)
        top_ks_mask = top_ks_mask >= top_ks.unsqueeze(dim=1)
        probs_sort = torch.where(top_ks_mask, 0, probs_sort)
        probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
        probs = torch.gather(probs_sort, dim=-1, index=torch.argsort(probs_idx, dim=-1))
        next_token_ids = torch.multinomial(probs, num_samples=1, replacement=True).squeeze(dim=-1)
        return next_token_ids, logits


class Linear(nn.Module):

    def __init__(self, in_features: 'int', out_features: 'int', quant: 'bool'):
        super().__init__()
        if quant:
            self.weight = nn.Parameter(torch.empty((out_features, in_features), dtype=torch.int8), requires_grad=False)
            self.weight_scaler = nn.Parameter(torch.Tensor(out_features))
        else:
            self.weight = nn.Parameter(torch.empty((out_features, in_features)), requires_grad=False)
        self.quant = quant

    def forward(self, x):
        weight = self.weight
        if self.quant:
            weight = weight * self.weight_scaler.unsqueeze(-1)
        output = F.linear(x, weight)
        return output


class Embedding(nn.Module):

    def __init__(self, num_embeddings: 'int', embedding_dim: 'int', quant: 'bool'):
        super().__init__()
        if quant:
            self.weight = nn.Parameter(torch.empty((num_embeddings, embedding_dim), dtype=torch.int8), requires_grad=False)
            self.weight_scaler = nn.Parameter(torch.Tensor(num_embeddings))
        else:
            self.weight = nn.Parameter(torch.empty((num_embeddings, embedding_dim)), requires_grad=False)
        self.quant = quant

    def forward(self, x):
        weight = self.weight
        if self.quant:
            weight = weight * self.weight_scaler.unsqueeze(-1)
        output = F.embedding(x, weight)
        return output


class RMSNorm(torch.nn.Module):

    def __init__(self, dim: 'int', eps: 'float'=1e-06, add_unit_offset: 'bool'=True):
        super().__init__()
        self.eps = eps
        self.add_unit_offset = add_unit_offset
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float())
        if self.add_unit_offset:
            output = output * (1 + self.weight.float())
        else:
            output = output * self.weight.float()
        return output.type_as(x)


def _initialize_affine_weight(weight: 'torch.Tensor', out_features: 'int', in_features: 'int', per_partition_size: 'int', partition_dim: 'int', init_method: 'Callable[[torch.Tensor], torch.Tensor]', world_size: 'int', rank: 'int', stride: 'int'=1, return_master_weight: 'bool'=False) ->Optional[torch.Tensor]:
    """Initialize affine weight for model parallel.

    Build the master weight on all processes and scatter
    the relevant chunk.
    """
    if world_size == 1:
        init_method(weight)
        if return_master_weight:
            return weight
        return None
    master_weight = torch.empty(out_features, in_features, dtype=weight.dtype, requires_grad=False)
    init_method(master_weight)
    per_partition_per_stride_size = divide_and_check_no_remainder(per_partition_size, stride)
    weight_list = torch.split(master_weight, per_partition_per_stride_size, dim=partition_dim)
    my_weight_list = weight_list[rank::world_size]
    with torch.no_grad():
        torch.cat(my_weight_list, dim=partition_dim, out=weight)
    if return_master_weight:
        return master_weight
    return None


class _CopyToModelParallelRegion(torch.autograd.Function):
    """Pass the input to the model parallel region."""

    @staticmethod
    def forward(ctx, input_, groups, world_size, rank):
        ctx.groups, ctx.world_size, ctx.rank = groups, world_size, rank
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        groups, world_size, rank = ctx.groups, ctx.world_size, ctx.rank
        return my_reduce(grad_output, groups, world_size, rank)


def copy_to_model_parallel_region(input_: 'torch.Tensor', groups, world_size, rank) ->torch.Tensor:
    return _CopyToModelParallelRegion.apply(input_, groups, world_size, rank)


class _GatherFromModelParallelRegion(torch.autograd.Function):
    """Gather the input from model parallel region and concatinate."""

    @staticmethod
    def forward(ctx, input_, groups, world_size, rank):
        ctx.groups, ctx.world_size, ctx.rank = groups, world_size, rank
        return my_gather(input_, groups, world_size, rank)

    @staticmethod
    def backward(ctx, grad_output):
        groups, world_size, rank = ctx.groups, ctx.world_size, ctx.rank
        return my_split(grad_output, groups, world_size, rank)


def gather_from_model_parallel_region(input_: 'torch.Tensor', groups, world_size, rank) ->torch.Tensor:
    return _GatherFromModelParallelRegion.apply(input_, groups, world_size, rank)


def get_model_parallel_group():
    return None


def get_model_parallel_rank():
    if USE_CUDA:
        return dist.get_rank()
    return xm.get_ordinal()


def get_model_parallel_world_size():
    if USE_CUDA:
        return dist.get_world_size()
    return xm.xrt_world_size()


EPS = torch.finfo(torch.float32).eps


def _find_per_channel_min_max(x: 'torch.Tensor', axis: 'int'):
    x_dim = x.size()
    new_axis_list = list(range(len(x_dim)))
    new_axis_list[axis] = 0
    new_axis_list[0] = axis
    y = x.permute(new_axis_list)
    y = torch.flatten(y, start_dim=1)
    return torch.aminmax(y, dim=1)


def _find_qparams(x: 'torch.Tensor', qconfig: 'TensorQConfig'):
    axis = qconfig.axis
    dtype = qconfig.dtype
    symmetric_quant = qconfig.symmetric_quant
    quant_min = qconfig.quant_min
    quant_max = qconfig.quant_max
    assert axis >= 0 and axis < len(x.shape)
    assert dtype == torch.int8
    min_val, max_val = _find_per_channel_min_max(x, axis)
    min_val_neg = torch.min(min_val, torch.zeros_like(min_val))
    max_val_pos = torch.max(max_val, torch.zeros_like(max_val))
    scale = torch.ones(min_val_neg.size(), dtype=torch.float32)
    if symmetric_quant:
        max_val_pos = torch.max(-min_val_neg, max_val_pos)
        scale = max_val_pos / (float(quant_max - quant_min) / 2)
        eps = torch.zeros_like(scale).fill_(EPS)
        scale = torch.max(scale, eps)
        return scale, None
    else:
        assert symmetric_quant


def _quantize_to_dtype(x: 'torch.Tensor', qconfig: 'TensorQConfig', scale: 'torch.Tensor', zero_point: 'Optional[torch.Tensor]'=None):
    if zero_point is None:
        zero_point = torch.zeros_like(scale)
    return torch.ops.quantized_decomposed.quantize_per_channel(x, scale, zero_point, qconfig.axis, qconfig.quant_min, qconfig.quant_max, qconfig.dtype)


def quantize_tensor(x: 'torch.Tensor', qconfig: 'TensorQConfig'):
    scale, zp = _find_qparams(x, qconfig)
    x_int = _quantize_to_dtype(x, qconfig, scale, zp)
    return x_int, scale, zp


class ColumnParallelLinear(torch.nn.Module):
    """Linear layer with column parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its second dimension as A = [A_1, ..., A_p].

    Arguments:
        in_features: first dimension of matrix A.
        out_features: second dimension of matrix A.
        bias: If true, add bias
        gather_output: If true, call all-gether on output and make Y available to
          all GPUs, otherwise, every GPU will have its output which is Y_i = XA_i
        init_method: method to initialize weights. Note that bias is always set to
          zero.
        stride: For the strided linear layers.
        keep_master_weight_for_test: This was added for testing and should be set
          to False. It returns the master weights used for initialization.
    """

    def __init__(self, in_features: 'int', out_features: 'int', bias: 'bool'=True, gather_output: 'bool'=True, init_method: 'Callable[[torch.Tensor], torch.Tensor]'=init.xavier_normal_, stride: 'int'=1, keep_master_weight_for_test: 'bool'=False, world_size: 'Optional[int]'=None, rank: 'Optional[int]'=None, groups: 'Optional[List]'=None, quant: 'bool'=False) ->None:
        super(ColumnParallelLinear, self).__init__()
        if world_size is None:
            self.groups = get_model_parallel_group()
            self.world_size = get_model_parallel_world_size()
            self.rank = get_model_parallel_rank()
        else:
            self.groups = groups
            self.world_size = world_size
            self.rank = rank
        self.in_features = in_features
        self.out_features = out_features
        self.gather_output = gather_output
        self.quant = quant
        self.output_size_per_partition = divide_and_check_no_remainder(out_features, self.world_size)
        if quant:
            self.weight = Parameter(torch.empty((self.output_size_per_partition, self.in_features), dtype=torch.int8), requires_grad=False)
            self.weight_scaler = Parameter(torch.Tensor(self.output_size_per_partition))
        else:
            self.weight = Parameter(torch.Tensor(self.output_size_per_partition, self.in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(self.output_size_per_partition))
            with torch.no_grad():
                self.bias.zero_()
        else:
            self.register_parameter('bias', None)
        self.master_weight = _initialize_affine_weight(self.weight, self.out_features, self.in_features, self.output_size_per_partition, 0, init_method, self.world_size, self.rank, stride=stride, return_master_weight=keep_master_weight_for_test)

    def get_master_weight(self) ->torch.Tensor:
        return gather_from_model_parallel_region(self.weight.data.transpose(0, 1), self.groups, self.world_size, self.rank).transpose_(0, 1)

    def set_quantize(self):
        assert not self.quant
        self.weight = Parameter(torch.empty((self.output_size_per_partition, self.in_features), dtype=torch.int8), requires_grad=False)
        self.weight_scaler = Parameter(torch.Tensor(self.output_size_per_partition))
        self.quant = True

    def quantize(self):
        assert not self.quant
        fp_w = deepcopy(self.weight.data)
        orig_dtype = fp_w.dtype
        fp_w = fp_w
        self.weight = Parameter(torch.empty((self.output_size_per_partition, self.in_features), dtype=torch.int8), requires_grad=False)
        self.weight_scaler = Parameter(torch.Tensor(self.output_size_per_partition))
        qconfig = TensorQConfig(axis=0)
        self.weight.data, scale, zero_point = quantize_tensor(fp_w, qconfig)
        self.weight_scaler.data = scale
        self.quant = True

    def forward(self, input_: 'torch.Tensor') ->torch.Tensor:
        input_parallel = copy_to_model_parallel_region(input_, self.groups, self.world_size, self.rank)
        if self.quant and USE_CUDA:
            scaled_weight = self.weight * self.weight_scaler
            output_parallel = F.linear(input_parallel, scaled_weight, self.bias)
        elif self.quant:
            output_parallel = F.linear(input_parallel, self.weight, self.bias)
            output_parallel = output_parallel * self.weight_scaler
        else:
            output_parallel = F.linear(input_parallel, self.weight, self.bias)
        if self.gather_output:
            output = gather_from_model_parallel_region(output_parallel, self.groups, self.world_size, self.rank)
        else:
            output = output_parallel
        return output


class RowParallelLinear(torch.nn.Module):
    """Linear layer with row parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its first dimension and X along its second dimension as:
               -   -
              | A_1 |
              | .   |
          A = | .   |        X = [X_1, ..., X_p]
              | .   |
              | A_p |
               -   -
    Arguments:
        in_features: first dimension of matrix A.
        out_features: second dimension of matrix A.
        bias: If true, add bias. Note that bias is not parallelized.
        input_is_parallel: If true, we assume that the input is already split
          across the GPUs and we do not split again.
        init_method: method to initialize weights. Note that bias is always set to
          zero.
        stride: For the strided linear layers.
        keep_master_weight_for_test: This was added for testing and should be set
          to False. It returns the master weights used for initialization.
    """

    def __init__(self, in_features: 'int', out_features: 'int', bias: 'bool'=True, input_is_parallel: 'bool'=False, init_method: 'Callable[[torch.Tensor], torch.Tensor]'=init.xavier_normal_, stride: 'int'=1, keep_master_weight_for_test: 'bool'=False, world_size: 'Optional[int]'=None, rank: 'Optional[int]'=None, groups: 'Optional[List]'=None, quant: 'bool'=False):
        super(RowParallelLinear, self).__init__()
        if world_size is None:
            self.groups = get_model_parallel_group()
            self.world_size = get_model_parallel_world_size()
            self.rank = get_model_parallel_rank()
        else:
            self.groups = groups
            self.world_size = world_size
            self.rank = rank
        self.in_features = in_features
        self.out_features = out_features
        self.input_is_parallel = input_is_parallel
        self.quant = quant
        self.input_size_per_partition = divide_and_check_no_remainder(in_features, self.world_size)
        if quant:
            self.weight = Parameter(torch.empty((self.out_features, self.input_size_per_partition), dtype=torch.int8), requires_grad=False)
            self.weight_scaler = Parameter(torch.Tensor(self.out_features))
        else:
            self.weight = Parameter(torch.Tensor(self.out_features, self.input_size_per_partition))
        if bias:
            self.bias = Parameter(torch.Tensor(self.out_features))
            with torch.no_grad():
                self.bias.zero_()
        else:
            self.register_parameter('bias', None)
        self.master_weight = _initialize_affine_weight(self.weight, self.out_features, self.in_features, self.input_size_per_partition, 1, init_method, self.world_size, self.rank, stride=stride, return_master_weight=keep_master_weight_for_test)

    def get_master_weight(self) ->torch.Tensor:
        return gather_from_model_parallel_region(self.weight.data, self.groups, self.world_size, self.rank)

    def set_quantize(self):
        assert not self.quant
        self.weight = Parameter(torch.empty((self.out_features, self.input_size_per_partition), dtype=torch.int8), requires_grad=False)
        self.weight_scaler = Parameter(torch.Tensor(self.out_features))
        self.quant = True

    def quantize(self):
        assert not self.quant
        fp_w = deepcopy(self.weight.data)
        orig_dtype = fp_w.dtype
        fp_w = fp_w
        self.weight = Parameter(torch.empty((self.out_features, self.input_size_per_partition), dtype=torch.int8), requires_grad=False)
        self.weight_scaler = Parameter(torch.Tensor(self.out_features))
        qconfig = TensorQConfig(axis=0)
        self.weight.data, scale, zero_point = quantize_tensor(fp_w, qconfig)
        self.weight_scaler.data = scale
        self.quant = True

    def forward(self, input_: 'torch.Tensor') ->torch.Tensor:
        if self.input_is_parallel:
            input_parallel = input_
        else:
            input_parallel = scatter_to_model_parallel_region(input_, self.groups, self.world_size, self.rank)
        if self.quant and USE_CUDA:
            scaled_weight = self.weight * self.weight_scaler
            output_parallel = F.linear(input_parallel, scaled_weight, self.bias)
        elif self.quant:
            output_parallel = F.linear(input_parallel, self.weight, self.bias)
            output_parallel = output_parallel * self.weight_scaler
        else:
            output_parallel = F.linear(input_parallel, self.weight)
        output_ = reduce_from_model_parallel_region(output_parallel, self.groups, self.world_size, self.rank)
        if self.bias is not None:
            output = output_ + self.bias
        else:
            output = output_
        return output


class GemmaMLP(nn.Module):

    def __init__(self, hidden_size: 'int', intermediate_size: 'int', world_size: 'int', rank: 'int', quant: 'bool'):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        def init_method(x):
            return x
        self.gate_proj = ColumnParallelLinear(hidden_size, intermediate_size, bias=False, gather_output=False, init_method=init_method, world_size=world_size, rank=rank, quant=quant)
        self.up_proj = ColumnParallelLinear(hidden_size, intermediate_size, bias=False, gather_output=False, init_method=init_method, world_size=world_size, rank=rank, quant=quant)
        self.down_proj = RowParallelLinear(intermediate_size, hidden_size, bias=False, input_is_parallel=True, init_method=init_method, world_size=world_size, rank=rank, quant=quant)

    def forward(self, x):
        gate = self.gate_proj(x)
        gate = F.gelu(gate, approximate='tanh')
        up = self.up_proj(x)
        fuse = gate * up
        outputs = self.down_proj(fuse)
        return outputs


def apply_rotary_emb(x: 'torch.Tensor', freqs_cis: 'torch.Tensor') ->torch.Tensor:
    """Applies the rotary embedding to the query and key tensors."""
    x_ = torch.view_as_complex(torch.stack(torch.chunk(x.transpose(1, 2).float(), 2, dim=-1), dim=-1))
    x_out = torch.view_as_real(x_ * freqs_cis).type_as(x)
    x_out = torch.cat(torch.chunk(x_out, 2, dim=-1), dim=-2)
    x_out = x_out.reshape(x_out.shape[0], x_out.shape[1], x_out.shape[2], -1).transpose(1, 2)
    return x_out


class GemmaAttention(nn.Module):

    def __init__(self, hidden_size: 'int', num_heads: 'int', num_kv_heads: 'int', attn_logit_softcapping: 'Optional[float]', query_pre_attn_scalar: 'Optional[int]', head_dim: 'int', world_size: 'int', rank: 'int', quant: 'bool', attn_type: 'gemma_config.AttentionType', sliding_window_size: 'Optional[int]'=None):
        super().__init__()
        self.rank = rank

        def init_method(x):
            return x
        self.total_num_heads = num_heads
        assert self.total_num_heads % world_size == 0
        self.num_heads = self.total_num_heads // world_size
        if num_kv_heads < world_size:
            assert world_size % num_kv_heads == 0
            self.total_num_kv_heads = world_size
        else:
            assert num_kv_heads % world_size == 0
            self.total_num_kv_heads = num_kv_heads
        self.num_kv_heads = self.total_num_kv_heads // world_size
        assert self.num_heads % self.num_kv_heads == 0
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads
        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        if query_pre_attn_scalar is not None:
            self.scaling = query_pre_attn_scalar ** -0.5
        else:
            self.scaling = self.head_dim ** -0.5
        self.qkv_proj = ColumnParallelLinear(self.hidden_size, (self.total_num_heads + 2 * self.total_num_kv_heads) * self.head_dim, bias=False, gather_output=False, init_method=init_method, world_size=world_size, rank=rank, quant=quant)
        self.o_proj = RowParallelLinear(self.total_num_heads * self.head_dim, self.hidden_size, bias=False, input_is_parallel=True, init_method=init_method, world_size=world_size, rank=rank, quant=quant)
        self.attn_type = attn_type
        self.sliding_window_size = sliding_window_size
        self.attn_logit_softcapping = attn_logit_softcapping

    def forward(self, hidden_states: 'torch.Tensor', freqs_cis: 'torch.Tensor', kv_write_indices: 'torch.Tensor', kv_cache: 'Tuple[torch.Tensor, torch.Tensor]', mask: 'torch.Tensor') ->torch.Tensor:
        hidden_states_shape = hidden_states.shape
        assert len(hidden_states_shape) == 3
        batch_size, input_len, _ = hidden_states_shape
        qkv = self.qkv_proj(hidden_states)
        xq, xk, xv = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        xq = xq.view(batch_size, -1, self.num_heads, self.head_dim)
        xk = xk.view(batch_size, -1, self.num_kv_heads, self.head_dim)
        xv = xv.view(batch_size, -1, self.num_kv_heads, self.head_dim)
        xq = apply_rotary_emb(xq, freqs_cis=freqs_cis)
        xk = apply_rotary_emb(xk, freqs_cis=freqs_cis)
        k_cache, v_cache = kv_cache
        k_cache.index_copy_(1, kv_write_indices, xk)
        v_cache.index_copy_(1, kv_write_indices, xv)
        key = k_cache
        value = v_cache
        if self.num_kv_heads != self.num_heads:
            key = torch.repeat_interleave(key, self.num_queries_per_kv, dim=2)
            value = torch.repeat_interleave(value, self.num_queries_per_kv, dim=2)
        q = xq.transpose(1, 2)
        k = key.transpose(1, 2)
        v = value.transpose(1, 2)
        q.mul_(self.scaling)
        scores = torch.matmul(q, k.transpose(2, 3))
        if self.attn_type == gemma_config.AttentionType.LOCAL_SLIDING and self.sliding_window_size is not None:
            all_ones = torch.ones_like(mask)
            sliding_mask = torch.triu(all_ones, -1 * self.sliding_window_size + 1) * torch.tril(all_ones, self.sliding_window_size - 1)
            mask = torch.where(sliding_mask == 1, mask, -2.3819763e+38)
        if self.attn_logit_softcapping is not None:
            scores = scores / self.attn_logit_softcapping
            scores = torch.tanh(scores)
            scores = scores * self.attn_logit_softcapping
        scores = scores + mask
        scores = F.softmax(scores.float(), dim=-1).type_as(q)
        output = torch.matmul(scores, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, input_len, -1)
        output = self.o_proj(output)
        return output


class GemmaDecoderLayer(nn.Module):

    def __init__(self, config: 'gemma_config.GemmaConfig', world_size: 'int', rank: 'int'):
        super().__init__()
        self.rank = rank
        self.self_attn = GemmaAttention(hidden_size=config.hidden_size, num_heads=config.num_attention_heads, num_kv_heads=config.num_key_value_heads, attn_logit_softcapping=config.attn_logit_softcapping, query_pre_attn_scalar=config.query_pre_attn_scalar, head_dim=config.head_dim, world_size=world_size, rank=rank, quant=config.quant, attn_type=gemma_config.AttentionType.GLOBAL)
        self.mlp = GemmaMLP(hidden_size=config.hidden_size, intermediate_size=config.intermediate_size, world_size=world_size, rank=rank, quant=config.quant)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states: 'torch.Tensor', freqs_cis: 'torch.Tensor', kv_write_indices: 'torch.Tensor', kv_cache: 'Tuple[torch.Tensor, torch.Tensor]', mask: 'torch.Tensor') ->torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states=hidden_states, freqs_cis=freqs_cis, kv_write_indices=kv_write_indices, kv_cache=kv_cache, mask=mask)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class Gemma2DecoderLayer(nn.Module):

    def __init__(self, config: 'gemma_config.GemmaConfig', attn_type: 'gemma_config.AttentionType', world_size: 'int', rank: 'int'):
        super().__init__()
        self.rank = rank
        self.self_attn = GemmaAttention(hidden_size=config.hidden_size, num_heads=config.num_attention_heads, num_kv_heads=config.num_key_value_heads, attn_logit_softcapping=config.attn_logit_softcapping, query_pre_attn_scalar=config.query_pre_attn_scalar, head_dim=config.head_dim, world_size=world_size, rank=rank, quant=config.quant, attn_type=attn_type, sliding_window_size=config.sliding_window_size)
        self.mlp = GemmaMLP(hidden_size=config.hidden_size, intermediate_size=config.intermediate_size, world_size=world_size, rank=rank, quant=config.quant)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_feedforward_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps) if config.use_pre_ffw_norm else None
        self.post_feedforward_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps) if config.use_post_ffw_norm else None

    def forward(self, hidden_states: 'torch.Tensor', freqs_cis: 'torch.Tensor', kv_write_indices: 'torch.Tensor', kv_cache: 'Tuple[torch.Tensor, torch.Tensor]', mask: 'torch.Tensor') ->torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states=hidden_states, freqs_cis=freqs_cis, kv_write_indices=kv_write_indices, kv_cache=kv_cache, mask=mask)
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states
        residual = hidden_states
        if self.pre_feedforward_layernorm is not None:
            hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        if self.post_feedforward_layernorm is not None:
            hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class GemmaModel(nn.Module):

    def __init__(self, config: 'gemma_config.GemmaConfig', world_size: 'int', rank: 'int'):
        super().__init__()
        self.config = config
        self.rank = rank
        self.vocab_size = config.vocab_size
        self.layers = nn.ModuleList()
        for i in range(config.num_hidden_layers):
            if config.architecture == gemma_config.Architecture.GEMMA_1:
                self.layers.append(GemmaDecoderLayer(config))
            elif config.architecture == gemma_config.Architecture.GEMMA_2:
                attn_type = config.attn_types[i] if config.attn_types is not None else gemma_config.AttentionType.GLOBAL
                self.layers.append(Gemma2DecoderLayer(config, attn_type, world_size, rank))
            else:
                raise ValueError(f'Unknown architecture: {config.architecture}')
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states: 'torch.Tensor', freqs_cis: 'torch.Tensor', kv_write_indices: 'torch.Tensor', kv_caches: 'List[Tuple[torch.Tensor, torch.Tensor]]', mask: 'torch.Tensor') ->torch.Tensor:
        for i in range(len(self.layers)):
            layer = self.layers[i]
            hidden_states = layer(hidden_states=hidden_states, freqs_cis=freqs_cis, kv_write_indices=kv_write_indices, kv_cache=kv_caches[i], mask=mask)
        hidden_states = self.norm(hidden_states)
        return hidden_states


class ParallelEmbedding(torch.nn.Module):
    """Embedding parallelized in the embedding dimension.

    This is mainly adapted from torch.nn.Embedding and all the default
    values are kept.
    Arguments:
        num_embeddings: vocabulary size.
        embedding_dim: size of hidden state.
        init_method: method to initialize weights.
    """

    def __init__(self, num_embeddings: 'int', embedding_dim: 'int', padding_idx: 'Optional[int]'=None, max_norm: 'Optional[float]'=None, norm_type: 'float'=2.0, scale_grad_by_freq: 'bool'=False, sparse: 'bool'=False, init_method: 'Callable[[torch.Tensor], torch.Tensor]'=init.xavier_normal_, keep_master_weight_for_test: 'bool'=False, world_size: 'Optional[int]'=None, rank: 'Optional[int]'=None, groups: 'Optional[List]'=None, quant: 'bool'=False) ->None:
        super(ParallelEmbedding, self).__init__()
        if world_size is None:
            self.groups = get_model_parallel_group()
            self.world_size = get_model_parallel_world_size()
            self.rank = get_model_parallel_rank()
        else:
            self.groups = groups
            self.world_size = world_size
            self.rank = rank
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        self.sparse = sparse
        self._weight = None
        self.quant = quant
        self.embedding_dim_per_partition = divide_and_check_no_remainder(self.embedding_dim, self.world_size)
        if quant:
            self.weight = Parameter(torch.empty((self.num_embeddings, self.embedding_dim_per_partition), dtype=torch.int8), requires_grad=False)
            self.weight_scaler = Parameter(torch.Tensor(self.num_embeddings))
        else:
            self.weight = Parameter(torch.Tensor(self.num_embeddings, self.embedding_dim_per_partition))
        _initialize_affine_weight(self.weight, self.num_embeddings, self.embedding_dim, self.embedding_dim_per_partition, 1, init_method, self.world_size, self.rank, stride=1, return_master_weight=False)

    def forward(self, input_: 'torch.Tensor') ->torch.Tensor:
        input_parallel = copy_to_model_parallel_region(input_, self.groups, self.world_size, self.rank)
        if USE_CUDA:
            input_parallel = torch.remainder(input_parallel, self.weight.shape[0])
        weight = self.weight
        if self.quant:
            weight = weight * self.weight_scaler.unsqueeze(-1)
        output_parallel = F.embedding(input_parallel, weight, self.padding_idx, self.max_norm, self.norm_type, self.scale_grad_by_freq, self.sparse)
        output = gather_from_model_parallel_region(output_parallel, self.groups, self.world_size, self.rank)
        return output


def precompute_freqs_cis(dim: 'int', end: 'int', theta: 'float'=10000.0):
    """Precomputes the frequency cis."""
    freqs = 1.0 / theta ** (torch.arange(0, dim, 2)[:dim // 2].float() / dim)
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


class GemmaForCausalLM(nn.Module):

    def __init__(self, config: 'gemma_config.GemmaConfig', world_size: 'int', rank: 'int', device: 'torch.device'):
        super().__init__()
        self.config = config
        self.world_size = world_size
        self.rank = rank
        self.device = device
        assert config.num_attention_heads % world_size == 0
        assert config.hidden_size % config.num_attention_heads == 0
        max_seq_len = config.max_position_embeddings
        head_dim = config.head_dim
        vocab_size = config.vocab_size

        def init_method(x):
            return x
        self.embedder = ParallelEmbedding(vocab_size, config.hidden_size, init_method=init_method, world_size=world_size, rank=rank, quant=config.quant)
        self.model = GemmaModel(config, world_size, rank)
        self.sampler = Sampler(vocab_size, world_size, rank, config)
        rope_theta = getattr(config, 'rope_theta', 10000)
        freqs_cis = precompute_freqs_cis(head_dim, max_seq_len * 2, theta=rope_theta)
        self.register_buffer('freqs_cis', freqs_cis)

    @torch.no_grad()
    def forward(self, input_token_ids: 'torch.Tensor', input_positions: 'torch.Tensor', kv_write_indices: 'torch.Tensor', kv_caches: 'List[Tuple[torch.Tensor, torch.Tensor]]', mask: 'torch.Tensor', output_positions: 'torch.Tensor', temperatures: 'Union[torch.Tensor, None]', top_ps: 'torch.Tensor', top_ks: 'torch.Tensor', **kwargs) ->Tuple[torch.Tensor, torch.Tensor]:
        freqs_cis = self.freqs_cis.index_select(0, input_positions)
        kv_write_indices = input_positions
        hidden_states = self.embedder(input_token_ids)
        normalizer = torch.tensor(self.config.hidden_size ** 0.5, dtype=hidden_states.dtype)
        hidden_states = hidden_states * normalizer
        hidden_states = self.model(hidden_states=hidden_states, freqs_cis=freqs_cis, kv_write_indices=kv_write_indices, kv_caches=kv_caches, mask=mask)
        embedder_weight = self.embedder.weight
        if self.config.quant:
            embedder_weight = embedder_weight * self.embedder.weight_scaler.unsqueeze(-1)
        next_tokens, logits = self.sampler(embedding=embedder_weight, hidden_states=hidden_states, output_positions=output_positions, temperatures=temperatures, top_ps=top_ps, top_ks=top_ks)
        return next_tokens, logits

    def _load_weights(self, model_state_dict: 'Mapping[str, torch.Tensor]'):
        num_attn_heads = self.config.num_attention_heads
        num_kv_heads = self.config.num_key_value_heads
        head_dim = self.config.head_dim
        hidden_size = self.config.hidden_size

        def split(tensor: 'torch.Tensor', axis: 'int') ->torch.Tensor:
            axis_len = tensor.shape[axis]
            split_len = axis_len // self.world_size
            split_start = split_len * self.rank
            split_end = split_start + split_len
            tensor = torch.moveaxis(tensor, axis, 0)
            tensor = tensor[split_start:split_end, ...]
            tensor = torch.moveaxis(tensor, 0, axis)
            return tensor
        for k, v in model_state_dict.items():
            if k == 'freqs_cis':
                continue
            if k == 'model.norm.weight' or k.endswith('_layernorm.weight') or k.endswith('weight_scaler'):
                pass
            elif k == 'embedder.weight' or re.fullmatch('model.layers.\\d+.mlp.down_proj.weight', k):
                v = split(v, 1)
            elif re.fullmatch('model.layers.\\d+.mlp.gate_proj.weight', k) or re.fullmatch('model.layers.\\d+.mlp.up_proj.weight', k):
                v = split(v, 0)
            elif re.fullmatch('model.layers.\\d+.self_attn.qkv_proj.weight', k):
                if num_kv_heads <= num_attn_heads:
                    num_replicas = max(self.world_size // num_kv_heads, 1)
                    v = v.reshape(num_attn_heads + num_kv_heads * 2, head_dim, hidden_size)
                    query = v[:num_attn_heads, ...]
                    key = v[num_attn_heads:num_attn_heads + num_kv_heads, ...].repeat(num_replicas, 1, 1)
                    value = v[-num_kv_heads:, ...].repeat(num_replicas, 1, 1)
                    v = torch.cat((split(query, 0), split(key, 0), split(value, 0)), dim=0)
                else:
                    v = v.reshape(3, num_attn_heads, head_dim, hidden_size)
                    v = split(v, 1)
                v = v.reshape(-1, hidden_size)
            elif re.fullmatch('model.layers.\\d+.self_attn.o_proj.weight', k):
                v = v.reshape(hidden_size, num_attn_heads, head_dim)
                v = split(v, 1)
                v = v.reshape(hidden_size, -1)
            else:
                raise ValueError(f'Unrecognized key: {k}')
            self.state_dict()[k].copy_(v)

    def load_weights(self, model_path: 'str'):
        if os.path.isfile(model_path):
            checkpoint = torch.load(model_path, weights_only=True)
            model_state_dict = checkpoint['model_state_dict']
            self._load_weights(model_state_dict)
        else:
            index_path = os.path.join(model_path, 'pytorch_model.bin.index.json')
            with open(index_path, 'r', encoding='utf-8') as f:
                index = json.load(f)
            shard_files = list(set(index['weight_map'].values()))
            for shard_file in shard_files:
                shard_path = os.path.join(model_path, shard_file)
                state_dict = torch.load(shard_path, map_location='cpu', weights_only=True)
                self._load_weights(state_dict)
                del state_dict
                gc.collect()


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (Linear,
     lambda: ([], {'in_features': 4, 'out_features': 4, 'quant': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (RMSNorm,
     lambda: ([], {'dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
]

