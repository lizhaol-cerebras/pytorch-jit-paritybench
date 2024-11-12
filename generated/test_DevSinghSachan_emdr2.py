
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


import torch


import random


import numpy as np


from torch.nn.parallel import DistributedDataParallel as torchDDP


from functools import lru_cache


from itertools import accumulate


import time


from abc import ABC


from torch.utils.data import Dataset


from torch.utils import data


from torch import nn


from torch.autograd import Variable


from torch.nn.parameter import Parameter


from torch._utils import _flatten_dense_tensors


from torch._utils import _unflatten_dense_tensors


import torch.nn as nn


from torch.utils import cpp_extension


import torch.distributed as dist


from torch.nn.modules import Module


import warnings


import math


import torch.nn.functional as F


import collections


import torch.nn.init as init


from torch import _C


from torch.cuda import _lazy_call


from torch.cuda import device as device_ctx_manager


from torch.utils.checkpoint import detach_variable


from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP


from collections import OrderedDict


from torch.utils.data import DataLoader


from torch.utils.data import BatchSampler


from abc import abstractmethod


from copy import deepcopy


class tofp16(nn.Module):
    """
    Utility module that implements::

        def forward(self, input):
            return input.half()
    """

    def __init__(self):
        super(tofp16, self).__init__()

    def forward(self, input):
        return input.half()


def convert_module(module, dtype):
    """
    Converts a module's immediate parameters and buffers to dtype.
    """
    for param in module.parameters(recurse=False):
        if param is not None:
            if param.data.dtype.is_floating_point:
                param.data = param.data
            if param._grad is not None and param._grad.data.dtype.is_floating_point:
                param._grad.data = param._grad.data
    for buf in module.buffers(recurse=False):
        if buf is not None and buf.data.dtype.is_floating_point:
            buf.data = buf.data


def convert_network(network, dtype):
    """
    Converts a network's parameters and buffers to dtype.
    """
    for module in network.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm) and module.affine is True:
            continue
        convert_module(module, dtype)
    return network


class FP16Model(nn.Module):
    """
    Convert model to half precision in a batchnorm-safe way.
    """

    def __init__(self, network):
        super(FP16Model, self).__init__()
        self.network = convert_network(network, dtype=torch.half)

    def forward(self, *inputs):
        inputs = tuple(t.half() for t in inputs)
        return self.network(*inputs)


class ScaledMaskedSoftmax(torch.autograd.Function):
    """
       Fused operation which performs following three operations in sequence
       1. Scale the tensor. 
       2. Apply the mask.
       3. Perform softmax.
    """

    @staticmethod
    def forward(ctx, inputs, mask, scale):
        scale_t = torch.tensor([scale])
        softmax_results = scaled_masked_softmax_cuda.forward(inputs, mask, scale_t[0])
        ctx.save_for_backward(softmax_results, scale_t)
        return softmax_results

    @staticmethod
    def backward(ctx, output_grads):
        softmax_results, scale_t = ctx.saved_tensors
        input_grads = scaled_masked_softmax_cuda.backward(output_grads, softmax_results, scale_t[0])
        return input_grads, None, None


class ScaledUpperTriangMaskedSoftmax(torch.autograd.Function):
    """
       Fused operation which performs following three operations in sequence
       1. Scale the tensor. 
       2. Apply upper triangular mask (typically used in gpt models).
       3. Perform softmax.
    """

    @staticmethod
    def forward(ctx, inputs, scale):
        scale_t = torch.tensor([scale])
        softmax_results = scaled_upper_triang_masked_softmax_cuda.forward(inputs, scale_t[0])
        ctx.save_for_backward(softmax_results, scale_t)
        return softmax_results

    @staticmethod
    def backward(ctx, output_grads):
        softmax_results, scale_t = ctx.saved_tensors
        input_grads = scaled_upper_triang_masked_softmax_cuda.backward(output_grads, softmax_results, scale_t[0])
        return input_grads, None


class FusedScaleMaskSoftmax(torch.nn.Module):
    """
       fused operation: scaling + mask + softmax
       Arguments:
           input_in_fp16: flag to indicate if input in fp16 data format.
           upper_triang_mask: if true, apply upper triangular masking.
                              (used in gpt family networks)
           mask_func: mask function to be applied.
           softmax_in_fp32: if true, softmax in performed at fp32 precision.
           scale: scaling factor used in input tensor scaling.

    """

    def __init__(self, input_in_fp16, upper_triang_mask_fusion, general_mask_fusion, mask_func, softmax_in_fp32, scale):
        super(FusedScaleMaskSoftmax, self).__init__()
        self.input_in_fp16 = input_in_fp16
        self.upper_triang_mask_fusion = upper_triang_mask_fusion
        self.general_mask_fusion = general_mask_fusion
        self.mask_func = mask_func
        self.softmax_in_fp32 = softmax_in_fp32
        self.scale = scale
        assert self.scale is None or softmax_in_fp32, 'softmax should be in fp32 when scaled'

    def forward(self, input, mask):
        data_size = input.size()
        assert input.dim() == 4
        if self.input_in_fp16 and data_size[-1] <= 2048 and (self.upper_triang_mask_fusion or self.general_mask_fusion) and input.size()[2] == input.size()[3]:
            scale = self.scale if self.scale is not None else 1.0
            if self.upper_triang_mask_fusion:
                input = input.view(-1, data_size[2], data_size[3])
                probs = ScaledUpperTriangMaskedSoftmax.apply(input, scale)
                probs = probs.view(*data_size)
            else:
                probs = ScaledMaskedSoftmax.apply(input, mask, scale)
        else:
            if self.input_in_fp16 and self.softmax_in_fp32:
                input = input.float()
            if self.scale is not None:
                input = input * self.scale
            mask_output = self.mask_func(input, mask)
            probs = torch.nn.Softmax(dim=-1)(mask_output)
            if self.input_in_fp16 and self.softmax_in_fp32:
                probs = probs.half()
        return probs


class MegatronModule(torch.nn.Module):
    """Megatron specific extentions of torch Module."""

    def __init__(self):
        super(MegatronModule, self).__init__()

    def state_dict_for_save_checkpoint(self, destination=None, prefix='', keep_vars=False):
        """Use this function to override the state dict for
        saving checkpoints."""
        return self.state_dict(destination, prefix, keep_vars)


def ensure_divisibility(numerator, denominator):
    """Ensure that numerator is divisible by the denominator."""
    assert numerator % denominator == 0, '{} is not divisible by {}'.format(numerator, denominator)


def divide(numerator, denominator):
    """Ensure that numerator is divisible by the denominator and return
    the division value."""
    ensure_divisibility(numerator, denominator)
    return numerator // denominator


class VocabUtility:
    """Split the vocabulary into `world_size` chunks amd return the
        first and last index of the vocabulary belonging to the `rank`
        partition: Note that indecies in [fist, last)"""

    @staticmethod
    def vocab_range_from_per_partition_vocab_size(per_partition_vocab_size, rank, world_size):
        index_f = rank * per_partition_vocab_size
        index_l = index_f + per_partition_vocab_size
        return index_f, index_l

    @staticmethod
    def vocab_range_from_global_vocab_size(global_vocab_size, rank, world_size):
        per_partition_vocab_size = divide(global_vocab_size, world_size)
        return VocabUtility.vocab_range_from_per_partition_vocab_size(per_partition_vocab_size, rank, world_size)


def get_args():
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group(title='save separate EMDR2 models')
    group.add_argument('--load', type=str, required=True, help='EMDR2 model path')
    group.add_argument('--save', type=str, required=True, help='path to save model')
    group.add_argument('--submodel-name', type=str, required=True, choices=['encoder', 'retriever'], help='sub model of EMDR2 to save')
    args = parser.parse_args()
    return args


_MODEL_PARALLEL_GROUP = None


def get_model_parallel_group():
    """Get the model parallel group the caller rank belongs to."""
    assert _MODEL_PARALLEL_GROUP is not None, 'model parallel group is not initialized'
    return _MODEL_PARALLEL_GROUP


def get_model_parallel_rank():
    """Return my rank for the model parallel group."""
    global _MPU_RANK
    if _MPU_RANK is not None:
        return _MPU_RANK
    return torch.distributed.get_rank(group=get_model_parallel_group())


def get_model_parallel_world_size():
    """Return world size for the model parallel group."""
    global _MPU_WORLD_SIZE
    if _MPU_WORLD_SIZE is not None:
        return _MPU_WORLD_SIZE
    return torch.distributed.get_world_size(group=get_model_parallel_group())


def _initialize_affine_weight_cpu(weight, output_size, input_size, per_partition_size, partition_dim, init_method, stride=1, return_master_weight=False):
    """Initialize affine weight for model parallel.

    Build the master weight on all processes and scatter
    the relevant chunk."""
    weight.model_parallel = True
    weight.partition_dim = partition_dim
    weight.partition_stride = stride
    master_weight = torch.empty(output_size, input_size, dtype=torch.float, requires_grad=False)
    init_method(master_weight)
    args = get_args()
    master_weight = master_weight
    per_partition_per_stride_size = divide(per_partition_size, stride)
    weight_list = torch.split(master_weight, per_partition_per_stride_size, dim=partition_dim)
    rank = get_model_parallel_rank()
    world_size = get_model_parallel_world_size()
    my_weight_list = weight_list[rank::world_size]
    with torch.no_grad():
        torch.cat(my_weight_list, dim=partition_dim, out=weight)
    if return_master_weight:
        return master_weight
    return None


_MODEL_PARALLEL_RNG_TRACKER_NAME = 'model-parallel-rng'


def _set_cuda_rng_state(new_state, device=-1):
    """Sets the random number generator state of the current GPU.

    Argumentss:
        new_state (torch.ByteTensor): The desired state
    This function is adapted from PyTorch repo (torch.cuda.set_rng_state)
    with a single change: the input state is not cloned. Cloning caused
    major performance issues for +4 GPU cases.
    """
    if hasattr(_C, '_cuda_setRNGState') and callable(_C._cuda_setRNGState):

        def cb():
            with device_ctx_manager(device):
                _C._cuda_setRNGState(new_state)
    else:
        if device == -1:
            device = torch.device('cuda')
        elif isinstance(device, str):
            device = torch.device(device)
        elif isinstance(device, int):
            device = torch.device('cuda', device)

        def cb():
            idx = device.index
            if idx is None:
                idx = torch.cuda.current_device()
            default_generator = torch.cuda.default_generators[idx]
            default_generator.set_state(new_state)
    _lazy_call(cb)


def get_cuda_rng_tracker():
    """Get cuda rng tracker."""
    return _CUDA_RNG_STATE_TRACKER


def _initialize_affine_weight_gpu(weight, init_method, partition_dim, stride=1):
    """Initialize affine weight for model parallel on GPU."""
    weight.model_parallel = True
    weight.partition_dim = partition_dim
    weight.partition_stride = stride
    with get_cuda_rng_tracker().fork():
        init_method(weight)


def _reduce(input_):
    """All-reduce the the input tensor across model parallel group."""
    if get_model_parallel_world_size() == 1:
        return input_
    torch.distributed.all_reduce(input_, group=get_model_parallel_group())
    return input_


class _ReduceFromModelParallelRegion(torch.autograd.Function):
    """All-redcue the input from the model parallel region."""

    @staticmethod
    def symbolic(graph, input_):
        return _reduce(input_)

    @staticmethod
    def forward(ctx, input_):
        return _reduce(input_)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


def reduce_from_model_parallel_region(input_):
    return _ReduceFromModelParallelRegion.apply(input_)


class VocabParallelEmbedding(torch.nn.Module):
    """Embedding parallelized in the vocabulary dimension.

    This is mainly adapted from torch.nn.Embedding and all the default
    values are kept.
    Arguments:
        num_embeddings: vocabulary size.
        embedding_dim: size of hidden state.
        init_method: method to initialize weights.
    """

    def __init__(self, num_embeddings, embedding_dim, init_method=init.xavier_normal_):
        super(VocabParallelEmbedding, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = None
        self.max_norm = None
        self.norm_type = 2.0
        self.scale_grad_by_freq = False
        self.sparse = False
        self._weight = None
        self.model_parallel_size = get_model_parallel_world_size()
        self.vocab_start_index, self.vocab_end_index = VocabUtility.vocab_range_from_global_vocab_size(self.num_embeddings, get_model_parallel_rank(), self.model_parallel_size)
        self.num_embeddings_per_partition = self.vocab_end_index - self.vocab_start_index
        args = get_args()
        if args.use_cpu_initialization:
            self.weight = Parameter(torch.empty(self.num_embeddings_per_partition, self.embedding_dim, dtype=args.params_dtype))
            _initialize_affine_weight_cpu(self.weight, self.num_embeddings, self.embedding_dim, self.num_embeddings_per_partition, 0, init_method)
        else:
            self.weight = Parameter(torch.empty(self.num_embeddings_per_partition, self.embedding_dim, device=torch.cuda.current_device(), dtype=args.params_dtype))
            _initialize_affine_weight_gpu(self.weight, init_method, partition_dim=0, stride=1)

    def forward(self, input_):
        if self.model_parallel_size > 1:
            input_mask = (input_ < self.vocab_start_index) | (input_ >= self.vocab_end_index)
            masked_input = input_.clone() - self.vocab_start_index
            masked_input[input_mask] = 0
        else:
            masked_input = input_
        output_parallel = F.embedding(masked_input, self.weight, self.padding_idx, self.max_norm, self.norm_type, self.scale_grad_by_freq, self.sparse)
        if self.model_parallel_size > 1:
            output_parallel[input_mask, :] = 0.0
        output = reduce_from_model_parallel_region(output_parallel)
        return output


class _CopyToModelParallelRegion(torch.autograd.Function):
    """Pass the input to the model parallel region."""

    @staticmethod
    def symbolic(graph, input_):
        return input_

    @staticmethod
    def forward(ctx, input_):
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        return _reduce(grad_output)


def copy_to_model_parallel_region(input_):
    return _CopyToModelParallelRegion.apply(input_)


def _gather(input_):
    """Gather tensors and concatinate along the last dimension."""
    world_size = get_model_parallel_world_size()
    if world_size == 1:
        return input_
    last_dim = input_.dim() - 1
    rank = get_model_parallel_rank()
    tensor_list = [torch.empty_like(input_) for _ in range(world_size)]
    tensor_list[rank] = input_
    torch.distributed.all_gather(tensor_list, input_, group=get_model_parallel_group())
    output = torch.cat(tensor_list, dim=last_dim).contiguous()
    return output


def split_tensor_along_last_dim(tensor, num_partitions, contiguous_split_chunks=False):
    """Split a tensor along its last dimension.
    Arguments:
        tensor: input tensor.
        num_partitions: number of partitions to split the tensor
        contiguous_split_chunks: If True, make each chunk contiguous
                                 in memory.
    """
    last_dim = tensor.dim() - 1
    last_dim_size = divide(tensor.size()[last_dim], num_partitions)
    tensor_list = torch.split(tensor, last_dim_size, dim=last_dim)
    if contiguous_split_chunks:
        return tuple(chunk.contiguous() for chunk in tensor_list)
    return tensor_list


def _split(input_):
    """Split the tensor along its last dimension and keep the
    corresponding slice."""
    world_size = get_model_parallel_world_size()
    if world_size == 1:
        return input_
    input_list = split_tensor_along_last_dim(input_, world_size)
    rank = get_model_parallel_rank()
    output = input_list[rank].contiguous()
    return output


class _GatherFromModelParallelRegion(torch.autograd.Function):
    """Gather the input from model parallel region and concatinate."""

    @staticmethod
    def symbolic(graph, input_):
        return _gather(input_)

    @staticmethod
    def forward(ctx, input_):
        return _gather(input_)

    @staticmethod
    def backward(ctx, grad_output):
        return _split(grad_output)


def gather_from_model_parallel_region(input_):
    return _GatherFromModelParallelRegion.apply(input_)


class ColumnParallelLinear(torch.nn.Module):
    """Linear layer with column parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its second dimension as A = [A_1, ..., A_p].

    Arguments:
        input_size: first dimension of matrix A.
        output_size: second dimension of matrix A.
        bias: If true, add bias
        gather_output: If true, call all-gether on output and make Y avaiable
                       to all GPUs, otherwise, every GPU will have its output
                       which is Y_i = XA_i
        init_method: method to initialize weights. Note that bias is always set
                     to zero.
        stride: For the strided linear layers.
        keep_master_weight_for_test: This was added for testing and should be
                                     set to False. It returns the master weights
                                     used for initialization.
        skip_bias_add: This was added to enable performance optimations where bias
                       can be fused with other elementwise operations. we skip 
                       adding bias but instead return it.
    """

    def __init__(self, input_size, output_size, bias=True, gather_output=True, init_method=init.xavier_normal_, stride=1, keep_master_weight_for_test=False, skip_bias_add=False):
        super(ColumnParallelLinear, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.gather_output = gather_output
        world_size = get_model_parallel_world_size()
        self.output_size_per_partition = divide(output_size, world_size)
        self.skip_bias_add = skip_bias_add
        args = get_args()
        if args.use_cpu_initialization:
            self.weight = Parameter(torch.empty(self.output_size_per_partition, self.input_size, dtype=args.params_dtype))
            self.master_weight = _initialize_affine_weight_cpu(self.weight, self.output_size, self.input_size, self.output_size_per_partition, 0, init_method, stride=stride, return_master_weight=keep_master_weight_for_test)
        else:
            self.weight = Parameter(torch.empty(self.output_size_per_partition, self.input_size, device=torch.cuda.current_device(), dtype=args.params_dtype))
            _initialize_affine_weight_gpu(self.weight, init_method, partition_dim=0, stride=stride)
        if bias:
            if args.use_cpu_initialization:
                self.bias = Parameter(torch.empty(self.output_size_per_partition, dtype=args.params_dtype))
            else:
                self.bias = Parameter(torch.empty(self.output_size_per_partition, device=torch.cuda.current_device(), dtype=args.params_dtype))
            self.bias.model_parallel = True
            self.bias.partition_dim = 0
            self.bias.stride = stride
            with torch.no_grad():
                self.bias.zero_()
        else:
            self.register_parameter('bias', None)

    def forward(self, input_):
        input_parallel = copy_to_model_parallel_region(input_)
        bias = self.bias if not self.skip_bias_add else None
        output_parallel = F.linear(input_parallel, self.weight, bias)
        if self.gather_output:
            output = gather_from_model_parallel_region(output_parallel)
        else:
            output = output_parallel
        output_bias = self.bias if self.skip_bias_add else None
        return output, output_bias


class _ScatterToModelParallelRegion(torch.autograd.Function):
    """Split the input and keep only the corresponding chuck to the rank."""

    @staticmethod
    def symbolic(graph, input_):
        return _split(input_)

    @staticmethod
    def forward(ctx, input_):
        return _split(input_)

    @staticmethod
    def backward(ctx, grad_output):
        return _gather(grad_output)


def scatter_to_model_parallel_region(input_):
    return _ScatterToModelParallelRegion.apply(input_)


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
        input_size: first dimension of matrix A.
        output_size: second dimension of matrix A.
        bias: If true, add bias. Note that bias is not parallelized.
        input_is_parallel: If true, we assume that the input is already
                           split across the GPUs and we do not split
                           again.
        init_method: method to initialize weights. Note that bias is always set
                     to zero.
        stride: For the strided linear layers.
        keep_master_weight_for_test: This was added for testing and should be
                                     set to False. It returns the master weights
                                     used for initialization.
        skip_bias_add: This was added to enable performance optimations where bias
                       can be fused with other elementwise operations. we skip 
                       adding bias but instead return it.
    """

    def __init__(self, input_size, output_size, bias=True, input_is_parallel=False, init_method=init.xavier_normal_, stride=1, keep_master_weight_for_test=False, skip_bias_add=False):
        super(RowParallelLinear, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.input_is_parallel = input_is_parallel
        world_size = get_model_parallel_world_size()
        self.input_size_per_partition = divide(input_size, world_size)
        self.skip_bias_add = skip_bias_add
        args = get_args()
        if args.use_cpu_initialization:
            self.weight = Parameter(torch.empty(self.output_size, self.input_size_per_partition, dtype=args.params_dtype))
            self.master_weight = _initialize_affine_weight_cpu(self.weight, self.output_size, self.input_size, self.input_size_per_partition, 1, init_method, stride=stride, return_master_weight=keep_master_weight_for_test)
        else:
            self.weight = Parameter(torch.empty(self.output_size, self.input_size_per_partition, device=torch.cuda.current_device(), dtype=args.params_dtype))
            _initialize_affine_weight_gpu(self.weight, init_method, partition_dim=1, stride=stride)
        if bias:
            if args.use_cpu_initialization:
                self.bias = Parameter(torch.empty(self.output_size, dtype=args.params_dtype))
            else:
                self.bias = Parameter(torch.empty(self.output_size, device=torch.cuda.current_device(), dtype=args.params_dtype))
            with torch.no_grad():
                self.bias.zero_()
        else:
            self.register_parameter('bias', None)

    def forward(self, input_):
        if self.input_is_parallel:
            input_parallel = input_
        else:
            input_parallel = scatter_to_model_parallel_region(input_)
        output_parallel = F.linear(input_parallel, self.weight)
        output_ = reduce_from_model_parallel_region(output_parallel)
        if not self.skip_bias_add:
            output = output_ + self.bias if self.bias is not None else output_
            output_bias = None
        else:
            output = output_
            output_bias = self.bias
        return output, output_bias


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (tofp16,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
]

