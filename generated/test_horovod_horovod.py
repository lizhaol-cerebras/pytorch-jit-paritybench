
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


import re


from typing import List


from typing import Dict


from typing import Set


import torch


import random


import numpy as np


import torch.backends.cudnn as cudnn


import torch.multiprocessing as mp


import torch.nn.functional as F


import torch.optim as optim


import torch.utils.data.distributed


from torch.utils.tensorboard import SummaryWriter


from torchvision import datasets


from torchvision import transforms


from torchvision import models


import math


import torch.nn as nn


import warnings


import logging


import copy


import numbers


import time


import torch.utils.data


import collections


from collections.abc import Iterable


from torch.autograd.function import Function


from torch.nn.modules.batchnorm import _BatchNorm


import uuid


import itertools


from torch.nn import functional as F


import inspect


from torch.utils.data import Dataset


from torch.utils.data import DataLoader


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)


version = __version__


class HorovodInternalError(RuntimeError):
    """Internal error raised when a Horovod collective operation (e.g., allreduce) fails.

    This is handled in elastic mode as a recoverable error, and will result in a reset event.
    """
    pass


_NULL = ''


def _allgather_function_factory(tensor):
    return 'horovod_torch_allgather_async_' + tensor.type().replace('.', '_')


def _check_function(function_factory, tensor):
    function = function_factory(tensor)
    if not hasattr(mpi_lib, function):
        raise ValueError('Tensor type %s is not supported.' % tensor.type())
    if not tensor.is_contiguous():
        raise ValueError('Tensor is required to be contiguous.')
    return function


_handle_map = {}


def _allgather_async(tensor, output, name, process_set: 'ProcessSet'):
    function = _check_function(_allgather_function_factory, tensor)
    try:
        handle = getattr(mpi_lib, function)(tensor, output, name.encode() if name is not None else _NULL, process_set.process_set_id)
    except RuntimeError as e:
        raise HorovodInternalError(e)
    _handle_map[handle] = tensor, output
    return handle


class MPI:


    class Comm:
        ...


def is_iterable(x):
    try:
        _ = iter(x)
    except TypeError:
        return False
    return True


def _allreduce_function_factory(tensor):
    return 'horovod_torch_allreduce_async_' + tensor.type().replace('.', '_')


def _cache(f):
    cache = dict()

    def wrapper(*args, **kwargs):
        key = args, frozenset(kwargs.items())
        if key in cache:
            return cache[key]
        else:
            retval = f(*args, **kwargs)
            cache[key] = retval
            return retval
    return wrapper


def _check_extension_lambda(ext_base_name, fn, fn_desc, verbose):
    """
    Tries to load the extension in a new process.  If successful, puts fn(ext)
    to the queue or False otherwise.  Mutes all stdout/stderr.
    """

    def _target_fn(ext_base_name, fn, fn_desc, queue, verbose):
        if verbose:
            None
        else:
            sys.stdout = open(os.devnull, 'w')
            sys.stderr = open(os.devnull, 'w')
        try:
            ext = importlib.import_module('.' + ext_base_name, 'horovod')
            result = fn(ext)
        except:
            traceback.print_exc()
            result = None
        if verbose:
            None
        queue.put(result)
    ctx = multiprocessing.get_context('fork')
    queue = ctx.Queue()
    p = ctx.Process(target=_target_fn, args=(ext_base_name, fn, fn_desc, queue, verbose))
    p.daemon = True
    p.start()
    p.join()
    return queue.get_nowait()


@_cache
def gpu_available(ext_base_name, verbose=False):
    available_fn = lambda ext: ext._check_has_gpu()
    return _check_extension_lambda(ext_base_name, available_fn, 'running with GPU', verbose) or False


def num_rank_is_power_2(num_rank):
    """
    Tests if the given number of ranks is of power of 2. This check is required
    for Adasum allreduce.
    TODO support non-power of 2 ranks.
    """
    return num_rank != 0 and num_rank & num_rank - 1 == 0


def _allreduce_async(tensor, output, name, op, prescale_factor, postscale_factor, process_set: 'ProcessSet'):
    if op == Average:
        if rocm_built():
            divisor = process_set.size()
            op = Sum
        else:
            divisor = 1
    elif op == Adasum:
        if process_set != global_process_set:
            raise NotImplementedError('Adasum does not support non-global process sets yet.')
        if tensor.device.type != 'cpu' and gpu_available('torch'):
            if nccl_built():
                if not is_homogeneous():
                    raise NotImplementedError('Running GPU Adasum on heterogeneous cluster is not supported yet.')
                elif not num_rank_is_power_2(int(size() / local_size())):
                    raise NotImplementedError('Running GPU Adasum with non-power of 2 nodes is not supported yet.')
                if rocm_built():
                    divisor = local_size()
                else:
                    divisor = 1
            else:
                warnings.warn('Adasum reduction does not currently support GPU reduction using MPI. Tensors are copied to CPU memory instead. To use Adasum for GPU reduction, please compile Horovod with HOROVOD_GPU_OPERATIONS=NCCL.')
                divisor = 1
        else:
            if not num_rank_is_power_2(size()):
                raise NotImplementedError('Running Adasum with non-power of 2 ranks is not supported yet.')
            divisor = 1
    else:
        divisor = 1
    function = _check_function(_allreduce_function_factory, tensor)
    try:
        handle = getattr(mpi_lib, function)(tensor, output, divisor, name.encode() if name is not None else _NULL, op, prescale_factor, postscale_factor, process_set.process_set_id)
    except RuntimeError as e:
        raise HorovodInternalError(e)
    _handle_map[handle] = tensor, output
    return handle


def get_average_backwards_compatibility_fun(reduce_ops):
    """
    Handle backwards compatibility between the old average and the new op parameters.
    Old code using the average parameter (e.g. hvd.allreduce(tensor, average=False))
    gets unchanged behavior, but mixing old and new is disallowed (e.g. no
    hvd.allreduce(tensor, average=False, op=hvd.Adasum)).
    """

    def impl(op, average):
        if op is not None:
            if average is not None:
                raise ValueError('The op parameter supersedes average. Please provide only one of them.')
            return op
        elif average is not None:
            warnings.warn('Parameter `average` has been replaced with `op` and will be removed in v1.0', DeprecationWarning)
            return reduce_ops.Average if average else reduce_ops.Sum
        else:
            return reduce_ops.Average
    return impl


def synchronize(handle):
    """
    Synchronizes an asynchronous allreduce, allgather, alltoall, broadcast, or reducescatter operation
    until  it's completed. Returns the result of the operation.

    Arguments:
        handle: A handle returned by an allreduce, allgather, alltoall, broadcast, or reducescatter
                asynchronous operation.

    Returns:
        A single output tensor of the operation or a tuple of multiple output tensors.
    """
    if handle not in _handle_map:
        return
    try:
        mpi_lib.horovod_torch_wait_and_clear(handle)
        output = _handle_map.pop(handle)[-1]
        return output
    except RuntimeError as e:
        _handle_map.pop(handle, None)
        raise HorovodInternalError(e)


class _SyncBatchNorm(Function):

    @staticmethod
    def forward(self, input, weight, bias, running_mean, running_var, eps, momentum):
        input = input.contiguous()
        size = input.numel() // input.size(1)
        count = torch.tensor([size])
        mean, invstd = torch.batch_norm_stats(input, eps)
        count_handle = allgather_async(count.unsqueeze(0), name='sync_batch_norm.count')
        mean_handle = allgather_async(mean.unsqueeze(0), name='sync_batch_norm.mean')
        invstd_handle = allgather_async(invstd.unsqueeze(0), name='sync_batch_norm.invstd')
        count_all = synchronize(count_handle)
        mean_all = synchronize(mean_handle)
        invstd_all = synchronize(invstd_handle)
        if _SYNC_BN_V3:
            counts_for_bngswc = count_all.view(-1).float()
        else:
            counts_for_bngswc = count_all.view(-1).tolist()
        mean, invstd = torch.batch_norm_gather_stats_with_counts(input, mean_all, invstd_all, running_mean, running_var, momentum, eps, counts_for_bngswc)
        self.save_for_backward(input, weight, mean, invstd, count_all)
        return torch.batch_norm_elemt(input, weight, bias, mean, invstd, eps)

    @staticmethod
    def backward(self, grad_output):
        grad_output = grad_output.contiguous()
        saved_input, weight, mean, invstd, count_all = self.saved_tensors
        need_input_grad, need_weight_grad, need_bias_grad = self.needs_input_grad[0:3]
        sum_dy, sum_dy_xmu, grad_weight, grad_bias = torch.batch_norm_backward_reduce(grad_output, saved_input, mean, invstd, weight, need_input_grad, need_weight_grad, need_bias_grad)
        if need_input_grad:
            sum_dy_handle = allreduce_async(sum_dy, op=Sum, name='sync_batch_norm.sum_dy')
            sum_dy_xmu_handle = allreduce_async(sum_dy_xmu, op=Sum, name='sync_batch_norm.sum_dy_xmu')
            sum_dy = synchronize(sum_dy_handle)
            sum_dy_xmu = synchronize(sum_dy_xmu_handle)
            if _SYNC_BN_V4:
                count_all = count_all
            elif _SYNC_BN_V2 or _SYNC_BN_V3:
                count = count_all.sum()
            else:
                count = size()
            if _SYNC_BN_V4:
                grad_input = torch.batch_norm_backward_elemt(grad_output, saved_input, mean, invstd, weight, sum_dy, sum_dy_xmu, count_all)
            else:
                grad_input = torch.batch_norm_backward_elemt(grad_output, saved_input, mean, invstd, weight, sum_dy / count, sum_dy_xmu / count)
        else:
            grad_input = None
        if weight is None or not need_weight_grad:
            grad_weight = None
        if weight is None or not need_bias_grad:
            grad_bias = None
        return grad_input, grad_weight, grad_bias, None, None, None, None, None, None


class SyncBatchNorm(_BatchNorm):
    """Applies synchronous version of N-dimensional BatchNorm.

    In this version, normalization parameters are synchronized across workers during forward pass.
    This is very useful in situations where each GPU can fit a very small number of examples.

    See https://pytorch.org/docs/stable/nn.html#batchnorm2d for more details about BatchNorm.

    Arguments:
        num_features: number of channels `C` from the shape `(N, C, ...)`
        eps: a value added to the denominator for numerical stability. Default: 1e-5
        momentum: the value used for the running_mean and running_var
            computation. Can be set to `None` for cumulative moving average
            (i.e. simple average). Default: 0.1
        affine: a boolean value that when set to `True`, this module has
            learnable affine parameters. Default: `True`
        track_running_stats: a boolean value that when set to `True`, this
            module tracks the running mean and variance, and when set to `False`,
            this module does not track such statistics and always uses batch
            statistics in both training and eval modes. Default: `True`
    
    .. note:: Only GPU input tensors are supported in the training mode.
    """

    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True):
        super().__init__(num_features, eps, momentum, affine, track_running_stats)

    def _check_input_dim(self, input):
        if input.dim() < 2:
            raise ValueError('expected at least 2D input (got {}D input)'.format(input.dim()))

    def _run_bn(self, input):
        return F.batch_norm(input, self.running_mean, self.running_var, self.weight, self.bias, self.training or not self.track_running_stats, self.momentum, self.eps)

    @torch.jit.unused
    def _maybe_run_sync_bn(self, input):
        if size() == 1:
            return self._run_bn(input)
        return _SyncBatchNorm.apply(input, self.weight, self.bias, self.running_mean, self.running_var, self.eps, self.momentum)

    def forward(self, input):
        if not input.is_cuda:
            raise ValueError('SyncBatchNorm expected input tensor to be on GPU')
        self._check_input_dim(input)
        if self.training and self.track_running_stats:
            self.num_batches_tracked = self.num_batches_tracked + 1
        if not self.training and self.track_running_stats:
            return self._run_bn(input)
        else:
            return self._maybe_run_sync_bn(input)


class LegacyXOR(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(LegacyXOR, self).__init__()
        self.lin1 = nn.Linear(input_dim, 8)
        self.lin2 = nn.Linear(8, output_dim)

    def forward(self, features):
        x = features.float()
        x = self.lin1(x)
        x = torch.tanh(x)
        x = self.lin2(x)
        x = torch.sigmoid(x)
        return x


class XOR(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(XOR, self).__init__()
        self.lin1 = nn.Linear(input_dim, 8)
        self.lin2 = nn.Linear(8, output_dim)

    def forward(self, features):
        x = features.float()
        x = self.lin1(x)
        x = torch.tanh(x)
        x = self.lin2(x)
        x = torch.sigmoid(x)
        return x


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (LegacyXOR,
     lambda: ([], {'input_dim': 4, 'output_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (XOR,
     lambda: ([], {'input_dim': 4, 'output_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
]

