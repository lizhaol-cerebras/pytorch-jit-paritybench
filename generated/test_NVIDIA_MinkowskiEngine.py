
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


from typing import Union


import torch


from torch.nn import Module


from torch.autograd import Function


import math


from torch.nn import Parameter


from collections.abc import Sequence


import numpy as np


from typing import List


from typing import Tuple


import warnings


import torch.nn.functional as F


from collections import namedtuple


from functools import reduce


from abc import ABC


from abc import abstractmethod


import torch.nn as nn


from torch.nn.modules import Module


import copy


from enum import Enum


import logging


import collections.abc


from torch.types import _TensorOrTensors


from typing import Callable


from typing import Optional


from torch.autograd.gradcheck import gradcheck as _gradcheck


from torch.autograd import Variable


from collections import OrderedDict


import sklearn.metrics as metrics


import torch.utils.data


from torch.utils.data import DataLoader


import torch.optim as optim


import random


import time


from torch.utils.data.sampler import Sampler


from time import time


from torch.optim import SGD


import torch.nn.parallel as parallel


import torch.multiprocessing as mp


import torch.distributed as dist


from torch.utils.data import Dataset


import re


from torch.utils.cpp_extension import BuildExtension


from torch.utils.cpp_extension import CppExtension


from torch.utils.cpp_extension import CUDAExtension


import torch.cuda


from torch.utils.cpp_extension import CUDA_HOME


from torch.utils.cpp_extension import ROCM_HOME


import collections


def convert_to_int_list(arg: 'Union[int, Sequence, np.ndarray, torch.Tensor]', dimension: 'int'):
    if isinstance(arg, list):
        assert len(arg) == dimension
        return arg
    if isinstance(arg, (Sequence, np.ndarray, torch.Tensor)):
        tmp = [i for i in arg]
        assert len(tmp) == dimension
    elif np.isscalar(arg):
        tmp = [int(arg) for i in range(dimension)]
    else:
        raise ValueError('Input must be a scalar or a sequence')
    return tmp


def get_postfix(tensor: 'torch.Tensor'):
    postfix = 'GPU' if tensor.is_cuda else 'CPU'
    return postfix


def get_minkowski_function(name, variable):
    fn_name = name + get_postfix(variable)
    if hasattr(MEB, fn_name):
        return getattr(MEB, fn_name)
    elif variable.is_cuda:
        raise ValueError(f'Function {fn_name} not available. Please compile MinkowskiEngine with `torch.cuda.is_available()` is `True`.')
    else:
        raise ValueError(f'Function {fn_name} not available.')


class MinkowskiInterpolationFunction(Function):

    @staticmethod
    def forward(ctx, input_features: 'torch.Tensor', tfield: 'torch.Tensor', in_coordinate_map_key: 'CoordinateMapKey', coordinate_manager: 'CoordinateManager'=None):
        input_features = input_features.contiguous()
        fw_fn = get_minkowski_function('InterpolationForward', input_features)
        out_feat, in_map, out_map, weights = fw_fn(input_features, tfield, in_coordinate_map_key, coordinate_manager._manager)
        ctx.save_for_backward(in_map, out_map, weights)
        ctx.inputs = in_coordinate_map_key, coordinate_manager
        return out_feat, in_map, out_map, weights

    @staticmethod
    def backward(ctx, grad_out_feat=None, grad_in_map=None, grad_out_map=None, grad_weights=None):
        grad_out_feat = grad_out_feat.contiguous()
        bw_fn = get_minkowski_function('InterpolationBackward', grad_out_feat)
        in_coordinate_map_key, coordinate_manager = ctx.inputs
        in_map, out_map, weights = ctx.saved_tensors
        grad_in_feat = bw_fn(grad_out_feat, in_map, out_map, weights, in_coordinate_map_key, coordinate_manager._manager)
        return grad_in_feat, None, None, None


def spmm(rows: 'torch.Tensor', cols: 'torch.Tensor', vals: 'torch.Tensor', size: 'torch.Size', mat: 'torch.Tensor', is_sorted: 'bool'=False, cuda_spmm_alg: 'int'=1) ->torch.Tensor:
    assert len(rows) == len(cols), 'Invalid length'
    assert len(rows) == len(vals), 'Invalid length'
    assert vals.dtype == mat.dtype, 'dtype mismatch'
    assert vals.device == mat.device, 'device mismatch'
    if mat.is_cuda:
        rows = rows.int()
        cols = cols.int()
        result = MEB.coo_spmm_int32(rows, cols, vals, size[0], size[1], mat, cuda_spmm_alg, is_sorted)
    else:
        COO = torch.stack((rows, cols), 0).long()
        torchSparseTensor = None
        if vals.dtype == torch.float64:
            torchSparseTensor = torch.sparse.DoubleTensor
        elif vals.dtype == torch.float32:
            torchSparseTensor = torch.sparse.FloatTensor
        else:
            raise ValueError(f'Unsupported data type: {vals.dtype}')
        sp = torchSparseTensor(COO, vals, size)
        result = sp.matmul(mat)
    return result


def spmm_average(rows: 'torch.Tensor', cols: 'torch.Tensor', size: 'torch.Size', mat: 'torch.Tensor', cuda_spmm_alg: 'int'=1) ->(torch.Tensor, torch.Tensor, torch.Tensor):
    assert len(rows) == len(cols), 'Invalid length'
    if mat.is_cuda:
        rows = rows.int()
        cols = cols.int()
        result, COO, vals = MEB.coo_spmm_average_int32(rows, cols, size[0], size[1], mat, cuda_spmm_alg)
    else:
        rows, sort_ind = torch.sort(rows)
        cols = cols[sort_ind]
        COO = torch.stack((rows, cols), 0).long()
        _, inverse_ind, counts = torch.unique(rows, return_counts=True, return_inverse=True)
        vals = 1 / counts[inverse_ind]
        torchSparseTensor = None
        if mat.dtype == torch.float64:
            torchSparseTensor = torch.sparse.DoubleTensor
        elif mat.dtype == torch.float32:
            torchSparseTensor = torch.sparse.FloatTensor
        else:
            raise ValueError(f'Unsupported data type: {mat.dtype}')
        sp = torchSparseTensor(COO, vals, size)
        result = sp.matmul(mat)
    return result, COO, vals


class MinkowskiSPMMAverageFunction(Function):

    @staticmethod
    def forward(ctx, rows: 'torch.Tensor', cols: 'torch.Tensor', size: 'torch.Size', mat: 'torch.Tensor', cuda_spmm_alg: 'int'=1):
        ctx.misc_args = size, cuda_spmm_alg
        result, COO, vals = spmm_average(rows, cols, size, mat, cuda_spmm_alg=cuda_spmm_alg)
        ctx.save_for_backward(COO, vals)
        return result

    @staticmethod
    def backward(ctx, grad: 'torch.Tensor'):
        size, cuda_spmm_alg = ctx.misc_args
        COO, vals = ctx.saved_tensors
        new_size = torch.Size([size[1], size[0]])
        grad = spmm(COO[1], COO[0], vals, new_size, grad, is_sorted=False, cuda_spmm_alg=cuda_spmm_alg)
        return None, None, None, grad, None


class MinkowskiSPMMFunction(Function):

    @staticmethod
    def forward(ctx, rows: 'torch.Tensor', cols: 'torch.Tensor', vals: 'torch.Tensor', size: 'torch.Size', mat: 'torch.Tensor', cuda_spmm_alg: 'int'=1):
        ctx.misc_args = size, cuda_spmm_alg
        ctx.save_for_backward(rows, cols, vals)
        result = spmm(rows, cols, vals, size, mat, is_sorted=False, cuda_spmm_alg=cuda_spmm_alg)
        return result

    @staticmethod
    def backward(ctx, grad: 'torch.Tensor'):
        size, cuda_spmm_alg = ctx.misc_args
        rows, cols, vals = ctx.saved_tensors
        new_size = torch.Size([size[1], size[0]])
        grad = spmm(cols, rows, vals, new_size, grad, is_sorted=False, cuda_spmm_alg=cuda_spmm_alg)
        return None, None, None, None, grad, None


class SparseTensorOperationMode(Enum):
    """Enum class for SparseTensor internal instantiation modes.

    :attr:`SEPARATE_COORDINATE_MANAGER`: always create a new coordinate manager.

    :attr:`SHARE_COORDINATE_MANAGER`: always use the globally defined coordinate
    manager. Must clear the coordinate manager manually by
    :attr:`MinkowskiEngine.SparseTensor.clear_global_coordinate_manager`.

    """
    SEPARATE_COORDINATE_MANAGER = 0
    SHARE_COORDINATE_MANAGER = 1


class SparseTensorQuantizationMode(Enum):
    """
    `RANDOM_SUBSAMPLE`: Subsample one coordinate per each quantization block randomly.
    `UNWEIGHTED_AVERAGE`: average all features within a quantization block equally.
    `UNWEIGHTED_SUM`: sum all features within a quantization block equally.
    `NO_QUANTIZATION`: No quantization is applied. Should not be used for normal operation.
    `MAX_POOL`: Voxel-wise max pooling is applied.
    `SPLAT_LINEAR_INTERPOLATION`: Splat features using N-dimensional linear interpolation to 2^N neighbors.
    """
    RANDOM_SUBSAMPLE = 0
    UNWEIGHTED_AVERAGE = 1
    UNWEIGHTED_SUM = 2
    NO_QUANTIZATION = 3
    MAX_POOL = 4
    SPLAT_LINEAR_INTERPOLATION = 5


COORDINATE_KEY_DIFFERENT_ERROR = 'SparseTensors must have the same coordinate_map_key.'


COORDINATE_MANAGER_DIFFERENT_ERROR = 'SparseTensors must share the same coordinate manager for this operation. Please refer to the SparseTensor creation API (https://nvidia.github.io/MinkowskiEngine/sparse_tensor.html) to share the coordinate manager, or set the sparse tensor operation mode with `set_sparse_tensor_operation_mode` to share it by default.'


class MinkowskiDirectMaxPoolingFunction(Function):

    @staticmethod
    def forward(ctx, in_map: 'torch.Tensor', out_map: 'torch.Tensor', in_feat: 'torch.Tensor', out_nrows: 'int', is_sorted: 'bool'=False):
        out_feat, max_mask = _C.direct_max_pool_fw(in_map, out_map, in_feat, out_nrows, is_sorted)
        ctx.in_nrows = in_feat.size(0)
        ctx.save_for_backward(max_mask)
        return out_feat

    @staticmethod
    def backward(ctx, grad_out_feat):
        grad_out_feat = grad_out_feat.contiguous()
        max_mask = ctx.saved_tensors[0]
        grad = _C.direct_max_pool_bw(grad_out_feat, max_mask, ctx.in_nrows)
        return None, None, grad, None, None

