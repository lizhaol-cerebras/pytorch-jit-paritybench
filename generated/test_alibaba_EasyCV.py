
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


import random


import time


import numpy as np


import torch


import torch.nn as nn


import copy


import logging


from collections import OrderedDict


from typing import Callable


from typing import Dict


from typing import List


from typing import Optional


from typing import Tuple


import torchvision.transforms.functional as t_f


import itertools


import torch.distributed as dist


import re


from torch import optim


from torch.nn.modules.utils import _pair


import math


from torch import Tensor


from abc import ABCMeta


from abc import abstractmethod


import warnings


from enum import IntEnum


from enum import unique


from logging import warning


from sklearn.metrics import confusion_matrix


from collections import defaultdict


import matplotlib.pyplot as plt


from sklearn.decomposition import PCA


from torch.optim import *


from torch.optim import AdamW as _AdamW


from torch.optim import Optimizer


from torch.optim.optimizer import Optimizer


from torch.optim.optimizer import required


import functools


from inspect import getfullargspec


from functools import partial


import collections


from numpy import random


import torch.utils.data as data


from torch.utils.data import DataLoader


from torch.utils.data import DistributedSampler as _DistributedSampler


from torch.utils.data import RandomSampler


from torch.utils.data import Sampler


from torchvision import transforms as _transforms


from torch.utils.data import Dataset


from torch.utils.data.dataset import ConcatDataset as _ConcatDataset


from collections.abc import Sequence


import inspect


from enum import EnumMeta


from torch.nn.modules.batchnorm import _BatchNorm


from torch import nn


from torch import distributed as dist


import torch.nn.functional as F


import uuid


from collections import namedtuple


import torch.utils.checkpoint as cp


from torch.nn import Conv2d


from torch.nn import Module


from torch.nn import ReLU


import torch.utils.checkpoint as checkpoint


from functools import lru_cache


from functools import reduce


from math import sqrt


from inspect import signature


from torch.nn.functional import dropout


from torch.nn.functional import linear


from torch.nn.functional import pad


from torch.nn.functional import softmax


from torch.nn.init import constant_


from torch.nn.modules.linear import Linear


from torch.nn.modules.module import Module


from torchvision.ops.boxes import box_area


from torchvision.ops.boxes import nms


import torchvision


from torch.autograd import Function


from torch.nn.init import normal_


from torchvision.transforms.functional import rotate


from torch.nn import functional as F


from scipy.optimize import linear_sum_assignment


import string


from typing import Union


from torch.cuda.amp import autocast


from typing import Sequence


from torch.hub import load_state_dict_from_url


from torchvision.transforms import Compose


from matplotlib.collections import PatchCollection


from matplotlib.patches import Polygon


from torch.autograd.function import once_differentiable


from torch.nn.init import xavier_uniform_


from torch.utils.cpp_extension import CUDA_HOME


from torch.utils.cpp_extension import CppExtension


from torch.utils.cpp_extension import CUDAExtension


from torch.autograd import gradcheck


from torch.nn.modules.utils import _single


from collections import deque


from torchvision.transforms import functional as F


from torch.autograd import Variable


from torchvision import transforms


import pandas as pd


from copy import deepcopy


import torch.distributed


from torch.distributed import ReduceOp


from torch.nn.modules.conv import _ConvNd


from torch.nn.modules.conv import _ConvTransposeMixin


from torch.nn.modules.pooling import _AdaptiveAvgPoolNd


from torch.nn.modules.pooling import _AdaptiveMaxPoolNd


from torch.nn.modules.pooling import _AvgPoolNd


from torch.nn.modules.pooling import _MaxPoolNd


import torch.multiprocessing as mp


from torch.optim.lr_scheduler import ExponentialLR


from torch.optim.lr_scheduler import ReduceLROnPlateau


from torch.optim.lr_scheduler import StepLR


from torch.testing._internal.common_utils import TestCase


from numpy.testing import assert_array_almost_equal


import scipy.io


from collections import Counter


from torch.backends import cudnn


class BaseError(Exception):
    """The base error class for exceptions.
  """
    code = None

    def __init__(self, message='', details=None, op=None):
        """Creates a new `OpError` indicating that a particular op failed.

      Args:
        message: The message string describing the failure.
        details: The help message that handle the error.
        op: The `ops.Operation` that failed, if known; otherwise None. During
          eager execution, this field is always `None`.
      """
        super(BaseError, self).__init__()
        self._op = op
        self._message = message
        self._details = details

    @property
    def message(self):
        """The error message that describes the error."""
        return self._message

    @property
    def details(self):
        """The help message that handle the error."""
        return self._details

    @property
    def op(self):
        """The operation that failed, if known.
      Returns:
        The `Operation` that failed, or None.
      """
        return self._op

    @property
    def error_code(self):
        """The integer error code that describes the error."""
        return hex(self.code)

    def __str__(self):
        print_str = 'ErrorCode: ' + self.error_code
        if self.op is not None:
            print_str += '\n' + 'Operation: ' + str(self.op)
        print_str += '\n' + 'Message: ' + self.message
        if self.details is not None:
            print_str += '\n' + 'Details: ' + self.details
        return print_str


RUNTIME = 5194902719927681025


class RuntimeError(BaseError):
    """Raised when the system experiences an internal error."""
    code = RUNTIME


class ModelExportWrapper(torch.nn.Module):

    def __init__(self, model, example_inputs, trace_model: 'bool'=True) ->None:
        super().__init__()
        self.model = model
        if hasattr(self.model, 'export_init'):
            self.model.export_init()
        self.example_inputs = example_inputs
        self.trace_model = trace_model
        if self.trace_model:
            try:
                self.trace_module()
            except RuntimeError:
                logging.warning('PAI-YOLOX: set model.test_conf=0.0 to avoid tensor in inference to be empty')
                model.test_conf = 0.0
                self.trace_module()

    def trace_module(self, **kwargs):
        trace_model = torch.jit.trace_module(self.model, {'forward_export': self.example_inputs}, **kwargs)
        self.model = trace_model

    def forward(self, image):
        with torch.no_grad():
            model_output = self.model.forward_export(image)
        return model_output


class ProcessExportWrapper(torch.nn.Module):
    """
        split the preprocess that can be wrapped as a preprocess jit model
        the preproprocess procedure cannot be optimized in an end2end blade model due to dynamic shape problem
    """

    def __init__(self, example_inputs, process_fn: 'Optional[Callable]'=None) ->None:
        super().__init__()
        self.process_fn = process_fn

    def forward(self, image):
        with torch.no_grad():
            output = self.process_fn(image)
        return output


class DistributedParallel:
    """Base class of parallelism."""

    def __init__(self, rank, world_size):
        self._rank = rank
        self._world_size = world_size

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def correct_mask(self, target, inputs):
        mask = torch.zeros(inputs.size(), device=inputs.device, dtype=inputs.dtype)
        mask.scatter_(1, target.view(-1, 1).long(), 1)
        return mask

    def correct_predictions(self, target, logits, k=1):
        if k == 1:
            pred = torch.max(logits, dim=1)[1]
            return (pred == target.view(-1, 1)).sum().item()
        pred = torch.topk(logits, k, dim=1)[1]
        return (pred == target.view(-1, 1)).sum().item()

    def xavier_uniform_(self, weight, gain=1.0):
        return torch.nn.init.xavier_uniform_(weight, gain=gain)


class _Cat(torch.autograd.Function):
    """Concat inputs."""

    @staticmethod
    def forward(ctx, inputs, dim, rank, world_size):
        """Cat is defined as:
    .. math::
      \\text{all_cat}(x_i) = \\bigoplus_j x_j
    """
        ctx.dim = dim
        ctx.rank = rank
        ctx.world_size = world_size
        all_inputs = [torch.zeros(inputs.size(), dtype=inputs.dtype, device=inputs.device) for _ in range(world_size)]
        torch.distributed.all_gather(all_inputs, inputs)
        output = torch.cat(all_inputs, dim=dim)
        output.requires_grad_()
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """Gradient of Cat is defined as:
    .. math::
      \\nabla \\text{all_cat}(x_i) =  \\text{split}(\\nabla x_i)
    """
        grad_input = grad_output.clone()
        torch.distributed.all_reduce(grad_input)
        grad_input_dim_size = grad_input.size()[ctx.dim]
        assert grad_input_dim_size % ctx.world_size == 0
        split_size = grad_input_dim_size // ctx.world_size
        grad_input_splits = torch.split(grad_input, split_size, dim=ctx.dim)
        return grad_input_splits[ctx.rank], None, None, None


def all_cat(inputs, dim=0, rank=0, world_size=1):
    return _Cat.apply(inputs, dim, rank, world_size)


class _LogSoftmax(torch.autograd.Function):
    """Compute log softmax of logits."""

    @staticmethod
    def forward(ctx, logits, epsilon):
        """LogSoftmax is defined as:
    .. math::
      \\log(\\text{softmax}(x_i))
        = \\log\\left(\\frac{\\text{e}^{x_i}}{\\sum_j\\text{e}^{x_j}}\\right)
        = x_i - \\log\\sum_j\\text{e}^{x_j}

    For numerical stability, it subtracts the maximum value for every logits:
    .. math::
      \\log(\\text{softmax}(x_i))
        = \\hat{x_i} - \\log\\sum_j\\text{e}^{\\hat{x_j}},
        \\hat{x} = x - \\max_j{x_j}
    """
        ctx.logits_dtype = logits.dtype
        logits_max = torch.max(logits, dim=1).values
        torch.distributed.all_reduce(logits_max, op=torch.distributed.ReduceOp.MAX)
        logits = logits - logits_max.view(-1, 1)
        logits_exp = torch.exp(logits)
        logits_exp_sum = torch.sum(logits_exp, dim=1)
        torch.distributed.all_reduce(logits_exp_sum)
        logits_exp_sum_log = torch.log(logits_exp_sum + epsilon)
        prob_log = logits - logits_exp_sum_log.view(-1, 1)
        ctx.save_for_backward(prob_log)
        prob_log.requires_grad_()
        return prob_log

    @staticmethod
    def backward(ctx, grad_output):
        """Gradient of LogSoftmax is defined as:
    .. math::
      \\nabla\\log(\\text{softmax}(x_i))
        = \\nabla x_i - \\text{softmax}(x_i) \\sum_j \\nabla x_j
    """
        grad_output_sum = torch.sum(grad_output, dim=1)
        torch.distributed.all_reduce(grad_output_sum)
        prob_log, = ctx.saved_tensors
        grad_input = torch.exp(prob_log) * grad_output_sum.view(-1, 1)
        grad_input = grad_output - grad_input
        grad_input = grad_input.type(dtype=ctx.logits_dtype)
        return grad_input, None


def all_log_softmax(logits, epsilon=1e-08):
    return _LogSoftmax.apply(logits, epsilon)


class _NLLLoss(torch.autograd.Function):
    """calculate NLLLoss from mask."""

    @staticmethod
    def forward(ctx, inputs, correct_mask):
        ctx.inputs_size = inputs.size()
        ctx.save_for_backward(correct_mask)
        loss = torch.sum(inputs * correct_mask) / -ctx.inputs_size[0]
        torch.distributed.all_reduce(loss)
        loss.requires_grad_()
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        correct_mask, = ctx.saved_tensors
        grad_input = grad_output.repeat(*ctx.inputs_size)
        grad_input = grad_input * correct_mask / -ctx.inputs_size[0]
        return grad_input, None


def all_nll_loss(inputs, correct_mask):
    return _NLLLoss.apply(inputs, correct_mask)


class _Sum(torch.autograd.Function):
    """Sum inputs."""

    @staticmethod
    def forward(_, inputs):
        """Sum is defined as:
    .. math::
      \\text{all_sum}(x_i) = \\sum_j x_j
    """
        inputs_sum = inputs.clone()
        torch.distributed.all_reduce(inputs_sum)
        inputs_sum.requires_grad_()
        return inputs_sum

    @staticmethod
    def backward(_, grad_output):
        """Gradient of Sum is defined as:
    .. math::
      \\nabla \\text{all_sum}(x_i) = \\sum_j\\nabla x_j
    """
        grad_input = grad_output.clone()
        torch.distributed.all_reduce(grad_input)
        return grad_input


def all_sum(inputs):
    return _Sum.apply(inputs)


def shard_correct_mask(target, inputs, rank=0):
    """Get correct mask of inputs."""
    inputs_size = inputs.size()
    target_shard_begin = inputs_size[1] * rank
    target_shard_end = inputs_size[1] * (rank + 1)
    target_shard_lmask = torch.ge(target, target_shard_begin)
    target_shard_rmask = torch.lt(target, target_shard_end)
    target_mask = target_shard_lmask * target_shard_rmask
    target_shard = (target - target_shard_begin) * target_mask.long()
    mask = torch.zeros(inputs_size, device=inputs.device, dtype=inputs.dtype)
    mask.scatter_(1, target_shard.view(-1, 1).long(), 1)
    mask.masked_fill_((~target_mask).view(-1, 1).expand(*inputs.size()), 0)
    return mask


def shard_correct_predictions(target, logits, world_size=1):
    """Calculate correct predictions for logits."""
    shard_max_logits, shard_expected_class = torch.max(logits, dim=1)
    all_max_logits = [torch.zeros(shard_max_logits.size(), dtype=shard_max_logits.dtype, device=shard_max_logits.device) for _ in range(world_size)]
    torch.distributed.all_gather(all_max_logits, shard_max_logits)
    all_max_logits = torch.cat([t.view(-1, 1) for t in all_max_logits], dim=1)
    rank_pred = torch.max(all_max_logits, dim=1)[1].view(-1, 1)
    all_shard_pred = [torch.zeros(shard_expected_class.size(), dtype=shard_expected_class.dtype, device=shard_expected_class.device) for _ in range(world_size)]
    torch.distributed.all_gather(all_shard_pred, shard_expected_class)
    all_shard_pred = torch.cat([t.view(-1, 1) for t in all_shard_pred], dim=1)
    all_shard_pred_mask = torch.zeros(all_shard_pred.size(), device=all_shard_pred.device, dtype=all_shard_pred.dtype)
    all_shard_pred_mask.scatter_(1, rank_pred.long(), 1)
    shard_pred = torch.sum(all_shard_pred * all_shard_pred_mask, dim=1).view(-1, 1)
    pred = shard_pred + rank_pred * logits.size()[1]
    return (pred == target.data.view_as(pred)).sum().item()


def shard_target_and_mask(target, output_features, rank=0):
    target_shard_begin = output_features * rank
    target_shard_end = output_features * (rank + 1)
    target_shard_lmask = torch.ge(target, target_shard_begin)
    target_shard_rmask = torch.lt(target, target_shard_end)
    target_mask = target_shard_lmask * target_shard_rmask
    target_shard = (target - target_shard_begin) * target_mask.long()
    return target_shard, target_mask


def shard_topk_correct_predictions(target, logits, k, world_size=1):
    """Calculate correct predictions for logits."""
    logits_topk, logits_topk_idx = torch.topk(logits, k, dim=1)
    all_logits_topk = [torch.zeros(logits_topk.size(), dtype=logits_topk.dtype, device=logits_topk.device) for _ in range(world_size)]
    torch.distributed.all_gather(all_logits_topk, logits_topk)
    all_logits_topk = torch.cat([t.view(-1, k) for t in all_logits_topk], dim=1)
    all_logits_topk_idx = [torch.zeros(logits_topk_idx.size(), dtype=logits_topk_idx.dtype, device=logits_topk_idx.device) for _ in range(world_size)]
    torch.distributed.all_gather(all_logits_topk_idx, logits_topk_idx)
    all_logits_topk_idx = torch.cat([t.view(-1, k) for t in all_logits_topk_idx], dim=1)
    _, all_logits_topk_topk_idx = torch.topk(all_logits_topk, k, dim=1)
    all_logits_topk_topk_idx = all_logits_topk_topk_idx.view(-1, k)
    all_logits_topk_mask = torch.zeros(all_logits_topk_idx.size(), device=all_logits_topk_idx.device, dtype=all_logits_topk_idx.dtype)
    all_logits_topk_mask.scatter_(1, all_logits_topk_topk_idx.long(), 1)
    batch_size, shard_num_classes = logits.size()
    all_logits_topk_base = torch.cat([(torch.ones([batch_size, k], device=all_logits_topk_idx.device, dtype=all_logits_topk_idx.dtype) * p * shard_num_classes) for p in range(world_size)], dim=1)
    all_logits_topk_gidx = all_logits_topk_base + all_logits_topk_idx
    pred = torch.masked_select(all_logits_topk_gidx, all_logits_topk_mask.type(torch.bool)).view(batch_size, k)
    return (pred == target.view(-1, 1)).sum().item()


class ModelParallel(DistributedParallel):
    """All-to-All Model Parallelism."""

    def gather(self, inputs, dim=0, requires_grad=True):
        if requires_grad:
            return all_cat(inputs, dim=dim, rank=self.rank, world_size=self.world_size)
        all_inputs = [torch.zeros(inputs.size(), dtype=inputs.dtype, device=inputs.device) for _ in range(self.world_size)]
        torch.distributed.all_gather(all_inputs, inputs)
        return torch.cat(all_inputs, dim=dim)

    def gather_target(self, target):
        return self.gather(target, requires_grad=False)

    def reduce_sum(self, inputs):
        return all_sum(inputs)

    def log_softmax(self, logits, epsilon=1e-08):
        return all_log_softmax(logits, epsilon=epsilon)

    def nll_loss(self, inputs, correct_mask):
        return all_nll_loss(inputs, correct_mask)

    def correct_mask(self, target, inputs):
        return shard_correct_mask(target, inputs, rank=self.rank)

    def target_and_mask(self, target, output_features):
        return shard_target_and_mask(target, output_features, rank=self.rank)

    def correct_predictions(self, target, logits, k=1):
        if k == 1:
            return shard_correct_predictions(target, logits, world_size=self.world_size)
        return shard_topk_correct_predictions(target, logits, k, world_size=self.world_size)


class LogSoftmax(torch.nn.Module):
    """Applies the :math:`\\log(\\text{Softmax}(x))` function to an n-dimensional
  input Tensor rescaling them so that the elements of the
  n-dimensional output Tensor lie in the range (0, 1].

  Shape:
    - Input: :math:`(*)` where `*` means, any number of additional
      dimensions
    - Output: :math:`(*)`, same shape as the input

  Returns:
    a Tensor of the same dimension and shape as the input with
    values in the range (-inf,0].

  Examples::
    >>> m = LogSoftmax()
    >>> input = torch.randn(2, 3)
    >>> output = m(input)
  """

    def __init__(self, epsilon=0, parallel=None):
        super(LogSoftmax, self).__init__()
        self.epsilon = epsilon
        self.parallel = parallel

    def forward(self, logits):
        if isinstance(self.parallel, ModelParallel):
            return self.parallel.log_softmax(logits, epsilon=self.epsilon)
        return torch.nn.functional.log_softmax(logits, _stacklevel=5)


def _get_right_parentheses_index_(s):
    left_paren_count = 0
    for index, x in enumerate(s):
        if x == '(':
            left_paren_count += 1
        elif x == ')':
            left_paren_count -= 1
            if left_paren_count == 0:
                return index
        else:
            pass
    return None


class PlainNetBasicBlockClass(nn.Module):

    def __init__(self, in_channels=0, out_channels=0, stride=1, no_create=False, block_name=None, **kwargs):
        super(PlainNetBasicBlockClass, self).__init__(**kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.no_create = no_create
        self.block_name = block_name

    def forward(self, x):
        return x

    @staticmethod
    def create_from_str(s, no_create=False):
        assert PlainNetBasicBlockClass.is_instance_from_str(s)
        idx = _get_right_parentheses_index_(s)
        assert idx is not None
        param_str = s[len('PlainNetBasicBlockClass('):idx]
        tmp_idx = param_str.find('|')
        if tmp_idx < 0:
            tmp_block_name = 'uuid{}'.format(uuid.uuid4().hex)
        else:
            tmp_block_name = param_str[0:tmp_idx]
            param_str = param_str[tmp_idx + 1:]
        param_str_split = param_str.split(',')
        in_channels = int(param_str_split[0])
        out_channels = int(param_str_split[1])
        stride = int(param_str_split[2])
        return PlainNetBasicBlockClass(in_channels=in_channels, out_channels=out_channels, stride=stride, block_name=tmp_block_name, no_create=no_create), s[idx + 1:]

    @staticmethod
    def is_instance_from_str(s):
        if s.startswith('PlainNetBasicBlockClass(') and s[-1] == ')':
            return True
        else:
            return False


class Linear(PlainNetBasicBlockClass):

    def __init__(self, in_channels=None, out_channels=None, bias=None, copy_from=None, no_create=False, block_name=None, **kwargs):
        super(Linear, self).__init__(**kwargs)
        self.block_name = block_name
        if copy_from is not None:
            assert isinstance(copy_from, nn.Linear)
            self.in_channels = copy_from.in_channels
            self.out_channels = copy_from.out_channels
            self.bias = copy_from.bias
            assert in_channels is None or in_channels == self.in_channels
            assert out_channels is None or out_channels == self.out_channels
            assert bias is None or bias == self.bias
            self.netblock = copy_from
        else:
            self.in_channels = in_channels
            self.out_channels = out_channels
            self.bias = bias
            if not no_create:
                self.netblock = nn.Linear(self.in_channels, self.out_channels, bias=self.bias)

    def forward(self, x):
        return self.netblock(x)

    @staticmethod
    def create_from_str(s, no_create=False):
        assert Linear.is_instance_from_str(s)
        idx = _get_right_parentheses_index_(s)
        assert idx is not None
        param_str = s[len('Linear('):idx]
        tmp_idx = param_str.find('|')
        if tmp_idx < 0:
            tmp_block_name = 'uuid{}'.format(uuid.uuid4().hex)
        else:
            tmp_block_name = param_str[0:tmp_idx]
            param_str = param_str[tmp_idx + 1:]
        split_str = param_str.split(',')
        in_channels = int(split_str[0])
        out_channels = int(split_str[1])
        bias = int(split_str[2])
        return Linear(in_channels=in_channels, out_channels=out_channels, bias=bias == 1, block_name=tmp_block_name, no_create=no_create), s[idx + 1:]

    @staticmethod
    def is_instance_from_str(s):
        if s.startswith('Linear(') and s[-1] == ')':
            return True
        else:
            return False


UNIMPLEMENTED = 5194902719927681026


class NotImplementedError(BaseError):
    """Raised when an operation has not been implemented."""
    code = UNIMPLEMENTED


class ParameterInitializer:
    """Base class for parameter initializer."""

    def __call__(self, param, shard_rank=0, num_shards=1):
        raise NotImplementedError


class RenormUniformInitializer(ParameterInitializer):

    def __init__(self, maxnorm=1e-05, scale=100000.0):
        self.maxnorm = maxnorm
        self.scale = scale

    def __call__(self, param, parallel=None):
        param.data.uniform_(-1, 1).renorm_(2, 0, maxnorm=self.maxnorm).mul_(self.scale)


INVALID_VALUE = 5194902719927681028


class ValueError(BaseError):
    """Raised when an operation receives an invalid value."""
    code = INVALID_VALUE


class ArcFaceLinear(torch.nn.Module):
    """Applies a ArcFace transformation to the incoming data.
      See https://arxiv.org/abs/1801.05599 .
  """

    def __init__(self, in_features, out_features, margin=0.5, scale=64.0, fast_phi=False, epsilon=0, weight_initializer=None, l2_norm=False, parallel=None):
        super(ArcFaceLinear, self).__init__()
        if isinstance(parallel, ModelParallel):
            if out_features % parallel.world_size != 0:
                raise ValueError('out_features must be divided by parallel.world_size')
            self.out_features = out_features // parallel.world_size
        else:
            self.out_features = out_features
        self.in_features = in_features
        self.margin = margin
        self.scale = scale
        self.fast_phi = fast_phi
        self.epsilon = epsilon
        self.weight_initializer = weight_initializer
        if weight_initializer is None:
            self.weight_initializer = RenormUniformInitializer()
        self.l2_norm = l2_norm
        self.parallel = parallel
        self.weight = torch.nn.Parameter(torch.Tensor(self.out_features, self.in_features))
        self.reset_parameters()
        self._cos_margin = math.cos(margin)
        self._sin_margin = math.sin(margin)
        self._threshold = math.cos(math.pi - margin)
        self._min = math.sin(math.pi - margin) * self.margin

    def reset_parameters(self):
        """Reset parameters."""
        self.weight_initializer(self.weight)

    def forward(self, features, target):
        """Compute ::math`\\phi = \\cos(\\theta + margin)` and logits."""
        features = features.type(dtype=self.weight.dtype)
        if self.l2_norm:
            features_norm = torch.norm(features, 2, 1, True)
            features = torch.div(features, features_norm)
            weight_norm = torch.norm(self.weight, 2, 0, True)
            weight = torch.div(self.weight, weight_norm)
        else:
            features = torch.nn.functional.normalize(features)
            weight = torch.nn.functional.normalize(self.weight)
        cosine = torch.nn.functional.linear(features, weight)
        cosine = cosine.clamp(-1, 1)
        sine = torch.sqrt(1.0 + self.epsilon - cosine * cosine)
        phi = cosine * self._cos_margin - sine * self._sin_margin
        phi = phi.type(dtype=cosine.dtype)
        if self.fast_phi:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self._threshold, phi, cosine - self._min)
        if isinstance(self.parallel, ModelParallel):
            mask = self.parallel.correct_mask(target, cosine)
        else:
            mask = torch.zeros(cosine.size(), device=cosine.device, dtype=cosine.dtype)
            mask.scatter_(1, target.view(-1, 1).long(), 1)
        logits = mask * phi + (1.0 - mask) * cosine
        logits *= self.scale
        return logits


class NLLLoss(torch.nn.Module):
    """The negative log likelihood loss for log probabilities. It is
  useful to train a classification problem with `C` classes.

  The `input` given through a forward call is expected to contain
  log-probabilities of each class. `input` has to be a Tensor of size either
  :math:`(N, C)` or :math:`(N, C, d_1, d_2, ..., d_K)`
  with :math:`K \\geq 1` for the `K`-dimensional case (described later).

  Obtaining log-probabilities in a neural network is easily achieved by
  adding a  `LogSoftmax` layer in the last layer of your network.
  You may use `CrossEntropyLoss` instead, if you prefer not to add an
  extra layer.

  The `target` that this loss expects should be a class index in the range
  :math:`[0, C-1]` where `C = number\\_classes`.

  NLLLoss is defined as:
  .. math::
      \\ell(x, y) = -\\frac{1}{N}\\sum_{n=1}^N L_{i}

  Args:
    num_classes: total number of classes.
    focal: whether to use FocalLoss implementation.
    focal_gamm: The focusing parameter of FocalLoss.
    rank: rank of current replica.
    world_size: size of replicas.

  Shape:
    - Input: :math:`(\\frac{N}{P}, C)` where `C = number of classes`, or
      :math:`(N, C, d_1, d_2, ..., d_K)` with :math:`K \\geq 1`
      in the case of `K`-dimensional loss.
    - Target: :math:`(\\frac{N}{P}, 1)` where each value is
      :math:`0 \\leq \\text{targets}[i] \\leq C-1`, or
      :math:`(N, d_1, d_2, ..., d_K)` with :math:`K \\geq 1` in the case of
      K-dimensional loss.
    - Output: scalar.
      If :attr:`reduction` is ``'none'``, then the same size as the target:
      :math:`(N)`, or :math:`(N, d_1, d_2, ..., d_K)` with :math:`K \\geq 1` in
      the case of K-dimensional loss.
  Examples::
    >>> m = LogSoftmax(...)
    >>> loss = NLLLoss(...)
    >>> # input is of size N x C = 3 x 5
    >>> input = torch.randn(3, 5, requires_grad=True)
    >>> # each element in target has to have 0 <= value < C
    >>> target = torch.tensor([1, 0, 4])
    >>> output = loss(m(input), target)
    >>> output.backward()
  """

    def __init__(self, focal=False, focal_gamma=0, parallel=None):
        super(NLLLoss, self).__init__()
        self.focal = focal
        self.focal_gamma = focal_gamma
        self.parallel = parallel

    def forward(self, logprob, target):
        """Compute negative log likelihood loss from log-probs and target."""
        if isinstance(self.parallel, ModelParallel):
            with torch.no_grad():
                mask = self.parallel.correct_mask(target, logprob)
            loss = self.parallel.nll_loss(logprob, mask)
            if self.focal:
                loss_exp = torch.exp(-loss)
                loss = (1 - loss_exp) ** self.focal_gamma * loss
            return loss
        loss = torch.nn.functional.nll_loss(logprob, target)
        if self.focal:
            loss_exp = torch.exp(-loss)
            loss = (1 - loss_exp) ** self.focal_gamma * loss
        return loss


def _expand_onehot_labels(labels, label_weights, target_shape, ignore_index):
    """Expand onehot labels to match the size of prediction."""
    bin_labels = labels.new_zeros(target_shape)
    valid_mask = (labels >= 0) & (labels != ignore_index)
    inds = torch.nonzero(valid_mask, as_tuple=True)
    if inds[0].numel() > 0:
        if labels.dim() == 3:
            bin_labels[inds[0], labels[valid_mask], inds[1], inds[2]] = 1
        else:
            bin_labels[inds[0], labels[valid_mask]] = 1
    valid_mask = valid_mask.unsqueeze(1).expand(target_shape).float()
    if label_weights is None:
        bin_label_weights = valid_mask
    else:
        bin_label_weights = label_weights.unsqueeze(1).expand(target_shape)
        bin_label_weights = bin_label_weights * valid_mask
    return bin_labels, bin_label_weights, valid_mask


def reduce_loss(loss, reduction):
    """Reduce loss as specified.

    Args:
        loss (Tensor): Elementwise loss tensor.
        reduction (str): Options are "none", "mean" and "sum".

    Return:
        Tensor: Reduced loss tensor.
    """
    reduction_enum = F._Reduction.get_enum(reduction)
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.mean()
    elif reduction_enum == 2:
        return loss.sum()


def weight_reduce_loss(loss, weight=None, reduction='mean', avg_factor=None):
    """Apply element-wise weight and reduce loss.

    Args:
        loss (Tensor): Element-wise loss.
        weight (Tensor): Element-wise weights.
        reduction (str): Same as built-in losses of PyTorch.
        avg_factor (float): Average factor when computing the mean of losses.

    Returns:
        Tensor: Processed loss values.
    """
    if weight is not None:
        assert weight.dim() == loss.dim()
        loss = loss * weight
    if avg_factor is None:
        loss = reduce_loss(loss, reduction)
    elif reduction == 'mean':
        eps = torch.finfo(torch.float32).eps
        loss = loss.sum() / (avg_factor + eps)
    elif reduction != 'none':
        raise ValueError('avg_factor can not be used with reduction="sum"')
    return loss


def binary_cross_entropy(pred, label, weight=None, reduction='mean', avg_factor=None, class_weight=None, ignore_index=-100, avg_non_ignore=False, label_ceil=False, **kwargs):
    """Calculate the binary CrossEntropy loss.

    Args:
        pred (torch.Tensor): The prediction with shape (N, 1).
        label (torch.Tensor): The learning label of the prediction.
            Note: In bce loss, label < 0 is invalid.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        reduction (str, optional): The method used to reduce the loss.
            Options are "none", "mean" and "sum".
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
        class_weight (list[float], optional): The weight for each class.
        ignore_index (int): The label index to be ignored. Default: -100.
        avg_non_ignore (bool): The flag decides to whether the loss is
            only averaged over non-ignored targets. Default: False.
            `New in version 0.23.0.`
        label_ceil (bool): When use bce and set label_ceil=True,
            it will make elements belong to (0, 1] in label change to 1.
            Default: False.

    Returns:
        torch.Tensor: The calculated loss
    """
    if len(pred.shape) > 1 and pred.shape[1] == 1:
        assert label[label != ignore_index].max() <= 1, 'For pred with shape [N, 1, H, W], its label must have at most 2 classes'
        pred = pred.squeeze()
    if pred.dim() != label.dim():
        assert pred.dim() == 2 and label.dim() == 1 or pred.dim() == 4 and label.dim() == 3, 'Only pred shape [N, C], label shape [N] or pred shape [N, C, H, W], label shape [N, H, W] are supported'
        label, weight, valid_mask = _expand_onehot_labels(label, weight, pred.shape, ignore_index)
    else:
        valid_mask = ((label >= 0) & (label != ignore_index)).float()
        if weight is not None:
            weight = weight * valid_mask
        else:
            weight = valid_mask
    if label_ceil:
        label = label.gt(0.0).type(label.dtype)
    if reduction == 'mean' and avg_factor is None and avg_non_ignore:
        avg_factor = valid_mask.sum().item()
    loss = F.binary_cross_entropy_with_logits(pred, label.float(), pos_weight=class_weight, reduction='none')
    loss = weight_reduce_loss(loss, weight, reduction=reduction, avg_factor=avg_factor)
    return loss


def cross_entropy(pred, label, weight=None, class_weight=None, reduction='mean', avg_factor=None, ignore_index=-100, avg_non_ignore=False):
    """cross_entropy. The wrapper function for :func:`F.cross_entropy`
    support sample-wise loss weight and the reduction average loss over non-ignored elements.

    Args:
        pred (torch.Tensor): The prediction with shape (N, 1).
        label (torch.Tensor): The learning label of the prediction.
        weight (torch.Tensor, optional): Sample-wise loss weight.
            Default: None.
        class_weight (list[float], optional): The weight for each class.
            Default: None.
        reduction (str, optional): The method used to reduce the loss.
            Options are 'none', 'mean' and 'sum'. Default: 'mean'.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Default: None.
        ignore_index (int): Specifies a target value that is ignored and
            does not contribute to the input gradients. When
            ``avg_non_ignore `` is ``True``, and the ``reduction`` is
            ``''mean''``, the loss is averaged over non-ignored targets.
            Defaults: -100.
        avg_non_ignore (bool): The flag decides to whether the loss is
            only averaged over non-ignored targets. Default: False.
            `New in version 0.23.0.`
    """
    loss = F.cross_entropy(pred, label, weight=class_weight, reduction='none', ignore_index=ignore_index)
    if avg_factor is None and avg_non_ignore and reduction == 'mean':
        avg_factor = label.numel() - (label == ignore_index).sum().item()
    if weight is not None:
        weight = weight.float()
    loss = weight_reduce_loss(loss, weight=weight, reduction=reduction, avg_factor=avg_factor)
    return loss


def get_class_weight(class_weight):
    """Get class weight for loss function.

    Args:
        class_weight (list[float] | str | None): If class_weight is a str,
            take it as a file name and read from it.
    """
    if isinstance(class_weight, str):
        if class_weight.endswith('.npy'):
            class_weight = np.load(class_weight)
        else:
            class_weight = mmcv.load(class_weight)
    return class_weight


def mask_cross_entropy(pred, target, label, reduction='mean', avg_factor=None, class_weight=None, ignore_index=None, **kwargs):
    """Calculate the CrossEntropy loss for masks.

    Args:
        pred (torch.Tensor): The prediction with shape (N, C), C is the number
            of classes.
        target (torch.Tensor): The learning label of the prediction.
        label (torch.Tensor): ``label`` indicates the class label of the mask'
            corresponding object. This will be used to select the mask in the
            of the class which the object belongs to when the mask prediction
            if not class-agnostic.
        reduction (str, optional): The method used to reduce the loss.
            Options are "none", "mean" and "sum".
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
        class_weight (list[float], optional): The weight for each class.
        ignore_index (None): Placeholder, to be consistent with other loss.
            Default: None.

    Returns:
        torch.Tensor: The calculated loss
    """
    assert ignore_index is None, 'BCE loss does not support ignore_index'
    assert reduction == 'mean' and avg_factor is None
    num_rois = pred.size()[0]
    inds = torch.arange(0, num_rois, dtype=torch.long, device=pred.device)
    pred_slice = pred[inds, label].squeeze(1)
    return F.binary_cross_entropy_with_logits(pred_slice, target, weight=class_weight, reduction='mean')[None]


class CrossEntropyLoss(nn.Module):
    """CrossEntropyLoss.

    Args:
        use_sigmoid (bool, optional): Whether the prediction uses sigmoid
            of softmax. Defaults to False.
        use_mask (bool, optional): Whether to use mask cross entropy loss.
            Defaults to False.
        reduction (str, optional): . Defaults to 'mean'.
            Options are "none", "mean" and "sum".
        class_weight (list[float] | str, optional): Weight of each class. If in
            str format, read them from a file. Defaults to None.
        loss_weight (float, optional): Weight of the loss. Defaults to 1.0.
        loss_name (str, optional): Name of the loss item. If you want this loss
            item to be included into the backward graph, `loss_` must be the
            prefix of the name. Defaults to 'loss_ce'.
        avg_non_ignore (bool): The flag decides to whether the loss is
            only averaged over non-ignored targets. Default: False.
            `New in version 0.23.0.`
        label_ceil (bool): When use bce and set label_ceil=True,
            it will make elements belong to (0, 1] in label change to 1.
            Default: False.
    """

    def __init__(self, use_sigmoid=False, use_mask=False, reduction='mean', class_weight=None, loss_weight=1.0, loss_name='loss_ce', avg_non_ignore=False, label_ceil=False):
        super(CrossEntropyLoss, self).__init__()
        assert use_sigmoid is False or use_mask is False
        self.use_sigmoid = use_sigmoid
        if label_ceil:
            if not use_sigmoid:
                raise ValueError('‘label_ceil’ is supported only when ‘use_sigmoid’ is true. If not use bce, please set ‘label_ceil’=False')
        self.use_mask = use_mask
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = get_class_weight(class_weight)
        self.avg_non_ignore = avg_non_ignore
        if not self.avg_non_ignore and self.reduction == 'mean':
            warnings.warn('Default ``avg_non_ignore`` is False, if you would like to ignore the certain label and average loss over non-ignore labels, which is the same with PyTorch official cross_entropy, set ``avg_non_ignore=True``.')
        if self.use_sigmoid:
            self.cls_criterion = binary_cross_entropy
        elif self.use_mask:
            self.cls_criterion = mask_cross_entropy
        else:
            self.cls_criterion = cross_entropy
        self._loss_name = loss_name
        self.label_ceil = label_ceil

    def extra_repr(self):
        """Extra repr."""
        s = f'avg_non_ignore={self.avg_non_ignore}'
        return s

    def forward(self, cls_score, label, weight=None, avg_factor=None, reduction_override=None, ignore_index=-100, **kwargs):
        """Forward function."""
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = reduction_override if reduction_override else self.reduction
        if self.class_weight is not None:
            class_weight = cls_score.new_tensor(self.class_weight)
        else:
            class_weight = None
        if self.use_sigmoid:
            loss_cls = self.loss_weight * self.cls_criterion(cls_score, label, weight, class_weight=class_weight, reduction=reduction, avg_factor=avg_factor, avg_non_ignore=self.avg_non_ignore, ignore_index=ignore_index, label_ceil=self.label_ceil, **kwargs)
        else:
            loss_cls = self.loss_weight * self.cls_criterion(cls_score, label, weight, class_weight=class_weight, reduction=reduction, avg_factor=avg_factor, avg_non_ignore=self.avg_non_ignore, ignore_index=ignore_index, **kwargs)
        return loss_cls

    @property
    def loss_name(self):
        """Loss Name.

        This function must be implemented and will return the name of this
        loss function. This name will be used to combine different loss items
        by simple sum operation. In addition, if you want this loss item to be
        included into the backward graph, `loss_` must be the prefix of the
        name.

        Returns:
            str: The name of this loss item.
        """
        return self._loss_name


def py_focal_loss_with_prob(pred, target, weight=None, gamma=2.0, alpha=0.25, reduction='mean', avg_factor=None):
    """PyTorch version of `Focal Loss <https://arxiv.org/abs/1708.02002>`_.
    Different from `py_sigmoid_focal_loss`, this function accepts probability
    as input.

    Args:
        pred (torch.Tensor): The prediction probability with shape (N, C),
            C is the number of classes.
        target (torch.Tensor): The learning label of the prediction.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        gamma (float, optional): The gamma for calculating the modulating
            factor. Defaults to 2.0.
        alpha (float, optional): A balanced form for Focal Loss.
            Defaults to 0.25.
        reduction (str, optional): The method used to reduce the loss into
            a scalar. Defaults to 'mean'.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
    """
    num_classes = pred.size(1)
    target = F.one_hot(target, num_classes=num_classes + 1)
    target = target[:, :num_classes]
    target = target.type_as(pred)
    pt = (1 - pred) * target + pred * (1 - target)
    focal_weight = (alpha * target + (1 - alpha) * (1 - target)) * pt.pow(gamma)
    loss = F.binary_cross_entropy(pred, target, reduction='none') * focal_weight
    if weight is not None:
        if weight.shape != loss.shape:
            if weight.size(0) == loss.size(0):
                weight = weight.view(-1, 1)
            else:
                assert weight.numel() == loss.numel()
                weight = weight.view(loss.size(0), -1)
        assert weight.ndim == loss.ndim
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss


def py_sigmoid_focal_loss(inputs, targets, weight=None, gamma=2.0, alpha=0.25, reduction='mean', avg_factor=None):
    """PyTorch version of `Focal Loss <https://arxiv.org/abs/1708.02002>`_.

    Args:
        inputs (torch.Tensor): The prediction with shape (N, C), C is the
            number of classes
        targets (torch.Tensor): The learning label of the prediction.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        gamma (float, optional): The gamma for calculating the modulating
            factor. Defaults to 2.0.
        alpha (float, optional): A balanced form for Focal Loss.
            Defaults to 0.25.
        reduction (str, optional): The method used to reduce the loss into
            a scalar. Defaults to 'mean'.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * (1 - p_t) ** gamma
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    if weight is not None:
        if weight.shape != loss.shape:
            if weight.size(0) == loss.size(0):
                weight = weight.view(-1, 1)
            else:
                assert weight.numel() == loss.numel()
                weight = weight.view(loss.size(0), -1)
        assert weight.ndim == loss.ndim
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss


def sigmoid_focal_loss(pred, target, weight=None, gamma=2.0, alpha=0.25, reduction='mean', avg_factor=None):
    """A warpper of cuda version `Focal Loss
    <https://arxiv.org/abs/1708.02002>`_.

    Args:
        pred (torch.Tensor): The prediction with shape (N, C), C is the number
            of classes.
        target (torch.Tensor): The learning label of the prediction.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        gamma (float, optional): The gamma for calculating the modulating
            factor. Defaults to 2.0.
        alpha (float, optional): A balanced form for Focal Loss.
            Defaults to 0.25.
        reduction (str, optional): The method used to reduce the loss into
            a scalar. Defaults to 'mean'. Options are "none", "mean" and "sum".
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
    """
    loss = _sigmoid_focal_loss(pred.contiguous(), target.contiguous(), gamma, alpha, None, 'none')
    if weight is not None:
        if weight.shape != loss.shape:
            if weight.size(0) == loss.size(0):
                weight = weight.view(-1, 1)
            else:
                assert weight.numel() == loss.numel()
                weight = weight.view(loss.size(0), -1)
        assert weight.ndim == loss.ndim
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss


class FocalLoss(nn.Module):

    def __init__(self, use_sigmoid=True, gamma=2.0, alpha=0.25, reduction='mean', loss_weight=1.0, activated=False):
        """`Focal Loss <https://arxiv.org/abs/1708.02002>`_

        Args:
            use_sigmoid (bool, optional): Whether to the prediction is
                used for sigmoid or softmax. Defaults to True.
            gamma (float, optional): The gamma for calculating the modulating
                factor. Defaults to 2.0.
            alpha (float, optional): A balanced form for Focal Loss.
                Defaults to 0.25.
            reduction (str, optional): The method used to reduce the loss into
                a scalar. Defaults to 'mean'. Options are "none", "mean" and
                "sum".
            loss_weight (float, optional): Weight of loss. Defaults to 1.0.
            activated (bool, optional): Whether the input is activated.
                If True, it means the input has been activated and can be
                treated as probabilities. Else, it should be treated as logits.
                Defaults to False.
        """
        super(FocalLoss, self).__init__()
        assert use_sigmoid is True, 'Only sigmoid focal loss supported now.'
        self.use_sigmoid = use_sigmoid
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.activated = activated

    def forward(self, pred, target, weight=None, avg_factor=None, reduction_override=None):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning label of the prediction.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Options are "none", "mean" and "sum".

        Returns:
            torch.Tensor: The calculated loss
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = reduction_override if reduction_override else self.reduction
        if self.use_sigmoid:
            if self.activated:
                calculate_loss_func = py_focal_loss_with_prob
            elif torch.cuda.is_available() and pred.is_cuda:
                calculate_loss_func = sigmoid_focal_loss
            else:
                num_classes = pred.size(1)
                target = F.one_hot(target, num_classes=num_classes + 1)
                target = target[:, :num_classes]
                calculate_loss_func = py_sigmoid_focal_loss
            loss_cls = self.loss_weight * calculate_loss_func(pred, target, weight, gamma=self.gamma, alpha=self.alpha, reduction=reduction, avg_factor=avg_factor)
        else:
            raise NotImplementedError
        return loss_cls


class SoftmaxLoss(torch.nn.Module):
    """This criterion combines :func:`Linear` and
    :func:`CrossEntropyLoss` in one single class.
  """

    def __init__(self, in_features, out_features, bias=True, epsilon=0, weight_initializer=None, bias_initializer=None, parallel=None):
        super(SoftmaxLoss, self).__init__()
        self._linear = Linear(in_features, out_features, bias=bias, weight_initializer=weight_initializer, bias_initializer=bias_initializer, parallel=parallel)
        self._log_softmax = LogSoftmax(epsilon=epsilon, parallel=parallel)
        self._nll_loss = NLLLoss(parallel=parallel)
        self._parallel = parallel

    def forward(self, features, target):
        if isinstance(self._parallel, ModelParallel):
            features = self._parallel.gather(features)
        logits = self._linear(features.squeeze())
        logprob = self._log_softmax(logits)
        if isinstance(self._parallel, ModelParallel):
            target = self._parallel.gather_target(target)
        return self._nll_loss(logprob, target)


class ArcMarginLoss(torch.nn.Module):
    """This criterion combines :func:`ArcFaceLinear` and
    :func:`CrossEntropyLoss` in one single class.
  """

    def __init__(self, in_features, out_features, margin=0.5, scale=64.0, fast_phi=False, epsilon=0, weight_initializer=None, l2_norm=False, parallel=None):
        super(ArcMarginLoss, self).__init__()
        self._linear = ArcFaceLinear(in_features, out_features, margin=margin, scale=scale, l2_norm=l2_norm, fast_phi=fast_phi, epsilon=epsilon, weight_initializer=weight_initializer, parallel=parallel)
        self._log_softmax = LogSoftmax(epsilon=epsilon, parallel=parallel)
        self._nll_loss = NLLLoss(parallel=parallel)
        self._parallel = parallel

    def forward(self, features, target):
        if isinstance(self._parallel, ModelParallel):
            features = self._parallel.gather(features)
            target = self._parallel.gather_target(target)
        logits = self._linear(features.squeeze(), target)
        logprob = self._log_softmax(logits)
        return self._nll_loss(logprob, target)


class BenchMarkMLP(nn.Module):

    def __init__(self, feature_num, num_classes=1000, avg_pool=False, **kwargs):
        super(BenchMarkMLP, self).__init__()
        self.fc1 = nn.Linear(feature_num, feature_num)
        self.relu1 = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.avg_pool = avg_pool

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        if self.avg_pool:
            x = self.pool(x)
        x = self.fc1(x)
        x = self.relu1(x)
        return tuple([x])


class BNInception(nn.Module):

    def __init__(self, num_classes=0):
        super(BNInception, self).__init__()
        inplace = True
        self.conv1_7x7_s2 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3))
        self.conv1_7x7_s2_bn = nn.BatchNorm2d(64, affine=True)
        self.conv1_relu_7x7 = nn.ReLU(inplace)
        self.pool1_3x3_s2 = nn.MaxPool2d((3, 3), stride=(2, 2), dilation=(1, 1), ceil_mode=True)
        self.conv2_3x3_reduce = nn.Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1))
        self.conv2_3x3_reduce_bn = nn.BatchNorm2d(64, affine=True)
        self.conv2_relu_3x3_reduce = nn.ReLU(inplace)
        self.conv2_3x3 = nn.Conv2d(64, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv2_3x3_bn = nn.BatchNorm2d(192, affine=True)
        self.conv2_relu_3x3 = nn.ReLU(inplace)
        self.pool2_3x3_s2 = nn.MaxPool2d((3, 3), stride=(2, 2), dilation=(1, 1), ceil_mode=True)
        self.inception_3a_1x1 = nn.Conv2d(192, 64, kernel_size=(1, 1), stride=(1, 1))
        self.inception_3a_1x1_bn = nn.BatchNorm2d(64, affine=True)
        self.inception_3a_relu_1x1 = nn.ReLU(inplace)
        self.inception_3a_3x3_reduce = nn.Conv2d(192, 64, kernel_size=(1, 1), stride=(1, 1))
        self.inception_3a_3x3_reduce_bn = nn.BatchNorm2d(64, affine=True)
        self.inception_3a_relu_3x3_reduce = nn.ReLU(inplace)
        self.inception_3a_3x3 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_3a_3x3_bn = nn.BatchNorm2d(64, affine=True)
        self.inception_3a_relu_3x3 = nn.ReLU(inplace)
        self.inception_3a_double_3x3_reduce = nn.Conv2d(192, 64, kernel_size=(1, 1), stride=(1, 1))
        self.inception_3a_double_3x3_reduce_bn = nn.BatchNorm2d(64, affine=True)
        self.inception_3a_relu_double_3x3_reduce = nn.ReLU(inplace)
        self.inception_3a_double_3x3_1 = nn.Conv2d(64, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_3a_double_3x3_1_bn = nn.BatchNorm2d(96, affine=True)
        self.inception_3a_relu_double_3x3_1 = nn.ReLU(inplace)
        self.inception_3a_double_3x3_2 = nn.Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_3a_double_3x3_2_bn = nn.BatchNorm2d(96, affine=True)
        self.inception_3a_relu_double_3x3_2 = nn.ReLU(inplace)
        self.inception_3a_pool = nn.AvgPool2d(3, stride=1, padding=1, ceil_mode=True, count_include_pad=True)
        self.inception_3a_pool_proj = nn.Conv2d(192, 32, kernel_size=(1, 1), stride=(1, 1))
        self.inception_3a_pool_proj_bn = nn.BatchNorm2d(32, affine=True)
        self.inception_3a_relu_pool_proj = nn.ReLU(inplace)
        self.inception_3b_1x1 = nn.Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1))
        self.inception_3b_1x1_bn = nn.BatchNorm2d(64, affine=True)
        self.inception_3b_relu_1x1 = nn.ReLU(inplace)
        self.inception_3b_3x3_reduce = nn.Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1))
        self.inception_3b_3x3_reduce_bn = nn.BatchNorm2d(64, affine=True)
        self.inception_3b_relu_3x3_reduce = nn.ReLU(inplace)
        self.inception_3b_3x3 = nn.Conv2d(64, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_3b_3x3_bn = nn.BatchNorm2d(96, affine=True)
        self.inception_3b_relu_3x3 = nn.ReLU(inplace)
        self.inception_3b_double_3x3_reduce = nn.Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1))
        self.inception_3b_double_3x3_reduce_bn = nn.BatchNorm2d(64, affine=True)
        self.inception_3b_relu_double_3x3_reduce = nn.ReLU(inplace)
        self.inception_3b_double_3x3_1 = nn.Conv2d(64, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_3b_double_3x3_1_bn = nn.BatchNorm2d(96, affine=True)
        self.inception_3b_relu_double_3x3_1 = nn.ReLU(inplace)
        self.inception_3b_double_3x3_2 = nn.Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_3b_double_3x3_2_bn = nn.BatchNorm2d(96, affine=True)
        self.inception_3b_relu_double_3x3_2 = nn.ReLU(inplace)
        self.inception_3b_pool = nn.AvgPool2d(3, stride=1, padding=1, ceil_mode=True, count_include_pad=True)
        self.inception_3b_pool_proj = nn.Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1))
        self.inception_3b_pool_proj_bn = nn.BatchNorm2d(64, affine=True)
        self.inception_3b_relu_pool_proj = nn.ReLU(inplace)
        self.inception_3c_3x3_reduce = nn.Conv2d(320, 128, kernel_size=(1, 1), stride=(1, 1))
        self.inception_3c_3x3_reduce_bn = nn.BatchNorm2d(128, affine=True)
        self.inception_3c_relu_3x3_reduce = nn.ReLU(inplace)
        self.inception_3c_3x3 = nn.Conv2d(128, 160, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.inception_3c_3x3_bn = nn.BatchNorm2d(160, affine=True)
        self.inception_3c_relu_3x3 = nn.ReLU(inplace)
        self.inception_3c_double_3x3_reduce = nn.Conv2d(320, 64, kernel_size=(1, 1), stride=(1, 1))
        self.inception_3c_double_3x3_reduce_bn = nn.BatchNorm2d(64, affine=True)
        self.inception_3c_relu_double_3x3_reduce = nn.ReLU(inplace)
        self.inception_3c_double_3x3_1 = nn.Conv2d(64, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_3c_double_3x3_1_bn = nn.BatchNorm2d(96, affine=True)
        self.inception_3c_relu_double_3x3_1 = nn.ReLU(inplace)
        self.inception_3c_double_3x3_2 = nn.Conv2d(96, 96, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.inception_3c_double_3x3_2_bn = nn.BatchNorm2d(96, affine=True)
        self.inception_3c_relu_double_3x3_2 = nn.ReLU(inplace)
        self.inception_3c_pool = nn.MaxPool2d((3, 3), stride=(2, 2), dilation=(1, 1), ceil_mode=True)
        self.inception_4a_1x1 = nn.Conv2d(576, 224, kernel_size=(1, 1), stride=(1, 1))
        self.inception_4a_1x1_bn = nn.BatchNorm2d(224, affine=True)
        self.inception_4a_relu_1x1 = nn.ReLU(inplace)
        self.inception_4a_3x3_reduce = nn.Conv2d(576, 64, kernel_size=(1, 1), stride=(1, 1))
        self.inception_4a_3x3_reduce_bn = nn.BatchNorm2d(64, affine=True)
        self.inception_4a_relu_3x3_reduce = nn.ReLU(inplace)
        self.inception_4a_3x3 = nn.Conv2d(64, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_4a_3x3_bn = nn.BatchNorm2d(96, affine=True)
        self.inception_4a_relu_3x3 = nn.ReLU(inplace)
        self.inception_4a_double_3x3_reduce = nn.Conv2d(576, 96, kernel_size=(1, 1), stride=(1, 1))
        self.inception_4a_double_3x3_reduce_bn = nn.BatchNorm2d(96, affine=True)
        self.inception_4a_relu_double_3x3_reduce = nn.ReLU(inplace)
        self.inception_4a_double_3x3_1 = nn.Conv2d(96, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_4a_double_3x3_1_bn = nn.BatchNorm2d(128, affine=True)
        self.inception_4a_relu_double_3x3_1 = nn.ReLU(inplace)
        self.inception_4a_double_3x3_2 = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_4a_double_3x3_2_bn = nn.BatchNorm2d(128, affine=True)
        self.inception_4a_relu_double_3x3_2 = nn.ReLU(inplace)
        self.inception_4a_pool = nn.AvgPool2d(3, stride=1, padding=1, ceil_mode=True, count_include_pad=True)
        self.inception_4a_pool_proj = nn.Conv2d(576, 128, kernel_size=(1, 1), stride=(1, 1))
        self.inception_4a_pool_proj_bn = nn.BatchNorm2d(128, affine=True)
        self.inception_4a_relu_pool_proj = nn.ReLU(inplace)
        self.inception_4b_1x1 = nn.Conv2d(576, 192, kernel_size=(1, 1), stride=(1, 1))
        self.inception_4b_1x1_bn = nn.BatchNorm2d(192, affine=True)
        self.inception_4b_relu_1x1 = nn.ReLU(inplace)
        self.inception_4b_3x3_reduce = nn.Conv2d(576, 96, kernel_size=(1, 1), stride=(1, 1))
        self.inception_4b_3x3_reduce_bn = nn.BatchNorm2d(96, affine=True)
        self.inception_4b_relu_3x3_reduce = nn.ReLU(inplace)
        self.inception_4b_3x3 = nn.Conv2d(96, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_4b_3x3_bn = nn.BatchNorm2d(128, affine=True)
        self.inception_4b_relu_3x3 = nn.ReLU(inplace)
        self.inception_4b_double_3x3_reduce = nn.Conv2d(576, 96, kernel_size=(1, 1), stride=(1, 1))
        self.inception_4b_double_3x3_reduce_bn = nn.BatchNorm2d(96, affine=True)
        self.inception_4b_relu_double_3x3_reduce = nn.ReLU(inplace)
        self.inception_4b_double_3x3_1 = nn.Conv2d(96, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_4b_double_3x3_1_bn = nn.BatchNorm2d(128, affine=True)
        self.inception_4b_relu_double_3x3_1 = nn.ReLU(inplace)
        self.inception_4b_double_3x3_2 = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_4b_double_3x3_2_bn = nn.BatchNorm2d(128, affine=True)
        self.inception_4b_relu_double_3x3_2 = nn.ReLU(inplace)
        self.inception_4b_pool = nn.AvgPool2d(3, stride=1, padding=1, ceil_mode=True, count_include_pad=True)
        self.inception_4b_pool_proj = nn.Conv2d(576, 128, kernel_size=(1, 1), stride=(1, 1))
        self.inception_4b_pool_proj_bn = nn.BatchNorm2d(128, affine=True)
        self.inception_4b_relu_pool_proj = nn.ReLU(inplace)
        self.inception_4c_1x1 = nn.Conv2d(576, 160, kernel_size=(1, 1), stride=(1, 1))
        self.inception_4c_1x1_bn = nn.BatchNorm2d(160, affine=True)
        self.inception_4c_relu_1x1 = nn.ReLU(inplace)
        self.inception_4c_3x3_reduce = nn.Conv2d(576, 128, kernel_size=(1, 1), stride=(1, 1))
        self.inception_4c_3x3_reduce_bn = nn.BatchNorm2d(128, affine=True)
        self.inception_4c_relu_3x3_reduce = nn.ReLU(inplace)
        self.inception_4c_3x3 = nn.Conv2d(128, 160, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_4c_3x3_bn = nn.BatchNorm2d(160, affine=True)
        self.inception_4c_relu_3x3 = nn.ReLU(inplace)
        self.inception_4c_double_3x3_reduce = nn.Conv2d(576, 128, kernel_size=(1, 1), stride=(1, 1))
        self.inception_4c_double_3x3_reduce_bn = nn.BatchNorm2d(128, affine=True)
        self.inception_4c_relu_double_3x3_reduce = nn.ReLU(inplace)
        self.inception_4c_double_3x3_1 = nn.Conv2d(128, 160, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_4c_double_3x3_1_bn = nn.BatchNorm2d(160, affine=True)
        self.inception_4c_relu_double_3x3_1 = nn.ReLU(inplace)
        self.inception_4c_double_3x3_2 = nn.Conv2d(160, 160, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_4c_double_3x3_2_bn = nn.BatchNorm2d(160, affine=True)
        self.inception_4c_relu_double_3x3_2 = nn.ReLU(inplace)
        self.inception_4c_pool = nn.AvgPool2d(3, stride=1, padding=1, ceil_mode=True, count_include_pad=True)
        self.inception_4c_pool_proj = nn.Conv2d(576, 128, kernel_size=(1, 1), stride=(1, 1))
        self.inception_4c_pool_proj_bn = nn.BatchNorm2d(128, affine=True)
        self.inception_4c_relu_pool_proj = nn.ReLU(inplace)
        self.inception_4d_1x1 = nn.Conv2d(608, 96, kernel_size=(1, 1), stride=(1, 1))
        self.inception_4d_1x1_bn = nn.BatchNorm2d(96, affine=True)
        self.inception_4d_relu_1x1 = nn.ReLU(inplace)
        self.inception_4d_3x3_reduce = nn.Conv2d(608, 128, kernel_size=(1, 1), stride=(1, 1))
        self.inception_4d_3x3_reduce_bn = nn.BatchNorm2d(128, affine=True)
        self.inception_4d_relu_3x3_reduce = nn.ReLU(inplace)
        self.inception_4d_3x3 = nn.Conv2d(128, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_4d_3x3_bn = nn.BatchNorm2d(192, affine=True)
        self.inception_4d_relu_3x3 = nn.ReLU(inplace)
        self.inception_4d_double_3x3_reduce = nn.Conv2d(608, 160, kernel_size=(1, 1), stride=(1, 1))
        self.inception_4d_double_3x3_reduce_bn = nn.BatchNorm2d(160, affine=True)
        self.inception_4d_relu_double_3x3_reduce = nn.ReLU(inplace)
        self.inception_4d_double_3x3_1 = nn.Conv2d(160, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_4d_double_3x3_1_bn = nn.BatchNorm2d(192, affine=True)
        self.inception_4d_relu_double_3x3_1 = nn.ReLU(inplace)
        self.inception_4d_double_3x3_2 = nn.Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_4d_double_3x3_2_bn = nn.BatchNorm2d(192, affine=True)
        self.inception_4d_relu_double_3x3_2 = nn.ReLU(inplace)
        self.inception_4d_pool = nn.AvgPool2d(3, stride=1, padding=1, ceil_mode=True, count_include_pad=True)
        self.inception_4d_pool_proj = nn.Conv2d(608, 128, kernel_size=(1, 1), stride=(1, 1))
        self.inception_4d_pool_proj_bn = nn.BatchNorm2d(128, affine=True)
        self.inception_4d_relu_pool_proj = nn.ReLU(inplace)
        self.inception_4e_3x3_reduce = nn.Conv2d(608, 128, kernel_size=(1, 1), stride=(1, 1))
        self.inception_4e_3x3_reduce_bn = nn.BatchNorm2d(128, affine=True)
        self.inception_4e_relu_3x3_reduce = nn.ReLU(inplace)
        self.inception_4e_3x3 = nn.Conv2d(128, 192, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.inception_4e_3x3_bn = nn.BatchNorm2d(192, affine=True)
        self.inception_4e_relu_3x3 = nn.ReLU(inplace)
        self.inception_4e_double_3x3_reduce = nn.Conv2d(608, 192, kernel_size=(1, 1), stride=(1, 1))
        self.inception_4e_double_3x3_reduce_bn = nn.BatchNorm2d(192, affine=True)
        self.inception_4e_relu_double_3x3_reduce = nn.ReLU(inplace)
        self.inception_4e_double_3x3_1 = nn.Conv2d(192, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_4e_double_3x3_1_bn = nn.BatchNorm2d(256, affine=True)
        self.inception_4e_relu_double_3x3_1 = nn.ReLU(inplace)
        self.inception_4e_double_3x3_2 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.inception_4e_double_3x3_2_bn = nn.BatchNorm2d(256, affine=True)
        self.inception_4e_relu_double_3x3_2 = nn.ReLU(inplace)
        self.inception_4e_pool = nn.MaxPool2d((3, 3), stride=(2, 2), dilation=(1, 1), ceil_mode=True)
        self.inception_5a_1x1 = nn.Conv2d(1056, 352, kernel_size=(1, 1), stride=(1, 1))
        self.inception_5a_1x1_bn = nn.BatchNorm2d(352, affine=True)
        self.inception_5a_relu_1x1 = nn.ReLU(inplace)
        self.inception_5a_3x3_reduce = nn.Conv2d(1056, 192, kernel_size=(1, 1), stride=(1, 1))
        self.inception_5a_3x3_reduce_bn = nn.BatchNorm2d(192, affine=True)
        self.inception_5a_relu_3x3_reduce = nn.ReLU(inplace)
        self.inception_5a_3x3 = nn.Conv2d(192, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_5a_3x3_bn = nn.BatchNorm2d(320, affine=True)
        self.inception_5a_relu_3x3 = nn.ReLU(inplace)
        self.inception_5a_double_3x3_reduce = nn.Conv2d(1056, 160, kernel_size=(1, 1), stride=(1, 1))
        self.inception_5a_double_3x3_reduce_bn = nn.BatchNorm2d(160, affine=True)
        self.inception_5a_relu_double_3x3_reduce = nn.ReLU(inplace)
        self.inception_5a_double_3x3_1 = nn.Conv2d(160, 224, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_5a_double_3x3_1_bn = nn.BatchNorm2d(224, affine=True)
        self.inception_5a_relu_double_3x3_1 = nn.ReLU(inplace)
        self.inception_5a_double_3x3_2 = nn.Conv2d(224, 224, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_5a_double_3x3_2_bn = nn.BatchNorm2d(224, affine=True)
        self.inception_5a_relu_double_3x3_2 = nn.ReLU(inplace)
        self.inception_5a_pool = nn.AvgPool2d(3, stride=1, padding=1, ceil_mode=True, count_include_pad=True)
        self.inception_5a_pool_proj = nn.Conv2d(1056, 128, kernel_size=(1, 1), stride=(1, 1))
        self.inception_5a_pool_proj_bn = nn.BatchNorm2d(128, affine=True)
        self.inception_5a_relu_pool_proj = nn.ReLU(inplace)
        self.inception_5b_1x1 = nn.Conv2d(1024, 352, kernel_size=(1, 1), stride=(1, 1))
        self.inception_5b_1x1_bn = nn.BatchNorm2d(352, affine=True)
        self.inception_5b_relu_1x1 = nn.ReLU(inplace)
        self.inception_5b_3x3_reduce = nn.Conv2d(1024, 192, kernel_size=(1, 1), stride=(1, 1))
        self.inception_5b_3x3_reduce_bn = nn.BatchNorm2d(192, affine=True)
        self.inception_5b_relu_3x3_reduce = nn.ReLU(inplace)
        self.inception_5b_3x3 = nn.Conv2d(192, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_5b_3x3_bn = nn.BatchNorm2d(320, affine=True)
        self.inception_5b_relu_3x3 = nn.ReLU(inplace)
        self.inception_5b_double_3x3_reduce = nn.Conv2d(1024, 192, kernel_size=(1, 1), stride=(1, 1))
        self.inception_5b_double_3x3_reduce_bn = nn.BatchNorm2d(192, affine=True)
        self.inception_5b_relu_double_3x3_reduce = nn.ReLU(inplace)
        self.inception_5b_double_3x3_1 = nn.Conv2d(192, 224, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_5b_double_3x3_1_bn = nn.BatchNorm2d(224, affine=True)
        self.inception_5b_relu_double_3x3_1 = nn.ReLU(inplace)
        self.inception_5b_double_3x3_2 = nn.Conv2d(224, 224, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_5b_double_3x3_2_bn = nn.BatchNorm2d(224, affine=True)
        self.inception_5b_relu_double_3x3_2 = nn.ReLU(inplace)
        self.inception_5b_pool = nn.MaxPool2d((3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), ceil_mode=True)
        self.inception_5b_pool_proj = nn.Conv2d(1024, 128, kernel_size=(1, 1), stride=(1, 1))
        self.inception_5b_pool_proj_bn = nn.BatchNorm2d(128, affine=True)
        self.inception_5b_relu_pool_proj = nn.ReLU(inplace)
        self.num_classes = num_classes
        if num_classes > 0:
            self.last_linear = nn.Linear(1024, num_classes)
        self.default_pretrained_model_path = model_urls[self.__class__.__name__]

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m, mode='fan_in', nonlinearity='relu')
            elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                constant_init(m, 1)

    def features(self, input):
        conv1_7x7_s2_out = self.conv1_7x7_s2(input)
        conv1_7x7_s2_bn_out = self.conv1_7x7_s2_bn(conv1_7x7_s2_out)
        conv1_relu_7x7_out = self.conv1_relu_7x7(conv1_7x7_s2_bn_out)
        pool1_3x3_s2_out = self.pool1_3x3_s2(conv1_relu_7x7_out)
        conv2_3x3_reduce_out = self.conv2_3x3_reduce(pool1_3x3_s2_out)
        conv2_3x3_reduce_bn_out = self.conv2_3x3_reduce_bn(conv2_3x3_reduce_out)
        conv2_relu_3x3_reduce_out = self.conv2_relu_3x3_reduce(conv2_3x3_reduce_bn_out)
        conv2_3x3_out = self.conv2_3x3(conv2_relu_3x3_reduce_out)
        conv2_3x3_bn_out = self.conv2_3x3_bn(conv2_3x3_out)
        conv2_relu_3x3_out = self.conv2_relu_3x3(conv2_3x3_bn_out)
        pool2_3x3_s2_out = self.pool2_3x3_s2(conv2_relu_3x3_out)
        inception_3a_1x1_out = self.inception_3a_1x1(pool2_3x3_s2_out)
        inception_3a_1x1_bn_out = self.inception_3a_1x1_bn(inception_3a_1x1_out)
        inception_3a_relu_1x1_out = self.inception_3a_relu_1x1(inception_3a_1x1_bn_out)
        inception_3a_3x3_reduce_out = self.inception_3a_3x3_reduce(pool2_3x3_s2_out)
        inception_3a_3x3_reduce_bn_out = self.inception_3a_3x3_reduce_bn(inception_3a_3x3_reduce_out)
        inception_3a_relu_3x3_reduce_out = self.inception_3a_relu_3x3_reduce(inception_3a_3x3_reduce_bn_out)
        inception_3a_3x3_out = self.inception_3a_3x3(inception_3a_relu_3x3_reduce_out)
        inception_3a_3x3_bn_out = self.inception_3a_3x3_bn(inception_3a_3x3_out)
        inception_3a_relu_3x3_out = self.inception_3a_relu_3x3(inception_3a_3x3_bn_out)
        inception_3a_double_3x3_reduce_out = self.inception_3a_double_3x3_reduce(pool2_3x3_s2_out)
        inception_3a_double_3x3_reduce_bn_out = self.inception_3a_double_3x3_reduce_bn(inception_3a_double_3x3_reduce_out)
        inception_3a_relu_double_3x3_reduce_out = self.inception_3a_relu_double_3x3_reduce(inception_3a_double_3x3_reduce_bn_out)
        inception_3a_double_3x3_1_out = self.inception_3a_double_3x3_1(inception_3a_relu_double_3x3_reduce_out)
        inception_3a_double_3x3_1_bn_out = self.inception_3a_double_3x3_1_bn(inception_3a_double_3x3_1_out)
        inception_3a_relu_double_3x3_1_out = self.inception_3a_relu_double_3x3_1(inception_3a_double_3x3_1_bn_out)
        inception_3a_double_3x3_2_out = self.inception_3a_double_3x3_2(inception_3a_relu_double_3x3_1_out)
        inception_3a_double_3x3_2_bn_out = self.inception_3a_double_3x3_2_bn(inception_3a_double_3x3_2_out)
        inception_3a_relu_double_3x3_2_out = self.inception_3a_relu_double_3x3_2(inception_3a_double_3x3_2_bn_out)
        inception_3a_pool_out = self.inception_3a_pool(pool2_3x3_s2_out)
        inception_3a_pool_proj_out = self.inception_3a_pool_proj(inception_3a_pool_out)
        inception_3a_pool_proj_bn_out = self.inception_3a_pool_proj_bn(inception_3a_pool_proj_out)
        inception_3a_relu_pool_proj_out = self.inception_3a_relu_pool_proj(inception_3a_pool_proj_bn_out)
        inception_3a_output_out = torch.cat([inception_3a_relu_1x1_out, inception_3a_relu_3x3_out, inception_3a_relu_double_3x3_2_out, inception_3a_relu_pool_proj_out], 1)
        inception_3b_1x1_out = self.inception_3b_1x1(inception_3a_output_out)
        inception_3b_1x1_bn_out = self.inception_3b_1x1_bn(inception_3b_1x1_out)
        inception_3b_relu_1x1_out = self.inception_3b_relu_1x1(inception_3b_1x1_bn_out)
        inception_3b_3x3_reduce_out = self.inception_3b_3x3_reduce(inception_3a_output_out)
        inception_3b_3x3_reduce_bn_out = self.inception_3b_3x3_reduce_bn(inception_3b_3x3_reduce_out)
        inception_3b_relu_3x3_reduce_out = self.inception_3b_relu_3x3_reduce(inception_3b_3x3_reduce_bn_out)
        inception_3b_3x3_out = self.inception_3b_3x3(inception_3b_relu_3x3_reduce_out)
        inception_3b_3x3_bn_out = self.inception_3b_3x3_bn(inception_3b_3x3_out)
        inception_3b_relu_3x3_out = self.inception_3b_relu_3x3(inception_3b_3x3_bn_out)
        inception_3b_double_3x3_reduce_out = self.inception_3b_double_3x3_reduce(inception_3a_output_out)
        inception_3b_double_3x3_reduce_bn_out = self.inception_3b_double_3x3_reduce_bn(inception_3b_double_3x3_reduce_out)
        inception_3b_relu_double_3x3_reduce_out = self.inception_3b_relu_double_3x3_reduce(inception_3b_double_3x3_reduce_bn_out)
        inception_3b_double_3x3_1_out = self.inception_3b_double_3x3_1(inception_3b_relu_double_3x3_reduce_out)
        inception_3b_double_3x3_1_bn_out = self.inception_3b_double_3x3_1_bn(inception_3b_double_3x3_1_out)
        inception_3b_relu_double_3x3_1_out = self.inception_3b_relu_double_3x3_1(inception_3b_double_3x3_1_bn_out)
        inception_3b_double_3x3_2_out = self.inception_3b_double_3x3_2(inception_3b_relu_double_3x3_1_out)
        inception_3b_double_3x3_2_bn_out = self.inception_3b_double_3x3_2_bn(inception_3b_double_3x3_2_out)
        inception_3b_relu_double_3x3_2_out = self.inception_3b_relu_double_3x3_2(inception_3b_double_3x3_2_bn_out)
        inception_3b_pool_out = self.inception_3b_pool(inception_3a_output_out)
        inception_3b_pool_proj_out = self.inception_3b_pool_proj(inception_3b_pool_out)
        inception_3b_pool_proj_bn_out = self.inception_3b_pool_proj_bn(inception_3b_pool_proj_out)
        inception_3b_relu_pool_proj_out = self.inception_3b_relu_pool_proj(inception_3b_pool_proj_bn_out)
        inception_3b_output_out = torch.cat([inception_3b_relu_1x1_out, inception_3b_relu_3x3_out, inception_3b_relu_double_3x3_2_out, inception_3b_relu_pool_proj_out], 1)
        inception_3c_3x3_reduce_out = self.inception_3c_3x3_reduce(inception_3b_output_out)
        inception_3c_3x3_reduce_bn_out = self.inception_3c_3x3_reduce_bn(inception_3c_3x3_reduce_out)
        inception_3c_relu_3x3_reduce_out = self.inception_3c_relu_3x3_reduce(inception_3c_3x3_reduce_bn_out)
        inception_3c_3x3_out = self.inception_3c_3x3(inception_3c_relu_3x3_reduce_out)
        inception_3c_3x3_bn_out = self.inception_3c_3x3_bn(inception_3c_3x3_out)
        inception_3c_relu_3x3_out = self.inception_3c_relu_3x3(inception_3c_3x3_bn_out)
        inception_3c_double_3x3_reduce_out = self.inception_3c_double_3x3_reduce(inception_3b_output_out)
        inception_3c_double_3x3_reduce_bn_out = self.inception_3c_double_3x3_reduce_bn(inception_3c_double_3x3_reduce_out)
        inception_3c_relu_double_3x3_reduce_out = self.inception_3c_relu_double_3x3_reduce(inception_3c_double_3x3_reduce_bn_out)
        inception_3c_double_3x3_1_out = self.inception_3c_double_3x3_1(inception_3c_relu_double_3x3_reduce_out)
        inception_3c_double_3x3_1_bn_out = self.inception_3c_double_3x3_1_bn(inception_3c_double_3x3_1_out)
        inception_3c_relu_double_3x3_1_out = self.inception_3c_relu_double_3x3_1(inception_3c_double_3x3_1_bn_out)
        inception_3c_double_3x3_2_out = self.inception_3c_double_3x3_2(inception_3c_relu_double_3x3_1_out)
        inception_3c_double_3x3_2_bn_out = self.inception_3c_double_3x3_2_bn(inception_3c_double_3x3_2_out)
        inception_3c_relu_double_3x3_2_out = self.inception_3c_relu_double_3x3_2(inception_3c_double_3x3_2_bn_out)
        inception_3c_pool_out = self.inception_3c_pool(inception_3b_output_out)
        inception_3c_output_out = torch.cat([inception_3c_relu_3x3_out, inception_3c_relu_double_3x3_2_out, inception_3c_pool_out], 1)
        inception_4a_1x1_out = self.inception_4a_1x1(inception_3c_output_out)
        inception_4a_1x1_bn_out = self.inception_4a_1x1_bn(inception_4a_1x1_out)
        inception_4a_relu_1x1_out = self.inception_4a_relu_1x1(inception_4a_1x1_bn_out)
        inception_4a_3x3_reduce_out = self.inception_4a_3x3_reduce(inception_3c_output_out)
        inception_4a_3x3_reduce_bn_out = self.inception_4a_3x3_reduce_bn(inception_4a_3x3_reduce_out)
        inception_4a_relu_3x3_reduce_out = self.inception_4a_relu_3x3_reduce(inception_4a_3x3_reduce_bn_out)
        inception_4a_3x3_out = self.inception_4a_3x3(inception_4a_relu_3x3_reduce_out)
        inception_4a_3x3_bn_out = self.inception_4a_3x3_bn(inception_4a_3x3_out)
        inception_4a_relu_3x3_out = self.inception_4a_relu_3x3(inception_4a_3x3_bn_out)
        inception_4a_double_3x3_reduce_out = self.inception_4a_double_3x3_reduce(inception_3c_output_out)
        inception_4a_double_3x3_reduce_bn_out = self.inception_4a_double_3x3_reduce_bn(inception_4a_double_3x3_reduce_out)
        inception_4a_relu_double_3x3_reduce_out = self.inception_4a_relu_double_3x3_reduce(inception_4a_double_3x3_reduce_bn_out)
        inception_4a_double_3x3_1_out = self.inception_4a_double_3x3_1(inception_4a_relu_double_3x3_reduce_out)
        inception_4a_double_3x3_1_bn_out = self.inception_4a_double_3x3_1_bn(inception_4a_double_3x3_1_out)
        inception_4a_relu_double_3x3_1_out = self.inception_4a_relu_double_3x3_1(inception_4a_double_3x3_1_bn_out)
        inception_4a_double_3x3_2_out = self.inception_4a_double_3x3_2(inception_4a_relu_double_3x3_1_out)
        inception_4a_double_3x3_2_bn_out = self.inception_4a_double_3x3_2_bn(inception_4a_double_3x3_2_out)
        inception_4a_relu_double_3x3_2_out = self.inception_4a_relu_double_3x3_2(inception_4a_double_3x3_2_bn_out)
        inception_4a_pool_out = self.inception_4a_pool(inception_3c_output_out)
        inception_4a_pool_proj_out = self.inception_4a_pool_proj(inception_4a_pool_out)
        inception_4a_pool_proj_bn_out = self.inception_4a_pool_proj_bn(inception_4a_pool_proj_out)
        inception_4a_relu_pool_proj_out = self.inception_4a_relu_pool_proj(inception_4a_pool_proj_bn_out)
        inception_4a_output_out = torch.cat([inception_4a_relu_1x1_out, inception_4a_relu_3x3_out, inception_4a_relu_double_3x3_2_out, inception_4a_relu_pool_proj_out], 1)
        inception_4b_1x1_out = self.inception_4b_1x1(inception_4a_output_out)
        inception_4b_1x1_bn_out = self.inception_4b_1x1_bn(inception_4b_1x1_out)
        inception_4b_relu_1x1_out = self.inception_4b_relu_1x1(inception_4b_1x1_bn_out)
        inception_4b_3x3_reduce_out = self.inception_4b_3x3_reduce(inception_4a_output_out)
        inception_4b_3x3_reduce_bn_out = self.inception_4b_3x3_reduce_bn(inception_4b_3x3_reduce_out)
        inception_4b_relu_3x3_reduce_out = self.inception_4b_relu_3x3_reduce(inception_4b_3x3_reduce_bn_out)
        inception_4b_3x3_out = self.inception_4b_3x3(inception_4b_relu_3x3_reduce_out)
        inception_4b_3x3_bn_out = self.inception_4b_3x3_bn(inception_4b_3x3_out)
        inception_4b_relu_3x3_out = self.inception_4b_relu_3x3(inception_4b_3x3_bn_out)
        inception_4b_double_3x3_reduce_out = self.inception_4b_double_3x3_reduce(inception_4a_output_out)
        inception_4b_double_3x3_reduce_bn_out = self.inception_4b_double_3x3_reduce_bn(inception_4b_double_3x3_reduce_out)
        inception_4b_relu_double_3x3_reduce_out = self.inception_4b_relu_double_3x3_reduce(inception_4b_double_3x3_reduce_bn_out)
        inception_4b_double_3x3_1_out = self.inception_4b_double_3x3_1(inception_4b_relu_double_3x3_reduce_out)
        inception_4b_double_3x3_1_bn_out = self.inception_4b_double_3x3_1_bn(inception_4b_double_3x3_1_out)
        inception_4b_relu_double_3x3_1_out = self.inception_4b_relu_double_3x3_1(inception_4b_double_3x3_1_bn_out)
        inception_4b_double_3x3_2_out = self.inception_4b_double_3x3_2(inception_4b_relu_double_3x3_1_out)
        inception_4b_double_3x3_2_bn_out = self.inception_4b_double_3x3_2_bn(inception_4b_double_3x3_2_out)
        inception_4b_relu_double_3x3_2_out = self.inception_4b_relu_double_3x3_2(inception_4b_double_3x3_2_bn_out)
        inception_4b_pool_out = self.inception_4b_pool(inception_4a_output_out)
        inception_4b_pool_proj_out = self.inception_4b_pool_proj(inception_4b_pool_out)
        inception_4b_pool_proj_bn_out = self.inception_4b_pool_proj_bn(inception_4b_pool_proj_out)
        inception_4b_relu_pool_proj_out = self.inception_4b_relu_pool_proj(inception_4b_pool_proj_bn_out)
        inception_4b_output_out = torch.cat([inception_4b_relu_1x1_out, inception_4b_relu_3x3_out, inception_4b_relu_double_3x3_2_out, inception_4b_relu_pool_proj_out], 1)
        inception_4c_1x1_out = self.inception_4c_1x1(inception_4b_output_out)
        inception_4c_1x1_bn_out = self.inception_4c_1x1_bn(inception_4c_1x1_out)
        inception_4c_relu_1x1_out = self.inception_4c_relu_1x1(inception_4c_1x1_bn_out)
        inception_4c_3x3_reduce_out = self.inception_4c_3x3_reduce(inception_4b_output_out)
        inception_4c_3x3_reduce_bn_out = self.inception_4c_3x3_reduce_bn(inception_4c_3x3_reduce_out)
        inception_4c_relu_3x3_reduce_out = self.inception_4c_relu_3x3_reduce(inception_4c_3x3_reduce_bn_out)
        inception_4c_3x3_out = self.inception_4c_3x3(inception_4c_relu_3x3_reduce_out)
        inception_4c_3x3_bn_out = self.inception_4c_3x3_bn(inception_4c_3x3_out)
        inception_4c_relu_3x3_out = self.inception_4c_relu_3x3(inception_4c_3x3_bn_out)
        inception_4c_double_3x3_reduce_out = self.inception_4c_double_3x3_reduce(inception_4b_output_out)
        inception_4c_double_3x3_reduce_bn_out = self.inception_4c_double_3x3_reduce_bn(inception_4c_double_3x3_reduce_out)
        inception_4c_relu_double_3x3_reduce_out = self.inception_4c_relu_double_3x3_reduce(inception_4c_double_3x3_reduce_bn_out)
        inception_4c_double_3x3_1_out = self.inception_4c_double_3x3_1(inception_4c_relu_double_3x3_reduce_out)
        inception_4c_double_3x3_1_bn_out = self.inception_4c_double_3x3_1_bn(inception_4c_double_3x3_1_out)
        inception_4c_relu_double_3x3_1_out = self.inception_4c_relu_double_3x3_1(inception_4c_double_3x3_1_bn_out)
        inception_4c_double_3x3_2_out = self.inception_4c_double_3x3_2(inception_4c_relu_double_3x3_1_out)
        inception_4c_double_3x3_2_bn_out = self.inception_4c_double_3x3_2_bn(inception_4c_double_3x3_2_out)
        inception_4c_relu_double_3x3_2_out = self.inception_4c_relu_double_3x3_2(inception_4c_double_3x3_2_bn_out)
        inception_4c_pool_out = self.inception_4c_pool(inception_4b_output_out)
        inception_4c_pool_proj_out = self.inception_4c_pool_proj(inception_4c_pool_out)
        inception_4c_pool_proj_bn_out = self.inception_4c_pool_proj_bn(inception_4c_pool_proj_out)
        inception_4c_relu_pool_proj_out = self.inception_4c_relu_pool_proj(inception_4c_pool_proj_bn_out)
        inception_4c_output_out = torch.cat([inception_4c_relu_1x1_out, inception_4c_relu_3x3_out, inception_4c_relu_double_3x3_2_out, inception_4c_relu_pool_proj_out], 1)
        inception_4d_1x1_out = self.inception_4d_1x1(inception_4c_output_out)
        inception_4d_1x1_bn_out = self.inception_4d_1x1_bn(inception_4d_1x1_out)
        inception_4d_relu_1x1_out = self.inception_4d_relu_1x1(inception_4d_1x1_bn_out)
        inception_4d_3x3_reduce_out = self.inception_4d_3x3_reduce(inception_4c_output_out)
        inception_4d_3x3_reduce_bn_out = self.inception_4d_3x3_reduce_bn(inception_4d_3x3_reduce_out)
        inception_4d_relu_3x3_reduce_out = self.inception_4d_relu_3x3_reduce(inception_4d_3x3_reduce_bn_out)
        inception_4d_3x3_out = self.inception_4d_3x3(inception_4d_relu_3x3_reduce_out)
        inception_4d_3x3_bn_out = self.inception_4d_3x3_bn(inception_4d_3x3_out)
        inception_4d_relu_3x3_out = self.inception_4d_relu_3x3(inception_4d_3x3_bn_out)
        inception_4d_double_3x3_reduce_out = self.inception_4d_double_3x3_reduce(inception_4c_output_out)
        inception_4d_double_3x3_reduce_bn_out = self.inception_4d_double_3x3_reduce_bn(inception_4d_double_3x3_reduce_out)
        inception_4d_relu_double_3x3_reduce_out = self.inception_4d_relu_double_3x3_reduce(inception_4d_double_3x3_reduce_bn_out)
        inception_4d_double_3x3_1_out = self.inception_4d_double_3x3_1(inception_4d_relu_double_3x3_reduce_out)
        inception_4d_double_3x3_1_bn_out = self.inception_4d_double_3x3_1_bn(inception_4d_double_3x3_1_out)
        inception_4d_relu_double_3x3_1_out = self.inception_4d_relu_double_3x3_1(inception_4d_double_3x3_1_bn_out)
        inception_4d_double_3x3_2_out = self.inception_4d_double_3x3_2(inception_4d_relu_double_3x3_1_out)
        inception_4d_double_3x3_2_bn_out = self.inception_4d_double_3x3_2_bn(inception_4d_double_3x3_2_out)
        inception_4d_relu_double_3x3_2_out = self.inception_4d_relu_double_3x3_2(inception_4d_double_3x3_2_bn_out)
        inception_4d_pool_out = self.inception_4d_pool(inception_4c_output_out)
        inception_4d_pool_proj_out = self.inception_4d_pool_proj(inception_4d_pool_out)
        inception_4d_pool_proj_bn_out = self.inception_4d_pool_proj_bn(inception_4d_pool_proj_out)
        inception_4d_relu_pool_proj_out = self.inception_4d_relu_pool_proj(inception_4d_pool_proj_bn_out)
        inception_4d_output_out = torch.cat([inception_4d_relu_1x1_out, inception_4d_relu_3x3_out, inception_4d_relu_double_3x3_2_out, inception_4d_relu_pool_proj_out], 1)
        inception_4e_3x3_reduce_out = self.inception_4e_3x3_reduce(inception_4d_output_out)
        inception_4e_3x3_reduce_bn_out = self.inception_4e_3x3_reduce_bn(inception_4e_3x3_reduce_out)
        inception_4e_relu_3x3_reduce_out = self.inception_4e_relu_3x3_reduce(inception_4e_3x3_reduce_bn_out)
        inception_4e_3x3_out = self.inception_4e_3x3(inception_4e_relu_3x3_reduce_out)
        inception_4e_3x3_bn_out = self.inception_4e_3x3_bn(inception_4e_3x3_out)
        inception_4e_relu_3x3_out = self.inception_4e_relu_3x3(inception_4e_3x3_bn_out)
        inception_4e_double_3x3_reduce_out = self.inception_4e_double_3x3_reduce(inception_4d_output_out)
        inception_4e_double_3x3_reduce_bn_out = self.inception_4e_double_3x3_reduce_bn(inception_4e_double_3x3_reduce_out)
        inception_4e_relu_double_3x3_reduce_out = self.inception_4e_relu_double_3x3_reduce(inception_4e_double_3x3_reduce_bn_out)
        inception_4e_double_3x3_1_out = self.inception_4e_double_3x3_1(inception_4e_relu_double_3x3_reduce_out)
        inception_4e_double_3x3_1_bn_out = self.inception_4e_double_3x3_1_bn(inception_4e_double_3x3_1_out)
        inception_4e_relu_double_3x3_1_out = self.inception_4e_relu_double_3x3_1(inception_4e_double_3x3_1_bn_out)
        inception_4e_double_3x3_2_out = self.inception_4e_double_3x3_2(inception_4e_relu_double_3x3_1_out)
        inception_4e_double_3x3_2_bn_out = self.inception_4e_double_3x3_2_bn(inception_4e_double_3x3_2_out)
        inception_4e_relu_double_3x3_2_out = self.inception_4e_relu_double_3x3_2(inception_4e_double_3x3_2_bn_out)
        inception_4e_pool_out = self.inception_4e_pool(inception_4d_output_out)
        inception_4e_output_out = torch.cat([inception_4e_relu_3x3_out, inception_4e_relu_double_3x3_2_out, inception_4e_pool_out], 1)
        inception_5a_1x1_out = self.inception_5a_1x1(inception_4e_output_out)
        inception_5a_1x1_bn_out = self.inception_5a_1x1_bn(inception_5a_1x1_out)
        inception_5a_relu_1x1_out = self.inception_5a_relu_1x1(inception_5a_1x1_bn_out)
        inception_5a_3x3_reduce_out = self.inception_5a_3x3_reduce(inception_4e_output_out)
        inception_5a_3x3_reduce_bn_out = self.inception_5a_3x3_reduce_bn(inception_5a_3x3_reduce_out)
        inception_5a_relu_3x3_reduce_out = self.inception_5a_relu_3x3_reduce(inception_5a_3x3_reduce_bn_out)
        inception_5a_3x3_out = self.inception_5a_3x3(inception_5a_relu_3x3_reduce_out)
        inception_5a_3x3_bn_out = self.inception_5a_3x3_bn(inception_5a_3x3_out)
        inception_5a_relu_3x3_out = self.inception_5a_relu_3x3(inception_5a_3x3_bn_out)
        inception_5a_double_3x3_reduce_out = self.inception_5a_double_3x3_reduce(inception_4e_output_out)
        inception_5a_double_3x3_reduce_bn_out = self.inception_5a_double_3x3_reduce_bn(inception_5a_double_3x3_reduce_out)
        inception_5a_relu_double_3x3_reduce_out = self.inception_5a_relu_double_3x3_reduce(inception_5a_double_3x3_reduce_bn_out)
        inception_5a_double_3x3_1_out = self.inception_5a_double_3x3_1(inception_5a_relu_double_3x3_reduce_out)
        inception_5a_double_3x3_1_bn_out = self.inception_5a_double_3x3_1_bn(inception_5a_double_3x3_1_out)
        inception_5a_relu_double_3x3_1_out = self.inception_5a_relu_double_3x3_1(inception_5a_double_3x3_1_bn_out)
        inception_5a_double_3x3_2_out = self.inception_5a_double_3x3_2(inception_5a_relu_double_3x3_1_out)
        inception_5a_double_3x3_2_bn_out = self.inception_5a_double_3x3_2_bn(inception_5a_double_3x3_2_out)
        inception_5a_relu_double_3x3_2_out = self.inception_5a_relu_double_3x3_2(inception_5a_double_3x3_2_bn_out)
        inception_5a_pool_out = self.inception_5a_pool(inception_4e_output_out)
        inception_5a_pool_proj_out = self.inception_5a_pool_proj(inception_5a_pool_out)
        inception_5a_pool_proj_bn_out = self.inception_5a_pool_proj_bn(inception_5a_pool_proj_out)
        inception_5a_relu_pool_proj_out = self.inception_5a_relu_pool_proj(inception_5a_pool_proj_bn_out)
        inception_5a_output_out = torch.cat([inception_5a_relu_1x1_out, inception_5a_relu_3x3_out, inception_5a_relu_double_3x3_2_out, inception_5a_relu_pool_proj_out], 1)
        inception_5b_1x1_out = self.inception_5b_1x1(inception_5a_output_out)
        inception_5b_1x1_bn_out = self.inception_5b_1x1_bn(inception_5b_1x1_out)
        inception_5b_relu_1x1_out = self.inception_5b_relu_1x1(inception_5b_1x1_bn_out)
        inception_5b_3x3_reduce_out = self.inception_5b_3x3_reduce(inception_5a_output_out)
        inception_5b_3x3_reduce_bn_out = self.inception_5b_3x3_reduce_bn(inception_5b_3x3_reduce_out)
        inception_5b_relu_3x3_reduce_out = self.inception_5b_relu_3x3_reduce(inception_5b_3x3_reduce_bn_out)
        inception_5b_3x3_out = self.inception_5b_3x3(inception_5b_relu_3x3_reduce_out)
        inception_5b_3x3_bn_out = self.inception_5b_3x3_bn(inception_5b_3x3_out)
        inception_5b_relu_3x3_out = self.inception_5b_relu_3x3(inception_5b_3x3_bn_out)
        inception_5b_double_3x3_reduce_out = self.inception_5b_double_3x3_reduce(inception_5a_output_out)
        inception_5b_double_3x3_reduce_bn_out = self.inception_5b_double_3x3_reduce_bn(inception_5b_double_3x3_reduce_out)
        inception_5b_relu_double_3x3_reduce_out = self.inception_5b_relu_double_3x3_reduce(inception_5b_double_3x3_reduce_bn_out)
        inception_5b_double_3x3_1_out = self.inception_5b_double_3x3_1(inception_5b_relu_double_3x3_reduce_out)
        inception_5b_double_3x3_1_bn_out = self.inception_5b_double_3x3_1_bn(inception_5b_double_3x3_1_out)
        inception_5b_relu_double_3x3_1_out = self.inception_5b_relu_double_3x3_1(inception_5b_double_3x3_1_bn_out)
        inception_5b_double_3x3_2_out = self.inception_5b_double_3x3_2(inception_5b_relu_double_3x3_1_out)
        inception_5b_double_3x3_2_bn_out = self.inception_5b_double_3x3_2_bn(inception_5b_double_3x3_2_out)
        inception_5b_relu_double_3x3_2_out = self.inception_5b_relu_double_3x3_2(inception_5b_double_3x3_2_bn_out)
        inception_5b_pool_out = self.inception_5b_pool(inception_5a_output_out)
        inception_5b_pool_proj_out = self.inception_5b_pool_proj(inception_5b_pool_out)
        inception_5b_pool_proj_bn_out = self.inception_5b_pool_proj_bn(inception_5b_pool_proj_out)
        inception_5b_relu_pool_proj_out = self.inception_5b_relu_pool_proj(inception_5b_pool_proj_bn_out)
        inception_5b_output_out = torch.cat([inception_5b_relu_1x1_out, inception_5b_relu_3x3_out, inception_5b_relu_double_3x3_2_out, inception_5b_relu_pool_proj_out], 1)
        return inception_5b_output_out

    def logits(self, features):
        adaptiveAvgPoolWidth = features.shape[2]
        x = F.avg_pool2d(features, kernel_size=adaptiveAvgPoolWidth)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

    def forward(self, input):
        x = self.features(input)
        if self.num_classes > 0:
            x = self.logits(x)
        return [x]


class BertLayerNorm(nn.Module):

    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
    """
        super(BertLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
  """

    def __init__(self, config):
        super(BertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size, padding_idx=0)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size, padding_idx=0)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertPooler(nn.Module):

    def __init__(self, config):
        super(BertPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


ACT2FN = {'gelu': nn.GELU(), 'relu': torch.nn.functional.relu}


class BertIntermediate(nn.Module):

    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str) or sys.version_info[0] == 2 and isinstance(config.hidden_act, unicode):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):

    def __init__(self, config):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class VisualInputEmbedding(nn.Module):
    """
    Takes input of both image and video (multi-frame)
    """

    def __init__(self, config):
        super(VisualInputEmbedding, self).__init__()
        self.config = config
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.row_position_embeddings = nn.Embedding(config.max_grid_row_position_embeddings, config.hidden_size)
        self.col_position_embeddings = nn.Embedding(config.max_grid_col_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(1, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, grid):
        """
        Args:
            grid: (B, n_frm, H, W, C), note that #frm can be 1

        Returns:

        """
        bsz, _, _, _, hsz = grid.shape
        grid = self.add_2d_positional_embeddings(grid)
        visual_tokens = grid.view(bsz, -1, hsz)
        visual_tokens_shape = visual_tokens.shape[:-1]
        device = visual_tokens.device
        token_type_ids = torch.zeros(visual_tokens_shape, dtype=torch.long, device=device)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = visual_tokens + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

    def add_temporal_postion_embeddings(self, grid):
        """
        Args:
            grid: (B, n_frms, H, W, d)

        Returns:
            (B, n_frms, H, W, d)
        """
        n_frms, height, width, hsz = grid.shape[-4:]
        temporal_position_ids = torch.arange(n_frms, dtype=torch.long, device=grid.device)
        t_position_embeddings = self.temporal_position_embeddings(temporal_position_ids)
        new_shape = 1, n_frms, 1, 1, hsz
        grid = grid + t_position_embeddings.view(*new_shape)
        return grid

    def add_2d_positional_embeddings(self, grid):
        """
        Args:
            grid: (B, *, H, W, d)

        Returns:
            (B, *, H, W, d)
        """
        height, width, hsz = grid.shape[-3:]
        row_position_ids = torch.arange(height, dtype=torch.long, device=grid.device)
        row_position_embeddings = self.row_position_embeddings(row_position_ids)
        row_shape = (1,) * (len(grid.shape) - 3) + (height, 1, hsz)
        grid = grid + row_position_embeddings.view(*row_shape)
        col_position_ids = torch.arange(width, dtype=torch.long, device=grid.device)
        col_position_embeddings = self.col_position_embeddings(col_position_ids)
        col_shape = (1,) * (len(grid.shape) - 3) + (1, width, hsz)
        grid = grid + col_position_embeddings.view(*col_shape)
        return grid


class BertSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, 'embedding_size'):
            raise ValueError('The hidden size (%d) is not a multiple of the number of attention heads (%d)' % (config.hidden_size, config.num_attention_heads))
        self.output_attentions = config.output_attentions
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask=None, head_mask=None, encoder_hidden_states=None, encoder_attention_mask=None):
        mixed_query_layer = self.query(hidden_states)
        if encoder_hidden_states is not None:
            mixed_key_layer = self.key(encoder_hidden_states)
            mixed_value_layer = self.value(encoder_hidden_states)
            attention_mask = encoder_attention_mask
        else:
            mixed_key_layer = self.key(hidden_states)
            mixed_value_layer = self.value(hidden_states)
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)
        if head_mask is not None:
            attention_probs = attention_probs * head_mask
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        outputs = (context_layer, attention_probs) if self.output_attentions else (context_layer,)
        return outputs


class BertSelfOutput(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)
        self.pruned_heads = set()

    def forward(self, hidden_states, attention_mask=None, head_mask=None, encoder_hidden_states=None, encoder_attention_mask=None):
        self_outputs = self.self(hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask)
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]
        return outputs


class BertLayer(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.attention = BertAttention(config)
        self.is_decoder = config.is_decoder
        if self.is_decoder:
            self.crossattention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask=None, head_mask=None, encoder_hidden_states=None, encoder_attention_mask=None):
        self_attention_outputs = self.attention(hidden_states, attention_mask, head_mask)
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]
        if self.is_decoder and encoder_hidden_states is not None:
            cross_attention_outputs = self.crossattention(attention_output, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask)
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:]
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output,) + outputs
        return outputs


class BertEncoder(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask=None, head_mask=None, encoder_hidden_states=None, encoder_attention_mask=None):
        all_hidden_states = ()
        all_attentions = ()
        for i, layer_module in enumerate(self.layer):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            layer_outputs = layer_module(hidden_states, attention_mask, head_mask[i], encoder_hidden_states, encoder_attention_mask)
            hidden_states = layer_outputs[0]
            if self.output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        outputs = hidden_states,
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs


class AdaptivePadding(nn.Module):
    """Applies padding to input (if needed) so that input can get fully covered
    by filter you specified. It support two modes "same" and "corner". The
    "same" mode is same with "SAME" padding mode in TensorFlow, pad zero around
    input. The "corner"  mode would pad zero to bottom right.

    Args:
        kernel_size (int | tuple): Size of the kernel:
        stride (int | tuple): Stride of the filter. Default: 1:
        dilation (int | tuple): Spacing between kernel elements.
            Default: 1.
        padding (str): Support "same" and "corner", "corner" mode
            would pad zero to bottom right, and "same" mode would
            pad zero around input. Default: "corner".
    Example:
        >>> kernel_size = 16
        >>> stride = 16
        >>> dilation = 1
        >>> input = torch.rand(1, 1, 15, 17)
        >>> adap_pad = AdaptivePadding(
        >>>     kernel_size=kernel_size,
        >>>     stride=stride,
        >>>     dilation=dilation,
        >>>     padding="corner")
        >>> out = adap_pad(input)
        >>> assert (out.shape[2], out.shape[3]) == (16, 32)
        >>> input = torch.rand(1, 1, 16, 17)
        >>> out = adap_pad(input)
        >>> assert (out.shape[2], out.shape[3]) == (16, 32)
    """

    def __init__(self, kernel_size=1, stride=1, dilation=1, padding='corner'):
        super(AdaptivePadding, self).__init__()
        assert padding in ('same', 'corner')
        kernel_size = to_2tuple(kernel_size)
        stride = to_2tuple(stride)
        dilation = to_2tuple(dilation)
        self.padding = padding
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation

    def get_pad_shape(self, input_shape):
        input_h, input_w = input_shape
        kernel_h, kernel_w = self.kernel_size
        stride_h, stride_w = self.stride
        output_h = math.ceil(input_h / stride_h)
        output_w = math.ceil(input_w / stride_w)
        pad_h = max((output_h - 1) * stride_h + (kernel_h - 1) * self.dilation[0] + 1 - input_h, 0)
        pad_w = max((output_w - 1) * stride_w + (kernel_w - 1) * self.dilation[1] + 1 - input_w, 0)
        return pad_h, pad_w

    def forward(self, x):
        pad_h, pad_w = self.get_pad_shape(x.size()[-2:])
        if pad_h > 0 or pad_w > 0:
            if self.padding == 'corner':
                x = F.pad(x, [0, pad_w, 0, pad_h])
            elif self.padding == 'same':
                x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2])
        return x


INVALID_KEY = 5194902719927681029


class KeyError(BaseError):
    """Raised when a mapping (dictionary) key is not found in the set of existing keys."""
    code = INVALID_KEY


def conv_ws_2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, eps=1e-05):
    c_in = weight.size(0)
    weight_flat = weight.view(c_in, -1)
    mean = weight_flat.mean(dim=1, keepdim=True).view(c_in, 1, 1, 1)
    std = weight_flat.std(dim=1, keepdim=True).view(c_in, 1, 1, 1)
    weight = (weight - mean) / (std + eps)
    return F.conv2d(input, weight, bias, stride, padding, dilation, groups)


class ConvWS2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, eps=1e-05):
        super(ConvWS2d, self).__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.eps = eps

    def forward(self, x):
        return conv_ws_2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups, self.eps)


class ModulatedDeformConv2dFunction(Function):

    @staticmethod
    def symbolic(g, input, offset, mask, weight, bias, stride, padding, dilation, groups, deform_groups):
        input_tensors = [input, offset, mask, weight]
        if bias is not None:
            input_tensors.append(bias)
        return g.op('mmcv::MMCVModulatedDeformConv2d', *input_tensors, stride_i=stride, padding_i=padding, dilation_i=dilation, groups_i=groups, deform_groups_i=deform_groups)

    @staticmethod
    def _jit_forward(input, offset, mask, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, deform_groups=1):
        if input is not None and input.dim() != 4:
            raise ValueError(f'Expected 4D tensor as input, got {input.dim()}D tensor                   instead.')
        with_bias = bias is not None
        if not bias:
            bias = input.new_empty(0)

        def _output_size(input, weight):
            channels = weight.size(0)
            output_size = input.size(0), channels
            for d in range(input.dim() - 2):
                in_size = input.size(d + 2)
                pad = padding[d]
                kernel = dilation[d] * (weight.size(d + 2) - 1) + 1
                stride_ = stride[d]
                output_size += (in_size + 2 * pad - kernel) // stride_ + 1,
            if not all(map(lambda s: s > 0, output_size)):
                raise ValueError('convolution input is too small (output would be ' + 'x'.join(map(str, output_size)) + ')')
            return output_size
        input = input.type_as(offset)
        weight = weight.type_as(input)
        output = input.new_empty(_output_size(input, weight))
        _bufs = [input.new_empty(0), input.new_empty(0)]
        if weight.dtype == torch.float16:
            output = torch.ops.mmcv.modulated_deform_conv(input, weight, bias, _bufs[0], offset, mask, output, _bufs[1], kernel_h=weight.size(2), kernel_w=weight.size(3), stride_h=stride[0], stride_w=stride[1], pad_h=padding[0], pad_w=padding[1], dilation_h=dilation[0], dilation_w=dilation[1], group=groups, deformable_group=deform_groups, with_bias=with_bias)
            output = output
        else:
            output = torch.ops.mmcv.modulated_deform_conv(input, weight, bias, _bufs[0], offset, mask, output, _bufs[1], kernel_h=weight.size(2), kernel_w=weight.size(3), stride_h=stride[0], stride_w=stride[1], pad_h=padding[0], pad_w=padding[1], dilation_h=dilation[0], dilation_w=dilation[1], group=groups, deformable_group=deform_groups, with_bias=with_bias)
        return output

    @staticmethod
    def forward(ctx, input: 'torch.Tensor', offset: 'torch.Tensor', mask: 'torch.Tensor', weight: 'nn.Parameter', bias: 'Optional[nn.Parameter]'=None, stride: 'int'=1, padding: 'int'=0, dilation: 'int'=1, groups: 'int'=1, deform_groups: 'int'=1) ->torch.Tensor:
        if input is not None and input.dim() != 4:
            raise ValueError(f'Expected 4D tensor as input, got {input.dim()}D tensor                   instead.')
        ctx.stride = _pair(stride)
        ctx.padding = _pair(padding)
        ctx.dilation = _pair(dilation)
        ctx.groups = groups
        ctx.deform_groups = deform_groups
        ctx.with_bias = bias is not None
        if not ctx.with_bias:
            bias = input.new_empty(0)
        input = input.type_as(offset)
        weight = weight.type_as(input)
        bias = bias.type_as(input)
        ctx.save_for_backward(input, offset, mask, weight, bias)
        output = input.new_empty(ModulatedDeformConv2dFunction._output_size(ctx, input, weight))
        ctx._bufs = [input.new_empty(0), input.new_empty(0)]
        ext_module.modulated_deform_conv_forward(input, weight, bias, ctx._bufs[0], offset, mask, output, ctx._bufs[1], kernel_h=weight.size(2), kernel_w=weight.size(3), stride_h=ctx.stride[0], stride_w=ctx.stride[1], pad_h=ctx.padding[0], pad_w=ctx.padding[1], dilation_h=ctx.dilation[0], dilation_w=ctx.dilation[1], group=ctx.groups, deformable_group=ctx.deform_groups, with_bias=ctx.with_bias)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output: 'torch.Tensor') ->tuple:
        input, offset, mask, weight, bias = ctx.saved_tensors
        grad_input = torch.zeros_like(input)
        grad_offset = torch.zeros_like(offset)
        grad_mask = torch.zeros_like(mask)
        grad_weight = torch.zeros_like(weight)
        grad_bias = torch.zeros_like(bias)
        grad_output = grad_output.contiguous()
        ext_module.modulated_deform_conv_backward(input, weight, bias, ctx._bufs[0], offset, mask, ctx._bufs[1], grad_input, grad_weight, grad_bias, grad_offset, grad_mask, grad_output, kernel_h=weight.size(2), kernel_w=weight.size(3), stride_h=ctx.stride[0], stride_w=ctx.stride[1], pad_h=ctx.padding[0], pad_w=ctx.padding[1], dilation_h=ctx.dilation[0], dilation_w=ctx.dilation[1], group=ctx.groups, deformable_group=ctx.deform_groups, with_bias=ctx.with_bias)
        if not ctx.with_bias:
            grad_bias = None
        return grad_input, grad_offset, grad_mask, grad_weight, grad_bias, None, None, None, None, None

    @staticmethod
    def _output_size(ctx, input, weight):
        channels = weight.size(0)
        output_size = input.size(0), channels
        for d in range(input.dim() - 2):
            in_size = input.size(d + 2)
            pad = ctx.padding[d]
            kernel = ctx.dilation[d] * (weight.size(d + 2) - 1) + 1
            stride_ = ctx.stride[d]
            output_size += (in_size + 2 * pad - kernel) // stride_ + 1,
        if not all(map(lambda s: s > 0, output_size)):
            raise ValueError('convolution input is too small (output would be ' + 'x'.join(map(str, output_size)) + ')')
        return output_size


modulated_deform_conv2d = ModulatedDeformConv2dFunction.apply


INVALID_TYPE = 5194902719927681030


class TypeError(BaseError):
    """Raised when an operation or function is applied to an object of inappropriate type."""
    code = INVALID_TYPE


def print_log(msg, logger=None, level=logging.INFO):
    """Print a log message.

    Args:
        msg (str): The message to be logged.
        logger (logging.Logger | str | None): The logger to be used. Some
            special loggers are:
            - "root": the root logger obtained with `get_root_logger()`.
            - "silent": no message will be printed.
            - None: The `print()` method will be used to print log messages.
        level (int): Logging level. Only available when `logger` is a Logger
            object or "root".
    """
    if logger is None:
        None
    elif isinstance(logger, logging.Logger):
        logger.log(level, msg)
    elif logger == 'silent':
        pass
    elif isinstance(logger, str):
        _logger = get_logger(logger)
        _logger.log(level, msg)
    else:
        raise TypeError(f'logger should be either a logging.Logger object, str, "silent" or None, but got {type(logger)}')


def build_conv_layer(cfg, *args, **kwargs):
    """ Build convolution layer

    Args:
        cfg (None or dict): cfg should contain:
            type (str): identify conv layer type.
            layer args: args needed to instantiate a conv layer.

    Returns:
        layer (nn.Module): created conv layer
    """
    if cfg is None:
        cfg_ = dict(type='Conv')
    else:
        assert isinstance(cfg, dict) and 'type' in cfg
        cfg_ = cfg.copy()
    layer_type = cfg_.pop('type')
    if layer_type not in conv_cfg:
        raise KeyError('Unrecognized norm type {}'.format(layer_type))
    else:
        conv_layer = conv_cfg[layer_type]
    layer = conv_layer(*args, **kwargs, **cfg_)
    return layer


class IBN(nn.Module):
    """Instance-Batch Normalization layer from
    `"Two at Once: Enhancing Learning and Generalization Capacities via IBN-Net"
    <https://arxiv.org/pdf/1807.09441.pdf>`
    Args:
        planes (int): Number of channels for the input tensor
        ratio (float): Ratio of instance normalization in the IBN layer
    """

    def __init__(self, planes, ratio=0.5, eps=1e-05):
        super(IBN, self).__init__()
        self.half = int(planes * ratio)
        self.IN = nn.InstanceNorm2d(self.half, affine=True)
        self.BN = nn.BatchNorm2d(planes - self.half, eps=eps)

    def forward(self, x):
        split = torch.split(x, self.half, 1)
        out1 = self.IN(split[0].contiguous())
        out2 = self.BN(split[1].contiguous())
        out = torch.cat((out1, out2), 1)
        return out


class SyncIBN(nn.Module):
    """Instance-Batch Normalization layer from
    `"Two at Once: Enhancing Learning and Generalization Capacities via IBN-Net"
    <https://arxiv.org/pdf/1807.09441.pdf>`
    Args:
        planes (int): Number of channels for the input tensor
        ratio (float): Ratio of instance normalization in the IBN layer
    """

    def __init__(self, planes, ratio=0.5, eps=1e-05):
        super(SyncIBN, self).__init__()
        self.half = int(planes * ratio)
        self.IN = nn.InstanceNorm2d(self.half, affine=True)
        self.BN = nn.SyncBatchNorm(planes - self.half, eps=eps)

    def forward(self, x):
        split = torch.split(x, self.half, 1)
        out1 = self.IN(split[0].contiguous())
        out2 = self.BN(split[1].contiguous())
        out = torch.cat((out1, out2), 1)
        return out


norm_cfg = {'BN': ('bn', nn.BatchNorm2d), 'SyncBN': ('bn', nn.SyncBatchNorm), 'GN': ('gn', nn.GroupNorm), 'IBN': ('ibn', IBN), 'SyncIBN': ('ibn', SyncIBN), 'IN': ('in', nn.InstanceNorm2d), 'LN': ('ln', nn.LayerNorm)}


def build_norm_layer(cfg, num_features, postfix=''):
    """ Build normalization layer

    Args:
        cfg (dict): cfg should contain:
            type (str): identify norm layer type.
            layer args: args needed to instantiate a norm layer.
            requires_grad (bool): [optional] whether stop gradient updates
        num_features (int): number of channels from input.
        postfix (int, str): appended into norm abbreviation to
            create named layer.

    Returns:
        name (str): abbreviation + postfix
        layer (nn.Module): created norm layer
    """
    assert isinstance(cfg, dict) and 'type' in cfg
    cfg_ = cfg.copy()
    layer_type = cfg_.pop('type')
    if layer_type not in norm_cfg:
        raise KeyError('Unrecognized norm type {}'.format(layer_type))
    else:
        abbr, norm_layer = norm_cfg[layer_type]
        if norm_layer is None:
            raise NotImplementedError
    assert isinstance(postfix, (int, str))
    name = abbr + str(postfix)
    requires_grad = cfg_.pop('requires_grad', True)
    cfg_.setdefault('eps', 1e-05)
    if layer_type != 'GN':
        layer = norm_layer(num_features, **cfg_)
        if layer_type == 'SyncBN' and hasattr(layer, '_specify_ddp_gpu_num'):
            layer._specify_ddp_gpu_num(1)
        elif layer_type == 'SyncIBN' and hasattr(layer, '_specify_ddp_gpu_num'):
            layer.BN._specify_ddp_gpu_num(1)
    else:
        assert 'num_groups' in cfg_
        layer = norm_layer(num_channels=num_features, **cfg_)
    for param in layer.parameters():
        param.requires_grad = requires_grad
    return name, layer


class ConvMlp(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def drop_path(x, drop_prob: 'float'=0.0, training: 'bool'=False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self):
        return 'p={}'.format(self.drop_prob)


class ConvBlock(nn.Module):

    def __init__(self, dim, mlp_ratio=4.0, drop=0.0, drop_path=0.0, act_layer=nn.GELU):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.conv1 = nn.Conv2d(dim, dim, 1)
        self.conv2 = nn.Conv2d(dim, dim, 1)
        self.attn = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = ConvMlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, mask=None):
        if mask is not None:
            residual = x
            x = self.conv1(self.norm1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2))
            x1 = self.attn(mask[0] * x)
            x2 = self.attn(mask[1] * x)
            x3 = self.attn(mask[2] * x)
            x4 = self.attn(mask[3] * x)
            x = mask[0] * x1 + mask[1] * x2 + mask[2] * x3 + mask[3] * x4
            x = residual + self.drop_path(self.conv2(x))
        else:
            x = x + self.drop_path(self.conv2(self.attn(self.conv1(self.norm1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)))))
        x = x + self.drop_path(self.mlp(self.norm2(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)))
        return x


class Attention(nn.Module):

    def __init__(self, dim, num_heads=8, mixer='Global', HW=[8, 25], local_k=[7, 11], qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.HW = HW
        if HW is not None:
            H = HW[0]
            W = HW[1]
            self.N = H * W
            self.C = dim
        if mixer == 'Local' and HW is not None:
            hk = local_k[0]
            wk = local_k[1]
            mask = torch.ones(H * W, H + hk - 1, W + wk - 1, dtype=torch.float32)
            for h in range(0, H):
                for w in range(0, W):
                    mask[h * W + w, h:h + hk, w:w + wk] = 0.0
            mask_paddle = mask[:, hk // 2:H + hk // 2, wk // 2:W + wk // 2].flatten(1)
            mask_inf = torch.full([H * W, H * W], fill_value=float('-Inf'), dtype=torch.float32)
            mask = torch.where(mask_paddle < 1, mask_paddle, mask_inf)
            self.mask = mask.unsqueeze(1).unsqueeze(0)
        self.mixer = mixer

    def forward(self, x):
        if self.HW is not None:
            N = self.N
            C = self.C
        else:
            _, N, C = x.shape
        qkv = self.qkv(x).reshape((-1, N, 3, self.num_heads, C // self.num_heads)).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        attn = q.matmul(k.permute(0, 1, 3, 2))
        if self.mixer == 'Local':
            attn += self.mask
        attn = nn.functional.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        x = attn.matmul(v).permute(0, 2, 1, 3).reshape((-1, N, C))
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class ConvMixer(nn.Module):

    def __init__(self, dim, num_heads=8, HW=[8, 25], local_k=[3, 3]):
        super().__init__()
        self.HW = HW
        self.dim = dim
        self.local_mixer = nn.Conv2d(dim, dim, local_k, 1, [local_k[0] // 2, local_k[1] // 2], groups=num_heads)

    def forward(self, x):
        h = self.HW[0]
        w = self.HW[1]
        x = x.transpose([0, 2, 1]).reshape([0, self.dim, h, w])
        x = self.local_mixer(x)
        x = x.flatten(2).permute(0, 2, 1)
        return x


class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, input):
        return input


class Mlp(nn.Module):
    """ Multilayer perceptron.
    Parameters:
        act_layer: Specify the activate function, default use nn.GELU.
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def eval(config_path, checkpoint_path, gpus=1, fp16=False, master_port=29600):
    tpath = config_path
    current_env = os.environ.copy()
    cmd = [sys.executable, '-m', 'torch.distributed.launch']
    cmd.append('--nproc_per_node={}'.format(gpus))
    cmd.append('--master_port={}'.format(master_port))
    eval_script = os.path.join(easycv_root_path, 'tools/eval.py')
    cmd.append(eval_script)
    cmd.append('{}'.format(tpath))
    cmd.append('{}'.format(checkpoint_path))
    cmd.append('--launcher=pytorch')
    cmd.append('--eval')
    if fp16:
        cmd.append('--fp16')
    process = subprocess.Popen(cmd, env=current_env)
    process.wait()
    if process.returncode != 0:
        raise subprocess.CalledProcessError(returncode=process.returncode, cmd=cmd)


class Block(nn.Module):

    def __init__(self, dim, num_heads, mixer='Global', local_mixer=[7, 11], HW=[8, 25], mlp_ratio=4.0, qkv_bias=False, qk_scale=None, drop=0.0, attn_drop=0.0, drop_path=0.0, act_layer='gelu', norm_layer='nn.LayerNorm', epsilon=1e-06, prenorm=True):
        super().__init__()
        if isinstance(norm_layer, str):
            self.norm1 = eval(norm_layer)(dim, eps=epsilon)
        else:
            self.norm1 = norm_layer(dim)
        if mixer == 'Global' or mixer == 'Local':
            self.mixer = Attention(dim, num_heads=num_heads, mixer=mixer, HW=HW, local_k=local_mixer, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        elif mixer == 'Conv':
            self.mixer = ConvMixer(dim, num_heads=num_heads, HW=HW, local_k=local_mixer)
        else:
            raise TypeError('The mixer must be one of [Global, Local, Conv]')
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else Identity()
        if isinstance(norm_layer, str):
            self.norm2 = eval(norm_layer)(dim, eps=epsilon)
        else:
            self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp_ratio = mlp_ratio
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.prenorm = prenorm

    def forward(self, x):
        if self.prenorm:
            x = self.norm1(x + self.drop_path(self.mixer(x)))
            x = self.norm2(x + self.drop_path(self.mlp(x)))
        else:
            x = x + self.drop_path(self.mixer(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000 ** omega
    pos = pos.reshape(-1)
    out = np.einsum('m,d->md', pos, omega)
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    emb = np.concatenate([emb_sin, emb_cos], axis=1)
    return emb


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    emb = np.concatenate([emb_h, emb_w], axis=1)
    return emb


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def _no_grad_trunc_normal_(tensor, mean, std, a, b):

    def norm_cdf(x):
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0
    if mean < a - 2 * std or mean > b + 2 * std:
        logging.warning('mean is more than 2 std from [a, b] in nn.init.trunc_normal_. The distribution of values may be incorrect.', stacklevel=2)
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


class FastConvMAEViT(nn.Module):
    """ Fast ConvMAE framework is a superiorly fast masked modeling scheme via
    complementary masking and mixture of reconstrunctors based on the ConvMAE(https://arxiv.org/abs/2205.03892).

    Args:
        img_size (list | tuple): Input image size for three stages.
        patch_size (list | tuple): The patch size for three stages.
        in_channels (int): The num of input channels. Default: 3
        embed_dim (list | tuple): The dimensions of embedding for three stages.
        depth (list | tuple): depth for three stages.
        num_heads (int): Parallel attention heads
        mlp_ratio (list | tuple): Mlp expansion ratio.
        drop_rate (float): Probability of an element to be zeroed
            after the feed forward layer. Defaults to 0.
        drop_path_rate (float): Stochastic depth rate. Defaults to 0.
        norm_layer (nn.Module): normalization layer
        init_pos_embed_by_sincos: initialize pos_embed by sincos strategy
        with_fuse(bool): Whether to use fuse layers.
        global_pool: global pool

    """

    def __init__(self, img_size=[224, 56, 28], patch_size=[4, 2, 2], in_channels=3, embed_dim=[256, 384, 768], depth=[2, 2, 11], num_heads=12, mlp_ratio=[4, 4, 4], drop_rate=0.0, drop_path_rate=0.1, norm_layer=partial(nn.LayerNorm, eps=1e-06), init_pos_embed_by_sincos=True, with_fuse=True, global_pool=False):
        super().__init__()
        self.init_pos_embed_by_sincos = init_pos_embed_by_sincos
        self.with_fuse = with_fuse
        self.global_pool = global_pool
        assert len(img_size) == len(patch_size) == len(embed_dim) == len(mlp_ratio)
        self.patch_size = patch_size[0] * patch_size[1] * patch_size[2]
        self.patch_embed1 = PatchEmbed(img_size=img_size[0], patch_size=patch_size[0], in_channels=in_channels, embed_dim=embed_dim[0])
        self.patch_embed2 = PatchEmbed(img_size=img_size[1], patch_size=patch_size[1], in_channels=embed_dim[0], embed_dim=embed_dim[1])
        self.patch_embed3 = PatchEmbed(img_size=img_size[2], patch_size=patch_size[2], in_channels=embed_dim[1], embed_dim=embed_dim[2])
        self.patch_embed4 = nn.Linear(embed_dim[2], embed_dim[2])
        if with_fuse:
            self._make_fuse_layers(embed_dim)
        self.num_patches = self.patch_embed3.num_patches
        if init_pos_embed_by_sincos:
            self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim[2]), requires_grad=False)
        else:
            self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim[2]))
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = np.linspace(0, drop_path_rate, sum(depth))
        self.blocks1 = nn.ModuleList([ConvBlock(dim=embed_dim[0], mlp_ratio=mlp_ratio[0], drop=drop_rate, drop_path=dpr[i]) for i in range(depth[0])])
        self.blocks2 = nn.ModuleList([ConvBlock(dim=embed_dim[1], mlp_ratio=mlp_ratio[1], drop=drop_rate, drop_path=dpr[depth[0] + i]) for i in range(depth[1])])
        self.blocks3 = nn.ModuleList([Block(dim=embed_dim[2], num_heads=num_heads, mlp_ratio=mlp_ratio[2], qkv_bias=True, qk_scale=None, drop=drop_rate, drop_path=dpr[depth[0] + depth[1] + i], norm_layer=norm_layer) for i in range(depth[2])])
        if self.global_pool:
            self.fc_norm = norm_layer(embed_dim[-1])
            self.norm = None
        else:
            self.norm = norm_layer(embed_dim[-1])
            self.fc_norm = None

    def init_weights(self):
        if self.init_pos_embed_by_sincos:
            pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.num_patches ** 0.5), cls_token=False)
            self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        else:
            trunc_normal_(self.pos_embed, std=0.02)
        w = self.patch_embed3.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _make_fuse_layers(self, embed_dim):
        self.stage1_output_decode = nn.Conv2d(embed_dim[0], embed_dim[2], 4, stride=4)
        self.stage2_output_decode = nn.Conv2d(embed_dim[1], embed_dim[2], 2, stride=2)

    def random_masking(self, x, mask_ratio=None):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N = x.shape[0]
        L = self.num_patches
        len_keep = int(L * (1 - mask_ratio))
        noise = torch.rand(N, L, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep1 = ids_shuffle[:, :len_keep]
        ids_keep2 = ids_shuffle[:, len_keep:2 * len_keep]
        ids_keep3 = ids_shuffle[:, 2 * len_keep:3 * len_keep]
        ids_keep4 = ids_shuffle[:, 3 * len_keep:]
        mask1 = torch.ones([N, L], device=x.device)
        mask1[:, :len_keep] = 0
        mask1 = torch.gather(mask1, dim=1, index=ids_restore)
        mask2 = torch.ones([N, L], device=x.device)
        mask2[:, len_keep:2 * len_keep] = 0
        mask2 = torch.gather(mask2, dim=1, index=ids_restore)
        mask3 = torch.ones([N, L], device=x.device)
        mask3[:, 2 * len_keep:3 * len_keep] = 0
        mask3 = torch.gather(mask3, dim=1, index=ids_restore)
        mask4 = torch.ones([N, L], device=x.device)
        mask4[:, 3 * len_keep:4 * len_keep] = 0
        mask4 = torch.gather(mask4, dim=1, index=ids_restore)
        return [ids_keep1, ids_keep2, ids_keep3, ids_keep4], [mask1, mask2, mask3, mask4], ids_restore

    def _fuse_forward(self, s1, s2, ids_keep=None, mask_ratio=None):
        stage1_embed = self.stage1_output_decode(s1).flatten(2).permute(0, 2, 1)
        stage2_embed = self.stage2_output_decode(s2).flatten(2).permute(0, 2, 1)
        if mask_ratio is not None:
            stage1_embed_1 = torch.gather(stage1_embed, dim=1, index=ids_keep[0].unsqueeze(-1).repeat(1, 1, stage1_embed.shape[-1]))
            stage2_embed_1 = torch.gather(stage2_embed, dim=1, index=ids_keep[0].unsqueeze(-1).repeat(1, 1, stage2_embed.shape[-1]))
            stage1_embed_2 = torch.gather(stage1_embed, dim=1, index=ids_keep[1].unsqueeze(-1).repeat(1, 1, stage1_embed.shape[-1]))
            stage2_embed_2 = torch.gather(stage2_embed, dim=1, index=ids_keep[1].unsqueeze(-1).repeat(1, 1, stage2_embed.shape[-1]))
            stage1_embed_3 = torch.gather(stage1_embed, dim=1, index=ids_keep[2].unsqueeze(-1).repeat(1, 1, stage1_embed.shape[-1]))
            stage2_embed_3 = torch.gather(stage2_embed, dim=1, index=ids_keep[2].unsqueeze(-1).repeat(1, 1, stage2_embed.shape[-1]))
            stage1_embed_4 = torch.gather(stage1_embed, dim=1, index=ids_keep[3].unsqueeze(-1).repeat(1, 1, stage1_embed.shape[-1]))
            stage2_embed_4 = torch.gather(stage2_embed, dim=1, index=ids_keep[3].unsqueeze(-1).repeat(1, 1, stage2_embed.shape[-1]))
            stage1_embed = torch.cat([stage1_embed_1, stage1_embed_2, stage1_embed_3, stage1_embed_4])
            stage2_embed = torch.cat([stage2_embed_1, stage2_embed_2, stage2_embed_3, stage2_embed_4])
        return stage1_embed, stage2_embed

    def forward(self, x, mask_ratio=None):
        if mask_ratio is not None:
            assert self.with_fuse
        if mask_ratio is not None:
            ids_keep, masks, ids_restore = self.random_masking(x, mask_ratio)
            mask_for_patch1 = [(1 - mask.reshape(-1, 14, 14).unsqueeze(-1).repeat(1, 1, 1, 16).reshape(-1, 14, 14, 4, 4).permute(0, 1, 3, 2, 4).reshape(x.shape[0], 56, 56).unsqueeze(1)) for mask in masks]
            mask_for_patch2 = [(1 - mask.reshape(-1, 14, 14).unsqueeze(-1).repeat(1, 1, 1, 4).reshape(-1, 14, 14, 2, 2).permute(0, 1, 3, 2, 4).reshape(x.shape[0], 28, 28).unsqueeze(1)) for mask in masks]
        else:
            mask_for_patch1 = None
            mask_for_patch2 = None
        s1 = self.patch_embed1(x)
        s1 = self.pos_drop(s1)
        for blk in self.blocks1:
            s1 = blk(s1, mask_for_patch1)
        s2 = self.patch_embed2(s1)
        for blk in self.blocks2:
            s2 = blk(s2, mask_for_patch2)
        if self.with_fuse:
            stage1_embed, stage2_embed = self._fuse_forward(s1, s2, ids_keep, mask_ratio)
        x = self.patch_embed3(s2)
        x = x.flatten(2).permute(0, 2, 1)
        x = self.patch_embed4(x)
        x = x + self.pos_embed
        if mask_ratio is not None:
            x1 = torch.gather(x, dim=1, index=ids_keep[0].unsqueeze(-1).repeat(1, 1, x.shape[-1]))
            x2 = torch.gather(x, dim=1, index=ids_keep[1].unsqueeze(-1).repeat(1, 1, x.shape[-1]))
            x3 = torch.gather(x, dim=1, index=ids_keep[2].unsqueeze(-1).repeat(1, 1, x.shape[-1]))
            x4 = torch.gather(x, dim=1, index=ids_keep[3].unsqueeze(-1).repeat(1, 1, x.shape[-1]))
            x = torch.cat([x1, x2, x3, x4])
        for blk in self.blocks3:
            x = blk(x)
        if self.with_fuse:
            x = x + stage1_embed + stage2_embed
        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)
            x = self.fc_norm(x)
        else:
            x = self.norm(x)
        if mask_ratio is not None:
            mask = torch.cat([masks[0], masks[1], masks[2], masks[3]])
            return x, mask, ids_restore
        return x, None, None


class RelativePositionBias(nn.Module):

    def __init__(self, window_size, num_heads):
        super().__init__()
        self.window_size = window_size
        self.num_relative_distance = (2 * window_size[0] - 1) * (2 * window_size[1] - 1)
        self.relative_position_bias_table = nn.Parameter(torch.zeros(self.num_relative_distance, num_heads))
        coords_h = torch.arange(window_size[0])
        coords_w = torch.arange(window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size[0] - 1
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer('relative_position_index', relative_position_index)

    def forward(self):
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        return relative_position_bias.permute(2, 0, 1).contiguous()


class BinaryOSSFile:

    def __init__(self, bucket, path):
        self.bucket = bucket
        self.path = path
        self.buffer = BytesIO()

    def __enter__(self):
        return self.buffer

    def __exit__(self, *args):
        value = self.buffer.getvalue()
        if len(value) > 100 * 1024 ** 2:
            with oss_progress('uploading') as callback:
                self.bucket.put_object(self.path, value, progress_callback=callback)
        else:
            self.bucket.put_object(self.path, value)


FILE_NOT_FOUND = 5194902719927681032


class FileNotFoundError(BaseError):
    """Raised when a requested entity was not found."""
    code = FILE_NOT_FOUND


IO_FAILED = 5194902719927681033


class IOError(BaseError):
    """Raised when an operation returns a system-related error, including I/O failures."""
    code = IO_FAILED


class NullContextWrapper:

    def __init__(self, obj):
        self._obj = obj

    def __getattr__(self, name):
        return getattr(self._obj, name)

    def __iter__(self):
        return self._obj.__iter__()

    def __next__(self):
        return self._obj.__next__()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


OSS_PREFIX = 'oss://'

