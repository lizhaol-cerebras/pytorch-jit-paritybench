
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


from torch.nn import BCEWithLogitsLoss


import numpy as np


import torch


import matplotlib.pyplot as plt


from typing import Optional


from typing import List


from typing import Union


from typing import Callable


from torch.utils.data import Dataset


from collections import defaultdict


import warnings


from typing import Tuple


import pandas as pd


from sklearn.model_selection import GroupKFold


from functools import partial


import random


import typing


from typing import Any


from torch.utils.data.dataloader import default_collate


import collections


from torch import nn


from torch import Tensor


from typing import Iterable


from typing import Dict


import itertools


from collections.abc import Sized


from collections.abc import Iterable


import math


from typing import Sequence


from typing import Mapping


import torch.nn.functional as F


from torch.nn.modules.loss import _Loss


from torch.autograd import Variable


from collections import OrderedDict


from torch.nn import functional as F


import torch.nn as nn


import torch.utils.model_zoo as model_zoo


from torch.utils import model_zoo


from typing import Type


import inspect


import logging


from torchvision.models import densenet121


from torchvision.models import densenet161


from torchvision.models import densenet169


from torchvision.models import densenet201


from torchvision.models import DenseNet


from torchvision.models import resnet50


from torchvision.models import resnet34


from torchvision.models import resnet18


from torchvision.models import resnet101


from torchvision.models import resnet152


from torchvision.models import squeezenet1_1


import torch.utils.checkpoint as checkpoint


import copy


import torch.jit


import torch.nn.init


from abc import abstractmethod


from typing import Protocol


from torch.nn.modules.module import _IncompatibleKeys


from enum import Enum


from math import hypot


import numbers


from typing import Iterator


from torch.optim.optimizer import Optimizer


from torch.optim.lr_scheduler import _LRScheduler


from torch.optim.lr_scheduler import LambdaLR


from torch.optim.lr_scheduler import ExponentialLR


from torch.optim.lr_scheduler import CyclicLR


from torch.optim.lr_scheduler import MultiStepLR


from torch.optim.lr_scheduler import ReduceLROnPlateau


from torch.optim.lr_scheduler import CosineAnnealingLR


from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts


from collections import namedtuple


from torchvision.ops import box_iou


import functools


import torch.distributed as dist


from torch.utils.data import ConcatDataset


import re


import torch.nn


from torch.optim import SGD


from torch.utils.data import DataLoader


class ApplySoftmaxTo(nn.Module):
    output_keys: 'Tuple'
    temperature: 'float'
    dim: 'int'

    def __init__(self, model: 'nn.Module', output_key: 'Union[str, int, Iterable[str]]'='logits', dim: 'int'=1, temperature: 'float'=1):
        """
        Apply softmax activation on given output(s) of the model
        :param model: Model to wrap
        :param output_key: string, index or list of strings, indicating to what outputs softmax activation should be applied.
        :param dim: Tensor dimension for softmax activation
        :param temperature: Temperature scaling coefficient. Values > 1 will make logits sharper.
        """
        super().__init__()
        output_key = (output_key,) if isinstance(output_key, (str, int)) else tuple(set(output_key))
        self.output_keys = output_key
        self.model = model
        self.dim = dim
        self.temperature = temperature

    def forward(self, *input, **kwargs):
        output = self.model(*input, **kwargs)
        for key in self.output_keys:
            output[key] = output[key].mul(self.temperature).softmax(dim=self.dim)
        return output


class ApplySigmoidTo(nn.Module):
    output_keys: 'Tuple'
    temperature: 'float'

    def __init__(self, model: 'nn.Module', output_key: 'Union[str, int, Iterable[str]]'='logits', temperature=1):
        """
        Apply sigmoid activation on given output(s) of the model
        :param model: Model to wrap
        :param output_key: string index, or list of strings, indicating to what outputs sigmoid activation should be applied.
        :param temperature: Temperature scaling coefficient. Values > 1 will make logits sharper.
        """
        super().__init__()
        output_key = (output_key,) if isinstance(output_key, (str, int)) else tuple(set(output_key))
        self.output_keys = output_key
        self.model = model
        self.temperature = temperature

    def forward(self, *input, **kwargs):
        output = self.model(*input, **kwargs)
        for key in self.output_keys:
            output[key] = output[key].mul(self.temperature).sigmoid_()
        return output


def _deaugment_averaging(x: 'Tensor', reduction: 'MaybeStrOrCallable') ->Tensor:
    """
    Average predictions of TTA-ed model.
    This function assumes TTA dimension is 0, e.g [T, B, C, Ci, Cj, ..]

    Args:
        x: Input tensor of shape [T, B, ... ]
        reduction: Reduction mode ("sum", "mean", "gmean", "hmean", function, None)

    Returns:
        Tensor of shape [B, C, Ci, Cj, ..]
    """
    if reduction == 'mean':
        x = x.mean(dim=0)
    elif reduction == 'sum':
        x = x.sum(dim=0)
    elif reduction in {'gmean', 'geometric_mean'}:
        x = F.geometric_mean(x, dim=0)
    elif reduction in {'hmean', 'harmonic_mean'}:
        x = F.harmonic_mean(x, dim=0)
    elif reduction in {'harmonic1p'}:
        x = F.harmonic1p_mean(x, dim=0)
    elif reduction == 'logodd':
        x = F.logodd_mean(x, dim=0)
    elif reduction == 'log1p':
        x = F.log1p_mean(x, dim=0)
    elif callable(reduction):
        x = reduction(x, dim=0)
    elif reduction in {None, 'None', 'none'}:
        pass
    else:
        raise KeyError(f'Unsupported reduction mode {reduction}')
    return x


class Ensembler(nn.Module):
    __slots__ = ['outputs', 'reduction', 'return_some_outputs']
    """
    Compute sum (or average) of outputs of several models.
    """

    def __init__(self, models: 'List[nn.Module]', reduction: 'str'='mean', outputs: 'Optional[Iterable[str]]'=None):
        """

        :param models:
        :param reduction: Reduction key ('mean', 'sum', 'gmean', 'hmean' or None)
        :param outputs: Name of model outputs to average and return from Ensembler.
            If None, all outputs from the first model will be used.
        """
        super().__init__()
        self.return_some_outputs = outputs is not None
        self.outputs = tuple(outputs) if outputs else tuple()
        self.models = nn.ModuleList(models)
        self.reduction = reduction

    def forward(self, *input, **kwargs):
        outputs = [model(*input, **kwargs) for model in self.models]
        output_is_dict = isinstance(outputs[0], dict)
        output_is_list = isinstance(outputs[0], (list, tuple))
        if self.return_some_outputs:
            keys = self.outputs
        elif isinstance(outputs[0], dict):
            keys = outputs[0].keys()
        elif isinstance(outputs[0], (list, tuple)):
            keys = list(range(len(outputs[0])))
        elif torch.is_tensor(outputs[0]):
            keys = None
        else:
            raise RuntimeError()
        if keys is None:
            predictions = torch.stack(outputs)
            predictions = _deaugment_averaging(predictions, self.reduction)
            averaged_output = predictions
        else:
            averaged_output = {} if output_is_dict else []
            for key in keys:
                predictions = [output[key] for output in outputs]
                predictions = torch.stack(predictions)
                predictions = _deaugment_averaging(predictions, self.reduction)
                if output_is_dict:
                    averaged_output[key] = predictions
                else:
                    averaged_output.append(predictions)
        return averaged_output


class PickModelOutput(nn.Module):
    """
    Wraps a model that returns dict or list and returns only a specific element.

    Usage example:
        >>> model = MyAwesomeSegmentationModel() # Returns dict {"OUTPUT_MASK": Tensor, ...}
        >>> net  = nn.Sequential(PickModelOutput(model, "OUTPUT_MASK")), nn.Sigmoid())
    """
    __slots__ = ['target_key']

    def __init__(self, model: 'nn.Module', key: 'Union[str, int]'):
        super().__init__()
        self.model = model
        self.target_key = key

    def forward(self, *input, **kwargs) ->Tensor:
        output = self.model(*input, **kwargs)
        return output[self.target_key]


class SelectByIndex(nn.Module):
    """
    Select a single Tensor from the dict or list of output tensors.

    Usage example:
        >>> model = MyAwesomeSegmentationModel() # Returns dict {"OUTPUT_MASK": Tensor, ...}
        >>> net  = nn.Sequential(model, SelectByIndex("OUTPUT_MASK"), nn.Sigmoid())
    """
    __slots__ = ['target_key']

    def __init__(self, key: 'Union[str, int]'):
        super().__init__()
        self.target_key = key

    def forward(self, outputs: 'Dict[str, Tensor]') ->Tensor:
        return outputs[self.target_key]


string_types = type(b''), type('')


def pytorch_toolbelt_deprecated(reason):
    """
    Mark function or class as deprecated.
    It will result in a warning being emitted when the function is used.
    """
    if isinstance(reason, string_types):

        def decorator(func1):
            if inspect.isclass(func1):
                fmt1 = 'Call to deprecated class {name} ({reason}).'
            else:
                fmt1 = 'Call to deprecated function {name} ({reason}).'

            @functools.wraps(func1)
            def new_func1(*args, **kwargs):
                warnings.simplefilter('always', DeprecationWarning)
                warnings.warn(fmt1.format(name=func1.__name__, reason=reason), category=DeprecationWarning, stacklevel=2)
                warnings.simplefilter('default', DeprecationWarning)
                return func1(*args, **kwargs)
            return new_func1
        return decorator
    elif inspect.isclass(reason) or inspect.isfunction(reason):
        func2 = reason
        if inspect.isclass(func2):
            fmt2 = 'Call to deprecated class {name}.'
        else:
            fmt2 = 'Call to deprecated function {name}.'

        @functools.wraps(func2)
        def new_func2(*args, **kwargs):
            warnings.simplefilter('always', DeprecationWarning)
            warnings.warn(fmt2.format(name=func2.__name__), category=DeprecationWarning, stacklevel=2)
            warnings.simplefilter('default', DeprecationWarning)
            return func2(*args, **kwargs)
        return new_func2
    else:
        raise TypeError(repr(type(reason)))


@pytorch_toolbelt_deprecated('This class is deprecated. Please use GeneralizedTTA instead')
class TTAWrapper(nn.Module):

    def __init__(self, model: 'nn.Module', tta_function, **kwargs):
        super().__init__()
        self.model = model
        self.tta = partial(tta_function, **kwargs)

    def forward(self, *input):
        return self.tta(self.model, *input)


class GeneralizedTTA(nn.Module):
    __slots__ = ['augment_fn', 'deaugment_fn']
    """
    Example:
        tta_model = GeneralizedTTA(model,
            augment_fn=tta.d2_image_augment,
            deaugment_fn={
                OUTPUT_MASK_KEY: tta.d2_image_deaugment,
                OUTPUT_EDGE_KEY: tta.d2_image_deaugment,
            },


    Notes:
        Input tensors must be square for D2/D4 or similar types of augmentation
    """

    def __init__(self, model: 'Union[nn.Module, nn.DataParallel]', augment_fn: 'Union[Callable, Dict[str, Callable], List[Callable]]', deaugment_fn: 'Union[Callable, Dict[str, Callable], List[Callable]]'):
        super().__init__()
        self.model = model
        self.augment_fn = augment_fn
        self.deaugment_fn = deaugment_fn

    def forward(self, *input, **kwargs):
        if isinstance(self.augment_fn, dict):
            if len(input) != 0:
                raise ValueError('Input for GeneralizedTTA must not have positional arguments when augment_fn is dictionary')
            augmented_inputs = dict((key, augment(kwargs[key])) for key, augment in self.augment_fn.items())
            outputs = self.model(**augmented_inputs)
        elif isinstance(self.augment_fn, (list, tuple)):
            if len(kwargs) != 0:
                raise ValueError('Input for GeneralizedTTA must be exactly one tensor')
            augmented_inputs = [augment(x) for x, augment in zip(input, self.augment_fn)]
            outputs = self.model(*augmented_inputs)
        else:
            if len(input) != 1:
                raise ValueError('Input for GeneralizedTTA must be exactly one tensor')
            if len(kwargs) != 0:
                raise ValueError('Input for GeneralizedTTA must be exactly one tensor')
            augmented_input = self.augment_fn(input[0])
            outputs = self.model(augmented_input)
        if isinstance(self.deaugment_fn, dict):
            if not isinstance(outputs, dict):
                raise ValueError('Output of the model must be a dict')
            deaugmented_output = dict((key, self.deaugment_fn[key](outputs[key])) for key in self.deaugment_fn.keys())
        elif isinstance(self.deaugment_fn, (list, tuple)):
            if not isinstance(outputs, (dict, tuple)):
                raise ValueError('Output of the model must be a dict')
            deaugmented_output = [deaugment(value) for value, deaugment in zip(outputs, self.deaugment_fn)]
        else:
            deaugmented_output = self.deaugment_fn(outputs)
        return deaugmented_output


def ms_image_augment(image: 'Tensor', size_offsets: 'List[Union[int, Tuple[int, int]]]', mode='bilinear', align_corners=False) ->List[Tensor]:
    """
    Multi-scale image augmentation. This function create list of resized tensors from the input one.
    """
    batch_size, channels, rows, cols = image.size()
    augmented_inputs = []
    for offset in size_offsets:
        if isinstance(offset, (tuple, list)):
            rows_offset, cols_offset = offset
        else:
            rows_offset, cols_offset = offset, offset
        if rows_offset == 0 and cols_offset == 0:
            augmented_inputs.append(image)
        else:
            scale_size = rows + rows_offset, cols + cols_offset
            scaled_input = torch.nn.functional.interpolate(image, size=scale_size, mode=mode, align_corners=align_corners)
            augmented_inputs.append(scaled_input)
    return augmented_inputs


def ms_image_deaugment(images: 'List[Tensor]', size_offsets: 'List[Union[int, Tuple[int, int]]]', reduction: 'MaybeStrOrCallable'='mean', mode: 'str'='bilinear', align_corners: 'bool'=True, stride: 'int'=1) ->Tensor:
    """
    Perform multi-scale deaugmentation of predicted feature maps.

    Args:
        images: List of tensors of shape [B, C, Hi, Wi], [B, C, Hj, Wj], [B, C, Hk, Wk]
        size_offsets:
        reduction:
        mode:
        align_corners:
        stride: Stride of the output feature map w.r.t to model input size.
        Used to correctly scale size_offsets to match with size of output feature maps

    Returns:
        Averaged feature-map of the original size
    """
    if len(images) != len(size_offsets):
        raise ValueError('Number of images must be equal to number of size offsets')
    deaugmented_outputs = []
    for feature_map, offset in zip(images, size_offsets):
        if isinstance(offset, (tuple, list)):
            rows_offset, cols_offset = offset
        else:
            rows_offset, cols_offset = offset, offset
        if rows_offset == 0 and cols_offset == 0:
            deaugmented_outputs.append(feature_map)
        else:
            _, _, rows, cols = feature_map.size()
            original_size = rows - rows_offset // stride, cols - cols_offset // stride
            scaled_image = torch.nn.functional.interpolate(feature_map, size=original_size, mode=mode, align_corners=align_corners)
            deaugmented_outputs.append(scaled_image)
    deaugmented_outputs = torch.stack(deaugmented_outputs)
    return _deaugment_averaging(deaugmented_outputs, reduction=reduction)


class MultiscaleTTA(nn.Module):

    def __init__(self, model: 'nn.Module', size_offsets: 'List[int]', mode: 'str'='bilinear', align_corners: 'bool'=False, augment_fn: 'Callable'=ms_image_augment, deaugment_fn: 'Union[Callable, Dict[str, Callable]]'=ms_image_deaugment):
        if isinstance(deaugment_fn, Mapping):
            self.keys = set(deaugment_fn.keys())
        else:
            self.keys = None
        super().__init__()
        self.model = model
        self.size_offsets = size_offsets
        self.mode = mode
        self.align_corners = align_corners
        self.augment_fn = augment_fn
        self.deaugment_fn = deaugment_fn

    def forward(self, x):
        ms_inputs = self.augment_fn(x, size_offsets=self.size_offsets, mode=self.mode, align_corners=self.align_corners)
        ms_outputs = [self.model(x) for x in ms_inputs]
        outputs = {}
        if self.keys is None:
            outputs = self.deaugment_fn(ms_outputs, self.size_offsets)
        else:
            keys = self.keys
            for key in keys:
                deaugment_fn: 'Callable' = self.deaugment_fn[key]
                values = [x[key] for x in ms_outputs]
                outputs[key] = deaugment_fn(values, size_offsets=self.size_offsets, mode=self.mode, align_corners=self.align_corners)
        return outputs


def balanced_binary_cross_entropy_with_logits(logits: 'Tensor', targets: 'Tensor', gamma: 'float'=1.0, ignore_index: 'Optional[int]'=None, reduction: 'str'='mean') ->Tensor:
    """
    Balanced binary cross entropy loss.

    Args:
        logits:
        targets: This loss function expects target values to be hard targets 0/1.
        gamma: Power factor for balancing weights
        ignore_index:
        reduction:

    Returns:
        Zero-sized tensor with reduced loss if `reduction` is `sum` or `mean`; Otherwise returns loss of the
        shape of `logits` tensor.
    """
    pos_targets: 'Tensor' = targets.eq(1).sum()
    neg_targets: 'Tensor' = targets.eq(0).sum()
    num_targets = pos_targets + neg_targets
    pos_weight = torch.pow(neg_targets / (num_targets + 1e-07), gamma)
    neg_weight = 1.0 - pos_weight
    pos_term = pos_weight.pow(gamma) * targets * torch.nn.functional.logsigmoid(logits)
    neg_term = neg_weight.pow(gamma) * (1 - targets) * torch.nn.functional.logsigmoid(-logits)
    loss = -(pos_term + neg_term)
    if ignore_index is not None:
        loss = torch.masked_fill(loss, targets.eq(ignore_index), 0)
    if reduction == 'mean':
        loss = loss.mean()
    if reduction == 'sum':
        loss = loss.sum()
    return loss


class BalancedBCEWithLogitsLoss(nn.Module):
    """
    Balanced binary cross-entropy loss.

    https://arxiv.org/pdf/1504.06375.pdf (Formula 2)
    """
    __constants__ = ['gamma', 'reduction', 'ignore_index']

    def __init__(self, gamma: 'float'=1.0, reduction='mean', ignore_index: 'Optional[int]'=None):
        """

        Args:
            gamma:
            ignore_index:
            reduction:
        """
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, output: 'Tensor', target: 'Tensor') ->Tensor:
        return balanced_binary_cross_entropy_with_logits(output, target, gamma=self.gamma, ignore_index=self.ignore_index, reduction=self.reduction)


def log_t(u, t):
    """Compute log_t for `u'."""
    if t == 1.0:
        return u.log()
    else:
        return (u.pow(1.0 - t) - 1.0) / (1.0 - t)


def exp_t(u, t):
    """Compute exp_t for `u'."""
    if t == 1:
        return u.exp()
    else:
        return (1.0 + (1.0 - t) * u).relu().pow(1.0 / (1.0 - t))


def compute_normalization_binary_search(activations: 'Tensor', t: 'float', num_iters: 'int') ->Tensor:
    """Compute normalization value for each example (t < 1.0).
    Args:
      activations: A multi-dimensional tensor with last dimension `num_classes`.
      t: Temperature 2 (< 1.0 for finite support).
      num_iters: Number of iterations to run the method.
    Return: A tensor of same rank as activation with the last dimension being 1.
    """
    mu, _ = torch.max(activations, -1, keepdim=True)
    normalized_activations = activations - mu
    effective_dim = torch.sum((normalized_activations > -1.0 / (1.0 - t)).to(torch.int32), dim=-1, keepdim=True)
    shape_partition = activations.shape[:-1] + (1,)
    lower = torch.zeros(shape_partition, dtype=activations.dtype, device=activations.device)
    upper = -log_t(1.0 / effective_dim, t) * torch.ones_like(lower)
    for _ in range(num_iters):
        logt_partition = (upper + lower) / 2.0
        sum_probs = torch.sum(exp_t(normalized_activations - logt_partition, t), dim=-1, keepdim=True)
        update = sum_probs < 1.0
        lower = torch.reshape(lower * update + (1.0 - update) * logt_partition, shape_partition)
        upper = torch.reshape(upper * (1.0 - update) + update * logt_partition, shape_partition)
    logt_partition = (upper + lower) / 2.0
    return logt_partition + mu


def compute_normalization_fixed_point(activations: 'Tensor', t: 'float', num_iters: 'int') ->Tensor:
    """Return the normalization value for each example (t > 1.0).
    Args:
      activations: A multi-dimensional tensor with last dimension `num_classes`.
      t: Temperature 2 (> 1.0 for tail heaviness).
      num_iters: Number of iterations to run the method.
    Return: A tensor of same shape as activation with the last dimension being 1.
    """
    mu, _ = torch.max(activations, -1, keepdim=True)
    normalized_activations_step_0 = activations - mu
    normalized_activations = normalized_activations_step_0
    for _ in range(num_iters):
        logt_partition = torch.sum(exp_t(normalized_activations, t), -1, keepdim=True)
        normalized_activations = normalized_activations_step_0 * logt_partition.pow(1.0 - t)
    logt_partition = torch.sum(exp_t(normalized_activations, t), -1, keepdim=True)
    normalization_constants = -log_t(1.0 / logt_partition, t) + mu
    return normalization_constants


class ComputeNormalization(torch.autograd.Function):
    """
    Class implementing custom backward pass for compute_normalization. See compute_normalization.
    """

    @staticmethod
    def forward(ctx, activations, t, num_iters):
        if t < 1.0:
            normalization_constants = compute_normalization_binary_search(activations, t, num_iters)
        else:
            normalization_constants = compute_normalization_fixed_point(activations, t, num_iters)
        ctx.save_for_backward(activations, normalization_constants)
        ctx.t = t
        return normalization_constants

    @staticmethod
    def backward(ctx, grad_output):
        activations, normalization_constants = ctx.saved_tensors
        t = ctx.t
        normalized_activations = activations - normalization_constants
        probabilities = exp_t(normalized_activations, t)
        escorts = probabilities.pow(t)
        escorts = escorts / escorts.sum(dim=-1, keepdim=True)
        grad_input = escorts * grad_output
        return grad_input, None, None


def compute_normalization(activations, t, num_iters=5):
    """Compute normalization value for each example.
    Backward pass is implemented.
    Args:
      activations: A multi-dimensional tensor with last dimension `num_classes`.
      t: Temperature 2 (> 1.0 for tail heaviness, < 1.0 for finite support).
      num_iters: Number of iterations to run the method.
    Return: A tensor of same rank as activation with the last dimension being 1.
    """
    return ComputeNormalization.apply(activations, t, num_iters)


def tempered_softmax(activations, t, num_iters=5):
    """Tempered softmax function.
    Args:
      activations: A multi-dimensional tensor with last dimension `num_classes`.
      t: Temperature > 1.0.
      num_iters: Number of iterations to run the method.
    Returns:
      A probabilities tensor.
    """
    if t == 1.0:
        return activations.softmax(dim=-1)
    normalization_constants = compute_normalization(activations, t, num_iters)
    return exp_t(activations - normalization_constants, t)


def bi_tempered_logistic_loss(activations, labels, t1, t2, label_smoothing=0.0, num_iters=5, reduction='mean'):
    """Bi-Tempered Logistic Loss.
    Args:
      activations: A multi-dimensional tensor with last dimension `num_classes`.
      labels: A tensor with shape and dtype as activations (onehot),
        or a long tensor of one dimension less than activations (pytorch standard)
      t1: Temperature 1 (< 1.0 for boundedness).
      t2: Temperature 2 (> 1.0 for tail heaviness, < 1.0 for finite support).
      label_smoothing: Label smoothing parameter between [0, 1). Default 0.0.
      num_iters: Number of iterations to run the method. Default 5.
      reduction: ``'none'`` | ``'mean'`` | ``'sum'``. Default ``'mean'``.
        ``'none'``: No reduction is applied, return shape is shape of
        activations without the last dimension.
        ``'mean'``: Loss is averaged over minibatch. Return shape (1,)
        ``'sum'``: Loss is summed over minibatch. Return shape (1,)
    Returns:
      A loss tensor.
    """
    if len(labels.shape) < len(activations.shape):
        labels_onehot = torch.zeros_like(activations)
        labels_onehot.scatter_(1, labels[..., None], 1)
    else:
        labels_onehot = labels
    if label_smoothing > 0:
        num_classes = labels_onehot.shape[-1]
        labels_onehot = (1 - label_smoothing * num_classes / (num_classes - 1)) * labels_onehot + label_smoothing / (num_classes - 1)
    probabilities = tempered_softmax(activations, t2, num_iters)
    loss_values = labels_onehot * log_t(labels_onehot + 1e-10, t1) - labels_onehot * log_t(probabilities, t1) - labels_onehot.pow(2.0 - t1) / (2.0 - t1) + probabilities.pow(2.0 - t1) / (2.0 - t1)
    loss_values = loss_values.sum(dim=-1)
    if reduction == 'none':
        return loss_values
    if reduction == 'sum':
        return loss_values.sum()
    if reduction == 'mean':
        return loss_values.mean()


class BiTemperedLogisticLoss(nn.Module):
    """

    https://ai.googleblog.com/2019/08/bi-tempered-logistic-loss-for-training.html
    https://arxiv.org/abs/1906.03361
    """

    def __init__(self, t1: 'float', t2: 'float', smoothing=0.0, ignore_index=None, reduction: 'str'='mean'):
        """

        Args:
            t1:
            t2:
            smoothing:
            ignore_index:
            reduction:
        """
        super(BiTemperedLogisticLoss, self).__init__()
        self.t1 = t1
        self.t2 = t2
        self.smoothing = smoothing
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, predictions: 'Tensor', targets: 'Tensor') ->Tensor:
        loss = bi_tempered_logistic_loss(predictions, targets, t1=self.t1, t2=self.t2, label_smoothing=self.smoothing, reduction='none')
        if self.ignore_index is not None:
            mask = ~targets.eq(self.ignore_index)
            loss *= mask
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss


class BinaryBiTemperedLogisticLoss(nn.Module):
    """
    Modification of BiTemperedLogisticLoss for binary classification case.
    It's signature matches nn.BCEWithLogitsLoss: Predictions and target tensors must have shape [B,1,...]

    References:
        https://ai.googleblog.com/2019/08/bi-tempered-logistic-loss-for-training.html
        https://arxiv.org/abs/1906.03361
    """

    def __init__(self, t1: 'float', t2: 'float', smoothing: 'float'=0.0, ignore_index: 'Optional[int]'=None, reduction: 'str'='mean'):
        """

        Args:
            t1:
            t2:
            smoothing:
            ignore_index:
            reduction:
        """
        super().__init__()
        self.t1 = t1
        self.t2 = t2
        self.smoothing = smoothing
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, predictions: 'Tensor', targets: 'Tensor') ->Tensor:
        """
        Forward method of the loss function

        Args:
            predictions: [B,1,...]
            targets: [B,1,...]

        Returns:
            Zero-sized tensor with reduced loss if self.reduction is `sum` or `mean`; Otherwise returns loss of the
            shape of `predictions` tensor.
        """
        if predictions.size(1) != 1 or targets.size(1) != 1:
            raise ValueError('Channel dimension for predictions and targets must be equal to 1')
        loss = bi_tempered_logistic_loss(torch.cat([-predictions, predictions], dim=1).moveaxis(1, -1), torch.cat([1 - targets, targets], dim=1).moveaxis(1, -1), t1=self.t1, t2=self.t2, label_smoothing=self.smoothing, reduction='none').unsqueeze(dim=1)
        if self.ignore_index is not None:
            mask = targets.eq(self.ignore_index)
            loss = torch.masked_fill(loss, mask, 0)
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss


BINARY_MODE = 'binary'


MULTICLASS_MODE = 'multiclass'


MULTILABEL_MODE = 'multilabel'


def soft_dice_score(output: 'torch.Tensor', target: 'torch.Tensor', smooth: 'float'=0.0, eps: 'float'=1e-07, dims=None) ->torch.Tensor:
    """

    :param output:
    :param target:
    :param smooth:
    :param eps:
    :return:

    Shape:
        - Input: :math:`(N, NC, *)` where :math:`*` means any number
            of additional dimensions
        - Target: :math:`(N, NC, *)`, same shape as the input
        - Output: scalar.

    """
    assert output.size() == target.size()
    if dims is not None:
        intersection = torch.sum(output * target, dim=dims)
        cardinality = torch.sum(output + target, dim=dims)
    else:
        intersection = torch.sum(output * target)
        cardinality = torch.sum(output + target)
    dice_score = (2.0 * intersection + smooth) / (cardinality + smooth).clamp_min(eps)
    return dice_score


def to_tensor(x, dtype=None) ->torch.Tensor:
    if isinstance(x, torch.Tensor):
        if dtype is not None:
            x = x.type(dtype)
        return x
    if isinstance(x, np.ndarray) and x.dtype.kind not in {'O', 'M', 'U', 'S'}:
        x = torch.from_numpy(x)
        if dtype is not None:
            x = x.type(dtype)
        return x
    if isinstance(x, (list, tuple)):
        x = np.ndarray(x)
        x = torch.from_numpy(x)
        if dtype is not None:
            x = x.type(dtype)
        return x
    raise ValueError('Unsupported input type' + str(type(x)))


class DiceLoss(_Loss):
    """
    Implementation of Dice loss for image segmentation task.
    It supports binary, multiclass and multilabel cases
    """

    def __init__(self, mode: 'str', classes: 'List[int]'=None, log_loss=False, from_logits=True, smooth: 'float'=0.0, ignore_index=None, eps=1e-07):
        """

        :param mode: Metric mode {'binary', 'multiclass', 'multilabel'}
        :param classes: Optional list of classes that contribute in loss computation;
        By default, all channels are included.
        :param log_loss: If True, loss computed as `-log(jaccard)`; otherwise `1 - jaccard`
        :param from_logits: If True assumes input is raw logits
        :param smooth:
        :param ignore_index: Label that indicates ignored pixels (does not contribute to loss)
        :param eps: Small epsilon for numerical stability
        """
        assert mode in {BINARY_MODE, MULTILABEL_MODE, MULTICLASS_MODE}
        super(DiceLoss, self).__init__()
        self.mode = mode
        if classes is not None:
            assert mode != BINARY_MODE, 'Masking classes is not supported with mode=binary'
            classes = to_tensor(classes, dtype=torch.long)
        self.classes = classes
        self.from_logits = from_logits
        self.smooth = smooth
        self.eps = eps
        self.ignore_index = ignore_index
        self.log_loss = log_loss

    def forward(self, y_pred: 'Tensor', y_true: 'Tensor') ->Tensor:
        """

        :param y_pred: NxCxHxW
        :param y_true: NxHxW
        :return: scalar
        """
        assert y_true.size(0) == y_pred.size(0)
        if self.from_logits:
            if self.mode == MULTICLASS_MODE:
                y_pred = y_pred.log_softmax(dim=1).exp()
            else:
                y_pred = F.logsigmoid(y_pred).exp()
        bs = y_true.size(0)
        num_classes = y_pred.size(1)
        dims = 0, 2
        if self.mode == BINARY_MODE:
            y_true = y_true.view(bs, 1, -1)
            y_pred = y_pred.view(bs, 1, -1)
            if self.ignore_index is not None:
                mask = y_true != self.ignore_index
                y_pred = y_pred * mask
                y_true = y_true * mask
        if self.mode == MULTICLASS_MODE:
            y_true = y_true.view(bs, -1)
            y_pred = y_pred.view(bs, num_classes, -1)
            if self.ignore_index is not None:
                mask = y_true != self.ignore_index
                y_pred = y_pred * mask.unsqueeze(1)
                y_true = F.one_hot(y_true * mask, num_classes)
                y_true = y_true.permute(0, 2, 1) * mask.unsqueeze(1)
            else:
                y_true = F.one_hot(y_true, num_classes)
                y_true = y_true.permute(0, 2, 1)
        if self.mode == MULTILABEL_MODE:
            y_true = y_true.view(bs, num_classes, -1)
            y_pred = y_pred.view(bs, num_classes, -1)
            if self.ignore_index is not None:
                mask = y_true != self.ignore_index
                y_pred = y_pred * mask
                y_true = y_true * mask
        scores = soft_dice_score(y_pred, y_true.type_as(y_pred), smooth=self.smooth, eps=self.eps, dims=dims)
        if self.log_loss:
            loss = -torch.log(scores.clamp_min(self.eps))
        else:
            loss = 1.0 - scores
        mask = y_true.sum(dims) > 0
        loss *= mask
        if self.classes is not None:
            loss = loss[self.classes]
        return loss.mean()


class BinaryFocalLoss(nn.Module):
    __constants__ = ['alpha', 'gamma', 'reduction', 'ignore_index', 'normalized', 'reduced_threshold', 'activation']

    def __init__(self, alpha: 'Optional[float]'=None, gamma: 'float'=2.0, ignore_index: 'Optional[int]'=None, reduction: 'str'='mean', normalized: 'bool'=False, reduced_threshold: 'Optional[float]'=None, activation: 'str'='sigmoid', softmax_dim: 'Optional[int]'=None, class_weights: 'Optional[Tensor]'=None):
        """

        :param alpha: Prior probability of having positive value in target.
        :param gamma: Power factor for dampening weight (focal strength).
        :param ignore_index: If not None, targets may contain values to be ignored.
        Target values equal to ignore_index will be ignored from loss computation.
        :param reduced: Switch to reduced focal loss. Note, when using this mode you should use `reduction="sum"`.
        :param activation: Either `sigmoid` or `softmax`. If `softmax` is used, `softmax_dim` must be also specified.

        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.normalized = normalized
        self.reduced_threshold = reduced_threshold
        self.activation = activation
        self.softmax_dim = softmax_dim
        if class_weights is not None and not torch.is_tensor(class_weights):
            class_weights = torch.tensor(list(class_weights), dtype=torch.float32)
        self.register_buffer('class_weights', class_weights, persistent=False)
        self.focal_loss_fn = partial(focal_loss_with_logits, alpha=alpha, gamma=gamma, reduced_threshold=reduced_threshold, reduction=reduction, normalized=normalized, ignore_index=ignore_index, activation=activation, softmax_dim=softmax_dim)
        self.get_one_hot_targets = self._one_hot_targets_with_ignore if ignore_index is not None else self._one_hot_targets

    def __repr__(self):
        repr = f'{self.__class__.__name__}(alpha={self.alpha}, gamma={self.gamma}, '
        repr += f'ignore_index={self.ignore_index}, reduction={self.reduction}, normalized={self.normalized}, '
        repr += f'reduced_threshold={self.reduced_threshold}, activation={self.activation}, '
        repr += f'softmax_dim={self.softmax_dim},'
        repr += f'class_weights={self.class_weights.tolist()}, '
        repr += f')'
        return repr

    def forward(self, inputs: 'Tensor', targets: 'Tensor') ->Tensor:
        """
        Compute focal loss for binary classification problem.
        Args:
            inputs: [B,C,H,W]
            targets: [B,C,H,W] one-hot or [B,H,W] long tensor that will be one-hot encoded (w.r.t to ignore_index)

        Returns:

        """
        if len(targets.shape) + 1 == len(inputs.shape):
            targets = self.get_one_hot_targets(targets, num_classes=inputs.size(1))
        loss = self.focal_loss_fn(inputs, targets, class_weights=self.class_weights)
        return loss

    def _one_hot_targets(self, targets, num_classes):
        targets_one_hot = torch.nn.functional.one_hot(targets, num_classes=num_classes)
        targets_one_hot = torch.moveaxis(targets_one_hot, -1, 1)
        return targets_one_hot

    def _one_hot_targets_with_ignore(self, targets, num_classes):
        ignored_mask = targets.eq(self.ignore_index)
        targets_masked = torch.masked_fill(targets, ignored_mask, 0)
        targets_one_hot = torch.nn.functional.one_hot(targets_masked, num_classes=num_classes)
        targets_one_hot = torch.moveaxis(targets_one_hot, -1, 1)
        targets_one_hot.masked_fill_(ignored_mask.unsqueeze(1), self.ignore_index)
        return targets_one_hot


def softmax_focal_loss_with_logits(output: 'torch.Tensor', target: 'torch.Tensor', class_weights: 'Optional[torch.Tensor]'=None, gamma: 'float'=2.0, reduction: 'str'='mean', normalized: 'bool'=False, reduced_threshold: 'Optional[float]'=None, eps: 'float'=1e-06, ignore_index: 'int'=-100) ->torch.Tensor:
    """
    Softmax version of focal loss between target and output logits.
    See :class:`~pytorch_toolbelt.losses.CrossEntropyFocalLoss` for details.

    Args:
        output: Tensor of shape [B, C, *] (Similar to nn.CrossEntropyLoss)
        target: Tensor of shape [B, *] (Similar to nn.CrossEntropyLoss)
        gamma: Focal loss power factor
        reduction (string, optional): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum' | 'batchwise_mean'. 'none': no reduction will be applied,
            'mean': the sum of the output will be divided by the number of
            elements in the output, 'sum': the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`.
            'batchwise_mean' computes mean loss per sample in batch. Default: 'mean'
        normalized (bool): Compute normalized focal loss (https://arxiv.org/pdf/1909.07829.pdf).
        reduced_threshold (float, optional): Compute reduced focal loss (https://arxiv.org/abs/1903.01347).
    """
    ignore_mask = target.eq(ignore_index)
    pos_mask = ~ignore_mask
    targets_masked = torch.masked_fill(target, ignore_mask, 0)
    targets_oh = F.one_hot(targets_masked, num_classes=output.size(1)).moveaxis(-1, 1).float()
    probs = F.softmax(output, dim=1)
    pt = (1 - targets_oh) * probs + targets_oh * (1 - probs)
    loss = F.binary_cross_entropy_with_logits(output, targets_oh, reduction='none')
    if reduced_threshold is None:
        focal_term = pt.pow(gamma)
    else:
        focal_term = (pt / reduced_threshold).pow(gamma)
        focal_term[pt < reduced_threshold] = 1
    loss = focal_term * loss
    if class_weights is not None:
        class_weights = class_weights.reshape(1, -1, *[(1) for _ in range(loss.dim() - 2)])
        loss = loss * class_weights
    loss = loss.sum(dim=1) * pos_mask
    if normalized:
        norm_factor = focal_term.sum().clamp_min(eps)
        loss = loss / norm_factor
    if reduction == 'mean':
        loss = loss.mean()
    if reduction == 'sum':
        loss = loss.sum()
    if reduction == 'batchwise_mean':
        loss = loss.sum(0)
    return loss


class CrossEntropyFocalLoss(nn.Module):
    """
    Focal loss for multi-class problem. It uses softmax to compute focal term instead of sigmoid as in
    original paper. This loss expects target labes to have one dimension less (like in nn.CrossEntropyLoss).

    """

    def __init__(self, gamma: 'float'=2.0, reduction: 'str'='mean', normalized: 'bool'=False, reduced_threshold: 'Optional[float]'=None, ignore_index: 'int'=-100, class_weights: 'Optional[Tensor]'=None):
        """

        :param alpha:
        :param gamma:
        :param ignore_index: If not None, targets with given index are ignored
        :param reduced_threshold: A threshold factor for computing reduced focal loss
        :param class_weights: A tensor of shape [C] where C is the number of classes.
               It represents weights for each class. If None, all classes are treated equally.
               Unreduced loss is multiplied by class_weights before reduction.
        """
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.reduced_threshold = reduced_threshold
        self.normalized = normalized
        self.ignore_index = ignore_index
        self.register_buffer('class_weights', class_weights, persistent=False)

    def forward(self, inputs: 'Tensor', targets: 'Tensor') ->Tensor:
        """

        Args:
            inputs: [B,C,H,W] tensor
            targets: [B,H,W] tensor

        Returns:

        """
        return softmax_focal_loss_with_logits(inputs, targets, gamma=self.gamma, reduction=self.reduction, normalized=self.normalized, reduced_threshold=self.reduced_threshold, ignore_index=self.ignore_index, class_weights=self.class_weights)


class FocalCosineLoss(nn.Module):
    """
    Implementation Focal cosine loss from the "Data-Efficient Deep Learning Method for Image Classification
    Using Data Augmentation, Focal Cosine Loss, and Ensemble" (https://arxiv.org/abs/2007.07805).

    Credit: https://www.kaggle.com/c/cassava-leaf-disease-classification/discussion/203271
    """

    def __init__(self, alpha: 'float'=1, gamma: 'float'=2, xent: 'float'=0.1, reduction='mean'):
        super(FocalCosineLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.xent = xent
        self.reduction = reduction

    def forward(self, input: 'Tensor', target: 'Tensor') ->Tensor:
        cosine_loss = F.cosine_embedding_loss(input, torch.nn.functional.one_hot(target, num_classes=input.size(-1)), torch.tensor([1], device=target.device), reduction=self.reduction)
        cent_loss = F.cross_entropy(F.normalize(input), target, reduction='none')
        pt = torch.exp(-cent_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * cent_loss
        if self.reduction == 'mean':
            focal_loss = torch.mean(focal_loss)
        return cosine_loss + self.xent * focal_loss


def soft_jaccard_score(output: 'torch.Tensor', target: 'torch.Tensor', smooth: 'float'=0.0, eps: 'float'=1e-07, dims=None) ->torch.Tensor:
    """

    :param output:
    :param target:
    :param smooth:
    :param eps:
    :param dims:
    :return:

    Shape:
        - Input: :math:`(N, NC, *)` where :math:`*` means
            any number of additional dimensions
        - Target: :math:`(N, NC, *)`, same shape as the input
        - Output: scalar.

    """
    assert output.size() == target.size()
    if dims is not None:
        intersection = torch.sum(output * target, dim=dims)
        cardinality = torch.sum(output + target, dim=dims)
    else:
        intersection = torch.sum(output * target)
        cardinality = torch.sum(output + target)
    union = cardinality - intersection
    jaccard_score = (intersection + smooth) / (union + smooth).clamp_min(eps)
    return jaccard_score


class JaccardLoss(_Loss):
    """
    Implementation of Jaccard loss for image segmentation task.
    It supports binary, multi-class and multi-label cases.
    """

    def __init__(self, mode: 'str', classes: 'List[int]'=None, log_loss=False, from_logits=True, smooth=0, eps=1e-07):
        """

        :param mode: Metric mode {'binary', 'multiclass', 'multilabel'}
        :param classes: Optional list of classes that contribute in loss computation;
        By default, all channels are included.
        :param log_loss: If True, loss computed as `-log(jaccard)`; otherwise `1 - jaccard`
        :param from_logits: If True assumes input is raw logits
        :param smooth:
        :param eps: Small epsilon for numerical stability
        """
        assert mode in {BINARY_MODE, MULTILABEL_MODE, MULTICLASS_MODE}
        super(JaccardLoss, self).__init__()
        self.mode = mode
        if classes is not None:
            assert mode != BINARY_MODE, 'Masking classes is not supported with mode=binary'
            classes = to_tensor(classes, dtype=torch.long)
        self.classes = classes
        self.from_logits = from_logits
        self.smooth = smooth
        self.eps = eps
        self.log_loss = log_loss

    def forward(self, y_pred: 'Tensor', y_true: 'Tensor') ->Tensor:
        """

        :param y_pred: NxCxHxW
        :param y_true: NxHxW
        :return: scalar
        """
        assert y_true.size(0) == y_pred.size(0)
        if self.from_logits:
            if self.mode == MULTICLASS_MODE:
                y_pred = y_pred.log_softmax(dim=1).exp()
            else:
                y_pred = F.logsigmoid(y_pred).exp()
        bs = y_true.size(0)
        num_classes = y_pred.size(1)
        dims = 0, 2
        if self.mode == BINARY_MODE:
            y_true = y_true.view(bs, 1, -1)
            y_pred = y_pred.view(bs, 1, -1)
        if self.mode == MULTICLASS_MODE:
            y_true = y_true.view(bs, -1)
            y_pred = y_pred.view(bs, num_classes, -1)
            y_true = F.one_hot(y_true, num_classes)
            y_true = y_true.permute(0, 2, 1)
        if self.mode == MULTILABEL_MODE:
            y_true = y_true.view(bs, num_classes, -1)
            y_pred = y_pred.view(bs, num_classes, -1)
        scores = soft_jaccard_score(y_pred, y_true.type(y_pred.dtype), smooth=self.smooth, eps=self.eps, dims=dims)
        if self.log_loss:
            loss = -torch.log(scores.clamp_min(self.eps))
        else:
            loss = 1.0 - scores
        mask = y_true.sum(dims) > 0
        loss *= mask.float()
        if self.classes is not None:
            loss = loss[self.classes]
        return loss.mean()


def log_cosh_loss(y_pred: 'torch.Tensor', y_true: 'torch.Tensor') ->torch.Tensor:
    """
    Numerically stable log-cosh implementation.
    Reference: https://datascience.stackexchange.com/questions/96271/logcoshloss-on-pytorch

    Args:
        y_pred:
        y_true:

    Returns:

    """

    def _log_cosh(x: 'torch.Tensor') ->torch.Tensor:
        return x + torch.nn.functional.softplus(-2.0 * x) - math.log(2.0)
    return torch.mean(_log_cosh(y_pred - y_true))


class LogCoshLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, y_pred: 'torch.Tensor', y_true: 'torch.Tensor') ->torch.Tensor:
        return log_cosh_loss(y_pred, y_true)


def _flatten_binary_scores(scores, labels, ignore_index=None):
    """Flattens predictions in the batch (binary case)
    Remove labels equal to 'ignore'
    """
    scores = scores.view(-1)
    labels = labels.view(-1)
    if ignore_index is None:
        return scores, labels
    valid = labels != ignore_index
    vscores = scores[valid]
    vlabels = labels[valid]
    return vscores, vlabels


def _lovasz_grad(gt_sorted):
    """Compute gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def _lovasz_hinge_flat(logits, labels):
    """Binary Lovasz hinge loss
    Args:
        logits: [P] Variable, logits at each prediction (between -iinfinity and +iinfinity)
        labels: [P] Tensor, binary ground truth labels (0 or 1)
        ignore: label to ignore
    """
    if len(labels) == 0:
        return logits.sum() * 0.0
    signs = 2.0 * labels.float() - 1.0
    errors = 1.0 - logits * Variable(signs)
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    perm = perm.data
    gt_sorted = labels[perm]
    grad = _lovasz_grad(gt_sorted)
    loss = torch.dot(F.relu(errors_sorted), Variable(grad))
    return loss


def isnan(x):
    return x != x


def mean(values, ignore_nan=False, empty=0):
    """Nanmean compatible with generators."""
    values = iter(values)
    if ignore_nan:
        values = ifilterfalse(isnan, values)
    try:
        n = 1
        acc = next(values)
    except StopIteration:
        if empty == 'raise':
            raise ValueError('Empty mean')
        return empty
    for n, v in enumerate(values, 2):
        acc += v
    if n == 1:
        return acc
    return acc / n


def _lovasz_hinge(logits, labels, per_image=True, ignore_index=None):
    """
    Binary Lovasz hinge loss
        logits: [B, H, W] Variable, logits at each pixel (between -infinity and +infinity)
        labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
        per_image: compute the loss per image instead of per batch
        ignore: void class id
    """
    if per_image:
        loss = mean(_lovasz_hinge_flat(*_flatten_binary_scores(log.unsqueeze(0), lab.unsqueeze(0), ignore_index)) for log, lab in zip(logits, labels))
    else:
        loss = _lovasz_hinge_flat(*_flatten_binary_scores(logits, labels, ignore_index))
    return loss


class BinaryLovaszLoss(_Loss):

    def __init__(self, per_image: 'bool'=False, ignore_index: 'Optional[Union[int, float]]'=None):
        super().__init__()
        self.ignore_index = ignore_index
        self.per_image = per_image

    def forward(self, logits, target):
        return _lovasz_hinge(logits, target, per_image=self.per_image, ignore_index=self.ignore_index)


def _flatten_probas(probas, labels, ignore=None):
    """Flattens predictions in the batch"""
    if probas.dim() == 3:
        B, H, W = probas.size()
        probas = probas.view(B, 1, H, W)
    C = probas.size(1)
    probas = torch.movedim(probas, 1, -1)
    probas = probas.contiguous().view(-1, C)
    labels = labels.view(-1)
    if ignore is None:
        return probas, labels
    valid = labels != ignore
    vprobas = probas[valid]
    vlabels = labels[valid]
    return vprobas, vlabels


def _lovasz_softmax_flat(probas, labels, classes='present'):
    """Multi-class Lovasz-Softmax loss
    Args:
        @param probas: [P, C] Variable, class probabilities at each prediction (between 0 and 1)
        @param labels: [P] Tensor, ground truth labels (between 0 and C - 1)
        @param classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
    """
    if probas.numel() == 0:
        return probas * 0.0
    C = probas.size(1)
    losses = []
    class_to_sum = list(range(C)) if classes in ['all', 'present'] else classes
    for c in class_to_sum:
        fg = (labels == c).type_as(probas)
        if classes == 'present' and fg.sum() == 0:
            continue
        if C == 1:
            if len(classes) > 1:
                raise ValueError('Sigmoid output possible only with 1 class')
            class_pred = probas[:, 0]
        else:
            class_pred = probas[:, c]
        errors = (fg - class_pred).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        perm = perm.data
        fg_sorted = fg[perm]
        losses.append(torch.dot(errors_sorted, _lovasz_grad(fg_sorted)))
    return mean(losses)


def _lovasz_softmax(probas, labels, classes='present', per_image=False, ignore_index=None):
    """Multi-class Lovasz-Softmax loss
    Args:
        @param probas: [B, C, H, W] Variable, class probabilities at each prediction (between 0 and 1).
        Interpreted as binary (sigmoid) output with outputs of size [B, H, W].
        @param labels: [B, H, W] Tensor, ground truth labels (between 0 and C - 1)
        @param classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
        @param per_image: compute the loss per image instead of per batch
        @param ignore_index: void class labels
    """
    if per_image:
        loss = mean(_lovasz_softmax_flat(*_flatten_probas(prob.unsqueeze(0), lab.unsqueeze(0), ignore_index), classes=classes) for prob, lab in zip(probas, labels))
    else:
        loss = _lovasz_softmax_flat(*_flatten_probas(probas, labels, ignore_index), classes=classes)
    return loss


class LovaszLoss(_Loss):

    def __init__(self, per_image=False, ignore=None):
        super().__init__()
        self.ignore = ignore
        self.per_image = per_image

    def forward(self, logits, target):
        return _lovasz_softmax(logits, target, per_image=self.per_image, ignore_index=self.ignore)


class SoftBCEWithLogitsLoss(nn.Module):
    """
    Drop-in replacement for nn.BCEWithLogitsLoss with few additions:
    - Support of ignore_index value
    - Support of label smoothing
    """
    __constants__ = ['weight', 'pos_weight', 'reduction', 'ignore_index', 'smooth_factor']

    def __init__(self, weight=None, ignore_index: 'Optional[int]'=-100, reduction='mean', smooth_factor=None, pos_weight=None):
        super().__init__()
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.smooth_factor = smooth_factor
        self.register_buffer('weight', weight)
        self.register_buffer('pos_weight', pos_weight)

    def forward(self, input: 'Tensor', target: 'Tensor') ->Tensor:
        if self.smooth_factor is not None:
            soft_targets = ((1 - target) * self.smooth_factor + target * (1 - self.smooth_factor)).type_as(input)
        else:
            soft_targets = target.type_as(input)
        loss = F.binary_cross_entropy_with_logits(input, soft_targets, self.weight, pos_weight=self.pos_weight, reduction='none')
        if self.ignore_index is not None:
            not_ignored_mask: 'Tensor' = target != self.ignore_index
            loss *= not_ignored_mask.type_as(loss)
        if self.reduction == 'mean':
            loss = loss.mean()
        if self.reduction == 'sum':
            loss = loss.sum()
        return loss


def label_smoothed_nll_loss(lprobs: 'torch.Tensor', target: 'torch.Tensor', epsilon: 'float', ignore_index=None, reduction='mean', dim=-1) ->torch.Tensor:
    """

    Source: https://github.com/pytorch/fairseq/blob/master/fairseq/criterions/label_smoothed_cross_entropy.py

    :param lprobs: Log-probabilities of predictions (e.g after log_softmax)
    :param target:
    :param epsilon:
    :param ignore_index:
    :param reduction:
    :return:
    """
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(dim)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        target = target.masked_fill(pad_mask, 0)
        nll_loss = -lprobs.gather(dim=dim, index=target)
        smooth_loss = -lprobs.sum(dim=dim, keepdim=True)
        nll_loss = nll_loss.masked_fill(pad_mask, 0.0)
        smooth_loss = smooth_loss.masked_fill(pad_mask, 0.0)
    else:
        nll_loss = -lprobs.gather(dim=dim, index=target)
        smooth_loss = -lprobs.sum(dim=dim, keepdim=True)
        nll_loss = nll_loss.squeeze(dim)
        smooth_loss = smooth_loss.squeeze(dim)
    if reduction == 'sum':
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    if reduction == 'mean':
        nll_loss = nll_loss.mean()
        smooth_loss = smooth_loss.mean()
    eps_i = epsilon / lprobs.size(dim)
    loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss
    return loss


class SoftCrossEntropyLoss(nn.Module):
    """
    Drop-in replacement for nn.CrossEntropyLoss with few additions:
    - Support of label smoothing
    """
    __constants__ = ['reduction', 'ignore_index', 'smooth_factor']

    def __init__(self, reduction: 'str'='mean', smooth_factor: 'float'=0.0, ignore_index: 'Optional[int]'=-100, dim=1):
        super().__init__()
        self.smooth_factor = smooth_factor
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.dim = dim

    def forward(self, input: 'Tensor', target: 'Tensor') ->Tensor:
        log_prob = F.log_softmax(input, dim=self.dim)
        return label_smoothed_nll_loss(log_prob, target, epsilon=self.smooth_factor, ignore_index=self.ignore_index, reduction=self.reduction, dim=self.dim)


def soft_micro_f1(preds: 'Tensor', targets: 'Tensor', eps=1e-06) ->Tensor:
    """Compute the macro soft F1-score as a cost.
    Average (1 - soft-F1) across all labels.
    Use probability values instead of binary predictions.

    Args:
        targets (Tensor): targets array of shape (Num Samples, Num Classes)
        preds (Tensor): probability matrix of shape (Num Samples, Num Classes)

    Returns:
        cost (scalar Tensor): value of the cost function for the batch

    References:
        https://towardsdatascience.com/the-unknown-benefits-of-using-a-soft-f1-loss-in-classification-systems-753902c0105d
    """
    tp = torch.sum(preds * targets, dim=0)
    fp = torch.sum(preds * (1 - targets), dim=0)
    fn = torch.sum((1 - preds) * targets, dim=0)
    soft_f1 = 2 * tp / (2 * tp + fn + fp + eps)
    loss = 1 - soft_f1
    return loss.mean()


class BinarySoftF1Loss(nn.Module):

    def __init__(self, ignore_index: 'Optional[int]'=None, eps=1e-06):
        super().__init__()
        self.ignore_index = ignore_index
        self.eps = eps

    def forward(self, preds: 'Tensor', targets: 'Tensor') ->Tensor:
        targets = targets.view(-1)
        preds = preds.view(-1)
        if self.ignore_index is not None:
            not_ignored = targets != self.ignore_index
            preds = preds[not_ignored]
            targets = targets[not_ignored]
            if targets.numel() == 0:
                return torch.tensor(0, dtype=preds.dtype, device=preds.device)
        preds = preds.sigmoid().clamp(self.eps, 1 - self.eps)
        return soft_micro_f1(preds.view(-1, 1), targets.view(-1, 1))


class SoftF1Loss(nn.Module):

    def __init__(self, ignore_index: 'Optional[int]'=None, eps=1e-06):
        super().__init__()
        self.ignore_index = ignore_index
        self.eps = eps

    def forward(self, preds: 'Tensor', targets: 'Tensor') ->Tensor:
        preds = preds.softmax(dim=1).clamp(self.eps, 1 - self.eps)
        targets = torch.nn.functional.one_hot(targets, preds.size(1))
        if self.ignore_index is not None:
            not_ignored = targets != self.ignore_index
            preds = preds[not_ignored]
            targets = targets[not_ignored]
            if targets.numel() == 0:
                return torch.tensor(0, dtype=preds.dtype, device=preds.device)
        return soft_micro_f1(preds, targets)


class WingLoss(_Loss):

    def __init__(self, width=5, curvature=0.5, reduction='mean'):
        super(WingLoss, self).__init__(reduction=reduction)
        self.width = width
        self.curvature = curvature

    def forward(self, prediction, target):
        return F.wing_loss(prediction, target, self.width, self.curvature, self.reduction)


def mish_naive(input):
    """
    Apply the mish function element-wise:
    mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
    See additional documentation for mish class.
    Credit: https://github.com/digantamisra98/Mish
    """
    return input * torch.tanh(F.softplus(input))


class MishNaive(nn.Module):

    def __init__(self, inplace=False):
        super().__init__()

    def forward(self, x):
        return mish_naive(x)


@torch.jit.script
def mish_jit_bwd(x, grad_output):
    x_sigmoid = torch.sigmoid(x)
    x_tanh_sp = F.softplus(x).tanh()
    return grad_output.mul(x_tanh_sp + x * x_sigmoid * (1 - x_tanh_sp * x_tanh_sp))


@torch.jit.script
def mish_jit_fwd(x):
    return x.mul(torch.tanh(F.softplus(x)))


class MishFunction(torch.autograd.Function):
    """
    Mish: A Self Regularized Non-Monotonic Neural Activation Function - https://arxiv.org/abs/1908.08681
    A memory efficient, jit scripted variant of Mish
    """

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return mish_jit_fwd(x)

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_tensors[0]
        return mish_jit_bwd(x, grad_output)


def mish(x):
    """
    Apply the mish function element-wise:
    mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
    See additional documentation for mish class.
    Credit: https://github.com/digantamisra98/Mish
    """
    return MishFunction.apply(x)


class Mish(nn.Module):
    """
    Applies the mish function element-wise:
    mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
    Shape:
        - Input: (N, *) where * means, any number of additional
          dimensions
        - Output: (N, *), same shape as the input
    Examples:
        >>> m = Mish()
        >>> input = torch.randn(2)
        >>> output = m(input)

    Credit: https://github.com/digantamisra98/Mish
    """

    def __init__(self, inplace=False):
        """
        Init method.
        :param inplace: Not used, exists only for compatibility
        """
        super().__init__()

    def forward(self, input):
        """
        Forward pass of the function.
        """
        return mish(input)


def hard_sigmoid(x, inplace=False):
    return F.relu6(x + 3, inplace) / 6


class HardSigmoid(nn.Module):

    def __init__(self, inplace=False):
        super(HardSigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return hard_sigmoid(x, inplace=self.inplace)


def swish_naive(x):
    return x * x.sigmoid()


class SwishNaive(nn.Module):

    def forward(self, input_tensor):
        return swish_naive(input_tensor)


@torch.jit.script
def swish_jit_bwd(x, grad_output):
    x_sigmoid = torch.sigmoid(x)
    return grad_output * (x_sigmoid * (1 + x * (1 - x_sigmoid)))


@torch.jit.script
def swish_jit_fwd(x):
    return x.mul(torch.sigmoid(x))


class SwishFunction(torch.autograd.Function):
    """
    Memory efficient Swish implementation.

    Credit:
        https://blog.ceshine.net/post/pytorch-memory-swish/
        https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/activations_jit.py

    """

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return swish_jit_fwd(x)

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_tensors[0]
        return swish_jit_bwd(x, grad_output)


def swish(x):
    return SwishFunction.apply(x)


class Swish(nn.Module):

    def __init__(self, inplace=False):
        super(Swish, self).__init__()

    def forward(self, input_tensor):
        return swish(input_tensor)


def hard_swish(x, inplace=False):
    return x * hard_sigmoid(x, inplace)


class HardSwish(nn.Module):

    def __init__(self, inplace=False):
        super(HardSwish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return hard_swish(x, inplace=self.inplace)


class BasicConv2d(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_planes, eps=0.001, momentum=0.1, affine=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Mixed_3a(nn.Module):

    def __init__(self):
        super(Mixed_3a, self).__init__()
        self.maxpool = nn.MaxPool2d(3, stride=2)
        self.conv = BasicConv2d(64, 96, kernel_size=3, stride=2)

    def forward(self, x):
        x0 = self.maxpool(x)
        x1 = self.conv(x)
        out = torch.cat((x0, x1), 1)
        return out


class Mixed_4a(nn.Module):

    def __init__(self):
        super(Mixed_4a, self).__init__()
        self.branch0 = nn.Sequential(BasicConv2d(160, 64, kernel_size=1, stride=1), BasicConv2d(64, 96, kernel_size=3, stride=1))
        self.branch1 = nn.Sequential(BasicConv2d(160, 64, kernel_size=1, stride=1), BasicConv2d(64, 64, kernel_size=(1, 7), stride=1, padding=(0, 3)), BasicConv2d(64, 64, kernel_size=(7, 1), stride=1, padding=(3, 0)), BasicConv2d(64, 96, kernel_size=(3, 3), stride=1))

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        out = torch.cat((x0, x1), 1)
        return out


class Mixed_5a(nn.Module):

    def __init__(self):
        super(Mixed_5a, self).__init__()
        self.conv = BasicConv2d(192, 192, kernel_size=3, stride=2)
        self.maxpool = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x0 = self.conv(x)
        x1 = self.maxpool(x)
        out = torch.cat((x0, x1), 1)
        return out


class Inception_A(nn.Module):

    def __init__(self):
        super(Inception_A, self).__init__()
        self.branch0 = BasicConv2d(384, 96, kernel_size=1, stride=1)
        self.branch1 = nn.Sequential(BasicConv2d(384, 64, kernel_size=1, stride=1), BasicConv2d(64, 96, kernel_size=3, stride=1, padding=1))
        self.branch2 = nn.Sequential(BasicConv2d(384, 64, kernel_size=1, stride=1), BasicConv2d(64, 96, kernel_size=3, stride=1, padding=1), BasicConv2d(96, 96, kernel_size=3, stride=1, padding=1))
        self.branch3 = nn.Sequential(nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False), BasicConv2d(384, 96, kernel_size=1, stride=1))

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Reduction_A(nn.Module):

    def __init__(self):
        super(Reduction_A, self).__init__()
        self.branch0 = BasicConv2d(384, 384, kernel_size=3, stride=2)
        self.branch1 = nn.Sequential(BasicConv2d(384, 192, kernel_size=1, stride=1), BasicConv2d(192, 224, kernel_size=3, stride=1, padding=1), BasicConv2d(224, 256, kernel_size=3, stride=2))
        self.branch2 = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        return out


class Inception_B(nn.Module):

    def __init__(self):
        super(Inception_B, self).__init__()
        self.branch0 = BasicConv2d(1024, 384, kernel_size=1, stride=1)
        self.branch1 = nn.Sequential(BasicConv2d(1024, 192, kernel_size=1, stride=1), BasicConv2d(192, 224, kernel_size=(1, 7), stride=1, padding=(0, 3)), BasicConv2d(224, 256, kernel_size=(7, 1), stride=1, padding=(3, 0)))
        self.branch2 = nn.Sequential(BasicConv2d(1024, 192, kernel_size=1, stride=1), BasicConv2d(192, 192, kernel_size=(7, 1), stride=1, padding=(3, 0)), BasicConv2d(192, 224, kernel_size=(1, 7), stride=1, padding=(0, 3)), BasicConv2d(224, 224, kernel_size=(7, 1), stride=1, padding=(3, 0)), BasicConv2d(224, 256, kernel_size=(1, 7), stride=1, padding=(0, 3)))
        self.branch3 = nn.Sequential(nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False), BasicConv2d(1024, 128, kernel_size=1, stride=1))

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Reduction_B(nn.Module):

    def __init__(self):
        super(Reduction_B, self).__init__()
        self.branch0 = nn.Sequential(BasicConv2d(1024, 192, kernel_size=1, stride=1), BasicConv2d(192, 192, kernel_size=3, stride=2))
        self.branch1 = nn.Sequential(BasicConv2d(1024, 256, kernel_size=1, stride=1), BasicConv2d(256, 256, kernel_size=(1, 7), stride=1, padding=(0, 3)), BasicConv2d(256, 320, kernel_size=(7, 1), stride=1, padding=(3, 0)), BasicConv2d(320, 320, kernel_size=3, stride=2))
        self.branch2 = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        return out


class Inception_C(nn.Module):

    def __init__(self):
        super(Inception_C, self).__init__()
        self.branch0 = BasicConv2d(1536, 256, kernel_size=1, stride=1)
        self.branch1_0 = BasicConv2d(1536, 384, kernel_size=1, stride=1)
        self.branch1_1a = BasicConv2d(384, 256, kernel_size=(1, 3), stride=1, padding=(0, 1))
        self.branch1_1b = BasicConv2d(384, 256, kernel_size=(3, 1), stride=1, padding=(1, 0))
        self.branch2_0 = BasicConv2d(1536, 384, kernel_size=1, stride=1)
        self.branch2_1 = BasicConv2d(384, 448, kernel_size=(3, 1), stride=1, padding=(1, 0))
        self.branch2_2 = BasicConv2d(448, 512, kernel_size=(1, 3), stride=1, padding=(0, 1))
        self.branch2_3a = BasicConv2d(512, 256, kernel_size=(1, 3), stride=1, padding=(0, 1))
        self.branch2_3b = BasicConv2d(512, 256, kernel_size=(3, 1), stride=1, padding=(1, 0))
        self.branch3 = nn.Sequential(nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False), BasicConv2d(1536, 256, kernel_size=1, stride=1))

    def forward(self, x):
        x0 = self.branch0(x)
        x1_0 = self.branch1_0(x)
        x1_1a = self.branch1_1a(x1_0)
        x1_1b = self.branch1_1b(x1_0)
        x1 = torch.cat((x1_1a, x1_1b), 1)
        x2_0 = self.branch2_0(x)
        x2_1 = self.branch2_1(x2_0)
        x2_2 = self.branch2_2(x2_1)
        x2_3a = self.branch2_3a(x2_2)
        x2_3b = self.branch2_3b(x2_2)
        x2 = torch.cat((x2_3a, x2_3b), 1)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class InceptionV4(nn.Module):

    def __init__(self, num_classes=1001):
        super(InceptionV4, self).__init__()
        self.input_space = None
        self.input_size = 299, 299, 3
        self.mean = None
        self.std = None
        self.features = nn.Sequential(BasicConv2d(3, 32, kernel_size=3, stride=2), BasicConv2d(32, 32, kernel_size=3, stride=1), BasicConv2d(32, 64, kernel_size=3, stride=1, padding=1), Mixed_3a(), Mixed_4a(), Mixed_5a(), Inception_A(), Inception_A(), Inception_A(), Inception_A(), Reduction_A(), Inception_B(), Inception_B(), Inception_B(), Inception_B(), Inception_B(), Inception_B(), Inception_B(), Reduction_B(), Inception_C(), Inception_C(), Inception_C())
        self.last_linear = nn.Linear(1536, num_classes)

    def logits(self, features):
        adaptiveAvgPoolWidth = features.shape[2]
        x = F.avg_pool2d(features, kernel_size=adaptiveAvgPoolWidth)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

    def forward(self, x):
        x = self.features(x)
        x = self.logits(x)
        return x


class InvertedResidual(nn.Module):

    def __init__(self, inp, oup, stride, expand_ratio, activation):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup
        if expand_ratio == 1:
            self.conv = nn.Sequential(nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False), nn.BatchNorm2d(hidden_dim), activation(), nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False), nn.BatchNorm2d(oup))
        else:
            self.conv = nn.Sequential(nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False), nn.BatchNorm2d(hidden_dim), activation(), nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False), nn.BatchNorm2d(hidden_dim), activation(), nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False), nn.BatchNorm2d(oup))

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


def conv_1x1_bn(inp, oup, activation):
    return nn.Sequential(nn.Conv2d(inp, oup, 1, 1, 0, bias=False), nn.BatchNorm2d(oup), activation())


def conv_bn(inp, oup, stride, activation):
    return nn.Sequential(nn.Conv2d(inp, oup, 3, stride, 1, bias=False), nn.BatchNorm2d(oup), activation())


ACT_CELU = 'celu'


ACT_ELU = 'elu'


ACT_GELU = 'gelu'


ACT_GLU = 'glu'


ACT_HARD_SIGMOID = 'hard_sigmoid'


ACT_HARD_SWISH = 'hard_swish'


ACT_LEAKY_RELU = 'leaky_relu'


ACT_MISH = 'mish'


ACT_MISH_NAIVE = 'mish_naive'


ACT_NONE = 'none'


ACT_PRELU = 'prelu'


ACT_RELU = 'relu'


ACT_RELU6 = 'relu6'


ACT_SELU = 'selu'


ACT_SIGMOID = 'sigmoid'


ACT_SILU = 'silu'


ACT_SOFTMAX = 'softmax'


ACT_SOFTPLUS = 'softplus'


ACT_SWISH = 'swish'


ACT_SWISH_NAIVE = 'swish_naive'


class Identity(nn.Module):
    """The most useful module. A pass-through module which does nothing."""

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        return x


def get_activation_block(activation_name: 'str'):
    ACTIVATIONS = {ACT_CELU: nn.CELU, ACT_ELU: nn.ELU, ACT_GELU: nn.GELU, ACT_GLU: nn.GLU, ACT_HARD_SIGMOID: HardSigmoid, ACT_HARD_SWISH: HardSwish, ACT_LEAKY_RELU: nn.LeakyReLU, ACT_MISH: Mish, ACT_MISH_NAIVE: MishNaive, ACT_NONE: Identity, ACT_PRELU: nn.PReLU, ACT_RELU6: nn.ReLU6, ACT_RELU: nn.ReLU, ACT_SELU: nn.SELU, ACT_SILU: nn.SiLU, ACT_SOFTPLUS: nn.Softplus, ACT_SWISH: Swish, ACT_SWISH_NAIVE: SwishNaive, ACT_SIGMOID: nn.Sigmoid, ACT_SOFTMAX: nn.Softmax}
    return ACTIVATIONS[activation_name.lower()]


class MobileNetV2(nn.Module):

    def __init__(self, n_class=1000, input_size=224, width_mult=1.0, activation='relu6'):
        super(MobileNetV2, self).__init__()
        activation_block = get_activation_block(activation)
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280
        interverted_residual_setting = [[1, 16, 1, 1], [6, 24, 2, 2], [6, 32, 3, 2], [6, 64, 4, 2], [6, 96, 3, 1], [6, 160, 3, 2], [6, 320, 1, 1]]
        assert input_size % 32 == 0
        input_channel = int(input_channel * width_mult)
        self.last_channel = int(last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.layer0 = conv_bn(3, input_channel, 2, activation_block)
        for layer_index, (t, c, n, s) in enumerate(interverted_residual_setting):
            output_channel = int(c * width_mult)
            blocks = []
            for i in range(n):
                if i == 0:
                    blocks.append(block(input_channel, output_channel, s, expand_ratio=t, activation=activation_block))
                else:
                    blocks.append(block(input_channel, output_channel, 1, expand_ratio=t, activation=activation_block))
                input_channel = output_channel
            self.add_module(f'layer{layer_index + 1}', nn.Sequential(*blocks))
        self.final_layer = conv_1x1_bn(input_channel, self.last_channel, activation=activation_block)
        self.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(self.last_channel, n_class))
        self._initialize_weights()

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.final_layer(x)
        x = x.mean(3).mean(2)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


class SEModule(nn.Module):

    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x


class Bottleneck(nn.Module):
    """
    Base class for bottlenecks that implements `forward()` method.
    """

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out = self.se_module(out) + residual
        out = self.relu(out)
        return out


class SEBottleneck(Bottleneck):
    """
    Bottleneck for SENet154.
    """
    expansion = 4

    def __init__(self, inplanes, planes, groups, reduction, stride=1, downsample=None):
        super(SEBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes * 2, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes * 2)
        self.conv2 = nn.Conv2d(planes * 2, planes * 4, kernel_size=3, stride=stride, padding=1, groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(planes * 4)
        self.conv3 = nn.Conv2d(planes * 4, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se_module = SEModule(planes * 4, reduction=reduction)
        self.downsample = downsample
        self.stride = stride


class SEResNetBottleneck(Bottleneck):
    """
    ResNet bottleneck with a Squeeze-and-Excitation module. It follows Caffe
    implementation and uses `stride=stride` in `conv1` and not in `conv2`
    (the latter is used in the torchvision implementation of ResNet).
    """
    expansion = 4

    def __init__(self, inplanes, planes, groups, reduction, stride=1, downsample=None):
        super(SEResNetBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False, stride=stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se_module = SEModule(planes * 4, reduction=reduction)
        self.downsample = downsample
        self.stride = stride


class SEResNeXtBottleneck(Bottleneck):
    """
    ResNeXt bottleneck type C with a Squeeze-and-Excitation module.
    """
    expansion = 4

    def __init__(self, inplanes, planes, groups, reduction, stride=1, downsample=None, base_width=4):
        super(SEResNeXtBottleneck, self).__init__()
        width = math.floor(planes * (base_width / 64)) * groups
        self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, bias=False, stride=1)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=1, groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se_module = SEModule(planes * 4, reduction=reduction)
        self.downsample = downsample
        self.stride = stride


class SENet(nn.Module):

    def __init__(self, block, layers, groups, reduction, dropout_p=0.2, inplanes=128, input_3x3=True, downsample_kernel_size=3, downsample_padding=1, num_classes=1000):
        """
        Parameters
        ----------
        block (nn.Module): Bottleneck class.
            - For SENet154: SEBottleneck
            - For SE-ResNet models: SEResNetBottleneck
            - For SE-ResNeXt models:  SEResNeXtBottleneck
        layers (list of ints): Number of residual blocks for 4 layers of the
            network (layer1...layer4).
        groups (int): Number of groups for the 3x3 convolution in each
            bottleneck block.
            - For SENet154: 64
            - For SE-ResNet models: 1
            - For SE-ResNeXt models:  32
        reduction (int): Reduction ratio for Squeeze-and-Excitation modules.
            - For all models: 16
        dropout_p (float or None): Drop probability for the Dropout layer.
            If `None` the Dropout layer is not used.
            - For SENet154: 0.2
            - For SE-ResNet models: None
            - For SE-ResNeXt models: None
        inplanes (int):  Number of input channels for layer1.
            - For SENet154: 128
            - For SE-ResNet models: 64
            - For SE-ResNeXt models: 64
        input_3x3 (bool): If `True`, use three 3x3 convolutions instead of
            a single 7x7 convolution in layer0.
            - For SENet154: True
            - For SE-ResNet models: False
            - For SE-ResNeXt models: False
        downsample_kernel_size (int): Kernel size for downsampling convolutions
            in layer2, layer3 and layer4.
            - For SENet154: 3
            - For SE-ResNet models: 1
            - For SE-ResNeXt models: 1
        downsample_padding (int): Padding for downsampling convolutions in
            layer2, layer3 and layer4.
            - For SENet154: 1
            - For SE-ResNet models: 0
            - For SE-ResNeXt models: 0
        num_classes (int): Number of outputs in `last_linear` layer.
            - For all models: 1000
        """
        super(SENet, self).__init__()
        self.inplanes = inplanes
        if input_3x3:
            layer0_modules = [('conv1', nn.Conv2d(3, 64, 3, stride=2, padding=1, bias=False)), ('bn1', nn.BatchNorm2d(64)), ('relu1', nn.ReLU(inplace=True)), ('conv2', nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=False)), ('bn2', nn.BatchNorm2d(64)), ('relu2', nn.ReLU(inplace=True)), ('conv3', nn.Conv2d(64, inplanes, 3, stride=1, padding=1, bias=False)), ('bn3', nn.BatchNorm2d(inplanes)), ('relu3', nn.ReLU(inplace=True))]
        else:
            layer0_modules = [('conv1', nn.Conv2d(3, inplanes, kernel_size=7, stride=2, padding=3, bias=False)), ('bn1', nn.BatchNorm2d(inplanes)), ('relu1', nn.ReLU(inplace=True))]
        layer0_modules.append(('pool', nn.MaxPool2d(3, stride=2, ceil_mode=True)))
        self.layer0 = nn.Sequential(OrderedDict(layer0_modules))
        self.layer1 = self._make_layer(block, planes=64, blocks=layers[0], groups=groups, reduction=reduction, downsample_kernel_size=1, downsample_padding=0)
        self.layer2 = self._make_layer(block, planes=128, blocks=layers[1], stride=2, groups=groups, reduction=reduction, downsample_kernel_size=downsample_kernel_size, downsample_padding=downsample_padding)
        self.layer3 = self._make_layer(block, planes=256, blocks=layers[2], stride=2, groups=groups, reduction=reduction, downsample_kernel_size=downsample_kernel_size, downsample_padding=downsample_padding)
        self.layer4 = self._make_layer(block, planes=512, blocks=layers[3], stride=2, groups=groups, reduction=reduction, downsample_kernel_size=downsample_kernel_size, downsample_padding=downsample_padding)
        self.avg_pool = nn.AvgPool2d(7, stride=1)
        self.dropout = nn.Dropout(dropout_p) if dropout_p is not None else None
        self.last_linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, groups, reduction, stride=1, downsample_kernel_size=1, downsample_padding=0):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=downsample_kernel_size, stride=stride, padding=downsample_padding, bias=False), nn.BatchNorm2d(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, groups, reduction, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups, reduction))
        return nn.Sequential(*layers)

    def features(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def logits(self, x):
        x = self.avg_pool(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

    def forward(self, x):
        x = self.features(x)
        x = self.logits(x)
        return x


def instantiate_activation_block(activation_name: 'str', **kwargs) ->nn.Module:
    block = get_activation_block(activation_name)
    act_params = {}
    if 'inplace' in kwargs and activation_name in {ACT_RELU, ACT_RELU6, ACT_LEAKY_RELU, ACT_SELU, ACT_SILU, ACT_CELU, ACT_ELU}:
        act_params['inplace'] = kwargs['inplace']
    if 'slope' in kwargs and activation_name in {ACT_LEAKY_RELU}:
        act_params['negative_slope'] = kwargs['slope']
    if activation_name == ACT_PRELU:
        act_params['num_parameters'] = kwargs['num_parameters']
    if 'dim' in kwargs and activation_name in {ACT_SOFTMAX}:
        act_params['dim'] = kwargs['dim']
    return block(**act_params)


def ABN(num_features: 'int', eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, activation=ACT_RELU, slope=0.01, inplace=True):
    bn = nn.BatchNorm2d(num_features, eps=eps, momentum=momentum, affine=affine, track_running_stats=track_running_stats)
    act = instantiate_activation_block(activation, inplace=inplace, slope=slope)
    return nn.Sequential(OrderedDict([('bn', bn), (activation, act)]))


class IdentityResidualBlock(nn.Module):

    def __init__(self, in_channels, channels, stride=1, dilation=1, groups=1, norm_act=ABN, dropout=None):
        """Identity-mapping residual block
        Parameters
        ----------
        in_channels : int
            Number of input channels.
        channels : list of int
            Number of channels in the internal feature maps. Can either have two or three elements: if three construct
            a residual block with two `3 x 3` convolutions, otherwise construct a bottleneck block with `1 x 1`, then
            `3 x 3` then `1 x 1` convolutions.
        stride : int
            Stride of the first `3 x 3` convolution
        dilation : int
            Dilation to apply to the `3 x 3` convolutions.
        groups : int
            Number of convolution groups. This is used to create ResNeXt-style blocks and is only compatible with
            bottleneck blocks.
        norm_act : callable
            Function to create normalization / activation Module.
        dropout: callable
            Function to create Dropout Module.
        """
        super(IdentityResidualBlock, self).__init__()
        if len(channels) != 2 and len(channels) != 3:
            raise ValueError('channels must contain either two or three values')
        if len(channels) == 2 and groups != 1:
            raise ValueError('groups > 1 are only valid if len(channels) == 3')
        is_bottleneck = len(channels) == 3
        need_proj_conv = stride != 1 or in_channels != channels[-1]
        self.bn1 = norm_act(in_channels)
        if not is_bottleneck:
            layers = [('conv1', nn.Conv2d(in_channels, channels[0], 3, stride=stride, padding=dilation, bias=False, dilation=dilation)), ('bn2', norm_act(channels[0])), ('conv2', nn.Conv2d(channels[0], channels[1], 3, stride=1, padding=dilation, bias=False, dilation=dilation))]
            if dropout is not None:
                layers = layers[0:2] + [('dropout', dropout())] + layers[2:]
        else:
            layers = [('conv1', nn.Conv2d(in_channels, channels[0], 1, stride=stride, padding=0, bias=False)), ('bn2', norm_act(channels[0])), ('conv2', nn.Conv2d(channels[0], channels[1], 3, stride=1, padding=dilation, bias=False, groups=groups, dilation=dilation)), ('bn3', norm_act(channels[1])), ('conv3', nn.Conv2d(channels[1], channels[2], 1, stride=1, padding=0, bias=False))]
            if dropout is not None:
                layers = layers[0:4] + [('dropout', dropout())] + layers[4:]
        self.convs = nn.Sequential(OrderedDict(layers))
        if need_proj_conv:
            self.proj_conv = nn.Conv2d(in_channels, channels[-1], 1, stride=stride, padding=0, bias=False)

    def forward(self, x):
        if hasattr(self, 'proj_conv'):
            bn1 = self.bn1(x)
            shortcut = self.proj_conv(bn1)
        else:
            shortcut = x.clone()
            bn1 = self.bn1(x)
        out = self.convs(bn1)
        out.add_(shortcut)
        return out


class GlobalAvgPool2d(nn.Module):

    def __init__(self, flatten=False):
        """Global average pooling over the input's spatial dimensions"""
        super().__init__()
        self.flatten = flatten

    def forward(self, x):
        x = F.adaptive_avg_pool2d(x, output_size=1)
        if self.flatten:
            x = x.view(x.size(0), x.size(1))
        return x


class WiderResNet(nn.Module):

    def __init__(self, structure, norm_act=ABN, classes=0):
        """Wider ResNet with pre-activation (identity mapping) blocks

        Parameters
        ----------
        structure : list of int
            Number of residual blocks in each of the six modules of the network.
        norm_act : callable
            Function to create normalization / activation Module.
        classes : int
            If not `0` also include global average pooling and a fully-connected layer with `classes` outputs at the end
            of the network.
        """
        super(WiderResNet, self).__init__()
        self.structure = structure
        if len(structure) != 6:
            raise ValueError('Expected a structure with six values')
        self.mod1 = nn.Sequential(OrderedDict([('conv1', nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False))]))
        in_channels = 64
        channels = [(128, 128), (256, 256), (512, 512), (512, 1024), (512, 1024, 2048), (1024, 2048, 4096)]
        for mod_id, num in enumerate(structure):
            blocks = []
            for block_id in range(num):
                blocks.append(('block%d' % (block_id + 1), IdentityResidualBlock(in_channels, channels[mod_id], norm_act=norm_act)))
                in_channels = channels[mod_id][-1]
            if mod_id <= 4:
                self.add_module('pool%d' % (mod_id + 2), nn.MaxPool2d(3, stride=2, padding=1))
            self.add_module('mod%d' % (mod_id + 2), nn.Sequential(OrderedDict(blocks)))
        self.bn_out = norm_act(in_channels)
        if classes != 0:
            self.classifier = nn.Sequential(OrderedDict([('avg_pool', GlobalAvgPool2d()), ('fc', nn.Linear(in_channels, classes))]))

    def forward(self, img):
        out = self.mod1(img)
        out = self.mod2(self.pool2(out))
        out = self.mod3(self.pool3(out))
        out = self.mod4(self.pool4(out))
        out = self.mod5(self.pool5(out))
        out = self.mod6(self.pool6(out))
        out = self.mod7(out)
        out = self.bn_out(out)
        if hasattr(self, 'classifier'):
            out = self.classifier(out)
        return out


class WiderResNetA2(nn.Module):

    def __init__(self, structure, norm_act=ABN, classes=0, dilation=False):
        """Wider ResNet with pre-activation (identity mapping) blocks.
        This variant uses down-sampling by max-pooling in the first two blocks and by strided convolution in the others.

        Parameters
        ----------
        structure : list of int
            Number of residual blocks in each of the six modules of the network.
        norm_act : callable
            Function to create normalization / activation Module.
        classes : int
            If not `0` also include global average pooling and a fully-connected layer with `classes` outputs at the end
            of the network.
        dilation : bool
            If `True` apply dilation to the last three modules and change the down-sampling factor from 32 to 8.
        """
        super(WiderResNetA2, self).__init__()
        self.structure = structure
        self.dilation = dilation
        if len(structure) != 6:
            raise ValueError('Expected a structure with six values')
        self.mod1 = nn.Sequential(OrderedDict([('conv1', nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False))]))
        in_channels = 64
        channels = [(128, 128), (256, 256), (512, 512), (512, 1024), (512, 1024, 2048), (1024, 2048, 4096)]
        for mod_id, num in enumerate(structure):
            blocks = []
            for block_id in range(num):
                if not dilation:
                    dil = 1
                    stride = 2 if block_id == 0 and 2 <= mod_id <= 4 else 1
                else:
                    if mod_id == 3:
                        dil = 2
                    elif mod_id > 3:
                        dil = 4
                    else:
                        dil = 1
                    stride = 2 if block_id == 0 and mod_id == 2 else 1
                if mod_id == 4:
                    drop = partial(nn.Dropout2d, p=0.3)
                elif mod_id == 5:
                    drop = partial(nn.Dropout2d, p=0.5)
                else:
                    drop = None
                blocks.append(('block%d' % (block_id + 1), IdentityResidualBlock(in_channels, channels[mod_id], norm_act=norm_act, stride=stride, dilation=dil, dropout=drop)))
                in_channels = channels[mod_id][-1]
            if mod_id < 2:
                self.add_module('pool%d' % (mod_id + 2), nn.MaxPool2d(3, stride=2, padding=1))
            self.add_module('mod%d' % (mod_id + 2), nn.Sequential(OrderedDict(blocks)))
        self.bn_out = norm_act(in_channels)
        if classes != 0:
            self.classifier = nn.Sequential(OrderedDict([('avg_pool', GlobalAvgPool2d()), ('fc', nn.Linear(in_channels, classes))]))

    def forward(self, img):
        out = self.mod1(img)
        out = self.mod2(self.pool2(out))
        out = self.mod3(self.pool3(out))
        out = self.mod4(out)
        out = self.mod5(out)
        out = self.mod6(out)
        out = self.mod7(out)
        out = self.bn_out(out)
        if hasattr(self, 'classifier'):
            return self.classifier(out)
        else:
            return out


def append_coords(input_tensor, with_r=False):
    batch_size, _, x_dim, y_dim = input_tensor.size()
    xx_channel = torch.arange(x_dim).repeat(1, y_dim, 1)
    yy_channel = torch.arange(y_dim).repeat(1, x_dim, 1).transpose(1, 2)
    xx_channel = xx_channel.float() / (x_dim - 1)
    yy_channel = yy_channel.float() / (y_dim - 1)
    xx_channel = xx_channel * 2 - 1
    yy_channel = yy_channel * 2 - 1
    xx_channel = xx_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)
    yy_channel = yy_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)
    ret = torch.cat([input_tensor, xx_channel.type_as(input_tensor), yy_channel.type_as(input_tensor)], dim=1)
    if with_r:
        rr = torch.sqrt(torch.pow(xx_channel.type_as(input_tensor) - 0.5, 2) + torch.pow(yy_channel.type_as(input_tensor) - 0.5, 2))
        ret = torch.cat([ret, rr], dim=1)
    return ret


class AddCoords(nn.Module):
    """
    An alternative implementation for PyTorch with auto-infering the x-y dimensions.
    https://github.com/mkocabas/CoordConv-pytorch/blob/master/CoordConv.py
    """

    def __init__(self, with_r=False):
        super().__init__()
        self.with_r = with_r

    def forward(self, x):
        """
        Args:
            x: shape(batch, channel, x_dim, y_dim)
        """
        return append_coords(x, self.with_r)


class CoordConv(nn.Module):

    def __init__(self, in_channels, out_channels, with_r=False, **kwargs):
        super().__init__()
        self.addcoords = AddCoords(with_r=with_r)
        in_size = in_channels + 2
        if with_r:
            in_size += 1
        self.conv = nn.Conv2d(in_size, out_channels, **kwargs)

    def forward(self, x):
        ret = self.addcoords(x)
        ret = self.conv(ret)
        return ret


NORM_BATCH = 'batch_norm'


def instantiate_normalization_block(normalization: 'str', in_channels: 'int', **kwargs):
    if normalization in ('bn', 'batch', 'batch2d', 'batch_norm', 'batch_norm_2d', 'batchnorm', 'batchnorm2d'):
        return nn.BatchNorm2d(num_features=in_channels)
    if normalization in ('bn3d', 'batch3d', 'batch_norm3d', 'batch_norm_3d', 'batchnorm3d'):
        return nn.BatchNorm3d(num_features=in_channels)
    if normalization in ('gn', 'group', 'group_norm', 'groupnorm'):
        return nn.GroupNorm(num_channels=in_channels, **kwargs)
    if normalization in ('in', 'instance', 'instance2d', 'instance_norm', 'instancenorm', 'instance_norm_2d', 'instancenorm2d'):
        return nn.InstanceNorm2d(num_features=in_channels, **kwargs)
    if normalization in ('in3d', 'instance3d', 'instance_norm_3d', 'instancenorm3d'):
        return nn.InstanceNorm3d(num_features=in_channels, **kwargs)
    raise KeyError(f"Unknown normalization type '{normalization}'")


class BiFPNConvBlock(nn.Module):
    """
    Convolution block with Batch Normalization and ReLU activation.

    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, activation: 'str'=ACT_RELU, dilation=1, normalization: 'str'=NORM_BATCH):
        super(BiFPNConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = instantiate_normalization_block(normalization, out_channels)
        self.act = instantiate_activation_block(activation, inplace=True)

    def forward(self, x: 'Tensor') ->Tensor:
        x = self.conv(x)
        x = self.bn(x)
        return self.act(x)


class BiFPNBlock(nn.Module):
    """
    Bi-Directional Feature Pyramid Network
    """
    __constants__ = ['epsilon']

    def __init__(self, feature_size: 'int', num_feature_maps: 'int', epsilon=0.0001, activation=ACT_RELU, normalization='batch', block: 'Union[Type[BiFPNConvBlock], Type[DepthwiseSeparableConv2dBlock]]'=BiFPNConvBlock):
        super(BiFPNBlock, self).__init__()
        self.epsilon = epsilon
        num_blocks = num_feature_maps - 1
        self.top_down_blocks = nn.ModuleList([block(feature_size, feature_size, activation=activation, normalization=normalization) for _ in range(num_blocks)])
        self.bottom_up_blocks = nn.ModuleList([block(feature_size, feature_size, activation=activation, normalization=normalization) for _ in range(num_blocks)])
        self.register_parameter('w1', nn.Parameter(torch.Tensor(2, num_blocks), requires_grad=True))
        self.register_parameter('w2', nn.Parameter(torch.Tensor(3, num_blocks), requires_grad=True))
        torch.nn.init.constant_(self.w1, 1)
        torch.nn.init.constant_(self.w2, 1)

    def forward(self, inputs: 'List[Tensor]') ->List[Tensor]:
        transition_features = self.top_down_pathway(inputs)
        return self.bottom_up_pathway(transition_features, inputs)

    def bottom_up_pathway(self, transition_features, inputs):
        w2 = torch.nn.functional.relu(self.w2)
        w2 = w2 / (torch.sum(w2, dim=0) + self.epsilon)
        outputs = [transition_features[-1]]
        transition_reversed = transition_features[:-1][::-1]
        for i, block in enumerate(self.bottom_up_blocks):
            x = inputs[i + 1]
            td = transition_reversed[i]
            y = block(x * w2[0, i] + td * w2[1, i] + F.interpolate(outputs[-1], size=x.size()[2:]) * w2[2, i])
            outputs.append(y)
        return outputs

    def top_down_pathway(self, inputs: 'List[Tensor]') ->List[Tensor]:
        w1 = torch.nn.functional.relu(self.w1)
        w1 = w1 / (torch.sum(w1, dim=0) + self.epsilon)
        features = [inputs[-1]]
        inputs_reversed = inputs[:-1][::-1]
        for i, block in enumerate(self.top_down_blocks):
            x = inputs_reversed[i]
            y = block(w1[0, i] * x + w1[1, i] * F.interpolate(features[-1], size=x.size()[2:]))
            features.append(y)
        return features


class RCM(nn.Module):

    def __init__(self, in_channels: 'int', out_channels: 'int'):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.block = nn.Sequential(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True), nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))

    def forward(self, x):
        x = self.conv1(x)
        return self.block(x) + x


class DepthwiseSeparableConv2d(nn.Module):
    """
    Depth-wise separable convolution.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True):
        super(DepthwiseSeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, dilation=dilation, padding=padding, stride=stride, bias=bias, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, groups=groups, bias=bias)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


def ds_cfm_branch(in_channels: 'int', out_channels: 'int', kernel_size: 'int'):
    return nn.Sequential(DepthwiseSeparableConv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True), DepthwiseSeparableConv2d(out_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2, bias=False), nn.BatchNorm2d(out_channels))


class CFM(nn.Module):

    def __init__(self, in_channels: 'int', out_channels: 'int', kernel_sizes=[3, 5, 7, 11]):
        super().__init__()
        self.gp_branch = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False), nn.BatchNorm2d(out_channels))
        self.conv_branches = nn.ModuleList(ds_cfm_branch(in_channels, out_channels, ks) for ks in kernel_sizes)

    def forward(self, x):
        gp = self.gp_branch(x)
        gp = gp.expand_as(x)
        conv_branches = [conv(x) for conv in self.conv_branches]
        return torch.cat(conv_branches + [gp], dim=1)


class AMM(nn.Module):

    def __init__(self, in_channels: 'int', out_channels: 'int'):
        super().__init__()
        self.conv_bn_relu = nn.Sequential(DepthwiseSeparableConv2d(in_channels + out_channels, out_channels, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))

    def forward(self, encoder, decoder):
        decoder = F.interpolate(decoder, size=encoder.size()[2:], mode='bilinear', align_corners=False)
        x = torch.cat([encoder, decoder], dim=1)
        x = self.conv_bn_relu(x)
        x = F.adaptive_avg_pool2d(x, 1) * x
        return encoder + x


def drop_path(x, drop_prob: 'float'=0.0, training: 'bool'=False, scale_by_keep: 'bool'=True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob: 'float'=0.0, scale_by_keep: 'bool'=True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob, 3):0.3f}'


class DropBlock2D(nn.Module):
    """Randomly zeroes 2D spatial blocks of the input tensor.
    As described in the paper
    `DropBlock: A regularization method for convolutional networks`_ ,
    dropping whole blocks of feature map allows to remove semantic
    information as compared to regular dropout.
    Args:
        drop_prob (float): probability of an element to be dropped.
        block_size (int): size of the block to drop
    Shape:
        - Input: `(N, C, H, W)`
        - Output: `(N, C, H, W)`
    .. _DropBlock: A regularization method for convolutional networks:
       https://arxiv.org/abs/1810.12890
    """

    def __init__(self, drop_prob, block_size):
        super(DropBlock2D, self).__init__()
        self.drop_prob = drop_prob
        self.block_size = block_size

    def forward(self, x):
        assert x.dim() == 4, 'Expected input with 4 dimensions (bsize, channels, height, width)'
        if not self.training or self.drop_prob == 0.0:
            return x
        else:
            gamma = self._compute_gamma(x)
            mask = torch.rand(x.shape[0], *x.shape[2:]) < gamma
            block_mask, keeped = self._compute_block_mask(mask)
            out = x * block_mask[:, None, :, :]
            out = out * (block_mask.numel() / keeped)
            return out

    def _compute_block_mask(self, mask):
        block_mask = F.max_pool2d(input=mask[:, None, :, :], kernel_size=(self.block_size, self.block_size), stride=(1, 1), padding=self.block_size // 2)
        if self.block_size % 2 == 0:
            block_mask = block_mask[:, :, :-1, :-1]
        keeped = block_mask.numel() - block_mask.sum()
        block_mask = 1 - block_mask.squeeze(1)
        return block_mask, keeped

    def _compute_gamma(self, x):
        return self.drop_prob / self.block_size ** 2


class DropBlock3D(DropBlock2D):
    """Randomly zeroes 3D spatial blocks of the input tensor.
    An extension to the concept described in the paper
    `DropBlock: A regularization method for convolutional networks`_ ,
    dropping whole blocks of feature map allows to remove semantic
    information as compared to regular dropout.
    Args:
        drop_prob (float): probability of an element to be dropped.
        block_size (int): size of the block to drop
    Shape:
        - Input: `(N, C, D, H, W)`
        - Output: `(N, C, D, H, W)`
    .. _DropBlock: A regularization method for convolutional networks:
       https://arxiv.org/abs/1810.12890
    """

    def __init__(self, drop_prob, block_size):
        super(DropBlock3D, self).__init__(drop_prob, block_size)

    def forward(self, x):
        assert x.dim() == 5, 'Expected input with 5 dimensions (bsize, channels, depth, height, width)'
        if not self.training or self.drop_prob == 0.0:
            return x
        else:
            gamma = self._compute_gamma(x)
            mask = torch.rand(x.shape[0], *x.shape[2:]) < gamma
            block_mask = self._compute_block_mask(mask)
            out = x * block_mask[:, None, :, :, :]
            out = out * block_mask.numel() / block_mask.sum()
            return out

    def _compute_block_mask(self, mask):
        block_mask = F.max_pool3d(input=mask[:, None, :, :, :], kernel_size=(self.block_size, self.block_size, self.block_size), stride=(1, 1, 1), padding=self.block_size // 2)
        if self.block_size % 2 == 0:
            block_mask = block_mask[:, :, :-1, :-1, :-1]
        block_mask = 1 - block_mask.squeeze(1)
        return block_mask

    def _compute_gamma(self, x):
        return self.drop_prob / self.block_size ** 3


class DropBlockScheduled(nn.Module):

    def __init__(self, dropblock, start_value, stop_value, nr_steps, start_step=0):
        super(DropBlockScheduled, self).__init__()
        self.dropblock = dropblock
        self.register_buffer('i', torch.zeros(1, dtype=torch.int64))
        self.start_step = start_step
        self.nr_steps = nr_steps
        self.step_size = (stop_value - start_value) / nr_steps

    def forward(self, x):
        if self.training:
            self.step()
        return self.dropblock(x)

    def step(self):
        idx = self.i.item()
        if self.start_step < idx < self.start_step + self.nr_steps:
            self.dropblock.drop_prob += self.step_size
        self.i += 1


class DepthwiseSeparableConv2dBlock(nn.Module):
    """
    Depthwise seperable convolution with batchnorm and activation.
    """

    def __init__(self, in_channels: 'int', out_channels: 'int', activation: 'str', kernel_size: 'int'=3, stride=1, padding=1, dilation=1, normalization: 'str'=NORM_BATCH):
        super(DepthwiseSeparableConv2dBlock, self).__init__()
        self.depthwise = DepthwiseSeparableConv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=False)
        self.bn = instantiate_normalization_block(normalization, out_channels)
        self.act = instantiate_activation_block(activation, inplace=True)

    def forward(self, x: 'Tensor') ->Tensor:
        x = self.depthwise(x)
        x = self.bn(x)
        return self.act(x)


class HGResidualBlock(nn.Module):

    def __init__(self, input_channels: 'int', output_channels: 'int', reduction=2, activation: 'Callable'=nn.ReLU):
        super(HGResidualBlock, self).__init__()
        mid_channels = input_channels // reduction
        self.bn1 = nn.BatchNorm2d(input_channels)
        self.act1 = activation(inplace=True)
        self.conv1 = nn.Conv2d(input_channels, mid_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.act2 = activation(inplace=True)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(mid_channels)
        self.act3 = activation(inplace=True)
        self.conv3 = nn.Conv2d(mid_channels, output_channels, kernel_size=1, bias=True)
        if input_channels == output_channels:
            self.skip_layer = nn.Identity()
        else:
            self.skip_layer = nn.Conv2d(input_channels, output_channels, kernel_size=1)
            torch.nn.init.zeros_(self.skip_layer.bias)
        torch.nn.init.zeros_(self.conv3.bias)

    def forward(self, x: 'Tensor') ->Tensor:
        residual = self.skip_layer(x)
        out = self.bn1(x)
        out = self.act1(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.act2(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.act3(out)
        out = self.conv3(out)
        out += residual
        return out


class HGStemBlock(nn.Module):

    def __init__(self, input_channels, output_channels, activation: 'Callable'=nn.ReLU):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, padding=1, stride=2, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.act1 = activation(inplace=True)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        self.act2 = activation(inplace=True)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2, bias=False)
        self.bn3 = nn.BatchNorm2d(64)
        self.act3 = activation(inplace=True)
        self.residual1 = HGResidualBlock(64, 128)
        self.residual2 = HGResidualBlock(128, output_channels)

    def forward(self, x: 'Tensor') ->Tensor:
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.act2(self.bn2(self.conv2(x)))
        x = self.act3(self.bn3(self.conv3(x)))
        x = self.residual1(x)
        x = self.residual2(x)
        return x


class HGBlock(nn.Module):
    """
    A single Hourglass model block.
    """

    def __init__(self, depth: 'int', input_features: 'int', features, increase=0, activation=nn.ReLU, repeats=1, pooling_block=nn.MaxPool2d):
        super(HGBlock, self).__init__()
        nf = features + increase
        if inspect.isclass(pooling_block) and issubclass(pooling_block, (nn.MaxPool2d, nn.AvgPool2d)):
            self.down = pooling_block(kernel_size=2, padding=0, stride=2)
        else:
            self.down = pooling_block(input_features)
        if repeats == 1:
            self.up1 = HGResidualBlock(input_features, features, activation=activation)
            self.low1 = HGResidualBlock(input_features, nf, activation=activation)
        else:
            up_blocks = []
            up_input_features = input_features
            for _ in range(repeats):
                up_blocks.append(HGResidualBlock(up_input_features, features))
                up_input_features = features
            self.up1 = nn.Sequential(*up_blocks)
            down_blocks = []
            down_input_features = input_features
            for _ in range(repeats):
                up_blocks.append(HGResidualBlock(down_input_features, nf))
                down_input_features = nf
            self.low1 = nn.Sequential(*down_blocks)
        self.depth = depth
        if self.depth > 1:
            self.low2 = HGBlock(depth - 1, nf, nf, increase=increase, pooling_block=pooling_block, activation=activation, repeats=repeats)
        else:
            self.low2 = HGResidualBlock(nf, nf, activation=activation)
        self.low3 = HGResidualBlock(nf, features, activation=activation)
        self.up = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x: 'Tensor') ->Tensor:
        up1 = self.up1(x)
        pool1 = self.down(x)
        low1 = self.low1(pool1)
        low2 = self.low2(low1)
        low3 = self.low3(low2)
        up2 = self.up(low3)
        hg = up1 + up2
        return hg


def conv1x1_bn_act(in_channels, out_channels, activation=nn.ReLU):
    return nn.Sequential(OrderedDict([('conv', nn.Conv2d(in_channels, out_channels, kernel_size=1)), ('bn', nn.BatchNorm2d(out_channels)), ('act', activation(inplace=True))]))


class HGFeaturesBlock(nn.Module):

    def __init__(self, features: 'int', activation: 'Callable', blocks=1):
        super().__init__()
        residual_blocks = [HGResidualBlock(features, features, activation=activation) for _ in range(blocks)]
        self.residuals = nn.Sequential(*residual_blocks)
        self.linear = conv1x1_bn_act(features, features, activation=activation)

    def forward(self, x: 'Tensor') ->Tensor:
        x = self.residuals(x)
        x = self.linear(x)
        return x


class HGSupervisionBlock(nn.Module):

    def __init__(self, features, supervision_channels: 'int'):
        super().__init__()
        self.squeeze = nn.Conv2d(features, supervision_channels, kernel_size=1)
        self.expand = nn.Conv2d(supervision_channels, features, kernel_size=1)

    def forward(self, x: 'Tensor') ->Tuple[Tensor, Tensor]:
        sup_mask = self.squeeze(x)
        sup_features = self.expand(sup_mask)
        return sup_mask, sup_features


HRNETV2_BN_MOMENTUM = 0.1


def hrnet_conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class HRNetBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(HRNetBasicBlock, self).__init__()
        self.conv1 = hrnet_conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=HRNETV2_BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = hrnet_conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=HRNETV2_BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class HRNetBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(HRNetBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=HRNETV2_BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=HRNETV2_BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion, momentum=HRNETV2_BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class HighResolutionModule(nn.Module):

    def __init__(self, num_branches, blocks, num_blocks, num_inchannels, num_channels, fuse_method, multi_scale_output=True):
        super(HighResolutionModule, self).__init__()
        self._check_branches(num_branches, blocks, num_blocks, num_inchannels, num_channels)
        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches
        self.multi_scale_output = multi_scale_output
        self.branches = self._make_branches(num_branches, blocks, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(inplace=True)

    def _check_branches(self, num_branches, blocks, num_blocks, num_inchannels, num_channels):
        if num_branches != len(num_blocks):
            error_msg = 'NUM_BRANCHES({}) <> NUM_BLOCKS({})'.format(num_branches, len(num_blocks))
            raise ValueError(error_msg)
        if num_branches != len(num_channels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_CHANNELS({})'.format(num_branches, len(num_channels))
            raise ValueError(error_msg)
        if num_branches != len(num_inchannels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_INCHANNELS({})'.format(num_branches, len(num_inchannels))
            raise ValueError(error_msg)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels, stride=1):
        downsample = None
        if stride != 1 or self.num_inchannels[branch_index] != num_channels[branch_index] * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.num_inchannels[branch_index], num_channels[branch_index] * block.expansion, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(num_channels[branch_index] * block.expansion, momentum=HRNETV2_BN_MOMENTUM))
        layers = []
        layers.append(block(self.num_inchannels[branch_index], num_channels[branch_index], stride, downsample))
        self.num_inchannels[branch_index] = num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(block(self.num_inchannels[branch_index], num_channels[branch_index]))
        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []
        for i in range(num_branches):
            branches.append(self._make_one_branch(i, block, num_blocks, num_channels))
        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None
        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(nn.Sequential(nn.Conv2d(num_inchannels[j], num_inchannels[i], 1, 1, 0, bias=False), nn.BatchNorm2d(num_inchannels[i], momentum=HRNETV2_BN_MOMENTUM)))
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(nn.Sequential(nn.Conv2d(num_inchannels[j], num_outchannels_conv3x3, 3, 2, 1, bias=False), nn.BatchNorm2d(num_outchannels_conv3x3, momentum=HRNETV2_BN_MOMENTUM)))
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(nn.Sequential(nn.Conv2d(num_inchannels[j], num_outchannels_conv3x3, 3, 2, 1, bias=False), nn.BatchNorm2d(num_outchannels_conv3x3, momentum=HRNETV2_BN_MOMENTUM), nn.ReLU(inplace=True)))
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))
        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]
        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])
        x_fuse = []
        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                elif j > i:
                    width_output = x[i].shape[-1]
                    height_output = x[i].shape[-2]
                    y = y + F.interpolate(self.fuse_layers[i][j](x[j]), size=(height_output, width_output), mode='bilinear', align_corners=False)
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))
        return x_fuse


class Mlp(nn.Module):
    """Multilayer perceptron."""

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


class Attention(nn.Module):

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0, sr_ratio=1):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f'dim {dim} should be divisible by num_heads {num_heads}.')
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        attn = q @ k.transpose(-2, -1) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio, qkv_bias: 'bool', qk_scale: 'Optional[float]', drop: 'float', attn_drop: 'float', drop_path: 'float', activation: 'str', sr_ratio: 'int', norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, activation=activation, drop=drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        """

        :param x: B, H * W, C
        :param H: Height
        :param W: Width
        :return: B, H * W, C
        """
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x


class OverlapPatchEmbed(nn.Module):
    """Image to Patch Embedding"""

    def __init__(self, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride, padding=patch_size // 2)
        self.norm = nn.LayerNorm(embed_dim)
        self.stride = stride
        self.apply(self._init_weights)

    def change_input_channels(self, input_channels: 'int', mode='auto', **kwargs):
        self.proj = nn.Conv2d(input_channels, self.embed_dim, kernel_size=self.patch_size, stride=self.stride, padding=self.patch_size // 2)
        return self

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W


class DWConv(nn.Module):

    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class WindowAttention(nn.Module):
    """Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.relative_position_bias_table = nn.Parameter(torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer('relative_position_index', relative_position_index)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """Forward function.

        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class SwinTransformerBlock(nn.Module):
    """Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, num_heads, window_size=7, shift_size=0, mlp_ratio=4.0, qkv_bias=True, qk_scale=None, drop=0.0, attn_drop=0.0, drop_path=0.0, act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert 0 <= self.shift_size < self.window_size, 'shift_size must in 0-window_size'
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(dim, window_size=to_2tuple(self.window_size), num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.H = None
        self.W = None

    def forward(self, x, mask_matrix):
        """Forward function.

        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
            mask_matrix: Attention mask for cyclic shift.
        """
        B, L, C = x.shape
        H, W = self.H, self.W
        assert L == H * W, 'input feature has wrong size'
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        attn_windows = self.attn(x_windows, mask=attn_mask)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()
        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchMerging(nn.Module):
    """Patch Merging Layer

    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x, H, W):
        """Forward function.

        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """
        B, L, C = x.shape
        assert L == H * W, 'input feature has wrong size'
        x = x.view(B, H, W, C)
        pad_input = H % 2 == 1 or W % 2 == 1
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)
        x = x.view(B, -1, 4 * C)
        x = self.norm(x)
        x = self.reduction(x)
        return x


class BasicLayer(nn.Module):
    """A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of feature channels
        depth (int): Depths of this stage.
        num_heads (int): Number of attention head.
        window_size (int): Local window size. Default: 7.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, depth, num_heads, window_size=7, mlp_ratio=4.0, qkv_bias=True, qk_scale=None, drop=0.0, attn_drop=0.0, drop_path=0.0, act_layer=nn.GELU, norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):
        super().__init__()
        self.window_size = window_size
        self.shift_size = window_size // 2
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.blocks = nn.ModuleList([SwinTransformerBlock(dim=dim, num_heads=num_heads, window_size=window_size, shift_size=0 if i % 2 == 0 else window_size // 2, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop, attn_drop=attn_drop, drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path, act_layer=act_layer, norm_layer=norm_layer) for i in range(depth)])
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, H, W):
        """
        Forward function.

        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)
        h_slices = slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None)
        w_slices = slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None)
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1
        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        for blk in self.blocks:
            blk.H, blk.W = H, W
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, attn_mask)
            else:
                x = blk(x, attn_mask)
        if self.downsample is not None:
            x_down = self.downsample(x, H, W)
            Wh, Ww = (H + 1) // 2, (W + 1) // 2
            return x, H, W, x_down, Wh, Ww
        else:
            return x, H, W, x, H, W


class PatchEmbed(nn.Module):
    """Image to Patch Embedding

    Args:
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """Forward function."""
        _, _, H, W = x.size()
        if W % self.patch_size[1] != 0:
            x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1]))
        if H % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))
        x = self.proj(x)
        if self.norm is not None:
            Wh, Ww = x.size(2), x.size(3)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, Wh, Ww)
        return x


def make_conv_bn_act(in_channels: 'int', out_channels: 'int', kernel_size: 'int'=3, stride: 'int'=1, zero_batch_norm: 'bool'=False, use_activation: 'bool'=True, activation: 'str'=ACT_RELU) ->torch.nn.Sequential:
    """
    Create a nn.Conv2d block followed by nn.BatchNorm2d and (optional) activation block.
    """
    batch_norm = nn.BatchNorm2d(out_channels)
    nn.init.constant_(batch_norm.weight, 0.0 if zero_batch_norm else 1.0)
    layers = [('conv', nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=kernel_size // 2, bias=False)), ('bn', batch_norm)]
    if use_activation:
        activation_block = instantiate_activation_block(activation, inplace=True)
        layers.append((activation, activation_block))
    return nn.Sequential(OrderedDict(layers))


class StemBlock(nn.Module):

    def __init__(self, input_channels: 'int', output_channels: 'int', activation: 'str'=ACT_RELU):
        super().__init__()
        self.conv_bn_relu_1 = make_conv_bn_act(input_channels, 8, stride=2, activation=activation)
        self.conv_bn_relu_2 = make_conv_bn_act(8, 64, activation=activation)
        self.conv_bn_relu_3 = make_conv_bn_act(64, output_channels, activation=activation)

    def forward(self, x):
        x = self.conv_bn_relu_1(x)
        x = self.conv_bn_relu_2(x)
        x = self.conv_bn_relu_3(x)
        return x


class XResNetBlock(nn.Module):
    """Creates the standard `XResNet` block."""

    def __init__(self, expansion: 'int', n_inputs: 'int', n_hidden: 'int', stride: 'int'=1, activation: 'str'=ACT_RELU):
        super().__init__()
        n_inputs = n_inputs * expansion
        n_filters = n_hidden * expansion
        if expansion == 1:
            layers = [make_conv_bn_act(n_inputs, n_hidden, 3, stride=stride, activation=activation), make_conv_bn_act(n_hidden, n_filters, 3, zero_batch_norm=True, use_activation=False)]
        else:
            layers = [make_conv_bn_act(n_inputs, n_hidden, 1, activation=activation), make_conv_bn_act(n_hidden, n_hidden, 3, stride=stride, activation=activation), make_conv_bn_act(n_hidden, n_filters, 1, zero_batch_norm=True, use_activation=False)]
        self.convs = nn.Sequential(*layers)
        self.activation = instantiate_activation_block(activation, inplace=True)
        if n_inputs == n_filters:
            self.id_conv = nn.Identity()
        else:
            self.id_conv = make_conv_bn_act(n_inputs, n_filters, kernel_size=1, use_activation=False)
        if stride == 1:
            self.pool = nn.Identity()
        else:
            self.pool = nn.AvgPool2d(2, ceil_mode=True)

    def forward(self, x):
        return self.activation(self.convs(x) + self.id_conv(self.pool(x)))


class ChannelGate2d(nn.Module):
    """
    Channel Squeeze module
    """

    def __init__(self, channels):
        super().__init__()
        self.squeeze = nn.Conv2d(channels, 1, kernel_size=1, padding=0)

    def forward(self, x: 'Tensor'):
        module_input = x
        x = self.squeeze(x)
        x = x.sigmoid()
        return module_input * x


class SpatialGate2d(nn.Module):
    """
    Spatial squeeze module
    """

    def __init__(self, channels, reduction=None, squeeze_channels=None):
        """
        Instantiate module

        :param channels: Number of input channels
        :param reduction: Reduction factor
        :param squeeze_channels: Number of channels in squeeze block.
        """
        super().__init__()
        assert reduction or squeeze_channels, "One of 'reduction' and 'squeeze_channels' must be set"
        assert not (reduction and squeeze_channels), "'reduction' and 'squeeze_channels' are mutually exclusive"
        if squeeze_channels is None:
            squeeze_channels = max(1, channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.squeeze = nn.Conv2d(channels, squeeze_channels, kernel_size=1)
        self.expand = nn.Conv2d(squeeze_channels, channels, kernel_size=1)

    def forward(self, x: 'Tensor'):
        module_input = x
        x = self.avg_pool(x)
        x = self.squeeze(x)
        x = F.relu(x, inplace=True)
        x = self.expand(x)
        x = x.sigmoid()
        return module_input * x


class ChannelSpatialGate2d(nn.Module):
    """
    Concurrent Spatial and Channel Squeeze & Excitation
    """

    def __init__(self, channels, reduction=4):
        super().__init__()
        self.channel_gate = ChannelGate2d(channels)
        self.spatial_gate = SpatialGate2d(channels, reduction=reduction)

    def forward(self, x):
        return self.channel_gate(x) + self.spatial_gate(x)


class SEXResNetBlock(nn.Module):
    """Creates the Squeeze&Excitation + XResNet block."""

    def __init__(self, expansion: 'int', n_inputs: 'int', n_hidden: 'int', stride: 'int'=1, activation: 'str'=ACT_RELU):
        super().__init__()
        n_inputs = n_inputs * expansion
        n_filters = n_hidden * expansion
        if expansion == 1:
            layers = [make_conv_bn_act(n_inputs, n_hidden, 3, stride=stride, activation=activation), make_conv_bn_act(n_hidden, n_filters, 3, zero_batch_norm=True, use_activation=False)]
        else:
            layers = [make_conv_bn_act(n_inputs, n_hidden, 1, activation=activation), make_conv_bn_act(n_hidden, n_hidden, 3, stride=stride, activation=activation), make_conv_bn_act(n_hidden, n_filters, 1, zero_batch_norm=True, use_activation=False)]
        self.convs = nn.Sequential(*layers)
        self.activation = instantiate_activation_block(activation, inplace=True)
        self.se = ChannelSpatialGate2d(n_filters, reduction=4)
        if n_inputs == n_filters:
            self.id_conv = nn.Identity()
        else:
            self.id_conv = make_conv_bn_act(n_inputs, n_filters, kernel_size=1, use_activation=False)
        if stride == 1:
            self.pool = nn.Identity()
        else:
            self.pool = nn.AvgPool2d(2, ceil_mode=True)

    def forward(self, x):
        return self.activation(self.convs(x) + self.id_conv(self.pool(x)))


class FPNContextBlock(nn.Module):

    def __init__(self, in_channels: 'int', out_channels: 'int', abn_block=ABN, dropout=0.0):
        """
        Center FPN block that aggregates multi-scale context using strided average poolings

        :param in_channels: Number of input features
        :param out_channels: Number of output features
        :param abn_block: Block for Activation + BatchNorm2d
        :param dropout: Dropout rate after context fusion
        """
        super().__init__()
        self.bottleneck = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.proj2 = nn.Conv2d(in_channels // 2, in_channels // 8, kernel_size=1)
        self.pool4 = nn.AvgPool2d(kernel_size=4, stride=4)
        self.proj4 = nn.Conv2d(in_channels // 2, in_channels // 8, kernel_size=1)
        self.pool8 = nn.AvgPool2d(kernel_size=8, stride=8)
        self.proj8 = nn.Conv2d(in_channels // 2, in_channels // 8, kernel_size=1)
        self.pool_global = nn.AdaptiveAvgPool2d(1)
        self.proj_global = nn.Conv2d(in_channels // 2, in_channels // 8, kernel_size=1)
        self.blend = nn.Conv2d(4 * in_channels // 8, out_channels, kernel_size=1)
        self.conv1 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.abn1 = abn_block(out_channels)
        self.dropout = nn.Dropout2d(dropout, inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.abn2 = abn_block(out_channels)

    def forward(self, x: 'Tensor') ->Tensor:
        x = self.bottleneck(x)
        p2 = self.proj2(self.pool2(x))
        p4 = self.proj4(self.pool4(x))
        p8 = self.proj8(self.pool8(x))
        pg = self.proj_global(self.pool_global(x))
        out_size = p2.size()[2:]
        x = torch.cat([p2, F.interpolate(p4, size=out_size, mode='nearest'), F.interpolate(p8, size=out_size, mode='nearest'), F.interpolate(pg, size=out_size, mode='nearest')], dim=1)
        x = self.blend(x)
        x = self.conv1(x)
        x = self.abn1(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.abn2(x)
        return x


class FPNBottleneckBlock(nn.Module):

    def __init__(self, in_channels: 'int', out_channels: 'int', abn_block=ABN, dropout=0.0):
        """

        Args:
            encoder_features:
            decoder_features:
            output_features:
            supervision_channels:
            abn_block:
            dropout:
        """
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.abn1 = abn_block(out_channels)
        self.drop1 = nn.Dropout2d(dropout, inplace=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.abn2 = abn_block(out_channels)

    def forward(self, x: 'Tensor') ->Tensor:
        x = self.conv1(x)
        x = self.abn1(x)
        x = self.drop1(x)
        x = self.conv2(x)
        x = self.abn2(x)
        return x


class FPNFuse(nn.Module):

    def __init__(self, mode='bilinear', align_corners=False):
        super().__init__()
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, features: 'List[Tensor]'):
        layers = []
        dst_size = features[0].size()[2:]
        for f in features:
            layers.append(F.interpolate(f, size=dst_size, mode=self.mode, align_corners=self.align_corners))
        return torch.cat(layers, dim=1)


class FPNFuseSum(nn.Module):
    """Compute a sum of individual FPN layers"""

    def __init__(self, mode='bilinear', align_corners=False):
        super().__init__()
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, features: 'List[Tensor]') ->Tensor:
        output = features[0]
        dst_size = features[0].size()[2:]
        for f in features[1:]:
            output = output + F.interpolate(f, size=dst_size, mode=self.mode, align_corners=self.align_corners)
        return output


class HFF(nn.Module):
    """
    Hierarchical feature fusion module.
    https://arxiv.org/pdf/1811.11431.pdf
    https://arxiv.org/pdf/1803.06815.pdf

    What it does is easily explained in code:
    feature_map_0 - feature_map of the highest resolution
    feature_map_N - feature_map of the smallest resolution

    >>> feature_map = feature_map_0 + up(feature_map[1] + up(feature_map[2] + up(feature_map[3] + ...))))
    """

    def __init__(self, sizes=None, upsample_scale=2, mode='nearest', align_corners=None):
        super().__init__()
        self.sizes = sizes
        self.interpolation_mode = mode
        self.align_corners = align_corners
        self.upsample_scale = upsample_scale

    def forward(self, features: 'List[Tensor]') ->Tensor:
        num_feature_maps = len(features)
        current_map = features[-1]
        for feature_map_index in reversed(range(num_feature_maps - 1)):
            if self.sizes is not None:
                prev_upsampled = self._upsample(current_map, self.sizes[feature_map_index])
            else:
                prev_upsampled = self._upsample(current_map)
            current_map = features[feature_map_index] + prev_upsampled
        return current_map

    def _upsample(self, x, output_size=None):
        if output_size is not None:
            x = F.interpolate(x, size=(output_size[0], output_size[1]), mode=self.interpolation_mode, align_corners=self.align_corners)
        else:
            x = F.interpolate(x, scale_factor=self.upsample_scale, mode=self.interpolation_mode, align_corners=self.align_corners)
        return x


class ProgressiveShuffleBlock(nn.Module):

    def __init__(self, in_channels: 'int', out_channels: 'int', activation: 'str', normalization: 'str'=NORM_BATCH, normalization_kwargs: 'Optional[Mapping]'=None):
        super().__init__()
        if normalization_kwargs is None:
            normalization_kwargs = {}
        self.conv = nn.Sequential(nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False), instantiate_normalization_block(normalization, in_channels, **normalization_kwargs), instantiate_activation_block(activation, inplace=True), nn.Conv2d(in_channels, out_channels * 4, kernel_size=1, padding=0, bias=False))
        self.shuffle = nn.PixelShuffle(2)

    def forward(self, x):
        return self.shuffle(self.conv(x))


class _SelfAttentionBlock(nn.Module):
    """
    The basic implementation for self-attention block/non-local block
    Input:
        N X C X H X W
    Parameters:
        in_channels       : the dimension of the input feature map
        key_channels      : the dimension after the key/query transform
        value_channels    : the dimension after the value transform
        scale             : choose the scale to downsample the input feature maps (save memory cost)
    Return:
        N X C X H X W
        position-aware context features.(w/o concate or add with the input)
    """

    def __init__(self, in_channels, key_channels, value_channels, out_channels=None, scale=1, abn_block=ABN):
        super(_SelfAttentionBlock, self).__init__()
        self.scale = scale
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.key_channels = key_channels
        self.value_channels = value_channels
        if out_channels is None:
            self.out_channels = in_channels
        self.pool = nn.MaxPool2d(kernel_size=(scale, scale))
        self.f_key = nn.Sequential(nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels, kernel_size=1, stride=1, padding=0), abn_block(self.key_channels))
        self.f_query = self.f_key
        self.f_value = nn.Conv2d(in_channels=self.in_channels, out_channels=self.value_channels, kernel_size=1, stride=1, padding=0)
        self.W = nn.Conv2d(in_channels=self.value_channels, out_channels=self.out_channels, kernel_size=1, stride=1, padding=0)
        nn.init.constant_(self.W.bias, 0)

    def forward(self, x):
        batch_size, h, w = x.size(0), x.size(2), x.size(3)
        if self.scale > 1:
            x = self.pool(x)
        value = self.f_value(x).view(batch_size, self.value_channels, -1)
        value = value.permute(0, 2, 1)
        query = self.f_query(x).view(batch_size, self.key_channels, -1)
        query = query.permute(0, 2, 1)
        key = self.f_key(x).view(batch_size, self.key_channels, -1)
        sim_map = torch.matmul(query, key)
        sim_map = self.key_channels ** -0.5 * sim_map
        sim_map = F.softmax(sim_map, dim=-1)
        context = torch.matmul(sim_map, value)
        context = context.permute(0, 2, 1).contiguous()
        context = context.view(batch_size, self.value_channels, *x.size()[2:])
        context = self.W(context)
        if self.scale > 1:
            context = F.interpolate(input=context, size=(h, w), mode='bilinear', align_corners=False)
        return context


class SelfAttentionBlock2D(_SelfAttentionBlock):

    def __init__(self, in_channels, key_channels, value_channels, out_channels=None, scale=1):
        super(SelfAttentionBlock2D, self).__init__(in_channels, key_channels, value_channels, out_channels, scale)


class BaseOC_Module(nn.Module):
    """
    Implementation of the BaseOC module
    Parameters:
        in_features / out_features: the channels of the input / output feature maps.
        dropout: we choose 0.05 as the default value.
        size: you can apply multiple sizes. Here we only use one size.
    Return:
        features fused with Object context information.
    """

    def __init__(self, in_channels, out_channels, key_channels, value_channels, dropout, sizes=[1], abn_block=ABN):
        super(BaseOC_Module, self).__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(in_channels, out_channels, key_channels, value_channels, size) for size in sizes])
        self.conv_bn_dropout = nn.Sequential(nn.Conv2d(2 * in_channels, out_channels, kernel_size=1, padding=0), abn_block(out_channels), nn.Dropout2d(dropout))

    def _make_stage(self, in_channels, output_channels, key_channels, value_channels, size):
        return SelfAttentionBlock2D(in_channels, key_channels, value_channels, output_channels, size)

    def forward(self, feats):
        priors = [stage(feats) for stage in self.stages]
        context = priors[0]
        for i in range(1, len(priors)):
            context += priors[i]
        output = self.conv_bn_dropout(torch.cat([context, feats], 1))
        return output


class ObjectContextBlock(nn.Module):
    """
    Output only the context features.
    Parameters:
        in_features / out_features: the channels of the input / output feature maps.
        dropout: specify the dropout ratio
        fusion: We provide two different fusion method, "concat" or "add"
        size: we find that directly learn the attention weights on even 1/8 feature maps is hard.
    Return:
        features after "concat" or "add"
    """

    def __init__(self, in_channels, out_channels, key_channels, value_channels, dropout, sizes=[1], abn_block=ABN):
        super(ObjectContextBlock, self).__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(in_channels, out_channels, key_channels, value_channels, size) for size in sizes])
        self.conv_bn_dropout = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=False), abn_block(out_channels))

    def _make_stage(self, in_channels, output_channels, key_channels, value_channels, size):
        return SelfAttentionBlock2D(in_channels, key_channels, value_channels, output_channels, size)

    def forward(self, feats):
        priors = [stage(feats) for stage in self.stages]
        context = priors[0]
        for i in range(1, len(priors)):
            context += priors[i]
        output = self.conv_bn_dropout(context)
        return output


class ASPObjectContextBlock(nn.Module):

    def __init__(self, features, out_features=256, dilations=(12, 24, 36), abn_block=ABN, dropout=0.1):
        super(ASPObjectContextBlock, self).__init__()
        self.context = nn.Sequential(nn.Conv2d(features, out_features, kernel_size=3, padding=1, dilation=1, bias=False), abn_block(out_features), ObjectContextBlock(in_channels=out_features, out_channels=out_features, key_channels=out_features // 2, value_channels=out_features, dropout=dropout, sizes=[2]))
        self.conv2 = nn.Sequential(nn.Conv2d(features, out_features, kernel_size=1, padding=0, dilation=1, bias=False), abn_block(out_features))
        self.conv3 = nn.Sequential(nn.Conv2d(features, out_features, kernel_size=3, padding=dilations[0], dilation=dilations[0], bias=False), abn_block(out_features))
        self.conv4 = nn.Sequential(nn.Conv2d(features, out_features, kernel_size=3, padding=dilations[1], dilation=dilations[1], bias=False), abn_block(out_features))
        self.conv5 = nn.Sequential(nn.Conv2d(features, out_features, kernel_size=3, padding=dilations[2], dilation=dilations[2], bias=False), abn_block(out_features))
        self.conv_bn_dropout = nn.Sequential(nn.Conv2d(out_features * 5, out_features * 2, kernel_size=1, padding=0, dilation=1, bias=False), abn_block(out_features * 2), nn.Dropout2d(dropout))

    def _cat_each(self, feat1, feat2, feat3, feat4, feat5):
        assert len(feat1) == len(feat2)
        z = []
        for i in range(len(feat1)):
            z.append(torch.cat((feat1[i], feat2[i], feat3[i], feat4[i], feat5[i]), dim=1))
        return z

    def forward(self, x):
        if isinstance(x, torch.Tensor):
            _, _, h, w = x.size()
        elif isinstance(x, tuple) or isinstance(x, list):
            _, _, h, w = x[0].size()
        else:
            raise RuntimeError('unknown input type')
        feat1 = self.context(x)
        feat2 = self.conv2(x)
        feat3 = self.conv3(x)
        feat4 = self.conv4(x)
        feat5 = self.conv5(x)
        if isinstance(x, torch.Tensor):
            out = torch.cat((feat1, feat2, feat3, feat4, feat5), dim=1)
        elif isinstance(x, tuple) or isinstance(x, list):
            out = self._cat_each(feat1, feat2, feat3, feat4, feat5)
        else:
            raise RuntimeError('unknown input type')
        output = self.conv_bn_dropout(out)
        return output


class _PyramidSelfAttentionBlock(nn.Module):
    """
    The basic implementation for self-attention block/non-local block
    Input:
        N X C X H X W
    Parameters:
        in_channels       : the dimension of the input feature map
        key_channels      : the dimension after the key/query transform
        value_channels    : the dimension after the value transform
        scale             : choose the scale to downsample the input feature maps
    Return:
        N X C X H X W
        position-aware context features.(w/o concate or add with the input)
    """

    def __init__(self, in_channels, key_channels, value_channels, out_channels=None, scale=1, abn_block=ABN):
        super(_PyramidSelfAttentionBlock, self).__init__()
        self.scale = scale
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.key_channels = key_channels
        self.value_channels = value_channels
        if out_channels is None:
            self.out_channels = in_channels
        self.f_key = nn.Sequential(nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels, kernel_size=1, stride=1, padding=0, bias=False), abn_block(self.key_channels))
        self.f_query = self.f_key
        self.f_value = nn.Conv2d(in_channels=self.in_channels, out_channels=self.value_channels, kernel_size=1, stride=1, padding=0)
        self.W = nn.Conv2d(in_channels=self.value_channels, out_channels=self.out_channels, kernel_size=1, stride=1, padding=0)
        nn.init.constant(self.W.bias, 0)

    def forward(self, x):
        batch_size, _, h, w = x.size(0), x.size(1), x.size(2), x.size(3)
        local_x = []
        local_y = []
        step_h, step_w = h // self.scale, w // self.scale
        for i in range(0, self.scale):
            for j in range(0, self.scale):
                start_x, start_y = i * step_h, j * step_w
                end_x, end_y = min(start_x + step_h, h), min(start_y + step_w, w)
                if i == self.scale - 1:
                    end_x = h
                if j == self.scale - 1:
                    end_y = w
                local_x += [start_x, end_x]
                local_y += [start_y, end_y]
        value = self.f_value(x)
        query = self.f_query(x)
        key = self.f_key(x)
        local_list = []
        local_block_cnt = 2 * self.scale * self.scale
        for i in range(0, local_block_cnt, 2):
            value_local = value[:, :, local_x[i]:local_x[i + 1], local_y[i]:local_y[i + 1]]
            query_local = query[:, :, local_x[i]:local_x[i + 1], local_y[i]:local_y[i + 1]]
            key_local = key[:, :, local_x[i]:local_x[i + 1], local_y[i]:local_y[i + 1]]
            h_local, w_local = value_local.size(2), value_local.size(3)
            value_local = value_local.contiguous().view(batch_size, self.value_channels, -1)
            value_local = value_local.permute(0, 2, 1)
            query_local = query_local.contiguous().view(batch_size, self.key_channels, -1)
            query_local = query_local.permute(0, 2, 1)
            key_local = key_local.contiguous().view(batch_size, self.key_channels, -1)
            sim_map = torch.matmul(query_local, key_local)
            sim_map = self.key_channels ** -0.5 * sim_map
            sim_map = F.softmax(sim_map, dim=-1)
            context_local = torch.matmul(sim_map, value_local)
            context_local = context_local.permute(0, 2, 1).contiguous()
            context_local = context_local.view(batch_size, self.value_channels, h_local, w_local)
            local_list.append(context_local)
        context_list = []
        for i in range(0, self.scale):
            row_tmp = []
            for j in range(0, self.scale):
                row_tmp.append(local_list[j + i * self.scale])
            context_list.append(torch.cat(row_tmp, 3))
        context = torch.cat(context_list, 2)
        context = self.W(context)
        return context


class PyramidSelfAttentionBlock2D(_PyramidSelfAttentionBlock):

    def __init__(self, in_channels, key_channels, value_channels, out_channels=None, scale=1):
        super(PyramidSelfAttentionBlock2D, self).__init__(in_channels, key_channels, value_channels, out_channels, scale)


class PyramidObjectContextBlock(nn.Module):
    """
    Output the combination of the context features and the original features.
    Parameters:
        in_features / out_features: the channels of the input / output feature maps.
        dropout: specify the dropout ratio
        size: we find that directly learn the attention weights on even 1/8 feature maps is hard.
    Return:
        features after "concat" or "add"
    """

    def __init__(self, in_channels, out_channels, dropout=0.05, sizes=[1, 2, 3, 6], abn_block=ABN):
        super(PyramidObjectContextBlock, self).__init__()
        self.group = len(sizes)
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(in_channels, out_channels, in_channels // 2, in_channels, size) for size in sizes])
        self.conv_bn_dropout = nn.Sequential(nn.Conv2d(2 * in_channels * self.group, out_channels, kernel_size=1, padding=0, bias=False), abn_block(out_channels), nn.Dropout2d(dropout))
        self.up_dr = nn.Sequential(nn.Conv2d(in_channels, in_channels * self.group, kernel_size=1, padding=0, bias=False), abn_block(in_channels * self.group))

    def _make_stage(self, in_channels, output_channels, key_channels, value_channels, size):
        return PyramidSelfAttentionBlock2D(in_channels, key_channels, value_channels, output_channels, size)

    def forward(self, feats):
        priors = [stage(feats) for stage in self.stages]
        context = [self.up_dr(feats)]
        for i in range(0, len(priors)):
            context += [priors[i]]
        output = self.conv_bn_dropout(torch.cat(context, 1))
        return output


class GlobalMaxPool2d(nn.Module):

    def __init__(self, flatten=False):
        """Global average pooling over the input's spatial dimensions"""
        super().__init__()
        self.flatten = flatten

    def forward(self, x):
        x = F.adaptive_max_pool2d(x, output_size=1)
        if self.flatten:
            x = x.view(x.size(0), x.size(1))
        return x


class GlobalKMaxPool2d(nn.Module):
    """
    K-max global pooling block

    https://arxiv.org/abs/1911.07344
    """

    def __init__(self, k=4, trainable=True, flatten=False):
        """Global average pooling over the input's spatial dimensions"""
        super().__init__()
        self.k = k
        self.flatten = flatten
        self.trainable = trainable
        weights = torch.ones((1, 1, k))
        if trainable:
            self.register_parameter('weights', torch.nn.Parameter(weights))
        else:
            self.register_buffer('weights', weights)

    def forward(self, x: 'Tensor'):
        input = x.view(x.size(0), x.size(1), -1)
        kmax = input.topk(k=self.k, dim=2)[0]
        kmax = (kmax * self.weights).mean(dim=2)
        if not self.flatten:
            kmax = kmax.view(kmax.size(0), kmax.size(1), 1, 1)
        return kmax

    def load_state_dict(self, state_dict: 'Union[Dict[str, Tensor], Dict[str, Tensor]]', strict: 'bool'=True):
        if not self.trainable:
            return _IncompatibleKeys([], [])
        super().load_state_dict(state_dict, strict)


class GlobalWeightedAvgPool2d(nn.Module):
    """
    Global Weighted Average Pooling from paper "Global Weighted Average
    Pooling Bridges Pixel-level Localization and Image-level Classification"
    """

    def __init__(self, features: 'int', flatten=False):
        super().__init__()
        self.conv = nn.Conv2d(features, 1, kernel_size=1, bias=True)
        self.flatten = flatten

    def fscore(self, x):
        m = self.conv(x)
        m = m.sigmoid().exp()
        return m

    def norm(self, x: 'torch.Tensor'):
        return x / x.sum(dim=[2, 3], keepdim=True)

    def forward(self, x):
        input_x = x
        x = self.fscore(x)
        x = self.norm(x)
        x = x * input_x
        x = x.sum(dim=[2, 3], keepdim=not self.flatten)
        return x


class RMSPool(nn.Module):
    """
    Root mean square pooling
    """

    def __init__(self):
        super().__init__()
        self.avg_pool = GlobalAvgPool2d()

    def forward(self, x):
        x_mean = x.mean(dim=[2, 3])
        avg_pool = self.avg_pool((x - x_mean) ** 2)
        return avg_pool.sqrt()


class MILCustomPoolingModule(nn.Module):

    def __init__(self, in_channels, out_channels, reduction=4):
        super().__init__()
        self.classifier = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.weight_generator = nn.Sequential(nn.BatchNorm2d(in_channels), nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1), nn.ReLU(True), nn.Conv2d(in_channels // reduction, out_channels, kernel_size=1), nn.Sigmoid())

    def forward(self, x):
        weight = self.weight_generator(x)
        loss = self.classifier(x)
        logits = torch.sum(weight * loss, dim=[2, 3]) / (torch.sum(weight, dim=[2, 3]) + 1e-06)
        return logits


class GlobalRankPooling(nn.Module):
    """
    https://arxiv.org/abs/1704.02112
    """

    def __init__(self, num_features, spatial_size, flatten=False):
        super().__init__()
        self.conv = nn.Conv1d(num_features, num_features, spatial_size, groups=num_features)
        self.flatten = flatten

    def forward(self, x: 'torch.Tensor'):
        spatial_size = x.size(2) * x.size(3)
        assert spatial_size == self.conv.kernel_size[0], f'Expected spatial size {self.conv.kernel_size[0]}, got {x.size(2)}x{x.size(3)}'
        x = x.view(x.size(0), x.size(1), -1)
        x_sorted, index = x.topk(spatial_size, dim=2)
        x = self.conv(x_sorted)
        if self.flatten:
            x = x.squeeze(2)
        return x


class GeneralizedMeanPooling2d(nn.Module):
    """

    https://arxiv.org/pdf/1902.05509v2.pdf
    https://amaarora.github.io/2020/08/30/gempool.html
    """

    def __init__(self, p: 'float'=3, eps=1e-06, flatten=False, l2_normalize: 'bool'=False):
        super(GeneralizedMeanPooling2d, self).__init__()
        self.register_parameter('p', nn.Parameter(torch.ones(1) * p))
        self.eps = eps
        self.flatten = flatten
        self.l2_normalize = l2_normalize

    def forward(self, x: 'Tensor') ->Tensor:
        p = F.softplus(self.p) + 1
        x = F.adaptive_avg_pool2d(x.clamp_min(self.eps).pow(p), output_size=1).pow(1.0 / p)
        if self.l2_normalize:
            x = F.normalize(x, p=2, dim=1)
        if self.flatten:
            x = x.view(x.size(0), x.size(1))
        return x

    def __repr__(self):
        p = torch.softplus(self.p) + 1
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(p.item()) + ', ' + 'eps=' + str(self.eps) + ')'


class GlobalMaxAvgPooling2d(nn.Module):

    def __init__(self, flatten: 'bool'=False):
        super().__init__()
        self.max_pooling = nn.AdaptiveMaxPool2d((1, 1))
        self.avg_pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = flatten

    def forward(self, x):
        x_max = self.max_pooling(x).flatten(start_dim=1)
        x_avg = self.avg_pooling(x).flatten(start_dim=1)
        y = torch.cat([x_max, x_avg], dim=1)
        if self.flatten:
            y = torch.flatten(y, 1)
        return y


class SpatialGate2dV2(nn.Module):
    """
    Spatial Squeeze and Channel Excitation module
    """

    def __init__(self, channels, reduction=4):
        super().__init__()
        squeeze_channels = max(1, channels // reduction)
        self.squeeze = nn.Conv2d(channels, squeeze_channels, kernel_size=1, padding=0)
        self.conv = nn.Conv2d(squeeze_channels, squeeze_channels, kernel_size=7, dilation=3, padding=3 * 3)
        self.expand = nn.Conv2d(squeeze_channels, channels, kernel_size=1, padding=0)

    def forward(self, x: 'Tensor'):
        module_input = x
        x = self.squeeze(x)
        x = self.conv(x)
        x = F.relu(x, inplace=True)
        x = self.expand(x)
        x = x.sigmoid()
        return module_input * x


class ChannelSpatialGate2dV2(nn.Module):

    def __init__(self, channels, reduction=4):
        super().__init__()
        self.channel_gate = ChannelGate2d(channels)
        self.spatial_gate = SpatialGate2dV2(channels, reduction)

    def forward(self, x):
        return self.channel_gate(x) + self.spatial_gate(x)


class ASPPModule(nn.Module):

    def __init__(self, in_channels: 'int', out_channels: 'int', kernel_size: 'int', padding: 'int', dilation: 'int', norm_layer=nn.BatchNorm2d, activation=ACT_RELU):
        super(ASPPModule, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation, bias=False)
        self.abn = nn.Sequential(OrderedDict([('norm', norm_layer(out_channels)), ('act', instantiate_activation_block(activation, inplace=True))]))

    def forward(self, x):
        x = self.conv(x)
        x = self.abn(x)
        return x


class SeparableASPPModule(nn.Module):

    def __init__(self, in_channels: 'int', out_channels: 'int', kernel_size: 'int', padding: 'int', dilation: 'int', norm_layer=nn.BatchNorm2d, activation=ACT_RELU):
        super().__init__()
        self.conv = DepthwiseSeparableConv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation, bias=False)
        self.abn = nn.Sequential(OrderedDict([('norm', norm_layer(out_channels)), ('act', instantiate_activation_block(activation, inplace=True))]))

    def forward(self, x):
        x = self.conv(x)
        x = self.abn(x)
        return x


class ASPPPooling(nn.Sequential):

    def __init__(self, in_channels: 'int', out_channels: 'int', norm_layer=nn.BatchNorm2d, activation: 'str'=ACT_RELU):
        super().__init__()
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.abn = nn.Sequential(OrderedDict([('norm', norm_layer(out_channels)), ('act', instantiate_activation_block(activation, inplace=True))]))

    def forward(self, x: 'Tensor') ->Tensor:
        size = x.shape[-2:]
        x = self.pooling(x)
        x = self.conv(x)
        x = self.abn(x)
        return torch.nn.functional.interpolate(x, size=size, mode='bilinear', align_corners=False)


class ASPP(nn.Module):

    def __init__(self, in_channels: 'int', out_channels: 'int', aspp_module: 'Union[Type[ASPPModule], Type[SeparableASPPModule]]', atrous_rates=(12, 24, 36), dropout: 'float'=0.5, activation: 'str'=ACT_RELU):
        super(ASPP, self).__init__()
        aspp_modules = [aspp_module(in_channels, out_channels, 3, padding=1, dilation=1, activation=activation), ASPPPooling(in_channels, out_channels)] + [aspp_module(in_channels, out_channels, 3, padding=ar, dilation=ar) for ar in atrous_rates]
        self.aspp = nn.ModuleList(aspp_modules)
        self.project = nn.Sequential(nn.Conv2d(len(self.aspp) * out_channels, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels), instantiate_activation_block(activation, inplace=True), nn.Dropout2d(dropout, inplace=False))

    def forward(self, x: 'Tensor') ->Tensor:
        res = []
        for aspp in self.aspp:
            res.append(aspp(x))
        res = torch.cat(res, dim=1)
        return self.project(res)


class SRMLayer(nn.Module):
    """An implementation of SRM block, proposed in
    "SRM : A Style-based Recalibration Module for Convolutional Neural Networks".

    """

    def __init__(self, channels: 'int'):
        super(SRMLayer, self).__init__()
        self.cfc = nn.Conv1d(channels, channels, kernel_size=2, bias=False, groups=channels)
        self.bn = nn.BatchNorm1d(channels)

    def forward(self, x):
        b, c, _, _ = x.size()
        mean = x.view(b, c, -1).mean(-1).unsqueeze(-1)
        std = x.view(b, c, -1).std(-1).unsqueeze(-1)
        u = torch.cat((mean, std), -1)
        z = self.cfc(u)
        z = self.bn(z)
        g = torch.sigmoid(z)
        g = g.view(b, c, 1, 1)
        return x * g.expand_as(x)


class UnetBlock(nn.Module):
    """
    Vanilla U-Net block containing of two convolutions interleaved with batch-norm and RELU
    """

    def __init__(self, in_channels: 'int', out_channels: 'int', activation=ACT_RELU, normalization=NORM_BATCH, normalization_kwargs=None, activation_kwargs=None):
        super().__init__()
        if activation_kwargs is None:
            activation_kwargs = {'inplace': True}
        if normalization_kwargs is None:
            normalization_kwargs = {}
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False)
        self.norm1 = instantiate_normalization_block(normalization, out_channels, **normalization_kwargs)
        self.act1 = instantiate_activation_block(activation, **activation_kwargs)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False)
        self.norm2 = instantiate_normalization_block(normalization, out_channels, **normalization_kwargs)
        self.act2 = instantiate_activation_block(activation, **activation_kwargs)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.act2(x)
        return x


class UnetResidualBlock(nn.Module):
    """ """

    def __init__(self, in_channels: 'int', out_channels: 'int', activation: 'str'=ACT_RELU, normalization: 'str'=NORM_BATCH, normalization_kwargs=None, activation_kwargs=None, drop_path_rate: 'float'=0.0):
        super().__init__()
        if activation_kwargs is None:
            activation_kwargs = {'inplace': True}
        if normalization_kwargs is None:
            normalization_kwargs = {}
        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, stride=1, bias=False) if in_channels != out_channels else nn.Identity()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False)
        self.norm1 = instantiate_normalization_block(normalization, out_channels, **normalization_kwargs)
        self.act1 = instantiate_activation_block(activation, **activation_kwargs)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False)
        self.norm2 = instantiate_normalization_block(normalization, out_channels, **normalization_kwargs)
        self.act2 = instantiate_activation_block(activation, **activation_kwargs)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()

    def forward(self, x):
        residual = self.residual(x)
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.act2(self.drop_path(x) + residual)
        return x


class AbstractResizeLayer(nn.Module):
    """
    Basic class for all upsampling blocks. It forces the upsample block to have a specific
    signature of forward method.
    """

    def forward(self, x: 'Tensor', output_size: 'Union[Tuple[int, int], torch.Size]') ->Tensor:
        """

        :param x: Input feature map to resize
        :param output_size: Target output size. This serves as a hint for the upsample block.
        :return:
        """
        raise NotImplementedError


class NearestNeighborResizeLayer(AbstractResizeLayer):

    def __init__(self, in_channels: 'int', scale_factor: 'int'):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.scale_factor = scale_factor

    def forward(self, x: 'Tensor', output_size: 'Union[Tuple[int, int], torch.Size]') ->Tensor:
        return nn.functional.interpolate(x, size=output_size, mode='nearest')


class BilinearInterpolationLayer(AbstractResizeLayer):

    def __init__(self, in_channels: 'int', scale_factor: 'int', align_corners=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.scale_factor = scale_factor
        self.align_corners = align_corners

    def forward(self, x: 'Tensor', output_size: 'Union[Tuple[int, int], torch.Size]') ->Tensor:
        return nn.functional.interpolate(x, size=output_size, mode='bilinear', align_corners=self.align_corners)


class PixelShuffle(AbstractResizeLayer):
    """
    Depth to Space feature map upsampling that produces a spatially larger feature map but with smaller number of
    channels. Of the input channels is not divisble by scale_factor^2, an additional 1x1 convolution will be
    applied.

    https://github.com/pytorch/pytorch/pull/5429
    https://arxiv.org/ftp/arxiv/papers/1707/1707.02937.pdf
    """

    def __init__(self, in_channels: 'int', scale_factor: 'int'):
        super().__init__()
        n = 2 ** scale_factor
        self.in_channels = in_channels
        self.out_channels = in_channels // n
        rounded_channels = self.out_channels * n
        self.conv = nn.Conv2d(rounded_channels, self.out_channels * n, kernel_size=1, padding=1, bias=False) if in_channels != rounded_channels else nn.Identity()
        self.shuffle = nn.PixelShuffle(upscale_factor=scale_factor)

    def forward(self, x: 'Tensor', output_size: 'Union[Tuple[int, int], torch.Size]'=None) ->Tensor:
        x = self.shuffle(self.conv(x))
        return x


class PixelShuffleWithLinear(AbstractResizeLayer):
    """
    Depth to Space feature map upsampling that preserves the input channels.
    This block performs grouped convolution to increase number of channels followed by pixel shuffle.

    https://github.com/pytorch/pytorch/pull/5429
    https://arxiv.org/ftp/arxiv/papers/1707/1707.02937.pdf
    """

    def __init__(self, in_channels: 'int', scale_factor: 'int', kernel_size: 'int'=3):
        super().__init__()
        n = scale_factor * scale_factor
        self.conv = nn.Conv2d(in_channels, in_channels * n, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.out_channels = in_channels
        self.shuffle = nn.PixelShuffle(upscale_factor=scale_factor)

    def forward(self, x: 'Tensor', output_size: 'Union[Tuple[int, int], torch.Size]'=None) ->Tensor:
        x = self.shuffle(self.conv(x))
        return x


class BilinearAdditiveUpsample2d(AbstractResizeLayer):
    """
    https://arxiv.org/abs/1707.05847
    """

    def __init__(self, in_channels: 'int', scale_factor: 'int'=2):
        super().__init__()
        self.n = 2 ** scale_factor
        self.in_channels = in_channels
        self.out_channels = in_channels // self.n
        if in_channels % self.n != 0:
            raise ValueError(f'Number of input channels ({in_channels}) must be divisable by n ({self.n})')
        self.in_channels = in_channels
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=scale_factor)

    def forward(self, x: 'Tensor', output_size: 'Optional[List[int]]'=None) ->Tensor:
        x = self.upsample(x)
        n, c, h, w = x.size()
        x = x.reshape(n, self.out_channels, self.n, h, w).mean(2)
        return x


class DeconvolutionUpsample2d(AbstractResizeLayer):

    def __init__(self, in_channels: 'int', scale_factor: 'int'=2):
        if scale_factor != 2:
            raise NotImplementedError('Scale factor other than 2 is not implemented')
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.conv = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=3, padding=1, stride=2)

    def forward(self, x: 'Tensor', output_size: 'Optional[List[int]]'=None) ->Tensor:
        return self.conv(x, output_size=output_size)


class ResidualDeconvolutionUpsample2d(AbstractResizeLayer):

    def __init__(self, in_channels: 'int', scale_factor=2):
        if scale_factor != 2:
            raise NotImplementedError(f'Scale factor other than 2 is not implemented. Got scale factor of {scale_factor}')
        super().__init__()
        n = scale_factor * scale_factor
        self.in_channels = in_channels
        self.out_channels = in_channels // n
        self.conv = nn.ConvTranspose2d(in_channels, in_channels // n, kernel_size=3, padding=1, stride=scale_factor, output_padding=1)
        self.residual = BilinearAdditiveUpsample2d(in_channels, scale_factor=scale_factor)

    def forward(self, x: 'Tensor', output_size: 'Optional[List[int]]') ->Tensor:
        residual_up = self.residual(x)
        return self.conv(x, output_size=residual_up.size()) + residual_up


class NoOp(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input


class SumAll(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input.sum(dim=[1, 2, 3])


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (AMM,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (ASPObjectContextBlock,
     lambda: ([], {'features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ASPPModule,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4, 'padding': 4, 'dilation': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ASPPPooling,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (AddCoords,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (BalancedBCEWithLogitsLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (BaseOC_Module,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'key_channels': 4, 'value_channels': 4, 'dropout': 0.5}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (BasicConv2d,
     lambda: ([], {'in_planes': 4, 'out_planes': 4, 'kernel_size': 4, 'stride': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (BiFPNConvBlock,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (BiTemperedLogisticLoss,
     lambda: ([], {'t1': 4, 't2': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (BilinearAdditiveUpsample2d,
     lambda: ([], {'in_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (BinaryLovaszLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (BinarySoftF1Loss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (CFM,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ChannelGate2d,
     lambda: ([], {'channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ChannelSpatialGate2d,
     lambda: ([], {'channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ChannelSpatialGate2dV2,
     lambda: ([], {'channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (CoordConv,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (DeconvolutionUpsample2d,
     lambda: ([], {'in_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (DepthwiseSeparableConv2d,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (DropBlock2D,
     lambda: ([], {'drop_prob': 4, 'block_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (DropBlock3D,
     lambda: ([], {'drop_prob': 4, 'block_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4, 4])], {})),
    (DropBlockScheduled,
     lambda: ([], {'dropblock': torch.nn.ReLU(), 'start_value': 4, 'stop_value': 4, 'nr_steps': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (DropPath,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Ensembler,
     lambda: ([], {'models': [torch.nn.ReLU()]}),
     lambda: ([], {'input': torch.rand([4, 4])})),
    (FPNBottleneckBlock,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (GeneralizedMeanPooling2d,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (GlobalAvgPool2d,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (GlobalKMaxPool2d,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (GlobalMaxAvgPooling2d,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (GlobalMaxPool2d,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (GlobalWeightedAvgPool2d,
     lambda: ([], {'features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (HGBlock,
     lambda: ([], {'depth': 1, 'input_features': 4, 'features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (HGFeaturesBlock,
     lambda: ([], {'features': 4, 'activation': torch.nn.ReLU}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (HGResidualBlock,
     lambda: ([], {'input_channels': 4, 'output_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (HGStemBlock,
     lambda: ([], {'input_channels': 4, 'output_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (HGSupervisionBlock,
     lambda: ([], {'features': 4, 'supervision_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (HRNetBasicBlock,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (HardSigmoid,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (HardSwish,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Identity,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (IdentityResidualBlock,
     lambda: ([], {'in_channels': 4, 'channels': [4, 4]}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (InceptionV4,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 128, 128])], {})),
    (Inception_A,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 384, 64, 64])], {})),
    (Inception_B,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1024, 64, 64])], {})),
    (Inception_C,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1536, 64, 64])], {})),
    (InvertedResidual,
     lambda: ([], {'inp': 4, 'oup': 4, 'stride': 1, 'expand_ratio': 4, 'activation': torch.nn.ReLU}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (LogCoshLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (LovaszLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (MILCustomPoolingModule,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Mish,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (MishNaive,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Mixed_3a,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 64, 64, 64])], {})),
    (Mixed_4a,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 160, 64, 64])], {})),
    (Mixed_5a,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 192, 64, 64])], {})),
    (Mlp,
     lambda: ([], {'in_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (MobileNetV2,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {})),
    (NoOp,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ObjectContextBlock,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'key_channels': 4, 'value_channels': 4, 'dropout': 0.5}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (OverlapPatchEmbed,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {})),
    (PickModelOutput,
     lambda: ([], {'model': torch.nn.ReLU(), 'key': 4}),
     lambda: ([], {'input': torch.rand([5, 4])})),
    (PyramidObjectContextBlock,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (PyramidSelfAttentionBlock2D,
     lambda: ([], {'in_channels': 4, 'key_channels': 4, 'value_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (RCM,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (RMSPool,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Reduction_A,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 384, 64, 64])], {})),
    (Reduction_B,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1024, 64, 64])], {})),
    (ResidualDeconvolutionUpsample2d,
     lambda: ([], {'in_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (SEModule,
     lambda: ([], {'channels': 4, 'reduction': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (SEXResNetBlock,
     lambda: ([], {'expansion': 4, 'n_inputs': 4, 'n_hidden': 4}),
     lambda: ([torch.rand([4, 16, 64, 64])], {})),
    (SRMLayer,
     lambda: ([], {'channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (SelectByIndex,
     lambda: ([], {'key': 4}),
     lambda: ([torch.rand([5, 4, 4, 4])], {})),
    (SelfAttentionBlock2D,
     lambda: ([], {'in_channels': 4, 'key_channels': 4, 'value_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (SeparableASPPModule,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4, 'padding': 4, 'dilation': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (SoftBCEWithLogitsLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (SoftCrossEntropyLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4]), torch.ones([4, 4], dtype=torch.int64)], {})),
    (SpatialGate2dV2,
     lambda: ([], {'channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (StemBlock,
     lambda: ([], {'input_channels': 4, 'output_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (SumAll,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Swish,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (SwishNaive,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (UnetBlock,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (UnetResidualBlock,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (XResNetBlock,
     lambda: ([], {'expansion': 4, 'n_inputs': 4, 'n_hidden': 4}),
     lambda: ([torch.rand([4, 16, 64, 64])], {})),
    (_PyramidSelfAttentionBlock,
     lambda: ([], {'in_channels': 4, 'key_channels': 4, 'value_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (_SelfAttentionBlock,
     lambda: ([], {'in_channels': 4, 'key_channels': 4, 'value_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
]

