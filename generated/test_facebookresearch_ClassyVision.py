
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


import logging


import torch


from torchvision import set_image_backend


from torchvision import set_video_backend


from typing import Any


from typing import Callable


from typing import Dict


from typing import Optional


from typing import Sequence


from typing import Union


from torch.utils.data import DataLoader


from torch.utils.data.distributed import DistributedSampler


from torchvision.datasets.hmdb51 import HMDB51


from typing import List


from torchvision.datasets.kinetics import Kinetics


from torchvision.datasets.ucf101 import UCF101


from torch.utils.data import Sampler


from torchvision import get_video_backend


from torchvision.datasets.samplers.clip_sampler import DistributedSampler


from torchvision.datasets.samplers.clip_sampler import RandomClipSampler


from torchvision.datasets.samplers.clip_sampler import UniformClipSampler


from typing import Iterable


from typing import Iterator


import collections.abc as abc


import math


from typing import Tuple


import numpy as np


from torch.distributions.beta import Beta


import collections


import torchvision.transforms as transforms


import random


import torchvision.transforms._transforms_video as transforms_video


from collections import defaultdict


from collections import deque


from time import perf_counter


from typing import Mapping


from torch.cuda import Event as CudaEvent


import torch.nn as nn


from torch.cuda import cudart


import collections.abc


import time


from functools import partial


import torch.nn.modules as nn


from collections.abc import Sequence


import copy


from collections import OrderedDict


from typing import Collection


import itertools


from itertools import accumulate


import torch.nn.modules.loss as torch_losses


import torch.nn.functional as F


from torch import Tensor


from enum import auto


from enum import Enum


import types


from typing import NamedTuple


import warnings


import re


import torch.optim


from abc import ABC


from abc import abstractmethod


from torch.optim import Optimizer


import torch.distributed as dist


import enum


from torch.distributed import broadcast


import torchvision.models as models


import functools


from torch.utils.data import Dataset


from torchvision import transforms


import numpy


import queue


from functools import wraps


from itertools import product


from torch.multiprocessing import Event


from torch.multiprocessing import Process


from torch.multiprocessing import Queue


from torchvision import models


from torch.utils.tensorboard import SummaryWriter


import torchvision.models


from torch.nn.modules.loss import CrossEntropyLoss


class ClassyHead(nn.Module):
    """
    Base class for heads that can be attached to :class:`ClassyModel`.

    A head is a regular :class:`torch.nn.Module` that can be attached to a
    pretrained model. This enables a form of transfer learning: utilizing a
    model trained for one dataset to extract features that can be used for
    other problems. A head must be attached to a :class:`models.ClassyBlock`
    within a :class:`models.ClassyModel`.
    """

    def __init__(self, unique_id: 'Optional[str]'=None, num_classes: 'Optional[int]'=None):
        """
        Constructs a ClassyHead.

        Args:
            unique_id: A unique identifier for the head. Multiple instances of
                the same head might be attached to a model, and unique_id is used
                to refer to them.

            num_classes: Number of classes for the head.
        """
        super().__init__()
        self.unique_id = unique_id or self.__class__.__name__
        self.num_classes = num_classes

    @classmethod
    def from_config(cls, config: 'Dict[str, Any]') ->'ClassyHead':
        """Instantiates a ClassyHead from a configuration.

        Args:
            config: A configuration for the ClassyHead.

        Returns:
            A ClassyHead instance.
        """
        raise NotImplementedError

    def forward(self, x):
        """
        Performs inference on the head.

        This is a regular PyTorch method, refer to :class:`torch.nn.Module` for
        more details
        """
        raise NotImplementedError


NORMALIZE_L2 = 'l2'


RELU_IN_PLACE = True


def get_torch_version():
    """Get the torch version as [major, minor].

    All comparisons must be done with the two version values. Revisions are not
    supported.
    """
    version_list = torch.__version__.split('.')[:2]
    return [int(version_str) for version_str in version_list]


def is_pos_int(number: 'int') ->bool:
    """
    Returns True if a number is a positive integer.
    """
    return type(number) == int and number >= 0


class FullyConnectedHead(ClassyHead):
    """This head defines a 2d average pooling layer
    (:class:`torch.nn.AdaptiveAvgPool2d`) followed by a fully connected
    layer (:class:`torch.nn.Linear`).
    """

    def __init__(self, unique_id: 'str', num_classes: 'Optional[int]', in_plane: 'int', conv_planes: 'Optional[int]'=None, activation: 'Optional[nn.Module]'=None, zero_init_bias: 'bool'=False, normalize_inputs: 'Optional[str]'=None):
        """Constructor for FullyConnectedHead

        Args:
            unique_id: A unique identifier for the head. Multiple instances of
                the same head might be attached to a model, and unique_id is used
                to refer to them.
            num_classes: Number of classes for the head. If None, then the fully
                connected layer is not applied.
            in_plane: Input size for the fully connected layer.
            conv_planes: If specified, applies a 1x1 convolutional layer to the input
                before passing it to the average pooling layer. The convolution is also
                followed by a BatchNorm and an activation.
            activation: The activation to be applied after the convolutional layer.
                Unused if `conv_planes` is not specified.
            zero_init_bias: Zero initialize the bias
            normalize_inputs: If specified, normalize the inputs after performing
                average pooling using the specified method. Supports "l2" normalization.
        """
        super().__init__(unique_id, num_classes)
        assert num_classes is None or is_pos_int(num_classes)
        assert is_pos_int(in_plane)
        if conv_planes is not None and activation is None:
            raise TypeError('activation cannot be None if conv_planes is specified')
        if normalize_inputs is not None and normalize_inputs != NORMALIZE_L2:
            raise ValueError(f'Unsupported value for normalize_inputs: {normalize_inputs}')
        self.conv = nn.Conv2d(in_plane, conv_planes, kernel_size=1, bias=False) if conv_planes else None
        self.bn = nn.BatchNorm2d(conv_planes) if conv_planes else None
        self.activation = activation
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = None if num_classes is None else nn.Linear(in_plane if conv_planes is None else conv_planes, num_classes)
        self.normalize_inputs = normalize_inputs
        if zero_init_bias:
            self.fc.bias.data.zero_()

    @classmethod
    def from_config(cls, config: 'Dict[str, Any]') ->'FullyConnectedHead':
        """Instantiates a FullyConnectedHead from a configuration.

        Args:
            config: A configuration for a FullyConnectedHead.
                See :func:`__init__` for parameters expected in the config.

        Returns:
            A FullyConnectedHead instance.
        """
        num_classes = config.get('num_classes', None)
        in_plane = config['in_plane']
        silu = None if get_torch_version() < [1, 7] else nn.SiLU()
        activation = {'relu': nn.ReLU(RELU_IN_PLACE), 'silu': silu}[config.get('activation', 'relu')]
        if activation is None:
            raise RuntimeError('SiLU activation is only supported since PyTorch 1.7')
        return cls(config['unique_id'], num_classes, in_plane, conv_planes=config.get('conv_planes', None), activation=activation, zero_init_bias=config.get('zero_init_bias', False), normalize_inputs=config.get('normalize_inputs', None))

    def forward(self, x):
        out = x
        if self.conv is not None:
            out = self.activation(self.bn(self.conv(x)))
        out = self.avgpool(out)
        out = out.flatten(start_dim=1)
        if self.normalize_inputs is not None:
            if self.normalize_inputs == NORMALIZE_L2:
                out = nn.functional.normalize(out, p=2.0, dim=1)
        if self.fc is not None:
            out = self.fc(out)
        return out


class FullyConvolutionalLinear(nn.Module):

    def __init__(self, dim_in, num_classes, act_func='softmax'):
        super(FullyConvolutionalLinear, self).__init__()
        self.projection = nn.Linear(dim_in, num_classes, bias=True)
        if act_func == 'softmax':
            self.act = nn.Softmax(dim=4)
        elif act_func == 'sigmoid':
            self.act = nn.Sigmoid()
        elif act_func == 'identity':
            self.act = nn.Identity()
        else:
            raise NotImplementedError('{} is not supported as an activationfunction.'.format(act_func))

    def forward(self, x):
        x = x.permute((0, 2, 3, 4, 1))
        x = self.projection(x)
        if not self.training:
            x = self.act(x)
            x = x.mean([1, 2, 3])
        x = x.flatten(start_dim=1)
        return x


class FullyConvolutionalLinearHead(ClassyHead):
    """
    This head defines a 3d average pooling layer (:class:`torch.nn.AvgPool3d` or
    :class:`torch.nn.AdaptiveAvgPool3d` if pool_size is None) followed by a fully
    convolutional linear layer. This layer performs a fully-connected projection
    during training, when the input size is 1x1x1.
    It performs a convolutional projection during testing when the input size
    is larger than 1x1x1.
    """

    def __init__(self, unique_id: 'str', num_classes: 'int', in_plane: 'int', pool_size: 'Optional[List[int]]', activation_func: 'str', use_dropout: 'Optional[bool]'=None, dropout_ratio: 'float'=0.5):
        """
        Constructor for FullyConvolutionalLinearHead.

        Args:
            unique_id: A unique identifier for the head. Multiple instances of
                the same head might be attached to a model, and unique_id is used
                to refer to them.
            num_classes: Number of classes for the head.
            in_plane: Input size for the fully connected layer.
            pool_size: Optional kernel size for the 3d pooling layer. If None, use
                :class:`torch.nn.AdaptiveAvgPool3d` with output size (1, 1, 1).
            activation_func: activation function to use. 'softmax': applies
                softmax on the output. 'sigmoid': applies sigmoid on the output.
            use_dropout: Whether to apply dropout after the pooling layer.
            dropout_ratio: dropout ratio.
        """
        super().__init__(unique_id, num_classes)
        if pool_size is not None:
            self.final_avgpool = nn.AvgPool3d(pool_size, stride=1)
        else:
            self.final_avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        if use_dropout:
            self.dropout = nn.Dropout(p=dropout_ratio)
        self.head_fcl = FullyConvolutionalLinear(in_plane, num_classes, act_func=activation_func)

    @classmethod
    def from_config(cls, config: 'Dict[str, Any]') ->'FullyConvolutionalLinearHead':
        """Instantiates a FullyConvolutionalLinearHead from a configuration.

        Args:
            config: A configuration for a FullyConvolutionalLinearHead.
                See :func:`__init__` for parameters expected in the config.

        Returns:
            A FullyConvolutionalLinearHead instance.
        """
        required_args = ['in_plane', 'num_classes']
        for arg in required_args:
            assert arg in config, 'argument %s is required' % arg
        config.update({'activation_func': config.get('activation_func', 'softmax')})
        config.update({'use_dropout': config.get('use_dropout', False)})
        pool_size = config.get('pool_size', None)
        if pool_size is not None:
            assert isinstance(pool_size, Sequence) and len(pool_size) == 3
            for pool_size_dim in pool_size:
                assert is_pos_int(pool_size_dim)
        assert is_pos_int(config['in_plane'])
        assert is_pos_int(config['num_classes'])
        num_classes = config.get('num_classes', None)
        in_plane = config['in_plane']
        return cls(config['unique_id'], num_classes, in_plane, pool_size, config['activation_func'], config['use_dropout'], config.get('dropout_ratio', 0.5))

    def forward(self, x):
        out = self.final_avgpool(x)
        if hasattr(self, 'dropout'):
            out = self.dropout(out)
        out = self.head_fcl(out)
        return out


def lecun_normal_init(tensor, fan_in):
    if get_torch_version() >= [1, 7]:
        trunc_normal_ = nn.init.trunc_normal_
    else:

        def trunc_normal_(tensor: 'Tensor', mean: 'float'=0.0, std: 'float'=1.0, a: 'float'=-2.0, b: 'float'=2.0) ->Tensor:

            def norm_cdf(x):
                return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0
            if mean < a - 2 * std or mean > b + 2 * std:
                warnings.warn('mean is more than 2 std from [a, b] in nn.init.trunc_normal_. The distribution of values may be incorrect.', stacklevel=2)
            with torch.no_grad():
                l = norm_cdf((a - mean) / std)
                u = norm_cdf((b - mean) / std)
                tensor.uniform_(2 * l - 1, 2 * u - 1)
                tensor.erfinv_()
                tensor.mul_(std * math.sqrt(2.0))
                tensor.add_(mean)
                tensor.clamp_(min=a, max=b)
                return tensor
    trunc_normal_(tensor, std=math.sqrt(1 / fan_in))


class VisionTransformerHead(ClassyHead):

    def __init__(self, unique_id: 'str', in_plane: 'int', num_classes: 'Optional[int]'=None, hidden_dim: 'Optional[int]'=None, normalize_inputs: 'Optional[str]'=None):
        """
        Args:
            unique_id: A unique identifier for the head
            in_plane: Input size for the fully connected layer
            num_classes: Number of output classes for the head
            hidden_dim: If not None, a hidden layer with the specific dimension is added
            normalize_inputs: If specified, normalize the inputs using the specified
                method. Supports "l2" normalization.
        """
        super().__init__(unique_id, num_classes)
        if normalize_inputs is not None and normalize_inputs != NORMALIZE_L2:
            raise ValueError(f'Unsupported value for normalize_inputs: {normalize_inputs}')
        if num_classes is None:
            layers = []
        elif hidden_dim is None:
            layers = [('head', nn.Linear(in_plane, num_classes))]
        else:
            layers = [('pre_logits', nn.Linear(in_plane, hidden_dim)), ('act', nn.Tanh()), ('head', nn.Linear(hidden_dim, num_classes))]
        self.layers = nn.Sequential(OrderedDict(layers))
        self.normalize_inputs = normalize_inputs
        self.init_weights()

    def init_weights(self):
        if hasattr(self.layers, 'pre_logits'):
            lecun_normal_init(self.layers.pre_logits.weight, fan_in=self.layers.pre_logits.in_features)
            nn.init.zeros_(self.layers.pre_logits.bias)
        if hasattr(self.layers, 'head'):
            nn.init.zeros_(self.layers.head.weight)
            nn.init.zeros_(self.layers.head.bias)

    @classmethod
    def from_config(cls, config):
        config = copy.deepcopy(config)
        return cls(**config)

    def forward(self, x):
        if self.normalize_inputs is not None:
            if self.normalize_inputs == NORMALIZE_L2:
                x = nn.functional.normalize(x, p=2.0, dim=1)
        return self.layers(x)


class ClassyLoss(nn.Module):
    """
    Base class to calculate the loss during training.

    This implementation of :class:`torch.nn.Module` allows building
    the loss object from a configuration file.
    """

    def __init__(self):
        """
        Constructor for ClassyLoss.
        """
        super(ClassyLoss, self).__init__()

    @classmethod
    def from_config(cls, config: 'Dict[str, Any]') ->'ClassyLoss':
        """Instantiates a ClassyLoss from a configuration.

        Args:
            config: A configuration for a ClassyLoss.

        Returns:
            A ClassyLoss instance.
        """
        raise NotImplementedError()

    def forward(self, output, target):
        """
        Compute the loss for the provided sample.

        Refer to :class:`torch.nn.Module` for more details.
        """
        raise NotImplementedError

    def get_classy_state(self) ->Dict[str, Any]:
        """Get the state of the ClassyLoss.

        The returned state is used for checkpointing. Note that most losses are
        stateless and do not need to save any state.

        Returns:
            A state dictionary containing the state of the loss.
        """
        return self.state_dict()

    def set_classy_state(self, state: 'Dict[str, Any]') ->None:
        """Set the state of the ClassyLoss.

        Args:
            state_dict: The state dictionary. Must be the output of a call to
                :func:`get_classy_state`.

        This is used to load the state of the loss from a checkpoint. Note
        that most losses are stateless and do not need to load any state.
        """
        return self.load_state_dict(state)

    def has_learned_parameters(self) ->bool:
        """Does this loss have learned parameters?"""
        return any(param.requires_grad for param in self.parameters(recurse=True))


LOSS_REGISTRY = {}


def log_class_usage(component_type, klass):
    """This function is used to log the usage of different Classy components."""
    identifier = 'ClassyVision'
    if klass and hasattr(klass, '__name__'):
        identifier += f'.{component_type}.{klass.__name__}'
    torch._C._log_api_usage_once(identifier)


def build_loss(config):
    """Builds a ClassyLoss from a config.

    This assumes a 'name' key in the config which is used to determine what
    model class to instantiate. For instance, a config `{"name": "my_loss",
    "foo": "bar"}` will find a class that was registered as "my_loss"
    (see :func:`register_loss`) and call .from_config on it.

    In addition to losses registered with :func:`register_loss`, we also
    support instantiating losses available in the `torch.nn.modules.loss <https:
    //pytorch.org/docs/stable/nn.html#loss-functions>`_
    module. Any keys in the config will get expanded to parameters of the loss
    constructor. For instance, the following call will instantiate a
    `torch.nn.modules.CrossEntropyLoss <https://pytorch.org/docs/stable/
    nn.html#torch.nn.CrossEntropyLoss>`_:

    .. code-block:: python

     build_loss({"name": "CrossEntropyLoss", "reduction": "sum"})
    """
    assert 'name' in config, f'name not provided for loss: {config}'
    name = config['name']
    args = copy.deepcopy(config)
    del args['name']
    if 'weight' in args and args['weight'] is not None:
        args['weight'] = torch.tensor(args['weight'], dtype=torch.float)
    if name in LOSS_REGISTRY:
        loss = LOSS_REGISTRY[name].from_config(config)
    else:
        assert hasattr(torch_losses, name), f"{name} isn't a registered loss, nor is it available in torch.nn.modules.loss"
        loss = getattr(torch_losses, name)(**args)
    log_class_usage('Loss', loss.__class__)
    return loss


class MultiOutputSumLoss(ClassyLoss):
    """
    Applies the provided loss to the list of outputs (or single output) and sums
    up the losses.
    """

    def __init__(self, loss) ->None:
        super().__init__()
        self._loss = loss

    @classmethod
    def from_config(cls, config: 'Dict[str, Any]') ->'MultiOutputSumLoss':
        """Instantiates a MultiOutputSumLoss from a configuration.

        Args:
            config: A configuration for a MultiOutpuSumLoss.
                See :func:`__init__` for parameters expected in the config.

        Returns:
            A MultiOutputSumLoss instance.
        """
        assert type(config['loss']) == dict, 'loss must be a dict containing a configuration for a registered loss'
        return cls(loss=build_loss(config['loss']))

    def forward(self, output, target):
        if torch.is_tensor(output):
            output = [output]
        loss = 0
        for pred in output:
            loss += self._loss(pred, target)
        return loss


def convert_to_one_hot(targets: 'torch.Tensor', classes) ->torch.Tensor:
    """
    This function converts target class indices to one-hot vectors,
    given the number of classes.

    """
    assert torch.max(targets).item() < classes, 'Class Index must be less than number of classes'
    one_hot_targets = torch.zeros((targets.shape[0], classes), dtype=torch.long, device=targets.device)
    one_hot_targets.scatter_(1, targets.long(), 1)
    return one_hot_targets


class SoftTargetCrossEntropyLoss(ClassyLoss):

    def __init__(self, ignore_index=-100, reduction='mean', normalize_targets=True):
        """Intializer for the soft target cross-entropy loss loss.
        This allows the targets for the cross entropy loss to be multilabel

        Args:
            ignore_index: sample should be ignored for loss if the class is this value
            reduction: specifies reduction to apply to the output
            normalize_targets: whether the targets should be normalized to a sum of 1
                based on the total count of positive targets for a given sample
        """
        super(SoftTargetCrossEntropyLoss, self).__init__()
        self._ignore_index = ignore_index
        self._reduction = reduction
        assert isinstance(normalize_targets, bool)
        self._normalize_targets = normalize_targets
        if self._reduction not in ['none', 'mean']:
            raise NotImplementedError('reduction type "{}" not implemented'.format(self._reduction))
        self._eps = torch.finfo(torch.float32).eps

    @classmethod
    def from_config(cls, config: 'Dict[str, Any]') ->'SoftTargetCrossEntropyLoss':
        """Instantiates a SoftTargetCrossEntropyLoss from a configuration.

        Args:
            config: A configuration for a SoftTargetCrossEntropyLoss.
                See :func:`__init__` for parameters expected in the config.

        Returns:
            A SoftTargetCrossEntropyLoss instance.
        """
        return cls(ignore_index=config.get('ignore_index', -100), reduction=config.get('reduction', 'mean'), normalize_targets=config.get('normalize_targets', True))

    def forward(self, output, target):
        """for N examples and C classes
        - output: N x C these are raw outputs (without softmax/sigmoid)
        - target: N x C or N corresponding targets

        Target elements set to ignore_index contribute 0 loss.

        Samples where all entries are ignore_index do not contribute to the loss
        reduction.
        """
        if target.ndim == 1:
            assert output.shape[0] == target.shape[0], 'SoftTargetCrossEntropyLoss requires output and target to have same batch size'
            target = convert_to_one_hot(target.view(-1, 1), output.shape[1])
        assert output.shape == target.shape, f'SoftTargetCrossEntropyLoss requires output and target to be same shape: {output.shape} != {target.shape}'
        valid_mask = target != self._ignore_index
        valid_targets = target.float() * valid_mask.float()
        if self._normalize_targets:
            valid_targets /= self._eps + valid_targets.sum(dim=1, keepdim=True)
        per_sample_per_target_loss = -valid_targets * F.log_softmax(output, -1)
        per_sample_loss = torch.sum(per_sample_per_target_loss, -1)
        if self._reduction == 'mean':
            loss = per_sample_loss.sum() / torch.sum(torch.sum(valid_mask, -1) > 0).clamp(min=1)
        elif self._reduction == 'none':
            loss = per_sample_loss
        return loss


class SumArbitraryLoss(ClassyLoss):
    """
    Sums a collection of (weighted) torch.nn losses.

    NOTE: this applies all the losses to the same output and does not support
    taking a list of outputs as input.
    """

    def __init__(self, losses: 'List[ClassyLoss]', weights: 'Optional[Tensor]'=None) ->None:
        super().__init__()
        if weights is None:
            weights = torch.ones(len(losses))
        self.losses = losses
        self.weights = weights

    @classmethod
    def from_config(cls, config: 'Dict[str, Any]') ->'SumArbitraryLoss':
        """Instantiates a SumArbitraryLoss from a configuration.

        Args:
            config: A configuration for a SumArbitraryLoss.
                See :func:`__init__` for parameters expected in the config.

        Returns:
            A SumArbitraryLoss instance.
        """
        assert type(config['losses']) == list and len(config['losses']) > 0, 'losses must be a list of registered losses with length > 0'
        assert type(config['weights']) == list and len(config['weights']) == len(config['losses']), 'weights must be None or a list and have same length as losses'
        loss_modules = []
        for loss_config in config['losses']:
            loss_modules.append(build_loss(loss_config))
        assert all(isinstance(loss_module, ClassyLoss) for loss_module in loss_modules), 'All losses must be registered, valid ClassyLosses'
        return cls(losses=loss_modules, weights=config.get('weights', None))

    def forward(self, prediction, target):
        for idx, loss in enumerate(self.losses):
            current_loss = loss(prediction, target)
            if idx == 0:
                total_loss = current_loss
            else:
                total_loss = total_loss.add(self.weights[idx], current_loss)
        return total_loss


class BasicTransform(nn.Sequential):
    """Basic transformation: [3x3 conv, BN, Relu] x2."""

    def __init__(self, width_in: 'int', width_out: 'int', stride: 'int', bn_epsilon: 'float', bn_momentum: 'float', activation: 'nn.Module'):
        super().__init__()
        self.a = nn.Sequential(nn.Conv2d(width_in, width_out, 3, stride=stride, padding=1, bias=False), nn.BatchNorm2d(width_out, eps=bn_epsilon, momentum=bn_momentum), activation, nn.Conv2d(width_out, width_out, 3, stride=1, padding=1, bias=False))
        self.final_bn = nn.BatchNorm2d(width_out, eps=bn_epsilon, momentum=bn_momentum)
        self.depth = 2


class ResStemCifar(nn.Sequential):
    """ResNet stem for CIFAR: 3x3, BN, ReLU."""

    def __init__(self, width_in: 'int', width_out: 'int', bn_epsilon: 'float', bn_momentum: 'float', activation: 'nn.Module'):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(width_in, width_out, 3, stride=1, padding=1, bias=False), nn.BatchNorm2d(width_out, eps=bn_epsilon, momentum=bn_momentum), activation)
        self.depth = 2


class ResStemIN(nn.Sequential):
    """ResNet stem for ImageNet: 7x7, BN, ReLU, MaxPool."""

    def __init__(self, width_in: 'int', width_out: 'int', bn_epsilon: 'float', bn_momentum: 'float', activation: 'nn.Module'):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(width_in, width_out, 7, stride=2, padding=3, bias=False), nn.BatchNorm2d(width_out, eps=bn_epsilon, momentum=bn_momentum), activation, nn.MaxPool2d(3, stride=2, padding=1))
        self.depth = 3


class SimpleStemIN(nn.Sequential):
    """Simple stem for ImageNet: 3x3, BN, ReLU."""

    def __init__(self, width_in: 'int', width_out: 'int', bn_epsilon: 'float', bn_momentum: 'float', activation: 'nn.Module'):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(width_in, width_out, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(width_out, eps=bn_epsilon, momentum=bn_momentum), activation)
        self.depth = 2


class VanillaBlock(nn.Sequential):
    """Vanilla block: [3x3 conv, BN, Relu] x2."""

    def __init__(self, width_in: 'int', width_out: 'int', stride: 'int', bn_epsilon: 'float', bn_momentum: 'float', activation: 'nn.Module', *args, **kwargs):
        super().__init__()
        self.a = nn.Sequential(nn.Conv2d(width_in, width_out, 3, stride=stride, padding=1, bias=False), nn.BatchNorm2d(width_out, eps=bn_epsilon, momentum=bn_momentum), activation)
        self.b = nn.Sequential(nn.Conv2d(width_out, width_out, 3, stride=1, padding=1, bias=False), nn.BatchNorm2d(width_out, eps=bn_epsilon, momentum=bn_momentum), activation)
        self.depth = 2


class ResBasicBlock(nn.Module):
    """Residual basic block: x + F(x), F = basic transform."""

    def __init__(self, width_in: 'int', width_out: 'int', stride: 'int', bn_epsilon: 'float', bn_momentum: 'float', activation: 'nn.Module', *args, **kwargs):
        super().__init__()
        self.proj_block = width_in != width_out or stride != 1
        if self.proj_block:
            self.proj = nn.Conv2d(width_in, width_out, 1, stride=stride, padding=0, bias=False)
            self.bn = nn.BatchNorm2d(width_out, eps=bn_epsilon, momentum=bn_momentum)
        self.f = BasicTransform(width_in, width_out, stride, bn_epsilon, bn_momentum, activation)
        self.activation = activation
        self.depth = self.f.depth

    def forward(self, x):
        if self.proj_block:
            x = self.bn(self.proj(x)) + self.f(x)
        else:
            x = x + self.f(x)
        return self.activation(x)


class SqueezeAndExcitationLayer(nn.Module):
    """Squeeze and excitation layer, as per https://arxiv.org/pdf/1709.01507.pdf"""

    def __init__(self, in_planes, reduction_ratio: 'Optional[int]'=16, reduced_planes: 'Optional[int]'=None, activation: 'Optional[nn.Module]'=None):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        assert bool(reduction_ratio) != bool(reduced_planes)
        if activation is None:
            activation = nn.ReLU()
        reduced_planes = in_planes // reduction_ratio if reduced_planes is None else reduced_planes
        self.excitation = nn.Sequential(nn.Conv2d(in_planes, reduced_planes, kernel_size=1, stride=1, bias=True), activation, nn.Conv2d(reduced_planes, in_planes, kernel_size=1, stride=1, bias=True), nn.Sigmoid())

    def forward(self, x):
        x_squeezed = self.avgpool(x)
        x_excited = self.excitation(x_squeezed)
        x_scaled = x * x_excited
        return x_scaled


class BottleneckTransform(nn.Sequential):
    """Bottleneck transformation: 1x1, 3x3 [+SE], 1x1."""

    def __init__(self, width_in: 'int', width_out: 'int', stride: 'int', bn_epsilon: 'float', bn_momentum: 'float', activation: 'nn.Module', group_width: 'int', bottleneck_multiplier: 'float', se_ratio: 'Optional[float]'):
        super().__init__()
        w_b = int(round(width_out * bottleneck_multiplier))
        g = w_b // group_width
        self.a = nn.Sequential(nn.Conv2d(width_in, w_b, 1, stride=1, padding=0, bias=False), nn.BatchNorm2d(w_b, eps=bn_epsilon, momentum=bn_momentum), activation)
        self.b = nn.Sequential(nn.Conv2d(w_b, w_b, 3, stride=stride, padding=1, groups=g, bias=False), nn.BatchNorm2d(w_b, eps=bn_epsilon, momentum=bn_momentum), activation)
        if se_ratio:
            width_se_out = int(round(se_ratio * width_in))
            self.se = SqueezeAndExcitationLayer(in_planes=w_b, reduction_ratio=None, reduced_planes=width_se_out, activation=activation)
        self.c = nn.Conv2d(w_b, width_out, 1, stride=1, padding=0, bias=False)
        self.final_bn = nn.BatchNorm2d(width_out, eps=bn_epsilon, momentum=bn_momentum)
        self.depth = 3 if not se_ratio else 4


class ResBottleneckBlock(nn.Module):
    """Residual bottleneck block: x + F(x), F = bottleneck transform."""

    def __init__(self, width_in: 'int', width_out: 'int', stride: 'int', bn_epsilon: 'float', bn_momentum: 'float', activation: 'nn.Module', group_width: 'int'=1, bottleneck_multiplier: 'float'=1.0, se_ratio: 'Optional[float]'=None):
        super().__init__()
        self.proj_block = width_in != width_out or stride != 1
        if self.proj_block:
            self.proj = nn.Conv2d(width_in, width_out, 1, stride=stride, padding=0, bias=False)
            self.bn = nn.BatchNorm2d(width_out, eps=bn_epsilon, momentum=bn_momentum)
        self.f = BottleneckTransform(width_in, width_out, stride, bn_epsilon, bn_momentum, activation, group_width, bottleneck_multiplier, se_ratio)
        self.activation = activation
        self.depth = self.f.depth

    def forward(self, x, *args):
        if self.proj_block:
            x = self.bn(self.proj(x)) + self.f(x)
        else:
            x = x + self.f(x)
        return self.activation(x)


class ResBottleneckLinearBlock(nn.Module):
    """Residual linear bottleneck block: x + F(x), F = bottleneck transform."""

    def __init__(self, width_in: 'int', width_out: 'int', stride: 'int', bn_epsilon: 'float', bn_momentum: 'float', activation: 'nn.Module', group_width: 'int'=1, bottleneck_multiplier: 'float'=4.0, se_ratio: 'Optional[float]'=None):
        super().__init__()
        self.has_skip = width_in == width_out and stride == 1
        self.f = BottleneckTransform(width_in, width_out, stride, bn_epsilon, bn_momentum, activation, group_width, bottleneck_multiplier, se_ratio)
        self.depth = self.f.depth

    def forward(self, x):
        return x + self.f(x) if self.has_skip else self.f(x)


class AnyStage(nn.Sequential):
    """AnyNet stage (sequence of blocks w/ the same output shape)."""

    def __init__(self, width_in: 'int', width_out: 'int', stride: 'int', depth: 'int', block_constructor: 'nn.Module', activation: 'nn.Module', group_width: 'int', bottleneck_multiplier: 'float', params: "'AnyNetParams'", stage_index: 'int'=0):
        super().__init__()
        self.stage_depth = 0
        for i in range(depth):
            block = block_constructor(width_in if i == 0 else width_out, width_out, stride if i == 0 else 1, params.bn_epsilon, params.bn_momentum, activation, group_width, bottleneck_multiplier, params.se_ratio)
            self.stage_depth += block.depth
            self.add_module(f'block{stage_index}-{i}', block)


class ClassyBlock(nn.Module):
    """
    This is a thin wrapper for head execution, which records the output of
    wrapped module for executing the heads forked from this module.
    """

    def __init__(self, name, module):
        super().__init__()
        self.name = name
        self.output = torch.zeros(0)
        self._module = module
        self._is_output_stateless = os.environ.get('CLASSY_BLOCK_STATELESS') == '1'

    def wrapped_module(self):
        return self._module

    def forward(self, input):
        if hasattr(self, '_is_output_stateless'):
            if self._is_output_stateless:
                return self._module(input)
        output = self._module(input)
        self.output = output
        return output


class _ClassyModelMethod:
    """Class to override ClassyModel method calls to ensure the wrapper is returned.

    This helps override calls like model.cuda() which return self, to return the
    wrapper instead of the underlying classy_model.
    """

    def __init__(self, wrapper, classy_method):
        self.wrapper = wrapper
        self.classy_method = classy_method

    def __call__(self, *args, **kwargs):
        ret_val = self.classy_method(*args, **kwargs)
        if ret_val is self.wrapper.classy_model:
            ret_val = self.wrapper
        return ret_val


class ClassyModelWrapper:
    """Base ClassyModel wrapper class.

    This class acts as a thin pass through wrapper which lets users modify the behavior
    of ClassyModels, such as changing the return output of the forward() call.
    This wrapper acts as a ClassyModel by itself and the underlying model can be
    accessed by the `classy_model` attribute.
    """

    def __init__(self, classy_model):
        self.classy_model = classy_model

    def __getattr__(self, name):
        if name != 'classy_model' and hasattr(self, 'classy_model'):
            attr = getattr(self.classy_model, name)
            if isinstance(attr, types.MethodType):
                attr = _ClassyModelMethod(self, attr)
            return attr
        else:
            return super().__getattr__(name)

    def __setattr__(self, name, value):
        if name not in ['classy_model', 'forward'] and hasattr(self, 'classy_model'):
            setattr(self.classy_model, name, value)
        else:
            super().__setattr__(name, value)

    def __delattr__(self, name):
        if name != 'classy_model' and hasattr(self, 'classy_model'):
            delattr(self.classy_model, name)
        else:
            return super().__delattr__(name)

    def forward(self, *args, **kwargs):
        return self.classy_model(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def __repr__(self):
        return f'Classy {type(self.classy_model)}:\n{self.classy_model.__repr__()}'

    @property
    def __class__(self):
        return self.classy_model.__class__


class ClassyModelHeadExecutorWrapper(ClassyModelWrapper):
    """Wrapper which changes the forward to also execute and return head output."""

    def forward(self, *args, **kwargs):
        out = self.classy_model(*args, **kwargs)
        if len(self._heads) == 0:
            return out
        head_outputs = self.execute_heads()
        if len(head_outputs) == 1:
            return list(head_outputs.values())[0]
        else:
            return head_outputs

