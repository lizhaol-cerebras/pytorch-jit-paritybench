
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


from typing import Optional


from typing import Sequence


from typing import Union


import torch


from torch.optim import Optimizer


import copy


import math


from types import MethodType


from torch import nn


import inspect


from typing import Callable


import functools


from typing import TypeVar


import numpy as np


import torch.utils.data


from torchvision.datasets import VisionDataset


import warnings


import torch.nn as nn


import torch.nn.functional as F


from torch.nn.common_types import _size_2_t


from torch.nn.modules.utils import _pair


from typing import Any


from torch import Tensor


import itertools


from typing import cast


import abc


from typing import Iterable


from torch.distributed.fsdp import FullyShardedDataParallel


from typing import TYPE_CHECKING


from functools import partial


import torchvision.transforms.functional


from torch.nn import functional as F


from typing import Mapping


from torchvision.models.resnet import Bottleneck


from torch.fx import GraphModule


from torch.optim.lr_scheduler import LRScheduler


from torch.optim.swa_utils import SWALR


from torch.optim.swa_utils import AveragedModel


import torch.nn.utils.parametrize as parametrize


from torch.fx import symbolic_trace


import time


from copy import deepcopy


import torch.cuda


from torch import distributed


from torch.utils.data import DataLoader


from torch.utils.data import IterableDataset


from collections import deque


import torch.distributed.checkpoint as DCP


from torch.distributed.fsdp import FullyShardedDataParallel as FSDP


from torch.distributed._shard.sharded_tensor import ShardedTensor


from torch.distributed._tensor import DTensor


from torch.utils.data import Dataset


from torch.nn.parallel import DistributedDataParallel


from abc import ABC


from abc import abstractmethod


import collections.abc


from torch.utils.data.distributed import DistributedSampler


from typing import Generator


from collections import OrderedDict


import torch.nn.modules.utils


from torch.distributed._tensor.device_mesh import DeviceMesh


from torch.distributed._tensor.device_mesh import init_device_mesh


from torch.distributed.fsdp.fully_sharded_data_parallel import FullOptimStateDictConfig


from torch.distributed.fsdp.fully_sharded_data_parallel import FullStateDictConfig


from torch.distributed.fsdp.fully_sharded_data_parallel import ShardedOptimStateDictConfig


from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType


from collections.abc import Mapping


from collections.abc import Sequence


import torch.nn


import torch.backends.cuda


import torch.backends.cudnn


import torch.cuda.amp


from typing import ContextManager


from typing import Iterator


from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointImpl


from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import apply_activation_checkpointing


from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper


from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import offload_wrapper


from torch.distributed.fsdp import ShardingStrategy


from torch.distributed.fsdp._common_utils import clean_tensor_name


from torch.distributed.fsdp.wrap import CustomPolicy


from torch.nn.parameter import Parameter


from torch.distributed import ProcessGroup


from torch.distributed.fsdp import BackwardPrefetch


from torch.distributed.fsdp import CPUOffload


from torch.distributed.fsdp import MixedPrecision


from torchvision.utils import draw_segmentation_masks


from functools import reduce


from typing import Literal


import queue


import uuid


import re


from torch.nn.modules.loss import _Loss


from torchvision.ops import box_convert


import random


import string


from torch import nn as nn


from torch.optim import SGD


from torch.optim import AdamW


from torch.optim.optimizer import required


from torch.optim.lr_scheduler import LambdaLR


from typing import OrderedDict


import torch.profiler


from torch.profiler.profiler import ProfilerAction as TorchProfilerAction


from torch.profiler._memory_profiler import _CATEGORY_TO_COLORS


from torch.profiler._memory_profiler import _CATEGORY_TO_INDEX


from torch.profiler._memory_profiler import MemoryProfileTimeline


from torch.profiler.profiler import profile as TorchProfile


from itertools import chain


from typing import no_type_check


from collections import Counter


from torch.optim.lr_scheduler import CosineAnnealingLR


from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts


from torch.optim.lr_scheduler import ExponentialLR


from torch.optim.lr_scheduler import MultiStepLR


from torch.optim.lr_scheduler import StepLR


from collections import defaultdict


from torch.cuda.amp.grad_scaler import GradScaler


from torch.cuda.amp.grad_scaler import OptState


from typing import TextIO


import torch.distributed


from torch._dynamo import OptimizedModule


from torch.distributed.fsdp._runtime_utils import _post_backward_final_callback


from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler


from torch.utils.data import DistributedSampler


from torch.distributed import checkpoint as dist_cp


from torch.distributed._tensor import DeviceMesh


from torch.distributed.checkpoint.metadata import Metadata


from torch.distributed.checkpoint.optimizer import load_sharded_optimizer_state_dict


from torch.distributed.checkpoint.planner import LoadPlan


from torch.distributed.checkpoint.planner import LoadPlanner


from torch.distributed.checkpoint.storage import StorageReader


from torch.distributed.distributed_c10d import ProcessGroup


import torch.distributed as dist


from torch.fx import Node


from torch.fx.passes.split_utils import split_by_tags


from torchvision import transforms


import collections


import types


from typing import Type


from typing import Callable as Callable


import torch.optim


from torchvision.datasets import MNIST


from torchvision.transforms import ToTensor


from torchvision import datasets


from torch.nn.functional import gelu


from torch.nn.functional import relu


from torch.nn import GroupNorm


from torch.nn import LayerNorm


import torch.fx


import scipy


from torch.distributed.fsdp.api import CPUOffload


from torch.optim import Adam


from torchvision.models import resnet


from torch.nn.functional import cross_entropy


from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointWrapper


from torch.distributed._tensor import Replicate


from collections import ChainMap


from typing import NamedTuple


from torch.fx.graph_module import GraphModule


from torchvision import models


def autocontrast(pil_img: 'Image.Image', level: 'float'=0.0):
    """Autocontrast an image.

    .. seealso:: :func:`PIL.ImageOps.autocontrast`.

    Args:
        pil_img (PIL.Image.Image): The image.
        level (float): The intensity.
    """
    del level
    return ImageOps.autocontrast(pil_img)


def _symmetric_sample(level: 'float'):
    """Helper function to sample from a symmetric distribution.

    The distribution over the domain [0.1, 10] with ``median == 1`` and uniform probability of ``x | 0.1 ≤ x ≤ 1``,
    and ``x | 1 ≤ x ≤ 10``.

    Used for sampling transforms that can range from intensity 0 to infinity and for which an intensity
    of 1 meaning no change.
    """
    if np.random.uniform() > 0.5:
        return np.random.uniform(1, level)
    else:
        return np.random.uniform(1 - 0.09 * level, 1)


def brightness(pil_img: 'Image.Image', level: 'float'):
    """Enhance brightness on an image.

    .. seealso:: :class:`PIL.ImageEnhance.Brightness`.

    Args:
        pil_img (PIL.Image.Image): The image.
        level (float): The intensity, which should be
            in ``[0, 10]``.
    """
    level = _symmetric_sample(level)
    if level > 1:
        level = level * 0.75
    return ImageEnhance.Brightness(pil_img).enhance(level)


def _float_parameter(level: 'float', maxval: 'float'):
    """Helper function to scale a value between ``0`` and ``maxval`` and return as a float.

    Args:
        level (float): Level of the operation that will be between [0, 10].
        maxval (float): Maximum value that the operation can have. This will be scaled to
            ``level/10``.

    Returns:
        float: The result from scaling ``maxval`` according to ``level``.
    """
    return float(level) * maxval / 10.0


def _sample_level(n: 'float'):
    """Helper function to sample from a uniform distribution between ``0.1`` and some value ``n``."""
    return np.random.uniform(low=0.1, high=n)


def brightness_original(pil_img: 'Image.Image', level: 'float'):
    """Enhance brightness on an image, following the
    corruptions in the ImageNet-C/CIFAR10-C test sets.

    .. seealso:: :class:`PIL.ImageEnhance.Brightness`.

    Args:
        pil_img (PIL.Image.Image): The image.
        level (float): The intensity, which should
            be in ``[0, 10]``.
    """
    level = _float_parameter(_sample_level(level), 1.8) + 0.1
    return ImageEnhance.Brightness(pil_img).enhance(level)


def color(pil_img: 'Image.Image', level: 'float'):
    """Enhance color on an image.

    .. seealso:: :class:`PIL.ImageEnhance.Color`.

    Args:
        pil_img (PIL.Image.Image): The image.
        level (float): The intensity, which should
            be in ``[0, 10]``.
    """
    level = _symmetric_sample(level)
    return ImageEnhance.Color(pil_img).enhance(level)


def color_original(pil_img: 'Image.Image', level: 'float'):
    """Enhance color on an image, following the
    corruptions in the ImageNet-C/CIFAR10-C test sets.

    .. seealso :class:`PIL.ImageEnhance.Color`.

    Args:
        pil_img (PIL.Image.Image): The image.
        level (float): The intensity, which should
            be in ``[0, 10]``.
    """
    level = _float_parameter(_sample_level(level), 1.8) + 0.1
    return ImageEnhance.Color(pil_img).enhance(level)


def contrast(pil_img: 'Image.Image', level: 'float'):
    """Enhance contrast on an image.

    .. seealso:: :class:`PIL.ImageEnhance.Contrast`.

    Args:
        pil_img (PIL.Image.Image): The image.
        level (float): The intensity, which should
            be in ``[0, 10]``.
    """
    level = _symmetric_sample(level)
    return ImageEnhance.Contrast(pil_img).enhance(level)


def contrast_original(pil_img: 'Image.Image', level: 'float'):
    """Enhance contrast on an image, following the
    corruptions in the ImageNet-C/CIFAR10-C test sets.

    .. seealso:: :class:`PIL.ImageEnhance.Contrast`.

    Args:
        pil_img (PIL.Image.Image): The image.
        level (float): The intensity, which should
            be in ``[0, 10]``.
    """
    level = _float_parameter(_sample_level(level), 1.8) + 0.1
    return ImageEnhance.Contrast(pil_img).enhance(level)


def equalize(pil_img: 'Image.Image', level: 'float'):
    """Equalize an image.

    .. seealso:: :func:`PIL.ImageOps.equalize`.

    Args:
        pil_img (PIL.Image.Image): The image.
        level (float): The intensity.
    """
    del level
    return ImageOps.equalize(pil_img)


def _int_parameter(level: 'float', maxval: 'float'):
    """Helper function to scale a value between ``0`` and ``maxval`` and return as an int.

    Args:
        level (float): Level of the operation that will be between ``[0, 10]``.
        maxval (float): Maximum value that the operation can have. This will be scaled to
            ``level/10``.

    Returns:
        int: The result from scaling ``maxval`` according to ``level``.
    """
    return int(level * maxval / 10)


def posterize(pil_img: 'Image.Image', level: 'float'):
    """Posterize an image.

    .. seealso:: :func:`PIL.ImageOps.posterize`.

    Args:
        pil_img (PIL.Image.Image): The image.
        level (float): The intensity, which should
            be in ``[0, 10]``.
    """
    level = _int_parameter(_sample_level(level), 4)
    return ImageOps.posterize(pil_img, 4 - level)


def rotate(pil_img: 'Image.Image', level: 'float'):
    """Rotate an image.

    Args:
        pil_img (PIL.Image.Image): The image.
        level (float): The intensity, which should
            be in ``[0, 10]``.
    """
    degrees = _int_parameter(_sample_level(level), 30)
    if np.random.uniform() > 0.5:
        degrees = -degrees
    return pil_img.rotate(degrees, resample=Resampling.BILINEAR)


def sharpness(pil_img: 'Image.Image', level: 'float'):
    """Enhance sharpness on an image.

    .. seealso:: :class:`PIL.ImageEnhance.Sharpness`.

    Args:
        pil_img (PIL.Image.Image): The image.
        level (float): The intensity, which should
            be in ``[0, 10]``.
    """
    level = _symmetric_sample(level)
    return ImageEnhance.Sharpness(pil_img).enhance(level)


def sharpness_original(pil_img: 'Image.Image', level: 'float'):
    """Enhance sharpness on an image, following the
    corruptions in the ImageNet-C/CIFAR10-C test sets.

    .. seealso:: :class:`PIL.ImageEnhance.Sharpness`.

    Args:
        pil_img (PIL.Image.Image): The image.
        level (float): The intensity, which should
            be in ``[0, 10]``.
    """
    level = _float_parameter(_sample_level(level), 1.8) + 0.1
    return ImageEnhance.Sharpness(pil_img).enhance(level)


Transform = Callable[[nn.Module], nn.Module]


def shear_x(pil_img: 'Image.Image', level: 'float'):
    """Shear an image horizontally.

    Args:
        pil_img (PIL.Image.Image): The image.
        level (float): The intensity, which should
            be in ``[0, 10]``.
    """
    level = _float_parameter(_sample_level(level), 0.3)
    if np.random.uniform() > 0.5:
        level = -level
    return pil_img.transform(pil_img.size, Transform.AFFINE, (1, level, 0, 0, 1, 0), resample=Resampling.BILINEAR)


def shear_y(pil_img: 'Image.Image', level: 'float'):
    """Shear an image vertically.

    Args:
        pil_img (PIL.Image.Image): The image.
        level (float): The intensity, which should
            be in ``[0, 10]``.
    """
    level = _float_parameter(_sample_level(level), 0.3)
    if np.random.uniform() > 0.5:
        level = -level
    return pil_img.transform(pil_img.size, Transform.AFFINE, (1, 0, 0, level, 1, 0), resample=Resampling.BILINEAR)


def solarize(pil_img: 'Image.Image', level: 'float'):
    """Solarize an image.

    .. seealso:: :func:`PIL.ImageOps.solarize`.

    Args:
        pil_img (PIL.Image.Image): The image.
        level (float): The intensity, which should
            be in ``[0, 10]``.
    """
    level = _int_parameter(_sample_level(level), 256)
    return ImageOps.solarize(pil_img, 256 - level)


def translate_x(pil_img: 'Image.Image', level: 'float'):
    """Shear an image horizontally.

    Args:
        pil_img (PIL.Image.Image): The image.
        level (float): The intensity, which should
            be in ``[0, 10]``.
    """
    level = _int_parameter(_sample_level(level), pil_img.size[0] / 3)
    if np.random.random() > 0.5:
        level = -level
    return pil_img.transform(pil_img.size, Transform.AFFINE, (1, 0, level, 0, 1, 0), resample=Resampling.BILINEAR)


def translate_y(pil_img: 'Image.Image', level: 'float'):
    """Shear an image vertically.

    Args:
        pil_img (PIL.Image.Image): The image.
        level (float): The intensity, which should
            be in ``[0, 10]``.
    """
    level = _int_parameter(_sample_level(level), pil_img.size[1] / 3)
    if np.random.random() > 0.5:
        level = -level
    return pil_img.transform(pil_img.size, Transform.AFFINE, (1, 0, 0, 0, 1, level), resample=Resampling.BILINEAR)


augmentation_sets = {'all': [autocontrast, equalize, posterize, rotate, solarize, shear_x, shear_y, translate_x, translate_y, color, contrast, brightness, sharpness], 'safe': [autocontrast, equalize, posterize, rotate, solarize, shear_x, shear_y, translate_x, translate_y], 'original': [autocontrast, equalize, posterize, rotate, solarize, shear_x, shear_y, translate_x, translate_y, color_original, contrast_original, brightness_original, sharpness_original]}


def _default_2d_filter():
    default_filter = torch.tensor([[[[1, 2, 1], [2, 4, 2], [1, 2, 1]]]]) * 1 / 16.0
    return default_filter


def _padding_for_filt_2d_same(filt: 'torch.Tensor'):
    _, _, h, w = filt.shape
    if h % 2 == 0:
        raise IndexError(f'Filter must have odd height; got {h}')
    if w % 2 == 0:
        raise IndexError(f'Filter must have odd width; got {w}')
    return int(torch.div(h, 2)), int(torch.div(w, 2))


def blur_2d(input: 'torch.Tensor', channels: 'int'=-1, stride: '_size_2_t'=1, filter: 'Optional[torch.Tensor]'=None) ->torch.Tensor:
    """Applies a spatial low-pass filter.

    Args:
        input (torch.Tensor): A 4d tensor of shape NCHW
        channels (int, optional): The number of channels in the input tensor.
            If non-positive, then dynamic control flow is used to determine the number of channels.
            If positive, then static control flow is used and the filter dimensions should be appropriate for
            the input size (note: this condition is always satisfied for the default filter and non-zero input size).
        stride (int | tuple, optional): Stride(s) along H and W axes. If a single value is passed, this
            value is used for both dimensions.
        filter (torch.Tensor, optional): A 2d or 4d tensor to be cross-correlated with the input tensor
            at each spatial position, within each channel. If 4d, the structure
            is required to be ``(C, 1, kH, kW)`` where ``C`` is the number of
            channels in the input tensor and ``kH`` and ``kW`` are the spatial
            sizes of the filter.

    By default, the filter used is:

    .. code-block:: python

            [1 2 1]
            [2 4 2] * 1/16
            [1 2 1]

    Returns:
        The blurred input
    """
    if filter is None:
        filter = _default_2d_filter()
    padding = _padding_for_filt_2d_same(filter)
    if channels < 1:
        _, channels, h, w = input.shape
        if filter.shape[0] == 1 and channels > 1:
            filter = filter.repeat((channels, 1, 1, 1))
        _, _, filter_h, filter_w = filter.shape
        if h + 2 * padding[0] < filter_h:
            return input
        if w + 2 * padding[1] < filter_w:
            return input
    return F.conv2d(input, filter, None, _pair(stride), _pair(padding), _pair(1), channels)


def blurmax_pool2d(input: 'torch.Tensor', kernel_size: 'Optional[_size_2_t]'=None, stride: '_size_2_t'=2, padding: '_size_2_t'=0, dilation: '_size_2_t'=1, ceil_mode: 'bool'=False, filter: 'Optional[torch.Tensor]'=None) ->torch.Tensor:
    """Max-pooling with anti-aliasing.

    This is a nearly drop-in replacement for PyTorch's :func:`torch.nn.functional.max_pool2d`.
    The only API difference is that the parameter ``return_indices`` is not
    available, because it is ill-defined when using anti-aliasing.

    See the associated `paper <http://proceedings.mlr.press/v97/zhang19a.html>`_
    for more details, experimental results, etc.

    This function can be understood as decoupling the max from the pooling, and
    inserting a low-pass filtering step between the two. Concretely, this
    function computes the max within spatial neighborhoods of shape
    ``kernel_size``, then applies an anti-aliasing filter to smooth the maxes,
    and only then pools according to ``stride``.

    See also: :func:`.blur_2d`.

    Args:
        input (torch.Tensor): A 4d tensor of shape NCHW
        kernel_size (int | tuple, optional): Size(s) of the spatial neighborhoods over which to pool.
            This is mostly commonly 2x2. If only a scalar ``s`` is provided, the
            neighborhood is of size ``(s, s)``. Default: ``(2, 2)``.
        stride (int | tuple, optional): Stride(s) along H and W axes. If a single value is passed, this
            value is used for both dimensions. Default: 2.
        padding (int | tuple, optional): implicit zero-padding to use. For the default 3x3 low-pass
            filter, ``padding=1`` (the default) returns output of the same size
            as the input. Default: 0.
        dilation (int | tuple, optional): Amount by which to "stretch" the pooling region for a given
            total size. See :class:`torch.nn.MaxPool2d`
            for our favorite explanation of how this works. Default: 1.
        ceil_mode (bool): When True, will use ceil instead of floor to compute the output shape. Default: ``False``.
        filter (torch.Tensor, optional): A 2d or 4d tensor to be cross-correlated with the input tensor
            at each spatial position, within each channel. If 4d, the structure
            is required to be ``(C, 1, kH, kW)`` where ``C`` is the number of
            channels in the input tensor and ``kH`` and ``kW`` are the spatial
            sizes of the filter.

    By default, the filter used is:

    .. code-block:: python

            [1 2 1]
            [2 4 2] * 1/16
            [1 2 1]

    Returns:
         The blurred and max-pooled input
    """
    if kernel_size is None:
        kernel_size = 2, 2
    maxs = F.max_pool2d(input, kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation, ceil_mode=ceil_mode)
    return blur_2d(maxs, channels=-1, stride=stride, filter=filter)


class BlurMaxPool2d(nn.Module):
    """This module is a (nearly) drop-in replacement for :class:`torch.nn.MaxPool2d`, but with an anti-aliasing filter.

    The only API difference is that the parameter ``return_indices`` is not
    available, because it is ill-defined when using anti-aliasing.

    See the associated `paper <http://proceedings.mlr.press/v97/zhang19a.html>`_
    for more details, experimental results, etc.

    See :func:`.blurmax_pool2d` for details.
    """

    def __init__(self, kernel_size: '_size_2_t', stride: 'Optional[_size_2_t]'=None, padding: '_size_2_t'=0, dilation: '_size_2_t'=1, ceil_mode: 'bool'=False):
        super(BlurMaxPool2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.dilation = dilation
        self.ceil_mode = ceil_mode
        self.register_buffer('filt2d', _default_2d_filter())

    def extra_repr(self) ->str:
        return 'kernel_size={kernel_size}, stride={stride}, padding={padding}, dilation={dilation}, ceil_mode={ceil_mode}'.format(**self.__dict__)

    def forward(self, input: 'torch.Tensor'):
        return blurmax_pool2d(input, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, dilation=self.dilation, ceil_mode=self.ceil_mode, filter=self.filt2d)

    @staticmethod
    def from_maxpool2d(module: 'torch.nn.MaxPool2d', module_index: 'int') ->'BlurMaxPool2d':
        return BlurMaxPool2d(kernel_size=module.kernel_size, stride=module.stride, padding=module.padding, dilation=module.dilation, ceil_mode=module.ceil_mode)


class BlurConv2d(nn.Module):
    """This module is a drop-in replacement for :class:`torch.nn.Conv2d`, but with an anti-aliasing filter.

    The one new parameter is ``blur_first``. When set to ``True``, the
    anti-aliasing filter is applied before the underlying convolution and
    vice-versa when set to ``False``. This mostly makes a difference when the
    stride is greater than one. In the former case, the only overhead is the
    cost of doing the anti-aliasing operation. In the latter case, the ``Conv2d``
    is applied with a stride of one to the input, and then the
    anti-aliasing is applied with the provided stride to the result. Setting
    the stride of the convolution to ``1`` can greatly increase the computational
    cost. E.g., replacing a stride of ``(2, 2)`` with a stride of ``1`` increases
    the number of operations by a factor of ``(2/1) * (2/1) = 4``. However,
    this approach most closely matches the behavior specified in the paper.

    This module should only be used to replace strided convolutions.

    See the associated `paper <http://proceedings.mlr.press/v97/zhang19a.html>`_
    for more details, experimental results, etc.

    See also: :func:`.blur_2d`.
    """

    def __init__(self, in_channels: 'int', out_channels: 'int', kernel_size: '_size_2_t', stride: '_size_2_t'=None, padding: '_size_2_t'=0, dilation: '_size_2_t'=1, groups: 'int'=1, bias: 'bool'=True, blur_first: 'bool'=True):
        super(BlurConv2d, self).__init__()
        self.blur_first = blur_first
        if self.blur_first:
            assert stride is not None
            conv_stride = stride
            self.blur_stride = 1
            self.blur_nchannels = in_channels
        else:
            conv_stride = 1
            self.blur_stride = kernel_size if stride is None else stride
            self.blur_nchannels = out_channels
        self.conv = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=conv_stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.conv._already_blurpooled = True
        filt = _default_2d_filter().repeat(self.blur_nchannels, 1, 1, 1)
        self.register_buffer('blur_filter', filt)

    def forward(self, input: 'torch.Tensor'):
        if self.blur_first:
            blurred = blur_2d(input, channels=self.blur_nchannels, filter=self.blur_filter, stride=self.blur_stride)
            return self.conv.forward(blurred)
        else:
            activations = self.conv.forward(input)
            return blur_2d(activations, channels=self.blur_nchannels, filter=self.blur_filter, stride=self.blur_stride)

    @staticmethod
    def from_conv2d(module: 'torch.nn.Conv2d', module_index: 'int'=-1, blur_first: 'bool'=True):
        has_bias = module.bias is not None and module.bias is not False
        blurconv = BlurConv2d(in_channels=module.in_channels, out_channels=module.out_channels, kernel_size=module.kernel_size, stride=module.stride, padding=module.padding, dilation=module.dilation, groups=module.groups, bias=has_bias, blur_first=blur_first)
        with torch.no_grad():
            blurconv.conv.weight.copy_(module.weight)
            if has_bias:
                blurconv.conv.bias.copy_(module.bias)
        return blurconv


class BlurPool2d(nn.Module):
    """This module just calls :func:`.blur_2d` in ``forward`` using the provided arguments."""

    def __init__(self, channels: 'int'=0, stride: '_size_2_t'=2, padding: '_size_2_t'=1) ->None:
        super(BlurPool2d, self).__init__()
        self.channels = channels
        self.stride = stride
        self.padding = padding
        self.register_buffer('blur_filter', _default_2d_filter())
        if self.channels > 0:
            self.blur_filter = self.blur_filter.repeat(channels, 1, 1, 1)

    def forward(self, input: 'torch.Tensor'):
        return blur_2d(input, channels=self.channels, stride=self.stride, filter=self.blur_filter)


def _max_rank_with_possible_speedup(in_channels: 'int', out_channels: 'int', kernel_size: 'Optional[_size_2_t]'=None) ->int:
    fan_in = in_channels
    if kernel_size is not None:
        fan_in *= np.prod(kernel_size)
    breakeven = fan_in * out_channels / (fan_in + out_channels)
    return int(math.ceil(breakeven - 1))


def _apply_solution_to_module_parameters(solution: 'LowRankSolution', module0: 'torch.nn.Module', module1: 'torch.nn.Module', transpose: 'bool') ->None:
    error_msg = "Can't apply unititalized solution!"
    assert solution.bias is not None, error_msg
    assert solution.Wa is not None, error_msg
    assert solution.Wb is not None, error_msg
    with torch.no_grad():
        if module0.bias is not None:
            assert isinstance(module0.bias, torch.Tensor)
            module0.bias = torch.nn.parameter.Parameter(torch.zeros(solution.rank, dtype=module0.bias.dtype))
        assert isinstance(module1.bias, torch.Tensor)
        module1.bias.copy_(solution.bias)
        Wa = solution.Wa
        Wb = solution.Wb
        if transpose:
            Wa = torch.transpose(Wa, 0, 1)
            Wb = torch.transpose(Wb, 0, 1)
        module0.weight = torch.nn.parameter.Parameter(Wa)
        module1.weight = torch.nn.parameter.Parameter(Wb)


def _activations_conv2d_to_mat(activations, kernel_size, padding=0, padding_mode='zeros', stride=1, dilation=1, groups=1):
    if np.max(stride) > 1:
        raise NotImplementedError(f'Stride != 1 not implemented; got {stride}')
    if np.max(dilation) > 1:
        raise NotImplementedError(f'Dilation != 1 not implemented; got {dilation}')
    if groups != 1:
        raise NotImplementedError(f'Groups != 1 not implemented; got {groups}')
    if np.max(padding) > 0 and padding_mode.lower() != 'zeros':
        if not isinstance(padding, list):
            padding = [padding]
        activations = F.pad(activations, pad=padding, mode=padding_mode)
        padding = 0
    ret = F.unfold(activations, kernel_size=kernel_size, padding=padding)
    ret = ret.transpose(1, 2)
    return ret.reshape(-1, ret.shape[2])


def _mat_to_weights_conv2d(mat: 'Optional[torch.Tensor]', kernel_size) ->Optional[torch.Tensor]:
    if mat is None:
        return None
    w = mat.T
    return w.reshape(w.shape[0], -1, *kernel_size)


def _weights_conv2d_to_mat(weights: 'torch.Tensor'):
    return weights.reshape(weights.shape[0], -1).T


def _lstsq(A: 'torch.Tensor', B: 'torch.Tensor') ->torch.Tensor:
    if A.shape[0] != B.shape[0]:
        raise RuntimeError(f'A has different number of rows than B! A.shape = {A.shape}, B.shape = {B.shape}')
    if A.ndim != 2:
        raise RuntimeError('A is not a rank 2 tensor: has shape', A.shape)
    if B.ndim != 2:
        raise RuntimeError('B is not a rank 2 tensor: has shape', A.shape)
    return torch.linalg.lstsq(A, B).solution


def _nmse(Y: 'torch.Tensor', Y_hat: 'torch.Tensor') ->float:
    diffs = Y.detach() - Y_hat.detach()
    return float((diffs * diffs).mean() / Y.var())


def _svd_initialize(Wa: 'torch.Tensor', Wb: 'Optional[torch.Tensor]', k: 'int') ->tuple[torch.Tensor, torch.Tensor]:
    if Wb is None:
        W = Wa
    else:
        W = Wa @ Wb
    U, s, Vt = torch.linalg.svd(W, full_matrices=False)
    Wa = U[:, :k]
    Wb = Vt[:k]
    s_sqrt = torch.sqrt(s[:k])
    Wa *= s_sqrt
    Wb *= s_sqrt.reshape(-1, 1)
    return Wa, Wb


class BERTGatedFFOutput(torch.nn.Module):
    """
    Defines a single feed-forward block that uses `Gated Linear Units <https://arxiv.org/abs/2002.05202>`_.

    Args:
        d_embed (int): The input dimension for the feed-forward network.
        d_ff (int): The hidden dimension for the feed-forward network.
        dropout_rate (float): The dropout rate to use between the two projection matricies in the feed-forward block.
        act_fn (Callable[torch.Tensor, torch.Tensor]): The activation function to use in the feed-forward network.
        layernorm_eps (float): The epsilon term to use in the LayerNorm operator. Useful for when the variance is small.
        gated_layer_bias (bool): Whether to use a bias term in the gated projection matrix.
        non_gated_layer_bias (bool): Whether to use a bias term in teh non-gated projection matrix.
    """

    def __init__(self, d_embed: 'int', d_ff: 'int', dropout_rate: 'float', act_fn: 'Callable[[torch.Tensor], torch.Tensor]', layernorm_eps: 'float', gated_layer_bias: 'bool'=False, non_gated_layer_bias: 'bool'=False):
        super().__init__()
        self.gated_layer = torch.nn.Linear(d_embed, d_ff, bias=gated_layer_bias)
        self.non_gated_layer = torch.nn.Linear(d_embed, d_ff, bias=non_gated_layer_bias)
        self.wo = torch.nn.Linear(d_ff, d_embed)
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.act = act_fn
        self.layernorm = torch.nn.LayerNorm(d_embed, eps=layernorm_eps)

    def forward(self, hidden_states: 'torch.Tensor', residual_connection: 'torch.Tensor'):
        """
        Args:
            hidden_states (torch.Tensor): The hidden states from the attention matrix.
            residual_connection (torch.Tensor): The residual connection to add before the LayerNorm operator.
        """
        hidden_states = self.act(self.gated_layer(hidden_states)) * self.non_gated_layer(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.wo(hidden_states)
        hidden_states = self.layernorm(hidden_states + residual_connection)
        return hidden_states


_TORCH_BATCHNORM_BASE_CLASS = torch.nn.modules.batchnorm._BatchNorm

