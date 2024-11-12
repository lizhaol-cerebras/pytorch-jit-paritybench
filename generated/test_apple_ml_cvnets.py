
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


from typing import Optional


from typing import Tuple


from typing import Union


import torch


from torch import Tensor


from itertools import product


from typing import List


import numpy as np


from torch import nn


from torch.nn import functional as F


from typing import Dict


import torch.nn


from typing import Any


from typing import Type


from torch import Size


import math


import random


from torchvision.ops import StochasticDepth as StochasticDepthTorch


from copy import deepcopy


import re


from types import MethodType


from typing import Mapping


from torch.utils.checkpoint import checkpoint as gradient_checkpoint_fn


from torch.nn import init


import copy


import torch.nn as nn


from functools import partial


from torch.utils.checkpoint import checkpoint_sequential as gradient_checkpoint_fn


from torchvision.models.detection.anchor_utils import AnchorGenerator


from torchvision.models.detection.mask_rcnn import MaskRCNN


from torchvision.ops import MultiScaleRoIAlign


from torchvision.ops import batched_nms


import torch.nn.functional as F


from typing import Sequence


from torchvision.ops.roi_align import RoIAlign


from torch.nn import functional


from torch.utils.data import default_collate


from torch.utils.data.sampler import Sampler


from torchvision.datasets import ImageFolder


from abc import ABC


from typing import TypedDict


from torch.utils import data


from typing import AnyStr


from torch.utils.data import DataLoader


from typing import Callable


from typing import Iterator


import torch.distributed as dist


import itertools


from torchvision import transforms as T


from torchvision.transforms import functional as F


from torch.nn import functional as F_torch


from torchvision.io import write_video


from torchvision.transforms import functional as FV


import numpy


from torchvision.transforms import functional as F_vision


import time


from torch.cuda.amp import autocast


import abc


import matplotlib.pyplot as plt


from matplotlib import animation


from matplotlib import cm


from torch.cuda.amp import GradScaler


from torch.distributed.elastic.multiprocessing import errors


from numbers import Number


from sklearn.metrics import average_precision_score


from typing import Iterable


from torch.optim import Adam


from torch.optim import AdamW


from torch.optim import SGD


import torch.utils.data as data


from collections import OrderedDict


import scipy.io.wavfile as wav


from torch.utils.mobile_optimizer import optimize_for_mobile


from torch import distributed as dist


from torch.autograd import Function


from matplotlib.colors import hsv_to_rgb


def parameter_list(named_parameters, weight_decay: 'Optional[float]'=0.0, no_decay_bn_filter_bias: 'Optional[bool]'=False, *args, **kwargs) ->List[Dict]:
    module_name = kwargs.get('module_name', '')
    with_decay = []
    without_decay = []
    with_decay_param_names = []
    without_decay_param_names = []
    if isinstance(named_parameters, list):
        for n_parameter in named_parameters:
            for p_name, param in n_parameter():
                if param.requires_grad and len(param.shape) == 1 and no_decay_bn_filter_bias:
                    without_decay.append(param)
                    without_decay_param_names.append(module_name + p_name)
                elif param.requires_grad:
                    with_decay.append(param)
                    with_decay_param_names.append(module_name + p_name)
    else:
        for p_name, param in named_parameters():
            if param.requires_grad and len(param.shape) == 1 and no_decay_bn_filter_bias:
                without_decay.append(param)
                without_decay_param_names.append(module_name + p_name)
            elif param.requires_grad:
                with_decay.append(param)
                with_decay_param_names.append(module_name + p_name)
    param_list = [{'params': with_decay, 'weight_decay': weight_decay, 'param_names': with_decay_param_names}]
    if len(without_decay) > 0:
        param_list.append({'params': without_decay, 'weight_decay': 0.0, 'param_names': without_decay_param_names})
    return param_list


class BaseImageProjectionHead(nn.Module):
    """Base class that projects image representations to the same space as text representations"""

    def __init__(self, opts, *args, **kwargs) ->None:
        super().__init__()
        self.lr_mult = getattr(opts, 'model.image_projection_head.lr_multiplier', 1.0)

    @classmethod
    def add_arguments(cls, parser: 'argparse.ArgumentParser'):
        """Add model specific arguments"""
        group = parser.add_argument_group(title=cls.__name__)
        group.add_argument('--model.image-projection-head.name', type=str, default=None, help='Name of the image projection head')
        group.add_argument('--model.image-projection-head.lr-multiplier', type=float, default=1.0, help='LR multiplier for image projection head')
        return parser

    def reset_parameters(self) ->None:
        """Reset weights of a given layer"""
        raise NotImplementedError

    def get_trainable_parameters(self, weight_decay: 'Optional[float]'=0.0, no_decay_bn_filter_bias: 'Optional[bool]'=False, *args, **kwargs):
        param_list = parameter_list(named_parameters=self.named_parameters, weight_decay=weight_decay, no_decay_bn_filter_bias=no_decay_bn_filter_bias)
        return param_list, [self.lr_mult] * len(param_list)

    def forward(self, input: 'Dict', *args, **kwargs) ->Dict:
        raise NotImplementedError


def is_master(opts) ->bool:
    node_rank = getattr(opts, 'ddp.rank', 0)
    return node_rank == 0


class GELU(nn.GELU):
    """
    Applies the `Gaussian Error Linear Units <https://arxiv.org/abs/1606.08415>`_ function
    """

    def __init__(self, *args, **kwargs) ->None:
        super().__init__()


class Hardsigmoid(nn.Hardsigmoid):
    """
    Applies the `Hard Sigmoid <https://arxiv.org/abs/1511.00363v3>`_ function
    """

    def __init__(self, inplace: 'Optional[bool]'=False, *args, **kwargs) ->None:
        super().__init__(inplace=inplace)

    def forward(self, input: 'Tensor', *args, **kwargs) ->Tensor:
        if hasattr(F, 'hardsigmoid'):
            return F.hardsigmoid(input, self.inplace)
        else:
            return F.relu(input + 3) / 6


class Hardswish(nn.Hardswish):
    """
    Applies the HardSwish function, as described in the paper
    `Searching for MobileNetv3 <https://arxiv.org/abs/1905.02244>`_
    """

    def __init__(self, inplace: 'Optional[bool]'=False, *args, **kwargs) ->None:
        super().__init__(inplace=inplace)

    def forward(self, input: 'Tensor', *args, **kwargs) ->Tensor:
        if hasattr(F, 'hardswish'):
            return F.hardswish(input, self.inplace)
        else:
            x_hard_sig = F.relu(input + 3) / 6
            return input * x_hard_sig


class LeakyReLU(nn.LeakyReLU):
    """
    Applies a leaky relu function. See `Rectifier Nonlinearities Improve Neural Network Acoustic Models`
    for more details.
    """

    def __init__(self, negative_slope: 'Optional[float]'=0.01, inplace: 'Optional[bool]'=False, *args, **kwargs) ->None:
        super().__init__(negative_slope=negative_slope, inplace=inplace)


class PReLU(nn.PReLU):
    """
    Applies the `Parametric Rectified Linear Unit <https://arxiv.org/abs/1502.01852>`_ function
    """

    def __init__(self, num_parameters: 'Optional[int]'=1, init: 'Optional[float]'=0.25, *args, **kwargs) ->None:
        super().__init__(num_parameters=num_parameters, init=init)


class ReLU(nn.ReLU):
    """
    Applies Rectified Linear Unit function
    """

    def __init__(self, inplace: 'Optional[bool]'=False, *args, **kwargs) ->None:
        super().__init__(inplace=inplace)


class ReLU6(nn.ReLU6):
    """
    Applies the ReLU6 function
    """

    def __init__(self, inplace: 'Optional[bool]'=False, *args, **kwargs) ->None:
        super().__init__(inplace=inplace)


class Sigmoid(nn.Sigmoid):
    """
    Applies the sigmoid function
    """

    def __init__(self, *args, **kwargs) ->None:
        super().__init__()


class Swish(nn.SiLU):
    """
    Applies the `Swish (also known as SiLU) <https://arxiv.org/abs/1702.03118>`_ function.
    """

    def __init__(self, inplace: 'Optional[bool]'=False, *args, **kwargs) ->None:
        super().__init__(inplace=inplace)


class Tanh(nn.Tanh):
    """
    Applies Tanh function
    """

    def __init__(self, *args, **kwargs) ->None:
        super().__init__()


class AdaptiveAvgPool2d(nn.AdaptiveAvgPool2d):
    """
    Applies a 2D adaptive average pooling over an input tensor.

    Args:
        output_size (Optional, int or Tuple[int, int]): The target output size. If a single int :math:`h` is passed,
        then a square output of size :math:`hxh` is produced. If a tuple of size :math:`hxw` is passed, then an
        output of size `hxw` is produced. Default is 1.
    Shape:
        - Input: :math:`(N, C, H, W)` where :math:`N` is the batch size, :math:`C` is the number of input channels,
        :math:`H` is the input height, and :math:`W` is the input width
        - Output: :math:`(N, C, h, h)` or :math:`(N, C, h, w)`
    """

    def __init__(self, output_size: 'Union[int, Tuple[int, int]]'=1, *args, **kwargs) ->None:
        super().__init__(output_size=output_size)


class Conv2d(nn.Conv2d):
    """
    Applies a 2D convolution over an input.

    Args:
        in_channels: :math:`C_{in}` from an expected input of size :math:`(N, C_{in}, H_{in}, W_{in})`.
        out_channels: :math:`C_{out}` from an expected output of size :math:`(N, C_{out}, H_{out}, W_{out})`.
        kernel_size: Kernel size for convolution.
        stride: Stride for convolution. Default: 1.
        padding: Padding for convolution. Default: 0.
        dilation: Dilation rate for convolution. Default: 1.
        groups: Number of groups in convolution. Default: 1.
        bias: Use bias. Default: ``False``.
        padding_mode: Padding mode ('zeros', 'reflect', 'replicate' or 'circular'). Default: ``zeros``.
        use_norm: Use normalization layer after convolution. Default: ``True``.
        use_act: Use activation layer after convolution (or convolution and normalization).
                                Default: ``True``.
        act_name: Use specific activation function. Overrides the one specified in command line args.

    Shape:
        - Input: :math:`(N, C_{in}, H_{in}, W_{in})`.
        - Output: :math:`(N, C_{out}, H_{out}, W_{out})`.
    """

    def __init__(self, in_channels: 'int', out_channels: 'int', kernel_size: 'Union[int, Tuple[int, int]]', stride: 'Optional[Union[int, Tuple[int, int]]]'=1, padding: 'Optional[Union[int, Tuple[int, int]]]'=0, dilation: 'Optional[Union[int, Tuple[int, int]]]'=1, groups: 'Optional[int]'=1, bias: 'Optional[bool]'=False, padding_mode: 'Optional[str]'='zeros', *args, **kwargs) ->None:
        super().__init__(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode)


class LayerNorm(nn.LayerNorm):
    """
    Applies `Layer Normalization <https://arxiv.org/abs/1607.06450>`_ over a input tensor

    Args:
        normalized_shape (int or list or torch.Size): input shape from an expected input
            of size

            .. math::
                [* \\times \\text{normalized\\_shape}[0] \\times \\text{normalized\\_shape}[1]
                    \\times \\ldots \\times \\text{normalized\\_shape}[-1]]

            If a single integer is used, it is treated as a singleton list, and this module will
            normalize over the last dimension which is expected to be of that specific size.
        eps (Optional, float): Value added to the denominator for numerical stability. Default: 1e-5
        elementwise_affine (bool): If ``True``, use learnable affine parameters. Default: ``True``

    Shape:
        - Input: :math:`(N, *)` where :math:`N` is the batch size
        - Output: same shape as the input
    """

    def __init__(self, normalized_shape: 'Union[int, List[int], Size]', eps: 'Optional[float]'=1e-05, elementwise_affine: 'Optional[bool]'=True, *args, **kwargs):
        super().__init__(normalized_shape=normalized_shape, eps=eps, elementwise_affine=elementwise_affine)

    def forward(self, x: 'Tensor') ->Tensor:
        n_dim = x.ndim
        if x.shape[1] == self.normalized_shape[0] and n_dim > 2:
            s, u = torch.std_mean(x, dim=1, keepdim=True, unbiased=False)
            x = (x - u) / (s + self.eps)
            if self.weight is not None:
                n_dim = x.ndim - 2
                new_shape = [1, self.normalized_shape[0]] + [1] * n_dim
                x = torch.addcmul(input=self.bias.reshape(*[new_shape]), value=1.0, tensor1=x, tensor2=self.weight.reshape(*[new_shape]))
            return x
        elif x.shape[-1] == self.normalized_shape[0]:
            return super().forward(x)
        else:
            raise NotImplementedError('LayerNorm is supported for channel-first and channel-last format only')


class LayerNorm2D_NCHW(nn.GroupNorm):
    """
    Applies `Layer Normalization <https://arxiv.org/abs/1607.06450>`_ over a 4D input tensor

    Args:
        num_features (int): :math:`C` from an expected input of size :math:`(N, C, H, W)`
        eps (Optional, float): Value added to the denominator for numerical stability. Default: 1e-5
        elementwise_affine (bool): If ``True``, use learnable affine parameters. Default: ``True``

    Shape:
        - Input: :math:`(N, C, H, W)` where :math:`N` is the batch size, :math:`C` is the number of input channels,
        :math:`H` is the input height, and :math:`W` is the input width
        - Output: same shape as the input
    """

    def __init__(self, num_features: 'int', eps: 'Optional[float]'=1e-05, elementwise_affine: 'Optional[bool]'=True, *args, **kwargs) ->None:
        super().__init__(num_channels=num_features, eps=eps, affine=elementwise_affine, num_groups=1)
        self.num_channels = num_features

    def __repr__(self):
        return '{}(num_channels={}, eps={}, affine={})'.format(self.__class__.__name__, self.num_channels, self.eps, self.affine)


ACT_FN_REGISTRY = {}


SUPPORTED_ACT_FNS = []


def build_activation_layer(opts: 'argparse.Namespace', act_type: 'Optional[str]'=None, inplace: 'Optional[bool]'=None, negative_slope: 'Optional[float]'=None, num_parameters: 'int'=-1) ->torch.nn.Module:
    """
    Helper function to build the activation function. If any of the optional
    arguments are not provided (i.e. None), the corresponding ``model.activation.*``
    config entry will be used as default value.

    Args:
        act_type: Name of the activation layer.
            Default: --model.activation.name config value.
        inplace: If true, operation will be inplace.
            Default: --model.activation.inplace config value.
        negative_slope: Negative slope parameter for leaky_relu.
            Default: --model.activation.neg_slop config value.
    """
    assert isinstance(opts, argparse.Namespace), f'Expected first argument to be an argparse.Namespace, but received a {type(opts)}.'
    if act_type is None:
        act_type = getattr(opts, 'model.activation.name')
    if inplace is None:
        inplace = getattr(opts, 'model.activation.inplace')
    if negative_slope is None:
        negative_slope = getattr(opts, 'model.activation.neg_slope')
    act_type = act_type.lower()
    act_layer = None
    if act_type in ACT_FN_REGISTRY:
        act_layer = ACT_FN_REGISTRY[act_type](num_parameters=num_parameters, inplace=inplace, negative_slope=negative_slope)
    else:
        logger.error('Supported activation layers are: {}. Supplied argument is: {}'.format(SUPPORTED_ACT_FNS, act_type))
    return act_layer


class Identity(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x: 'Any') ->Any:
        return x


NORM_LAYER_REGISTRY = {}


SUPPORTED_NORM_FNS = []


def build_normalization_layer(opts: 'argparse.Namespace', num_features: 'int', norm_type: 'Optional[str]'=None, num_groups: 'Optional[int]'=None, momentum: 'Optional[float]'=None) ->torch.nn.Module:
    """
    Helper function to build the normalization layer.
    The function can be used in either of below mentioned ways:
    Scenario 1: Set the default normalization layers using command line arguments. This is useful when the same normalization
    layer is used for the entire network (e.g., ResNet).
    Scenario 2: Network uses different normalization layers. In that case, we can override the default normalization
    layer by specifying the name using `norm_type` argument.
    """
    if norm_type is None:
        norm_type = getattr(opts, 'model.normalization.name')
    if num_groups is None:
        num_groups = getattr(opts, 'model.normalization.groups')
    if momentum is None:
        momentum = getattr(opts, 'model.normalization.momentum')
    norm_layer = None
    norm_type = norm_type.lower()
    if norm_type in NORM_LAYER_REGISTRY:
        if 'cuda' not in str(getattr(opts, 'dev.device', 'cpu')) and 'sync_batch' in norm_type:
            norm_type = norm_type.replace('sync_', '')
        norm_layer = NORM_LAYER_REGISTRY[norm_type](normalized_shape=num_features, num_features=num_features, momentum=momentum, num_groups=num_groups)
    elif norm_type == 'identity':
        norm_layer = Identity()
    else:
        logger.error('Supported normalization layer arguments are: {}. Got: {}'.format(SUPPORTED_NORM_FNS, norm_type))
    return norm_layer


get_normalization_layer = build_normalization_layer


class Dropout(nn.Dropout):
    """
    This layer, during training, randomly zeroes some of the elements of the input tensor with probability `p`
    using samples from a Bernoulli distribution.

    Args:
        p: probability of an element to be zeroed. Default: 0.5
        inplace: If set to ``True``, will do this operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(N, *)` where :math:`N` is the batch size
        - Output: same as the input

    """

    def __init__(self, p: 'Optional[float]'=0.5, inplace: 'Optional[bool]'=False, *args, **kwargs) ->None:
        super().__init__(p=p, inplace=inplace)


class Dropout2d(nn.Dropout2d):
    """
    This layer, during training, randomly zeroes some of the elements of the 4D input tensor with probability `p`
    using samples from a Bernoulli distribution.

    Args:
        p: probability of an element to be zeroed. Default: 0.5
        inplace: If set to ``True``, will do this operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(N, C, H, W)` where :math:`N` is the batch size, :math:`C` is the input channels,
            :math:`H` is the input tensor height, and :math:`W` is the input tensor width
        - Output: same as the input

    """

    def __init__(self, p: 'float'=0.5, inplace: 'bool'=False):
        super().__init__(p=p, inplace=inplace)


class Embedding(nn.Embedding):
    """A lookup table that stores embeddings of a fixed dictionary and size.

    Args:
        num_embeddings (int): size of the dictionary of embeddings
        embedding_dim (int): the size of each embedding vector
        padding_idx (int, optional): If specified, the entries at :attr:`padding_idx` do not contribute to the gradient;
                                     therefore, the embedding vector at :attr:`padding_idx` is not updated during training,
                                     i.e. it remains as a fixed "pad". For a newly constructed Embedding,
                                     the embedding vector at :attr:`padding_idx` will default to all zeros,
                                     but can be updated to another value to be used as the padding vector.

    Shape:
        - Input: :math:`(*)`, IntTensor or LongTensor of arbitrary shape containing the indices to extract
        - Output: :math:`(*, H)`, where `*` is the input shape and :math:`H=\\text{embedding\\_dim}`
    """

    def __init__(self, opts, num_embeddings: 'int', embedding_dim: 'int', padding_idx: 'Optional[int]'=None, *args, **kwargs):
        super().__init__(num_embeddings=num_embeddings, embedding_dim=embedding_dim, padding_idx=padding_idx)

    def reset_parameters(self) ->None:
        nn.init.normal_(self.weight, mean=0, std=self.embedding_dim ** -0.5)
        if self.padding_idx is not None:
            nn.init.constant_(self.weight[self.padding_idx], 0)


class Flatten(nn.Flatten):
    """
    This layer flattens a contiguous range of dimensions into a tensor.

    Args:
        start_dim (Optional[int]): first dim to flatten. Default: 1
        end_dim (Optional[int]): last dim to flatten. Default: -1

    Shape:
        - Input: :math:`(*, S_{\\text{start}},..., S_{i}, ..., S_{\\text{end}}, *)`,'
          where :math:`S_{i}` is the size at dimension :math:`i` and :math:`*` means any
          number of dimensions including none.
        - Output: :math:`(*, \\prod_{i=\\text{start}}^{\\text{end}} S_{i}, *)`.
    """

    def __init__(self, start_dim: 'Optional[int]'=1, end_dim: 'Optional[int]'=-1):
        super(Flatten, self).__init__(start_dim=start_dim, end_dim=end_dim)


class BatchNorm2d(nn.BatchNorm2d):
    """
    Applies a `Batch Normalization <https://arxiv.org/abs/1502.03167>`_ over a 4D input tensor

    Args:
        num_features (Optional, int): :math:`C` from an expected input of size :math:`(N, C, H, W)`
        eps (Optional, float): Value added to the denominator for numerical stability. Default: 1e-5
        momentum (Optional, float): Value used for the running_mean and running_var computation. Default: 0.1
        affine (bool): If ``True``, use learnable affine parameters. Default: ``True``
        track_running_stats: If ``True``, tracks running mean and variance. Default: ``True``

    Shape:
        - Input: :math:`(N, C, H, W)` where :math:`N` is the batch size, :math:`C` is the number of input channels,
        :math:`H` is the input height, and :math:`W` is the input width
        - Output: same shape as the input
    """

    def __init__(self, num_features: 'int', eps: 'Optional[float]'=1e-05, momentum: 'Optional[float]'=0.1, affine: 'Optional[bool]'=True, track_running_stats: 'Optional[bool]'=True, *args, **kwargs) ->None:
        super().__init__(num_features=num_features, eps=eps, momentum=momentum, affine=affine, track_running_stats=track_running_stats)


class BatchNorm2dFP32(BatchNorm2d):
    """
    Applies a `Batch Normalization <https://arxiv.org/abs/1502.03167>`_ over a 4D input tensor in FP32
    """

    def __init__(self, num_features: 'int', eps: 'Optional[float]'=1e-05, momentum: 'Optional[float]'=0.1, affine: 'Optional[bool]'=True, track_running_stats: 'Optional[bool]'=True, *args, **kwargs) ->None:
        super().__init__(*args, num_features=num_features, eps=eps, momentum=momentum, affine=affine, track_running_stats=track_running_stats, **kwargs)

    def forward(self, input: 'Tensor') ->Tensor:
        inp_dtype = input.dtype
        return super().forward(input.to(torch.float32))


class BatchNorm1d(nn.BatchNorm1d):
    """
    Applies a `Batch Normalization <https://arxiv.org/abs/1502.03167>`_ over a 2D or 3D input tensor

    Args:
        num_features (Optional, int): :math:`C` from an expected input of size :math:`(N, C)` or :math:`(N, C, L)`
        eps (Optional, float): Value added to the denominator for numerical stability. Default: 1e-5
        momentum (Optional, float): Value used for the running_mean and running_var computation. Default: 0.1
        affine (bool): If ``True``, use learnable affine parameters. Default: ``True``
        track_running_stats: If ``True``, tracks running mean and variance. Default: ``True``

    Shape:
        - Input: :math:`(N, C)` or :math:`(N, C, L)` where :math:`N` is the batch size,
        :math:`C` is the number of input channels,  and :math:`L` is the sequence length
        - Output: same shape as the input
    """

    def __init__(self, num_features: 'int', eps: 'Optional[float]'=1e-05, momentum: 'Optional[float]'=0.1, affine: 'Optional[bool]'=True, track_running_stats: 'Optional[bool]'=True, *args, **kwargs) ->None:
        super().__init__(num_features=num_features, eps=eps, momentum=momentum, affine=affine, track_running_stats=track_running_stats)


class BatchNorm3d(nn.BatchNorm3d):

    def __init__(self, num_features: 'int', eps: 'Optional[float]'=1e-05, momentum: 'Optional[float]'=0.1, affine: 'Optional[bool]'=True, track_running_stats: 'Optional[bool]'=True, *args, **kwargs) ->None:
        """
        Applies a `Batch Normalization <https://arxiv.org/abs/1502.03167>`_ over a 5D input tensor

        Args:
            num_features (Optional, int): :math:`C` from an expected input of size :math:`(N, C, D, H, W)`
            eps (Optional, float): Value added to the denominator for numerical stability. Default: 1e-5
            momentum (Optional, float): Value used for the running_mean and running_var computation. Default: 0.1
            affine (bool): If ``True``, use learnable affine parameters. Default: ``True``
            track_running_stats: If ``True``, tracks running mean and variance. Default: ``True``

        Shape:
            - Input: :math:`(N, C, D, H, W)` where :math:`N` is the batch size, :math:`C` is the number of input
            channels, :math:`D` is the input depth, :math:`H` is the input height, and :math:`W` is the input width
            - Output: same shape as the input
        """
        super().__init__(num_features=num_features, eps=eps, momentum=momentum, affine=affine, track_running_stats=track_running_stats)


class GroupNorm(nn.GroupNorm):
    """
    Applies a `Group Normalization <https://arxiv.org/abs/1803.08494>`_ over an input tensor

    Args:
        num_groups (int): number of groups to separate the input channels into
        num_features (int): :math:`C` from an expected input of size :math:`(N, C, *)`
        eps (Optional, float): Value added to the denominator for numerical stability. Default: 1e-5
        affine (bool): If ``True``, use learnable affine parameters. Default: ``True``

    Shape:
        - Input: :math:`(N, C, *)` where :math:`N` is the batch size, :math:`C` is the number of input channels,
        and :math:`*` is the remaining dimensions of the input tensor
        - Output: same shape as the input

    .. note::
        GroupNorm is the same as LayerNorm when `num_groups=1` and it is the same as InstanceNorm when
        `num_groups=C`.
    """

    def __init__(self, num_groups: 'int', num_features: 'int', eps: 'Optional[float]'=1e-05, affine: 'Optional[bool]'=True, *args, **kwargs) ->None:
        super().__init__(num_groups=num_groups, num_channels=num_features, eps=eps, affine=affine)


class InstanceNorm2d(nn.InstanceNorm2d):
    """
    Applies a `Instance Normalization <https://arxiv.org/abs/1607.08022>`_ over a 4D input tensor

    Args:
        num_features (int): :math:`C` from an expected input of size :math:`(N, C, H, W)`
        eps (Optional, float): Value added to the denominator for numerical stability. Default: 1e-5
        momentum (Optional, float): Value used for the running_mean and running_var computation. Default: 0.1
        affine (bool): If ``True``, use learnable affine parameters. Default: ``True``
        track_running_stats: If ``True``, tracks running mean and variance. Default: ``True``

    Shape:
        - Input: :math:`(N, C, H, W)` where :math:`N` is the batch size, :math:`C` is the number of input channels,
        :math:`H` is the input height, and :math:`W` is the input width
        - Output: same shape as the input
    """

    def __init__(self, num_features: 'int', eps: 'Optional[float]'=1e-05, momentum: 'Optional[float]'=0.1, affine: 'Optional[bool]'=True, track_running_stats: 'Optional[bool]'=True, *args, **kwargs) ->None:
        super().__init__(num_features=num_features, eps=eps, momentum=momentum, affine=affine, track_running_stats=track_running_stats)


class InstanceNorm1d(nn.InstanceNorm1d):
    """
    Applies a `Instance Normalization <https://arxiv.org/abs/1607.08022>`_ over a 2D or 3D input tensor

    Args:
        num_features (int): :math:`C` from an expected input of size :math:`(N, C)` or :math:`(N, C, L)`
        eps (Optional, float): Value added to the denominator for numerical stability. Default: 1e-5
        momentum (Optional, float): Value used for the running_mean and running_var computation. Default: 0.1
        affine (bool): If ``True``, use learnable affine parameters. Default: ``True``
        track_running_stats: If ``True``, tracks running mean and variance. Default: ``True``

    Shape:
        - Input: :math:`(N, C)` or :math:`(N, C, L)` where :math:`N` is the batch size, :math:`C` is the number
        of input channels,  and :math:`L` is the sequence length
    - Output: same shape as the input
    """

    def __init__(self, num_features: 'int', eps: 'Optional[float]'=1e-05, momentum: 'Optional[float]'=0.1, affine: 'Optional[bool]'=True, track_running_stats: 'Optional[bool]'=True, *args, **kwargs) ->None:
        super().__init__(num_features=num_features, eps=eps, momentum=momentum, affine=affine, track_running_stats=track_running_stats)


class LayerNormFP32(LayerNorm):
    """
    Applies `Layer Normalization <https://arxiv.org/abs/1607.06450>`_ over a input tensor with FP32 precision
    """

    def __init__(self, normalized_shape: 'Union[int, List[int], Size]', eps: 'Optional[float]'=1e-05, elementwise_affine: 'Optional[bool]'=True, *args, **kwargs):
        super().__init__(*args, normalized_shape=normalized_shape, eps=eps, elementwise_affine=elementwise_affine, **kwargs)

    def forward(self, x: 'Tensor') ->Tensor:
        inp_dtype = x.dtype
        return super().forward(x.to(torch.float32))


class SyncBatchNorm(nn.SyncBatchNorm):
    """
    Applies a `Syncronized Batch Normalization <https://arxiv.org/abs/1502.03167>`_ over the input tensor

    Args:
        num_features (Optional, int): :math:`C` from an expected input of size :math:`(N, C, *)`
        eps (Optional, float): Value added to the denominator for numerical stability. Default: 1e-5
        momentum (Optional, float): Value used for the running_mean and running_var computation. Default: 0.1
        affine (bool): If ``True``, use learnable affine parameters. Default: ``True``
        track_running_stats: If ``True``, tracks running mean and variance. Default: ``True``

    Shape:
        - Input: :math:`(N, C, *)` where :math:`N` is the batch size, :math:`C` is the number of input channels,
        :math:`*` is the remaining input dimensions
        - Output: same shape as the input

    """

    def __init__(self, num_features: 'int', eps: 'Optional[float]'=1e-05, momentum: 'Optional[float]'=0.1, affine: 'Optional[bool]'=True, track_running_stats: 'Optional[bool]'=True, *args, **kwargs) ->None:
        super().__init__(num_features=num_features, eps=eps, momentum=momentum, affine=affine, track_running_stats=track_running_stats)


class SyncBatchNormFP32(SyncBatchNorm):
    """
    Synchronized BN in FP32
    """

    def __init__(self, num_features: 'int', eps: 'Optional[float]'=1e-05, momentum: 'Optional[float]'=0.1, affine: 'Optional[bool]'=True, track_running_stats: 'Optional[bool]'=True, *args, **kwargs) ->None:
        super().__init__(num_features=num_features, eps=eps, momentum=momentum, affine=affine, track_running_stats=track_running_stats)

    def forward(self, x: 'Tensor', *args, **kwargs) ->Tensor:
        in_dtype = x.dtype
        return super().forward(x.to(dtype=torch.float))


class PixelShuffle(nn.PixelShuffle):
    """
    Rearranges elements in a tensor of shape :math:`(*, C 	imes r^2, H, W)`
    to a tensor of shape :math:`(*, C, H 	imes r, W 	imes r)`, where r is an upscale factor.

    Args:
        upscale_factor (int): factor to increase spatial resolution by

    Shape:
        - Input: :math:`(*, C 	imes r^2, H, W)`, where * is zero or more dimensions
        - Output: :math:`(*, C, H 	imes r, W 	imes r)`
    """

    def __init__(self, upscale_factor: 'int', *args, **kwargs) ->None:
        super(PixelShuffle, self).__init__(upscale_factor=upscale_factor)

    def __repr__(self):
        return '{}(upscale_factor={})'.format(self.__class__.__name__, self.upscale_factor)


class MaxPool2d(nn.MaxPool2d):
    """
    Applies a 2D max pooling over a 4D input tensor.

    Args:
        kernel_size (Optional[int]): the size of the window to take a max over
        stride (Optional[int]): The stride of the window. Default: 2
        padding (Optional[int]): Padding to be added on both sides of the tensor. Default: 1

    Shape:
        - Input: :math:`(N, C, H_{in}, W_{in})` where :math:`N` is the batch size, :math:`C` is the input channels,
            :math:`H_{in}` is the input height, and :math:`W_{in}` is the input width
        - Output: :math:`(N, C, H_{out}, W_{out})` where :math:`H_{out}` is the output height, and :math:`W_{in}` is
            the output width
    """

    def __init__(self, kernel_size: 'Optional[int]'=3, stride: 'Optional[int]'=2, padding: 'Optional[int]'=1, *args, **kwargs) ->None:
        super().__init__(kernel_size=kernel_size, stride=stride, padding=padding)

    def __repr__(self):
        return '{}(kernel_size={}, stride={})'.format(self.__class__.__name__, self.kernel_size, self.stride)


class AvgPool2d(nn.AvgPool2d):
    """
    Applies a 2D average pooling over a 4D input tensor.

    Args:
        kernel_size (Optional[int]): the size of the window to take a max over
        stride (Optional[int]): The stride of the window. Default: 2
        padding (Optional[int]): Padding to be added on both sides of the tensor. Default: 1
        ceil_mode (Optional[bool]): When True, will use `ceil` instead of `floor` to compute the output shape. Default: False
        count_include_pad (Optional[bool]): When True, will include the zero-padding in the averaging calculation. Default: True
        divisor_override: if specified, it will be used as divisor, otherwise size of the pooling region will be used. Default: None

    Shape:
        - Input: :math:`(N, C, H_{in}, W_{in})` where :math:`N` is the batch size, :math:`C` is the input channels,
            :math:`H_{in}` is the input height, and :math:`W_{in}` is the input width
        - Output: :math:`(N, C, H_{out}, W_{out})` where :math:`H_{out}` is the output height, and :math:`W_{in}` is
            the output width
    """

    def __init__(self, kernel_size: 'tuple', stride: 'Optional[tuple]'=None, padding: 'Optional[tuple]'=(0, 0), ceil_mode: 'Optional[bool]'=False, count_include_pad: 'Optional[bool]'=True, divisor_override: 'Optional[bool]'=None):
        super(AvgPool2d, self).__init__(kernel_size=kernel_size, stride=stride, padding=padding, ceil_mode=ceil_mode, count_include_pad=count_include_pad, divisor_override=divisor_override)

    def __repr__(self):
        return '{}(upscale_factor={})'.format(self.__class__.__name__, self.upscale_factor)


class LearnablePositionalEmbedding(nn.Module):
    """Learnable Positional embedding"""

    def __init__(self, opts, num_embeddings: 'int', embedding_dim: 'int', padding_idx: 'Optional[int]'=None, sequence_first: 'Optional[bool]'=False, interpolation_mode: 'Optional[str]'='bilinear', *args, **kwargs):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.empty(1, 1, num_embeddings, embedding_dim))
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.padding_idx = padding_idx
        self.sequence_first = sequence_first
        self.interpolation_mode = interpolation_mode
        self.reset_parameters()

    def reset_parameters(self) ->None:
        nn.init.trunc_normal_(self.pos_embed, mean=0, std=self.embedding_dim ** -0.5)
        if self.padding_idx is not None:
            with torch.no_grad():
                self.pos_embed[:, :, self.padding_idx, ...] = 0.0

    def forward(self, seq_len: 'int', *args, **kwargs) ->Tensor:
        pos_embed = self.pos_embed
        if self.padding_idx is not None:
            with torch.no_grad():
                pos_embed[:, :, self.padding_idx, ...] = 0.0
        if seq_len != self.num_embeddings:
            pos_embed = F.interpolate(pos_embed, size=(seq_len, self.embedding_dim), mode=self.interpolation_mode)
        if self.sequence_first:
            return pos_embed.reshape(seq_len, 1, self.embedding_dim)
        else:
            return pos_embed.reshape(1, seq_len, self.embedding_dim)

    def __repr__(self):
        return '{}(num_embeddings={}, embedding_dim={}, padding_idx={}, sequence_first={})'.format(self.__class__.__name__, self.num_embeddings, self.embedding_dim, self.padding_idx, self.sequence_first)


class SinusoidalPositionalEmbedding(nn.Module):

    def __init__(self, opts, num_embeddings: 'int', embedding_dim: 'int', padding_idx: 'Optional[int]'=None, sequence_first: 'Optional[bool]'=False, interpolation_mode: 'Optional[str]'='bilinear', *args, **kwargs):
        super().__init__()
        self.padding_idx = padding_idx
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.sequence_first = sequence_first
        self.interpolation_mode = interpolation_mode
        self.register_buffer('pos_embed', self.get_weights())

    def get_weights(self) ->Tensor:
        """Build sinusoidal embeddings. Adapted from Fairseq."""
        half_dim = self.embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(self.num_embeddings, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).reshape(self.num_embeddings, -1)
        if self.embedding_dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros(self.num_embeddings, 1)], dim=1)
        if self.padding_idx is not None:
            emb[self.padding_idx, :] = 0
        return emb.unsqueeze(0).unsqueeze(0)

    def forward(self, seq_len: 'int', *args, **kwargs) ->Tensor:
        pos_embed = self.pos_embed
        if seq_len != self.num_embeddings:
            pos_embed = F.interpolate(pos_embed, size=(seq_len, self.embedding_dim), mode=self.interpolation_mode)
        if self.sequence_first:
            return pos_embed.reshape(seq_len, 1, self.embedding_dim)
        else:
            return pos_embed.reshape(1, seq_len, self.embedding_dim)

    def __repr__(self):
        return '{}(num_embeddings={}, embedding_dim={}, padding_idx={}, sequence_first={})'.format(self.__class__.__name__, self.num_embeddings, self.embedding_dim, self.padding_idx, self.sequence_first)


def bound_fn(min_val: 'Union[float, int]', max_val: 'Union[float, int]', value: 'Union[float, int]') ->Union[float, int]:
    return max(min_val, min(max_val, value))


class Softmax(nn.Softmax):
    """
    Applies the Softmax function to an input tensor along the specified dimension

    Args:
        dim (int): Dimension along which softmax to be applied. Default: -1

    Shape:
        - Input: :math:`(*)` where :math:`*` is one or more dimensions
        - Output: same shape as the input
    """

    def __init__(self, dim: 'Optional[int]'=-1, *args, **kwargs):
        super().__init__(dim=dim)


def pad_x_and_mask(x: 'torch.Tensor', key_padding_mask: 'torch.Tensor', window_size: 'int') ->Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply padding to @x and @key_padding_mask to make their lengths divisible
    by @window_size.

    Args:
        x: The input tensor of shape [B, N, C].
        key_padding_mask: The mask of shape [B, N].
        window_size: the N dimension of @x and @key_padding_mask will be padded
            to make them divisble by this number.

    Returns:
        A tuple containing @x and @key_padding_mask, with padding applied.
    """
    B, N, _ = x.shape
    padding = (window_size - N % window_size) % window_size
    if key_padding_mask is not None:
        key_padding_mask = F.pad(key_padding_mask, (0, padding), value=float('-inf'))
    x = F.pad(x, (0, 0, 0, padding), value=0)
    return x, key_padding_mask


class TokenMerging(nn.Module):
    """
    Merge tokens from a [batch_size, sequence_length, num_channels] tensor
    using a linear projection.

    This function also updates masks and adds padding as needed to make the
    sequence length divisible by the window size before merging tokens.

    Args:
        dim: Number of input channels.
        window: The size of the window to merge into a single token.
    """

    def __init__(self, dim: 'int', window: 'int'=2) ->None:
        super().__init__()
        self.dim = dim
        self.reduction = linear_layer.LinearLayer(window * dim, dim, bias=False)
        self.norm = layer_norm.LayerNorm(dim)
        self.window = window

    def forward(self, x: 'torch.Tensor', key_padding_mask: 'torch.Tensor') ->Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform token merging.

        Args:
            x: A tensor of shape [batch_size, sequence_length, num_channels].
            key_padding_mask: A tensor of shape [batch_size, sequence_length]
                with "-inf" values at mask tokens, and "0" values at unmasked
                tokens.

        Returns:
            A tensor of shape [batch_size, math.ceil(sequence_length /
                self.window), num_channels], where @self.window is the window
                size.
        """
        if key_padding_mask is not None:
            x[key_padding_mask == float('-inf')] = 0
        x, key_padding_mask = pad_x_and_mask(x, key_padding_mask, self.window)
        B, N, C = x.shape
        x = x.unfold(1, self.window, self.window)
        x = x.reshape(B, N // self.window, C * self.window)
        x = self.reduction(x)
        x = self.norm(x)
        if key_padding_mask is not None:
            assert key_padding_mask.shape == (B, N)
            key_padding_mask = key_padding_mask.unfold(1, self.window, self.window)
            key_padding_mask = key_padding_mask.max(dim=-1).values
        return x, key_padding_mask

    def extra_repr(self) ->str:
        return f'dim={self.dim}, window={self.window}'


class UpSample(nn.Upsample):
    """
    This layer upsamples a given input tensor.

    Args:
        size (Optional[Union[int, Tuple[int, ...]]): Output spatial size. Default: None
        scale_factor (Optional[float]): Scale each spatial dimension of the input by this factor. Default: None
        mode (Optional[str]): Upsampling algorithm (``'nearest'``, ``'linear'``, ``'bilinear'``, ``'bicubic'`` and ``'trilinear'``. Default: ``'nearest'``
        align_corners (Optional[bool]): if ``True``, the corner pixels of the input and output tensors are aligned, and thus preserving the values at
            those pixels. This only has effect when :attr:`mode` is ``'linear'``, ``'bilinear'``, ``'bicubic'``, or ``'trilinear'``.
            Default: ``None``

    Shape:
        - Input: :math:`(N, C, W_{in})` or :math:`(N, C, H_{in}, W_{in})` or :math:`(N, C, D_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, C, W_{out})` or :math:`(N, C, H_{out}, W_{out})` or :math:`(N, C, D_{out}, H_{out}, W_{out})`
    """

    def __init__(self, size: 'Optional[Union[int, Tuple[int, ...]]]'=None, scale_factor: 'Optional[float]'=None, mode: 'Optional[str]'='nearest', align_corners: 'Optional[bool]'=None, *args, **kwargs) ->None:
        super().__init__(size=size, scale_factor=scale_factor, mode=mode, align_corners=align_corners)


class JsonValidator:

    def __init__(self, expected_type: 'type'):
        """
        JsonValidator(T) is function (s)->x that parses json string s into python value x, where x is of type T.

        Example Usage:
        >>> from typing import Union, List
        >>> import argparse
        >>> parser = argparse.ArgumentParser()
        >>> parser.add_argument("--x", type=JsonValidator(Union[int, List[float]]))
        >>> assert parser.parse_args(["--x=123"]).x == 123
        >>> assert parser.parse_args(["--x=[1, 2]"]).x == [1., 2.]
        """
        self.expected_type = expected_type

    @classmethod
    def _validate_and_cast(cls, json_value: 'Any', expected_type: 'Any'):
        type_cls = typing.get_origin(expected_type) or expected_type
        type_args = typing.get_args(expected_type)
        if type_cls is typing.Any:
            return json_value
        if type_cls is float and isinstance(json_value, (int, float)):
            return float(json_value)
        elif type_cls in (int, str, bool) and isinstance(json_value, type_cls):
            return json_value
        elif type_cls is None and json_value is None:
            return None
        elif type_cls is typing.Union:
            for arg in type_args:
                try:
                    return cls._validate_and_cast(json_value, arg)
                except TypeError:
                    continue
        elif type_cls is dict and isinstance(json_value, dict):
            if not type_args:
                type_args = Any, Any
            type_key, type_value = type_args
            return {cls._validate_and_cast(key, type_key): cls._validate_and_cast(value, type_value) for key, value in json_value.items()}
        elif type_cls is list and isinstance(json_value, list):
            if not type_args:
                type_args = [Any]
            return [cls._validate_and_cast(x, type_args[0]) for x in json_value]
        elif type_cls is tuple and isinstance(json_value, list) and (type_args is None or len(type_args) == len(json_value)):
            if type_args is None:
                type_args = [Any] * len(json_value)
            return tuple(type_cls(cls._validate_and_cast(item, type_arg) for item, type_arg in zip(json_value, type_args)))
        raise TypeError(f'Cannot cast {json_value} with type {type(json_value)} to {expected_type}')

    def __call__(self, str_value: 'str') ->Any:
        try:
            value = json.loads(str_value)
        except json.JSONDecodeError:
            raise TypeError(f"Cannot parse json value '{str_value}' for {self}")
        return self._validate_and_cast(value, self.expected_type)

    def __repr__(self):
        return f'JSON[{self.expected_type}]'


NORM_LAYER_CLS = []


norm_layers_tuple = tuple(NORM_LAYER_CLS)


def unwrap_model_fn(model: 'torch.nn.Module') ->torch.nn.Module:
    """Helper function to unwrap the model.

    Args:
        model: An instance of torch.nn.Module.

    Returns:
        Unwrapped instance of torch.nn.Module.
    """
    unwrapped_model = model
    while True:
        if hasattr(unwrapped_model, 'module'):
            unwrapped_model = unwrapped_model.module
        elif hasattr(unwrapped_model, '_fsdp_wrapped_module'):
            unwrapped_model = unwrapped_model._fsdp_wrapped_module
        else:
            break
    return unwrapped_model


def check_frozen_norm_layer(model: 'torch.nn.Module') ->Tuple[bool, int]:
    unwrapped_model = unwrap_model_fn(model)
    count_norm = 0
    frozen_state = False
    for m in unwrapped_model.modules():
        if isinstance(m, norm_layers_tuple):
            frozen_state = m.weight.requires_grad
    return frozen_state, count_norm


def get_tensor_sizes(data: 'Union[Dict, Tensor]') ->Union[List[str], List[Tuple[int]]]:
    """Utility function for extracting tensor shapes (for printing purposes only)."""
    if isinstance(data, Dict):
        tensor_sizes = []
        for k, v in data.items():
            size_ = get_tensor_sizes(v)
            if size_:
                tensor_sizes.append(f'{k}: {size_}')
        return tensor_sizes
    elif isinstance(data, Tensor):
        return [*data.shape]
    else:
        return []


supported_conv_inits = ['kaiming_normal', 'kaiming_uniform', 'xavier_normal', 'xavier_uniform', 'normal', 'trunc_normal']


def _init_nn_layers(module, init_method: 'Optional[str]'='kaiming_normal', std_val: 'Optional[float]'=None) ->None:
    """
    Helper function to initialize neural network module
    """
    init_method = init_method.lower()
    if init_method == 'kaiming_normal':
        if module.weight is not None:
            nn.init.kaiming_normal_(module.weight, mode='fan_out')
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif init_method == 'kaiming_uniform':
        if module.weight is not None:
            nn.init.kaiming_uniform_(module.weight, mode='fan_out')
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif init_method == 'xavier_normal':
        if module.weight is not None:
            nn.init.xavier_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif init_method == 'xavier_uniform':
        if module.weight is not None:
            nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif init_method == 'normal':
        if module.weight is not None:
            std = 1.0 / module.weight.size(1) ** 0.5 if std_val is None else std_val
            nn.init.normal_(module.weight, mean=0.0, std=std)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif init_method == 'trunc_normal':
        if module.weight is not None:
            std = 1.0 / module.weight.size(1) ** 0.5 if std_val is None else std_val
            nn.init.trunc_normal_(module.weight, mean=0.0, std=std)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    else:
        supported_conv_message = 'Supported initialization methods are:'
        for i, l in enumerate(supported_conv_inits):
            supported_conv_message += '\n \t {}) {}'.format(i, l)
        logger.error('{} \n Got: {}'.format(supported_conv_message, init_method))


def initialize_conv_layer(module, init_method: 'Optional[str]'='kaiming_normal', std_val: 'Optional[float]'=0.01) ->None:
    """Helper function to initialize convolution layers"""
    _init_nn_layers(module=module, init_method=init_method, std_val=std_val)


def initialize_fc_layer(module, init_method: 'Optional[str]'='normal', std_val: 'Optional[float]'=0.01) ->None:
    """Helper function to initialize fully-connected layers"""
    if hasattr(module, 'layer'):
        _init_nn_layers(module=module.layer, init_method=init_method, std_val=std_val)
    else:
        _init_nn_layers(module=module, init_method=init_method, std_val=std_val)


def initialize_norm_layers(module) ->None:
    """Helper function to initialize normalization layers"""

    def _init_fn(module):
        if hasattr(module, 'weight') and module.weight is not None:
            nn.init.ones_(module.weight)
        if hasattr(module, 'bias') and module.bias is not None:
            nn.init.zeros_(module.bias)
    _init_fn(module.layer) if hasattr(module, 'layer') else _init_fn(module=module)


def initialize_weights(opts, modules) ->None:
    """Helper function to initialize differnet layers in a model"""
    conv_init_type = getattr(opts, 'model.layer.conv_init', 'kaiming_normal')
    linear_init_type = getattr(opts, 'model.layer.linear_init', 'normal')
    conv_std = getattr(opts, 'model.layer.conv_init_std_dev', None)
    linear_std = getattr(opts, 'model.layer.linear_init_std_dev', 0.01)
    group_linear_std = getattr(opts, 'model.layer.group_linear_init_std_dev', 0.01)
    if isinstance(modules, nn.Sequential):
        for m in modules:
            if isinstance(m, (nn.Conv2d, nn.Conv3d)):
                initialize_conv_layer(module=m, init_method=conv_init_type, std_val=conv_std)
            elif isinstance(m, norm_layers_tuple):
                initialize_norm_layers(module=m)
            elif isinstance(m, (nn.Linear, LinearLayer)):
                initialize_fc_layer(module=m, init_method=linear_init_type, std_val=linear_std)
            elif isinstance(m, GroupLinear):
                initialize_fc_layer(module=m, init_method=linear_init_type, std_val=group_linear_std)
    elif isinstance(modules, (nn.Conv2d, nn.Conv3d)):
        initialize_conv_layer(module=modules, init_method=conv_init_type, std_val=conv_std)
    elif isinstance(modules, norm_layers_tuple):
        initialize_norm_layers(module=modules)
    elif isinstance(modules, (nn.Linear, LinearLayer)):
        initialize_fc_layer(module=modules, init_method=linear_init_type, std_val=linear_std)
    elif isinstance(modules, GroupLinear):
        initialize_fc_layer(module=modules, init_method=linear_init_type, std_val=group_linear_std)


class UniformSampler(nn.Module):

    def __init__(self, low: 'float', high: 'float', min_fn: 'Optional[nn.Module]'=Identity(), max_fn: 'Optional[nn.Module]'=Identity(), *args, **kwargs):
        super().__init__()
        self._low = nn.Parameter(torch.tensor(low, dtype=torch.float))
        self._high = nn.Parameter(torch.tensor(high, dtype=torch.float))
        self.min_fn = min_fn
        self.max_fn = max_fn

    def forward(self, sample_shape=(), data_type=torch.float, device=torch.device('cpu')) ->Tensor:
        rand_tensor = torch.rand(sample_shape, dtype=data_type, device=device)
        return self.low + rand_tensor * (self.high - self.low)

    @property
    def high(self):
        return self.max_fn(self._high)

    @property
    def low(self):
        return self.min_fn(self._low)

    def __repr__(self):
        return '{}(min_fn={}, max_fn={})'.format(self.__class__.__name__, self.min_fn, self.max_fn)


_distribution_tuple = UniformSampler,


def random_brightness(x: 'Tensor', magnitude: 'Tensor', *args, **kwargs) ->Tensor:
    """
    Brightness function.
    """
    x = x * magnitude
    return x


def random_contrast(x: 'Tensor', magnitude: 'Tensor', *args, **kwargs) ->Tensor:
    per_channel_mean = torch.mean(x, dim=[-1, -2], keepdim=True)
    x = (1.0 - magnitude) * per_channel_mean + x * magnitude
    return x


def random_noise(x: 'Tensor', variance: 'Tensor', *args, **kwargs) ->Tensor:
    """Apply random noise sampled."""
    noise = torch.randn_like(x) * variance
    x = x + noise
    return x


class BaseNeuralAugmentor(nn.Module):
    """
    Base class for `neural (or range) augmentation <https://arxiv.org/abs/2212.10553>`_
    """

    def __init__(self, opts, *args, **kwargs):
        super().__init__()
        self.opts = opts
        self.lr_multiplier = getattr(opts, 'model.learn_augmentation.lr_multiplier', 1.0)
        self.brightness = None
        self.contrast = None
        self.noise = None
        self.aug_fns = []

    def _is_valid_aug_fn_list(self, aug_fns):
        if self.training:
            if len(aug_fns) == 0:
                logger.error('{} needs at least one learnable function.'.format(self.__class__.__name__))

    def get_trainable_parameters(self, weight_decay: 'Optional[float]'=0.0, no_decay_bn_filter_bias: 'Optional[bool]'=False, *args, **kwargs):
        """Get trainable parameters"""
        param_list = parameter_list(named_parameters=self.named_parameters, weight_decay=weight_decay, no_decay_bn_filter_bias=no_decay_bn_filter_bias)
        return param_list, [self.lr_multiplier] * len(param_list)

    def __repr__(self):
        aug_str = '{}('.format(self.__class__.__name__)
        if self.brightness is not None:
            aug_str += '\n\tBrightness={}, '.format(self.brightness.data.shape if isinstance(self.brightness, nn.Parameter) else self.brightness)
        if self.contrast is not None:
            aug_str += '\n\tContrast={}, '.format(self.contrast.data.shape if isinstance(self.contrast, nn.Parameter) else self.contrast)
        if self.noise is not None:
            aug_str += '\n\tNoise={}, '.format(self.noise.data.shape if isinstance(self.noise, nn.Parameter) else self.noise)
        aug_str += self.extra_repr()
        aug_str += ')'
        return aug_str

    @classmethod
    def add_arguments(cls, parser: 'argparse.ArgumentParser'):
        """Add model-specific arguments"""
        group = parser.add_argument_group(title=cls.__name__)
        group.add_argument('--model.learn-augmentation.mode', type=str, default=None, choices=['basic', 'distribution'], help='Neural augmentation mode')
        group.add_argument('--model.learn-augmentation.brightness', action='store_true', help='Learn parameters for brightness')
        group.add_argument('--model.learn-augmentation.contrast', action='store_true', help='Learn parameters for contrast')
        group.add_argument('--model.learn-augmentation.noise', action='store_true', help='Learn parameters for noise')
        group.add_argument('--model.learn-augmentation.lr-multiplier', type=float, default=1.0, help='LR multiplier for neural aug parameters')
        return parser

    def _build_aug_fns(self, opts) ->List:
        raise NotImplementedError

    def _apply_brightness(self, x: 'Tensor', *args, **kwargs) ->Tensor:
        """
        Apply brightness augmentation function with learnable parameters.
        """
        x_shape = [*x.shape]
        x_shape[1:] = [1] * (len(x_shape) - 1)
        if isinstance(self.brightness, nn.Parameter):
            magnitude = self.brightness
        elif isinstance(self.brightness, _distribution_tuple):
            magnitude = self.brightness(x_shape, device=x.device, data_type=x.dtype)
        else:
            raise NotImplementedError
        return random_brightness(x, magnitude, *args, **kwargs)

    def _apply_contrast(self, x: 'Tensor', *args, **kwargs) ->Tensor:
        """
        Apply contrast augmentation function with learnable parameters.
        """
        x_shape = [*x.shape]
        x_shape[1:] = [1] * (len(x_shape) - 1)
        if isinstance(self.contrast, nn.Parameter):
            magnitude = self.contrast
        elif isinstance(self.contrast, _distribution_tuple):
            magnitude = self.contrast(x_shape, device=x.device, data_type=x.dtype)
        else:
            raise NotImplementedError
        return random_contrast(x, magnitude, *args, *kwargs)

    def _apply_noise(self, x: 'Tensor', *args, **kwargs) ->Tensor:
        x_shape = [*x.shape]
        x_shape[1:] = [1] * (len(x_shape) - 1)
        if isinstance(self.noise, nn.Parameter):
            variance = self.noise
        elif isinstance(self.noise, _distribution_tuple):
            variance = self.noise(x_shape, device=x.device, data_type=x.dtype)
        else:
            raise NotImplementedError
        return random_noise(x, variance, *args, *kwargs)

    def forward(self, x: 'Tensor', *args, **kwargs) ->Tensor:
        batch_size, in_channels, in_height, in_width = x.shape
        n_aug_samples = max(1, batch_size // 2)
        random.shuffle(self.aug_fns)
        for aug_fn in self.aug_fns:
            sample_ids = torch.randperm(n=batch_size, dtype=torch.long, device=x.device)[:n_aug_samples]
            x_aug = torch.index_select(x, dim=0, index=sample_ids)
            x_aug = aug_fn(x=x_aug)
            x = torch.index_copy(x, dim=0, source=x_aug, index=sample_ids)
        x = torch.clip(x, min=0.0, max=1.0)
        return x


class Clip(nn.Module):

    def __init__(self, min_val: 'float', max_val: 'float', hard_clip: 'Optional[bool]'=False, *args, **kwargs) ->None:
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val
        self.hard_clip = hard_clip

    def forward(self, x: 'Any') ->Any:
        if self.hard_clip:
            with torch.no_grad():
                return x.clamp_(min=self.min_val, max=self.max_val)
        else:
            return torch.sigmoid(x) * (self.max_val - self.min_val) + self.min_val

    def __repr__(self):
        return '{}(min={}, max={}, clipping={})'.format(self.__class__.__name__, self.min_val, self.max_val, 'hard' if self.hard_clip else 'soft')


class FixedSampler(nn.Module):

    def __init__(self, value: 'float', clip_fn: 'Optional[nn.Module]'=Identity(), *args, **kwargs):
        super().__init__()
        self._value = nn.Parameter(torch.FloatTensor(1, 3, 1, 1).fill_(value))
        self.clip_fn = clip_fn

    def forward(self, sample_shape=(), data_type=torch.float, device=torch.device('cpu')) ->Tensor:
        return self.clip_fn(self._value)

    def __repr__(self):
        return '{}(clip_fn={})'.format(self.__class__.__name__, self.clip_fn)


class BasicNeuralAugmentor(BaseNeuralAugmentor):
    """
    Basic neural augmentation. This class learns per-channel augmentation parameters
    and apply the same parameter to all images in a batch.

    See `neural (or range) augmentation <https://arxiv.org/abs/2212.10553>`_ paper for details.
    """

    def __init__(self, opts, *args, **kwargs) ->None:
        super().__init__(*args, opts=opts, **kwargs)
        aug_fns = self._build_aug_fns(opts=opts)
        self._is_valid_aug_fn_list(aug_fns)
        self.aug_fns = aug_fns

    def _build_aug_fns(self, opts) ->List:
        aug_fns = []
        if getattr(opts, 'model.learn_augmentation.brightness', False):
            self.brightness = FixedSampler(value=1.0, clip_fn=Clip(min_val=0.1, max_val=10.0))
            aug_fns.append(self._apply_brightness)
        if getattr(opts, 'model.learn_augmentation.contrast', False):
            self.contrast = FixedSampler(value=1.0, clip_fn=Clip(min_val=0.1, max_val=10.0))
            aug_fns.append(self._apply_contrast)
        if getattr(opts, 'model.learn_augmentation.noise', False):
            self.noise = FixedSampler(value=0.0, clip_fn=Clip(min_val=0.0, max_val=1.0))
            aug_fns.append(self._apply_noise)
        return aug_fns


class DistributionNeuralAugmentor(BaseNeuralAugmentor):
    """
    Distribution-based neural (or range) augmentation. This class samples the augmentation parameters
    from a specified distribution with learnable range.

    See `neural (or range) augmentation <https://arxiv.org/abs/2212.10553>`_ paper for details.
    """

    def __init__(self, opts, *args, **kwargs) ->None:
        super().__init__(*args, opts=opts, **kwargs)
        aug_fns = self._build_aug_fns_with_uniform_dist(opts=opts)
        self._is_valid_aug_fn_list(aug_fns)
        self.aug_fns = aug_fns

    def _build_aug_fns_with_uniform_dist(self, opts) ->List:
        aug_fns = []
        if getattr(opts, 'model.learn_augmentation.brightness', False):
            self.brightness = UniformSampler(low=0.5, high=1.5, min_fn=Clip(min_val=0.1, max_val=0.9), max_fn=Clip(min_val=1.1, max_val=10.0))
            aug_fns.append(self._apply_brightness)
        if getattr(opts, 'model.learn_augmentation.contrast', False):
            self.contrast = UniformSampler(low=0.5, high=1.5, min_fn=Clip(min_val=0.1, max_val=0.9), max_fn=Clip(min_val=1.1, max_val=10.0))
            aug_fns.append(self._apply_contrast)
        if getattr(opts, 'model.learn_augmentation.noise', False):
            self.noise = UniformSampler(low=0.0, high=0.1, min_fn=Clip(min_val=0.0, max_val=5e-05), max_fn=Clip(min_val=0.0001, max_val=1.0))
            aug_fns.append(self._apply_noise)
        return aug_fns


def build_neural_augmentor(opts, *args, **kwargs):
    mode = getattr(opts, 'model.learn_augmentation.mode', None)
    if mode is None:
        mode = 'none'
    mode = mode.lower()
    if mode == 'distribution':
        return DistributionNeuralAugmentor(*args, opts=opts, **kwargs)
    elif mode == 'basic':
        return BasicNeuralAugmentor(*args, opts=opts, **kwargs)
    else:
        return None


def is_test_env() ->bool:
    return 'PYTEST_CURRENT_TEST' in os.environ


def set_model_specific_opts_before_model_building(opts: 'argparse.Namespace') ->Dict[str, Any]:
    """Override library-level defaults with model-specific default values.

    Args:
        opts: Command-line arguments

    Returns:
        A dictionary containing the name of arguments that are updated along with their original values.
        This dictionary is used in `unset_model_specific_opts_after_model_building` function to unset the
        model-specific to library-specific defaults.
    """
    seg_act_fn = getattr(opts, 'model.segmentation.activation.name')
    if seg_act_fn is not None:
        default_act_fn = getattr(opts, 'model.activation.name', 'relu')
        default_act_inplace = getattr(opts, 'model.activation.inplace', False)
        default_act_neg_slope = getattr(opts, 'model.activation.neg_slope', 0.1)
        setattr(opts, 'model.activation.name', seg_act_fn)
        setattr(opts, 'model.activation.inplace', getattr(opts, 'model.segmentation.activation.inplace', False))
        setattr(opts, 'model.activation.neg_slope', getattr(opts, 'model.segmentation.activation.neg_slope', 0.1))
        return {'model.activation.name': default_act_fn, 'model.activation.inplace': default_act_inplace, 'model.activation.neg_slope': default_act_neg_slope}
    return {}


def unset_model_specific_opts_after_model_building(opts: 'argparse.Namespace', default_opts_info: 'Dict[str, Any]', *ars, **kwargs) ->None:
    """Given command-line arguments and a mapping of opts that needs to be unset, this function
    unsets the library-level defaults that were over-ridden previously
    in `set_model_specific_opts_before_model_building`.
    """
    assert isinstance(default_opts_info, dict), f'Please ensure set_model_specific_opts_before_model_building() returns a dict.'
    for k, v in default_opts_info.items():
        setattr(opts, k, v)


def window_partition_reverse(t: 'torch.Tensor', B: 'int', num_windows: 'int', C: 'int') ->torch.Tensor:
    """
    Undo the @window_partition operation.

    Args:
        t: The input tensor of shape [batch_size * num_windows, window_size,
            embed_dim].
        B: The batch size.
        num_windows: The number of windows.
        C: The embedding dimension.

    Returns:
        A tensor of shape [batch_size, num_windows * window_size, embed_dim].
    """
    t = t.reshape(B, num_windows * t.shape[1], C)
    return t


def unwindow_x(x_windows: 'torch.Tensor', B: 'int', N: 'int', C: 'int', window_shift: 'int'):
    """
    Undoes the operation of @window_x_and_attention on the input tensor @x_windows.

    Args:
        x_windows: The input tensor to unwindow. Its shape is [batch_size *
              padded_sequence_length // window_size, window_size, embed_dim].
        B: The batch size. Referred to as batch_size in this docstring.
        N: The sequence length of the tensor before windowing. Referred to as
            sequence_length in this docstring.
        C: The number of channels. Referred to as embed_dim in this docstring.
        window_shift: The shift applied to the sequence before the windowing
            originally occurred.

    Returns:
        A tensor of shape [batch_size, sequence_length, embed_dim].
    """
    num_windows = x_windows.shape[0] // B
    x = window_partition_reverse(x_windows, B, num_windows, C)
    if window_shift > 0:
        x = torch.roll(x, shifts=window_shift, dims=1)
    x = x[:, :N]
    return x


def get_windows_shift_mask(N: 'int', window_size: 'int', window_shift: 'int', device: 'torch.device') ->torch.Tensor:
    """
    Get the mask window required due to window shifting (needed for shifted
    window attention).

    This produces a tensor with mask values for each window. Most windows don't
    require masking, but windows that bleed across the beginning/end of the
    tensor (due to shifting) require it.

    Args:
        N: The sequence length.
        window_size: The window size.
        window_shift: The window shift.
        device: The device on which to create the tensor.

    Returns:
        A tensor of shape [N // window_size, window_size, window_size]
        containing mask values. The values are 0 (unmasked) or float("-inf")
        (masked).
    """
    ret = torch.zeros(N // window_size, window_size, window_size, device=device)
    ret[-1].fill_(float('-inf'))
    ret[-1, :window_size - window_shift, :window_size - window_shift] = 0
    ret[-1, -window_shift:, -window_shift:] = 0
    return ret


def window_partition(t: 'torch.Tensor', window_size: 'int') ->torch.Tensor:
    """
    Partition tensor @t into chunks of size @window_size.

    @t's sequence length must be divisible by @window_size.

    Args:
        t: A tensor of shape [batch_size, sequence_length, embed_dim].
        window_size: The desired window size.

    Returns:
        A tensor of shape [batch_size * sequence_length // window_size,
        window_size, embed_dim].
    """
    B, N, C = t.shape
    if not N % window_size == 0:
        raise ValueError(f'sequence length {N} must be divisible by window size {window_size}')
    t = t.reshape(B * N // window_size, window_size, C)
    return t


def window_x_and_key_padding_mask(x: 'torch.Tensor', key_padding_mask: 'torch.Tensor', window_size: 'int', window_shift: 'int') ->Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Perform windowing on @x and @key_padding_mask in preparation for windowed
    attention.

    Args:
        x: The input tensor of shape [batch_size, sequence_length, num_channels].
        key_padding_mask: The mask, as a tensor of shape [batch_size, sequence_length].
        window_size: The window size to be used for windowed attention.
        window_shift: The window shift to be used for windowed attention.

    Returns:
        A tuple containing 3 tensors. The first is the windowed input. The second
        is the windowed mask. The third is the mask needed to perform shifted
        window attention (to avoid the first and last windows from bleeding
        into each other).
    """
    B, N = key_padding_mask.shape
    assert x.shape[:2] == (B, N)
    x, key_padding_mask = token_merging.pad_x_and_mask(x, key_padding_mask, window_size)
    if window_shift > 0:
        x = torch.roll(x, shifts=-window_shift, dims=1)
        key_padding_mask = torch.roll(key_padding_mask, shifts=-window_shift, dims=1)
    x_windows = window_partition(x, window_size)
    token_mask_windows = key_padding_mask.reshape(B * x.shape[1] // window_size, window_size)
    window_mask = get_windows_shift_mask(x.shape[1], window_size, window_shift, x_windows.device).expand(B, -1, -1, -1)
    window_mask = window_mask.reshape(window_mask.shape[0] * window_mask.shape[1], window_mask.shape[2], window_mask.shape[3])
    return x_windows, token_mask_windows, window_mask


def get_configuration(opts: 'argparse.Namespace') ->Dict:
    """
    Get configuration parameters associated with ByteFormer.

    These parameters are similar to those of DeIT
    (https://arxiv.org/pdf/2012.12877.pdf).

    Args:
        opts: The options configuration.

    Returns:
        A dict with keys specifying the parameters needed for ByteFormer.
    """
    mode = getattr(opts, 'model.classification.byteformer.mode')
    mode = mode.lower()
    dropout = getattr(opts, 'model.classification.byteformer.dropout')
    norm_layer = getattr(opts, 'model.classification.byteformer.norm_layer')
    byteformer_config = dict()
    if mode == 'tiny':
        byteformer_config = {'embed_dim': 192, 'n_transformer_layers': 12, 'n_attn_heads': 3, 'ffn_dim': 192 * 4, 'norm_layer': norm_layer, 'pos_emb_drop_p': 0.1, 'attn_dropout': 0.0, 'ffn_dropout': 0.0, 'dropout': dropout}
    elif mode == 'small':
        byteformer_config = {'embed_dim': 384, 'n_transformer_layers': 12, 'n_attn_heads': 6, 'ffn_dim': 384 * 4, 'norm_layer': norm_layer, 'pos_emb_drop_p': 0.0, 'attn_dropout': 0.0, 'ffn_dropout': 0.0, 'dropout': dropout}
    elif mode == 'base':
        byteformer_config = {'embed_dim': 768, 'n_transformer_layers': 12, 'n_attn_heads': 12, 'ffn_dim': 768 * 4, 'norm_layer': norm_layer, 'pos_emb_drop_p': 0.0, 'attn_dropout': 0.0, 'ffn_dropout': 0.0, 'dropout': dropout}
    elif mode == 'huge':
        byteformer_config = {'embed_dim': 1280, 'n_transformer_layers': 32, 'n_attn_heads': 20, 'ffn_dim': 1280 * 4, 'norm_layer': norm_layer, 'pos_emb_drop_p': 0.0, 'attn_dropout': 0.0, 'ffn_dropout': 0.0, 'dropout': dropout}
    else:
        logger.error('Got unsupported ByteFormer configuration: {}'.format(mode))
    return byteformer_config


def unfold_tokens(t: 'Tensor', kernel_size: 'int') ->Tensor:
    """
    Group tokens from tensor @t using torch.Tensor.unfold, using the given
    kernel size. This amounts to windowing @t using overlapping windows
    of size @kernel_size, with overlap of @kernel_size // 2.

    Args:
        t: A tensor of shape [batch_size, sequence_length, num_channels].
        kernel_size: The kernel size.

    Returns:
        A tensor of shape [batch_size * (sequence_length - kernel_size)
        // (kernel_size // 2) + 1, kernel_size, num_channels].
    """
    t = t.unfold(dimension=1, size=kernel_size, step=kernel_size // 2)
    B, L, C, _ = t.shape
    t = t.reshape(B * L, C, kernel_size)
    t = t.transpose(1, 2)
    return t


class BaseModule(nn.Module):
    """Base class for all modules"""

    def __init__(self, *args, **kwargs):
        super(BaseModule, self).__init__()

    def forward(self, x: 'Any', *args, **kwargs) ->Any:
        raise NotImplementedError

    def __repr__(self):
        return '{}'.format(self.__class__.__name__)


def make_divisible(v: 'Union[float, int]', divisor: 'Optional[int]'=8, min_value: 'Optional[Union[float, int]]'=None) ->Union[float, int]:
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class SqueezeExcitation(BaseModule):
    """
    This class defines the Squeeze-excitation module, in the `SENet paper <https://arxiv.org/abs/1709.01507>`_

    Args:
        opts: command-line arguments
        in_channels (int): :math:`C` from an expected input of size :math:`(N, C, H, W)`
        squeeze_factor (Optional[int]): Reduce :math:`C` by this factor. Default: 4
        squeeze_channels (Optional[int]): This module's output channels. Overrides squeeze_factor if specified
        scale_fn_name (Optional[str]): Scaling function name. Default: sigmoid

    Shape:
        - Input: :math:`(N, C, H, W)`
        - Output: :math:`(N, C, H, W)`
    """

    def __init__(self, opts, in_channels: 'int', squeeze_factor: 'Optional[int]'=4, squeeze_channels: 'Optional[int]'=None, scale_fn_name: 'Optional[str]'='sigmoid', *args, **kwargs) ->None:
        if squeeze_channels is None:
            squeeze_channels = max(make_divisible(in_channels // squeeze_factor, 8), 32)
        fc1 = ConvLayer2d(opts=opts, in_channels=in_channels, out_channels=squeeze_channels, kernel_size=1, stride=1, bias=True, use_norm=False, use_act=True)
        fc2 = ConvLayer2d(opts=opts, in_channels=squeeze_channels, out_channels=in_channels, kernel_size=1, stride=1, bias=True, use_norm=False, use_act=False)
        act_fn = build_activation_layer(opts, act_type=scale_fn_name, inplace=True)
        super().__init__()
        self.se_layer = nn.Sequential()
        self.se_layer.add_module(name='global_pool', module=AdaptiveAvgPool2d(output_size=1))
        self.se_layer.add_module(name='fc1', module=fc1)
        self.se_layer.add_module(name='fc2', module=fc2)
        self.se_layer.add_module(name='scale_act', module=act_fn)
        self.in_channels = in_channels
        self.squeeze_factor = squeeze_factor
        self.scale_fn = scale_fn_name

    def forward(self, x: 'Tensor', *args, **kwargs) ->Tensor:
        return x * self.se_layer(x)

    def __repr__(self) ->str:
        return '{}(in_channels={}, squeeze_factor={}, scale_fn={})'.format(self.__class__.__name__, self.in_channels, self.squeeze_factor, self.scale_fn)


class InvertedResidualSE(BaseModule):
    """
    This class implements the inverted residual block with squeeze-excitation unit, as described in
    `MobileNetv3 <https://arxiv.org/abs/1905.02244>`_ paper

    Args:
        opts: command-line arguments
        in_channels (int): :math:`C_{in}` from an expected input of size :math:`(N, C_{in}, H_{in}, W_{in})`
        out_channels (int): :math:`C_{out}` from an expected output of size :math:`(N, C_{out}, H_{out}, W_{out)`
        expand_ratio (Union[int, float]): Expand the input channels by this factor in depth-wise conv
        dilation (Optional[int]): Use conv with dilation. Default: 1
        stride (Optional[int]): Use convolutions with a stride. Default: 1
        use_se (Optional[bool]): Use squeeze-excitation block. Default: False
        act_fn_name (Optional[str]): Activation function name. Default: relu
        se_scale_fn_name (Optional [str]): Scale activation function inside SE unit. Defaults to hard_sigmoid
        kernel_size (Optional[int]): Kernel size in depth-wise convolution. Defaults to 3.
        squeeze_factor (Optional[bool]): Squeezing factor in SE unit. Defaults to 4.

    Shape:
        - Input: :math:`(N, C_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, C_{out}, H_{out}, W_{out})`
    """

    def __init__(self, opts, in_channels: 'int', out_channels: 'int', expand_ratio: 'Union[int, float]', dilation: 'Optional[int]'=1, stride: 'Optional[int]'=1, use_se: 'Optional[bool]'=False, act_fn_name: 'Optional[str]'='relu', se_scale_fn_name: 'Optional[str]'='hard_sigmoid', kernel_size: 'Optional[int]'=3, squeeze_factor: 'Optional[int]'=4, *args, **kwargs) ->None:
        hidden_dim = make_divisible(int(round(in_channels * expand_ratio)), 8)
        act_fn = build_activation_layer(opts, act_type=act_fn_name, inplace=True)
        super().__init__()
        block = nn.Sequential()
        if expand_ratio != 1:
            block.add_module(name='exp_1x1', module=ConvLayer2d(opts, in_channels=in_channels, out_channels=hidden_dim, kernel_size=1, use_act=False, use_norm=True))
            block.add_module(name='act_fn_1', module=act_fn)
        block.add_module(name='conv_3x3', module=ConvLayer2d(opts, in_channels=hidden_dim, out_channels=hidden_dim, stride=stride, kernel_size=kernel_size, groups=hidden_dim, use_act=False, use_norm=True, dilation=dilation))
        block.add_module(name='act_fn_2', module=act_fn)
        if use_se:
            se = SqueezeExcitation(opts=opts, in_channels=hidden_dim, squeeze_factor=squeeze_factor, scale_fn_name=se_scale_fn_name)
            block.add_module(name='se', module=se)
        block.add_module(name='red_1x1', module=ConvLayer2d(opts, in_channels=hidden_dim, out_channels=out_channels, kernel_size=1, use_act=False, use_norm=True))
        self.block = block
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.exp = expand_ratio
        self.dilation = dilation
        self.use_se = use_se
        self.stride = stride
        self.act_fn_name = act_fn_name
        self.kernel_size = kernel_size
        self.use_res_connect = self.stride == 1 and in_channels == out_channels

    def forward(self, x: 'Tensor', *args, **kwargs) ->Tensor:
        y = self.block(x)
        return x + y if self.use_res_connect else y

    def __repr__(self) ->str:
        return '{}(in_channels={}, out_channels={}, stride={}, exp={}, dilation={}, use_se={}, kernel_size={}, act_fn={})'.format(self.__class__.__name__, self.in_channels, self.out_channels, self.stride, self.exp, self.dilation, self.use_se, self.kernel_size, self.act_fn_name)


class StochasticDepth(StochasticDepthTorch):
    """
    Implements the Stochastic Depth `"Deep Networks with Stochastic Depth"
    <https://arxiv.org/abs/1603.09382>`_ used for randomly dropping residual
    branches of residual architectures.
    """

    def __init__(self, p: 'float', mode: 'str') ->None:
        super().__init__(p=p, mode=mode)


class EfficientNetBlock(InvertedResidualSE):
    """
    This class implements a variant of the inverted residual block with squeeze-excitation unit,
    as described in `MobileNetv3 <https://arxiv.org/abs/1905.02244>`_ paper. This variant
    includes stochastic depth, as used in `EfficientNet <https://arxiv.org/abs/1905.11946>`_ paper.

    Args:
        stochastic_depth_prob: float,
        For other arguments, refer to the parent class.

    Shape:
        - Input: :math:`(N, C_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, C_{out}, H_{out}, W_{out})`
    """

    def __init__(self, stochastic_depth_prob: 'float', *args, **kwargs) ->None:
        super().__init__(*args, **kwargs)
        self.stochastic_depth = StochasticDepth(p=stochastic_depth_prob, mode='row')

    def forward(self, x: 'Tensor', *args, **kwargs) ->Tensor:
        y = self.block(x)
        if self.use_res_connect:
            y = self.stochastic_depth(y)
            y = y + x
        return y

    def __repr__(self) ->str:
        return super().__repr__()[:-1] + f', stochastic_depth_prob={self.stochastic_depth.p})'


@dataclass
class EfficientNetBlockConfig:
    """This class stores the config for each block in EfficientNet i.e. MBConv layers
    in Table 1 of `EfficientNet paper <https://arxiv.org/abs/1905.11946>`_
    Notably, this class takes width_mult and depth_mult as input too and adjusts
    layers' depth and width, as is required in different modes of EfficientNet.
    """

    def __init__(self, expand_ratio: 'float', kernel: 'int', stride: 'int', in_channels: 'int', out_channels: 'int', num_layers: 'int', width_mult: 'float', depth_mult: 'float'):
        self.expand_ratio = expand_ratio
        self.kernel = kernel
        self.stride = stride
        self.in_channels = int(make_divisible(in_channels * width_mult, 8))
        self.out_channels = int(make_divisible(out_channels * width_mult, 8))
        self.num_layers = int(math.ceil(num_layers * depth_mult))


class MobileOneBlock(BaseModule):
    """
    MobileOne building block.

    For more details, please refer to our paper:
    `An Improved One millisecond Mobile Backbone <https://arxiv.org/pdf/2206.04040.pdf>`
    """

    def __init__(self, opts: 'argparse.Namespace', in_channels: 'int', out_channels: 'int', kernel_size: 'int', stride: 'int'=1, padding: 'int'=0, dilation: 'int'=1, groups: 'int'=1, inference_mode: 'bool'=False, use_se: 'bool'=False, use_act: 'bool'=True, use_scale_branch: 'bool'=True, num_conv_branches: 'int'=1) ->None:
        """
        Construct a MobileOneBlock.

        Args:
            opts: Command line arguments.
            in_channels: Number of channels in the input.
            out_channels: Number of channels produced by the block.
            kernel_size: Size of the convolution kernel.
            stride: Stride size. Default: 1
            padding: Zero-padding size. Default: 0
            dilation: Kernel dilation factor. Default: 1
            groups: Group number. Default: 1
            inference_mode: If True, instantiates model in inference mode. Default: ``False``
            use_se: Whether to use SE-ReLU activations. Default: ``False``
            use_act: Whether to use activation. Default: ``True``
            use_scale_branch: Whether to use scale branch. Default: ``True``
            num_conv_branches: Number of linear conv branches. Default: 1
        """
        super(MobileOneBlock, self).__init__()
        self.inference_mode = inference_mode
        self.groups = groups
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_conv_branches = num_conv_branches
        if use_se:
            self.se = SqueezeExcitation(opts, out_channels, squeeze_factor=16)
        else:
            self.se = Identity()
        if use_act:
            self.activation = build_activation_layer(opts)
        else:
            self.activation = Identity()
        if inference_mode:
            self.reparam_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=True)
        else:
            self.rbr_skip = BatchNorm2d(num_features=in_channels, affine=True) if out_channels == in_channels and stride == 1 else None
            if num_conv_branches > 0:
                rbr_conv = list()
                for _ in range(self.num_conv_branches):
                    rbr_conv.append(ConvLayer2d(opts, in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=kernel_size, stride=self.stride, padding=padding, groups=self.groups, bias=False, use_act=False))
                self.rbr_conv = nn.ModuleList(rbr_conv)
            else:
                self.rbr_conv = None
            self.rbr_scale = None
            if kernel_size > 1 and use_scale_branch:
                self.rbr_scale = ConvLayer2d(opts, in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=1, stride=self.stride, padding=0, groups=self.groups, bias=False, use_act=False)

    def forward(self, x: 'torch.Tensor', *args, **kwargs) ->torch.Tensor:
        """
        Forward pass implements inference logic for module
        before and after reparameterization.

        Args:
            x: Input tensor of shape :math:`(B, C, H, W)`.

        Returns:
            torch.Tensor of shape :math:`(B, C, H, W)`.
        """
        if self.inference_mode:
            return self.activation(self.se(self.reparam_conv(x)))
        identity_out = 0
        if self.rbr_skip is not None:
            identity_out = self.rbr_skip(x)
        scale_out = 0
        if self.rbr_scale is not None:
            scale_out = self.rbr_scale(x)
        out = scale_out + identity_out
        if self.rbr_conv is not None:
            for ix in range(self.num_conv_branches):
                out += self.rbr_conv[ix](x)
        return self.activation(self.se(out))

    def reparameterize(self) ->None:
        """
        Following works like `RepVGG: Making VGG-style ConvNets Great Again` -
        https://arxiv.org/pdf/2101.03697.pdf. We re-parameterize multi-branched
        architecture used at training time to obtain a plain CNN-like structure
        for inference.
        """
        if self.inference_mode:
            return
        kernel, bias = self._get_kernel_bias()
        self.reparam_conv = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups, bias=True)
        self.reparam_conv.weight.data = kernel
        self.reparam_conv.bias.data = bias
        for para in self.parameters():
            para.detach_()
        if hasattr(self, 'rbr_conv'):
            self.__delattr__('rbr_conv')
        if hasattr(self, 'rbr_scale'):
            self.__delattr__('rbr_scale')
        if hasattr(self, 'rbr_skip'):
            self.__delattr__('rbr_skip')
        self.inference_mode = True

    def _get_kernel_bias(self) ->Tuple[torch.Tensor, torch.Tensor]:
        """
        Method to obtain re-parameterized kernel and bias.
        Reference: https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py#L83

        Returns:
            Tuple of (kernel, bias) after fusing branches.
        """
        kernel_scale = 0
        bias_scale = 0
        if self.rbr_scale is not None:
            kernel_scale, bias_scale = self._fuse_branch_ops(self.rbr_scale.block)
            pad = self.kernel_size // 2
            kernel_scale = torch.nn.functional.pad(kernel_scale, [pad, pad, pad, pad])
        kernel_identity = 0
        bias_identity = 0
        if self.rbr_skip is not None:
            kernel_identity, bias_identity = self._fuse_branch_ops(self.rbr_skip)
        kernel_conv = 0
        bias_conv = 0
        if self.rbr_conv is not None:
            for ix in range(self.num_conv_branches):
                _kernel, _bias = self._fuse_branch_ops(self.rbr_conv[ix].block)
                kernel_conv += _kernel
                bias_conv += _bias
        kernel_final = kernel_conv + kernel_scale + kernel_identity
        bias_final = bias_conv + bias_scale + bias_identity
        return kernel_final, bias_final

    def _fuse_branch_ops(self, branch: 'Union[nn.Sequential, nn.BatchNorm2d]') ->Tuple[torch.Tensor, torch.Tensor]:
        """
        Method to fuse all linear ops in a branch.
        Reference: https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py#L95

        Args:
            branch: Sequence of ops to be fused.

        Returns:
            Tuple of (kernel, bias) after fusing batchnorm.
        """
        if isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight
            return self._fuse_conv_bn(kernel, branch.norm)
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels // self.groups
                kernel_value = torch.zeros((self.in_channels, input_dim, self.kernel_size, self.kernel_size), dtype=branch.weight.dtype, device=branch.weight.device)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, self.kernel_size // 2, self.kernel_size // 2] = 1
                self.id_tensor = kernel_value
            kernel = self.id_tensor
            return self._fuse_conv_bn(kernel, branch)

    @staticmethod
    def _fuse_conv_bn(kernel: 'torch.Tensor', bn: 'nn.BatchNorm2d') ->Tuple[torch.Tensor, torch.Tensor]:
        """
        Method to fuse batchnorm layer with conv layer.

        Args:
            kernel: Convolutional kernel weights.
            bn: Batchnorm 2d layer.

        Returns:
            Tuple of (kernel, bias) after fusing batchnorm.
        """
        assert bn.affine, 'Expected BatchNorm layer to have affine parameters instead got BatchNorm layer without affine parameters.'
        running_mean = bn.running_mean
        running_var = bn.running_var
        gamma = bn.weight
        beta = bn.bias
        eps = bn.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std


class RepLKBlock(BaseModule):
    """
    This class defines overparameterized large kernel conv block in `RepLKNet <https://arxiv.org/abs/2203.06717>`_
    Reference: https://github.com/DingXiaoH/RepLKNet-pytorch

    Args:
        opts: Command-line arguments.
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Kernel size of the large kernel conv branch.
        stride: Stride size. Default: 1
        dilation: Kernel dilation factor. Default: 1
        groups: Group number. Default: 1
        small_kernel_size: Kernel size of small kernel conv branch.
        inference_mode: If True, instantiates model in inference mode. Default: ``False``
        use_act: If True, activation is used. Default: ``True``
    """

    def __init__(self, opts: 'argparse.Namespace', in_channels: 'int', out_channels: 'int', kernel_size: 'int', stride: 'int'=1, dilation: 'int'=1, groups: 'int'=1, small_kernel_size: 'int'=None, inference_mode: 'bool'=False, use_act: 'bool'=True) ->None:
        super().__init__()
        self.stride = stride
        self.groups = groups
        self.dilation = dilation
        self.in_channels = in_channels
        self.out_channels = out_channels
        if use_act:
            self.activation = build_activation_layer(opts)
        else:
            self.activation = Identity()
        self.kernel_size = kernel_size
        self.small_kernel_size = small_kernel_size
        self.padding = kernel_size // 2
        if inference_mode:
            self.lkb_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=self.padding, dilation=self.dilation, groups=groups, bias=True)
        else:
            self.lkb_origin = ConvLayer2d(opts, in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, groups=self.groups, bias=False, use_act=False)
            if small_kernel_size is not None:
                assert small_kernel_size <= kernel_size, 'The kernel size for re-param cannot be larger than the large kernel'
                self.small_conv = ConvLayer2d(opts, in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=self.small_kernel_size, stride=self.stride, padding=self.small_kernel_size // 2, groups=self.groups, bias=False, use_act=False)

    def forward(self, x: 'torch.Tensor', *args, **kwargs) ->torch.Tensor:
        """
        Forward pass implements inference logic for module
        before and after reparameterization.

        Args:
            x: Input tensor of shape :math:`(B, C, H, W)`.

        Returns:
            torch.Tensor of shape :math:`(B, C, H, W)`.
        """
        if hasattr(self, 'lkb_reparam'):
            out = self.lkb_reparam(x)
        else:
            out = self.lkb_origin(x)
            if hasattr(self, 'small_conv'):
                out += self.small_conv(x)
        self.activation(out)
        return out

    def _get_kernel_bias(self) ->Tuple[torch.Tensor, torch.Tensor]:
        """
        Method to obtain re-parameterized kernel and bias.
        Reference: https://github.com/DingXiaoH/RepLKNet-pytorch

        Returns:
            Tuple of (kernel, bias) after fusing branches.
        """
        lk_kernel, lk_bias = MobileOneBlock._fuse_conv_bn(self.lkb_origin.block.conv.weight, self.lkb_origin.block.norm)
        if hasattr(self, 'small_conv'):
            sk_kernel, sk_bias = MobileOneBlock._fuse_conv_bn(self.small_conv.block.conv.weight, self.small_conv.block.norm)
            lk_bias += sk_bias
            lk_kernel += nn.functional.pad(sk_kernel, [(self.kernel_size - self.small_kernel_size) // 2] * 4)
        return lk_kernel, lk_bias

    def reparameterize(self) ->None:
        """
        Following works like `RepVGG: Making VGG-style ConvNets Great Again` -
        https://arxiv.org/pdf/2101.03697.pdf. We re-parameterize multi-branched
        architecture used at training time to obtain a plain CNN-like structure
        for inference.
        """
        kernel, bias = self._get_kernel_bias()
        self.lkb_reparam = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups, bias=True)
        self.lkb_reparam.weight.data = kernel
        self.lkb_reparam.bias.data = bias
        self.__delattr__('lkb_origin')
        if hasattr(self, 'small_conv'):
            self.__delattr__('small_conv')


class PatchEmbed(BaseModule):
    """
    Convolutional Patch embedding layer.

    Args:
        opts: Command line arguments.
        patch_size: Patch size for embedding computation.
        stride: Stride for convolutional embedding layer.
        in_channels: Number of channels of input tensor.
        embed_dim: Number of embedding dimensions.
    """

    def __init__(self, opts: 'argparse.Namespace', patch_size: 'int', stride: 'int', in_channels: 'int', embed_dim: 'int'):
        super().__init__()
        inference_mode = getattr(opts, 'model.classification.fastvit.inference_mode')
        block = list()
        block.append(RepLKBlock(opts, in_channels=in_channels, out_channels=embed_dim, kernel_size=patch_size, stride=stride, groups=in_channels, small_kernel_size=3, inference_mode=inference_mode))
        block.append(MobileOneBlock(opts, in_channels=embed_dim, out_channels=embed_dim, kernel_size=1, stride=1, padding=0, groups=1, inference_mode=inference_mode, use_se=False, num_conv_branches=1))
        self.proj = nn.Sequential(*block)

    def forward(self, x: 'torch.Tensor', *args, **kwargs) ->torch.Tensor:
        """
        Forward pass

        Args:
            x: Input tensor of shape :math:`(B, C, H, W)`.

        Returns:
            torch.Tensor of shape :math:`(B, C, H//s, W//s)`,
            where `s` is the stride provide while instantiating the layer.
        """
        x = self.proj(x)
        return x


class ConvFFN(BaseModule):
    """
    Convolutional FFN Module.

    Args:
        opts: Command line arguments.
        in_channels: Number of input channels.
        hidden_channels: Number of channels after expansion. Default: None
        out_channels: Number of output channels. Default: None
        drop: Dropout rate. Default: ``0.0``.
    """

    def __init__(self, opts: 'argparse.Namespace', in_channels: 'int', hidden_channels: 'Optional[int]'=None, out_channels: 'Optional[int]'=None, drop: 'float'=0.0):
        super().__init__()
        out_channels = out_channels or in_channels
        hidden_channels = hidden_channels or in_channels
        self.conv = ConvLayer2d(opts, in_channels=in_channels, out_channels=in_channels, kernel_size=7, padding=3, groups=in_channels, bias=False, use_act=False)
        self.fc1 = ConvLayer2d(opts, in_channels, hidden_channels, kernel_size=1, use_norm=False, bias=True)
        self.fc2 = ConvLayer2d(opts, hidden_channels, out_channels, kernel_size=1, use_norm=False, use_act=False, bias=True)
        self.drop = nn.Dropout(drop)

    def forward(self, x: 'torch.Tensor', *args, **kwargs) ->torch.Tensor:
        """
        Forward pass

        Args:
            x: Input tensor of shape :math:`(B, C, H, W)`.

        Returns:
            torch.Tensor of shape :math:`(B, C, H, W)`.
        """
        x = self.conv(x)
        x = self.fc1(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class AttentionBlock(BaseModule):
    """
    Implementation of metaformer block with MHSA as token mixer.
    For more details on Metaformer structure, please refer to:
    `MetaFormer Is Actually What You Need for Vision <https://arxiv.org/pdf/2111.11418.pdf>`_

    Args:
        opts: Command line arguments.
        dim: Number of embedding dimensions.
        mlp_ratio: MLP expansion ratio. Default: 4.0
        drop: Dropout rate. Default: 0.0
        drop_path: Drop path rate. Default: 0.0
        use_layer_scale: Flag to turn on layer scale. Default: ``True``
        layer_scale_init_value: Layer scale value at initialization. Default: 1e-5
    """

    def __init__(self, opts: 'argparse.Namespace', dim: 'int', mlp_ratio: 'float'=4.0, drop: 'float'=0.0, drop_path: 'float'=0.0, use_layer_scale: 'bool'=True, layer_scale_init_value: 'float'=1e-05):
        super().__init__()
        self.norm = BatchNorm2d(num_features=dim)
        self.head_dim = 32
        num_heads = dim // self.head_dim
        self.token_mixer = MultiHeadAttention(embed_dim=dim, num_heads=num_heads, bias=False)
        assert mlp_ratio > 0, 'MLP ratio should be greater than 0, found: {}'.format(mlp_ratio)
        hidden_dim = int(dim * mlp_ratio)
        self.convffn = ConvFFN(opts, in_channels=dim, hidden_channels=hidden_dim, drop=drop)
        self.drop_path = StochasticDepth(drop_path, mode='row') if drop_path > 0.0 else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_1 = nn.Parameter(layer_scale_init_value * torch.ones((dim, 1, 1)))
            self.layer_scale_2 = nn.Parameter(layer_scale_init_value * torch.ones((dim, 1, 1)))

    def _apply_mhsa(self, x: 'torch.Tensor') ->torch.Tensor:
        """
        Perform appropriate reshaping before and after MHSA block.

        Args:
            x: Input tensor of shape :math:`(B, C, H, W)`.

        Returns:
            torch.Tensor of shape :math:`(B, C, H, W)`.
        """
        x_norm = self.norm(x)
        B, C, H, W = x_norm.shape
        x_norm_reshaped = torch.flatten(x_norm, start_dim=2).transpose(-2, -1)
        out = self.token_mixer(x_norm_reshaped)
        out = out.transpose(-2, -1).reshape(B, C, H, W)
        return out

    def forward(self, x: 'torch.Tensor', *args, **kwargs) ->torch.Tensor:
        """
        Forward pass

        Args:
            x: Input tensor of shape :math:`(B, C, H, W)`.

        Returns:
            torch.Tensor output from the attention block.
        """
        if self.use_layer_scale:
            x = x + self.drop_path(self.layer_scale_1 * self._apply_mhsa(x))
            x = x + self.drop_path(self.layer_scale_2 * self.convffn(x))
        else:
            x = x + self.drop_path(self._apply_mhsa(x))
            x = x + self.drop_path(self.convffn(x))
        return x


class RepMixer(BaseModule):
    """
    Reparameterizable token mixer

    For more details, please refer to our paper:
    `FastViT: A Fast Hybrid Vision Transformer using Structural Reparameterization`

    Args:
        opts: Command line arguments.
        dim: Input feature map dimension. :math:`C_{in}` from an expected input of size :math:`(B, C_{in}, H, W)`.
        kernel_size: Kernel size for spatial mixing. Default: 3
        use_layer_scale: If True, learnable layer scale is used. Default: ``True``
        layer_scale_init_value: Initial value for layer scale. Default: 1e-5
        inference_mode: If True, instantiates model in inference mode. Default: ``False``
    """

    def __init__(self, opts: 'argparse.Namespace', dim: 'int', kernel_size: 'int'=3, use_layer_scale: 'bool'=True, layer_scale_init_value: 'float'=1e-05, inference_mode: 'bool'=False):
        super().__init__()
        self.dim = dim
        self.kernel_size = kernel_size
        if inference_mode:
            self.reparam_conv = nn.Conv2d(in_channels=self.dim, out_channels=self.dim, kernel_size=self.kernel_size, stride=1, padding=self.kernel_size // 2, groups=self.dim, bias=True)
        else:
            self.norm = MobileOneBlock(opts, dim, dim, kernel_size, padding=kernel_size // 2, groups=dim, use_act=False, use_scale_branch=False, num_conv_branches=0)
            self.mixer = MobileOneBlock(opts, dim, dim, kernel_size, padding=kernel_size // 2, groups=dim, use_act=False)
            self.use_layer_scale = use_layer_scale
            if use_layer_scale:
                self.layer_scale = nn.Parameter(layer_scale_init_value * torch.ones((dim, 1, 1)))

    def forward(self, x: 'torch.Tensor', *args, **kwargs) ->torch.Tensor:
        """
        Forward pass implements inference logic for module
        before and after reparameterization.

        Args:
            x: Input tensor of shape :math:`(B, C, H, W)`.

        Returns:
            torch.Tensor of shape :math:`(B, C, H, W)`.
        """
        if hasattr(self, 'reparam_conv'):
            x = self.reparam_conv(x)
            return x
        else:
            if self.use_layer_scale:
                x = x + self.layer_scale * (self.mixer(x) - self.norm(x))
            else:
                x = x + self.mixer(x) - self.norm(x)
            return x

    def reparameterize(self) ->None:
        """
        Reparameterize mixer and norm into a single
        convolutional layer for efficient inference.
        """
        self.mixer.reparameterize()
        self.norm.reparameterize()
        if self.use_layer_scale:
            w = self.mixer.id_tensor + self.layer_scale.unsqueeze(-1) * (self.mixer.reparam_conv.weight - self.norm.reparam_conv.weight)
            b = torch.squeeze(self.layer_scale) * (self.mixer.reparam_conv.bias - self.norm.reparam_conv.bias)
        else:
            w = self.mixer.id_tensor + self.mixer.reparam_conv.weight - self.norm.reparam_conv.weight
            b = self.mixer.reparam_conv.bias - self.norm.reparam_conv.bias
        self.reparam_conv = nn.Conv2d(in_channels=self.dim, out_channels=self.dim, kernel_size=self.kernel_size, stride=1, padding=self.kernel_size // 2, groups=self.dim, bias=True)
        self.reparam_conv.weight.data = w
        self.reparam_conv.bias.data = b
        for para in self.parameters():
            para.detach_()
        self.__delattr__('mixer')
        self.__delattr__('norm')


class RepMixerBlock(BaseModule):
    """
    Implementation of Metaformer block with RepMixer as token mixer.
    For more details on Metaformer structure, please refer to:
    `MetaFormer Is Actually What You Need for Vision <https://arxiv.org/pdf/2111.11418.pdf>`_

    Args:
        opts: Command line arguments.
        dim: Number of embedding dimensions.
        kernel_size: Kernel size for repmixer. Default: 3
        mlp_ratio: MLP expansion ratio. Default: 4.0
        drop: Dropout rate. Default: 0.0
        drop_path: Drop path rate. Default: 0.0
        use_layer_scale: Flag to turn on layer scale. Default: ``True``
        layer_scale_init_value: Layer scale value at initialization. Default: 1e-5
        inference_mode: Flag to instantiate block in inference mode. Default: ``False``
    """

    def __init__(self, opts: 'argparse.Namespace', dim: 'int', kernel_size: 'int'=3, mlp_ratio: 'float'=4.0, drop: 'float'=0.0, drop_path: 'float'=0.0, use_layer_scale: 'bool'=True, layer_scale_init_value: 'float'=1e-05, inference_mode: 'bool'=False):
        super().__init__()
        self.token_mixer = RepMixer(opts, dim=dim, kernel_size=kernel_size, use_layer_scale=use_layer_scale, layer_scale_init_value=layer_scale_init_value, inference_mode=inference_mode)
        assert mlp_ratio > 0, 'MLP ratio should be greater than 0, found: {}'.format(mlp_ratio)
        hidden_dim = int(dim * mlp_ratio)
        self.convffn = ConvFFN(opts, in_channels=dim, hidden_channels=hidden_dim, drop=drop)
        self.drop_path = StochasticDepth(drop_path, mode='row') if drop_path > 0.0 else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale = nn.Parameter(layer_scale_init_value * torch.ones((dim, 1, 1)))

    def forward(self, x: 'torch.Tensor', *args, **kwargs) ->torch.Tensor:
        """
        Forward pass

        Args:
            x: Input tensor of shape :math:`(B, C, H, W)`.

        Returns:
            torch.Tensor of shape :math:`(B, C, H, W)`.
        """
        if self.use_layer_scale:
            x = self.token_mixer(x)
            x = x + self.drop_path(self.layer_scale * self.convffn(x))
        else:
            x = self.token_mixer(x)
            x = x + self.drop_path(self.convffn(x))
        return x


def basic_blocks(opts: 'argparse.Namespace', dim: 'int', block_index: 'int', num_blocks: 'List[int]', token_mixer_type: 'str', kernel_size: 'int'=3, mlp_ratio: 'float'=4.0, drop_rate: 'float'=0.0, drop_path_rate: 'float'=0.0, inference_mode: 'bool'=False, use_layer_scale: 'bool'=True, layer_scale_init_value: 'float'=1e-05) ->nn.Sequential:
    """Build FastViT blocks within a stage.

    Args:
        opts: Command line arguments.
        dim: Number of embedding dimensions.
        block_index: block index.
        num_blocks: List containing number of blocks per stage.
        token_mixer_type: Token mixer type.
        kernel_size: Kernel size for repmixer.
        mlp_ratio: MLP expansion ratio.
        drop_rate: Dropout rate.
        drop_path_rate: Drop path rate.
        inference_mode: Flag to instantiate block in inference mode.
        use_layer_scale: Flag to turn on layer scale regularization.
        layer_scale_init_value: Layer scale value at initialization.

    Returns:
        nn.Sequential object of all the blocks within the stage.
    """
    blocks = []
    for block_idx in range(num_blocks[block_index]):
        block_dpr = drop_path_rate * (block_idx + sum(num_blocks[:block_index])) / (sum(num_blocks) - 1)
        if token_mixer_type == 'repmixer':
            blocks.append(RepMixerBlock(opts, dim, kernel_size=kernel_size, mlp_ratio=mlp_ratio, drop=drop_rate, drop_path=block_dpr, inference_mode=inference_mode, use_layer_scale=use_layer_scale, layer_scale_init_value=layer_scale_init_value))
        elif token_mixer_type == 'attention':
            blocks.append(AttentionBlock(opts, dim, mlp_ratio=mlp_ratio, drop=drop_rate, drop_path=block_dpr, use_layer_scale=use_layer_scale, layer_scale_init_value=layer_scale_init_value))
        else:
            raise ValueError('Token mixer type: {} not supported'.format(token_mixer_type))
    blocks = nn.Sequential(*blocks)
    return blocks


def convolutional_stem(opts: 'argparse.Namespace', in_channels: 'int', out_channels: 'int') ->nn.Sequential:
    """
    Build convolutional stem with MobileOne blocks.

    Args:
        opts: Command line arguments.
        in_channels: Number of input channels.
        out_channels: Number of output channels.

    Returns:
        nn.Sequential object with stem elements.
    """
    inference_mode = getattr(opts, 'model.classification.fastvit.inference_mode')
    return nn.Sequential(MobileOneBlock(opts, in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1, groups=1, inference_mode=inference_mode, use_se=False, num_conv_branches=1), MobileOneBlock(opts, in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1, groups=out_channels, inference_mode=inference_mode, use_se=False, num_conv_branches=1), MobileOneBlock(opts, in_channels=out_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, groups=1, inference_mode=inference_mode, use_se=False, num_conv_branches=1))


class InvertedResidual(BaseModule):
    """
    This class implements the inverted residual block, as described in `MobileNetv2 <https://arxiv.org/abs/1801.04381>`_ paper

    Args:
        opts: command-line arguments
        in_channels (int): :math:`C_{in}` from an expected input of size :math:`(N, C_{in}, H_{in}, W_{in})`
        out_channels (int): :math:`C_{out}` from an expected output of size :math:`(N, C_{out}, H_{out}, W_{out)`
        stride (Optional[int]): Use convolutions with a stride. Default: 1
        expand_ratio (Union[int, float]): Expand the input channels by this factor in depth-wise conv
        dilation (Optional[int]): Use conv with dilation. Default: 1
        skip_connection (Optional[bool]): Use skip-connection. Default: True

    Shape:
        - Input: :math:`(N, C_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, C_{out}, H_{out}, W_{out})`

    .. note::
        If `in_channels =! out_channels` and `stride > 1`, we set `skip_connection=False`

    """

    def __init__(self, opts, in_channels: 'int', out_channels: 'int', stride: 'int', expand_ratio: 'Union[int, float]', dilation: 'int'=1, skip_connection: 'Optional[bool]'=True, *args, **kwargs) ->None:
        assert stride in [1, 2]
        hidden_dim = make_divisible(int(round(in_channels * expand_ratio)), 8)
        super().__init__()
        block = nn.Sequential()
        if expand_ratio != 1:
            block.add_module(name='exp_1x1', module=ConvLayer2d(opts, in_channels=in_channels, out_channels=hidden_dim, kernel_size=1, use_act=True, use_norm=True))
        block.add_module(name='conv_3x3', module=ConvLayer2d(opts, in_channels=hidden_dim, out_channels=hidden_dim, stride=stride, kernel_size=3, groups=hidden_dim, use_act=True, use_norm=True, dilation=dilation))
        block.add_module(name='red_1x1', module=ConvLayer2d(opts, in_channels=hidden_dim, out_channels=out_channels, kernel_size=1, use_act=False, use_norm=True))
        self.block = block
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.exp = expand_ratio
        self.dilation = dilation
        self.stride = stride
        self.use_res_connect = self.stride == 1 and in_channels == out_channels and skip_connection

    def forward(self, x: 'Tensor', *args, **kwargs) ->Tensor:
        if self.use_res_connect:
            return x + self.block(x)
        else:
            return self.block(x)

    def __repr__(self) ->str:
        return '{}(in_channels={}, out_channels={}, stride={}, exp={}, dilation={}, skip_conn={})'.format(self.__class__.__name__, self.in_channels, self.out_channels, self.stride, self.exp, self.dilation, self.use_res_connect)


class TransformerEncoder(BaseModule):
    """
    This class defines the pre-norm `Transformer encoder <https://arxiv.org/abs/1706.03762>`_
    Args:
        opts: Command line arguments.
        embed_dim: :math:`C_{in}` from an expected input of size :math:`(N, P, C_{in})`.
        ffn_latent_dim: Inner dimension of the FFN.
        num_heads: Number of heads in multi-head attention. Default: 8.
        attn_dropout: Dropout rate for attention in multi-head attention. Default: 0.0
        dropout: Dropout rate. Default: 0.0.
        ffn_dropout: Dropout between FFN layers. Default: 0.0.
        transformer_norm_layer: Normalization layer. Default: layer_norm.
        stochastic_dropout: Stochastic dropout setting. Default: 0.0.

    Shape:
        - Input: :math:`(N, P, C_{in})` where :math:`N` is batch size, :math:`P` is number of patches,
        and :math:`C_{in}` is input embedding dim
        - Output: same shape as the input
    """

    def __init__(self, opts: 'argparse.Namespace', embed_dim: 'int', ffn_latent_dim: 'int', num_heads: 'Optional[int]'=8, attn_dropout: 'Optional[float]'=0.0, dropout: 'Optional[float]'=0.0, ffn_dropout: 'Optional[float]'=0.0, transformer_norm_layer: 'Optional[str]'='layer_norm', stochastic_dropout: 'Optional[float]'=0.0, *args, **kwargs) ->None:
        super().__init__()
        attn_unit = SingleHeadAttention(embed_dim=embed_dim, attn_dropout=attn_dropout, bias=True)
        if num_heads > 1:
            attn_unit = MultiHeadAttention(embed_dim, num_heads, attn_dropout=attn_dropout, bias=True, coreml_compatible=getattr(opts, 'common.enable_coreml_compatible_module', False))
        self.pre_norm_mha = nn.Sequential(get_normalization_layer(opts=opts, norm_type=transformer_norm_layer, num_features=embed_dim), attn_unit, Dropout(p=dropout))
        act_name = build_activation_layer(opts, num_parameters=1)
        self.pre_norm_ffn = nn.Sequential(get_normalization_layer(opts=opts, norm_type=transformer_norm_layer, num_features=embed_dim), LinearLayer(in_features=embed_dim, out_features=ffn_latent_dim, bias=True), act_name, Dropout(p=ffn_dropout), LinearLayer(in_features=ffn_latent_dim, out_features=embed_dim, bias=True), Dropout(p=dropout))
        self.drop_path = Identity()
        if stochastic_dropout > 0.0:
            if dropout > 0.0:
                logger.error('Stochastic dropout and dropout are mutually exclusive. Use either of them, but not both.Got: {} and {}'.format(stochastic_dropout, dropout))
            self.drop_path = StochasticDepth(p=stochastic_dropout, mode='row')
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_latent_dim
        self.ffn_dropout = ffn_dropout
        self.stochastic_dropout = stochastic_dropout
        self.std_dropout = dropout
        self.attn_fn_name = attn_unit.__class__.__name__
        self.act_fn_name = act_name.__class__.__name__
        self.norm_type = transformer_norm_layer

    def __repr__(self) ->str:
        return '{}(embed_dim={}, ffn_dim={}, dropout={}, ffn_dropout={}, stochastic_dropout={}, attn_fn={}, act_fn={}, norm_fn={})'.format(self.__class__.__name__, self.embed_dim, self.ffn_dim, self.std_dropout, self.ffn_dropout, self.stochastic_dropout, self.attn_fn_name, self.act_fn_name, self.norm_type)

    def forward(self, x: 'Tensor', x_prev: 'Optional[Tensor]'=None, key_padding_mask: 'Optional[Tensor]'=None, attn_mask: 'Optional[Tensor]'=None, *args, **kwargs) ->Tensor:
        res = x
        x = self.pre_norm_mha[0](x)
        x = self.pre_norm_mha[1](*args, x_q=x, x_kv=x_prev, key_padding_mask=key_padding_mask, attn_mask=attn_mask, **kwargs)
        x = self.drop_path(self.pre_norm_mha[2](x))
        x = x + res
        x = x + self.drop_path(self.pre_norm_ffn(x))
        return x


class MobileViTBlock(BaseModule):
    """
    This class defines the `MobileViT block <https://arxiv.org/abs/2110.02178?context=cs.LG>`_

    Args:
        opts: command line arguments
        in_channels (int): :math:`C_{in}` from an expected input of size :math:`(N, C_{in}, H, W)`
        transformer_dim (int): Input dimension to the transformer unit
        ffn_dim (int): Dimension of the FFN block
        n_transformer_blocks (Optional[int]): Number of transformer blocks. Default: 2
        head_dim (Optional[int]): Head dimension in the multi-head attention. Default: 32
        attn_dropout (Optional[float]): Dropout in multi-head attention. Default: 0.0
        dropout (Optional[float]): Dropout rate. Default: 0.0
        ffn_dropout (Optional[float]): Dropout between FFN layers in transformer. Default: 0.0
        patch_h (Optional[int]): Patch height for unfolding operation. Default: 8
        patch_w (Optional[int]): Patch width for unfolding operation. Default: 8
        transformer_norm_layer (Optional[str]): Normalization layer in the transformer block. Default: layer_norm
        conv_ksize (Optional[int]): Kernel size to learn local representations in MobileViT block. Default: 3
        dilation (Optional[int]): Dilation rate in convolutions. Default: 1
        no_fusion (Optional[bool]): Do not combine the input and output feature maps. Default: False
    """

    def __init__(self, opts, in_channels: 'int', transformer_dim: 'int', ffn_dim: 'int', n_transformer_blocks: 'Optional[int]'=2, head_dim: 'Optional[int]'=32, attn_dropout: 'Optional[float]'=0.0, dropout: 'Optional[int]'=0.0, ffn_dropout: 'Optional[int]'=0.0, patch_h: 'Optional[int]'=8, patch_w: 'Optional[int]'=8, transformer_norm_layer: 'Optional[str]'='layer_norm', conv_ksize: 'Optional[int]'=3, dilation: 'Optional[int]'=1, no_fusion: 'Optional[bool]'=False, *args, **kwargs) ->None:
        conv_3x3_in = ConvLayer2d(opts=opts, in_channels=in_channels, out_channels=in_channels, kernel_size=conv_ksize, stride=1, use_norm=True, use_act=True, dilation=dilation)
        conv_1x1_in = ConvLayer2d(opts=opts, in_channels=in_channels, out_channels=transformer_dim, kernel_size=1, stride=1, use_norm=False, use_act=False)
        conv_1x1_out = ConvLayer2d(opts=opts, in_channels=transformer_dim, out_channels=in_channels, kernel_size=1, stride=1, use_norm=True, use_act=True)
        conv_3x3_out = None
        if not no_fusion:
            conv_3x3_out = ConvLayer2d(opts=opts, in_channels=2 * in_channels, out_channels=in_channels, kernel_size=conv_ksize, stride=1, use_norm=True, use_act=True)
        super().__init__()
        self.local_rep = nn.Sequential()
        self.local_rep.add_module(name='conv_3x3', module=conv_3x3_in)
        self.local_rep.add_module(name='conv_1x1', module=conv_1x1_in)
        assert transformer_dim % head_dim == 0
        num_heads = transformer_dim // head_dim
        global_rep = [TransformerEncoder(opts=opts, embed_dim=transformer_dim, ffn_latent_dim=ffn_dim, num_heads=num_heads, attn_dropout=attn_dropout, dropout=dropout, ffn_dropout=ffn_dropout, transformer_norm_layer=transformer_norm_layer) for _ in range(n_transformer_blocks)]
        global_rep.append(get_normalization_layer(opts=opts, norm_type=transformer_norm_layer, num_features=transformer_dim))
        self.global_rep = nn.Sequential(*global_rep)
        self.conv_proj = conv_1x1_out
        self.fusion = conv_3x3_out
        self.patch_h = patch_h
        self.patch_w = patch_w
        self.patch_area = self.patch_w * self.patch_h
        self.cnn_in_dim = in_channels
        self.cnn_out_dim = transformer_dim
        self.n_heads = num_heads
        self.ffn_dim = ffn_dim
        self.dropout = dropout
        self.attn_dropout = attn_dropout
        self.ffn_dropout = ffn_dropout
        self.dilation = dilation
        self.n_blocks = n_transformer_blocks
        self.conv_ksize = conv_ksize

    def __repr__(self) ->str:
        repr_str = '{}('.format(self.__class__.__name__)
        repr_str += '\n\t Local representations'
        if isinstance(self.local_rep, nn.Sequential):
            for m in self.local_rep:
                repr_str += '\n\t\t {}'.format(m)
        else:
            repr_str += '\n\t\t {}'.format(self.local_rep)
        repr_str += '\n\t Global representations with patch size of {}x{}'.format(self.patch_h, self.patch_w)
        if isinstance(self.global_rep, nn.Sequential):
            for m in self.global_rep:
                repr_str += '\n\t\t {}'.format(m)
        else:
            repr_str += '\n\t\t {}'.format(self.global_rep)
        if isinstance(self.conv_proj, nn.Sequential):
            for m in self.conv_proj:
                repr_str += '\n\t\t {}'.format(m)
        else:
            repr_str += '\n\t\t {}'.format(self.conv_proj)
        if self.fusion is not None:
            repr_str += '\n\t Feature fusion'
            if isinstance(self.fusion, nn.Sequential):
                for m in self.fusion:
                    repr_str += '\n\t\t {}'.format(m)
            else:
                repr_str += '\n\t\t {}'.format(self.fusion)
        repr_str += '\n)'
        return repr_str

    def unfolding(self, feature_map: 'Tensor') ->Tuple[Tensor, Dict]:
        patch_w, patch_h = self.patch_w, self.patch_h
        patch_area = int(patch_w * patch_h)
        batch_size, in_channels, orig_h, orig_w = feature_map.shape
        new_h = int(math.ceil(orig_h / self.patch_h) * self.patch_h)
        new_w = int(math.ceil(orig_w / self.patch_w) * self.patch_w)
        interpolate = False
        if new_w != orig_w or new_h != orig_h:
            feature_map = F.interpolate(feature_map, size=(new_h, new_w), mode='bilinear', align_corners=False)
            interpolate = True
        num_patch_w = new_w // patch_w
        num_patch_h = new_h // patch_h
        num_patches = num_patch_h * num_patch_w
        reshaped_fm = feature_map.reshape(batch_size * in_channels * num_patch_h, patch_h, num_patch_w, patch_w)
        transposed_fm = reshaped_fm.transpose(1, 2)
        reshaped_fm = transposed_fm.reshape(batch_size, in_channels, num_patches, patch_area)
        transposed_fm = reshaped_fm.transpose(1, 3)
        patches = transposed_fm.reshape(batch_size * patch_area, num_patches, -1)
        info_dict = {'orig_size': (orig_h, orig_w), 'batch_size': batch_size, 'interpolate': interpolate, 'total_patches': num_patches, 'num_patches_w': num_patch_w, 'num_patches_h': num_patch_h}
        return patches, info_dict

    def folding(self, patches: 'Tensor', info_dict: 'Dict') ->Tensor:
        n_dim = patches.dim()
        assert n_dim == 3, 'Tensor should be of shape BPxNxC. Got: {}'.format(patches.shape)
        patches = patches.contiguous().view(info_dict['batch_size'], self.patch_area, info_dict['total_patches'], -1)
        batch_size, pixels, num_patches, channels = patches.size()
        num_patch_h = info_dict['num_patches_h']
        num_patch_w = info_dict['num_patches_w']
        patches = patches.transpose(1, 3)
        feature_map = patches.reshape(batch_size * channels * num_patch_h, num_patch_w, self.patch_h, self.patch_w)
        feature_map = feature_map.transpose(1, 2)
        feature_map = feature_map.reshape(batch_size, channels, num_patch_h * self.patch_h, num_patch_w * self.patch_w)
        if info_dict['interpolate']:
            feature_map = F.interpolate(feature_map, size=info_dict['orig_size'], mode='bilinear', align_corners=False)
        return feature_map

    def forward_spatial(self, x: 'Tensor') ->Tensor:
        res = x
        fm = self.local_rep(x)
        patches, info_dict = self.unfolding(fm)
        for transformer_layer in self.global_rep:
            patches = transformer_layer(patches)
        fm = self.folding(patches=patches, info_dict=info_dict)
        fm = self.conv_proj(fm)
        if self.fusion is not None:
            fm = self.fusion(torch.cat((res, fm), dim=1))
        return fm

    def forward_temporal(self, x: 'Tensor', x_prev: 'Optional[Tensor]'=None) ->Union[Tensor, Tuple[Tensor, Tensor]]:
        res = x
        fm = self.local_rep(x)
        patches, info_dict = self.unfolding(fm)
        for global_layer in self.global_rep:
            if isinstance(global_layer, TransformerEncoder):
                patches = global_layer(x=patches, x_prev=x_prev)
            else:
                patches = global_layer(patches)
        fm = self.folding(patches=patches, info_dict=info_dict)
        fm = self.conv_proj(fm)
        if self.fusion is not None:
            fm = self.fusion(torch.cat((res, fm), dim=1))
        return fm, patches

    def forward(self, x: 'Union[Tensor, Tuple[Tensor]]', *args, **kwargs) ->Union[Tensor, Tuple[Tensor, Tensor]]:
        if isinstance(x, Tuple) and len(x) == 2:
            return self.forward_temporal(x=x[0], x_prev=x[1])
        elif isinstance(x, Tensor):
            return self.forward_spatial(x)
        else:
            raise NotImplementedError


class XRegNetBlock(BaseModule):
    """
    This class implements the `X` block based on the ResNet bottleneck block. See figure 4 of RegNet
    paper `RegNet model <https://arxiv.org/pdf/2003.13678.pdf>`_

    Args:
        opts: command-line arguments
        width_in: The number of input channels
        width_out: The number of output channels
        stride: Stride for convolution
        groups: Number of groups for convolution
        bottleneck_multiplier: The number of in/out channels of the intermediate
            conv layer will be scaled by this value
        se_ratio: The numer squeeze-excitation ratio. The number of channels in the SE
            module will be scaled by this value
        stochastic_depth_prob: The stochastic depth probability
    """

    def __init__(self, opts: 'argparse.Namespace', width_in: 'int', width_out: 'int', stride: 'int', groups: 'int', bottleneck_multiplier: 'float', se_ratio: 'float', stochastic_depth_prob: 'float'=0.0) ->None:
        super().__init__()
        bottleneck_width = int(round(width_out * bottleneck_multiplier))
        bottleneck_groups = bottleneck_width // groups
        conv_1x1_1 = ConvLayer2d(opts=opts, in_channels=width_in, out_channels=bottleneck_width, kernel_size=1, stride=1, use_norm=True, use_act=True)
        conv_3x3 = ConvLayer2d(opts=opts, in_channels=bottleneck_width, out_channels=bottleneck_width, kernel_size=3, stride=stride, groups=bottleneck_groups, use_norm=True, use_act=True)
        se = Identity()
        if se_ratio > 0:
            squeeze_channels = int(round(se_ratio * width_in))
            se = SqueezeExcitation(opts, in_channels=bottleneck_width, squeeze_channels=squeeze_channels)
        conv_1x1_2 = ConvLayer2d(opts=opts, in_channels=bottleneck_width, out_channels=width_out, kernel_size=1, stride=1, use_norm=True, use_act=True)
        block = nn.Sequential()
        block.add_module('conv_1x1_1', module=conv_1x1_1)
        block.add_module('conv_3x3', module=conv_3x3)
        block.add_module('se', module=se)
        block.add_module('conv_1x1_2', module=conv_1x1_2)
        down_sample = Identity()
        if stride != 1 or width_out != width_in:
            down_sample = ConvLayer2d(opts, in_channels=width_in, out_channels=width_out, kernel_size=1, stride=stride, use_act=False)
        act_type = getattr(opts, 'model.activation.name')
        neg_slope = getattr(opts, 'model.activation.neg_slope')
        inplace = getattr(opts, 'model.activation.inplace')
        final_act = build_activation_layer(opts=opts, act_type=act_type, inplace=inplace, negative_slope=neg_slope, num_parameters=width_out)
        self.stochastic_depth = StochasticDepth(p=stochastic_depth_prob, mode='row')
        self.block = block
        self.down_sample = down_sample
        self.final_act = final_act
        self.width_in = width_in
        self.width_out = width_out
        self.stride = stride
        self.groups = groups
        self.bottleneck_multiplier = bottleneck_multiplier
        self.se_ratio = se_ratio
        self.stochastic_depth_prob = stochastic_depth_prob

    def forward(self, x: 'Tensor') ->Tensor:
        """Forward pass for XRegNetBlock.

        Args:
            x: Batch of images

        Retruns:
            * output of XRegNetBlock including stochastic depth layer and
                residual.

        Shape:
            x: :math:`(N, C_{in}, H_{in}, W_{in})`
            Output: :math:`(N, C_{out}, H_{out}, W_{out})`
        """
        out = self.block(x)
        out = self.stochastic_depth(out)
        res = self.down_sample(x)
        out = res + out
        return self.final_act(out)

    def __repr__(self) ->str:
        return '{}(width_in={}, width_out={}, stride={}, groups={}, bottleneck_multiplier={}, se_ratio={}, stochastic_depth_prob={})'.format(self.__class__.__name__, self.width_in, self.width_out, self.stride, self.groups, self.bottleneck_multiplier, self.se_ratio, self.stochastic_depth_prob)


class AnyRegNetStage(BaseModule):
    """
    This class implements a 'stage' as defined in the `RegNet paper <https://arxiv.org/pdf/2003.13678.pdf>`_.
    It consists of a sequence of bottleneck blocks.

    Args:
        opts: command-line arguments
        depth: The number of XRegNetBlocks in the stage
        width_in: The number of input channels of the first block
        width_out: The number of output channels of each block
        stride: Stride for convolution of first block
        groups: Number of groups for the intermediate convolution (bottleneck) layer in each block
        bottleneck_multiplier: The number of in/out channels of the intermediate
            conv layer of each block will be scaled by this value
        se_ratio: The numer squeeze-excitation ratio. The number of channels in the SE
            module of each block will be scaled by this value
        stage_depths: A list of the number of blocks in each stage
        stage_index: The index of the current stage being constructed
        stochastic_depth_prob: The stochastic depth probability
    """

    def __init__(self, opts: 'argparse.Namespace', depth: 'int', width_in: 'int', width_out: 'int', stride: 'int', groups: 'int', bottleneck_multiplier: 'float', se_ratio: 'float', stage_index: 'int', stochastic_depth_probs: 'List[float]') ->None:
        super().__init__()
        stage_blocks = nn.Sequential()
        for i, sd_prob in enumerate(stochastic_depth_probs):
            block = XRegNetBlock(opts, width_in=width_in if i == 0 else width_out, width_out=width_out, stride=stride if i == 0 else 1, groups=groups, bottleneck_multiplier=bottleneck_multiplier, se_ratio=se_ratio, stochastic_depth_prob=sd_prob)
            stage_blocks.add_module(f'Stage{stage_index}-Block{i}', module=block)
        self.stage = stage_blocks
        self.depth = depth
        self.width_in = width_in
        self.width_out = width_out
        self.stride = stride
        self.groups = groups
        self.bottleneck_multiplier = bottleneck_multiplier
        self.se_ratio = se_ratio
        self.stage_index = stage_index
        self.stochastic_depth_probs = stochastic_depth_probs

    def forward(self, x: 'Tensor') ->Tensor:
        """Forward pass through all blocks in the stage.

        Args:
            x: Batch of images.

        Returns:
            * output of passing x through all blocks in the stage.

        Shape:
            x: :math:`(N, C_{in}, H_{in}, W_{in})`
            Output: :math:`(N, C_{out}, H_{out}, W_{out})`
        """
        return self.stage(x)

    def __repr__(self) ->str:
        return '{}(depth={}, width_in={}, width_out={}, stride={}, groups={}, bottleneck_multiplier={}, se_ratio={}, stage_index={}, stochastic_depth_probs={})'.format(self.__class__.__name__, self.depth, self.width_in, self.width_out, self.stride, self.groups, self.bottleneck_multiplier, self.se_ratio, self.stage_index, self.stochastic_depth_probs)


supported_modes = ['x_200mf', 'x_400mf', 'x_600mf', 'x_800mf', 'x_1.6gf', 'x_3.2gf', 'x_4.0gf', 'x_6.4gf', 'x_8.0gf', 'x_12gf', 'x_16gf', 'x_32gf', 'y_200mf', 'y_400mf', 'y_800mf', 'y_600mf', 'y_1.6gf', 'y_3.2gf', 'y_4.0gf', 'y_6.4gf', 'y_8.0gf', 'y_12gf', 'y_16gf', 'y_32gf']


class BasicResNetBlock(BaseModule):
    """
    This class defines the Basic block in the `ResNet model <https://arxiv.org/abs/1512.03385>`_
    Args:
        opts: command-line arguments
        in_channels (int): :math:`C_{in}` from an expected input of size :math:`(N, C_{in}, H_{in}, W_{in})`
        mid_channels (int): :math:`C_{mid}` from an expected tensor of size :math:`(N, C_{mid}, H_{out}, W_{out})`
        out_channels (int): :math:`C_{out}` from an expected output of size :math:`(N, C_{out}, H_{out}, W_{out})`
        stride (Optional[int]): Stride for convolution. Default: 1
        dilation (Optional[int]): Dilation for convolution. Default: 1
        dropout (Optional[float]): Dropout after second convolution. Default: 0.0
        stochastic_depth_prob (Optional[float]): Stochastic depth drop probability (1 - survival_prob). Default: 0.0
        squeeze_channels (Optional[int]): The number of channels to use in the Squeeze-Excitation block for SE-ResNet.
            Default: None.

    Shape:
        - Input: :math:`(N, C_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, C_{out}, H_{out}, W_{out})`

    """
    expansion: 'int' = 1

    def __init__(self, opts: 'argparse.Namespace', in_channels: 'int', mid_channels: 'int', out_channels: 'int', stride: 'Optional[int]'=1, dilation: 'Optional[int]'=1, dropout: 'Optional[float]'=0.0, stochastic_depth_prob: 'Optional[float]'=0.0, squeeze_channels: 'Optional[int]'=None, *args, **kwargs) ->None:
        act_type = getattr(opts, 'model.activation.name')
        neg_slope = getattr(opts, 'model.activation.neg_slope')
        inplace = getattr(opts, 'model.activation.inplace')
        cbr_1 = ConvLayer2d(opts=opts, in_channels=in_channels, out_channels=mid_channels, kernel_size=3, stride=stride, dilation=dilation, use_norm=True, use_act=True)
        cb_2 = ConvLayer2d(opts=opts, in_channels=mid_channels, out_channels=out_channels, kernel_size=3, stride=1, use_norm=True, use_act=False, dilation=dilation)
        block = nn.Sequential()
        block.add_module(name='conv_batch_act_1', module=cbr_1)
        block.add_module(name='conv_batch_2', module=cb_2)
        if 0.0 < dropout < 1.0:
            block.add_module(name='dropout', module=Dropout(p=dropout))
        down_sample = Identity()
        if stride == 2:
            down_sample = ConvLayer2d(opts=opts, in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, use_norm=True, use_act=False)
        se_block = Identity()
        if squeeze_channels is not None:
            se_block = SqueezeExcitation(opts=opts, in_channels=out_channels, squeeze_channels=squeeze_channels)
        super().__init__()
        self.block = block
        self.down_sample = down_sample
        self.final_act = build_activation_layer(opts, act_type=act_type, inplace=inplace, negative_slope=neg_slope, num_parameters=out_channels)
        self.stochastic_depth = StochasticDepth(p=stochastic_depth_prob, mode='row')
        self.se_block = se_block
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dilation = dilation
        self.dropout = dropout
        self.stochastic_depth_prob = stochastic_depth_prob
        self.squeeze_channels = squeeze_channels

    def forward(self, x: 'Tensor', *args, **kwargs) ->Tensor:
        out = self.block(x)
        out = self.se_block(out)
        res = self.down_sample(x)
        out = self.stochastic_depth(out)
        out = out + res
        return self.final_act(out)

    def __repr__(self) ->str:
        return '{}(in_channels={}, out_channels={}, stride={}, dilation={}, dropout={}, stochastic_depth_prob={}, squeeze_channels={})'.format(self.__class__.__name__, self.in_channels, self.out_channels, self.stride, self.dilation, self.dropout, self.stochastic_depth_prob, self.squeeze_channels)


class BottleneckResNetBlock(BaseModule):
    """
    This class defines the Bottleneck block in the `ResNet model <https://arxiv.org/abs/1512.03385>`_
    Args:
        opts: command-line arguments
        in_channels (int): :math:`C_{in}` from an expected input of size :math:`(N, C_{in}, H_{in}, W_{in})`
        mid_channels (int): :math:`C_{mid}` from an expected tensor of size :math:`(N, C_{mid}, H_{out}, W_{out})`
        out_channels (int): :math:`C_{out}` from an expected output of size :math:`(N, C_{out}, H_{out}, W_{out})`
        stride (Optional[int]): Stride for convolution. Default: 1
        dilation (Optional[int]): Dilation for convolution. Default: 1
        dropout (Optional[float]): Dropout after third convolution. Default: 0.0
        stochastic_depth_prob (Optional[float]): Stochastic depth drop probability (1 - survival_prob). Default: 0.0
        squeeze_channels (Optional[int]): The number of channels to use in the Squeeze-Excitation block for SE-ResNet.

    Shape:
        - Input: :math:`(N, C_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, C_{out}, H_{out}, W_{out})`

    """
    expansion: 'int' = 4

    def __init__(self, opts: 'argparse.Namespace', in_channels: 'int', mid_channels: 'int', out_channels: 'int', stride: 'Optional[int]'=1, dilation: 'Optional[int]'=1, dropout: 'Optional[float]'=0.0, stochastic_depth_prob: 'Optional[float]'=0.0, squeeze_channels: 'Optional[int]'=None, *args, **kwargs) ->None:
        act_type = getattr(opts, 'model.activation.name')
        neg_slope = getattr(opts, 'model.activation.neg_slope')
        inplace = getattr(opts, 'model.activation.inplace')
        cbr_1 = ConvLayer2d(opts=opts, in_channels=in_channels, out_channels=mid_channels, kernel_size=1, stride=1, use_norm=True, use_act=True)
        cbr_2 = ConvLayer2d(opts=opts, in_channels=mid_channels, out_channels=mid_channels, kernel_size=3, stride=stride, use_norm=True, use_act=True, dilation=dilation)
        cb_3 = ConvLayer2d(opts=opts, in_channels=mid_channels, out_channels=out_channels, kernel_size=1, stride=1, use_norm=True, use_act=False)
        block = nn.Sequential()
        block.add_module(name='conv_batch_act_1', module=cbr_1)
        block.add_module(name='conv_batch_act_2', module=cbr_2)
        block.add_module(name='conv_batch_3', module=cb_3)
        if 0.0 < dropout < 1.0:
            block.add_module(name='dropout', module=Dropout(p=dropout))
        down_sample = Identity()
        if stride == 2:
            down_sample = ConvLayer2d(opts=opts, in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, use_norm=True, use_act=False)
        elif in_channels != out_channels:
            down_sample = ConvLayer2d(opts=opts, in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, use_norm=True, use_act=False)
        se_block = Identity()
        if squeeze_channels is not None:
            se_block = SqueezeExcitation(opts=opts, in_channels=out_channels, squeeze_channels=squeeze_channels)
        super().__init__()
        self.block = block
        self.down_sample = down_sample
        self.final_act = build_activation_layer(opts, act_type=act_type, inplace=inplace, negative_slope=neg_slope, num_parameters=out_channels)
        self.stochastic_depth = StochasticDepth(p=stochastic_depth_prob, mode='row')
        self.se_block = se_block
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mid_channels = mid_channels
        self.dilation = dilation
        self.dropout = dropout
        self.stochastic_depth_prob = stochastic_depth_prob
        self.squeeze_channels = squeeze_channels

    def forward(self, x: 'Tensor', *args, **kwargs) ->Tensor:
        out = self.block(x)
        out = self.se_block(out)
        res = self.down_sample(x)
        out = self.stochastic_depth(out)
        out = out + res
        return self.final_act(out)

    def __repr__(self) ->str:
        return '{}(in_channels={}, mid_channels={}, out_channels={}, stride={}, dilation={}, dropout={}, stochastic_depth_prob={}, squeeze_channels={})'.format(self.__class__.__name__, self.in_channels, self.mid_channels, self.out_channels, self.stride, self.dilation, self.dropout, self.stochastic_depth_prob, self.squeeze_channels)


def _patch_merging_pad(x):
    H, W, _ = x.shape[-3:]
    x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
    return x


class PatchMerging(BaseModule):
    """Patch Merging Layer.
    Args:
        dim (int): Number of input channels.
        norm_layer (str): Normalization layer name.
        strided (Optional[bool]): Down-sample the input by a factor of 2. Default is True.
    """

    def __init__(self, opts, dim: 'int', norm_layer: 'str', strided: 'Optional[bool]'=True):
        super().__init__()
        self.dim = dim
        self.reduction = LinearLayer(in_features=4 * dim, out_features=2 * dim, bias=False)
        self.norm = get_normalization_layer(opts=opts, norm_type=norm_layer, num_features=4 * dim)
        self.strided = strided

    def forward(self, x: 'Tensor', *args, **kwargs) ->Tensor:
        """
        Args:
            x (Tensor): input tensor with expected layout of [..., H, W, C]
        Returns:
            Tensor with layout of [..., H/2, W/2, 2*C]
        """
        x = _patch_merging_pad(x)
        if self.strided:
            x0 = x[..., 0::2, 0::2, :]
            x1 = x[..., 1::2, 0::2, :]
            x2 = x[..., 0::2, 1::2, :]
            x3 = x[..., 1::2, 1::2, :]
            x = torch.cat([x0, x1, x2, x3], -1)
        else:
            x = torch.cat([x, x, x, x], -1)
        x = self.norm(x)
        x = self.reduction(x)
        return x

    def __repr__(self) ->str:
        s = f'{self.__class__.__name__}(dim={self.dim})'
        return s


class Permute(BaseModule):
    """This module returns a view of the tensor input with its dimensions permuted.
    Args:
        dims (List[int]): The desired ordering of dimensions
    """

    def __init__(self, dims: 'List[int]'):
        super().__init__()
        self.dims = dims

    def forward(self, x: 'Tensor') ->Tensor:
        return torch.permute(x, self.dims)

    def __repr__(self) ->str:
        s = f'{self.__class__.__name__}(dims={self.dims})'
        return s


def shifted_window_attention(input: 'Tensor', qkv_weight: 'Tensor', proj_weight: 'Tensor', relative_position_bias: 'Tensor', window_size: 'List[int]', num_heads: 'int', shift_size: 'List[int]', attention_dropout: 'float'=0.0, dropout: 'float'=0.0, qkv_bias: 'Optional[Tensor]'=None, proj_bias: 'Optional[Tensor]'=None):
    """
    Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        input (Tensor[N, H, W, C]): The input tensor or 4-dimensions.
        qkv_weight (Tensor[in_dim, out_dim]): The weight tensor of query, key, value.
        proj_weight (Tensor[out_dim, out_dim]): The weight tensor of projection.
        relative_position_bias (Tensor): The learned relative position bias added to attention.
        window_size (List[int]): Window size.
        num_heads (int): Number of attention heads.
        shift_size (List[int]): Shift size for shifted window attention.
        attention_dropout (float): Dropout ratio of attention weight. Default: 0.0.
        dropout (float): Dropout ratio of output. Default: 0.0.
        qkv_bias (Tensor[out_dim], optional): The bias tensor of query, key, value. Default: None.
        proj_bias (Tensor[out_dim], optional): The bias tensor of projection. Default: None.
    Returns:
        Tensor[N, H, W, C]: The output tensor after shifted window attention.
    """
    B, H, W, C = input.shape
    pad_r = (window_size[1] - W % window_size[1]) % window_size[1]
    pad_b = (window_size[0] - H % window_size[0]) % window_size[0]
    x = F.pad(input, (0, 0, 0, pad_r, 0, pad_b))
    _, pad_H, pad_W, _ = x.shape
    shift_size = shift_size.copy()
    if window_size[0] >= pad_H:
        shift_size[0] = 0
    if window_size[1] >= pad_W:
        shift_size[1] = 0
    if sum(shift_size) > 0:
        x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1]), dims=(1, 2))
    num_windows = pad_H // window_size[0] * (pad_W // window_size[1])
    x = x.view(B, pad_H // window_size[0], window_size[0], pad_W // window_size[1], window_size[1], C)
    x = x.permute(0, 1, 3, 2, 4, 5).reshape(B * num_windows, window_size[0] * window_size[1], C)
    qkv = F.linear(x, qkv_weight, qkv_bias)
    qkv = qkv.reshape(x.size(0), x.size(1), 3, num_heads, C // num_heads).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]
    q = q * (C // num_heads) ** -0.5
    attn = q.matmul(k.transpose(-2, -1))
    attn = attn + relative_position_bias
    if sum(shift_size) > 0:
        attn_mask = x.new_zeros((pad_H, pad_W))
        h_slices = (0, -window_size[0]), (-window_size[0], -shift_size[0]), (-shift_size[0], None)
        w_slices = (0, -window_size[1]), (-window_size[1], -shift_size[1]), (-shift_size[1], None)
        count = 0
        for h in h_slices:
            for w in w_slices:
                attn_mask[h[0]:h[1], w[0]:w[1]] = count
                count += 1
        attn_mask = attn_mask.view(pad_H // window_size[0], window_size[0], pad_W // window_size[1], window_size[1])
        attn_mask = attn_mask.permute(0, 2, 1, 3).reshape(num_windows, window_size[0] * window_size[1])
        attn_mask = attn_mask.unsqueeze(1) - attn_mask.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        attn = attn.view(x.size(0) // num_windows, num_windows, num_heads, x.size(1), x.size(1))
        attn = attn + attn_mask.unsqueeze(1).unsqueeze(0)
        attn = attn.view(-1, num_heads, x.size(1), x.size(1))
    attn = F.softmax(attn, dim=-1)
    attn = F.dropout(attn, p=attention_dropout)
    x = attn.matmul(v).transpose(1, 2).reshape(x.size(0), x.size(1), C)
    x = F.linear(x, proj_weight, proj_bias)
    x = F.dropout(x, p=dropout)
    x = x.view(B, pad_H // window_size[0], pad_W // window_size[1], window_size[0], window_size[1], C)
    x = x.permute(0, 1, 3, 2, 4, 5).reshape(B, pad_H, pad_W, C)
    if sum(shift_size) > 0:
        x = torch.roll(x, shifts=(shift_size[0], shift_size[1]), dims=(1, 2))
    x = x[:, :H, :W, :].contiguous()
    return x


class ShiftedWindowAttention(BaseModule):
    """
    See :func:`shifted_window_attention`.
    """

    def __init__(self, dim: 'int', window_size: 'List[int]', shift_size: 'List[int]', num_heads: 'int', qkv_bias: 'bool'=True, proj_bias: 'bool'=True, attention_dropout: 'float'=0.0, dropout: 'float'=0.0):
        super().__init__()
        if len(window_size) != 2 or len(shift_size) != 2:
            raise ValueError('window_size and shift_size must be of length 2')
        self.window_size = window_size
        self.shift_size = shift_size
        self.num_heads = num_heads
        self.attention_dropout = attention_dropout
        self.dropout = dropout
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.relative_position_bias_table = nn.Parameter(torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1).view(-1)
        self.register_buffer('relative_position_index', relative_position_index)
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.embed_dim = dim

    def __repr__(self) ->str:
        return '{}(embed_dim={}, window_size={}, shift_size={}, num_heads={}, dropout={}, attn_dropout={}, dropout={})'.format(self.__class__.__name__, self.embed_dim, self.window_size, self.shift_size, self.num_heads, self.attention_dropout, self.dropout)

    def forward(self, x: 'Tensor', *args, **kwargs) ->Tensor:
        """
        Args:
            x (Tensor): Tensor with layout of [B, H, W, C]
        Returns:
            Tensor with same layout as input, i.e. [B, H, W, C]
        """
        N = self.window_size[0] * self.window_size[1]
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index]
        relative_position_bias = relative_position_bias.view(N, N, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous().unsqueeze(0)
        return shifted_window_attention(x, self.qkv.weight, self.proj.weight, relative_position_bias, self.window_size, self.num_heads, shift_size=self.shift_size, attention_dropout=self.attention_dropout, dropout=self.dropout, qkv_bias=self.qkv.bias, proj_bias=self.proj.bias)


class SwinTransformerBlock(BaseModule):
    """
    Swin Transformer Block.
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (List[int]): Window size.
        shift_size (List[int]): Shift size for shifted window attention.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.0.
        dropout (float): Dropout rate. Default: 0.0.
        attention_dropout (float): Attention dropout rate. Default: 0.0.
        stochastic_depth_prob: (float): Stochastic depth rate. Default: 0.0.
        norm_layer (nn.Module): Normalization layer.  Default: nn.LayerNorm.
        attn_layer (nn.Module): Attention layer. Default: ShiftedWindowAttention
    """

    def __init__(self, opts, embed_dim: 'int', num_heads: 'int', window_size: 'List[int]', shift_size: 'List[int]', mlp_ratio: 'float'=4.0, dropout: 'float'=0.0, attn_dropout: 'Optional[float]'=0.0, ffn_dropout: 'Optional[float]'=0.0, stochastic_depth_prob: 'float'=0.0, norm_layer: 'Optional[str]'='layer_norm'):
        super().__init__()
        attn_unit = ShiftedWindowAttention(embed_dim, window_size, shift_size, num_heads, attention_dropout=attn_dropout, dropout=dropout)
        self.attn = nn.Sequential(get_normalization_layer(opts=opts, norm_type=norm_layer, num_features=embed_dim), attn_unit, Dropout(p=dropout))
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, 'row')
        ffn_latent_dim = int(embed_dim * mlp_ratio)
        act_name = build_activation_layer(opts, num_parameters=1)
        self.mlp = nn.Sequential(get_normalization_layer(opts=opts, norm_type=norm_layer, num_features=embed_dim), LinearLayer(in_features=embed_dim, out_features=ffn_latent_dim, bias=True), act_name, Dropout(p=ffn_dropout), LinearLayer(in_features=ffn_latent_dim, out_features=embed_dim, bias=True), Dropout(p=dropout))
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_latent_dim
        self.ffn_dropout = ffn_dropout
        self.std_dropout = dropout
        self.attn_fn_name = attn_unit.__class__.__name__
        self.act_fn_name = act_name.__class__.__name__
        self.norm_type = norm_layer

    def __repr__(self) ->str:
        return '{}(embed_dim={}, ffn_dim={}, dropout={}, ffn_dropout={}, attn_fn={}, act_fn={}, norm_fn={})'.format(self.__class__.__name__, self.embed_dim, self.ffn_dim, self.std_dropout, self.ffn_dropout, self.attn_fn_name, self.act_fn_name, self.norm_type)

    def forward(self, x: 'Tensor', *args, **kwargs) ->Tensor:
        x = x + self.stochastic_depth(self.attn(x))
        x = x + self.stochastic_depth(self.mlp(x))
        return x


def check_feature_map_output_channels(config: 'Dict', layer_name: 'str') ->int:
    enc_ch_l: 'Dict' = config.get(layer_name, None)
    if enc_ch_l is None or not enc_ch_l:
        logger.error('Encoder does not define input-output mapping for {}: Got: {}'.format(layer_name, config))
    enc_ch_l_out = enc_ch_l.get('out', None)
    if enc_ch_l_out is None or not enc_ch_l_out:
        logger.error('Output channels are not defined in {} of the encoder. Got: {}'.format(layer_name, enc_ch_l))
    return enc_ch_l_out


def import_modules_from_folder(folder_name: 'str', extra_roots: 'Sequence[str]'=()) ->None:
    """Automatically import all modules from public library root folder, in addition
    to the @extra_roots directories.

    The @folder_name directory must exist in LIBRARY_ROOT, but existence in @extra_roots
    is optional.

    Args:
        folder_name: Name of the folder to search for its internal and public modules.
        extra_roots: By default, this function only imports from
            `LIBRARY_ROOT/{folder_name}/**/*.py`. For any extra_root provided, it will
            also import `LIBRARY_ROOT/{extra_root}/{folder_name}/**/*.py` modules.
    """
    if not LIBRARY_ROOT.joinpath(folder_name).exists():
        logger.error(f"{folder_name} doesn't exist in the public library root directory.")
    for base_dir in ['.', *extra_roots]:
        for path in LIBRARY_ROOT.glob(os.path.join(base_dir, folder_name, '**/*.py')):
            filename = path.name
            if filename[0] not in ('.', '_'):
                module_name = str(path.relative_to(LIBRARY_ROOT).with_suffix('')).replace(os.sep, '.')
                importlib.import_module(module_name)


def clean_strip(obj: 'Union[str, List[str]]', sep: 'Optional[str]'=',', strip: 'bool'=True) ->List[str]:
    if isinstance(obj, list):
        strings = obj
    else:
        strings = obj.split(sep)
    if strip:
        strings = [x.strip() for x in strings]
    strings = [x for x in strings if x]
    return strings


def freeze_module(module: 'torch.nn.Module', force_eval: 'bool'=True) ->torch.nn.Module:
    """
    Sets requires_grad = False on all the given module parameters, and put the module in eval mode.
    By default, it also overrides the module's `train` method to make sure that it always stays in eval mode
    (ie calling ``module.train(mode=True)`` executes ``module.train(mode=False)``)

    >>> module = nn.Linear(10, 20).train()
    >>> module.training
    True
    >>> module.weight.requires_grad
    True
    >>> freeze_module(module).train().training
    False
    >>> module.weight.requires_grad
    False
    """
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad = False
    if force_eval:

        def _force_train_in_eval(self: 'torch.nn.Module', mode: 'bool'=True) ->torch.nn.Module:
            return self
        module.train = MethodType(_force_train_in_eval, module)
    return module


def freeze_modules_based_on_opts(opts: 'argparse.Namespace', model: 'torch.nn.Module', verbose: 'bool'=True) ->torch.nn.Module:
    """
    Allows for freezing immediate modules and parameters of the model using --model.freeze-modules.

    --model.freeze-modules should be a list of strings or a comma-separated list of regex expressions.

    Examples of --model.freeze-modules:
        "conv.*"  # see example below: can freeze all (top-level) conv layers
        "^((?!classifier).)*$"   # freezes everything except for "classifier": useful for linear probing
        "conv1,layer1,layer2,layer3"  # freeze all layers up to layer3

    >>> model = nn.Sequential(OrderedDict([
          ('conv1', nn.Conv2d(1, 20, 5)),
          ('relu1', nn.ReLU()),
          ('conv2', nn.Conv2d(20, 64, 5)),
          ('relu2', nn.ReLU())
        ]))
    >>> opts = argparse.Namespace(**{"model.freeze_modules": "conv1"})
    >>> _ = freeze_modules_based_on_opts(opts, model)
    INFO    - Freezing module: conv1
    >>> model.train()
    >>> model.conv1.training
    False
    >>> model.conv2.training
    True
    """
    freeze_patterns = getattr(opts, 'model.freeze_modules', '')
    freeze_patterns = clean_strip(freeze_patterns)
    verbose = verbose and is_master(opts)
    if freeze_patterns:
        for name, module in model.named_children():
            if any([re.match(p, name) for p in freeze_patterns]):
                freeze_module(module)
                if verbose:
                    logger.info('Freezing module: {}'.format(name))
        for name, param in model.named_parameters(recurse=False):
            if any([re.match(p, name) for p in freeze_patterns]):
                param.requires_grad = False
                if verbose:
                    logger.info('Freezing parameter: {}'.format(name))
    if verbose and hasattr(model, 'get_trainable_parameters'):
        param_list, _ = model.get_trainable_parameters()
        for params in param_list:
            if not isinstance(params['param_names'], List) or not isinstance(params['params'], List) or not isinstance(params['weight_decay'], (float, int)):
                param_types = {k: type(v) for k, v in params.items()}
                logger.error('Expected parameter format: {{ params: List, weight_decay: float, param_names: List }}. Got: {}'.format(param_types))
        trainable_param_names = [p for x in param_list for p in x['param_names']]
        logger.info('Trainable parameters: {}'.format(trainable_param_names))
    return model


def is_start_rank_node(opts) ->bool:
    node_rank = getattr(opts, 'ddp.rank', 0)
    def_rank = getattr(opts, 'ddp.start_rank', 0)
    return node_rank == def_rank


def load_pretrained_model(model: 'torch.nn.Module', wt_loc: 'str', opts: 'argparse.Namespace', *args, **kwargs) ->torch.nn.Module:
    """Helper function to load pre-trained weights.
    Args:
        model: Model whose weights will be loaded.
        wt_loc: Path to file to load state_dict from.
        opts: Input arguments.
    Returns:
        The model loaded with the given weights.

    """
    if not os.path.isfile(wt_loc):
        logger.error('Pretrained file is not found here: {}'.format(wt_loc))
    wts = torch.load(wt_loc, map_location='cpu')
    is_master_node = is_start_rank_node(opts)
    exclude_scopes = getattr(opts, 'model.resume_exclude_scopes', '')
    exclude_scopes: 'List[str]' = clean_strip(exclude_scopes)
    missing_scopes = getattr(opts, 'model.ignore_missing_scopes', '')
    missing_scopes: 'List[str]' = clean_strip(missing_scopes)
    rename_scopes_map: 'List[List[str]]' = getattr(opts, 'model.rename_scopes_map', [])
    if rename_scopes_map:
        for entry in rename_scopes_map:
            if len(entry) != 2:
                raise ValueError('Every entry in model.rename_scopes_map must contain exactly two string elements for before and after. Got {}.'.format(str(entry)))
    missing_scopes += exclude_scopes
    if exclude_scopes:
        for key in wts.copy():
            if any([re.match(x, key) for x in exclude_scopes]):
                del wts[key]
    if rename_scopes_map:
        for before, after in rename_scopes_map:
            wts = {re.sub(before, after, key): value for key, value in wts.items()}
    strict = not bool(missing_scopes)
    try:
        module = unwrap_model_fn(model)
        missing_keys, unexpected_keys = module.load_state_dict(wts, strict=strict)
        if unexpected_keys:
            raise Exception('Found unexpected keys: {}.You can ignore these keys using `model.resume_exclude_scopes`.'.format(','.join(unexpected_keys)))
        missing_keys = [key for key in missing_keys if not any([re.match(x, key) for x in missing_scopes])]
        if missing_keys:
            raise Exception('Missing keys detected. Did not find the following keys in pre-trained model: {}. You can ignore the keys using `model.ignore_missing_scopes`.'.format(','.join(missing_keys)))
        if is_master_node:
            logger.log('Pretrained weights are loaded from {}'.format(wt_loc))
    except Exception as e:
        if is_master_node:
            logger.error('Unable to load pretrained weights from {}. Error: {}'.format(wt_loc, e))
    return model


class MaskRCNNEncoder(nn.Module):

    def __init__(self, opts: 'argparse.Namespace', encoder: 'BaseImageEncoder', output_strides: 'List', projection_channels: 'int', encoder_lr_multiplier: 'Optional[float]'=1.0, *args, **kwargs) ->None:
        use_fpn = not getattr(opts, 'model.detection.mask_rcnn.disable_fpn', False)
        super().__init__()
        encoder.conv_1x1_exp = Identity()
        encoder.classifier = Identity()
        backbone_proj_layers = nn.ModuleDict()
        self.backbone_output_strides = sorted(list({4, 8, 16, 32}.intersection(output_strides)))
        model_config = encoder.model_conf_dict
        self.backbone_map = {}
        fpn_proj_layers = nn.ModuleDict() if use_fpn else None
        for os in self.backbone_output_strides:
            if os == 4:
                in_channels = model_config['layer2']['out']
                backbone_os_str = 'out_l2'
            elif os == 8:
                in_channels = model_config['layer3']['out']
                backbone_os_str = 'out_l3'
            elif os == 16:
                in_channels = model_config['layer4']['out']
                backbone_os_str = 'out_l4'
            elif os == 32:
                in_channels = model_config['layer5']['out']
                backbone_os_str = 'out_l5'
            else:
                raise NotImplementedError
            conv_layer = ConvLayer2d(opts=opts, in_channels=in_channels, out_channels=projection_channels, kernel_size=1, use_norm=True, use_act=False)
            backbone_proj_layers.add_module(str(os), conv_layer)
            self.backbone_map[os] = backbone_os_str
            if use_fpn:
                fpn_layer = ConvLayer2d(opts=opts, in_channels=projection_channels, out_channels=projection_channels, kernel_size=3, use_norm=True, use_act=False)
                fpn_proj_layers.add_module(str(os), fpn_layer)
        extra_layers = nn.ModuleDict()
        extra_layer_os = sorted(list(set(self.backbone_output_strides) ^ set(output_strides)))
        for os in extra_layer_os:
            conv_layer = ConvLayer2d(opts=opts, in_channels=projection_channels, out_channels=projection_channels, kernel_size=3, stride=2, use_norm=True, use_act=False)
            extra_layers.add_module(str(os), conv_layer)
        self.encoder = encoder
        self.backbone_proj_layers = backbone_proj_layers
        self.fpn_proj_layers = fpn_proj_layers
        self.use_fpn = use_fpn
        self.extra_layers = extra_layers
        self.out_channels = projection_channels
        self.augmented_tensor = None
        self.encoder_lr_multiplier = encoder_lr_multiplier

    def get_augmented_tensor(self) ->Tensor:
        return self.augmented_tensor

    def forward(self, x: 'Tensor') ->Dict[str, Tensor]:
        enc_end_points: 'Dict' = self.encoder.extract_end_points_all(x)
        self.augmented_tensor = enc_end_points.pop('augmented_tensor', None)
        outputs_backbone: 'Dict' = {}
        for os, enc_key_name in self.backbone_map.items():
            x_proj = self.backbone_proj_layers[str(os)](enc_end_points.pop(enc_key_name))
            outputs_backbone[f'{os}'] = x_proj
        if self.fpn_proj_layers:
            last_os = self.backbone_output_strides[-1]
            prev_fm = outputs_backbone[f'{last_os}']
            prev_fm = self.fpn_proj_layers[f'{last_os}'](prev_fm)
            for os in self.backbone_output_strides[:-1][::-1]:
                curr_fm = outputs_backbone[f'{os}']
                feat_shape = curr_fm.shape[-2:]
                inner_top_down = F.interpolate(prev_fm, size=feat_shape, mode='nearest')
                prev_fm = self.fpn_proj_layers[f'{os}'](curr_fm + inner_top_down)
                outputs_backbone[f'{os}'] = prev_fm
        if self.extra_layers:
            prev_os = self.backbone_output_strides[-1]
            for os, extra_layer in self.extra_layers.items():
                x_proj = extra_layer(outputs_backbone[f'{prev_os}'])
                outputs_backbone[f'{os}'] = x_proj
                prev_os = os
        return outputs_backbone

    def get_trainable_parameters(self, weight_decay: 'float'=0.0, no_decay_bn_filter_bias: 'bool'=False, *args, **kwargs) ->Tuple[List, List]:
        module_name = kwargs.pop('module_name', '')
        """Returns a list of trainable parameters"""
        all_params = []
        all_params_lr = []
        if hasattr(self.encoder, 'enable_layer_wise_lr_decay') and self.encoder.enable_layer_wise_lr_decay:
            backbone_param_list, backbone_lr_list = self.encoder.get_trainable_parameters(*args, weight_decay=weight_decay, no_decay_bn_filter_bias=no_decay_bn_filter_bias, module_name=module_name + 'encoder.', **kwargs)
            all_params.extend(backbone_param_list)
            if self.encoder_lr_multiplier != 1.0:
                backbone_lr_list = [(lr * self.encoder_lr_multiplier) for lr in backbone_lr_list]
            all_params_lr.extend(backbone_lr_list)
        else:
            backbone_param_list = parameter_list(*args, named_parameters=self.encoder.named_parameters, weight_decay=weight_decay, no_decay_bn_filter_bias=no_decay_bn_filter_bias, module_name=module_name + 'encoder.', **kwargs)
            all_params.extend(backbone_param_list)
            all_params_lr.extend([self.encoder_lr_multiplier] * len(backbone_param_list))
        if self.backbone_proj_layers:
            projection_param_list = parameter_list(*args, named_parameters=self.backbone_proj_layers.named_parameters, weight_decay=weight_decay, no_decay_bn_filter_bias=no_decay_bn_filter_bias, module_name=module_name + 'backbone_proj_layers.', **kwargs)
            all_params.extend(projection_param_list)
            all_params_lr.extend([1.0] * len(projection_param_list))
        if self.fpn_proj_layers:
            fpn_projection_param_list = parameter_list(*args, named_parameters=self.fpn_proj_layers.named_parameters, weight_decay=weight_decay, no_decay_bn_filter_bias=no_decay_bn_filter_bias, module_name=module_name + 'fpn_proj_layers.', **kwargs)
            all_params.extend(fpn_projection_param_list)
            all_params_lr.extend([1.0] * len(fpn_projection_param_list))
        if self.extra_layers:
            extra_layer_param_list = parameter_list(*args, named_parameters=self.extra_layers.named_parameters, weight_decay=weight_decay, no_decay_bn_filter_bias=no_decay_bn_filter_bias, module_name=module_name + 'extra_layers.', **kwargs)
            all_params.extend(extra_layer_param_list)
            all_params_lr.extend([1.0] * len(extra_layer_param_list))
        return all_params, all_params_lr


def replace_syncbn_with_syncbnfp32(opts, num_features: 'int') ->nn.Module:
    norm_layer = getattr(opts, 'model.normalization.name', None)
    if norm_layer.find('sync') > -1:
        return get_normalization_layer(opts, num_features=num_features, norm_type='sync_batch_norm_fp32')
    else:
        return get_normalization_layer(opts=opts, num_features=num_features)


class FastRCNNConvFCHead(nn.Sequential):

    def __init__(self, opts, input_size: 'Tuple[int, int, int]', conv_layers: 'List[int]', fc_layers: 'List[int]', *args, **kwargs):
        """
        Args:
            input_size (Tuple[int, int, int]): the input size in CHW format.
            conv_layers (list): feature dimensions of each Convolution layer
            fc_layers (list): feature dimensions of each FCN layer
        """
        in_channels, in_height, in_width = input_size
        blocks = []
        previous_channels = in_channels
        for current_channels in conv_layers:
            blocks.extend([ConvLayer2d(opts, in_channels=previous_channels, out_channels=current_channels, kernel_size=3, stride=1, use_norm=False, use_act=False), replace_syncbn_with_syncbnfp32(opts, num_features=current_channels), nn.ReLU(inplace=False)])
            previous_channels = current_channels
        blocks.append(nn.Flatten())
        previous_channels = previous_channels * in_height * in_width
        for current_channels in fc_layers:
            blocks.append(LinearLayer(previous_channels, current_channels, bias=True))
            blocks.append(nn.ReLU(inplace=True))
            previous_channels = current_channels
        super().__init__(*blocks)
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                initialize_conv_layer(module=layer, init_method='kaiming_normal')
            elif isinstance(layer, LinearLayer):
                initialize_fc_layer(module=layer, init_method='kaiming_uniform')


class FastRCNNPredictor(nn.Module):
    """
    Standard classification + bounding box regression layers
    for Fast R-CNN.

    Args:
        in_channels (int): number of input channels
        num_classes (int): number of output classes (including background)
    """

    def __init__(self, in_channels: 'int', num_classes: 'int') ->None:
        super().__init__()
        self.cls_score = LinearLayer(in_channels, num_classes, bias=True)
        self.bbox_pred = LinearLayer(in_channels, num_classes * 4, bias=True)
        for layer in self.modules():
            if isinstance(layer, LinearLayer):
                initialize_fc_layer(module=layer, init_method='kaiming_uniform')

    def forward(self, x: 'Tensor') ->Tuple[Tensor, Tensor]:
        if x.dim() == 4:
            torch._assert(list(x.shape[2:]) == [1, 1], f'x has the wrong shape, expecting the last two dimensions to be [1,1] instead of {list(x.shape[2:])}')
        x = x.flatten(start_dim=1)
        scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)
        return scores, bbox_deltas


class MaskRCNNHeads(nn.Sequential):

    def __init__(self, opts, in_channels: 'int', layers: 'List', dilation: 'int'):
        """
        Args:
            in_channels (int): number of input channels
            layers (list): feature dimensions of each FCN layer
            dilation (int): dilation rate of kernel
            norm_layer (callable, optional): Module specifying the normalization layer to use. Default: None
        """
        blocks = []
        next_feature = in_channels
        for layer_features in layers:
            blocks.extend([ConvLayer2d(opts=opts, in_channels=next_feature, out_channels=layer_features, kernel_size=3, stride=1, dilation=dilation, use_norm=False, use_act=False, bias=False), replace_syncbn_with_syncbnfp32(opts=opts, num_features=layer_features), nn.ReLU(inplace=False)])
            next_feature = layer_features
        super().__init__(*blocks)
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                initialize_conv_layer(module=layer, init_method='kaiming_normal')


class MaskRCNNPredictor(nn.Sequential):

    def __init__(self, opts, in_channels: 'int', dim_reduced: 'int', num_classes: 'int') ->None:
        super().__init__(*[TransposeConvLayer2d(opts, in_channels=in_channels, out_channels=dim_reduced, kernel_size=2, stride=2, padding=0, output_padding=0, use_norm=False, use_act=False, bias=False, groups=1), replace_syncbn_with_syncbnfp32(opts, num_features=dim_reduced), nn.ReLU(inplace=False), ConvLayer2d(opts, in_channels=dim_reduced, out_channels=num_classes, kernel_size=1, stride=1, bias=True, use_norm=False, use_act=False)])
        for layer in self.modules():
            if isinstance(layer, (nn.Conv2d, nn.ConvTranspose2d)):
                initialize_conv_layer(module=layer, init_method='kaiming_normal')


class RPNHead(nn.Module):
    """
    Adds a simple RPN Head with classification and regression heads

    Args:
        in_channels (int): number of channels of the input feature
        num_anchors (int): number of anchors to be predicted
        conv_depth (int, optional): number of convolutions
    """

    def __init__(self, opts, in_channels: 'int', num_anchors: 'int', conv_depth=1) ->None:
        super().__init__()
        convs = []
        for _ in range(conv_depth):
            convs.extend([ConvLayer2d(opts, in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, use_norm=False, use_act=False, bias=False), replace_syncbn_with_syncbnfp32(opts, num_features=in_channels), nn.ReLU(inplace=False)])
        self.conv = nn.Sequential(*convs)
        self.cls_logits = ConvLayer2d(opts, in_channels=in_channels, out_channels=num_anchors, kernel_size=1, stride=1, use_norm=False, use_act=False, bias=True)
        self.bbox_pred = ConvLayer2d(opts, in_channels=in_channels, out_channels=num_anchors * 4, kernel_size=1, stride=1, use_act=False, use_norm=False, bias=True)
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                initialize_conv_layer(module=layer, init_method='normal', std_val=0.01)

    def forward(self, x: 'List[Tensor]') ->Tuple[List[Tensor], List[Tensor]]:
        logits = []
        bbox_reg = []
        for feature in x:
            t = self.conv(feature)
            logits.append(self.cls_logits(t))
            bbox_reg.append(self.bbox_pred(t))
        return logits, bbox_reg


class FeaturePyramidNetwork(BaseModule):
    """
    This class implements the `Feature Pyramid Network <https://arxiv.org/abs/1612.03144>`_ module for object detection.

    Args:
        opts: command-line arguments
        in_channels (List[int]): List of channels at different output strides
        output_strides (List[int]): Feature maps from these output strides will be used in FPN
        out_channels (int): Output channels

    """

    def __init__(self, opts, in_channels: 'List[int]', output_strides: 'List[str]', out_channels: 'int', *args, **kwargs) ->None:
        if isinstance(in_channels, int):
            in_channels = [in_channels]
        if isinstance(output_strides, int):
            output_strides = [output_strides]
        if len(in_channels) != len(output_strides):
            logger.error('For {}, we need the length of input_channels to be the same as the length of output stride. Got: {} and {}'.format(self.__class__.__name__, len(in_channels), len(output_strides)))
        assert len(in_channels) == len(output_strides)
        super().__init__(*args, **kwargs)
        self.proj_layers = nn.ModuleDict()
        self.nxn_convs = nn.ModuleDict()
        for os, in_channel in zip(output_strides, in_channels):
            proj_layer = ConvLayer2d(opts=opts, in_channels=in_channel, out_channels=out_channels, kernel_size=1, bias=False, use_norm=True, use_act=False)
            nxn_conv = ConvLayer2d(opts=opts, in_channels=out_channels, out_channels=out_channels, kernel_size=3, bias=False, use_norm=True, use_act=False)
            self.proj_layers.add_module(name='os_{}'.format(os), module=proj_layer)
            self.nxn_convs.add_module(name='os_{}'.format(os), module=nxn_conv)
        self.num_fpn_layers = len(in_channels)
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.output_strides = output_strides
        self.reset_weights()

    def reset_weights(self) ->None:
        """Resets the weights of FPN layers"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                initialize_conv_layer(m, init_method='xavier_uniform')
            elif isinstance(m, norm_layers_tuple):
                initialize_norm_layers(m)

    def forward(self, x: 'Dict[str, Tensor]', *args, **kwargs) ->Dict[str, Tensor]:
        assert len(x) == self.num_fpn_layers
        fpn_out_dict = {'os_'.format(os): None for os in self.output_strides}
        os_key = 'os_{}'.format(self.output_strides[-1])
        prev_x = self.proj_layers[os_key](x[os_key])
        prev_x = self.nxn_convs[os_key](prev_x)
        fpn_out_dict[os_key] = prev_x
        remaining_output_strides = self.output_strides[:-1]
        for os in remaining_output_strides[::-1]:
            os_key = 'os_{}'.format(os)
            curr_x = self.proj_layers[os_key](x[os_key])
            prev_x = F.interpolate(prev_x, size=curr_x.shape[-2:], mode='nearest')
            prev_x = curr_x + prev_x
            prev_x = self.nxn_convs[os_key](prev_x)
            fpn_out_dict[os_key] = prev_x
        return fpn_out_dict

    def __repr__(self):
        return '{}(in_channels={}, output_strides={} out_channels={})'.format(self.__class__.__name__, self.in_channels, self.output_strides, self.out_channels)


class SSDHead(BaseModule):
    """
    This class defines the `SSD object detection Head <https://arxiv.org/abs/1512.02325>`_

    Args:
        opts: command-line arguments
        in_channels (int): :math:`C` from an expected input of size :math:`(N, C, H, W)`
        n_anchors (int): Number of anchors
        n_classes (int): Number of classes in the dataset
        n_coordinates (Optional[int]): Number of coordinates. Default: 4 (x, y, w, h)
        proj_channels (Optional[int]): Number of projected channels. If `-1`, then projection layer is not used
        kernel_size (Optional[int]): Kernel size in convolutional layer. If kernel_size=1, then standard
            point-wise convolution is used. Otherwise, separable convolution is used
        stride (Optional[int]): stride for feature map. If stride > 1, then feature map is sampled at this rate
            and predictions are made on fewer pixels as compared to the input tensor. Default: 1
    """

    def __init__(self, opts, in_channels: 'int', n_anchors: 'int', n_classes: 'int', n_coordinates: 'Optional[int]'=4, proj_channels: 'Optional[int]'=-1, kernel_size: 'Optional[int]'=3, stride: 'Optional[int]'=1, *args, **kwargs) ->None:
        super().__init__()
        proj_layer = None
        self.proj_channels = None
        if proj_channels != -1 and proj_channels != in_channels and kernel_size > 1:
            proj_layer = ConvLayer2d(opts=opts, in_channels=in_channels, out_channels=proj_channels, kernel_size=1, stride=1, groups=1, bias=False, use_norm=True, use_act=True)
            in_channels = proj_channels
            self.proj_channels = proj_channels
        self.proj_layer = proj_layer
        conv_fn = ConvLayer2d if kernel_size == 1 else SeparableConv2d
        if kernel_size > 1 and stride > 1:
            kernel_size = max(kernel_size, stride if stride % 2 != 0 else stride + 1)
        self.loc_cls_layer = conv_fn(opts=opts, in_channels=in_channels, out_channels=n_anchors * (n_coordinates + n_classes), kernel_size=kernel_size, stride=1, groups=1, bias=True, use_norm=False, use_act=False)
        self.n_coordinates = n_coordinates
        self.n_classes = n_classes
        self.n_anchors = n_anchors
        self.k_size = kernel_size
        self.stride = stride
        self.in_channel = in_channels
        self.reset_parameters()

    def __repr__(self) ->str:
        repr_str = '{}(in_channels={}, n_anchors={}, n_classes={}, n_coordinates={}, kernel_size={}, stride={}'.format(self.__class__.__name__, self.in_channel, self.n_anchors, self.n_classes, self.n_coordinates, self.k_size, self.stride)
        if self.proj_layer is not None:
            repr_str += ', proj=True, proj_channels={}'.format(self.proj_channels)
        repr_str += ')'
        return repr_str

    def reset_parameters(self) ->None:
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                initialize_conv_layer(module=layer, init_method='xavier_uniform')

    def _sample_fm(self, x: 'Tensor') ->Tensor:
        height, width = x.shape[-2:]
        device = x.device
        start_step = max(0, self.stride // 2)
        indices_h = torch.arange(start=start_step, end=height, step=self.stride, dtype=torch.int64, device=device)
        indices_w = torch.arange(start=start_step, end=width, step=self.stride, dtype=torch.int64, device=device)
        x_sampled = torch.index_select(x, dim=-1, index=indices_w)
        x_sampled = torch.index_select(x_sampled, dim=-2, index=indices_h)
        return x_sampled

    def forward(self, x: 'Tensor', *args, **kwargs) ->Tuple[Tensor, Tensor]:
        batch_size = x.shape[0]
        if self.proj_layer is not None:
            x = self.proj_layer(x)
        x = self.loc_cls_layer(x)
        if self.stride > 1:
            x = self._sample_fm(x)
        x = x.permute(0, 2, 3, 1)
        x = x.contiguous().view(batch_size, -1, self.n_coordinates + self.n_classes)
        box_locations, box_classes = torch.split(x, [self.n_coordinates, self.n_classes], dim=-1)
        return box_locations, box_classes


def build_anchor_generator(opts, *args, **kwargs):
    """Build anchor generator for object detection"""
    anchor_gen_name = getattr(opts, 'anchor_generator.name')
    if anchor_gen_name == '__base__':
        logger.error("__base__ can't be used as a projection name. Please check.")
    anchor_gen = ANCHOR_GEN_REGISTRY[anchor_gen_name](opts, *args, **kwargs)
    return anchor_gen


class BaseMatcher(object):
    """
    Base class for matching anchor boxes and labels for the task of object detection
    """

    def __init__(self, opts, *args, **kwargs) ->None:
        super(BaseMatcher, self).__init__()
        self.opts = opts

    @classmethod
    def add_arguments(cls, parser: 'argparse.ArgumentParser'):
        """Add class-specific arguments"""
        return parser

    def __call__(self, *args, **kwargs):
        raise NotImplementedError


def build_matcher(opts, *args, **kwargs):
    matcher_name = getattr(opts, 'matcher.name', None)
    if matcher_name == '__base__':
        logger.error("__base__ can't be used as a projection name. Please check.")
    matcher = MATCHER_REGISTRY[matcher_name](opts, *args, **kwargs)
    return matcher


def is_coreml_conversion(opts) ->bool:
    if getattr(opts, 'common.enable_coreml_compatible_module', False):
        return True
    return False


def _check_out_channels(config: 'dict', layer_name: 'str') ->int:
    enc_ch_l: 'dict' = config.get(layer_name, None)
    if enc_ch_l is None or not enc_ch_l:
        logger.error('Encoder does not define input-output mapping for {}: Got: {}'.format(layer_name, config))
    enc_ch_l_out = enc_ch_l.get('out', None)
    if enc_ch_l_out is None or not enc_ch_l_out:
        logger.error('Output channels are not defined in {} of the encoder. Got: {}'.format(layer_name, enc_ch_l))
    return enc_ch_l_out


class ASPP(BaseModule):
    """
    ASPP module defined in DeepLab papers, `here <https://arxiv.org/abs/1606.00915>`_ and `here <https://arxiv.org/abs/1706.05587>`_

    Args:
        opts: command-line arguments
        in_channels (int): :math:`C_{in}` from an expected input of size :math:`(N, C_{in}, H, W)`
        out_channels (int): :math:`C_{out}` from an expected output of size :math:`(N, C_{out}, H, W)`
        atrous_rates (Tuple[int]): atrous rates for different branches.
        is_sep_conv (Optional[bool]): Use separable convolution instead of standaard conv. Default: False
        dropout (Optional[float]): Apply dropout. Default is 0.0

    Shape:
        - Input: :math:`(N, C_{in}, H, W)`
        - Output: :math:`(N, C_{out}, H, W)`
    """

    def __init__(self, opts, in_channels: 'int', out_channels: 'int', atrous_rates: 'Tuple[int]', is_sep_conv: 'Optional[bool]'=False, dropout: 'Optional[float]'=0.0, *args, **kwargs) ->None:
        in_proj = ConvLayer2d(opts=opts, in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, use_norm=True, use_act=True)
        out_proj = ConvLayer2d(opts=opts, in_channels=5 * out_channels, out_channels=out_channels, kernel_size=1, stride=1, use_norm=True, use_act=True)
        aspp_layer = ASPPSeparableConv2d if is_sep_conv else ASPPConv2d
        assert len(atrous_rates) == 3
        modules = [in_proj]
        modules.extend([aspp_layer(opts=opts, in_channels=in_channels, out_channels=out_channels, dilation=rate) for rate in atrous_rates])
        modules.append(ASPPPooling(opts=opts, in_channels=in_channels, out_channels=out_channels))
        if not 0.0 <= dropout < 1.0:
            if is_master(opts):
                logger.warning('Dropout value in {} should be between 0 and 1. Got: {}. Setting it to 0.0'.format(self.__class__.__name__, dropout))
            dropout = 0.0
        super().__init__()
        self.convs = nn.ModuleList(modules)
        self.project = out_proj
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.atrous_rates = atrous_rates
        self.is_sep_conv_layer = is_sep_conv
        self.n_atrous_branches = len(atrous_rates)
        self.dropout_layer = Dropout2d(p=dropout)

    def forward(self, x: 'Tensor', *args, **kwargs) ->Tensor:
        out = []
        for conv in self.convs:
            out.append(conv(x))
        out = torch.cat(out, dim=1)
        out = self.project(out)
        out = self.dropout_layer(out)
        return out

    def __repr__(self):
        return '{}(in_channels={}, out_channels={}, atrous_rates={}, is_aspp_sep={}, dropout={})'.format(self.__class__.__name__, self.in_channels, self.out_channels, self.atrous_rates, self.is_sep_conv_layer, self.dropout_layer.p)


class PSP(BaseModule):
    """
    This class defines the Pyramid Scene Parsing module in the `PSPNet paper <https://arxiv.org/abs/1612.01105>`_

    Args:
        opts: command-line arguments
        in_channels (int): :math:`C_{in}` from an expected input of size :math:`(N, C_{in}, H, W)`
        out_channels (int): :math:`C_{out}` from an expected output of size :math:`(N, C_{out}, H, W)`
        pool_sizes Optional[Tuple[int, ...]]: List or Tuple of pool sizes. Default: (1, 2, 3, 6)
        dropout (Optional[float]): Apply dropout. Default is 0.0
    """

    def __init__(self, opts, in_channels: 'int', out_channels: 'int', pool_sizes: 'Optional[Tuple[int, ...]]'=(1, 2, 3, 6), dropout: 'Optional[float]'=0.0, *args, **kwargs) ->None:
        if not 0.0 <= dropout < 1.0:
            logger.error('Dropout value in {} should be between 0 and 1. Got: {}'.format(self.__class__.__name__, dropout))
        reduction_dim = in_channels // len(pool_sizes)
        reduction_dim = reduction_dim // 16 * 16
        channels_after_concat = reduction_dim * len(pool_sizes) + in_channels
        super().__init__()
        self.psp_branches = nn.ModuleList([self._make_psp_layer(opts, o_size=ps, in_channels=in_channels, out_channels=reduction_dim) for ps in pool_sizes])
        self.fusion = nn.Sequential(ConvLayer2d(opts=opts, in_channels=channels_after_concat, out_channels=out_channels, kernel_size=3, stride=1, use_norm=True, use_act=True), Dropout2d(p=dropout))
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pool_sizes = pool_sizes
        self.inner_channels = reduction_dim
        self.dropout = dropout

    @staticmethod
    def _make_psp_layer(opts, o_size: 'int', in_channels: 'int', out_channels: 'int') ->nn.Module:
        return nn.Sequential(AdaptiveAvgPool2d(output_size=(o_size, o_size)), ConvLayer2d(opts, in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=False, use_norm=True, use_act=True))

    def forward(self, x: 'Tensor', *args, **kwargs) ->Tensor:
        x_size = x.shape[2:]
        out = [x] + [F.interpolate(input=psp_branch(x), size=x_size, mode='bilinear', align_corners=True) for psp_branch in self.psp_branches]
        out = torch.cat(out, dim=1)
        out = self.fusion(out)
        return out

    def __repr__(self):
        return '{}(in_channels={}, out_channels={}, pool_sizes={}, inner_channels={}, dropout_2d={})'.format(self.__class__.__name__, self.in_channels, self.out_channels, self.pool_sizes, self.inner_channels, self.dropout)


class RepCPE(BaseModule):
    """
    Implementation of reparameterizable conditional positional encoding.
    For more details refer to paper:
    `Conditional Positional Encodings for Vision Transformers <https://arxiv.org/pdf/2102.10882.pdf>`_

    Args:
        opts: Command line arguments.
        in_channels: Number of input channels.
        embed_dim: Number of embedding dimensions. Default: 768
        spatial_shape: Spatial shape of kernel for positional encoding. Default: (7, 7)
        inference_mode: Flag to instantiate block in inference mode. Default: ``False``
    """

    def __init__(self, opts: 'argparse.Namespace', in_channels: 'int', embed_dim: 'int'=768, spatial_shape: 'Union[int, Tuple[int, int]]'=(7, 7), inference_mode: 'bool'=False):
        super(RepCPE, self).__init__()
        if isinstance(spatial_shape, int):
            spatial_shape = tuple([spatial_shape] * 2)
        assert isinstance(spatial_shape, Tuple), f'"spatial_shape" must by a sequence or int, get {type(spatial_shape)} instead.'
        assert len(spatial_shape) == 2, f'Length of "spatial_shape" should be 2, got {len(spatial_shape)} instead.'
        self.spatial_shape = spatial_shape
        self.embed_dim = embed_dim
        self.in_channels = in_channels
        self.groups = embed_dim
        if inference_mode:
            self.reparam_conv = nn.Conv2d(in_channels=self.in_channels, out_channels=self.embed_dim, kernel_size=self.spatial_shape, stride=1, padding=int(self.spatial_shape[0] // 2), groups=self.embed_dim, bias=True)
        else:
            self.pe = ConvLayer2d(opts, in_channels=in_channels, out_channels=embed_dim, kernel_size=spatial_shape, stride=1, padding=int(spatial_shape[0] // 2), use_norm=False, use_act=False, bias=True, groups=embed_dim)

    def forward(self, x: 'torch.Tensor', *args, **kwargs) ->torch.Tensor:
        """
        Forward pass implements inference logic for module
        before and after reparameterization.

        Args:
            x: Input tensor of shape :math:`(B, C, H, W)`.

        Returns:
            torch.Tensor of shape :math:`(B, C, H, W)`.
        """
        if hasattr(self, 'reparam_conv'):
            x = self.reparam_conv(x)
            return x
        else:
            x = self.pe(x) + x
            return x

    def reparameterize(self) ->None:
        """Reparameterize linear branches."""
        input_dim = self.in_channels // self.groups
        kernel_value = torch.zeros((self.in_channels, input_dim, self.spatial_shape[0], self.spatial_shape[1]), dtype=self.pe.block.conv.weight.dtype, device=self.pe.block.conv.weight.device)
        for i in range(self.in_channels):
            kernel_value[i, i % input_dim, self.spatial_shape[0] // 2, self.spatial_shape[1] // 2] = 1
        id_tensor = kernel_value
        w_final = id_tensor + self.pe.block.conv.weight
        b_final = self.pe.block.conv.bias
        self.reparam_conv = nn.Conv2d(in_channels=self.in_channels, out_channels=self.embed_dim, kernel_size=self.spatial_shape, stride=1, padding=int(self.spatial_shape[0] // 2), groups=self.embed_dim, bias=True)
        self.reparam_conv.weight.data = w_final
        self.reparam_conv.bias.data = b_final
        for para in self.parameters():
            para.detach_()
        self.__delattr__('pe')


class LinearAttnFFN(BaseModule):
    """
    This class defines the pre-norm transformer encoder with linear self-attention in `MobileViTv2 <https://arxiv.org/abs/2206.02680>`_ paper
    Args:
        opts: command line arguments
        embed_dim (int): :math:`C_{in}` from an expected input of size :math:`(B, C_{in}, P, N)`
        ffn_latent_dim (int): Inner dimension of the FFN
        attn_dropout (Optional[float]): Dropout rate for attention in multi-head attention. Default: 0.0
        dropout (Optional[float]): Dropout rate. Default: 0.0
        ffn_dropout (Optional[float]): Dropout between FFN layers. Default: 0.0
        norm_layer (Optional[str]): Normalization layer. Default: layer_norm_2d

    Shape:
        - Input: :math:`(B, C_{in}, P, N)` where :math:`B` is batch size, :math:`C_{in}` is input embedding dim,
            :math:`P` is number of pixels in a patch, and :math:`N` is number of patches,
        - Output: same shape as the input
    """

    def __init__(self, opts, embed_dim: 'int', ffn_latent_dim: 'int', attn_dropout: 'Optional[float]'=0.0, dropout: 'Optional[float]'=0.1, ffn_dropout: 'Optional[float]'=0.0, norm_layer: 'Optional[str]'='layer_norm_2d', *args, **kwargs) ->None:
        super().__init__()
        attn_unit = LinearSelfAttention(opts, embed_dim=embed_dim, attn_dropout=attn_dropout, bias=True)
        self.pre_norm_attn = nn.Sequential(get_normalization_layer(opts=opts, norm_type=norm_layer, num_features=embed_dim), attn_unit, Dropout(p=dropout))
        self.pre_norm_ffn = nn.Sequential(get_normalization_layer(opts=opts, norm_type=norm_layer, num_features=embed_dim), ConvLayer2d(opts=opts, in_channels=embed_dim, out_channels=ffn_latent_dim, kernel_size=1, stride=1, bias=True, use_norm=False, use_act=True), Dropout(p=ffn_dropout), ConvLayer2d(opts=opts, in_channels=ffn_latent_dim, out_channels=embed_dim, kernel_size=1, stride=1, bias=True, use_norm=False, use_act=False), Dropout(p=dropout))
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_latent_dim
        self.ffn_dropout = ffn_dropout
        self.std_dropout = dropout
        self.attn_fn_name = attn_unit.__repr__()
        self.norm_name = norm_layer

    def __repr__(self) ->str:
        return '{}(embed_dim={}, ffn_dim={}, dropout={}, ffn_dropout={}, attn_fn={}, norm_layer={})'.format(self.__class__.__name__, self.embed_dim, self.ffn_dim, self.std_dropout, self.ffn_dropout, self.attn_fn_name, self.norm_name)

    def forward(self, x: 'Tensor', x_prev: 'Optional[Tensor]'=None, *args, **kwargs) ->Tensor:
        if x_prev is None:
            x = x + self.pre_norm_attn(x)
        else:
            res = x
            x = self.pre_norm_attn[0](x)
            x = self.pre_norm_attn[1](x, x_prev)
            x = self.pre_norm_attn[2](x)
            x = x + res
        x = x + self.pre_norm_ffn(x)
        return x


class MobileViTBlockv2(BaseModule):
    """
    This class defines the `MobileViTv2 <https://arxiv.org/abs/2206.02680>`_ block

    Args:
        opts: command line arguments
        in_channels (int): :math:`C_{in}` from an expected input of size :math:`(N, C_{in}, H, W)`
        attn_unit_dim (int): Input dimension to the attention unit
        ffn_multiplier (int): Expand the input dimensions by this factor in FFN. Default is 2.
        n_attn_blocks (Optional[int]): Number of attention units. Default: 2
        attn_dropout (Optional[float]): Dropout in multi-head attention. Default: 0.0
        dropout (Optional[float]): Dropout rate. Default: 0.0
        ffn_dropout (Optional[float]): Dropout between FFN layers in transformer. Default: 0.0
        patch_h (Optional[int]): Patch height for unfolding operation. Default: 8
        patch_w (Optional[int]): Patch width for unfolding operation. Default: 8
        conv_ksize (Optional[int]): Kernel size to learn local representations in MobileViT block. Default: 3
        dilation (Optional[int]): Dilation rate in convolutions. Default: 1
        attn_norm_layer (Optional[str]): Normalization layer in the attention block. Default: layer_norm_2d
    """

    def __init__(self, opts, in_channels: 'int', attn_unit_dim: 'int', ffn_multiplier: 'Optional[Union[Sequence[Union[int, float]], int, float]]'=2.0, n_attn_blocks: 'Optional[int]'=2, attn_dropout: 'Optional[float]'=0.0, dropout: 'Optional[float]'=0.0, ffn_dropout: 'Optional[float]'=0.0, patch_h: 'Optional[int]'=8, patch_w: 'Optional[int]'=8, conv_ksize: 'Optional[int]'=3, dilation: 'Optional[int]'=1, attn_norm_layer: 'Optional[str]'='layer_norm_2d', *args, **kwargs) ->None:
        cnn_out_dim = attn_unit_dim
        conv_3x3_in = ConvLayer2d(opts=opts, in_channels=in_channels, out_channels=in_channels, kernel_size=conv_ksize, stride=1, use_norm=True, use_act=True, dilation=dilation, groups=in_channels)
        conv_1x1_in = ConvLayer2d(opts=opts, in_channels=in_channels, out_channels=cnn_out_dim, kernel_size=1, stride=1, use_norm=False, use_act=False)
        super(MobileViTBlockv2, self).__init__()
        self.local_rep = nn.Sequential(conv_3x3_in, conv_1x1_in)
        self.global_rep, attn_unit_dim = self._build_attn_layer(opts=opts, d_model=attn_unit_dim, ffn_mult=ffn_multiplier, n_layers=n_attn_blocks, attn_dropout=attn_dropout, dropout=dropout, ffn_dropout=ffn_dropout, attn_norm_layer=attn_norm_layer)
        self.conv_proj = ConvLayer2d(opts=opts, in_channels=cnn_out_dim, out_channels=in_channels, kernel_size=1, stride=1, use_norm=True, use_act=False)
        self.patch_h = patch_h
        self.patch_w = patch_w
        self.patch_area = self.patch_w * self.patch_h
        self.cnn_in_dim = in_channels
        self.cnn_out_dim = cnn_out_dim
        self.transformer_in_dim = attn_unit_dim
        self.dropout = dropout
        self.attn_dropout = attn_dropout
        self.ffn_dropout = ffn_dropout
        self.n_blocks = n_attn_blocks
        self.conv_ksize = conv_ksize
        self.enable_coreml_compatible_fn = getattr(opts, 'common.enable_coreml_compatible_module', False)
        if self.enable_coreml_compatible_fn:
            self.register_buffer(name='unfolding_weights', tensor=self._compute_unfolding_weights(), persistent=False)

    def _compute_unfolding_weights(self) ->Tensor:
        weights = torch.eye(self.patch_h * self.patch_w, dtype=torch.float)
        weights = weights.reshape((self.patch_h * self.patch_w, 1, self.patch_h, self.patch_w))
        weights = weights.repeat(self.cnn_out_dim, 1, 1, 1)
        return weights

    def _build_attn_layer(self, opts, d_model: 'int', ffn_mult: 'Union[Sequence, int, float]', n_layers: 'int', attn_dropout: 'float', dropout: 'float', ffn_dropout: 'float', attn_norm_layer: 'str', *args, **kwargs) ->Tuple[nn.Module, int]:
        if isinstance(ffn_mult, Sequence) and len(ffn_mult) == 2:
            ffn_dims = np.linspace(ffn_mult[0], ffn_mult[1], n_layers, dtype=float) * d_model
        elif isinstance(ffn_mult, Sequence) and len(ffn_mult) == 1:
            ffn_dims = [ffn_mult[0] * d_model] * n_layers
        elif isinstance(ffn_mult, (int, float)):
            ffn_dims = [ffn_mult * d_model] * n_layers
        else:
            raise NotImplementedError
        ffn_dims = [int(d // 16 * 16) for d in ffn_dims]
        global_rep = [LinearAttnFFN(opts=opts, embed_dim=d_model, ffn_latent_dim=ffn_dims[block_idx], attn_dropout=attn_dropout, dropout=dropout, ffn_dropout=ffn_dropout, norm_layer=attn_norm_layer) for block_idx in range(n_layers)]
        global_rep.append(get_normalization_layer(opts=opts, norm_type=attn_norm_layer, num_features=d_model))
        return nn.Sequential(*global_rep), d_model

    def __repr__(self) ->str:
        repr_str = '{}('.format(self.__class__.__name__)
        repr_str += '\n\t Local representations'
        if isinstance(self.local_rep, nn.Sequential):
            for m in self.local_rep:
                repr_str += '\n\t\t {}'.format(m)
        else:
            repr_str += '\n\t\t {}'.format(self.local_rep)
        repr_str += '\n\t Global representations with patch size of {}x{}'.format(self.patch_h, self.patch_w)
        if isinstance(self.global_rep, nn.Sequential):
            for m in self.global_rep:
                repr_str += '\n\t\t {}'.format(m)
        else:
            repr_str += '\n\t\t {}'.format(self.global_rep)
        if isinstance(self.conv_proj, nn.Sequential):
            for m in self.conv_proj:
                repr_str += '\n\t\t {}'.format(m)
        else:
            repr_str += '\n\t\t {}'.format(self.conv_proj)
        repr_str += '\n)'
        return repr_str

    def unfolding_pytorch(self, feature_map: 'Tensor') ->Tuple[Tensor, Tuple[int, int]]:
        batch_size, in_channels, img_h, img_w = feature_map.shape
        patches = F.unfold(feature_map, kernel_size=(self.patch_h, self.patch_w), stride=(self.patch_h, self.patch_w))
        patches = patches.reshape(batch_size, in_channels, self.patch_h * self.patch_w, -1)
        return patches, (img_h, img_w)

    def folding_pytorch(self, patches: 'Tensor', output_size: 'Tuple[int, int]') ->Tensor:
        batch_size, in_dim, patch_size, n_patches = patches.shape
        patches = patches.reshape(batch_size, in_dim * patch_size, n_patches)
        feature_map = F.fold(patches, output_size=output_size, kernel_size=(self.patch_h, self.patch_w), stride=(self.patch_h, self.patch_w))
        return feature_map

    def unfolding_coreml(self, feature_map: 'Tensor') ->Tuple[Tensor, Tuple[int, int]]:
        batch_size, in_channels, img_h, img_w = feature_map.shape
        patches = F.conv2d(feature_map, self.unfolding_weights, bias=None, stride=(self.patch_h, self.patch_w), padding=0, dilation=1, groups=in_channels)
        patches = patches.reshape(batch_size, in_channels, self.patch_h * self.patch_w, -1)
        return patches, (img_h, img_w)

    def folding_coreml(self, patches: 'Tensor', output_size: 'Tuple[int, int]') ->Tensor:
        batch_size, in_dim, patch_size, n_patches = patches.shape
        n_patches_h = output_size[0] // self.patch_h
        n_patches_w = output_size[1] // self.patch_w
        feature_map = patches.reshape(batch_size, in_dim * self.patch_h * self.patch_w, n_patches_h, n_patches_w)
        assert self.patch_h == self.patch_w, 'For Coreml, we need patch_h and patch_w are the same'
        feature_map = F.pixel_shuffle(feature_map, upscale_factor=self.patch_h)
        return feature_map

    def resize_input_if_needed(self, x):
        batch_size, in_channels, orig_h, orig_w = x.shape
        if orig_h % self.patch_h != 0 or orig_w % self.patch_w != 0:
            new_h = int(math.ceil(orig_h / self.patch_h) * self.patch_h)
            new_w = int(math.ceil(orig_w / self.patch_w) * self.patch_w)
            x = F.interpolate(x, size=(new_h, new_w), mode='bilinear', align_corners=True)
        return x

    def forward_spatial(self, x: 'Tensor', *args, **kwargs) ->Tensor:
        x = self.resize_input_if_needed(x)
        fm = self.local_rep(x)
        if self.enable_coreml_compatible_fn:
            patches, output_size = self.unfolding_coreml(fm)
        else:
            patches, output_size = self.unfolding_pytorch(fm)
        patches = self.global_rep(patches)
        if self.enable_coreml_compatible_fn:
            fm = self.folding_coreml(patches=patches, output_size=output_size)
        else:
            fm = self.folding_pytorch(patches=patches, output_size=output_size)
        fm = self.conv_proj(fm)
        return fm

    def forward_temporal(self, x: 'Tensor', x_prev: 'Tensor', *args, **kwargs) ->Union[Tensor, Tuple[Tensor, Tensor]]:
        x = self.resize_input_if_needed(x)
        fm = self.local_rep(x)
        if self.enable_coreml_compatible_fn:
            patches, output_size = self.unfolding_coreml(fm)
        else:
            patches, output_size = self.unfolding_pytorch(fm)
        for global_layer in self.global_rep:
            if isinstance(global_layer, LinearAttnFFN):
                patches = global_layer(x=patches, x_prev=x_prev)
            else:
                patches = global_layer(patches)
        if self.enable_coreml_compatible_fn:
            fm = self.folding_coreml(patches=patches, output_size=output_size)
        else:
            fm = self.folding_pytorch(patches=patches, output_size=output_size)
        fm = self.conv_proj(fm)
        return fm, patches

    def forward(self, x: 'Union[Tensor, Tuple[Tensor]]', *args, **kwargs) ->Union[Tensor, Tuple[Tensor, Tensor]]:
        if isinstance(x, Tuple) and len(x) == 2:
            return self.forward_temporal(x=x[0], x_prev=x[1])
        elif isinstance(x, Tensor):
            return self.forward_spatial(x)
        else:
            raise NotImplementedError


class SSDInstanceHead(BaseModule):
    """
    Instance segmentation head for SSD model.
    """

    def __init__(self, opts, in_channels: 'int', n_classes: 'Optional[int]'=1, inner_dim: 'Optional[int]'=256, output_stride: 'Optional[int]'=1, output_size: 'Optional[int]'=8, *args, **kwargs) ->None:
        """

        Args:
            opts: command-line arguments
            in_channels (int): :math:`C` from an expected input of size :math:`(N, C, H, W)`
            n_classes (Optional[int]): Number of classes. Default: 1
            inner_dim: (Optional[int]): Inner dimension of the instance head. Default: 256
            output_stride (Optional[int]): Output stride of the feature map. Output stride is the ratio of input to
                the feature map size. Default: 1
            output_size (Optional[int]): Output size of the instances extracted from RoIAlign layer. Default: 8
        """
        super().__init__()
        self.roi_align = RoIAlign(output_size=output_size, spatial_scale=1.0 / output_stride, sampling_ratio=2, aligned=True)
        self.seg_head = nn.Sequential(TransposeConvLayer2d(opts=opts, in_channels=in_channels, out_channels=inner_dim, kernel_size=2, stride=2, bias=True, use_norm=False, use_act=True, auto_padding=False, padding=0, output_padding=0), ConvLayer2d(opts=opts, in_channels=inner_dim, out_channels=n_classes, kernel_size=1, stride=1, use_norm=False, use_act=False, bias=True))
        self.inner_channels = inner_dim
        self.in_channels = in_channels
        self.mask_classes = n_classes
        self.reset_parameters()

    def __repr__(self) ->str:
        return '{}(in_channels={}, up_out_channels={}, n_classes={})'.format(self.__class__.__name__, self.in_channels, self.inner_channels, self.mask_classes)

    def reset_parameters(self) ->None:
        for layer in self.modules():
            if isinstance(layer, (nn.Conv2d, nn.ConvTranspose2d)):
                initialize_conv_layer(module=layer, init_method='kaiming_normal')

    def forward(self, x: 'Tensor', boxes: 'Tensor', *args, **kwargs) ->Tensor:
        rois = self.roi_align(x, boxes)
        rois = self.seg_head(rois)
        return rois


def compute_class_weights(target: 'Tensor', n_classes: 'int', norm_val: 'float'=1.1) ->Tensor:
    """Implementation of a class-weighting scheme, as defined in Section 5.2
    of `ENet <https://arxiv.org/pdf/1606.02147.pdf>`_ paper.

    Args:
        target: Tensor of shape [Batch_size, *] containing values in the range `[0, C)`.
        n_classes: Integer specifying the number of classes :math:`C`
        norm_val: Normalization value. Defaults to 1.1. This value is decided based on the
        `ESPNetv2 paper <https://arxiv.org/abs/1811.11431>`_.
        Link: https://github.com/sacmehta/ESPNetv2/blob/b78e323039908f31347d8ca17f49d5502ef1a594/segmentation/loadData.py#L16

    Returns:
        A :math:`C`-dimensional tensor containing class weights
    """
    class_hist = torch.histc(target.float(), bins=n_classes, min=0, max=n_classes - 1)
    None
    mask_indices = class_hist == 0
    norm_hist = torch.div(class_hist, class_hist.sum())
    None
    norm_hist = torch.add(norm_hist, norm_val)
    class_wts = torch.div(torch.ones_like(class_hist), torch.log(norm_hist))
    class_wts[mask_indices] = 0.0
    return class_wts


def flatten_yaml_as_dict(d, parent_key='', sep='.'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections_abc.MutableMapping):
            items.extend(flatten_yaml_as_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def build_cls_teacher_from_opts(opts: 'argparse.Namespace') ->nn.Module:
    """Helper function to build a classification teacher model from command-line arguments

    Args:
        opts: command-line arguments

    Returns:
        A teacher model
    """
    pretrained_model = getattr(opts, 'teacher.model.classification.pretrained')
    pytest_env = is_test_env()
    if not pytest_env and pretrained_model is None:
        logger.error('For distillation, please specify teacher weights using teacher.model.classification.pretrained')
    teacher_opts = extract_opts_with_prefix_replacement(opts, 'teacher.model.', 'model.')
    return get_model(teacher_opts, category='classification')


def cosine_curriculum(start: 'int', end: 'int', period: 'int') ->Tensor:
    """This function implements cosine curriculum
    Args:
        start: the starting value for the set of points
        end: the ending value for the set of points
        period: size of the constructed tensor

    Returns:
        A float tensor of length period
    """
    curr = [(end + 0.5 * (start - end) * (1 + math.cos(math.pi * i / (period + 1)))) for i in range(period + 1)]
    curr = torch.tensor(curr, dtype=torch.float)
    return curr


def linear_curriculum(start: 'int', end: 'int', period: 'int') ->Tensor:
    """This function implements linear curriculum

    Args:
        start: the starting value for the set of points
        end: the ending value for the set of points
        period: size of the constructed tensor

    Returns:
        A float tensor of length period
    """
    return torch.linspace(start=start, end=end, steps=period + 1, dtype=torch.float)


CURRICULUM_METHOD = {'linear': linear_curriculum, 'cosine': cosine_curriculum}


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (AdaptiveAvgPool2d,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (AvgPool2d,
     lambda: ([], {'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (BaseNeuralAugmentor,
     lambda: ([], {'opts': SimpleNamespace(model.learn_augmentation.lr_multiplier=4)}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (BatchNorm1d,
     lambda: ([], {'num_features': 4}),
     lambda: ([torch.rand([4, 4, 4])], {})),
    (BatchNorm2d,
     lambda: ([], {'num_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (BatchNorm2dFP32,
     lambda: ([], {'num_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (BatchNorm3d,
     lambda: ([], {'num_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4, 4])], {})),
    (Clip,
     lambda: ([], {'min_val': 4, 'max_val': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Conv2d,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (DistributionNeuralAugmentor,
     lambda: ([], {'opts': SimpleNamespace(model.learn_augmentation.lr_multiplier=4, model.learn_augmentation.brightness=4, model.learn_augmentation.contrast=4, model.learn_augmentation.noise=4)}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Dropout,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Dropout2d,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (FixedSampler,
     lambda: ([], {'value': 4}),
     lambda: ([], {})),
    (Flatten,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (GELU,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (GroupNorm,
     lambda: ([], {'num_groups': 1, 'num_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Identity,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (InstanceNorm1d,
     lambda: ([], {'num_features': 4}),
     lambda: ([torch.rand([4, 4, 4])], {})),
    (InstanceNorm2d,
     lambda: ([], {'num_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (LayerNorm,
     lambda: ([], {'normalized_shape': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (LayerNorm2D_NCHW,
     lambda: ([], {'num_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (LayerNormFP32,
     lambda: ([], {'normalized_shape': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (LeakyReLU,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (MaxPool2d,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (PReLU,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ReLU,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ReLU6,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Sigmoid,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Softmax,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Swish,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (SyncBatchNorm,
     lambda: ([], {'num_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (SyncBatchNormFP32,
     lambda: ([], {'num_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Tanh,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (UniformSampler,
     lambda: ([], {'low': 4, 'high': 4}),
     lambda: ([], {})),
]

