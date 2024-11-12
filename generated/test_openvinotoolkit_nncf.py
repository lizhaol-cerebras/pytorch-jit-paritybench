
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


import inspect


from typing import Any


from typing import Dict


import warnings


import torch


from torch import nn


from functools import partial


import numpy as np


import re


from typing import List


from typing import Optional


from sklearn.metrics import accuracy_score


from torchvision import datasets


from torchvision import transforms


from typing import Tuple


from typing import Iterable


from torchvision import models


from typing import Callable


import torchvision


from torchvision.models.detection.ssd import SSD


from torchvision.models.detection.ssd import GeneralizedRCNNTransform


from torchvision.models.detection.anchor_utils import DefaultBoxGenerator


from time import time


import torch.nn as nn


import torch.nn.parallel


import torch.optim


import torch.utils.data


import torch.utils.data.distributed


import torchvision.datasets as datasets


import torchvision.models as models


import torchvision.transforms as transforms


from torch._dynamo.exc import BackendCompilerFailed


from copy import deepcopy


from torch.jit import TracerWarning


import time


from torch.backends import cudnn


from torch.cuda.amp.autocast_mode import autocast


from torch.nn.modules.loss import _Loss


from torch.optim.lr_scheduler import ReduceLROnPlateau


from torchvision.datasets import CIFAR10


from torchvision.datasets import CIFAR100


from torchvision.models import InceptionOutputs


from torch import Tensor


import copy


from torch import distributed as dist


from torch.utils.data import Sampler


import torch.multiprocessing as mp


import torchvision.models


from collections import namedtuple


import torch.nn.functional as F


from torch.utils import model_zoo


from torch.hub import load_state_dict_from_url


from typing import Sequence


from torch.nn import functional as F


from collections import OrderedDict


from numpy import lcm


from torch.optim import SGD


from torch.optim import Adam


from torch.optim.lr_scheduler import ExponentialLR


from torch.optim.lr_scheduler import LambdaLR


from torch.optim.lr_scheduler import MultiStepLR


from torch.optim.lr_scheduler import StepLR


import collections


import logging


from torch.utils import data


from torch.utils.tensorboard import SummaryWriter


from torchvision.ops import nms as torch_nms


from itertools import product


from math import sqrt


from torch.nn import init


import types


from numpy import random


import functools


import torchvision.transforms as T


from torchvision.transforms import ToPILImage


import math


import random


from torchvision import transforms as T


from torchvision.transforms import InterpolationMode


from torchvision.transforms import functional as F


from abc import ABC


from abc import abstractmethod


from typing import Union


from typing import TypeVar


from typing import cast


from collections import deque


from copy import copy


from typing import Deque


from typing import Set


from typing import Type


from collections import Counter


from enum import Enum


import torch.fx


import torch.utils._pytree as pytree


from collections import defaultdict


from torch.ao.quantization.pt2e.duplicate_dq_pass import DuplicateDQPass


from torch.ao.quantization.pt2e.port_metadata_pass import PortNodeMetaForQDQ


from torch.ao.quantization.pt2e.qat_utils import _fold_conv_bn_qat


from torch.ao.quantization.pt2e.utils import _disallow_eval_train


from torch.fx import GraphModule


from torch.fx.passes.infra.pass_manager import PassManager


from torch.ao.quantization.fx.utils import create_getattr_from_value


from torch.ao.quantization.pt2e.utils import fold_bn_weights_into_conv_node


from torch.quantization.fake_quantize import FakeQuantize


from itertools import combinations


from torch.nn import Parameter


from typing import OrderedDict as OrderedDictType


from typing import NoReturn


from torch.utils.data.dataloader import DataLoader


from functools import cmp_to_key


import torch.distributed as dist


from enum import IntEnum


from functools import reduce


import pandas as pd


from torch.nn.functional import _canonical_mask


from torch.nn.functional import _in_projection


from torch.nn.functional import _in_projection_packed


from torch.nn.functional import _mha_shape_check


from torch.nn.functional import _none_or_dtype


from torch.nn.functional import _verify_batch_size


from torch.nn.functional import dropout


from torch.nn.functional import linear


from torch.nn.functional import pad


from torch.nn.functional import scaled_dot_product_attention


from torch.nn.functional import softmax


from itertools import chain


from types import MethodType


from types import TracebackType


from typing import Iterator


from torch.overrides import TorchFunctionMode


from enum import auto


import tensorflow


import tensorflow as tf


from torchvision.transforms import ToTensor


from types import SimpleNamespace


from torch.autograd import Variable


from sklearn.preprocessing import MinMaxScaler


import torch.nn


from torch.nn import Module


from typing import DefaultDict


from typing import Generator


import abc


from inspect import Parameter


from inspect import Signature


from typing import Protocol


from itertools import islice


import torch.utils.cpp_extension


from torch._jit_internal import createResolutionCallbackFromFrame


from torch.jit import is_tracing


from torch.nn import DataParallel


from torch.nn.parallel import DistributedDataParallel


from torch.onnx import OperatorExportTypes


import enum


from torch.utils.cpp_extension import _get_build_directory


from torch.nn.modules.batchnorm import _BatchNorm


from torch.utils.data import DataLoader


from abc import abstractclassmethod


import numbers


from torch.nn.utils.rnn import PackedSequence


from torch.nn.utils.weight_norm import WeightNorm


from torch.distributed import barrier


from torch.nn.parameter import Parameter


from math import isclose


import queue


from torch import optim


from typing import NamedTuple


from string import Template


from typing import TYPE_CHECKING


from torch import distributed


import itertools


from random import randint


from random import seed


import torch.utils


from torch._export import capture_pre_autograd_graph


from numpy.testing import assert_allclose


from torch.fx.passes.graph_drawer import FxGraphDrawer


import torch.ao.quantization


from torch.ao.quantization.observer import MinMaxObserver


from torch.ao.quantization.observer import PerChannelMinMaxObserver


from torch.utils.data import Dataset


import torchvision.transforms.functional as F


from torch.nn.utils.rnn import pack_padded_sequence


from torch.nn.utils.rnn import pad_packed_sequence


from torch.nn.functional import log_softmax


from torch.nn.utils import weight_norm


from torchvision.models import resnet50


from abc import ABCMeta


from torch.quantization import FakeQuantize


from torch import autocast


from torchvision.models import squeezenet1_1


from random import random


from torch.distributions.uniform import Uniform


from numpy.random import random_sample


from torchvision.transforms import transforms


from scipy.special import softmax


import torch.cuda


from torch import cuda


import torch.nn.functional


from torch.nn import AvgPool2d


from torch.nn import BatchNorm2d


from torch.nn import Conv2d


from torch.nn import MaxPool2d


from torch.nn import ReLU


from torch.nn import Sequential


from math import ceil


from collections.abc import Iterable


from torch import Size


from torch.nn import Dropout


from torchvision.transforms.functional import normalize


from torch.nn.parallel import DistributedDataParallel as DDP


class ConvBN(nn.Module):

    def __init__(self, in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1, dilation=1, bias=False):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels, momentum=0.05)

    def forward(self, inputs):
        x = self.conv(inputs)
        x = self.bn(x)
        return x


class ConvBNReLU(nn.Module):

    def __init__(self, in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1, dilation=1, bias=False):
        super().__init__()
        self.convbn = ConvBN(in_channels, out_channels, kernel_size, stride, padding, dilation, bias)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs):
        x = self.convbn(inputs)
        x = self.relu(x)
        return x


class Conv2dNormActivation(nn.Sequential):

    def __init__(self, in_planes: 'int', out_planes: 'int', kernel_size: 'int'=3, stride: 'int'=1, groups: 'int'=1, norm_layer: 'Optional[Callable[..., nn.Module]]'=None, activation_layer: 'Optional[Callable[..., nn.Module]]'=None, dilation: 'int'=1) ->None:
        padding = (kernel_size - 1) // 2 * dilation
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if activation_layer is None:
            activation_layer = nn.ReLU6
        super(ConvBNReLU, self).__init__(nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, dilation=dilation, groups=groups, bias=False), norm_layer(out_planes), activation_layer(inplace=True))
        self.out_channels = out_planes


class InvertedResidual(nn.Module):

    def __init__(self, inp, hidden_dim, oup, kernel_size, stride, se_layer):
        super().__init__()
        assert stride in [1, 2]
        self.identity = stride == 1 and inp == oup
        self.conv = nn.Sequential(nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, (kernel_size - 1) // 2, groups=hidden_dim, bias=False), nn.BatchNorm2d(hidden_dim), nn.ReLU(inplace=True), se_layer(hidden_dim), nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False), nn.BatchNorm2d(oup))

    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        return self.conv(x)


SPARSITY_TYPES = ['magnitude', 'rb', 'const']


SPARSITY_ALGOS = {'_'.join([type, 'sparsity']) for type in SPARSITY_TYPES}


def tensor_guard(func: 'callable'):
    """
    A decorator that ensures that the first argument to the decorated function is a Tensor.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if isinstance(args[0], Tensor):
            return func(*args, **kwargs)
        raise NotImplementedError(f'Function `{func.__name__}` is not implemented for {type(args[0])}')
    return wrapper


@functools.singledispatch
@tensor_guard
def max(a: 'Tensor', axis: 'Optional[Union[int, Tuple[int, ...]]]'=None, keepdims: 'bool'=False) ->Tensor:
    """
    Return the maximum of an array or maximum along an axis.

    :param a: The input tensor.
    :param axis: Axis or axes along which to operate. By default, flattened input is used.
    :param keepdim: If this is set to True, the axes which are reduced are left in the result as dimensions with size
        one. With this option, the result will broadcast correctly against the input array. False, by default.
    :return: Maximum of a.
    """
    return Tensor(max(a.data, axis, keepdims))


def _make_divisible(v: 'float', divisor: 'int', min_value: 'Optional[int]'=None) ->int:
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class MobileNetV2For32x32(nn.Module):

    def __init__(self, num_classes: 'int'=100, width_mult: 'float'=1.0, inverted_residual_setting: 'Optional[List[List[int]]]'=None, round_nearest: 'int'=8, block: 'Optional[Callable[..., nn.Module]]'=None, norm_layer: 'Optional[Callable[..., nn.Module]]'=None) ->None:
        """
        A special version for MobileNet that is supposed to give solid
        accuracy on CIFAR100 (~68% top1). The differences from the
        regular ImageNet version are:
        1) stride 1, kernel size 1 instead of stride 2, kernel size 3 for the first conv
        2) tcns config [6, 160, 3, 1] for the second-to-last feature block instead of [6, 160, 3, 2]
        (e.g. stride 1 vs stride 2)


        Args:
            num_classes (int): Number of classes
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
            block: Module specifying inverted residual building block for mobilenet
            norm_layer: Module specifying the normalization layer to use

        """
        super().__init__()
        if block is None:
            block = InvertedResidual
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        input_channel = 32
        last_channel = 1280
        if inverted_residual_setting is None:
            inverted_residual_setting = [[1, 16, 1, 1], [6, 24, 2, 2], [6, 32, 3, 2], [6, 64, 4, 2], [6, 96, 3, 1], [6, 160, 3, 1], [6, 320, 1, 1]]
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError('inverted_residual_setting should be non-empty or a 4-element list, got {}'.format(inverted_residual_setting))
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        features: 'List[nn.Module]' = [ConvBNReLU(3, input_channel, kernel_size=1, stride=1, norm_layer=norm_layer)]
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t, norm_layer=norm_layer))
                input_channel = output_channel
        features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1, norm_layer=norm_layer))
        self.features = nn.Sequential(*features)
        self.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(self.last_channel, num_classes))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def _forward_impl(self, x: 'Tensor') ->Tensor:
        x = self.features(x)
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1)).reshape(x.shape[0], -1)
        x = self.classifier(x)
        return x

    def forward(self, x: 'Tensor') ->Tensor:
        return self._forward_impl(x)


class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


class InceptionA(nn.Module):

    def __init__(self, in_channels, pool_features):
        super().__init__()
        self.branch1x1 = BasicConv2d(in_channels, 64, kernel_size=1)
        self.branch5x5_1 = BasicConv2d(in_channels, 48, kernel_size=1)
        self.branch5x5_2 = BasicConv2d(48, 64, kernel_size=5, padding=2)
        self.branch3x3dbl_1 = BasicConv2d(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = BasicConv2d(96, 96, kernel_size=3, padding=1)
        self.branch_pool = BasicConv2d(in_channels, pool_features, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)
        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)
        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)
        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionAux(nn.Module):

    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.conv0 = BasicConv2d(in_channels, 128, kernel_size=1)
        self.conv1 = BasicConv2d(128, 768, kernel_size=5)
        self.conv1.stddev = 0.01
        self.fc = nn.Linear(768, num_classes)
        self.fc.stddev = 0.001

    def forward(self, x):
        x = F.avg_pool2d(x, kernel_size=5, stride=3)
        x = self.conv0(x)
        x = self.conv1(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class InceptionB(nn.Module):

    def __init__(self, in_channels):
        super().__init__()
        self.branch3x3 = BasicConv2d(in_channels, 384, kernel_size=3, stride=2)
        self.branch3x3dbl_1 = BasicConv2d(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = BasicConv2d(96, 96, kernel_size=3, stride=2)

    def forward(self, x):
        branch3x3 = self.branch3x3(x)
        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)
        branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)
        outputs = [branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionC(nn.Module):

    def __init__(self, in_channels, channels_7x7):
        super().__init__()
        self.branch1x1 = BasicConv2d(in_channels, 192, kernel_size=1)
        c7 = channels_7x7
        self.branch7x7_1 = BasicConv2d(in_channels, c7, kernel_size=1)
        self.branch7x7_2 = BasicConv2d(c7, c7, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7_3 = BasicConv2d(c7, 192, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7dbl_1 = BasicConv2d(in_channels, c7, kernel_size=1)
        self.branch7x7dbl_2 = BasicConv2d(c7, c7, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7dbl_3 = BasicConv2d(c7, c7, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7dbl_4 = BasicConv2d(c7, c7, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7dbl_5 = BasicConv2d(c7, 192, kernel_size=(1, 7), padding=(0, 3))
        self.branch_pool = BasicConv2d(in_channels, 192, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch7x7 = self.branch7x7_1(x)
        branch7x7 = self.branch7x7_2(branch7x7)
        branch7x7 = self.branch7x7_3(branch7x7)
        branch7x7dbl = self.branch7x7dbl_1(x)
        branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_4(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_5(branch7x7dbl)
        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)
        outputs = [branch1x1, branch7x7, branch7x7dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionD(nn.Module):

    def __init__(self, in_channels):
        super().__init__()
        self.branch3x3_1 = BasicConv2d(in_channels, 192, kernel_size=1)
        self.branch3x3_2 = BasicConv2d(192, 320, kernel_size=3, stride=2)
        self.branch7x7x3_1 = BasicConv2d(in_channels, 192, kernel_size=1)
        self.branch7x7x3_2 = BasicConv2d(192, 192, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7x3_3 = BasicConv2d(192, 192, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7x3_4 = BasicConv2d(192, 192, kernel_size=3, stride=2)

    def forward(self, x):
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)
        branch7x7x3 = self.branch7x7x3_1(x)
        branch7x7x3 = self.branch7x7x3_2(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_3(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_4(branch7x7x3)
        branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)
        outputs = [branch3x3, branch7x7x3, branch_pool]
        return torch.cat(outputs, 1)


class InceptionE(nn.Module):

    def __init__(self, in_channels):
        super().__init__()
        self.branch1x1 = BasicConv2d(in_channels, 320, kernel_size=1)
        self.branch3x3_1 = BasicConv2d(in_channels, 384, kernel_size=1)
        self.branch3x3_2a = BasicConv2d(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3_2b = BasicConv2d(384, 384, kernel_size=(3, 1), padding=(1, 0))
        self.branch3x3dbl_1 = BasicConv2d(in_channels, 448, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(448, 384, kernel_size=3, padding=1)
        self.branch3x3dbl_3a = BasicConv2d(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3dbl_3b = BasicConv2d(384, 384, kernel_size=(3, 1), padding=(1, 0))
        self.branch_pool = BasicConv2d(in_channels, 192, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [self.branch3x3_2a(branch3x3), self.branch3x3_2b(branch3x3)]
        branch3x3 = torch.cat(branch3x3, 1)
        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [self.branch3x3dbl_3a(branch3x3dbl), self.branch3x3dbl_3b(branch3x3dbl)]
        branch3x3dbl = torch.cat(branch3x3dbl, 1)
        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)
        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class Inception3(nn.Module):

    def __init__(self, num_classes=1000, aux_logits=True, transform_input=False):
        super().__init__()
        self.aux_logits = aux_logits
        self.transform_input = transform_input
        self.Conv2d_1a_3x3 = BasicConv2d(3, 32, kernel_size=3, stride=2)
        self.Conv2d_2a_3x3 = BasicConv2d(32, 32, kernel_size=3)
        self.Conv2d_2b_3x3 = BasicConv2d(32, 64, kernel_size=3, padding=1)
        self.Conv2d_3b_1x1 = BasicConv2d(64, 80, kernel_size=1)
        self.Conv2d_4a_3x3 = BasicConv2d(80, 192, kernel_size=3)
        self.Mixed_5b = InceptionA(192, pool_features=32)
        self.Mixed_5c = InceptionA(256, pool_features=64)
        self.Mixed_5d = InceptionA(288, pool_features=64)
        self.Mixed_6a = InceptionB(288)
        self.Mixed_6b = InceptionC(768, channels_7x7=128)
        self.Mixed_6c = InceptionC(768, channels_7x7=160)
        self.Mixed_6d = InceptionC(768, channels_7x7=160)
        self.Mixed_6e = InceptionC(768, channels_7x7=192)
        if aux_logits:
            self.AuxLogits = InceptionAux(768, num_classes)
        self.Mixed_7a = InceptionD(768)
        self.Mixed_7b = InceptionE(1280)
        self.Mixed_7c = InceptionE(2048)
        self.fc = nn.Linear(2048, num_classes)
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                from scipy import stats
                stddev = m.stddev if hasattr(m, 'stddev') else 0.1
                X = stats.truncnorm(-2, 2, scale=stddev)
                values = torch.Tensor(X.rvs(m.weight.numel()))
                values = values.view(m.weight.size())
                m.weight.data.copy_(values)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        if self.transform_input:
            x_ch0 = torch.unsqueeze(x[:, 0], 1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x_ch1 = torch.unsqueeze(x[:, 1], 1) * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x_ch2 = torch.unsqueeze(x[:, 2], 1) * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
            x = torch.cat((x_ch0, x_ch1, x_ch2), 1)
        x = self.Conv2d_1a_3x3(x)
        x = self.Conv2d_2a_3x3(x)
        x = self.Conv2d_2b_3x3(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = self.Conv2d_3b_1x1(x)
        x = self.Conv2d_4a_3x3(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = self.Mixed_5b(x)
        x = self.Mixed_5c(x)
        x = self.Mixed_5d(x)
        x = self.Mixed_6a(x)
        x = self.Mixed_6b(x)
        x = self.Mixed_6c(x)
        x = self.Mixed_6d(x)
        x = self.Mixed_6e(x)
        if self.training and self.aux_logits:
            aux = self.AuxLogits(x)
        x = self.Mixed_7a(x)
        x = self.Mixed_7b(x)
        x = self.Mixed_7c(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = F.dropout(x, training=self.training)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        if self.training and self.aux_logits:
            return x, aux
        return x


class ConvBNActivation(nn.Sequential):

    def __init__(self, in_planes: 'int', out_planes: 'int', kernel_size: 'int'=3, stride: 'int'=1, groups: 'int'=1, norm_layer: 'Optional[Callable[..., nn.Module]]'=None, activation_layer: 'Optional[Callable[..., nn.Module]]'=None, dilation: 'int'=1) ->None:
        padding = (kernel_size - 1) // 2 * dilation
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if activation_layer is None:
            activation_layer = nn.ReLU6
        super(ConvBNReLU, self).__init__(nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, dilation=dilation, groups=groups, bias=False), norm_layer(out_planes), activation_layer(inplace=True))
        self.out_channels = out_planes


class SqueezeExcitation(nn.Module):

    def __init__(self, input_channels: 'int', squeeze_factor: 'int'=4):
        super().__init__()
        squeeze_channels = _make_divisible(input_channels // squeeze_factor, 8)
        self.fc1 = nn.Conv2d(input_channels, squeeze_channels, 1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(squeeze_channels, input_channels, 1)

    def _scale(self, input_: 'Tensor', inplace: 'bool') ->Tensor:
        scale = F.adaptive_avg_pool2d(input_, 1)
        scale = self.fc1(scale)
        scale = self.relu(scale)
        scale = self.fc2(scale)
        return F.hardsigmoid(scale, inplace=inplace)

    def forward(self, input_: 'Tensor') ->Tensor:
        scale = self._scale(input_, True)
        return scale * input_


class InvertedResidualConfig:

    def __init__(self, input_channels: 'int', kernel: 'int', expanded_channels: 'int', out_channels: 'int', use_se: 'bool', activation: 'str', stride: 'int', dilation: 'int', width_mult: 'float'):
        self.input_channels = self.adjust_channels(input_channels, width_mult)
        self.kernel = kernel
        self.expanded_channels = self.adjust_channels(expanded_channels, width_mult)
        self.out_channels = self.adjust_channels(out_channels, width_mult)
        self.use_se = use_se
        self.use_hs = activation == 'HS'
        self.stride = stride
        self.dilation = dilation

    @staticmethod
    def adjust_channels(channels: 'int', width_mult: 'float'):
        return _make_divisible(channels * width_mult, 8)


@functools.singledispatch
@tensor_guard
def all(a: 'Tensor', axis: 'Optional[Union[int, Tuple[int, ...]]]'=None) ->Tensor:
    """
    Test whether all tensor elements along a given axis evaluate to True.

    :param a: The input tensor.
    :param axis: Axis or axes along which a logical AND reduction is performed.
    :return: A new tensor.
    """
    return Tensor(all(a.data, axis=axis))


class MobileNetV3(nn.Module):

    def __init__(self, inverted_residual_setting: 'List[InvertedResidualConfig]', last_channel: 'int', num_classes: 'int'=1000, block: 'Optional[Callable[..., nn.Module]]'=None, norm_layer: 'Optional[Callable[..., nn.Module]]'=None) ->None:
        """
        MobileNet V3 main class

        Args:
            inverted_residual_setting (List[InvertedResidualConfig]): Network structure
            last_channel (int): The number of channels on the penultimate layer
            num_classes (int): Number of classes
            block (Optional[Callable[..., nn.Module]]): Module specifying inverted residual building block for mobilenet
            norm_layer (Optional[Callable[..., nn.Module]]): Module specifying the normalization layer to use
        """
        super().__init__()
        if not inverted_residual_setting:
            raise ValueError('The inverted_residual_setting should not be empty')
        if not (isinstance(inverted_residual_setting, Sequence) and all(isinstance(s, InvertedResidualConfig) for s in inverted_residual_setting)):
            raise TypeError('The inverted_residual_setting should be List[InvertedResidualConfig]')
        if block is None:
            block = InvertedResidual
        if norm_layer is None:
            norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.01)
        layers: 'List[nn.Module]' = []
        firstconv_output_channels = inverted_residual_setting[0].input_channels
        layers.append(ConvBNActivation(3, firstconv_output_channels, kernel_size=3, stride=2, norm_layer=norm_layer, activation_layer=nn.Hardswish))
        for cnf in inverted_residual_setting:
            layers.append(block(cnf, norm_layer))
        lastconv_input_channels = inverted_residual_setting[-1].out_channels
        lastconv_output_channels = 6 * lastconv_input_channels
        layers.append(ConvBNActivation(lastconv_input_channels, lastconv_output_channels, kernel_size=1, norm_layer=norm_layer, activation_layer=nn.Hardswish))
        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Linear(lastconv_output_channels, last_channel), nn.Hardswish(inplace=True), nn.Dropout(p=0.2, inplace=True), nn.Linear(last_channel, num_classes))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def _forward_impl(self, x: 'Tensor') ->Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def forward(self, x: 'Tensor') ->Tensor:
        return self._forward_impl(x)


class ShuffleBlock(nn.Module):

    def __init__(self, groups=2):
        super().__init__()
        self.groups = groups

    def forward(self, x):
        """Channel shuffle: [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,w] -> [N,C,H,W]"""
        N, C, H, W = x.size()
        g = self.groups
        return x.view(N, g, C // g, H, W).permute(0, 2, 1, 3, 4).reshape(N, C, H, W)


class SplitBlock(nn.Module):

    def __init__(self, ratio):
        super().__init__()
        self.ratio = ratio

    def forward(self, x):
        c = int(x.size(1) * self.ratio)
        return x[:, :c, :, :], x[:, c:, :, :]


class BasicBlock(nn.Module):

    def __init__(self, in_channels, split_ratio=0.5):
        super().__init__()
        self.split = SplitBlock(split_ratio)
        in_channels = int(in_channels * split_ratio)
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, groups=in_channels, bias=False)
        self.bn2 = nn.BatchNorm2d(in_channels)
        self.conv3 = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(in_channels)
        self.shuffle = ShuffleBlock()

    def forward(self, x):
        x1, x2 = self.split(x)
        out = F.relu(self.bn1(self.conv1(x2)))
        out = self.bn2(self.conv2(out))
        out = F.relu(self.bn3(self.conv3(out)))
        out = torch.cat([x1, out], 1)
        out = self.shuffle(out)
        return out


class Bottleneck(nn.Module):

    def __init__(self, in_planes, out_planes, stride, groups):
        super().__init__()
        self.stride = stride
        mid_planes = out_planes // 4
        g = 1 if in_planes == 24 else groups
        self.conv1 = nn.Conv2d(in_planes, mid_planes, kernel_size=1, groups=g, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_planes)
        self.shuffle1 = ShuffleBlock(groups=g)
        self.conv2 = nn.Conv2d(mid_planes, mid_planes, kernel_size=3, stride=stride, padding=1, groups=mid_planes, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_planes)
        self.conv3 = nn.Conv2d(mid_planes, out_planes, kernel_size=1, groups=groups, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)
        self.shortcut = nn.Sequential()
        if stride == 2:
            self.shortcut = nn.Sequential(nn.AvgPool2d(3, stride=2, padding=1))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.shuffle1(out)
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        res = self.shortcut(x)
        out = F.relu(torch.cat([out, res], 1)) if self.stride == 2 else F.relu(out + res)
        return out


class ResNet(nn.Module):

    def __init__(self, block, num_blocks, num_classes=10):
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for block_stride in strides:
            layers.append(block(self.in_planes, planes, block_stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.linear(out)
        return out


class RMBlock(nn.Module):

    def __init__(self, input_planes, squeeze_planes, output_planes, downsample=False, dropout_ratio=0.1, activation=nn.ELU):
        super().__init__()
        self.downsample = downsample
        self.input_planes = input_planes
        self.output_planes = output_planes
        self.squeeze_conv = nn.Conv2d(input_planes, squeeze_planes, kernel_size=1, bias=False)
        self.squeeze_bn = nn.BatchNorm2d(squeeze_planes)
        self.dw_conv = nn.Conv2d(squeeze_planes, squeeze_planes, groups=squeeze_planes, kernel_size=3, padding=1, stride=2 if downsample else 1, bias=False)
        self.dw_bn = nn.BatchNorm2d(squeeze_planes)
        self.expand_conv = nn.Conv2d(squeeze_planes, output_planes, kernel_size=1, bias=False)
        self.expand_bn = nn.BatchNorm2d(output_planes)
        self.activation = activation(inplace=True)
        self.dropout_ratio = dropout_ratio
        if self.downsample:
            self.skip_conv = nn.Conv2d(input_planes, output_planes, kernel_size=1, bias=False)
            self.skip_conv_bn = nn.BatchNorm2d(output_planes)
        self.init_weights()

    def init_weights(self):
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        residual = x
        out = self.activation(self.squeeze_bn(self.squeeze_conv(x)))
        out = self.activation(self.dw_bn(self.dw_conv(out)))
        out = self.expand_bn(self.expand_conv(out))
        if self.dropout_ratio > 0:
            out = F.dropout(out, p=self.dropout_ratio, training=self.training, inplace=True)
        if self.downsample:
            residual = F.max_pool2d(x, kernel_size=2, stride=2, padding=0)
            residual = self.skip_conv(residual)
            residual = self.skip_conv_bn(residual)
        out += residual
        return self.activation(out)


class RMNetBody(nn.Module):

    def __init__(self, block=RMBlock, blocks_per_stage=(None, 4, 8, 10, 11), trunk_width=(32, 32, 64, 128, 256), bottleneck_width=(None, 8, 16, 32, 64)):
        super().__init__()
        assert len(blocks_per_stage) == len(trunk_width) == len(bottleneck_width)
        self.dim_out = trunk_width[-1]
        stages = [nn.Sequential(OrderedDict([('data_bn', nn.BatchNorm2d(3)), ('conv1', nn.Conv2d(3, trunk_width[0], kernel_size=3, stride=2, padding=1, bias=False)), ('bn1', nn.BatchNorm2d(trunk_width[0])), ('relu1', nn.ReLU(inplace=True))]))]
        for i, (blocks_num, w, wb) in enumerate(zip(blocks_per_stage, trunk_width, bottleneck_width)):
            if i == 0:
                continue
            stage = []
            if i > 1:
                stage.append(block(trunk_width[i - 1], wb, w, downsample=True))
            for _ in range(blocks_num):
                stage.append(block(w, wb, w))
            stages.append(nn.Sequential(*stage))
        self.stages = nn.Sequential(OrderedDict([('stage_{}'.format(i), stage) for i, stage in enumerate(stages)]))
        self.init_weights()

    def init_weights(self):
        m = self.stages[0][0]
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
        m = self.stages[0][1]
        nn.init.kaiming_normal_(m.weight, mode='fan_out')
        m = self.stages[0][2]
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.stages(x)


class RMNetClassifierCifar(nn.Module):

    def __init__(self, num_classes, pretrained=False, body=RMNetBody, dropout_ratio=0.1):
        super().__init__()
        self.dropout_ratio = dropout_ratio
        self.backbone = body()
        self.extra_conv = nn.Conv2d(256, 512, 3, stride=2, padding=1, bias=False)
        self.extra_conv_bn = nn.BatchNorm2d(512)
        self.extra_conv_2 = nn.Conv2d(512, 1024, 3, stride=2, padding=1, bias=False)
        self.extra_conv_2_bn = nn.BatchNorm2d(1024)
        self.fc = nn.Conv2d(1024, num_classes, 1, stride=1, padding=0)

    def forward(self, x):
        x = self.backbone(x)
        x = F.elu(self.extra_conv_bn(self.extra_conv(x)))
        x = F.relu(self.extra_conv_2_bn(self.extra_conv_2(x)))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = self.fc(x)
        x = x.view(-1, x.size(1))
        return x


VGG_CONFIGS = {'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'], 'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'], 'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'], 'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']}


class VGG(nn.Module):

    def __init__(self, vgg_name):
        super().__init__()
        self.features = self._make_layers(VGG_CONFIGS[vgg_name])
        self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, config):
        layers = []
        in_channels = 3
        for x in config:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1), nn.BatchNorm2d(x), nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


class InitialBlock(nn.Module):
    """The initial block is composed of two branches:
    1. a main branch which performs a regular convolution with stride 2;
    2. an extension branch which performs max-pooling.

    Doing both operations in parallel and concatenating their results
    allows for efficient downsampling and expansion. The main branch
    outputs 13 feature maps while the extension branch outputs 3, for a
    total of 16 feature maps after concatenation.

    Keyword arguments:
    - in_channels (int): the number of input channels.
    - out_channels (int): the number output channels.
    - kernel_size (int, optional): the kernel size of the filters used in
    the convolution layer. Default: 3.
    - padding (int, optional): zero-padding added to both sides of the
    input. Default: 0.
    - bias (bool, optional): Adds a learnable bias to the output if
    ``True``. Default: False.
    - relu (bool, optional): When ``True`` ReLU is used as the activation
    function; otherwise, PReLU is used. Default: True.

    """

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=0, bias=False, relu=True):
        super().__init__()
        if relu:
            activation = nn.ReLU()
        else:
            activation = nn.PReLU()
        self.main_branch = nn.Conv2d(in_channels, out_channels - 3, kernel_size=kernel_size, stride=2, padding=padding, bias=bias)
        self.ext_branch = nn.MaxPool2d(kernel_size, stride=2, padding=padding)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.out_prelu = activation

    def forward(self, x):
        main = self.main_branch(x)
        ext = self.ext_branch(x)
        out = torch.cat((main, ext), 1)
        out = self.batch_norm(out)
        return self.out_prelu(out)


class RegularBottleneck(nn.Module):
    """Regular bottlenecks are the main building block of ENet.
    Main branch:
    1. Shortcut connection.

    Extension branch:
    1. 1x1 convolution which decreases the number of channels by
    ``internal_ratio``, also called a projection;
    2. regular, dilated or asymmetric convolution;
    3. 1x1 convolution which increases the number of channels back to
    ``channels``, also called an expansion;
    4. dropout as a regularizer.

    Keyword arguments:
    - channels (int): the number of input and output channels.
    - internal_ratio (int, optional): a scale factor applied to
    ``channels`` used to compute the number of
    channels after the projection. eg. given ``channels`` equal to 128 and
    internal_ratio equal to 2 the number of channels after the projection
    is 64. Default: 4.
    - kernel_size (int, optional): the kernel size of the filters used in
    the convolution layer described above in item 2 of the extension
    branch. Default: 3.
    - padding (int, optional): zero-padding added to both sides of the
    input. Default: 0.
    - dilation (int, optional): spacing between kernel elements for the
    convolution described in item 2 of the extension branch. Default: 1.
    asymmetric (bool, optional): flags if the convolution described in
    item 2 of the extension branch is asymmetric or not. Default: False.
    - dropout_prob (float, optional): probability of an element to be
    zeroed. Default: 0 (no dropout).
    - bias (bool, optional): Adds a learnable bias to the output if
    ``True``. Default: False.
    - relu (bool, optional): When ``True`` ReLU is used as the activation
    function; otherwise, PReLU is used. Default: True.

    """

    def __init__(self, channels, internal_ratio=4, kernel_size=3, padding=0, dilation=1, asymmetric=False, dropout_prob=0, bias=False, relu=True):
        super().__init__()
        if internal_ratio <= 1 or internal_ratio > channels:
            raise nncf.ValidationError('Value out of range. Expected value in the interval [1, {0}], got internal_scale={1}.'.format(channels, internal_ratio))
        internal_channels = channels // internal_ratio
        if relu:
            activation = nn.ReLU()
        else:
            activation = nn.PReLU()
        self.ext_conv1 = nn.Sequential(nn.Conv2d(channels, internal_channels, kernel_size=1, stride=1, bias=bias), nn.BatchNorm2d(internal_channels), activation)
        if asymmetric:
            self.ext_conv2 = nn.Sequential(nn.Conv2d(internal_channels, internal_channels, kernel_size=(kernel_size, 1), stride=1, padding=(padding, 0), dilation=dilation, bias=bias), nn.BatchNorm2d(internal_channels), activation, nn.Conv2d(internal_channels, internal_channels, kernel_size=(1, kernel_size), stride=1, padding=(0, padding), dilation=dilation, bias=bias), nn.BatchNorm2d(internal_channels), activation)
        else:
            self.ext_conv2 = nn.Sequential(nn.Conv2d(internal_channels, internal_channels, kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation, bias=bias), nn.BatchNorm2d(internal_channels), activation)
        self.ext_conv3 = nn.Sequential(nn.Conv2d(internal_channels, channels, kernel_size=1, stride=1, bias=bias), nn.BatchNorm2d(channels), activation)
        self.ext_regul = nn.Dropout2d(p=dropout_prob)
        self.out_prelu = activation

    def forward(self, x):
        main = x
        ext = self.ext_conv1(x)
        ext = self.ext_conv2(ext)
        ext = self.ext_conv3(ext)
        ext = self.ext_regul(ext)
        out = main + ext
        return self.out_prelu(out)


class DownsamplingBottleneck(nn.Module):
    """Downsampling bottlenecks further downsample the feature map size.

    Main branch:
    1. max pooling with stride 2; indices are saved to be used for
    unpooling later.

    Extension branch:
    1. 2x2 convolution with stride 2 that decreases the number of channels
    by ``internal_ratio``, also called a projection;
    2. regular convolution (by default, 3x3);
    3. 1x1 convolution which increases the number of channels to
    ``out_channels``, also called an expansion;
    4. dropout as a regularizer.

    Keyword arguments:
    - in_channels (int): the number of input channels.
    - out_channels (int): the number of output channels.
    - internal_ratio (int, optional): a scale factor applied to ``channels``
    used to compute the number of channels after the projection. eg. given
    ``channels`` equal to 128 and internal_ratio equal to 2 the number of
    channels after the projection is 64. Default: 4.
    - kernel_size (int, optional): the kernel size of the filters used in
    the convolution layer described above in item 2 of the extension branch.
    Default: 3.
    - padding (int, optional): zero-padding added to both sides of the
    input. Default: 0.
    - dilation (int, optional): spacing between kernel elements for the
    convolution described in item 2 of the extension branch. Default: 1.
    - asymmetric (bool, optional): flags if the convolution described in
    item 2 of the extension branch is asymmetric or not. Default: False.
    - return_indices (bool, optional):  if ``True``, will return the max
    indices along with the outputs. Useful when unpooling later.
    - dropout_prob (float, optional): probability of an element to be
    zeroed. Default: 0 (no dropout).
    - bias (bool, optional): Adds a learnable bias to the output if
    ``True``. Default: False.
    - relu (bool, optional): When ``True`` ReLU is used as the activation
    function; otherwise, PReLU is used. Default: True.

    """

    def __init__(self, in_channels, out_channels, internal_ratio=4, kernel_size=3, padding=0, return_indices=False, dropout_prob=0, bias=False, relu=True):
        super().__init__()
        self.return_indices = return_indices
        if internal_ratio <= 1 or internal_ratio > in_channels:
            raise nncf.ValidationError('Value out of range. Expected value in the interval [1, {0}], got internal_scale={1}. '.format(in_channels, internal_ratio))
        internal_channels = in_channels // internal_ratio
        if relu:
            activation = nn.ReLU()
        else:
            activation = nn.PReLU()
        self.main_max1 = nn.MaxPool2d(kernel_size, stride=2, padding=padding, return_indices=return_indices)
        self.ext_conv1 = nn.Sequential(nn.Conv2d(in_channels, internal_channels, kernel_size=2, stride=2, bias=bias), nn.BatchNorm2d(internal_channels), activation)
        self.ext_conv2 = nn.Sequential(nn.Conv2d(internal_channels, internal_channels, kernel_size=kernel_size, stride=1, padding=padding, bias=bias), nn.BatchNorm2d(internal_channels), activation)
        self.ext_conv3 = nn.Sequential(nn.Conv2d(internal_channels, out_channels, kernel_size=1, stride=1, bias=bias), nn.BatchNorm2d(out_channels), activation)
        self.ext_regul = nn.Dropout2d(p=dropout_prob)
        self.out_prelu = activation

    def forward(self, x):
        if self.return_indices:
            main, max_indices = self.main_max1(x)
        else:
            main = self.main_max1(x)
        ext = self.ext_conv1(x)
        ext = self.ext_conv2(ext)
        ext = self.ext_conv3(ext)
        ext = self.ext_regul(ext)
        n, ch_ext, h, w = ext.size()
        ch_main = main.size()[1]
        padding = torch.zeros(n, ch_ext - ch_main, h, w)
        if main.is_cuda:
            padding = padding
        main = torch.cat((main, padding), 1)
        out = main + ext
        return self.out_prelu(out), max_indices


class UpsamplingBottleneck(nn.Module):
    """The upsampling bottlenecks upsample the feature map resolution using max
    pooling indices stored from the corresponding downsampling bottleneck.

    Main branch:
    1. 1x1 convolution with stride 1 that decreases the number of channels by
    ``internal_ratio``, also called a projection;
    2. max unpool layer using the max pool indices from the corresponding
    downsampling max pool layer.

    Extension branch:
    1. 1x1 convolution with stride 1 that decreases the number of channels by
    ``internal_ratio``, also called a projection;
    2. transposed convolution (by default, 3x3);
    3. 1x1 convolution which increases the number of channels to
    ``out_channels``, also called an expansion;
    4. dropout as a regularizer.

    Keyword arguments:
    - in_channels (int): the number of input channels.
    - out_channels (int): the number of output channels.
    - internal_ratio (int, optional): a scale factor applied to ``in_channels``
     used to compute the number of channels after the projection. eg. given
     ``in_channels`` equal to 128 and ``internal_ratio`` equal to 2 the number
     of channels after the projection is 64. Default: 4.
    - kernel_size (int, optional): the kernel size of the filters used in the
    convolution layer described above in item 2 of the extension branch.
    Default: 3.
    - padding (int, optional): zero-padding added to both sides of the input.
    Default: 0.
    - dropout_prob (float, optional): probability of an element to be zeroed.
    Default: 0 (no dropout).
    - bias (bool, optional): Adds a learnable bias to the output if ``True``.
    Default: False.
    - relu (bool, optional): When ``True`` ReLU is used as the activation
    function; otherwise, PReLU is used. Default: True.

    """

    def __init__(self, in_channels, out_channels, internal_ratio=4, kernel_size=3, padding=0, dropout_prob=0, bias=False, relu=True):
        super().__init__()
        if internal_ratio <= 1 or internal_ratio > in_channels:
            raise nncf.ValidationError('Value out of range. Expected value in the interval [1, {0}], got internal_scale={1}. '.format(in_channels, internal_ratio))
        internal_channels = in_channels // internal_ratio
        if relu:
            activation = nn.ReLU()
        else:
            activation = nn.PReLU()
        self.main_conv1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias), nn.BatchNorm2d(out_channels))
        self.main_unpool1 = nn.MaxUnpool2d(kernel_size=2)
        self.ext_conv1 = nn.Sequential(nn.Conv2d(in_channels, internal_channels, kernel_size=1, bias=bias), nn.BatchNorm2d(internal_channels), activation)
        self.ext_conv2 = nn.Sequential(nn.ConvTranspose2d(internal_channels, internal_channels, kernel_size=kernel_size, stride=2, padding=padding, output_padding=1, bias=bias), nn.BatchNorm2d(internal_channels), activation)
        self.ext_conv3 = nn.Sequential(nn.Conv2d(internal_channels, out_channels, kernel_size=1, bias=bias), nn.BatchNorm2d(out_channels), activation)
        self.ext_regul = nn.Dropout2d(p=dropout_prob)
        self.out_prelu = activation

    def forward(self, x, max_indices):
        main = self.main_conv1(x)
        main = self.main_unpool1(main, max_indices)
        ext = self.ext_conv1(x)
        ext = self.ext_conv2(ext)
        ext = self.ext_conv3(ext)
        ext = self.ext_regul(ext)
        out = main + ext
        return self.out_prelu(out)


class ENet(nn.Module):
    """Generate the ENet model.

    Keyword arguments:
    - num_classes (int): the number of classes to segment.
    - encoder_relu (bool, optional): When ``True`` ReLU is used as the
    activation function in the encoder blocks/layers; otherwise, PReLU
    is used. Default: False.
    - decoder_relu (bool, optional): When ``True`` ReLU is used as the
    activation function in the decoder blocks/layers; otherwise, PReLU
    is used. Default: True.

    """

    def __init__(self, num_classes, encoder_relu=False, decoder_relu=True):
        super().__init__()
        self.initial_block = InitialBlock(3, 16, padding=1, relu=encoder_relu)
        self.downsample1_0 = DownsamplingBottleneck(16, 64, padding=1, return_indices=True, dropout_prob=0.01, relu=encoder_relu)
        self.regular1_1 = RegularBottleneck(64, padding=1, dropout_prob=0.01, relu=encoder_relu)
        self.regular1_2 = RegularBottleneck(64, padding=1, dropout_prob=0.01, relu=encoder_relu)
        self.regular1_3 = RegularBottleneck(64, padding=1, dropout_prob=0.01, relu=encoder_relu)
        self.regular1_4 = RegularBottleneck(64, padding=1, dropout_prob=0.01, relu=encoder_relu)
        self.downsample2_0 = DownsamplingBottleneck(64, 128, padding=1, return_indices=True, dropout_prob=0.1, relu=encoder_relu)
        self.regular2_1 = RegularBottleneck(128, padding=1, dropout_prob=0.1, relu=encoder_relu)
        self.dilated2_2 = RegularBottleneck(128, dilation=2, padding=2, dropout_prob=0.1, relu=encoder_relu)
        self.asymmetric2_3 = RegularBottleneck(128, kernel_size=5, padding=2, asymmetric=True, dropout_prob=0.1, relu=encoder_relu)
        self.dilated2_4 = RegularBottleneck(128, dilation=4, padding=4, dropout_prob=0.1, relu=encoder_relu)
        self.regular2_5 = RegularBottleneck(128, padding=1, dropout_prob=0.1, relu=encoder_relu)
        self.dilated2_6 = RegularBottleneck(128, dilation=8, padding=8, dropout_prob=0.1, relu=encoder_relu)
        self.asymmetric2_7 = RegularBottleneck(128, kernel_size=5, asymmetric=True, padding=2, dropout_prob=0.1, relu=encoder_relu)
        self.dilated2_8 = RegularBottleneck(128, dilation=16, padding=16, dropout_prob=0.1, relu=encoder_relu)
        self.regular3_0 = RegularBottleneck(128, padding=1, dropout_prob=0.1, relu=encoder_relu)
        self.dilated3_1 = RegularBottleneck(128, dilation=2, padding=2, dropout_prob=0.1, relu=encoder_relu)
        self.asymmetric3_2 = RegularBottleneck(128, kernel_size=5, padding=2, asymmetric=True, dropout_prob=0.1, relu=encoder_relu)
        self.dilated3_3 = RegularBottleneck(128, dilation=4, padding=4, dropout_prob=0.1, relu=encoder_relu)
        self.regular3_4 = RegularBottleneck(128, padding=1, dropout_prob=0.1, relu=encoder_relu)
        self.dilated3_5 = RegularBottleneck(128, dilation=8, padding=8, dropout_prob=0.1, relu=encoder_relu)
        self.asymmetric3_6 = RegularBottleneck(128, kernel_size=5, asymmetric=True, padding=2, dropout_prob=0.1, relu=encoder_relu)
        self.dilated3_7 = RegularBottleneck(128, dilation=16, padding=16, dropout_prob=0.1, relu=encoder_relu)
        self.upsample4_0 = UpsamplingBottleneck(128, 64, padding=1, dropout_prob=0.1, relu=decoder_relu)
        self.regular4_1 = RegularBottleneck(64, padding=1, dropout_prob=0.1, relu=decoder_relu)
        self.regular4_2 = RegularBottleneck(64, padding=1, dropout_prob=0.1, relu=decoder_relu)
        self.upsample5_0 = UpsamplingBottleneck(64, 16, padding=1, dropout_prob=0.1, relu=decoder_relu)
        self.regular5_1 = RegularBottleneck(16, padding=1, dropout_prob=0.1, relu=decoder_relu)
        self.transposed_conv = nn.ConvTranspose2d(16, num_classes, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)

    def forward(self, x):
        x = self.initial_block(x)
        x, max_indices1_0 = self.downsample1_0(x)
        x = self.regular1_1(x)
        x = self.regular1_2(x)
        x = self.regular1_3(x)
        x = self.regular1_4(x)
        x, max_indices2_0 = self.downsample2_0(x)
        x = self.regular2_1(x)
        x = self.dilated2_2(x)
        x = self.asymmetric2_3(x)
        x = self.dilated2_4(x)
        x = self.regular2_5(x)
        x = self.dilated2_6(x)
        x = self.asymmetric2_7(x)
        x = self.dilated2_8(x)
        x = self.regular3_0(x)
        x = self.dilated3_1(x)
        x = self.asymmetric3_2(x)
        x = self.dilated3_3(x)
        x = self.regular3_4(x)
        x = self.dilated3_5(x)
        x = self.asymmetric3_6(x)
        x = self.dilated3_7(x)
        x = self.upsample4_0(x, max_indices2_0)
        x = self.regular4_1(x)
        x = self.regular4_2(x)
        x = self.upsample5_0(x, max_indices1_0)
        x = self.regular5_1(x)
        x = self.transposed_conv(x)
        return x


class ResNetBlock(nn.Module):

    def __init__(self, in_channels, reduce_channels, increase_channels, dilation=1, stride=1):
        super().__init__()
        nonshrinking_padding = dilation
        self.conv_1x1_reduce_bnrelu = ConvBNReLU(in_channels, out_channels=reduce_channels, kernel_size=1, stride=stride, padding=0, dilation=1, bias=False)
        self.conv_3x3_bnrelu = ConvBNReLU(in_channels=reduce_channels, out_channels=reduce_channels, kernel_size=3, stride=1, padding=nonshrinking_padding, dilation=dilation, bias=False)
        self.conv_1x1_increase_bn = ConvBN(in_channels=reduce_channels, out_channels=increase_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False)
        self.need_proj = in_channels != increase_channels
        if self.need_proj:
            self.conv_1x1_proj_bn = ConvBN(in_channels, out_channels=increase_channels, kernel_size=1, stride=stride, padding=0, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs):
        fx = self.conv_1x1_reduce_bnrelu(inputs)
        fx = self.conv_3x3_bnrelu(fx)
        fx = self.conv_1x1_increase_bn(fx)
        x = inputs
        if self.need_proj:
            x = self.conv_1x1_proj_bn(x)
        out = fx + x
        out = self.relu(out)
        return out


class ICNetBackbone(nn.Module):

    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Sequential(OrderedDict([('conv1_1_3x3_s2', ConvBNReLU(in_channels, out_channels=32, kernel_size=3, stride=2, padding=1, dilation=1, bias=False)), ('conv1_2_3x3', ConvBNReLU(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, dilation=1, bias=False)), ('conv1_3_3x3', ConvBNReLU(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1, dilation=1, bias=False))]))
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Sequential(OrderedDict([('conv2_1', ResNetBlock(64, 32, 128)), ('conv2_2', ResNetBlock(128, 32, 128)), ('conv2_3', ResNetBlock(128, 32, 128))]))
        self.conv3_1 = ResNetBlock(128, 64, 256, stride=2)
        self.conv3_rest = nn.Sequential(OrderedDict([('conv3_2', ResNetBlock(256, 64, 256)), ('conv3_3', ResNetBlock(256, 64, 256)), ('conv3_4', ResNetBlock(256, 64, 256))]))
        self.conv4 = nn.Sequential(OrderedDict([('conv4_1', ResNetBlock(256, 128, 512, dilation=2)), ('conv4_2', ResNetBlock(512, 128, 512, dilation=2)), ('conv4_3', ResNetBlock(512, 128, 512, dilation=2)), ('conv4_4', ResNetBlock(512, 128, 512, dilation=2)), ('conv4_5', ResNetBlock(512, 128, 512, dilation=2)), ('conv4_6', ResNetBlock(512, 128, 512, dilation=2))]))
        self.conv4 = nn.Sequential(OrderedDict([('conv4_1', ResNetBlock(256, 128, 512, dilation=2)), ('conv4_2', ResNetBlock(512, 128, 512, dilation=2)), ('conv4_3', ResNetBlock(512, 128, 512, dilation=2)), ('conv4_4', ResNetBlock(512, 128, 512, dilation=2)), ('conv4_5', ResNetBlock(512, 128, 512, dilation=2)), ('conv4_6', ResNetBlock(512, 128, 512, dilation=2))]))
        self.conv5 = nn.Sequential(OrderedDict([('conv5_1', ResNetBlock(512, 256, 1024, dilation=4)), ('conv5_2', ResNetBlock(1024, 256, 1024, dilation=4)), ('conv5_3', ResNetBlock(1024, 256, 1024, dilation=4))]))

    def forward(self):
        pass


class PyramidPooling(nn.Module):

    def __init__(self, input_size_hw, bin_dimensions=None, mode='sum'):
        super().__init__()
        if mode not in ['sum', 'cat']:
            raise NotImplementedError
        self.mode = mode
        self.input_size_hw = input_size_hw
        self.sampling_params = {'mode': 'nearest'}
        if bin_dimensions is None:
            self.bin_dimensions = [1, 2, 3, 6]
        else:
            self.bin_dimensions = bin_dimensions
        self.paddings = {}
        for dim in self.bin_dimensions:
            pad_h = (dim - input_size_hw[0] % dim) % dim
            pad_w = (dim - input_size_hw[1] % dim) % dim
            self.paddings[dim] = 0, pad_w, 0, pad_h

    def forward(self, inputs):
        x = inputs.clone()
        for dim in self.bin_dimensions:
            pooled_feature = F.adaptive_avg_pool2d(inputs, dim)
            pooled_feature = F.interpolate(pooled_feature, self.input_size_hw, **self.sampling_params)
            if self.mode == 'sum':
                x += pooled_feature
            elif self.mode == 'cat':
                x = torch.cat(pooled_feature)
            else:
                raise NotImplementedError
        return x


class CascadeFeatureFusion(nn.Module):

    def __init__(self, in_channels_lowres, in_channels_highres, highres_size_hw, num_classes):
        super().__init__()
        self.sampling_params = {'mode': 'nearest'}
        self.conv = ConvBN(in_channels_lowres, out_channels=128, kernel_size=3, padding=2, dilation=2, bias=False)
        self.conv_proj = ConvBN(in_channels_highres, out_channels=128, kernel_size=1, padding=0, dilation=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.classifier = nn.Conv2d(in_channels_lowres, out_channels=num_classes, kernel_size=1, padding=0, dilation=1, bias=True)
        self.highres_size_hw = highres_size_hw

    def forward(self, lowres_input, highres_input):
        upsampled = F.interpolate(lowres_input, self.highres_size_hw, **self.sampling_params)
        lr = self.conv(upsampled)
        hr = self.conv_proj(highres_input)
        x = lr + hr
        x = self.relu(x)
        if self.training:
            aux_labels = self.classifier(upsampled)
            return x, aux_labels
        return x


def get_backbone(backbone, in_channels):
    if backbone == 'icnet':
        return ICNetBackbone(in_channels)
    raise NotImplementedError


def is_tracing_state():
    return torch._C._get_tracing_state() is not None


class ICNet(nn.Module):

    def __init__(self, input_size_hw, in_channels=3, n_classes=20, backbone='icnet'):
        super().__init__()
        self._input_size_hw = input_size_hw
        self._input_size_hw_ds2 = self._input_size_hw[0] // 2, self._input_size_hw[1] // 2
        self._input_size_hw_ds4 = self._input_size_hw[0] // 4, self._input_size_hw[1] // 4
        self._input_size_hw_ds8 = self._input_size_hw[0] // 8, self._input_size_hw[1] // 8
        self._input_size_hw_ds16 = self._input_size_hw[0] // 16, self._input_size_hw[1] // 16
        self._input_size_hw_ds32 = self._input_size_hw[0] // 32, self._input_size_hw[1] // 32
        self.sampling_params = {'mode': 'nearest'}
        self.backbone = get_backbone(backbone, in_channels)
        self.highres_conv = nn.Sequential(OrderedDict([('conv1_sub1', ConvBNReLU(in_channels, out_channels=32, kernel_size=3, stride=2, padding=1, bias=False)), ('conv2_sub1', ConvBNReLU(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1, bias=False)), ('conv3_sub1', ConvBNReLU(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1, bias=False))]))
        self.conv5_4_k1 = ConvBNReLU(in_channels=1024, out_channels=256, kernel_size=1, stride=1, padding=0, dilation=1, bias=False)
        self.ppm = PyramidPooling(self._input_size_hw_ds32)
        self.cff42 = CascadeFeatureFusion(in_channels_lowres=256, in_channels_highres=256, highres_size_hw=self._input_size_hw_ds16, num_classes=n_classes)
        self.cff421 = CascadeFeatureFusion(in_channels_lowres=128, in_channels_highres=32, highres_size_hw=self._input_size_hw_ds8, num_classes=n_classes)
        self.conv6_cls = nn.Conv2d(128, out_channels=n_classes, kernel_size=1, padding=0, dilation=1, bias=True)
        required_alignment = 32
        for bin_dim in self.ppm.bin_dimensions:
            required_alignment = lcm(required_alignment, bin_dim)
        if input_size_hw[0] % required_alignment or input_size_hw[1] % required_alignment:
            raise ValueError('ICNet may only operate on {}-aligned input resolutions'.format(required_alignment))

    def highres_branch(self, inputs):
        x = self.highres_conv(inputs)
        return x

    def mediumres_branch(self, inputs):
        x = self.backbone.conv1(inputs)
        x = self.backbone.maxpool(x)
        x = self.backbone.conv2(x)
        x = self.backbone.conv3_1(x)
        return x

    def lowres_branch(self, inputs):
        x = self.backbone.conv3_rest(inputs)
        x = self.backbone.conv4(x)
        x = self.backbone.conv5(x)
        x = self.ppm(x)
        x = self.conv5_4_k1(x)
        return x

    def forward(self, inputs):
        data_sub1 = inputs
        features_sub1 = self.highres_branch(data_sub1)
        data_sub2 = F.interpolate(data_sub1, self._input_size_hw_ds2, **self.sampling_params)
        features_sub2 = self.mediumres_branch(data_sub2)
        data_sub4 = F.interpolate(features_sub2, self._input_size_hw_ds32, **self.sampling_params)
        features_sub4 = self.lowres_branch(data_sub4)
        if self.training:
            fused_features_sub42, label_scores_ds16 = self.cff42(features_sub4, features_sub2)
            fused_features_sub421, label_scores_ds8 = self.cff421(fused_features_sub42, features_sub1)
            fused_features_ds4 = F.interpolate(fused_features_sub421, self._input_size_hw_ds4, **self.sampling_params)
            label_scores_ds4 = self.conv6_cls(fused_features_ds4)
            return OrderedDict([('ds4', label_scores_ds4), ('ds8', label_scores_ds8), ('ds16', label_scores_ds16)])
        fused_features_sub42 = self.cff42(features_sub4, features_sub2)
        fused_features_sub421 = self.cff421(fused_features_sub42, features_sub1)
        fused_features_ds4 = F.interpolate(fused_features_sub421, self._input_size_hw_ds4, **self.sampling_params)
        label_scores_ds4 = self.conv6_cls(fused_features_ds4)
        label_scores = F.interpolate(label_scores_ds4, self._input_size_hw, **self.sampling_params)
        if is_tracing_state() and version.parse(torch.__version__) >= version.parse('1.1.0'):
            softmaxed = F.softmax(label_scores, dim=1)
            return softmaxed
        return label_scores


class UNetConvBlock(nn.Module):

    def __init__(self, in_size, out_size, padding, batch_norm):
        super().__init__()
        block = []
        block.append(nn.Conv2d(in_size, out_size, kernel_size=3, padding=int(padding)))
        if batch_norm:
            block.append(nn.BatchNorm2d(out_size))
        block.append(nn.ReLU())
        block.append(nn.Conv2d(out_size, out_size, kernel_size=3, padding=int(padding)))
        if batch_norm:
            block.append(nn.BatchNorm2d(out_size))
        block.append(nn.ReLU())
        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        return out


def center_crop(layer, target_size):
    if layer.dim() == 4:
        _, _, layer_height, layer_width = layer.size()
        diff_y = (layer_height - target_size[0]) // 2
        diff_x = (layer_width - target_size[1]) // 2
        return layer[:, :, diff_y:diff_y + target_size[0], diff_x:diff_x + target_size[1]]
    assert layer.dim() == 3
    _, layer_height, layer_width = layer.size()
    diff_y = (layer_height - target_size[0]) // 2
    diff_x = (layer_width - target_size[1]) // 2
    return layer[:, diff_y:diff_y + target_size[0], diff_x:diff_x + target_size[1]]


class UNetUpBlock(nn.Module):

    def __init__(self, in_size, out_size, up_mode, padding, batch_norm):
        super().__init__()
        if up_mode == 'upconv':
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2)
        elif up_mode == 'upsample':
            self.up = nn.Sequential(nn.Upsample(mode='bilinear', scale_factor=2), nn.Conv2d(in_size, out_size, kernel_size=1))
        self.conv_block = UNetConvBlock(in_size, out_size, padding, batch_norm)

    def forward(self, x, bridge):
        up = self.up(x)
        crop1 = center_crop(bridge, up.shape[2:])
        out = torch.cat([up, crop1], 1)
        out = self.conv_block(out)
        return out


class UNet(nn.Module):

    def __init__(self, in_channels=3, n_classes=12, depth=5, wf=6, padding=False, batch_norm=True, up_mode='upconv'):
        """
        Implementation of
        U-Net: Convolutional Networks for Biomedical Image Segmentation
        (Ronneberger et al., 2015)
        https://arxiv.org/abs/1505.04597
        Using the default arguments will yield the exact version used
        in the original paper
        Args:
            in_channels (int): number of input channels
            n_classes (int): number of output channels
            depth (int): depth of the network
            wf (int): number of filters in the first layer is 2**wf
            padding (bool): if True, apply padding such that the input shape
                            is the same as the output.
                            This may introduce artifacts
            batch_norm (bool): Use BatchNorm prior to layers with an
                               activation function
            up_mode (str): one of 'upconv' or 'upsample'.
                           'upconv' will use transposed convolutions for
                           learned upsampling.
                           'upsample' will use bilinear upsampling.
        """
        super().__init__()
        assert up_mode in ('upconv', 'upsample')
        self.padding = padding
        self.depth = depth
        prev_channels = in_channels
        self.down_path = nn.ModuleList()
        for i in range(depth):
            self.down_path.append(UNetConvBlock(prev_channels, 2 ** (wf + i), padding, batch_norm))
            prev_channels = 2 ** (wf + i)
        self.up_path = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            self.up_path.append(UNetUpBlock(prev_channels, 2 ** (wf + i), up_mode, padding, batch_norm))
            prev_channels = 2 ** (wf + i)
        self.last = nn.Conv2d(prev_channels, n_classes, kernel_size=1)

    def forward(self, x):
        blocks = []
        for i, down in enumerate(self.down_path):
            x = down(x)
            if i != len(self.down_path) - 1:
                blocks.append(x)
                x = F.max_pool2d(x, 2)
        for i, up in enumerate(self.up_path):
            x = up(x, blocks[-i - 1])
        x = self.last(x)
        return x


DOMAIN_CUSTOM_OPS_NAME = 'org.openvinotoolkit'


def add_domain(name_operator: 'str') ->str:
    return DOMAIN_CUSTOM_OPS_NAME + '::' + name_operator


def decode(loc, priors):
    """Decode locations from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        loc (tensor): location predictions for loc layers,
            Shape: [num_priors,4]
        priors (tensor): Prior boxes in point-form with variances.
            Shape: [2, num_priors,4].
    Return:
        decoded bounding box predictions
    """
    variances = priors[1].squeeze(0)
    priors = priors[0].squeeze(0)
    decoded_boxes_cx_cy = variances[:, :2] * loc[:, :2] * (priors[:, 2:] - priors[:, :2]) + (priors[:, :2] + priors[:, 2:]) / 2
    decoded_boxes_w_h = torch.exp(variances[:, 2:] * loc[:, 2:]) * (priors[:, 2:] - priors[:, :2])
    decoded_boxes_xmin_ymin = decoded_boxes_cx_cy - decoded_boxes_w_h / 2
    decoded_boxes_xmax_ymax = decoded_boxes_cx_cy + decoded_boxes_w_h / 2
    encoded_boxes = torch.cat((decoded_boxes_xmin_ymin, decoded_boxes_xmax_ymax), 1)
    return encoded_boxes


class Registry:
    REGISTERED_NAME_ATTR = '_registered_name'

    def __init__(self, name: 'str', add_name_as_attr: 'bool'=False):
        self._name = name
        self._registry_dict: 'Dict[str, Any]' = {}
        self._add_name_as_attr = add_name_as_attr

    @property
    def registry_dict(self) ->Dict[str, Any]:
        return self._registry_dict

    def values(self) ->Any:
        return self._registry_dict.values()

    def _register(self, obj: 'Any', name: 'str') ->None:
        if name in self._registry_dict:
            raise KeyError('{} is already registered in {}'.format(name, self._name))
        self._registry_dict[name] = obj

    def register(self, name: 'str'=None) ->Callable[[Any], Any]:

        def wrap(obj: 'Any') ->Any:
            cls_name = name
            if cls_name is None:
                cls_name = obj.__name__
            if self._add_name_as_attr:
                setattr(obj, self.REGISTERED_NAME_ATTR, name)
            self._register(obj, cls_name)
            return obj
        return wrap

    def get(self, name: 'str') ->Any:
        if name not in self._registry_dict:
            self._key_not_found(name)
        return self._registry_dict[name]

    def _key_not_found(self, name: 'str') ->None:
        raise KeyError('{} is unknown type of {} '.format(name, self._name))

    def __contains__(self, item: 'Any') ->bool:
        return item in self._registry_dict.values()


EXTENSIONS = Registry('extensions')


class NMSFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, boxes, scores, threshold, top_k=200):
        if scores.size(0) == 0:
            return torch.tensor([], dtype=torch.int), torch.tensor(0)
        if scores.dim() == 1:
            scores = scores.unsqueeze(1)
        if not boxes.is_cuda:
            keep = torch_nms(boxes, scores.flatten(), threshold)
        else:
            keep = EXTENSIONS.nms(torch.cat((boxes, scores), dim=1), threshold, top_k)
        return keep, torch.tensor(keep.size(0))

    @staticmethod
    def backward(ctx: 'Any', *grad_outputs: Any) ->Any:
        raise NotImplementedError


nms = NMSFunction.apply


class no_jit_trace:

    def __enter__(self):
        self.state = torch._C._get_tracing_state()
        torch._C._set_tracing_state(None)

    def __exit__(self, *args):
        torch._C._set_tracing_state(self.state)
        self.state = None


class api:
    API_MARKER_ATTR = '_nncf_api_marker'
    CANONICAL_ALIAS_ATTR = '_nncf_canonical_alias'

    def __init__(self, canonical_alias: 'str'=None):
        self._canonical_alias = canonical_alias

    def __call__(self, obj: 'Any') ->Any:
        setattr(obj, api.API_MARKER_ATTR, obj.__name__)
        if self._canonical_alias is not None:
            setattr(obj, api.CANONICAL_ALIAS_ATTR, self._canonical_alias)
        return obj


class CopySafeThreadingVars:
    """A class holding variables that are related to threading and
    thus impossible to deepcopy. The deepcopy will simply return a
    new object without copying, but won't fail."""

    def __init__(self):
        self.thread_local = TracingThreadLocals()
        self.cond = threading.Condition()

    def __deepcopy__(self, memo):
        return CopySafeThreadingVars()

