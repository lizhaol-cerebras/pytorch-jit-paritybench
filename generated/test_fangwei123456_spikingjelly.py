
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


from matplotlib import pyplot as plt


import torch


import numpy as np


import matplotlib


import torch.nn as nn


import torchvision


import torch.nn.functional as F


import torchvision.transforms


from torchvision.datasets import FashionMNIST


from typing import Type


from typing import Dict


from typing import Any


from typing import Tuple


from typing import Iterable


from torch import fx


from torch.nn.utils.fusion import fuse_conv_bn_eval


import matplotlib.pyplot as plt


import logging


import re


import copy


from typing import Callable


from typing import Optional


import math


from abc import abstractmethod


import time


from typing import Union


import random


import torch.optim as optim


from torch.distributions import Categorical


from torch.utils.tensorboard import SummaryWriter


from collections import namedtuple


from itertools import count


import torchvision.transforms as T


import collections


from torch.autograd import Variable


from collections import deque


from copy import deepcopy


import itertools


from torch.optim import Adam


from collections import OrderedDict


from torch.distributions import Normal


import torch.backends.cudnn as cudnn


import torchvision.transforms as transforms


from torch.cuda import amp


from torch.utils.data import DataLoader


from torchvision import transforms


import torch.utils.data as data


import torchvision.datasets


from torch import Tensor


from torch import nn


from scipy.signal import savgol_filter


from sklearn.metrics import confusion_matrix


import matplotlib.ticker as ticker


import string


from torch.nn.common_types import _size_any_t


from torch.nn.common_types import _size_1_t


from torch.nn.common_types import _size_2_t


from torch.nn.common_types import _size_3_t


from torch.nn.common_types import _ratio_any_t


from typing import List


from torch.nn.modules.batchnorm import _BatchNorm


from torch.nn.utils.fusion import *


from torch.autograd import Function


from torch.nn.functional import interpolate


from math import tanh


from torch.jit import script


import warnings


import torch.utils.data


from torch.utils.data.dataloader import default_collate


from torchvision.transforms.functional import InterpolationMode


from torchvision.transforms import autoaugment


from torchvision.transforms import transforms


import torch.distributed as dist


from torchvision.transforms import functional as F


from collections import defaultdict


from numpy import sqrt


from numpy import newaxis


from numpy import integer


from numpy.fft import irfft


from numpy.fft import rfftfreq


from numpy.random import default_rng


from numpy.random import Generator


from numpy.random import RandomState


from numpy import sum as npsum


from torch.utils.cpp_extension import load_inline


from torch.cuda.amp import custom_fwd


from torch.cuda.amp import custom_bwd


from torch.types import _int


from torch.types import _size


from torch.nn.modules.utils import _single


from torch.nn.modules.utils import _pair


from torch.nn.modules.utils import _triple


from torchvision.datasets import DatasetFolder


import scipy.io


from torchvision.datasets import utils


from torch.utils.data import Dataset


from torchvision.datasets.utils import extract_archive


from torchvision.datasets.utils import download_url


from torchvision.datasets.utils import verify_str_arg


from random import choice


class VoltageHook(nn.Module):

    def __init__(self, scale=1.0, momentum=0.1, mode='Max'):
        """
        * :ref:`API in English <VoltageHook.__init__-en>`

        .. _voltageHook.__init__-cn:

        :param scale: 缩放初始值
        :type scale: float
        :param momentum: 动量值
        :type momentum: float
        :param mode: 模式。输入“Max”表示记录ANN激活最大值，“99.9%”表示记录ANN激活的99.9%分位点，输入0-1的float型浮点数表示记录激活最大值的对应倍数。
        :type mode: str, float

        ``VoltageHook`` 被置于ReLU后，用于在ANN推理中确定激活的范围。

        * :ref:`中文API <VoltageHook.__init__-cn>`

        .. _voltageHook.__init__-en:

        :param scale: initial scaling value
        :type scale: float
        :param momentum: momentum value
        :type momentum: float
        :param mode: The mode. Value "Max" means recording the maximum value of ANN activation, "99.9%" means recording the 99.9% precentile of ANN activation, and a float of 0-1 means recording the corresponding multiple of the maximum activation value.
        :type mode: str, float

        ``VoltageHook`` is placed behind ReLU and used to determine the range of activations in ANN inference.

        """
        super().__init__()
        self.register_buffer('scale', torch.tensor(scale))
        self.mode = mode
        self.num_batches_tracked = 0
        self.momentum = momentum

    def forward(self, x):
        """
        * :ref:`API in English <VoltageHook.forward-en>`

        .. _VoltageHook.forward-cn:

        :param x: 输入张量
        :type x: torch.Tensor
        :return: 原输入张量
        :rtype: torch.Tensor

        不对输入张量做任何处理，只是抓取ReLU的激活值

        * :ref:`中文API <VoltageHook.forward-cn>`

        .. _VoltageHook.forward-en:

        :param x: input tensor
        :type x: torch.Tensor
        :return: original input tensor
        :rtype: torch.Tensor

        It doesn't process input tensors, but hooks the activation values of ReLU.

        """
        err_msg = 'You have used a non-defined VoltageScale Method.'
        if isinstance(self.mode, str):
            if self.mode[-1] == '%':
                try:
                    s_t = torch.tensor(np.percentile(x.detach().cpu(), float(self.mode[:-1])))
                except ValueError:
                    raise NotImplementedError(err_msg)
            elif self.mode.lower() in ['max']:
                s_t = x.max().detach()
            else:
                raise NotImplementedError(err_msg)
        elif isinstance(self.mode, float) and self.mode <= 1 and self.mode > 0:
            s_t = x.max().detach() * self.mode
        else:
            raise NotImplementedError(err_msg)
        if self.num_batches_tracked == 0:
            self.scale = s_t
        else:
            self.scale = (1 - self.momentum) * self.scale + self.momentum * s_t
        self.num_batches_tracked += x.shape[0]
        return x


class VoltageScaler(nn.Module):

    def __init__(self, scale=1.0):
        """
        * :ref:`API in English <VoltageScaler.__init__-en>`

        .. _VoltageScaler.__init__-cn:

        :param scale: 缩放值
        :type scale: float

        ``VoltageScaler`` 用于SNN推理中缩放电流。

        * :ref:`中文API <VoltageScaler.__init__-cn>`

        .. _VoltageScaler.__init__-en:

        :param scale: scaling value
        :type scale: float

        ``VoltageScaler`` is used for scaling current in SNN inference.

        """
        super().__init__()
        self.register_buffer('scale', torch.tensor(scale))

    def forward(self, x):
        """
        * :ref:`API in English <VoltageScaler.forward-en>`

        .. _VoltageScaler.forward-cn:

        :param x: 输入张量，亦即输入电流
        :type x: torch.Tensor
        :return: 缩放后的电流
        :rtype: torch.Tensor

        * :ref:`中文API <VoltageScaler.forward-cn>`

        .. _VoltageScaler.forward-en:

        :param x: input tensor, or input current
        :type x: torch.Tensor
        :return: current after scaling
        :rtype: torch.Tensor

        """
        return x * self.scale

    def extra_repr(self):
        return '%f' % self.scale.item()


class Converter(nn.Module):

    def __init__(self, dataloader, device=None, mode='Max', momentum=0.1, fuse_flag=True):
        """
        * :ref:`API in English <Converter.__init__-en>`

        .. _Converter.__init__-cn:

        :param dataloader: 数据加载器
        :type dataloader: Dataloader
        :param device: Device
        :type device: str
        :param mode: 转换模式。目前支持三种模式: 最大电流转换模式mode='max'，99.9%电流转换模式mode='99.9%'，以及缩放转换模式mode=x（0<x<=1）
        :type mode: str, float
        :param momentum: 动量值，用于modules.VoltageHook
        :type momentum: float
        :param fuse_flag: 标志位，设置为True，则进行conv与bn的融合，反之不进行。
        :type fuse_flag: bool

        ``Converter`` 用于将带有ReLU的ANN转换为SNN。

        ANN2SNN教程见此处 `ANN转换SNN <https://spikingjelly.readthedocs.io/zh_CN/latest/activation_based/ann2snn.html>`_ 。

        目前支持三种转换模式，由参数mode进行设置。

        转换后ReLU模块被删除，SNN需要的新模块（包括VoltageScaler、IFNode等)被创建并存放在snn tailor父模块中。

        由于返回值的类型为fx.GraphModule，建议使用print(fx.GraphModule.graph)查看计算图及前向传播关系。更多API参见 `GraphModule <https://pytorch.org/docs/stable/fx.html?highlight=graphmodule#torch.fx.GraphModule>`_ 。

        .. warning::

            必须确保ANN中的 ``ReLU`` 为module而非function。

            您最好在ANN模型中使用平均池化而不是最大池化。否则，可能会损害转换后的SNN模型的性能。

        * :ref:`中文API <Converter.__init__-cn>`

        .. _Converter.__init__-en:

        :param dataloader: Dataloader for converting
        :type dataloader: Dataloader
        :param device: Device
        :type device: str
        :param mode: Conversion mode. Now support three mode, MaxNorm(mode='max'), RobustNorm(mode='99.9%'), and scaling mode(mode=x, where 0<x<=1)
        :type mode: str, float
        :param momentum: Momentum value used by modules.VoltageHook
        :type momentum: float
        :param fuse_flag: Bool specifying if fusion of the conv and the bn happens, by default it happens.
        :type fuse_flag: bool

        ``Converter`` is used to convert ANN with to SNN.

        ANN2SNN tutorial is here `ANN2SNN <https://spikingjelly.readthedocs.io/zh_CN/latest/activation_based_en/ann2snn.html>`_ .

        Three common methods are implemented here, which can be selected by the value of parameter mode.

        After converting, ReLU modules will be removed. And new modules needed by SNN, such as VoltageScaler and IFNode, will be created and stored in the parent module 'snn tailor'.

        Due to the type of the return model is fx.GraphModule, you can use 'print(fx.GraphModule.graph)' to view how modules links and the how the forward method works. More APIs are here `GraphModule <https://pytorch.org/docs/stable/fx.html?highlight=graphmodule#torch.fx.GraphModule>`_ .

        .. warning::

            Make sure that ``ReLU`` is module rather than function.

            You'd better use ``avgpool`` rather than ``maxpool`` in your ann model. If not, the performance of the converted snn model may be ruined.
        """
        super().__init__()
        self.mode = mode
        self.fuse_flag = fuse_flag
        self.dataloader = dataloader
        self._check_mode()
        self.device = device
        self.momentum = momentum

    def forward(self, ann: 'nn.Module'):
        """
        * :ref:`API in English <Converter.forward-en>`

        .. _Converter.forward-cn:
        :param ann: 待转换的ann
        :type ann: torch.nn.Module
        :return: 转换得到的snn
        :rtype: torch.fx.GraphModule

        * :ref:`API in Chinese <Converter.forward-cn>`

        .. _Converter.forward-en:
        :param ann: ann to be converted
        :type ann: torch.nn.Module
        :return: snn
        :rtype: torch.fx.GraphModule

        """
        if self.device is None:
            self.device = next(ann.parameters()).device
        ann = fx.symbolic_trace(ann)
        ann.eval()
        ann_fused = self.fuse(ann, fuse_flag=self.fuse_flag)
        ann_with_hook = self.set_voltagehook(ann_fused, momentum=self.momentum, mode=self.mode)
        for _, (imgs, _) in enumerate(tqdm(self.dataloader)):
            ann_with_hook(imgs)
        snn = self.replace_by_ifnode(ann_with_hook)
        return snn

    def _check_mode(self):
        err_msg = 'You have used a non-defined VoltageScale Method.'
        if isinstance(self.mode, str):
            if self.mode[-1] == '%':
                try:
                    float(self.mode[:-1])
                except ValueError:
                    raise NotImplementedError(err_msg)
            elif self.mode.lower() in ['max']:
                pass
            else:
                raise NotImplementedError(err_msg)
        elif isinstance(self.mode, float):
            try:
                assert self.mode <= 1 and self.mode > 0
            except AssertionError:
                raise NotImplementedError(err_msg)
        else:
            raise NotImplementedError(err_msg)

    @staticmethod
    def fuse(fx_model: 'torch.fx.GraphModule', fuse_flag: 'bool'=True) ->torch.fx.GraphModule:
        """
        * :ref:`API in English <Converter.fuse-en>`

        .. _Converter.fuse-cn:

        :param fx_model: 原模型
        :type fx_model: torch.fx.GraphModule
        :param fuse_flag: 标志位，设置为True，则进行conv与bn的融合，反之不进行。
        :type fuse_flag: bool
        :return: conv层和bn层融合后的模型.
        :rtype: torch.fx.GraphModule

        ``fuse`` 用于conv与bn的融合。

        * :ref:`中文API <Converter.fuse-cn>`

        .. _Converter.fuse-en:

        :param fx_model: Original fx_model
        :type fx_model: torch.fx.GraphModule
        :param fuse_flag: Bool specifying if fusion of the conv and the bn happens, by default it happens.
        :type fuse_flag: bool
        :return: fx_model whose conv layer and bn layer have been fused.
        :rtype: torch.fx.GraphModule

        ``fuse`` is used to fuse conv layer and bn layer.

        """

        def matches_module_pattern(pattern: 'Iterable[Type]', node: 'fx.Node', modules: 'Dict[str, Any]') ->bool:
            if len(node.args) == 0:
                return False
            nodes: 'Tuple[Any, fx.Node]' = (node.args[0], node)
            for expected_type, current_node in zip(pattern, nodes):
                if not isinstance(current_node, fx.Node):
                    return False
                if current_node.op != 'call_module':
                    return False
                if not isinstance(current_node.target, str):
                    return False
                if current_node.target not in modules:
                    return False
                if type(modules[current_node.target]) is not expected_type:
                    return False
            return True

        def replace_node_module(node: 'fx.Node', modules: 'Dict[str, Any]', new_module: 'torch.nn.Module'):

            def parent_name(target: 'str') ->Tuple[str, str]:
                """
                Splits a qualname into parent path and last atom.
                For example, `foo.bar.baz` -> (`foo.bar`, `baz`)
                """
                *parent, name = target.rsplit('.', 1)
                return parent[0] if parent else '', name
            assert isinstance(node.target, str)
            parent_name, name = parent_name(node.target)
            modules[node.target] = new_module
            setattr(modules[parent_name], name, new_module)
        if not fuse_flag:
            return fx_model
        patterns = [(nn.Conv1d, nn.BatchNorm1d), (nn.Conv2d, nn.BatchNorm2d), (nn.Conv3d, nn.BatchNorm3d)]
        modules = dict(fx_model.named_modules())
        for pattern in patterns:
            for node in fx_model.graph.nodes:
                if matches_module_pattern(pattern, node, modules):
                    if len(node.args[0].users) > 1:
                        continue
                    conv = modules[node.args[0].target]
                    bn = modules[node.target]
                    fused_conv = fuse_conv_bn_eval(conv, bn)
                    replace_node_module(node.args[0], modules, fused_conv)
                    node.replace_all_uses_with(node.args[0])
                    fx_model.graph.erase_node(node)
        fx_model.graph.lint()
        fx_model.delete_all_unused_submodules()
        fx_model.recompile()
        return fx_model

    @staticmethod
    def set_voltagehook(fx_model: 'torch.fx.GraphModule', mode='Max', momentum=0.1) ->torch.fx.GraphModule:
        """
        * :ref:`API in English <Converter.set_voltagehook-en>`

        .. _Converter.set_voltagehook-cn:

        :param fx_model: 原模型
        :type fx_model: torch.fx.GraphModule
        :param mode: 转换模式。目前支持三种模式，最大电流转换模式，99.9%电流转换模式，以及缩放转换模式
        :type mode: str, float
        :param momentum: 动量值，用于VoltageHook
        :type momentum: float
        :return: 带有VoltageHook的模型.
        :rtype: torch.fx.GraphModule

        ``set_voltagehook`` 用于给模型添加VoltageHook模块。这里实现了常见的三种模式，同上。

        * :ref:`中文API <Converter.set_voltagehook-cn>`

        .. _Converter.set_voltagehook-en:

        :param fx_model: Original fx_model
        :type fx_model: torch.fx.GraphModule
        :param mode: Conversion mode. Now support three mode, MaxNorm, RobustNorm(99.9%), and scaling mode
        :type mode: str, float
        :param momentum: momentum value used by VoltageHook
        :type momentum: float
        :return: fx_model with VoltageHook.
        :rtype: torch.fx.GraphModule

        ``set_voltagehook`` is used to add VoltageHook to fx_model. Three common methods are implemented here, the same as Converter.mode.

        """
        hook_cnt = -1
        for node in fx_model.graph.nodes:
            if node.op != 'call_module':
                continue
            if type(fx_model.get_submodule(node.target)) is nn.ReLU:
                hook_cnt += 1
                target = 'snn tailor.' + str(hook_cnt) + '.0'
                m = VoltageHook(momentum=momentum, mode=mode)
                new_node = Converter._add_module_and_node(fx_model, target, node, m, (node,))
        fx_model.graph.lint()
        fx_model.recompile()
        return fx_model

    @staticmethod
    def replace_by_ifnode(fx_model: 'torch.fx.GraphModule') ->torch.fx.GraphModule:
        """
        * :ref:`API in English <Converter.replace_by_ifnode-en>`

        .. _Converter.replace_by_ifnode-cn:

        :param fx_model: 原模型
        :type fx_model: torch.fx.GraphModule
        :return: 将ReLU替换为IF脉冲神经元后的模型.
        :rtype: torch.fx.GraphModule

        ``replace_by_ifnode`` 用于将模型的ReLU替换为IF脉冲神经元。

        * :ref:`中文API <Converter.replace_by_ifnode-cn>`

        .. _Converter.replace_by_ifnode-en:

        :param fx_model: Original fx_model
        :type fx_model: torch.fx.GraphModule
        :return: fx_model whose ReLU has been replaced by IF neuron.
        :rtype: torch.fx.GraphModule

        ``replace_by_ifnode`` is used to replace ReLU with IF neuron.

        """
        hook_cnt = -1
        for node in fx_model.graph.nodes:
            if node.op != 'call_module':
                continue
            if type(fx_model.get_submodule(node.target)) is VoltageHook:
                if type(fx_model.get_submodule(node.args[0].target)) is nn.ReLU:
                    hook_cnt += 1
                    hook_node = node
                    relu_node = node.args[0]
                    if len(relu_node.args) != 1:
                        raise NotImplementedError('The number of relu_node.args should be 1.')
                    s = fx_model.get_submodule(node.target).scale.item()
                    target0 = 'snn tailor.' + str(hook_cnt) + '.0'
                    target1 = 'snn tailor.' + str(hook_cnt) + '.1'
                    target2 = 'snn tailor.' + str(hook_cnt) + '.2'
                    m0 = VoltageScaler(1.0 / s)
                    m1 = neuron.IFNode(v_threshold=1.0, v_reset=None)
                    m2 = VoltageScaler(s)
                    node0 = Converter._add_module_and_node(fx_model, target0, hook_node, m0, relu_node.args)
                    node1 = Converter._add_module_and_node(fx_model, target1, node0, m1, (node0,))
                    node2 = Converter._add_module_and_node(fx_model, target2, node1, m2, args=(node1,))
                    relu_node.replace_all_uses_with(node2)
                    node2.args = node1,
                    fx_model.graph.erase_node(hook_node)
                    fx_model.graph.erase_node(relu_node)
                    fx_model.delete_all_unused_submodules()
        fx_model.graph.lint()
        fx_model.recompile()
        return fx_model

    @staticmethod
    def _add_module_and_node(fx_model: 'fx.GraphModule', target: 'str', after: 'fx.Node', m: 'nn.Module', args: 'Tuple') ->fx.Node:
        fx_model.add_submodule(target=target, m=m)
        with fx_model.graph.inserting_after(n=after):
            new_node = fx_model.graph.call_module(module_name=target, args=args)
        return new_node


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return layer.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, groups=groups, bias=False, dilation=dilation)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1, norm_layer=None, spiking_neuron: 'callable'=None, **kwargs):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = layer.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError('Dilation > 1 not supported in BasicBlock')
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.sn1 = spiking_neuron(**deepcopy(kwargs))
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.sn2 = spiking_neuron(**deepcopy(kwargs))
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.sn1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.sn2(out)
        return out


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return layer.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1, norm_layer=None, spiking_neuron: 'callable'=None, **kwargs):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = layer.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.sn1 = spiking_neuron(**deepcopy(kwargs))
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.sn2 = spiking_neuron(**deepcopy(kwargs))
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.sn3 = spiking_neuron(**deepcopy(kwargs))
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.sn1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.sn2(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.sn3(out)
        return out


class ResNet(nn.Module):

    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)
        self.flatten = nn.Flatten()
        self.relu = nn.ReLU()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = self.flatten(out)
        out = self.linear(out)
        return out


class CNN(nn.Module):

    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(nn.Conv2d(1, 32, 3, 1), nn.BatchNorm2d(32), nn.ReLU(), nn.AvgPool2d(2, 2), nn.Conv2d(32, 32, 3, 1), nn.BatchNorm2d(32), nn.ReLU(), nn.AvgPool2d(2, 2), nn.Conv2d(32, 32, 3, 1), nn.BatchNorm2d(32), nn.ReLU(), nn.AvgPool2d(2, 2), nn.Flatten(), nn.Linear(32, 10))

    def forward(self, x):
        x = self.network(x)
        return x


class StepModule:

    def supported_step_mode(self):
        """
        * :ref:`API in English <StepModule.supported_step_mode-en>`

        .. _StepModule.supported_step_mode-cn:

        :return: 包含支持的后端的tuple
        :rtype: tuple[str]

        返回此模块支持的步进模式。

        * :ref:`中文 API <StepModule.supported_step_mode-cn>`

        .. _StepModule.supported_step_mode-en:

        :return: a tuple that contains the supported backends
        :rtype: tuple[str]

        """
        return 's', 'm'

    @property
    def step_mode(self):
        """
        * :ref:`API in English <StepModule.step_mode-en>`

        .. _StepModule.step_mode-cn:

        :return: 模块当前使用的步进模式
        :rtype: str

        * :ref:`中文 API <StepModule.step_mode-cn>`

        .. _StepModule.step_mode-en:

        :return: the current step mode of this module
        :rtype: str
        """
        return self._step_mode

    @step_mode.setter
    def step_mode(self, value: 'str'):
        """
        * :ref:`API in English <StepModule.step_mode-setter-en>`

        .. _StepModule.step_mode-setter-cn:

        :param value: 步进模式
        :type value: str

        将本模块的步进模式设置为 ``value``

        * :ref:`中文 API <StepModule.step_mode-setter-cn>`

        .. _StepModule.step_mode-setter-en:

        :param value: the step mode
        :type value: str

        Set the step mode of this module to be ``value``

        """
        if value not in self.supported_step_mode():
            raise ValueError(f'step_mode can only be {self.supported_step_mode()}, but got "{value}"!')
        self._step_mode = value


def check_backend_library(backend: 'str'):
    """
    * :ref:`API in English <check_backend_library-en>`

    .. _check_backend_library-cn:

    :param backend: ``'torch'``, ``'cupy'`` 或 ``'lava'``
    :type backend: str

    检查某个后端的python库是否已经安装。若未安装则此函数会报错。

    * :ref:`中文 API <check_backend_library-cn>`

    .. _check_backend_library-en:

    :param backend: ``'torch'``, ``'cupy'`` or ``'lava'``
    :type backend: str

    Check whether the python lib for backend is installed. If not, this function will raise an error.
    """
    if backend == 'torch':
        return
    elif backend == 'cupy':
        if cupy is None:
            raise ImportError('CuPy is not installed! You can install it from "https://github.com/cupy/cupy".')
    elif backend == 'lava':
        if slayer is None:
            raise ImportError('Lava-DL is not installed! You can install it from "https://github.com/lava-nc/lava-dl". ')
    else:
        pass


class MemoryModule(nn.Module, StepModule):

    def __init__(self):
        """
        * :ref:`API in English <MemoryModule.__init__-en>`

        .. _MemoryModule.__init__-cn:

        ``MemoryModule`` 是SpikingJelly中所有有状态（记忆）模块的基类。

        * :ref:`中文API <MemoryModule.__init__-cn>`

        .. _MemoryModule.__init__-en:

        ``MemoryModule`` is the base class of all stateful modules in SpikingJelly.

        """
        super().__init__()
        self._memories = {}
        self._memories_rv = {}
        self._backend = 'torch'
        self.step_mode = 's'

    @property
    def supported_backends(self):
        """
        * :ref:`API in English <MemoryModule.supported_backends-en>`

        .. _MemoryModule.supported_backends-cn:

        返回支持的后端，默认情况下只有 `('torch', )`

        :return: 支持的后端
        :rtype: tuple[str]

        * :ref:`中文API <MemoryModule.supported_backends-cn>`

        .. _MemoryModule.supported_backends-en:

        Return the supported backends. The default return value is `('torch', )`

        :return: supported backends
        :rtype: tuple[str]

        """
        return 'torch',

    @property
    def backend(self):
        return self._backend

    @backend.setter
    def backend(self, value: 'str'):
        if value not in self.supported_backends:
            raise NotImplementedError(f'{value} is not a supported backend of {self._get_name()}!')
        check_backend_library(value)
        self._backend = value

    @abstractmethod
    def single_step_forward(self, x: 'torch.Tensor', *args, **kwargs):
        """
        * :ref:`API in English <MemoryModule.single_step_forward-en>`

        .. _MemoryModule.single_step_forward-cn:

        :param x: input tensor with ``shape = [N, *] ``
        :type x: torch.Tensor

        本模块的单步的前向传播函数


        * :ref:`中文 API <MemoryModule.single_step_forward-cn>`

        .. _MemoryModule.single_step_forward-en:

        :param x: input tensor with ``shape = [N, *] ``
        :type x: torch.Tensor

        The single-step forward function for this module

        """
        pass

    def multi_step_forward(self, x_seq: 'torch.Tensor', *args, **kwargs):
        """
        * :ref:`API in English <MemoryModule.multi_step_forward-en>`

        .. _MemoryModule.multi_step_forward-cn:

        :param x_seq: input tensor with ``shape = [T, N, *] ``
        :type x_seq: torch.Tensor

        本模块的多步的前向传播函数，通过调用 ``T`` 次 ``single_step_forward(x[t], *args, **kwargs)`` 实现


        * :ref:`中文 API <MemoryModule.multi_step_forward-cn>`

        .. _MemoryModule.multi_step_forward-en:

        :param x_seq: input tensor with ``shape = [T, N, *] ``
        :type x_seq: torch.Tensor

        The multi-step forward function for this module, which is implemented by calling ``single_step_forward(x[t], *args, **kwargs)`` over ``T`` times

        """
        T = x_seq.shape[0]
        y_seq = []
        for t in range(T):
            y = self.single_step_forward(x_seq[t], *args, **kwargs)
            y_seq.append(y.unsqueeze(0))
        return torch.cat(y_seq, 0)

    def forward(self, *args, **kwargs):
        if self.step_mode == 's':
            return self.single_step_forward(*args, **kwargs)
        elif self.step_mode == 'm':
            return self.multi_step_forward(*args, **kwargs)
        else:
            raise ValueError(self.step_mode)

    def extra_repr(self):
        return f'step_mode={self.step_mode}, backend={self.backend}'

    def register_memory(self, name: 'str', value):
        """
        * :ref:`API in English <MemoryModule.register_memory-en>`

        .. _MemoryModule.register_memory-cn:

        :param name: 变量的名字
        :type name: str
        :param value: 变量的值
        :type value: any

        将变量存入用于保存有状态变量（例如脉冲神经元的膜电位）的字典中。这个变量的重置值会被设置为 ``value``。每次调用 ``self.reset()``
        函数后， ``self.name`` 都会被重置为 ``value``。

        * :ref:`中文API <MemoryModule.register_memory-cn>`

        .. _MemoryModule.register_memory-en:

        :param name: variable's name
        :type name: str
        :param value: variable's value
        :type value: any

        Register the variable to memory dict, which saves stateful variables (e.g., the membrane potential of a
        spiking neuron). The reset value of this variable will be ``value``. ``self.name`` will be set to ``value`` after
        each calling of ``self.reset()``.

        """
        assert not hasattr(self, name), f'{name} has been set as a member variable!'
        self._memories[name] = value
        self.set_reset_value(name, value)

    def reset(self):
        """
        * :ref:`API in English <MemoryModule.reset-en>`

        .. _MemoryModule.reset-cn:

        重置所有有状态变量为默认值。

        * :ref:`中文API <MemoryModule.reset-cn>`

        .. _MemoryModule.reset-en:

        Reset all stateful variables to their default values.
        """
        for key in self._memories.keys():
            self._memories[key] = copy.deepcopy(self._memories_rv[key])

    def set_reset_value(self, name: 'str', value):
        self._memories_rv[name] = copy.deepcopy(value)

    def __getattr__(self, name: 'str'):
        if '_memories' in self.__dict__:
            memories = self.__dict__['_memories']
            if name in memories:
                return memories[name]
        return super().__getattr__(name)

    def __setattr__(self, name: 'str', value) ->None:
        _memories = self.__dict__.get('_memories')
        if _memories is not None and name in _memories:
            _memories[name] = value
        else:
            super().__setattr__(name, value)

    def __delattr__(self, name):
        if name in self._memories:
            del self._memories[name]
            del self._memories_rv[name]
        else:
            return super().__delattr__(name)

    def __dir__(self):
        module_attrs = dir(self.__class__)
        attrs = list(self.__dict__.keys())
        parameters = list(self._parameters.keys())
        modules = list(self._modules.keys())
        buffers = list(self._buffers.keys())
        memories = list(self._memories.keys())
        keys = module_attrs + attrs + parameters + modules + buffers + memories
        keys = [key for key in keys if not key[0].isdigit()]
        return sorted(keys)

    def memories(self):
        """
        * :ref:`API in English <MemoryModule.memories-en>`

        .. _MemoryModule.memories-cn:

        :return: 返回一个所有状态变量的迭代器
        :rtype: Iterator

        * :ref:`中文API <MemoryModule.memories-cn>`

        .. _MemoryModule.memories-en:

        :return: an iterator over all stateful variables
        :rtype: Iterator
        """
        for name, value in self._memories.items():
            yield value

    def named_memories(self):
        """
        * :ref:`API in English <MemoryModule.named_memories-en>`

        .. _MemoryModule.named_memories-cn:

        :return: 返回一个所有状态变量及其名称的迭代器
        :rtype: Iterator

        * :ref:`中文API <MemoryModule.named_memories-cn>`

        .. _MemoryModule.named_memories-en:

        :return: an iterator over all stateful variables and their names
        :rtype: Iterator
        """
        for name, value in self._memories.items():
            yield name, value

    def detach(self):
        """
        * :ref:`API in English <MemoryModule.detach-en>`

        .. _MemoryModule.detach-cn:

        从计算图中分离所有有状态变量。

        .. tip::

            可以使用这个函数实现TBPTT(Truncated Back Propagation Through Time)。


        * :ref:`中文API <MemoryModule.detach-cn>`

        .. _MemoryModule.detach-en:

        Detach all stateful variables.

        .. admonition:: Tip
            :class: tip

            We can use this function to implement TBPTT(Truncated Back Propagation Through Time).

        """
        for key in self._memories.keys():
            if isinstance(self._memories[key], torch.Tensor):
                self._memories[key].detach_()

    def _apply(self, fn):
        for key, value in self._memories.items():
            if isinstance(value, torch.Tensor):
                self._memories[key] = fn(value)
        return super()._apply(fn)

    def _replicate_for_data_parallel(self):
        replica = super()._replicate_for_data_parallel()
        replica._memories = self._memories.copy()
        return replica


class PopSpikeEncoderDeterministic(nn.Module):
    """ Learnable Population Coding Spike Encoder with Deterministic Spike Trains"""

    def __init__(self, obs_dim, pop_dim, spike_ts, mean_range, std):
        super().__init__()
        self.obs_dim = obs_dim
        self.pop_dim = pop_dim
        self.encoder_neuron_num = obs_dim * pop_dim
        self.spike_ts = spike_ts
        tmp_mean = torch.zeros(1, obs_dim, pop_dim)
        delta_mean = (mean_range[1] - mean_range[0]) / (pop_dim - 1)
        for num in range(pop_dim):
            tmp_mean[0, :, num] = mean_range[0] + delta_mean * num
        tmp_std = torch.zeros(1, obs_dim, pop_dim) + std
        self.mean = nn.Parameter(tmp_mean)
        self.std = nn.Parameter(tmp_std)
        self.neurons = neuron.IFNode(v_threshold=0.999, v_reset=None, surrogate_function=surrogate.DeterministicPass(), detach_reset=True)
        functional.set_step_mode(self, step_mode='m')
        functional.set_backend(self, backend='torch')

    def forward(self, obs):
        obs = obs.view(-1, self.obs_dim, 1)
        pop_act = torch.exp(-(1.0 / 2.0) * (obs - self.mean).pow(2) / self.std.pow(2)).view(-1, self.encoder_neuron_num)
        pop_act = pop_act.unsqueeze(0).repeat(self.spike_ts, 1, 1)
        return self.neurons(pop_act)


class PopSpikeEncoderRandom(nn.Module):
    """ Learnable Population Coding Spike Encoder with Random Spike Trains """

    def __init__(self, obs_dim, pop_dim, spike_ts, mean_range, std):
        super().__init__()
        self.obs_dim = obs_dim
        self.pop_dim = pop_dim
        self.encoder_neuron_num = obs_dim * pop_dim
        self.spike_ts = spike_ts
        tmp_mean = torch.zeros(1, obs_dim, pop_dim)
        delta_mean = (mean_range[1] - mean_range[0]) / (pop_dim - 1)
        for num in range(pop_dim):
            tmp_mean[0, :, num] = mean_range[0] + delta_mean * num
        tmp_std = torch.zeros(1, obs_dim, pop_dim) + std
        self.mean = nn.Parameter(tmp_mean)
        self.std = nn.Parameter(tmp_std)
        self.pseudo_spike = surrogate.poisson_pass.apply

    def forward(self, obs):
        obs = obs.view(-1, self.obs_dim, 1)
        batch_size = obs.shape[0]
        pop_act = torch.exp(-(1.0 / 2.0) * (obs - self.mean).pow(2) / self.std.pow(2)).view(-1, self.encoder_neuron_num)
        pop_spikes = torch.zeros(self.spike_ts, batch_size, self.encoder_neuron_num, device=obs.device)
        for step in range(self.spike_ts):
            pop_spikes[step, :, :] = self.pseudo_spike(pop_act)
        return pop_spikes


class PopEncoder(nn.Module):
    """ Learnable Population Coding Encoder """

    def __init__(self, obs_dim, pop_dim, spike_ts, mean_range, std):
        super().__init__()
        self.obs_dim = obs_dim
        self.pop_dim = pop_dim
        self.encoder_neuron_num = obs_dim * pop_dim
        self.spike_ts = spike_ts
        tmp_mean = torch.zeros(1, obs_dim, pop_dim)
        delta_mean = (mean_range[1] - mean_range[0]) / (pop_dim - 1)
        for num in range(pop_dim):
            tmp_mean[0, :, num] = mean_range[0] + delta_mean * num
        tmp_std = torch.zeros(1, obs_dim, pop_dim) + std
        self.mean = nn.Parameter(tmp_mean)
        self.std = nn.Parameter(tmp_std)

    def forward(self, obs):
        obs = obs.view(-1, self.obs_dim, 1)
        batch_size = obs.shape[0]
        pop_act = torch.exp(-(1.0 / 2.0) * (obs - self.mean).pow(2) / self.std.pow(2)).view(-1, self.encoder_neuron_num)
        pop_inputs = torch.zeros(self.spike_ts, batch_size, self.encoder_neuron_num, device=obs.device)
        for step in range(self.spike_ts):
            pop_inputs[step, :, :] = pop_act
        return pop_inputs


class DQN(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class WeightedMSELoss(nn.Module):

    def __init__(self, size_average=True):
        super(WeightedMSELoss, self).__init__()
        self.size_average = size_average

    def forward(self, input, target, weights=None):
        if weights is None:
            return nn.MSELoss(self.size_average)(input, target)
        loss_rows = (input - target) ** 2
        if len(loss_rows.size()) != 1:
            loss_rows = torch.sum(loss_rows, dim=1)
        res = (weights * loss_rows).sum()
        if self.size_average:
            res /= len(weights)
        return res


class DSQN(nn.Module):

    def __init__(self, input_shape, n_actions, T=5, dec_type='max-mem', use_cuda=False):
        super(DSQN, self).__init__()
        self.model_name = 'spiking_dqn'
        self.dec_type = dec_type
        if 'mem' in dec_type:
            self.network = nn.Sequential(layer.Conv2d(input_shape[0], 32, kernel_size=8, stride=4), neuron.LIFNode(surrogate_function=surrogate.ATan(), detach_reset=True), layer.Conv2d(32, 64, kernel_size=4, stride=2), neuron.LIFNode(surrogate_function=surrogate.ATan(), detach_reset=True), layer.Conv2d(64, 64, kernel_size=3, stride=1), neuron.LIFNode(surrogate_function=surrogate.ATan(), detach_reset=True), layer.Flatten(), layer.Linear(64 * 7 * 7, 512), neuron.LIFNode(surrogate_function=surrogate.ATan(), detach_reset=True), layer.Linear(512, n_actions), neuron.NonSpikingLIFNode(decode=dec_type))
        else:
            self.network = nn.Sequential(layer.Conv2d(input_shape[0], 32, kernel_size=8, stride=4), neuron.LIFNode(surrogate_function=surrogate.ATan(), detach_reset=True), layer.Conv2d(32, 64, kernel_size=4, stride=2), neuron.LIFNode(surrogate_function=surrogate.ATan(), detach_reset=True), layer.Conv2d(64, 64, kernel_size=3, stride=1), neuron.LIFNode(surrogate_function=surrogate.ATan(), detach_reset=True), layer.Flatten(), layer.Linear(64 * 7 * 7, 512), neuron.LIFNode(surrogate_function=surrogate.ATan(), detach_reset=True))
            self.decoder = nn.Linear(512, n_actions)
        self.T = T
        functional.set_step_mode(self.network, step_mode='m')
        if use_cuda:
            functional.set_backend(self.network, backend='cupy')

    def forward(self, x):
        x_seq = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)
        if 'mem' in self.dec_type:
            return self.network(x_seq)
        x_seq = self.network(x_seq)
        fr = x_seq.mean(0)
        return self.decoder(fr)


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


class MLPQFunction(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.q = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs, act):
        q = self.q(torch.cat([obs, act], dim=-1))
        return torch.squeeze(q, -1)


class NoisyPopSpikeDecoder(nn.Module):

    def __init__(self, act_dim, pop_dim, spike_ts, beta, sigma_init):
        super().__init__()
        self.act_dim = act_dim
        self.pop_dim = pop_dim
        self.spike_ts = spike_ts
        self.group_fc = layer.Conv1d(act_dim, act_dim, pop_dim, groups=act_dim)
        self.decoder = neuron.NoisyNonSpikingIFNode(act_dim, T=spike_ts, sigma_init=sigma_init, beta=beta, decode='last-mem')
        functional.set_step_mode(self, step_mode='m')

    def forward(self, out_pop_spikes):
        out_pop_spikes = out_pop_spikes.view(self.spike_ts, -1, self.act_dim, self.pop_dim)
        return self.decoder(self.group_fc(out_pop_spikes).view(self.spike_ts, -1, self.act_dim))

    def use_noise(self, is_training=True):
        self.decoder.is_training = is_training

    def reset_noise(self, num_steps):
        self.decoder.reset_noise(num_steps)

    def get_colored_noise(self):
        return self.decoder.get_colored_noise()

    def get_colored_noise_length(self):
        return self.act_dim

    def load_colored_noise(self, cn):
        self.decoder.load_colored_noise(cn)

    def cancel_load(self):
        self.decoder.cancel_load()

    def get_noise_sigma(self):
        return self.decoder.sigma.mean()


class NoisySpikeMLP(nn.Module):

    def __init__(self, in_pop_dim, act_dim, dec_pop_dim, hidden_sizes, spike_ts, beta, sigma_init):
        super().__init__()
        hidden_num = len(hidden_sizes)
        hidden_layers = OrderedDict([('linear0', layer.Linear(in_pop_dim, hidden_sizes[0])), ('sn0', neuron.NoisyCLIFNode(hidden_sizes[0], T=spike_ts, sigma_init=sigma_init, beta=beta))])
        if hidden_num > 1:
            for hidden_layer in range(1, hidden_num):
                hidden_layers['linear' + str(hidden_layer)] = layer.Linear(hidden_sizes[hidden_layer - 1], hidden_sizes[hidden_layer])
                hidden_layers['sn' + str(hidden_layer)] = neuron.NoisyCLIFNode(hidden_sizes[hidden_layer], T=spike_ts, sigma_init=sigma_init, beta=beta)
        hidden_layers['linear' + str(hidden_num)] = layer.Linear(hidden_sizes[-1], act_dim * dec_pop_dim)
        hidden_layers['sn' + str(hidden_num)] = neuron.NoisyILCCLIFNode(act_dim, dec_pop_dim, T=spike_ts, sigma_init=sigma_init, beta=beta)
        self.hidden_layers = nn.Sequential(hidden_layers)
        functional.set_step_mode(self, step_mode='m')

    def forward(self, in_pop_spikes):
        return self.hidden_layers(in_pop_spikes)

    def use_noise(self, is_training=True):
        for name, module in self.hidden_layers.named_modules():
            if not isinstance(module, layer.Linear):
                module.is_training = is_training

    def reset_noise(self, num_steps):
        for name, module in self.hidden_layers.named_modules():
            if not isinstance(module, layer.Linear):
                module.reset_noise(num_steps)

    def get_colored_noise(self):
        cn = []
        for name, module in self.hidden_layers.named_modules():
            if not isinstance(module, layer.Linear):
                cn.append(module.get_colored_noise())
        cn = torch.cat(cn, dim=1)
        return cn

    def get_colored_noise_length(self):
        length = 0
        for name, module in self.hidden_layers.named_modules():
            if not isinstance(module, layer.Linear):
                length += module.num_node * 2
        self.cn_length = length
        return length

    def load_colored_noise(self, cn):
        start_idx = 0
        for name, module in self.hidden_layers.named_modules():
            if not isinstance(module, layer.Linear):
                length = module.num_node * 2
                module.load_colored_noise(cn[:, :, start_idx:start_idx + length])
                start_idx += length

    def cancel_load(self):
        for name, module in self.hidden_layers.named_modules():
            if not isinstance(module, layer.Linear):
                module.cancel_load()


class NoisyPopSpikeActor(nn.Module):

    def __init__(self, obs_dim, act_dim, enc_pop_dim, dec_pop_dim, hidden_sizes, mean_range, std, spike_ts, act_limit, beta, sigma_init):
        super().__init__()
        self.act_limit = act_limit
        self.encoder = encoding.PopSpikeEncoderDeterministic(obs_dim, enc_pop_dim, spike_ts, mean_range, std)
        self.snn = NoisySpikeMLP(obs_dim * enc_pop_dim, act_dim, dec_pop_dim, hidden_sizes, spike_ts, beta, sigma_init)
        self.decoder = NoisyPopSpikeDecoder(act_dim, dec_pop_dim, spike_ts, beta, sigma_init)

    def forward(self, obs):
        in_pop_spikes = self.encoder(obs)
        out_pop_spikes = self.snn(in_pop_spikes)
        return self.act_limit * torch.tanh(self.decoder(out_pop_spikes))

    def act(self, obs):
        self.use_noise(False)
        in_pop_spikes = self.encoder(obs)
        out_pop_spikes = self.snn(in_pop_spikes)
        action = self.act_limit * torch.tanh(self.decoder(out_pop_spikes))
        self.use_noise(True)
        return action

    def use_noise(self, is_training=True):
        self.snn.use_noise(is_training)
        self.decoder.use_noise(is_training)

    def reset_noise(self, num_steps):
        self.snn.reset_noise(num_steps)
        self.decoder.reset_noise(num_steps)

    def get_colored_noise(self):
        cn = [self.snn.get_colored_noise(), self.decoder.get_colored_noise()]
        return torch.cat(cn, dim=1).cpu().numpy()

    def get_colored_noise_length(self):
        return self.snn.get_colored_noise_length() + self.decoder.get_colored_noise_length()

    def load_colored_noise(self, cn):
        self.snn.load_colored_noise(cn[:, :, :self.snn.cn_length])
        self.decoder.load_colored_noise(cn[:, :, self.snn.cn_length:])

    def cancel_load(self):
        self.snn.cancel_load()
        self.decoder.cancel_load()

    def get_noise_sigma(self):
        return self.decoder.get_noise_sigma()


class SpikeActorDeepCritic(nn.Module):

    def __init__(self, observation_space, action_space, encoder_pop_dim, decoder_pop_dim, mean_range, std, spike_ts, beta, sigma_init, hidden_sizes=(256, 256), activation=nn.ReLU):
        super().__init__()
        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]
        act_limit = action_space.high[0]
        self.san = NoisyPopSpikeActor(obs_dim, act_dim, encoder_pop_dim, decoder_pop_dim, hidden_sizes, mean_range, std, spike_ts, act_limit, beta, sigma_init)
        self.q1 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)
        self.q2 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)

    def act(self, obs, use_noise=True):
        with torch.no_grad():
            action = self.san.act(obs).cpu().numpy() if not use_noise else self.san(obs).cpu().numpy()
            functional.reset_net(self.san)
            return action


class SpikeMLP(nn.Module):

    def __init__(self, in_pop_dim, act_dim, dec_pop_dim, hidden_sizes):
        super().__init__()
        hidden_num = len(hidden_sizes)
        hidden_layers = OrderedDict([('Linear0', layer.Linear(in_pop_dim, hidden_sizes[0])), (neuron_type + '0', neuron.CLIFNode(surrogate_function=surrogate.Rect()))])
        if hidden_num > 1:
            for hidden_layer in range(1, hidden_num):
                hidden_layers['Linear' + str(hidden_layer)] = layer.Linear(hidden_sizes[hidden_layer - 1], hidden_sizes[hidden_layer])
                hidden_layers[neuron_type + str(hidden_layer)] = neuron.CLIFNode(surrogate_function=surrogate.Rect())
        hidden_layers['Linear' + str(hidden_num)] = layer.Linear(hidden_sizes[-1], act_dim * dec_pop_dim)
        hidden_layers[neuron_type + str(hidden_num)] = neuron.ILCCLIFNode(act_dim, dec_pop_dim, surrogate_function=surrogate.Rect())
        self.hidden_layers = nn.Sequential(hidden_layers)
        functional.set_step_mode(self, step_mode='m')

    def forward(self, in_pop_spikes):
        return self.hidden_layers(in_pop_spikes)


class PopDecoder(nn.Module):
    """ Learnable Population Coding Decoder """

    def __init__(self, act_dim, pop_dim, spike_ts, decode='last-mem'):
        super().__init__()
        self.act_dim = act_dim
        self.pop_dim = pop_dim
        self.spike_ts = spike_ts
        self.decode = decode
        if decode == 'fr-mlp':
            self.decoder = nn.Conv1d(act_dim, act_dim, pop_dim, groups=act_dim)
        else:
            self.decoder = nn.Sequential(layer.Conv1d(act_dim, act_dim, pop_dim, groups=act_dim), neuron.NonSpikingIFNode(decode=decode))
            functional.set_step_mode(self, step_mode='m')

    def forward(self, out_pop_spikes):
        if self.decode == 'fr-mlp':
            out_pop_fr = out_pop_spikes.mean(dim=0).view(-1, self.act_dim, self.pop_dim)
            return self.decoder(out_pop_fr).view(-1, self.act_dim)
        out_pop_spikes = out_pop_spikes.view(self.spike_ts, -1, self.act_dim, self.pop_dim)
        return self.decoder(out_pop_spikes).view(-1, self.act_dim)


class PopSpikeActor(nn.Module):

    def __init__(self, obs_dim, act_dim, enc_pop_dim, dec_pop_dim, hidden_sizes, mean_range, std, spike_ts, encode, decode, act_limit):
        super().__init__()
        self.act_limit = act_limit
        if encode == 'pop-det':
            self.encoder = encoding.PopSpikeEncoderDeterministic(obs_dim, enc_pop_dim, spike_ts, mean_range, std)
        elif encode == 'pop-ran':
            self.encoder = encoding.PopSpikeEncoderRandom(obs_dim, enc_pop_dim, spike_ts, mean_range, std)
        else:
            self.encoder = encoding.PopEncoder(obs_dim, enc_pop_dim, spike_ts, mean_range, std)
        self.snn = SpikeMLP(obs_dim * enc_pop_dim, act_dim, dec_pop_dim, hidden_sizes)
        self.decoder = PopDecoder(act_dim, dec_pop_dim, spike_ts, decode)

    def forward(self, obs):
        in_pop_vals = self.encoder(obs)
        out_pop_spikes = self.snn(in_pop_vals)
        return self.act_limit * torch.tanh(self.decoder(out_pop_spikes))


class ActorCritic(nn.Module):

    def __init__(self, num_inputs, num_outputs, hidden_size, T=16):
        super(ActorCritic, self).__init__()
        self.critic = nn.Sequential(layer.Linear(num_inputs, hidden_size), neuron.IFNode(), layer.Linear(hidden_size, 1), NonSpikingLIFNode(tau=2.0))
        self.actor = nn.Sequential(layer.Linear(num_inputs, hidden_size), neuron.IFNode(), layer.Linear(hidden_size, num_outputs), NonSpikingLIFNode(tau=2.0))
        self.T = T

    def forward(self, x):
        for t in range(self.T):
            self.critic(x)
            self.actor(x)
        value = self.critic[-1].v
        probs = F.softmax(self.actor[-1].v, dim=1)
        dist = Categorical(probs)
        return dist, value


class DQSN(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, T=16):
        super().__init__()
        self.fc = nn.Sequential(layer.Linear(input_size, hidden_size), neuron.IFNode(), layer.Linear(hidden_size, output_size), NonSpikingLIFNode(tau=2.0))
        self.T = T

    def forward(self, x):
        for t in range(self.T):
            self.fc(x)
        return self.fc[-1].v


class ResNet11(nn.Module):

    def __init__(self):
        super().__init__()
        self.train_epoch = 0
        self.cnn11 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.lif11 = nn.Sequential(LIFNode(), layer.Dropout(0.25))
        self.avgpool1 = nn.AvgPool2d(kernel_size=2)
        self.if1 = IFNode()
        self.cnn21 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        self.lif21 = nn.Sequential(LIFNode(), layer.Dropout(0.25))
        self.cnn22 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        self.shortcut1 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=1, stride=1, bias=False))
        self.lif2 = nn.Sequential(LIFNode(), layer.Dropout(0.25))
        self.cnn31 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)
        self.lif31 = nn.Sequential(LIFNode(), layer.Dropout(0.25))
        self.cnn32 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1, bias=False)
        self.shortcut2 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=1, stride=2, bias=False))
        self.lif3 = nn.Sequential(LIFNode(), layer.Dropout(0.25))
        self.cnn41 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False)
        self.lif41 = nn.Sequential(LIFNode(), layer.Dropout(0.25))
        self.cnn42 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False)
        self.shortcut3 = nn.Sequential(nn.Conv2d(256, 512, kernel_size=1, stride=1, bias=False))
        self.lif4 = nn.Sequential(LIFNode(), layer.Dropout(0.25))
        self.cnn51 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False)
        self.lif51 = nn.Sequential(LIFNode(), layer.Dropout(0.25))
        self.cnn52 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=1, bias=False)
        self.shortcut4 = nn.Sequential(nn.AvgPool2d(kernel_size=(1, 1), stride=(2, 2), padding=(0, 0)))
        self.lif5 = nn.Sequential(LIFNode(), layer.Dropout(0.25))
        self.fc0 = nn.Linear(512 * 4 * 4, 1024, bias=False)
        self.lif6 = nn.Sequential(LIFNode(), layer.Dropout(0.25))
        self.fc1 = nn.Linear(1024, 10, bias=False)
        self.lif_out = LIFNode(fire=False)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                variance1 = math.sqrt(1.0 / n)
                m.weight.data.normal_(0, variance1)
            elif isinstance(m, nn.Linear):
                size = m.weight.size()
                fan_in = size[1]
                variance2 = math.sqrt(1.0 / fan_in)
                m.weight.data.normal_(0.0, variance2)

    def forward(self, x):
        x = self.if1(self.avgpool1(self.lif11(self.cnn11(x))))
        x = self.lif2(self.cnn22(self.lif21(self.cnn21(x))) + self.shortcut1(x))
        x = self.lif3(self.cnn32(self.lif31(self.cnn31(x))) + self.shortcut2(x))
        x = self.lif4(self.cnn42(self.lif41(self.cnn41(x))) + self.shortcut3(x))
        x = self.lif5(self.cnn52(self.lif51(self.cnn51(x))) + self.shortcut4(x))
        out = x.view(x.size(0), -1)
        out = self.lif_out(self.fc1(self.lif6(self.fc0(out))))
        return out

    def reset_(self):
        for item in self.modules():
            if hasattr(item, 'reset'):
                item.reset()


class CSNN(nn.Module):

    def __init__(self, T: 'int', channels: 'int', use_cupy=False):
        super().__init__()
        self.T = T
        self.conv_fc = nn.Sequential(layer.Conv2d(1, channels, kernel_size=3, padding=1, bias=False), layer.BatchNorm2d(channels), neuron.IFNode(surrogate_function=surrogate.ATan()), layer.MaxPool2d(2, 2), layer.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False), layer.BatchNorm2d(channels), neuron.IFNode(surrogate_function=surrogate.ATan()), layer.MaxPool2d(2, 2), layer.Flatten(), layer.Linear(channels * 7 * 7, channels * 4 * 4, bias=False), neuron.IFNode(surrogate_function=surrogate.ATan()), layer.Linear(channels * 4 * 4, 10, bias=False), neuron.IFNode(surrogate_function=surrogate.ATan()))
        functional.set_step_mode(self, step_mode='m')
        if use_cupy:
            functional.set_backend(self, backend='cupy')

    def forward(self, x: 'torch.Tensor'):
        x_seq = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)
        x_seq = self.conv_fc(x_seq)
        fr = x_seq.mean(0)
        return fr

    def spiking_encoder(self):
        return self.conv_fc[0:3]


class MNISTNet(nn.Module):

    def __init__(self, channels=128, spiking_neuron: 'callable'=None, **kwargs):
        super().__init__()
        self.conv_fc = nn.Sequential(layer.Conv2d(1, channels, kernel_size=3, padding=1, bias=False), layer.BatchNorm2d(channels), spiking_neuron(**deepcopy(kwargs)), layer.MaxPool2d(2, 2), layer.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False), layer.BatchNorm2d(channels), spiking_neuron(**deepcopy(kwargs)), layer.MaxPool2d(2, 2), layer.Flatten(), layer.Dropout(0.5), layer.Linear(channels * 7 * 7, 2048), spiking_neuron(**deepcopy(kwargs)), layer.Dropout(0.5), layer.Linear(2048, 100), spiking_neuron(**deepcopy(kwargs)), layer.VotingLayer())

    def forward(self, x: 'torch.Tensor'):
        return self.conv_fc(x)


class SNN(nn.Module):

    def __init__(self, tau):
        super().__init__()
        self.layer = nn.Sequential(layer.Flatten(), layer.Linear(28 * 28, 10, bias=False), neuron.LIFNode(tau=tau, surrogate_function=surrogate.ATan()))

    def forward(self, x: 'torch.Tensor'):
        return self.layer(x)


class InferenceNet(nn.Module):

    def __init__(self, T: 'int', modules_list: 'list'):
        super().__init__()
        self.T = T
        self.module_list = nn.Sequential(*modules_list)

    def forward(self, x: 'torch.Tensor'):
        x = x.repeat(self.T, 1, 1, 1)
        x = self.module_list(x)
        x = x.reshape(self.T, x.shape[0] // self.T, -1)
        return x.mean(0)


class PlainNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(layer.Linear(28, 32), neuron.IFNode(surrogate_function=surrogate.ATan()), layer.Linear(32, 10), neuron.IFNode(surrogate_function=surrogate.ATan()))

    def forward(self, x: 'torch.Tensor'):
        return self.fc(x).mean(0)


class StatefulSynapseNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(layer.Linear(28, 32), neuron.IFNode(surrogate_function=surrogate.ATan()), layer.SynapseFilter(tau=2.0, learnable=True), layer.Linear(32, 10), neuron.IFNode(surrogate_function=surrogate.ATan()))

    def forward(self, x: 'torch.Tensor'):
        return self.fc(x).mean(0)


class FeedBackNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(layer.Linear(28, 32), layer.LinearRecurrentContainer(neuron.IFNode(surrogate_function=surrogate.ATan(), detach_reset=True), in_features=32, out_features=32, bias=True), layer.Linear(32, 10), neuron.IFNode(surrogate_function=surrogate.ATan()))

    def forward(self, x: 'torch.Tensor'):
        return self.fc(x).mean(0)


def hz_to_mel(frequencies, dct_type):
    if dct_type == 'htk':
        if torch.is_tensor(frequencies) and frequencies.ndim:
            return 2595.0 * torch.log10(1.0 + frequencies / 700.0)
        return 2595.0 * math.log10(1.0 + frequencies / 700.0)
    f_min = 0.0
    f_sp = 200.0 / 3
    mels = (frequencies - f_min) / f_sp
    min_log_hz = 1000.0
    min_log_mel = (min_log_hz - f_min) / f_sp
    logstep = math.log(6.4) / 27.0
    if torch.is_tensor(frequencies) and frequencies.ndim:
        log_t = frequencies >= min_log_hz
        mels[log_t] = min_log_mel + torch.log(frequencies[log_t] / min_log_hz) / logstep
    elif frequencies >= min_log_hz:
        mels = min_log_mel + math.log(frequencies / min_log_hz) / logstep
    return mels


def mel_to_hz(mels, dct_type):
    if dct_type == 'htk':
        return 700.0 * (10 ** (mels / 2595.0) - 1.0)
    f_min = 0.0
    f_sp = 200.0 / 3
    freqs = f_min + f_sp * mels
    min_log_hz = 1000.0
    min_log_mel = (min_log_hz - f_min) / f_sp
    logstep = math.log(6.4) / 27.0
    if torch.is_tensor(mels) and mels.ndim:
        log_t = mels >= min_log_mel
        freqs[log_t] = min_log_hz * torch.exp(logstep * (mels[log_t] - min_log_mel))
    elif mels >= min_log_mel:
        freqs = min_log_hz * math.exp(logstep * (mels - min_log_mel))
    return freqs


def create_fb_matrix(n_freqs: 'int', f_min: 'float', f_max: 'float', n_mels: 'int', sample_rate: 'int', dct_type: 'Optional[str]'='slaney') ->Tensor:
    if dct_type != 'htk' and dct_type != 'slaney':
        raise ValueError("DCT type must be either 'htk' or 'slaney'")
    all_freqs = torch.linspace(0, sample_rate // 2, n_freqs)
    m_min = hz_to_mel(f_min, dct_type)
    m_max = hz_to_mel(f_max, dct_type)
    m_pts = torch.linspace(m_min, m_max, n_mels + 2)
    f_pts = mel_to_hz(m_pts, dct_type)
    f_diff = f_pts[1:] - f_pts[:-1]
    slopes = f_pts.unsqueeze(0) - all_freqs.unsqueeze(1)
    zero = torch.zeros(1)
    down_slopes = -1.0 * slopes[:, :-2] / f_diff[:-1]
    up_slopes = slopes[:, 2:] / f_diff[1:]
    fb = torch.max(zero, torch.min(down_slopes, up_slopes))
    if dct_type == 'slaney':
        enorm = 2.0 / (f_pts[2:n_mels + 2] - f_pts[:n_mels])
        fb *= enorm.unsqueeze(0)
    return fb


class MelScaleDelta(nn.Module):
    __constants__ = ['n_mels', 'sample_rate', 'f_min', 'f_max']

    def __init__(self, order, n_mels: 'int'=128, sample_rate: 'int'=16000, f_min: 'float'=0.0, f_max: 'Optional[float]'=None, n_stft: 'Optional[int]'=None, dct_type: 'Optional[str]'='slaney') ->None:
        super(MelScaleDelta, self).__init__()
        self.order = order
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.f_max = f_max if f_max is not None else float(sample_rate // 2)
        self.f_min = f_min
        self.dct_type = dct_type
        assert f_min <= self.f_max, 'Require f_min: {} < f_max: {}'.format(f_min, self.f_max)
        fb = torch.empty(0) if n_stft is None else create_fb_matrix(n_stft, self.f_min, self.f_max, self.n_mels, self.sample_rate, self.dct_type)
        self.register_buffer('fb', fb)

    def forward(self, specgram: 'Tensor') ->Tensor:
        shape = specgram.size()
        specgram = specgram.reshape(-1, shape[-2], shape[-1])
        if self.fb.numel() == 0:
            tmp_fb = create_fb_matrix(specgram.size(1), self.f_min, self.f_max, self.n_mels, self.sample_rate, self.dct_type)
            self.fb.resize_(tmp_fb.size())
            self.fb.copy_(tmp_fb)
        mel_specgram = torch.matmul(specgram.transpose(1, 2), self.fb).transpose(1, 2)
        mel_specgram = mel_specgram.reshape(shape[:-2] + mel_specgram.shape[-2:]).squeeze()
        M = torch.max(torch.abs(mel_specgram))
        if M > 0:
            feat = torch.log1p(mel_specgram / M)
        else:
            feat = mel_specgram
        feat_list = [feat.numpy().T]
        for k in range(1, self.order + 1):
            feat_list.append(savgol_filter(feat.numpy(), 9, deriv=k, axis=-1, mode='interp', polyorder=k).T)
        return torch.as_tensor(np.expand_dims(np.stack(feat_list), axis=0))


class LIFWrapper(nn.Module):

    def __init__(self, module, flatten=False):
        super().__init__()
        self.module = module
        self.flatten = flatten

    def forward(self, x_seq: 'torch.Tensor'):
        """
        :param x_seq: shape=[batch size, channel, T, n_mel]
        :type x_seq: torch.Tensor
        :return: y_seq, shape=[batch size, channel, T, n_mel]
        :rtype: torch.Tensor
        """
        y_seq = self.module(x_seq.transpose(0, 2))
        if self.flatten:
            y_seq = y_seq.permute(2, 0, 1, 3)
            shape = y_seq.shape[:2]
            return y_seq.reshape(shape + (-1,))
        else:
            return y_seq.transpose(0, 2)


class Net(nn.Module):

    def __init__(self, m, T):
        super().__init__()
        self.tempotron = neuron.Tempotron(28 * 28 * m, 10, T)

    def forward(self, x: 'torch.Tensor'):
        return self.tempotron(x, 'v_max')


_hw_bits = 12


@torch.jit.script
def _listep_backward(grad_output: 'torch.Tensor', decay: 'torch.Tensor', state: 'torch.Tensor', hw_bits: 'int'=12):
    grad_state = (1 - decay / (1 << hw_bits)) * grad_output
    grad_decay = -state / (1 << hw_bits) * grad_output
    grad_decay = grad_decay.sum()
    return grad_output, grad_decay, grad_state


@torch.jit.script
def right_shift_to_zero(x: 'torch.Tensor', bits: 'int'):
    dtype = x.dtype
    assert dtype in (torch.int32, torch.int64)
    return torch.sign(x) * (torch.abs(x) >> bits)


@torch.jit.script
def _listep_forward(x: 'torch.Tensor', decay: 'torch.Tensor', state: 'torch.Tensor', w_scale: 'int', dtype: 'torch.dtype'=torch.int32, hw_bits: 'int'=12):
    scaled_state = state * w_scale
    decay_int = (1 << hw_bits) - decay
    output = right_shift_to_zero(scaled_state * decay_int, hw_bits) + w_scale * x
    return output / w_scale


class LeakyIntegratorStep(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, decay, state, w_scale):
        output = _listep_forward(x, decay, state, w_scale, dtype=torch.int64, hw_bits=_hw_bits)
        if x.requires_grad or state.requires_grad:
            ctx.save_for_backward(decay, state)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        decay, state = ctx.saved_tensors
        grad_input, grad_decay, grad_state = _listep_backward(grad_output, decay, state, hw_bits=_hw_bits)
        return grad_input, grad_decay, grad_state, None


@torch.jit.script
def step_quantize_forward(x: 'torch.Tensor', step: 'float'):
    return torch.round_(x / step) * step


class step_quantize_atgf(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x: 'torch.Tensor', step: 'float'):
        return step_quantize_forward(x, step)

    @staticmethod
    def backward(ctx, grad_output: 'torch.Tensor'):
        return grad_output, None


@torch.jit.ignore
def step_quantize(x: 'torch.Tensor', step: 'float'):
    """
    :param x: the input tensor
    :type x: torch.Tensor
    :param step: the quantize step
    :type step: float
    :return: the quantized tensor
    :rtype: torch.Tensor

    Quantize ``x`` to the nearest ``i * step``, where ``i`` is an integer.

    Note that the gradient is defined by :math:`\\frac{\\partial y}{\\partial x} = 1`.

    .. image:: ../_static/API/activation_based//quantize/step_quantize.*
        :width: 100%

    """
    return step_quantize_atgf.apply(x, step)


def quantize_8b(x, scale, descale=False):
    """
    Denote ``k`` as an ``int``, ``x[i]`` will be quantized to the nearest ``2 * k / scale``,     and ``k = {-128, -127, ..., 126, 127}``.
    """
    if not descale:
        return step_quantize(x, step=2 / scale).clamp(-256 / scale, 255 / scale)
    else:
        return step_quantize(x, step=2 / scale).clamp(-256 / scale, 255 / scale) * scale


class PrintShapeModule(nn.Module):

    def __init__(self, ext_str='PrintShapeModule'):
        """
        * :ref:`API in English <PrintModule.__init__-en>`

        .. _PrintModule.__init__-cn:

        :param ext_str: 额外打印的字符串
        :type ext_str: str

        只打印 ``ext_str`` 和输入的 ``shape``，不进行任何操作的网络层，可以用于debug。

        * :ref:`中文API <PrintModule.__init__-cn>`

        .. _PrintModule.__init__-en:

        :param ext_str: extra strings for printing
        :type ext_str: str

        This layer will not do any operation but print ``ext_str`` and the shape of input, which can be used for debugging.

        """
        super().__init__()
        self.ext_str = ext_str

    def forward(self, x: 'Tensor'):
        None
        return x


class ReplaceforGrad(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, x_r):
        return x_r

    @staticmethod
    def backward(ctx, grad):
        return grad, grad


class GradwithTrace(nn.Module):

    def __init__(self, module):
        """
        * :ref:`API in English <GradwithTrace-en>`

        .. _GradwithTrace-cn:

        :param module: 需要包装的模块

        用于随时间在线训练时，根据神经元的迹计算梯度
        出处：'Online Training Through Time for Spiking Neural Networks <https://openreview.net/forum?id=Siv3nHYHheI>'

        * :ref:`中文 API <GradwithTrace-cn>`

        .. _GradwithTrace-en:

        :param module: the module that requires wrapping

        Used for online training through time, calculate gradients by the traces of neurons
        Reference: 'Online Training Through Time for Spiking Neural Networks <https://openreview.net/forum?id=Siv3nHYHheI>'

        """
        super().__init__()
        self.module = module

    def forward(self, x: 'Tensor'):
        spike, trace = x[0], x[1]
        with torch.no_grad():
            out = self.module(spike).detach()
        in_for_grad = ReplaceforGrad.apply(spike, trace)
        out_for_grad = self.module(in_for_grad)
        x = ReplaceforGrad.apply(out_for_grad, out)
        return x


class SpikeTraceOp(nn.Module):

    def __init__(self, module):
        """
        * :ref:`API in English <SpikeTraceOp-en>`

        .. _SpikeTraceOp-cn:

        :param module: 需要包装的模块

        对脉冲和迹进行相同的运算，如Dropout，AvgPool等

        * :ref:`中文 API <GradwithTrace-cn>`

        .. _SpikeTraceOp-en:

        :param module: the module that requires wrapping

        perform the same operations for spike and trace, such as Dropout, Avgpool, etc.

        """
        super().__init__()
        self.module = module

    def forward(self, x: 'Tensor'):
        spike, trace = x[0], x[1]
        spike = self.module(spike)
        with torch.no_grad():
            trace = self.module(trace)
        x = [spike, trace]
        return x


class OTTTSequential(nn.Sequential):

    def __init__(self, *args):
        super().__init__(*args)

    def forward(self, input):
        for module in self:
            if not isinstance(input, list):
                input = module(input)
            else:
                if len(list(module.parameters())) > 0:
                    module = GradwithTrace(module)
                else:
                    module = SpikeTraceOp(module)
                input = module(input)
        return input


def stdp_conv1d_single_step(conv: 'nn.Conv1d', in_spike: 'torch.Tensor', out_spike: 'torch.Tensor', trace_pre: 'Union[torch.Tensor, None]', trace_post: 'Union[torch.Tensor, None]', tau_pre: 'float', tau_post: 'float', f_pre: 'Callable'=lambda x: x, f_post: 'Callable'=lambda x: x):
    if conv.dilation != (1,):
        raise NotImplementedError('STDP with dilation != 1 for Conv1d has not been implemented!')
    if conv.groups != 1:
        raise NotImplementedError('STDP with groups != 1 for Conv1d has not been implemented!')
    stride_l = conv.stride[0]
    if conv.padding == (0,):
        pass
    else:
        pL = conv.padding[0]
        if conv.padding_mode != 'zeros':
            in_spike = F.pad(in_spike, conv._reversed_padding_repeated_twice, mode=conv.padding_mode)
        else:
            in_spike = F.pad(in_spike, pad=(pL, pL))
    if trace_pre is None:
        trace_pre = torch.zeros_like(in_spike, device=in_spike.device, dtype=in_spike.dtype)
    if trace_post is None:
        trace_post = torch.zeros_like(out_spike, device=in_spike.device, dtype=in_spike.dtype)
    trace_pre = trace_pre - trace_pre / tau_pre + in_spike
    trace_post = trace_post - trace_post / tau_post + out_spike
    delta_w = torch.zeros_like(conv.weight.data)
    for l in range(conv.weight.shape[2]):
        l_end = in_spike.shape[2] - conv.weight.shape[2] + 1 + l
        pre_spike = in_spike[:, :, l:l_end:stride_l]
        post_spike = out_spike
        weight = conv.weight.data[:, :, l]
        tr_pre = trace_pre[:, :, l:l_end:stride_l]
        tr_post = trace_post
        delta_w_pre = -(f_pre(weight) * (tr_post.unsqueeze(2) * pre_spike.unsqueeze(1)).permute([1, 2, 0, 3]).sum(dim=[2, 3]))
        delta_w_post = f_post(weight) * (tr_pre.unsqueeze(1) * post_spike.unsqueeze(2)).permute([1, 2, 0, 3]).sum(dim=[2, 3])
        delta_w[:, :, l] += delta_w_pre + delta_w_post
    return trace_pre, trace_post, delta_w


def stdp_conv2d_single_step(conv: 'nn.Conv2d', in_spike: 'torch.Tensor', out_spike: 'torch.Tensor', trace_pre: 'Union[torch.Tensor, None]', trace_post: 'Union[torch.Tensor, None]', tau_pre: 'float', tau_post: 'float', f_pre: 'Callable'=lambda x: x, f_post: 'Callable'=lambda x: x):
    if conv.dilation != (1, 1):
        raise NotImplementedError('STDP with dilation != 1 for Conv2d has not been implemented!')
    if conv.groups != 1:
        raise NotImplementedError('STDP with groups != 1 for Conv2d has not been implemented!')
    stride_h = conv.stride[0]
    stride_w = conv.stride[1]
    if conv.padding == (0, 0):
        pass
    else:
        pH = conv.padding[0]
        pW = conv.padding[1]
        if conv.padding_mode != 'zeros':
            in_spike = F.pad(in_spike, conv._reversed_padding_repeated_twice, mode=conv.padding_mode)
        else:
            in_spike = F.pad(in_spike, pad=(pW, pW, pH, pH))
    if trace_pre is None:
        trace_pre = torch.zeros_like(in_spike, device=in_spike.device, dtype=in_spike.dtype)
    if trace_post is None:
        trace_post = torch.zeros_like(out_spike, device=in_spike.device, dtype=in_spike.dtype)
    trace_pre = trace_pre - trace_pre / tau_pre + in_spike
    trace_post = trace_post - trace_post / tau_post + out_spike
    delta_w = torch.zeros_like(conv.weight.data)
    for h in range(conv.weight.shape[2]):
        for w in range(conv.weight.shape[3]):
            h_end = in_spike.shape[2] - conv.weight.shape[2] + 1 + h
            w_end = in_spike.shape[3] - conv.weight.shape[3] + 1 + w
            pre_spike = in_spike[:, :, h:h_end:stride_h, w:w_end:stride_w]
            post_spike = out_spike
            weight = conv.weight.data[:, :, h, w]
            tr_pre = trace_pre[:, :, h:h_end:stride_h, w:w_end:stride_w]
            tr_post = trace_post
            delta_w_pre = -(f_pre(weight) * (tr_post.unsqueeze(2) * pre_spike.unsqueeze(1)).permute([1, 2, 0, 3, 4]).sum(dim=[2, 3, 4]))
            delta_w_post = f_post(weight) * (tr_pre.unsqueeze(1) * post_spike.unsqueeze(2)).permute([1, 2, 0, 3, 4]).sum(dim=[2, 3, 4])
            delta_w[:, :, h, w] += delta_w_pre + delta_w_post
    return trace_pre, trace_post, delta_w


def stdp_linear_single_step(fc: 'nn.Linear', in_spike: 'torch.Tensor', out_spike: 'torch.Tensor', trace_pre: 'Union[float, torch.Tensor, None]', trace_post: 'Union[float, torch.Tensor, None]', tau_pre: 'float', tau_post: 'float', f_pre: 'Callable'=lambda x: x, f_post: 'Callable'=lambda x: x):
    if trace_pre is None:
        trace_pre = 0.0
    if trace_post is None:
        trace_post = 0.0
    weight = fc.weight.data
    trace_pre = trace_pre - trace_pre / tau_pre + in_spike
    trace_post = trace_post - trace_post / tau_post + out_spike
    delta_w_pre = -f_pre(weight) * (trace_post.unsqueeze(2) * in_spike.unsqueeze(1)).sum(0)
    delta_w_post = f_post(weight) * (trace_pre.unsqueeze(1) * out_spike.unsqueeze(2)).sum(0)
    return trace_pre, trace_post, delta_w_pre + delta_w_post


def stdp_multi_step(layer: 'Union[nn.Linear, nn.Conv1d, nn.Conv2d]', in_spike: 'torch.Tensor', out_spike: 'torch.Tensor', trace_pre: 'Union[float, torch.Tensor, None]', trace_post: 'Union[float, torch.Tensor, None]', tau_pre: 'float', tau_post: 'float', f_pre: 'Callable'=lambda x: x, f_post: 'Callable'=lambda x: x):
    weight = layer.weight.data
    delta_w = torch.zeros_like(weight)
    T = in_spike.shape[0]
    if isinstance(layer, nn.Linear):
        stdp_single_step = stdp_linear_single_step
    elif isinstance(layer, nn.Conv1d):
        stdp_single_step = stdp_conv1d_single_step
    elif isinstance(layer, nn.Conv2d):
        stdp_single_step = stdp_conv2d_single_step
    for t in range(T):
        trace_pre, trace_post, dw = stdp_single_step(layer, in_spike[t], out_spike[t], trace_pre, trace_post, tau_pre, tau_post, f_pre, f_post)
        delta_w += dw
    return trace_pre, trace_post, delta_w


def mstdp_linear_single_step(fc: 'nn.Linear', in_spike: 'torch.Tensor', out_spike: 'torch.Tensor', trace_pre: 'Union[float, torch.Tensor, None]', trace_post: 'Union[float, torch.Tensor, None]', tau_pre: 'float', tau_post: 'float', f_pre: 'Callable'=lambda x: x, f_post: 'Callable'=lambda x: x):
    if trace_pre is None:
        trace_pre = 0.0
    if trace_post is None:
        trace_post = 0.0
    weight = fc.weight.data
    trace_pre = trace_pre * math.exp(-1 / tau_pre) + in_spike
    trace_post = trace_post * math.exp(-1 / tau_post) + out_spike
    eligibility = f_post(weight) * (trace_pre.unsqueeze(1) * out_spike.unsqueeze(2)) - f_pre(weight) * (trace_post.unsqueeze(2) * in_spike.unsqueeze(1))
    return trace_pre, trace_post, eligibility


def mstdpet_linear_single_step(fc: 'nn.Linear', in_spike: 'torch.Tensor', out_spike: 'torch.Tensor', trace_pre: 'Union[float, torch.Tensor, None]', trace_post: 'Union[float, torch.Tensor, None]', tau_pre: 'float', tau_post: 'float', tau_trace: 'float', f_pre: 'Callable'=lambda x: x, f_post: 'Callable'=lambda x: x):
    if trace_pre is None:
        trace_pre = 0.0
    if trace_post is None:
        trace_post = 0.0
    weight = fc.weight.data
    trace_pre = trace_pre * math.exp(-1 / tau_pre) + in_spike
    trace_post = trace_post * math.exp(-1 / tau_post) + out_spike
    eligibility = f_post(weight) * torch.outer(out_spike, trace_pre) - f_pre(weight) * torch.outer(trace_post, in_spike)
    return trace_pre, trace_post, eligibility


class FashionMNISTNet(MNISTNet):
    pass


class NMNISTNet(MNISTNet):

    def __init__(self, channels=128, spiking_neuron: 'callable'=None, **kwargs):
        super().__init__(channels, spiking_neuron, **kwargs)
        self.conv_fc[0] = layer.Conv2d(2, channels, kernel_size=3, padding=1, bias=False)
        self.conv_fc[-6] = layer.Linear(channels * 8 * 8, 2048)


class CIFAR10Net(nn.Module):

    def __init__(self, channels=256, spiking_neuron: 'callable'=None, **kwargs):
        super().__init__()
        conv = []
        for i in range(2):
            for j in range(3):
                if conv.__len__() == 0:
                    in_channels = 3
                else:
                    in_channels = channels
                conv.append(layer.Conv2d(in_channels, channels, kernel_size=3, padding=1, bias=False))
                conv.append(layer.BatchNorm2d(channels))
                conv.append(spiking_neuron(**deepcopy(kwargs)))
            conv.append(layer.MaxPool2d(2, 2))
        self.conv_fc = nn.Sequential(*conv, layer.Flatten(), layer.Dropout(0.5), layer.Linear(channels * 8 * 8, 2048), spiking_neuron(**deepcopy(kwargs)), layer.Dropout(0.5), layer.Linear(2048, 100), spiking_neuron(**deepcopy(kwargs)), layer.VotingLayer(10))

    def forward(self, x):
        return self.conv_fc(x)


class CIFAR10DVSNet(nn.Module):

    def __init__(self, channels=128, spiking_neuron: 'callable'=None, **kwargs):
        super().__init__()
        conv = []
        for i in range(4):
            if conv.__len__() == 0:
                in_channels = 2
            else:
                in_channels = channels
            conv.append(layer.Conv2d(in_channels, channels, kernel_size=3, padding=1, bias=False))
            conv.append(layer.BatchNorm2d(channels))
            conv.append(spiking_neuron(**deepcopy(kwargs)))
            conv.append(layer.MaxPool2d(2, 2))
        self.conv_fc = nn.Sequential(*conv, layer.Flatten(), layer.Dropout(0.5), layer.Linear(channels * 8 * 8, 512), spiking_neuron(**deepcopy(kwargs)), layer.Dropout(0.5), layer.Linear(512, 100), spiking_neuron(**deepcopy(kwargs)), layer.VotingLayer(10))

    def forward(self, x: 'torch.Tensor'):
        return self.conv_fc(x)


class DVSGestureNet(nn.Module):

    def __init__(self, channels=128, spiking_neuron: 'callable'=None, **kwargs):
        super().__init__()
        conv = []
        for i in range(5):
            if conv.__len__() == 0:
                in_channels = 2
            else:
                in_channels = channels
            conv.append(layer.Conv2d(in_channels, channels, kernel_size=3, padding=1, bias=False))
            conv.append(layer.BatchNorm2d(channels))
            conv.append(spiking_neuron(**deepcopy(kwargs)))
            conv.append(layer.MaxPool2d(2, 2))
        self.conv_fc = nn.Sequential(*conv, layer.Flatten(), layer.Dropout(0.5), layer.Linear(channels * 4 * 4, 512), spiking_neuron(**deepcopy(kwargs)), layer.Dropout(0.5), layer.Linear(512, 110), spiking_neuron(**deepcopy(kwargs)), layer.VotingLayer(10))

    def forward(self, x: 'torch.Tensor'):
        return self.conv_fc(x)


class SEWResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False, groups=1, width_per_group=64, replace_stride_with_dilation=None, norm_layer=None, cnf: 'str'=None, spiking_neuron: 'callable'=None, **kwargs):
        super().__init__()
        if norm_layer is None:
            norm_layer = layer.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError('replace_stride_with_dilation should be None or a 3-element tuple, got {}'.format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = layer.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.sn1 = spiking_neuron(**deepcopy(kwargs))
        self.maxpool = layer.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], cnf=cnf, spiking_neuron=spiking_neuron, **kwargs)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0], cnf=cnf, spiking_neuron=spiking_neuron, **kwargs)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1], cnf=cnf, spiking_neuron=spiking_neuron, **kwargs)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2], cnf=cnf, spiking_neuron=spiking_neuron, **kwargs)
        self.avgpool = layer.AdaptiveAvgPool2d((1, 1))
        self.fc = layer.Linear(512 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, layer.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (layer.BatchNorm2d, layer.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False, cnf: 'str'=None, spiking_neuron: 'callable'=None, **kwargs):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(conv1x1(self.inplanes, planes * block.expansion, stride), norm_layer(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer, cnf, spiking_neuron, **kwargs))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups, base_width=self.base_width, dilation=self.dilation, norm_layer=norm_layer, cnf=cnf, spiking_neuron=spiking_neuron, **kwargs))
        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.sn1(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        if self.avgpool.step_mode == 's':
            x = torch.flatten(x, 1)
        elif self.avgpool.step_mode == 'm':
            x = torch.flatten(x, 2)
        x = self.fc(x)
        return x

    def forward(self, x):
        return self._forward_impl(x)


class ScaleLayer(nn.Module):

    def __init__(self):
        super().__init__()
        self.scale = torch.tensor(0.0)

    def forward(self, input):
        return input * self.scale


class Neuronal_Cell(nn.Module):

    def __init__(self, args, in_channel, out_channel, con_mat):
        """
        :param args: additional arguments
        :param in_channel: input channel
        :type in_channel: int
        :param out_channel: output channel
        :type out_channel: int
        :param con_mat: connection matrix
        :type con_mat: torch.Tensor

        Neuronal forward cell.

        """
        super(Neuronal_Cell, self).__init__()
        self.cell_architecture = nn.ModuleList([])
        self.con_mat = con_mat
        for col in range(1, 4):
            for row in range(col):
                op = con_mat[row, col]
                if op == 0:
                    self.cell_architecture.append(ScaleLayer())
                elif op == 1:
                    self.cell_architecture.append(nn.Identity())
                elif op == 2:
                    self.cell_architecture.append(nn.Sequential(neuron.LIFNode(v_threshold=args.threshold, v_reset=0.0, tau=args.tau, surrogate_function=surrogate.ATan(), detach_reset=True), nn.Conv2d(in_channel, out_channel, kernel_size=(1, 1), stride=(1, 1), bias=False), nn.BatchNorm2d(out_channel, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)))
                elif op == 3:
                    self.cell_architecture.append(nn.Sequential(neuron.LIFNode(v_threshold=args.threshold, v_reset=0.0, tau=args.tau, surrogate_function=surrogate.ATan(), detach_reset=True), nn.Conv2d(in_channel, out_channel, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False), nn.BatchNorm2d(out_channel, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)))
                elif op == 4:
                    self.cell_architecture.append(nn.AvgPool2d(kernel_size=3, stride=1, padding=1))

    def forward(self, x_in):
        x_1 = self.cell_architecture[0](x_in)
        x_2 = self.cell_architecture[1](x_in) + self.cell_architecture[2](x_1)
        x_3 = self.cell_architecture[3](x_in) + self.cell_architecture[4](x_1) + self.cell_architecture[5](x_2)
        return x_3


class Neuronal_Cell_backward(nn.Module):

    def __init__(self, args, in_channel, out_channel, con_mat):
        """
        :param args: additional arguments
        :param in_channel: input channel
        :type in_channel: int
        :param out_channel: output channel
        :type out_channel: int
        :param con_mat: connection matrix
        :type con_mat: torch.Tensor

        Neuronal backward cell.

        """
        super(Neuronal_Cell_backward, self).__init__()
        self.cell_architecture = nn.ModuleList([])
        self.con_mat = con_mat
        self.cell_architecture_back = nn.ModuleList([])
        self.last_xin = 0.0
        self.last_x1 = 0.0
        self.last_x2 = 0.0
        for col in range(1, 4):
            for row in range(col):
                op = con_mat[row, col]
                if op == 0:
                    self.cell_architecture.append(ScaleLayer())
                elif op == 1:
                    self.cell_architecture.append(nn.Identity())
                elif op == 2:
                    self.cell_architecture.append(nn.Sequential(neuron.LIFNode(v_threshold=args.threshold, v_reset=0.0, tau=args.tau, surrogate_function=surrogate.ATan(), detach_reset=True), nn.Conv2d(in_channel, out_channel, kernel_size=(1, 1), stride=(1, 1), bias=False), nn.BatchNorm2d(out_channel, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)))
                elif op == 3:
                    self.cell_architecture.append(nn.Sequential(neuron.LIFNode(v_threshold=args.threshold, v_reset=0.0, tau=args.tau, surrogate_function=surrogate.ATan(), detach_reset=True), nn.Conv2d(in_channel, out_channel, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False), nn.BatchNorm2d(out_channel, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)))
                elif op == 4:
                    self.cell_architecture.append(nn.AvgPool2d(kernel_size=3, stride=1, padding=1))
        for col in range(0, 3):
            for row in range(col + 1, 4):
                op = con_mat[row, col]
                if op == 0:
                    self.cell_architecture_back.append(ScaleLayer())
                elif op == 1:
                    self.cell_architecture_back.append(nn.Identity())
                elif op == 2:
                    self.cell_architecture_back.append(nn.Sequential(neuron.LIFNode(v_threshold=args.threshold, v_reset=0.0, tau=args.tau, surrogate_function=surrogate.ATan(), detach_reset=True), nn.Conv2d(in_channel, out_channel, kernel_size=(1, 1), stride=(1, 1), bias=False), nn.BatchNorm2d(out_channel, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)))
                elif op == 3:
                    self.cell_architecture_back.append(nn.Sequential(neuron.LIFNode(v_threshold=args.threshold, v_reset=0.0, tau=args.tau, surrogate_function=surrogate.ATan(), detach_reset=True), nn.Conv2d(in_channel, out_channel, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False), nn.BatchNorm2d(out_channel, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)))
                elif op == 4:
                    self.cell_architecture_back.append(nn.AvgPool2d(kernel_size=3, stride=1, padding=1))

    def forward(self, x_in):
        x_1 = self.cell_architecture[0](x_in + self.last_xin)
        x_2 = self.cell_architecture[1](x_in + self.last_xin) + self.cell_architecture[2](x_1 + self.last_x1)
        x_3 = self.cell_architecture[3](x_in + self.last_xin) + self.cell_architecture[4](x_1 + self.last_x1) + self.cell_architecture[5](x_2 + self.last_x2)
        self.last_xin = self.cell_architecture_back[0](x_1 + self.last_x1) + self.cell_architecture_back[1](x_2 + self.last_x2) + self.cell_architecture_back[2](x_3)
        self.last_x1 = self.cell_architecture_back[3](x_2 + self.last_x2) + self.cell_architecture_back[4](x_3)
        self.last_x2 = self.cell_architecture_back[5](x_3)
        return x_3


class SNASNet(nn.Module):

    def __init__(self, args, con_mat):
        """
        :param args: additional arguments
        :param con_mat: connection matrix
        :type con_mat: torch.Tensor

        The SNASNet `Neural Architecture Search for Spiking Neural Networks <https://arxiv.org/abs/2201.10355>`_ implementation by Spikingjelly.

        """
        super(SNASNet, self).__init__()
        self.con_mat = con_mat
        self.total_timestep = args.timestep
        self.second_avgpooling = args.second_avgpooling
        if args.dataset == 'cifar10':
            self.num_class = 10
            self.num_final_neuron = 100
            self.num_cluster = 10
            self.in_channel = 3
            self.img_size = 32
            self.first_out_channel = 128
            self.channel_ratio = 2
            self.spatial_decay = 2 * self.second_avgpooling
            self.classifier_inter_ch = 1024
            self.stem_stride = 1
        elif args.dataset == 'cifar100':
            self.num_class = 100
            self.num_final_neuron = 500
            self.num_cluster = 5
            self.in_channel = 3
            self.img_size = 32
            self.channel_ratio = 1
            self.first_out_channel = 128
            self.spatial_decay = 2 * self.second_avgpooling
            self.classifier_inter_ch = 1024
            self.stem_stride = 1
        elif args.dataset == 'tinyimagenet':
            self.num_class = 200
            self.num_final_neuron = 1000
            self.num_cluster = 5
            self.in_channel = 3
            self.img_size = 64
            self.first_out_channel = 128
            self.channel_ratio = 1
            self.spatial_decay = 4 * self.second_avgpooling
            self.classifier_inter_ch = 4096
            self.stem_stride = 2
        self.stem = nn.Sequential(nn.Conv2d(self.in_channel, self.first_out_channel * self.channel_ratio, kernel_size=3, stride=self.stem_stride, padding=1, bias=False), nn.BatchNorm2d(self.first_out_channel * self.channel_ratio, affine=True))
        if args.celltype == 'forward':
            self.cell1 = Neuronal_Cell(args, self.first_out_channel * self.channel_ratio, self.first_out_channel * self.channel_ratio, self.con_mat)
        elif args.celltype == 'backward':
            self.cell1 = Neuronal_Cell_backward(args, self.first_out_channel * self.channel_ratio, self.first_out_channel * self.channel_ratio, self.con_mat)
        else:
            None
            exit()
        self.downconv1 = nn.Sequential(nn.BatchNorm2d(128 * self.channel_ratio, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), neuron.LIFNode(v_threshold=args.threshold, v_reset=0.0, tau=args.tau, surrogate_function=surrogate.ATan(), detach_reset=True), nn.Conv2d(128 * self.channel_ratio, 256 * self.channel_ratio, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False), nn.BatchNorm2d(256 * self.channel_ratio, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
        self.resdownsample1 = nn.AvgPool2d(2, 2)
        if args.celltype == 'forward':
            self.cell2 = Neuronal_Cell(args, 256 * self.channel_ratio, 256 * self.channel_ratio, self.con_mat)
        elif args.celltype == 'backward':
            self.cell2 = Neuronal_Cell_backward(args, 256 * self.channel_ratio, 256 * self.channel_ratio, self.con_mat)
        else:
            None
            exit()
        self.last_act = nn.Sequential(nn.BatchNorm2d(256 * self.channel_ratio, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), neuron.LIFNode(v_threshold=args.threshold, v_reset=0.0, tau=args.tau, surrogate_function=surrogate.ATan(), detach_reset=True))
        self.resdownsample2 = nn.AvgPool2d(self.second_avgpooling, self.second_avgpooling)
        self.classifier = nn.Sequential(layer.Dropout(0.5), nn.Linear(256 * self.channel_ratio * (self.img_size // self.spatial_decay) * (self.img_size // self.spatial_decay), self.classifier_inter_ch, bias=False), neuron.LIFNode(v_threshold=args.threshold, v_reset=0.0, tau=args.tau, surrogate_function=surrogate.ATan(), detach_reset=True), nn.Linear(self.classifier_inter_ch, self.num_final_neuron, bias=True))
        self.boost = nn.AvgPool1d(self.num_cluster, self.num_cluster)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=2)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, input):
        self.neuron_init()
        acc_voltage = 0
        batch_size = input.size(0)
        static_x = self.stem(input)
        for t in range(self.total_timestep):
            x = self.cell1(static_x)
            x = self.downconv1(x)
            x = self.resdownsample1(x)
            x = self.cell2(x)
            x = self.last_act(x)
            x = self.resdownsample2(x)
            x = x.view(batch_size, -1)
            x = self.classifier(x)
            acc_voltage = acc_voltage + self.boost(x.unsqueeze(1)).squeeze(1)
        acc_voltage = acc_voltage / self.total_timestep
        return acc_voltage

    def neuron_init(self):
        self.cell1.last_xin = 0.0
        self.cell1.last_x1 = 0.0
        self.cell1.last_x2 = 0.0
        self.cell2.last_xin = 0.0
        self.cell2.last_x1 = 0.0
        self.cell2.last_x2 = 0.0


class Identity(nn.Module):

    def __init__(self, C_in, C_out, signal):
        super(Identity, self).__init__()
        self.signal = signal

    def forward(self, x):
        return x


class SurrogateFunctionBase(nn.Module):

    def __init__(self, alpha, spiking=True):
        super().__init__()
        self.spiking = spiking
        self.alpha = alpha

    def set_spiking_mode(self, spiking: 'bool'):
        self.spiking = spiking

    def extra_repr(self):
        return f'alpha={self.alpha}, spiking={self.spiking}'

    @staticmethod
    def spiking_function(x, alpha):
        raise NotImplementedError

    @staticmethod
    def primitive_function(x, alpha):
        raise NotImplementedError

    def cuda_code(self, x: 'str', y: 'str', dtype='fp32'):
        raise NotImplementedError

    def cuda_code_start_comments(self):
        return f'// start: spikingjelly.activation_based.surrogate.{self._get_name()}.cuda_code'

    def cuda_code_end_comments(self):
        return f'// end: spikingjelly.activation_based.surrogate.{self._get_name()}.cuda_code'

    def forward(self, x: 'torch.Tensor'):
        if self.spiking:
            return self.spiking_function(x, self.alpha)
        else:
            return self.primitive_function(x, self.alpha)

    def cuda_codes(self, y: 'str', x: 'str', dtype: 'str'):
        raise NotImplementedError


@script
def dSpike_backward(grad_output: 'Tensor', x: 'Tensor', alpha: 'float'):
    mask = x.abs() > 0.5
    const = alpha / (2.0 * tanh(alpha / 2.0))
    grad_x = (grad_output * const / (alpha * x).cosh_().square_()).masked_fill_(mask, 0)
    return grad_x, None


@torch.jit.script
def heaviside(x: 'torch.Tensor'):
    """
    * :ref:`API in English <heaviside.__init__-en>`
    .. _heaviside.__init__-cn:

    :param x: 输入tensor
    :return: 输出tensor

    heaviside阶跃函数，定义为

    .. math::
        g(x) =
        \\begin{cases}
        1, & x \\geq 0 \\\\
        0, & x < 0 \\\\
        \\end{cases}

    阅读 `HeavisideStepFunction <https://mathworld.wolfram.com/HeavisideStepFunction.html>`_ 以获得更多信息。

    * :ref:`中文API <heaviside.__init__-cn>`
    .. _heaviside.__init__-en:

    :param x: the input tensor
    :return: the output tensor

    The heaviside function, which is defined by

    .. math::
        g(x) =
        \\begin{cases}
        1, & x \\geq 0 \\\\
        0, & x < 0 \\\\
        \\end{cases}

    For more information, see `HeavisideStepFunction <https://mathworld.wolfram.com/HeavisideStepFunction.html>`_.

    """
    return x >= 0


class dSpike(Function):

    @staticmethod
    def forward(ctx, x: 'Tensor', alpha: 'float') ->Tensor:
        if x.requires_grad:
            ctx.save_for_backward(x)
            ctx.alpha = alpha
        return heaviside(x)

    @staticmethod
    def backward(ctx, grad_output):
        return dSpike_backward(grad_output, ctx.saved_tensors[0], ctx.alpha)


class DSpike(SurrogateFunctionBase):

    def __init__(self, alpha: 'float'=3, spiking=True):
        super().__init__(alpha, spiking)
        assert alpha > 0, 'alpha must be lager than 0.'

    @staticmethod
    def spiking_function(x: 'Tensor', alpha: 'float'):
        return dSpike.apply(x, alpha)


def getSpikingNode(v_threshold=0.5):
    return LIFNode(tau=1.25, decay_input=False, v_threshold=v_threshold, detach_reset=True, surrogate_function=DSpike())


class SpikingConv2d(nn.Module):

    def __init__(self, input_c, output_c, kernel_size=3, stride=1, padding=1, b=3, spiking=True, v_threshold=0.5):
        super(SpikingConv2d, self).__init__()
        self.conv = layer.Conv2d(input_c, output_c, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = layer.BatchNorm2d(output_c)
        self.spiking = spiking
        if self.spiking:
            self.spike = getSpikingNode(v_threshold=v_threshold)

    def forward(self, x):
        x = self.bn(self.conv(x))
        if self.spiking:
            x = self.spike(x)
        return x


def get_save_v_SpikingNode(v_threshold=0.5):
    return save_v_LIFNode(tau=1.25, decay_input=False, v_threshold=v_threshold, detach_reset=True, surrogate_function=DSpike())


class SearchSpikingConv2d_stem(nn.Module):

    def __init__(self, input_c, output_c, kernel_size=3, stride=1, padding=1, b=3, spiking=True, v_threshold=0.5):
        super(SearchSpikingConv2d_stem, self).__init__()
        self.conv_m = layer.Conv2d(input_c, output_c, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv_b = layer.Conv2d(input_c, output_c, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv_s = layer.Conv2d(input_c, output_c, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn_m = layer.BatchNorm2d(output_c)
        self.bn_b = layer.BatchNorm2d(output_c)
        self.bn_s = layer.BatchNorm2d(output_c)
        self.spike_m = get_save_v_SpikingNode()
        self.spike_b = get_save_v_SpikingNode()
        self.spike_s = get_save_v_SpikingNode()
        self.is_DGS = False
        self.dgs_alpha = nn.Parameter(0.001 * torch.ones(3), requires_grad=True)
        self.dgs_step = 0.2

    def dgs_init_stage(self):
        self.is_DGS = True
        self.conv_s.load_state_dict(self.conv_m.state_dict())
        self.conv_b.load_state_dict(self.conv_m.state_dict())
        self.bn_s.load_state_dict(self.bn_m.state_dict())
        self.bn_b.load_state_dict(self.bn_m.state_dict())
        self.spike_s.surrogate_function.alpha = self.spike_m.surrogate_function.alpha - self.dgs_step
        self.spike_b.surrogate_function.alpha = self.spike_m.surrogate_function.alpha + self.dgs_step
        self.dgs_alpha = nn.Parameter(0.001 * torch.ones(3), requires_grad=True)
        for name, value in self.named_parameters():
            value.requires_grad_(True)

    def dgs_finish_stage(self, dgs_direction):
        self.is_DGS = False
        value_list = [-self.dgs_step, 0, self.dgs_step]
        value = value_list[dgs_direction]
        self.spike_m.surrogate_function.alpha += value
        if self.spike_m.surrogate_function.alpha < 0.2:
            self.spike_m.surrogate_function.alpha = 0.2

    def forward(self, x):
        if self.is_DGS:
            n_a = F.softmax(self.dgs_alpha, dim=0)
            x = n_a[0] * self.spike_s(self.bn_s(self.conv_s(x))) + n_a[1] * self.spike_m(self.bn_m(self.conv_m(x))) + n_a[2] * self.spike_b(self.bn_b(self.conv_b(x)))
        else:
            x = self.spike_m(self.bn_m(self.conv_m(x)))
        return x


class SearchSpikingConv2d_cell(nn.Module):

    def __init__(self, io_c, kernel_size=3, stride=1, padding=1, b=3, spiking=True, v_threshold=0.5):
        super(SearchSpikingConv2d_cell, self).__init__()
        [input_c1, output_c1, primitive1], [input_c2, output_c2, primitive2] = io_c
        self.conv1_m = layer.Conv2d(input_c1, output_c1, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv1_b = layer.Conv2d(input_c1, output_c1, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv1_s = layer.Conv2d(input_c1, output_c1, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn1_m = layer.BatchNorm2d(output_c1)
        self.bn1_b = layer.BatchNorm2d(output_c1)
        self.bn1_s = layer.BatchNorm2d(output_c1)
        self.conv2_m = layer.Conv2d(input_c2, output_c2, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv2_b = layer.Conv2d(input_c2, output_c2, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv2_s = layer.Conv2d(input_c2, output_c2, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn2_m = layer.BatchNorm2d(output_c2)
        self.bn2_b = layer.BatchNorm2d(output_c2)
        self.bn2_s = layer.BatchNorm2d(output_c2)
        self.spike_m = get_save_v_SpikingNode()
        self.spike_b = get_save_v_SpikingNode()
        self.spike_s = get_save_v_SpikingNode()
        self.is_DGS = False
        self.dgs_alpha = nn.Parameter(0.001 * torch.ones(3), requires_grad=True)
        self.dgs_step = 0.2

    def dgs_init_stage(self):
        self.is_DGS = True
        self.conv1_s.load_state_dict(self.conv1_m.state_dict())
        self.conv1_b.load_state_dict(self.conv1_m.state_dict())
        self.bn1_s.load_state_dict(self.bn1_m.state_dict())
        self.bn1_b.load_state_dict(self.bn1_m.state_dict())
        self.conv2_s.load_state_dict(self.conv2_m.state_dict())
        self.conv2_b.load_state_dict(self.conv2_m.state_dict())
        self.bn2_s.load_state_dict(self.bn2_m.state_dict())
        self.bn2_b.load_state_dict(self.bn2_m.state_dict())
        self.spike_s.surrogate_function.alpha = self.spike_m.surrogate_function.alpha - self.dgs_step
        self.spike_b.surrogate_function.alpha = self.spike_m.surrogate_function.alpha + self.dgs_step
        self.dgs_alpha = nn.Parameter(0.001 * torch.ones(3), requires_grad=True)
        for name, value in self.named_parameters():
            value.requires_grad_(True)

    def dgs_finish_stage(self, dgs_direction):
        self.is_DGS = False
        value_list = [-self.dgs_step, 0, self.dgs_step]
        value = value_list[dgs_direction]
        self.spike_m.surrogate_function.alpha += value
        if self.spike_m.surrogate_function.alpha < 0.2:
            self.spike_m.surrogate_function.alpha = 0.2

    def forward(self, x1, x2):
        if self.is_DGS:
            n_a = F.softmax(self.dgs_alpha, dim=0)
            x = n_a[0] * self.spike_s(self.bn1_s(self.conv1_s(x1)) + self.bn2_s(self.conv2_s(x2))) + n_a[1] * self.spike_m(self.bn1_m(self.conv1_m(x1)) + self.bn2_m(self.conv2_m(x2))) + n_a[2] * self.spike_b(self.bn1_b(self.conv1_b(x1)) + self.bn2_b(self.conv2_b(x2)))
        else:
            x = self.spike_m(self.bn1_m(self.conv1_m(x1)) + self.bn2_m(self.conv2_m(x2)))
        return x


class SpikingLinear(nn.Module):

    def __init__(self, input_c, output_c, spiking=True):
        super(SpikingLinear, self).__init__()
        self.linear = layer.Linear(input_c, output_c)
        self.bn = layer.SeqToANNContainer(nn.BatchNorm1d(output_c))
        self.spiking = spiking
        if self.spiking:
            self.spike = getSpikingNode()

    def forward(self, x):
        x = self.bn(self.linear(x))
        if self.spiking:
            x = self.spike(x)
        return x


class SpikingAvgPool2d(nn.Module):

    def __init__(self, kernel_size=5, stride=3, padding=0, b=3, spiking=True):
        super(SpikingAvgPool2d, self).__init__()
        self.pooling = layer.SeqToANNContainer(nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=padding, count_include_pad=False))
        self.spike = getSpikingNode()

    def forward(self, x):
        return self.spike(self.pooling(x))


class SpikingAdaptiveAvgPool2d(nn.Module):

    def __init__(self, dimension, b=3, spiking=True):
        super(SpikingAdaptiveAvgPool2d, self).__init__()
        self.pooling = layer.SeqToANNContainer(nn.AdaptiveAvgPool2d(dimension))
        self.spike = getSpikingNode()

    def forward(self, x):
        return self.spike(self.pooling(x))


class Nearest(nn.Module):

    def __init__(self, shape) ->None:
        super().__init__()
        self._shape = shape

    def forward(self, x):
        return interpolate(x, self._shape, mode='nearest')


PRIMITIVES = ['skip_connect', 'snn_b3', 'snn_b5']


class Cell(nn.Module):

    def __init__(self, steps, block_multiplier, prev_prev_fmultiplier, prev_filter_multiplier, cell_arch, network_arch, filter_multiplier, downup_sample, args=None):
        """
        :param steps: number of nodes
        :type steps: int
        :param block_multiplier: The change factor for the channel for current node
        :type block_multiplier: int
        :param prev_prev_fmultiplier: The change factor for the channel for previous previous node
        :type prev_prev_fmultiplier: int
        :param prev_filter_multiplier: The change factor for the channel for previous node
        :type prev_filter_multiplier: int
        :param cell_arch: cell level architecture
        :type cell_arch: numpy.ndarray
        :param network_arch: layer level architecture
        :type network_arch: numpy.ndarray
        :param filter_multiplier: filter channel multiplier
        :type filter_multiplier: int
        :param downup_sample: sample rate, -1:downsample, 1:upsample, 0: no change
        :type downup_sample: int
        :param args: additional arguments

        A cell is defined as a repeated and searchable unit, which is a directed acyclic graph with N nodes.
        """
        super(Cell, self).__init__()
        self.cell_arch = cell_arch
        self.C_in = block_multiplier * filter_multiplier
        self.C_out = filter_multiplier
        self.C_prev = int(block_multiplier * prev_filter_multiplier)
        self.C_prev_prev = int(block_multiplier * prev_prev_fmultiplier)
        self.downup_sample = downup_sample
        self._steps = steps
        self.block_multiplier = block_multiplier
        self._ops = nn.ModuleList()
        if downup_sample == -1:
            self.scale = 0.5
        elif downup_sample == 1:
            self.scale = 2
        self.cell_arch = torch.sort(self.cell_arch, dim=0)[0]
        C_out = self.C_out
        ops_channel = []
        for i, x in enumerate(self.cell_arch):
            primitive = PRIMITIVES[x[1]]
            if x[0] in [0, 2, 5]:
                C_in = self.C_prev_prev
            elif x[0] in [1, 3, 6]:
                C_in = self.C_prev
            else:
                C_in = self.C_out
            ops_channel.append([C_in, C_out, primitive])
            if i % 2 == 1:
                op = SearchSpikingConv2d_cell(io_c=ops_channel)
                self._ops.append(op)
                ops_channel = []
        self.spikes = nn.ModuleList([getSpikingNode() for _ in range(self._steps)])

    def scale_dimension(self, dim, scale):
        return int((float(dim) - 1.0) * scale + 1.0) if dim % 2 == 1 else int(float(dim) * scale)

    def forward(self, prev_prev_input, prev_input):
        s0 = prev_prev_input
        s1 = prev_input
        if self.downup_sample != 0:
            feature_size_h = self.scale_dimension(s1.shape[3], self.scale)
            feature_size_w = self.scale_dimension(s1.shape[4], self.scale)
            interpolate = layer.SeqToANNContainer(Nearest([feature_size_h, feature_size_w]))
            s1 = interpolate(s1)
        if s0.shape[3] != s1.shape[3] or s0.shape[4] != s1.shape[4]:
            interpolate = layer.SeqToANNContainer(Nearest([s1.shape[3], s1.shape[4]]))
            s0 = interpolate(s0)
        device = prev_input.device
        states = [s0, s1]
        spike = self._ops[0](states[0], states[1])
        states.append(spike)
        spike = self._ops[1](states[0], states[1])
        states.append(spike)
        spike = self._ops[2](states[2], states[3])
        states.append(spike)
        concat_feature = torch.cat(states[-self.block_multiplier:], dim=2)
        return prev_input, concat_feature


class AuxiliaryHeadCIFAR(nn.Module):

    def __init__(self, C, num_classes):
        """assuming input size 8x8"""
        super(AuxiliaryHeadCIFAR, self).__init__()
        self.pooling = SpikingAvgPool2d(kernel_size=5, stride=3, padding=0)
        self.conv1 = SpikingConv2d(C, 128, 1, padding=0, b=3)
        self.conv2 = SpikingConv2d(128, 768, 2, padding=0, b=3)
        self.classifier = SpikingLinear(768, num_classes, spiking=False)

    def forward(self, x):
        x = self.pooling(x)
        spike1 = self.conv1(x)
        spike2 = self.conv2(spike1)
        shape = spike2.shape[:2]
        result = self.classifier(spike2.view(*shape, -1))
        return result


class newFeature(nn.Module):

    def __init__(self, frame_rate, network_arch, cell_arch, cell=Cell, args=None):
        """
        :param frame_rate: input channel
        :type frame_rate: int
        :param network_arch: layer level architecture
        :type network_arch: numpy.ndarray
        :param cell_arch: cell level architecture
        :type cell_arch: numpy.ndarray
        :param cell: choice the type of cell, defaults to Cell
        :type cell: Cell class
        :param args: additional arguments

        newFeature is used to extract feature.
        """
        super(newFeature, self).__init__()
        self.args = args
        self.cells = nn.ModuleList()
        self.network_arch = torch.from_numpy(network_arch)
        self.cell_arch = torch.from_numpy(cell_arch)
        self._step = args.fea_step
        self._num_layers = args.fea_num_layers
        self._block_multiplier = args.fea_block_multiplier
        self._filter_multiplier = args.fea_filter_multiplier
        f_initial = int(self._filter_multiplier)
        half_f_initial = int(f_initial / 2)
        self._num_end = self._filter_multiplier * self._block_multiplier
        self.stem0 = SearchSpikingConv2d_stem(frame_rate, f_initial * self._block_multiplier, kernel_size=3, stride=1, padding=1, b=3)
        self.auxiliary_head = AuxiliaryHeadCIFAR(576, 100)
        filter_param_dict = {(0): 1, (1): 2, (2): 4, (3): 8}
        for i in range(self._num_layers):
            level_option = torch.sum(self.network_arch[i], dim=1)
            prev_level_option = torch.sum(self.network_arch[i - 1], dim=1)
            prev_prev_level_option = torch.sum(self.network_arch[i - 2], dim=1)
            level = torch.argmax(level_option).item()
            prev_level = torch.argmax(prev_level_option).item()
            prev_prev_level = torch.argmax(prev_prev_level_option).item()
            if i == 0:
                downup_sample = -torch.argmax(torch.sum(self.network_arch[0], dim=1))
                _cell = cell(self._step, self._block_multiplier, self._filter_multiplier, self._filter_multiplier, self.cell_arch, self.network_arch[i], int(self._filter_multiplier * filter_param_dict[level]), downup_sample, self.args)
            else:
                three_branch_options = torch.sum(self.network_arch[i], dim=0)
                downup_sample = torch.argmax(three_branch_options).item() - 1
                if i == 1:
                    _cell = cell(self._step, self._block_multiplier, self._filter_multiplier, int(self._filter_multiplier * filter_param_dict[prev_level]), self.cell_arch, self.network_arch[i], int(self._filter_multiplier * filter_param_dict[level]), downup_sample, self.args)
                else:
                    _cell = cell(self._step, self._block_multiplier, int(self._filter_multiplier * filter_param_dict[prev_prev_level]), int(self._filter_multiplier * filter_param_dict[prev_level]), self.cell_arch, self.network_arch[i], int(self._filter_multiplier * filter_param_dict[level]), downup_sample, self.args)
            self.cells += [_cell]

    def forward(self, x):
        stem0 = self.stem0(x)
        stem1 = stem0
        out = stem0, stem1
        for i in range(self._num_layers):
            out = self.cells[i](out[0], out[1])
            """
            cell torch.Size([50, 144, 32, 32])
            cell torch.Size([50, 144, 32, 32])
            cell torch.Size([50, 288, 16, 16])
            cell torch.Size([50, 288, 16, 16])
            cell torch.Size([50, 288, 16, 16])
            cell torch.Size([50, 576, 8, 8] -> auxiliary [50, 10]
            cell torch.Size([50, 576, 8, 8])
            cell torch.Size([50, 576, 8, 8])
            """
            if i == 2 * 8 // 3:
                if self.training:
                    logits_aux = self.auxiliary_head(out[-1])
        last_output = out[-1]
        if self.training:
            return last_output, logits_aux
        else:
            return last_output, None

    def get_params(self):
        bn_params = []
        non_bn_params = []
        for name, param in self.named_parameters():
            if 'bn' in name or 'downsample.1' in name:
                bn_params.append(param)
            else:
                bn_params.append(param)
        return bn_params, non_bn_params


def network_layer_to_space(net_arch):
    """
    :param net_arch: network level sample rate
        0: down 1: None 2: Up
    :type net_arch: numpy.ndarray
    :return: network level architecture
        network_space[layer][level][sample]:
        layer: 0 - 8
        level: sample_level {0: 1, 1: 2, 2: 4, 3: 8}
        sample: 0: down 1: None 2: Up
    :rtype: numpy.ndarray

    Convert network level sample rate like [0,0,1,1,1,2,2,2] to network architecture.
    """
    for i, layer in enumerate(net_arch):
        if i == 0:
            space = np.zeros((1, 4, 3))
            space[0][layer][0] = 1
            prev = layer
        else:
            if layer == prev + 1:
                sample = 0
            elif layer == prev:
                sample = 1
            elif layer == prev - 1:
                sample = 2
            space1 = np.zeros((1, 4, 3))
            space1[0][layer][sample] = 1
            space = np.concatenate([space, space1], axis=0)
            prev = layer
    return space


class SpikeDHS(nn.Module):

    def __init__(self, init_channels=3, args=None):
        """
        :param init_channels: channel size, defaults to 3
        :type init_channels: int
        :param args: additional arguments

        The SpikeDHS `Auto-Spikformer: Spikformer Architecture Search <https://arxiv.org/abs/2306.00807>`_ implementation by Spikingjelly.

        """
        super(SpikeDHS, self).__init__()
        p = 0.0
        network_path_fea = [0, 0, 1, 1, 1, 2, 2, 2]
        network_path_fea = np.array(network_path_fea)
        network_arch_fea = network_layer_to_space(network_path_fea)
        cell_arch_fea = [[1, 1], [0, 1], [3, 2], [2, 1], [7, 1], [8, 1]]
        cell_arch_fea = np.array(cell_arch_fea)
        self.feature = newFeature(init_channels, network_arch_fea, cell_arch_fea, args=args)
        self.global_pooling = SpikingAdaptiveAvgPool2d(1)
        self.classifier = SpikingLinear(576, 100, spiking=False)
        self._time_step = 6

    def weight_parameters(self):
        return [param for name, param in self.named_parameters()]

    def forward(self, input):
        input = input.expand(self._time_step, -1, -1, -1, -1)
        shape = input.shape[:2]
        feature_out, logits_aux = self.feature(input)
        pooling_out = self.global_pooling(feature_out)
        shape = pooling_out.shape[:2]
        logits_buf = self.classifier(pooling_out.view(*shape, -1))
        logits = logits_buf.mean(0)
        if logits_aux is not None:
            logits_aux = logits_aux.mean(0)
        if self.training:
            return logits, logits_aux
        else:
            return logits, None

    def dgs_freeze_weights(self):
        for name, value in self.named_parameters():
            value.requires_grad_(False)

    def dgs_unfreeze_weights(self):
        for name, value in self.named_parameters():
            value.requires_grad_(True)


class SpikingResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False, groups=1, width_per_group=64, replace_stride_with_dilation=None, norm_layer=None, spiking_neuron: 'callable'=None, **kwargs):
        super(SpikingResNet, self).__init__()
        if norm_layer is None:
            norm_layer = layer.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError('replace_stride_with_dilation should be None or a 3-element tuple, got {}'.format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = layer.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.sn1 = spiking_neuron(**deepcopy(kwargs))
        self.maxpool = layer.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], spiking_neuron=spiking_neuron, **kwargs)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0], spiking_neuron=spiking_neuron, **kwargs)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1], spiking_neuron=spiking_neuron, **kwargs)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2], spiking_neuron=spiking_neuron, **kwargs)
        self.avgpool = layer.AdaptiveAvgPool2d((1, 1))
        self.fc = layer.Linear(512 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, layer.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (layer.BatchNorm2d, layer.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False, spiking_neuron: 'callable'=None, **kwargs):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(conv1x1(self.inplanes, planes * block.expansion, stride), norm_layer(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer, spiking_neuron, **kwargs))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups, base_width=self.base_width, dilation=self.dilation, norm_layer=norm_layer, spiking_neuron=spiking_neuron, **kwargs))
        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.sn1(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        if self.avgpool.step_mode == 's':
            x = torch.flatten(x, 1)
        elif self.avgpool.step_mode == 'm':
            x = torch.flatten(x, 2)
        x = self.fc(x)
        return x

    def forward(self, x):
        return self._forward_impl(x)


class SpikingVGG(nn.Module):

    def __init__(self, cfg, batch_norm=False, norm_layer=None, num_classes=1000, init_weights=True, spiking_neuron: 'callable'=None, **kwargs):
        super(SpikingVGG, self).__init__()
        self.features = self.make_layers(cfg=cfg, batch_norm=batch_norm, norm_layer=norm_layer, neuron=spiking_neuron, **kwargs)
        self.avgpool = layer.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(layer.Linear(512 * 7 * 7, 4096), spiking_neuron(**deepcopy(kwargs)), layer.Dropout(), layer.Linear(4096, 4096), spiking_neuron(**deepcopy(kwargs)), layer.Dropout(), layer.Linear(4096, num_classes))
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        if self.avgpool.step_mode == 's':
            x = torch.flatten(x, 1)
        elif self.avgpool.step_mode == 'm':
            x = torch.flatten(x, 2)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, layer.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, layer.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, layer.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    @staticmethod
    def make_layers(cfg, batch_norm=False, norm_layer=None, neuron: 'callable'=None, **kwargs):
        if norm_layer is None:
            norm_layer = layer.BatchNorm2d
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [layer.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = layer.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, norm_layer(v), neuron(**deepcopy(kwargs))]
                else:
                    layers += [conv2d, neuron(**deepcopy(kwargs))]
                in_channels = v
        return nn.Sequential(*layers)


class Scale(nn.Module):

    def __init__(self, scale):
        super(Scale, self).__init__()
        self.scale = scale

    def forward(self, x):
        return x * self.scale


class OTTTSpikingVGG(nn.Module):

    def __init__(self, cfg, weight_standardization=True, num_classes=1000, init_weights=True, spiking_neuron: 'callable'=None, light_classifier=True, drop_rate=0.0, **kwargs):
        super(OTTTSpikingVGG, self).__init__()
        self.fc_hw = kwargs.get('fc_hw', 1)
        if weight_standardization:
            ws_scale = 2.74
        else:
            ws_scale = 1.0
        self.neuron = spiking_neuron
        self.features = self.make_layers(cfg=cfg, weight_standardization=weight_standardization, neuron=spiking_neuron, drop_rate=0.0, **kwargs)
        if light_classifier:
            self.classifier = layer.OTTTSequential(layer.AdaptiveAvgPool2d((self.fc_hw, self.fc_hw)), layer.Flatten(1), layer.Linear(512 * self.fc_hw ** 2, num_classes))
        else:
            Linear = layer.WSLinear if weight_standardization else layer.Linear
            self.classifier = layer.OTTTSequential(layer.AdaptiveAvgPool2d((7, 7)), layer.Flatten(1), Linear(512 * 7 * 7, 4096), spiking_neuron(**deepcopy(kwargs)), Scale(ws_scale), layer.Dropout(), Linear(4096, 4096), spiking_neuron(**deepcopy(kwargs)), Scale(ws_scale), layer.Dropout(), layer.Linear(4096, num_classes))
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, layer.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, layer.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, layer.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    @staticmethod
    def make_layers(cfg, weight_standardization=True, neuron: 'callable'=None, drop_rate=0.0, **kwargs):
        layers = []
        in_channels = 3
        Conv2d = layer.WSConv2d if weight_standardization else layer.Conv2d
        for v in cfg:
            if v == 'M':
                layers += [layer.MaxPool2d(kernel_size=2, stride=2)]
            elif v == 'A':
                layers += [layer.AvgPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = Conv2d(in_channels, v, kernel_size=3, padding=1)
                layers += [conv2d, neuron(**deepcopy(kwargs))]
                if weight_standardization:
                    layers += [Scale(2.74)]
                in_channels = v
                if drop_rate > 0.0:
                    layers += [layer.Dropout(drop_rate)]
        return layer.OTTTSequential(*layers)


class RandomMixup(torch.nn.Module):
    """Randomly apply Mixup to the provided batch and targets.
    The class implements the data augmentations as described in the paper
    `"mixup: Beyond Empirical Risk Minimization" <https://arxiv.org/abs/1710.09412>`_.

    Args:
        num_classes (int): number of classes used for one-hot encoding.
        p (float): probability of the batch being transformed. Default value is 0.5.
        alpha (float): hyperparameter of the Beta distribution used for mixup.
            Default value is 1.0.
        inplace (bool): boolean to make this transform inplace. Default set to False.
    """

    def __init__(self, num_classes: 'int', p: 'float'=0.5, alpha: 'float'=1.0, inplace: 'bool'=False) ->None:
        super().__init__()
        assert num_classes > 0, 'Please provide a valid positive value for the num_classes.'
        assert alpha > 0, "Alpha param can't be zero."
        self.num_classes = num_classes
        self.p = p
        self.alpha = alpha
        self.inplace = inplace

    def forward(self, batch: 'Tensor', target: 'Tensor') ->Tuple[Tensor, Tensor]:
        """
        Args:
            batch (Tensor): Float tensor of size (B, C, H, W)
            target (Tensor): Integer tensor of size (B, )

        Returns:
            Tensor: Randomly transformed batch.
        """
        if batch.ndim != 4:
            raise ValueError(f'Batch ndim should be 4. Got {batch.ndim}')
        if target.ndim != 1:
            raise ValueError(f'Target ndim should be 1. Got {target.ndim}')
        if not batch.is_floating_point():
            raise TypeError(f'Batch dtype should be a float tensor. Got {batch.dtype}.')
        if target.dtype != torch.int64:
            raise TypeError(f'Target dtype should be torch.int64. Got {target.dtype}')
        if not self.inplace:
            batch = batch.clone()
            target = target.clone()
        if target.ndim == 1:
            target = torch.nn.functional.one_hot(target, num_classes=self.num_classes)
        if torch.rand(1).item() >= self.p:
            return batch, target
        batch_rolled = batch.roll(1, 0)
        target_rolled = target.roll(1, 0)
        lambda_param = float(torch._sample_dirichlet(torch.tensor([self.alpha, self.alpha]))[0])
        batch_rolled.mul_(1.0 - lambda_param)
        batch.mul_(lambda_param).add_(batch_rolled)
        target_rolled.mul_(1.0 - lambda_param)
        target.mul_(lambda_param).add_(target_rolled)
        return batch, target

    def __repr__(self) ->str:
        s = f'{self.__class__.__name__}(num_classes={self.num_classes}, p={self.p}, alpha={self.alpha}, inplace={self.inplace})'
        return s


class RandomCutmix(torch.nn.Module):
    """Randomly apply Cutmix to the provided batch and targets.
    The class implements the data augmentations as described in the paper
    `"CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features"
    <https://arxiv.org/abs/1905.04899>`_.

    Args:
        num_classes (int): number of classes used for one-hot encoding.
        p (float): probability of the batch being transformed. Default value is 0.5.
        alpha (float): hyperparameter of the Beta distribution used for cutmix.
            Default value is 1.0.
        inplace (bool): boolean to make this transform inplace. Default set to False.
    """

    def __init__(self, num_classes: 'int', p: 'float'=0.5, alpha: 'float'=1.0, inplace: 'bool'=False) ->None:
        super().__init__()
        assert num_classes > 0, 'Please provide a valid positive value for the num_classes.'
        assert alpha > 0, "Alpha param can't be zero."
        self.num_classes = num_classes
        self.p = p
        self.alpha = alpha
        self.inplace = inplace

    def forward(self, batch: 'Tensor', target: 'Tensor') ->Tuple[Tensor, Tensor]:
        """
        Args:
            batch (Tensor): Float tensor of size (B, C, H, W)
            target (Tensor): Integer tensor of size (B, )

        Returns:
            Tensor: Randomly transformed batch.
        """
        if batch.ndim != 4:
            raise ValueError(f'Batch ndim should be 4. Got {batch.ndim}')
        if target.ndim != 1:
            raise ValueError(f'Target ndim should be 1. Got {target.ndim}')
        if not batch.is_floating_point():
            raise TypeError(f'Batch dtype should be a float tensor. Got {batch.dtype}.')
        if target.dtype != torch.int64:
            raise TypeError(f'Target dtype should be torch.int64. Got {target.dtype}')
        if not self.inplace:
            batch = batch.clone()
            target = target.clone()
        if target.ndim == 1:
            target = torch.nn.functional.one_hot(target, num_classes=self.num_classes)
        if torch.rand(1).item() >= self.p:
            return batch, target
        batch_rolled = batch.roll(1, 0)
        target_rolled = target.roll(1, 0)
        lambda_param = float(torch._sample_dirichlet(torch.tensor([self.alpha, self.alpha]))[0])
        W, H = F.get_image_size(batch)
        r_x = torch.randint(W, (1,))
        r_y = torch.randint(H, (1,))
        r = 0.5 * math.sqrt(1.0 - lambda_param)
        r_w_half = int(r * W)
        r_h_half = int(r * H)
        x1 = int(torch.clamp(r_x - r_w_half, min=0))
        y1 = int(torch.clamp(r_y - r_h_half, min=0))
        x2 = int(torch.clamp(r_x + r_w_half, max=W))
        y2 = int(torch.clamp(r_y + r_h_half, max=H))
        batch[:, :, y1:y2, x1:x2] = batch_rolled[:, :, y1:y2, x1:x2]
        lambda_param = float(1.0 - (x2 - x1) * (y2 - y1) / (W * H))
        target_rolled.mul_(1.0 - lambda_param)
        target.mul_(lambda_param).add_(target_rolled)
        return batch, target

    def __repr__(self) ->str:
        s = f'{self.__class__.__name__}(num_classes={self.num_classes}, p={self.p}, alpha={self.alpha}, inplace={self.inplace})'
        return s


class ExponentialMovingAverage(torch.optim.swa_utils.AveragedModel):
    """Maintains moving averages of model parameters using an exponential decay.
    ``ema_avg = decay * avg_model_param + (1 - decay) * model_param``
    `torch.optim.swa_utils.AveragedModel <https://pytorch.org/docs/stable/optim.html#custom-averaging-strategies>`_
    is used to compute the EMA.
    """

    def __init__(self, model, decay, device='cpu'):

        def ema_avg(avg_model_param, model_param, num_averaged):
            return decay * avg_model_param + (1 - decay) * model_param
        super().__init__(model, device, ema_avg)

    def update_parameters(self, model):
        for p_swa, p_model in zip(self.module.state_dict().values(), model.state_dict().values()):
            device = p_swa.device
            p_model_ = p_model.detach()
            if self.n_averaged == 0:
                p_swa.detach().copy_(p_model_)
            else:
                p_swa.detach().copy_(self.avg_fn(p_swa.detach(), p_model_, self.n_averaged))
        self.n_averaged += 1


def _get_normal_distribution(random_state: 'Optional[Union[int, Generator, RandomState]]'):
    normal_dist = None
    if isinstance(random_state, (integer, int)) or random_state is None:
        random_state = default_rng(random_state)
        normal_dist = random_state.normal
    elif isinstance(random_state, (Generator, RandomState)):
        normal_dist = random_state.normal
    else:
        raise ValueError('random_state must be one of integer, numpy.random.Generator, numpy.random.Randomstate')
    return normal_dist


def powerlaw_psd_gaussian(exponent: 'float', size: 'Union[int, Iterable[int]]', fmin: 'float'=0.0, random_state: 'Optional[Union[int, Generator, RandomState]]'=None):
    """Gaussian (1/f)**beta noise.

    Based on the algorithm in:
    Timmer, J. and Koenig, M.:
    On generating power law noise.
    Astron. Astrophys. 300, 707-710 (1995)

    Normalised to unit variance

    Parameters:
    -----------

    exponent : float
        The power-spectrum of the generated noise is proportional to

        S(f) = (1 / f)**beta
        flicker / pink noise:   exponent beta = 1
        brown noise:            exponent beta = 2

        Furthermore, the autocorrelation decays proportional to lag**-gamma
        with gamma = 1 - beta for 0 < beta < 1.
        There may be finite-size issues for beta close to one.

    size : Union[int, Iterable[int]]
        The output has the given shape, and the desired power spectrum in
        the last coordinate. That is, the last dimension is taken as time,
        and all other components are independent.

    fmin : float, optional
        Low-frequency cutoff.
        Default: 0 corresponds to original paper. 
        
        The power-spectrum below fmin is flat. fmin is defined relative
        to a unit sampling rate (see numpy's rfftfreq). For convenience,
        the passed value is mapped to max(fmin, 1/samples) internally
        since 1/samples is the lowest possible finite frequency in the
        sample. The largest possible value is fmin = 0.5, the Nyquist
        frequency. The output for this value is white noise.

    random_state :  int, numpy.integer, numpy.random.Generator, numpy.random.RandomState, 
                    optional
        Optionally sets the state of NumPy's underlying random number generator.
        Integer-compatible values or None are passed to np.random.default_rng.
        np.random.RandomState or np.random.Generator are used directly.
        Default: None.

    Returns
    -------
    out : array
        The samples.


    Examples:
    ---------

    # generate 1/f noise == pink noise == flicker noise
    >>> import colorednoise as cn
    >>> y = cn.powerlaw_psd_gaussian(1, 5)
    """
    if isinstance(size, (integer, int)):
        size = [size]
    elif isinstance(size, Iterable):
        size = list(size)
    else:
        raise ValueError('Size must be of type int or Iterable[int]')
    samples = size[-1]
    f = rfftfreq(samples)
    if 0 <= fmin <= 0.5:
        fmin = max(fmin, 1.0 / samples)
    else:
        raise ValueError('fmin must be chosen between 0 and 0.5.')
    s_scale = f
    ix = npsum(s_scale < fmin)
    if ix and ix < len(s_scale):
        s_scale[:ix] = s_scale[ix]
    s_scale = s_scale ** (-exponent / 2.0)
    w = s_scale[1:].copy()
    w[-1] *= (1 + samples % 2) / 2.0
    sigma = 2 * sqrt(npsum(w ** 2)) / samples
    size[-1] = len(f)
    dims_to_add = len(size) - 1
    s_scale = s_scale[(newaxis,) * dims_to_add + (Ellipsis,)]
    normal_dist = _get_normal_distribution(random_state)
    sr = normal_dist(scale=s_scale, size=size)
    si = normal_dist(scale=s_scale, size=size)
    if not samples % 2:
        si[..., -1] = 0
        sr[..., -1] *= sqrt(2)
    si[..., 0] = 0
    sr[..., 0] *= sqrt(2)
    s = sr + 1.0j * si
    y = irfft(s, n=samples, axis=-1) / sigma
    return y


class SpikingRNNCellBase(nn.Module):

    def __init__(self, input_size: 'int', hidden_size: 'int', bias=True):
        """
        * :ref:`API in English <SpikingRNNCellBase.__init__-en>`

        .. _SpikingRNNCellBase.__init__-cn:

        Spiking RNN Cell 的基类。

        :param input_size: 输入 ``x`` 的特征数
        :type input_size: int

        :param hidden_size: 隐藏状态 ``h`` 的特征数
        :type hidden_size: int

        :param bias: 若为 ``False``, 则内部的隐藏层不会带有偏置项 ``b_ih`` 和 ``b_hh``。 默认为 ``True``
        :type bias: bool

        .. note::

            所有权重和偏置项都会按照 :math:`\\mathcal{U}(-\\sqrt{k}, \\sqrt{k})` 进行初始化。
            其中 :math:`k = \\frac{1}{\\text{hidden_size}}`.

        * :ref:`中文API <SpikingRNNCellBase.__init__-cn>`

        .. _SpikingRNNCellBase.__init__-en:

        The base class of Spiking RNN Cell.

        :param input_size: The number of expected features in the input ``x``
        :type input_size: int

        :param hidden_size: The number of features in the hidden state ``h``
        :type hidden_size: int

        :param bias: If ``False``, then the layer does not use bias weights ``b_ih`` and
            ``b_hh``. Default: ``True``
        :type bias: bool

        .. admonition:: Note
            :class: note

            All the weights and biases are initialized from :math:`\\mathcal{U}(-\\sqrt{k}, \\sqrt{k})`
            where :math:`k = \\frac{1}{\\text{hidden_size}}`.

        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias

    def reset_parameters(self):
        """
        * :ref:`API in English <SpikingRNNCellBase.reset_parameters-en>`

        .. _SpikingRNNCellBase.reset_parameters-cn:

        初始化所有可学习参数。

        * :ref:`中文API <SpikingRNNCellBase.reset_parameters-cn>`

        .. _SpikingRNNCellBase.reset_parameters-en:

        Initialize all learnable parameters.
        """
        sqrt_k = math.sqrt(1 / self.hidden_size)
        for param in self.parameters():
            nn.init.uniform_(param, -sqrt_k, sqrt_k)

    def weight_ih(self):
        """
        * :ref:`API in English <SpikingRNNCellBase.weight_ih-en>`

        .. _SpikingRNNCellBase.weight_ih-cn:

        :return: 输入到隐藏状态的连接权重
        :rtype: torch.Tensor

        * :ref:`中文API <SpikingRNNCellBase.weight_ih-cn>`

        .. _SpikingRNNCellBase.weight_ih-en:

        :return: the learnable input-hidden weights
        :rtype: torch.Tensor
        """
        return self.linear_ih.weight

    def weight_hh(self):
        """
        * :ref:`API in English <SpikingRNNCellBase.weight_hh-en>`

        .. _SpikingRNNCellBase.weight_hh-cn:

        :return: 隐藏状态到隐藏状态的连接权重
        :rtype: torch.Tensor

        * :ref:`中文API <SpikingRNNCellBase.weight_hh-cn>`

        .. _SpikingRNNCellBase.weight_hh-en:

        :return: the learnable hidden-hidden weights
        :rtype: torch.Tensor
        """
        return self.linear_hh.weight

    def bias_ih(self):
        """
        * :ref:`API in English <SpikingRNNCellBase.bias_ih-en>`

        .. _SpikingRNNCellBase.bias_ih-cn:

        :return: 输入到隐藏状态的连接偏置项
        :rtype: torch.Tensor

        * :ref:`中文API <SpikingRNNCellBase.bias_ih-cn>`

        .. _SpikingRNNCellBase.bias_ih-en:

        :return: the learnable input-hidden bias
        :rtype: torch.Tensor
        """
        return self.linear_ih.bias

    def bias_hh(self):
        """
        * :ref:`API in English <SpikingRNNCellBase.bias_hh-en>`

        .. _SpikingRNNCellBase.bias_hh-cn:

        :return: 隐藏状态到隐藏状态的连接偏置项
        :rtype: torch.Tensor

        * :ref:`中文API <SpikingRNNCellBase.bias_hh-cn>`

        .. _SpikingRNNCellBase.bias_hh-en:

        :return: the learnable hidden-hidden bias
        :rtype: torch.Tensor
        """
        return self.linear_hh.bias


def bidirectional_rnn_cell_forward(cell: 'nn.Module', cell_reverse: 'nn.Module', x: 'torch.Tensor', states: 'torch.Tensor', states_reverse: 'torch.Tensor'):
    """
    :param cell: 正向RNN cell，输入是正向序列
    :type cell: nn.Module
    :param cell_reverse: 反向的RNN cell，输入是反向序列
    :type cell_reverse: nn.Module
    :param x: ``shape = [T, batch_size, input_size]`` 的输入
    :type x: torch.Tensor
    :param states: 正向RNN cell的起始状态
        若RNN cell只有单个隐藏状态，则 ``shape = [batch_size, hidden_size]`` ；
        否则 ``shape = [states_num, batch_size, hidden_size]``
    :type states: torch.Tensor
    :param states_reverse: 反向RNN cell的起始状态
        若RNN cell只有单个隐藏状态，则 ``shape = [batch_size, hidden_size]`` ；
        否则 ``shape = [states_num, batch_size, hidden_size]``
    :type states: torch.Tensor
    :return: y, ss, ss_r

        y: torch.Tensor
            ``shape = [T, batch_size, 2 * hidden_size]`` 的输出。``y[t]`` 由正向cell在 ``t`` 时刻和反向cell在 ``T - t - 1``
            时刻的输出拼接而来
        ss: torch.Tensor
            ``shape`` 与 ``states`` 相同，正向cell在 ``T-1`` 时刻的状态
        ss_r: torch.Tensor
            ``shape`` 与 ``states_reverse`` 相同，反向cell在 ``0`` 时刻的状态

    计算单个正向和反向RNN cell沿着时间维度的循环并输出结果和两个cell的最终状态。
    """
    T = x.shape[0]
    ss = states
    ss_r = states_reverse
    output = []
    output_r = []
    for t in range(T):
        ss = cell(x[t], ss)
        ss_r = cell_reverse(x[T - t - 1], ss_r)
        if states.dim() == 2:
            output.append(ss)
            output_r.append(ss_r)
        elif states.dim() == 3:
            output.append(ss[0])
            output_r.append(ss_r[0])
    ret = []
    for t in range(T):
        ret.append(torch.cat((output[t], output_r[T - t - 1]), dim=-1))
    return torch.stack(ret), ss, ss_r


def directional_rnn_cell_forward(cell: 'nn.Module', x: 'torch.Tensor', states: 'torch.Tensor'):
    T = x.shape[0]
    ss = states
    output = []
    for t in range(T):
        ss = cell(x[t], ss)
        if states.dim() == 2:
            output.append(ss)
        elif states.dim() == 3:
            output.append(ss[0])
    return torch.stack(output), ss


class SpikingRNNBase(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, bias=True, dropout_p=0, invariant_dropout_mask=False, bidirectional=False, *args, **kwargs):
        """
        * :ref:`API in English <SpikingRNNBase.__init__-en>`

        .. _SpikingRNNBase.__init__-cn:

        多层 `脉冲` RNN的基类。

        :param input_size: 输入 ``x`` 的特征数
        :type input_size: int
        :param hidden_size: 隐藏状态 ``h`` 的特征数
        :type hidden_size: int
        :param num_layers: 内部RNN的层数，例如 ``num_layers = 2`` 将会创建堆栈式的两层RNN，第1层接收第0层的输出作为输入，
            并计算最终输出
        :type num_layers: int
        :param bias: 若为 ``False``, 则内部的隐藏层不会带有偏置项 ``b_ih`` 和 ``b_hh``。 默认为 ``True``
        :type bias: bool
        :param dropout_p: 若非 ``0``，则除了最后一层，每个RNN层后会增加一个丢弃概率为 ``dropout_p`` 的 `Dropout` 层。
            默认为 ``0``
        :type dropout_p: float
        :param invariant_dropout_mask: 若为 ``False``，则使用普通的 `Dropout`；若为 ``True``，则使用SNN中特有的，`mask` 不
            随着时间变化的 `Dropout``，参见 :class:`~spikingjelly.activation_based.layer.Dropout`。默认为 ``False``
        :type invariant_dropout_mask: bool
        :param bidirectional: 若为 ``True``，则使用双向RNN。默认为 ``False``
        :type bidirectional: bool
        :param args: 子类使用的额外参数
        :param kwargs: 子类使用的额外参数

        * :ref:`中文API <SpikingRNNBase.__init__-cn>`

        .. _SpikingRNNBase.__init__-en:

        The base-class of a multi-layer `spiking` RNN.

        :param input_size: The number of expected features in the input ``x``
        :type input_size: int
        :param hidden_size: The number of features in the hidden state ``h``
        :type hidden_size: int
        :param num_layers: Number of recurrent layers. E.g., setting ``num_layers=2`` would mean stacking two LSTMs
            together to form a `stacked RNN`, with the second RNN taking in outputs of the first RNN and computing the
            final results
        :type num_layers: int
        :param bias: If ``False``, then the layer does not use bias weights `b_ih` and `b_hh`. Default: ``True``
        :type bias: bool
        :param dropout_p: If non-zero, introduces a `Dropout` layer on the outputs of each RNN layer except the last
            layer, with dropout probability equal to :attr:`dropout`. Default: 0
        :type dropout_p: float
        :param invariant_dropout_mask: If ``False``，use the naive `Dropout`；If ``True``，use the dropout in SNN that
            `mask` doesn't change in different time steps, see :class:`~spikingjelly.activation_based.layer.Dropout` for more
            information. Defaule: ``False``
        :type invariant_dropout_mask: bool
        :param bidirectional: If ``True``, becomes a bidirectional LSTM. Default: ``False``
        :type bidirectional: bool
        :param args: additional arguments for sub-class
        :param kwargs: additional arguments for sub-class
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.dropout_p = dropout_p
        self.invariant_dropout_mask = invariant_dropout_mask
        self.bidirectional = bidirectional
        if self.bidirectional:
            self.cells, self.cells_reverse = self.create_cells(*args, **kwargs)
        else:
            self.cells = self.create_cells(*args, **kwargs)

    def create_cells(self, *args, **kwargs):
        """
        * :ref:`API in English <SpikingRNNBase.create_cells-en>`

        .. _SpikingRNNBase.create_cells-cn:

        :param args: 子类使用的额外参数
        :param kwargs: 子类使用的额外参数
        :return: 若 ``self.bidirectional == True`` 则会返回正反两个堆栈式RNN；否则返回单个堆栈式RNN
        :rtype: nn.Sequential

        * :ref:`中文API <SpikingRNNBase.create_cells-cn>`

        .. _SpikingRNNBase.create_cells-en:

        :param args: additional arguments for sub-class
        :param kwargs: additional arguments for sub-class
        :return: If ``self.bidirectional == True``, return a RNN for forward direction and a RNN for reverse direction;
            else, return a single stacking RNN
        :rtype: nn.Sequential
        """
        if self.bidirectional:
            cells = []
            cells_reverse = []
            cells.append(self.base_cell()(self.input_size, self.hidden_size, self.bias, *args, **kwargs))
            cells_reverse.append(self.base_cell()(self.input_size, self.hidden_size, self.bias, *args, **kwargs))
            for i in range(self.num_layers - 1):
                cells.append(self.base_cell()(self.hidden_size * 2, self.hidden_size, self.bias, *args, **kwargs))
                cells_reverse.append(self.base_cell()(self.hidden_size * 2, self.hidden_size, self.bias, *args, **kwargs))
            return nn.Sequential(*cells), nn.Sequential(*cells_reverse)
        else:
            cells = []
            cells.append(self.base_cell()(self.input_size, self.hidden_size, self.bias, *args, **kwargs))
            for i in range(self.num_layers - 1):
                cells.append(self.base_cell()(self.hidden_size, self.hidden_size, self.bias, *args, **kwargs))
            return nn.Sequential(*cells)

    @staticmethod
    def base_cell():
        """
        * :ref:`API in English <SpikingRNNBase.base_cell-en>`

        .. _SpikingRNNBase.base_cell-cn:

        :return: 构成该RNN的基本RNN Cell。例如对于 :class:`~spikingjelly.activation_based.rnn.SpikingLSTM`，
            返回的是 :class:`~spikingjelly.activation_based.rnn.SpikingLSTMCell`
        :rtype: nn.Module

        * :ref:`中文API <SpikingRNNBase.base_cell-cn>`

        .. _SpikingRNNBase.base_cell-en:

        :return: The base cell of this RNN. E.g., in :class:`~spikingjelly.activation_based.rnn.SpikingLSTM` this function
            will return :class:`~spikingjelly.activation_based.rnn.SpikingLSTMCell`
        :rtype: nn.Module
        """
        raise NotImplementedError

    @staticmethod
    def states_num():
        """
        * :ref:`API in English <SpikingRNNBase.states_num-en>`

        .. _SpikingRNNBase.states_num-cn:

        :return: 状态变量的数量。例如对于 :class:`~spikingjelly.activation_based.rnn.SpikingLSTM`，由于其输出是 ``h`` 和 ``c``，
            因此返回 ``2``；而对于 :class:`~spikingjelly.activation_based.rnn.SpikingGRU`，由于其输出是 ``h``，因此返回 ``1``
        :rtype: int

        * :ref:`中文API <SpikingRNNBase.states_num-cn>`

        .. _SpikingRNNBase.states_num-en:

        :return: The states number. E.g., for :class:`~spikingjelly.activation_based.rnn.SpikingLSTM` the output are ``h``
            and ``c``, this function will return ``2``; for :class:`~spikingjelly.activation_based.rnn.SpikingGRU` the output
            is ``h``, this function will return ``1``
        :rtype: int
        """
        raise NotImplementedError

    def forward(self, x: 'torch.Tensor', states=None):
        """
        * :ref:`API in English <SpikingRNNBase.forward-en>`

        .. _SpikingRNNBase.forward-cn:

        :param x: ``shape = [T, batch_size, input_size]``，输入序列
        :type x: torch.Tensor
        :param states: ``self.states_num()`` 为 ``1`` 时是单个tensor, 否则是一个tuple，包含 ``self.states_num()`` 个tensors。
            所有的tensor的尺寸均为 ``shape = [num_layers * num_directions, batch, hidden_size]``, 包含 ``self.states_num()``
            个初始状态
            如果RNN是双向的, ``num_directions`` 为 ``2``, 否则为 ``1``
        :type states: Union[torch.Tensor, tuple]
        :return: output, output_states
            output: torch.Tensor
                ``shape = [T, batch, num_directions * hidden_size]``，最后一层在所有时刻的输出
            output_states: Union[torch.Tensor, tuple]
                ``self.states_num()`` 为 ``1`` 时是单个tensor, 否则是一个tuple，包含 ``self.states_num()`` 个tensors。
                所有的tensor的尺寸均为 ``shape = [num_layers * num_directions, batch, hidden_size]``, 包含 ``self.states_num()``
                个最后时刻的状态

        * :ref:`中文API <SpikingRNNBase.forward-cn>`

        .. _SpikingRNNBase.forward-en:

        :param x: ``shape = [T, batch_size, input_size]``, tensor containing the features of the input sequence
        :type x: torch.Tensor
        :param states: a single tensor when ``self.states_num()`` is ``1``, otherwise a tuple with ``self.states_num()``
            tensors.
            ``shape = [num_layers * num_directions, batch, hidden_size]`` for all tensors, containing the ``self.states_num()``
            initial states for each element in the batch.
            If the RNN is bidirectional, ``num_directions`` should be ``2``, else it should be ``1``
        :type states: Union[torch.Tensor, tuple]
        :return: output, output_states
            output: torch.Tensor
                ``shape = [T, batch, num_directions * hidden_size]``, tensor containing the output features from the last
                layer of the RNN, for each ``t``
            output_states: Union[torch.Tensor, tuple]
                a single tensor when ``self.states_num()`` is ``1``, otherwise a tuple with ``self.states_num()``
                tensors.
                ``shape = [num_layers * num_directions, batch, hidden_size]`` for all tensors, containing the ``self.states_num()``
                states for ``t = T - 1``
        """
        T = x.shape[0]
        batch_size = x.shape[1]
        if isinstance(states, tuple):
            states_list = torch.stack(states)
        elif isinstance(states, torch.Tensor):
            if states.dim() == 3:
                states_list = states
            else:
                raise TypeError
        elif states == None:
            if self.bidirectional == True:
                states_list = torch.zeros(size=[self.states_num(), self.num_layers * 2, x.shape[1], self.hidden_size], dtype=torch.float, device=x.device).squeeze(0)
            else:
                states_list = torch.zeros(size=[self.states_num(), self.num_layers, x.shape[1], self.hidden_size], dtype=torch.float, device=x.device).squeeze(0)
        else:
            raise TypeError
        if self.bidirectional:
            if states_list.dim() == 4 and states_list.shape[1] != 2 * self.num_layers or states_list.dim() == 3 and states_list.shape[0] != 2 * self.num_layers:
                raise ValueError
            y = x.clone()
            if self.training and self.dropout_p > 0 and self.invariant_dropout_mask:
                mask = F.dropout(torch.ones(size=[self.num_layers - 1, batch_size, self.hidden_size * 2]), p=self.dropout_p, training=True, inplace=True)
            for i in range(self.num_layers):
                new_states_list = torch.zeros_like(states_list.data)
                if self.states_num() == 1:
                    cell_init_states = states_list[i]
                    cell_init_states_reverse = states_list[i + self.num_layers]
                else:
                    cell_init_states = states_list[:, i]
                    cell_init_states_reverse = states_list[:, i + self.num_layers]
                if self.training and self.dropout_p > 0:
                    if i > 1:
                        if self.invariant_dropout_mask:
                            y = y * mask[i - 1]
                        else:
                            y = F.dropout(y, p=self.dropout_p, training=True)
                y, ss, ss_r = bidirectional_rnn_cell_forward(self.cells[i], self.cells_reverse[i], y, cell_init_states, cell_init_states_reverse)
                if self.states_num() == 1:
                    new_states_list[i] = ss
                    new_states_list[i + self.num_layers] = ss_r
                else:
                    new_states_list[:, i] = torch.stack(ss)
                    new_states_list[:, i + self.num_layers] = torch.stack(ss_r)
                states_list = new_states_list.clone()
            if self.states_num() == 1:
                return y, new_states_list
            else:
                return y, tuple(new_states_list)
        else:
            if states_list.dim() == 4 and states_list.shape[1] != self.num_layers or states_list.dim() == 3 and states_list.shape[0] != self.num_layers:
                raise ValueError
            y = x.clone()
            if self.training and self.dropout_p > 0 and self.invariant_dropout_mask:
                mask = F.dropout(torch.ones(size=[self.num_layers - 1, batch_size, self.hidden_size * 2]), p=self.dropout_p, training=True, inplace=True)
            for i in range(self.num_layers):
                new_states_list = torch.zeros_like(states_list.data)
                if self.states_num() == 1:
                    cell_init_states = states_list[i]
                else:
                    cell_init_states = states_list[:, i]
                if self.training and self.dropout_p > 0:
                    if i > 1:
                        if self.invariant_dropout_mask:
                            y = y * mask[i - 1]
                        else:
                            y = F.dropout(y, p=self.dropout_p, training=True)
                y, ss = directional_rnn_cell_forward(self.cells[i], y, cell_init_states)
                if self.states_num() == 1:
                    new_states_list[i] = ss
                else:
                    new_states_list[:, i] = torch.stack(ss)
                states_list = new_states_list.clone()
            if self.states_num() == 1:
                return y, new_states_list
            else:
                return y, tuple(new_states_list)


class spikeLinear(torch.autograd.Function):

    @staticmethod
    @custom_fwd
    def forward(ctx, spike, weight, bias=None):
        if ctx.needs_input_grad[0] or ctx.needs_input_grad[1] or ctx.needs_input_grad[2]:
            if ctx.needs_input_grad[1]:
                ctx.s_shape = spike.shape
                ctx.s_tk = tensor_cache.BOOL_TENSOR_CACHE.store_bool(spike)
            if ctx.needs_input_grad[0]:
                ctx.save_for_backward(weight)
        return F.linear(spike, weight, bias)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        if ctx.needs_input_grad[1]:
            weight = ctx.saved_tensors[0]
        if ctx.needs_input_grad[0]:
            spike = tensor_cache.BOOL_TENSOR_CACHE.get_float(ctx.s_tk, ctx.s_shape)
        grad_spike = grad_weight = grad_bias = None
        if ctx.needs_input_grad[0]:
            grad_spike = F.linear(grad_output, weight.t(), bias=None)
        if ctx.needs_input_grad[1]:
            in_features = spike.shape[-1]
            out_features = grad_output.shape[-1]
            grad_weight = torch.mm(grad_output.reshape(-1, out_features).t(), spike.reshape(-1, in_features))
        if ctx.needs_input_grad[2]:
            out_features = grad_output.shape[-1]
            grad_bias = grad_output.reshape(-1, out_features).sum(0)
        return grad_spike, grad_weight, grad_bias


def spike_linear(spike: 'Tensor', weight: 'Tensor', bias: 'Optional[Tensor]'=None) ->Tensor:
    """
    * :ref:`API in English <spike_linear-en>`

    .. _spike_linear-cn:

    :class:`torch.nn.functional.linear` 在输入为脉冲时的特例。

    .. note::

        在CUDA设备上训练时拥有比 :class:`torch.nn.functional.linear` 更低的显存消耗。

    .. warning::

        `spike` 中的任何元素都必须为0或1。

    * :ref:`中文API <spike_linear-cn>`

    .. _spike_linear-en:

    A specific case of :class:`torch.nn.functional.linear` with inputs are spikes.

    .. admonition:: Note
        :class: note

        This function has less memory consumption than :class:`torch.nn.functional.linear` when training on CUDA devices.

    .. admonition:: Warning
        :class: warning

        Any element in `spike` must be 0 or 1.
    """
    if spike.get_device() < 0:
        return F.linear(spike, weight, bias)
    else:
        return spikeLinear.apply(spike, weight, bias)


class SpikeLinear(nn.Linear):
    """
    * :ref:`API in English <SpikeLinear-en>`

    .. _SpikeLinear-cn:

    :class:`torch.nn.Linear` 在输入为脉冲时的特例。

    .. note::

        在CUDA设备上运行时拥有比 :class:`torch.nn.Linear` 更低的显存消耗。

    .. warning::

        `spike` 中的任何元素都必须为0或1。

    * :ref:`中文API <SpikeLinear-cn>`

    .. _SpikeLinear-en:

    A specific case of :class:`torch.nn.Linear` with inputs are spikes.

    .. admonition:: Note
        :class: note

        This function has less memory consumption than :class:`torch.nn.Linear` when training on CUDA devices.

    .. admonition:: Warning
        :class: warning

        Any element in `spike` must be 0 or 1.
    """

    def forward(self, spike: 'Tensor') ->Tensor:
        return spike_linear(spike, self.weight, self.bias)


class spikeConvolution(torch.autograd.Function):

    @staticmethod
    @custom_fwd
    def forward(ctx, spike, weight, bias, stride, padding, dilation, groups):
        if ctx.needs_input_grad[0] or ctx.needs_input_grad[1] or ctx.needs_input_grad[2]:
            if ctx.needs_input_grad[1]:
                ctx.s_shape = spike.shape
                ctx.s_tk = tensor_cache.BOOL_TENSOR_CACHE.store_bool(spike)
            if ctx.needs_input_grad[0]:
                ctx.save_for_backward(weight)
            ctx.padding = padding
            ctx.stride = stride
            ctx.dilation = dilation
            ctx.groups = groups
            ctx.weight_shape = weight.shape
        if spike.dim() == 3:
            return F.conv1d(input=spike, weight=weight, bias=bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
        elif spike.dim() == 4:
            return F.conv2d(input=spike, weight=weight, bias=bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
        elif spike.dim() == 5:
            return F.conv3d(input=spike, weight=weight, bias=bias, stride=stride, padding=padding, dilation=dilation, groups=groups)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        grad_spike = None
        grad_weight = None
        grad_bias = None
        if ctx.needs_input_grad[0] and ctx.needs_input_grad[1]:
            weight = ctx.saved_tensors[0]
            spike = tensor_cache.BOOL_TENSOR_CACHE.get_float(ctx.s_tk, ctx.s_shape)
            weight = weight
            grad_spike, grad_weight = cpp_wrapper.cudnn_convolution_backward(spike, grad_output, weight, ctx.padding, ctx.stride, ctx.dilation, ctx.groups, torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic, torch.backends.cudnn.allow_tf32, (True, True))
        elif not ctx.needs_input_grad[0] and ctx.needs_input_grad[1]:
            spike = tensor_cache.BOOL_TENSOR_CACHE.get_float(ctx.s_tk, ctx.s_shape)
            grad_weight = cpp_wrapper.cudnn_convolution_backward_weight(ctx.weight_shape, grad_output, spike, ctx.padding, ctx.stride, ctx.dilation, ctx.groups, torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic, torch.backends.cudnn.allow_tf32)
        elif ctx.needs_input_grad[0] and not ctx.needs_input_grad[1]:
            weight = ctx.saved_tensors[0]
            weight = weight
            grad_spike = cpp_wrapper.cudnn_convolution_backward_input(ctx.spike_shape, grad_output, weight, ctx.padding, ctx.stride, ctx.dilation, ctx.groups, torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic, torch.backends.cudnn.allow_tf32)
        if ctx.needs_input_grad[2]:
            out_channels = grad_output.shape[1]
            grad_bias = grad_output.transpose(0, 1).reshape(out_channels, -1).sum(1)
        return grad_spike, grad_weight, grad_bias, None, None, None, None


def spike_conv1d(spike: 'Tensor', weight: 'Tensor', bias: 'Tensor'=None, stride: 'Union[_int, _size]'=1, padding: 'str'='valid', dilation: 'Union[_int, _size]'=1, groups: '_int'=1) ->Tensor:
    """
    * :ref:`API in English <spike_conv1d-en>`

    .. _spike_conv1d-cn:

    :class:`torch.nn.functional.conv1d` 在输入为脉冲时的特例。

    .. note::

        在CUDA设备上训练时拥有比 :class:`torch.nn.functional.conv1d` 更低的显存消耗。

    .. warning::

        `spike` 中的任何元素都必须为0或1。

    * :ref:`中文API <spike_conv1d-cn>`

    .. _spike_conv1d-en:

    A specific case of :class:`torch.nn.functional.conv1d` with inputs are spikes.

    .. admonition:: Note
        :class: note

        This function has less memory consumption than :class:`torch.nn.functional.conv1d` when training on CUDA devices.

    .. admonition:: Warning
        :class: warning

        Any element in `spike` must be 0 or 1.
    """
    if spike.get_device() < 0:
        return F.conv1d(spike, weight, bias, stride, padding, dilation, groups)
    else:
        return spikeConvolution.apply(spike, weight, bias, stride, padding, dilation, groups)


class SpikeConv1d(nn.Conv1d):
    """
    * :ref:`API in English <SpikeConv1d-en>`

    .. _SpikeConv1d-cn:

    :class:`torch.nn.Conv1d` 在输入为脉冲时的特例。

    .. note::

        在CUDA设备上运行时拥有比 :class:`torch.nn.Conv1d` 更低的显存消耗。

    .. warning::

        `spike` 中的任何元素都必须为0或1。

    * :ref:`中文API <SpikeConv1d-cn>`

    .. _SpikeConv1d-en:

    A specific case of :class:`torch.nn.Conv1d` with inputs are spikes.

    .. admonition:: Note
        :class: note

        This function has less memory consumption than :class:`torch.nn.Conv1d` when training on CUDA devices.

    .. admonition:: Warning
        :class: warning

        Any element in `spike` must be 0 or 1.
    """

    def _conv_forward(self, spike: 'Tensor', weight: 'Tensor', bias: 'Optional[Tensor]'):
        if self.padding_mode != 'zeros':
            return spike_conv1d(F.pad(spike, self._reversed_padding_repeated_twice, mode=self.padding_mode), weight, bias, self.stride, _single(0), self.dilation, self.groups)
        return spike_conv1d(spike, weight, bias, self.stride, self.padding, self.dilation, self.groups)


def spike_conv2d(spike: 'Tensor', weight: 'Tensor', bias: 'Optional[Tensor]'=None, stride: 'Union[_int, _size]'=1, padding: 'str'='valid', dilation: 'Union[_int, _size]'=1, groups: '_int'=1) ->Tensor:
    """
    * :ref:`API in English <spike_conv2d-en>`

    .. _spike_conv2d-cn:

    :class:`torch.nn.functional.conv2d` 在输入为脉冲时的特例。

    .. note::

        在CUDA设备上训练时拥有比 :class:`torch.nn.functional.conv2d` 更低的显存消耗。

    .. warning::

        `spike` 中的任何元素都必须为0或1。

    * :ref:`中文API <spike_conv2d-cn>`

    .. _spike_conv2d-en:

    A specific case of :class:`torch.nn.functional.conv2d` with inputs are spikes.

    .. admonition:: Note
        :class: note

        This function has less memory consumption than :class:`torch.nn.functional.conv2d` when training on CUDA devices.

    .. admonition:: Warning
        :class: warning

        Any element in `spike` must be 0 or 1.
    """
    if spike.get_device() < 0:
        return F.conv2d(spike, weight, bias, stride, padding, dilation, groups)
    else:
        return spikeConvolution.apply(spike, weight, bias, stride, padding, dilation, groups)


class SpikeConv2d(nn.Conv2d):
    """
    * :ref:`API in English <SpikeConv2d-en>`

    .. _SpikeConv2d-cn:

    :class:`torch.nn.Conv2d` 在输入为脉冲时的特例。

    .. note::

        在CUDA设备上运行时拥有比 :class:`torch.nn.Conv2d` 更低的显存消耗。

    .. warning::

        `spike` 中的任何元素都必须为0或1。

    * :ref:`中文API <SpikeConv2d-cn>`

    .. _SpikeConv2d-en:

    A specific case of :class:`torch.nn.Conv2d` with inputs are spikes.

    .. admonition:: Note
        :class: note

        This function has less memory consumption than :class:`torch.nn.Conv2d` when training on CUDA devices.

    .. admonition:: Warning
        :class: warning

        Any element in `spike` must be 0 or 1.
    """

    def _conv_forward(self, spike: 'Tensor', weight: 'Tensor', bias: 'Optional[Tensor]'):
        if self.padding_mode != 'zeros':
            return spike_conv2d(F.pad(spike, self._reversed_padding_repeated_twice, mode=self.padding_mode), weight, bias, self.stride, _pair(0), self.dilation, self.groups)
        return spike_conv2d(spike, weight, bias, self.stride, self.padding, self.dilation, self.groups)


def spike_conv3d(spike: 'Tensor', weight: 'Tensor', bias: 'Optional[Tensor]'=None, stride: 'Union[_int, _size]'=1, padding: 'str'='valid', dilation: 'Union[_int, _size]'=1, groups: '_int'=1) ->Tensor:
    """
    * :ref:`API in English <spike_conv3d-en>`

    .. _spike_conv3d-cn:

    :class:`torch.nn.functional.conv3d` 在输入为脉冲时的特例。

    .. note::

        在CUDA设备上训练时拥有比 :class:`torch.nn.functional.conv3d` 更低的显存消耗。

    .. warning::

        `spike` 中的任何元素都必须为0或1。

    * :ref:`中文API <spike_conv3d-cn>`

    .. _spike_conv3d-en:

    A specific case of :class:`torch.nn.functional.conv3d` with inputs are spikes.

    .. admonition:: Note
        :class: note

        This function has less memory consumption than :class:`torch.nn.functional.conv3d` when training on CUDA devices.

    .. admonition:: Warning
        :class: warning

        Any element in `spike` must be 0 or 1.
    """
    if spike.get_device() < 0:
        return F.conv3d(spike, weight, bias, stride, padding, dilation, groups)
    else:
        return spikeConvolution.apply(spike, weight, bias, stride, padding, dilation, groups)


class SpikeConv3d(nn.Conv3d):
    """
    * :ref:`API in English <SpikeConv3d-en>`

    .. _SpikeConv3d-cn:

    :class:`torch.nn.Conv3d` 在输入为脉冲时的特例。

    .. note::

        在CUDA设备上运行时拥有比 :class:`torch.nn.Conv3d` 更低的显存消耗。

    .. warning::

        `spike` 中的任何元素都必须为0或1。

    * :ref:`中文API <SpikeConv3d-cn>`

    .. _SpikeConv3d-en:

    A specific case of :class:`torch.nn.Conv3d` with inputs are spikes.

    .. admonition:: Note
        :class: note

        This function has less memory consumption than :class:`torch.nn.Conv3d` when training on CUDA devices.

    .. admonition:: Warning
        :class: warning

        Any element in `spike` must be 0 or 1.
    """

    def _conv_forward(self, spike: 'Tensor', weight: 'Tensor', bias: 'Optional[Tensor]'):
        if self.padding_mode != 'zeros':
            return spike_conv3d(F.pad(spike, self._reversed_padding_repeated_twice, mode=self.padding_mode), weight, bias, self.stride, _triple(0), self.dilation, self.groups)
        return spike_conv3d(spike, weight, bias, self.stride, self.padding, self.dilation, self.groups)


class MultiArgsSurrogateFunctionBase(nn.Module):

    def __init__(self, spiking: 'bool', *args, **kwargs):
        super().__init__()
        self.spiking = spiking

    def set_spiking_mode(self, spiking: 'bool'):
        self.spiking = spiking

    def cuda_code(self, x: 'str', y: 'str', dtype='fp32'):
        raise NotImplementedError

    def cuda_code_start_comments(self):
        return f'// start: spikingjelly.activation_based.surrogate.{self._get_name()}.cuda_code'

    def cuda_code_end_comments(self):
        return f'// end: spikingjelly.activation_based.surrogate.{self._get_name()}.cuda_code'

    def cuda_codes(self, y: 'str', x: 'str', dtype: 'str'):
        raise NotImplementedError


@torch.jit.script
def piecewise_quadratic_backward(grad_output: 'torch.Tensor', x: 'torch.Tensor', alpha: 'float'):
    x_abs = x.abs()
    mask = x_abs > 1 / alpha
    grad_x = (grad_output * (-alpha ** 2 * x_abs + alpha)).masked_fill_(mask, 0)
    return grad_x, None


class piecewise_quadratic(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, alpha):
        if x.requires_grad:
            ctx.save_for_backward(x)
            ctx.alpha = alpha
        return heaviside(x)

    @staticmethod
    def backward(ctx, grad_output):
        return piecewise_quadratic_backward(grad_output, ctx.saved_tensors[0], ctx.alpha)


class PiecewiseQuadratic(SurrogateFunctionBase):

    def __init__(self, alpha=1.0, spiking=True):
        """
        * :ref:`API in English <PiecewiseQuadratic.__init__-en>`
        .. _PiecewiseQuadratic.__init__-cn:

        :param alpha: 控制反向传播时梯度的平滑程度的参数
        :param spiking: 是否输出脉冲，默认为 ``True``，在前向传播时使用 ``heaviside`` 而在反向传播使用替代梯度。若为 ``False``
            则不使用替代梯度，前向传播时，使用反向传播时的梯度替代函数对应的原函数

        反向传播时使用分段二次函数的梯度（三角形函数）的脉冲发放函数。反向传播为

        .. math::
            g'(x) =
            \\begin{cases}
            0, & |x| > \\frac{1}{\\alpha} \\\\
            -\\alpha^2|x|+\\alpha, & |x| \\leq \\frac{1}{\\alpha}
            \\end{cases}

        对应的原函数为

        .. math::
            g(x) =
            \\begin{cases}
            0, & x < -\\frac{1}{\\alpha} \\\\
            -\\frac{1}{2}\\alpha^2|x|x + \\alpha x + \\frac{1}{2}, & |x| \\leq \\frac{1}{\\alpha}  \\\\
            1, & x > \\frac{1}{\\alpha} \\\\
            \\end{cases}

        .. image:: ../_static/API/activation_based/surrogate/PiecewiseQuadratic.*
            :width: 100%

        该函数在文章 [#esser2016convolutional]_ [#STBP]_ [#LSNN]_ [#neftci2019surrogate]_ [#panda2020toward]_ 中使用。

        * :ref:`中文API <PiecewiseQuadratic.__init__-cn>`
        .. _PiecewiseQuadratic.__init__-en:

        :param alpha: parameter to control smoothness of gradient
        :param spiking: whether output spikes. The default is ``True`` which means that using ``heaviside`` in forward
            propagation and using surrogate gradient in backward propagation. If ``False``, in forward propagation,
            using the primitive function of the surrogate gradient function used in backward propagation

        The piecewise quadratic surrogate spiking function. The gradient is defined by

        .. math::
            g'(x) =
            \\begin{cases}
            0, & |x| > \\frac{1}{\\alpha} \\\\
            -\\alpha^2|x|+\\alpha, & |x| \\leq \\frac{1}{\\alpha}
            \\end{cases}

        The primitive function is defined by

        .. math::
            g(x) =
            \\begin{cases}
            0, & x < -\\frac{1}{\\alpha} \\\\
            -\\frac{1}{2}\\alpha^2|x|x + \\alpha x + \\frac{1}{2}, & |x| \\leq \\frac{1}{\\alpha}  \\\\
            1, & x > \\frac{1}{\\alpha} \\\\
            \\end{cases}

        .. image:: ../_static/API/activation_based/surrogate/PiecewiseQuadratic.*
            :width: 100%

        The function is used in [#esser2016convolutional]_ [#STBP]_ [#LSNN]_ [#neftci2019surrogate]_ [#panda2020toward]_.

        """
        super().__init__(alpha, spiking)

    @staticmethod
    def spiking_function(x, alpha):
        return piecewise_quadratic.apply(x, alpha)

    @staticmethod
    @torch.jit.script
    def primitive_function(x: 'torch.Tensor', alpha: 'float'):
        mask0 = x > 1.0 / alpha
        mask1 = x.abs() <= 1.0 / alpha
        return mask0 + mask1 * (-alpha ** 2 / 2 * x.square() * x.sign() + alpha * x + 0.5)

    @staticmethod
    def backward(grad_output, x, alpha):
        return piecewise_quadratic_backward(grad_output, x, alpha)[0]


@torch.jit.script
def piecewise_exp_backward(grad_output: 'torch.Tensor', x: 'torch.Tensor', alpha: 'float'):
    return alpha / 2 * (-alpha * x.abs()).exp_() * grad_output, None


class piecewise_exp(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, alpha):
        if x.requires_grad:
            ctx.save_for_backward(x)
            ctx.alpha = alpha
        return heaviside(x)

    @staticmethod
    def backward(ctx, grad_output):
        return piecewise_exp_backward(grad_output, ctx.saved_tensors[0], ctx.alpha)


class PiecewiseExp(SurrogateFunctionBase):

    def __init__(self, alpha=1.0, spiking=True):
        """
        * :ref:`API in English <PiecewiseExp.__init__-en>`
        .. _PiecewiseExp.__init__-cn:

        :param alpha: 控制反向传播时梯度的平滑程度的参数
        :param spiking: 是否输出脉冲，默认为 ``True``，在前向传播时使用 ``heaviside`` 而在反向传播使用替代梯度。若为 ``False``
            则不使用替代梯度，前向传播时，使用反向传播时的梯度替代函数对应的原函数

        反向传播时使用分段指数函数的梯度的脉冲发放函数。反向传播为

        .. math::
            g'(x) = \\frac{\\alpha}{2}e^{-\\alpha |x|}

        对应的原函数为

        .. math::
            g(x) =
            \\begin{cases}
            \\frac{1}{2}e^{\\alpha x}, & x < 0 \\\\
            1 - \\frac{1}{2}e^{-\\alpha x}, & x \\geq 0
            \\end{cases}

        .. image:: ../_static/API/activation_based/surrogate/PiecewiseExp.*
            :width: 100%

        该函数在文章 [#SLAYER]_ [#neftci2019surrogate]_ 中使用。

        * :ref:`中文API <PiecewiseExp.__init__-cn>`
        .. _PiecewiseExp.__init__-en:

        :param alpha: parameter to control smoothness of gradient
        :param spiking: whether output spikes. The default is ``True`` which means that using ``heaviside`` in forward
            propagation and using surrogate gradient in backward propagation. If ``False``, in forward propagation,
            using the primitive function of the surrogate gradient function used in backward propagation

        The piecewise exponential surrogate spiking function. The gradient is defined by

        .. math::
            g'(x) = \\frac{\\alpha}{2}e^{-\\alpha |x|}

        The primitive function is defined by

        .. math::
            g(x) =
            \\begin{cases}
            \\frac{1}{2}e^{\\alpha x}, & x < 0 \\\\
            1 - \\frac{1}{2}e^{-\\alpha x}, & x \\geq 0
            \\end{cases}

        .. image:: ../_static/API/activation_based/surrogate/PiecewiseExp.*
            :width: 100%

        The function is used in [#SLAYER]_ [#neftci2019surrogate]_ .
        """
        super().__init__(alpha, spiking)

    @staticmethod
    def spiking_function(x, alpha):
        return piecewise_exp.apply(x, alpha)

    @staticmethod
    @torch.jit.script
    def primitive_function(x: 'torch.Tensor', alpha: 'float'):
        mask_nonnegative = heaviside(x)
        mask_sign = mask_nonnegative * 2.0 - 1.0
        exp_x = (mask_sign * x * -alpha).exp_() / 2.0
        return mask_nonnegative - exp_x * mask_sign

    @staticmethod
    def backward(grad_output, x, alpha):
        return piecewise_exp_backward(grad_output, x, alpha)[0]


@torch.jit.script
def sigmoid_backward(grad_output: 'torch.Tensor', x: 'torch.Tensor', alpha: 'float'):
    sgax = (x * alpha).sigmoid_()
    return grad_output * (1.0 - sgax) * sgax * alpha, None


class sigmoid(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, alpha):
        if x.requires_grad:
            ctx.save_for_backward(x)
            ctx.alpha = alpha
        return heaviside(x)

    @staticmethod
    def backward(ctx, grad_output):
        return sigmoid_backward(grad_output, ctx.saved_tensors[0], ctx.alpha)


tab4_str = '\t\t\t\t'


class Sigmoid(SurrogateFunctionBase):

    def __init__(self, alpha=4.0, spiking=True):
        """
        * :ref:`API in English <Sigmoid.__init__-en>`
        .. _Sigmoid.__init__-cn:

        :param alpha: 控制反向传播时梯度的平滑程度的参数
        :param spiking: 是否输出脉冲，默认为 ``True``，在前向传播时使用 ``heaviside`` 而在反向传播使用替代梯度。若为 ``False``
            则不使用替代梯度，前向传播时，使用反向传播时的梯度替代函数对应的原函数

        反向传播时使用sigmoid的梯度的脉冲发放函数。反向传播为

        .. math::
            g'(x) = \\alpha * (1 - \\mathrm{sigmoid} (\\alpha x)) \\mathrm{sigmoid} (\\alpha x)

        对应的原函数为

        .. math::
            g(x) = \\mathrm{sigmoid}(\\alpha x) = \\frac{1}{1+e^{-\\alpha x}}

        .. image:: ../_static/API/activation_based/surrogate/Sigmoid.*
            :width: 100%

        该函数在文章 [#STBP]_ [#roy2019scaling]_ [#SNNLSTM]_ [#SNU]_ 中使用。

        * :ref:`中文API <Sigmoid.__init__-cn>`
        .. _Sigmoid.__init__-en:

        :param alpha: parameter to control smoothness of gradient
        :param spiking: whether output spikes. The default is ``True`` which means that using ``heaviside`` in forward
            propagation and using surrogate gradient in backward propagation. If ``False``, in forward propagation,
            using the primitive function of the surrogate gradient function used in backward propagation

        The sigmoid surrogate spiking function. The gradient is defined by

        .. math::
            g'(x) = \\alpha * (1 - \\mathrm{sigmoid} (\\alpha x)) \\mathrm{sigmoid} (\\alpha x)

        The primitive function is defined by

        .. math::
            g(x) = \\mathrm{sigmoid}(\\alpha x) = \\frac{1}{1+e^{-\\alpha x}}

        .. image:: ../_static/API/activation_based/surrogate/Sigmoid.*
            :width: 100%

        The function is used in  [#STBP]_ [#roy2019scaling]_ [#SNNLSTM]_ [#SNU]_ .
        """
        super().__init__(alpha, spiking)

    @staticmethod
    def spiking_function(x, alpha):
        return sigmoid.apply(x, alpha)

    @staticmethod
    @torch.jit.script
    def primitive_function(x: 'torch.Tensor', alpha: 'float'):
        return (x * alpha).sigmoid()

    @staticmethod
    def backward(grad_output, x, alpha):
        return sigmoid_backward(grad_output, x, alpha)[0]

    def cuda_code(self, x: 'str', y: 'str', dtype='fp32'):
        sg_name = 'sg_' + self._get_name()
        alpha = str(self.alpha) + 'f'
        code = f"""
            {tab4_str}{self.cuda_code_start_comments()}
        """
        if dtype == 'fp32':
            code += f"""
            {tab4_str}const float {sg_name}_sigmoid_ax = 1.0f / (1.0f + expf(- {alpha} * {x}));
            {tab4_str}const float {y} = (1.0f - {sg_name}_sigmoid_ax) * {sg_name}_sigmoid_ax * {alpha};
            """
        elif dtype == 'fp16':
            code += f"""
            {tab4_str}const half2 {sg_name}_alpha = __float2half2_rn({alpha});
            {tab4_str}const half2 {sg_name}_sigmoid_ax = __h2div(__float2half2_rn(1.0f), __hadd2(h2exp(__hneg2(__hmul2({sg_name}_alpha, {x}))), __float2half2_rn(1.0f)));
            {tab4_str}const half2 {y} = __hmul2(__hmul2(__hsub2(__float2half2_rn(1.0f), {sg_name}_sigmoid_ax), {sg_name}_sigmoid_ax), {sg_name}_alpha);
            """
        else:
            raise NotImplementedError
        code += f"""
            {tab4_str}{self.cuda_code_end_comments()}
        """
        return code

    def cuda_codes(self, y: 'str', x: 'str', dtype: 'str'):
        return cfunction.sigmoid_backward(y=y, x=x, alpha=self.alpha, dtype=dtype)


@torch.jit.script
def soft_sign_backward(grad_output: 'torch.Tensor', x: 'torch.Tensor', alpha: 'float'):
    return grad_output / (2 * alpha * (1 / alpha + x.abs()).pow_(2)), None


class soft_sign(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, alpha):
        if x.requires_grad:
            ctx.save_for_backward(x)
            ctx.alpha = alpha
        return heaviside(x)

    @staticmethod
    def backward(ctx, grad_output):
        return soft_sign_backward(grad_output, ctx.saved_tensors[0], ctx.alpha)


@torch.jit.script
def atan_backward(grad_output: 'torch.Tensor', x: 'torch.Tensor', alpha: 'float'):
    return alpha / 2 / (1 + (math.pi / 2 * alpha * x).pow_(2)) * grad_output, None


class atan(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, alpha):
        if x.requires_grad:
            ctx.save_for_backward(x)
            ctx.alpha = alpha
        return heaviside(x)

    @staticmethod
    def backward(ctx, grad_output):
        return atan_backward(grad_output, ctx.saved_tensors[0], ctx.alpha)


@torch.jit.script
def super_spike_backward(grad_output: 'torch.Tensor', x: 'torch.Tensor', alpha: 'float'):
    return alpha * grad_output / torch.pow(torch.abs(x) + 1.0, 2), None


class SuperSpike(SurrogateFunctionBase):

    def __init__(self, alpha=1.0, spiking=True):
        """
        * :ref:`API in English <SuperSpike.__init__-en>`
        .. _SuperSpike.__init__-cn:
    
        `SuperSpike: Supervised learning in multi-layer spiking neural networks <https://arxiv.org/abs/1705.11146>`_ 提出的反向传播时使用SuperSpike的梯度的脉冲发放函数。反向传播为

        .. math::
            g'(x) = \\frac{\\alpha}{(1 + (|x|))^2}


        * :ref:`中文API <SuperSpike.__init__-cn>`
        .. _SuperSpike.__init__-en:

        The SuperSpike surrogate spiking function proposed by `SuperSpike: Supervised learning in multi-layer spiking neural networks <https://arxiv.org/abs/1705.11146>`_. The gradient is defined by

        .. math::
            g'(x) = \\frac{\\alpha}{(1 + (|x|))^2}
        """
        super().__init__(alpha, spiking)

    @staticmethod
    def spiking_function(x, alpha):
        return atan.apply(x, alpha)

    @staticmethod
    @torch.jit.script
    def primitive_function(x: 'torch.Tensor', alpha: 'float'):
        raise NotImplementedError

    @staticmethod
    def backward(grad_output, x, alpha):
        return super_spike_backward(grad_output, x, alpha)[0]

    def cuda_code(self, x: 'str', y: 'str', dtype='fp32'):
        raise NotImplementedError

    def cuda_codes(self, y: 'str', x: 'str', dtype: 'str'):
        raise NotImplementedError


class ATan(SurrogateFunctionBase):

    def __init__(self, alpha=2.0, spiking=True):
        """
        * :ref:`API in English <ATan.__init__-en>`
        .. _ATan.__init__-cn:

        反向传播时使用反正切函数arc tangent的梯度的脉冲发放函数。反向传播为

        .. math::
            g'(x) = \\frac{\\alpha}{2(1 + (\\frac{\\pi}{2}\\alpha x)^2)}

        对应的原函数为

        .. math::
            g(x) = \\frac{1}{\\pi} \\arctan(\\frac{\\pi}{2}\\alpha x) + \\frac{1}{2}

        .. image:: ../_static/API/activation_based/surrogate/ATan.*
            :width: 100%

        * :ref:`中文API <ATan.__init__-cn>`
        .. _ATan.__init__-en:

        The arc tangent surrogate spiking function. The gradient is defined by

        .. math::
            g'(x) = \\frac{\\alpha}{2(1 + (\\frac{\\pi}{2}\\alpha x)^2)}

        The primitive function is defined by

        .. math::
            g(x) = \\frac{1}{\\pi} \\arctan(\\frac{\\pi}{2}\\alpha x) + \\frac{1}{2}

        .. image:: ../_static/API/activation_based/surrogate/ATan.*
            :width: 100%
        """
        super().__init__(alpha, spiking)

    @staticmethod
    def spiking_function(x, alpha):
        return atan.apply(x, alpha)

    @staticmethod
    @torch.jit.script
    def primitive_function(x: 'torch.Tensor', alpha: 'float'):
        return (math.pi / 2 * alpha * x).atan_() / math.pi + 0.5

    @staticmethod
    def backward(grad_output, x, alpha):
        return atan_backward(grad_output, x, alpha)[0]

    def cuda_code(self, x: 'str', y: 'str', dtype='fp32'):
        sg_name = 'sg_' + self._get_name()
        alpha = str(self.alpha) + 'f'
        code = f"""
            {tab4_str}{self.cuda_code_start_comments()}
        """
        if dtype == 'fp32':
            code += f"""
            {tab4_str}const float {sg_name}_M_PI_2__alpha__x = ((float) 1.57079632679489661923) * {alpha} * {x};
            {tab4_str}const float {y} = {alpha} / 2.0f / (1.0f + {sg_name}_M_PI_2__alpha__x * {sg_name}_M_PI_2__alpha__x);
            """
        elif dtype == 'fp16':
            code += f"""
            {tab4_str}const half2 {sg_name}_alpha =  __float2half2_rn({alpha});
            {tab4_str}const half2 {sg_name}_M_PI_2__alpha__x = __hmul2(__hmul2(__float2half2_rn((float) 1.57079632679489661923), {sg_name}_alpha), {x});
            {tab4_str}const half2 {y} = __h2div(__h2div({sg_name}_alpha, __float2half2_rn(2.0f)), __hfma2({sg_name}_M_PI_2__alpha__x, {sg_name}_M_PI_2__alpha__x, __float2half2_rn(1.0f)));
            """
        else:
            raise NotImplementedError
        code += f"""
            {tab4_str}{self.cuda_code_end_comments()}
        """
        return code

    def cuda_codes(self, y: 'str', x: 'str', dtype: 'str'):
        return cfunction.atan_backward(y=y, x=x, alpha=self.alpha, dtype=dtype)


@torch.jit.script
def nonzero_sign_log_abs_backward(grad_output: 'torch.Tensor', x: 'torch.Tensor', alpha: 'float'):
    return grad_output / (1 / alpha + x.abs()), None


class nonzero_sign_log_abs(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, alpha):
        if x.requires_grad:
            ctx.save_for_backward(x)
            ctx.alpha = alpha
        return heaviside(x)

    @staticmethod
    def backward(ctx, grad_output):
        return nonzero_sign_log_abs_backward((grad_output, ctx.saved_tensors[0], ctx.alpha))


class NonzeroSignLogAbs(SurrogateFunctionBase):

    def __init__(self, alpha=1.0, spiking=True):
        """
        * :ref:`API in English <LogAbs.__init__-en>`
        .. _LogAbs.__init__-cn:

        :param alpha: 控制反向传播时梯度的平滑程度的参数
        :param spiking: 是否输出脉冲，默认为 ``True``，在前向传播时使用 ``heaviside`` 而在反向传播使用替代梯度。若为 ``False``
            则不使用替代梯度，前向传播时，使用反向传播时的梯度替代函数对应的原函数

        .. warning::
            原函数的输出范围并不是(0, 1)。它的优势是反向传播的计算量特别小。

        反向传播时使用NonzeroSignLogAbs的梯度的脉冲发放函数。反向传播为

        .. math::
            g'(x) = \\frac{\\alpha}{1 + |\\alpha x|} = \\frac{1}{\\frac{1}{\\alpha} + |x|}

        对应的原函数为

        .. math::
            g(x) = \\mathrm{NonzeroSign}(x) \\log (|\\alpha x| + 1)

        其中

            .. math::
                \\mathrm{NonzeroSign}(x) =
                \\begin{cases}
                1, & x \\geq 0 \\\\
                -1, & x < 0 \\\\
                \\end{cases}

        .. image:: ../_static/API/activation_based/surrogate/NonzeroSignLogAbs.*
            :width: 100%

        * :ref:`中文API <LogAbs.__init__-cn>`
        .. _LogAbs.__init__-en:

        :param alpha: parameter to control smoothness of gradient
        :param spiking: whether output spikes. The default is ``True`` which means that using ``heaviside`` in forward
            propagation and using surrogate gradient in backward propagation. If ``False``, in forward propagation,
            using the primitive function of the surrogate gradient function used in backward propagation

        .. admonition:: Warning
            :class: warning

            The output range the primitive function is not (0, 1). The advantage of this function is that computation
            cost is small when backward.

        The NonzeroSignLogAbs surrogate spiking function. The gradient is defined by

        .. math::
            g'(x) = \\frac{\\alpha}{1 + |\\alpha x|} = \\frac{1}{\\frac{1}{\\alpha} + |x|}

        The primitive function is defined by

        .. math::
            g(x) = \\mathrm{NonzeroSign}(x) \\log (|\\alpha x| + 1)

        where

        .. math::
            \\mathrm{NonzeroSign}(x) =
            \\begin{cases}
            1, & x \\geq 0 \\\\
            -1, & x < 0 \\\\
            \\end{cases}

        .. image:: ../_static/API/activation_based/surrogate/NonzeroSignLogAbs.*
            :width: 100%

        """
        super().__init__(alpha, spiking)

    @staticmethod
    def spiking_function(x, alpha):
        return nonzero_sign_log_abs.apply(x, alpha)

    @staticmethod
    def backward(grad_output, x, alpha):
        return nonzero_sign_log_abs_backward(grad_output, x, alpha)[0]

    @staticmethod
    @torch.jit.script
    def primitive_function(x: 'torch.Tensor', alpha: 'float'):
        mask_p = heaviside(x) * 2.0 - 1.0
        return mask_p * (alpha * mask_p * x + 1).log()


@torch.jit.script
def erf_backward(grad_output: 'torch.Tensor', x: 'torch.Tensor', alpha: 'float'):
    return grad_output * (-(x * alpha).pow_(2)).exp_() * (alpha / math.sqrt(math.pi)), None


class erf(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, alpha):
        if x.requires_grad:
            ctx.save_for_backward(x)
            ctx.alpha = alpha
        return heaviside(x)

    @staticmethod
    def backward(ctx, grad_output):
        return erf_backward(grad_output, ctx.saved_tensors[0], ctx.alpha)


class Erf(SurrogateFunctionBase):

    def __init__(self, alpha=2.0, spiking=True):
        """
        * :ref:`API in English <Erf.__init__-en>`
        .. _Erf.__init__-cn:

        :param alpha: 控制反向传播时梯度的平滑程度的参数
        :param spiking: 是否输出脉冲，默认为 ``True``，在前向传播时使用 ``heaviside`` 而在反向传播使用替代梯度。若为 ``False``
            则不使用替代梯度，前向传播时，使用反向传播时的梯度替代函数对应的原函数

        反向传播时使用高斯误差函数(erf)的梯度的脉冲发放函数。反向传播为

        .. math::
            g'(x) = \\frac{\\alpha}{\\sqrt{\\pi}}e^{-\\alpha^2x^2}

        对应的原函数为

        .. math::
            :nowrap:

            \\begin{split}
            g(x) &= \\frac{1}{2}(1-\\text{erf}(-\\alpha x)) \\\\
            &= \\frac{1}{2} \\text{erfc}(-\\alpha x) \\\\
            &= \\frac{1}{\\sqrt{\\pi}}\\int_{-\\infty}^{\\alpha x}e^{-t^2}dt
            \\end{split}

        .. image:: ../_static/API/activation_based/surrogate/Erf.*
            :width: 100%

        该函数在文章 [#esser2015backpropagation]_ [#STBP]_ [#SRNN]_ 中使用。

        * :ref:`中文API <Erf.__init__-cn>`
        .. _Erf.__init__-en:

        :param alpha: parameter to control smoothness of gradient
        :param spiking: whether output spikes. The default is ``True`` which means that using ``heaviside`` in forward
            propagation and using surrogate gradient in backward propagation. If ``False``, in forward propagation,
            using the primitive function of the surrogate gradient function used in backward propagation

        The Gaussian error (erf) surrogate spiking function. The gradient is defined by

        .. math::
            g'(x) = \\frac{\\alpha}{\\sqrt{\\pi}}e^{-\\alpha^2x^2}

        The primitive function is defined by

        .. math::
            :nowrap:

            \\begin{split}
            g(x) &= \\frac{1}{2}(1-\\text{erf}(-\\alpha x)) \\\\
            &= \\frac{1}{2} \\text{erfc}(-\\alpha x) \\\\
            &= \\frac{1}{\\sqrt{\\pi}}\\int_{-\\infty}^{\\alpha x}e^{-t^2}dt
            \\end{split}

        .. image:: ../_static/API/activation_based/surrogate/Erf.*
            :width: 100%

        The function is used in [#esser2015backpropagation]_ [#STBP]_ [#SRNN]_.
        """
        super().__init__(alpha, spiking)

    @staticmethod
    def spiking_function(x, alpha):
        return erf.apply(x, alpha)

    @staticmethod
    @torch.jit.script
    def primitive_function(x: 'torch.Tensor', alpha: 'float'):
        return torch.erfc_(-alpha * x) / 2.0

    @staticmethod
    def backward(grad_output, x, alpha):
        return erf_backward(grad_output, x, alpha)[0]


curly_bracket_l = '{'


curly_bracket_r = '}'


@torch.jit.script
def piecewise_leaky_relu_backward(grad_output: 'torch.Tensor', x: 'torch.Tensor', w: 'float', c: 'float'):
    mask_width = x.abs() < w
    mask_c = mask_width.logical_not()
    return grad_output * x.masked_fill(mask_width, 1 / (2 * w)).masked_fill(mask_c, c), None, None


class piecewise_leaky_relu(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x: 'torch.Tensor', w=1, c=0.01):
        if x.requires_grad:
            ctx.save_for_backward(x)
            ctx.w = w
            ctx.c = c
        return heaviside(x)

    @staticmethod
    def backward(ctx, grad_output):
        return piecewise_leaky_relu_backward(grad_output, ctx.saved_tensors[0], ctx.w, ctx.c)


class PiecewiseLeakyReLU(MultiArgsSurrogateFunctionBase):

    def __init__(self, w=1.0, c=0.01, spiking=True):
        """
        * :ref:`API in English <PiecewiseLeakyReLU.__init__-en>`
        .. _PiecewiseLeakyReLU.__init__-cn:

        :param w: ``-w <= x <= w`` 时反向传播的梯度为 ``1 / 2w``
        :param c: ``x > w`` 或 ``x < -w`` 时反向传播的梯度为 ``c``
        :param spiking: 是否输出脉冲，默认为 ``True``，在前向传播时使用 ``heaviside`` 而在反向传播使用替代梯度。若为 ``False``
            则不使用替代梯度，前向传播时，使用反向传播时的梯度替代函数对应的原函数

        分段线性的近似脉冲发放函数。梯度为

        .. math::
            g'(x) =
            \\begin{cases}
            \\frac{1}{2w}, & -w \\leq x \\leq w \\\\
            c, & x < -w ~or~ x > w
            \\end{cases}

        对应的原函数为

        .. math::
            g(x) =
            \\begin{cases}
            cx + cw, & x < -w \\\\
            \\frac{1}{2w}x + \\frac{1}{2}, & -w \\leq x \\leq w \\\\
            cx - cw + 1, & x > w \\\\
            \\end{cases}

        .. image:: ../_static/API/activation_based/surrogate/PiecewiseLeakyReLU.*
            :width: 100%

        该函数在文章 [#yin2017algorithm]_ [#STBP]_ [#huh2018gradient]_ [#wu2019direct]_ [#STCA]_ [#roy2019scaling]_ [#LISNN]_ [#DECOLLE]_ 中使用。

        * :ref:`中文API <PiecewiseLeakyReLU.__init__-cn>`
        .. _PiecewiseLeakyReLU.__init__-en:

        :param w: when ``-w <= x <= w`` the gradient is ``1 / 2w``
        :param c: when ``x > w`` or ``x < -w`` the gradient is ``c``
        :param spiking: whether output spikes. The default is ``True`` which means that using ``heaviside`` in forward
            propagation and using surrogate gradient in backward propagation. If ``False``, in forward propagation,
            using the primitive function of the surrogate gradient function used in backward propagation

        The piecewise surrogate spiking function. The gradient is defined by

        .. math::
            g'(x) =
            \\begin{cases}
            \\frac{1}{2w}, & -w \\leq x \\leq w \\\\
            c, & x < -w ~or~ x > w
            \\end{cases}

        The primitive function is defined by

        .. math::
            g(x) =
            \\begin{cases}
            cx + cw, & x < -w \\\\
            \\frac{1}{2w}x + \\frac{1}{2}, & -w \\leq x \\leq w \\\\
            cx - cw + 1, & x > w
            \\end{cases}

        .. image:: ../_static/API/activation_based/surrogate/PiecewiseLeakyReLU.*
            :width: 100%

        The function is used in [#yin2017algorithm]_ [#STBP]_ [#huh2018gradient]_ [#wu2019direct]_ [#STCA]_ [#roy2019scaling]_ [#LISNN]_ [#DECOLLE]_.
        """
        super().__init__(spiking)
        assert w > 0.0
        self.w = w
        self.c = c
        self.spiking = spiking

    def forward(self, x):
        if self.spiking:
            f = self.spiking_function
        else:
            f = self.primitive_function
        return f(x, self.w, self.c)

    @staticmethod
    def spiking_function(x: 'torch.Tensor', w, c):
        return piecewise_leaky_relu.apply(x, w, c)

    @staticmethod
    def backward(grad_output, x, w, c):
        return piecewise_leaky_relu_backward(grad_output, x, w, c)[0]

    @staticmethod
    @torch.jit.script
    def primitive_function(x: 'torch.Tensor', w: 'float', c: 'float'):
        mask0 = x < -w
        mask1 = x > w
        mask2 = torch.ones_like(x.data) - mask0 - mask1
        if c == 0:
            return mask2 * (x / (2 * w) + 1 / 2) + mask1
        else:
            cw = c * w
            return mask0 * (c * x + cw) + mask1 * (c * x + (-cw + 1)) + mask2 * (x / (2 * w) + 1 / 2)

    def cuda_code(self, x: 'str', y: 'str', dtype='fp32'):
        sg_name = 'sg_' + self._get_name()
        w = str(self.w) + 'f'
        w_inv = str(1.0 / self.w) + 'f'
        c = str(self.c) + 'f'
        code = f"""
            {tab4_str}{self.cuda_code_start_comments()}
        """
        if dtype == 'fp32':
            code += f"""
            {tab4_str}const float {sg_name}_x_abs = fabsf({x});
            float {y};
            if ({sg_name}_x_abs > {w})
            {curly_bracket_l}
                {y} = {c};
            {curly_bracket_r}
            else
            {curly_bracket_l}
                {y} = {w_inv};
            {curly_bracket_r}
            """
        elif dtype == 'fp16':
            code += f"""
            {tab4_str}const half2 {sg_name}_x_abs = __habs2({x});
            {tab4_str}const half2 {sg_name}_x_abs_ge_w = __hge2({sg_name}_x_abs, __float2half2_rn({w}));
            {tab4_str}half2 {y} = __hadd2(__hmul2(__float2half2_rn({c}),  {sg_name}_x_abs_ge_w), __hmul2(__hsub2(__float2half2_rn(1.0f), {sg_name}_x_abs_ge_w), __float2half2_rn({w_inv})));
            """
        else:
            raise NotImplementedError
        code += f"""
            {tab4_str}{self.cuda_code_end_comments()}
        """
        return code

    def cuda_codes(self, y: 'str', x: 'str', dtype: 'str'):
        return cfunction.piecewise_leaky_relu_backward(y=y, x=x, w=self.w, c=self.c, dtype=dtype)


class squarewave_fourier_series(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x: 'torch.Tensor', n: 'int', T_period: 'float'):
        if x.requires_grad:
            ctx.save_for_backward(x)
            ctx.n = n
            ctx.T_period = T_period
        return heaviside(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_x = 0.0
        x = ctx.saved_tensors[0]
        w = math.pi * 2.0 / ctx.T_period
        for i in range(1, ctx.n):
            grad_x += torch.cos_((2 * i - 1.0) * w * x)
        grad_x *= 4.0 / ctx.T_period
        grad_x *= grad_output
        return grad_x, None, None


class SquarewaveFourierSeries(MultiArgsSurrogateFunctionBase):

    def __init__(self, n: 'int'=2, T_period: 'float'=8, spiking=True):
        super().__init__(spiking)
        assert isinstance(n, int) and T_period > 0.0
        self.n = n
        self.T_period = T_period
        self.spiking = spiking

    def forward(self, x):
        if self.spiking:
            f = self.spiking_function
        else:
            f = self.primitive_function
        return f(x, self.n, self.T_period)

    @staticmethod
    def spiking_function(x: 'torch.Tensor', w, c):
        return squarewave_fourier_series.apply(x, w, c)

    @staticmethod
    def primitive_function(x: 'torch.Tensor', n: 'int', T_period: 'float'):
        w = math.pi * 2.0 / T_period
        ret = torch.zeros_like(x.data)
        for i in range(1, n):
            c = 2 * i - 1.0
            ret += torch.sin(c * w * x) / c
        return 0.5 + 2.0 / math.pi * ret

    def cuda_code(self, x: 'str', y: 'str', dtype='fp32'):
        sg_name = 'sg_' + self._get_name()
        w = str(self.w) + 'f'
        w_inv = str(1.0 / self.w) + 'f'
        c = str(self.c) + 'f'
        code = f"""
            {tab4_str}{self.cuda_code_start_comments()}
        """
        if dtype == 'fp32':
            raise NotImplementedError
        elif dtype == 'fp16':
            raise NotImplementedError
        else:
            raise NotImplementedError
        code += f"""
            {tab4_str}{self.cuda_code_end_comments()}
        """
        return code


class s2nn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x: 'torch.Tensor', alpha: 'float', beta: 'float'):
        if x.requires_grad:
            ctx.save_for_backward(x)
            ctx.alpha = alpha
            ctx.beta = beta
        return heaviside(x)

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_tensors[0]
        sgax = torch.sigmoid(ctx.alpha * x)
        grad_x = torch.where(x < 0.0, ctx.alpha * sgax * (1.0 - sgax), ctx.beta / (x + 1.0))
        return grad_x * grad_output, None, None


class S2NN(MultiArgsSurrogateFunctionBase):

    def __init__(self, alpha=4.0, beta=1.0, spiking=True):
        """
        * :ref:`API in English <S2NN.__init__-en>`
        .. _S2NN.__init__-cn:

        :param alpha: 控制 ``x < 0`` 时梯度的参数
        :param beta: 控制 ``x >= 0`` 时梯度的参数
        :param spiking: 是否输出脉冲，默认为 ``True``，在前向传播时使用 ``heaviside`` 而在反向传播使用替代梯度。若为 ``False``
            则不使用替代梯度，前向传播时，使用反向传播时的梯度替代函数对应的原函数

        `S2NN: Time Step Reduction of Spiking Surrogate Gradients for Training Energy Efficient Single-Step Neural Networks <https://arxiv.org/abs/2201.10879>`_ 提出的S2NN替代函数。反向传播为

        .. math::
            g'(x) = \\begin{cases}
                \\alpha * (1 - \\mathrm{sigmoid} (\\alpha x)) \\mathrm{sigmoid} (\\alpha x), x < 0 \\\\
                \\\\frac{beta}{(x + 1)}, x \\ge 0
            \\end{cases}

        对应的原函数为

        .. math::
            g(x) = \\begin{cases}
                \\mathrm{sigmoid} (\\alpha x), x < 0 \\\\
                \\beta \\mathrm{ln}(x + 1) + 1, x \\ge 0
            \\end{cases}

        .. image:: ../_static/API/activation_based/surrogate/S2NN.*
            :width: 100%


        * :ref:`中文API <S2NN.__init__-cn>`
        .. _S2NN.__init__-en:

        :param alpha: the param that controls the gradient when ``x < 0``
        :param beta: the param that controls the gradient when ``x >= 0``
        :param spiking: whether output spikes. The default is ``True`` which means that using ``heaviside`` in forward
            propagation and using surrogate gradient in backward propagation. If ``False``, in forward propagation,
            using the primitive function of the surrogate gradient function used in backward propagation

        The S2NN surrogate spiking function, which is proposed by `S2NN: Time Step Reduction of Spiking Surrogate Gradients for Training Energy Efficient Single-Step Neural Networks <https://arxiv.org/abs/2201.10879>`_. The gradient is defined by

        .. math::
            g'(x) = \\begin{cases}
                \\alpha * (1 - \\mathrm{sigmoid} (\\alpha x)) \\mathrm{sigmoid} (\\alpha x), x < 0 \\\\
                \\beta (x + 1), x \\ge 0
            \\end{cases}

        The primitive function is defined by

        .. math::
            g(x) = \\begin{cases}
                \\mathrm{sigmoid} (\\alpha x), x < 0 \\\\
                \\beta \\mathrm{ln}(x + 1) + 1, x \\ge 0
            \\end{cases}

        .. image:: ../_static/API/activation_based/surrogate/S2NN.*
            :width: 100%
        """
        super().__init__(spiking)
        self.alpha = alpha
        self.beta = beta
        self.spiking = spiking

    def forward(self, x):
        if self.spiking:
            f = self.spiking_function
        else:
            f = self.primitive_function
        return f(x, self.alpha, self.beta)

    @staticmethod
    def spiking_function(x: 'torch.Tensor', alpha, beta):
        return s2nn.apply(x, alpha, beta)

    @staticmethod
    def primitive_function(x: 'torch.Tensor', alpha: 'float', beta: 'float'):
        return torch.where(x < 0.0, torch.sigmoid(x * alpha), beta * torch.log((x + 1.0).abs_() + 1e-05) + 0.5)

    def cuda_code(self, x: 'str', y: 'str', dtype='fp32'):
        sg_name = 'sg_' + self._get_name()
        alpha = str(self.alpha) + 'f'
        beta = str(self.beta) + 'f'
        code = f"""
            {tab4_str}{self.cuda_code_start_comments()}
        """
        if dtype == 'fp32':
            code += f"""
            {tab4_str}const float {sg_name}_sigmoid_ax = 1.0f / (1.0f + expf(- {alpha} * {x}));
            {tab4_str}const float {sg_name}_mask_l = (float)({x} < 0.0f);
            {tab4_str}const float {y} = (1.0f - {sg_name}_sigmoid_ax) * {sg_name}_sigmoid_ax * {alpha} * {sg_name}_mask_l + {beta} / ({x} + 1.0f) * (1.0f - {sg_name}_mask_l);
            """
        elif dtype == 'fp16':
            code += f"""
            {tab4_str}const half2 {sg_name}_alpha = __float2half2_rn({alpha});
            {tab4_str}const half2 {sg_name}_sigmoid_ax = __h2div(__float2half2_rn(1.0f), __hadd2(h2exp(__hneg2(__hmul2({sg_name}_alpha, {x}))), __float2half2_rn(1.0f)));
            {tab4_str}const half2 {sg_name}_mask_l = __hlt2({x}, __float2half2_rn(0.0f));
            {tab4_str}const half2 {y} = __hadd2(__hmul2(__hmul2(__hmul2(__hsub2(__float2half2_rn(1.0f), {sg_name}_sigmoid_ax), {sg_name}_sigmoid_ax), {sg_name}_alpha), {sg_name}_mask_l), __hmul2(__h2div(__float2half2_rn({beta}), __hadd2({x}, __float2half2_rn(1.0f))), __hsub2(__float2half2_rn(1.0f), {sg_name}_mask_l)));
            """
        else:
            raise NotImplementedError
        code += f"""
            {tab4_str}{self.cuda_code_end_comments()}
        """
        return code

    def cuda_codes(self, y: 'str', x: 'str', dtype: 'str'):
        return cfunction.s2nn_backward(y=y, x=x, alpha=self.alpha, beta=self.beta, dtype=dtype)


class q_pseudo_spike(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, alpha):
        if x.requires_grad:
            ctx.save_for_backward(x)
            ctx.alpha = alpha
        return heaviside(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_x = None
        x = ctx.saved_tensors[0]
        if ctx.needs_input_grad[0]:
            grad_x = (1 + 2 / (ctx.alpha - 1) * x.abs()).pow_(-ctx.alpha) * grad_output
        return grad_x, None


class QPseudoSpike(SurrogateFunctionBase):

    def __init__(self, alpha=2.0, spiking=True):
        """
        * :ref:`API in English <QPseudoSpike.__init__-en>`
        .. _QPseudoSpike.__init__-cn:

        :param alpha: 控制反向传播时梯度函数尾部厚度的参数
        :param spiking: 是否输出脉冲，默认为 ``True``，在前向传播时使用 ``heaviside`` 而在反向传播使用替代梯度。若为 ``False``
            则不使用替代梯度，前向传播时，使用反向传播时的梯度替代函数对应的原函数

        `Surrogate Gradients Design <https://arxiv.org/abs/2202.00282>`_ 提出的 :math:`q`-PseudoSpike替代函数。反向传播为

        .. math::
            g'(x) = (1+\\frac{2|x|}{\\alpha-1})^{-\\alpha}

        其中 :math:`\\alpha>1` 对应原文中的 :math:`q`。

        对应的原函数为

        .. math::
            g(x) =
            \\begin{cases}
            \\frac{1}{2}(1-\\frac{2x}{\\alpha-1})^{1-\\alpha}, & x < 0 \\\\
            1 - \\frac{1}{2}(1+\\frac{2x}{\\alpha-1})^{1-\\alpha}, & x \\geq 0.
            \\end{cases}

        .. image:: ../_static/API/activation_based/surrogate/QPseudoSpike.*
            :width: 100%

        * :ref:`中文API <QPseudoSpike.__init__-cn>`
        .. _QPseudoSpike.__init__-en:

        :param alpha: parameter to control tail fatness of gradient
        :param spiking: whether output spikes. The default is ``True`` which means that using ``heaviside`` in forward
            propagation and using surrogate gradient in backward propagation. If ``False``, in forward propagation,
            using the primitive function of the surrogate gradient function used in backward propagation

        The :math:`q`-PseudoSpike surrogate spiking function, which is first proposed in `Surrogate Gradients Design <https://arxiv.org/abs/2202.00282>`_. The gradient is defined by

        .. math::
            g'(x) = (1+\\frac{2|x|}{\\alpha-1})^{-\\alpha}

        where :math:`\\alpha>1` corresponds to :math:`q` in paper.

        The primitive function is defined by

        .. math::
            g(x) =
            \\begin{cases}
            \\frac{1}{2}(1-\\frac{2x}{\\alpha-1})^{1-\\alpha}, & x < 0 \\\\
            1 - \\frac{1}{2}(1+\\frac{2x}{\\alpha-1})^{1-\\alpha}, & x \\geq 0.
            \\end{cases}

        .. image:: ../_static/API/activation_based/surrogate/QPseudoSpike.*
            :width: 100%
        """
        super().__init__(alpha, spiking)

    @staticmethod
    def spiking_function(x, alpha):
        return q_pseudo_spike.apply(x, alpha)

    @staticmethod
    def primitive_function(x: 'torch.Tensor', alpha: 'float'):
        mask_nonnegative = heaviside(x)
        mask_sign = mask_nonnegative * 2.0 - 1.0
        return mask_nonnegative - mask_sign * (0.5 * (1.0 + 2.0 / (alpha - 1.0) * x * mask_sign).pow_(1.0 - alpha))

    def cuda_code(self, x: 'str', y: 'str', dtype='fp32'):
        sg_name = 'sg_' + self._get_name()
        alpha = str(self.alpha) + 'f'
        code = f"""
            {tab4_str}{self.cuda_code_start_comments()}
        """
        if dtype == 'fp32':
            code += f"""
            {tab4_str}const float {sg_name}_base = 1.0f + 2.0f / ({alpha} - 1.0f) * fabsf({x});
            {tab4_str}const float {y} = powf({sg_name}_base, -{alpha});
            """
        elif dtype == 'fp16':
            code += f"""
            {tab4_str}const half2 {sg_name}_alpha = __float2half2_rn({alpha});
            {tab4_str}const half2 {sg_name}_base = __hadd2(__float2half2_rn(1.0f), __h2div(__hmul2(__float2half2_rn(2.0f), __habs2({x})), __hsub2({sg_name}_alpha, __float2half2_rn(1.0f))));
            {tab4_str}const half2 {y} = h2exp2(__hmul2(h2log2({sg_name}_base), __hneg2({sg_name}_alpha))); // Replace power with combination of log and exp, since CUDA has no power function for FP16.
            """
        else:
            raise NotImplementedError
        code += f"""
            {tab4_str}{self.cuda_code_end_comments()}
        """
        return code

    def cuda_codes(self, y: 'str', x: 'str', dtype: 'str'):
        return cfunction.q_pseudo_spike_backward(y=y, x=x, alpha=self.alpha, dtype=dtype)


@torch.jit.script
def leaky_k_relu_backward(grad_output: 'torch.Tensor', x: 'torch.Tensor', leak: 'float', k: 'float'):
    mask1 = x >= 0.0
    grad_x = mask1 * k + (1.0 - mask1) * leak
    return grad_output * grad_x, None, None


class leaky_k_relu(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, leak, k):
        if x.requires_grad:
            ctx.save_for_backward(x)
            ctx.leak = leak
            ctx.k = k
        return heaviside(x)

    @staticmethod
    def backward(ctx, grad_output):
        return leaky_k_relu_backward(grad_output, ctx.saved_tensors[0], ctx.leak, ctx.k)


class LeakyKReLU(MultiArgsSurrogateFunctionBase):

    def __init__(self, spiking=True, leak: 'float'=0.0, k: 'float'=1.0):
        """
        * :ref:`API in English <LeakyKReLU.__init__-en>`
        .. _LeakyKReLU.__init__-cn:

        :param spiking: whether output spikes. The default is ``True`` which means that using ``heaviside`` in forward
            propagation and using surrogate gradient in backward propagation. If ``False``, in forward propagation,
            using the primitive function of the surrogate gradient function used in backward propagation
        :type spiking: bool
        :param leak: gradient when ``x < 0``
        :type leak: float
        :param k: gradient when ``x >= 0 ``
        :type k: float

        反向传播时使用LeakyKReLU的梯度的脉冲发放函数。反向传播为

        .. math::
            g'(x) =
            \\begin{cases}
            k, & x \\geq 0 \\\\
            leak, & x < 0 \\\\
            \\end{cases}

        对应的原函数为

        .. math::
            g(x) =
            \\begin{cases}
            k \\cdot x, & x \\geq 0 \\\\
            leak \\cdot x, & x < 0 \\\\
            \\end{cases}

        .. image:: ../_static/API/activation_based/surrogate/LeakyKReLU.*
            :width: 100%

        * :ref:`中文API <LeakyKReLU.__init__-cn>`
        .. _LeakyKReLU.__init__-en:

        :param spiking: 是否输出脉冲，默认为 ``True``，在前向传播时使用 ``heaviside`` 而在反向传播使用替代梯度。若为 ``False``
            则不使用替代梯度，前向传播时，使用反向传播时的梯度替代函数对应的原函数
        :type spiking: bool
        :param leak: ``x < 0`` 时的梯度值
        :type leak: float
        :param k: ``x >= 0 `` 时的梯度值
        :type k: float

        The LeakyKReLU surrogate spiking function. The gradient is defined by

        .. math::
            g'(x) =
            \\begin{cases}
            k, & x \\geq 0 \\\\
            leak, & x < 0 \\\\
            \\end{cases}

        The primitive function is defined by

        .. math::
            g(x) =
            \\begin{cases}
            k \\cdot x, & x \\geq 0 \\\\
            leak \\cdot x, & x < 0 \\\\
            \\end{cases}

        .. image:: ../_static/API/activation_based/surrogate/LeakyKReLU.*
            :width: 100%

        """
        super().__init__(spiking, leak, k)
        self.leak = leak
        self.k = k

    @staticmethod
    def spiking_function(x, leak, k):
        return leaky_k_relu.apply(x, leak, k)

    @staticmethod
    @torch.jit.script
    def primitive_function(x: 'torch.Tensor', leak: 'float', k: 'float'):
        mask1 = x >= 0.0
        return (leak * (1.0 - mask1) + k * mask1) * x

    @staticmethod
    def backward(grad_output, x, leak, k):
        return leaky_k_relu_backward(grad_output, x, leak, k)[0]

    def forward(self, x):
        if self.spiking:
            f = self.spiking_function
        else:
            f = self.primitive_function
        return f(x, self.leak, self.k)

    def cuda_code(self, x: 'str', y: 'str', dtype='fp32'):
        sg_name = 'sg_' + self._get_name()
        leak = str(self.leak) + 'f'
        k = str(self.k) + 'f'
        code = f"""
            {tab4_str}{self.cuda_code_start_comments()}
        """
        if dtype == 'fp32':
            code += f"""
            {tab4_str}const float {sg_name}_mask1 = (float) ({x} >= 0.0f);
            {tab4_str}const float {y} = {leak} * (1.0f - {sg_name}_mask1) + {k} * {sg_name}_mask1;
            """
        elif dtype == 'fp16':
            code += f"""
            {tab4_str}const half2 {sg_name}_mask1 = __hgeu2({x}, __float2half2_rn(0.0f));
            {tab4_str}const half2 {y} = __hfma2(__float2half2_rn({k}), {sg_name}_mask1, __hmul2(__float2half2_rn({leak}), __hsub2(__float2half2_rn(1.0f), {sg_name}_mask1)));
            """
        else:
            raise NotImplementedError
        code += f"""
            {tab4_str}{self.cuda_code_end_comments()}
        """
        return code

    def cuda_codes(self, y: 'str', x: 'str', dtype: 'str'):
        return cfunction.leaky_k_relu_backward(y=y, x=x, leak=self.leak, k=self.k, dtype=dtype)


@torch.jit.script
def fake_numerical_gradient_backward(grad_output: 'torch.Tensor', x: 'torch.Tensor', alpha: 'float'):
    grad_x = torch.clamp_max(((x >= 0.0) * 2.0 - 1.0) / x, alpha)
    return grad_output * grad_x, None


class fake_numerical_gradient(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, alpha):
        if x.requires_grad:
            ctx.save_for_backward(x)
            ctx.alpha = alpha
        return heaviside(x)

    @staticmethod
    def backward(ctx, grad_output):
        return fake_numerical_gradient_backward(grad_output, ctx.saved_tensors[0], ctx.alpha)


class FakeNumericalGradient(SurrogateFunctionBase):

    def __init__(self, alpha=0.3):
        super().__init__(alpha, spiking=True)

    @staticmethod
    def spiking_function(x, alpha):
        return fake_numerical_gradient.apply(x, alpha)

    @staticmethod
    def backward(grad_output, x, alpha):
        return fake_numerical_gradient_backward(grad_output, x, alpha)[0]

    def cuda_code(self, x: 'str', y: 'str', dtype='fp32'):
        sg_name = 'sg_' + self._get_name()
        alpha = str(self.alpha) + 'f'
        code = f"""
            {tab4_str}{self.cuda_code_start_comments()}
        """
        if dtype == 'fp32':
            code += f"""
            {tab4_str}const float {sg_name}_sign = (float) ({x} >= 0.0f) * 2.0f - 1.0f;
            {tab4_str}const float {y} = min({sg_name}_sign / {x}, {alpha});
            """
        elif dtype == 'fp16':
            code += f"""
            {tab4_str}const half2 {sg_name}_sign = __hfma2(__hgeu2({x}, __float2half2_rn(0.0f)), __float2half2_rn(2.0f), __float2half2_rn(-1.0f));
            #if (__CUDA_ARCH__ < 800)
            {tab4_str}const half2 {sg_name}_grad_x = __h2div({sg_name}_sign, {x});
            {tab4_str}const half2 {sg_name}_grad_max = __float2half2_rn({alpha});
            {tab4_str}const half2 {y} = make_half2({sg_name}_grad_x.x <= {sg_name}_grad_max.x ? {sg_name}_grad_x.x : {sg_name}_grad_max.x, {sg_name}_grad_x.y <= {sg_name}_grad_max.y ? {sg_name}_grad_x.y : {sg_name}_grad_max.y);
            #else
            {tab4_str}const half2 {y} = __hmin2(__h2div({sg_name}_sign, {x}), __float2half2_rn({alpha}));
            #endif
            """
        else:
            raise NotImplementedError
        code += f"""
            {tab4_str}{self.cuda_code_end_comments()}
        """
        return code

    def cuda_codes(self, y: 'str', x: 'str', dtype: 'str'):
        return cfunction.fake_numerical_gradient_backward(y=y, x=x, alpha=self.alpha, dtype=dtype)


@torch.jit.script
def log_tailed_relu_backward(grad_output: 'torch.Tensor', x: 'torch.Tensor', alpha: 'float'):
    mask_gt1 = x > 1.0
    mask_le0 = x <= 0.0
    grad_x = torch.ones_like(grad_output)
    grad_x[mask_gt1] = 1.0 / x[mask_gt1]
    grad_x[mask_le0] = alpha
    return grad_output * grad_x, None


class log_tailed_relu(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, alpha):
        if x.requires_grad:
            ctx.save_for_backward(x)
            ctx.alpha = alpha
        return heaviside(x)

    @staticmethod
    def backward(ctx, grad_output):
        return log_tailed_relu_backward(grad_output, ctx.saved_tensors[0], ctx.alpha)


@torch.jit.script
def deterministic_pass_backward(grad_output: 'torch.Tensor', x: 'torch.Tensor', alpha: 'float'):
    return grad_output, None


class deterministic_pass(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, alpha):
        if x.requires_grad:
            ctx.save_for_backward(x)
            ctx.alpha = alpha
        return heaviside(x)

    @staticmethod
    def backward(ctx, grad_output):
        return deterministic_pass_backward(grad_output, ctx.saved_tensors[0], ctx.alpha)


class DeterministicPass(SurrogateFunctionBase):

    def __init__(self, alpha=1.0, spiking=True):
        super().__init__(alpha, spiking)

    @staticmethod
    def spiking_function(x, alpha):
        return deterministic_pass.apply(x, alpha)

    @staticmethod
    @torch.jit.script
    def primitive_function(x: 'torch.Tensor', alpha: 'float'):
        return x

    @staticmethod
    def backward(grad_output, x, alpha):
        return deterministic_pass_backward(grad_output, x, alpha)[0]


@torch.jit.script
def rect_backward(grad_output: 'torch.Tensor', x: 'torch.Tensor', alpha: 'float'):
    return alpha * (x.abs() < 0.5 / alpha) * grad_output, None


class rect(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, alpha):
        if x.requires_grad:
            ctx.save_for_backward(x)
            ctx.alpha = alpha
        return heaviside(x)

    @staticmethod
    def backward(ctx, grad_output):
        return rect_backward(grad_output, ctx.saved_tensors[0], ctx.alpha)


class Rect(SurrogateFunctionBase):

    def __init__(self, alpha=1.0, spiking=True):
        super().__init__(alpha, spiking)

    @staticmethod
    def spiking_function(x, alpha):
        return rect.apply(x, alpha)

    @staticmethod
    @torch.jit.script
    def primitive_function(x: 'torch.Tensor', alpha: 'float'):
        return torch.clamp(alpha * x + 0.5, min=0.0, max=1.0)

    @staticmethod
    def backward(grad_output, x, alpha):
        return rect_backward(grad_output, x, alpha)[0]


def random_temporal_delete(x_seq: 'Union[torch.Tensor, np.ndarray]', T_remain: 'int', batch_first):
    """
    :param x_seq: a sequence with `shape = [T, N, *]`, where `T` is the sequence length and `N` is the batch size
    :type x_seq: Union[torch.Tensor, np.ndarray]
    :param T_remain: the remained length
    :type T_remain: int
    :param batch_first: if `True`, `x_seq` will be regarded as `shape = [N, T, *]`
    :type batch_first: bool
    :return: the sequence with length `T_remain`, which is obtained by randomly removing `T - T_remain` slices
    :rtype: Union[torch.Tensor, np.ndarray]
    The random temporal delete data augmentation used in `Deep Residual Learning in Spiking Neural Networks <https://arxiv.org/abs/2102.04159>`_.
    Codes example:

    .. code-block:: python

        import torch
        from spikingjelly.datasets import random_temporal_delete
        T = 8
        T_remain = 5
        N = 4
        x_seq = torch.arange(0, N*T).view([N, T])
        print('x_seq=\\n', x_seq)
        print('random_temporal_delete(x_seq)=\\n', random_temporal_delete(x_seq, T_remain, batch_first=True))

    Outputs:

    .. code-block:: shell

        x_seq=
         tensor([[ 0,  1,  2,  3,  4,  5,  6,  7],
                [ 8,  9, 10, 11, 12, 13, 14, 15],
                [16, 17, 18, 19, 20, 21, 22, 23],
                [24, 25, 26, 27, 28, 29, 30, 31]])
        random_temporal_delete(x_seq)=
         tensor([[ 0,  1,  4,  6,  7],
                [ 8,  9, 12, 14, 15],
                [16, 17, 20, 22, 23],
                [24, 25, 28, 30, 31]])
    """
    if batch_first:
        sec_list = np.random.choice(x_seq.shape[1], T_remain, replace=False)
    else:
        sec_list = np.random.choice(x_seq.shape[0], T_remain, replace=False)
    sec_list.sort()
    if batch_first:
        return x_seq[:, sec_list]
    else:
        return x_seq[sec_list]


class RandomTemporalDelete(torch.nn.Module):

    def __init__(self, T_remain: 'int', batch_first: 'bool'):
        """
        :param T_remain: the remained length
        :type T_remain: int
        :type T_remain: int
        :param batch_first: if `True`, `x_seq` will be regarded as `shape = [N, T, *]`
        The random temporal delete data augmentation used in `Deep Residual Learning in Spiking Neural Networks <https://arxiv.org/abs/2102.04159>`_.
        Refer to :class:`random_temporal_delete` for more details.
        """
        super().__init__()
        self.T_remain = T_remain
        self.batch_first = batch_first

    def forward(self, x_seq: 'Union[torch.Tensor, np.ndarray]'):
        return random_temporal_delete(x_seq, self.T_remain, self.batch_first)


class Tempotron(nn.Module):

    def __init__(self, in_features, out_features, T, tau=15.0, tau_s=15.0 / 4, v_threshold=1.0):
        """
        :param in_features: 输入数量，含义与nn.Linear的in_features参数相同
        :param out_features: 输出数量，含义与nn.Linear的out_features参数相同
        :param T: 仿真周期
        :param tau: LIF神经元的积分时间常数
        :param tau_s: 突触上的电流的衰减时间常数
        :param v_threshold: 阈值电压

        Gutig R, Sompolinsky H. The tempotron: a neuron that learns spike timing–based decisions[J]. Nature         Neuroscience, 2006, 9(3): 420-428. 中提出的Tempotron模型

        """
        super().__init__()
        self.tau = tau
        self.tau_s = tau_s
        self.T = T
        self.v_threshold = v_threshold
        self.fc = nn.Linear(in_features, out_features, bias=False)
        t_max = tau * tau_s * math.log(tau / tau_s) / (tau - tau_s)
        self.v0 = self.v_threshold / (math.exp(-t_max / tau) - math.exp(-t_max / tau_s))

    @staticmethod
    def psp_kernel(t: 'torch.Tensor', tau, tau_s):
        """
        :param t: 表示时刻的tensor
        :param tau: LIF神经元的积分时间常数
        :param tau_s: 突触上的电流的衰减时间常数
        :return: t时刻突触后的LIF神经元的电压值
        """
        return (torch.exp(-t / tau) - torch.exp(-t / tau_s)) * (t >= 0).float()

    @staticmethod
    def mse_loss(v_max, v_threshold, label, num_classes):
        """
        :param v_max: Tempotron神经元在仿真周期内输出的最大电压值，与forward函数在ret_type == 'v_max'时的返回值相        同。shape=[batch_size, out_features]的tensor
        :param v_threshold: Tempotron的阈值电压，float或shape=[batch_size, out_features]的tensor
        :param label: 样本的真实标签，shape=[batch_size]的tensor
        :param num_classes: 样本的类别总数，int
        :return: 分类错误的神经元的电压，与阈值电压之差的均方误差
        """
        wrong_mask = ((v_max >= v_threshold).float() != F.one_hot(label, num_classes)).float()
        return torch.sum(torch.pow((v_max - v_threshold) * wrong_mask, 2)) / label.shape[0]

    def forward(self, in_spikes: 'torch.Tensor', ret_type):
        """
        :param in_spikes: shape=[batch_size, in_features]

        in_spikes[:, i]表示第i个输入脉冲的脉冲发放时刻，介于0到T之间，T是仿真时长

        in_spikes[:, i] < 0则表示无脉冲发放
        :param ret_type: 返回值的类项，可以为'v','v_max','spikes'
        :return:

        ret_type == 'v': 返回一个shape=[batch_size, out_features, T]的tensor，表示out_features个Tempotron神经元在仿真时长T        内的电压值

        ret_type == 'v_max': 返回一个shape=[batch_size, out_features]的tensor，表示out_features个Tempotron神经元在仿真时长T        内的峰值电压

        ret_type == 'spikes': 返回一个out_spikes，shape=[batch_size, out_features]的tensor，表示out_features个Tempotron神        经元的脉冲发放时刻，out_spikes[:, i]表示第i个输出脉冲的脉冲发放时刻，介于0到T之间，T是仿真时长。out_spikes[:, i] < 0        表示无脉冲发放
        """
        t = torch.arange(0, self.T)
        t = t.view(1, 1, t.shape[0]).repeat(in_spikes.shape[0], in_spikes.shape[1], 1)
        in_spikes = in_spikes.unsqueeze(-1).repeat(1, 1, self.T)
        v_in = self.v0 * self.psp_kernel(t - in_spikes, self.tau, self.tau_s) * (in_spikes >= 0).float()
        v_out = self.fc(v_in.permute(0, 2, 1)).permute(0, 2, 1)
        if ret_type == 'v':
            return v_out
        elif ret_type == 'v_max':
            return F.max_pool1d(v_out, kernel_size=self.T).squeeze()
        elif ret_type == 'spikes':
            max_index = v_out.argmax(dim=2)
            t = torch.arange(0, self.T)
            t = t.view(1, 1, t.shape[0]).repeat(in_spikes.shape[0], v_out.shape[1], 1)
            max_index_soft = (F.softmax(v_out * self.T, dim=2) * t).sum(dim=2)
            v_max = F.max_pool1d(v_out, kernel_size=self.T).squeeze()
            mask = (v_max >= self.v_threshold).float() * 2 - 1
            max_index = max_index * mask
            max_index_soft = max_index_soft * mask
            return max_index_soft + (max_index - max_index_soft).detach()
        else:
            raise ValueError


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (ATan,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (DSpike,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (DeterministicPass,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Erf,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (FakeNumericalGradient,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (GradwithTrace,
     lambda: ([], {'module': torch.nn.ReLU()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Identity,
     lambda: ([], {'C_in': 4, 'C_out': 4, 'signal': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (LIFWrapper,
     lambda: ([], {'module': torch.nn.ReLU()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (LeakyKReLU,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (MemoryModule,
     lambda: ([], {}),
     lambda: ([], {'x': 4})),
    (Nearest,
     lambda: ([], {'shape': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (NonzeroSignLogAbs,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (OTTTSequential,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (PiecewiseExp,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (PiecewiseLeakyReLU,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (PiecewiseQuadratic,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (PrintShapeModule,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (QPseudoSpike,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (RandomTemporalDelete,
     lambda: ([], {'T_remain': 4, 'batch_first': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Rect,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (S2NN,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Scale,
     lambda: ([], {'scale': 1.0}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ScaleLayer,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Sigmoid,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (SpikeTraceOp,
     lambda: ([], {'module': torch.nn.ReLU()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (SquarewaveFourierSeries,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (SuperSpike,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (VoltageHook,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (VoltageScaler,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (WeightedMSELoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
]

