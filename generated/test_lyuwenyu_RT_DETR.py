
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


import torch.utils.data as data


import torchvision


import torchvision.transforms as T


import torchvision.transforms.functional as F


import time


from collections import namedtuple


from collections import OrderedDict


import numpy as np


import torch.nn as nn


import math


from copy import deepcopy


from torch.utils.data import Dataset


from torch.utils.data import DataLoader


from torch.optim import Optimizer


from torch.optim.lr_scheduler import LRScheduler


from torch.cuda.amp.grad_scaler import GradScaler


from typing import Callable


from typing import List


from typing import Dict


import re


import copy


import torch.utils.data


from typing import Optional


from torch import Tensor


import torchvision.transforms.v2 as T


import torchvision.transforms.v2.functional as F


from typing import Any


import random


import torch.distributed


import torch.distributed as tdist


from torch.nn.parallel import DistributedDataParallel as DDP


from torch.utils.data import DistributedSampler


from torch.utils.data.dataloader import DataLoader


from collections import defaultdict


from collections import deque


import logging


from torch import nn


import torch.utils.model_zoo as model_zoo


import torch.nn.functional as F


import torch.cuda.amp as amp


import torch.optim as optim


import torch.optim.lr_scheduler as lr_scheduler


from typing import Iterable


import torch.amp


from torchvision.ops.boxes import box_area


from scipy.optimize import linear_sum_assignment


import torch.nn.init as init


from torch.cuda.amp import autocast


import collections


from torch.utils.tensorboard import SummaryWriter


from torch.utils.data import default_collate


import torchvision.transforms.v2 as VT


from torchvision.transforms.v2 import functional as VF


from torchvision.transforms.v2 import InterpolationMode


from functools import partial


import torchvision.transforms.functional as TVF


from typing import Tuple


import torch.backends.cudnn


from torch.nn.parallel import DataParallel as DP


from torch.distributed.fsdp import FullyShardedDataParallel as FSDP


import warnings


from torchvision.models.feature_extraction import get_graph_node_names


from torchvision.models.feature_extraction import create_feature_extractor


from enum import Enum


import functools


class YOLO(torch.nn.Module):
    __inject__ = ['backbone', 'neck', 'head']

    def __init__(self, backbone: 'torch.nn.Module', neck, head):
        super().__init__()
        self.backbone = backbone
        self.neck = neck
        self.head = head

    def forward(self, x, **kwargs):
        x = self.backbone(x)
        x = self.neck(x)
        x = self.head(x)
        return x

    def deploy(self):
        self.eval()
        for m in self.modules():
            if m is not self and hasattr(m, 'deploy'):
                m.deploy()
        return self


class YOLOv8(torch.nn.Module):

    def __init__(self, name) ->None:
        super().__init__()
        model = YOLO(f'{name}.pt')
        self.model = model.model

    def forward(self, x):
        """https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/tasks.py#L216
        """
        pred: 'torch.Tensor' = self.model(x)[0]
        pred = pred.permute(0, 2, 1)
        boxes, scores = pred.split([4, 80], dim=-1)
        boxes = torchvision.ops.box_convert(boxes, in_fmt='cxcywh', out_fmt='xyxy')
        return boxes, scores


class Classification(torch.nn.Module):
    __inject__ = ['backbone', 'head']

    def __init__(self, backbone: 'nn.Module', head: 'nn.Module'=None):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x):
        x = self.backbone(x)
        if self.head is not None:
            x = self.head(x)
        return x


class ClassHead(nn.Module):

    def __init__(self, hidden_dim, num_classes):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = x[0] if isinstance(x, (list, tuple)) else x
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.proj(x)
        return x


def get_activation(act: 'str', inpace: 'bool'=True):
    """get activation
    """
    if act is None:
        return nn.Identity()
    elif isinstance(act, nn.Module):
        return act
    act = act.lower()
    if act == 'silu' or act == 'swish':
        m = nn.SiLU()
    elif act == 'relu':
        m = nn.ReLU()
    elif act == 'leaky_relu':
        m = nn.LeakyReLU()
    elif act == 'silu':
        m = nn.SiLU()
    elif act == 'gelu':
        m = nn.GELU()
    elif act == 'hardsigmoid':
        m = nn.Hardsigmoid()
    else:
        raise RuntimeError('')
    if hasattr(m, 'inplace'):
        m.inplace = inpace
    return m


class ConvNormLayer(nn.Module):

    def __init__(self, ch_in, ch_out, kernel_size, stride, padding=None, bias=False, act=None):
        super().__init__()
        self.conv = nn.Conv2d(ch_in, ch_out, kernel_size, stride, padding=(kernel_size - 1) // 2 if padding is None else padding, bias=bias)
        self.norm = nn.BatchNorm2d(ch_out)
        self.act = nn.Identity() if act is None else get_activation(act)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class FrozenBatchNorm2d(nn.Module):
    """copy and modified from https://github.com/facebookresearch/detr/blob/master/models/backbone.py
    BatchNorm2d where the batch statistics and the affine parameters are fixed.
    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, num_features, eps=1e-05):
        super(FrozenBatchNorm2d, self).__init__()
        n = num_features
        self.register_buffer('weight', torch.ones(n))
        self.register_buffer('bias', torch.zeros(n))
        self.register_buffer('running_mean', torch.zeros(n))
        self.register_buffer('running_var', torch.ones(n))
        self.eps = eps
        self.num_features = n

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]
        super(FrozenBatchNorm2d, self)._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        scale = w * (rv + self.eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias

    def extra_repr(self):
        return '{num_features}, eps={eps}'.format(**self.__dict__)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(self.expansion * planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


def autopad(k, p=None):
    if p is None:
        p = k // 2 if isinstance(k, int) else [(x // 2) for x in k]
    return p


class Conv(nn.Module):

    def __init__(self, cin, cout, k=1, s=1, p=None, g=1, act='silu') ->None:
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(cout)
        self.act = get_activation(act, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):

    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5, act='silu'):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1, act=act)
        self.cv2 = Conv(c_, c2, 3, 1, g=g, act=act)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


BN_MOMENTUM = 0.1


class BottleneckX(nn.Module):
    expansion = 2
    cardinality = 32

    def __init__(self, inplanes, planes, stride=1, dilation=1):
        super(BottleneckX, self).__init__()
        cardinality = BottleneckX.cardinality
        bottle_planes = planes * cardinality // 32
        self.conv1 = nn.Conv2d(inplanes, bottle_planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(bottle_planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(bottle_planes, bottle_planes, kernel_size=3, stride=stride, padding=dilation, bias=False, dilation=dilation, groups=cardinality)
        self.bn2 = nn.BatchNorm2d(bottle_planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(bottle_planes, planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x, residual=None):
        if residual is None:
            residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out += residual
        out = self.relu(out)
        return out


class Root(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, residual):
        super(Root, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1, stride=1, bias=False, padding=(kernel_size - 1) // 2)
        self.bn = nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.residual = residual

    def forward(self, *x):
        children = x
        x = self.conv(torch.cat(x, 1))
        x = self.bn(x)
        if self.residual:
            x += children[0]
        x = self.relu(x)
        return x


class Tree(nn.Module):

    def __init__(self, levels, block, in_channels, out_channels, stride=1, level_root=False, root_dim=0, root_kernel_size=1, dilation=1, root_residual=False):
        super(Tree, self).__init__()
        if root_dim == 0:
            root_dim = 2 * out_channels
        if level_root:
            root_dim += in_channels
        if levels == 1:
            self.tree1 = block(in_channels, out_channels, stride, dilation=dilation)
            self.tree2 = block(out_channels, out_channels, 1, dilation=dilation)
        else:
            self.tree1 = Tree(levels - 1, block, in_channels, out_channels, stride, root_dim=0, root_kernel_size=root_kernel_size, dilation=dilation, root_residual=root_residual)
            self.tree2 = Tree(levels - 1, block, out_channels, out_channels, root_dim=root_dim + out_channels, root_kernel_size=root_kernel_size, dilation=dilation, root_residual=root_residual)
        if levels == 1:
            self.root = Root(root_dim, out_channels, root_kernel_size, root_residual)
        self.level_root = level_root
        self.root_dim = root_dim
        self.downsample = None
        self.project = None
        self.levels = levels
        if stride > 1:
            self.downsample = nn.MaxPool2d(stride, stride=stride)
        if levels == 1 and in_channels != out_channels:
            self.project = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False), nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM))

    def forward(self, x, residual=None, children=None):
        children = [] if children is None else children
        bottom = self.downsample(x) if self.downsample else x
        residual = self.project(bottom) if self.project else bottom
        if self.level_root:
            children.append(bottom)
        x1 = self.tree1(x, residual)
        if self.levels == 1:
            x2 = self.tree2(x1)
            x = self.root(x2, x1, *children)
        else:
            children.append(x1)
            x = self.tree2(x1, children=children)
        return x


def get_model_url(data='imagenet', name='dla34', hash='ba72cf86'):
    return join('http://dl.yf.io/dla/models', data, '{}-{}.pth'.format(name, hash))


class DLA(nn.Module):

    def __init__(self, levels, channels, num_classes=1000, block=BasicBlock, out_indices=(2, 3, 4, 5), residual_root=False, linear_root=False):
        super(DLA, self).__init__()
        self.channels = channels
        self.num_classes = num_classes
        self.out_indices = out_indices
        self.base_layer = nn.Sequential(nn.Conv2d(3, channels[0], kernel_size=7, stride=1, padding=3, bias=False), nn.BatchNorm2d(channels[0], momentum=BN_MOMENTUM), nn.ReLU(inplace=True))
        self.level0 = self._make_conv_level(channels[0], channels[0], levels[0])
        self.level1 = self._make_conv_level(channels[0], channels[1], levels[1], stride=2)
        self.level2 = Tree(levels[2], block, channels[1], channels[2], 2, level_root=False, root_residual=residual_root)
        self.level3 = Tree(levels[3], block, channels[2], channels[3], 2, level_root=True, root_residual=residual_root)
        self.level4 = Tree(levels[4], block, channels[3], channels[4], 2, level_root=True, root_residual=residual_root)
        self.level5 = Tree(levels[5], block, channels[4], channels[5], 2, level_root=True, root_residual=residual_root)

    def _make_level(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(nn.MaxPool2d(stride, stride=stride), nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False), nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
        layers = []
        layers.append(block(inplanes, planes, stride, downsample=downsample))
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))
        return nn.Sequential(*layers)

    def _make_conv_level(self, inplanes, planes, convs, stride=1, dilation=1):
        modules = []
        for i in range(convs):
            modules.extend([nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride if i == 0 else 1, padding=dilation, bias=False, dilation=dilation), nn.BatchNorm2d(planes, momentum=BN_MOMENTUM), nn.ReLU(inplace=True)])
            inplanes = planes
        return nn.Sequential(*modules)

    def forward(self, x):
        y = []
        x = self.base_layer(x)
        for i in range(6):
            x = getattr(self, 'level{}'.format(i))(x)
            if i in self.out_indices:
                y.append(x)
        return y

    def load_pretrained_model(self, data='imagenet', name='dla34', hash='ba72cf86'):
        if name.endswith('.pth'):
            model_weights = torch.load(data + name)
        else:
            model_url = get_model_url(data, name, hash)
            model_weights = model_zoo.load_url(model_url)
        self.load_state_dict(model_weights, strict=False)


class DLANet(nn.Module):

    def __init__(self, dla='dla34', pretrained=True, levels=[1, 1, 1, 2, 2, 1], in_channels=[16, 32, 64, 128, 256, 512], return_index=[1, 2, 3], cfg=None):
        super(DLANet, self).__init__()
        self.cfg = cfg
        self.in_channels = in_channels
        self.model = eval(dla)(pretrained=pretrained, levels=levels, in_channels=in_channels)
        self.return_index = return_index

    def forward(self, x):
        x = self.model(x)
        max_list = max(self.return_index)
        min_list = min(self.return_index)
        return x[min_list:max_list + 1]


class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class BottleNeck(nn.Module):
    expansion = 4

    def __init__(self, ch_in, ch_out, stride, shortcut, act='relu', variant='b'):
        super().__init__()
        if variant == 'a':
            stride1, stride2 = stride, 1
        else:
            stride1, stride2 = 1, stride
        width = ch_out
        self.branch2a = ConvNormLayer(ch_in, width, 1, stride1, act=act)
        self.branch2b = ConvNormLayer(width, width, 3, stride2, act=act)
        self.branch2c = ConvNormLayer(width, ch_out * self.expansion, 1, 1)
        self.shortcut = shortcut
        if not shortcut:
            if variant == 'd' and stride == 2:
                self.short = nn.Sequential(OrderedDict([('pool', nn.AvgPool2d(2, 2, 0, ceil_mode=True)), ('conv', ConvNormLayer(ch_in, ch_out * self.expansion, 1, 1))]))
            else:
                self.short = ConvNormLayer(ch_in, ch_out * self.expansion, 1, stride)
        self.act = nn.Identity() if act is None else get_activation(act)

    def forward(self, x):
        out = self.branch2a(x)
        out = self.branch2b(out)
        out = self.branch2c(out)
        if self.shortcut:
            short = x
        else:
            short = self.short(x)
        out = out + short
        out = self.act(out)
        return out


class Blocks(nn.Module):

    def __init__(self, block, ch_in, ch_out, count, stage_num, act='relu', variant='b'):
        super().__init__()
        self.blocks = nn.ModuleList()
        for i in range(count):
            self.blocks.append(block(ch_in, ch_out, stride=2 if i == 0 and stage_num != 2 else 1, shortcut=False if i == 0 else True, variant=variant, act=act))
            if i == 0:
                ch_in = ch_out * block.expansion

    def forward(self, x):
        out = x
        for block in self.blocks:
            out = block(out)
        return out


ResNet_cfg = {(18): [2, 2, 2, 2], (34): [3, 4, 6, 3], (50): [3, 4, 6, 3], (101): [3, 4, 23, 3]}


donwload_url = {(18): 'https://github.com/lyuwenyu/storage/releases/download/v0.1/ResNet18_vd_pretrained_from_paddle.pth', (34): 'https://github.com/lyuwenyu/storage/releases/download/v0.1/ResNet34_vd_pretrained_from_paddle.pth', (50): 'https://github.com/lyuwenyu/storage/releases/download/v0.1/ResNet50_vd_ssld_v2_pretrained_from_paddle.pth', (101): 'https://github.com/lyuwenyu/storage/releases/download/v0.1/ResNet101_vd_ssld_pretrained_from_paddle.pth'}


class PResNet(nn.Module):

    def __init__(self, depth, variant='d', num_stages=4, return_idx=[0, 1, 2, 3], act='relu', freeze_at=-1, freeze_norm=True, pretrained=False):
        super().__init__()
        block_nums = ResNet_cfg[depth]
        ch_in = 64
        if variant in ['c', 'd']:
            conv_def = [[3, ch_in // 2, 3, 2, 'conv1_1'], [ch_in // 2, ch_in // 2, 3, 1, 'conv1_2'], [ch_in // 2, ch_in, 3, 1, 'conv1_3']]
        else:
            conv_def = [[3, ch_in, 7, 2, 'conv1_1']]
        self.conv1 = nn.Sequential(OrderedDict([(name, ConvNormLayer(cin, cout, k, s, act=act)) for cin, cout, k, s, name in conv_def]))
        ch_out_list = [64, 128, 256, 512]
        block = BottleNeck if depth >= 50 else BasicBlock
        _out_channels = [(block.expansion * v) for v in ch_out_list]
        _out_strides = [4, 8, 16, 32]
        self.res_layers = nn.ModuleList()
        for i in range(num_stages):
            stage_num = i + 2
            self.res_layers.append(Blocks(block, ch_in, ch_out_list[i], block_nums[i], stage_num, act=act, variant=variant))
            ch_in = _out_channels[i]
        self.return_idx = return_idx
        self.out_channels = [_out_channels[_i] for _i in return_idx]
        self.out_strides = [_out_strides[_i] for _i in return_idx]
        if freeze_at >= 0:
            self._freeze_parameters(self.conv1)
            for i in range(min(freeze_at, num_stages)):
                self._freeze_parameters(self.res_layers[i])
        if freeze_norm:
            self._freeze_norm(self)
        if pretrained:
            if isinstance(pretrained, bool) or 'http' in pretrained:
                state = torch.hub.load_state_dict_from_url(donwload_url[depth], map_location='cpu')
            else:
                state = torch.load(pretrained, map_location='cpu')
            self.load_state_dict(state)
            None

    def _freeze_parameters(self, m: 'nn.Module'):
        for p in m.parameters():
            p.requires_grad = False

    def _freeze_norm(self, m: 'nn.Module'):
        if isinstance(m, nn.BatchNorm2d):
            m = FrozenBatchNorm2d(m.num_features)
        else:
            for name, child in m.named_children():
                _child = self._freeze_norm(child)
                if _child is not child:
                    setattr(m, name, _child)
        return m

    def forward(self, x):
        conv1 = self.conv1(x)
        x = F.max_pool2d(conv1, kernel_size=3, stride=2, padding=1)
        outs = []
        for idx, stage in enumerate(self.res_layers):
            x = stage(x)
            if idx in self.return_idx:
                outs.append(x)
        return outs


class RegNet(nn.Module):

    def __init__(self, configuration, return_idx=[0, 1, 2, 3]):
        super(RegNet, self).__init__()
        self.model = RegNetModel.from_pretrained('facebook/regnet-y-040')
        self.return_idx = return_idx

    def forward(self, x):
        outputs = self.model(x, output_hidden_states=True)
        x = outputs.hidden_states[2:5]
        return x


class _ResNet(nn.Module):

    def __init__(self, block, num_blocks, num_classes=10):
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class MResNet(nn.Module):

    def __init__(self, num_classes=10, num_blocks=[2, 2, 2, 2]) ->None:
        super().__init__()
        self.model = _ResNet(BasicBlock, num_blocks, num_classes)

    def forward(self, x):
        return self.model(x)


class IntermediateLayerGetter(nn.ModuleDict):
    """
    Module wrapper that returns intermediate layers from a model

    It has a strong assumption that the modules have been registered
    into the model in the same order as they are used.
    This means that one should **not** reuse the same nn.Module
    twice in the forward if you want this to work.

    Additionally, it is only able to query submodules that are directly
    assigned to the model. So if `model` is passed, `model.feature1` can
    be returned, but not `model.feature1.layer2`.
    """
    _version = 3

    def __init__(self, model: 'nn.Module', return_layers: 'List[str]') ->None:
        if not set(return_layers).issubset([name for name, _ in model.named_children()]):
            raise ValueError('return_layers are not present in model. {}'.format([name for name, _ in model.named_children()]))
        orig_return_layers = return_layers
        return_layers = {str(k): str(k) for k in return_layers}
        layers = OrderedDict()
        for name, module in model.named_children():
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break
        super().__init__(layers)
        self.return_layers = orig_return_layers

    def forward(self, x):
        outputs = []
        for name, module in self.items():
            x = module(x)
            if name in self.return_layers:
                outputs.append(x)
        return outputs


class ExponentialMovingAverage(torch.optim.swa_utils.AveragedModel):
    """Maintains moving averages of model parameters using an exponential decay.
    ``ema_avg = decay * avg_model_param + (1 - decay) * model_param``
    `torch.optim.swa_utils.AveragedModel <https://pytorch.org/docs/stable/optim.html#custom-averaging-strategies>`_
    is used to compute the EMA.
    """

    def __init__(self, model, decay, device='cpu', use_buffers=True):
        self.decay_fn = lambda x: decay * (1 - math.exp(-x / 2000))

        def ema_avg(avg_model_param, model_param, num_averaged):
            decay = self.decay_fn(num_averaged)
            return decay * avg_model_param + (1 - decay) * model_param
        super().__init__(model, device, ema_avg, use_buffers=use_buffers)


class RepVggBlock(nn.Module):

    def __init__(self, ch_in, ch_out, act='relu'):
        super().__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.conv1 = ConvNormLayer(ch_in, ch_out, 3, 1, padding=1, act=None)
        self.conv2 = ConvNormLayer(ch_in, ch_out, 1, 1, padding=0, act=None)
        self.act = nn.Identity() if act is None else get_activation(act)

    def forward(self, x):
        if hasattr(self, 'conv'):
            y = self.conv(x)
        else:
            y = self.conv1(x) + self.conv2(x)
        return self.act(y)

    def convert_to_deploy(self):
        if not hasattr(self, 'conv'):
            self.conv = nn.Conv2d(self.ch_in, self.ch_out, 3, 1, padding=1)
        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv.weight.data = kernel
        self.conv.bias.data = bias

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1), bias3x3 + bias1x1

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return F.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch: 'ConvNormLayer'):
        if branch is None:
            return 0, 0
        kernel = branch.conv.weight
        running_mean = branch.norm.running_mean
        running_var = branch.norm.running_var
        gamma = branch.norm.weight
        beta = branch.norm.bias
        eps = branch.norm.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std


class CSPRepLayer(nn.Module):

    def __init__(self, in_channels, out_channels, num_blocks=3, expansion=1.0, bias=None, act='silu'):
        super(CSPRepLayer, self).__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = ConvNormLayer(in_channels, hidden_channels, 1, 1, bias=bias, act=act)
        self.conv2 = ConvNormLayer(in_channels, hidden_channels, 1, 1, bias=bias, act=act)
        self.bottlenecks = nn.Sequential(*[RepVggBlock(hidden_channels, hidden_channels, act=act) for _ in range(num_blocks)])
        if hidden_channels != out_channels:
            self.conv3 = ConvNormLayer(hidden_channels, out_channels, 1, 1, bias=bias, act=act)
        else:
            self.conv3 = nn.Identity()

    def forward(self, x):
        x_1 = self.conv1(x)
        x_1 = self.bottlenecks(x_1)
        x_2 = self.conv2(x)
        return self.conv3(x_1 + x_2)


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation='relu', normalize_before=False):
        super().__init__()
        self.normalize_before = normalize_before
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = get_activation(activation)

    @staticmethod
    def with_pos_embed(tensor, pos_embed):
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(self, src, src_mask=None, pos_embed=None) ->torch.Tensor:
        residual = src
        if self.normalize_before:
            src = self.norm1(src)
        q = k = self.with_pos_embed(src, pos_embed)
        src, _ = self.self_attn(q, k, value=src, attn_mask=src_mask)
        src = residual + self.dropout1(src)
        if not self.normalize_before:
            src = self.norm1(src)
        residual = src
        if self.normalize_before:
            src = self.norm2(src)
        src = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = residual + self.dropout2(src)
        if not self.normalize_before:
            src = self.norm2(src)
        return src


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, src_mask=None, pos_embed=None) ->torch.Tensor:
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=src_mask, pos_embed=pos_embed)
        if self.norm is not None:
            output = self.norm(output)
        return output


class HybridEncoder(nn.Module):
    __share__ = ['eval_spatial_size']

    def __init__(self, in_channels=[512, 1024, 2048], feat_strides=[8, 16, 32], hidden_dim=256, nhead=8, dim_feedforward=1024, dropout=0.0, enc_act='gelu', use_encoder_idx=[2], num_encoder_layers=1, pe_temperature=10000, expansion=1.0, depth_mult=1.0, act='silu', eval_spatial_size=None, version='v2'):
        super().__init__()
        self.in_channels = in_channels
        self.feat_strides = feat_strides
        self.hidden_dim = hidden_dim
        self.use_encoder_idx = use_encoder_idx
        self.num_encoder_layers = num_encoder_layers
        self.pe_temperature = pe_temperature
        self.eval_spatial_size = eval_spatial_size
        self.out_channels = [hidden_dim for _ in range(len(in_channels))]
        self.out_strides = feat_strides
        self.input_proj = nn.ModuleList()
        for in_channel in in_channels:
            if version == 'v1':
                proj = nn.Sequential(nn.Conv2d(in_channel, hidden_dim, kernel_size=1, bias=False), nn.BatchNorm2d(hidden_dim))
            elif version == 'v2':
                proj = nn.Sequential(OrderedDict([('conv', nn.Conv2d(in_channel, hidden_dim, kernel_size=1, bias=False)), ('norm', nn.BatchNorm2d(hidden_dim))]))
            else:
                raise AttributeError()
            self.input_proj.append(proj)
        encoder_layer = TransformerEncoderLayer(hidden_dim, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, activation=enc_act)
        self.encoder = nn.ModuleList([TransformerEncoder(copy.deepcopy(encoder_layer), num_encoder_layers) for _ in range(len(use_encoder_idx))])
        self.lateral_convs = nn.ModuleList()
        self.fpn_blocks = nn.ModuleList()
        for _ in range(len(in_channels) - 1, 0, -1):
            self.lateral_convs.append(ConvNormLayer(hidden_dim, hidden_dim, 1, 1, act=act))
            self.fpn_blocks.append(CSPRepLayer(hidden_dim * 2, hidden_dim, round(3 * depth_mult), act=act, expansion=expansion))
        self.downsample_convs = nn.ModuleList()
        self.pan_blocks = nn.ModuleList()
        for _ in range(len(in_channels) - 1):
            self.downsample_convs.append(ConvNormLayer(hidden_dim, hidden_dim, 3, 2, act=act))
            self.pan_blocks.append(CSPRepLayer(hidden_dim * 2, hidden_dim, round(3 * depth_mult), act=act, expansion=expansion))
        self._reset_parameters()

    def _reset_parameters(self):
        if self.eval_spatial_size:
            for idx in self.use_encoder_idx:
                stride = self.feat_strides[idx]
                pos_embed = self.build_2d_sincos_position_embedding(self.eval_spatial_size[1] // stride, self.eval_spatial_size[0] // stride, self.hidden_dim, self.pe_temperature)
                setattr(self, f'pos_embed{idx}', pos_embed)

    @staticmethod
    def build_2d_sincos_position_embedding(w, h, embed_dim=256, temperature=10000.0):
        """
        """
        grid_w = torch.arange(int(w), dtype=torch.float32)
        grid_h = torch.arange(int(h), dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing='ij')
        assert embed_dim % 4 == 0, 'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1.0 / temperature ** omega
        out_w = grid_w.flatten()[..., None] @ omega[None]
        out_h = grid_h.flatten()[..., None] @ omega[None]
        return torch.concat([out_w.sin(), out_w.cos(), out_h.sin(), out_h.cos()], dim=1)[None, :, :]

    def forward(self, feats):
        assert len(feats) == len(self.in_channels)
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]
        if self.num_encoder_layers > 0:
            for i, enc_ind in enumerate(self.use_encoder_idx):
                h, w = proj_feats[enc_ind].shape[2:]
                src_flatten = proj_feats[enc_ind].flatten(2).permute(0, 2, 1)
                if self.training or self.eval_spatial_size is None:
                    pos_embed = self.build_2d_sincos_position_embedding(w, h, self.hidden_dim, self.pe_temperature)
                else:
                    pos_embed = getattr(self, f'pos_embed{enc_ind}', None)
                memory: 'torch.Tensor' = self.encoder[i](src_flatten, pos_embed=pos_embed)
                proj_feats[enc_ind] = memory.permute(0, 2, 1).reshape(-1, self.hidden_dim, h, w).contiguous()
        inner_outs = [proj_feats[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_heigh = inner_outs[0]
            feat_low = proj_feats[idx - 1]
            feat_heigh = self.lateral_convs[len(self.in_channels) - 1 - idx](feat_heigh)
            inner_outs[0] = feat_heigh
            upsample_feat = F.interpolate(feat_heigh, scale_factor=2.0, mode='nearest')
            inner_out = self.fpn_blocks[len(self.in_channels) - 1 - idx](torch.concat([upsample_feat, feat_low], dim=1))
            inner_outs.insert(0, inner_out)
        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_height = inner_outs[idx + 1]
            downsample_feat = self.downsample_convs[idx](feat_low)
            out = self.pan_blocks[idx](torch.concat([downsample_feat, feat_height], dim=1))
            outs.append(out)
        return outs


def box_cxcywh_to_xyxy(x: 'Tensor') ->Tensor:
    x_c, y_c, w, h = x.unbind(-1)
    b = [x_c - 0.5 * w, y_c - 0.5 * h, x_c + 0.5 * w, y_c + 0.5 * h]
    return torch.stack(b, dim=-1)


def box_iou(boxes1: 'Tensor', boxes2: 'Tensor'):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2 - inter
    iou = inter / union
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)
    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    area = wh[:, :, 0] * wh[:, :, 1]
    return iou - (area - union) / area


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """
    __share__ = ['use_focal_loss']

    def __init__(self, weight_dict, use_focal_loss=False, alpha=0.25, gamma=2.0):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = weight_dict['cost_class']
        self.cost_bbox = weight_dict['cost_bbox']
        self.cost_giou = weight_dict['cost_giou']
        self.use_focal_loss = use_focal_loss
        self.alpha = alpha
        self.gamma = gamma
        assert self.cost_class != 0 or self.cost_bbox != 0 or self.cost_giou != 0, 'all costs cant be 0'

    @torch.no_grad()
    def forward(self, outputs: 'Dict[str, torch.Tensor]', targets):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bs, num_queries = outputs['pred_logits'].shape[:2]
        if self.use_focal_loss:
            out_prob = F.sigmoid(outputs['pred_logits'].flatten(0, 1))
        else:
            out_prob = outputs['pred_logits'].flatten(0, 1).softmax(-1)
        out_bbox = outputs['pred_boxes'].flatten(0, 1)
        tgt_ids = torch.cat([v['labels'] for v in targets])
        tgt_bbox = torch.cat([v['boxes'] for v in targets])
        if self.use_focal_loss:
            out_prob = out_prob[:, tgt_ids]
            neg_cost_class = (1 - self.alpha) * out_prob ** self.gamma * -(1 - out_prob + 1e-08).log()
            pos_cost_class = self.alpha * (1 - out_prob) ** self.gamma * -(out_prob + 1e-08).log()
            cost_class = pos_cost_class - neg_cost_class
        else:
            cost_class = -out_prob[:, tgt_ids]
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1).cpu()
        sizes = [len(v['boxes']) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        indices = [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]
        return {'indices': indices}


class RTDETR(nn.Module):
    __inject__ = ['backbone', 'encoder', 'decoder']

    def __init__(self, backbone: 'nn.Module', encoder: 'nn.Module', decoder: 'nn.Module'):
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder
        self.encoder = encoder

    def forward(self, x, targets=None):
        x = self.backbone(x)
        x = self.encoder(x)
        x = self.decoder(x, targets)
        return x

    def deploy(self):
        self.eval()
        for m in self.modules():
            if hasattr(m, 'convert_to_deploy'):
                m.convert_to_deploy()
        return self


@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    if target.numel() == 0:
        return [torch.zeros([], device=output.device)]
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def is_dist_available_and_initialized():
    if not torch.distributed.is_available():
        return False
    if not torch.distributed.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_available_and_initialized():
        return 1
    return torch.distributed.get_world_size()


def interpolate(input, size=None, scale_factor=None, mode='nearest', align_corners=None):
    """
    Equivalent to nn.functional.interpolate, but with support for empty batch sizes.
    This will eventually be supported natively by PyTorch, and this
    class can go away.
    """
    if version.parse(torchvision.__version__) < version.parse('0.7'):
        if input.numel() > 0:
            return torch.nn.functional.interpolate(input, size, scale_factor, mode, align_corners)
        output_shape = _output_size(2, input, size, scale_factor)
        output_shape = list(input.shape[:-2]) + list(output_shape)
        return _new_empty_tensor(input, output_shape)
    else:
        return torchvision.ops.misc.interpolate(input, size, scale_factor, mode, align_corners)


def sigmoid_focal_loss(logit, label, normalizer=1.0, alpha=0.25, gamma=2.0):
    prob = F.sigmoid(logit)
    ce_loss = F.binary_cross_entropy_with_logits(logit, label, reduction='none')
    p_t = prob * label + (1 - prob) * (1 - label)
    loss = ce_loss * (1 - p_t) ** gamma
    if alpha >= 0:
        alpha_t = alpha * label + (1 - alpha) * (1 - label)
        loss = alpha_t * loss
    return loss.mean(1).sum() / normalizer


class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    __share__ = ['num_classes']
    __inject__ = ['matcher']

    def __init__(self, matcher, weight_dict, losses, alpha=0.2, gamma=2.0, eos_coef=0.0001, num_classes=80):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = eos_coef
        self.register_buffer('empty_weight', empty_weight)
        self.alpha = alpha
        self.gamma = gamma

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t['labels'][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}
        if log:
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    def loss_labels_bce(self, outputs, targets, indices, num_boxes, log=True):
        src_logits = outputs['pred_logits']
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t['labels'][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]
        loss = F.binary_cross_entropy_with_logits(src_logits, target * 1.0, reduction='none')
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes
        return {'loss_bce': loss}

    def loss_labels_focal(self, outputs, targets, indices, num_boxes, log=True):
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t['labels'][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]
        loss = torchvision.ops.sigmoid_focal_loss(src_logits, target, self.alpha, self.gamma, reduction='none')
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes
        return {'loss_focal': loss}

    def loss_labels_vfl(self, outputs, targets, indices, num_boxes, log=True):
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        ious, _ = box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes))
        ious = torch.diag(ious).detach()
        src_logits = outputs['pred_logits']
        target_classes_o = torch.cat([t['labels'][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]
        target_score_o = torch.zeros_like(target_classes, dtype=src_logits.dtype)
        target_score_o[idx] = ious
        target_score = target_score_o.unsqueeze(-1) * target
        pred_score = F.sigmoid(src_logits).detach()
        weight = self.alpha * pred_score.pow(self.gamma) * (1 - target) + target_score
        loss = F.binary_cross_entropy_with_logits(src_logits, target_score, weight=weight, reduction='none')
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes
        return {'loss_vfl': loss}

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v['labels']) for v in targets], device=device)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        losses = {}
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes
        loss_giou = 1 - torch.diag(generalized_box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert 'pred_masks' in outputs
        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs['pred_masks']
        src_masks = src_masks[src_idx]
        masks = [t['masks'] for t in targets]
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks
        target_masks = target_masks[tgt_idx]
        src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:], mode='bilinear', align_corners=False)
        src_masks = src_masks[:, 0].flatten(1)
        target_masks = target_masks.flatten(1)
        target_masks = target_masks.view(src_masks.shape)
        losses = {'loss_mask': sigmoid_focal_loss(src_masks, target_masks, num_boxes), 'loss_dice': dice_loss(src_masks, target_masks, num_boxes)}
        return losses

    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for src, _ in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for _, tgt in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {'labels': self.loss_labels, 'cardinality': self.loss_cardinality, 'boxes': self.loss_boxes, 'masks': self.loss_masks, 'bce': self.loss_labels_bce, 'focal': self.loss_labels_focal, 'vfl': self.loss_labels_vfl}
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if 'aux' not in k}
        indices = self.matcher(outputs_without_aux, targets)
        num_boxes = sum(len(t['labels']) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_available_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()
        losses = {}
        for loss in self.losses:
            l_dict = self.get_loss(loss, outputs, targets, indices, num_boxes)
            l_dict = {k: (l_dict[k] * self.weight_dict[k]) for k in l_dict if k in self.weight_dict}
            losses.update(l_dict)
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == 'masks':
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k: (l_dict[k] * self.weight_dict[k]) for k in l_dict if k in self.weight_dict}
                    l_dict = {(k + f'_aux_{i}'): v for k, v in l_dict.items()}
                    losses.update(l_dict)
        if 'dn_aux_outputs' in outputs:
            assert 'dn_meta' in outputs, ''
            indices = self.get_cdn_matched_indices(outputs['dn_meta'], targets)
            num_boxes = num_boxes * outputs['dn_meta']['dn_num_group']
            for i, aux_outputs in enumerate(outputs['dn_aux_outputs']):
                for loss in self.losses:
                    if loss == 'masks':
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k: (l_dict[k] * self.weight_dict[k]) for k in l_dict if k in self.weight_dict}
                    l_dict = {(k + f'_dn_{i}'): v for k, v in l_dict.items()}
                    losses.update(l_dict)
        return losses

    @staticmethod
    def get_cdn_matched_indices(dn_meta, targets):
        """get_cdn_matched_indices
        """
        dn_positive_idx, dn_num_group = dn_meta['dn_positive_idx'], dn_meta['dn_num_group']
        num_gts = [len(t['labels']) for t in targets]
        device = targets[0]['labels'].device
        dn_match_indices = []
        for i, num_gt in enumerate(num_gts):
            if num_gt > 0:
                gt_idx = torch.arange(num_gt, dtype=torch.int64, device=device)
                gt_idx = gt_idx.tile(dn_num_group)
                assert len(dn_positive_idx[i]) == len(gt_idx)
                dn_match_indices.append((dn_positive_idx[i], gt_idx))
            else:
                dn_match_indices.append((torch.zeros(0, dtype=torch.int64, device=device), torch.zeros(0, dtype=torch.int64, device=device)))
        return dn_match_indices


class MLP(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, act='relu'):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.act = get_activation(act)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = self.act(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def deformable_attention_core_func_v2(value: 'torch.Tensor', value_spatial_shapes, sampling_locations: 'torch.Tensor', attention_weights: 'torch.Tensor', num_points_list: 'List[int]', method='default'):
    """
    Args:
        value (Tensor): [bs, value_length, n_head, c]
        value_spatial_shapes (Tensor|List): [n_levels, 2]
        value_level_start_index (Tensor|List): [n_levels]
        sampling_locations (Tensor): [bs, query_length, n_head, n_levels * n_points, 2]
        attention_weights (Tensor): [bs, query_length, n_head, n_levels * n_points]

    Returns:
        output (Tensor): [bs, Length_{query}, C]
    """
    bs, _, n_head, c = value.shape
    _, Len_q, _, _, _ = sampling_locations.shape
    split_shape = [(h * w) for h, w in value_spatial_shapes]
    value_list = value.permute(0, 2, 3, 1).flatten(0, 1).split(split_shape, dim=-1)
    if method == 'default':
        sampling_grids = 2 * sampling_locations - 1
    elif method == 'discrete':
        sampling_grids = sampling_locations
    sampling_grids = sampling_grids.permute(0, 2, 1, 3, 4).flatten(0, 1)
    sampling_locations_list = sampling_grids.split(num_points_list, dim=-2)
    sampling_value_list = []
    for level, (h, w) in enumerate(value_spatial_shapes):
        value_l = value_list[level].reshape(bs * n_head, c, h, w)
        sampling_grid_l: 'torch.Tensor' = sampling_locations_list[level]
        if method == 'default':
            sampling_value_l = F.grid_sample(value_l, sampling_grid_l, mode='bilinear', padding_mode='zeros', align_corners=False)
        elif method == 'discrete':
            sampling_coord = sampling_grid_l * torch.tensor([[w, h]], device=value.device) + 0.5
            sampling_coord = sampling_coord.clamp(0, h - 1)
            sampling_coord = sampling_coord.reshape(bs * n_head, Len_q * num_points_list[level], 2)
            s_idx = torch.arange(sampling_coord.shape[0], device=value.device).unsqueeze(-1).repeat(1, sampling_coord.shape[1])
            sampling_value_l: 'torch.Tensor' = value_l[s_idx, :, sampling_coord[..., 1], sampling_coord[..., 0]]
            sampling_value_l = sampling_value_l.permute(0, 2, 1).reshape(bs * n_head, c, Len_q, num_points_list[level])
        sampling_value_list.append(sampling_value_l)
    attn_weights = attention_weights.permute(0, 2, 1, 3).reshape(bs * n_head, 1, Len_q, sum(num_points_list))
    weighted_sample_locs = torch.concat(sampling_value_list, dim=-1) * attn_weights
    output = weighted_sample_locs.sum(-1).reshape(bs, n_head * c, Len_q)
    return output.permute(0, 2, 1)


class MSDeformableAttention(nn.Module):

    def __init__(self, embed_dim=256, num_heads=8, num_levels=4, num_points=4, method='default', offset_scale=0.5):
        """Multi-Scale Deformable Attention
        """
        super(MSDeformableAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.offset_scale = offset_scale
        if isinstance(num_points, list):
            assert len(num_points) == num_levels, ''
            num_points_list = num_points
        else:
            num_points_list = [num_points for _ in range(num_levels)]
        self.num_points_list = num_points_list
        num_points_scale = [(1 / n) for n in num_points_list for _ in range(n)]
        self.register_buffer('num_points_scale', torch.tensor(num_points_scale, dtype=torch.float32))
        self.total_points = num_heads * sum(num_points_list)
        self.method = method
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, 'embed_dim must be divisible by num_heads'
        self.sampling_offsets = nn.Linear(embed_dim, self.total_points * 2)
        self.attention_weights = nn.Linear(embed_dim, self.total_points)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)
        self.ms_deformable_attn_core = functools.partial(deformable_attention_core_func_v2, method=self.method)
        self._reset_parameters()
        if method == 'discrete':
            for p in self.sampling_offsets.parameters():
                p.requires_grad = False

    def _reset_parameters(self):
        init.constant_(self.sampling_offsets.weight, 0)
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = grid_init / grid_init.abs().max(-1, keepdim=True).values
        grid_init = grid_init.reshape(self.num_heads, 1, 2).tile([1, sum(self.num_points_list), 1])
        scaling = torch.concat([torch.arange(1, n + 1) for n in self.num_points_list]).reshape(1, -1, 1)
        grid_init *= scaling
        self.sampling_offsets.bias.data[...] = grid_init.flatten()
        init.constant_(self.attention_weights.weight, 0)
        init.constant_(self.attention_weights.bias, 0)
        init.xavier_uniform_(self.value_proj.weight)
        init.constant_(self.value_proj.bias, 0)
        init.xavier_uniform_(self.output_proj.weight)
        init.constant_(self.output_proj.bias, 0)

    def forward(self, query: 'torch.Tensor', reference_points: 'torch.Tensor', value: 'torch.Tensor', value_spatial_shapes: 'List[int]', value_mask: 'torch.Tensor'=None):
        """
        Args:
            query (Tensor): [bs, query_length, C]
            reference_points (Tensor): [bs, query_length, n_levels, 2], range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area
            value (Tensor): [bs, value_length, C]
            value_spatial_shapes (List): [n_levels, 2], [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
            value_mask (Tensor): [bs, value_length], True for non-padding elements, False for padding elements

        Returns:
            output (Tensor): [bs, Length_{query}, C]
        """
        bs, Len_q = query.shape[:2]
        Len_v = value.shape[1]
        value = self.value_proj(value)
        if value_mask is not None:
            value = value * value_mask.unsqueeze(-1)
        value = value.reshape(bs, Len_v, self.num_heads, self.head_dim)
        sampling_offsets: 'torch.Tensor' = self.sampling_offsets(query)
        sampling_offsets = sampling_offsets.reshape(bs, Len_q, self.num_heads, sum(self.num_points_list), 2)
        attention_weights = self.attention_weights(query).reshape(bs, Len_q, self.num_heads, sum(self.num_points_list))
        attention_weights = F.softmax(attention_weights, dim=-1).reshape(bs, Len_q, self.num_heads, sum(self.num_points_list))
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.tensor(value_spatial_shapes)
            offset_normalizer = offset_normalizer.flip([1]).reshape(1, 1, 1, self.num_levels, 1, 2)
            sampling_locations = reference_points.reshape(bs, Len_q, 1, self.num_levels, 1, 2) + sampling_offsets / offset_normalizer
        elif reference_points.shape[-1] == 4:
            num_points_scale = self.num_points_scale.unsqueeze(-1)
            offset = sampling_offsets * num_points_scale * reference_points[:, :, None, :, 2:] * self.offset_scale
            sampling_locations = reference_points[:, :, None, :, :2] + offset
        else:
            raise ValueError('Last dim of reference_points must be 2 or 4, but get {} instead.'.format(reference_points.shape[-1]))
        output = self.ms_deformable_attn_core(value, value_spatial_shapes, sampling_locations, attention_weights, self.num_points_list)
        output = self.output_proj(output)
        return output


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model=256, n_head=8, dim_feedforward=1024, dropout=0.0, activation='relu', n_levels=4, n_points=4, cross_attn_method='default'):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.cross_attn = MSDeformableAttention(d_model, n_head, n_levels, n_points, method=cross_attn_method)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = get_activation(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)
        self._reset_parameters()

    def _reset_parameters(self):
        init.xavier_uniform_(self.linear1.weight)
        init.xavier_uniform_(self.linear2.weight)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        return self.linear2(self.dropout3(self.activation(self.linear1(tgt))))

    def forward(self, target, reference_points, memory, memory_spatial_shapes, attn_mask=None, memory_mask=None, query_pos_embed=None):
        q = k = self.with_pos_embed(target, query_pos_embed)
        target2, _ = self.self_attn(q, k, value=target, attn_mask=attn_mask)
        target = target + self.dropout1(target2)
        target = self.norm1(target)
        target2 = self.cross_attn(self.with_pos_embed(target, query_pos_embed), reference_points, memory, memory_spatial_shapes, memory_mask)
        target = target + self.dropout2(target2)
        target = self.norm2(target)
        target2 = self.forward_ffn(target)
        target = target + self.dropout4(target2)
        target = self.norm3(target)
        return target


def inverse_sigmoid(x: 'torch.Tensor', eps: 'float'=1e-05) ->torch.Tensor:
    x = x.clip(min=0.0, max=1.0)
    return torch.log(x.clip(min=eps) / (1 - x).clip(min=eps))


class TransformerDecoder(nn.Module):

    def __init__(self, hidden_dim, decoder_layer, num_layers, eval_idx=-1):
        super(TransformerDecoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.eval_idx = eval_idx if eval_idx >= 0 else num_layers + eval_idx

    def forward(self, target, ref_points_unact, memory, memory_spatial_shapes, bbox_head, score_head, query_pos_head, attn_mask=None, memory_mask=None):
        dec_out_bboxes = []
        dec_out_logits = []
        ref_points_detach = F.sigmoid(ref_points_unact)
        output = target
        for i, layer in enumerate(self.layers):
            ref_points_input = ref_points_detach.unsqueeze(2)
            query_pos_embed = query_pos_head(ref_points_detach)
            output = layer(output, ref_points_input, memory, memory_spatial_shapes, attn_mask, memory_mask, query_pos_embed)
            inter_ref_bbox = F.sigmoid(bbox_head[i](output) + inverse_sigmoid(ref_points_detach))
            if self.training:
                dec_out_logits.append(score_head[i](output))
                if i == 0:
                    dec_out_bboxes.append(inter_ref_bbox)
                else:
                    dec_out_bboxes.append(F.sigmoid(bbox_head[i](output) + inverse_sigmoid(ref_points)))
            elif i == self.eval_idx:
                dec_out_logits.append(score_head[i](output))
                dec_out_bboxes.append(inter_ref_bbox)
                break
            ref_points = inter_ref_bbox
            ref_points_detach = inter_ref_bbox.detach()
        return torch.stack(dec_out_bboxes), torch.stack(dec_out_logits)


def bias_init_with_prob(prior_prob=0.01):
    """initialize conv/fc bias value according to a given probability value."""
    bias_init = float(-math.log((1 - prior_prob) / prior_prob))
    return bias_init


def box_xyxy_to_cxcywh(x: 'Tensor') ->Tensor:
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0]
    return torch.stack(b, dim=-1)


def get_contrastive_denoising_training_group(targets, num_classes, num_queries, class_embed, num_denoising=100, label_noise_ratio=0.5, box_noise_scale=1.0):
    """cnd"""
    if num_denoising <= 0:
        return None, None, None, None
    num_gts = [len(t['labels']) for t in targets]
    device = targets[0]['labels'].device
    max_gt_num = max(num_gts)
    if max_gt_num == 0:
        return None, None, None, None
    num_group = num_denoising // max_gt_num
    num_group = 1 if num_group == 0 else num_group
    bs = len(num_gts)
    input_query_class = torch.full([bs, max_gt_num], num_classes, dtype=torch.int32, device=device)
    input_query_bbox = torch.zeros([bs, max_gt_num, 4], device=device)
    pad_gt_mask = torch.zeros([bs, max_gt_num], dtype=torch.bool, device=device)
    for i in range(bs):
        num_gt = num_gts[i]
        if num_gt > 0:
            input_query_class[i, :num_gt] = targets[i]['labels']
            input_query_bbox[i, :num_gt] = targets[i]['boxes']
            pad_gt_mask[i, :num_gt] = 1
    input_query_class = input_query_class.tile([1, 2 * num_group])
    input_query_bbox = input_query_bbox.tile([1, 2 * num_group, 1])
    pad_gt_mask = pad_gt_mask.tile([1, 2 * num_group])
    negative_gt_mask = torch.zeros([bs, max_gt_num * 2, 1], device=device)
    negative_gt_mask[:, max_gt_num:] = 1
    negative_gt_mask = negative_gt_mask.tile([1, num_group, 1])
    positive_gt_mask = 1 - negative_gt_mask
    positive_gt_mask = positive_gt_mask.squeeze(-1) * pad_gt_mask
    dn_positive_idx = torch.nonzero(positive_gt_mask)[:, 1]
    dn_positive_idx = torch.split(dn_positive_idx, [(n * num_group) for n in num_gts])
    num_denoising = int(max_gt_num * 2 * num_group)
    if label_noise_ratio > 0:
        mask = torch.rand_like(input_query_class, dtype=torch.float) < label_noise_ratio * 0.5
        new_label = torch.randint_like(mask, 0, num_classes, dtype=input_query_class.dtype)
        input_query_class = torch.where(mask & pad_gt_mask, new_label, input_query_class)
    if box_noise_scale > 0:
        known_bbox = box_cxcywh_to_xyxy(input_query_bbox)
        diff = torch.tile(input_query_bbox[..., 2:] * 0.5, [1, 1, 2]) * box_noise_scale
        rand_sign = torch.randint_like(input_query_bbox, 0, 2) * 2.0 - 1.0
        rand_part = torch.rand_like(input_query_bbox)
        rand_part = (rand_part + 1.0) * negative_gt_mask + rand_part * (1 - negative_gt_mask)
        known_bbox += rand_sign * rand_part * diff
        known_bbox = torch.clip(known_bbox, min=0.0, max=1.0)
        input_query_bbox = box_xyxy_to_cxcywh(known_bbox)
        input_query_bbox_unact = inverse_sigmoid(input_query_bbox)
    input_query_logits = class_embed(input_query_class)
    tgt_size = num_denoising + num_queries
    attn_mask = torch.full([tgt_size, tgt_size], False, dtype=torch.bool, device=device)
    attn_mask[num_denoising:, :num_denoising] = True
    for i in range(num_group):
        if i == 0:
            attn_mask[max_gt_num * 2 * i:max_gt_num * 2 * (i + 1), max_gt_num * 2 * (i + 1):num_denoising] = True
        if i == num_group - 1:
            attn_mask[max_gt_num * 2 * i:max_gt_num * 2 * (i + 1), :max_gt_num * i * 2] = True
        else:
            attn_mask[max_gt_num * 2 * i:max_gt_num * 2 * (i + 1), max_gt_num * 2 * (i + 1):num_denoising] = True
            attn_mask[max_gt_num * 2 * i:max_gt_num * 2 * (i + 1), :max_gt_num * 2 * i] = True
    dn_meta = {'dn_positive_idx': dn_positive_idx, 'dn_num_group': num_group, 'dn_num_split': [num_denoising, num_queries]}
    return input_query_logits, input_query_bbox_unact, attn_mask, dn_meta


class RTDETRTransformer(nn.Module):
    __share__ = ['num_classes']

    def __init__(self, num_classes=80, hidden_dim=256, num_queries=300, position_embed_type='sine', feat_channels=[512, 1024, 2048], feat_strides=[8, 16, 32], num_levels=3, num_points=4, nhead=8, num_layers=6, dim_feedforward=1024, dropout=0.0, activation='relu', num_denoising=100, label_noise_ratio=0.5, box_noise_scale=1.0, learnt_init_query=False, eval_spatial_size=None, eval_idx=-1, eps=0.01, aux_loss=True, version='v1'):
        super(RTDETRTransformer, self).__init__()
        assert position_embed_type in ['sine', 'learned'], f'ValueError: position_embed_type not supported {position_embed_type}!'
        assert len(feat_channels) <= num_levels
        assert len(feat_strides) == len(feat_channels)
        for _ in range(num_levels - len(feat_strides)):
            feat_strides.append(feat_strides[-1] * 2)
        self.hidden_dim = hidden_dim
        self.nhead = nhead
        self.feat_strides = feat_strides
        self.num_levels = num_levels
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.eps = eps
        self.num_layers = num_layers
        self.eval_spatial_size = eval_spatial_size
        self.aux_loss = aux_loss
        self._build_input_proj_layer(feat_channels)
        decoder_layer = TransformerDecoderLayer(hidden_dim, nhead, dim_feedforward, dropout, activation, num_levels, num_points)
        self.decoder = TransformerDecoder(hidden_dim, decoder_layer, num_layers, eval_idx)
        self.num_denoising = num_denoising
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale
        if num_denoising > 0:
            self.denoising_class_embed = nn.Embedding(num_classes + 1, hidden_dim, padding_idx=num_classes)
            init.normal_(self.denoising_class_embed.weight[:-1])
        self.learnt_init_query = learnt_init_query
        if learnt_init_query:
            self.tgt_embed = nn.Embedding(num_queries, hidden_dim)
        self.query_pos_head = MLP(4, 2 * hidden_dim, hidden_dim, num_layers=2)
        if version == 'v1':
            self.enc_output = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim))
        else:
            self.enc_output = nn.Sequential(OrderedDict([('proj', nn.Linear(hidden_dim, hidden_dim)), ('norm', nn.LayerNorm(hidden_dim))]))
        self.enc_score_head = nn.Linear(hidden_dim, num_classes)
        self.enc_bbox_head = MLP(hidden_dim, hidden_dim, 4, num_layers=3)
        self.dec_score_head = nn.ModuleList([nn.Linear(hidden_dim, num_classes) for _ in range(num_layers)])
        self.dec_bbox_head = nn.ModuleList([MLP(hidden_dim, hidden_dim, 4, num_layers=3) for _ in range(num_layers)])
        if self.eval_spatial_size:
            self.anchors, self.valid_mask = self._generate_anchors()
        self._reset_parameters()

    def _reset_parameters(self):
        bias = bias_init_with_prob(0.01)
        init.constant_(self.enc_score_head.bias, bias)
        init.constant_(self.enc_bbox_head.layers[-1].weight, 0)
        init.constant_(self.enc_bbox_head.layers[-1].bias, 0)
        for cls_, reg_ in zip(self.dec_score_head, self.dec_bbox_head):
            init.constant_(cls_.bias, bias)
            init.constant_(reg_.layers[-1].weight, 0)
            init.constant_(reg_.layers[-1].bias, 0)
        init.xavier_uniform_(self.enc_output[0].weight)
        if self.learnt_init_query:
            init.xavier_uniform_(self.tgt_embed.weight)
        init.xavier_uniform_(self.query_pos_head.layers[0].weight)
        init.xavier_uniform_(self.query_pos_head.layers[1].weight)

    def _build_input_proj_layer(self, feat_channels):
        self.input_proj = nn.ModuleList()
        for in_channels in feat_channels:
            self.input_proj.append(nn.Sequential(OrderedDict([('conv', nn.Conv2d(in_channels, self.hidden_dim, 1, bias=False)), ('norm', nn.BatchNorm2d(self.hidden_dim))])))
        in_channels = feat_channels[-1]
        for _ in range(self.num_levels - len(feat_channels)):
            self.input_proj.append(nn.Sequential(OrderedDict([('conv', nn.Conv2d(in_channels, self.hidden_dim, 3, 2, padding=1, bias=False)), ('norm', nn.BatchNorm2d(self.hidden_dim))])))
            in_channels = self.hidden_dim

    def _get_encoder_input(self, feats):
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]
        if self.num_levels > len(proj_feats):
            len_srcs = len(proj_feats)
            for i in range(len_srcs, self.num_levels):
                if i == len_srcs:
                    proj_feats.append(self.input_proj[i](feats[-1]))
                else:
                    proj_feats.append(self.input_proj[i](proj_feats[-1]))
        feat_flatten = []
        spatial_shapes = []
        level_start_index = [0]
        for i, feat in enumerate(proj_feats):
            _, _, h, w = feat.shape
            feat_flatten.append(feat.flatten(2).permute(0, 2, 1))
            spatial_shapes.append([h, w])
            level_start_index.append(h * w + level_start_index[-1])
        feat_flatten = torch.concat(feat_flatten, 1)
        level_start_index.pop()
        return feat_flatten, spatial_shapes, level_start_index

    def _generate_anchors(self, spatial_shapes=None, grid_size=0.05, dtype=torch.float32, device='cpu'):
        if spatial_shapes is None:
            spatial_shapes = [[int(self.eval_spatial_size[0] / s), int(self.eval_spatial_size[1] / s)] for s in self.feat_strides]
        anchors = []
        for lvl, (h, w) in enumerate(spatial_shapes):
            grid_y, grid_x = torch.meshgrid(torch.arange(end=h, dtype=dtype), torch.arange(end=w, dtype=dtype), indexing='ij')
            grid_xy = torch.stack([grid_x, grid_y], -1)
            valid_WH = torch.tensor([w, h])
            grid_xy = (grid_xy.unsqueeze(0) + 0.5) / valid_WH
            wh = torch.ones_like(grid_xy) * grid_size * 2.0 ** lvl
            anchors.append(torch.concat([grid_xy, wh], -1).reshape(-1, h * w, 4))
        anchors = torch.concat(anchors, 1)
        valid_mask = ((anchors > self.eps) * (anchors < 1 - self.eps)).all(-1, keepdim=True)
        anchors = torch.log(anchors / (1 - anchors))
        anchors = torch.where(valid_mask, anchors, torch.inf)
        return anchors, valid_mask

    def _get_decoder_input(self, memory, spatial_shapes, denoising_class=None, denoising_bbox_unact=None):
        bs, _, _ = memory.shape
        if self.training or self.eval_spatial_size is None:
            anchors, valid_mask = self._generate_anchors(spatial_shapes, device=memory.device)
        else:
            anchors, valid_mask = self.anchors, self.valid_mask
        memory = valid_mask * memory
        output_memory = self.enc_output(memory)
        enc_outputs_class = self.enc_score_head(output_memory)
        enc_outputs_coord_unact = self.enc_bbox_head(output_memory) + anchors
        _, topk_ind = torch.topk(enc_outputs_class.max(-1).values, self.num_queries, dim=1)
        reference_points_unact = enc_outputs_coord_unact.gather(dim=1, index=topk_ind.unsqueeze(-1).repeat(1, 1, enc_outputs_coord_unact.shape[-1]))
        enc_topk_bboxes = F.sigmoid(reference_points_unact)
        if denoising_bbox_unact is not None:
            reference_points_unact = torch.concat([denoising_bbox_unact, reference_points_unact], 1)
        enc_topk_logits = enc_outputs_class.gather(dim=1, index=topk_ind.unsqueeze(-1).repeat(1, 1, enc_outputs_class.shape[-1]))
        if self.learnt_init_query:
            target = self.tgt_embed.weight.unsqueeze(0).tile([bs, 1, 1])
        else:
            target = output_memory.gather(dim=1, index=topk_ind.unsqueeze(-1).repeat(1, 1, output_memory.shape[-1]))
            target = target.detach()
        if denoising_class is not None:
            target = torch.concat([denoising_class, target], 1)
        return target, reference_points_unact.detach(), enc_topk_bboxes, enc_topk_logits

    def forward(self, feats, targets=None):
        memory, spatial_shapes, level_start_index = self._get_encoder_input(feats)
        if self.training and self.num_denoising > 0:
            denoising_class, denoising_bbox_unact, attn_mask, dn_meta = get_contrastive_denoising_training_group(targets, self.num_classes, self.num_queries, self.denoising_class_embed, num_denoising=self.num_denoising, label_noise_ratio=self.label_noise_ratio, box_noise_scale=self.box_noise_scale)
        else:
            denoising_class, denoising_bbox_unact, attn_mask, dn_meta = None, None, None, None
        target, init_ref_points_unact, enc_topk_bboxes, enc_topk_logits = self._get_decoder_input(memory, spatial_shapes, denoising_class, denoising_bbox_unact)
        out_bboxes, out_logits = self.decoder(target, init_ref_points_unact, memory, spatial_shapes, level_start_index, self.dec_bbox_head, self.dec_score_head, self.query_pos_head, attn_mask=attn_mask)
        if self.training and dn_meta is not None:
            dn_out_bboxes, out_bboxes = torch.split(out_bboxes, dn_meta['dn_num_split'], dim=2)
            dn_out_logits, out_logits = torch.split(out_logits, dn_meta['dn_num_split'], dim=2)
        out = {'pred_logits': out_logits[-1], 'pred_boxes': out_bboxes[-1]}
        if self.training and self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(out_logits[:-1], out_bboxes[:-1])
            out['aux_outputs'].extend(self._set_aux_loss([enc_topk_logits], [enc_topk_bboxes]))
            if self.training and dn_meta is not None:
                out['dn_aux_outputs'] = self._set_aux_loss(dn_out_logits, dn_out_bboxes)
                out['dn_meta'] = dn_meta
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        return [{'pred_logits': a, 'pred_boxes': b} for a, b in zip(outputs_class, outputs_coord)]


def mod(a, b):
    out = a - a // b * b
    return out


mscoco_category2name = {(1): 'person', (2): 'bicycle', (3): 'car', (4): 'motorcycle', (5): 'airplane', (6): 'bus', (7): 'train', (8): 'truck', (9): 'boat', (10): 'traffic light', (11): 'fire hydrant', (13): 'stop sign', (14): 'parking meter', (15): 'bench', (16): 'bird', (17): 'cat', (18): 'dog', (19): 'horse', (20): 'sheep', (21): 'cow', (22): 'elephant', (23): 'bear', (24): 'zebra', (25): 'giraffe', (27): 'backpack', (28): 'umbrella', (31): 'handbag', (32): 'tie', (33): 'suitcase', (34): 'frisbee', (35): 'skis', (36): 'snowboard', (37): 'sports ball', (38): 'kite', (39): 'baseball bat', (40): 'baseball glove', (41): 'skateboard', (42): 'surfboard', (43): 'tennis racket', (44): 'bottle', (46): 'wine glass', (47): 'cup', (48): 'fork', (49): 'knife', (50): 'spoon', (51): 'bowl', (52): 'banana', (53): 'apple', (54): 'sandwich', (55): 'orange', (56): 'broccoli', (57): 'carrot', (58): 'hot dog', (59): 'pizza', (60): 'donut', (61): 'cake', (62): 'chair', (63): 'couch', (64): 'potted plant', (65): 'bed', (67): 'dining table', (70): 'toilet', (72): 'tv', (73): 'laptop', (74): 'mouse', (75): 'remote', (76): 'keyboard', (77): 'cell phone', (78): 'microwave', (79): 'oven', (80): 'toaster', (81): 'sink', (82): 'refrigerator', (84): 'book', (85): 'clock', (86): 'vase', (87): 'scissors', (88): 'teddy bear', (89): 'hair drier', (90): 'toothbrush'}


mscoco_category2label = {k: i for i, k in enumerate(mscoco_category2name.keys())}


mscoco_label2category = {v: k for k, v in mscoco_category2label.items()}


class RTDETRPostProcessor(nn.Module):
    __share__ = ['num_classes', 'use_focal_loss', 'num_top_queries', 'remap_mscoco_category']

    def __init__(self, num_classes=80, use_focal_loss=True, num_top_queries=300, remap_mscoco_category=False) ->None:
        super().__init__()
        self.use_focal_loss = use_focal_loss
        self.num_top_queries = num_top_queries
        self.num_classes = int(num_classes)
        self.remap_mscoco_category = remap_mscoco_category
        self.deploy_mode = False

    def extra_repr(self) ->str:
        return f'use_focal_loss={self.use_focal_loss}, num_classes={self.num_classes}, num_top_queries={self.num_top_queries}'

    def forward(self, outputs, orig_target_sizes: 'torch.Tensor'):
        logits, boxes = outputs['pred_logits'], outputs['pred_boxes']
        bbox_pred = torchvision.ops.box_convert(boxes, in_fmt='cxcywh', out_fmt='xyxy')
        bbox_pred *= orig_target_sizes.repeat(1, 2).unsqueeze(1)
        if self.use_focal_loss:
            scores = F.sigmoid(logits)
            scores, index = torch.topk(scores.flatten(1), self.num_top_queries, dim=-1)
            labels = mod(index, self.num_classes)
            index = index // self.num_classes
            boxes = bbox_pred.gather(dim=1, index=index.unsqueeze(-1).repeat(1, 1, bbox_pred.shape[-1]))
        else:
            scores = F.softmax(logits)[:, :, :-1]
            scores, labels = scores.max(dim=-1)
            if scores.shape[1] > self.num_top_queries:
                scores, index = torch.topk(scores, self.num_top_queries, dim=-1)
                labels = torch.gather(labels, dim=1, index=index)
                boxes = torch.gather(boxes, dim=1, index=index.unsqueeze(-1).tile(1, 1, boxes.shape[-1]))
        if self.deploy_mode:
            return labels, boxes, scores
        if self.remap_mscoco_category:
            labels = torch.tensor([mscoco_label2category[int(x.item())] for x in labels.flatten()]).reshape(labels.shape)
        results = []
        for lab, box, sco in zip(labels, boxes, scores):
            result = dict(labels=lab, boxes=box, scores=sco)
            results.append(result)
        return results

    def deploy(self):
        self.eval()
        self.deploy_mode = True
        return self


class C3(nn.Module):

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, act='silu'):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1, act=act)
        self.cv2 = Conv(c1, c_, 1, 1, act=act)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0, act=act) for _ in range(n)))
        self.cv3 = Conv(2 * c_, c2, 1, act=act)

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))


class SPPF(nn.Module):

    def __init__(self, c1, c2, k=5, act='silu'):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1, act=act)
        self.cv2 = Conv(c_ * 4, c2, 1, 1, act=act)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            y1 = self.m(x)
            y2 = self.m(y1)
            return self.cv2(torch.cat([x, y1, y2, self.m(y2)], 1))


def make_divisible(c, d):
    return math.ceil(c / d) * d


class CSPDarkNet(nn.Module):
    __share__ = ['depth_multi', 'width_multi']

    def __init__(self, in_channels=3, width_multi=1.0, depth_multi=1.0, return_idx=[2, 3, -1], act='silu') ->None:
        super().__init__()
        channels = [64, 128, 256, 512, 1024]
        channels = [make_divisible(c * width_multi, 8) for c in channels]
        depths = [3, 6, 9, 3]
        depths = [max(round(d * depth_multi), 1) for d in depths]
        self.layers = nn.ModuleList([Conv(in_channels, channels[0], 6, 2, 2, act=act)])
        for i, (c, d) in enumerate(zip(channels, depths), 1):
            layer = nn.Sequential(*[Conv(c, channels[i], 3, 2, act=act), C3(channels[i], channels[i], n=d, act=act)])
            self.layers.append(layer)
        self.layers.append(SPPF(channels[-1], channels[-1], k=5, act=act))
        self.return_idx = return_idx
        self.out_channels = [channels[i] for i in self.return_idx]
        self.strides = [[2, 4, 8, 16, 32][i] for i in self.return_idx]
        self.depths = depths
        self.act = act

    def forward(self, x):
        outputs = []
        for _, m in enumerate(self.layers):
            x = m(x)
            outputs.append(x)
        return [outputs[i] for i in self.return_idx]


class CSPPAN(nn.Module):
    """
    P5 ---> 1x1  ---------------------------------> concat --> c3 --> det
             | up                                     | conv /2 
    P4 ---> concat ---> c3 ---> 1x1  -->  concat ---> c3 -----------> det
                                 | up       | conv /2
    P3 -----------------------> concat ---> c3 ---------------------> det
    """
    __share__ = ['depth_multi']

    def __init__(self, in_channels=[256, 512, 1024], depth_multi=1.0, act='silu') ->None:
        super().__init__()
        depth = max(round(3 * depth_multi), 1)
        self.out_channels = in_channels
        self.fpn_stems = nn.ModuleList([Conv(cin, cout, 1, 1, act=act) for cin, cout in zip(in_channels[::-1], in_channels[::-1][1:])])
        self.fpn_csps = nn.ModuleList([C3(cin, cout, depth, False, act=act) for cin, cout in zip(in_channels[::-1], in_channels[::-1][1:])])
        self.pan_stems = nn.ModuleList([Conv(c, c, 3, 2, act=act) for c in in_channels[:-1]])
        self.pan_csps = nn.ModuleList([C3(c, c, depth, False, act=act) for c in in_channels[1:]])

    def forward(self, feats):
        fpn_feats = []
        for i, feat in enumerate(feats[::-1]):
            if i == 0:
                feat = self.fpn_stems[i](feat)
                fpn_feats.append(feat)
            else:
                _feat = F.interpolate(fpn_feats[-1], scale_factor=2, mode='nearest')
                feat = torch.concat([_feat, feat], dim=1)
                feat = self.fpn_csps[i - 1](feat)
                if i < len(self.fpn_stems):
                    feat = self.fpn_stems[i](feat)
                fpn_feats.append(feat)
        pan_feats = []
        for i, feat in enumerate(fpn_feats[::-1]):
            if i == 0:
                pan_feats.append(feat)
            else:
                _feat = self.pan_stems[i - 1](pan_feats[-1])
                feat = torch.concat([_feat, feat], dim=1)
                feat = self.pan_csps[i - 1](feat)
                pan_feats.append(feat)
        return pan_feats


class ConvBNLayer(nn.Module):

    def __init__(self, ch_in, ch_out, filter_size=3, stride=1, groups=1, padding=0, act=None):
        super().__init__()
        self.conv = nn.Conv2d(ch_in, ch_out, filter_size, stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(ch_out)
        self.act = get_activation(act)

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class EffectiveSELayer(nn.Module):
    """ Effective Squeeze-Excitation
    From `CenterMask : Real-Time Anchor-Free Instance Segmentation` - https://arxiv.org/abs/1911.06667
    """

    def __init__(self, channels, act='hardsigmoid'):
        super(EffectiveSELayer, self).__init__()
        self.fc = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
        self.act = get_activation(act)

    def forward(self, x: 'torch.Tensor'):
        x_se = x.mean((2, 3), keepdim=True)
        x_se = self.fc(x_se)
        x_se = self.act(x_se)
        return x * x_se


class CSPResStage(nn.Module):

    def __init__(self, block_fn, ch_in, ch_out, n, stride, act='relu', attn='eca', use_alpha=False):
        super().__init__()
        ch_mid = (ch_in + ch_out) // 2
        if stride == 2:
            self.conv_down = ConvBNLayer(ch_in, ch_mid, 3, stride=2, padding=1, act=act)
        else:
            self.conv_down = None
        self.conv1 = ConvBNLayer(ch_mid, ch_mid // 2, 1, act=act)
        self.conv2 = ConvBNLayer(ch_mid, ch_mid // 2, 1, act=act)
        self.blocks = nn.Sequential(*[block_fn(ch_mid // 2, ch_mid // 2, act=act, shortcut=True, use_alpha=use_alpha) for i in range(n)])
        if attn:
            self.attn = EffectiveSELayer(ch_mid, act='hardsigmoid')
        else:
            self.attn = None
        self.conv3 = ConvBNLayer(ch_mid, ch_out, 1, act=act)

    def forward(self, x):
        if self.conv_down is not None:
            x = self.conv_down(x)
        y1 = self.conv1(x)
        y2 = self.blocks(self.conv2(x))
        y = torch.concat([y1, y2], dim=1)
        if self.attn is not None:
            y = self.attn(y)
        y = self.conv3(y)
        return y


class CSPResNet(nn.Module):
    layers = [3, 6, 6, 3]
    channels = [64, 128, 256, 512, 1024]
    model_cfg = {'s': {'depth_mult': 0.33, 'width_mult': 0.5}, 'm': {'depth_mult': 0.67, 'width_mult': 0.75}, 'l': {'depth_mult': 1.0, 'width_mult': 1.0}, 'x': {'depth_mult': 1.33, 'width_mult': 1.25}}

    def __init__(self, name: 'str', act='silu', return_idx=[1, 2, 3], use_large_stem=True, use_alpha=False, pretrained=False):
        super().__init__()
        depth_mult = self.model_cfg[name]['depth_mult']
        width_mult = self.model_cfg[name]['width_mult']
        channels = [max(round(c * width_mult), 1) for c in self.channels]
        layers = [max(round(l * depth_mult), 1) for l in self.layers]
        act = get_activation(act)
        if use_large_stem:
            self.stem = nn.Sequential(OrderedDict([('conv1', ConvBNLayer(3, channels[0] // 2, 3, stride=2, padding=1, act=act)), ('conv2', ConvBNLayer(channels[0] // 2, channels[0] // 2, 3, stride=1, padding=1, act=act)), ('conv3', ConvBNLayer(channels[0] // 2, channels[0], 3, stride=1, padding=1, act=act))]))
        else:
            self.stem = nn.Sequential(OrderedDict([('conv1', ConvBNLayer(3, channels[0] // 2, 3, stride=2, padding=1, act=act)), ('conv2', ConvBNLayer(channels[0] // 2, channels[0], 3, stride=1, padding=1, act=act))]))
        n = len(channels) - 1
        self.stages = nn.Sequential(OrderedDict([(str(i), CSPResStage(BasicBlock, channels[i], channels[i + 1], layers[i], 2, act=act, use_alpha=use_alpha)) for i in range(n)]))
        self._out_channels = channels[1:]
        self._out_strides = [(4 * 2 ** i) for i in range(n)]
        self.return_idx = return_idx
        if pretrained:
            if isinstance(pretrained, bool) or 'http' in pretrained:
                state = torch.hub.load_state_dict_from_url(donwload_url[name], map_location='cpu')
            else:
                state = torch.load(pretrained, map_location='cpu')
            self.load_state_dict(state)
            None

    def forward(self, x):
        x = self.stem(x)
        outs = []
        for idx, stage in enumerate(self.stages):
            x = stage(x)
            if idx in self.return_idx:
                outs.append(x)
        return outs


class LearnableAffineBlock(nn.Module):

    def __init__(self, scale_value=1.0, bias_value=0.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor([scale_value]))
        self.bias = nn.Parameter(torch.tensor([bias_value]))

    def forward(self, x: 'Tensor') ->Tensor:
        return self.scale * x + self.bias


class ConvBNAct(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, groups=1, use_act=True, use_lab=False):
        super().__init__()
        self.use_act = use_act
        self.use_lab = use_lab
        if padding == 'same':
            self.conv = nn.Sequential(nn.ZeroPad2d([0, 1, 0, 1]), nn.Conv2d(in_channels, out_channels, kernel_size, stride, groups=groups, bias=False))
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=(kernel_size - 1) // 2, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        if self.use_act:
            self.act = nn.ReLU()
            if self.use_lab:
                self.lab = LearnableAffineBlock()

    def forward(self, x: 'Tensor') ->Tensor:
        x = self.conv(x)
        x = self.bn(x)
        if self.use_act:
            x = self.act(x)
            if self.use_lab:
                x = self.lab(x)
        return x


class LightConvBNAct(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, groups=1, use_lab=False):
        super().__init__()
        self.conv1 = ConvBNAct(in_channels=in_channels, out_channels=out_channels, kernel_size=1, use_act=False, use_lab=use_lab)
        self.conv2 = ConvBNAct(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, groups=out_channels, use_act=True, use_lab=use_lab)

    def forward(self, x: 'Tensor') ->Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class StemBlock(nn.Module):

    def __init__(self, in_channels, mid_channels, out_channels, use_lab=False):
        super().__init__()
        self.stem1 = ConvBNAct(in_channels=in_channels, out_channels=mid_channels, kernel_size=3, stride=2, use_lab=use_lab)
        self.stem2a = ConvBNAct(in_channels=mid_channels, out_channels=mid_channels // 2, kernel_size=2, stride=1, padding='same', use_lab=use_lab)
        self.stem2b = ConvBNAct(in_channels=mid_channels // 2, out_channels=mid_channels, kernel_size=2, stride=1, padding='same', use_lab=use_lab)
        self.stem3 = ConvBNAct(in_channels=mid_channels * 2, out_channels=mid_channels, kernel_size=3, stride=2, use_lab=use_lab)
        self.stem4 = ConvBNAct(in_channels=mid_channels, out_channels=out_channels, kernel_size=1, stride=1, use_lab=use_lab)
        self.pool = nn.Sequential(nn.ZeroPad2d([0, 1, 0, 1]), nn.MaxPool2d(2, 1, ceil_mode=True))

    def forward(self, x: 'Tensor') ->Tensor:
        x = self.stem1(x)
        x2 = self.stem2a(x)
        x2 = self.stem2b(x2)
        x1 = self.pool(x)
        x = torch.concat([x1, x2], dim=1)
        x = self.stem3(x)
        x = self.stem4(x)
        return x


class HG_Block(nn.Module):

    def __init__(self, in_channels, mid_channels, out_channels, kernel_size=3, layer_num=6, identity=False, light_block=True, use_lab=False):
        super().__init__()
        self.identity = identity
        self.layers = nn.ModuleList()
        block_type = 'LightConvBNAct' if light_block else 'ConvBNAct'
        for i in range(layer_num):
            self.layers.append(eval(block_type)(in_channels=in_channels if i == 0 else mid_channels, out_channels=mid_channels, stride=1, kernel_size=kernel_size, use_lab=use_lab))
        total_channels = in_channels + layer_num * mid_channels
        self.aggregation_squeeze_conv = ConvBNAct(in_channels=total_channels, out_channels=out_channels // 2, kernel_size=1, stride=1, use_lab=use_lab)
        self.aggregation_excitation_conv = ConvBNAct(in_channels=out_channels // 2, out_channels=out_channels, kernel_size=1, stride=1, use_lab=use_lab)

    def forward(self, x):
        identity = x
        output = []
        output.append(x)
        for layer in self.layers:
            x = layer(x)
            output.append(x)
        x = torch.concat(output, dim=1)
        x = self.aggregation_squeeze_conv(x)
        x = self.aggregation_excitation_conv(x)
        if self.identity:
            x = x + identity
        return x


class HG_Stage(nn.Module):

    def __init__(self, in_channels, mid_channels, out_channels, block_num, layer_num=6, downsample=True, light_block=True, kernel_size=3, use_lab=False):
        super().__init__()
        self.downsample = downsample
        if downsample:
            self.downsample = ConvBNAct(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=2, groups=in_channels, use_act=False, use_lab=use_lab)
        blocks_list = []
        for i in range(block_num):
            blocks_list.append(HG_Block(in_channels=in_channels if i == 0 else out_channels, mid_channels=mid_channels, out_channels=out_channels, kernel_size=kernel_size, layer_num=layer_num, identity=False if i == 0 else True, light_block=light_block, use_lab=use_lab))
        self.blocks = nn.Sequential(*blocks_list)

    def forward(self, x):
        if self.downsample:
            x = self.downsample(x)
        x = self.blocks(x)
        return x


class HGNetv2(nn.Module):
    """
    Args:
        stem_channels: list. Number of channels for the stem block.
        stage_type: str. The stage configuration of PPHGNet. such as the number of channels, stride, etc.
        use_lab: boolean. Whether to use LearnableAffineBlock in network.
        lr_mult_list: list. Control the learning rate of different stages.
    Returns:
        model: nn.Module.
    """
    arch_configs = {'L': {'stem_channels': [3, 32, 48], 'stage_config': {'stage1': [48, 48, 128, 1, False, False, 3, 6], 'stage2': [128, 96, 512, 1, True, False, 3, 6], 'stage3': [512, 192, 1024, 3, True, True, 5, 6], 'stage4': [1024, 384, 2048, 1, True, True, 5, 6]}, 'url': 'https://github.com/lyuwenyu/storage/releases/download/v0.1/PPHGNetV2_L_ssld_pretrained_from_paddle.pth'}, 'X': {'stem_channels': [3, 32, 64], 'stage_config': {'stage1': [64, 64, 128, 1, False, False, 3, 6], 'stage2': [128, 128, 512, 2, True, False, 3, 6], 'stage3': [512, 256, 1024, 5, True, True, 5, 6], 'stage4': [1024, 512, 2048, 2, True, True, 5, 6]}, 'url': 'https://github.com/lyuwenyu/storage/releases/download/v0.1/PPHGNetV2_X_ssld_pretrained_from_paddle.pth'}, 'H': {'stem_channels': [3, 48, 96], 'stage_config': {'stage1': [96, 96, 192, 2, False, False, 3, 6], 'stage2': [192, 192, 512, 3, True, False, 3, 6], 'stage3': [512, 384, 1024, 6, True, True, 5, 6], 'stage4': [1024, 768, 2048, 3, True, True, 5, 6]}, 'url': 'https://github.com/lyuwenyu/storage/releases/download/v0.1/PPHGNetV2_H_ssld_pretrained_from_paddle.pth'}}

    def __init__(self, name, use_lab=False, return_idx=[1, 2, 3], freeze_at=-1, freeze_norm=False, pretrained=False):
        super().__init__()
        self.use_lab = use_lab
        self.return_idx = return_idx
        stem_channels = self.arch_configs[name]['stem_channels']
        stage_config = self.arch_configs[name]['stage_config']
        download_url = self.arch_configs[name]['url']
        self._out_strides = [4, 8, 16, 32]
        self._out_channels = [stage_config[k][2] for k in stage_config]
        self.stem = StemBlock(in_channels=stem_channels[0], mid_channels=stem_channels[1], out_channels=stem_channels[2], use_lab=use_lab)
        self.stages = nn.ModuleList()
        for i, k in enumerate(stage_config):
            in_channels, mid_channels, out_channels, block_num, downsample, light_block, kernel_size, layer_num = stage_config[k]
            self.stages.append(HG_Stage(in_channels, mid_channels, out_channels, block_num, layer_num, downsample, light_block, kernel_size, use_lab))
        self._init_weights()
        if freeze_at >= 0:
            self._freeze_parameters(self.stem)
            for i in range(min(freeze_at, 4)):
                self._freeze_parameters(self.stages[i])
        if freeze_norm:
            self._freeze_norm(self)
        if pretrained:
            if isinstance(pretrained, bool) or 'http' in pretrained:
                state = torch.hub.load_state_dict_from_url(download_url, map_location='cpu')
            else:
                state = torch.load(pretrained, map_location='cpu')
            self.load_state_dict(state)
            None

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.constant_(m.bias, 0)

    def _freeze_parameters(self, m: 'nn.Module'):
        for p in m.parameters():
            p.requires_grad = False

    def _freeze_norm(self, m: 'nn.Module'):
        if isinstance(m, nn.BatchNorm2d):
            m = FrozenBatchNorm2d(m.num_features)
        else:
            for name, child in m.named_children():
                _child = self._freeze_norm(child)
                if _child is not child:
                    setattr(m, name, _child)
        return m

    def forward(self, x: 'Tensor') ->List[Tensor]:
        x = self.stem(x)
        outs = []
        for idx, stage in enumerate(self.stages):
            x = stage(x)
            if idx in self.return_idx:
                outs.append(x)
        return outs


class TimmModel(torch.nn.Module):

    def __init__(self, name, return_layers, pretrained=False, exportable=True, features_only=True, **kwargs) ->None:
        super().__init__()
        model = timm.create_model(name, pretrained=pretrained, exportable=exportable, features_only=features_only, **kwargs)
        assert set(return_layers).issubset(model.feature_info.module_name()), f'return_layers should be a subset of {model.feature_info.module_name()}'
        self.model = IntermediateLayerGetter(model, return_layers)
        return_idx = [model.feature_info.module_name().index(name) for name in return_layers]
        self.strides = [model.feature_info.reduction()[i] for i in return_idx]
        self.channels = [model.feature_info.channels()[i] for i in return_idx]
        self.return_idx = return_idx
        self.return_layers = return_layers

    def forward(self, x: 'torch.Tensor'):
        outputs = self.model(x)
        return outputs


class TorchVisionModel(torch.nn.Module):

    def __init__(self, name, return_layers, weights=None, **kwargs) ->None:
        super().__init__()
        if weights is not None:
            weights = getattr(torchvision.models.get_model_weights(name), weights)
        model = torchvision.models.get_model(name, weights=weights, **kwargs)
        if hasattr(model, 'features'):
            model = IntermediateLayerGetter(model.features, return_layers)
        else:
            model = IntermediateLayerGetter(model, return_layers)
        self.model = model

    def forward(self, x):
        return self.model(x)


class DetCriterion(torch.nn.Module):
    """Default Detection Criterion
    """
    __share__ = ['num_classes']
    __inject__ = ['matcher']

    def __init__(self, losses, weight_dict, num_classes=80, alpha=0.75, gamma=2.0, box_fmt='cxcywh', matcher=None):
        """
        Args:
            losses (list[str]): requested losses, support ['boxes', 'vfl', 'focal']
            weight_dict (dict[str, float)]: corresponding losses weight, including
                ['loss_bbox', 'loss_giou', 'loss_vfl', 'loss_focal']
            box_fmt (str): in box format, 'cxcywh' or 'xyxy'
            matcher (Matcher): matcher used to match source to target
        """
        super().__init__()
        self.losses = losses
        self.weight_dict = weight_dict
        self.alpha = alpha
        self.gamma = gamma
        self.num_classes = num_classes
        self.box_fmt = box_fmt
        assert matcher is not None, ''
        self.matcher = matcher

    def forward(self, outputs, targets, **kwargs):
        """
        Args:
            outputs: Dict[Tensor], 'pred_boxes', 'pred_logits', 'meta'.
            targets, List[Dict[str, Tensor]], len(targets) == batch_size.
            kwargs, store other information such as current epoch id.
        Return:
            losses, Dict[str, Tensor]
        """
        matched = self.matcher(outputs, targets)
        values = matched['values']
        indices = matched['indices']
        num_boxes = self._get_positive_nums(indices)
        losses = {}
        for loss in self.losses:
            l_dict = self.get_loss(loss, outputs, targets, indices, num_boxes)
            l_dict = {k: (l_dict[k] * self.weight_dict[k]) for k in l_dict if k in self.weight_dict}
            losses.update(l_dict)
        return losses

    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for src, _ in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for _, tgt in indices])
        return batch_idx, tgt_idx

    def _get_positive_nums(self, indices):
        num_pos = sum(len(i) for i, _ in indices)
        num_pos = torch.as_tensor([num_pos], dtype=torch.float32, device=indices[0][0].device)
        if dist_utils.is_dist_available_and_initialized():
            torch.distributed.all_reduce(num_pos)
        num_pos = torch.clamp(num_pos / dist_utils.get_world_size(), min=1).item()
        return num_pos

    def loss_labels_focal(self, outputs, targets, indices, num_boxes):
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t['labels'][j] for t, (_, j) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]
        loss = torchvision.ops.sigmoid_focal_loss(src_logits, target, self.alpha, self.gamma, reduction='none')
        loss = loss.sum() / num_boxes
        return {'loss_focal': loss}

    def loss_labels_vfl(self, outputs, targets, indices, num_boxes):
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][j] for t, (_, j) in zip(targets, indices)], dim=0)
        src_boxes = torchvision.ops.box_convert(src_boxes, in_fmt=self.box_fmt, out_fmt='xyxy')
        target_boxes = torchvision.ops.box_convert(target_boxes, in_fmt=self.box_fmt, out_fmt='xyxy')
        iou, _ = box_ops.elementwise_box_iou(src_boxes.detach(), target_boxes)
        src_logits: 'torch.Tensor' = outputs['pred_logits']
        target_classes_o = torch.cat([t['labels'][j] for t, (_, j) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]
        target_score_o = torch.zeros_like(target_classes, dtype=src_logits.dtype)
        target_score_o[idx] = iou
        target_score = target_score_o.unsqueeze(-1) * target
        src_score = F.sigmoid(src_logits.detach())
        weight = self.alpha * src_score.pow(self.gamma) * (1 - target) + target_score
        loss = F.binary_cross_entropy_with_logits(src_logits, target_score, weight=weight, reduction='none')
        loss = loss.sum() / num_boxes
        return {'loss_vfl': loss}

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        losses = {}
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes
        src_boxes = torchvision.ops.box_convert(src_boxes, in_fmt=self.box_fmt, out_fmt='xyxy')
        target_boxes = torchvision.ops.box_convert(target_boxes, in_fmt=self.box_fmt, out_fmt='xyxy')
        loss_giou = 1 - box_ops.elementwise_generalized_box_iou(src_boxes, target_boxes)
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def loss_boxes_giou(self, outputs, targets, indices, num_boxes):
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        losses = {}
        src_boxes = torchvision.ops.box_convert(src_boxes, in_fmt=self.box_fmt, out_fmt='xyxy')
        target_boxes = torchvision.ops.box_convert(target_boxes, in_fmt=self.box_fmt, out_fmt='xyxy')
        loss_giou = 1 - box_ops.elementwise_generalized_box_iou(src_boxes, target_boxes)
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {'boxes': self.loss_boxes, 'giou': self.loss_boxes_giou, 'vfl': self.loss_labels_vfl, 'focal': self.loss_labels_focal}
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)


class BoxProcessFormat(Enum):
    """Box process format 

    Available formats are
    * ``RESIZE``
    * ``RESIZE_KEEP_RATIO``
    * ``RESIZE_KEEP_RATIO_PADDING``
    """
    RESIZE = 1
    RESIZE_KEEP_RATIO = 2
    RESIZE_KEEP_RATIO_PADDING = 3


def box_revert(boxes: 'Tensor', orig_sizes: 'Tensor'=None, eval_sizes: 'Tensor'=None, inpt_sizes: 'Tensor'=None, inpt_padding: 'Tensor'=None, normalized: 'bool'=True, in_fmt: 'str'='cxcywh', out_fmt: 'str'='xyxy', process_fmt=BoxProcessFormat.RESIZE) ->Tensor:
    """
    Args:
        boxes(Tensor), [N, :, 4], (x1, y1, x2, y2), pred boxes.
        inpt_sizes(Tensor), [N, 2], (w, h). input sizes.
        orig_sizes(Tensor), [N, 2], (w, h). origin sizes.
        inpt_padding (Tensor), [N, 2], (w_pad, h_pad, ...).
        (inpt_sizes + inpt_padding) == eval_sizes
    """
    assert in_fmt in ('cxcywh', 'xyxy'), ''
    if normalized and eval_sizes is not None:
        boxes = boxes * eval_sizes.repeat(1, 2).unsqueeze(1)
    if inpt_padding is not None:
        if in_fmt == 'xyxy':
            boxes -= inpt_padding[:, :2].repeat(1, 2).unsqueeze(1)
        elif in_fmt == 'cxcywh':
            boxes[..., :2] -= inpt_padding[:, :2].repeat(1, 2).unsqueeze(1)
    if orig_sizes is not None:
        orig_sizes = orig_sizes.repeat(1, 2).unsqueeze(1)
        if inpt_sizes is not None:
            inpt_sizes = inpt_sizes.repeat(1, 2).unsqueeze(1)
            boxes = boxes * (orig_sizes / inpt_sizes)
        else:
            boxes = boxes * orig_sizes
    boxes = torchvision.ops.box_convert(boxes, in_fmt=in_fmt, out_fmt=out_fmt)
    return boxes


class DetDETRPostProcessor(nn.Module):

    def __init__(self, num_classes=80, use_focal_loss=True, num_top_queries=300, box_process_format=BoxProcessFormat.RESIZE) ->None:
        super().__init__()
        self.use_focal_loss = use_focal_loss
        self.num_top_queries = num_top_queries
        self.num_classes = int(num_classes)
        self.box_process_format = box_process_format
        self.deploy_mode = False

    def extra_repr(self) ->str:
        return f'use_focal_loss={self.use_focal_loss}, num_classes={self.num_classes}, num_top_queries={self.num_top_queries}'

    def forward(self, outputs, **kwargs):
        logits, boxes = outputs['pred_logits'], outputs['pred_boxes']
        if self.use_focal_loss:
            scores = F.sigmoid(logits)
            scores, index = torch.topk(scores.flatten(1), self.num_top_queries, dim=-1)
            labels = index % self.num_classes
            index = index // self.num_classes
            boxes = boxes.gather(dim=1, index=index.unsqueeze(-1).repeat(1, 1, boxes.shape[-1]))
        else:
            scores = F.softmax(logits)[:, :, :-1]
            scores, labels = scores.max(dim=-1)
            if scores.shape[1] > self.num_top_queries:
                scores, index = torch.topk(scores, self.num_top_queries, dim=-1)
                labels = torch.gather(labels, dim=1, index=index)
                boxes = torch.gather(boxes, dim=1, index=index.unsqueeze(-1).tile(1, 1, boxes.shape[-1]))
        if kwargs is not None:
            boxes = box_revert(boxes, in_fmt='cxcywh', out_fmt='xyxy', process_fmt=self.box_process_format, normalized=True, **kwargs)
        if self.deploy_mode:
            return labels, boxes, scores
        results = []
        for lab, box, sco in zip(labels, boxes, scores):
            result = dict(labels=lab, boxes=box, scores=sco)
            results.append(result)
        return results

    def deploy(self):
        self.eval()
        self.deploy_mode = True
        return self


class DetNMSPostProcessor(torch.nn.Module):

    def __init__(self, iou_threshold=0.7, score_threshold=0.01, keep_topk=300, box_fmt='cxcywh', logit_fmt='sigmoid') ->None:
        super().__init__()
        self.iou_threshold = iou_threshold
        self.score_threshold = score_threshold
        self.keep_topk = keep_topk
        self.box_fmt = box_fmt.lower()
        self.logit_fmt = logit_fmt.lower()
        self.logit_func = getattr(F, self.logit_fmt, None)
        self.deploy_mode = False

    def forward(self, outputs: 'Dict[str, Tensor]', orig_target_sizes: 'Tensor'):
        logits, boxes = outputs['pred_logits'], outputs['pred_boxes']
        pred_boxes = torchvision.ops.box_convert(boxes, in_fmt=self.box_fmt, out_fmt='xyxy')
        pred_boxes *= orig_target_sizes.repeat(1, 2).unsqueeze(1)
        values, pred_labels = torch.max(logits, dim=-1)
        if self.logit_func:
            pred_scores = self.logit_func(values)
        else:
            pred_scores = values
        if self.deploy_mode:
            blobs = {'pred_labels': pred_labels, 'pred_boxes': pred_boxes, 'pred_scores': pred_scores}
            return blobs
        results = []
        for i in range(logits.shape[0]):
            score_keep = pred_scores[i] > self.score_threshold
            pred_box = pred_boxes[i][score_keep]
            pred_label = pred_labels[i][score_keep]
            pred_score = pred_scores[i][score_keep]
            keep = torchvision.ops.batched_nms(pred_box, pred_score, pred_label, self.iou_threshold)
            keep = keep[:self.keep_topk]
            blob = {'labels': pred_label[keep], 'boxes': pred_box[keep], 'scores': pred_score[keep]}
            results.append(blob)
        return results

    def deploy(self):
        self.eval()
        self.deploy_mode = True
        return self


class RTDETRCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    __share__ = ['num_classes']
    __inject__ = ['matcher']

    def __init__(self, matcher, weight_dict, losses, alpha=0.2, gamma=2.0, eos_coef=0.0001, num_classes=80):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = eos_coef
        self.register_buffer('empty_weight', empty_weight)
        self.alpha = alpha
        self.gamma = gamma

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t['labels'][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}
        if log:
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    def loss_labels_focal(self, outputs, targets, indices, num_boxes, log=True):
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t['labels'][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]
        loss = torchvision.ops.sigmoid_focal_loss(src_logits, target, self.alpha, self.gamma, reduction='none')
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes
        return {'loss_focal': loss}

    def loss_labels_vfl(self, outputs, targets, indices, num_boxes, log=True):
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        ious, _ = box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes))
        ious = torch.diag(ious).detach()
        src_logits = outputs['pred_logits']
        target_classes_o = torch.cat([t['labels'][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]
        target_score_o = torch.zeros_like(target_classes, dtype=src_logits.dtype)
        target_score_o[idx] = ious
        target_score = target_score_o.unsqueeze(-1) * target
        pred_score = F.sigmoid(src_logits).detach()
        weight = self.alpha * pred_score.pow(self.gamma) * (1 - target) + target_score
        loss = F.binary_cross_entropy_with_logits(src_logits, target_score, weight=weight, reduction='none')
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes
        return {'loss_vfl': loss}

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v['labels']) for v in targets], device=device)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        losses = {}
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes
        loss_giou = 1 - torch.diag(generalized_box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for src, _ in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for _, tgt in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {'labels': self.loss_labels, 'boxes': self.loss_boxes, 'cardinality': self.loss_cardinality, 'focal': self.loss_labels_focal, 'vfl': self.loss_labels_vfl}
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets, **kwargs):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if 'aux' not in k}
        num_boxes = sum(len(t['labels']) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_available_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()
        indices = self.matcher(outputs_without_aux, targets)['indices']
        losses = {}
        for loss in self.losses:
            l_dict = self.get_loss(loss, outputs, targets, indices, num_boxes)
            l_dict = {k: (l_dict[k] * self.weight_dict[k]) for k in l_dict if k in self.weight_dict}
            losses.update(l_dict)
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)['indices']
                for loss in self.losses:
                    if loss == 'masks':
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k: (l_dict[k] * self.weight_dict[k]) for k in l_dict if k in self.weight_dict}
                    l_dict = {(k + f'_aux_{i}'): v for k, v in l_dict.items()}
                    losses.update(l_dict)
        if 'dn_aux_outputs' in outputs:
            assert 'dn_meta' in outputs, ''
            indices = self.get_cdn_matched_indices(outputs['dn_meta'], targets)
            dn_num_boxes = num_boxes * outputs['dn_meta']['dn_num_group']
            for i, aux_outputs in enumerate(outputs['dn_aux_outputs']):
                for loss in self.losses:
                    if loss == 'masks':
                        continue
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, dn_num_boxes, **kwargs)
                    l_dict = {k: (l_dict[k] * self.weight_dict[k]) for k in l_dict if k in self.weight_dict}
                    l_dict = {(k + f'_dn_{i}'): v for k, v in l_dict.items()}
                    losses.update(l_dict)
        return losses

    @staticmethod
    def get_cdn_matched_indices(dn_meta, targets):
        """get_cdn_matched_indices
        """
        dn_positive_idx, dn_num_group = dn_meta['dn_positive_idx'], dn_meta['dn_num_group']
        num_gts = [len(t['labels']) for t in targets]
        device = targets[0]['labels'].device
        dn_match_indices = []
        for i, num_gt in enumerate(num_gts):
            if num_gt > 0:
                gt_idx = torch.arange(num_gt, dtype=torch.int64, device=device)
                gt_idx = gt_idx.tile(dn_num_group)
                assert len(dn_positive_idx[i]) == len(gt_idx)
                dn_match_indices.append((dn_positive_idx[i], gt_idx))
            else:
                dn_match_indices.append((torch.zeros(0, dtype=torch.int64, device=device), torch.zeros(0, dtype=torch.int64, device=device)))
        return dn_match_indices


class RTDETRCriterionv2(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    __share__ = ['num_classes']
    __inject__ = ['matcher']

    def __init__(self, matcher, weight_dict, losses, alpha=0.2, gamma=2.0, num_classes=80, boxes_weight_format=None, share_matched_indices=False):
        """Create the criterion.
        Parameters:
            matcher: module able to compute a matching between targets and proposals
            num_classes: number of object categories, omitting the special no-object category
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            boxes_weight_format: format for boxes weight (iou, )
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.boxes_weight_format = boxes_weight_format
        self.share_matched_indices = share_matched_indices
        self.alpha = alpha
        self.gamma = gamma

    def loss_labels_focal(self, outputs, targets, indices, num_boxes):
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t['labels'][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]
        loss = torchvision.ops.sigmoid_focal_loss(src_logits, target, self.alpha, self.gamma, reduction='none')
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes
        return {'loss_focal': loss}

    def loss_labels_vfl(self, outputs, targets, indices, num_boxes, values=None):
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        if values is None:
            src_boxes = outputs['pred_boxes'][idx]
            target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
            ious, _ = box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes))
            ious = torch.diag(ious).detach()
        else:
            ious = values
        src_logits = outputs['pred_logits']
        target_classes_o = torch.cat([t['labels'][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]
        target_score_o = torch.zeros_like(target_classes, dtype=src_logits.dtype)
        target_score_o[idx] = ious
        target_score = target_score_o.unsqueeze(-1) * target
        pred_score = F.sigmoid(src_logits).detach()
        weight = self.alpha * pred_score.pow(self.gamma) * (1 - target) + target_score
        loss = F.binary_cross_entropy_with_logits(src_logits, target_score, weight=weight, reduction='none')
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes
        return {'loss_vfl': loss}

    def loss_boxes(self, outputs, targets, indices, num_boxes, boxes_weight=None):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        losses = {}
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes
        loss_giou = 1 - torch.diag(generalized_box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes)))
        loss_giou = loss_giou if boxes_weight is None else loss_giou * boxes_weight
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for src, _ in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for _, tgt in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {'boxes': self.loss_boxes, 'focal': self.loss_labels_focal, 'vfl': self.loss_labels_vfl}
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets, **kwargs):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if 'aux' not in k}
        num_boxes = sum(len(t['labels']) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_available_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()
        matched = self.matcher(outputs_without_aux, targets)
        indices = matched['indices']
        losses = {}
        for loss in self.losses:
            meta = self.get_loss_meta_info(loss, outputs, targets, indices)
            l_dict = self.get_loss(loss, outputs, targets, indices, num_boxes, **meta)
            l_dict = {k: (l_dict[k] * self.weight_dict[k]) for k in l_dict if k in self.weight_dict}
            losses.update(l_dict)
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                if not self.share_matched_indices:
                    matched = self.matcher(aux_outputs, targets)
                    indices = matched['indices']
                for loss in self.losses:
                    meta = self.get_loss_meta_info(loss, aux_outputs, targets, indices)
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **meta)
                    l_dict = {k: (l_dict[k] * self.weight_dict[k]) for k in l_dict if k in self.weight_dict}
                    l_dict = {(k + f'_aux_{i}'): v for k, v in l_dict.items()}
                    losses.update(l_dict)
        if 'dn_aux_outputs' in outputs:
            assert 'dn_meta' in outputs, ''
            indices = self.get_cdn_matched_indices(outputs['dn_meta'], targets)
            dn_num_boxes = num_boxes * outputs['dn_meta']['dn_num_group']
            for i, aux_outputs in enumerate(outputs['dn_aux_outputs']):
                for loss in self.losses:
                    meta = self.get_loss_meta_info(loss, aux_outputs, targets, indices)
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, dn_num_boxes, **meta)
                    l_dict = {k: (l_dict[k] * self.weight_dict[k]) for k in l_dict if k in self.weight_dict}
                    l_dict = {(k + f'_dn_{i}'): v for k, v in l_dict.items()}
                    losses.update(l_dict)
        if 'enc_aux_outputs' in outputs:
            assert 'enc_meta' in outputs, ''
            class_agnostic = outputs['enc_meta']['class_agnostic']
            if class_agnostic:
                orig_num_classes = self.num_classes
                self.num_classes = 1
                enc_targets = copy.deepcopy(targets)
                for t in enc_targets:
                    t['labels'] = torch.zeros_like(t['labels'])
            else:
                enc_targets = targets
            for i, aux_outputs in enumerate(outputs['enc_aux_outputs']):
                matched = self.matcher(aux_outputs, targets)
                indices = matched['indices']
                for loss in self.losses:
                    meta = self.get_loss_meta_info(loss, aux_outputs, enc_targets, indices)
                    l_dict = self.get_loss(loss, aux_outputs, enc_targets, indices, num_boxes, **meta)
                    l_dict = {k: (l_dict[k] * self.weight_dict[k]) for k in l_dict if k in self.weight_dict}
                    l_dict = {(k + f'_enc_{i}'): v for k, v in l_dict.items()}
                    losses.update(l_dict)
            if class_agnostic:
                self.num_classes = orig_num_classes
        return losses

    def get_loss_meta_info(self, loss, outputs, targets, indices):
        if self.boxes_weight_format is None:
            return {}
        src_boxes = outputs['pred_boxes'][self._get_src_permutation_idx(indices)]
        target_boxes = torch.cat([t['boxes'][j] for t, (_, j) in zip(targets, indices)], dim=0)
        if self.boxes_weight_format == 'iou':
            iou, _ = box_iou(box_cxcywh_to_xyxy(src_boxes.detach()), box_cxcywh_to_xyxy(target_boxes))
            iou = torch.diag(iou)
        elif self.boxes_weight_format == 'giou':
            iou = torch.diag(generalized_box_iou(box_cxcywh_to_xyxy(src_boxes.detach()), box_cxcywh_to_xyxy(target_boxes)))
        else:
            raise AttributeError()
        if loss in ('boxes',):
            meta = {'boxes_weight': iou}
        elif loss in ('vfl',):
            meta = {'values': iou}
        else:
            meta = {}
        return meta

    @staticmethod
    def get_cdn_matched_indices(dn_meta, targets):
        """get_cdn_matched_indices
        """
        dn_positive_idx, dn_num_group = dn_meta['dn_positive_idx'], dn_meta['dn_num_group']
        num_gts = [len(t['labels']) for t in targets]
        device = targets[0]['labels'].device
        dn_match_indices = []
        for i, num_gt in enumerate(num_gts):
            if num_gt > 0:
                gt_idx = torch.arange(num_gt, dtype=torch.int64, device=device)
                gt_idx = gt_idx.tile(dn_num_group)
                assert len(dn_positive_idx[i]) == len(gt_idx)
                dn_match_indices.append((dn_positive_idx[i], gt_idx))
            else:
                dn_match_indices.append((torch.zeros(0, dtype=torch.int64, device=device), torch.zeros(0, dtype=torch.int64, device=device)))
        return dn_match_indices


class RTDETRTransformerv2(nn.Module):
    __share__ = ['num_classes', 'eval_spatial_size']

    def __init__(self, num_classes=80, hidden_dim=256, num_queries=300, feat_channels=[512, 1024, 2048], feat_strides=[8, 16, 32], num_levels=3, num_points=4, nhead=8, num_layers=6, dim_feedforward=1024, dropout=0.0, activation='relu', num_denoising=100, label_noise_ratio=0.5, box_noise_scale=1.0, learn_query_content=False, eval_spatial_size=None, eval_idx=-1, eps=0.01, aux_loss=True, cross_attn_method='default', query_select_method='default'):
        super().__init__()
        assert len(feat_channels) <= num_levels
        assert len(feat_strides) == len(feat_channels)
        for _ in range(num_levels - len(feat_strides)):
            feat_strides.append(feat_strides[-1] * 2)
        self.hidden_dim = hidden_dim
        self.nhead = nhead
        self.feat_strides = feat_strides
        self.num_levels = num_levels
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.eps = eps
        self.num_layers = num_layers
        self.eval_spatial_size = eval_spatial_size
        self.aux_loss = aux_loss
        assert query_select_method in ('default', 'one2many', 'agnostic'), ''
        assert cross_attn_method in ('default', 'discrete'), ''
        self.cross_attn_method = cross_attn_method
        self.query_select_method = query_select_method
        self._build_input_proj_layer(feat_channels)
        decoder_layer = TransformerDecoderLayer(hidden_dim, nhead, dim_feedforward, dropout, activation, num_levels, num_points, cross_attn_method=cross_attn_method)
        self.decoder = TransformerDecoder(hidden_dim, decoder_layer, num_layers, eval_idx)
        self.num_denoising = num_denoising
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale
        if num_denoising > 0:
            self.denoising_class_embed = nn.Embedding(num_classes + 1, hidden_dim, padding_idx=num_classes)
            init.normal_(self.denoising_class_embed.weight[:-1])
        self.learn_query_content = learn_query_content
        if learn_query_content:
            self.tgt_embed = nn.Embedding(num_queries, hidden_dim)
        self.query_pos_head = MLP(4, 2 * hidden_dim, hidden_dim, 2)
        self.enc_output = nn.Sequential(OrderedDict([('proj', nn.Linear(hidden_dim, hidden_dim)), ('norm', nn.LayerNorm(hidden_dim))]))
        if query_select_method == 'agnostic':
            self.enc_score_head = nn.Linear(hidden_dim, 1)
        else:
            self.enc_score_head = nn.Linear(hidden_dim, num_classes)
        self.enc_bbox_head = MLP(hidden_dim, hidden_dim, 4, 3)
        self.dec_score_head = nn.ModuleList([nn.Linear(hidden_dim, num_classes) for _ in range(num_layers)])
        self.dec_bbox_head = nn.ModuleList([MLP(hidden_dim, hidden_dim, 4, 3) for _ in range(num_layers)])
        if self.eval_spatial_size:
            anchors, valid_mask = self._generate_anchors()
            self.register_buffer('anchors', anchors)
            self.register_buffer('valid_mask', valid_mask)
        self._reset_parameters()

    def _reset_parameters(self):
        bias = bias_init_with_prob(0.01)
        init.constant_(self.enc_score_head.bias, bias)
        init.constant_(self.enc_bbox_head.layers[-1].weight, 0)
        init.constant_(self.enc_bbox_head.layers[-1].bias, 0)
        for _cls, _reg in zip(self.dec_score_head, self.dec_bbox_head):
            init.constant_(_cls.bias, bias)
            init.constant_(_reg.layers[-1].weight, 0)
            init.constant_(_reg.layers[-1].bias, 0)
        init.xavier_uniform_(self.enc_output[0].weight)
        if self.learn_query_content:
            init.xavier_uniform_(self.tgt_embed.weight)
        init.xavier_uniform_(self.query_pos_head.layers[0].weight)
        init.xavier_uniform_(self.query_pos_head.layers[1].weight)
        for m in self.input_proj:
            init.xavier_uniform_(m[0].weight)

    def _build_input_proj_layer(self, feat_channels):
        self.input_proj = nn.ModuleList()
        for in_channels in feat_channels:
            self.input_proj.append(nn.Sequential(OrderedDict([('conv', nn.Conv2d(in_channels, self.hidden_dim, 1, bias=False)), ('norm', nn.BatchNorm2d(self.hidden_dim))])))
        in_channels = feat_channels[-1]
        for _ in range(self.num_levels - len(feat_channels)):
            self.input_proj.append(nn.Sequential(OrderedDict([('conv', nn.Conv2d(in_channels, self.hidden_dim, 3, 2, padding=1, bias=False)), ('norm', nn.BatchNorm2d(self.hidden_dim))])))
            in_channels = self.hidden_dim

    def _get_encoder_input(self, feats: 'List[torch.Tensor]'):
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]
        if self.num_levels > len(proj_feats):
            len_srcs = len(proj_feats)
            for i in range(len_srcs, self.num_levels):
                if i == len_srcs:
                    proj_feats.append(self.input_proj[i](feats[-1]))
                else:
                    proj_feats.append(self.input_proj[i](proj_feats[-1]))
        feat_flatten = []
        spatial_shapes = []
        for i, feat in enumerate(proj_feats):
            _, _, h, w = feat.shape
            feat_flatten.append(feat.flatten(2).permute(0, 2, 1))
            spatial_shapes.append([h, w])
        feat_flatten = torch.concat(feat_flatten, 1)
        return feat_flatten, spatial_shapes

    def _generate_anchors(self, spatial_shapes=None, grid_size=0.05, dtype=torch.float32, device='cpu'):
        if spatial_shapes is None:
            spatial_shapes = []
            eval_h, eval_w = self.eval_spatial_size
            for s in self.feat_strides:
                spatial_shapes.append([int(eval_h / s), int(eval_w / s)])
        anchors = []
        for lvl, (h, w) in enumerate(spatial_shapes):
            grid_y, grid_x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
            grid_xy = torch.stack([grid_x, grid_y], dim=-1)
            grid_xy = (grid_xy.unsqueeze(0) + 0.5) / torch.tensor([w, h], dtype=dtype)
            wh = torch.ones_like(grid_xy) * grid_size * 2.0 ** lvl
            lvl_anchors = torch.concat([grid_xy, wh], dim=-1).reshape(-1, h * w, 4)
            anchors.append(lvl_anchors)
        anchors = torch.concat(anchors, dim=1)
        valid_mask = ((anchors > self.eps) * (anchors < 1 - self.eps)).all(-1, keepdim=True)
        anchors = torch.log(anchors / (1 - anchors))
        anchors = torch.where(valid_mask, anchors, torch.inf)
        return anchors, valid_mask

    def _get_decoder_input(self, memory: 'torch.Tensor', spatial_shapes, denoising_logits=None, denoising_bbox_unact=None):
        if self.training or self.eval_spatial_size is None:
            anchors, valid_mask = self._generate_anchors(spatial_shapes, device=memory.device)
        else:
            anchors = self.anchors
            valid_mask = self.valid_mask
        memory = valid_mask * memory
        output_memory: 'torch.Tensor' = self.enc_output(memory)
        enc_outputs_logits: 'torch.Tensor' = self.enc_score_head(output_memory)
        enc_outputs_coord_unact: 'torch.Tensor' = self.enc_bbox_head(output_memory) + anchors
        enc_topk_bboxes_list, enc_topk_logits_list = [], []
        enc_topk_memory, enc_topk_logits, enc_topk_bbox_unact = self._select_topk(output_memory, enc_outputs_logits, enc_outputs_coord_unact, self.num_queries)
        if self.training:
            enc_topk_bboxes = F.sigmoid(enc_topk_bbox_unact)
            enc_topk_bboxes_list.append(enc_topk_bboxes)
            enc_topk_logits_list.append(enc_topk_logits)
        if self.learn_query_content:
            content = self.tgt_embed.weight.unsqueeze(0).tile([memory.shape[0], 1, 1])
        else:
            content = enc_topk_memory.detach()
        enc_topk_bbox_unact = enc_topk_bbox_unact.detach()
        if denoising_bbox_unact is not None:
            enc_topk_bbox_unact = torch.concat([denoising_bbox_unact, enc_topk_bbox_unact], dim=1)
            content = torch.concat([denoising_logits, content], dim=1)
        return content, enc_topk_bbox_unact, enc_topk_bboxes_list, enc_topk_logits_list

    def _select_topk(self, memory: 'torch.Tensor', outputs_logits: 'torch.Tensor', outputs_coords_unact: 'torch.Tensor', topk: 'int'):
        if self.query_select_method == 'default':
            _, topk_ind = torch.topk(outputs_logits.max(-1).values, topk, dim=-1)
        elif self.query_select_method == 'one2many':
            _, topk_ind = torch.topk(outputs_logits.flatten(1), topk, dim=-1)
            topk_ind = topk_ind // self.num_classes
        elif self.query_select_method == 'agnostic':
            _, topk_ind = torch.topk(outputs_logits.squeeze(-1), topk, dim=-1)
        topk_ind: 'torch.Tensor'
        topk_coords = outputs_coords_unact.gather(dim=1, index=topk_ind.unsqueeze(-1).repeat(1, 1, outputs_coords_unact.shape[-1]))
        topk_logits = outputs_logits.gather(dim=1, index=topk_ind.unsqueeze(-1).repeat(1, 1, outputs_logits.shape[-1]))
        topk_memory = memory.gather(dim=1, index=topk_ind.unsqueeze(-1).repeat(1, 1, memory.shape[-1]))
        return topk_memory, topk_logits, topk_coords

    def forward(self, feats, targets=None):
        memory, spatial_shapes = self._get_encoder_input(feats)
        if self.training and self.num_denoising > 0:
            denoising_logits, denoising_bbox_unact, attn_mask, dn_meta = get_contrastive_denoising_training_group(targets, self.num_classes, self.num_queries, self.denoising_class_embed, num_denoising=self.num_denoising, label_noise_ratio=self.label_noise_ratio, box_noise_scale=self.box_noise_scale)
        else:
            denoising_logits, denoising_bbox_unact, attn_mask, dn_meta = None, None, None, None
        init_ref_contents, init_ref_points_unact, enc_topk_bboxes_list, enc_topk_logits_list = self._get_decoder_input(memory, spatial_shapes, denoising_logits, denoising_bbox_unact)
        out_bboxes, out_logits = self.decoder(init_ref_contents, init_ref_points_unact, memory, spatial_shapes, self.dec_bbox_head, self.dec_score_head, self.query_pos_head, attn_mask=attn_mask)
        if self.training and dn_meta is not None:
            dn_out_bboxes, out_bboxes = torch.split(out_bboxes, dn_meta['dn_num_split'], dim=2)
            dn_out_logits, out_logits = torch.split(out_logits, dn_meta['dn_num_split'], dim=2)
        out = {'pred_logits': out_logits[-1], 'pred_boxes': out_bboxes[-1]}
        if self.training and self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(out_logits[:-1], out_bboxes[:-1])
            out['enc_aux_outputs'] = self._set_aux_loss(enc_topk_logits_list, enc_topk_bboxes_list)
            out['enc_meta'] = {'class_agnostic': self.query_select_method == 'agnostic'}
            if dn_meta is not None:
                out['dn_aux_outputs'] = self._set_aux_loss(dn_out_logits, dn_out_bboxes)
                out['dn_meta'] = dn_meta
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        return [{'pred_logits': a, 'pred_boxes': b} for a, b in zip(outputs_class, outputs_coord)]


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (BasicBlock,
     lambda: ([], {'in_planes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (CSPRepLayer,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ClassHead,
     lambda: ([], {'hidden_dim': 4, 'num_classes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Classification,
     lambda: ([], {'backbone': torch.nn.ReLU()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ConvBNAct,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ConvBNLayer,
     lambda: ([], {'ch_in': 4, 'ch_out': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ConvNormLayer,
     lambda: ([], {'ch_in': 4, 'ch_out': 4, 'kernel_size': 4, 'stride': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (EffectiveSELayer,
     lambda: ([], {'channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (FrozenBatchNorm2d,
     lambda: ([], {'num_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (HG_Block,
     lambda: ([], {'in_channels': 4, 'mid_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (HG_Stage,
     lambda: ([], {'in_channels': 4, 'mid_channels': 4, 'out_channels': 4, 'block_num': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Identity,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (LearnableAffineBlock,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (LightConvBNAct,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4, 'stride': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (MLP,
     lambda: ([], {'input_dim': 4, 'hidden_dim': 4, 'output_dim': 4, 'num_layers': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (RepVggBlock,
     lambda: ([], {'ch_in': 4, 'ch_out': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (StemBlock,
     lambda: ([], {'in_channels': 4, 'mid_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (TransformerEncoderLayer,
     lambda: ([], {'d_model': 4, 'nhead': 4}),
     lambda: ([torch.rand([4, 4])], {})),
    (YOLO,
     lambda: ([], {'backbone': torch.nn.ReLU(), 'neck': torch.nn.ReLU(), 'head': torch.nn.ReLU()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
]

