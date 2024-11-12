
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


import torch.nn as nn


import torch


import math


import numpy as np


from scipy.optimize import minimize_scalar


from torch import nn


import torch.nn.functional as F


import torch.nn.init as init


import logging


import time


import random


import torchvision.transforms as transforms


import torchvision.datasets as datasets


import copy


BN = nn.BatchNorm2d


class _InvertedResidual(nn.Module):

    def __init__(self, in_ch, out_ch, kernel_size, stride, expansion_factor):
        super(_InvertedResidual, self).__init__()
        assert stride in [1, 2]
        assert kernel_size in [3, 5]
        mid_ch = in_ch * expansion_factor
        self.apply_residual = in_ch == out_ch and stride == 1
        self.layers = nn.Sequential(nn.Conv2d(in_ch, mid_ch, 1, bias=False), BN(mid_ch), nn.ReLU(inplace=True), nn.Conv2d(mid_ch, mid_ch, kernel_size, padding=kernel_size // 2, stride=stride, groups=mid_ch, bias=False), BN(mid_ch), nn.ReLU(inplace=True), nn.Conv2d(mid_ch, out_ch, 1, bias=False), BN(out_ch))

    def forward(self, input):
        if self.apply_residual:
            return self.layers(input) + input
        else:
            return self.layers(input)


def _round_to_multiple_of(val, divisor, round_up_bias=0.9):
    """ Asymmetric rounding to make `val` divisible by `divisor`. With default
    bias, will round up, unless the number is no more than 10% greater than the
    smaller divisible value, i.e. (83, 8) -> 80, but (84, 8) -> 88. """
    assert 0.0 < round_up_bias < 1.0
    new_val = max(divisor, int(val + divisor / 2) // divisor * divisor)
    return new_val if new_val >= round_up_bias * val else new_val + divisor


def _get_depths(scale):
    """ Scales tensor depths as in reference MobileNet code, prefers rouding up
    rather than down. """
    depths = [32, 16, 24, 40, 80, 96, 192, 320]
    return [_round_to_multiple_of(depth * scale, 8) for depth in depths]


def _stack(in_ch, out_ch, kernel_size, stride, exp_factor, repeats):
    """ Creates a stack of inverted residuals. """
    assert repeats >= 1
    first = _InvertedResidual(in_ch, out_ch, kernel_size, stride, exp_factor)
    remaining = []
    for _ in range(1, repeats):
        remaining.append(_InvertedResidual(out_ch, out_ch, kernel_size, 1, exp_factor))
    return nn.Sequential(first, *remaining)


class MNASNet(torch.nn.Module):
    _version = 2

    def __init__(self, scale=2.0, num_classes=1000, dropout=0.0):
        super(MNASNet, self).__init__()
        global BN
        BN = nn.BatchNorm2d
        assert scale > 0.0
        self.scale = scale
        self.num_classes = num_classes
        depths = _get_depths(scale)
        layers = [nn.Conv2d(3, depths[0], 3, padding=1, stride=2, bias=False), BN(depths[0]), nn.ReLU(inplace=True), nn.Conv2d(depths[0], depths[0], 3, padding=1, stride=1, groups=depths[0], bias=False), BN(depths[0]), nn.ReLU(inplace=True), nn.Conv2d(depths[0], depths[1], 1, padding=0, stride=1, bias=False), BN(depths[1]), _stack(depths[1], depths[2], 3, 2, 3, 3), _stack(depths[2], depths[3], 5, 2, 3, 3), _stack(depths[3], depths[4], 5, 2, 6, 3), _stack(depths[4], depths[5], 3, 1, 6, 2), _stack(depths[5], depths[6], 5, 2, 6, 4), _stack(depths[6], depths[7], 3, 1, 6, 1), nn.Conv2d(depths[7], 1280, 1, padding=0, stride=1, bias=False), BN(1280), nn.ReLU(inplace=True)]
        self.layers = nn.Sequential(*layers)
        self.classifier = nn.Sequential(nn.Dropout(p=dropout, inplace=True), nn.Linear(1280, num_classes))
        self._initialize_weights()

    def forward(self, x):
        x = self.layers(x)
        x = x.mean([2, 3])
        return self.classifier(x)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, mode='fan_out', nonlinearity='sigmoid')
                nn.init.zeros_(m.bias)


class InvertedResidual(nn.Module):

    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup
        self.expand_ratio = expand_ratio
        if expand_ratio == 1:
            self.conv = nn.Sequential(nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False), nn.BatchNorm2d(hidden_dim), nn.ReLU6(inplace=True), nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False), nn.BatchNorm2d(oup))
        else:
            self.conv = nn.Sequential(nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False), nn.BatchNorm2d(hidden_dim), nn.ReLU6(inplace=True), nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False), nn.BatchNorm2d(hidden_dim), nn.ReLU6(inplace=True), nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False), nn.BatchNorm2d(oup))

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


def conv_1x1_bn(inp, oup):
    return nn.Sequential(nn.Conv2d(inp, oup, 1, 1, 0, bias=False), nn.BatchNorm2d(oup), nn.ReLU6(inplace=True))


def conv_bn(inp, oup, stride):
    return nn.Sequential(nn.Conv2d(inp, oup, 3, stride, 1, bias=False), nn.BatchNorm2d(oup), nn.ReLU6(inplace=True))


class MobileNetV2(nn.Module):

    def __init__(self, num_classes=1000, input_size=224, width_mult=1.0, dropout=0.0):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280
        interverted_residual_setting = [[1, 16, 1, 1], [6, 24, 2, 2], [6, 32, 3, 2], [6, 64, 4, 2], [6, 96, 3, 1], [6, 160, 3, 2], [6, 320, 1, 1]]
        assert input_size % 32 == 0
        input_channel = int(input_channel * width_mult)
        self.last_channel = int(last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.features = [conv_bn(3, input_channel, 2)]
        for t, c, n, s in interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                if i == 0:
                    self.features.append(block(input_channel, output_channel, s, expand_ratio=t))
                else:
                    self.features.append(block(input_channel, output_channel, 1, expand_ratio=t))
                input_channel = output_channel
        self.features.append(conv_1x1_bn(input_channel, self.last_channel))
        self.features = nn.Sequential(*self.features)
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(self.last_channel, num_classes))
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.mean([2, 3])
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


class SimpleStemIN(nn.Module):
    """Simple stem for ImageNet."""

    def __init__(self, in_w, out_w):
        super(SimpleStemIN, self).__init__()
        self._construct(in_w, out_w)

    def _construct(self, in_w, out_w):
        self.conv = nn.Conv2d(in_w, out_w, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn = BN(out_w)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x


class SE(nn.Module):
    """Squeeze-and-Excitation (SE) block"""

    def __init__(self, w_in, w_se):
        super(SE, self).__init__()
        self._construct(w_in, w_se)

    def _construct(self, w_in, w_se):
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.f_ex = nn.Sequential(nn.Conv2d(w_in, w_se, kernel_size=1, bias=True), nn.ReLU(inplace=True), nn.Conv2d(w_se, w_in, kernel_size=1, bias=True), nn.Sigmoid())

    def forward(self, x):
        return x * self.f_ex(self.avg_pool(x))


class BottleneckTransform(nn.Module):
    """Bottlenect transformation: 1x1, 3x3, 1x1"""

    def __init__(self, w_in, w_out, stride, bm, gw, se_r):
        super(BottleneckTransform, self).__init__()
        self._construct(w_in, w_out, stride, bm, gw, se_r)

    def _construct(self, w_in, w_out, stride, bm, gw, se_r):
        w_b = int(round(w_out * bm))
        num_gs = w_b // gw
        self.a = nn.Conv2d(w_in, w_b, kernel_size=1, stride=1, padding=0, bias=False)
        self.a_bn = BN(w_b)
        self.a_relu = nn.ReLU(True)
        self.b = nn.Conv2d(w_b, w_b, kernel_size=3, stride=stride, padding=1, groups=num_gs, bias=False)
        self.b_bn = BN(w_b)
        self.b_relu = nn.ReLU(True)
        if se_r:
            w_se = int(round(w_in * se_r))
            self.se = SE(w_b, w_se)
        self.c = nn.Conv2d(w_b, w_out, kernel_size=1, stride=1, padding=0, bias=False)
        self.c_bn = BN(w_out)
        self.c_bn.final_bn = True

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x


class ResBottleneckBlock(nn.Module):
    """Residual bottleneck block: x + F(x), F = bottleneck transform"""

    def __init__(self, w_in, w_out, stride, bm=1.0, gw=1, se_r=None):
        super(ResBottleneckBlock, self).__init__()
        self._construct(w_in, w_out, stride, bm, gw, se_r)

    def _add_skip_proj(self, w_in, w_out, stride):
        self.proj = nn.Conv2d(w_in, w_out, kernel_size=1, stride=stride, padding=0, bias=False)
        self.bn = BN(w_out)

    def _construct(self, w_in, w_out, stride, bm, gw, se_r):
        self.proj_block = w_in != w_out or stride != 1
        if self.proj_block:
            self._add_skip_proj(w_in, w_out, stride)
        self.f = BottleneckTransform(w_in, w_out, stride, bm, gw, se_r)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        if self.proj_block:
            x = self.bn(self.proj(x)) + self.f(x)
        else:
            x = x + self.f(x)
        x = self.relu(x)
        return x


class AnyHead(nn.Module):
    """AnyNet head."""

    def __init__(self, w_in, nc):
        super(AnyHead, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(w_in, nc, bias=True)

    def forward(self, x):
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class AnyStage(nn.Module):
    """AnyNet stage (sequence of blocks w/ the same output shape)."""

    def __init__(self, w_in, w_out, stride, d, block_fun, bm, gw, se_r):
        super(AnyStage, self).__init__()
        self._construct(w_in, w_out, stride, d, block_fun, bm, gw, se_r)

    def _construct(self, w_in, w_out, stride, d, block_fun, bm, gw, se_r):
        for i in range(d):
            b_stride = stride if i == 0 else 1
            b_w_in = w_in if i == 0 else w_out
            self.add_module('b{}'.format(i + 1), block_fun(b_w_in, w_out, b_stride, bm, gw, se_r))

    def forward(self, x):
        for block in self.children():
            x = block(x)
        return x


class AnyNet(nn.Module):
    """AnyNet model."""

    def __init__(self, **kwargs):
        super(AnyNet, self).__init__()
        if kwargs:
            self._construct(stem_w=kwargs['stem_w'], ds=kwargs['ds'], ws=kwargs['ws'], ss=kwargs['ss'], bms=kwargs['bms'], gws=kwargs['gws'], se_r=kwargs['se_r'], nc=kwargs['nc'])
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(mean=0.0, std=math.sqrt(2.0 / fan_out))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 1.0 / float(n))
                m.bias.data.zero_()

    def _construct(self, stem_w, ds, ws, ss, bms, gws, se_r, nc):
        bms = bms if bms else [(1.0) for _d in ds]
        gws = gws if gws else [(1) for _d in ds]
        stage_params = list(zip(ds, ws, ss, bms, gws))
        self.stem = SimpleStemIN(3, stem_w)
        block_fun = ResBottleneckBlock
        prev_w = stem_w
        for i, (d, w, s, bm, gw) in enumerate(stage_params):
            self.add_module('s{}'.format(i + 1), AnyStage(prev_w, w, s, d, block_fun, bm, gw, se_r))
            prev_w = w
        self.head = AnyHead(w_in=prev_w, nc=nc)

    def forward(self, x):
        for module in self.children():
            x = module(x)
        return x


def quantize_float(f, q):
    """Converts a float to closest non-zero int divisible by q."""
    return int(round(f / q) * q)


def adjust_ws_gs_comp(ws, bms, gs):
    """Adjusts the compatibility of widths and groups."""
    ws_bot = [int(w * b) for w, b in zip(ws, bms)]
    gs = [min(g, w_bot) for g, w_bot in zip(gs, ws_bot)]
    ws_bot = [quantize_float(w_bot, g) for w_bot, g in zip(ws_bot, gs)]
    ws = [int(w_bot / b) for w_bot, b in zip(ws_bot, bms)]
    return ws, gs


def generate_regnet(w_a, w_0, w_m, d, q=8):
    """Generates per block ws from RegNet parameters.

    args:
        w_a(float): slope
        w_0(int): initial width
        w_m(float): an additional parameter that controls quantization
        d(int): number of depth
        q(int): the coefficient of division

    procedure:
        1. generate a linear parameterization for block widths. Eql(2)
        2. compute corresponding stage for each block $log_{w_m}^{w_j/w_0}$. Eql(3)
        3. compute per-block width via $w_0*w_m^(s_j)$ and qunatize them that can be divided by q. Eql(4)

    return:
        ws(list of quantized float): quantized width list for blocks in different stages
        num_stages(int): total number of stages
        max_stage(float): the maximal index of stage
        ws_cont(list of float): original width list for blocks in different stages
    """
    assert w_a >= 0 and w_0 > 0 and w_m > 1 and w_0 % q == 0
    ws_cont = np.arange(d) * w_a + w_0
    ks = np.round(np.log(ws_cont / w_0) / np.log(w_m))
    ws = w_0 * np.power(w_m, ks)
    ws = np.round(np.divide(ws, q)) * q
    num_stages, max_stage = len(np.unique(ws)), ks.max() + 1
    ws, ws_cont = ws.astype(int).tolist(), ws_cont.tolist()
    return ws, num_stages, max_stage, ws_cont


def get_stages_from_blocks(ws, rs):
    """Gets ws/ds of network at each stage from per block values."""
    ts_temp = zip(ws + [0], [0] + ws, rs + [0], [0] + rs)
    ts = [(w != wp or r != rp) for w, wp, r, rp in ts_temp]
    s_ws = [w for w, t in zip(ws, ts[:-1]) if t]
    s_ds = np.diff([d for d, t in zip(range(len(ts)), ts) if t]).tolist()
    return s_ws, s_ds


class RegNet(AnyNet):
    """RegNet model class, based on
    `"Designing Network Design Spaces" <https://arxiv.org/abs/2003.13678>`_
    """

    def __init__(self, cfg, bn=None):
        b_ws, num_s, _, _ = generate_regnet(cfg['WA'], cfg['W0'], cfg['WM'], cfg['DEPTH'])
        ws, ds = get_stages_from_blocks(b_ws, b_ws)
        gws = [cfg['GROUP_W'] for _ in range(num_s)]
        bms = [(1) for _ in range(num_s)]
        ws, gws = adjust_ws_gs_comp(ws, bms, gws)
        ss = [(2) for _ in range(num_s)]
        se_r = 0.25 if cfg['SE_ON'] else None
        STEM_W = 32
        global BN
        kwargs = {'stem_w': STEM_W, 'ss': ss, 'ds': ds, 'ws': ws, 'bms': bms, 'gws': gws, 'se_r': se_r, 'nc': 1000}
        super(RegNet, self).__init__(**kwargs)


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, groups=groups, bias=False, dilation=dilation)


class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = BN
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError('Dilation > 1 not supported in BasicBlock')
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.relu2 = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu2(out)
        return out


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Bottleneck(nn.Module):
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = BN
        width = int(planes * (base_width / 64.0)) * groups
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu3 = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu3(out)
        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False, groups=1, width_per_group=64, replace_stride_with_dilation=None, deep_stem=False, avg_down=False):
        super(ResNet, self).__init__()
        global BN
        BN = torch.nn.BatchNorm2d
        norm_layer = BN
        self._norm_layer = norm_layer
        self.inplanes = 64
        self.dilation = 1
        self.deep_stem = deep_stem
        self.avg_down = avg_down
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError('replace_stride_with_dilation should be None or a 3-element tuple, got {}'.format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        if self.deep_stem:
            self.conv1 = nn.Sequential(nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False), norm_layer(32), nn.ReLU(inplace=True), nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False), norm_layer(32), nn.ReLU(inplace=True), nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False))
        else:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            if self.avg_down:
                downsample = nn.Sequential(nn.AvgPool2d(stride, stride=stride, ceil_mode=True, count_include_pad=False), conv1x1(self.inplanes, planes * block.expansion), norm_layer(planes * block.expansion))
            else:
                downsample = nn.Sequential(conv1x1(self.inplanes, planes * block.expansion, stride), norm_layer(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups, base_width=self.base_width, dilation=self.dilation, norm_layer=norm_layer))
        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def forward(self, x):
        return self._forward_impl(x)


class ObserverBase(nn.Module):

    def __init__(self, bit=8, symmetric=False, ch_axis=-1):
        super(ObserverBase, self).__init__()
        self.bit = bit
        self.symmetric = symmetric
        self.ch_axis = ch_axis
        self.eps = torch.tensor(1e-08, dtype=torch.float32)
        if self.symmetric:
            self.quant_min = -2 ** (self.bit - 1)
            self.quant_max = 2 ** (self.bit - 1) - 1
        else:
            self.quant_min = 0
            self.quant_max = 2 ** self.bit - 1
        self.register_buffer('min_val', torch.tensor(float('inf')))
        self.register_buffer('max_val', torch.tensor(float('-inf')))

    def set_bit(self, bit):
        self.bit = bit
        if self.symmetric:
            self.quant_min = -2 ** (self.bit - 1)
            self.quant_max = 2 ** (self.bit - 1) - 1
        else:
            self.quant_min = 0
            self.quant_max = 2 ** self.bit - 1

    def set_name(self, name):
        self.name = name

    @torch.jit.export
    def calculate_qparams(self, min_val, max_val):
        quant_min, quant_max = self.quant_min, self.quant_max
        min_val_neg = torch.min(min_val, torch.zeros_like(min_val))
        max_val_pos = torch.max(max_val, torch.zeros_like(max_val))
        device = min_val_neg.device
        scale = torch.ones(min_val_neg.size(), dtype=torch.float32, device=device)
        zero_point = torch.zeros(min_val_neg.size(), dtype=torch.int, device=device)
        if self.symmetric:
            max_val_pos = torch.max(-min_val_neg, max_val_pos)
            scale = max_val_pos / (float(quant_max - quant_min) / 2)
            scale = torch.max(scale, self.eps)
        else:
            scale = (max_val_pos - min_val_neg) / float(quant_max - quant_min)
            scale = torch.max(scale, self.eps)
            zero_point = quant_min - torch.round(min_val_neg / scale)
            zero_point = torch.clamp(zero_point, quant_min, quant_max)
        return scale, zero_point


class QuantizeBase(nn.Module):

    def __init__(self, observer=ObserverBase, bit=8, symmetric=False, ch_axis=-1):
        super().__init__()
        self.observer = observer(bit=bit, symmetric=symmetric, ch_axis=ch_axis)
        self.bit = bit
        self.symmetric = symmetric
        self.ch_axis = ch_axis
        self.observer_enabled = 0
        self.fake_quant_enabled = 0
        self.quant_min = self.observer.quant_min
        self.quant_max = self.observer.quant_max
        self.drop_prob = 1.0

    def set_bit(self, bit):
        self.observer.set_bit(bit)
        self.bit = bit
        self.quant_min = self.observer.quant_min
        self.quant_max = self.observer.quant_max

    def set_name(self, name):
        self.name = name

    @torch.jit.export
    def calculate_qparams(self):
        return self.observer.calculate_qparams()

    @torch.jit.export
    def disable_observer(self):
        self.observer_enabled = 0

    @torch.jit.export
    def enable_observer(self):
        self.observer_enabled = 1

    @torch.jit.export
    def disable_fake_quant(self):
        self.fake_quant_enabled = 0

    @torch.jit.export
    def enable_fake_quant(self):
        self.fake_quant_enabled = 1

    @torch.jit.export
    def extra_repr(self):
        return 'fake_quant_enabled={}, observer_enabled={}, symmetric={}, bit={}, ch_axis={}, quant_min={}, quant_max={}'.format(self.fake_quant_enabled, self.observer_enabled, self.symmetric, self.bit, self.ch_axis, self.quant_min, self.quant_max)

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        super()._save_to_state_dict(destination, prefix, keep_vars)
        destination[prefix + 'scale'] = self.scale
        destination[prefix + 'zero_point'] = self.zero_point

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        local_state = ['scale', 'zero_point']
        for name in local_state:
            key = prefix + name
            if key in state_dict:
                val = state_dict[key]
                if name == 'scale':
                    if isinstance(self.scale, nn.Parameter):
                        self.scale.data = torch.ones_like(val)
                    else:
                        self.scale.resize_(val.shape)
                else:
                    assert name == 'zero_point'
                    if isinstance(self.zero_point, nn.Parameter):
                        self.zero_point.data = torch.ones_like(val)
                    else:
                        self.zero_point.resize_(val.shape)
                if torch.jit.is_scripting():
                    if name == 'scale':
                        self.scale.copy_(val)
                    else:
                        assert name == 'zero_point'
                        self.zero_point.copy_(val)
            elif strict:
                missing_keys.append(key)
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)


def round_ste(x: 'torch.Tensor'):
    """
    Implement Straight-Through Estimator for rounding operation.
    """
    return (x.round() - x).detach() + x


def fake_quantize_per_channel_affine(x, scale, zero_point, ch_axis, quant_min, quant_max):
    new_shape = [1] * len(x.shape)
    new_shape[ch_axis] = x.shape[ch_axis]
    scale = scale.reshape(new_shape)
    zero_point = zero_point.reshape(new_shape)
    x_int = round_ste(x / scale) + zero_point
    x_quant = torch.clamp(x_int, quant_min, quant_max)
    x_dequant = (x_quant - zero_point) * scale
    return x_dequant


def fake_quantize_per_tensor_affine(x, scale, zero_point, quant_min, quant_max):
    x_int = round_ste(x / scale) + zero_point
    x_quant = torch.clamp(x_int, quant_min, quant_max)
    x_dequant = (x_quant - zero_point) * scale
    return x_dequant


class FixedFakeQuantize(QuantizeBase):

    def __init__(self, observer, bit=8, symmetric=False, ch_axis=-1):
        super().__init__(observer, bit=bit, symmetric=symmetric, ch_axis=ch_axis)
        self.register_buffer('scale', torch.tensor([1.0], dtype=torch.float))
        self.register_buffer('zero_point', torch.tensor([0], dtype=torch.int))
        self.drop_prob = 1.0

    def forward(self, X):
        if self.observer_enabled == 1:
            self.observer(X.detach())
            _scale, _zero_point = self.observer.calculate_qparams(self.observer.min_val, self.observer.max_val)
            _scale, _zero_point = _scale, _zero_point
            if self.scale.shape != _scale.shape:
                self.scale.resize_(_scale.shape)
                self.zero_point.resize_(_zero_point.shape)
            self.scale.copy_(_scale)
            self.zero_point.copy_(_zero_point)
        if self.fake_quant_enabled == 1:
            if self.drop_prob < 1.0:
                x_orig = X
            if self.ch_axis != -1:
                X = fake_quantize_per_channel_affine(X, self.scale.data, self.zero_point.data.int(), self.ch_axis, self.quant_min, self.quant_max)
            else:
                X = fake_quantize_per_tensor_affine(X, self.scale.item(), self.zero_point.item(), self.quant_min, self.quant_max)
            if self.drop_prob < 1.0:
                x_prob = torch.where(torch.rand_like(X) < self.drop_prob, X, x_orig)
                return x_prob
        return X


def grad_scale(t, scale):
    return (t - t * scale).detach() + t * scale


def fake_quantize_learnable_per_channel_affine_training(x, scale, zero_point, ch_axis, quant_min, quant_max, grad_factor):
    new_shape = [1] * len(x.shape)
    new_shape[ch_axis] = x.shape[ch_axis]
    scale = grad_scale(scale, grad_factor).reshape(new_shape)
    zero_point = zero_point.reshape(new_shape)
    x_int = round_ste(x / scale) + zero_point
    x_quant = torch.clamp(x_int, quant_min, quant_max)
    x_dequant = (x_quant - zero_point) * scale
    return x_dequant


def fake_quantize_learnable_per_tensor_affine_training(x, scale, zero_point, quant_min, quant_max, grad_factor):
    scale = grad_scale(scale, grad_factor)
    x_int = round_ste(x / scale) + zero_point
    x_quant = torch.clamp(x_int, quant_min, quant_max)
    x_dequant = (x_quant - zero_point) * scale
    return x_dequant


class LSQFakeQuantize(QuantizeBase):

    def __init__(self, observer, bit=8, symmetric=False, ch_axis=-1, use_grad_scaling=True):
        super().__init__(observer, bit=bit, symmetric=symmetric, ch_axis=ch_axis)
        self.scale = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float))
        self.register_buffer('zero_point', torch.tensor([0], dtype=torch.int))
        self.register_buffer('eps', torch.tensor([torch.finfo(torch.float32).eps]))
        self.use_grad_scaling = use_grad_scaling
        self.drop_prob = 1.0

    def forward(self, X):
        if self.observer_enabled == 1:
            self.observer(X.detach())
            _scale, _zero_point = self.observer.calculate_qparams(self.observer.min_val, self.observer.max_val)
            _scale, _zero_point = _scale, _zero_point
            if self.ch_axis != -1:
                self.scale.data = torch.ones_like(_scale)
                self.zero_point.resize_(_zero_point.shape)
            self.scale.data.copy_(_scale)
            self.zero_point.copy_(_zero_point)
        else:
            self.scale.data.abs_()
            self.scale.data.clamp_(min=self.eps.item())
        if self.fake_quant_enabled == 1:
            if self.drop_prob < 1.0:
                x_orig = X
            if self.ch_axis != -1:
                if self.use_grad_scaling:
                    grad_factor = 1.0 / (X.numel() / X.shape[self.ch_axis] * self.quant_max) ** 0.5
                else:
                    grad_factor = 1.0
                X = fake_quantize_learnable_per_channel_affine_training(X, self.scale, self.zero_point.data.int(), self.ch_axis, self.quant_min, self.quant_max, grad_factor)
            else:
                if self.use_grad_scaling:
                    grad_factor = 1.0 / (X.numel() * self.quant_max) ** 0.5
                else:
                    grad_factor = 1.0
                X = fake_quantize_learnable_per_tensor_affine_training(X, self.scale, self.zero_point.item(), self.quant_min, self.quant_max, grad_factor)
            if self.drop_prob < 1.0:
                x_prob = torch.where(torch.rand_like(X) < self.drop_prob, X, x_orig)
                return x_prob
        return X


def fake_quantize_learnableplus_per_channel_affine_training(x, scale, zero_point, ch_axis, quant_min, quant_max, grad_factor):
    zero_point = (zero_point.round() - zero_point).detach() + zero_point
    new_shape = [1] * len(x.shape)
    new_shape[ch_axis] = x.shape[ch_axis]
    scale = grad_scale(scale, grad_factor).reshape(new_shape)
    zero_point = grad_scale(zero_point, grad_factor).reshape(new_shape)
    x_int = round_ste(x / scale) + zero_point
    x_quant = torch.clamp(x_int, quant_min, quant_max)
    x_dequant = (x_quant - zero_point) * scale
    return x_dequant


def fake_quantize_learnableplus_per_tensor_affine_training(x, scale, zero_point, quant_min, quant_max, grad_factor):
    zero_point = (zero_point.round() - zero_point).detach() + zero_point
    scale = grad_scale(scale, grad_factor)
    zero_point = grad_scale(zero_point, grad_factor)
    x_int = round_ste(x / scale) + zero_point
    x_quant = torch.clamp(x_int, quant_min, quant_max)
    x_dequant = (x_quant - zero_point) * scale
    return x_dequant


class LSQPlusFakeQuantize(QuantizeBase):

    def __init__(self, observer, bit=8, symmetric=False, ch_axis=-1, use_grad_scaling=True):
        super().__init__(observer, bit=bit, symmetric=symmetric, ch_axis=ch_axis)
        self.scale = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float))
        self.zero_point = torch.nn.Parameter(torch.tensor([0.0], dtype=torch.float))
        self.register_buffer('eps', torch.tensor([torch.finfo(torch.float32).eps]))
        self.use_grad_scaling = use_grad_scaling
        self.drop_prob = 1.0

    def forward(self, X):
        if self.observer_enabled == 1:
            self.observer(X.detach())
            _scale, _zero_point = self.observer.calculate_qparams(self.observer.min_val, self.observer.max_val)
            _scale, _zero_point = _scale, _zero_point
            if self.ch_axis != -1:
                self.scale.data = torch.ones_like(_scale)
                self.zero_point.data = torch.zeros_like(_zero_point.float())
            self.scale.data.copy_(_scale)
            self.zero_point.data.copy_(_zero_point.float())
        else:
            self.scale.data.abs_()
            self.scale.data.clamp_(min=self.eps.item())
            self.zero_point.data.clamp_(self.quant_min, self.quant_max).float()
        if self.fake_quant_enabled == 1:
            if self.drop_prob < 1.0:
                x_orig = X
            if self.ch_axis != -1:
                if self.use_grad_scaling:
                    grad_factor = 1.0 / (X.numel() / X.shape[self.ch_axis] * self.quant_max) ** 0.5
                else:
                    grad_factor = 1.0
                X = fake_quantize_learnableplus_per_channel_affine_training(X, self.scale, self.zero_point, self.ch_axis, self.quant_min, self.quant_max, grad_factor)
            else:
                if self.use_grad_scaling:
                    grad_factor = 1.0 / (X.numel() * self.quant_max) ** 0.5
                else:
                    grad_factor = 1.0
                X = fake_quantize_learnableplus_per_tensor_affine_training(X, self.scale, self.zero_point, self.quant_min, self.quant_max, grad_factor)
            if self.drop_prob < 1.0:
                x_prob = torch.where(torch.rand_like(X) < self.drop_prob, X, x_orig)
                return x_prob
        return X


class AdaRoundFakeQuantize(QuantizeBase):
    """
    self.adaround=True: turn on up or down forward
    self.adaround=False: turn on round-to-nearest forward
    based on the FixedFakeQuantize
    """

    def __init__(self, observer, bit=8, symmetric=False, ch_axis=-1):
        super().__init__(observer, bit=bit, symmetric=symmetric, ch_axis=ch_axis)
        self.register_buffer('scale', torch.tensor([1.0], dtype=torch.float))
        self.register_buffer('zero_point', torch.tensor([0], dtype=torch.int))
        self.adaround = False
        self.gamma, self.zeta = -0.1, 1.1

    def init(self, weight_tensor: 'torch.Tensor', round_mode):
        self.adaround = True
        self.round_mode = round_mode
        self.init_alpha(x=weight_tensor.data.clone().detach())

    def init_alpha(self, x: 'torch.Tensor'):
        if self.ch_axis != -1:
            new_shape = [1] * len(x.shape)
            new_shape[self.ch_axis] = x.shape[self.ch_axis]
            scale = self.scale.data.reshape(new_shape)
        else:
            scale = self.scale.data
        x_floor = torch.floor(x / scale)
        if self.round_mode == 'learned_hard_sigmoid':
            rest = x / scale - x_floor
            alpha = -torch.log((self.zeta - self.gamma) / (rest - self.gamma) - 1)
            self.alpha = torch.nn.Parameter(alpha)
        else:
            raise NotImplementedError

    def rectified_sigmoid(self):
        """generate rounding mask.
        """
        return ((self.zeta - self.gamma) * torch.sigmoid(self.alpha) + self.gamma).clamp(0, 1)

    def adaround_forward(self, X, hard_value=False):
        if self.ch_axis != -1:
            new_shape = [1] * len(X.shape)
            new_shape[self.ch_axis] = X.shape[self.ch_axis]
            scale = self.scale.data.reshape(new_shape)
            zero_point = self.zero_point.data.int().reshape(new_shape)
        else:
            scale = self.scale.item()
            zero_point = self.zero_point.item()
        X = torch.floor(X / scale)
        if hard_value:
            X += (self.alpha >= 0).float()
        else:
            X += self.rectified_sigmoid()
        X += zero_point
        X = torch.clamp(X, self.quant_min, self.quant_max)
        X = (X - zero_point) * scale
        return X

    def get_hard_value(self, X):
        X = self.adaround_forward(X, hard_value=True)
        return X

    def forward(self, X):
        if self.observer_enabled == 1:
            self.observer(X.detach())
            _scale, _zero_point = self.observer.calculate_qparams(self.observer.min_val, self.observer.max_val)
            _scale, _zero_point = _scale, _zero_point
            if self.scale.shape != _scale.shape:
                self.scale.resize_(_scale.shape)
                self.zero_point.resize_(_zero_point.shape)
            self.scale.copy_(_scale)
            self.zero_point.copy_(_zero_point)
        if self.fake_quant_enabled == 1:
            if not self.adaround:
                if self.ch_axis != -1:
                    X = fake_quantize_per_channel_affine(X, self.scale.data, self.zero_point.data.int(), self.ch_axis, self.quant_min, self.quant_max)
                else:
                    X = fake_quantize_per_tensor_affine(X, self.scale.item(), self.zero_point.item(), self.quant_min, self.quant_max)
            else:
                if not hasattr(self, 'alpha'):
                    raise NotImplementedError
                if self.round_mode == 'learned_hard_sigmoid':
                    X = self.adaround_forward(X)
                else:
                    raise NotImplementedError
        return X


def _transform_to_ch_axis(x, ch_axis):
    if ch_axis == -1:
        return x
    else:
        x_dim = x.size()
        new_axis_list = [i for i in range(len(x_dim))]
        new_axis_list[ch_axis] = 0
        new_axis_list[0] = ch_axis
        x_channel = x.permute(new_axis_list)
        y = torch.flatten(x_channel, start_dim=1)
        return y


class MinMaxObserver(ObserverBase):
    """
    Calculate minmax of whole calibration dataset.
    """

    def __init__(self, bit=8, symmetric=False, ch_axis=-1):
        super(MinMaxObserver, self).__init__(bit=bit, symmetric=symmetric, ch_axis=ch_axis)

    def forward(self, x_orig):
        """Records the running minimum and maximum of ``x``."""
        if x_orig.numel() == 0:
            return x_orig
        x = x_orig.clone().detach()
        if self.ch_axis == -1:
            min_val_cur, max_val_cur = torch._aminmax(x)
        else:
            y = _transform_to_ch_axis(x, self.ch_axis)
            min_val_cur, max_val_cur = torch._aminmax(y, 1)
        self.min_val = torch.min(self.min_val, min_val_cur)
        self.max_val = torch.max(self.max_val, max_val_cur)


class AvgMinMaxObserver(ObserverBase):
    """
    Average min/max calibration
    """

    def __init__(self, bit=8, symmetric=False, ch_axis=-1):
        super(AvgMinMaxObserver, self).__init__(bit=bit, symmetric=symmetric, ch_axis=ch_axis)
        self.cnt = 0
        assert self.ch_axis == -1

    def forward(self, x_orig):
        """Records the average minimum and maximum of ``x``."""
        if x_orig.numel() == 0:
            return x_orig
        x = x_orig.clone().detach()
        if self.ch_axis == -1:
            min_val_cur, max_val_cur = torch._aminmax(x)
        if self.max_val.numel() <= 1 and self.max_val.isinf():
            self.min_val = min_val_cur
            self.max_val = max_val_cur
        else:
            self.min_val = self.min_val * self.cnt + min_val_cur
            self.max_val = self.max_val * self.cnt + max_val_cur
        self.cnt += 1
        self.min_val /= self.cnt
        self.max_val /= self.cnt


class MSEObserver(ObserverBase):

    def __init__(self, bit=8, symmetric=False, ch_axis=-1):
        super(MSEObserver, self).__init__(bit=bit, symmetric=symmetric, ch_axis=ch_axis)
        self.p = 2.4
        self.num = 100
        self.one_side_dist = None

    def lp_loss(self, pred, tgt, p=2.0):
        x = (pred - tgt).abs().pow(p)
        if self.ch_axis == -1:
            return x.mean()
        else:
            y = _transform_to_ch_axis(x, self.ch_axis)
            return y.mean(1)

    def loss_fx(self, x, new_min, new_max):
        scale, zero_point = self.calculate_qparams(new_min, new_max)
        if self.ch_axis != -1:
            x_q = fake_quantize_per_channel_affine(x, scale.data, zero_point.data.int(), self.ch_axis, self.quant_min, self.quant_max)
        else:
            x_q = fake_quantize_per_tensor_affine(x, scale.item(), int(zero_point.item()), self.quant_min, self.quant_max)
        score = self.lp_loss(x_q, x, p=self.p)
        return score

    def perform_2D_search(self, x):
        if self.ch_axis != -1:
            y = _transform_to_ch_axis(x, self.ch_axis)
            x_min, x_max = torch._aminmax(y, 1)
            x_max = torch.max(x_max, torch.zeros_like(x_max))
            x_min = torch.min(x_min, torch.zeros_like(x_min))
        else:
            x_min, x_max = torch._aminmax(x)
        xrange = x_max - x_min
        best_score = torch.zeros_like(x_min) + 10000000000.0
        best_min = x_min.clone()
        best_max = x_max.clone()
        for i in range(1, self.num + 1):
            tmp_min = torch.zeros_like(x_min)
            tmp_max = xrange / self.num * i
            tmp_delta = (tmp_max - tmp_min) / float(self.quant_max - self.quant_min)
            for zp in range(self.quant_min, self.quant_max + 1):
                new_min = tmp_min - zp * tmp_delta
                new_max = tmp_max - zp * tmp_delta
                score = self.loss_fx(x, new_min, new_max)
                best_min = torch.where(score < best_score, new_min, best_min)
                best_max = torch.where(score < best_score, new_max, best_max)
                best_score = torch.min(best_score, score)
        return best_min, best_max

    def perform_1D_search(self, x):
        if self.ch_axis != -1:
            y = _transform_to_ch_axis(x, self.ch_axis)
            x_min, x_max = torch._aminmax(y, 1)
        else:
            x_min, x_max = torch._aminmax(x)
        xrange = torch.max(x_min.abs(), x_max)
        best_score = torch.zeros_like(x_min) + 10000000000.0
        best_min = x_min.clone()
        best_max = x_max.clone()
        for i in range(1, self.num + 1):
            thres = xrange / self.num * i
            new_min = torch.zeros_like(x_min) if self.one_side_dist == 'pos' else -thres
            new_max = torch.zeros_like(x_max) if self.one_side_dist == 'neg' else thres
            score = self.loss_fx(x, new_min, new_max)
            best_min = torch.where(score < best_score, new_min, best_min)
            best_max = torch.where(score < best_score, new_max, best_max)
            best_score = torch.min(score, best_score)
        return best_min, best_max

    def forward(self, x_orig):
        if x_orig.numel() == 0:
            return x_orig
        x = x_orig.clone().detach()
        if self.one_side_dist is None:
            self.one_side_dist = 'pos' if x.min() >= 0.0 else 'neg' if x.max() <= 0.0 else 'no'
        if self.one_side_dist != 'no' or self.symmetric:
            best_min, best_max = self.perform_1D_search(x)
        else:
            best_min, best_max = self.perform_2D_search(x)
        self.min_val = torch.min(self.min_val, best_min)
        self.max_val = torch.max(self.max_val, best_max)


class AvgMSEObserver(MSEObserver):

    def __init__(self, bit=8, symmetric=False, ch_axis=-1):
        super(AvgMSEObserver, self).__init__(bit=bit, symmetric=symmetric, ch_axis=ch_axis)
        self.cnt = 0
        assert self.ch_axis == -1

    def forward(self, x_orig):
        if x_orig.numel() == 0:
            return x_orig
        x = x_orig.clone().detach()
        if self.one_side_dist is None:
            self.one_side_dist = 'pos' if x.min() >= 0.0 else 'neg' if x.max() <= 0.0 else 'no'
        if self.one_side_dist != 'no' or self.symmetric:
            best_min, best_max = self.perform_1D_search(x)
        else:
            best_min, best_max = self.perform_2D_search(x)
        if self.max_val.numel() <= 1 and self.max_val.isinf():
            self.min_val = best_min
            self.max_val = best_max
        else:
            self.min_val = self.min_val * self.cnt + best_min
            self.max_val = self.max_val * self.cnt + best_max
        self.cnt += 1
        self.min_val /= self.cnt
        self.max_val /= self.cnt


class MSEFastObserver(ObserverBase):

    def __init__(self, bit=8, symmetric=False, ch_axis=-1):
        super(MSEFastObserver, self).__init__(bit=bit, symmetric=symmetric, ch_axis=ch_axis)
        self.p = 2.4
        self.num = 100
        self.one_side_dist = None

    def lp_loss(self, pred, tgt, p=2.0):
        return (pred - tgt).abs().pow(p).mean()

    def loss_fx(self, x, new_min, new_max):
        new_min = torch.tensor(new_min)
        new_max = torch.tensor(new_max)
        scale, zero_point = self.calculate_qparams(new_min, new_max)
        x_q = fake_quantize_per_tensor_affine(x, scale.item(), int(zero_point.item()), self.quant_min, self.quant_max)
        score = self.lp_loss(x_q, x, p=self.p)
        return score

    def golden_asym_shift_loss(self, shift, xrange, x, x_min, x_max):
        tmp_min = 0.0
        tmp_max = xrange
        new_min = tmp_min - shift
        new_max = tmp_max - shift
        return self.loss_fx(x, new_min, new_max).cpu().numpy()

    def golden_asym_range_loss(self, xrange, x, x_min, x_max):
        tmp_delta = xrange / float(self.quant_max - self.quant_min)
        max_shift = tmp_delta * self.quant_max
        min_shift = tmp_delta * self.quant_min
        result = minimize_scalar(self.golden_asym_shift_loss, args=(xrange, x, x_min, x_max), bounds=(min_shift, max_shift), method='Bounded')
        return result.fun

    def golden_sym_range_loss(self, xrange, x):
        new_min = 0.0 if self.one_side_dist == 'pos' else -xrange
        new_max = 0.0 if self.one_side_dist == 'neg' else xrange
        return self.loss_fx(x, new_min, new_max).cpu().numpy()

    def golden_section_search_2D_channel(self, x, x_min, x_max):
        xrange = x_max - x_min
        result = minimize_scalar(self.golden_asym_range_loss, args=(x, x_min, x_max), bounds=(min(0.1, 0.01 * xrange.item()), xrange.item()), method='Bounded')
        final_range = result.x
        tmp_min = 0.0
        tmp_max = final_range
        tmp_delta = final_range / float(self.quant_max - self.quant_min)
        max_shift = tmp_delta * self.quant_max
        min_shift = tmp_delta * self.quant_min
        subresult = minimize_scalar(self.golden_asym_shift_loss, args=(final_range, x, x_min, x_max), bounds=(min_shift, max_shift), method='Bounded')
        final_shift = subresult.x
        best_min = max(tmp_min - final_shift, x_min)
        best_max = min(tmp_max - final_shift, x_max)
        return torch.tensor(best_min), torch.tensor(best_max)

    def golden_section_search_1D_channel(self, x, x_min, x_max):
        xrange = torch.max(x_min.abs(), x_max)
        result = minimize_scalar(self.golden_sym_range_loss, args=(x,), bounds=(min(0.1, 0.01 * xrange.item()), xrange.item()), method='Bounded')
        final_range = result.x
        best_min = torch.zeros_like(x_min) if self.one_side_dist == 'pos' else -torch.tensor(final_range)
        best_max = torch.zeros_like(x_max) if self.one_side_dist == 'neg' else torch.tensor(final_range)
        return torch.tensor(best_min), torch.tensor(best_max)

    def golden_section_2D_search(self, x):
        if self.ch_axis == -1:
            x_min, x_max = torch._aminmax(x)
            x_min, x_max = self.golden_section_search_2D_channel(x, x_min, x_max)
        else:
            y = _transform_to_ch_axis(x, self.ch_axis)
            x_min, x_max = torch._aminmax(y, 1)
            for ch, val in enumerate(y):
                x_min[ch], x_max[ch] = self.golden_section_search_2D_channel(y[ch], x_min[ch], x_max[ch])
        return x_min, x_max

    def golden_section_1D_search(self, x):
        if self.ch_axis == -1:
            x_min, x_max = torch._aminmax(x)
            x_min, x_max = self.golden_section_search_1D_channel(x, x_min, x_max)
        else:
            y = _transform_to_ch_axis(x, self.ch_axis)
            x_min, x_max = torch._aminmax(y, 1)
            for ch, val in enumerate(y):
                x_min[ch], x_max[ch] = self.golden_section_search_1D_channel(y[ch], x_min[ch], x_max[ch])
        return x_min, x_max

    def forward(self, x_orig):
        if x_orig.numel() == 0:
            return x_orig
        x = x_orig.clone().detach()
        if self.one_side_dist is None:
            self.one_side_dist = 'pos' if x.min() >= 0.0 else 'neg' if x.max() <= 0.0 else 'no'
        if self.one_side_dist != 'no' or self.symmetric:
            best_min, best_max = self.golden_section_1D_search(x)
        else:
            best_min, best_max = self.golden_section_2D_search(x)
        self.min_val = torch.min(self.min_val, best_min)
        self.max_val = torch.max(self.max_val, best_max)


class AvgMSEFastObserver(MSEFastObserver):

    def __init__(self, bit=8, symmetric=False, ch_axis=-1):
        super().__init__(bit=bit, symmetric=symmetric, ch_axis=ch_axis)
        self.cnt = 0
        assert self.ch_axis == -1

    def forward(self, x_orig):
        if x_orig.numel() == 0:
            return x_orig
        x = x_orig.clone().detach()
        if self.one_side_dist is None:
            self.one_side_dist = 'pos' if x.min() >= 0.0 else 'neg' if x.max() <= 0.0 else 'no'
        if self.one_side_dist != 'no' or self.symmetric:
            best_min, best_max = self.golden_section_1D_search(x)
        else:
            best_min, best_max = self.golden_section_2D_search(x)
        if self.max_val.numel() <= 1 and self.max_val.isinf():
            self.min_val = best_min
            self.max_val = best_max
        else:
            self.min_val = self.min_val * self.cnt + best_min
            self.max_val = self.max_val * self.cnt + best_max
        self.cnt += 1
        self.min_val /= self.cnt
        self.max_val /= self.cnt


class QuantizedOperator:
    pass


FakeQuantizeDict = {'FixedFakeQuantize': FixedFakeQuantize, 'LSQFakeQuantize': LSQFakeQuantize, 'LSQPlusFakeQuantize': LSQPlusFakeQuantize, 'AdaRoundFakeQuantize': AdaRoundFakeQuantize}


ObserverDict = {'MinMaxObserver': MinMaxObserver, 'AvgMinMaxObserver': AvgMinMaxObserver, 'MSEObserver': MSEObserver, 'AvgMSEObserver': AvgMSEObserver, 'MSEFastObserver': MSEFastObserver, 'AvgMSEFastObserver': AvgMSEFastObserver}


def WeightQuantizer(w_qconfig):
    return FakeQuantizeDict[w_qconfig.quantizer](ObserverDict[w_qconfig.observer], bit=w_qconfig.bit, symmetric=w_qconfig.symmetric, ch_axis=w_qconfig.ch_axis)


class QConv2d(QuantizedOperator, nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, w_qconfig):
        super().__init__(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode)
        self.weight_fake_quant = WeightQuantizer(w_qconfig)

    def forward(self, input):
        return self._conv_forward(input, self.weight_fake_quant(self.weight), self.bias)


class QLinear(QuantizedOperator, nn.Linear):

    def __init__(self, in_features, out_features, bias, w_qconfig):
        super().__init__(in_features=in_features, out_features=out_features, bias=bias)
        self.weight_fake_quant = WeightQuantizer(w_qconfig)

    def forward(self, input):
        return F.linear(input, self.weight_fake_quant(self.weight), self.bias)


class QEmbedding(QuantizedOperator, nn.Embedding):

    def __init__(self, num_embeddings, embedding_dim, padding_idx, max_norm, norm_type, scale_grad_by_freq, sparse, _weight, w_qconfig):
        super().__init__(num_embeddings=num_embeddings, embedding_dim=embedding_dim, padding_idx=padding_idx, max_norm=max_norm, norm_type=norm_type, scale_grad_by_freq=scale_grad_by_freq, sparse=sparse, _weight=_weight)
        self.weight_fake_quant = WeightQuantizer(w_qconfig)

    def forward(self, input):
        return F.embedding(input, self.weight_fake_quant(self.weight), self.padding_idx, self.max_norm, self.norm_type, self.scale_grad_by_freq, self.sparse)


class QuantizedModule(nn.Module):

    def __init__(self):
        super().__init__()


def ActivationQuantizer(a_qconfig):
    return FakeQuantizeDict[a_qconfig.quantizer](ObserverDict[a_qconfig.observer], bit=a_qconfig.bit, symmetric=a_qconfig.symmetric, ch_axis=a_qconfig.ch_axis)


def get_module_args(module):
    if isinstance(module, nn.Linear):
        return dict(in_features=module.in_features, out_features=module.out_features, bias=module.bias is not None)
    elif isinstance(module, nn.Conv2d):
        return dict(in_channels=module.in_channels, out_channels=module.out_channels, kernel_size=module.kernel_size, stride=module.stride, padding=module.padding, dilation=module.dilation, groups=module.groups, bias=module.bias is not None, padding_mode=module.padding_mode)
    elif isinstance(module, nn.Embedding):
        return dict(num_embeddings=module.num_embeddings, embedding_dim=module.embedding_dim, padding_idx=module.padding_idx, max_norm=module.max_norm, norm_type=module.norm_type, scale_grad_by_freq=module.scale_grad_by_freq, sparse=module.sparse, _weight=None)
    else:
        raise NotImplementedError


module_type_to_quant_weight = {nn.Linear: QLinear, nn.Conv2d: QConv2d, nn.Embedding: QEmbedding}


def Quantizer(module, config):
    if module is None:
        return ActivationQuantizer(a_qconfig=config)
    module_type = type(module)
    if module_type in module_type_to_quant_weight:
        kwargs = get_module_args(module)
        qmodule = module_type_to_quant_weight[module_type](**kwargs, w_qconfig=config)
        qmodule.weight.data = module.weight.data.clone()
        if getattr(module, 'bias', None) is not None:
            qmodule.bias.data = module.bias.data.clone()
        return qmodule
    return module


class QuantizedLayer(QuantizedModule):

    def __init__(self, module, activation, w_qconfig, a_qconfig, qoutput=True):
        super().__init__()
        self.qoutput = qoutput
        self.module = Quantizer(module, w_qconfig)
        self.activation = activation
        if qoutput:
            self.layer_post_act_fake_quantize = Quantizer(None, a_qconfig)

    def forward(self, x):
        x = self.module(x)
        if self.activation is not None:
            x = self.activation(x)
        if self.qoutput:
            x = self.layer_post_act_fake_quantize(x)
        return x


class QuantizedBlock(QuantizedModule):

    def __init__(self):
        super().__init__()


class StraightThrough(nn.Module):

    def __int__(self):
        super().__init__()

    def forward(self, input):
        return input


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (AnyHead,
     lambda: ([], {'w_in': 4, 'nc': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (AnyNet,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (AnyStage,
     lambda: ([], {'w_in': 4, 'w_out': 4, 'stride': 1, 'd': 4, 'block_fun': torch.nn.ReLU, 'bm': 4, 'gw': 4, 'se_r': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (AvgMSEFastObserver,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (AvgMSEObserver,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (AvgMinMaxObserver,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (BasicBlock,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (BottleneckTransform,
     lambda: ([], {'w_in': 4, 'w_out': 4, 'stride': 1, 'bm': 4, 'gw': 4, 'se_r': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (InvertedResidual,
     lambda: ([], {'inp': 4, 'oup': 4, 'stride': 1, 'expand_ratio': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (MNASNet,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {})),
    (MSEFastObserver,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (MSEObserver,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (MinMaxObserver,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (MobileNetV2,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {})),
    (ResBottleneckBlock,
     lambda: ([], {'w_in': 4, 'w_out': 4, 'stride': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (SE,
     lambda: ([], {'w_in': 4, 'w_se': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (SimpleStemIN,
     lambda: ([], {'in_w': 4, 'out_w': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (StraightThrough,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (_InvertedResidual,
     lambda: ([], {'in_ch': 4, 'out_ch': 4, 'kernel_size': 3, 'stride': 1, 'expansion_factor': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
]

