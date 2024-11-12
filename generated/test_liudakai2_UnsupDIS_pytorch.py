
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


import time


import numpy as np


import torch


import math


import torch.nn as nn


import torch.nn.functional as F


from torch.autograd import Variable


from collections import OrderedDict


import logging


from copy import deepcopy


from collections import Iterable


import torch.distributed as dist


import torch.optim as optim


import torch.optim.lr_scheduler as lr_scheduler


from torch.cuda import amp


from torch.nn.parallel import DistributedDataParallel as DDP


import random


from time import time


from torch.utils.data import Dataset


import re


import torchvision


import torch.backends.cudnn as cudnn


def autopad(k, p=None):
    if p is None:
        p = k // 2 if isinstance(k, int) else [(x // 2) for x in k]
    return p


def forward_norm_layer(x, norm, norm_type):
    if norm_type == 'HIN':
        x1, x2 = torch.chunk(x, 2, dim=1)
        x2 = norm(x2)
        x = torch.cat((x1, x2), dim=1)
    else:
        x = norm(x)
    return x


def parse_norm_layer(norm, c2):
    if norm == 'BN':
        m = nn.BatchNorm2d(c2)
    elif norm == 'LN':
        m = nn.GroupNorm(1, c2)
    elif norm == 'IN':
        m = nn.InstanceNorm2d(c2)
    elif norm == 'HIN':
        m = nn.InstanceNorm2d(c2 // 2)
    else:
        m = nn.Identity()
    return m


class Conv(nn.Module):

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True, d=1, norm='BN'):
        super(Conv, self).__init__()
        no_norm = norm == 'None' or norm is None
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), dilation=d, groups=g, bias=no_norm)
        self.norm_type = norm
        self.bn = parse_norm_layer(norm, c2)
        if act is True:
            self.act = nn.ReLU(inplace=True) if no_norm else nn.SiLU()
        else:
            self.act = act if isinstance(act, nn.Module) else nn.Identity()
        if no_norm:
            nn.init.xavier_normal_(self.conv.weight, gain=2.0)

    def forward(self, x):
        out = self.conv(x)
        out = forward_norm_layer(out, self.bn, self.norm_type)
        out = self.act(out)
        return out

    def fuseforward(self, x):
        return self.act(self.conv(x))


class Resizer(nn.Module):

    def __init__(self, size, shape, mode='nearest'):
        super(Resizer, self).__init__()
        self.size = size
        self.mode = mode
        assert shape == 'square'
        self.shape = shape

    def forward(self, x):
        if isinstance(x, torch.Tensor):
            size = self.size, self.size
        else:
            x, template = x
            size = template.shape[-2:]
        return F.interpolate(x, mode=self.mode, size=size, align_corners=False if self.mode == 'bilinear' else None)


class ResBlock(nn.Module):

    def __init__(self, c1, c2, norm='None'):
        super(ResBlock, self).__init__()
        self.body = nn.Sequential(Conv(c1, c1, k=3, s=1, norm=norm), nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1, bias=True))
        self.shortcut = nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1, bias=False) if c1 != c2 else nn.Identity()
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.act(self.shortcut(x) + self.body(x))
        return x


class Add(nn.Module):

    def __init__(self, act=False):
        super(Add, self).__init__()
        self.act = nn.ReLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(sum(x))


class FuseBlock(nn.Module):

    def __init__(self, c1, c2, bn=False):
        super(FuseBlock, self).__init__()
        self.conv1 = nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(c1) if bn else nn.Identity()
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.conv2(x)
        return x


class TransformerLayer(nn.Module):

    def __init__(self, c, num_heads, norm=False):
        super().__init__()
        self.q = nn.Linear(c, c, bias=False)
        self.k = nn.Linear(c, c, bias=False)
        self.v = nn.Linear(c, c, bias=False)
        self.ma = nn.MultiheadAttention(embed_dim=c, num_heads=num_heads)
        self.fc1 = nn.Linear(c, c, bias=False)
        self.fc2 = nn.Linear(c, c, bias=False)
        if norm:
            self.norm1 = nn.LayerNorm(c)
            self.norm2 = nn.LayerNorm(c)
        else:
            self.norm1 = None
            self.norm2 = None

    def forward(self, x, pos_embed=None):
        if pos_embed is not None:
            q, k, v = self.q(x), self.k(x + pos_embed), self.v(x + pos_embed)
        else:
            q, k, v = self.q(x), self.k(x), self.v(x)
        x = self.ma(q, k, v)[0] + x
        if self.norm1 is not None:
            x = self.norm1(x)
        x = self.fc2(self.fc1(x)) + x
        if self.norm2 is not None:
            x = self.norm2(x)
        return x


class TransformerBlock(nn.Module):

    def __init__(self, c1, c2, num_heads, num_layers, norm='BN'):
        super().__init__()
        self.conv = None
        if c1 != c2:
            self.conv = Conv(c1, c2, norm=norm)
        self.linear = nn.Linear(c2, c2)
        self.tr = nn.Sequential(*[TransformerLayer(c2, num_heads) for _ in range(num_layers)])
        self.c2 = c2

    def forward(self, x):
        if self.conv is not None:
            x = self.conv(x)
        b, _, w, h = x.shape
        p = x.flatten(2)
        p = p.unsqueeze(0)
        p = p.transpose(0, 3)
        p = p.squeeze(3)
        e = self.linear(p)
        x = p + e
        x = self.tr(x)
        x = x.unsqueeze(3)
        x = x.transpose(0, 3)
        x = x.reshape(b, self.c2, w, h)
        return x


class ASFF(nn.Module):

    def __init__(self, ch1, c2, e=0.5, norm='BN'):
        super().__init__()
        assert tuple(ch1) == tuple(sorted(ch1))
        self.num_layers = len(ch1)
        self.c = int(c2 * e)
        self.c_reducer = nn.ModuleList(Conv(x, self.c, 3, 1, norm=norm) for x in ch1)
        self.adapter = nn.Conv2d(self.c * self.num_layers, self.num_layers, kernel_size=3, stride=1, padding=1, bias=False)
        self.c_expander = Conv(self.c, c2, 1, norm=norm)

    def forward(self, x):
        assert len(x) == self.num_layers
        bs, c0, h, w = x[0].shape
        for i in range(self.num_layers):
            x[i] = self.c_reducer[i](x[i])
            scale_factor = w // x[i].shape[-1]
            if scale_factor > 1:
                x[i] = nn.Upsample(size=None, scale_factor=scale_factor, mode='nearest')(x[i])
        layer_weights = self.adapter(torch.cat(x, dim=1)).chunk(self.num_layers, dim=1)
        fused = sum([(x[i] * layer_weights[i]) for i in range(self.num_layers)])
        out = self.c_expander(fused)
        return out


class TransformerEncoder(nn.Module):

    def __init__(self, c, num_heads, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.tr = nn.ModuleList([TransformerLayer(c, num_heads, norm=False) for _ in range(num_layers)])
        self.c = c

    def forward(self, x):
        x, pos, pos_embed = x
        for i in range(self.num_layers):
            x = self.tr[i](x, pos_embed)
        return self._organize_shape(x), self._organize_shape(pos)

    @staticmethod
    def _organize_shape(x):
        p = x.transpose(0, 1)
        p = p.transpose(1, 2)
        p = p.unsqueeze(2)
        return p


class Bottleneck(nn.Module):

    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5, norm='BN'):
        super(Bottleneck, self).__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1, norm=norm)
        self.cv2 = Conv(c_, c2, 3, 1, g=g, norm=norm)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, norm='BN'):
        super(BottleneckCSP, self).__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1, norm=norm)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1, norm=norm)
        self.bn = parse_norm_layer(norm, 2 * c_)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0, norm=norm) for _ in range(n)])
        self.norm_type = norm

    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        y = torch.cat((y1, y2), dim=1)
        y = forward_norm_layer(y, self.bn, self.norm_type)
        y = self.act(y)
        return self.cv4(y)


class C3(nn.Module):

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, norm='BN'):
        super(C3, self).__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1, norm=norm)
        self.cv2 = Conv(c1, c_, 1, 1, norm=norm)
        self.cv3 = Conv(2 * c_, c2, 1, norm=norm)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0, norm=norm) for _ in range(n)])

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))


class C3TR(C3):

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = TransformerBlock(c_, c_, 4, n)


class SPP(nn.Module):

    def __init__(self, c1, c2, k=(5, 9, 13), norm='BN'):
        super(SPP, self).__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1, norm=norm)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1, norm=norm)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class ASPP(nn.Module):

    def __init__(self, c1, c2, d=(1, 2, 4, 6), norm='BN'):
        super(ASPP, self).__init__()
        assert c1 == c2 and c2 % len(d) == 0
        c_ = c2 // len(d)
        self.cv1 = Conv(c1, c_, 1, 1, norm=norm)
        self.m = nn.ModuleList([Conv(c_, c_, k=3, s=1, p=x, d=x, norm=norm) for x in d])

    def forward(self, x):
        x = self.cv1(x)
        return torch.cat([m(x) for m in self.m], 1)


class Focus(nn.Module):

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True, norm='BN'):
        super(Focus, self).__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act, norm=norm)

    def forward(self, x):
        return self.conv(self.contract(x))

    @staticmethod
    def contract(x):
        return torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1)


class Focus2(Focus):

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True, norm='BN'):
        super().__init__(c1, c2, k, s, p, g, act, norm)

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return self.conv(torch.cat((self.contract(x1), self.contract(x2)), dim=1))


class Blur(nn.Module):

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True, norm='BN'):
        super(Blur, self).__init__()
        self.conv = Conv(c1 // 4, c2, k, s, p, g, act, norm=norm)

    def forward(self, x):
        return self.conv(F.pixel_shuffle(x, 2))


class Contract(nn.Module):

    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        N, C, H, W = x.size()
        s = self.gain
        x = x.view(N, C, H // s, s, W // s, s)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()
        return x.view(N, C * s * s, H // s, W // s)


class Expand(nn.Module):

    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        N, C, H, W = x.size()
        s = self.gain
        x = x.view(N, s, s, C // s ** 2, H, W)
        x = x.permute(0, 3, 4, 1, 5, 2).contiguous()
        return x.view(N, C // s ** 2, H * s, W * s)


class Concat(nn.Module):

    def __init__(self, dimension=1):
        super(Concat, self).__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)


class Classify(nn.Module):

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1):
        super(Classify, self).__init__()
        self.aap = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g)
        self.flat = nn.Flatten()

    def forward(self, x):
        z = torch.cat([self.aap(y) for y in (x if isinstance(x, list) else [x])], 1)
        return self.flat(self.conv(z))


class CrossConv(nn.Module):

    def __init__(self, c1, c2, k=3, s=1, g=1, e=1.0, shortcut=False):
        super(CrossConv, self).__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, (1, k), (1, s))
        self.cv2 = Conv(c_, c2, (k, 1), (s, 1), g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class Sum(nn.Module):

    def __init__(self, n, weight=False):
        super(Sum, self).__init__()
        self.weight = weight
        self.iter = range(n - 1)
        if weight:
            self.w = nn.Parameter(-torch.arange(1.0, n) / 2, requires_grad=True)

    def forward(self, x):
        y = x[0]
        if self.weight:
            w = torch.sigmoid(self.w) * 2
            for i in self.iter:
                y = y + x[i + 1] * w[i]
        else:
            for i in self.iter:
                y = y + x[i + 1]
        return y


class GhostConv(nn.Module):

    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        super(GhostConv, self).__init__()
        c_ = c2 // 2
        self.cv1 = Conv(c1, c_, k, s, None, g, act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act)

    def forward(self, x):
        y = self.cv1(x)
        return torch.cat([y, self.cv2(y)], 1)


def DWConv(c1, c2, k=1, s=1, p=None, act=True, norm='BN'):
    assert norm == 'BN', 'Warning: depth-wise conv is recommended to take a batch-norm. Comment this line if needed.'
    return Conv(c1, c2, k, s, p=p, g=math.gcd(c1, c2), act=act, norm=norm)


class GhostBottleneck(nn.Module):

    def __init__(self, c1, c2, k=3, s=1):
        super(GhostBottleneck, self).__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(GhostConv(c1, c_, 1, 1), DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(), GhostConv(c_, c2, 1, 1, act=False))
        self.shortcut = nn.Sequential(DWConv(c1, c1, k, s, act=False), Conv(c1, c2, 1, 1, act=False)) if s == 2 else nn.Identity()

    def forward(self, x):
        return self.conv(x) + self.shortcut(x)


class MixConv2d(nn.Module):

    def __init__(self, c1, c2, k=(1, 3), s=1, equal_ch=True):
        super(MixConv2d, self).__init__()
        groups = len(k)
        if equal_ch:
            i = torch.linspace(0, groups - 1e-06, c2).floor()
            c_ = [(i == g).sum() for g in range(groups)]
        else:
            b = [c2] + [0] * groups
            a = np.eye(groups + 1, groups, k=-1)
            a -= np.roll(a, 1, axis=1)
            a *= np.array(k) ** 2
            a[0] = 1
            c_ = np.linalg.lstsq(a, b, rcond=None)[0].round()
        self.m = nn.ModuleList([nn.Conv2d(c1, int(c_[g]), k[g], s, k[g] // 2, bias=False) for g in range(groups)])
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        return x + self.act(self.bn(torch.cat([m(x) for m in self.m], 1)))


class Ensemble(nn.ModuleList):

    def __init__(self):
        super(Ensemble, self).__init__()

    def forward(self, x, augment=False):
        y = []
        for module in self:
            y.append(module(x, augment)[0])
        y = torch.cat(y, 1)
        return y, None


class Detect(nn.Module):
    stride = None
    onnx_dynamic = False

    def __init__(self, nc=80, anchors=(), ch=(), inplace=True):
        super(Detect, self).__init__()
        self.nc = nc
        self.no = nc + 5
        self.nl = len(anchors)
        self.na = len(anchors[0]) // 2
        self.grid = [torch.zeros(1)] * self.nl
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer('anchors', a)
        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)
        self.inplace = inplace

    def forward(self, x):
        z = []
        for i in range(self.nl):
            x[i] = self.m[i](x[i])
            bs, _, ny, nx = x[i].shape
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            if not self.training:
                if self.grid[i].shape[2:4] != x[i].shape[2:4] or self.onnx_dynamic:
                    self.grid[i] = self._make_grid(nx, ny)
                y = x[i].sigmoid()
                if self.inplace:
                    y[..., 0:2] = (y[..., 0:2] * 2.0 - 0.5 + self.grid[i]) * self.stride[i]
                    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]
                else:
                    xy = (y[..., 0:2] * 2.0 - 0.5 + self.grid[i]) * self.stride[i]
                    wh = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i].view(1, self.na, 1, 1, 2)
                    y = torch.cat((xy, wh, y[..., 4:]), -1)
                z.append(y.view(bs, -1, self.no))
        return x if self.training else (torch.cat(z, 1), x)

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()


class DLTSolver(object):

    def __init__(self):
        self.M1 = torch.tensor([[0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0]], dtype=torch.float32).unsqueeze(0)
        self.M2 = torch.tensor([[0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1]], dtype=torch.float32).unsqueeze(0)
        self.M3 = torch.tensor([[0], [1], [0], [1], [0], [1], [0], [1]], dtype=torch.float32).unsqueeze(0)
        self.M4 = torch.tensor([[-1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, -1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, -1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, -1, 0], [0, 0, 0, 0, 0, 0, 0, 0]], dtype=torch.float32).unsqueeze(0)
        self.M5 = torch.tensor([[0, -1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, -1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, -1, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, -1], [0, 0, 0, 0, 0, 0, 0, 0]], dtype=torch.float32).unsqueeze(0)
        self.M6 = torch.tensor([[-1], [0], [-1], [0], [-1], [0], [-1], [0]], dtype=torch.float32).unsqueeze(0)
        self.M71 = torch.tensor([[0, 1, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 1, 0]], dtype=torch.float32).unsqueeze(0)
        self.M72 = torch.tensor([[1, 0, 0, 0, 0, 0, 0, 0], [-1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0], [0, 0, -1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, -1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, -1, 0]], dtype=torch.float32).unsqueeze(0)
        self.M8 = torch.tensor([[0, 1, 0, 0, 0, 0, 0, 0], [0, -1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 0, -1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, -1, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0, -1]], dtype=torch.float32).unsqueeze(0)
        self.Mb = torch.tensor([[0, -1, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, -1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, -1, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, -1], [0, 0, 0, 0, 0, 0, 1, 0]], dtype=torch.float32).unsqueeze(0)

    def solve(self, pred_4pt_shift, patch_size=128.0):
        bs, device, dtype = pred_4pt_shift.size(0), pred_4pt_shift.device, pred_4pt_shift.dtype
        if isinstance(patch_size, float):
            p_width, p_height = patch_size, patch_size
        else:
            p_width, p_height = patch_size
        pts_1_tile = torch.tensor([0.0, 0.0, p_width, 0.0, 0.0, p_height, p_width, p_height], dtype=torch.float32)
        pred_pt4 = pts_1_tile.reshape((8, 1)).unsqueeze(0).expand(bs, -1, -1)
        orig_pt4 = pred_4pt_shift + pred_pt4
        self.check_mat_device(device)
        A1 = torch.bmm(self.M1.expand(bs, -1, -1), orig_pt4)
        A2 = torch.bmm(self.M2.expand(bs, -1, -1), orig_pt4)
        A3 = self.M3.expand(bs, -1, -1)
        A4 = torch.bmm(self.M4.expand(bs, -1, -1), orig_pt4)
        A5 = torch.bmm(self.M5.expand(bs, -1, -1), orig_pt4)
        A6 = self.M6.expand(bs, -1, -1)
        A7 = torch.bmm(self.M71.expand(bs, -1, -1), pred_pt4) * torch.bmm(self.M72.expand(bs, -1, -1), orig_pt4)
        A8 = torch.bmm(self.M71.expand(bs, -1, -1), pred_pt4) * torch.bmm(self.M8.expand(bs, -1, -1), orig_pt4)
        A_mat = torch.cat((A1, A2, A3, A4, A5, A6, A7, A8), dim=-1)
        b_mat = torch.bmm(self.Mb.expand(bs, -1, -1), pred_pt4)
        H_8el = torch.linalg.solve(A_mat.float(), b_mat.float()).type(dtype).squeeze(-1)
        h_ones = torch.ones((bs, 1)).type_as(H_8el)
        H_mat = torch.cat((H_8el, h_ones), dim=1).reshape(-1, 3, 3)
        return H_mat

    def check_mat_device(self, device):
        if self.M1.device != device:
            self.M1 = self.M1
            self.M2 = self.M2
            self.M3 = self.M3
            self.M4 = self.M4
            self.M5 = self.M5
            self.M6 = self.M6
            self.M71 = self.M71
            self.M72 = self.M72
            self.M8 = self.M8
            self.Mb = self.Mb


def check_anomaly(tensor, prefix='tensor', replace=0, _exit=False, _skip=False):
    is_nan, is_inf = torch.isnan(tensor), torch.isinf(tensor)
    for anomaly_type, anomaly_idx in zip(['nan', 'inf'], [is_nan, is_inf]):
        if anomaly_idx.any():
            None
            if replace is not None:
                tensor[anomaly_idx] = replace
                None
            else:
                None
            if _exit:
                exit(1)
    return tensor, is_nan, is_inf


def STN(image2_tensor, H_tf, offsets=()):
    """Spatial Transformer Layer"""

    def _repeat(x, n_repeats):
        rep = torch.ones(1, n_repeats, dtype=x.dtype)
        x = torch.mm(x.reshape(-1, 1), rep)
        return x.reshape(-1)

    def _interpolate(im, x, y, out_size):
        num_batch, channels, height, width = im.shape
        device = im.device
        x, y = x.float(), y.float()
        height_f, width_f = torch.tensor(height).float(), torch.tensor(width).float()
        out_height, out_width = out_size
        x = (x + 1.0) * width_f / 2.0
        y = (y + 1.0) * height_f / 2.0
        x0 = x.floor().int()
        x1 = x0 + 1
        y0 = y.floor().int()
        y1 = y0 + 1
        x0 = torch.clamp(x0, 0, width - 1)
        x1 = torch.clamp(x1, 0, width - 1)
        y0 = torch.clamp(y0, 0, height - 1)
        y1 = torch.clamp(y1, 0, height - 1)
        dim2 = width
        dim1 = width * height
        base = _repeat(torch.arange(num_batch) * dim1, out_height * out_width)
        base_y0 = base + y0 * dim2
        base_y1 = base + y1 * dim2
        idx_a = base_y0 + x0
        idx_b = base_y1 + x0
        idx_c = base_y0 + x1
        idx_d = base_y1 + x1
        im_flat = im.permute(0, 2, 3, 1).reshape(-1, channels).float()
        Ia, Ib, Ic, Id = im_flat[idx_a], im_flat[idx_b], im_flat[idx_c], im_flat[idx_d]
        x0_f = x0.float()
        x1_f = x1.float()
        y0_f = y0.float()
        y1_f = y1.float()
        wa = torch.unsqueeze((x1_f - x) * (y1_f - y), 1)
        wb = torch.unsqueeze((x1_f - x) * (y - y0_f), 1)
        wc = torch.unsqueeze((x - x0_f) * (y1_f - y), 1)
        wd = torch.unsqueeze((x - x0_f) * (y - y0_f), 1)
        output = wa * Ia + wb * Ib + wc * Ic + wd * Id
        return output

    def _meshgrid(height, width):
        x_t = torch.mm(torch.ones(height, 1), torch.linspace(-1.0, 1.0, width).unsqueeze(0))
        y_t = torch.mm(torch.linspace(-1.0, 1.0, height).unsqueeze(1), torch.ones(1, width))
        x_t_flat = x_t.reshape(1, -1)
        y_t_flat = y_t.reshape(1, -1)
        ones = torch.ones_like(x_t_flat)
        grid = torch.cat([x_t_flat, y_t_flat, ones], dim=0)
        return grid
    bs, nc, height, width = image2_tensor.shape
    device = image2_tensor.device
    is_nan = torch.isnan(H_tf.view(bs, 9)).any(dim=1)
    assert is_nan.sum() == 0, f'{image2_tensor.shape} {len(offsets)}, {[off.view(-1, 8)[is_nan] for off in offsets]}'
    H_tf = H_tf.reshape(-1, 3, 3).float()
    grid = _meshgrid(height, width).unsqueeze(0).expand(bs, -1, -1)
    T_g = torch.bmm(H_tf, grid)
    x_s, y_s, t_s = torch.chunk(T_g, 3, dim=1)
    t_s_flat = t_s.reshape(-1)
    eps, maximal = 0.01, 10.0
    t_s_flat[t_s_flat.abs() < eps] = eps
    x_s_flat = x_s.reshape(-1) / t_s_flat
    y_s_flat = y_s.reshape(-1) / t_s_flat
    input_transformed = _interpolate(image2_tensor, x_s_flat, y_s_flat, (height, width))
    output = input_transformed.reshape(bs, height, width, nc).permute(0, 3, 1, 2)
    check_anomaly(output, prefix='transformed feature map', _exit=True)
    return output


class HEstimator(nn.Module):

    def __init__(self, input_size=128, strides=(2, 4, 8), keep_prob=0.5, norm='BN', ch=()):
        super(HEstimator, self).__init__()
        self.ch = ch
        self.stride = torch.tensor([4, 32])
        self.input_size = input_size
        self.strides = strides
        self.keep_prob = keep_prob
        self.search_ranges = [16, 8, 4]
        self.patch_sizes = [input_size / 4, input_size / 2, input_size / 1]
        self.aux_matrices = torch.stack([self.gen_aux_mat(patch_size) for patch_size in self.patch_sizes])
        self.DLT_solver = DLTSolver()
        self._init_layers(norm=norm)

    def _init_layers(self, norm='BN'):
        m = []
        s = self.input_size // (128 // 8)
        k, p = s, 0
        for i, x in enumerate(self.ch[::-1]):
            ch1 = x * 2
            ch_conv = 512 // 2 ** i
            ch_flat = (self.input_size // self.strides[-1] // s) ** 2 * ch_conv
            ch_fc = 512 // 2 ** i
            m.append(nn.Sequential(Conv(ch1, ch_conv, k=3, s=1, norm=norm), Conv(ch_conv, ch_conv, k=3, s=2 if i >= 2 else 1, norm=norm), Conv(ch_conv, ch_conv, k=3, s=2 if i >= 1 else 1, norm=norm), DWConv(ch_conv, ch_conv, k=k, s=s, p=p, norm='BN'), nn.Flatten(), nn.Linear(ch_flat, ch_fc), nn.SiLU(), nn.Linear(ch_fc, 8, bias=False)))
        self.m = nn.ModuleList(m)

    def forward(self, feature1, feature2, image2, mask2):
        bs = image2.size(0)
        assert len(self.search_ranges) == len(feature1) == len(feature2)
        device, dtype = image2.device, image2.dtype
        if self.aux_matrices.device != device:
            self.aux_matrices = self.aux_matrices
        if self.aux_matrices.dtype != dtype:
            self.aux_matrices = self.aux_matrices.type(dtype)
        vertices_offsets = []
        for i, search_range in enumerate(self.search_ranges):
            x = self._feat_fuse(feature1[-(i + 1)], feature2[-(i + 1)], i=i, search_range=search_range)
            off = self.m[i](x).unsqueeze(-1)
            assert torch.isnan(off).sum() == 0
            vertices_offsets.append(off)
            if i == len(self.search_ranges) - 1:
                break
            H = self.DLT_solver.solve(sum(vertices_offsets) / 4.0, self.patch_sizes[0])
            M, M_inv = torch.chunk(self.aux_matrices[0], 2, dim=0)
            H = torch.bmm(torch.bmm(M_inv.expand(bs, -1, -1), H), M.expand(bs, -1, -1))
            feature2[-(i + 2)] = self._feat_warp(feature2[-(i + 2)], H, vertices_offsets)
        warped_imgs, warped_msks = [], []
        patch_level = 0
        M, M_inv = torch.chunk(self.aux_matrices[patch_level], 2, dim=0)
        img_with_msk = torch.cat((image2, mask2), dim=1)
        for i in range(len(vertices_offsets)):
            H_inv = self.DLT_solver.solve(sum(vertices_offsets[:i + 1]) / 2 ** (2 - patch_level), self.patch_sizes[patch_level])
            H = torch.bmm(torch.bmm(M_inv.expand(bs, -1, -1), H_inv), M.expand(bs, -1, -1))
            warped_img, warped_msk = STN(img_with_msk, H, vertices_offsets[:i + 1]).split([3, 1], dim=1)
            warped_imgs.append(warped_img)
            warped_msks.append(warped_msk)
        return sum(vertices_offsets), warped_imgs, warped_msks

    def _feat_fuse(self, x1, x2, i, search_range):
        x = torch.cat((x1, x2), dim=1)
        return x

    @staticmethod
    def _feat_warp(x2, H, vertices_offsets):
        return STN(x2, H, vertices_offsets)

    @staticmethod
    def cost_volume(x1, x2, search_range, norm=True, fast=True):
        if norm:
            x1 = F.normalize(x1, p=2, dim=1)
            x2 = F.normalize(x2, p=2, dim=1)
        bs, c, h, w = x1.shape
        padded_x2 = F.pad(x2, [search_range] * 4)
        max_offset = search_range * 2 + 1
        if fast:
            patches = F.unfold(padded_x2, (max_offset, max_offset)).reshape(bs, c, max_offset ** 2, h, w)
            cost_vol = (x1.unsqueeze(2) * patches).mean(dim=1, keepdim=False)
        else:
            cost_vol = []
            for j in range(0, max_offset):
                for i in range(0, max_offset):
                    x2_slice = padded_x2[:, :, j:j + h, i:i + w]
                    cost = torch.mean(x1 * x2_slice, dim=1, keepdim=True)
                    cost_vol.append(cost)
            cost_vol = torch.cat(cost_vol, dim=1)
        cost_vol = F.leaky_relu(cost_vol, 0.1)
        return cost_vol

    @staticmethod
    def gen_aux_mat(patch_size):
        M = np.array([[patch_size / 2.0, 0.0, patch_size / 2.0], [0.0, patch_size / 2.0, patch_size / 2.0], [0.0, 0.0, 1.0]]).astype(np.float32)
        M_inv = np.linalg.inv(M)
        return torch.from_numpy(np.stack((M, M_inv)))


class HEstimatorOrigin(HEstimator):

    def __init__(self, input_size=128, strides=(2, 4, 8), keep_prob=0.5, norm='None', ch=()):
        super(HEstimatorOrigin, self).__init__(input_size, strides, keep_prob, norm, ch)

    def _init_layers(self, norm='None'):
        m = []
        for i, x in enumerate(self.ch[::-1]):
            ch1 = (self.search_ranges[i] * 2 + 1) ** 2
            ch_conv = 512 // 2 ** i
            ch_flat = (self.input_size // self.strides[-1]) ** 2 * ch_conv
            ch_fc = 1024 // 2 ** i
            m.append(nn.Sequential(Conv(ch1, ch_conv, k=3, s=1, norm=norm), Conv(ch_conv, ch_conv, k=3, s=2 if i >= 2 else 1, norm=norm), Conv(ch_conv, ch_conv, k=3, s=2 if i >= 1 else 1, norm=norm), nn.Flatten(), nn.Linear(ch_flat, ch_fc), nn.ReLU(inplace=True), nn.Dropout(self.keep_prob), nn.Linear(ch_fc, 8, bias=False)))
        self.m = nn.ModuleList(m)

    def _feat_fuse(self, x1, x2, i, search_range):
        x1, x2 = F.normalize(x1, p=2, dim=1), F.normalize(x2, p=2, dim=1) if i == 0 else x2
        x = self.cost_volume(x1, x2, search_range, norm=False)
        return x

    @staticmethod
    def _feat_warp(x2, H, vertices_offsets):
        return STN(F.normalize(x2, p=2, dim=1), H, vertices_offsets)


class Reconstructor(nn.Module):

    def __init__(self, norm='BN', ch=()):
        super(Reconstructor, self).__init__()
        ch_lr = ch[0]
        self.m_lr = nn.Sequential(nn.Conv2d(ch_lr, 3, kernel_size=3, stride=1, padding=1, bias=False))
        self.m_hr = nn.Sequential(Conv(3 * 3, 64, norm=norm), C3(64, 64, 3, norm=norm), nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1, bias=False))
        self.stride = torch.tensor([4, 32])

    def forward(self, x):
        out_lr, in_hr = x
        out_lr = self.m_lr(out_lr).sigmoid_()
        out_lr_sr = F.interpolate(out_lr, mode='bilinear', size=in_hr.shape[2:], align_corners=False)
        out_hr = torch.cat((in_hr, out_lr_sr), dim=1)
        out_hr = self.m_hr(out_hr).sigmoid_()
        return out_lr, out_hr


def fuse_conv_and_bn(conv, bn):
    fusedconv = torch.nn.Conv2d(conv.in_channels, conv.out_channels, kernel_size=conv.kernel_size, stride=conv.stride, padding=conv.padding, groups=conv.groups, bias=True).requires_grad_(False)
    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))
    b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)
    return fusedconv


def initialize_weights(model):
    for m in model.modules():
        t = type(m)
        if t is torch.nn.Conv2d:
            pass
        elif t is torch.nn.BatchNorm2d:
            m.eps = 0.001
            m.momentum = 0.03
        elif t in [torch.nn.Hardswish, torch.nn.LeakyReLU, torch.nn.ReLU, torch.nn.ReLU6]:
            m.inplace = True


logger = logging.getLogger(__name__)


def model_info(model, verbose=False, img_size=640):
    n_p = sum(x.numel() for x in model.parameters())
    n_g = sum(x.numel() for x in model.parameters() if x.requires_grad)
    if verbose:
        None
        for i, (name, p) in enumerate(model.named_parameters()):
            name = name.replace('module_list.', '')
            None

    def thop_forward(model_align):
        img_size = 128 if model_align else 640
        stride = img_size
        img = torch.zeros((1, 8, stride, stride), device=next(model.parameters()).device)
        flops = profile(deepcopy(model), inputs=(img, False, model_align), verbose=False)[0] / 1000000000.0 * 2
        img_size = img_size if isinstance(img_size, list) else [img_size, img_size]
        fs = ', %.1f GFLOPs' % (flops * img_size[0] / stride * img_size[1] / stride)
        return fs
    try:
        fs = thop_forward(model_align=False)
    except RuntimeError as e:
        try:
            fs = thop_forward(model_align=True)
        except (ImportError, Exception) as e:
            None
            fs = ''
    except (ImportError, Exception) as e:
        None
        fs = ''
    logger.info(f'Model Summary: {len(list(model.modules()))} layers, {n_p} parameters, {n_g} gradients{fs}')


def make_divisible(x, divisor):
    return math.ceil(x / divisor) * divisor


def parse_model(d, ch):
    logger.info('\n%3s%18s%3s%10s  %-40s%-30s' % ('', 'from', 'n', 'params', 'module', 'arguments'))
    gd, gw = d['depth_multiple'], d['width_multiple']
    no = 3
    layers, save, c2 = [], [], ch[-1]
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):
        m = eval(m) if isinstance(m, str) else m
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a
            except:
                pass
        n = max(round(n * gd), 1) if n > 1 else n
        if m in [Conv, GhostConv, Bottleneck, GhostBottleneck, SPP, DWConv, MixConv2d, Focus, Blur, CrossConv, BottleneckCSP, C3, C3TR, Focus2, ResBlock]:
            c1, c2 = ch[f], args[0]
            if c2 != no:
                c2 = make_divisible(c2 * gw, 8)
            args = [c1, c2, *args[1:]]
            if m in [BottleneckCSP, C3, C3TR]:
                args.insert(2, n)
                n = 1
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum([ch[x] for x in f])
        elif m is Add:
            c2 = ch[f[0]]
        elif m is Resizer:
            c2 = ch[f[0]] if isinstance(f, Iterable) else ch[f]
        elif m is Reconstructor:
            args.append([ch[x] for x in f])
        elif m in [HEstimator, HEstimatorOrigin]:
            args.append(ch[f])
        elif m is Contract:
            c2 = ch[f] * args[0] ** 2
        elif m is Expand:
            c2 = ch[f] // args[0] ** 2
        elif m is nn.Identity:
            c2 = [ch[x] for x in f] if isinstance(f, Iterable) else ch[f]
        else:
            c2 = ch[f]
        m_ = nn.Sequential(*[m(*args) for _ in range(n)]) if n > 1 else m(*args)
        t = str(m)[8:-2].replace('__main__.', '')
        np = sum([x.numel() for x in m_.parameters()])
        m_.i, m_.f, m_.type, m_.np = i, f, t, np
        logger.info('%3s%18s%3s%10.0f  %-40s%-30s' % (i, f, n, np, t, args))
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)


def time_synchronized():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()


class Model(nn.Module):

    def __init__(self, cfg='yolov5s.yaml', ch=3, mode_align=True):
        super(Model, self).__init__()
        if isinstance(cfg, dict):
            self.yaml = cfg
        else:
            self.yaml_file = Path(cfg).name
            with open(cfg) as f:
                self.yaml = yaml.safe_load(f)
        self.mode_align = mode_align
        self.ch = self.yaml['ch'] = self.yaml.get('ch', ch)
        ch = 3 if mode_align else 6
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch])
        self.inplace = self.yaml.get('inplace', True)
        m = self.model[-1]
        if isinstance(m, Reconstructor) or isinstance(m, HEstimator):
            self.stride = m.stride
        else:
            self.stride = torch.tensor([4, 32])
        initialize_weights(self)
        self.info()
        logger.info('')

    def forward(self, x, profile=False, mode_align=True):
        x1, m1, x2, m2 = torch.split(x, [3, 1, 3, 1], dim=1)
        mode_align = self.mode_align if hasattr(self, 'mode_align') else mode_align
        if mode_align:
            module_range = 0, -1
            feature1 = self.forward_once(x1, profile, module_range=module_range)
            feature2 = self.forward_once(x2, profile, module_range=module_range)
            return self.model[-1](feature1, feature2, x2, m2)
        else:
            x = torch.cat((x1, x2), dim=1)
            out = self.forward_once(x, profile)
            if not self.training:
                mask = (m1 + m2 > 0).type_as(x)
                out = out[0], out[1] * mask
            return out

    def forward_once(self, x, profile=False, module_range=None):
        y, dt = [], []
        inputs = x
        modules = self.model if module_range is None else self.model[module_range[0]:module_range[1]]
        for m in modules:
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [(x if j == -1 else y[j]) for j in m.f]
            if profile:
                o = thop.profile(m, inputs=(x,), verbose=False)[0] / 1000000000.0 * 2 if thop else 0
                t = time_synchronized()
                for _ in range(10):
                    _ = m(x)
                dt.append((time_synchronized() - t) * 100)
                if m == self.model[0]:
                    logger.info(f"{'time (ms)':>10s} {'GFLOPs':>10s} {'params':>10s}  {'module'}")
                logger.info(f'{dt[-1]:10.2f} {o:10.2f} {m.np:10.0f}  {m.type}')
            if str(m.type) in 'models.yolo.Reconstructor':
                x = *x, inputs
            x = m(x)
            y.append(x if m.i in self.save else None)
        if profile:
            logger.info('%.1fms total' % sum(dt))
        return x

    def fuse(self):
        logger.info('Fusing layers... ')
        for m in self.model.modules():
            if type(m) is Conv and hasattr(m, 'bn') and isinstance(m.bn, nn.BatchNorm2d):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)
                delattr(m, 'bn')
                m.forward = m.fuseforward
        self.info()
        return self

    def info(self, verbose=False, img_size=640):
        model_info(self, verbose, img_size)


class VGGPerceptualLoss(torch.nn.Module):

    def __init__(self, device, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        vgg_features = torchvision.models.vgg19(pretrained=True).features.eval()
        blocks = [vgg_features[:18], vgg_features[18:36]]
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.resize = torch.nn.functional.interpolate if resize else None
        self.normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def forward(self, x, y, mask=None):
        if mask is None:
            mask = torch.ones_like(x[:, :1, :, :])
        x, y = x * mask, y * mask
        if self.resize:
            x = self.resize(x, mode='bilinear', size=(224, 224), align_corners=False)
            y = self.resize(y, mode='bilinear', size=(224, 224), align_corners=False)
        x = self.normalize(x).float()
        y = self.normalize(y).float()
        losses = []
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            losses.append(F.mse_loss(x, y))
        return losses


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (ASPP,
     lambda: ([], {'c1': 4, 'c2': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Add,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Blur,
     lambda: ([], {'c1': 4, 'c2': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Bottleneck,
     lambda: ([], {'c1': 4, 'c2': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (BottleneckCSP,
     lambda: ([], {'c1': 4, 'c2': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (C3,
     lambda: ([], {'c1': 4, 'c2': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Classify,
     lambda: ([], {'c1': 4, 'c2': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Contract,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Conv,
     lambda: ([], {'c1': 4, 'c2': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (CrossConv,
     lambda: ([], {'c1': 4, 'c2': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Expand,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Focus,
     lambda: ([], {'c1': 4, 'c2': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Focus2,
     lambda: ([], {'c1': 4, 'c2': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (FuseBlock,
     lambda: ([], {'c1': 4, 'c2': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (GhostBottleneck,
     lambda: ([], {'c1': 4, 'c2': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (GhostConv,
     lambda: ([], {'c1': 4, 'c2': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (MixConv2d,
     lambda: ([], {'c1': 4, 'c2': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ResBlock,
     lambda: ([], {'c1': 4, 'c2': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (SPP,
     lambda: ([], {'c1': 4, 'c2': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Sum,
     lambda: ([], {'n': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (TransformerBlock,
     lambda: ([], {'c1': 4, 'c2': 4, 'num_heads': 4, 'num_layers': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (TransformerLayer,
     lambda: ([], {'c': 4, 'num_heads': 4}),
     lambda: ([torch.rand([4, 4])], {})),
    (VGGPerceptualLoss,
     lambda: ([], {'device': 0}),
     lambda: ([torch.rand([4, 3, 4, 4]), torch.rand([4, 3, 4, 4])], {})),
]

