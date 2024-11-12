
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


import torch.nn as nn


from torchvision import transforms as T


import torchvision.datasets as datasets


import random


import numpy as np


import torch.distributed as dist


from torch.nn.parallel import DistributedDataParallel as DDP


from torch.utils.data.distributed import DistributedSampler


from typing import Callable


from typing import Any


from typing import Optional


from typing import List


from torchvision import datasets


class ConvNormAct(nn.Module):

    def __init__(self, in_channels: 'int', out_channels: 'int', kernel_size: 'int'=3, stride=1, padding: 'Optional[int]'=None, groups: 'int'=1, norm_layer: 'Optional[Callable[..., torch.nn.Module]]'=torch.nn.BatchNorm2d, activation_layer: 'Optional[Callable[..., torch.nn.Module]]'=torch.nn.SiLU, dilation: 'int'=1):
        super(ConvNormAct, self).__init__()
        if padding is None:
            padding = (kernel_size - 1) // 2 * dilation
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation=dilation, groups=groups, bias=norm_layer is None)
        self.norm_layer = nn.BatchNorm2d(out_channels) if norm_layer is None else norm_layer(out_channels)
        self.act = activation_layer() if activation_layer is not None else activation_layer

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        x = self.conv(x)
        if self.norm_layer is not None:
            x = self.norm_layer(x)
        if self.act is not None:
            x = self.act(x)
        return x


class InvertedResidual(nn.Module):
    """
    MobileNetv2 InvertedResidual block
    """

    def __init__(self, in_channels, out_channels, stride=1, expand_ratio=2, act_layer=nn.SiLU):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        self.use_res_connect = self.stride == 1 and in_channels == out_channels
        hidden_dim = int(round(in_channels * expand_ratio))
        layers = []
        if expand_ratio != 1:
            layers.append(ConvNormAct(in_channels, hidden_dim, kernel_size=1, activation_layer=None))
        layers.append(ConvNormAct(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1, groups=hidden_dim, activation_layer=act_layer))
        layers.append(nn.Conv2d(hidden_dim, out_channels, kernel_size=1, stride=1, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class FFN(nn.Module):

    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden_dim), nn.SiLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, dim), nn.Dropout(dropout))

    def forward(self, x):
        return self.net(x)


class MultiHeadSelfAttention(nn.Module):
    """
    Implement multi head self attention layer using the "Einstein summation convention".
    Paper: https://arxiv.org/abs/1706.03762
    Blog: https://towardsdatascience.com/illustrated-self-attention-2d627e33b20a
    Parameters
    ----------
    dim:
        Token's dimension, EX: word embedding vector size
    num_heads:
        The number of distinct representations to learn
    dim_head:
        The dimension of the each head
    """

    def __init__(self, dim, num_heads=8, dim_head=None):
        super(MultiHeadSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.dim_head = int(dim / num_heads) if dim_head is None else dim_head
        _weight_dim = self.num_heads * self.dim_head
        self.to_qvk = nn.Linear(dim, _weight_dim * 3, bias=False)
        self.scale_factor = dim ** -0.5
        self.scale_factor = dim ** -0.5
        self.w_out = nn.Linear(_weight_dim, dim, bias=False)

    def forward(self, x):
        qkv = self.to_qvk(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b p n (h d) -> b p h n d', h=self.num_heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale_factor
        attn = torch.softmax(dots, dim=-1)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b p h n d -> b p n (h d)')
        return self.w_out(out)


class PreNorm(nn.Module):

    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class Transformer(nn.Module):

    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([PreNorm(dim, MultiHeadSelfAttention(dim, heads, dim_head)), PreNorm(dim, FFN(dim, mlp_dim, dropout))]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class MobileVitBlock(nn.Module):

    def __init__(self, in_channels, out_channels, d_model, layers, mlp_dim):
        super(MobileVitBlock, self).__init__()
        self.local_representation = nn.Sequential(ConvNormAct(in_channels, in_channels, 3), ConvNormAct(in_channels, d_model, 1))
        self.transformer = Transformer(d_model, layers, 1, 32, mlp_dim, 0.1)
        self.fusion_block1 = nn.Conv2d(d_model, in_channels, kernel_size=1)
        self.fusion_block2 = nn.Conv2d(in_channels * 2, out_channels, 3, padding=1)

    def forward(self, x):
        local_repr = self.local_representation(x)
        _, _, h, w = local_repr.shape
        global_repr = rearrange(local_repr, 'b d (h ph) (w pw) -> b (ph pw) (h w) d', ph=2, pw=2)
        global_repr = self.transformer(global_repr)
        global_repr = rearrange(global_repr, 'b (ph pw) (h w) d -> b d (h ph) (w pw)', h=h // 2, w=w // 2, ph=2, pw=2)
        fuse_repr = self.fusion_block1(global_repr)
        result = self.fusion_block2(torch.cat([x, fuse_repr], dim=1))
        return result


class MobileViT(nn.Module):

    def __init__(self, img_size, features_list, d_list, transformer_depth, expansion, num_classes=1000):
        super(MobileViT, self).__init__()
        self.stem = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=features_list[0], kernel_size=3, stride=2, padding=1), InvertedResidual(in_channels=features_list[0], out_channels=features_list[1], stride=1, expand_ratio=expansion))
        self.stage1 = nn.Sequential(InvertedResidual(in_channels=features_list[1], out_channels=features_list[2], stride=2, expand_ratio=expansion), InvertedResidual(in_channels=features_list[2], out_channels=features_list[2], stride=1, expand_ratio=expansion), InvertedResidual(in_channels=features_list[2], out_channels=features_list[3], stride=1, expand_ratio=expansion))
        self.stage2 = nn.Sequential(InvertedResidual(in_channels=features_list[3], out_channels=features_list[4], stride=2, expand_ratio=expansion), MobileVitBlock(in_channels=features_list[4], out_channels=features_list[5], d_model=d_list[0], layers=transformer_depth[0], mlp_dim=d_list[0] * 2))
        self.stage3 = nn.Sequential(InvertedResidual(in_channels=features_list[5], out_channels=features_list[6], stride=2, expand_ratio=expansion), MobileVitBlock(in_channels=features_list[6], out_channels=features_list[7], d_model=d_list[1], layers=transformer_depth[1], mlp_dim=d_list[1] * 4))
        self.stage4 = nn.Sequential(InvertedResidual(in_channels=features_list[7], out_channels=features_list[8], stride=2, expand_ratio=expansion), MobileVitBlock(in_channels=features_list[8], out_channels=features_list[9], d_model=d_list[2], layers=transformer_depth[2], mlp_dim=d_list[2] * 4), nn.Conv2d(in_channels=features_list[9], out_channels=features_list[10], kernel_size=1, stride=1, padding=0))
        self.avgpool = nn.AvgPool2d(kernel_size=img_size // 32)
        self.fc = nn.Linear(features_list[10], num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (ConvNormAct,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (FFN,
     lambda: ([], {'dim': 4, 'hidden_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (InvertedResidual,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (PreNorm,
     lambda: ([], {'dim': 4, 'fn': torch.nn.ReLU()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
]

