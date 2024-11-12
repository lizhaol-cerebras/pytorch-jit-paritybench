
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


import numpy as np


from torch.utils.data import Dataset


import logging


from scipy import integrate


import torch.optim as optim


import torch.nn as nn


import functools


import math


import string


from functools import partial


import torch.nn.functional as F


import abc


import time


from torch.utils import tensorboard


from random import randint


from time import sleep


import torch.utils.cpp_extension


from torch.utils.cpp_extension import CUDAExtension


from torch.utils.cpp_extension import CppExtension


from torch.utils.cpp_extension import BuildExtension


def variance_scaling(scale, mode, distribution, in_axis=1, out_axis=0, dtype=torch.float32, device='cpu'):
    """Ported from JAX. """

    def _compute_fans(shape, in_axis=1, out_axis=0):
        receptive_field_size = np.prod(shape) / shape[in_axis] / shape[out_axis]
        fan_in = shape[in_axis] * receptive_field_size
        fan_out = shape[out_axis] * receptive_field_size
        return fan_in, fan_out

    def init(shape, dtype=dtype, device=device):
        fan_in, fan_out = _compute_fans(shape, in_axis, out_axis)
        if mode == 'fan_in':
            denominator = fan_in
        elif mode == 'fan_out':
            denominator = fan_out
        elif mode == 'fan_avg':
            denominator = (fan_in + fan_out) / 2
        else:
            raise ValueError('invalid mode for variance scaling initializer: {}'.format(mode))
        variance = scale / denominator
        if distribution == 'normal':
            return torch.randn(*shape, dtype=dtype, device=device) * np.sqrt(variance)
        elif distribution == 'uniform':
            return (torch.rand(*shape, dtype=dtype, device=device) * 2.0 - 1.0) * np.sqrt(3 * variance)
        else:
            raise ValueError('invalid distribution for variance scaling initializer')
    return init


def default_init(scale=1.0):
    """The same initialization used in DDPM."""
    scale = 1e-10 if scale == 0 else scale
    return variance_scaling(scale, 'fan_avg', 'uniform')


def ddpm_conv3x3(in_planes, out_planes, stride=1, bias=True, dilation=1, init_scale=1.0, padding=1):
    """3x3 convolution with DDPM initialization."""
    conv = nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride, padding=padding, dilation=dilation, bias=bias)
    conv.weight.data = default_init(init_scale)(conv.weight.data.shape)
    nn.init.zeros_(conv.bias)
    return conv


class Downsample(nn.Module):

    def __init__(self, channels, with_conv=False):
        super().__init__()
        if with_conv:
            self.Conv_0 = ddpm_conv3x3(channels, channels, stride=2, padding=0)
        self.with_conv = with_conv

    def forward(self, x):
        B, C, D, H, W = x.shape
        if self.with_conv:
            x = F.pad(x, (0, 1, 0, 1, 0, 1))
            x = self.Conv_0(x)
        else:
            x = F.avg_pool3d(x, kernel_size=2, stride=2, padding=0)
        assert x.shape == (B, C, D // 2, H // 2, W // 2)
        return x


def _einsum(a, b, c, x, y):
    einsum_str = '{},{}->{}'.format(''.join(a), ''.join(b), ''.join(c))
    return torch.einsum(einsum_str, x, y)


def contract_inner(x, y):
    """tensordot(x, y, 1)."""
    x_chars = list(string.ascii_lowercase[:len(x.shape)])
    y_chars = list(string.ascii_lowercase[len(x.shape):len(y.shape) + len(x.shape)])
    y_chars[0] = x_chars[-1]
    out_chars = x_chars[:-1] + y_chars[1:]
    return _einsum(x_chars, y_chars, out_chars, x, y)


class NIN(nn.Module):

    def __init__(self, in_dim, num_units, init_scale=0.1):
        super().__init__()
        self.W = nn.Parameter(default_init(scale=init_scale)((in_dim, num_units)), requires_grad=True)
        self.b = nn.Parameter(torch.zeros(num_units), requires_grad=True)

    def forward(self, x):
        x = x.permute(0, 2, 3, 4, 1)
        y = contract_inner(x, self.W) + self.b
        return y.permute(0, 4, 1, 2, 3)


class ResnetBlockDDPM(nn.Module):
    """The ResNet Blocks used in DDPM."""

    def __init__(self, act, in_ch, out_ch=None, temb_dim=None, conv_shortcut=False, dropout=0.1):
        super().__init__()
        if out_ch is None:
            out_ch = in_ch
        self.GroupNorm_0 = nn.GroupNorm(num_groups=32, num_channels=in_ch, eps=1e-06)
        self.act = act
        self.Conv_0 = ddpm_conv3x3(in_ch, out_ch)
        if temb_dim is not None:
            self.Dense_0 = nn.Linear(temb_dim, out_ch)
            self.Dense_0.weight.data = default_init()(self.Dense_0.weight.data.shape)
            nn.init.zeros_(self.Dense_0.bias)
        self.GroupNorm_1 = nn.GroupNorm(num_groups=32, num_channels=out_ch, eps=1e-06)
        self.Dropout_0 = nn.Dropout(dropout)
        self.Conv_1 = ddpm_conv3x3(out_ch, out_ch, init_scale=0.0)
        if in_ch != out_ch:
            if conv_shortcut:
                self.Conv_2 = ddpm_conv3x3(in_ch, out_ch)
            else:
                self.NIN_0 = NIN(in_ch, out_ch)
        self.out_ch = out_ch
        self.in_ch = in_ch
        self.conv_shortcut = conv_shortcut

    def forward(self, x, temb=None):
        B, C, D, H, W = x.shape
        assert C == self.in_ch
        out_ch = self.out_ch if self.out_ch else self.in_ch
        h = self.act(self.GroupNorm_0(x))
        h = self.Conv_0(h)
        if temb is not None:
            h += self.Dense_0(self.act(temb))[:, :, None, None, None]
        h = self.act(self.GroupNorm_1(h))
        h = self.Dropout_0(h)
        h = self.Conv_1(h)
        if C != out_ch:
            if self.conv_shortcut:
                x = self.Conv_2(x)
            else:
                x = self.NIN_0(x)
        return x + h


class Upsample(nn.Module):

    def __init__(self, channels, with_conv=False):
        super().__init__()
        if with_conv:
            self.Conv_0 = ddpm_conv3x3(channels, channels)
        self.with_conv = with_conv

    def forward(self, x):
        B, C, D, H, W = x.shape
        h = F.interpolate(x, (D * 2, H * 2, W * 2), mode='nearest')
        if self.with_conv:
            h = self.Conv_0(h)
        return h


def get_act(config):
    """Get activation functions from the config file."""
    if config.model.nonlinearity.lower() == 'elu':
        return nn.ELU()
    elif config.model.nonlinearity.lower() == 'relu':
        return nn.ReLU()
    elif config.model.nonlinearity.lower() == 'lrelu':
        return nn.LeakyReLU(negative_slope=0.2)
    elif config.model.nonlinearity.lower() == 'swish':
        return nn.SiLU()
    else:
        raise NotImplementedError('activation function does not exist!')


class DDPMRes128(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.act = act = get_act(config)
        self.register_buffer('sigmas', torch.tensor(utils.get_sigmas(config)))
        self.nf = nf = config.model.nf
        ch_mult = config.model.ch_mult
        self.num_res_blocks = num_res_blocks = config.model.num_res_blocks
        self.attn_resolutions = attn_resolutions = config.model.attn_resolutions
        dropout = config.model.dropout
        resamp_with_conv = config.model.resamp_with_conv
        self.num_resolutions = num_resolutions = len(ch_mult)
        self.all_resolutions = all_resolutions = [(config.data.image_size // 2 ** i) for i in range(num_resolutions)]
        AttnBlock = functools.partial(layers.AttnBlock)
        self.conditional = conditional = config.model.conditional
        ResnetBlock = functools.partial(ResnetBlockDDPM, act=act, temb_dim=4 * nf, dropout=dropout)
        if conditional:
            modules = [nn.Linear(nf, nf * 4)]
            modules[0].weight.data = default_initializer()(modules[0].weight.data.shape)
            nn.init.zeros_(modules[0].bias)
            modules.append(nn.Linear(nf * 4, nf * 4))
            modules[1].weight.data = default_initializer()(modules[1].weight.data.shape)
            nn.init.zeros_(modules[1].bias)
        self.centered = config.data.centered
        channels = config.data.num_channels
        self.img_size = img_size = config.data.image_size
        self.num_freq = int(np.log2(img_size))
        coord_x, coord_y, coord_z = torch.meshgrid(torch.arange(img_size), torch.arange(img_size), torch.arange(img_size))
        self.use_coords = False
        if self.use_coords:
            self.coords = torch.nn.Parameter(torch.stack([coord_x, coord_y, coord_z]).view(1, 3, img_size, img_size, img_size), requires_grad=False)
        self.mask = torch.nn.Parameter(torch.zeros(1, 1, img_size, img_size, img_size), requires_grad=False)
        self.pos_layer = conv5x5(3, nf, stride=1, padding=2)
        self.mask_layer = conv5x5(1, nf, stride=1, padding=2)
        modules.append(conv5x5(channels, nf, stride=1, padding=2))
        hs_c = [nf]
        in_ch = nf
        for i_level in range(num_resolutions):
            num_res_blocks_curr = self.num_res_blocks if i_level != 0 else 2
            for i_block in range(num_res_blocks_curr):
                out_ch = nf * ch_mult[i_level]
                modules.append(ResnetBlock(in_ch=in_ch, out_ch=out_ch))
                in_ch = out_ch
                if all_resolutions[i_level] in attn_resolutions:
                    modules.append(AttnBlock(channels=in_ch))
                hs_c.append(in_ch)
            if i_level != num_resolutions - 1:
                modules.append(Downsample(channels=in_ch, with_conv=resamp_with_conv))
                hs_c.append(in_ch)
        in_ch = hs_c[-1]
        modules.append(ResnetBlock(in_ch=in_ch))
        modules.append(AttnBlock(channels=in_ch))
        modules.append(ResnetBlock(in_ch=in_ch))
        for i_level in reversed(range(num_resolutions)):
            num_res_blocks_curr = self.num_res_blocks if i_level != 0 else 2
            for i_block in range(num_res_blocks_curr + 1):
                out_ch = nf * ch_mult[i_level]
                modules.append(ResnetBlock(in_ch=in_ch + hs_c.pop(), out_ch=out_ch))
                in_ch = out_ch
            if all_resolutions[i_level] in attn_resolutions:
                modules.append(AttnBlock(channels=in_ch))
            if i_level != 0:
                modules.append(Upsample(channels=in_ch, with_conv=resamp_with_conv))
        assert not hs_c
        modules.append(nn.GroupNorm(num_channels=in_ch, num_groups=32, eps=1e-06))
        modules.append(conv5x5(in_ch, channels, init_scale=0.0, stride=1, padding=2))
        self.all_modules = nn.ModuleList(modules)
        self.scale_by_sigma = config.model.scale_by_sigma

    def forward(self, x, labels):
        modules = self.all_modules
        m_idx = 0
        if self.conditional:
            timesteps = labels
            temb = layers.get_timestep_embedding(timesteps, self.nf)
            temb = modules[m_idx](temb)
            m_idx += 1
            temb = modules[m_idx](self.act(temb))
            m_idx += 1
        else:
            temb = None
        if self.centered:
            h = x
        else:
            h = 2 * x - 1.0
        if self.use_coords:
            hs = [modules[m_idx](h) + self.pos_layer(self.coords) + self.mask_layer(self.mask)]
        else:
            hs = [modules[m_idx](h) + self.mask_layer(self.mask)]
        m_idx += 1
        for i_level in range(self.num_resolutions):
            num_res_blocks = self.num_res_blocks if i_level != 0 else 2
            for i_block in range(num_res_blocks):
                h = modules[m_idx](hs[-1], temb)
                m_idx += 1
                if h.shape[-1] in self.attn_resolutions:
                    h = modules[m_idx](h)
                    m_idx += 1
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(modules[m_idx](hs[-1]))
                m_idx += 1
        h = hs[-1]
        h = modules[m_idx](h, temb)
        m_idx += 1
        h = modules[m_idx](h)
        m_idx += 1
        h = modules[m_idx](h, temb)
        m_idx += 1
        for i_level in reversed(range(self.num_resolutions)):
            num_res_blocks = self.num_res_blocks if i_level != 0 else 2
            for i_block in range(num_res_blocks + 1):
                hspop = hs.pop()
                input = torch.cat([h, hspop], dim=1)
                h = modules[m_idx](input, temb)
                m_idx += 1
            if h.shape[-1] in self.attn_resolutions:
                h = modules[m_idx](h)
                m_idx += 1
            if i_level != 0:
                h = modules[m_idx](h)
                m_idx += 1
        assert not hs
        h = self.act(modules[m_idx](h))
        m_idx += 1
        h = modules[m_idx](h)
        m_idx += 1
        assert m_idx == len(modules)
        if self.scale_by_sigma:
            used_sigmas = self.sigmas[labels, None, None, None]
            h = h / used_sigmas
        return h


class DDPMRes64(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.act = act = get_act(config)
        self.register_buffer('sigmas', torch.tensor(utils.get_sigmas(config)))
        self.nf = nf = config.model.nf
        ch_mult = config.model.ch_mult
        self.num_res_blocks = num_res_blocks = config.model.num_res_blocks
        self.attn_resolutions = attn_resolutions = config.model.attn_resolutions
        dropout = config.model.dropout
        resamp_with_conv = config.model.resamp_with_conv
        self.num_resolutions = num_resolutions = len(ch_mult)
        self.all_resolutions = all_resolutions = [(config.data.image_size // 2 ** i) for i in range(num_resolutions)]
        AttnBlock = functools.partial(layers.AttnBlock)
        self.conditional = conditional = config.model.conditional
        ResnetBlock = functools.partial(ResnetBlockDDPM, act=act, temb_dim=4 * nf, dropout=dropout)
        if conditional:
            modules = [nn.Linear(nf, nf * 4)]
            modules[0].weight.data = default_initializer()(modules[0].weight.data.shape)
            nn.init.zeros_(modules[0].bias)
            modules.append(nn.Linear(nf * 4, nf * 4))
            modules[1].weight.data = default_initializer()(modules[1].weight.data.shape)
            nn.init.zeros_(modules[1].bias)
        self.centered = config.data.centered
        channels = config.data.num_channels
        self.img_size = img_size = config.data.image_size
        self.num_freq = int(np.log2(img_size))
        coord_x, coord_y, coord_z = torch.meshgrid(torch.arange(img_size), torch.arange(img_size), torch.arange(img_size))
        self.coords = torch.nn.Parameter(torch.stack([coord_x, coord_y, coord_z]).view(1, 3, img_size, img_size, img_size) * 0.0, requires_grad=False)
        self.mask = torch.nn.Parameter(torch.zeros(1, 1, img_size, img_size, img_size), requires_grad=False)
        self.pos_layer = conv3x3(3, nf)
        self.mask_layer = conv3x3(1, nf)
        modules.append(conv3x3(channels, nf))
        hs_c = [nf]
        in_ch = nf
        for i_level in range(num_resolutions):
            for i_block in range(num_res_blocks):
                out_ch = nf * ch_mult[i_level]
                modules.append(ResnetBlock(in_ch=in_ch, out_ch=out_ch))
                in_ch = out_ch
                if all_resolutions[i_level] in attn_resolutions:
                    modules.append(AttnBlock(channels=in_ch))
                hs_c.append(in_ch)
            if i_level != num_resolutions - 1:
                modules.append(Downsample(channels=in_ch, with_conv=resamp_with_conv))
                hs_c.append(in_ch)
        in_ch = hs_c[-1]
        modules.append(ResnetBlock(in_ch=in_ch))
        modules.append(AttnBlock(channels=in_ch))
        modules.append(ResnetBlock(in_ch=in_ch))
        for i_level in reversed(range(num_resolutions)):
            for i_block in range(num_res_blocks + 1):
                out_ch = nf * ch_mult[i_level]
                modules.append(ResnetBlock(in_ch=in_ch + hs_c.pop(), out_ch=out_ch))
                in_ch = out_ch
            if all_resolutions[i_level] in attn_resolutions:
                modules.append(AttnBlock(channels=in_ch))
            if i_level != 0:
                modules.append(Upsample(channels=in_ch, with_conv=resamp_with_conv))
        assert not hs_c
        modules.append(nn.GroupNorm(num_channels=in_ch, num_groups=32, eps=1e-06))
        modules.append(conv3x3(in_ch, channels, init_scale=0.0))
        self.all_modules = nn.ModuleList(modules)
        self.scale_by_sigma = config.model.scale_by_sigma

    def forward(self, x, labels):
        modules = self.all_modules
        m_idx = 0
        if self.conditional:
            timesteps = labels
            temb = layers.get_timestep_embedding(timesteps, self.nf)
            temb = modules[m_idx](temb)
            m_idx += 1
            temb = modules[m_idx](self.act(temb))
            m_idx += 1
        else:
            temb = None
        if self.centered:
            h = x
        else:
            h = 2 * x - 1.0
        hs = [modules[m_idx](h) + self.pos_layer(self.coords) + self.mask_layer(self.mask)]
        m_idx += 1
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = modules[m_idx](hs[-1], temb)
                m_idx += 1
                if h.shape[-1] in self.attn_resolutions:
                    h = modules[m_idx](h)
                    m_idx += 1
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(modules[m_idx](hs[-1]))
                m_idx += 1
        h = hs[-1]
        h = modules[m_idx](h, temb)
        m_idx += 1
        h = modules[m_idx](h)
        m_idx += 1
        h = modules[m_idx](h, temb)
        m_idx += 1
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                hspop = hs.pop()
                input = torch.cat([h, hspop], dim=1)
                h = modules[m_idx](input, temb)
                m_idx += 1
            if h.shape[-1] in self.attn_resolutions:
                h = modules[m_idx](h)
                m_idx += 1
            if i_level != 0:
                h = modules[m_idx](h)
                m_idx += 1
        assert not hs
        h = self.act(modules[m_idx](h))
        m_idx += 1
        h = modules[m_idx](h)
        m_idx += 1
        assert m_idx == len(modules)
        if self.scale_by_sigma:
            used_sigmas = self.sigmas[labels, None, None, None]
            h = h / used_sigmas
        return h


class Dense(nn.Module):
    """Linear layer with `default_init`."""

    def __init__(self):
        super().__init__()


def ncsn_conv3x3(in_planes, out_planes, stride=1, bias=True, dilation=1, init_scale=1.0, padding=1):
    """3x3 convolution with PyTorch initialization. Same as NCSNv1/NCSNv2."""
    init_scale = 1e-10 if init_scale == 0 else init_scale
    conv = nn.Conv3d(in_planes, out_planes, stride=stride, bias=bias, dilation=dilation, padding=padding, kernel_size=3)
    conv.weight.data *= init_scale
    conv.bias.data *= init_scale
    return conv


class CRPBlock(nn.Module):

    def __init__(self, features, n_stages, act=nn.ReLU(), maxpool=True):
        super().__init__()
        self.convs = nn.ModuleList()
        for i in range(n_stages):
            self.convs.append(ncsn_conv3x3(features, features, stride=1, bias=False))
        self.n_stages = n_stages
        if maxpool:
            self.pool = nn.MaxPool3d(kernel_size=5, stride=1, padding=2)
        else:
            self.pool = nn.AvgPool3d(kernel_size=5, stride=1, padding=2)
        self.act = act

    def forward(self, x):
        x = self.act(x)
        path = x
        for i in range(self.n_stages):
            path = self.pool(path)
            path = self.convs[i](path)
            x = path + x
        return x


class CondCRPBlock(nn.Module):

    def __init__(self, features, n_stages, num_classes, normalizer, act=nn.ReLU()):
        super().__init__()
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.normalizer = normalizer
        for i in range(n_stages):
            self.norms.append(normalizer(features, num_classes, bias=True))
            self.convs.append(ncsn_conv3x3(features, features, stride=1, bias=False))
        self.n_stages = n_stages
        self.pool = nn.AvgPool3d(kernel_size=5, stride=1, padding=2)
        self.act = act

    def forward(self, x, y):
        x = self.act(x)
        path = x
        for i in range(self.n_stages):
            path = self.norms[i](path, y)
            path = self.pool(path)
            path = self.convs[i](path)
            x = path + x
        return x


class RCUBlock(nn.Module):

    def __init__(self, features, n_blocks, n_stages, act=nn.ReLU()):
        super().__init__()
        for i in range(n_blocks):
            for j in range(n_stages):
                setattr(self, '{}_{}_conv'.format(i + 1, j + 1), ncsn_conv3x3(features, features, stride=1, bias=False))
        self.stride = 1
        self.n_blocks = n_blocks
        self.n_stages = n_stages
        self.act = act

    def forward(self, x):
        for i in range(self.n_blocks):
            residual = x
            for j in range(self.n_stages):
                x = self.act(x)
                x = getattr(self, '{}_{}_conv'.format(i + 1, j + 1))(x)
            x += residual
        return x


class CondRCUBlock(nn.Module):

    def __init__(self, features, n_blocks, n_stages, num_classes, normalizer, act=nn.ReLU()):
        super().__init__()
        for i in range(n_blocks):
            for j in range(n_stages):
                setattr(self, '{}_{}_norm'.format(i + 1, j + 1), normalizer(features, num_classes, bias=True))
                setattr(self, '{}_{}_conv'.format(i + 1, j + 1), ncsn_conv3x3(features, features, stride=1, bias=False))
        self.stride = 1
        self.n_blocks = n_blocks
        self.n_stages = n_stages
        self.act = act
        self.normalizer = normalizer

    def forward(self, x, y):
        for i in range(self.n_blocks):
            residual = x
            for j in range(self.n_stages):
                x = getattr(self, '{}_{}_norm'.format(i + 1, j + 1))(x, y)
                x = self.act(x)
                x = getattr(self, '{}_{}_conv'.format(i + 1, j + 1))(x)
            x += residual
        return x


class MSFBlock(nn.Module):

    def __init__(self, in_planes, features):
        super().__init__()
        assert isinstance(in_planes, list) or isinstance(in_planes, tuple)
        self.convs = nn.ModuleList()
        self.features = features
        for i in range(len(in_planes)):
            self.convs.append(ncsn_conv3x3(in_planes[i], features, stride=1, bias=True))

    def forward(self, xs, shape):
        sums = torch.zeros(xs[0].shape[0], self.features, *shape, device=xs[0].device)
        for i in range(len(self.convs)):
            h = self.convs[i](xs[i])
            h = F.interpolate(h, size=shape, mode='bilinear', align_corners=True)
            sums += h
        return sums


class CondMSFBlock(nn.Module):

    def __init__(self, in_planes, features, num_classes, normalizer):
        super().__init__()
        assert isinstance(in_planes, list) or isinstance(in_planes, tuple)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.features = features
        self.normalizer = normalizer
        for i in range(len(in_planes)):
            self.convs.append(ncsn_conv3x3(in_planes[i], features, stride=1, bias=True))
            self.norms.append(normalizer(in_planes[i], num_classes, bias=True))

    def forward(self, xs, y, shape):
        sums = torch.zeros(xs[0].shape[0], self.features, *shape, device=xs[0].device)
        for i in range(len(self.convs)):
            h = self.norms[i](xs[i], y)
            h = self.convs[i](h)
            h = F.interpolate(h, size=shape, mode='bilinear', align_corners=True)
            sums += h
        return sums


class RefineBlock(nn.Module):

    def __init__(self, in_planes, features, act=nn.ReLU(), start=False, end=False, maxpool=True):
        super().__init__()
        assert isinstance(in_planes, tuple) or isinstance(in_planes, list)
        self.n_blocks = n_blocks = len(in_planes)
        self.adapt_convs = nn.ModuleList()
        for i in range(n_blocks):
            self.adapt_convs.append(RCUBlock(in_planes[i], 2, 2, act))
        self.output_convs = RCUBlock(features, 3 if end else 1, 2, act)
        if not start:
            self.msf = MSFBlock(in_planes, features)
        self.crp = CRPBlock(features, 2, act, maxpool=maxpool)

    def forward(self, xs, output_shape):
        assert isinstance(xs, tuple) or isinstance(xs, list)
        hs = []
        for i in range(len(xs)):
            h = self.adapt_convs[i](xs[i])
            hs.append(h)
        if self.n_blocks > 1:
            h = self.msf(hs, output_shape)
        else:
            h = hs[0]
        h = self.crp(h)
        h = self.output_convs(h)
        return h


class CondRefineBlock(nn.Module):

    def __init__(self, in_planes, features, num_classes, normalizer, act=nn.ReLU(), start=False, end=False):
        super().__init__()
        assert isinstance(in_planes, tuple) or isinstance(in_planes, list)
        self.n_blocks = n_blocks = len(in_planes)
        self.adapt_convs = nn.ModuleList()
        for i in range(n_blocks):
            self.adapt_convs.append(CondRCUBlock(in_planes[i], 2, 2, num_classes, normalizer, act))
        self.output_convs = CondRCUBlock(features, 3 if end else 1, 2, num_classes, normalizer, act)
        if not start:
            self.msf = CondMSFBlock(in_planes, features, num_classes, normalizer)
        self.crp = CondCRPBlock(features, 2, num_classes, normalizer, act)

    def forward(self, xs, y, output_shape):
        assert isinstance(xs, tuple) or isinstance(xs, list)
        hs = []
        for i in range(len(xs)):
            h = self.adapt_convs[i](xs[i], y)
            hs.append(h)
        if self.n_blocks > 1:
            h = self.msf(hs, y, output_shape)
        else:
            h = hs[0]
        h = self.crp(h, y)
        h = self.output_convs(h, y)
        return h


class ConvMeanPool(nn.Module):

    def __init__(self, input_dim, output_dim, kernel_size=3, biases=True, adjust_padding=False):
        super().__init__()
        if not adjust_padding:
            conv = nn.Conv3d(input_dim, output_dim, kernel_size, stride=1, padding=kernel_size // 2, bias=biases)
            self.conv = conv
        else:
            conv = nn.Conv3d(input_dim, output_dim, kernel_size, stride=1, padding=kernel_size // 2, bias=biases)
            self.conv = nn.Sequential(nn.ZeroPad3d((1, 0, 1, 0)), conv)

    def forward(self, inputs):
        output = self.conv(inputs)
        output = sum([output[:, :, ::2, ::2], output[:, :, 1::2, ::2], output[:, :, ::2, 1::2], output[:, :, 1::2, 1::2]]) / 4.0
        return output


class MeanPoolConv(nn.Module):

    def __init__(self, input_dim, output_dim, kernel_size=3, biases=True):
        super().__init__()
        self.conv = nn.Conv3d(input_dim, output_dim, kernel_size, stride=1, padding=kernel_size // 2, bias=biases)

    def forward(self, inputs):
        output = inputs
        output = sum([output[:, :, ::2, ::2], output[:, :, 1::2, ::2], output[:, :, ::2, 1::2], output[:, :, 1::2, 1::2]]) / 4.0
        return self.conv(output)


class UpsampleConv(nn.Module):

    def __init__(self, input_dim, output_dim, kernel_size=3, biases=True):
        super().__init__()
        self.conv = nn.Conv3d(input_dim, output_dim, kernel_size, stride=1, padding=kernel_size // 2, bias=biases)
        self.pixelshuffle = nn.PixelShuffle(upscale_factor=2)

    def forward(self, inputs):
        output = inputs
        output = torch.cat([output, output, output, output], dim=1)
        output = self.pixelshuffle(output)
        return self.conv(output)


class ConditionalInstanceNorm3dPlus(nn.Module):

    def __init__(self, num_features, num_classes, bias=True):
        super().__init__()
        self.num_features = num_features
        self.bias = bias
        self.instance_norm = nn.InstanceNorm3d(num_features, affine=False, track_running_stats=False)
        if bias:
            self.embed = nn.Embedding(num_classes, num_features * 3)
            self.embed.weight.data[:, :2 * num_features].normal_(1, 0.02)
            self.embed.weight.data[:, 2 * num_features:].zero_()
        else:
            self.embed = nn.Embedding(num_classes, 2 * num_features)
            self.embed.weight.data.normal_(1, 0.02)

    def forward(self, x, y):
        means = torch.mean(x, dim=(2, 3, 4))
        m = torch.mean(means, dim=-1, keepdim=True)
        v = torch.var(means, dim=-1, keepdim=True)
        means = (means - m) / torch.sqrt(v + 1e-05)
        h = self.instance_norm(x)
        if self.bias:
            gamma, alpha, beta = self.embed(y).chunk(3, dim=-1)
            h = h + means[..., None, None, None] * alpha[..., None, None, None]
            out = gamma.view(-1, self.num_features, 1, 1, 1) * h + beta.view(-1, self.num_features, 1, 1, 1)
        else:
            gamma, alpha = self.embed(y).chunk(2, dim=-1)
            h = h + means[..., None, None, None] * alpha[..., None, None, None]
            out = gamma.view(-1, self.num_features, 1, 1, 1) * h
        return out


class ConditionalResidualBlock(nn.Module):

    def __init__(self, input_dim, output_dim, num_classes, resample=1, act=nn.ELU(), normalization=ConditionalInstanceNorm3dPlus, adjust_padding=False, dilation=None):
        super().__init__()
        self.non_linearity = act
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.resample = resample
        self.normalization = normalization
        if resample == 'down':
            if dilation > 1:
                self.conv1 = ncsn_conv3x3(input_dim, input_dim, dilation=dilation)
                self.normalize2 = normalization(input_dim, num_classes)
                self.conv2 = ncsn_conv3x3(input_dim, output_dim, dilation=dilation)
                conv_shortcut = partial(ncsn_conv3x3, dilation=dilation)
            else:
                self.conv1 = ncsn_conv3x3(input_dim, input_dim)
                self.normalize2 = normalization(input_dim, num_classes)
                self.conv2 = ConvMeanPool(input_dim, output_dim, 3, adjust_padding=adjust_padding)
                conv_shortcut = partial(ConvMeanPool, kernel_size=1, adjust_padding=adjust_padding)
        elif resample is None:
            if dilation > 1:
                conv_shortcut = partial(ncsn_conv3x3, dilation=dilation)
                self.conv1 = ncsn_conv3x3(input_dim, output_dim, dilation=dilation)
                self.normalize2 = normalization(output_dim, num_classes)
                self.conv2 = ncsn_conv3x3(output_dim, output_dim, dilation=dilation)
            else:
                conv_shortcut = nn.Conv3d
                self.conv1 = ncsn_conv3x3(input_dim, output_dim)
                self.normalize2 = normalization(output_dim, num_classes)
                self.conv2 = ncsn_conv3x3(output_dim, output_dim)
        else:
            raise Exception('invalid resample value')
        if output_dim != input_dim or resample is not None:
            self.shortcut = conv_shortcut(input_dim, output_dim)
        self.normalize1 = normalization(input_dim, num_classes)

    def forward(self, x, y):
        output = self.normalize1(x, y)
        output = self.non_linearity(output)
        output = self.conv1(output)
        output = self.normalize2(output, y)
        output = self.non_linearity(output)
        output = self.conv2(output)
        if self.output_dim == self.input_dim and self.resample is None:
            shortcut = x
        else:
            shortcut = self.shortcut(x)
        return shortcut + output


def ncsn_conv1x1(in_planes, out_planes, stride=1, bias=True, dilation=1, init_scale=1.0, padding=0):
    """1x1 convolution. Same as NCSNv1/v2."""
    conv = nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride, bias=bias, dilation=dilation, padding=padding)
    init_scale = 1e-10 if init_scale == 0 else init_scale
    conv.weight.data *= init_scale
    conv.bias.data *= init_scale
    return conv


class ResidualBlock(nn.Module):

    def __init__(self, input_dim, output_dim, resample=None, act=nn.ELU(), normalization=nn.InstanceNorm3d, adjust_padding=False, dilation=1):
        super().__init__()
        self.non_linearity = act
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.resample = resample
        self.normalization = normalization
        if resample == 'down':
            if dilation > 1:
                self.conv1 = ncsn_conv3x3(input_dim, input_dim, dilation=dilation)
                self.normalize2 = normalization(input_dim)
                self.conv2 = ncsn_conv3x3(input_dim, output_dim, dilation=dilation)
                conv_shortcut = partial(ncsn_conv3x3, dilation=dilation)
            else:
                self.conv1 = ncsn_conv3x3(input_dim, input_dim)
                self.normalize2 = normalization(input_dim)
                self.conv2 = ConvMeanPool(input_dim, output_dim, 3, adjust_padding=adjust_padding)
                conv_shortcut = partial(ConvMeanPool, kernel_size=1, adjust_padding=adjust_padding)
        elif resample is None:
            if dilation > 1:
                conv_shortcut = partial(ncsn_conv3x3, dilation=dilation)
                self.conv1 = ncsn_conv3x3(input_dim, output_dim, dilation=dilation)
                self.normalize2 = normalization(output_dim)
                self.conv2 = ncsn_conv3x3(output_dim, output_dim, dilation=dilation)
            else:
                conv_shortcut = partial(ncsn_conv1x1)
                self.conv1 = ncsn_conv3x3(input_dim, output_dim)
                self.normalize2 = normalization(output_dim)
                self.conv2 = ncsn_conv3x3(output_dim, output_dim)
        else:
            raise Exception('invalid resample value')
        if output_dim != input_dim or resample is not None:
            self.shortcut = conv_shortcut(input_dim, output_dim)
        self.normalize1 = normalization(input_dim)

    def forward(self, x):
        output = self.normalize1(x)
        output = self.non_linearity(output)
        output = self.conv1(output)
        output = self.normalize2(output)
        output = self.non_linearity(output)
        output = self.conv2(output)
        if self.output_dim == self.input_dim and self.resample is None:
            shortcut = x
        else:
            shortcut = self.shortcut(x)
        return shortcut + output


class AttnBlock(nn.Module):
    """Channel-wise self-attention block."""

    def __init__(self, channels):
        super().__init__()
        self.GroupNorm_0 = nn.GroupNorm(num_groups=32, num_channels=channels, eps=1e-06)
        self.NIN_0 = NIN(channels, channels)
        self.NIN_1 = NIN(channels, channels)
        self.NIN_2 = NIN(channels, channels)
        self.NIN_3 = NIN(channels, channels, init_scale=0.0)

    def forward(self, x):
        B, C, D, H, W = x.shape
        h = self.GroupNorm_0(x)
        q = self.NIN_0(h)
        k = self.NIN_1(h)
        v = self.NIN_2(h)
        w = torch.einsum('bcdhw,bckij->bdhwkij', q, k) * int(C) ** -0.5
        w = torch.reshape(w, (B, D, H, W, D * H * W))
        w = F.softmax(w, dim=-1)
        w = torch.reshape(w, (B, D, H, W, D, H, W))
        h = torch.einsum('bdhwkij,bckij->bcdhw', w, v)
        h = self.NIN_3(h)
        return x + h


class ResnetBlockDDPMPosEncoding(nn.Module):
    """The ResNet Blocks used in DDPM."""

    def __init__(self, act, in_ch, out_ch=None, temb_dim=None, conv_shortcut=False, dropout=0.1, img_size=64):
        super().__init__()
        coord_x, coord_y, coord_z = torch.meshgrid(torch.arange(img_size), torch.arange(img_size), torch.arange(img_size))
        coords = torch.stack([coord_x, coord_y, coord_z])
        self.num_freq = int(np.log2(img_size))
        pos_encoding = torch.zeros(1, 2 * self.num_freq, 3, img_size, img_size, img_size)
        with torch.no_grad():
            for i in range(self.num_freq):
                pos_encoding[0, 2 * i, :, :, :, :] = torch.cos((i + 1) * np.pi * coords)
                pos_encoding[0, 2 * i + 1, :, :, :, :] = torch.sin((i + 1) * np.pi * coords)
        self.pos_encoding = nn.Parameter(pos_encoding.view(1, 2 * self.num_freq * 3, img_size, img_size, img_size) / img_size, requires_grad=False)
        if out_ch is None:
            out_ch = in_ch
        self.GroupNorm_0 = nn.GroupNorm(num_groups=32, num_channels=in_ch, eps=1e-06)
        self.act = act
        self.Conv_0 = ddpm_conv3x3(in_ch, out_ch)
        self.Conv_0_pos = ddpm_conv3x3(2 * self.num_freq * 3, out_ch)
        if temb_dim is not None:
            self.Dense_0 = nn.Linear(temb_dim, out_ch)
            self.Dense_0.weight.data = default_init()(self.Dense_0.weight.data.shape)
            nn.init.zeros_(self.Dense_0.bias)
        self.GroupNorm_1 = nn.GroupNorm(num_groups=32, num_channels=out_ch, eps=1e-06)
        self.Dropout_0 = nn.Dropout(dropout)
        self.Conv_1 = ddpm_conv3x3(out_ch, out_ch, init_scale=0.0)
        if in_ch != out_ch:
            if conv_shortcut:
                self.Conv_2 = ddpm_conv3x3(in_ch, out_ch)
            else:
                self.NIN_0 = NIN(in_ch, out_ch)
        self.out_ch = out_ch
        self.in_ch = in_ch
        self.conv_shortcut = conv_shortcut

    def forward(self, x, temb=None):
        B, C, D, H, W = x.shape
        assert C == self.in_ch
        out_ch = self.out_ch if self.out_ch else self.in_ch
        h = self.act(self.GroupNorm_0(x))
        h = self.Conv_0(h) + self.Conv_0_pos(self.pos_encoding).expand(h.size(0), -1, -1, -1, -1)
        if temb is not None:
            h += self.Dense_0(self.act(temb))[:, :, None, None, None]
        h = self.act(self.GroupNorm_1(h))
        h = self.Dropout_0(h)
        h = self.Conv_1(h)
        if C != out_ch:
            if self.conv_shortcut:
                x = self.Conv_2(x)
            else:
                x = self.NIN_0(x)
        return x + h


class ConditionalBatchNorm3d(nn.Module):

    def __init__(self, num_features, num_classes, bias=True):
        super().__init__()
        self.num_features = num_features
        self.bias = bias
        self.bn = nn.BatchNorm3d(num_features, affine=False)
        if self.bias:
            self.embed = nn.Embedding(num_classes, num_features * 2)
            self.embed.weight.data[:, :num_features].uniform_()
            self.embed.weight.data[:, num_features:].zero_()
        else:
            self.embed = nn.Embedding(num_classes, num_features)
            self.embed.weight.data.uniform_()

    def forward(self, x, y):
        out = self.bn(x)
        if self.bias:
            gamma, beta = self.embed(y).chunk(2, dim=1)
            out = gamma.view(-1, self.num_features, 1, 1, 1) * out + beta.view(-1, self.num_features, 1, 1, 1)
        else:
            gamma = self.embed(y)
            out = gamma.view(-1, self.num_features, 1, 1, 1) * out
        return out


class ConditionalInstanceNorm3d(nn.Module):

    def __init__(self, num_features, num_classes, bias=True):
        super().__init__()
        self.num_features = num_features
        self.bias = bias
        self.instance_norm = nn.InstanceNorm3d(num_features, affine=False, track_running_stats=False)
        if bias:
            self.embed = nn.Embedding(num_classes, num_features * 2)
            self.embed.weight.data[:, :num_features].uniform_()
            self.embed.weight.data[:, num_features:].zero_()
        else:
            self.embed = nn.Embedding(num_classes, num_features)
            self.embed.weight.data.uniform_()

    def forward(self, x, y):
        h = self.instance_norm(x)
        if self.bias:
            gamma, beta = self.embed(y).chunk(2, dim=-1)
            out = gamma.view(-1, self.num_features, 1, 1, 1) * h + beta.view(-1, self.num_features, 1, 1, 1)
        else:
            gamma = self.embed(y)
            out = gamma.view(-1, self.num_features, 1, 1, 1) * h
        return out


class ConditionalVarianceNorm3d(nn.Module):

    def __init__(self, num_features, num_classes, bias=False):
        super().__init__()
        self.num_features = num_features
        self.bias = bias
        self.embed = nn.Embedding(num_classes, num_features)
        self.embed.weight.data.normal_(1, 0.02)

    def forward(self, x, y):
        vars = torch.var(x, dim=(2, 3, 4), keepdim=True)
        h = x / torch.sqrt(vars + 1e-05)
        gamma = self.embed(y)
        out = gamma.view(-1, self.num_features, 1, 1, 1) * h
        return out


class VarianceNorm3d(nn.Module):

    def __init__(self, num_features, bias=False):
        super().__init__()
        self.num_features = num_features
        self.bias = bias
        self.alpha = nn.Parameter(torch.zeros(num_features))
        self.alpha.data.normal_(1, 0.02)

    def forward(self, x):
        vars = torch.var(x, dim=(2, 3, 4), keepdim=True)
        h = x / torch.sqrt(vars + 1e-05)
        out = self.alpha.view(-1, self.num_features, 1, 1, 1) * h
        return out


class ConditionalNoneNorm3d(nn.Module):

    def __init__(self, num_features, num_classes, bias=True):
        super().__init__()
        self.num_features = num_features
        self.bias = bias
        if bias:
            self.embed = nn.Embedding(num_classes, num_features * 2)
            self.embed.weight.data[:, :num_features].uniform_()
            self.embed.weight.data[:, num_features:].zero_()
        else:
            self.embed = nn.Embedding(num_classes, num_features)
            self.embed.weight.data.uniform_()

    def forward(self, x, y):
        if self.bias:
            gamma, beta = self.embed(y).chunk(2, dim=-1)
            out = gamma.view(-1, self.num_features, 1, 1, 1) * x + beta.view(-1, self.num_features, 1, 1, 1)
        else:
            gamma = self.embed(y)
            out = gamma.view(-1, self.num_features, 1, 1, 1) * x
        return out


class NoneNorm3d(nn.Module):

    def __init__(self, num_features, bias=True):
        super().__init__()

    def forward(self, x):
        return x


class InstanceNorm3dPlus(nn.Module):

    def __init__(self, num_features, bias=True):
        super().__init__()
        self.num_features = num_features
        self.bias = bias
        self.instance_norm = nn.InstanceNorm3d(num_features, affine=False, track_running_stats=False)
        self.alpha = nn.Parameter(torch.zeros(num_features))
        self.gamma = nn.Parameter(torch.zeros(num_features))
        self.alpha.data.normal_(1, 0.02)
        self.gamma.data.normal_(1, 0.02)
        if bias:
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        means = torch.mean(x, dim=(2, 3, 4))
        m = torch.mean(means, dim=-1, keepdim=True)
        v = torch.var(means, dim=-1, keepdim=True)
        means = (means - m) / torch.sqrt(v + 1e-05)
        h = self.instance_norm(x)
        if self.bias:
            h = h + means[..., None, None, None] * self.alpha[..., None, None, None]
            out = self.gamma.view(-1, self.num_features, 1, 1, 1) * h + self.beta.view(-1, self.num_features, 1, 1, 1)
        else:
            h = h + means[..., None, None, None] * self.alpha[..., None, None, None]
            out = self.gamma.view(-1, self.num_features, 1, 1, 1) * h
        return out


def _setup_kernel(k):
    k = np.asarray(k, dtype=np.float32)
    if k.ndim == 1:
        k = np.outer(k, k)
    k /= np.sum(k)
    assert k.ndim == 2
    assert k.shape[0] == k.shape[1]
    return k


def conv_downsample_3d(x, w, k=None, factor=2, gain=1):
    """Fused `tf.nn.conv3d()` followed by `downsample_3d()`.

    Padding is performed only once at the beginning, not between the operations.
    The fused op is considerably more efficient than performing the same
    calculation
    using standard TensorFlow ops. It supports gradients of arbitrary order.
    Args:
        x:            Input tensor of the shape `[N, C, H, W]` or `[N, H, W,
          C]`.
        w:            Weight tensor of the shape `[filterH, filterW, inChannels,
          outChannels]`. Grouped convolution can be performed by `inChannels =
          x.shape[0] // numGroups`.
        k:            FIR filter of the shape `[firH, firW]` or `[firN]`
          (separable). The default is `[1] * factor`, which corresponds to
          average pooling.
        factor:       Integer downsampling factor (default: 2).
        gain:         Scaling factor for signal magnitude (default: 1.0).

    Returns:
        Tensor of the shape `[N, C, H // factor, W // factor]` or
        `[N, H // factor, W // factor, C]`, and same datatype as `x`.
  """
    assert isinstance(factor, int) and factor >= 1
    _outC, _inC, convH, convW = w.shape
    assert convW == convH
    if k is None:
        k = [1] * factor
    k = _setup_kernel(k) * gain
    p = k.shape[0] - factor + (convW - 1)
    s = [factor, factor]
    x = upfirdn2d(x, torch.tensor(k, device=x.device), pad=((p + 1) // 2, p // 2))
    return F.conv3d(x, w, stride=s, padding=0)


def _shape(x, dim):
    return x.shape[dim]


def upsample_conv_3d(x, w, k=None, factor=2, gain=1):
    """Fused `upsample_3d()` followed by `tf.nn.conv3d()`.

     Padding is performed only once at the beginning, not between the
     operations.
     The fused op is considerably more efficient than performing the same
     calculation
     using standard TensorFlow ops. It supports gradients of arbitrary order.
     Args:
       x:            Input tensor of the shape `[N, C, H, W]` or `[N, H, W,
         C]`.
       w:            Weight tensor of the shape `[filterH, filterW, inChannels,
         outChannels]`. Grouped convolution can be performed by `inChannels =
         x.shape[0] // numGroups`.
       k:            FIR filter of the shape `[firH, firW]` or `[firN]`
         (separable). The default is `[1] * factor`, which corresponds to
         nearest-neighbor upsampling.
       factor:       Integer upsampling factor (default: 2).
       gain:         Scaling factor for signal magnitude (default: 1.0).

     Returns:
       Tensor of the shape `[N, C, H * factor, W * factor]` or
       `[N, H * factor, W * factor, C]`, and same datatype as `x`.
  """
    assert isinstance(factor, int) and factor >= 1
    assert len(w.shape) == 5
    convD = w.shape[2]
    convH = w.shape[3]
    convW = w.shape[4]
    inC = w.shape[1]
    outC = w.shape[0]
    assert convW == convH
    if k is None:
        k = [1] * factor
    k = _setup_kernel(k) * (gain * factor ** 2)
    p = k.shape[0] - factor - (convW - 1)
    stride = factor, factor
    stride = [1, 1, factor, factor]
    output_shape = (_shape(x, 2) - 1) * factor + convD, (_shape(x, 3) - 1) * factor + convH, (_shape(x, 4) - 1) * factor + convW
    output_padding = output_shape[0] - (_shape(x, 2) - 1) * stride[0] - convD, output_shape[1] - (_shape(x, 3) - 1) * stride[1] - convH, output_shape[2] - (_shape(x, 4) - 1) * stride[2] - convW
    assert output_padding[0] >= 0 and output_padding[1] >= 0 and output_padding[2] >= 0
    num_groups = _shape(x, 1) // inC
    w = torch.reshape(w, (num_groups, -1, inC, convD, convH, convW))
    w = w[..., ::-1, ::-1].permute(0, 2, 1, 3, 4, 5)
    w = torch.reshape(w, (num_groups * inC, -1, convD, convH, convW))
    x = F.conv_transpose3d(x, w, stride=stride, output_padding=output_padding, padding=0)
    return upfirdn2d(x, torch.tensor(k, device=x.device), pad=((p + 1) // 2 + factor - 1, p // 2 + 1))


class Conv3d(nn.Module):
    """Conv3d layer with optimal upsampling and downsampling (StyleGAN2)."""

    def __init__(self, in_ch, out_ch, kernel, up=False, down=False, resample_kernel=(1, 3, 3, 1), use_bias=True, kernel_init=None):
        super().__init__()
        assert not (up and down)
        assert kernel >= 1 and kernel % 2 == 1
        self.weight = nn.Parameter(torch.zeros(out_ch, in_ch, kernel, kernel, kernel))
        if kernel_init is not None:
            self.weight.data = kernel_init(self.weight.data.shape)
        if use_bias:
            self.bias = nn.Parameter(torch.zeros(out_ch))
        self.up = up
        self.down = down
        self.resample_kernel = resample_kernel
        self.kernel = kernel
        self.use_bias = use_bias

    def forward(self, x):
        if self.up:
            x = upsample_conv_3d(x, self.weight, k=self.resample_kernel)
        elif self.down:
            x = conv_downsample_3d(x, self.weight, k=self.resample_kernel)
        else:
            x = F.conv3d(x, self.weight, stride=1, padding=self.kernel // 2)
        if self.use_bias:
            x = x + self.bias.reshape(1, -1, 1, 1, 1)
        return x


class Trainer(torch.nn.Module):

    def __init__(self, glctx, geometry, lgt, mat, optimize_geometry, optimize_light, image_loss_fn, FLAGS):
        super(Trainer, self).__init__()
        self.glctx = glctx
        self.geometry = geometry
        self.light = lgt
        self.material = mat
        self.optimize_geometry = optimize_geometry
        self.optimize_light = optimize_light
        self.image_loss_fn = image_loss_fn
        self.FLAGS = FLAGS
        if not self.optimize_light:
            with torch.no_grad():
                self.light.build_mips()
        self.params = list(self.material.parameters())
        self.params += list(self.light.parameters()) if optimize_light else []
        self.geo_params = list(self.geometry.parameters()) if optimize_geometry else []
        try:
            self.sdf_params = [self.geometry.sdf]
        except:
            self.sdf_params = []
        self.deform_params = [self.geometry.deform]

    def forward(self, target, it):
        if self.optimize_light:
            self.light.build_mips()
            if self.FLAGS.camera_space_light:
                self.light.xfm(target['mv'])
        self.light.xfm(target['envlight_transform'])
        return self.geometry.tick(glctx, target, self.light, self.material, self.image_loss_fn, it, xfm_lgt=target['envlight_transform'], no_depth_thin=False)


class Buffer(object):

    def __init__(self, shape, capacity, device) ->None:
        self.len_curr = 0
        self.pointer = 0
        self.capacity = capacity
        self.buffer = torch.zeros((capacity,) + shape, device=device)

    def push(self, x):
        """
            Push one single data point into the buffer
        """
        self.buffer[self.pointer] = x
        self.pointer = (self.pointer + 1) % self.capacity
        if self.len_curr < self.capacity:
            self.len_curr += 1

    def avg(self):
        return torch.sign(torch.sign(self.buffer[:self.len_curr]).float().mean(dim=0)).float()


class DMTet:

    def __init__(self):
        self.triangle_table = torch.tensor([[-1, -1, -1, -1, -1, -1], [1, 0, 2, -1, -1, -1], [4, 0, 3, -1, -1, -1], [1, 4, 2, 1, 3, 4], [3, 1, 5, -1, -1, -1], [2, 3, 0, 2, 5, 3], [1, 4, 0, 1, 5, 4], [4, 2, 5, -1, -1, -1], [4, 5, 2, -1, -1, -1], [4, 1, 0, 4, 5, 1], [3, 2, 0, 3, 5, 2], [1, 3, 5, -1, -1, -1], [4, 1, 2, 4, 3, 1], [3, 0, 4, -1, -1, -1], [2, 0, 1, -1, -1, -1], [-1, -1, -1, -1, -1, -1]], dtype=torch.long, device='cuda')
        self.num_triangles_table = torch.tensor([0, 1, 1, 2, 1, 2, 2, 1, 1, 2, 2, 1, 2, 1, 1, 0], dtype=torch.long, device='cuda')
        self.base_tet_edges = torch.tensor([0, 1, 0, 2, 0, 3, 1, 2, 1, 3, 2, 3], dtype=torch.long, device='cuda')

    def sort_edges(self, edges_ex2):
        with torch.no_grad():
            order = (edges_ex2[:, 0] > edges_ex2[:, 1]).long()
            order = order.unsqueeze(dim=1)
            a = torch.gather(input=edges_ex2, index=order, dim=1)
            b = torch.gather(input=edges_ex2, index=1 - order, dim=1)
        return torch.stack([a, b], -1)

    def map_uv(self, faces, face_gidx, max_idx):
        N = int(np.ceil(np.sqrt((max_idx + 1) // 2)))
        tex_y, tex_x = torch.meshgrid(torch.linspace(0, 1 - 1 / N, N, dtype=torch.float32, device='cuda'), torch.linspace(0, 1 - 1 / N, N, dtype=torch.float32, device='cuda'), indexing='ij')
        pad = 0.9 / N
        uvs = torch.stack([tex_x, tex_y, tex_x + pad, tex_y, tex_x + pad, tex_y + pad, tex_x, tex_y + pad], dim=-1).view(-1, 2)

        def _idx(tet_idx, N):
            x = tet_idx % N
            y = torch.div(tet_idx, N, rounding_mode='trunc')
            return y * N + x
        tet_idx = _idx(torch.div(face_gidx, 2, rounding_mode='trunc'), N)
        tri_idx = face_gidx % 2
        uv_idx = torch.stack((tet_idx * 4, tet_idx * 4 + tri_idx + 1, tet_idx * 4 + tri_idx + 2), dim=-1).view(-1, 3)
        return uvs, uv_idx

    def __call__(self, pos_nx3, sdf_n, tet_fx4):
        with torch.no_grad():
            occ_n = sdf_n > 0
            occ_fx4 = occ_n[tet_fx4.reshape(-1)].reshape(-1, 4)
            occ_sum = torch.sum(occ_fx4, -1)
            valid_tets = (occ_sum > 0) & (occ_sum < 4)
            occ_sum = occ_sum[valid_tets]
            all_edges = tet_fx4[valid_tets][:, self.base_tet_edges].reshape(-1, 2)
            all_edges = self.sort_edges(all_edges)
            unique_edges, idx_map = torch.unique(all_edges, dim=0, return_inverse=True)
            unique_edges = unique_edges.long()
            mask_edges = occ_n[unique_edges.reshape(-1)].reshape(-1, 2).sum(-1) == 1
            mapping = torch.ones(unique_edges.shape[0], dtype=torch.long, device='cuda') * -1
            mapping[mask_edges] = torch.arange(mask_edges.sum(), dtype=torch.long, device='cuda')
            idx_map = mapping[idx_map]
            interp_v = unique_edges[mask_edges]
        edges_to_interp = pos_nx3[interp_v.reshape(-1)].reshape(-1, 2, 3)
        edges_to_interp_sdf = sdf_n[interp_v.reshape(-1)].reshape(-1, 2, 1)
        edges_to_interp_sdf[:, -1] *= -1
        denominator = edges_to_interp_sdf.sum(1, keepdim=True)
        edges_to_interp_sdf = torch.flip(edges_to_interp_sdf, [1]) / denominator
        verts = (edges_to_interp * edges_to_interp_sdf).sum(1)
        idx_map = idx_map.reshape(-1, 6)
        v_id = torch.pow(2, torch.arange(4, dtype=torch.long, device='cuda'))
        tetindex = (occ_fx4[valid_tets] * v_id.unsqueeze(0)).sum(-1)
        num_triangles = self.num_triangles_table[tetindex]
        faces = torch.cat((torch.gather(input=idx_map[num_triangles == 1], dim=1, index=self.triangle_table[tetindex[num_triangles == 1]][:, :3]).reshape(-1, 3), torch.gather(input=idx_map[num_triangles == 2], dim=1, index=self.triangle_table[tetindex[num_triangles == 2]][:, :6]).reshape(-1, 3)), dim=0)
        num_tets = tet_fx4.shape[0]
        tet_gidx = torch.arange(num_tets, dtype=torch.long, device='cuda')[valid_tets]
        face_gidx = torch.cat((tet_gidx[num_triangles == 1] * 2, torch.stack((tet_gidx[num_triangles == 2] * 2, tet_gidx[num_triangles == 2] * 2 + 1), dim=-1).view(-1)), dim=0)
        uvs, uv_idx = self.map_uv(faces, face_gidx, num_tets * 2)
        face_to_valid_tet = torch.cat((tet_gidx[num_triangles == 1], torch.stack((tet_gidx[num_triangles == 2], tet_gidx[num_triangles == 2]), dim=-1).view(-1)), dim=0)
        valid_vert_idx = tet_fx4[tet_gidx[num_triangles > 0]].long().unique()
        return verts, faces, uvs, uv_idx, face_to_valid_tet.long(), valid_vert_idx


def sdf_reg_loss(sdf, all_edges):
    sdf_f1x6x2 = sdf[all_edges.reshape(-1)].reshape(-1, 2)
    mask = torch.sign(sdf_f1x6x2[..., 0]) != torch.sign(sdf_f1x6x2[..., 1])
    sdf_f1x6x2 = sdf_f1x6x2[mask]
    sdf_diff = torch.nn.functional.binary_cross_entropy_with_logits(sdf_f1x6x2[..., 0], (sdf_f1x6x2[..., 1] > 0).float()) + torch.nn.functional.binary_cross_entropy_with_logits(sdf_f1x6x2[..., 1], (sdf_f1x6x2[..., 0] > 0).float())
    return sdf_diff


class DMTetGeometry(torch.nn.Module):

    def __init__(self, grid_res, scale, FLAGS, root='./', grid_to_tet=None, deform_scale=2.0, **kwargs):
        super(DMTetGeometry, self).__init__()
        self.FLAGS = FLAGS
        self.grid_res = grid_res
        self.marching_tets = DMTet()
        self.cropped = True
        self.tanh = False
        self.deform_scale = deform_scale
        self.grid_to_tet = grid_to_tet
        if self.cropped:
            None
            tets = np.load(os.path.join(root, 'data/tets/{}_tets_cropped.npz'.format(self.grid_res)))
        else:
            tets = np.load(os.path.join(root, 'data/tets/{}_tets.npz'.format(self.grid_res)))
        None
        self.verts = torch.tensor(tets['vertices'], dtype=torch.float32, device='cuda') * scale
        self.indices = torch.tensor(tets['indices'], dtype=torch.long, device='cuda')
        self.generate_edges()
        sdf = torch.rand_like(self.verts[:, 0]).clamp(-1.0, 1.0) - 0.1
        self.sdf = torch.nn.Parameter(sdf.clone().detach(), requires_grad=True)
        self.register_parameter('sdf', self.sdf)
        self.deform = torch.nn.Parameter(torch.zeros_like(self.verts), requires_grad=True)
        self.register_parameter('deform', self.deform)
        self.alpha = None
        self.sdf_ema = torch.nn.Parameter(sdf.clone().detach(), requires_grad=False)
        self.deform_ema = torch.nn.Parameter(self.deform.clone().detach(), requires_grad=False)
        self.ema_coeff = 0.9
        self.sdf_buffer = Buffer(sdf.size(), capacity=200, device='cuda')

    def generate_edges(self):
        with torch.no_grad():
            edges = torch.tensor([0, 1, 0, 2, 0, 3, 1, 2, 1, 3, 2, 3], dtype=torch.long, device='cuda')
            all_edges = self.indices[:, edges].reshape(-1, 2)
            all_edges_sorted = torch.sort(all_edges, dim=1)[0]
            self.all_edges = torch.unique(all_edges_sorted, dim=0)

    def getAABB(self):
        return torch.min(self.verts, dim=0).values, torch.max(self.verts, dim=0).values

    def getVertNNDist(self):
        v_deformed = (self.verts + 2 / (self.grid_res * 2) * torch.tanh(self.deform)).unsqueeze(0)
        return pytorch3d.ops.knn.knn_points(v_deformed, v_deformed, K=2).dists[0, :, -1].detach()

    def getTetCenters(self):
        v_deformed = self.get_deformed()
        face_verts = v_deformed[self.indices]
        face_centers = face_verts.mean(dim=1)
        return face_centers

    def getValidTetIdx(self):
        v_deformed = self.get_deformed()
        verts, faces, uvs, uv_idx, tet_gidx, valid_vert_idx = self.marching_tets(v_deformed, self.sdf, self.indices)
        return tet_gidx.long()

    def getValidVertsIdx(self):
        v_deformed = self.get_deformed()
        verts, faces, uvs, uv_idx, tet_gidx, valid_vert_idx = self.marching_tets(v_deformed, self.sdf, self.indices)
        return self.indices[tet_gidx.long()].unique()

    def getMesh(self, material, noise=0.0, ema=False):
        v_deformed = self.get_deformed(ema=ema)
        if ema:
            sdf = self.sdf_ema
        else:
            sdf = self.sdf
        verts, faces, uvs, uv_idx, tet_gidx, valid_vert_idx = self.marching_tets(v_deformed, sdf, self.indices)
        imesh = mesh.Mesh(verts, faces, v_tex=uvs, t_tex_idx=uv_idx, material=material)
        if material is not None:
            imesh = mesh.auto_normals(imesh)
            imesh = mesh.compute_tangents(imesh)
        imesh.valid_vert_idx = valid_vert_idx
        return imesh

    def getMesh_no_deform(self, material, noise=0.0, ema=False):
        if ema:
            sdf = self.sdf_ema
        else:
            sdf = self.sdf
        verts, faces, uvs, uv_idx, tet_gidx, valid_vert_idx = self.marching_tets(self.verts, torch.sign(sdf), self.indices)
        imesh = mesh.Mesh(verts, faces, v_tex=uvs, t_tex_idx=uv_idx, material=material)
        imesh = mesh.auto_normals(imesh)
        imesh = mesh.compute_tangents(imesh)
        return imesh

    def getMesh_no_deform_gd(self, material, noise=0.0, ema=False):
        v_deformed = self.get_deformed(no_grad=True)
        if ema:
            sdf = self.sdf_ema
        else:
            sdf = self.sdf
        verts, faces, uvs, uv_idx, tet_gidx, valid_vert_idx = self.marching_tets(v_deformed, sdf, self.indices)
        imesh = mesh.Mesh(verts, faces, v_tex=uvs, t_tex_idx=uv_idx, material=material)
        imesh = mesh.auto_normals(imesh)
        imesh = mesh.compute_tangents(imesh)
        return imesh

    def get_deformed(self, no_grad=False, ema=False):
        if no_grad:
            deform = self.deform.detach()
        else:
            deform = self.deform
        if self.tanh:
            v_deformed = self.verts + 2 / (self.grid_res * 2) * torch.tanh(deform) * self.deform_scale
        else:
            v_deformed = self.verts + 2 / (self.grid_res * 2) * deform * self.deform_scale
        return v_deformed

    def get_angle(self):
        with torch.no_grad():
            comb_list = [(0, 1, 2, 3), (0, 1, 3, 2), (0, 2, 3, 1), (1, 2, 3, 0)]
            directions = torch.zeros(self.indices.size(0), 4)
            dir_vec = torch.zeros(self.indices.size(0), 4, 3)
            vert_inds = torch.zeros(self.indices.size(0), 4).long()
            count = 0
            vpos_list = self.get_deformed()
            for comb in comb_list:
                face = self.indices[:, comb[:3]]
                face_pos = vpos_list[face, :]
                face_center = face_pos.mean(1, keepdim=False)
                v = self.indices[:, comb[3]]
                test_vec = vpos_list[v]
                ref_vec = render_utils.safe_normalize(vpos_list[face[:, 0]] - face_center)
                distance_vec = test_vec - render_utils.dot(test_vec, ref_vec) * ref_vec
                directions[:, count] = torch.sign(render_utils.dot(test_vec, distance_vec)[:, 0])
                dir_vec[:, count, :] = distance_vec
                vert_inds[:, count] = v
                count += 1
            return directions, dir_vec, vert_inds

    def clamp_deform(self):
        if not self.tanh:
            self.deform.data[:] = self.deform.data.clamp(-0.99, 0.99)
            self.sdf.data[:] = self.sdf.data.clamp(-1.0, 1.0)

    def render(self, glctx, target, lgt, opt_material, bsdf=None, ema=False, xfm_lgt=None, get_visible_tets=False):
        opt_mesh = self.getMesh(opt_material, ema=ema)
        tet_centers = self.getTetCenters() if get_visible_tets else None
        return render.render_mesh(glctx, opt_mesh, target['mvp'], target['campos'], lgt, target['resolution'], spp=target['spp'], msaa=True, background=target['background'], bsdf=bsdf, xfm_lgt=xfm_lgt, tet_centers=tet_centers)

    def render_with_mesh(self, glctx, target, lgt, opt_material, bsdf=None, noise=0.0, ema=False, xfm_lgt=None):
        opt_mesh = self.getMesh(opt_material, noise=noise, ema=ema)
        return opt_mesh, render.render_mesh(glctx, opt_mesh, target['mvp'], target['campos'], lgt, target['resolution'], spp=target['spp'], msaa=True, background=target['background'], bsdf=bsdf, xfm_lgt=xfm_lgt)

    def update_ema(self, ema_coeff=0.9):
        self.sdf_buffer.push(self.sdf)
        self.sdf_ema.data[:] = self.sdf_buffer.avg()
        self.deform_ema.data[:] = self.deform.data[:]

    def render_ema(self, glctx, target, lgt, opt_material, bsdf=None, xfm_lgt=None):
        opt_mesh = self.getMesh(opt_material, ema=True)
        return render.render_mesh(glctx, opt_mesh, target['mvp'], target['campos'], lgt, target['resolution'], spp=target['spp'], msaa=True, background=target['background'], bsdf=bsdf, xfm_lgt=xfm_lgt)

    def init_with_gt_surface(self, gt_verts, surface_faces, campos):
        with torch.no_grad():
            surface_face_verts = gt_verts[surface_faces]
            surface_centers = surface_face_verts.mean(dim=1)
            v_pos = self.get_deformed()
            results = pytorch3d.ops.knn_points(v_pos[None, ...], surface_centers[None, ...])
            dists, nn_idx = results.dists, results.idx
            displacement = v_pos - surface_centers[nn_idx[0, :, 0]]
            view_dirs = campos - surface_centers
            normals = torch.cross(surface_face_verts[:, 0] - surface_face_verts[:, 1], surface_face_verts[:, 0] - surface_face_verts[:, 2])
            mask = ((normals * view_dirs).sum(dim=-1, keepdim=True) >= 0).float()
            normals = normals * mask - normals * (1 - mask)
            outside_verts_idx = (displacement * normals[nn_idx[0, :, 0]]).sum(dim=-1) > 0
            self.sdf.data[outside_verts_idx] = 1.0

    def tick(self, glctx, target, lgt, opt_material, loss_fn, iteration, with_reg=True, xfm_lgt=None, no_depth_thin=True):
        if iteration < 100:
            self.deform.requires_grad = False
            self.deform_scale = 2.0
        else:
            self.deform.requires_grad = True
            self.deform_scale = 2.0
        if iteration > 200 and iteration < 2000 and iteration % 20 == 0:
            with torch.no_grad():
                v_pos = self.get_deformed()
                v_pos_camera_homo = ru.xfm_points(v_pos[None, ...], target['mvp'])
                v_pos_camera = v_pos_camera_homo[:, :, :2] / v_pos_camera_homo[:, :, -1:]
                v_pos_camera_discrete = ((v_pos_camera * 0.5 + 0.5).clip(0, 1) * (target['resolution'][0] - 1)).long()
                target_mask = target['mask_cont'][:, :, :, 0] == 0
                for k in range(target_mask.size(0)):
                    assert v_pos_camera_discrete[k].min() >= 0 and v_pos_camera_discrete[k].max() < target['resolution'][0]
                    v_mask = target_mask[k, v_pos_camera_discrete[k, :, 1], v_pos_camera_discrete[k, :, 0]].view(v_pos.size(0))
                    self.sdf.data[v_mask] = self.sdf.data[v_mask].abs().clamp(0.0, 1.0)
        imesh, buffers = self.render_with_mesh(glctx, target, lgt, opt_material, noise=0.0, xfm_lgt=xfm_lgt)
        color_ref = target['img']
        img_loss = torch.tensor(0.0)
        img_loss = torch.nn.functional.mse_loss(buffers['shaded'][..., 3:], color_ref[..., 3:])
        img_loss = img_loss + loss_fn(buffers['shaded'][..., 0:3] * color_ref[..., 3:], color_ref[..., 0:3] * color_ref[..., 3:])
        mask = (target['mask_cont'][:, :, :, 0] == 1.0).float()
        mask_curr = (buffers['mask_cont'][:, :, :, 0] == 1.0).float()
        if iteration % 300 == 0 and iteration < 1790:
            self.deform.data[:] *= 0.4
        if no_depth_thin:
            valid_depth_mask = ((target['depth_second'] >= 0).float() * ((target['depth_second'] - target['depth']).abs() >= 0.005).float()).detach()
        else:
            valid_depth_mask = 1.0
        depth_diff = (buffers['depth'][:, :, :, :1] - target['depth'][:, :, :, :1]).abs() * mask.unsqueeze(-1) * valid_depth_mask
        l1_loss_mask = (depth_diff < 1.0).float()
        img_loss = img_loss + (l1_loss_mask * depth_diff + (1 - l1_loss_mask) * depth_diff.pow(2)).mean() * 100.0
        reg_loss = torch.tensor(0.0)
        iter_thres = 0
        sdf_weight = self.FLAGS.sdf_regularizer - (self.FLAGS.sdf_regularizer - 0.01) * min(1.0, 4.0 * ((iteration - iter_thres) / (self.FLAGS.iter - iter_thres)))
        sdf_mask = torch.zeros_like(self.sdf, device=self.sdf.device)
        sdf_mask[imesh.valid_vert_idx] = 1.0
        sdf_masked = self.sdf.detach() * sdf_mask + self.sdf * (1 - sdf_mask)
        reg_loss = sdf_reg_loss(sdf_masked, self.all_edges).mean() * sdf_weight * 2.5
        reg_loss += torch.mean(buffers['kd_grad'][..., :-1] * buffers['kd_grad'][..., -1:]) * 0.03 * min(1.0, iteration / 500)
        reg_loss += torch.mean(buffers['occlusion'][..., :-1] * buffers['occlusion'][..., -1:]) * 1.0 * min(1.0, iteration / 500)
        pred_points = kaolin.ops.mesh.sample_points(imesh.v_pos.unsqueeze(0), imesh.t_pos_idx, 50000)[0][0]
        target_pts = target['spts']
        chamfer = kaolin.metrics.pointcloud.chamfer_distance(pred_points.unsqueeze(0), target_pts.unsqueeze(0)).mean()
        reg_loss += chamfer
        return img_loss, reg_loss


class DMTetGeometryFixedTopo(torch.nn.Module):

    def __init__(self, dmt_geometry, base_mesh, grid_res, scale, FLAGS, deform_scale=1.0, **kwargs):
        super(DMTetGeometryFixedTopo, self).__init__()
        self.FLAGS = FLAGS
        self.grid_res = grid_res
        self.marching_tets = DMTet()
        self.initial_guess = base_mesh
        self.scale = scale
        self.tanh = False
        self.deform_scale = deform_scale
        tets = np.load('./data/tets/{}_tets_cropped.npz'.format(self.grid_res))
        self.verts = torch.tensor(tets['vertices'], dtype=torch.float32, device='cuda') * scale
        self.indices = torch.tensor(tets['indices'], dtype=torch.long, device='cuda')
        self.generate_edges()
        self.sdf_sign = torch.nn.Parameter(torch.sign(dmt_geometry.sdf.data + 1e-08).float(), requires_grad=False)
        self.sdf_sign.data[self.sdf_sign.data == 0] = 1.0
        self.register_parameter('sdf_sign', self.sdf_sign)
        self.sdf_abs = torch.nn.Parameter(torch.ones_like(dmt_geometry.sdf), requires_grad=False)
        self.register_parameter('sdf_abs', self.sdf_abs)
        self.deform = torch.nn.Parameter(dmt_geometry.deform.data, requires_grad=True)
        self.register_parameter('deform', self.deform)
        self.sdf_abs_ema = torch.nn.Parameter(self.sdf_abs.clone().detach(), requires_grad=False)
        self.deform_ema = torch.nn.Parameter(self.deform.clone().detach(), requires_grad=False)

    def set_init_v_pos(self):
        with torch.no_grad():
            v_deformed = self.get_deformed()
            verts, faces, uvs, uv_idx = self.marching_tets(v_deformed, self.sdf_sign * self.sdf_abs.abs(), self.indices)
            self.initial_guess_v_pos = verts

    def generate_edges(self):
        with torch.no_grad():
            edges = torch.tensor([0, 1, 0, 2, 0, 3, 1, 2, 1, 3, 2, 3], dtype=torch.long, device='cuda')
            all_edges = self.indices[:, edges].reshape(-1, 2)
            all_edges_sorted = torch.sort(all_edges, dim=1)[0]
            self.all_edges = torch.unique(all_edges_sorted, dim=0)

    def getAABB(self):
        return torch.min(self.verts, dim=0).values, torch.max(self.verts, dim=0).values

    def getVertNNDist(self):
        raise NotImplementedError
        v_deformed = (self.verts + 2 / (self.grid_res * 2) * torch.tanh(self.deform)).unsqueeze(0)
        return pytorch3d.ops.knn.knn_points(v_deformed, v_deformed, K=2).dists[0, :, -1].detach()

    def getMesh(self, material):
        v_deformed = self.get_deformed()
        verts, faces, uvs, uv_idx = self.marching_tets(v_deformed, self.sdf_sign * self.sdf_abs.abs(), self.indices)
        imesh = mesh.Mesh(verts, faces, v_tex=uvs, t_tex_idx=uv_idx, material=material)
        imesh = mesh.auto_normals(imesh)
        imesh = mesh.compute_tangents(imesh)
        return imesh

    def getMesh_tet_gidx(self, material):
        v_deformed = self.get_deformed()
        verts, faces, uvs, uv_idx, tet_gidx = self.marching_tets(v_deformed, self.sdf_sign * self.sdf_abs.abs(), self.indices, get_tet_gidx=True)
        imesh = mesh.Mesh(verts, faces, v_tex=uvs, t_tex_idx=uv_idx, material=material)
        imesh = mesh.auto_normals(imesh)
        imesh = mesh.compute_tangents(imesh)
        return imesh, tet_gidx

    def update_ema(self, ema_coeff=0.9):
        return

    def get_deformed(self):
        if self.tanh:
            v_deformed = self.verts + 2 / (self.grid_res * 2) * torch.tanh(self.deform) * self.deform_scale
        else:
            v_deformed = self.verts + 2 / (self.grid_res * 2) * self.deform * self.deform_scale
        return v_deformed

    def getValidTetIdx(self):
        v_deformed = self.get_deformed()
        verts, faces, uvs, uv_idx, tet_gidx = self.marching_tets(v_deformed, self.sdf_sign * self.sdf_abs.abs(), self.indices, get_tet_gidx=True)
        return tet_gidx.long()

    def getValidVertsIdx(self):
        v_deformed = self.get_deformed()
        verts, faces, uvs, uv_idx, tet_gidx = self.marching_tets(v_deformed, self.sdf_sign * self.sdf_abs.abs(), self.indices, get_tet_gidx=True)
        return self.indices[tet_gidx.long()].unique()

    def getTetCenters(self):
        v_deformed = self.get_deformed()
        face_verts = v_deformed[self.indices]
        face_centers = face_verts.mean(dim=1)
        return face_centers

    def clamp_deform(self):
        if not self.tanh:
            self.deform.data[:] = self.deform.data.clamp(-0.99, 0.99)

    def render(self, glctx, target, lgt, opt_material, bsdf=None, ema=False, xfm_lgt=None, get_visible_tets=False):
        opt_mesh = self.getMesh(opt_material)
        tet_centers = self.getTetCenters() if get_visible_tets else None
        return render.render_mesh(glctx, opt_mesh, target['mvp'], target['campos'], lgt, target['resolution'], spp=target['spp'], msaa=True, background=target['background'], bsdf=bsdf, xfm_lgt=xfm_lgt, tet_centers=tet_centers)

    def render_with_mesh(self, glctx, target, lgt, opt_material, bsdf=None, xfm_lgt=None):
        opt_mesh = self.getMesh(opt_material)
        return opt_mesh, render.render_mesh(glctx, opt_mesh, target['mvp'], target['campos'], lgt, target['resolution'], spp=target['spp'], num_layers=self.FLAGS.layers, msaa=True, background=target['background'], bsdf=bsdf, xfm_lgt=xfm_lgt)

    def tick(self, glctx, target, lgt, opt_material, loss_fn, iteration, with_reg=True, xfm_lgt=None, no_depth_thin=True):
        imesh, buffers = self.render_with_mesh(glctx, target, lgt, opt_material, xfm_lgt=xfm_lgt)
        t_iter = iteration / self.FLAGS.iter
        color_ref = target['img']
        img_loss = torch.nn.functional.mse_loss(buffers['shaded'][..., 3:], color_ref[..., 3:])
        img_loss = img_loss + loss_fn(buffers['shaded'][..., 0:3] * color_ref[..., 3:], color_ref[..., 0:3] * color_ref[..., 3:])
        mask = target['mask'][:, :, :, 0]
        if no_depth_thin:
            valid_depth_mask = ((target['depth_second'] >= 0).float() * ((target['depth_second'] - target['depth']).abs() >= 0.005).float()).detach()
        else:
            valid_depth_mask = 1.0
        depth_diff = (buffers['depth'][:, :, :, :1] - target['depth'][:, :, :, :1]).abs() * mask.unsqueeze(-1) * valid_depth_mask
        depth_diff = (buffers['depth_second'][:, :, :, :1] - target['depth_second'][:, :, :, :1]).abs() * mask.unsqueeze(-1) * valid_depth_mask * 0.1
        l1_loss_mask = (depth_diff < 1.0).float()
        img_loss = img_loss + (l1_loss_mask * depth_diff + (1 - l1_loss_mask) * depth_diff.pow(2)).mean() * 100.0
        reg_loss = torch.tensor([0], dtype=torch.float32, device='cuda')
        reg_loss += regularizer.laplace_regularizer_const(imesh.v_pos - self.initial_guess_v_pos, imesh.t_pos_idx) * self.FLAGS.laplace_scale * (1 - t_iter) * 0.01
        pred_points = kaolin.ops.mesh.sample_points(imesh.v_pos.unsqueeze(0), imesh.t_pos_idx, 50000)[0][0]
        target_pts = target['spts']
        chamfer = kaolin.metrics.pointcloud.chamfer_distance(pred_points.unsqueeze(0), target_pts.unsqueeze(0)).mean()
        reg_loss += chamfer
        return img_loss, reg_loss


class cubemap_mip(torch.autograd.Function):

    @staticmethod
    def forward(ctx, cubemap):
        return util.avg_pool_nhwc(cubemap, (2, 2))

    @staticmethod
    def backward(ctx, dout):
        res = dout.shape[1] * 2
        out = torch.zeros(6, res, res, dout.shape[-1], dtype=torch.float32, device='cuda')
        for s in range(6):
            gy, gx = torch.meshgrid(torch.linspace(-1.0 + 1.0 / res, 1.0 - 1.0 / res, res, device='cuda'), torch.linspace(-1.0 + 1.0 / res, 1.0 - 1.0 / res, res, device='cuda'), indexing='ij')
            v = util.safe_normalize(util.cube_to_dir(s, gx, gy))
            out[s, ...] = dr.texture(dout[None, ...] * 0.25, v[None, ...].contiguous(), filter_mode='linear', boundary_mode='cube')
        return out


class EnvironmentLight(torch.nn.Module):
    LIGHT_MIN_RES = 16
    MIN_ROUGHNESS = 0.08
    MAX_ROUGHNESS = 0.5

    def __init__(self, base, trainable=True):
        super(EnvironmentLight, self).__init__()
        self.mtx = None
        self.base = torch.nn.Parameter(base.clone().detach(), requires_grad=trainable)
        None
        if trainable:
            self.register_parameter('env_base', self.base)

    def xfm(self, mtx):
        self.mtx = mtx

    def clone(self):
        return EnvironmentLight(self.base.clone().detach())

    def clamp_(self, min=None, max=None):
        self.base.clamp_(min, max)

    def get_mip(self, roughness):
        return torch.where(roughness < self.MAX_ROUGHNESS, (torch.clamp(roughness, self.MIN_ROUGHNESS, self.MAX_ROUGHNESS) - self.MIN_ROUGHNESS) / (self.MAX_ROUGHNESS - self.MIN_ROUGHNESS) * (len(self.specular) - 2), (torch.clamp(roughness, self.MAX_ROUGHNESS, 1.0) - self.MAX_ROUGHNESS) / (1.0 - self.MAX_ROUGHNESS) + len(self.specular) - 2)

    def build_mips(self, cutoff=0.99):
        self.specular = [self.base]
        while self.specular[-1].shape[1] > self.LIGHT_MIN_RES:
            self.specular += [cubemap_mip.apply(self.specular[-1])]
        self.diffuse = ru.diffuse_cubemap(self.specular[-1])
        for idx in range(len(self.specular) - 1):
            roughness = idx / (len(self.specular) - 2) * (self.MAX_ROUGHNESS - self.MIN_ROUGHNESS) + self.MIN_ROUGHNESS
            self.specular[idx] = ru.specular_cubemap(self.specular[idx], roughness, cutoff)
        self.specular[-1] = ru.specular_cubemap(self.specular[-1], 1.0, cutoff)

    def regularizer(self):
        white = (self.base[..., 0:1] + self.base[..., 1:2] + self.base[..., 2:3]) / 3.0
        return torch.mean(torch.abs(self.base - white))

    def shade(self, gb_pos, gb_normal, kd, ks, view_pos, specular=True, xfm_lgt=None):
        wo = util.safe_normalize(view_pos - gb_pos)
        if specular:
            roughness = ks[..., 1:2]
            metallic = ks[..., 2:3]
            spec_col = (1.0 - metallic) * 0.04 + kd * metallic
            diff_col = kd * (1.0 - metallic)
        else:
            diff_col = kd
        reflvec = util.safe_normalize(util.reflect(wo, gb_normal))
        nrmvec = gb_normal
        if xfm_lgt is not None:
            mtx = torch.as_tensor(xfm_lgt, dtype=torch.float32, device='cuda')
            reflvec = ru.xfm_vectors(reflvec.view(reflvec.shape[0], reflvec.shape[1] * reflvec.shape[2], reflvec.shape[3]), mtx).view(*reflvec.shape)
            nrmvec = ru.xfm_vectors(nrmvec.view(nrmvec.shape[0], nrmvec.shape[1] * nrmvec.shape[2], nrmvec.shape[3]), mtx).view(*nrmvec.shape)
        elif self.mtx is not None:
            raise NotImplementedError
            mtx = torch.as_tensor(self.mtx, dtype=torch.float32, device='cuda')
            reflvec = ru.xfm_vectors(reflvec.view(reflvec.shape[0], reflvec.shape[1] * reflvec.shape[2], reflvec.shape[3]), mtx).view(*reflvec.shape)
            nrmvec = ru.xfm_vectors(nrmvec.view(nrmvec.shape[0], nrmvec.shape[1] * nrmvec.shape[2], nrmvec.shape[3]), mtx).view(*nrmvec.shape)
        diffuse = dr.texture(self.diffuse[None, ...], nrmvec.contiguous(), filter_mode='linear', boundary_mode='cube')
        shaded_col = diffuse * diff_col
        if specular:
            raise NotImplementedError
            NdotV = torch.clamp(util.dot(wo, gb_normal), min=0.0001)
            fg_uv = torch.cat((NdotV, roughness), dim=-1)
            if not hasattr(self, '_FG_LUT'):
                self._FG_LUT = torch.as_tensor(np.fromfile('data/irrmaps/bsdf_256_256.bin', dtype=np.float32).reshape(1, 256, 256, 2), dtype=torch.float32, device='cuda')
            fg_lookup = dr.texture(self._FG_LUT, fg_uv, filter_mode='linear', boundary_mode='clamp')
            miplevel = self.get_mip(roughness)
            spec = dr.texture(self.specular[0][None, ...], reflvec.contiguous(), mip=list(m[None, ...] for m in self.specular[1:]), mip_level_bias=miplevel[..., 0], filter_mode='linear-mipmap-linear', boundary_mode='cube')
            reflectance = spec_col * fg_lookup[..., 0:1] + fg_lookup[..., 1:2]
            shaded_col += spec * reflectance
        assert ks[..., 0:1].sum().item() == 0
        return shaded_col * (1.0 - ks[..., 0:1])


class Material(torch.nn.Module):

    def __init__(self, mat_dict):
        super(Material, self).__init__()
        self.mat_keys = set()
        for key in mat_dict.keys():
            self.mat_keys.add(key)
            self[key] = mat_dict[key]

    def __contains__(self, key):
        return hasattr(self, key)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, val):
        self.mat_keys.add(key)
        setattr(self, key, val)

    def __delitem__(self, key):
        self.mat_keys.remove(key)
        delattr(self, key)

    def keys(self):
        return self.mat_keys


class _MLP(torch.nn.Module):

    def __init__(self, cfg, loss_scale=1.0):
        super(_MLP, self).__init__()
        self.loss_scale = loss_scale
        net = torch.nn.Linear(cfg['n_input_dims'], cfg['n_neurons'], bias=False), torch.nn.ReLU()
        for i in range(cfg['n_hidden_layers'] - 1):
            net = net + (torch.nn.Linear(cfg['n_neurons'], cfg['n_neurons'], bias=False), torch.nn.ReLU())
        net = net + (torch.nn.Linear(cfg['n_neurons'], cfg['n_output_dims'], bias=False),)
        self.net = torch.nn.Sequential(*net)
        self.net.apply(self._init_weights)
        if self.loss_scale != 1.0:
            self.net.register_full_backward_hook(lambda module, grad_i, grad_o: (grad_i[0] * self.loss_scale,))

    def forward(self, x):
        return self.net(x)

    @staticmethod
    def _init_weights(m):
        if type(m) == torch.nn.Linear:
            torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            if hasattr(m.bias, 'data'):
                m.bias.data.fill_(0.0)


class MLPTexture3D(torch.nn.Module):

    def __init__(self, AABB, channels=3, internal_dims=32, hidden=2, min_max=None):
        super(MLPTexture3D, self).__init__()
        self.channels = channels
        self.internal_dims = internal_dims
        self.AABB = AABB
        self.min_max = min_max
        desired_resolution = 4096
        base_grid_resolution = 16
        num_levels = 16
        per_level_scale = np.exp(np.log(desired_resolution / base_grid_resolution) / (num_levels - 1))
        enc_cfg = {'otype': 'HashGrid', 'n_levels': num_levels, 'n_features_per_level': 2, 'log2_hashmap_size': 19, 'base_resolution': base_grid_resolution, 'per_level_scale': per_level_scale}
        gradient_scaling = 128.0
        self.encoder = tcnn.Encoding(3, enc_cfg)
        self.encoder.register_full_backward_hook(lambda module, grad_i, grad_o: (grad_i[0] / gradient_scaling,))
        mlp_cfg = {'n_input_dims': self.encoder.n_output_dims, 'n_output_dims': self.channels, 'n_hidden_layers': hidden, 'n_neurons': self.internal_dims}
        self.net = _MLP(mlp_cfg, gradient_scaling)
        None

    def sample(self, texc):
        _texc = (texc.view(-1, 3) - self.AABB[0][None, ...]) / (self.AABB[1][None, ...] - self.AABB[0][None, ...])
        _texc = torch.clamp(_texc, min=0, max=1)
        p_enc = self.encoder(_texc.contiguous())
        out = self.net.forward(p_enc)
        if self.min_max is not None:
            out = torch.sigmoid(out) * (self.min_max[1][None, :] - self.min_max[0][None, :]) + self.min_max[0][None, :]
        return out.view(*texc.shape[:-1], self.channels)

    def clamp_(self):
        pass

    def cleanup(self):
        tcnn.free_temporary_memory()


class texture2d_mip(torch.autograd.Function):

    @staticmethod
    def forward(ctx, texture):
        return util.avg_pool_nhwc(texture, (2, 2))

    @staticmethod
    def backward(ctx, dout):
        gy, gx = torch.meshgrid(torch.linspace(0.0 + 0.25 / dout.shape[1], 1.0 - 0.25 / dout.shape[1], dout.shape[1] * 2, device='cuda'), torch.linspace(0.0 + 0.25 / dout.shape[2], 1.0 - 0.25 / dout.shape[2], dout.shape[2] * 2, device='cuda'), indexing='ij')
        uv = torch.stack((gx, gy), dim=-1)
        return dr.texture(dout * 0.25, uv[None, ...].contiguous(), filter_mode='linear', boundary_mode='clamp')


class Texture2D(torch.nn.Module):

    def __init__(self, init, min_max=None, trainable=True):
        super(Texture2D, self).__init__()
        if isinstance(init, np.ndarray):
            init = torch.tensor(init, dtype=torch.float32, device='cuda')
        elif isinstance(init, list) and len(init) == 1:
            init = init[0]
        if isinstance(init, list):
            self.data = list(torch.nn.Parameter(mip.clone().detach(), requires_grad=trainable) for mip in init)
        elif len(init.shape) == 4:
            self.data = torch.nn.Parameter(init.clone().detach(), requires_grad=trainable)
        elif len(init.shape) == 3:
            self.data = torch.nn.Parameter(init[None, ...].clone().detach(), requires_grad=trainable)
        elif len(init.shape) == 1:
            self.data = torch.nn.Parameter(init[None, None, None, :].clone().detach(), requires_grad=trainable)
        else:
            assert False, 'Invalid texture object'
        self.min_max = min_max

    def sample(self, texc, texc_deriv, filter_mode='linear-mipmap-linear'):
        if isinstance(self.data, list):
            out = dr.texture(self.data[0], texc, texc_deriv, mip=self.data[1:], filter_mode=filter_mode)
        elif self.data.shape[1] > 1 and self.data.shape[2] > 1:
            mips = [self.data]
            while mips[-1].shape[1] > 1 and mips[-1].shape[2] > 1:
                mips += [texture2d_mip.apply(mips[-1])]
            out = dr.texture(mips[0], texc, texc_deriv, mip=mips[1:], filter_mode=filter_mode)
        else:
            out = dr.texture(self.data, texc, texc_deriv, filter_mode=filter_mode)
        return out

    def getRes(self):
        return self.getMips()[0].shape[1:3]

    def getChannels(self):
        return self.getMips()[0].shape[3]

    def getMips(self):
        if isinstance(self.data, list):
            return self.data
        else:
            return [self.data]

    def clamp_(self):
        if self.min_max is not None:
            for mip in self.getMips():
                for i in range(mip.shape[-1]):
                    mip[..., i].clamp_(min=self.min_max[0][i], max=self.min_max[1][i])

    def normalize_(self):
        with torch.no_grad():
            for mip in self.getMips():
                mip = util.safe_normalize(mip)


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (Conv3d,
     lambda: ([], {'in_ch': 4, 'out_ch': 4, 'kernel': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ConvMeanPool,
     lambda: ([], {'input_dim': 4, 'output_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Downsample,
     lambda: ([], {'channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4, 4])], {})),
    (InstanceNorm3dPlus,
     lambda: ([], {'num_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4, 4])], {})),
    (MeanPoolConv,
     lambda: ([], {'input_dim': 4, 'output_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (NoneNorm3d,
     lambda: ([], {'num_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ResidualBlock,
     lambda: ([], {'input_dim': 4, 'output_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Upsample,
     lambda: ([], {'channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4, 4])], {})),
    (UpsampleConv,
     lambda: ([], {'input_dim': 4, 'output_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (VarianceNorm3d,
     lambda: ([], {'num_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4, 4])], {})),
    (_MLP,
     lambda: ([], {'cfg': SimpleNamespace(n_input_dims=4, n_neurons=4, n_hidden_layers=1, n_output_dims=4)}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
]

