
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


import re


import time


from typing import Dict


from typing import List


from typing import Callable


from typing import Union


from copy import deepcopy


from collections import OrderedDict


import torch


from typing import Optional


import torch.nn as nn


import torch.nn.functional as F


import numpy as np


from torch.nn.utils import spectral_norm


from torch import Tensor


from collections import defaultdict


import math


from typing import Tuple


import copy


from collections import namedtuple


import warnings


from copy import copy


import torchvision


from torch.hub import download_url_to_file as _torchhub_download_url_to_file


from torch.hub import get_dir


class LambdaLayer(nn.Module):

    def __init__(self, f):
        super(LambdaLayer, self).__init__()
        self.f = f

    def forward(self, x):
        return self.f(x)


class ScaledWSConv2d(nn.Conv2d):
    """2D Conv layer with Scaled Weight Standardization."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, gain=True, eps=0.0001):
        nn.Conv2d.__init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        if gain:
            self.gain = nn.Parameter(torch.ones(self.out_channels, 1, 1, 1))
        else:
            self.gain = None
        self.eps = eps

    def get_weight(self):
        fan_in = np.prod(self.weight.shape[1:])
        var, mean = torch.var_mean(self.weight, dim=(1, 2, 3), keepdims=True)
        scale = torch.rsqrt(torch.max(var * fan_in, torch.tensor(self.eps))) * self.gain.view_as(var)
        shift = mean * scale
        return self.weight * scale - shift

    def forward(self, x):
        return F.conv2d(x, self.get_weight(), self.bias, self.stride, self.padding, self.dilation, self.groups)


class ScaledWSTransposeConv2d(nn.ConvTranspose2d):
    """2D Transpose Conv layer with Scaled Weight Standardization."""

    def __init__(self, in_channels: 'int', out_channels: 'int', kernel_size, stride=1, padding=0, output_padding=0, groups: 'int'=1, bias: 'bool'=True, dilation: 'int'=1, gain=True, eps=0.0001):
        nn.ConvTranspose2d.__init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, groups, bias, dilation, 'zeros')
        if gain:
            self.gain = nn.Parameter(torch.ones(self.in_channels, 1, 1, 1))
        else:
            self.gain = None
        self.eps = eps

    def get_weight(self):
        fan_in = np.prod(self.weight.shape[1:])
        var, mean = torch.var_mean(self.weight, dim=(1, 2, 3), keepdims=True)
        scale = torch.rsqrt(torch.max(var * fan_in, torch.tensor(self.eps))) * self.gain.view_as(var)
        shift = mean * scale
        return self.weight * scale - shift

    def forward(self, x, output_size: 'Optional[List[int]]'=None):
        output_padding = self._output_padding(input, output_size, self.stride, self.padding, self.kernel_size, self.dilation)
        return F.conv_transpose2d(x, self.get_weight(), self.bias, self.stride, self.padding, output_padding, self.groups, self.dilation)


class GatedWSConvPadded(nn.Module):

    def __init__(self, in_ch, out_ch, ks, stride=1, dilation=1):
        super(GatedWSConvPadded, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.padding = nn.ReflectionPad2d((ks - 1) * dilation // 2)
        self.conv = ScaledWSConv2d(in_ch, out_ch, kernel_size=ks, stride=stride, dilation=dilation)
        self.conv_gate = ScaledWSConv2d(in_ch, out_ch, kernel_size=ks, stride=stride, dilation=dilation)

    def forward(self, x):
        x = self.padding(x)
        signal = self.conv(x)
        gate = torch.sigmoid(self.conv_gate(x))
        return signal * gate * 1.8


class GatedWSTransposeConvPadded(nn.Module):

    def __init__(self, in_ch, out_ch, ks, stride=1):
        super(GatedWSTransposeConvPadded, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.conv = ScaledWSTransposeConv2d(in_ch, out_ch, kernel_size=ks, stride=stride, padding=(ks - 1) // 2)
        self.conv_gate = ScaledWSTransposeConv2d(in_ch, out_ch, kernel_size=ks, stride=stride, padding=(ks - 1) // 2)

    def forward(self, x):
        signal = self.conv(x)
        gate = torch.sigmoid(self.conv_gate(x))
        return signal * gate * 1.8


def relu_nf(x):
    return F.relu(x) * 1.7139588594436646


class ResBlock(nn.Module):

    def __init__(self, ch, alpha=0.2, beta=1.0, dilation=1):
        super(ResBlock, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.c1 = GatedWSConvPadded(ch, ch, 3, dilation=dilation)
        self.c2 = GatedWSConvPadded(ch, ch, 3, dilation=dilation)

    def forward(self, x):
        skip = x
        x = self.c1(relu_nf(x / self.beta))
        x = self.c2(relu_nf(x))
        x = x * self.alpha
        return x + skip


def my_layer_norm(feat):
    mean = feat.mean((2, 3), keepdim=True)
    std = feat.std((2, 3), keepdim=True) + 1e-09
    feat = 2 * (feat - mean) / std - 1
    feat = 5 * feat
    return feat


class AOTBlock(nn.Module):

    def __init__(self, dim, rates=[2, 4, 8, 16]):
        super(AOTBlock, self).__init__()
        self.rates = rates
        for i, rate in enumerate(rates):
            self.__setattr__('block{}'.format(str(i).zfill(2)), nn.Sequential(nn.ReflectionPad2d(rate), nn.Conv2d(dim, dim // 4, 3, padding=0, dilation=rate), nn.ReLU(True)))
        self.fuse = nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(dim, dim, 3, padding=0, dilation=1))
        self.gate = nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(dim, dim, 3, padding=0, dilation=1))

    def forward(self, x):
        out = [self.__getattr__(f'block{str(i).zfill(2)}')(x) for i in range(len(self.rates))]
        out = torch.cat(out, 1)
        out = self.fuse(out)
        mask = my_layer_norm(self.gate(x))
        mask = torch.sigmoid(mask)
        return x * (1 - mask) + out * mask


class ResBlockDis(nn.Module):

    def __init__(self, in_planes, planes, stride=1):
        super(ResBlockDis, self).__init__()
        self.bn1 = nn.InstanceNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3 if stride == 1 else 4, stride=stride, padding=1)
        self.bn2 = nn.InstanceNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1)
        self.planes = planes
        self.in_planes = in_planes
        self.stride = stride
        self.shortcut = nn.Sequential()
        if stride > 1:
            self.shortcut = nn.Sequential(nn.AvgPool2d(2, 2), nn.Conv2d(in_planes, planes, kernel_size=1))
        elif in_planes != planes and stride == 1:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, planes, kernel_size=1))

    def forward(self, x):
        sc = self.shortcut(x)
        x = self.conv1(F.leaky_relu(self.bn1(x), 0.2))
        x = self.conv2(F.leaky_relu(self.bn2(x), 0.2))
        return sc + x


class Discriminator(nn.Module):

    def __init__(self, in_ch=3, in_planes=64, blocks=[2, 2, 2], alpha=0.2):
        super(Discriminator, self).__init__()
        self.in_planes = in_planes
        self.conv = nn.Sequential(spectral_norm(nn.Conv2d(in_ch, in_planes, 4, stride=2, padding=1, bias=False)), nn.LeakyReLU(0.2, inplace=True), spectral_norm(nn.Conv2d(in_planes, in_planes * 2, 4, stride=2, padding=1, bias=False)), nn.LeakyReLU(0.2, inplace=True), spectral_norm(nn.Conv2d(in_planes * 2, in_planes * 4, 4, stride=2, padding=1, bias=False)), nn.LeakyReLU(0.2, inplace=True), spectral_norm(nn.Conv2d(in_planes * 4, in_planes * 8, 4, stride=1, padding=1, bias=False)), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(512, 1, 4, stride=1, padding=1))

    def forward(self, x):
        x = self.conv(x)
        return x


class AOTGenerator(nn.Module):

    def __init__(self, in_ch=4, out_ch=3, ch=32, alpha=0.0):
        super(AOTGenerator, self).__init__()
        self.head = nn.Sequential(GatedWSConvPadded(in_ch, ch, 3, stride=1), LambdaLayer(relu_nf), GatedWSConvPadded(ch, ch * 2, 4, stride=2), LambdaLayer(relu_nf), GatedWSConvPadded(ch * 2, ch * 4, 4, stride=2))
        self.body_conv = nn.Sequential(*[AOTBlock(ch * 4) for _ in range(10)])
        self.tail = nn.Sequential(GatedWSConvPadded(ch * 4, ch * 4, 3, 1), LambdaLayer(relu_nf), GatedWSConvPadded(ch * 4, ch * 4, 3, 1), LambdaLayer(relu_nf), GatedWSTransposeConvPadded(ch * 4, ch * 2, 4, 2), LambdaLayer(relu_nf), GatedWSTransposeConvPadded(ch * 2, ch, 4, 2), LambdaLayer(relu_nf), GatedWSConvPadded(ch, out_ch, 3, stride=1))

    def forward(self, img, mask):
        x = torch.cat([mask, img], dim=1)
        x = self.head(x)
        conv = self.body_conv(x)
        x = self.tail(conv)
        if self.training:
            return x
        else:
            return torch.clip(x, -1, 1)


class FFCSE_block(nn.Module):

    def __init__(self, channels, ratio_g):
        super(FFCSE_block, self).__init__()
        in_cg = int(channels * ratio_g)
        in_cl = channels - in_cg
        r = 16
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv1 = nn.Conv2d(channels, channels // r, kernel_size=1, bias=True)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv_a2l = None if in_cl == 0 else nn.Conv2d(channels // r, in_cl, kernel_size=1, bias=True)
        self.conv_a2g = None if in_cg == 0 else nn.Conv2d(channels // r, in_cg, kernel_size=1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x if type(x) is tuple else (x, 0)
        id_l, id_g = x
        x = id_l if type(id_g) is int else torch.cat([id_l, id_g], dim=1)
        x = self.avgpool(x)
        x = self.relu1(self.conv1(x))
        x_l = 0 if self.conv_a2l is None else id_l * self.sigmoid(self.conv_a2l(x))
        x_g = 0 if self.conv_a2g is None else id_g * self.sigmoid(self.conv_a2g(x))
        return x_l, x_g


class FourierUnit(nn.Module):

    def __init__(self, in_channels, out_channels, groups=1, spatial_scale_factor=None, spatial_scale_mode='bilinear', spectral_pos_encoding=False, use_se=False, se_kwargs=None, ffc3d=False, fft_norm='ortho'):
        super(FourierUnit, self).__init__()
        self.groups = groups
        self.conv_layer = torch.nn.Conv2d(in_channels=in_channels * 2 + (2 if spectral_pos_encoding else 0), out_channels=out_channels * 2, kernel_size=1, stride=1, padding=0, groups=self.groups, bias=False)
        self.bn = torch.nn.BatchNorm2d(out_channels * 2)
        self.relu = torch.nn.ReLU(inplace=True)
        self.use_se = use_se
        self.spatial_scale_factor = spatial_scale_factor
        self.spatial_scale_mode = spatial_scale_mode
        self.spectral_pos_encoding = spectral_pos_encoding
        self.ffc3d = ffc3d
        self.fft_norm = fft_norm

    def forward(self, x):
        batch = x.shape[0]
        if self.spatial_scale_factor is not None:
            orig_size = x.shape[-2:]
            x = F.interpolate(x, scale_factor=self.spatial_scale_factor, mode=self.spatial_scale_mode, align_corners=False)
        r_size = x.size()
        fft_dim = (-3, -2, -1) if self.ffc3d else (-2, -1)
        if x.dtype in (torch.float16, torch.bfloat16):
            x = x.type(torch.float32)
        ffted = torch.fft.rfftn(x, dim=fft_dim, norm=self.fft_norm)
        ffted = torch.stack((ffted.real, ffted.imag), dim=-1)
        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()
        ffted = ffted.view((batch, -1) + ffted.size()[3:])
        if self.spectral_pos_encoding:
            height, width = ffted.shape[-2:]
            coords_vert = torch.linspace(0, 1, height)[None, None, :, None].expand(batch, 1, height, width)
            coords_hor = torch.linspace(0, 1, width)[None, None, None, :].expand(batch, 1, height, width)
            ffted = torch.cat((coords_vert, coords_hor, ffted), dim=1)
        if self.use_se:
            ffted = self.se(ffted)
        ffted = self.conv_layer(ffted)
        ffted = self.relu(self.bn(ffted))
        ffted = ffted.view((batch, -1, 2) + ffted.size()[2:]).permute(0, 1, 3, 4, 2).contiguous()
        if ffted.dtype in (torch.float16, torch.bfloat16):
            ffted = ffted.type(torch.float32)
        ffted = torch.complex(ffted[..., 0], ffted[..., 1])
        ifft_shape_slice = x.shape[-3:] if self.ffc3d else x.shape[-2:]
        output = torch.fft.irfftn(ffted, s=ifft_shape_slice, dim=fft_dim, norm=self.fft_norm)
        if self.spatial_scale_factor is not None:
            output = F.interpolate(output, size=orig_size, mode=self.spatial_scale_mode, align_corners=False)
        return output


class SpectralTransform(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, groups=1, enable_lfu=True, **fu_kwargs):
        super(SpectralTransform, self).__init__()
        self.enable_lfu = enable_lfu
        if stride == 2:
            self.downsample = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        else:
            self.downsample = nn.Identity()
        self.stride = stride
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, out_channels // 2, kernel_size=1, groups=groups, bias=False), nn.BatchNorm2d(out_channels // 2), nn.ReLU(inplace=True))
        self.fu = FourierUnit(out_channels // 2, out_channels // 2, groups, **fu_kwargs)
        if self.enable_lfu:
            self.lfu = FourierUnit(out_channels // 2, out_channels // 2, groups)
        self.conv2 = torch.nn.Conv2d(out_channels // 2, out_channels, kernel_size=1, groups=groups, bias=False)

    def forward(self, x):
        x = self.downsample(x)
        x = self.conv1(x)
        output = self.fu(x)
        if self.enable_lfu:
            n, c, h, w = x.shape
            split_no = 2
            split_s = h // split_no
            xs = torch.cat(torch.split(x[:, :c // 4], split_s, dim=-2), dim=1).contiguous()
            xs = torch.cat(torch.split(xs, split_s, dim=-1), dim=1).contiguous()
            xs = self.lfu(xs)
            xs = xs.repeat(1, 1, split_no, split_no).contiguous()
        else:
            xs = 0
        output = self.conv2(x + output + xs)
        return output


class FFC(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, ratio_gin, ratio_gout, stride=1, padding=0, dilation=1, groups=1, bias=False, enable_lfu=True, padding_type='reflect', gated=False, **spectral_kwargs):
        super(FFC, self).__init__()
        assert stride == 1 or stride == 2, 'Stride should be 1 or 2.'
        self.stride = stride
        in_cg = int(in_channels * ratio_gin)
        in_cl = in_channels - in_cg
        out_cg = int(out_channels * ratio_gout)
        out_cl = out_channels - out_cg
        self.ratio_gin = ratio_gin
        self.ratio_gout = ratio_gout
        self.global_in_num = in_cg
        module = nn.Identity if in_cl == 0 or out_cl == 0 else nn.Conv2d
        self.convl2l = module(in_cl, out_cl, kernel_size, stride, padding, dilation, groups, bias, padding_mode=padding_type)
        module = nn.Identity if in_cl == 0 or out_cg == 0 else nn.Conv2d
        self.convl2g = module(in_cl, out_cg, kernel_size, stride, padding, dilation, groups, bias, padding_mode=padding_type)
        module = nn.Identity if in_cg == 0 or out_cl == 0 else nn.Conv2d
        self.convg2l = module(in_cg, out_cl, kernel_size, stride, padding, dilation, groups, bias, padding_mode=padding_type)
        module = nn.Identity if in_cg == 0 or out_cg == 0 else SpectralTransform
        self.convg2g = module(in_cg, out_cg, stride, 1 if groups == 1 else groups // 2, enable_lfu, **spectral_kwargs)
        self.gated = gated
        module = nn.Identity if in_cg == 0 or out_cl == 0 or not self.gated else nn.Conv2d
        self.gate = module(in_channels, 2, 1)

    def forward(self, x):
        x_l, x_g = x if type(x) is tuple else (x, 0)
        out_xl, out_xg = 0, 0
        if self.gated:
            total_input_parts = [x_l]
            if torch.is_tensor(x_g):
                total_input_parts.append(x_g)
            total_input = torch.cat(total_input_parts, dim=1)
            gates = torch.sigmoid(self.gate(total_input))
            g2l_gate, l2g_gate = gates.chunk(2, dim=1)
        else:
            g2l_gate, l2g_gate = 1, 1
        if self.ratio_gout != 1:
            out_xl = self.convl2l(x_l) + self.convg2l(x_g) * g2l_gate
        if self.ratio_gout != 0:
            out_xg = self.convl2g(x_l) * l2g_gate + self.convg2g(x_g)
        return out_xl, out_xg


class FFC_BN_ACT(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, ratio_gin, ratio_gout, stride=1, padding=0, dilation=1, groups=1, bias=False, norm_layer=nn.BatchNorm2d, activation_layer=nn.Identity, padding_type='reflect', enable_lfu=True, **kwargs):
        super(FFC_BN_ACT, self).__init__()
        self.ffc = FFC(in_channels, out_channels, kernel_size, ratio_gin, ratio_gout, stride, padding, dilation, groups, bias, enable_lfu, padding_type=padding_type, **kwargs)
        lnorm = nn.Identity if ratio_gout == 1 else norm_layer
        gnorm = nn.Identity if ratio_gout == 0 else norm_layer
        global_channels = int(out_channels * ratio_gout)
        self.bn_l = lnorm(out_channels - global_channels)
        self.bn_g = gnorm(global_channels)
        lact = nn.Identity if ratio_gout == 1 else activation_layer
        gact = nn.Identity if ratio_gout == 0 else activation_layer
        self.act_l = lact(inplace=True)
        self.act_g = gact(inplace=True)

    def forward(self, x):
        x_l, x_g = self.ffc(x)
        x_l = self.act_l(self.bn_l(x_l))
        x_g = self.act_g(self.bn_g(x_g))
        return x_l, x_g


class FFCResnetBlock(nn.Module):

    def __init__(self, dim, padding_type, norm_layer, activation_layer=nn.ReLU, dilation=1, inline=False, **conv_kwargs):
        super().__init__()
        self.conv1 = FFC_BN_ACT(dim, dim, kernel_size=3, padding=dilation, dilation=dilation, norm_layer=norm_layer, activation_layer=activation_layer, padding_type=padding_type, **conv_kwargs)
        self.conv2 = FFC_BN_ACT(dim, dim, kernel_size=3, padding=dilation, dilation=dilation, norm_layer=norm_layer, activation_layer=activation_layer, padding_type=padding_type, **conv_kwargs)
        self.inline = inline

    def forward(self, x):
        if self.inline:
            x_l, x_g = x[:, :-self.conv1.ffc.global_in_num], x[:, -self.conv1.ffc.global_in_num:]
        else:
            x_l, x_g = x if type(x) is tuple else (x, 0)
        id_l, id_g = x_l, x_g
        x_l, x_g = self.conv1((x_l, x_g))
        x_l, x_g = self.conv2((x_l, x_g))
        x_l, x_g = id_l + x_l, id_g + x_g
        out = x_l, x_g
        if self.inline:
            out = torch.cat(out, dim=1)
        return out


class ConcatTupleLayer(nn.Module):

    def forward(self, x):
        assert isinstance(x, tuple)
        x_l, x_g = x
        assert torch.is_tensor(x_l) or torch.is_tensor(x_g)
        if not torch.is_tensor(x_g):
            return x_l
        return torch.cat(x, dim=1)


def get_activation(kind='tanh'):
    if kind == 'tanh':
        return nn.Tanh()
    if kind == 'sigmoid':
        return nn.Sigmoid()
    if kind is False:
        return nn.Identity()
    raise ValueError(f'Unknown activation kind {kind}')


class FFCResNetGenerator(nn.Module):

    def __init__(self, input_nc=4, output_nc=3, ngf=64, n_downsampling=3, n_blocks=9, norm_layer=nn.BatchNorm2d, padding_type='reflect', activation_layer=nn.ReLU, up_norm_layer=nn.BatchNorm2d, up_activation=nn.ReLU(True), init_conv_kwargs={}, downsample_conv_kwargs={}, resnet_conv_kwargs={}, spatial_transform_kwargs={}, add_out_act=True, max_features=1024, out_ffc=False, out_ffc_kwargs={}):
        assert n_blocks >= 0
        super().__init__()
        model = [nn.ReflectionPad2d(3), FFC_BN_ACT(input_nc, ngf, kernel_size=7, padding=0, norm_layer=norm_layer, activation_layer=activation_layer, **init_conv_kwargs)]
        for i in range(n_downsampling):
            mult = 2 ** i
            if i == n_downsampling - 1:
                cur_conv_kwargs = dict(downsample_conv_kwargs)
                cur_conv_kwargs['ratio_gout'] = resnet_conv_kwargs.get('ratio_gin', 0)
            else:
                cur_conv_kwargs = downsample_conv_kwargs
            model += [FFC_BN_ACT(min(max_features, ngf * mult), min(max_features, ngf * mult * 2), kernel_size=3, stride=2, padding=1, norm_layer=norm_layer, activation_layer=activation_layer, **cur_conv_kwargs)]
        mult = 2 ** n_downsampling
        feats_num_bottleneck = min(max_features, ngf * mult)
        for i in range(n_blocks):
            cur_resblock = FFCResnetBlock(feats_num_bottleneck, padding_type=padding_type, activation_layer=activation_layer, norm_layer=norm_layer, **resnet_conv_kwargs)
            model += [cur_resblock]
        model += [ConcatTupleLayer()]
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(min(max_features, ngf * mult), min(max_features, int(ngf * mult / 2)), kernel_size=3, stride=2, padding=1, output_padding=1), up_norm_layer(min(max_features, int(ngf * mult / 2))), up_activation]
        if out_ffc:
            model += [FFCResnetBlock(ngf, padding_type=padding_type, activation_layer=activation_layer, norm_layer=norm_layer, inline=True, **out_ffc_kwargs)]
        model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        if add_out_act:
            model.append(get_activation('tanh' if add_out_act is True else add_out_act))
        self.model = nn.Sequential(*model)

    def forward(self, img, mask, rel_pos=None, direct=None) ->Tensor:
        masked_img = torch.cat([img * (1 - mask), mask], dim=1)
        if rel_pos is None:
            return self.model(masked_img)
        else:
            x_l, x_g = self.model[:2](masked_img)
            x_l = x_l
            x_l += rel_pos
            x_l += direct
            return self.model[2:]((x_l, x_g))


class NLayerDiscriminator(nn.Module):

    def __init__(self, input_nc=3, ndf=64, n_layers=4, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.n_layers = n_layers
        kw = 4
        padw = int(np.ceil((kw - 1.0) / 2))
        sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]]
        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            cur_model = []
            cur_model += [nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw), norm_layer(nf), nn.LeakyReLU(0.2, True)]
            sequence.append(cur_model)
        nf_prev = nf
        nf = min(nf * 2, 512)
        cur_model = []
        cur_model += [nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw), norm_layer(nf), nn.LeakyReLU(0.2, True)]
        sequence.append(cur_model)
        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]
        for n in range(len(sequence)):
            setattr(self, 'model' + str(n), nn.Sequential(*sequence[n]))

    def get_all_activations(self, x):
        res = [x]
        for n in range(self.n_layers + 2):
            model = getattr(self, 'model' + str(n))
            res.append(model(res[-1]))
        return res[1:]

    def forward(self, x):
        act = self.get_all_activations(x)
        return act[-1], act[:-1]


class MaskedSinusoidalPositionalEmbedding(nn.Embedding):
    """This module produces sinusoidal positional embeddings of any length."""

    def __init__(self, num_embeddings: 'int', embedding_dim: 'int'):
        super().__init__(num_embeddings, embedding_dim)
        self.weight = self._init_weight(self.weight)

    @staticmethod
    def _init_weight(out: 'nn.Parameter'):
        """
        Identical to the XLM create_sinusoidal_embeddings except features are not interleaved. The cos features are in
        the 2nd half of the vector. [dim // 2:]
        """
        n_pos, dim = out.shape
        position_enc = np.array([[(pos / np.power(10000, 2 * (j // 2) / dim)) for j in range(dim)] for pos in range(n_pos)])
        out.requires_grad = False
        sentinel = dim // 2 if dim % 2 == 0 else dim // 2 + 1
        out[:, 0:sentinel] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))
        out[:, sentinel:] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))
        out.detach_()
        return out

    @torch.no_grad()
    def forward(self, input_ids):
        """`input_ids` is expected to be [bsz x seqlen]."""
        return super().forward(input_ids)


class MultiLabelEmbedding(nn.Module):

    def __init__(self, num_positions: 'int', embedding_dim: 'int'):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(num_positions, embedding_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.weight)

    def forward(self, input_ids):
        out = torch.matmul(input_ids, self.weight)
        return out


class MPE(nn.Module):

    def __init__(self):
        super().__init__()
        self.rel_pos_emb = MaskedSinusoidalPositionalEmbedding(num_embeddings=128, embedding_dim=64)
        self.direct_emb = MultiLabelEmbedding(num_positions=4, embedding_dim=64)
        self.alpha5 = nn.Parameter(torch.tensor(0, dtype=torch.float32), requires_grad=True)
        self.alpha6 = nn.Parameter(torch.tensor(0, dtype=torch.float32), requires_grad=True)

    def forward(self, rel_pos=None, direct=None):
        b, h, w = rel_pos.shape
        rel_pos = rel_pos.reshape(b, h * w)
        rel_pos_emb = self.rel_pos_emb(rel_pos).reshape(b, h, w, -1).permute(0, 3, 1, 2) * self.alpha5
        direct = direct.reshape(b, h * w, 4)
        direct_emb = self.direct_emb(direct).reshape(b, h, w, -1).permute(0, 3, 1, 2) * self.alpha6
        return rel_pos_emb, direct_emb


class ResNet(nn.Module):

    def __init__(self, input_channel, output_channel, block, layers):
        super(ResNet, self).__init__()
        self.output_channel_block = [int(output_channel / 4), int(output_channel / 2), output_channel, output_channel]
        self.inplanes = int(output_channel / 8)
        self.conv0_1 = nn.Conv2d(input_channel, int(output_channel / 8), kernel_size=3, stride=1, padding=1, bias=False)
        self.bn0_1 = nn.BatchNorm2d(int(output_channel / 8))
        self.conv0_2 = nn.Conv2d(int(output_channel / 8), self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.maxpool1 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.layer1 = self._make_layer(block, self.output_channel_block[0], layers[0])
        self.bn1 = nn.BatchNorm2d(self.output_channel_block[0])
        self.conv1 = nn.Conv2d(self.output_channel_block[0], self.output_channel_block[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.maxpool2 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.layer2 = self._make_layer(block, self.output_channel_block[1], layers[1], stride=1)
        self.bn2 = nn.BatchNorm2d(self.output_channel_block[1])
        self.conv2 = nn.Conv2d(self.output_channel_block[1], self.output_channel_block[1], kernel_size=3, stride=1, padding=1, bias=False)
        self.maxpool3 = nn.AvgPool2d(kernel_size=2, stride=(2, 1), padding=(0, 1))
        self.layer3 = self._make_layer(block, self.output_channel_block[2], layers[2], stride=1)
        self.bn3 = nn.BatchNorm2d(self.output_channel_block[2])
        self.conv3 = nn.Conv2d(self.output_channel_block[2], self.output_channel_block[2], kernel_size=3, stride=1, padding=1, bias=False)
        self.layer4 = self._make_layer(block, self.output_channel_block[3], layers[3], stride=1)
        self.bn4_1 = nn.BatchNorm2d(self.output_channel_block[3])
        self.conv4_1 = nn.Conv2d(self.output_channel_block[3], self.output_channel_block[3], kernel_size=3, stride=(2, 1), padding=(1, 1), bias=False)
        self.bn4_2 = nn.BatchNorm2d(self.output_channel_block[3])
        self.conv4_2 = nn.Conv2d(self.output_channel_block[3], self.output_channel_block[3], kernel_size=3, stride=1, padding=0, bias=False)
        self.bn4_3 = nn.BatchNorm2d(self.output_channel_block[3])

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.BatchNorm2d(self.inplanes), nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv0_1(x)
        x = self.bn0_1(x)
        x = F.relu(x)
        x = self.conv0_2(x)
        x = self.maxpool1(x)
        x = self.layer1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv1(x)
        x = self.maxpool2(x)
        x = self.layer2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.maxpool3(x)
        x = self.layer3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = self.layer4(x)
        x = self.bn4_1(x)
        x = F.relu(x)
        x = self.conv4_1(x)
        x = self.bn4_2(x)
        x = F.relu(x)
        x = self.conv4_2(x)
        x = self.bn4_3(x)
        return x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = self._conv3x3(inplanes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = self._conv3x3(planes, planes)
        self.downsample = downsample
        self.stride = stride

    def _conv3x3(self, in_planes, out_planes, stride=1):
        """3x3 convolution with padding"""
        return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

    def forward(self, x):
        residual = x
        out = self.bn1(x)
        out = F.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = F.relu(out)
        out = self.conv2(out)
        if self.downsample is not None:
            residual = self.downsample(residual)
        return out + residual


class ResNet_FeatureExtractor(nn.Module):
    """ FeatureExtractor of FAN (http://openaccess.thecvf.com/content_ICCV_2017/papers/Cheng_Focusing_Attention_Towards_ICCV_2017_paper.pdf) """

    def __init__(self, input_channel, output_channel=128):
        super(ResNet_FeatureExtractor, self).__init__()
        self.ConvNet = ResNet(input_channel, output_channel, BasicBlock, [4, 6, 8, 6, 3])

    def forward(self, input):
        return self.ConvNet(input)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x, offset=0):
        x = x + self.pe[:, offset:offset + x.size(1), :]
        return x


class AddCoords(nn.Module):

    def __init__(self, with_r=False):
        super().__init__()
        self.with_r = with_r

    def forward(self, input_tensor):
        """
        Args:
            input_tensor: shape(batch, channel, x_dim, y_dim)
        """
        batch_size, _, x_dim, y_dim = input_tensor.size()
        xx_channel = torch.arange(x_dim).repeat(1, y_dim, 1)
        yy_channel = torch.arange(y_dim).repeat(1, x_dim, 1).transpose(1, 2)
        xx_channel = xx_channel.float() / (x_dim - 1)
        yy_channel = yy_channel.float() / (y_dim - 1)
        xx_channel = xx_channel * 2 - 1
        yy_channel = yy_channel * 2 - 1
        xx_channel = xx_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)
        yy_channel = yy_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)
        ret = torch.cat([input_tensor, xx_channel.type_as(input_tensor), yy_channel.type_as(input_tensor)], dim=1)
        if self.with_r:
            rr = torch.sqrt(torch.pow(xx_channel.type_as(input_tensor) - 0.5, 2) + torch.pow(yy_channel.type_as(input_tensor) - 0.5, 2))
            ret = torch.cat([ret, rr], dim=1)
        return ret


class CustomTransformerEncoderLayer(nn.Module):
    """TransformerEncoderLayer is made up of self-attn and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.
    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).
        layer_norm_eps: the eps value in layer normalization components (default=1e-5).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False``.
        norm_first: if ``True``, layer norm is done prior to attention and feedforward
            operations, respectivaly. Otherwise it's done after. Default: ``False`` (after).
    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)
    Alternatively, when ``batch_first`` is ``True``:
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True)
        >>> src = torch.rand(32, 10, 512)
        >>> out = encoder_layer(src)
    """
    __constants__ = ['batch_first', 'norm_first']

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation='gelu', layer_norm_eps=1e-05, batch_first=False, norm_first=False, device=None, dtype=None) ->None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(CustomTransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first, **factory_kwargs)
        self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)
        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.pe = PositionalEncoding(d_model, max_len=2048)
        self.activation = F.gelu

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(CustomTransformerEncoderLayer, self).__setstate__(state)

    def forward(self, src: 'torch.Tensor', src_mask: 'Optional[torch.Tensor]'=None, src_key_padding_mask: 'Optional[torch.Tensor]'=None, is_causal=None) ->torch.Tensor:
        """Pass the input through the encoder layer.
        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
        Shape:
            see the docs in Transformer class.
        """
        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask))
            x = self.norm2(x + self._ff_block(x))
        return x

    def _sa_block(self, x: 'torch.Tensor', attn_mask: 'Optional[torch.Tensor]', key_padding_mask: 'Optional[torch.Tensor]') ->torch.Tensor:
        x = self.self_attn(self.pe(x), self.pe(x), x, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=False)[0]
        return self.dropout1(x)

    def _ff_block(self, x: 'torch.Tensor') ->torch.Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


class OCR(nn.Module):

    def __init__(self, dictionary, max_len):
        super(OCR, self).__init__()
        self.max_len = max_len
        self.dictionary = dictionary
        self.dict_size = len(dictionary)
        self.backbone = ResNet_FeatureExtractor(3, 320)
        enc = CustomTransformerEncoderLayer(320, 8, 320 * 4, dropout=0.05, batch_first=True, norm_first=True)
        self.encoders = nn.TransformerEncoder(enc, 3)
        self.char_pred_norm = nn.Sequential(nn.LayerNorm(320), nn.Dropout(0.1), nn.GELU())
        self.char_pred = nn.Linear(320, self.dict_size)
        self.color_pred1 = nn.Sequential(nn.Linear(320, 6))

    def forward(self, img: 'torch.FloatTensor'):
        feats = self.backbone(img).squeeze(2)
        feats = self.encoders(feats.permute(0, 2, 1))
        pred_char_logits = self.char_pred(self.char_pred_norm(feats))
        pred_color_values = self.color_pred1(feats)
        return pred_char_logits, pred_color_values

    def decode(self, img: 'torch.Tensor', img_widths: 'List[int]', blank, verbose=False) ->List[List[Tuple[str, float, int, int, int, int, int, int]]]:
        N, C, H, W = img.shape
        assert H == 48 and C == 3
        feats = self.backbone(img).squeeze(2)
        feats = self.encoders(feats.permute(0, 2, 1))
        pred_char_logits = self.char_pred(self.char_pred_norm(feats))
        pred_color_values = self.color_pred1(feats)
        return self.decode_ctc_top1(pred_char_logits, pred_color_values, blank, verbose=verbose)

    def decode_ctc_top1(self, pred_char_logits, pred_color_values, blank, verbose=False) ->List[List[Tuple[str, float, int, int, int, int, int, int]]]:
        pred_chars: 'List[List[Tuple[str, float, int, int, int, int, int, int]]]' = []
        for _ in range(pred_char_logits.size(0)):
            pred_chars.append([])
        logprobs = pred_char_logits.log_softmax(2)
        _, preds_index = logprobs.max(2)
        preds_index = preds_index.cpu()
        pred_color_values = pred_color_values.cpu().clamp_(0, 1)
        for b in range(pred_char_logits.size(0)):
            if verbose:
                None
            last_ch = blank
            for t in range(pred_char_logits.size(1)):
                pred_ch = preds_index[b, t]
                if pred_ch != last_ch and pred_ch != blank:
                    lp = logprobs[b, t, pred_ch].item()
                    if verbose:
                        if lp < math.log(0.9):
                            top5 = torch.topk(logprobs[b, t], 5)
                            top5_idx = top5.indices
                            top5_val = top5.values
                            r = ''
                            for i in range(5):
                                r += f'{self.dictionary[top5_idx[i]]}: {math.exp(top5_val[i])}, '
                            None
                        else:
                            None
                    pred_chars[b].append((pred_ch, lp, pred_color_values[b, t][0].item(), pred_color_values[b, t][1].item(), pred_color_values[b, t][2].item(), pred_color_values[b, t][3].item(), pred_color_values[b, t][4].item(), pred_color_values[b, t][5].item()))
                last_ch = pred_ch
        return pred_chars

    def eval_ocr(self, input_lengths, target_lengths, pred_char_logits, pred_color_values, gt_char_index, gt_color_values, blank, blank1):
        correct_char = 0
        total_char = 0
        color_diff = 0
        color_diff_dom = 0
        _, preds_index = pred_char_logits.max(2)
        pred_chars = torch.zeros_like(gt_char_index).cpu()
        for b in range(pred_char_logits.size(0)):
            last_ch = blank
            i = 0
            for t in range(input_lengths[b]):
                pred_ch = preds_index[b, t]
                if pred_ch != last_ch and pred_ch != blank:
                    total_char += 1
                    if gt_char_index[b, i] == pred_ch:
                        correct_char += 1
                        if pred_ch != blank1:
                            color_diff += ((pred_color_values[b, t] - gt_color_values[b, i]).abs().mean() * 255.0).item()
                            color_diff_dom += 1
                    pred_chars[b, i] = pred_ch
                    i += 1
                    if i >= gt_color_values.size(1) or i >= gt_char_index.size(1):
                        break
                last_ch = pred_ch
        return correct_char / (total_char + 1), color_diff / (color_diff_dom + 1), pred_chars


def duplicate_interleave(m):
    """
    A simple version of `torch.repeat_interleave` for duplicating a matrix while interleaving the copy.
    """
    dim0 = m.shape[0]
    m = m.view(-1, 1)
    m = m.repeat(1, 2)
    m = m.view(dim0, -1)
    return m


def rotate_every_two(x):
    x1 = x[:, :, ::2]
    x2 = x[:, :, 1::2]
    x = torch.stack((-x2, x1), dim=-1)
    return x.flatten(-2)


def apply_rotary_pos_emb(x, sin, cos, scale=1):
    sin, cos = map(lambda t: duplicate_interleave(t * scale), (sin, cos))
    return x * cos + rotate_every_two(x) * sin


def fixed_pos_embedding(x):
    seq_len, dim = x.shape
    inv_freq = 1.0 / 10000 ** (torch.arange(0, dim) / dim)
    sinusoid_inp = torch.einsum('i , j -> i j', torch.arange(0, seq_len, dtype=torch.float), inv_freq)
    return torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)


class XPOS(nn.Module):

    def __init__(self, head_dim, scale_base=512):
        super().__init__()
        self.head_dim = head_dim
        self.scale_base = scale_base
        self.register_buffer('scale', (torch.arange(0, head_dim, 2) + 0.4 * head_dim) / (1.4 * head_dim))

    def forward(self, x, offset=0, downscale=False):
        length = x.shape[1]
        min_pos = -(length + offset) // 2
        max_pos = length + offset + min_pos
        scale = self.scale ** torch.arange(min_pos, max_pos, 1).div(self.scale_base)[:, None]
        sin, cos = fixed_pos_embedding(scale)
        if scale.shape[0] > length:
            scale = scale[-length:]
            sin = sin[-length:]
            cos = cos[-length:]
        if downscale:
            scale = 1 / scale
        x = apply_rotary_pos_emb(x, sin, cos, scale)
        return x


class XPOS2D(nn.Module):

    def __init__(self, head_dim, scale_base=512):
        super().__init__()
        self.xpos = XPOS(head_dim // 2, scale_base)

    def forward(self, x: 'torch.Tensor', offset_x=0, offset_y=0, downscale=False):
        """
            x: N, H, W, C
        """
        N, H, W, C = x.shape
        C = C // 2
        [dir_x, dir_y] = x.chunk(2, dim=3)
        dir_x = einops.rearrange(dir_x, 'N H W C -> (N H) W C', N=N, H=H, W=W, C=C)
        dir_y = einops.rearrange(dir_y, 'N H W C -> (N W) H C', N=N, H=H, W=W, C=C)
        dir_x = self.xpos(dir_x, offset=offset_x, downscale=downscale)
        dir_y = self.xpos(dir_y, offset=offset_y, downscale=downscale)
        dir_x = einops.rearrange(dir_x, '(N H) W C -> N H W C', N=N, H=H, W=W, C=C)
        dir_y = einops.rearrange(dir_y, '(N W) H C -> N H W C', N=N, H=H, W=W, C=C)
        return torch.cat([dir_x, dir_y], dim=3)


class ConvNeXtBlock(nn.Module):
    """ ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, dim, layer_scale_init_value=1e-06, ks=7, padding=3):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=ks, padding=padding, groups=dim)
        self.norm = nn.BatchNorm2d(dim, eps=1e-06)
        self.pwconv1 = nn.Conv2d(dim, 4 * dim, 1, 1, 0)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(4 * dim, dim, 1, 1, 0)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(1, dim, 1, 1), requires_grad=True) if layer_scale_init_value > 0 else None

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = input + x
        return x


class ConvNext_FeatureExtractor(nn.Module):

    def __init__(self, img_height=48, in_dim=3, dim=512, n_layers=12) ->None:
        super().__init__()
        base = dim // 8
        self.stem = nn.Sequential(nn.Conv2d(in_dim, base, kernel_size=7, stride=1, padding=3), nn.BatchNorm2d(base), nn.ReLU(), nn.Conv2d(base, base * 2, kernel_size=2, stride=2, padding=0), nn.BatchNorm2d(base * 2), nn.ReLU(), nn.Conv2d(base * 2, base * 2, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(base * 2), nn.ReLU())
        self.block1 = self.make_layers(base * 2, 4)
        self.down1 = nn.Sequential(nn.Conv2d(base * 2, base * 4, kernel_size=2, stride=2, padding=0), nn.BatchNorm2d(base * 4), nn.ReLU())
        self.block2 = self.make_layers(base * 4, 12)
        self.down2 = nn.Sequential(nn.Conv2d(base * 4, base * 8, kernel_size=(2, 1), stride=(2, 1), padding=(0, 0)), nn.BatchNorm2d(base * 8), nn.ReLU())
        self.block3 = self.make_layers(base * 8, 10, ks=5, padding=2)
        self.down3 = nn.Sequential(nn.Conv2d(base * 8, base * 8, kernel_size=(2, 1), stride=(2, 1), padding=(0, 0)), nn.BatchNorm2d(base * 8), nn.ReLU())
        self.block4 = self.make_layers(base * 8, 8, ks=3, padding=1)
        self.down4 = nn.Sequential(nn.Conv2d(base * 8, base * 8, kernel_size=(3, 1), stride=(1, 1), padding=(0, 0)), nn.BatchNorm2d(base * 8), nn.ReLU())

    def make_layers(self, dim, n, ks=7, padding=3):
        layers = []
        for i in range(n):
            layers.append(ConvNeXtBlock(dim, ks=ks, padding=padding))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.down1(x)
        x = self.block2(x)
        x = self.down2(x)
        x = self.block3(x)
        x = self.down3(x)
        x = self.block4(x)
        x = self.down4(x)
        return x


class XposMultiheadAttention(nn.Module):

    def __init__(self, embed_dim, num_heads, self_attention=False, encoder_decoder_attention=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim ** -0.5
        self.self_attention = self_attention
        self.encoder_decoder_attention = encoder_decoder_attention
        assert self.self_attention ^ self.encoder_decoder_attention
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.xpos = XPOS(self.head_dim, embed_dim)
        self.batch_first = True
        self._qkv_same_embed_dim = True

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(self, query, key, value, key_padding_mask=None, attn_mask=None, need_weights=False, is_causal=False, k_offset=0, q_offset=0):
        assert not is_causal
        bsz, tgt_len, embed_dim = query.size()
        src_len = tgt_len
        assert embed_dim == self.embed_dim, f'query dim {embed_dim} != {self.embed_dim}'
        key_bsz, src_len, _ = key.size()
        assert key_bsz == bsz, f'{query.size(), key.size()}'
        assert value is not None
        assert bsz, src_len == value.shape[:2]
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        q *= self.scaling
        q = q.view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, src_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, src_len, self.num_heads, self.head_dim).transpose(1, 2)
        q = q.reshape(bsz * self.num_heads, tgt_len, self.head_dim)
        k = k.reshape(bsz * self.num_heads, src_len, self.head_dim)
        v = v.reshape(bsz * self.num_heads, src_len, self.head_dim)
        if self.xpos is not None:
            k = self.xpos(k, offset=k_offset, downscale=True)
            q = self.xpos(q, offset=q_offset, downscale=False)
        attn_weights = torch.bmm(q, k.transpose(1, 2))
        if attn_mask is not None:
            attn_weights = torch.nan_to_num(attn_weights)
            attn_mask = attn_mask.unsqueeze(0)
            attn_weights += attn_mask
        if key_padding_mask is not None:
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).type_as(attn_weights)
        attn = torch.bmm(attn_weights, v)
        attn = attn.transpose(0, 1).reshape(tgt_len, bsz, embed_dim).transpose(0, 1)
        attn = self.out_proj(attn)
        attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len).transpose(1, 0)
        if need_weights:
            return attn, attn_weights
        else:
            return attn, None


def autopad(k, p=None):
    if p is None:
        p = k // 2 if isinstance(k, int) else [(x // 2) for x in k]
    return p


class Conv(nn.Module):

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        if isinstance(act, bool):
            self.act = nn.SiLU() if act is True else act if isinstance(act, nn.Module) else nn.Identity()
        elif isinstance(act, str):
            if act == 'leaky':
                self.act = nn.LeakyReLU(0.1, inplace=True)
            elif act == 'relu':
                self.act = nn.ReLU(inplace=True)
            else:
                self.act = None

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class Bottleneck(nn.Module):

    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5, act=True):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1, act=act)
        self.cv2 = Conv(c_, c2, 3, 1, g=g, act=act)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C3(nn.Module):

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, act=True):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1, act=act)
        self.cv2 = Conv(c1, c_, 1, 1, act=act)
        self.cv3 = Conv(2 * c_, c2, 1, act=act)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0, act=act) for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))


class double_conv_up_c3(nn.Module):

    def __init__(self, in_ch, mid_ch, out_ch, act=True):
        super(double_conv_up_c3, self).__init__()
        self.conv = nn.Sequential(C3(in_ch + mid_ch, mid_ch, act=act), nn.ConvTranspose2d(mid_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.conv(x)


class double_conv_c3(nn.Module):

    def __init__(self, in_ch, out_ch, stride=1, act=True):
        super(double_conv_c3, self).__init__()
        if stride > 1:
            self.down = nn.AvgPool2d(2, stride=2) if stride > 1 else None
        self.conv = C3(in_ch, out_ch, act=act)

    def forward(self, x):
        if self.down is not None:
            x = self.down(x)
        x = self.conv(x)
        return x


TEXTDET_DET = 1


TEXTDET_MASK = 0


class UnetHead(nn.Module):

    def __init__(self, act=True) ->None:
        super(UnetHead, self).__init__()
        self.down_conv1 = double_conv_c3(512, 512, 2, act=act)
        self.upconv0 = double_conv_up_c3(0, 512, 256, act=act)
        self.upconv2 = double_conv_up_c3(256, 512, 256, act=act)
        self.upconv3 = double_conv_up_c3(0, 512, 256, act=act)
        self.upconv4 = double_conv_up_c3(128, 256, 128, act=act)
        self.upconv5 = double_conv_up_c3(64, 128, 64, act=act)
        self.upconv6 = nn.Sequential(nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1, bias=False), nn.Sigmoid())

    def forward(self, f160, f80, f40, f20, f3, forward_mode=TEXTDET_MASK):
        d10 = self.down_conv1(f3)
        u20 = self.upconv0(d10)
        u40 = self.upconv2(torch.cat([f20, u20], dim=1))
        if forward_mode == TEXTDET_DET:
            return f80, f40, u40
        else:
            u80 = self.upconv3(torch.cat([f40, u40], dim=1))
            u160 = self.upconv4(torch.cat([f80, u80], dim=1))
            u320 = self.upconv5(torch.cat([f160, u160], dim=1))
            mask = self.upconv6(u320)
            if forward_mode == TEXTDET_MASK:
                return mask
            else:
                return mask, [f80, f40, u40]

    def init_weight(self, init_func):
        self.apply(init_func)


class DBHead(nn.Module):

    def __init__(self, in_channels, k=50, shrink_with_sigmoid=True, act=True):
        super().__init__()
        self.k = k
        self.shrink_with_sigmoid = shrink_with_sigmoid
        self.upconv3 = double_conv_up_c3(0, 512, 256, act=act)
        self.upconv4 = double_conv_up_c3(128, 256, 128, act=act)
        self.conv = nn.Sequential(nn.Conv2d(128, in_channels, 1), nn.BatchNorm2d(in_channels), nn.ReLU(inplace=True))
        self.binarize = nn.Sequential(nn.Conv2d(in_channels, in_channels // 4, 3, padding=1), nn.BatchNorm2d(in_channels // 4), nn.ReLU(inplace=True), nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 2, 2), nn.BatchNorm2d(in_channels // 4), nn.ReLU(inplace=True), nn.ConvTranspose2d(in_channels // 4, 1, 2, 2))
        self.thresh = self._init_thresh(in_channels)

    def forward(self, f80, f40, u40, shrink_with_sigmoid=True, step_eval=False):
        shrink_with_sigmoid = self.shrink_with_sigmoid
        u80 = self.upconv3(torch.cat([f40, u40], dim=1))
        x = self.upconv4(torch.cat([f80, u80], dim=1))
        x = self.conv(x)
        threshold_maps = self.thresh(x)
        x = self.binarize(x)
        shrink_maps = torch.sigmoid(x)
        if self.training:
            binary_maps = self.step_function(shrink_maps, threshold_maps)
            if shrink_with_sigmoid:
                return torch.cat((shrink_maps, threshold_maps, binary_maps), dim=1)
            else:
                return torch.cat((shrink_maps, threshold_maps, binary_maps, x), dim=1)
        elif step_eval:
            return self.step_function(shrink_maps, threshold_maps)
        else:
            return torch.cat((shrink_maps, threshold_maps), dim=1)

    def init_weight(self, init_func):
        self.apply(init_func)

    def _init_thresh(self, inner_channels, serial=False, smooth=False, bias=False):
        in_channels = inner_channels
        if serial:
            in_channels += 1
        self.thresh = nn.Sequential(nn.Conv2d(in_channels, inner_channels // 4, 3, padding=1, bias=bias), nn.BatchNorm2d(inner_channels // 4), nn.ReLU(inplace=True), self._init_upsample(inner_channels // 4, inner_channels // 4, smooth=smooth, bias=bias), nn.BatchNorm2d(inner_channels // 4), nn.ReLU(inplace=True), self._init_upsample(inner_channels // 4, 1, smooth=smooth, bias=bias), nn.Sigmoid())
        return self.thresh

    def _init_upsample(self, in_channels, out_channels, smooth=False, bias=False):
        if smooth:
            inter_out_channels = out_channels
            if out_channels == 1:
                inter_out_channels = in_channels
            module_list = [nn.Upsample(scale_factor=2, mode='nearest'), nn.Conv2d(in_channels, inter_out_channels, 3, 1, 1, bias=bias)]
            if out_channels == 1:
                module_list.append(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=1, bias=True))
            return nn.Sequential(module_list)
        else:
            return nn.ConvTranspose2d(in_channels, out_channels, 2, 2)

    def step_function(self, x, y):
        return torch.reciprocal(1 + torch.exp(-self.k * (x - y)))


CTD_MODEL_PATH = 'data/models/comictextdetector.pt'


REFINEMASK_INPAINT = 0


class SegDetectorRepresenter:

    def __init__(self, thresh=0.3, box_thresh=0.7, max_candidates=1000, unclip_ratio=1.5):
        self.min_size = 3
        self.thresh = thresh
        self.box_thresh = box_thresh
        self.max_candidates = max_candidates
        self.unclip_ratio = unclip_ratio

    def __call__(self, batch, pred, is_output_polygon=False, height=None, width=None):
        """
        batch: (image, polygons, ignore_tags
        batch: a dict produced by dataloaders.
            image: tensor of shape (N, C, H, W).
            polygons: tensor of shape (N, K, 4, 2), the polygons of objective regions.
            ignore_tags: tensor of shape (N, K), indicates whether a region is ignorable or not.
            shape: the original shape of images.
            filename: the original filenames of images.
        pred:
            binary: text region segmentation map, with shape (N, H, W)
            thresh: [if exists] thresh hold prediction with shape (N, H, W)
            thresh_binary: [if exists] binarized with threshhold, (N, H, W)
        """
        pred = pred[:, 0, :, :]
        segmentation = self.binarize(pred)
        boxes_batch = []
        scores_batch = []
        batch_size = pred.size(0) if isinstance(pred, torch.Tensor) else pred.shape[0]
        if height is None:
            height = pred.shape[1]
        if width is None:
            width = pred.shape[2]
        for batch_index in range(batch_size):
            if is_output_polygon:
                boxes, scores = self.polygons_from_bitmap(pred[batch_index], segmentation[batch_index], width, height)
            else:
                boxes, scores = self.boxes_from_bitmap(pred[batch_index], segmentation[batch_index], width, height)
            boxes_batch.append(boxes)
            scores_batch.append(scores)
        return boxes_batch, scores_batch

    def binarize(self, pred):
        return pred > self.thresh

    def polygons_from_bitmap(self, pred, _bitmap, dest_width, dest_height):
        """
        _bitmap: single map with shape (H, W),
            whose values are binarized as {0, 1}
        """
        assert len(_bitmap.shape) == 2
        bitmap = _bitmap.cpu().numpy()
        pred = pred.cpu().detach().numpy()
        height, width = bitmap.shape
        boxes = []
        scores = []
        contours, _ = cv2.findContours((bitmap * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours[:self.max_candidates]:
            epsilon = 0.005 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            points = approx.reshape((-1, 2))
            if points.shape[0] < 4:
                continue
            score = self.box_score_fast(pred, contour.squeeze(1))
            if self.box_thresh > score:
                continue
            if points.shape[0] > 2:
                box = self.unclip(points, unclip_ratio=self.unclip_ratio)
                if len(box) > 1:
                    continue
            else:
                continue
            box = box.reshape(-1, 2)
            _, sside = self.get_mini_boxes(box.reshape((-1, 1, 2)))
            if sside < self.min_size + 2:
                continue
            if not isinstance(dest_width, int):
                dest_width = dest_width.item()
                dest_height = dest_height.item()
            box[:, 0] = np.clip(np.round(box[:, 0] / width * dest_width), 0, dest_width)
            box[:, 1] = np.clip(np.round(box[:, 1] / height * dest_height), 0, dest_height)
            boxes.append(box)
            scores.append(score)
        return boxes, scores

    def boxes_from_bitmap(self, pred, _bitmap, dest_width, dest_height):
        """
        _bitmap: single map with shape (H, W),
            whose values are binarized as {0, 1}
        """
        assert len(_bitmap.shape) == 2
        if isinstance(pred, torch.Tensor):
            bitmap = _bitmap.cpu().numpy()
            pred = pred.cpu().detach().numpy()
        else:
            bitmap = _bitmap
        height, width = bitmap.shape
        contours, _ = cv2.findContours((bitmap * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        num_contours = min(len(contours), self.max_candidates)
        boxes = np.zeros((num_contours, 4, 2), dtype=np.int64)
        scores = np.zeros((num_contours,), dtype=np.float32)
        for index in range(num_contours):
            contour = contours[index].squeeze(1)
            points, sside = self.get_mini_boxes(contour)
            if sside < 2:
                continue
            points = np.array(points)
            score = self.box_score_fast(pred, contour)
            box = self.unclip(points, unclip_ratio=self.unclip_ratio).reshape(-1, 1, 2)
            box, sside = self.get_mini_boxes(box)
            box = np.array(box)
            if not isinstance(dest_width, int):
                dest_width = dest_width.item()
                dest_height = dest_height.item()
            box[:, 0] = np.clip(np.round(box[:, 0] / width * dest_width), 0, dest_width)
            box[:, 1] = np.clip(np.round(box[:, 1] / height * dest_height), 0, dest_height)
            boxes[index, :, :] = box.astype(np.int64)
            scores[index] = score
        return boxes, scores

    def unclip(self, box, unclip_ratio=1.5):
        poly = Polygon(box)
        distance = poly.area * unclip_ratio / poly.length
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        expanded = np.array(offset.Execute(distance))
        return expanded

    def get_mini_boxes(self, contour):
        bounding_box = cv2.minAreaRect(contour)
        points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])
        index_1, index_2, index_3, index_4 = 0, 1, 2, 3
        if points[1][1] > points[0][1]:
            index_1 = 0
            index_4 = 1
        else:
            index_1 = 1
            index_4 = 0
        if points[3][1] > points[2][1]:
            index_2 = 2
            index_3 = 3
        else:
            index_2 = 3
            index_3 = 2
        box = [points[index_1], points[index_2], points[index_3], points[index_4]]
        return box, min(bounding_box[1])

    def box_score_fast(self, bitmap, _box):
        h, w = bitmap.shape[:2]
        box = _box.copy()
        xmin = np.clip(np.floor(box[:, 0].min()).astype(np.int64), 0, w - 1)
        xmax = np.clip(np.ceil(box[:, 0].max()).astype(np.int64), 0, w - 1)
        ymin = np.clip(np.floor(box[:, 1].min()).astype(np.int64), 0, h - 1)
        ymax = np.clip(np.ceil(box[:, 1].max()).astype(np.int64), 0, h - 1)
        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
        box[:, 0] = box[:, 0] - xmin
        box[:, 1] = box[:, 1] - ymin
        cv2.fillPoly(mask, box.reshape(1, -1, 2).astype(np.int32), 1)
        if bitmap.dtype == np.float16:
            bitmap = bitmap.astype(np.float32)
        return cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0]


CJKPATTERN = re.compile('[\\uac00-\\ud7a3\\u3040-\\u30ff\\u4e00-\\u9FFF]')


@dataclass
class Config:

    def update(self, key: 'str', value):
        assert key in self.__annotations__, f"type object '{self.__class__.__name__}' has no attribute {key}"
        self.__setattr__(key, value)

    @classmethod
    def annotations_set(cls):
        return set(list(cls.__annotations__))

    def __getitem__(self, key: 'str'):
        assert key in self.__annotations__, f"type object '{self.__class__.__name__}' has no attribute {key}"
        return self.__getattribute__(key)

    def __setitem__(self, key: 'str', value):
        self.__setattr__(key, value)

    @classmethod
    def params(cls):
        return cls.__annotations__

    def merge(self, target):
        tgt_keys = target.annotations_set()
        for key in tgt_keys:
            self.update(key, target[key])

    def copy(self):
        return copy.deepcopy(self)


class LineSpacingType(enum.IntEnum):
    Proportional = 0
    Distance = 1


fontweight_pattern = re.compile('font-weight:(\\d+)', re.DOTALL)


fontweight_qt5_to_qt6 = {(0): 100, (12): 200, (25): 300, (50): 400, (57): 500, (63): 600, (75): 700, (81): 800, (87): 900}


fontweight_qt6_to_qt5 = {(100): 0, (200): 12, (300): 25, (400): 50, (500): 57, (600): 63, (700): 75, (800): 81, (900): 87}


def fix_fontweight_qt(weight: 'Union[str, int]'):

    def _fix_html_fntweight(matched):
        weight = int(matched.group(1))
        return f'font-weight:{fix_fontweight_qt(weight)}'
    if weight is None:
        return None
    if isinstance(weight, int):
        if shared.FLAG_QT6 and weight < 100:
            if weight in fontweight_qt5_to_qt6:
                weight = fontweight_qt5_to_qt6[weight]
        if not shared.FLAG_QT6 and weight >= 100:
            if weight in fontweight_qt6_to_qt5:
                weight = fontweight_qt6_to_qt5[weight]
    if isinstance(weight, str):
        weight = fontweight_pattern.sub(lambda matched: _fix_html_fntweight(matched), weight)
    return weight


def nested_dataclass(*args, **dataclass_kwargs):
    """
    nested dataclass support 

    also ignore extra arguments 
    """

    def wrapper(check_class):
        check_class = dataclass(check_class, **dataclass_kwargs)
        o_init = check_class.__init__

        def __init__(self, *args, **kwargs):
            store_deprecated = 'deprecated_attributes' in self.__annotations__
            deprecated = {}
            for name in list(kwargs.keys()):
                if name not in self.__annotations__:
                    val = kwargs.pop(name)
                    if store_deprecated:
                        deprecated[name] = val
                    continue
                value = kwargs[name]
                ft = check_class.__annotations__.get(name, None)
                if is_dataclass(ft) and isinstance(value, dict):
                    obj = ft(**value)
                    kwargs[name] = obj
            if len(deprecated) > 0:
                kwargs['deprecated_attributes'] = deprecated
            o_init(self, *args, **kwargs)
        check_class.__init__ = __init__
        return check_class
    return wrapper(args[0]) if args else wrapper


def pt2px(pt, to_int=False) ->float:
    if to_int:
        return int(round(pt * shared.LDPI / 72.0))
    else:
        return pt * shared.LDPI / 72.0


def px2pt(px) ->float:
    return px / shared.LDPI * 72.0


class TextAlignment(enum.IntEnum):
    Left = 0
    Center = 1
    Right = 2


def color_difference(rgb1: 'List', rgb2: 'List') ->float:
    color1 = np.array(rgb1, dtype=np.uint8).reshape(1, 1, 3)
    color2 = np.array(rgb2, dtype=np.uint8).reshape(1, 1, 3)
    diff = cv2.cvtColor(color1, cv2.COLOR_RGB2LAB).astype(np.float64) - cv2.cvtColor(color2, cv2.COLOR_RGB2LAB).astype(np.float64)
    diff[..., 0] *= 0.392
    diff = np.linalg.norm(diff, axis=2)
    return diff.item()


def rotate_polygons(center, polygons, rotation, new_center=None, to_int=True):
    if new_center is None:
        new_center = center
    rotation = np.deg2rad(rotation)
    s, c = np.sin(rotation), np.cos(rotation)
    polygons = polygons.astype(np.float32)
    polygons[:, 1::2] -= center[1]
    polygons[:, ::2] -= center[0]
    rotated = np.copy(polygons)
    rotated[:, 1::2] = polygons[:, 1::2] * c - polygons[:, ::2] * s
    rotated[:, ::2] = polygons[:, 1::2] * s + polygons[:, ::2] * c
    rotated[:, 1::2] += new_center[1]
    rotated[:, ::2] += new_center[0]
    if to_int:
        return rotated.astype(np.int64)
    return rotated


def xywh2xyxypoly(xywh, to_int=True):
    xyxypoly = np.tile(xywh[:, [0, 1]], 4)
    xyxypoly[:, [2, 4]] += xywh[:, [2]]
    xyxypoly[:, [5, 7]] += xywh[:, [3]]
    if to_int:
        xyxypoly = xyxypoly.astype(np.int64)
    return xyxypoly


TEXTDET_INFERENCE = 2


def fuse_conv_and_bn(conv, bn):
    fusedconv = nn.Conv2d(conv.in_channels, conv.out_channels, kernel_size=conv.kernel_size, stride=conv.stride, padding=conv.padding, groups=conv.groups, bias=True).requires_grad_(False)
    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))
    b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)
    return fusedconv


def check_version(current='0.0.0', minimum='0.0.0', name='version ', pinned=False, hard=False):
    current, minimum = (pkg.parse_version(x) for x in (current, minimum))
    result = current == minimum if pinned else current >= minimum
    if hard:
        assert result, f'{name}{minimum} required by YOLOv5, but {name}{current} is currently installed'
    else:
        return result


class Detect(nn.Module):
    stride = None
    onnx_dynamic = False

    def __init__(self, nc=80, anchors=(), ch=(), inplace=True):
        super().__init__()
        self.nc = nc
        self.no = nc + 5
        self.nl = len(anchors)
        self.na = len(anchors[0]) // 2
        self.grid = [torch.zeros(1)] * self.nl
        self.anchor_grid = [torch.zeros(1)] * self.nl
        self.register_buffer('anchors', torch.tensor(anchors).float().view(self.nl, -1, 2))
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)
        self.inplace = inplace

    def forward(self, x):
        z = []
        for i in range(self.nl):
            x[i] = self.m[i](x[i])
            bs, _, ny, nx = x[i].shape
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            if not self.training:
                if self.onnx_dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)
                y = x[i].sigmoid()
                if self.inplace:
                    y[..., 0:2] = (y[..., 0:2] * 2 - 0.5 + self.grid[i]) * self.stride[i]
                    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]
                else:
                    xy = (y[..., 0:2] * 2 - 0.5 + self.grid[i]) * self.stride[i]
                    wh = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]
                    y = torch.cat((xy, wh, y[..., 4:]), -1)
                z.append(y.view(bs, -1, self.no))
        return x if self.training else (torch.cat(z, 1), x)

    def _make_grid(self, nx=20, ny=20, i=0):
        d = self.anchors[i].device
        if check_version(torch.__version__, '1.10.0'):
            yv, xv = torch.meshgrid([torch.arange(ny, device=d), torch.arange(nx, device=d)], indexing='ij')
        else:
            yv, xv = torch.meshgrid([torch.arange(ny, device=d), torch.arange(nx, device=d)])
        grid = torch.stack((xv, yv), 2).expand((1, self.na, ny, nx, 2)).float()
        anchor_grid = (self.anchors[i].clone() * self.stride[i]).view((1, self.na, 1, 1, 2)).expand((1, self.na, ny, nx, 2)).float()
        return grid, anchor_grid


class DWConv(Conv):

    def __init__(self, c1, c2, k=1, s=1, act=True):
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), act=act)


def check_anchor_order(m):
    a = m.anchors.prod(-1).view(-1)
    da = a[-1] - a[0]
    ds = m.stride[-1] - m.stride[0]
    if da.sign() != ds.sign():
        m.anchors[:] = m.anchors.flip(0)


def initialize_weights(model):
    for m in model.modules():
        t = type(m)
        if t is nn.Conv2d:
            pass
        elif t is nn.BatchNorm2d:
            m.eps = 0.001
            m.momentum = 0.03
        elif t in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
            m.inplace = True


class BottleneckCSP(nn.Module):

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)
        self.act = nn.SiLU()
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), dim=1))))


class GhostConv(nn.Module):

    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        super().__init__()
        c_ = c2 // 2
        self.cv1 = Conv(c1, c_, k, s, None, g, act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act)

    def forward(self, x):
        y = self.cv1(x)
        return torch.cat([y, self.cv2(y)], 1)


class GhostBottleneck(nn.Module):

    def __init__(self, c1, c2, k=3, s=1):
        super().__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(GhostConv(c1, c_, 1, 1), DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(), GhostConv(c_, c2, 1, 1, act=False))
        self.shortcut = nn.Sequential(DWConv(c1, c1, k, s, act=False), Conv(c1, c2, 1, 1, act=False)) if s == 2 else nn.Identity()

    def forward(self, x):
        return self.conv(x) + self.shortcut(x)


class C3Ghost(C3):

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*(GhostBottleneck(c_, c_) for _ in range(n)))


class SPP(nn.Module):

    def __init__(self, c1, c2, k=(5, 9, 13)):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class C3SPP(C3):

    def __init__(self, c1, c2, k=(5, 9, 13), n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = SPP(c_, c_, k)


class TransformerLayer(nn.Module):

    def __init__(self, c, num_heads):
        super().__init__()
        self.q = nn.Linear(c, c, bias=False)
        self.k = nn.Linear(c, c, bias=False)
        self.v = nn.Linear(c, c, bias=False)
        self.ma = nn.MultiheadAttention(embed_dim=c, num_heads=num_heads)
        self.fc1 = nn.Linear(c, c, bias=False)
        self.fc2 = nn.Linear(c, c, bias=False)

    def forward(self, x):
        x = self.ma(self.q(x), self.k(x), self.v(x))[0] + x
        x = self.fc2(self.fc1(x)) + x
        return x


class TransformerBlock(nn.Module):

    def __init__(self, c1, c2, num_heads, num_layers):
        super().__init__()
        self.conv = None
        if c1 != c2:
            self.conv = Conv(c1, c2)
        self.linear = nn.Linear(c2, c2)
        self.tr = nn.Sequential(*(TransformerLayer(c2, num_heads) for _ in range(num_layers)))
        self.c2 = c2

    def forward(self, x):
        if self.conv is not None:
            x = self.conv(x)
        b, _, w, h = x.shape
        p = x.flatten(2).permute(2, 0, 1)
        return self.tr(p + self.linear(p)).permute(1, 2, 0).reshape(b, self.c2, w, h)


class C3TR(C3):

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = TransformerBlock(c_, c_, 4, n)


class Concat(nn.Module):

    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)


class Contract(nn.Module):

    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        b, c, h, w = x.size()
        s = self.gain
        x = x.view(b, c, h // s, s, w // s, s)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()
        return x.view(b, c * s * s, h // s, w // s)


class Expand(nn.Module):

    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        b, c, h, w = x.size()
        s = self.gain
        x = x.view(b, s, s, c // s ** 2, h, w)
        x = x.permute(0, 3, 4, 1, 5, 2).contiguous()
        return x.view(b, c // s ** 2, h * s, w * s)


class Focus(nn.Module):

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super().__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act)

    def forward(self, x):
        return self.conv(torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1))


class SPPF(nn.Module):

    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            y1 = self.m(x)
            y2 = self.m(y1)
            return self.cv2(torch.cat([x, y1, y2, self.m(y2)], 1))


def make_divisible(x, divisor):
    if isinstance(divisor, torch.Tensor):
        divisor = int(divisor.max())
    return math.ceil(x / divisor) * divisor


def parse_model(d, ch):
    anchors, nc, gd, gw = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple']
    na = len(anchors[0]) // 2 if isinstance(anchors, list) else anchors
    no = na * (nc + 5)
    layers, save, c2 = [], [], ch[-1]
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):
        m = eval(m) if isinstance(m, str) else m
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a
            except NameError:
                pass
        n = n_ = max(round(n * gd), 1) if n > 1 else n
        if m in [Conv, GhostConv, Bottleneck, GhostBottleneck, SPP, SPPF, DWConv, Focus, BottleneckCSP, C3, C3TR, C3SPP, C3Ghost]:
            c1, c2 = ch[f], args[0]
            if c2 != no:
                c2 = make_divisible(c2 * gw, 8)
            args = [c1, c2, *args[1:]]
            if m in [BottleneckCSP, C3, C3TR, C3Ghost]:
                args.insert(2, n)
                n = 1
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum(ch[x] for x in f)
        elif m is Detect:
            args.append([ch[x] for x in f])
            if isinstance(args[1], int):
                args[1] = [list(range(args[1] * 2))] * len(f)
        elif m is Contract:
            c2 = ch[f] * args[0] ** 2
        elif m is Expand:
            c2 = ch[f] // args[0] ** 2
        else:
            c2 = ch[f]
        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)
        t = str(m)[8:-2].replace('__main__.', '')
        np = sum(x.numel() for x in m_.parameters())
        m_.i, m_.f, m_.type, m_.np = i, f, t, np
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)


def scale_img(img, ratio=1.0, same_shape=False, gs=32):
    if ratio == 1.0:
        return img
    else:
        h, w = img.shape[2:]
        s = int(h * ratio), int(w * ratio)
        img = F.interpolate(img, size=s, mode='bilinear', align_corners=False)
        if not same_shape:
            h, w = (math.ceil(x * ratio / gs) * gs for x in (h, w))
        return F.pad(img, [0, w - s[1], 0, h - s[0]], value=0.447)


class Model(nn.Module):

    def __init__(self, cfg='yolov5s.yaml', ch=3, nc=None, anchors=None):
        super().__init__()
        self.out_indices = None
        if isinstance(cfg, dict):
            self.yaml = cfg
        else:
            self.yaml_file = Path(cfg).name
            with open(cfg, encoding='ascii', errors='ignore') as f:
                self.yaml = yaml.safe_load(f)
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)
        if nc and nc != self.yaml['nc']:
            self.yaml['nc'] = nc
        if anchors:
            self.yaml['anchors'] = round(anchors)
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch])
        self.names = [str(i) for i in range(self.yaml['nc'])]
        self.inplace = self.yaml.get('inplace', True)
        m = self.model[-1]
        if isinstance(m, Detect):
            s = 256
            m.inplace = self.inplace
            m.stride = torch.tensor([(s / x.shape[-2]) for x in self.forward(torch.zeros(1, ch, s, s))])
            m.anchors /= m.stride.view(-1, 1, 1)
            check_anchor_order(m)
            self.stride = m.stride
            self._initialize_biases()
        initialize_weights(self)

    def forward(self, x, augment=False, profile=False, visualize=False, detect=False):
        if augment:
            return self._forward_augment(x)
        return self._forward_once(x, profile, visualize, detect=detect)

    def _forward_augment(self, x):
        img_size = x.shape[-2:]
        s = [1, 0.83, 0.67]
        f = [None, 3, None]
        y = []
        for si, fi in zip(s, f):
            xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
            yi = self._forward_once(xi)[0]
            yi = self._descale_pred(yi, fi, si, img_size)
            y.append(yi)
        y = self._clip_augmented(y)
        return torch.cat(y, 1), None

    def _forward_once(self, x, profile=False, visualize=False, detect=False):
        y, dt = [], []
        z = []
        for ii, m in enumerate(self.model):
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [(x if j == -1 else y[j]) for j in m.f]
            if profile:
                self._profile_one_layer(m, x, dt)
            x = m(x)
            y.append(x if m.i in self.save else None)
            if self.out_indices is not None:
                if m.i in self.out_indices:
                    z.append(x)
        if self.out_indices is not None:
            if detect:
                return x, z
            else:
                return z
        else:
            return x

    def _descale_pred(self, p, flips, scale, img_size):
        if self.inplace:
            p[..., :4] /= scale
            if flips == 2:
                p[..., 1] = img_size[0] - p[..., 1]
            elif flips == 3:
                p[..., 0] = img_size[1] - p[..., 0]
        else:
            x, y, wh = p[..., 0:1] / scale, p[..., 1:2] / scale, p[..., 2:4] / scale
            if flips == 2:
                y = img_size[0] - y
            elif flips == 3:
                x = img_size[1] - x
            p = torch.cat((x, y, wh, p[..., 4:]), -1)
        return p

    def _clip_augmented(self, y):
        nl = self.model[-1].nl
        g = sum(4 ** x for x in range(nl))
        e = 1
        i = y[0].shape[1] // g * sum(4 ** x for x in range(e))
        y[0] = y[0][:, :-i]
        i = y[-1].shape[1] // g * sum(4 ** (nl - 1 - x) for x in range(e))
        y[-1] = y[-1][:, i:]
        return y

    def _profile_one_layer(self, m, x, dt):
        c = isinstance(m, Detect)
        for _ in range(10):
            m(x.copy() if c else x)

    def _initialize_biases(self, cf=None):
        m = self.model[-1]
        for mi, s in zip(m.m, m.stride):
            b = mi.bias.view(m.na, -1)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)
            b.data[:, 5:] += math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def _print_biases(self):
        m = self.model[-1]
        for mi in m.m:
            b = mi.bias.detach().view(m.na, -1).T

    def fuse(self):
        for m in self.model.modules():
            if isinstance(m, (Conv, DWConv)) and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)
                delattr(m, 'bn')
                m.forward = m.forward_fuse
        return self

    def _apply(self, fn):
        self = super()._apply(fn)
        m = self.model[-1]
        if isinstance(m, Detect):
            m.stride = fn(m.stride)
            m.grid = list(map(fn, m.grid))
            if isinstance(m.anchor_grid, list):
                m.anchor_grid = list(map(fn, m.anchor_grid))
        return self


@torch.no_grad()
def load_yolov5_ckpt(weights, map_location='cpu', fuse=True, inplace=True, out_indices=[1, 3, 5, 7, 9]):
    if isinstance(weights, str):
        ckpt = torch.load(weights, map_location=map_location)
    else:
        ckpt = weights
    model = Model(ckpt['cfg'])
    model.load_state_dict(ckpt['weights'], strict=True)
    if fuse:
        model = model.float().fuse().eval()
    else:
        model = model.float().eval()
    for m in model.modules():
        if type(m) in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU, Detect, Model]:
            m.inplace = inplace
            if type(m) is Detect:
                if not isinstance(m.anchor_grid, list):
                    delattr(m, 'anchor_grid')
                    setattr(m, 'anchor_grid', [torch.zeros(1)] * m.nl)
        elif type(m) is Conv:
            m._non_persistent_buffers_set = set()
    model.out_indices = out_indices
    return model


def get_base_det_models(model_path, device='cpu', half=False, act='leaky'):
    textdetector_dict = torch.load(model_path, map_location='cpu')
    blk_det = load_yolov5_ckpt(textdetector_dict['blk_det'])
    text_seg = UnetHead(act=act)
    text_seg.load_state_dict(textdetector_dict['text_seg'])
    text_det = DBHead(64, act=act)
    text_det.load_state_dict(textdetector_dict['text_det'])
    if half:
        return blk_det.eval().half(), text_seg.eval().half(), text_det.eval().half()
    return blk_det.eval(), text_seg.eval(), text_det.eval()


class TextDetBase(nn.Module):

    def __init__(self, model_path, device='cpu', half=False, fuse=False, act='leaky'):
        super(TextDetBase, self).__init__()
        self.blk_det, self.text_seg, self.text_det = get_base_det_models(model_path, device, half, act=act)
        if fuse:
            self.fuse()

    def fuse(self):

        def _fuse(model):
            for m in model.modules():
                if isinstance(m, Conv) and hasattr(m, 'bn'):
                    m.conv = fuse_conv_and_bn(m.conv, m.bn)
                    delattr(m, 'bn')
                    m.forward = m.forward_fuse
            return model
        self.text_seg = _fuse(self.text_seg)
        self.text_det = _fuse(self.text_det)

    def forward(self, features):
        blks, features = self.blk_det(features, detect=True)
        mask, features = self.text_seg(*features, forward_mode=TEXTDET_INFERENCE)
        lines = self.text_det(*features, step_eval=False)
        return blks[0], mask, lines


class TextDetBaseDNN:

    def __init__(self, input_size, model_path):
        self.input_size = input_size
        self.model = cv2.dnn.readNetFromONNX(model_path)
        self.uoln = self.model.getUnconnectedOutLayersNames()

    def __call__(self, im_in):
        blob = cv2.dnn.blobFromImage(im_in, scalefactor=1 / 255.0, size=(self.input_size, self.input_size))
        self.model.setInput(blob)
        blks, mask, lines_map = self.model.forward(self.uoln)
        return blks, mask, lines_map


def square_pad_resize(img: 'np.ndarray', tgt_size: 'int'):
    h, w = img.shape[:2]
    pad_h, pad_w = 0, 0
    if w < h:
        pad_w = h - w
        w += pad_w
    elif h < w:
        pad_h = w - h
        h += pad_h
    pad_size = tgt_size - h
    if pad_size > 0:
        pad_h += pad_size
        pad_w += pad_size
    if pad_h > 0 or pad_w > 0:
        img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT)
    down_scale_ratio = tgt_size / img.shape[0]
    assert down_scale_ratio <= 1
    if down_scale_ratio < 1:
        img = cv2.resize(img, (tgt_size, tgt_size), interpolation=cv2.INTER_AREA)
    return img, down_scale_ratio, pad_h, pad_w


def det_rearrange_forward(img: 'np.ndarray', dbnet_batch_forward: 'Callable[[np.ndarray, str], Tuple[np.ndarray, np.ndarray]]', tgt_size: 'int'=1280, max_batch_size: 'int'=4, device='cuda', crop_as_square=False, verbose=False):
    """
    Rearrange image to square batches before feeding into network if following conditions are satisfied: 

    1. Extreme aspect ratio
    2. Is too tall or wide for detect size (tgt_size)

    Returns:
        DBNet output, mask or None, None if rearrangement is not required
    """

    def _unrearrange(patch_lst: 'List[np.ndarray]', transpose: 'bool', channel=1, pad_num=0):
        _psize = _h = patch_lst[0].shape[-1]
        _step = int(ph_step * _psize / patch_size)
        _pw = int(_psize / pw_num)
        _h = int(_pw / w * h)
        tgtmap = np.zeros((channel, _h, _pw), dtype=np.float32)
        num_patches = len(patch_lst) * pw_num - pad_num
        for ii, p in enumerate(patch_lst):
            if transpose:
                p = einops.rearrange(p, 'c h w -> c w h')
            for jj in range(pw_num):
                pidx = ii * pw_num + jj
                rel_t = rel_step_list[pidx]
                t = int(round(rel_t * _h))
                b = min(t + _psize, _h)
                l = jj * _pw
                r = l + _pw
                tgtmap[..., t:b, :] += p[..., :b - t, l:r]
                if pidx > 0:
                    interleave = _psize - _step
                    tgtmap[..., t:t + interleave, :] /= 2.0
                if pidx >= num_patches - 1:
                    break
        if transpose:
            tgtmap = einops.rearrange(tgtmap, 'c h w -> c w h')
        return tgtmap[None, ...]

    def _patch2batches(patch_lst: 'List[np.ndarray]', p_num: 'int', transpose: 'bool'):
        if transpose:
            patch_lst = einops.rearrange(patch_lst, '(p_num pw_num) ph pw c -> p_num (pw_num pw) ph c', p_num=p_num)
        else:
            patch_lst = einops.rearrange(patch_lst, '(p_num pw_num) ph pw c -> p_num ph (pw_num pw) c', p_num=p_num)
        batches = [[]]
        for ii, patch in enumerate(patch_lst):
            if len(batches[-1]) >= max_batch_size:
                batches.append([])
            p, down_scale_ratio, pad_h, pad_w = square_pad_resize(patch, tgt_size=tgt_size)
            assert pad_h == pad_w
            pad_size = pad_h
            batches[-1].append(p)
            if verbose:
                cv2.imwrite(f'result/rearrange_{ii}.png', p[..., ::-1])
        return batches, down_scale_ratio, pad_size
    h, w = img.shape[:2]
    transpose = False
    if h < w:
        transpose = True
        h, w = img.shape[1], img.shape[0]
    asp_ratio = h / w
    down_scale_ratio = h / tgt_size
    require_rearrange = down_scale_ratio > 2.5 and asp_ratio > 3
    if not require_rearrange:
        return None, None
    if verbose:
        None
    if transpose:
        img = einops.rearrange(img, 'h w c -> w h c')
    if crop_as_square:
        pw_num = 1
    else:
        pw_num = max(int(np.floor(2 * tgt_size / w)), 2)
    patch_size = ph = pw_num * w
    ph_num = int(np.ceil(h / ph))
    ph_step = int((h - ph) / (ph_num - 1)) if ph_num > 1 else 0
    rel_step_list = []
    patch_list = []
    for ii in range(ph_num):
        t = ii * ph_step
        b = t + ph
        rel_step_list.append(t / h)
        patch_list.append(img[t:b])
    p_num = int(np.ceil(ph_num / pw_num))
    pad_num = p_num * pw_num - ph_num
    for ii in range(pad_num):
        patch_list.append(np.zeros_like(patch_list[0]))
    batches, down_scale_ratio, pad_size = _patch2batches(patch_list, p_num, transpose)
    db_lst, mask_lst = [], []
    for batch in batches:
        batch = np.array(batch)
        db, mask = dbnet_batch_forward(batch, device=device)
        for ii, (d, m) in enumerate(zip(db, mask)):
            if pad_size > 0:
                paddb = int(db.shape[-1] / tgt_size * pad_size)
                padmsk = int(mask.shape[-1] / tgt_size * pad_size)
                d = d[..., :-paddb, :-paddb]
                m = m[..., :-padmsk, :-padmsk]
            db_lst.append(d)
            mask_lst.append(m)
            if verbose:
                cv2.imwrite(f'result/rearrange_db_{ii}.png', (d[0] * 255).astype(np.uint8))
                cv2.imwrite(f'result/rearrange_thr_{ii}.png', (d[1] * 255).astype(np.uint8))
    db = _unrearrange(db_lst, transpose, channel=2, pad_num=pad_num)
    mask = _unrearrange(mask_lst, transpose, channel=1, pad_num=pad_num)
    return db, mask


LANG_LIST = ['eng', 'ja', 'unknown']


def examine_textblk(blk: 'TextBlock', im_w: 'int', im_h: 'int', sort: 'bool'=False) ->None:
    lines = blk.lines_array()
    middle_pnts = (lines[:, [1, 2, 3, 0]] + lines) / 2
    vec_v = middle_pnts[:, 2] - middle_pnts[:, 0]
    vec_h = middle_pnts[:, 1] - middle_pnts[:, 3]
    center_pnts = (lines[:, 0] + lines[:, 2]) / 2
    v = np.sum(vec_v, axis=0)
    h = np.sum(vec_h, axis=0)
    norm_v, norm_h = np.linalg.norm(v), np.linalg.norm(h)
    if blk.language == 'ja':
        vertical = norm_v > norm_h
    else:
        vertical = norm_v > norm_h * 2
    if vertical:
        primary_vec, primary_norm = v, norm_v
        distance_vectors = center_pnts - np.array([[im_w, 0]], dtype=np.float64)
        font_size = int(round(norm_h / len(lines)))
    else:
        primary_vec, primary_norm = h, norm_h
        distance_vectors = center_pnts - np.array([[0, 0]], dtype=np.float64)
        font_size = int(round(norm_v / len(lines)))
    rotation_angle = int(math.atan2(primary_vec[1], primary_vec[0]) / math.pi * 180)
    distance = np.linalg.norm(distance_vectors, axis=1)
    rad_matrix = np.arccos(np.einsum('ij, j->i', distance_vectors, primary_vec) / (distance * primary_norm))
    distance = np.abs(np.sin(rad_matrix) * distance)
    blk.lines = lines.astype(np.int32).tolist()
    blk.distance = distance
    blk.angle = rotation_angle
    if vertical:
        blk.angle -= 90
    if abs(blk.angle) < 3:
        blk.angle = 0
    blk.font_size = font_size
    blk.vertical = blk.src_is_vertical = vertical
    blk.vec = primary_vec
    blk.norm = primary_norm
    if sort:
        blk.sort_lines()


def try_merge_textline(blk: 'TextBlock', blk2: 'TextBlock', fntsize_tol=1.7, distance_tol=2) ->bool:
    if blk2.merged:
        return False
    fntsize_div = blk.font_size / blk2.font_size
    num_l1, num_l2 = len(blk), len(blk2)
    fntsz_avg = (blk.font_size * num_l1 + blk2.font_size * num_l2) / (num_l1 + num_l2)
    vec_prod = blk.vec @ blk2.vec
    vec_sum = blk.vec + blk2.vec
    cos_vec = vec_prod / blk.norm / blk2.norm
    distance = blk2.distance[-1] - blk.distance[-1]
    distance_p1 = np.linalg.norm(np.array(blk2.lines[-1][0]) - np.array(blk.lines[-1][0]))
    l1, l2 = Polygon(blk.lines[-1]), Polygon(blk2.lines[-1])
    if not l1.intersects(l2):
        if fntsize_div > fntsize_tol or 1 / fntsize_div > fntsize_tol:
            return False
        if abs(cos_vec) < 0.866:
            return False
        if distance > distance_tol * fntsz_avg:
            return False
        if blk.vertical and blk2.vertical and distance_p1 > fntsz_avg * 2.5:
            return False
    blk.lines.append(blk2.lines[0])
    blk.vec = vec_sum
    blk.angle = int(round(np.rad2deg(math.atan2(vec_sum[1], vec_sum[0]))))
    if blk.vertical:
        blk.angle -= 90
    blk.norm = np.linalg.norm(vec_sum)
    blk.distance = np.append(blk.distance, blk2.distance[-1])
    blk.font_size = fntsz_avg
    blk2.merged = True
    return True


def sort_pnts(pts: 'np.ndarray'):
    """
    Direction must be provided for sorting.
    The longer structure vector (mean of long side vectors) of input points is used to determine the direction.
    It is reliable enough for text lines but not for blocks.
    """
    if isinstance(pts, List):
        pts = np.array(pts)
    assert isinstance(pts, np.ndarray) and pts.shape == (4, 2)
    pairwise_vec = (pts[:, None] - pts[None]).reshape((16, -1))
    pairwise_vec_norm = np.linalg.norm(pairwise_vec, axis=1)
    long_side_ids = np.argsort(pairwise_vec_norm)[[8, 10]]
    long_side_vecs = pairwise_vec[long_side_ids]
    inner_prod = (long_side_vecs[0] * long_side_vecs[1]).sum()
    if inner_prod < 0:
        long_side_vecs[0] = -long_side_vecs[0]
    struc_vec = np.abs(long_side_vecs.mean(axis=0))
    is_vertical = struc_vec[0] <= struc_vec[1]
    if is_vertical:
        pts = pts[np.argsort(pts[:, 1])]
        pts = pts[[*np.argsort(pts[:2, 0]), *(np.argsort(pts[2:, 0])[::-1] + 2)]]
        return pts, is_vertical
    else:
        pts = pts[np.argsort(pts[:, 0])]
        pts_sorted = np.zeros_like(pts)
        pts_sorted[[0, 3]] = sorted(pts[[0, 1]], key=lambda x: x[1])
        pts_sorted[[1, 2]] = sorted(pts[[2, 3]], key=lambda x: x[1])
        return pts_sorted, is_vertical


def split_textblk(blk: 'TextBlock'):
    font_size, distance, lines = blk.font_size, blk.distance, blk.lines
    l0 = np.array(blk.lines[0])
    lines.sort(key=lambda line: np.linalg.norm(np.array(line[0]) - l0[0]))
    distance_tol = font_size * 2
    current_blk = copy.deepcopy(blk)
    current_blk.lines = [l0]
    sub_blk_list = [current_blk]
    textblock_splitted = False
    for jj, line in enumerate(lines[1:]):
        l1, l2 = Polygon(lines[jj]), Polygon(line)
        split = False
        if not l1.intersects(l2):
            line_disance = abs(distance[jj + 1] - distance[jj])
            if line_disance > distance_tol:
                split = True
            elif blk.vertical and abs(blk.angle) < 15:
                if len(current_blk.lines) > 1 or line_disance > font_size:
                    split = abs(lines[jj][0][1] - line[0][1]) > font_size
        if split:
            current_blk = copy.deepcopy(current_blk)
            current_blk.lines = [line]
            sub_blk_list.append(current_blk)
        else:
            current_blk.lines.append(line)
    if len(sub_blk_list) > 1:
        textblock_splitted = True
        for current_blk in sub_blk_list:
            current_blk.adjust_bbox(with_bbox=False)
    return textblock_splitted, sub_blk_list


def union_area(bboxa, bboxb):
    x1 = max(bboxa[0], bboxb[0])
    y1 = max(bboxa[1], bboxb[1])
    x2 = min(bboxa[2], bboxb[2])
    y2 = min(bboxa[3], bboxb[3])
    if y2 < y1 or x2 < x1:
        return -1
    return (y2 - y1) * (x2 - x1)


def postprocess_mask(img: 'Union[torch.Tensor, np.ndarray]', thresh=None):
    if isinstance(img, torch.Tensor):
        img = img.squeeze_()
        if img.device != 'cpu':
            img = img.detach().cpu()
        img = img.numpy()
    else:
        img = img.squeeze()
    if thresh is not None:
        img = img > thresh
    img = img * 255
    return img.astype(np.uint8)


def box_iou(box1, box2):
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        return (box[2] - box[0]) * (box[3] - box[1])
    area1 = box_area(box1.T)
    area2 = box_area(box2.T)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)


def xywh2xyxy(x):
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y


def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, multi_label=False, labels=(), max_det=300):
    """Runs Non-Maximum Suppression (NMS) on inference results

    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """
    if isinstance(prediction, np.ndarray):
        prediction = torch.from_numpy(prediction)
    nc = prediction.shape[2] - 5
    xc = prediction[..., 4] > conf_thres
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'
    min_wh, max_wh = 2, 4096
    max_nms = 30000
    time_limit = 10.0
    redundant = True
    multi_label &= nc > 1
    merge = False
    t = time.time()
    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
    for xi, x in enumerate(prediction):
        x = x[xc[xi]]
        if labels and len(labels[xi]):
            l = labels[xi]
            v = torch.zeros((len(l), nc + 5), device=x.device)
            v[:, :4] = l[:, 1:5]
            v[:, 4] = 1.0
            v[range(len(l)), l[:, 0].long() + 5] = 1.0
            x = torch.cat((x, v), 0)
        if not x.shape[0]:
            continue
        x[:, 5:] *= x[:, 4:5]
        box = xywh2xyxy(x[:, :4])
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]
        n = x.shape[0]
        if not n:
            continue
        elif n > max_nms:
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]
        c = x[:, 5:6] * (0 if agnostic else max_wh)
        boxes, scores = x[:, :4] + c, x[:, 4]
        i = torchvision.ops.nms(boxes, scores, iou_thres)
        if i.shape[0] > max_det:
            i = i[:max_det]
        if merge and 1 < n < 3000.0:
            iou = box_iou(boxes[i], boxes) > iou_thres
            weights = iou * scores[None]
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)
            if redundant:
                i = i[iou.sum(1) > 1]
        output[xi] = x[i]
        if time.time() - t > time_limit:
            None
            break
    return output


def postprocess_yolo(det, conf_thresh, nms_thresh, resize_ratio, sort_func=None):
    det = non_max_suppression(det, conf_thresh, nms_thresh)[0]
    if det.device != 'cpu':
        det = det.detach_().cpu().numpy()
    det[..., [0, 2]] = det[..., [0, 2]] * resize_ratio[0]
    det[..., [1, 3]] = det[..., [1, 3]] * resize_ratio[1]
    if sort_func is not None:
        det = sort_func(det)
    blines = det[..., 0:4].astype(np.int32)
    confs = np.round(det[..., 4], 3)
    cls = det[..., 5].astype(np.int32)
    return blines, cls, confs


def letterbox(im, new_shape=(640, 640), color=(0, 0, 0), auto=False, scaleFill=False, scaleup=True, stride=128):
    shape = im.shape[:2]
    if not isinstance(new_shape, tuple):
        new_shape = new_shape, new_shape
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)
    ratio = r, r
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    if auto:
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    elif scaleFill:
        dw, dh = 0.0, 0.0
        new_unpad = new_shape[1], new_shape[0]
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]
    dh, dw = int(dh), int(dw)
    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, 0, dh, 0, dw, cv2.BORDER_CONSTANT, value=color)
    return im, ratio, (dw, dh)


def preprocess_img(img, detect_size=(1024, 1024), device='cpu', bgr2rgb=True, half=False, to_tensor=True):
    if isinstance(detect_size, int):
        detect_size = detect_size, detect_size
    if bgr2rgb:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_in, ratio, (dw, dh) = letterbox(img, new_shape=detect_size, auto=False, stride=64)
    if to_tensor:
        img_in = img_in.transpose((2, 0, 1))[::-1]
        img_in = np.array([np.ascontiguousarray(img_in)]).astype(np.float32) / 255
        if to_tensor:
            img_in = torch.from_numpy(img_in)
            if half:
                img_in = img_in.half()
    return img_in, ratio, int(dw), int(dh)


def enlarge_window(rect, im_w, im_h, ratio=2.5, aspect_ratio=1.0) ->List:
    assert ratio > 1.0
    x1, y1, x2, y2 = rect
    w = x2 - x1
    h = y2 - y1
    if w <= 0 or h <= 0:
        return [0, 0, 0, 0]
    coeff = [aspect_ratio, w + h * aspect_ratio, (1 - ratio) * w * h]
    roots = np.roots(coeff)
    roots.sort()
    delta = int(round(roots[-1] / 2))
    delta_w = int(delta * aspect_ratio)
    delta_w = min(x1, im_w - x2, delta_w)
    delta = min(y1, im_h - y2, delta)
    rect = np.array([x1 - delta_w, y1 - delta, x2 + delta_w, y2 + delta], dtype=np.int64)
    rect[::2] = np.clip(rect[::2], 0, im_w)
    rect[1::2] = np.clip(rect[1::2], 0, im_h)
    return rect.tolist()


def minxor_thresh(threshed, mask, dilate=False):
    neg_threshed = 255 - threshed
    e_size = 1
    if dilate:
        element = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * e_size + 1, 2 * e_size + 1), (e_size, e_size))
        neg_threshed = cv2.dilate(neg_threshed, element, iterations=1)
        threshed = cv2.dilate(threshed, element, iterations=1)
    neg_xor_sum = cv2.bitwise_xor(neg_threshed, mask).sum()
    xor_sum = cv2.bitwise_xor(threshed, mask).sum()
    if neg_xor_sum < xor_sum:
        return neg_threshed, neg_xor_sum
    else:
        return threshed, xor_sum


def get_otsuthresh_masklist(img, pred_mask, per_channel=False) ->List[np.ndarray]:
    channels = [img[..., 0], img[..., 1], img[..., 2]]
    mask_list = []
    for c in channels:
        _, threshed = cv2.threshold(c, 1, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
        threshed, xor_sum = minxor_thresh(threshed, pred_mask, dilate=False)
        mask_list.append([threshed, xor_sum])
    mask_list.sort(key=lambda x: x[1])
    if per_channel:
        return mask_list
    else:
        return [mask_list[0]]


def get_topk_color(color_list, bins, k=3, color_var=10, bin_tol=0.001):
    idx = np.argsort(bins * -1)
    color_list, bins = color_list[idx], bins[idx]
    top_colors = [color_list[0]]
    bin_tol = np.sum(bins) * bin_tol
    if len(color_list) > 1:
        for color, bin in zip(color_list[1:], bins[1:]):
            if np.abs(np.array(top_colors) - color).min() > color_var:
                top_colors.append(color)
            if len(top_colors) >= k or bin < bin_tol:
                break
    return top_colors


def get_topk_masklist(im_grey, pred_mask):
    if len(im_grey.shape) == 3 and im_grey.shape[-1] == 3:
        im_grey = cv2.cvtColor(im_grey, cv2.COLOR_BGR2GRAY)
    msk = np.ascontiguousarray(pred_mask)
    candidate_grey_px = im_grey[np.where(cv2.erode(msk, np.ones((3, 3), np.uint8), iterations=1) > 127)]
    bin, his = np.histogram(candidate_grey_px, bins=255)
    topk_color = get_topk_color(his, bin, color_var=10, k=3)
    color_range = 30
    mask_list = list()
    for ii, color in enumerate(topk_color):
        c_top = min(color + color_range, 255)
        c_bottom = c_top - 2 * color_range
        threshed = cv2.inRange(im_grey, c_bottom, c_top)
        threshed, xor_sum = minxor_thresh(threshed, msk)
        mask_list.append([threshed, xor_sum])
    return mask_list


def merge_mask_list(mask_list, pred_mask, blk: 'TextBlock'=None, pred_thresh=30, text_window=None, filter_with_lines=False, refine_mode=REFINEMASK_INPAINT):
    mask_list.sort(key=lambda x: x[1])
    linemask = None
    if blk is not None and filter_with_lines:
        linemask = np.zeros_like(pred_mask)
        lines = blk.lines_array(dtype=np.int64)
        for line in lines:
            line[..., 0] -= text_window[0]
            line[..., 1] -= text_window[1]
            cv2.fillPoly(linemask, [line], 255)
        linemask = cv2.dilate(linemask, np.ones((3, 3), np.uint8), iterations=3)
    if pred_thresh > 0:
        e_size = 1
        element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * e_size + 1, 2 * e_size + 1), (e_size, e_size))
        pred_mask = cv2.erode(pred_mask, element, iterations=1)
        _, pred_mask = cv2.threshold(pred_mask, 60, 255, cv2.THRESH_BINARY)
    connectivity = 8
    mask_merged = np.zeros_like(pred_mask)
    for ii, (candidate_mask, xor_sum) in enumerate(mask_list):
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(candidate_mask, connectivity, cv2.CV_16U)
        for label_index, stat, centroid in zip(range(num_labels), stats, centroids):
            if label_index != 0:
                x, y, w, h, area = stat
                if w * h < 3:
                    continue
                x1, y1, x2, y2 = x, y, x + w, y + h
                label_local = labels[y1:y2, x1:x2]
                label_cordinates = np.where(label_local == label_index)
                tmp_merged = np.zeros_like(label_local, np.uint8)
                tmp_merged[label_cordinates] = 255
                tmp_merged = cv2.bitwise_or(mask_merged[y1:y2, x1:x2], tmp_merged)
                xor_merged = cv2.bitwise_xor(tmp_merged, pred_mask[y1:y2, x1:x2]).sum()
                xor_origin = cv2.bitwise_xor(mask_merged[y1:y2, x1:x2], pred_mask[y1:y2, x1:x2]).sum()
                if xor_merged < xor_origin:
                    mask_merged[y1:y2, x1:x2] = tmp_merged
    if refine_mode == REFINEMASK_INPAINT:
        mask_merged = cv2.dilate(mask_merged, np.ones((5, 5), np.uint8), iterations=1)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(255 - mask_merged, connectivity, cv2.CV_16U)
    sorted_area = np.sort(stats[:, -1])
    if len(sorted_area) > 1:
        area_thresh = sorted_area[-2]
    else:
        area_thresh = sorted_area[-1]
    for label_index, stat, centroid in zip(range(num_labels), stats, centroids):
        x, y, w, h, area = stat
        if area < area_thresh:
            x1, y1, x2, y2 = x, y, x + w, y + h
            label_local = labels[y1:y2, x1:x2]
            label_cordinates = np.where(label_local == label_index)
            tmp_merged = np.zeros_like(label_local, np.uint8)
            tmp_merged[label_cordinates] = 255
            tmp_merged = cv2.bitwise_or(mask_merged[y1:y2, x1:x2], tmp_merged)
            xor_merged = cv2.bitwise_xor(tmp_merged, pred_mask[y1:y2, x1:x2]).sum()
            xor_origin = cv2.bitwise_xor(mask_merged[y1:y2, x1:x2], pred_mask[y1:y2, x1:x2]).sum()
            if xor_merged < xor_origin:
                mask_merged[y1:y2, x1:x2] = tmp_merged
    return mask_merged


def refine_mask(img: 'np.ndarray', pred_mask: 'np.ndarray', blk_list: 'List[TextBlock]', refine_mode: 'int'=REFINEMASK_INPAINT) ->np.ndarray:
    mask_refined = np.zeros_like(pred_mask)
    for blk in blk_list:
        bx1, by1, bx2, by2 = enlarge_window(blk.xyxy, img.shape[1], img.shape[0])
        im = np.ascontiguousarray(img[by1:by2, bx1:bx2])
        msk = np.ascontiguousarray(pred_mask[by1:by2, bx1:bx2])
        mask_list = get_topk_masklist(im, msk)
        mask_list += get_otsuthresh_masklist(im, msk, per_channel=False)
        mask_merged = merge_mask_list(mask_list, msk, blk=blk, text_window=[bx1, by1, bx2, by2], refine_mode=refine_mode)
        mask_refined[by1:by2, bx1:bx2] = cv2.bitwise_or(mask_refined[by1:by2, bx1:bx2], mask_merged)
    return mask_refined


def refine_undetected_mask(img: 'np.ndarray', mask_pred: 'np.ndarray', mask_refined: 'np.ndarray', blk_list: 'List[TextBlock]', refine_mode=REFINEMASK_INPAINT):
    mask_pred[np.where(mask_refined > 30)] = 0
    _, pred_mask_t = cv2.threshold(mask_pred, 30, 255, cv2.THRESH_BINARY)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(pred_mask_t, 4, cv2.CV_16U)
    valid_labels = np.where(stats[:, -1] > 50)[0]
    seg_blk_list = []
    if len(valid_labels) > 0:
        for lab_index in valid_labels[1:]:
            x, y, w, h, area = stats[lab_index]
            bx1, by1 = x, y
            bx2, by2 = x + w, y + h
            bbox = [bx1, by1, bx2, by2]
            bbox_score = -1
            for blk in blk_list:
                bbox_s = union_area(blk.xyxy, bbox)
                if bbox_s > bbox_score:
                    bbox_score = bbox_s
            if bbox_score / w / h < 0.5:
                seg_blk_list.append(TextBlock(bbox))
    if len(seg_blk_list) > 0:
        mask_refined = cv2.bitwise_or(mask_refined, refine_mask(img, mask_pred, seg_blk_list, refine_mode=refine_mode))
    return mask_refined


class Classify(nn.Module):

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1):
        super().__init__()
        self.aap = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g)
        self.flat = nn.Flatten()

    def forward(self, x):
        z = torch.cat([self.aap(y) for y in (x if isinstance(x, list) else [x])], 1)
        return self.flat(self.conv(z))


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (AOTBlock,
     lambda: ([], {'dim': 4}),
     lambda: ([torch.rand([4, 4, 64, 64])], {})),
    (AddCoords,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (BasicBlock,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
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
    (C3SPP,
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
    (ConvNeXtBlock,
     lambda: ([], {'dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ConvNext_FeatureExtractor,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {})),
    (CustomTransformerEncoderLayer,
     lambda: ([], {'d_model': 4, 'nhead': 4}),
     lambda: ([torch.rand([4, 4, 4])], {})),
    (DWConv,
     lambda: ([], {'c1': 4, 'c2': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Discriminator,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {})),
    (Expand,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Focus,
     lambda: ([], {'c1': 4, 'c2': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (FourierUnit,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (GatedWSConvPadded,
     lambda: ([], {'in_ch': 4, 'out_ch': 4, 'ks': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (GatedWSTransposeConvPadded,
     lambda: ([], {'in_ch': 4, 'out_ch': 4, 'ks': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (GhostBottleneck,
     lambda: ([], {'c1': 4, 'c2': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (GhostConv,
     lambda: ([], {'c1': 4, 'c2': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (LambdaLayer,
     lambda: ([], {'f': torch.nn.ReLU()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (MultiLabelEmbedding,
     lambda: ([], {'num_positions': 4, 'embedding_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (NLayerDiscriminator,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {})),
    (PositionalEncoding,
     lambda: ([], {'d_model': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ResBlock,
     lambda: ([], {'ch': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ResBlockDis,
     lambda: ([], {'in_planes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ResNet_FeatureExtractor,
     lambda: ([], {'input_channel': 4}),
     lambda: ([torch.rand([4, 4, 64, 64])], {})),
    (SPP,
     lambda: ([], {'c1': 4, 'c2': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (SPPF,
     lambda: ([], {'c1': 4, 'c2': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ScaledWSConv2d,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ScaledWSTransposeConv2d,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (TransformerBlock,
     lambda: ([], {'c1': 4, 'c2': 4, 'num_heads': 4, 'num_layers': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (TransformerLayer,
     lambda: ([], {'c': 4, 'num_heads': 4}),
     lambda: ([torch.rand([4, 4])], {})),
    (XPOS,
     lambda: ([], {'head_dim': 4}),
     lambda: ([torch.rand([4, 4, 4])], {})),
    (double_conv_up_c3,
     lambda: ([], {'in_ch': 4, 'mid_ch': 4, 'out_ch': 4}),
     lambda: ([torch.rand([4, 8, 64, 64])], {})),
]

