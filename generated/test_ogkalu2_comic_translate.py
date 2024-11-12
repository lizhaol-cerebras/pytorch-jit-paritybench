
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


import abc


from typing import Optional


import torch


import numpy as np


import torch.nn as nn


import torch.nn.functional as F


from torch import Tensor


import re


from torch.utils.data import Dataset


from collections import namedtuple


import torch.nn.init as init


from torchvision import models


from collections import OrderedDict


import torch.backends.cudnn as cudnn


from torch.autograd import Variable


import math


import torch.utils.data


import torchvision.transforms as transforms


import logging


from torch.hub import download_url_to_file


from typing import List


from torch.hub import get_dir


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


def init_weights(modules):
    for m in modules:
        if isinstance(m, nn.Conv2d):
            init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.01)
            m.bias.data.zero_()


model_urls = {'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth', 'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth', 'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth', 'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth', 'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth', 'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth', 'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth', 'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth'}


class Vgg16BN(torch.nn.Module):

    def __init__(self, pretrained: 'bool'=True, freeze: 'bool'=True):
        super(Vgg16BN, self).__init__()
        model_urls['vgg16_bn'] = model_urls['vgg16_bn'].replace('https://', 'http://')
        vgg_pretrained_features = models.vgg16_bn(pretrained=pretrained).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(12):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 19):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(19, 29):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(29, 39):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        self.slice5 = torch.nn.Sequential(nn.MaxPool2d(kernel_size=3, stride=1, padding=1), nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6), nn.Conv2d(1024, 1024, kernel_size=1))
        if not pretrained:
            init_weights(self.slice1.modules())
            init_weights(self.slice2.modules())
            init_weights(self.slice3.modules())
            init_weights(self.slice4.modules())
        init_weights(self.slice5.modules())
        if freeze:
            for param in self.slice1.parameters():
                param.requires_grad = False

    def forward(self, x):
        h = self.slice1(x)
        h_relu2_2 = h
        h = self.slice2(h)
        h_relu3_2 = h
        h = self.slice3(h)
        h_relu4_3 = h
        h = self.slice4(h)
        h_relu5_3 = h
        h = self.slice5(h)
        h_fc7 = h
        vgg_outputs = namedtuple('VggOutputs', ['fc7', 'relu5_3', 'relu4_3', 'relu3_2', 'relu2_2'])
        out = vgg_outputs(h_fc7, h_relu5_3, h_relu4_3, h_relu3_2, h_relu2_2)
        return out


class VGGFeatureExtractor(nn.Module):
    """ FeatureExtractor of CRNN (https://arxiv.org/pdf/1507.05717.pdf) """

    def __init__(self, n_input_channels: 'int'=1, n_output_channels: 'int'=512, opt2val=None):
        super(VGGFeatureExtractor, self).__init__()
        self.output_channel = [int(n_output_channels / 8), int(n_output_channels / 4), int(n_output_channels / 2), n_output_channels]
        rec_model_ckpt_fp = opt2val['rec_model_ckpt_fp']
        if 'baseline' in rec_model_ckpt_fp:
            self.ConvNet = nn.Sequential(nn.Conv2d(n_input_channels, self.output_channel[0], 3, 1, 1), nn.ReLU(True), nn.MaxPool2d(2, 2), nn.Conv2d(self.output_channel[0], self.output_channel[1], 3, 1, 1), nn.ReLU(True), nn.MaxPool2d(2, 2), nn.Conv2d(self.output_channel[1], self.output_channel[2], 3, 1, 1), nn.ReLU(True), nn.Conv2d(self.output_channel[2], self.output_channel[2], 3, 1, 1), nn.ReLU(True), nn.MaxPool2d((2, 1), (2, 1)), nn.Conv2d(self.output_channel[2], self.output_channel[3], 3, 1, 1, bias=False), nn.BatchNorm2d(self.output_channel[3]), nn.ReLU(True), nn.Conv2d(self.output_channel[3], self.output_channel[3], 3, 1, 1, bias=False), nn.BatchNorm2d(self.output_channel[3]), nn.ReLU(True), nn.MaxPool2d((2, 1), (2, 1)), nn.ConvTranspose2d(self.output_channel[3], self.output_channel[3], 2, 2), nn.ReLU(True))
        else:
            self.ConvNet = nn.Sequential(nn.Conv2d(n_input_channels, self.output_channel[0], 3, 1, 1), nn.ReLU(True), nn.MaxPool2d(2, 2), nn.Conv2d(self.output_channel[0], self.output_channel[1], 3, 1, 1), nn.ReLU(True), nn.MaxPool2d(2, 2), nn.Conv2d(self.output_channel[1], self.output_channel[2], 3, 1, 1), nn.ReLU(True), nn.Conv2d(self.output_channel[2], self.output_channel[2], 3, 1, 1), nn.ReLU(True), nn.MaxPool2d((2, 1), (2, 1)), nn.Conv2d(self.output_channel[2], self.output_channel[3], 3, 1, 1, bias=False), nn.BatchNorm2d(self.output_channel[3]), nn.ReLU(True), nn.Conv2d(self.output_channel[3], self.output_channel[3], 3, 1, 1, bias=False), nn.BatchNorm2d(self.output_channel[3]), nn.ReLU(True), nn.MaxPool2d((2, 1), (2, 1)), nn.ConvTranspose2d(self.output_channel[3], self.output_channel[3], 2, 2), nn.ReLU(True), nn.ConvTranspose2d(self.output_channel[3], self.output_channel[3], 2, 2), nn.ReLU(True))

    def forward(self, x):
        return self.ConvNet(x)


class BidirectionalLSTM(nn.Module):

    def __init__(self, input_size: 'int', hidden_size: 'int', output_size: 'int'):
        super(BidirectionalLSTM, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        """
        x : visual feature [batch_size x T=24 x input_size=512]
        output : contextual feature [batch_size x T x output_size]
        """
        self.rnn.flatten_parameters()
        recurrent, _ = self.rnn(x)
        output = self.linear(recurrent)
        return output


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes: 'int', planes: 'int', stride: 'int'=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = self._conv3x3(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = self._conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def _conv3x3(self, in_planes, out_planes, stride=1):
        """3x3 convolution with padding"""
        return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):

    def __init__(self, n_input_channels: 'int', n_output_channels: 'int', block, layers):
        """
        :param n_input_channels (int): The number of input channels of the feature extractor
        :param n_output_channels (int): The number of output channels of the feature extractor
        :param block:
        :param layers:
        """
        super(ResNet, self).__init__()
        self.output_channel_blocks = [int(n_output_channels / 4), int(n_output_channels / 2), n_output_channels, n_output_channels]
        self.inplanes = int(n_output_channels / 8)
        self.conv0_1 = nn.Conv2d(n_input_channels, int(n_output_channels / 16), kernel_size=3, stride=1, padding=1, bias=False)
        self.bn0_1 = nn.BatchNorm2d(int(n_output_channels / 16))
        self.conv0_2 = nn.Conv2d(int(n_output_channels / 16), self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn0_2 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.layer1 = self._make_layer(block, self.output_channel_blocks[0], layers[0])
        self.conv1 = nn.Conv2d(self.output_channel_blocks[0], self.output_channel_blocks[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.output_channel_blocks[0])
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.layer2 = self._make_layer(block, self.output_channel_blocks[1], layers[1], stride=1)
        self.conv2 = nn.Conv2d(self.output_channel_blocks[1], self.output_channel_blocks[1], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(self.output_channel_blocks[1])
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=(2, 1), padding=(0, 1))
        self.layer3 = self._make_layer(block, self.output_channel_blocks[2], layers[2], stride=1)
        self.conv3 = nn.Conv2d(self.output_channel_blocks[2], self.output_channel_blocks[2], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.output_channel_blocks[2])
        self.layer4 = self._make_layer(block, self.output_channel_blocks[3], layers[3], stride=1)
        self.conv4_1 = nn.Conv2d(self.output_channel_blocks[3], self.output_channel_blocks[3], kernel_size=2, stride=(2, 1), padding=(0, 1), bias=False)
        self.bn4_1 = nn.BatchNorm2d(self.output_channel_blocks[3])
        self.conv4_2 = nn.Conv2d(self.output_channel_blocks[3], self.output_channel_blocks[3], kernel_size=2, stride=1, padding=0, bias=False)
        self.bn4_2 = nn.BatchNorm2d(self.output_channel_blocks[3])

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv0_1(x)
        x = self.bn0_1(x)
        x = self.relu(x)
        x = self.conv0_2(x)
        x = self.bn0_2(x)
        x = self.relu(x)
        x = self.maxpool1(x)
        x = self.layer1(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool2(x)
        x = self.layer2(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.maxpool3(x)
        x = self.layer3(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.layer4(x)
        x = self.conv4_1(x)
        x = self.bn4_1(x)
        x = self.relu(x)
        x = self.conv4_2(x)
        x = self.bn4_2(x)
        x = self.relu(x)
        return x


class ResNetFeatureExtractor(nn.Module):
    """
    FeatureExtractor of FAN
    (http://openaccess.thecvf.com/content_ICCV_2017/papers/Cheng_Focusing_Attention_Towards_ICCV_2017_paper.pdf)
    """

    def __init__(self, n_input_channels: 'int'=1, n_output_channels: 'int'=512):
        super(ResNetFeatureExtractor, self).__init__()
        self.ConvNet = ResNet(n_input_channels, n_output_channels, BasicBlock, [1, 2, 5, 3])

    def forward(self, inputs):
        return self.ConvNet(inputs)


class GridGenerator(nn.Module):
    """ Grid Generator of RARE, which produces P_prime by multipling T with P """

    def __init__(self, F, I_r_size):
        """ Generate P_hat and inv_delta_C for later """
        super(GridGenerator, self).__init__()
        self.eps = 1e-06
        self.I_r_height, self.I_r_width = I_r_size
        self.F = F
        self.C = self._build_C(self.F)
        self.P = self._build_P(self.I_r_width, self.I_r_height)
        self.register_buffer('inv_delta_C', torch.tensor(self._build_inv_delta_C(self.F, self.C)).float())
        self.register_buffer('P_hat', torch.tensor(self._build_P_hat(self.F, self.C, self.P)).float())

    def _build_C(self, F):
        """ Return coordinates of fiducial points in I_r; C """
        ctrl_pts_x = np.linspace(-1.0, 1.0, int(F / 2))
        ctrl_pts_y_top = -1 * np.ones(int(F / 2))
        ctrl_pts_y_bottom = np.ones(int(F / 2))
        ctrl_pts_top = np.stack([ctrl_pts_x, ctrl_pts_y_top], axis=1)
        ctrl_pts_bottom = np.stack([ctrl_pts_x, ctrl_pts_y_bottom], axis=1)
        C = np.concatenate([ctrl_pts_top, ctrl_pts_bottom], axis=0)
        return C

    def _build_inv_delta_C(self, F, C):
        """ Return inv_delta_C which is needed to calculate T """
        hat_C = np.zeros((F, F), dtype=float)
        for i in range(0, F):
            for j in range(i, F):
                r = np.linalg.norm(C[i] - C[j])
                hat_C[i, j] = r
                hat_C[j, i] = r
        np.fill_diagonal(hat_C, 1)
        hat_C = hat_C ** 2 * np.log(hat_C)
        delta_C = np.concatenate([np.concatenate([np.ones((F, 1)), C, hat_C], axis=1), np.concatenate([np.zeros((2, 3)), np.transpose(C)], axis=1), np.concatenate([np.zeros((1, 3)), np.ones((1, F))], axis=1)], axis=0)
        inv_delta_C = np.linalg.inv(delta_C)
        return inv_delta_C

    def _build_P(self, I_r_width, I_r_height):
        I_r_grid_x = (np.arange(-I_r_width, I_r_width, 2) + 1.0) / I_r_width
        I_r_grid_y = (np.arange(-I_r_height, I_r_height, 2) + 1.0) / I_r_height
        P = np.stack(np.meshgrid(I_r_grid_x, I_r_grid_y), axis=2)
        return P.reshape([-1, 2])

    def _build_P_hat(self, F, C, P):
        n = P.shape[0]
        P_tile = np.tile(np.expand_dims(P, axis=1), (1, F, 1))
        C_tile = np.expand_dims(C, axis=0)
        P_diff = P_tile - C_tile
        rbf_norm = np.linalg.norm(P_diff, ord=2, axis=2, keepdims=False)
        rbf = np.multiply(np.square(rbf_norm), np.log(rbf_norm + self.eps))
        P_hat = np.concatenate([np.ones((n, 1)), P, rbf], axis=1)
        return P_hat

    def build_P_prime(self, batch_C_prime):
        """ Generate Grid from batch_C_prime [batch_size x F x 2] """
        batch_size = batch_C_prime.size(0)
        batch_inv_delta_C = self.inv_delta_C.repeat(batch_size, 1, 1)
        batch_P_hat = self.P_hat.repeat(batch_size, 1, 1)
        batch_C_prime_with_zeros = torch.cat((batch_C_prime, torch.zeros(batch_size, 3, 2).float()), dim=1)
        batch_T = torch.bmm(batch_inv_delta_C, batch_C_prime_with_zeros)
        batch_P_prime = torch.bmm(batch_P_hat, batch_T)
        return batch_P_prime


class LocalizationNetwork(nn.Module):
    """ Localization Network of RARE, which predicts C' (K x 2) from I (I_width x I_height) """

    def __init__(self, F, I_channel_num: 'int'):
        super(LocalizationNetwork, self).__init__()
        self.F = F
        self.I_channel_num = I_channel_num
        self.conv = nn.Sequential(nn.Conv2d(in_channels=self.I_channel_num, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(True), nn.MaxPool2d(2, 2), nn.Conv2d(64, 128, 3, 1, 1, bias=False), nn.BatchNorm2d(128), nn.ReLU(True), nn.MaxPool2d(2, 2), nn.Conv2d(128, 256, 3, 1, 1, bias=False), nn.BatchNorm2d(256), nn.ReLU(True), nn.MaxPool2d(2, 2), nn.Conv2d(256, 512, 3, 1, 1, bias=False), nn.BatchNorm2d(512), nn.ReLU(True), nn.AdaptiveAvgPool2d(1))
        self.localization_fc1 = nn.Sequential(nn.Linear(512, 256), nn.ReLU(True))
        self.localization_fc2 = nn.Linear(256, self.F * 2)
        self.localization_fc2.weight.data.fill_(0)
        ctrl_pts_x = np.linspace(-1.0, 1.0, int(F / 2))
        ctrl_pts_y_top = np.linspace(0.0, -1.0, num=int(F / 2))
        ctrl_pts_y_bottom = np.linspace(1.0, 0.0, num=int(F / 2))
        ctrl_pts_top = np.stack([ctrl_pts_x, ctrl_pts_y_top], axis=1)
        ctrl_pts_bottom = np.stack([ctrl_pts_x, ctrl_pts_y_bottom], axis=1)
        initial_bias = np.concatenate([ctrl_pts_top, ctrl_pts_bottom], axis=0)
        self.localization_fc2.bias.data = torch.from_numpy(initial_bias).float().view(-1)

    def forward(self, batch_I):
        """
        :param batch_I : Batch Input Image [batch_size x I_channel_num x I_height x I_width]
        :return: batch_C_prime : Predicted coordinates of fiducial points for input batch [batch_size x F x 2]
        """
        batch_size = batch_I.size(0)
        features = self.conv(batch_I).view(batch_size, -1)
        batch_C_prime = self.localization_fc2(self.localization_fc1(features)).view(batch_size, self.F, 2)
        return batch_C_prime


class TpsSpatialTransformerNetwork(nn.Module):
    """ Rectification Network of RARE, namely TPS based STN """

    def __init__(self, F, I_size, I_r_size, I_channel_num: 'int'=1):
        """Based on RARE TPS
        input:
            batch_I: Batch Input Image [batch_size x I_channel_num x I_height x I_width]
            I_size : (height, width) of the input image I
            I_r_size : (height, width) of the rectified image I_r
            I_channel_num : the number of channels of the input image I
        output:
            batch_I_r: rectified image [batch_size x I_channel_num x I_r_height x I_r_width]
        """
        super(TpsSpatialTransformerNetwork, self).__init__()
        self.F = F
        self.I_size = I_size
        self.I_r_size = I_r_size
        self.I_channel_num = I_channel_num
        self.LocalizationNetwork = LocalizationNetwork(self.F, self.I_channel_num)
        self.GridGenerator = GridGenerator(self.F, self.I_r_size)

    def forward(self, batch_I):
        batch_C_prime = self.LocalizationNetwork(batch_I)
        build_P_prime = self.GridGenerator.build_P_prime(batch_C_prime)
        build_P_prime_reshape = build_P_prime.reshape([build_P_prime.size(0), self.I_r_size[0], self.I_r_size[1], 2])
        batch_I_r = F.grid_sample(batch_I, build_P_prime_reshape, padding_mode='border')
        return batch_I_r


class DoubleConv(nn.Module):

    def __init__(self, in_ch: 'int', mid_ch: 'int', out_ch: 'int') ->None:
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_ch + mid_ch, mid_ch, kernel_size=1), nn.BatchNorm2d(mid_ch), nn.ReLU(inplace=True), nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))

    def forward(self, x: 'Tensor'):
        x = self.conv(x)
        return x


class CRAFT(nn.Module):

    def __init__(self, pretrained: 'bool'=False, freeze: 'bool'=False) ->None:
        super(CRAFT, self).__init__()
        self.basenet = Vgg16BN(pretrained, freeze)
        self.upconv1 = DoubleConv(1024, 512, 256)
        self.upconv2 = DoubleConv(512, 256, 128)
        self.upconv3 = DoubleConv(256, 128, 64)
        self.upconv4 = DoubleConv(128, 64, 32)
        num_class = 2
        self.conv_cls = nn.Sequential(nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(32, 16, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(16, 16, kernel_size=1), nn.ReLU(inplace=True), nn.Conv2d(16, num_class, kernel_size=1))
        init_weights(self.upconv1.modules())
        init_weights(self.upconv2.modules())
        init_weights(self.upconv3.modules())
        init_weights(self.upconv4.modules())
        init_weights(self.conv_cls.modules())

    def forward(self, x: 'Tensor'):
        sources = self.basenet(x)
        y = torch.cat([sources[0], sources[1]], dim=1)
        y = self.upconv1(y)
        y = F.interpolate(y, size=sources[2].size()[2:], mode='bilinear', align_corners=False)
        y = torch.cat([y, sources[2]], dim=1)
        y = self.upconv2(y)
        y = F.interpolate(y, size=sources[3].size()[2:], mode='bilinear', align_corners=False)
        y = torch.cat([y, sources[3]], dim=1)
        y = self.upconv3(y)
        y = F.interpolate(y, size=sources[4].size()[2:], mode='bilinear', align_corners=False)
        y = torch.cat([y, sources[4]], dim=1)
        feature = self.upconv4(y)
        y = self.conv_cls(feature)
        return y.permute(0, 2, 3, 1), feature


class AttentionCell(nn.Module):

    def __init__(self, input_size, hidden_size, num_embeddings):
        super(AttentionCell, self).__init__()
        self.i2h = nn.Linear(input_size, hidden_size, bias=False)
        self.h2h = nn.Linear(hidden_size, hidden_size)
        self.score = nn.Linear(hidden_size, 1, bias=False)
        self.rnn = nn.LSTMCell(input_size + num_embeddings, hidden_size)
        self.hidden_size = hidden_size

    def forward(self, prev_hidden, batch_H, char_onehots):
        batch_H_proj = self.i2h(batch_H)
        prev_hidden_proj = self.h2h(prev_hidden[0]).unsqueeze(1)
        e = self.score(torch.tanh(batch_H_proj + prev_hidden_proj))
        alpha = F.softmax(e, dim=1)
        context = torch.bmm(alpha.permute(0, 2, 1), batch_H).squeeze(1)
        concat_context = torch.cat([context, char_onehots], 1)
        cur_hidden = self.rnn(concat_context, prev_hidden)
        return cur_hidden, alpha


class Attention(nn.Module):

    def __init__(self, input_size, hidden_size, num_classes):
        super(Attention, self).__init__()
        self.attention_cell = AttentionCell(input_size, hidden_size, num_classes)
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.generator = nn.Linear(hidden_size, num_classes)

    def _char_to_onehot(self, input_char, onehot_dim=38):
        input_char = input_char.unsqueeze(1)
        batch_size = input_char.size(0)
        one_hot = torch.FloatTensor(batch_size, onehot_dim).zero_()
        one_hot = one_hot.scatter_(1, input_char, 1)
        return one_hot

    def forward(self, batch_H, text, is_train=True, batch_max_length=25):
        """
        input:
            batch_H : contextual_feature H = hidden state of encoder. [batch_size x num_steps x contextual_feature_channels]
            text : the text-index of each image. [batch_size x (max_length+1)]. +1 for [GO] token. text[:, 0] = [GO].
        output: probability distribution at each step [batch_size x num_steps x num_classes]
        """
        batch_size = batch_H.size(0)
        num_steps = batch_max_length + 1
        output_hiddens = torch.FloatTensor(batch_size, num_steps, self.hidden_size).fill_(0)
        hidden = torch.FloatTensor(batch_size, self.hidden_size).fill_(0), torch.FloatTensor(batch_size, self.hidden_size).fill_(0)
        if is_train:
            for i in range(num_steps):
                char_onehots = self._char_to_onehot(text[:, i], onehot_dim=self.num_classes)
                hidden, alpha = self.attention_cell(hidden, batch_H, char_onehots)
                output_hiddens[:, i, :] = hidden[0]
            probs = self.generator(output_hiddens)
        else:
            targets = torch.LongTensor(batch_size).fill_(0)
            probs = torch.FloatTensor(batch_size, num_steps, self.num_classes).fill_(0)
            for i in range(num_steps):
                char_onehots = self._char_to_onehot(targets, onehot_dim=self.num_classes)
                hidden, alpha = self.attention_cell(hidden, batch_H, char_onehots)
                probs_step = self.generator(hidden[0])
                probs[:, i, :] = probs_step
                _, next_input = probs_step.max(1)
                targets = next_input
        return probs


class Model(nn.Module):

    def __init__(self, opt2val: 'dict'):
        super(Model, self).__init__()
        input_channel = opt2val['input_channel']
        output_channel = opt2val['output_channel']
        hidden_size = opt2val['hidden_size']
        vocab_size = opt2val['vocab_size']
        num_fiducial = opt2val['num_fiducial']
        imgH = opt2val['imgH']
        imgW = opt2val['imgW']
        FeatureExtraction = opt2val['FeatureExtraction']
        Transformation = opt2val['Transformation']
        SequenceModeling = opt2val['SequenceModeling']
        Prediction = opt2val['Prediction']
        if Transformation == 'TPS':
            self.Transformation = TpsSpatialTransformerNetwork(F=num_fiducial, I_size=(imgH, imgW), I_r_size=(imgH, imgW), I_channel_num=input_channel)
        else:
            None
        if FeatureExtraction == 'VGG':
            extractor = VGGFeatureExtractor
        else:
            extractor = ResNetFeatureExtractor
        self.FeatureExtraction = extractor(input_channel, output_channel, opt2val)
        self.FeatureExtraction_output = output_channel
        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((None, 1))
        if SequenceModeling == 'BiLSTM':
            self.SequenceModeling = nn.Sequential(BidirectionalLSTM(self.FeatureExtraction_output, hidden_size, hidden_size), BidirectionalLSTM(hidden_size, hidden_size, hidden_size))
            self.SequenceModeling_output = hidden_size
        else:
            None
            self.SequenceModeling_output = self.FeatureExtraction_output
        if Prediction == 'CTC':
            self.Prediction = nn.Linear(self.SequenceModeling_output, vocab_size)
        elif Prediction == 'Attn':
            self.Prediction = Attention(self.SequenceModeling_output, hidden_size, vocab_size)
        elif Prediction == 'Transformer':
            pass
        else:
            raise Exception('Prediction is neither CTC or Attn')

    def forward(self, x: 'Tensor'):
        """
        :param x: (batch, input_channel, height, width)
        :return:
        """
        x = self.Transformation(x)
        visual_feature = self.FeatureExtraction(x)
        visual_feature = self.AdaptiveAvgPool(visual_feature.permute(0, 3, 1, 2))
        visual_feature = visual_feature.squeeze(3)
        self.SequenceModeling.eval()
        contextual_feature = self.SequenceModeling(visual_feature)
        prediction = self.Prediction(contextual_feature.contiguous())
        return prediction


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (BasicBlock,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (BidirectionalLSTM,
     lambda: ([], {'input_size': 4, 'hidden_size': 4, 'output_size': 4}),
     lambda: ([torch.rand([4, 4])], {})),
    (CRAFT,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {})),
    (DoubleConv,
     lambda: ([], {'in_ch': 4, 'mid_ch': 4, 'out_ch': 4}),
     lambda: ([torch.rand([4, 8, 64, 64])], {})),
    (FourierUnit,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (LocalizationNetwork,
     lambda: ([], {'F': 4, 'I_channel_num': 4}),
     lambda: ([torch.rand([4, 4, 64, 64])], {})),
    (MultiLabelEmbedding,
     lambda: ([], {'num_positions': 4, 'embedding_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ResNetFeatureExtractor,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1, 64, 64])], {})),
    (TpsSpatialTransformerNetwork,
     lambda: ([], {'F': 4, 'I_size': 4, 'I_r_size': [4, 4]}),
     lambda: ([torch.rand([4, 1, 64, 64])], {})),
    (Vgg16BN,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {})),
]

