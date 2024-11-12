
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


from torch.utils.cpp_extension import CUDAExtension


from torch.utils.cpp_extension import CppExtension


from torch.utils.cpp_extension import BuildExtension


import numpy as np


import random


from torch.utils.data import Dataset


import math


import torch.nn as nn


import torch.utils.data


import torch.nn.functional as F


import torch.backends.cudnn as cudnn


from torch.utils.data import DataLoader


import torch.nn.parallel


import torchvision.utils as vutils


import copy


from torch.autograd import Variable


from torch import Tensor


import torch.distributed as dist


from torch.nn.modules.batchnorm import _BatchNorm


from functools import partial


from torch.utils.data import DistributedSampler as _DistributedSampler


from torch.utils.data import Sampler


from torchvision import transforms


from numpy.linalg import inv


from torch.optim.optimizer import Optimizer


import functools


import time


from collections import defaultdict


import warnings


import matplotlib.pylab as plt


from torch.cuda.amp import custom_bwd


from torch.cuda.amp import custom_fwd


from torch.autograd.function import Function


from torch.autograd.function import once_differentiable


from torch.nn.init import normal_


from torchvision.transforms.functional import rotate


class MobileV2_Residual(nn.Module):

    def __init__(self, inp, oup, stride, expanse_ratio, dilation=1):
        super(MobileV2_Residual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
        hidden_dim = int(inp * expanse_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup
        pad = dilation
        if expanse_ratio == 1:
            self.conv = nn.Sequential(nn.Conv2d(hidden_dim, hidden_dim, 3, stride, pad, dilation=dilation, groups=hidden_dim, bias=False), nn.BatchNorm2d(hidden_dim), nn.ReLU6(inplace=True), nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False), nn.BatchNorm2d(oup))
        else:
            self.conv = nn.Sequential(nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False), nn.BatchNorm2d(hidden_dim), nn.ReLU6(inplace=True), nn.Conv2d(hidden_dim, hidden_dim, 3, stride, pad, dilation=dilation, groups=hidden_dim, bias=False), nn.BatchNorm2d(hidden_dim), nn.ReLU6(inplace=True), nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False), nn.BatchNorm2d(oup))

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class hourglass2D(nn.Module):

    def __init__(self, in_channels):
        super(hourglass2D, self).__init__()
        self.expanse_ratio = 2
        self.conv1 = MobileV2_Residual(in_channels, in_channels * 2, stride=2, expanse_ratio=self.expanse_ratio)
        self.conv2 = MobileV2_Residual(in_channels * 2, in_channels * 2, stride=1, expanse_ratio=self.expanse_ratio)
        self.conv3 = MobileV2_Residual(in_channels * 2, in_channels * 4, stride=2, expanse_ratio=self.expanse_ratio)
        self.conv4 = MobileV2_Residual(in_channels * 4, in_channels * 4, stride=1, expanse_ratio=self.expanse_ratio)
        self.conv5 = nn.Sequential(nn.ConvTranspose2d(in_channels * 4, in_channels * 2, 3, padding=1, output_padding=1, stride=2, bias=False), nn.BatchNorm2d(in_channels * 2))
        self.conv6 = nn.Sequential(nn.ConvTranspose2d(in_channels * 2, in_channels, 3, padding=1, output_padding=1, stride=2, bias=False), nn.BatchNorm2d(in_channels))
        self.redir1 = MobileV2_Residual(in_channels, in_channels, stride=1, expanse_ratio=self.expanse_ratio)
        self.redir2 = MobileV2_Residual(in_channels * 2, in_channels * 2, stride=1, expanse_ratio=self.expanse_ratio)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = F.relu(self.conv5(conv4) + self.redir2(conv2), inplace=True)
        conv6 = F.relu(self.conv6(conv5) + self.redir1(x), inplace=True)
        return conv6


def convbn(in_channels, out_channels, kernel_size, stride, pad, dilation):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=dilation if dilation > 1 else pad, dilation=dilation, bias=False), nn.BatchNorm2d(out_channels))


def disparity_regression(x, maxdisp):
    assert len(x.shape) == 4
    disp_values = torch.arange(0, maxdisp, dtype=x.dtype, device=x.device)
    disp_values = disp_values.view(1, maxdisp, 1, 1)
    return torch.sum(x * disp_values, 1, keepdim=False)


def convbn_dws(inp, oup, kernel_size, stride, pad, dilation, second_relu=True):
    if second_relu:
        return nn.Sequential(nn.Conv2d(inp, inp, kernel_size=kernel_size, stride=stride, padding=dilation if dilation > 1 else pad, dilation=dilation, groups=inp, bias=False), nn.BatchNorm2d(inp), nn.ReLU6(inplace=True), nn.Conv2d(inp, oup, 1, 1, 0, bias=False), nn.BatchNorm2d(oup), nn.ReLU6(inplace=False))
    else:
        return nn.Sequential(nn.Conv2d(inp, inp, kernel_size=kernel_size, stride=stride, padding=dilation if dilation > 1 else pad, dilation=dilation, groups=inp, bias=False), nn.BatchNorm2d(inp), nn.ReLU6(inplace=True), nn.Conv2d(inp, oup, 1, 1, 0, bias=False), nn.BatchNorm2d(oup))


class MobileV1_Residual(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride, downsample, pad, dilation):
        super(MobileV1_Residual, self).__init__()
        self.stride = stride
        self.downsample = downsample
        self.conv1 = convbn_dws(inplanes, planes, 3, stride, pad, dilation)
        self.conv2 = convbn_dws(planes, planes, 3, 1, pad, dilation, second_relu=False)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample is not None:
            x = self.downsample(x)
        out += x
        return out


class feature_extraction(nn.Module):

    def __init__(self, add_relus=False):
        super(feature_extraction, self).__init__()
        self.expanse_ratio = 3
        self.inplanes = 32
        if add_relus:
            self.firstconv = nn.Sequential(MobileV2_Residual(3, 32, 2, self.expanse_ratio), nn.ReLU(inplace=True), MobileV2_Residual(32, 32, 1, self.expanse_ratio), nn.ReLU(inplace=True), MobileV2_Residual(32, 32, 1, self.expanse_ratio), nn.ReLU(inplace=True))
        else:
            self.firstconv = nn.Sequential(MobileV2_Residual(3, 32, 2, self.expanse_ratio), MobileV2_Residual(32, 32, 1, self.expanse_ratio), MobileV2_Residual(32, 32, 1, self.expanse_ratio))
        self.layer1 = self._make_layer(MobileV1_Residual, 32, 3, 1, 1, 1)
        self.layer2 = self._make_layer(MobileV1_Residual, 64, 16, 2, 1, 1)
        self.layer3 = self._make_layer(MobileV1_Residual, 128, 3, 1, 1, 1)
        self.layer4 = self._make_layer(MobileV1_Residual, 128, 3, 1, 1, 2)

    def _make_layer(self, block, planes, blocks, stride, pad, dilation):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes))
        layers = [block(self.inplanes, planes, stride, downsample, pad, dilation)]
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, 1, None, pad, dilation))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.firstconv(x)
        x = self.layer1(x)
        l2 = self.layer2(x)
        l3 = self.layer3(l2)
        l4 = self.layer4(l3)
        feature_volume = torch.cat((l2, l3, l4), dim=1)
        return feature_volume


def interweave_tensors(refimg_fea, targetimg_fea):
    B, C, H, W = refimg_fea.shape
    interwoven_features = refimg_fea.new_zeros([B, 2 * C, H, W])
    interwoven_features[:, ::2, :, :] = refimg_fea
    interwoven_features[:, 1::2, :, :] = targetimg_fea
    interwoven_features = interwoven_features.contiguous()
    return interwoven_features


class MSNet2D(nn.Module):

    def __init__(self, maxdisp):
        super(MSNet2D, self).__init__()
        self.maxdisp = maxdisp
        self.num_groups = 1
        self.volume_size = 48
        self.hg_size = 48
        self.dres_expanse_ratio = 3
        self.feature_extraction = feature_extraction(add_relus=True)
        self.preconv11 = nn.Sequential(convbn(320, 256, 1, 1, 0, 1), nn.ReLU(inplace=True), convbn(256, 128, 1, 1, 0, 1), nn.ReLU(inplace=True), convbn(128, 64, 1, 1, 0, 1), nn.ReLU(inplace=True), nn.Conv2d(64, 32, 1, 1, 0, 1))
        self.conv3d = nn.Sequential(nn.Conv3d(1, 16, kernel_size=(8, 3, 3), stride=[8, 1, 1], padding=[0, 1, 1]), nn.BatchNorm3d(16), nn.ReLU(), nn.Conv3d(16, 32, kernel_size=(4, 3, 3), stride=[4, 1, 1], padding=[0, 1, 1]), nn.BatchNorm3d(32), nn.ReLU(), nn.Conv3d(32, 16, kernel_size=(2, 3, 3), stride=[2, 1, 1], padding=[0, 1, 1]), nn.BatchNorm3d(16), nn.ReLU())
        self.volume11 = nn.Sequential(convbn(16, 1, 1, 1, 0, 1), nn.ReLU(inplace=True))
        self.dres0 = nn.Sequential(MobileV2_Residual(self.volume_size, self.hg_size, 1, self.dres_expanse_ratio), nn.ReLU(inplace=True), MobileV2_Residual(self.hg_size, self.hg_size, 1, self.dres_expanse_ratio), nn.ReLU(inplace=True))
        self.dres1 = nn.Sequential(MobileV2_Residual(self.hg_size, self.hg_size, 1, self.dres_expanse_ratio), nn.ReLU(inplace=True), MobileV2_Residual(self.hg_size, self.hg_size, 1, self.dres_expanse_ratio))
        self.encoder_decoder1 = hourglass2D(self.hg_size)
        self.encoder_decoder2 = hourglass2D(self.hg_size)
        self.encoder_decoder3 = hourglass2D(self.hg_size)
        self.classif0 = nn.Sequential(convbn(self.hg_size, self.hg_size, 3, 1, 1, 1), nn.ReLU(inplace=True), nn.Conv2d(self.hg_size, self.hg_size, kernel_size=3, padding=1, stride=1, bias=False, dilation=1))
        self.classif1 = nn.Sequential(convbn(self.hg_size, self.hg_size, 3, 1, 1, 1), nn.ReLU(inplace=True), nn.Conv2d(self.hg_size, self.hg_size, kernel_size=3, padding=1, stride=1, bias=False, dilation=1))
        self.classif2 = nn.Sequential(convbn(self.hg_size, self.hg_size, 3, 1, 1, 1), nn.ReLU(inplace=True), nn.Conv2d(self.hg_size, self.hg_size, kernel_size=3, padding=1, stride=1, bias=False, dilation=1))
        self.classif3 = nn.Sequential(convbn(self.hg_size, self.hg_size, 3, 1, 1, 1), nn.ReLU(inplace=True), nn.Conv2d(self.hg_size, self.hg_size, kernel_size=3, padding=1, stride=1, bias=False, dilation=1))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, L, R):
        features_L = self.feature_extraction(L)
        features_R = self.feature_extraction(R)
        featL = self.preconv11(features_L)
        featR = self.preconv11(features_R)
        B, C, H, W = featL.shape
        volume = featL.new_zeros([B, self.num_groups, self.volume_size, H, W])
        for i in range(self.volume_size):
            if i > 0:
                x = interweave_tensors(featL[:, :, :, i:], featR[:, :, :, :-i])
                x = torch.unsqueeze(x, 1)
                x = self.conv3d(x)
                x = torch.squeeze(x, 2)
                x = self.volume11(x)
                volume[:, :, i, :, i:] = x
            else:
                x = interweave_tensors(featL, featR)
                x = torch.unsqueeze(x, 1)
                x = self.conv3d(x)
                x = torch.squeeze(x, 2)
                x = self.volume11(x)
                volume[:, :, i, :, :] = x
        volume = volume.contiguous()
        volume = torch.squeeze(volume, 1)
        cost0 = self.dres0(volume)
        cost0 = self.dres1(cost0) + cost0
        out1 = self.encoder_decoder1(cost0)
        out2 = self.encoder_decoder2(out1)
        out3 = self.encoder_decoder3(out2)
        if self.training:
            cost0 = self.classif0(cost0)
            cost1 = self.classif1(out1)
            cost2 = self.classif2(out2)
            cost3 = self.classif3(out3)
            cost0 = torch.unsqueeze(cost0, 1)
            cost0 = F.interpolate(cost0, [self.maxdisp, L.size()[2], L.size()[3]], mode='trilinear')
            cost0 = torch.squeeze(cost0, 1)
            pred0 = F.softmax(cost0, dim=1)
            pred0 = disparity_regression(pred0, self.maxdisp)
            cost1 = torch.unsqueeze(cost1, 1)
            cost1 = F.interpolate(cost1, [self.maxdisp, L.size()[2], L.size()[3]], mode='trilinear')
            cost1 = torch.squeeze(cost1, 1)
            pred1 = F.softmax(cost1, dim=1)
            pred1 = disparity_regression(pred1, self.maxdisp)
            cost2 = torch.unsqueeze(cost2, 1)
            cost2 = F.interpolate(cost2, [self.maxdisp, L.size()[2], L.size()[3]], mode='trilinear')
            cost2 = torch.squeeze(cost2, 1)
            pred2 = F.softmax(cost2, dim=1)
            pred2 = disparity_regression(pred2, self.maxdisp)
            cost3 = torch.unsqueeze(cost3, 1)
            cost3 = F.interpolate(cost3, [self.maxdisp, L.size()[2], L.size()[3]], mode='trilinear')
            cost3 = torch.squeeze(cost3, 1)
            pred3 = F.softmax(cost3, dim=1)
            pred3 = disparity_regression(pred3, self.maxdisp)
            return [pred0, pred1, pred2, pred3]
        else:
            cost3 = self.classif3(out3)
            cost3 = torch.unsqueeze(cost3, 1)
            cost3 = F.interpolate(cost3, [self.maxdisp, L.size()[2], L.size()[3]], mode='trilinear')
            cost3 = torch.squeeze(cost3, 1)
            pred3 = F.softmax(cost3, dim=1)
            pred3 = disparity_regression(pred3, self.maxdisp)
            return [pred3]


class MobileV2_Residual_3D(nn.Module):

    def __init__(self, inp, oup, stride, expanse_ratio):
        super(MobileV2_Residual_3D, self).__init__()
        self.stride = stride
        hidden_dim = round(inp * expanse_ratio)
        self.use_res_connect = self.stride == (1, 1, 1) and inp == oup
        if expanse_ratio == 1:
            self.conv = nn.Sequential(nn.Conv3d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False), nn.BatchNorm3d(hidden_dim), nn.ReLU6(inplace=True), nn.Conv3d(hidden_dim, oup, 1, 1, 0, bias=False), nn.BatchNorm3d(oup))
        else:
            self.conv = nn.Sequential(nn.Conv3d(inp, hidden_dim, 1, 1, 0, bias=False), nn.BatchNorm3d(hidden_dim), nn.ReLU6(inplace=True), nn.Conv3d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False), nn.BatchNorm3d(hidden_dim), nn.ReLU6(inplace=True), nn.Conv3d(hidden_dim, oup, 1, 1, 0, bias=False), nn.BatchNorm3d(oup))

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class hourglass3D(nn.Module):

    def __init__(self, in_channels):
        super(hourglass3D, self).__init__()
        self.expanse_ratio = 2
        self.conv1 = MobileV2_Residual_3D(in_channels, in_channels * 2, 2, self.expanse_ratio)
        self.conv2 = MobileV2_Residual_3D(in_channels * 2, in_channels * 2, 1, self.expanse_ratio)
        self.conv3 = MobileV2_Residual_3D(in_channels * 2, in_channels * 4, 2, self.expanse_ratio)
        self.conv4 = MobileV2_Residual_3D(in_channels * 4, in_channels * 4, 1, self.expanse_ratio)
        self.conv5 = nn.Sequential(nn.ConvTranspose3d(in_channels * 4, in_channels * 2, 3, padding=1, output_padding=1, stride=2, bias=False), nn.BatchNorm3d(in_channels * 2))
        self.conv6 = nn.Sequential(nn.ConvTranspose3d(in_channels * 2, in_channels, 3, padding=1, output_padding=1, stride=2, bias=False), nn.BatchNorm3d(in_channels))
        self.redir1 = MobileV2_Residual_3D(in_channels, in_channels, 1, self.expanse_ratio)
        self.redir2 = MobileV2_Residual_3D(in_channels * 2, in_channels * 2, 1, self.expanse_ratio)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = F.relu(self.conv5(conv4) + self.redir2(conv2), inplace=True)
        conv6 = F.relu(self.conv6(conv5) + self.redir1(x), inplace=True)
        return conv6


def groupwise_correlation(fea1, fea2, num_groups):
    B, C, H, W = fea1.shape
    assert C % num_groups == 0
    channels_per_group = C // num_groups
    cost = (fea1 * fea2).view([B, num_groups, channels_per_group, H, W]).mean(dim=2)
    assert cost.shape == (B, num_groups, H, W)
    return cost


def build_gwc_volume(refimg_fea, targetimg_fea, maxdisp, num_groups):
    B, C, H, W = refimg_fea.shape
    volume = refimg_fea.new_zeros([B, num_groups, maxdisp, H, W])
    for i in range(maxdisp):
        if i > 0:
            volume[:, :, i, :, i:] = groupwise_correlation(refimg_fea[:, :, :, i:], targetimg_fea[:, :, :, :-i], num_groups)
        else:
            volume[:, :, i, :, :] = groupwise_correlation(refimg_fea, targetimg_fea, num_groups)
    volume = volume.contiguous()
    return volume


def convbn_3d(in_channels, out_channels, kernel_size, stride, pad):
    return nn.Sequential(nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=pad, bias=False), nn.BatchNorm3d(out_channels))


class MSNet3D(nn.Module):

    def __init__(self, maxdisp):
        super(MSNet3D, self).__init__()
        self.maxdisp = maxdisp
        self.hourglass_size = 32
        self.dres_expanse_ratio = 3
        self.num_groups = 40
        self.feature_extraction = feature_extraction()
        self.dres0 = nn.Sequential(MobileV2_Residual_3D(self.num_groups, self.hourglass_size, 1, self.dres_expanse_ratio), MobileV2_Residual_3D(self.hourglass_size, self.hourglass_size, 1, self.dres_expanse_ratio))
        self.dres1 = nn.Sequential(MobileV2_Residual_3D(self.hourglass_size, self.hourglass_size, 1, self.dres_expanse_ratio), MobileV2_Residual_3D(self.hourglass_size, self.hourglass_size, 1, self.dres_expanse_ratio))
        self.encoder_decoder1 = hourglass3D(self.hourglass_size)
        self.encoder_decoder2 = hourglass3D(self.hourglass_size)
        self.encoder_decoder3 = hourglass3D(self.hourglass_size)
        self.classif0 = nn.Sequential(convbn_3d(self.hourglass_size, self.hourglass_size, 3, 1, 1), nn.ReLU(inplace=True), nn.Conv3d(self.hourglass_size, 1, kernel_size=3, padding=1, stride=1, bias=False, dilation=1))
        self.classif1 = nn.Sequential(convbn_3d(self.hourglass_size, self.hourglass_size, 3, 1, 1), nn.ReLU(inplace=True), nn.Conv3d(self.hourglass_size, 1, kernel_size=3, padding=1, stride=1, bias=False, dilation=1))
        self.classif2 = nn.Sequential(convbn_3d(self.hourglass_size, self.hourglass_size, 3, 1, 1), nn.ReLU(inplace=True), nn.Conv3d(self.hourglass_size, 1, kernel_size=3, padding=1, stride=1, bias=False, dilation=1))
        self.classif3 = nn.Sequential(convbn_3d(self.hourglass_size, self.hourglass_size, 3, 1, 1), nn.ReLU(inplace=True), nn.Conv3d(self.hourglass_size, 1, kernel_size=3, padding=1, stride=1, bias=False, dilation=1))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, L, R):
        features_left = self.feature_extraction(L)
        features_right = self.feature_extraction(R)
        volume = build_gwc_volume(features_left, features_right, self.maxdisp // 4, self.num_groups)
        cost0 = self.dres0(volume)
        cost0 = self.dres1(cost0) + cost0
        out1 = self.encoder_decoder1(cost0)
        out2 = self.encoder_decoder2(out1)
        out3 = self.encoder_decoder3(out2)
        if self.training:
            cost0 = self.classif0(cost0)
            cost1 = self.classif1(out1)
            cost2 = self.classif2(out2)
            cost3 = self.classif3(out3)
            cost0 = F.interpolate(cost0, [self.maxdisp, L.size()[2], L.size()[3]], mode='trilinear')
            cost0 = torch.squeeze(cost0, 1)
            pred0 = F.softmax(cost0, dim=1)
            pred0 = disparity_regression(pred0, self.maxdisp)
            cost1 = F.interpolate(cost1, [self.maxdisp, L.size()[2], L.size()[3]], mode='trilinear')
            cost1 = torch.squeeze(cost1, 1)
            pred1 = F.softmax(cost1, dim=1)
            pred1 = disparity_regression(pred1, self.maxdisp)
            cost2 = F.interpolate(cost2, [self.maxdisp, L.size()[2], L.size()[3]], mode='trilinear')
            cost2 = torch.squeeze(cost2, 1)
            pred2 = F.softmax(cost2, dim=1)
            pred2 = disparity_regression(pred2, self.maxdisp)
            cost3 = F.interpolate(cost3, [self.maxdisp, L.size()[2], L.size()[3]], mode='trilinear')
            cost3 = torch.squeeze(cost3, 1)
            pred3 = F.softmax(cost3, dim=1)
            pred3 = disparity_regression(pred3, self.maxdisp)
            return [pred0, pred1, pred2, pred3]
        else:
            cost3 = self.classif3(out3)
            cost3 = F.interpolate(cost3, [self.maxdisp, L.size()[2], L.size()[3]], mode='trilinear')
            cost3 = torch.squeeze(cost3, 1)
            pred3 = F.softmax(cost3, dim=1)
            pred3 = disparity_regression(pred3, self.maxdisp)
            return [pred3]


def CE_ssc_loss(pred, target, class_weights):
    criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=255, reduction='none')
    loss = criterion(pred, target.long())
    loss_valid = loss[target != 255]
    loss_valid_mean = torch.mean(loss_valid)
    return loss_valid_mean


class Header(nn.Module):

    def __init__(self, class_num, norm_layer, feature):
        super(Header, self).__init__()
        self.feature = feature
        self.class_num = class_num
        self.mlp_head = nn.Sequential(nn.LayerNorm(self.feature), nn.Linear(self.feature, self.class_num))
        self.up_scale_2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)

    def forward(self, input_dict):
        res = {}
        x3d_l1 = input_dict['x3d']
        x3d_up_l1 = self.up_scale_2(x3d_l1)
        _, feat_dim, w, l, h = x3d_up_l1.shape
        x3d_up_l1 = x3d_up_l1.squeeze().permute(1, 2, 3, 0).reshape(-1, feat_dim)
        ssc_logit_full = self.mlp_head(x3d_up_l1)
        res['ssc_logit'] = ssc_logit_full.reshape(w, l, h, self.class_num).permute(3, 0, 1, 2).unsqueeze(0)
        return res


def geo_scal_loss(pred, ssc_target):
    pred = F.softmax(pred, dim=1)
    empty_probs = pred[:, 0, :, :, :]
    nonempty_probs = 1 - empty_probs
    mask = ssc_target != 255
    nonempty_target = ssc_target != 0
    nonempty_target = nonempty_target[mask].float()
    nonempty_probs = nonempty_probs[mask]
    empty_probs = empty_probs[mask]
    intersection = (nonempty_target * nonempty_probs).sum()
    precision = intersection / nonempty_probs.sum()
    recall = intersection / nonempty_target.sum()
    spec = ((1 - nonempty_target) * empty_probs).sum() / (1 - nonempty_target).sum()
    return F.binary_cross_entropy(precision, torch.ones_like(precision)) + F.binary_cross_entropy(recall, torch.ones_like(recall)) + F.binary_cross_entropy(spec, torch.ones_like(spec))


def sem_scal_loss(pred, ssc_target):
    pred = F.softmax(pred, dim=1)
    loss = 0
    count = 0
    mask = ssc_target != 255
    n_classes = pred.shape[1]
    for i in range(0, n_classes):
        p = pred[:, i, :, :, :]
        target_ori = ssc_target
        p = p[mask]
        target = ssc_target[mask]
        completion_target = torch.ones_like(target)
        completion_target[target != i] = 0
        completion_target_ori = torch.ones_like(target_ori).float()
        completion_target_ori[target_ori != i] = 0
        if torch.sum(completion_target) > 0:
            count += 1.0
            nominator = torch.sum(p * completion_target)
            loss_class = 0
            if torch.sum(p) > 0:
                precision = nominator / torch.sum(p)
                loss_precision = F.binary_cross_entropy(precision, torch.ones_like(precision))
                loss_class += loss_precision
            if torch.sum(completion_target) > 0:
                recall = nominator / torch.sum(completion_target)
                loss_recall = F.binary_cross_entropy(recall, torch.ones_like(recall))
                loss_class += loss_recall
            if torch.sum(1 - completion_target) > 0:
                specificity = torch.sum((1 - p) * (1 - completion_target)) / torch.sum(1 - completion_target)
                loss_specificity = F.binary_cross_entropy(specificity, torch.ones_like(specificity))
                loss_class += loss_specificity
            loss += loss_class
    return loss / count


class VoxFormerHead(nn.Module):

    def __init__(self, *args, bev_h, bev_w, bev_z, cross_transformer, self_transformer, positional_encoding, embed_dims, CE_ssc_loss=True, geo_scal_loss=True, sem_scal_loss=True, save_flag=False, **kwargs):
        super().__init__()
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.bev_z = bev_z
        self.real_w = 51.2
        self.real_h = 51.2
        self.n_classes = 20
        self.embed_dims = embed_dims
        self.bev_embed = nn.Embedding(self.bev_h * self.bev_w * self.bev_z, self.embed_dims)
        self.mask_embed = nn.Embedding(1, self.embed_dims)
        self.positional_encoding = build_positional_encoding(positional_encoding)
        self.cross_transformer = build_transformer(cross_transformer)
        self.self_transformer = build_transformer(self_transformer)
        self.header = Header(self.n_classes, nn.BatchNorm3d, feature=self.embed_dims)
        self.class_names = ['empty', 'car', 'bicycle', 'motorcycle', 'truck', 'other-vehicle', 'person', 'bicyclist', 'motorcyclist', 'road', 'parking', 'sidewalk', 'other-ground', 'building', 'fence', 'vegetation', 'trunk', 'terrain', 'pole', 'traffic-sign']
        self.class_weights = torch.from_numpy(np.array([0.446, 0.603, 0.852, 0.856, 0.747, 0.734, 0.801, 0.796, 0.818, 0.557, 0.653, 0.568, 0.683, 0.56, 0.603, 0.53, 0.688, 0.574, 0.716, 0.786]))
        self.CE_ssc_loss = CE_ssc_loss
        self.sem_scal_loss = sem_scal_loss
        self.geo_scal_loss = geo_scal_loss
        self.save_flag = save_flag

    def forward(self, mlvl_feats, img_metas, target):
        """Forward function.
        Args:
            mlvl_feats (tuple[Tensor]): Features from the upstream
                network, each is a 5D-tensor with shape
                (B, N, C, H, W).
            img_metas: Meta information such as camera intrinsics.
            target: Semantic completion ground truth. 
        Returns:
            ssc_logit (Tensor): Outputs from the segmentation head.
        """
        bs, num_cam, _, _, _ = mlvl_feats[0].shape
        dtype = mlvl_feats[0].dtype
        bev_queries = self.bev_embed.weight
        bev_pos_cross_attn = self.positional_encoding(torch.zeros((bs, 512, 512), device=bev_queries.device).to(dtype))
        bev_pos_self_attn = self.positional_encoding(torch.zeros((bs, 512, 512), device=bev_queries.device).to(dtype))
        proposal = img_metas[0]['proposal'].reshape(self.bev_h, self.bev_w, self.bev_z)
        unmasked_idx = np.asarray(np.where(proposal.reshape(-1) > 0)).astype(np.int32)
        masked_idx = np.asarray(np.where(proposal.reshape(-1) == 0)).astype(np.int32)
        vox_coords, ref_3d = self.get_ref_3d()
        seed_feats = self.cross_transformer.get_vox_features(mlvl_feats, bev_queries, self.bev_h, self.bev_w, ref_3d=ref_3d, vox_coords=vox_coords, unmasked_idx=unmasked_idx, grid_length=(self.real_h / self.bev_h, self.real_w / self.bev_w), bev_pos=bev_pos_cross_attn, img_metas=img_metas, prev_bev=None)
        vox_feats = torch.empty((self.bev_h, self.bev_w, self.bev_z, self.embed_dims), device=bev_queries.device)
        vox_feats_flatten = vox_feats.reshape(-1, self.embed_dims)
        vox_feats_flatten[vox_coords[unmasked_idx[0], 3], :] = seed_feats[0]
        vox_feats_flatten[vox_coords[masked_idx[0], 3], :] = self.mask_embed.weight.view(1, self.embed_dims).expand(masked_idx.shape[1], self.embed_dims)
        vox_feats_diff = self.self_transformer.diffuse_vox_features(mlvl_feats, vox_feats_flatten, 512, 512, ref_3d=ref_3d, vox_coords=vox_coords, unmasked_idx=unmasked_idx, grid_length=(self.real_h / self.bev_h, self.real_w / self.bev_w), bev_pos=bev_pos_self_attn, img_metas=img_metas, prev_bev=None)
        vox_feats_diff = vox_feats_diff.reshape(self.bev_h, self.bev_w, self.bev_z, self.embed_dims)
        input_dict = {'x3d': vox_feats_diff.permute(3, 0, 1, 2).unsqueeze(0)}
        out = self.header(input_dict)
        return out

    def step(self, out_dict, target, img_metas, step_type):
        """Training/validation function.
        Args:
            out_dict (dict[Tensor]): Segmentation output.
            img_metas: Meta information such as camera intrinsics.
            target: Semantic completion ground truth. 
            step_type: Train or test.
        Returns:
            loss or predictions
        """
        ssc_pred = out_dict['ssc_logit']
        if step_type == 'train':
            loss_dict = dict()
            class_weight = self.class_weights.type_as(target)
            if self.CE_ssc_loss:
                loss_ssc = CE_ssc_loss(ssc_pred, target, class_weight)
                loss_dict['loss_ssc'] = loss_ssc
            if self.sem_scal_loss:
                loss_sem_scal = sem_scal_loss(ssc_pred, target)
                loss_dict['loss_sem_scal'] = loss_sem_scal
            if self.geo_scal_loss:
                loss_geo_scal = geo_scal_loss(ssc_pred, target)
                loss_dict['loss_geo_scal'] = loss_geo_scal
            return loss_dict
        elif step_type == 'val' or 'test':
            y_true = target.cpu().numpy()
            y_pred = ssc_pred.detach().cpu().numpy()
            y_pred = np.argmax(y_pred, axis=1)
            result = dict()
            result['y_pred'] = y_pred
            result['y_true'] = y_true
            if self.save_flag:
                self.save_pred(img_metas, y_pred)
            return result

    def training_step(self, out_dict, target, img_metas):
        """Training step.
        """
        return self.step(out_dict, target, img_metas, 'train')

    def validation_step(self, out_dict, target, img_metas):
        """Validation step.
        """
        return self.step(out_dict, target, img_metas, 'val')

    def get_ref_3d(self):
        """Get reference points in 3D.
        Args:
            self.real_h, self.bev_h
        Returns:
            vox_coords (Array): Voxel indices
            ref_3d (Array): 3D reference points
        """
        scene_size = 51.2, 51.2, 6.4
        vox_origin = np.array([0, -25.6, -2])
        voxel_size = self.real_h / self.bev_h
        vol_bnds = np.zeros((3, 2))
        vol_bnds[:, 0] = vox_origin
        vol_bnds[:, 1] = vox_origin + np.array(scene_size)
        vol_dim = np.ceil((vol_bnds[:, 1] - vol_bnds[:, 0]) / voxel_size).copy(order='C').astype(int)
        idx = np.array([range(vol_dim[0] * vol_dim[1] * vol_dim[2])])
        xv, yv, zv = np.meshgrid(range(vol_dim[0]), range(vol_dim[1]), range(vol_dim[2]), indexing='ij')
        vox_coords = np.concatenate([xv.reshape(1, -1), yv.reshape(1, -1), zv.reshape(1, -1), idx], axis=0).astype(int).T
        ref_3d = np.concatenate([(xv.reshape(1, -1) + 0.5) / self.bev_h, (yv.reshape(1, -1) + 0.5) / self.bev_w, (zv.reshape(1, -1) + 0.5) / self.bev_z], axis=0).astype(np.float64).T
        return vox_coords, ref_3d

    def save_pred(self, img_metas, y_pred):
        """Save predictions for evaluations and visualizations.

        learning_map_inv: inverse of previous map
        
        0: 0    # "unlabeled/ignored"  # 1: 10   # "car"        # 2: 11   # "bicycle"       # 3: 15   # "motorcycle"     # 4: 18   # "truck" 
        5: 20   # "other-vehicle"      # 6: 30   # "person"     # 7: 31   # "bicyclist"     # 8: 32   # "motorcyclist"   # 9: 40   # "road"   
        10: 44  # "parking"            # 11: 48  # "sidewalk"   # 12: 49  # "other-ground"  # 13: 50  # "building"       # 14: 51  # "fence"          
        15: 70  # "vegetation"         # 16: 71  # "trunk"      # 17: 72  # "terrain"       # 18: 80  # "pole"           # 19: 81  # "traffic-sign"
        """
        y_pred[y_pred == 10] = 44
        y_pred[y_pred == 11] = 48
        y_pred[y_pred == 12] = 49
        y_pred[y_pred == 13] = 50
        y_pred[y_pred == 14] = 51
        y_pred[y_pred == 15] = 70
        y_pred[y_pred == 16] = 71
        y_pred[y_pred == 17] = 72
        y_pred[y_pred == 18] = 80
        y_pred[y_pred == 19] = 81
        y_pred[y_pred == 1] = 10
        y_pred[y_pred == 2] = 11
        y_pred[y_pred == 3] = 15
        y_pred[y_pred == 4] = 18
        y_pred[y_pred == 5] = 20
        y_pred[y_pred == 6] = 30
        y_pred[y_pred == 7] = 31
        y_pred[y_pred == 8] = 32
        y_pred[y_pred == 9] = 40
        pred_folder = os.path.join('./voxformer', 'sequences', img_metas[0]['sequence_id'], 'predictions')
        if not os.path.exists(pred_folder):
            os.makedirs(pred_folder)
        y_pred_bin = y_pred.astype(np.uint16)
        y_pred_bin.tofile(os.path.join(pred_folder, img_metas[0]['frame_id'] + '.label'))


class SegmentationHead(nn.Module):
    """
  3D Segmentation heads to retrieve semantic segmentation at each scale.
  Formed by Dim expansion, Conv3D, ASPP block, Conv3D.
  """

    def __init__(self, inplanes, planes, nbr_classes, dilations_conv_list):
        super().__init__()
        self.conv0 = nn.Conv3d(inplanes, planes, kernel_size=3, padding=1, stride=1)
        self.conv_list = dilations_conv_list
        self.conv1 = nn.ModuleList([nn.Conv3d(planes, planes, kernel_size=3, padding=dil, dilation=dil, bias=False) for dil in dilations_conv_list])
        self.bn1 = nn.ModuleList([nn.BatchNorm3d(planes) for dil in dilations_conv_list])
        self.conv2 = nn.ModuleList([nn.Conv3d(planes, planes, kernel_size=3, padding=dil, dilation=dil, bias=False) for dil in dilations_conv_list])
        self.bn2 = nn.ModuleList([nn.BatchNorm3d(planes) for dil in dilations_conv_list])
        self.relu = nn.ReLU(inplace=True)
        self.conv_classes = nn.Conv3d(planes, nbr_classes, kernel_size=3, padding=1, stride=1)

    def forward(self, x_in):
        x_in = x_in[:, None, :, :, :]
        x_in = self.relu(self.conv0(x_in))
        y = self.bn2[0](self.conv2[0](self.relu(self.bn1[0](self.conv1[0](x_in)))))
        for i in range(1, len(self.conv_list)):
            y += self.bn2[i](self.conv2[i](self.relu(self.bn1[i](self.conv1[i](x_in)))))
        x_in = self.relu(y + x_in)
        x_in = self.conv_classes(x_in)
        return x_in


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (MSNet3D,
     lambda: ([], {'maxdisp': 4}),
     lambda: ([torch.rand([4, 3, 64, 64]), torch.rand([4, 3, 64, 64])], {})),
    (MobileV2_Residual,
     lambda: ([], {'inp': 4, 'oup': 4, 'stride': 1, 'expanse_ratio': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (MobileV2_Residual_3D,
     lambda: ([], {'inp': 4, 'oup': 4, 'stride': 1, 'expanse_ratio': 4}),
     lambda: ([torch.rand([4, 4, 4, 4, 4])], {})),
    (feature_extraction,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {})),
    (hourglass2D,
     lambda: ([], {'in_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (hourglass3D,
     lambda: ([], {'in_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4, 4])], {})),
]

