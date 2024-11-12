
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


from torch.utils.data import Dataset


import torch


import numpy as np


import torchvision.transforms as tt


import random


import numbers


import logging


import time


from torch import nn


import torch.backends.cudnn as cudnn


from torch.autograd import Variable


import torch.nn as nn


import torch.nn.functional as F


from torch.nn import init


import torch.nn.init as init


import scipy.misc as misc


import math


from sklearn import metrics


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.orthogonal_(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.orthogonal_(m.weight.data, gain=1)
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_xavier(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_normal_(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.xavier_normal_(m.weight.data, gain=1)
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def init_weights(net, init_type='normal'):
    if init_type == 'normal':
        net.apply(weights_init_normal)
    elif init_type == 'xavier':
        net.apply(weights_init_xavier)
    elif init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)


class unetConv2(nn.Module):

    def __init__(self, in_size, out_size, is_batchnorm, n=2, ks=3, stride=1, padding=1):
        super(unetConv2, self).__init__()
        self.n = n
        self.ks = ks
        self.stride = stride
        self.padding = padding
        s = stride
        p = padding
        if is_batchnorm:
            for i in range(1, n + 1):
                conv = nn.Sequential(nn.Conv2d(in_size, out_size, ks, s, p), nn.BatchNorm2d(out_size), nn.ReLU(inplace=True))
                setattr(self, 'conv%d' % i, conv)
                in_size = out_size
        else:
            for i in range(1, n + 1):
                conv = nn.Sequential(nn.Conv2d(in_size, out_size, ks, s, p), nn.ReLU(inplace=True))
                setattr(self, 'conv%d' % i, conv)
                in_size = out_size
        for m in self.children():
            init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        x = inputs
        for i in range(1, self.n + 1):
            conv = getattr(self, 'conv%d' % i)
            x = conv(x)
        return x


class unetUp(nn.Module):

    def __init__(self, in_size, out_size, is_deconv, n_concat=2):
        super(unetUp, self).__init__()
        self.conv = unetConv2(in_size + (n_concat - 2) * out_size, out_size, False)
        if is_deconv:
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=4, stride=2, padding=1)
        else:
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        for m in self.children():
            if m.__class__.__name__.find('unetConv2') != -1:
                continue
            init_weights(m, init_type='kaiming')

    def forward(self, inputs0, *input):
        outputs0 = self.up(inputs0)
        for i in range(len(input)):
            outputs0 = torch.cat([outputs0, input[i]], 1)
        return self.conv(outputs0)


class UNet(nn.Module):

    def __init__(self, in_channels=3, n_classes=4, feature_scale=4, is_deconv=True, is_batchnorm=True):
        super(UNet, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale
        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]
        self.conv1 = unetConv2(self.in_channels, filters[0], self.is_batchnorm)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = unetConv2(filters[0], filters[1], self.is_batchnorm)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.conv3 = unetConv2(filters[1], filters[2], self.is_batchnorm)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)
        self.conv4 = unetConv2(filters[2], filters[3], self.is_batchnorm)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)
        self.center = unetConv2(filters[3], filters[4], self.is_batchnorm)
        self.cls = nn.Sequential(nn.Dropout(p=0.5), nn.Conv2d(256, 3, 1), nn.AdaptiveMaxPool2d(1), nn.Sigmoid())
        self.up_concat4 = unetUp(filters[4], filters[3], self.is_deconv)
        self.up_concat3 = unetUp(filters[3], filters[2], self.is_deconv)
        self.up_concat2 = unetUp(filters[2], filters[1], self.is_deconv)
        self.up_concat1 = unetUp(filters[1], filters[0], self.is_deconv)
        self.final_1 = nn.Conv2d(filters[0], n_classes, 1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        conv1 = self.conv1(inputs)
        maxpool1 = self.maxpool1(conv1)
        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)
        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)
        conv4 = self.conv4(maxpool3)
        maxpool4 = self.maxpool4(conv4)
        center = self.center(maxpool4)
        cls_branch = self.cls(center).squeeze()
        up4 = self.up_concat4(center, conv4)
        up3 = self.up_concat3(up4, conv3)
        up2 = self.up_concat2(up3, conv2)
        up1 = self.up_concat1(up2, conv1)
        final_1 = self.final_1(up1)
        return F.log_softmax(final_1, dim=1), cls_branch


class UNet_Nested(nn.Module):

    def __init__(self, in_channels=3, n_classes=4, feature_scale=4, is_deconv=True, is_batchnorm=True, is_ds=True):
        super(UNet_Nested, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.is_ds = is_ds
        self.feature_scale = feature_scale
        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]
        self.conv00 = unetConv2(self.in_channels, filters[0], self.is_batchnorm)
        self.maxpool0 = nn.MaxPool2d(kernel_size=2)
        self.conv10 = unetConv2(filters[0], filters[1], self.is_batchnorm)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.conv20 = unetConv2(filters[1], filters[2], self.is_batchnorm)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.conv30 = unetConv2(filters[2], filters[3], self.is_batchnorm)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)
        self.conv40 = unetConv2(filters[3], filters[4], self.is_batchnorm)
        self.cls = nn.Sequential(nn.Dropout(p=0.5), nn.Conv2d(256, 3, 1), nn.AdaptiveMaxPool2d(1), nn.Sigmoid())
        self.up_concat01 = unetUp(filters[1], filters[0], self.is_deconv)
        self.up_concat11 = unetUp(filters[2], filters[1], self.is_deconv)
        self.up_concat21 = unetUp(filters[3], filters[2], self.is_deconv)
        self.up_concat31 = unetUp(filters[4], filters[3], self.is_deconv)
        self.up_concat02 = unetUp(filters[1], filters[0], self.is_deconv, 3)
        self.up_concat12 = unetUp(filters[2], filters[1], self.is_deconv, 3)
        self.up_concat22 = unetUp(filters[3], filters[2], self.is_deconv, 3)
        self.up_concat03 = unetUp(filters[1], filters[0], self.is_deconv, 4)
        self.up_concat13 = unetUp(filters[2], filters[1], self.is_deconv, 4)
        self.up_concat04 = unetUp(filters[1], filters[0], self.is_deconv, 5)
        self.final_1 = nn.Conv2d(filters[0], n_classes, 1)
        self.final_2 = nn.Conv2d(filters[0], n_classes, 1)
        self.final_3 = nn.Conv2d(filters[0], n_classes, 1)
        self.final_4 = nn.Conv2d(filters[0], n_classes, 1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        X_00 = self.conv00(inputs)
        maxpool0 = self.maxpool0(X_00)
        X_10 = self.conv10(maxpool0)
        maxpool1 = self.maxpool1(X_10)
        X_20 = self.conv20(maxpool1)
        maxpool2 = self.maxpool2(X_20)
        X_30 = self.conv30(maxpool2)
        maxpool3 = self.maxpool3(X_30)
        X_40 = self.conv40(maxpool3)
        cls_branch = self.cls(X_40).squeeze()
        X_01 = self.up_concat01(X_10, X_00)
        X_11 = self.up_concat11(X_20, X_10)
        X_21 = self.up_concat21(X_30, X_20)
        X_31 = self.up_concat31(X_40, X_30)
        X_02 = self.up_concat02(X_11, X_00, X_01)
        X_12 = self.up_concat12(X_21, X_10, X_11)
        X_22 = self.up_concat22(X_31, X_20, X_21)
        X_03 = self.up_concat03(X_12, X_00, X_01, X_02)
        X_13 = self.up_concat13(X_22, X_10, X_11, X_12)
        X_04 = self.up_concat04(X_13, X_00, X_01, X_02, X_03)
        final_1 = self.final_1(X_01)
        final_2 = self.final_2(X_02)
        final_3 = self.final_3(X_03)
        final_4 = self.final_4(X_04)
        final = (final_1 + final_2 + final_3 + final_4) / 4
        if self.is_ds:
            return F.log_softmax(final, dim=1), cls_branch
        else:
            return F.log_softmax(final_4), cls_branch


class unetConv2_dilation(nn.Module):

    def __init__(self, in_size, out_size, is_batchnorm=True, n=4, ks=3, stride=1):
        super(unetConv2_dilation, self).__init__()
        self.n = n
        self.ks = ks
        self.stride = stride
        s = stride
        if is_batchnorm:
            for i in range(1, n + 1):
                conv = nn.Sequential(nn.Conv2d(in_size, out_size, ks, s, 2 ** (i - 1), 2 ** (i - 1)), nn.BatchNorm2d(out_size), nn.ReLU(inplace=True))
                setattr(self, 'conv%d' % i, conv)
                in_size = out_size
        else:
            for i in range(1, n + 1):
                conv = nn.Sequential(nn.Conv2d(in_size, out_size, ks, s, p, r), nn.ReLU(inplace=True))
                setattr(self, 'conv%d' % i, conv)
                in_size = out_size
        for m in self.children():
            init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        output = inputs
        x_0 = inputs
        conv = getattr(self, 'conv1')
        x_1 = conv(x_0)
        conv = getattr(self, 'conv2')
        x_2 = conv(x_1)
        conv = getattr(self, 'conv3')
        x_3 = conv(x_2)
        conv = getattr(self, 'conv4')
        x_4 = conv(x_3)
        return x_0 + x_1 + x_2 + x_3 + x_4


class UNet_Nested_dilated(nn.Module):

    def __init__(self, in_channels=3, n_classes=4, feature_scale=4, is_deconv=True, is_batchnorm=True, is_ds=True):
        super(UNet_Nested_dilated, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.is_ds = is_ds
        self.feature_scale = feature_scale
        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]
        self.conv00 = unetConv2(self.in_channels, filters[0], self.is_batchnorm)
        self.maxpool0 = nn.MaxPool2d(kernel_size=2)
        self.conv10 = unetConv2(filters[0], filters[1], self.is_batchnorm)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.conv20 = unetConv2(filters[1], filters[2], self.is_batchnorm)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.conv30 = unetConv2(filters[2], filters[3], self.is_batchnorm)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)
        self.conv40 = unetConv2(filters[3], filters[4], self.is_batchnorm)
        self.dilated = unetConv2_dilation(filters[4], filters[4], self.is_batchnorm)
        self.cls = nn.Sequential(nn.Dropout(p=0.5), nn.Conv2d(256, 3, 1), nn.AdaptiveMaxPool2d(1), nn.Sigmoid())
        self.up_concat01 = unetUp(filters[1], filters[0], self.is_deconv)
        self.up_concat11 = unetUp(filters[2], filters[1], self.is_deconv)
        self.up_concat21 = unetUp(filters[3], filters[2], self.is_deconv)
        self.up_concat31 = unetUp(filters[4], filters[3], self.is_deconv)
        self.up_concat02 = unetUp(filters[1], filters[0], self.is_deconv, 3)
        self.up_concat12 = unetUp(filters[2], filters[1], self.is_deconv, 3)
        self.up_concat22 = unetUp(filters[3], filters[2], self.is_deconv, 3)
        self.up_concat03 = unetUp(filters[1], filters[0], self.is_deconv, 4)
        self.up_concat13 = unetUp(filters[2], filters[1], self.is_deconv, 4)
        self.up_concat04 = unetUp(filters[1], filters[0], self.is_deconv, 5)
        self.final_1 = nn.Conv2d(filters[0], n_classes, 1)
        self.final_2 = nn.Conv2d(filters[0], n_classes, 1)
        self.final_3 = nn.Conv2d(filters[0], n_classes, 1)
        self.final_4 = nn.Conv2d(filters[0], n_classes, 1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        X_00 = self.conv00(inputs)
        maxpool0 = self.maxpool0(X_00)
        X_10 = self.conv10(maxpool0)
        maxpool1 = self.maxpool1(X_10)
        X_20 = self.conv20(maxpool1)
        maxpool2 = self.maxpool2(X_20)
        X_30 = self.conv30(maxpool2)
        maxpool3 = self.maxpool3(X_30)
        X_40 = self.conv40(maxpool3)
        X_40_d = self.dilated(X_40)
        cls_branch = self.cls(X_40_d).squeeze()
        X_01 = self.up_concat01(X_10, X_00)
        X_11 = self.up_concat11(X_20, X_10)
        X_21 = self.up_concat21(X_30, X_20)
        X_31 = self.up_concat31(X_40_d, X_30)
        X_02 = self.up_concat02(X_11, X_00, X_01)
        X_12 = self.up_concat12(X_21, X_10, X_11)
        X_22 = self.up_concat22(X_31, X_20, X_21)
        X_03 = self.up_concat03(X_12, X_00, X_01, X_02)
        X_13 = self.up_concat13(X_22, X_10, X_11, X_12)
        X_04 = self.up_concat04(X_13, X_00, X_01, X_02, X_03)
        final_1 = self.final_1(X_01)
        final_2 = self.final_2(X_02)
        final_3 = self.final_3(X_03)
        final_4 = self.final_4(X_04)
        final = (final_1 + final_2 + final_3 + final_4) / 4
        if self.is_ds:
            return F.log_softmax(final), cls_branch
        else:
            return F.log_softmax(final_4), cls_branch


class unetConv2_res(nn.Module):

    def __init__(self, in_size, out_size, is_batchnorm, n=2, ks=3, stride=1, padding=1):
        super(unetConv2_res, self).__init__()
        self.n = n
        self.ks = ks
        self.stride = stride
        self.padding = padding
        s = stride
        p = padding
        self.conv0 = nn.Conv2d(in_size, out_size, 1)
        if is_batchnorm:
            for i in range(1, n + 1):
                conv = nn.Sequential(nn.Conv2d(in_size, out_size, ks, s, p), nn.BatchNorm2d(out_size), nn.ReLU(inplace=True))
                setattr(self, 'conv%d' % i, conv)
                in_size = out_size
        else:
            for i in range(1, n + 1):
                conv = nn.Sequential(nn.Conv2d(in_size, out_size, ks, s, p), nn.ReLU(inplace=True))
                setattr(self, 'conv%d' % i, conv)
                in_size = out_size
        for m in self.children():
            init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        inputs_ori = self.conv0(inputs)
        x = inputs
        for i in range(1, self.n + 1):
            conv = getattr(self, 'conv%d' % i)
            x = conv(x)
        return x + inputs_ori


class UnetConv3(nn.Module):

    def __init__(self, in_size, out_size, is_batchnorm, kernel_size=(3, 3, 3), padding_size=(1, 1, 1), init_stride=(1, 1, 1)):
        super(UnetConv3, self).__init__()
        if is_batchnorm:
            self.conv1 = nn.Sequential(nn.Conv3d(in_size, out_size, kernel_size, init_stride, padding_size), nn.BatchNorm3d(out_size), nn.ReLU(inplace=True))
            self.conv2 = nn.Sequential(nn.Conv3d(out_size, out_size, kernel_size, 1, padding_size), nn.BatchNorm3d(out_size), nn.ReLU(inplace=True))
        else:
            self.conv1 = nn.Sequential(nn.Conv3d(in_size, out_size, kernel_size, init_stride, padding_size), nn.ReLU(inplace=True))
            self.conv2 = nn.Sequential(nn.Conv3d(out_size, out_size, kernel_size, 1, padding_size), nn.ReLU(inplace=True))
        for m in self.children():
            init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        return outputs


class UnetUp3(nn.Module):

    def __init__(self, in_size, out_size, is_deconv, is_batchnorm=True):
        super(UnetUp3, self).__init__()
        if is_deconv:
            self.conv = UnetConv3(in_size, out_size, is_batchnorm)
            self.up = nn.ConvTranspose3d(in_size, out_size, kernel_size=(4, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1))
        else:
            self.conv = UnetConv3(in_size + out_size, out_size, is_batchnorm)
            self.up = nn.Upsample(scale_factor=(2, 2, 2), mode='trilinear')
        for m in self.children():
            if m.__class__.__name__.find('UnetConv3') != -1:
                continue
            init_weights(m, init_type='kaiming')

    def forward(self, inputs1, inputs2):
        outputs2 = self.up(inputs2)
        offset = outputs2.size()[2] - inputs1.size()[2]
        padding = 2 * [offset // 2, offset // 2, 0]
        outputs1 = F.pad(inputs1, padding)
        return self.conv(torch.cat([outputs1, outputs2], 1))


class Upold(nn.Module):

    def __init__(self, in_size, out_size, is_deconv):
        super(Upold, self).__init__()
        self.conv = unetConv2(in_size, out_size, False)
        if is_deconv:
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=4, stride=2, padding=1)
        else:
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        for m in self.children():
            if m.__class__.__name__.find('unetConv2') != -1:
                continue
            init_weights(m, init_type='kaiming')

    def forward(self, inputs1, inputs2):
        outputs2 = self.up(inputs2)
        offset = outputs2.size()[2] - inputs1.size()[2]
        padding = 2 * [offset // 2, offset // 2]
        outputs1 = F.pad(inputs1, padding)
        return self.conv(torch.cat([outputs1, outputs2], 1))


class unetConv2_SELU(nn.Module):

    def __init__(self, in_size, out_size, is_batchnorm, n=2, ks=3, stride=1, padding=1):
        super(unetConv2_SELU, self).__init__()
        self.n = n
        self.ks = ks
        self.stride = stride
        self.padding = padding
        s = stride
        p = padding
        if is_batchnorm:
            for i in range(1, n + 1):
                conv = nn.Sequential(nn.Conv2d(in_size, out_size, ks, s, p), nn.BatchNorm2d(out_size), nn.SELU(inplace=True))
                setattr(self, 'conv%d' % i, conv)
                in_size = out_size
        else:
            for i in range(1, n + 1):
                conv = nn.Sequential(nn.Conv2d(in_size, out_size, ks, s, p), nn.SELU(inplace=True))
                setattr(self, 'conv%d' % i, conv)
                in_size = out_size
        for m in self.children():
            init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        x = inputs
        for i in range(1, self.n + 1):
            conv = getattr(self, 'conv%d' % i)
            x = conv(x)
        return x


class unetUp_SELU(nn.Module):

    def __init__(self, in_size, out_size, is_deconv):
        super(unetUp_SELU, self).__init__()
        self.conv = unetConv2_SELU(in_size, out_size, False)
        if is_deconv:
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=4, stride=2, padding=1)
        else:
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        for m in self.children():
            if m.__class__.__name__.find('unetConv2') != -1:
                continue
            init_weights(m, init_type='kaiming')

    def forward(self, inputs1, inputs2):
        outputs2 = self.up(inputs2)
        offset = outputs2.size()[2] - inputs1.size()[2]
        padding = 2 * [offset // 2, offset // 2]
        outputs1 = F.pad(inputs1, padding)
        return self.conv(torch.cat([outputs1, outputs2], 1))


class unetConv2_dilation2(nn.Module):

    def __init__(self, in_size, out_size, is_batchnorm=True, n=3, ks=3, stride=1):
        super(unetConv2_dilation2, self).__init__()
        self.n = n
        self.ks = ks
        self.stride = stride
        s = stride
        if is_batchnorm:
            for i in range(1, n + 1):
                conv = nn.Sequential(nn.Conv2d(in_size, out_size, ks, s, 2 ** (i - 1), 2 ** (i - 1)), nn.BatchNorm2d(out_size), nn.ReLU(inplace=True))
                setattr(self, 'conv%d' % i, conv)
                in_size = out_size
        else:
            for i in range(1, n + 1):
                conv = nn.Sequential(nn.Conv2d(in_size, out_size, ks, s, p, r), nn.ReLU(inplace=True))
                setattr(self, 'conv%d' % i, conv)
                in_size = out_size
        for m in self.children():
            init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        output = inputs
        x_0 = inputs
        conv = getattr(self, 'conv1')
        x_1 = conv(x_0)
        conv = getattr(self, 'conv2')
        x_2 = conv(x_1)
        conv = getattr(self, 'conv3')
        x_3 = conv(x_2)
        return x_0 + x_1 + x_2 + x_3


class SELayer(nn.Module):

    def __init__(self, channel, reduction=2, is_bn=True, is_cse=True, is_sse=True):
        super(SELayer, self).__init__()
        self.is_cse = is_cse
        self.is_sse = is_sse
        self.is_bn = is_bn
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(channel, channel // reduction), nn.ReLU(inplace=True), nn.Linear(channel // reduction, channel), nn.Sigmoid())
        self.sse1 = nn.Conv2d(channel, 1, 1)
        self.sse2 = nn.Sigmoid()
        self.bn = nn.BatchNorm2d(channel)

    def forward(self, x):
        b, c, _, _ = x.size()
        y_c = self.avg_pool(x).view(b, c)
        y_c = self.fc(y_c).view(b, c, 1, 1)
        if self.is_bn:
            out_c = self.bn(x * y_c)
        else:
            out_c = x * y_c
        y_s = self.sse2(self.sse1(x))
        if self.is_bn:
            out_s = self.bn(x * y_s)
        else:
            out_s = x * y_s
        if self.is_cse and not self.is_sse:
            return out_c
        elif self.is_sse and not self.is_cse:
            return out_s
        else:
            return out_c + out_s


class Downsample(nn.Module):

    def __init__(self, in_size, out_size, is_batchnorm, ks=4, stride=2, padding=1):
        super(Downsample, self).__init__()
        self.ks = ks
        self.stride = stride
        self.padding = padding
        s = stride
        p = padding
        if is_batchnorm:
            self.conv = nn.Sequential(nn.Conv2d(in_size, out_size, ks, s, p), nn.BatchNorm2d(out_size), nn.ReLU(inplace=True))
        else:
            self.conv = nn.Sequential(nn.Conv2d(in_size, out_size, ks, s, p), nn.ReLU(inplace=True))
        for m in self.children():
            init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        x = self.conv(inputs)
        return x


class Atrous_module(nn.Module):

    def __init__(self, inplanes, planes, rate):
        super(Atrous_module, self).__init__()
        self.atrous_convolution = nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding=rate, dilation=rate)
        self.batch_norm = nn.BatchNorm2d(planes)

    def forward(self, x):
        x = self.atrous_convolution(x)
        x = self.batch_norm(x)
        return x


class DiceLoss(nn.Module):

    def __init__(self, class_num=4, smooth=1):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.class_num = class_num

    def forward(self, input, target):
        input = torch.exp(input)
        self.smooth = 0.0
        Dice = Variable(torch.Tensor([0]).float())
        for i in range(1, self.class_num):
            input_i = input[:, i, :, :]
            target_i = (target == i).float()
            intersect = (input_i * target_i).sum()
            union = torch.sum(input_i) + torch.sum(target_i)
            if target_i.sum() == 0:
                dice = Variable(torch.Tensor([1]).float())
            else:
                dice = (2 * intersect + self.smooth) / (union + self.smooth)
            Dice += dice
        dice_loss = 1 - Dice / (self.class_num - 1)
        return dice_loss


class EL_DiceLoss(nn.Module):

    def __init__(self, class_num=4, smooth=1, gamma=0.5):
        super(EL_DiceLoss, self).__init__()
        self.smooth = smooth
        self.class_num = class_num
        self.gamma = gamma

    def forward(self, input, target):
        input = torch.exp(input)
        self.smooth = 0.0
        Dice = Variable(torch.Tensor([0]).float())
        for i in range(1, self.class_num):
            input_i = input[:, i, :, :]
            target_i = (target == i).float()
            intersect = (input_i * target_i).sum()
            union = torch.sum(input_i) + torch.sum(target_i)
            if target_i.sum() == 0:
                dice = Variable(torch.Tensor([1]).float())
            else:
                dice = (2 * intersect + self.smooth) / (union + self.smooth)
            Dice += (-torch.log(dice)) ** self.gamma
        dice_loss = Dice / (self.class_num - 1)
        return dice_loss


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (Atrous_module,
     lambda: ([], {'inplanes': 4, 'planes': 4, 'rate': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (DiceLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (Downsample,
     lambda: ([], {'in_size': 4, 'out_size': 4, 'is_batchnorm': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (EL_DiceLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (SELayer,
     lambda: ([], {'channel': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (UNet,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {})),
    (UNet_Nested,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {})),
    (UNet_Nested_dilated,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {})),
    (UnetConv3,
     lambda: ([], {'in_size': 4, 'out_size': 4, 'is_batchnorm': 4}),
     lambda: ([torch.rand([4, 4, 4, 4, 4])], {})),
    (Upold,
     lambda: ([], {'in_size': 4, 'out_size': 4, 'is_deconv': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {})),
    (unetConv2,
     lambda: ([], {'in_size': 4, 'out_size': 4, 'is_batchnorm': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (unetConv2_SELU,
     lambda: ([], {'in_size': 4, 'out_size': 4, 'is_batchnorm': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (unetConv2_dilation,
     lambda: ([], {'in_size': 4, 'out_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (unetConv2_dilation2,
     lambda: ([], {'in_size': 4, 'out_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (unetConv2_res,
     lambda: ([], {'in_size': 4, 'out_size': 4, 'is_batchnorm': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (unetUp,
     lambda: ([], {'in_size': 4, 'out_size': 4, 'is_deconv': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (unetUp_SELU,
     lambda: ([], {'in_size': 4, 'out_size': 4, 'is_deconv': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {})),
]

