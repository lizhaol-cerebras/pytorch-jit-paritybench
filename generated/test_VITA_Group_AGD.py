
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


import math


import numpy as np


import torch


import torchvision.transforms as transforms


from torch.autograd import Variable


import time


import matplotlib


from matplotlib import pyplot as plt


import warnings


from matplotlib.patches import Rectangle


import torch.nn as nn


from collections import OrderedDict


from collections import defaultdict


import torch.utils.model_zoo as model_zoo


import torch.distributed as dist


import functools


import torch.nn.functional as F


from torch.nn import functional as F


import random


from torch.utils.data import Dataset


import logging


import torch.utils


import torch.backends.cudnn as cudnn


from torch.utils.data import DataLoader


from torchvision.utils import save_image


from collections import namedtuple


from torch.autograd.function import InplaceFunction


from torch.autograd.function import Function


from scipy import linalg


from torch.nn.functional import adaptive_avg_pool2d


from torchvision import models


from torchvision.models.vgg import vgg19


from torchvision.models.vgg import vgg16


class ResidualDenseBlock_5C(nn.Module):

    def __init__(self, nf=64, gc=32, bias=True):
        super(ResidualDenseBlock_5C, self).__init__()
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    """Residual in Residual Dense Block"""

    def __init__(self, nf, gc=32):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(nf, gc)
        self.RDB2 = ResidualDenseBlock_5C(nf, gc)
        self.RDB3 = ResidualDenseBlock_5C(nf, gc)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + x


def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)


class RRDBNet(nn.Module):

    def __init__(self, in_nc, out_nc, nf, nb, gc=32):
        super(RRDBNet, self).__init__()
        RRDB_block_f = functools.partial(RRDB, nf=nf, gc=gc)
        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.RRDB_trunk = make_layer(RRDB_block_f, nb)
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.upconv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        fea = self.conv_first(x)
        trunk = self.trunk_conv(self.RRDB_trunk(fea))
        fea = fea + trunk
        fea = self.lrelu(self.upconv1(F.interpolate(fea, scale_factor=2, mode='nearest')))
        fea = self.lrelu(self.upconv2(F.interpolate(fea, scale_factor=2, mode='nearest')))
        out = self.conv_last(self.lrelu(self.HRconv(fea)))
        return out


BatchNorm2d = nn.InstanceNorm2d


QParams = namedtuple('QParams', ['range', 'zero_point', 'num_bits'])


_DEFAULT_FLATTEN = 1, -1


def _deflatten_as(x, x_full):
    shape = list(x.shape) + [1] * (x_full.dim() - x.dim())
    return x.view(*shape)


def calculate_qparams(x, num_bits, flatten_dims=_DEFAULT_FLATTEN, reduce_dim=0, reduce_type='mean', keepdim=False, true_zero=False):
    with torch.no_grad():
        x_flat = x.flatten(*flatten_dims)
        if x_flat.dim() == 1:
            min_values = _deflatten_as(x_flat.min(), x)
            max_values = _deflatten_as(x_flat.max(), x)
        else:
            min_values = _deflatten_as(x_flat.min(-1)[0], x)
            max_values = _deflatten_as(x_flat.max(-1)[0], x)
        if reduce_dim is not None:
            if reduce_type == 'mean':
                min_values = min_values.mean(reduce_dim, keepdim=keepdim)
                max_values = max_values.mean(reduce_dim, keepdim=keepdim)
            else:
                min_values = min_values.min(reduce_dim, keepdim=keepdim)[0]
                max_values = max_values.max(reduce_dim, keepdim=keepdim)[0]
        range_values = max_values - min_values
        return QParams(range=range_values, zero_point=min_values, num_bits=num_bits)


class UniformQuantize(InplaceFunction):

    @staticmethod
    def forward(ctx, input, num_bits=None, qparams=None, flatten_dims=_DEFAULT_FLATTEN, reduce_dim=0, dequantize=True, signed=False, stochastic=False, inplace=False):
        ctx.inplace = inplace
        if ctx.inplace:
            ctx.mark_dirty(input)
            output = input
        else:
            output = input.clone()
        if qparams is None:
            assert num_bits is not None, 'either provide qparams of num_bits to quantize'
            qparams = calculate_qparams(input, num_bits=num_bits, flatten_dims=flatten_dims, reduce_dim=reduce_dim)
        zero_point = qparams.zero_point
        num_bits = qparams.num_bits
        qmin = -2.0 ** (num_bits - 1) if signed else 0.0
        qmax = qmin + 2.0 ** num_bits - 1.0
        scale = qparams.range / (qmax - qmin)
        with torch.no_grad():
            output.add_(qmin * scale - zero_point).div_(scale)
            if stochastic:
                noise = output.new(output.shape).uniform_(-0.5, 0.5)
                output.add_(noise)
            output.clamp_(qmin, qmax).round_()
            if dequantize:
                output.mul_(scale).add_(zero_point - qmin * scale)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output
        return grad_input, None, None, None, None, None, None, None, None


def Quantize(x, num_bits=None, qparams=None, flatten_dims=_DEFAULT_FLATTEN, reduce_dim=0, dequantize=True, signed=False, stochastic=False, inplace=False):
    return UniformQuantize().apply(x, num_bits, qparams, flatten_dims, reduce_dim, dequantize, signed, stochastic, inplace)


class QuantMeasure(nn.Module):
    """docstring for QuantMeasure."""

    def __init__(self, num_bits=8, shape_measure=(1,), flatten_dims=_DEFAULT_FLATTEN, inplace=False, dequantize=True, stochastic=False, momentum=0.1, measure=False):
        super(QuantMeasure, self).__init__()
        self.register_buffer('running_zero_point', torch.zeros(*shape_measure))
        self.register_buffer('running_range', torch.zeros(*shape_measure))
        self.measure = measure
        if self.measure:
            self.register_buffer('num_measured', torch.zeros(1))
        self.flatten_dims = flatten_dims
        self.momentum = momentum
        self.dequantize = dequantize
        self.stochastic = stochastic
        self.inplace = inplace
        self.num_bits = num_bits

    def forward(self, input, qparams=None):
        if self.training or self.measure:
            if qparams is None:
                qparams = calculate_qparams(input, num_bits=self.num_bits, flatten_dims=self.flatten_dims, reduce_dim=0)
            with torch.no_grad():
                if self.measure:
                    momentum = self.num_measured / (self.num_measured + 1)
                    self.num_measured += 1
                else:
                    momentum = self.momentum
                self.running_zero_point.mul_(momentum).add_(qparams.zero_point * (1 - momentum))
                self.running_range.mul_(momentum).add_(qparams.range * (1 - momentum))
        else:
            qparams = QParams(range=self.running_range, zero_point=self.running_zero_point, num_bits=self.num_bits)
        if self.measure:
            return input
        else:
            q_input = Quantize(input, qparams=qparams, dequantize=self.dequantize, stochastic=self.stochastic, inplace=self.inplace)
            return q_input


class QConv2d(nn.Conv2d):
    """docstring for QConv2d."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, num_bits=8, num_bits_weight=8):
        super(QConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.num_bits = num_bits
        self.num_bits_weight = num_bits_weight or num_bits

    def forward(self, input, quantize=False):
        if quantize:
            if not hasattr(self, 'quantize_input'):
                self.quantize_input = QuantMeasure(self.num_bits, shape_measure=(1, 1, 1, 1), flatten_dims=(1, -1))
            qinput = self.quantize_input(input)
            weight_qparams = calculate_qparams(self.weight, num_bits=self.num_bits_weight, flatten_dims=(1, -1), reduce_dim=None)
            qweight = Quantize(self.weight, qparams=weight_qparams)
            if self.bias is not None:
                qbias = Quantize(self.bias, num_bits=self.num_bits_weight + self.num_bits, flatten_dims=(0, -1))
            else:
                qbias = None
            output = F.conv2d(qinput, qweight, qbias, self.stride, self.padding, self.dilation, self.groups)
        else:
            output = F.conv2d(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return output


Conv2d = QConv2d


def make_divisible(v, divisor=8, min_value=3):
    """
    forked from slim:
    https://github.com/tensorflow/models/blob/    0344c5503ee55e24f0de7f37336a6e08f10976fd/    research/slim/nets/mobilenet/mobilenet.py#L62-L69
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class USBatchNorm2d(nn.InstanceNorm2d):

    def __init__(self, num_features, width_mult_list=[1.0]):
        super(USBatchNorm2d, self).__init__(num_features, affine=False, track_running_stats=False)
        self.num_features_max = num_features
        self.width_mult_list = width_mult_list
        self.bn = nn.ModuleList([nn.InstanceNorm2d(i, affine=False) for i in [make_divisible(self.num_features_max * width_mult) for width_mult in width_mult_list]])
        self.ratio = 1.0

    def set_ratio(self, ratio):
        self.ratio = ratio

    def forward(self, input):
        weight = self.weight
        bias = self.bias
        assert self.ratio in self.width_mult_list
        idx = self.width_mult_list.index(self.ratio)
        y = self.bn[idx](input)
        return y


class USConv2d(QConv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, width_mult_list=[1.0], num_bits=8, num_bits_weight=8):
        super(USConv2d, self).__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias, num_bits=num_bits, num_bits_weight=num_bits_weight)
        self.in_channels_max = in_channels
        self.out_channels_max = out_channels
        self.width_mult_list = width_mult_list
        self.ratio = 1.0, 1.0

    def set_ratio(self, ratio):
        self.ratio = ratio

    def forward(self, input, quantize=False):
        self.in_channels = make_divisible(self.in_channels_max * self.ratio[0])
        assert self.ratio[1] in self.width_mult_list, str(self.ratio[1]) + ' in? ' + str(self.width_mult_list)
        self.out_channels = make_divisible(self.out_channels_max * self.ratio[1])
        weight = self.weight[:self.out_channels, :self.in_channels, :, :]
        if self.groups != 1:
            self.groups = self.out_channels
        if self.bias is not None:
            bias = self.bias[:self.out_channels]
        else:
            bias = self.bias
        if quantize:
            if not hasattr(self, 'quantize_input'):
                self.quantize_input = QuantMeasure(self.num_bits, shape_measure=(1, 1, 1, 1), flatten_dims=(1, -1))
            qinput = self.quantize_input(input)
            weight_qparams = calculate_qparams(weight, num_bits=self.num_bits_weight, flatten_dims=(1, -1), reduce_dim=None)
            qweight = Quantize(weight, qparams=weight_qparams)
            if self.bias is not None:
                qbias = Quantize(bias, num_bits=self.num_bits_weight + self.num_bits, flatten_dims=(0, -1))
            else:
                qbias = None
            output = F.conv2d(qinput, qweight, qbias, self.stride, self.padding, self.dilation, self.groups)
        else:
            output = F.conv2d(input, weight, bias, self.stride, self.padding, self.dilation, self.groups)
        return output


class QConvTranspose2d(nn.ConvTranspose2d):
    """docstring for QConv2d."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=2, padding=1, output_padding=1, dilation=1, groups=1, bias=True, num_bits=8, num_bits_weight=8):
        super(QConvTranspose2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, output_padding, groups, bias, dilation)
        self.num_bits = num_bits
        self.num_bits_weight = num_bits_weight or num_bits

    def forward(self, input, quantize=False):
        if quantize:
            if not hasattr(self, 'quantize_input'):
                self.quantize_input = QuantMeasure(self.num_bits, shape_measure=(1, 1, 1, 1), flatten_dims=(1, -1))
            qinput = self.quantize_input(input)
            weight_qparams = calculate_qparams(self.weight, num_bits=self.num_bits_weight, flatten_dims=(1, -1), reduce_dim=None)
            qweight = Quantize(self.weight, qparams=weight_qparams)
            if self.bias is not None:
                qbias = Quantize(self.bias, num_bits=self.num_bits_weight + self.num_bits, flatten_dims=(0, -1))
            else:
                qbias = None
            output = F.conv_transpose2d(qinput, qweight, qbias, self.stride, self.padding, self.output_padding, self.groups, self.dilation)
        else:
            output = F.conv_transpose2d(input, self.weight, self.bias, self.stride, self.padding, self.output_padding, self.groups, self.dilation)
        return output


def count_custom(m, x, y):
    m.total_ops += 0


flops_lookup_table = {}


latency_lookup_table = {}


table_file_name = 'flops_lookup_table.npy'


class BasicResidual(nn.Module):

    def __init__(self, C_in, C_out, kernel_size=3, stride=1, dilation=1, groups=1, slimmable=True, width_mult_list=[1.0]):
        super(BasicResidual, self).__init__()
        self.C_in = C_in
        self.C_out = C_out
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.slimmable = slimmable
        self.width_mult_list = width_mult_list
        assert stride in [1, 2]
        if self.stride == 2:
            self.dilation = 1
        self.ratio = 1.0, 1.0
        self.relu = nn.ReLU(inplace=True)
        if self.slimmable:
            self.conv1 = USConv2d(C_in, C_out, 3, stride, padding=dilation, dilation=dilation, groups=groups, bias=False, width_mult_list=width_mult_list)
            self.bn1 = USBatchNorm2d(C_out, width_mult_list)
            self.conv2 = USConv2d(C_out, C_out, 3, 1, padding=dilation, dilation=dilation, groups=groups, bias=False, width_mult_list=width_mult_list)
            self.bn2 = USBatchNorm2d(C_out, width_mult_list)
            self.skip = USConv2d(C_in, C_out, 1, stride, padding=0, dilation=dilation, groups=1, bias=False, width_mult_list=width_mult_list)
            self.bn3 = USBatchNorm2d(C_out, width_mult_list)
        else:
            self.conv1 = Conv2d(C_in, C_out, 3, stride, padding=dilation, dilation=dilation, groups=groups, bias=False)
            self.bn1 = BatchNorm2d(C_out)
            self.conv2 = Conv2d(C_out, C_out, 3, 1, padding=dilation, dilation=dilation, groups=groups, bias=False)
            self.bn2 = BatchNorm2d(C_out)
            if self.C_in != self.C_out or self.stride != 1:
                self.skip = Conv2d(C_in, C_out, 1, stride, padding=0, dilation=dilation, groups=1, bias=False)
                self.bn3 = BatchNorm2d(C_out)

    def set_ratio(self, ratio):
        assert len(ratio) == 2
        self.ratio = ratio
        self.conv1.set_ratio(ratio)
        self.bn1.set_ratio(ratio[1])
        self.conv2.set_ratio((ratio[1], ratio[1]))
        self.bn2.set_ratio(ratio[1])
        if hasattr(self, 'skip'):
            self.skip.set_ratio(ratio)
            self.bn3.set_ratio(ratio[1])

    @staticmethod
    def _flops(h, w, C_in, C_out, kernel_size=3, stride=1, dilation=1, groups=1):
        layer = BasicResidual(C_in, C_out, kernel_size, stride, dilation, groups, slimmable=False)
        flops, params = profile(layer, inputs=(torch.randn(1, C_in, h, w),), custom_ops=custom_ops)
        return flops

    @staticmethod
    def _latency(h, w, C_in, C_out, kernel_size=3, stride=1, dilation=1, groups=1):
        layer = BasicResidual(C_in, C_out, kernel_size, stride, dilation, groups, slimmable=False)
        latency = compute_latency(layer, (1, C_in, h, w))
        return latency

    def forward_latency(self, size):
        c_in, h_in, w_in = size
        if self.slimmable:
            assert c_in == make_divisible(self.C_in * self.ratio[0])
            c_out = make_divisible(self.C_out * self.ratio[1])
        else:
            assert c_in == self.C_in, 'c_in %d, self.C_in%d' % (c_in, self.C_in)
            c_out = self.C_out
        if self.stride == 1:
            h_out = h_in
            w_out = w_in
        else:
            h_out = h_in // 2
            w_out = w_in // 2
        name = 'BasicResidual_H%d_W%d_Cin%d_Cout%d_stride%d_dilation%d' % (h_in, w_in, c_in, c_out, self.stride, self.dilation)
        if name in latency_lookup_table:
            latency = latency_lookup_table[name]
        else:
            None
            latency = BasicResidual._latency(h_in, w_in, c_in, c_out, self.kernel_size, self.stride, self.dilation, self.groups)
            latency_lookup_table[name] = latency
            np.save(table_file_name, latency_lookup_table)
        return latency, (c_out, h_out, w_out)

    def forward_flops(self, size, quantize=False):
        c_in, h_in, w_in = size
        if self.slimmable:
            assert c_in == make_divisible(self.C_in * self.ratio[0]), 'c_in %d, self.C_in * self.ratio[0] %d' % (c_in, self.C_in * self.ratio[0])
            c_out = make_divisible(self.C_out * self.ratio[1])
        else:
            assert c_in == self.C_in, 'c_in %d, self.C_in %d' % (c_in, self.C_in)
            c_out = self.C_out
        if self.stride == 1:
            h_out = h_in
            w_out = w_in
        else:
            h_out = h_in // 2
            w_out = w_in // 2
        name = 'BasicResidual_H%d_W%d_Cin%d_Cout%d_kernel%d_stride%d' % (h_in, w_in, c_in, c_out, self.kernel_size, self.stride)
        if name in flops_lookup_table:
            flops = flops_lookup_table[name]
        else:
            None
            flops = BasicResidual._flops(h_in, w_in, c_in, c_out, self.kernel_size, self.stride, self.dilation, self.groups)
            flops_lookup_table[name] = flops
            np.save(table_file_name, flops_lookup_table)
        return flops, (c_out, h_out, w_out)

    def forward(self, x, quantize=False):
        identity = x
        out = self.conv1(x, quantize=quantize)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out, quantize=quantize)
        out = self.bn2(out)
        if hasattr(self, 'skip'):
            identity = self.bn3(self.skip(identity, quantize=quantize))
        out += identity
        out = self.relu(out)
        return out


class Conv3x3(nn.Module):

    def __init__(self, C_in, C_out, kernel_size=3, stride=1, dilation=1, groups=1, slimmable=True, width_mult_list=[1.0]):
        super(Conv3x3, self).__init__()
        self.C_in = C_in
        self.C_out = C_out
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.slimmable = slimmable
        self.width_mult_list = width_mult_list
        assert stride in [1, 2]
        if self.stride == 2:
            self.dilation = 1
        self.ratio = 1.0, 1.0
        self.relu = nn.ReLU(inplace=True)
        if slimmable:
            self.conv1 = USConv2d(C_in, C_out, 3, stride, padding=dilation, dilation=dilation, groups=groups, bias=False, width_mult_list=width_mult_list)
            self.bn1 = USBatchNorm2d(C_out, width_mult_list)
        else:
            self.conv1 = Conv2d(C_in, C_out, 3, stride, padding=dilation, dilation=dilation, groups=groups, bias=False)
            self.bn1 = BatchNorm2d(C_out)

    def set_ratio(self, ratio):
        assert len(ratio) == 2
        self.ratio = ratio
        self.conv1.set_ratio(ratio)
        self.bn1.set_ratio(ratio[1])

    @staticmethod
    def _flops(h, w, C_in, C_out, kernel_size=3, stride=1, dilation=1, groups=1):
        layer = Conv3x3(C_in, C_out, kernel_size, stride, dilation, groups, slimmable=False)
        flops, params = profile(layer, inputs=(torch.randn(1, C_in, h, w),), custom_ops=custom_ops)
        return flops

    @staticmethod
    def _latency(h, w, C_in, C_out, kernel_size=3, stride=1, dilation=1, groups=1):
        layer = Conv3x3(C_in, C_out, kernel_size, stride, dilation, groups, slimmable=False)
        latency = compute_latency(layer, (1, C_in, h, w))
        return latency

    def forward_latency(self, size):
        c_in, h_in, w_in = size
        if self.slimmable:
            assert c_in == make_divisible(self.C_in * self.ratio[0]), 'c_in %d, int(self.C_in * self.ratio[0]) %d' % (c_in, int(self.C_in * self.ratio[0]))
            c_out = make_divisible(self.C_out * self.ratio[1])
        else:
            assert c_in == self.C_in, 'c_in %d, self.C_in %d' % (c_in, self.C_in)
            c_out = self.C_out
        if self.stride == 1:
            h_out = h_in
            w_out = w_in
        else:
            h_out = h_in // 2
            w_out = w_in // 2
        name = 'Conv3x3_H%d_W%d_Cin%d_Cout%d_stride%d_dilation%d' % (h_in, w_in, c_in, c_out, self.stride, self.dilation)
        if name in latency_lookup_table:
            latency = latency_lookup_table[name]
        else:
            None
            latency = Conv3x3._latency(h_in, w_in, c_in, c_out, self.kernel_size, self.stride, self.dilation, self.groups)
            latency_lookup_table[name] = latency
            np.save(table_file_name, latency_lookup_table)
        return latency, (c_out, h_out, w_out)

    def forward_flops(self, size, quantize=False):
        c_in, h_in, w_in = size
        if self.slimmable:
            assert c_in == make_divisible(self.C_in * self.ratio[0]), 'c_in %d, self.C_in * self.ratio[0] %d' % (c_in, self.C_in * self.ratio[0])
            c_out = make_divisible(self.C_out * self.ratio[1])
        else:
            assert c_in == self.C_in, 'c_in %d, self.C_in %d' % (c_in, self.C_in)
            c_out = self.C_out
        if self.stride == 1:
            h_out = h_in
            w_out = w_in
        else:
            h_out = h_in // 2
            w_out = w_in // 2
        name = 'Conv3x3_H%d_W%d_Cin%d_Cout%d_kernel%d_stride%d' % (h_in, w_in, c_in, c_out, self.kernel_size, self.stride)
        if name in flops_lookup_table:
            flops = flops_lookup_table[name]
        else:
            None
            flops = Conv3x3._flops(h_in, w_in, c_in, c_out, self.kernel_size, self.stride, self.dilation, self.groups)
            flops_lookup_table[name] = flops
            np.save(table_file_name, flops_lookup_table)
        return flops, (c_out, h_out, w_out)

    def forward(self, x, quantize=False):
        identity = x
        out = self.conv1(x, quantize=quantize)
        out = self.bn1(out)
        out = self.relu(out)
        return out


class DwsBlock(nn.Module):

    def __init__(self, C_in, C_out, kernel_size=3, stride=1, dilation=1, groups=1, slimmable=True, width_mult_list=[1.0]):
        super(DwsBlock, self).__init__()
        self.C_in = C_in
        self.C_out = C_out
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.slimmable = slimmable
        self.width_mult_list = width_mult_list
        assert stride in [1, 2]
        if self.stride == 2:
            self.dilation = 1
        self.ratio = 1.0, 1.0
        self.relu = nn.ReLU(inplace=True)
        if self.slimmable:
            self.conv1 = USConv2d(C_in, C_in, 3, stride, padding=dilation, dilation=dilation, groups=C_in, bias=False, width_mult_list=width_mult_list)
            self.bn1 = USBatchNorm2d(C_out, width_mult_list)
            self.conv2 = USConv2d(C_in, C_out, 1, 1, padding=0, dilation=dilation, groups=groups, bias=False, width_mult_list=width_mult_list)
            self.bn2 = USBatchNorm2d(C_out, width_mult_list)
        else:
            self.conv1 = Conv2d(C_in, C_in, 3, stride, padding=dilation, dilation=dilation, groups=C_in, bias=False)
            self.bn1 = BatchNorm2d(C_out)
            self.conv2 = Conv2d(C_in, C_out, 1, 1, padding=0, dilation=dilation, groups=groups, bias=False)
            self.bn2 = BatchNorm2d(C_out)

    def set_ratio(self, ratio):
        assert len(ratio) == 2
        self.ratio = ratio
        self.conv1.set_ratio((1, ratio[0]))
        self.bn1.set_ratio(ratio[0])
        self.conv2.set_ratio(ratio)
        self.bn2.set_ratio(ratio[1])

    @staticmethod
    def _flops(h, w, C_in, C_out, kernel_size=3, stride=1, dilation=1, groups=1):
        layer = DwsBlock(C_in, C_out, kernel_size, stride, dilation, groups=1, slimmable=False)
        flops, params = profile(layer, inputs=(torch.randn(1, C_in, h, w),), custom_ops=custom_ops)
        return flops

    @staticmethod
    def _latency(h, w, C_in, C_out, kernel_size=3, stride=1, dilation=1, groups=1):
        layer = DwsBlock(C_in, C_out, kernel_size, stride, dilation, groups, slimmable=False)
        latency = compute_latency(layer, (1, C_in, h, w))
        return latency

    def forward_latency(self, size):
        c_in, h_in, w_in = size
        if self.slimmable:
            c_out = make_divisible(self.C_out * self.ratio[1])
        else:
            c_out = self.C_out
        if self.stride == 1:
            h_out = h_in
            w_out = w_in
        else:
            h_out = h_in // 2
            w_out = w_in // 2
        name = 'DwsBlock_H%d_W%d_Cin%d_Cout%d_stride%d_dilation%d' % (h_in, w_in, c_in, c_out, self.stride, self.dilation)
        if name in latency_lookup_table:
            latency = latency_lookup_table[name]
        else:
            None
            latency = DwsBlock._latency(h_in, w_in, c_in, c_out, self.kernel_size, self.stride, self.dilation, self.groups)
            latency_lookup_table[name] = latency
            np.save(table_file_name, latency_lookup_table)
        return latency, (c_out, h_out, w_out)

    def forward_flops(self, size, quantize=False):
        c_in, h_in, w_in = size
        if self.slimmable:
            c_out = make_divisible(self.C_out * self.ratio[1])
        else:
            c_out = self.C_out
        if self.stride == 1:
            h_out = h_in
            w_out = w_in
        else:
            h_out = h_in // 2
            w_out = w_in // 2
        name = 'DwsBlock_H%d_W%d_Cin%d_Cout%d_kernel%d_stride%d' % (h_in, w_in, c_in, c_out, self.kernel_size, self.stride)
        if name in flops_lookup_table:
            flops = flops_lookup_table[name]
        else:
            None
            flops = DwsBlock._flops(h_in, w_in, c_in, c_out, self.kernel_size, self.stride, self.dilation, self.groups)
            flops_lookup_table[name] = flops
            np.save(table_file_name, flops_lookup_table)
        return flops, (c_out, h_out, w_out)

    def forward(self, x, quantize=False):
        identity = x
        out = self.conv1(x, quantize=False)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out, quantize=quantize)
        out = self.bn2(out)
        out = self.relu(out)
        return out


class SkipConnect(nn.Module):

    def __init__(self, C_in, C_out, stride=1, slimmable=True, width_mult_list=[1.0]):
        super(SkipConnect, self).__init__()
        assert stride in [1, 2]
        assert C_out % 2 == 0, 'C_out=%d' % C_out
        self.C_in = C_in
        self.C_out = C_out
        self.stride = stride
        self.slimmable = slimmable
        self.width_mult_list = width_mult_list
        self.ratio = 1.0, 1.0
        self.kernel_size = 1
        self.padding = 0
        if slimmable:
            self.conv = USConv2d(C_in, C_out, 1, stride=stride, padding=0, bias=False, width_mult_list=width_mult_list)
            self.bn = USBatchNorm2d(C_out, width_mult_list)
            self.relu = nn.ReLU(inplace=True)
        else:
            self.conv = Conv2d(C_in, C_out, 1, stride=stride, padding=0, bias=False)
            self.bn = BatchNorm2d(C_out)
            self.relu = nn.ReLU(inplace=True)

    def set_ratio(self, ratio):
        assert len(ratio) == 2
        self.ratio = ratio
        self.conv.set_ratio(ratio)
        self.bn.set_ratio(ratio[1])

    @staticmethod
    def _flops(h, w, C_in, C_out, stride=1):
        layer = SkipConnect(C_in, C_out, stride, slimmable=False)
        flops, params = profile(layer, inputs=(torch.randn(1, C_in, h, w),), custom_ops=custom_ops)
        return flops

    @staticmethod
    def _latency(h, w, C_in, C_out, stride=1):
        layer = SkipConnect(C_in, C_out, stride, slimmable=False)
        latency = compute_latency(layer, (1, C_in, h, w))
        return latency

    def forward_latency(self, size):
        c_in, h_in, w_in = size
        if self.slimmable:
            assert c_in == make_divisible(self.C_in * self.ratio[0])
            c_out = make_divisible(self.C_out * self.ratio[1])
        else:
            assert c_in == self.C_in
            c_out = self.C_out
        if self.stride == 1:
            h_out = h_in
            w_out = w_in
        else:
            h_out = h_in // 2
            w_out = w_in // 2
        name = 'SkipConnect_H%d_W%d_Cin%d_Cout%d_stride%d' % (h_in, w_in, c_in, c_out, self.stride)
        if name in latency_lookup_table:
            latency = latency_lookup_table[name]
        else:
            None
            latency = SkipConnect._latency(h_in, w_in, c_in, c_out, self.stride)
            latency_lookup_table[name] = latency
            np.save(table_file_name, latency_lookup_table)
        return latency, (c_out, h_out, w_out)

    def forward_flops(self, size, quantize=False):
        c_in, h_in, w_in = size
        if self.slimmable:
            assert c_in == make_divisible(self.C_in * self.ratio[0]), 'c_in %d, self.C_in * self.ratio[0] %d' % (c_in, self.C_in * self.ratio[0])
            c_out = make_divisible(self.C_out * self.ratio[1])
        else:
            assert c_in == self.C_in, 'c_in %d, self.C_in %d' % (c_in, self.C_in)
            c_out = self.C_out
        if self.stride == 1:
            h_out = h_in
            w_out = w_in
        else:
            h_out = h_in // 2
            w_out = w_in // 2
        name = 'SkipConnect_H%d_W%d_Cin%d_Cout%d_stride%d' % (h_in, w_in, c_in, c_out, self.stride)
        if name in flops_lookup_table:
            flops = flops_lookup_table[name]
        else:
            None
            flops = SkipConnect._flops(h_in, w_in, c_in, c_out, self.stride)
            flops_lookup_table[name] = flops
            np.save(table_file_name, flops_lookup_table)
        return flops, (c_out, h_out, w_out)

    def forward(self, x, quantize=False):
        if hasattr(self, 'conv'):
            out = self.conv(x, quantize=quantize)
            out = self.bn(out)
            out = self.relu(out)
        else:
            out = x
        return out


OPS = {'skip': lambda C_in, C_out, stride, slimmable, width_mult_list: SkipConnect(C_in, C_out, stride, slimmable, width_mult_list), 'conv3x3': lambda C_in, C_out, stride, slimmable, width_mult_list: Conv3x3(C_in, C_out, kernel_size=3, stride=stride, dilation=1, slimmable=slimmable, width_mult_list=width_mult_list), 'conv3x3_d2': lambda C_in, C_out, stride, slimmable, width_mult_list: Conv3x3(C_in, C_out, kernel_size=3, stride=stride, dilation=2, slimmable=slimmable, width_mult_list=width_mult_list), 'conv3x3_d4': lambda C_in, C_out, stride, slimmable, width_mult_list: Conv3x3(C_in, C_out, kernel_size=3, stride=stride, dilation=4, slimmable=slimmable, width_mult_list=width_mult_list), 'residual': lambda C_in, C_out, stride, slimmable, width_mult_list: BasicResidual(C_in, C_out, kernel_size=3, stride=stride, dilation=1, slimmable=slimmable, width_mult_list=width_mult_list), 'dwsblock': lambda C_in, C_out, stride, slimmable, width_mult_list: DwsBlock(C_in, C_out, kernel_size=3, stride=stride, dilation=1, slimmable=slimmable, width_mult_list=width_mult_list)}


PRIMITIVES = ['skip', 'conv', 'conv_di', 'conv_2x', 'conv_2x_di']


class MixedOp(nn.Module):

    def __init__(self, C_in, C_out, stride=1, slimmable=True, width_mult_list=[1.0], quantize=False, width_mult_list_left=None):
        super(MixedOp, self).__init__()
        self._ops = nn.ModuleList()
        self._width_mult_list = width_mult_list
        self.quantize = quantize
        self.slimmable = slimmable
        self._width_mult_list_left = width_mult_list_left if width_mult_list_left is not None else width_mult_list
        for primitive in PRIMITIVES:
            op = OPS[primitive](C_in, C_out, stride, slimmable=slimmable, width_mult_list=width_mult_list)
            self._ops.append(op)

    def set_prun_ratio(self, ratio):
        for op in self._ops:
            op.set_ratio(ratio)

    def forward(self, x, alpha, beta, ratio):
        result = 0
        if isinstance(ratio[0], torch.Tensor):
            ratio0 = self._width_mult_list_left[ratio[0].argmax()]
            r_score0 = ratio[0][ratio[0].argmax()]
        else:
            ratio0 = ratio[0]
            r_score0 = 1.0
        if isinstance(ratio[1], torch.Tensor):
            ratio1 = self._width_mult_list[ratio[1].argmax()]
            r_score1 = ratio[1][ratio[1].argmax()]
        else:
            ratio1 = ratio[1]
            r_score1 = 1.0
        if self.slimmable:
            self.set_prun_ratio((ratio0, ratio1))
        for w, op in zip(alpha, self._ops):
            if self.quantize == 'search':
                result = result + (beta[0] * op(x, quantize=False) + beta[1] * op(x, quantize=True)) * w * r_score0 * r_score1
            elif self.quantize:
                result = result + op(x, quantize=True) * w * r_score0 * r_score1
            else:
                result = result + op(x, quantize=False) * w * r_score0 * r_score1
        return result

    def forward_latency(self, size, alpha, ratio):
        result = 0
        if isinstance(ratio[0], torch.Tensor):
            ratio0 = self._width_mult_list_left[ratio[0].argmax()]
            r_score0 = ratio[0][ratio[0].argmax()]
        else:
            ratio0 = ratio[0]
            r_score0 = 1.0
        if isinstance(ratio[1], torch.Tensor):
            ratio1 = self._width_mult_list[ratio[1].argmax()]
            r_score1 = ratio[1][ratio[1].argmax()]
        else:
            ratio1 = ratio[1]
            r_score1 = 1.0
        if self.slimmable:
            self.set_prun_ratio((ratio0, ratio1))
        for w, op in zip(alpha, self._ops):
            latency, size_out = op.forward_latency(size)
            result = result + latency * w * r_score0 * r_score1
        return result, size_out

    def forward_flops(self, size, alpha, beta, ratio):
        result = 0
        if isinstance(ratio[0], torch.Tensor):
            ratio0 = self._width_mult_list_left[ratio[0].argmax()]
            r_score0 = ratio[0][ratio[0].argmax()]
        else:
            ratio0 = ratio[0]
            r_score0 = 1.0
        if isinstance(ratio[1], torch.Tensor):
            ratio1 = self._width_mult_list[ratio[1].argmax()]
            r_score1 = ratio[1][ratio[1].argmax()]
        else:
            ratio1 = ratio[1]
            r_score1 = 1.0
        if self.slimmable:
            self.set_prun_ratio((ratio0, ratio1))
        for w, op in zip(alpha, self._ops):
            if self.quantize == 'search':
                flops_full, size_out = op.forward_flops(size, quantize=False)
                flops_quant, _ = op.forward_flops(size, quantize=True)
                flops = beta[0] * flops_full + beta[1] * flops_quant
            elif self.quantize:
                flops, size_out = op.forward_flops(size, quantize=True)
            else:
                flops, size_out = op.forward_flops(size, quantize=False)
            result = result + flops * w * r_score0 * r_score1
        return result, size_out


class Cell(nn.Module):
    """Residual in Residual Dense Block"""

    def __init__(self, nf=64, op_per_cell=5, slimmable=True, width_mult_list=[1.0], quantize=False):
        super(Cell, self).__init__()
        self.nf = nf
        self.op_per_cell = op_per_cell
        self.slimmable = slimmable
        self._width_mult_list = width_mult_list
        self.quantize = quantize
        self.ops = nn.ModuleList()
        for _ in range(op_per_cell):
            self.ops.append(MixedOp(self.nf, self.nf, slimmable=slimmable, width_mult_list=width_mult_list, quantize=quantize))

    def forward(self, x, alpha, beta, ratio):
        out = x
        for i, op in enumerate(self.ops):
            if i == 0:
                out = op(out, alpha[i], beta[i], [1, ratio[i]])
            elif i == self.op_per_cell - 1:
                out = op(out, alpha[i], beta[i], [ratio[i - 1], 1])
            else:
                out = op(out, alpha[i], beta[i], [ratio[i - 1], ratio[i]])
        return out * 0.2 + x

    def forward_flops(self, size, alpha, beta, ratio):
        flops_total = []
        for i, op in enumerate(self.ops):
            if i == 0:
                flops, size = op.forward_flops(size, alpha[i], beta[i], [1, ratio[i]])
                flops_total.append(flops)
            elif i == self.op_per_cell - 1:
                flops, size = op.forward_flops(size, alpha[i], beta[i], [ratio[i - 1], 1])
                flops_total.append(flops)
            else:
                flops, size = op.forward_flops(size, alpha[i], beta[i], [ratio[i - 1], ratio[i]])
                flops_total.append(flops)
        return sum(flops_total), size


class Conv(nn.Module):
    """
    conv => norm => activation
    use native Conv2d, not slimmable
    """

    def __init__(self, C_in, C_out, kernel_size=3, stride=1, padding=None, dilation=1, groups=1, bias=False, slimmable=False, width_mult_list=[1.0]):
        super(Conv, self).__init__()
        self.C_in = C_in
        self.C_out = C_out
        self.kernel_size = kernel_size
        assert stride in [1, 2]
        self.stride = stride
        if padding is None:
            self.padding = int(np.ceil((dilation * (kernel_size - 1) + 1 - stride) / 2.0))
        else:
            self.padding = padding
        self.dilation = dilation
        assert type(groups) == int
        if kernel_size == 1:
            self.groups = 1
        else:
            self.groups = groups
        self.bias = bias
        self.slimmable = slimmable
        self.width_mult_list = width_mult_list
        self.ratio = 1.0, 1.0
        if slimmable:
            self.conv = USConv2d(C_in, C_out, kernel_size, stride, padding=self.padding, dilation=dilation, groups=self.groups, bias=bias, width_mult_list=width_mult_list)
        else:
            self.conv = Conv2d(C_in, C_out, kernel_size, stride, padding=self.padding, dilation=dilation, groups=self.groups, bias=bias)

    def set_ratio(self, ratio):
        assert self.slimmable
        assert len(ratio) == 2
        self.ratio = ratio
        self.conv.set_ratio(ratio)

    @staticmethod
    def _flops(h, w, C_in, C_out, kernel_size=3, stride=1, padding=None, dilation=1, groups=1, bias=False):
        layer = Conv(C_in, C_out, kernel_size, stride, padding, dilation, groups, bias, slimmable=False)
        flops, params = profile(layer, inputs=(torch.randn(1, C_in, h, w),), custom_ops=custom_ops)
        return flops

    @staticmethod
    def _latency(h, w, C_in, C_out, kernel_size=3, stride=1, padding=None, dilation=1, groups=1, bias=False):
        layer = Conv(C_in, C_out, kernel_size, stride, padding, dilation, groups, bias, slimmable=False)
        latency = compute_latency(layer, (1, C_in, h, w))
        return latency

    def forward_latency(self, size):
        c_in, h_in, w_in = size
        if self.slimmable:
            assert c_in == make_divisible(self.C_in * self.ratio[0]), 'c_in %d, self.C_in * self.ratio[0] %d' % (c_in, self.C_in * self.ratio[0])
            c_out = make_divisible(self.C_out * self.ratio[1])
        else:
            assert c_in == self.C_in, 'c_in %d, self.C_in %d' % (c_in, self.C_in)
            c_out = self.C_out
        if self.stride == 1:
            h_out = h_in
            w_out = w_in
        else:
            h_out = h_in // 2
            w_out = w_in // 2
        name = 'Conv_H%d_W%d_Cin%d_Cout%d_kernel%d_stride%d' % (h_in, w_in, c_in, c_out, self.kernel_size, self.stride)
        if name in latency_lookup_table:
            latency = latency_lookup_table[name]
        else:
            None
            latency = Conv._latency(h_in, w_in, c_in, c_out, self.kernel_size, self.stride, self.padding, self.dilation, self.groups, self.bias)
            latency_lookup_table[name] = latency
            np.save(table_file_name, latency_lookup_table)
        return latency, (c_out, h_out, w_out)

    def forward_flops(self, size, quantize=False):
        c_in, h_in, w_in = size
        if self.slimmable:
            assert c_in == make_divisible(self.C_in * self.ratio[0]), 'c_in %d, self.C_in * self.ratio[0] %d' % (c_in, self.C_in * self.ratio[0])
            c_out = make_divisible(self.C_out * self.ratio[1])
        else:
            assert c_in == self.C_in, 'c_in %d, self.C_in %d' % (c_in, self.C_in)
            c_out = self.C_out
        if self.stride == 1:
            h_out = h_in
            w_out = w_in
        else:
            h_out = h_in // 2
            w_out = w_in // 2
        name = 'Conv_H%d_W%d_Cin%d_Cout%d_kernel%d_stride%d' % (h_in, w_in, c_in, c_out, self.kernel_size, self.stride)
        if name in flops_lookup_table:
            flops = flops_lookup_table[name]
        else:
            None
            flops = Conv._flops(h_in, w_in, c_in, c_out, self.kernel_size, self.stride, self.padding, self.dilation, self.groups, self.bias)
            flops_lookup_table[name] = flops
            np.save(table_file_name, flops_lookup_table)
        return flops, (c_out, h_out, w_out)

    def forward(self, x, quantize=False):
        x = self.conv(x, quantize=quantize)
        return x


class ConvNorm(nn.Module):
    """
    conv => norm => activation
    use native Conv2d, not slimmable
    """

    def __init__(self, C_in, C_out, kernel_size=3, stride=1, padding=None, dilation=1, groups=1, bias=False, slimmable=False, width_mult_list=[1.0]):
        super(ConvNorm, self).__init__()
        self.C_in = C_in
        self.C_out = C_out
        self.kernel_size = kernel_size
        assert stride in [1, 2]
        self.stride = stride
        if padding is None:
            self.padding = int(np.ceil((dilation * (kernel_size - 1) + 1 - stride) / 2.0))
        else:
            self.padding = padding
        self.dilation = dilation
        assert type(groups) == int
        if kernel_size == 1:
            self.groups = 1
        else:
            self.groups = groups
        self.bias = bias
        self.slimmable = slimmable
        self.width_mult_list = width_mult_list
        self.ratio = 1.0, 1.0
        if slimmable:
            self.conv = USConv2d(C_in, C_out, kernel_size, stride, padding=self.padding, dilation=dilation, groups=self.groups, bias=bias, width_mult_list=width_mult_list)
            self.bn = USBatchNorm2d(C_out, width_mult_list)
            self.relu = nn.ReLU(inplace=True)
        else:
            self.conv = Conv2d(C_in, C_out, kernel_size, stride, padding=self.padding, dilation=dilation, groups=self.groups, bias=bias)
            self.bn = BatchNorm2d(C_out)
            self.relu = nn.ReLU(inplace=True)

    def set_ratio(self, ratio):
        assert self.slimmable
        assert len(ratio) == 2
        self.ratio = ratio
        self.conv.set_ratio(ratio)
        self.bn.set_ratio(ratio[1])

    @staticmethod
    def _flops(h, w, C_in, C_out, kernel_size=3, stride=1, padding=None, dilation=1, groups=1, bias=False):
        layer = ConvNorm(C_in, C_out, kernel_size, stride, padding, dilation, groups, bias, slimmable=False)
        flops, params = profile(layer, inputs=(torch.randn(1, C_in, h, w),), custom_ops=custom_ops)
        return flops

    @staticmethod
    def _latency(h, w, C_in, C_out, kernel_size=3, stride=1, padding=None, dilation=1, groups=1, bias=False):
        layer = ConvNorm(C_in, C_out, kernel_size, stride, padding, dilation, groups, bias, slimmable=False)
        latency = compute_latency(layer, (1, C_in, h, w))
        return latency

    def forward_latency(self, size):
        c_in, h_in, w_in = size
        if self.slimmable:
            assert c_in == make_divisible(self.C_in * self.ratio[0]), 'c_in %d, self.C_in * self.ratio[0] %d' % (c_in, self.C_in * self.ratio[0])
            c_out = make_divisible(self.C_out * self.ratio[1])
        else:
            assert c_in == self.C_in, 'c_in %d, self.C_in %d' % (c_in, self.C_in)
            c_out = self.C_out
        if self.stride == 1:
            h_out = h_in
            w_out = w_in
        else:
            h_out = h_in // 2
            w_out = w_in // 2
        name = 'ConvNorm_H%d_W%d_Cin%d_Cout%d_kernel%d_stride%d' % (h_in, w_in, c_in, c_out, self.kernel_size, self.stride)
        if name in latency_lookup_table:
            latency = latency_lookup_table[name]
        else:
            None
            latency = ConvNorm._latency(h_in, w_in, c_in, c_out, self.kernel_size, self.stride, self.padding, self.dilation, self.groups, self.bias)
            latency_lookup_table[name] = latency
            np.save(table_file_name, latency_lookup_table)
        return latency, (c_out, h_out, w_out)

    def forward_flops(self, size, quantize=False):
        c_in, h_in, w_in = size
        if self.slimmable:
            assert c_in == make_divisible(self.C_in * self.ratio[0]), 'c_in %d, self.C_in * self.ratio[0] %d' % (c_in, self.C_in * self.ratio[0])
            c_out = make_divisible(self.C_out * self.ratio[1])
        else:
            assert c_in == self.C_in, 'c_in %d, self.C_in %d' % (c_in, self.C_in)
            c_out = self.C_out
        if self.stride == 1:
            h_out = h_in
            w_out = w_in
        else:
            h_out = h_in // 2
            w_out = w_in // 2
        name = 'ConvNorm_H%d_W%d_Cin%d_Cout%d_kernel%d_stride%d' % (h_in, w_in, c_in, c_out, self.kernel_size, self.stride)
        if name in flops_lookup_table:
            flops = flops_lookup_table[name]
        else:
            None
            flops = ConvNorm._flops(h_in, w_in, c_in, c_out, self.kernel_size, self.stride, self.padding, self.dilation, self.groups, self.bias)
            flops_lookup_table[name] = flops
            np.save(table_file_name, flops_lookup_table)
        return flops, (c_out, h_out, w_out)

    def forward(self, x, quantize=False):
        x = self.conv(x, quantize=quantize)
        x = self.bn(x)
        x = self.relu(x)
        return x


ConvTranspose2d = QConvTranspose2d


class USConvTranspose2d(QConvTranspose2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=2, padding=1, output_padding=1, dilation=1, groups=1, bias=True, width_mult_list=[1.0], num_bits=8, num_bits_weight=8):
        super(USConvTranspose2d, self).__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding, dilation=dilation, groups=groups, bias=bias, num_bits=num_bits, num_bits_weight=num_bits_weight)
        self.in_channels_max = in_channels
        self.out_channels_max = out_channels
        self.width_mult_list = width_mult_list
        self.ratio = 1.0, 1.0

    def set_ratio(self, ratio):
        self.ratio = ratio

    def forward(self, input, quantize=False):
        self.in_channels = make_divisible(self.in_channels_max * self.ratio[0])
        assert self.ratio[1] in self.width_mult_list, str(self.ratio[1]) + ' in? ' + str(self.width_mult_list)
        self.out_channels = make_divisible(self.out_channels_max * self.ratio[1])
        weight = self.weight[:self.in_channels, :self.out_channels, :, :]
        if self.groups != 1:
            self.groups = self.out_channels
        if self.bias is not None:
            bias = self.bias[:self.out_channels]
        else:
            bias = self.bias
        if quantize:
            if not hasattr(self, 'quantize_input'):
                self.quantize_input = QuantMeasure(self.num_bits, shape_measure=(1, 1, 1, 1), flatten_dims=(1, -1))
            qinput = self.quantize_input(input)
            weight_qparams = calculate_qparams(weight, num_bits=self.num_bits_weight, flatten_dims=(1, -1), reduce_dim=None)
            qweight = Quantize(weight, qparams=weight_qparams)
            if self.bias is not None:
                qbias = Quantize(bias, num_bits=self.num_bits_weight + self.num_bits, flatten_dims=(0, -1))
            else:
                qbias = None
            output = F.conv_transpose2d(qinput, qweight, qbias, self.stride, self.padding, self.output_padding, self.groups, self.dilation)
        else:
            output = F.conv_transpose2d(input, weight, bias, self.stride, self.padding, self.output_padding, self.groups, self.dilation)
        return output


class ConvTranspose2dNorm(nn.Module):
    """
    conv => norm => activation
    use native Conv2d, not slimmable
    """

    def __init__(self, C_in, C_out, kernel_size=3, stride=2, padding=None, dilation=1, groups=1, bias=False, slimmable=True, width_mult_list=[1.0]):
        super(ConvTranspose2dNorm, self).__init__()
        self.C_in = C_in
        self.C_out = C_out
        self.kernel_size = kernel_size
        assert stride in [1, 2]
        self.stride = stride
        self.padding = 1
        self.dilation = dilation
        assert type(groups) == int
        if kernel_size == 1:
            self.groups = 1
        else:
            self.groups = groups
        self.bias = bias
        self.slimmable = slimmable
        self.width_mult_list = width_mult_list
        self.ratio = 1.0, 1.0
        if slimmable:
            self.conv = USConvTranspose2d(C_in, C_out, kernel_size, stride, padding=self.padding, output_padding=1, dilation=dilation, groups=self.groups, bias=bias, width_mult_list=width_mult_list)
            self.bn = USBatchNorm2d(C_out, width_mult_list)
            self.relu = nn.ReLU(inplace=True)
        else:
            self.conv = ConvTranspose2d(C_in, C_out, kernel_size, stride, padding=self.padding, output_padding=1, dilation=dilation, groups=self.groups, bias=bias)
            self.bn = BatchNorm2d(C_out)
            self.relu = nn.ReLU(inplace=True)

    def set_ratio(self, ratio):
        assert self.slimmable
        assert len(ratio) == 2
        self.ratio = ratio
        self.conv.set_ratio(ratio)
        self.bn.set_ratio(ratio[1])

    @staticmethod
    def _flops(h, w, C_in, C_out, kernel_size=3, stride=1, padding=None, dilation=1, groups=1, bias=False):
        layer = ConvTranspose2dNorm(C_in, C_out, kernel_size, stride, padding, dilation, groups, bias, slimmable=False)
        flops, params = profile(layer, inputs=(torch.randn(1, C_in, h, w),), custom_ops=custom_ops)
        return flops

    @staticmethod
    def _latency(h, w, C_in, C_out, kernel_size=3, stride=1, padding=None, dilation=1, groups=1, bias=False):
        layer = ConvTranspose2dNorm(C_in, C_out, kernel_size, stride, padding, dilation, groups, bias, slimmable=False)
        latency = compute_latency(layer, (1, C_in, h, w))
        return latency

    def forward_latency(self, size):
        c_in, h_in, w_in = size
        if self.slimmable:
            assert c_in == make_divisible(self.C_in * self.ratio[0]), 'c_in %d, self.C_in * self.ratio[0] %d' % (c_in, self.C_in * self.ratio[0])
            c_out = make_divisible(self.C_out * self.ratio[1])
        else:
            assert c_in == self.C_in, 'c_in %d, self.C_in %d' % (c_in, self.C_in)
            c_out = self.C_out
        if self.stride == 1:
            h_out = h_in
            w_out = w_in
        else:
            h_out = h_in // 2
            w_out = w_in // 2
        name = 'ConvTranspose2dNorm_H%d_W%d_Cin%d_Cout%d_kernel%d_stride%d' % (h_in, w_in, c_in, c_out, self.kernel_size, self.stride)
        if name in latency_lookup_table:
            latency = latency_lookup_table[name]
        else:
            None
            latency = ConvTranspose2dNorm._latency(h_in, w_in, c_in, c_out, self.kernel_size, self.stride, self.padding, self.dilation, self.groups, self.bias)
            latency_lookup_table[name] = latency
            np.save(table_file_name, latency_lookup_table)
        return latency, (c_out, h_out, w_out)

    def forward_flops(self, size, quantize=False):
        c_in, h_in, w_in = size
        if self.slimmable:
            assert c_in == make_divisible(self.C_in * self.ratio[0]), 'c_in %d, self.C_in * self.ratio[0] %d' % (c_in, self.C_in * self.ratio[0])
            c_out = make_divisible(self.C_out * self.ratio[1])
        else:
            assert c_in == self.C_in, 'c_in %d, self.C_in %d' % (c_in, self.C_in)
            c_out = self.C_out
        if self.stride == 1:
            h_out = h_in
            w_out = w_in
        else:
            h_out = h_in // 2
            w_out = w_in // 2
        name = 'ConvTranspose2dNorm_H%d_W%d_Cin%d_Cout%d_kernel%d_stride%d' % (h_in, w_in, c_in, c_out, self.kernel_size, self.stride)
        if name in flops_lookup_table:
            flops = flops_lookup_table[name]
        else:
            None
            flops = ConvTranspose2dNorm._flops(h_in, w_in, c_in, c_out, self.kernel_size, self.stride, self.padding, self.dilation, self.groups, self.bias)
            flops_lookup_table[name] = flops
            np.save(table_file_name, flops_lookup_table)
        return flops, (c_out, h_out, w_out)

    def forward(self, x, quantize=False):
        x = self.conv(x, quantize=quantize)
        x = self.bn(x)
        x = self.relu(x)
        return x


class SingleOp(nn.Module):

    def __init__(self, op, C_in, C_out, kernel_size=3, stride=1, slimmable=True, width_mult_list=[1.0], quantize=False, width_mult_list_left=None):
        super(SingleOp, self).__init__()
        self._op = op(C_in, C_out, kernel_size=kernel_size, stride=stride, slimmable=slimmable, width_mult_list=width_mult_list)
        self._width_mult_list = width_mult_list
        self.quantize = quantize
        self.slimmable = slimmable
        self._width_mult_list_left = width_mult_list_left if width_mult_list_left is not None else width_mult_list

    def set_prun_ratio(self, ratio):
        self._op.set_ratio(ratio)

    def forward(self, x, beta, ratio):
        if isinstance(ratio[0], torch.Tensor):
            ratio0 = self._width_mult_list_left[ratio[0].argmax()]
            r_score0 = ratio[0][ratio[0].argmax()]
        else:
            ratio0 = ratio[0]
            r_score0 = 1.0
        if isinstance(ratio[1], torch.Tensor):
            ratio1 = self._width_mult_list[ratio[1].argmax()]
            r_score1 = ratio[1][ratio[1].argmax()]
        else:
            ratio1 = ratio[1]
            r_score1 = 1.0
        if self.slimmable:
            self.set_prun_ratio((ratio0, ratio1))
        if self.quantize == 'search':
            result = (beta[0] * self._op(x, quantize=False) + beta[1] * self._op(x, quantize=True)) * r_score0 * r_score1
        elif self.quantize:
            result = self._op(x, quantize=True) * r_score0 * r_score1
        else:
            result = self._op(x, quantize=False) * r_score0 * r_score1
        return result

    def forward_flops(self, size, beta, ratio):
        if isinstance(ratio[0], torch.Tensor):
            ratio0 = self._width_mult_list_left[ratio[0].argmax()]
            r_score0 = ratio[0][ratio[0].argmax()]
        else:
            ratio0 = ratio[0]
            r_score0 = 1.0
        if isinstance(ratio[1], torch.Tensor):
            ratio1 = self._width_mult_list[ratio[1].argmax()]
            r_score1 = ratio[1][ratio[1].argmax()]
        else:
            ratio1 = ratio[1]
            r_score1 = 1.0
        if self.slimmable:
            self.set_prun_ratio((ratio0, ratio1))
        if self.quantize == 'search':
            flops_full, size_out = self._op.forward_flops(size, quantize=False)
            flops_quant, _ = self._op.forward_flops(size, quantize=True)
            flops = beta[0] * flops_full + beta[1] * flops_quant
        elif self.quantize:
            flops, size_out = self._op.forward_flops(size, quantize=True)
        else:
            flops, size_out = self._op.forward_flops(size, quantize=False)
        flops = flops * r_score0 * r_score1
        return flops, size_out


class NAS_GAN_Eval(nn.Module):

    def __init__(self, alpha, beta, ratio, beta_sh, ratio_sh, layers=16, width_mult_list=[1.0], width_mult_list_sh=[1.0], quantize=True):
        super(NAS_GAN_Eval, self).__init__()
        assert layers >= 3
        self._layers = layers
        self._width_mult_list = width_mult_list
        self._width_mult_list_sh = width_mult_list_sh
        self._flops = 0
        self._params = 0
        self.len_stem = 3
        self.len_header = 3
        self.len_beta_sh = self.len_stem + self.len_header
        self.len_ratio_sh = self.len_stem + self.len_header - 1
        op_idx_list = F.softmax(alpha, dim=-1).argmax(-1)
        if quantize == 'search':
            quantize_list = F.softmax(beta, dim=-1).argmax(-1) == 1
            quantize_list_sh = F.softmax(beta_sh, dim=-1).argmax(-1) == 1
        elif quantize:
            quantize_list = [(True) for _ in range(layers)]
            quantize_list_sh = [(True) for _ in range(beta_sh.size(0))]
        else:
            quantize_list = [(False) for _ in range(layers)]
            quantize_list_sh = [(False) for _ in range(beta_sh.size(0))]
        ratio_list = F.softmax(ratio, dim=-1).argmax(-1)
        ratio_list_sh = F.softmax(ratio_sh, dim=-1).argmax(-1)
        self.stem = nn.ModuleList()
        self.stem.append(SingleOp(ConvNorm, 3, make_divisible(64 * width_mult_list_sh[ratio_list_sh[0]]), 7, quantize=quantize_list_sh[0]))
        in_features = 64
        out_features = in_features * 2
        for i in range(2):
            self.stem.append(SingleOp(ConvNorm, make_divisible(in_features * width_mult_list_sh[ratio_list_sh[i]]), make_divisible(out_features * width_mult_list_sh[ratio_list_sh[i + 1]]), 3, stride=2, quantize=quantize_list_sh[1 + i]))
            in_features = out_features
            out_features = in_features * 2
        self.cells = nn.ModuleList()
        for i in range(layers):
            if i == 0:
                self.cells.append(MixedOp(make_divisible(in_features * width_mult_list_sh[ratio_list_sh[self.len_stem - 1]]), make_divisible(in_features * width_mult_list[ratio_list[i]]), op_idx_list[i], quantize_list[i]))
            else:
                self.cells.append(MixedOp(make_divisible(in_features * width_mult_list[ratio_list[i - 1]]), make_divisible(in_features * width_mult_list[ratio_list[i]]), op_idx_list[i], quantize_list[i]))
        self.header = nn.ModuleList()
        out_features = in_features // 2
        self.header.append(SingleOp(ConvTranspose2dNorm, make_divisible(in_features * width_mult_list[ratio_list[self._layers - 1]]), make_divisible(out_features * width_mult_list_sh[ratio_list_sh[self.len_stem]]), 3, stride=2, quantize=quantize_list_sh[self.len_stem]))
        in_features = out_features
        out_features = in_features // 2
        self.header.append(SingleOp(ConvTranspose2dNorm, make_divisible(in_features * width_mult_list_sh[ratio_list_sh[self.len_stem]]), make_divisible(out_features * width_mult_list_sh[ratio_list_sh[self.len_stem + 1]]), 3, stride=2, quantize=quantize_list_sh[self.len_stem + 1]))
        self.header.append(SingleOp(Conv, make_divisible(64 * width_mult_list_sh[ratio_list_sh[self.len_stem + 1]]), 3, 7, quantize=quantize_list_sh[self.len_stem + 2]))
        self.tanh = nn.Tanh()

    def forward(self, input):
        out = input
        for i, module in enumerate(self.stem):
            out = module(out)
        for i, cell in enumerate(self.cells):
            out = cell(out)
        for i, module in enumerate(self.header):
            out = module(out)
        out = self.tanh(out)
        return out

    def forward_flops(self, size):
        flops_total = []
        for i, module in enumerate(self.stem):
            flops, size = module.forward_flops(size)
            flops_total.append(flops)
        for i, cell in enumerate(self.cells):
            flops, size = cell.forward_flops(size)
            flops_total.append(flops)
        for i, module in enumerate(self.header):
            flops, size = module.forward_flops(size)
            flops_total.append(flops)
        return sum(flops_total)


class Vgg16(nn.Module):

    def __init__(self):
        super(Vgg16, self).__init__()
        features = models.vgg16(pretrained=True).features
        self.to_relu_1_2 = nn.Sequential()
        self.to_relu_2_2 = nn.Sequential()
        self.to_relu_3_3 = nn.Sequential()
        self.to_relu_4_3 = nn.Sequential()
        for x in range(4):
            self.to_relu_1_2.add_module(str(x), features[x])
        for x in range(4, 9):
            self.to_relu_2_2.add_module(str(x), features[x])
        for x in range(9, 16):
            self.to_relu_3_3.add_module(str(x), features[x])
        for x in range(16, 23):
            self.to_relu_4_3.add_module(str(x), features[x])
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        h = self.to_relu_1_2(x)
        h_relu_1_2 = h
        h = self.to_relu_2_2(h)
        h_relu_2_2 = h
        h = self.to_relu_3_3(h)
        h_relu_3_3 = h
        h = self.to_relu_4_3(h)
        h_relu_4_3 = h
        out = h_relu_1_2, h_relu_2_2, h_relu_3_3, h_relu_4_3
        return out


class VGGFeature(nn.Module):

    def __init__(self):
        super(VGGFeature, self).__init__()
        self.add_module('vgg', Vgg16())

    def __call__(self, x):
        x = (x.clone() + 1.0) / 2.0
        x_vgg = self.vgg(x)
        return x_vgg


class NAS_GAN_Infer(nn.Module):

    def __init__(self, alpha, beta, ratio, beta_sh, ratio_sh, layers=16, width_mult_list=[1.0], width_mult_list_sh=[1.0], loss_weight=[1.0, 100000.0, 1.0, 1e-07], quantize=True):
        super(NAS_GAN_Infer, self).__init__()
        assert layers >= 3
        self._layers = layers
        self._width_mult_list = width_mult_list
        self._width_mult_list_sh = width_mult_list_sh
        self._flops = 0
        self._params = 0
        self.len_stem = 3
        self.len_header = 3
        self.len_beta_sh = self.len_stem + self.len_header
        self.len_ratio_sh = self.len_stem + self.len_header - 1
        self.base_weight = loss_weight[0]
        self.style_weight = loss_weight[1]
        self.content_weight = loss_weight[2]
        self.tv_weight = loss_weight[3]
        op_idx_list = F.softmax(alpha, dim=-1).argmax(-1)
        if quantize == 'search':
            quantize_list = F.softmax(beta, dim=-1).argmax(-1) == 1
            quantize_list_sh = F.softmax(beta_sh, dim=-1).argmax(-1) == 1
        elif quantize:
            quantize_list = [(True) for _ in range(layers)]
            quantize_list_sh = [(True) for _ in range(beta_sh.size(0))]
        else:
            quantize_list = [(False) for _ in range(layers)]
            quantize_list_sh = [(False) for _ in range(beta_sh.size(0))]
        ratio_list = F.softmax(ratio, dim=-1).argmax(-1)
        ratio_list_sh = F.softmax(ratio_sh, dim=-1).argmax(-1)
        self.vgg = torch.nn.DataParallel(VGGFeature())
        self.stem = nn.ModuleList()
        self.stem.append(SingleOp(ConvNorm, 3, make_divisible(64 * width_mult_list_sh[ratio_list_sh[0]]), 7, quantize=quantize_list_sh[0]))
        in_features = 64
        out_features = in_features * 2
        for i in range(2):
            self.stem.append(SingleOp(ConvNorm, make_divisible(in_features * width_mult_list_sh[ratio_list_sh[i]]), make_divisible(out_features * width_mult_list_sh[ratio_list_sh[i + 1]]), 3, stride=2, quantize=quantize_list_sh[1 + i]))
            in_features = out_features
            out_features = in_features * 2
        self.cells = nn.ModuleList()
        for i in range(layers):
            if i == 0:
                self.cells.append(MixedOp(make_divisible(in_features * width_mult_list_sh[ratio_list_sh[self.len_stem - 1]]), make_divisible(in_features * width_mult_list[ratio_list[i]]), op_idx_list[i], quantize_list[i]))
            else:
                self.cells.append(MixedOp(make_divisible(in_features * width_mult_list[ratio_list[i - 1]]), make_divisible(in_features * width_mult_list[ratio_list[i]]), op_idx_list[i], quantize_list[i]))
        self.header = nn.ModuleList()
        out_features = in_features // 2
        self.header.append(SingleOp(ConvTranspose2dNorm, make_divisible(in_features * width_mult_list[ratio_list[self._layers - 1]]), make_divisible(out_features * width_mult_list_sh[ratio_list_sh[self.len_stem]]), 3, stride=2, quantize=quantize_list_sh[self.len_stem]))
        in_features = out_features
        out_features = in_features // 2
        self.header.append(SingleOp(ConvTranspose2dNorm, make_divisible(in_features * width_mult_list_sh[ratio_list_sh[self.len_stem]]), make_divisible(out_features * width_mult_list_sh[ratio_list_sh[self.len_stem + 1]]), 3, stride=2, quantize=quantize_list_sh[self.len_stem + 1]))
        self.header.append(SingleOp(Conv, make_divisible(64 * width_mult_list_sh[ratio_list_sh[self.len_stem + 1]]), 3, 7, quantize=quantize_list_sh[self.len_stem + 2]))
        self.tanh = nn.Tanh()

    def forward(self, input):
        out = input
        for i, module in enumerate(self.stem):
            out = module(out)
        for i, cell in enumerate(self.cells):
            out = cell(out)
        for i, module in enumerate(self.header):
            out = module(out)
        out = self.tanh(out)
        return out

    def forward_flops(self, size):
        flops_total = []
        for i, module in enumerate(self.stem):
            flops, size = module.forward_flops(size)
            flops_total.append(flops)
        for i, cell in enumerate(self.cells):
            flops, size = cell.forward_flops(size)
            flops_total.append(flops)
        for i, module in enumerate(self.header):
            flops, size = module.forward_flops(size)
            flops_total.append(flops)
        return sum(flops_total)

    def gram(self, x):
        bs, ch, h, w = x.size()
        f = x.view(bs, ch, w * h)
        f_T = f.transpose(1, 2)
        G = f.bmm(f_T) / (ch * h * w)
        return G

    def _criterion(self, y_hat, x):
        base_loss = self.base_weight * nn.L1Loss()(y_hat, x)
        y_c_features = self.vgg(x)
        y_hat_features = self.vgg(y_hat)
        y_hat_gram = [self.gram(fmap) for fmap in y_hat_features]
        x_gram = [self.gram(fmap) for fmap in y_c_features]
        style_loss = 0
        for j in range(4):
            style_loss += self.style_weight * nn.functional.mse_loss(y_hat_gram[j], x_gram[j])
        recon = y_c_features[1]
        recon_hat = y_hat_features[1]
        content_loss = self.content_weight * nn.L1Loss()(recon_hat, recon)
        diff_i = torch.sum(torch.abs(y_hat[:, :, :, 1:] - y_hat[:, :, :, :-1]))
        diff_j = torch.sum(torch.abs(y_hat[:, :, 1:, :] - y_hat[:, :, :-1, :]))
        tv_loss = self.tv_weight * (diff_i + diff_j)
        total_loss = base_loss + style_loss + content_loss + tv_loss
        return total_loss

    def _loss(self, input, target):
        logit = self(input)
        loss = self._criterion(logit, target)
        return loss


def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape)
    U = U
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature=1):
    y = logits + sample_gumbel(logits.size())
    return F.softmax(y / temperature, dim=-1)


def gumbel_softmax(logits, temperature=1, hard=False):
    """
    ST-gumple-softmax
    input: [*, n_class]
    return: flatten --> [*, n_class] an one-hot vector
    """
    y = gumbel_softmax_sample(logits, temperature)
    if not hard:
        return y
    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    y_hard = (y_hard - y).detach() + y
    return y_hard


class NAS_GAN(nn.Module):

    def __init__(self, layers=16, slimmable=True, width_mult_list=[1.0], width_mult_list_sh=[1.0], loss_weight=[1.0, 100000.0, 1.0, 1e-07], prun_modes='arch_ratio', quantize=False):
        super(NAS_GAN, self).__init__()
        assert layers >= 3
        self._layers = layers
        self._width_mult_list = width_mult_list
        self._width_mult_list_sh = width_mult_list_sh
        self._prun_modes = prun_modes
        self.prun_mode = None
        self._flops = 0
        self._params = 0
        self.len_stem = 3
        self.len_header = 3
        self.len_beta_sh = self.len_stem + self.len_header
        self.len_ratio_sh = self.len_stem + self.len_header - 1
        self.base_weight = loss_weight[0]
        self.style_weight = loss_weight[1]
        self.content_weight = loss_weight[2]
        self.tv_weight = loss_weight[3]
        self.vgg = torch.nn.DataParallel(VGGFeature())
        self.quantize = quantize
        self.slimmable = slimmable
        self.stem = nn.ModuleList()
        self.stem.append(SingleOp(ConvNorm, 3, 64, 7, slimmable=slimmable, width_mult_list=width_mult_list_sh, quantize=quantize))
        in_features = 64
        out_features = in_features * 2
        for _ in range(2):
            self.stem.append(SingleOp(ConvNorm, in_features, out_features, 3, stride=2, slimmable=slimmable, width_mult_list=width_mult_list_sh, quantize=quantize))
            in_features = out_features
            out_features = in_features * 2
        self.cells = nn.ModuleList()
        for i in range(layers):
            if i == 0:
                op = MixedOp(in_features, in_features, slimmable=slimmable, width_mult_list=width_mult_list, quantize=quantize, width_mult_list_left=width_mult_list_sh)
            else:
                op = MixedOp(in_features, in_features, slimmable=slimmable, width_mult_list=width_mult_list, quantize=quantize)
            self.cells.append(op)
        self.header = nn.ModuleList()
        out_features = in_features // 2
        self.header.append(SingleOp(ConvTranspose2dNorm, in_features, out_features, 3, stride=2, slimmable=slimmable, width_mult_list=width_mult_list_sh, quantize=quantize, width_mult_list_left=width_mult_list))
        in_features = out_features
        out_features = in_features // 2
        self.header.append(SingleOp(ConvTranspose2dNorm, in_features, out_features, 3, stride=2, slimmable=slimmable, width_mult_list=width_mult_list_sh, quantize=quantize))
        self.header.append(SingleOp(Conv, 64, 3, 7, slimmable=slimmable, width_mult_list=width_mult_list_sh, quantize=quantize))
        self.tanh = nn.Tanh()
        self._arch_params = self._build_arch_parameters()
        self._reset_arch_parameters()

    def sample_prun_ratio(self, mode='arch_ratio'):
        """
        mode: "min"|"max"|"random"|"arch_ratio"(default)
        """
        assert mode in ['min', 'max', 'random', 'arch_ratio']
        if mode == 'arch_ratio':
            ratio = self._arch_params['ratio']
            ratio_sampled = []
            for layer in range(self._layers):
                ratio_sampled.append(gumbel_softmax(F.log_softmax(ratio[layer], dim=-1), hard=True))
            ratio_sh = self._arch_params['ratio_sh']
            ratio_sampled_sh = []
            for layer in range(self.len_ratio_sh):
                ratio_sampled_sh.append(gumbel_softmax(F.log_softmax(ratio_sh[layer], dim=-1), hard=True))
            return ratio_sampled, ratio_sampled_sh
        elif mode == 'min':
            ratio_sampled = []
            for layer in range(self._layers):
                ratio_sampled.append(self._width_mult_list[0])
            ratio_sampled_sh = []
            for layer in range(self.len_ratio_sh):
                ratio_sampled_sh.append(self._width_mult_list_sh[0])
            return ratio_sampled, ratio_sampled_sh
        elif mode == 'max':
            ratio_sampled = []
            for layer in range(self._layers):
                ratio_sampled.append(self._width_mult_list[-1])
            ratio_sampled_sh = []
            for layer in range(self.len_ratio_sh):
                ratio_sampled_sh.append(self._width_mult_list_sh[-1])
            return ratio_sampled, ratio_sampled_sh
        elif mode == 'random':
            ratio_sampled = []
            for layer in range(self._layers):
                ratio_sampled.append(np.random.choice(self._width_mult_list))
            ratio_sampled_sh = []
            for layer in range(self.len_ratio_sh):
                ratio_sampled_sh.append(np.random.choice(self._width_mult_list_sh))
            return ratio_sampled, ratio_sampled_sh

    def forward(self, input):
        alpha = F.softmax(getattr(self, 'alpha'), dim=-1)
        beta = F.softmax(getattr(self, 'beta'), dim=-1)
        beta_sh = F.softmax(getattr(self, 'beta_sh'), dim=-1)
        if self.prun_mode is not None:
            ratio, ratio_sh = self.sample_prun_ratio(mode=self.prun_mode)
        else:
            ratio, ratio_sh = self.sample_prun_ratio(mode=self._prun_modes)
        out = input
        for i, module in enumerate(self.stem):
            if i == 0:
                out = module(out, beta_sh[i], [1, ratio_sh[i]])
            else:
                out = module(out, beta_sh[i], [ratio_sh[i - 1], ratio_sh[i]])
        for i, cell in enumerate(self.cells):
            if i == 0:
                out = cell(out, alpha[i], beta[i], [ratio_sh[self.len_stem - 1], ratio[i]])
            else:
                out = cell(out, alpha[i], beta[i], [ratio[i - 1], ratio[i]])
        for i, module in enumerate(self.header):
            if i == 0:
                out = module(out, beta_sh[self.len_stem + i], [ratio[self._layers - 1], ratio_sh[self.len_stem + i]])
            elif i == self.len_header - 1:
                out = module(out, beta_sh[self.len_stem + i], [ratio_sh[self.len_stem + i - 1], 1])
            else:
                out = module(out, beta_sh[self.len_stem + i], [ratio_sh[self.len_stem + i - 1], ratio_sh[self.len_stem + i]])
        out = self.tanh(out)
        return out

    def forward_flops(self, size, alpha=True, beta=True, ratio=True):
        if alpha:
            alpha = F.softmax(getattr(self, 'alpha'), dim=-1)
        else:
            alpha = torch.ones_like(getattr(self, 'alpha')) * 1.0 / len(PRIMITIVES)
        if beta:
            beta = F.softmax(getattr(self, 'beta'), dim=-1)
            beta_sh = F.softmax(getattr(self, 'beta_sh'), dim=-1)
        else:
            beta = torch.ones_like(getattr(self, 'beta')) * 1.0 / 2
            beta_sh = torch.ones_like(getattr(self, 'beta_sh')) * 1.0 / 2
        if ratio:
            if self.prun_mode is not None:
                ratio, ratio_sh = self.sample_prun_ratio(mode=self.prun_mode)
            else:
                ratio, ratio_sh = self.sample_prun_ratio(mode=self._prun_modes)
        else:
            ratio, ratio_sh = self.sample_prun_ratio(mode='max')
        flops_total = []
        for i, module in enumerate(self.stem):
            if i == 0:
                flops, size = module.forward_flops(size, beta_sh[i], [1, ratio_sh[i]])
                flops_total.append(flops)
            else:
                flops, size = module.forward_flops(size, beta_sh[i], [ratio_sh[i - 1], ratio_sh[i]])
                flops_total.append(flops)
        for i, cell in enumerate(self.cells):
            if i == 0:
                flops, size = cell.forward_flops(size, alpha[i], beta[i], [ratio_sh[self.len_stem - 1], ratio[i]])
                flops_total.append(flops)
            else:
                flops, size = cell.forward_flops(size, alpha[i], beta[i], [ratio[i - 1], ratio[i]])
                flops_total.append(flops)
        for i, module in enumerate(self.header):
            if i == 0:
                flops, size = module.forward_flops(size, beta_sh[self.len_stem + i], [ratio[self._layers - 1], ratio_sh[self.len_stem + i]])
                flops_total.append(flops)
            elif i == self.len_header - 1:
                flops, size = module.forward_flops(size, beta_sh[self.len_stem + i], [ratio_sh[self.len_stem + i - 1], 1])
                flops_total.append(flops)
            else:
                flops, size = module.forward_flops(size, beta_sh[self.len_stem + i], [ratio_sh[self.len_stem + i - 1], ratio_sh[self.len_stem + i]])
                flops_total.append(flops)
        return sum(flops_total)

    def gram(self, x):
        bs, ch, h, w = x.size()
        f = x.view(bs, ch, w * h)
        f_T = f.transpose(1, 2)
        G = f.bmm(f_T) / (ch * h * w)
        return G

    def _criterion(self, y_hat, x):
        base_loss = self.base_weight * nn.L1Loss()(y_hat, x)
        y_c_features = self.vgg(x)
        y_hat_features = self.vgg(y_hat)
        y_hat_gram = [self.gram(fmap) for fmap in y_hat_features]
        x_gram = [self.gram(fmap) for fmap in y_c_features]
        style_loss = 0
        for j in range(4):
            style_loss += self.style_weight * nn.functional.mse_loss(y_hat_gram[j], x_gram[j])
        recon = y_c_features[1]
        recon_hat = y_hat_features[1]
        content_loss = self.content_weight * nn.L1Loss()(recon_hat, recon)
        diff_i = torch.sum(torch.abs(y_hat[:, :, :, 1:] - y_hat[:, :, :, :-1]))
        diff_j = torch.sum(torch.abs(y_hat[:, :, 1:, :] - y_hat[:, :, :-1, :]))
        tv_loss = self.tv_weight * (diff_i + diff_j)
        total_loss = base_loss + style_loss + content_loss + tv_loss
        return total_loss

    def _loss(self, input, target, pretrain=False):
        loss = 0
        if pretrain is not True:
            self.prun_mode = None
            logit = self(input)
            loss = loss + self._criterion(logit, target)
        if len(self._width_mult_list) > 1:
            self.prun_mode = 'max'
            logit = self(input)
            loss = loss + self._criterion(logit, target)
            self.prun_mode = 'min'
            logit = self(input)
            loss = loss + self._criterion(logit, target)
            if pretrain == True:
                self.prun_mode = 'random'
                logit = self(input)
                loss = loss + self._criterion(logit, target)
                self.prun_mode = 'random'
                logit = self(input)
                loss = loss + self._criterion(logit, target)
        elif pretrain == True and len(self._width_mult_list) == 1:
            self.prun_mode = 'max'
            logit = self(input)
            loss = loss + self._criterion(logit, target)
        return loss

    def _build_arch_parameters(self):
        num_ops = len(PRIMITIVES)
        setattr(self, 'alpha', nn.Parameter(Variable(0.001 * torch.ones(self._layers, num_ops), requires_grad=True)))
        setattr(self, 'beta', nn.Parameter(Variable(0.001 * torch.ones(self._layers, 2), requires_grad=True)))
        if self._prun_modes == 'arch_ratio':
            num_widths = len(self._width_mult_list)
            num_widths_sh = len(self._width_mult_list_sh)
        else:
            num_widths = 1
            num_widths_sh = 1
        setattr(self, 'ratio', nn.Parameter(Variable(0.001 * torch.ones(self._layers, num_widths), requires_grad=True)))
        setattr(self, 'beta_sh', nn.Parameter(Variable(0.001 * torch.ones(self.len_beta_sh, 2), requires_grad=True)))
        setattr(self, 'ratio_sh', nn.Parameter(Variable(0.001 * torch.ones(self.len_ratio_sh, num_widths_sh), requires_grad=True)))
        return {'alpha': self.alpha, 'beta': self.beta, 'ratio': self.ratio, 'beta_sh': self.beta, 'ratio_sh': self.ratio_sh}

    def _reset_arch_parameters(self):
        num_ops = len(PRIMITIVES)
        if self._prun_modes == 'arch_ratio':
            num_widths = len(self._width_mult_list)
            num_widths_sh = len(self._width_mult_list_sh)
        else:
            num_widths = 1
            num_widths_sh = 1
        getattr(self, 'alpha').data = Variable(0.001 * torch.ones(self._layers, num_ops), requires_grad=True)
        getattr(self, 'beta').data = Variable(0.001 * torch.ones(self._layers, 2), requires_grad=True)
        getattr(self, 'ratio').data = Variable(0.001 * torch.ones(self._layers, num_widths), requires_grad=True)
        getattr(self, 'beta_sh').data = Variable(0.001 * torch.ones(self.len_beta_sh, 2), requires_grad=True)
        getattr(self, 'ratio_sh').data = Variable(0.001 * torch.ones(self.len_ratio_sh, num_widths_sh), requires_grad=True)


class Conv7x7(nn.Module):

    def __init__(self, C_in, C_out, kernel_size=7, stride=1, dilation=1, groups=1, slimmable=True, width_mult_list=[1.0]):
        super(Conv7x7, self).__init__()
        self.C_in = C_in
        self.C_out = C_out
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.slimmable = slimmable
        self.width_mult_list = width_mult_list
        assert stride in [1, 2]
        if self.stride == 2:
            self.dilation = 1
        self.ratio = 1.0, 1.0
        if slimmable:
            self.conv1 = USConv2d(C_in, C_out, 7, stride, padding=3, dilation=dilation, groups=groups, bias=False, width_mult_list=width_mult_list)
        else:
            self.conv1 = Conv2d(C_in, C_out, 7, stride, padding=3, dilation=dilation, groups=groups, bias=False)

    def set_ratio(self, ratio):
        assert len(ratio) == 2
        self.ratio = ratio
        self.conv1.set_ratio(ratio)

    @staticmethod
    def _flops(h, w, C_in, C_out, kernel_size=7, stride=1, dilation=1, groups=1):
        layer = Conv7x7(C_in, C_out, kernel_size, stride, dilation, groups, slimmable=False)
        flops, params = profile(layer, inputs=(torch.randn(1, C_in, h, w),), custom_ops=custom_ops)
        return flops

    @staticmethod
    def _latency(h, w, C_in, C_out, kernel_size=7, stride=1, dilation=1, groups=1):
        layer = Conv7x7(C_in, C_out, kernel_size, stride, dilation, groups, slimmable=False)
        latency = compute_latency(layer, (1, C_in, h, w))
        return latency

    def forward_latency(self, size):
        c_in, h_in, w_in = size
        if self.slimmable:
            assert c_in == make_divisible(self.C_in * self.ratio[0]), 'c_in %d, int(self.C_in * self.ratio[0]) %d' % (c_in, int(self.C_in * self.ratio[0]))
            c_out = make_divisible(self.C_out * self.ratio[1])
        else:
            assert c_in == self.C_in, 'c_in %d, self.C_in %d' % (c_in, self.C_in)
            c_out = self.C_out
        if self.stride == 1:
            h_out = h_in
            w_out = w_in
        else:
            h_out = h_in // 2
            w_out = w_in // 2
        name = 'Conv7x7_H%d_W%d_Cin%d_Cout%d_stride%d_dilation%d' % (h_in, w_in, c_in, c_out, self.stride, self.dilation)
        if name in latency_lookup_table:
            latency = latency_lookup_table[name]
        else:
            None
            latency = Conv7x7._latency(h_in, w_in, c_in, c_out, self.kernel_size, self.stride, self.dilation, self.groups)
            latency_lookup_table[name] = latency
            np.save(table_file_name, latency_lookup_table)
        return latency, (c_out, h_out, w_out)

    def forward_flops(self, size, quantize=False):
        c_in, h_in, w_in = size
        if self.slimmable:
            assert c_in == make_divisible(self.C_in * self.ratio[0]), 'c_in %d, self.C_in * self.ratio[0] %d' % (c_in, self.C_in * self.ratio[0])
            c_out = make_divisible(self.C_out * self.ratio[1])
        else:
            assert c_in == self.C_in, 'c_in %d, self.C_in %d' % (c_in, self.C_in)
            c_out = self.C_out
        if self.stride == 1:
            h_out = h_in
            w_out = w_in
        else:
            h_out = h_in // 2
            w_out = w_in // 2
        name = 'Conv7x7_H%d_W%d_Cin%d_Cout%d_kernel%d_stride%d' % (h_in, w_in, c_in, c_out, self.kernel_size, self.stride)
        if name in flops_lookup_table:
            flops = flops_lookup_table[name]
        else:
            None
            flops = Conv7x7._flops(h_in, w_in, c_in, c_out, self.kernel_size, self.stride, self.dilation, self.groups)
            flops_lookup_table[name] = flops
            np.save(table_file_name, flops_lookup_table)
        return flops, (c_out, h_out, w_out)

    def forward(self, x, quantize=False):
        identity = x
        out = self.conv1(x, quantize=quantize)
        return out


_DEFAULT_FLATTEN_GRAD = 0, -1


class UniformQuantizeGrad(InplaceFunction):

    @staticmethod
    def forward(ctx, input, num_bits=None, qparams=None, flatten_dims=_DEFAULT_FLATTEN_GRAD, reduce_dim=0, dequantize=True, signed=False, stochastic=True):
        ctx.num_bits = num_bits
        ctx.qparams = qparams
        ctx.flatten_dims = flatten_dims
        ctx.stochastic = stochastic
        ctx.signed = signed
        ctx.dequantize = dequantize
        ctx.reduce_dim = reduce_dim
        ctx.inplace = False
        return input

    @staticmethod
    def backward(ctx, grad_output):
        qparams = ctx.qparams
        with torch.no_grad():
            if qparams is None:
                assert ctx.num_bits is not None, 'either provide qparams of num_bits to quantize'
                qparams = calculate_qparams(grad_output, num_bits=ctx.num_bits, flatten_dims=ctx.flatten_dims, reduce_dim=ctx.reduce_dim, reduce_type='extreme')
            grad_input = Quantize(grad_output, num_bits=None, qparams=qparams, flatten_dims=ctx.flatten_dims, reduce_dim=ctx.reduce_dim, dequantize=True, signed=ctx.signed, stochastic=ctx.stochastic, inplace=False)
        return grad_input, None, None, None, None, None, None, None


def quantize_grad(x, num_bits=None, qparams=None, flatten_dims=_DEFAULT_FLATTEN_GRAD, reduce_dim=0, dequantize=True, signed=False, stochastic=True):
    return UniformQuantizeGrad().apply(x, num_bits, qparams, flatten_dims, reduce_dim, dequantize, signed, stochastic)


def linear_biprec(input, weight, bias=None, num_bits_grad=None):
    out1 = F.linear(input.detach(), weight, bias)
    out2 = F.linear(input, weight.detach(), bias.detach() if bias is not None else None)
    out2 = quantize_grad(out2, num_bits=num_bits_grad)
    return out1 + out2 - out1.detach()


class QLinear(nn.Linear):
    """docstring for QConv2d."""

    def __init__(self, in_features, out_features, bias=True, num_bits=8, num_bits_weight=8, num_bits_grad=8, biprecision=True):
        super(QLinear, self).__init__(in_features, out_features, bias)
        self.num_bits = num_bits
        self.num_bits_weight = num_bits_weight or num_bits
        self.num_bits_grad = num_bits_grad
        self.biprecision = biprecision
        self.quantize_input = QuantMeasure(self.num_bits)

    def forward(self, input):
        qinput = self.quantize_input(input)
        weight_qparams = calculate_qparams(self.weight, num_bits=self.num_bits_weight, flatten_dims=(1, -1), reduce_dim=None)
        qweight = Quantize(self.weight, qparams=weight_qparams)
        if self.bias is not None:
            qbias = Quantize(self.bias, num_bits=self.num_bits_weight + self.num_bits, flatten_dims=(0, -1))
        else:
            qbias = None
        if not self.biprecision or self.num_bits_grad is None:
            output = F.linear(qinput, qweight, qbias)
            if self.num_bits_grad is not None:
                output = quantize_grad(output, num_bits=self.num_bits_grad)
        else:
            output = linear_biprec(qinput, qweight, qbias, self.num_bits_grad)
        return output


class RangeBN(nn.Module):

    def __init__(self, num_features, dim=1, momentum=0.1, affine=True, num_chunks=16, eps=1e-05, num_bits=8, num_bits_grad=8):
        super(RangeBN, self).__init__()
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.zeros(num_features))
        self.momentum = momentum
        self.dim = dim
        if affine:
            self.bias = nn.Parameter(torch.Tensor(num_features))
            self.weight = nn.Parameter(torch.Tensor(num_features))
        self.num_bits = num_bits
        self.num_bits_grad = num_bits_grad
        self.quantize_input = QuantMeasure(self.num_bits, inplace=True, shape_measure=(1, 1, 1, 1), flatten_dims=(1, -1))
        self.eps = eps
        self.num_chunks = num_chunks
        self.reset_params()

    def reset_params(self):
        if self.weight is not None:
            self.weight.data.uniform_()
        if self.bias is not None:
            self.bias.data.zero_()

    def forward(self, x):
        x = self.quantize_input(x)
        if x.dim() == 2:
            x = x.unsqueeze(-1).unsqueeze(-1)
        if self.training:
            B, C, H, W = x.shape
            y = x.transpose(0, 1).contiguous()
            y = y.view(C, self.num_chunks, B * H * W // self.num_chunks)
            mean_max = y.max(-1)[0].mean(-1)
            mean_min = y.min(-1)[0].mean(-1)
            mean = y.view(C, -1).mean(-1)
            scale_fix = 0.5 * 0.35 * (1 + (math.pi * math.log(4)) ** 0.5) / (2 * math.log(y.size(-1))) ** 0.5
            scale = (mean_max - mean_min) * scale_fix
            with torch.no_grad():
                self.running_mean.mul_(self.momentum).add_(mean * (1 - self.momentum))
                self.running_var.mul_(self.momentum).add_(scale * (1 - self.momentum))
        else:
            mean = self.running_mean
            scale = self.running_var
        out = (x - mean.view(1, -1, 1, 1)) / (scale.view(1, -1, 1, 1) + self.eps)
        if self.weight is not None:
            qweight = self.weight
            out = out * qweight.view(1, -1, 1, 1)
        if self.bias is not None:
            qbias = self.bias
            out = out + qbias.view(1, -1, 1, 1)
        if self.num_bits_grad is not None:
            out = quantize_grad(out, num_bits=self.num_bits_grad, flatten_dims=(1, -1))
        if out.size(3) == 1 and out.size(2) == 1:
            out = out.squeeze(-1).squeeze(-1)
        return out


class RangeBN1d(RangeBN):

    def __init__(self, num_features, dim=1, momentum=0.1, affine=True, num_chunks=16, eps=1e-05, num_bits=8, num_bits_grad=8):
        super(RangeBN1d, self).__init__(num_features, dim, momentum, affine, num_chunks, eps, num_bits, num_bits_grad)
        self.quantize_input = QuantMeasure(self.num_bits, inplace=True, shape_measure=(1, 1), flatten_dims=(1, -1))


class ResidualBlock(nn.Module):

    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        conv_block = [nn.ReflectionPad2d(1), nn.Conv2d(in_features, in_features, 3), nn.InstanceNorm2d(in_features, affine=True), nn.ReLU(inplace=True), nn.ReflectionPad2d(1), nn.Conv2d(in_features, in_features, 3), nn.InstanceNorm2d(in_features)]
        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)


class Generator(nn.Module):

    def __init__(self, input_nc, output_nc, n_residual_blocks=9):
        super(Generator, self).__init__()
        model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, 64, 7), nn.InstanceNorm2d(64, affine=True), nn.ReLU(inplace=True)]
        in_features = 64
        out_features = in_features * 2
        affine = True
        for _ in range(2):
            model += [nn.Conv2d(in_features, out_features, 3, stride=2, padding=1), nn.InstanceNorm2d(out_features, affine=affine), nn.ReLU(inplace=True)]
            in_features = out_features
            out_features = in_features * 2
            affine = False
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_features)]
        out_features = in_features // 2
        for _ in range(2):
            model += [nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1), nn.InstanceNorm2d(out_features, affine=True), nn.ReLU(inplace=True)]
            in_features = out_features
            out_features = in_features // 2
        model += [nn.ReflectionPad2d(3), nn.Conv2d(64, output_nc, 7), nn.Tanh()]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):

    def __init__(self, input_nc):
        super(Discriminator, self).__init__()
        model = [nn.Conv2d(input_nc, 64, 4, stride=2, padding=1), nn.LeakyReLU(0.2, inplace=True)]
        model += [nn.Conv2d(64, 128, 4, stride=2, padding=1), nn.InstanceNorm2d(128), nn.LeakyReLU(0.2, inplace=True)]
        model += [nn.Conv2d(128, 256, 4, stride=2, padding=1), nn.InstanceNorm2d(256), nn.LeakyReLU(0.2, inplace=True)]
        model += [nn.Conv2d(256, 512, 4, padding=1), nn.InstanceNorm2d(512), nn.LeakyReLU(0.2, inplace=True)]
        model += [nn.Conv2d(512, 1, 4, padding=1)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        x = self.model(x)
        return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)


class FIDInceptionA(models.inception.InceptionA):
    """InceptionA block patched for FID computation"""

    def __init__(self, in_channels, pool_features):
        super(FIDInceptionA, self).__init__(in_channels, pool_features)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)
        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)
        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1, count_include_pad=False)
        branch_pool = self.branch_pool(branch_pool)
        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class FIDInceptionC(models.inception.InceptionC):
    """InceptionC block patched for FID computation"""

    def __init__(self, in_channels, channels_7x7):
        super(FIDInceptionC, self).__init__(in_channels, channels_7x7)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch7x7 = self.branch7x7_1(x)
        branch7x7 = self.branch7x7_2(branch7x7)
        branch7x7 = self.branch7x7_3(branch7x7)
        branch7x7dbl = self.branch7x7dbl_1(x)
        branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_4(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_5(branch7x7dbl)
        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1, count_include_pad=False)
        branch_pool = self.branch_pool(branch_pool)
        outputs = [branch1x1, branch7x7, branch7x7dbl, branch_pool]
        return torch.cat(outputs, 1)


class FIDInceptionE_1(models.inception.InceptionE):
    """First InceptionE block patched for FID computation"""

    def __init__(self, in_channels):
        super(FIDInceptionE_1, self).__init__(in_channels)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [self.branch3x3_2a(branch3x3), self.branch3x3_2b(branch3x3)]
        branch3x3 = torch.cat(branch3x3, 1)
        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [self.branch3x3dbl_3a(branch3x3dbl), self.branch3x3dbl_3b(branch3x3dbl)]
        branch3x3dbl = torch.cat(branch3x3dbl, 1)
        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1, count_include_pad=False)
        branch_pool = self.branch_pool(branch_pool)
        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class FIDInceptionE_2(models.inception.InceptionE):
    """Second InceptionE block patched for FID computation"""

    def __init__(self, in_channels):
        super(FIDInceptionE_2, self).__init__(in_channels)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [self.branch3x3_2a(branch3x3), self.branch3x3_2b(branch3x3)]
        branch3x3 = torch.cat(branch3x3, 1)
        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [self.branch3x3dbl_3a(branch3x3dbl), self.branch3x3dbl_3b(branch3x3dbl)]
        branch3x3dbl = torch.cat(branch3x3dbl, 1)
        branch_pool = F.max_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)
        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


FID_WEIGHTS_URL = 'https://github.com/mseitzer/pytorch-fid/releases/download/fid_weights/pt_inception-2015-12-05-6726825d.pth'


def fid_inception_v3():
    """Build pretrained Inception model for FID computation

    The Inception model for FID computation uses a different set of weights
    and has a slightly different structure than torchvision's Inception.

    This method first constructs torchvision's Inception and then patches the
    necessary parts that are different in the FID Inception model.
    """
    inception = models.inception_v3(num_classes=1008, aux_logits=False, pretrained=False)
    inception.Mixed_5b = FIDInceptionA(192, pool_features=32)
    inception.Mixed_5c = FIDInceptionA(256, pool_features=64)
    inception.Mixed_5d = FIDInceptionA(288, pool_features=64)
    inception.Mixed_6b = FIDInceptionC(768, channels_7x7=128)
    inception.Mixed_6c = FIDInceptionC(768, channels_7x7=160)
    inception.Mixed_6d = FIDInceptionC(768, channels_7x7=160)
    inception.Mixed_6e = FIDInceptionC(768, channels_7x7=192)
    inception.Mixed_7b = FIDInceptionE_1(1280)
    inception.Mixed_7c = FIDInceptionE_2(2048)
    state_dict = load_state_dict_from_url(FID_WEIGHTS_URL, progress=True)
    inception.load_state_dict(state_dict)
    return inception


class InceptionV3(nn.Module):
    """Pretrained InceptionV3 network returning feature maps"""
    DEFAULT_BLOCK_INDEX = 3
    BLOCK_INDEX_BY_DIM = {(64): 0, (192): 1, (768): 2, (2048): 3}

    def __init__(self, output_blocks=[DEFAULT_BLOCK_INDEX], resize_input=True, normalize_input=True, requires_grad=False, use_fid_inception=True):
        """Build pretrained InceptionV3

        Parameters
        ----------
        output_blocks : list of int
            Indices of blocks to return features of. Possible values are:
                - 0: corresponds to output of first max pooling
                - 1: corresponds to output of second max pooling
                - 2: corresponds to output which is fed to aux classifier
                - 3: corresponds to output of final average pooling
        resize_input : bool
            If true, bilinearly resizes input to width and height 299 before
            feeding input to model. As the network without fully connected
            layers is fully convolutional, it should be able to handle inputs
            of arbitrary size, so resizing might not be strictly needed
        normalize_input : bool
            If true, scales the input from range (0, 1) to the range the
            pretrained Inception network expects, namely (-1, 1)
        requires_grad : bool
            If true, parameters of the model require gradients. Possibly useful
            for finetuning the network
        use_fid_inception : bool
            If true, uses the pretrained Inception model used in Tensorflow's
            FID implementation. If false, uses the pretrained Inception model
            available in torchvision. The FID Inception model has different
            weights and a slightly different structure from torchvision's
            Inception model. If you want to compute FID scores, you are
            strongly advised to set this parameter to true to get comparable
            results.
        """
        super(InceptionV3, self).__init__()
        self.resize_input = resize_input
        self.normalize_input = normalize_input
        self.output_blocks = sorted(output_blocks)
        self.last_needed_block = max(output_blocks)
        assert self.last_needed_block <= 3, 'Last possible output block index is 3'
        self.blocks = nn.ModuleList()
        if use_fid_inception:
            inception = fid_inception_v3()
        else:
            inception = models.inception_v3(pretrained=True)
        block0 = [inception.Conv2d_1a_3x3, inception.Conv2d_2a_3x3, inception.Conv2d_2b_3x3, nn.MaxPool2d(kernel_size=3, stride=2)]
        self.blocks.append(nn.Sequential(*block0))
        if self.last_needed_block >= 1:
            block1 = [inception.Conv2d_3b_1x1, inception.Conv2d_4a_3x3, nn.MaxPool2d(kernel_size=3, stride=2)]
            self.blocks.append(nn.Sequential(*block1))
        if self.last_needed_block >= 2:
            block2 = [inception.Mixed_5b, inception.Mixed_5c, inception.Mixed_5d, inception.Mixed_6a, inception.Mixed_6b, inception.Mixed_6c, inception.Mixed_6d, inception.Mixed_6e]
            self.blocks.append(nn.Sequential(*block2))
        if self.last_needed_block >= 3:
            block3 = [inception.Mixed_7a, inception.Mixed_7b, inception.Mixed_7c, nn.AdaptiveAvgPool2d(output_size=(1, 1))]
            self.blocks.append(nn.Sequential(*block3))
        for param in self.parameters():
            param.requires_grad = requires_grad

    def forward(self, inp):
        """Get Inception feature maps

        Parameters
        ----------
        inp : torch.autograd.Variable
            Input tensor of shape Bx3xHxW. Values are expected to be in
            range (0, 1)

        Returns
        -------
        List of torch.autograd.Variable, corresponding to the selected output
        block, sorted ascending by index
        """
        outp = []
        x = inp
        if self.resize_input:
            x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
        if self.normalize_input:
            x = 2 * x - 1
        for idx, block in enumerate(self.blocks):
            x = block(x)
            if idx in self.output_blocks:
                outp.append(x)
            if idx == self.last_needed_block:
                break
        return outp


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (BasicResidual,
     lambda: ([], {'C_in': 4, 'C_out': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Conv,
     lambda: ([], {'C_in': 4, 'C_out': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Conv2d,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Conv3x3,
     lambda: ([], {'C_in': 4, 'C_out': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Conv7x7,
     lambda: ([], {'C_in': 4, 'C_out': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ConvNorm,
     lambda: ([], {'C_in': 4, 'C_out': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ConvTranspose2d,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ConvTranspose2dNorm,
     lambda: ([], {'C_in': 4, 'C_out': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (FIDInceptionA,
     lambda: ([], {'in_channels': 4, 'pool_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (FIDInceptionC,
     lambda: ([], {'in_channels': 4, 'channels_7x7': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (FIDInceptionE_1,
     lambda: ([], {'in_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (FIDInceptionE_2,
     lambda: ([], {'in_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (QConv2d,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (QConvTranspose2d,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (QLinear,
     lambda: ([], {'in_features': 4, 'out_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (QuantMeasure,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (RRDB,
     lambda: ([], {'nf': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (RRDBNet,
     lambda: ([], {'in_nc': 4, 'out_nc': 4, 'nf': 4, 'nb': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (RangeBN,
     lambda: ([], {'num_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (RangeBN1d,
     lambda: ([], {'num_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ResidualBlock,
     lambda: ([], {'in_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ResidualDenseBlock_5C,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 64, 64, 64])], {})),
    (SkipConnect,
     lambda: ([], {'C_in': 4, 'C_out': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (USBatchNorm2d,
     lambda: ([], {'num_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (USConv2d,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (USConvTranspose2d,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Vgg16,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {})),
]

