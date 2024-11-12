
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


from torch.utils import data


import random


import numpy as np


import scipy.io as io


import numpy.linalg as linalg


import matplotlib.pyplot as plt


from scipy.ndimage.morphology import distance_transform_edt


from scipy.ndimage.morphology import distance_transform_cdt


import collections


import torchvision


import scipy.io as sio


import torch.nn.functional as F


from torch.utils.data.dataloader import default_collate


import torch.autograd as autograd


import torch.nn as nn


from torch.autograd.function import once_differentiable


from torch.autograd import Function


from torch.nn.modules.utils import _pair


import math


from torch.nn.modules.module import Module


from torch import nn


from torch.autograd import Variable


import time


from torch.autograd import gradcheck


import torch.nn.functional as functional


import torch.cuda.comm as comm


from torch.utils.cpp_extension import load


import torch.distributed as dist


from numbers import Number


from itertools import repeat


from torch.autograd.function import Function


from torch.nn.parameter import Parameter


import torch as th


from functools import wraps


from torch.nn.parallel._functions import _get_stream


import functools


from torch.nn.parallel._functions import Broadcast


from torch.nn.parallel.data_parallel import DataParallel


from torch.nn.parallel.parallel_apply import get_a_var


from torch.nn.parallel.scatter_gather import gather


from torch._utils import _flatten_dense_tensors


from torch._utils import _unflatten_dense_tensors


from torch._utils import _take_tensors


from torch.nn.parallel._functions import Scatter as OrigScatter


from torch.nn.functional import batch_norm


from torch.nn.modules.batchnorm import _BatchNorm


from torch.nn.parallel._functions import ReduceAddCoalesced


import torch.utils.checkpoint as cp


from collections import OrderedDict


from torch.nn import Conv2d


from torch.nn import Module


from torch.nn import Linear


from torch.nn import ReLU


from functools import partial


from torch.hub import load_state_dict_from_url


from torchvision.models.resnet import ResNet


from torchvision.models.resnet import Bottleneck


from torch.nn import functional as F


import matplotlib


from sklearn import svm


from sklearn import datasets


from sklearn.model_selection import train_test_split


from sklearn.metrics import confusion_matrix


import torch.backends.cudnn as cudnn


import scipy


from scipy import ndimage


from math import ceil


from collections import Counter


from torch.nn.parallel.scatter_gather import gather as torch_gather


from torch.optim import SGD


from torch.optim import Adam


from torch.optim import lr_scheduler


def _check_contiguous(*args):
    if not all([(mod is None or mod.is_contiguous()) for mod in args]):
        raise ValueError('Non-contiguous input')


class CA_Map(autograd.Function):

    @staticmethod
    def forward(ctx, weight, g):
        out = torch.zeros_like(g)
        _ext.ca_map_forward_cuda(weight, g, out)
        ctx.save_for_backward(weight, g)
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, dout):
        weight, g = ctx.saved_tensors
        dw = torch.zeros_like(weight)
        dg = torch.zeros_like(g)
        _ext.ca_map_backward_cuda(dout.contiguous(), weight, g, dw, dg)
        _check_contiguous(dw, dg)
        return dw, dg


ca_map = CA_Map.apply


class CA_Weight(autograd.Function):

    @staticmethod
    def forward(ctx, t, f):
        n, c, h, w = t.size()
        size = n, h + w - 1, h, w
        weight = torch.zeros(size, dtype=t.dtype, layout=t.layout, device=t.device)
        _ext.ca_forward_cuda(t, f, weight)
        ctx.save_for_backward(t, f)
        return weight

    @staticmethod
    @once_differentiable
    def backward(ctx, dw):
        t, f = ctx.saved_tensors
        dt = torch.zeros_like(t)
        df = torch.zeros_like(f)
        _ext.ca_backward_cuda(dw.contiguous(), t, f, dt, df)
        _check_contiguous(dt, df)
        return dt, df


ca_weight = CA_Weight.apply


class CrossAttention(nn.Module):

    def __init__(self, dim_in, dim_inner, dim_out):
        super(CrossAttention, self).__init__()
        self.t_func = nn.Conv2d(in_channels=dim_in, out_channels=dim_inner, kernel_size=1, stride=1, padding=0)
        self.f_func = nn.Conv2d(in_channels=dim_in, out_channels=dim_inner, kernel_size=1, stride=1, padding=0)
        self.g_func = nn.Conv2d(in_channels=dim_in, out_channels=dim_out, kernel_size=1, stride=1, padding=0)
        self.inc = nn.Conv2d(in_channels=dim_out, out_channels=dim_in, kernel_size=1, stride=1, padding=0)
        nn.init.constant_(self.inc.weight, 0)
        nn.init.constant_(self.inc.bias, 0)

    def forward(self, x):
        t = self.t_func(x)
        f = self.f_func(x)
        g = self.g_func(x)
        w = ca_weight(t, f)
        w = F.softmax(w, 1)
        out = ca_map(w, g)
        x = x + self.inc(out)
        return x


class CrissCrossAttention(nn.Module):
    """ Pixel-wise attention module"""

    def __init__(self, in_dim):
        super(CrissCrossAttention, self).__init__()
        self.chanel_in = in_dim
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        proj_query = self.query_conv(x)
        proj_key = self.key_conv(x)
        proj_value = self.value_conv(x)
        energy = ca_weight(proj_query, proj_key)
        attention = F.softmax(energy, 1)
        out = ca_map(attention, proj_value)
        out = self.gamma * out + x
        return out


class PAM_Module(nn.Module):
    """ Position attention module"""

    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = F.softmax(energy, 1)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)
        out = self.gamma * out + x
        return out


class DeformConvFunction(Function):

    def __init__(self, stride, padding, dilation, deformable_groups=1, im2col_step=64):
        super(DeformConvFunction, self).__init__()
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.deformable_groups = deformable_groups
        self.im2col_step = im2col_step

    def forward(self, input, offset, weight):
        self.save_for_backward(input, offset, weight)
        output = input.new(*self._output_size(input, weight))
        self.bufs_ = [input.new(), input.new()]
        if not input.is_cuda:
            raise NotImplementedError
        else:
            if isinstance(input, torch.autograd.Variable):
                if not isinstance(input.data, torch.FloatTensor):
                    raise NotImplementedError
            elif not isinstance(input, torch.FloatTensor):
                raise NotImplementedError
            cur_im2col_step = min(self.im2col_step, input.shape[0])
            assert input.shape[0] % cur_im2col_step == 0, 'im2col step must divide batchsize'
            deform_conv.deform_conv_forward_cuda(input, weight, offset, output, self.bufs_[0], self.bufs_[1], weight.size(3), weight.size(2), self.stride[1], self.stride[0], self.padding[1], self.padding[0], self.dilation[1], self.dilation[0], self.deformable_groups, cur_im2col_step)
        return output

    def backward(self, grad_output):
        input, offset, weight = self.saved_tensors
        grad_input = grad_offset = grad_weight = None
        if not grad_output.is_cuda:
            raise NotImplementedError
        else:
            if isinstance(grad_output, torch.autograd.Variable):
                if not isinstance(grad_output.data, torch.FloatTensor):
                    raise NotImplementedError
            elif not isinstance(grad_output, torch.FloatTensor):
                raise NotImplementedError
            cur_im2col_step = min(self.im2col_step, input.shape[0])
            assert input.shape[0] % cur_im2col_step == 0, 'im2col step must divide batchsize'
            if self.needs_input_grad[0] or self.needs_input_grad[1]:
                grad_input = input.new(*input.size()).zero_()
                grad_offset = offset.new(*offset.size()).zero_()
                deform_conv.deform_conv_backward_input_cuda(input, offset, grad_output, grad_input, grad_offset, weight, self.bufs_[0], weight.size(3), weight.size(2), self.stride[1], self.stride[0], self.padding[1], self.padding[0], self.dilation[1], self.dilation[0], self.deformable_groups, cur_im2col_step)
            if self.needs_input_grad[2]:
                grad_weight = weight.new(*weight.size()).zero_()
                deform_conv.deform_conv_backward_parameters_cuda(input, offset, grad_output, grad_weight, self.bufs_[0], self.bufs_[1], weight.size(3), weight.size(2), self.stride[1], self.stride[0], self.padding[1], self.padding[0], self.dilation[1], self.dilation[0], self.deformable_groups, 1, cur_im2col_step)
        return grad_input, grad_offset, grad_weight

    def _output_size(self, input, weight):
        channels = weight.size(0)
        output_size = input.size(0), channels
        for d in range(input.dim() - 2):
            in_size = input.size(d + 2)
            pad = self.padding[d]
            kernel = self.dilation[d] * (weight.size(d + 2) - 1) + 1
            stride = self.stride[d]
            output_size += (in_size + 2 * pad - kernel) // stride + 1,
        if not all(map(lambda s: s > 0, output_size)):
            raise ValueError('convolution input is too small (output would be {})'.format('x'.join(map(str, output_size))))
        return output_size


def deform_conv_function(input, offset, weight, stride=1, padding=0, dilation=1, deform_groups=1, im2col_step=64):
    if input is not None and input.dim() != 4:
        raise ValueError('Expected 4D tensor as input, got {}D tensor instead.'.format(input.dim()))
    f = DeformConvFunction(_pair(stride), _pair(padding), _pair(dilation), deform_groups, im2col_step)
    return f(input, offset, weight)


class DeformConv(Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, num_deformable_groups=1):
        super(DeformConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.num_deformable_groups = num_deformable_groups
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, *self.kernel_size))
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1.0 / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, offset):
        return deform_conv_function(input, offset, self.weight, self.stride, self.padding, self.dilation, self.num_deformable_groups)


class ModulatedDeformConvFunction(Function):

    def __init__(self, stride, padding, dilation=1, deformable_groups=1):
        super(ModulatedDeformConvFunction, self).__init__()
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.deformable_groups = deformable_groups

    def forward(self, input, offset, mask, weight, bias):
        if not input.is_cuda:
            raise NotImplementedError
        if weight.requires_grad or mask.requires_grad or offset.requires_grad or input.requires_grad:
            self.save_for_backward(input, offset, mask, weight, bias)
        output = input.new(*self._infer_shape(input, weight))
        self._bufs = [input.new(), input.new()]
        _backend.modulated_deform_conv_cuda_forward(input, weight, bias, self._bufs[0], offset, mask, output, self._bufs[1], weight.shape[2], weight.shape[3], self.stride, self.stride, self.padding, self.padding, self.dilation, self.dilation, self.deformable_groups)
        return output

    def backward(self, grad_output):
        if not grad_output.is_cuda:
            raise NotImplementedError
        input, offset, mask, weight, bias = self.saved_tensors
        grad_input = input.new(*input.size()).zero_()
        grad_offset = offset.new(*offset.size()).zero_()
        grad_mask = mask.new(*mask.size()).zero_()
        grad_weight = weight.new(*weight.size()).zero_()
        grad_bias = bias.new(*bias.size()).zero_()
        _backend.modulated_deform_conv_cuda_backward(input, weight, bias, self._bufs[0], offset, mask, self._bufs[1], grad_input, grad_weight, grad_bias, grad_offset, grad_mask, grad_output, weight.shape[2], weight.shape[3], self.stride, self.stride, self.padding, self.padding, self.dilation, self.dilation, self.deformable_groups)
        return grad_input, grad_offset, grad_mask, grad_weight, grad_bias

    def _infer_shape(self, input, weight):
        n = input.size(0)
        channels_out = weight.size(0)
        height, width = input.shape[2:4]
        kernel_h, kernel_w = weight.shape[2:4]
        height_out = (height + 2 * self.padding - (self.dilation * (kernel_h - 1) + 1)) // self.stride + 1
        width_out = (width + 2 * self.padding - (self.dilation * (kernel_w - 1) + 1)) // self.stride + 1
        return n, channels_out, height_out, width_out


class ModulatedDeformConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation=1, deformable_groups=1, no_bias=True):
        super(ModulatedDeformConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.deformable_groups = deformable_groups
        self.no_bias = no_bias
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, *self.kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.reset_parameters()
        if self.no_bias:
            self.bias.requires_grad = False

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1.0 / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.zero_()

    def forward(self, input, offset, mask):
        func = ModulatedDeformConvFunction(self.stride, self.padding, self.dilation, self.deformable_groups)
        return func(input, offset, mask, self.weight, self.bias)


class ModulatedDeformConvPack(ModulatedDeformConv):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation=1, deformable_groups=1, no_bias=False):
        super(ModulatedDeformConvPack, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, deformable_groups, no_bias)
        self.conv_offset_mask = nn.Conv2d(self.in_channels, self.deformable_groups * 3 * self.kernel_size[0] * self.kernel_size[1], kernel_size=self.kernel_size, stride=(self.stride, self.stride), padding=(self.padding, self.padding), bias=True)
        self.init_offset()

    def init_offset(self):
        self.conv_offset_mask.weight.data.zero_()
        self.conv_offset_mask.bias.data.zero_()

    def forward(self, input):
        out = self.conv_offset_mask(input)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        func = ModulatedDeformConvFunction(self.stride, self.padding, self.dilation, self.deformable_groups)
        return func(input, offset, mask, self.weight, self.bias)


class DeformRoIPoolingFunction(Function):

    def __init__(self, spatial_scale, pooled_size, output_dim, no_trans, group_size=1, part_size=None, sample_per_part=4, trans_std=0.0):
        super(DeformRoIPoolingFunction, self).__init__()
        self.spatial_scale = spatial_scale
        self.pooled_size = pooled_size
        self.output_dim = output_dim
        self.no_trans = no_trans
        self.group_size = group_size
        self.part_size = pooled_size if part_size is None else part_size
        self.sample_per_part = sample_per_part
        self.trans_std = trans_std
        assert self.trans_std >= 0.0 and self.trans_std <= 1.0

    def forward(self, data, rois, offset):
        if not data.is_cuda:
            raise NotImplementedError
        output = data.new(*self._infer_shape(data, rois))
        output_count = data.new(*self._infer_shape(data, rois))
        _backend.deform_psroi_pooling_cuda_forward(data, rois, offset, output, output_count, self.no_trans, self.spatial_scale, self.output_dim, self.group_size, self.pooled_size, self.part_size, self.sample_per_part, self.trans_std)
        self.data = data
        self.rois = rois
        self.offset = offset
        self.output_count = output_count
        return output

    def backward(self, grad_output):
        if not grad_output.is_cuda:
            raise NotImplementedError
        data = self.data
        rois = self.rois
        offset = self.offset
        output_count = self.output_count
        grad_input = data.new(*data.size()).zero_()
        grad_offset = offset.new(*offset.size()).zero_()
        _backend.deform_psroi_pooling_cuda_backward(grad_output, data, rois, offset, output_count, grad_input, grad_offset, self.no_trans, self.spatial_scale, self.output_dim, self.group_size, self.pooled_size, self.part_size, self.sample_per_part, self.trans_std)
        return grad_input, torch.zeros(rois.shape), grad_offset

    def _infer_shape(self, data, rois):
        c = data.shape[1]
        n = rois.shape[0]
        return n, self.output_dim, self.pooled_size, self.pooled_size


class DeformRoIPooling(nn.Module):

    def __init__(self, spatial_scale, pooled_size, output_dim, no_trans, group_size=1, part_size=None, sample_per_part=4, trans_std=0.0):
        super(DeformRoIPooling, self).__init__()
        self.spatial_scale = spatial_scale
        self.pooled_size = pooled_size
        self.output_dim = output_dim
        self.no_trans = no_trans
        self.group_size = group_size
        self.part_size = pooled_size if part_size is None else part_size
        self.sample_per_part = sample_per_part
        self.trans_std = trans_std
        self.func = DeformRoIPoolingFunction(self.spatial_scale, self.pooled_size, self.output_dim, self.no_trans, self.group_size, self.part_size, self.sample_per_part, self.trans_std)

    def forward(self, data, rois, offset):
        if self.no_trans:
            offset = data.new()
        return self.func(data, rois, offset)


class ModulatedDeformRoIPoolingPack(DeformRoIPooling):

    def __init__(self, spatial_scale, pooled_size, output_dim, no_trans, group_size=1, part_size=None, sample_per_part=4, trans_std=0.0, deform_fc_dim=1024):
        super(ModulatedDeformRoIPoolingPack, self).__init__(spatial_scale, pooled_size, output_dim, no_trans, group_size, part_size, sample_per_part, trans_std)
        self.deform_fc_dim = deform_fc_dim
        if not no_trans:
            self.func_offset = DeformRoIPoolingFunction(self.spatial_scale, self.pooled_size, self.output_dim, True, self.group_size, self.part_size, self.sample_per_part, self.trans_std)
            self.offset_fc = nn.Sequential(nn.Linear(self.pooled_size * self.pooled_size * self.output_dim, self.deform_fc_dim), nn.ReLU(inplace=True), nn.Linear(self.deform_fc_dim, self.deform_fc_dim), nn.ReLU(inplace=True), nn.Linear(self.deform_fc_dim, self.pooled_size * self.pooled_size * 2))
            self.offset_fc[4].weight.data.zero_()
            self.offset_fc[4].bias.data.zero_()
            self.mask_fc = nn.Sequential(nn.Linear(self.pooled_size * self.pooled_size * self.output_dim, self.deform_fc_dim), nn.ReLU(inplace=True), nn.Linear(self.deform_fc_dim, self.pooled_size * self.pooled_size * 1), nn.Sigmoid())
            self.mask_fc[2].weight.data.zero_()
            self.mask_fc[2].bias.data.zero_()

    def forward(self, data, rois):
        if self.no_trans:
            offset = data.new()
        else:
            n = rois.shape[0]
            offset = data.new()
            x = self.func_offset(data, rois, offset)
            offset = self.offset_fc(x.view(n, -1))
            offset = offset.view(n, 2, self.pooled_size, self.pooled_size)
            mask = self.mask_fc(x.view(n, -1))
            mask = mask.view(n, 1, self.pooled_size, self.pooled_size)
            feat = self.func(data, rois, offset) * mask
            return feat
        return self.func(data, rois, offset)


class FilterResponseNormalization(nn.Module):

    def __init__(self, beta, gamma, tau, eps=1e-06):
        """
        Input Variables:
        ----------------
            beta, gamma, tau: Variables of shape [1, C, 1, 1].
            eps: A scalar constant or learnable variable.
        """
        super(FilterResponseNormalization, self).__init__()
        self.beta = beta
        self.gamma = gamma
        self.tau = tau
        self.eps = torch.Tensor([eps])

    def forward(self, x):
        """
        Input Variables:
        ----------------
            x: Input tensor of shape [NxCxHxW]
        """
        n, c, h, w = x.shape
        assert (self.gamma.shape[1], self.beta.shape[1], self.tau.shape[1]) == (c, c, c)
        nu2 = torch.mean(x.pow(2), (2, 3), keepdims=True)
        x = x * torch.rsqrt(nu2 + torch.abs(self.eps))
        return torch.max(self.gamma * x + self.beta, self.tau)


ACT_ELU = 'elu'


ACT_LEAKY_RELU = 'leaky_relu'


ACT_RELU = 'relu'


class ABN(nn.Module):
    """Activated Batch Normalization

    This gathers a `BatchNorm2d` and an activation function in a single module
    """

    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True, activation='leaky_relu', slope=0.01):
        """Creates an Activated Batch Normalization module

        Parameters
        ----------
        num_features : int
            Number of feature channels in the input and output.
        eps : float
            Small constant to prevent numerical issues.
        momentum : float
            Momentum factor applied to compute running statistics as.
        affine : bool
            If `True` apply learned scale and shift transformation after normalization.
        activation : str
            Name of the activation functions, one of: `leaky_relu`, `elu` or `none`.
        slope : float
            Negative slope for the `leaky_relu` activation.
        """
        super(ABN, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps
        self.momentum = momentum
        self.activation = activation
        self.slope = slope
        if self.affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self.running_mean, 0)
        nn.init.constant_(self.running_var, 1)
        if self.affine:
            nn.init.constant_(self.weight, 1)
            nn.init.constant_(self.bias, 0)

    def forward(self, x):
        x = functional.batch_norm(x, self.running_mean, self.running_var, self.weight, self.bias, self.training, self.momentum, self.eps)
        if self.activation == ACT_RELU:
            return functional.relu(x, inplace=True)
        elif self.activation == ACT_LEAKY_RELU:
            return functional.leaky_relu(x, negative_slope=self.slope, inplace=True)
        elif self.activation == ACT_ELU:
            return functional.elu(x, inplace=True)
        else:
            return x

    def __repr__(self):
        rep = '{name}({num_features}, eps={eps}, momentum={momentum}, affine={affine}, activation={activation}'
        if self.activation == 'leaky_relu':
            rep += ', slope={slope})'
        else:
            rep += ')'
        return rep.format(name=self.__class__.__name__, **self.__dict__)


ACT_NONE = 'none'


def _act_backward(ctx, x, dx):
    if ctx.activation == ACT_LEAKY_RELU:
        _backend.leaky_relu_backward(x, dx, ctx.slope)
    elif ctx.activation == ACT_ELU:
        _backend.elu_backward(x, dx)
    elif ctx.activation == ACT_NONE:
        pass


def _act_forward(ctx, x):
    if ctx.activation == ACT_LEAKY_RELU:
        _backend.leaky_relu_forward(x, ctx.slope)
    elif ctx.activation == ACT_ELU:
        _backend.elu_forward(x)
    elif ctx.activation == ACT_NONE:
        pass


def _count_samples(x):
    count = 1
    for i, s in enumerate(x.size()):
        if i != 1:
            count *= s
    return count


class InPlaceABN(autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, running_mean, running_var, training=True, momentum=0.1, eps=1e-05, activation=ACT_LEAKY_RELU, slope=0.01):
        ctx.training = training
        ctx.momentum = momentum
        ctx.eps = eps
        ctx.activation = activation
        ctx.slope = slope
        ctx.affine = weight is not None and bias is not None
        count = _count_samples(x)
        x = x.contiguous()
        weight = weight.contiguous() if ctx.affine else x.new_empty(0, dtype=torch.float32)
        bias = bias.contiguous() if ctx.affine else x.new_empty(0, dtype=torch.float32)
        if ctx.training:
            mean, var = _backend.mean_var(x)
            running_mean.mul_(1 - ctx.momentum).add_(ctx.momentum * mean)
            running_var.mul_(1 - ctx.momentum).add_(ctx.momentum * var * count / (count - 1))
            ctx.mark_dirty(x, running_mean, running_var)
        else:
            mean, var = running_mean.contiguous(), running_var.contiguous()
            ctx.mark_dirty(x)
        _backend.forward(x, mean, var, weight, bias, ctx.affine, ctx.eps)
        _act_forward(ctx, x)
        ctx.var = var
        ctx.save_for_backward(x, var, weight, bias)
        return x

    @staticmethod
    @once_differentiable
    def backward(ctx, dz):
        z, var, weight, bias = ctx.saved_tensors
        dz = dz.contiguous()
        _act_backward(ctx, z, dz)
        if ctx.training:
            edz, eydz = _backend.edz_eydz(z, dz, weight, bias, ctx.affine, ctx.eps)
        else:
            edz = dz.new_zeros(dz.size(1))
            eydz = dz.new_zeros(dz.size(1))
        dx = _backend.backward(z, dz, var, weight, bias, edz, eydz, ctx.affine, ctx.eps)
        dweight = eydz if ctx.affine else None
        if dweight is not None:
            dweight[weight < 0] *= -1
        dbias = edz if ctx.affine else None
        return dx, dweight, dbias, None, None, None, None, None, None, None


class InPlaceABNSync(autograd.Function):

    @classmethod
    def forward(cls, ctx, x, weight, bias, running_mean, running_var, training=True, momentum=0.1, eps=1e-05, activation=ACT_LEAKY_RELU, slope=0.01, equal_batches=True):
        ctx.training = training
        ctx.momentum = momentum
        ctx.eps = eps
        ctx.activation = activation
        ctx.slope = slope
        ctx.affine = weight is not None and bias is not None
        ctx.world_size = dist.get_world_size() if dist.is_initialized() else 1
        batch_size = x.new_tensor([x.shape[0]], dtype=torch.long)
        x = x.contiguous()
        weight = weight.contiguous() if ctx.affine else x.new_empty(0, dtype=torch.float32)
        bias = bias.contiguous() if ctx.affine else x.new_empty(0, dtype=torch.float32)
        if ctx.training:
            mean, var = _backend.mean_var(x)
            if ctx.world_size > 1:
                if equal_batches:
                    batch_size *= ctx.world_size
                else:
                    dist.all_reduce(batch_size, dist.ReduceOp.SUM)
                ctx.factor = x.shape[0] / float(batch_size.item())
                mean_all = mean.clone() * ctx.factor
                dist.all_reduce(mean_all, dist.ReduceOp.SUM)
                var_all = (var + (mean - mean_all) ** 2) * ctx.factor
                dist.all_reduce(var_all, dist.ReduceOp.SUM)
                mean = mean_all
                var = var_all
            running_mean.mul_(1 - ctx.momentum).add_(ctx.momentum * mean)
            count = batch_size.item() * x.view(x.shape[0], x.shape[1], -1).shape[-1]
            running_var.mul_(1 - ctx.momentum).add_(ctx.momentum * var * (float(count) / (count - 1)))
            ctx.mark_dirty(x, running_mean, running_var)
        else:
            mean, var = running_mean.contiguous(), running_var.contiguous()
            ctx.mark_dirty(x)
        _backend.forward(x, mean, var, weight, bias, ctx.affine, ctx.eps)
        _act_forward(ctx, x)
        ctx.var = var
        ctx.save_for_backward(x, var, weight, bias)
        return x

    @staticmethod
    @once_differentiable
    def backward(ctx, dz):
        z, var, weight, bias = ctx.saved_tensors
        dz = dz.contiguous()
        _act_backward(ctx, z, dz)
        if ctx.training:
            edz, eydz = _backend.edz_eydz(z, dz, weight, bias, ctx.affine, ctx.eps)
            edz_local = edz.clone()
            eydz_local = eydz.clone()
            if ctx.world_size > 1:
                edz *= ctx.factor
                dist.all_reduce(edz, dist.ReduceOp.SUM)
                eydz *= ctx.factor
                dist.all_reduce(eydz, dist.ReduceOp.SUM)
        else:
            edz_local = edz = dz.new_zeros(dz.size(1))
            eydz_local = eydz = dz.new_zeros(dz.size(1))
        dx = _backend.backward(z, dz, var, weight, bias, edz, eydz, ctx.affine, ctx.eps)
        dweight = eydz_local if ctx.affine else None
        if dweight is not None:
            dweight[weight < 0] *= -1
        dbias = edz_local if ctx.affine else None
        return dx, dweight, dbias, None, None, None, None, None, None, None


class GlobalAvgPool2d(nn.Module):

    def __init__(self):
        """Global average pooling over the input's spatial dimensions"""
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, inputs):
        in_size = inputs.size()
        return inputs.view((in_size[0], in_size[1], -1)).mean(dim=2)


class SingleGPU(nn.Module):

    def __init__(self, module):
        super(SingleGPU, self).__init__()
        self.module = module

    def forward(self, input):
        return self.module(input)


def np_gaussian_2d(width, sigma=-1):
    """Truncated 2D Gaussian filter"""
    assert width % 2 == 1
    if sigma <= 0:
        sigma = float(width) / 4
    r = np.arange(-(width // 2), width // 2 + 1, dtype=np.float32)
    gaussian_1d = np.exp(-0.5 * r * r / (sigma * sigma))
    gaussian_2d = gaussian_1d.reshape(-1, 1) * gaussian_1d
    gaussian_2d /= gaussian_2d.sum()
    return gaussian_2d


class _PacConvNd(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, transposed, output_padding, bias, pool_only, kernel_type, smooth_kernel_type, channel_wise, normalize_kernel, shared_filters, filler):
        super(_PacConvNd, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.pool_only = pool_only
        self.kernel_type = kernel_type
        self.smooth_kernel_type = smooth_kernel_type
        self.channel_wise = channel_wise
        self.normalize_kernel = normalize_kernel
        self.shared_filters = shared_filters
        self.filler = filler
        if any([(k % 2 != 1) for k in kernel_size]):
            raise ValueError('kernel_size only accept odd numbers')
        if smooth_kernel_type.find('_') >= 0 and int(smooth_kernel_type[smooth_kernel_type.rfind('_') + 1:]) % 2 != 1:
            raise ValueError('smooth_kernel_type only accept kernels of odd widths')
        if shared_filters:
            assert in_channels == out_channels, 'when specifying shared_filters, number of channels should not change'
        if any([(p > d * (k - 1) / 2) for p, d, k in zip(padding, dilation, kernel_size)]):
            pass
        if not pool_only:
            if self.filler in {'pool', 'crf_pool'}:
                assert shared_filters
                self.register_buffer('weight', torch.ones(1, 1, *kernel_size))
                if self.filler == 'crf_pool':
                    self.weight[(0, 0) + tuple(k // 2 for k in kernel_size)] = 0
            elif shared_filters:
                self.weight = Parameter(torch.Tensor(1, 1, *kernel_size))
            elif transposed:
                self.weight = Parameter(torch.Tensor(in_channels, out_channels, *kernel_size))
            else:
                self.weight = Parameter(torch.Tensor(out_channels, in_channels, *kernel_size))
            if bias:
                self.bias = Parameter(torch.Tensor(out_channels))
            else:
                self.register_parameter('bias', None)
        if kernel_type.startswith('inv_'):
            self.inv_alpha_init = float(kernel_type.split('_')[1])
            self.inv_lambda_init = float(kernel_type.split('_')[2])
            if self.channel_wise and kernel_type.find('_fixed') < 0:
                if out_channels <= 0:
                    raise ValueError('out_channels needed for channel_wise {}'.format(kernel_type))
                inv_alpha = self.inv_alpha_init * torch.ones(out_channels)
                inv_lambda = self.inv_lambda_init * torch.ones(out_channels)
            else:
                inv_alpha = torch.tensor(float(self.inv_alpha_init))
                inv_lambda = torch.tensor(float(self.inv_lambda_init))
            if kernel_type.find('_fixed') < 0:
                self.register_parameter('inv_alpha', Parameter(inv_alpha))
                self.register_parameter('inv_lambda', Parameter(inv_lambda))
            else:
                self.register_buffer('inv_alpha', inv_alpha)
                self.register_buffer('inv_lambda', inv_lambda)
        elif kernel_type != 'gaussian':
            raise ValueError('kernel_type set to invalid value ({})'.format(kernel_type))
        if smooth_kernel_type.startswith('full_'):
            smooth_kernel_size = int(smooth_kernel_type.split('_')[-1])
            self.smooth_kernel = Parameter(torch.Tensor(1, 1, *repeat(smooth_kernel_size, len(kernel_size))))
        elif smooth_kernel_type == 'gaussian':
            smooth_1d = torch.tensor([0.25, 0.5, 0.25])
            smooth_kernel = smooth_1d
            for d in range(1, len(kernel_size)):
                smooth_kernel = smooth_kernel * smooth_1d.view(-1, *repeat(1, d))
            self.register_buffer('smooth_kernel', smooth_kernel.unsqueeze(0).unsqueeze(0))
        elif smooth_kernel_type.startswith('average_'):
            smooth_kernel_size = int(smooth_kernel_type.split('_')[-1])
            smooth_1d = torch.tensor((1.0 / smooth_kernel_size,) * smooth_kernel_size)
            smooth_kernel = smooth_1d
            for d in range(1, len(kernel_size)):
                smooth_kernel = smooth_kernel * smooth_1d.view(-1, *repeat(1, d))
            self.register_buffer('smooth_kernel', smooth_kernel.unsqueeze(0).unsqueeze(0))
        elif smooth_kernel_type != 'none':
            raise ValueError('smooth_kernel_type set to invalid value ({})'.format(smooth_kernel_type))
        self.reset_parameters()

    def reset_parameters(self):
        if not (self.pool_only or self.filler in {'pool', 'crf_pool'}):
            if self.filler == 'uniform':
                n = self.in_channels
                for k in self.kernel_size:
                    n *= k
                stdv = 1.0 / math.sqrt(n)
                if self.shared_filters:
                    stdv *= self.in_channels
                self.weight.data.uniform_(-stdv, stdv)
                if self.bias is not None:
                    self.bias.data.uniform_(-stdv, stdv)
            elif self.filler == 'linear':
                effective_kernel_size = tuple(2 * s - 1 for s in self.stride)
                pad = tuple(int((k - ek) // 2) for k, ek in zip(self.kernel_size, effective_kernel_size))
                assert self.transposed and self.in_channels == self.out_channels
                assert all(k >= ek for k, ek in zip(self.kernel_size, effective_kernel_size))
                w = 1.0
                for i, (p, s, k) in enumerate(zip(pad, self.stride, self.kernel_size)):
                    d = len(pad) - i - 1
                    w = w * (np.array((0.0,) * p + tuple(range(1, s)) + tuple(range(s, 0, -1)) + (0,) * p) / s).reshape((-1,) + (1,) * d)
                    if self.normalize_kernel:
                        w = w * np.array(tuple((k - j - 1) // s + j // s + 1.0 for j in range(k))).reshape((-1,) + (1,) * d)
                self.weight.data.fill_(0.0)
                for c in range(1 if self.shared_filters else self.in_channels):
                    self.weight.data[c, c, :] = torch.tensor(w)
                if self.bias is not None:
                    self.bias.data.fill_(0.0)
            elif self.filler in {'crf', 'crf_perturbed'}:
                assert len(self.kernel_size) == 2 and self.kernel_size[0] == self.kernel_size[1] and self.in_channels == self.out_channels
                perturb_range = 0.001
                n_classes = self.in_channels
                gauss = np_gaussian_2d(self.kernel_size[0]) * self.kernel_size[0] * self.kernel_size[0]
                gauss[self.kernel_size[0] // 2, self.kernel_size[1] // 2] = 0
                if self.shared_filters:
                    self.weight.data[0, 0, :] = torch.tensor(gauss)
                else:
                    compat = 1.0 - np.eye(n_classes, dtype=np.float32)
                    self.weight.data[:] = torch.tensor(compat.reshape(n_classes, n_classes, 1, 1) * gauss)
                if self.filler == 'crf_perturbed':
                    self.weight.data.add_((torch.rand_like(self.weight.data) - 0.5) * perturb_range)
                if self.bias is not None:
                    self.bias.data.fill_(0.0)
            else:
                raise ValueError('Initialization method ({}) not supported.'.format(self.filler))
        if hasattr(self, 'inv_alpha') and isinstance(self.inv_alpha, Parameter):
            self.inv_alpha.data.fill_(self.inv_alpha_init)
            self.inv_lambda.data.fill_(self.inv_lambda_init)
        if hasattr(self, 'smooth_kernel') and isinstance(self.smooth_kernel, Parameter):
            self.smooth_kernel.data.fill_(1.0 / np.multiply.reduce(self.smooth_kernel.shape))

    def extra_repr(self):
        s = '{in_channels}, {out_channels}, kernel_size={kernel_size}, kernel_type={kernel_type}'
        if self.stride != (1,) * len(self.stride):
            s += ', stride={stride}'
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.bias is None:
            s += ', bias=False'
        if self.smooth_kernel_type != 'none':
            s += ', smooth_kernel_type={smooth_kernel_type}'
        if self.channel_wise:
            s += ', channel_wise=True'
        if self.normalize_kernel:
            s += ', normalize_kernel=True'
        if self.shared_filters:
            s += ', shared_filters=True'
        return s.format(**self.__dict__)


class PacConv2dFn(Function):

    @staticmethod
    def forward(ctx, input, kernel, weight, bias=None, stride=1, padding=0, dilation=1, shared_filters=False):
        (bs, ch), in_sz = input.shape[:2], input.shape[2:]
        if kernel.size(1) > 1:
            raise ValueError('Non-singleton channel is not allowed for kernel.')
        ctx.input_size = in_sz
        ctx.in_ch = ch
        ctx.kernel_size = tuple(weight.shape[-2:])
        ctx.dilation = _pair(dilation)
        ctx.padding = _pair(padding)
        ctx.stride = _pair(stride)
        ctx.shared_filters = shared_filters
        ctx.save_for_backward(input if ctx.needs_input_grad[1] or ctx.needs_input_grad[2] else None, kernel if ctx.needs_input_grad[0] or ctx.needs_input_grad[2] else None, weight if ctx.needs_input_grad[0] or ctx.needs_input_grad[1] else None)
        ctx._backend = type2backend[input.type()]
        cols = F.unfold(input, ctx.kernel_size, ctx.dilation, ctx.padding, ctx.stride)
        in_mul_k = cols.view(bs, ch, *kernel.shape[2:]) * kernel
        if shared_filters:
            output = torch.einsum('ijklmn,zykl->ijmn', (in_mul_k, weight))
        else:
            output = torch.einsum('ijklmn,ojkl->iomn', (in_mul_k, weight))
        if bias is not None:
            output += bias.view(1, -1, 1, 1)
        return output.clone()

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        grad_input = grad_kernel = grad_weight = grad_bias = None
        (bs, out_ch), out_sz = grad_output.shape[:2], grad_output.shape[2:]
        in_ch = ctx.in_ch
        input, kernel, weight = ctx.saved_tensors
        if ctx.needs_input_grad[0] or ctx.needs_input_grad[1]:
            if ctx.shared_filters:
                grad_in_mul_k = grad_output.view(bs, out_ch, 1, 1, out_sz[0], out_sz[1]) * weight.view(ctx.kernel_size[0], ctx.kernel_size[1], 1, 1)
            else:
                grad_in_mul_k = torch.einsum('iomn,ojkl->ijklmn', (grad_output, weight))
        if ctx.needs_input_grad[1] or ctx.needs_input_grad[2]:
            in_cols = F.unfold(input, ctx.kernel_size, ctx.dilation, ctx.padding, ctx.stride)
            in_cols = in_cols.view(bs, in_ch, ctx.kernel_size[0], ctx.kernel_size[1], out_sz[0], out_sz[1])
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.new()
            grad_im2col_output = grad_in_mul_k * kernel
            grad_im2col_output = grad_im2col_output.view(bs, -1, out_sz[0] * out_sz[1])
            ctx._backend.Im2Col_updateGradInput(ctx._backend.library_state, grad_im2col_output, grad_input, ctx.input_size[0], ctx.input_size[1], ctx.kernel_size[0], ctx.kernel_size[1], ctx.dilation[0], ctx.dilation[1], ctx.padding[0], ctx.padding[1], ctx.stride[0], ctx.stride[1])
        if ctx.needs_input_grad[1]:
            grad_kernel = in_cols * grad_in_mul_k
            grad_kernel = grad_kernel.sum(dim=1, keepdim=True)
        if ctx.needs_input_grad[2]:
            in_mul_k = in_cols * kernel
            if ctx.shared_filters:
                grad_weight = torch.einsum('ijmn,ijklmn->kl', (grad_output, in_mul_k))
                grad_weight = grad_weight.view(1, 1, ctx.kernel_size[0], ctx.kernel_size[1]).contiguous()
            else:
                grad_weight = torch.einsum('iomn,ijklmn->ojkl', (grad_output, in_mul_k))
        if ctx.needs_input_grad[3]:
            grad_bias = torch.einsum('iomn->o', (grad_output,))
        return grad_input, grad_kernel, grad_weight, grad_bias, None, None, None, None


def nd2col(input_nd, kernel_size, stride=1, padding=0, output_padding=0, dilation=1, transposed=False, use_pyinn_if_possible=False):
    """
    Shape:
        - Input: :math:`(N, C, L_{in})`
        - Output: :math:`(N, C, *kernel_size, *L_{out})` where
          :math:`L_{out} = floor((L_{in} + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1)` for non-transposed
          :math:`L_{out} = (L_{in} - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1 + output_padding` for transposed
    """
    n_dims = len(input_nd.shape[2:])
    kernel_size = (kernel_size,) * n_dims if isinstance(kernel_size, Number) else kernel_size
    stride = (stride,) * n_dims if isinstance(stride, Number) else stride
    padding = (padding,) * n_dims if isinstance(padding, Number) else padding
    output_padding = (output_padding,) * n_dims if isinstance(output_padding, Number) else output_padding
    dilation = (dilation,) * n_dims if isinstance(dilation, Number) else dilation
    if transposed:
        assert n_dims == 2, 'Only 2D is supported for fractional strides.'
        w_one = input_nd.new_ones(1, 1, 1, 1)
        pad = [((k - 1) * d - p) for k, d, p in zip(kernel_size, dilation, padding)]
        input_nd = F.conv_transpose2d(input_nd, w_one, stride=stride)
        input_nd = F.pad(input_nd, (pad[1], pad[1] + output_padding[1], pad[0], pad[0] + output_padding[0]))
        stride = _pair(1)
        padding = _pair(0)
    (bs, nch), in_sz = input_nd.shape[:2], input_nd.shape[2:]
    out_sz = tuple([((i + 2 * p - d * (k - 1) - 1) // s + 1) for i, k, d, p, s in zip(in_sz, kernel_size, dilation, padding, stride)])
    if n_dims == 2 and dilation == 1 and has_pyinn and torch.cuda.is_available() and use_pyinn_if_possible:
        output = P.im2col(input_nd, kernel_size, stride, padding)
    else:
        output = F.unfold(input_nd, kernel_size, dilation, padding, stride)
        out_shape = (bs, nch) + tuple(kernel_size) + out_sz
        output = output.view(*out_shape).contiguous()
    return output


def pacconv2d(input, kernel, weight, bias=None, stride=1, padding=0, dilation=1, shared_filters=False, native_impl=False):
    kernel_size = tuple(weight.shape[-2:])
    stride = _pair(stride)
    padding = _pair(padding)
    dilation = _pair(dilation)
    if native_impl:
        im_cols = nd2col(input, kernel_size, stride=stride, padding=padding, dilation=dilation)
        if shared_filters:
            output = torch.einsum('ijklmn,zykl->ijmn', (im_cols * kernel, weight))
        else:
            output = torch.einsum('ijklmn,ojkl->iomn', (im_cols * kernel, weight))
        if bias is not None:
            output += bias.view(1, -1, 1, 1)
    else:
        output = PacConv2dFn.apply(input, kernel, weight, bias, stride, padding, dilation, shared_filters)
    return output


class GaussKernel2dFn(Function):

    @staticmethod
    def forward(ctx, input, kernel_size, stride, padding, dilation, channel_wise):
        ctx.kernel_size = _pair(kernel_size)
        ctx.dilation = _pair(dilation)
        ctx.padding = _pair(padding)
        ctx.stride = _pair(stride)
        bs, ch, in_h, in_w = input.shape
        out_h = (in_h + 2 * ctx.padding[0] - ctx.dilation[0] * (ctx.kernel_size[0] - 1) - 1) // ctx.stride[0] + 1
        out_w = (in_w + 2 * ctx.padding[1] - ctx.dilation[1] * (ctx.kernel_size[1] - 1) - 1) // ctx.stride[1] + 1
        cols = F.unfold(input, ctx.kernel_size, ctx.dilation, ctx.padding, ctx.stride)
        cols = cols.view(bs, ch, ctx.kernel_size[0], ctx.kernel_size[1], out_h, out_w)
        center_y, center_x = ctx.kernel_size[0] // 2, ctx.kernel_size[1] // 2
        feat_0 = cols.contiguous()[:, :, center_y:center_y + 1, center_x:center_x + 1, :, :]
        diff_sq = (cols - feat_0).pow(2)
        if not channel_wise:
            diff_sq = diff_sq.sum(dim=1, keepdim=True)
        output = torch.exp(-0.5 * diff_sq)
        ctx._backend = type2backend[input.type()]
        ctx.save_for_backward(input, output)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        input, output = ctx.saved_tensors
        bs, ch, in_h, in_w = input.shape
        out_h, out_w = output.shape[-2:]
        cols = F.unfold(input, ctx.kernel_size, ctx.dilation, ctx.padding, ctx.stride)
        cols = cols.view(bs, ch, ctx.kernel_size[0], ctx.kernel_size[1], out_h, out_w)
        center_y, center_x = ctx.kernel_size[0] // 2, ctx.kernel_size[1] // 2
        feat_0 = cols.contiguous()[:, :, center_y:center_y + 1, center_x:center_x + 1, :, :]
        diff = cols - feat_0
        grad = -0.5 * grad_output * output
        grad_diff = grad.expand_as(cols) * (2 * diff)
        grad_diff[:, :, center_y:center_y + 1, center_x:center_x + 1, :, :] -= grad_diff.sum(dim=2, keepdim=True).sum(dim=3, keepdim=True)
        grad_input = grad_output.new()
        ctx._backend.Im2Col_updateGradInput(ctx._backend.library_state, grad_diff.view(bs, ch * ctx.kernel_size[0] * ctx.kernel_size[1], -1), grad_input, in_h, in_w, ctx.kernel_size[0], ctx.kernel_size[1], ctx.dilation[0], ctx.dilation[1], ctx.padding[0], ctx.padding[1], ctx.stride[0], ctx.stride[1])
        return grad_input, None, None, None, None, None


def _neg_idx(idx):
    return None if idx == 0 else -idx


def packernel2d(input, mask=None, kernel_size=0, stride=1, padding=0, output_padding=0, dilation=1, kernel_type='gaussian', smooth_kernel_type='none', smooth_kernel=None, inv_alpha=None, inv_lambda=None, channel_wise=False, normalize_kernel=False, transposed=False, native_impl=False):
    kernel_size = _pair(kernel_size)
    dilation = _pair(dilation)
    padding = _pair(padding)
    output_padding = _pair(output_padding)
    stride = _pair(stride)
    output_mask = False if mask is None else True
    norm = None
    if mask is not None and mask.dtype != input.dtype:
        mask = torch.tensor(mask, dtype=input.dtype, device=input.device)
    if transposed:
        in_sz = tuple(int((o - op - 1 - (k - 1) * d + 2 * p) // s) + 1 for o, k, s, p, op, d in zip(input.shape[-2:], kernel_size, stride, padding, output_padding, dilation))
    else:
        in_sz = input.shape[-2:]
    if mask is not None or normalize_kernel:
        mask_pattern = input.new_ones(1, 1, *in_sz)
        mask_pattern = nd2col(mask_pattern, kernel_size, stride=stride, padding=padding, output_padding=output_padding, dilation=dilation, transposed=transposed)
        if mask is not None:
            mask = nd2col(mask, kernel_size, stride=stride, padding=padding, output_padding=output_padding, dilation=dilation, transposed=transposed)
            if not normalize_kernel:
                norm = mask.sum(dim=2, keepdim=True).sum(dim=3, keepdim=True) / mask_pattern.sum(dim=2, keepdim=True).sum(dim=3, keepdim=True)
        else:
            mask = mask_pattern
    if transposed:
        stride = _pair(1)
        padding = tuple((k - 1) * d // 2 for k, d in zip(kernel_size, dilation))
    if native_impl:
        bs, k_ch, in_h, in_w = input.shape
        x = nd2col(input, kernel_size, stride=stride, padding=padding, dilation=dilation)
        x = x.view(bs, k_ch, -1, *x.shape[-2:]).contiguous()
        if smooth_kernel_type == 'none':
            self_idx = kernel_size[0] * kernel_size[1] // 2
            feat_0 = x[:, :, self_idx:self_idx + 1, :, :]
        else:
            smooth_kernel_size = smooth_kernel.shape[2:]
            smooth_padding = int(padding[0] - (kernel_size[0] - smooth_kernel_size[0]) / 2), int(padding[1] - (kernel_size[1] - smooth_kernel_size[1]) / 2)
            crop = tuple(-1 * np.minimum(0, smooth_padding))
            input_for_kernel_crop = input.view(-1, 1, in_h, in_w)[:, :, crop[0]:_neg_idx(crop[0]), crop[1]:_neg_idx(crop[1])]
            smoothed = F.conv2d(input_for_kernel_crop, smooth_kernel, stride=stride, padding=tuple(np.maximum(0, smooth_padding)))
            feat_0 = smoothed.view(bs, k_ch, 1, *x.shape[-2:])
        x = x - feat_0
        if kernel_type.find('_asym') >= 0:
            x = F.relu(x, inplace=True)
        x = x * x
        if not channel_wise:
            x = torch.sum(x, dim=1, keepdim=True)
        if kernel_type == 'gaussian':
            x = torch.exp_(x.mul_(-0.5))
        elif kernel_type.startswith('inv_'):
            epsilon = 0.0001
            x = inv_alpha.view(1, -1, 1, 1, 1) + torch.pow(x + epsilon, 0.5 * inv_lambda.view(1, -1, 1, 1, 1))
        else:
            raise ValueError()
        output = x.view(*(x.shape[:2] + tuple(kernel_size) + x.shape[-2:])).contiguous()
    else:
        assert smooth_kernel_type == 'none' and kernel_type == 'gaussian'
        output = GaussKernel2dFn.apply(input, kernel_size, stride, padding, dilation, channel_wise)
    if mask is not None:
        output = output * mask
    if normalize_kernel:
        norm = output.sum(dim=2, keepdim=True).sum(dim=3, keepdim=True)
    if norm is not None:
        empty_mask = norm == 0
        output = output / (norm + torch.tensor(empty_mask, dtype=input.dtype, device=input.device))
        output_mask = 1 - empty_mask if output_mask else None
    else:
        output_mask = None
    return output, output_mask


class PacConv2d(_PacConvNd):
    """
    Args (in addition to those of Conv2d):
        kernel_type (str): 'gaussian' | 'inv_{alpha}_{lambda}[_asym][_fixed]'. Default: 'gaussian'
        smooth_kernel_type (str): 'none' | 'gaussian' | 'average_{sz}' | 'full_{sz}'. Default: 'none'
        normalize_kernel (bool): Default: False
        shared_filters (bool): Default: False
        filler (str): 'uniform'. Default: 'uniform'

    Note:
        - kernel_size only accepts odd numbers
        - padding should not be larger than :math:`dilation * (kernel_size - 1) / 2`
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True, kernel_type='gaussian', smooth_kernel_type='none', normalize_kernel=False, shared_filters=False, filler='uniform', native_impl=False):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(PacConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, False, _pair(0), bias, False, kernel_type, smooth_kernel_type, False, normalize_kernel, shared_filters, filler)
        self.native_impl = native_impl

    def compute_kernel(self, input_for_kernel, input_mask=None):
        return packernel2d(input_for_kernel, input_mask, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, dilation=self.dilation, kernel_type=self.kernel_type, smooth_kernel_type=self.smooth_kernel_type, smooth_kernel=self.smooth_kernel if hasattr(self, 'smooth_kernel') else None, inv_alpha=self.inv_alpha if hasattr(self, 'inv_alpha') else None, inv_lambda=self.inv_lambda if hasattr(self, 'inv_lambda') else None, channel_wise=False, normalize_kernel=self.normalize_kernel, transposed=False, native_impl=self.native_impl)

    def forward(self, input_2d, input_for_kernel, kernel=None, mask=None):
        output_mask = None
        if kernel is None:
            kernel, output_mask = self.compute_kernel(input_for_kernel, mask)
        output = pacconv2d(input_2d, kernel, self.weight, self.bias, self.stride, self.padding, self.dilation, self.shared_filters, self.native_impl)
        return output if output_mask is None else (output, output_mask)


class PacConvTranspose2dFn(Function):

    @staticmethod
    def forward(ctx, input, kernel, weight, bias=None, stride=1, padding=0, output_padding=0, dilation=1, shared_filters=False):
        (bs, ch), in_sz = input.shape[:2], input.shape[2:]
        if kernel.size(1) > 1:
            raise ValueError('Non-singleton channel is not allowed for kernel.')
        ctx.in_ch = ch
        ctx.kernel_size = tuple(weight.shape[-2:])
        ctx.dilation = _pair(dilation)
        ctx.padding = _pair(padding)
        ctx.output_padding = _pair(output_padding)
        ctx.stride = _pair(stride)
        ctx.shared_filters = shared_filters
        ctx.save_for_backward(input if ctx.needs_input_grad[1] or ctx.needs_input_grad[2] else None, kernel if ctx.needs_input_grad[0] or ctx.needs_input_grad[2] else None, weight if ctx.needs_input_grad[0] or ctx.needs_input_grad[1] else None)
        ctx._backend = type2backend[input.type()]
        w = input.new_ones((ch, 1, 1, 1))
        x = F.conv_transpose2d(input, w, stride=stride, groups=ch)
        pad = [((k - 1) * d - p) for k, d, p in zip(ctx.kernel_size, ctx.dilation, ctx.padding)]
        x = F.pad(x, (pad[1], pad[1] + ctx.output_padding[1], pad[0], pad[0] + ctx.output_padding[0]))
        cols = F.unfold(x, ctx.kernel_size, ctx.dilation, _pair(0), _pair(1))
        in_mul_k = cols.view(bs, ch, *kernel.shape[2:]) * kernel
        if shared_filters:
            output = torch.einsum('ijklmn,jokl->iomn', (in_mul_k, weight))
        else:
            output = torch.einsum('ijklmn,jokl->iomn', (in_mul_k, weight))
        if bias is not None:
            output += bias.view(1, -1, 1, 1)
        return output.clone()

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        grad_input = grad_kernel = grad_weight = grad_bias = None
        (bs, out_ch), out_sz = grad_output.shape[:2], grad_output.shape[2:]
        in_ch = ctx.in_ch
        pad = [((k - 1) * d - p) for k, d, p in zip(ctx.kernel_size, ctx.dilation, ctx.padding)]
        pad = [(p, p + op) for p, op in zip(pad, ctx.output_padding)]
        input, kernel, weight = ctx.saved_tensors
        if ctx.needs_input_grad[0] or ctx.needs_input_grad[1]:
            if ctx.shared_filters:
                grad_in_mul_k = grad_output.view(bs, out_ch, 1, 1, out_sz[0], out_sz[1]) * weight.view(ctx.kernel_size[0], ctx.kernel_size[1], 1, 1)
            else:
                grad_in_mul_k = torch.einsum('iomn,jokl->ijklmn', (grad_output, weight))
        if ctx.needs_input_grad[1] or ctx.needs_input_grad[2]:
            w = input.new_ones((in_ch, 1, 1, 1))
            x = F.conv_transpose2d(input, w, stride=ctx.stride, groups=in_ch)
            x = F.pad(x, (pad[1][0], pad[1][1], pad[0][0], pad[0][1]))
            in_cols = F.unfold(x, ctx.kernel_size, ctx.dilation, _pair(0), _pair(1))
            in_cols = in_cols.view(bs, in_ch, ctx.kernel_size[0], ctx.kernel_size[1], out_sz[0], out_sz[1])
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.new()
            grad_im2col_output = grad_in_mul_k * kernel
            grad_im2col_output = grad_im2col_output.view(bs, -1, out_sz[0] * out_sz[1])
            im2col_input_sz = [(o + (k - 1) * d) for o, k, d in zip(out_sz, ctx.kernel_size, ctx.dilation)]
            ctx._backend.Im2Col_updateGradInput(ctx._backend.library_state, grad_im2col_output, grad_input, im2col_input_sz[0], im2col_input_sz[1], ctx.kernel_size[0], ctx.kernel_size[1], ctx.dilation[0], ctx.dilation[1], 0, 0, 1, 1)
            grad_input = grad_input[:, :, pad[0][0]:-pad[0][1]:ctx.stride[0], pad[1][0]:-pad[1][1]:ctx.stride[1]]
        if ctx.needs_input_grad[1]:
            grad_kernel = in_cols * grad_in_mul_k
            grad_kernel = grad_kernel.sum(dim=1, keepdim=True)
        if ctx.needs_input_grad[2]:
            in_mul_k = in_cols * kernel
            if ctx.shared_filters:
                grad_weight = torch.einsum('ijmn,ijklmn->kl', (grad_output, in_mul_k))
                grad_weight = grad_weight.view(1, 1, ctx.kernel_size[0], ctx.kernel_size[1]).contiguous()
            else:
                grad_weight = torch.einsum('iomn,ijklmn->jokl', (grad_output, in_mul_k))
        if ctx.needs_input_grad[3]:
            grad_bias = torch.einsum('iomn->o', (grad_output,))
        return grad_input, grad_kernel, grad_weight, grad_bias, None, None, None, None, None


def pacconv_transpose2d(input, kernel, weight, bias=None, stride=1, padding=0, output_padding=0, dilation=1, shared_filters=False, native_impl=False):
    kernel_size = tuple(weight.shape[-2:])
    stride = _pair(stride)
    padding = _pair(padding)
    output_padding = _pair(output_padding)
    dilation = _pair(dilation)
    if native_impl:
        ch = input.shape[1]
        w = input.new_ones((ch, 1, 1, 1))
        x = F.conv_transpose2d(input, w, stride=stride, groups=ch)
        pad = [((kernel_size[i] - 1) * dilation[i] - padding[i]) for i in range(2)]
        x = F.pad(x, (pad[1], pad[1] + output_padding[1], pad[0], pad[0] + output_padding[0]))
        output = pacconv2d(x, kernel, weight.permute(1, 0, 2, 3), bias, dilation=dilation, shared_filters=shared_filters, native_impl=True)
    else:
        output = PacConvTranspose2dFn.apply(input, kernel, weight, bias, stride, padding, output_padding, dilation, shared_filters)
    return output


class PacConvTranspose2d(_PacConvNd):
    """
    Args (in addition to those of ConvTranspose2d):
        kernel_type (str): 'gaussian' | 'inv_{alpha}_{lambda}[_asym][_fixed]'. Default: 'gaussian'
        smooth_kernel_type (str): 'none' | 'gaussian' | 'average_{sz}' | 'full_{sz}'. Default: 'none'
        normalize_kernel (bool): Default: False
        shared_filters (bool): Default: False
        filler (str): 'uniform' | 'linear'. Default: 'uniform'

    Note:
        - kernel_size only accepts odd numbers
        - padding should not be larger than :math:`dilation * (kernel_size - 1) / 2`
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, dilation=1, bias=True, kernel_type='gaussian', smooth_kernel_type='none', normalize_kernel=False, shared_filters=False, filler='uniform', native_impl=False):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        output_padding = _pair(output_padding)
        dilation = _pair(dilation)
        super(PacConvTranspose2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, True, output_padding, bias, False, kernel_type, smooth_kernel_type, False, normalize_kernel, shared_filters, filler)
        self.native_impl = native_impl

    def compute_kernel(self, input_for_kernel, input_mask=None):
        return packernel2d(input_for_kernel, input_mask, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, output_padding=self.output_padding, dilation=self.dilation, kernel_type=self.kernel_type, smooth_kernel_type=self.smooth_kernel_type, smooth_kernel=self.smooth_kernel if hasattr(self, 'smooth_kernel') else None, inv_alpha=self.inv_alpha if hasattr(self, 'inv_alpha') else None, inv_lambda=self.inv_lambda if hasattr(self, 'inv_lambda') else None, channel_wise=False, normalize_kernel=self.normalize_kernel, transposed=True, native_impl=self.native_impl)

    def forward(self, input_2d, input_for_kernel, kernel=None, mask=None):
        output_mask = None
        if kernel is None:
            kernel, output_mask = self.compute_kernel(input_for_kernel, mask)
        output = pacconv_transpose2d(input_2d, kernel, self.weight, self.bias, self.stride, self.padding, self.output_padding, self.dilation, self.shared_filters, self.native_impl)
        return output if output_mask is None else (output, output_mask)


class PacPool2dFn(Function):

    @staticmethod
    def forward(ctx, input, kernel, kernel_size, stride=1, padding=0, dilation=1):
        (bs, ch), in_sz = input.shape[:2], input.shape[2:]
        if kernel.size(1) > 1 and kernel.size(1) != ch:
            raise ValueError('Incompatible input and kernel sizes.')
        ctx.input_size = in_sz
        ctx.kernel_size = _pair(kernel_size)
        ctx.kernel_ch = kernel.size(1)
        ctx.dilation = _pair(dilation)
        ctx.padding = _pair(padding)
        ctx.stride = _pair(stride)
        ctx.save_for_backward(input if ctx.needs_input_grad[1] else None, kernel if ctx.needs_input_grad[0] else None)
        ctx._backend = type2backend[input.type()]
        cols = F.unfold(input, ctx.kernel_size, ctx.dilation, ctx.padding, ctx.stride)
        output = cols.view(bs, ch, *kernel.shape[2:]) * kernel
        output = torch.einsum('ijklmn->ijmn', (output,))
        return output.clone()

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        input, kernel = ctx.saved_tensors
        grad_input = grad_kernel = None
        (bs, ch), out_sz = grad_output.shape[:2], grad_output.shape[2:]
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.new()
            grad_im2col_output = torch.einsum('ijmn,izklmn->ijklmn', (grad_output, kernel))
            grad_im2col_output = grad_im2col_output.view(bs, -1, out_sz[0] * out_sz[1])
            ctx._backend.Im2Col_updateGradInput(ctx._backend.library_state, grad_im2col_output, grad_input, ctx.input_size[0], ctx.input_size[1], ctx.kernel_size[0], ctx.kernel_size[1], ctx.dilation[0], ctx.dilation[1], ctx.padding[0], ctx.padding[1], ctx.stride[0], ctx.stride[1])
        if ctx.needs_input_grad[1]:
            cols = F.unfold(input, ctx.kernel_size, ctx.dilation, ctx.padding, ctx.stride)
            cols = cols.view(bs, ch, ctx.kernel_size[0], ctx.kernel_size[1], out_sz[0], out_sz[1])
            grad_kernel = torch.einsum('ijmn,ijklmn->ijklmn', (grad_output, cols))
            if ctx.kernel_ch == 1:
                grad_kernel = grad_kernel.sum(dim=1, keepdim=True)
        return grad_input, grad_kernel, None, None, None, None


def pacpool2d(input, kernel, kernel_size, stride=1, padding=0, dilation=1, native_impl=False):
    kernel_size = _pair(kernel_size)
    stride = _pair(stride)
    padding = _pair(padding)
    dilation = _pair(dilation)
    if native_impl:
        bs, in_ch, in_h, in_w = input.shape
        out_h = (in_h + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) // stride[0] + 1
        out_w = (in_w + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) // stride[1] + 1
        im_cols = nd2col(input, kernel_size, stride=stride, padding=padding, dilation=dilation)
        im_cols *= kernel
        output = im_cols.view(bs, in_ch, -1, out_h, out_w).sum(dim=2, keepdim=False)
    else:
        output = PacPool2dFn.apply(input, kernel, kernel_size, stride, padding, dilation)
    return output


class PacPool2d(_PacConvNd):
    """
    Args:
        kernel_size, stride, padding, dilation
        kernel_type (str): 'gaussian' | 'inv_{alpha}_{lambda}[_asym][_fixed]'. Default: 'gaussian'
        smooth_kernel_type (str): 'none' | 'gaussian' | 'average_{sz}' | 'full_{sz}'. Default: 'none'
        channel_wise (bool): Default: False
        normalize_kernel (bool): Default: False
        out_channels (int): needs to be specified for channel_wise 'inv_*' (non-fixed) kernels. Default: -1

    Note:
        - kernel_size only accepts odd numbers
        - padding should not be larger than :math:`dilation * (kernel_size - 1) / 2`
    """

    def __init__(self, kernel_size, stride=1, padding=0, dilation=1, kernel_type='gaussian', smooth_kernel_type='none', channel_wise=False, normalize_kernel=False, out_channels=-1, native_impl=False):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(PacPool2d, self).__init__(-1, out_channels, kernel_size, stride, padding, dilation, False, _pair(0), False, True, kernel_type, smooth_kernel_type, channel_wise, normalize_kernel, False, None)
        self.native_impl = native_impl

    def compute_kernel(self, input_for_kernel, input_mask=None):
        return packernel2d(input_for_kernel, input_mask, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, dilation=self.dilation, kernel_type=self.kernel_type, smooth_kernel_type=self.smooth_kernel_type, smooth_kernel=self.smooth_kernel if hasattr(self, 'smooth_kernel') else None, inv_alpha=self.inv_alpha if hasattr(self, 'inv_alpha') else None, inv_lambda=self.inv_lambda if hasattr(self, 'inv_lambda') else None, channel_wise=self.channel_wise, normalize_kernel=self.normalize_kernel, transposed=False, native_impl=self.native_impl)

    def forward(self, input_2d, input_for_kernel, kernel=None, mask=None):
        output_mask = None
        if kernel is None:
            kernel, output_mask = self.compute_kernel(input_for_kernel, mask)
        bs, in_ch, in_h, in_w = input_2d.shape
        if self.channel_wise and kernel.shape[1] != in_ch:
            raise ValueError('input and kernel must have the same number of channels when channel_wise=True')
        assert self.out_channels <= 0 or self.out_channels == in_ch
        output = pacpool2d(input_2d, kernel, self.kernel_size, self.stride, self.padding, self.dilation, self.native_impl)
        return output if output_mask is None else (output, output_mask)


def _ceil_pad_factor(sizes, factor):
    offs = tuple((factor - sz % factor) % factor for sz in sizes)
    pad = tuple((off + 1) // 2 for off in offs)
    return pad


class PacCRF(nn.Module):
    """
    Args:
        channels (int): number of categories.
        num_steps (int): number of mean-field update steps.
        final_output (str): 'log_softmax' | 'softmax' | 'log_Q'. Default: 'log_Q'
        perturbed_init (bool): whether to perturb initialization. Default: True
        native_impl (bool): Default: False
        fixed_weighting (bool): whether to use fixed weighting for unary/pairwise terms. Default: False
        unary_weight (float): Default: 1.0
        pairwise_kernels (dict or list): pairwise kernels, see add_pairwise_kernel() for details. Default: None
    """

    def __init__(self, channels, num_steps, final_output='log_Q', perturbed_init=True, native_impl=False, fixed_weighting=False, unary_weight=1.0, pairwise_kernels=None):
        super(PacCRF, self).__init__()
        self.channels = channels
        self.num_steps = num_steps
        self.final_output = final_output
        self.perturbed_init = perturbed_init
        self.native_impl = native_impl
        self.fixed_weighting = fixed_weighting
        self.init_unary_weight = unary_weight
        self.messengers = nn.ModuleList()
        self.compat = nn.ModuleList()
        self.init_pairwise_weights = []
        self.pairwise_weights = nn.ParameterList()
        self._use_pairwise_weights = []
        self.unary_weight = unary_weight if self.fixed_weighting else nn.Parameter(th.tensor(float(unary_weight)))
        self.blur = []
        self.pairwise_repr = []
        if pairwise_kernels is not None:
            if type(pairwise_kernels) == dict:
                self.add_pairwise_kernel(**pairwise_kernels)
            else:
                for k in pairwise_kernels:
                    self.add_pairwise_kernel(**k)

    def reset_parameters(self, pairwise_idx=None):
        if pairwise_idx is None:
            idxs = range(len(self.messengers))
            if not self.fixed_weighting:
                self.unary_weight.data.fill_(self.init_unary_weight)
        else:
            idxs = [pairwise_idx]
        for i in idxs:
            self.messengers[i].reset_parameters()
            if isinstance(self.messengers[i], nn.Conv2d):
                pass
            if self.compat[i] is not None:
                self.compat[i].weight.data[:, :, 0, 0] = 1.0 - th.eye(self.channels, dtype=th.float32)
                if self.perturbed_init:
                    perturb_range = 0.001
                    self.compat[i].weight.data.add_((th.rand_like(self.compat[i].weight.data) - 0.5) * perturb_range)
            self.pairwise_weights[i].data = th.ones_like(self.pairwise_weights[i]) * self.init_pairwise_weights[i]

    def extra_repr(self):
        s = 'categories={channels}, num_steps={num_steps}, final_output={final_output}'
        if self.perturbed_init:
            s += ', perturbed_init=True'
        if self.fixed_weighting:
            s += ', fixed_weighting=True'
        if self.pairwise_repr:
            s += ', pairwise_kernels=({})'.format(', '.join(self.pairwise_repr))
        return s.format(**self.__dict__)

    def add_pairwise_kernel(self, kernel_size=3, dilation=1, blur=1, compat_type='4d', spatial_filter=True, pairwise_weight=1.0):
        assert kernel_size % 2 == 1
        self.pairwise_repr.append('{}{}_{}_{}_{}'.format('0d' if compat_type == 'potts' else compat_type, 's' if spatial_filter else '', kernel_size, dilation, blur))
        if compat_type == 'potts':
            pairwise_weight *= -1.0
        if compat_type == 'potts' and not spatial_filter and not self.fixed_weighting:
            self._use_pairwise_weights.append(True)
        else:
            self._use_pairwise_weights.append(False)
        self.pairwise_weights.append(nn.Parameter(th.tensor(pairwise_weight, dtype=th.float32)))
        self.init_pairwise_weights.append(pairwise_weight)
        self.blur.append(blur)
        self.compat.append(nn.Conv2d(self.channels, self.channels, kernel_size=1, bias=False) if compat_type == '2d' else None)
        pad = int(kernel_size // 2) * dilation
        if compat_type == 'na':
            messenger = nn.Conv2d(self.channels, self.channels, kernel_size, padding=pad, dilation=dilation, bias=False)
        elif compat_type == '4d':
            messenger = pac.PacConv2d(self.channels, self.channels, kernel_size, padding=pad, dilation=dilation, bias=False, shared_filters=False, native_impl=self.native_impl, filler='crf_perturbed' if self.perturbed_init else 'crf')
        elif spatial_filter:
            messenger = pac.PacConv2d(self.channels, self.channels, kernel_size, padding=pad, dilation=dilation, bias=False, shared_filters=True, native_impl=self.native_impl, filler='crf_perturbed' if self.perturbed_init else 'crf')
        else:
            messenger = pac.PacConv2d(self.channels, self.channels, kernel_size, padding=pad, dilation=dilation, bias=False, shared_filters=True, native_impl=self.native_impl, filler='crf_pool')
        self.messengers.append(messenger)
        self.reset_parameters(-1)

    def num_pairwise_kernels(self):
        return len(self.messengers)

    def forward(self, unary, edge_feat, edge_kernel=None, logQ=None):
        n_kernels = len(self.messengers)
        edge_kernel = [edge_kernel] * n_kernels if isinstance(edge_kernel, th.Tensor) else edge_kernel
        if edge_kernel is None:
            edge_kernel = [None] * n_kernels
            _shared = isinstance(edge_feat, th.Tensor)
            if _shared:
                edge_feat = {(1): edge_feat}
            for i in range(n_kernels):
                if isinstance(self.messengers[i], nn.Conv2d):
                    continue
                if _shared and self.blur[i] in edge_feat:
                    feat = edge_feat[self.blur[i]]
                elif self.blur[i] == 1:
                    feat = edge_feat[i]
                else:
                    feat = edge_feat[1] if _shared else edge_feat[i]
                    pad = _ceil_pad_factor(feat.shape[2:], self.blur[i])
                    feat = F.avg_pool2d(feat, kernel_size=self.blur[i], padding=pad, count_include_pad=False)
                    if _shared:
                        edge_feat[self.blur[i]] = feat
                edge_kernel[i], _ = self.messengers[i].compute_kernel(feat)
                del feat
            del edge_feat
        if logQ is None:
            logQ = unary
        for step in range(self.num_steps):
            Q = F.softmax(logQ, dim=1)
            Q_blur = {(1): Q}
            logQ = unary * self.unary_weight
            for i in range(n_kernels):
                pad = _ceil_pad_factor(Q.shape[2:], self.blur[i])
                if self.blur[i] not in Q_blur:
                    Q_blur[self.blur[i]] = F.avg_pool2d(Q, kernel_size=self.blur[i], padding=pad, count_include_pad=False)
                if isinstance(self.messengers[i], nn.Conv2d):
                    msg = self.messengers[i](Q_blur[self.blur[i]])
                else:
                    msg = self.messengers[i](Q_blur[self.blur[i]], None, edge_kernel[i])
                if self.compat[i] is not None:
                    msg = self.compat[i](msg)
                if self.blur[i] > 1:
                    msg = F.interpolate(msg, scale_factor=self.blur[i], mode='bilinear', align_corners=False)
                    msg = msg[:, :, pad[0]:pad[0] + unary.shape[2], pad[1]:pad[1] + unary.shape[3]].contiguous()
                pw = self.pairwise_weights[i] if self._use_pairwise_weights[i] else self.init_pairwise_weights[i]
                logQ = logQ - msg * pw
        if self.final_output == 'softmax':
            out = F.softmax(logQ, dim=1)
        elif self.final_output == 'log_softmax':
            out = F.log_softmax(logQ, dim=1)
        elif self.final_output == 'log_Q':
            out = logQ
        else:
            raise ValueError('Unknown value for final_output: {}'.format(self.final_output))
        return out


class PacCRFLoose(nn.Module):

    def __init__(self, channels, num_steps, final_output='log_Q', perturbed_init=True, native_impl=False, fixed_weighting=False, unary_weight=1.0, pairwise_kernels=None):
        super(PacCRFLoose, self).__init__()
        self.channels = channels
        self.num_steps = num_steps
        self.final_output = final_output
        self.steps = nn.ModuleList()
        for i in range(num_steps):
            self.steps.append(PacCRF(channels, 1, 'log_Q', perturbed_init, native_impl, fixed_weighting, unary_weight, pairwise_kernels))
        self.reset_parameters()

    def reset_parameters(self):
        for i in range(self.num_steps):
            self.steps[i].reset_parameters()

    def extra_repr(self):
        s = 'categories={channels}, num_steps={num_steps}, final_output={final_output}'
        return s.format(**self.__dict__)

    def add_pairwise_kernel(self, kernel_size=3, dilation=1, blur=1, compat_type='4d', spatial_filter=True, pairwise_weight=1.0):
        for i in range(self.num_steps):
            self.steps[i].add_pairwise_kernel(kernel_size, dilation, blur, compat_type, spatial_filter, pairwise_weight)

    def num_pairwise_kernels(self):
        return self.steps[0].num_pairwise_kernels()

    def forward(self, unary, edge_feat, edge_kernel=None):
        n_kernels = self.num_pairwise_kernels()
        edge_kernel = [edge_kernel] * n_kernels if isinstance(edge_kernel, th.Tensor) else edge_kernel
        blurs = self.steps[0].blur
        if edge_kernel is None:
            edge_kernel = [None] * n_kernels
            _shared = isinstance(edge_feat, th.Tensor)
            if _shared:
                edge_feat = {(1): edge_feat}
            for i in range(n_kernels):
                if _shared and blurs[i] in edge_feat:
                    feat = edge_feat[blurs[i]]
                elif blurs[i] == 1:
                    feat = edge_feat[i]
                else:
                    feat = edge_feat[1] if _shared else edge_feat[i]
                    pad = _ceil_pad_factor(feat.shape[2:], blurs[i])
                    feat = F.avg_pool2d(feat, kernel_size=blurs[i], padding=pad, count_include_pad=False)
                    if _shared:
                        edge_feat[blurs[i]] = feat
                edge_kernel[i], _ = self.steps[0].messengers[i].compute_kernel(feat)
                del feat
            del edge_feat
        logQ = unary
        for step in self.steps:
            logQ = step(unary, None, edge_kernel, logQ)
        if self.final_output == 'softmax':
            out = F.softmax(logQ, dim=1)
        elif self.final_output == 'log_softmax':
            out = F.log_softmax(logQ, dim=1)
        elif self.final_output == 'log_Q':
            out = logQ
        else:
            raise ValueError('Unknown value for final_output: {}'.format(self.final_output))
        return out


class CallbackContext(object):
    pass


def execute_replication_callbacks(modules):
    """
    Execute an replication callback `__data_parallel_replicate__` on each module created
    by original replication.

    The callback will be invoked with arguments `__data_parallel_replicate__(ctx, copy_id)`

    Note that, as all modules are isomorphism, we assign each sub-module with a context
    (shared among multiple copies of this module on different devices).
    Through this context, different copies can share some information.

    We guarantee that the callback on the master copy (the first copy) will be called ahead
    of calling the callback of any slave copies.
    """
    master_copy = modules[0]
    nr_modules = len(list(master_copy.modules()))
    ctxs = [CallbackContext() for _ in range(nr_modules)]
    for i, module in enumerate(modules):
        for j, m in enumerate(module.modules()):
            if hasattr(m, '__data_parallel_replicate__'):
                m.__data_parallel_replicate__(ctxs[j], i)


def assert_tensor_type(func):

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if not isinstance(args[0].data, torch.Tensor):
            raise AttributeError('{} has no attribute {} for type {}'.format(args[0].__class__.__name__, func.__name__, args[0].datatype))
        return func(*args, **kwargs)
    return wrapper


class DataContainer(object):
    """A container for any type of objects.

    Typically tensors will be stacked in the collate function and sliced along
    some dimension in the scatter function. This behavior has some limitations.
    1. All tensors have to be the same size.
    2. Types are limited (numpy array or Tensor).

    We design `DataContainer` and `MMDataParallel` to overcome these
    limitations. The behavior can be either of the following.

    - copy to GPU, pad all tensors to the same size and stack them
    - copy to GPU without stacking
    - leave the objects as is and pass it to the model
    """

    def __init__(self, data, stack=False, padding_value=0, cpu_only=False):
        self._data = data
        self._cpu_only = cpu_only
        self._stack = stack
        self._padding_value = padding_value

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, repr(self.data))

    @property
    def data(self):
        return self._data

    @property
    def datatype(self):
        if isinstance(self.data, torch.Tensor):
            return self.data.type()
        else:
            return type(self.data)

    @property
    def cpu_only(self):
        return self._cpu_only

    @property
    def stack(self):
        return self._stack

    @property
    def padding_value(self):
        return self._padding_value

    @assert_tensor_type
    def size(self, *args, **kwargs):
        return self.data.size(*args, **kwargs)

    @assert_tensor_type
    def dim(self):
        return self.data.dim()

    @assert_tensor_type
    def numel(self):
        return self.data.numel()


def get_input_device(input):
    if isinstance(input, list):
        for item in input:
            input_device = get_input_device(item)
            if input_device != -1:
                return input_device
        return -1
    elif isinstance(input, torch.Tensor):
        return input.get_device() if input.is_cuda else -1
    else:
        raise Exception('Unknown type {}.'.format(type(input)))

