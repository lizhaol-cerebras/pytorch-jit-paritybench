
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


from collections import deque


import time


import math


import copy


import random


import numpy as np


from collections import OrderedDict


import logging


from torch import nn


from torch.autograd import Function


from torch.nn.modules.utils import _pair


from torch.autograd.function import once_differentiable


from torch.utils.cpp_extension import CUDA_HOME


from torch.utils.cpp_extension import CppExtension


from torch.utils.cpp_extension import CUDAExtension


import torch.nn as nn


from torch.autograd import gradcheck


import torch as T


import torch.nn.functional as F


class DCNv2(torch.nn.Module):
    """
    咩酱自实现的DCNv2，咩酱的得意之作，Pytorch的纯python接口实现，效率极高。
    """

    def __init__(self, input_dim, filters, filter_size, stride=1, padding=0, bias_attr=False, distribution='normal', gain=1):
        super(DCNv2, self).__init__()
        assert distribution in ['uniform', 'normal']
        self.input_dim = input_dim
        self.filters = filters
        self.filter_size = filter_size
        self.stride = stride
        self.padding = padding
        self.conv_offset = torch.nn.Conv2d(input_dim, filter_size * filter_size * 3, kernel_size=filter_size, stride=stride, padding=padding, bias=True)
        torch.nn.init.constant_(self.conv_offset.weight, 0.0)
        torch.nn.init.constant_(self.conv_offset.bias, 0.0)
        self.sigmoid = torch.nn.Sigmoid()
        self.dcn_weight = torch.nn.Parameter(torch.randn(filters, input_dim, filter_size, filter_size))
        self.dcn_bias = None
        if bias_attr:
            self.dcn_bias = torch.nn.Parameter(torch.randn(filters))
            torch.nn.init.constant_(self.dcn_bias, 0.0)
        if distribution == 'uniform':
            torch.nn.init.xavier_uniform_(self.dcn_weight, gain=gain)
        else:
            torch.nn.init.xavier_normal_(self.dcn_weight, gain=gain)

    def gather_nd(self, input, index):
        keep_dims = []
        first_dims = []
        dim_idx = []
        dims = index.shape[1]
        for i, number in enumerate(input.shape):
            if i < dims:
                dim_ = index[:, i]
                dim_idx.append(dim_)
                first_dims.append(number)
            else:
                keep_dims.append(number)
        target_dix = torch.zeros((index.shape[0],), dtype=torch.long, device=input.device) + dim_idx[-1]
        new_shape = (-1,) + tuple(keep_dims)
        input2 = torch.reshape(input, new_shape)
        mul2 = 1
        for i in range(dims - 1, 0, -1):
            mul2 *= first_dims[i]
            target_dix += mul2 * dim_idx[i - 1]
        o = input2[target_dix]
        return o

    def forward(self, x):
        filter_size = self.filter_size
        stride = self.stride
        padding = self.padding
        dcn_weight = self.dcn_weight
        dcn_bias = self.dcn_bias
        offset_mask = self.conv_offset(x)
        offset = offset_mask[:, :filter_size ** 2 * 2, :, :]
        mask = offset_mask[:, filter_size ** 2 * 2:, :, :]
        mask = self.sigmoid(mask)
        N, in_C, H, W = x.shape
        out_C, in_C, kH, kW = dcn_weight.shape
        out_W = (W + 2 * padding - (kW - 1)) // stride
        out_H = (H + 2 * padding - (kH - 1)) // stride
        pad_x_H = H + padding * 2 + 1
        pad_x_W = W + padding * 2 + 1
        pad_x = torch.zeros((N, in_C, pad_x_H, pad_x_W), dtype=torch.float32, device=x.device)
        pad_x[:, :, padding:padding + H, padding:padding + W] = x
        rows = torch.arange(0, out_W, dtype=torch.float32, device=dcn_weight.device) * stride + padding
        cols = torch.arange(0, out_H, dtype=torch.float32, device=dcn_weight.device) * stride + padding
        rows = rows[np.newaxis, np.newaxis, :, np.newaxis, np.newaxis].repeat((1, out_H, 1, 1, 1))
        cols = cols[np.newaxis, :, np.newaxis, np.newaxis, np.newaxis].repeat((1, 1, out_W, 1, 1))
        start_pos_yx = torch.cat([cols, rows], dim=-1)
        start_pos_yx = start_pos_yx.repeat((N, 1, 1, kH * kW, 1))
        start_pos_y = start_pos_yx[:, :, :, :, :1]
        start_pos_x = start_pos_yx[:, :, :, :, 1:]
        half_W = (kW - 1) // 2
        half_H = (kW - 1) // 2
        rows2 = torch.arange(0, kW, dtype=torch.float32, device=dcn_weight.device) - half_W
        cols2 = torch.arange(0, kH, dtype=torch.float32, device=dcn_weight.device) - half_H
        rows2 = rows2[np.newaxis, :, np.newaxis].repeat((kH, 1, 1))
        cols2 = cols2[:, np.newaxis, np.newaxis].repeat((1, kW, 1))
        filter_inner_offset_yx = torch.cat([cols2, rows2], dim=-1)
        filter_inner_offset_yx = torch.reshape(filter_inner_offset_yx, (1, 1, 1, kH * kW, 2))
        filter_inner_offset_yx = filter_inner_offset_yx.repeat((N, out_H, out_W, 1, 1))
        filter_inner_offset_y = filter_inner_offset_yx[:, :, :, :, :1]
        filter_inner_offset_x = filter_inner_offset_yx[:, :, :, :, 1:]
        mask = mask.permute(0, 2, 3, 1)
        offset = offset.permute(0, 2, 3, 1)
        offset_yx = torch.reshape(offset, (N, out_H, out_W, kH * kW, 2))
        offset_y = offset_yx[:, :, :, :, :1]
        offset_x = offset_yx[:, :, :, :, 1:]
        start_pos_y.requires_grad = False
        start_pos_x.requires_grad = False
        filter_inner_offset_y.requires_grad = False
        filter_inner_offset_x.requires_grad = False
        pos_y = start_pos_y + filter_inner_offset_y + offset_y
        pos_x = start_pos_x + filter_inner_offset_x + offset_x
        pos_y = torch.clamp(pos_y, 0.0, H + padding * 2 - 1.0)
        pos_x = torch.clamp(pos_x, 0.0, W + padding * 2 - 1.0)
        ytxt = torch.cat([pos_y, pos_x], -1)
        pad_x = pad_x.permute(0, 2, 3, 1)
        pad_x = torch.reshape(pad_x, (N * pad_x_H, pad_x_W, in_C))
        ytxt = torch.reshape(ytxt, (N * out_H * out_W * kH * kW, 2))
        _yt = ytxt[:, :1]
        _xt = ytxt[:, 1:]
        row_offset = torch.arange(0, N, dtype=torch.float32, device=dcn_weight.device) * pad_x_H
        row_offset = row_offset[:, np.newaxis, np.newaxis].repeat((1, out_H * out_W * kH * kW, 1))
        row_offset = torch.reshape(row_offset, (N * out_H * out_W * kH * kW, 1))
        row_offset.requires_grad = False
        _yt += row_offset
        _y1 = torch.floor(_yt)
        _x1 = torch.floor(_xt)
        _y2 = _y1 + 1.0
        _x2 = _x1 + 1.0
        _y1x1 = torch.cat([_y1, _x1], -1)
        _y1x2 = torch.cat([_y1, _x2], -1)
        _y2x1 = torch.cat([_y2, _x1], -1)
        _y2x2 = torch.cat([_y2, _x2], -1)
        _y1x1_int = _y1x1.long()
        v1 = self.gather_nd(pad_x, _y1x1_int)
        _y1x2_int = _y1x2.long()
        v2 = self.gather_nd(pad_x, _y1x2_int)
        _y2x1_int = _y2x1.long()
        v3 = self.gather_nd(pad_x, _y2x1_int)
        _y2x2_int = _y2x2.long()
        v4 = self.gather_nd(pad_x, _y2x2_int)
        lh = _yt - _y1
        lw = _xt - _x1
        hh = 1 - lh
        hw = 1 - lw
        w1 = hh * hw
        w2 = hh * lw
        w3 = lh * hw
        w4 = lh * lw
        value = w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4
        mask = torch.reshape(mask, (N * out_H * out_W * kH * kW, 1))
        value = value * mask
        value = torch.reshape(value, (N, out_H, out_W, kH, kW, in_C))
        new_x = value.permute(0, 1, 2, 5, 3, 4)
        new_x = torch.reshape(new_x, (N, out_H, out_W, in_C * kH * kW))
        new_x = new_x.permute(0, 3, 1, 2)
        rw = torch.reshape(dcn_weight, (out_C, in_C * kH * kW, 1, 1))
        out = F.conv2d(new_x, rw, stride=1)
        return out


class MyNet(torch.nn.Module):

    def __init__(self):
        super(MyNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, kernel_size=1, stride=1, padding=0, bias=True)
        self.act1 = torch.nn.LeakyReLU(0.1)
        self.dcnv2 = DCNv2(8, 512, filter_size=3, stride=2, padding=1, bias_attr=False)

    def __call__(self, input_tensor):
        x = self.conv1(input_tensor)
        x = self.act1(x)
        x = self.dcnv2(x)
        return x


class _DCNv2(Function):

    @staticmethod
    def forward(ctx, input, offset, mask, weight, bias, stride, padding, dilation, deformable_groups):
        ctx.stride = _pair(stride)
        ctx.padding = _pair(padding)
        ctx.dilation = _pair(dilation)
        ctx.kernel_size = _pair(weight.shape[2:4])
        ctx.deformable_groups = deformable_groups
        output = _backend.dcn_v2_forward(input, weight, bias, offset, mask, ctx.kernel_size[0], ctx.kernel_size[1], ctx.stride[0], ctx.stride[1], ctx.padding[0], ctx.padding[1], ctx.dilation[0], ctx.dilation[1], ctx.deformable_groups)
        ctx.save_for_backward(input, offset, mask, weight, bias)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        input, offset, mask, weight, bias = ctx.saved_tensors
        grad_input, grad_offset, grad_mask, grad_weight, grad_bias = _backend.dcn_v2_backward(input, weight, bias, offset, mask, grad_output, ctx.kernel_size[0], ctx.kernel_size[1], ctx.stride[0], ctx.stride[1], ctx.padding[0], ctx.padding[1], ctx.dilation[0], ctx.dilation[1], ctx.deformable_groups)
        return grad_input, grad_offset, grad_mask, grad_weight, grad_bias, None, None, None, None


dcn_v2_conv = _DCNv2.apply


class DCN(DCNv2):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation=1, deformable_groups=1):
        super(DCN, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, deformable_groups)
        channels_ = self.deformable_groups * 3 * self.kernel_size[0] * self.kernel_size[1]
        self.conv_offset_mask = nn.Conv2d(self.in_channels, channels_, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, bias=True)
        self.init_offset()

    def init_offset(self):
        self.conv_offset_mask.weight.data.zero_()
        self.conv_offset_mask.bias.data.zero_()

    def forward(self, input):
        out = self.conv_offset_mask(input)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        return dcn_v2_conv(input, offset, mask, self.weight, self.bias, self.stride, self.padding, self.dilation, self.deformable_groups)


class _DCNv2Pooling(Function):

    @staticmethod
    def forward(ctx, input, rois, offset, spatial_scale, pooled_size, output_dim, no_trans, group_size=1, part_size=None, sample_per_part=4, trans_std=0.0):
        ctx.spatial_scale = spatial_scale
        ctx.no_trans = int(no_trans)
        ctx.output_dim = output_dim
        ctx.group_size = group_size
        ctx.pooled_size = pooled_size
        ctx.part_size = pooled_size if part_size is None else part_size
        ctx.sample_per_part = sample_per_part
        ctx.trans_std = trans_std
        output, output_count = _backend.dcn_v2_psroi_pooling_forward(input, rois, offset, ctx.no_trans, ctx.spatial_scale, ctx.output_dim, ctx.group_size, ctx.pooled_size, ctx.part_size, ctx.sample_per_part, ctx.trans_std)
        ctx.save_for_backward(input, rois, offset, output_count)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        input, rois, offset, output_count = ctx.saved_tensors
        grad_input, grad_offset = _backend.dcn_v2_psroi_pooling_backward(grad_output, input, rois, offset, output_count, ctx.no_trans, ctx.spatial_scale, ctx.output_dim, ctx.group_size, ctx.pooled_size, ctx.part_size, ctx.sample_per_part, ctx.trans_std)
        return grad_input, None, grad_offset, None, None, None, None, None, None, None, None


dcn_v2_pooling = _DCNv2Pooling.apply


class DCNv2Pooling(nn.Module):

    def __init__(self, spatial_scale, pooled_size, output_dim, no_trans, group_size=1, part_size=None, sample_per_part=4, trans_std=0.0):
        super(DCNv2Pooling, self).__init__()
        self.spatial_scale = spatial_scale
        self.pooled_size = pooled_size
        self.output_dim = output_dim
        self.no_trans = no_trans
        self.group_size = group_size
        self.part_size = pooled_size if part_size is None else part_size
        self.sample_per_part = sample_per_part
        self.trans_std = trans_std

    def forward(self, input, rois, offset):
        assert input.shape[1] == self.output_dim
        if self.no_trans:
            offset = input.new()
        return dcn_v2_pooling(input, rois, offset, self.spatial_scale, self.pooled_size, self.output_dim, self.no_trans, self.group_size, self.part_size, self.sample_per_part, self.trans_std)


class DCNPooling(DCNv2Pooling):

    def __init__(self, spatial_scale, pooled_size, output_dim, no_trans, group_size=1, part_size=None, sample_per_part=4, trans_std=0.0, deform_fc_dim=1024):
        super(DCNPooling, self).__init__(spatial_scale, pooled_size, output_dim, no_trans, group_size, part_size, sample_per_part, trans_std)
        self.deform_fc_dim = deform_fc_dim
        if not no_trans:
            self.offset_mask_fc = nn.Sequential(nn.Linear(self.pooled_size * self.pooled_size * self.output_dim, self.deform_fc_dim), nn.ReLU(inplace=True), nn.Linear(self.deform_fc_dim, self.deform_fc_dim), nn.ReLU(inplace=True), nn.Linear(self.deform_fc_dim, self.pooled_size * self.pooled_size * 3))
            self.offset_mask_fc[4].weight.data.zero_()
            self.offset_mask_fc[4].bias.data.zero_()

    def forward(self, input, rois):
        offset = input.new()
        if not self.no_trans:
            n = rois.shape[0]
            roi = dcn_v2_pooling(input, rois, offset, self.spatial_scale, self.pooled_size, self.output_dim, True, self.group_size, self.part_size, self.sample_per_part, self.trans_std)
            offset_mask = self.offset_mask_fc(roi.view(n, -1))
            offset_mask = offset_mask.view(n, 3, self.pooled_size, self.pooled_size)
            o1, o2, mask = torch.chunk(offset_mask, 3, dim=1)
            offset = torch.cat((o1, o2), dim=1)
            mask = torch.sigmoid(mask)
            return dcn_v2_pooling(input, rois, offset, self.spatial_scale, self.pooled_size, self.output_dim, self.no_trans, self.group_size, self.part_size, self.sample_per_part, self.trans_std) * mask
        return dcn_v2_pooling(input, rois, offset, self.spatial_scale, self.pooled_size, self.output_dim, self.no_trans, self.group_size, self.part_size, self.sample_per_part, self.trans_std)


class Mish(torch.nn.Module):

    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, x):
        x = x * torch.tanh(F.softplus(x))
        return x


class AffineChannel(torch.nn.Module):

    def __init__(self, num_features):
        super(AffineChannel, self).__init__()
        self.weight = torch.nn.Parameter(torch.randn(num_features))
        self.bias = torch.nn.Parameter(torch.randn(num_features))

    def forward(self, x):
        N = x.size(0)
        C = x.size(1)
        H = x.size(2)
        W = x.size(3)
        transpose_x = x.permute(0, 2, 3, 1)
        flatten_x = transpose_x.reshape(N * H * W, C)
        out = flatten_x * self.weight + self.bias
        out = out.reshape(N, H, W, C)
        out = out.permute(0, 3, 1, 2)
        return out


class Conv2dUnit(torch.nn.Module):

    def __init__(self, input_dim, filters, filter_size, stride=1, bias_attr=False, bn=0, gn=0, af=0, groups=32, act=None, freeze_norm=False, is_test=False, norm_decay=0.0, lr=1.0, bias_lr=None, weight_init=None, bias_init=None, use_dcn=False, name=''):
        super(Conv2dUnit, self).__init__()
        self.groups = groups
        self.filters = filters
        self.filter_size = filter_size
        self.stride = stride
        self.padding = (filter_size - 1) // 2
        self.act = act
        self.freeze_norm = freeze_norm
        self.is_test = is_test
        self.norm_decay = norm_decay
        self.use_dcn = use_dcn
        self.name = name
        self.lr = lr
        if use_dcn:
            self.conv = DCNv2(input_dim, filters, filter_size=filter_size, stride=stride, padding=(filter_size - 1) // 2, bias_attr=False)
        else:
            if bias_attr:
                blr = lr
                if bias_lr:
                    blr = bias_lr
                self.blr = blr
            self.conv = torch.nn.Conv2d(input_dim, filters, kernel_size=filter_size, stride=stride, padding=(filter_size - 1) // 2, bias=bias_attr)
        self.bn = None
        self.gn = None
        self.af = None
        if bn:
            self.bn = torch.nn.BatchNorm2d(filters)
        if gn:
            self.gn = torch.nn.GroupNorm(num_groups=groups, num_channels=filters)
        if af:
            self.af = AffineChannel(filters)
        self.act = None
        if act == 'relu':
            self.act = torch.nn.ReLU(inplace=True)
        elif act == 'leaky':
            self.act = torch.nn.LeakyReLU(0.1)
        elif act == 'mish':
            self.act = Mish()
        elif act is None:
            pass
        else:
            raise NotImplementedError("Activation '{}' is not implemented.".format(act))

    def freeze(self):
        if isinstance(self.conv, torch.nn.Conv2d):
            self.conv.weight.requires_grad = False
            if self.conv.bias is not None:
                self.conv.bias.requires_grad = False
        elif isinstance(self.conv, DCNv2):
            self.conv.conv_offset.weight.requires_grad = False
            self.conv.conv_offset.bias.requires_grad = False
            self.conv.dcn_weight.requires_grad = False
            if self.conv.dcn_bias is not None:
                self.conv.dcn_bias.requires_grad = False
        else:
            self.conv.weight.requires_grad = False
            if self.conv.bias is not None:
                self.conv.bias.requires_grad = False
        if self.bn is not None:
            self.bn.weight.requires_grad = False
            self.bn.bias.requires_grad = False
        if self.gn is not None:
            self.gn.weight.requires_grad = False
            self.gn.bias.requires_grad = False
        if self.af is not None:
            self.af.weight.requires_grad = False
            self.af.bias.requires_grad = False

    def add_param_group(self, param_groups, base_lr, base_wd):
        if isinstance(self.conv, torch.nn.Conv2d):
            if self.conv.weight.requires_grad:
                param_group_conv = {'params': [self.conv.weight]}
                param_group_conv['lr'] = base_lr * self.lr
                param_group_conv['base_lr'] = base_lr * self.lr
                param_group_conv['weight_decay'] = base_wd
                param_groups.append(param_group_conv)
                if self.conv.bias is not None:
                    if self.conv.bias.requires_grad:
                        param_group_conv_bias = {'params': [self.conv.bias]}
                        param_group_conv_bias['lr'] = base_lr * self.blr
                        param_group_conv_bias['base_lr'] = base_lr * self.blr
                        param_group_conv_bias['weight_decay'] = 0.0
                        param_groups.append(param_group_conv_bias)
        elif isinstance(self.conv, DCNv2):
            if self.conv.conv_offset.weight.requires_grad:
                param_group_conv_offset_w = {'params': [self.conv.conv_offset.weight]}
                param_group_conv_offset_w['lr'] = base_lr * self.lr
                param_group_conv_offset_w['base_lr'] = base_lr * self.lr
                param_group_conv_offset_w['weight_decay'] = base_wd
                param_groups.append(param_group_conv_offset_w)
            if self.conv.conv_offset.bias.requires_grad:
                param_group_conv_offset_b = {'params': [self.conv.conv_offset.bias]}
                param_group_conv_offset_b['lr'] = base_lr * self.lr
                param_group_conv_offset_b['base_lr'] = base_lr * self.lr
                param_group_conv_offset_b['weight_decay'] = base_wd
                param_groups.append(param_group_conv_offset_b)
            if self.conv.dcn_weight.requires_grad:
                param_group_dcn_weight = {'params': [self.conv.dcn_weight]}
                param_group_dcn_weight['lr'] = base_lr * self.lr
                param_group_dcn_weight['base_lr'] = base_lr * self.lr
                param_group_dcn_weight['weight_decay'] = base_wd
                param_groups.append(param_group_dcn_weight)
        else:
            pass
        if self.bn is not None:
            if self.bn.weight.requires_grad:
                param_group_norm_weight = {'params': [self.bn.weight]}
                param_group_norm_weight['lr'] = base_lr * self.lr
                param_group_norm_weight['base_lr'] = base_lr * self.lr
                param_group_norm_weight['weight_decay'] = 0.0
                param_groups.append(param_group_norm_weight)
            if self.bn.bias.requires_grad:
                param_group_norm_bias = {'params': [self.bn.bias]}
                param_group_norm_bias['lr'] = base_lr * self.lr
                param_group_norm_bias['base_lr'] = base_lr * self.lr
                param_group_norm_bias['weight_decay'] = 0.0
                param_groups.append(param_group_norm_bias)
        if self.gn is not None:
            if self.gn.weight.requires_grad:
                param_group_norm_weight = {'params': [self.gn.weight]}
                param_group_norm_weight['lr'] = base_lr * self.lr
                param_group_norm_weight['base_lr'] = base_lr * self.lr
                param_group_norm_weight['weight_decay'] = 0.0
                param_groups.append(param_group_norm_weight)
            if self.gn.bias.requires_grad:
                param_group_norm_bias = {'params': [self.gn.bias]}
                param_group_norm_bias['lr'] = base_lr * self.lr
                param_group_norm_bias['base_lr'] = base_lr * self.lr
                param_group_norm_bias['weight_decay'] = 0.0
                param_groups.append(param_group_norm_bias)
        if self.af is not None:
            if self.af.weight.requires_grad:
                param_group_norm_weight = {'params': [self.af.weight]}
                param_group_norm_weight['lr'] = base_lr * self.lr
                param_group_norm_weight['base_lr'] = base_lr * self.lr
                param_group_norm_weight['weight_decay'] = 0.0
                param_groups.append(param_group_norm_weight)
            if self.af.bias.requires_grad:
                param_group_norm_bias = {'params': [self.af.bias]}
                param_group_norm_bias['lr'] = base_lr * self.lr
                param_group_norm_bias['base_lr'] = base_lr * self.lr
                param_group_norm_bias['weight_decay'] = 0.0
                param_groups.append(param_group_norm_bias)

    def forward(self, x):
        x = self.conv(x)
        if self.bn:
            x = self.bn(x)
        if self.gn:
            x = self.gn(x)
        if self.af:
            x = self.af(x)
        if self.act:
            x = self.act(x)
        return x


class CoordConv(torch.nn.Module):

    def __init__(self, coord_conv=True):
        super(CoordConv, self).__init__()
        self.coord_conv = coord_conv

    def __call__(self, input):
        if not self.coord_conv:
            return input
        b = input.shape[0]
        h = input.shape[2]
        w = input.shape[3]
        x_range = T.arange(0, w, dtype=T.float32, device=input.device) / (w - 1) * 2.0 - 1
        y_range = T.arange(0, h, dtype=T.float32, device=input.device) / (h - 1) * 2.0 - 1
        x_range = x_range[np.newaxis, np.newaxis, np.newaxis, :].repeat((b, 1, h, 1))
        y_range = y_range[np.newaxis, np.newaxis, :, np.newaxis].repeat((b, 1, 1, w))
        offset = T.cat([input, x_range, y_range], dim=1)
        return offset


class SPP(torch.nn.Module):

    def __init__(self, seq='asc'):
        super(SPP, self).__init__()
        assert seq in ['desc', 'asc']
        self.seq = seq

    def __call__(self, x):
        x_1 = x
        x_2 = F.max_pool2d(x, 5, 1, 2)
        x_3 = F.max_pool2d(x, 9, 1, 4)
        x_4 = F.max_pool2d(x, 13, 1, 6)
        if self.seq == 'desc':
            out = torch.cat([x_4, x_3, x_2, x_1], dim=1)
        else:
            out = torch.cat([x_1, x_2, x_3, x_4], dim=1)
        return out


class DropBlock(torch.nn.Module):

    def __init__(self, block_size=3, keep_prob=0.9, is_test=False):
        super(DropBlock, self).__init__()
        self.block_size = block_size
        self.keep_prob = keep_prob
        self.is_test = is_test

    def __call__(self, input):
        if self.is_test:
            return input

        def CalculateGamma(input, block_size, keep_prob):
            h = input.shape[2]
            h = np.array([h])
            h = torch.tensor(h, dtype=torch.float32, device=input.device)
            feat_shape_t = h.reshape((1, 1, 1, 1))
            feat_area = torch.pow(feat_shape_t, 2)
            block_shape_t = torch.zeros((1, 1, 1, 1), dtype=torch.float32, device=input.device) + block_size
            block_area = torch.pow(block_shape_t, 2)
            useful_shape_t = feat_shape_t - block_shape_t + 1
            useful_area = torch.pow(useful_shape_t, 2)
            upper_t = feat_area * (1 - keep_prob)
            bottom_t = block_area * useful_area
            output = upper_t / bottom_t
            return output
        gamma = CalculateGamma(input, block_size=self.block_size, keep_prob=self.keep_prob)
        input_shape = input.shape
        p = gamma.repeat(input_shape)
        input_shape_tmp = input.shape
        random_matrix = torch.rand(input_shape_tmp, device=input.device)
        one_zero_m = (random_matrix < p).float()
        mask_flag = torch.nn.functional.max_pool2d(one_zero_m, (self.block_size, self.block_size), stride=1, padding=1)
        mask = 1.0 - mask_flag
        elem_numel = input_shape[0] * input_shape[1] * input_shape[2] * input_shape[3]
        elem_numel_m = float(elem_numel)
        elem_sum = mask.sum()
        output = input * mask * elem_numel_m / elem_sum
        return output


class DCNv2_Slow(torch.nn.Module):
    """
    自实现的DCNv2，非常慢=_=!
    """

    def __init__(self, input_dim, filters, filter_size, stride=1, padding=0, bias_attr=False, distribution='normal', gain=1):
        super(DCNv2_Slow, self).__init__()
        assert distribution in ['uniform', 'normal']
        self.input_dim = input_dim
        self.filters = filters
        self.filter_size = filter_size
        self.stride = stride
        self.padding = padding
        self.conv_offset = torch.nn.Conv2d(input_dim, filter_size * filter_size * 3, kernel_size=filter_size, stride=stride, padding=padding, bias=True)
        torch.nn.init.constant_(self.conv_offset.weight, 0.0)
        torch.nn.init.constant_(self.conv_offset.bias, 0.0)
        self.sigmoid = torch.nn.Sigmoid()
        self.dcn_weight = torch.nn.Parameter(torch.randn(filters, input_dim, filter_size, filter_size))
        self.dcn_bias = None
        if bias_attr:
            self.dcn_bias = torch.nn.Parameter(torch.randn(filters))
            torch.nn.init.constant_(self.dcn_bias, 0.0)
        if distribution == 'uniform':
            torch.nn.init.xavier_uniform_(self.dcn_weight, gain=gain)
        else:
            torch.nn.init.xavier_normal_(self.dcn_weight, gain=gain)

    def get_pix(self, input, y, x):
        _x = x.int()
        _y = y.int()
        return input[:, _y:_y + 1, _x:_x + 1]

    def bilinear(self, input, h, w):
        C, height, width = input.shape
        y0 = torch.floor(h)
        x0 = torch.floor(w)
        y1 = y0 + 1
        x1 = x0 + 1
        lh = h - y0
        lw = w - x0
        hh = 1 - lh
        hw = 1 - lw
        v1 = torch.zeros((C, 1, 1), device=input.device)
        if y0 >= 0 and x0 >= 0 and y0 <= height - 1 and x0 <= width - 1:
            v1 = self.get_pix(input, y0, x0)
        v2 = torch.zeros((C, 1, 1), device=input.device)
        if y0 >= 0 and x1 >= 0 and y0 <= height - 1 and x1 <= width - 1:
            v2 = self.get_pix(input, y0, x1)
        v3 = torch.zeros((C, 1, 1), device=input.device)
        if y1 >= 0 and x0 >= 0 and y1 <= height - 1 and x0 <= width - 1:
            v3 = self.get_pix(input, y1, x0)
        v4 = torch.zeros((C, 1, 1), device=input.device)
        if y1 >= 0 and x1 >= 0 and y1 <= height - 1 and x1 <= width - 1:
            v4 = self.get_pix(input, y1, x1)
        w1 = hh * hw
        w2 = hh * lw
        w3 = lh * hw
        w4 = lh * lw
        out = w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4
        return out

    def forward(self, x):
        filter_size = self.filter_size
        stride = self.stride
        padding = self.padding
        dcn_weight = self.dcn_weight
        dcn_bias = self.dcn_bias
        offset_mask = self.conv_offset(x)
        offset = offset_mask[:, :filter_size ** 2 * 2, :, :]
        mask = offset_mask[:, filter_size ** 2 * 2:, :, :]
        mask = self.sigmoid(mask)
        N, C, H, W = x.shape
        out_C, in_C, kH, kW = dcn_weight.shape
        out_W = (W + 2 * padding - (kW - 1)) // stride
        out_H = (H + 2 * padding - (kH - 1)) // stride
        out = torch.zeros((N, out_C, out_H, out_W), device=x.device)
        for bid in range(N):
            input = x[bid]
            sample_offset = offset[bid]
            sample_mask = mask[bid]
            for i in range(out_H):
                for j in range(out_W):
                    ori_x = j * stride - padding
                    ori_y = i * stride - padding
                    point_offset = sample_offset[:, i, j]
                    point_mask = sample_mask[:, i, j]
                    part_x = []
                    for i2 in range(filter_size):
                        for j2 in range(filter_size):
                            _offset_y = point_offset[2 * (i2 * filter_size + j2)]
                            _offset_x = point_offset[2 * (i2 * filter_size + j2) + 1]
                            mask_ = point_mask[i2 * filter_size + j2]
                            h_im = ori_y + i2 + _offset_y
                            w_im = ori_x + j2 + _offset_x
                            value = self.bilinear(input, h_im, w_im)
                            value = value * mask_
                            part_x.append(value)
                    part_x = torch.cat(part_x, 1)
                    part_x = torch.reshape(part_x, (in_C, filter_size, filter_size))
                    exp_part_x = part_x.unsqueeze(0)
                    mul = exp_part_x * dcn_weight
                    mul = mul.sum((1, 2, 3))
                    if dcn_bias is not None:
                        mul += dcn_bias
                    out[bid, :, i, j] = mul
        return out


class DetectionBlock(torch.nn.Module):

    def __init__(self, in_c, channel, coord_conv=True, bn=0, gn=0, af=0, norm_decay=0.0, conv_block_num=2, is_first=False, use_spp=True, drop_block=True, block_size=3, keep_prob=0.9, is_test=True, name=''):
        super(DetectionBlock, self).__init__()
        assert channel % 2 == 0, 'channel {} cannot be divided by 2'.format(channel)
        self.norm_decay = norm_decay
        self.use_spp = use_spp
        self.coord_conv = coord_conv
        self.is_first = is_first
        self.is_test = is_test
        self.drop_block = drop_block
        self.block_size = block_size
        self.keep_prob = keep_prob
        self.layers = torch.nn.ModuleList()
        self.tip_layers = torch.nn.ModuleList()
        for j in range(conv_block_num):
            coordConv = CoordConv(coord_conv)
            input_c = in_c + 2 if coord_conv else in_c
            conv_unit1 = Conv2dUnit(input_c, channel, 1, stride=1, bn=bn, gn=gn, af=af, act='leaky', norm_decay=self.norm_decay, name='{}.{}.0'.format(name, j))
            self.layers.append(coordConv)
            self.layers.append(conv_unit1)
            if self.use_spp and is_first and j == 1:
                spp = SPP()
                conv_unit2 = Conv2dUnit(channel * 4, 512, 1, stride=1, bn=bn, gn=gn, af=af, act='leaky', norm_decay=self.norm_decay, name='{}.{}.spp.conv'.format(name, j))
                conv_unit3 = Conv2dUnit(512, channel * 2, 3, stride=1, bn=bn, gn=gn, af=af, act='leaky', norm_decay=self.norm_decay, name='{}.{}.1'.format(name, j))
                self.layers.append(spp)
                self.layers.append(conv_unit2)
                self.layers.append(conv_unit3)
            else:
                conv_unit3 = Conv2dUnit(channel, channel * 2, 3, stride=1, bn=bn, gn=gn, af=af, act='leaky', norm_decay=self.norm_decay, name='{}.{}.1'.format(name, j))
                self.layers.append(conv_unit3)
            if self.drop_block and j == 0 and not is_first:
                dropBlock = DropBlock(block_size=self.block_size, keep_prob=self.keep_prob, is_test=is_test)
                self.layers.append(dropBlock)
            in_c = channel * 2
        if self.drop_block and is_first:
            dropBlock = DropBlock(block_size=self.block_size, keep_prob=self.keep_prob, is_test=is_test)
            self.layers.append(dropBlock)
        coordConv = CoordConv(coord_conv)
        if conv_block_num == 0:
            input_c = in_c + 2 if coord_conv else in_c
        else:
            input_c = channel * 2 + 2 if coord_conv else channel * 2
        conv_unit = Conv2dUnit(input_c, channel, 1, stride=1, bn=bn, gn=gn, af=af, act='leaky', norm_decay=self.norm_decay, name='{}.2'.format(name))
        self.layers.append(coordConv)
        self.layers.append(conv_unit)
        coordConv = CoordConv(coord_conv)
        input_c = channel + 2 if coord_conv else channel
        conv_unit = Conv2dUnit(input_c, channel * 2, 3, stride=1, bn=bn, gn=gn, af=af, act='leaky', norm_decay=self.norm_decay, name='{}.tip'.format(name))
        self.tip_layers.append(coordConv)
        self.tip_layers.append(conv_unit)

    def __call__(self, input):
        conv = input
        for ly in self.layers:
            conv = ly(conv)
        route = conv
        tip = conv
        for ly in self.tip_layers:
            tip = ly(tip)
        return route, tip

    def add_param_group(self, param_groups, base_lr, base_wd):
        for layer in self.layers:
            if isinstance(layer, Conv2dUnit):
                layer.add_param_group(param_groups, base_lr, base_wd)
        for layer in self.tip_layers:
            if isinstance(layer, Conv2dUnit):
                layer.add_param_group(param_groups, base_lr, base_wd)


def _de_sigmoid(x, eps=1e-07):
    x = torch.clamp(x, eps, 1 / eps)
    x = 1.0 / x - 1.0
    x = torch.clamp(x, eps, 1 / eps)
    x = -torch.log(x)
    return x


def _postprocess_output(ioup, output, an_num, num_classes, iou_aware_factor):
    """
    post process output objectness score
    """
    tensors = []
    stride = output.shape[1] // an_num
    for m in range(an_num):
        tensors.append(output[:, stride * m:stride * m + 4, :, :])
        obj = output[:, stride * m + 4:stride * m + 5, :, :]
        obj = torch.sigmoid(obj)
        ip = ioup[:, m:m + 1, :, :]
        new_obj = torch.pow(obj, 1 - iou_aware_factor) * torch.pow(ip, iou_aware_factor)
        new_obj = _de_sigmoid(new_obj)
        tensors.append(new_obj)
        tensors.append(output[:, stride * m + 5:stride * m + 5 + num_classes, :, :])
    output = torch.cat(tensors, dim=1)
    return output


def _split_ioup(output, an_num, num_classes):
    """
    Split new output feature map to output, predicted iou
    along channel dimension
    """
    ioup = output[:, :an_num, :, :]
    ioup = torch.sigmoid(ioup)
    oriout = output[:, an_num:, :, :]
    return ioup, oriout


def get_iou_aware_score(output, an_num, num_classes, iou_aware_factor):
    ioup, output = _split_ioup(output, an_num, num_classes)
    output = _postprocess_output(ioup, output, an_num, num_classes, iou_aware_factor)
    return output


def get_norm(norm_type):
    bn = 0
    gn = 0
    af = 0
    if norm_type == 'bn':
        bn = 1
    elif norm_type == 'sync_bn':
        bn = 1
    elif norm_type == 'gn':
        gn = 1
    elif norm_type == 'affine_channel':
        af = 1
    return bn, gn, af


def intersect(box_a, box_b):
    """计算两组矩形两两之间相交区域的面积
    Args:
        box_a: (tensor) bounding boxes, Shape: [A, 4].
        box_b: (tensor) bounding boxes, Shape: [B, 4].
    Return:
      (tensor) intersection area, Shape: [A, B].
    """
    A = box_a.size(0)
    B = box_b.size(0)
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2), box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2), box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp(max_xy - min_xy, min=0)
    return inter[:, :, 0] * inter[:, :, 1]


def jaccard(box_a, box_b):
    """计算两组矩形两两之间的iou
    Args:
        box_a: (tensor) bounding boxes, Shape: [A, 4].
        box_b: (tensor) bounding boxes, Shape: [B, 4].
    Return:
        ious: (tensor) Shape: [A, B]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2] - box_a[:, 0]) * (box_a[:, 3] - box_a[:, 1])).unsqueeze(1).expand_as(inter)
    area_b = ((box_b[:, 2] - box_b[:, 0]) * (box_b[:, 3] - box_b[:, 1])).unsqueeze(0).expand_as(inter)
    union = area_a + area_b - inter
    return inter / union


def _matrix_nms(bboxes, cate_labels, cate_scores, kernel='gaussian', sigma=2.0):
    """Matrix NMS for multi-class bboxes.
    Args:
        bboxes (Tensor): shape (n, 4)
        cate_labels (Tensor): shape (n), mask labels in descending order
        cate_scores (Tensor): shape (n), mask scores in descending order
        kernel (str):  'linear' or 'gaussian'
        sigma (float): std in gaussian method
    Returns:
        Tensor: cate_scores_update, tensors of shape (n)
    """
    n_samples = len(cate_labels)
    if n_samples == 0:
        return []
    iou_matrix = jaccard(bboxes, bboxes)
    iou_matrix = iou_matrix.triu(diagonal=1)
    cate_labels_x = cate_labels.expand(n_samples, n_samples)
    label_matrix = (cate_labels_x == cate_labels_x.transpose(1, 0)).float().triu(diagonal=1)
    compensate_iou, _ = (iou_matrix * label_matrix).max(0)
    compensate_iou = compensate_iou.expand(n_samples, n_samples).transpose(1, 0)
    decay_iou = iou_matrix * label_matrix
    if kernel == 'gaussian':
        decay_matrix = torch.exp(-1 * sigma * decay_iou ** 2)
        compensate_matrix = torch.exp(-1 * sigma * compensate_iou ** 2)
        decay_coefficient, _ = (decay_matrix / compensate_matrix).min(0)
    elif kernel == 'linear':
        decay_matrix = (1 - decay_iou) / (1 - compensate_iou)
        decay_coefficient, _ = decay_matrix.min(0)
    else:
        raise NotImplementedError
    cate_scores_update = cate_scores * decay_coefficient
    return cate_scores_update


def matrix_nms(bboxes, scores, score_threshold, post_threshold, nms_top_k, keep_top_k, use_gaussian=False, gaussian_sigma=2.0):
    inds = scores > score_threshold
    cate_scores = scores[inds]
    if len(cate_scores) == 0:
        return torch.zeros((1, 6), device=bboxes.device) - 1.0
    inds = inds.nonzero()
    cate_labels = inds[:, 1]
    bboxes = bboxes[inds[:, 0]]
    sort_inds = torch.argsort(cate_scores, descending=True)
    if nms_top_k > 0 and len(sort_inds) > nms_top_k:
        sort_inds = sort_inds[:nms_top_k]
    bboxes = bboxes[sort_inds, :]
    cate_scores = cate_scores[sort_inds]
    cate_labels = cate_labels[sort_inds]
    kernel = 'gaussian' if use_gaussian else 'linear'
    cate_scores = _matrix_nms(bboxes, cate_labels, cate_scores, kernel=kernel, sigma=gaussian_sigma)
    keep = cate_scores >= post_threshold
    if keep.sum() == 0:
        return torch.zeros((1, 6), device=bboxes.device) - 1.0
    bboxes = bboxes[keep, :]
    cate_scores = cate_scores[keep]
    cate_labels = cate_labels[keep]
    sort_inds = torch.argsort(cate_scores, descending=True)
    if len(sort_inds) > keep_top_k:
        sort_inds = sort_inds[:keep_top_k]
    bboxes = bboxes[sort_inds, :]
    cate_scores = cate_scores[sort_inds]
    cate_labels = cate_labels[sort_inds]
    cate_scores = cate_scores.unsqueeze(1)
    cate_labels = cate_labels.unsqueeze(1).float()
    pred = torch.cat([cate_labels, cate_scores, bboxes], 1)
    return pred


def yolo_box(conv_output, anchors, stride, num_classes, scale_x_y, im_size, clip_bbox, conf_thresh):
    conv_output = conv_output.permute(0, 2, 3, 1)
    conv_shape = conv_output.shape
    batch_size = conv_shape[0]
    output_size = conv_shape[1]
    anchor_per_scale = len(anchors)
    conv_output = conv_output.reshape((batch_size, output_size, output_size, anchor_per_scale, 5 + num_classes))
    conv_raw_dxdy = conv_output[:, :, :, :, 0:2]
    conv_raw_dwdh = conv_output[:, :, :, :, 2:4]
    conv_raw_conf = conv_output[:, :, :, :, 4:5]
    conv_raw_prob = conv_output[:, :, :, :, 5:]
    rows = T.arange(0, output_size, dtype=T.float32, device=conv_raw_dxdy.device)
    cols = T.arange(0, output_size, dtype=T.float32, device=conv_raw_dxdy.device)
    rows = rows[np.newaxis, np.newaxis, :, np.newaxis, np.newaxis].repeat((1, output_size, 1, 1, 1))
    cols = cols[np.newaxis, :, np.newaxis, np.newaxis, np.newaxis].repeat((1, 1, output_size, 1, 1))
    offset = T.cat([rows, cols], dim=-1)
    offset = offset.repeat((batch_size, 1, 1, anchor_per_scale, 1))
    pred_xy = (scale_x_y * T.sigmoid(conv_raw_dxdy) + offset - (scale_x_y - 1.0) * 0.5) * stride
    _anchors = T.Tensor(anchors)
    pred_wh = T.exp(conv_raw_dwdh) * _anchors
    pred_xyxy = T.cat([pred_xy - pred_wh / 2, pred_xy + pred_wh / 2], dim=-1)
    pred_conf = T.sigmoid(conv_raw_conf)
    pred_prob = T.sigmoid(conv_raw_prob)
    pred_scores = pred_conf * pred_prob
    pred_xyxy = pred_xyxy.reshape((batch_size, output_size * output_size * anchor_per_scale, 4))
    pred_scores = pred_scores.reshape((batch_size, pred_xyxy.shape[1], num_classes))
    _im_size_h = im_size[:, 0:1]
    _im_size_w = im_size[:, 1:2]
    _im_size = T.cat([_im_size_w, _im_size_h], 1)
    _im_size = _im_size.unsqueeze(1)
    _im_size = _im_size.repeat((1, pred_xyxy.shape[1], 1))
    pred_x0y0 = pred_xyxy[:, :, 0:2] / output_size / stride * _im_size
    pred_x1y1 = pred_xyxy[:, :, 2:4] / output_size / stride * _im_size
    if clip_bbox:
        x0 = pred_x0y0[:, :, 0:1]
        y0 = pred_x0y0[:, :, 1:2]
        x1 = pred_x1y1[:, :, 0:1]
        y1 = pred_x1y1[:, :, 1:2]
        x0 = torch.where(x0 < 0, x0 * 0, x0)
        y0 = torch.where(y0 < 0, y0 * 0, y0)
        x1 = torch.where(x1 > _im_size[:, :, 0:1], _im_size[:, :, 0:1], x1)
        y1 = torch.where(y1 > _im_size[:, :, 1:2], _im_size[:, :, 1:2], y1)
        pred_xyxy = T.cat([x0, y0, x1, y1], -1)
    else:
        pred_xyxy = T.cat([pred_x0y0, pred_x1y1], -1)
    return pred_xyxy, pred_scores


class YOLOv3Head(torch.nn.Module):

    def __init__(self, conv_block_num=2, num_classes=80, anchors=[[10, 13], [16, 30], [33, 23], [30, 61], [62, 45], [59, 119], [116, 90], [156, 198], [373, 326]], anchor_masks=[[6, 7, 8], [3, 4, 5], [0, 1, 2]], norm_type='bn', norm_decay=0.0, coord_conv=True, iou_aware=True, iou_aware_factor=0.4, block_size=3, scale_x_y=1.05, spp=True, drop_block=True, keep_prob=0.9, clip_bbox=True, yolo_loss=None, downsample=[32, 16, 8], in_channels=[2048, 1024, 512], nms_cfg=None, focalloss_on_obj=False, prior_prob=0.01, is_train=False):
        super(YOLOv3Head, self).__init__()
        self.conv_block_num = conv_block_num
        self.num_classes = num_classes
        self.norm_type = norm_type
        self.norm_decay = norm_decay
        self.coord_conv = coord_conv
        self.iou_aware = iou_aware
        self.iou_aware_factor = iou_aware_factor
        self.scale_x_y = scale_x_y
        self.use_spp = spp
        self.drop_block = drop_block
        self.keep_prob = keep_prob
        self.clip_bbox = clip_bbox
        self.anchors = anchors
        self.anchor_masks = anchor_masks
        self.block_size = block_size
        self.downsample = downsample
        self.in_channels = in_channels
        self.yolo_loss = yolo_loss
        self.nms_cfg = nms_cfg
        self.focalloss_on_obj = focalloss_on_obj
        self.prior_prob = prior_prob
        self.is_train = is_train
        _anchors = copy.deepcopy(anchors)
        _anchors = np.array(_anchors)
        _anchors = _anchors.astype(np.float32)
        self._anchors = _anchors
        self.mask_anchors = []
        for m in anchor_masks:
            temp = []
            for aid in m:
                temp += anchors[aid]
            self.mask_anchors.append(temp)
        assert norm_type in ['bn', 'sync_bn', 'gn', 'affine_channel']
        bn, gn, af = get_norm(norm_type)
        self.detection_blocks = torch.nn.ModuleList()
        self.yolo_output_convs = torch.nn.ModuleList()
        self.upsample_layers = torch.nn.ModuleList()
        out_layer_num = len(downsample)
        for i in range(out_layer_num):
            in_c = self.in_channels[i]
            if i > 0:
                in_c = self.in_channels[i] + 512 // 2 ** i
            _detection_block = DetectionBlock(in_c=in_c, channel=64 * 2 ** out_layer_num // 2 ** i, coord_conv=self.coord_conv, bn=bn, gn=gn, af=af, norm_decay=self.norm_decay, is_first=i == 0, conv_block_num=self.conv_block_num, use_spp=self.use_spp, drop_block=self.drop_block, block_size=self.block_size, keep_prob=self.keep_prob, is_test=not self.is_train, name='yolo_block.{}'.format(i))
            if self.iou_aware:
                num_filters = len(self.anchor_masks[i]) * (self.num_classes + 6)
            else:
                num_filters = len(self.anchor_masks[i]) * (self.num_classes + 5)
            bias_init = None
            yolo_output_conv = Conv2dUnit(64 * 2 ** out_layer_num // 2 ** i * 2, num_filters, 1, stride=1, bias_attr=True, act=None, bias_init=bias_init, name='yolo_output.{}.conv'.format(i))
            self.detection_blocks.append(_detection_block)
            self.yolo_output_convs.append(yolo_output_conv)
            if i < out_layer_num - 1:
                conv_unit = Conv2dUnit(64 * 2 ** out_layer_num // 2 ** i, 256 // 2 ** i, 1, stride=1, bn=bn, gn=gn, af=af, act='leaky', norm_decay=self.norm_decay, name='yolo_transition.{}'.format(i))
                upsample = torch.nn.Upsample(scale_factor=2, mode='nearest')
                self.upsample_layers.append(conv_unit)
                self.upsample_layers.append(upsample)

    def add_param_group(self, param_groups, base_lr, base_wd):
        for detection_block in self.detection_blocks:
            detection_block.add_param_group(param_groups, base_lr, base_wd)
        for layer in self.yolo_output_convs:
            layer.add_param_group(param_groups, base_lr, base_wd)
        for layer in self.upsample_layers:
            if isinstance(layer, Conv2dUnit):
                layer.add_param_group(param_groups, base_lr, base_wd)

    def set_dropblock(self, is_test):
        for detection_block in self.detection_blocks:
            for l in detection_block.layers:
                if isinstance(l, DropBlock):
                    l.is_test = is_test

    def _get_outputs(self, body_feats):
        outputs = []
        out_layer_num = len(self.anchor_masks)
        blocks = body_feats[-1:-out_layer_num - 1:-1]
        route = None
        for i, block in enumerate(blocks):
            if i > 0:
                block = torch.cat([route, block], dim=1)
            route, tip = self.detection_blocks[i](block)
            block_out = self.yolo_output_convs[i](tip)
            outputs.append(block_out)
            if i < out_layer_num - 1:
                route = self.upsample_layers[i * 2](route)
                route = self.upsample_layers[i * 2 + 1](route)
        return outputs

    def get_loss(self, input, gt_box, gt_label, gt_score, targets):
        """
        Get final loss of network of YOLOv3.

        Args:
            input (list): List of Variables, output of backbone stages
            gt_box (Variable): The ground-truth boudding boxes.
            gt_label (Variable): The ground-truth class labels.
            gt_score (Variable): The ground-truth boudding boxes mixup scores.
            targets ([Variables]): List of Variables, the targets for yolo
                                   loss calculatation.

        Returns:
            loss (Variable): The loss Variable of YOLOv3 network.

        """
        outputs = self._get_outputs(input)
        return self.yolo_loss(outputs, gt_box, gt_label, gt_score, targets, self.anchors, self.anchor_masks, self.mask_anchors, self.num_classes)

    def get_prediction(self, body_feats, im_size):
        """
        Get prediction result of YOLOv3 network

        Args:
            input (list): List of Variables, output of backbone stages
            im_size (Variable): Variable of size([h, w]) of each image

        Returns:
            pred (Variable): shape = [bs, keep_top_k, 6]

        """
        outputs = self._get_outputs(body_feats)
        boxes = []
        scores = []
        for i, output in enumerate(outputs):
            if self.iou_aware:
                output = get_iou_aware_score(output, len(self.anchor_masks[i]), self.num_classes, self.iou_aware_factor)
            box, score = yolo_box(output, self._anchors[self.anchor_masks[i]], self.downsample[i], self.num_classes, self.scale_x_y, im_size, self.clip_bbox, conf_thresh=self.nms_cfg['score_threshold'])
            boxes.append(box)
            scores.append(score)
        yolo_boxes = torch.cat(boxes, dim=1)
        yolo_scores = torch.cat(scores, dim=1)
        preds = []
        nms_cfg = copy.deepcopy(self.nms_cfg)
        nms_type = nms_cfg.pop('nms_type')
        batch_size = yolo_boxes.shape[0]
        if nms_type == 'matrix_nms':
            for i in range(batch_size):
                pred = matrix_nms(yolo_boxes[i, :, :], yolo_scores[i, :, :], **nms_cfg)
                preds.append(pred)
        return preds


class PPYOLO(torch.nn.Module):

    def __init__(self, backbone, head):
        super(PPYOLO, self).__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x, im_size, eval=True, gt_box=None, gt_label=None, gt_score=None, targets=None):
        body_feats = self.backbone(x)
        if eval:
            out = self.head.get_prediction(body_feats, im_size)
        else:
            out = self.head.get_loss(body_feats, gt_box, gt_label, gt_score, targets)
        return out

    def add_param_group(self, param_groups, base_lr, base_wd):
        self.backbone.add_param_group(param_groups, base_lr, base_wd)
        self.head.add_param_group(param_groups, base_lr, base_wd)


class ConvBlock(torch.nn.Module):

    def __init__(self, in_c, filters, bn, gn, af, freeze_norm, norm_decay, lr, use_dcn=False, stride=2, downsample_in3x3=True, is_first=False, block_name=''):
        super(ConvBlock, self).__init__()
        filters1, filters2, filters3 = filters
        if downsample_in3x3 == True:
            stride1, stride2 = 1, stride
        else:
            stride1, stride2 = stride, 1
        self.is_first = is_first
        self.conv1 = Conv2dUnit(in_c, filters1, 1, stride=stride1, bn=bn, gn=gn, af=af, freeze_norm=freeze_norm, norm_decay=norm_decay, lr=lr, act='relu', name=block_name + '_branch2a')
        self.conv2 = Conv2dUnit(filters1, filters2, 3, stride=stride2, bn=bn, gn=gn, af=af, freeze_norm=freeze_norm, norm_decay=norm_decay, lr=lr, act='relu', use_dcn=use_dcn, name=block_name + '_branch2b')
        self.conv3 = Conv2dUnit(filters2, filters3, 1, stride=1, bn=bn, gn=gn, af=af, freeze_norm=freeze_norm, norm_decay=norm_decay, lr=lr, act=None, name=block_name + '_branch2c')
        if not self.is_first:
            self.avg_pool = torch.nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
            self.conv4 = Conv2dUnit(in_c, filters3, 1, stride=1, bn=bn, gn=gn, af=af, freeze_norm=freeze_norm, norm_decay=norm_decay, lr=lr, act=None, name=block_name + '_branch1')
        else:
            self.conv4 = Conv2dUnit(in_c, filters3, 1, stride=stride, bn=bn, gn=gn, af=af, freeze_norm=freeze_norm, norm_decay=norm_decay, lr=lr, act=None, name=block_name + '_branch1')
        self.act = torch.nn.ReLU(inplace=True)

    def freeze(self):
        self.conv1.freeze()
        self.conv2.freeze()
        self.conv3.freeze()
        self.conv4.freeze()

    def add_param_group(self, param_groups, base_lr, base_wd):
        self.conv1.add_param_group(param_groups, base_lr, base_wd)
        self.conv2.add_param_group(param_groups, base_lr, base_wd)
        self.conv3.add_param_group(param_groups, base_lr, base_wd)
        self.conv4.add_param_group(param_groups, base_lr, base_wd)

    def __call__(self, input_tensor):
        x = self.conv1(input_tensor)
        x = self.conv2(x)
        x = self.conv3(x)
        if not self.is_first:
            input_tensor = self.avg_pool(input_tensor)
        shortcut = self.conv4(input_tensor)
        x = x + shortcut
        x = self.act(x)
        return x


class IdentityBlock(torch.nn.Module):

    def __init__(self, in_c, filters, bn, gn, af, freeze_norm, norm_decay, lr, use_dcn=False, block_name=''):
        super(IdentityBlock, self).__init__()
        filters1, filters2, filters3 = filters
        self.conv1 = Conv2dUnit(in_c, filters1, 1, stride=1, bn=bn, gn=gn, af=af, freeze_norm=freeze_norm, norm_decay=norm_decay, lr=lr, act='relu', name=block_name + '_branch2a')
        self.conv2 = Conv2dUnit(filters1, filters2, 3, stride=1, bn=bn, gn=gn, af=af, freeze_norm=freeze_norm, norm_decay=norm_decay, lr=lr, act='relu', use_dcn=use_dcn, name=block_name + '_branch2b')
        self.conv3 = Conv2dUnit(filters2, filters3, 1, stride=1, bn=bn, gn=gn, af=af, freeze_norm=freeze_norm, norm_decay=norm_decay, lr=lr, act=None, name=block_name + '_branch2c')
        self.act = torch.nn.ReLU(inplace=True)

    def freeze(self):
        self.conv1.freeze()
        self.conv2.freeze()
        self.conv3.freeze()

    def add_param_group(self, param_groups, base_lr, base_wd):
        self.conv1.add_param_group(param_groups, base_lr, base_wd)
        self.conv2.add_param_group(param_groups, base_lr, base_wd)
        self.conv3.add_param_group(param_groups, base_lr, base_wd)

    def __call__(self, input_tensor):
        x = self.conv1(input_tensor)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x + input_tensor
        x = self.act(x)
        return x


class Resnet50Vd(torch.nn.Module):

    def __init__(self, norm_type='bn', feature_maps=[3, 4, 5], dcn_v2_stages=[5], downsample_in3x3=True, freeze_at=0, freeze_norm=False, norm_decay=0.0, lr_mult_list=[1.0, 1.0, 1.0, 1.0]):
        super(Resnet50Vd, self).__init__()
        self.norm_type = norm_type
        self.feature_maps = feature_maps
        assert freeze_at in [0, 1, 2, 3, 4, 5]
        assert len(lr_mult_list) == 4, 'lr_mult_list length must be 4 but got {}'.format(len(lr_mult_list))
        self.lr_mult_list = lr_mult_list
        self.freeze_at = freeze_at
        assert norm_type in ['bn', 'sync_bn', 'gn', 'affine_channel']
        bn, gn, af = get_norm(norm_type)
        self.stage1_conv1_1 = Conv2dUnit(3, 32, 3, stride=2, bn=bn, gn=gn, af=af, freeze_norm=freeze_norm, norm_decay=norm_decay, act='relu', name='conv1_1')
        self.stage1_conv1_2 = Conv2dUnit(32, 32, 3, stride=1, bn=bn, gn=gn, af=af, freeze_norm=freeze_norm, norm_decay=norm_decay, act='relu', name='conv1_2')
        self.stage1_conv1_3 = Conv2dUnit(32, 64, 3, stride=1, bn=bn, gn=gn, af=af, freeze_norm=freeze_norm, norm_decay=norm_decay, act='relu', name='conv1_3')
        self.pool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.stage2_0 = ConvBlock(64, [64, 64, 256], bn, gn, af, freeze_norm, norm_decay, lr_mult_list[0], stride=1, downsample_in3x3=downsample_in3x3, is_first=True, block_name='res2a')
        self.stage2_1 = IdentityBlock(256, [64, 64, 256], bn, gn, af, freeze_norm, norm_decay, lr_mult_list[0], block_name='res2b')
        self.stage2_2 = IdentityBlock(256, [64, 64, 256], bn, gn, af, freeze_norm, norm_decay, lr_mult_list[0], block_name='res2c')
        use_dcn = 3 in dcn_v2_stages
        self.stage3_0 = ConvBlock(256, [128, 128, 512], bn, gn, af, freeze_norm, norm_decay, lr_mult_list[1], use_dcn=use_dcn, downsample_in3x3=downsample_in3x3, block_name='res3a')
        self.stage3_1 = IdentityBlock(512, [128, 128, 512], bn, gn, af, freeze_norm, norm_decay, lr_mult_list[1], use_dcn=use_dcn, block_name='res3b')
        self.stage3_2 = IdentityBlock(512, [128, 128, 512], bn, gn, af, freeze_norm, norm_decay, lr_mult_list[1], use_dcn=use_dcn, block_name='res3c')
        self.stage3_3 = IdentityBlock(512, [128, 128, 512], bn, gn, af, freeze_norm, norm_decay, lr_mult_list[1], use_dcn=use_dcn, block_name='res3d')
        use_dcn = 4 in dcn_v2_stages
        self.stage4_0 = ConvBlock(512, [256, 256, 1024], bn, gn, af, freeze_norm, norm_decay, lr_mult_list[2], use_dcn=use_dcn, downsample_in3x3=downsample_in3x3, block_name='res4a')
        self.stage4_1 = IdentityBlock(1024, [256, 256, 1024], bn, gn, af, freeze_norm, norm_decay, lr_mult_list[2], use_dcn=use_dcn, block_name='res4b')
        self.stage4_2 = IdentityBlock(1024, [256, 256, 1024], bn, gn, af, freeze_norm, norm_decay, lr_mult_list[2], use_dcn=use_dcn, block_name='res4c')
        self.stage4_3 = IdentityBlock(1024, [256, 256, 1024], bn, gn, af, freeze_norm, norm_decay, lr_mult_list[2], use_dcn=use_dcn, block_name='res4d')
        self.stage4_4 = IdentityBlock(1024, [256, 256, 1024], bn, gn, af, freeze_norm, norm_decay, lr_mult_list[2], use_dcn=use_dcn, block_name='res4e')
        self.stage4_5 = IdentityBlock(1024, [256, 256, 1024], bn, gn, af, freeze_norm, norm_decay, lr_mult_list[2], use_dcn=use_dcn, block_name='res4f')
        use_dcn = 5 in dcn_v2_stages
        self.stage5_0 = ConvBlock(1024, [512, 512, 2048], bn, gn, af, freeze_norm, norm_decay, lr_mult_list[3], use_dcn=use_dcn, downsample_in3x3=downsample_in3x3, block_name='res5a')
        self.stage5_1 = IdentityBlock(2048, [512, 512, 2048], bn, gn, af, freeze_norm, norm_decay, lr_mult_list[3], use_dcn=use_dcn, block_name='res5b')
        self.stage5_2 = IdentityBlock(2048, [512, 512, 2048], bn, gn, af, freeze_norm, norm_decay, lr_mult_list[3], use_dcn=use_dcn, block_name='res5c')

    def forward(self, input_tensor):
        x = self.stage1_conv1_1(input_tensor)
        x = self.stage1_conv1_2(x)
        x = self.stage1_conv1_3(x)
        x = self.pool(x)
        x = self.stage2_0(x)
        x = self.stage2_1(x)
        s4 = self.stage2_2(x)
        x = self.stage3_0(s4)
        x = self.stage3_1(x)
        x = self.stage3_2(x)
        s8 = self.stage3_3(x)
        x = self.stage4_0(s8)
        x = self.stage4_1(x)
        x = self.stage4_2(x)
        x = self.stage4_3(x)
        x = self.stage4_4(x)
        s16 = self.stage4_5(x)
        x = self.stage5_0(s16)
        x = self.stage5_1(x)
        s32 = self.stage5_2(x)
        outs = []
        if 2 in self.feature_maps:
            outs.append(s4)
        if 3 in self.feature_maps:
            outs.append(s8)
        if 4 in self.feature_maps:
            outs.append(s16)
        if 5 in self.feature_maps:
            outs.append(s32)
        return outs

    def get_block(self, name):
        layer = getattr(self, name)
        return layer

    def freeze(self):
        freeze_at = self.freeze_at
        if freeze_at >= 1:
            self.stage1_conv1_1.freeze()
            self.stage1_conv1_2.freeze()
            self.stage1_conv1_3.freeze()
        if freeze_at >= 2:
            self.stage2_0.freeze()
            self.stage2_1.freeze()
            self.stage2_2.freeze()
        if freeze_at >= 3:
            self.stage3_0.freeze()
            self.stage3_1.freeze()
            self.stage3_2.freeze()
            self.stage3_3.freeze()
        if freeze_at >= 4:
            self.stage4_0.freeze()
            self.stage4_1.freeze()
            self.stage4_2.freeze()
            self.stage4_3.freeze()
            self.stage4_4.freeze()
            self.stage4_5.freeze()
        if freeze_at >= 5:
            self.stage5_0.freeze()
            self.stage5_1.freeze()
            self.stage5_2.freeze()

    def add_param_group(self, param_groups, base_lr, base_wd):
        self.stage1_conv1_1.add_param_group(param_groups, base_lr, base_wd)
        self.stage1_conv1_2.add_param_group(param_groups, base_lr, base_wd)
        self.stage1_conv1_3.add_param_group(param_groups, base_lr, base_wd)
        self.stage2_0.add_param_group(param_groups, base_lr, base_wd)
        self.stage2_1.add_param_group(param_groups, base_lr, base_wd)
        self.stage2_2.add_param_group(param_groups, base_lr, base_wd)
        self.stage3_0.add_param_group(param_groups, base_lr, base_wd)
        self.stage3_1.add_param_group(param_groups, base_lr, base_wd)
        self.stage3_2.add_param_group(param_groups, base_lr, base_wd)
        self.stage3_3.add_param_group(param_groups, base_lr, base_wd)
        self.stage4_0.add_param_group(param_groups, base_lr, base_wd)
        self.stage4_1.add_param_group(param_groups, base_lr, base_wd)
        self.stage4_2.add_param_group(param_groups, base_lr, base_wd)
        self.stage4_3.add_param_group(param_groups, base_lr, base_wd)
        self.stage4_4.add_param_group(param_groups, base_lr, base_wd)
        self.stage4_5.add_param_group(param_groups, base_lr, base_wd)
        self.stage5_0.add_param_group(param_groups, base_lr, base_wd)
        self.stage5_1.add_param_group(param_groups, base_lr, base_wd)
        self.stage5_2.add_param_group(param_groups, base_lr, base_wd)


class BasicBlock(torch.nn.Module):

    def __init__(self, in_c, filters, bn, gn, af, freeze_norm, norm_decay, lr, stride=1, is_first=False, block_name=''):
        super(BasicBlock, self).__init__()
        filters1, filters2 = filters
        stride1, stride2 = stride, 1
        self.is_first = is_first
        self.stride = stride
        self.conv1 = Conv2dUnit(in_c, filters1, 3, stride=stride1, bn=bn, gn=gn, af=af, freeze_norm=freeze_norm, norm_decay=norm_decay, lr=lr, act='relu', name=block_name + '_branch2a')
        self.conv2 = Conv2dUnit(filters1, filters2, 3, stride=stride2, bn=bn, gn=gn, af=af, freeze_norm=freeze_norm, norm_decay=norm_decay, lr=lr, act=None, name=block_name + '_branch2b')
        self.conv3 = None
        if self.stride == 2 or self.is_first:
            if not self.is_first:
                self.avg_pool = torch.nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
                self.conv3 = Conv2dUnit(in_c, filters2, 1, stride=1, bn=bn, gn=gn, af=af, freeze_norm=freeze_norm, norm_decay=norm_decay, lr=lr, act=None, name=block_name + '_branch1')
            else:
                self.conv3 = Conv2dUnit(in_c, filters2, 1, stride=stride, bn=bn, gn=gn, freeze_norm=freeze_norm, norm_decay=norm_decay, lr=lr, af=af, act=None, name=block_name + '_branch1')
        self.act = torch.nn.ReLU(inplace=True)

    def freeze(self):
        self.conv1.freeze()
        self.conv2.freeze()
        if self.conv3 is not None:
            self.conv3.freeze()

    def add_param_group(self, param_groups, base_lr, base_wd):
        self.conv1.add_param_group(param_groups, base_lr, base_wd)
        self.conv2.add_param_group(param_groups, base_lr, base_wd)
        if self.conv3 is not None:
            self.conv3.add_param_group(param_groups, base_lr, base_wd)

    def forward(self, input_tensor):
        x = self.conv1(input_tensor)
        x = self.conv2(x)
        if self.stride == 2 or self.is_first:
            if not self.is_first:
                input_tensor = self.avg_pool(input_tensor)
            shortcut = self.conv3(input_tensor)
        else:
            shortcut = input_tensor
        x = x + shortcut
        x = self.act(x)
        return x


class Resnet18Vd(torch.nn.Module):

    def __init__(self, norm_type='bn', feature_maps=[4, 5], dcn_v2_stages=[], freeze_at=0, freeze_norm=False, norm_decay=0.0, lr_mult_list=[1.0, 1.0, 1.0, 1.0]):
        super(Resnet18Vd, self).__init__()
        self.norm_type = norm_type
        self.feature_maps = feature_maps
        assert freeze_at in [0, 1, 2, 3, 4, 5]
        assert len(lr_mult_list) == 4, 'lr_mult_list length must be 4 but got {}'.format(len(lr_mult_list))
        self.lr_mult_list = lr_mult_list
        self.freeze_at = freeze_at
        assert norm_type in ['bn', 'sync_bn', 'gn', 'affine_channel']
        bn, gn, af = get_norm(norm_type)
        self.stage1_conv1_1 = Conv2dUnit(3, 32, 3, stride=2, bn=bn, gn=gn, af=af, freeze_norm=freeze_norm, norm_decay=norm_decay, act='relu', name='conv1_1')
        self.stage1_conv1_2 = Conv2dUnit(32, 32, 3, stride=1, bn=bn, gn=gn, af=af, freeze_norm=freeze_norm, norm_decay=norm_decay, act='relu', name='conv1_2')
        self.stage1_conv1_3 = Conv2dUnit(32, 64, 3, stride=1, bn=bn, gn=gn, af=af, freeze_norm=freeze_norm, norm_decay=norm_decay, act='relu', name='conv1_3')
        self.pool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.stage2_0 = BasicBlock(64, [64, 64], bn, gn, af, freeze_norm, norm_decay, lr_mult_list[0], stride=1, is_first=True, block_name='res2a')
        self.stage2_1 = BasicBlock(64, [64, 64], bn, gn, af, freeze_norm, norm_decay, lr_mult_list[0], stride=1, block_name='res2b')
        self.stage3_0 = BasicBlock(64, [128, 128], bn, gn, af, freeze_norm, norm_decay, lr_mult_list[1], stride=2, block_name='res3a')
        self.stage3_1 = BasicBlock(128, [128, 128], bn, gn, af, freeze_norm, norm_decay, lr_mult_list[1], stride=1, block_name='res3b')
        self.stage4_0 = BasicBlock(128, [256, 256], bn, gn, af, freeze_norm, norm_decay, lr_mult_list[2], stride=2, block_name='res4a')
        self.stage4_1 = BasicBlock(256, [256, 256], bn, gn, af, freeze_norm, norm_decay, lr_mult_list[2], stride=1, block_name='res4b')
        self.stage5_0 = BasicBlock(256, [512, 512], bn, gn, af, freeze_norm, norm_decay, lr_mult_list[3], stride=2, block_name='res5a')
        self.stage5_1 = BasicBlock(512, [512, 512], bn, gn, af, freeze_norm, norm_decay, lr_mult_list[3], stride=1, block_name='res5b')

    def forward(self, input_tensor):
        x = self.stage1_conv1_1(input_tensor)
        x = self.stage1_conv1_2(x)
        x = self.stage1_conv1_3(x)
        x = self.pool(x)
        x = self.stage2_0(x)
        s4 = self.stage2_1(x)
        x = self.stage3_0(s4)
        s8 = self.stage3_1(x)
        x = self.stage4_0(s8)
        s16 = self.stage4_1(x)
        x = self.stage5_0(s16)
        s32 = self.stage5_1(x)
        outs = []
        if 2 in self.feature_maps:
            outs.append(s4)
        if 3 in self.feature_maps:
            outs.append(s8)
        if 4 in self.feature_maps:
            outs.append(s16)
        if 5 in self.feature_maps:
            outs.append(s32)
        return outs

    def get_block(self, name):
        layer = getattr(self, name)
        return layer

    def freeze(self):
        freeze_at = self.freeze_at
        if freeze_at >= 1:
            self.stage1_conv1_1.freeze()
            self.stage1_conv1_2.freeze()
            self.stage1_conv1_3.freeze()
        if freeze_at >= 2:
            self.stage2_0.freeze()
            self.stage2_1.freeze()
        if freeze_at >= 3:
            self.stage3_0.freeze()
            self.stage3_1.freeze()
        if freeze_at >= 4:
            self.stage4_0.freeze()
            self.stage4_1.freeze()
        if freeze_at >= 5:
            self.stage5_0.freeze()
            self.stage5_1.freeze()

    def add_param_group(self, param_groups, base_lr, base_wd):
        self.stage1_conv1_1.add_param_group(param_groups, base_lr, base_wd)
        self.stage1_conv1_2.add_param_group(param_groups, base_lr, base_wd)
        self.stage1_conv1_3.add_param_group(param_groups, base_lr, base_wd)
        self.stage2_0.add_param_group(param_groups, base_lr, base_wd)
        self.stage2_1.add_param_group(param_groups, base_lr, base_wd)
        self.stage3_0.add_param_group(param_groups, base_lr, base_wd)
        self.stage3_1.add_param_group(param_groups, base_lr, base_wd)
        self.stage4_0.add_param_group(param_groups, base_lr, base_wd)
        self.stage4_1.add_param_group(param_groups, base_lr, base_wd)
        self.stage5_0.add_param_group(param_groups, base_lr, base_wd)
        self.stage5_1.add_param_group(param_groups, base_lr, base_wd)


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (AffineChannel,
     lambda: ([], {'num_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Conv2dUnit,
     lambda: ([], {'input_dim': 4, 'filters': 4, 'filter_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (DCNv2,
     lambda: ([], {'input_dim': 4, 'filters': 4, 'filter_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (DCNv2_Slow,
     lambda: ([], {'input_dim': 4, 'filters': 4, 'filter_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Mish,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Resnet18Vd,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {})),
    (Resnet50Vd,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {})),
    (SPP,
     lambda: ([], {}),
     lambda: ([], {'x': torch.rand([4, 4, 4])})),
]

