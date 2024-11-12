
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


import torch.nn.functional as F


import math


import numpy as np


from torch.autograd import Function


from torch.autograd.function import once_differentiable


from torch.nn.modules.utils import _pair


from torch.nn.modules.utils import _single


from torch.utils.cpp_extension import BuildExtension


from torch.utils.cpp_extension import CUDAExtension


import torch.utils.data


from torch.autograd import Variable


import torch.utils.data as data


import torchvision.transforms as transforms


from torch.utils.data import DataLoader


from torch.nn.modules import padding


class DeformConvFunction(Function):

    @staticmethod
    def forward(ctx, input, offset, weight, stride=1, padding=0, dilation=1, groups=1, deformable_groups=1, im2col_step=64):
        if input is not None and input.dim() != 4:
            raise ValueError('Expected 4D tensor as input, got {}D tensor instead.'.format(input.dim()))
        ctx.stride = _pair(stride)
        ctx.padding = _pair(padding)
        ctx.dilation = _pair(dilation)
        ctx.groups = groups
        ctx.deformable_groups = deformable_groups
        ctx.im2col_step = im2col_step
        ctx.save_for_backward(input, offset, weight)
        output = input.new_empty(DeformConvFunction._output_size(input, weight, ctx.padding, ctx.dilation, ctx.stride))
        ctx.bufs_ = [input.new_empty(0), input.new_empty(0)]
        if not input.is_cuda:
            raise NotImplementedError
        else:
            cur_im2col_step = min(ctx.im2col_step, input.shape[0])
            assert input.shape[0] % cur_im2col_step == 0, 'im2col step must divide batchsize'
            deform_conv_cuda.deform_conv_forward_cuda(input, weight, offset, output, ctx.bufs_[0], ctx.bufs_[1], weight.size(3), weight.size(2), ctx.stride[1], ctx.stride[0], ctx.padding[1], ctx.padding[0], ctx.dilation[1], ctx.dilation[0], ctx.groups, ctx.deformable_groups, cur_im2col_step)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        input, offset, weight = ctx.saved_tensors
        grad_input = grad_offset = grad_weight = None
        if not grad_output.is_cuda:
            raise NotImplementedError
        else:
            cur_im2col_step = min(ctx.im2col_step, input.shape[0])
            assert input.shape[0] % cur_im2col_step == 0, 'im2col step must divide batchsize'
            if ctx.needs_input_grad[0] or ctx.needs_input_grad[1]:
                grad_input = torch.zeros_like(input)
                grad_offset = torch.zeros_like(offset)
                deform_conv_cuda.deform_conv_backward_input_cuda(input, offset, grad_output, grad_input, grad_offset, weight, ctx.bufs_[0], weight.size(3), weight.size(2), ctx.stride[1], ctx.stride[0], ctx.padding[1], ctx.padding[0], ctx.dilation[1], ctx.dilation[0], ctx.groups, ctx.deformable_groups, cur_im2col_step)
            if ctx.needs_input_grad[2]:
                grad_weight = torch.zeros_like(weight)
                deform_conv_cuda.deform_conv_backward_parameters_cuda(input, offset, grad_output, grad_weight, ctx.bufs_[0], ctx.bufs_[1], weight.size(3), weight.size(2), ctx.stride[1], ctx.stride[0], ctx.padding[1], ctx.padding[0], ctx.dilation[1], ctx.dilation[0], ctx.groups, ctx.deformable_groups, 1, cur_im2col_step)
        return grad_input, grad_offset, grad_weight, None, None, None, None, None

    @staticmethod
    def _output_size(input, weight, padding, dilation, stride):
        channels = weight.size(0)
        output_size = input.size(0), channels
        for d in range(input.dim() - 2):
            in_size = input.size(d + 2)
            pad = padding[d]
            kernel = dilation[d] * (weight.size(d + 2) - 1) + 1
            stride_ = stride[d]
            output_size += (in_size + 2 * pad - kernel) // stride_ + 1,
        if not all(map(lambda s: s > 0, output_size)):
            raise ValueError('convolution input is too small (output would be {})'.format('x'.join(map(str, output_size))))
        return output_size


deform_conv = DeformConvFunction.apply


class DeformConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, deformable_groups=1, bias=False):
        super(DeformConv, self).__init__()
        assert not bias
        assert in_channels % groups == 0, 'in_channels {} cannot be divisible by groups {}'.format(in_channels, groups)
        assert out_channels % groups == 0, 'out_channels {} cannot be divisible by groups {}'.format(out_channels, groups)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.deformable_groups = deformable_groups
        self.transposed = False
        self.output_padding = _single(0)
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // self.groups, *self.kernel_size))
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1.0 / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, x, offset):
        return deform_conv(x, offset, self.weight, self.stride, self.padding, self.dilation, self.groups, self.deformable_groups)


class ModulatedDeformConvFunction(Function):

    @staticmethod
    def forward(ctx, input, offset, mask, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, deformable_groups=1):
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.groups = groups
        ctx.deformable_groups = deformable_groups
        ctx.with_bias = bias is not None
        if not ctx.with_bias:
            bias = input.new_empty(1)
        if not input.is_cuda:
            raise NotImplementedError
        if weight.requires_grad or mask.requires_grad or offset.requires_grad or input.requires_grad:
            ctx.save_for_backward(input, offset, mask, weight, bias)
        output = input.new_empty(ModulatedDeformConvFunction._infer_shape(ctx, input, weight))
        ctx._bufs = [input.new_empty(0), input.new_empty(0)]
        deform_conv_cuda.modulated_deform_conv_cuda_forward(input, weight, bias, ctx._bufs[0], offset, mask, output, ctx._bufs[1], weight.shape[2], weight.shape[3], ctx.stride, ctx.stride, ctx.padding, ctx.padding, ctx.dilation, ctx.dilation, ctx.groups, ctx.deformable_groups, ctx.with_bias)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        if not grad_output.is_cuda:
            raise NotImplementedError
        input, offset, mask, weight, bias = ctx.saved_tensors
        grad_input = torch.zeros_like(input)
        grad_offset = torch.zeros_like(offset)
        grad_mask = torch.zeros_like(mask)
        grad_weight = torch.zeros_like(weight)
        grad_bias = torch.zeros_like(bias)
        deform_conv_cuda.modulated_deform_conv_cuda_backward(input, weight, bias, ctx._bufs[0], offset, mask, ctx._bufs[1], grad_input, grad_weight, grad_bias, grad_offset, grad_mask, grad_output, weight.shape[2], weight.shape[3], ctx.stride, ctx.stride, ctx.padding, ctx.padding, ctx.dilation, ctx.dilation, ctx.groups, ctx.deformable_groups, ctx.with_bias)
        if not ctx.with_bias:
            grad_bias = None
        return grad_input, grad_offset, grad_mask, grad_weight, grad_bias, None, None, None, None, None

    @staticmethod
    def _infer_shape(ctx, input, weight):
        n = input.size(0)
        channels_out = weight.size(0)
        height, width = input.shape[2:4]
        kernel_h, kernel_w = weight.shape[2:4]
        height_out = (height + 2 * ctx.padding - (ctx.dilation * (kernel_h - 1) + 1)) // ctx.stride + 1
        width_out = (width + 2 * ctx.padding - (ctx.dilation * (kernel_w - 1) + 1)) // ctx.stride + 1
        return n, channels_out, height_out, width_out


modulated_deform_conv = ModulatedDeformConvFunction.apply


class ModulatedDeformConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, deformable_groups=1, bias=True):
        super(ModulatedDeformConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.deformable_groups = deformable_groups
        self.with_bias = bias
        self.transposed = False
        self.output_padding = _single(0)
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, *self.kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1.0 / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.zero_()

    def forward(self, x, offset, mask):
        return modulated_deform_conv(x, offset, mask, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups, self.deformable_groups)


class DeformConv2d(nn.Module):
    """A single (modulated) deformable conv layer"""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=2, groups=1, deformable_groups=2, modulation=True, double_mask=True, bias=False):
        super(DeformConv2d, self).__init__()
        self.modulation = modulation
        self.deformable_groups = deformable_groups
        self.kernel_size = kernel_size
        self.double_mask = double_mask
        if self.modulation:
            self.deform_conv = ModulatedDeformConv(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=dilation, dilation=dilation, groups=groups, deformable_groups=deformable_groups, bias=bias)
        else:
            self.deform_conv = DeformConv(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=dilation, dilation=dilation, groups=groups, deformable_groups=deformable_groups, bias=bias)
        k = 3 if self.modulation else 2
        offset_out_channels = deformable_groups * k * kernel_size * kernel_size
        self.offset_conv = nn.Conv2d(in_channels, offset_out_channels, kernel_size=kernel_size, stride=stride, padding=dilation, dilation=dilation, groups=deformable_groups, bias=True)
        nn.init.constant_(self.offset_conv.weight, 0.0)
        nn.init.constant_(self.offset_conv.bias, 0.0)

    def forward(self, x):
        if self.modulation:
            offset_mask = self.offset_conv(x)
            offset_channel = self.deformable_groups * 2 * self.kernel_size * self.kernel_size
            offset = offset_mask[:, :offset_channel, :, :]
            mask = offset_mask[:, offset_channel:, :, :]
            mask = mask.sigmoid()
            if self.double_mask:
                mask = mask * 2
            out = self.deform_conv(x, offset, mask)
        else:
            offset = self.offset_conv(x)
            out = self.deform_conv(x, offset)
        return out


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class DeformSimpleBottleneck(nn.Module):
    """Used for cost aggregation"""

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, norm_layer=None, mdconv_dilation=2, deformable_groups=2, modulation=True, double_mask=True):
        super(DeformSimpleBottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = DeformConv2d(width, width, stride=stride, dilation=mdconv_dilation, deformable_groups=deformable_groups, modulation=modulation, double_mask=double_mask)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes)
        self.bn3 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


def conv3x3(in_planes, out_planes, strd=1, padding=1, bias=False):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=strd, padding=padding, bias=bias)


class SimpleBottleneck(nn.Module):
    """Simple bottleneck block without channel expansion"""

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1, norm_layer=None):
        super(SimpleBottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes)
        self.bn3 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class AdaptiveAggregationModule(nn.Module):

    def __init__(self, num_scales, num_output_branches, max_disp, num_blocks=1, simple_bottleneck=False, deformable_groups=2, mdconv_dilation=2):
        super(AdaptiveAggregationModule, self).__init__()
        self.num_scales = num_scales
        self.num_output_branches = num_output_branches
        self.max_disp = max_disp
        self.num_blocks = num_blocks
        self.branches = nn.ModuleList()
        for i in range(self.num_scales):
            num_candidates = max_disp // 2 ** i
            branch = nn.ModuleList()
            for j in range(num_blocks):
                if simple_bottleneck:
                    branch.append(SimpleBottleneck(num_candidates, num_candidates))
                else:
                    branch.append(DeformSimpleBottleneck(num_candidates, num_candidates, modulation=True, mdconv_dilation=mdconv_dilation, deformable_groups=deformable_groups))
            self.branches.append(nn.Sequential(*branch))
        self.fuse_layers = nn.ModuleList()
        for i in range(self.num_output_branches):
            self.fuse_layers.append(nn.ModuleList())
            for j in range(self.num_scales):
                if i == j:
                    self.fuse_layers[-1].append(nn.Identity())
                elif i < j:
                    self.fuse_layers[-1].append(nn.Sequential(nn.Conv2d(max_disp // 2 ** j, max_disp // 2 ** i, kernel_size=1, bias=False), nn.BatchNorm2d(max_disp // 2 ** i)))
                elif i > j:
                    layers = nn.ModuleList()
                    for k in range(i - j - 1):
                        layers.append(nn.Sequential(nn.Conv2d(max_disp // 2 ** j, max_disp // 2 ** j, kernel_size=3, stride=2, padding=1, bias=False), nn.BatchNorm2d(max_disp // 2 ** j), nn.LeakyReLU(0.2, inplace=True)))
                    layers.append(nn.Sequential(nn.Conv2d(max_disp // 2 ** j, max_disp // 2 ** i, kernel_size=3, stride=2, padding=1, bias=False), nn.BatchNorm2d(max_disp // 2 ** i)))
                    self.fuse_layers[-1].append(nn.Sequential(*layers))
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        assert len(self.branches) == len(x)
        for i in range(len(self.branches)):
            branch = self.branches[i]
            for j in range(self.num_blocks):
                dconv = branch[j]
                x[i] = dconv(x[i])
        if self.num_scales == 1:
            return x
        x_fused = []
        for i in range(len(self.fuse_layers)):
            for j in range(len(self.branches)):
                if j == 0:
                    x_fused.append(self.fuse_layers[i][0](x[0]))
                else:
                    exchange = self.fuse_layers[i][j](x[j])
                    if exchange.size()[2:] != x_fused[i].size()[2:]:
                        exchange = F.interpolate(exchange, size=x_fused[i].size()[2:], mode='bilinear')
                    x_fused[i] = x_fused[i] + exchange
        for i in range(len(x_fused)):
            x_fused[i] = self.relu(x_fused[i])
        return x_fused


class AdaptiveAggregation(nn.Module):

    def __init__(self, max_disp, num_scales=3, num_fusions=6, num_stage_blocks=1, num_deform_blocks=2, intermediate_supervision=True, deformable_groups=2, mdconv_dilation=2):
        super(AdaptiveAggregation, self).__init__()
        self.max_disp = max_disp
        self.num_scales = num_scales
        self.num_fusions = num_fusions
        self.intermediate_supervision = intermediate_supervision
        fusions = nn.ModuleList()
        for i in range(num_fusions):
            if self.intermediate_supervision:
                num_out_branches = self.num_scales
            else:
                num_out_branches = 1 if i == num_fusions - 1 else self.num_scales
            if i >= num_fusions - num_deform_blocks:
                simple_bottleneck_module = False
            else:
                simple_bottleneck_module = True
            fusions.append(AdaptiveAggregationModule(num_scales=self.num_scales, num_output_branches=num_out_branches, max_disp=max_disp, num_blocks=num_stage_blocks, mdconv_dilation=mdconv_dilation, deformable_groups=deformable_groups, simple_bottleneck=simple_bottleneck_module))
        self.fusions = nn.Sequential(*fusions)
        self.final_conv = nn.ModuleList()
        for i in range(self.num_scales):
            in_channels = max_disp // 2 ** i
            self.final_conv.append(nn.Conv2d(in_channels, max_disp // 2 ** i, kernel_size=1))
            if not self.intermediate_supervision:
                break

    def forward(self, cost_volume):
        assert isinstance(cost_volume, list)
        for i in range(self.num_fusions):
            fusion = self.fusions[i]
            cost_volume = fusion(cost_volume)
        out = []
        for i in range(len(self.final_conv)):
            out = out + [self.final_conv[i](cost_volume[i])]
        return out


class CostVolume(nn.Module):

    def __init__(self, max_disp, feature_similarity='correlation'):
        """Construct cost volume based on different
        similarity measures

        Args:
            max_disp: max disparity candidate
            feature_similarity: type of similarity measure
        """
        super(CostVolume, self).__init__()
        self.max_disp = max_disp
        self.feature_similarity = feature_similarity

    def forward(self, left_feature, right_feature):
        b, c, h, w = left_feature.size()
        if self.feature_similarity == 'difference':
            cost_volume = left_feature.new_zeros(b, c, self.max_disp, h, w)
            for i in range(self.max_disp):
                if i > 0:
                    cost_volume[:, :, i, :, i:] = left_feature[:, :, :, i:] - right_feature[:, :, :, :-i]
                else:
                    cost_volume[:, :, i, :, :] = left_feature - right_feature
        elif self.feature_similarity == 'concat':
            cost_volume = left_feature.new_zeros(b, 2 * c, self.max_disp, h, w)
            for i in range(self.max_disp):
                if i > 0:
                    cost_volume[:, :, i, :, i:] = torch.cat((left_feature[:, :, :, i:], right_feature[:, :, :, :-i]), dim=1)
                else:
                    cost_volume[:, :, i, :, :] = torch.cat((left_feature, right_feature), dim=1)
        elif self.feature_similarity == 'correlation':
            cost_volume = left_feature.new_zeros(b, self.max_disp, h, w)
            for i in range(self.max_disp):
                if i > 0:
                    cost_volume[:, i, :, i:] = (left_feature[:, :, :, i:] * right_feature[:, :, :, :-i]).mean(dim=1)
                else:
                    cost_volume[:, i, :, :] = (left_feature * right_feature).mean(dim=1)
        else:
            raise NotImplementedError
        cost_volume = cost_volume.contiguous()
        return cost_volume


class CostVolumePyramid(nn.Module):

    def __init__(self, max_disp, feature_similarity='correlation'):
        super(CostVolumePyramid, self).__init__()
        self.max_disp = max_disp
        self.feature_similarity = feature_similarity

    def forward(self, left_feature_pyramid, right_feature_pyramid):
        num_scales = len(left_feature_pyramid)
        cost_volume_pyramid = []
        for s in range(num_scales):
            max_disp = self.max_disp // 2 ** s
            cost_volume_module = CostVolume(max_disp, self.feature_similarity)
            cost_volume = cost_volume_module(left_feature_pyramid[s], right_feature_pyramid[s])
            cost_volume_pyramid.append(cost_volume)
        return cost_volume_pyramid


class DisparityEstimation(nn.Module):

    def __init__(self, max_disp, match_similarity=True):
        super(DisparityEstimation, self).__init__()
        self.max_disp = max_disp
        self.match_similarity = match_similarity

    def forward(self, cost_volume):
        assert cost_volume.dim() == 4
        cost_volume = cost_volume if self.match_similarity else -cost_volume
        prob_volume = F.softmax(cost_volume, dim=1)
        if cost_volume.size(1) == self.max_disp:
            disp_candidates = torch.arange(0, self.max_disp).type_as(prob_volume)
        else:
            max_disp = prob_volume.size(1)
            disp_candidates = torch.arange(0, max_disp).type_as(prob_volume)
        disp_candidates = disp_candidates.view(1, cost_volume.size(1), 1, 1)
        disp = torch.sum(prob_volume * disp_candidates, 1, keepdim=False)
        return disp


class FeaturePyrmaid(nn.Module):

    def __init__(self, in_channel=32):
        super(FeaturePyrmaid, self).__init__()
        self.out1 = nn.Sequential(nn.Conv2d(in_channel, in_channel * 2, kernel_size=3, stride=2, padding=1, bias=False), nn.BatchNorm2d(in_channel * 2), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(in_channel * 2, in_channel * 2, kernel_size=1, stride=1, padding=0, bias=False), nn.BatchNorm2d(in_channel * 2), nn.LeakyReLU(0.2, inplace=True))
        self.out2 = nn.Sequential(nn.Conv2d(in_channel * 2, in_channel * 4, kernel_size=3, stride=2, padding=1, bias=False), nn.BatchNorm2d(in_channel * 4), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(in_channel * 4, in_channel * 4, kernel_size=1, stride=1, padding=0, bias=False), nn.BatchNorm2d(in_channel * 4), nn.LeakyReLU(0.2, inplace=True))

    def forward(self, x):
        out1 = self.out1(x)
        out2 = self.out2(out1)
        return [x, out1, out2]


class BasicConv(nn.Module):

    def __init__(self, in_channels, out_channels, deconv=False, is_3d=False, bn=True, relu=True, **kwargs):
        super(BasicConv, self).__init__()
        self.relu = relu
        self.use_bn = bn
        if is_3d:
            if deconv:
                self.conv = nn.ConvTranspose3d(in_channels, out_channels, bias=False, **kwargs)
            else:
                self.conv = nn.Conv3d(in_channels, out_channels, bias=False, **kwargs)
            self.bn = nn.BatchNorm3d(out_channels)
        else:
            if deconv:
                self.conv = nn.ConvTranspose2d(in_channels, out_channels, bias=False, **kwargs)
            else:
                self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
            self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x


class Conv2x(nn.Module):

    def __init__(self, in_channels, out_channels, deconv=False, is_3d=False, concat=True, bn=True, relu=True, mdconv=False):
        super(Conv2x, self).__init__()
        self.concat = concat
        if deconv and is_3d:
            kernel = 3, 4, 4
        elif deconv:
            kernel = 4
        else:
            kernel = 3
        self.conv1 = BasicConv(in_channels, out_channels, deconv, is_3d, bn=True, relu=True, kernel_size=kernel, stride=2, padding=1)
        if self.concat:
            if mdconv:
                self.conv2 = DeformConv2d(out_channels * 2, out_channels, kernel_size=3, stride=1)
            else:
                self.conv2 = BasicConv(out_channels * 2, out_channels, False, is_3d, bn, relu, kernel_size=3, stride=1, padding=1)
        else:
            self.conv2 = BasicConv(out_channels, out_channels, False, is_3d, bn, relu, kernel_size=3, stride=1, padding=1)

    def forward(self, x, rem):
        x = self.conv1(x)
        assert x.size() == rem.size()
        if self.concat:
            x = torch.cat((x, rem), 1)
        else:
            x = x + rem
        x = self.conv2(x)
        return x


class GANetFeature(nn.Module):
    """Height and width need to be divided by 48, downsampled by 1/3"""

    def __init__(self, feature_mdconv=False):
        super(GANetFeature, self).__init__()
        if feature_mdconv:
            self.conv_start = nn.Sequential(BasicConv(3, 32, kernel_size=3, padding=1), BasicConv(32, 32, kernel_size=5, stride=3, padding=2), DeformConv2d(32, 32))
        else:
            self.conv_start = nn.Sequential(BasicConv(3, 32, kernel_size=3, padding=1), BasicConv(32, 32, kernel_size=5, stride=3, padding=2), BasicConv(32, 32, kernel_size=3, padding=1))
        self.conv1a = BasicConv(32, 48, kernel_size=3, stride=2, padding=1)
        self.conv2a = BasicConv(48, 64, kernel_size=3, stride=2, padding=1)
        if feature_mdconv:
            self.conv3a = DeformConv2d(64, 96, kernel_size=3, stride=2)
            self.conv4a = DeformConv2d(96, 128, kernel_size=3, stride=2)
        else:
            self.conv3a = BasicConv(64, 96, kernel_size=3, stride=2, padding=1)
            self.conv4a = BasicConv(96, 128, kernel_size=3, stride=2, padding=1)
        self.deconv4a = Conv2x(128, 96, deconv=True)
        self.deconv3a = Conv2x(96, 64, deconv=True)
        self.deconv2a = Conv2x(64, 48, deconv=True)
        self.deconv1a = Conv2x(48, 32, deconv=True)
        self.conv1b = Conv2x(32, 48)
        self.conv2b = Conv2x(48, 64)
        if feature_mdconv:
            self.conv3b = Conv2x(64, 96, mdconv=True)
            self.conv4b = Conv2x(96, 128, mdconv=True)
        else:
            self.conv3b = Conv2x(64, 96)
            self.conv4b = Conv2x(96, 128)
        self.deconv4b = Conv2x(128, 96, deconv=True)
        self.deconv3b = Conv2x(96, 64, deconv=True)
        self.deconv2b = Conv2x(64, 48, deconv=True)
        self.deconv1b = Conv2x(48, 32, deconv=True)

    def forward(self, x):
        x = self.conv_start(x)
        rem0 = x
        x = self.conv1a(x)
        rem1 = x
        x = self.conv2a(x)
        rem2 = x
        x = self.conv3a(x)
        rem3 = x
        x = self.conv4a(x)
        rem4 = x
        x = self.deconv4a(x, rem3)
        rem3 = x
        x = self.deconv3a(x, rem2)
        rem2 = x
        x = self.deconv2a(x, rem1)
        rem1 = x
        x = self.deconv1a(x, rem0)
        rem0 = x
        x = self.conv1b(x, rem1)
        rem1 = x
        x = self.conv2b(x, rem2)
        rem2 = x
        x = self.conv3b(x, rem3)
        rem3 = x
        x = self.conv4b(x, rem4)
        x = self.deconv4b(x, rem3)
        x = self.deconv3b(x, rem2)
        x = self.deconv2b(x, rem1)
        x = self.deconv1b(x, rem0)
        return x


def conv2d(in_channels, out_channels, kernel_size=3, stride=1, dilation=1, groups=1):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=dilation, dilation=dilation, bias=False, groups=groups), nn.BatchNorm2d(out_channels), nn.LeakyReLU(0.2, inplace=True))


def meshgrid(img, homogeneous=False):
    """Generate meshgrid in image scale
    Args:
        img: [B, _, H, W]
        homogeneous: whether to return homogeneous coordinates
    Return:
        grid: [B, 2, H, W]
    """
    b, _, h, w = img.size()
    x_range = torch.arange(0, w).view(1, 1, w).expand(1, h, w).type_as(img)
    y_range = torch.arange(0, h).view(1, h, 1).expand(1, h, w).type_as(img)
    grid = torch.cat((x_range, y_range), dim=0)
    grid = grid.unsqueeze(0).expand(b, 2, h, w)
    if homogeneous:
        ones = torch.ones_like(x_range).unsqueeze(0).expand(b, 1, h, w)
        grid = torch.cat((grid, ones), dim=1)
        assert grid.size(1) == 3
    return grid


def normalize_coords(grid):
    """Normalize coordinates of image scale to [-1, 1]
    Args:
        grid: [B, 2, H, W]
    """
    assert grid.size(1) == 2
    h, w = grid.size()[2:]
    grid[:, 0, :, :] = 2 * (grid[:, 0, :, :].clone() / (w - 1)) - 1
    grid[:, 1, :, :] = 2 * (grid[:, 1, :, :].clone() / (h - 1)) - 1
    grid = grid.permute((0, 2, 3, 1))
    return grid


def disp_warp(img, disp, padding_mode='border'):
    """Warping by disparity
    Args:
        img: [B, 3, H, W]
        disp: [B, 1, H, W], positive
        padding_mode: 'zeros' or 'border'
    Returns:
        warped_img: [B, 3, H, W]
        valid_mask: [B, 3, H, W]
    """
    assert disp.min() >= 0
    grid = meshgrid(img)
    offset = torch.cat((-disp, torch.zeros_like(disp)), dim=1)
    sample_grid = grid + offset
    corres_L2R = sample_grid.clone()
    sample_grid = normalize_coords(sample_grid)
    warped_img = F.grid_sample(img, sample_grid, mode='bilinear', padding_mode=padding_mode, align_corners=True)
    return warped_img, corres_L2R.permute(0, 2, 3, 1)


class HourglassRefinement(nn.Module):
    """Height and width need to be divided by 16"""

    def __init__(self):
        super(HourglassRefinement, self).__init__()
        in_channels = 6
        self.conv1 = conv2d(in_channels, 16)
        self.conv2 = conv2d(1, 16)
        self.conv_start = DeformConv2d(32, 32)
        self.conv1a = BasicConv(32, 48, kernel_size=3, stride=2, padding=1)
        self.conv2a = BasicConv(48, 64, kernel_size=3, stride=2, padding=1)
        self.conv3a = DeformConv2d(64, 96, kernel_size=3, stride=2)
        self.conv4a = DeformConv2d(96, 128, kernel_size=3, stride=2)
        self.deconv4a = Conv2x(128, 96, deconv=True)
        self.deconv3a = Conv2x(96, 64, deconv=True)
        self.deconv2a = Conv2x(64, 48, deconv=True)
        self.deconv1a = Conv2x(48, 32, deconv=True)
        self.conv1b = Conv2x(32, 48)
        self.conv2b = Conv2x(48, 64)
        self.conv3b = Conv2x(64, 96, mdconv=True)
        self.conv4b = Conv2x(96, 128, mdconv=True)
        self.deconv4b = Conv2x(128, 96, deconv=True)
        self.deconv3b = Conv2x(96, 64, deconv=True)
        self.deconv2b = Conv2x(64, 48, deconv=True)
        self.deconv1b = Conv2x(48, 32, deconv=True)
        self.final_conv = nn.Conv2d(32, 1, 3, 1, 1)

    def forward(self, low_disp, left_img, right_img):
        assert low_disp.dim() == 3
        low_disp = low_disp.unsqueeze(1)
        scale_factor = left_img.size(-1) / low_disp.size(-1)
        if scale_factor == 1.0:
            disp = low_disp
        else:
            disp = F.interpolate(low_disp, size=left_img.size()[-2:], mode='bilinear')
            disp = disp * scale_factor
        warped_right = disp_warp(right_img, disp)[0]
        error = warped_right - left_img
        concat1 = torch.cat((error, left_img), dim=1)
        conv1 = self.conv1(concat1)
        conv2 = self.conv2(disp)
        x = torch.cat((conv1, conv2), dim=1)
        x = self.conv_start(x)
        rem0 = x
        x = self.conv1a(x)
        rem1 = x
        x = self.conv2a(x)
        rem2 = x
        x = self.conv3a(x)
        rem3 = x
        x = self.conv4a(x)
        rem4 = x
        x = self.deconv4a(x, rem3)
        rem3 = x
        x = self.deconv3a(x, rem2)
        rem2 = x
        x = self.deconv2a(x, rem1)
        rem1 = x
        x = self.deconv1a(x, rem0)
        rem0 = x
        x = self.conv1b(x, rem1)
        rem1 = x
        x = self.conv2b(x, rem2)
        rem2 = x
        x = self.conv3b(x, rem3)
        rem3 = x
        x = self.conv4b(x, rem4)
        x = self.deconv4b(x, rem3)
        x = self.deconv3b(x, rem2)
        x = self.deconv2b(x, rem1)
        x = self.deconv1b(x, rem0)
        residual_disp = self.final_conv(x)
        disp = F.relu(disp + residual_disp, inplace=True)
        disp = disp.squeeze(1)
        return disp


class AANetPlusFeature(nn.Module):

    def __init__(self, input_reso, max_disp=72) ->None:
        super().__init__()
        num_scales = 3
        num_fusions = 6
        num_stage_blocks = 1
        num_deform_blocks = 3
        mdconv_dilation = 2
        deformable_groups = 2
        no_intermediate_supervision = False
        no_feature_mdconv = False
        feature_similarity = 'correlation'
        self.refinement_type = 'hourglass'
        self.feature_type = 'ganet'
        self.num_downsample = 2
        self.aggregation_type = 'adaptive'
        self.num_scales = num_scales
        self.max_disp = max_disp // 3
        self.feature_extractor = GANetFeature(feature_mdconv=not no_feature_mdconv)
        self.feature_reso_list = [math.ceil(input_reso / 3), math.ceil(input_reso / 6), math.ceil(input_reso / 12)]
        self.fpn = FeaturePyrmaid()
        self.cost_volume = CostVolumePyramid(self.max_disp, feature_similarity=feature_similarity)
        self.cost_volume3D = CostVolume(self.max_disp, feature_similarity='difference')
        self.aggregation = AdaptiveAggregation(max_disp=self.max_disp, num_scales=num_scales, num_fusions=num_fusions, num_stage_blocks=num_stage_blocks, num_deform_blocks=num_deform_blocks, mdconv_dilation=mdconv_dilation, deformable_groups=deformable_groups, intermediate_supervision=not no_intermediate_supervision)
        self.disparity_estimation = DisparityEstimation(self.max_disp, match_similarity=True)
        refine_module_list = nn.ModuleList()
        for _ in range(self.num_downsample):
            refine_module_list.append(HourglassRefinement())
        self.refinement = refine_module_list

    def feature_extraction(self, img):
        feature = self.feature_extractor(img)
        feature = self.fpn(feature)
        return feature

    def cost_volume_construction(self, left_feature, right_feature):
        cost_volume = self.cost_volume(left_feature, right_feature)
        return cost_volume

    def Construct3DFeatureVolume(self, left_feature_list, right_feature_list):
        target_reso = self.feature_reso_list[0]
        num = len(left_feature_list)
        left_res = []
        right_res = []
        for i in range(num):
            cur_left_feature = left_feature_list[i]
            if cur_left_feature.size(-1) != target_reso:
                left_res.append(F.interpolate(cur_left_feature, size=(target_reso, target_reso), mode='bilinear'))
            else:
                left_res.append(cur_left_feature)
            cur_right_feature = right_feature_list[i]
            if cur_right_feature.size(-1) != target_reso:
                right_res.append(F.interpolate(cur_right_feature, size=(target_reso, target_reso), mode='bilinear'))
            else:
                right_res.append(cur_right_feature)
        left_feature = torch.cat(left_res, dim=1)
        right_feature = torch.cat(right_res, dim=1)
        cost_volume3D = self.cost_volume3D(left_feature, right_feature)
        return left_feature, right_feature, cost_volume3D

    def ConstructConfindenceVolume(self, aggregation_list):
        target_reso = self.feature_reso_list[0]
        res = []
        for aggregation in aggregation_list:
            confidence_volume = F.softmax(aggregation, dim=1)
            confidence_volume = confidence_volume.unsqueeze(1)
            if confidence_volume.size(-1) != target_reso:
                res.append(F.interpolate(confidence_volume, size=(self.max_disp, target_reso, target_reso), mode='trilinear'))
            else:
                res.append(confidence_volume)
        temp_confidence_volume = torch.cat(res, dim=1).mean(dim=1, keepdim=True)
        return temp_confidence_volume

    def disparity_computation(self, aggregation):
        assert isinstance(aggregation, list)
        disparity_pyramid = []
        length = len(aggregation)
        for i in range(length):
            disp = self.disparity_estimation(aggregation[length - 1 - i])
            disparity_pyramid.append(disp)
        return disparity_pyramid

    def disparity_refinement(self, left_img, right_img, disparity):
        disparity_pyramid = []
        for i in range(self.num_downsample):
            scale_factor = 1.0 / pow(2, self.num_downsample - i - 1)
            if scale_factor == 1.0:
                curr_left_img = left_img
                curr_right_img = right_img
            else:
                curr_left_img = F.interpolate(left_img, scale_factor=scale_factor, mode='bilinear')
                curr_right_img = F.interpolate(right_img, scale_factor=scale_factor, mode='bilinear')
            inputs = disparity, curr_left_img, curr_right_img
            disparity = self.refinement[i](*inputs)
            disparity_pyramid.append(disparity)
        return disparity_pyramid

    def forward(self, left_img, right_img):
        left_feature_list = self.feature_extraction(left_img)
        right_feature_list = self.feature_extraction(right_img)
        left_feature, right_feature, cost_volume3D = self.Construct3DFeatureVolume(left_feature_list, right_feature_list)
        cost_volume = self.cost_volume_construction(left_feature_list, right_feature_list)
        aggregation = self.aggregation(cost_volume)
        confidence_volume = self.ConstructConfindenceVolume(aggregation)
        disparity_pyramid = self.disparity_computation(aggregation)
        disparity_pyramid += self.disparity_refinement(left_img, right_img, disparity_pyramid[-1])
        return left_feature, right_feature, cost_volume3D, confidence_volume, disparity_pyramid[-1]


class DeformBottleneck(nn.Module):
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1, norm_layer=None):
        super(DeformBottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = DeformConv2d(width, width, stride=stride)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class DeformConvPack(DeformConv):
    """A Deformable Conv Encapsulation that acts as normal Conv layers.

    Args:
        in_channels (int): Same as nn.Conv2d.
        out_channels (int): Same as nn.Conv2d.
        kernel_size (int or tuple[int]): Same as nn.Conv2d.
        stride (int or tuple[int]): Same as nn.Conv2d.
        padding (int or tuple[int]): Same as nn.Conv2d.
        dilation (int or tuple[int]): Same as nn.Conv2d.
        groups (int): Same as nn.Conv2d.
        bias (bool or str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias will be set as True if norm_cfg is None, otherwise
            False.
    """
    _version = 2

    def __init__(self, *args, **kwargs):
        super(DeformConvPack, self).__init__(*args, **kwargs)
        self.conv_offset = nn.Conv2d(self.in_channels, self.deformable_groups * 2 * self.kernel_size[0] * self.kernel_size[1], kernel_size=self.kernel_size, stride=_pair(self.stride), padding=_pair(self.padding), bias=True)
        self.init_offset()

    def init_offset(self):
        self.conv_offset.weight.data.zero_()
        self.conv_offset.bias.data.zero_()

    def forward(self, x):
        offset = self.conv_offset(x)
        return deform_conv(x, offset, self.weight, self.stride, self.padding, self.dilation, self.groups, self.deformable_groups)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        version = local_metadata.get('version', None)
        if version is None or version < 2:
            if prefix + 'conv_offset.weight' not in state_dict and prefix[:-1] + '_offset.weight' in state_dict:
                state_dict[prefix + 'conv_offset.weight'] = state_dict.pop(prefix[:-1] + '_offset.weight')
            if prefix + 'conv_offset.bias' not in state_dict and prefix[:-1] + '_offset.bias' in state_dict:
                state_dict[prefix + 'conv_offset.bias'] = state_dict.pop(prefix[:-1] + '_offset.bias')
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)


class ModulatedDeformConvPack(ModulatedDeformConv):
    """A ModulatedDeformable Conv Encapsulation that acts as normal Conv layers.

    Args:
        in_channels (int): Same as nn.Conv2d.
        out_channels (int): Same as nn.Conv2d.
        kernel_size (int or tuple[int]): Same as nn.Conv2d.
        stride (int or tuple[int]): Same as nn.Conv2d.
        padding (int or tuple[int]): Same as nn.Conv2d.
        dilation (int or tuple[int]): Same as nn.Conv2d.
        groups (int): Same as nn.Conv2d.
        bias (bool or str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias will be set as True if norm_cfg is None, otherwise
            False.
    """
    _version = 2

    def __init__(self, *args, **kwargs):
        super(ModulatedDeformConvPack, self).__init__(*args, **kwargs)
        self.conv_offset = nn.Conv2d(self.in_channels, self.deformable_groups * 3 * self.kernel_size[0] * self.kernel_size[1], kernel_size=self.kernel_size, stride=_pair(self.stride), padding=_pair(self.padding), bias=True)
        self.init_offset()

    def init_offset(self):
        self.conv_offset.weight.data.zero_()
        self.conv_offset.bias.data.zero_()

    def forward(self, x):
        out = self.conv_offset(x)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        return modulated_deform_conv(x, offset, mask, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups, self.deformable_groups)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        version = local_metadata.get('version', None)
        if version is None or version < 2:
            if prefix + 'conv_offset.weight' not in state_dict and prefix[:-1] + '_offset.weight' in state_dict:
                state_dict[prefix + 'conv_offset.weight'] = state_dict.pop(prefix[:-1] + '_offset.weight')
            if prefix + 'conv_offset.bias' not in state_dict and prefix[:-1] + '_offset.bias' in state_dict:
                state_dict[prefix + 'conv_offset.bias'] = state_dict.pop(prefix[:-1] + '_offset.bias')
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)


def convbn_3d(in_planes, out_planes, kernel_size, stride, pad):
    return nn.Sequential(nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, padding=pad, stride=stride, bias=False), nn.BatchNorm3d(out_planes))


def UpsampleConv3D_BN(in_nfeat, out_nfeat, kernel_size=3, padding=1, stride=1):
    return nn.Sequential(nn.Upsample(scale_factor=2, mode='trilinear'), nn.Conv3d(in_nfeat, out_nfeat, kernel_size, stride=stride, padding=padding, bias=False), nn.BatchNorm3d(out_nfeat))


class hourglass3D(nn.Module):

    def __init__(self, inplanes):
        super(hourglass3D, self).__init__()
        self.conv1 = nn.Sequential(convbn_3d(inplanes, inplanes * 2, kernel_size=3, stride=2, pad=1), nn.ReLU(inplace=True))
        self.conv2 = convbn_3d(inplanes * 2, inplanes * 2, kernel_size=3, stride=1, pad=1)
        self.conv3 = nn.Sequential(convbn_3d(inplanes * 2, inplanes * 2, kernel_size=3, stride=2, pad=1), nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(convbn_3d(inplanes * 2, inplanes * 2, kernel_size=3, stride=1, pad=1), nn.ReLU(inplace=True))
        self.conv5 = UpsampleConv3D_BN(inplanes * 2, inplanes * 2)
        self.conv6 = UpsampleConv3D_BN(inplanes * 2, inplanes)

    def forward(self, x, presqu, postsqu):
        out = self.conv1(x)
        pre = self.conv2(out)
        if postsqu is not None:
            pre = F.relu(pre + postsqu, inplace=True)
        else:
            pre = F.relu(pre, inplace=True)
        out = self.conv3(pre)
        out = self.conv4(out)
        if presqu is not None:
            post = F.relu(self.conv5(out) + presqu, inplace=True)
        else:
            post = F.relu(self.conv5(out) + pre, inplace=True)
        out = self.conv6(post)
        return out, pre, post


class CostVolumeFilter(nn.Module):

    def __init__(self, in_nfeat, out_nfeat=32, num_stacks=4):
        super(CostVolumeFilter, self).__init__()
        self.dres0 = nn.Sequential(nn.Conv3d(in_nfeat, out_nfeat, kernel_size=3, padding=1, stride=(1, 2, 2), bias=False), nn.BatchNorm3d(out_nfeat), nn.ReLU(inplace=True), convbn_3d(out_nfeat, out_nfeat, 3, 1, 1), nn.ReLU(inplace=True))
        self.dres1 = nn.Sequential(convbn_3d(out_nfeat, out_nfeat, 3, 1, 1), nn.ReLU(inplace=True), convbn_3d(out_nfeat, out_nfeat, 3, 1, 1))
        self.num_stacks = num_stacks
        for i in range(self.num_stacks):
            self.add_module('HG3D_%d' % i, hourglass3D(out_nfeat))
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

    def forward(self, cost):
        cost0 = self.dres0(cost)
        cost0 = self.dres1(cost0) + cost0
        res = []
        out, pre, post = self._modules['HG3D_0'](cost0, None, None)
        out = out + cost0
        res.append(out)
        for i in range(1, self.num_stacks):
            out, _, post = self._modules['HG3D_%d' % i](cost0, pre, post)
            out = out + cost0
            res.append(out)
        return res


class ConvBlock(nn.Module):

    def __init__(self, in_planes, out_planes, norm='batch'):
        super(ConvBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, int(out_planes / 2))
        self.conv2 = conv3x3(int(out_planes / 2), int(out_planes / 4))
        self.conv3 = conv3x3(int(out_planes / 4), int(out_planes / 4))
        if norm == 'batch':
            self.bn1 = nn.BatchNorm2d(in_planes)
            self.bn2 = nn.BatchNorm2d(int(out_planes / 2))
            self.bn3 = nn.BatchNorm2d(int(out_planes / 4))
            self.bn4 = nn.BatchNorm2d(in_planes)
        elif norm == 'group':
            self.bn1 = nn.GroupNorm(32, in_planes)
            self.bn2 = nn.GroupNorm(32, int(out_planes / 2))
            self.bn3 = nn.GroupNorm(32, int(out_planes / 4))
            self.bn4 = nn.GroupNorm(32, in_planes)
        if in_planes != out_planes:
            self.downsample = nn.Sequential(self.bn4, nn.ReLU(True), nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, bias=False))
        else:
            self.downsample = None

    def forward(self, x):
        residual = x
        out1 = self.bn1(x)
        out1 = F.relu(out1, True)
        out1 = self.conv1(out1)
        out2 = self.bn2(out1)
        out2 = F.relu(out2, True)
        out2 = self.conv2(out2)
        out3 = self.bn3(out2)
        out3 = F.relu(out3, True)
        out3 = self.conv3(out3)
        out3 = torch.cat((out1, out2, out3), 1)
        if self.downsample is not None:
            residual = self.downsample(residual)
        out3 += residual
        return out3


class HourGlass(nn.Module):

    def __init__(self, num_modules, depth, num_features, norm='batch'):
        super(HourGlass, self).__init__()
        self.num_modules = num_modules
        self.depth = depth
        self.features = num_features
        self.norm = norm
        self._generate_network(self.depth)

    def _generate_network(self, level):
        self.add_module('b1_' + str(level), ConvBlock(self.features, self.features, norm=self.norm))
        self.add_module('b2_' + str(level), ConvBlock(self.features, self.features, norm=self.norm))
        if level > 1:
            self._generate_network(level - 1)
        else:
            self.add_module('b2_plus_' + str(level), ConvBlock(self.features, self.features, norm=self.norm))
        self.add_module('b3_' + str(level), ConvBlock(self.features, self.features, norm=self.norm))

    def _forward(self, level, inp):
        up1 = inp
        up1 = self._modules['b1_' + str(level)](up1)
        low1 = F.avg_pool2d(inp, 2, stride=2)
        low1 = self._modules['b2_' + str(level)](low1)
        if level > 1:
            low2 = self._forward(level - 1, low1)
        else:
            low2 = low1
            low2 = self._modules['b2_plus_' + str(level)](low2)
        low3 = low2
        low3 = self._modules['b3_' + str(level)](low3)
        up2 = F.interpolate(low3, scale_factor=2, mode='bicubic', align_corners=True)
        return up1 + up2

    def forward(self, x):
        return self._forward(self.depth, x)


class HGFilter(nn.Module):

    def __init__(self, in_nfeat=3, num_stack=4, norm_type='group', hg_down='ave_pool', num_hourglass=2, hourglass_dim=256):
        super(HGFilter, self).__init__()
        self.num_modules = num_stack
        self.norm_type = norm_type
        self.hg_down = hg_down
        self.num_hourglass = num_hourglass
        self.hourglass_dim = hourglass_dim
        self.conv1 = nn.Conv2d(in_nfeat, 64, kernel_size=7, stride=2, padding=3)
        if self.norm_type == 'batch':
            self.bn1 = nn.BatchNorm2d(64)
        elif self.norm_type == 'group':
            self.bn1 = nn.GroupNorm(32, 64)
        if self.hg_down == 'conv64':
            self.conv2 = ConvBlock(64, 64, self.norm_type)
            self.down_conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        elif self.hg_down == 'conv128':
            self.conv2 = ConvBlock(64, 128, self.norm_type)
            self.down_conv2 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1)
        elif self.hg_down == 'ave_pool':
            self.conv2 = ConvBlock(64, 128, self.norm_type)
        else:
            raise NameError('Unknown Fan Filter setting!')
        self.conv3 = ConvBlock(128, 128, self.norm_type)
        self.conv4 = ConvBlock(128, 256, self.norm_type)
        for hg_module in range(self.num_modules):
            self.add_module('m' + str(hg_module), HourGlass(1, self.num_hourglass, 256, self.norm_type))
            self.add_module('top_m_' + str(hg_module), ConvBlock(256, 256, self.norm_type))
            self.add_module('conv_last' + str(hg_module), nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0))
            if self.norm_type == 'batch':
                self.add_module('bn_end' + str(hg_module), nn.BatchNorm2d(256))
            elif self.norm_type == 'group':
                self.add_module('bn_end' + str(hg_module), nn.GroupNorm(32, 256))
            self.add_module('l' + str(hg_module), nn.Conv2d(256, self.hourglass_dim, kernel_size=1, stride=1, padding=0))
            if hg_module < self.num_modules - 1:
                self.add_module('bl' + str(hg_module), nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0))
                self.add_module('al' + str(hg_module), nn.Conv2d(self.hourglass_dim, 256, kernel_size=1, stride=1, padding=0))

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)), True)
        tmpx = x
        if self.hg_down == 'ave_pool':
            x = F.avg_pool2d(self.conv2(x), 2, stride=2)
        elif self.hg_down in ['conv64', 'conv128']:
            x = self.conv2(x)
            x = self.down_conv2(x)
        else:
            raise NameError('Unknown Fan Filter setting!')
        normx = x
        x = self.conv3(x)
        x = self.conv4(x)
        previous = x
        outputs = []
        for i in range(self.num_modules):
            hg = self._modules['m' + str(i)](previous)
            ll = hg
            ll = self._modules['top_m_' + str(i)](ll)
            ll = F.relu(self._modules['bn_end' + str(i)](self._modules['conv_last' + str(i)](ll)), True)
            tmp_out = self._modules['l' + str(i)](ll)
            outputs.append(tmp_out)
            if i < self.num_modules - 1:
                ll = self._modules['bl' + str(i)](ll)
                tmp_out_ = self._modules['al' + str(i)](tmp_out)
                previous = previous + ll + tmp_out_
        return outputs, tmpx.detach(), normx


class DilateMask(nn.Module):

    def __init__(self) ->None:
        super().__init__()
        kernel = [[0.0, -0.25, 0.0], [-0.25, 1.0, -0.25], [0.0, -0.25, 0.0]]
        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)
        self.padding_func = nn.ReplicationPad2d(1)

    def forward(self, batch_mask, iter_num):
        for _ in range(iter_num):
            padding_mask = self.padding_func(batch_mask)
            res = F.conv2d(padding_mask, self.weight, bias=None, stride=1, padding=0)
            batch_mask[res.abs() > 0.0001] = 1.0
        return batch_mask


class ExtractDepthEdgeMask(nn.Module):

    def __init__(self, thres_) ->None:
        super().__init__()
        self.thres = thres_

    def forward(self, batch_depth):
        B, _, H, W = batch_depth.size()
        patch = F.unfold(batch_depth, kernel_size=3, padding=1, stride=1)
        min_v, _ = patch.min(dim=1, keepdim=True)
        max_v, _ = patch.max(dim=1, keepdim=True)
        mask = max_v - min_v > self.thres
        mask = mask.view(B, 1, H, W).float()
        return mask


class ImgNormalizationAndInv(nn.Module):

    def __init__(self, RGB_mean, RGB_std):
        super().__init__()
        mean = torch.FloatTensor(RGB_mean).view(1, 3, 1, 1)
        std = torch.FloatTensor(RGB_std).view(1, 3, 1, 1)
        self.register_buffer('rgb_mean', mean)
        self.register_buffer('rgb_std', std)

    def forward(self, image_tensor, inv):
        if inv:
            image_tensor = image_tensor * self.rgb_std + self.rgb_mean
        else:
            image_tensor = (image_tensor - self.rgb_mean) / self.rgb_std
        return image_tensor


class SurfaceClassifier(nn.Module):

    def __init__(self, filter_channels, no_residual=False, last_op=nn.Sigmoid()):
        super(SurfaceClassifier, self).__init__()
        self.filters = []
        self.no_residual = no_residual
        filter_channels = filter_channels
        self.last_op = last_op
        if self.no_residual:
            for l in range(0, len(filter_channels) - 1):
                self.filters.append(nn.Conv1d(filter_channels[l], filter_channels[l + 1], 1))
                self.add_module('conv%d' % l, self.filters[l])
        else:
            for l in range(0, len(filter_channels) - 1):
                if 0 != l:
                    self.filters.append(nn.Conv1d(filter_channels[l] + filter_channels[0], filter_channels[l + 1], 1))
                else:
                    self.filters.append(nn.Conv1d(filter_channels[l], filter_channels[l + 1], 1))
                self.add_module('conv%d' % l, self.filters[l])

    def _forward(self, feature, return_inter_var=False):
        y = feature
        tmpy = feature
        num_filter = len(self.filters)
        for i in range(num_filter):
            if self.no_residual:
                y = self._modules['conv' + str(i)](y)
            else:
                y = self._modules['conv' + str(i)](y if i == 0 else torch.cat([y, tmpy], 1))
            if i != len(self.filters) - 1:
                y = F.leaky_relu(y)
            if return_inter_var and i == num_filter // 2:
                return y
        if self.last_op:
            y = self.last_op(y)
        return y

    def forward(self, feature_1, feature_2=None):
        """

        :param feature: list of [BxC_inxHxW] tensors of image features
        :param xy: [Bx3xN] tensor of (x,y) coodinates in the image plane
        :return: [BxC_outxN] tensor of features extracted at the coordinates
        """
        if feature_2 is None:
            return self._forward(feature_1)
        else:
            inter_var1 = self._forward(feature_1, return_inter_var=True)
            inter_var2 = self._forward(feature_2, return_inter_var=True)
            y = torch.stack([inter_var1, inter_var2], dim=1).mean(dim=1)
            tmpy = torch.stack([feature_1, feature_2], dim=1).mean(dim=1)
            num_filter = len(self.filters)
            inter_layer_index = 1 + num_filter // 2
            for i in range(inter_layer_index, num_filter):
                if self.no_residual:
                    y = self._modules['conv' + str(i)](y)
                else:
                    y = self._modules['conv' + str(i)](y if i == 0 else torch.cat([y, tmpy], 1))
                if i != len(self.filters) - 1:
                    y = F.leaky_relu(y)
        if self.last_op:
            y = self.last_op(y)
        return y


class ZEDProject(nn.Module):

    def __init__(self, camera, use_oriZ) ->None:
        super().__init__()
        ExterMat = torch.from_numpy(camera['ExterMat']).unsqueeze(0)
        InterMat = torch.from_numpy(camera['InterMat']).unsqueeze(0)
        self.register_buffer('exter_rot', ExterMat[:, :3, :3])
        self.register_buffer('exter_trans', ExterMat[:, :3, 3:4])
        self.register_buffer('inter_scale', InterMat[:, :2, :2])
        self.register_buffer('inter_trans', InterMat[:, :2, 2:])
        self.use_oriZ = use_oriZ

    def __call__(self, points):
        """
        points: [B, 3, N]
        """
        batch_size = points.size(0)
        exter_trans = self.exter_trans.expand(batch_size, -1, -1)
        exter_rot = self.exter_rot.expand(batch_size, -1, -1)
        inter_trans = self.inter_trans.expand(batch_size, -1, -1)
        inter_scale = self.inter_scale.expand(batch_size, -1, -1)
        homo = torch.baddbmm(exter_trans, exter_rot, points)
        xy = homo[:, :2, :] / homo[:, 2:3, :]
        uv = torch.baddbmm(inter_trans, inter_scale, xy)
        if self.use_oriZ:
            return uv, points[:, 2:3, :]
        else:
            return uv, homo[:, 2:3, :]


class ZEDCamera(object):

    def __init__(self, reso_X_Y, crop_X_Y_ori, crop_X_Y_lenght, position_3d, normalize_camera):
        super().__init__()
        self.normalize_camera = normalize_camera
        self.L_camera_path = './ConfigData/L_Cam_1920x1080_30fps.yml'
        self.R_camera_path = './ConfigData/R_Cam_1920x1080_30fps.yml'
        self.position_3d = position_3d
        self.crop_X_Y_ori = crop_X_Y_ori
        self.crop_X_Y_lenght = crop_X_Y_lenght
        self.reso_X_Y = reso_X_Y
        self._build_camera_para(is_left=True)
        self._build_camera_para(is_left=False)
        self._calc_baseline()

    def _build_camera_para(self, is_left):
        if is_left:
            fs = cv2.FileStorage(self.L_camera_path, cv2.FILE_STORAGE_READ)
        else:
            fs = cv2.FileStorage(self.R_camera_path, cv2.FILE_STORAGE_READ)
        ExterMat = fs.getNode('ExterMat').mat().astype(np.float32)
        InterMat = fs.getNode('InterMat').mat().astype(np.float32)
        CameraReso = fs.getNode('CameraSize').mat().astype(np.float32)
        fs.release()
        ExterMat[0, 3] += self.position_3d[0]
        ExterMat[1, 3] += self.position_3d[1]
        ExterMat[2, 3] += self.position_3d[2]
        InterMat[0, 2] = InterMat[0, 2] - self.crop_X_Y_ori[0]
        InterMat[1, 2] = InterMat[1, 2] - self.crop_X_Y_ori[1]
        if self.normalize_camera:
            InterMat[0, :] = 2.0 * InterMat[0, :] / self.crop_X_Y_lenght[0]
            InterMat[1, :] = 2.0 * InterMat[1, :] / self.crop_X_Y_lenght[1]
            InterMat[:2, 2] = InterMat[:2, 2] - 1.0
        else:
            InterMat[0, :] = InterMat[0, :] * self.reso_X_Y[0] / self.crop_X_Y_lenght[0]
            InterMat[1, :] = InterMat[1, :] * self.reso_X_Y[1] / self.crop_X_Y_lenght[1]
        CameraReso = np.array([self.reso_X_Y[0], self.reso_X_Y[1]])
        if is_left:
            self.l_camera = {'ExterMat': ExterMat, 'InterMat': InterMat, 'CameraReso': CameraReso}
        else:
            self.r_camera = {'ExterMat': ExterMat, 'InterMat': InterMat, 'CameraReso': CameraReso}

    def Calc_Z(self, one_point, use_l_camera):
        if use_l_camera:
            exter_mat = self.l_camera['ExterMat']
        else:
            exter_mat = self.r_camera['ExterMat']
        one_point = one_point.reshape(3, 1)
        res = exter_mat[:3, :3].dot(one_point) + exter_mat[:3, 3:]
        return res[2, 0]

    def _calc_baseline(self):
        self.baseline = abs(self.r_camera['ExterMat'][0, 3] - self.l_camera['ExterMat'][0, 3]) * self.r_camera['InterMat'][0, 0]

    def print_camera_info(self):
        None
        None
        None
        None
        None
        None
        None
        None
        None
        None
        None
        None
        None
        None
        None
        None
        None
        None
        None
        None


class StereoPIFuNet(nn.Module):

    def __init__(self, use_VisualHull, use_bb_of_rootjoint):
        super().__init__()
        self.use_VisualHull = use_VisualHull
        self.use_bb_of_rootjoint = use_bb_of_rootjoint
        self.max_disp = 72
        self.num_stacks = 4
        self.normalize_rgb = True
        self.use_OriZ = False
        self.img_size = int(zed_camera_regu.l_camera['CameraReso'][0])
        self.baseline = zed_camera_regu.baseline
        self.Stereo_RGB_mean = [0.485, 0.456, 0.406]
        self.Stereo_RGB_std = [0.229, 0.224, 0.225]
        self.HG_RGB_mean = [0.5, 0.5, 0.5]
        self.HG_RGB_std = [0.5, 0.5, 0.5]
        self.depth_edge_thres = 0.08
        self.num_3Dnfeat = 64
        self.point_feature = 256 + self.num_3Dnfeat + 1 + 1
        self.im_nfeat = 3
        self.sigmoid_coef = 50.0
        if self.sigmoid_coef > 0:
            self.use_sigmoid_z = True
        else:
            self.use_sigmoid_z = False
        self.LeftFeature_List = None
        self.CostVolume3D_List = None
        self.ConfidVolume3D = None
        self.DepthTensor = None
        self.DepthEdgeMask = None
        self.LMaskTensor = None
        self.RMaskTensor = None
        self.l_rgb = None
        self.r_rgb = None
        self._build_tool_funcs()

    def _build_tool_funcs(self):
        self.Cost3DFilter = CostVolumeFilter(in_nfeat=224, out_nfeat=self.num_3Dnfeat, num_stacks=self.num_stacks)
        self.image_filter = HGFilter(in_nfeat=self.im_nfeat, num_stack=self.num_stacks)
        self.surface_classifier = SurfaceClassifier(filter_channels=[self.point_feature, 1024, 512, 256, 128, 1])
        self.l_projection = ZEDProject(zed_camera_normalized.l_camera, self.use_OriZ)
        self.r_projection = ZEDProject(zed_camera_normalized.r_camera, self.use_OriZ)
        self.aanet_feat_extractor = AANetPlusFeature(self.img_size, self.max_disp)
        self.dilate_mask_func = DilateMask()
        self.StereoRGBNormalizer = ImgNormalizationAndInv(self.Stereo_RGB_mean, self.Stereo_RGB_std)
        self.HGNormalizer = ImgNormalizationAndInv(self.HG_RGB_mean, self.HG_RGB_std)
        self.extract_depth_edge_mask_func = ExtractDepthEdgeMask(thres_=self.depth_edge_thres)

    @staticmethod
    def calc_mean_z(depth_tensor, mask_c1b):
        batch_size = depth_tensor.size(0)
        res = []
        for i in range(batch_size):
            res.append(torch.mean(depth_tensor[i, :, :, :][mask_c1b[i, :, :, :]]))
        return torch.stack(res, dim=0).view(batch_size, 1, 1)

    def DepthDispConvertor(self, batch_input, batch_mask):
        if batch_input.dim() == 3:
            batch_input = batch_input.unsqueeze(1)
        assert 4 == batch_mask.dim()
        if batch_mask.dtype != torch.bool:
            batch_mask = batch_mask > 0.5
        mask = batch_mask & (batch_input > 0.0001)
        res = batch_input.clone()
        res[mask] = self.baseline / batch_input[mask]
        res[~mask] = 0.0
        return res

    def update_feature_by_imgs(self, l_rgb, r_rgb, l_mask, r_mask):
        l_mask_c1b = l_mask > 0.5
        l_mask_c3b = l_mask_c1b.expand(-1, 3, -1, -1)
        left_feature, right_feature, cost_volume3D, confidence_volume, disparity_map = self.aanet_feat_extractor(l_rgb, r_rgb)
        depth_tensor = self.DepthDispConvertor(disparity_map.detach(), l_mask)
        ori_l_rgb = self.StereoRGBNormalizer(l_rgb, inv=True)
        hg_l_rgb = self.HGNormalizer(ori_l_rgb, inv=False)
        hg_l_rgb[~l_mask_c3b] = 0.0
        im_input = hg_l_rgb
        self.LeftFeature_List = self.image_filter(im_input)[0]
        self.CostVolume3D_List = self.Cost3DFilter(cost_volume3D.detach())
        self.ConfidVolume3D = confidence_volume.detach()
        l_mask_tensor = self.dilate_mask_func(l_mask, 2).detach() if self.use_VisualHull else None
        r_mask_tensor = self.dilate_mask_func(r_mask, 2).detach() if self.use_VisualHull else None
        self.DepthTensor = depth_tensor.detach()
        self.DepthEdgeMask = self.extract_depth_edge_mask_func(self.DepthTensor)
        self.LMaskTensor = l_mask_tensor
        self.RMaskTensor = r_mask_tensor
        self.mean_z = self.calc_mean_z(self.DepthTensor, l_mask_c1b)

    @staticmethod
    def index2D(feat, uv, mode='bilinear'):
        uv = uv.transpose(1, 2)
        uv = uv.unsqueeze(2)
        samples = torch.nn.functional.grid_sample(feat, uv, align_corners=True, mode=mode)
        return samples[:, :, :, 0]

    @staticmethod
    def index3D(feat, uv3d, mode='bilinear'):
        grid_ = uv3d.permute(0, 2, 1).unsqueeze(1).unsqueeze(1)
        samples = torch.nn.functional.grid_sample(feat, grid_, mode=mode)
        return samples[:, :, 0, 0, :]

    def get_feat2D(self, feat, uv, edge_mask):
        if edge_mask is not None:
            feat_bilinear = self.index2D(feat, uv, mode='bilinear')
            feat_nearest = self.index2D(feat, uv, mode='nearest')
            feat = torch.where(edge_mask, feat_nearest, feat_bilinear)
        else:
            feat = self.index2D(feat, uv, mode='bilinear')
        return feat

    def get_feat3D(self, feat, uv, edge_mask):
        if edge_mask is not None:
            feat_bilinear = self.index3D(feat, uv, mode='bilinear')
            feat_nearest = self.index3D(feat, uv, mode='nearest')
            feat = torch.where(edge_mask, feat_nearest, feat_bilinear)
        else:
            feat = self.index3D(feat, uv, mode='bilinear')
        return feat

    def sample_feature(self, points):
        assert self.DepthTensor is not None
        assert self.DepthEdgeMask is not None
        point_num = points.size(-1)
        batch_size = points.size(0)
        left_uv, left_z = self.l_projection(points)
        right_uv, right_z = self.r_projection(points)
        disp = (left_uv[:, 0:1, :] - right_uv[:, 0:1, :]) * self.img_size / self.max_disp - 1.0
        uv3d = torch.cat([left_uv, disp], dim=1)
        in_img = (uv3d[:, 0] >= -1.0) & (uv3d[:, 0] <= 1.0) & (uv3d[:, 1] >= -1.0) & (uv3d[:, 1] <= 1.0) & (uv3d[:, 2] >= -1.0) & (uv3d[:, 2] <= 1.0)
        if self.use_VisualHull:
            assert self.LMaskTensor is not None
            assert self.LMaskTensor.dtype == torch.float32
            assert self.RMaskTensor is not None
            assert self.RMaskTensor.dtype == torch.float32
            l_mask_value = self.index2D(self.LMaskTensor, left_uv, mode='nearest')
            r_mask_value = self.index2D(self.RMaskTensor, right_uv, mode='nearest')
            in_img = in_img & (l_mask_value[:, 0] > 0.5) & (r_mask_value[:, 0] > 0.5)
        if self.use_bb_of_rootjoint:
            temp_z = left_z - self.mean_z
            in_img = in_img & (temp_z[:, 0, :] > -0.7) & (temp_z[:, 0, :] < 0.7)
        point_feat_mask_c1f = in_img.view(batch_size, 1, point_num).float()
        edge_mask_c1f = self.index2D(self.DepthEdgeMask, left_uv, mode='bilinear')
        edge_mask_c1b = edge_mask_c1f > 0.01
        z_predict = self.get_feat2D(self.DepthTensor, left_uv, edge_mask=edge_mask_c1b)
        left_z = left_z - z_predict
        if self.use_sigmoid_z:
            left_z = 2.0 / (1.0 + torch.exp(-1.0 * self.sigmoid_coef * left_z)) - 1.0
        confid_feature = self.get_feat3D(self.ConfidVolume3D, uv3d, edge_mask_c1b)
        res = []
        for i in range(self.num_stacks):
            left_feature = self.index2D(self.LeftFeature_List[i], left_uv)
            cost_feature = self.index3D(self.CostVolume3D_List[i], uv3d)
            res.append(torch.cat([left_feature, cost_feature, left_z, confid_feature], dim=1))
        return res, point_feat_mask_c1f

    def query(self, points):
        assert self.LeftFeature_List is not None
        assert self.DepthTensor is not None
        point_feat_list, point_feat_mask_c1f = self.sample_feature(points)
        pred = point_feat_mask_c1f * self.surface_classifier(point_feat_list[-1])
        return pred


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (BasicConv,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ConvBlock,
     lambda: ([], {'in_planes': 4, 'out_planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (CostVolume,
     lambda: ([], {'max_disp': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (CostVolumeFilter,
     lambda: ([], {'in_nfeat': 4}),
     lambda: ([torch.rand([4, 4, 4, 2, 2])], {})),
    (DilateMask,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), 0], {})),
    (DisparityEstimation,
     lambda: ([], {'max_disp': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ExtractDepthEdgeMask,
     lambda: ([], {'thres_': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (FeaturePyrmaid,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 32, 64, 64])], {})),
    (HGFilter,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {})),
    (HourGlass,
     lambda: ([], {'num_modules': torch.nn.ReLU(), 'depth': 1, 'num_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (SimpleBottleneck,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (SurfaceClassifier,
     lambda: ([], {'filter_channels': [4, 4]}),
     lambda: ([torch.rand([4, 4])], {})),
]

