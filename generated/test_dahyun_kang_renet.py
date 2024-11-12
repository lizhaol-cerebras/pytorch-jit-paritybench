
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


import torch


import random


import numpy as np


import torch.nn as nn


from torch.utils.data import Dataset


from torchvision import transforms


import torch.nn.functional as F


import torch.nn.init as init


from torch import nn


from torch.nn import functional as F


import torch.nn.functional as Func


from torch.nn.modules.utils import _quadruple


from torch.utils.data import DataLoader


import time


class SepConv4d(nn.Module):
    """ approximates 3 x 3 x 3 x 3 kernels via two subsequent 3 x 3 x 1 x 1 and 1 x 1 x 3 x 3 """

    def __init__(self, in_planes, out_planes, stride=(1, 1, 1), ksize=3, do_padding=True, bias=False):
        super(SepConv4d, self).__init__()
        self.isproj = False
        padding1 = (0, ksize // 2, ksize // 2) if do_padding else (0, 0, 0)
        padding2 = (ksize // 2, ksize // 2, 0) if do_padding else (0, 0, 0)
        if in_planes != out_planes:
            self.isproj = True
            self.proj = nn.Sequential(nn.Conv2d(in_channels=in_planes, out_channels=out_planes, kernel_size=1, bias=bias, padding=0), nn.BatchNorm2d(out_planes))
        self.conv1 = nn.Sequential(nn.Conv3d(in_channels=in_planes, out_channels=in_planes, kernel_size=(1, ksize, ksize), stride=stride, bias=bias, padding=padding1), nn.BatchNorm3d(in_planes))
        self.conv2 = nn.Sequential(nn.Conv3d(in_channels=in_planes, out_channels=in_planes, kernel_size=(ksize, ksize, 1), stride=stride, bias=bias, padding=padding2), nn.BatchNorm3d(in_planes))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        b, c, u, v, h, w = x.shape
        x = self.conv2(x.view(b, c, u, v, -1))
        b, c, u, v, _ = x.shape
        x = self.relu(x)
        x = self.conv1(x.view(b, c, -1, h, w))
        b, c, _, h, w = x.shape
        if self.isproj:
            x = self.proj(x.view(b, c, -1, w))
        x = x.view(b, -1, u, v, h, w)
        return x


class CCA(torch.nn.Module):

    def __init__(self, kernel_sizes=[3, 3], planes=[16, 1]):
        super(CCA, self).__init__()
        num_layers = len(kernel_sizes)
        nn_modules = list()
        for i in range(num_layers):
            ch_in = 1 if i == 0 else planes[i - 1]
            ch_out = planes[i]
            k_size = kernel_sizes[i]
            nn_modules.append(SepConv4d(in_planes=ch_in, out_planes=ch_out, ksize=k_size, do_padding=True))
            if i != num_layers - 1:
                nn_modules.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*nn_modules)

    def forward(self, x):
        x = self.conv(x) + self.conv(x.permute(0, 1, 4, 5, 2, 3)).permute(0, 1, 4, 5, 2, 3)
        return x


class LocalSelfAttention(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, bias=False):
        super(LocalSelfAttention, self).__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        assert self.out_channels % self.groups == 0, 'out_channels should be divided by groups. (example: out_channels: 40, groups: 4)'
        self.rel_h = nn.Parameter(torch.randn(out_channels // 2, 1, 1, kernel_size, 1), requires_grad=True)
        self.rel_w = nn.Parameter(torch.randn(out_channels // 2, 1, 1, 1, kernel_size), requires_grad=True)
        self.key_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.query_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.value_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.agg = nn.Sequential(nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True), nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False), nn.BatchNorm2d(out_channels))
        self.reset_parameters()

    def forward(self, x):
        batch, channels, height, width = x.size()
        padded_x = F.pad(x, [self.padding, self.padding, self.padding, self.padding])
        q_out = self.query_conv(x)
        k_out = self.key_conv(padded_x)
        v_out = self.value_conv(padded_x)
        k_out = k_out.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)
        v_out = v_out.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)
        k_out_h, k_out_w = k_out.split(self.out_channels // 2, dim=1)
        k_out = torch.cat((k_out_h + self.rel_h, k_out_w + self.rel_w), dim=1)
        k_out = k_out.contiguous().view(batch, self.groups, self.out_channels // self.groups, height, width, -1)
        v_out = v_out.contiguous().view(batch, self.groups, self.out_channels // self.groups, height, width, -1)
        q_out = q_out.view(batch, self.groups, self.out_channels // self.groups, height, width, 1)
        out = q_out * k_out
        out = F.softmax(out, dim=-1)
        out = torch.einsum('bnchwk,bnchwk -> bnchw', out, v_out).view(batch, -1, height, width)
        out = self.agg(out)
        return out

    def reset_parameters(self):
        init.kaiming_normal_(self.key_conv.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.value_conv.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.query_conv.weight, mode='fan_out', nonlinearity='relu')
        init.normal_(self.rel_h, 0, 1)
        init.normal_(self.rel_w, 0, 1)


class _NonLocalBlockND(nn.Module):

    def __init__(self, in_channels, inter_channels=None, dimension=3, sub_sample=True):
        super(_NonLocalBlockND, self).__init__()
        assert dimension in [1, 2, 3]
        self.dimension = dimension
        self.sub_sample = sub_sample
        self.in_channels = in_channels
        self.inter_channels = inter_channels
        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1
        conv_nd = nn.Conv2d
        max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
        bn = nn.BatchNorm2d
        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0)
        self.W = nn.Sequential(conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1, stride=1, padding=0), bn(self.in_channels))
        nn.init.constant_(self.W[1].weight, 0)
        nn.init.constant_(self.W[1].bias, 0)
        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0)
        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0)
        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)

    def forward(self, x):
        """
        :param x: (b, c, t, h, w)
        :return:
        """
        batch_size = x.size(0)
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)
        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1)
        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x
        return z


class NonLocalSelfAttention(_NonLocalBlockND):

    def __init__(self, in_channels, inter_channels=None, sub_sample=True):
        super(NonLocalSelfAttention, self).__init__(in_channels, inter_channels=inter_channels, dimension=2, sub_sample=sub_sample)


def featureL2Norm(feature):
    epsilon = 1e-06
    norm = torch.pow(torch.sum(torch.pow(feature, 2), 1) + epsilon, 0.5).unsqueeze(1).expand_as(feature)
    return torch.div(feature, norm)


def generate_spatial_descriptor(data, kernel_size):
    """
    Applies self local similarity with fixed sliding window.
    Args:
        data: featuren map, variable of shape (b,c,h,w)
        kernel_size: width/heigh of local window, int
    Returns:
        output: global spatial map, variable of shape (b,c,h,w)
    """
    padding = int(kernel_size // 2)
    b, c, h, w = data.shape
    p2d = _quadruple(padding)
    data_padded = Func.pad(data, p2d, 'constant', 0)
    assert data_padded.shape == (b, c, h + 2 * padding, w + 2 * padding), 'Error: data_padded shape{} wrong!'.format(data_padded.shape)
    output = torch.zeros(size=[b, kernel_size * kernel_size, h, w], requires_grad=data.requires_grad)
    if data.is_cuda:
        output = output
    for hi in range(h):
        for wj in range(w):
            q = data[:, :, hi, wj].contiguous()
            i = hi + padding
            j = wj + padding
            hs = i - padding
            he = i + padding + 1
            ws = j - padding
            we = j + padding + 1
            patch = data_padded[:, :, hs:he, ws:we].contiguous()
            assert patch.shape == (b, c, kernel_size, kernel_size)
            hk, wk = kernel_size, kernel_size
            feature_a = q.view(b, c, 1 * 1).transpose(1, 2)
            feature_b = patch.view(b, c, hk * wk)
            feature_mul = torch.bmm(feature_a, feature_b)
            assert feature_mul.shape == (b, 1, hk * wk)
            correlation_tensor = feature_mul.unsqueeze(1)
            output[:, :, hi, wj] = correlation_tensor.squeeze(1).squeeze(1)
    return output


class SpatialContextEncoder(torch.nn.Module):
    """
    Spatial Context Encoder.
    Author: Shuaiyi Huang
    Input:
        x: feature of shape (b,c,h,w)
    Output:
        feature_embd: context-aware semantic feature of shape (b,c+k**2,h,w), where k is the kernel size of spatial descriptor
    """

    def __init__(self, planes=None, kernel_size=None):
        super(SpatialContextEncoder, self).__init__()
        self.kernel_size = kernel_size
        self.conv1x1_in = nn.Sequential(nn.Conv2d(planes[0], planes[1], kernel_size=1, bias=False, padding=0), nn.BatchNorm2d(planes[1]), nn.ReLU(inplace=True))
        self.embeddingFea = nn.Sequential(nn.Conv2d(planes[1] + kernel_size * kernel_size, planes[2], kernel_size=1, bias=False, padding=0), nn.BatchNorm2d(planes[2]), nn.ReLU(inplace=True))
        self.conv1x1_out = nn.Sequential(nn.Conv2d(planes[2], planes[3], kernel_size=1, bias=False, padding=0), nn.BatchNorm2d(planes[3]))
        None
        return

    def forward(self, x):
        x = self.conv1x1_in(x)
        feature_gs = generate_spatial_descriptor(x, kernel_size=self.kernel_size)
        feature_gs = featureL2Norm(feature_gs)
        feature_cat = torch.cat([x, feature_gs], 1)
        feature_embd = self.embeddingFea(feature_cat)
        feature_embd = self.conv1x1_out(feature_embd)
        return feature_embd


class SqueezeExcitation(nn.Module):

    def __init__(self, channel, reduction=16):
        super(SqueezeExcitation, self).__init__()
        hdim = 64
        self.conv1x1_in = nn.Sequential(nn.Conv2d(channel, hdim, kernel_size=1, bias=False, padding=0), nn.BatchNorm2d(hdim), nn.ReLU(inplace=False))
        self.conv1x1_out = nn.Sequential(nn.Conv2d(hdim, channel, kernel_size=1, bias=False, padding=0), nn.BatchNorm2d(channel), nn.ReLU(inplace=False))
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(hdim, hdim // reduction, bias=False), nn.ReLU(inplace=False), nn.Linear(hdim // reduction, hdim, bias=False), nn.Sigmoid())

    def forward(self, x):
        x = self.conv1x1_in(x)
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        x = x * y.expand_as(x)
        x = self.conv1x1_out(x)
        return x


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.LeakyReLU(0.1)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv3x3(planes, planes)
        self.bn3 = nn.BatchNorm2d(planes)
        self.maxpool = nn.MaxPool2d(stride)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        out = self.maxpool(out)
        return out


class ResNet(nn.Module):

    def __init__(self, args, block=BasicBlock):
        self.inplanes = 3
        super(ResNet, self).__init__()
        self.args = args
        self.layer1 = self._make_layer(block, 64, stride=2)
        self.layer2 = self._make_layer(block, 160, stride=2)
        self.layer3 = self._make_layer(block, 320, stride=2)
        self.layer4 = self._make_layer(block, 640, stride=2)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=1, bias=False), nn.BatchNorm2d(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


class SCR(nn.Module):

    def __init__(self, planes=[640, 64, 64, 64, 640], stride=(1, 1, 1), ksize=3, do_padding=False, bias=False):
        super(SCR, self).__init__()
        self.ksize = _quadruple(ksize) if isinstance(ksize, int) else ksize
        padding1 = (0, self.ksize[2] // 2, self.ksize[3] // 2) if do_padding else (0, 0, 0)
        self.conv1x1_in = nn.Sequential(nn.Conv2d(planes[0], planes[1], kernel_size=1, bias=False, padding=0), nn.BatchNorm2d(planes[1]), nn.ReLU(inplace=True))
        self.conv1 = nn.Sequential(nn.Conv3d(planes[1], planes[2], (1, self.ksize[2], self.ksize[3]), stride=stride, bias=bias, padding=padding1), nn.BatchNorm3d(planes[2]), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv3d(planes[2], planes[3], (1, self.ksize[2], self.ksize[3]), stride=stride, bias=bias, padding=padding1), nn.BatchNorm3d(planes[3]), nn.ReLU(inplace=True))
        self.conv1x1_out = nn.Sequential(nn.Conv2d(planes[3], planes[4], kernel_size=1, bias=False, padding=0), nn.BatchNorm2d(planes[4]))

    def forward(self, x):
        b, c, h, w, u, v = x.shape
        x = x.view(b, c, h * w, u * v)
        x = self.conv1x1_in(x)
        c = x.shape[1]
        x = x.view(b, c, h * w, u, v)
        x = self.conv1(x)
        x = self.conv2(x)
        c = x.shape[1]
        x = x.view(b, c, h, w)
        x = self.conv1x1_out(x)
        return x


class SelfCorrelationComputation(nn.Module):

    def __init__(self, kernel_size=(5, 5), padding=2):
        super(SelfCorrelationComputation, self).__init__()
        self.kernel_size = kernel_size
        self.unfold = nn.Unfold(kernel_size=kernel_size, padding=padding)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        b, c, h, w = x.shape
        x = self.relu(x)
        x = F.normalize(x, dim=1, p=2)
        identity = x
        x = self.unfold(x)
        x = x.view(b, c, self.kernel_size[0], self.kernel_size[1], h, w)
        x = x * identity.unsqueeze(2).unsqueeze(2)
        x = x.permute(0, 1, 4, 5, 2, 3).contiguous()
        return x


class RENet(nn.Module):

    def __init__(self, args, mode=None):
        super().__init__()
        self.mode = mode
        self.args = args
        self.encoder = ResNet(args=args)
        self.encoder_dim = 640
        self.fc = nn.Linear(self.encoder_dim, self.args.num_class)
        self.scr_module = self._make_scr_layer(planes=[640, 64, 64, 64, 640])
        self.cca_module = CCA(kernel_sizes=[3, 3], planes=[16, 1])
        self.cca_1x1 = nn.Sequential(nn.Conv2d(self.encoder_dim, 64, kernel_size=1, bias=False), nn.BatchNorm2d(64), nn.ReLU())

    def _make_scr_layer(self, planes):
        stride, kernel_size, padding = (1, 1, 1), (5, 5), 2
        layers = list()
        if self.args.self_method == 'scr':
            corr_block = SelfCorrelationComputation(kernel_size=kernel_size, padding=padding)
            self_block = SCR(planes=planes, stride=stride)
        elif self.args.self_method == 'sce':
            planes = [640, 64, 64, 640]
            self_block = SpatialContextEncoder(planes=planes, kernel_size=kernel_size[0])
        elif self.args.self_method == 'se':
            self_block = SqueezeExcitation(channel=planes[0])
        elif self.args.self_method == 'lsa':
            self_block = LocalSelfAttention(in_channels=planes[0], out_channels=planes[0], kernel_size=kernel_size[0])
        elif self.args.self_method == 'nlsa':
            self_block = NonLocalSelfAttention(planes[0], sub_sample=False)
        else:
            raise NotImplementedError
        if self.args.self_method == 'scr':
            layers.append(corr_block)
        layers.append(self_block)
        return nn.Sequential(*layers)

    def forward(self, input):
        if self.mode == 'fc':
            return self.fc_forward(input)
        elif self.mode == 'encoder':
            return self.encode(input, False)
        elif self.mode == 'cca':
            spt, qry = input
            return self.cca(spt, qry)
        else:
            raise ValueError('Unknown mode')

    def fc_forward(self, x):
        x = x.mean(dim=[-1, -2])
        return self.fc(x)

    def cca(self, spt, qry):
        spt = spt.squeeze(0)
        spt = self.normalize_feature(spt)
        qry = self.normalize_feature(qry)
        corr4d = self.get_4d_correlation_map(spt, qry)
        num_qry, way, H_s, W_s, H_q, W_q = corr4d.size()
        corr4d = self.cca_module(corr4d.view(-1, 1, H_s, W_s, H_q, W_q))
        corr4d_s = corr4d.view(num_qry, way, H_s * W_s, H_q, W_q)
        corr4d_q = corr4d.view(num_qry, way, H_s, W_s, H_q * W_q)
        corr4d_s = self.gaussian_normalize(corr4d_s, dim=2)
        corr4d_q = self.gaussian_normalize(corr4d_q, dim=4)
        corr4d_s = F.softmax(corr4d_s / self.args.temperature_attn, dim=2)
        corr4d_s = corr4d_s.view(num_qry, way, H_s, W_s, H_q, W_q)
        corr4d_q = F.softmax(corr4d_q / self.args.temperature_attn, dim=4)
        corr4d_q = corr4d_q.view(num_qry, way, H_s, W_s, H_q, W_q)
        attn_s = corr4d_s.sum(dim=[4, 5])
        attn_q = corr4d_q.sum(dim=[2, 3])
        spt_attended = attn_s.unsqueeze(2) * spt.unsqueeze(0)
        qry_attended = attn_q.unsqueeze(2) * qry.unsqueeze(1)
        if self.args.shot > 1:
            spt_attended = spt_attended.view(num_qry, self.args.shot, self.args.way, *spt_attended.shape[2:])
            qry_attended = qry_attended.view(num_qry, self.args.shot, self.args.way, *qry_attended.shape[2:])
            spt_attended = spt_attended.mean(dim=1)
            qry_attended = qry_attended.mean(dim=1)
        spt_attended_pooled = spt_attended.mean(dim=[-1, -2])
        qry_attended_pooled = qry_attended.mean(dim=[-1, -2])
        qry_pooled = qry.mean(dim=[-1, -2])
        similarity_matrix = F.cosine_similarity(spt_attended_pooled, qry_attended_pooled, dim=-1)
        if self.training:
            return similarity_matrix / self.args.temperature, self.fc(qry_pooled)
        else:
            return similarity_matrix / self.args.temperature

    def gaussian_normalize(self, x, dim, eps=1e-05):
        x_mean = torch.mean(x, dim=dim, keepdim=True)
        x_var = torch.var(x, dim=dim, keepdim=True)
        x = torch.div(x - x_mean, torch.sqrt(x_var + eps))
        return x

    def get_4d_correlation_map(self, spt, qry):
        """
        The value H and W both for support and query is the same, but their subscripts are symbolic.
        :param spt: way * C * H_s * W_s
        :param qry: num_qry * C * H_q * W_q
        :return: 4d correlation tensor: num_qry * way * H_s * W_s * H_q * W_q
        :rtype:
        """
        way = spt.shape[0]
        num_qry = qry.shape[0]
        spt = self.cca_1x1(spt)
        qry = self.cca_1x1(qry)
        spt = F.normalize(spt, p=2, dim=1, eps=1e-08)
        qry = F.normalize(qry, p=2, dim=1, eps=1e-08)
        spt = spt.unsqueeze(0).repeat(num_qry, 1, 1, 1, 1)
        qry = qry.unsqueeze(1).repeat(1, way, 1, 1, 1)
        similarity_map_einsum = torch.einsum('qncij,qnckl->qnijkl', spt, qry)
        return similarity_map_einsum

    def normalize_feature(self, x):
        return x - x.mean(1).unsqueeze(1)

    def encode(self, x, do_gap=True):
        x = self.encoder(x)
        if self.args.self_method:
            identity = x
            x = self.scr_module(x)
            if self.args.self_method == 'scr':
                x = x + identity
            x = F.relu(x, inplace=True)
        if do_gap:
            return F.adaptive_avg_pool2d(x, 1)
        else:
            return x


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (BasicBlock,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (CCA,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1, 4, 4, 4, 4])], {})),
    (LocalSelfAttention,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (NonLocalSelfAttention,
     lambda: ([], {'in_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ResNet,
     lambda: ([], {'args': SimpleNamespace()}),
     lambda: ([torch.rand([4, 3, 64, 64])], {})),
    (SelfCorrelationComputation,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (SepConv4d,
     lambda: ([], {'in_planes': 4, 'out_planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4, 4, 4])], {})),
    (SqueezeExcitation,
     lambda: ([], {'channel': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (_NonLocalBlockND,
     lambda: ([], {'in_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
]

