
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


import torch.nn as nn


import torch.nn.functional as F


import torch as torch


from torch.nn import functional as F


from copy import deepcopy


from torchvision.models._utils import IntermediateLayerGetter as IntermediateLayerGetter


import torchvision


from itertools import product as product


from math import ceil


import math


from torch import nn


from collections import namedtuple


from torch.nn import AdaptiveAvgPool2d


from torch.nn import BatchNorm1d


from torch.nn import BatchNorm2d


from torch.nn import Conv2d


from torch.nn import Dropout


from torch.nn import Linear


from torch.nn import MaxPool2d


from torch.nn import Module


from torch.nn import PReLU


from torch.nn import ReLU


from torch.nn import Sequential


from torch.nn import Sigmoid


from torchvision.transforms.functional import normalize


from torch.hub import download_url_to_file


from torch.hub import get_dir


class AddCoordsTh(nn.Module):

    def __init__(self, x_dim=64, y_dim=64, with_r=False, with_boundary=False):
        super(AddCoordsTh, self).__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.with_r = with_r
        self.with_boundary = with_boundary

    def forward(self, input_tensor, heatmap=None):
        """
        input_tensor: (batch, c, x_dim, y_dim)
        """
        batch_size_tensor = input_tensor.shape[0]
        xx_ones = torch.ones([1, self.y_dim], dtype=torch.int32, device=input_tensor.device)
        xx_ones = xx_ones.unsqueeze(-1)
        xx_range = torch.arange(self.x_dim, dtype=torch.int32, device=input_tensor.device).unsqueeze(0)
        xx_range = xx_range.unsqueeze(1)
        xx_channel = torch.matmul(xx_ones.float(), xx_range.float())
        xx_channel = xx_channel.unsqueeze(-1)
        yy_ones = torch.ones([1, self.x_dim], dtype=torch.int32, device=input_tensor.device)
        yy_ones = yy_ones.unsqueeze(1)
        yy_range = torch.arange(self.y_dim, dtype=torch.int32, device=input_tensor.device).unsqueeze(0)
        yy_range = yy_range.unsqueeze(-1)
        yy_channel = torch.matmul(yy_range.float(), yy_ones.float())
        yy_channel = yy_channel.unsqueeze(-1)
        xx_channel = xx_channel.permute(0, 3, 2, 1)
        yy_channel = yy_channel.permute(0, 3, 2, 1)
        xx_channel = xx_channel / (self.x_dim - 1)
        yy_channel = yy_channel / (self.y_dim - 1)
        xx_channel = xx_channel * 2 - 1
        yy_channel = yy_channel * 2 - 1
        xx_channel = xx_channel.repeat(batch_size_tensor, 1, 1, 1)
        yy_channel = yy_channel.repeat(batch_size_tensor, 1, 1, 1)
        if self.with_boundary and heatmap is not None:
            boundary_channel = torch.clamp(heatmap[:, -1:, :, :], 0.0, 1.0)
            zero_tensor = torch.zeros_like(xx_channel)
            xx_boundary_channel = torch.where(boundary_channel > 0.05, xx_channel, zero_tensor)
            yy_boundary_channel = torch.where(boundary_channel > 0.05, yy_channel, zero_tensor)
        if self.with_boundary and heatmap is not None:
            xx_boundary_channel = xx_boundary_channel
            yy_boundary_channel = yy_boundary_channel
        ret = torch.cat([input_tensor, xx_channel, yy_channel], dim=1)
        if self.with_r:
            rr = torch.sqrt(torch.pow(xx_channel, 2) + torch.pow(yy_channel, 2))
            rr = rr / torch.max(rr)
            ret = torch.cat([ret, rr], dim=1)
        if self.with_boundary and heatmap is not None:
            ret = torch.cat([ret, xx_boundary_channel, yy_boundary_channel], dim=1)
        return ret


class CoordConvTh(nn.Module):
    """CoordConv layer as in the paper."""

    def __init__(self, x_dim, y_dim, with_r, with_boundary, in_channels, first_one=False, *args, **kwargs):
        super(CoordConvTh, self).__init__()
        self.addcoords = AddCoordsTh(x_dim=x_dim, y_dim=y_dim, with_r=with_r, with_boundary=with_boundary)
        in_channels += 2
        if with_r:
            in_channels += 1
        if with_boundary and not first_one:
            in_channels += 2
        self.conv = nn.Conv2d(*args, in_channels=in_channels, **kwargs)

    def forward(self, input_tensor, heatmap=None):
        ret = self.addcoords(input_tensor, heatmap)
        last_channel = ret[:, -2:, :, :]
        ret = self.conv(ret)
        return ret, last_channel


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):

    def __init__(self, in_chan, out_chan, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_chan, out_chan, stride)
        self.bn1 = nn.BatchNorm2d(out_chan)
        self.conv2 = conv3x3(out_chan, out_chan)
        self.bn2 = nn.BatchNorm2d(out_chan)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        if in_chan != out_chan or stride != 1:
            self.downsample = nn.Sequential(nn.Conv2d(in_chan, out_chan, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(out_chan))

    def forward(self, x):
        residual = self.conv1(x)
        residual = F.relu(self.bn1(residual))
        residual = self.conv2(residual)
        residual = self.bn2(residual)
        shortcut = x
        if self.downsample is not None:
            shortcut = self.downsample(x)
        out = shortcut + residual
        out = self.relu(out)
        return out


class ConvBlock(nn.Module):

    def __init__(self, in_planes, out_planes):
        super(ConvBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = conv3x3(in_planes, int(out_planes / 2))
        self.bn2 = nn.BatchNorm2d(int(out_planes / 2))
        self.conv2 = conv3x3(int(out_planes / 2), int(out_planes / 4), padding=1, dilation=1)
        self.bn3 = nn.BatchNorm2d(int(out_planes / 4))
        self.conv3 = conv3x3(int(out_planes / 4), int(out_planes / 4), padding=1, dilation=1)
        if in_planes != out_planes:
            self.downsample = nn.Sequential(nn.BatchNorm2d(in_planes), nn.ReLU(True), nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, bias=False))
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

    def __init__(self, num_modules, depth, num_features, first_one=False):
        super(HourGlass, self).__init__()
        self.num_modules = num_modules
        self.depth = depth
        self.features = num_features
        self.coordconv = CoordConvTh(x_dim=64, y_dim=64, with_r=True, with_boundary=True, in_channels=256, first_one=first_one, out_channels=256, kernel_size=1, stride=1, padding=0)
        self._generate_network(self.depth)

    def _generate_network(self, level):
        self.add_module('b1_' + str(level), ConvBlock(256, 256))
        self.add_module('b2_' + str(level), ConvBlock(256, 256))
        if level > 1:
            self._generate_network(level - 1)
        else:
            self.add_module('b2_plus_' + str(level), ConvBlock(256, 256))
        self.add_module('b3_' + str(level), ConvBlock(256, 256))

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
        up2 = F.interpolate(low3, scale_factor=2, mode='nearest')
        return up1 + up2

    def forward(self, x, heatmap):
        x, last_channel = self.coordconv(x, heatmap)
        return self._forward(self.depth, x), last_channel


def calculate_points(heatmaps):
    B, N, H, W = heatmaps.shape
    HW = H * W
    BN_range = np.arange(B * N)
    heatline = heatmaps.reshape(B, N, HW)
    indexes = np.argmax(heatline, axis=2)
    preds = np.stack((indexes % W, indexes // W), axis=2)
    preds = preds.astype(np.float, copy=False)
    inr = indexes.ravel()
    heatline = heatline.reshape(B * N, HW)
    x_up = heatline[BN_range, inr + 1]
    x_down = heatline[BN_range, inr - 1]
    if any(inr + W >= 4096):
        y_up = heatline[BN_range, 4095]
    else:
        y_up = heatline[BN_range, inr + W]
    if any(inr - W <= 0):
        y_down = heatline[BN_range, 0]
    else:
        y_down = heatline[BN_range, inr - W]
    think_diff = np.sign(np.stack((x_up - x_down, y_up - y_down), axis=1))
    think_diff *= 0.25
    preds += think_diff.reshape(B, N, 2)
    preds += 0.5
    return preds


class FAN(nn.Module):

    def __init__(self, num_modules=1, end_relu=False, gray_scale=False, num_landmarks=68, device='cuda'):
        super(FAN, self).__init__()
        self.device = device
        self.num_modules = num_modules
        self.gray_scale = gray_scale
        self.end_relu = end_relu
        self.num_landmarks = num_landmarks
        if self.gray_scale:
            self.conv1 = CoordConvTh(x_dim=256, y_dim=256, with_r=True, with_boundary=False, in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3)
        else:
            self.conv1 = CoordConvTh(x_dim=256, y_dim=256, with_r=True, with_boundary=False, in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = ConvBlock(64, 128)
        self.conv3 = ConvBlock(128, 128)
        self.conv4 = ConvBlock(128, 256)
        for hg_module in range(self.num_modules):
            if hg_module == 0:
                first_one = True
            else:
                first_one = False
            self.add_module('m' + str(hg_module), HourGlass(1, 4, 256, first_one))
            self.add_module('top_m_' + str(hg_module), ConvBlock(256, 256))
            self.add_module('conv_last' + str(hg_module), nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0))
            self.add_module('bn_end' + str(hg_module), nn.BatchNorm2d(256))
            self.add_module('l' + str(hg_module), nn.Conv2d(256, num_landmarks + 1, kernel_size=1, stride=1, padding=0))
            if hg_module < self.num_modules - 1:
                self.add_module('bl' + str(hg_module), nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0))
                self.add_module('al' + str(hg_module), nn.Conv2d(num_landmarks + 1, 256, kernel_size=1, stride=1, padding=0))

    def forward(self, x):
        x, _ = self.conv1(x)
        x = F.relu(self.bn1(x), True)
        x = F.avg_pool2d(self.conv2(x), 2, stride=2)
        x = self.conv3(x)
        x = self.conv4(x)
        previous = x
        outputs = []
        boundary_channels = []
        tmp_out = None
        for i in range(self.num_modules):
            hg, boundary_channel = self._modules['m' + str(i)](previous, tmp_out)
            ll = hg
            ll = self._modules['top_m_' + str(i)](ll)
            ll = F.relu(self._modules['bn_end' + str(i)](self._modules['conv_last' + str(i)](ll)), True)
            tmp_out = self._modules['l' + str(i)](ll)
            if self.end_relu:
                tmp_out = F.relu(tmp_out)
            outputs.append(tmp_out)
            boundary_channels.append(boundary_channel)
            if i < self.num_modules - 1:
                ll = self._modules['bl' + str(i)](ll)
                tmp_out_ = self._modules['al' + str(i)](tmp_out)
                previous = previous + ll + tmp_out_
        return outputs, boundary_channels

    def get_landmarks(self, img):
        H, W, _ = img.shape
        offset = W / 64, H / 64, 0, 0
        img = cv2.resize(img, (256, 256))
        inp = img[..., ::-1]
        inp = torch.from_numpy(np.ascontiguousarray(inp.transpose((2, 0, 1)))).float()
        inp = inp
        inp.div_(255.0).unsqueeze_(0)
        outputs, _ = self.forward(inp)
        out = outputs[-1][:, :-1, :, :]
        heatmaps = out.detach().cpu().numpy()
        pred = calculate_points(heatmaps).reshape(-1, 2)
        pred *= offset[:2]
        pred += offset[-2:]
        return pred


class Bottleneck(namedtuple('Block', ['in_channel', 'depth', 'stride'])):
    """A named tuple describing a ResNet block."""


class ResNetBackbone(nn.Module):

    def __init__(self, lda_out_channels, in_chn, block, layers, num_classes=1000):
        super(ResNetBackbone, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.lda1_pool = nn.Sequential(nn.Conv2d(256, 16, kernel_size=1, stride=1, padding=0, bias=False), nn.AvgPool2d(7, stride=7))
        self.lda1_fc = nn.Linear(16 * 64, lda_out_channels)
        self.lda2_pool = nn.Sequential(nn.Conv2d(512, 32, kernel_size=1, stride=1, padding=0, bias=False), nn.AvgPool2d(7, stride=7))
        self.lda2_fc = nn.Linear(32 * 16, lda_out_channels)
        self.lda3_pool = nn.Sequential(nn.Conv2d(1024, 64, kernel_size=1, stride=1, padding=0, bias=False), nn.AvgPool2d(7, stride=7))
        self.lda3_fc = nn.Linear(64 * 4, lda_out_channels)
        self.lda4_pool = nn.AvgPool2d(7, stride=7)
        self.lda4_fc = nn.Linear(2048, in_chn - lda_out_channels * 3)

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
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        lda_1 = self.lda1_fc(self.lda1_pool(x).view(x.size(0), -1))
        x = self.layer2(x)
        lda_2 = self.lda2_fc(self.lda2_pool(x).view(x.size(0), -1))
        x = self.layer3(x)
        lda_3 = self.lda3_fc(self.lda3_pool(x).view(x.size(0), -1))
        x = self.layer4(x)
        lda_4 = self.lda4_fc(self.lda4_pool(x).view(x.size(0), -1))
        vec = torch.cat((lda_1, lda_2, lda_3, lda_4), 1)
        out = {}
        out['hyper_in_feat'] = x
        out['target_in_vec'] = vec
        return out


def resnet50_backbone(lda_out_channels, in_chn, **kwargs):
    """Constructs a ResNet-50 model_hyper."""
    model = ResNetBackbone(lda_out_channels, in_chn, Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


class HyperNet(nn.Module):
    """
    Hyper network for learning perceptual rules.
    Args:
        lda_out_channels: local distortion aware module output size.
        hyper_in_channels: input feature channels for hyper network.
        target_in_size: input vector size for target network.
        target_fc(i)_size: fully connection layer size of target network.
        feature_size: input feature map width/height for hyper network.
    Note:
        For size match, input args must satisfy: 'target_fc(i)_size * target_fc(i+1)_size' is divisible by 'feature_size ^ 2'. # noqa E501
    """

    def __init__(self, lda_out_channels, hyper_in_channels, target_in_size, target_fc1_size, target_fc2_size, target_fc3_size, target_fc4_size, feature_size):
        super(HyperNet, self).__init__()
        self.hyperInChn = hyper_in_channels
        self.target_in_size = target_in_size
        self.f1 = target_fc1_size
        self.f2 = target_fc2_size
        self.f3 = target_fc3_size
        self.f4 = target_fc4_size
        self.feature_size = feature_size
        self.res = resnet50_backbone(lda_out_channels, target_in_size)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv1 = nn.Sequential(nn.Conv2d(2048, 1024, 1, padding=(0, 0)), nn.ReLU(inplace=True), nn.Conv2d(1024, 512, 1, padding=(0, 0)), nn.ReLU(inplace=True), nn.Conv2d(512, self.hyperInChn, 1, padding=(0, 0)), nn.ReLU(inplace=True))
        self.fc1w_conv = nn.Conv2d(self.hyperInChn, int(self.target_in_size * self.f1 / feature_size ** 2), 3, padding=(1, 1))
        self.fc1b_fc = nn.Linear(self.hyperInChn, self.f1)
        self.fc2w_conv = nn.Conv2d(self.hyperInChn, int(self.f1 * self.f2 / feature_size ** 2), 3, padding=(1, 1))
        self.fc2b_fc = nn.Linear(self.hyperInChn, self.f2)
        self.fc3w_conv = nn.Conv2d(self.hyperInChn, int(self.f2 * self.f3 / feature_size ** 2), 3, padding=(1, 1))
        self.fc3b_fc = nn.Linear(self.hyperInChn, self.f3)
        self.fc4w_conv = nn.Conv2d(self.hyperInChn, int(self.f3 * self.f4 / feature_size ** 2), 3, padding=(1, 1))
        self.fc4b_fc = nn.Linear(self.hyperInChn, self.f4)
        self.fc5w_fc = nn.Linear(self.hyperInChn, self.f4)
        self.fc5b_fc = nn.Linear(self.hyperInChn, 1)

    def forward(self, img):
        feature_size = self.feature_size
        res_out = self.res(img)
        target_in_vec = res_out['target_in_vec'].view(-1, self.target_in_size, 1, 1)
        hyper_in_feat = self.conv1(res_out['hyper_in_feat']).view(-1, self.hyperInChn, feature_size, feature_size)
        target_fc1w = self.fc1w_conv(hyper_in_feat).view(-1, self.f1, self.target_in_size, 1, 1)
        target_fc1b = self.fc1b_fc(self.pool(hyper_in_feat).squeeze()).view(-1, self.f1)
        target_fc2w = self.fc2w_conv(hyper_in_feat).view(-1, self.f2, self.f1, 1, 1)
        target_fc2b = self.fc2b_fc(self.pool(hyper_in_feat).squeeze()).view(-1, self.f2)
        target_fc3w = self.fc3w_conv(hyper_in_feat).view(-1, self.f3, self.f2, 1, 1)
        target_fc3b = self.fc3b_fc(self.pool(hyper_in_feat).squeeze()).view(-1, self.f3)
        target_fc4w = self.fc4w_conv(hyper_in_feat).view(-1, self.f4, self.f3, 1, 1)
        target_fc4b = self.fc4b_fc(self.pool(hyper_in_feat).squeeze()).view(-1, self.f4)
        target_fc5w = self.fc5w_fc(self.pool(hyper_in_feat).squeeze()).view(-1, 1, self.f4, 1, 1)
        target_fc5b = self.fc5b_fc(self.pool(hyper_in_feat).squeeze()).view(-1, 1)
        out = {}
        out['target_in_vec'] = target_in_vec
        out['target_fc1w'] = target_fc1w
        out['target_fc1b'] = target_fc1b
        out['target_fc2w'] = target_fc2w
        out['target_fc2b'] = target_fc2b
        out['target_fc3w'] = target_fc3w
        out['target_fc3b'] = target_fc3b
        out['target_fc4w'] = target_fc4w
        out['target_fc4b'] = target_fc4b
        out['target_fc5w'] = target_fc5w
        out['target_fc5b'] = target_fc5b
        return out


class TargetFC(nn.Module):
    """
    Fully connection operations for target net
    Note:
        Weights & biases are different for different images in a batch,
        thus here we use group convolution for calculating images in a batch with individual weights & biases.
    """

    def __init__(self, weight, bias):
        super(TargetFC, self).__init__()
        self.weight = weight
        self.bias = bias

    def forward(self, input_):
        input_re = input_.view(-1, input_.shape[0] * input_.shape[1], input_.shape[2], input_.shape[3])
        weight_re = self.weight.view(self.weight.shape[0] * self.weight.shape[1], self.weight.shape[2], self.weight.shape[3], self.weight.shape[4])
        bias_re = self.bias.view(self.bias.shape[0] * self.bias.shape[1])
        out = F.conv2d(input=input_re, weight=weight_re, bias=bias_re, groups=self.weight.shape[0])
        return out.view(input_.shape[0], self.weight.shape[1], input_.shape[2], input_.shape[3])


class TargetNet(nn.Module):
    """
    Target network for quality prediction.
    """

    def __init__(self, paras):
        super(TargetNet, self).__init__()
        self.l1 = nn.Sequential(TargetFC(paras['target_fc1w'], paras['target_fc1b']), nn.Sigmoid())
        self.l2 = nn.Sequential(TargetFC(paras['target_fc2w'], paras['target_fc2b']), nn.Sigmoid())
        self.l3 = nn.Sequential(TargetFC(paras['target_fc3w'], paras['target_fc3b']), nn.Sigmoid())
        self.l4 = nn.Sequential(TargetFC(paras['target_fc4w'], paras['target_fc4b']), nn.Sigmoid(), TargetFC(paras['target_fc5w'], paras['target_fc5b']))

    def forward(self, x):
        q = self.l1(x)
        q = self.l2(q)
        q = self.l3(q)
        q = self.l4(q).squeeze()
        return q


class HyperIQA(nn.Module):
    """
    Combine the hypernet and target network within a network.
    """

    def __init__(self, *args):
        super(HyperIQA, self).__init__()
        self.hypernet = HyperNet(*args)

    def forward(self, img):
        net_params = self.hypernet(img)
        target_net = TargetNet(net_params)
        for param in target_net.parameters():
            param.requires_grad = False
        pred = target_net(net_params['target_in_vec'])
        return pred


def conv_bn(inp, oup, stride):
    return nn.Sequential(nn.Conv2d(inp, oup, 3, stride, 1, bias=False), nn.BatchNorm2d(oup), nn.ReLU6(inplace=True))


def conv_bn1X1(inp, oup, stride, leaky=0):
    return nn.Sequential(nn.Conv2d(inp, oup, 1, stride, padding=0, bias=False), nn.BatchNorm2d(oup), nn.LeakyReLU(negative_slope=leaky, inplace=True))


class FPN(nn.Module):

    def __init__(self, in_channels_list, out_channels):
        super(FPN, self).__init__()
        leaky = 0
        if out_channels <= 64:
            leaky = 0.1
        self.output1 = conv_bn1X1(in_channels_list[0], out_channels, stride=1, leaky=leaky)
        self.output2 = conv_bn1X1(in_channels_list[1], out_channels, stride=1, leaky=leaky)
        self.output3 = conv_bn1X1(in_channels_list[2], out_channels, stride=1, leaky=leaky)
        self.merge1 = conv_bn(out_channels, out_channels, leaky=leaky)
        self.merge2 = conv_bn(out_channels, out_channels, leaky=leaky)

    def forward(self, input):
        output1 = self.output1(input[0])
        output2 = self.output2(input[1])
        output3 = self.output3(input[2])
        up3 = F.interpolate(output3, size=[output2.size(2), output2.size(3)], mode='nearest')
        output2 = output2 + up3
        output2 = self.merge2(output2)
        up2 = F.interpolate(output2, size=[output1.size(2), output1.size(3)], mode='nearest')
        output1 = output1 + up2
        output1 = self.merge1(output1)
        out = [output1, output2, output3]
        return out


def conv_dw(inp, oup, stride, leaky=0.1):
    return nn.Sequential(nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False), nn.BatchNorm2d(inp), nn.LeakyReLU(negative_slope=leaky, inplace=True), nn.Conv2d(inp, oup, 1, 1, 0, bias=False), nn.BatchNorm2d(oup), nn.LeakyReLU(negative_slope=leaky, inplace=True))


class MobileNetV1(nn.Module):

    def __init__(self):
        super(MobileNetV1, self).__init__()
        self.stage1 = nn.Sequential(conv_bn(3, 8, 2, leaky=0.1), conv_dw(8, 16, 1), conv_dw(16, 32, 2), conv_dw(32, 32, 1), conv_dw(32, 64, 2), conv_dw(64, 64, 1))
        self.stage2 = nn.Sequential(conv_dw(64, 128, 2), conv_dw(128, 128, 1), conv_dw(128, 128, 1), conv_dw(128, 128, 1), conv_dw(128, 128, 1), conv_dw(128, 128, 1))
        self.stage3 = nn.Sequential(conv_dw(128, 256, 2), conv_dw(256, 256, 1))
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, 1000)

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.avg(x)
        x = x.view(-1, 256)
        x = self.fc(x)
        return x


class PriorBox(object):

    def __init__(self, cfg, image_size=None, phase='train'):
        super(PriorBox, self).__init__()
        self.min_sizes = cfg['min_sizes']
        self.steps = cfg['steps']
        self.clip = cfg['clip']
        self.image_size = image_size
        self.feature_maps = [[ceil(self.image_size[0] / step), ceil(self.image_size[1] / step)] for step in self.steps]
        self.name = 's'

    def forward(self):
        anchors = []
        for k, f in enumerate(self.feature_maps):
            min_sizes = self.min_sizes[k]
            for i, j in product(range(f[0]), range(f[1])):
                for min_size in min_sizes:
                    s_kx = min_size / self.image_size[1]
                    s_ky = min_size / self.image_size[0]
                    dense_cx = [(x * self.steps[k] / self.image_size[1]) for x in [j + 0.5]]
                    dense_cy = [(y * self.steps[k] / self.image_size[0]) for y in [i + 0.5]]
                    for cy, cx in product(dense_cy, dense_cx):
                        anchors += [cx, cy, s_kx, s_ky]
        output = torch.Tensor(anchors).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output


def conv_bn_no_relu(inp, oup, stride):
    return nn.Sequential(nn.Conv2d(inp, oup, 3, stride, 1, bias=False), nn.BatchNorm2d(oup))


class SSH(nn.Module):

    def __init__(self, in_channel, out_channel):
        super(SSH, self).__init__()
        assert out_channel % 4 == 0
        leaky = 0
        if out_channel <= 64:
            leaky = 0.1
        self.conv3X3 = conv_bn_no_relu(in_channel, out_channel // 2, stride=1)
        self.conv5X5_1 = conv_bn(in_channel, out_channel // 4, stride=1, leaky=leaky)
        self.conv5X5_2 = conv_bn_no_relu(out_channel // 4, out_channel // 4, stride=1)
        self.conv7X7_2 = conv_bn(out_channel // 4, out_channel // 4, stride=1, leaky=leaky)
        self.conv7x7_3 = conv_bn_no_relu(out_channel // 4, out_channel // 4, stride=1)

    def forward(self, input):
        conv3X3 = self.conv3X3(input)
        conv5X5_1 = self.conv5X5_1(input)
        conv5X5 = self.conv5X5_2(conv5X5_1)
        conv7X7_2 = self.conv7X7_2(conv5X5_1)
        conv7X7 = self.conv7x7_3(conv7X7_2)
        out = torch.cat([conv3X3, conv5X5, conv7X7], dim=1)
        out = F.relu(out)
        return out


def batched_decode(b_loc, priors, variances):
    """Decode locations from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        b_loc (tensor): location predictions for loc layers,
            Shape: [num_batches,num_priors,4]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [1,num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded bounding box predictions
    """
    boxes = priors[:, :, :2] + b_loc[:, :, :2] * variances[0] * priors[:, :, 2:], priors[:, :, 2:] * torch.exp(b_loc[:, :, 2:] * variances[1])
    boxes = torch.cat(boxes, dim=2)
    boxes[:, :, :2] -= boxes[:, :, 2:] / 2
    boxes[:, :, 2:] += boxes[:, :, :2]
    return boxes


def batched_decode_landm(pre, priors, variances):
    """Decode landm from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        pre (tensor): landm predictions for loc layers,
            Shape: [num_batches,num_priors,10]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [1,num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded landm predictions
    """
    landms = priors[:, :, :2] + pre[:, :, :2] * variances[0] * priors[:, :, 2:], priors[:, :, :2] + pre[:, :, 2:4] * variances[0] * priors[:, :, 2:], priors[:, :, :2] + pre[:, :, 4:6] * variances[0] * priors[:, :, 2:], priors[:, :, :2] + pre[:, :, 6:8] * variances[0] * priors[:, :, 2:], priors[:, :, :2] + pre[:, :, 8:10] * variances[0] * priors[:, :, 2:]
    landms = torch.cat(landms, dim=2)
    return landms


def decode(loc, priors, variances):
    """Decode locations from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        loc (tensor): location predictions for loc layers,
            Shape: [num_priors,4]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded bounding box predictions
    """
    boxes = torch.cat((priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:], priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])), 1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes


def decode_landm(pre, priors, variances):
    """Decode landm from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        pre (tensor): landm predictions for loc layers,
            Shape: [num_priors,10]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded landm predictions
    """
    tmp = priors[:, :2] + pre[:, :2] * variances[0] * priors[:, 2:], priors[:, :2] + pre[:, 2:4] * variances[0] * priors[:, 2:], priors[:, :2] + pre[:, 4:6] * variances[0] * priors[:, 2:], priors[:, :2] + pre[:, 6:8] * variances[0] * priors[:, 2:], priors[:, :2] + pre[:, 8:10] * variances[0] * priors[:, 2:]
    landms = torch.cat(tmp, dim=1)
    return landms


def generate_config(network_name):
    cfg_mnet = {'name': 'mobilenet0.25', 'min_sizes': [[16, 32], [64, 128], [256, 512]], 'steps': [8, 16, 32], 'variance': [0.1, 0.2], 'clip': False, 'loc_weight': 2.0, 'gpu_train': True, 'batch_size': 32, 'ngpu': 1, 'epoch': 250, 'decay1': 190, 'decay2': 220, 'image_size': 640, 'return_layers': {'stage1': 1, 'stage2': 2, 'stage3': 3}, 'in_channel': 32, 'out_channel': 64}
    cfg_re50 = {'name': 'Resnet50', 'min_sizes': [[16, 32], [64, 128], [256, 512]], 'steps': [8, 16, 32], 'variance': [0.1, 0.2], 'clip': False, 'loc_weight': 2.0, 'gpu_train': True, 'batch_size': 24, 'ngpu': 4, 'epoch': 100, 'decay1': 70, 'decay2': 90, 'image_size': 840, 'return_layers': {'layer2': 1, 'layer3': 2, 'layer4': 3}, 'in_channel': 256, 'out_channel': 256}
    if network_name == 'mobile0.25':
        return cfg_mnet
    elif network_name == 'resnet50':
        return cfg_re50
    else:
        raise NotImplementedError(f'network_name={network_name}')


DEFAULT_CROP_SIZE = 96, 112


class FaceWarpException(Exception):

    def __str__(self):
        return 'In File {}:{}'.format(__file__, super.__str__(self))


REFERENCE_FACIAL_POINTS = [[30.29459953, 51.69630051], [65.53179932, 51.50139999], [48.02519989, 71.73660278], [33.54930115, 92.3655014], [62.72990036, 92.20410156]]


def get_reference_facial_points(output_size=None, inner_padding_factor=0.0, outer_padding=(0, 0), default_square=False):
    """
    Function:
    ----------
        get reference 5 key points according to crop settings:
        0. Set default crop_size:
            if default_square:
                crop_size = (112, 112)
            else:
                crop_size = (96, 112)
        1. Pad the crop_size by inner_padding_factor in each side;
        2. Resize crop_size into (output_size - outer_padding*2),
            pad into output_size with outer_padding;
        3. Output reference_5point;
    Parameters:
    ----------
        @output_size: (w, h) or None
            size of aligned face image
        @inner_padding_factor: (w_factor, h_factor)
            padding factor for inner (w, h)
        @outer_padding: (w_pad, h_pad)
            each row is a pair of coordinates (x, y)
        @default_square: True or False
            if True:
                default crop_size = (112, 112)
            else:
                default crop_size = (96, 112);
        !!! make sure, if output_size is not None:
                (output_size - outer_padding)
                = some_scale * (default crop_size * (1.0 +
                inner_padding_factor))
    Returns:
    ----------
        @reference_5point: 5x2 np.array
            each row is a pair of transformed coordinates (x, y)
    """
    tmp_5pts = np.array(REFERENCE_FACIAL_POINTS)
    tmp_crop_size = np.array(DEFAULT_CROP_SIZE)
    if default_square:
        size_diff = max(tmp_crop_size) - tmp_crop_size
        tmp_5pts += size_diff / 2
        tmp_crop_size += size_diff
    if output_size and output_size[0] == tmp_crop_size[0] and output_size[1] == tmp_crop_size[1]:
        return tmp_5pts
    if inner_padding_factor == 0 and outer_padding == (0, 0):
        if output_size is None:
            return tmp_5pts
        else:
            raise FaceWarpException('No paddings to do, output_size must be None or {}'.format(tmp_crop_size))
    if not 0 <= inner_padding_factor <= 1.0:
        raise FaceWarpException('Not (0 <= inner_padding_factor <= 1.0)')
    if (inner_padding_factor > 0 or outer_padding[0] > 0 or outer_padding[1] > 0) and output_size is None:
        output_size = tmp_crop_size * (1 + inner_padding_factor * 2).astype(np.int32)
        output_size += np.array(outer_padding)
    if not (outer_padding[0] < output_size[0] and outer_padding[1] < output_size[1]):
        raise FaceWarpException('Not (outer_padding[0] < output_size[0] and outer_padding[1] < output_size[1])')
    if inner_padding_factor > 0:
        size_diff = tmp_crop_size * inner_padding_factor * 2
        tmp_5pts += size_diff / 2
        tmp_crop_size += np.round(size_diff).astype(np.int32)
    size_bf_outer_pad = np.array(output_size) - np.array(outer_padding) * 2
    if size_bf_outer_pad[0] * tmp_crop_size[1] != size_bf_outer_pad[1] * tmp_crop_size[0]:
        raise FaceWarpException('Must have (output_size - outer_padding)= some_scale * (crop_size * (1.0 + inner_padding_factor)')
    scale_factor = size_bf_outer_pad[0].astype(np.float32) / tmp_crop_size[0]
    tmp_5pts = tmp_5pts * scale_factor
    tmp_crop_size = size_bf_outer_pad
    reference_5point = tmp_5pts + np.array(outer_padding)
    tmp_crop_size = output_size
    return reference_5point


class BboxHead(nn.Module):

    def __init__(self, inchannels=512, num_anchors=3):
        super(BboxHead, self).__init__()
        self.conv1x1 = nn.Conv2d(inchannels, num_anchors * 4, kernel_size=(1, 1), stride=1, padding=0)

    def forward(self, x):
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1).contiguous()
        return out.view(out.shape[0], -1, 4)


def make_bbox_head(fpn_num=3, inchannels=64, anchor_num=2):
    bboxhead = nn.ModuleList()
    for i in range(fpn_num):
        bboxhead.append(BboxHead(inchannels, anchor_num))
    return bboxhead


class ClassHead(nn.Module):

    def __init__(self, inchannels=512, num_anchors=3):
        super(ClassHead, self).__init__()
        self.num_anchors = num_anchors
        self.conv1x1 = nn.Conv2d(inchannels, self.num_anchors * 2, kernel_size=(1, 1), stride=1, padding=0)

    def forward(self, x):
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1).contiguous()
        return out.view(out.shape[0], -1, 2)


def make_class_head(fpn_num=3, inchannels=64, anchor_num=2):
    classhead = nn.ModuleList()
    for i in range(fpn_num):
        classhead.append(ClassHead(inchannels, anchor_num))
    return classhead


class LandmarkHead(nn.Module):

    def __init__(self, inchannels=512, num_anchors=3):
        super(LandmarkHead, self).__init__()
        self.conv1x1 = nn.Conv2d(inchannels, num_anchors * 10, kernel_size=(1, 1), stride=1, padding=0)

    def forward(self, x):
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1).contiguous()
        return out.view(out.shape[0], -1, 10)


def make_landmark_head(fpn_num=3, inchannels=64, anchor_num=2):
    landmarkhead = nn.ModuleList()
    for i in range(fpn_num):
        landmarkhead.append(LandmarkHead(inchannels, anchor_num))
    return landmarkhead


def py_cpu_nms(dets, thresh):
    """Pure Python NMS baseline."""
    keep = torchvision.ops.nms(boxes=torch.Tensor(dets[:, :4]), scores=torch.Tensor(dets[:, 4]), iou_threshold=thresh)
    return list(keep)


def get_affine_transform_matrix(src_pts, dst_pts):
    """
    Function:
    ----------
        get affine transform matrix 'tfm' from src_pts to dst_pts
    Parameters:
    ----------
        @src_pts: Kx2 np.array
            source points matrix, each row is a pair of coordinates (x, y)
        @dst_pts: Kx2 np.array
            destination points matrix, each row is a pair of coordinates (x, y)
    Returns:
    ----------
        @tfm: 2x3 np.array
            transform matrix from src_pts to dst_pts
    """
    tfm = np.float32([[1, 0, 0], [0, 1, 0]])
    n_pts = src_pts.shape[0]
    ones = np.ones((n_pts, 1), src_pts.dtype)
    src_pts_ = np.hstack([src_pts, ones])
    dst_pts_ = np.hstack([dst_pts, ones])
    A, res, rank, s = np.linalg.lstsq(src_pts_, dst_pts_)
    if rank == 3:
        tfm = np.float32([[A[0, 0], A[1, 0], A[2, 0]], [A[0, 1], A[1, 1], A[2, 1]]])
    elif rank == 2:
        tfm = np.float32([[A[0, 0], A[1, 0], 0], [A[0, 1], A[1, 1], 0]])
    return tfm


def cvt_tform_mat_for_cv2(trans):
    """
    Function:
    ----------
        Convert Transform Matrix 'trans' into 'cv2_trans' which could be
        directly used by cv2.warpAffine():
            u = src_pts[:, 0]
            v = src_pts[:, 1]
            x = dst_pts[:, 0]
            y = dst_pts[:, 1]
            [x, y].T = cv_trans * [u, v, 1].T

    Parameters:
    ----------
        @trans: 3x3 np.array
            transform matrix from uv to xy

    Returns:
    ----------
        @cv2_trans: 2x3 np.array
            transform matrix from src_pts to dst_pts, could be directly used
            for cv2.warpAffine()
    """
    cv2_trans = trans[:, 0:2].T
    return cv2_trans


def findNonreflectiveSimilarity(uv, xy, options=None):
    options = {'K': 2}
    K = options['K']
    M = xy.shape[0]
    x = xy[:, 0].reshape((-1, 1))
    y = xy[:, 1].reshape((-1, 1))
    tmp1 = np.hstack((x, y, np.ones((M, 1)), np.zeros((M, 1))))
    tmp2 = np.hstack((y, -x, np.zeros((M, 1)), np.ones((M, 1))))
    X = np.vstack((tmp1, tmp2))
    u = uv[:, 0].reshape((-1, 1))
    v = uv[:, 1].reshape((-1, 1))
    U = np.vstack((u, v))
    if rank(X) >= 2 * K:
        r, _, _, _ = lstsq(X, U, rcond=-1)
        r = np.squeeze(r)
    else:
        raise Exception('cp2tform:twoUniquePointsReq')
    sc = r[0]
    ss = r[1]
    tx = r[2]
    ty = r[3]
    Tinv = np.array([[sc, -ss, 0], [ss, sc, 0], [tx, ty, 1]])
    T = inv(Tinv)
    T[:, 2] = np.array([0, 0, 1])
    return T, Tinv


def tformfwd(trans, uv):
    """
    Function:
    ----------
        apply affine transform 'trans' to uv

    Parameters:
    ----------
        @trans: 3x3 np.array
            transform matrix
        @uv: Kx2 np.array
            each row is a pair of coordinates (x, y)

    Returns:
    ----------
        @xy: Kx2 np.array
            each row is a pair of transformed coordinates (x, y)
    """
    uv = np.hstack((uv, np.ones((uv.shape[0], 1))))
    xy = np.dot(uv, trans)
    xy = xy[:, 0:-1]
    return xy


def findSimilarity(uv, xy, options=None):
    options = {'K': 2}
    trans1, trans1_inv = findNonreflectiveSimilarity(uv, xy, options)
    xyR = xy
    xyR[:, 0] = -1 * xyR[:, 0]
    trans2r, trans2r_inv = findNonreflectiveSimilarity(uv, xyR, options)
    TreflectY = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]])
    trans2 = np.dot(trans2r, TreflectY)
    xy1 = tformfwd(trans1, uv)
    norm1 = norm(xy1 - xy)
    xy2 = tformfwd(trans2, uv)
    norm2 = norm(xy2 - xy)
    if norm1 <= norm2:
        return trans1, trans1_inv
    else:
        trans2_inv = inv(trans2)
        return trans2, trans2_inv


def get_similarity_transform(src_pts, dst_pts, reflective=True):
    """
    Function:
    ----------
        Find Similarity Transform Matrix 'trans':
            u = src_pts[:, 0]
            v = src_pts[:, 1]
            x = dst_pts[:, 0]
            y = dst_pts[:, 1]
            [x, y, 1] = [u, v, 1] * trans

    Parameters:
    ----------
        @src_pts: Kx2 np.array
            source points, each row is a pair of coordinates (x, y)
        @dst_pts: Kx2 np.array
            destination points, each row is a pair of transformed
            coordinates (x, y)
        @reflective: True or False
            if True:
                use reflective similarity transform
            else:
                use non-reflective similarity transform

    Returns:
    ----------
       @trans: 3x3 np.array
            transform matrix from uv to xy
        trans_inv: 3x3 np.array
            inverse of trans, transform matrix from xy to uv
    """
    if reflective:
        trans, trans_inv = findSimilarity(src_pts, dst_pts)
    else:
        trans, trans_inv = findNonreflectiveSimilarity(src_pts, dst_pts)
    return trans, trans_inv


def get_similarity_transform_for_cv2(src_pts, dst_pts, reflective=True):
    """
    Function:
    ----------
        Find Similarity Transform Matrix 'cv2_trans' which could be
        directly used by cv2.warpAffine():
            u = src_pts[:, 0]
            v = src_pts[:, 1]
            x = dst_pts[:, 0]
            y = dst_pts[:, 1]
            [x, y].T = cv_trans * [u, v, 1].T

    Parameters:
    ----------
        @src_pts: Kx2 np.array
            source points, each row is a pair of coordinates (x, y)
        @dst_pts: Kx2 np.array
            destination points, each row is a pair of transformed
            coordinates (x, y)
        reflective: True or False
            if True:
                use reflective similarity transform
            else:
                use non-reflective similarity transform

    Returns:
    ----------
        @cv2_trans: 2x3 np.array
            transform matrix from src_pts to dst_pts, could be directly used
            for cv2.warpAffine()
    """
    trans, trans_inv = get_similarity_transform(src_pts, dst_pts, reflective)
    cv2_trans = cvt_tform_mat_for_cv2(trans)
    return cv2_trans


def warp_and_crop_face(src_img, facial_pts, reference_pts=None, crop_size=(96, 112), align_type='smilarity'):
    """
    Function:
    ----------
        apply affine transform 'trans' to uv
    Parameters:
    ----------
        @src_img: 3x3 np.array
            input image
        @facial_pts: could be
            1)a list of K coordinates (x,y)
        or
            2) Kx2 or 2xK np.array
            each row or col is a pair of coordinates (x, y)
        @reference_pts: could be
            1) a list of K coordinates (x,y)
        or
            2) Kx2 or 2xK np.array
            each row or col is a pair of coordinates (x, y)
        or
            3) None
            if None, use default reference facial points
        @crop_size: (w, h)
            output face image size
        @align_type: transform type, could be one of
            1) 'similarity': use similarity transform
            2) 'cv2_affine': use the first 3 points to do affine transform,
                    by calling cv2.getAffineTransform()
            3) 'affine': use all points to do affine transform
    Returns:
    ----------
        @face_img: output face image with size (w, h) = @crop_size
    """
    if reference_pts is None:
        if crop_size[0] == 96 and crop_size[1] == 112:
            reference_pts = REFERENCE_FACIAL_POINTS
        else:
            default_square = False
            inner_padding_factor = 0
            outer_padding = 0, 0
            output_size = crop_size
            reference_pts = get_reference_facial_points(output_size, inner_padding_factor, outer_padding, default_square)
    ref_pts = np.float32(reference_pts)
    ref_pts_shp = ref_pts.shape
    if max(ref_pts_shp) < 3 or min(ref_pts_shp) != 2:
        raise FaceWarpException('reference_pts.shape must be (K,2) or (2,K) and K>2')
    if ref_pts_shp[0] == 2:
        ref_pts = ref_pts.T
    src_pts = np.float32(facial_pts)
    src_pts_shp = src_pts.shape
    if max(src_pts_shp) < 3 or min(src_pts_shp) != 2:
        raise FaceWarpException('facial_pts.shape must be (K,2) or (2,K) and K>2')
    if src_pts_shp[0] == 2:
        src_pts = src_pts.T
    if src_pts.shape != ref_pts.shape:
        raise FaceWarpException('facial_pts and reference_pts must have the same shape')
    if align_type == 'cv2_affine':
        tfm = cv2.getAffineTransform(src_pts[0:3], ref_pts[0:3])
    elif align_type == 'affine':
        tfm = get_affine_transform_matrix(src_pts, ref_pts)
    else:
        tfm = get_similarity_transform_for_cv2(src_pts, ref_pts)
    face_img = cv2.warpAffine(src_img, tfm, (crop_size[0], crop_size[1]))
    return face_img


class RetinaFace(nn.Module):

    def __init__(self, network_name='resnet50', half=False, phase='test', device=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device
        super(RetinaFace, self).__init__()
        self.half_inference = half
        cfg = generate_config(network_name)
        self.backbone = cfg['name']
        self.model_name = f'retinaface_{network_name}'
        self.cfg = cfg
        self.phase = phase
        self.target_size, self.max_size = 1600, 2150
        self.resize, self.scale, self.scale1 = 1.0, None, None
        self.mean_tensor = torch.tensor([[[[104.0]], [[117.0]], [[123.0]]]], device=self.device)
        self.reference = get_reference_facial_points(default_square=True)
        backbone = None
        if cfg['name'] == 'mobilenet0.25':
            backbone = MobileNetV1()
            self.body = IntermediateLayerGetter(backbone, cfg['return_layers'])
        elif cfg['name'] == 'Resnet50':
            import torchvision.models as models
            backbone = models.resnet50(pretrained=False)
            self.body = IntermediateLayerGetter(backbone, cfg['return_layers'])
        in_channels_stage2 = cfg['in_channel']
        in_channels_list = [in_channels_stage2 * 2, in_channels_stage2 * 4, in_channels_stage2 * 8]
        out_channels = cfg['out_channel']
        self.fpn = FPN(in_channels_list, out_channels)
        self.ssh1 = SSH(out_channels, out_channels)
        self.ssh2 = SSH(out_channels, out_channels)
        self.ssh3 = SSH(out_channels, out_channels)
        self.ClassHead = make_class_head(fpn_num=3, inchannels=cfg['out_channel'])
        self.BboxHead = make_bbox_head(fpn_num=3, inchannels=cfg['out_channel'])
        self.LandmarkHead = make_landmark_head(fpn_num=3, inchannels=cfg['out_channel'])
        self
        self.eval()
        if self.half_inference:
            self.half()

    def forward(self, inputs):
        out = self.body(inputs)
        if self.backbone == 'mobilenet0.25' or self.backbone == 'Resnet50':
            out = list(out.values())
        fpn = self.fpn(out)
        feature1 = self.ssh1(fpn[0])
        feature2 = self.ssh2(fpn[1])
        feature3 = self.ssh3(fpn[2])
        features = [feature1, feature2, feature3]
        bbox_regressions = torch.cat([self.BboxHead[i](feature) for i, feature in enumerate(features)], dim=1)
        classifications = torch.cat([self.ClassHead[i](feature) for i, feature in enumerate(features)], dim=1)
        tmp = [self.LandmarkHead[i](feature) for i, feature in enumerate(features)]
        ldm_regressions = torch.cat(tmp, dim=1)
        if self.phase == 'train':
            output = bbox_regressions, classifications, ldm_regressions
        else:
            output = bbox_regressions, F.softmax(classifications, dim=-1), ldm_regressions
        return output

    def __detect_faces(self, inputs):
        height, width = inputs.shape[2:]
        self.scale = torch.tensor([width, height, width, height], dtype=torch.float32, device=self.device)
        tmp = [width, height, width, height, width, height, width, height, width, height]
        self.scale1 = torch.tensor(tmp, dtype=torch.float32, device=self.device)
        inputs = inputs
        if self.half_inference:
            inputs = inputs.half()
        loc, conf, landmarks = self(inputs)
        priorbox = PriorBox(self.cfg, image_size=inputs.shape[2:])
        priors = priorbox.forward()
        return loc, conf, landmarks, priors

    def transform(self, image, use_origin_size):
        if isinstance(image, Image.Image):
            image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        image = image.astype(np.float32)
        im_size_min = np.min(image.shape[0:2])
        im_size_max = np.max(image.shape[0:2])
        resize = float(self.target_size) / float(im_size_min)
        if np.round(resize * im_size_max) > self.max_size:
            resize = float(self.max_size) / float(im_size_max)
        resize = 1 if use_origin_size else resize
        if resize != 1:
            image = cv2.resize(image, None, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR)
        image = image.transpose(2, 0, 1)
        image = torch.from_numpy(image).unsqueeze(0)
        return image, resize

    def detect_faces(self, image, conf_threshold=0.8, nms_threshold=0.4, use_origin_size=True):
        image, self.resize = self.transform(image, use_origin_size)
        image = image
        if self.half_inference:
            image = image.half()
        image = image - self.mean_tensor
        loc, conf, landmarks, priors = self.__detect_faces(image)
        boxes = decode(loc.data.squeeze(0), priors.data, self.cfg['variance'])
        boxes = boxes * self.scale / self.resize
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
        landmarks = decode_landm(landmarks.squeeze(0), priors, self.cfg['variance'])
        landmarks = landmarks * self.scale1 / self.resize
        landmarks = landmarks.cpu().numpy()
        inds = np.where(scores > conf_threshold)[0]
        boxes, landmarks, scores = boxes[inds], landmarks[inds], scores[inds]
        order = scores.argsort()[::-1]
        boxes, landmarks, scores = boxes[order], landmarks[order], scores[order]
        bounding_boxes = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(bounding_boxes, nms_threshold)
        bounding_boxes, landmarks = bounding_boxes[keep, :], landmarks[keep]
        return np.concatenate((bounding_boxes, landmarks), axis=1)

    def __align_multi(self, image, boxes, landmarks, limit=None):
        if len(boxes) < 1:
            return [], []
        if limit:
            boxes = boxes[:limit]
            landmarks = landmarks[:limit]
        faces = []
        for landmark in landmarks:
            facial5points = [[landmark[2 * j], landmark[2 * j + 1]] for j in range(5)]
            warped_face = warp_and_crop_face(np.array(image), facial5points, self.reference, crop_size=(112, 112))
            faces.append(warped_face)
        return np.concatenate((boxes, landmarks), axis=1), faces

    def align_multi(self, img, conf_threshold=0.8, limit=None):
        rlt = self.detect_faces(img, conf_threshold=conf_threshold)
        boxes, landmarks = rlt[:, 0:5], rlt[:, 5:]
        return self.__align_multi(img, boxes, landmarks, limit)

    def batched_transform(self, frames, use_origin_size):
        """
        Arguments:
            frames: a list of PIL.Image, or torch.Tensor(shape=[n, h, w, c],
                type=np.float32, BGR format).
            use_origin_size: whether to use origin size.
        """
        from_PIL = True if isinstance(frames[0], Image.Image) else False
        if from_PIL:
            frames = [cv2.cvtColor(np.asarray(frame), cv2.COLOR_RGB2BGR) for frame in frames]
            frames = np.asarray(frames, dtype=np.float32)
        im_size_min = np.min(frames[0].shape[0:2])
        im_size_max = np.max(frames[0].shape[0:2])
        resize = float(self.target_size) / float(im_size_min)
        if np.round(resize * im_size_max) > self.max_size:
            resize = float(self.max_size) / float(im_size_max)
        resize = 1 if use_origin_size else resize
        if resize != 1:
            if not from_PIL:
                frames = F.interpolate(frames, scale_factor=resize)
            else:
                frames = [cv2.resize(frame, None, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR) for frame in frames]
        if not from_PIL:
            frames = frames.transpose(1, 2).transpose(1, 3).contiguous()
        else:
            frames = frames.transpose((0, 3, 1, 2))
            frames = torch.from_numpy(frames)
        return frames, resize

    def batched_detect_faces(self, frames, conf_threshold=0.8, nms_threshold=0.4, use_origin_size=True):
        """
        Arguments:
            frames: a list of PIL.Image, or np.array(shape=[n, h, w, c],
                type=np.uint8, BGR format).
            conf_threshold: confidence threshold.
            nms_threshold: nms threshold.
            use_origin_size: whether to use origin size.
        Returns:
            final_bounding_boxes: list of np.array ([n_boxes, 5],
                type=np.float32).
            final_landmarks: list of np.array ([n_boxes, 10], type=np.float32).
        """
        frames, self.resize = self.batched_transform(frames, use_origin_size)
        frames = frames
        frames = frames - self.mean_tensor
        b_loc, b_conf, b_landmarks, priors = self.__detect_faces(frames)
        final_bounding_boxes, final_landmarks = [], []
        priors = priors.unsqueeze(0)
        b_loc = batched_decode(b_loc, priors, self.cfg['variance']) * self.scale / self.resize
        b_landmarks = batched_decode_landm(b_landmarks, priors, self.cfg['variance']) * self.scale1 / self.resize
        b_conf = b_conf[:, :, 1]
        b_indice = b_conf > conf_threshold
        b_loc_and_conf = torch.cat((b_loc, b_conf.unsqueeze(-1)), dim=2).float()
        for pred, landm, inds in zip(b_loc_and_conf, b_landmarks, b_indice):
            pred, landm = pred[inds, :], landm[inds, :]
            if pred.shape[0] == 0:
                final_bounding_boxes.append(np.array([], dtype=np.float32))
                final_landmarks.append(np.array([], dtype=np.float32))
                continue
            bounding_boxes, landm = pred.cpu().numpy(), landm.cpu().numpy()
            keep = py_cpu_nms(bounding_boxes, nms_threshold)
            bounding_boxes, landmarks = bounding_boxes[keep, :], landm[keep]
            final_bounding_boxes.append(bounding_boxes)
            final_landmarks.append(landmarks)
        return final_bounding_boxes, final_landmarks


class HopeNet(nn.Module):

    def __init__(self, block, layers, num_bins):
        super(HopeNet, self).__init__()
        if block == 'resnet':
            block = torchvision.models.resnet.Bottleneck
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7)
        self.fc_yaw = nn.Linear(512 * block.expansion, num_bins)
        self.fc_pitch = nn.Linear(512 * block.expansion, num_bins)
        self.fc_roll = nn.Linear(512 * block.expansion, num_bins)
        self.idx_tensor = torch.arange(66).float()

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

    @staticmethod
    def softmax_temperature(tensor, temperature):
        result = torch.exp(tensor / temperature)
        result = torch.div(result, torch.sum(result, 1).unsqueeze(1).expand_as(result))
        return result

    def bin2degree(self, predict):
        predict = self.softmax_temperature(predict, 1)
        return torch.sum(predict * self.idx_tensor.type_as(predict), 1) * 3 - 99

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        pre_yaw = self.fc_yaw(x)
        pre_pitch = self.fc_pitch(x)
        pre_roll = self.fc_roll(x)
        yaw = self.bin2degree(pre_yaw)
        pitch = self.bin2degree(pre_pitch)
        roll = self.bin2degree(pre_roll)
        return yaw, pitch, roll


class BaseBackbone(nn.Module):
    """ Superclass of Replaceable Backbone Model for Semantic Estimation
    """

    def __init__(self, in_channels):
        super(BaseBackbone, self).__init__()
        self.in_channels = in_channels
        self.model = None
        self.enc_channels = []

    def forward(self, x):
        raise NotImplementedError

    def load_pretrained_ckpt(self):
        raise NotImplementedError


class InvertedResidual(nn.Module):

    def __init__(self, inp, oup, stride, expansion, dilation=1):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
        hidden_dim = round(inp * expansion)
        self.use_res_connect = self.stride == 1 and inp == oup
        if expansion == 1:
            self.conv = nn.Sequential(nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, dilation=dilation, bias=False), nn.BatchNorm2d(hidden_dim), nn.ReLU6(inplace=True), nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False), nn.BatchNorm2d(oup))
        else:
            self.conv = nn.Sequential(nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False), nn.BatchNorm2d(hidden_dim), nn.ReLU6(inplace=True), nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, dilation=dilation, bias=False), nn.BatchNorm2d(hidden_dim), nn.ReLU6(inplace=True), nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False), nn.BatchNorm2d(oup))

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def conv_1x1_bn(inp, oup):
    return nn.Sequential(nn.Conv2d(inp, oup, 1, 1, 0, bias=False), nn.BatchNorm2d(oup), nn.ReLU6(inplace=True))


class MobileNetV2(nn.Module):

    def __init__(self, in_channels, alpha=1.0, expansion=6, num_classes=1000):
        super(MobileNetV2, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        input_channel = 32
        last_channel = 1280
        interverted_residual_setting = [[1, 16, 1, 1], [expansion, 24, 2, 2], [expansion, 32, 3, 2], [expansion, 64, 4, 2], [expansion, 96, 3, 1], [expansion, 160, 3, 2], [expansion, 320, 1, 1]]
        input_channel = _make_divisible(input_channel * alpha, 8)
        self.last_channel = _make_divisible(last_channel * alpha, 8) if alpha > 1.0 else last_channel
        self.features = [conv_bn(self.in_channels, input_channel, 2)]
        for t, c, n, s in interverted_residual_setting:
            output_channel = _make_divisible(int(c * alpha), 8)
            for i in range(n):
                if i == 0:
                    self.features.append(InvertedResidual(input_channel, output_channel, s, expansion=t))
                else:
                    self.features.append(InvertedResidual(input_channel, output_channel, 1, expansion=t))
                input_channel = output_channel
        self.features.append(conv_1x1_bn(input_channel, self.last_channel))
        self.features = nn.Sequential(*self.features)
        if self.num_classes is not None:
            self.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(self.last_channel, num_classes))
        self._init_weights()

    def forward(self, x):
        x = self.features[0](x)
        x = self.features[1](x)
        x = self.features[2](x)
        x = self.features[3](x)
        x = self.features[4](x)
        x = self.features[5](x)
        x = self.features[6](x)
        x = self.features[7](x)
        x = self.features[8](x)
        x = self.features[9](x)
        x = self.features[10](x)
        x = self.features[11](x)
        x = self.features[12](x)
        x = self.features[13](x)
        x = self.features[14](x)
        x = self.features[15](x)
        x = self.features[16](x)
        x = self.features[17](x)
        x = self.features[18](x)
        if self.num_classes is not None:
            x = x.mean(dim=(2, 3))
            x = self.classifier(x)
        return x

    def _load_pretrained_model(self, pretrained_file):
        pretrain_dict = torch.load(pretrained_file, map_location='cpu')
        model_dict = {}
        state_dict = self.state_dict()
        None
        for k, v in pretrain_dict.items():
            if k in state_dict:
                model_dict[k] = v
            else:
                None
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


class MobileNetV2Backbone(BaseBackbone):
    """ MobileNetV2 Backbone
    """

    def __init__(self, in_channels):
        super(MobileNetV2Backbone, self).__init__(in_channels)
        self.model = MobileNetV2(self.in_channels, alpha=1.0, expansion=6, num_classes=None)
        self.enc_channels = [16, 24, 32, 96, 1280]

    def forward(self, x):
        x = self.model.features[0](x)
        x = self.model.features[1](x)
        enc2x = x
        x = self.model.features[2](x)
        x = self.model.features[3](x)
        enc4x = x
        x = self.model.features[4](x)
        x = self.model.features[5](x)
        x = self.model.features[6](x)
        enc8x = x
        x = self.model.features[7](x)
        x = self.model.features[8](x)
        x = self.model.features[9](x)
        x = self.model.features[10](x)
        x = self.model.features[11](x)
        x = self.model.features[12](x)
        x = self.model.features[13](x)
        enc16x = x
        x = self.model.features[14](x)
        x = self.model.features[15](x)
        x = self.model.features[16](x)
        x = self.model.features[17](x)
        x = self.model.features[18](x)
        enc32x = x
        return [enc2x, enc4x, enc8x, enc16x, enc32x]

    def load_pretrained_ckpt(self):
        ckpt_path = './pretrained/mobilenetv2_human_seg.ckpt'
        if not os.path.exists(ckpt_path):
            None
            exit()
        ckpt = torch.load(ckpt_path)
        self.model.load_state_dict(ckpt)


class IBNorm(nn.Module):
    """ Combine Instance Norm and Batch Norm into One Layer
    """

    def __init__(self, in_channels):
        super(IBNorm, self).__init__()
        in_channels = in_channels
        self.bnorm_channels = int(in_channels / 2)
        self.inorm_channels = in_channels - self.bnorm_channels
        self.bnorm = nn.BatchNorm2d(self.bnorm_channels, affine=True)
        self.inorm = nn.InstanceNorm2d(self.inorm_channels, affine=False)

    def forward(self, x):
        bn_x = self.bnorm(x[:, :self.bnorm_channels, ...].contiguous())
        in_x = self.inorm(x[:, self.bnorm_channels:, ...].contiguous())
        return torch.cat((bn_x, in_x), 1)


class Conv2dIBNormRelu(nn.Module):
    """ Convolution + IBNorm + ReLu
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, with_ibn=True, with_relu=True):
        super(Conv2dIBNormRelu, self).__init__()
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)]
        if with_ibn:
            layers.append(IBNorm(out_channels))
        if with_relu:
            layers.append(nn.ReLU(inplace=True))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class SEBlock(nn.Module):
    """ SE Block Proposed in https://arxiv.org/pdf/1709.01507.pdf
    """

    def __init__(self, in_channels, out_channels, reduction=1):
        super(SEBlock, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(in_channels, int(in_channels // reduction), bias=False), nn.ReLU(inplace=True), nn.Linear(int(in_channels // reduction), out_channels, bias=False), nn.Sigmoid())

    def forward(self, x):
        b, c, _, _ = x.size()
        w = self.pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)
        return x * w.expand_as(x)


class LRBranch(nn.Module):
    """ Low Resolution Branch of MODNet
    """

    def __init__(self, backbone):
        super(LRBranch, self).__init__()
        enc_channels = backbone.enc_channels
        self.backbone = backbone
        self.se_block = SEBlock(enc_channels[4], enc_channels[4], reduction=4)
        self.conv_lr16x = Conv2dIBNormRelu(enc_channels[4], enc_channels[3], 5, stride=1, padding=2)
        self.conv_lr8x = Conv2dIBNormRelu(enc_channels[3], enc_channels[2], 5, stride=1, padding=2)
        self.conv_lr = Conv2dIBNormRelu(enc_channels[2], 1, kernel_size=3, stride=2, padding=1, with_ibn=False, with_relu=False)

    def forward(self, img, inference):
        enc_features = self.backbone.forward(img)
        enc2x, enc4x, enc32x = enc_features[0], enc_features[1], enc_features[4]
        enc32x = self.se_block(enc32x)
        lr16x = F.interpolate(enc32x, scale_factor=2, mode='bilinear', align_corners=False)
        lr16x = self.conv_lr16x(lr16x)
        lr8x = F.interpolate(lr16x, scale_factor=2, mode='bilinear', align_corners=False)
        lr8x = self.conv_lr8x(lr8x)
        pred_semantic = None
        if not inference:
            lr = self.conv_lr(lr8x)
            pred_semantic = torch.sigmoid(lr)
        return pred_semantic, lr8x, [enc2x, enc4x]


class HRBranch(nn.Module):
    """ High Resolution Branch of MODNet
    """

    def __init__(self, hr_channels, enc_channels):
        super(HRBranch, self).__init__()
        self.tohr_enc2x = Conv2dIBNormRelu(enc_channels[0], hr_channels, 1, stride=1, padding=0)
        self.conv_enc2x = Conv2dIBNormRelu(hr_channels + 3, hr_channels, 3, stride=2, padding=1)
        self.tohr_enc4x = Conv2dIBNormRelu(enc_channels[1], hr_channels, 1, stride=1, padding=0)
        self.conv_enc4x = Conv2dIBNormRelu(2 * hr_channels, 2 * hr_channels, 3, stride=1, padding=1)
        self.conv_hr4x = nn.Sequential(Conv2dIBNormRelu(3 * hr_channels + 3, 2 * hr_channels, 3, stride=1, padding=1), Conv2dIBNormRelu(2 * hr_channels, 2 * hr_channels, 3, stride=1, padding=1), Conv2dIBNormRelu(2 * hr_channels, hr_channels, 3, stride=1, padding=1))
        self.conv_hr2x = nn.Sequential(Conv2dIBNormRelu(2 * hr_channels, 2 * hr_channels, 3, stride=1, padding=1), Conv2dIBNormRelu(2 * hr_channels, hr_channels, 3, stride=1, padding=1), Conv2dIBNormRelu(hr_channels, hr_channels, 3, stride=1, padding=1), Conv2dIBNormRelu(hr_channels, hr_channels, 3, stride=1, padding=1))
        self.conv_hr = nn.Sequential(Conv2dIBNormRelu(hr_channels + 3, hr_channels, 3, stride=1, padding=1), Conv2dIBNormRelu(hr_channels, 1, kernel_size=1, stride=1, padding=0, with_ibn=False, with_relu=False))

    def forward(self, img, enc2x, enc4x, lr8x, inference):
        img2x = F.interpolate(img, scale_factor=1 / 2, mode='bilinear', align_corners=False)
        img4x = F.interpolate(img, scale_factor=1 / 4, mode='bilinear', align_corners=False)
        enc2x = self.tohr_enc2x(enc2x)
        hr4x = self.conv_enc2x(torch.cat((img2x, enc2x), dim=1))
        enc4x = self.tohr_enc4x(enc4x)
        hr4x = self.conv_enc4x(torch.cat((hr4x, enc4x), dim=1))
        lr4x = F.interpolate(lr8x, scale_factor=2, mode='bilinear', align_corners=False)
        hr4x = self.conv_hr4x(torch.cat((hr4x, lr4x, img4x), dim=1))
        hr2x = F.interpolate(hr4x, scale_factor=2, mode='bilinear', align_corners=False)
        hr2x = self.conv_hr2x(torch.cat((hr2x, enc2x), dim=1))
        pred_detail = None
        if not inference:
            hr = F.interpolate(hr2x, scale_factor=2, mode='bilinear', align_corners=False)
            hr = self.conv_hr(torch.cat((hr, img), dim=1))
            pred_detail = torch.sigmoid(hr)
        return pred_detail, hr2x


class FusionBranch(nn.Module):
    """ Fusion Branch of MODNet
    """

    def __init__(self, hr_channels, enc_channels):
        super(FusionBranch, self).__init__()
        self.conv_lr4x = Conv2dIBNormRelu(enc_channels[2], hr_channels, 5, stride=1, padding=2)
        self.conv_f2x = Conv2dIBNormRelu(2 * hr_channels, hr_channels, 3, stride=1, padding=1)
        self.conv_f = nn.Sequential(Conv2dIBNormRelu(hr_channels + 3, int(hr_channels / 2), 3, stride=1, padding=1), Conv2dIBNormRelu(int(hr_channels / 2), 1, 1, stride=1, padding=0, with_ibn=False, with_relu=False))

    def forward(self, img, lr8x, hr2x):
        lr4x = F.interpolate(lr8x, scale_factor=2, mode='bilinear', align_corners=False)
        lr4x = self.conv_lr4x(lr4x)
        lr2x = F.interpolate(lr4x, scale_factor=2, mode='bilinear', align_corners=False)
        f2x = self.conv_f2x(torch.cat((lr2x, hr2x), dim=1))
        f = F.interpolate(f2x, scale_factor=2, mode='bilinear', align_corners=False)
        f = self.conv_f(torch.cat((f, img), dim=1))
        pred_matte = torch.sigmoid(f)
        return pred_matte


class MODNet(nn.Module):
    """ Architecture of MODNet
    """

    def __init__(self, in_channels=3, hr_channels=32, backbone_pretrained=True):
        super(MODNet, self).__init__()
        self.in_channels = in_channels
        self.hr_channels = hr_channels
        self.backbone_pretrained = backbone_pretrained
        self.backbone = MobileNetV2Backbone(self.in_channels)
        self.lr_branch = LRBranch(self.backbone)
        self.hr_branch = HRBranch(self.hr_channels, self.backbone.enc_channels)
        self.f_branch = FusionBranch(self.hr_channels, self.backbone.enc_channels)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                self._init_conv(m)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.InstanceNorm2d):
                self._init_norm(m)
        if self.backbone_pretrained:
            self.backbone.load_pretrained_ckpt()

    def forward(self, img, inference):
        pred_semantic, lr8x, [enc2x, enc4x] = self.lr_branch(img, inference)
        pred_detail, hr2x = self.hr_branch(img, enc2x, enc4x, lr8x, inference)
        pred_matte = self.f_branch(img, lr8x, hr2x)
        return pred_semantic, pred_detail, pred_matte

    def freeze_norm(self):
        norm_types = [nn.BatchNorm2d, nn.InstanceNorm2d]
        for m in self.modules():
            for n in norm_types:
                if isinstance(m, n):
                    m.eval()
                    continue

    def _init_conv(self, conv):
        nn.init.kaiming_uniform_(conv.weight, a=0, mode='fan_in', nonlinearity='relu')
        if conv.bias is not None:
            nn.init.constant_(conv.bias, 0)

    def _init_norm(self, norm):
        if norm.weight is not None:
            nn.init.constant_(norm.weight, 1)
            nn.init.constant_(norm.bias, 0)


class ConvBNReLU(nn.Module):

    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_chan, out_chan, kernel_size=ks, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_chan)

    def forward(self, x):
        x = self.conv(x)
        x = F.relu(self.bn(x))
        return x


class BiSeNetOutput(nn.Module):

    def __init__(self, in_chan, mid_chan, num_class):
        super(BiSeNetOutput, self).__init__()
        self.conv = ConvBNReLU(in_chan, mid_chan, ks=3, stride=1, padding=1)
        self.conv_out = nn.Conv2d(mid_chan, num_class, kernel_size=1, bias=False)

    def forward(self, x):
        feat = self.conv(x)
        out = self.conv_out(feat)
        return out, feat


class AttentionRefinementModule(nn.Module):

    def __init__(self, in_chan, out_chan):
        super(AttentionRefinementModule, self).__init__()
        self.conv = ConvBNReLU(in_chan, out_chan, ks=3, stride=1, padding=1)
        self.conv_atten = nn.Conv2d(out_chan, out_chan, kernel_size=1, bias=False)
        self.bn_atten = nn.BatchNorm2d(out_chan)
        self.sigmoid_atten = nn.Sigmoid()

    def forward(self, x):
        feat = self.conv(x)
        atten = F.avg_pool2d(feat, feat.size()[2:])
        atten = self.conv_atten(atten)
        atten = self.bn_atten(atten)
        atten = self.sigmoid_atten(atten)
        out = torch.mul(feat, atten)
        return out


def create_layer_basic(in_chan, out_chan, bnum, stride=1):
    layers = [BasicBlock(in_chan, out_chan, stride=stride)]
    for i in range(bnum - 1):
        layers.append(BasicBlock(out_chan, out_chan, stride=1))
    return nn.Sequential(*layers)


class ResNet18(nn.Module):

    def __init__(self):
        super(ResNet18, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = create_layer_basic(64, 64, bnum=2, stride=1)
        self.layer2 = create_layer_basic(64, 128, bnum=2, stride=2)
        self.layer3 = create_layer_basic(128, 256, bnum=2, stride=2)
        self.layer4 = create_layer_basic(256, 512, bnum=2, stride=2)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.maxpool(x)
        x = self.layer1(x)
        feat8 = self.layer2(x)
        feat16 = self.layer3(feat8)
        feat32 = self.layer4(feat16)
        return feat8, feat16, feat32


class ContextPath(nn.Module):

    def __init__(self):
        super(ContextPath, self).__init__()
        self.resnet = ResNet18()
        self.arm16 = AttentionRefinementModule(256, 128)
        self.arm32 = AttentionRefinementModule(512, 128)
        self.conv_head32 = ConvBNReLU(128, 128, ks=3, stride=1, padding=1)
        self.conv_head16 = ConvBNReLU(128, 128, ks=3, stride=1, padding=1)
        self.conv_avg = ConvBNReLU(512, 128, ks=1, stride=1, padding=0)

    def forward(self, x):
        feat8, feat16, feat32 = self.resnet(x)
        h8, w8 = feat8.size()[2:]
        h16, w16 = feat16.size()[2:]
        h32, w32 = feat32.size()[2:]
        avg = F.avg_pool2d(feat32, feat32.size()[2:])
        avg = self.conv_avg(avg)
        avg_up = F.interpolate(avg, (h32, w32), mode='nearest')
        feat32_arm = self.arm32(feat32)
        feat32_sum = feat32_arm + avg_up
        feat32_up = F.interpolate(feat32_sum, (h16, w16), mode='nearest')
        feat32_up = self.conv_head32(feat32_up)
        feat16_arm = self.arm16(feat16)
        feat16_sum = feat16_arm + feat32_up
        feat16_up = F.interpolate(feat16_sum, (h8, w8), mode='nearest')
        feat16_up = self.conv_head16(feat16_up)
        return feat8, feat16_up, feat32_up


class FeatureFusionModule(nn.Module):

    def __init__(self, in_chan, out_chan):
        super(FeatureFusionModule, self).__init__()
        self.convblk = ConvBNReLU(in_chan, out_chan, ks=1, stride=1, padding=0)
        self.conv1 = nn.Conv2d(out_chan, out_chan // 4, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv2 = nn.Conv2d(out_chan // 4, out_chan, kernel_size=1, stride=1, padding=0, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, fsp, fcp):
        fcat = torch.cat([fsp, fcp], dim=1)
        feat = self.convblk(fcat)
        atten = F.avg_pool2d(feat, feat.size()[2:])
        atten = self.conv1(atten)
        atten = self.relu(atten)
        atten = self.conv2(atten)
        atten = self.sigmoid(atten)
        feat_atten = torch.mul(feat, atten)
        feat_out = feat_atten + feat
        return feat_out


class BiSeNet(nn.Module):

    def __init__(self, num_class):
        super(BiSeNet, self).__init__()
        self.cp = ContextPath()
        self.ffm = FeatureFusionModule(256, 256)
        self.conv_out = BiSeNetOutput(256, 256, num_class)
        self.conv_out16 = BiSeNetOutput(128, 64, num_class)
        self.conv_out32 = BiSeNetOutput(128, 64, num_class)

    def forward(self, x, return_feat=False):
        h, w = x.size()[2:]
        feat_res8, feat_cp8, feat_cp16 = self.cp(x)
        feat_sp = feat_res8
        feat_fuse = self.ffm(feat_sp, feat_cp8)
        out, feat = self.conv_out(feat_fuse)
        out16, feat16 = self.conv_out16(feat_cp8)
        out32, feat32 = self.conv_out32(feat_cp16)
        out = F.interpolate(out, (h, w), mode='bilinear', align_corners=True)
        out16 = F.interpolate(out16, (h, w), mode='bilinear', align_corners=True)
        out32 = F.interpolate(out32, (h, w), mode='bilinear', align_corners=True)
        if return_feat:
            feat = F.interpolate(feat, (h, w), mode='bilinear', align_corners=True)
            feat16 = F.interpolate(feat16, (h, w), mode='bilinear', align_corners=True)
            feat32 = F.interpolate(feat32, (h, w), mode='bilinear', align_corners=True)
            return out, out16, out32, feat, feat16, feat32
        else:
            return out, out16, out32


class NormLayer(nn.Module):
    """Normalization Layers.

    Args:
        channels: input channels, for batch norm and instance norm.
        input_size: input shape without batch size, for layer norm.
    """

    def __init__(self, channels, normalize_shape=None, norm_type='bn'):
        super(NormLayer, self).__init__()
        norm_type = norm_type.lower()
        self.norm_type = norm_type
        if norm_type == 'bn':
            self.norm = nn.BatchNorm2d(channels, affine=True)
        elif norm_type == 'in':
            self.norm = nn.InstanceNorm2d(channels, affine=False)
        elif norm_type == 'gn':
            self.norm = nn.GroupNorm(32, channels, affine=True)
        elif norm_type == 'pixel':
            self.norm = lambda x: F.normalize(x, p=2, dim=1)
        elif norm_type == 'layer':
            self.norm = nn.LayerNorm(normalize_shape)
        elif norm_type == 'none':
            self.norm = lambda x: x * 1.0
        else:
            assert 1 == 0, f'Norm type {norm_type} not support.'

    def forward(self, x, ref=None):
        if self.norm_type == 'spade':
            return self.norm(x, ref)
        else:
            return self.norm(x)


class ReluLayer(nn.Module):
    """Relu Layer.

    Args:
        relu type: type of relu layer, candidates are
            - ReLU
            - LeakyReLU: default relu slope 0.2
            - PRelu
            - SELU
            - none: direct pass
    """

    def __init__(self, channels, relu_type='relu'):
        super(ReluLayer, self).__init__()
        relu_type = relu_type.lower()
        if relu_type == 'relu':
            self.func = nn.ReLU(True)
        elif relu_type == 'leakyrelu':
            self.func = nn.LeakyReLU(0.2, inplace=True)
        elif relu_type == 'prelu':
            self.func = nn.PReLU(channels)
        elif relu_type == 'selu':
            self.func = nn.SELU(True)
        elif relu_type == 'none':
            self.func = lambda x: x * 1.0
        else:
            assert 1 == 0, f'Relu type {relu_type} not support.'

    def forward(self, x):
        return self.func(x)


class ConvLayer(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, scale='none', norm_type='none', relu_type='none', use_pad=True, bias=True):
        super(ConvLayer, self).__init__()
        self.use_pad = use_pad
        self.norm_type = norm_type
        if norm_type in ['bn']:
            bias = False
        stride = 2 if scale == 'down' else 1
        self.scale_func = lambda x: x
        if scale == 'up':
            self.scale_func = lambda x: nn.functional.interpolate(x, scale_factor=2, mode='nearest')
        self.reflection_pad = nn.ReflectionPad2d(int(np.ceil((kernel_size - 1.0) / 2)))
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, bias=bias)
        self.relu = ReluLayer(out_channels, relu_type)
        self.norm = NormLayer(out_channels, norm_type=norm_type)

    def forward(self, x):
        out = self.scale_func(x)
        if self.use_pad:
            out = self.reflection_pad(out)
        out = self.conv2d(out)
        out = self.norm(out)
        out = self.relu(out)
        return out


class ResidualBlock(nn.Module):
    """
    Residual block recommended in: http://torch.ch/blog/2016/02/04/resnets.html
    """

    def __init__(self, c_in, c_out, relu_type='prelu', norm_type='bn', scale='none'):
        super(ResidualBlock, self).__init__()
        if scale == 'none' and c_in == c_out:
            self.shortcut_func = lambda x: x
        else:
            self.shortcut_func = ConvLayer(c_in, c_out, 3, scale)
        scale_config_dict = {'down': ['none', 'down'], 'up': ['up', 'none'], 'none': ['none', 'none']}
        scale_conf = scale_config_dict[scale]
        self.conv1 = ConvLayer(c_in, c_out, 3, scale_conf[0], norm_type=norm_type, relu_type=relu_type)
        self.conv2 = ConvLayer(c_out, c_out, 3, scale_conf[1], norm_type=norm_type, relu_type='none')

    def forward(self, x):
        identity = self.shortcut_func(x)
        res = self.conv1(x)
        res = self.conv2(res)
        return identity + res


class ParseNet(nn.Module):

    def __init__(self, in_size=128, out_size=128, min_feat_size=32, base_ch=64, parsing_ch=19, res_depth=10, relu_type='LeakyReLU', norm_type='bn', ch_range=[32, 256]):
        super().__init__()
        self.res_depth = res_depth
        act_args = {'norm_type': norm_type, 'relu_type': relu_type}
        min_ch, max_ch = ch_range
        ch_clip = lambda x: max(min_ch, min(x, max_ch))
        min_feat_size = min(in_size, min_feat_size)
        down_steps = int(np.log2(in_size // min_feat_size))
        up_steps = int(np.log2(out_size // min_feat_size))
        self.encoder = []
        self.encoder.append(ConvLayer(3, base_ch, 3, 1))
        head_ch = base_ch
        for i in range(down_steps):
            cin, cout = ch_clip(head_ch), ch_clip(head_ch * 2)
            self.encoder.append(ResidualBlock(cin, cout, scale='down', **act_args))
            head_ch = head_ch * 2
        self.body = []
        for i in range(res_depth):
            self.body.append(ResidualBlock(ch_clip(head_ch), ch_clip(head_ch), **act_args))
        self.decoder = []
        for i in range(up_steps):
            cin, cout = ch_clip(head_ch), ch_clip(head_ch // 2)
            self.decoder.append(ResidualBlock(cin, cout, scale='up', **act_args))
            head_ch = head_ch // 2
        self.encoder = nn.Sequential(*self.encoder)
        self.body = nn.Sequential(*self.body)
        self.decoder = nn.Sequential(*self.decoder)
        self.out_img_conv = ConvLayer(ch_clip(head_ch), 3)
        self.out_mask_conv = ConvLayer(ch_clip(head_ch), parsing_ch)

    def forward(self, x):
        feat = self.encoder(x)
        x = feat + self.body(feat)
        x = self.decoder(x)
        out_img = self.out_img_conv(x)
        out_mask = self.out_mask_conv(x)
        return out_mask, out_img


class Flatten(Module):

    def forward(self, input):
        return input.view(input.size(0), -1)


class SEModule(Module):

    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = AdaptiveAvgPool2d(1)
        self.fc1 = Conv2d(channels, channels // reduction, kernel_size=1, padding=0, bias=False)
        self.relu = ReLU(inplace=True)
        self.fc2 = Conv2d(channels // reduction, channels, kernel_size=1, padding=0, bias=False)
        self.sigmoid = Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x


class bottleneck_IR(Module):

    def __init__(self, in_channel, depth, stride):
        super(bottleneck_IR, self).__init__()
        if in_channel == depth:
            self.shortcut_layer = MaxPool2d(1, stride)
        else:
            self.shortcut_layer = Sequential(Conv2d(in_channel, depth, (1, 1), stride, bias=False), BatchNorm2d(depth))
        self.res_layer = Sequential(BatchNorm2d(in_channel), Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False), PReLU(depth), Conv2d(depth, depth, (3, 3), stride, 1, bias=False), BatchNorm2d(depth))

    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)
        return res + shortcut


class bottleneck_IR_SE(Module):

    def __init__(self, in_channel, depth, stride):
        super(bottleneck_IR_SE, self).__init__()
        if in_channel == depth:
            self.shortcut_layer = MaxPool2d(1, stride)
        else:
            self.shortcut_layer = Sequential(Conv2d(in_channel, depth, (1, 1), stride, bias=False), BatchNorm2d(depth))
        self.res_layer = Sequential(BatchNorm2d(in_channel), Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False), PReLU(depth), Conv2d(depth, depth, (3, 3), stride, 1, bias=False), BatchNorm2d(depth), SEModule(depth, 16))

    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)
        return res + shortcut


def get_block(in_channel, depth, num_units, stride=2):
    return [Bottleneck(in_channel, depth, stride)] + [Bottleneck(depth, depth, 1) for i in range(num_units - 1)]


def get_blocks(num_layers):
    if num_layers == 50:
        blocks = [get_block(in_channel=64, depth=64, num_units=3), get_block(in_channel=64, depth=128, num_units=4), get_block(in_channel=128, depth=256, num_units=14), get_block(in_channel=256, depth=512, num_units=3)]
    elif num_layers == 100:
        blocks = [get_block(in_channel=64, depth=64, num_units=3), get_block(in_channel=64, depth=128, num_units=13), get_block(in_channel=128, depth=256, num_units=30), get_block(in_channel=256, depth=512, num_units=3)]
    elif num_layers == 152:
        blocks = [get_block(in_channel=64, depth=64, num_units=3), get_block(in_channel=64, depth=128, num_units=8), get_block(in_channel=128, depth=256, num_units=36), get_block(in_channel=256, depth=512, num_units=3)]
    return blocks


def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output


class Backbone(Module):

    def __init__(self, num_layers, drop_ratio, mode='ir'):
        super(Backbone, self).__init__()
        assert num_layers in [50, 100, 152], 'num_layers should be 50,100, or 152'
        assert mode in ['ir', 'ir_se'], 'mode should be ir or ir_se'
        blocks = get_blocks(num_layers)
        if mode == 'ir':
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE
        self.input_layer = Sequential(Conv2d(3, 64, (3, 3), 1, 1, bias=False), BatchNorm2d(64), PReLU(64))
        self.output_layer = Sequential(BatchNorm2d(512), Dropout(drop_ratio), Flatten(), Linear(512 * 7 * 7, 512), BatchNorm1d(512))
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(unit_module(bottleneck.in_channel, bottleneck.depth, bottleneck.stride))
        self.body = Sequential(*modules)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.body(x)
        x = self.output_layer(x)
        return l2_norm(x)


class Conv_block(Module):

    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        super(Conv_block, self).__init__()
        self.conv = Conv2d(in_c, out_channels=out_c, kernel_size=kernel, groups=groups, stride=stride, padding=padding, bias=False)
        self.bn = BatchNorm2d(out_c)
        self.prelu = PReLU(out_c)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.prelu(x)
        return x


class Linear_block(Module):

    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        super(Linear_block, self).__init__()
        self.conv = Conv2d(in_c, out_channels=out_c, kernel_size=kernel, groups=groups, stride=stride, padding=padding, bias=False)
        self.bn = BatchNorm2d(out_c)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class Depth_Wise(Module):

    def __init__(self, in_c, out_c, residual=False, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=1):
        super(Depth_Wise, self).__init__()
        self.conv = Conv_block(in_c, out_c=groups, kernel=(1, 1), padding=(0, 0), stride=(1, 1))
        self.conv_dw = Conv_block(groups, groups, groups=groups, kernel=kernel, padding=padding, stride=stride)
        self.project = Linear_block(groups, out_c, kernel=(1, 1), padding=(0, 0), stride=(1, 1))
        self.residual = residual

    def forward(self, x):
        if self.residual:
            short_cut = x
        x = self.conv(x)
        x = self.conv_dw(x)
        x = self.project(x)
        if self.residual:
            output = short_cut + x
        else:
            output = x
        return output


class Residual(Module):

    def __init__(self, c, num_block, groups, kernel=(3, 3), stride=(1, 1), padding=(1, 1)):
        super(Residual, self).__init__()
        modules = []
        for _ in range(num_block):
            modules.append(Depth_Wise(c, c, residual=True, kernel=kernel, padding=padding, stride=stride, groups=groups))
        self.model = Sequential(*modules)

    def forward(self, x):
        return self.model(x)


class MobileFaceNet(Module):

    def __init__(self, embedding_size):
        super(MobileFaceNet, self).__init__()
        self.conv1 = Conv_block(3, 64, kernel=(3, 3), stride=(2, 2), padding=(1, 1))
        self.conv2_dw = Conv_block(64, 64, kernel=(3, 3), stride=(1, 1), padding=(1, 1), groups=64)
        self.conv_23 = Depth_Wise(64, 64, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=128)
        self.conv_3 = Residual(64, num_block=4, groups=128, kernel=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_34 = Depth_Wise(64, 128, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=256)
        self.conv_4 = Residual(128, num_block=6, groups=256, kernel=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_45 = Depth_Wise(128, 128, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=512)
        self.conv_5 = Residual(128, num_block=2, groups=256, kernel=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_6_sep = Conv_block(128, 512, kernel=(1, 1), stride=(1, 1), padding=(0, 0))
        self.conv_6_dw = Linear_block(512, 512, groups=512, kernel=(7, 7), stride=(1, 1), padding=(0, 0))
        self.conv_6_flatten = Flatten()
        self.linear = Linear(512, embedding_size, bias=False)
        self.bn = BatchNorm1d(embedding_size)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2_dw(out)
        out = self.conv_23(out)
        out = self.conv_3(out)
        out = self.conv_34(out)
        out = self.conv_4(out)
        out = self.conv_45(out)
        out = self.conv_5(out)
        out = self.conv_6_sep(out)
        out = self.conv_6_dw(out)
        out = self.conv_6_flatten(out)
        out = self.linear(out)
        out = self.bn(out)
        return l2_norm(out)


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (AttentionRefinementModule,
     lambda: ([], {'in_chan': 4, 'out_chan': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (BasicBlock,
     lambda: ([], {'in_chan': 4, 'out_chan': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (BboxHead,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 512, 64, 64])], {})),
    (BiSeNet,
     lambda: ([], {'num_class': 4}),
     lambda: ([torch.rand([4, 3, 64, 64])], {})),
    (BiSeNetOutput,
     lambda: ([], {'in_chan': 4, 'mid_chan': 4, 'num_class': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ClassHead,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 512, 64, 64])], {})),
    (ContextPath,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {})),
    (ConvBNReLU,
     lambda: ([], {'in_chan': 4, 'out_chan': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ConvLayer,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Conv_block,
     lambda: ([], {'in_c': 4, 'out_c': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Depth_Wise,
     lambda: ([], {'in_c': 4, 'out_c': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (FeatureFusionModule,
     lambda: ([], {'in_chan': 4, 'out_chan': 4}),
     lambda: ([torch.rand([4, 1, 4, 4]), torch.rand([4, 3, 4, 4])], {})),
    (Flatten,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (IBNorm,
     lambda: ([], {'in_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (InvertedResidual,
     lambda: ([], {'inp': 4, 'oup': 4, 'stride': 1, 'expansion': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (LandmarkHead,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 512, 64, 64])], {})),
    (Linear_block,
     lambda: ([], {'in_c': 4, 'out_c': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (MobileNetV2,
     lambda: ([], {'in_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (MobileNetV2Backbone,
     lambda: ([], {'in_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (NormLayer,
     lambda: ([], {'channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ParseNet,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {})),
    (ReluLayer,
     lambda: ([], {'channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ResNet18,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {})),
    (Residual,
     lambda: ([], {'c': 4, 'num_block': 4, 'groups': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ResidualBlock,
     lambda: ([], {'c_in': 4, 'c_out': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (SEBlock,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (SEModule,
     lambda: ([], {'channels': 4, 'reduction': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (bottleneck_IR,
     lambda: ([], {'in_channel': 4, 'depth': 1, 'stride': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
]

