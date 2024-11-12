
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


from torchvision.models.vgg import vgg16


import numpy as np


import torch.utils.data as data


import random


import matplotlib.pyplot as plt


import torch.utils.model_zoo as model_zoo


import time


import torchvision


import torch.optim


class L_color(nn.Module):

    def __init__(self):
        super(L_color, self).__init__()
        self.eps = 0

    def forward(self, x):
        mean_rgb = torch.mean(x, [2, 3], keepdim=True)
        mr, mg, mb = torch.split(mean_rgb, 1, dim=1)
        Drg = torch.pow(mr - mg, 2)
        Drb = torch.pow(mr - mb, 2)
        Dgb = torch.pow(mb - mg, 2)
        k = torch.pow(torch.pow(Drg, 2) + torch.pow(Drb, 2) + torch.pow(Dgb, 2) + self.eps, 0.5)
        return k


class L_spa(nn.Module):

    def __init__(self):
        super(L_spa, self).__init__()
        kernel_left = torch.FloatTensor([[0, 0, 0], [-1, 1, 0], [0, 0, 0]]).unsqueeze(0).unsqueeze(0)
        kernel_right = torch.FloatTensor([[0, 0, 0], [0, 1, -1], [0, 0, 0]]).unsqueeze(0).unsqueeze(0)
        kernel_up = torch.FloatTensor([[0, -1, 0], [0, 1, 0], [0, 0, 0]]).unsqueeze(0).unsqueeze(0)
        kernel_down = torch.FloatTensor([[0, 0, 0], [0, 1, 0], [0, -1, 0]]).unsqueeze(0).unsqueeze(0)
        self.weight_left = nn.Parameter(data=kernel_left, requires_grad=False)
        self.weight_right = nn.Parameter(data=kernel_right, requires_grad=False)
        self.weight_up = nn.Parameter(data=kernel_up, requires_grad=False)
        self.weight_down = nn.Parameter(data=kernel_down, requires_grad=False)
        self.pool = nn.AvgPool2d(4)

    def forward(self, org, enhance):
        b, c, h, w = org.shape
        org_mean = torch.mean(org, 1, keepdim=True)
        enhance_mean = torch.mean(enhance, 1, keepdim=True)
        org_pool = self.pool(org_mean)
        enhance_pool = self.pool(enhance_mean)
        D_org_letf = F.conv2d(org_pool, self.weight_left, padding=1)
        D_org_right = F.conv2d(org_pool, self.weight_right, padding=1)
        D_org_up = F.conv2d(org_pool, self.weight_up, padding=1)
        D_org_down = F.conv2d(org_pool, self.weight_down, padding=1)
        D_enhance_letf = F.conv2d(enhance_pool, self.weight_left, padding=1)
        D_enhance_right = F.conv2d(enhance_pool, self.weight_right, padding=1)
        D_enhance_up = F.conv2d(enhance_pool, self.weight_up, padding=1)
        D_enhance_down = F.conv2d(enhance_pool, self.weight_down, padding=1)
        D_left = torch.pow(D_org_letf - D_enhance_letf, 2)
        D_right = torch.pow(D_org_right - D_enhance_right, 2)
        D_up = torch.pow(D_org_up - D_enhance_up, 2)
        D_down = torch.pow(D_org_down - D_enhance_down, 2)
        E = D_left + D_right + D_up + D_down
        return E


class L_spa8(nn.Module):

    def __init__(self, patch_size):
        super(L_spa8, self).__init__()
        kernel_left = torch.FloatTensor([[0, 0, 0], [-1, 1, 0], [0, 0, 0]]).unsqueeze(0).unsqueeze(0)
        kernel_right = torch.FloatTensor([[0, 0, 0], [0, 1, -1], [0, 0, 0]]).unsqueeze(0).unsqueeze(0)
        kernel_up = torch.FloatTensor([[0, -1, 0], [0, 1, 0], [0, 0, 0]]).unsqueeze(0).unsqueeze(0)
        kernel_down = torch.FloatTensor([[0, 0, 0], [0, 1, 0], [0, -1, 0]]).unsqueeze(0).unsqueeze(0)
        kernel_upleft = torch.FloatTensor([[-1, 0, 0], [0, 1, 0], [0, 0, 0]]).unsqueeze(0).unsqueeze(0)
        kernel_upright = torch.FloatTensor([[0, 0, -1], [0, 1, 0], [0, 0, 0]]).unsqueeze(0).unsqueeze(0)
        kernel_loleft = torch.FloatTensor([[0, 0, 0], [0, 1, 0], [-1, 0, 0]]).unsqueeze(0).unsqueeze(0)
        kernel_loright = torch.FloatTensor([[0, 0, 0], [0, 1, 0], [0, 0, -1]]).unsqueeze(0).unsqueeze(0)
        self.weight_left = nn.Parameter(data=kernel_left, requires_grad=False)
        self.weight_right = nn.Parameter(data=kernel_right, requires_grad=False)
        self.weight_up = nn.Parameter(data=kernel_up, requires_grad=False)
        self.weight_down = nn.Parameter(data=kernel_down, requires_grad=False)
        self.weight_upleft = nn.Parameter(data=kernel_upleft, requires_grad=False)
        self.weight_upright = nn.Parameter(data=kernel_upright, requires_grad=False)
        self.weight_loleft = nn.Parameter(data=kernel_loleft, requires_grad=False)
        self.weight_loright = nn.Parameter(data=kernel_loright, requires_grad=False)
        self.pool = nn.AvgPool2d(patch_size)

    def forward(self, org, enhance):
        org_mean = torch.mean(org, 1, keepdim=True)
        enhance_mean = torch.mean(enhance, 1, keepdim=True)
        org_pool = self.pool(org_mean)
        enhance_pool = self.pool(enhance_mean)
        D_org_letf = F.conv2d(org_pool, self.weight_left, padding=1)
        D_org_right = F.conv2d(org_pool, self.weight_right, padding=1)
        D_org_up = F.conv2d(org_pool, self.weight_up, padding=1)
        D_org_down = F.conv2d(org_pool, self.weight_down, padding=1)
        D_org_upleft = F.conv2d(org_pool, self.weight_upleft, padding=1)
        D_org_upright = F.conv2d(org_pool, self.weight_upright, padding=1)
        D_org_loleft = F.conv2d(org_pool, self.weight_loleft, padding=1)
        D_org_loright = F.conv2d(org_pool, self.weight_loright, padding=1)
        D_enhance_letf = F.conv2d(enhance_pool, self.weight_left, padding=1)
        D_enhance_right = F.conv2d(enhance_pool, self.weight_right, padding=1)
        D_enhance_up = F.conv2d(enhance_pool, self.weight_up, padding=1)
        D_enhance_down = F.conv2d(enhance_pool, self.weight_down, padding=1)
        D_enhance_upleft = F.conv2d(enhance_pool, self.weight_upleft, padding=1)
        D_enhance_upright = F.conv2d(enhance_pool, self.weight_upright, padding=1)
        D_enhance_loleft = F.conv2d(enhance_pool, self.weight_loleft, padding=1)
        D_enhance_loright = F.conv2d(enhance_pool, self.weight_loright, padding=1)
        D_left = torch.pow(D_org_letf - D_enhance_letf, 2)
        D_right = torch.pow(D_org_right - D_enhance_right, 2)
        D_up = torch.pow(D_org_up - D_enhance_up, 2)
        D_down = torch.pow(D_org_down - D_enhance_down, 2)
        D_upleft = torch.pow(D_org_upleft - D_enhance_upleft, 2)
        D_upright = torch.pow(D_org_upright - D_enhance_upright, 2)
        D_loleft = torch.pow(D_org_loleft - D_enhance_loleft, 2)
        D_loright = torch.pow(D_org_loright - D_enhance_loright, 2)
        E = D_left + D_right + D_up + D_down + 0.5 * (D_upleft + D_upright + D_loleft + D_loright)
        return E


class L_exp(nn.Module):

    def __init__(self, patch_size):
        super(L_exp, self).__init__()
        self.pool = nn.AvgPool2d(patch_size)

    def forward(self, x, mean_val):
        x = torch.mean(x, 1, keepdim=True)
        mean = self.pool(x)
        meanTensor = torch.FloatTensor([mean_val])
        d = torch.mean(torch.pow(mean - meanTensor, 2))
        return d


class L1_exp(nn.Module):

    def __init__(self, patch_size):
        super(L1_exp, self).__init__()
        self.pool = nn.AvgPool2d(patch_size)

    def forward(self, x, mean_val):
        crit = torch.nn.SmoothL1Loss()
        x = torch.mean(x, 1, keepdim=True)
        mean = self.pool(x)
        meanTensor = torch.FloatTensor([mean_val])
        d = torch.mean(crit(mean, meanTensor))
        return d


class L_TV(nn.Module):

    def __init__(self, TVLoss_weight=1):
        super(L_TV, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = (x.size()[2] - 1) * x.size()[3]
        count_w = x.size()[2] * (x.size()[3] - 1)
        h_tv = torch.pow(x[:, :, 1:, :] - x[:, :, :h_x - 1, :], 2).sum()
        w_tv = torch.pow(x[:, :, :, 1:] - x[:, :, :, :w_x - 1], 2).sum()
        return self.TVLoss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size


class fpn_module(nn.Module):

    def __init__(self, numClass):
        super(fpn_module, self).__init__()
        self.toplayer = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0)
        self.smooth1_1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth2_1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth3_1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth4_1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth1_2 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.smooth2_2 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.smooth3_2 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.smooth4_2 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.latlayer1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)
        self.classify = nn.Conv2d(128 * 4, numClass, kernel_size=3, stride=1, padding=1)

    def _concatenate(self, p5, p4, p3, p2):
        _, _, H, W = p2.size()
        p5 = F.upsample(p5, size=(H, W), mode='bilinear')
        p4 = F.upsample(p4, size=(H, W), mode='bilinear')
        p3 = F.upsample(p3, size=(H, W), mode='bilinear')
        return torch.cat([p5, p4, p3, p2], dim=1)

    def _upsample_add(self, x, y):
        """Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        """
        _, _, H, W = y.size()
        return F.upsample(x, size=(H, W), mode='bilinear') + y

    def forward(self, c2, c3, c4, c5):
        p5 = self.toplayer(c5)
        p4 = self._upsample_add(p5, self.latlayer1(c4))
        p3 = self._upsample_add(p4, self.latlayer2(c3))
        p2 = self._upsample_add(p3, self.latlayer3(c2))
        p5 = self.smooth1_2(self.smooth1_1(p5))
        p4 = self.smooth2_2(self.smooth2_1(p4))
        p3 = self.smooth3_2(self.smooth3_1(p3))
        p2 = self.smooth4_2(self.smooth4_1(p2))
        output = self.classify(self._concatenate(p5, p4, p3, p2))
        return output


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
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
        return out


class ResNet(nn.Module):

    def __init__(self, block, layers):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

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
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        return c2, c3, c4, c5


model_urls = {'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth', 'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth', 'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth', 'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth', 'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth'}


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3])
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']), strict=False)
    return model


class fpn(nn.Module):

    def __init__(self, numClass):
        super(fpn, self).__init__()
        self.resnet = resnet50(True)
        self.fpn = fpn_module(numClass)
        for m in self.fpn.children():
            nn.init.normal_(m.weight, mean=0, std=0.01)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        c2, c3, c4, c5 = self.resnet.forward(x)
        return self.fpn.forward(c2, c3, c4, c5)


def one_hot(index, classes):
    size = index.size()[:1] + (classes,) + index.size()[1:]
    view = index.size()[:1] + (1,) + index.size()[1:]
    if torch.cuda.is_available():
        mask = torch.Tensor(size).fill_(0)
    else:
        mask = torch.Tensor(size).fill_(0)
    index = index.view(view)
    ones = 1.0
    return mask.scatter_(1, index, ones)


class FocalLoss(nn.Module):

    def __init__(self, gamma=0, eps=1e-07, size_average=True, reduce=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.size_average = size_average
        self.reduce = reduce

    def forward(self, input, target):
        y = one_hot(target, input.size(1))
        probs = F.softmax(input, dim=1)
        probs = (probs * y).sum(1)
        probs = probs.clamp(self.eps, 1.0 - self.eps)
        log_p = probs.log()
        batch_loss = -torch.pow(1 - probs, self.gamma) * log_p
        if self.reduce:
            if self.size_average:
                loss = batch_loss.mean()
            else:
                loss = batch_loss.sum()
        else:
            loss = batch_loss
        return loss


class TC(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(TC, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, input):
        out = self.conv(input)
        return out


class DSC(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(DSC, self).__init__()
        self.depth_conv = nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size=3, stride=1, padding=1, groups=in_ch)
        self.point_conv = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=1, stride=1, padding=0, groups=1)

    def forward(self, input):
        out = self.depth_conv(input)
        out = self.point_conv(out)
        return out


class enhance_net_nopool(nn.Module):

    def __init__(self, scale_factor, conv_type='dsc'):
        super(enhance_net_nopool, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.scale_factor = scale_factor
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=self.scale_factor)
        number_f = 32
        if conv_type == 'dsc':
            self.conv = DSC
        elif conv_type == 'dc':
            self.conv = DC
        elif conv_type == 'tc':
            self.conv = TC
        else:
            None
        self.e_conv1 = self.conv(3, number_f)
        self.e_conv2 = self.conv(number_f, number_f)
        self.e_conv3 = self.conv(number_f, number_f)
        self.e_conv4 = self.conv(number_f, number_f)
        self.e_conv5 = self.conv(number_f * 2, number_f)
        self.e_conv6 = self.conv(number_f * 2, number_f)
        self.e_conv7 = self.conv(number_f * 2, 3)

    def enhance(self, x, x_r):
        x = x + x_r * (torch.pow(x, 2) - x)
        x = x + x_r * (torch.pow(x, 2) - x)
        x = x + x_r * (torch.pow(x, 2) - x)
        enhance_image_1 = x + x_r * (torch.pow(x, 2) - x)
        x = enhance_image_1 + x_r * (torch.pow(enhance_image_1, 2) - enhance_image_1)
        x = x + x_r * (torch.pow(x, 2) - x)
        x = x + x_r * (torch.pow(x, 2) - x)
        enhance_image = x + x_r * (torch.pow(x, 2) - x)
        return enhance_image

    def forward(self, x):
        if self.scale_factor == 1:
            x_down = x
        else:
            x_down = F.interpolate(x, scale_factor=1 / self.scale_factor, mode='bilinear')
        x1 = self.relu(self.e_conv1(x_down))
        x2 = self.relu(self.e_conv2(x1))
        x3 = self.relu(self.e_conv3(x2))
        x4 = self.relu(self.e_conv4(x3))
        x5 = self.relu(self.e_conv5(torch.cat([x3, x4], 1)))
        x6 = self.relu(self.e_conv6(torch.cat([x2, x5], 1)))
        x_r = F.tanh(self.e_conv7(torch.cat([x1, x6], 1)))
        if self.scale_factor == 1:
            x_r = x_r
        else:
            x_r = self.upsample(x_r)
        enhance_image = self.enhance(x, x_r)
        return enhance_image, x_r


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (DSC,
     lambda: ([], {'in_ch': 4, 'out_ch': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (L_TV,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (L_spa,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (L_spa8,
     lambda: ([], {'patch_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (TC,
     lambda: ([], {'in_ch': 4, 'out_ch': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (enhance_net_nopool,
     lambda: ([], {'scale_factor': 1.0}),
     lambda: ([torch.rand([4, 3, 64, 64])], {})),
    (fpn,
     lambda: ([], {'numClass': 4}),
     lambda: ([torch.rand([4, 3, 64, 64])], {})),
    (fpn_module,
     lambda: ([], {'numClass': 4}),
     lambda: ([torch.rand([4, 256, 64, 64]), torch.rand([4, 512, 64, 64]), torch.rand([4, 1024, 64, 64]), torch.rand([4, 2048, 64, 64])], {})),
]

