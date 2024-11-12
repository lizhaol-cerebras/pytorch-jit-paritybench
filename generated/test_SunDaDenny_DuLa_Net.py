
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


import torchvision


import time


import numpy as np


from torch.autograd import Variable


import math


import torch.utils.model_zoo as model_zoo


from torchvision import transforms


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv3x3_relu(in_planes, out_planes):
    return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=3, padding=1), nn.ReLU(inplace=True))


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

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

    def __init__(self, block, layers, num_classes=1000):
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
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

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
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
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


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


class DulaNet_Branch(nn.Module):

    def __init__(self, backbone):
        super(DulaNet_Branch, self).__init__()
        bb_dict = {'resnet18': resnet18, 'resnet34': resnet34, 'resnet50': resnet50}
        self.encoder = bb_dict[backbone]()
        feat_dim = 512 if backbone != 'resnet50' else 2048
        self.decoder = nn.ModuleList([conv3x3_relu(feat_dim, 256), conv3x3_relu(256, 128), conv3x3_relu(128, 64), conv3x3_relu(64, 32), conv3x3_relu(32, 16)])
        self.last = conv3x3(16, 1)

    def forward_get_feats(self, x):
        x = self.encoder(x)
        feats = [x]
        for conv in self.decoder:
            x = F.interpolate(x, scale_factor=(2, 2), mode='nearest')
            x = conv(x)
            feats.append(x)
        out = self.last(x)
        return out, feats

    def forward_from_feats(self, x, feats):
        x = self.encoder(x)
        for i, conv in enumerate(self.decoder):
            x = x + feats[i]
            x = F.interpolate(x, scale_factor=(2, 2), mode='nearest')
            x = conv(x)
        out = self.last(x)
        return out


class E2P(nn.Module):

    def __init__(self, equ_size, out_dim, fov, radius=128, up_flip=True, gpu=True):
        super(E2P, self).__init__()
        self.equ_h = equ_size[0]
        self.equ_w = equ_size[1]
        self.out_dim = out_dim
        self.fov = fov
        self.radius = radius
        self.up_flip = up_flip
        self.gpu = gpu
        R_lst = []
        theta_lst = np.array([-90, 0, 90, 180], np.float) / 180 * np.pi
        phi_lst = np.array([90, -90], np.float) / 180 * np.pi
        for theta in theta_lst:
            angle_axis = theta * np.array([0, 1, 0], np.float)
            R = cv2.Rodrigues(angle_axis)[0]
            R_lst.append(R)
        for phi in phi_lst:
            angle_axis = phi * np.array([1, 0, 0], np.float)
            R = cv2.Rodrigues(angle_axis)[0]
            R_lst.append(R)
        if gpu:
            R_lst = [Variable(torch.FloatTensor(x)) for x in R_lst]
        else:
            R_lst = [Variable(torch.FloatTensor(x)) for x in R_lst]
        R_lst = R_lst[4:]
        equ_cx = (self.equ_w - 1) / 2.0
        equ_cy = (self.equ_h - 1) / 2.0
        c_x = (out_dim - 1) / 2.0
        c_y = (out_dim - 1) / 2.0
        wangle = (180 - fov) / 2.0
        w_len = 2 * radius * np.sin(np.radians(fov / 2.0)) / np.sin(np.radians(wangle))
        f = radius / w_len * out_dim
        cx = c_x
        cy = c_y
        self.intrisic = {'f': f, 'cx': cx, 'cy': cy}
        interval = w_len / (out_dim - 1)
        z_map = np.zeros([out_dim, out_dim], np.float32) + radius
        x_map = np.tile((np.arange(out_dim) - c_x) * interval, [out_dim, 1])
        y_map = np.tile((np.arange(out_dim) - c_y) * interval, [out_dim, 1]).T
        D = np.sqrt(x_map ** 2 + y_map ** 2 + z_map ** 2)
        xyz = np.zeros([out_dim, out_dim, 3], np.float)
        xyz[:, :, 0] = radius / D * x_map[:, :]
        xyz[:, :, 1] = radius / D * y_map[:, :]
        xyz[:, :, 2] = radius / D * z_map[:, :]
        if gpu:
            xyz = Variable(torch.FloatTensor(xyz))
        else:
            xyz = Variable(torch.FloatTensor(xyz))
        self.xyz = xyz.clone()
        self.xyz = self.xyz.unsqueeze(0)
        self.xyz /= torch.norm(self.xyz, p=2, dim=3).unsqueeze(-1)
        reshape_xyz = xyz.view(out_dim * out_dim, 3).transpose(0, 1)
        self.loc = []
        self.grid = []
        for i, R in enumerate(R_lst):
            result = torch.matmul(R, reshape_xyz).transpose(0, 1)
            tmp_xyz = result.contiguous().view(1, out_dim, out_dim, 3)
            self.grid.append(tmp_xyz)
            lon = torch.atan2(result[:, 0], result[:, 2]).view(1, out_dim, out_dim, 1) / np.pi
            lat = torch.asin(result[:, 1] / radius).view(1, out_dim, out_dim, 1) / (np.pi / 2)
            self.loc.append(torch.cat([lon, lat], dim=3))

    def forward(self, batch):
        batch_size = batch.size()[0]
        up_views = []
        down_views = []
        for i in range(batch_size):
            up_coor, down_coor = self.loc
            up_view = F.grid_sample(batch[i:i + 1], up_coor)
            down_view = F.grid_sample(batch[i:i + 1], down_coor)
            up_views.append(up_view)
            down_views.append(down_view)
        up_views = torch.cat(up_views, dim=0)
        if self.up_flip:
            up_views = torch.flip(up_views, dims=[2])
        down_views = torch.cat(down_views, dim=0)
        return up_views, down_views

    def GetGrid(self):
        return self.xyz


class DuLaNet(nn.Module):

    def __init__(self, backbone):
        super(DuLaNet, self).__init__()
        self.model_equi = DulaNet_Branch(backbone)
        self.model_up = DulaNet_Branch(backbone)
        self.model_h = nn.Sequential(nn.Linear(512, 256), nn.Dropout(inplace=True), nn.Linear(256, 64), nn.Dropout(inplace=True), nn.Linear(64, 1))
        self.e2p = E2P(cf.pano_size, cf.fp_size, cf.fp_fov)
        fuse_dim = [int(cf.pano_size[0] / 32 * 2 ** i) for i in range(6)]
        self.e2ps_f = [E2P((n, n * 2), n, cf.fp_fov) for n in fuse_dim]

    def forward(self, pano_view):
        [up_view, down_view] = self.e2p(pano_view)
        fcmap, feats_equi = self.model_equi.forward_get_feats(pano_view)
        feats_fuse = []
        for i, feat in enumerate(feats_equi):
            [feat_up, _] = self.e2ps_f[i](feat)
            feats_fuse.append(feat_up * 0.6 * (1 / 3) ** i)
        fpmap = self.model_up.forward_from_feats(up_view, feats_fuse)
        fpmap = torch.sigmoid(fpmap)
        fcmap = torch.sigmoid(fcmap)
        height = self.model_h(feats_equi[0].mean(1).view(-1, 512))
        return fpmap, fcmap, height


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (BasicBlock,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
]

