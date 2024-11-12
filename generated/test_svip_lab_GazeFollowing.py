
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


from torch.autograd import Variable


import torch.optim as optim


import torch.nn as nn


import torch.nn.functional as F


from torch.nn import DataParallel


from torch.utils.data import Dataset


from torch.utils.data import DataLoader


from torchvision import transforms


import time


import numpy as np


import random


import itertools


from scipy.io import loadmat


import logging


from scipy import signal


import math


import torch.utils.model_zoo as model_zoo


from scipy.io import savemat


class FPN(nn.Module):

    def __init__(self):
        super(FPN, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.resnet = resnet_fpn.resnet50(pretrained=True)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.c5_conv = nn.Conv2d(2048, 256, (1, 1))
        self.c4_conv = nn.Conv2d(1024, 256, (1, 1))
        self.c3_conv = nn.Conv2d(512, 256, (1, 1))
        self.c2_conv = nn.Conv2d(256, 256, (1, 1))
        self.p5_conv = nn.Conv2d(256, 256, (3, 3), padding=1)
        self.p4_conv = nn.Conv2d(256, 256, (3, 3), padding=1)
        self.p3_conv = nn.Conv2d(256, 256, (3, 3), padding=1)
        self.p2_conv = nn.Conv2d(256, 256, (3, 3), padding=1)
        self.sigmoid = nn.Sigmoid()
        self.predict = nn.Conv2d(256, 1, (3, 3), padding=1)

    def top_down(self, x):
        c2, c3, c4, c5 = x
        p5 = self.c5_conv(c5)
        p4 = self.upsample(p5) + self.c4_conv(c4)
        p3 = self.upsample(p4) + self.c3_conv(c3)
        p2 = self.upsample(p3) + self.c2_conv(c2)
        p5 = self.relu(self.p5_conv(p5))
        p4 = self.relu(self.p4_conv(p4))
        p3 = self.relu(self.p3_conv(p3))
        p2 = self.relu(self.p2_conv(p2))
        return p2, p3, p4, p5

    def forward(self, x):
        c2, c3, c4, c5 = self.resnet(x)
        p2, p3, p4, p5 = self.top_down((c2, c3, c4, c5))
        heatmap = self.sigmoid(self.predict(p2))
        return heatmap


class GazeNet(nn.Module):

    def __init__(self):
        super(GazeNet, self).__init__()
        self.face_net = M.resnet50(pretrained=True)
        self.face_process = nn.Sequential(nn.Linear(2048, 512), nn.ReLU(inplace=True))
        self.fpn_net = FPN()
        self.eye_position_transform = nn.Sequential(nn.Linear(2, 256), nn.ReLU(inplace=True))
        self.fusion = nn.Sequential(nn.Linear(512 + 256, 256), nn.ReLU(inplace=True), nn.Linear(256, 2))
        self.relu = nn.ReLU(inplace=False)
        conv = [x.clone() for x in self.fpn_net.resnet.conv1.parameters()][0]
        new_kernel_channel = conv.data.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)
        new_kernel = torch.cat((conv.data, new_kernel_channel), 1)
        new_conv = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)
        new_conv.weight.data = new_kernel
        self.fpn_net.resnet.conv1 = new_conv

    def forward(self, x):
        image, face_image, gaze_field, eye_position = x
        face_feature = self.face_net(face_image)
        face_feature = self.face_process(face_feature)
        eye_feature = self.eye_position_transform(eye_position)
        feature = torch.cat((face_feature, eye_feature), 1)
        direction = self.fusion(feature)
        norm = torch.norm(direction, 2, dim=1)
        normalized_direction = direction / norm.view([-1, 1])
        batch_size, channel, height, width = gaze_field.size()
        gaze_field = gaze_field.permute([0, 2, 3, 1]).contiguous()
        gaze_field = gaze_field.view([batch_size, -1, 2])
        gaze_field = torch.matmul(gaze_field, normalized_direction.view([batch_size, 2, 1]))
        gaze_field_map = gaze_field.view([batch_size, height, width, 1])
        gaze_field_map = gaze_field_map.permute([0, 3, 1, 2]).contiguous()
        gaze_field_map = self.relu(gaze_field_map)
        gaze_field_map_2 = torch.pow(gaze_field_map, 2)
        gaze_field_map_3 = torch.pow(gaze_field_map, 5)
        image = torch.cat([image, gaze_field_map, gaze_field_map_2, gaze_field_map_3], dim=1)
        heatmap = self.fpn_net(image)
        return direction, heatmap


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


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
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
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
        p2 = self.layer1(x)
        p3 = self.layer2(p2)
        p4 = self.layer3(p3)
        p5 = self.layer4(p4)
        return p2, p3, p4, p5


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (BasicBlock,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
]

