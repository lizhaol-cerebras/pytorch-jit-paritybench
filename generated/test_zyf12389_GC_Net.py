
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


import torchvision.transforms as transforms


from torch.utils.data import Dataset


from torch.utils.data import DataLoader


from torch.autograd import Variable


import numpy as np


import torch.optim as optim


import time


import math


import torch.nn.init as init


class BasicBlock(nn.Module):

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ThreeDConv(nn.Module):

    def __init__(self, in_planes, planes, stride=1):
        super(ThreeDConv, self).__init__()
        self.conv1 = nn.Conv3d(in_planes, planes, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm3d(planes)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = F.relu(self.bn3(self.conv3(out)))
        return out


class GC_NET(nn.Module):

    def __init__(self, block, block_3d, num_block, height, width, maxdisp):
        super(GC_NET, self).__init__()
        self.height = height
        self.width = width
        self.maxdisp = int(maxdisp / 2)
        self.in_planes = 32
        self.conv0 = nn.Conv2d(3, 32, 5, 2, 2)
        self.bn0 = nn.BatchNorm2d(32)
        self.res_block = self._make_layer(block, self.in_planes, 32, num_block[0], stride=1)
        self.conv1 = nn.Conv2d(32, 32, 3, 1, 1)
        self.conv3d_1 = nn.Conv3d(64, 32, 3, 1, 1)
        self.bn3d_1 = nn.BatchNorm3d(32)
        self.conv3d_2 = nn.Conv3d(32, 32, 3, 1, 1)
        self.bn3d_2 = nn.BatchNorm3d(32)
        self.conv3d_3 = nn.Conv3d(64, 64, 3, 2, 1)
        self.bn3d_3 = nn.BatchNorm3d(64)
        self.conv3d_4 = nn.Conv3d(64, 64, 3, 2, 1)
        self.bn3d_4 = nn.BatchNorm3d(64)
        self.conv3d_5 = nn.Conv3d(64, 64, 3, 2, 1)
        self.bn3d_5 = nn.BatchNorm3d(64)
        self.block_3d_1 = self._make_layer(block_3d, 64, 64, num_block[1], stride=2)
        self.block_3d_2 = self._make_layer(block_3d, 64, 64, num_block[1], stride=2)
        self.block_3d_3 = self._make_layer(block_3d, 64, 64, num_block[1], stride=2)
        self.block_3d_4 = self._make_layer(block_3d, 64, 128, num_block[1], stride=2)
        self.deconv1 = nn.ConvTranspose3d(128, 64, 3, 2, 1, 1)
        self.debn1 = nn.BatchNorm3d(64)
        self.deconv2 = nn.ConvTranspose3d(64, 64, 3, 2, 1, 1)
        self.debn2 = nn.BatchNorm3d(64)
        self.deconv3 = nn.ConvTranspose3d(64, 64, 3, 2, 1, 1)
        self.debn3 = nn.BatchNorm3d(64)
        self.deconv4 = nn.ConvTranspose3d(64, 32, 3, 2, 1, 1)
        self.debn4 = nn.BatchNorm3d(32)
        self.deconv5 = nn.ConvTranspose3d(32, 1, 3, 2, 1, 1)

    def forward(self, imgLeft, imgRight):
        imgl0 = F.relu(self.bn0(self.conv0(imgLeft)))
        imgr0 = F.relu(self.bn0(self.conv0(imgRight)))
        imgl_block = self.res_block(imgl0)
        imgr_block = self.res_block(imgr0)
        imgl1 = self.conv1(imgl_block)
        imgr1 = self.conv1(imgr_block)
        cost_volum = self.cost_volume(imgl1, imgr1)
        conv3d_out = F.relu(self.bn3d_1(self.conv3d_1(cost_volum)))
        conv3d_out = F.relu(self.bn3d_2(self.conv3d_2(conv3d_out)))
        conv3d_block_1 = self.block_3d_1(cost_volum)
        conv3d_21 = F.relu(self.bn3d_3(self.conv3d_3(cost_volum)))
        conv3d_block_2 = self.block_3d_2(conv3d_21)
        conv3d_24 = F.relu(self.bn3d_4(self.conv3d_4(conv3d_21)))
        conv3d_block_3 = self.block_3d_3(conv3d_24)
        conv3d_27 = F.relu(self.bn3d_5(self.conv3d_5(conv3d_24)))
        conv3d_block_4 = self.block_3d_4(conv3d_27)
        deconv3d = F.relu(self.debn1(self.deconv1(conv3d_block_4)) + conv3d_block_3)
        deconv3d = F.relu(self.debn2(self.deconv2(deconv3d)) + conv3d_block_2)
        deconv3d = F.relu(self.debn3(self.deconv3(deconv3d)) + conv3d_block_1)
        deconv3d = F.relu(self.debn4(self.deconv4(deconv3d)) + conv3d_out)
        deconv3d = self.deconv5(deconv3d)
        out = deconv3d.view(1, self.maxdisp * 2, self.height, self.width)
        prob = F.softmax(-out, 1)
        return prob

    def _make_layer(self, block, in_planes, planes, num_block, stride):
        strides = [stride] + [1] * (num_block - 1)
        layers = []
        for step in strides:
            layers.append(block(in_planes, planes, step))
        return nn.Sequential(*layers)

    def cost_volume(self, imgl, imgr):
        xx_list = []
        pad_opr1 = nn.ZeroPad2d((0, self.maxdisp, 0, 0))
        xleft = pad_opr1(imgl)
        for d in range(self.maxdisp):
            pad_opr2 = nn.ZeroPad2d((d, self.maxdisp - d, 0, 0))
            xright = pad_opr2(imgr)
            xx_temp = torch.cat((xleft, xright), 1)
            xx_list.append(xx_temp)
        xx = torch.cat(xx_list, 1)
        xx = xx.view(1, self.maxdisp, 64, int(self.height / 2), int(self.width / 2) + self.maxdisp)
        xx0 = xx.permute(0, 2, 1, 3, 4)
        xx0 = xx0[:, :, :, :, :int(self.width / 2)]
        return xx0


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (BasicBlock,
     lambda: ([], {'in_planes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ThreeDConv,
     lambda: ([], {'in_planes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4, 4])], {})),
]

