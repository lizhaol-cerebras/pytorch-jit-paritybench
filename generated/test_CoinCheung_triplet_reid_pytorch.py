
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


import torch.utils.model_zoo as model_zoo


from torch.utils.data import Dataset


from torch.utils.data import DataLoader


from torch.utils.data.sampler import Sampler


import numpy as np


import random


import logging


import torchvision.transforms as transforms


from torchvision.transforms import *


import math


import time


import itertools


class DenseNormReLU(nn.Module):

    def __init__(self, in_feats, out_feats, *args, **kwargs):
        super(DenseNormReLU, self).__init__(*args, **kwargs)
        self.dense = nn.Linear(in_features=in_feats, out_features=out_feats)
        self.bn = nn.BatchNorm1d(out_feats)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.dense(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Bottleneck(nn.Module):

    def __init__(self, in_chan, mid_chan, stride=1, stride_at_1x1=False, *args, **kwargs):
        super(Bottleneck, self).__init__(*args, **kwargs)
        stride1x1, stride3x3 = (stride, 1) if stride_at_1x1 else (1, stride)
        out_chan = 4 * mid_chan
        self.conv1 = nn.Conv2d(in_chan, mid_chan, kernel_size=1, stride=stride1x1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_chan)
        self.conv2 = nn.Conv2d(mid_chan, mid_chan, kernel_size=3, stride=stride3x3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_chan)
        self.conv3 = nn.Conv2d(mid_chan, out_chan, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_chan)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        if in_chan != out_chan or stride != 1:
            self.downsample = nn.Sequential(nn.Conv2d(in_chan, out_chan, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(out_chan))

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample == None:
            residual = x
        else:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


def create_layer(in_chan, mid_chan, b_num, stride):
    out_chan = mid_chan * 4
    blocks = [Bottleneck(in_chan, mid_chan, stride=stride)]
    for i in range(1, b_num):
        blocks.append(Bottleneck(out_chan, mid_chan, stride=1))
    return nn.Sequential(*blocks)


param_url = 'https://download.pytorch.org/models/resnet50-19c8e357.pth'


class EmbedNetwork(nn.Module):

    def __init__(self, dims=128, pretrained_base=True, *args, **kwargs):
        super(EmbedNetwork, self).__init__(*args, **kwargs)
        self.pretrained_base = pretrained_base
        resnet50 = torchvision.models.resnet50()
        self.conv1 = resnet50.conv1
        self.bn1 = resnet50.bn1
        self.relu = resnet50.relu
        self.maxpool = resnet50.maxpool
        self.layer1 = create_layer(64, 64, 3, stride=1)
        self.layer2 = create_layer(256, 128, 4, stride=2)
        self.layer3 = create_layer(512, 256, 6, stride=2)
        self.layer4 = create_layer(1024, 512, 3, stride=1)
        self.fc_head = DenseNormReLU(in_feats=2048, out_feats=1024)
        self.embed = nn.Linear(in_features=1024, out_features=dims)
        if self.pretrained_base:
            new_state = model_zoo.load_url(param_url)
            state_dict = self.state_dict()
            for k, v in new_state.items():
                if 'fc' in k:
                    continue
                state_dict.update({k: v})
            self.load_state_dict(state_dict)
        for el in self.fc_head.children():
            if isinstance(el, nn.Linear):
                nn.init.kaiming_normal_(el.weight, a=1)
                nn.init.constant_(el.bias, 0)
        nn.init.kaiming_normal_(self.embed.weight, a=1)
        nn.init.constant_(self.embed.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.contiguous().view(-1, 2048)
        x = self.fc_head(x)
        x = self.embed(x)
        return x


class TripletLoss(nn.Module):
    """
    Compute normal triplet loss or soft margin triplet loss given triplets
    """

    def __init__(self, margin=None):
        super(TripletLoss, self).__init__()
        self.margin = margin
        if self.margin is None:
            self.Loss = nn.SoftMarginLoss()
        else:
            self.Loss = nn.TripletMarginLoss(margin=margin, p=2)

    def forward(self, anchor, pos, neg):
        if self.margin is None:
            num_samples = anchor.shape[0]
            y = torch.ones((num_samples, 1)).view(-1)
            if anchor.is_cuda:
                y = y
            ap_dist = torch.norm(anchor - pos, 2, dim=1).view(-1)
            an_dist = torch.norm(anchor - neg, 2, dim=1).view(-1)
            loss = self.Loss(an_dist - ap_dist, y)
        else:
            loss = self.Loss(anchor, pos, neg)
        return loss


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (Bottleneck,
     lambda: ([], {'in_chan': 4, 'mid_chan': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (DenseNormReLU,
     lambda: ([], {'in_feats': 4, 'out_feats': 4}),
     lambda: ([torch.rand([4, 4, 4])], {})),
    (EmbedNetwork,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {})),
]

