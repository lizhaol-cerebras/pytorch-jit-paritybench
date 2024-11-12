
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


from collections import OrderedDict


from torch.utils.model_zoo import load_url


import copy


import torch.optim as optim


import torchvision.transforms as transforms


class VGG(nn.Module):

    def __init__(self, features, num_classes=1000):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(nn.Linear(512 * 7 * 7, 4096), nn.ReLU(True), nn.Dropout(), nn.Linear(4096, 4096), nn.ReLU(True), nn.Dropout(), nn.Linear(4096, num_classes))


class VGG_SOD(nn.Module):

    def __init__(self, features, num_classes=100):
        super(VGG_SOD, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(nn.Linear(512 * 7 * 7, 4096), nn.ReLU(True), nn.Dropout(), nn.Linear(4096, 4096), nn.ReLU(True), nn.Dropout(), nn.Linear(4096, 100))


class VGG_FCN32S(nn.Module):

    def __init__(self, features, num_classes=1000):
        super(VGG_FCN32S, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(nn.Conv2d(512, 4096, (7, 7)), nn.ReLU(True), nn.Dropout(0.5), nn.Conv2d(4096, 4096, (1, 1)), nn.ReLU(True), nn.Dropout(0.5))


class VGG_PRUNED(nn.Module):

    def __init__(self, features, num_classes=1000):
        super(VGG_PRUNED, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(nn.Linear(512 * 7 * 7, 4096), nn.ReLU(True), nn.Dropout(0.5), nn.Linear(4096, 4096), nn.ReLU(True), nn.Dropout(0.5))


class NIN(nn.Module):

    def __init__(self, pooling):
        super(NIN, self).__init__()
        if pooling == 'max':
            pool2d = nn.MaxPool2d((3, 3), (2, 2), (0, 0), ceil_mode=True)
        elif pooling == 'avg':
            pool2d = nn.AvgPool2d((3, 3), (2, 2), (0, 0), ceil_mode=True)
        self.features = nn.Sequential(nn.Conv2d(3, 96, (11, 11), (4, 4)), nn.ReLU(inplace=True), nn.Conv2d(96, 96, (1, 1)), nn.ReLU(inplace=True), nn.Conv2d(96, 96, (1, 1)), nn.ReLU(inplace=True), pool2d, nn.Conv2d(96, 256, (5, 5), (1, 1), (2, 2)), nn.ReLU(inplace=True), nn.Conv2d(256, 256, (1, 1)), nn.ReLU(inplace=True), nn.Conv2d(256, 256, (1, 1)), nn.ReLU(inplace=True), pool2d, nn.Conv2d(256, 384, (3, 3), (1, 1), (1, 1)), nn.ReLU(inplace=True), nn.Conv2d(384, 384, (1, 1)), nn.ReLU(inplace=True), nn.Conv2d(384, 384, (1, 1)), nn.ReLU(inplace=True), pool2d, nn.Dropout(0.5), nn.Conv2d(384, 1024, (3, 3), (1, 1), (1, 1)), nn.ReLU(inplace=True), nn.Conv2d(1024, 1024, (1, 1)), nn.ReLU(inplace=True), nn.Conv2d(1024, 1000, (1, 1)), nn.ReLU(inplace=True), nn.AvgPool2d((6, 6), (1, 1), (0, 0), ceil_mode=True), nn.Softmax())


class ModelParallel(nn.Module):

    def __init__(self, net, device_ids, device_splits):
        super(ModelParallel, self).__init__()
        self.device_list = self.name_devices(device_ids.split(','))
        self.chunks = self.chunks_to_devices(self.split_net(net, device_splits.split(',')))

    def name_devices(self, input_list):
        device_list = []
        for i, device in enumerate(input_list):
            if str(device).lower() != 'c':
                device_list.append('cuda:' + str(device))
            else:
                device_list.append('cpu')
        return device_list

    def split_net(self, net, device_splits):
        chunks, cur_chunk = [], nn.Sequential()
        for i, l in enumerate(net):
            cur_chunk.add_module(str(i), net[i])
            if str(i) in device_splits and device_splits != '':
                del device_splits[0]
                chunks.append(cur_chunk)
                cur_chunk = nn.Sequential()
        chunks.append(cur_chunk)
        return chunks

    def chunks_to_devices(self, chunks):
        for i, chunk in enumerate(chunks):
            chunk
        return chunks

    def c(self, input, i):
        if input.type() == 'torch.FloatTensor' and 'cuda' in self.device_list[i]:
            input = input.type('torch.cuda.FloatTensor')
        elif input.type() == 'torch.cuda.FloatTensor' and 'cpu' in self.device_list[i]:
            input = input.type('torch.FloatTensor')
        return input

    def forward(self, input):
        for i, chunk in enumerate(self.chunks):
            if i < len(self.chunks) - 1:
                input = self.c(chunk(self.c(input, i).to(self.device_list[i])), i + 1)
            else:
                input = chunk(input)
        return input


class ScaleGradients(torch.autograd.Function):

    @staticmethod
    def forward(self, input_tensor, strength):
        self.strength = strength
        return input_tensor

    @staticmethod
    def backward(self, grad_output):
        grad_input = grad_output.clone()
        grad_input = grad_input / (torch.norm(grad_input, keepdim=True) + 1e-08)
        return grad_input * self.strength * self.strength, None


class ContentLoss(nn.Module):

    def __init__(self, strength, normalize):
        super(ContentLoss, self).__init__()
        self.strength = strength
        self.crit = nn.MSELoss()
        self.mode = 'None'
        self.normalize = normalize

    def forward(self, input):
        if self.mode == 'loss':
            loss = self.crit(input, self.target)
            if self.normalize:
                loss = ScaleGradients.apply(loss, self.strength)
            self.loss = loss * self.strength
        elif self.mode == 'capture':
            self.target = input.detach()
        return input


class GramMatrix(nn.Module):

    def forward(self, input):
        B, C, H, W = input.size()
        x_flat = input.view(C, H * W)
        return torch.mm(x_flat, x_flat.t())


class StyleLoss(nn.Module):

    def __init__(self, strength, normalize):
        super(StyleLoss, self).__init__()
        self.target = torch.Tensor()
        self.strength = strength
        self.gram = GramMatrix()
        self.crit = nn.MSELoss()
        self.mode = 'None'
        self.blend_weight = None
        self.normalize = normalize

    def forward(self, input):
        self.G = self.gram(input)
        self.G = self.G.div(input.nelement())
        if self.mode == 'capture':
            if self.blend_weight == None:
                self.target = self.G.detach()
            elif self.target.nelement() == 0:
                self.target = self.G.detach().mul(self.blend_weight)
            else:
                self.target = self.target.add(self.blend_weight, self.G.detach())
        elif self.mode == 'loss':
            loss = self.crit(self.G, self.target)
            if self.normalize:
                loss = ScaleGradients.apply(loss, self.strength)
            self.loss = self.strength * loss
        return input


class TVLoss(nn.Module):

    def __init__(self, strength):
        super(TVLoss, self).__init__()
        self.strength = strength

    def forward(self, input):
        self.x_diff = input[:, :, 1:, :] - input[:, :, :-1, :]
        self.y_diff = input[:, :, :, 1:] - input[:, :, :, :-1]
        self.loss = self.strength * (torch.sum(torch.abs(self.x_diff)) + torch.sum(torch.abs(self.y_diff)))
        return input


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (ContentLoss,
     lambda: ([], {'strength': 4, 'normalize': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (TVLoss,
     lambda: ([], {'strength': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
]

