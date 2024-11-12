
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


import torchvision


import numpy as np


class HBP(torch.nn.Module):

    def __init__(self):
        torch.nn.Module.__init__(self)
        self.features = torchvision.models.vgg16(pretrained=True).features
        self.features_conv5_1 = torch.nn.Sequential(*list(self.features.children())[:-5])
        self.features_conv5_2 = torch.nn.Sequential(*list(self.features.children())[-5:-3])
        self.features_conv5_3 = torch.nn.Sequential(*list(self.features.children())[-3:-1])
        self.bilinear_proj_1 = torch.nn.Conv2d(512, 8192, kernel_size=1, bias=True)
        self.bilinear_proj_2 = torch.nn.Conv2d(512, 8192, kernel_size=1, bias=True)
        self.bilinear_proj_3 = torch.nn.Conv2d(512, 8192, kernel_size=1, bias=True)
        self.fc = torch.nn.Linear(8192 * 3, 200)
        for param in self.features_conv5_1.parameters():
            param.requires_grad = False
        for param in self.features_conv5_2.parameters():
            param.requires_grad = False
        for param in self.features_conv5_3.parameters():
            param.requires_grad = False
        torch.nn.init.xavier_normal_(self.fc.weight.data)
        if self.fc.bias is not None:
            torch.nn.init.constant_(self.fc.bias.data, val=0)

    def hbp_1_2(self, conv1, conv2):
        N = conv1.size()[0]
        proj_1 = self.bilinear_proj_1(conv1)
        proj_2 = self.bilinear_proj_2(conv2)
        assert proj_1.size() == (N, 8192, 28, 28)
        X = proj_1 * proj_2
        assert X.size() == (N, 8192, 28, 28)
        X = torch.sum(X.view(X.size()[0], X.size()[1], -1), dim=2)
        X = X.view(N, 8192)
        X = torch.sign(X) * torch.sqrt(torch.abs(X) + 1e-05)
        X = torch.nn.functional.normalize(X)
        return X

    def hbp_1_3(self, conv1, conv3):
        N = conv1.size()[0]
        proj_1 = self.bilinear_proj_1(conv1)
        proj_3 = self.bilinear_proj_3(conv3)
        assert proj_1.size() == (N, 8192, 28, 28)
        X = proj_1 * proj_3
        assert X.size() == (N, 8192, 28, 28)
        X = torch.sum(X.view(X.size()[0], X.size()[1], -1), dim=2)
        X = X.view(N, 8192)
        X = torch.sign(X) * torch.sqrt(torch.abs(X) + 1e-05)
        X = torch.nn.functional.normalize(X)
        return X

    def hbp_2_3(self, conv2, conv3):
        N = conv2.size()[0]
        proj_2 = self.bilinear_proj_2(conv2)
        proj_3 = self.bilinear_proj_3(conv3)
        assert proj_2.size() == (N, 8192, 28, 28)
        X = proj_2 * proj_3
        assert X.size() == (N, 8192, 28, 28)
        X = torch.sum(X.view(X.size()[0], X.size()[1], -1), dim=2)
        X = X.view(N, 8192)
        X = torch.sign(X) * torch.sqrt(torch.abs(X) + 1e-05)
        X = torch.nn.functional.normalize(X)
        return X

    def forward(self, X):
        N = X.size()[0]
        assert X.size() == (N, 3, 448, 448)
        X_conv5_1 = self.features_conv5_1(X)
        X_conv5_2 = self.features_conv5_2(X_conv5_1)
        X_conv5_3 = self.features_conv5_3(X_conv5_2)
        X_branch_1 = self.hbp_1_2(X_conv5_1, X_conv5_2)
        X_branch_2 = self.hbp_1_3(X_conv5_1, X_conv5_3)
        X_branch_3 = self.hbp_2_3(X_conv5_2, X_conv5_3)
        X_branch = torch.cat([X_branch_1, X_branch_2, X_branch_3], dim=1)
        assert X_branch.size() == (N, 8192 * 3)
        X = self.fc(X_branch)
        assert X.size() == (N, 200)
        return X

