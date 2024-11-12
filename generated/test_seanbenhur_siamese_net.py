
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


import torch.nn as nn


import numpy as np


import torch.nn.functional as F


import pandas as pd


import torch


import torch.utils.data as utils


import torchvision.transforms as transforms


from torch.utils.data import DataLoader


from torch.utils.data import Dataset


import matplotlib.pyplot as plt


import torchvision.utils


import torchvision


from torch.autograd import Variable


class ContrastiveLoss(nn.Module):
    """Contrastive loss function"""

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) + label * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive


class SiameseNetwork(nn.Module):

    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.cnn1 = nn.Sequential(nn.Conv2d(1, 96, kernel_size=11, stride=1), nn.BatchNorm2d(96), nn.ReLU(inplace=True), nn.MaxPool2d(3, stride=2), nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2), nn.BatchNorm2d(256), nn.ReLU(inplace=True), nn.MaxPool2d(3, stride=2), nn.Dropout2d(p=0.3), nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(384), nn.ReLU(inplace=True), nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True), nn.MaxPool2d(3, stride=2), nn.Dropout2d(p=0.3))
        self.fc1 = nn.Sequential(nn.Linear(30976, 1024), nn.ReLU(inplace=True), nn.Dropout2d(p=0.5), nn.Linear(1024, 128), nn.ReLU(inplace=True), nn.Linear(128, 2))

    def forward_once(self, x):
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (ContrastiveLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
]

