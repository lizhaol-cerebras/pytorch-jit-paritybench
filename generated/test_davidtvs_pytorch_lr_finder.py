
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


import random


import time


import numpy as np


import torch


import torch.nn.functional as F


import torch.optim as optim


from torch.utils.data import Subset


from torch.utils.data import DataLoader


from torchvision import datasets


from torchvision import transforms


from torch.utils.data import Dataset


import matplotlib.pyplot as plt


import copy


from torch.optim.lr_scheduler import _LRScheduler


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


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Cifar10ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=10, ch_width=2):
        super(Cifar10ResNet, self).__init__()
        width = [16, 16 * ch_width, 16 * ch_width * ch_width]
        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, width[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width[0])
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, width[0], layers[0])
        self.layer2 = self._make_layer(block, width[1], layers[1], stride=2)
        self.layer3 = self._make_layer(block, width[2], layers[2], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(width[2] * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(conv1x1(self.inplanes, planes * block.expansion, stride), nn.BatchNorm2d(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class ConvNet(nn.Module):

    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1)
        self.conv2_drop = nn.Dropout2d()
        self.net = nn.Sequential(self.conv1, nn.MaxPool2d(2), nn.ReLU(True), self.conv2, self.conv2_drop, nn.MaxPool2d(2), nn.ReLU(True))
        self.fc1 = nn.Linear(5 * 5 * 32, 64)
        self.fc2 = nn.Linear(64, 16)

    def forward(self, x):
        x = self.net(x)
        x = x.view(-1, 5 * 5 * 32)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class LinearMLP(nn.Module):

    def __init__(self, layer_dim):
        super(LinearMLP, self).__init__()
        io_pairs = zip(layer_dim[:-1], layer_dim[1:])
        layers = [nn.Linear(idim, odim) for idim, odim in io_pairs]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class LSTMTagger(nn.Module):
    """
    A POS (part-of-speech) tagger using LSTM.
    """

    def __init__(self, embedding_dim, hidden_dim, tagset_size, vocab_to_ix):
        """
        Arguments:
            embedding_dim (int): dimension of output embedding vector.
            hidden_dim (int): dimension of LSTM hidden layers.
            target_size (int): number of tags for tagger to learn.
            vocab_to_ix (dict): a dict for vocab to index conversion.
        """
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim
        self.vocab_to_ix = vocab_to_ix
        self.word_embeddings = nn.Embedding(len(vocab_to_ix), embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def get_indices_of_vocabs(self, vocabs):
        return torch.LongTensor([self.vocab_to_ix[v] for v in vocabs])

    def forward(self, vocabs):
        """
        Arguments:
            vocabs (list of str): tokenized sentence.
        """
        device = next(self.lstm.parameters()).device
        indices = self.get_indices_of_vocabs(vocabs)
        embeds = self.word_embeddings(indices)
        lstm_out, _ = self.lstm(embeds.view(len(vocabs), 1, -1))
        tag_space = self.hidden2tag(lstm_out.view(len(vocabs), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (BasicBlock,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (LinearMLP,
     lambda: ([], {'layer_dim': [4, 4]}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
]

