
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


import numpy as np


import torch


from torch import nn


from torch.autograd import Variable


from torch.optim import SGD


import torch as t


import time


from torch.nn import functional as F


class BasicModule(t.nn.Module):
    """
    封装nn.Module，主要提供save和load两个方法
    """

    def __init__(self):
        super(BasicModule, self).__init__()
        self.module_name = str(type(self))

    def load(self, path, use_gpu=False):
        """
        可加载指定路径的模型
        """
        if not use_gpu:
            self.load_state_dict(t.load(path, map_location=lambda storage, loc: storage))
        else:
            self.load_state_dict(t.load(path))

    def save(self, name=None):
        """
        保存模型，默认使用"模型名字+时间"作为文件名
        """
        if name is None:
            prefix = self.module_name + '_'
            name = time.strftime(prefix + '%m%d_%H:%M:%S.pth')
        t.save(self.state_dict(), 'checkpoints/' + name)
        return name

    def forward(self, *input):
        pass


class ImgModule(BasicModule):

    def __init__(self, bit, pretrain_model=None):
        super(ImgModule, self).__init__()
        self.module_name = 'image_model'
        self.features = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=64, kernel_size=11, stride=4), nn.ReLU(inplace=True), nn.LocalResponseNorm(size=2, k=2), nn.ZeroPad2d((0, 1, 0, 1)), nn.MaxPool2d(kernel_size=(3, 3), stride=2), nn.Conv2d(in_channels=64, out_channels=256, kernel_size=5, stride=1, padding=2), nn.ReLU(inplace=True), nn.LocalResponseNorm(size=2, k=2), nn.MaxPool2d(kernel_size=(3, 3), stride=2), nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True), nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True), nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(0, 0)), nn.Conv2d(in_channels=256, out_channels=4096, kernel_size=6), nn.ReLU(inplace=True), nn.Conv2d(in_channels=4096, out_channels=4096, kernel_size=1), nn.ReLU(inplace=True))
        self.classifier = nn.Linear(in_features=4096, out_features=bit)
        self.classifier.weight.data = torch.randn(bit, 4096) * 0.01
        self.classifier.bias.data = torch.randn(bit) * 0.01
        self.mean = torch.zeros(3, 224, 224)
        if pretrain_model:
            self._init(pretrain_model)

    def _init(self, data):
        weights = data['layers'][0]
        self.mean = torch.from_numpy(data['normalization'][0][0][0].transpose()).type(torch.float)
        for k, v in self.features.named_children():
            k = int(k)
            if isinstance(v, nn.Conv2d):
                if k > 1:
                    k -= 1
                v.weight.data = torch.from_numpy(weights[k][0][0][0][0][0].transpose())
                v.bias.data = torch.from_numpy(weights[k][0][0][0][0][1].reshape(-1))

    def forward(self, x):
        if x.is_cuda:
            x = x - self.mean
        else:
            x = x - self.mean
        x = self.features(x)
        x = x.squeeze()
        x = self.classifier(x)
        return x


LAYER1_NODE = 8192


def weights_init(m):
    if type(m) == nn.Conv2d:
        nn.init.normal_(m.weight.data, 0.0, 0.01)
        nn.init.normal_(m.bias.data, 0.0, 0.01)


class TxtModule(BasicModule):

    def __init__(self, y_dim, bit):
        """
        :param y_dim: dimension of tags
        :param bit: bit number of the final binary code
        """
        super(TxtModule, self).__init__()
        self.module_name = 'text_model'
        self.conv1 = nn.Conv2d(1, LAYER1_NODE, kernel_size=(y_dim, 1), stride=(1, 1))
        self.conv2 = nn.Conv2d(LAYER1_NODE, bit, kernel_size=1, stride=(1, 1))
        self.apply(weights_init)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = x.squeeze()
        return x


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (BasicModule,
     lambda: ([], {}),
     lambda: ([], {})),
    (ImgModule,
     lambda: ([], {'bit': 4}),
     lambda: ([torch.rand([4, 3, 224, 224])], {})),
    (TxtModule,
     lambda: ([], {'y_dim': 4, 'bit': 4}),
     lambda: ([torch.rand([4, 1, 64, 64])], {})),
]

