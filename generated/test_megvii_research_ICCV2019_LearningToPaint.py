import sys
_module = sys.modules[__name__]
del sys
actor = _module
critic = _module
ddpg = _module
evaluator = _module
multi = _module
rpm = _module
wgan = _module
Renderer = _module
model = _module
stroke_gen = _module
env = _module
test = _module
train = _module
train_renderer = _module
tensorboard = _module
util = _module
actor = _module
critic = _module
ddpg = _module
multi = _module
rpm = _module
wgan = _module
model = _module
env = _module
test = _module
train = _module
train_renderer = _module
util = _module
predict = _module

from paritybench._paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, matplotlib, numbers, numpy, pandas, queue, random, re, scipy, sklearn, string, tensorflow, time, torch, torchaudio, torchvision, types, typing, uuid, warnings
import numpy as np
from torch import Tensor
patch_functional()
open = mock_open()
yaml = logging = sys = argparse = MagicMock()
ArgumentParser = argparse.ArgumentParser
_global_config = args = argv = cfg = config = params = _mock_config()
argparse.ArgumentParser.return_value.parse_args.return_value = _global_config
yaml.load.return_value = _global_config
sys.argv = _global_config
__version__ = '1.0.0'
xrange = range
wraps = functools.wraps


import numpy as np


import torch


import torch.nn as nn


import torch.nn.functional as F


import torch.nn.utils.weight_norm as weightNorm


from torch.autograd import Variable


from torch.optim import Adam


from torch.optim import SGD


import random


from torch import autograd


from torch.autograd import grad as torch_grad


import torchvision.transforms as transforms


from torchvision import transforms


from torchvision import utils


import time


import torch.optim as optim


class TReLU(nn.Module):

    def __init__(self):
        super(TReLU, self).__init__()
        self.alpha = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.alpha.data.fill_(0)

    def forward(self, x):
        x = F.relu(x - self.alpha) + self.alpha
        return x


def conv3x3(in_planes, out_planes, stride=1):
    return weightNorm(nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True))


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.conv2 = conv3x3(planes, planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(weightNorm(nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=True)))
        self.relu_1 = TReLU()
        self.relu_2 = TReLU()

    def forward(self, x):
        out = self.relu_1(self.conv1(x))
        out = self.conv2(out)
        out += self.shortcut(x)
        out = self.relu_2(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = weightNorm(nn.Conv2d(in_planes, planes, kernel_size=1, bias=True))
        self.conv2 = weightNorm(nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True))
        self.conv3 = weightNorm(nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=True))
        self.relu_1 = TReLU()
        self.relu_2 = TReLU()
        self.relu_3 = TReLU()
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(weightNorm(nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=True)))

    def forward(self, x):
        out = self.relu_1(self.conv1(x))
        out = self.relu_2(self.conv2(out))
        out = self.conv3(out)
        out += self.shortcut(x)
        out = self.relu_3(out)
        return out


class ResNet(nn.Module):

    def __init__(self, num_inputs, depth, num_outputs):
        super(ResNet, self).__init__()
        self.in_planes = 64
        block, num_blocks = cfg(depth)
        self.conv1 = conv3x3(num_inputs, 64, 2)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=2)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.fc = nn.Linear(512, num_outputs)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.avg_pool2d(x, 4)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = torch.sigmoid(x)
        return x


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ResNet_wobn(nn.Module):

    def __init__(self, num_inputs, depth, num_outputs):
        super(ResNet_wobn, self).__init__()
        self.in_planes = 64
        block, num_blocks = cfg(depth)
        self.conv0 = conv3x3(num_inputs, 32, 2)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=2)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=1)
        self.conv4 = weightNorm(nn.Conv2d(512, 1, 1, 1, 0))
        self.relu_1 = TReLU()
        self.conv1 = weightNorm(nn.Conv2d(65 + 2, 64, 1, 1, 0))
        self.conv2 = weightNorm(nn.Conv2d(64, 64, 1, 1, 0))
        self.conv3 = weightNorm(nn.Conv2d(64, 32, 1, 1, 0))
        self.relu_2 = TReLU()
        self.relu_3 = TReLU()
        self.relu_4 = TReLU()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def a2img(self, x):
        tmp = coord.expand(x.shape[0], 2, 64, 64)
        x = x.repeat(64, 64, 1, 1).permute(2, 3, 0, 1)
        x = self.relu_2(self.conv1(torch.cat([x, tmp], 1)))
        x = self.relu_3(self.conv2(x))
        x = self.relu_4(self.conv3(x))
        return x

    def forward(self, input):
        x, a = input
        a = self.a2img(a)
        x = self.relu_1(self.conv0(x))
        x = torch.cat([x, a], 1)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.conv4(x)
        return x.view(x.size(0), 64)


class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv0 = weightNorm(nn.Conv2d(6, 16, 5, 2, 2))
        self.conv1 = weightNorm(nn.Conv2d(16, 32, 5, 2, 2))
        self.conv2 = weightNorm(nn.Conv2d(32, 64, 5, 2, 2))
        self.conv3 = weightNorm(nn.Conv2d(64, 128, 5, 2, 2))
        self.conv4 = weightNorm(nn.Conv2d(128, 1, 1, 1, 0))
        self.relu0 = TReLU()
        self.relu1 = TReLU()
        self.relu2 = TReLU()
        self.relu3 = TReLU()

    def forward(self, x):
        x = self.conv0(x)
        x = self.relu0(x)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        x = x.view(-1, 64)
        return x


class FCN(nn.Module):

    def __init__(self):
        super(FCN, self).__init__()
        self.fc1 = nn.Linear(10, 512)
        self.fc2 = nn.Linear(512, 1024)
        self.fc3 = nn.Linear(1024, 2048)
        self.fc4 = nn.Linear(2048, 4096)
        self.conv1 = nn.Conv2d(16, 32, 3, 1, 1)
        self.conv2 = nn.Conv2d(32, 32, 3, 1, 1)
        self.conv3 = nn.Conv2d(8, 16, 3, 1, 1)
        self.conv4 = nn.Conv2d(16, 16, 3, 1, 1)
        self.conv5 = nn.Conv2d(4, 8, 3, 1, 1)
        self.conv6 = nn.Conv2d(8, 4, 3, 1, 1)
        self.pixel_shuffle = nn.PixelShuffle(2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = x.view(-1, 16, 16, 16)
        x = F.relu(self.conv1(x))
        x = self.pixel_shuffle(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pixel_shuffle(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.pixel_shuffle(self.conv6(x))
        x = torch.sigmoid(x)
        return 1 - x.view(-1, 128, 128)


import torch
from torch.nn import MSELoss, ReLU
from paritybench._paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BasicBlock,
     lambda: ([], {'in_planes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Bottleneck,
     lambda: ([], {'in_planes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Discriminator,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 6, 64, 64])], {}),
     False),
    (TReLU,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_megvii_research_ICCV2019_LearningToPaint(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

    def test_003(self):
        self._check(*TESTCASES[3])

