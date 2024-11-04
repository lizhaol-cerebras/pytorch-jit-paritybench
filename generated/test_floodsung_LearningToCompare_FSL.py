import sys
_module = sys.modules[__name__]
del sys
proc_images = _module
miniimagenet_test_few_shot = _module
miniimagenet_test_one_shot = _module
miniimagenet_train_few_shot = _module
miniimagenet_train_one_shot = _module
task_generator = _module
task_generator_test = _module
omniglot_test_few_shot = _module
omniglot_test_one_shot = _module
omniglot_train_few_shot = _module
omniglot_train_one_shot = _module
task_generator = _module

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


import torch


import torch.nn as nn


import torch.nn.functional as F


from torch.autograd import Variable


from torch.optim.lr_scheduler import StepLR


import numpy as np


import math


import scipy as sp


import scipy.stats


import torchvision


import torchvision.datasets as dset


import torchvision.transforms as transforms


from torch.utils.data import DataLoader


from torch.utils.data import Dataset


import random


import matplotlib.pyplot as plt


from torch.utils.data.sampler import Sampler


class CNNEncoder(nn.Module):
    """docstring for ClassName"""

    def __init__(self):
        super(CNNEncoder, self).__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=3, padding=0), nn.BatchNorm2d(64, momentum=1, affine=True), nn.ReLU(), nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=0), nn.BatchNorm2d(64, momentum=1, affine=True), nn.ReLU(), nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64, momentum=1, affine=True), nn.ReLU())
        self.layer4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64, momentum=1, affine=True), nn.ReLU())

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        return out


class RelationNetwork(nn.Module):
    """docstring for RelationNetwork"""

    def __init__(self, input_size, hidden_size):
        super(RelationNetwork, self).__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64, momentum=1, affine=True), nn.ReLU(), nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64, momentum=1, affine=True), nn.ReLU(), nn.MaxPool2d(2))
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.sigmoid(self.fc2(out))
        return out


import torch
from torch.nn import MSELoss, ReLU
from paritybench._paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (CNNEncoder,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1, 64, 64])], {}),
     False),
]

class Test_floodsung_LearningToCompare_FSL(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

