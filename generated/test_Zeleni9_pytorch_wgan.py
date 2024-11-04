import sys
_module = sys.modules[__name__]
del sys
main = _module
models = _module
dcgan = _module
gan = _module
wgan_clipping = _module
wgan_gradient_penalty = _module
utils = _module
config = _module
data_loader = _module
fashion_mnist = _module
feature_extraction_test = _module
inception_score = _module
tensorboard_logger = _module

from _paritybench_helpers import _mock_config, patch_functional
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


from torch.autograd import Variable


import time as t


from itertools import chain


from torchvision import utils


import time


import matplotlib.pyplot as plt


import torch.optim as optim


from torch import autograd


import torchvision.datasets as dset


import torchvision.transforms as transforms


import torch.utils.data as data_utils


import torch.utils.data as data


import torchvision.models as models


from sklearn.metrics import accuracy_score


from sklearn.linear_model import LogisticRegression


from torch import nn


from torch.nn import functional as F


import torch.utils.data


from torchvision.models.inception import inception_v3


import numpy as np


from scipy.stats import entropy


class Generator(torch.nn.Module):

    def __init__(self, channels):
        super().__init__()
        self.main_module = nn.Sequential(nn.ConvTranspose2d(in_channels=100, out_channels=1024, kernel_size=4, stride=1, padding=0), nn.BatchNorm2d(num_features=1024), nn.ReLU(True), nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=4, stride=2, padding=1), nn.BatchNorm2d(num_features=512), nn.ReLU(True), nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1), nn.BatchNorm2d(num_features=256), nn.ReLU(True), nn.ConvTranspose2d(in_channels=256, out_channels=channels, kernel_size=4, stride=2, padding=1))
        self.output = nn.Tanh()

    def forward(self, x):
        x = self.main_module(x)
        return self.output(x)


class Discriminator(torch.nn.Module):

    def __init__(self, channels):
        super().__init__()
        self.main_module = nn.Sequential(nn.Conv2d(in_channels=channels, out_channels=256, kernel_size=4, stride=2, padding=1), nn.InstanceNorm2d(256, affine=True), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1), nn.InstanceNorm2d(512, affine=True), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=4, stride=2, padding=1), nn.InstanceNorm2d(1024, affine=True), nn.LeakyReLU(0.2, inplace=True))
        self.output = nn.Sequential(nn.Conv2d(in_channels=1024, out_channels=1, kernel_size=4, stride=1, padding=0))

    def forward(self, x):
        x = self.main_module(x)
        return self.output(x)

    def feature_extraction(self, x):
        x = self.main_module(x)
        return x.view(-1, 1024 * 4 * 4)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Generator,
     lambda: ([], {'channels': 4}),
     lambda: ([torch.rand([4, 100, 4, 4])], {}),
     True),
]

class Test_Zeleni9_pytorch_wgan(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

