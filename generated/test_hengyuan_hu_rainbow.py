import sys
_module = sys.modules[__name__]
del sys
core = _module
dqn = _module
env = _module
logger = _module
main = _module
model = _module
policy = _module
train = _module
utils = _module

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


import random


import torch


import numpy as np


import copy


import torch.nn as nn


from torch.autograd import Variable


import math


import time


import torch.nn


class BasicNetwork(nn.Module):

    def __init__(self, conv, fc):
        super(BasicNetwork, self).__init__()
        self.conv = conv
        self.fc = fc

    def forward(self, x):
        assert x.data.max() <= 1.0
        batch = x.size(0)
        y = self.conv(x)
        y = y.view(batch, -1)
        y = self.fc(y)
        return y


class DuelingNetwork(nn.Module):

    def __init__(self, conv, adv, val):
        super(DuelingNetwork, self).__init__()
        self.conv = conv
        self.adv = adv
        self.val = val

    def forward(self, x):
        assert x.data.max() <= 1.0
        batch = x.size(0)
        feat = self.conv(x)
        feat = feat.view(batch, -1)
        adv = self.adv(feat)
        val = self.val(feat)
        q = val - adv.mean(1, keepdim=True) + adv
        return q


class DistributionalBasicNetwork(nn.Module):

    def __init__(self, conv, fc, num_actions, num_atoms):
        super(DistributionalBasicNetwork, self).__init__()
        self.conv = conv
        self.fc = fc
        self.num_actions = num_actions
        self.num_atoms = num_atoms

    def forward(self, x):
        batch = x.size(0)
        y = self.conv(x)
        y = y.view(batch, -1)
        y = self.fc(y)
        logits = y.view(batch, self.num_actions, self.num_atoms)
        probs = nn.functional.softmax(logits, 2)
        return probs


class NoisyLinear(nn.Module):
    """Factorised Gaussian NoisyNet"""

    def __init__(self, in_features, out_features, sigma0):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.noisy_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.noisy_bias = nn.Parameter(torch.Tensor(out_features))
        self.reset_parameters()
        self.noise_std = sigma0 / math.sqrt(self.in_features)
        self.in_noise = torch.FloatTensor(in_features)
        self.out_noise = torch.FloatTensor(out_features)
        self.noise = None
        self.sample_noise()

    def sample_noise(self):
        self.in_noise.normal_(0, self.noise_std)
        self.out_noise.normal_(0, self.noise_std)
        self.noise = torch.mm(self.out_noise.view(-1, 1), self.in_noise.view(1, -1))

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.noisy_weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
            self.noisy_bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        normal_y = nn.functional.linear(x, self.weight, self.bias)
        if not x.volatile:
            self.sample_noise()
        noisy_weight = self.noisy_weight * Variable(self.noise)
        noisy_bias = self.noisy_bias * Variable(self.out_noise)
        noisy_y = nn.functional.linear(x, noisy_weight, noisy_bias)
        return noisy_y + normal_y

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'in_features=' + str(self.in_features) + ', out_features=' + str(self.out_features) + ')'


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BasicNetwork,
     lambda: ([], {'conv': _mock_layer(), 'fc': _mock_layer()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (DistributionalBasicNetwork,
     lambda: ([], {'conv': _mock_layer(), 'fc': _mock_layer(), 'num_actions': 4, 'num_atoms': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     True),
    (DuelingNetwork,
     lambda: ([], {'conv': _mock_layer(), 'adv': _mock_layer(), 'val': _mock_layer()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (NoisyLinear,
     lambda: ([], {'in_features': 4, 'out_features': 4, 'sigma0': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_hengyuan_hu_rainbow(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

    def test_003(self):
        self._check(*TESTCASES[3])

