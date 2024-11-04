import sys
_module = sys.modules[__name__]
del sys
main = _module
model = _module
trainer = _module

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


import torch.nn as nn


import torch


import torch.optim as optim


from torch.autograd import Variable


import torch.autograd as autograd


import torchvision.datasets as dset


import torchvision.transforms as transforms


from torch.utils.data import DataLoader


from torchvision.utils import save_image


import numpy as np


class FrontEnd(nn.Module):
    """ front end part of discriminator and Q"""

    def __init__(self):
        super(FrontEnd, self).__init__()
        self.main = nn.Sequential(nn.Conv2d(1, 64, 4, 2, 1), nn.LeakyReLU(0.1, inplace=True), nn.Conv2d(64, 128, 4, 2, 1, bias=False), nn.BatchNorm2d(128), nn.LeakyReLU(0.1, inplace=True), nn.Conv2d(128, 1024, 7, bias=False), nn.BatchNorm2d(1024), nn.LeakyReLU(0.1, inplace=True))

    def forward(self, x):
        output = self.main(x)
        return output


class D(nn.Module):

    def __init__(self):
        super(D, self).__init__()
        self.main = nn.Sequential(nn.Conv2d(1024, 1, 1), nn.Sigmoid())

    def forward(self, x):
        output = self.main(x).view(-1, 1)
        return output


class Q(nn.Module):

    def __init__(self):
        super(Q, self).__init__()
        self.conv = nn.Conv2d(1024, 128, 1, bias=False)
        self.bn = nn.BatchNorm2d(128)
        self.lReLU = nn.LeakyReLU(0.1, inplace=True)
        self.conv_disc = nn.Conv2d(128, 10, 1)
        self.conv_mu = nn.Conv2d(128, 2, 1)
        self.conv_var = nn.Conv2d(128, 2, 1)

    def forward(self, x):
        y = self.conv(x)
        disc_logits = self.conv_disc(y).squeeze()
        mu = self.conv_mu(y).squeeze()
        var = self.conv_var(y).squeeze().exp()
        return disc_logits, mu, var


class G(nn.Module):

    def __init__(self):
        super(G, self).__init__()
        self.main = nn.Sequential(nn.ConvTranspose2d(74, 1024, 1, 1, bias=False), nn.BatchNorm2d(1024), nn.ReLU(True), nn.ConvTranspose2d(1024, 128, 7, 1, bias=False), nn.BatchNorm2d(128), nn.ReLU(True), nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(True), nn.ConvTranspose2d(64, 1, 4, 2, 1, bias=False), nn.Sigmoid())

    def forward(self, x):
        output = self.main(x)
        return output


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (D,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1024, 64, 64])], {}),
     True),
    (FrontEnd,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1, 64, 64])], {}),
     True),
    (G,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 74, 4, 4])], {}),
     True),
    (Q,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1024, 64, 64])], {}),
     True),
]

class Test_pianomania_infoGAN_pytorch(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

    def test_003(self):
        self._check(*TESTCASES[3])

