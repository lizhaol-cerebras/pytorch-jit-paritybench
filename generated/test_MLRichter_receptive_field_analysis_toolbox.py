
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


from typing import Callable


from typing import Dict


from typing import List


from typing import Optional


from typing import Tuple


from typing import Union


import warnings


from collections import Sequence


import numpy as np


from typing import Sequence


from torchvision.models import efficientnet_b0


from torchvision.models.alexnet import alexnet


from torchvision.models.inception import inception_v3


from torchvision.models.mnasnet import mnasnet1_3


from torchvision.models.resnet import resnet18


from torchvision.models.resnet import resnet152


from torchvision.models.vgg import vgg19


class SomeModule(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.k_size = 3
        self.s_size = 1
        self.conv1 = torch.nn.Conv2d(64, 64, kernel_size=self.k_size, stride=self.s_size, padding=1, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.nn.functional.max_pool2d(x, kernel_size=self.k_size * 2, stride=self.s_size * 2, padding=2)
        return x


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (SomeModule,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 64, 64, 64])], {})),
]

