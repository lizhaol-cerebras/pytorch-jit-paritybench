
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


from typing import Any


import torch


from torch import Tensor


import torchvision.models as models


import torchvision.transforms as transforms


import torch.nn as nn


from typing import Union


from typing import List


from typing import Tuple


import numpy as np


import math


from typing import cast


import warnings


from typing import Optional


from typing import Callable


from typing import Dict


from typing import NamedTuple


import functools


import copy


class RandomizedResNet18(torch.nn.Module):

    def __init__(self) ->None:
        super().__init__()
        self.model = models.resnet18(pretrained=True)
        self.transforms = transforms.RandomRotation(degrees=25)

    def forward(self, x: 'Tensor') ->Any:
        x = self.transforms(x)
        return self.model(x)


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (RandomizedResNet18,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {})),
]

