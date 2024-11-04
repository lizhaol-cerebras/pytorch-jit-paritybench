import sys
_module = sys.modules[__name__]
del sys
conf = _module
mnist = _module
viz_optimizers = _module
setup = _module
conftest = _module
test_basic = _module
test_optimizer = _module
test_optimizer_with_nn = _module
test_param_validation = _module
utils = _module
torch_optimizer = _module
a2grad = _module
accsgd = _module
adabelief = _module
adabound = _module
adafactor = _module
adahessian = _module
adamod = _module
adamp = _module
aggmo = _module
apollo = _module
diffgrad = _module
lamb = _module
lars = _module
lookahead = _module
madgrad = _module
novograd = _module
pid = _module
qhadam = _module
qhm = _module
radam = _module
sgdp = _module
sgdw = _module
shampoo = _module
swats = _module
types = _module
yogi = _module

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


import torch.nn.functional as F


from torch.optim.lr_scheduler import StepLR


from torch.utils.tensorboard import SummaryWriter


from torchvision import datasets


from torchvision import transforms


from torchvision import utils


import math


import matplotlib.pyplot as plt


import numpy as np


import re


import functools


from copy import deepcopy


from torch.autograd import Variable


from torch.optim.lr_scheduler import ExponentialLR


from torch.optim.lr_scheduler import ReduceLROnPlateau


from torch import nn


from typing import Dict


from typing import List


from typing import Type


from torch.optim.optimizer import Optimizer


import copy


from typing import Optional


from typing import Any


from typing import Tuple


from typing import TypeVar


from typing import Union


from collections import defaultdict


from typing import Callable


import torch.optim


import warnings


from typing import Iterable


from torch import Tensor


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


class LogisticRegression(nn.Module):

    def __init__(self):
        super(LogisticRegression, self).__init__()
        self.linear1 = nn.Linear(2, 4)
        self.linear2 = nn.Linear(4, 1)

    def forward(self, x):
        output = torch.relu(self.linear1(x))
        output = self.linear2(output)
        y_pred = torch.sigmoid(output)
        return y_pred

