import sys
_module = sys.modules[__name__]
del sys
mnist_example = _module
mnist_loader_example = _module
setup = _module
multi_input_multi_target = _module
single_input_multi_target = _module
single_input_single_target = _module
simple_multi_input_multi_target = _module
simple_multi_input_no_target = _module
simple_multi_input_single_target = _module
single_input_multi_target = _module
single_input_no_target = _module
single_input_single_target = _module
test_metrics = _module
test_affine_transforms = _module
test_image_transforms = _module
test_tensor_transforms = _module
utils = _module
torchsample = _module
callbacks = _module
constraints = _module
datasets = _module
functions = _module
affine = _module
initializers = _module
metrics = _module
modules = _module
_utils = _module
module_trainer = _module
regularizers = _module
samplers = _module
transforms = _module
affine_transforms = _module
distortion_transforms = _module
image_transforms = _module
tensor_transforms = _module
utils = _module
version = _module

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


import torch as th


import torch.nn as nn


import torch.nn.functional as F


from torchvision import datasets


from torch.utils.data import DataLoader


import torch


from torch.autograd import Variable


import numpy as np


from collections import OrderedDict


from collections import Iterable


import warnings


import time


import pandas as pd


import torch.nn.init


import torch.optim as optim


import functools


import math


import random


class Network(nn.Module):

    def __init__(self):
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(1600, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 1600)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x)

