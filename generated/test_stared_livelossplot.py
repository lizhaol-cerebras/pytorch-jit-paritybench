
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


import matplotlib.pyplot as plt


import numpy as np


from matplotlib.colors import ListedColormap


import torch


from torch import nn


from torch import optim


from torch.utils.data import TensorDataset


from torch.utils.data import DataLoader


class Network(nn.Sequential):

    def __init__(self):
        super().__init__(nn.Linear(12, 5))


class Model(nn.Sequential):

    def __init__(self):
        super().__init__(nn.Linear(12, 5))

