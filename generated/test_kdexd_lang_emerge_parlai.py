
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


import math


import torch


from torch import nn


from torch.autograd import Variable


from torch.autograd import backward as autograd_backward


from functools import reduce


import random


from torch.utils.data import Dataset


from torch.utils.data import DataLoader


import numpy as np


from torch import optim


def xavier_init(module):
    """Xavier initializer for module parameters."""
    for parameter in module.parameters():
        if len(parameter.data.shape) == 1:
            parameter.data.fill_(0)
        else:
            fan_in = parameter.data.size(0)
            fan_out = parameter.data.size(1)
            parameter.data.normal_(0, math.sqrt(2 / (fan_in + fan_out)))
    return module

