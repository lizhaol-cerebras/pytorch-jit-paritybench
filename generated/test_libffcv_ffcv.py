
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


from typing import List


import time


import numpy as np


import torch as ch


from torch.cuda.amp import GradScaler


from torch.cuda.amp import autocast


from torch.nn import CrossEntropyLoss


from torch.nn import Conv2d


from torch.nn import BatchNorm2d


from torch.optim import SGD


from torch.optim import lr_scheduler


import torchvision


from torch.utils.data import TensorDataset


from torch.utils.data import DataLoader


import logging


from time import sleep


from time import time


from torch.utils.data import Dataset


from typing import Callable


from typing import TYPE_CHECKING


from typing import Tuple


from typing import Type


import warnings


from collections import defaultdict


from queue import Queue


from queue import Full


from typing import Sequence


import enum


from re import sub


from typing import Any


from typing import Mapping


from typing import Union


from typing import Literal


from collections.abc import Collection


from enum import Enum


from enum import unique


from enum import auto


from typing import Optional


from typing import Dict


from typing import Set


from abc import ABC


from abc import abstractmethod


import torch.nn.functional as F


from numpy.random import permutation


from numpy.random import rand


from collections.abc import Sequence


from numpy import dtype


import random


from torch.utils.data import DistributedSampler


import uuid


from torchvision import transforms as tvt


from torchvision.datasets import CIFAR10


from torchvision.utils import save_image


from torchvision.utils import make_grid


from torch.utils.data import Subset


import string


from typing import Counter


from torch.utils.data import distributed


from torch.multiprocessing import spawn


from torch.multiprocessing import Queue


from torch.distributed import init_process_group


from numpy.random import shuffle


class Mul(ch.nn.Module):

    def __init__(self, weight):
        super(Mul, self).__init__()
        self.weight = weight

    def forward(self, x):
        return x * self.weight


class Flatten(ch.nn.Module):

    def forward(self, x):
        return x.view(x.size(0), -1)


class Residual(ch.nn.Module):

    def __init__(self, module):
        super(Residual, self).__init__()
        self.module = module

    def forward(self, x):
        return x + self.module(x)


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (Flatten,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Mul,
     lambda: ([], {'weight': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Residual,
     lambda: ([], {'module': torch.nn.ReLU()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
]

