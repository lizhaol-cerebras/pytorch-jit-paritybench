
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


import torchvision.transforms.functional as FT


from torch.utils.data import Dataset


from torchvision.transforms import Compose


from torchvision.transforms import CenterCrop


from torchvision.transforms import ToTensor


from torchvision.transforms import Normalize


from torchvision.transforms import Resize


from torchvision.transforms import InterpolationMode


import time


import numpy as np


from typing import List


from typing import Tuple


from typing import Dict


import torch.nn.functional as F


from torch.utils.data import DataLoader


import functools


import random


from typing import Optional


from typing import Union


from typing import Literal


import torch.nn as nn


import math


from typing import Set


import logging


class Phi(nn.Module):
    """
    Textual Inversion Phi network.
    Takes as input the visual features of an image and outputs the pseudo-work embedding.
    Copy-paste from https://github.com/miccunifi/SEARLE/blob/main/src/phi.py
    """

    def __init__(self, input_dim: 'int', hidden_dim: 'int', output_dim: 'int', dropout: 'int'):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Dropout(p=dropout), nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(p=dropout), nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        return self.layers(x)


class PIC2WORD(nn.Module):

    def __init__(self, embed_dim=512, middle_dim=512, output_dim=512, n_layer=2, dropout=0.1):
        super().__init__()
        self.fc_out = nn.Linear(middle_dim, output_dim)
        layers = []
        dim = embed_dim
        for _ in range(n_layer):
            block = []
            block.append(nn.Linear(dim, middle_dim))
            block.append(nn.Dropout(dropout))
            block.append(nn.ReLU())
            dim = middle_dim
            layers.append(nn.Sequential(*block))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: 'torch.Tensor'):
        for layer in self.layers:
            x = layer(x)
        return self.fc_out(x)


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (Phi,
     lambda: ([], {'input_dim': 4, 'hidden_dim': 4, 'output_dim': 4, 'dropout': 0.5}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
]

