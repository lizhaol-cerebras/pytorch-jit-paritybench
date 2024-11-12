
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


from sklearn import manifold


from sklearn import datasets


from sklearn.cluster import KMeans


from sklearn import neighbors


import torch


import numpy as np


import torch.utils.data as data


from torchvision import datasets


from torchvision import transforms


import torch.nn.functional as F


from itertools import combinations


import matplotlib


import pandas as pd


from torch import nn


import torch.nn as nn


class Discriminator(nn.Module):
    """Discriminator model for source domain."""

    def __init__(self, input_dims, hidden_dims, output_dims):
        """Init discriminator."""
        super(Discriminator, self).__init__()
        self.restored = False
        self.layer = nn.Sequential(nn.Linear(input_dims, hidden_dims), nn.ReLU(), nn.Linear(hidden_dims, hidden_dims), nn.ReLU(), nn.Linear(hidden_dims, output_dims))

    def forward(self, input):
        """Forward the discriminator."""
        out = self.layer(input)
        return out


class EmbeddingNet(nn.Module):

    def __init__(self, n_outputs=128):
        super(EmbeddingNet, self).__init__()
        self.convnet = nn.Sequential(nn.Conv2d(1, 20, kernel_size=5), nn.MaxPool2d(kernel_size=2), nn.ReLU(), nn.Conv2d(20, 50, kernel_size=5), nn.MaxPool2d(kernel_size=2), nn.ReLU())
        self.n_classes = 10
        self.n_outputs = n_outputs
        self.fc = nn.Sequential(nn.Linear(50 * 4 * 4, 500), nn.ReLU(), nn.Linear(500, self.n_outputs))

    def extract_features(self, x):
        output = self.convnet(x)
        output = output.view(output.size()[0], -1)
        output = self.fc[0](output)
        return output

    def forward(self, x):
        output = self.convnet(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output

    def get_embedding(self, x):
        return self.forward(x)


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (Discriminator,
     lambda: ([], {'input_dims': 4, 'hidden_dims': 4, 'output_dims': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
]

