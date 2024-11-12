
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


import time


import numpy as np


import torch


import torchvision


from torch import nn


from torch import autograd


from torch import optim


import torch.autograd as autograd


import torch.nn as nn


import torch.nn.functional as F


import torch.optim as optim


from sklearn.preprocessing import OneHotEncoder


import matplotlib


import matplotlib.pyplot as plt


import sklearn.datasets


import random


DIM = 512


FIXED_GENERATOR = False


class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()
        main = nn.Sequential(nn.Linear(2, DIM), nn.ReLU(True), nn.Linear(DIM, DIM), nn.ReLU(True), nn.Linear(DIM, DIM), nn.ReLU(True), nn.Linear(DIM, 2))
        self.main = main

    def forward(self, noise, real_data):
        if FIXED_GENERATOR:
            return noise + real_data
        else:
            output = self.main(noise)
            return output


class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()
        main = nn.Sequential(nn.Linear(2, DIM), nn.ReLU(True), nn.Linear(DIM, DIM), nn.ReLU(True), nn.Linear(DIM, DIM), nn.ReLU(True), nn.Linear(DIM, 1))
        self.main = main

    def forward(self, inputs):
        output = self.main(inputs)
        return output.view(-1)


class ResBlock(nn.Module):

    def __init__(self):
        super(ResBlock, self).__init__()
        self.res_block = nn.Sequential(nn.ReLU(True), nn.Conv1d(DIM, DIM, 5, padding=2), nn.ReLU(True), nn.Conv1d(DIM, DIM, 5, padding=2))

    def forward(self, input):
        output = self.res_block(input)
        return input + 0.3 * output

