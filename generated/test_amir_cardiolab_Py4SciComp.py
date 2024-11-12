
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


import torch.nn as nn


import torch.nn.functional as F


import torchvision.transforms as transforms


import torchvision.datasets as dsets


from torch.utils.data import DataLoader


from torch.utils.data import TensorDataset


from torch.utils.data import RandomSampler


import numpy as np


from sklearn.model_selection import train_test_split


import matplotlib.pyplot as plt


from matplotlib import pyplot as plt


from torch.autograd import Variable


import torch.optim as optim


from math import exp


from math import sqrt


from math import pi


import time


import math


from torch import nn


from scipy.io import savemat


class myCNN(nn.Module):

    def __init__(self):
        super(myCNN, self).__init__()
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5)
        self.relu1 = nn.ReLU()
        self.cnn11 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=5)
        self.relu11 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.cnn2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(288, 512)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(512, 28 * 28)

    def forward(self, x):
        out = self.cnn1(x)
        out = self.relu1(out)
        out = self.cnn11(out)
        out = self.relu1(out)
        out = self.maxpool1(out)
        out = self.cnn2(out)
        out = self.relu2(out)
        out = self.maxpool2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.relu3(out)
        out = self.fc2(out)
        return out


class autoencoder(nn.Module):

    def __init__(self):
        super(autoencoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=4, kernel_size=5, stride=1)
        self.maxpool = nn.MaxPool2d(2, 2)
        self.tconv1 = nn.ConvTranspose2d(in_channels=4, out_channels=16, kernel_size=5, stride=1, padding=2)
        self.tconv2 = nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=5, stride=1)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.maxpool(x)
        x = F.relu(self.conv2(x))
        x = self.maxpool(x)
        x = F.relu(self.tconv1(x))
        x = F.sigmoid(self.tconv2(x))
        return x


class DeepAutoencoder_original(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Sequential(torch.nn.Linear(28 * 28, 256), torch.nn.ReLU(), torch.nn.Linear(256, 128), torch.nn.ReLU(), torch.nn.Linear(128, 64), torch.nn.ReLU(), torch.nn.Linear(64, 8))
        self.decoder = torch.nn.Sequential(torch.nn.Linear(8, 32), torch.nn.ReLU(), torch.nn.Linear(32, 64), torch.nn.ReLU(), torch.nn.Linear(64, 28 * 28))

    def forward(self, x):
        x = x.view(x.size(0), -1)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class DeepAutoencoder(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Sequential(torch.nn.Linear(28 * 28, 256), torch.nn.ReLU(), torch.nn.Linear(256, 128), torch.nn.ReLU(), torch.nn.Linear(128, 64), torch.nn.ReLU(), torch.nn.Linear(64, 32))
        self.decoder = torch.nn.Sequential(torch.nn.Linear(32, 44), torch.nn.ReLU(), torch.nn.Linear(44, 64), torch.nn.ReLU(), torch.nn.Linear(64, 80), torch.nn.ReLU(), torch.nn.Linear(80, 28 * 28))

    def forward(self, x):
        x = x.view(x.size(0), -1)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class DeepAutoencoder_deeper(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Sequential(torch.nn.Linear(28 * 28, 512), torch.nn.ReLU(), torch.nn.Linear(512, 256), torch.nn.ReLU(), torch.nn.Linear(256, 128), torch.nn.ReLU(), torch.nn.Linear(128, 64), torch.nn.ReLU(), torch.nn.Linear(64, 32))
        self.decoder = torch.nn.Sequential(torch.nn.Linear(32, 64), torch.nn.ReLU(), torch.nn.Linear(64, 128), torch.nn.ReLU(), torch.nn.Linear(128, 256), torch.nn.ReLU(), torch.nn.Linear(256, 512), torch.nn.ReLU(), torch.nn.Linear(512, 28 * 28))

    def forward(self, x):
        x = x.view(x.size(0), -1)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (autoencoder,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1, 64, 64])], {})),
]

