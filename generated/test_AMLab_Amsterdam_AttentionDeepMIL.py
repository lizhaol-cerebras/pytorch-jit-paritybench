
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


import numpy as np


import torch


import torch.utils.data as data_utils


from torchvision import datasets


from torchvision import transforms


import torch.optim as optim


from torch.autograd import Variable


import matplotlib.pyplot as plt


import torch.nn as nn


import torch.nn.functional as F


class Attention(nn.Module):

    def __init__(self):
        super(Attention, self).__init__()
        self.M = 500
        self.L = 128
        self.ATTENTION_BRANCHES = 1
        self.feature_extractor_part1 = nn.Sequential(nn.Conv2d(1, 20, kernel_size=5), nn.ReLU(), nn.MaxPool2d(2, stride=2), nn.Conv2d(20, 50, kernel_size=5), nn.ReLU(), nn.MaxPool2d(2, stride=2))
        self.feature_extractor_part2 = nn.Sequential(nn.Linear(50 * 4 * 4, self.M), nn.ReLU())
        self.attention = nn.Sequential(nn.Linear(self.M, self.L), nn.Tanh(), nn.Linear(self.L, self.ATTENTION_BRANCHES))
        self.classifier = nn.Sequential(nn.Linear(self.M * self.ATTENTION_BRANCHES, 1), nn.Sigmoid())

    def forward(self, x):
        x = x.squeeze(0)
        H = self.feature_extractor_part1(x)
        H = H.view(-1, 50 * 4 * 4)
        H = self.feature_extractor_part2(H)
        A = self.attention(H)
        A = torch.transpose(A, 1, 0)
        A = F.softmax(A, dim=1)
        Z = torch.mm(A, H)
        Y_prob = self.classifier(Z)
        Y_hat = torch.ge(Y_prob, 0.5).float()
        return Y_prob, Y_hat, A

    def calculate_classification_error(self, X, Y):
        Y = Y.float()
        _, Y_hat, _ = self.forward(X)
        error = 1.0 - Y_hat.eq(Y).cpu().float().mean().data.item()
        return error, Y_hat

    def calculate_objective(self, X, Y):
        Y = Y.float()
        Y_prob, _, A = self.forward(X)
        Y_prob = torch.clamp(Y_prob, min=1e-05, max=1.0 - 1e-05)
        neg_log_likelihood = -1.0 * (Y * torch.log(Y_prob) + (1.0 - Y) * torch.log(1.0 - Y_prob))
        return neg_log_likelihood, A


class GatedAttention(nn.Module):

    def __init__(self):
        super(GatedAttention, self).__init__()
        self.M = 500
        self.L = 128
        self.ATTENTION_BRANCHES = 1
        self.feature_extractor_part1 = nn.Sequential(nn.Conv2d(1, 20, kernel_size=5), nn.ReLU(), nn.MaxPool2d(2, stride=2), nn.Conv2d(20, 50, kernel_size=5), nn.ReLU(), nn.MaxPool2d(2, stride=2))
        self.feature_extractor_part2 = nn.Sequential(nn.Linear(50 * 4 * 4, self.M), nn.ReLU())
        self.attention_V = nn.Sequential(nn.Linear(self.M, self.L), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(self.M, self.L), nn.Sigmoid())
        self.attention_w = nn.Linear(self.L, self.ATTENTION_BRANCHES)
        self.classifier = nn.Sequential(nn.Linear(self.M * self.ATTENTION_BRANCHES, 1), nn.Sigmoid())

    def forward(self, x):
        x = x.squeeze(0)
        H = self.feature_extractor_part1(x)
        H = H.view(-1, 50 * 4 * 4)
        H = self.feature_extractor_part2(H)
        A_V = self.attention_V(H)
        A_U = self.attention_U(H)
        A = self.attention_w(A_V * A_U)
        A = torch.transpose(A, 1, 0)
        A = F.softmax(A, dim=1)
        Z = torch.mm(A, H)
        Y_prob = self.classifier(Z)
        Y_hat = torch.ge(Y_prob, 0.5).float()
        return Y_prob, Y_hat, A

    def calculate_classification_error(self, X, Y):
        Y = Y.float()
        _, Y_hat, _ = self.forward(X)
        error = 1.0 - Y_hat.eq(Y).cpu().float().mean().item()
        return error, Y_hat

    def calculate_objective(self, X, Y):
        Y = Y.float()
        Y_prob, _, A = self.forward(X)
        Y_prob = torch.clamp(Y_prob, min=1e-05, max=1.0 - 1e-05)
        neg_log_likelihood = -1.0 * (Y * torch.log(Y_prob) + (1.0 - Y) * torch.log(1.0 - Y_prob))
        return neg_log_likelihood, A

