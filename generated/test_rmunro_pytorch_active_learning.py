
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


import torch.optim as optim


import random


import math


import re


from random import shuffle


from collections import defaultdict


import copy


class SimpleTextClassifier(nn.Module):
    """Text Classifier with 1 hidden layer 

    """

    def __init__(self, num_labels, vocab_size):
        super(SimpleTextClassifier, self).__init__()
        self.linear1 = nn.Linear(vocab_size, 128)
        self.linear2 = nn.Linear(128, num_labels)

    def forward(self, feature_vec):
        hidden1 = self.linear1(feature_vec).clamp(min=0)
        output = self.linear2(hidden1)
        return F.log_softmax(output, dim=1)


class SimpleUncertaintyPredictor(nn.Module):
    """Simple model to predict whether an item will be classified correctly    

    """

    def __init__(self, vocab_size):
        super(SimpleUncertaintyPredictor, self).__init__()
        self.linear = nn.Linear(vocab_size, 2)

    def forward(self, feature_vec, return_all_layers=False):
        output = self.linear(feature_vec).clamp(min=-1)
        log_softmax = F.log_softmax(output, dim=1)
        if return_all_layers:
            return [output, log_softmax]
        else:
            return log_softmax


class AdvancedUncertaintyPredictor(nn.Module):
    """Simple model to predict whether an item will be classified correctly    

    """

    def __init__(self, vocab_size):
        super(AdvancedUncertaintyPredictor, self).__init__()
        self.linear1 = nn.Linear(vocab_size, 128)
        self.linear2 = nn.Linear(128, num_labels)

    def forward(self, feature_vec, return_all_layers=False):
        hidden1 = self.linear1(feature_vec).clamp(min=0)
        output = self.linear2(hidden1)
        log_softmax = F.log_softmax(output, dim=1)
        if return_all_layers:
            return [hidden1, output, log_softmax]
        else:
            return log_softmax


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (SimpleTextClassifier,
     lambda: ([], {'num_labels': 4, 'vocab_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (SimpleUncertaintyPredictor,
     lambda: ([], {'vocab_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
]

