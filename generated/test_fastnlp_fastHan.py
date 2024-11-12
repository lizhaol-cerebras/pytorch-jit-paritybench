
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


import copy


import torch


import pandas as pd


from torch.utils.data.dataloader import DataLoader


from torch.utils.data.dataloader import SequentialSampler


from torch.utils.data.dataloader import RandomSampler


import warnings


import torch.nn as nn


import torch.nn.functional as F


from torch.nn import CrossEntropyLoss


import re


from torch import nn


class PrefixEncoder(torch.nn.Module):
    """
    The torch.nn model to encode the prefix
    Input shape: (batch-size, prefix-length)
    Output shape: (batch-size, prefix-length, 2*layers*hidden)
    """

    def __init__(self, config, num_tokens, prefix_projection, pre_seq_len, prefix_hidden_size=500):
        super().__init__()
        self.prefix_projection = prefix_projection
        if self.prefix_projection:
            self.embedding = torch.nn.Embedding(num_tokens, config.hidden_size)
            self.trans = torch.nn.Sequential(torch.nn.Linear(config.hidden_size, prefix_hidden_size), torch.nn.Tanh(), torch.nn.Linear(prefix_hidden_size, config.num_hidden_layers * 2 * config.hidden_size))
        else:
            self.embedding = torch.nn.Embedding(num_tokens, config.num_hidden_layers * 2 * config.hidden_size)

    def forward(self, prefix: 'torch.Tensor'):
        if self.prefix_projection:
            prefix_tokens = self.embedding(prefix)
            past_key_values = self.trans(prefix_tokens)
        else:
            past_key_values = self.embedding(prefix)
        return past_key_values

