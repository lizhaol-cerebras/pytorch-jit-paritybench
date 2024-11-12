
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


import pandas as pd


import random


import copy


import itertools


import torch


import torch.nn as nn


import torchvision.models as models


import torch.optim as optim


from torchvision import transforms


from torch.utils.data import DataLoader


from torch.utils.data import Dataset


from typing import Dict


import logging


from torch import Tensor


from torch.optim import SGD


from torch.utils.data import TensorDataset


from itertools import product


from torch import LongTensor


from torch.nn import CrossEntropyLoss


from torch.optim import Adam


from typing import List


from typing import Union


from typing import Tuple


from sklearn.feature_extraction.text import TfidfVectorizer


import scipy.sparse as sp


from torch.nn.utils.rnn import pack_padded_sequence


from torch.nn.utils.rnn import pad_packed_sequence


from torch import nn


from typing import Callable


from torch.optim.optimizer import Optimizer


from scipy.sparse import csr_matrix


from sklearn.neighbors import NearestNeighbors


from abc import ABC


from abc import abstractmethod


import torch.nn.functional as F


from sklearn.metrics import classification_report


from torch.nn import Module


from torch.nn.modules.loss import _Loss


from torch import argmax


import matplotlib.pyplot as plt


from torch.optim import Optimizer


from copy import copy


import warnings


from scipy import sparse as ss


from torch.nn import BCEWithLogitsLoss


from numpy.testing import assert_array_equal


from torch import equal


class BidirectionalLSTM(nn.Module):

    def __init__(self, word_input_dim, word_output_dim, word_embedding_matrix, num_classes, size_factor=200):
        super(BidirectionalLSTM, self).__init__()
        self.word_input_dim = word_input_dim
        self.word_output_dim = word_output_dim
        self.word_embedding_matrix = word_embedding_matrix
        self.size_factor = size_factor
        self.num_classes = num_classes
        self.word_embedding = nn.Embedding(word_input_dim, word_output_dim, padding_idx=0)
        self.word_embedding.weight = nn.Parameter(torch.tensor(word_embedding_matrix, dtype=torch.float32))
        self.word_embedding.weight.requires_grad = False
        self.type_linear = nn.Linear(20, size_factor * 2)
        self.td_dense = nn.Linear(word_output_dim, size_factor)
        self.biLSTM = nn.LSTM(size_factor, size_factor, bidirectional=True, batch_first=True)
        self.predict = nn.Linear(size_factor * 2, num_classes)
        self.init_weights()

    def forward(self, x):
        word_embeddings = self.word_embedding(x)
        td_dense = self.td_dense(word_embeddings)
        biLSTM, (h_n, c_n) = self.biLSTM(td_dense)
        self.biLSTM.flatten_parameters()
        final_state = h_n.view(1, 2, x.shape[0], self.size_factor)[-1]
        h_1, h_2 = final_state[0], final_state[1]
        concat = torch.cat((h_1, h_2), 1)
        final = self.predict(concat)
        return final

    def init_weights(self):
        """
        Here we reproduce Keras default initialization weights to initialize Embeddings/LSTM weights
        """
        torch.manual_seed(12345)
        ih = (param.data for name, param in self.named_parameters() if 'weight_ih' in name)
        hh = (param.data for name, param in self.named_parameters() if 'weight_hh' in name)
        b = (param.data for name, param in self.named_parameters() if 'bias' in name)
        for t in ih:
            nn.init.xavier_uniform_(t)
        for t in hh:
            nn.init.orthogonal_(t)
        for t in b:
            nn.init.constant_(t, 0)


class LogisticRegressionModel(nn.Module):

    def __init__(self, input_dim: 'int', output_classes: 'int'):
        super(LogisticRegressionModel, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_classes)

    def forward(self, x):
        x = x.float()
        outputs = self.linear(x)
        return outputs


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (LogisticRegressionModel,
     lambda: ([], {'input_dim': 4, 'output_classes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
]

