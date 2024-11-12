
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


from torch.nn.modules.loss import _Loss


import re


import string


import copy


import random


import numpy as np


from torch.utils.data import DataLoader


import torch.utils.data


from torch import optim


import torch.nn as nn


from torch.nn import functional as F


class Criterion(_Loss):

    def __init__(self, way=2, shot=5):
        super(Criterion, self).__init__()
        self.amount = way * shot

    def forward(self, probs, target):
        target = target[self.amount:]
        target_onehot = torch.zeros_like(probs)
        target_onehot = target_onehot.scatter(1, target.reshape(-1, 1), 1)
        loss = torch.mean((probs - target_onehot) ** 2)
        pred = torch.argmax(probs, dim=1)
        acc = torch.sum(target == pred).float() / target.shape[0]
        return loss, acc


class Encoder(nn.Module):

    def __init__(self, num_classes, num_support_per_class, vocab_size, embed_size, hidden_size, output_dim, weights):
        super(Encoder, self).__init__()
        self.num_support = num_classes * num_support_per_class
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, embed_size)
        if weights is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(weights))
        self.bilstm = nn.LSTM(embed_size, hidden_size, num_layers=1, bidirectional=True, batch_first=True)
        self.fc1 = nn.Linear(2 * hidden_size, output_dim)
        self.fc2 = nn.Linear(output_dim, output_dim)

    def attention(self, x):
        weights = torch.tanh(self.fc1(x))
        weights = self.fc2(weights)
        batch, seq_len, d_a = weights.shape
        weights = weights.transpose(1, 2)
        weights = weights.contiguous().view(-1, seq_len)
        weights = F.softmax(weights, dim=1).view(batch, d_a, seq_len)
        sentence_embeddings = torch.bmm(weights, x)
        avg_sentence_embeddings = torch.mean(sentence_embeddings, dim=1)
        return avg_sentence_embeddings

    def forward(self, x, hidden=None):
        batch_size, _ = x.shape
        if hidden is None:
            h = x.data.new(2, batch_size, self.hidden_size).fill_(0).float()
            c = x.data.new(2, batch_size, self.hidden_size).fill_(0).float()
        else:
            h, c = hidden
        x = self.embedding(x)
        outputs, _ = self.bilstm(x, (h, c))
        outputs = self.attention(outputs)
        support, query = outputs[0:self.num_support], outputs[self.num_support:]
        return support, query


class Induction(nn.Module):

    def __init__(self, C, S, H, iterations):
        super(Induction, self).__init__()
        self.C = C
        self.S = S
        self.H = H
        self.iterations = iterations
        self.W = torch.nn.Parameter(torch.randn(H, H))

    def forward(self, x):
        b_ij = torch.zeros(self.C, self.S)
        for _ in range(self.iterations):
            d_i = F.softmax(b_ij.unsqueeze(2), dim=1)
            e_ij = torch.mm(x.reshape(-1, self.H), self.W).reshape(self.C, self.S, self.H)
            c_i = torch.sum(d_i * e_ij, dim=1)
            squared = torch.sum(c_i ** 2, dim=1).reshape(self.C, -1)
            coeff = squared / (1 + squared) / torch.sqrt(squared + 1e-09)
            c_i = coeff * c_i
            c_produce_e = torch.bmm(e_ij, c_i.unsqueeze(2))
            b_ij = b_ij + c_produce_e.squeeze(2)
        return c_i


class Relation(nn.Module):

    def __init__(self, C, H, out_size):
        super(Relation, self).__init__()
        self.out_size = out_size
        self.M = torch.nn.Parameter(torch.randn(H, H, out_size))
        self.W = torch.nn.Parameter(torch.randn(C * out_size, C))
        self.b = torch.nn.Parameter(torch.randn(C))

    def forward(self, class_vector, query_encoder):
        mid_pro = []
        for slice in range(self.out_size):
            slice_inter = torch.mm(torch.mm(class_vector, self.M[:, :, slice]), query_encoder.transpose(1, 0))
            mid_pro.append(slice_inter)
        mid_pro = torch.cat(mid_pro, dim=0)
        V = F.relu(mid_pro.transpose(0, 1))
        probs = torch.sigmoid(torch.mm(V, self.W) + self.b)
        return probs


class FewShotInduction(nn.Module):

    def __init__(self, C, S, vocab_size, embed_size, hidden_size, d_a, iterations, outsize, weights=None):
        super(FewShotInduction, self).__init__()
        self.encoder = Encoder(C, S, vocab_size, embed_size, hidden_size, d_a, weights)
        self.induction = Induction(C, S, 2 * hidden_size, iterations)
        self.relation = Relation(C, 2 * hidden_size, outsize)

    def forward(self, x):
        support_encoder, query_encoder = self.encoder(x)
        class_vector = self.induction(support_encoder)
        probs = self.relation(class_vector, query_encoder)
        return probs


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (Criterion,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (Induction,
     lambda: ([], {'C': 4, 'S': 4, 'H': 4, 'iterations': 4}),
     lambda: ([torch.rand([4, 4, 4])], {})),
    (Relation,
     lambda: ([], {'C': 4, 'H': 4, 'out_size': 4}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4])], {})),
]

