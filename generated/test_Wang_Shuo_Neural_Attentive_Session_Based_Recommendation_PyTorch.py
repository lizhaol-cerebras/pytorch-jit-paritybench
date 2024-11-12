
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


from torch.utils.data import Dataset


import numpy as np


import time


import random


from torch import nn


from torch.utils.data import DataLoader


import torch.nn.functional as F


import torch.optim as optim


from torch.optim.lr_scheduler import StepLR


from torch.autograd import Variable


from torch.backends import cudnn


import torch.nn as nn


from torch.nn.utils.rnn import pack_padded_sequence


from torch.nn.utils.rnn import pad_packed_sequence


class NARM(nn.Module):
    """Neural Attentive Session Based Recommendation Model Class

    Args:
        n_items(int): the number of items
        hidden_size(int): the hidden size of gru
        embedding_dim(int): the dimension of item embedding
        batch_size(int): 
        n_layers(int): the number of gru layers

    """

    def __init__(self, n_items, hidden_size, embedding_dim, batch_size, n_layers=1):
        super(NARM, self).__init__()
        self.n_items = n_items
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.n_layers = n_layers
        self.embedding_dim = embedding_dim
        self.emb = nn.Embedding(self.n_items, self.embedding_dim, padding_idx=0)
        self.emb_dropout = nn.Dropout(0.25)
        self.gru = nn.GRU(self.embedding_dim, self.hidden_size, self.n_layers)
        self.a_1 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.a_2 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.v_t = nn.Linear(self.hidden_size, 1, bias=False)
        self.ct_dropout = nn.Dropout(0.5)
        self.b = nn.Linear(self.embedding_dim, 2 * self.hidden_size, bias=False)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, seq, lengths):
        hidden = self.init_hidden(seq.size(1))
        embs = self.emb_dropout(self.emb(seq))
        embs = pack_padded_sequence(embs, lengths)
        gru_out, hidden = self.gru(embs, hidden)
        gru_out, lengths = pad_packed_sequence(gru_out)
        ht = hidden[-1]
        gru_out = gru_out.permute(1, 0, 2)
        c_global = ht
        q1 = self.a_1(gru_out.contiguous().view(-1, self.hidden_size)).view(gru_out.size())
        q2 = self.a_2(ht)
        mask = torch.where(seq.permute(1, 0) > 0, torch.tensor([1.0], device=self.device), torch.tensor([0.0], device=self.device))
        q2_expand = q2.unsqueeze(1).expand_as(q1)
        q2_masked = mask.unsqueeze(2).expand_as(q1) * q2_expand
        alpha = self.v_t(torch.sigmoid(q1 + q2_masked).view(-1, self.hidden_size)).view(mask.size())
        c_local = torch.sum(alpha.unsqueeze(2).expand_as(gru_out) * gru_out, 1)
        c_t = torch.cat([c_local, c_global], 1)
        c_t = self.ct_dropout(c_t)
        item_embs = self.emb(torch.arange(self.n_items))
        scores = torch.matmul(c_t, self.b(item_embs).permute(1, 0))
        return scores

    def init_hidden(self, batch_size):
        return torch.zeros((self.n_layers, batch_size, self.hidden_size), requires_grad=True)

