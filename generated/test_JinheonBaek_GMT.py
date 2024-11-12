
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


import math


import torch


import torch.nn as nn


import torch.nn.functional as F


from math import ceil


import time


import random


import numpy as np


from functools import reduce


import matplotlib.pyplot as plt


from collections import Counter


class MAB(nn.Module):

    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False, cluster=False, conv=None):
        super(MAB, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k, self.fc_v = self.get_fc_kv(dim_K, dim_V, conv)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)
        self.softmax_dim = 2
        if cluster == True:
            self.softmax_dim = 1

    def forward(self, Q, K, attention_mask=None, graph=None, return_attn=False):
        Q = self.fc_q(Q)
        if graph is not None:
            x, edge_index, batch = graph
            K, V = self.fc_k(x, edge_index), self.fc_v(x, edge_index)
            K, _ = to_dense_batch(K, batch)
            V, _ = to_dense_batch(V, batch)
        else:
            K, V = self.fc_k(K), self.fc_v(K)
        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)
        if attention_mask is not None:
            attention_mask = torch.cat([attention_mask for _ in range(self.num_heads)], 0)
            attention_score = Q_.bmm(K_.transpose(1, 2)) / math.sqrt(self.dim_V)
            A = torch.softmax(attention_mask + attention_score, self.softmax_dim)
        else:
            A = torch.softmax(Q_.bmm(K_.transpose(1, 2)) / math.sqrt(self.dim_V), self.softmax_dim)
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        if return_attn:
            return O, A
        else:
            return O

    def get_fc_kv(self, dim_K, dim_V, conv):
        if conv == 'GCN':
            fc_k = GCNConv(dim_K, dim_V)
            fc_v = GCNConv(dim_K, dim_V)
        elif conv == 'GIN':
            fc_k = GINConv(nn.Sequential(nn.Linear(dim_K, dim_K), nn.ReLU(), nn.Linear(dim_K, dim_V), nn.ReLU(), nn.BatchNorm1d(dim_V)), train_eps=False)
            fc_v = GINConv(nn.Sequential(nn.Linear(dim_K, dim_K), nn.ReLU(), nn.Linear(dim_K, dim_V), nn.ReLU(), nn.BatchNorm1d(dim_V)), train_eps=False)
        else:
            fc_k = nn.Linear(dim_K, dim_V)
            fc_v = nn.Linear(dim_K, dim_V)
        return fc_k, fc_v


class SAB(nn.Module):

    def __init__(self, dim_in, dim_out, num_heads, ln=False, cluster=False, mab_conv=None):
        super(SAB, self).__init__()
        self.mab = MAB(dim_in, dim_in, dim_out, num_heads, ln=ln, cluster=cluster, conv=mab_conv)

    def forward(self, X, attention_mask=None, graph=None):
        return self.mab(X, X, attention_mask, graph)


class ISAB(nn.Module):

    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False, cluster=False, mab_conv=None):
        super(ISAB, self).__init__()
        self.I = nn.Parameter(torch.Tensor(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads, ln=ln, cluster=cluster, conv=mab_conv)
        self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads, ln=ln, cluster=cluster, conv=mab_conv)

    def forward(self, X, attention_mask=None, graph=None):
        H = self.mab0(self.I.repeat(X.size(0), 1, 1), X, attention_mask, graph)
        return self.mab1(X, H)


class PMA(nn.Module):

    def __init__(self, dim, num_heads, num_seeds, ln=False, cluster=False, mab_conv=None):
        super(PMA, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(dim, dim, dim, num_heads, ln=ln, cluster=cluster, conv=mab_conv)

    def forward(self, X, attention_mask=None, graph=None, return_attn=False):
        return self.mab(self.S.repeat(X.size(0), 1, 1), X, attention_mask, graph, return_attn)


class GraphRepresentation(torch.nn.Module):

    def __init__(self, args):
        super(GraphRepresentation, self).__init__()
        self.args = args
        self.num_features = args.num_features
        self.nhid = args.num_hidden
        self.num_classes = args.num_classes
        self.pooling_ratio = args.pooling_ratio
        self.dropout_ratio = args.dropout

    def get_convs(self):
        convs = nn.ModuleList()
        _input_dim = self.num_features
        _output_dim = self.nhid
        for _ in range(self.args.num_convs):
            if self.args.conv == 'GCN':
                conv = GCNConv(_input_dim, _output_dim)
            elif self.args.conv == 'GIN':
                conv = GINConv(nn.Sequential(nn.Linear(_input_dim, _output_dim), nn.ReLU(), nn.Linear(_output_dim, _output_dim), nn.ReLU(), nn.BatchNorm1d(_output_dim)), train_eps=False)
            convs.append(conv)
            _input_dim = _output_dim
            _output_dim = _output_dim
        return convs

    def get_pools(self):
        pools = nn.ModuleList([gap])
        return pools

    def get_classifier(self):
        return nn.Sequential(nn.Linear(self.nhid, self.nhid), nn.ReLU(), nn.Dropout(p=self.dropout_ratio), nn.Linear(self.nhid, self.nhid // 2), nn.ReLU(), nn.Dropout(p=self.dropout_ratio), nn.Linear(self.nhid // 2, self.num_classes))


class GraphMultisetTransformer(GraphRepresentation):

    def __init__(self, args):
        super(GraphMultisetTransformer, self).__init__(args)
        self.ln = args.ln
        self.num_heads = args.num_heads
        self.cluster = args.cluster
        self.model_sequence = args.model_string.split('-')
        self.convs = self.get_convs()
        self.pools = self.get_pools()
        self.classifier = self.get_classifier()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        xs = []
        for _ in range(self.args.num_convs):
            x = F.relu(self.convs[_](x, edge_index))
            xs.append(x)
        x = torch.cat(xs, dim=1)
        for _index, _model_str in enumerate(self.model_sequence):
            if _index == 0:
                batch_x, mask = to_dense_batch(x, batch)
                extended_attention_mask = mask.unsqueeze(1)
                extended_attention_mask = extended_attention_mask
                extended_attention_mask = (1.0 - extended_attention_mask) * -1000000000.0
            if _model_str == 'GMPool_G':
                batch_x = self.pools[_index](batch_x, attention_mask=extended_attention_mask, graph=(x, edge_index, batch))
            else:
                batch_x = self.pools[_index](batch_x, attention_mask=extended_attention_mask)
            extended_attention_mask = None
        batch_x = self.pools[len(self.model_sequence)](batch_x)
        x = batch_x.squeeze(1)
        x = self.classifier(x)
        return F.log_softmax(x, dim=-1)

    def get_pools(self, _input_dim=None, reconstruction=False):
        pools = nn.ModuleList()
        _input_dim = self.nhid * self.args.num_convs if _input_dim is None else _input_dim
        _output_dim = self.nhid
        _num_nodes = ceil(self.pooling_ratio * self.args.avg_num_nodes)
        for _index, _model_str in enumerate(self.model_sequence):
            if _index == len(self.model_sequence) - 1 and reconstruction == False:
                _num_nodes = 1
            if _model_str == 'GMPool_G':
                pools.append(PMA(_input_dim, self.num_heads, _num_nodes, ln=self.ln, cluster=self.cluster, mab_conv=self.args.mab_conv))
                _num_nodes = ceil(self.pooling_ratio * _num_nodes)
            elif _model_str == 'GMPool_I':
                pools.append(PMA(_input_dim, self.num_heads, _num_nodes, ln=self.ln, cluster=self.cluster, mab_conv=None))
                _num_nodes = ceil(self.pooling_ratio * _num_nodes)
            elif _model_str == 'SelfAtt':
                pools.append(SAB(_input_dim, _output_dim, self.num_heads, ln=self.ln, cluster=self.cluster))
                _input_dim = _output_dim
                _output_dim = _output_dim
            else:
                raise ValueError('Model Name in Model String <{}> is Unknown'.format(_model_str))
        pools.append(nn.Linear(_input_dim, self.nhid))
        return pools


class GraphMultisetTransformer_for_OGB(GraphMultisetTransformer):

    def __init__(self, args):
        super(GraphMultisetTransformer_for_OGB, self).__init__(args)
        self.atom_encoder = AtomEncoder(self.nhid)
        self.convs = self.get_convs()

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x = self.atom_encoder(x)
        xs = []
        for _ in range(self.args.num_convs):
            x = F.relu(self.convs[_](x, edge_index, edge_attr))
            xs.append(x)
        x = torch.cat(xs, dim=1)
        for _index, _model_str in enumerate(self.model_sequence):
            if _index == 0:
                batch_x, mask = to_dense_batch(x, batch)
                extended_attention_mask = mask.unsqueeze(1)
                extended_attention_mask = extended_attention_mask
                extended_attention_mask = (1.0 - extended_attention_mask) * -1000000000.0
            if _model_str == 'GMPool_G':
                batch_x = self.pools[_index](batch_x, attention_mask=extended_attention_mask, graph=(x, edge_index, batch))
            else:
                batch_x = self.pools[_index](batch_x, attention_mask=extended_attention_mask)
            extended_attention_mask = None
        batch_x = self.pools[len(self.model_sequence)](batch_x)
        x = batch_x.squeeze(1)
        x = self.classifier(x)
        return x

    def get_convs(self):
        convs = nn.ModuleList()
        for _ in range(self.args.num_convs):
            if self.args.conv == 'GCN':
                conv = GCNConv_for_OGB(self.nhid)
            elif self.args.conv == 'GIN':
                conv = GINConv_for_OGB(self.nhid)
            convs.append(conv)
        return convs


class GraphMultisetTransformer_for_Recon(GraphMultisetTransformer):

    def __init__(self, args):
        super(GraphMultisetTransformer_for_Recon, self).__init__(args)
        self.pools = self.get_pools(_input_dim=self.nhid, reconstruction=True)
        self.unconvs = self.get_unconvs()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for _ in range(self.args.num_convs):
            x = F.relu(self.convs[_](x, edge_index))
        for _index, _model_str in enumerate(self.model_sequence):
            if _index == 0:
                batch_x, mask = to_dense_batch(x, batch)
                extended_attention_mask = mask.unsqueeze(1)
                extended_attention_mask = extended_attention_mask
                extended_attention_mask = (1.0 - extended_attention_mask) * -1000000000.0
            if _model_str == 'GMPool_G':
                batch_x, attn = self.pools[_index](batch_x, attention_mask=extended_attention_mask, graph=(x, edge_index, batch), return_attn=True)
            else:
                batch_x, attn = self.pools[_index](batch_x, attention_mask=extended_attention_mask, return_attn=True)
            extended_attention_mask = None
        batch_x = self.pools[len(self.model_sequence)](batch_x)
        x = torch.bmm(attn.transpose(1, 2), batch_x)
        x = x[mask]
        for _ in range(self.args.num_unconvs):
            x = self.unconvs[_](x, edge_index)
            if _ < self.args.num_unconvs - 1:
                x = F.relu(x)
        return x

    def get_unconvs(self):
        unconvs = nn.ModuleList()
        _input_dim = self.nhid
        _output_dim = self.nhid
        for _ in range(self.args.num_unconvs):
            if _ == self.args.num_unconvs - 1:
                _output_dim = self.num_features
            if self.args.conv == 'GCN':
                conv = GCNConv(_input_dim, _output_dim)
            elif self.args.conv == 'GIN':
                conv = GINConv(nn.Sequential(nn.Linear(_input_dim, _input_dim), nn.ReLU(), nn.Linear(_input_dim, _output_dim), nn.ReLU(), nn.BatchNorm1d(_output_dim)), train_eps=False)
            unconvs.append(conv)
        return unconvs


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (ISAB,
     lambda: ([], {'dim_in': 4, 'dim_out': 4, 'num_heads': 4, 'num_inds': 4}),
     lambda: ([torch.rand([4, 4, 4])], {})),
    (MAB,
     lambda: ([], {'dim_Q': 4, 'dim_K': 4, 'dim_V': 4, 'num_heads': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {})),
    (PMA,
     lambda: ([], {'dim': 4, 'num_heads': 4, 'num_seeds': 4}),
     lambda: ([torch.rand([4, 4, 4])], {})),
    (SAB,
     lambda: ([], {'dim_in': 4, 'dim_out': 4, 'num_heads': 4}),
     lambda: ([torch.rand([4, 4, 4])], {})),
]

