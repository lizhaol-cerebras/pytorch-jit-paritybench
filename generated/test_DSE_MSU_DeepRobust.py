
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


import torch.nn as nn


import torch.nn.functional as F


import torchvision


import torchvision.transforms as transforms


import numpy as np


import torch


import torch.optim as optim


import torch.utils.data as data_utils


import math


import torchvision.models as models


import random


import time


import collections


import scipy.sparse as sp


from itertools import repeat


import warnings


from torch.nn.modules.module import Module


from torch.nn.parameter import Parameter


from copy import deepcopy


from sklearn.metrics import f1_score


from torch import Tensor


from torch import optim


from torch.nn import Linear


from torch.optim.sgd import SGD


from torch.optim.optimizer import required


from torch.optim import Optimizer


import sklearn


from torch.distributions.multivariate_normal import MultivariateNormal


from sklearn.metrics.pairwise import cosine_similarity


from itertools import product


from typing import Optional


from typing import Tuple


import scipy.sparse


from typing import Union


from torch.nn import Parameter


from torch.nn import functional as F


from itertools import count


import scipy.linalg as spl


from scipy.sparse.linalg import eigsh


import torch.multiprocessing as mp


from torch import spmm


from collections import namedtuple


from functools import lru_cache


import copy


from sklearn.cluster import KMeans


from sklearn.model_selection import train_test_split


import torch.sparse as ts


import logging


from torch.autograd import Variable


from abc import ABCMeta


import torch as torch


from numpy import linalg as LA


import scipy.optimize as so


import torch.backends.cudnn as cudnn


from torchvision import datasets


from torchvision import transforms


from torch.nn.modules.loss import _Loss


from collections import OrderedDict


from typing import List


from typing import Dict


from torch.utils.data import DataLoader


from torch.utils.data import Dataset


from torchvision import models


import matplotlib.pyplot as plt


import torch.nn


from torch.utils.data import TensorDataset


import numpy as py


class ChebNet(nn.Module):
    """ 2 Layer ChebNet based on pytorch geometric.

    Parameters
    ----------
    nfeat : int
        size of input feature dimension
    nhid : int
        number of hidden units
    nclass : int
        size of output dimension
    num_hops: int
        number of hops in ChebConv
    dropout : float
        dropout rate for ChebNet
    lr : float
        learning rate for ChebNet
    weight_decay : float
        weight decay coefficient (l2 normalization) for GCN.
        When `with_relu` is True, `weight_decay` will be set to 0.
    with_bias: bool
        whether to include bias term in ChebNet weights.
    device: str
        'cpu' or 'cuda'.

    Examples
    --------
	We can first load dataset and then train ChebNet.

    >>> from deeprobust.graph.data import Dataset
    >>> from deeprobust.graph.defense import ChebNet
    >>> data = Dataset(root='/tmp/', name='cora')
    >>> adj, features, labels = data.adj, data.features, data.labels
    >>> idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
    >>> cheby = ChebNet(nfeat=features.shape[1],
              nhid=16, num_hops=3,
              nclass=labels.max().item() + 1,
              dropout=0.5, device='cpu')
    >>> cheby = cheby.to('cpu')
    >>> pyg_data = Dpr2Pyg(data) # convert deeprobust dataset to pyg dataset
    >>> cheby.fit(pyg_data, patience=10, verbose=True) # train with earlystopping
    """

    def __init__(self, nfeat, nhid, nclass, num_hops=3, dropout=0.5, lr=0.01, weight_decay=0.0005, with_bias=True, device=None):
        super(ChebNet, self).__init__()
        assert device is not None, "Please specify 'device'!"
        self.device = device
        self.conv1 = ChebConv(nfeat, nhid, K=num_hops, bias=with_bias)
        self.conv2 = ChebConv(nhid, nclass, K=num_hops, bias=with_bias)
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.lr = lr
        self.output = None
        self.best_model = None
        self.best_output = None

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

    def initialize(self):
        """Initialize parameters of ChebNet.
        """
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def fit(self, pyg_data, train_iters=200, initialize=True, verbose=False, patience=500, **kwargs):
        """Train the ChebNet model, when idx_val is not None, pick the best model
        according to the validation loss.

        Parameters
        ----------
        pyg_data :
            pytorch geometric dataset object
        train_iters : int
            number of training epochs
        initialize : bool
            whether to initialize parameters before training
        verbose : bool
            whether to show verbose logs
        patience : int
            patience for early stopping, only valid when `idx_val` is given
        """
        self.device = self.conv1.weight.device
        if initialize:
            self.initialize()
        self.data = pyg_data[0]
        self.train_with_early_stopping(train_iters, patience, verbose)

    def train_with_early_stopping(self, train_iters, patience, verbose):
        """early stopping based on the validation loss
        """
        if verbose:
            None
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        labels = self.data.y
        train_mask, val_mask = self.data.train_mask, self.data.val_mask
        early_stopping = patience
        best_loss_val = 100
        for i in range(train_iters):
            self.train()
            optimizer.zero_grad()
            output = self.forward(self.data)
            loss_train = F.nll_loss(output[train_mask], labels[train_mask])
            loss_train.backward()
            optimizer.step()
            if verbose and i % 10 == 0:
                None
            self.eval()
            output = self.forward(self.data)
            loss_val = F.nll_loss(output[val_mask], labels[val_mask])
            if best_loss_val > loss_val:
                best_loss_val = loss_val
                self.output = output
                weights = deepcopy(self.state_dict())
                patience = early_stopping
            else:
                patience -= 1
            if i > early_stopping and patience <= 0:
                break
        if verbose:
            None
        self.load_state_dict(weights)

    def test(self):
        """Evaluate ChebNet performance on test set.

        Parameters
        ----------
        idx_test :
            node testing indices
        """
        self.eval()
        test_mask = self.data.test_mask
        labels = self.data.y
        output = self.forward(self.data)
        loss_test = F.nll_loss(output[test_mask], labels[test_mask])
        acc_test = utils.accuracy(output[test_mask], labels[test_mask])
        None
        return acc_test.item()

    def predict(self):
        """
        Returns
        -------
        torch.FloatTensor
            output (log probabilities) of ChebNet
        """
        self.eval()
        return self.forward(self.data)


class BaseModel(nn.Module):

    def __init__(self):
        super(BaseModel, self).__init__()
        pass

    def fit(self, pyg_data, train_iters=1000, initialize=True, verbose=False, patience=100, **kwargs):
        if initialize:
            self.initialize()
        self.data = pyg_data
        self.train_with_early_stopping(train_iters, patience, verbose)

    def finetune(self, edge_index, edge_weight, feat=None, train_iters=10, verbose=True):
        if verbose:
            None
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        labels = self.data.y
        if feat is None:
            x = self.data.x
        else:
            x = feat
        train_mask, val_mask = self.data.train_mask, self.data.val_mask
        best_loss_val = 100
        best_acc_val = 0
        for i in range(train_iters):
            self.train()
            optimizer.zero_grad()
            output = self.forward(x, edge_index, edge_weight)
            loss_train = F.nll_loss(output[train_mask], labels[train_mask])
            loss_train.backward()
            optimizer.step()
            if verbose and i % 50 == 0:
                None
            self.eval()
            with torch.no_grad():
                output = self.forward(x, edge_index)
            loss_val = F.nll_loss(output[val_mask], labels[val_mask])
            acc_val = utils.accuracy(output[val_mask], labels[val_mask])
            if best_acc_val < acc_val:
                best_acc_val = acc_val
                best_output = output
                weights = deepcopy(self.state_dict())
        None
        self.load_state_dict(weights)
        return best_output

    def _fit_with_val(self, pyg_data, train_iters=1000, initialize=True, verbose=False, **kwargs):
        if initialize:
            self.initialize()
        self.data = pyg_data
        if verbose:
            None
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        labels = self.data.y
        train_mask, val_mask = self.data.train_mask, self.data.val_mask
        x, edge_index = self.data.x, self.data.edge_index
        for i in range(train_iters):
            self.train()
            optimizer.zero_grad()
            output = self.forward(x, edge_index)
            loss_train = F.nll_loss(output[train_mask + val_mask], labels[train_mask + val_mask])
            loss_train.backward()
            optimizer.step()
            if verbose and i % 50 == 0:
                None

    def fit_with_val(self, pyg_data, train_iters=1000, initialize=True, patience=100, verbose=False, **kwargs):
        if initialize:
            self.initialize()
        self.data = pyg_data
        self.data.train_mask = self.data.train_mask + self.data.val1_mask
        self.data.val_mask = self.data.val2_mask
        self.train_with_early_stopping(train_iters, patience, verbose)

    def train_with_early_stopping(self, train_iters, patience, verbose):
        """early stopping based on the validation loss
        """
        if verbose:
            None
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        labels = self.data.y
        train_mask, val_mask = self.data.train_mask, self.data.val_mask
        early_stopping = patience
        best_loss_val = 100
        best_acc_val = 0
        best_epoch = 0
        x, edge_index = self.data.x, self.data.edge_index
        for i in range(train_iters):
            self.train()
            optimizer.zero_grad()
            output = self.forward(x, edge_index)
            loss_train = F.nll_loss(output[train_mask], labels[train_mask])
            loss_train.backward()
            optimizer.step()
            if verbose and i % 50 == 0:
                None
            self.eval()
            output = self.forward(x, edge_index)
            loss_val = F.nll_loss(output[val_mask], labels[val_mask])
            acc_val = utils.accuracy(output[val_mask], labels[val_mask])
            if best_acc_val < acc_val:
                best_acc_val = acc_val
                self.output = output
                weights = deepcopy(self.state_dict())
                patience = early_stopping
                best_epoch = i
            else:
                patience -= 1
            if i > early_stopping and patience <= 0:
                break
        if verbose:
            None
        self.load_state_dict(weights)

    def test(self):
        """Evaluate model performance on test set.
        Parameters
        ----------
        idx_test :
            node testing indices
        """
        self.eval()
        test_mask = self.data.test_mask
        labels = self.data.y
        output = self.forward(self.data.x, self.data.edge_index)
        loss_test = F.nll_loss(output[test_mask], labels[test_mask])
        acc_test = utils.accuracy(output[test_mask], labels[test_mask])
        None
        return acc_test.item()

    def predict(self, x=None, edge_index=None, edge_weight=None):
        """
        Returns
        -------
        torch.FloatTensor
            output (log probabilities)
        """
        self.eval()
        if x is None or edge_index is None:
            x, edge_index = self.data.x, self.data.edge_index
        return self.forward(x, edge_index, edge_weight)

    def _ensure_contiguousness(self, x, edge_idx, edge_weight):
        if not x.is_sparse:
            x = x.contiguous()
        if hasattr(edge_idx, 'contiguous'):
            edge_idx = edge_idx.contiguous()
        if edge_weight is not None:
            edge_weight = edge_weight.contiguous()
        return x, edge_idx, edge_weight


def add_self_loops(edge_index, edge_weight=None, fill_value=1, num_nodes=None):
    loop_index = torch.arange(0, num_nodes, dtype=torch.long, device=edge_index.device)
    loop_index = loop_index.unsqueeze(0).repeat(2, 1)
    if edge_weight is not None:
        assert edge_weight.numel() == edge_index.size(1)
        loop_weight = edge_weight.new_full((num_nodes,), fill_value)
        edge_weight = torch.cat([edge_weight, loop_weight], dim=0)
    edge_index = torch.cat([edge_index, loop_index], dim=1)
    return edge_index, edge_weight


class GAT(BaseModel):

    def __init__(self, nfeat, nhid, nclass, heads=8, output_heads=1, dropout=0.5, lr=0.01, nlayers=2, with_bn=False, weight_decay=0.0005, with_bias=True, device=None):
        super(GAT, self).__init__()
        assert device is not None, "Please specify 'device'!"
        self.device = device
        self.convs = nn.ModuleList([])
        if with_bn:
            self.bns = nn.ModuleList([])
            self.bns.append(nn.BatchNorm1d(nhid * heads))
        self.convs.append(GATConv(nfeat, nhid, heads=heads, dropout=dropout, bias=with_bias))
        for i in range(nlayers - 2):
            self.convs.append(GATConv(nhid * heads, nhid, heads=heads, dropout=dropout, bias=with_bias))
            if with_bn:
                self.bns.append(nn.BatchNorm1d(nhid * heads))
        self.convs.append(GATConv(nhid * heads, nclass, heads=output_heads, concat=False, dropout=dropout, bias=with_bias))
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.lr = lr
        self.output = None
        self.best_model = None
        self.best_output = None
        self.name = 'GAT'
        self.with_bn = with_bn

    def forward(self, x, edge_index, edge_weight=None):
        for ii, conv in enumerate(self.convs[:-1]):
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = conv(x, edge_index, edge_weight)
            if self.with_bn:
                x = self.bns[ii](x)
                x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index, edge_weight)
        return F.log_softmax(x, dim=1)

    def get_embed(self, x, edge_index, edge_weight=None):
        for ii, conv in enumerate(self.convs[:-1]):
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = conv(x, edge_index, edge_weight)
            if self.with_bn:
                x = self.bns[ii](x)
                x = F.elu(x)
        return x

    def initialize(self):
        for conv in self.convs:
            conv.reset_parameters()
        if self.with_bn:
            for bn in self.bns:
                bn.reset_parameters()


class GraphConvolution(Module):
    """Simple GCN layer, similar to https://github.com/tkipf/pygcn
    """

    def __init__(self, in_features, out_features, with_bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if with_bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        """ Graph Convolutional Layer forward function
        """
        if input.data.is_sparse:
            support = torch.spmm(input, self.weight)
        else:
            support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GCN(nn.Module):

    def __init__(self, nfeat, nhid, nclass, dropout=0.5, lr=0.01, weight_decay=0.0005, layer=2, device=None, layer_norm_first=False, use_ln=False):
        super(GCN, self).__init__()
        assert device is not None, "Please specify 'device'!"
        self.device = device
        self.nfeat = nfeat
        self.hidden_sizes = [nhid]
        self.nclass = nclass
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(nfeat, nhid))
        self.lns = nn.ModuleList()
        self.lns.append(torch.nn.LayerNorm(nfeat))
        for _ in range(layer - 2):
            self.convs.append(GCNConv(nhid, nhid))
            self.lns.append(nn.LayerNorm(nhid))
        self.lns.append(nn.LayerNorm(nhid))
        self.gc2 = GCNConv(nhid, nclass)
        self.dropout = dropout
        self.lr = lr
        self.output = None
        self.edge_index = None
        self.edge_weight = None
        self.features = None
        self.weight_decay = weight_decay
        self.layer_norm_first = layer_norm_first
        self.use_ln = use_ln

    def forward(self, x, edge_index, edge_weight=None):
        if self.layer_norm_first:
            x = self.lns[0](x)
        i = 0
        for conv in self.convs:
            x = F.relu(conv(x, edge_index, edge_weight))
            if self.use_ln:
                x = self.lns[i + 1](x)
            i += 1
            x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, edge_index, edge_weight)
        return F.log_softmax(x, dim=1)

    def get_h(self, x, edge_index):
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
        return x

    def fit(self, features, edge_index, edge_weight, labels, idx_train, idx_val=None, train_iters=200, verbose=False):
        """Train the gcn model, when idx_val is not None, pick the best model according to the validation loss.
        Parameters
        ----------
        features :
            node features
        adj :
            the adjacency matrix. The format could be torch.tensor or scipy matrix
        labels :
            node labels
        idx_train :
            node training indices
        idx_val :
            node validation indices. If not given (None), GCN training process will not adpot early stopping
        train_iters : int
            number of training epochs
        initialize : bool
            whether to initialize parameters before training
        verbose : bool
            whether to show verbose logs
        """
        self.edge_index, self.edge_weight = edge_index, edge_weight
        self.features = features
        self.labels = labels
        if idx_val is None:
            self._train_without_val(self.labels, idx_train, train_iters, verbose)
        else:
            self._train_with_val(self.labels, idx_train, idx_val, train_iters, verbose)

    def _train_without_val(self, labels, idx_train, train_iters, verbose):
        self.train()
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        for i in range(train_iters):
            optimizer.zero_grad()
            output = self.forward(self.features, self.edge_index, self.edge_weight)
            loss_train = F.nll_loss(output[idx_train], labels[idx_train])
            loss_train.backward()
            optimizer.step()
            if verbose and i % 10 == 0:
                None
        self.eval()
        output = self.forward(self.features, self.edge_index, self.edge_weight)
        self.output = output

    def _train_with_val(self, labels, idx_train, idx_val, train_iters, verbose):
        if verbose:
            None
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        best_loss_val = 100
        best_acc_val = 0
        for i in range(train_iters):
            self.train()
            optimizer.zero_grad()
            output = self.forward(self.features, self.edge_index, self.edge_weight)
            loss_train = F.nll_loss(output[idx_train], labels[idx_train])
            loss_train.backward()
            optimizer.step()
            self.eval()
            output = self.forward(self.features, self.edge_index, self.edge_weight)
            loss_val = F.nll_loss(output[idx_val], labels[idx_val])
            acc_val = utils.accuracy(output[idx_val], labels[idx_val])
            if verbose and i % 10 == 0:
                None
                None
            if acc_val > best_acc_val:
                best_acc_val = acc_val
                self.output = output
                weights = deepcopy(self.state_dict())
        if verbose:
            None
        self.load_state_dict(weights)

    def test(self, features, edge_index, edge_weight, labels, idx_test):
        """Evaluate GCN performance on test set.
        Parameters
        ----------
        idx_test :
            node testing indices
        """
        self.eval()
        with torch.no_grad():
            output = self.forward(features, edge_index, edge_weight)
            acc_test = utils.accuracy(output[idx_test], labels[idx_test])
        return float(acc_test)

    def test_with_correct_nodes(self, features, edge_index, edge_weight, labels, idx_test):
        self.eval()
        output = self.forward(features, edge_index, edge_weight)
        correct_nids = (output.argmax(dim=1)[idx_test] == labels[idx_test]).nonzero().flatten()
        acc_test = utils.accuracy(output[idx_test], labels[idx_test])
        return acc_test, correct_nids


class GCNSVD(GCN):
    """GCNSVD is a 2 Layer Graph Convolutional Network with Truncated SVD as
    preprocessing. See more details in All You Need Is Low (Rank): Defending
    Against Adversarial Attacks on Graphs,
    https://dl.acm.org/doi/abs/10.1145/3336191.3371789.

    Parameters
    ----------
    nfeat : int
        size of input feature dimension
    nhid : int
        number of hidden units
    nclass : int
        size of output dimension
    dropout : float
        dropout rate for GCN
    lr : float
        learning rate for GCN
    weight_decay : float
        weight decay coefficient (l2 normalization) for GCN. When `with_relu` is True, `weight_decay` will be set to 0.
    with_relu : bool
        whether to use relu activation function. If False, GCN will be linearized.
    with_bias: bool
        whether to include bias term in GCN weights.
    device: str
        'cpu' or 'cuda'.

    Examples
    --------
	We can first load dataset and then train GCNSVD.

    >>> from deeprobust.graph.data import PrePtbDataset, Dataset
    >>> from deeprobust.graph.defense import GCNSVD
    >>> # load clean graph data
    >>> data = Dataset(root='/tmp/', name='cora', seed=15)
    >>> adj, features, labels = data.adj, data.features, data.labels
    >>> idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
    >>> # load perturbed graph data
    >>> perturbed_data = PrePtbDataset(root='/tmp/', name='cora')
    >>> perturbed_adj = perturbed_data.adj
    >>> # train defense model
    >>> model = GCNSVD(nfeat=features.shape[1],
              nhid=16,
              nclass=labels.max().item() + 1,
              dropout=0.5, device='cpu').to('cpu')
    >>> model.fit(features, perturbed_adj, labels, idx_train, idx_val, k=20)

    """

    def __init__(self, nfeat, nhid, nclass, dropout=0.5, lr=0.01, weight_decay=0.0005, with_relu=True, with_bias=True, device='cpu'):
        super(GCNSVD, self).__init__(nfeat, nhid, nclass, dropout, lr, weight_decay, with_relu, with_bias, device=device)
        self.device = device
        self.k = None

    def fit(self, features, adj, labels, idx_train, idx_val=None, k=50, train_iters=200, initialize=True, verbose=True, **kwargs):
        """First perform rank-k approximation of adjacency matrix via
        truncated SVD, and then train the gcn model on the processed graph,
        when idx_val is not None, pick the best model according to
        the validation loss.

        Parameters
        ----------
        features :
            node features
        adj :
            the adjacency matrix. The format could be torch.tensor or scipy matrix
        labels :
            node labels
        idx_train :
            node training indices
        idx_val :
            node validation indices. If not given (None), GCN training process will not adpot early stopping
        k : int
            number of singular values and vectors to compute.
        train_iters : int
            number of training epochs
        initialize : bool
            whether to initialize parameters before training
        verbose : bool
            whether to show verbose logs
        """
        modified_adj = self.truncatedSVD(adj, k=k)
        self.k = k
        features, modified_adj, labels = utils.to_tensor(features, modified_adj, labels, device=self.device)
        self.modified_adj = modified_adj
        self.features = features
        self.labels = labels
        super().fit(features, modified_adj, labels, idx_train, idx_val, train_iters=train_iters, initialize=initialize, verbose=verbose)

    def truncatedSVD(self, data, k=50):
        """Truncated SVD on input data.

        Parameters
        ----------
        data :
            input matrix to be decomposed
        k : int
            number of singular values and vectors to compute.

        Returns
        -------
        numpy.array
            reconstructed matrix.
        """
        None
        if sp.issparse(data):
            data = data.asfptype()
            U, S, V = sp.linalg.svds(data, k=k)
            None
            diag_S = np.diag(S)
        else:
            U, S, V = np.linalg.svd(data)
            U = U[:, :k]
            S = S[:k]
            V = V[:k, :]
            None
            diag_S = np.diag(S)
            None
        return U @ diag_S @ V

    def predict(self, features=None, adj=None):
        """By default, the inputs should be unnormalized adjacency

        Parameters
        ----------
        features :
            node features. If `features` and `adj` are not given, this function will use previous stored `features` and `adj` from training to make predictions.
        adj :
            adjcency matrix. If `features` and `adj` are not given, this function will use previous stored `features` and `adj` from training to make predictions.


        Returns
        -------
        torch.FloatTensor
            output (log probabilities) of GCNSVD
        """
        self.eval()
        if features is None and adj is None:
            return self.forward(self.features, self.adj_norm)
        else:
            adj = self.truncatedSVD(adj, k=self.k)
            if type(adj) is not torch.Tensor:
                features, adj = utils.to_tensor(features, adj, device=self.device)
            self.features = features
            if utils.is_sparse_tensor(adj):
                self.adj_norm = utils.normalize_adj_tensor(adj, sparse=True)
            else:
                self.adj_norm = utils.normalize_adj_tensor(adj)
            return self.forward(self.features, self.adj_norm)


class GCNJaccard(GCN):
    """GCNJaccard first preprocesses input graph via droppining dissimilar
    edges and train a GCN based on the processed graph. See more details in
    Adversarial Examples on Graph Data: Deep Insights into Attack and Defense,
    https://arxiv.org/pdf/1903.01610.pdf.

    Parameters
    ----------
    nfeat : int
        size of input feature dimension
    nhid : int
        number of hidden units
    nclass : int
        size of output dimension
    dropout : float
        dropout rate for GCN
    lr : float
        learning rate for GCN
    weight_decay : float
        weight decay coefficient (l2 normalization) for GCN. When `with_relu` is True, `weight_decay` will be set to 0.
    with_relu : bool
        whether to use relu activation function. If False, GCN will be linearized.
    with_bias: bool
        whether to include bias term in GCN weights.
    device: str
        'cpu' or 'cuda'.

    Examples
    --------
	We can first load dataset and then train GCNJaccard.

    >>> from deeprobust.graph.data import PrePtbDataset, Dataset
    >>> from deeprobust.graph.defense import GCNJaccard
    >>> # load clean graph data
    >>> data = Dataset(root='/tmp/', name='cora', seed=15)
    >>> adj, features, labels = data.adj, data.features, data.labels
    >>> idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
    >>> # load perturbed graph data
    >>> perturbed_data = PrePtbDataset(root='/tmp/', name='cora')
    >>> perturbed_adj = perturbed_data.adj
    >>> # train defense model
    >>> model = GCNJaccard(nfeat=features.shape[1],
              nhid=16,
              nclass=labels.max().item() + 1,
              dropout=0.5, device='cpu').to('cpu')
    >>> model.fit(features, perturbed_adj, labels, idx_train, idx_val, threshold=0.03)

    """

    def __init__(self, nfeat, nhid, nclass, binary_feature=True, dropout=0.5, lr=0.01, weight_decay=0.0005, with_relu=True, with_bias=True, device='cpu'):
        super(GCNJaccard, self).__init__(nfeat, nhid, nclass, dropout, lr, weight_decay, with_relu, with_bias, device=device)
        self.device = device
        self.binary_feature = binary_feature

    def fit(self, features, adj, labels, idx_train, idx_val=None, threshold=0.01, train_iters=200, initialize=True, verbose=True, **kwargs):
        """First drop dissimilar edges with similarity smaller than given
        threshold and then train the gcn model on the processed graph.
        When idx_val is not None, pick the best model according to the
        validation loss.

        Parameters
        ----------
        features :
            node features. The format can be numpy.array or scipy matrix
        adj :
            the adjacency matrix.
        labels :
            node labels
        idx_train :
            node training indices
        idx_val :
            node validation indices. If not given (None), GCN training process will not adpot early stopping
        threshold : float
            similarity threshold for dropping edges. If two connected nodes with similarity smaller than threshold, the edge between them will be removed.
        train_iters : int
            number of training epochs
        initialize : bool
            whether to initialize parameters before training
        verbose : bool
            whether to show verbose logs
        """
        self.threshold = threshold
        modified_adj = self.drop_dissimilar_edges(features, adj)
        features, modified_adj, labels = utils.to_tensor(features, modified_adj, labels, device=self.device)
        self.modified_adj = modified_adj
        self.features = features
        self.labels = labels
        super().fit(features, modified_adj, labels, idx_train, idx_val, train_iters=train_iters, initialize=initialize, verbose=verbose)

    def drop_dissimilar_edges(self, features, adj, metric='similarity'):
        """Drop dissimilar edges.(Faster version using numba)
        """
        if not sp.issparse(adj):
            adj = sp.csr_matrix(adj)
        adj_triu = sp.triu(adj, format='csr')
        if sp.issparse(features):
            features = features.todense().A
        if metric == 'distance':
            removed_cnt = dropedge_dis(adj_triu.data, adj_triu.indptr, adj_triu.indices, features, threshold=self.threshold)
        elif self.binary_feature:
            removed_cnt = dropedge_jaccard(adj_triu.data, adj_triu.indptr, adj_triu.indices, features, threshold=self.threshold)
        else:
            removed_cnt = dropedge_cosine(adj_triu.data, adj_triu.indptr, adj_triu.indices, features, threshold=self.threshold)
        None
        modified_adj = adj_triu + adj_triu.transpose()
        return modified_adj

    def predict(self, features=None, adj=None):
        """By default, the inputs should be unnormalized adjacency

        Parameters
        ----------
        features :
            node features. If `features` and `adj` are not given, this function will use previous stored `features` and `adj` from training to make predictions.
        adj :
            adjcency matrix. If `features` and `adj` are not given, this function will use previous stored `features` and `adj` from training to make predictions.


        Returns
        -------
        torch.FloatTensor
            output (log probabilities) of GCNJaccard
        """
        self.eval()
        if features is None and adj is None:
            return self.forward(self.features, self.adj_norm)
        else:
            adj = self.drop_dissimilar_edges(features, adj)
            if type(adj) is not torch.Tensor:
                features, adj = utils.to_tensor(features, adj, device=self.device)
            self.features = features
            if utils.is_sparse_tensor(adj):
                self.adj_norm = utils.normalize_adj_tensor(adj, sparse=True)
            else:
                self.adj_norm = utils.normalize_adj_tensor(adj)
            return self.forward(self.features, self.adj_norm)

    def _drop_dissimilar_edges(self, features, adj):
        """Drop dissimilar edges. (Slower version)
        """
        if not sp.issparse(adj):
            adj = sp.csr_matrix(adj)
        modified_adj = adj.copy().tolil()
        None
        edges = np.array(modified_adj.nonzero()).T
        removed_cnt = 0
        for edge in tqdm(edges):
            n1 = edge[0]
            n2 = edge[1]
            if n1 > n2:
                continue
            if self.binary_feature:
                J = self._jaccard_similarity(features[n1], features[n2])
                if J < self.threshold:
                    modified_adj[n1, n2] = 0
                    modified_adj[n2, n1] = 0
                    removed_cnt += 1
            else:
                C = self._cosine_similarity(features[n1], features[n2])
                if C < self.threshold:
                    modified_adj[n1, n2] = 0
                    modified_adj[n2, n1] = 0
                    removed_cnt += 1
        None
        return modified_adj

    def _jaccard_similarity(self, a, b):
        intersection = a.multiply(b).count_nonzero()
        J = intersection * 1.0 / (a.count_nonzero() + b.count_nonzero() - intersection)
        return J

    def _cosine_similarity(self, a, b):
        inner_product = (a * b).sum()
        C = inner_product / (np.sqrt(np.square(a).sum()) * np.sqrt(np.square(b).sum()) + 1e-10)
        return C


class MedianGCN(torch.nn.Module):
    """Graph Convolutional Networks with Median aggregation (MedianGCN) 
    based on pytorch geometric. 

    `Understanding Structural Vulnerability in Graph Convolutional Networks 
    <https://arxiv.org/abs/2108.06280>`

    MedianGCN uses median aggregation function instead of 
    `weighted mean` adopted in GCN, which improves the robustness 
    of the model against adversarial structural attack.

    Parameters
    ----------
    nfeat : int
        size of input feature dimension
    nhid : int
        number of hidden units        
    nclass : int
        size of output dimension
    lr : float
        learning rate for MedianGCN
    weight_decay : float
        weight decay coefficient (l2 normalization) for MedianGCN.
    with_bias: bool
        whether to include bias term in MedianGCN weights.
    device: str
        'cpu' or 'cuda'.

    Examples
    --------
        We can first load dataset and then train MedianGCN.

    >>> from deeprobust.graph.data import Dataset
    >>> from deeprobust.graph.defense import MedianGCN
    >>> data = Dataset(root='/tmp/', name='cora')
    >>> adj, features, labels = data.adj, data.features, data.labels
    >>> idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
    >>> MedianGCN = MedianGCN(nfeat=features.shape[1],
                          nhid=16, nclass=labels.max().item() + 1, 
                          device='cuda')
    >>> MedianGCN = MedianGCN.to('cuda')
    >>> pyg_data = Dpr2Pyg(data) # convert deeprobust dataset to pyg dataset
    >>> MedianGCN.fit(pyg_data, verbose=True) # train with earlystopping
    """

    def __init__(self, nfeat, nhid, nclass, dropout=0.5, lr=0.01, weight_decay=0.0005, with_bias=True, device=None):
        super(MedianGCN, self).__init__()
        assert device is not None, "Please specify 'device'!"
        self.device = device
        self.conv1 = MedianConv(nfeat, nhid, bias=with_bias)
        self.conv2 = MedianConv(nhid, nclass, bias=with_bias)
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.with_bias = with_bias
        self.output = None

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

    def initialize(self):
        """Initialize parameters of MedianGCN.
        """
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def fit(self, pyg_data, train_iters=200, initialize=True, verbose=False, patience=500, **kwargs):
        """Train the MedianGCN model, when idx_val is not None, pick the best model
        according to the validation loss.

        Parameters
        ----------
        pyg_data :
            pytorch geometric dataset object
        train_iters : int
            number of training epochs
        initialize : bool
            whether to initialize parameters before training
        verbose : bool
            whether to show verbose logs
        patience : int
            patience for early stopping, only valid when `idx_val` is given
        """
        if initialize:
            self.initialize()
        self.data = pyg_data[0]
        self.train_with_early_stopping(train_iters, patience, verbose)

    def train_with_early_stopping(self, train_iters, patience, verbose):
        """early stopping based on the validation loss
        """
        if verbose:
            None
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        labels = self.data.y
        train_mask, val_mask = self.data.train_mask, self.data.val_mask
        early_stopping = patience
        best_loss_val = 100
        for i in range(train_iters):
            self.train()
            optimizer.zero_grad()
            output = self.forward(self.data)
            loss_train = F.nll_loss(output[train_mask], labels[train_mask])
            loss_train.backward()
            optimizer.step()
            if verbose and i % 10 == 0:
                None
            self.eval()
            output = self.forward(self.data)
            loss_val = F.nll_loss(output[val_mask], labels[val_mask])
            if best_loss_val > loss_val:
                best_loss_val = loss_val
                self.output = output
                weights = deepcopy(self.state_dict())
                patience = early_stopping
            else:
                patience -= 1
            if i > early_stopping and patience <= 0:
                break
        if verbose:
            None
        self.load_state_dict(weights)

    @torch.no_grad()
    def test(self, pyg_data=None):
        """Evaluate MedianGCN performance on test set.

        Parameters
        ----------
        pyg_data :
            pytorch geometric dataset object        
        idx_test :
            node testing indices
        """
        self.eval()
        data = pyg_data[0] if pyg_data is not None else self.data
        test_mask = data.test_mask
        labels = data.y
        output = self.forward(data)
        loss_test = F.nll_loss(output[test_mask], labels[test_mask])
        acc_test = utils.accuracy(output[test_mask], labels[test_mask])
        None
        return acc_test.item()

    @torch.no_grad()
    def predict(self, pyg_data=None):
        """
        Parameters
        ----------
        pyg_data :
            pytorch geometric dataset object    

        Returns
        -------
        torch.FloatTensor
            output (log probabilities) of MedianGCN
        """
        self.eval()
        data = pyg_data[0] if pyg_data is not None else self.data
        return self.forward(data)


class EstimateAdj(nn.Module):
    """Provide a pytorch parameter matrix for estimated
    adjacency matrix and corresponding operations.
    """

    def __init__(self, adj, symmetric=False, device='cpu'):
        super(EstimateAdj, self).__init__()
        n = len(adj)
        self.estimated_adj = nn.Parameter(torch.FloatTensor(n, n))
        self._init_estimation(adj)
        self.symmetric = symmetric
        self.device = device

    def _init_estimation(self, adj):
        with torch.no_grad():
            n = len(adj)
            self.estimated_adj.data.copy_(adj)

    def forward(self):
        return self.estimated_adj

    def normalize(self):
        if self.symmetric:
            adj = (self.estimated_adj + self.estimated_adj.t()) / 2
        else:
            adj = self.estimated_adj
        normalized_adj = self._normalize(adj + torch.eye(adj.shape[0]))
        return normalized_adj

    def _normalize(self, mx):
        rowsum = mx.sum(1)
        r_inv = rowsum.pow(-1 / 2).flatten()
        r_inv[torch.isinf(r_inv)] = 0.0
        r_mat_inv = torch.diag(r_inv)
        mx = r_mat_inv @ mx
        mx = mx @ r_mat_inv
        return mx


class GGCL_F(Module):
    """Graph Gaussian Convolution Layer (GGCL) when the input is feature"""

    def __init__(self, in_features, out_features, dropout=0.6):
        super(GGCL_F, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.weight_miu = Parameter(torch.FloatTensor(in_features, out_features))
        self.weight_sigma = Parameter(torch.FloatTensor(in_features, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight_miu)
        torch.nn.init.xavier_uniform_(self.weight_sigma)

    def forward(self, features, adj_norm1, adj_norm2, gamma=1):
        features = F.dropout(features, self.dropout, training=self.training)
        self.miu = F.elu(torch.mm(features, self.weight_miu))
        self.sigma = F.relu(torch.mm(features, self.weight_sigma))
        Att = torch.exp(-gamma * self.sigma)
        miu_out = adj_norm1 @ (self.miu * Att)
        sigma_out = adj_norm2 @ (self.sigma * Att * Att)
        return miu_out, sigma_out


class GGCL_D(Module):
    """Graph Gaussian Convolution Layer (GGCL) when the input is distribution"""

    def __init__(self, in_features, out_features, dropout):
        super(GGCL_D, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.weight_miu = Parameter(torch.FloatTensor(in_features, out_features))
        self.weight_sigma = Parameter(torch.FloatTensor(in_features, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight_miu)
        torch.nn.init.xavier_uniform_(self.weight_sigma)

    def forward(self, miu, sigma, adj_norm1, adj_norm2, gamma=1):
        miu = F.dropout(miu, self.dropout, training=self.training)
        sigma = F.dropout(sigma, self.dropout, training=self.training)
        miu = F.elu(miu @ self.weight_miu)
        sigma = F.relu(sigma @ self.weight_sigma)
        Att = torch.exp(-gamma * sigma)
        mean_out = adj_norm1 @ (miu * Att)
        sigma_out = adj_norm2 @ (sigma * Att * Att)
        return mean_out, sigma_out


class GaussianConvolution(Module):
    """[Deprecated] Alternative gaussion convolution layer.
    """

    def __init__(self, in_features, out_features):
        super(GaussianConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_miu = Parameter(torch.FloatTensor(in_features, out_features))
        self.weight_sigma = Parameter(torch.FloatTensor(in_features, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight_miu)
        torch.nn.init.xavier_uniform_(self.weight_sigma)

    def forward(self, previous_miu, previous_sigma, adj_norm1=None, adj_norm2=None, gamma=1):
        if adj_norm1 is None and adj_norm2 is None:
            return torch.mm(previous_miu, self.weight_miu), torch.mm(previous_miu, self.weight_miu)
        Att = torch.exp(-gamma * previous_sigma)
        M = adj_norm1 @ (previous_miu * Att) @ self.weight_miu
        Sigma = adj_norm2 @ (previous_sigma * Att * Att) @ self.weight_sigma
        return M, Sigma

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class RGCN(Module):
    """Robust Graph Convolutional Networks Against Adversarial Attacks. KDD 2019.

    Parameters
    ----------
    nnodes : int
        number of nodes in the input grpah
    nfeat : int
        size of input feature dimension
    nhid : int
        number of hidden units
    nclass : int
        size of output dimension
    gamma : float
        hyper-parameter for RGCN. See more details in the paper.
    beta1 : float
        hyper-parameter for RGCN. See more details in the paper.
    beta2 : float
        hyper-parameter for RGCN. See more details in the paper.
    lr : float
        learning rate for GCN
    dropout : float
        dropout rate for GCN
    device: str
        'cpu' or 'cuda'.

    """

    def __init__(self, nnodes, nfeat, nhid, nclass, gamma=1.0, beta1=0.0005, beta2=0.0005, lr=0.01, dropout=0.6, device='cpu'):
        super(RGCN, self).__init__()
        self.device = device
        self.lr = lr
        self.gamma = gamma
        self.beta1 = beta1
        self.beta2 = beta2
        self.nclass = nclass
        self.nhid = nhid // 2
        self.gc1 = GGCL_F(nfeat, nhid, dropout=dropout)
        self.gc2 = GGCL_D(nhid, nclass, dropout=dropout)
        self.dropout = dropout
        self.gaussian = MultivariateNormal(torch.zeros(nnodes, self.nclass), torch.diag_embed(torch.ones(nnodes, self.nclass)))
        self.adj_norm1, self.adj_norm2 = None, None
        self.features, self.labels = None, None

    def forward(self):
        features = self.features
        miu, sigma = self.gc1(features, self.adj_norm1, self.adj_norm2, self.gamma)
        miu, sigma = self.gc2(miu, sigma, self.adj_norm1, self.adj_norm2, self.gamma)
        output = miu + self.gaussian.sample() * torch.sqrt(sigma + 1e-08)
        return F.log_softmax(output, dim=1)

    def fit(self, features, adj, labels, idx_train, idx_val=None, train_iters=200, verbose=True, **kwargs):
        """Train RGCN.

        Parameters
        ----------
        features :
            node features
        adj :
            the adjacency matrix. The format could be torch.tensor or scipy matrix
        labels :
            node labels
        idx_train :
            node training indices
        idx_val :
            node validation indices. If not given (None), GCN training process will not adpot early stopping
        train_iters : int
            number of training epochs
        verbose : bool
            whether to show verbose logs

        Examples
        --------
        We can first load dataset and then train RGCN.

        >>> from deeprobust.graph.data import PrePtbDataset, Dataset
        >>> from deeprobust.graph.defense import RGCN
        >>> # load clean graph data
        >>> data = Dataset(root='/tmp/', name='cora', seed=15)
        >>> adj, features, labels = data.adj, data.features, data.labels
        >>> idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
        >>> # load perturbed graph data
        >>> perturbed_data = PrePtbDataset(root='/tmp/', name='cora')
        >>> perturbed_adj = perturbed_data.adj
        >>> # train defense model
        >>> model = RGCN(nnodes=perturbed_adj.shape[0], nfeat=features.shape[1],
                         nclass=labels.max()+1, nhid=32, device='cpu')
        >>> model.fit(features, perturbed_adj, labels, idx_train, idx_val,
                      train_iters=200, verbose=True)
        >>> model.test(idx_test)

        """
        adj, features, labels = utils.to_tensor(adj.todense(), features.todense(), labels, device=self.device)
        self.features, self.labels = features, labels
        self.adj_norm1 = self._normalize_adj(adj, power=-1 / 2)
        self.adj_norm2 = self._normalize_adj(adj, power=-1)
        None
        self._initialize()
        if idx_val is None:
            self._train_without_val(labels, idx_train, train_iters, verbose)
        else:
            self._train_with_val(labels, idx_train, idx_val, train_iters, verbose)

    def _train_without_val(self, labels, idx_train, train_iters, verbose=True):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        self.train()
        for i in range(train_iters):
            optimizer.zero_grad()
            output = self.forward()
            loss_train = self._loss(output[idx_train], labels[idx_train])
            loss_train.backward()
            optimizer.step()
            if verbose and i % 10 == 0:
                None
        self.eval()
        output = self.forward()
        self.output = output

    def _train_with_val(self, labels, idx_train, idx_val, train_iters, verbose):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        best_loss_val = 100
        best_acc_val = 0
        for i in range(train_iters):
            self.train()
            optimizer.zero_grad()
            output = self.forward()
            loss_train = self._loss(output[idx_train], labels[idx_train])
            loss_train.backward()
            optimizer.step()
            if verbose and i % 10 == 0:
                None
            self.eval()
            output = self.forward()
            loss_val = F.nll_loss(output[idx_val], labels[idx_val])
            acc_val = utils.accuracy(output[idx_val], labels[idx_val])
            if best_loss_val > loss_val:
                best_loss_val = loss_val
                self.output = output
            if acc_val > best_acc_val:
                best_acc_val = acc_val
                self.output = output
        None

    def test(self, idx_test):
        """Evaluate the peformance on test set
        """
        self.eval()
        output = self.output
        loss_test = F.nll_loss(output[idx_test], self.labels[idx_test])
        acc_test = utils.accuracy(output[idx_test], self.labels[idx_test])
        None
        return acc_test.item()

    def predict(self):
        """
        Returns
        -------
        torch.FloatTensor
            output (log probabilities) of RGCN
        """
        self.eval()
        return self.forward()

    def _loss(self, input, labels):
        loss = F.nll_loss(input, labels)
        miu1 = self.gc1.miu
        sigma1 = self.gc1.sigma
        kl_loss = 0.5 * (miu1.pow(2) + sigma1 - torch.log(1e-08 + sigma1)).mean(1)
        kl_loss = kl_loss.sum()
        norm2 = torch.norm(self.gc1.weight_miu, 2).pow(2) + torch.norm(self.gc1.weight_sigma, 2).pow(2)
        return loss + self.beta1 * kl_loss + self.beta2 * norm2

    def _initialize(self):
        self.gc1.reset_parameters()
        self.gc2.reset_parameters()

    def _normalize_adj(self, adj, power=-1 / 2):
        """Row-normalize sparse matrix"""
        A = adj + torch.eye(len(adj))
        D_power = A.sum(1).pow(power)
        D_power[torch.isinf(D_power)] = 0.0
        D_power = torch.diag(D_power)
        return D_power @ A @ D_power


class SGC(torch.nn.Module):
    """ SGC based on pytorch geometric. Simplifying Graph Convolutional Networks.

    Parameters
    ----------
    nfeat : int
        size of input feature dimension
    nclass : int
        size of output dimension
    K: int
        number of propagation in SGC
    cached : bool
        whether to set the cache flag in SGConv
    lr : float
        learning rate for SGC
    weight_decay : float
        weight decay coefficient (l2 normalization) for GCN.
        When `with_relu` is True, `weight_decay` will be set to 0.
    with_bias: bool
        whether to include bias term in SGC weights.
    device: str
        'cpu' or 'cuda'.

    Examples
    --------
	We can first load dataset and then train SGC.

    >>> from deeprobust.graph.data import Dataset
    >>> from deeprobust.graph.defense import SGC
    >>> data = Dataset(root='/tmp/', name='cora')
    >>> adj, features, labels = data.adj, data.features, data.labels
    >>> idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
    >>> sgc = SGC(nfeat=features.shape[1], K=3, lr=0.1,
              nclass=labels.max().item() + 1, device='cuda')
    >>> sgc = sgc.to('cuda')
    >>> pyg_data = Dpr2Pyg(data) # convert deeprobust dataset to pyg dataset
    >>> sgc.fit(pyg_data, train_iters=200, patience=200, verbose=True) # train with earlystopping
    """

    def __init__(self, nfeat, nclass, K=3, cached=True, lr=0.01, weight_decay=0.0005, with_bias=True, device=None):
        super(SGC, self).__init__()
        assert device is not None, "Please specify 'device'!"
        self.device = device
        self.conv1 = SGConv(nfeat, nclass, bias=with_bias, K=K, cached=cached)
        self.weight_decay = weight_decay
        self.lr = lr
        self.output = None
        self.best_model = None
        self.best_output = None

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        return F.log_softmax(x, dim=1)

    def initialize(self):
        """Initialize parameters of SGC.
        """
        self.conv1.reset_parameters()

    def fit(self, pyg_data, train_iters=200, initialize=True, verbose=False, patience=500, **kwargs):
        """Train the SGC model, when idx_val is not None, pick the best model
        according to the validation loss.

        Parameters
        ----------
        pyg_data :
            pytorch geometric dataset object
        train_iters : int
            number of training epochs
        initialize : bool
            whether to initialize parameters before training
        verbose : bool
            whether to show verbose logs
        patience : int
            patience for early stopping, only valid when `idx_val` is given
        """
        if initialize:
            self.initialize()
        self.data = pyg_data[0]
        self.train_with_early_stopping(train_iters, patience, verbose)

    def train_with_early_stopping(self, train_iters, patience, verbose):
        """early stopping based on the validation loss
        """
        if verbose:
            None
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        labels = self.data.y
        train_mask, val_mask = self.data.train_mask, self.data.val_mask
        early_stopping = patience
        best_loss_val = 100
        for i in range(train_iters):
            self.train()
            optimizer.zero_grad()
            output = self.forward(self.data)
            loss_train = F.nll_loss(output[train_mask], labels[train_mask])
            loss_train.backward()
            optimizer.step()
            if verbose and i % 10 == 0:
                None
            self.eval()
            output = self.forward(self.data)
            loss_val = F.nll_loss(output[val_mask], labels[val_mask])
            if best_loss_val > loss_val:
                best_loss_val = loss_val
                self.output = output
                weights = deepcopy(self.state_dict())
                patience = early_stopping
            else:
                patience -= 1
            if i > early_stopping and patience <= 0:
                break
        if verbose:
            None
        self.load_state_dict(weights)

    def test(self):
        """Evaluate SGC performance on test set.

        Parameters
        ----------
        idx_test :
            node testing indices
        """
        self.eval()
        test_mask = self.data.test_mask
        labels = self.data.y
        output = self.forward(self.data)
        loss_test = F.nll_loss(output[test_mask], labels[test_mask])
        acc_test = utils.accuracy(output[test_mask], labels[test_mask])
        None
        return acc_test.item()

    def predict(self):
        """
        Returns
        -------
        torch.FloatTensor
            output (log probabilities) of SGC
        """
        self.eval()
        return self.forward(self.data)


class AttrSim:

    def __init__(self, features):
        self.features = features.cpu().numpy()
        self.features[self.features != 0] = 1

    def get_label(self, k=5):
        features = self.features
        if not os.path.exists('saved_knn/cosine_sims_{}.npy'.format(features.shape)):
            sims = cosine_similarity(features)
            np.save('saved_knn/cosine_sims_{}.npy'.format(features.shape), sims)
        else:
            None
            sims = np.load('saved_knn/cosine_sims_{}.npy'.format(features.shape))
        if not os.path.exists('saved_knn/attrsim_sampled_idx_{}.npy'.format(features.shape)):
            try:
                indices_sorted = sims.argsort(1)
                idx = np.arange(k, sims.shape[0] - k)
                selected = np.hstack((indices_sorted[:, :k], indices_sorted[:, -k - 1:]))
                selected_set = set()
                for i in range(len(sims)):
                    for pair in product([i], selected[i]):
                        if pair[0] > pair[1]:
                            pair = pair[1], pair[0]
                        if pair[0] == pair[1]:
                            continue
                        selected_set.add(pair)
            except MemoryError:
                selected_set = set()
                for ii, row in tqdm(enumerate(sims)):
                    row = row.argsort()
                    idx = np.arange(k, sims.shape[0] - k)
                    sampled = np.random.choice(idx, k, replace=False)
                    for node in np.hstack((row[:k], row[-k - 1:], row[sampled])):
                        if ii > node:
                            pair = node, ii
                        else:
                            pair = ii, node
                        selected_set.add(pair)
            sampled = np.array(list(selected_set)).transpose()
            np.save('saved_knn/attrsim_sampled_idx_{}.npy'.format(features.shape), sampled)
        else:
            None
            sampled = np.load('saved_knn/attrsim_sampled_idx_{}.npy'.format(features.shape))
        None
        self.node_pairs = sampled[0], sampled[1]
        self.sims = sims
        return torch.FloatTensor(sims[self.node_pairs]).reshape(-1, 1)


def noaug_normalized_adjacency(adj):
    adj = sp.coo_matrix(adj)
    row_sum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj_noloop(adj, device):
    adj_normalizer = noaug_normalized_adjacency
    r_adj = adj_normalizer(adj)
    r_adj = utils.sparse_mx_to_torch_sparse_tensor(r_adj).float()
    r_adj = r_adj
    return r_adj


class SimPGCN(nn.Module):
    """SimP-GCN: Node similarity preserving graph convolutional networks.
       https://arxiv.org/abs/2011.09643

    Parameters
    ----------
    nnodes : int
        number of nodes in the input grpah
    nfeat : int
        size of input feature dimension
    nhid : int
        number of hidden units
    nclass : int
        size of output dimension
    lambda_ : float
        coefficients for SSL loss in SimP-GCN
    gamma : float
        coefficients for adaptive learnable self-loops
    bias_init : float
        bias init for the score
    dropout : float
        dropout rate for GCN
    lr : float
        learning rate for GCN
    weight_decay : float
        weight decay coefficient (l2 normalization) for GCN. When `with_relu` is True, `weight_decay` will be set to 0.
    with_bias: bool
        whether to include bias term in GCN weights.
    device: str
        'cpu' or 'cuda'.

    Examples
    --------
	We can first load dataset and then train SimPGCN.
    See the detailed hyper-parameter setting in https://github.com/ChandlerBang/SimP-GCN.

    >>> from deeprobust.graph.data import PrePtbDataset, Dataset
    >>> from deeprobust.graph.defense import SimPGCN
    >>> # load clean graph data
    >>> data = Dataset(root='/tmp/', name='cora', seed=15)
    >>> adj, features, labels = data.adj, data.features, data.labels
    >>> idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
    >>> # load perturbed graph data
    >>> perturbed_data = PrePtbDataset(root='/tmp/', name='cora')
    >>> perturbed_adj = perturbed_data.adj
    >>> model = SimPGCN(nnodes=features.shape[0], nfeat=features.shape[1],
        nhid=16, nclass=labels.max()+1, device='cuda')
    >>> model = model.to('cuda')
    >>> model.fit(features, perturbed_adj, labels, idx_train, idx_val, train_iters=200, verbose=True)
    >>> model.test(idx_test)
    """

    def __init__(self, nnodes, nfeat, nhid, nclass, dropout=0.5, lr=0.01, weight_decay=0.0005, lambda_=5, gamma=0.1, bias_init=0, with_bias=True, device=None):
        super(SimPGCN, self).__init__()
        assert device is not None, "Please specify 'device'!"
        self.device = device
        self.nfeat = nfeat
        self.hidden_sizes = [nhid]
        self.nclass = nclass
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.bias_init = bias_init
        self.gamma = gamma
        self.lambda_ = lambda_
        self.output = None
        self.best_model = None
        self.best_output = None
        self.adj_norm = None
        self.features = None
        self.gc1 = GraphConvolution(nfeat, nhid, with_bias=with_bias)
        self.gc2 = GraphConvolution(nhid, nclass, with_bias=with_bias)
        self.scores = nn.ParameterList()
        self.scores.append(Parameter(torch.FloatTensor(nfeat, 1)))
        for i in range(1):
            self.scores.append(Parameter(torch.FloatTensor(nhid, 1)))
        self.bias = nn.ParameterList()
        self.bias.append(Parameter(torch.FloatTensor(1)))
        for i in range(1):
            self.bias.append(Parameter(torch.FloatTensor(1)))
        self.D_k = nn.ParameterList()
        self.D_k.append(Parameter(torch.FloatTensor(nfeat, 1)))
        for i in range(1):
            self.D_k.append(Parameter(torch.FloatTensor(nhid, 1)))
        self.identity = utils.sparse_mx_to_torch_sparse_tensor(sp.eye(nnodes))
        self.D_bias = nn.ParameterList()
        self.D_bias.append(Parameter(torch.FloatTensor(1)))
        for i in range(1):
            self.D_bias.append(Parameter(torch.FloatTensor(1)))
        self.linear = nn.Linear(nhid, 1)
        self.adj_knn = None
        self.pseudo_labels = None

    def get_knn_graph(self, features, k=20):
        if not os.path.exists('saved_knn/'):
            os.mkdir('saved_knn')
        if not os.path.exists('saved_knn/knn_graph_{}.npz'.format(features.shape)):
            features[features != 0] = 1
            sims = cosine_similarity(features)
            np.save('saved_knn/cosine_sims_{}.npy'.format(features.shape), sims)
            sims[np.arange(len(sims)), np.arange(len(sims))] = 0
            for i in range(len(sims)):
                indices_argsort = np.argsort(sims[i])
                sims[i, indices_argsort[:-k]] = 0
            adj_knn = sp.csr_matrix(sims)
            sp.save_npz('saved_knn/knn_graph_{}.npz'.format(features.shape), adj_knn)
        else:
            None
            adj_knn = sp.load_npz('saved_knn/knn_graph_{}.npz'.format(features.shape))
        return preprocess_adj_noloop(adj_knn, self.device)

    def initialize(self):
        """Initialize parameters of SimPGCN.
        """
        self.gc1.reset_parameters()
        self.gc2.reset_parameters()
        for s in self.scores:
            stdv = 1.0 / math.sqrt(s.size(1))
            s.data.uniform_(-stdv, stdv)
        for b in self.bias:
            b.data.fill_(self.bias_init)
        for Dk in self.D_k:
            stdv = 1.0 / math.sqrt(Dk.size(1))
            Dk.data.uniform_(-stdv, stdv)
        for b in self.D_bias:
            b.data.fill_(0)

    def fit(self, features, adj, labels, idx_train, idx_val=None, train_iters=200, initialize=True, verbose=False, normalize=True, patience=500, **kwargs):
        if initialize:
            self.initialize()
        if type(adj) is not torch.Tensor:
            features, adj, labels = utils.to_tensor(features, adj, labels, device=self.device)
        else:
            features = features
            adj = adj
            labels = labels
        if normalize:
            if utils.is_sparse_tensor(adj):
                adj_norm = utils.normalize_adj_tensor(adj, sparse=True)
            else:
                adj_norm = utils.normalize_adj_tensor(adj)
        else:
            adj_norm = adj
        self.adj_norm = adj_norm
        self.features = features
        self.labels = labels
        if idx_val is None:
            self._train_without_val(labels, idx_train, train_iters, verbose)
        elif patience < train_iters:
            self._train_with_early_stopping(labels, idx_train, idx_val, train_iters, patience, verbose)
        else:
            self._train_with_val(labels, idx_train, idx_val, train_iters, verbose)

    def forward(self, fea, adj):
        x, _ = self.myforward(fea, adj)
        return x

    def myforward(self, fea, adj):
        """output embedding and log_softmax"""
        if self.adj_knn is None:
            self.adj_knn = self.get_knn_graph(fea.to_dense().cpu().numpy())
        adj_knn = self.adj_knn
        gamma = self.gamma
        s_i = torch.sigmoid(fea @ self.scores[0] + self.bias[0])
        Dk_i = fea @ self.D_k[0] + self.D_bias[0]
        x = s_i * self.gc1(fea, adj) + (1 - s_i) * self.gc1(fea, adj_knn) + gamma * Dk_i * self.gc1(fea, self.identity)
        x = F.dropout(x, self.dropout, training=self.training)
        embedding = x.clone()
        s_o = torch.sigmoid(x @ self.scores[-1] + self.bias[-1])
        Dk_o = x @ self.D_k[-1] + self.D_bias[-1]
        x = s_o * self.gc2(x, adj) + (1 - s_o) * self.gc2(x, adj_knn) + gamma * Dk_o * self.gc2(x, self.identity)
        x = F.log_softmax(x, dim=1)
        self.ss = torch.cat((s_i.view(1, -1), s_o.view(1, -1), gamma * Dk_i.view(1, -1), gamma * Dk_o.view(1, -1)), dim=0)
        return x, embedding

    def regression_loss(self, embeddings):
        if self.pseudo_labels is None:
            agent = AttrSim(self.features.to_dense())
            self.pseudo_labels = agent.get_label()
            node_pairs = agent.node_pairs
            self.node_pairs = node_pairs
        k = 10000
        node_pairs = self.node_pairs
        if len(self.node_pairs[0]) > k:
            sampled = np.random.choice(len(self.node_pairs[0]), k, replace=False)
            embeddings0 = embeddings[node_pairs[0][sampled]]
            embeddings1 = embeddings[node_pairs[1][sampled]]
            embeddings = self.linear(torch.abs(embeddings0 - embeddings1))
            loss = F.mse_loss(embeddings, self.pseudo_labels[sampled], reduction='mean')
        else:
            embeddings0 = embeddings[node_pairs[0]]
            embeddings1 = embeddings[node_pairs[1]]
            embeddings = self.linear(torch.abs(embeddings0 - embeddings1))
            loss = F.mse_loss(embeddings, self.pseudo_labels, reduction='mean')
        return loss

    def _train_without_val(self, labels, idx_train, train_iters, verbose):
        self.train()
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        for i in range(train_iters):
            self.train()
            optimizer.zero_grad()
            output, embeddings = self.myforward(self.features, self.adj_norm)
            loss_train = F.nll_loss(output[idx_train], labels[idx_train])
            loss_ssl = self.lambda_ * self.regression_loss(embeddings)
            loss_total = loss_train + loss_ssl
            loss_total.backward()
            optimizer.step()
            if verbose and i % 10 == 0:
                None
        self.eval()
        output = self.forward(self.features, self.adj_norm)
        self.output = output

    def _train_with_val(self, labels, idx_train, idx_val, train_iters, verbose):
        if verbose:
            None
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        best_loss_val = 100
        best_acc_val = 0
        for i in range(train_iters):
            self.train()
            optimizer.zero_grad()
            output, embeddings = self.myforward(self.features, self.adj_norm)
            loss_train = F.nll_loss(output[idx_train], labels[idx_train])
            loss_ssl = self.lambda_ * self.regression_loss(embeddings)
            loss_total = loss_train + loss_ssl
            loss_total.backward()
            optimizer.step()
            if verbose and i % 10 == 0:
                None
            self.eval()
            output = self.forward(self.features, self.adj_norm)
            loss_val = F.nll_loss(output[idx_val], labels[idx_val])
            acc_val = utils.accuracy(output[idx_val], labels[idx_val])
            if best_loss_val > loss_val:
                best_loss_val = loss_val
                self.output = output
                weights = deepcopy(self.state_dict())
            if acc_val > best_acc_val:
                best_acc_val = acc_val
                self.output = output
                weights = deepcopy(self.state_dict())
        if verbose:
            None
        self.load_state_dict(weights)

    def _train_with_early_stopping(self, labels, idx_train, idx_val, train_iters, patience, verbose):
        if verbose:
            None
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        early_stopping = patience
        best_loss_val = 100
        for i in range(train_iters):
            self.train()
            optimizer.zero_grad()
            output, embeddings = self.myforward(self.features, self.adj_norm)
            loss_train = F.nll_loss(output[idx_train], labels[idx_train])
            loss_ssl = self.lambda_ * self.regression_loss(embeddings)
            loss_total = loss_train + loss_ssl
            loss_total.backward()
            optimizer.step()
            if verbose and i % 10 == 0:
                None
            self.eval()
            output = self.forward(self.features, self.adj_norm)
            loss_val = F.nll_loss(output[idx_val], labels[idx_val])
            if best_loss_val > loss_val:
                best_loss_val = loss_val
                self.output = output
                weights = deepcopy(self.state_dict())
                patience = early_stopping
            else:
                patience -= 1
            if i > early_stopping and patience <= 0:
                break
        if verbose:
            None
        self.load_state_dict(weights)

    def test(self, idx_test):
        """Evaluate GCN performance on test set.

        Parameters
        ----------
        idx_test :
            node testing indices
        """
        self.eval()
        output = self.predict()
        loss_test = F.nll_loss(output[idx_test], self.labels[idx_test])
        acc_test = utils.accuracy(output[idx_test], self.labels[idx_test])
        None
        return acc_test.item()

    def predict(self, features=None, adj=None):
        """By default, the inputs should be unnormalized data

        Parameters
        ----------
        features :
            node features. If `features` and `adj` are not given, this function will use previous stored `features` and `adj` from training to make predictions.
        adj :
            adjcency matrix. If `features` and `adj` are not given, this function will use previous stored `features` and `adj` from training to make predictions.


        Returns
        -------
        torch.FloatTensor
            output (log probabilities) of GCN
        """
        self.eval()
        if features is None and adj is None:
            return self.forward(self.features, self.adj_norm)
        else:
            if type(adj) is not torch.Tensor:
                features, adj = utils.to_tensor(features, adj, device=self.device)
            self.features = features
            if utils.is_sparse_tensor(adj):
                self.adj_norm = utils.normalize_adj_tensor(adj, sparse=True)
            else:
                self.adj_norm = utils.normalize_adj_tensor(adj)
            return self.forward(self.features, self.adj_norm)


class APPNP(BaseModel):

    def __init__(self, nfeat, nhid, nclass, K=10, alpha=0.1, dropout=0.5, lr=0.01, with_bn=False, weight_decay=0.0005, with_bias=True, device=None):
        super(APPNP, self).__init__()
        assert device is not None, "Please specify 'device'!"
        self.device = device
        self.lin1 = Linear(nfeat, nhid)
        if with_bn:
            self.bn1 = nn.BatchNorm1d(nhid)
            self.bn2 = nn.BatchNorm1d(nclass)
        self.lin2 = Linear(nhid, nclass)
        self.prop1 = APPNPConv(K, alpha)
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.lr = lr
        self.output = None
        self.best_model = None
        self.best_output = None
        self.name = 'APPNP'
        self.with_bn = with_bn

    def forward(self, x, edge_index, edge_weight=None):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin1(x)
        if self.with_bn:
            x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        if self.with_bn:
            x = self.bn2(x)
        x = self.prop1(x, edge_index, edge_weight)
        return F.log_softmax(x, dim=1)

    def initialize(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        if self.with_bn:
            self.bn1.reset_parameters()
            self.bn2.reset_parameters()


class GPRGNN(BaseModel):
    """GPRGNN, from original repo https://github.com/jianhao2016/GPRGNN"""

    def __init__(self, nfeat, nhid, nclass, Init='PPR', dprate=0.5, dropout=0.5, lr=0.01, weight_decay=0, device='cpu', K=10, alpha=0.1, Gamma=None, ppnp='GPR_prop'):
        super(GPRGNN, self).__init__()
        self.lin1 = nn.Linear(nfeat, nhid)
        self.lin2 = nn.Linear(nhid, nclass)
        if ppnp == 'PPNP':
            self.prop1 = APPNP(K, alpha)
        elif ppnp == 'GPR_prop':
            self.prop1 = GPR_prop(K, alpha, Init, Gamma)
        self.Init = Init
        self.dprate = dprate
        self.dropout = dropout
        self.name = 'GPR'
        self.weight_decay = weight_decay
        self.lr = lr
        self.device = device

    def initialize(self):
        self.reset_parameters()

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.prop1.reset_parameters()

    def forward(self, x, edge_index, edge_weight=None):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        if edge_weight is not None:
            adj = SparseTensor.from_edge_index(edge_index, edge_weight, sparse_sizes=2 * x.shape[:1])
            if self.dprate == 0.0:
                x = self.prop1(x, adj)
            else:
                x = F.dropout(x, p=self.dprate, training=self.training)
                x = self.prop1(x, adj)
        elif self.dprate == 0.0:
            x = self.prop1(x, edge_index, edge_weight)
        else:
            x = F.dropout(x, p=self.dprate, training=self.training)
            x = self.prop1(x, edge_index, edge_weight)
        return F.log_softmax(x, dim=1)


class SAGE(BaseModel):

    def __init__(self, nfeat, nhid, nclass, num_layers=2, dropout=0.5, lr=0.01, weight_decay=0, device='cpu', with_bn=False, **kwargs):
        super(SAGE, self).__init__()
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(nfeat, nhid))
        self.bns = nn.ModuleList()
        if 'nlayers' in kwargs:
            num_layers = kwargs['nlayers']
        self.bns.append(nn.BatchNorm1d(nhid))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(nhid, nhid))
            self.bns.append(nn.BatchNorm1d(nhid))
        self.convs.append(SAGEConv(nhid, nclass))
        self.weight_decay = weight_decay
        self.lr = lr
        self.dropout = dropout
        self.activation = F.relu
        self.with_bn = with_bn
        self.device = device
        self.name = 'SAGE'

    def initialize(self):
        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, edge_index, edge_weight=None):
        if edge_weight is not None:
            adj = SparseTensor.from_edge_index(edge_index, edge_weight, sparse_sizes=2 * x.shape[:1]).t()
        for i, conv in enumerate(self.convs[:-1]):
            if edge_weight is not None:
                x = conv(x, adj)
            else:
                x = conv(x, edge_index, edge_weight)
            if self.with_bn:
                x = self.bns[i](x)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        if edge_weight is not None:
            x = self.convs[-1](x, adj)
        else:
            x = self.convs[-1](x, edge_index, edge_weight)
        return F.log_softmax(x, dim=1)


class BaseAttack(object):
    """
    Attack base class.
    """
    __metaclass__ = ABCMeta

    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device

    def generate(self, image, label, **kwargs):
        """
        Overide this function for the main body of attack algorithm.

        Parameters
        ----------
        image :
            original image
        label :
            original label
        kwargs :
            user defined parameters
        """
        return input

    def parse_params(self, **kwargs):
        """
        Parse user defined parameters.
        """
        return True

    def check_type_device(self, image, label):
        """
        Check device, match variable type to device type.

        Parameters
        ----------
        image :
            image
        label :
            label
        """
        if self.device == 'cuda':
            image = image
            label = label
            self.model = self.model
        elif self.device == 'cpu':
            image = image.cpu()
            label = label.cpu()
            self.model = self.model.cpu()
        else:
            raise ValueError('Please input cpu or cuda')
        if type(image).__name__ == 'Tensor':
            image = image.float()
            image = image.float().clone().detach().requires_grad_(True)
        elif type(image).__name__ == 'ndarray':
            image = image.astype('float')
            image = torch.tensor(image, requires_grad=True)
        else:
            raise ValueError('Input values only take numpy arrays or torch tensors')
        if type(label).__name__ == 'Tensor':
            label = label.long()
        elif type(label).__name__ == 'ndarray':
            label = label.astype('long')
            label = torch.tensor(y)
        else:
            raise ValueError('Input labels only take numpy arrays or torch tensors')
        self.image = image
        self.label = label
        return True

    def get_or_predict_lable(self, image):
        output = self.model(image)
        pred = output.argmax(dim=1, keepdim=True)
        return pred


class BaseMeta(BaseAttack):
    """Abstract base class for meta attack. Adversarial Attacks on Graph Neural
    Networks via Meta Learning, ICLR 2019,
    https://openreview.net/pdf?id=Bylnx209YX

    Parameters
    ----------
    model :
        model to attack. Default `None`.
    nnodes : int
        number of nodes in the input graph
    lambda_ : float
        lambda_ is used to weight the two objectives in Eq. (10) in the paper.
    feature_shape : tuple
        shape of the input node features
    attack_structure : bool
        whether to attack graph structure
    attack_features : bool
        whether to attack node features
    undirected : bool
        whether the graph is undirected
    device: str
        'cpu' or 'cuda'

    """

    def __init__(self, model=None, nnodes=None, feature_shape=None, lambda_=0.5, attack_structure=True, attack_features=False, undirected=True, device='cpu'):
        super(BaseMeta, self).__init__(model, nnodes, attack_structure, attack_features, device)
        self.lambda_ = lambda_
        assert attack_features or attack_structure, 'attack_features or attack_structure cannot be both False'
        self.modified_adj = None
        self.modified_features = None
        if attack_structure:
            self.undirected = undirected
            assert nnodes is not None, 'Please give nnodes='
            self.adj_changes = Parameter(torch.FloatTensor(nnodes, nnodes))
            self.adj_changes.data.fill_(0)
        if attack_features:
            assert feature_shape is not None, 'Please give feature_shape='
            self.feature_changes = Parameter(torch.FloatTensor(feature_shape))
            self.feature_changes.data.fill_(0)
        self.with_relu = model.with_relu

    def attack(self, adj, labels, n_perturbations):
        pass

    def get_modified_adj(self, ori_adj):
        adj_changes_square = self.adj_changes - torch.diag(torch.diag(self.adj_changes, 0))
        if self.undirected:
            adj_changes_square = adj_changes_square + torch.transpose(adj_changes_square, 1, 0)
        adj_changes_square = torch.clamp(adj_changes_square, -1, 1)
        modified_adj = adj_changes_square + ori_adj
        return modified_adj

    def get_modified_features(self, ori_features):
        return ori_features + self.feature_changes

    def filter_potential_singletons(self, modified_adj):
        """
        Computes a mask for entries potentially leading to singleton nodes, i.e. one of the two nodes corresponding to
        the entry have degree 1 and there is an edge between the two nodes.
        """
        degrees = modified_adj.sum(0)
        degree_one = degrees == 1
        resh = degree_one.repeat(modified_adj.shape[0], 1).float()
        l_and = resh * modified_adj
        if self.undirected:
            l_and = l_and + l_and.t()
        flat_mask = 1 - l_and
        return flat_mask

    def self_training_label(self, labels, idx_train):
        output = self.surrogate.output
        labels_self_training = output.argmax(1)
        labels_self_training[idx_train] = labels[idx_train]
        return labels_self_training

    def log_likelihood_constraint(self, modified_adj, ori_adj, ll_cutoff):
        """
        Computes a mask for entries that, if the edge corresponding to the entry is added/removed, would lead to the
        log likelihood constraint to be violated.

        Note that different data type (float, double) can effect the final results.
        """
        t_d_min = torch.tensor(2.0)
        if self.undirected:
            t_possible_edges = np.array(np.triu(np.ones((self.nnodes, self.nnodes)), k=1).nonzero()).T
        else:
            t_possible_edges = np.array((np.ones((self.nnodes, self.nnodes)) - np.eye(self.nnodes)).nonzero()).T
        allowed_mask, current_ratio = utils.likelihood_ratio_filter(t_possible_edges, modified_adj, ori_adj, t_d_min, ll_cutoff, undirected=self.undirected)
        return allowed_mask, current_ratio

    def get_adj_score(self, adj_grad, modified_adj, ori_adj, ll_constraint, ll_cutoff):
        adj_meta_grad = adj_grad * (-2 * modified_adj + 1)
        adj_meta_grad = adj_meta_grad - adj_meta_grad.min()
        adj_meta_grad = adj_meta_grad - torch.diag(torch.diag(adj_meta_grad, 0))
        singleton_mask = self.filter_potential_singletons(modified_adj)
        adj_meta_grad = adj_meta_grad * singleton_mask
        if ll_constraint:
            allowed_mask, self.ll_ratio = self.log_likelihood_constraint(modified_adj, ori_adj, ll_cutoff)
            allowed_mask = allowed_mask
            adj_meta_grad = adj_meta_grad * allowed_mask
        return adj_meta_grad

    def get_feature_score(self, feature_grad, modified_features):
        feature_meta_grad = feature_grad * (-2 * modified_features + 1)
        feature_meta_grad -= feature_meta_grad.min()
        return feature_meta_grad


class Metattack(BaseMeta):
    """Meta attack. Adversarial Attacks on Graph Neural Networks
    via Meta Learning, ICLR 2019.

    Examples
    --------

    >>> import numpy as np
    >>> from deeprobust.graph.data import Dataset
    >>> from deeprobust.graph.defense import GCN
    >>> from deeprobust.graph.global_attack import Metattack
    >>> data = Dataset(root='/tmp/', name='cora')
    >>> adj, features, labels = data.adj, data.features, data.labels
    >>> idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
    >>> idx_unlabeled = np.union1d(idx_val, idx_test)
    >>> idx_unlabeled = np.union1d(idx_val, idx_test)
    >>> # Setup Surrogate model
    >>> surrogate = GCN(nfeat=features.shape[1], nclass=labels.max().item()+1,
                    nhid=16, dropout=0, with_relu=False, with_bias=False, device='cpu').to('cpu')
    >>> surrogate.fit(features, adj, labels, idx_train, idx_val, patience=30)
    >>> # Setup Attack Model
    >>> model = Metattack(surrogate, nnodes=adj.shape[0], feature_shape=features.shape,
            attack_structure=True, attack_features=False, device='cpu', lambda_=0).to('cpu')
    >>> # Attack
    >>> model.attack(features, adj, labels, idx_train, idx_unlabeled, n_perturbations=10, ll_constraint=False)
    >>> modified_adj = model.modified_adj

    """

    def __init__(self, model, nnodes, feature_shape=None, attack_structure=True, attack_features=False, undirected=True, device='cpu', with_bias=False, lambda_=0.5, train_iters=100, lr=0.1, momentum=0.9):
        super(Metattack, self).__init__(model, nnodes, feature_shape, lambda_, attack_structure, attack_features, undirected, device)
        self.momentum = momentum
        self.lr = lr
        self.train_iters = train_iters
        self.with_bias = with_bias
        self.weights = []
        self.biases = []
        self.w_velocities = []
        self.b_velocities = []
        self.hidden_sizes = self.surrogate.hidden_sizes
        self.nfeat = self.surrogate.nfeat
        self.nclass = self.surrogate.nclass
        previous_size = self.nfeat
        for ix, nhid in enumerate(self.hidden_sizes):
            weight = Parameter(torch.FloatTensor(previous_size, nhid))
            w_velocity = torch.zeros(weight.shape)
            self.weights.append(weight)
            self.w_velocities.append(w_velocity)
            if self.with_bias:
                bias = Parameter(torch.FloatTensor(nhid))
                b_velocity = torch.zeros(bias.shape)
                self.biases.append(bias)
                self.b_velocities.append(b_velocity)
            previous_size = nhid
        output_weight = Parameter(torch.FloatTensor(previous_size, self.nclass))
        output_w_velocity = torch.zeros(output_weight.shape)
        self.weights.append(output_weight)
        self.w_velocities.append(output_w_velocity)
        if self.with_bias:
            output_bias = Parameter(torch.FloatTensor(self.nclass))
            output_b_velocity = torch.zeros(output_bias.shape)
            self.biases.append(output_bias)
            self.b_velocities.append(output_b_velocity)
        self._initialize()

    def _initialize(self):
        for w, v in zip(self.weights, self.w_velocities):
            stdv = 1.0 / math.sqrt(w.size(1))
            w.data.uniform_(-stdv, stdv)
            v.data.fill_(0)
        if self.with_bias:
            for b, v in zip(self.biases, self.b_velocities):
                stdv = 1.0 / math.sqrt(w.size(1))
                b.data.uniform_(-stdv, stdv)
                v.data.fill_(0)

    def inner_train(self, features, adj_norm, idx_train, idx_unlabeled, labels):
        self._initialize()
        for ix in range(len(self.hidden_sizes) + 1):
            self.weights[ix] = self.weights[ix].detach()
            self.weights[ix].requires_grad = True
            self.w_velocities[ix] = self.w_velocities[ix].detach()
            self.w_velocities[ix].requires_grad = True
            if self.with_bias:
                self.biases[ix] = self.biases[ix].detach()
                self.biases[ix].requires_grad = True
                self.b_velocities[ix] = self.b_velocities[ix].detach()
                self.b_velocities[ix].requires_grad = True
        for j in range(self.train_iters):
            hidden = features
            for ix, w in enumerate(self.weights):
                b = self.biases[ix] if self.with_bias else 0
                if self.sparse_features:
                    hidden = adj_norm @ torch.spmm(hidden, w) + b
                else:
                    hidden = adj_norm @ hidden @ w + b
                if self.with_relu and ix != len(self.weights) - 1:
                    hidden = F.relu(hidden)
            output = F.log_softmax(hidden, dim=1)
            loss_labeled = F.nll_loss(output[idx_train], labels[idx_train])
            weight_grads = torch.autograd.grad(loss_labeled, self.weights, create_graph=True)
            self.w_velocities = [(self.momentum * v + g) for v, g in zip(self.w_velocities, weight_grads)]
            if self.with_bias:
                bias_grads = torch.autograd.grad(loss_labeled, self.biases, create_graph=True)
                self.b_velocities = [(self.momentum * v + g) for v, g in zip(self.b_velocities, bias_grads)]
            self.weights = [(w - self.lr * v) for w, v in zip(self.weights, self.w_velocities)]
            if self.with_bias:
                self.biases = [(b - self.lr * v) for b, v in zip(self.biases, self.b_velocities)]

    def get_meta_grad(self, features, adj_norm, idx_train, idx_unlabeled, labels, labels_self_training):
        hidden = features
        for ix, w in enumerate(self.weights):
            b = self.biases[ix] if self.with_bias else 0
            if self.sparse_features:
                hidden = adj_norm @ torch.spmm(hidden, w) + b
            else:
                hidden = adj_norm @ hidden @ w + b
            if self.with_relu and ix != len(self.weights) - 1:
                hidden = F.relu(hidden)
        output = F.log_softmax(hidden, dim=1)
        loss_labeled = F.nll_loss(output[idx_train], labels[idx_train])
        loss_unlabeled = F.nll_loss(output[idx_unlabeled], labels_self_training[idx_unlabeled])
        loss_test_val = F.nll_loss(output[idx_unlabeled], labels[idx_unlabeled])
        if self.lambda_ == 1:
            attack_loss = loss_labeled
        elif self.lambda_ == 0:
            attack_loss = loss_unlabeled
        else:
            attack_loss = self.lambda_ * loss_labeled + (1 - self.lambda_) * loss_unlabeled
        None
        None
        None
        adj_grad, feature_grad = None, None
        if self.attack_structure:
            adj_grad = torch.autograd.grad(attack_loss, self.adj_changes, retain_graph=True)[0]
        if self.attack_features:
            feature_grad = torch.autograd.grad(attack_loss, self.feature_changes, retain_graph=True)[0]
        return adj_grad, feature_grad

    def attack(self, ori_features, ori_adj, labels, idx_train, idx_unlabeled, n_perturbations, ll_constraint=True, ll_cutoff=0.004):
        """Generate n_perturbations on the input graph.

        Parameters
        ----------
        ori_features :
            Original (unperturbed) node feature matrix
        ori_adj :
            Original (unperturbed) adjacency matrix
        labels :
            node labels
        idx_train :
            node training indices
        idx_unlabeled:
            unlabeled nodes indices
        n_perturbations : int
            Number of perturbations on the input graph. Perturbations could
            be edge removals/additions or feature removals/additions.
        ll_constraint: bool
            whether to exert the likelihood ratio test constraint
        ll_cutoff : float
            The critical value for the likelihood ratio test of the power law distributions.
            See the Chi square distribution with one degree of freedom. Default value 0.004
            corresponds to a p-value of roughly 0.95. It would be ignored if `ll_constraint`
            is False.

        """
        self.sparse_features = sp.issparse(ori_features)
        ori_adj, ori_features, labels = utils.to_tensor(ori_adj, ori_features, labels, device=self.device)
        labels_self_training = self.self_training_label(labels, idx_train)
        modified_adj = ori_adj
        modified_features = ori_features
        for i in tqdm(range(n_perturbations), desc='Perturbing graph'):
            if self.attack_structure:
                modified_adj = self.get_modified_adj(ori_adj)
            if self.attack_features:
                modified_features = ori_features + self.feature_changes
            adj_norm = utils.normalize_adj_tensor(modified_adj)
            self.inner_train(modified_features, adj_norm, idx_train, idx_unlabeled, labels)
            adj_grad, feature_grad = self.get_meta_grad(modified_features, adj_norm, idx_train, idx_unlabeled, labels, labels_self_training)
            adj_meta_score = torch.tensor(0.0)
            feature_meta_score = torch.tensor(0.0)
            if self.attack_structure:
                adj_meta_score = self.get_adj_score(adj_grad, modified_adj, ori_adj, ll_constraint, ll_cutoff)
            if self.attack_features:
                feature_meta_score = self.get_feature_score(feature_grad, modified_features)
            if adj_meta_score.max() >= feature_meta_score.max():
                adj_meta_argmax = torch.argmax(adj_meta_score)
                row_idx, col_idx = utils.unravel_index(adj_meta_argmax, ori_adj.shape)
                self.adj_changes.data[row_idx][col_idx] += -2 * modified_adj[row_idx][col_idx] + 1
                if self.undirected:
                    self.adj_changes.data[col_idx][row_idx] += -2 * modified_adj[row_idx][col_idx] + 1
            else:
                feature_meta_argmax = torch.argmax(feature_meta_score)
                row_idx, col_idx = utils.unravel_index(feature_meta_argmax, ori_features.shape)
                self.feature_changes.data[row_idx][col_idx] += -2 * modified_features[row_idx][col_idx] + 1
        if self.attack_structure:
            self.modified_adj = self.get_modified_adj(ori_adj).detach()
        if self.attack_features:
            self.modified_features = self.get_modified_features(ori_features).detach()


class MetaApprox(BaseMeta):
    """Approximated version of Meta Attack. Adversarial Attacks on
    Graph Neural Networks via Meta Learning, ICLR 2019.

    Examples
    --------

    >>> import numpy as np
    >>> from deeprobust.graph.data import Dataset
    >>> from deeprobust.graph.defense import GCN
    >>> from deeprobust.graph.global_attack import MetaApprox
    >>> from deeprobust.graph.utils import preprocess
    >>> data = Dataset(root='/tmp/', name='cora')
    >>> adj, features, labels = data.adj, data.features, data.labels
    >>> adj, features, labels = preprocess(adj, features, labels, preprocess_adj=False) # conver to tensor
    >>> idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
    >>> idx_unlabeled = np.union1d(idx_val, idx_test)
    >>> # Setup Surrogate model
    >>> surrogate = GCN(nfeat=features.shape[1], nclass=labels.max().item()+1,
                    nhid=16, dropout=0, with_relu=False, with_bias=False, device='cpu').to('cpu')
    >>> surrogate.fit(features, adj, labels, idx_train, idx_val, patience=30)
    >>> # Setup Attack Model
    >>> model = MetaApprox(surrogate, nnodes=adj.shape[0], feature_shape=features.shape,
            attack_structure=True, attack_features=False, device='cpu', lambda_=0).to('cpu')
    >>> # Attack
    >>> model.attack(features, adj, labels, idx_train, idx_unlabeled, n_perturbations=10, ll_constraint=True)
    >>> modified_adj = model.modified_adj

    """

    def __init__(self, model, nnodes, feature_shape=None, attack_structure=True, attack_features=False, undirected=True, device='cpu', with_bias=False, lambda_=0.5, train_iters=100, lr=0.01):
        super(MetaApprox, self).__init__(model, nnodes, feature_shape, lambda_, attack_structure, attack_features, undirected, device)
        self.lr = lr
        self.train_iters = train_iters
        self.adj_meta_grad = None
        self.features_meta_grad = None
        if self.attack_structure:
            self.adj_grad_sum = torch.zeros(nnodes, nnodes)
        if self.attack_features:
            self.feature_grad_sum = torch.zeros(feature_shape)
        self.with_bias = with_bias
        self.weights = []
        self.biases = []
        previous_size = self.nfeat
        for ix, nhid in enumerate(self.hidden_sizes):
            weight = Parameter(torch.FloatTensor(previous_size, nhid))
            bias = Parameter(torch.FloatTensor(nhid))
            previous_size = nhid
            self.weights.append(weight)
            self.biases.append(bias)
        output_weight = Parameter(torch.FloatTensor(previous_size, self.nclass))
        output_bias = Parameter(torch.FloatTensor(self.nclass))
        self.weights.append(output_weight)
        self.biases.append(output_bias)
        self.optimizer = optim.Adam(self.weights + self.biases, lr=lr)
        self._initialize()

    def _initialize(self):
        for w, b in zip(self.weights, self.biases):
            stdv = 1.0 / math.sqrt(w.size(1))
            w.data.uniform_(-stdv, stdv)
            b.data.uniform_(-stdv, stdv)
        self.optimizer = optim.Adam(self.weights + self.biases, lr=self.lr)

    def inner_train(self, features, modified_adj, idx_train, idx_unlabeled, labels, labels_self_training):
        adj_norm = utils.normalize_adj_tensor(modified_adj)
        for j in range(self.train_iters):
            hidden = features
            for ix, w in enumerate(self.weights):
                b = self.biases[ix] if self.with_bias else 0
                if self.sparse_features:
                    hidden = adj_norm @ torch.spmm(hidden, w) + b
                else:
                    hidden = adj_norm @ hidden @ w + b
                if self.with_relu:
                    hidden = F.relu(hidden)
            output = F.log_softmax(hidden, dim=1)
            loss_labeled = F.nll_loss(output[idx_train], labels[idx_train])
            loss_unlabeled = F.nll_loss(output[idx_unlabeled], labels_self_training[idx_unlabeled])
            if self.lambda_ == 1:
                attack_loss = loss_labeled
            elif self.lambda_ == 0:
                attack_loss = loss_unlabeled
            else:
                attack_loss = self.lambda_ * loss_labeled + (1 - self.lambda_) * loss_unlabeled
            self.optimizer.zero_grad()
            loss_labeled.backward(retain_graph=True)
            if self.attack_structure:
                self.adj_changes.grad.zero_()
                self.adj_grad_sum += torch.autograd.grad(attack_loss, self.adj_changes, retain_graph=True)[0]
            if self.attack_features:
                self.feature_changes.grad.zero_()
                self.feature_grad_sum += torch.autograd.grad(attack_loss, self.feature_changes, retain_graph=True)[0]
            self.optimizer.step()
        loss_test_val = F.nll_loss(output[idx_unlabeled], labels[idx_unlabeled])
        None
        None

    def attack(self, ori_features, ori_adj, labels, idx_train, idx_unlabeled, n_perturbations, ll_constraint=True, ll_cutoff=0.004):
        """Generate n_perturbations on the input graph.

        Parameters
        ----------
        ori_features :
            Original (unperturbed) node feature matrix
        ori_adj :
            Original (unperturbed) adjacency matrix
        labels :
            node labels
        idx_train :
            node training indices
        idx_unlabeled:
            unlabeled nodes indices
        n_perturbations : int
            Number of perturbations on the input graph. Perturbations could
            be edge removals/additions or feature removals/additions.
        ll_constraint: bool
            whether to exert the likelihood ratio test constraint
        ll_cutoff : float
            The critical value for the likelihood ratio test of the power law distributions.
            See the Chi square distribution with one degree of freedom. Default value 0.004
            corresponds to a p-value of roughly 0.95. It would be ignored if `ll_constraint`
            is False.

        """
        ori_adj, ori_features, labels = utils.to_tensor(ori_adj, ori_features, labels, device=self.device)
        labels_self_training = self.self_training_label(labels, idx_train)
        self.sparse_features = sp.issparse(ori_features)
        modified_adj = ori_adj
        modified_features = ori_features
        for i in tqdm(range(n_perturbations), desc='Perturbing graph'):
            self._initialize()
            if self.attack_structure:
                modified_adj = self.get_modified_adj(ori_adj)
                self.adj_grad_sum.data.fill_(0)
            if self.attack_features:
                modified_features = ori_features + self.feature_changes
                self.feature_grad_sum.data.fill_(0)
            self.inner_train(modified_features, modified_adj, idx_train, idx_unlabeled, labels, labels_self_training)
            adj_meta_score = torch.tensor(0.0)
            feature_meta_score = torch.tensor(0.0)
            if self.attack_structure:
                adj_meta_score = self.get_adj_score(self.adj_grad_sum, modified_adj, ori_adj, ll_constraint, ll_cutoff)
            if self.attack_features:
                feature_meta_score = self.get_feature_score(self.feature_grad_sum, modified_features)
            if adj_meta_score.max() >= feature_meta_score.max():
                adj_meta_argmax = torch.argmax(adj_meta_score)
                row_idx, col_idx = utils.unravel_index(adj_meta_argmax, ori_adj.shape)
                self.adj_changes.data[row_idx][col_idx] += -2 * modified_adj[row_idx][col_idx] + 1
                if self.undirected:
                    self.adj_changes.data[col_idx][row_idx] += -2 * modified_adj[row_idx][col_idx] + 1
            else:
                feature_meta_argmax = torch.argmax(feature_meta_score)
                row_idx, col_idx = utils.unravel_index(feature_meta_argmax, ori_features.shape)
                self.feature_changes.data[row_idx][col_idx] += -2 * modified_features[row_idx][col_idx] + 1
        if self.attack_structure:
            self.modified_adj = self.get_modified_adj(ori_adj).detach()
        if self.attack_features:
            self.modified_features = self.get_modified_features(ori_features).detach()


def edges_to_sparse(edges, num_nodes, weights=None):
    if weights is None:
        weights = np.ones(edges.shape[0])
    return sp.coo_matrix((weights, (edges[:, 0], edges[:, 1])), shape=(num_nodes, num_nodes)).tocsr()


class NodeEmbeddingAttack(BaseAttack):
    """Node embedding attack. Adversarial Attacks on Node Embeddings via Graph
    Poisoning. Aleksandar Bojchevski and Stephan Günnemann, ICML 2019
    http://proceedings.mlr.press/v97/bojchevski19a.html

    Examples
    -----
    >>> from deeprobust.graph.data import Dataset
    >>> from deeprobust.graph.global_attack import NodeEmbeddingAttack
    >>> data = Dataset(root='/tmp/', name='cora_ml', seed=15)
    >>> adj, features, labels = data.adj, data.features, data.labels
    >>> model = NodeEmbeddingAttack()
    >>> model.attack(adj, attack_type="remove")
    >>> modified_adj = model.modified_adj
    >>> model.attack(adj, attack_type="remove", min_span_tree=True)
    >>> modified_adj = model.modified_adj
    >>> model.attack(adj, attack_type="add", n_candidates=10000)
    >>> modified_adj = model.modified_adj
    >>> model.attack(adj, attack_type="add_by_remove", n_candidates=10000)
    >>> modified_adj = model.modified_adj
    """

    def __init__(self):
        pass

    def attack(self, adj, n_perturbations=1000, dim=32, window_size=5, attack_type='remove', min_span_tree=False, n_candidates=None, seed=None, **kwargs):
        """Selects the top (n_perturbations) number of flips using our perturbation attack.

        :param adj: sp.spmatrix
            The graph represented as a sparse scipy matrix
        :param n_perturbations: int
            Number of flips to select
        :param dim: int
            Dimensionality of the embeddings.
        :param window_size: int
            Co-occurence window size.
        :param attack_type: str
            can be chosed from ["remove", "add", "add_by_remove"]
        :param min_span_tree: bool
            Whether to disallow edges that lie on the minimum spanning tree;
            only valid when `attack_type` is "remove"
        :param n_candidates: int
            Number of candiates for addition; only valid when `attack_type` is "add" or "add_by_remove";
        :param seed: int
            Random seed
        """
        assert attack_type in ['remove', 'add', 'add_by_remove'], 'attack_type can only be `remove` or `add`'
        if attack_type == 'remove':
            if min_span_tree:
                candidates = self.generate_candidates_removal_minimum_spanning_tree(adj)
            else:
                candidates = self.generate_candidates_removal(adj, seed)
        elif attack_type == 'add' or attack_type == 'add_by_remove':
            assert n_candidates, 'please specify the value of `n_candidates`, ' + 'i.e. how many candiate you want to genereate for addition'
            candidates = self.generate_candidates_addition(adj, n_candidates, seed)
        n_nodes = adj.shape[0]
        if attack_type == 'add_by_remove':
            candidates_add = candidates
            adj_add = self.flip_candidates(adj, candidates_add)
            vals_org_add, vecs_org_add = spl.eigh(adj_add.toarray(), np.diag(adj_add.sum(1).A1))
            flip_indicator = 1 - 2 * adj_add[candidates[:, 0], candidates[:, 1]].A1
            loss_est = estimate_loss_with_delta_eigenvals(candidates_add, flip_indicator, vals_org_add, vecs_org_add, n_nodes, dim, window_size)
            loss_argsort = loss_est.argsort()
            top_flips = candidates_add[loss_argsort[:n_perturbations]]
        else:
            delta_w = 1 - 2 * adj[candidates[:, 0], candidates[:, 1]].A1
            deg_matrix = np.diag(adj.sum(1).A1)
            vals_org, vecs_org = spl.eigh(adj.toarray(), deg_matrix)
            loss_for_candidates = estimate_loss_with_delta_eigenvals(candidates, delta_w, vals_org, vecs_org, n_nodes, dim, window_size)
            top_flips = candidates[loss_for_candidates.argsort()[-n_perturbations:]]
        assert len(top_flips) == n_perturbations
        modified_adj = self.flip_candidates(adj, top_flips)
        self.check_adj(modified_adj)
        self.modified_adj = modified_adj

    def generate_candidates_removal(self, adj, seed=None):
        """Generates candidate edge flips for removal (edge -> non-edge),
        disallowing one random edge per node to prevent singleton nodes.

        :param adj: sp.csr_matrix, shape [n_nodes, n_nodes]
            Adjacency matrix of the graph
        :param seed: int
            Random seed
        :return: np.ndarray, shape [?, 2]
            Candidate set of edge flips
        """
        n_nodes = adj.shape[0]
        if seed is not None:
            np.random.seed(seed)
        deg = np.where(adj.sum(1).A1 == 1)[0]
        hiddeen = np.column_stack((np.arange(n_nodes), np.fromiter(map(np.random.choice, adj.tolil().rows), dtype=np.int32)))
        adj_hidden = edges_to_sparse(hiddeen, adj.shape[0])
        adj_hidden = adj_hidden.maximum(adj_hidden.T)
        adj_keep = adj - adj_hidden
        candidates = np.column_stack(sp.triu(adj_keep).nonzero())
        candidates = candidates[np.logical_not(np.in1d(candidates[:, 0], deg) | np.in1d(candidates[:, 1], deg))]
        return candidates

    def generate_candidates_removal_minimum_spanning_tree(self, adj):
        """Generates candidate edge flips for removal (edge -> non-edge),
         disallowing edges that lie on the minimum spanning tree.
        adj: sp.csr_matrix, shape [n_nodes, n_nodes]
            Adjacency matrix of the graph
        :return: np.ndarray, shape [?, 2]
            Candidate set of edge flips
        """
        mst = sp.csgraph.minimum_spanning_tree(adj)
        mst = mst.maximum(mst.T)
        adj_sample = adj - mst
        candidates = np.column_stack(sp.triu(adj_sample, 1).nonzero())
        return candidates

    def generate_candidates_addition(self, adj, n_candidates, seed=None):
        """Generates candidate edge flips for addition (non-edge -> edge).

        :param adj: sp.csr_matrix, shape [n_nodes, n_nodes]
            Adjacency matrix of the graph
        :param n_candidates: int
            Number of candidates to generate.
        :param seed: int
            Random seed
        :return: np.ndarray, shape [?, 2]
            Candidate set of edge flips
        """
        if seed is not None:
            np.random.seed(seed)
        num_nodes = adj.shape[0]
        candidates = np.random.randint(0, num_nodes, [n_candidates * 5, 2])
        candidates = candidates[candidates[:, 0] < candidates[:, 1]]
        candidates = candidates[adj[candidates[:, 0], candidates[:, 1]].A1 == 0]
        candidates = np.array(list(set(map(tuple, candidates))))
        candidates = candidates[:n_candidates]
        assert len(candidates) == n_candidates
        return candidates

    def flip_candidates(self, adj, candidates):
        """Flip the edges in the candidate set to non-edges and vise-versa.

        :param adj: sp.csr_matrix, shape [n_nodes, n_nodes]
            Adjacency matrix of the graph
        :param candidates: np.ndarray, shape [?, 2]
            Candidate set of edge flips
        :return: sp.csr_matrix, shape [n_nodes, n_nodes]
            Adjacency matrix of the graph with the flipped edges/non-edges.
        """
        adj_flipped = adj.copy().tolil()
        adj_flipped[candidates[:, 0], candidates[:, 1]] = 1 - adj[candidates[:, 0], candidates[:, 1]]
        adj_flipped[candidates[:, 1], candidates[:, 0]] = 1 - adj[candidates[:, 1], candidates[:, 0]]
        adj_flipped = adj_flipped.tocsr()
        adj_flipped.eliminate_zeros()
        return adj_flipped


def construct_line_graph(adj):
    """Construct a line graph from an undirected original graph.

    Parameters
    ----------
    adj : sp.spmatrix [n_samples ,n_samples]
        Symmetric binary adjacency matrix.
    Returns
    -------
    L : sp.spmatrix, shape [A.nnz/2, A.nnz/2]
        Symmetric binary adjacency matrix of the line graph.
    """
    N = adj.shape[0]
    edges = np.column_stack(sp.triu(adj, 1).nonzero())
    e1, e2 = edges[:, 0], edges[:, 1]
    I = sp.eye(N).tocsr()
    E1 = I[e1]
    E2 = I[e2]
    L = E1.dot(E1.T) + E1.dot(E2.T) + E2.dot(E1.T) + E2.dot(E2.T)
    return L - 2 * sp.eye(L.shape[0])


class OtherNodeEmbeddingAttack(NodeEmbeddingAttack):
    """ Baseline methods from the paper Adversarial Attacks on Node Embeddings
    via Graph Poisoning. Aleksandar Bojchevski and Stephan Günnemann, ICML 2019.
    http://proceedings.mlr.press/v97/bojchevski19a.html

    Examples
    -----
    >>> from deeprobust.graph.data import Dataset
    >>> from deeprobust.graph.global_attack import OtherNodeEmbeddingAttack
    >>> data = Dataset(root='/tmp/', name='cora_ml', seed=15)
    >>> adj, features, labels = data.adj, data.features, data.labels
    >>> model = OtherNodeEmbeddingAttack(type='degree')
    >>> model.attack(adj, attack_type="remove")
    >>> modified_adj = model.modified_adj
    >>> #
    >>> model = OtherNodeEmbeddingAttack(type='eigencentrality')
    >>> model.attack(adj, attack_type="remove")
    >>> modified_adj = model.modified_adj
    >>> #
    >>> model = OtherNodeEmbeddingAttack(type='random')
    >>> model.attack(adj, attack_type="add", n_candidates=10000)
    >>> modified_adj = model.modified_adj
    """

    def __init__(self, type):
        assert type in ['degree', 'eigencentrality', 'random']
        self.type = type

    def attack(self, adj, n_perturbations=1000, attack_type='remove', min_span_tree=False, n_candidates=None, seed=None, **kwargs):
        """Selects the top (n_perturbations) number of flips using our perturbation attack.

        :param adj: sp.spmatrix
            The graph represented as a sparse scipy matrix
        :param n_perturbations: int
            Number of flips to select
        :param dim: int
            Dimensionality of the embeddings.
        :param attack_type: str
            can be chosed from ["remove", "add"]
        :param min_span_tree: bool
            Whether to disallow edges that lie on the minimum spanning tree;
            only valid when `attack_type` is "remove"
        :param n_candidates: int
            Number of candiates for addition; only valid when `attack_type` is "add";
        :param seed: int
            Random seed;
        :return: np.ndarray, shape [?, 2]
            The top edge flips from the candidate set
        """
        assert attack_type in ['remove', 'add'], 'attack_type can only be `remove` or `add`'
        if attack_type == 'remove':
            if min_span_tree:
                candidates = self.generate_candidates_removal_minimum_spanning_tree(adj)
            else:
                candidates = self.generate_candidates_removal(adj, seed)
        elif attack_type == 'add':
            assert n_candidates, 'please specify the value of `n_candidates`, ' + 'i.e. how many candiate you want to genereate for addition'
            candidates = self.generate_candidates_addition(adj, n_candidates, seed)
        else:
            raise NotImplementedError
        if self.type == 'random':
            top_flips = self.random_top_flips(candidates, n_perturbations, seed)
        elif self.type == 'eigencentrality':
            top_flips = self.eigencentrality_top_flips(adj, candidates, n_perturbations)
        elif self.type == 'degree':
            top_flips = self.degree_top_flips(adj, candidates, n_perturbations, complement=False)
        else:
            raise NotImplementedError
        assert len(top_flips) == n_perturbations
        modified_adj = self.flip_candidates(adj, top_flips)
        self.check_adj(modified_adj)
        self.modified_adj = modified_adj

    def random_top_flips(self, candidates, n_perturbations, seed=None):
        """Selects (n_perturbations) number of flips at random.

        :param candidates: np.ndarray, shape [?, 2]
            Candidate set of edge flips
        :param n_perturbations: int
            Number of flips to select
        :param seed: int
            Random seed
        :return: np.ndarray, shape [?, 2]
            The top edge flips from the candidate set
        """
        if seed is not None:
            np.random.seed(seed)
        return candidates[np.random.permutation(len(candidates))[:n_perturbations]]

    def eigencentrality_top_flips(self, adj, candidates, n_perturbations):
        """Selects the top (n_perturbations) number of flips using eigencentrality score of the edges.
        Applicable only when removing edges.

        :param adj: sp.spmatrix
            The graph represented as a sparse scipy matrix
        :param candidates: np.ndarray, shape [?, 2]
            Candidate set of edge flips
        :param n_perturbations: int
            Number of flips to select
        :return: np.ndarray, shape [?, 2]
            The top edge flips from the candidate set
        """
        edges = np.column_stack(sp.triu(adj, 1).nonzero())
        line_graph = construct_line_graph(adj)
        eigcentrality_scores = nx.eigenvector_centrality_numpy(nx.Graph(line_graph))
        eigcentrality_scores = {tuple(edges[k]): eigcentrality_scores[k] for k, v in eigcentrality_scores.items()}
        eigcentrality_scores = np.array([eigcentrality_scores[tuple(cnd)] for cnd in candidates])
        scores_argsrt = eigcentrality_scores.argsort()
        return candidates[scores_argsrt[-n_perturbations:]]

    def degree_top_flips(self, adj, candidates, n_perturbations, complement):
        """Selects the top (n_perturbations) number of flips using degree centrality score of the edges.

        :param adj: sp.spmatrix
            The graph represented as a sparse scipy matrix
        :param candidates: np.ndarray, shape [?, 2]
            Candidate set of edge flips
        :param n_perturbations: int
            Number of flips to select
        :param complement: bool
            Whether to look at the complement graph
        :return: np.ndarray, shape [?, 2]
            The top edge flips from the candidate set
        """
        if complement:
            adj = sp.csr_matrix(1 - adj.toarray())
        deg = adj.sum(1).A1
        deg_argsort = (deg[candidates[:, 0]] + deg[candidates[:, 1]]).argsort()
        return candidates[deg_argsort[-n_perturbations:]]


class PGDAttack(BaseAttack):
    """PGD attack for graph data.

    Parameters
    ----------
    model :
        model to attack. Default `None`.
    nnodes : int
        number of nodes in the input graph
    loss_type: str
        attack loss type, chosen from ['CE', 'CW']
    feature_shape : tuple
        shape of the input node features
    attack_structure : bool
        whether to attack graph structure
    attack_features : bool
        whether to attack node features
    device: str
        'cpu' or 'cuda'

    Examples
    --------

    >>> from deeprobust.graph.data import Dataset
    >>> from deeprobust.graph.defense import GCN
    >>> from deeprobust.graph.global_attack import PGDAttack
    >>> from deeprobust.graph.utils import preprocess
    >>> data = Dataset(root='/tmp/', name='cora')
    >>> adj, features, labels = data.adj, data.features, data.labels
    >>> adj, features, labels = preprocess(adj, features, labels, preprocess_adj=False) # conver to tensor
    >>> idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
    >>> # Setup Victim Model
    >>> victim_model = GCN(nfeat=features.shape[1], nclass=labels.max().item()+1,
                        nhid=16, dropout=0.5, weight_decay=5e-4, device='cpu').to('cpu')
    >>> victim_model.fit(features, adj, labels, idx_train)
    >>> # Setup Attack Model
    >>> model = PGDAttack(model=victim_model, nnodes=adj.shape[0], loss_type='CE', device='cpu').to('cpu')
    >>> model.attack(features, adj, labels, idx_train, n_perturbations=10)
    >>> modified_adj = model.modified_adj

    """

    def __init__(self, model=None, nnodes=None, loss_type='CE', feature_shape=None, attack_structure=True, attack_features=False, device='cpu'):
        super(PGDAttack, self).__init__(model, nnodes, attack_structure, attack_features, device)
        assert attack_features or attack_structure, 'attack_features or attack_structure cannot be both False'
        self.loss_type = loss_type
        self.modified_adj = None
        self.modified_features = None
        if attack_structure:
            assert nnodes is not None, 'Please give nnodes='
            self.adj_changes = Parameter(torch.FloatTensor(int(nnodes * (nnodes - 1) / 2)))
            self.adj_changes.data.fill_(0)
        if attack_features:
            assert True, 'Topology Attack does not support attack feature'
        self.complementary = None

    def attack(self, ori_features, ori_adj, labels, idx_train, n_perturbations, epochs=200, **kwargs):
        """Generate perturbations on the input graph.

        Parameters
        ----------
        ori_features :
            Original (unperturbed) node feature matrix
        ori_adj :
            Original (unperturbed) adjacency matrix
        labels :
            node labels
        idx_train :
            node training indices
        n_perturbations : int
            Number of perturbations on the input graph. Perturbations could
            be edge removals/additions or feature removals/additions.
        epochs:
            number of training epochs

        """
        victim_model = self.surrogate
        self.sparse_features = sp.issparse(ori_features)
        ori_adj, ori_features, labels = utils.to_tensor(ori_adj, ori_features, labels, device=self.device)
        victim_model.eval()
        for t in tqdm(range(epochs)):
            modified_adj = self.get_modified_adj(ori_adj)
            adj_norm = utils.normalize_adj_tensor(modified_adj)
            output = victim_model(ori_features, adj_norm)
            loss = self._loss(output[idx_train], labels[idx_train])
            adj_grad = torch.autograd.grad(loss, self.adj_changes)[0]
            if self.loss_type == 'CE':
                lr = 200 / np.sqrt(t + 1)
                self.adj_changes.data.add_(lr * adj_grad)
            if self.loss_type == 'CW':
                lr = 0.1 / np.sqrt(t + 1)
                self.adj_changes.data.add_(lr * adj_grad)
            self.projection(n_perturbations)
        self.random_sample(ori_adj, ori_features, labels, idx_train, n_perturbations)
        self.modified_adj = self.get_modified_adj(ori_adj).detach()
        self.check_adj_tensor(self.modified_adj)

    def random_sample(self, ori_adj, ori_features, labels, idx_train, n_perturbations):
        K = 20
        best_loss = -1000
        victim_model = self.surrogate
        victim_model.eval()
        with torch.no_grad():
            s = self.adj_changes.cpu().detach().numpy()
            for i in range(K):
                sampled = np.random.binomial(1, s)
                if sampled.sum() > n_perturbations:
                    continue
                self.adj_changes.data.copy_(torch.tensor(sampled))
                modified_adj = self.get_modified_adj(ori_adj)
                adj_norm = utils.normalize_adj_tensor(modified_adj)
                output = victim_model(ori_features, adj_norm)
                loss = self._loss(output[idx_train], labels[idx_train])
                if best_loss < loss:
                    best_loss = loss
                    best_s = sampled
            self.adj_changes.data.copy_(torch.tensor(best_s))

    def _loss(self, output, labels):
        if self.loss_type == 'CE':
            loss = F.nll_loss(output, labels)
        if self.loss_type == 'CW':
            onehot = utils.tensor2onehot(labels)
            best_second_class = (output - 1000 * onehot).argmax(1)
            margin = output[np.arange(len(output)), labels] - output[np.arange(len(output)), best_second_class]
            k = 0
            loss = -torch.clamp(margin, min=k).mean()
        return loss

    def projection(self, n_perturbations):
        if torch.clamp(self.adj_changes, 0, 1).sum() > n_perturbations:
            left = (self.adj_changes - 1).min()
            right = self.adj_changes.max()
            miu = self.bisection(left, right, n_perturbations, epsilon=1e-05)
            self.adj_changes.data.copy_(torch.clamp(self.adj_changes.data - miu, min=0, max=1))
        else:
            self.adj_changes.data.copy_(torch.clamp(self.adj_changes.data, min=0, max=1))

    def get_modified_adj(self, ori_adj):
        if self.complementary is None:
            self.complementary = torch.ones_like(ori_adj) - torch.eye(self.nnodes) - ori_adj - ori_adj
        m = torch.zeros((self.nnodes, self.nnodes))
        tril_indices = torch.tril_indices(row=self.nnodes, col=self.nnodes, offset=-1)
        m[tril_indices[0], tril_indices[1]] = self.adj_changes
        m = m + m.t()
        modified_adj = self.complementary * m + ori_adj
        return modified_adj

    def bisection(self, a, b, n_perturbations, epsilon):

        def func(x):
            return torch.clamp(self.adj_changes - x, 0, 1).sum() - n_perturbations
        miu = a
        while b - a >= epsilon:
            miu = (a + b) / 2
            if func(miu) == 0.0:
                break
            if func(miu) * func(a) < 0:
                b = miu
            else:
                a = miu
        return miu


class MinMax(PGDAttack):
    """MinMax attack for graph data.

    Parameters
    ----------
    model :
        model to attack. Default `None`.
    nnodes : int
        number of nodes in the input graph
    loss_type: str
        attack loss type, chosen from ['CE', 'CW']
    feature_shape : tuple
        shape of the input node features
    attack_structure : bool
        whether to attack graph structure
    attack_features : bool
        whether to attack node features
    device: str
        'cpu' or 'cuda'

    Examples
    --------

    >>> from deeprobust.graph.data import Dataset
    >>> from deeprobust.graph.defense import GCN
    >>> from deeprobust.graph.global_attack import MinMax
    >>> from deeprobust.graph.utils import preprocess
    >>> data = Dataset(root='/tmp/', name='cora')
    >>> adj, features, labels = data.adj, data.features, data.labels
    >>> adj, features, labels = preprocess(adj, features, labels, preprocess_adj=False) # conver to tensor
    >>> idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
    >>> # Setup Victim Model
    >>> victim_model = GCN(nfeat=features.shape[1], nclass=labels.max().item()+1,
                        nhid=16, dropout=0.5, weight_decay=5e-4, device='cpu').to('cpu')
    >>> victim_model.fit(features, adj, labels, idx_train)
    >>> # Setup Attack Model
    >>> model = MinMax(model=victim_model, nnodes=adj.shape[0], loss_type='CE', device='cpu').to('cpu')
    >>> model.attack(features, adj, labels, idx_train, n_perturbations=10)
    >>> modified_adj = model.modified_adj

    """

    def __init__(self, model=None, nnodes=None, loss_type='CE', feature_shape=None, attack_structure=True, attack_features=False, device='cpu'):
        super(MinMax, self).__init__(model, nnodes, loss_type, feature_shape, attack_structure, attack_features, device=device)

    def attack(self, ori_features, ori_adj, labels, idx_train, n_perturbations, **kwargs):
        """Generate perturbations on the input graph.

        Parameters
        ----------
        ori_features :
            Original (unperturbed) node feature matrix
        ori_adj :
            Original (unperturbed) adjacency matrix
        labels :
            node labels
        idx_train :
            node training indices
        n_perturbations : int
            Number of perturbations on the input graph. Perturbations could
            be edge removals/additions or feature removals/additions.
        epochs:
            number of training epochs

        """
        victim_model = self.surrogate
        self.sparse_features = sp.issparse(ori_features)
        ori_adj, ori_features, labels = utils.to_tensor(ori_adj, ori_features, labels, device=self.device)
        optimizer = optim.Adam(victim_model.parameters(), lr=0.01)
        epochs = 200
        victim_model.eval()
        for t in tqdm(range(epochs)):
            victim_model.train()
            modified_adj = self.get_modified_adj(ori_adj)
            adj_norm = utils.normalize_adj_tensor(modified_adj)
            output = victim_model(ori_features, adj_norm)
            loss = self._loss(output[idx_train], labels[idx_train])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            victim_model.eval()
            modified_adj = self.get_modified_adj(ori_adj)
            adj_norm = utils.normalize_adj_tensor(modified_adj)
            output = victim_model(ori_features, adj_norm)
            loss = self._loss(output[idx_train], labels[idx_train])
            adj_grad = torch.autograd.grad(loss, self.adj_changes)[0]
            if self.loss_type == 'CE':
                lr = 200 / np.sqrt(t + 1)
                self.adj_changes.data.add_(lr * adj_grad)
            if self.loss_type == 'CW':
                lr = 0.1 / np.sqrt(t + 1)
                self.adj_changes.data.add_(lr * adj_grad)
            self.projection(n_perturbations)
        self.random_sample(ori_adj, ori_features, labels, idx_train, n_perturbations)
        self.modified_adj = self.get_modified_adj(ori_adj).detach()


class StaticGraph(object):
    graph = None

    @staticmethod
    def get_gsize():
        return torch.Size((len(StaticGraph.graph), len(StaticGraph.graph)))


class GraphNormTool(object):

    def __init__(self, normalize, gm, device):
        self.adj_norm = normalize
        self.gm = gm
        g = StaticGraph.graph
        edges = np.array(g.edges(), dtype=np.int64)
        rev_edges = np.array([edges[:, 1], edges[:, 0]], dtype=np.int64)
        edges = np.hstack((edges.T, rev_edges))
        idxes = torch.LongTensor(edges)
        values = torch.ones(idxes.size()[1])
        self.raw_adj = torch.sparse.FloatTensor(idxes, values, StaticGraph.get_gsize())
        self.raw_adj = self.raw_adj
        self.normed_adj = self.raw_adj.clone()
        if self.adj_norm:
            if self.gm == 'gcn':
                self.normed_adj = utils.normalize_adj_tensor(self.normed_adj, sparse=True)
            else:
                self.normed_adj = utils.degree_normalize_adj_tensor(self.normed_adj, sparse=True)

    def norm_extra(self, added_adj=None):
        if added_adj is None:
            return self.normed_adj
        new_adj = self.raw_adj + added_adj
        if self.adj_norm:
            if self.gm == 'gcn':
                new_adj = utils.normalize_adj_tensor(new_adj, sparse=True)
            else:
                new_adj = utils.degree_normalize_adj_tensor(new_adj, sparse=True)
        return new_adj


def node_greedy_actions(target_nodes, picked_nodes, list_q, net):
    assert len(target_nodes) == len(list_q)
    actions = []
    values = []
    for i in range(len(target_nodes)):
        region = net.list_action_space[target_nodes[i]]
        if picked_nodes is not None and picked_nodes[i] is not None:
            region = net.list_action_space[picked_nodes[i]]
        if region is None:
            assert list_q[i].size()[0] == net.total_nodes
        else:
            assert len(region) == list_q[i].size()[0]
        val, act = torch.max(list_q[i], dim=0)
        values.append(val)
        if region is not None:
            act = region[act.data.cpu().numpy()[0]]
            act = torch.LongTensor([act])
            actions.append(act)
        else:
            actions.append(act)
    return torch.cat(actions, dim=0).data, torch.cat(values, dim=0).data


def glorot_uniform(t):
    if len(t.size()) == 2:
        fan_in, fan_out = t.size()
    elif len(t.size()) == 3:
        fan_in = t.size()[1] * t.size()[2]
        fan_out = t.size()[0] * t.size()[2]
    else:
        fan_in = np.prod(t.size())
        fan_out = np.prod(t.size())
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    t.uniform_(-limit, limit)


def _param_init(m):
    if isinstance(m, Parameter):
        glorot_uniform(m.data)
    elif isinstance(m, nn.Linear):
        m.bias.data.zero_()
        glorot_uniform(m.weight.data)


def weights_init(m):
    for p in m.modules():
        if isinstance(p, nn.ParameterList):
            for pp in p:
                _param_init(pp)
        else:
            _param_init(p)
    for name, p in m.named_parameters():
        if not '.' in name:
            _param_init(p)


class QNetNode(nn.Module):

    def __init__(self, node_features, node_labels, list_action_space, bilin_q=1, embed_dim=64, mlp_hidden=64, max_lv=1, gm='mean_field', device='cpu'):
        """
        bilin_q: bilinear q or not
        mlp_hidden: mlp hidden layer size
        mav_lv: max rounds of message passing
        """
        super(QNetNode, self).__init__()
        self.node_features = node_features
        self.node_labels = node_labels
        self.list_action_space = list_action_space
        self.total_nodes = len(list_action_space)
        self.bilin_q = bilin_q
        self.embed_dim = embed_dim
        self.mlp_hidden = mlp_hidden
        self.max_lv = max_lv
        self.gm = gm
        if bilin_q:
            last_wout = embed_dim
        else:
            last_wout = 1
            self.bias_target = Parameter(torch.Tensor(1, embed_dim))
        if mlp_hidden:
            self.linear_1 = nn.Linear(embed_dim * 2, mlp_hidden)
            self.linear_out = nn.Linear(mlp_hidden, last_wout)
        else:
            self.linear_out = nn.Linear(embed_dim * 2, last_wout)
        self.w_n2l = Parameter(torch.Tensor(node_features.size()[1], embed_dim))
        self.bias_n2l = Parameter(torch.Tensor(embed_dim))
        self.bias_picked = Parameter(torch.Tensor(1, embed_dim))
        self.conv_params = nn.Linear(embed_dim, embed_dim)
        self.norm_tool = GraphNormTool(normalize=True, gm=self.gm, device=device)
        weights_init(self)

    def make_spmat(self, n_rows, n_cols, row_idx, col_idx):
        idxes = torch.LongTensor([[row_idx], [col_idx]])
        values = torch.ones(1)
        sp = torch.sparse.FloatTensor(idxes, values, torch.Size([n_rows, n_cols]))
        if next(self.parameters()).is_cuda:
            sp = sp
        return sp

    def forward(self, time_t, states, actions, greedy_acts=False, is_inference=False):
        if self.node_features.data.is_sparse:
            input_node_linear = torch.spmm(self.node_features, self.w_n2l)
        else:
            input_node_linear = torch.mm(self.node_features, self.w_n2l)
        input_node_linear += self.bias_n2l
        target_nodes, batch_graph, picked_nodes = zip(*states)
        list_pred = []
        prefix_sum = []
        for i in range(len(batch_graph)):
            region = self.list_action_space[target_nodes[i]]
            node_embed = input_node_linear.clone()
            if picked_nodes is not None and picked_nodes[i] is not None:
                with torch.set_grad_enabled(mode=not is_inference):
                    picked_sp = self.make_spmat(self.total_nodes, 1, picked_nodes[i], 0)
                    node_embed += torch.spmm(picked_sp, self.bias_picked)
                    region = self.list_action_space[picked_nodes[i]]
            if not self.bilin_q:
                with torch.set_grad_enabled(mode=not is_inference):
                    target_sp = self.make_spmat(self.total_nodes, 1, target_nodes[i], 0)
                    node_embed += torch.spmm(target_sp, self.bias_target)
            with torch.set_grad_enabled(mode=not is_inference):
                device = self.node_features.device
                adj = self.norm_tool.norm_extra(batch_graph[i].get_extra_adj(device))
                lv = 0
                input_message = node_embed
                node_embed = F.relu(input_message)
                while lv < self.max_lv:
                    n2npool = torch.spmm(adj, node_embed)
                    node_linear = self.conv_params(n2npool)
                    merged_linear = node_linear + input_message
                    node_embed = F.relu(merged_linear)
                    lv += 1
                target_embed = node_embed[target_nodes[i], :].view(-1, 1)
                if region is not None:
                    node_embed = node_embed[region]
                graph_embed = torch.mean(node_embed, dim=0, keepdim=True)
                if actions is None:
                    graph_embed = graph_embed.repeat(node_embed.size()[0], 1)
                else:
                    if region is not None:
                        act_idx = region.index(actions[i])
                    else:
                        act_idx = actions[i]
                    node_embed = node_embed[act_idx, :].view(1, -1)
                embed_s_a = torch.cat((node_embed, graph_embed), dim=1)
                if self.mlp_hidden:
                    embed_s_a = F.relu(self.linear_1(embed_s_a))
                raw_pred = self.linear_out(embed_s_a)
                if self.bilin_q:
                    raw_pred = torch.mm(raw_pred, target_embed)
                list_pred.append(raw_pred)
        if greedy_acts:
            actions, _ = node_greedy_actions(target_nodes, picked_nodes, list_pred, self)
        return actions, list_pred


class NStepQNetNode(nn.Module):

    def __init__(self, num_steps, node_features, node_labels, list_action_space, bilin_q=1, embed_dim=64, mlp_hidden=64, max_lv=1, gm='mean_field', device='cpu'):
        super(NStepQNetNode, self).__init__()
        self.node_features = node_features
        self.node_labels = node_labels
        self.list_action_space = list_action_space
        self.total_nodes = len(list_action_space)
        list_mod = []
        for i in range(0, num_steps):
            list_mod.append(QNetNode(node_features, node_labels, list_action_space, bilin_q, embed_dim, mlp_hidden, max_lv, gm=gm, device=device))
        self.list_mod = nn.ModuleList(list_mod)
        self.num_steps = num_steps

    def forward(self, time_t, states, actions, greedy_acts=False, is_inference=False):
        assert time_t >= 0 and time_t < self.num_steps
        return self.list_mod[time_t](time_t, states, actions, greedy_acts, is_inference)


class FGA(BaseAttack):
    """FGA/FGSM.

    Parameters
    ----------
    model :
        model to attack
    nnodes : int
        number of nodes in the input graph
    feature_shape : tuple
        shape of the input node features
    attack_structure : bool
        whether to attack graph structure
    attack_features : bool
        whether to attack node features
    device: str
        'cpu' or 'cuda'

    Examples
    --------

    >>> from deeprobust.graph.data import Dataset
    >>> from deeprobust.graph.defense import GCN
    >>> from deeprobust.graph.targeted_attack import FGA
    >>> data = Dataset(root='/tmp/', name='cora')
    >>> adj, features, labels = data.adj, data.features, data.labels
    >>> idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
    >>> # Setup Surrogate model
    >>> surrogate = GCN(nfeat=features.shape[1], nclass=labels.max().item()+1,
                    nhid=16, dropout=0, with_relu=False, with_bias=False, device='cpu').to('cpu')
    >>> surrogate.fit(features, adj, labels, idx_train, idx_val, patience=30)
    >>> # Setup Attack Model
    >>> target_node = 0
    >>> model = FGA(surrogate, nnodes=adj.shape[0], attack_structure=True, attack_features=False, device='cpu').to('cpu')
    >>> # Attack
    >>> model.attack(features, adj, labels, idx_train, target_node, n_perturbations=5)
    >>> modified_adj = model.modified_adj

    """

    def __init__(self, model, nnodes, feature_shape=None, attack_structure=True, attack_features=False, device='cpu'):
        super(FGA, self).__init__(model, nnodes, attack_structure=attack_structure, attack_features=attack_features, device=device)
        assert not self.attack_features, 'not support attacking features'
        if self.attack_features:
            self.feature_changes = Parameter(torch.FloatTensor(feature_shape))
            self.feature_changes.data.fill_(0)

    def attack(self, ori_features, ori_adj, labels, idx_train, target_node, n_perturbations, verbose=False, **kwargs):
        """Generate perturbations on the input graph.

        Parameters
        ----------
        ori_features : scipy.sparse.csr_matrix
            Original (unperturbed) adjacency matrix
        ori_adj : scipy.sparse.csr_matrix
            Original (unperturbed) node feature matrix
        labels :
            node labels
        idx_train:
            training node indices
        target_node : int
            target node index to be attacked
        n_perturbations : int
            Number of perturbations on the input graph. Perturbations could
            be edge removals/additions or feature removals/additions.
        """
        modified_adj = ori_adj.todense()
        modified_features = ori_features.todense()
        modified_adj, modified_features, labels = utils.to_tensor(modified_adj, modified_features, labels, device=self.device)
        self.surrogate.eval()
        if verbose == True:
            None
        pseudo_labels = self.surrogate.predict().detach().argmax(1)
        pseudo_labels[idx_train] = labels[idx_train]
        modified_adj.requires_grad = True
        for i in range(n_perturbations):
            adj_norm = utils.normalize_adj_tensor(modified_adj)
            if self.attack_structure:
                output = self.surrogate(modified_features, adj_norm)
                loss = F.nll_loss(output[[target_node]], pseudo_labels[[target_node]])
                grad = torch.autograd.grad(loss, modified_adj)[0]
                grad = (grad[target_node] + grad[:, target_node]) * (-2 * modified_adj[target_node] + 1)
                grad[target_node] = -10
                grad_argmax = torch.argmax(grad)
            value = -2 * modified_adj[target_node][grad_argmax] + 1
            modified_adj.data[target_node][grad_argmax] += value
            modified_adj.data[grad_argmax][target_node] += value
            if self.attack_features:
                pass
        modified_adj = modified_adj.detach().cpu().numpy()
        modified_adj = sp.csr_matrix(modified_adj)
        self.check_adj(modified_adj)
        self.modified_adj = modified_adj


class IGAttack(BaseAttack):
    """IGAttack: IG-FGSM. Adversarial Examples on Graph Data: Deep Insights into Attack and Defense, https://arxiv.org/pdf/1903.01610.pdf.

    Parameters
    ----------
    model :
        model to attack
    nnodes : int
        number of nodes in the input graph
    feature_shape : tuple
        shape of the input node features
    attack_structure : bool
        whether to attack graph structure
    attack_features : bool
        whether to attack node features
    device: str
        'cpu' or 'cuda'

    Examples
    --------

    >>> from deeprobust.graph.data import Dataset
    >>> from deeprobust.graph.defense import GCN
    >>> from deeprobust.graph.targeted_attack import IGAttack
    >>> data = Dataset(root='/tmp/', name='cora')
    >>> adj, features, labels = data.adj, data.features, data.labels
    >>> idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
    >>> # Setup Surrogate model
    >>> surrogate = GCN(nfeat=features.shape[1], nclass=labels.max().item()+1,
                    nhid=16, dropout=0, with_relu=False, with_bias=False, device='cpu').to('cpu')
    >>> surrogate.fit(features, adj, labels, idx_train, idx_val, patience=30)
    >>> # Setup Attack Model
    >>> target_node = 0
    >>> model = IGAttack(surrogate, nnodes=adj.shape[0], attack_structure=True, attack_features=True, device='cpu').to('cpu')
    >>> # Attack
    >>> model.attack(features, adj, labels, idx_train, target_node, n_perturbations=5, steps=10)
    >>> modified_adj = model.modified_adj
    >>> modified_features = model.modified_features

    """

    def __init__(self, model, nnodes=None, feature_shape=None, attack_structure=True, attack_features=True, device='cpu'):
        super(IGAttack, self).__init__(model, nnodes, attack_structure, attack_features, device)
        assert attack_features or attack_structure, 'attack_features or attack_structure cannot be both False'
        self.modified_adj = None
        self.modified_features = None
        self.target_node = None

    def attack(self, ori_features, ori_adj, labels, idx_train, target_node, n_perturbations, steps=10, **kwargs):
        """Generate perturbations on the input graph.

        Parameters
        ----------
        ori_features :
            Original (unperturbed) node feature matrix
        ori_adj :
            Original (unperturbed) adjacency matrix
        labels :
            node labels
        idx_train:
            training nodes indices
        target_node : int
            target node index to be attacked
        n_perturbations : int
            Number of perturbations on the input graph. Perturbations could
            be edge removals/additions or feature removals/additions.
        steps : int
            steps for computing integrated gradients
        """
        self.surrogate.eval()
        self.target_node = target_node
        modified_adj = ori_adj.todense()
        modified_features = ori_features.todense()
        adj, features, labels = utils.to_tensor(modified_adj, modified_features, labels, device=self.device)
        adj_norm = utils.normalize_adj_tensor(adj)
        pseudo_labels = self.surrogate.predict().detach().argmax(1)
        pseudo_labels[idx_train] = labels[idx_train]
        self.pseudo_labels = pseudo_labels
        s_e = np.zeros(adj.shape[1])
        s_f = np.zeros(features.shape[1])
        if self.attack_structure:
            s_e = self.calc_importance_edge(features, adj_norm, labels, steps)
        if self.attack_features:
            s_f = self.calc_importance_feature(features, adj_norm, labels, steps)
        for t in range(n_perturbations):
            s_e_max = np.argmax(s_e)
            s_f_max = np.argmax(s_f)
            if s_e[s_e_max] >= s_f[s_f_max]:
                if self.attack_structure:
                    value = np.abs(1 - modified_adj[target_node, s_e_max])
                    modified_adj[target_node, s_e_max] = value
                    modified_adj[s_e_max, target_node] = value
                    s_e[s_e_max] = 0
                else:
                    raise Exception("""No posisble perturbation on the structure can be made!
                            See https://github.com/DSE-MSU/DeepRobust/issues/42 for more details.""")
            elif self.attack_features:
                modified_features[target_node, s_f_max] = np.abs(1 - modified_features[target_node, s_f_max])
                s_f[s_f_max] = 0
            else:
                raise Exception("""No posisble perturbation on the features can be made!
                            See https://github.com/DSE-MSU/DeepRobust/issues/42 for more details.""")
        self.modified_adj = sp.csr_matrix(modified_adj)
        self.modified_features = sp.csr_matrix(modified_features)
        self.check_adj(modified_adj)

    def calc_importance_edge(self, features, adj_norm, labels, steps):
        """Calculate integrated gradient for edges. Although I think the the gradient should be
        with respect to adj instead of adj_norm, but the calculation is too time-consuming. So I
        finally decided to calculate the gradient of loss with respect to adj_norm
        """
        baseline_add = adj_norm.clone()
        baseline_remove = adj_norm.clone()
        baseline_add.data[self.target_node] = 1
        baseline_remove.data[self.target_node] = 0
        adj_norm.requires_grad = True
        integrated_grad_list = []
        i = self.target_node
        for j in tqdm(range(adj_norm.shape[1])):
            if adj_norm[i][j]:
                scaled_inputs = [(baseline_remove + float(k) / steps * (adj_norm - baseline_remove)) for k in range(0, steps + 1)]
            else:
                scaled_inputs = [(baseline_add - float(k) / steps * (baseline_add - adj_norm)) for k in range(0, steps + 1)]
            _sum = 0
            for new_adj in scaled_inputs:
                output = self.surrogate(features, new_adj)
                loss = F.nll_loss(output[[self.target_node]], self.pseudo_labels[[self.target_node]])
                adj_grad = torch.autograd.grad(loss, adj_norm)[0]
                adj_grad = adj_grad[i][j]
                _sum += adj_grad
            if adj_norm[i][j]:
                avg_grad = (adj_norm[i][j] - 0) * _sum.mean()
            else:
                avg_grad = (1 - adj_norm[i][j]) * _sum.mean()
            integrated_grad_list.append(avg_grad.detach().item())
        integrated_grad_list[i] = 0
        integrated_grad_list = np.array(integrated_grad_list)
        adj = (adj_norm > 0).cpu().numpy()
        integrated_grad_list = (-2 * adj[self.target_node] + 1) * integrated_grad_list
        integrated_grad_list[self.target_node] = -10
        return integrated_grad_list

    def calc_importance_feature(self, features, adj_norm, labels, steps):
        """Calculate integrated gradient for features
        """
        baseline_add = features.clone()
        baseline_remove = features.clone()
        baseline_add.data[self.target_node] = 1
        baseline_remove.data[self.target_node] = 0
        features.requires_grad = True
        integrated_grad_list = []
        i = self.target_node
        for j in tqdm(range(features.shape[1])):
            if features[i][j]:
                scaled_inputs = [(baseline_add + float(k) / steps * (features - baseline_add)) for k in range(0, steps + 1)]
            else:
                scaled_inputs = [(baseline_remove - float(k) / steps * (baseline_remove - features)) for k in range(0, steps + 1)]
            _sum = 0
            for new_features in scaled_inputs:
                output = self.surrogate(new_features, adj_norm)
                loss = F.nll_loss(output[[self.target_node]], self.pseudo_labels[[self.target_node]])
                feature_grad = torch.autograd.grad(loss, features)[0]
                feature_grad = feature_grad[i][j]
                _sum += feature_grad
            if features[i][j]:
                avg_grad = (features[i][j] - 0) * _sum.mean()
            else:
                avg_grad = (1 - features[i][j]) * _sum.mean()
            integrated_grad_list.append(avg_grad.detach().item())
        features = (features > 0).cpu().numpy()
        integrated_grad_list = np.array(integrated_grad_list)
        integrated_grad_list = (-2 * features[self.target_node] + 1) * integrated_grad_list
        return integrated_grad_list


def compute_alpha(n, sum_log_degrees, d_min):
    try:
        alpha = 1 + n / (sum_log_degrees - n * torch.log(d_min - 0.5))
    except:
        alpha = 1 + n / (sum_log_degrees - n * np.log(d_min - 0.5))
    return alpha


def compute_log_likelihood(n, alpha, sum_log_degrees, d_min):
    try:
        ll = n * torch.log(alpha) + n * alpha * torch.log(d_min) + (alpha + 1) * sum_log_degrees
    except:
        ll = n * np.log(alpha) + n * alpha * np.log(d_min) + (alpha + 1) * sum_log_degrees
    return ll


def filter_chisquare(ll_ratios, cutoff):
    return ll_ratios < cutoff


def filter_singletons(edges, adj):
    """
    Filter edges that, if removed, would turn one or more nodes into singleton nodes.
    """
    degs = np.squeeze(np.array(np.sum(adj, 0)))
    existing_edges = np.squeeze(np.array(adj.tocsr()[tuple(edges.T)]))
    if existing_edges.size > 0:
        edge_degrees = degs[np.array(edges)] + 2 * (1 - existing_edges[:, None]) - 1
    else:
        edge_degrees = degs[np.array(edges)] + 1
    zeros = edge_degrees == 0
    zeros_sum = zeros.sum(1)
    return zeros_sum == 0


def update_Sx(S_old, n_old, d_old, d_new, d_min):
    """
    Update on the sum of log degrees S_d and n based on degree distribution resulting from inserting or deleting
    a single edge.
    """
    old_in_range = d_old >= d_min
    new_in_range = d_new >= d_min
    d_old_in_range = np.multiply(d_old, old_in_range)
    d_new_in_range = np.multiply(d_new, new_in_range)
    new_S_d = S_old - np.log(np.maximum(d_old_in_range, 1)).sum(1) + np.log(np.maximum(d_new_in_range, 1)).sum(1)
    new_n = n_old - np.sum(old_in_range, 1) + np.sum(new_in_range, 1)
    return new_S_d, new_n


class Nettack(BaseAttack):
    """Nettack.

    Parameters
    ----------
    model :
        model to attack
    nnodes : int
        number of nodes in the input graph
    attack_structure : bool
        whether to attack graph structure
    attack_features : bool
        whether to attack node features
    device: str
        'cpu' or 'cuda'

    Examples
    --------

    >>> from deeprobust.graph.data import Dataset
    >>> from deeprobust.graph.defense import GCN
    >>> from deeprobust.graph.targeted_attack import Nettack
    >>> data = Dataset(root='/tmp/', name='cora')
    >>> adj, features, labels = data.adj, data.features, data.labels
    >>> idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
    >>> # Setup Surrogate model
    >>> surrogate = GCN(nfeat=features.shape[1], nclass=labels.max().item()+1,
                    nhid=16, dropout=0, with_relu=False, with_bias=False, device='cpu').to('cpu')
    >>> surrogate.fit(features, adj, labels, idx_train, idx_val, patience=30)
    >>> # Setup Attack Model
    >>> target_node = 0
    >>> model = Nettack(surrogate, nnodes=adj.shape[0], attack_structure=True, attack_features=True, device='cpu').to('cpu')
    >>> # Attack
    >>> model.attack(features, adj, labels, target_node, n_perturbations=5)
    >>> modified_adj = model.modified_adj
    >>> modified_features = model.modified_features

    """

    def __init__(self, model, nnodes=None, attack_structure=True, attack_features=False, device='cpu'):
        super(Nettack, self).__init__(model, nnodes, attack_structure=attack_structure, attack_features=attack_features, device=device)
        self.structure_perturbations = []
        self.feature_perturbations = []
        self.influencer_nodes = []
        self.potential_edges = []
        self.cooc_constraint = None

    def filter_potential_singletons(self, modified_adj):
        """Computes a mask for entries potentially leading to singleton nodes, i.e.
        one of the two nodes corresponding to the entry have degree 1 and there
        is an edge between the two nodes.
        """
        degrees = modified_adj.sum(0)
        degree_one = degrees == 1
        resh = degree_one.repeat(self.nnodes, 1).float()
        l_and = resh * modified_adj
        logical_and_symmetric = l_and + l_and.t()
        flat_mask = 1 - logical_and_symmetric
        return flat_mask

    def get_linearized_weight(self):
        surrogate = self.surrogate
        W = surrogate.gc1.weight @ surrogate.gc2.weight
        return W.detach().cpu().numpy()

    def attack(self, features, adj, labels, target_node, n_perturbations, direct=True, n_influencers=0, ll_cutoff=0.004, verbose=True, **kwargs):
        """Generate perturbations on the input graph.

        Parameters
        ----------
        ori_features : torch.Tensor or scipy.sparse.csr_matrix
            Origina (unperturbed) node feature matrix. Note that
            torch.Tensor will be automatically transformed into
            scipy.sparse.csr_matrix
        ori_adj : torch.Tensor or scipy.sparse.csr_matrix
            Original (unperturbed) adjacency matrix. Note that
            torch.Tensor will be automatically transformed into
            scipy.sparse.csr_matrix
        labels :
            node labels
        target_node : int
            target node index to be attacked
        n_perturbations : int
            Number of perturbations on the input graph. Perturbations could
            be edge removals/additions or feature removals/additions.
        direct: bool
            whether to conduct direct attack
        n_influencers:
            number of influencer nodes when performing indirect attack.
            (setting `direct` to False). When `direct` is True, it would be ignored.
        ll_cutoff : float
            The critical value for the likelihood ratio test of the power law distributions.
            See the Chi square distribution with one degree of freedom. Default value 0.004
            corresponds to a p-value of roughly 0.95.
        verbose : bool
            whether to show verbose logs
        """
        if self.nnodes is None:
            self.nnodes = adj.shape[0]
        self.target_node = target_node
        if type(adj) is torch.Tensor:
            self.ori_adj = utils.to_scipy(adj).tolil()
            self.modified_adj = utils.to_scipy(adj).tolil()
            self.ori_features = utils.to_scipy(features).tolil()
            self.modified_features = utils.to_scipy(features).tolil()
        else:
            self.ori_adj = adj.tolil()
            self.modified_adj = adj.tolil()
            self.ori_features = features.tolil()
            self.modified_features = features.tolil()
        self.cooc_matrix = self.modified_features.T.dot(self.modified_features).tolil()
        attack_features = self.attack_features
        attack_structure = self.attack_structure
        assert not (direct == False and n_influencers == 0), 'indirect mode requires at least one influencer node'
        assert n_perturbations > 0, 'need at least one perturbation'
        assert attack_features or attack_structure, 'either attack_features or attack_structure must be true'
        self.adj_norm = utils.normalize_adj(self.modified_adj)
        self.W = self.get_linearized_weight()
        logits = (self.adj_norm @ self.adj_norm @ self.modified_features @ self.W)[target_node]
        self.label_u = labels[target_node]
        label_target_onehot = np.eye(int(self.nclass))[labels[target_node]]
        best_wrong_class = (logits - 1000 * label_target_onehot).argmax()
        surrogate_losses = [logits[labels[target_node]] - logits[best_wrong_class]]
        if verbose:
            None
            if attack_structure and attack_features:
                None
            elif attack_features:
                None
            elif attack_structure:
                None
            if direct:
                None
            else:
                None
            None
        if attack_structure:
            degree_sequence_start = self.ori_adj.sum(0).A1
            current_degree_sequence = self.modified_adj.sum(0).A1
            d_min = 2
            S_d_start = np.sum(np.log(degree_sequence_start[degree_sequence_start >= d_min]))
            current_S_d = np.sum(np.log(current_degree_sequence[current_degree_sequence >= d_min]))
            n_start = np.sum(degree_sequence_start >= d_min)
            current_n = np.sum(current_degree_sequence >= d_min)
            alpha_start = compute_alpha(n_start, S_d_start, d_min)
            log_likelihood_orig = compute_log_likelihood(n_start, alpha_start, S_d_start, d_min)
        if len(self.influencer_nodes) == 0:
            if not direct:
                infls, add_infls = self.get_attacker_nodes(n_influencers, add_additional_nodes=True)
                self.influencer_nodes = np.concatenate((infls, add_infls)).astype('int')
                self.potential_edges = np.row_stack([np.column_stack((np.tile(infl, self.nnodes - 2), np.setdiff1d(np.arange(self.nnodes), np.array([target_node, infl])))) for infl in self.influencer_nodes])
                if verbose:
                    None
            else:
                influencers = [target_node]
                self.potential_edges = np.column_stack((np.tile(target_node, self.nnodes - 1), np.setdiff1d(np.arange(self.nnodes), target_node)))
                self.influencer_nodes = np.array(influencers)
        self.potential_edges = self.potential_edges.astype('int32')
        for _ in range(n_perturbations):
            if verbose:
                None
            if attack_structure:
                singleton_filter = filter_singletons(self.potential_edges, self.modified_adj)
                filtered_edges = self.potential_edges[singleton_filter]
                deltas = 2 * (1 - self.modified_adj[tuple(filtered_edges.T)].toarray()[0]) - 1
                d_edges_old = current_degree_sequence[filtered_edges]
                d_edges_new = current_degree_sequence[filtered_edges] + deltas[:, None]
                new_S_d, new_n = update_Sx(current_S_d, current_n, d_edges_old, d_edges_new, d_min)
                new_alphas = compute_alpha(new_n, new_S_d, d_min)
                new_ll = compute_log_likelihood(new_n, new_alphas, new_S_d, d_min)
                alphas_combined = compute_alpha(new_n + n_start, new_S_d + S_d_start, d_min)
                new_ll_combined = compute_log_likelihood(new_n + n_start, alphas_combined, new_S_d + S_d_start, d_min)
                new_ratios = -2 * new_ll_combined + 2 * (new_ll + log_likelihood_orig)
                powerlaw_filter = filter_chisquare(new_ratios, ll_cutoff)
                filtered_edges_final = filtered_edges[powerlaw_filter]
                a_hat_uv_new = self.compute_new_a_hat_uv(filtered_edges_final, target_node)
                struct_scores = self.struct_score(a_hat_uv_new, self.modified_features @ self.W)
                best_edge_ix = struct_scores.argmin()
                best_edge_score = struct_scores.min()
                best_edge = filtered_edges_final[best_edge_ix]
            if attack_features:
                feature_ixs, feature_scores = self.feature_scores()
                best_feature_ix = feature_ixs[0]
                best_feature_score = feature_scores[0]
            if attack_structure and attack_features:
                if best_edge_score < best_feature_score:
                    if verbose:
                        None
                    change_structure = True
                else:
                    if verbose:
                        None
                    change_structure = False
            elif attack_structure:
                change_structure = True
            elif attack_features:
                change_structure = False
            if change_structure:
                self.modified_adj[tuple(best_edge)] = self.modified_adj[tuple(best_edge[::-1])] = 1 - self.modified_adj[tuple(best_edge)]
                self.adj_norm = utils.normalize_adj(self.modified_adj)
                self.structure_perturbations.append(tuple(best_edge))
                self.feature_perturbations.append(())
                surrogate_losses.append(best_edge_score)
                current_S_d = new_S_d[powerlaw_filter][best_edge_ix]
                current_n = new_n[powerlaw_filter][best_edge_ix]
                current_degree_sequence[best_edge] += deltas[powerlaw_filter][best_edge_ix]
            else:
                self.modified_features[tuple(best_feature_ix)] = 1 - self.modified_features[tuple(best_feature_ix)]
                self.feature_perturbations.append(tuple(best_feature_ix))
                self.structure_perturbations.append(())
                surrogate_losses.append(best_feature_score)

    def get_attacker_nodes(self, n=5, add_additional_nodes=False):
        """Determine the influencer nodes to attack node i based on
        the weights W and the attributes X.
        """
        assert n < self.nnodes - 1, 'number of influencers cannot be >= number of nodes in the graph!'
        neighbors = self.ori_adj[self.target_node].nonzero()[1]
        assert self.target_node not in neighbors
        potential_edges = np.column_stack((np.tile(self.target_node, len(neighbors)), neighbors)).astype('int32')
        a_hat_uv = self.compute_new_a_hat_uv(potential_edges, self.target_node)
        XW = self.modified_features @ self.W
        struct_scores = self.struct_score(a_hat_uv, XW)
        if len(neighbors) >= n:
            influencer_nodes = neighbors[np.argsort(struct_scores)[:n]]
            if add_additional_nodes:
                return influencer_nodes, np.array([])
            return influencer_nodes
        else:
            influencer_nodes = neighbors
            if add_additional_nodes:
                poss_add_infl = np.setdiff1d(np.setdiff1d(np.arange(self.nnodes), neighbors), self.target_node)
                n_possible_additional = len(poss_add_infl)
                n_additional_attackers = n - len(neighbors)
                possible_edges = np.column_stack((np.tile(self.target_node, n_possible_additional), poss_add_infl))
                a_hat_uv_additional = self.compute_new_a_hat_uv(possible_edges, self.target_node)
                additional_struct_scores = self.struct_score(a_hat_uv_additional, XW)
                additional_influencers = poss_add_infl[np.argsort(additional_struct_scores)[-n_additional_attackers:]]
                return influencer_nodes, additional_influencers
            else:
                return influencer_nodes

    def compute_logits(self):
        return (self.adj_norm @ self.adj_norm @ self.modified_features @ self.W)[self.target_node]

    def strongest_wrong_class(self, logits):
        label_u_onehot = np.eye(self.nclass)[self.label_u]
        return (logits - 1000 * label_u_onehot).argmax()

    def feature_scores(self):
        """Compute feature scores for all possible feature changes.
        """
        if self.cooc_constraint is None:
            self.compute_cooccurrence_constraint(self.influencer_nodes)
        logits = self.compute_logits()
        best_wrong_class = self.strongest_wrong_class(logits)
        surrogate_loss = logits[self.label_u] - logits[best_wrong_class]
        gradient = self.gradient_wrt_x(self.label_u) - self.gradient_wrt_x(best_wrong_class)
        gradients_flipped = sp.lil_matrix(gradient * -1)
        gradients_flipped[self.modified_features.nonzero()] *= -1
        X_influencers = sp.lil_matrix(self.modified_features.shape)
        X_influencers[self.influencer_nodes] = self.modified_features[self.influencer_nodes]
        gradients_flipped = gradients_flipped.multiply(self.cooc_constraint + X_influencers > 0)
        nnz_ixs = np.array(gradients_flipped.nonzero()).T
        sorting = np.argsort(gradients_flipped[tuple(nnz_ixs.T)]).A1
        sorted_ixs = nnz_ixs[sorting]
        grads = gradients_flipped[tuple(nnz_ixs[sorting].T)]
        scores = surrogate_loss - grads
        return sorted_ixs[::-1], scores.A1[::-1]

    def compute_cooccurrence_constraint(self, nodes):
        """
        Co-occurrence constraint as described in the paper.

        Parameters
        ----------
        nodes: np.array
            Nodes whose features are considered for change

        Returns
        -------
        np.array [len(nodes), D], dtype bool
            Binary matrix of dimension len(nodes) x D. A 1 in entry n,d indicates that
            we are allowed to add feature d to the features of node n.

        """
        words_graph = self.cooc_matrix.copy()
        D = self.modified_features.shape[1]
        words_graph.setdiag(0)
        words_graph = words_graph > 0
        word_degrees = np.sum(words_graph, axis=0).A1
        inv_word_degrees = np.reciprocal(word_degrees.astype(float) + 1e-08)
        sd = np.zeros([self.nnodes])
        for n in range(self.nnodes):
            n_idx = self.modified_features[n, :].nonzero()[1]
            sd[n] = np.sum(inv_word_degrees[n_idx.tolist()])
        scores_matrix = sp.lil_matrix((self.nnodes, D))
        for n in nodes:
            common_words = words_graph.multiply(self.modified_features[n])
            idegs = inv_word_degrees[common_words.nonzero()[1]]
            nnz = common_words.nonzero()[0]
            scores = np.array([idegs[nnz == ix].sum() for ix in range(D)])
            scores_matrix[n] = scores
        self.cooc_constraint = sp.csr_matrix(scores_matrix - 0.5 * sd[:, None] > 0)

    def gradient_wrt_x(self, label):
        return self.adj_norm.dot(self.adj_norm)[self.target_node].T.dot(self.W[:, label].reshape(1, -1))

    def reset(self):
        """Reset Nettack
        """
        self.modified_adj = self.ori_adj.copy()
        self.modified_features = self.ori_features.copy()
        self.structure_perturbations = []
        self.feature_perturbations = []
        self.influencer_nodes = []
        self.potential_edges = []
        self.cooc_constraint = None

    def struct_score(self, a_hat_uv, XW):
        """
        Compute structure scores, cf. Eq. 15 in the paper

        Parameters
        ----------
        a_hat_uv: sp.sparse_matrix, shape [P,2]
            Entries of matrix A_hat^2_u for each potential edge (see paper for explanation)

        XW: sp.sparse_matrix, shape [N, K], dtype float
            The class logits for each node.

        Returns
        -------
        np.array [P,]
            The struct score for every row in a_hat_uv
        """
        logits = a_hat_uv.dot(XW)
        label_onehot = np.eye(XW.shape[1])[self.label_u]
        best_wrong_class_logits = (logits - 1000 * label_onehot).max(1)
        logits_for_correct_class = logits[:, self.label_u]
        struct_scores = logits_for_correct_class - best_wrong_class_logits
        return struct_scores

    def compute_new_a_hat_uv(self, potential_edges, target_node):
        """
        Compute the updated A_hat_square_uv entries that would result from inserting/deleting the input edges,
        for every edge.

        Parameters
        ----------
        potential_edges: np.array, shape [P,2], dtype int
            The edges to check.

        Returns
        -------
        sp.sparse_matrix: updated A_hat_square_u entries, a sparse PxN matrix, where P is len(possible_edges).
        """
        edges = np.array(self.modified_adj.nonzero()).T
        edges_set = {tuple(x) for x in edges}
        A_hat_sq = self.adj_norm @ self.adj_norm
        values_before = A_hat_sq[target_node].toarray()[0]
        node_ixs = np.unique(edges[:, 0], return_index=True)[1]
        twohop_ixs = np.array(A_hat_sq.nonzero()).T
        degrees = self.modified_adj.sum(0).A1 + 1
        ixs, vals = compute_new_a_hat_uv(edges, node_ixs, edges_set, twohop_ixs, values_before, degrees, potential_edges.astype(np.int32), target_node)
        ixs_arr = np.array(ixs)
        a_hat_uv = sp.coo_matrix((vals, (ixs_arr[:, 0], ixs_arr[:, 1])), shape=[len(potential_edges), self.nnodes])
        return a_hat_uv


class RND(BaseAttack):
    """As is described in Adversarial Attacks on Neural Networks for Graph Data (KDD'19),
    'Rnd is an attack in which we modify the structure of the graph. Given our target node v,
    in each step we randomly sample nodes u whose lable is different from v and
    add the edge u,v to the graph structure

    Parameters
    ----------
    model :
        model to attack
    nnodes : int
        number of nodes in the input graph
    attack_structure : bool
        whether to attack graph structure
    attack_features : bool
        whether to attack node features
    device: str
        'cpu' or 'cuda'

    Examples
    --------

    >>> from deeprobust.graph.data import Dataset
    >>> from deeprobust.graph.targeted_attack import RND
    >>> data = Dataset(root='/tmp/', name='cora')
    >>> adj, features, labels = data.adj, data.features, data.labels
    >>> idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
    >>> # Setup Attack Model
    >>> target_node = 0
    >>> model = RND()
    >>> # Attack
    >>> model.attack(adj, labels, idx_train, target_node, n_perturbations=5)
    >>> modified_adj = model.modified_adj
    >>> # # You can also inject nodes
    >>> # model.add_nodes(features, adj, labels, idx_train, target_node, n_added=10, n_perturbations=100)
    >>> # modified_adj = model.modified_adj

    """

    def __init__(self, model=None, nnodes=None, attack_structure=True, attack_features=False, device='cpu'):
        super(RND, self).__init__(model, nnodes, attack_structure=attack_structure, attack_features=attack_features, device=device)
        assert not self.attack_features, 'RND does NOT support attacking features except adding nodes'

    def attack(self, ori_adj, labels, idx_train, target_node, n_perturbations, **kwargs):
        """
        Randomly sample nodes u whose lable is different from v and
        add the edge u,v to the graph structure. This baseline only
        has access to true class labels in training set

        Parameters
        ----------
        ori_adj : scipy.sparse.csr_matrix
            Original (unperturbed) adjacency matrix
        labels :
            node labels
        idx_train :
            node training indices
        target_node : int
            target node index to be attacked
        n_perturbations : int
            Number of perturbations on the input graph. Perturbations could
            be edge removals/additions or feature removals/additions.
        """
        None
        modified_adj = ori_adj.tolil()
        row = ori_adj[target_node].todense().A1
        diff_label_nodes = [x for x in idx_train if labels[x] != labels[target_node] and row[x] == 0]
        diff_label_nodes = np.random.permutation(diff_label_nodes)
        if len(diff_label_nodes) >= n_perturbations:
            changed_nodes = diff_label_nodes[:n_perturbations]
            modified_adj[target_node, changed_nodes] = 1
            modified_adj[changed_nodes, target_node] = 1
        else:
            changed_nodes = diff_label_nodes
            unlabeled_nodes = [x for x in range(ori_adj.shape[0]) if x not in idx_train and row[x] == 0]
            unlabeled_nodes = np.random.permutation(unlabeled_nodes)
            changed_nodes = np.concatenate([changed_nodes, unlabeled_nodes[:n_perturbations - len(diff_label_nodes)]])
            modified_adj[target_node, changed_nodes] = 1
            modified_adj[changed_nodes, target_node] = 1
        self.check_adj(modified_adj)
        self.modified_adj = modified_adj

    def add_nodes(self, features, ori_adj, labels, idx_train, target_node, n_added=1, n_perturbations=10, **kwargs):
        """
        For each added node, first connect the target node with added fake nodes.
        Then randomly connect the fake nodes with other nodes whose label is
        different from target node. As for the node feature, simply copy arbitary node
        """
        None
        N = ori_adj.shape[0]
        D = features.shape[1]
        modified_adj = self.reshape_mx(ori_adj, shape=(N + n_added, N + n_added))
        modified_features = self.reshape_mx(features, shape=(N + n_added, D))
        diff_labels = [l for l in range(labels.max() + 1) if l != labels[target_node]]
        diff_labels = np.random.permutation(diff_labels)
        possible_nodes = [x for x in idx_train if labels[x] == diff_labels[0]]
        for fake_node in range(N, N + n_added):
            sampled_nodes = np.random.permutation(possible_nodes)[:n_perturbations]
            modified_adj[fake_node, target_node] = 1
            modified_adj[target_node, fake_node] = 1
            for node in sampled_nodes:
                modified_adj[fake_node, node] = 1
                modified_adj[node, fake_node] = 1
            modified_features[fake_node] = features[node]
        self.check_adj(modified_adj)
        self.modified_adj = modified_adj
        self.modified_features = modified_features

    def reshape_mx(self, mx, shape):
        indices = mx.nonzero()
        return sp.csr_matrix((mx.data, (indices[0], indices[1])), shape=shape).tolil()


SubGraph = namedtuple('SubGraph', ['edge_index', 'non_edge_index', 'self_loop', 'self_loop_weight', 'edge_weight', 'non_edge_weight', 'edges_all'])


class SGAttack(BaseAttack):
    """SGAttack proposed in `Adversarial Attack on Large Scale Graph` TKDE 2021
    <https://arxiv.org/abs/2009.03488>

    SGAttack follows these steps::
    + training a surrogate SGC model with hop K
    + extrack a K-hop subgraph centered at target node
    + choose top-N attacker nodes that belong to the best wrong classes of the target node
    + compute gradients w.r.t to the subgraph to add or remove edges iteratively

    Parameters
    ----------
    model :
        model to attack
    nnodes : int
        number of nodes in the input graph
    attack_structure : bool
        whether to attack graph structure
    attack_features : bool
        whether to attack node features
    device: str
        'cpu' or 'cuda'

    Examples
    --------

    >>> from deeprobust.graph.data import Dataset
    >>> from deeprobust.graph.defense import SGC
    >>> data = Dataset(root='/tmp/', name='cora')
    >>> adj, features, labels = data.adj, data.features, data.labels
    >>> idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
    >>> surrogate = SGC(nfeat=features.shape[1], K=3, lr=0.1,
              nclass=labels.max().item() + 1, device='cuda')
    >>> surrogate = surrogate.to('cuda')
    >>> pyg_data = Dpr2Pyg(data) # convert deeprobust dataset to pyg dataset
    >>> surrogate.fit(pyg_data, train_iters=200, patience=200, verbose=True) # train with earlystopping
    >>> from deeprobust.graph.targeted_attack import SGAttack
    >>> # Setup Attack Model
    >>> target_node = 0
    >>> model = SGAttack(surrogate, attack_structure=True, device=device)
    >>> # Attack
    >>> model.attack(features, adj, labels, target_node, n_perturbations=5)
    >>> modified_adj = model.modified_adj
    >>> modified_features = model.modified_features
    """

    def __init__(self, model, nnodes=None, attack_structure=True, attack_features=False, device='cpu'):
        super(SGAttack, self).__init__(model=None, nnodes=nnodes, attack_structure=attack_structure, attack_features=attack_features, device=device)
        self.target_node = None
        self.logits = model.predict()
        self.K = model.conv1.K
        W = model.conv1.lin.weight
        b = model.conv1.lin.bias
        if b is not None:
            b = b
        self.weight, self.bias = W, b

    @lru_cache(maxsize=1)
    def compute_XW(self):
        return F.linear(self.modified_features, self.weight)

    def attack(self, features, adj, labels, target_node, n_perturbations, direct=True, n_influencers=3, **kwargs):
        """Generate perturbations on the input graph.

        Parameters
        ----------
        features :
            Original (unperturbed) node feature matrix
        adj :
            Original (unperturbed) adjacency matrix
        labels :
            node labels
        target_node : int
            target_node node index to be attacked
        n_perturbations : int
            Number of perturbations on the input graph. Perturbations could
            be edge removals/additions or feature removals/additions.
        direct: bool
            whether to conduct direct attack
        n_influencers : int
            number of the top influencers to choose. For direct attack, it will set as `n_perturbations`.
        """
        if sp.issparse(features):
            features = features.A
        if not torch.is_tensor(features):
            features = torch.tensor(features, device=self.device)
        if torch.is_tensor(adj):
            adj = utils.to_scipy(adj).csr()
        self.modified_features = features.requires_grad_(bool(self.attack_features))
        target_label = torch.LongTensor([labels[target_node]])
        best_wrong_label = torch.LongTensor([(self.logits[target_node].cpu() - 1000 * torch.eye(self.logits.size(1))[target_label]).argmax()])
        self.selfloop_degree = torch.tensor(adj.sum(1).A1 + 1, device=self.device)
        self.target_label = target_label
        self.best_wrong_label = best_wrong_label
        self.n_perturbations = n_perturbations
        self.ori_adj = adj
        self.target_node = target_node
        self.direct = direct
        attacker_nodes = torch.where(torch.as_tensor(labels) == best_wrong_label)[0]
        subgraph = self.get_subgraph(attacker_nodes, n_influencers)
        if not direct:
            mask = torch.logical_or(subgraph.edge_index[0] == target_node, subgraph.edge_index[1] == target_node)
        structure_perturbations = []
        feature_perturbations = []
        num_features = features.shape[-1]
        for _ in range(n_perturbations):
            edge_grad, non_edge_grad, features_grad = self.compute_gradient(subgraph)
            max_structure_score = max_feature_score = 0.0
            if self.attack_structure:
                edge_grad *= -2 * subgraph.edge_weight + 1
                non_edge_grad *= -2 * subgraph.non_edge_weight + 1
                min_grad = min(edge_grad.min().item(), non_edge_grad.min().item())
                edge_grad -= min_grad
                non_edge_grad -= min_grad
                if not direct:
                    edge_grad[mask] = 0.0
                max_edge_grad, max_edge_idx = torch.max(edge_grad, dim=0)
                max_non_edge_grad, max_non_edge_idx = torch.max(non_edge_grad, dim=0)
                max_structure_score = max(max_edge_grad.item(), max_non_edge_grad.item())
            if self.attack_features:
                features_grad *= -2 * self.modified_features + 1
                features_grad -= features_grad.min()
                if not direct:
                    features_grad[target_node] = 0.0
                max_feature_grad, max_feature_idx = torch.max(features_grad.view(-1), dim=0)
                max_feature_score = max_feature_grad.item()
            if max_structure_score >= max_feature_score:
                if max_edge_grad > max_non_edge_grad:
                    best_edge = subgraph.edge_index[:, max_edge_idx]
                    subgraph.edge_weight.data[max_edge_idx] = 0.0
                    self.selfloop_degree[best_edge] -= 1.0
                else:
                    best_edge = subgraph.non_edge_index[:, max_non_edge_idx]
                    subgraph.non_edge_weight.data[max_non_edge_idx] = 1.0
                    self.selfloop_degree[best_edge] += 1.0
                u, v = best_edge.tolist()
                structure_perturbations.append((u, v))
            else:
                u, v = divmod(max_feature_idx.item(), num_features)
                feature_perturbations.append((u, v))
                self.modified_features[u, v].data.fill_(1.0 - self.modified_features[u, v].data)
        if structure_perturbations:
            modified_adj = adj.tolil(copy=True)
            row, col = list(zip(*structure_perturbations))
            modified_adj[row, col] = modified_adj[col, row] = 1 - modified_adj[row, col].A
            modified_adj = modified_adj.tocsr(copy=False)
            modified_adj.eliminate_zeros()
        else:
            modified_adj = adj.copy()
        self.modified_adj = modified_adj
        self.modified_features = self.modified_features.detach().cpu().numpy()
        self.structure_perturbations = structure_perturbations
        self.feature_perturbations = feature_perturbations

    def get_subgraph(self, attacker_nodes, n_influencers=None):
        target_node = self.target_node
        neighbors = self.ori_adj[target_node].indices
        sub_nodes, sub_edges = self.ego_subgraph()
        if self.direct or n_influencers is not None:
            influencers = [target_node]
            attacker_nodes = np.setdiff1d(attacker_nodes, neighbors)
        else:
            influencers = neighbors
        subgraph = self.subgraph_processing(influencers, attacker_nodes, sub_nodes, sub_edges)
        if n_influencers is not None and self.attack_structure:
            if self.direct:
                influencers = [target_node]
                attacker_nodes = self.get_topk_influencers(subgraph, k=self.n_perturbations + 1)
            else:
                influencers = neighbors
                attacker_nodes = self.get_topk_influencers(subgraph, k=n_influencers)
            subgraph = self.subgraph_processing(influencers, attacker_nodes, sub_nodes, sub_edges)
        return subgraph

    def get_topk_influencers(self, subgraph, k):
        _, non_edge_grad, _ = self.compute_gradient(subgraph)
        _, topk_nodes = torch.topk(non_edge_grad, k=k, sorted=False)
        influencers = subgraph.non_edge_index[1][topk_nodes.cpu()]
        return influencers.cpu().numpy()

    def subgraph_processing(self, influencers, attacker_nodes, sub_nodes, sub_edges):
        if not self.attack_structure:
            self_loop = sub_nodes.repeat((2, 1))
            edges_all = torch.cat([sub_edges, sub_edges[[1, 0]], self_loop], dim=1)
            edge_weight = torch.ones(edges_all.size(1), device=self.device)
            return SubGraph(edge_index=sub_edges, non_edge_index=None, self_loop=None, edges_all=edges_all, edge_weight=edge_weight, non_edge_weight=None, self_loop_weight=None)
        row = np.repeat(influencers, len(attacker_nodes))
        col = np.tile(attacker_nodes, len(influencers))
        non_edges = np.row_stack([row, col])
        if len(influencers) > 1:
            mask = self.ori_adj[non_edges[0], non_edges[1]].A1 == 0
            non_edges = non_edges[:, mask]
        non_edges = torch.as_tensor(non_edges, device=self.device)
        unique_nodes = np.union1d(sub_nodes.tolist(), attacker_nodes)
        unique_nodes = torch.as_tensor(unique_nodes, device=self.device)
        self_loop = unique_nodes.repeat((2, 1))
        edges_all = torch.cat([sub_edges, sub_edges[[1, 0]], non_edges, non_edges[[1, 0]], self_loop], dim=1)
        edge_weight = torch.ones(sub_edges.size(1), device=self.device).requires_grad_(bool(self.attack_structure))
        non_edge_weight = torch.zeros(non_edges.size(1), device=self.device).requires_grad_(bool(self.attack_structure))
        self_loop_weight = torch.ones(self_loop.size(1), device=self.device)
        edge_index = sub_edges
        non_edge_index = non_edges
        self_loop = self_loop
        subgraph = SubGraph(edge_index=edge_index, non_edge_index=non_edge_index, self_loop=self_loop, edges_all=edges_all, edge_weight=edge_weight, non_edge_weight=non_edge_weight, self_loop_weight=self_loop_weight)
        return subgraph

    def SGCCov(self, x, edge_index, edge_weight):
        row, col = edge_index
        for _ in range(self.K):
            src = x[row] * edge_weight.view(-1, 1)
            x = scatter_add(src, col, dim=-2, dim_size=x.size(0))
        return x

    def compute_gradient(self, subgraph, eps=5.0):
        if self.attack_structure:
            edge_weight = subgraph.edge_weight
            non_edge_weight = subgraph.non_edge_weight
            self_loop_weight = subgraph.self_loop_weight
            weights = torch.cat([edge_weight, edge_weight, non_edge_weight, non_edge_weight, self_loop_weight], dim=0)
        else:
            weights = subgraph.edge_weight
        weights = self.gcn_norm(subgraph.edges_all, weights, self.selfloop_degree)
        logit = self.SGCCov(self.compute_XW(), subgraph.edges_all, weights)
        logit = logit[self.target_node]
        if self.bias is not None:
            logit += self.bias
        logit = F.log_softmax(logit.view(1, -1) / eps, dim=1)
        loss = F.nll_loss(logit, self.target_label) - F.nll_loss(logit, self.best_wrong_label)
        edge_grad = non_edge_grad = features_grad = None
        if self.attack_structure and self.attack_features:
            edge_grad, non_edge_grad, features_grad = torch.autograd.grad(loss, [edge_weight, non_edge_weight, self.modified_features], create_graph=False)
        elif self.attack_structure:
            edge_grad, non_edge_grad = torch.autograd.grad(loss, [edge_weight, non_edge_weight], create_graph=False)
        else:
            features_grad = torch.autograd.grad(loss, self.modified_features, create_graph=False)[0]
        if self.attack_features:
            self.compute_XW.cache_clear()
        return edge_grad, non_edge_grad, features_grad

    def ego_subgraph(self):
        edge_index = np.asarray(self.ori_adj.nonzero())
        edge_index = torch.as_tensor(edge_index, dtype=torch.long, device=self.device)
        sub_nodes, sub_edges, *_ = k_hop_subgraph(int(self.target_node), self.K, edge_index)
        sub_edges = sub_edges[:, sub_edges[0] < sub_edges[1]]
        return sub_nodes, sub_edges

    @staticmethod
    def gcn_norm(edge_index, weights, degree):
        row, col = edge_index
        inv_degree = torch.pow(degree, -0.5)
        normed_weights = weights * inv_degree[row] * inv_degree[col]
        return normed_weights


class GradWhere(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    @staticmethod
    def forward(ctx, input, thrd, device):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        ctx.save_for_backward(input)
        rst = torch.where(input > thrd, torch.tensor(1.0, device=device, requires_grad=True), torch.tensor(0.0, device=device, requires_grad=True))
        return rst

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        """
        Return results number should corresponding with .forward inputs (besides ctx),
        for each input, return a corresponding backward grad
        """
        return grad_input, None, None


class GraphTrojanNet(nn.Module):

    def __init__(self, device, nfeat, nout, layernum=1, dropout=0.0):
        super(GraphTrojanNet, self).__init__()
        layers = []
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
        for l in range(layernum - 1):
            layers.append(nn.Linear(nfeat, nfeat))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))
        self.layers = nn.Sequential(*layers)
        self.feat = nn.Linear(nfeat, nout * nfeat)
        self.edge = nn.Linear(nfeat, int(nout * (nout - 1) / 2))
        self.device = device

    def forward(self, input, thrd):
        """
        "input", "mask" and "thrd", should already in cuda before sent to this function.
        If using sparse format, corresponding tensor should already in sparse format before
        sent into this function
        """
        GW = GradWhere.apply
        self.layers = self.layers
        h = self.layers(input)
        feat = self.feat(h)
        edge_weight = self.edge(h)
        edge_weight = GW(edge_weight, thrd, self.device)
        return feat, edge_weight


class HomoLoss(nn.Module):

    def __init__(self, device):
        super(HomoLoss, self).__init__()
        self.device = device

    def forward(self, trigger_edge_index, trigger_edge_weights, x, thrd):
        trigger_edge_index = trigger_edge_index[:, trigger_edge_weights > 0.0]
        edge_sims = F.cosine_similarity(x[trigger_edge_index[0]], x[trigger_edge_index[1]])
        loss = torch.relu(thrd - edge_sims).mean()
        return loss


def accuracy(output, labels):
    """Return accuracy of output compared to labels.

    Parameters
    ----------
    output : torch.Tensor
        output from model
    labels : torch.Tensor or numpy.array
        node labels

    Returns
    -------
    float
        accuracy
    """
    if not hasattr(labels, '__len__'):
        labels = [labels]
    if type(labels) is not torch.Tensor:
        labels = torch.LongTensor(labels)
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


class Backdoor:

    def __init__(self, target_class, trigger_size, target_loss_weight, homo_loss_weight, homo_boost_thrd, trojan_epochs, inner, thrd, lr, hidden, weight_decay, seed, debug, device):
        self.device = device
        self.weights = None
        self.trigger_size = trigger_size
        self.thrd = thrd
        self.trigger_index = self.get_trigger_index(self.trigger_size)
        self.hidden = hidden
        self.target_class = target_class
        self.lr = lr
        self.weight_decay = weight_decay
        self.trojan_epochs = trojan_epochs
        self.inner = inner
        self.seed = seed
        self.target_loss_weight = target_loss_weight
        self.homo_boost_thrd = homo_boost_thrd
        self.homo_loss_weight = homo_loss_weight
        self.debug = debug

    def get_trigger_index(self, trigger_size):
        edge_list = []
        edge_list.append([0, 0])
        for j in range(trigger_size):
            for k in range(j):
                edge_list.append([j, k])
        edge_index = torch.tensor(edge_list, device=self.device).long().T
        return edge_index

    def get_trojan_edge(self, start, idx_attach, trigger_size):
        edge_list = []
        for idx in idx_attach:
            edges = self.trigger_index.clone()
            edges[0, 0] = idx
            edges[1, 0] = start
            edges[:, 1:] = edges[:, 1:] + start
            edge_list.append(edges)
            start += trigger_size
        edge_index = torch.cat(edge_list, dim=1)
        row = torch.cat([edge_index[0], edge_index[1]])
        col = torch.cat([edge_index[1], edge_index[0]])
        edge_index = torch.stack([row, col])
        return edge_index

    def inject_trigger(self, idx_attach, features, edge_index, edge_weight, y, device):
        self.trojan = self.trojan
        idx_attach = idx_attach
        features = features
        edge_index = edge_index
        edge_weight = edge_weight
        self.trojan.eval()
        trojan_feat, trojan_weights = self.trojan(features[idx_attach], self.thrd)
        trojan_weights = torch.cat([torch.ones([len(idx_attach), 1], dtype=torch.float, device=device), trojan_weights], dim=1)
        trojan_weights = trojan_weights.flatten()
        trojan_feat = trojan_feat.view([-1, features.shape[1]])
        trojan_edge = self.get_trojan_edge(len(features), idx_attach, self.trigger_size)
        update_edge_weights = torch.cat([edge_weight, trojan_weights, trojan_weights])
        update_feat = torch.cat([features, trojan_feat])
        update_edge_index = torch.cat([edge_index, trojan_edge], dim=1)
        update_y = torch.cat([y, -1 * torch.ones([len(idx_attach) * self.trigger_size], dtype=torch.long, device=device)])
        self.trojan = self.trojan.cpu()
        idx_attach = idx_attach.cpu()
        features = features.cpu()
        edge_index = edge_index.cpu()
        edge_weight = edge_weight.cpu()
        return update_feat, update_edge_index, update_edge_weights, update_y

    def fit(self, features, edge_index, edge_weight, labels, idx_train, idx_attach, idx_unlabeled):
        if edge_weight is None:
            edge_weight = torch.ones([edge_index.shape[1]], device=self.device, dtype=torch.float)
        self.idx_attach = idx_attach
        self.features = features
        self.edge_index = edge_index
        self.edge_weights = edge_weight
        self.shadow_model = GCN(nfeat=features.shape[1], nhid=self.hidden, nclass=labels.max().item() + 1, dropout=0.0, device=self.device)
        self.trojan = GraphTrojanNet(self.device, features.shape[1], self.trigger_size, layernum=2)
        self.homo_loss = HomoLoss(self.device)
        optimizer_shadow = optim.Adam(self.shadow_model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        optimizer_trigger = optim.Adam(self.trojan.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.labels = labels.clone()
        self.labels[idx_attach] = self.target_class
        trojan_edge = self.get_trojan_edge(len(features), idx_attach, self.trigger_size)
        poison_edge_index = torch.cat([edge_index, trojan_edge], dim=1)
        loss_best = 100000000.0
        for i in range(self.trojan_epochs):
            self.trojan.train()
            for j in range(self.inner):
                optimizer_shadow.zero_grad()
                trojan_feat, trojan_weights = self.trojan(features[idx_attach], self.thrd)
                trojan_weights = torch.cat([torch.ones([len(trojan_feat), 1], dtype=torch.float, device=self.device), trojan_weights], dim=1)
                trojan_weights = trojan_weights.flatten()
                trojan_feat = trojan_feat.view([-1, features.shape[1]])
                poison_edge_weights = torch.cat([edge_weight, trojan_weights, trojan_weights]).detach()
                poison_x = torch.cat([features, trojan_feat]).detach()
                output = self.shadow_model(poison_x, poison_edge_index, poison_edge_weights)
                loss_inner = F.nll_loss(output[torch.cat([idx_train, idx_attach])], self.labels[torch.cat([idx_train, idx_attach])])
                loss_inner.backward()
                optimizer_shadow.step()
            acc_train_clean = accuracy(output[idx_train], self.labels[idx_train])
            acc_train_attach = accuracy(output[idx_attach], self.labels[idx_attach])
            self.trojan.eval()
            optimizer_trigger.zero_grad()
            rs = np.random.RandomState(self.seed)
            idx_outter = torch.cat([idx_attach, idx_unlabeled[rs.choice(len(idx_unlabeled), size=512, replace=False)]])
            trojan_feat, trojan_weights = self.trojan(features[idx_outter], self.thrd)
            trojan_weights = torch.cat([torch.ones([len(idx_outter), 1], dtype=torch.float, device=self.device), trojan_weights], dim=1)
            trojan_weights = trojan_weights.flatten()
            trojan_feat = trojan_feat.view([-1, features.shape[1]])
            trojan_edge = self.get_trojan_edge(len(features), idx_outter, self.trigger_size)
            update_edge_weights = torch.cat([edge_weight, trojan_weights, trojan_weights])
            update_feat = torch.cat([features, trojan_feat])
            update_edge_index = torch.cat([edge_index, trojan_edge], dim=1)
            output = self.shadow_model(update_feat, update_edge_index, update_edge_weights)
            labels_outter = labels.clone()
            labels_outter[idx_outter] = self.target_class
            loss_target = self.target_loss_weight * F.nll_loss(output[torch.cat([idx_train, idx_outter])], labels_outter[torch.cat([idx_train, idx_outter])])
            loss_homo = 0.0
            if self.homo_loss_weight > 0:
                loss_homo = self.homo_loss(trojan_edge[:, :int(trojan_edge.shape[1] / 2)], trojan_weights, update_feat, self.homo_boost_thrd)
            loss_outter = loss_target + self.homo_loss_weight * loss_homo
            loss_outter.backward()
            optimizer_trigger.step()
            acc_train_outter = (output[idx_outter].argmax(dim=1) == self.target_class).float().mean()
            if loss_outter < loss_best:
                self.weights = deepcopy(self.trojan.state_dict())
                loss_best = float(loss_outter)
            if self.debug and i % 10 == 0:
                None
                None
        if self.debug:
            None
        self.trojan.load_state_dict(self.weights)
        self.trojan.eval()

    def get_poisoned(self):
        with torch.no_grad():
            poison_x, poison_edge_index, poison_edge_weights, poison_labels = self.inject_trigger(self.idx_attach, self.features, self.edge_index, self.edge_weights, self.labels, self.device)
        poison_edge_index = poison_edge_index[:, poison_edge_weights > 0.0]
        poison_edge_weights = poison_edge_weights[poison_edge_weights > 0.0]
        return poison_x, poison_edge_index, poison_edge_weights, poison_labels


class GCN_body(nn.Module):

    def __init__(self, nfeat, nhid, dropout=0.5, layer=2, device=None, layer_norm_first=False, use_ln=False):
        super(GCN_body, self).__init__()
        self.device = device
        self.nfeat = nfeat
        self.hidden_sizes = [nhid]
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(nfeat, nhid))
        self.lns = nn.ModuleList()
        self.lns.append(torch.nn.LayerNorm(nfeat))
        for _ in range(layer - 1):
            self.convs.append(GCNConv(nhid, nhid))
            self.lns.append(nn.LayerNorm(nhid))
        self.lns.append(torch.nn.LayerNorm(nhid))
        self.layer_norm_first = layer_norm_first
        self.use_ln = use_ln

    def forward(self, x, edge_index, edge_weight=None):
        if self.layer_norm_first:
            x = self.lns[0](x)
        i = 0
        for conv in self.convs:
            x = F.relu(conv(x, edge_index, edge_weight))
            if self.use_ln:
                x = self.lns[i + 1](x)
            i += 1
            x = F.dropout(x, self.dropout, training=self.training)
        return x


class GCN_Encoder(nn.Module):

    def __init__(self, nfeat, nhid, nclass, dropout=0.5, lr=0.01, weight_decay=0.0005, layer=2, device=None, use_ln=False, layer_norm_first=False):
        super(GCN_Encoder, self).__init__()
        assert device is not None, "Please specify 'device'!"
        self.device = device
        self.nfeat = nfeat
        self.hidden_sizes = [nhid]
        self.nclass = nclass
        self.use_ln = use_ln
        self.layer_norm_first = layer_norm_first
        self.body = GCN_body(nfeat, nhid, dropout, layer, device=None, use_ln=use_ln, layer_norm_first=layer_norm_first)
        self.fc = nn.Linear(nhid, nclass)
        self.dropout = dropout
        self.lr = lr
        self.output = None
        self.edge_index = None
        self.edge_weight = None
        self.features = None
        self.weight_decay = weight_decay

    def forward(self, x, edge_index, edge_weight=None):
        x = self.body(x, edge_index, edge_weight)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)

    def get_h(self, x, edge_index, edge_weight):
        self.eval()
        x = self.body(x, edge_index, edge_weight)
        return x

    def fit(self, features, edge_index, edge_weight, labels, idx_train, idx_val=None, train_iters=200, verbose=False):
        """Train the gcn model, when idx_val is not None, pick the best model according to the validation loss.
        Parameters
        ----------
        features :
            node features
        adj :
            the adjacency matrix. The format could be torch.tensor or scipy matrix
        labels :
            node labels
        idx_train :
            node training indices
        idx_val :
            node validation indices. If not given (None), GCN training process will not adpot early stopping
        train_iters : int
            number of training epochs
        initialize : bool
            whether to initialize parameters before training
        verbose : bool
            whether to show verbose logs
        """
        self.edge_index, self.edge_weight = edge_index, edge_weight
        self.features = features
        self.labels = labels
        if idx_val is None:
            self._train_without_val(self.labels, idx_train, train_iters, verbose)
        else:
            self._train_with_val(self.labels, idx_train, idx_val, train_iters, verbose)

    def _train_without_val(self, labels, idx_train, train_iters, verbose):
        self.train()
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        for i in range(train_iters):
            optimizer.zero_grad()
            output = self.forward(self.features, self.edge_index, self.edge_weight)
            loss_train = F.nll_loss(output[idx_train], labels[idx_train])
            loss_train.backward()
            optimizer.step()
            if verbose and i % 10 == 0:
                None
        self.eval()
        output = self.forward(self.features, self.edge_index, self.edge_weight)
        self.output = output

    def _train_with_val(self, labels, idx_train, idx_val, train_iters, verbose):
        if verbose:
            None
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        best_loss_val = 100
        best_acc_val = 0
        for i in range(train_iters):
            self.train()
            optimizer.zero_grad()
            output = self.forward(self.features, self.edge_index, self.edge_weight)
            loss_train = F.nll_loss(output[idx_train], labels[idx_train])
            loss_train.backward()
            optimizer.step()
            self.eval()
            output = self.forward(self.features, self.edge_index, self.edge_weight)
            loss_val = F.nll_loss(output[idx_val], labels[idx_val])
            acc_val = accuracy(output[idx_val], labels[idx_val])
            if verbose and i % 10 == 0:
                None
                None
            if acc_val > best_acc_val:
                best_acc_val = acc_val
                self.output = output
                weights = deepcopy(self.state_dict())
        if verbose:
            None
        self.load_state_dict(weights)

    def test(self, features, edge_index, edge_weight, labels, idx_test):
        """Evaluate GCN performance on test set.
        Parameters
        ----------
        idx_test :
            node testing indices
        """
        self.eval()
        with torch.no_grad():
            output = self.forward(features, edge_index, edge_weight)
            acc_test = accuracy(output[idx_test], labels[idx_test])
        return float(acc_test)

    def test_with_correct_nodes(self, features, edge_index, edge_weight, labels, idx_test):
        self.eval()
        output = self.forward(features, edge_index, edge_weight)
        correct_nids = (output.argmax(dim=1)[idx_test] == labels[idx_test]).nonzero().flatten()
        acc_test = accuracy(output[idx_test], labels[idx_test])
        return acc_test, correct_nids


class UGBA(BaseAttack):
    """
    Modified from Unnoticeable Backdoor Attacks on Graph Neural Networks (WWW 2023).

    see example in examples/graph/test_ugba.py

    Parameters
    ----------
    vs_number: int
        number of selected poisoned for training backdoor model
    
    device: str
        'cpu' or 'cuda'

    target_class: int
        the class that the attacker aim to misclassify into

    trigger_size: int
        the number of nodes in a trigger
    
    target_loss_weight: float

    homo_loss_weight: float
        the weight of homophily loss

    homo_boost_thrd: float
        the upper bound of similarity 

    train_epochs: int
        the number of epochs when training GCN encoder 
    
    trojan_epochs: int
        the number of epochs when training trigger generator


    """

    def __init__(self, data, vs_number, target_class=0, trigger_size=3, target_loss_weight=1, homo_loss_weight=100, homo_boost_thrd=0.8, train_epochs=200, trojan_epochs=800, dis_weight=1, inner=1, thrd=0.5, lr=0.01, hidden=32, weight_decay=0.0005, seed=10, debug=True, device='cpu'):
        self.device = device
        self.data = data
        self.size = vs_number
        self.target_class = target_class
        self.trigger_size = trigger_size
        self.target_loss_weight = target_loss_weight
        self.homo_loss_weight = homo_loss_weight
        self.homo_boost_thrd = homo_boost_thrd
        self.train_epochs = train_epochs
        self.trojan_epochs = trojan_epochs
        self.dis_weight = dis_weight
        self.inner = inner
        self.thrd = thrd
        self.lr = lr
        self.hidden = hidden
        self.weight_decay = weight_decay
        self.seed = seed
        self.debug = debug
        self.unlabeled_idx = (torch.bitwise_not(data.test_mask) & torch.bitwise_not(data.train_mask)).nonzero().flatten()
        self.idx_val = utils.index_to_mask(data.val_mask, size=data.x.shape[0])

    def attack(self, target_node, x, y, edge_index, edge_weights=None):
        """
        inject the generated trigger to the target node (a single node)

        Parameters
        ----------
        target_node: int
            the index of target node
        x: tensor:
            features of nodes
        y: tensor:
            node labels
        edge_index: tensor:
            edge index of the graph
        edge_weights: tensor:
            the weights of edges
        """
        idx_target = torch.tensor([target_node])
        None
        if edge_weights == None:
            edge_weights = torch.ones([edge_index.shape[1]])
        x, edge_index, edge_weights, y = self.inject_trigger(idx_target, x, y, edge_index, edge_weights)
        return x, edge_index, edge_weights, y

    def get_poisoned_graph(self):
        """
        Obtain the poisoned training graph for training backdoor GNN
        """
        assert self.trigger_generator, 'please first use train_trigger_generator() to train trigger generator and get poisoned nodes'
        poison_x, poison_edge_index, poison_edge_weights, poison_labels = self.trigger_generator.get_poisoned()
        idx_bkd_tn = torch.cat([self.idx_train, self.idx_attach])
        poison_data = copy.deepcopy(self.data)
        idx_val = poison_data.val_mask.nonzero().flatten()
        idx_test = poison_data.test_mask.nonzero().flatten()
        poison_data.x, poison_data.edge_index, poison_data.edge_weights, poison_data.y = poison_x, poison_edge_index, poison_edge_weights, poison_labels
        poison_data.train_mask = utils.index_to_mask(idx_bkd_tn, poison_data.x.shape[0])
        poison_data.val_mask = utils.index_to_mask(idx_val, poison_data.x.shape[0])
        poison_data.test_mask = utils.index_to_mask(idx_test, poison_data.x.shape[0])
        return poison_data

    def train_trigger_generator(self, idx_train, edge_index, edge_weights=None, selection_method='cluster', **kwargs):
        """
        Train the adpative trigger generator 
        
        Parameters
        ----------
        idx_train: tensor: 
            indexs of training nodes
        edge_index: tensor:
            edge index of the graph
        edge_weights: tensor:
            the weights of edges
        selection method : ['none', 'cluster']
            the method to select poisoned nodes
        """
        self.idx_train = idx_train
        idx_attach = self.select_idx_attach(selection_method, edge_index, edge_weights)
        self.idx_attach = idx_attach
        None
        trigger_generator = Backdoor(self.target_class, self.trigger_size, self.target_loss_weight, self.homo_loss_weight, self.homo_boost_thrd, self.trojan_epochs, self.inner, self.thrd, self.lr, self.hidden, self.weight_decay, self.seed, self.debug, self.device)
        self.trigger_generator = trigger_generator
        self.trigger_generator.fit(self.data.x, edge_index, edge_weights, self.data.y, idx_train, idx_attach, self.unlabeled_idx)
        return self.trigger_generator, idx_attach

    def inject_trigger(self, idx_attach, x, y, edge_index, edge_weights):
        """
        Attach the generated triggers with the attachde nodes
        
        Parameters
        ----------
        idx_attach: tensor: 
            indexs of to-be attached nodes
        x: tensor:
            features of nodes
        y: tensor:
            node labels
        edge_index: tensor:
            edge index of the graph
        edge_weights: tensor:
            the weights of edges
        """
        assert self.trigger_generator, 'please first use train_trigger_generator() to train trigger generator'
        update_x, update_edge_index, update_edge_weights, update_y = self.trigger_generator.inject_trigger(idx_attach, x, edge_index, edge_weights, y, self.device)
        return update_x, update_edge_index, update_edge_weights, update_y

    def select_idx_attach(self, selection_method, edge_index, edge_weights=None):
        if selection_method == 'none':
            idx_attach = self.obtain_attach_nodes(self.unlabeled_idx, self.size)
        elif selection_method == 'cluster':
            idx_attach = self.cluster_selection(self.data, self.idx_train, self.idx_val, self.unlabeled_idx, self.size, edge_index, edge_weights)
            idx_attach = torch.LongTensor(idx_attach)
        return idx_attach

    def obtain_attach_nodes(self, node_idxs, size):
        size = min(len(node_idxs), size)
        rs = np.random.RandomState(self.seed)
        choice = np.arange(len(node_idxs))
        rs.shuffle(choice)
        return node_idxs[choice[:size]]

    def cluster_selection(self, data, idx_train, idx_val, unlabeled_idx, size, edge_index, edge_weights=None):
        gcn_encoder = GCN_Encoder(nfeat=data.x.shape[1], nhid=32, nclass=int(data.y.max() + 1), dropout=0.5, lr=0.01, weight_decay=0.0005, device=self.device, use_ln=False, layer_norm_first=False)
        t_total = time.time()
        None
        gcn_encoder.fit(data.x, edge_index, edge_weights, data.y, idx_train, idx_val=idx_val, train_iters=self.train_epochs, verbose=True)
        None
        None
        seen_node_idx = torch.concat([idx_train, unlabeled_idx])
        nclass = np.unique(data.y.cpu().numpy()).shape[0]
        encoder_x = gcn_encoder.get_h(data.x, edge_index, edge_weights).clone().detach()
        kmeans = KMeans(n_clusters=nclass, random_state=1)
        kmeans.fit(encoder_x[seen_node_idx].detach().cpu().numpy())
        cluster_centers = kmeans.cluster_centers_
        y_pred = kmeans.predict(encoder_x.cpu().numpy())
        idx_attach = self.obtain_attach_nodes_by_cluster_degree_all(edge_index, y_pred, cluster_centers, unlabeled_idx.cpu().tolist(), encoder_x, size).astype(int)
        idx_attach = idx_attach[:size]
        return idx_attach

    def obtain_attach_nodes_by_cluster_degree_all(self, edge_index, y_pred, cluster_centers, node_idxs, x, size):
        dis_weight = self.dis_weight
        degrees = (degree(edge_index[0]) + degree(edge_index[1])).cpu().numpy()
        distances = []
        for id in range(x.shape[0]):
            tmp_center_label = y_pred[id]
            tmp_center_x = cluster_centers[tmp_center_label]
            dis = np.linalg.norm(tmp_center_x - x[id].detach().cpu().numpy())
            distances.append(dis)
        distances = np.array(distances)
        None
        nontarget_nodes = np.where(y_pred != self.target_class)[0]
        non_target_node_idxs = np.array(list(set(nontarget_nodes) & set(node_idxs)))
        node_idxs = np.array(non_target_node_idxs)
        candiadate_distances = distances[node_idxs]
        candiadate_degrees = degrees[node_idxs]
        candiadate_distances = self.max_norm(candiadate_distances)
        candiadate_degrees = self.max_norm(candiadate_degrees)
        dis_score = candiadate_distances + dis_weight * candiadate_degrees
        candidate_nid_index = np.argsort(dis_score)
        sorted_node_idex = np.array(node_idxs[candidate_nid_index])
        selected_nodes = sorted_node_idex
        return selected_nodes

    def max_norm(self, data):
        _range = np.max(data) - np.min(data)
        return (data - np.min(data)) / _range

