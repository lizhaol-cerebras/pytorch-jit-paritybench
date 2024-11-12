
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


import torch as th


import torch.nn as nn


import torch.nn.functional as F


import re


import numpy as np


import warnings


import random


from scipy.sparse import csr_matrix


from collections import defaultdict


import torch as t


import copy


from torch.utils.data import Dataset


from scipy.sparse import vstack as s_vstack


import time


import math


from copy import deepcopy


import itertools


from random import shuffle


from random import choice


from collections import Counter


import logging


from scipy.sparse import csc_matrix


from scipy.special import softmax


import scipy.sparse as ssp


from typing import Any


from typing import Callable


from typing import List


from typing import Optional


from collections.abc import Sequence


import scipy.sparse as sp


from abc import ABC


import collections


import pandas as pd


import scipy


from torch.utils.data import DataLoader


from scipy import sparse


from scipy import io as sio


from torch import nn


import functools


from torch.optim import Adam


from torch.optim.lr_scheduler import ExponentialLR


from torch.optim.lr_scheduler import ReduceLROnPlateau


from torch.nn import Module


from torch.nn.parameter import Parameter


from scipy.sparse import coo_matrix


from torch.nn import Identity


import abc


from torch.autograd import Variable


from torch import Tensor


from torch.nn import functional as F


from torch.nn.init import xavier_normal_


from torch.nn import Parameter


from typing import Dict


from typing import Tuple


from typing import Union


import scipy.sparse


from itertools import product


import numpy


import torch.sparse as sparse


from collections import OrderedDict


from sklearn.model_selection import train_test_split


from sklearn.metrics import f1_score


from sklearn.svm import LinearSVC


from torch.nn import init


from torch import autograd


from functools import reduce


from numpy.random.mtrand import set_state


import pandas


import torch.utils.data


from abc import ABCMeta


from functools import partial


from torch.optim import SparseAdam


from torch.optim.lr_scheduler import CosineAnnealingLR


from torch.nn import Embedding


from torch.nn import ModuleList


from torch.utils.data import IterableDataset


from torch.serialization import save


from itertools import combinations


from sklearn.metrics import roc_auc_score


from sklearn.metrics import average_precision_score


from sklearn import metrics


from scipy.io import loadmat


import scipy.io as sio


import torch.optim as optim


from abc import abstractmethod


import logging.config


from collections import defaultdict as ddict


from itertools import chain


from time import time


from scipy.sparse import lil_matrix


import torch.utils.data as dataloader


from random import sample


from sklearn.metrics.pairwise import cosine_similarity


from sklearn.cluster import KMeans


from numpy import random


from torch.nn.functional import softmax


from sklearn.metrics import accuracy_score


from sklearn.metrics import precision_score


from sklearn.metrics import recall_score


from torch.utils.tensorboard import SummaryWriter


from sklearn.metrics import precision_recall_curve


from sklearn.metrics import auc


from torch import distributed as dist


from torch.utils import data as torch_data


from re import S


from scipy.stats import rankdata


import uuid


from collections.abc import Mapping


import torch.utils.data as data


from torch.nn.modules.loss import _Loss


from torch.nn.utils.rnn import pad_sequence


from sklearn.metrics import precision_recall_fscore_support


from sklearn.metrics import mean_squared_error


from sklearn.metrics import mean_absolute_error


from torch.utils.data.dataloader import DataLoader


from torch.utils.data.sampler import BatchSampler


from torch.utils.data import TensorDataset


from sklearn.metrics import roc_curve


from sklearn.metrics import normalized_mutual_info_score


from sklearn.metrics import adjusted_rand_score


from sklearn.metrics import ndcg_score


from sklearn.linear_model import LogisticRegression


import sklearn.metrics as Metric


from sklearn import preprocessing


from sklearn.svm import SVC


class MyRGCN(nn.Module):

    def __init__(self, in_feats, hid_feats, out_feats, rel_names):
        super().__init__()
        self.conv1 = dglnn.HeteroGraphConv({rel: dglnn.GraphConv(in_feats, hid_feats) for rel in rel_names}, aggregate='sum')
        self.conv2 = dglnn.HeteroGraphConv({rel: dglnn.GraphConv(hid_feats, out_feats) for rel in rel_names}, aggregate='sum')

    def forward(self, graph, inputs):
        h = self.conv1(graph, inputs)
        h = {k: F.relu(v) for k, v in h.items()}
        h = self.conv2(graph, h)
        return h


def Aggr_max(z):
    z = torch.stack(z, dim=1)
    return z.max(1)[0]


def Aggr_mean(z):
    z = torch.stack(z, dim=1)
    return z.mean(1)


def Aggr_sum(z):
    z = torch.stack(z, dim=1)
    return z.sum(1)


class MetapathConv(nn.Module):
    """
    MetapathConv is an aggregation function based on meta-path, which is similar with `dgl.nn.pytorch.HeteroGraphConv`.
    We could choose Attention/ APPNP or any GraphConvLayer to aggregate node features.
    After that we will get embeddings based on different meta-paths and fusion them.

    .. math::
        \\mathbf{Z}=\\mathcal{F}(Z^{\\Phi_1},Z^{\\Phi_2},...,Z^{\\Phi_p})=\\mathcal{F}(f(H,\\Phi_1),f(H,\\Phi_2),...,f(H,\\Phi_p))

    where :math:`\\mathcal{F}` denotes semantic fusion function, such as semantic-attention. :math:`\\Phi_i` denotes meta-path and
    :math:`f` denotes the aggregation function, such as GAT, APPNP.

    Parameters
    ------------
    meta_paths_dict : dict[str, list[tuple(meta-path)]]
        contain multiple meta-paths.
    mods : nn.ModuleDict
        aggregation function
    macro_func : callable aggregation func
        A semantic aggregation way, e.g. 'mean', 'max', 'sum' or 'attention'

    """

    def __init__(self, meta_paths_dict, mods, macro_func, **kargs):
        super(MetapathConv, self).__init__()
        self.mods = mods
        self.meta_paths_dict = meta_paths_dict
        self.SemanticConv = macro_func

    def forward(self, g_dict, h_dict):
        """
        Parameters
        -----------
        g_dict : dict[str: dgl.DGLGraph]
            A dict of DGLGraph(full batch) or DGLBlock(mini batch) extracted by metapaths.
        h_dict : dict[str: torch.Tensor]
            The input features

        Returns
        --------
        h : dict[str: torch.Tensor]
            The output features dict
        """
        outputs = {g.dsttypes[0]: [] for s, g in g_dict.items()}
        for meta_path_name, meta_path in self.meta_paths_dict.items():
            new_g = g_dict[meta_path_name]
            if h_dict.get(meta_path_name) is not None:
                h = h_dict[meta_path_name][new_g.srctypes[0]]
            else:
                h = h_dict[new_g.srctypes[0]]
            outputs[new_g.dsttypes[0]].append(self.mods[meta_path_name](new_g, h).flatten(1))
        rsts = {}
        for ntype, ntype_outputs in outputs.items():
            if len(ntype_outputs) != 0:
                rsts[ntype] = self.SemanticConv(ntype_outputs)
        return rsts


class SemanticAttention(nn.Module):

    def __init__(self, in_size, hidden_size=128):
        super(SemanticAttention, self).__init__()
        self.project = nn.Sequential(nn.Linear(in_size, hidden_size), nn.Tanh(), nn.Linear(hidden_size, 1, bias=False))

    def forward(self, z):
        w = self.project(z).mean(0)
        beta = torch.softmax(w, dim=0)
        beta = beta.expand((z.shape[0],) + beta.shape)
        out_emb = (beta * z).sum(1)
        att_mp = beta.mean(0).squeeze()
        return out_emb, att_mp


class APPNPConv(nn.Module):

    def __init__(self, dim_in, dim_out, bias=False, **kwargs):
        super(APPNPConv, self).__init__()
        self.model = dgl.nn.pytorch.APPNPConv(k=3, alpha=0.5)
        self.lin = nn.Linear(dim_in, dim_out, bias)

    def forward(self, g, h):
        h = self.model(g, h)
        h = self.lin(h)
        return h


class GATConv(nn.Module):

    def __init__(self, dim_in, dim_out, bias=False, **kwargs):
        super(GATConv, self).__init__()
        self.model = dgl.nn.pytorch.GATConv(dim_in, dim_out, num_heads=kwargs['num_heads'], bias=bias, allow_zero_in_degree=True)

    def forward(self, g, h):
        h = self.model(g, h).mean(1)
        return h


class GCNConv(nn.Module):

    def __init__(self):
        super(GCNConv, self).__init__()

    def forward(self, graph, feat, edge_weight=None):
        with graph.local_scope():
            if edge_weight is not None:
                assert edge_weight.shape[0] == graph.number_of_edges()
                graph.edata['_edge_weight'] = edge_weight
                aggregate_fn = fn.u_mul_e('h', '_edge_weight', 'm')
                graph.srcdata['h'] = feat
                graph.update_all(aggregate_fn, fn.sum(msg='m', out='h'))
                rst = graph.dstdata['h']
        return rst


class GINConv(nn.Module):

    def __init__(self, dim_in, dim_out, bias=False, **kwargs):
        super(GINConv, self).__init__()
        lin = nn.Sequential(nn.Linear(dim_in, dim_out, bias), nn.ReLU(), nn.Linear(dim_out, dim_out))
        self.model = dgl.nn.pytorch.GINConv(lin, 'max')

    def forward(self, g, h):
        h = self.model(g, h)
        return h


class HGTConv(nn.Module):
    """Heterogeneous graph transformer convolution from `Heterogeneous Graph Transformer
    <https://arxiv.org/abs/2003.01332>`__

    Given a graph :math:`G(V, E)` and input node features :math:`H^{(l-1)}`,
    it computes the new node features as follows:

    Compute a multi-head attention score for each edge :math:`(s, e, t)` in the graph:

    .. math::

      Attention(s, e, t) = \\text{Softmax}\\left(||_{i\\in[1,h]}ATT-head^i(s, e, t)\\right) \\\\
      ATT-head^i(s, e, t) = \\left(K^i(s)W^{ATT}_{\\phi(e)}Q^i(t)^{\\top}\\right)\\cdot
        \\frac{\\mu_{(\\tau(s),\\phi(e),\\tau(t)}}{\\sqrt{d}} \\\\
      K^i(s) = \\text{K-Linear}^i_{\\tau(s)}(H^{(l-1)}[s]) \\\\
      Q^i(t) = \\text{Q-Linear}^i_{\\tau(t)}(H^{(l-1)}[t]) \\\\

    Compute the message to send on each edge :math:`(s, e, t)`:

    .. math::

      Message(s, e, t) = ||_{i\\in[1, h]} MSG-head^i(s, e, t) \\\\
      MSG-head^i(s, e, t) = \\text{M-Linear}^i_{\\tau(s)}(H^{(l-1)}[s])W^{MSG}_{\\phi(e)} \\\\

    Send messages to target nodes :math:`t` and aggregate:

    .. math::

      \\tilde{H}^{(l)}[t] = \\sum_{\\forall s\\in \\mathcal{N}(t)}\\left( Attention(s,e,t)
      \\cdot Message(s,e,t)\\right)

    Compute new node features:

    .. math::

      H^{(l)}[t]=\\text{A-Linear}_{\\tau(t)}(\\sigma(\\tilde(H)^{(l)}[t])) + H^{(l-1)}[t]

    Parameters
    ----------
    in_size : int
        Input node feature size.
    head_size : int
        Output head size. The output node feature size is ``head_size * num_heads``.
    num_heads : int
        Number of heads. The output node feature size is ``head_size * num_heads``.
    num_ntypes : int
        Number of node types.
    num_etypes : int
        Number of edge types.
    dropout : optional, float
        Dropout rate.
    use_norm : optiona, bool
        If true, apply a layer norm on the output node feature.

    Examples
    --------
    """

    def __init__(self, in_size, head_size, num_heads, num_ntypes, num_etypes, dropout=0.2, use_norm=False):
        super().__init__()
        self.in_size = in_size
        self.head_size = head_size
        self.num_heads = num_heads
        self.sqrt_d = math.sqrt(head_size)
        self.use_norm = use_norm
        self.linear_k = TypedLinear(in_size, head_size * num_heads, num_ntypes)
        self.linear_q = TypedLinear(in_size, head_size * num_heads, num_ntypes)
        self.linear_v = TypedLinear(in_size, head_size * num_heads, num_ntypes)
        self.linear_a = TypedLinear(head_size * num_heads, head_size * num_heads, num_ntypes)
        self.relation_pri = nn.ParameterList([nn.Parameter(torch.ones(num_etypes)) for i in range(num_heads)])
        self.relation_att = nn.ModuleList([TypedLinear(head_size, head_size, num_etypes) for i in range(num_heads)])
        self.relation_msg = nn.ModuleList([TypedLinear(head_size, head_size, num_etypes) for i in range(num_heads)])
        self.skip = nn.Parameter(torch.ones(num_ntypes))
        self.drop = nn.Dropout(dropout)
        if use_norm:
            self.norm = nn.LayerNorm(head_size * num_heads)
        if in_size != head_size * num_heads:
            self.residual_w = nn.Parameter(torch.Tensor(in_size, head_size * num_heads))
            nn.init.xavier_uniform_(self.residual_w)

    def forward(self, g, x, ntype, etype, *, presorted=False):
        """Forward computation.

        Parameters
        ----------
        g : DGLGraph
            The input graph.
        x : torch.Tensor
            A 2D tensor of node features. Shape: :math:`(|V|, D_{in})`.
        ntype : torch.Tensor
            An 1D integer tensor of node types. Shape: :math:`(|V|,)`.
        etype : torch.Tensor
            An 1D integer tensor of edge types. Shape: :math:`(|E|,)`.
        presorted : bool, optional
            Whether *both* the nodes and the edges of the input graph have been sorted by
            their types. Forward on pre-sorted graph may be faster. Graphs created by
            :func:`~dgl.to_homogeneous` automatically satisfy the condition.
            Also see :func:`~dgl.reorder_graph` for manually reordering the nodes and edges.

        Returns
        -------
        torch.Tensor
            New node features. Shape: :math:`(|V|, D_{head} * N_{head})`.
        """
        self.presorted = presorted
        if g.is_block:
            x_src = x
            x_dst = x[:g.num_dst_nodes()]
            srcntype = ntype
            dstntype = ntype[:g.num_dst_nodes()]
        else:
            x_src = x
            x_dst = x
            srcntype = ntype
            dstntype = ntype
        with g.local_scope():
            k = self.linear_k(x_src, srcntype, presorted).view(-1, self.num_heads, self.head_size)
            q = self.linear_q(x_dst, dstntype, presorted).view(-1, self.num_heads, self.head_size)
            v = self.linear_v(x_src, srcntype, presorted).view(-1, self.num_heads, self.head_size)
            g.srcdata['k'] = k
            g.dstdata['q'] = q
            g.srcdata['v'] = v
            g.edata['etype'] = etype
            g.apply_edges(self.message)
            g.edata['m'] = g.edata['m'] * edge_softmax(g, g.edata['a']).unsqueeze(-1)
            g.update_all(fn.copy_e('m', 'm'), fn.sum('m', 'h'))
            h = g.dstdata['h'].view(-1, self.num_heads * self.head_size)
            h = self.drop(self.linear_a(h, dstntype, presorted))
            alpha = torch.sigmoid(self.skip[dstntype]).unsqueeze(-1)
            if x_dst.shape != h.shape:
                h = h * alpha + x_dst @ self.residual_w * (1 - alpha)
            else:
                h = h * alpha + x_dst * (1 - alpha)
            if self.use_norm:
                h = self.norm(h)
            return h

    def message(self, edges):
        """Message function."""
        a, m = [], []
        etype = edges.data['etype']
        k = torch.unbind(edges.src['k'], dim=1)
        q = torch.unbind(edges.dst['q'], dim=1)
        v = torch.unbind(edges.src['v'], dim=1)
        for i in range(self.num_heads):
            kw = self.relation_att[i](k[i], etype, self.presorted)
            a.append((kw * q[i]).sum(-1) * self.relation_pri[i][etype] / self.sqrt_d)
            m.append(self.relation_msg[i](v[i], etype, self.presorted))
        return {'a': torch.stack(a, dim=1), 'm': torch.stack(m, dim=1)}


class HgtConv(nn.Module):

    def __init__(self, dim_in, dim_out, bias=False, **kwargs):
        super(HgtConv, self).__init__()
        self.model = HGTConv(dim_in, dim_out, n_heads=kwargs['num_heads'], n_etypes=kwargs['num_etypes'], n_ntypes=kwargs['num_ntypes'])

    def forward(self, g, h):
        h = self.model(g, h)
        return h


class SAGEConv(nn.Module):

    def __init__(self, dim_in, dim_out, bias=False, **kwargs):
        super(SAGEConv, self).__init__()
        self.model = dgl.nn.pytorch.SAGEConv(dim_in, dim_out, aggregator_type='mean', bias=bias)

    def forward(self, g, h):
        h = self.model(g, h)
        return h


class SimpleHGNConv(nn.Module):
    """
    The SimpleHGN convolution layer.

    Parameters
    ----------
    edge_dim: int
        the edge dimension
    num_etypes: int
        the number of the edge type
    in_dim: int
        the input dimension
    out_dim: int
        the output dimension
    num_heads: int
        the number of heads
    num_etypes: int
        the number of edge type
    feat_drop: float
        the feature drop rate
    negative_slope: float
        the negative slope used in the LeakyReLU
    residual: boolean
        if we need the residual operation
    activation: str
        the activation function
    beta: float
        the hyperparameter used in edge residual
    """

    def __init__(self, edge_dim, in_dim, out_dim, num_heads, num_etypes, feat_drop=0.0, negative_slope=0.2, residual=True, activation=F.elu, beta=0.0):
        super(SimpleHGNConv, self).__init__()
        self.edge_dim = edge_dim
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_etypes = num_etypes
        self.edge_emb = nn.Parameter(torch.empty(size=(num_etypes, edge_dim)))
        self.W = nn.Parameter(torch.FloatTensor(in_dim, out_dim * num_heads))
        self.W_r = TypedLinear(edge_dim, edge_dim * num_heads, num_etypes)
        self.a_l = nn.Parameter(torch.empty(size=(1, num_heads, out_dim)))
        self.a_r = nn.Parameter(torch.empty(size=(1, num_heads, out_dim)))
        self.a_e = nn.Parameter(torch.empty(size=(1, num_heads, edge_dim)))
        nn.init.xavier_uniform_(self.edge_emb, gain=1.414)
        nn.init.xavier_uniform_(self.W, gain=1.414)
        nn.init.xavier_uniform_(self.a_l.data, gain=1.414)
        nn.init.xavier_uniform_(self.a_r.data, gain=1.414)
        nn.init.xavier_uniform_(self.a_e.data, gain=1.414)
        self.feat_drop = nn.Dropout(feat_drop)
        self.leakyrelu = nn.LeakyReLU(negative_slope)
        self.activation = activation
        if residual:
            self.residual = nn.Linear(in_dim, out_dim * num_heads)
        else:
            self.register_buffer('residual', None)
        self.beta = beta

    def forward(self, g, h, ntype, etype, presorted=False):
        """
        The forward part of the SimpleHGNConv.

        Parameters
        ----------
        g : object
            the dgl homogeneous graph
        h: tensor
            the original features of the graph
        ntype: tensor
            the node type of the graph
        etype: tensor
            the edge type of the graph
        presorted: boolean
            if the ntype and etype are preordered, default: ``False``
            
        Returns
        -------
        tensor
            The embeddings after aggregation.
        """
        emb = self.feat_drop(h)
        emb = torch.matmul(emb, self.W).view(-1, self.num_heads, self.out_dim)
        emb[torch.isnan(emb)] = 0.0
        edge_emb = self.W_r(self.edge_emb[etype], etype, presorted).view(-1, self.num_heads, self.edge_dim)
        row = g.edges()[0]
        col = g.edges()[1]
        h_l = (self.a_l * emb).sum(dim=-1)[row]
        h_r = (self.a_r * emb).sum(dim=-1)[col]
        h_e = (self.a_e * edge_emb).sum(dim=-1)
        edge_attention = self.leakyrelu(h_l + h_r + h_e)
        edge_attention = edge_softmax(g, edge_attention)
        if 'alpha' in g.edata.keys():
            res_attn = g.edata['alpha']
            edge_attention = edge_attention * (1 - self.beta) + res_attn * self.beta
        if self.num_heads == 1:
            edge_attention = edge_attention[:, 0]
            edge_attention = edge_attention.unsqueeze(1)
        with g.local_scope():
            emb = emb.permute(0, 2, 1).contiguous()
            g.edata['alpha'] = edge_attention
            g.srcdata['emb'] = emb
            g.update_all(Fn.u_mul_e('emb', 'alpha', 'm'), Fn.sum('m', 'emb'))
            h_output = g.dstdata['emb'].view(-1, self.out_dim * self.num_heads)
        g.edata['alpha'] = edge_attention
        if g.is_block:
            h = h[:g.num_dst_nodes()]
        if self.residual:
            res = self.residual(h)
            h_output += res
        if self.activation is not None:
            h_output = self.activation(h_output)
        return h_output


class SimpleConv(nn.Module):

    def __init__(self, dim_in, dim_out, bias=False, **kwargs):
        super(SimpleConv, self).__init__()
        self.model = SimpleHGNConv(dim_in, dim_in, int(dim_out / kwargs['num_heads']), kwargs['num_heads'], kwargs['num_etypes'], beta=0.0)

    def forward(self, g, h):
        h = self.model(g, h)
        return h


homo_layer_dict = {'gcnconv': GCNConv, 'sageconv': SAGEConv, 'gatconv': GATConv, 'ginconv': GINConv, 'simpleconv': SimpleConv, 'hgtconv': HgtConv, 'appnpconv': APPNPConv}


class MPConv(nn.Module):

    def __init__(self, name, dim_in, dim_out, bias=False, **kwargs):
        super(MPConv, self).__init__()
        macro_func = kwargs['macro_func']
        meta_paths = kwargs['meta_paths']
        if macro_func == 'attention':
            macro_func = SemanticAttention(dim_out)
        elif macro_func == 'sum':
            macro_func = Aggr_sum
        elif macro_func == 'mean':
            macro_func = Aggr_mean
        elif macro_func == 'max':
            macro_func = Aggr_max
        self.model = MetapathConv(meta_paths, [homo_layer_dict[name](dim_in, dim_out, bias=bias, **kwargs) for _ in meta_paths], macro_func)
        self.meta_paths = meta_paths

    def forward(self, mp_g_list, h):
        h = self.model(mp_g_list, h)
        return h


class GeneralLayer(nn.Module):
    """General wrapper for layers"""

    def __init__(self, name, dim_in, dim_out, dropout, act=None, has_bn=True, has_l2norm=False, **kwargs):
        super(GeneralLayer, self).__init__()
        self.has_l2norm = has_l2norm
        if kwargs.get('meta_paths') is not None:
            self.layer = MPConv(name, dim_in, dim_out, bias=not has_bn, **kwargs)
        else:
            self.layer = homo_layer_dict[name](dim_in, dim_out, bias=not has_bn, **kwargs)
        layer_wrapper = []
        if has_bn:
            layer_wrapper.append(nn.BatchNorm1d(dim_out))
        if dropout > 0:
            layer_wrapper.append(nn.Dropout(p=dropout))
        if act is not None:
            layer_wrapper.append(act)
        self.post_layer = nn.Sequential(*layer_wrapper)

    def forward(self, homo_g, h):
        h = self.layer(homo_g, h)
        h = self.post_layer(h)
        if self.has_l2norm:
            h = F.normalize(h, p=2, dim=1)
        return h


def GNNLayer(gnn_type, dim_in, dim_out, dropout, act, has_bn, has_l2norm, **kwargs):
    return GeneralLayer(gnn_type, dim_in, dim_out, dropout, act, has_bn, has_l2norm, **kwargs)


class GNNModel(torch.nn.Module):

    def __init__(self, params, loader):
        super(GNNModel, self).__init__()
        self.n_layer = params.n_layer
        self.hidden_dim = params.hidden_dim
        self.attn_dim = params.attn_dim
        self.n_ent = params.n_ent
        self.n_rel = params.n_rel
        self.n_node_topk = params.n_node_topk
        self.n_edge_topk = params.n_edge_topk
        self.loader = loader
        acts = {'relu': nn.ReLU(), 'tanh': torch.tanh, 'idd': lambda x: x}
        act = acts[params.act]
        self.gnn_layers = []
        for i in range(self.n_layer):
            i_n_node_topk = self.n_node_topk if 'int' in str(type(self.n_node_topk)) else self.n_node_topk[i]
            self.gnn_layers.append(GNNLayer(self.hidden_dim, self.hidden_dim, self.attn_dim, self.n_rel, self.n_ent, n_node_topk=i_n_node_topk, n_edge_topk=self.n_edge_topk, tau=params.tau, act=act))
        self.gnn_layers = nn.ModuleList(self.gnn_layers)
        self.dropout = nn.Dropout(params.dropout)
        self.W_final = nn.Linear(self.hidden_dim, 1, bias=False)
        self.gate = nn.GRU(self.hidden_dim, self.hidden_dim)

    def updateTopkNums(self, topk_list):
        assert len(topk_list) == self.n_layer
        for idx in range(self.n_layer):
            self.gnn_layers[idx].n_node_topk = topk_list[idx]

    def fixSamplingWeight(self):

        def freeze(m):
            m.requires_grad = False
        for i in range(self.n_layer):
            self.gnn_layers[i].W_samp.apply(freeze)

    def forward(self, subs, rels, mode='train'):
        n = len(subs)
        q_sub = torch.LongTensor(subs)
        q_rel = torch.LongTensor(rels)
        h0 = torch.zeros((1, n, self.hidden_dim))
        nodes = torch.cat([torch.arange(n).unsqueeze(1), q_sub.unsqueeze(1)], 1)
        hidden = torch.zeros(n, self.hidden_dim)
        for i in range(self.n_layer):
            nodes, edges, old_nodes_new_idx = self.loader.get_neighbors(nodes.data.cpu().numpy(), n, mode=mode)
            n_node = nodes.size(0)
            hidden, nodes, sampled_nodes_idx = self.gnn_layers[i](q_sub, q_rel, hidden, edges, nodes, old_nodes_new_idx, n)
            h0 = torch.zeros(1, n_node, hidden.size(1)).index_copy_(1, old_nodes_new_idx, h0)
            h0 = h0[0, sampled_nodes_idx, :].unsqueeze(0)
            hidden = self.dropout(hidden)
            hidden, h0 = self.gate(hidden.unsqueeze(0), h0)
            hidden = hidden.squeeze(0)
        scores = self.W_final(hidden).squeeze(-1)
        scores_all = torch.zeros((n, self.loader.n_ent))
        scores_all[[nodes[:, 0], nodes[:, 1]]] = scores
        return scores_all


class HeteroEmbedLayer(nn.Module):
    """
    Embedding layer for featureless heterograph.

    Parameters
    -----------
    n_nodes_dict : dict[str, int]
        Key of dict means node type,
        value of dict means number of nodes.
    embed_size : int
        Dimension of embedding,
    activation : callable activation function/layer or None, optional
        If not None, applies an activation function to the updated node features.
        Default: ``None``.
    dropout : float, optional
        Dropout rate. Default: ``0.0``
    """

    def __init__(self, n_nodes_dict, embed_size, embed_name='embed', activation=None, dropout=0.0):
        super(HeteroEmbedLayer, self).__init__()
        self.embed_size = embed_size
        self.embed_name = embed_name
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        self.embeds = nn.ParameterDict()
        for ntype, nodes in n_nodes_dict.items():
            embed = nn.Parameter(th.FloatTensor(nodes, self.embed_size))
            nn.init.xavier_uniform_(embed, gain=nn.init.calculate_gain('relu'))
            self.embeds[ntype] = embed

    def forward(self):
        """
        Returns
        -------
        The output embeddings.
        """
        out_feature = {}
        for key, embed in self.embeds.items():
            out_feature[key] = embed
        return out_feature

    def forward_nodes(self, nodes_dict):
        """

        Parameters
        ----------
        nodes_dict : dict[str, th.Tensor]
            Key of dict means node type, value of dict means idx of nodes.

        Returns
        -------
        out_feature : dict[str, th.Tensor]
            Output feature.
        """
        out_feature = {}
        for key, nid in nodes_dict.items():
            out_feature[key] = self.embeds[key][nid]
        return out_feature


class multi_Linear(nn.Module):

    def __init__(self, linear_list, bias=False):
        super(multi_Linear, self).__init__()
        self.encoder = nn.ModuleDict({})
        for linear in linear_list:
            self.encoder[linear[0]] = nn.Linear(in_features=linear[1], out_features=linear[2], bias=bias)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.encoder.weight)

    def forward(self, name_linear, h):
        h = self.encoder[name_linear](h)
        return h


class multi_2Linear(nn.Module):

    def __init__(self, linear_list, bias=False):
        super(multi_2Linear, self).__init__()
        hidden_dim = 16
        self.hidden_layer = nn.ModuleDict({})
        self.output_layer = nn.ModuleDict({})
        for linear in linear_list:
            self.hidden_layer[linear[0]] = nn.Linear(in_features=linear[1], out_features=hidden_dim, bias=bias)
            self.output_layer[linear[0]] = nn.Linear(in_features=hidden_dim, out_features=linear[2], bias=bias)

    def forward(self, name_linear, h):
        h = F.relu(self.hidden_layer[name_linear](h))
        h = self.output_layer[name_linear](h)
        return h


class hetero_linear(nn.Module):

    def __init__(self, linear_list, bias=False):
        super(hetero_linear, self).__init__()
        self.encoder = multi_Linear(linear_list, bias)

    def forward(self, h_dict):
        h_out = {}
        for ntype, h in h_dict.items():
            h = self.encoder(ntype, h)
            h_out[ntype] = h
        return h_out


class Linear(nn.Module):

    def __init__(self, dim_in, dim_out, dropout, act=None, has_bn=True, has_l2norm=False, **kwargs):
        super(Linear, self).__init__()
        self.has_l2norm = has_l2norm
        self.layer = nn.Linear(dim_in, dim_out, bias=not has_bn, **kwargs)
        layer_wrapper = []
        if has_bn:
            layer_wrapper.append(nn.BatchNorm1d(dim_out))
        if dropout > 0:
            layer_wrapper.append(nn.Dropout(p=dropout))
        if act is not None:
            layer_wrapper.append(act)
        self.post_layer = nn.Sequential(*layer_wrapper)

    def forward(self, h):
        h = self.layer(h)
        h = self.post_layer(h)
        if self.has_l2norm:
            h = F.normalize(h, p=2, dim=1)
        return h


class MultiLinearLayer(nn.Module):

    def __init__(self, linear_list, dropout, act=None, has_bn=True, has_l2norm=False, **kwargs):
        super(MultiLinearLayer, self).__init__()
        for i in range(len(linear_list) - 1):
            d_in = linear_list[i]
            d_out = linear_list[i + 1]
            layer = Linear(d_in, d_out, dropout, act, has_bn, has_l2norm)
            self.add_module('Layer_{}'.format(i), layer)

    def forward(self, h):
        for layer in self.children():
            h = layer(h)
        return h


class BatchNorm1dNode(nn.Module):
    """General wrapper for layers"""

    def __init__(self, dim_in):
        super(BatchNorm1dNode, self).__init__()
        self.bn = nn.BatchNorm1d(dim_in)

    def forward(self, h):
        h = self.bn(h)
        return h


class HeteroGraphConv(nn.Module):
    """
    A generic module for computing convolution on heterogeneous graphs.

    The heterograph convolution applies sub-modules on their associating
    relation graphs, which reads the features from source nodes and writes the
    updated ones to destination nodes. If multiple relations have the same
    destination node types, their results are aggregated by the specified method.

    If the relation graph has no edge, the corresponding module will not be called.

    Parameters
    ----------
    mods : dict[str, nn.Module]
        Modules associated with every edge types.
    """

    def __init__(self, mods: 'dict'):
        super(HeteroGraphConv, self).__init__()
        self.mods = nn.ModuleDict(mods)

    def forward(self, graph: 'dgl.DGLHeteroGraph', input_src: 'dict', input_dst: 'dict', relation_embedding: 'dict', node_transformation_weight: 'nn.ParameterDict', relation_transformation_weight: 'nn.ParameterDict'):
        """
        call the forward function with each module.

        Parameters
        ----------
        graph: DGLHeteroGraph
            The Heterogeneous Graph.
        input_src: dict[tuple, Tensor]
            Input source node features {relation_type: features, }
        input_dst: dict[tuple, Tensor]
            Input destination node features {relation_type: features, }
        relation_embedding: dict[etype, Tensor]
            Input relation features {etype: feature}
        node_transformation_weight: nn.ParameterDict
            weights {ntype, (inp_dim, hidden_dim)}
        relation_transformation_weight: nn.ParameterDict
            weights {etype, (n_heads, 2 * hidden_dim)}

        Returns
        -------
        outputs: dict[tuple, Tensor]
            Output representations for every relation -> {(stype, etype, dtype): features}.
        """
        reverse_relation_dict = {}
        for srctype, reltype, dsttype in list(input_src.keys()):
            for stype, etype, dtype in input_src:
                if stype == dsttype and dtype == srctype and etype != reltype:
                    reverse_relation_dict[reltype] = etype
                    break
        outputs = dict()
        for stype, etype, dtype in graph.canonical_etypes:
            rel_graph = graph[stype, etype, dtype]
            if rel_graph.number_of_edges() == 0:
                continue
            dst_representation = self.mods[etype](rel_graph, (input_src[dtype, reverse_relation_dict[etype], stype], input_dst[stype, etype, dtype]), node_transformation_weight[dtype], node_transformation_weight[stype], relation_embedding[etype], relation_transformation_weight[etype])
            outputs[stype, etype, dtype] = dst_representation
        return outputs


class RelationConv(nn.Module):

    def __init__(self, name, rel_names, dim_in, dim_out, bias=False, **kwargs):
        super(RelationConv, self).__init__()
        macro_func = kwargs['macro_func']
        if macro_func == 'attention':
            macro_func = SemanticAttention(dim_out)
        self.model = HeteroGraphConv({rel: homo_layer_dict[name](dim_in, dim_out, bias=bias, **kwargs) for rel in rel_names}, aggregate=macro_func)

    def forward(self, g, h_dict):
        h_dict = self.model(g, h_dict)
        return h_dict


class HeteroGeneralLayer(nn.Module):
    """
    General wrapper for layers
    """

    def __init__(self, name, rel_names, dim_in, dim_out, dropout, act=None, has_bn=True, has_l2norm=False, **kwargs):
        super(HeteroGeneralLayer, self).__init__()
        self.has_l2norm = has_l2norm
        has_bn = has_bn
        self.layer = RelationConv(name, rel_names, dim_in, dim_out, bias=not has_bn, **kwargs)
        layer_wrapper = []
        if has_bn:
            layer_wrapper.append(nn.BatchNorm1d(dim_out))
        if dropout > 0:
            layer_wrapper.append(nn.Dropout(p=dropout))
        if act is not None:
            layer_wrapper.append(act)
        self.post_layer = nn.Sequential(*layer_wrapper)

    def forward(self, g, h_dict):
        h_dict = self.layer(g, h_dict)
        if self.has_l2norm:
            for name, batch_h in h_dict.items():
                h_dict[name] = F.normalize(self.post_layer(batch_h), p=2, dim=-1)
        return h_dict


class GeneralLinear(nn.Module):
    """
    General Linear, combined with activation, normalization(batch and L2), dropout and so on.

    Parameters
    ------------
    in_features : int
        size of each input sample, which is fed into nn.Linear
    out_features : int
        size of each output sample, which is fed into nn.Linear
    act : callable activation function/layer or None, optional
        If not None, applies an activation function to the updated node features.
        Default: ``None``.
    dropout : float, optional
        Dropout rate. Default: ``0.0``
    has_l2norm : bool
        If True, applies torch.nn.functional.normalize to the node features at last of forward(). Default: ``True``
    has_bn : bool
        If True, applies torch.nn.BatchNorm1d to the node features after applying nn.Linear.

    """

    def __init__(self, in_features, out_features, act=None, dropout=0.0, has_l2norm=True, has_bn=True, **kwargs):
        super(GeneralLinear, self).__init__()
        self.has_l2norm = has_l2norm
        has_bn = has_bn
        self.layer = nn.Linear(in_features, out_features, bias=not has_bn)
        layer_wrapper = []
        if has_bn:
            layer_wrapper.append(nn.BatchNorm1d(out_features))
        if dropout > 0:
            layer_wrapper.append(nn.Dropout(p=dropout))
        if act is not None:
            layer_wrapper.append(act)
        self.post_layer = nn.Sequential(*layer_wrapper)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.layer.weight)

    def forward(self, batch_h: 'torch.Tensor') ->torch.Tensor:
        """
        Apply Linear, BatchNorm1d, Dropout and normalize(if need).
        """
        batch_h = self.layer(batch_h)
        batch_h = self.post_layer(batch_h)
        if self.has_l2norm:
            batch_h = F.normalize(batch_h, p=2, dim=1)
        return batch_h


class HeteroLinearLayer(nn.Module):
    """
    Transform feature with nn.Linear. In general, heterogeneous feature has different dimension as input.
    Even though they may have same dimension, they may have different semantic in every dimension.
    So we use a linear layer for each node type to map all node features to a shared feature space.

    Parameters
    ----------
    linear_dict : dict
        Key of dict can be node type(node name), value of dict is a list contains input dimension and output dimension.

    Examples
    ----------

    >>> import torch as th
    >>> linear_dict = {}
    >>> linear_dict['author'] = [110, 64]
    >>> linear_dict['paper'] = [128,64]
    >>> h_dict = {}
    >>> h_dict['author'] = th.tensor(10, 110)
    >>> h_dict['paper'] = th.tensor(5, 128)
    >>> layer = HeteroLinearLayer(linear_dict)
    >>> out_dict = layer(h_dict)

    """

    def __init__(self, linear_dict, act=None, dropout=0.0, has_l2norm=True, has_bn=True, **kwargs):
        super(HeteroLinearLayer, self).__init__()
        self.layer = nn.ModuleDict({})
        for name, linear_dim in linear_dict.items():
            self.layer[name] = GeneralLinear(in_features=linear_dim[0], out_features=linear_dim[1], act=act, dropot=dropout, has_l2norm=has_l2norm, has_bn=has_bn)

    def forward(self, dict_h: 'dict') ->dict:
        """
        Parameters
        ----------
        dict_h : dict
            A dict of heterogeneous feature

        return dict_h
        """
        new_h = {}
        if isinstance(dict_h, dict):
            for name, batch_h in dict_h.items():
                new_h[name] = self.layer[name](batch_h)
        return new_h


class HeteroMLPLayer(nn.Module):
    """
    HeteroMLPLayer contains multiple GeneralLinears, different with HeteroLinearLayer.
    The latter contains only one layer.

    Parameters
    ----------
    linear_dict : dict
        Key of dict can be node type(node name), value of dict is a list contains input, hidden and output dimension.

    """

    def __init__(self, linear_dict, act=None, dropout=0.0, has_l2norm=True, has_bn=True, final_act=False, **kwargs):
        super(HeteroMLPLayer, self).__init__()
        self.layers = nn.ModuleDict({})
        for name, linear_dim in linear_dict.items():
            nn_list = []
            n_layer = len(linear_dim) - 1
            for i in range(n_layer):
                in_dim = linear_dim[i]
                out_dim = linear_dim[i + 1]
                if i == n_layer - 1:
                    if final_act:
                        layer = GeneralLinear(in_features=in_dim, out_features=out_dim, act=act, dropot=dropout, has_l2norm=has_l2norm, has_bn=has_bn)
                    else:
                        layer = GeneralLinear(in_features=in_dim, out_features=out_dim, act=None, dropot=dropout, has_l2norm=has_l2norm, has_bn=has_bn)
                else:
                    layer = GeneralLinear(in_features=in_dim, out_features=out_dim, act=act, dropot=dropout, has_l2norm=has_l2norm, has_bn=has_bn)
                nn_list.append(layer)
            self.layers[name] = nn.Sequential(*nn_list)

    def forward(self, dict_h):
        new_h = {}
        if isinstance(dict_h, dict):
            for name, batch_h in dict_h.items():
                new_h[name] = self.layers[name](batch_h)
        return new_h


class HeteroFeature(nn.Module):
    """
    This is a feature preprocessing component which is dealt with various heterogeneous feature situation.

    In general, we will face the following three situations.

        1. The dataset has not feature at all.

        2. The dataset has features in every node type.

        3. The dataset has features of a part of node types.

    To deal with that, we implement the HeteroFeature.In every situation, we can see that

        1. We will build embeddings for all node types.

        2. We will build linear layer for all node types.

        3. We will build embeddings for parts of node types and linear layer for parts of node types which have original feature.

    Parameters
    ----------
    h_dict: dict
        Input heterogeneous feature dict,
        key of dict means node type,
        value of dict means corresponding feature of the node type.
        It can be None if the dataset has no feature.
    n_nodes_dict: dict
        Key of dict means node type,
        value of dict means number of nodes.
    embed_size: int
        Dimension of embedding, and used to assign to the output dimension of Linear which transform the original feature.
    need_trans: bool, optional
        A flag to control whether to transform original feature linearly. Default is ``True``.
    act : callable activation function/layer or None, optional
        If not None, applies an activation function to the updated node features.
        Default: ``None``.

    Attributes
    -----------
    embed_dict : nn.ParameterDict
        store the embeddings

    hetero_linear : HeteroLinearLayer
        A heterogeneous linear layer to transform original feature.
    """

    def __init__(self, h_dict, n_nodes_dict, embed_size, act=None, need_trans=True, all_feats=True):
        super(HeteroFeature, self).__init__()
        self.n_nodes_dict = n_nodes_dict
        self.embed_size = embed_size
        self.h_dict = h_dict
        self.need_trans = need_trans
        self.type_node_num_sum = [0]
        self.all_type = []
        for ntype, type_num in n_nodes_dict.items():
            num_now = self.type_node_num_sum[-1]
            num_now += type_num
            self.type_node_num_sum.append(num_now)
            self.all_type.append(ntype)
        self.type_node_num_sum = torch.tensor(self.type_node_num_sum)
        linear_dict = {}
        embed_dict = {}
        for ntype, n_nodes in self.n_nodes_dict.items():
            h = h_dict.get(ntype)
            if h is None:
                if all_feats:
                    embed_dict[ntype] = n_nodes
            else:
                linear_dict[ntype] = h.shape[1]
        self.embes = HeteroEmbedding(embed_dict, embed_size)
        if need_trans:
            self.linear = HeteroLinear(linear_dict, embed_size)
        self.act = act

    def forward(self):
        out_dict = {}
        out_dict.update(self.embes.weight)
        tmp = self.linear(self.h_dict)
        if self.act:
            for x, y in tmp.items():
                tmp.update({x: self.act(y)})
        out_dict.update(tmp)
        return out_dict

    def forward_nodes(self, id_dict):
        id_tensor = None
        if torch.is_tensor(id_dict):
            device = id_dict.device
        else:
            device = id_dict.get(next(iter(id_dict))).device
        if torch.is_tensor(id_dict):
            id_tensor = id_dict
            self.type_node_num_sum = self.type_node_num_sum
            id_dict = {}
            to_pos = {}
            for i, x in enumerate(id_tensor):
                tmp = torch.where(self.type_node_num_sum <= x)[0]
                if len(tmp) > 0:
                    tmp = tmp.max()
                    now_type = self.all_type[tmp]
                    now_id = x - self.type_node_num_sum[tmp]
                    if now_type not in id_dict.keys():
                        id_dict[now_type] = []
                    id_dict[now_type].append(now_id)
                    if now_type not in to_pos.keys():
                        to_pos[now_type] = []
                    to_pos[now_type].append(i)
            for ntype in id_dict.keys():
                id_dict[ntype] = torch.tensor(id_dict[ntype], device=device)
        embed_id_dict = {}
        linear_id_dict = {}
        for entype, id in id_dict.items():
            if self.h_dict.get(entype) is None:
                embed_id_dict[entype] = id
            else:
                linear_id_dict[entype] = id
        out_dict = {}
        tmp = self.embes(embed_id_dict)
        out_dict.update(tmp)
        h_dict = {}
        for key in linear_id_dict:
            linear_id_dict[key] = linear_id_dict[key]
        for key in linear_id_dict:
            h_dict[key] = self.h_dict[key][linear_id_dict[key]]
        tmp = self.linear(h_dict)
        if self.act:
            for x, y in tmp.items():
                tmp.update({x: self.act(y)})
        for entype in linear_id_dict:
            out_dict[entype] = tmp[entype]
        if id_tensor is not None:
            out_feat = [None] * len(id_tensor)
            for ntype, feat_list in out_dict.items():
                for i, feat in enumerate(feat_list):
                    now_pos = to_pos[ntype][i]
                    out_feat[now_pos] = feat.data
            out_dict = torch.stack(out_feat, dim=0)
        return out_dict


def HGNNLayer(gnn_type, rel_names, dim_in, dim_out, dropout, act, has_bn, has_l2norm, **kwargs):
    return HeteroGeneralLayer(gnn_type, rel_names, dim_in, dim_out, dropout, act, has_bn, has_l2norm, **kwargs)


class HGNNSkipBlock(nn.Module):
    """Skip block for HGNN"""

    def __init__(self, gnn_type, rel_names, dim_in, dim_out, num_layers, stage_type, dropout, act, has_bn, has_l2norm, **kwargs):
        super(HGNNSkipBlock, self).__init__()
        self.stage_type = stage_type
        self.f = nn.ModuleList()
        if num_layers == 1:
            self.f.append(HGNNLayer(gnn_type, rel_names, dim_in, dim_out, dropout, act, has_bn, has_l2norm, **kwargs))
        else:
            self.f = []
            for i in range(num_layers - 1):
                d_in = dim_in if i == 0 else dim_out
                self.f.append(HGNNLayer(gnn_type, rel_names, d_in, dim_out, dropout, act, has_bn, has_l2norm, **kwargs))
            d_in = dim_in if num_layers == 1 else dim_out
            self.f.append(HGNNLayer(gnn_type, rel_names, d_in, dim_out, dropout, act, has_bn, has_l2norm, **kwargs))
        self.act = act
        if stage_type == 'skipsum':
            assert dim_in == dim_out, 'Sum skip must have same dim_in, dim_out'

    def forward(self, g, h):
        h_0 = h
        for layer in self.f:
            h = layer(g, h)
        out_h = {}
        for key, value in h_0.items():
            if self.stage_type == 'skipsum':
                out_h[key] = self.act(h[key] + h_0[key])
            elif self.stage_type == 'skipconcat':
                out_h[key] = self.act(torch.cat((h[key], h_0[key]), 1))
            else:
                raise ValueError('stage_type must in [skipsum, skipconcat]')
        return out_h


class HGNNStackStage(nn.Module):
    """Simple Stage that stack GNN layers"""

    def __init__(self, gnn_type, rel_names, stage_type, dim_in, dim_out, num_layers, skip_every, dropout, act, has_bn, has_l2norm, **kwargs):
        super(HGNNStackStage, self).__init__()
        for i in range(num_layers):
            d_in = dim_in if i == 0 else dim_out
            layer = HGNNLayer(gnn_type, rel_names, d_in, dim_out, dropout, act, has_bn, has_l2norm, **kwargs)
            self.add_module('layer{}'.format(i), layer)
        self.dim_out = dim_out
        self.has_l2norm = has_l2norm

    def forward(self, g, h_dict):
        for layer in self.children():
            h_dict = layer(g, h_dict)
        if self.has_l2norm:
            for name, batch_h in h_dict.items():
                h_dict[name] = F.normalize(batch_h, p=2, dim=-1)
        return h_dict


class HGNNSkipStage(nn.Module):
    """ Stage with skip connections"""

    def __init__(self, gnn_type, rel_names, stage_type, dim_in, dim_out, num_layers, skip_every, dropout, act, has_bn, has_l2norm, **kwargs):
        super(HGNNSkipStage, self).__init__()
        assert num_layers % skip_every == 0, 'cfg.gnn.skip_every must be multiples of cfg.gnn.layer_mp(excluding head layer)'
        for i in range(num_layers // skip_every):
            if stage_type == 'skipsum':
                d_in = dim_in if i == 0 else dim_out
            elif stage_type == 'skipconcat':
                d_in = dim_in if i == 0 else dim_in + i * dim_out
            block = HGNNSkipBlock(gnn_type, rel_names, d_in, dim_out, skip_every, stage_type, dropout, act, has_bn, has_l2norm, **kwargs)
            self.add_module('block{}'.format(i), block)
        if stage_type == 'skipconcat':
            self.dim_out = d_in + dim_out
        else:
            self.dim_out = dim_out
        self.has_l2norm = has_l2norm

    def forward(self, g, h_dict):
        for layer in self.children():
            h_dict = layer(g, h_dict)
        if self.has_l2norm:
            for name, batch_h in h_dict.items():
                h_dict[name] = F.normalize(batch_h, p=2, dim=-1)
        return h_dict


class GNNSkipBlock(nn.Module):
    """
    Skip block for HGNN
    """

    def __init__(self, gnn_type, dim_in, dim_out, num_layers, stage_type, dropout, act, has_bn, has_l2norm, *args, **kwargs):
        super(GNNSkipBlock, self).__init__()
        self.stage_type = stage_type
        self.f = nn.ModuleList()
        if num_layers == 1:
            self.f.append(GNNLayer(gnn_type, dim_in, dim_out, dropout, None, has_bn, has_l2norm, **kwargs))
        else:
            for i in range(num_layers - 1):
                d_in = dim_in if i == 0 else dim_out
                self.f.append(GNNLayer(gnn_type, d_in, dim_out, dropout, act, has_bn, has_l2norm, **kwargs))
            d_in = dim_in if num_layers == 1 else dim_out
            self.f.append(GNNLayer(gnn_type, d_in, dim_out, dropout, None, has_bn, has_l2norm, **kwargs))
        self.act = act
        if stage_type == 'skipsum':
            assert dim_in == dim_out, 'Sum skip must have same dim_in, dim_out'

    def forward(self, g, h):
        h_0 = h
        for layer in self.f:
            h = layer(g, h)
        if self.stage_type == 'skipsum':
            h = h + h_0
        elif self.stage_type == 'skipconcat':
            h = torch.cat((h_0, h), 1)
        else:
            raise ValueError('stage_type must in [skipsum, skipconcat]')
        h = self.act(h)
        return h


class GNNStackStage(nn.Module):
    """Simple Stage that stack GNN layers"""

    def __init__(self, gnn_type, dim_in, dim_out, num_layers, dropout, act, has_bn, has_l2norm, *args, **kwargs):
        super(GNNStackStage, self).__init__()
        for i in range(num_layers):
            d_in = dim_in if i == 0 else dim_out
            layer = GNNLayer(gnn_type, d_in, dim_out, dropout, act, has_bn, has_l2norm, **kwargs)
            self.add_module('layer{}'.format(i), layer)
        self.dim_out = dim_out
        self.has_l2norm = has_l2norm

    def forward(self, g, h):
        for layer in self.children():
            h = layer(g, h)
        if self.has_l2norm:
            h = F.normalize(h, p=2, dim=-1)
        return h


class GNNSkipStage(nn.Module):
    """ Stage with skip connections"""

    def __init__(self, gnn_type, stage_type, dim_in, dim_out, num_layers, skip_every, dropout, act, has_bn, has_l2norm, *args, **kwargs):
        super(GNNSkipStage, self).__init__()
        assert num_layers % skip_every == 0, 'cfg.gnn.skip_every must be multiples of cfg.gnn.layer_mp(excluding head layer)'
        for i in range(num_layers // skip_every):
            if stage_type == 'skipsum':
                d_in = dim_in if i == 0 else dim_out
            elif stage_type == 'skipconcat':
                d_in = dim_in if i == 0 else dim_in + i * dim_out
            block = GNNSkipBlock(gnn_type, d_in, dim_out, skip_every, stage_type, dropout, act, has_bn, has_l2norm, **kwargs)
            self.add_module('block{}'.format(i), block)
        if stage_type == 'skipconcat':
            self.dim_out = d_in + dim_out
        else:
            self.dim_out = dim_out
        self.has_l2norm = has_l2norm

    def forward(self, g, h):
        for layer in self.children():
            h = layer(g, h)
        if self.has_l2norm:
            h = F.normalize(h, p=2, dim=-1)
        return h


class CompGCNCov(nn.Module):

    def __init__(self, in_channels, out_channels, act=lambda x: x, bias=True, drop_rate=0.0, opn='corr', num_base=-1, num_rel=None, wni=False, wsi=False, use_bn=True, ltr=True):
        super(CompGCNCov, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.act = act
        self.device = None
        self.rel = None
        self.opn = opn
        self.use_bn = use_bn
        self.ltr = ltr
        self.in_w = self.get_param([in_channels, out_channels])
        self.out_w = self.get_param([in_channels, out_channels])
        self.loop_w = self.get_param([in_channels, out_channels])
        self.w_rel = self.get_param([in_channels, out_channels])
        self.loop_rel = self.get_param([1, in_channels])
        self.drop = nn.Dropout(drop_rate)
        self.bn = torch.nn.BatchNorm1d(out_channels)
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None
        if num_base > 0:
            self.rel_wt = self.get_param([num_rel * 2, num_base])
        else:
            self.rel_wt = None
        self.wni = wni
        self.wsi = wsi

    def get_param(self, shape):
        param = nn.Parameter(torch.Tensor(*shape))
        nn.init.xavier_normal_(param, gain=nn.init.calculate_gain('relu'))
        return param

    def message_func(self, edges):
        edge_type = edges.data['type']
        edge_num = edge_type.shape[0]
        edge_data = self.comp(edges.src['h'], self.rel[edge_type])
        msg = torch.cat([torch.matmul(edge_data[:edge_num // 2, :], self.in_w), torch.matmul(edge_data[edge_num // 2:, :], self.out_w)])
        msg = msg * edges.data['norm'].reshape(-1, 1)
        return {'msg': msg}

    def reduce_func(self, nodes):
        return {'h': self.drop(nodes.data['h'])}

    def comp(self, h, edge_data):

        def com_mult(a, b):
            r1, i1 = a[..., 0], a[..., 1]
            r2, i2 = b[..., 0], b[..., 1]
            return torch.stack([r1 * r2 - i1 * i2, r1 * i2 + i1 * r2], dim=-1)

        def conj(a):
            a[..., 1] = -a[..., 1]
            return a

        def ccorr(a, b):
            return torch.irfft(com_mult(conj(torch.rfft(a, 1)), torch.rfft(b, 1)), 1, signal_sizes=(a.shape[-1],))
        if self.opn == 'mult':
            return h * edge_data
        elif self.opn == 'sub':
            return h - edge_data
        elif self.opn == 'corr':
            return ccorr(h, edge_data.expand_as(h))
        else:
            raise KeyError(f'composition operator {self.opn} not recognized.')

    def forward(self, g: 'dgl.DGLGraph', x, rel_repr, edge_type, edge_norm):
        """
        :param g: dgl Graph, a graph without self-loop
        :param x: input node features, [V, in_channel]
        :param rel_repr: input relation features: 1. not using bases: [num_rel*2, in_channel]
                                                  2. using bases: [num_base, in_channel]
        :param edge_type: edge type, [E]
        :param edge_norm: edge normalization, [E]
        :return: x: output node features: [V, out_channel]
                 rel: output relation features: [num_rel*2, out_channel]
        """
        self.device = x.device
        g = g.local_var()
        g.ndata['h'] = x
        g.edata['type'] = edge_type
        g.edata['norm'] = edge_norm
        if self.rel_wt is None:
            self.rel = rel_repr
        else:
            self.rel = torch.mm(self.rel_wt, rel_repr)
        g.update_all(self.message_func, fn.sum(msg='msg', out='h'), self.reduce_func)
        if not self.wni and not self.wsi:
            x = (g.ndata.pop('h') + torch.mm(self.comp(x, self.loop_rel), self.loop_w)) / 3
        else:
            if self.wsi:
                x = g.ndata.pop('h') / 2
            if self.wni:
                x = torch.mm(self.comp(x, self.loop_rel), self.loop_w)
        if self.bias is not None:
            x = x + self.bias
        if self.use_bn:
            x = self.bn(x)
        if self.ltr:
            return self.act(x), torch.matmul(self.rel, self.w_rel)
        else:
            return self.act(x), self.rel


class ATTConv(nn.Module):
    """
    It is macro_layer of the models [HetGNN].
    It presents in the 3.3.2 Types Combination of the paper.
    
    In this framework, to make embedding dimension consistent and models tuning easy,
    we use the same dimension d for content embedding in Section 3.2,
    aggregated content embedding in Section 3.3, and output node embedding in Section 3.3.
        
    So just give one dim parameter.

    Parameters
    ----------
    dim : int
        Input feature dimension.
    ntypes : list
        Node types.

    Note:
        We don't implement multi-heads version.

        atten_w is specific to the center node type, agnostic to the neighbor node type.
    """

    def __init__(self, ntypes, dim):
        super(ATTConv, self).__init__()
        self.ntypes = ntypes
        self.activation = nn.LeakyReLU()
        self.atten_w = nn.ModuleDict({})
        for n in self.ntypes:
            self.atten_w[n] = nn.Linear(in_features=dim * 2, out_features=1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, hg, h_neigh, h_center):
        with hg.local_scope():
            if hg.is_block:
                h_dst = {k: v[:hg.number_of_dst_nodes(k)] for k, v in h_center.items()}
            else:
                h_dst = h_center
            n_types = len(self.ntypes) + 1
            outputs = {}
            for n in self.ntypes:
                h = h_dst[n]
                batch_size = h.shape[0]
                concat_h = []
                concat_emd = []
                for i in range(len(h_neigh[n])):
                    concat_h.append(th.cat((h, h_neigh[n][i]), 1))
                    concat_emd.append(h_neigh[n][i])
                concat_h.append(th.cat((h, h), 1))
                concat_emd.append(h)
                concat_h = th.hstack(concat_h).view(batch_size * n_types, self.dim * 2)
                atten_w = self.activation(self.atten_w[n](concat_h)).view(batch_size, n_types)
                atten_w = self.softmax(atten_w).view(batch_size, 1, 4)
                concat_emd = th.hstack(concat_emd).view(batch_size, n_types, self.dim)
                weight_agg_batch = th.bmm(atten_w, concat_emd).view(batch_size, self.dim)
                outputs[n] = weight_agg_batch
            return outputs


class MacroConv(nn.Module):
    """
    Parameters
    ----------
    in_feats : int
        Input feature size.
    out_feats : int
        Output feature size.
    num_heads : int
        Number of heads in Multi-Head Attention.
    dropout : float, optional
        Dropout rate, defaults: ``0``.
    """

    def __init__(self, in_feats: 'int', out_feats: 'int', num_heads: 'int', dropout: 'float'=0.0, negative_slope: 'float'=0.2):
        super(MacroConv, self).__init__()
        self._in_feats = in_feats
        self._out_feats = out_feats
        self._num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(negative_slope)

    def forward(self, graph, input_dst: 'dict', relation_features: 'dict', edge_type_transformation_weight: 'nn.ParameterDict', central_node_transformation_weight: 'nn.ParameterDict', edge_types_attention_weight: 'nn.Parameter'):
        """
        :param graph: dgl.DGLHeteroGraph
        :param input_dst: dict: {ntype: features}
        :param relation_features: dict: {(stype, etype, dtype): features}
        :param edge_type_transformation_weight: ParameterDict {etype: (n_heads * hidden_dim, n_heads * hidden_dim)}
        :param central_node_transformation_weight:  ParameterDict {ntype: (input_central_node_dim, n_heads * hidden_dim)}
        :param edge_types_attention_weight: Parameter (n_heads, 2 * hidden_dim)
        :return: output_features: dict, {"type": features}
        """
        output_features = {}
        for ntype in input_dst:
            if graph.number_of_dst_nodes(ntype) != 0:
                central_node_feature = input_dst[ntype]
                central_node_feature = torch.matmul(central_node_feature, central_node_transformation_weight[ntype]).view(-1, self._num_heads, self._out_feats)
                types_features = []
                for relation_tuple in relation_features:
                    stype, etype, dtype = relation_tuple
                    if dtype == ntype:
                        types_features.append(torch.matmul(relation_features[relation_tuple], edge_type_transformation_weight[etype]))
                types_features = torch.stack(types_features, dim=0)
                if types_features.shape[0] == 1:
                    output_features[ntype] = types_features.squeeze(dim=0)
                else:
                    types_features = types_features.view(types_features.shape[0], -1, self._num_heads, self._out_feats)
                    stacked_central_features = torch.stack([central_node_feature for _ in range(types_features.shape[0])], dim=0)
                    concat_features = torch.cat((stacked_central_features, types_features), dim=-1)
                    attention_scores = (edge_types_attention_weight * concat_features).sum(dim=-1, keepdim=True)
                    attention_scores = self.leaky_relu(attention_scores)
                    attention_scores = F.softmax(attention_scores, dim=0)
                    output_feature = (attention_scores * types_features).sum(dim=0)
                    output_feature = self.dropout(output_feature)
                    output_feature = output_feature.reshape(-1, self._num_heads * self._out_feats)
                    output_features[ntype] = output_feature
        return output_features


def com_mult(a, b):
    r1, i1 = a[..., 0], a[..., 1]
    r2, i2 = b[..., 0], b[..., 1]
    return th.stack([r1 * r2 - i1 * i2, r1 * i2 + i1 * r2], dim=-1)


def conj(a):
    a[..., 1] = -a[..., 1]
    return a


def ccorr(a, b):
    """
    Compute circular correlation of two tensors.
    Parameters
    ----------
    a: Tensor, 1D or 2D
    b: Tensor, 1D or 2D
    Notes
    -----
    Input a and b should have the same dimensions. And this operation supports broadcasting.
    Returns
    -------
    Tensor, having the same dimension as the input a.
    """
    try:
        from torch import irfft
        from torch import rfft
    except ImportError:
        from torch.fft import irfft2
        from torch.fft import rfft2

        def rfft(x, d):
            t = rfft2(x, dim=-d)
            return th.stack((t.real, t.imag), -1)

        def irfft(x, d, signal_sizes):
            return irfft2(th.complex(x[:, :, 0], x[:, :, 1]), s=signal_sizes, dim=-d)
    return irfft(com_mult(conj(rfft(a, 1)), rfft(b, 1)), 1, signal_sizes=(a.shape[-1],))


class CompConv(nn.Module):
    """
    Composition-based convolution was introduced in `Composition-based Multi-Relational Graph Convolutional Networks
    <https://arxiv.org/abs/1911.03082>`__
    and mathematically is defined as follows:

    Parameters
    ----------
    comp_fn : str, one of 'sub', 'mul', 'ccorr'
    """

    def __init__(self, comp_fn, norm='right', linear=False, in_feats=None, out_feats=None, bias=False, activation=None, _allow_zero_in_degree=False):
        super(CompConv, self).__init__()
        if norm not in ('none', 'both', 'right'):
            raise DGLError('Invalid norm value. Must be either "none", "both" or "right". But got "{}".'.format(norm))
        self._norm = norm
        self.comp_fn = comp_fn
        if self.comp_fn == 'sub':
            self.aggregate = fn.u_sub_e('h', '_edge_weight', out='comp_h')
        elif self.comp_fn == 'mul':
            self.aggregate = fn.u_mul_e('h', '_edge_weight', out='comp_h')
        elif self.comp_fn == 'ccorr':
            self.aggregate = lambda edges: {'comp_h': ccorr(edges.src['h'], edges.data['_edge_weight'])}
        else:
            raise Exception('Only supports sub, mul, and ccorr')
        if linear:
            if in_feats is None or out_feats is None:
                raise DGLError('linear is True, so you must specify the in/out feats')
            else:
                self.Linear = nn.Parameter(th.Tensor(in_feats, out_feats))
        else:
            self.register_parameter('Linear', None)
        if bias:
            self.bias = nn.Parameter(th.Tensor(out_feats))
        else:
            self.register_parameter('bias', None)
        self._activation = activation
        self._allow_zero_in_degree = _allow_zero_in_degree

    def forward(self, graph, feat, h_e, Linear=None):
        """
        Compute Composition-based  convolution.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : torch.Tensor or pair of torch.Tensor
            If a torch.Tensor is given, it represents the input feature of shape
            :math:`(N, D_{in})`
            where :math:`D_{in}` is size of input feature, :math:`N` is the number of nodes.
            If a pair of torch.Tensor is given, which is the case for bipartite graph, the pair
            must contain two tensors of shape :math:`(N_{in}, D_{in_{src}})` and
            :math:`(N_{out}, D_{in_{dst}})`.
        Linear : a Linear nn.Module, optional
            Optional external weight tensor.
        h_e : torch.Tensor
            :math:`(1, D_{in})`
            means the edge type feature.

        Returns
        -------
        torch.Tensor
            The output feature

        Raises
        ------
        DGLError
            Case 1:
            If there are 0-in-degree nodes in the input graph, it will raise DGLError
            since no message will be passed to those nodes. This will cause invalid output.
            The error can be ignored by setting ``allow_zero_in_degree`` parameter to ``True``.

            Case 2:
            External weight is provided while at the same time the module
            has defined its own weight parameter.

        Note
        ----
        The h_e is a tensor of size `(1, D_{in})`
        
        * Input shape: :math:`(N, *, \\text{in_feats})` where * means any number of additional
          dimensions, :math:`N` is the number of nodes.
        * Output shape: :math:`(N, *, \\text{out_feats})` where all but the last dimension are
          the same shape as the input.
        * Linear shape: :math:`(\\text{in_feats}, \\text{out_feats})`.
                """
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise DGLError('There are 0-in-degree nodes in the graph, output for those nodes will be invalid. This is harmful for some applications, causing silent performance regression. Adding self-loop on the input graph by calling `g = dgl.add_self_loop(g)` will resolve the issue. Setting ``allow_zero_in_degree`` to be `True` when constructing this module will suppress the check and let the code run.')
            graph.edata['_edge_weight'] = h_e.expand(graph.num_edges(), -1)
            feat_src, feat_dst = expand_as_pair(feat, graph)
            graph.srcdata['h'] = feat_src
            graph.apply_edges(self.aggregate)
            if Linear is not None:
                if self.Linear is not None:
                    raise DGLError('External Linear is provided while at the same time the module has defined its own Linear module. Please create the module with flag Linear=False.')
            else:
                Linear = self.Linear
            graph.edata['comp_h'] = Linear(graph.edata['comp_h'])
            graph.update_all(fn.copy_e('comp_h', 'm'), fn.sum('m', 'h'))
            rst = graph.dstdata['h']
            if self._norm != 'none':
                degs = graph.in_degrees().float().clamp(min=1)
                if self._norm == 'both':
                    norm = th.pow(degs, -0.5)
                else:
                    norm = 1.0 / degs
                shp = norm.shape + (1,) * (feat_dst.dim() - 1)
                norm = th.reshape(norm, shp)
                rst = rst * norm
            if self.bias is not None:
                rst = rst + self.bias
            if self._activation is not None:
                rst = self._activation(rst)
            return rst


class AttConv(nn.Module):
    """
    Attention-based convolution was introduced in `Hybrid Micro/Macro Level Convolution for Heterogeneous Graph Learning
    <https://arxiv.org/abs/>`__
    and mathematically is defined as follows:

    """

    def __init__(self, in_feats: 'tuple', out_feats: 'int', num_heads: 'int', dropout: 'float'=0.0, negative_slope: 'float'=0.2):
        """
        Parameters
        ----------
        in_feats : pair of ints
            Input feature size.
        out_feats : int
            Output feature size.
        num_heads : int
            Number of heads in Multi-Head Attention.
        dropout : float, optional
            Dropout rate, defaults: 0.
        negative_slope : float, optional
            Negative slope rate, defaults: 0.2.
        """
        super(AttConv, self).__init__()
        self._in_src_feats, self._in_dst_feats = in_feats[0], in_feats[1]
        self._out_feats = out_feats
        self._num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(negative_slope)

    def forward(self, graph: 'dgl.DGLHeteroGraph', feat: 'tuple', dst_node_transformation_weight: 'nn.Parameter', src_node_transformation_weight: 'nn.Parameter', src_nodes_attention_weight: 'nn.Parameter'):
        """Compute graph attention network layer.
        
        Parameters
        ----------
        graph: 
            specific relational DGLHeteroGraph
        feat: pair of torch.Tensor
            The pair contains two tensors of shape (N_{in}, D_{in_{src}})` and (N_{out}, D_{in_{dst}}).
        dst_node_transformation_weight: 
            Parameter (input_dst_dim, n_heads * hidden_dim)
        src_node_transformation_weight: 
            Parameter (input_src_dim, n_heads * hidden_dim)
        src_nodes_attention_weight: 
            Parameter (n_heads, 2 * hidden_dim)

        Returns
        -------
        torch.Tensor, shape (N, H, D_out)` where H is the number of heads, and D_out is size of output feature.
        """
        with graph.local_scope():
            feat_src = self.dropout(feat[0])
            feat_dst = self.dropout(feat[1])
            feat_src = torch.matmul(feat_src, src_node_transformation_weight).view(-1, self._num_heads, self._out_feats)
            feat_dst = torch.matmul(feat_dst, dst_node_transformation_weight).view(-1, self._num_heads, self._out_feats)
            e_dst = (feat_dst * src_nodes_attention_weight[:, :self._out_feats]).sum(dim=-1, keepdim=True)
            e_src = (feat_src * src_nodes_attention_weight[:, self._out_feats:]).sum(dim=-1, keepdim=True)
            graph.srcdata.update({'ft': feat_src, 'e_src': e_src})
            graph.dstdata.update({'e_dst': e_dst})
            graph.apply_edges(fn.u_add_v('e_src', 'e_dst', 'e'))
            e = self.leaky_relu(graph.edata.pop('e'))
            graph.edata['a'] = edge_softmax(graph, e)
            graph.update_all(fn.u_mul_e('ft', 'a', 'msg'), fn.sum('msg', 'ft'))
            dst_features = graph.dstdata.pop('ft').reshape(-1, self._num_heads * self._out_feats)
            dst_features = F.relu(dst_features)
        return dst_features


class LSTMConv(nn.Module):
    """
    Aggregate the neighbors with LSTM
    """

    def __init__(self, dim):
        super(LSTMConv, self).__init__()
        self.lstm = nn.LSTM(dim, int(dim / 2), 1, batch_first=True, bidirectional=True)
        self.reset_parameters()

    def reset_parameters(self):
        """
        Reinitialize learnable parameters.

        Note
        ----
        The LSTM module is using xavier initialization method for its weights.
        """
        self.lstm.reset_parameters()

    def _lstm_reducer(self, nodes):
        m = nodes.mailbox['m']
        batch_size = m.shape[0]
        all_state, last_state = self.lstm(m)
        return {'neigh': th.mean(all_state, 1)}

    def forward(self, g, inputs):
        with g.local_scope():
            if isinstance(inputs, tuple) or g.is_block:
                if isinstance(inputs, tuple):
                    src_inputs, dst_inputs = inputs
                else:
                    src_inputs = inputs
                    dst_inputs = {k: v[:g.number_of_dst_nodes(k)] for k, v in inputs.items()}
                g.srcdata['h'] = src_inputs
                g.update_all(fn.copy_u('h', 'm'), self._lstm_reducer)
                h_neigh = g.dstdata['neigh']
            else:
                g.srcdata['h'] = inputs
                g.update_all(fn.copy_u('h', 'm'), self._lstm_reducer)
                h_neigh = g.dstdata['neigh']
            return h_neigh


_TORCH_HAS_SEARCHSORTED = getattr(th, 'searchsorted', None)


def _searchsorted(sorted_sequence, values):
    if _TORCH_HAS_SEARCHSORTED:
        return th.searchsorted(sorted_sequence, values)
    else:
        device = values.device
        return th.from_numpy(np.searchsorted(sorted_sequence.cpu().numpy(), values.cpu().numpy()))


class RelGraphConv(nn.Module):
    """Relational graph convolution layer.

    Relational graph convolution is introduced in "`Modeling Relational Data with Graph
    Convolutional Networks <https://arxiv.org/abs/1703.06103>`__"
    and can be described in DGL as below:

    .. math::

       h_i^{(l+1)} = \\sigma(\\sum_{r\\in\\mathcal{R}}
       \\sum_{j\\in\\mathcal{N}^r(i)}e_{j,i}W_r^{(l)}h_j^{(l)}+W_0^{(l)}h_i^{(l)})

    where :math:`\\mathcal{N}^r(i)` is the neighbor set of node :math:`i` w.r.t. relation
    :math:`r`. :math:`e_{j,i}` is the normalizer. :math:`\\sigma` is an activation
    function. :math:`W_0` is the self-loop weight.

    The basis regularization decomposes :math:`W_r` by:

    .. math::

       W_r^{(l)} = \\sum_{b=1}^B a_{rb}^{(l)}V_b^{(l)}

    where :math:`B` is the number of bases, :math:`V_b^{(l)}` are linearly combined
    with coefficients :math:`a_{rb}^{(l)}`.

    The block-diagonal-decomposition regularization decomposes :math:`W_r` into :math:`B`
    number of block diagonal matrices. We refer :math:`B` as the number of bases.

    The block regularization decomposes :math:`W_r` by:

    .. math::

       W_r^{(l)} = \\oplus_{b=1}^B Q_{rb}^{(l)}

    where :math:`B` is the number of bases, :math:`Q_{rb}^{(l)}` are block
    bases with shape :math:`R^{(d^{(l+1)}/B)*(d^{l}/B)}`.

    Parameters
    ----------
    in_feat : int
        Input feature size; i.e, the number of dimensions of :math:`h_j^{(l)}`.
    out_feat : int
        Output feature size; i.e., the number of dimensions of :math:`h_i^{(l+1)}`.
    num_rels : int
        Number of relations. .
    regularizer : str
        Which weight regularizer to use "basis" or "bdd".
        "basis" is short for basis-diagonal-decomposition.
        "bdd" is short for block-diagonal-decomposition.
    num_bases : int, optional
        Number of bases. If is none, use number of relations. Default: ``None``.
    bias : bool, optional
        True if bias is added. Default: ``True``.
    activation : callable, optional
        Activation function. Default: ``None``.
    self_loop : bool, optional
        True to include self loop message. Default: ``True``.
    low_mem : bool, optional
        True to use low memory implementation of relation message passing function. Default: False.
        This option trades speed with memory consumption, and will slowdown the forward/backward.
        Turn it on when you encounter OOM problem during training or evaluation. Default: ``False``.
    dropout : float, optional
        Dropout rate. Default: ``0.0``
    layer_norm: float, optional
        Add layer norm. Default: ``False``

    Examples
    --------
    >>> import dgl
    >>> import numpy as np
    >>> import torch as th
    >>> from dgl.nn import RelGraphConv
    >>>
    >>> g = dgl.graph(([0,1,2,3,2,5], [1,2,3,4,0,3]))
    >>> feat = th.ones(6, 10)
    >>> conv = RelGraphConv(10, 2, 3, regularizer='basis', num_bases=2)
    >>> conv.weight.shape
    torch.Size([2, 10, 2])
    >>> etype = th.tensor(np.array([0,1,2,0,1,2]).astype(np.int64))
    >>> res = conv(g, feat, etype)
    >>> res
    tensor([[ 0.3996, -2.3303],
            [-0.4323, -0.1440],
            [ 0.3996, -2.3303],
            [ 2.1046, -2.8654],
            [-0.4323, -0.1440],
            [-0.1309, -1.0000]], grad_fn=<AddBackward0>)

    >>> # One-hot input
    >>> one_hot_feat = th.tensor(np.array([0,1,2,3,4,5]).astype(np.int64))
    >>> res = conv(g, one_hot_feat, etype)
    >>> res
    tensor([[ 0.5925,  0.0985],
            [-0.3953,  0.8408],
            [-0.9819,  0.5284],
            [-1.0085, -0.1721],
            [ 0.5962,  1.2002],
            [ 0.0365, -0.3532]], grad_fn=<AddBackward0>)
    """

    def __init__(self, in_feat, out_feat, num_rels, regularizer='basis', num_bases=None, bias=True, activation=None, self_loop=True, low_mem=False, dropout=0.0, layer_norm=False, wni=False):
        super(RelGraphConv, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.num_rels = num_rels
        self.regularizer = regularizer
        self.num_bases = num_bases
        if self.num_bases is None or self.num_bases > self.num_rels or self.num_bases <= 0:
            self.num_bases = self.num_rels
        self.bias = bias
        self.activation = activation
        self.self_loop = self_loop
        self.low_mem = low_mem
        self.layer_norm = layer_norm
        self.wni = wni
        if regularizer == 'basis':
            self.weight = nn.Parameter(th.Tensor(self.num_bases, self.in_feat, self.out_feat))
            if self.num_bases < self.num_rels:
                self.w_comp = nn.Parameter(th.Tensor(self.num_rels, self.num_bases))
            nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))
            if self.num_bases < self.num_rels:
                nn.init.xavier_uniform_(self.w_comp, gain=nn.init.calculate_gain('relu'))
            self.message_func = self.basis_message_func
        elif regularizer == 'bdd':
            if in_feat % self.num_bases != 0 or out_feat % self.num_bases != 0:
                raise ValueError('Feature size must be a multiplier of num_bases (%d).' % self.num_bases)
            self.submat_in = in_feat // self.num_bases
            self.submat_out = out_feat // self.num_bases
            self.weight = nn.Parameter(th.Tensor(self.num_rels, self.num_bases * self.submat_in * self.submat_out))
            nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))
            self.message_func = self.bdd_message_func
        else:
            raise ValueError("Regularizer must be either 'basis' or 'bdd'")
        if self.bias:
            self.h_bias = nn.Parameter(th.Tensor(out_feat))
            nn.init.zeros_(self.h_bias)
        if self.layer_norm:
            self.layer_norm_weight = nn.LayerNorm(out_feat, elementwise_affine=True)
        if self.self_loop:
            self.loop_weight = nn.Parameter(th.Tensor(in_feat, out_feat))
            nn.init.xavier_uniform_(self.loop_weight, gain=nn.init.calculate_gain('relu'))
        self.dropout = nn.Dropout(dropout)

    def basis_message_func(self, edges, etypes):
        """Message function for basis regularizer.

        Parameters
        ----------
        edges : dgl.EdgeBatch
            Input to DGL message UDF.
        etypes : torch.Tensor or list[int]
            Edge type data. Could be either:

                * An :math:`(|E|,)` dense tensor. Each element corresponds to the edge's type ID.
                  Preferred format if ``lowmem == False``.
                * An integer list. The i^th element is the number of edges of the i^th type.
                  This requires the input graph to store edges sorted by their type IDs.
                  Preferred format if ``lowmem == True``.
        """
        if self.num_bases < self.num_rels:
            weight = self.weight.view(self.num_bases, self.in_feat * self.out_feat)
            weight = th.matmul(self.w_comp, weight).view(self.num_rels, self.in_feat, self.out_feat)
        else:
            weight = self.weight
        h = edges.src['h']
        device = h.device
        if h.dtype == th.int64 and h.ndim == 1:
            if isinstance(etypes, list):
                etypes = th.repeat_interleave(th.arange(len(etypes), device=device), th.tensor(etypes, device=device))
            idim = weight.shape[1]
            weight = weight.view(-1, weight.shape[2])
            flatidx = etypes * idim + h
            msg = weight.index_select(0, flatidx)
        elif self.low_mem:
            assert isinstance(etypes, list)
            h_t = th.split(h, etypes)
            msg = []
            for etype in range(self.num_rels):
                if h_t[etype].shape[0] == 0:
                    continue
                msg.append(th.matmul(h_t[etype], weight[etype]))
            msg = th.cat(msg)
        else:
            if isinstance(etypes, list):
                etypes = th.repeat_interleave(th.arange(len(etypes), device=device), th.tensor(etypes, device=device))
            weight = weight.index_select(0, etypes)
            msg = th.bmm(h.unsqueeze(1), weight).squeeze(1)
        if 'norm' in edges.data:
            msg = msg * edges.data['norm']
        return {'msg': msg}

    def bdd_message_func(self, edges, etypes):
        """Message function for block-diagonal-decomposition regularizer.

        Parameters
        ----------
        edges : dgl.EdgeBatch
            Input to DGL message UDF.
        etypes : torch.Tensor or list[int]
            Edge type data. Could be either:

                * An :math:`(|E|,)` dense tensor. Each element corresponds to the edge's type ID.
                  Preferred format if ``lowmem == False``.
                * An integer list. The i^th element is the number of edges of the i^th type.
                  This requires the input graph to store edges sorted by their type IDs.
                  Preferred format if ``lowmem == True``.
        """
        h = edges.src['h']
        device = h.device
        if h.dtype == th.int64 and h.ndim == 1:
            raise TypeError('Block decomposition does not allow integer ID feature.')
        if self.low_mem:
            assert isinstance(etypes, list)
            h_t = th.split(h, etypes)
            msg = []
            for etype in range(self.num_rels):
                if h_t[etype].shape[0] == 0:
                    continue
                tmp_w = self.weight[etype].view(self.num_bases, self.submat_in, self.submat_out)
                tmp_h = h_t[etype].view(-1, self.num_bases, self.submat_in)
                msg.append(th.einsum('abc,bcd->abd', tmp_h, tmp_w).reshape(-1, self.out_feat))
            msg = th.cat(msg)
        else:
            if isinstance(etypes, list):
                etypes = th.repeat_interleave(th.arange(len(etypes), device=device), th.tensor(etypes, device=device))
            weight = self.weight.index_select(0, etypes).view(-1, self.submat_in, self.submat_out)
            node = h.view(-1, 1, self.submat_in)
            msg = th.bmm(node, weight).view(-1, self.out_feat)
        if 'norm' in edges.data:
            msg = msg * edges.data['norm']
        return {'msg': msg}

    def forward(self, g, feat, etypes, norm=None):
        """Forward computation.

        Parameters
        ----------
        g : DGLGraph
            The graph.
        feat : torch.Tensor
            Input node features. Could be either

                * :math:`(|V|, D)` dense tensor
                * :math:`(|V|,)` int64 vector, representing the categorical values of each
                  node. It then treat the input feature as an one-hot encoding feature.
        etypes : torch.Tensor or list[int]
            Edge type data. Could be either

                * An :math:`(|E|,)` dense tensor. Each element corresponds to the edge's type ID.
                  Preferred format if ``lowmem == False``.
                * An integer list. The i^th element is the number of edges of the i^th type.
                  This requires the input graph to store edges sorted by their type IDs.
                  Preferred format if ``lowmem == True``.
        norm : torch.Tensor, optional
            Edge normalizer. Could be either

                * An :math:`(|E|, 1)` tensor storing the normalizer on each edge.

        Returns
        -------
        torch.Tensor
            New node features.

        Notes
        -----
        Under the ``low_mem`` mode, DGL will sort the graph based on the edge types
        and compute message passing one type at a time. DGL recommends sorts the
        graph beforehand (and cache it if possible) and provides the integer list
        format to the ``etypes`` argument. Use DGL's :func:`~dgl.to_homogeneous` API
        to get a sorted homogeneous graph from a heterogeneous graph. Pass ``return_count=True``
        to it to get the ``etypes`` in integer list.
        """
        if isinstance(etypes, th.Tensor):
            if len(etypes) != g.num_edges():
                raise DGLError('"etypes" tensor must have length equal to the number of edges in the graph. But got {} and {}.'.format(len(etypes), g.num_edges()))
            if self.low_mem and not (feat.dtype == th.int64 and feat.ndim == 1):
                sorted_etypes, index = th.sort(etypes)
                g = edge_subgraph(g, index, relabel_nodes=False)
                pos = _searchsorted(sorted_etypes, th.arange(self.num_rels, device=g.device))
                num = th.tensor([len(etypes)], device=g.device)
                etypes = (th.cat([pos[1:], num]) - pos).tolist()
                if norm is not None:
                    norm = norm[index]
        with g.local_scope():
            g.srcdata['h'] = feat
            if norm is not None:
                g.edata['norm'] = norm
            if self.self_loop:
                loop_message = utils.matmul_maybe_select(feat[:g.number_of_dst_nodes()], self.loop_weight)
            if not self.wni:
                g.update_all(functools.partial(self.message_func, etypes=etypes), fn.sum(msg='msg', out='h'))
                node_repr = g.dstdata['h']
                if self.layer_norm:
                    node_repr = self.layer_norm_weight(node_repr)
                if self.bias:
                    node_repr = node_repr + self.h_bias
            else:
                node_repr = 0
            if self.self_loop:
                node_repr = node_repr + loop_message
            if self.activation:
                node_repr = self.activation(node_repr)
            node_repr = self.dropout(node_repr)
            return node_repr


class Attention(nn.Module):

    def __init__(self, embedding_dims):
        super(Attention, self).__init__()
        self.embed_dim = embedding_dims
        self.att1 = nn.Linear(self.embed_dim * 2, self.embed_dim)
        self.att2 = nn.Linear(self.embed_dim, self.embed_dim)
        self.att3 = nn.Linear(self.embed_dim, 1)

    def forward(self, node, neighs, num_neighs):
        node = node.repeat(num_neighs, 1)
        x = torch.cat((neighs, node), 1)
        x = F.relu(self.att1(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.att2(x))
        x = F.dropout(x, training=self.training)
        x = self.att3(x)
        att = F.softmax(x, dim=0)
        return att


def init_drop(dropout):
    if dropout > 0:
        return nn.Dropout(dropout)
    else:
        return lambda x: x


class SelfAttention(nn.Module):

    def __init__(self, hidden_dim, attn_drop, txt):
        """
        This part is used to calculate type-level attention and semantic-level attention, and utilize them to generate :math:`z^{sc}` and :math:`z^{mp}`.

        .. math::
           w_{n}&=\\frac{1}{|V|}\\sum\\limits_{i\\in V} \\textbf{a}^\\top \\cdot \\tanh\\left(\\textbf{W}h_i^{n}+\\textbf{b}\\right) \\\\
           \\beta_{n}&=\\frac{\\exp\\left(w_{n}\\right)}{\\sum_{i=1}^M\\exp\\left(w_{i}\\right)} \\\\
           z &= \\sum_{n=1}^M \\beta_{n}\\cdot h^{n}

        Parameters
        ----------
        txt : str
            A str to identify view, MP or SC

        Returns
        -------
        z : matrix
            The fused embedding matrix

        """
        super(SelfAttention, self).__init__()
        self.fc = nn.Linear(hidden_dim, hidden_dim, bias=True)
        nn.init.xavier_normal_(self.fc.weight, gain=1.414)
        self.tanh = nn.Tanh()
        self.att = nn.Parameter(torch.empty(size=(1, hidden_dim)), requires_grad=True)
        nn.init.xavier_normal_(self.att.data, gain=1.414)
        self.softmax = nn.Softmax(dim=0)
        self.attn_drop = init_drop(attn_drop)
        self.txt = txt

    def forward(self, embeds):
        beta = []
        attn_curr = self.attn_drop(self.att)
        for embed in embeds:
            sp = self.tanh(self.fc(embed)).mean(dim=0)
            beta.append(attn_curr.matmul(sp.t()))
        beta = torch.cat(beta, dim=-1).view(-1)
        beta = self.softmax(beta)
        None
        z = 0
        for i in range(len(embeds)):
            z += embeds[i] * beta[i]
        return z


class _ATT_HGCN(nn.Module):

    def __init__(self):
        pass

    def forward(self):
        pass


class HeteAggregateLayer(nn.Module):

    def __init__(self, curr_k, nb_list, in_layer_shape, out_shape, type_fusion, type_att_size):
        super(HeteAggregateLayer, self).__init__()
        self.nb_list = nb_list
        self.curr_k = curr_k
        self.type_fusion = type_fusion
        self.W_rel = nn.ParameterDict()
        for k in nb_list:
            try:
                self.W_rel[k] = nn.Parameter(torch.FloatTensor(in_layer_shape[k], out_shape))
            except KeyError as ke:
                self.W_rel[k] = nn.Parameter(torch.FloatTensor(in_layer_shape[self.curr_k], out_shape))
            finally:
                nn.init.xavier_uniform_(self.W_rel[k].data, gain=1.414)
        self.w_self = nn.Parameter(torch.FloatTensor(in_layer_shape[curr_k], out_shape))
        nn.init.xavier_uniform_(self.w_self.data, gain=1.414)
        self.bias = nn.Parameter(torch.FloatTensor(1, out_shape))
        nn.init.xavier_uniform_(self.bias.data, gain=1.414)
        if type_fusion == 'att':
            self.w_query = nn.Parameter(torch.FloatTensor(out_shape, type_att_size))
            nn.init.xavier_uniform_(self.w_query.data, gain=1.414)
            self.w_keys = nn.Parameter(torch.FloatTensor(out_shape, type_att_size))
            nn.init.xavier_uniform_(self.w_keys.data, gain=1.414)
            self.w_att = nn.Parameter(torch.FloatTensor(2 * type_att_size, 1))
            nn.init.xavier_uniform_(self.w_att.data, gain=1.414)

    def forward(self, x_dict, adj_dict):
        attention_curr_k = 0
        self_ft = torch.mm(x_dict[self.curr_k], self.w_self)
        nb_ft_list = [self_ft]
        nb_name = [self.curr_k + '_self']
        for k in self.nb_list:
            try:
                nb_ft = torch.mm(x_dict[k], self.W_rel[k])
            except KeyError as ke:
                nb_ft = torch.mm(x_dict[self.curr_k], self.W_rel[k])
            finally:
                nb_ft = torch.spmm(adj_dict[k], nb_ft)
                nb_ft_list.append(nb_ft)
                nb_name.append(k)
        if self.type_fusion == 'mean':
            agg_nb_ft = torch.cat([nb_ft.unsqueeze(1) for nb_ft in nb_ft_list], 1).mean(1)
            attention = []
        elif self.type_fusion == 'att':
            att_query = torch.mm(self_ft, self.w_query).repeat(len(nb_ft_list), 1)
            att_keys = torch.mm(torch.cat(nb_ft_list, 0), self.w_keys)
            att_input = torch.cat([att_keys, att_query], 1)
            att_input = F.dropout(att_input, 0.5, training=self.training)
            e = F.elu(torch.matmul(att_input, self.w_att))
            attention = F.softmax(e.view(len(nb_ft_list), -1).transpose(0, 1), dim=1)
            agg_nb_ft = torch.cat([nb_ft.unsqueeze(1) for nb_ft in nb_ft_list], 1).mul(attention.unsqueeze(-1)).sum(1)
        output = agg_nb_ft + self.bias
        return output, attention


class HeteGCNLayer(nn.Module):

    def __init__(self, net_schema, in_layer_shape, out_layer_shape, type_fusion, type_att_size):
        super(HeteGCNLayer, self).__init__()
        self.net_schema = net_schema
        self.in_layer_shape = in_layer_shape
        self.out_layer_shape = out_layer_shape
        self.hete_agg = nn.ModuleDict()
        for k in net_schema:
            self.hete_agg[k] = HeteAggregateLayer(k, net_schema[k], in_layer_shape, out_layer_shape[k], type_fusion, type_att_size)

    def forward(self, x_dict, adj_dict):
        attention_dict = {}
        ret_x_dict = {}
        for k in self.hete_agg.keys():
            ret_x_dict[k], attention_dict[k] = self.hete_agg[k](x_dict, adj_dict[k])
        return ret_x_dict, attention_dict


class GraphConvolution(torch.nn.Module):

    def __init__(self, in_features, out_features, num_relations, bias=True, wsi=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.num_relations = num_relations
        self.alpha = torch.nn.Embedding(num_relations + 1, 1, padding_idx=0)
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.wsi = wsi

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, g, all_edge_type, input):
        with g.local_scope():
            feats = torch.mm(input, self.weight)
            g.srcdata['ft'] = feats
            if not self.wsi:
                train_edge_num = int((all_edge_type.shape[0] - input.shape[0]) / 2)
                transpose_all_edge_type = torch.cat((all_edge_type[train_edge_num:train_edge_num * 2], all_edge_type[:train_edge_num], all_edge_type[-input.shape[0]:]))
            else:
                train_edge_num = int(all_edge_type.shape[0])
                transpose_all_edge_type = torch.cat((all_edge_type[train_edge_num:train_edge_num * 2], all_edge_type[:train_edge_num]))
            alp = self.alpha(all_edge_type) + self.alpha(transpose_all_edge_type)
            g.edata['a'] = alp
            g.update_all(fn.u_mul_e('ft', 'a', 'm'), fn.sum('m', 'ft'))
            output = g.dstdata['ft']
            if self.bias is not None:
                return output + self.bias
            else:
                return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GCN(nn.Module):
    """
    The downstream GCN model.
    """

    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        """
        Parameters
        ----------
        x : tensor
            The feature matrix.
        adj : tensor
            The adjacent matrix.
        """
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return x


def coototensor(A):
    """
    Convert a coo_matrix to a torch sparse tensor
    """
    values = A.data
    indices = np.vstack((A.row, A.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = A.shape
    return torch.sparse.FloatTensor(i, v, torch.Size(shape))


def adj_matrix_weight_merge(A, adj_weight):
    """
    Multiplex Relation Aggregation
    """
    N = A[0][0].shape[0]
    temp = coo_matrix((N, N))
    temp = coototensor(temp)
    a = coototensor(A[0][0].tocoo())
    b = coototensor(A[0][1].tocoo())
    c = coototensor(A[0][2].tocoo())
    d = coototensor(A[0][3].tocoo())
    e = coototensor(A[0][4].tocoo())
    f = coototensor(A[0][5].tocoo())
    g = coototensor(A[0][6].tocoo())
    A_t = torch.stack([a, b, c, d, e, f, g], dim=2).to_dense()
    temp = torch.matmul(A_t, adj_weight)
    temp = torch.squeeze(temp, 2)
    return temp + temp.transpose(0, 1)


def construct_adj(encode, struct_weight):
    weight = torch.diag(struct_weight)
    adjust_encode = torch.mm(encode, weight)
    struct_adj = torch.mm(adjust_encode, adjust_encode.t())
    normal_struct_adj = torch.nn.functional.softmax(struct_adj, dim=1)
    return normal_struct_adj


class BPHGNN(nn.Module):

    def __init__(self, nfeat, nhid, out, dropout):
        super(BPHGNN, self).__init__()
        """
        # Multilayer Graph Convolution
        """
        self.gc1 = GraphConvolution(nfeat, out)
        self.gc2 = GraphConvolution(out, out)
        self.dropout = dropout
        """
        Set the trainable weight of adjacency matrix aggregation
        """
        self.weight_b = torch.nn.Parameter(torch.FloatTensor(7, 1), requires_grad=True)
        torch.nn.init.uniform_(self.weight_b, a=0, b=0.1)
        self.struct_weight = torch.nn.Parameter(torch.ones(7), requires_grad=True)
        torch.nn.init.uniform_(self.struct_weight, a=0, b=0.1)

    def forward(self, feature, A, encode, use_relu=True):
        final_A = adj_matrix_weight_merge(A, self.weight_b)
        try:
            feature = torch.tensor(feature.astype(float).toarray())
        except:
            try:
                feature = torch.from_numpy(feature.toarray())
            except:
                pass
        U1 = self.gc1(feature, final_A)
        U2 = self.gc2(U1, final_A)
        struct_adj = construct_adj(encode, self.struct_weight)
        None
        U3 = self.gc1(feature, struct_adj)
        U4 = self.gc2(U3, struct_adj)
        result = ((U1 + U2) / 2 + U4) / 2
        return result, (U1 + U2) / 2, U4


class BatchGRU(nn.Module):

    def __init__(self, hidden_size=300):
        super(BatchGRU, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, batch_first=True, bidirectional=True)
        self.bias = nn.Parameter(torch.Tensor(self.hidden_size))
        self.bias.data.uniform_(-1.0 / math.sqrt(self.hidden_size), 1.0 / math.sqrt(self.hidden_size))

    def forward(self, node, a_scope):
        hidden = node
        message = F.relu(node + self.bias)
        MAX_node_len = max(a_scope)
        message_lst = []
        hidden_lst = []
        a_start = 0
        for i in a_scope:
            i = int(i)
            if i == 0:
                assert 0
            cur_message = message.narrow(0, a_start, i)
            cur_hidden = hidden.narrow(0, a_start, i)
            hidden_lst.append(cur_hidden.max(0)[0].unsqueeze(0).unsqueeze(0))
            a_start += i
            cur_message = torch.nn.ZeroPad2d((0, 0, 0, MAX_node_len - cur_message.shape[0]))(cur_message)
            message_lst.append(cur_message.unsqueeze(0))
        message_lst = torch.cat(message_lst, 0)
        hidden_lst = torch.cat(hidden_lst, 1)
        hidden_lst = hidden_lst.repeat(2, 1, 1)
        cur_message, cur_hidden = self.gru(message_lst, hidden_lst)
        cur_message_unpadding = []
        kk = 0
        for a_size in a_scope:
            a_size = int(a_size)
            cur_message_unpadding.append(cur_message[kk, :a_size].view(-1, 2 * self.hidden_size))
            kk += 1
        cur_message_unpadding = torch.cat(cur_message_unpadding, 0)
        return cur_message_unpadding


class BaseModel(nn.Module, metaclass=ABCMeta):

    @classmethod
    def build_model_from_args(cls, args, hg):
        """
        Build the model instance from args and hg.

        So every subclass inheriting it should override the method.
        """
        raise NotImplementedError('Models must implement the build_model_from_args method')

    def __init__(self):
        super(BaseModel, self).__init__()

    def forward(self, *args):
        """
        The model plays a role of encoder. So the forward will encoder original features into new features.

        Parameters
        -----------
        hg : dgl.DGlHeteroGraph
            the heterogeneous graph
        h_dict : dict[str, th.Tensor]
            the dict of heterogeneous feature

        Return
        -------
        out_dic : dict[str, th.Tensor]
            A dict of encoded feature. In general, it should ouput all nodes embedding.
            It is allowed that just output the embedding of target nodes which are participated in loss calculation.
        """
        raise NotImplementedError

    def extra_loss(self):
        """
        Some model want to use L2Norm which is not applied all parameters.

        Returns
        -------
        th.Tensor
        """
        raise NotImplementedError

    def h2dict(self, h, hdict):
        pre = 0
        out_dict = {}
        for i, value in hdict.items():
            out_dict[i] = h[pre:value.shape[0] + pre]
            pre += value.shape[0]
        return out_dict

    def get_emb(self):
        """
        Return the embedding of a model for further analysis.

        Returns
        -------
        numpy.array
        """
        raise NotImplementedError


class RelGraphConvLayer(nn.Module):
    """Relational graph convolution layer.

    We use `HeteroGraphConv <https://docs.dgl.ai/api/python/nn.pytorch.html#heterographconv>`_ to implement the model.

    Parameters
    ----------
    in_feat : int
        Input feature size.
    out_feat : int
        Output feature size.
    rel_names : list[str]
        Relation names.
    num_bases : int, optional
        Number of bases. If is none, use number of relations. Default: None.
    weight : bool, optional
        True if a linear layer is applied after message passing. Default: True
    bias : bool, optional
        True if bias is added. Default: True
    activation : callable, optional
        Activation function. Default: None
    self_loop : bool, optional
        True to include self loop message. Default: False
    dropout : float, optional
        Dropout rate. Default: 0.0
    """

    def __init__(self, in_feat, out_feat, rel_names, num_bases, *, weight=True, bias=True, activation=None, self_loop=False, dropout=0.0):
        super(RelGraphConvLayer, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.rel_names = rel_names
        self.num_bases = num_bases
        self.bias = bias
        self.activation = activation
        self.self_loop = self_loop
        self.batchnorm = False
        self.conv = dglnn.HeteroGraphConv({rel: dglnn.GraphConv(in_feat, out_feat, norm='right', weight=False, bias=False) for rel in rel_names})
        self.use_weight = weight
        self.use_basis = num_bases < len(self.rel_names) and weight
        if self.use_weight:
            if self.use_basis:
                self.basis = dglnn.WeightBasis((in_feat, out_feat), num_bases, len(self.rel_names))
            else:
                self.weight = nn.Parameter(th.Tensor(len(self.rel_names), in_feat, out_feat))
                nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))
        if bias:
            self.h_bias = nn.Parameter(th.Tensor(out_feat))
            nn.init.zeros_(self.h_bias)
        if self.self_loop:
            self.loop_weight = nn.Parameter(th.Tensor(in_feat, out_feat))
            nn.init.xavier_uniform_(self.loop_weight, gain=nn.init.calculate_gain('relu'))
        if self.batchnorm:
            self.bn = nn.BatchNorm1d(out_feat)
        self.dropout = nn.Dropout(dropout)

    def forward(self, g, inputs):
        """Forward computation
        Parameters
        ----------
        g : DGLHeteroGraph
            Input graph.
        inputs : dict[str, torch.Tensor]
            Node feature for each node type.
        Returns
        -------
        dict[str, torch.Tensor]
            New node features for each node type.
        """
        g = g.local_var()
        if self.use_weight:
            weight = self.basis() if self.use_basis else self.weight
            wdict = {self.rel_names[i]: {'weight': w.squeeze(0)} for i, w in enumerate(th.split(weight, 1, dim=0))}
        else:
            wdict = {}
        if g.is_block:
            inputs_src = inputs
            inputs_dst = {k: v[:g.number_of_dst_nodes(k)] for k, v in inputs.items()}
        else:
            inputs_src = inputs_dst = inputs
        hs = self.conv(g, inputs_src, mod_kwargs=wdict)

        def _apply(ntype, h):
            if self.self_loop:
                h = h + th.matmul(inputs_dst[ntype], self.loop_weight)
            if self.bias:
                h = h + self.h_bias
            if self.activation:
                h = self.activation(h)
            if self.batchnorm:
                h = self.bn(h)
            return self.dropout(h)
        return {ntype: _apply(ntype, h) for ntype, h in hs.items()}


class RGCN(BaseModel):
    """
    **Title:** `Modeling Relational Data with Graph Convolutional Networks <https://arxiv.org/abs/1703.06103>`_

    **Authors:** Michael Schlichtkrull, Thomas N. Kipf, Peter Bloem, Rianne van den Berg, Ivan Titov, Max Welling

    Parameters
    ----------
    in_dim : int
        Input feature size.
    hidden_dim : int
        Hidden dimension .
    out_dim : int
        Output feature size.
    etypes : list[str]
        Relation names.
    num_bases : int, optional
        Number of bases. If is none, use number of relations. Default: None.
    num_hidden_layers: int
        Number of RelGraphConvLayer
    dropout : float, optional
        Dropout rate. Default: 0.0
    use_self_loop : bool, optional
        True to include self loop message. Default: False

    Attributes
    -----------
    RelGraphConvLayer: RelGraphConvLayer

    """

    @classmethod
    def build_model_from_args(cls, args, hg):
        return cls(args.hidden_dim, args.hidden_dim, args.out_dim, hg.etypes, args.n_bases, args.num_layers - 2, dropout=args.dropout)

    def __init__(self, in_dim, hidden_dim, out_dim, etypes, num_bases, num_hidden_layers=1, dropout=0, use_self_loop=False):
        super(RGCN, self).__init__()
        self.in_dim = in_dim
        self.h_dim = hidden_dim
        self.out_dim = out_dim
        self.rel_names = list(set(etypes))
        self.rel_names.sort()
        if num_bases < 0 or num_bases > len(self.rel_names):
            self.num_bases = len(self.rel_names)
        else:
            self.num_bases = num_bases
        self.num_hidden_layers = num_hidden_layers
        self.dropout = dropout
        self.use_self_loop = use_self_loop
        self.layers = nn.ModuleList()
        self.layers.append(RelGraphConvLayer(self.in_dim, self.h_dim, self.rel_names, self.num_bases, activation=F.relu, self_loop=self.use_self_loop, dropout=self.dropout, weight=True))
        for i in range(self.num_hidden_layers):
            self.layers.append(RelGraphConvLayer(self.h_dim, self.h_dim, self.rel_names, self.num_bases, activation=F.relu, self_loop=self.use_self_loop, dropout=self.dropout))
        self.layers.append(RelGraphConvLayer(self.h_dim, self.out_dim, self.rel_names, self.num_bases, activation=None, self_loop=self.use_self_loop))

    def forward(self, hg, h_dict):
        """
        Support full-batch and mini-batch training.

        Parameters
        ----------
        hg: dgl.HeteroGraph or dgl.blocks
            Input graph
        h_dict: dict[str, th.Tensor]
            Input feature
        Returns
        -------
        h: dict[str, th.Tensor]
            output feature
        """
        if hasattr(hg, 'ntypes'):
            for layer in self.layers:
                h_dict = layer(hg, h_dict)
        else:
            for layer, block in zip(self.layers, hg):
                h_dict = layer(block, h_dict)
        return h_dict

    def l2_penalty(self):
        loss = 0.0005 * th.norm(self.layers[0].weight, p=2, dim=1)
        return loss


class RGCNLayer(nn.Module):

    def __init__(self, inp_dim, out_dim, aggregator, bias=None, activation=None, dropout=0.0, edge_dropout=0.0, is_input_layer=False):
        super(RGCNLayer, self).__init__()
        self.bias = bias
        self.activation = activation
        if self.bias:
            self.bias = nn.Parameter(torch.Tensor(out_dim))
            nn.init.xavier_uniform_(self.bias, gain=nn.init.calculate_gain('relu'))
        self.aggregator = aggregator
        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None
        if edge_dropout:
            self.edge_dropout = nn.Dropout(edge_dropout)
        else:
            self.edge_dropout = Identity()

    def propagate(self, g):
        raise NotImplementedError

    def forward(self, g, attn_rel_emb=None):
        self.propagate(g, attn_rel_emb)
        node_repr = g.ndata['h']
        if self.bias:
            node_repr = node_repr + self.bias
        if self.activation:
            node_repr = self.activation(node_repr)
        if self.dropout:
            node_repr = self.dropout(node_repr)
        g.ndata['h'] = node_repr
        if self.is_input_layer:
            g.ndata['repr'] = g.ndata['h'].unsqueeze(1)
        else:
            g.ndata['repr'] = torch.cat([g.ndata['repr'], g.ndata['h'].unsqueeze(1)], dim=1)


class RGCNBasisLayer(RGCNLayer):

    def __init__(self, inp_dim, out_dim, aggregator, attn_rel_emb_dim, num_rels, num_bases=-1, bias=None, activation=None, dropout=0.0, edge_dropout=0.0, is_input_layer=False, has_attn=False):
        super(RGCNBasisLayer, self).__init__(inp_dim, out_dim, aggregator, bias, activation, dropout=dropout, edge_dropout=edge_dropout, is_input_layer=is_input_layer)
        self.inp_dim = inp_dim
        self.out_dim = out_dim
        self.attn_rel_emb_dim = attn_rel_emb_dim
        self.num_rels = num_rels
        self.num_bases = num_bases
        self.is_input_layer = is_input_layer
        self.has_attn = has_attn
        if self.num_bases <= 0 or self.num_bases > self.num_rels:
            self.num_bases = self.num_rels
        self.weight = nn.Parameter(torch.Tensor(self.num_bases, self.inp_dim, self.out_dim))
        self.w_comp = nn.Parameter(torch.Tensor(self.num_rels, self.num_bases))
        if self.has_attn:
            self.A = nn.Linear(2 * self.inp_dim + 2 * self.attn_rel_emb_dim, inp_dim)
            self.B = nn.Linear(inp_dim, 1)
        self.self_loop_weight = nn.Parameter(torch.Tensor(self.inp_dim, self.out_dim))
        nn.init.xavier_uniform_(self.self_loop_weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.w_comp, gain=nn.init.calculate_gain('relu'))

    def propagate(self, g, attn_rel_emb=None):
        weight = self.weight.view(self.num_bases, self.inp_dim * self.out_dim)
        weight = torch.matmul(self.w_comp, weight).view(self.num_rels, self.inp_dim, self.out_dim)
        g = g
        g.edata['w'] = self.edge_dropout(torch.ones(g.number_of_edges(), 1))
        input_ = 'feat' if self.is_input_layer else 'h'

        def msg_func(edges):
            w = weight.index_select(0, edges.data['type'])
            msg = edges.data['w'] * torch.bmm(edges.src[input_].unsqueeze(1), w).squeeze(1)
            curr_emb = torch.mm(edges.dst[input_], self.self_loop_weight)
            if self.has_attn:
                e = torch.cat([edges.src[input_], edges.dst[input_], attn_rel_emb(edges.data['type']), attn_rel_emb(edges.data['label'])], dim=1)
                a = torch.sigmoid(self.B(F.relu(self.A(e))))
            else:
                a = torch.ones((len(edges), 1))
            return {'curr_emb': curr_emb, 'msg': msg, 'alpha': a}
        g.update_all(msg_func, self.aggregator, None)


class Aggregator(nn.Module):

    def __init__(self, emb_dim):
        super(Aggregator, self).__init__()

    def forward(self, node):
        curr_emb = node.mailbox['curr_emb'][:, 0, :]
        nei_msg = torch.bmm(node.mailbox['alpha'].transpose(1, 2), node.mailbox['msg']).squeeze(1)
        new_emb = self.update_embedding(curr_emb, nei_msg)
        return {'h': new_emb}

    @abc.abstractmethod
    def update_embedding(curr_emb, nei_msg):
        raise NotImplementedError


class SumAggregator(Aggregator):

    def __init__(self, emb_dim):
        super(SumAggregator, self).__init__(emb_dim)

    def update_embedding(self, curr_emb, nei_msg):
        new_emb = nei_msg + curr_emb
        return new_emb


class MLPAggregator(Aggregator):

    def __init__(self, emb_dim):
        super(MLPAggregator, self).__init__(emb_dim)
        self.linear = nn.Linear(2 * emb_dim, emb_dim)

    def update_embedding(self, curr_emb, nei_msg):
        inp = torch.cat((nei_msg, curr_emb), 1)
        new_emb = F.relu(self.linear(inp))
        return new_emb


class GRUAggregator(Aggregator):

    def __init__(self, emb_dim):
        super(GRUAggregator, self).__init__(emb_dim)
        self.gru = nn.GRUCell(emb_dim, emb_dim)

    def update_embedding(self, curr_emb, nei_msg):
        new_emb = self.gru(nei_msg, curr_emb)
        return new_emb


class CompGraphConvLayer(nn.Module):
    """One layer of simplified CompGCN."""

    def __init__(self, in_dim, out_dim, rel_names, comp_fn='sub', activation=None, batchnorm=False, dropout=0):
        super(CompGraphConvLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.comp_fn = comp_fn
        self.actvation = activation
        self.batchnorm = batchnorm
        self.rel_names = rel_names
        self.dropout = nn.Dropout(dropout)
        if self.batchnorm:
            self.bn = nn.BatchNorm1d(out_dim)
        self.W_O = nn.Linear(self.in_dim, self.out_dim)
        self.W_I = nn.Linear(self.in_dim, self.out_dim)
        self.W_S = nn.Linear(self.in_dim, self.out_dim)
        self.W_R = nn.Linear(self.in_dim, self.out_dim)
        self.conv = dglnn.HeteroGraphConv({rel: CompConv(comp_fn=comp_fn, norm='right', _allow_zero_in_degree=True) for rel in rel_names})

    def forward(self, hg, n_in_feats, r_feats):
        """
        Compute one layer of composition transfer for one relation only in a
        homogeneous graph with bidirectional edges.
        """
        with hg.local_scope():
            wdict = {}
            for i, etype in enumerate(self.rel_names):
                if etype[:4] == 'rev-' or etype[-4:] == '-rev':
                    W = self.W_I
                else:
                    W = self.W_O
                wdict[etype] = {'Linear': W, 'h_e': r_feats[i + 1]}
            if hg.is_block:
                inputs_src = n_in_feats
                inputs_dst = {k: v[:hg.number_of_dst_nodes(k)] for k, v in n_in_feats.items()}
                outputs = self.conv(hg, (inputs_src, inputs_dst), mod_kwargs=wdict)
            else:
                inputs_src = inputs_dst = n_in_feats
                outputs = self.conv(hg, inputs_src, mod_kwargs=wdict)
            for n, emd in outputs.items():
                if self.comp_fn == 'sub':
                    h_self = self.W_S(inputs_dst[n] - r_feats[-1])
                elif self.comp_fn == 'mul':
                    h_self = self.W_S(inputs_dst[n] * r_feats[-1])
                elif self.comp_fn == 'ccorr':
                    h_self = self.W_S(ccorr(inputs_dst[n], r_feats[-1]))
                else:
                    raise Exception('Only supports sub, mul, and ccorr')
                h_self.add_(emd)
                if self.batchnorm:
                    if h_self.shape[0] > 1:
                        h_self = self.bn(h_self)
                n_out_feats = self.dropout(h_self)
                if self.actvation is not None:
                    n_out_feats = self.actvation(n_out_feats)
                outputs[n] = n_out_feats
        r_out_feats = self.W_R(r_feats)
        r_out_feats = self.dropout(r_out_feats)
        if self.actvation is not None:
            r_out_feats = self.actvation(r_out_feats)
        return outputs, r_out_feats


class Discriminator(nn.Module):
    """
    A generator :math:`G` samples fake node embeddings from a continuous distribution. The distribution is Gaussian distribution:

    .. math::
        \\mathcal{N}(\\mathbf{e}_u^{G^T} \\mathbf{M}_r^G, \\mathbf{\\sigma}^2 \\mathbf{I})

    where :math:`e_u^G \\in \\mathbb{R}^{d \\times 1}` and :math:`M_r^G \\in \\mathbb{R}^{d \\times d}` denote the node embedding of :math:`u \\in \\mathcal{V}` and the relation matrix of :math:`r \\in \\mathcal{R}` for the generator.

    There are also a two-layer MLP integrated into the generator for enhancing the expression of the fake samples:

    .. math::
        G(\\mathbf{u}, \\mathbf{r}; \\mathbf{\\theta}^G) = f(\\mathbf{W_2}f(\\mathbf{W}_1 \\mathbf{e} + \\mathbf{b}_1) + \\mathbf{b}_2)

    where :math:`e` is drawn from Gaussian distribution. :math:`\\{W_i, b_i}` denote the weight matrix and bias vector for :math:`i`-th layer.

    The discriminator Loss is:

    .. math::
        L_1^D = \\mathbb{E}_{\\langle u,v,r\\rangle \\sim P_G} = -\\log D(e_v^u|u,r))

        L_2^D = \\mathbb{E}_{\\langle u,v\\rangle \\sim P_G, r' \\sim P_{R'}} = -\\log (1-D(e_v^u|u,r')))

        L_3^D = \\mathbb{E}_{\\langle u,v\\rangle \\sim P_G, e'_v \\sim G(u,r;\\theta^G)} = -\\log (1-D(e_v'|u,r)))

        L_G = L_1^D + L_2^D + L_2^D + \\lambda^D || \\theta^D ||_2^2

    where :math:`\\theta^D` denote all the learnable parameters in Discriminator.

    Parameters
    -----------
    emb_size: int
        embeddings size.
    hg: dgl.heteroGraph
        heterogenous graph.

    """

    def __init__(self, emb_size, hg):
        super().__init__()
        self.n_relation = len(hg.etypes)
        self.node_emb_dim = emb_size
        self.nodes_embedding = nn.ParameterDict()
        for nodes_type, nodes_emb in hg.ndata['h'].items():
            self.nodes_embedding[nodes_type] = nn.Parameter(nodes_emb, requires_grad=True)
        self.relation_matrix = nn.ParameterDict()
        for et in hg.etypes:
            rm = torch.empty(self.node_emb_dim, self.node_emb_dim)
            rm = nn.init.xavier_normal_(rm)
            self.relation_matrix[et] = nn.Parameter(rm, requires_grad=True)

    def forward(self, pos_hg, neg_hg1, neg_hg2, generate_neighbor_emb):
        """
        Parameters
        ----------
        pos_hg:
            sampled postive graph.
        neg_hg1:
            sampled negative graph with wrong relation.
        neg_hg2:
            sampled negative graph wtih wrong node.
        generate_neighbor_emb:
            generator node embeddings.
        """
        self.assign_node_data(pos_hg)
        self.assign_node_data(neg_hg1)
        self.assign_node_data(neg_hg2, generate_neighbor_emb)
        self.assign_edge_data(pos_hg)
        self.assign_edge_data(neg_hg1)
        self.assign_edge_data(neg_hg2)
        pos_score = self.score_pred(pos_hg)
        neg_score1 = self.score_pred(neg_hg1)
        neg_score2 = self.score_pred(neg_hg2)
        return pos_score, neg_score1, neg_score2

    def get_parameters(self):
        """
        return discriminator node embeddings and relation embeddings.
        """
        return {k: self.nodes_embedding[k] for k in self.nodes_embedding.keys()}, {k: self.relation_matrix[k] for k in self.relation_matrix.keys()}

    def score_pred(self, hg):
        """
        predict the discriminator score for sampled heterogeneous graph.
        """
        score_list = []
        with hg.local_scope():
            for et in hg.canonical_etypes:
                hg.apply_edges(lambda edges: {'s': edges.src['h'].unsqueeze(1).matmul(edges.data['e']).reshape(hg.num_edges(et), 64)}, etype=et)
                if len(hg.edata['f']) == 0:
                    hg.apply_edges(lambda edges: {'score': edges.data['s'].multiply(edges.dst['h'])}, etype=et)
                else:
                    hg.apply_edges(lambda edges: {'score': edges.data['s'].multiply(edges.data['f'])}, etype=et)
                score = torch.sum(hg.edata['score'].pop(et), dim=1)
                score_list.append(score)
        return torch.cat(score_list)

    def assign_edge_data(self, hg):
        d = {}
        for et in hg.canonical_etypes:
            e = self.relation_matrix[et[1]]
            n = hg.num_edges(et)
            d[et] = e.expand(n, -1, -1)
        hg.edata['e'] = d

    def assign_node_data(self, hg, generate_neighbor_emb=None):
        for nt in hg.ntypes:
            hg.nodes[nt].data['h'] = self.nodes_embedding[nt]
        if generate_neighbor_emb:
            hg.edata['f'] = generate_neighbor_emb


class AvgReadout(nn.Module):
    """
    Considering the efficiency of the method, we simply employ average pooling, computing the average of the set of embedding matrices

    .. math::
      \\begin{equation}
        \\mathbf{H}=\\mathcal{Q}\\left(\\left\\{\\mathbf{H}^{(r)} \\mid r \\in \\mathcal{R}\\right\\}\\right)=\\frac{1}{|\\mathcal{R}|} \\sum_{r \\in \\mathcal{R}} \\mathbf{H}^{(r)}
      \\end{equation}
    """

    def __init__(self):
        super(AvgReadout, self).__init__()

    def forward(self, seq):
        return torch.mean(seq, 0)


class LogReg(nn.Module):
    """
    Logical classifier
    """

    def __init__(self, ft_in, nb_classes):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(ft_in, nb_classes)
        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq):
        ret = self.fc(seq)
        return ret


class MLP(torch.nn.Module):

    def __init__(self, input_dim, feature_dim, hidden_dim, output_dim, feature_pre=True, layer_num=2, dropout=True, **kwargs):
        super(MLP, self).__init__()
        self.feature_pre = feature_pre
        self.layer_num = layer_num
        self.dropout = dropout
        if feature_pre:
            self.linear_pre = nn.Linear(input_dim, feature_dim, bias=True)
        else:
            self.linear_first = nn.Linear(input_dim, hidden_dim)
        self.linear_hidden = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for i in range(layer_num - 2)])
        self.linear_out = nn.Linear(feature_dim, output_dim, bias=True)

    def forward(self, data):
        x = data
        if self.feature_pre:
            x = self.linear_pre(x)
        prelu = nn.PReLU()
        x = prelu(x)
        for i in range(self.layer_num - 2):
            x = self.linear_hidden[i](x)
            x = F.tanh(x)
            if self.dropout:
                x = F.dropout(x, training=self.training)
        x = self.linear_out(x)
        x = F.normalize(x, p=2, dim=-1)
        return x


class LogisticRegression(nn.Module):

    def __init__(self, num_dim, num_class):
        super().__init__()
        self.linear = nn.Linear(num_dim, num_class)
        torch.nn.init.xavier_uniform_(self.linear.weight.data)
        self.linear.bias.data.fill_(0.0)
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, x, y):
        logits = self.linear(x)
        loss = self.cross_entropy(logits, y)
        return logits, loss


class Op(nn.Module):

    def __init__(self):
        super(Op, self).__init__()

    def forward(self, x, adjs, idx):
        return torch.spmm(adjs[idx], x)


class Cell(nn.Module):

    def __init__(self, n_step, n_hid_prev, n_hid, use_norm=True, use_nl=True):
        super(Cell, self).__init__()
        self.affine = nn.Linear(n_hid_prev, n_hid)
        self.n_step = n_step
        self.norm = nn.LayerNorm(n_hid) if use_norm is True else lambda x: x
        self.use_nl = use_nl
        self.ops_seq = nn.ModuleList()
        self.ops_res = nn.ModuleList()
        for i in range(self.n_step):
            self.ops_seq.append(Op())
        for i in range(1, self.n_step):
            for j in range(i):
                self.ops_res.append(Op())

    def forward(self, x, adjs, idxes_seq, idxes_res):
        x = self.affine(x)
        states = [x]
        offset = 0
        for i in range(self.n_step):
            seqi = self.ops_seq[i](states[i], adjs[:-1], idxes_seq[i])
            resi = sum(self.ops_res[offset + j](h, adjs, idxes_res[offset + j]) for j, h in enumerate(states[:i]))
            offset += i
            states.append(seqi + resi)
        output = self.norm(states[-1])
        if self.use_nl:
            output = F.gelu(output)
        return output


class Op1(nn.Module):
    """
    operation for one link in the DAG search space
    """

    def __init__(self):
        super(Op1, self).__init__()

    def forward(self, x, adjs, ws, idx):
        return ws[idx] * torch.spmm(adjs[idx], x)


class Cell1(nn.Module):
    """
    the DAG search space
    """

    def __init__(self, n_step, n_hid_prev, n_hid, cstr, use_norm=True, use_nl=True):
        super(Cell1, self).__init__()
        self.affine = nn.Linear(n_hid_prev, n_hid)
        self.n_step = n_step
        self.norm = nn.LayerNorm(n_hid, elementwise_affine=False) if use_norm is True else lambda x: x
        self.use_nl = use_nl
        assert isinstance(cstr, list)
        self.cstr = cstr
        self.ops_seq = nn.ModuleList()
        for i in range(1, self.n_step):
            self.ops_seq.append(Op1())
        self.ops_res = nn.ModuleList()
        for i in range(2, self.n_step):
            for j in range(i - 1):
                self.ops_res.append(Op1())
        self.last_seq = Op1()
        self.last_res = nn.ModuleList()
        for i in range(self.n_step - 1):
            self.last_res.append(Op1())

    def forward(self, x, adjs, ws_seq, idxes_seq, ws_res, idxes_res):
        x = self.affine(x)
        states = [x]
        offset = 0
        for i in range(self.n_step - 1):
            seqi = self.ops_seq[i](states[i], adjs[:-1], ws_seq[0][i], idxes_seq[0][i])
            resi = sum(self.ops_res[offset + j](h, adjs, ws_res[0][offset + j], idxes_res[0][offset + j]) for j, h in enumerate(states[:i]))
            offset += i
            states.append(seqi + resi)
        adjs_cstr = [adjs[i] for i in self.cstr]
        out_seq = self.last_seq(states[-1], adjs_cstr, ws_seq[1], idxes_seq[1])
        adjs_cstr.append(adjs[-1])
        out_res = sum(self.last_res[i](h, adjs_cstr, ws_res[1][i], idxes_res[1][i]) for i, h in enumerate(states[:-1]))
        output = self.norm(out_seq + out_res)
        if self.use_nl:
            output = F.gelu(output)
        return output


class search_model(nn.Module):

    def __init__(self, in_dims, n_hid, n_adjs, n_steps, cstr, attn_dim=64, use_norm=True, out_nl=True):
        super(search_model, self).__init__()
        self.cstr = cstr
        self.n_adjs = n_adjs
        self.n_hid = n_hid
        self.ws = nn.ModuleList()
        assert isinstance(in_dims, list)
        for i in range(len(in_dims)):
            self.ws.append(nn.Linear(in_dims[i], n_hid))
        assert isinstance(n_steps, list)
        self.metas = nn.ModuleList()
        for i in range(len(n_steps)):
            self.metas.append(Cell1(n_steps[i], n_hid, n_hid, cstr, use_norm=use_norm, use_nl=out_nl))
        self.as_seq = []
        self.as_last_seq = []
        for i in range(len(n_steps)):
            if n_steps[i] > 1:
                ai = 0.001 * torch.randn(n_steps[i] - 1, n_adjs - 1)
                ai = ai
                ai.requires_grad_(True)
                self.as_seq.append(ai)
            else:
                self.as_seq.append(None)
            ai_last = 0.001 * torch.randn(len(cstr))
            ai_last = ai_last
            ai_last.requires_grad_(True)
            self.as_last_seq.append(ai_last)
        ks = [sum(1 for i in range(2, n_steps[k]) for j in range(i - 1)) for k in range(len(n_steps))]
        self.as_res = []
        self.as_last_res = []
        for i in range(len(n_steps)):
            if ks[i] > 0:
                ai = 0.001 * torch.randn(ks[i], n_adjs)
                ai = ai
                ai.requires_grad_(True)
                self.as_res.append(ai)
            else:
                self.as_res.append(None)
            if n_steps[i] > 1:
                ai_last = 0.001 * torch.randn(n_steps[i] - 1, len(cstr) + 1)
                ai_last = ai_last
                ai_last.requires_grad_(True)
                self.as_last_res.append(ai_last)
            else:
                self.as_last_res.append(None)
        assert ks[0] + n_steps[0] + (0 if self.as_last_res[0] is None else self.as_last_res[0].size(0)) == (1 + n_steps[0]) * n_steps[0] // 2
        self.attn_fc1 = nn.Linear(n_hid, attn_dim)
        self.attn_fc2 = nn.Linear(attn_dim, 1)

    def alphas(self):
        alphas = []
        for each in self.as_seq:
            if each is not None:
                alphas.append(each)
        for each in self.as_last_seq:
            alphas.append(each)
        for each in self.as_res:
            if each is not None:
                alphas.append(each)
        for each in self.as_last_res:
            if each is not None:
                alphas.append(each)
        return alphas

    def sample(self, eps):
        """
        to sample one candidate edge type per link
        """
        idxes_seq = []
        idxes_res = []
        if np.random.uniform() < eps:
            for i in range(len(self.metas)):
                temp = []
                temp.append(None if self.as_seq[i] is None else torch.randint(low=0, high=self.as_seq[i].size(-1), size=self.as_seq[i].size()[:-1]))
                temp.append(torch.randint(low=0, high=self.as_last_seq[i].size(-1), size=(1,)))
                idxes_seq.append(temp)
            for i in range(len(self.metas)):
                temp = []
                temp.append(None if self.as_res[i] is None else torch.randint(low=0, high=self.as_res[i].size(-1), size=self.as_res[i].size()[:-1]))
                temp.append(None if self.as_last_res[i] is None else torch.randint(low=0, high=self.as_last_res[i].size(-1), size=self.as_last_res[i].size()[:-1]))
                idxes_res.append(temp)
        else:
            for i in range(len(self.metas)):
                temp = []
                temp.append(None if self.as_seq[i] is None else torch.argmax(F.softmax(self.as_seq[i], dim=-1), dim=-1))
                temp.append(torch.argmax(F.softmax(self.as_last_seq[i], dim=-1), dim=-1))
                idxes_seq.append(temp)
            for i in range(len(self.metas)):
                temp = []
                temp.append(None if self.as_res[i] is None else torch.argmax(F.softmax(self.as_res[i], dim=-1), dim=-1))
                temp.append(None if self.as_last_res[i] is None else torch.argmax(F.softmax(self.as_last_res[i], dim=-1), dim=-1))
                idxes_res.append(temp)
        return idxes_seq, idxes_res

    def forward(self, node_feats, node_types, adjs, idxes_seq, idxes_res):
        hid = torch.zeros((len(node_types), self.n_hid))
        for i in range(len(node_feats)):
            hid[node_types == i] = self.ws[i](node_feats[i])
        temps = []
        attns = []
        for i, meta in enumerate(self.metas):
            ws_seq = []
            ws_seq.append(None if self.as_seq[i] is None else F.softmax(self.as_seq[i], dim=-1))
            ws_seq.append(F.softmax(self.as_last_seq[i], dim=-1))
            ws_res = []
            ws_res.append(None if self.as_res[i] is None else F.softmax(self.as_res[i], dim=-1))
            ws_res.append(None if self.as_last_res[i] is None else F.softmax(self.as_last_res[i], dim=-1))
            hidi = meta(hid, adjs, ws_seq, idxes_seq[i], ws_res, idxes_res[i])
            temps.append(hidi)
            attni = self.attn_fc2(torch.tanh(self.attn_fc1(temps[-1])))
            attns.append(attni)
        hids = torch.stack(temps, dim=0).transpose(0, 1)
        attns = F.softmax(torch.cat(attns, dim=-1), dim=-1)
        out = (attns.unsqueeze(dim=-1) * hids).sum(dim=1)
        return out

    def parse(self):
        """
        to derive a meta graph indicated by arch parameters
        """
        idxes_seq, idxes_res = self.sample(0.0)
        msg_seq = []
        msg_res = []
        for i in range(len(idxes_seq)):
            map_seq = [self.cstr[idxes_seq[i][1].item()]]
            msg_seq.append(map_seq if idxes_seq[i][0] is None else idxes_seq[i][0].tolist() + map_seq)
            assert len(msg_seq[i]) == self.metas[i].n_step
            temp_res = []
            if idxes_res[i][1] is not None:
                for item in idxes_res[i][1].tolist():
                    if item < len(self.cstr):
                        temp_res.append(self.cstr[item])
                    else:
                        assert item == len(self.cstr)
                        temp_res.append(self.n_adjs - 1)
                if idxes_res[i][0] is not None:
                    temp_res = idxes_res[i][0].tolist() + temp_res
            assert len(temp_res) == self.metas[i].n_step * (self.metas[i].n_step - 1) // 2
            msg_res.append(temp_res)
        return msg_seq, msg_res


def get_param(shape):
    param = nn.Parameter(torch.Tensor(*shape))
    nn.init.xavier_normal_(param.data)
    return param


def broadcast(src: 'torch.Tensor', other: 'torch.Tensor', dim: 'int'):
    if dim < 0:
        dim = other.dim() + dim
    if src.dim() == 1:
        for _ in range(0, dim):
            src = src.unsqueeze(0)
    for _ in range(src.dim(), other.dim()):
        src = src.unsqueeze(-1)
    src = src.expand(other.size())
    return src


def scatter_sum(src: 'torch.Tensor', index: 'torch.Tensor', dim: 'int'=-1, out: 'Optional[torch.Tensor]'=None, dim_size: 'Optional[int]'=None) ->torch.Tensor:
    index = broadcast(index, src, dim)
    if out is None:
        size = list(src.size())
        if dim_size is not None:
            size[dim] = dim_size
        elif index.numel() == 0:
            size[dim] = 0
        else:
            size[dim] = int(index.max()) + 1
        out = torch.zeros(size, dtype=src.dtype, device=src.device)
        return out.scatter_add_(dim, index, src)
    else:
        return out.scatter_add_(dim, index, src)


class DisenLayer(nn.Module):

    def __init__(self, edge_index, edge_type, in_channels, out_channels, num_rels, act=lambda x: x, params=None, head_num=1):
        super(DisenLayer, self).__init__()
        self.node_dim = 0
        self.edge_index = edge_index
        self.edge_type = edge_type
        self.p = params
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.act = act
        self.device = None
        self.head_num = head_num
        self.num_rels = num_rels
        self.drop = torch.nn.Dropout(self.p.gcn_drop)
        self.dropout = torch.nn.Dropout(0.3)
        self.bn = torch.nn.BatchNorm1d(self.p.num_factors * out_channels)
        if self.p.bias:
            self.register_parameter('bias', Parameter(torch.zeros(out_channels)))
        num_edges = self.edge_index.size(1) // 2
        if self.device is None:
            self.device = self.edge_index.device
        self.in_index, self.out_index = self.edge_index[:, :num_edges], self.edge_index[:, num_edges:]
        self.in_type, self.out_type = self.edge_type[:num_edges], self.edge_type[num_edges:]
        self.loop_index = torch.stack([torch.arange(self.p.num_ent), torch.arange(self.p.num_ent)])
        self.loop_type = torch.full((self.p.num_ent,), 2 * self.num_rels, dtype=torch.long)
        num_ent = self.p.num_ent
        self.leakyrelu = nn.LeakyReLU(0.2)
        if self.p.att_mode == 'cat_emb' or self.p.att_mode == 'cat_weight':
            self.att_weight = get_param((1, self.p.num_factors, 2 * out_channels))
        else:
            self.att_weight = get_param((1, self.p.num_factors, out_channels))
        self.rel_weight = get_param((2 * self.num_rels + 1, self.p.num_factors, out_channels))
        self.loop_rel = get_param((1, out_channels))
        self.w_rel = get_param((out_channels, out_channels))

    def forward(self, x, rel_embed, mode):
        rel_embed = torch.cat([rel_embed, self.loop_rel], dim=0)
        edge_index = torch.cat([self.edge_index, self.loop_index], dim=1)
        edge_type = torch.cat([self.edge_type, self.loop_type])
        edge_index_i = edge_index[0]
        edge_index_j = edge_index[1]
        x_i = torch.index_select(x, dim=0, index=edge_index_i)
        x_j = torch.index_select(x, dim=0, index=edge_index_j)
        message_res = self.message(edge_index_i=edge_index_i, edge_index_j=edge_index_j, x_i=x_i, x_j=x_j, edge_type=edge_type, rel_embed=rel_embed, rel_weight=self.rel_weight)
        aggr_res = self.aggregate(input=message_res, edge_index_i=edge_index_i)
        out = self.update(aggr_res)
        if self.p.bias:
            out = out + self.bias
        out = self.bn(out.view(-1, self.p.num_factors * self.p.gcn_dim)).view(-1, self.p.num_factors, self.p.gcn_dim)
        entity1 = out if self.p.no_act else self.act(out)
        return entity1, torch.matmul(rel_embed, self.w_rel)[:-1]

    def message(self, edge_index_i, edge_index_j, x_i, x_j, edge_type, rel_embed, rel_weight):
        """
        edge_index_i : [E]
        x_i: [E, F]
        x_j: [E, F]
        """
        rel_embed = torch.index_select(rel_embed, 0, edge_type)
        rel_weight = torch.index_select(rel_weight, 0, edge_type)
        xj_rel = self.rel_transform(x_j, rel_embed, rel_weight)
        alpha = self._get_attention(edge_index_i, edge_index_j, x_i, x_j, rel_embed, rel_weight, xj_rel)
        alpha = self.drop(alpha)
        return xj_rel * alpha

    def aggregate(self, input, edge_index_i):
        return scatter_sum(input, edge_index_i, dim=0)

    def update(self, aggr_out):
        return aggr_out

    def _get_attention(self, edge_index_i, edge_index_j, x_i, x_j, rel_embed, rel_weight, mes_xj):
        if self.p.att_mode == 'learn':
            alpha = self.leakyrelu(torch.einsum('ekf, xkf->ek', [mes_xj, self.att_weight]))
            alpha = softmax(alpha, edge_index_i, num_nodes=self.p.num_ent)
        elif self.p.att_mode == 'dot_weight':
            sub_rel_emb = x_i * rel_weight
            obj_rel_emb = x_j * rel_weight
            alpha = self.leakyrelu(torch.einsum('ekf,ekf->ek', [sub_rel_emb, obj_rel_emb]))
            alpha = softmax(alpha, edge_index_i, num_nodes=self.p.num_ent)
        elif self.p.att_mode == 'dot_emb':
            sub_rel_emb = x_i * rel_embed.unsqueeze(1)
            obj_rel_emb = x_j * rel_embed.unsqueeze(1)
            alpha = self.leakyrelu(torch.einsum('ekf,ekf->ek', [sub_rel_emb, obj_rel_emb]))
            alpha = softmax(alpha, edge_index_i, num_nodes=self.p.num_ent)
        elif self.p.att_mode == 'cat_weight':
            sub_rel_emb = x_i * rel_weight
            obj_rel_emb = x_j * rel_weight
            alpha = self.leakyrelu(torch.einsum('ekf,xkf->ek', torch.cat([sub_rel_emb, obj_rel_emb], dim=2), self.att_weight))
            alpha = softmax(alpha, edge_index_i, num_nodes=self.p.num_ent)
        elif self.p.att_mode == 'cat_emb':
            sub_rel_emb = x_i * rel_embed.unsqueeze(1)
            obj_rel_emb = x_j * rel_embed.unsqueeze(1)
            alpha = self.leakyrelu(torch.einsum('ekf,xkf->ek', torch.cat([sub_rel_emb, obj_rel_emb], dim=2), self.att_weight))
            alpha = softmax(alpha, edge_index_i, num_nodes=self.p.num_ent)
        else:
            raise NotImplementedError
        return alpha.unsqueeze(2)

    def rel_transform(self, ent_embed, rel_embed, rel_weight, opn=None):
        if opn is None:
            opn = self.p.opn
        if opn == 'corr':
            trans_embed = ccorr(ent_embed * rel_weight, rel_embed.unsqueeze(1))
        elif opn == 'corr_ra':
            trans_embed = ccorr(ent_embed * rel_weight, rel_embed)
        elif opn == 'sub':
            trans_embed = ent_embed * rel_weight - rel_embed.unsqueeze(1)
        elif opn == 'es':
            trans_embed = ent_embed
        elif opn == 'sub_ra':
            trans_embed = ent_embed * rel_weight - rel_embed.unsqueeze(1)
        elif opn == 'mult':
            trans_embed = ent_embed * rel_embed.unsqueeze(1) * rel_weight
        elif opn == 'mult_ra':
            trans_embed = ent_embed * rel_embed * rel_weight
        elif opn == 'cross':
            trans_embed = ent_embed * rel_embed.unsqueeze(1) * rel_weight + ent_embed * rel_weight
        elif opn == 'cross_wo_rel':
            trans_embed = ent_embed * rel_weight
        elif opn == 'cross_simplfy':
            trans_embed = ent_embed * rel_embed + ent_embed
        elif opn == 'concat':
            trans_embed = torch.cat([ent_embed, rel_embed], dim=1)
        elif opn == 'concat_ra':
            trans_embed = torch.cat([ent_embed, rel_embed], dim=1) * rel_weight
        elif opn == 'ent_ra':
            trans_embed = ent_embed * rel_weight + rel_embed
        else:
            raise NotImplementedError
        return trans_embed

    def __repr__(self):
        return '{}({}, {}, num_rels={})'.format(self.__class__.__name__, self.in_channels, self.out_channels, self.num_rels)


class CLUBSample(nn.Module):

    def __init__(self, x_dim, y_dim, hidden_size):
        super(CLUBSample, self).__init__()
        self.p_mu = nn.Sequential(nn.Linear(x_dim, hidden_size // 2), nn.ReLU(), nn.Linear(hidden_size // 2, y_dim))
        self.p_logvar = nn.Sequential(nn.Linear(x_dim, hidden_size // 2), nn.ReLU(), nn.Linear(hidden_size // 2, y_dim), nn.Tanh())

    def get_mu_logvar(self, x_samples):
        mu = self.p_mu(x_samples)
        logvar = self.p_logvar(x_samples)
        return mu, logvar

    def loglikeli(self, x_samples, y_samples):
        mu, logvar = self.get_mu_logvar(x_samples)
        return (-(mu - y_samples) ** 2 / 2.0 / logvar.exp()).sum(dim=1).mean(dim=0)

    def forward(self, x_samples, y_samples):
        mu, logvar = self.get_mu_logvar(x_samples)
        sample_size = x_samples.shape[0]
        random_index = torch.randperm(sample_size).long()
        positive = -(mu - y_samples) ** 2 / logvar.exp()
        negative = -(mu - y_samples[random_index]) ** 2 / logvar.exp()
        upper_bound = (positive.sum(dim=-1) - negative.sum(dim=-1)).mean()
        return upper_bound / 2.0

    def learning_loss(self, x_samples, y_samples):
        return -self.loglikeli(x_samples, y_samples)


class SparseInputLinear(nn.Module):

    def __init__(self, inp_dim, out_dim):
        super(SparseInputLinear, self).__init__()
        weight = np.zeros((inp_dim, out_dim), dtype=np.float32)
        weight = nn.Parameter(torch.from_numpy(weight))
        bias = np.zeros(out_dim, dtype=np.float32)
        bias = nn.Parameter(torch.from_numpy(bias))
        self.inp_dim, self.out_dim = inp_dim, out_dim
        self.weight, self.bias = weight, bias
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / np.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        return torch.mm(x, self.weight) + self.bias


class CapsuleBase(BaseModel):

    def __init__(self, edge_index, edge_type, num_rel, params=None):
        super(CapsuleBase, self).__init__(params)
        self.edge_index = edge_index
        self.edge_type = edge_type
        self.device = self.edge_index.device
        self.init_embed = get_param((self.p.num_ent, self.p.init_dim))
        self.init_rel = get_param((num_rel * 2, self.p.gcn_dim))
        self.pca = SparseInputLinear(self.p.init_dim, self.p.num_factors * self.p.gcn_dim)
        conv_ls = []
        for i in range(self.p.gcn_layer):
            conv = DisenLayer(self.edge_index, self.edge_type, self.p.init_dim, self.p.gcn_dim, num_rel, act=self.act, params=self.p, head_num=self.p.head_num)
            self.add_module('conv_{}'.format(i), conv)
            conv_ls.append(conv)
        self.conv_ls = conv_ls
        if self.p.mi_train:
            if self.p.mi_method == 'club_b':
                num_dis = int(self.p.num_factors * (self.p.num_factors - 1) / 2)
                self.mi_Discs = nn.ModuleList([CLUBSample(self.p.gcn_dim, self.p.gcn_dim, self.p.gcn_dim) for fac in range(num_dis)])
            elif self.p.mi_method == 'club_s':
                self.mi_Discs = nn.ModuleList([CLUBSample((fac + 1) * self.p.gcn_dim, self.p.gcn_dim, (fac + 1) * self.p.gcn_dim) for fac in range(self.p.num_factors - 1)])
        self.register_parameter('bias', Parameter(torch.zeros(self.p.num_ent)))
        self.rel_drop = nn.Dropout(0.1)
        self.leakyrelu = nn.LeakyReLU(0.2)

    def lld_bst(self, sub, rel, drop1, mode='train'):
        x = self.act(self.pca(self.init_embed)).view(-1, self.p.num_factors, self.p.gcn_dim)
        r = self.init_rel
        for conv in self.conv_ls:
            x, r = conv(x, r, mode)
            if self.p.mi_drop:
                x = drop1(x)
            else:
                continue
        sub_emb = torch.index_select(x, 0, sub)
        lld_loss = 0.0
        sub_emb = sub_emb.view(-1, self.p.gcn_dim * self.p.num_factors)
        if self.p.mi_method == 'club_s':
            for i in range(self.p.num_factors - 1):
                bnd = i + 1
                lld_loss += self.mi_Discs[i].learning_loss(sub_emb[:, :bnd * self.p.gcn_dim], sub_emb[:, bnd * self.p.gcn_dim:(bnd + 1) * self.p.gcn_dim])
        elif self.p.mi_method == 'club_b':
            cnt = 0
            for i in range(self.p.num_factors):
                for j in range(i + 1, self.p.num_factors):
                    lld_loss += self.mi_Discs[cnt].learning_loss(sub_emb[:, i * self.p.gcn_dim:(i + 1) * self.p.gcn_dim], sub_emb[:, j * self.p.gcn_dim:(j + 1) * self.p.gcn_dim])
                    cnt += 1
        return lld_loss

    def mi_cal(self, sub_emb):

        def loss_dependence_hisc(zdata_trn, ncaps, nhidden):
            loss_dep = torch.zeros(1)
            hH = -1 / nhidden * torch.ones(nhidden, nhidden) + torch.eye(nhidden)
            kfactor = torch.zeros(ncaps, nhidden, nhidden)
            for mm in range(ncaps):
                data_temp = zdata_trn[:, mm * nhidden:(mm + 1) * nhidden]
                kfactor[mm, :, :] = torch.mm(data_temp.t(), data_temp)
            for mm in range(ncaps):
                for mn in range(mm + 1, ncaps):
                    mat1 = torch.mm(hH, kfactor[mm, :, :])
                    mat2 = torch.mm(hH, kfactor[mn, :, :])
                    mat3 = torch.mm(mat1, mat2)
                    teststat = torch.trace(mat3)
                    loss_dep = loss_dep + teststat
            return loss_dep

        def loss_dependence_club_s(sub_emb):
            mi_loss = 0.0
            for i in range(self.p.num_factors - 1):
                bnd = i + 1
                mi_loss += self.mi_Discs[i](sub_emb[:, :bnd * self.p.gcn_dim], sub_emb[:, bnd * self.p.gcn_dim:(bnd + 1) * self.p.gcn_dim])
            return mi_loss

        def loss_dependence_club_b(sub_emb):
            mi_loss = 0.0
            cnt = 0
            for i in range(self.p.num_factors):
                for j in range(i + 1, self.p.num_factors):
                    mi_loss += self.mi_Discs[cnt](sub_emb[:, i * self.p.gcn_dim:(i + 1) * self.p.gcn_dim], sub_emb[:, j * self.p.gcn_dim:(j + 1) * self.p.gcn_dim])
                    cnt += 1
            return mi_loss

        def DistanceCorrelation(tensor_1, tensor_2):
            channel = tensor_1.shape[0]
            zeros = torch.zeros(channel, channel)
            zero = torch.zeros(1)
            tensor_1, tensor_2 = tensor_1.unsqueeze(-1), tensor_2.unsqueeze(-1)
            """cul distance matrix"""
            a_, b_ = torch.matmul(tensor_1, tensor_1.t()) * 2, torch.matmul(tensor_2, tensor_2.t()) * 2
            tensor_1_square, tensor_2_square = tensor_1 ** 2, tensor_2 ** 2
            a, b = torch.sqrt(torch.max(tensor_1_square - a_ + tensor_1_square.t(), zeros) + 1e-08), torch.sqrt(torch.max(tensor_2_square - b_ + tensor_2_square.t(), zeros) + 1e-08)
            """cul distance correlation"""
            A = a - a.mean(dim=0, keepdim=True) - a.mean(dim=1, keepdim=True) + a.mean()
            B = b - b.mean(dim=0, keepdim=True) - b.mean(dim=1, keepdim=True) + b.mean()
            dcov_AB = torch.sqrt(torch.max((A * B).sum() / channel ** 2, zero) + 1e-08)
            dcov_AA = torch.sqrt(torch.max((A * A).sum() / channel ** 2, zero) + 1e-08)
            dcov_BB = torch.sqrt(torch.max((B * B).sum() / channel ** 2, zero) + 1e-08)
            return dcov_AB / torch.sqrt(dcov_AA * dcov_BB + 1e-08)
        if self.p.mi_method == 'club_s':
            mi_loss = loss_dependence_club_s(sub_emb)
        elif self.p.mi_method == 'club_b':
            mi_loss = loss_dependence_club_b(sub_emb)
        elif self.p.mi_method == 'hisc':
            mi_loss = loss_dependence_hisc(sub_emb, self.p.num_factors, self.p.gcn_dim)
        elif self.p.mi_method == 'dist':
            cor = 0.0
            for i in range(self.p.num_factors):
                for j in range(i + 1, self.p.num_factors):
                    cor += DistanceCorrelation(sub_emb[:, i * self.p.gcn_dim:(i + 1) * self.p.gcn_dim], sub_emb[:, j * self.p.gcn_dim:(j + 1) * self.p.gcn_dim])
            return cor
        else:
            raise NotImplementedError
        return mi_loss

    def forward_base(self, sub, rel, drop1, drop2, mode):
        if not self.p.no_enc:
            x = self.act(self.pca(self.init_embed)).view(-1, self.p.num_factors, self.p.gcn_dim)
            r = self.init_rel
            for conv in self.conv_ls:
                x, r = conv(x, r, mode)
                x = drop1(x)
        else:
            x = self.init_embed
            r = self.init_rel
            x = drop1(x)
        sub_emb = torch.index_select(x, 0, sub)
        rel_emb = torch.index_select(self.init_rel, 0, rel).repeat(1, self.p.num_factors)
        mi_loss = 0.0
        sub_emb = sub_emb.view(-1, self.p.gcn_dim * self.p.num_factors)
        mi_loss = self.mi_cal(sub_emb)
        return sub_emb, rel_emb, x, mi_loss

    def test_base(self, sub, rel, drop1, drop2, mode):
        if not self.p.no_enc:
            x = self.act(self.pca(self.init_embed)).view(-1, self.p.num_factors, self.p.gcn_dim)
            r = self.init_rel
            for conv in self.conv_ls:
                x, r = conv(x, r, mode)
                x = drop1(x)
        else:
            x = self.init_embed.view(-1, self.p.num_factors, self.p.gcn_dim)
            r = self.init_rel
            x = drop1(x)
        sub_emb = torch.index_select(x, 0, sub)
        rel_emb = torch.index_select(self.init_rel, 0, rel).repeat(1, self.p.num_factors)
        return sub_emb, rel_emb, x, 0.0


class DisenKGAT_TransE(CapsuleBase):

    def __init__(self, edge_index, edge_type, params=None):
        super(self.__class__, self).__init__(edge_index, edge_type, params.num_rel, params)
        self.drop = torch.nn.Dropout(self.p.hid_drop)
        self.rel_weight = self.conv_ls[-1].rel_weight
        gamma_init = torch.FloatTensor([self.p.init_gamma])
        if not self.p.fix_gamma:
            self.register_parameter('gamma', Parameter(gamma_init))

    def lld_best(self, sub, rel):
        return self.lld_bst(sub, rel, self.drop)

    def forward(self, sub, rel, neg_ents=None, mode='train'):
        if mode == 'train' and self.p.mi_train:
            sub_emb, rel_emb, all_ent, corr = self.forward_base(sub, rel, self.drop, self.drop, mode)
            sub_emb = sub_emb.view(-1, self.p.num_factors, self.p.gcn_dim)
        else:
            sub_emb, rel_emb, all_ent, corr = self.test_base(sub, rel, self.drop, self.drop, mode)
        rel_weight = torch.index_select(self.rel_weight, 0, rel)
        rel_emb = rel_emb.view(-1, self.p.num_factors, self.p.gcn_dim)
        sub_emb = sub_emb.view(-1, self.p.num_factors, self.p.gcn_dim)
        if self.p.score_method == 'dot_rel':
            sub_rel_emb = sub_emb * rel_weight
            rel_emb = rel_emb
            attention = self.leakyrelu(torch.einsum('bkf,bkf->bk', [sub_rel_emb, rel_emb]))
        elif self.p.score_method == 'dot_sub':
            sub_rel_emb = sub_emb
            attention = self.leakyrelu(torch.einsum('bkf,bkf->bk', [sub_rel_emb, rel_emb]))
        elif self.p.score_method == 'cat_rel':
            sub_rel_emb = sub_emb * rel_weight
            attention = self.leakyrelu(self.fc_a(torch.cat([sub_rel_emb, rel_emb], dim=2)).squeeze())
        elif self.p.score_method == 'cat_sub':
            sub_rel_emb = sub_emb
            attention = self.leakyrelu(self.fc_a(torch.cat([sub_rel_emb, rel_emb], dim=2)).squeeze())
        elif self.p.score_method == 'learn':
            att_rel = torch.index_select(self.fc_att, 0, rel)
            attention = self.leakyrelu(att_rel)
        attention = nn.Softmax(dim=-1)(attention)
        obj_emb = sub_emb + rel_emb
        if self.p.gamma_method == 'ada':
            x = self.gamma - torch.norm(obj_emb.unsqueeze(1) - all_ent, p=1, dim=3).transpose(1, 2)
        elif self.p.gamma_method == 'norm':
            x2 = torch.sum(obj_emb * obj_emb, dim=-1)
            y2 = torch.sum(all_ent * all_ent, dim=-1)
            xy = torch.einsum('bkf,nkf->bkn', [obj_emb, all_ent])
            x = self.gamma - (x2.unsqueeze(2) + y2.t() - 2 * xy)
        elif self.p.gamma_method == 'fix':
            x = self.p.gamma - torch.norm(obj_emb.unsqueeze(1) - all_ent, p=1, dim=3).transpose(1, 2)
        if self.p.score_order == 'before':
            x = torch.einsum('bk,bkn->bn', [attention, x])
            pred = torch.sigmoid(x)
        elif self.p.score_order == 'after':
            x = torch.sigmoid(x)
            pred = torch.einsum('bk,bkn->bn', [attention, x])
            pred = torch.clamp(pred, min=0.0, max=1.0)
        return pred, corr


class DisenKGAT_DistMult(CapsuleBase):

    def __init__(self, edge_index, edge_type, params=None):
        super(self.__class__, self).__init__(edge_index, edge_type, params.num_rel, params)
        self.drop = torch.nn.Dropout(self.p.hid_drop)
        self.rel_weight = self.conv_ls[-1].rel_weight

    def lld_best(self, sub, rel):
        return self.lld_bst(sub, rel, self.drop)

    def forward(self, sub, rel, neg_ents=None, mode='train'):
        if mode == 'train' and self.p.mi_train:
            sub_emb, rel_emb, all_ent, corr = self.forward_base(sub, rel, self.drop, self.drop, mode)
            sub_emb = sub_emb.view(-1, self.p.num_factors, self.p.gcn_dim)
        else:
            sub_emb, rel_emb, all_ent, corr = self.test_base(sub, rel, self.drop, self.drop, mode)
        rel_weight = torch.index_select(self.rel_weight, 0, rel)
        rel_emb = rel_emb.view(-1, self.p.num_factors, self.p.gcn_dim)
        sub_emb = sub_emb.view(-1, self.p.num_factors, self.p.gcn_dim)
        if self.p.score_method == 'dot_rel':
            sub_rel_emb = sub_emb * rel_weight
            rel_emb = rel_emb
            attention = self.leakyrelu(torch.einsum('bkf,bkf->bk', [sub_rel_emb, rel_emb]))
        elif self.p.score_method == 'dot_sub':
            sub_rel_emb = sub_emb
            attention = self.leakyrelu(torch.einsum('bkf,bkf->bk', [sub_rel_emb, rel_emb]))
        elif self.p.score_method == 'cat_rel':
            sub_rel_emb = sub_emb * rel_weight
            attention = self.leakyrelu(self.fc_a(torch.cat([sub_rel_emb, rel_emb], dim=2)).squeeze())
        elif self.p.score_method == 'cat_sub':
            sub_rel_emb = sub_emb
            attention = self.leakyrelu(self.fc_a(torch.cat([sub_rel_emb, rel_emb], dim=2)).squeeze())
        elif self.p.score_method == 'learn':
            att_rel = torch.index_select(self.fc_att, 0, rel)
            attention = self.leakyrelu(att_rel)
        attention = nn.Softmax(dim=-1)(attention)
        obj_emb = sub_emb * rel_emb
        x = torch.einsum('bkf,nkf->bkn', [obj_emb, all_ent])
        x += self.bias.expand_as(x)
        if self.p.score_order == 'before':
            x = torch.einsum('bk,bkn->bn', [attention, x])
            pred = torch.sigmoid(x)
        elif self.p.score_order == 'after':
            x = torch.sigmoid(x)
            pred = torch.einsum('bk,bkn->bn', [attention, x])
            pred = torch.clamp(pred, min=0.0, max=1.0)
        return pred, corr


class DisenKGAT_ConvE(CapsuleBase):

    def __init__(self, edge_index, edge_type, params=None):
        super(self.__class__, self).__init__(edge_index, edge_type, params.num_rel, params)
        self.embed_dim = self.p.embed_dim
        self.bn0 = torch.nn.BatchNorm2d(1)
        self.bn1 = torch.nn.BatchNorm2d(self.p.num_filt)
        self.bn2 = torch.nn.BatchNorm1d(self.embed_dim)
        self.hidden_drop = torch.nn.Dropout(self.p.hid_drop)
        self.hidden_drop2 = torch.nn.Dropout(self.p.hid_drop2)
        self.feature_drop = torch.nn.Dropout(self.p.feat_drop)
        self.m_conv1 = torch.nn.Conv2d(1, out_channels=self.p.num_filt, kernel_size=(self.p.ker_sz, self.p.ker_sz), stride=1, padding=0, bias=self.p.bias)
        flat_sz_h = int(2 * self.p.k_w) - self.p.ker_sz + 1
        flat_sz_w = self.p.k_h - self.p.ker_sz + 1
        self.flat_sz = flat_sz_h * flat_sz_w * self.p.num_filt
        self.fc = torch.nn.Linear(self.flat_sz, self.embed_dim)
        if self.p.score_method.startswith('cat'):
            self.fc_a = nn.Linear(2 * self.p.gcn_dim, 1)
        elif self.p.score_method == 'learn':
            self.fc_att = get_param((2 * self.p.num_rel, self.p.num_factors))
        self.rel_weight = self.conv_ls[-1].rel_weight

    def concat(self, e1_embed, rel_embed):
        e1_embed = e1_embed.view(-1, 1, self.embed_dim)
        rel_embed = rel_embed.view(-1, 1, self.embed_dim)
        stack_inp = torch.cat([e1_embed, rel_embed], 1)
        stack_inp = torch.transpose(stack_inp, 2, 1).reshape((-1, 1, 2 * self.p.k_w, self.p.k_h))
        return stack_inp

    def lld_best(self, sub, rel):
        return self.lld_bst(sub, rel, self.hidden_drop)

    def forward(self, sub, rel, neg_ents=None, mode='train'):
        if mode == 'train' and self.p.mi_train:
            sub_emb, rel_emb, all_ent, corr = self.forward_base(sub, rel, self.hidden_drop, self.feature_drop, mode)
        else:
            sub_emb, rel_emb, all_ent, corr = self.test_base(sub, rel, self.hidden_drop, self.feature_drop, mode)
        sub_emb = sub_emb.view(-1, self.p.gcn_dim)
        rel_emb = rel_emb.view(-1, self.p.gcn_dim)
        all_ent = all_ent.view(-1, self.p.num_factors, self.p.gcn_dim)
        stk_inp = self.concat(sub_emb, rel_emb)
        x = self.bn0(stk_inp)
        x = self.m_conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_drop(x)
        x = x.view(-1, self.flat_sz)
        x = self.fc(x)
        x = self.hidden_drop2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = x.view(-1, self.p.num_factors, self.p.gcn_dim)
        rel_weight = torch.index_select(self.rel_weight, 0, rel)
        rel_emb = rel_emb.view(-1, self.p.num_factors, self.p.gcn_dim)
        sub_emb = sub_emb.view(-1, self.p.num_factors, self.p.gcn_dim)
        if self.p.score_method == 'dot_rel':
            sub_rel_emb = sub_emb * rel_weight
            attention = self.leakyrelu(torch.einsum('bkf,bkf->bk', [sub_rel_emb, rel_emb]))
        elif self.p.score_method == 'dot_sub':
            sub_rel_emb = sub_emb
            attention = self.leakyrelu(torch.einsum('bkf,bkf->bk', [sub_rel_emb, rel_emb]))
        elif self.p.score_method == 'cat_rel':
            sub_rel_emb = sub_emb * rel_weight
            attention = self.leakyrelu(self.fc_a(torch.cat([sub_rel_emb, rel_emb], dim=2)).squeeze())
        elif self.p.score_method == 'cat_sub':
            sub_rel_emb = sub_emb
            attention = self.leakyrelu(self.fc_a(torch.cat([sub_rel_emb, rel_emb], dim=2)).squeeze())
        elif self.p.score_method == 'learn':
            att_rel = torch.index_select(self.fc_att, 0, rel)
            attention = self.leakyrelu(att_rel)
        attention = nn.Softmax(dim=-1)(attention)
        x = torch.einsum('bkf,nkf->bkn', [x, all_ent])
        x += self.bias.expand_as(x)
        if self.p.score_order == 'before':
            x = torch.einsum('bk,bkn->bn', [attention, x])
            pred = torch.sigmoid(x)
        elif self.p.score_order == 'after':
            x = torch.sigmoid(x)
            pred = torch.einsum('bk,bkn->bn', [attention, x])
            pred = torch.clamp(pred, min=0.0, max=1.0)
        return pred, corr


class DisenKGAT_InteractE(CapsuleBase):

    def __init__(self, edge_index, edge_type, params=None):
        super(self.__class__, self).__init__(edge_index, edge_type, params.num_rel, params)
        self.inp_drop = torch.nn.Dropout(self.p.iinp_drop)
        self.feature_map_drop = torch.nn.Dropout2d(self.p.ifeat_drop)
        self.hidden_drop = torch.nn.Dropout(self.p.ihid_drop)
        self.hidden_drop_gcn = torch.nn.Dropout(0)
        self.bn0 = torch.nn.BatchNorm2d(self.p.iperm)
        flat_sz_h = self.p.ik_h
        flat_sz_w = 2 * self.p.ik_w
        self.padding = 0
        self.bn1 = torch.nn.BatchNorm2d(self.p.inum_filt * self.p.iperm)
        self.flat_sz = flat_sz_h * flat_sz_w * self.p.inum_filt * self.p.iperm
        self.bn2 = torch.nn.BatchNorm1d(self.p.embed_dim)
        self.fc = torch.nn.Linear(self.flat_sz, self.p.embed_dim)
        self.chequer_perm = self.get_chequer_perm()
        if self.p.score_method.startswith('cat'):
            self.fc_a = nn.Linear(2 * self.p.gcn_dim, 1)
        elif self.p.score_method == 'learn':
            self.fc_att = get_param((2 * self.p.num_rel, self.p.num_factors))
        self.rel_weight = self.conv_ls[-1].rel_weight
        self.register_parameter('bias', Parameter(torch.zeros(self.p.num_ent)))
        self.register_parameter('conv_filt', Parameter(torch.zeros(self.p.inum_filt, 1, self.p.iker_sz, self.p.iker_sz)))
        xavier_normal_(self.conv_filt)

    def circular_padding_chw(self, batch, padding):
        upper_pad = batch[..., -padding:, :]
        lower_pad = batch[..., :padding, :]
        temp = torch.cat([upper_pad, batch, lower_pad], dim=2)
        left_pad = temp[..., -padding:]
        right_pad = temp[..., :padding]
        padded = torch.cat([left_pad, temp, right_pad], dim=3)
        return padded

    def lld_best(self, sub, rel):
        return self.lld_bst(sub, rel, self.inp_drop)

    def forward(self, sub, rel, neg_ents=None, mode='train'):
        if mode == 'train' and self.p.mi_train:
            sub_emb, rel_emb, all_ent, corr = self.forward_base(sub, rel, self.inp_drop, self.hidden_drop_gcn, mode)
        else:
            sub_emb, rel_emb, all_ent, corr = self.test_base(sub, rel, self.inp_drop, self.hidden_drop_gcn, mode)
        sub_emb = sub_emb.view(-1, self.p.gcn_dim)
        rel_emb = rel_emb.view(-1, self.p.gcn_dim)
        all_ent = all_ent.view(-1, self.p.num_factors, self.p.gcn_dim)
        comb_emb = torch.cat([sub_emb, rel_emb], dim=1)
        chequer_perm = comb_emb[:, self.chequer_perm]
        stack_inp = chequer_perm.reshape((-1, self.p.iperm, 2 * self.p.ik_w, self.p.ik_h))
        stack_inp = self.bn0(stack_inp)
        x = stack_inp
        x = self.circular_padding_chw(x, self.p.iker_sz // 2)
        x = F.conv2d(x, self.conv_filt.repeat(self.p.iperm, 1, 1, 1), padding=self.padding, groups=self.p.iperm)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_map_drop(x)
        x = x.view(-1, self.flat_sz)
        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = x.view(-1, self.p.num_factors, self.p.gcn_dim)
        rel_weight = torch.index_select(self.rel_weight, 0, rel)
        rel_emb = rel_emb.view(-1, self.p.num_factors, self.p.gcn_dim)
        sub_emb = sub_emb.view(-1, self.p.num_factors, self.p.gcn_dim)
        if self.p.score_method == 'dot_rel':
            sub_rel_emb = sub_emb * rel_weight
            attention = self.leakyrelu(torch.einsum('bkf,bkf->bk', [sub_rel_emb, rel_emb]))
        elif self.p.score_method == 'dot_sub':
            sub_rel_emb = sub_emb
            attention = self.leakyrelu(torch.einsum('bkf,bkf->bk', [sub_rel_emb, rel_emb]))
        elif self.p.score_method == 'cat_rel':
            sub_rel_emb = sub_emb * rel_weight
            attention = self.leakyrelu(self.fc_a(torch.cat([sub_rel_emb, rel_emb], dim=2)).squeeze())
        elif self.p.score_method == 'cat_sub':
            sub_rel_emb = sub_emb
            attention = self.leakyrelu(self.fc_a(torch.cat([sub_rel_emb, rel_emb], dim=2)).squeeze())
        elif self.p.score_method == 'learn':
            att_rel = torch.index_select(self.fc_att, 0, rel)
            attention = self.leakyrelu(att_rel)
        attention = nn.Softmax(dim=-1)(attention)
        if self.p.strategy == 'one_to_n' or neg_ents is None:
            x = torch.einsum('bkf,nkf->bkn', [x, all_ent])
            x += self.bias.expand_as(x)
        else:
            x = torch.mul(x.unsqueeze(1), all_ent[neg_ents]).sum(dim=-1)
            x += self.bias[neg_ents]
        if self.p.score_order == 'before':
            x = torch.einsum('bk,bkn->bn', [attention, x])
            pred = torch.sigmoid(x)
        elif self.p.score_order == 'after':
            x = torch.sigmoid(x)
            pred = torch.einsum('bk,bkn->bn', [attention, x])
            pred = torch.clamp(pred, min=0.0, max=1.0)
        return pred, corr

    def get_chequer_perm(self):
        """
        Function to generate the chequer permutation required for InteractE model

        Parameters
        ----------

        Returns
        -------

        """
        ent_perm = np.int32([np.random.permutation(self.p.embed_dim) for _ in range(self.p.iperm)])
        rel_perm = np.int32([np.random.permutation(self.p.embed_dim) for _ in range(self.p.iperm)])
        comb_idx = []
        for k in range(self.p.iperm):
            temp = []
            ent_idx, rel_idx = 0, 0
            for i in range(self.p.ik_h):
                for j in range(self.p.ik_w):
                    if k % 2 == 0:
                        if i % 2 == 0:
                            temp.append(ent_perm[k, ent_idx])
                            ent_idx += 1
                            temp.append(rel_perm[k, rel_idx] + self.p.embed_dim)
                            rel_idx += 1
                        else:
                            temp.append(rel_perm[k, rel_idx] + self.p.embed_dim)
                            rel_idx += 1
                            temp.append(ent_perm[k, ent_idx])
                            ent_idx += 1
                    elif i % 2 == 0:
                        temp.append(rel_perm[k, rel_idx] + self.p.embed_dim)
                        rel_idx += 1
                        temp.append(ent_perm[k, ent_idx])
                        ent_idx += 1
                    else:
                        temp.append(ent_perm[k, ent_idx])
                        ent_idx += 1
                        temp.append(rel_perm[k, rel_idx] + self.p.embed_dim)
                        rel_idx += 1
            comb_idx.append(temp)
        chequer_perm = torch.LongTensor(np.int32(comb_idx))
        return chequer_perm


class ExpressGNN(BaseModel):

    @classmethod
    def build_model_from_args(cls, args, hg):
        return cls(args=args, latent_dim=args.embedding_size - args.gcn_free_size, free_dim=args.gcn_free_size, device=args.device, load_method=args.load_method, rule_list=args.rule_list, rule_weights_learning=args.rule_weights_learning, graph=hg, PRED_DICT=args.PRED_DICT, slice_dim=args.slice_dim, transductive=args.trans == 1)

    def __init__(self, args, graph, latent_dim, free_dim, device, load_method, rule_list, rule_weights_learning, PRED_DICT, num_hops=5, num_layers=2, slice_dim=5, transductive=True):
        """

        Parameters
        ----------
        graph: knowledge graph
        latent_dim: embedding_size - gcn_free_size
        free_dim: gcn_free_size
        device: device
        load_method: Factorized Posterior's load method, use args to get
        rule_list: MLN's rules, should come from dataset
        rule_weights_learning: MLN's args, should come from args
        num_hops: number of hops of GCN
        num_layers: number of layers of GCN
        slice_dim: Used by Factorized Posterior
        transductive: Used by GCN
        """
        super(ExpressGNN, self).__init__()
        self.graph = graph
        self.latent_dim = latent_dim
        self.free_dim = free_dim
        self.num_hops = num_hops
        self.num_layers = num_layers
        self.PRED_DICT = PRED_DICT
        self.args = args
        self.num_ents = graph.num_ents
        self.num_rels = graph.num_rels
        self.num_nodes = graph.num_nodes
        self.num_edges = graph.num_edges
        self.num_edge_types = len(graph.edge_type2idx)
        self.xent_loss = F.binary_cross_entropy_with_logits
        self.load_method = load_method
        self.num_rels = graph.num_rels
        self.ent2idx = graph.ent2idx
        self.rel2idx = graph.rel2idx
        self.idx2rel = graph.idx2rel
        self.num_ents = self.graph.num_ents
        self.ent_embeds = nn.Embedding(self.num_ents, self.args.embedding_size)
        self.ents = torch.arange(self.num_ents)
        self.edge2node_in, self.edge2node_out, self.node_degree, self.edge_type_masks, self.edge_direction_masks = self.gen_edge2node_mapping()
        self.node_feat, self.const_nodes = self.prepare_node_feature(graph, transductive=transductive)
        if not transductive:
            self.node_feat_dim = 1 + self.num_rels
        else:
            self.node_feat_dim = self.num_ents + self.num_rels
        self.init_node_linear = nn.Linear(self.node_feat_dim, latent_dim, bias=False)
        for param in self.init_node_linear.parameters():
            param.requires_grad = False
        self.node_feat = self.node_feat
        self.const_nodes = self.const_nodes
        self.edge2node_in = self.edge2node_in
        self.edge2node_out = self.edge2node_out
        self.edge_type_masks = [mask for mask in self.edge_type_masks]
        self.edge_direction_masks = [mask for mask in self.edge_direction_masks]
        self.MLPs = nn.ModuleList()
        for _ in range(self.num_hops):
            self.MLPs.append(MLP(input_size=self.latent_dim, num_layers=self.num_layers, hidden_size=self.latent_dim, output_size=self.latent_dim))
        self.edge_type_W = nn.ModuleList()
        for _ in range(self.num_edge_types):
            ml_edge_type = nn.ModuleList()
            for _ in range(self.num_hops):
                ml_hop = nn.ModuleList()
                for _ in range(2):
                    ml_hop.append(nn.Linear(latent_dim, latent_dim, bias=False))
                ml_edge_type.append(ml_hop)
            self.edge_type_W.append(ml_edge_type)
        self.const_nodes_free_params = nn.Parameter(nn.init.kaiming_uniform_(torch.zeros(self.num_ents, free_dim)))
        if load_method == 1:
            self.params_u_R = nn.ModuleList()
            self.params_W_R = nn.ModuleList()
            self.params_V_R = nn.ModuleList()
            for idx in range(self.num_rels):
                rel = self.idx2rel[idx]
                num_args = self.PRED_DICT[rel].num_args
                self.params_W_R.append(nn.Bilinear(num_args * args.embedding_size, num_args * args.embedding_size, slice_dim, bias=False))
                self.params_V_R.append(nn.Linear(num_args * args.embedding_size, slice_dim, bias=True))
                self.params_u_R.append(nn.Linear(slice_dim, 1, bias=False))
        elif load_method == 0:
            self.params_u_R = nn.ParameterList()
            self.params_W_R = nn.ModuleList()
            self.params_V_R = nn.ModuleList()
            self.params_b_R = nn.ParameterList()
            for idx in range(self.num_rels):
                rel = self.idx2rel[idx]
                num_args = self.PRED_DICT[rel].num_args
                self.params_u_R.append(nn.Parameter(nn.init.kaiming_uniform_(torch.zeros(slice_dim, 1)).view(-1)))
                self.params_W_R.append(nn.Bilinear(num_args * args.embedding_size, num_args * args.embedding_size, slice_dim, bias=False))
                self.params_V_R.append(nn.Linear(num_args * args.embedding_size, slice_dim, bias=False))
                self.params_b_R.append(nn.Parameter(nn.init.kaiming_uniform_(torch.zeros(slice_dim, 1)).view(-1)))
        self.rule_weights_lin = nn.Linear(len(rule_list), 1, bias=False)
        self.num_rules = len(rule_list)
        self.soft_logic = False
        self.alpha_table = nn.Parameter(torch.tensor([(10.0) for _ in range(len(self.PRED_DICT))], requires_grad=True))
        self.predname2ind = dict(e for e in zip(self.PRED_DICT.keys(), range(len(self.PRED_DICT))))
        if rule_weights_learning == 0:
            self.rule_weights_lin.weight.data = torch.tensor([[rule.weight for rule in rule_list]], dtype=torch.float)
            None
        else:
            self.rule_weights_lin.weight = nn.Parameter(torch.tensor([[rule.weight for rule in rule_list]], dtype=torch.float))
            None

    def gcn_forward(self, batch_data):
        if self.args.use_gcn == 0:
            node_embeds = self.ent_embeds(self.ents)
            return node_embeds
        else:
            node_embeds = self.init_node_linear(self.node_feat)
            hop = 0
            hidden = node_embeds
            while hop < self.num_hops:
                node_aggregate = torch.zeros_like(hidden)
                for edge_type in set(self.graph.edge_types):
                    for direction in range(2):
                        W = self.edge_type_W[edge_type][hop][direction]
                        W_nodes = W(hidden)
                        nodes_attached_on_edges_out = torch.gather(W_nodes, 0, self.edge2node_out)
                        nodes_attached_on_edges_out *= self.edge_type_masks[edge_type].view(-1, 1)
                        nodes_attached_on_edges_out *= self.edge_direction_masks[direction].view(-1, 1)
                        node_aggregate.scatter_add_(0, self.edge2node_in, nodes_attached_on_edges_out)
                hidden = self.MLPs[hop](hidden + node_aggregate)
                hop += 1
            read_out_const_nodes_embed = torch.cat((hidden[self.const_nodes], self.const_nodes_free_params), dim=1)
            return read_out_const_nodes_embed

    def posterior_forward(self, latent_vars, node_embeds, batch_mode=False, fast_mode=False, fast_inference_mode=False):
        """
        compute posterior probabilities of specified latent variables

        :param latent_vars:
            list of latent variables (i.e. unobserved facts)
        :param node_embeds:
            node embeddings
        :return:
            n-dim vector, probability of corresponding latent variable being True

        Parameters
        ----------
        fast_inference_mode
        fast_mode
        batch_mode
        """
        if fast_inference_mode:
            assert self.load_method == 1
            samples = latent_vars
            scores = []
            for ind in range(len(samples)):
                pred_name, pred_sample = samples[ind]
                rel_idx = self.rel2idx[pred_name]
                sample_mat = torch.tensor(pred_sample, dtype=torch.long)
                sample_query = torch.cat([node_embeds[sample_mat[:, 0]], node_embeds[sample_mat[:, 1]]], dim=1)
                sample_score = self.params_u_R[rel_idx](torch.tanh(self.params_W_R[rel_idx](sample_query, sample_query) + self.params_V_R[rel_idx](sample_query))).view(-1)
                scores.append(torch.sigmoid(sample_score))
            return scores
        elif fast_mode:
            assert self.load_method == 1
            samples, neg_mask, latent_mask, obs_var, neg_var = latent_vars
            scores = []
            obs_probs = []
            neg_probs = []
            a = []
            for pred_mask in neg_mask:
                a.append(pred_mask[1])
            pos_mask_mat = torch.tensor(a)
            pos_mask_mat = pos_mask_mat
            neg_mask_mat = (pos_mask_mat == 0).type(torch.float)
            latent_mask_mat = torch.tensor([pred_mask[1] for pred_mask in latent_mask], dtype=torch.float)
            obs_mask_mat = (latent_mask_mat == 0).type(torch.float)
            for ind in range(len(samples)):
                pred_name, pred_sample = samples[ind]
                _, obs_sample = obs_var[ind]
                _, neg_sample = neg_var[ind]
                rel_idx = self.rel2idx[pred_name]
                sample_mat = torch.tensor(pred_sample, dtype=torch.long)
                obs_mat = torch.tensor(obs_sample, dtype=torch.long)
                neg_mat = torch.tensor(neg_sample, dtype=torch.long)
                sample_mat = torch.cat([sample_mat, obs_mat, neg_mat], dim=0)
                sample_query = torch.cat([node_embeds[sample_mat[:, 0]], node_embeds[sample_mat[:, 1]]], dim=1)
                sample_score = self.params_u_R[rel_idx](torch.tanh(self.params_W_R[rel_idx](sample_query, sample_query) + self.params_V_R[rel_idx](sample_query))).view(-1)
                var_prob = sample_score[len(pred_sample):]
                obs_prob = var_prob[:len(obs_sample)]
                neg_prob = var_prob[len(obs_sample):]
                sample_score = sample_score[:len(pred_sample)]
                scores.append(sample_score)
                obs_probs.append(obs_prob)
                neg_probs.append(neg_prob)
            score_mat = torch.stack(scores, dim=0)
            score_mat = torch.sigmoid(score_mat)
            pos_score = (1 - score_mat) * pos_mask_mat
            neg_score = score_mat * neg_mask_mat
            potential = 1 - ((pos_score + neg_score) * latent_mask_mat + obs_mask_mat).prod(dim=0)
            obs_mat = torch.cat(obs_probs, dim=0)
            if obs_mat.size(0) == 0:
                obs_loss = 0.0
            else:
                obs_loss = self.xent_loss(obs_mat, torch.ones_like(obs_mat), reduction='sum')
            neg_mat = torch.cat(neg_probs, dim=0)
            if neg_mat.size(0) != 0:
                obs_loss += self.xent_loss(obs_mat, torch.zeros_like(neg_mat), reduction='sum')
            obs_loss /= obs_mat.size(0) + neg_mat.size(0) + 1e-06
            return potential, (score_mat * latent_mask_mat).view(-1), obs_loss
        elif batch_mode:
            assert self.load_method == 1
            pred_name, x_mat, invx_mat, sample_mat = latent_vars
            rel_idx = self.rel2idx[pred_name]
            x_mat = torch.tensor(x_mat, dtype=torch.long)
            invx_mat = torch.tensor(invx_mat, dtype=torch.long)
            sample_mat = torch.tensor(sample_mat, dtype=torch.long)
            tail_query = torch.cat([node_embeds[x_mat[:, 0]], node_embeds[x_mat[:, 1]]], dim=1)
            head_query = torch.cat([node_embeds[invx_mat[:, 0]], node_embeds[invx_mat[:, 1]]], dim=1)
            true_query = torch.cat([node_embeds[sample_mat[:, 0]], node_embeds[sample_mat[:, 1]]], dim=1)
            tail_score = self.params_u_R[rel_idx](torch.tanh(self.params_W_R[rel_idx](tail_query, tail_query) + self.params_V_R[rel_idx](tail_query))).view(-1)
            head_score = self.params_u_R[rel_idx](torch.tanh(self.params_W_R[rel_idx](head_query, head_query) + self.params_V_R[rel_idx](head_query))).view(-1)
            true_score = self.params_u_R[rel_idx](torch.tanh(self.params_W_R[rel_idx](true_query, true_query) + self.params_V_R[rel_idx](true_query))).view(-1)
            probas_tail = torch.sigmoid(tail_score)
            probas_head = torch.sigmoid(head_score)
            probas_true = torch.sigmoid(true_score)
            return probas_tail, probas_head, probas_true
        else:
            assert self.load_method == 0
            probas = torch.zeros(len(latent_vars))
            for i in range(len(latent_vars)):
                rel, args = latent_vars[i]
                args_embed = torch.cat([node_embeds[self.ent2idx[arg]] for arg in args], 0)
                rel_idx = self.rel2idx[rel]
                score = self.params_u_R[rel_idx].dot(torch.tanh(self.params_W_R[rel_idx](args_embed, args_embed) + self.params_V_R[rel_idx](args_embed) + self.params_b_R[rel_idx]))
                proba = torch.sigmoid(score)
                probas[i] = proba
            return probas

    def mln_forward(self, neg_mask_ls_ls, latent_var_inds_ls_ls, observed_rule_cnts, posterior_prob, flat_list, observed_vars_ls_ls):
        """
        compute the MLN potential given the posterior probability of latent variables
        :param neg_mask_ls_ls:

        :return:

        Parameters
        ----------
        flat_list
        posterior_prob
        observed_vars_ls_ls
        latent_var_inds_ls_ls
        observed_rule_cnts
        """
        scores = torch.zeros(self.num_rules, dtype=torch.float, device=self.args.device)
        pred_ind_flat_list = []
        if self.soft_logic:
            pred_name_ls = [e[0] for e in flat_list]
            pred_ind_flat_list = [self.predname2ind[pred_name] for pred_name in pred_name_ls]
        for i in range(len(neg_mask_ls_ls)):
            neg_mask_ls = neg_mask_ls_ls[i]
            latent_var_inds_ls = latent_var_inds_ls_ls[i]
            observed_vars_ls = observed_vars_ls_ls[i]
            for j in range(len(neg_mask_ls)):
                latent_neg_mask, observed_neg_mask = neg_mask_ls[j]
                latent_var_inds = latent_var_inds_ls[j]
                observed_vars = observed_vars_ls[j]
                z_probs = posterior_prob[latent_var_inds].unsqueeze(0)
                z_probs = torch.cat([1 - z_probs, z_probs], dim=0)
                cartesian_prod = z_probs[:, 0]
                for j in range(1, z_probs.shape[1]):
                    cartesian_prod = torch.ger(cartesian_prod, z_probs[:, j])
                    cartesian_prod = cartesian_prod.view(-1)
                view_ls = [(2) for _ in range(len(latent_neg_mask))]
                cartesian_prod = cartesian_prod.view(*[view_ls])
                if self.soft_logic:
                    obs_vals = [e[0] for e in observed_vars]
                    pred_names = [e[1] for e in observed_vars]
                    pred_inds = [self.predname2ind[pn] for pn in pred_names]
                    alpha = self.alpha_table[pred_inds]
                    act_alpha = torch.sigmoid(alpha)
                    obs_neg_flag = [(1 if observed_vars[i] != observed_neg_mask[i] else 0) for i in range(len(observed_vars))]
                    tn_obs_neg_flag = torch.tensor(obs_neg_flag, dtype=torch.float)
                    val = torch.abs(1 - torch.tensor(obs_vals, dtype=torch.float) - act_alpha)
                    obs_score = torch.abs(tn_obs_neg_flag - val)
                    inds = product(*[[0, 1] for _ in range(len(latent_neg_mask))])
                    pred_inds = [pred_ind_flat_list[i] for i in latent_var_inds]
                    alpha = self.alpha_table[pred_inds]
                    act_alpha = torch.sigmoid(alpha)
                    tn_latent_neg_mask = torch.tensor(latent_neg_mask, dtype=torch.float)
                    for ind in inds:
                        val = torch.abs(1 - torch.tensor(ind, dtype=torch.float) - act_alpha)
                        val = torch.abs(tn_latent_neg_mask - val)
                        cartesian_prod[tuple(ind)] *= torch.max(torch.cat([val, obs_score], dim=0))
                elif sum(observed_neg_mask) == 0:
                    cartesian_prod[tuple(latent_neg_mask)] = 0.0
                scores[i] += cartesian_prod.sum()
            scores[i] += observed_rule_cnts[i]
        return self.rule_weights_lin(scores)

    def gen_edge2node_mapping(self):
        """
        A GCN's function
        Returns
        -------

        """
        ei = 0
        edge_idx = 0
        edge2node_in = torch.zeros(self.num_edges * 2, dtype=torch.long)
        edge2node_out = torch.zeros(self.num_edges * 2, dtype=torch.long)
        node_degree = torch.zeros(self.num_nodes)
        edge_type_masks = []
        for _ in range(self.num_edge_types):
            edge_type_masks.append(torch.zeros(self.num_edges * 2))
        edge_direction_masks = []
        for _ in range(2):
            edge_direction_masks.append(torch.zeros(self.num_edges * 2))
        for ni, nj in torch.as_tensor(self.graph.edge_pairs):
            edge_type = self.graph.edge_types[edge_idx]
            edge_idx += 1
            edge2node_in[ei] = nj
            edge2node_out[ei] = ni
            node_degree[ni] += 1
            edge_type_masks[edge_type][ei] = 1
            edge_direction_masks[0][ei] = 1
            ei += 1
            edge2node_in[ei] = ni
            edge2node_out[ei] = nj
            node_degree[nj] += 1
            edge_type_masks[edge_type][ei] = 1
            edge_direction_masks[1][ei] = 1
            ei += 1
        edge2node_in = edge2node_in.view(-1, 1).expand(-1, self.latent_dim)
        edge2node_out = edge2node_out.view(-1, 1).expand(-1, self.latent_dim)
        node_degree = node_degree.view(-1, 1)
        return edge2node_in, edge2node_out, node_degree, edge_type_masks, edge_direction_masks

    def weight_update(self, neg_mask_ls_ls, latent_var_inds_ls_ls, observed_rule_cnts, posterior_prob, flat_list, observed_vars_ls_ls):
        """
        A MLN's Function
        Parameters
        ----------
        neg_mask_ls_ls
        latent_var_inds_ls_ls
        observed_rule_cnts
        posterior_prob
        flat_list
        observed_vars_ls_ls

        Returns
        -------

        """
        closed_wolrd_potentials = torch.zeros(self.num_rules, dtype=torch.float)
        pred_ind_flat_list = []
        if self.soft_logic:
            pred_name_ls = [e[0] for e in flat_list]
            pred_ind_flat_list = [self.predname2ind[pred_name] for pred_name in pred_name_ls]
        for i in range(len(neg_mask_ls_ls)):
            neg_mask_ls = neg_mask_ls_ls[i]
            latent_var_inds_ls = latent_var_inds_ls_ls[i]
            observed_vars_ls = observed_vars_ls_ls[i]
            for j in range(len(neg_mask_ls)):
                latent_neg_mask, observed_neg_mask = neg_mask_ls[j]
                latent_var_inds = latent_var_inds_ls[j]
                observed_vars = observed_vars_ls[j]
                has_pos_atom = False
                for val in (observed_neg_mask + latent_neg_mask):
                    if val == 1:
                        has_pos_atom = True
                        break
                if has_pos_atom:
                    closed_wolrd_potentials[i] += 1
                z_probs = posterior_prob[latent_var_inds].unsqueeze(0)
                z_probs = torch.cat([1 - z_probs, z_probs], dim=0)
                cartesian_prod = z_probs[:, 0]
                for j in range(1, z_probs.shape[1]):
                    cartesian_prod = torch.ger(cartesian_prod, z_probs[:, j])
                    cartesian_prod = cartesian_prod.view(-1)
                view_ls = [(2) for _ in range(len(latent_neg_mask))]
                cartesian_prod = cartesian_prod.view(*[view_ls])
                if self.soft_logic:
                    obs_vals = [e[0] for e in observed_vars]
                    pred_names = [e[1] for e in observed_vars]
                    pred_inds = [self.predname2ind[pn] for pn in pred_names]
                    alpha = self.alpha_table[pred_inds]
                    act_alpha = torch.sigmoid(alpha)
                    obs_neg_flag = [(1 if observed_vars[i] != observed_neg_mask[i] else 0) for i in range(len(observed_vars))]
                    tn_obs_neg_flag = torch.tensor(obs_neg_flag, dtype=torch.float)
                    val = torch.abs(1 - torch.tensor(obs_vals, dtype=torch.float) - act_alpha)
                    obs_score = torch.abs(tn_obs_neg_flag - val)
                    inds = product(*[[0, 1] for _ in range(len(latent_neg_mask))])
                    pred_inds = [pred_ind_flat_list[i] for i in latent_var_inds]
                    alpha = self.alpha_table[pred_inds]
                    act_alpha = torch.sigmoid(alpha)
                    tn_latent_neg_mask = torch.tensor(latent_neg_mask, dtype=torch.float)
                    for ind in inds:
                        val = torch.abs(1 - torch.tensor(ind, dtype=torch.float) - act_alpha)
                        val = torch.abs(tn_latent_neg_mask - val)
                        cartesian_prod[tuple(ind)] *= torch.max(torch.cat([val, obs_score], dim=0))
                elif sum(observed_neg_mask) == 0:
                    cartesian_prod[tuple(latent_neg_mask)] = 0.0
            weight_grad = closed_wolrd_potentials
            return weight_grad

    def gen_index(self, facts, predicates, dataset):
        rel2idx = dict()
        idx_rel = 0
        for rel in sorted(predicates.keys()):
            if rel not in rel2idx:
                rel2idx[rel] = idx_rel
                idx_rel += 1
        idx2rel = dict(zip(rel2idx.values(), rel2idx.keys()))
        ent2idx = dict()
        idx_ent = 0
        for type_name in sorted(dataset.const_sort_dict.keys()):
            for const in dataset.const_sort_dict[type_name]:
                ent2idx[const] = idx_ent
                idx_ent += 1
        idx2ent = dict(zip(ent2idx.values(), ent2idx.keys()))
        node2idx = ent2idx.copy()
        idx_node = len(node2idx)
        for rel in sorted(facts.keys()):
            for fact in sorted(list(facts[rel])):
                val, args = fact
                if (rel, args) not in node2idx:
                    node2idx[rel, args] = idx_node
                    idx_node += 1
        idx2node = dict(zip(node2idx.values(), node2idx.keys()))
        return ent2idx, idx2ent, rel2idx, idx2rel, node2idx, idx2node

    def gen_edge_type(self):
        edge_type2idx = dict()
        num_args_set = set()
        for rel in self.PRED_DICT:
            num_args = self.PRED_DICT[rel].num_args
            num_args_set.add(num_args)
        idx = 0
        for num_args in sorted(list(num_args_set)):
            for pos_code in product(['0', '1'], repeat=num_args):
                if '1' in pos_code:
                    edge_type2idx[0, ''.join(pos_code)] = idx
                    idx += 1
                    edge_type2idx[1, ''.join(pos_code)] = idx
                    idx += 1
        return edge_type2idx

    def gen_graph(self, facts, predicates, dataset):
        """
            generate directed knowledge graph, where each edge is from subject to object
        :param facts:
            dictionary of facts
        :param predicates:
            dictionary of predicates
        :param dataset:
            dataset object
        :return:
            graph object, entity to index, index to entity, relation to index, index to relation
        """
        g = nx.Graph()
        ent2idx, idx2ent, rel2idx, idx2rel, node2idx, idx2node = self.gen_index(facts, predicates, dataset)
        edge_type2idx = self.gen_edge_type()
        for node_idx in idx2node:
            g.add_node(node_idx)
        for rel in facts.keys():
            for fact in facts[rel]:
                val, args = fact
                fact_node_idx = node2idx[rel, args]
                for arg in args:
                    pos_code = ''.join([('%d' % (arg == v)) for v in args])
                    g.add_edge(fact_node_idx, node2idx[arg], edge_type=edge_type2idx[val, pos_code])
        return g, edge_type2idx, ent2idx, idx2ent, rel2idx, idx2rel, node2idx, idx2node

    def prepare_node_feature(self, graph, transductive=True):
        if transductive:
            node_feat = torch.zeros(graph.num_nodes, graph.num_ents + graph.num_rels)
            const_nodes = []
            for i in graph.idx2node:
                if isinstance(graph.idx2node[i], str):
                    const_nodes.append(i)
                    node_feat[i][i] = 1
                elif isinstance(graph.idx2node[i], tuple):
                    rel, args = graph.idx2node[i]
                    node_feat[i][graph.num_ents + graph.rel2idx[rel]] = 1
        else:
            node_feat = torch.zeros(graph.num_nodes, 1 + graph.num_rels)
            const_nodes = []
            for i in graph.idx2node:
                if isinstance(graph.idx2node[i], str):
                    node_feat[i][0] = 1
                    const_nodes.append(i)
                elif isinstance(graph.idx2node[i], tuple):
                    rel, args = graph.idx2node[i]
                    node_feat[i][1 + graph.rel2idx[rel]] = 1
        return node_feat, torch.LongTensor(const_nodes)


class GATConv_norm(nn.Module):

    def __init__(self, in_feats, out_feats, num_heads, feat_drop=0.0, attn_drop=0.0, negative_slope=0.2, residual=False, activation=None, allow_zero_in_degree=False, bias=True, norm=None):
        super(GATConv_norm, self).__init__()
        self._num_heads = num_heads
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree
        if isinstance(in_feats, tuple):
            self.fc_src = nn.Linear(self._in_src_feats, out_feats * num_heads, bias=False)
            self.fc_dst = nn.Linear(self._in_dst_feats, out_feats * num_heads, bias=False)
        else:
            self.fc = nn.Linear(self._in_src_feats, out_feats * num_heads, bias=False)
        self.attn_l = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
        self.attn_r = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(size=(num_heads * out_feats,)))
        else:
            self.register_buffer('bias', None)
        if residual:
            if self._in_dst_feats != out_feats * num_heads:
                self.res_fc = nn.Linear(self._in_dst_feats, num_heads * out_feats, bias=False)
            else:
                self.res_fc = nn.Identity()
        else:
            self.register_buffer('res_fc', None)
        self.reset_parameters()
        self.activation = activation
        self.norm = norm
        if norm is not None:
            self.norm = norm(num_heads * out_feats)

    def reset_parameters(self):
        """

        Description
        -----------
        Reinitialize learnable parameters.

        Note
        ----
        The fc weights :math:`W^{(l)}` are initialized using Glorot uniform initialization.
        The attention weights are using xavier initialization method.
        """
        gain = nn.init.calculate_gain('relu')
        if hasattr(self, 'fc'):
            nn.init.xavier_normal_(self.fc.weight, gain=gain)
        else:
            nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)

    def set_allow_zero_in_degree(self, set_value):
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat, get_attention=False):
        """
            feat: Tensor of shape [num_nodes,feat_dim]
        """
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise RuntimeError('There are 0-in-degree nodes in the graph, output for those nodes will be invalid. This is harmful for some applications, causing silent performance regression. Adding self-loop on the input graph by calling `g = dgl.add_self_loop(g)` will resolve the issue. Setting ``allow_zero_in_degree`` to be `True` when constructing this module will suppress the check and let the code run.')
            if isinstance(feat, tuple):
                src_prefix_shape = feat[0].shape[:-1]
                dst_prefix_shape = feat[1].shape[:-1]
                h_src = self.feat_drop(feat[0])
                h_dst = self.feat_drop(feat[1])
                if not hasattr(self, 'fc_src'):
                    feat_src = self.fc(h_src).view(*src_prefix_shape, self._num_heads, self._out_feats)
                    feat_dst = self.fc(h_dst).view(*dst_prefix_shape, self._num_heads, self._out_feats)
                else:
                    feat_src = self.fc_src(h_src).view(*src_prefix_shape, self._num_heads, self._out_feats)
                    feat_dst = self.fc_dst(h_dst).view(*dst_prefix_shape, self._num_heads, self._out_feats)
            else:
                src_prefix_shape = dst_prefix_shape = feat.shape[:-1]
                h_src = h_dst = self.feat_drop(feat)
                feat_src = feat_dst = self.fc(h_src).view(*src_prefix_shape, self._num_heads, self._out_feats)
                if graph.is_block:
                    feat_dst = feat_src[:graph.number_of_dst_nodes()]
                    h_dst = h_dst[:graph.number_of_dst_nodes()]
                    dst_prefix_shape = (graph.number_of_dst_nodes(),) + dst_prefix_shape[1:]
            el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
            er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)
            graph.srcdata.update({'ft': feat_src, 'el': el})
            graph.dstdata.update({'er': er})
            graph.apply_edges(fn.u_add_v('el', 'er', 'e'))
            e = self.leaky_relu(graph.edata.pop('e'))
            graph.edata['a'] = self.attn_drop(edge_softmax(graph, e))
            graph.update_all(fn.u_mul_e('ft', 'a', 'm'), fn.sum('m', 'ft'))
            rst = graph.dstdata['ft']
            if self.bias is not None:
                rst = rst + self.bias.view(*((1,) * len(dst_prefix_shape)), self._num_heads, self._out_feats)
            if self.res_fc is not None:
                resval = self.res_fc(h_dst).view(*dst_prefix_shape, -1, self._out_feats)
                rst = rst + resval
            rst = rst.flatten(1)
            if self.norm is not None:
                rst = self.norm(rst)
            if self.activation:
                rst = self.activation(rst)
            if get_attention:
                return rst, graph.edata['a']
            else:
                return rst


class HANLayer(nn.Module):

    def __init__(self, num_metapaths, in_dim, out_dim, num_heads, feat_drop, attn_drop, negative_slope, residual, activation, norm):
        super(HANLayer, self).__init__()
        self.gat_layers = nn.ModuleList()
        for i in range(num_metapaths):
            self.gat_layers.append(GATConv_norm(in_dim, out_dim, num_heads, feat_drop, attn_drop, negative_slope, residual, activation, norm=norm))
        self.semantic_attention = SemanticAttention(in_size=out_dim * num_heads)

    def forward(self, gs, h):
        semantic_embeddings = []
        for i, new_g in enumerate(gs):
            semantic_embeddings.append(self.gat_layers[i](new_g, h).flatten(1))
        semantic_embeddings = torch.stack(semantic_embeddings, dim=1)
        out, att_mp = self.semantic_attention(semantic_embeddings)
        return out, att_mp


class FedHGNN(BaseModel):

    @classmethod
    def build_model_from_args(cls, args):
        return cls(meta_paths=args.meta_paths, in_size=args.in_size, hidden_size=args.hidden_size, out_size=args.out_size, num_heads=args.num_heads, dropout=args.dropout)

    def __init__(self, meta_paths, in_size, hidden_size, out_size, num_heads, dropout):
        super(FedHGNN, self).__init__()
        num_heads = json.loads(num_heads)
        self.layers = nn.ModuleList()
        self.layers.append(HANLayer(meta_paths, in_size, hidden_size, num_heads[0], dropout))
        for l in range(1, len(num_heads)):
            self.layers.append(HANLayer(meta_paths, hidden_size * num_heads[l - 1], hidden_size, num_heads[l], dropout))
        self.predict = nn.Linear(hidden_size * num_heads[-1], out_size)

    def forward(self, g, h):
        for gnn in self.layers:
            h = gnn(g, h)
        return self.predict(h)


TASK_REGISTRY = {}


SUPPORTED_TASKS = {'coldstart_recommendation': 'openhgnn.tasks.coldstart_recommendation', 'KTN_trainer': 'openhgnn.tasks.KTN', 'demo': 'openhgnn.tasks.demo', 'node_classification': 'openhgnn.tasks.node_classification', 'link_prediction': 'openhgnn.tasks.link_prediction', 'recommendation': 'openhgnn.tasks.recommendation', 'embedding': 'openhgnn.tasks.embedding', 'edge_classification': 'openhgnn.tasks.edge_classification', 'hypergraph': 'openhgnn.tasks.hypergraph', 'meirec': 'openhgnn.tasks.meirec', 'pretrain': 'openhgnn.tasks.pretrain', 'abnorm_event_detection': 'openhgnn.tasks.AbnormEventDetection', 'DSSL_trainer': 'openhgnn.tasks.DSSL_task', 'NBF_link_prediction': 'openhgnn.tasks.link_prediction', 'Ingram': 'openhgnn.tasks.Ingram', 'DisenKGAT_link_prediction': 'openhgnn.tasks.link_prediction'}


def try_import_task(task):
    if task not in TASK_REGISTRY:
        if task in SUPPORTED_TASKS:
            importlib.import_module(SUPPORTED_TASKS[task])
        else:
            None
            return False
    return True


def build_task(args):
    if not try_import_task(args.task):
        exit(1)
    return TASK_REGISTRY[args.task](args)


def get_nodes_dict(hg):
    n_dict = {}
    for n in hg.ntypes:
        n_dict[n] = hg.num_nodes(n)
    return n_dict


class BaseFlow(ABC):
    candidate_optimizer = {'Adam': torch.optim.Adam, 'SGD': torch.optim.SGD, 'Adadelta': torch.optim.Adadelta}

    def __init__(self, args):
        """

        Parameters
        ----------
        args

        Attributes
        -------------
        evaluate_interval: int
            the interval of evaluation in validation
        """
        super(BaseFlow, self).__init__()
        self.evaluator = None
        self.evaluate_interval = getattr(args, 'evaluate_interval', 1)
        if hasattr(args, 'model_path'):
            self._checkpoint = args.model_path
        elif hasattr(args, '_checkpoint'):
            self._checkpoint = os.path.join(args._checkpoint, f'{args.model_name}_{args.dataset_name}.pt')
        elif hasattr(args, 'load_from_pretrained'):
            self._checkpoint = os.path.join(args.output_dir, f'{args.model_name}_{args.dataset_name}_{args.task}.pt')
        else:
            self._checkpoint = None
        if not hasattr(args, 'HGB_results_path') and args.dataset_name[:3] == 'HGB':
            args.HGB_results_path = os.path.join(args.output_dir, '{}_{}_{}.txt'.format(args.model_name, args.dataset_name[5:], args.seed))
        self.use_distributed = args.use_distributed
        args.test_flag = getattr(args, 'test_flag', True)
        args.prediction_flag = getattr(args, 'prediction_flag', False)
        args.use_uva = getattr(args, 'use_uva', False)
        self.args = args
        self.logger = self.args.logger
        self.model_name = args.model_name
        self.model = args.model
        self.device = args.device
        self.task = build_task(args)
        self.max_epoch = args.max_epoch
        self.optimizer = None
        if self.model_name in ['SIAN', 'MeiREC', 'ExpressGNN', 'Ingram', 'RedGNN', 'RedGNNT', 'AdapropI', 'AdapropT', 'RedGNNT', 'Grail', 'ComPILE', 'DisenKGAT', 'MetaHIN']:
            return
        if self.model_name == 'Ingram':
            return
        if self.args.use_uva:
            self.hg = self.task.get_graph()
        else:
            self.hg = self.task.get_graph()
        self.args.meta_paths_dict = self.task.dataset.meta_paths_dict
        self.patience = args.patience
        self.loss_fn = self.task.get_loss_fn()

    def preprocess(self):
        """
        Every trainerflow should run the preprocess_feature if you want to get a feature preprocessing.
        The Parameters in input_feature will be added into optimizer and input_feature will be added into the model.

        Attributes
        -----------
        input_feature : HeteroFeature
            It will return the processed feature if call it.

        """
        if hasattr(self.args, 'activation'):
            if hasattr(self.args.activation, 'weight'):
                import torch.nn as nn
                act = nn.PReLU()
            else:
                act = self.args.activation
        else:
            act = None
        if hasattr(self.args, 'feat'):
            pass
        else:
            self.args.feat = 0
        self.feature_preprocess(act)
        self.optimizer.add_param_group({'params': self.input_feature.parameters()})
        self.model.add_module('input_feature', self.input_feature)
        self.load_from_pretrained()

    def feature_preprocess(self, act):
        """
        Feat
            0, 1 ,2
        Node feature
            1 node type & more than 1 node types
            no feature

        Returns
        -------

        """
        if self.hg.ndata.get('h', {}) == {} or self.args.feat == 2:
            if self.hg.ndata.get('h', {}) == {}:
                self.logger.feature_info('Assign embedding as features, because hg.ndata is empty.')
            else:
                self.logger.feature_info('feat2, drop features!')
                self.hg.ndata.pop('h')
            self.input_feature = HeteroFeature({}, get_nodes_dict(self.hg), self.args.hidden_dim, act=act)
        elif self.args.feat == 0:
            self.input_feature = self.init_feature(act)
        elif self.args.feat == 1:
            if self.args.task != 'node_classification':
                self.logger.feature_info("'feat 1' is only for node classification task, set feat 0!")
                self.input_feature = self.init_feature(act)
            else:
                h_dict = self.hg.ndata.pop('h')
                self.logger.feature_info('feat1, preserve target nodes!')
                self.input_feature = HeteroFeature({self.category: h_dict[self.category]}, get_nodes_dict(self.hg), self.args.hidden_dim, act=act)

    def init_feature(self, act):
        self.logger.feature_info('Feat is 0, nothing to do!')
        if isinstance(self.hg.ndata['h'], dict):
            input_feature = HeteroFeature(self.hg.ndata['h'], get_nodes_dict(self.hg), self.args.hidden_dim, act=act)
        elif isinstance(self.hg.ndata['h'], torch.Tensor):
            input_feature = HeteroFeature({self.hg.ntypes[0]: self.hg.ndata['h']}, get_nodes_dict(self.hg), self.args.hidden_dim, act=act)
        return input_feature

    @abstractmethod
    def train(self):
        pass

    def _full_train_step(self):
        """
        Train with a full_batch graph
        """
        raise NotImplementedError

    def _mini_train_step(self):
        """
        Train with a mini_batch seed nodes graph
        """
        raise NotImplementedError

    def _full_test_step(self):
        """
        Test with a full_batch graph
        """
        raise NotImplementedError

    def _mini_test_step(self):
        """
        Test with a mini_batch seed nodes graph
        """
        raise NotImplementedError

    def load_from_pretrained(self):
        if hasattr(self.args, 'load_from_pretrained') and self.args.load_from_pretrained:
            try:
                ck_pt = torch.load(self._checkpoint)
                self.model.load_state_dict(ck_pt)
                self.logger.info('[Load Model] Load model from pretrained model:' + self._checkpoint)
            except FileNotFoundError:
                self.logger.info("[Load Model] Do not load the model from pretrained, {} doesn't exists".format(self._checkpoint))

    def save_checkpoint(self):
        if self._checkpoint and hasattr(self.model, '_parameters()'):
            torch.save(self.model.state_dict(), self._checkpoint)


class NSLoss(nn.Module):

    def __init__(self, num_nodes, num_sampled, embedding_size):
        super(NSLoss, self).__init__()
        self.num_nodes = num_nodes
        self.num_sampled = num_sampled
        self.embedding_size = embedding_size
        self.weights = Parameter(torch.FloatTensor(num_nodes, embedding_size))
        self.sample_weights = F.normalize(torch.Tensor([((math.log(k + 2) - math.log(k + 1)) / math.log(num_nodes + 1)) for k in range(num_nodes)]), dim=0)
        self.reset_parameters()

    def reset_parameters(self):
        self.weights.data.normal_(std=1.0 / math.sqrt(self.embedding_size))

    def forward(self, input, embs, label):
        n = input.shape[0]
        log_target = torch.log(torch.sigmoid(torch.sum(torch.mul(embs, self.weights[label]), 1)))
        negs = torch.multinomial(self.sample_weights, self.num_sampled * n, replacement=True).view(n, self.num_sampled)
        noise = torch.neg(self.weights[negs])
        sum_log_sampled = torch.sum(torch.log(torch.sigmoid(torch.bmm(noise, embs.unsqueeze(2)))), 1).squeeze()
        loss = log_target + sum_log_sampled
        return -loss.sum() / n


class NeighborSampler(object):

    def __init__(self, hg, ntypes, num_nodes, device):
        self.hg = hg
        self.ntypes = ntypes
        self.num_nodes = num_nodes
        self.sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
        self.device = device

    def build_hetero_graph(self, heads, tails):
        edge_dict = {}
        num_ntypes = len(self.ntypes)
        for i in range(num_ntypes):
            for j in range(num_ntypes):
                edge = self.ntypes[i], self.ntypes[i] + '-' + self.ntypes[j], self.ntypes[j]
                mask = (heads[1] == i) & (tails[1] == j)
                edge_dict[edge] = heads[0][mask], tails[0][mask]
        hg = dgl.heterograph(edge_dict, self.num_nodes)
        return hg

    def sample_from_item_pairs(self, heads, tails, neg_tails):
        pos_graph = self.build_hetero_graph(heads, tails)
        neg_graph = self.build_hetero_graph(heads, neg_tails)
        pos_graph, neg_graph = dgl.compact_graphs([pos_graph, neg_graph])
        pos_nodes = pos_graph.ndata[dgl.NID]
        seed_nodes = pos_nodes
        blocks = self.sampler.sample_blocks(self.hg, seed_nodes, exclude_eids=None)
        return pos_graph, neg_graph, blocks


MODEL_REGISTRY = {}


SUPPORTED_MODELS = {'MHGCN': 'openhgnn.models.MHGCN', 'BPHGNN': 'openhgnn.models.BPHGNN', 'MetaHIN': 'openhgnn.models.MetaHIN', 'HGA': 'openhgnn.models.HGA', 'RHINE': 'openhgnn.models.RHINE', 'FedHGNN': 'openhgnn.models.FedHGNN', 'SIAN': 'openhgnn.models.SIAN', 'CompGCN': 'openhgnn.models.CompGCN', 'HetGNN': 'openhgnn.models.HetGNN', 'HMPNN': 'openhgnn.models.HMPNN', 'RGCN': 'openhgnn.models.RGCN', 'RGAT': 'openhgnn.models.RGAT', 'RSHN': 'openhgnn.models.RSHN', 'Metapath2vec': 'openhgnn.models.SkipGram', 'HERec': 'openhgnn.models.SkipGram', 'HAN': 'openhgnn.models.HAN', 'RoHe': 'openhgnn.models.RoHe', 'HeCo': 'openhgnn.models.HeCo', 'HGT': 'openhgnn.models.HGT', 'GTN': 'openhgnn.models.GTN_sparse', 'fastGTN': 'openhgnn.models.fastGTN', 'MHNF': 'openhgnn.models.MHNF', 'MAGNN': 'openhgnn.models.MAGNN', 'HeGAN': 'openhgnn.models.HeGAN', 'NSHE': 'openhgnn.models.NSHE', 'NARS': 'openhgnn.models.NARS', 'RHGNN': 'openhgnn.models.RHGNN', 'HPN': 'openhgnn.models.HPN', 'KGCN': 'openhgnn.models.KGCN', 'SLiCE': 'openhgnn.models.SLiCE', 'HGSL': 'openhgnn.models.HGSL', 'GCN': 'space4hgnn.homo_models.GCN', 'GAT': 'space4hgnn.homo_models.GAT', 'homo_GNN': 'openhgnn.models.homo_GNN', 'general_HGNN': 'openhgnn.models.general_HGNN', 'HDE': 'openhgnn.models.HDE', 'SimpleHGN': 'openhgnn.models.SimpleHGN', 'GATNE-T': 'openhgnn.models.GATNE', 'HetSANN': 'openhgnn.models.HetSANN', 'HGAT': 'openhgnn.models.HGAT', 'ieHGCN': 'openhgnn.models.ieHGCN', 'TransE': 'openhgnn.models.TransE', 'TransH': 'openhgnn.models.TransH', 'TransR': 'openhgnn.models.TransR', 'TransD': 'openhgnn.models.TransD', 'GIE': 'openhgnn.models.GIE', 'GIN': 'openhgnn.models.GIN', 'Rsage': 'openhgnn.models.Rsage', 'Mg2vec': 'openhgnn.models.MG2vec', 'DHNE': 'openhgnn.models.DHNE', 'DiffMG': 'openhgnn.models.DiffMG', 'MeiREC': 'openhgnn.models.MeiREC', 'HGNN_AC': 'openhgnn.models.HGNN_AC', 'AEHCL': 'openhgnn.models.AEHCL', 'KGAT': 'openhgnn.models.KGAT', 'SHGP': 'openhgnn.models.ATT_HGCN', 'DSSL': 'openhgnn.models.DSSL', 'HGCL': 'openhgnn.models.HGCL', 'lightGCN': 'openhgnn.models.lightGCN', 'SeHGNN': 'openhgnn.models.SeHGNN', 'Grail': 'openhgnn.models.Grail', 'ComPILE': 'openhgnn.models.ComPILE', 'AdapropT': 'openhgnn.models.AdapropT', 'AdapropI': 'openhgnn.models.AdapropI', 'LTE': 'openhgnn.models.LTE', 'LTE_Transe': 'openhgnn.models.LTE_Transe', 'SACN': 'openhgnn.models.SACN', 'ExpressGNN': 'openhgnn.models.ExpressGNN', 'NBF': 'openhgnn.models.NBF', 'Ingram': 'openhgnn.models.Ingram', 'RedGNN': 'openhgnn.models.RedGNN', 'RedGNNT': 'openhgnn.models.RedGNNT'}


def try_import_model(model):
    if model not in MODEL_REGISTRY:
        if model in SUPPORTED_MODELS:
            importlib.import_module(SUPPORTED_MODELS[model])
        else:
            None
            return False
    return True


def build_model(model):
    if isinstance(model, nn.Module):
        if not hasattr(model, 'build_model_from_args'):

            def build_model_from_args(args, hg):
                return model
            model.build_model_from_args = build_model_from_args
        return model
    if not try_import_model(model):
        exit(1)
    return MODEL_REGISTRY[model]


def generate_pairs_parallel(walks, skip_window=None, layer_id=None):
    pairs = []
    for walk in walks:
        walk = walk.tolist()
        for i in range(len(walk)):
            for j in range(1, skip_window + 1):
                if i - j >= 0:
                    pairs.append((walk[i], walk[i - j], layer_id))
                if i + j < len(walk):
                    pairs.append((walk[i], walk[i + j], layer_id))
    return pairs


def generate_pairs(all_walks, window_size, num_workers):
    start_time = time.time()
    None
    pool = multiprocessing.Pool(processes=num_workers)
    pairs = []
    skip_window = window_size // 2
    for layer_id, walks in enumerate(all_walks):
        block_num = len(walks) // num_workers
        if block_num > 0:
            walks_list = [walks[i * block_num:min((i + 1) * block_num, len(walks))] for i in range(num_workers)]
        else:
            walks_list = [walks]
        tmp_result = pool.map(partial(generate_pairs_parallel, skip_window=skip_window, layer_id=layer_id), walks_list)
        pairs += reduce(lambda x, y: x + y, tmp_result)
    pool.close()
    end_time = time.time()
    None
    return np.array([list(pair) for pair in set(pairs)])


class GATNE(BaseFlow):

    def __init__(self, args):
        super(GATNE, self).__init__(args)
        self.model = build_model(self.model).build_model_from_args(self.args, self.hg)
        self.train_pairs = None
        self.train_dataloader = None
        self.nsloss = None
        self.neighbor_sampler = None
        self.orig_val_hg = self.task.val_hg
        self.orig_test_hg = self.task.test_hg
        self.preprocess()

    def preprocess(self):
        assert len(self.hg.ntypes) == 1
        bidirected_hg = dgl.to_bidirected(dgl.to_simple(self.hg))
        all_walks = []
        for etype in self.hg.etypes:
            nodes = torch.unique(bidirected_hg.edges(etype=etype)[0]).repeat(self.args.rw_walks)
            traces, types = dgl.sampling.random_walk(bidirected_hg, nodes, metapath=[etype] * (self.args.rw_length - 1))
            all_walks.append(traces)
        self.train_pairs = generate_pairs(all_walks, self.args.window_size, self.args.num_workers)
        self.neighbor_sampler = NeighborSampler(bidirected_hg, [self.args.neighbor_samples])
        self.train_dataloader = torch.utils.data.DataLoader(self.train_pairs, batch_size=self.args.batch_size, collate_fn=self.neighbor_sampler.sample, shuffle=True, num_workers=self.args.num_workers, pin_memory=True)
        self.nsloss = NSLoss(self.hg.num_nodes(), self.args.neg_size, self.args.dim)
        self.optimizer = torch.optim.Adam([{'params': self.model.parameters()}, {'params': self.nsloss.parameters()}], lr=self.args.learning_rate)
        return

    def train(self):
        best_score = 0
        patience = 0
        for self.epoch in range(self.args.max_epoch):
            self._full_train_step()
            cur_score = self._full_test_step()
            if cur_score > best_score:
                best_score = cur_score
                patience = 0
            else:
                patience += 1
                if patience > self.args.patience:
                    self.logger.train_info(f'Early Stop!\tEpoch:{self.epoch:03d}.')
                    break

    def _full_train_step(self):
        self.model.train()
        random.shuffle(self.train_pairs)
        data_iter = tqdm(self.train_dataloader, desc='epoch %d' % self.epoch, total=(len(self.train_pairs) + (self.args.batch_size - 1)) // self.args.batch_size)
        avg_loss = 0.0
        for i, (block, head_invmap, tails, block_types) in enumerate(data_iter):
            self.optimizer.zero_grad()
            block_types = block_types
            embs = self.model(block[0])[head_invmap]
            embs = embs.gather(1, block_types.view(-1, 1, 1).expand(embs.shape[0], 1, embs.shape[2]))[:, 0]
            loss = self.nsloss(block[0].dstdata[dgl.NID][head_invmap], embs, tails)
            loss.backward()
            self.optimizer.step()
            avg_loss += loss.item()
            post_fix = {'epoch': self.epoch, 'iter': i, 'avg_loss': avg_loss / (i + 1), 'loss': loss.item()}
            data_iter.set_postfix(post_fix)

    def _full_test_step(self):
        self.model.eval()
        final_model = dict(zip(self.hg.etypes, [th.empty(self.hg.num_nodes(), self.args.dim) for _ in range(len(self.hg.etypes))]))
        for i in tqdm(range(self.hg.num_nodes()), desc='Evaluating...'):
            train_inputs = torch.tensor([i for _ in range(len(self.hg.etypes))]).unsqueeze(1)
            train_types = torch.tensor(list(range(len(self.hg.etypes)))).unsqueeze(1)
            pairs = torch.cat((train_inputs, train_inputs, train_types), dim=1)
            train_blocks, train_invmap, fake_tails, train_types = self.neighbor_sampler.sample(pairs)
            node_emb = self.model(train_blocks[0])[train_invmap]
            node_emb = node_emb.gather(1, train_types.view(-1, 1, 1).expand(node_emb.shape[0], 1, node_emb.shape[2]))[:, 0]
            for j in range(len(self.hg.etypes)):
                final_model[self.hg.etypes[j]][i] = node_emb[j].detach()
        metric = {}
        score = []
        for etype in self.hg.etypes:
            self.task.val_hg = dgl.edge_type_subgraph(self.orig_val_hg, [etype])
            self.task.test_hg = dgl.edge_type_subgraph(self.orig_test_hg, [etype])
            for split in ['test', 'valid']:
                n_embedding = {self.hg.ntypes[0]: final_model[etype]}
                res = self.task.evaluate(n_embedding=n_embedding, mode=split)
                metric[split] = res
                if split == 'valid':
                    score.append(res.get('roc_auc'))
            self.logger.train_info(etype + self.logger.metric2str(metric))
        avg_score = sum(score) / len(score)
        return avg_score


class Artanh(th.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        x = x.clamp(-1 + 1e-05, 1 - 1e-05)
        ctx.save_for_backward(x)
        dtype = x.dtype
        x = x.double()
        return th.log_(1 + x).sub_(th.log_(1 - x)).mul_(0.5)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return grad_output / (1 - input ** 2)


class GIE(BaseModel):

    @classmethod
    def build_model_from_args(cls, args, hg):
        return cls(args, hg)

    def __init__(self, args, hg):
        super(GIE, self).__init__()
        self.device = args.device
        self.ent_num = hg.num_nodes()
        self.rel_num = len(hg.canonical_etypes)
        self.dim = args.hidden_dim
        self.margin = args.margin
        self.dis_norm = args.dis_norm
        self.scale = th.Tensor([1.0 / np.sqrt(self.dim)]).double()
        self.n_emb = nn.Embedding(self.ent_num, self.dim)
        self.r_emb = nn.Embedding(self.rel_num, self.dim)
        self.r_diagE = nn.Embedding(self.rel_num, self.dim)
        self.r_diagH = nn.Embedding(self.rel_num, self.dim)
        self.r_diagS = nn.Embedding(self.rel_num, self.dim)
        self.context_vec = nn.Embedding(self.rel_num, self.dim)
        self.act = nn.Softmax(dim=1)
        self.c = nn.Parameter(th.ones((self.rel_num, 1), dtype=th.double), requires_grad=True)
        self.v = nn.Parameter(th.ones((self.rel_num, 1), dtype=th.double), requires_grad=True)
        self.u = nn.Parameter(th.ones((self.rel_num, 1), dtype=th.double), requires_grad=True)
        self.MIN_NORM = 1e-15
        nn.init.xavier_uniform_(self.n_emb.weight.data)
        nn.init.xavier_uniform_(self.r_emb.weight.data)
        nn.init.xavier_uniform_(self.r_diagE.weight.data)
        nn.init.xavier_uniform_(self.r_diagH.weight.data)
        nn.init.xavier_uniform_(self.r_diagS.weight.data)
        nn.init.xavier_uniform_(self.context_vec.weight.data)

    def forward(self, h, r, t):
        h = h
        r = r
        t = t
        if h.shape == th.Size([]):
            h = h.view(1).repeat(t.shape[0])
        if r.shape == th.Size([]):
            r = r.view(1).repeat(h.shape[0])
        if t.shape == th.Size([]):
            t = t.view(1).repeat(h.shape[0])
        h_emb = self.n_emb(h)
        r_emb = self.r_emb(r)
        t_emb = self.n_emb(t)
        c = F.softplus(self.c[r])
        r_exp0c = self.expmap0(r_emb, c)
        res_E = self.givens_rotations(self.r_diagE(r), h_emb).view((-1, 1, self.dim))
        v = F.softplus(self.v[r])
        h_exp0v = self.expmap0(h_emb, v)
        r_exp0v = self.expmap0(r_emb, v)
        lh_H = self.project(self.mobius_add(h_exp0v, r_exp0v, v), v)
        res_H = self.logmap0(self.givens_rotations(self.r_diagH(r), lh_H), v).view((-1, 1, self.dim))
        u = F.softplus(self.u[r])
        h_exp0u = self.expmap0(h_emb, u)
        r_exp0u = self.expmap0(r_emb, u)
        lh_S = self.project(self.mobius_add(h_exp0u, r_exp0u, u), u)
        res_S = self.logmap0(self.givens_rotations(self.r_diagS(r), lh_S), u).view((-1, 1, self.dim))
        cands = th.cat([res_E, res_H, res_S], dim=1)
        context_vec = self.context_vec(r).view((-1, 1, self.dim))
        att_W = self.act(th.sum(context_vec * cands * self.scale, dim=-1, keepdim=True))
        hr_emb = self.project(self.mobius_add(self.expmap0(th.sum(att_W * cands, dim=1), c), r_exp0c, c), c)
        return (self.similarity_score(hr_emb, t_emb, c) + self.margin).view(-1)

    def similarity_score(self, x, v, c):
        sqrt_c = c ** 0.5
        v_norm = th.norm(v, p=2, dim=-1, keepdim=True)
        xv = th.sum(x * v / v_norm, dim=-1, keepdim=True)
        gamma = self.tanh(sqrt_c * v_norm) / sqrt_c
        x2 = th.sum(x * x, dim=-1, keepdim=True)
        c1 = 1 - 2 * c * gamma * xv + c * gamma ** 2
        c2 = 1 - c * x2
        num = th.sqrt(c1 ** 2 * x2 + c2 ** 2 * gamma ** 2 - 2 * c1 * c2 * gamma * xv)
        denom = 1 - 2 * c * gamma * xv + c ** 2 * gamma ** 2 * x2
        pairwise_norm = num / denom.clamp_min(self.MIN_NORM)
        dist = self.artanh(sqrt_c * pairwise_norm)
        return -(2 * dist / sqrt_c) ** 2

    def tanh(self, x):
        return x.clamp(-15, 15).tanh()

    def artanh(self, x):
        return Artanh.apply(x)

    def givens_rotations(self, r, x):
        givens = r.view((r.shape[0], -1, 2))
        givens = givens / th.norm(givens, p=2, dim=-1, keepdim=True).clamp_min(1e-15)
        x = x.view((r.shape[0], -1, 2))
        x_rot = givens[:, :, 0:1] * x + givens[:, :, 1:] * th.cat((-x[:, :, 1:], x[:, :, 0:1]), dim=-1)
        return x_rot.view((r.shape[0], -1))

    def mobius_add(self, x, y, c):
        x2 = th.sum(x * x, dim=-1, keepdim=True)
        y2 = th.sum(y * y, dim=-1, keepdim=True)
        xy = th.sum(x * y, dim=-1, keepdim=True)
        num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
        denom = 1 + 2 * c * xy + c ** 2 * x2 * y2
        return num / denom.clamp_min(self.MIN_NORM)

    def expmap0(self, u, c):
        sqrt_c = c ** 0.5
        u_norm = u.norm(dim=-1, p=2, keepdim=True).clamp_min(self.MIN_NORM)
        temp = self.tanh(sqrt_c * u_norm) * u
        gamma_1 = temp / (sqrt_c * u_norm)
        return self.project(gamma_1, c)

    def logmap0(self, y, c):
        sqrt_c = c ** 0.5
        y_norm = y.norm(dim=-1, p=2, keepdim=True).clamp_min(self.MIN_NORM)
        return y / y_norm / sqrt_c * self.artanh(sqrt_c * y_norm)

    def project(self, x, c):
        norm = x.norm(dim=-1, p=2, keepdim=True).clamp_min(self.MIN_NORM)
        eps = 1e-05
        maxnorm = (1 - eps) / c ** 0.5
        cond = norm > maxnorm
        projected = x / norm * maxnorm
        return th.where(cond, projected, x)


class GIN(nn.Module):

    def __init__(self, g, in_dims, num_hidden, num_classes, num_layers, activation, dropout):
        super(GIN, self).__init__()
        self.g = g
        self.layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        self.act = torch.nn.ReLU()
        self.activation = activation
        self.fc_list = nn.ModuleList([nn.Linear(in_dim, num_hidden, bias=True) for in_dim in in_dims])
        for fc in self.fc_list:
            nn.init.xavier_normal_(fc.weight, gain=1.414)
        self.ginfunc = torch.nn.Sequential(torch.nn.Linear(num_hidden, num_hidden), self.act, torch.nn.Linear(num_hidden, num_hidden))
        self.GINlayer = dgl.nn.pytorch.conv.GINConv(apply_func=self.ginfunc, aggregator_type='sum')
        self.bn = torch.nn.BatchNorm1d(num_hidden)
        self.layers.append(self.GINlayer)
        self.bn_layers.append(self.bn)
        for i in range(num_layers - 1):
            self.layers.append(self.GINlayer)
            self.bn_layers.append(self.bn)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, features_list):
        h = []
        for fc, feature in zip(self.fc_list, features_list):
            h.append(fc(feature))
        h = torch.cat(h, 0)
        for i, layer in enumerate(self.layers):
            h = self.dropout(h)
            h = layer(self.g, h)
            h = self.activation(h)
            h = self.bn_layers[i](h)
        return h


class GINBase(nn.Module):

    def __init__(self, input_dim, output_dim, learn_eps):
        super(GINBase, self).__init__()
        mlp = MLP(input_dim, output_dim)
        self.ginlayer = GINConv(mlp, learn_eps=learn_eps)
        self.batch_norm = nn.BatchNorm1d(output_dim)

    def forward(self, g, h):
        h = self.ginlayer(g, h)
        h = self.batch_norm(h)
        h = F.relu(h)
        return h


class GINLayer(nn.Module):

    def __init__(self, input_dim, output_dim, rel_names, learn_eps, aggregate):
        super(GINLayer, self).__init__()
        self.conv = dglnn.HeteroGraphConv({rel: GINBase(input_dim, output_dim, learn_eps) for rel in rel_names}, aggregate)

    def forward(self, g, h_dict):
        h_dict = self.conv(g, h_dict)
        out_put = {}
        for n_type, h in h_dict.items():
            out_put[n_type] = h.squeeze()
        return out_put


class GTConv(nn.Module):
    """
        We conv each sub adjacency matrix :math:`A_{R_{i}}` to a combination adjacency matrix :math:`A_{1}`:

        .. math::
            A_{1} = conv\\left(A ; W_{c}\\right)=\\sum_{R_{i} \\in R} w_{R_{i}} A_{R_{i}}

        where :math:`R_i \\subseteq \\mathcal{R}` and :math:`W_{c}` is the weight of each relation matrix
    """

    def __init__(self, in_channels, out_channels, softmax_flag=True):
        super(GTConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = nn.Parameter(th.Tensor(out_channels, in_channels))
        self.softmax_flag = softmax_flag
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.weight, std=0.01)

    def forward(self, A):
        if self.softmax_flag:
            Filter = F.softmax(self.weight, dim=1)
        else:
            Filter = self.weight
        num_channels = Filter.shape[0]
        results = []
        for i in range(num_channels):
            for j, g in enumerate(A):
                A[j].edata['w_sum'] = g.edata['w'] * Filter[i][j]
            sum_g = dgl.adj_sum_graph(A, 'w_sum')
            results.append(sum_g)
        return results


class GTLayer(nn.Module):
    """
        CTLayer multiply each combination adjacency matrix :math:`l` times to a :math:`l-length`
        meta-paths adjacency matrix.

        The method to generate :math:`l-length` meta-path adjacency matrix can be described as:

        .. math::
            A_{(l)}=\\Pi_{i=1}^{l} A_{i}

        where :math:`A_{i}` is the combination adjacency matrix generated by GT conv.

        Parameters
        ----------
            in_channels: int
                The input dimension of GTConv which is numerically equal to the number of relations.
            out_channels: int
                The input dimension of GTConv which is numerically equal to the number of channel in GTN.
            first: bool
                If true, the first combination adjacency matrix multiply the combination adjacency matrix.

    """

    def __init__(self, in_channels, out_channels, first=True):
        super(GTLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.first = first
        if self.first:
            self.conv1 = GTConv(in_channels, out_channels)
            self.conv2 = GTConv(in_channels, out_channels)
        else:
            self.conv1 = GTConv(in_channels, out_channels)

    def forward(self, A, H_=None):
        if self.first:
            result_A = self.conv1(A)
            result_B = self.conv2(A)
            W = [F.softmax(self.conv1.weight, dim=1).detach(), F.softmax(self.conv2.weight, dim=1).detach()]
        else:
            result_A = H_
            result_B = self.conv1(A)
            W = [F.softmax(self.conv1.weight, dim=1).detach()]
        H = []
        for i in range(len(result_A)):
            g = dgl.adj_product_graph(result_A[i], result_B[i], 'w_sum')
            H.append(g)
        return H, W


class GraphConv(nn.Module):

    def __init__(self, in_feats, out_feats, dropout, activation=None):
        super(GraphConv, self).__init__()
        self._in_feats = in_feats
        self._out_feats = out_feats
        self.weight1 = nn.Parameter(th.Tensor(in_feats, out_feats))
        self.reset_parameters()
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight1)

    def forward(self, hg, feat, edge_weight=None):
        with hg.local_scope():
            outputs = {}
            norm = {}
            aggregate_fn = fn.copy_u('h', 'm')
            if edge_weight is not None:
                hg.edata['_edge_weight'] = edge_weight
                aggregate_fn = fn.u_mul_e('h', '_edge_weight', 'm')
            for e in hg.canonical_etypes:
                if e[0] == e[1]:
                    hg = dgl.remove_self_loop(hg, etype=e)
            feat_src, feat_dst = expand_as_pair(feat, hg)
            hg.srcdata['h'] = feat_src
            for e in hg.canonical_etypes:
                stype, etype, dtype = e
                sub_graph = hg[stype, etype, dtype]
                sub_graph.update_all(aggregate_fn, fn.sum(msg='m', out='out'))
                temp = hg.ndata['out'].pop(dtype)
                degs = sub_graph.in_degrees().float().clamp(min=1)
                if isinstance(temp, dict):
                    temp = temp[dtype]
                if outputs.get(dtype) is None:
                    outputs[dtype] = temp
                    norm[dtype] = degs
                else:
                    outputs[dtype].add_(temp)
                    norm[dtype].add_(degs)

            def _apply(ntype, h, norm):
                h = th.matmul(h + feat[ntype], self.weight1)
                if self.activation:
                    h = self.activation(h)
                return self.dropout(h)
            return {ntype: _apply(ntype, h, norm) for ntype, h in outputs.items()}


def transform_relation_graph_list(hg, category, identity=True):
    """
        extract subgraph :math:`G_i` from :math:`G` in which
        only edges whose type :math:`R_i` belongs to :math:`\\mathcal{R}`

        Parameters
        ----------
            hg : dgl.heterograph
                Input heterogeneous graph
            category : string
                Type of predicted nodes.
            identity : bool
                If True, the identity matrix will be added to relation matrix set.
    """
    for i, ntype in enumerate(hg.ntypes):
        if ntype == category:
            category_id = i
    g = dgl.to_homogeneous(hg, ndata='h')
    loc = g.ndata[dgl.NTYPE] == category_id
    category_idx = th.arange(g.num_nodes())[loc]
    edges = g.edges()
    etype = g.edata[dgl.ETYPE]
    ctx = g.device
    num_edge_type = th.max(etype).item()
    graph_list = []
    for i in range(num_edge_type + 1):
        e_ids = th.nonzero(etype == i).squeeze(-1)
        sg = dgl.graph((edges[0][e_ids], edges[1][e_ids]), num_nodes=g.num_nodes())
        sg.edata['w'] = th.ones(sg.num_edges(), device=ctx)
        graph_list.append(sg)
    if identity == True:
        x = th.arange(0, g.num_nodes(), device=ctx)
        sg = dgl.graph((x, x))
        sg.edata['w'] = th.ones(g.num_nodes(), device=ctx)
        graph_list.append(sg)
    return graph_list, g.ndata['h'], category_idx


class GTN(BaseModel):
    """
        GTN from paper `Graph Transformer Networks <https://arxiv.org/abs/1911.06455>`__
        in NeurIPS_2019. You can also see the extension paper `Graph Transformer
        Networks: Learning Meta-path Graphs to Improve GNNs <https://arxiv.org/abs/2106.06218.pdf>`__.

        `Code from author <https://github.com/seongjunyun/Graph_Transformer_Networks>`__.

        Given a heterogeneous graph :math:`G` and its edge relation type set :math:`\\mathcal{R}`.Then we extract
        the single relation adjacency matrix list. In that, we can generate combination adjacency matrix by conv
        the single relation adjacency matrix list. We can generate :math:'l-length' meta-path adjacency matrix
        by multiplying combination adjacency matrix. Then we can generate node representation using a GCN layer.

        Parameters
        ----------
        num_edge_type : int
            Number of relations.
        num_channels : int
            Number of conv channels.
        in_dim : int
            The dimension of input feature.
        hidden_dim : int
            The dimension of hidden layer.
        num_class : int
            Number of classification type.
        num_layers : int
            Length of hybrid metapath.
        category : string
            Type of predicted nodes.
        norm : bool
            If True, the adjacency matrix will be normalized.
        identity : bool
            If True, the identity matrix will be added to relation matrix set.

    """

    @classmethod
    def build_model_from_args(cls, args, hg):
        if args.identity:
            num_edge_type = len(hg.canonical_etypes) + 1
        else:
            num_edge_type = len(hg.canonical_etypes)
        return cls(num_edge_type=num_edge_type, num_channels=args.num_channels, in_dim=args.hidden_dim, hidden_dim=args.hidden_dim, num_class=args.out_dim, num_layers=args.num_layers, category=args.category, norm=args.norm_emd_flag, identity=args.identity)

    def __init__(self, num_edge_type, num_channels, in_dim, hidden_dim, num_class, num_layers, category, norm, identity):
        super(GTN, self).__init__()
        self.num_edge_type = num_edge_type
        self.num_channels = num_channels
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_class = num_class
        self.num_layers = num_layers
        self.is_norm = norm
        self.category = category
        self.identity = identity
        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.append(GTLayer(num_edge_type, num_channels, first=True))
            else:
                layers.append(GTLayer(num_edge_type, num_channels, first=False))
        self.layers = nn.ModuleList(layers)
        self.gcn = GraphConv(in_feats=self.in_dim, out_feats=hidden_dim, norm='none', activation=F.relu)
        self.norm = EdgeWeightNorm(norm='right')
        self.linear1 = nn.Linear(self.hidden_dim * self.num_channels, self.hidden_dim)
        self.linear2 = nn.Linear(self.hidden_dim, self.num_class)
        self.category_idx = None
        self.A = None
        self.h = None

    def normalization(self, H):
        norm_H = []
        for i in range(self.num_channels):
            g = H[i]
            g = dgl.remove_self_loop(g)
            g.edata['w_sum'] = self.norm(g, g.edata['w_sum'])
            norm_H.append(g)
        return norm_H

    def forward(self, hg, h):
        with hg.local_scope():
            hg.ndata['h'] = h
            if self.category_idx is None:
                self.A, h, self.category_idx = transform_relation_graph_list(hg, category=self.category, identity=self.identity)
            else:
                g = dgl.to_homogeneous(hg, ndata='h')
                h = g.ndata['h']
            A = self.A
            for i in range(self.num_layers):
                if i == 0:
                    H, W = self.layers[i](A)
                else:
                    H, W = self.layers[i](A, H)
                if self.is_norm == True:
                    H = self.normalization(H)
            for i in range(self.num_channels):
                g = dgl.remove_self_loop(H[i])
                edge_weight = g.edata['w_sum']
                g = dgl.add_self_loop(g)
                edge_weight = th.cat((edge_weight, th.full((g.number_of_nodes(),), 1, device=g.device)))
                edge_weight = self.norm(g, edge_weight)
                if i == 0:
                    X_ = self.gcn(g, h, edge_weight=edge_weight)
                else:
                    X_ = th.cat((X_, self.gcn(g, h, edge_weight=edge_weight)), dim=1)
            X_ = self.linear1(X_)
            X_ = F.relu(X_)
            y = self.linear2(X_)
            return {self.category: y[self.category_idx]}


class Grail(BaseModel):

    @classmethod
    def build_model_from_args(cls, args, relation2id):
        return cls(args, relation2id)

    def __init__(self, args, relation2id):
        super(Grail, self).__init__()
        self.params = args
        self.relation2id = relation2id
        self.gnn = RGCN(args)
        self.rel_emb = nn.Embedding(self.params.num_rels, self.params.rel_emb_dim, sparse=False)
        if self.params.add_ht_emb:
            self.fc_layer = nn.Linear(3 * self.params.num_gcn_layers * self.params.emb_dim + self.params.rel_emb_dim, 1)
        else:
            self.fc_layer = nn.Linear(self.params.num_gcn_layers * self.params.emb_dim + self.params.rel_emb_dim, 1)

    def forward(self, hg):
        g, rel_labels = hg
        g.ndata['h'] = self.gnn(g)
        g_out = mean_nodes(g, 'repr')
        head_ids = (g.ndata['id'] == 1).nonzero().squeeze(1)
        head_embs = g.ndata['repr'][head_ids]
        tail_ids = (g.ndata['id'] == 2).nonzero().squeeze(1)
        tail_embs = g.ndata['repr'][tail_ids]
        if self.params.add_ht_emb:
            g_rep = torch.cat([g_out.view(-1, self.params.num_gcn_layers * self.params.emb_dim), head_embs.view(-1, self.params.num_gcn_layers * self.params.emb_dim), tail_embs.view(-1, self.params.num_gcn_layers * self.params.emb_dim), self.rel_emb(rel_labels)], dim=1)
        else:
            g_rep = torch.cat([g_out.view(-1, self.params.num_gcn_layers * self.params.emb_dim), self.rel_emb(rel_labels)], dim=1)
        output = self.fc_layer(g_rep)
        return output


class HAN(nn.Module):

    def __init__(self, num_metapaths, in_dim, hidden_dim, out_dim, num_layers, num_heads, num_out_heads, activation, feat_drop, attn_drop, negative_slope, residual, norm, encoding=False):
        super(HAN, self).__init__()
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.han_layers = nn.ModuleList()
        self.activation = activation
        last_activation = activation if encoding else None
        last_residual = encoding and residual
        last_norm = norm if encoding else None
        if num_layers == 1:
            self.han_layers.append(HANLayer(num_metapaths, in_dim, out_dim, num_out_heads, feat_drop, attn_drop, negative_slope, last_residual, last_activation, norm=last_norm))
        else:
            self.han_layers.append(HANLayer(num_metapaths, in_dim, hidden_dim, num_heads, feat_drop, attn_drop, negative_slope, residual, self.activation, norm=norm))
            for l in range(1, num_layers - 1):
                self.han_layers.append(HANLayer(num_metapaths, hidden_dim * num_heads, hidden_dim, num_heads, feat_drop, attn_drop, negative_slope, residual, self.activation, norm=norm))
            self.han_layers.append(HANLayer(num_metapaths, hidden_dim * num_heads, out_dim, num_out_heads, feat_drop, attn_drop, negative_slope, last_residual, activation=last_activation, norm=last_norm))

    def forward(self, gs: 'list[dgl.DGLGraph]', h, return_hidden=False):
        for gnn in self.han_layers:
            h, att_mp = gnn(gs, h)
        return h, att_mp


class _HAN(nn.Module):

    def __init__(self, meta_paths_dict, in_dim, hidden_dim, out_dim, num_heads, dropout, settings):
        super(_HAN, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(HANLayer(meta_paths_dict, in_dim, hidden_dim, num_heads[0], dropout, settings))
        for l in range(1, len(num_heads)):
            self.layers.append(HANLayer(meta_paths_dict, hidden_dim * num_heads[l - 1], hidden_dim, num_heads[l], dropout, settings))
        self.linear = nn.Linear(hidden_dim * num_heads[-1], out_dim)

    def forward(self, g, h_dict):
        for gnn in self.layers:
            h_dict = gnn(g, h_dict)
        out_dict = {}
        for ntype, h in h_dict.items():
            out_dict[ntype] = self.linear(h_dict[ntype])
        return out_dict

    def get_emb(self, g, h_dict):
        h = h_dict[self.category]
        for gnn in self.layers:
            h = gnn(g, h)
        return {self.category: h.detach().cpu().numpy()}


class GNN(nn.Module):
    """
    Aggregate 2-hop neighbor.
    """

    def __init__(self, input_dim, output_dim, num_neighbor, use_bias=True):
        super(GNN, self).__init__()
        self.input_dim = int(input_dim)
        self.num_fea = int(input_dim)
        self.output_dim = int(output_dim)
        self.num_neighbor = num_neighbor
        self.use_bias = use_bias
        self.linear1 = nn.Linear(self.input_dim * 2, 64)
        self.linear2 = nn.Linear(64 + self.num_fea, 64)
        self.linear3 = nn.Linear(64, self.output_dim)

    def forward(self, fea):
        node = fea[:, :self.num_fea]
        neigh1 = fea[:, self.num_fea:self.num_fea * (self.num_neighbor + 1)]
        neigh1 = torch.reshape(neigh1, [-1, self.num_neighbor, self.num_fea])
        neigh2 = fea[:, self.num_fea * (self.num_neighbor + 1):]
        neigh2 = torch.reshape(neigh2, [-1, self.num_neighbor, self.num_neighbor, self.num_fea])
        neigh2_agg = torch.mean(neigh2, dim=2)
        tmp = torch.cat([neigh1, neigh2_agg], dim=2)
        tmp = F.relu(self.linear1(tmp))
        emb = torch.cat([node, torch.mean(tmp, dim=1)], dim=1)
        emb = F.relu(self.linear2(emb))
        emb = F.relu(self.linear3(emb))
        return emb


class HDE(BaseModel):

    def __init__(self, input_dim, output_dim, num_neighbor, use_bias=True):
        super(HDE, self).__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.num_neighbor = num_neighbor
        self.use_bias = use_bias
        self.aggregator = GNN(input_dim=input_dim, output_dim=output_dim, num_neighbor=num_neighbor)
        self.linear1 = nn.Linear(2 * self.output_dim, 32)
        self.linear2 = nn.Linear(32, 2)

    def forward(self, fea_a, fea_b):
        emb_a = self.aggregator(fea_a)
        emb_b = self.aggregator(fea_b)
        emb = torch.cat([emb_a, emb_b], dim=1)
        emb = F.relu(self.linear1(emb))
        output = self.linear2(emb)
        return output

    @classmethod
    def build_model_from_args(cls, args, hg):
        return cls(input_dim=args.input_dim, output_dim=args.output_dim, num_neighbor=args.num_neighbor, use_bias=args.use_bias)


class HeteroAttLayer(nn.Module):

    def __init__(self, ntype_meta_path, in_dim, att_dim, dropout):
        super(HeteroAttLayer, self).__init__()
        self.ntype_meta_type = ntype_meta_path
        self.nchannel = len(ntype_meta_path)
        self.in_dim = in_dim
        self.att_dim = att_dim
        self.meta_att = nn.Parameter(torch.zeros(size=(len(ntype_meta_path), att_dim)))
        nn.init.xavier_uniform_(self.meta_att.data, gain=1.414)
        self.linear_block2 = nn.Sequential(nn.Linear(att_dim, att_dim), nn.Tanh())

    def forward(self, hs, nnode):
        hs = torch.cat([self.linear_block2(hs[i]).view(1, nnode, -1) for i in range(self.nchannel)], dim=0)
        meta_att = []
        for i in range(self.nchannel):
            meta_att.append(torch.sum(torch.mm(hs[i], self.meta_att[i].view(-1, 1)).squeeze(1)) / nnode)
        meta_att = torch.stack(meta_att, dim=0)
        meta_att = F.softmax(meta_att, dim=0)
        aggre_hid = []
        aggre_hid = torch.bmm
        meta_att_expanded = meta_att.unsqueeze(0).expand(nnode, -1, -1)
        hs_transposed = hs.permute(1, 0, 2)
        aggre_hid = torch.bmm(meta_att_expanded, hs_transposed)
        aggre_hid = aggre_hid.view(nnode, self.att_dim)
        return aggre_hid


class NodeAttLayer(nn.Module):

    def __init__(self, meta_paths_dict, nfeat, hidden_dim, nheads, dropout, *args, **kwargs) ->None:
        super().__init__(*args, **kwargs)
        self.meta_paths_dict = meta_paths_dict
        self.layers = nn.ModuleList()
        self._cached_graph = None
        self._cached_coalesced_graph = {}
        self.mods = nn.ModuleDict({mp: GATConv(nfeat, hidden_dim, nheads, dropout, dropout, activation=F.elu, allow_zero_in_degree=True) for mp in meta_paths_dict})

    def metapath_reachable_graph(self, g, metapath):
        adj = 1
        for etype in metapath:
            adj = adj * g.adj_external(etype=etype, scipy_fmt='csr', transpose=False)
        adj = (adj != 0).tocsr()
        srctype = g.to_canonical_etype(metapath[0])[0]
        dsttype = g.to_canonical_etype(metapath[-1])[2]
        new_g = convert.heterograph({(srctype, '_E', dsttype): adj.nonzero()}, {srctype: adj.shape[0], dsttype: adj.shape[0]}, idtype=g.idtype, device=g.device)
        new_g.nodes[srctype].data.update(g.nodes[srctype].data)
        return new_g

    def forward(self, g, h_dict):
        if isinstance(g, dict):
            g_dict = g
        else:
            if self._cached_graph is None or self._cached_graph is not g:
                self._cached_graph = g
                self._cached_coalesced_graph.clear()
                for mp, mp_value in self.meta_paths_dict.items():
                    self._cached_coalesced_graph[mp] = self.metapath_reachable_graph(g, mp_value)
            g_dict = self._cached_coalesced_graph
        outputs = {}
        for meta_path_name, meta_path in self.meta_paths_dict.items():
            new_g = g_dict[meta_path_name]
            if h_dict.get(meta_path_name) is not None:
                h = h_dict[meta_path_name][new_g.srctypes[0]]
            else:
                h = h_dict[new_g.srctypes[0]]
            outputs[meta_path_name] = self.mods[meta_path_name](new_g, h).flatten(1)
        return outputs


class NodeAttEmb(nn.Module):

    def __init__(self, ntype_meta_paths_dict, nfeat, hidden_dim, nheads, dropout, *args, **kwargs) ->None:
        super().__init__(*args, **kwargs)
        self.NodeAttLayers = nn.ModuleList()
        for ntype, meta_path_dict in ntype_meta_paths_dict:
            self.NodeAttLayers(NodeAttLayer(meta_path_dict, nfeat, hidden_dim, nheads, dropout))
        self.linear = nn.Linear(hidden_dim * nheads[-1], hidden_dim * nheads[-1])

    def forward(self, g, h_dict=None):
        if h_dict == None:
            h_dict = g.ndata['h']
        for gnn in self.NodeAttLayers:
            h_dict = gnn(g, h_dict)
        out_dict = {}
        for ntype, h in h_dict.items():
            out_dict[ntype] = self.linear(h_dict[ntype])
        return out_dict


class _HGA(BaseModel):

    def __init__(self, ntype_meta_paths_dict, nfeat, nlabel, hidden_dim, nheads, dropout):
        super().__init__()
        self.sharedNet = NodeAttLayer(ntype_meta_paths_dict, nfeat, hidden_dim, nheads, dropout)
        self.linear_block = nn.Sequential(nn.Linear(hidden_dim * nheads, hidden_dim), nn.Tanh())
        self.HeteroAttLayerT = HeteroAttLayer(ntype_meta_paths_dict, hidden_dim * nheads, nlabel, dropout)
        self.ntype_meta_paths_dict = ntype_meta_paths_dict
        self.cls_fcs = nn.ModuleList()
        for i in ntype_meta_paths_dict:
            self.cls_fcs.append(torch.nn.Linear(hidden_dim, nlabel))

    def forward(self, gT, gT_dict, gS=None, gS_dict=None):
        if gS is not None:
            homo_outS = self.sharedNet(gS, gS_dict)
            homo_outT = self.sharedNet(gT, gT_dict)
            new_hsS = {i: self.linear_block(homo_outS[i]).view(list(homo_outS.values())[0].shape[0], -1) for i in homo_outS}
            new_hsT = {i: self.linear_block(homo_outT[i]).view(list(homo_outT.values())[0].shape[0], -1) for i in homo_outT}
            clabel_predSs = []
            clabel_predTs = []
            for idx, (path_name, meta_path) in enumerate(self.ntype_meta_paths_dict.items()):
                clabel_predSs.append(self.cls_fcs[idx](new_hsS[path_name]))
                clabel_predTs.append(self.cls_fcs[idx](new_hsT[path_name]))
            tworeS = torch.cat([i.unsqueeze(0) for i in clabel_predSs], dim=0)
            clabel_predS = self.HeteroAttLayerT(tworeS, tworeS.shape[1])
            twore = torch.cat([i.unsqueeze(0) for i in clabel_predTs], dim=0)
            clabel_predF = self.HeteroAttLayerT(twore, twore.shape[1])
            target_probs = F.softmax(clabel_predF, dim=-1)
            target_probs = torch.clamp(target_probs, min=1e-09, max=1.0)
            return homo_outS, homo_outT, clabel_predSs, clabel_predTs, target_probs, clabel_predS
        else:
            homo_outT = self.sharedNet(gT, gT_dict)
            new_hsT = {i: self.linear_block(homo_outT[i]).view(list(homo_outT.values())[0].shape[0], -1) for i in homo_outT}
            clabel_predTs = []
            for idx, (path_name, meta_path) in enumerate(self.ntype_meta_paths_dict.items()):
                clabel_predTs.append(self.cls_fcs[idx](new_hsT[path_name]))
            twore = torch.cat([i.unsqueeze(0) for i in clabel_predTs], dim=0)
            clabel_predF = self.HeteroAttLayerT(twore, twore.shape[1])
            target_probs = F.softmax(clabel_predF, dim=-1)
            target_probs = torch.clamp(target_probs, min=1e-09, max=1.0)
            return target_probs


def extract_metapaths(category, canonical_etypes, self_loop=False):
    meta_paths_dict = {}
    for etype in canonical_etypes:
        if etype[0] in category:
            for dst_e in canonical_etypes:
                if etype[0] == dst_e[2] and etype[2] == dst_e[0]:
                    if self_loop:
                        mp_name = 'mp' + str(len(meta_paths_dict))
                        meta_paths_dict[mp_name] = [etype, dst_e]
                    elif etype[0] != etype[2]:
                        mp_name = 'mp' + str(len(meta_paths_dict))
                        meta_paths_dict[mp_name] = [etype, dst_e]
    return meta_paths_dict


def get_ntypes_from_canonical_etypes(canonical_etypes=None):
    ntypes = set()
    for etype in canonical_etypes:
        src = etype[0]
        dst = etype[2]
        ntypes.add(src)
        ntypes.add(dst)
    return ntypes


class HGA(BaseModel):

    @classmethod
    def build_model_from_args(cls, args, hg):
        ntypes = set()
        if hasattr(args, 'target_link'):
            ntypes = get_ntypes_from_canonical_etypes(args.target_link)
        elif hasattr(args, 'category'):
            ntypes.add(args.category)
        else:
            raise ValueError
        ntype_meta_paths_dict = {}
        for ntype in ntypes:
            ntype_meta_paths_dict[ntype] = {}
            for meta_path_name, meta_path in args.meta_paths_dict.items():
                if meta_path[0][0] == ntype:
                    ntype_meta_paths_dict[ntype][meta_path_name] = meta_path
        for ntype, meta_paths_dict in ntype_meta_paths_dict.items():
            if len(meta_paths_dict) == 0:
                ntype_meta_paths_dict[ntype] = extract_metapaths(ntype, hg.canonical_etypes)
        return cls(ntype_meta_paths_dict=ntype_meta_paths_dict, nfeat=args.hidden_dim, hidden_dim=args.hidden_dim, nlabel=args.out_dim, nheads=args.num_heads, dropout=args.dropout)

    def __init__(self, ntype_meta_paths_dict, nfeat, nlabel, hidden_dim, nheads, dropout):
        super().__init__()
        self.out_dim = nlabel
        self.mod_dict = nn.ModuleDict()
        for ntype, meta_paths_dict in ntype_meta_paths_dict.items():
            self.mod_dict[ntype] = _HGA(meta_paths_dict, nfeat, nlabel, hidden_dim, nheads, dropout)

    def forward(self, gT, h_dictT, gS=None, h_dictS=None):
        """
        Parameters
        -----------
        g : DGLHeteroGraph or dict[str, dict[str, DGLBlock]]
            For full batch, it is a heterogeneous graph. For mini batch, it is a dict from node type to dict from
            mata path name to DGLBlock.
        h_dict : dict[str, Tensor] or dict[str, dict[str, dict[str, Tensor]]]
            The input features. For full batch, it is a dict from node type to node features. For mini batch, it is
            a dict from node type to dict from meta path name to dict from node type to node features.
        Returns
        --------
        out_dict : dict[str, Tensor]
            The output features. Dict from node type to node features.
        """
        if gS is not None:
            for ntype, hga in self.mod_dict.items():
                if isinstance(gS, dict):
                    if ntype not in gS:
                        continue
                    _gS = gS[ntype]
                    _gT = gT[ntype]
                    _in_hS = h_dictS[ntype]
                    _in_hT = h_dictT[ntype]
                else:
                    _gS = gS
                    _gT = gT
                    _in_hS = h_dictS
                    _in_hT = h_dictT
                homo_outS, homo_outT, clabel_predSs, clabel_predTs, target_probs, clabel_predS = hga(gS=_gS, gS_dict=_in_hS, gT=_gT, gT_dict=_in_hT)
            return homo_outS, homo_outT, clabel_predSs, clabel_predTs, target_probs, clabel_predS
        else:
            for ntype, hga in self.mod_dict.items():
                if isinstance(gT, dict):
                    if ntype not in gT:
                        continue
                    _gT = gT[ntype]
                    _in_hT = h_dictT[ntype]
                else:
                    _gT = gT
                    _in_hT = h_dictT
                target_probs = hga(gT=_gT, gT_dict=_in_hT)
            return target_probs


class NodeAttention(nn.Module):
    """
    The node-level attention layer

    Parameters
    ----------
    in_dim: int
        the input dimension of the feature
    out_dim: int
        the output dimension
    slope: float
        the negative slope used in the LeakyReLU
    """

    def __init__(self, in_dim, out_dim, slope):
        super(NodeAttention, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.Mu_l = nn.Linear(in_dim, in_dim)
        self.Mu_r = nn.Linear(in_dim, in_dim)
        self.leakyrelu = nn.LeakyReLU(slope)

    def forward(self, g, x, ntype, etype, presorted=False):
        """
        The forward part of the NodeAttention.

        Parameters
        ----------
        g : object
            the dgl homogeneous graph
        x: tensor
            the original features of the graph
        ntype: tensor
            the node type of the graph
        etype: tensor
            the edge type of the graph
        presorted: boolean
            if the ntype and etype are preordered, default: ``False``
            
        Returns
        -------
        tensor
            The embeddings after aggregation.
        """
        with g.local_scope():
            src = g.edges()[0]
            dst = g.edges()[1]
            h_l = self.Mu_l(x)[src]
            h_r = self.Mu_r(x)[dst]
            edge_attention = self.leakyrelu((h_l + h_r) * g.edata['alpha'])
            edge_attention = edge_softmax(g, edge_attention)
            g.edata['alpha'] = edge_attention
            g.srcdata['x'] = x
            g.update_all(Fn.u_mul_e('x', 'alpha', 'm'), Fn.sum('m', 'x'))
            h = g.ndata['x']
        return h


class TypeAttention(nn.Module):
    """
    The type-level attention layer

    Parameters
    ----------
    in_dim: int
        the input dimension of the feature
    ntypes: list
        the list of the node type in the graph
    slope: float
        the negative slope used in the LeakyReLU
    """

    def __init__(self, in_dim, ntypes, slope):
        super(TypeAttention, self).__init__()
        attn_vector = {}
        for ntype in ntypes:
            attn_vector[ntype] = in_dim
        self.mu_l = HeteroLinear(attn_vector, in_dim)
        self.mu_r = HeteroLinear(attn_vector, in_dim)
        self.leakyrelu = nn.LeakyReLU(slope)

    def forward(self, hg, h_dict):
        """
        The forward part of the TypeAttention.
        
        Parameters
        ----------
        hg : object
            the dgl heterogeneous graph
        h_dict: dict
            the feature dict of different node types
            
        Returns
        -------
        dict
            The embeddings after the output projection.
        """
        h_t = {}
        attention = {}
        with hg.local_scope():
            hg.ndata['h'] = h_dict
            for srctype, etype, dsttype in hg.canonical_etypes:
                rel_graph = hg[srctype, etype, dsttype]
                if srctype not in h_dict:
                    continue
                with rel_graph.local_scope():
                    degs = rel_graph.out_degrees().float().clamp(min=1)
                    norm = torch.pow(degs, -0.5)
                    feat_src = h_dict[srctype]
                    shp = norm.shape + (1,) * (feat_src.dim() - 1)
                    norm = torch.reshape(norm, shp)
                    feat_src = feat_src * norm
                    rel_graph.srcdata['h'] = feat_src
                    rel_graph.update_all(Fn.copy_u('h', 'm'), Fn.sum(msg='m', out='h'))
                    rst = rel_graph.dstdata['h']
                    degs = rel_graph.in_degrees().float().clamp(min=1)
                    norm = torch.pow(degs, -0.5)
                    shp = norm.shape + (1,) * (feat_src.dim() - 1)
                    norm = torch.reshape(norm, shp)
                    rst = rst * norm
                    h_t[srctype] = rst
                    h_l = self.mu_l(h_dict)[dsttype]
                    h_r = self.mu_r(h_t)[srctype]
                    edge_attention = F.elu(h_l + h_r)
                    rel_graph.ndata['m'] = {dsttype: edge_attention, srctype: torch.zeros((rel_graph.num_nodes(ntype=srctype),))}
                    reverse_graph = dgl.reverse(rel_graph)
                    reverse_graph.apply_edges(Fn.copy_u('m', 'alpha'))
                hg.edata['alpha'] = {(srctype, etype, dsttype): reverse_graph.edata['alpha']}
            attention = edge_softmax(hg, hg.edata['alpha'])
        return attention


def to_hetero_feat(h, type, name):
    """Feature convert API.

    It uses information about the type of the specified node
    to convert features ``h`` in homogeneous graph into a heteorgeneous
    feature dictionay ``h_dict``.

    Parameters
    ----------
    h: Tensor
        Input features of homogeneous graph
    type: Tensor
        Represent the type of each node or edge with a number.
        It should correspond to the parameter ``name``.
    name: list
        The node or edge types list.

    Return
    ------
    h_dict: dict
        output feature dictionary of heterogeneous graph

    Example
    -------

    >>> h = torch.tensor([[1, 2, 3],
                          [1, 1, 1],
                          [0, 2, 1],
                          [1, 3, 3],
                          [2, 1, 1]])
    >>> print(h.shape)
    torch.Size([5, 3])
    >>> type = torch.tensor([0, 1, 0, 0, 1])
    >>> name = ['author', 'paper']
    >>> h_dict = to_hetero_feat(h, type, name)
    >>> print(h_dict)
    {'author': tensor([[1, 2, 3],
    [0, 2, 1],
    [1, 3, 3]]), 'paper': tensor([[1, 1, 1],
    [2, 1, 1]])}

    """
    h_dict = {}
    for index, ntype in enumerate(name):
        h_dict[ntype] = h[th.where(type == index)]
    return h_dict


class HGAT(BaseModel):
    """
    This is a model HGAT from `Heterogeneous Graph Attention Networks for Semi-supervised Short Text Classification
    <https://dl.acm.org/doi/abs/10.1145/3450352>`__

    It contains the following parts:

    Type-level Attention: Given a specific node :math:`v`, we need to calculate the type-level attention scores based on the current node 
    embedding and the type embedding.
    
    .. math::
       a_{\\tau} = \\sigma(\\mu_{\\tau}^T \\cdot [h_v \\parallel h_{\\tau}]) \\quad (1)
    
    The type embedding is :math:`h_{\\tau}=\\sum_{v^{'}}\\widetilde{A}_{vv^{'}}h_{v^{'}}`, 
    which is the sum of the neighboring node features :math:`h_{v^{'}}` 
    where the nodes :math:`v^{'} \\in \\mathcal{N}_v` and are with the type :math:`h_{\\tau}`.
    :math:`\\mu_{\\tau}` is the attention vector for the type :math:`\\tau`.
    
    And the type-level attention weights is:
    
    .. math::
       \\alpha_{\\tau} = \\frac{exp(a_{\\tau})}{\\sum_{\\tau^{'}\\in \\mathcal{T}} exp(a_{\\tau^{'}})} \\quad (2)

    Node-level Attention: Given a specific node :math:`v` and its neightoring node :math:`v^{'}\\in \\mathcal{N}_v`, 
    we need to calculate the node-level attention scores based on the node embeddings :math:`h_v` and :math:`h_{v^{'}}`
    and with the type-level attention weight :math:`\\alpha_{\\tau^{'}}` for the node :math:`v^{'}`:
    
    .. math::
       b_{vv^{'}} = \\sigma(\\nu^T \\cdot \\alpha_{\\tau^{'}}[h_v \\parallel h_{v^{'}}]) \\quad (3)
    
    where :math:`\\nu` is the attention vector.
    
    And the node-level attention weights is:
    
    .. math::
       \\beta_{vv^{'}} = \\frac{exp(b_{vv^{'}})}{\\sum_{i\\in \\mathcal{N}_v} exp(b_{vi})} \\quad (4)
    
    The final output is:
    
    .. math::
       H^{(l+1)} = \\sigma(\\sum_{\\tau \\in \\mathcal{T}}B_{\\tau}\\cdot H_{\\tau}^{(l)}\\cdot W_{\\tau}^{(l)}) \\quad (5)
    
    Parameters
    ----------
    num_layers: int
        the number of layers we used in the computing
    in_dim: int
        the input dimension
    hidden_dim: int
        the hidden dimension
    num_classes: int
        the number of the output classes
    ntypes: list
        the list of the node type in the graph
    negative_slope: float
        the negative slope used in the LeakyReLU
    """

    @classmethod
    def build_model_from_args(cls, args, hg):
        return cls(args.num_layers, args.hidden_dim, args.num_classes, hg.ntypes, args.negative_slope)

    def __init__(self, num_layers, hidden_dim, num_classes, ntypes, negative_slope):
        super(HGAT, self).__init__()
        self.num_layers = num_layers
        self.activation = F.elu
        self.hgat_layers = nn.ModuleList()
        self.hgat_layers.append(TypeAttention(hidden_dim, ntypes, negative_slope))
        self.hgat_layers.append(NodeAttention(hidden_dim, hidden_dim, negative_slope))
        for l in range(num_layers - 1):
            self.hgat_layers.append(TypeAttention(hidden_dim, ntypes, negative_slope))
            self.hgat_layers.append(NodeAttention(hidden_dim, hidden_dim, negative_slope))
        self.hgat_layers.append(TypeAttention(hidden_dim, ntypes, negative_slope))
        self.hgat_layers.append(NodeAttention(hidden_dim, num_classes, negative_slope))

    def forward(self, hg, h_dict):
        """
        The forward part of the HGAT.
        
        Parameters
        ----------
        hg : object
            the dgl heterogeneous graph
        h_dict: dict
            the feature dict of different node types
            
        Returns
        -------
        dict
            The embeddings after the output projection.
        """
        with hg.local_scope():
            hg.ndata['h'] = h_dict
            for l in range(self.num_layers):
                attention = self.hgat_layers[2 * l](hg, hg.ndata['h'])
                hg.edata['alpha'] = attention
                g = dgl.to_homogeneous(hg, ndata='h', edata=['alpha'])
                h = self.hgat_layers[2 * l + 1](g, g.ndata['h'], g.ndata['_TYPE'], g.ndata['_TYPE'], presorted=True)
                h_dict = to_hetero_feat(h, g.ndata['_TYPE'], hg.ntypes)
                hg.ndata['h'] = h_dict
        return h_dict


class GCN_layer(nn.Module):

    def __init__(self):
        super(GCN_layer, self).__init__()

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        if type(sparse_mx) != sp.coo_matrix:
            sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data).float()
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)

    def normalize_adj(self, adj):
        """Symmetrically normalize adjacency matrix."""
        adj = sp.coo_matrix(adj)
        rowsum = np.array(adj.sum(1))
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()

    def forward(self, features, Mat, index):
        subset_Mat = Mat
        subset_features = features
        subset_Mat = self.normalize_adj(subset_Mat)
        subset_sparse_tensor = self.sparse_mx_to_torch_sparse_tensor(subset_Mat)
        out_features = torch.spmm(subset_sparse_tensor, subset_features)
        new_features = torch.empty(features.shape)
        new_features[index] = out_features
        dif_index = np.setdiff1d(torch.arange(features.shape[0]), index)
        new_features[dif_index] = features[dif_index]
        return new_features


class HGCL(BaseModel):

    def build_model_from_args(args, hg):
        userNum = hg.number_of_nodes('user')
        itemNum = hg.number_of_nodes('item')
        userMat = hg.adj_external(etype=('user', 'distance', 'user'), scipy_fmt='csr')
        itemMat = hg.adj_external(etype=('item', 'distance', 'item'), scipy_fmt='csr')
        uiMat = hg.adj_external(etype=('user+item', 'distance', 'user+item'), scipy_fmt='csr')
        return HGCL(userNum=userNum, itemNum=itemNum, userMat=userMat, itemMat=itemMat, uiMat=uiMat, hide_dim=args.hide_dim, Layers=args.Layers, rank=args.rank, wu1=args.wu1, wu2=args.wu2, wi1=args.wi1, wi2=args.wi2)

    def __init__(self, userNum, itemNum, userMat, itemMat, uiMat, hide_dim, Layers, rank, wu1, wu2, wi1, wi2):
        super(HGCL, self).__init__()
        self.userNum = userNum
        self.itemNum = itemNum
        self.uuMat = userMat
        self.iiMat = itemMat
        self.uiMat = uiMat
        self.hide_dim = hide_dim
        self.LayerNums = Layers
        self.wu1 = wu1
        self.wu2 = wu2
        self.wi1 = wi1
        self.wi2 = wi2
        uimat = self.uiMat[:self.userNum, self.userNum:]
        values = torch.FloatTensor(uimat.tocoo().data)
        indices = np.vstack((uimat.tocoo().row, uimat.tocoo().col))
        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        shape = uimat.tocoo().shape
        uimat1 = torch.sparse.FloatTensor(i, v, torch.Size(shape))
        self.uiadj = uimat1
        self.iuadj = uimat1.transpose(0, 1)
        self.gating_weightub = nn.Parameter(torch.FloatTensor(1, hide_dim))
        nn.init.xavier_normal_(self.gating_weightub.data)
        self.gating_weightu = nn.Parameter(torch.FloatTensor(hide_dim, hide_dim))
        nn.init.xavier_normal_(self.gating_weightu.data)
        self.gating_weightib = nn.Parameter(torch.FloatTensor(1, hide_dim))
        nn.init.xavier_normal_(self.gating_weightib.data)
        self.gating_weighti = nn.Parameter(torch.FloatTensor(hide_dim, hide_dim))
        nn.init.xavier_normal_(self.gating_weighti.data)
        self.encoder = nn.ModuleList()
        for i in range(0, self.LayerNums):
            self.encoder.append(GCN_layer())
        self.k = rank
        k = self.k
        self.mlp = MLP(hide_dim, hide_dim * k, hide_dim // 2, hide_dim * k)
        self.mlp1 = MLP(hide_dim, hide_dim * k, hide_dim // 2, hide_dim * k)
        self.mlp2 = MLP(hide_dim, hide_dim * k, hide_dim // 2, hide_dim * k)
        self.mlp3 = MLP(hide_dim, hide_dim * k, hide_dim // 2, hide_dim * k)
        self.meta_netu = nn.Linear(hide_dim * 3, hide_dim, bias=True)
        self.meta_neti = nn.Linear(hide_dim * 3, hide_dim, bias=True)
        self.embedding_dict = nn.ModuleDict({'uu_emb': torch.nn.Embedding(userNum, hide_dim), 'ii_emb': torch.nn.Embedding(itemNum, hide_dim), 'user_emb': torch.nn.Embedding(userNum, hide_dim), 'item_emb': torch.nn.Embedding(itemNum, hide_dim)})

    def init_weight(self, userNum, itemNum, hide_dim):
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({'user_emb': nn.Parameter(initializer(t.empty(userNum, hide_dim))), 'item_emb': nn.Parameter(initializer(t.empty(itemNum, hide_dim)))})
        return embedding_dict

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        if type(sparse_mx) != sp.coo_matrix:
            sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data).float()
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)

    def metaregular(self, em0, em, adj):

        def row_column_shuffle(embedding):
            corrupted_embedding = embedding[:, torch.randperm(embedding.shape[1])]
            corrupted_embedding = corrupted_embedding[torch.randperm(embedding.shape[0])]
            return corrupted_embedding

        def score(x1, x2):
            x1 = F.normalize(x1, p=2, dim=-1)
            x2 = F.normalize(x2, p=2, dim=-1)
            return torch.sum(torch.multiply(x1, x2), 1)
        user_embeddings = em
        Adj_Norm = t.from_numpy(np.sum(adj, axis=1)).float()
        adj = self.sparse_mx_to_torch_sparse_tensor(adj)
        edge_embeddings = torch.spmm(adj, user_embeddings) / Adj_Norm
        user_embeddings = em0
        graph = torch.mean(edge_embeddings, 0)
        pos = score(user_embeddings, graph)
        neg1 = score(row_column_shuffle(user_embeddings), graph)
        global_loss = torch.mean(-torch.log(torch.sigmoid(pos - neg1)))
        return global_loss

    def self_gatingu(self, em):
        return torch.multiply(em, torch.sigmoid(torch.matmul(em, self.gating_weightu) + self.gating_weightub))

    def self_gatingi(self, em):
        return torch.multiply(em, torch.sigmoid(torch.matmul(em, self.gating_weighti) + self.gating_weightib))

    def metafortansform(self, auxiembedu, targetembedu, auxiembedi, targetembedi):
        uneighbor = t.matmul(self.uiadj, self.ui_itemEmbedding)
        ineighbor = t.matmul(self.iuadj, self.ui_userEmbedding)
        tembedu = self.meta_netu(t.cat((auxiembedu, targetembedu, uneighbor), dim=1).detach())
        tembedi = self.meta_neti(t.cat((auxiembedi, targetembedi, ineighbor), dim=1).detach())
        """ Personalized transformation parameter matrix """
        metau1 = self.mlp(tembedu).reshape(-1, self.hide_dim, self.k)
        metau2 = self.mlp1(tembedu).reshape(-1, self.k, self.hide_dim)
        metai1 = self.mlp2(tembedi).reshape(-1, self.hide_dim, self.k)
        metai2 = self.mlp3(tembedi).reshape(-1, self.k, self.hide_dim)
        meta_biasu = torch.mean(metau1, dim=0)
        meta_biasu1 = torch.mean(metau2, dim=0)
        meta_biasi = torch.mean(metai1, dim=0)
        meta_biasi1 = torch.mean(metai2, dim=0)
        low_weightu1 = F.softmax(metau1 + meta_biasu, dim=1)
        low_weightu2 = F.softmax(metau2 + meta_biasu1, dim=1)
        low_weighti1 = F.softmax(metai1 + meta_biasi, dim=1)
        low_weighti2 = F.softmax(metai2 + meta_biasi1, dim=1)
        tembedus = t.sum(t.multiply(auxiembedu.unsqueeze(-1), low_weightu1), dim=1)
        tembedus = t.sum(t.multiply(tembedus.unsqueeze(-1), low_weightu2), dim=1)
        tembedis = t.sum(t.multiply(auxiembedi.unsqueeze(-1), low_weighti1), dim=1)
        tembedis = t.sum(t.multiply(tembedis.unsqueeze(-1), low_weighti2), dim=1)
        transfuEmbed = tembedus
        transfiEmbed = tembedis
        return transfuEmbed, transfiEmbed

    def forward(self, iftraining, uid, iid, norm=1):
        item_index = np.arange(0, self.itemNum)
        user_index = np.arange(0, self.userNum)
        ui_index = np.array(user_index.tolist() + [(i + self.userNum) for i in item_index])
        userembed0 = self.embedding_dict['user_emb'].weight
        itemembed0 = self.embedding_dict['item_emb'].weight
        uu_embed0 = self.self_gatingu(userembed0)
        ii_embed0 = self.self_gatingi(itemembed0)
        self.ui_embeddings = t.cat([userembed0, itemembed0], 0)
        self.all_user_embeddings = [uu_embed0]
        self.all_item_embeddings = [ii_embed0]
        self.all_ui_embeddings = [self.ui_embeddings]
        for i in range(len(self.encoder)):
            layer = self.encoder[i]
            if i == 0:
                userEmbeddings0 = layer(uu_embed0, self.uuMat, user_index)
                itemEmbeddings0 = layer(ii_embed0, self.iiMat, item_index)
                uiEmbeddings0 = layer(self.ui_embeddings, self.uiMat, ui_index)
            else:
                userEmbeddings0 = layer(userEmbeddings, self.uuMat, user_index)
                itemEmbeddings0 = layer(itemEmbeddings, self.iiMat, item_index)
                uiEmbeddings0 = layer(uiEmbeddings, self.uiMat, ui_index)
            self.ui_userEmbedding0, self.ui_itemEmbedding0 = t.split(uiEmbeddings0, [self.userNum, self.itemNum])
            userEd = (userEmbeddings0 + self.ui_userEmbedding0) / 2.0
            itemEd = (itemEmbeddings0 + self.ui_itemEmbedding0) / 2.0
            userEmbeddings = userEd
            itemEmbeddings = itemEd
            uiEmbeddings = torch.cat([userEd, itemEd], 0)
            if norm == 1:
                norm_embeddings = F.normalize(userEmbeddings0, p=2, dim=1)
                self.all_user_embeddings += [norm_embeddings]
                norm_embeddings = F.normalize(itemEmbeddings0, p=2, dim=1)
                self.all_item_embeddings += [norm_embeddings]
                norm_embeddings = F.normalize(uiEmbeddings0, p=2, dim=1)
                self.all_ui_embeddings += [norm_embeddings]
            else:
                self.all_user_embeddings += [userEmbeddings]
                self.all_item_embeddings += [norm_embeddings]
                self.all_ui_embeddings += [norm_embeddings]
        self.userEmbedding = t.stack(self.all_user_embeddings, dim=1)
        self.userEmbedding = t.mean(self.userEmbedding, dim=1)
        self.itemEmbedding = t.stack(self.all_item_embeddings, dim=1)
        self.itemEmbedding = t.mean(self.itemEmbedding, dim=1)
        self.uiEmbedding = t.stack(self.all_ui_embeddings, dim=1)
        self.uiEmbedding = t.mean(self.uiEmbedding, dim=1)
        self.ui_userEmbedding, self.ui_itemEmbedding = t.split(self.uiEmbedding, [self.userNum, self.itemNum])
        metatsuembed, metatsiembed = self.metafortansform(self.userEmbedding, self.ui_userEmbedding, self.itemEmbedding, self.ui_itemEmbedding)
        self.userEmbedding = self.userEmbedding + metatsuembed
        self.itemEmbedding = self.itemEmbedding + metatsiembed
        metaregloss = 0
        if iftraining == True:
            self.reg_lossu = self.metaregular(self.ui_userEmbedding[uid.cpu().numpy()], self.userEmbedding, self.uuMat[uid.cpu().numpy()])
            self.reg_lossi = self.metaregular(self.ui_itemEmbedding[iid.cpu().numpy()], self.itemEmbedding, self.iiMat[iid.cpu().numpy()])
            metaregloss = (self.reg_lossu + self.reg_lossi) / 2.0
        return self.userEmbedding, self.itemEmbedding, self.wu1 * self.ui_userEmbedding + self.wu2 * self.userEmbedding, self.wi1 * self.ui_itemEmbedding + self.wi2 * self.itemEmbedding, self.ui_userEmbedding, self.ui_itemEmbedding, metaregloss


class AttentionLayer(nn.Module):
    """
    This is the attention process used in HGNN\\_AC. For more details, you can check here_.
    
    Parameters
    -------------------
    in_dim: int
        nodes' topological embedding dimension
    hidden_dim: int
        hidden dimension
    dropout: float
        the drop rate used in the attention
    activation: callable activation function
        the activation function used in HGNN_AC.  default: ``F.elu``
    """

    def __init__(self, in_dim, hidden_dim, dropout, activation, cuda=False):
        super(AttentionLayer, self).__init__()
        self.dropout = dropout
        self.activation = activation
        self.is_cuda = cuda
        self.W = nn.Parameter(nn.init.xavier_normal_(torch.Tensor(in_dim, hidden_dim).type(torch.FloatTensor if cuda else torch.FloatTensor), gain=np.sqrt(2.0)), requires_grad=True)
        self.W2 = nn.Parameter(nn.init.xavier_normal_(torch.Tensor(hidden_dim, hidden_dim).type(torch.FloatTensor if cuda else torch.FloatTensor), gain=np.sqrt(2.0)), requires_grad=True)
        self.leakyrelu = nn.LeakyReLU(0.2)

    def forward(self, bias, emb_dest, emb_src, feature_src):
        """
        This is the forward part of the attention process.
        
        Parameters
        --------------
        bias: matrix
            the processed adjacency matrix related to the source nodes
        emb_dest: matrix
            the embeddings of the destination nodes
        emb_src: matrix
            the embeddings of the source nodes
        feature_src: matrix
            the features of the source nodes
        
        Returns
        ------------
        features: matrix
            the new features of the nodes
        """
        h_1 = torch.mm(emb_src, self.W)
        h_2 = torch.mm(emb_dest, self.W)
        e = self.leakyrelu(torch.mm(torch.mm(h_2, self.W2), h_1.t()))
        zero_vec = -9000000000000000.0 * torch.ones_like(e)
        attention = torch.where(bias > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, feature_src)
        return self.activation(h_prime)


class HGNN_AC(BaseModel):
    """
    HGNN_AC was introduced in `HGNN_AC <https://dl.acm.org/doi/10.1145/3442381.3449914>`__.
        
    It included four parts:

    - Pre-learning of Topological Embedding
        HGNN-AC first obtains more comprehensive node sequences by random walk according to the frequently used multiple meta-paths, 
        and then feeds these sequences to the skip-gram model to learn node embeddings :math:`H`.
        
    - Attribute Completion with Attention Mechanism
        HGNN-AC adopts a masked attention mechanism which means we only calculate :math:`e_{vu}` for nodes :math:`u\\in{N_v^+}`, 
        where :math:`u\\in{N_v^+}` denotes the first-order neighbors of node :math:`v` 
        in set :math:`V^+`, where :math:`V^+` is the set of nodes with attributes.
        
        .. math::
           e_{vu}=\\sigma(h_v^{T}Wh_u)
        
        where :math:`W` is the parametric matrix, and :math:`\\sigma` an activation function.
    
        Then, softmax function is applied to get normalized weighted coefficient :math:`a_{vu}`

        .. math::
           a_{vu}=softmax(e_{vu})=\\frac{exp(e_{vu})}{\\sum_{s\\in{N_v^+}}{exp(e_{vs})}}

        HGNN-AC can perform weighted aggregation of attributes
        for node :math:`v`  according to weighted coefficient :math:`a_{vu}`  :

        .. math::
           X_v^C=\\sum_{u\\in{N_v^+}}{a_{vu}x_u}

        where :math:`N_v^+` denotes the set of neighbors of node :math:`v\\in{V^+}`,
        and :math:`x_u` denotes the attributes of nodes :math:`u`.

        .. _here:
        
        Specially, the attention process is extended to a multi-head attention
        to stabilize the learning process and reduce the high variance

        .. math::
           X_v^C=mean(\\sum_k^K {\\sum_{u\\in{N_v^+}}{a_{vu}x_u}})

        where :math:`K` means that we perform :math:`K` independent attention process.

    - Dropping some Attributes
        To be specific, for nodes in :math:`V^+`, HGNN-AC randomly divides them into two parts
        :math:`V_{drop}^+` and :math:`V_{keep}^+` according to a small ratio :math:`\\alpha`, i.e. :math:`|V_{drop}^+|=\\alpha|V^+|`.
        HGNN-AC first drops attributes of nodes in :math:`V_{drop}^+` and then 
        reconstructs these attributes via attributes of nodes :math:`V_{drop}^+` by conducting
        attribute completion.
        
        .. math::
           X_v^C=mean(\\sum_k^K {\\sum_{u\\in{V_{keep}^+ \\cap V_i^+}}{a_{vu}x_u}})

        It introduced a weakly supervised loss to optimize the parameters of attribute completion 
        and use euclidean distance as the metric to design the loss function as:
    
        .. math::
           L_{completion}=\\frac{1}{|V_{drop}^+|}\\sum_{i \\in V_{drop}^+} \\sqrt{(X_i^C-X_i)^2}
    
    - Combination with HIN Model
        Now, we have completed attributes nodes in :math:`V^-`(the set of nodes without attribute), and the raw attributes nodes in :math:`V+`, 
        Wthen the new attributes of all nodes are defined as:

        .. math::
           X^{new}=\\{X_i^C,X_j|\\forall i \\in V^-, \\forall j \\in V^+\\}

        the new attributes :math:`X^{new}`, together with network topology :math:`A`, as
        a new graph, are sent to the HIN model:

        .. math::
           \\overline{Y}=\\Phi(A,X^{new})
           L_{prediction}=f(\\overline{Y},Y)
        
        where :math:`\\Phi` denotes an arbitrary HINs model.

        the overall model can be optimized via back propagation in an end-to-end
        manner:

        .. math::
           L=\\lambda L_{completion}+L_{prediction}
    
        where :math:`\\lambda` is a weighted coefficient to balance these two parts.
        
    Parameters
    ----------
    in_dim: int
        nodes' topological embedding dimension
    hidden_dim: int
        hidden dimension 
    dropout: float
        the dropout rate of neighbor nodes dropout
    activation: callable activation function
        the activation function used in HGNN_AC.  default: ``F.elu``
    num_heads: int
        the number of heads in attribute completion with attention mechanism
    """

    @classmethod
    def build_model_from_args(cls, args, hg):
        return cls(in_dim=hg.nodes[hg.ntypes[0]].data['h'].shape[1], hidden_dim=args.attn_vec_dim, dropout=args.dropout, activation=F.elu, num_heads=args.num_heads, cuda=False if args.device == torch.device('cpu') else True)

    def __init__(self, in_dim, hidden_dim, dropout, activation, num_heads, cuda):
        super(HGNN_AC, self).__init__()
        self.dropout = dropout
        self.attentions = [AttentionLayer(in_dim, hidden_dim, dropout, activation, cuda) for _ in range(num_heads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

    def forward(self, bias, emb_dest, emb_src, feature_src):
        """
        This is the forward part of model HGNN_AC

        Parameters
        ----------
        bias: matrix
            adjacency matrix related to the source nodes
        emb_dest: matrix
            embeddings of the destination node
        emb_src: matrix
            embeddings of the source node
        feature_src: matrix
            features of the source node
            
        Returns
        -------
        features: matrix
            the new features of the type of node
        """
        adj = F.dropout(bias, self.dropout, training=self.training)
        x = torch.cat([att(adj, emb_dest, emb_src, feature_src).unsqueeze(0) for att in self.attentions], dim=0)
        return torch.mean(x, dim=0, keepdim=False)


class myGATConv(nn.Module):
    """
    Adapted from
    https://docs.dgl.ai/_modules/dgl/nn/pytorch/conv/gatconv.html#GATConv
    """

    def __init__(self, edge_feats, num_etypes, in_feats, out_feats, num_heads, feat_drop=0.0, attn_drop=0.0, negative_slope=0.2, residual=False, activation=None, allow_zero_in_degree=False, bias=False, alpha=0.0):
        super(myGATConv, self).__init__()
        self._edge_feats = edge_feats
        self._num_heads = num_heads
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree
        self.edge_emb = nn.Embedding(num_etypes, edge_feats)
        if isinstance(in_feats, tuple):
            self.fc_src = nn.Linear(self._in_src_feats, out_feats * num_heads, bias=False)
            self.fc_dst = nn.Linear(self._in_dst_feats, out_feats * num_heads, bias=False)
        else:
            self.fc = nn.Linear(self._in_src_feats, out_feats * num_heads, bias=False)
        self.fc_e = nn.Linear(edge_feats, edge_feats * num_heads, bias=False)
        self.attn_l = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats)))
        self.attn_r = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats)))
        self.attn_e = nn.Parameter(th.FloatTensor(size=(1, num_heads, edge_feats)))
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        if residual:
            if self._in_dst_feats != out_feats:
                self.res_fc = nn.Linear(self._in_dst_feats, num_heads * out_feats, bias=False)
            else:
                self.res_fc = Identity()
        else:
            self.register_buffer('res_fc', None)
        self.reset_parameters()
        self.activation = activation
        self.bias = bias
        if bias:
            self.bias_param = nn.Parameter(th.zeros((1, num_heads, out_feats)))
        self.alpha = alpha

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        if hasattr(self, 'fc'):
            nn.init.xavier_normal_(self.fc.weight, gain=gain)
        else:
            nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)
        nn.init.xavier_normal_(self.attn_e, gain=gain)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)
        nn.init.xavier_normal_(self.fc_e.weight, gain=gain)

    def set_allow_zero_in_degree(self, set_value):
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat, e_feat, res_attn=None):
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise DGLError('There are 0-in-degree nodes in the graph, output for those nodes will be invalid. This is harmful for some applications, causing silent performance regression. Adding self-loop on the input graph by calling `g = dgl.add_self_loop(g)` will resolve the issue. Setting ``allow_zero_in_degree`` to be `True` when constructing this module will suppress the check and let the code run.')
            if isinstance(feat, tuple):
                h_src = self.feat_drop(feat[0])
                h_dst = self.feat_drop(feat[1])
                if not hasattr(self, 'fc_src'):
                    self.fc_src, self.fc_dst = self.fc, self.fc
                feat_src = self.fc_src(h_src).view(-1, self._num_heads, self._out_feats)
                feat_dst = self.fc_dst(h_dst).view(-1, self._num_heads, self._out_feats)
            else:
                h_src = h_dst = self.feat_drop(feat)
                feat_src = feat_dst = self.fc(h_src).view(-1, self._num_heads, self._out_feats)
                if graph.is_block:
                    feat_dst = feat_src[:graph.number_of_dst_nodes()]
            e_feat = self.edge_emb(e_feat)
            e_feat = self.fc_e(e_feat).view(-1, self._num_heads, self._edge_feats)
            ee = (e_feat * self.attn_e).sum(dim=-1).unsqueeze(-1)
            el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
            er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)
            graph.srcdata.update({'ft': feat_src, 'el': el})
            graph.dstdata.update({'er': er})
            graph.edata.update({'ee': ee})
            graph.apply_edges(fn.u_add_v('el', 'er', 'e'))
            e = self.leaky_relu(graph.edata.pop('e') + graph.edata.pop('ee'))
            graph.edata['a'] = self.attn_drop(edge_softmax(graph, e))
            if res_attn is not None:
                graph.edata['a'] = graph.edata['a'] * (1 - self.alpha) + res_attn * self.alpha
            graph.update_all(fn.u_mul_e('ft', 'a', 'm'), fn.sum('m', 'ft'))
            rst = graph.dstdata['ft']
            if self.res_fc is not None:
                resval = self.res_fc(h_dst).view(h_dst.shape[0], -1, self._out_feats)
                rst = rst + resval
            if self.bias:
                rst = rst + self.bias_param
            if self.activation:
                rst = self.activation(rst)
            return rst, graph.edata.pop('a').detach()


class myGAT(nn.Module):

    def __init__(self, g, edge_dim, num_etypes, in_dims, num_hidden, num_classes, num_layers, heads, activation, feat_drop, attn_drop, negative_slope, residual, alpha):
        super(myGAT, self).__init__()
        self.g = g
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation
        self.fc_list = nn.ModuleList([nn.Linear(in_dim, num_hidden, bias=True) for in_dim in in_dims])
        for fc in self.fc_list:
            nn.init.xavier_normal_(fc.weight, gain=1.414)
        self.gat_layers.append(myGATConv(edge_dim, num_etypes, num_hidden, num_hidden, heads[0], feat_drop, attn_drop, negative_slope, False, self.activation, alpha=alpha))
        for l in range(1, num_layers):
            self.gat_layers.append(myGATConv(edge_dim, num_etypes, num_hidden * heads[l - 1], num_hidden, heads[l], feat_drop, attn_drop, negative_slope, residual, self.activation, alpha=alpha))
        self.gat_layers.append(myGATConv(edge_dim, num_etypes, num_hidden * heads[-2], num_classes, heads[-1], feat_drop, attn_drop, negative_slope, residual, None, alpha=alpha))
        self.epsilon = torch.FloatTensor([1e-12])

    def forward(self, features_list, e_feat):
        h = []
        for fc, feature in zip(self.fc_list, features_list):
            h.append(fc(feature))
        h = torch.cat(h, 0)
        res_attn = None
        for l in range(self.num_layers):
            h, res_attn = self.gat_layers[l](self.g, h, e_feat, res_attn=res_attn)
            h = h.flatten(1)
        logits, _ = self.gat_layers[-1](self.g, h, e_feat, res_attn=None)
        logits = logits.mean(1)
        logits = logits / torch.max(torch.norm(logits, dim=1, keepdim=True), self.epsilon)
        return logits


class acm_hGCN(nn.Module):

    def __init__(self, g, in_dims, num_hidden, num_classes, num_layers, activation, dropout):
        super(acm_hGCN, self).__init__()
        self.g = g
        self.layers = nn.ModuleList()
        self.fc_list = nn.ModuleList([nn.Linear(in_dim, num_hidden, bias=True) for in_dim in in_dims])
        for fc in self.fc_list:
            nn.init.xavier_normal_(fc.weight, gain=1.414)
        self.layers.append(GraphConv(num_hidden, num_hidden, activation=activation, weight=False))
        for i in range(num_layers - 1):
            self.layers.append(GraphConv(num_hidden, num_hidden, activation=activation))
        self.dropout = nn.Dropout(p=dropout)
        self.sigmoid = nn.Sigmoid()

    def message_func(self, edges):
        res = edges.src['ft']
        _mask = edges.data['e'] == 0
        mask = torch.broadcast_to(_mask.permute(1, 0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft0'], torch.zeros_like(edges.src['ft']))
        res0 = msg
        _mask = edges.data['e'] == 2
        mask = torch.broadcast_to(_mask.permute(1, 0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft2'], torch.zeros_like(edges.src['ft']))
        res2 = msg
        _mask = edges.data['e'] == 4
        mask = torch.broadcast_to(_mask.permute(1, 0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft4'], torch.zeros_like(edges.src['ft']))
        res4 = msg
        _mask = edges.data['e'] == 6
        mask = torch.broadcast_to(_mask.permute(1, 0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft6'], torch.zeros_like(edges.src['ft']))
        res6 = msg
        return {'m': res + res0 + res2 + res4 + res6}

    def forward(self, graph, features_list, e_feat):
        e_feat = e_feat.reshape(-1, 1).float()
        h = []
        for fc, feature in zip(self.fc_list, features_list):
            h.append(fc(feature))
        h = torch.cat(h, 0)
        for i, layer in enumerate(self.layers):
            h = self.dropout(h)
            h = layer(self.g, h)
        graph.edata.update({'e': e_feat})
        graph.srcdata.update({'ft': h, 'ft0': h, 'ft2': h, 'ft4': h, 'ft6': h})
        graph.update_all(self.message_func, fn.sum('m', 'ft'))
        res = graph.dstdata['ft']
        return res


class acm_hGCN_each_loss(nn.Module):

    def __init__(self, g, in_dims, num_hidden, num_classes, num_layers, activation, dropout):
        super(acm_hGCN_each_loss, self).__init__()
        self.g = g
        self.layers = nn.ModuleList()
        self.fc_list = nn.ModuleList([nn.Linear(in_dim, num_hidden, bias=True) for in_dim in in_dims])
        for fc in self.fc_list:
            nn.init.xavier_normal_(fc.weight, gain=1.414)
        self.layers.append(GraphConv(num_hidden, num_hidden, activation=activation, weight=False))
        for i in range(num_layers - 1):
            self.layers.append(GraphConv(num_hidden, num_hidden, activation=activation))
        self.dropout = nn.Dropout(p=dropout)
        self.sigmoid = nn.Sigmoid()

    def message_func(self, edges):
        res = edges.src['ft']
        return {'m': res}

    def message_func0(self, edges):
        _mask = edges.data['e'] == 0
        mask = torch.broadcast_to(_mask.permute(1, 0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft0'], torch.zeros_like(edges.src['ft']))
        return {'m': msg}

    def message_func2(self, edges):
        _mask = edges.data['e'] == 2
        mask = torch.broadcast_to(_mask.permute(1, 0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft0'], torch.zeros_like(edges.src['ft']))
        return {'m': msg}

    def message_func4(self, edges):
        _mask = edges.data['e'] == 4
        mask = torch.broadcast_to(_mask.permute(1, 0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft0'], torch.zeros_like(edges.src['ft']))
        return {'m': msg}

    def message_func6(self, edges):
        _mask = edges.data['e'] == 6
        mask = torch.broadcast_to(_mask.permute(1, 0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft0'], torch.zeros_like(edges.src['ft']))
        return {'m': msg}

    def forward(self, graph, features_list, e_feat):
        e_feat = e_feat.reshape(-1, 1).float()
        h = []
        for fc, feature in zip(self.fc_list, features_list):
            h.append(fc(feature))
        h = torch.cat(h, 0)
        for i, layer in enumerate(self.layers):
            h = self.dropout(h)
            h = layer(self.g, h)
        graph.edata.update({'e': e_feat})
        graph.srcdata.update({'ft': h, 'ft0': h, 'ft2': h, 'ft4': h, 'ft6': h})
        graph.update_all(self.message_func, fn.sum('m', 'ft'))
        graph.update_all(self.message_func0, fn.sum('m', 'ft0'))
        graph.update_all(self.message_func0, fn.sum('m', 'ft2'))
        graph.update_all(self.message_func0, fn.sum('m', 'ft4'))
        graph.update_all(self.message_func0, fn.sum('m', 'ft6'))
        res = graph.dstdata['ft']
        res0 = graph.dstdata['ft0']
        res2 = graph.dstdata['ft2']
        res4 = graph.dstdata['ft4']
        res6 = graph.dstdata['ft6']
        return res, res0, res2, res4, res6


class acm_sem_hGCN(nn.Module):

    def __init__(self, g, in_dims, num_hidden, num_classes, num_layers, activation, dropout):
        super(acm_sem_hGCN, self).__init__()
        self.g = g
        self.layers = nn.ModuleList()
        self.fc_list = nn.ModuleList([nn.Linear(in_dim, num_hidden, bias=True) for in_dim in in_dims])
        for fc in self.fc_list:
            nn.init.xavier_normal_(fc.weight, gain=1.414)
        self.layers.append(GraphConv(num_hidden, num_hidden, activation=activation, weight=False))
        for i in range(num_layers - 1):
            self.layers.append(GraphConv(num_hidden, num_hidden, activation=activation))
        self.semantic_weight = torch.nn.Parameter(torch.Tensor(1, 5))
        self.dropout = nn.Dropout(p=dropout)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.semantic_weight)

    def message_func(self, edges):
        res = edges.src['ft'] * self.semantic_weight[0, 0]
        _mask = edges.data['e'] == 0
        mask = torch.broadcast_to(_mask.permute(1, 0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft0'], torch.zeros_like(edges.src['ft']))
        res0 = msg * self.semantic_weight[0, 1]
        _mask = edges.data['e'] == 2
        mask = torch.broadcast_to(_mask.permute(1, 0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft2'], torch.zeros_like(edges.src['ft']))
        res2 = msg * self.semantic_weight[0, 2]
        _mask = edges.data['e'] == 4
        mask = torch.broadcast_to(_mask.permute(1, 0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft4'], torch.zeros_like(edges.src['ft']))
        res4 = msg * self.semantic_weight[0, 3]
        _mask = edges.data['e'] == 6
        mask = torch.broadcast_to(_mask.permute(1, 0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft6'], torch.zeros_like(edges.src['ft']))
        res6 = msg * self.semantic_weight[0, 4]
        return {'m': res + res0 + res2 + res4 + res6}

    def forward(self, graph, features_list, e_feat):
        e_feat = e_feat.reshape(-1, 1).float()
        h = []
        for fc, feature in zip(self.fc_list, features_list):
            h.append(fc(feature))
        h = torch.cat(h, 0)
        for i, layer in enumerate(self.layers):
            h = self.dropout(h)
            h = layer(self.g, h)
        graph.edata.update({'e': e_feat})
        graph.srcdata.update({'ft': h, 'ft0': h, 'ft2': h, 'ft4': h, 'ft6': h})
        graph.update_all(self.message_func, fn.sum('m', 'ft'))
        res = graph.dstdata['ft']
        return res


class semantic_GCN(nn.Module):

    def __init__(self, g, in_dims, num_hidden, num_classes, num_layers, activation, dropout):
        super(semantic_GCN, self).__init__()
        self.g = g
        self.layers = nn.ModuleList()
        self.fc_list = nn.ModuleList([nn.Linear(in_dim, num_hidden, bias=True) for in_dim in in_dims])
        for fc in self.fc_list:
            nn.init.xavier_normal_(fc.weight, gain=1.414)
        self.layers.append(GraphConv(num_hidden, num_hidden, activation=activation, weight=False))
        for i in range(num_layers - 1):
            self.layers.append(GraphConv(num_hidden, num_hidden, activation=activation))
        self.semantic_weight = torch.nn.Parameter(torch.Tensor(1, 5))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, features_list):
        h = []
        for fc, feature in zip(self.fc_list, features_list):
            h.append(fc(feature))
        h = torch.cat(h, 0)
        for i, layer in enumerate(self.layers):
            h = self.dropout(h)
            h = layer(self.g, h)
        return h, self.semantic_weight


class freebase_source_hGCN(nn.Module):

    def __init__(self, g, in_dims, num_hidden, num_classes, num_layers, activation, dropout):
        super(freebase_source_hGCN, self).__init__()
        self.g = g
        self.layers = nn.ModuleList()
        self.fc_list = nn.ModuleList([nn.Linear(in_dim, num_hidden, bias=True) for in_dim in in_dims])
        for fc in self.fc_list:
            nn.init.xavier_normal_(fc.weight, gain=1.414)
        self.layers.append(GraphConv(num_hidden, num_hidden, activation=activation, weight=False))
        for i in range(num_layers - 1):
            self.layers.append(GraphConv(num_hidden, num_hidden, activation=activation))
        self.dropout = nn.Dropout(p=dropout)
        self.sigmoid = nn.Sigmoid()

    def message_func(self, edges):
        res = edges.src['ft']
        _mask = edges.data['e'] == 0
        mask = torch.broadcast_to(_mask.permute(1, 0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft0'], torch.zeros_like(edges.src['ft']))
        res0 = msg
        _mask = edges.data['e'] == 1
        mask = torch.broadcast_to(_mask.permute(1, 0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft1'], torch.zeros_like(edges.src['ft']))
        res1 = msg
        _mask = edges.data['e'] == 2
        mask = torch.broadcast_to(_mask.permute(1, 0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft2'], torch.zeros_like(edges.src['ft']))
        res2 = msg
        _mask = edges.data['e'] == 3
        mask = torch.broadcast_to(_mask.permute(1, 0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft3'], torch.zeros_like(edges.src['ft']))
        res3 = msg
        _mask = edges.data['e'] == 4
        mask = torch.broadcast_to(_mask.permute(1, 0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft4'], torch.zeros_like(edges.src['ft']))
        res4 = msg
        return {'m': res + res0 + res1 + res2 + res3 + res4}

    def forward(self, graph, features_list, e_feat):
        e_feat = e_feat.reshape(-1, 1).float()
        h = []
        for fc, feature in zip(self.fc_list, features_list):
            h.append(fc(feature))
        h = torch.cat(h, 0)
        for i, layer in enumerate(self.layers):
            h = self.dropout(h)
            h = layer(self.g, h)
        graph.edata.update({'e': e_feat})
        graph.srcdata.update({'ft': h, 'ft0': h, 'ft1': h, 'ft2': h, 'ft3': h, 'ft4': h})
        graph.update_all(self.message_func, fn.sum('m', 'ft'))
        res = graph.dstdata['ft']
        return res


class freebase_des_hGCN(nn.Module):

    def __init__(self, g, in_dims, num_hidden, num_classes, num_layers, activation, dropout):
        super(freebase_des_hGCN, self).__init__()
        self.g = g
        self.layers = nn.ModuleList()
        self.fc_list = nn.ModuleList([nn.Linear(in_dim, num_hidden, bias=True) for in_dim in in_dims])
        for fc in self.fc_list:
            nn.init.xavier_normal_(fc.weight, gain=1.414)
        self.layers.append(GraphConv(num_hidden, num_hidden, activation=activation, weight=False))
        for i in range(num_layers - 1):
            self.layers.append(GraphConv(num_hidden, num_hidden, activation=activation))
        self.dropout = nn.Dropout(p=dropout)
        self.sigmoid = nn.Sigmoid()

    def message_func(self, edges):
        res = edges.src['ft']
        _mask = edges.data['e'] == 0
        mask = torch.broadcast_to(_mask.permute(1, 0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft0'], torch.zeros_like(edges.src['ft']))
        res0 = msg
        _mask = edges.data['e'] == 6
        mask = torch.broadcast_to(_mask.permute(1, 0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft1'], torch.zeros_like(edges.src['ft']))
        res1 = msg
        _mask = edges.data['e'] == 14
        mask = torch.broadcast_to(_mask.permute(1, 0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft2'], torch.zeros_like(edges.src['ft']))
        res2 = msg
        _mask = edges.data['e'] == 30
        mask = torch.broadcast_to(_mask.permute(1, 0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft3'], torch.zeros_like(edges.src['ft']))
        res3 = msg
        return {'m': res + res0 + res1 + res2 + res3}

    def forward(self, graph, features_list, e_feat):
        e_feat = e_feat.reshape(-1, 1).float()
        h = []
        for fc, feature in zip(self.fc_list, features_list):
            h.append(fc(feature))
        h = torch.cat(h, 0)
        for i, layer in enumerate(self.layers):
            h = self.dropout(h)
            h = layer(self.g, h)
        graph.edata.update({'e': e_feat})
        graph.srcdata.update({'ft': h, 'ft0': h, 'ft1': h, 'ft2': h, 'ft3': h})
        graph.update_all(self.message_func, fn.sum('m', 'ft'))
        res = graph.dstdata['ft']
        return res


class freebase_bi_hGCN(nn.Module):

    def __init__(self, g, in_dims, num_hidden, num_classes, num_layers, activation, dropout):
        super(freebase_bi_hGCN, self).__init__()
        self.g = g
        self.layers = nn.ModuleList()
        self.fc_list = nn.ModuleList([nn.Linear(in_dim, num_hidden, bias=True) for in_dim in in_dims])
        for fc in self.fc_list:
            nn.init.xavier_normal_(fc.weight, gain=1.414)
        self.layers.append(GraphConv(num_hidden, num_hidden, activation=activation, weight=False))
        for i in range(num_layers - 1):
            self.layers.append(GraphConv(num_hidden, num_hidden, activation=activation))
        self.dropout = nn.Dropout(p=dropout)
        self.sigmoid = nn.Sigmoid()

    def message_func(self, edges):
        res = edges.src['ft']
        _mask = edges.data['e'] == 0
        mask = torch.broadcast_to(_mask.permute(1, 0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft0'], torch.zeros_like(edges.src['ft']))
        res0 = msg
        _mask = edges.data['e'] == 1
        mask = torch.broadcast_to(_mask.permute(1, 0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft1'], torch.zeros_like(edges.src['ft']))
        res1 = msg
        _mask = edges.data['e'] == 2
        mask = torch.broadcast_to(_mask.permute(1, 0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft2'], torch.zeros_like(edges.src['ft']))
        res2 = msg
        _mask = edges.data['e'] == 3
        mask = torch.broadcast_to(_mask.permute(1, 0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft3'], torch.zeros_like(edges.src['ft']))
        res3 = msg
        _mask = edges.data['e'] == 4
        mask = torch.broadcast_to(_mask.permute(1, 0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft4'], torch.zeros_like(edges.src['ft']))
        res4 = msg
        return {'m': res + res0 + res1 + res2 + res3 + res4}

    def message_func0(self, edges):
        _mask = edges.data['e'] == 6
        mask = torch.broadcast_to(_mask.permute(1, 0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft6'], torch.zeros_like(edges.src['ft']))
        res1 = msg
        _mask = edges.data['e'] == 14
        mask = torch.broadcast_to(_mask.permute(1, 0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft14'], torch.zeros_like(edges.src['ft']))
        res2 = msg
        _mask = edges.data['e'] == 30
        mask = torch.broadcast_to(_mask.permute(1, 0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft30'], torch.zeros_like(edges.src['ft']))
        res3 = msg
        return {'m': res1 + res2 + res3}

    def forward(self, graph, trans_graph, features_list, e_feat):
        e_feat = e_feat.reshape(-1, 1).float()
        h = []
        for fc, feature in zip(self.fc_list, features_list):
            h.append(fc(feature))
        h = torch.cat(h, 0)
        for i, layer in enumerate(self.layers):
            h = self.dropout(h)
            h = layer(self.g, h)
        graph.edata.update({'e': e_feat})
        graph.srcdata.update({'ft': h, 'ft0': h, 'ft1': h, 'ft2': h, 'ft3': h, 'ft4': h})
        graph.update_all(self.message_func, fn.sum('m', 'ft'))
        trans_graph.edata.update({'e': e_feat})
        trans_graph.srcdata.update({'ft': h, 'ft6': h, 'ft14': h, 'ft30': h})
        trans_graph.update_all(self.message_func0, fn.sum('m', 'ft'))
        res = graph.dstdata['ft'] + trans_graph.dstdata['ft']
        return res


class GAT(nn.Module):

    def __init__(self, g, in_dims, num_hidden, num_classes, num_layers, heads, activation, feat_drop, attn_drop, negative_slope, residual):
        super(GAT, self).__init__()
        self.g = g
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation
        self.fc_list = nn.ModuleList([nn.Linear(in_dim, num_hidden, bias=True) for in_dim in in_dims])
        for fc in self.fc_list:
            nn.init.xavier_normal_(fc.weight, gain=1.414)
        self.gat_layers.append(GATConv(num_hidden, num_hidden, heads[0], feat_drop, attn_drop, negative_slope, False, self.activation))
        for l in range(1, num_layers):
            self.gat_layers.append(GATConv(num_hidden * heads[l - 1], num_hidden, heads[l], feat_drop, attn_drop, negative_slope, residual, self.activation))
        self.gat_layers.append(GATConv(num_hidden * heads[-2], num_hidden, heads[-1], feat_drop, attn_drop, negative_slope, residual, None))

    def forward(self, features_list):
        h = []
        for fc, feature in zip(self.fc_list, features_list):
            h.append(fc(feature))
        h = torch.cat(h, 0)
        for l in range(self.num_layers):
            h = self.gat_layers[l](self.g, h).flatten(1)
        logits = self.gat_layers[-1](self.g, h).mean(1)
        return logits


class dblp_hGAT(nn.Module):

    def __init__(self, g, in_dims, num_hidden, num_classes, num_layers, heads, activation, feat_drop, attn_drop, negative_slope, residual):
        super(dblp_hGAT, self).__init__()
        self.g = g
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation
        self.fc_list = nn.ModuleList([nn.Linear(in_dim, num_hidden, bias=True) for in_dim in in_dims])
        for fc in self.fc_list:
            nn.init.xavier_normal_(fc.weight, gain=1.414)
        self.gat_layers.append(GATConv(num_hidden, num_hidden, heads[0], feat_drop, attn_drop, negative_slope, False, self.activation))
        for l in range(1, num_layers):
            self.gat_layers.append(GATConv(num_hidden * heads[l - 1], num_hidden, heads[l], feat_drop, attn_drop, negative_slope, residual, self.activation))
        self.gat_layers.append(GATConv(num_hidden * heads[-2], num_hidden, heads[-1], feat_drop, attn_drop, negative_slope, residual, None))

    def message_func_onehop(self, edges):
        _mask = edges.data['e'] == 3
        mask = torch.broadcast_to(_mask.permute(1, 0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft_twohop'], torch.zeros_like(edges.src['ft']))
        res0 = msg
        return {'m': res0}

    def message_func_twohop(self, edges):
        res = edges.src['ft']
        _mask = edges.data['e'] == 0
        mask = torch.broadcast_to(_mask.permute(1, 0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft0'], torch.zeros_like(edges.src['ft']))
        res0 = msg
        _mask = edges.data['e'] == 4
        mask = torch.broadcast_to(_mask.permute(1, 0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft4'], torch.zeros_like(edges.src['ft']))
        res1 = msg
        _mask = edges.data['e'] == 5
        mask = torch.broadcast_to(_mask.permute(1, 0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft5'], torch.zeros_like(edges.src['ft']))
        res2 = msg
        return {'m': res + res0 + res1 + res2}

    def forward(self, graph, features_list, e_feat):
        e_feat = e_feat.reshape(-1, 1).float()
        h = []
        for fc, feature in zip(self.fc_list, features_list):
            h.append(fc(feature))
        h = torch.cat(h, 0)
        for l in range(self.num_layers):
            h = self.gat_layers[l](self.g, h).flatten(1)
        h = self.gat_layers[-1](self.g, h).mean(1)
        graph.edata.update({'e': e_feat})
        graph.srcdata.update({'ft': h, 'ft0': h, 'ft3': h, 'ft4': h, 'ft5': h})
        graph.update_all(self.message_func_twohop, fn.sum('m', 'ft_twohop'))
        graph.srcdata.update({'ft_twohop': graph.dstdata['ft_twohop']})
        graph.update_all(self.message_func_onehop, fn.sum('m', 'ft_onehop'))
        res = graph.dstdata['ft_onehop']
        return res


class node_prompt_layer_feature_weighted_sum(nn.Module):

    def __init__(self, input_dim):
        super(node_prompt_layer_feature_weighted_sum, self).__init__()
        self.weight = torch.nn.Parameter(torch.Tensor(1, input_dim))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, graph, graph_embedding):
        emb = F.elu(graph_embedding * self.weight)
        graph.srcdata.update({'ft': emb})
        graph.update_all(fn.copy_u('ft', 'm'), fn.sum('m', 'ft'))
        return graph.dstdata['ft']


class freebase_node_prompt_layer_feature_weighted_sum(nn.Module):

    def __init__(self, input_dim, g):
        super(freebase_node_prompt_layer_feature_weighted_sum, self).__init__()
        self.weight = torch.nn.Parameter(torch.Tensor(1, input_dim))
        self.r_graph = dgl.reverse(g)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, graph, graph_embedding):
        emb = F.elu(graph_embedding * self.weight)
        r_graph = self.r_graph
        graph.srcdata.update({'ft': emb})
        graph.update_all(fn.copy_u('ft', 'm'), fn.sum('m', 'ft'))
        r_graph.srcdata.update({'ft': emb})
        r_graph.update_all(fn.copy_u('ft', 'm'), fn.sum('m', 'ft'))
        return graph.dstdata['ft'] + r_graph.dstdata['ft']


class node_bottle_net(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(node_bottle_net, self).__init__()
        self.linear0 = torch.nn.Linear(input_dim, hidden_dim)
        self.linear1 = torch.nn.Linear(hidden_dim, output_dim)

    def message_func(edges):
        return {'m': edges.dst['h']}

    def forward(self, graph, graph_embedding):
        emb = F.elu(self.linear1(self.linear0(graph_embedding)))
        graph.srcdata.update({'ft': emb})
        graph.dstdata.update({'ft': emb})
        graph.update_all(fn.copy_u('ft', 'm'), fn.sum('m', 'ft'))
        return graph.dstdata['ft']


class hnode_prompt_layer_feature_weighted_sum(nn.Module):

    def __init__(self, input_dim, negative_slope=0.2):
        super(hnode_prompt_layer_feature_weighted_sum, self).__init__()
        self.weight = torch.nn.Parameter(torch.Tensor(1, input_dim))
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)

    def message_func(self, edges):
        return {'m': torch.cat((edges.src['ft'], edges.data['e']), dim=1)}

    def forward(self, graph, graph_embedding, e_feat):
        e_feat = e_feat.reshape(-1, 1).float()
        emb = graph_embedding * self.weight
        graph.srcdata.update({'ft': emb})
        graph.edata.update({'e': e_feat})
        graph.update_all(self.message_func, fn.sum('m', 'ft'))
        res = graph.dstdata['ft']
        return res


class acm_hnode_prompt_layer_feature_weighted_sum(nn.Module):

    def __init__(self, input_dim, pre_semantic_weight=None, negative_slope=0.2):
        super(acm_hnode_prompt_layer_feature_weighted_sum, self).__init__()
        self.weight = torch.nn.Parameter(torch.Tensor(1, input_dim))
        self.weight0 = torch.nn.Parameter(torch.Tensor(1, input_dim))
        self.weight2 = torch.nn.Parameter(torch.Tensor(1, input_dim))
        self.weight4 = torch.nn.Parameter(torch.Tensor(1, input_dim))
        self.weight6 = torch.nn.Parameter(torch.Tensor(1, input_dim))
        self.pre_semantic_weight = pre_semantic_weight
        self.semantic_weight = torch.nn.Parameter(torch.Tensor(1, 5))
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.reset_parameters()

    def weighted_sum(self, input, weights):
        weights = weights.expand(input.permute(1, 2, 0).shape).permute(2, 0, 1)
        weighted_inputs = input * weights
        output = weighted_inputs
        return output

    def message_func(self, edges):
        res = edges.src['ft']
        _mask = edges.data['e'] == 0
        mask = torch.broadcast_to(_mask.permute(1, 0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft0'], torch.zeros_like(edges.src['ft']))
        res0 = msg
        _mask = edges.data['e'] == 2
        mask = torch.broadcast_to(_mask.permute(1, 0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft2'], torch.zeros_like(edges.src['ft']))
        res2 = msg
        _mask = edges.data['e'] == 4
        mask = torch.broadcast_to(_mask.permute(1, 0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft4'], torch.zeros_like(edges.src['ft']))
        res4 = msg
        _mask = edges.data['e'] == 6
        mask = torch.broadcast_to(_mask.permute(1, 0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft6'], torch.zeros_like(edges.src['ft']))
        res6 = msg
        return {'m': res + res0 + res2 + res4 + res6}

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)
        torch.nn.init.xavier_uniform_(self.weight0)
        torch.nn.init.xavier_uniform_(self.weight2)
        torch.nn.init.xavier_uniform_(self.weight4)
        torch.nn.init.xavier_uniform_(self.weight6)
        torch.nn.init.xavier_uniform_(self.semantic_weight)

    def forward(self, graph, graph_embedding, e_feat):
        e_feat = e_feat.reshape(-1, 1).float()
        emb = emb0 = emb2 = emb4 = emb6 = F.elu(graph_embedding * self.weight)
        graph.edata.update({'e': e_feat})
        graph.srcdata.update({'ft': emb, 'ft0': emb0, 'ft2': emb2, 'ft4': emb4, 'ft6': emb6})
        graph.update_all(self.message_func, fn.sum('m', 'ft'))
        res = graph.dstdata['ft']
        return res


class acm_hnode_semantic_prompt_layer_feature_weighted_sum(nn.Module):

    def __init__(self, input_dim, pre_semantic_weight=None, negative_slope=0.2, semantic_prompt_weight=0.1):
        super(acm_hnode_semantic_prompt_layer_feature_weighted_sum, self).__init__()
        self.weight = torch.nn.Parameter(torch.Tensor(1, input_dim))
        self.weight0 = torch.nn.Parameter(torch.Tensor(1, input_dim))
        self.weight2 = torch.nn.Parameter(torch.Tensor(1, input_dim))
        self.weight4 = torch.nn.Parameter(torch.Tensor(1, input_dim))
        self.weight6 = torch.nn.Parameter(torch.Tensor(1, input_dim))
        self.pre_semantic_weight = pre_semantic_weight
        self.semantic_weight = torch.nn.Parameter(torch.Tensor(1, 5))
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.semantic_prompt_weight = semantic_prompt_weight
        self.reset_parameters()

    def weighted_sum(self, input, weights):
        weights = weights.expand(input.permute(1, 2, 0).shape).permute(2, 0, 1)
        weighted_inputs = input * weights
        output = weighted_inputs
        return output

    def message_func(self, edges):
        res = edges.src['ft']
        _mask = edges.data['e'] == 0
        mask = torch.broadcast_to(_mask.permute(1, 0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft0'], torch.zeros_like(edges.src['ft']))
        res0 = msg
        _mask = edges.data['e'] == 2
        mask = torch.broadcast_to(_mask.permute(1, 0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft2'], torch.zeros_like(edges.src['ft']))
        res2 = msg
        _mask = edges.data['e'] == 4
        mask = torch.broadcast_to(_mask.permute(1, 0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft4'], torch.zeros_like(edges.src['ft']))
        res4 = msg
        _mask = edges.data['e'] == 6
        mask = torch.broadcast_to(_mask.permute(1, 0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft6'], torch.zeros_like(edges.src['ft']))
        res6 = msg
        return {'m': res + res0 + res2 + res4 + res6}

    def message_func_semantic(self, edges):
        semantic = self.semantic_weight
        semantic = F.normalize(semantic, p=2, dim=1)
        res = edges.src['ft'] * semantic[0, 0]
        _mask = edges.data['e'] == 0
        mask = torch.broadcast_to(_mask.permute(1, 0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft0'], torch.zeros_like(edges.src['ft']))
        res0 = msg * semantic[0, 1]
        _mask = edges.data['e'] == 2
        mask = torch.broadcast_to(_mask.permute(1, 0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft2'], torch.zeros_like(edges.src['ft']))
        res2 = msg * semantic[0, 2]
        _mask = edges.data['e'] == 4
        mask = torch.broadcast_to(_mask.permute(1, 0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft4'], torch.zeros_like(edges.src['ft']))
        res4 = msg * semantic[0, 3]
        _mask = edges.data['e'] == 6
        mask = torch.broadcast_to(_mask.permute(1, 0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft6'], torch.zeros_like(edges.src['ft']))
        res6 = msg * semantic[0, 4]
        return {'m': res + res0 + res2 + res4 + res6}

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)
        torch.nn.init.xavier_uniform_(self.weight0)
        torch.nn.init.xavier_uniform_(self.weight2)
        torch.nn.init.xavier_uniform_(self.weight4)
        torch.nn.init.xavier_uniform_(self.weight6)
        torch.nn.init.xavier_uniform_(self.semantic_weight)

    def forward(self, graph, graph_embedding, e_feat):
        e_feat = e_feat.reshape(-1, 1).float()
        emb = emb0 = emb2 = emb4 = emb6 = F.elu(graph_embedding * self.weight)
        graph.edata.update({'e': e_feat})
        graph.srcdata.update({'ft': emb, 'ft0': emb0, 'ft2': emb2, 'ft4': emb4, 'ft6': emb6})
        graph.update_all(self.message_func_semantic, fn.sum('m', 'ft_s'))
        graph.update_all(self.message_func, fn.sum('m', 'ft'))
        res = graph.dstdata['ft'] + self.semantic_prompt_weight * graph.dstdata['ft_s']
        return res


class acm_eachloss_hnode_prompt_layer_feature_weighted_sum(nn.Module):

    def __init__(self, input_dim, pre_semantic_weight=None, negative_slope=0.2):
        super(acm_eachloss_hnode_prompt_layer_feature_weighted_sum, self).__init__()
        self.weight = torch.nn.Parameter(torch.Tensor(1, input_dim))
        self.weight0 = torch.nn.Parameter(torch.Tensor(1, input_dim))
        self.weight2 = torch.nn.Parameter(torch.Tensor(1, input_dim))
        self.weight4 = torch.nn.Parameter(torch.Tensor(1, input_dim))
        self.weight6 = torch.nn.Parameter(torch.Tensor(1, input_dim))
        self.pre_semantic_weight = pre_semantic_weight
        self.semantic_weight = torch.nn.Parameter(torch.Tensor(1, 5))
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.reset_parameters()

    def weighted_sum(self, input, weights):
        weights = weights.expand(input.permute(1, 2, 0).shape).permute(2, 0, 1)
        weighted_inputs = input * weights
        output = weighted_inputs
        return output

    def message_func(self, edges):
        res = edges.src['ft']
        return {'m': res}

    def message_func0(self, edges):
        _mask = edges.data['e'] == 0
        mask = torch.broadcast_to(_mask.permute(1, 0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft0'], torch.zeros_like(edges.src['ft']))
        return {'m': msg}

    def message_func2(self, edges):
        _mask = edges.data['e'] == 2
        mask = torch.broadcast_to(_mask.permute(1, 0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft0'], torch.zeros_like(edges.src['ft']))
        return {'m': msg}

    def message_func4(self, edges):
        _mask = edges.data['e'] == 4
        mask = torch.broadcast_to(_mask.permute(1, 0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft0'], torch.zeros_like(edges.src['ft']))
        return {'m': msg}

    def message_func6(self, edges):
        _mask = edges.data['e'] == 6
        mask = torch.broadcast_to(_mask.permute(1, 0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft0'], torch.zeros_like(edges.src['ft']))
        return {'m': msg}

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)
        torch.nn.init.xavier_uniform_(self.weight0)
        torch.nn.init.xavier_uniform_(self.weight2)
        torch.nn.init.xavier_uniform_(self.weight4)
        torch.nn.init.xavier_uniform_(self.weight6)
        torch.nn.init.xavier_uniform_(self.semantic_weight)

    def forward(self, graph, graph_embedding, e_feat):
        e_feat = e_feat.reshape(-1, 1).float()
        emb = graph_embedding * self.weight
        emb0 = graph_embedding * self.weight
        emb2 = graph_embedding * self.weight
        emb4 = graph_embedding * self.weight
        emb6 = graph_embedding * self.weight
        graph.edata.update({'e': e_feat})
        graph.srcdata.update({'ft': emb, 'ft0': emb0, 'ft2': emb2, 'ft4': emb4, 'ft6': emb6})
        graph.update_all(self.message_func, fn.sum('m', 'ft'))
        graph.update_all(self.message_func0, fn.sum('m', 'ft0'))
        graph.update_all(self.message_func0, fn.sum('m', 'ft2'))
        graph.update_all(self.message_func0, fn.sum('m', 'ft4'))
        graph.update_all(self.message_func0, fn.sum('m', 'ft6'))
        res = graph.dstdata['ft']
        res0 = graph.dstdata['ft0']
        res2 = graph.dstdata['ft2']
        res4 = graph.dstdata['ft4']
        res6 = graph.dstdata['ft6']
        return res, res0, res2, res4, res6


class acm_meta_path_hnode_prompt_layer_feature_weighted_sum(nn.Module):

    def __init__(self, input_dim, negative_slope=0.2):
        super(acm_meta_path_hnode_prompt_layer_feature_weighted_sum, self).__init__()
        self.weight = torch.nn.Parameter(torch.Tensor(1, input_dim))
        self.weight0 = torch.nn.Parameter(torch.Tensor(1, input_dim))
        self.weight2 = torch.nn.Parameter(torch.Tensor(1, input_dim))
        self.weight4 = torch.nn.Parameter(torch.Tensor(1, input_dim))
        self.weight6 = torch.nn.Parameter(torch.Tensor(1, input_dim))
        self.semantic_weight = torch.nn.Parameter(torch.Tensor(1, 5))
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.reset_parameters()

    def weighted_sum(self, input, weights):
        weights = weights.expand(input.permute(1, 2, 0).shape).permute(2, 0, 1)
        weighted_inputs = input * weights
        output = weighted_inputs
        return output

    def message_func(self, edges):
        res = edges.src['ft'] * self.semantic_weight[0, 0]
        _mask = edges.data['e'] == 0
        mask = torch.broadcast_to(_mask.permute(1, 0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft0'], torch.zeros_like(edges.src['ft']))
        res0 = msg * self.semantic_weight[0, 1]
        _mask = edges.data['e'] == 2
        mask = torch.broadcast_to(_mask.permute(1, 0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft2'], torch.zeros_like(edges.src['ft']))
        res2 = msg * self.semantic_weight[0, 2]
        _mask = edges.data['e'] == 4
        mask = torch.broadcast_to(_mask.permute(1, 0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft4'], torch.zeros_like(edges.src['ft']))
        res4 = msg * self.semantic_weight[0, 3]
        _mask = edges.data['e'] == 6
        mask = torch.broadcast_to(_mask.permute(1, 0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft6'], torch.zeros_like(edges.src['ft']))
        res6 = msg * self.semantic_weight[0, 4]
        return {'m': res + res0 + res2 + res4 + res6}

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)
        torch.nn.init.xavier_uniform_(self.weight0)
        torch.nn.init.xavier_uniform_(self.weight2)
        torch.nn.init.xavier_uniform_(self.weight4)
        torch.nn.init.xavier_uniform_(self.weight6)
        torch.nn.init.xavier_uniform_(self.semantic_weight)

    def forward(self, graph, graph_embedding, e_feat):
        e_feat = e_feat.reshape(-1, 1).float()
        emb = graph_embedding * self.weight
        emb0 = graph_embedding * self.weight
        emb2 = graph_embedding * self.weight
        emb4 = graph_embedding * self.weight
        emb6 = graph_embedding * self.weight
        graph.edata.update({'e': e_feat})
        graph.srcdata.update({'ft': emb, 'ft0': emb0, 'ft2': emb2, 'ft4': emb4, 'ft6': emb6})
        graph.update_all(self.message_func, fn.sum('m', 'ft'))
        res = graph.dstdata['ft']
        return res


class freebase_bidirection_semantic_hnode_prompt_layer_feature_weighted_sum(nn.Module):

    def __init__(self, input_dim, pre_semantic_weight=None, negative_slope=0.2, semantic_prompt_weight=0.1):
        super(freebase_bidirection_semantic_hnode_prompt_layer_feature_weighted_sum, self).__init__()
        self.weight = torch.nn.Parameter(torch.Tensor(1, input_dim))
        self.semantic_weight = torch.nn.Parameter(torch.Tensor(1, 9))
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.semantic_prompt_weight = semantic_prompt_weight
        self.reset_parameters()

    def weighted_sum(self, input, weights):
        weights = weights.expand(input.permute(1, 2, 0).shape).permute(2, 0, 1)
        weighted_inputs = input * weights
        output = weighted_inputs
        return output

    def message_func(self, edges):
        res = edges.src['ft']
        _mask = edges.data['e'] == 0
        mask = torch.broadcast_to(_mask.permute(1, 0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft0'], torch.zeros_like(edges.src['ft']))
        res0 = msg
        _mask = edges.data['e'] == 1
        mask = torch.broadcast_to(_mask.permute(1, 0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft1'], torch.zeros_like(edges.src['ft']))
        res1 = msg
        _mask = edges.data['e'] == 2
        mask = torch.broadcast_to(_mask.permute(1, 0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft2'], torch.zeros_like(edges.src['ft']))
        res2 = msg
        _mask = edges.data['e'] == 3
        mask = torch.broadcast_to(_mask.permute(1, 0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft3'], torch.zeros_like(edges.src['ft']))
        res3 = msg
        _mask = edges.data['e'] == 4
        mask = torch.broadcast_to(_mask.permute(1, 0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft4'], torch.zeros_like(edges.src['ft']))
        res4 = msg
        return {'m': res + res0 + res1 + res2 + res3 + res4}

    def message_func0(self, edges):
        _mask = edges.data['e'] == 6
        mask = torch.broadcast_to(_mask.permute(1, 0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft6'], torch.zeros_like(edges.src['ft']))
        res1 = msg
        _mask = edges.data['e'] == 14
        mask = torch.broadcast_to(_mask.permute(1, 0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft14'], torch.zeros_like(edges.src['ft']))
        res2 = msg
        _mask = edges.data['e'] == 30
        mask = torch.broadcast_to(_mask.permute(1, 0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft30'], torch.zeros_like(edges.src['ft']))
        res3 = msg
        return {'m': res1 + res2 + res3}

    def message_func_semantic(self, edges):
        semantic = self.semantic_weight
        semantic = F.normalize(semantic, p=2, dim=1)
        res = edges.src['ft']
        res = res * semantic[0, 0]
        _mask = edges.data['e'] == 0
        mask = torch.broadcast_to(_mask.permute(1, 0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft0'], torch.zeros_like(edges.src['ft']))
        res0 = msg * semantic[0, 1]
        _mask = edges.data['e'] == 1
        mask = torch.broadcast_to(_mask.permute(1, 0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft1'], torch.zeros_like(edges.src['ft']))
        res1 = msg * semantic[0, 2]
        _mask = edges.data['e'] == 2
        mask = torch.broadcast_to(_mask.permute(1, 0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft2'], torch.zeros_like(edges.src['ft']))
        res2 = msg * semantic[0, 3]
        _mask = edges.data['e'] == 3
        mask = torch.broadcast_to(_mask.permute(1, 0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft3'], torch.zeros_like(edges.src['ft']))
        res3 = msg * semantic[0, 4]
        _mask = edges.data['e'] == 4
        mask = torch.broadcast_to(_mask.permute(1, 0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft4'], torch.zeros_like(edges.src['ft']))
        res4 = msg * semantic[0, 5]
        return {'m': res + res0 + res1 + res2 + res3 + res4}

    def message_func0_semantic(self, edges):
        semantic = self.semantic_weight
        semantic = F.normalize(semantic, p=2, dim=1)
        _mask = edges.data['e'] == 6
        mask = torch.broadcast_to(_mask.permute(1, 0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft6'], torch.zeros_like(edges.src['ft']))
        res1 = msg * semantic[0, 6]
        _mask = edges.data['e'] == 14
        mask = torch.broadcast_to(_mask.permute(1, 0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft14'], torch.zeros_like(edges.src['ft']))
        res2 = msg * semantic[0, 7]
        _mask = edges.data['e'] == 30
        mask = torch.broadcast_to(_mask.permute(1, 0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft30'], torch.zeros_like(edges.src['ft']))
        res3 = msg * semantic[0, 8]
        return {'m': res1 + res2 + res3}

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)
        torch.nn.init.xavier_uniform_(self.semantic_weight)

    def forward(self, graph, trans_graph, graph_embedding, e_feat):
        e_feat = e_feat.reshape(-1, 1).float()
        emb = emb0 = emb1 = emb2 = emb3 = emb4 = emb6 = emb14 = emb30 = F.elu(graph_embedding * self.weight)
        graph.edata.update({'e': e_feat})
        graph.srcdata.update({'ft': emb, 'ft0': emb0, 'ft1': emb1, 'ft2': emb2, 'ft3': emb3, 'ft4': emb4})
        graph.update_all(self.message_func_semantic, fn.sum('m', 'ft_semantic'))
        trans_graph.edata.update({'e': e_feat})
        trans_graph.srcdata.update({'ft': emb, 'ft6': emb6, 'ft14': emb14, 'ft30': emb30})
        trans_graph.update_all(self.message_func0_semantic, fn.sum('m', 'ft_semantic'))
        res = graph.dstdata['ft'] + trans_graph.dstdata['ft'] + self.semantic_prompt_weight * (graph.dstdata['ft_semantic'] + trans_graph.dstdata['ft_semantic'])
        return res


class freebase_bidirection_hnode_prompt_layer_feature_weighted_sum(nn.Module):

    def __init__(self, input_dim, pre_semantic_weight=None, negative_slope=0.2):
        super(freebase_bidirection_hnode_prompt_layer_feature_weighted_sum, self).__init__()
        self.weight = torch.nn.Parameter(torch.Tensor(1, input_dim))
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.reset_parameters()

    def weighted_sum(self, input, weights):
        weights = weights.expand(input.permute(1, 2, 0).shape).permute(2, 0, 1)
        weighted_inputs = input * weights
        output = weighted_inputs
        return output

    def message_func(self, edges):
        res = edges.src['ft']
        _mask = edges.data['e'] == 0
        mask = torch.broadcast_to(_mask.permute(1, 0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft0'], torch.zeros_like(edges.src['ft']))
        res0 = msg
        _mask = edges.data['e'] == 1
        mask = torch.broadcast_to(_mask.permute(1, 0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft1'], torch.zeros_like(edges.src['ft']))
        res1 = msg
        _mask = edges.data['e'] == 2
        mask = torch.broadcast_to(_mask.permute(1, 0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft2'], torch.zeros_like(edges.src['ft']))
        res2 = msg
        _mask = edges.data['e'] == 3
        mask = torch.broadcast_to(_mask.permute(1, 0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft3'], torch.zeros_like(edges.src['ft']))
        res3 = msg
        _mask = edges.data['e'] == 4
        mask = torch.broadcast_to(_mask.permute(1, 0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft4'], torch.zeros_like(edges.src['ft']))
        res4 = msg
        return {'m': res + res0 + res1 + res2 + res3 + res4}

    def message_func0(self, edges):
        _mask = edges.data['e'] == 6
        mask = torch.broadcast_to(_mask.permute(1, 0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft6'], torch.zeros_like(edges.src['ft']))
        res1 = msg
        _mask = edges.data['e'] == 14
        mask = torch.broadcast_to(_mask.permute(1, 0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft14'], torch.zeros_like(edges.src['ft']))
        res2 = msg
        _mask = edges.data['e'] == 30
        mask = torch.broadcast_to(_mask.permute(1, 0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft30'], torch.zeros_like(edges.src['ft']))
        res3 = msg
        return {'m': res1 + res2 + res3}

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, graph, trans_graph, graph_embedding, e_feat):
        e_feat = e_feat.reshape(-1, 1).float()
        emb = emb0 = emb1 = emb2 = emb3 = emb4 = emb6 = emb14 = emb30 = F.elu(graph_embedding * self.weight)
        graph.edata.update({'e': e_feat})
        graph.srcdata.update({'ft': emb, 'ft0': emb0, 'ft1': emb1, 'ft2': emb2, 'ft3': emb3, 'ft4': emb4})
        graph.update_all(self.message_func, fn.sum('m', 'ft'))
        trans_graph.edata.update({'e': e_feat})
        trans_graph.srcdata.update({'ft': emb, 'ft6': emb6, 'ft14': emb14, 'ft30': emb30})
        trans_graph.update_all(self.message_func0, fn.sum('m', 'ft'))
        res = graph.dstdata['ft'] + trans_graph.dstdata['ft']
        return res


class freebase_source_hnode_prompt_layer_feature_weighted_sum(nn.Module):

    def __init__(self, input_dim, pre_semantic_weight=None, negative_slope=0.2):
        super(freebase_source_hnode_prompt_layer_feature_weighted_sum, self).__init__()
        self.weight = torch.nn.Parameter(torch.Tensor(1, input_dim))
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.reset_parameters()

    def weighted_sum(self, input, weights):
        weights = weights.expand(input.permute(1, 2, 0).shape).permute(2, 0, 1)
        weighted_inputs = input * weights
        output = weighted_inputs
        return output

    def message_func(self, edges):
        res = edges.src['ft']
        _mask = edges.data['e'] == 0
        mask = torch.broadcast_to(_mask.permute(1, 0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft0'], torch.zeros_like(edges.src['ft']))
        res0 = msg
        _mask = edges.data['e'] == 1
        mask = torch.broadcast_to(_mask.permute(1, 0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft1'], torch.zeros_like(edges.src['ft']))
        res1 = msg
        _mask = edges.data['e'] == 2
        mask = torch.broadcast_to(_mask.permute(1, 0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft2'], torch.zeros_like(edges.src['ft']))
        res2 = msg
        _mask = edges.data['e'] == 3
        mask = torch.broadcast_to(_mask.permute(1, 0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft3'], torch.zeros_like(edges.src['ft']))
        res3 = msg
        _mask = edges.data['e'] == 4
        mask = torch.broadcast_to(_mask.permute(1, 0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft4'], torch.zeros_like(edges.src['ft']))
        res4 = msg
        return {'m': res + res0 + res1 + res2 + res3 + res4}

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, graph, graph_embedding, e_feat):
        e_feat = e_feat.reshape(-1, 1).float()
        emb = emb0 = emb1 = emb2 = emb3 = emb4 = F.elu(graph_embedding * self.weight)
        graph.edata.update({'e': e_feat})
        graph.srcdata.update({'ft': emb, 'ft0': emb0, 'ft1': emb1, 'ft2': emb2, 'ft3': emb3, 'ft4': emb4})
        graph.update_all(self.message_func, fn.sum('m', 'ft'))
        res = graph.dstdata['ft']
        return res


class freebase_des_hnode_prompt_layer_feature_weighted_sum(nn.Module):

    def __init__(self, input_dim, pre_semantic_weight=None, negative_slope=0.2):
        super(freebase_des_hnode_prompt_layer_feature_weighted_sum, self).__init__()
        self.weight = torch.nn.Parameter(torch.Tensor(1, input_dim))
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.reset_parameters()

    def weighted_sum(self, input, weights):
        weights = weights.expand(input.permute(1, 2, 0).shape).permute(2, 0, 1)
        weighted_inputs = input * weights
        output = weighted_inputs
        return output

    def message_func(self, edges):
        res = edges.src['ft']
        _mask = edges.data['e'] == 0
        mask = torch.broadcast_to(_mask.permute(1, 0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft0'], torch.zeros_like(edges.src['ft']))
        res0 = msg
        _mask = edges.data['e'] == 6
        mask = torch.broadcast_to(_mask.permute(1, 0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft1'], torch.zeros_like(edges.src['ft']))
        res1 = msg
        _mask = edges.data['e'] == 14
        mask = torch.broadcast_to(_mask.permute(1, 0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft2'], torch.zeros_like(edges.src['ft']))
        res2 = msg
        _mask = edges.data['e'] == 30
        mask = torch.broadcast_to(_mask.permute(1, 0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft3'], torch.zeros_like(edges.src['ft']))
        res3 = msg
        return {'m': res + res0 + res1 + res2 + res3}

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, graph, graph_embedding, e_feat):
        e_feat = e_feat.reshape(-1, 1).float()
        emb = emb0 = emb1 = emb2 = emb3 = F.elu(graph_embedding * self.weight)
        graph.edata.update({'e': e_feat})
        graph.srcdata.update({'ft': emb, 'ft0': emb0, 'ft1': emb1, 'ft2': emb2, 'ft3': emb3})
        graph.update_all(self.message_func, fn.sum('m', 'ft'))
        res = graph.dstdata['ft']
        return res


class dblp_hnode_prompt_layer_feature_weighted_sum(nn.Module):

    def __init__(self, input_dim, pre_semantic_weight=None, negative_slope=0.2):
        super(dblp_hnode_prompt_layer_feature_weighted_sum, self).__init__()
        self.weight = torch.nn.Parameter(torch.Tensor(1, input_dim))
        self.weight0 = torch.nn.Parameter(torch.Tensor(1, input_dim))
        self.weight1 = torch.nn.Parameter(torch.Tensor(1, input_dim))
        self.weight2 = torch.nn.Parameter(torch.Tensor(1, input_dim))
        self.pre_semantic_weight = pre_semantic_weight
        self.semantic_weight = torch.nn.Parameter(torch.Tensor(1, 4))
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.reset_parameters()

    def weighted_sum(self, input, weights):
        weights = weights.expand(input.permute(1, 2, 0).shape).permute(2, 0, 1)
        weighted_inputs = input * weights
        output = weighted_inputs
        return output

    def message_func_onehop(self, edges):
        _mask = edges.data['e'] == 3
        mask = torch.broadcast_to(_mask.permute(1, 0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft_twohop'], torch.zeros_like(edges.src['ft']))
        res0 = msg
        return {'m': res0}

    def message_func_twohop(self, edges):
        res = edges.src['ft']
        _mask = edges.data['e'] == 0
        mask = torch.broadcast_to(_mask.permute(1, 0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft0'], torch.zeros_like(edges.src['ft']))
        res0 = msg
        _mask = edges.data['e'] == 4
        mask = torch.broadcast_to(_mask.permute(1, 0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft4'], torch.zeros_like(edges.src['ft']))
        res1 = msg
        _mask = edges.data['e'] == 5
        mask = torch.broadcast_to(_mask.permute(1, 0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft5'], torch.zeros_like(edges.src['ft']))
        res2 = msg
        return {'m': res + res0 + res1 + res2}

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)
        torch.nn.init.xavier_uniform_(self.weight0)
        torch.nn.init.xavier_uniform_(self.weight1)
        torch.nn.init.xavier_uniform_(self.weight2)
        torch.nn.init.xavier_uniform_(self.semantic_weight)

    def forward(self, graph, graph_embedding, e_feat):
        e_feat = e_feat.reshape(-1, 1).float()
        emb = emb0 = emb3 = emb4 = emb5 = F.elu(graph_embedding * self.weight)
        graph.edata.update({'e': e_feat})
        graph.srcdata.update({'ft': emb, 'ft0': emb0, 'ft3': emb3, 'ft4': emb4, 'ft5': emb5})
        graph.update_all(self.message_func_twohop, fn.sum('m', 'ft_twohop'))
        graph.srcdata.update({'ft_twohop': graph.dstdata['ft_twohop']})
        graph.update_all(self.message_func_onehop, fn.sum('m', 'ft_onehop'))
        res = graph.dstdata['ft_onehop']
        return res


class dblp_hnode_semantic_prompt_layer_feature_weighted_sum(nn.Module):

    def __init__(self, input_dim, pre_semantic_weight=None, negative_slope=0.2, semantic_prompt_weight=0.1):
        super(dblp_hnode_semantic_prompt_layer_feature_weighted_sum, self).__init__()
        self.weight = torch.nn.Parameter(torch.Tensor(1, input_dim))
        self.weight0 = torch.nn.Parameter(torch.Tensor(1, input_dim))
        self.weight1 = torch.nn.Parameter(torch.Tensor(1, input_dim))
        self.weight2 = torch.nn.Parameter(torch.Tensor(1, input_dim))
        self.pre_semantic_weight = pre_semantic_weight
        self.semantic_weight = torch.nn.Parameter(torch.Tensor(1, 5))
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.semantic_prompt_weight = semantic_prompt_weight
        self.reset_parameters()

    def weighted_sum(self, input, weights):
        weights = weights.expand(input.permute(1, 2, 0).shape).permute(2, 0, 1)
        weighted_inputs = input * weights
        output = weighted_inputs
        return output

    def message_func_onehop(self, edges):
        _mask = edges.data['e'] == 3
        mask = torch.broadcast_to(_mask.permute(1, 0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft_twohop'], torch.zeros_like(edges.src['ft']))
        res0 = msg
        return {'m': res0}

    def message_func_twohop(self, edges):
        res = edges.src['ft']
        _mask = edges.data['e'] == 0
        mask = torch.broadcast_to(_mask.permute(1, 0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft0'], torch.zeros_like(edges.src['ft']))
        res0 = msg
        _mask = edges.data['e'] == 4
        mask = torch.broadcast_to(_mask.permute(1, 0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft4'], torch.zeros_like(edges.src['ft']))
        res1 = msg
        _mask = edges.data['e'] == 5
        mask = torch.broadcast_to(_mask.permute(1, 0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft5'], torch.zeros_like(edges.src['ft']))
        res2 = msg
        return {'m': res + res0 + res1 + res2}

    def message_func_onehop_semantic(self, edges):
        semantic = self.semantic_weight
        semantic = F.normalize(semantic, p=2, dim=1)
        _mask = edges.data['e'] == 3
        mask = torch.broadcast_to(_mask.permute(1, 0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft_twohop_s'], torch.zeros_like(edges.src['ft']))
        res0 = msg * semantic[0, 0]
        return {'m': res0}

    def message_func_twohop_semantic(self, edges):
        semantic = self.semantic_weight
        semantic = F.normalize(semantic, p=2, dim=1)
        res = edges.src['ft'] * semantic[0, 1]
        _mask = edges.data['e'] == 0
        mask = torch.broadcast_to(_mask.permute(1, 0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft0'], torch.zeros_like(edges.src['ft']))
        res0 = msg * semantic[0, 2]
        _mask = edges.data['e'] == 4
        mask = torch.broadcast_to(_mask.permute(1, 0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft4'], torch.zeros_like(edges.src['ft']))
        res1 = msg * semantic[0, 3]
        _mask = edges.data['e'] == 5
        mask = torch.broadcast_to(_mask.permute(1, 0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft5'], torch.zeros_like(edges.src['ft']))
        res2 = msg * semantic[0, 4]
        return {'m': res + res0 + res1 + res2}

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)
        torch.nn.init.xavier_uniform_(self.weight0)
        torch.nn.init.xavier_uniform_(self.weight1)
        torch.nn.init.xavier_uniform_(self.weight2)
        torch.nn.init.xavier_uniform_(self.semantic_weight)

    def forward(self, graph, graph_embedding, e_feat):
        e_feat = e_feat.reshape(-1, 1).float()
        emb = emb0 = emb3 = emb4 = emb5 = F.elu(graph_embedding * self.weight)
        graph.edata.update({'e': e_feat})
        graph.srcdata.update({'ft': emb, 'ft0': emb0, 'ft3': emb3, 'ft4': emb4, 'ft5': emb5})
        graph.update_all(self.message_func_twohop, fn.sum('m', 'ft_twohop'))
        graph.srcdata.update({'ft_twohop': graph.dstdata['ft_twohop']})
        graph.update_all(self.message_func_onehop, fn.sum('m', 'ft_onehop'))
        graph.update_all(self.message_func_twohop_semantic, fn.sum('m', 'ft_twohop_s'))
        graph.srcdata.update({'ft_twohop_s': graph.dstdata['ft_twohop_s']})
        graph.update_all(self.message_func_onehop_semantic, fn.sum('m', 'ft_onehop_s'))
        res = graph.dstdata['ft_onehop'] + self.semantic_prompt_weight * graph.dstdata['ft_onehop_s']
        return res


class node_prompt_layer_feature_cat(nn.Module):

    def __init__(self, prompt_dim):
        super(node_prompt_layer_feature_cat, self).__init__()
        self.weight = torch.nn.Parameter(torch.Tensor(1, prompt_dim))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, graph, graph_embedding):
        graph_embedding = torch.cat([graph_embedding, torch.broadcast_to(self.weight, (graph_embedding.size(0), self.weight.size(1)))], dim=1)
        emb = graph_embedding
        graph.srcdata.update({'ft': emb})
        graph.update_all(fn.copy_u('ft', 'm'), fn.sum('m', 'ft'))
        res = graph.dstdata['ft']
        return res


class node_prompt_layer_feature_cat_edge(nn.Module):

    def __init__(self, prompt_dim):
        super(node_prompt_layer_feature_cat_edge, self).__init__()
        self.weight = torch.nn.Parameter(torch.Tensor(1, prompt_dim))
        self.prompt_dim = prompt_dim
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)

    def message_func(self, edges):
        return {'m': torch.cat((edges.src['ft'], edges.data['p']), dim=1)}

    def forward(self, graph, graph_embedding):
        emb = graph_embedding
        graph.srcdata.update({'ft': emb})
        enum = graph.num_edges()
        graph.edata.update({'p': torch.broadcast_to(self.weight, (enum, self.prompt_dim))})
        graph.update_all(self.message_func, fn.sum('m', 'ft'))
        res = graph.dstdata['ft']
        return res


class hnode_prompt_layer_feature_cat_edge(nn.Module):

    def __init__(self, prompt_dim, heterprompt_dim):
        super(hnode_prompt_layer_feature_cat_edge, self).__init__()
        self.weight = torch.nn.Parameter(torch.Tensor(1, prompt_dim))
        self.hetero_prompt = torch.nn.Parameter(torch.Tensor(1, heterprompt_dim))
        self.hetero_dim = heterprompt_dim
        self.hetero_prompt = torch.nn.Parameter(torch.Tensor(1, heterprompt_dim))
        self.prompt_dim = prompt_dim
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)
        torch.nn.init.xavier_uniform_(self.hetero_prompt)

    def message_func(self, edges):
        return {'m': torch.cat((edges.src['ft'] * edges.data['p'], edges.data['e']), dim=1)}

    def forward(self, graph, graph_embedding, e_feat):
        graph.srcdata.update({'ft': graph_embedding})
        enum = graph.num_edges()
        graph.edata.update({'p': torch.broadcast_to(self.weight, (enum, self.prompt_dim))})
        graph.edata.update({'hp': torch.broadcast_to(self.hetero_prompt, (enum, self.hetero_dim))})
        graph.edata.update({'e': e_feat})
        graph.update_all(self.message_func, fn.sum('m', 'ft'))
        res = graph.dstdata['ft']
        return res


class node_prompt_layer_feature_sum(nn.Module):

    def __init__(self):
        super(node_prompt_layer_feature_sum, self).__init__()

    def forward(self, graph, graph_embedding):
        graph.srcdata.update({'ft': graph_embedding})
        graph.update_all(fn.copy_u('ft', 'm'), fn.sum('m', 'ft'))
        return graph.dstdata['ft']


class hnode_prompt_layer_feature_sum(nn.Module):

    def __init__(self, negative_slope=0.2):
        super(hnode_prompt_layer_feature_sum, self).__init__()
        self.leaky_relu = nn.LeakyReLU(negative_slope)

    def message_func(self, edges):
        return {'m': torch.cat((edges.src['ft'], edges.data['e']), dim=1)}

    def forward(self, graph, graph_embedding, e_feat):
        e_feat = e_feat.reshape(-1, 1).float()
        emb = graph_embedding
        graph.srcdata.update({'ft': emb})
        graph.edata.update({'e': e_feat})
        graph.update_all(self.message_func, fn.sum('m', 'ft'))
        res = graph.dstdata['ft']
        return res


class hprompt_gcn(nn.Module):

    def __init__(self, input_dim, negative_slope=0.2):
        super(hprompt_gcn, self).__init__()
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.layers = nn.ModuleList()
        self.layers.append(GraphConv(input_dim, input_dim, weight=False))
        for i in range(2 - 1):
            self.layers.append(GraphConv(input_dim, input_dim))
        self.dropout = nn.Dropout(p=0.2)

    def message_func(self, edges):
        res = edges.src['ft']
        _mask = edges.data['e'] == 0
        mask = torch.broadcast_to(_mask.permute(1, 0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft0'], torch.zeros_like(edges.src['ft']))
        res0 = msg
        _mask = edges.data['e'] == 2
        mask = torch.broadcast_to(_mask.permute(1, 0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft2'], torch.zeros_like(edges.src['ft']))
        res2 = msg
        _mask = edges.data['e'] == 4
        mask = torch.broadcast_to(_mask.permute(1, 0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft4'], torch.zeros_like(edges.src['ft']))
        res4 = msg
        _mask = edges.data['e'] == 6
        mask = torch.broadcast_to(_mask.permute(1, 0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft6'], torch.zeros_like(edges.src['ft']))
        res6 = msg
        return {'m': res + res0 + res2 + res4 + res6}

    def forward(self, graph, graph_embedding, e_feat):
        e_feat = e_feat.reshape(-1, 1).float()
        for i, layer in enumerate(self.layers):
            graph_embedding = self.dropout(graph_embedding)
            graph_embedding = layer(graph, graph_embedding)
        emb = graph_embedding
        emb0 = graph_embedding
        emb2 = graph_embedding
        emb4 = graph_embedding
        emb6 = graph_embedding
        graph.edata.update({'e': e_feat})
        graph.srcdata.update({'ft': emb, 'ft0': emb0, 'ft2': emb2, 'ft4': emb4, 'ft6': emb6})
        graph.update_all(self.message_func, fn.sum('m', 'ft'))
        res = graph.dstdata['ft']
        return res


class prompt_gcn(nn.Module):

    def __init__(self, input_dim):
        super(prompt_gcn, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GraphConv(input_dim, input_dim, weight=False))
        for i in range(2 - 1):
            self.layers.append(GraphConv(input_dim, input_dim))
        self.dropout = nn.Dropout(p=0.2)
        self.gcn = GraphConv(input_dim, input_dim)

    def forward(self, graph, graph_embedding):
        for i, layer in enumerate(self.layers):
            graph_embedding = self.dropout(graph_embedding)
            graph_embedding = layer(graph, graph_embedding)
        emb = graph_embedding
        graph.srcdata.update({'ft': emb})
        graph.update_all(fn.copy_u('ft', 'm'), fn.sum('m', 'ft'))
        return graph.dstdata['ft']


class GraphChannelAttLayer(nn.Module):
    """
    The graph channel attention layer in equation 7, 9 and 10 of paper.
    """

    def __init__(self, num_channel):
        super(GraphChannelAttLayer, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(num_channel, 1, 1))
        nn.init.constant_(self.weight, 0.1)

    def forward(self, adj_list):
        """
        Parameters
        ----------
        adj_list : list
            The list of adjacent matrices.
        """
        adj_list = torch.stack(adj_list)
        adj_list = F.normalize(adj_list, dim=1, p=1)
        return torch.sum(adj_list * F.softmax(self.weight, dim=0), dim=0)


class MetricCalcLayer(nn.Module):
    """
    Calculate metric in equation 3 of paper.

    Parameters
    ----------
    nhid : int
        The dimension of mapped features in the graph generating procedure.
    """

    def __init__(self, nhid):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(1, nhid))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, h):
        """
        Parameters
        ----------
        h : tensor
            The result of the Hadamard product in equation 3 of paper.
        """
        return h * self.weight


class GraphGenerator(nn.Module):
    """
    Generate a graph using similarity.
    """

    def __init__(self, dim, num_head=2, threshold=0.1, dev=None):
        super(GraphGenerator, self).__init__()
        self.threshold = threshold
        self.metric_layer = nn.ModuleList()
        for i in range(num_head):
            self.metric_layer.append(MetricCalcLayer(dim))
        self.num_head = num_head
        self.dev = dev

    def forward(self, left_h, right_h):
        """
        Parameters
        ----------
        left_h : tensor
            The first input embedding matrix.
        right_h : tensor
            The second input embedding matrix.
        """

        def cos_sim(a, b, eps=1e-08):
            a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
            a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
            b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
            sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
            return sim_mt
        if torch.sum(left_h) == 0 or torch.sum(right_h) == 0:
            return torch.zeros((left_h.shape[0], right_h.shape[0]))
        s = torch.zeros((left_h.shape[0], right_h.shape[0]))
        zero_lines = torch.nonzero(torch.sum(left_h, 1) == 0)
        if len(zero_lines) > 0:
            left_h[zero_lines, :] += 1e-08
        for i in range(self.num_head):
            weighted_left_h = self.metric_layer[i](left_h)
            weighted_right_h = self.metric_layer[i](right_h)
            s += cos_sim(weighted_left_h, weighted_right_h)
        s /= self.num_head
        s = torch.where(s < self.threshold, torch.zeros_like(s), s)
        return s


class HGSL(BaseModel):
    """
    HGSL, Heterogeneous Graph Structure Learning from `paper <http://www.shichuan.org/doc/100.pdf>`_.

    Parameters
    ----------
    feat_dims : dict
        The feature dimensions of different node types.
    undirected_relations : str
        The HGSL model can only handle undirected heterographs, while in the dgl.heterograph format, directed edges are
        stored in two different edge types, separately and symmetrically, to represent undirected edge. Hence you have
        to specify which relations are those distinct undirected relations. In this parameter, each undirected relation
        is separated with a comma. For example, in a heterograph with 2 undirected relations: paper-author and
        paper-subject, there are 4 type of edges stored in the dgl.heterograph: paper-author, author-paper,
        paper-subject, subject-paper. Then this parameter can be "paper-author,paper-subject",
        "author-paper,paper-subject", "paper-author,subject-paper" or "author-paper,subject-paper".
    device: str
        The GPU device to select, like 'cuda:0'.
    metapaths : list
        The metapath name list.
    mp_emb_dim : int
        The dimension of metapath embeddings from metapath2vec.
    hidden_dim : int
        The dimension of mapped features in the graph generating procedure.
    num_heads: int
        Number of heads in the K-head weighted cosine similarity function.
    fs_eps : float
        Threshold of feature similarity graph :math:`\\epsilon^{FS}`.
    fp_eps : float
        Threshold of feature propagation graph :math:`\\epsilon^{FP}`.
    mp_eps : float
        Threshold of semantic graph :math:`\\epsilon^{MP}`.
    gnn_emd_dim : int
        The dimension of hidden layers of the downstream GNN.
    gnn_dropout : float
        The dropout ratio of features in the downstream GNN.
    category : str
        The target node type which the model will predict on.
    out_dim : int
        number of classes of the target node type.

    Attributes
    -----------
    fgg_direct : nn.ModuleDict
        Feature similarity graph generator(:math:`S_r^{FS}`) dict in equation 2 of paper, in which keys are
        undirected-relation strs.
    fgg_left: nn.ModuleDict
        Feature propagation graph generator(:math:`S_r^{FH}`) dict which generates the graphs in equation 5 of paper.
    fgg_right: nn.ModuleDict
        Feature propagation graph generator(:math:`S_r^{FT}`) dict which generates the graphs in equation 6 of paper.
    fg_agg : nn.ModuleDict
        A channel attention layer, in which a layer fuses one feature similarity graph and two feature propagation
        graphs generated, in equation 7 of paper.
    sgg_gen : nn.ModuleDict
        Semantic subgraph generator(:math:`S_{r,m}^{MP}`) dict, in equation 8 of paper.
    sg_agg : nn.ModuleDict
        The channel attention layer which fuses semantic subgraphs, in equation 9 of paper.
    overall_g_agg : nn.ModuleDict
        The channel attention layer which fuses the learned feature graph, semantic graph and the original graph.
    encoder : nn.ModuleDict
        The type-specific mapping layer in equation 1 of paper.

    Note
    ----
    This model under the best config has some slight differences compared with the code given by the paper author,
    which seems having little impact on performance:

    1. The regularization item in loss is on all parameters of the model, while in the author's code, it is only on the
       generated adjacent matrix. If you want to implement the latter, a new task of OpenHGNN is needed.

    2. The normalization of input adjacent matrix is separately on different adjacent matrices of different
       relations, while in the author's code, it is on the entire adjacent matrix composed of adjacent matrices of all
       relations.
    """

    @classmethod
    def build_model_from_args(cls, args, hg):
        feat_dims = dict()
        for ntype in hg.ntypes:
            if 'h' in hg.nodes[ntype].data.keys():
                feat_dims[ntype] = hg.nodes[ntype].data['h'].shape[1]
            else:
                feat_dims[ntype] = 128
        und_rels = args.undirected_relations.split(',')
        undirected_relations = list()
        for etype in hg.canonical_etypes:
            if etype[1] in und_rels:
                undirected_relations.append(etype)
        device = hg.device
        metapaths = list()
        if args.dataset_name == 'acm4GTN':
            for feature_name in hg.nodes['paper'].data.keys():
                if 'm2v' in feature_name:
                    metapaths.append(feature_name)
            mp_emb_dim = hg.nodes['paper'].data['pap_m2v_emb'].shape[1]
        elif args.dataset_name == 'dblp4GTN':
            for feature_name in hg.nodes['paper'].data.keys():
                if 'h' not in feature_name:
                    metapaths.append(feature_name)
            mp_emb_dim = hg.nodes['paper'].data['PAPCP'].shape[1]
        elif args.dataset_name == 'yelp4HGSL':
            for feature_name in hg.nodes['b'].data.keys():
                if 'h' not in feature_name:
                    metapaths.append(feature_name)
            mp_emb_dim = hg.nodes['b'].data['bub'].shape[1]
        else:
            raise NotImplemented('HGSL on dataset {} has not been implemented'.format(args.dataset_name))
        return cls(feat_dims=feat_dims, undirected_relations=undirected_relations, device=device, metapaths=metapaths, mp_emb_dim=mp_emb_dim, hidden_dim=args.hidden_dim, num_heads=args.num_heads, fs_eps=args.fs_eps, fp_eps=args.fp_eps, mp_eps=args.mp_eps, gnn_emd_dim=args.gnn_emd_dim, gnn_dropout=args.gnn_dropout, category=args.category, num_class=args.out_dim)

    def __init__(self, feat_dims, undirected_relations, device, metapaths, mp_emb_dim, hidden_dim, num_heads, fs_eps, fp_eps, mp_eps, gnn_emd_dim, gnn_dropout, category, num_class):
        super().__init__()
        self.device = device
        self.ud_rels = undirected_relations
        self.node_types = list(feat_dims.keys())
        self.feat_dims = feat_dims
        self.non_linear = nn.ReLU()
        self.category = category
        self.metapaths = metapaths
        nnmd = nn.ModuleDict
        self.fgg_direct, self.fgg_left, self.fgg_right, self.fg_agg, self.sgg_gen, self.sg_agg, self.overall_g_agg = nnmd({}), nnmd({}), nnmd({}), nnmd({}), nnmd({}), nnmd({}), nnmd({})
        self.encoder = nnmd(dict(zip(self.node_types, [nn.Linear(feat_dims[node_type], hidden_dim) for node_type in self.node_types])))
        for canonical_etype in undirected_relations:
            undirected_relation = canonical_etype[1]
            self.fgg_direct[undirected_relation] = GraphGenerator(hidden_dim, num_heads, fs_eps, self.device)
            self.fgg_left[undirected_relation] = GraphGenerator(feat_dims[canonical_etype[0]], num_heads, fp_eps, self.device)
            self.fgg_right[undirected_relation] = GraphGenerator(feat_dims[canonical_etype[2]], num_heads, fp_eps, self.device)
            self.fg_agg[undirected_relation] = GraphChannelAttLayer(3)
            self.sgg_gen[undirected_relation] = nnmd(dict(zip(metapaths, [GraphGenerator(mp_emb_dim, num_heads, mp_eps, self.device) for _ in metapaths])))
            self.sg_agg[undirected_relation] = GraphChannelAttLayer(len(metapaths))
            self.overall_g_agg[undirected_relation] = GraphChannelAttLayer(3)
        if len(set(feat_dims.values())) == 1:
            self.GCN = GCN(list(self.feat_dims.values())[0], gnn_emd_dim, num_class, gnn_dropout)
        else:
            raise Exception('Downstream model GCN can only accept features for different node types of the same dimension')

    def forward(self, hg, h_features):
        """
        Parameters
        ----------
        hg : dgl.DGlHeteroGraph
            All input data is stored in this graph.
            The graph should be an undirected heterogeneous graph.
            Every node type in graph should have its feature named 'h' and the same feature dimension.
            Every node type in graph should have its metapath2vec embedding feature named 'xxx_m2v_emb'
            and the same feature dimension.
        h_features : dict

        Returns
        --------
        result : dict
            The target node type and the corresponding node embeddings.
        """

        def generate_node_indexes(hg):
            indexes = dict()
            index = 0
            for node_type in hg.ntypes:
                indexes[node_type] = index, index + hg.num_nodes(node_type)
                index += hg.num_nodes(node_type)
            return indexes

        def construct_homo_adj(new_adjs, hg, node_indexes, device):
            new_homo_adj = torch.zeros(size=(hg.num_nodes(), hg.num_nodes()))
            for canonical_etype, new_adj in new_adjs.items():
                row_range = node_indexes[canonical_etype[0]]
                column_range = node_indexes[canonical_etype[2]]
                new_homo_adj[row_range[0]:row_range[1], column_range[0]:column_range[1]] = new_adj
            temp = new_homo_adj.clone()
            new_homo_adj = temp + new_homo_adj.t()
            new_homo_adj = F.normalize(new_homo_adj, dim=0, p=1)
            return new_homo_adj

        def construct_homo_feature(hg, device):
            homo_feature = list()
            for ntype in hg.ntypes:
                homo_feature.append(hg.nodes[ntype].data['h'])
            homo_feature = torch.cat(homo_feature, dim=0)
            return homo_feature
        mapped_feats = dict()
        for ntype in self.node_types:
            if 'h' in hg.nodes[ntype].data.keys():
                mapped_feats[ntype] = self.non_linear(self.encoder[ntype](hg.nodes[ntype].data['h'].clone()))
            else:
                mapped_feats[ntype] = self.non_linear(self.encoder[ntype](h_features[ntype].clone()))
        new_adjs = dict()
        for canonical_etype in self.ud_rels:
            undirected_relation = canonical_etype[1]
            ori_g = F.normalize(hg.adj(etype=canonical_etype).to_dense(), dim=1, p=2)
            fg_direct = self.fgg_direct[undirected_relation](mapped_feats[canonical_etype[0]], mapped_feats[canonical_etype[2]])
            if 'h' in hg.nodes[canonical_etype[0]].data.keys() and 'h' in hg.nodes[canonical_etype[2]].data.keys():
                fmat_l, fmat_r = hg.nodes[canonical_etype[0]].data['h'], hg.nodes[canonical_etype[2]].data['h']
            else:
                fmat_l, fmat_r = h_features[canonical_etype[0]], h_features[canonical_etype[2]]
            sim_l, sim_r = self.fgg_left[undirected_relation](fmat_l, fmat_l), self.fgg_right[undirected_relation](fmat_r, fmat_r)
            fg_left, fg_right = sim_l.mm(ori_g), sim_r.mm(ori_g.t()).t()
            feat_g = self.fg_agg[undirected_relation]([fg_direct, fg_left, fg_right])
            sem_g_list = [self.sgg_gen[undirected_relation][mp](hg.nodes[canonical_etype[0]].data[mp], hg.nodes[canonical_etype[2]].data[mp]) for mp in self.metapaths]
            sem_g = self.sg_agg[undirected_relation](sem_g_list)
            new_adjs[canonical_etype] = self.overall_g_agg[undirected_relation]([feat_g, sem_g, ori_g])
        node_indexes = generate_node_indexes(hg)
        new_homo_adj = construct_homo_adj(new_adjs, hg, node_indexes, self.device)
        homo_feature = construct_homo_feature(hg, self.device)
        x = self.GCN(homo_feature, new_homo_adj)
        result = {self.category: x[node_indexes[self.category][0]:node_indexes[self.category][1], :]}
        return result


class HGTLayer(nn.Module):

    def __init__(self, in_dim, out_dim, node_dict, edge_dict, n_heads, dropout=0.2, use_norm=False):
        super(HGTLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.node_dict = node_dict
        self.edge_dict = edge_dict
        self.num_types = len(node_dict)
        self.num_relations = len(edge_dict)
        self.total_rel = self.num_types * self.num_relations * self.num_types
        self.n_heads = n_heads
        self.d_k = out_dim // n_heads
        self.sqrt_dk = math.sqrt(self.d_k)
        self.att = None
        self.k_linears = nn.ModuleList()
        self.q_linears = nn.ModuleList()
        self.v_linears = nn.ModuleList()
        self.a_linears = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.use_norm = use_norm
        for t in range(self.num_types):
            self.k_linears.append(nn.Linear(in_dim, out_dim))
            self.q_linears.append(nn.Linear(in_dim, out_dim))
            self.v_linears.append(nn.Linear(in_dim, out_dim))
            self.a_linears.append(nn.Linear(out_dim, out_dim))
            if use_norm:
                self.norms.append(nn.LayerNorm(out_dim))
        self.relation_pri = nn.Parameter(th.ones(self.num_relations, self.n_heads))
        self.relation_att = nn.Parameter(th.Tensor(self.num_relations, n_heads, self.d_k, self.d_k))
        self.relation_msg = nn.Parameter(th.Tensor(self.num_relations, n_heads, self.d_k, self.d_k))
        self.skip = nn.Parameter(th.ones(self.num_types))
        self.drop = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.relation_att)
        nn.init.xavier_uniform_(self.relation_msg)

    def forward(self, G, h):
        with G.local_scope():
            node_dict, edge_dict = self.node_dict, self.edge_dict
            for srctype, etype, dsttype in G.canonical_etypes:
                sub_graph = G[srctype, etype, dsttype]
                k_linear = self.k_linears[node_dict[srctype]]
                v_linear = self.v_linears[node_dict[srctype]]
                q_linear = self.q_linears[node_dict[dsttype]]
                k = k_linear(h[srctype]).view(-1, self.n_heads, self.d_k)
                v = v_linear(h[srctype]).view(-1, self.n_heads, self.d_k)
                q = q_linear(h[dsttype]).view(-1, self.n_heads, self.d_k)
                e_id = self.edge_dict[etype]
                relation_att = self.relation_att[e_id]
                relation_pri = self.relation_pri[e_id]
                relation_msg = self.relation_msg[e_id]
                k = th.einsum('bij,ijk->bik', k, relation_att)
                v = th.einsum('bij,ijk->bik', v, relation_msg)
                sub_graph.srcdata['k'] = k
                sub_graph.dstdata['q'] = q
                sub_graph.srcdata['v_%d' % e_id] = v
                sub_graph.apply_edges(fn.v_dot_u('q', 'k', 't'))
                attn_score = sub_graph.edata.pop('t').sum(-1) * relation_pri / self.sqrt_dk
                attn_score = edge_softmax(sub_graph, attn_score, norm_by='dst')
                sub_graph.edata['t'] = attn_score.unsqueeze(-1)
            G.multi_update_all({etype: (fn.u_mul_e('v_%d' % e_id, 't', 'm'), fn.sum('m', 't')) for etype, e_id in edge_dict.items()}, cross_reducer='mean')
            new_h = {}
            for ntype in G.ntypes:
                """
                    Step 3: Target-specific Aggregation
                    x = norm( W[node_type] * gelu( Agg(x) ) + x )
                """
                n_id = node_dict[ntype]
                alpha = th.sigmoid(self.skip[n_id])
                t = G.nodes[ntype].data['t'].view(-1, self.out_dim)
                trans_out = self.drop(self.a_linears[n_id](t))
                trans_out = trans_out * alpha + h[ntype] * (1 - alpha)
                if self.use_norm:
                    new_h[ntype] = self.norms[n_id](trans_out)
                else:
                    new_h[ntype] = trans_out
            return new_h


class HGT(BaseModel):

    @classmethod
    def build_model_from_args(cls, args, hg):
        node_dict = {}
        edge_dict = {}
        for ntype in hg.ntypes:
            node_dict[ntype] = len(node_dict)
        for etype in hg.etypes:
            edge_dict[etype] = len(edge_dict)
            hg.edges[etype].data['id'] = th.ones(hg.number_of_edges(etype), dtype=th.long) * edge_dict[etype]
        return cls(node_dict, edge_dict, args.hidden_dim, args.out_dim, args.num_layers, args.num_heads, args.dropout, category=args.category)

    def __init__(self, node_dict, edge_dict, hidden_dim, out_dim, num_layers, n_heads, dropout, category, use_norm=True):
        super(HGT, self).__init__()
        self.node_dict = node_dict
        self.edge_dict = edge_dict
        self.category = category
        self.gcs = nn.ModuleList()
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.adapt_ws = nn.ModuleList()
        for _ in range(num_layers):
            self.gcs.append(HGTLayer(hidden_dim, hidden_dim, node_dict, edge_dict, n_heads, dropout, use_norm=use_norm))
        self.out = nn.Linear(hidden_dim, out_dim)

    def forward(self, G, h_in=None):
        h = h_in
        for i in range(self.num_layers):
            h = self.gcs[i](G, h)
        return {self.category: self.out(h[self.category])}


class HMPNNLayer(nn.Module):

    def __init__(self, in_feat, out_feat, etypes, activation=None):
        super(HMPNNLayer, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.etypes = etypes
        self.conv = dglnn.HeteroGraphConv({rel: dglnn.GraphConv(in_feat, out_feat, activation=activation) for rel in self.etypes}, aggregate='sum')

    def forward(self, g, h_dict):
        with g.local_scope():
            h_dict = self.conv(g, h_dict)
        return h_dict


class HMPNN(BaseModel):
    """
    Heterogeneous Message-Passing Neural Network (HMPNN)

    A simple implementation of HMPNN from paper for experimenting in KTN.

    Parameters
    ----------
    in_dim : int
        Input feature size.
    hid_dim : int
        Hidden layer size.
    out_dim : int
        Output feature size.
    etypes : list
        Edge types.
    num_layers : int
        Number of layers.
    device : str
        Device to run the model.
        
    """

    @classmethod
    def build_model_from_args(cls, args, hg):
        return HMPNN(args.in_dim, args.hid_dim, args.hid_dim, hg.etypes, args.num_layers, args.device)

    def __init__(self, in_dim, hid_dim, out_dim, etypes, num_layers, device):
        super(HMPNN, self).__init__()
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.etypes = etypes
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        self.device = device
        None
        self.layers.append(HMPNNLayer(in_dim, hid_dim, etypes, activation=F.relu))
        for i in range(num_layers - 2):
            self.layers.append(HMPNNLayer(hid_dim, hid_dim, etypes, activation=F.relu))
        self.layers.append(HMPNNLayer(hid_dim, out_dim, etypes, activation=None))

    def forward(self, hg, h_dict):
        if hasattr(hg, 'ntypes'):
            for layer in self.layers:
                h_dict = layer(hg, h_dict)
        else:
            for layer, block in zip(self.layers, hg):
                block = block
                h_dict = layer(block, h_dict)
        return h_dict

    def input_feature(self):
        return self.dataset.get_features()


class HPNLayer(nn.Module):

    def __init__(self, meta_paths_dict, in_size, dropout, k_layer, alpha, edge_drop):
        super(HPNLayer, self).__init__()
        self.hidden = nn.Sequential(nn.Linear(in_features=in_size, out_features=in_size, bias=True), nn.ReLU())
        self.meta_paths_dict = meta_paths_dict
        semantic_attention = SemanticAttention(in_size=in_size)
        mods = nn.ModuleDict({mp: APPNPConv(k_layer, alpha, edge_drop) for mp in meta_paths_dict})
        self.model = MetapathConv(meta_paths_dict, mods, semantic_attention)
        self._cached_graph = None
        self._cached_coalesced_graph = {}

    def forward(self, g, h_dict):
        """
        Parameters
        -----------
        g : DGLHeteroGraph
            The heterogeneous graph
        h : tensor
            The input features

        Returns
        --------
        h : tensor
            The output features
        """
        h_dict = {ntype: self.hidden(h_dict[ntype]) for ntype in h_dict}
        if self._cached_graph is None or self._cached_graph is not g:
            self._cached_graph = g
            self._cached_coalesced_graph.clear()
            for mp, mp_value in self.meta_paths_dict.items():
                self._cached_coalesced_graph[mp] = dgl.metapath_reachable_graph(g, mp_value)
        h_dict = self.model(self._cached_coalesced_graph, h_dict)
        return h_dict


class HPN(BaseModel):
    """
    This model shows an example of using dgl.metapath_reachable_graph on the original heterogeneous
    graph.HPN from paper `Heterogeneous Graph Propagation Network
    <https://ieeexplore.ieee.org/abstract/document/9428609>`__.
    The author did not provide codes. So, we complete it according to the implementation of HAN


    .. math::
        \\mathbf{Z}^{\\Phi}=\\mathcal{P}_{\\Phi}(\\mathbf{X})=g_\\Phi(f_\\Phi(\\mathbf{X}))

    where :math:`\\mathbf{X}` denotes initial feature matrix and :math:`\\mathbf{Z^\\Phi}` denotes semantic-specific node embedding.

    .. math::
        \\mathbf{H}^{\\Phi}=f_\\Phi(\\mathbf{X})=\\sigma(\\mathbf{X} \\cdot \\mathbf{W}^\\Phi+\\mathbf{b}^{\\Phi})

    where :math:`\\mathbf{H}^{\\Phi}` is projected node feature matrix

    .. math::
        \\mathbf{Z}^{\\Phi, k}=g_{\\Phi}\\left(\\mathbf{Z}^{\\Phi, k-1}\\right)=(1-\\gamma) \\cdot \\mathbf{M}^{\\Phi} \\cdot \\mathbf{Z}^{\\Phi, k-1}+\\gamma \\cdot \\mathbf{H}^{\\Phi}

    where :math:`\\mathbf{Z}^{\\Phi,k}` denotes node embeddings learned by k-th layer semantic propagation mechanism. :math:`\\gamma` is a weight scalar which indicates the
    importance of characteristic of node in aggregating process.
    We use MetapathConv to finish Semantic Propagation and Semantic Fusion.



    Parameters
    ------------
    meta_paths : list
        contain multiple meta-paths.
    category : str
        The category means the head and tail node of metapaths.
    in_size : int
        input feature dimension.
    out_size : int
        out dimension.
    dropout : float
        Dropout probability.
    k_layer : int
        propagation times.
    alpha : float
        Value of restart probability.
    edge_drop : float, optional
        The dropout rate on edges that controls the
        messages received by each node. Default: ``0``.


    """

    @classmethod
    def build_model_from_args(cls, args, hg):
        if args.meta_paths_dict is None:
            meta_paths = extract_metapaths(args.category, hg.canonical_etypes)
        else:
            meta_paths = args.meta_paths_dict
        return cls(meta_paths=meta_paths, category=args.out_node_type, in_size=args.hidden_dim, out_size=args.out_dim, dropout=args.dropout, k_layer=args.k_layer, alpha=args.alpha, edge_drop=args.edge_drop)

    def __init__(self, meta_paths, category, in_size, out_size, dropout, k_layer, alpha, edge_drop):
        super(HPN, self).__init__()
        self.category = category
        self.layers = nn.ModuleList()
        self.layers.append(HPNLayer(meta_paths, in_size, dropout, k_layer, alpha, edge_drop))
        self.linear = nn.Linear(in_size, out_size)

    def forward(self, g, h_dict):
        for gnn in self.layers:
            h_dict = gnn(g, h_dict)
        out_dict = {ntype: self.linear(h_dict[ntype]) for ntype in self.category}
        return out_dict


class Contrast(nn.Module):

    def __init__(self, hidden_dim, tau, lam):
        """
        This part is used to calculate the contrastive loss.

        Returns
        -------
        contra_loss : float
            The calculated loss

        """
        super(Contrast, self).__init__()
        self.proj = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ELU(), nn.Linear(hidden_dim, hidden_dim))
        self.tau = tau
        self.lam = lam
        for model in self.proj:
            if isinstance(model, nn.Linear):
                nn.init.xavier_normal_(model.weight, gain=1.414)

    def sim(self, z1, z2):
        """
        This part is used to calculate the cosine similarity of each pair of nodes from different views.

        """
        z1_norm = torch.norm(z1, dim=-1, keepdim=True)
        z2_norm = torch.norm(z2, dim=-1, keepdim=True)
        dot_numerator = torch.mm(z1, z2.t())
        dot_denominator = torch.mm(z1_norm, z2_norm.t())
        sim_matrix = torch.exp(dot_numerator / dot_denominator / self.tau)
        return sim_matrix

    def forward(self, z_mp, z_sc, pos):
        """
        This is the forward part of contrast part.

        We firstly project the embeddings under two views into the space where contrastive loss is calculated. Then, we calculate the contrastive loss with projected embeddings in a cross-view way.

        .. math::
           \\mathcal{L}_i^{sc}=-\\log\\frac{\\sum_{j\\in\\mathbb{P}_i}exp\\left(sim\\left(z_i^{sc}\\_proj,z_j^{mp}\\_proj\\right)/\\tau\\right)}{\\sum_{k\\in\\{\\mathbb{P}_i\\bigcup\\mathbb{N}_i\\}}exp\\left(sim\\left(z_i^{sc}\\_proj,z_k^{mp}\\_proj\\right)/\\tau\\right)}

        where we show the contrastive loss :math:`\\mathcal{L}_i^{sc}` under network schema view, and :math:`\\mathbb{P}_i` and :math:`\\mathbb{N}_i` are positives and negatives for node :math:`i`.

        In a similar way, we can get the contrastive loss :math:`\\mathcal{L}_i^{mp}` under meta-path view. Finally, we utilize combination parameter :math:`\\lambda` to add this two losses.

        Note
        -----------
        In implementation, each row of 'matrix_mp2sc' means the similarity with exponential between one node in meta-path view and all nodes in network schema view. Then, we conduct normalization for this row,
        and pick the results where the pair of nodes are positives. Finally, we sum these results for each row, and give a log to get the final loss.

        """
        z_proj_mp = self.proj(z_mp)
        z_proj_sc = self.proj(z_sc)
        matrix_mp2sc = self.sim(z_proj_mp, z_proj_sc)
        matrix_sc2mp = matrix_mp2sc.t()
        matrix_mp2sc = matrix_mp2sc / (torch.sum(matrix_mp2sc, dim=1).view(-1, 1) + 1e-08)
        lori_mp = -torch.log(matrix_mp2sc.mul(pos.to_dense()).sum(dim=-1)).mean()
        matrix_sc2mp = matrix_sc2mp / (torch.sum(matrix_sc2mp, dim=1).view(-1, 1) + 1e-08)
        lori_sc = -torch.log(matrix_sc2mp.mul(pos.to_dense()).sum(dim=-1)).mean()
        contra_loss = self.lam * lori_mp + (1 - self.lam) * lori_sc
        return contra_loss


class Mp_encoder(nn.Module):

    def __init__(self, meta_paths_dict, hidden_size, attn_drop):
        """
        This part is to encode meta-path view.

        Returns
        -------
        z_mp : matrix
            The embedding matrix under meta-path view.

        """
        super(Mp_encoder, self).__init__()
        self.act = nn.PReLU()
        self.gcn_layers = nn.ModuleDict()
        for mp in meta_paths_dict:
            one_layer = GraphConv(hidden_size, hidden_size, activation=self.act, allow_zero_in_degree=True)
            one_layer.reset_parameters()
            self.gcn_layers[mp] = one_layer
        self.meta_paths_dict = meta_paths_dict
        self._cached_graph = None
        self._cached_coalesced_graph = {}
        self.semantic_attention = SelfAttention(hidden_size, attn_drop, 'mp')

    def forward(self, g, h):
        semantic_embeddings = []
        if self._cached_graph is None or self._cached_graph is not g:
            self._cached_graph = g
            self._cached_coalesced_graph.clear()
            for mp, meta_path in self.meta_paths_dict.items():
                self._cached_coalesced_graph[mp] = dgl.metapath_reachable_graph(g, meta_path)
        for mp, meta_path in self.meta_paths_dict.items():
            new_g = self._cached_coalesced_graph[mp]
            one = self.gcn_layers[mp](new_g, h)
            semantic_embeddings.append(one)
        z_mp = self.semantic_attention(semantic_embeddings)
        return z_mp


class Sc_encoder(nn.Module):

    def __init__(self, network_schema, hidden_size, attn_drop, sample_rate, category):
        """
        This part is to encode network schema view.

        Returns
        -------
        z_mp : matrix
            The embedding matrix under network schema view.

        Note
        -----------
        There is a different sampling strategy between original code and this code. In original code, the authors implement sampling without replacement if the number of neighbors exceeds a threshold,
        and with replacement if not. In this version, we simply use the API dgl.sampling.sample_neighbors to implement this operation, and set replacement as True.

        """
        super(Sc_encoder, self).__init__()
        self.gat_layers = nn.ModuleList()
        for i in range(len(network_schema)):
            one_layer = GATConv((hidden_size, hidden_size), hidden_size, num_heads=1, attn_drop=attn_drop, allow_zero_in_degree=True)
            one_layer.reset_parameters()
            self.gat_layers.append(one_layer)
        self.network_schema = list(tuple(ns) for ns in network_schema)
        self._cached_graph = None
        self._cached_coalesced_graph = {}
        self.inter = SelfAttention(hidden_size, attn_drop, 'sc')
        self.sample_rate = sample_rate
        self.category = category

    def forward(self, g, h):
        intra_embeddings = []
        for i, network_schema in enumerate(self.network_schema):
            src_type = network_schema[0]
            one_graph = g[network_schema]
            cate_num = torch.arange(0, g.num_nodes(self.category))
            sub_graph = sample_neighbors(one_graph, {self.category: cate_num}, {network_schema[1]: self.sample_rate[src_type]}, replace=True)
            one = self.gat_layers[i](sub_graph, (h[src_type], h[self.category]))
            one = one.squeeze(1)
            intra_embeddings.append(one)
        z_sc = self.inter(intra_embeddings)
        return z_sc


class HeCo(BaseModel):
    """
    **Title:** Self-supervised Heterogeneous Graph Neural Network with Co-contrastive Learning

    **Authors:** Xiao Wang, Nian Liu, Hui Han, Chuan Shi

    HeCo was introduced in `[paper] <http://shichuan.org/doc/112.pdf>`_
    and parameters are defined as follows:

    Parameters
    ----------
    meta_paths : dict
        Extract metapaths from graph
    network_schema : dict
        Directed edges from other types to target type
    category : string
        The category of the nodes to be classificated
    hidden_size : int
        Hidden units size
    feat_drop : float
        Dropout rate for projected feature
    attn_drop : float
        Dropout rate for attentions used in two view guided encoders
    sample_rate : dict
        The nuber of neighbors of each type sampled for network schema view
    tau : float
        Temperature parameter used for contrastive loss
    lam : float
        Balance parameter for two contrastive losses

    """

    @classmethod
    def build_model_from_args(cls, args, hg):
        if args.meta_paths_dict is None:
            meta_paths_dict = extract_metapaths(args.category, hg.canonical_etypes)
        else:
            meta_paths_dict = args.meta_paths_dict
        schema = []
        for etype in hg.canonical_etypes:
            if etype[2] == args.category:
                schema.append(etype)
        return cls(meta_paths_dict=meta_paths_dict, network_schema=schema, category=args.category, hidden_size=args.hidden_dim, feat_drop=args.feat_drop, attn_drop=args.attn_drop, sample_rate=args.sample_rate, tau=args.tau, lam=args.lam)

    def __init__(self, meta_paths_dict, network_schema, category, hidden_size, feat_drop, attn_drop, sample_rate, tau, lam):
        super(HeCo, self).__init__()
        self.category = category
        self.feat_drop = init_drop(feat_drop)
        self.attn_drop = attn_drop
        self.mp = Mp_encoder(meta_paths_dict, hidden_size, self.attn_drop)
        self.sc = Sc_encoder(network_schema, hidden_size, self.attn_drop, sample_rate, self.category)
        self.contrast = Contrast(hidden_size, tau, lam)

    def forward(self, g, h_dict, pos):
        """
        This is the forward part of model HeCo.

        Parameters
        ----------
        g : DGLGraph
            A DGLGraph
        h_dict: dict
            Projected features after linear projection
        pos: matrix
            A matrix to indicate the postives for each node

        Returns
        -------
        loss : float
            The optimize objective

        Note
        -----------
        Pos matrix is pre-defined by users. The relative tool is given in original code.
        """
        new_h = {}
        for key, value in h_dict.items():
            new_h[key] = F.elu(self.feat_drop(value))
        z_mp = self.mp(g, new_h[self.category])
        z_sc = self.sc(g, new_h)
        loss = self.contrast(z_mp, z_sc, pos)
        return loss

    def get_embeds(self, g, h_dict):
        """
        This is to get final embeddings of target nodes

        """
        z_mp = F.elu(h_dict[self.category])
        z_mp = self.mp(g, z_mp)
        return z_mp.detach()


class Generator(nn.Module):
    """
     A Discriminator :math:`D` eveluates the connectivity between the pair of nodes :math:`u` and :math:`v` w.r.t. a relation :math:`r`. It is formulated as follow:

    .. math::
        D(\\mathbf{e}_v|\\mathbf{u},\\mathbf{r};\\mathbf{\\theta}^D) = \\frac{1}{1+\\exp(-\\mathbf{e}_u^{D^T}) \\mathbf{M}_r^D \\mathbf{e}_v}

    where :math:`e_v \\in \\mathbb{R}^{d\\times 1}` is the input embeddings of the sample :math:`v`,
    :math:`e_u^D \\in \\mathbb{R}^{d \\times 1}` is the learnable embedding of node :math:`u`,
    :math:`M_r^D \\in \\mathbb{R}^{d \\times d}` is a learnable relation matrix for relation :math:`r`.

    There are also a two-layer MLP integrated into the generator for enhancing the expression of the fake samples:

    .. math::
        G(\\mathbf{u}, \\mathbf{r}; \\mathbf{\\theta}^G) = f(\\mathbf{W_2}f(\\mathbf{W}_1 \\mathbf{e} + \\mathbf{b}_1) + \\mathbf{b}_2)

    where :math:`e` is drawn from Gaussian distribution. :math:`\\{W_i, b_i}` denote the weight matrix and bias vector for :math:`i`-th layer.

    The discriminator Loss is :

    .. math::
        L_G = \\mathbb{E}_{\\langle u,v\\rangle \\sim P_G, e'_v \\sim G(u,r;\\theta^G)} = -\\log -D(e'_v|u,r)) +\\lambda^G || \\theta^G ||_2^2

    where :math:`\\theta^G` denote all the learnable parameters in Generator.

    Parameters
    -----------
    emb_size: int
        embeddings size.
    hg: dgl.heteroGraph
        heterogenous graph.

    """

    def __init__(self, emb_size, hg):
        super().__init__()
        self.n_relation = len(hg.etypes)
        self.node_emb_dim = emb_size
        self.nodes_embedding = nn.ParameterDict()
        for nodes_type, nodes_emb in hg.ndata['h'].items():
            self.nodes_embedding[nodes_type] = nn.Parameter(nodes_emb, requires_grad=True)
        self.relation_matrix = nn.ParameterDict()
        for et in hg.etypes:
            rm = torch.empty(self.node_emb_dim, self.node_emb_dim)
            rm = nn.init.xavier_normal_(rm)
            self.relation_matrix[et] = nn.Parameter(rm, requires_grad=True)
        self.fc = nn.Sequential(OrderedDict([('w_1', nn.Linear(in_features=self.node_emb_dim, out_features=self.node_emb_dim, bias=True)), ('a_1', nn.LeakyReLU()), ('w_2', nn.Linear(in_features=self.node_emb_dim, out_features=self.node_emb_dim)), ('a_2', nn.LeakyReLU())]))

    def forward(self, gen_hg, dis_node_emb, dis_relation_matrix, noise_emb):
        """
        Parameters
        -----------
        gen_hg: dgl.heterograph
            sampled graph for generator.
        dis_node_emb: dict[str: Tensor]
            discriminator node embedding.
        dis_relation_matrix: dict[str: Tensor]
            discriminator relation embedding.
        noise_emb: dict[str: Tensor]
            noise embedding.
        """
        score_list = []
        with gen_hg.local_scope():
            self.assign_node_data(gen_hg, dis_node_emb)
            self.assign_edge_data(gen_hg, dis_relation_matrix)
            self.generate_neighbor_emb(gen_hg, noise_emb)
            for et in gen_hg.canonical_etypes:
                gen_hg.apply_edges(lambda edges: {'s': edges.src['dh'].unsqueeze(1).matmul(edges.data['de']).squeeze()}, etype=et)
                gen_hg.apply_edges(lambda edges: {'score': edges.data['s'].multiply(edges.data['g'])}, etype=et)
                score = torch.sum(gen_hg.edata['score'].pop(et), dim=1)
                score_list.append(score)
        return torch.cat(score_list)

    def get_parameters(self):
        return {k: self.nodes_embedding[k] for k in self.nodes_embedding.keys()}

    def generate_neighbor_emb(self, hg, noise_emb):
        for et in hg.canonical_etypes:
            hg.apply_edges(lambda edges: {'g': edges.src['h'].unsqueeze(1).matmul(edges.data['e']).squeeze()}, etype=et)
            hg.apply_edges(lambda edges: {'g': edges.data['g'] + noise_emb[et]}, etype=et)
            hg.apply_edges(lambda edges: {'g': self.fc(edges.data['g'])}, etype=et)
        return {et: hg.edata['g'][et] for et in hg.canonical_etypes}

    def assign_edge_data(self, hg, dis_relation_matrix=None):
        for et in hg.canonical_etypes:
            n = hg.num_edges(et)
            e = self.relation_matrix[et[1]]
            hg.edata['e'] = {et: e.expand(n, -1, -1)}
            if dis_relation_matrix:
                de = dis_relation_matrix[et[1]]
                hg.edata['de'] = {et: de.expand(n, -1, -1)}

    def assign_node_data(self, hg, dis_node_emb=None):
        for nt in hg.ntypes:
            hg.nodes[nt].data['h'] = self.nodes_embedding[nt]
        if dis_node_emb:
            hg.ndata['dh'] = dis_node_emb


class HeGAN(BaseModel):
    """
    HeGAN was introduced in `Adversarial Learning on Heterogeneous Information Networks <https://dl.acm.org/doi/10.1145/3292500.3330970>`_

    It included a **Discriminator** and a **Generator**. For more details please read docs of both.

    Parameters
    ----------
    emb_size: int
        embedding size
    hg: dgl.heteroGraph
        hetorogeneous graph
    """

    @classmethod
    def build_model_from_args(cls, args, hg):
        return cls(args.emb_size, hg)

    def __init__(self, emb_size, hg):
        super().__init__()
        self.generator = Generator(emb_size, hg)
        self.discriminator = Discriminator(emb_size, hg)

    def forward(self, *args):
        pass

    def extra_loss(self):
        pass


class lstm_aggr(nn.Module):
    """
    Aggregate the same neighbors with LSTM
    """

    def __init__(self, dim):
        super(lstm_aggr, self).__init__()
        self.lstm = nn.LSTM(dim, int(dim / 2), 1, batch_first=True, bidirectional=True)
        self.lstm.flatten_parameters()

    def _lstm_reducer(self, nodes):
        m = nodes.mailbox['m']
        batch_size = m.shape[0]
        all_state, last_state = self.lstm(m)
        return {'neigh': th.mean(all_state, 1)}

    def forward(self, g, inputs):
        with g.local_scope():
            if isinstance(inputs, tuple) or g.is_block:
                if isinstance(inputs, tuple):
                    src_inputs, dst_inputs = inputs
                else:
                    src_inputs = inputs
                    dst_inputs = {k: v[:g.number_of_dst_nodes(k)] for k, v in inputs.items()}
                g.srcdata['h'] = src_inputs
                g.update_all(fn.copy_u('h', 'm'), self._lstm_reducer)
                h_neigh = g.dstdata['neigh']
            else:
                g.srcdata['h'] = inputs
                g.update_all(fn.copy_u('h', 'm'), self._lstm_reducer)
                h_neigh = g.dstdata['neigh']
            return h_neigh


class aggregate_het_neigh(nn.Module):
    """
    It is a Aggregating Heterogeneous Neighbors(C3)
    Same Type Neighbors Aggregation

    """

    def __init__(self, ntypes, dim):
        super(aggregate_het_neigh, self).__init__()
        self.neigh_rnn = nn.ModuleDict({})
        self.ntypes = ntypes
        for n in ntypes:
            self.neigh_rnn[n] = lstm_aggr(dim)

    def forward(self, hg, inputs):
        with hg.local_scope():
            outputs = {}
            for i in self.ntypes:
                outputs[i] = []
            if isinstance(inputs, tuple) or hg.is_block:
                if isinstance(inputs, tuple):
                    src_inputs, dst_inputs = inputs
                else:
                    src_inputs = inputs
                    dst_inputs = {k: v[:hg.number_of_dst_nodes(k)] for k, v in inputs.items()}
                for stype, etype, dtype in hg.canonical_etypes:
                    rel_graph = hg[stype, etype, dtype]
                    if rel_graph.number_of_edges() == 0:
                        continue
                    if stype not in src_inputs or dtype not in dst_inputs:
                        continue
                    dstdata = self.neigh_rnn[stype](rel_graph, (src_inputs[stype], dst_inputs[dtype]))
                    outputs[dtype].append(dstdata)
            else:
                for stype, etype, dtype in hg.canonical_etypes:
                    rel_graph = hg[stype, etype, dtype]
                    if rel_graph.number_of_edges() == 0:
                        continue
                    if stype not in inputs:
                        continue
                    dstdata = self.neigh_rnn[stype](rel_graph, inputs[stype])
                    outputs[dtype].append(dstdata)
            return outputs


class het_content_encoder(nn.Module):
    """
    The Encoding Heterogeneous Contents(C2) in the paper
    For a specific node type, encoder different content features with a LSTM.

    In paper, it is (b) NN-1: node heterogeneous contents encoder in figure 2.

    Parameters
    ------------
    dim : int
        input dimension

    Attributes
    ------------
    content_rnn : nn.Module
        nn.LSTM encode different content feature
    """

    def __init__(self, dim):
        super(het_content_encoder, self).__init__()
        self.content_rnn = nn.LSTM(dim, int(dim / 2), 1, batch_first=True, bidirectional=True)
        self.content_rnn.flatten_parameters()
        self.dim = dim

    def forward(self, h_dict):
        """

        Parameters
        ----------
        h_dict: dict[str, th.Tensor]
            key means different content feature

        Returns
        -------
        content_h : th.tensor
        """
        concate_embed = []
        for _, h in h_dict.items():
            concate_embed.append(h)
        concate_embed = th.cat(concate_embed, 1)
        concate_embed = concate_embed.view(concate_embed.shape[0], -1, self.dim)
        all_state, last_state = self.content_rnn(concate_embed)
        out_h = th.mean(all_state, 1).squeeze()
        return out_h


class Het_Aggregate(nn.Module):
    """
    The whole model of HetGNN

    Attributes
    -----------
    content_rnn : nn.Module
        het_content_encoder
    neigh_rnn : nn.Module
        aggregate_het_neigh
    atten_w : nn.ModuleDict[str, nn.Module]


    """

    def __init__(self, ntypes, dim):
        super(Het_Aggregate, self).__init__()
        self.ntypes = ntypes
        self.dim = dim
        self.content_rnn = het_content_encoder(dim)
        self.neigh_rnn = aggregate_het_neigh(ntypes, dim)
        self.atten_w = nn.ModuleDict({})
        for n in self.ntypes:
            self.atten_w[n] = nn.Linear(in_features=dim * 2, out_features=1)
        self.softmax = nn.Softmax(dim=1)
        self.activation = nn.LeakyReLU()
        self.drop = nn.Dropout(p=0.5)
        self.bn = nn.BatchNorm1d(dim)
        self.embed_d = dim

    def forward(self, hg, h_dict):
        with hg.local_scope():
            content_h = {}
            for ntype, h in h_dict.items():
                content_h[ntype] = self.content_rnn(h)
            neigh_h = self.neigh_rnn(hg, content_h)
            dst_h = {k: v[:hg.number_of_dst_nodes(k)] for k, v in content_h.items()}
            out_h = {}
            for n in self.ntypes:
                d_h = dst_h[n]
                batch_size = d_h.shape[0]
                concat_h = []
                concat_emd = []
                for i in range(len(neigh_h[n])):
                    concat_h.append(th.cat((d_h, neigh_h[n][i]), 1))
                    concat_emd.append(neigh_h[n][i])
                concat_h.append(th.cat((d_h, d_h), 1))
                concat_emd.append(d_h)
                concat_h = th.hstack(concat_h).view(batch_size * (len(self.ntypes) + 1), self.dim * 2)
                atten_w = self.activation(self.atten_w[n](concat_h)).view(batch_size, len(self.ntypes) + 1)
                atten_w = self.softmax(atten_w).view(batch_size, 1, 4)
                concat_emd = th.hstack(concat_emd).view(batch_size, len(self.ntypes) + 1, self.dim)
                weight_agg_batch = th.bmm(atten_w, concat_emd).view(batch_size, self.dim)
                out_h[n] = weight_agg_batch
            return out_h


class HetGNN(BaseModel):
    """
    HetGNN[KDD2019]-
    `Heterogeneous Graph Neural Network <https://dl.acm.org/doi/abs/10.1145/3292500.3330961>`_
    `Source Code Link <https://github.com/chuxuzhang/KDD2019_HetGNN>`_

    The author of the paper only gives the academic dataset.

    Attributes
    -----------
    Het_Aggrate : nn.Module
        Het_Aggregate
    """

    @classmethod
    def build_model_from_args(cls, args, hg):
        return cls(hg, args)

    def __init__(self, hg, args):
        super(HetGNN, self).__init__()
        self.Het_Aggregate = Het_Aggregate(hg.ntypes, args.dim)
        self.ntypes = hg.ntypes
        self.device = args.device
        self.loss_fn = HetGNN.compute_loss

    def forward(self, hg, h=None):
        if h is None:
            h = self.extract_feature(hg, self.ntypes)
        x = self.Het_Aggregate(hg, h)
        return x

    def evaluator(self):
        self.link_preddiction()
        self.node_classification()

    def get_embedding(self):
        input_features = self.model.extract_feature(self.hg, self.hg.ntypes)
        x = self.model(self.model.preprocess(self.hg, self.args), input_features)
        return x

    def link_preddiction(self):
        x = self.get_embedding()
        self.model.lp_evaluator(x[self.category].detach(), self.train_batch, self.test_batch)

    def node_classification(self):
        x = self.get_embedding()
        self.model.nc_evaluator(x[self.category].detach(), self.labels, self.train_idx, self.test_idx)

    @staticmethod
    def compute_loss(pos_score, neg_score):
        loss = []
        for i in pos_score:
            loss.append(F.logsigmoid(pos_score[i]))
            loss.append(F.logsigmoid(-neg_score[i]))
        loss = th.cat(loss)
        return -loss.mean()

    @staticmethod
    def extract_feature(g, ntypes):
        input_features = {}
        for n in ntypes:
            ndata = g.srcnodes[n].data
            data = {}
            data['dw_embedding'] = ndata['dw_embedding']
            data['abstract'] = ndata['abstract']
            if n == 'paper':
                data['title'] = ndata['title']
                data['venue'] = ndata['venue']
                data['author'] = ndata['author']
                data['reference'] = ndata['reference']
            input_features[n] = data
        return input_features

    @staticmethod
    def pred(edge_subgraph, x):
        with edge_subgraph.local_scope():
            edge_subgraph.ndata['x'] = x
            for etype in edge_subgraph.canonical_etypes:
                edge_subgraph.apply_edges(dgl.function.u_dot_v('x', 'x', 'score'), etype=etype)
            return edge_subgraph.edata['score']


class ScorePredictor(nn.Module):

    def forward(self, edge_subgraph, x):
        with edge_subgraph.local_scope():
            edge_subgraph.ndata['x'] = x
            for etype in edge_subgraph.canonical_etypes:
                edge_subgraph.apply_edges(dgl.function.u_dot_v('x', 'x', 'score'), etype=etype)
            return edge_subgraph.edata['score']


class HetSANNConv(nn.Module):
    """
    The HetSANN convolution layer.

    Parameters
    ----------
    num_heads: int
        the number of heads in the attention computing
    in_dim: int
        the input dimension of the features
    hidden_dim: int
        the hidden dimension of the features
    num_etypes: int
        the number of the edge types
    dropout: float
        the dropout rate
    negative_slope: float
        the negative slope used in the LeakyReLU
    residual: boolean
        if we need the residual operation
    activation: str
        the activation function
    """

    def __init__(self, num_heads, in_dim, hidden_dim, num_etypes, dropout, negative_slope, residual, activation):
        super(HetSANNConv, self).__init__()
        self.num_heads = num_heads
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.W = TypedLinear(self.in_dim, self.hidden_dim * self.num_heads, num_etypes)
        self.a_l = TypedLinear(self.hidden_dim * self.num_heads, self.hidden_dim * self.num_heads, num_etypes)
        self.a_r = TypedLinear(self.hidden_dim * self.num_heads, self.hidden_dim * self.num_heads, num_etypes)
        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(negative_slope)
        if residual:
            self.residual = nn.Linear(in_dim, self.hidden_dim * num_heads)
        else:
            self.register_buffer('residual', None)
        self.activation = activation

    def forward(self, g, x, ntype, etype, presorted=False):
        """
        The forward part of the HetSANNConv.

        Parameters
        ----------
        g : object
            the dgl homogeneous graph
        x: tensor
            the original features of the graph
        ntype: tensor
            the node type of the graph
        etype: tensor
            the edge type of the graph
        presorted: boolean
            if the ntype and etype are preordered, default: ``False``
            
        Returns
        -------
        tensor
            The embeddings after aggregation.
        """
        g.srcdata['h'] = x
        g.apply_edges(Fn.copy_u('h', 'm'))
        h = g.edata['m']
        feat = self.W(h, etype, presorted)
        h = self.dropout(feat)
        g.edata['m'] = h
        h = h.view(-1, self.num_heads, self.hidden_dim)
        h_l = self.a_l(h.view(-1, self.num_heads * self.hidden_dim), etype, presorted).view(-1, self.num_heads, self.hidden_dim).sum(dim=-1)
        h_r = self.a_r(h.view(-1, self.num_heads * self.hidden_dim), etype, presorted).view(-1, self.num_heads, self.hidden_dim).sum(dim=-1)
        attention = self.leakyrelu(h_l + h_r)
        attention = edge_softmax(g, attention)
        with g.local_scope():
            h = h.permute(0, 2, 1).contiguous()
            g.edata['alpha'] = h @ attention.reshape(-1, self.num_heads, 1)
            g.update_all(Fn.copy_e('m', 'w'), Fn.sum('w', 'emb'))
            h_output = g.dstdata['emb']
        if g.is_block:
            x = x[:g.num_dst_nodes()]
        if self.residual:
            res = self.residual(x)
            h_output += res
        if self.activation is not None:
            h_output = self.activation(h_output)
        return h_output


class HetSANN(BaseModel):
    """
    This is a model HetSANN from `An Attention-Based Graph Neural Network for Heterogeneous Structural Learning
    <https://arxiv.org/abs/1912.10832>`__

    It contains the following part:

    Apply a linear transformation:
    
    .. math::
       h^{(l+1, m)}_{\\phi(j),i} = W^{(l+1, m)}_{\\phi(j),\\phi(i)} h^{(l)}_i \\quad (1)
    
    And return the new embeddings.
    
    You may refer to the paper HetSANN-Section 2.1-Type-aware Attention Layer-(1)

    Aggregation of Neighborhood:
    
    Computing the attention coefficient:
    
    .. math::
       o^{(l+1,m)}_e = \\sigma(f^{(l+1,m)}_r(h^{(l+1, m)}_{\\phi(j),j}, h^{(l+1, m)}_{\\phi(j),i})) \\quad (2)
        
    .. math::
       f^{(l+1,m)}_r(e) = [h^{(l+1, m)^T}_{\\phi(j),j}||h^{(l+1, m)^T}_{\\phi(j),i}]a^{(l+1, m)}_r ] \\quad (3)
    
    .. math::
       \\alpha^{(l+1,m)}_e = exp(o^{(l+1,m)}_e) / \\sum_{k\\in \\varepsilon_j} exp(o^{(l+1,m)}_k) \\quad (4)
    
    Getting new embeddings with multi-head and residual
    
    .. math::
       h^{(l + 1, m)}_j = \\sigma(\\sum_{e = (i,j,r)\\in \\varepsilon_j} \\alpha^{(l+1,m)}_e h^{(l+1, m)}_{\\phi(j),i}) \\quad (5)
    
    Multi-heads:
    
    .. math::
       h^{(l+1)}_j = \\parallel^M_{m = 1}h^{(l + 1, m)}_j \\quad (6)
    
    Residual:
    
    .. math::
       h^{(l+1)}_j = h^{(l)}_j + \\parallel^M_{m = 1}h^{(l + 1, m)}_j \\quad (7)
    
    Parameters
    ----------
    num_heads: int
        the number of heads in the attention computing
    num_layers: int
        the number of layers we used in the computing
    in_dim: int
        the input dimension
    num_classes: int
        the number of the output classes
    num_etypes: int
        the number of the edge types
    dropout: float
        the dropout rate
    negative_slope: float
        the negative slope used in the LeakyReLU
    residual: boolean
        if we need the residual operation
    ntype: list
        the list of node type
    """

    @classmethod
    def build_model_from_args(cls, args, hg):
        return cls(args.num_heads, args.num_layers, args.hidden_dim, args.out_dim, hg.ntypes, len(hg.etypes), args.dropout, args.slope, args.residual)

    def __init__(self, num_heads, num_layers, in_dim, num_classes, ntypes, num_etypes, dropout, negative_slope, residual):
        super(HetSANN, self).__init__()
        self.num_layers = num_layers
        self.ntypes = ntypes
        self.residual = residual
        self.activation = F.elu
        self.het_layers = nn.ModuleList()
        self.het_layers.append(HetSANNConv(num_heads, in_dim, in_dim // num_heads, num_etypes, dropout, negative_slope, False, self.activation))
        for i in range(1, num_layers - 1):
            self.het_layers.append(HetSANNConv(num_heads, in_dim, in_dim // num_heads, num_etypes, dropout, negative_slope, residual, self.activation))
        self.het_layers.append(HetSANNConv(1, in_dim, num_classes, num_etypes, dropout, negative_slope, residual, None))

    def forward(self, hg, h_dict):
        """
        The forward part of the HetSANN.
        
        Parameters
        ----------
        hg : object
            the dgl heterogeneous graph
        h_dict: dict
            the feature dict of different node types
            
        Returns
        -------
        dict
            The embeddings after the output projection.
        """
        if hasattr(hg, 'ntypes'):
            with hg.local_scope():
                hg.ndata['h'] = h_dict
                g = dgl.to_homogeneous(hg, ndata='h')
                h = g.ndata['h']
                for i in range(self.num_layers - 1):
                    h = self.het_layers[i](g, h, g.ndata['_TYPE'], g.edata['_TYPE'], True)
                h = self.het_layers[-1](g, h, g.ndata['_TYPE'], g.edata['_TYPE'], True)
                h_dict = to_hetero_feat(h, g.ndata['_TYPE'], self.ntypes)
        else:
            h = h_dict
            for layer, block in zip(self.het_layers, hg):
                h = layer(block, h, block.ndata['_TYPE']['_N'], block.edata['_TYPE'], presorted=False)
            h_dict = to_hetero_feat(h, block.ndata['_TYPE']['_N'][:block.num_dst_nodes()], self.ntypes)
        return h_dict

    @property
    def to_homo_flag(self):
        return True


class BaseTask(ABC):

    def __init__(self):
        super(BaseTask, self).__init__()
        self.loss_fn = None
        self.evaluator = None

    @abstractmethod
    def evaluate(self, *args, **kwargs):
        pass


def load_dgl_graph(path_file):
    g, _ = load_graphs(path_file)
    return g[0]


def load_HIN(dataset):
    if dataset == 'acm':
        data_path = './openhgnn/dataset/acm_graph.bin'
        category = 'paper'
        num_classes = 3
    elif dataset == 'imdb':
        data_path = './openhgnn/dataset/imdb_graph.bin'
        category = 'movie'
        num_classes = 3
    elif dataset == 'acm1':
        data_path = './openhgnn/dataset/acm_graph1.bin'
        category = 'paper'
        num_classes = 3
    elif dataset == 'academic':
        data_path = './openhgnn/dataset/academic.bin'
        category = 'author'
        num_classes = 4
    g = load_dgl_graph(data_path)
    g = g.long()
    return g, category, num_classes


def load_KG(dataset):
    if dataset == 'aifb':
        kg_dataset = AIFBDataset()
    elif dataset == 'mutag':
        kg_dataset = MUTAGDataset()
    elif dataset == 'bgs':
        kg_dataset = BGSDataset()
    elif dataset == 'am':
        kg_dataset = AMDataset()
    else:
        raise ValueError()
    kg = kg_dataset[0]
    category = kg_dataset.predict_category
    num_classes = kg_dataset.num_classes
    return kg, category, num_classes


def load_OGB(dataset):
    if dataset == 'mag':
        dataset = DglNodePropPredDataset(name='ogbn-mag')
        return dataset
    elif dataset in ['biokg', 'wikikg']:
        d_name = 'ogbl-' + dataset
        dataset = DglLinkPropPredDataset(name=d_name)
        split_edge = dataset.get_edge_split()
        train_edge, valid_edge, test_edge = split_edge['train'], split_edge['valid'], split_edge['test']
        graph = dataset[0]


def build_dataset(model_name, dataset_name):
    if dataset_name in ['mag']:
        dataset = load_OGB(dataset_name)
        return dataset
    if model_name in ['GTN', 'NSHE', 'HetGNN']:
        g, category, num_classes = load_HIN(dataset_name)
    elif model_name in ['RSHN', 'RGCN', 'CompGCN']:
        g, category, num_classes = load_KG(dataset_name)
    return g, category, num_classes


class Ingram(BaseTask):
    """Recommendation tasks."""

    def __init__(self, args):
        super().__init__()
        self.logger = args.logger
        self.name_dataset = args.dataset
        self.train_dataloader, self.valid_dataloader, self.test_dataloader = build_dataset(args.dataset, 'Ingram')

    def evaluate(self, y_true, y_score):
        pass


class Model(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.user_seq_length = 15
        self.user_item_term_length = 10
        self.user_query_term_length = 10
        self.query_length = 10
        self.query_topcate_length = 3
        self.query_leafcate_length = 3
        self.embed_size_word = 64
        self._generate_model_layer()
        self.keep_prob = 0.8

    def set_mode(self, mode='train'):
        self.keep_prob = 0.8 if mode == 'train' else 1.0

    def _generate_model_layer(self):
        self._word_embed = nn.Parameter(Variable(torch.Tensor(self.args.vocab, self.embed_size_word), requires_grad=True))
        self.user_word_lstm, self.h_1, self.c_1, self.w_l1, self.b_l1, self.loss1 = self._rnn_lstm(1, self.args.batch_num, 64, 64)
        self.user_item_query_lstm, self.h_2, self.c_2, self.w_l2, self.b_l2, self.loss2 = self._rnn_lstm(1, self.args.batch_num, 64, 64)
        self.user_query_item_lstm, self.h_3, self.c_3, self.w_l3, self.b_l3, self.loss3 = self._rnn_lstm(1, self.args.batch_num, 64, 64)
        self.user_query_seq_lstm, self.h_4, self.c_4, self.w_l4, self.b_l4, self.loss4 = self._rnn_lstm(1, self.args.batch_num, 64, 64)
        self.query_item_query_cnn, self.conv_w1, self.loss5 = self._cnn(64)
        self.query_item_query_linear = nn.Linear(12 * 2 * 1, 64)
        self.query_item_query_relu = nn.ReLU()
        self.query_user_item_cnn, self.conv_w2, self.loss6 = self._cnn(64)
        self.query_user_item_linear = nn.Linear(12 * 2 * 1, 64)
        self.query_user_item_relu = nn.ReLU()
        self.wide_feat_w, self.loss7 = self.get_weights_variables([64, 81], self.args.weight_decay)
        self.wide_feat_b = self.get_bias_variables(64)
        self.concat_query_w, self.loss8 = self.get_weights_variables([64, 64 * 7], self.args.weight_decay)
        self.concat_query_b = self.get_bias_variables(64)
        self.concat_query_user_wide_w, self.loss9 = self.get_weights_variables([64, 128], self.args.weight_decay)
        self.concat_query_user_wide_b = self.get_bias_variables(64)
        self.deep_wide_feat_w, self.loss10 = self.get_weights_variables([1, 64], self.args.weight_decay)
        self.deep_wide_feat_b = self.get_bias_variables(1)

    @property
    def regular_loss(self):
        return self.get_regular_loss(self.w_l1) + self.get_regular_loss(self.w_l2) + self.get_regular_loss(self.w_l3) + self.get_regular_loss(self.w_l4) + self.get_regular_loss(self.wide_feat_w) + self.get_regular_loss(self.concat_query_w) + self.get_regular_loss(self.concat_query_user_wide_w) + self.get_regular_loss(self.deep_wide_feat_w)

    def get_regular_loss(self, params):
        return torch.sum(torch.pow(params, 2)) / 2 * self.args.weight_decay

    def get_weights_variables(self, shape, weight_decay, trainable=True):
        params = nn.Parameter(Variable(torch.Tensor(*shape), requires_grad=trainable))
        if weight_decay == 0:
            regular_loss = 0.0
        else:
            nn.init.xavier_uniform_(params, gain=1.0)
            regular_loss = torch.sum(torch.pow(params, 2)) / 2 * weight_decay
        return params, regular_loss

    def get_bias_variables(self, size, trainable=True):
        params = nn.Parameter(Variable(torch.Tensor(size, 1), requires_grad=trainable))
        nn.init.constant_(params, 0.0)
        return params

    def _rnn_lstm(self, layers_num, batches_num, features_num, hidden_size):
        lstm = nn.LSTM(features_num, hidden_size, layers_num, bias=True)
        h_0, regular_loss_h0 = self.get_weights_variables([layers_num, batches_num, hidden_size], self.args.weight_decay)
        c_0, regular_loss_c0 = self.get_weights_variables([layers_num, batches_num, hidden_size], self.args.weight_decay)
        w_l, regular_loss_wl = self.get_weights_variables([hidden_size, hidden_size], self.args.weight_decay)
        b_l = self.get_bias_variables(hidden_size)
        loss_total = regular_loss_h0 + regular_loss_c0 + regular_loss_wl
        return lstm, h_0, c_0, w_l, b_l, loss_total

    def _cnn(self, features_num):
        conv_w, regular_loss_w = self.get_weights_variables([12, 1, 2, features_num], self.args.weight_decay)
        conv_b, _ = self.get_weights_variables([12], 0)
        conv = nn.Conv2d(1, 12, kernel_size=(2, features_num), stride=1, padding='valid')
        return conv, conv_w, regular_loss_w

    def forward(self, x):
        wide_feat = x[:81, :]
        user_item_seq = x[81:276, :]
        query_feat = x[276:292, :]
        user_query_seq = x[292:462, :]
        query_item_query = x[462:562, :]
        user_query_item = x[562:662, :]
        user_item_query = x[662:812, :]
        query_user_item = x[812:, :]
        query_terms, query_topcate, query_leafcate = torch.split(query_feat, [self.query_length, self.query_topcate_length, self.query_leafcate_length], 0)
        inputs_query_raw = torch.nn.functional.embedding(query_terms, self._word_embed)
        input_num = torch.sum(torch.sign(query_terms)) if torch.sum(torch.sign(query_terms)) > 1 else torch.tensor(1)
        self.query_w2v_sum = torch.mean(inputs_query_raw[:int(input_num.item())], 0).T
        raw_word_embedding_list = torch.split(user_item_seq[-13 * 5:, :], [13] * 5, 0)
        step_embedding_list = []
        for raw_word_embed in raw_word_embedding_list:
            item_terms, item_topcate, item_leafcate, time_delta = torch.split(raw_word_embed, [self.user_item_term_length, 1, 1, 1], 0)
            step_embedding = torch.nn.functional.embedding(item_terms, self._word_embed)
            input_num = torch.sum(torch.sign(item_terms)) if torch.sum(torch.sign(item_terms)) > 1 else torch.tensor(1)
            step_avg_embedding = torch.mean(step_embedding[:int(input_num.item())], 0)
            step_embedding_list.append(step_avg_embedding.unsqueeze(0))
        step_embedding_vec = torch.cat(step_embedding_list, 0)
        lstm_cells, lstm_hiddens = self.user_word_lstm(step_embedding_vec, (self.h_1, self.c_1))
        self.user_item_term_lstm_output = torch.tanh(torch.matmul(self.w_l1, lstm_cells[-1].T) + self.b_l1)
        raw_user_item_query_embedding_list = torch.split(user_item_query[:10 * 5, :], [10] * 5, 0)
        step_embedding_list = []
        for raw_user_item_embed in raw_user_item_query_embedding_list:
            item_terms = raw_user_item_embed[:5]
            step_embedding = torch.nn.functional.embedding(item_terms, self._word_embed)
            input_num = torch.sum(torch.sign(item_terms)) if torch.sum(torch.sign(item_terms)) > 1 else torch.tensor(1)
            step_avg_embedding = torch.mean(step_embedding[:int(input_num.item())], 0)
            step_embedding_list.append(step_avg_embedding.unsqueeze(0))
        step_embedding_vec = torch.cat(step_embedding_list, 0)
        lstm_cells, lstm_hiddens = self.user_item_query_lstm(step_embedding_vec, (self.h_2, self.c_2))
        self.user_item_query_term_lstm_output = torch.tanh(torch.matmul(self.w_l2, lstm_cells[-1].T) + self.b_l2)
        raw_user_query_item_embedding_list = torch.split(user_query_item[-10 * 5:, :], [10] * 5, 0)
        step_embedding_list = []
        for raw_user_item_embed in raw_user_query_item_embedding_list:
            item_terms = raw_user_item_embed[:5]
            step_embedding = torch.nn.functional.embedding(item_terms, self._word_embed)
            input_num = torch.sum(torch.sign(item_terms)) if torch.sum(torch.sign(item_terms)) > 1 else torch.tensor(1)
            step_avg_embedding = torch.mean(step_embedding[:int(input_num.item())], 0)
            step_embedding_list.append(step_avg_embedding.unsqueeze(0))
        step_embedding_vec = torch.cat(step_embedding_list, 0)
        lstm_cells, lstm_hiddens = self.user_query_item_lstm(step_embedding_vec, (self.h_3, self.c_3))
        self.user_query_item_term_lstm_output = torch.tanh(torch.matmul(self.w_l3, lstm_cells[-1].T) + self.b_l3)
        raw_query_item_query_embedding_list = torch.split(query_item_query[-10 * 5:, :], [10] * 5, 0)
        step_embedding_list = []
        for raw_user_item_embed in raw_query_item_query_embedding_list:
            query_terms = raw_user_item_embed[:5]
            step_embedding = torch.nn.functional.embedding(query_terms, self._word_embed)
            input_num = torch.sum(torch.sign(query_terms)) if torch.sum(torch.sign(query_terms)) > 1 else torch.tensor(1)
            step_avg_embedding = torch.mean(step_embedding[:int(input_num.item())], 0)
            step_embedding_list.append(step_avg_embedding.unsqueeze(0))
        step_embedding_vec = torch.cat(step_embedding_list, 0)
        convd = self.query_item_query_cnn(torch.transpose(step_embedding_vec, 0, 1).unsqueeze(1))
        convd_active = F.relu(convd)
        pooled = F.max_pool2d(convd_active, (2, 1), stride=2)
        pooled = torch.transpose(pooled, 1, 2)
        pooled = torch.transpose(pooled, 2, 3)
        pool_flat = pooled.reshape(-1, 2 * 1 * 12)
        self.query_item_query_cnn_output = self.query_item_query_relu(self.query_item_query_linear(pool_flat)).T
        raw_query_user_item_embedding_list = torch.split(query_user_item[-10 * 5:, :], [10] * 5, 0)
        step_embedding_list = []
        for raw_user_item_embed in raw_query_user_item_embedding_list:
            item_terms = raw_user_item_embed[:5]
            step_embedding = torch.nn.functional.embedding(item_terms, self._word_embed)
            input_num = torch.sum(torch.sign(item_terms)) if torch.sum(torch.sign(item_terms)) > 1 else torch.tensor(1)
            step_avg_embedding = torch.mean(step_embedding[:int(input_num.item())], 0)
            step_embedding_list.append(step_avg_embedding.unsqueeze(0))
        step_embedding_vec = torch.cat(step_embedding_list, 0)
        convd = self.query_user_item_cnn(torch.transpose(step_embedding_vec, 0, 1).unsqueeze(1))
        convd_active = F.relu(convd)
        pooled = F.max_pool2d(convd_active, (2, 1), stride=2)
        pooled = torch.transpose(pooled, 1, 2)
        pooled = torch.transpose(pooled, 2, 3)
        pool_flat = pooled.reshape(-1, 2 * 1 * 12)
        self.query_user_item_cnn_output = self.query_user_item_relu(self.query_user_item_linear(pool_flat)).T
        raw_user_query_seq_embedding_list = torch.split(user_query_seq[-17 * 5:, :], [17] * 5, 0)
        step_embedding_list = []
        for raw_user_item_embed in raw_user_query_seq_embedding_list[::-1]:
            query_terms, query_topcate, query_leafcate, time_delta = torch.split(raw_user_item_embed, [self.user_query_term_length, 3, 3, 1], 0)
            step_embedding = torch.nn.functional.embedding(query_terms, self._word_embed)
            input_num = torch.sum(torch.sign(query_terms)) if torch.sum(torch.sign(query_terms)) > 1 else torch.tensor(1)
            step_avg_embedding = torch.mean(step_embedding[:int(input_num.item())], 0)
            step_embedding_list.append(step_avg_embedding.unsqueeze(0))
        step_embedding_vec = torch.cat(step_embedding_list, 0)
        lstm_cells, lstm_hiddens = self.user_query_seq_lstm(step_embedding_vec, (self.h_4, self.c_4))
        self.user_query_term_lstm_output = torch.tanh(torch.matmul(self.w_l4, lstm_cells[-1].T) + self.b_l4)
        self.wide_hidden_layer1 = torch.tanh(F.dropout(torch.matmul(self.wide_feat_w, wide_feat.float()) + self.wide_feat_b, self.keep_prob))
        concat_seq = [self.user_item_term_lstm_output, self.user_query_term_lstm_output, self.query_w2v_sum, self.user_item_query_term_lstm_output, self.user_query_item_term_lstm_output, self.query_item_query_cnn_output, self.query_user_item_cnn_output]
        qu_term_concat = F.dropout(torch.cat(concat_seq, 0), self.keep_prob)
        self.qu_term_hidden_layer1 = torch.tanh(F.dropout(torch.matmul(self.concat_query_w, qu_term_concat) + self.concat_query_b, self.keep_prob))
        deep_wide_concat = torch.cat([self.qu_term_hidden_layer1, self.wide_hidden_layer1], 0)
        self.dw_hidden_layer0 = torch.tanh(F.dropout(torch.matmul(self.concat_query_user_wide_w, deep_wide_concat) + self.concat_query_user_wide_b, self.keep_prob))
        self.dw_hidden_layer1 = torch.sigmoid(torch.matmul(self.deep_wide_feat_w, self.dw_hidden_layer0) + self.deep_wide_feat_b)
        self.predict_labels = self.dw_hidden_layer1.squeeze()
        return self.predict_labels


class InGramEntityLayer(nn.Module):

    def __init__(self, dim_in_ent, dim_out_ent, dim_rel, bias=True, num_head=8):
        super(InGramEntityLayer, self).__init__()
        self.dim_out_ent = dim_out_ent
        self.dim_hid_ent = dim_out_ent // num_head
        assert dim_out_ent == self.dim_hid_ent * num_head
        self.num_head = num_head
        self.attn_proj = nn.Linear(2 * dim_in_ent + dim_rel, dim_out_ent, bias=bias)
        self.attn_vec = nn.Parameter(torch.zeros((1, num_head, self.dim_hid_ent)))
        self.aggr_proj = nn.Linear(dim_in_ent + dim_rel, dim_out_ent, bias=bias)
        self.dim_rel = dim_rel
        self.act = nn.LeakyReLU(negative_slope=0.2)
        self.bias = bias
        self.param_init()

    def param_init(self):
        nn.init.xavier_normal_(self.attn_proj.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(self.attn_vec, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(self.aggr_proj.weight, gain=nn.init.calculate_gain('relu'))
        if self.bias:
            nn.init.zeros_(self.attn_proj.bias)
            nn.init.zeros_(self.aggr_proj.bias)

    def forward(self, emb_ent, emb_rel, triplets):
        num_ent = len(emb_ent)
        num_rel = len(emb_rel)
        head_idxs = triplets[..., 0]
        rel_idxs = triplets[..., 1]
        tail_idxs = triplets[..., 2]
        ent_freq = torch.zeros((num_ent,)).index_add(dim=0, index=tail_idxs, source=torch.ones_like(tail_idxs, dtype=torch.float)).unsqueeze(dim=1)
        self_rel = torch.zeros((num_ent, self.dim_rel)).index_add(dim=0, index=tail_idxs, source=emb_rel[rel_idxs]) / ent_freq
        emb_rels = torch.cat([emb_rel[rel_idxs], self_rel], dim=0)
        head_idxs = torch.cat([head_idxs, torch.arange(num_ent)], dim=0)
        tail_idxs = torch.cat([tail_idxs, torch.arange(num_ent)], dim=0)
        concat_mat_att = torch.cat([emb_ent[tail_idxs], emb_ent[head_idxs], emb_rels], dim=-1)
        attn_val_raw = (self.act(self.attn_proj(concat_mat_att).view(-1, self.num_head, self.dim_hid_ent)) * self.attn_vec).sum(dim=-1, keepdim=True)
        scatter_idx = tail_idxs.unsqueeze(dim=-1).repeat(1, self.num_head).unsqueeze(dim=-1)
        attn_val_max = torch.zeros((num_ent, self.num_head, 1)).scatter_reduce(dim=0, index=scatter_idx, src=attn_val_raw, reduce='amax', include_self=False)
        attn_val = torch.exp(attn_val_raw - attn_val_max[tail_idxs])
        attn_sums = torch.zeros((num_ent, self.num_head, 1)).index_add(dim=0, index=tail_idxs, source=attn_val)
        beta = attn_val / (attn_sums[tail_idxs] + 1e-16)
        concat_mat = torch.cat([emb_ent[head_idxs], emb_rels], dim=-1)
        aggr_val = beta * self.aggr_proj(concat_mat).view(-1, self.num_head, self.dim_hid_ent)
        output = torch.zeros((num_ent, self.num_head, self.dim_hid_ent)).index_add(dim=0, index=tail_idxs, source=aggr_val)
        return output.flatten(1, -1)


class InGramRelationLayer(nn.Module):

    def __init__(self, dim_in_rel, dim_out_rel, num_bin, bias=True, num_head=8):
        super(InGramRelationLayer, self).__init__()
        self.dim_out_rel = dim_out_rel
        self.dim_hid_rel = dim_out_rel // num_head
        assert dim_out_rel == self.dim_hid_rel * num_head
        self.attn_proj = nn.Linear(2 * dim_in_rel, dim_out_rel, bias=bias)
        self.attn_bin = nn.Parameter(torch.zeros(num_bin, num_head, 1))
        self.attn_vec = nn.Parameter(torch.zeros(1, num_head, self.dim_hid_rel))
        self.aggr_proj = nn.Linear(dim_in_rel, dim_out_rel, bias=bias)
        self.num_head = num_head
        self.act = nn.LeakyReLU(negative_slope=0.2)
        self.num_bin = num_bin
        self.bias = bias
        self.param_init()

    def param_init(self):
        nn.init.xavier_normal_(self.attn_proj.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(self.attn_vec, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(self.aggr_proj.weight, gain=nn.init.calculate_gain('relu'))
        if self.bias:
            nn.init.zeros_(self.attn_proj.bias)
            nn.init.zeros_(self.aggr_proj.bias)

    def forward(self, emb_rel, relation_triplets):
        num_rel = len(emb_rel)
        head_idxs = relation_triplets[..., 0]
        tail_idxs = relation_triplets[..., 1]
        concat_mat = torch.cat([emb_rel[head_idxs], emb_rel[tail_idxs]], dim=-1)
        attn_val_raw = (self.act(self.attn_proj(concat_mat).view(-1, self.num_head, self.dim_hid_rel)) * self.attn_vec).sum(dim=-1, keepdim=True) + self.attn_bin[relation_triplets[..., 2]]
        scatter_idx = head_idxs.unsqueeze(dim=-1).repeat(1, self.num_head).unsqueeze(dim=-1)
        attn_val_max = torch.zeros((num_rel, self.num_head, 1)).scatter_reduce(dim=0, index=scatter_idx, src=attn_val_raw, reduce='amax', include_self=False)
        attn_val = torch.exp(attn_val_raw - attn_val_max[head_idxs])
        attn_sums = torch.zeros((num_rel, self.num_head, 1)).index_add(dim=0, index=head_idxs, source=attn_val)
        beta = attn_val / (attn_sums[head_idxs] + 1e-16)
        output = torch.zeros((num_rel, self.num_head, self.dim_hid_rel)).index_add(dim=0, index=head_idxs, source=beta * self.aggr_proj(emb_rel[tail_idxs]).view(-1, self.num_head, self.dim_hid_rel))
        return output.flatten(1, -1)


class KGAT_Aggregator(nn.Module):

    def __init__(self, in_dim, out_dim, dropout, aggregator_type):
        super(KGAT_Aggregator, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dropout = dropout
        self.aggregator_type = aggregator_type
        self.message_dropout = nn.Dropout(dropout)
        if aggregator_type == 'gcn':
            self.W = nn.Linear(self.in_dim, self.out_dim)
        elif aggregator_type == 'graphsage':
            self.W = nn.Linear(self.in_dim * 2, self.out_dim)
        elif aggregator_type == 'bi-interaction':
            self.W1 = nn.Linear(self.in_dim, self.out_dim)
            self.W2 = nn.Linear(self.in_dim, self.out_dim)
        else:
            raise NotImplementedError
        self.activation = nn.LeakyReLU()

    def forward(self, mode, g, entity_embed):
        g = g.local_var()
        g.ndata['node'] = entity_embed
        if mode == 'predict':
            g.update_all(dgl.function.u_mul_e('node', 'att', 'side'), lambda nodes: {'N_h': torch.sum(nodes.mailbox['side'], 1)})
        else:
            g.update_all(dgl.function.u_mul_e('node', 'att', 'side'), dgl.function.sum('side', 'N_h'))
        if self.aggregator_type == 'gcn':
            out = self.activation(self.W(g.ndata['node'] + g.ndata['N_h']))
        elif self.aggregator_type == 'graphsage':
            out = self.activation(self.W(torch.cat([g.ndata['node'], g.ndata['N_h']], dim=1)))
        elif self.aggregator_type == 'bi-interaction':
            out1 = self.activation(self.W1(g.ndata['node'] + g.ndata['N_h']))
            out2 = self.activation(self.W2(g.ndata['node'] * g.ndata['N_h']))
            out = out1 + out2
        else:
            raise NotImplementedError
        out = self.message_dropout(out)
        return out


class KGAT(BaseModel):
    """
    This model KGAT was introduced in `KGAT <https://arxiv.org/pdf/1905.07854v2.pdf>`__.

    It involves some careful design.

    Embedding Layer:

    Knowledge graph embedding is an effective way to parameterize entities and relations as vector representations.
    KGAT employ TransR, a widely used method, on CKG.

    Attentive Embedding Propagation Layers:

    A single attentive embedding propagation layer consists of three components: information propagation, knowledge-aware attention, and information aggregation.

    1. Information Propagation:

    Considering an entity :math:`h`, we use :math:`\\mathcal{N}_{h} = {(h,r,t)|(h,r,t)\\in\\mathcal{G}}` to denote the set of triplets where :math:`h` is the head entity, termed ego-network.
    To characterize the first-order connectivity structure of entity :math:`h`, we compute the linear combination of :math:`h`’s ego-network:

    :math:`e_{\\mathcal{N}_{h}}=\\sum_{(h,r,t)\\in\\mathcal{N}_{h}}\\pi(h,r,t)e_{t}`

    where :math:`\\pi(h,r,t)` controls the decay factor on each propagation on edge :math:`(h,r,t)`, indicating how much information being propagated from :math:`t` to :math:`h` conditioned to relation :math:`r`.

    2. Knowledge-aware Attention:

    GAT implement :math:`\\pi(h,r,t)` via relational attention mechanism, which is formulated as follows:

    :math:`\\pi(h,r,t)=(\\mathcal{N}_{r}\\mathcal{e}_{t})^{\\mathsf{T}}tanh((\\mathcal{W}_{r}\\mathcal{e}_{h}+e_{r}))`

    This makes the attention score dependent on the distance between :math:`e_h` and :math:`e_t` in the relation :math:`r`’s space
    Hereafter, GAT normalize the coefficients across all triplets connected with :math:`h` by adopting the softmax function:

    :math:`\\pi(h,r,t)=\\frac{exp(\\pi(h,r,t))}{\\sum_{(h,r',t')\\in\\mathcal{N}_{h}}exp(\\pi(h,r',t'))}`

    As a result, the final attention score is capable of suggesting which neighbor nodes should be given more attention to capture collaborative signals.

    3. Information Aggregation:
    The final phase is to aggregate the entity representation :math:`e_h` and its ego-network representations :math:`e_{\\mathcal{N}_h}` as the new representation of entity :math:`h`

    GCN Aggregator : :math:`\\mathcal{f}_{GCN}=LeakyReLU(\\mathcal{W}(e_h+e_{\\mathcal{N}_h}))`

    GraphSage Aggregator : :math:`\\mathcal{f}_{GraphSage}=LeakyReLU(\\mathcal{W}(e_h||e_{\\mathcal{N}_h}))`

    Bi-Interaction Aggregator : :math:`\\mathcal{f}_{Bi-Interaction}=LeakyReLU(\\mathcal{W}(e_h+e_{\\mathcal{N}_h})+LeakyReLU(\\mathcal{W}(e_h\\odote_{\\mathcal{N}_h})`

    High-order Propagation:
    We can further stack more propagation layers to explore the high-order connectivity information, gathering the information propagated from the higher-hop neighbors.

    :math:`e_{h}^{(l)}=\\mathcal{f}(e_{h}^{(l-1)}_{\\mathcal{N}_h})`

    Model Prediction：

    After performing :math:`L` layers, we obtain multiple representations for user node :math:`u`, namely :math:`{e_{u}^{(1)},...,{e_{u}^{(L)}}`; Analogous to item node i, :math:`{e_{i}^{(1)},...,{e_{i}^{(L)}}` are obtained.
    GAT hence adopt the layer-aggregation mechanism to concatenate the representations at each step into a single vector, as follows:

    :math:`e^*_u=e_u^{(0)}||...||e_u^{(L)},e^*_i=e_i^{(0)}||...||e_i^{(L)}`

    Finally, we conduct inner product of user and item representations, so as to predict their matching score:

    :math:`\\check{\\mathcal{y}}(u,i)=e^*_u`\\mathsf{T}e^*_i`

    Parameters
    ----------
    entity_dim ：User / entity Embedding size
    relation_dim ： Relation Embedding size
    aggregation_type ： Specify the type of the aggregation layer from {gcn, graphsage, bi-interaction}
    conv_dim_list ： Output sizes of every aggregation layer
    mess_dropout ： Dropout probability w.r.t. message dropout for each deep layer. 0: no dropout
    """

    @classmethod
    def build_model_from_args(cls, args, hg):
        return cls(args=args)

    def __init__(self, args):
        super(KGAT, self).__init__()
        if args.aggregation_type not in ['bi-interaction', 'gcn', 'graphsage']:
            raise KeyError('Aggregator type {} not supported.'.format(args.aggregation_type))
        self.use_pretrain = args.use_pretrain
        self.entity_dim = args.entity_dim
        self.relation_dim = args.relation_dim
        self.aggregation_type = args.aggregation_type
        self.conv_dim_list = [args.entity_dim] + eval(args.conv_dim_list)
        self.mess_dropout = eval(args.mess_dropout)
        self.n_layers = len(eval(args.conv_dim_list))
        self.kg_l2loss_lambda = args.kg_l2loss_lambda
        self.cf_l2loss_lambda = args.cf_l2loss_lambda
        self.aggregator_layers = nn.ModuleList()
        for k in range(self.n_layers):
            self.aggregator_layers.append(KGAT_Aggregator(self.conv_dim_list[k], self.conv_dim_list[k + 1], self.mess_dropout[k], self.aggregation_type))

    def set_parameters(self, n_users, n_entities, n_relations, user_pre_embed=None, item_pre_embed=None):
        self.n_users = n_users
        self.n_entities = n_entities
        self.n_relations = n_relations
        self.relation_embed = nn.Embedding(self.n_relations, self.relation_dim)
        self.entity_user_embed = nn.Embedding(self.n_entities + self.n_users, self.entity_dim)
        if self.use_pretrain == 1 and user_pre_embed is not None and item_pre_embed is not None:
            other_entity_embed = nn.Parameter(torch.Tensor(self.n_entities - item_pre_embed.shape[0], self.entity_dim))
            nn.init.xavier_uniform_(other_entity_embed, gain=nn.init.calculate_gain('relu'))
            entity_user_embed = torch.cat([item_pre_embed, other_entity_embed, user_pre_embed], dim=0)
            self.entity_user_embed.weight = nn.Parameter(entity_user_embed)
        self.W_R = nn.Parameter(torch.Tensor(self.n_relations, self.entity_dim, self.relation_dim))
        nn.init.xavier_uniform_(self.W_R, gain=nn.init.calculate_gain('relu'))
    """
    # DGL: dgl-cu101(0.4.3)
    # We will get different results when using the function `fn.sum`, and the randomness is due to `atomicAdd`.
    # Use custom function to ensure deterministic behavior.
    """

    def edge_softmax_fix(self, graph, score):

        def reduce_sum(nodes):
            accum = torch.sum(nodes.mailbox['temp'], 1)
            return {'out_sum': accum}
        graph = graph.local_var()
        graph.edata['out'] = score
        graph.edata['out'] = torch.exp(graph.edata['out'])
        graph.update_all(fn.copy_e('out', 'temp'), reduce_sum)
        graph.apply_edges(fn.e_div_v('out', 'out_sum', 'out'))
        out = graph.edata['out']
        return out

    def att_score(self, edges):
        r_mul_t = torch.matmul(self.entity_user_embed(edges.src['id']), self.W_r)
        r_mul_h = torch.matmul(self.entity_user_embed(edges.dst['id']), self.W_r)
        r_embed = self.relation_embed(edges.data['type'])
        att = torch.bmm(r_mul_t.unsqueeze(1), torch.tanh(r_mul_h + r_embed).unsqueeze(2)).squeeze(-1)
        return {'att': att}

    def compute_attention(self, g):
        g = g.local_var()
        for i in range(self.n_relations):
            edge_idxs = g.filter_edges(lambda edge: edge.data['type'] == i)
            self.W_r = self.W_R[i]
            g.apply_edges(self.att_score, edge_idxs)
        g.edata['att'] = self.edge_softmax_fix(g, g.edata.pop('att'))
        return g.edata.pop('att')

    def _L2_loss_mean(self, x):
        return torch.mean(torch.sum(torch.pow(x, 2), dim=1, keepdim=False) / 2.0)

    def calc_kg_loss(self, h, r, pos_t, neg_t):
        """
        h:      (kg_batch_size)
        r:      (kg_batch_size)
        pos_t:  (kg_batch_size)
        neg_t:  (kg_batch_size)
        """
        r_embed = self.relation_embed(r)
        W_r = self.W_R[r]
        h_embed = self.entity_user_embed(h)
        pos_t_embed = self.entity_user_embed(pos_t)
        neg_t_embed = self.entity_user_embed(neg_t)
        r_mul_h = torch.bmm(h_embed.unsqueeze(1), W_r).squeeze(1)
        r_mul_pos_t = torch.bmm(pos_t_embed.unsqueeze(1), W_r).squeeze(1)
        r_mul_neg_t = torch.bmm(neg_t_embed.unsqueeze(1), W_r).squeeze(1)
        pos_score = torch.sum(torch.pow(r_mul_h + r_embed - r_mul_pos_t, 2), dim=1)
        neg_score = torch.sum(torch.pow(r_mul_h + r_embed - r_mul_neg_t, 2), dim=1)
        kg_loss = -1.0 * F.logsigmoid(neg_score - pos_score)
        kg_loss = torch.mean(kg_loss)
        l2_loss = self._L2_loss_mean(r_mul_h) + self._L2_loss_mean(r_embed) + self._L2_loss_mean(r_mul_pos_t) + self._L2_loss_mean(r_mul_neg_t)
        loss = kg_loss + self.kg_l2loss_lambda * l2_loss
        return loss

    def cf_embedding(self, mode, g):
        g = g.local_var()
        ego_embed = self.entity_user_embed(g.ndata['id'])
        all_embed = [ego_embed]
        for i, layer in enumerate(self.aggregator_layers):
            ego_embed = layer(mode, g, ego_embed)
            norm_embed = F.normalize(ego_embed, p=2, dim=1)
            all_embed.append(norm_embed)
        all_embed = torch.cat(all_embed, dim=1)
        return all_embed

    def cf_score(self, mode, g, user_ids, item_ids):
        """
        user_ids:   number of users to evaluate   (n_eval_users)
        item_ids:   number of items to evaluate   (n_eval_items)
        """
        all_embed = self.cf_embedding(mode, g)
        user_embed = all_embed[user_ids]
        item_embed = all_embed[item_ids]
        cf_score = torch.matmul(user_embed, item_embed.transpose(0, 1))
        return cf_score

    def calc_cf_loss(self, mode, g, user_ids, item_pos_ids, item_neg_ids):
        """
        user_ids:       (cf_batch_size)
        item_pos_ids:   (cf_batch_size)
        item_neg_ids:   (cf_batch_size)
        """
        all_embed = self.cf_embedding(mode, g)
        user_embed = all_embed[user_ids]
        item_pos_embed = all_embed[item_pos_ids]
        item_neg_embed = all_embed[item_neg_ids]
        pos_score = torch.sum(user_embed * item_pos_embed, dim=1)
        neg_score = torch.sum(user_embed * item_neg_embed, dim=1)
        cf_loss = -1.0 * F.logsigmoid(pos_score - neg_score)
        cf_loss = torch.mean(cf_loss)
        l2_loss = self._L2_loss_mean(user_embed) + self._L2_loss_mean(item_pos_embed) + self._L2_loss_mean(item_neg_embed)
        loss = cf_loss + self.cf_l2loss_lambda * l2_loss
        return loss

    def forward(self, mode, *input):
        if mode == 'calc_att':
            return self.compute_attention(*input)
        if mode == 'calc_cf_loss':
            return self.calc_cf_loss(mode, *input)
        if mode == 'calc_kg_loss':
            return self.calc_kg_loss(*input)
        if mode == 'predict':
            return self.cf_score(mode, *input)


class KGCN_Aggregate(nn.Module):

    def __init__(self, args):
        super(KGCN_Aggregate, self).__init__()
        self.args = args
        self.in_dim = args.in_dim
        self.out_dim = args.out_dim
        if self.args.aggregate == 'CONCAT':
            self.agg = nn.Linear(self.in_dim * 2, self.out_dim)
        else:
            self.agg = nn.Linear(self.in_dim, self.out_dim)

    def aggregate(self):
        self.sub_g.update_all(fn.u_mul_e('embedding', 'weight', 'm'), fn.sum('m', 'ft'))
        self.userList = []
        self.labelList = []
        embeddingList = []
        for i in range(len(self.data)):
            weightIndex = np.where(self.itemlist == int(self.sub_g.dstdata['_ID'][i]))
            if self.args.aggregate == 'SUM':
                embeddingList.append(self.sub_g.dstdata['embedding'][i] + self.sub_g.dstdata['ft'][i][weightIndex])
            elif self.args.aggregate == 'CONCAT':
                embeddingList.append(th.cat([self.sub_g.dstdata['embedding'][i], self.sub_g.dstdata['ft'][i][weightIndex].squeeze(0)], dim=-1))
            elif self.args.aggregate == 'NEIGHBOR':
                embeddingList.append(self.sub_g.dstdata['embedding'][i])
            self.userList.append(int(self.user_indices[weightIndex]))
            self.labelList.append(int(self.labels[weightIndex]))
        self.sub_g.dstdata['embedding'] = th.stack(embeddingList).squeeze(1)
        output = F.dropout(self.sub_g.dstdata['embedding'], p=0)
        if self.layer + 1 == len(self.blocks):
            self.item_embeddings = th.tanh(self.agg(output))
        else:
            self.item_embeddings = th.relu(self.agg(output))

    def forward(self, blocks, inputdata):
        """
        Aggregate the entity representation and its neighborhood representation

        Parameters
        ----------
        blocks : list
            Blocks saves the information of neighbor nodes in each layer
        inputdata : numpy.ndarray
            Inputdata contains the relationship between the user and the entity

        Returns
        -------
        item_embeddings : torch.Tensor
            items' embeddings after aggregated
        userList : list
            Users corresponding to items
        labelList : list
            Labels corresponding to items
        """
        self.data = inputdata
        self.blocks = blocks
        self.user_indices = self.data[:, 0]
        self.itemlist = self.data[:, 1]
        self.labels = self.data[:, 2]
        for self.layer in range(len(blocks)):
            self.sub_g = blocks[self.layer]
            self.aggregate()
        return self.item_embeddings, self.userList, self.labelList


class KGCN(BaseModel):
    """
    This module KGCN was introduced in `KGCN <https://dl.acm.org/doi/10.1145/3308558.3313417>`__.

    It included two parts:

    Aggregate the entity representation and its neighborhood representation into the entity's embedding.
    The message function is defined as follow:

    :math:`\\mathrm{v}_{\\mathcal{N}(v)}^{u}=\\sum_{e \\in \\mathcal{N}(v)} \\tilde{\\pi}_{r_{v, e}}^{u} \\mathrm{e}`

    where :math:`\\mathrm{e}` is the representation of entity,
    :math:`\\tilde{\\pi}_{r_{v, e}}^{u}` is the scalar weight on the edge from entity to entity,
    the result :math:`\\mathrm{v}_{\\mathcal{N}(v)}^{u}` saves message which is passed from neighbor nodes

    There are three types of aggregators.
    Sum aggregator takes the summation of two representation vectors,
    Concat aggregator concatenates the two representation vectors and
    Neighbor aggregator directly takes the neighborhood representation of entity as the output representation

    :math:`a g g_{s u m}=\\sigma\\left(\\mathbf{W} \\cdot\\left(\\mathrm{v}+\\mathrm{v}_{\\mathcal{S}(v)}^{u}\\right)+\\mathbf{b}\\right)`

    :math:`agg $_{\\text {concat }}=\\sigma\\left(\\mathbf{W} \\cdot \\text{concat}\\left(\\mathrm{v}, \\mathrm{v}_{\\mathcal{S}(v)}^{u}\\right)+\\mathbf{b}\\right)$`

    :math:`\\text { agg }_{\\text {neighbor }}=\\sigma\\left(\\mathrm{W} \\cdot \\mathrm{v}_{\\mathcal{S}(v)}^{u}+\\mathrm{b}\\right)`

    In the above equations, :math:`\\sigma` is the nonlinear function and
    :math:`\\mathrm{W}` and :math:`\\mathrm{b}` are transformation weight and bias.
    the representation of an item is bound up with its neighbors by aggregation

    Obtain scores using final entity representation and user representation
    The final entity representation is denoted as :math:`\\mathrm{v}^{u}`,
    :math:`\\mathrm{v}^{u}` do dot product with user representation :math:`\\mathrm{u}`
    can obtain the probability. The math formula for the above function is:

    :math:`$\\hat{y}_{u v}=f\\left(\\mathbf{u}, \\mathrm{v}^{u}\\right)$`

    Parameters
    ----------
    g : DGLGraph
        A knowledge Graph preserves relationships between entities
    args : Config
        Model's config
    """

    @classmethod
    def build_model_from_args(cls, args, g):
        return cls(g, args)

    def __init__(self, g, args):
        super(KGCN, self).__init__()
        self.g = g
        self.args = args
        self.in_dim = args.in_dim
        self.out_dim = args.out_dim
        self.entity_emb_matrix = nn.Parameter(th.FloatTensor(self.g.num_nodes(), self.in_dim))
        self.relation_emb_matrix = nn.Parameter(th.FloatTensor(args.n_relation, self.in_dim))
        self.user_emb_matrix = nn.Parameter(th.FloatTensor(args.n_user, self.in_dim))
        self.Aggregate = KGCN_Aggregate(args)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.entity_emb_matrix, -1, 1)
        nn.init.uniform_(self.relation_emb_matrix, -1, 1)
        nn.init.uniform_(self.user_emb_matrix, -1, 1)

    def get_score(self):
        """
        Obtain scores using final entity representation and user representation
        
        Returns
        -------

        """
        self.user_embeddings = self.user_emb_matrix[np.array(self.userList)]
        self.scores = th.sum(self.user_embeddings * self.item_embeddings, dim=1)
        self.scores_normalized = th.sigmoid(self.scores)

    def get_embeddings(self):
        return self.user_emb_matrix, self.entity_emb_matrix, self.relation_emb_matrix

    def forward(self, blocks, inputdata):
        """
        Predict the probability between user and entity

        Parameters
        ----------
        blocks : list
            Blocks saves the information of neighbor nodes in each layer
        inputdata : numpy.ndarray
            Inputdata contains the relationship between the user and the entity

        Returns
        -------
        labels : torch.Tensor
            the label between users and entities
        scores : torch.Tensor
            Probability of users clicking on entitys
        """
        self.data = inputdata
        self.blocks = blocks
        self.user_indices = self.data[:, 0]
        self.itemlist = self.data[:, 1]
        self.labels = self.data[:, 2]
        self.item_embeddings, self.userList, self.labelList = self.Aggregate(blocks, inputdata)
        self.get_score()
        self.labels = th.tensor(self.labelList)
        return self.labels, self.scores


class TransE(BaseModel):

    @classmethod
    def build_model_from_args(cls, args, hg):
        return cls(args, hg)

    def __init__(self, args, hg):
        super(TransE, self).__init__()
        self.device = args.device
        self.ent_num = hg.num_nodes()
        self.rel_num = len(hg.canonical_etypes)
        self.dim = args.hidden_dim
        self.margin = args.margin
        self.dis_norm = args.dis_norm
        self.n_emb = nn.Embedding(self.ent_num, self.dim)
        self.r_emb = nn.Embedding(self.rel_num, self.dim)
        nn.init.xavier_uniform_(self.n_emb.weight.data)
        nn.init.xavier_uniform_(self.r_emb.weight.data)

    def forward(self, h, r, t):
        if self.training:
            self.r_emb.weight.data = F.normalize(self.r_emb.weight.data, p=2.0, dim=-1)
            self.n_emb.weight.data = F.normalize(self.n_emb.weight.data, p=2.0, dim=-1)
        h_emb = self.n_emb(h)
        r_emb = self.r_emb(r)
        t_emb = self.n_emb(t)
        h_emb = F.normalize(h_emb, 2.0, -1)
        r_emb = F.normalize(r_emb, 2.0, -1)
        t_emb = F.normalize(t_emb, 2.0, -1)
        score = th.norm(h_emb + r_emb - t_emb, self.dis_norm, dim=-1)
        return score


class LTE(BaseModel):

    @classmethod
    def build_model_from_args(cls, config):
        return cls(config)

    def __init__(self, config):
        super().__init__()
        self.model = TransE(config)

    def forward(self, *args):
        return self.model(*args)

    def extra_loss(self):
        pass


class LTEModel(nn.Module):

    def __init__(self, params=None):
        super(LTEModel, self).__init__()
        self.bceloss = torch.nn.BCELoss()
        self.p = params
        num_ents = self.p.num_ents
        num_rels = self.p.num_rels
        self.init_embed = get_param((num_ents, self.p.init_dim))
        self.device = 'cuda'
        self.init_rel = get_param((num_rels * 2, self.p.init_dim))
        self.bias = nn.Parameter(torch.zeros(num_ents))
        self.h_ops_dict = nn.ModuleDict({'p': nn.Linear(self.p.init_dim, self.p.gcn_dim, bias=False), 'b': nn.BatchNorm1d(self.p.gcn_dim), 'd': nn.Dropout(self.p.hid_drop), 'a': nn.Tanh()})
        self.t_ops_dict = nn.ModuleDict({'p': nn.Linear(self.p.init_dim, self.p.gcn_dim, bias=False), 'b': nn.BatchNorm1d(self.p.gcn_dim), 'd': nn.Dropout(self.p.hid_drop), 'a': nn.Tanh()})
        self.r_ops_dict = nn.ModuleDict({'p': nn.Linear(self.p.init_dim, self.p.gcn_dim, bias=False), 'b': nn.BatchNorm1d(self.p.gcn_dim), 'd': nn.Dropout(self.p.hid_drop), 'a': nn.Tanh()})
        self.x_ops = self.p.x_ops
        self.r_ops = self.p.r_ops
        self.diff_ht = False

    def calc_loss(self, pred, label):
        return self.loss(pred, label)

    def loss(self, pred, true_label):
        return self.bceloss(pred, true_label)

    def exop(self, x, r, x_ops=None, r_ops=None, diff_ht=False):
        x_head = x_tail = x
        if len(x_ops) > 0:
            for x_op in x_ops.split('.'):
                if diff_ht:
                    x_head = self.h_ops_dict[x_op](x_head)
                    x_tail = self.t_ops_dict[x_op](x_tail)
                else:
                    x_head = x_tail = self.h_ops_dict[x_op](x_head)
        if len(r_ops) > 0:
            for r_op in r_ops.split('.'):
                r = self.r_ops_dict[r_op](r)
        return x_head, x_tail, r


class DistMult(LTEModel):

    def __init__(self, num_ents, num_rels, params=None):
        super(self.__class__, self).__init__(num_ents, num_rels, params)

    def forward(self, g, sub, rel):
        x = self.init_embed
        r = self.init_rel
        x_h, x_t, r = self.exop(x, r, self.x_ops, self.r_ops)
        sub_emb = torch.index_select(x_h, 0, sub)
        rel_emb = torch.index_select(r, 0, rel)
        all_ent = x_t
        obj_emb = sub_emb * rel_emb
        x = torch.mm(obj_emb, all_ent.transpose(1, 0))
        x += self.bias.expand_as(x)
        score = torch.sigmoid(x)
        return score


class ConvE(LTEModel):

    def __init__(self, num_ents, num_rels, params=None):
        super(self.__class__, self).__init__(num_ents, num_rels, params)
        self.bn0 = torch.nn.BatchNorm2d(1)
        self.bn1 = torch.nn.BatchNorm2d(self.p.num_filt)
        self.bn2 = torch.nn.BatchNorm1d(self.p.embed_dim)
        self.hidden_drop = torch.nn.Dropout(self.p.hid_drop)
        self.hidden_drop2 = torch.nn.Dropout(self.p.conve_hid_drop)
        self.feature_drop = torch.nn.Dropout(self.p.feat_drop)
        self.m_conv1 = torch.nn.Conv2d(1, out_channels=self.p.num_filt, kernel_size=(self.p.ker_sz, self.p.ker_sz), stride=1, padding=0, bias=self.p.bias)
        flat_sz_h = int(2 * self.p.k_w) - self.p.ker_sz + 1
        flat_sz_w = self.p.k_h - self.p.ker_sz + 1
        self.flat_sz = flat_sz_h * flat_sz_w * self.p.num_filt
        self.fc = torch.nn.Linear(self.flat_sz, self.p.embed_dim)

    def concat(self, e1_embed, rel_embed):
        e1_embed = e1_embed.view(-1, 1, self.p.embed_dim)
        rel_embed = rel_embed.view(-1, 1, self.p.embed_dim)
        stack_inp = torch.cat([e1_embed, rel_embed], 1)
        stack_inp = torch.transpose(stack_inp, 2, 1).reshape((-1, 1, 2 * self.p.k_w, self.p.k_h))
        return stack_inp

    def forward(self, g, sub, rel):
        x = self.init_embed
        r = self.init_rel
        x_h, x_t, r = self.exop(x, r, self.x_ops, self.r_ops)
        sub_emb = torch.index_select(x_h, 0, sub)
        rel_emb = torch.index_select(r, 0, rel)
        all_ent = x_t
        stk_inp = self.concat(sub_emb, rel_emb)
        x = self.bn0(stk_inp)
        x = self.m_conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_drop(x)
        x = x.view(-1, self.flat_sz)
        x = self.fc(x)
        x = self.hidden_drop2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = torch.mm(x, all_ent.transpose(1, 0))
        x += self.bias.expand_as(x)
        score = torch.sigmoid(x)
        return score


class GCNs(nn.Module):

    def __init__(self, args, num_ent, num_rel, num_base, init_dim, gcn_dim, embed_dim, n_layer, edge_type, edge_norm, conv_bias=True, gcn_drop=0.0, opn='mult', wni=False, wsi=False, encoder='compgcn', use_bn=True, ltr=True):
        super(GCNs, self).__init__()
        num_ent = args.num_ent
        num_rel = args.num_rel
        num_base = args.num_base
        init_dim = args.init_dim
        gcn_dim = args.gcn_dim
        embed_dim = args.embed_dim
        n_layer = args.n_layer
        edge_type = args.edge_type
        edge_norm = args.edge_norm
        conv_bias = True
        if args.conv_bias is not None:
            conv_bias = args.conv_bias
        gcn_drop = 0.0
        if args.gcn_drop is not None:
            gcn_drop = args.gcn_drop
        opn = 'mult'
        if args.opn is not None:
            opn = args.opn
        wni = False
        if args.wni is not None:
            wni = args.wni
        wsi = False
        if args.wsi is not None:
            wsi = args.wsi
        encoder = 'compgcn'
        if args.encoder is not None:
            encoder = args.encoder
        use_bn = True
        if args.use_bn is not None:
            use_bn = args.use_bn
        ltr = True
        if args.ltr is not None:
            ltr = args.ltr
        self.act = torch.tanh
        self.loss = nn.BCELoss()
        self.num_ent, self.num_rel, self.num_base = num_ent, num_rel, num_base
        self.init_dim, self.gcn_dim, self.embed_dim = init_dim, gcn_dim, embed_dim
        self.conv_bias = conv_bias
        self.gcn_drop = gcn_drop
        self.opn = opn
        self.edge_type = edge_type
        self.edge_norm = edge_norm
        self.n_layer = n_layer
        self.wni = wni
        self.encoder = encoder
        self.init_embed = self.get_param([self.num_ent, self.init_dim])
        self.init_rel = self.get_param([self.num_rel * 2, self.init_dim])
        if encoder == 'compgcn':
            if n_layer < 3:
                self.conv1 = CompGCNCov(self.init_dim, self.gcn_dim, self.act, conv_bias, gcn_drop, opn, num_base=-1, num_rel=self.num_rel, wni=wni, wsi=wsi, use_bn=use_bn, ltr=ltr)
                self.conv2 = CompGCNCov(self.gcn_dim, self.embed_dim, self.act, conv_bias, gcn_drop, opn, wni=wni, wsi=wsi, use_bn=use_bn, ltr=ltr) if n_layer == 2 else None
            else:
                self.conv1 = CompGCNCov(self.init_dim, self.gcn_dim, self.act, conv_bias, gcn_drop, opn, num_base=-1, num_rel=self.num_rel, wni=wni, wsi=wsi, use_bn=use_bn, ltr=ltr)
                self.conv2 = CompGCNCov(self.gcn_dim, self.gcn_dim, self.act, conv_bias, gcn_drop, opn, num_base=-1, num_rel=self.num_rel, wni=wni, wsi=wsi, use_bn=use_bn, ltr=ltr)
                self.conv3 = CompGCNCov(self.gcn_dim, self.embed_dim, self.act, conv_bias, gcn_drop, opn, wni=wni, wsi=wsi, use_bn=use_bn, ltr=ltr)
        elif encoder == 'rgcn':
            self.conv1 = RelGraphConv(self.init_dim, self.gcn_dim, self.num_rel * 2, 'bdd', num_bases=self.num_base, activation=self.act, self_loop=not wsi, dropout=gcn_drop, wni=wni)
            self.conv2 = RelGraphConv(self.gcn_dim, self.embed_dim, self.num_rel * 2, 'bdd', num_bases=self.num_base, activation=self.act, self_loop=not wsi, dropout=gcn_drop, wni=wni) if n_layer == 2 else None
        self.bias = nn.Parameter(torch.zeros(self.num_ent))

    def get_param(self, shape):
        param = nn.Parameter(torch.Tensor(*shape))
        nn.init.xavier_normal_(param, gain=nn.init.calculate_gain('relu'))
        return param

    def calc_loss(self, pred, label):
        return self.loss(pred, label)

    def forward_base(self, g, subj, rel, drop1, drop2):
        """
        :param g: graph
        :param sub: subjects in a batch [batch]
        :param rel: relations in a batch [batch]
        :param drop1: dropout rate in first layer
        :param drop2: dropout rate in second layer
        :return: sub_emb: [batch, D]
                 rel_emb: [num_rel*2, D]
                 x: [num_ent, D]
        """
        x, r = self.init_embed, self.init_rel
        if self.n_layer > 0:
            if self.encoder == 'compgcn':
                if self.n_layer < 3:
                    x, r = self.conv1(g, x, r, self.edge_type, self.edge_norm)
                    x = drop1(x)
                    x, r = self.conv2(g, x, r, self.edge_type, self.edge_norm) if self.n_layer == 2 else (x, r)
                    x = drop2(x) if self.n_layer == 2 else x
                else:
                    x, r = self.conv1(g, x, r, self.edge_type, self.edge_norm)
                    x = drop1(x)
                    x, r = self.conv2(g, x, r, self.edge_type, self.edge_norm)
                    x = drop1(x)
                    x, r = self.conv3(g, x, r, self.edge_type, self.edge_norm)
                    x = drop2(x)
            elif self.encoder == 'rgcn':
                x = self.conv1(g, x, self.edge_type, self.edge_norm.unsqueeze(-1))
                x = drop1(x)
                x = self.conv2(g, x, self.edge_type, self.edge_norm.unsqueeze(-1)) if self.n_layer == 2 else x
                x = drop2(x) if self.n_layer == 2 else x
        sub_emb = torch.index_select(x, 0, subj)
        rel_emb = torch.index_select(r, 0, rel)
        return sub_emb, rel_emb, x


class GCN_TransE(GCNs):

    def __init__(self, args):
        super(GCN_TransE, self).__init__(args)
        self.drop = nn.Dropout(args.hid_drop)
        self.gamma = args.gamma

    def forward(self, g, subj, rel):
        """
        :param g: dgl graph
        :param sub: subject in batch [batch_size]
        :param rel: relation in batch [batch_size]
        :return: score: [batch_size, ent_num], the prob in link-prediction
        """
        sub_emb, rel_emb, all_ent = self.forward_base(g, subj, rel, self.drop, self.drop)
        obj_emb = sub_emb + rel_emb
        x = self.gamma - torch.norm(obj_emb.unsqueeze(1) - all_ent, p=1, dim=2)
        score = torch.sigmoid(x)
        return score


class LTE_Transe(BaseModel):

    @classmethod
    def build_model_from_args(cls, config):
        return cls(config)

    def __init__(self, config):
        super().__init__()
        self.model = GCN_TransE(config)

    def forward(self, *args):
        return self.model(*args)

    def extra_loss(self):
        pass


class MAGNN_attn_intra(nn.Module):

    def __init__(self, in_feats, out_feats, num_heads, feat_drop=0.5, attn_drop=0.5, negative_slope=0.01, activation=F.elu):
        super(MAGNN_attn_intra, self).__init__()
        self._num_heads = num_heads
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self.attn_r = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats)))
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.reset_parameters()
        self.activation = activation

    def reset_parameters(self):
        nn.init.xavier_normal_(self.attn_r, gain=1.414)

    def forward(self, feat, metapath, metapath_idx):
        _metapath = metapath.split('-')
        device = feat[0].device
        h_meta = self.feat_drop(feat[0]).view(-1, self._num_heads, self._out_feats)
        er = (h_meta * self.attn_r).sum(dim=-1).unsqueeze(-1)
        graph_data = {('meta_inst', 'meta2{}'.format(_metapath[0]), _metapath[0]): (th.arange(0, metapath_idx.shape[0]), th.tensor(metapath_idx[:, 0]))}
        num_nodes_dict = {'meta_inst': metapath_idx.shape[0], _metapath[0]: feat[1].shape[0]}
        g_meta = dgl.heterograph(graph_data, num_nodes_dict=num_nodes_dict)
        g_meta.nodes['meta_inst'].data.update({'feat_src': h_meta, 'er': er})
        g_meta.apply_edges(func=fn.copy_u('er', 'e'), etype='meta2{}'.format(_metapath[0]))
        e = self.leaky_relu(g_meta.edata.pop('e'))
        g_meta.edata['a'] = self.attn_drop(edge_softmax(g_meta, e))
        g_meta.update_all(message_func=fn.u_mul_e('feat_src', 'a', 'm'), reduce_func=fn.sum('m', 'feat'))
        feat = self.activation(g_meta.dstdata['feat'])
        return feat.flatten(1)


class MAGNN_layer(nn.Module):

    def __init__(self, in_feats, inter_attn_feats, out_feats, num_heads, metapath_list, ntypes, edge_type_list, dst_ntypes, encoder_type='RotateE', last_layer=False):
        super(MAGNN_layer, self).__init__()
        self.in_feats = in_feats
        self.inter_attn_feats = inter_attn_feats
        self.out_feats = out_feats
        self.num_heads = num_heads
        self.metapath_list = metapath_list
        self.ntypes = ntypes
        self.edge_type_list = edge_type_list
        self.dst_ntypes = dst_ntypes
        self.encoder_type = encoder_type
        self.last_layer = last_layer
        in_feats_dst_meta = tuple((in_feats, in_feats))
        self.intra_attn_layers = nn.ModuleDict()
        for metapath in self.metapath_list:
            self.intra_attn_layers[metapath] = MAGNN_attn_intra(in_feats=in_feats_dst_meta, out_feats=in_feats, num_heads=num_heads)
        self.inter_linear = nn.ModuleDict()
        self.inter_attn_vec = nn.ModuleDict()
        for ntype in dst_ntypes:
            self.inter_linear[ntype] = nn.Linear(in_features=in_feats * num_heads, out_features=inter_attn_feats, bias=True)
            self.inter_attn_vec[ntype] = nn.Linear(in_features=inter_attn_feats, out_features=1, bias=False)
            nn.init.xavier_normal_(self.inter_linear[ntype].weight, gain=1.414)
            nn.init.xavier_normal_(self.inter_attn_vec[ntype].weight, gain=1.414)
        if encoder_type == 'RotateE':
            r_vec_ = nn.Parameter(th.empty(size=(len(edge_type_list) // 2, in_feats * num_heads // 2, 2)))
            nn.init.xavier_normal_(r_vec_.data, gain=1.414)
            self.r_vec = F.normalize(r_vec_, p=2, dim=2)
            self.r_vec = th.stack([self.r_vec, self.r_vec], dim=1)
            self.r_vec[:, 1, :, 1] = -self.r_vec[:, 1, :, 1]
            self.r_vec = self.r_vec.reshape(r_vec_.shape[0] * 2, r_vec_.shape[1], 2)
            self.r_vec_dict = nn.ParameterDict()
            for i, edge_type in zip(range(len(edge_type_list)), edge_type_list):
                self.r_vec_dict[edge_type] = nn.Parameter(self.r_vec[i])
        elif encoder_type == 'Linear':
            self.encoder_linear = nn.Linear(in_features=in_feats * num_heads, out_features=in_feats * num_heads)
        if last_layer:
            self._output_projection = nn.Linear(in_features=num_heads * in_feats, out_features=out_feats)
        else:
            self._output_projection = nn.Linear(in_features=num_heads * in_feats, out_features=num_heads * out_feats)
        nn.init.xavier_normal_(self._output_projection.weight, gain=1.414)

    def forward(self, feat_dict, metapath_idx_dict):
        feat_intra = {}
        for _metapath in self.metapath_list:
            feat_intra[_metapath] = self.intra_metapath_trans(feat_dict, metapath=_metapath, metapath_idx_dict=metapath_idx_dict)
        feat_inter = self.inter_metapath_trans(feat_dict=feat_dict, feat_intra=feat_intra, metapath_list=self.metapath_list)
        feat_final = self.output_projection(feat_inter=feat_inter)
        return feat_final, feat_inter

    def intra_metapath_trans(self, feat_dict, metapath, metapath_idx_dict):
        metapath_idx = metapath_idx_dict[metapath]
        intra_metapath_feat = self.encoder(feat_dict, metapath, metapath_idx)
        feat_intra = self.intra_attn_layers[metapath]([intra_metapath_feat, feat_dict[metapath.split('-')[0]]], metapath, metapath_idx)
        return feat_intra

    def inter_metapath_trans(self, feat_dict, feat_intra, metapath_list):
        meta_s = {}
        feat_inter = {}
        for metapath in metapath_list:
            _metapath = metapath.split('-')
            meta_feat = feat_intra[metapath]
            meta_feat = th.tanh(self.inter_linear[_metapath[0]](meta_feat)).mean(dim=0)
            meta_s[metapath] = self.inter_attn_vec[_metapath[0]](meta_feat)
        for ntype in self.ntypes:
            if ntype in self.dst_ntypes:
                metapaths = np.array(metapath_list)[[(meta.split('-')[0] == ntype) for meta in metapath_list]]
                meta_b = th.tensor(itemgetter(*metapaths)(meta_s))
                meta_b = F.softmax(meta_b, dim=0)
                meta_feat = itemgetter(*metapaths)(feat_intra)
                feat_inter[ntype] = th.stack([(meta_b[i] * meta_feat[i]) for i in range(len(meta_b))], dim=0).sum(dim=0)
            else:
                feat_inter[ntype] = feat_dict[ntype]
        return feat_inter

    def encoder(self, feat_dict, metapath, metapath_idx):
        _metapath = metapath.split('-')
        device = feat_dict[_metapath[0]].device
        feat = th.zeros((len(_metapath), metapath_idx.shape[0], feat_dict[_metapath[0]].shape[1]), device=device)
        for i, ntype in zip(range(len(_metapath)), _metapath):
            feat[i] = feat_dict[ntype][metapath_idx[:, i]]
        feat = feat.reshape(feat.shape[0], feat.shape[1], feat.shape[2] // 2, 2)
        if self.encoder_type == 'RotateE':
            temp_r_vec = th.zeros((len(_metapath), feat.shape[-2], 2), device=device)
            temp_r_vec[0, :, 0] = 1
            for i in range(1, len(_metapath), 1):
                edge_type = '{}-{}'.format(_metapath[i - 1], _metapath[i])
                temp_r_vec[i] = self.complex_hada(temp_r_vec[i - 1], self.r_vec_dict[edge_type])
                feat[i] = self.complex_hada(feat[i], temp_r_vec[i], opt='feat')
            feat = feat.reshape(feat.shape[0], feat.shape[1], -1)
            return th.mean(feat, dim=0)
        elif self.encoder_type == 'Linear':
            feat = feat.reshape(feat.shape[0], feat.shape[1], -1)
            feat = self.encoder_linear(th.mean(feat, dim=0))
            return feat
        elif self.encoder_type == 'Average':
            feat = feat.reshape(feat.shape[0], feat.shape[1], -1)
            return th.mean(feat, dim=0)
        else:
            raise ValueError('The encoder type {} has not been implemented yet.'.format(self.encoder_type))

    @staticmethod
    def complex_hada(h, v, opt='r_vec'):
        if opt == 'r_vec':
            h_h, l_h = h[:, 0].clone(), h[:, 1].clone()
        else:
            h_h, l_h = h[:, :, 0].clone(), h[:, :, 1].clone()
        h_v, l_v = v[:, 0].clone(), v[:, 1].clone()
        res = th.zeros_like(h)
        if opt == 'r_vec':
            res[:, 0] = h_h * h_v - l_h * l_v
            res[:, 1] = h_h * l_v + l_h * h_v
        else:
            res[:, :, 0] = h_h * h_v - l_h * l_v
            res[:, :, 1] = h_h * l_v + l_h * h_v
        return res

    def output_projection(self, feat_inter):
        feat_final = {}
        for ntype in self.ntypes:
            feat_final[ntype] = self._output_projection(feat_inter[ntype])
        return feat_final


def mp_instance_sampler(g, metapath_list, dataset):
    """
    Sampling the indices of all metapath instances in g according to the metapath list

    Parameters
    ----------
    g : object
        the dgl heterogeneous graph
    metapath_list : list
        the list of metapaths in g, e.g. ['M-A-M', M-D-M', ...]
    dataset : str
        the name of dataset, e.g. 'imdb4MAGNN'

    Returns
    -------
    dict
        the indices of all metapath instances. e.g dict['MAM'] contains the indices of all MAM instances

    Notes
    -----
    Please make sure that the metapath in metapath_list are all symmetric

    We'd store the metapath instances in the disk after one metapath instances sampling and next time the
    metapath instances will be extracted directly from the disk if they exists.

    """
    file_dir = 'openhgnn/output/MAGNN/'
    file_addr = file_dir + '{}'.format(dataset) + '_mp_inst.pkl'
    test = True
    if os.path.exists(file_addr) and test is False:
        with open(file_addr, 'rb') as file:
            res = pickle.load(file)
    else:
        etype_idx_dict = {}
        for etype in g.etypes:
            edges_idx_i = g.edges(etype=etype)[0].cpu().numpy()
            edges_idx_j = g.edges(etype=etype)[1].cpu().numpy()
            etype_idx_dict[etype] = pd.DataFrame([edges_idx_i, edges_idx_j]).T
            _etype = etype.split('-')
            etype_idx_dict[etype].columns = [_etype[0], _etype[1]]
        res = {}
        for metapath in metapath_list:
            res[metapath] = None
            _metapath = metapath.split('-')
            for i in range(1, len(_metapath) - 1):
                if i == 1:
                    res[metapath] = etype_idx_dict['-'.join(_metapath[:i + 1])]
                feat_j = etype_idx_dict['-'.join(_metapath[i:i + 2])]
                col_i = res[metapath].columns[-1]
                col_j = feat_j.columns[0]
                res[metapath] = pd.merge(res[metapath], feat_j, left_on=col_i, right_on=col_j, how='inner')
                if col_i != col_j:
                    res[metapath].drop(columns=col_j, inplace=True)
            res[metapath] = res[metapath].values
        with open(file_addr, 'wb') as file:
            pickle.dump(res, file)
    return res


class MAGNN(BaseModel):
    """
    This is the main method of model MAGNN

    Parameters
    ----------
    ntypes: list
        the nodes' types of the dataset
    h_feats: int
        hidden dimension
    inter_attn_feats: int
        the dimension of attention vector in inter-metapath aggregation
    num_heads: int
        the number of heads in intra metapath attention
    num_classes: int
        the number of output classes
    num_layers: int
        the number of hidden layers
    metapath_list: list
        the list of metapaths, e.g ['M-D-M', 'M-A-M', ...],
    edge_type_list: list
        the list of edge types, e.g ['M-A', 'A-M', 'M-D', 'D-M'],
    dropout_rate: float
        the dropout rate of feat dropout and attention dropout
    mp_instances : dict
        the metapath instances indices dict. e.g mp_instances['MAM'] stores MAM instances indices.
    encoder_type: str
        the type of encoder, e.g ['RotateE', 'Average', 'Linear']
    activation: callable activation function
        the activation function used in MAGNN.  default: F.elu

    Notes
    -----
    Please make sure that the please make sure that all the metapath is symmetric, e.g ['MDM', 'MAM' ...] are symmetric,
    while ['MAD', 'DAM', ...] are not symmetric.

    please make sure that the edge_type_list meets the following form:
    [edge_type_1, edge_type_1_reverse, edge_type_2, edge_type_2_reverse, ...], like the example above.

    All the activation in MAGNN are the same according to the codes of author.

    """

    @classmethod
    def build_model_from_args(cls, args, hg):
        ntypes = hg.ntypes
        if args.dataset == 'imdb4MAGNN':
            metapath_list = ['M-D-M', 'M-A-M', 'D-M-D', 'D-M-A-M-D', 'A-M-A', 'A-M-D-M-A']
            edge_type_list = ['A-M', 'M-A', 'D-M', 'M-D']
            in_feats = {'M': 3066, 'D': 2081, 'A': 5257}
            metapath_idx_dict = mp_instance_sampler(hg, metapath_list, 'imdb4MAGNN')
        elif args.dataset == 'dblp4MAGNN':
            metapath_list = ['A-P-A', 'A-P-T-P-A', 'A-P-V-P-A']
            edge_type_list = ['A-P', 'P-A', 'P-T', 'T-P', 'P-V', 'V-P']
            in_feats = {'A': 334, 'P': 14328, 'T': 7723, 'V': 20}
            metapath_idx_dict = mp_instance_sampler(hg, metapath_list, 'dblp4MAGNN')
        else:
            raise NotImplementedError('MAGNN on dataset {} has not been implemented'.format(args.dataset))
        return cls(ntypes=ntypes, h_feats=args.hidden_dim // args.num_heads, inter_attn_feats=args.inter_attn_feats, num_heads=args.num_heads, num_classes=args.out_dim, num_layers=args.num_layers, metapath_list=metapath_list, edge_type_list=edge_type_list, dropout_rate=args.dropout, encoder_type=args.encoder_type, metapath_idx_dict=metapath_idx_dict)

    def __init__(self, ntypes, h_feats, inter_attn_feats, num_heads, num_classes, num_layers, metapath_list, edge_type_list, dropout_rate, metapath_idx_dict, encoder_type='RotateE', activation=F.elu):
        super(MAGNN, self).__init__()
        self.encoder_type = encoder_type
        self.ntypes = ntypes
        self.h_feats = h_feats
        self.inter_attn_feats = inter_attn_feats
        self.num_heads = num_heads
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.metapath_list = metapath_list
        self.edge_type_list = edge_type_list
        self.activation = activation
        self.backup = {}
        self.is_backup = False
        self.feat_drop = nn.Dropout(p=dropout_rate)
        self.dst_ntypes = set([metapath.split('-')[0] for metapath in metapath_list])
        self.layers = nn.ModuleList()
        for i in range(num_layers - 1):
            self.layers.append(MAGNN_layer(in_feats=h_feats, inter_attn_feats=inter_attn_feats, out_feats=h_feats, num_heads=num_heads, metapath_list=metapath_list, ntypes=self.ntypes, edge_type_list=edge_type_list, dst_ntypes=self.dst_ntypes, encoder_type=encoder_type, last_layer=False))
        self.layers.append(MAGNN_layer(in_feats=h_feats, inter_attn_feats=inter_attn_feats, out_feats=num_classes, num_heads=num_heads, metapath_list=metapath_list, ntypes=self.ntypes, edge_type_list=edge_type_list, dst_ntypes=self.dst_ntypes, encoder_type=encoder_type, last_layer=True))
        self.metapath_idx_dict = metapath_idx_dict

    def mini_reset_params(self, new_metapth_idx_dict):
        """
        This method is utilized for reset some parameters including metapath_idx_dict, metapath_list, dst_ntypes...
        Other Parameters like weight matrix don't need to be updated.

        """
        if not self.is_backup:
            self.backup['metapath_idx_dict'] = self.metapath_idx_dict
            self.backup['metapath_list'] = self.metapath_list
            self.backup['dst_ntypes'] = self.dst_ntypes
            self.is_backup = True
        self.metapath_idx_dict = new_metapth_idx_dict
        self.metapath_list = list(new_metapth_idx_dict.keys())
        self.dst_ntypes = set([meta[0] for meta in self.metapath_list])
        for layer in self.layers:
            layer.metapath_list = self.metapath_list
            layer.dst_ntypes = self.dst_ntypes

    def restore_params(self):
        assert self.backup, 'The model.backup is empty'
        self.metapath_idx_dict = self.backup['metapath_idx_dict']
        self.metapath_list = self.backup['metapath_list']
        self.dst_ntypes = self.backup['dst_ntypes']
        for layer in self.layers:
            layer.metapath_list = self.metapath_list
            layer.dst_ntypes = self.dst_ntypes

    def forward(self, g, feat_dict=None):
        """
        The forward part of MAGNN

        Parameters
        ----------
        g : object
            the dgl heterogeneous graph
        feat_dict : dict
            the feature matrix dict of different node types, e.g {'M':feat_of_M, 'D':feat_of_D, ...}

        Returns
        -------
        dict
            The predicted logit after the output projection. e.g For the predicted node type, such as M(movie),
            dict['M'] contains the probability that each node is classified as each class. For other node types, such as
            D(director), dict['D'] contains the result after the output projection.

        dict
            The embeddings before the output projection. e.g dict['M'] contains embeddings of every node of M type.
        """
        for i in range(self.num_layers - 1):
            h, _ = self.layers[i](feat_dict, self.metapath_idx_dict)
            for key in h.keys():
                h[key] = self.activation(h[key])
        h_output, embedding = self.layers[-1](feat_dict, self.metapath_idx_dict)
        return h_output


class HMAELayer(nn.Module):
    """
        HMAE: Hybrid Metapath Autonomous Extraction

        The method to generate l-hop hybrid adjacency matrix

        .. math::
            A_{(l)}=\\Pi_{i=1}^{l} A_{i}
    """

    def __init__(self, in_channels, out_channels, first=True):
        super(HMAELayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.first = first
        self.norm = EdgeWeightNorm(norm='right')
        if self.first == True:
            self.conv1 = GTConv(in_channels, out_channels, softmax_flag=False)
            self.conv2 = GTConv(in_channels, out_channels, softmax_flag=False)
        else:
            self.conv1 = GTConv(in_channels, out_channels, softmax_flag=False)

    def softmax_norm(self, H):
        norm_H = []
        for i in range(len(H)):
            g = H[i]
            g.edata['w_sum'] = self.norm(g, th.exp(g.edata['w_sum']))
            norm_H.append(g)
        return norm_H

    def forward(self, A, H_=None):
        if self.first == True:
            result_A = self.softmax_norm(self.conv1(A))
            result_B = self.softmax_norm(self.conv2(A))
            W = [self.conv1.weight.detach(), self.conv2.weight.detach()]
        else:
            result_A = H_
            result_B = self.conv1(A)
            W = [self.conv1.weight.detach().detach()]
        H = []
        for i in range(len(result_A)):
            g = dgl.adj_product_graph(result_A[i], result_B[i], 'w_sum')
            H.append(g)
        return H, W, result_A


class HLHIA(nn.Module):
    """
        HLHIA: The Hop-Level Heterogeneous Information Aggregation

        The l-hop representation :math:`Z_{l}` is generated by the original node feature through a graph conv

        .. math::
           Z_{l}^{\\Phi_{p}} = \\sigma\\left[\\left(D_{(l)}^{\\Phi_{p}}\\right)^{-1} A_{(l)}^{\\Phi_{p}} h W^{\\Phi_{p}}\\right]

        where :math:`\\Phi_{p}` is the hybrid l-hop metapath and `\\mathcal{h}` is the original node feature.

    """

    def __init__(self, num_edge_type, num_channels, num_layers, in_dim, hidden_dim):
        super(HLHIA, self).__init__()
        self.num_channels = num_channels
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.append(HMAELayer(num_edge_type, num_channels, first=True))
            else:
                layers.append(HMAELayer(num_edge_type, num_channels, first=False))
        self.layers = nn.ModuleList(layers)
        self.gcn_list = nn.ModuleList()
        for i in range(num_channels):
            self.gcn_list.append(GraphConv(in_feats=self.in_dim, out_feats=hidden_dim, norm='none', activation=F.relu))
        self.norm = EdgeWeightNorm(norm='right')

    def forward(self, A, h):
        layer_list = []
        for i in range(len(self.layers)):
            if i == 0:
                H, W, first_adj = self.layers[i](A)
                layer_list.append(first_adj)
                layer_list.append(H)
            else:
                H, W, first_adj = self.layers[i](A, H)
                layer_list.append(H)
        channel_attention_list = []
        for i in range(self.num_channels):
            gcn = self.gcn_list[i]
            layer_attention_list = []
            for j in range(len(layer_list)):
                layer = layer_list[j][i]
                layer = dgl.remove_self_loop(layer)
                edge_weight = layer.edata['w_sum']
                layer = dgl.add_self_loop(layer)
                edge_weight = th.cat((edge_weight, th.full((layer.number_of_nodes(),), 1, device=layer.device)))
                edge_weight = self.norm(layer, edge_weight)
                layer_attention_list.append(gcn(layer, h, edge_weight=edge_weight))
            channel_attention_list.append(layer_attention_list)
        return channel_attention_list


class HSAF(nn.Module):
    """
        HSAF: Hierarchical Semantic Attention Fusion

        The HSAF model use two level attention mechanism to generate final representation

        * Hop-level attention

          .. math::
              \\alpha_{i, l}^{\\Phi_{p}}=\\sigma\\left[\\delta^{\\Phi_{p}} \\tanh \\left(W^{\\Phi_{p}} Z_{i, l}^{\\Phi_{p}}\\right)\\right]

          In which, :math:`\\alpha_{i, l}^{\\Phi_{p}}` is the importance of the information :math:`\\left(Z_{i, l}^{\\Phi_{p}}\\right)`
          of the l-th-hop neighbors of node i under the path :math:`\\Phi_{p}`, and :math:`\\delta^{\\Phi_{p}}` represents the learnable matrix.

          Then normalize :math:`\\alpha_{i, l}^{\\Phi_{p}}`

          .. math::
              \\beta_{i, l}^{\\Phi_{p}}=\\frac{\\exp \\left(\\alpha_{i, l}^{\\Phi_{p}}\\right)}{\\sum_{j=1}^{L} \\exp \\left(\\alpha_{i, j}^{\\Phi_{p}}\\right)}

          Finally, we get hop-level attention representation in one hybrid metapath.

          .. math::
              Z_{i}^{\\Phi_{p}}=\\sum_{l=1}^{L} \\beta_{l}^{\\Phi_{p}} Z_{l}^{\\Phi_{p}}

        * Channel-level attention

          It also can be seen as multi-head attention mechanism.

          .. math::
              \\alpha_{i, \\Phi_{p}}=\\sigma\\left[\\delta \\tanh \\left(W Z_{i}^{\\Phi_{p}}\\right)\\right.

          Then normalize :math:`\\alpha_{i, \\Phi_{p}}`

          .. math::
              \\beta_{i, \\Phi_{p}}=\\frac{\\exp \\left(\\alpha_{i, \\Phi_{p}}\\right)}{\\sum_{p^{\\prime} \\in P} \\exp \\left(\\alpha_{\\Phi_{p^{\\prime}}}\\right)}

          Finally, we get final representation of every nodes.

          .. math::
              Z_{i}=\\sum_{p \\in P} \\beta_{i, \\Phi_{p}} Z_{i, \\Phi_{p}}

    """

    def __init__(self, num_edge_type, num_channels, num_layers, in_dim, hidden_dim):
        super(HSAF, self).__init__()
        self.num_channels = num_channels
        self.num_layers = num_layers
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.HLHIA_layer = HLHIA(num_edge_type, self.num_channels, self.num_layers, self.in_dim, self.hidden_dim)
        self.channel_attention = nn.Sequential(nn.Linear(self.hidden_dim, 1), nn.Tanh(), nn.Linear(1, 1, bias=False), nn.ReLU())
        self.layers_attention = nn.ModuleList()
        for i in range(num_channels):
            self.layers_attention.append(nn.Sequential(nn.Linear(self.hidden_dim, 1), nn.Tanh(), nn.Linear(1, 1, bias=False), nn.ReLU()))

    def forward(self, A, h):
        attention_list = self.HLHIA_layer(A, h)
        channel_attention_list = []
        for i in range(self.num_channels):
            layer_level_feature_list = attention_list[i]
            layer_attention = self.layers_attention[i]
            for j in range(self.num_layers + 1):
                layer_level_feature = layer_level_feature_list[j]
                if j == 0:
                    layer_level_alpha = layer_attention(layer_level_feature)
                else:
                    layer_level_alpha = th.cat((layer_level_alpha, layer_attention(layer_level_feature)), dim=-1)
            layer_level_beta = th.softmax(layer_level_alpha, dim=-1)
            channel_attention_list.append(th.bmm(th.stack(layer_level_feature_list, dim=-1), layer_level_beta.unsqueeze(-1)).squeeze(-1))
        for i in range(self.num_channels):
            channel_level_feature = channel_attention_list[i]
            if i == 0:
                channel_level_alpha = self.channel_attention(channel_level_feature)
            else:
                channel_level_alpha = th.cat((channel_level_alpha, self.channel_attention(channel_level_feature)), dim=-1)
        channel_level_beta = th.softmax(channel_level_alpha, dim=-1)
        channel_attention = th.bmm(th.stack(channel_attention_list, dim=-1), channel_level_beta.unsqueeze(-1)).squeeze(-1)
        return channel_attention


class MHNF(BaseModel):
    """
        MHNF from paper `Multi-hop Heterogeneous Neighborhood information Fusion graph representation learning
        <https://arxiv.org/pdf/2106.09289.pdf>`__.

        Given a heterogeneous graph :math:`G` and its edge relation type set :math:`\\mathcal{R}`.Then we can extract l-hops hybrid adjacency matrix list
        in HMAE model. The hybrid adjacency matrix list can be used in HLHIA model to generate l-hops representations. Then HSAF
        model use attention mechanism to aggregate l-hops representations and because of multi-channel conv, the
        HSAF model also  aggregates different channels l-hops representations to generate a final representation.
        You can see detail operation in correspond model.

        Parameters
        ----------
        num_edge_type : int
            Number of relations.
        num_channels : int
            Number of conv channels.
        in_dim : int
            The dimension of input feature.
        hidden_dim : int
            The dimension of hidden layer.
        num_class : int
            Number of classification type.
        num_layers : int
            Length of hybrid metapath.
        category : string
            Type of predicted nodes.
        norm : bool
            If True, the adjacency matrix will be normalized.
        identity : bool
            If True, the identity matrix will be added to relation matrix set.

    """

    @classmethod
    def build_model_from_args(cls, args, hg):
        if args.identity == True:
            num_edge_type = len(hg.canonical_etypes) + 1
        else:
            num_edge_type = len(hg.canonical_etypes)
        return cls(num_edge_type=num_edge_type, num_channels=args.num_channels, in_dim=args.hidden_dim, hidden_dim=args.hidden_dim, num_class=args.out_dim, num_layers=args.num_layers, category=args.category, norm=args.norm_emd_flag, identity=args.identity)

    def __init__(self, num_edge_type, num_channels, in_dim, hidden_dim, num_class, num_layers, category, norm, identity):
        super(MHNF, self).__init__()
        self.num_edge_type = num_edge_type
        self.num_channels = num_channels
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_class = num_class
        self.num_layers = num_layers
        self.is_norm = norm
        self.category = category
        self.identity = identity
        self.HSAF = HSAF(num_edge_type, self.num_channels, self.num_layers, self.in_dim, self.hidden_dim)
        self.linear = nn.Linear(self.hidden_dim, self.num_class)
        self.category_idx = None
        self.A = None
        self.h = None

    def forward(self, hg, h=None):
        with hg.local_scope():
            hg.ndata['h'] = h
            if self.category_idx is None:
                self.A, h, self.category_idx = transform_relation_graph_list(hg, category=self.category, identity=self.identity)
            else:
                g = dgl.to_homogeneous(hg, ndata='h')
                h = g.ndata['h']
            A = self.A
            final_representation = self.HSAF(A, h)
            y = self.linear(final_representation)
            return {self.category: y[self.category_idx]}


class MeiREC(BaseModel):
    """
        MeiREC from paper `Metapath-guided Heterogeneous Graph Neural Network for
        Intent Recommendation <https://dl.acm.org/doi/abs/10.1145/3292500.3330673>`__
        in KDD_2019.

        `Code from author <https://github.com/googlebaba/KDD2019-MEIRec>`__.

        We leverage metapaths to obtain different-step neighbors of an object, and the embeddings of us
        ers and queries are the aggregation of their neighbors under different metapaths.And we propose
        to represent the queries and items with a small number of term embeddings.we need to learn the
        term embeddings, rather than all object embeddings. This method is able to significantly reduc
        e the number of parameters.

        Parameters
        ----------
        user_seq_length : int
            Number for process dataset.
        ...
        batch_num : int
            Number of batch.
        weight_decay : float
            Number of weight_decay.
        lr : float
            learning rate.
        train_epochs : int
            Number of train epoch.
        -----------
    """

    @classmethod
    def build_model_from_args(cls, config):
        return cls(config)

    def __init__(self, config):
        super().__init__()
        self.model = Model(config)

    def forward(self, *args):
        return self.model(*args)

    def extra_loss(self):
        pass


def filter(triplets_to_filter, target_s, target_r, target_o, num_entities, mode):
    triplets_to_filter = triplets_to_filter.copy()
    target_s, target_r, target_o = int(target_s), int(target_r), int(target_o)
    filtered = []
    if (target_s, target_r, target_o) in triplets_to_filter:
        triplets_to_filter.remove((target_s, target_r, target_o))
    if mode == 's':
        for s in range(num_entities):
            if (s, target_r, target_o) not in triplets_to_filter:
                filtered.append(s)
    elif mode == 'o':
        for o in range(num_entities):
            if (target_s, target_r, o) not in triplets_to_filter:
                filtered.append(o)
    return th.LongTensor(filtered)


def perturb_and_get_rank(n_embedding, r_embedding, eval_triplets, triplets_to_filter, score_predictor, filtered, preturb_side):
    """ Perturb object in the triplets
    """
    ranks = []
    num_entities = n_embedding.shape[0]
    eval_range = tqdm(range(eval_triplets.shape[0]), ncols=100)
    for idx in eval_range:
        target_s = eval_triplets[idx, 0]
        target_r = eval_triplets[idx, 1]
        target_o = eval_triplets[idx, 2]
        if filtered == 'filtered':
            if preturb_side == 'o':
                select_s = target_s
                select_o = filter(triplets_to_filter, target_s, target_r, target_o, num_entities, 'o')
                target_idx = int((select_o == target_o).nonzero())
            elif preturb_side == 's':
                select_s = filter(triplets_to_filter, target_s, target_r, target_o, num_entities, 's')
                select_o = target_o
                target_idx = int((select_s == target_s).nonzero())
        elif filtered == 'raw':
            if preturb_side == 'o':
                select_s = target_s
                select_o = th.arange(num_entities)
                target_idx = target_o
            elif preturb_side == 's':
                select_o = target_o
                select_s = th.arange(num_entities)
                target_idx = target_s
        emb_s = n_embedding[select_s]
        emb_r = r_embedding[int(target_r)]
        emb_o = n_embedding[select_o]
        scores = score_predictor(emb_s, emb_r, emb_o)
        _, indices = th.sort(scores, descending=False)
        rank = int((indices == target_idx).nonzero())
        ranks.append(rank)
    return th.LongTensor(ranks)


def cal_mrr(n_embedding, r_embedding, valid_triplets, test_triplets, triplets_to_filter, score_predictor, hits=[], filtered='raw', eval_mode='test'):
    with th.no_grad():
        eval_triplets = test_triplets if eval_mode == 'test' else valid_triplets
        None
        ranks_s = perturb_and_get_rank(n_embedding, r_embedding, eval_triplets, triplets_to_filter, score_predictor, filtered, 's')
        None
        ranks_o = perturb_and_get_rank(n_embedding, r_embedding, eval_triplets, triplets_to_filter, score_predictor, filtered, 'o')
        ranks = th.cat([ranks_s, ranks_o])
        ranks += 1
        mrr_matrix = {'Mode': filtered, 'MR': th.mean(ranks.float()).item(), 'MRR': th.mean(1.0 / ranks.float()).item()}
        for hit in hits:
            mrr_matrix['Hits@' + str(hit)] = th.mean((ranks <= hit).float()).item()
        return mrr_matrix


def concat_u_v(x, u_idx, v_idx):
    u = x[u_idx]
    v = x[v_idx]
    emd = th.cat((u, v), dim=1)
    return emd


class Evaluator:

    def __init__(self, seed):
        self.seed = seed

    def cluster(self, n, X, Y):
        X = np.array(X)
        Y = np.array(Y)
        Y_pred = KMeans(n, random_state=self.seed).fit(X).predict(X)
        nmi = normalized_mutual_info_score(Y, Y_pred)
        ari = adjusted_rand_score(Y, Y_pred)
        return nmi, ari

    def classification(self, X, Y):
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=self.seed)
        LR = LogisticRegression(max_iter=10000)
        LR.fit(X_train, Y_train)
        Y_pred = LR.predict(X_test)
        macro_f1, micro_f1 = f1_node_classification(Y_test, Y_pred)
        return micro_f1, macro_f1

    def f1_node_classification(self, y_label, y_pred):
        macro_f1 = f1_score(y_label, y_pred, average='macro')
        micro_f1 = f1_score(y_label, y_pred, average='micro')
        return dict(Macro_f1=macro_f1, Micro_f1=micro_f1)

    def cal_acc(self, y_true, y_pred):
        return accuracy_score(y_true, y_pred)

    def cal_roc_auc(self, y_true, y_pred):
        return roc_auc_score(y_true, y_pred)

    def mrr_(self, n_embedding, r_embedding, train_triplets, valid_triplets, test_triplets, score_predictor, hits=[], filtered='raw', eval_mode='test'):
        if not hasattr(self, 'triplets_to_filter'):
            triplets_to_filter = th.cat([train_triplets, valid_triplets, test_triplets]).tolist()
            self.triplets_to_filter = {tuple(triplet) for triplet in triplets_to_filter}
        return cal_mrr(n_embedding, r_embedding, valid_triplets, test_triplets, self.triplets_to_filter, score_predictor, hits, filtered, eval_mode)

    def ndcg(self, y_score, y_true):
        return ndcg_score(y_true, y_score, 10)
    """"""

    def LR_pred(self, train_X, train_Y, test_X):
        LR = LogisticRegression(max_iter=10000)
        LR.fit(train_X, train_Y)
        pred_Y = LR.predict(test_X)
        return pred_Y

    def link_prediction(self, train_X, train_Y, test_X, test_Y):
        pred_Y = self.LR_pred(train_X, train_Y, test_X)
        AUC_score = Metric.roc_auc_score(test_Y, pred_Y)
        macro_f1, micro_f1 = f1_node_classification(test_Y, pred_Y)
        return AUC_score, macro_f1, micro_f1

    def author_link_prediction(self, x, train_batch, test_batch):
        train_u, train_v, train_Y = train_batch
        test_u, test_v, test_Y = test_batch
        train_X = concat_u_v(x, th.tensor(train_u), th.tensor(train_v))
        test_X = concat_u_v(x, th.tensor(test_u), th.tensor(test_v))
        train_Y = th.tensor(train_Y)
        test_Y = th.tensor(test_Y)
        return self.link_prediction(train_X, train_Y, test_X, test_Y)
    """"""

    def nc_with_LR(self, emd, labels, train_idx, test_idx):
        Y_train = labels[train_idx]
        Y_test = labels[test_idx]
        LR = LogisticRegression(max_iter=10000)
        X_train = emd[train_idx]
        X_test = emd[test_idx]
        LR.fit(X_train, Y_train)
        Y_pred = LR.predict(X_test)
        macro_f1, micro_f1 = f1_node_classification(Y_test, Y_pred)
        return micro_f1, macro_f1

    def ec_with_SVC(self, C, gamma, emd, labels, train_idx, test_idx):
        X_train = emd[train_idx]
        Y_train = labels[train_idx]
        X_test = emd[test_idx]
        Y_test = labels[test_idx]
        scaler = preprocessing.StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        clf = SVC(C=C, gamma=gamma).fit(X_train, Y_train)
        Y_pred = clf.predict(X_test)
        macro_f1 = metrics.f1_score(Y_test, Y_pred, average='macro')
        micro_f1 = metrics.f1_score(Y_test, Y_pred, average='micro')
        acc = metrics.accuracy_score(Y_test, Y_pred)
        return micro_f1, macro_f1, acc

    def prediction(self, real_score, pred_score):
        MAE = mean_absolute_error(real_score, pred_score)
        RMSE = math.sqrt(mean_squared_error(real_score, pred_score))
        return MAE, RMSE

    def dcg_at_k(self, scores):
        return scores[0] + sum(sc / math.log(ind + 1, 2) for sc, ind in zip(scores[1:], range(2, len(scores) + 1)))

    def ndcg_at_k(self, real_scores, predicted_scores):
        idcg = self.dcg_at_k(sorted(real_scores, reverse=True))
        return self.dcg_at_k(predicted_scores) / idcg if idcg > 0.0 else 0.0

    def ranking(self, real_score, pred_score, k):
        sorted_idx = sorted(np.argsort(real_score)[::-1][:k])
        r_s_at_k = real_score[sorted_idx]
        p_s_at_k = pred_score[sorted_idx]
        ndcg_5 = self.ndcg_at_k(r_s_at_k, p_s_at_k)
        return ndcg_5


class ItemEmbeddingDB(torch.nn.Module):

    def __init__(self, config):
        super(ItemEmbeddingDB, self).__init__()
        self.num_publisher = config.num_publisher
        self.embedding_dim = config.embedding_dim
        self.embedding_publisher = torch.nn.Embedding(num_embeddings=self.num_publisher, embedding_dim=self.embedding_dim)

    def forward(self, item_fea):
        """
        :param item_fea:
        :return:
        """
        publisher_idx = Variable(item_fea[:, 0], requires_grad=False)
        publisher_emb = self.embedding_publisher(publisher_idx)
        return publisher_emb


class MetaLearner(torch.nn.Module):

    def __init__(self, config):
        super(MetaLearner, self).__init__()
        self.embedding_dim = config.embedding_dim
        self.fc1_in_dim = 32 + config.item_embedding_dim
        self.fc2_in_dim = config.first_fc_hidden_dim
        self.fc2_out_dim = config.second_fc_hidden_dim
        self.use_cuda = config.use_cuda
        self.config = config
        self.vars = torch.nn.ParameterDict()
        self.vars_bn = torch.nn.ParameterList()
        w1 = torch.nn.Parameter(torch.ones([self.fc2_in_dim, self.fc1_in_dim]))
        torch.nn.init.xavier_normal_(w1)
        self.vars['ml_fc_w1'] = w1
        self.vars['ml_fc_b1'] = torch.nn.Parameter(torch.zeros(self.fc2_in_dim))
        w2 = torch.nn.Parameter(torch.ones([self.fc2_out_dim, self.fc2_in_dim]))
        torch.nn.init.xavier_normal_(w2)
        self.vars['ml_fc_w2'] = w2
        self.vars['ml_fc_b2'] = torch.nn.Parameter(torch.zeros(self.fc2_in_dim))
        w3 = torch.nn.Parameter(torch.ones([1, self.fc2_out_dim]))
        torch.nn.init.xavier_normal_(w3)
        self.vars['ml_fc_w3'] = w3
        self.vars['ml_fc_b3'] = torch.nn.Parameter(torch.zeros(1))

    def forward(self, user_emb, item_emb, user_neigh_emb, vars_dict=None):
        """ """
        if vars_dict is None:
            vars_dict = self.vars
        x_i = item_emb
        x_u = user_neigh_emb
        x = torch.cat((x_i, x_u), 1)
        x = F.relu(F.linear(x, vars_dict['ml_fc_w1'], vars_dict['ml_fc_b1']))
        x = F.relu(F.linear(x, vars_dict['ml_fc_w2'], vars_dict['ml_fc_b2']))
        x = F.linear(x, vars_dict['ml_fc_w3'], vars_dict['ml_fc_b3'])
        return x.squeeze()

    def zero_grad(self, vars_dict=None):
        with torch.no_grad():
            if vars_dict is None:
                for p in self.vars.values():
                    if p.grad is not None:
                        p.grad.zero_()
            else:
                for p in vars_dict.values():
                    if p.grad is not None:
                        p.grad.zero_()

    def update_parameters(self):
        return self.vars


class MetapathLearner(torch.nn.Module):

    def __init__(self, config):
        super(MetapathLearner, self).__init__()
        self.config = config
        self.vars = torch.nn.ParameterDict()
        neigh_w = torch.nn.Parameter(torch.ones([32, config.item_embedding_dim]))
        torch.nn.init.xavier_normal_(neigh_w)
        self.vars['neigh_w'] = neigh_w
        self.vars['neigh_b'] = torch.nn.Parameter(torch.zeros(32))

    def forward(self, user_emb, item_emb, neighs_emb, mp, index_list, vars_dict=None):
        """ """
        if vars_dict is None:
            vars_dict = self.vars
        agg_neighbor_emb = F.linear(neighs_emb, vars_dict['neigh_w'], vars_dict['neigh_b'])
        output_emb = F.leaky_relu(torch.mean(agg_neighbor_emb, 0)).repeat(user_emb.shape[0], 1)
        return output_emb

    def zero_grad(self, vars_dict=None):
        with torch.no_grad():
            if vars_dict is None:
                for p in self.vars.values():
                    if p.grad is not None:
                        p.grad.zero_()
            else:
                for p in vars_dict.values():
                    if p.grad is not None:
                        p.grad.zero_()

    def update_parameters(self):
        return self.vars


class UserEmbeddingDB(torch.nn.Module):

    def __init__(self, config):
        super(UserEmbeddingDB, self).__init__()
        self.num_location = config.num_location
        self.embedding_dim = config.embedding_dim
        self.embedding_location = torch.nn.Embedding(num_embeddings=self.num_location, embedding_dim=self.embedding_dim)

    def forward(self, user_fea):
        """
        :param user_fea: tensor, shape = [#sample, #user_fea]
        :return:
        """
        location_idx = Variable(user_fea[:, 0], requires_grad=False)
        location_emb = self.embedding_location(location_idx)
        return location_emb


class MetaHIN(BaseModel):

    @classmethod
    def build_model_from_args(cls, args):
        return cls(args, args.model_name)

    def __init__(self, config, model_name):
        super(MetaHIN, self).__init__()
        self.config = config
        self.mp = ['ub', 'ubab', 'ubub']
        self.device = torch.device('cuda' if self.config.use_cuda else 'cpu')
        self.model_name = model_name
        self.item_emb = ItemEmbeddingDB(config)
        self.user_emb = UserEmbeddingDB(config)
        self.mp_learner = MetapathLearner(config)
        self.meta_learner = MetaLearner(config)
        self.mp_lr = config.mp_lr
        self.local_lr = config.local_lr
        self.emb_dim = self.config.embedding_dim
        self.cal_metrics = Evaluator(config.seed)
        self.ml_weight_len = len(self.meta_learner.update_parameters())
        self.ml_weight_name = list(self.meta_learner.update_parameters().keys())
        self.mp_weight_len = len(self.mp_learner.update_parameters())
        self.mp_weight_name = list(self.mp_learner.update_parameters().keys())
        self.transformer_liners = self.transform_mp2task()
        self.meta_optimizer = torch.optim.Adam(self.parameters(), lr=config.lr)

    def transform_mp2task(self):
        liners = {}
        ml_parameters = self.meta_learner.update_parameters()
        output_dim_of_mp = 32
        for w in self.ml_weight_name:
            liners[w.replace('.', '-')] = torch.nn.Linear(output_dim_of_mp, np.prod(ml_parameters[w].shape))
        return torch.nn.ModuleDict(liners)

    def forward(self, support_user_emb, support_item_emb, support_set_y, support_mp_user_emb, vars_dict=None):
        """ """
        if vars_dict is None:
            vars_dict = self.meta_learner.update_parameters()
        support_set_y_pred = self.meta_learner(support_user_emb, support_item_emb, support_mp_user_emb, vars_dict)
        loss = F.mse_loss(support_set_y_pred, support_set_y)
        grad = torch.autograd.grad(loss, vars_dict.values(), create_graph=True)
        fast_weights = {}
        for i, w in enumerate(vars_dict.keys()):
            fast_weights[w] = vars_dict[w] - self.local_lr * grad[i]
        for idx in range(1, self.config.local_update):
            support_set_y_pred = self.meta_learner(support_user_emb, support_item_emb, support_mp_user_emb, vars_dict=fast_weights)
            loss = F.mse_loss(support_set_y_pred, support_set_y)
            grad = torch.autograd.grad(loss, fast_weights.values(), create_graph=True)
            for i, w in enumerate(fast_weights.keys()):
                fast_weights[w] = fast_weights[w] - self.local_lr * grad[i]
        return fast_weights

    def mp_update(self, support_set_x, support_set_y, support_set_mps, query_set_x, query_set_y, query_set_mps):
        """
        Mete-update the parameters of MetaPathLearner, AggLearner and MetaLearner.
        """
        support_mp_enhanced_user_emb_s, query_mp_enhanced_user_emb_s = [], []
        mp_task_fast_weights_s = {}
        mp_task_loss_s = {}
        mp_initial_weights = self.mp_learner.update_parameters()
        ml_initial_weights = self.meta_learner.update_parameters()
        support_user_emb = self.user_emb(support_set_x[:, self.config.item_fea_len:])
        support_item_emb = self.item_emb(support_set_x[:, 0:self.config.item_fea_len])
        query_user_emb = self.user_emb(query_set_x[:, self.config.item_fea_len:])
        query_item_emb = self.item_emb(query_set_x[:, 0:self.config.item_fea_len])
        for mp in self.mp:
            support_set_mp = list(support_set_mps[mp])
            query_set_mp = list(query_set_mps[mp])
            support_neighs_emb = self.item_emb(torch.cat(support_set_mp))
            support_index_list = list(map(lambda _: _.shape[0], support_set_mp))
            query_neighs_emb = self.item_emb(torch.cat(query_set_mp))
            query_index_list = list(map(lambda _: _.shape[0], query_set_mp))
            support_mp_enhanced_user_emb = self.mp_learner(support_user_emb, support_item_emb, support_neighs_emb, mp, support_index_list)
            support_set_y_pred = self.meta_learner(support_user_emb, support_item_emb, support_mp_enhanced_user_emb)
            loss = F.mse_loss(support_set_y_pred, support_set_y)
            grad = torch.autograd.grad(loss, mp_initial_weights.values(), create_graph=True)
            fast_weights = {}
            for i in range(self.mp_weight_len):
                weight_name = self.mp_weight_name[i]
                fast_weights[weight_name] = mp_initial_weights[weight_name] - self.mp_lr * grad[i]
            for idx in range(1, self.config.mp_update):
                support_mp_enhanced_user_emb = self.mp_learner(support_user_emb, support_item_emb, support_neighs_emb, mp, support_index_list, vars_dict=fast_weights)
                support_set_y_pred = self.meta_learner(support_user_emb, support_item_emb, support_mp_enhanced_user_emb)
                loss = F.mse_loss(support_set_y_pred, support_set_y)
                grad = torch.autograd.grad(loss, fast_weights.values(), create_graph=True)
                for i in range(self.mp_weight_len):
                    weight_name = self.mp_weight_name[i]
                    fast_weights[weight_name] = fast_weights[weight_name] - self.mp_lr * grad[i]
            support_mp_enhanced_user_emb = self.mp_learner(support_user_emb, support_item_emb, support_neighs_emb, mp, support_index_list, vars_dict=fast_weights)
            support_mp_enhanced_user_emb_s.append(support_mp_enhanced_user_emb)
            query_mp_enhanced_user_emb = self.mp_learner(query_user_emb, query_item_emb, query_neighs_emb, mp, query_index_list, vars_dict=fast_weights)
            query_mp_enhanced_user_emb_s.append(query_mp_enhanced_user_emb)
            f_fast_weights = {}
            for w, liner in self.transformer_liners.items():
                w = w.replace('-', '.')
                f_fast_weights[w] = ml_initial_weights[w] * torch.sigmoid(liner(support_mp_enhanced_user_emb.mean(0))).view(ml_initial_weights[w].shape)
            mp_task_fast_weights = self.forward(support_user_emb, support_item_emb, support_set_y, support_mp_enhanced_user_emb, vars_dict=f_fast_weights)
            mp_task_fast_weights_s[mp] = mp_task_fast_weights
            query_set_y_pred = self.meta_learner(query_user_emb, query_item_emb, query_mp_enhanced_user_emb, vars_dict=mp_task_fast_weights)
            q_loss = F.mse_loss(query_set_y_pred, query_set_y)
            mp_task_loss_s[mp] = q_loss.data
        mp_att = F.softmax(-torch.stack(list(mp_task_loss_s.values())), dim=0)
        agg_task_fast_weights = self.aggregator(mp_task_fast_weights_s, mp_att)
        agg_mp_emb = torch.stack(query_mp_enhanced_user_emb_s, 1)
        query_agg_enhanced_user_emb = torch.sum(agg_mp_emb * mp_att.unsqueeze(1), 1)
        query_y_pred = self.meta_learner(query_user_emb, query_item_emb, query_agg_enhanced_user_emb, vars_dict=agg_task_fast_weights)
        loss = F.mse_loss(query_y_pred, query_set_y)
        query_y_real = query_set_y.data.cpu().numpy()
        query_y_pred = query_y_pred.data.cpu().numpy()
        mae, rmse = self.cal_metrics.prediction(query_y_real, query_y_pred)
        ndcg_5 = self.cal_metrics.ranking(query_y_real, query_y_pred, k=5)
        return loss, mae, rmse, ndcg_5

    def global_update(self, support_xs, support_ys, support_mps, query_xs, query_ys, query_mps, device='cpu'):
        """ """
        batch_sz = len(support_xs)
        loss_s = []
        mae_s = []
        rmse_s = []
        ndcg_at_5_s = []
        for i in range(batch_sz):
            support_mp = dict(support_mps[i])
            query_mp = dict(query_mps[i])
            for mp in self.mp:
                support_mp[mp] = map(lambda x: x, support_mp[mp])
                query_mp[mp] = map(lambda x: x, query_mp[mp])
            _loss, _mae, _rmse, _ndcg_5 = self.mp_update(support_xs[i], support_ys[i], support_mp, query_xs[i], query_ys[i], query_mp)
            loss_s.append(_loss)
            mae_s.append(_mae)
            rmse_s.append(_rmse)
            ndcg_at_5_s.append(_ndcg_5)
        loss = torch.stack(loss_s).mean(0)
        mae = np.mean(mae_s)
        rmse = np.mean(rmse_s)
        ndcg_at_5 = np.mean(ndcg_at_5_s)
        self.meta_optimizer.zero_grad()
        loss.backward()
        self.meta_optimizer.step()
        return loss.cpu().data.numpy(), mae, rmse, ndcg_at_5

    def evaluation(self, support_x, support_y, support_mp, query_x, query_y, query_mp, device='cpu'):
        """ """
        support_mp = dict(support_mp)
        query_mp = dict(query_mp)
        for mp in self.mp:
            support_mp[mp] = map(lambda x: x, support_mp[mp])
            query_mp[mp] = map(lambda x: x, query_mp[mp])
        _, mae, rmse, ndcg_5 = self.mp_update(support_x, support_y, support_mp, query_x, query_y, query_mp)
        return mae, rmse, ndcg_5

    def aggregator(self, task_weights_s, att):
        for idx, mp in enumerate(self.mp):
            if idx == 0:
                att_task_weights = dict({k: (v * att[idx]) for k, v in task_weights_s[mp].items()})
                continue
            tmp_att_task_weights = dict({k: (v * att[idx]) for k, v in task_weights_s[mp].items()})
            att_task_weights = dict(zip(att_task_weights.keys(), list(map(lambda x: x[0] + x[1], zip(att_task_weights.values(), tmp_att_task_weights.values())))))
        return att_task_weights

    def eval_no_MAML(self, query_set_x, query_set_y, query_set_mps):
        query_mp_enhanced_user_emb_s = []
        query_user_emb = self.user_emb(query_set_x[:, self.config.item_fea_len:])
        query_item_emb = self.item_emb(query_set_x[:, 0:self.config.item_fea_len])
        for mp in self.mp:
            query_set_mp = list(query_set_mps[mp])
            query_neighs_emb = self.item_emb(torch.cat(query_set_mp))
            query_index_list = map(lambda _: _.shape[0], query_set_mp)
            query_mp_enhanced_user_emb = self.mp_learner(query_user_emb, query_item_emb, query_neighs_emb, mp, query_index_list)
            query_mp_enhanced_user_emb_s.append(query_mp_enhanced_user_emb)
        mp_att = torch.FloatTensor([1.0 / len(self.mp)] * len(self.mp))
        agg_mp_emb = torch.stack(query_mp_enhanced_user_emb_s, 1)
        query_agg_enhanced_user_emb = torch.sum(agg_mp_emb * mp_att.unsqueeze(1), 1)
        query_y_pred = self.meta_learner(query_user_emb, query_item_emb, query_agg_enhanced_user_emb)
        query_mae, query_rmse = self.cal_metrics.prediction(query_set_y.data.cpu().numpy(), query_y_pred.data.cpu().numpy())
        query_ndcg_5 = self.cal_metrics.ranking(query_set_y.data.cpu().numpy(), query_y_pred.data.cpu().numpy(), 5)
        return query_mae, query_rmse, query_ndcg_5


class Mg2vec(BaseModel):
    """
    This is a model mg2vec from `mg2vec: Learning Relationship-Preserving Heterogeneous Graph Representations via
    Metagraph Embedding<https://ieeexplore.ieee.org/document/9089251>`__

    It contains following parts:

    Achieve the metagraph and metagraph instances by mining the raw graph. Please go to
    `DataMaker-For-Mg2vec<https://github.com/null-xyj/DataMaker-For-Mg2vec>`__ for more details.

    Initialize the embedding for every node and metagraph and adopt an unsupervised method to train the node embeddings
    and metagraph embeddings. In detail, for every node, we keep its embedding close to the metagraph it belongs to and
    far away from the metagraph we get by negative sampling.

    Every node and meta-graph can be represented as an n-dim vector.We define the first-order loss and second-order
    loss.
    First-Order Loss is for single core node in every meta-graph. We compute the dot product of the node embedding and
    the positive meta-graph embedding as the true logit. Then We compute the dot product of the node embedding and
    the sampled negative meta-graph embedding as the neg logit. We use the binary_cross_entropy_with_logits function to
    compute the first-order loss.
    Second-Order Loss consider two core nodes in every meta-graph. First, we cancat the two node's embedding, what is a
    2n-dim vector. Then we use a 2n*n matrix and an n-dim vector to map the 2n-dim vector to an n-dim vector. The map
    function is showed below:
    .. math::
        f(u,v) = RELU([u||v]W + b)
    u and v means the origin embedding of the two nodes, || is the concatenation operator. W is the 2n*n matrix and b is
    the n-dim vector. RELU is the an activation function. f(u,v) means the n-dim vector after transforming.
    Then, the computation of second-order loss is the same as the first-order loss.
    Finally, we use a parameter alpha to balance the first-order loss and second-order loss.
    .. math::
        L=(1-alpha)*L_1 + alpha*L_2

    After we train the node embeddings, we use the embeddings to complete the relation prediction task.
    The relation prediction task is achieved by edge classification task. If two nodes are connected with a relation, we
    see the relation as an edge. Then we can adopt the edge classification to complete relation prediction task.

    Parameters
    ----------
    node_num: int
        the number of core-nodes
    mg_num: int
        the number of meta-graphs
    emb_dimension: int
        the embedding dimension of nodes and meta-graphs
    unigram: float
        the frequency of every meta-graph, for negative sampling
    sample_num: int
        the number of sampled negative meta-graph

    """

    @classmethod
    def build_model_from_args(cls, args, hg):
        return cls(args.node_num, args.mg_num, args.emb_dimension, args.unigram, args.sample_num)

    def __init__(self, node_num, mg_num, emb_dimension, unigram, sample_num):
        super(Mg2vec, self).__init__()
        self.node_num = node_num
        self.mg_num = mg_num
        self.mg_unigrams = torch.tensor(unigram, dtype=torch.float64)
        self.sample_num = sample_num
        self.emb_dimension = emb_dimension
        self.n_embedding = nn.Embedding(node_num, emb_dimension, sparse=False)
        self.m_embedding = nn.Embedding(mg_num, emb_dimension, sparse=False)
        self.n_w_t = nn.Parameter(torch.empty([emb_dimension * 2, emb_dimension]), requires_grad=True)
        self.n_b = nn.Parameter(torch.empty(emb_dimension), requires_grad=True)
        init.xavier_normal_(self.n_embedding.weight.data)
        init.xavier_normal_(self.m_embedding.weight.data)
        init.xavier_normal_(self.n_w_t)
        init.constant_(self.n_b, 0)

    def forward(self, train_a, train_b, train_labels, train_freq, train_weight, device):
        batch_size = len(train_a)
        n_embed_a = self.n_embedding(train_a)
        n_embed_b = self.n_embedding(train_b)
        n_embed_con = torch.cat([n_embed_a, n_embed_b], dim=1)
        mask_o1 = torch.eq(train_a, train_b).type(torch.FloatTensor).reshape(batch_size, 1)
        mask_o2 = torch.not_equal(train_a, train_b).type(torch.FloatTensor).reshape(batch_size, 1)
        m_embed_pos = self.m_embedding(train_labels)
        neg_sample_id = torch.multinomial(self.mg_unigrams, min(self.mg_num, self.sample_num))
        neg_m_embed = self.m_embedding(neg_sample_id)
        n_embed_o1 = n_embed_a * mask_o1
        n_embed_o2 = F.relu(torch.mm(n_embed_con, self.n_w_t) + self.n_b) * mask_o2
        n_embed = torch.add(n_embed_o1, n_embed_o2)
        true_logit = torch.sum(n_embed * m_embed_pos, dim=1, keepdim=True)
        neg_logit = torch.mm(n_embed, neg_m_embed.T)
        logit = torch.cat([true_logit, neg_logit], dim=1)
        labels = torch.cat([torch.ones_like(true_logit), torch.zeros_like(neg_logit)], dim=1)
        xent = torch.sum(F.binary_cross_entropy_with_logits(logit, labels, reduction='none'), dim=1, keepdim=True)
        unsupervised_loss = torch.mean(train_weight * (train_freq * xent))
        return unsupervised_loss

    def normalize_embedding(self):
        norm = torch.sqrt_(torch.sum(torch.square(self.n_embedding.weight.data), dim=1, keepdim=True))
        self.n_embedding.weight.data = self.n_embedding.weight.data / norm
        m_norm = torch.sqrt_(torch.sum(torch.square(self.m_embedding.weight.data), dim=1, keepdim=True))
        self.m_embedding.weight.data = self.m_embedding.weight.data / m_norm

    def save_embedding(self, id2node, file_name):
        self.normalize_embedding()
        embedding = self.n_embedding.weight.cpu().data.numpy()
        with open(file_name, 'w') as f:
            for nId, node in id2node.items():
                to_write = str(node) + ' ' + ' '.join(map(lambda x: str(x), embedding[nId])) + '\n'
                f.write(to_write)

    def save_embedding_np(self, file_name):
        self.normalize_embedding()
        embedding = self.n_embedding.weight.cpu().data.numpy()
        np.save(file_name, embedding)


class MicroConv(nn.Module):
    """
    Parameters
    ----------
    in_feats : pair of ints
        Input feature size.
    out_feats : int
        Output feature size.
    num_heads : int
        Number of heads in Multi-Head Attention.
    dropout : float, optional
        Dropout rate, defaults: 0.
    negative_slope : float, optional
        Negative slope rate, defaults: 0.2.
    """

    def __init__(self, in_feats: 'tuple', out_feats: 'int', num_heads: 'int', dropout: 'float'=0.0, negative_slope: 'float'=0.2):
        super(MicroConv, self).__init__()
        self._in_src_feats, self._in_dst_feats = in_feats[0], in_feats[1]
        self._out_feats = out_feats
        self._num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(negative_slope)

    def forward(self, graph: 'dgl.DGLHeteroGraph', feat: 'tuple', dst_node_transformation_weight: 'nn.Parameter', src_node_transformation_weight: 'nn.Parameter', src_nodes_attention_weight: 'nn.Parameter'):
        """Compute graph attention network layer.
        Parameters
        ----------
        graph : specific relational DGLHeteroGraph
        feat : pair of torch.Tensor
            The pair contains two tensors of shape (N_{in}, D_{in_{src}})` and (N_{out}, D_{in_{dst}}).
        dst_node_transformation_weight: Parameter (input_dst_dim, n_heads * hidden_dim)
        src_node_transformation_weight: Parameter (input_src_dim, n_heads * hidden_dim)
        src_nodes_attention_weight: Parameter (n_heads, 2 * hidden_dim)
        Returns
        -------
        torch.Tensor, shape (N, H, D_out)` where H is the number of heads, and D_out is size of output feature.
        """
        graph = graph.local_var()
        feat_src = self.dropout(feat[0])
        feat_dst = self.dropout(feat[1])
        feat_src = torch.matmul(feat_src, src_node_transformation_weight).view(-1, self._num_heads, self._out_feats)
        feat_dst = torch.matmul(feat_dst, dst_node_transformation_weight).view(-1, self._num_heads, self._out_feats)
        e_dst = (feat_dst * src_nodes_attention_weight[:, :self._out_feats]).sum(dim=-1, keepdim=True)
        e_src = (feat_src * src_nodes_attention_weight[:, self._out_feats:]).sum(dim=-1, keepdim=True)
        graph.srcdata.update({'ft': feat_src, 'e_src': e_src})
        graph.dstdata.update({'e_dst': e_dst})
        graph.apply_edges(fn.u_add_v('e_src', 'e_dst', 'e'))
        e = self.leaky_relu(graph.edata.pop('e'))
        graph.edata['a'] = edge_softmax(graph, e)
        graph.update_all(fn.u_mul_e('ft', 'a', 'msg'), fn.sum('msg', 'ft'))
        dst_features = graph.dstdata.pop('ft').reshape(-1, self._num_heads * self._out_feats)
        dst_features = F.relu(dst_features)
        return dst_features


class Base_model(nn.Module):

    def __init__(self):
        super(Base_model, self).__init__()

    def Micro_layer(self, h_dict):
        return h_dict

    def Macro_layer(self, h_dict):
        return h_dict

    def forward(self, h_dict):
        h_dict = self.Micro_layer(h_dict)
        h_dict = self.Macro_layer(h_dict)
        return h_dict


class Multi_level(nn.Module):

    def __init__(self):
        super(Multi_level, self).__init__()
        self.micro_layer = None
        self.macro_layer = None

    def forward(self):
        return


class HGConvLayer(nn.Module):

    def __init__(self, graph: 'dgl.DGLHeteroGraph', input_dim: 'int', hidden_dim: 'int', n_heads: 'int'=4, dropout: 'float'=0.2, residual: 'bool'=True):
        """
        :param graph: a heterogeneous graph
        :param input_dim: int, input dimension
        :param hidden_dim: int, hidden dimension
        :param n_heads: int, number of attention heads
        :param dropout: float, dropout rate
        :param residual: boolean, residual connections or not
        """
        super(HGConvLayer, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.dropout = dropout
        self.residual = residual
        self.micro_conv = dglnn.HeteroGraphConv({etype: LSTMConv(dim=input_dim) for srctype, etype, dsttype in graph.canonical_etypes})
        self.macro_conv = MacroConv(in_feats=hidden_dim * n_heads, out_feats=hidden_dim, num_heads=n_heads, dropout=dropout, negative_slope=0.2)
        if self.residual:
            self.res_fc = nn.ModuleDict()
            self.residual_weight = nn.ParameterDict()
            for ntype in graph.ntypes:
                self.res_fc[ntype] = nn.Linear(input_dim, n_heads * hidden_dim, bias=True)
                self.residual_weight[ntype] = nn.Parameter(torch.randn(1))
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')
        for weight in self.node_transformation_weight:
            nn.init.xavier_normal_(self.node_transformation_weight[weight], gain=gain)
        for weight in self.nodes_attention_weight:
            nn.init.xavier_normal_(self.nodes_attention_weight[weight], gain=gain)
        for weight in self.edge_type_transformation_weight:
            nn.init.xavier_normal_(self.edge_type_transformation_weight[weight], gain=gain)
        for weight in self.central_node_transformation_weight:
            nn.init.xavier_normal_(self.central_node_transformation_weight[weight], gain=gain)
        nn.init.xavier_normal_(self.edge_types_attention_weight, gain=gain)
        if self.residual:
            for ntype in self.res_fc:
                nn.init.xavier_normal_(self.res_fc[ntype].weight, gain=gain)

    def forward(self, graph: 'dgl.DGLHeteroGraph', node_features: 'dict'):
        """
        :param graph: dgl.DGLHeteroGraph
        :param node_features: dict, {"type": features}
        :return: output_features: dict, {"type": features}
        """
        input_src = node_features
        if graph.is_block:
            input_dst = {}
            for ntype in node_features:
                input_dst[ntype] = node_features[ntype][:graph.number_of_dst_nodes(ntype)]
        else:
            input_dst = node_features
        relation_features = self.micro_conv(graph, input_src, input_dst, self.node_transformation_weight, self.nodes_attention_weight)
        output_features = self.macro_conv(graph, input_dst, relation_features, self.edge_type_transformation_weight, self.central_node_transformation_weight, self.edge_types_attention_weight)
        if self.residual:
            for ntype in output_features:
                alpha = F.sigmoid(self.residual_weight[ntype])
                output_features[ntype] = output_features[ntype] * alpha + self.res_fc[ntype](input_dst[ntype]) * (1 - alpha)
        return output_features


class FeedForwardNet(nn.Module):
    """
        A feedforward net.

        Input
        ------
        in_feats :
            input feature dimention
        hidden :
            hidden layer dimention
        out_feats :
            output feature dimention
        num_layers :
            number of layers
        dropout :
            dropout rate
    """

    def __init__(self, in_feats, hidden, out_feats, num_layers, dropout):
        super(FeedForwardNet, self).__init__()
        self.layers = nn.ModuleList()
        self.num_layers = num_layers
        if num_layers == 1:
            self.layers.append(nn.Linear(in_feats, out_feats))
        else:
            self.layers.append(nn.Linear(in_feats, hidden))
            for i in range(num_layers - 2):
                self.layers.append(nn.Linear(hidden, hidden))
            self.layers.append(nn.Linear(hidden, out_feats))
        if self.num_layers > 1:
            self.prelu = nn.PReLU()
            self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        for layer in self.layers:
            nn.init.xavier_uniform_(layer.weight, gain=gain)
            nn.init.zeros_(layer.bias)

    def forward(self, x):
        for layer_id, layer in enumerate(self.layers):
            x = layer(x)
            if layer_id < self.num_layers - 1:
                x = self.dropout(self.prelu(x))
        return x


class SIGN(nn.Module):
    """
        The SIGN model.

        Parameters
        ------------
        in_feats :
            input feature dimention
        hidden :
            hidden layer dimention
        out_feats :
            output feature dimention
        num_hops :
            number of hops
        num_layers :
            number of layers
        dropout :
            dropout rate
        input_drop :
            whether or not to dropout when inputting features

    """

    def __init__(self, in_feats, hidden, out_feats, num_hops, num_layers, dropout, input_drop):
        super(SIGN, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.prelu = nn.PReLU()
        self.inception_ffs = nn.ModuleList()
        self.input_drop = input_drop
        for i in range(num_hops):
            self.inception_ffs.append(FeedForwardNet(in_feats, hidden, hidden, num_layers, dropout))
        self.project = FeedForwardNet(num_hops * hidden, hidden, out_feats, num_layers, dropout)

    def forward(self, feats):
        hidden = []
        for feat, ff in zip(feats, self.inception_ffs):
            if self.input_drop:
                feat = self.dropout(feat)
            hidden.append(ff(feat))
        out = self.project(self.dropout(self.prelu(th.cat(hidden, dim=-1))))
        return out


class WeightedAggregator(nn.Module):
    """
        Get new features by multiplying the old features by the weight matrix.

        Parameters
        -------------
        num_feats :
            number of subsets
        in_feats :
            input feature dimention
        num_hops :
            number of hops


    """

    def __init__(self, num_feats, in_feats, num_hops):
        super(WeightedAggregator, self).__init__()
        self.agg_feats = nn.ParameterList()
        for _ in range(num_hops):
            self.agg_feats.append(nn.Parameter(th.Tensor(num_feats, in_feats)))
            nn.init.xavier_uniform_(self.agg_feats[-1])

    def forward(self, feats):
        new_feats = []
        for feat, weight in zip(feats, self.agg_feats):
            new_feats.append((feat * weight.unsqueeze(0)).sum(dim=1).squeeze())
        return new_feats


def gen_rel_subset_feature(g, rel_subset, args, device, predict):
    """
        Build relation subgraph given relation subset and generate multi-hop
        neighbor-averaged feature on this subgraph

        Parameters
        ----------
        g :
            Heterogeneous graph
        rel_subset :
            relation of subsets
        args :
            arguments
        device :
            device

        Returns
        ------
        new features of a relation subsets
    """
    new_g = g.edge_type_subgraph(rel_subset)
    ntypes = new_g.ntypes
    for ntype in ntypes:
        num_nodes = new_g.number_of_nodes(ntype)
        if num_nodes < g.nodes[ntype].data['feat'].shape[0]:
            new_g.nodes[ntype].data['hop_0'] = g.nodes[ntype].data['feat'][:num_nodes, :]
        else:
            new_g.nodes[ntype].data['hop_0'] = g.nodes[ntype].data['feat']
        deg = 0
        for etype in new_g.etypes:
            _, _, dtype = new_g.to_canonical_etype(etype)
            if ntype == dtype:
                deg = deg + new_g.in_degrees(etype=etype)
        if th.is_tensor(deg):
            norm = 1.0 / deg.float()
            norm[th.isinf(norm)] = 0
            new_g.nodes[ntype].data['norm'] = norm.view(-1, 1)
    res = []
    for hop in range(1, args.num_hops + 1):
        ntype2feat = {}
        for etype in new_g.etypes:
            stype, _, dtype = new_g.to_canonical_etype(etype)
            new_g[etype].update_all(fn.copy_u(f'hop_{hop - 1}', 'm'), fn.sum('m', 'new_feat'))
            new_feat = new_g.nodes[dtype].data.pop('new_feat')
            assert 'new_feat' not in new_g.nodes[stype].data
            if dtype in ntype2feat:
                ntype2feat[dtype] += new_feat
            else:
                ntype2feat[dtype] = new_feat
        for ntype in new_g.ntypes:
            assert ntype in ntype2feat
            feat_dict = new_g.nodes[ntype].data
            old_feat = feat_dict.pop(f'hop_{hop - 1}')
            if ntype == predict:
                res.append(old_feat.cpu())
            feat_dict[f'hop_{hop}'] = ntype2feat.pop(ntype).mul_(feat_dict['norm'])
    res.append(new_g.nodes[predict].data.pop(f'hop_{args.num_hops}').cpu())
    return res


def preprocess_features(g, mps, args, device, predict):
    """
        pre-process heterogeneous graph g to generate neighbor-averaged features
        for each relation subsets

        Parameters
        -----------
        g :
            heterogeneous graph
        rel_subsets :
            relations of subsets
        args :
            arguments
        device :
            device

        Return
        ------
            new features of each relation subsets

    """
    category_dim = g.nodes[predict].data['feat'].shape[1]
    for ntype in g.ntypes:
        ntype_dim = g.nodes[ntype].data['feat'].shape[1]
        if category_dim != ntype_dim:
            rand_weight = th.Tensor(ntype_dim, category_dim).uniform_(-0.5, 0.5)
            g.nodes[ntype].data['feat'] = th.matmul(g.nodes[ntype].data['feat'], rand_weight)
    num_paper, feat_size = g.nodes[predict].data['feat'].shape
    new_feats = [th.zeros(num_paper, len(mps), feat_size) for _ in range(args.num_hops + 1)]
    for subset_id, subset in enumerate(mps):
        feats = gen_rel_subset_feature(g, subset, args, device, predict)
        for i in range(args.num_hops + 1):
            feat = feats[i]
            new_feats[i][:feat.shape[0], subset_id, :] = feat
        feats = None
    return new_feats


class NARS(BaseModel):
    """
        `SCALABLE GRAPH NEURAL NETWORKS FOR HETEROGENEOUS GRAPHS <https://arxiv.org/pdf/2011.09679.pdf>`_.

        Given a heterogeneous graph :math:`G` and its edge relation type set :math:`\\mathcal{R}`,
        our proposed method first samples :math:`K` unique subsets from :math:`\\mathcal{R}`.
        Then for each sampled subset :math:`R_i \\subseteq \\mathcal{R}`, we generate a relation subgraph
        :math:`G_i` from :math:`G` in which only edges whose type belongs to :math:`R_i` are kept.
        We treat :math:`G_i` as a homogeneous graph or a bipartite graph, and perform neighbor aggregation to generate :math:`L`-hop neighbor features for each node.
        Let :math:`H_{v,0}` be the input features (of dimension :math:`D`) for node :math:`v`. For each subgraph :math:`G_i`
        , the :math:`l`-th hop
        features :math:`H_{v,l}^{i}` are computed as

        .. math::
            H_{v, l}^{i}=\\sum_{u \\in N_{i}(v)} \\frac{1}{\\left|N_{i}(v)\\right|} H_{u, l-1}^{i}


        where :math:`N_i(v)` is the set of neighbors of node :math:`v` in :math:`G_i`.

        For each layer :math:`l`, we let the model adaptively learn which relation-subgraph features to use by aggregating
        features from different subgraphs :math:`G_i` with learnable 1-D convolution. The aggregated :math:`l`-hop
        features across all subgraphs are calculated as


        .. math::
            H_{v, l}^{a g g}=\\sum_{i=1}^{K} a_{i, l} \\cdot H_{v, l}^{i}


        where :math:`H^i` is the neighbor averaging features on subgraph :math:`G_i` and :math:`a_{i,l}` is a learned vector of length equal
        to the feature dimension :math:`D`.

        Parameters
        ----------
        num_hops : int
            Number of hops.
        category : str
            Type of predicted nodes.
        hidden_dim : int
            The dimention of hidden layer.
        num_feats : int
            The number of relation subsets.

        Note
        ----
        We do not support the dataset without feature, (e.g. HGBn-Freebase
        because the model performs neighbor aggregation to generate :math:`L`-hop neighbor features at once.

        """

    @classmethod
    def build_model_from_args(cls, args, hg):
        num_hops = args.num_hops + 1
        return cls(num_hops=num_hops, args=args, hg=hg)

    def __init__(self, num_hops, args, hg):
        super(NARS, self).__init__()
        self.category = args.category
        self.dropout = args.dropout
        self.input_dropout = args.input_dropout
        self.device = args.device
        self.num_hops = num_hops
        self.args = args
        in_size = hg.nodes[args.category].data['h'].shape[1]
        etypes = hg.canonical_etypes
        mps = []
        for etype in etypes:
            if etype[0] == args.category:
                for dst_e in etypes:
                    if etype[0] == dst_e[2] and etype[2] == dst_e[0] and etype[0] != etype[2]:
                        mps.append([etype, dst_e])
        self.mps = mps
        self.num_feats = len(mps)
        with th.no_grad():
            self.feats = preprocess_features(hg, mps, args, args.device, self.args.category)
            None
        self.seq = nn.Sequential(WeightedAggregator(self.num_feats, in_size, num_hops), SIGN(in_size, args.hidden_dim, args.out_dim, num_hops, args.ff_layer, args.dropout, args.input_dropout))

    def forward(self, hg, h_dict):
        ffeats = [x for x in self.feats]
        return {self.category: self.seq.forward(ffeats)}

    def reset_parameters(self):
        self.seq.register_parameter()


def is_torch_sparse_tensor(src: 'Any') ->bool:
    if isinstance(src, Tensor):
        if src.layout == torch.sparse_coo:
            return True
        if src.layout == torch.sparse_csr:
            return True
        if src.layout == torch.sparse_csc:
            return True
    return False


def maybe_num_nodes(edge_index, num_nodes=None):
    if num_nodes is not None:
        return num_nodes
    elif isinstance(edge_index, Tensor):
        if is_torch_sparse_tensor(edge_index):
            return max(edge_index.size(0), edge_index.size(1))
        return int(edge_index.max()) + 1 if edge_index.numel() > 0 else 0
    else:
        return max(edge_index.size(0), edge_index.size(1))


def degree(index: 'Tensor', num_nodes: 'Optional[int]'=None, dtype: 'Optional[torch.dtype]'=None) ->Tensor:
    N = maybe_num_nodes(index, num_nodes)
    out = torch.zeros((N,), dtype=dtype, device=index.device)
    one = torch.ones((index.size(0),), dtype=out.dtype, device=out.device)
    return out.scatter_add_(0, index, one)


def scatter_max(src: 'torch.Tensor', index: 'torch.Tensor', dim: 'int'=-1, out: 'Optional[torch.Tensor]'=None, dim_size: 'Optional[int]'=None) ->Tuple[torch.Tensor, torch.Tensor]:
    return torch.ops.torch_scatter.scatter_max(src, index, dim, out, dim_size)


def scatter_mean(src: 'torch.Tensor', index: 'torch.Tensor', dim: 'int'=-1, out: 'Optional[torch.Tensor]'=None, dim_size: 'Optional[int]'=None) ->torch.Tensor:
    out = scatter_sum(src, index, dim, out, dim_size)
    dim_size = out.size(dim)
    index_dim = dim
    if index_dim < 0:
        index_dim = index_dim + src.dim()
    if index.dim() <= index_dim:
        index_dim = index.dim() - 1
    ones = torch.ones(index.size(), dtype=src.dtype, device=src.device)
    count = scatter_sum(ones, index, index_dim, None, dim_size)
    count[count < 1] = 1
    count = broadcast(count, out, dim)
    if out.is_floating_point():
        out.true_divide_(count)
    else:
        out.div_(count, rounding_mode='floor')
    return out


def scatter_min(src: 'torch.Tensor', index: 'torch.Tensor', dim: 'int'=-1, out: 'Optional[torch.Tensor]'=None, dim_size: 'Optional[int]'=None) ->Tuple[torch.Tensor, torch.Tensor]:
    return torch.ops.torch_scatter.scatter_min(src, index, dim, out, dim_size)


def scatter_mul(src: 'torch.Tensor', index: 'torch.Tensor', dim: 'int'=-1, out: 'Optional[torch.Tensor]'=None, dim_size: 'Optional[int]'=None) ->torch.Tensor:
    return torch.ops.torch_scatter.scatter_mul(src, index, dim, out, dim_size)


def scatter(src: 'torch.Tensor', index: 'torch.Tensor', dim: 'int'=-1, out: 'Optional[torch.Tensor]'=None, dim_size: 'Optional[int]'=None, reduce: 'str'='sum') ->torch.Tensor:
    if reduce == 'sum' or reduce == 'add':
        return scatter_sum(src, index, dim, out, dim_size)
    if reduce == 'mul':
        return scatter_mul(src, index, dim, out, dim_size)
    elif reduce == 'mean':
        return scatter_mean(src, index, dim, out, dim_size)
    elif reduce == 'min':
        return scatter_min(src, index, dim, out, dim_size)[0]
    elif reduce == 'max':
        return scatter_max(src, index, dim, out, dim_size)[0]
    else:
        raise ValueError


class GeneralizedRelationalConv(torch.nn.Module):
    eps = 1e-06
    message2mul = {'transe': 'add', 'distmult': 'mul'}

    def __init__(self, input_dim, output_dim, num_relation, query_input_dim, message_func='distmult', aggregate_func='pna', layer_norm=False, activation='relu', dependent=True):
        super(GeneralizedRelationalConv, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_relation = num_relation
        self.query_input_dim = query_input_dim
        self.message_func = message_func
        self.aggregate_func = aggregate_func
        self.dependent = dependent
        self.node_dim = -2
        if layer_norm:
            self.layer_norm = nn.LayerNorm(output_dim)
        else:
            self.layer_norm = None
        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = activation
        if self.aggregate_func == 'pna':
            self.linear = nn.Linear(input_dim * 13, output_dim)
        else:
            self.linear = nn.Linear(input_dim * 2, output_dim)
        if dependent:
            self.relation_linear = nn.Linear(query_input_dim, num_relation * input_dim)
        else:
            self.relation = nn.Embedding(num_relation, input_dim)

    def forward(self, input, query, boundary, edge_index, edge_type, size, edge_weight=None):
        batch_size = len(query)
        if self.dependent:
            relation = self.relation_linear(query).view(batch_size, self.num_relation, self.input_dim)
        else:
            relation = self.relation.weight.expand(batch_size, -1, -1)
        if edge_weight is None:
            edge_weight = torch.ones(len(edge_type), device=input.device)
        input_j = input.index_select(1, edge_index[0])
        message_res = self.message(input_j=input_j, relation=relation, boundary=boundary, edge_type=edge_type)
        aggr_res = self.aggregate(input=message_res, edge_weight=edge_weight, index=edge_index[1], dim_size=input.shape[1])
        return self.update(update=aggr_res, input=input)

    def message(self, input_j, relation, boundary, edge_type):
        relation_j = relation.index_select(self.node_dim, edge_type)
        if self.message_func == 'transe':
            message = input_j + relation_j
        elif self.message_func == 'distmult':
            message = input_j * relation_j
        elif self.message_func == 'rotate':
            x_j_re, x_j_im = input_j.chunk(2, dim=-1)
            r_j_re, r_j_im = relation_j.chunk(2, dim=-1)
            message_re = x_j_re * r_j_re - x_j_im * r_j_im
            message_im = x_j_re * r_j_im + x_j_im * r_j_re
            message = torch.cat([message_re, message_im], dim=-1)
        else:
            raise ValueError('Unknown message function `%s`' % self.message_func)
        message = torch.cat([message, boundary], dim=self.node_dim)
        return message

    def aggregate(self, input, edge_weight, index, dim_size):
        index = torch.cat([index, torch.arange(dim_size, device=input.device)])
        edge_weight = torch.cat([edge_weight, torch.ones(dim_size, device=input.device)])
        shape = [1] * input.ndim
        shape[self.node_dim] = -1
        edge_weight = edge_weight.view(shape)
        if self.aggregate_func == 'pna':
            mean = scatter_mean(input * edge_weight, index, dim=self.node_dim, dim_size=dim_size)
            sq_mean = scatter_mean(input ** 2 * edge_weight, index, dim=self.node_dim, dim_size=dim_size)
            max = scatter(input * edge_weight, index, dim=self.node_dim, dim_size=dim_size, reduce='max')
            min = scatter(input * edge_weight, index, dim=self.node_dim, dim_size=dim_size, reduce='min')
            std = (sq_mean - mean ** 2).clamp(min=self.eps).sqrt()
            features = torch.cat([mean.unsqueeze(-1), max.unsqueeze(-1), min.unsqueeze(-1), std.unsqueeze(-1)], dim=-1)
            features = features.flatten(-2)
            degree_out = degree(index, dim_size).unsqueeze(0).unsqueeze(-1)
            scale = degree_out.log()
            scale = scale / scale.mean()
            scales = torch.cat([torch.ones_like(scale), scale, 1 / scale.clamp(min=0.01)], dim=-1)
            output = (features.unsqueeze(-1) * scales.unsqueeze(-2)).flatten(-2)
        return output

    def update(self, update, input):
        output = self.linear(torch.cat([input, update], dim=-1))
        if self.layer_norm:
            output = self.layer_norm(output)
        if self.activation:
            output = self.activation(output)
        return output

    def message_and_aggregate(self, edge_index, input, relation, boundary, edge_type, edge_weight, index, dim_size):
        batch_size, num_node = input.shape[:2]
        input = input.transpose(0, 1).flatten(1)
        relation = relation.transpose(0, 1).flatten(1)
        boundary = boundary.transpose(0, 1).flatten(1)
        degree_out = degree(index, dim_size).unsqueeze(-1) + 1
        if self.message_func in self.message2mul:
            mul = self.message2mul[self.message_func]
        else:
            raise ValueError('Unknown message function `%s`' % self.message_func)
        if self.aggregate_func == 'sum':
            update = generalized_rspmm(edge_index, edge_type, edge_weight, relation, input, sum='add', mul=mul)
            update = update + boundary
        elif self.aggregate_func == 'mean':
            update = generalized_rspmm(edge_index, edge_type, edge_weight, relation, input, sum='add', mul=mul)
            update = (update + boundary) / degree_out
        elif self.aggregate_func == 'max':
            update = generalized_rspmm(edge_index, edge_type, edge_weight, relation, input, sum='max', mul=mul)
            update = torch.max(update, boundary)
        elif self.aggregate_func == 'pna':
            sum = generalized_rspmm(edge_index, edge_type, edge_weight, relation, input, sum='add', mul=mul)
            sq_sum = generalized_rspmm(edge_index, edge_type, edge_weight, relation ** 2, input ** 2, sum='add', mul=mul)
            max = generalized_rspmm(edge_index, edge_type, edge_weight, relation, input, sum='max', mul=mul)
            min = generalized_rspmm(edge_index, edge_type, edge_weight, relation, input, sum='min', mul=mul)
            mean = (sum + boundary) / degree_out
            sq_mean = (sq_sum + boundary ** 2) / degree_out
            max = torch.max(max, boundary)
            min = torch.min(min, boundary)
            std = (sq_mean - mean ** 2).clamp(min=self.eps).sqrt()
            features = torch.cat([mean.unsqueeze(-1), max.unsqueeze(-1), min.unsqueeze(-1), std.unsqueeze(-1)], dim=-1)
            features = features.flatten(-2)
            scale = degree_out.log()
            scale = scale / scale.mean()
            scales = torch.cat([torch.ones_like(scale), scale, 1 / scale.clamp(min=0.01)], dim=-1)
            update = (features.unsqueeze(-1) * scales.unsqueeze(-2)).flatten(-2)
        else:
            raise ValueError('Unknown aggregation function `%s`' % self.aggregate_func)
        update = update.view(num_node, batch_size, -1).transpose(0, 1)
        return update

    def propagate(self, edge_index, size=None, **kwargs):
        if kwargs['edge_weight'].requires_grad or self.message_func == 'rotate':
            return super(GeneralizedRelationalConv, self).propagate(edge_index, size, **kwargs)
        for hook in self._propagate_forward_pre_hooks.values():
            res = hook(self, (edge_index, size, kwargs))
            if res is not None:
                edge_index, size, kwargs = res
        size = self._check_input(edge_index, size)
        coll_dict = self._collect(self._fused_user_args, edge_index, size, kwargs)
        msg_aggr_kwargs = self.inspector.distribute('message_and_aggregate', coll_dict)
        for hook in self._message_and_aggregate_forward_pre_hooks.values():
            res = hook(self, (edge_index, msg_aggr_kwargs))
            if res is not None:
                edge_index, msg_aggr_kwargs = res
        out = self.message_and_aggregate(edge_index, **msg_aggr_kwargs)
        for hook in self._message_and_aggregate_forward_hooks.values():
            res = hook(self, (edge_index, msg_aggr_kwargs), out)
            if res is not None:
                out = res
        update_kwargs = self.inspector.distribute('update', coll_dict)
        out = self.update(out, **update_kwargs)
        for hook in self._propagate_forward_hooks.values():
            res = hook(self, (edge_index, size, kwargs), out)
            if res is not None:
                out = res
        return out


def edge_match(edge_index, query_index):
    base = edge_index.max(dim=1)[0] + 1
    assert reduce(int.__mul__, base.tolist()) < torch.iinfo(torch.long).max
    scale = base.cumprod(0)
    scale = scale[-1] // scale
    edge_hash = (edge_index * scale.unsqueeze(-1)).sum(dim=0)
    edge_hash, order = edge_hash.sort()
    query_hash = (query_index * scale.unsqueeze(-1)).sum(dim=0)
    start = torch.bucketize(query_hash, edge_hash)
    end = torch.bucketize(query_hash, edge_hash, right=True)
    num_match = end - start
    offset = num_match.cumsum(0) - num_match
    range = torch.arange(num_match.sum(), device=edge_index.device)
    range = range + (start - offset).repeat_interleave(num_match)
    return order[range], num_match


def index_to_mask(index, size):
    index = index.view(-1)
    size = int(index.max()) + 1 if size is None else size
    mask = index.new_zeros(size, dtype=torch.bool)
    mask[index] = True
    return mask


def multi_slice_mask(starts, ends, length):
    values = torch.cat([torch.ones_like(starts), -torch.ones_like(ends)])
    slices = torch.cat([starts, ends])
    mask = scatter_sum(values, slices, dim=0, dim_size=length + 1)[:-1]
    mask = mask.cumsum(0).bool()
    return mask


def scatter_extend(data, size, input, input_size):
    new_size = size + input_size
    new_cum_size = new_size.cumsum(0)
    new_data = torch.zeros(new_cum_size[-1], *data.shape[1:], dtype=data.dtype, device=data.device)
    starts = new_cum_size - new_size
    ends = starts + size
    index = multi_slice_mask(starts, ends, new_cum_size[-1])
    new_data[index] = data
    new_data[~index] = input
    return new_data, new_size


def size_to_index(size):
    range = torch.arange(len(size), device=size.device)
    index2sample = range.repeat_interleave(size)
    return index2sample


def scatter_topk(input, size, k, largest=True):
    index2graph = size_to_index(size)
    index2graph = index2graph.view([-1] + [1] * (input.ndim - 1))
    mask = ~torch.isinf(input)
    max = input[mask].max().item()
    min = input[mask].min().item()
    safe_input = input.clamp(2 * min - max, 2 * max - min)
    offset = (max - min) * 4
    if largest:
        offset = -offset
    input_ext = safe_input + offset * index2graph
    index_ext = input_ext.argsort(dim=0, descending=largest)
    num_actual = size.clamp(max=k)
    num_padding = k - num_actual
    starts = size.cumsum(0) - size
    ends = starts + num_actual
    mask = multi_slice_mask(starts, ends, len(index_ext)).nonzero().flatten()
    if (num_padding > 0).any():
        padding = ends - 1
        padding2graph = size_to_index(num_padding)
        mask = scatter_extend(mask, num_actual, padding[padding2graph], num_padding)[0]
    index = index_ext[mask]
    value = input.gather(0, index)
    if isinstance(k, torch.Tensor) and k.shape == size.shape:
        value = value.view(-1, *input.shape[1:])
        index = index.view(-1, *input.shape[1:])
        index = index - (size.cumsum(0) - size).repeat_interleave(k).view([-1] + [1] * (index.ndim - 1))
    else:
        value = value.view(-1, k, *input.shape[1:])
        index = index.view(-1, k, *input.shape[1:])
        index = index - (size.cumsum(0) - size).view([-1] + [1] * (index.ndim - 1))
    return value, index


class NBFNet(BaseModel):

    @classmethod
    def build_model_from_args(cls, args, hg):
        return cls(input_dim=args.input_dim, hidden_dims=args.hidden_dims, num_relation=args.num_relation, message_func=args.message_func, aggregate_func=args.aggregate_func, short_cut=args.short_cut, layer_norm=args.layer_norm, dependent=args.dependent)

    def __init__(self, input_dim, hidden_dims, num_relation, message_func='distmult', aggregate_func='pna', short_cut=False, layer_norm=False, activation='relu', concat_hidden=False, num_mlp_layer=2, dependent=True, remove_one_hop=False, num_beam=10, path_topk=10):
        super(NBFNet, self).__init__()
        if not isinstance(hidden_dims, Sequence):
            hidden_dims = [hidden_dims]
        self.dims = [input_dim] + list(hidden_dims)
        self.num_relation = num_relation
        self.short_cut = short_cut
        self.concat_hidden = concat_hidden
        self.remove_one_hop = remove_one_hop
        self.num_beam = num_beam
        self.path_topk = path_topk
        self.layers = nn.ModuleList()
        for i in range(len(self.dims) - 1):
            self.layers.append(GeneralizedRelationalConv(self.dims[i], self.dims[i + 1], num_relation, self.dims[0], message_func, aggregate_func, layer_norm, activation, dependent))
        feature_dim = (sum(hidden_dims) if concat_hidden else hidden_dims[-1]) + input_dim
        self.query = nn.Embedding(num_relation, input_dim)
        self.mlp = nn.Sequential()
        mlp = []
        for i in range(num_mlp_layer - 1):
            mlp.append(nn.Linear(feature_dim, feature_dim))
            mlp.append(nn.ReLU())
        mlp.append(nn.Linear(feature_dim, 1))
        self.mlp = nn.Sequential(*mlp)

    def remove_easy_edges(self, data, h_index, t_index, r_index=None):
        h_index_ext = torch.cat([h_index, t_index], dim=-1)
        t_index_ext = torch.cat([t_index, h_index], dim=-1)
        r_index_ext = torch.cat([r_index, r_index + self.num_relation // 2], dim=-1)
        if self.remove_one_hop:
            edge_index = data.edge_index
            easy_edge = torch.stack([h_index_ext, t_index_ext]).flatten(1)
            index = edge_match(edge_index, easy_edge)[0]
            mask = ~index_to_mask(index, data.num_edges)
        else:
            edge_index = torch.cat([data.edge_index, data.edge_type.unsqueeze(0)])
            easy_edge = torch.stack([h_index_ext, t_index_ext, r_index_ext]).flatten(1)
            index = edge_match(edge_index, easy_edge)[0]
            mask = ~index_to_mask(index, data.num_edges)
        data = copy.copy(data)
        data.edge_index = data.edge_index[:, mask]
        data.edge_type = data.edge_type[mask]
        return data

    def negative_sample_to_tail(self, h_index, t_index, r_index):
        is_t_neg = (h_index == h_index[:, [0]]).all(dim=-1, keepdim=True)
        new_h_index = torch.where(is_t_neg, h_index, t_index)
        new_t_index = torch.where(is_t_neg, t_index, h_index)
        new_r_index = torch.where(is_t_neg, r_index, r_index + self.num_relation // 2)
        return new_h_index, new_t_index, new_r_index

    def bellmanford(self, data, h_index, r_index, separate_grad=False):
        batch_size = len(r_index)
        query = self.query(r_index)
        index = h_index.unsqueeze(-1).expand_as(query)
        boundary = torch.zeros(batch_size, data.num_nodes, self.dims[0], device=h_index.device)
        boundary.scatter_add_(1, index.unsqueeze(1), query.unsqueeze(1))
        size = data.num_nodes, data.num_nodes
        edge_weight = torch.ones(data.num_edges, device=h_index.device)
        hiddens = []
        edge_weights = []
        layer_input = boundary
        for layer in self.layers:
            if separate_grad:
                edge_weight = edge_weight.clone().requires_grad_()
            hidden = layer(layer_input, query, boundary, data.edge_index, data.edge_type, size, edge_weight)
            if self.short_cut and hidden.shape == layer_input.shape:
                hidden = hidden + layer_input
            hiddens.append(hidden)
            edge_weights.append(edge_weight)
            layer_input = hidden
        node_query = query.unsqueeze(1).expand(-1, data.num_nodes, -1)
        if self.concat_hidden:
            output = torch.cat(hiddens + [node_query], dim=-1)
        else:
            output = torch.cat([hiddens[-1], node_query], dim=-1)
        return {'node_feature': output, 'edge_weights': edge_weights}

    def forward(self, data, batch):
        h_index, t_index, r_index = batch.unbind(-1)
        if self.training:
            data = self.remove_easy_edges(data, h_index, t_index, r_index)
        data.num_edges = data.edge_index.shape[1]
        shape = h_index.shape
        h_index, t_index, r_index = self.negative_sample_to_tail(h_index, t_index, r_index)
        assert (h_index[:, [0]] == h_index).all()
        assert (r_index[:, [0]] == r_index).all()
        output = self.bellmanford(data, h_index[:, 0], r_index[:, 0])
        feature = output['node_feature']
        index = t_index.unsqueeze(-1).expand(-1, -1, feature.shape[-1])
        feature = feature.gather(1, index)
        score = self.mlp(feature).squeeze(-1)
        return score.view(shape)

    def visualize(self, data, batch):
        assert batch.shape == (1, 3)
        h_index, t_index, r_index = batch.unbind(-1)
        output = self.bellmanford(data, h_index, r_index, separate_grad=True)
        feature = output['node_feature']
        edge_weights = output['edge_weights']
        index = t_index.unsqueeze(0).unsqueeze(-1).expand(-1, -1, feature.shape[-1])
        feature = feature.gather(1, index).squeeze(0)
        score = self.mlp(feature).squeeze(-1)
        edge_grads = autograd.grad(score, edge_weights)
        distances, back_edges = self.beam_search_distance(data, edge_grads, h_index, t_index, self.num_beam)
        paths, weights = self.topk_average_length(distances, back_edges, t_index, self.path_topk)
        return paths, weights

    @torch.no_grad()
    def beam_search_distance(self, data, edge_grads, h_index, t_index, num_beam=10):
        num_nodes = data.num_nodes
        input = torch.full((num_nodes, num_beam), float('-inf'), device=h_index.device)
        input[h_index, 0] = 0
        edge_mask = data.edge_index[0, :] != t_index
        distances = []
        back_edges = []
        for edge_grad in edge_grads:
            node_in, node_out = data.edge_index[:, edge_mask]
            relation = data.edge_type[edge_mask]
            edge_grad = edge_grad[edge_mask]
            message = input[node_in] + edge_grad.unsqueeze(-1)
            msg_source = torch.stack([node_in, node_out, relation], dim=-1).unsqueeze(1).expand(-1, num_beam, -1)
            is_duplicate = torch.isclose(message.unsqueeze(-1), message.unsqueeze(-2)) & (msg_source.unsqueeze(-2) == msg_source.unsqueeze(-3)).all(dim=-1)
            is_duplicate = is_duplicate.float() - torch.arange(num_beam, dtype=torch.float, device=message.device) / (num_beam + 1)
            prev_rank = is_duplicate.argmax(dim=-1, keepdim=True)
            msg_source = torch.cat([msg_source, prev_rank], dim=-1)
            node_out, order = node_out.sort()
            node_out_set = torch.unique(node_out)
            message = message[order].flatten()
            msg_source = msg_source[order].flatten(0, -2)
            size = node_out.bincount(minlength=num_nodes)
            msg2out = size_to_index(size[node_out_set] * num_beam)
            is_duplicate = (msg_source[1:] == msg_source[:-1]).all(dim=-1)
            is_duplicate = torch.cat([torch.zeros(1, dtype=torch.bool, device=message.device), is_duplicate])
            message = message[~is_duplicate]
            msg_source = msg_source[~is_duplicate]
            msg2out = msg2out[~is_duplicate]
            size = msg2out.bincount(minlength=len(node_out_set))
            if not torch.isinf(message).all():
                distance, rel_index = scatter_topk(message, size, k=num_beam)
                abs_index = rel_index + (size.cumsum(0) - size).unsqueeze(-1)
                back_edge = msg_source[abs_index]
                distance = distance.view(len(node_out_set), num_beam)
                back_edge = back_edge.view(len(node_out_set), num_beam, 4)
                distance = scatter_sum(distance, node_out_set, dim=0, dim_size=num_nodes)
                back_edge = scatter_sum(back_edge, node_out_set, dim=0, dim_size=num_nodes)
            else:
                distance = torch.full((num_nodes, num_beam), float('-inf'), device=message.device)
                back_edge = torch.zeros(num_nodes, num_beam, 4, dtype=torch.long, device=message.device)
            distances.append(distance)
            back_edges.append(back_edge)
            input = distance
        return distances, back_edges

    def topk_average_length(self, distances, back_edges, t_index, k=10):
        paths = []
        average_lengths = []
        for i in range(len(distances)):
            distance, order = distances[i][t_index].flatten(0, -1).sort(descending=True)
            back_edge = back_edges[i][t_index].flatten(0, -2)[order]
            for d, (h, t, r, prev_rank) in zip(distance[:k].tolist(), back_edge[:k].tolist()):
                if d == float('-inf'):
                    break
                path = [(h, t, r)]
                for j in range(i - 1, -1, -1):
                    h, t, r, prev_rank = back_edges[j][h, prev_rank].tolist()
                    path.append((h, t, r))
                paths.append(path[::-1])
                average_lengths.append(d / len(path))
        if paths:
            average_lengths, paths = zip(*sorted(zip(average_lengths, paths), reverse=True)[:k])
        return paths, average_lengths


class MLP_follow_model(nn.Module):

    def __init__(self, model, h_dim, out_dim):
        super(MLP_follow_model, self).__init__()
        self.gnn_model = model
        self.project = nn.Sequential(nn.Linear(h_dim, out_dim, bias=False))

    def forward(self, hg, h=None, category=None):
        if h is None:
            h = self.gnn_model(hg)
        else:
            h = self.gnn_model(hg, h)
        for i in h:
            h[i] = self.project(h[i])
        return h


class NSHE(BaseModel):
    """
    NSHE[IJCAI2020]
    Network Schema Preserving Heterogeneous Information Network Embedding
    `Paper Link <http://www.shichuan.org/doc/87.pdf>`
    `Code Link https://github.com/Andy-Border/NSHE`

    """

    @classmethod
    def build_model_from_args(cls, args, hg):
        return cls(hg, 'GCN', project_dim=args.dim_size['project'], emd_dim=args.dim_size['emd'], context_dim=args.dim_size['context'], num_heads=args.num_heads, dropout=args.dropout)

    def __init__(self, g, gnn_model, project_dim, emd_dim, context_dim, num_heads, dropout):
        super(NSHE, self).__init__()
        self.gnn_model = gnn_model
        self.norm_emb = True
        self.project_dim = project_dim
        self.emd_dim = emd_dim
        self.context_dim = context_dim
        self.num_heads = num_heads
        self.dropout = dropout
        linear_dict1 = {}
        linear_dict2 = {}
        linear_dict3 = {}
        cla_dim = self.emd_dim + self.context_dim * (len(g.ntypes) - 1)
        for ntype in g.ntypes:
            in_dim = g.nodes[ntype].data['h'].shape[1]
            linear_dict1[ntype] = in_dim, self.project_dim
            linear_dict2[ntype] = self.emd_dim, self.context_dim
            linear_dict3[ntype] = cla_dim, 1
        self.feature_proj = HeteroLinearLayer(linear_dict1, has_l2norm=False, has_bn=False)
        if self.gnn_model == 'GCN':
            self.gnn1 = GraphConv(self.project_dim, self.emd_dim, norm='none', activation=F.relu)
            self.gnn2 = GraphConv(self.emd_dim, self.emd_dim, norm='none', activation=None)
        elif self.gnn_model == 'GAT':
            self.gnn1 = GraphConv(self.project_dim, self.emd_dim, activation=F.relu)
            self.gnn2 = GraphConv(self.emd_dim, self.emd_dim, activation=None)
        self.context_encoder = HeteroLinearLayer(linear_dict2, has_l2norm=False, has_bn=False)
        self.linear_classifier = HeteroMLPLayer(linear_dict3, has_l2norm=False, has_bn=False)

    def forward(self, hg, h):
        with hg.local_scope():
            h_dict = self.feature_proj(h)
            hg.ndata['h_proj'] = h_dict
            g_homo = dgl.to_homogeneous(hg, ndata=['h_proj'])
            h = g_homo.ndata['h_proj']
            h = self.gnn2(g_homo, h)
            if self.norm_emb:
                h = F.normalize(h, p=2, dim=1)
            emd = self.h2dict(h, h_dict)
        return emd, h

    def h2dict(self, h, hdict):
        pre = 0
        for i, value in hdict.items():
            hdict[i] = h[pre:value.shape[0] + pre]
            pre += value.shape[0]
        return hdict


class NSHELayer(nn.Module):

    def __init__(self, in_feat, out_feat, num_heads, rel_names, activation=None, dropout=0.0):
        super(NSHELayer, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.rel_names = rel_names
        self.num_heads = num_heads
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        self.conv = dgl.nn.HeteroGraphConv({rel: dgl.nn.pytorch.GATConv(in_feat, out_feat // num_heads, num_heads) for rel in rel_names})
        self.loop_weight = nn.Parameter(th.Tensor(in_feat, out_feat))
        nn.init.xavier_uniform_(self.loop_weight, gain=nn.init.calculate_gain('relu'))
        self.h_bias = nn.Parameter(th.Tensor(out_feat))
        nn.init.zeros_(self.h_bias)

    def forward(self, g, inputs):
        g = g.local_var()
        if g.is_block:
            inputs_src = inputs
            inputs_dst = {k: v[:g.number_of_dst_nodes(k)] for k, v in inputs.items()}
        else:
            inputs_src = inputs_dst = inputs
        hs = self.conv(g, inputs_src)

        def _apply(ntype, h):
            h = h + th.matmul(inputs_dst[ntype], self.loop_weight)
            h = h + self.h_bias
            if self.activation:
                h = self.activation(h)
            return self.dropout(h)
        return {ntype: _apply(ntype, h) for ntype, h in hs.items()}


class RGATLayer(nn.Module):

    def __init__(self, in_feat, out_feat, num_heads, rel_names, activation=None, dropout=0.0, last_layer_flag=False, bias=True):
        super(RGATLayer, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.num_heads = num_heads
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        self.last_layer_flag = last_layer_flag
        self.conv = dglnn.HeteroGraphConv({rel: dgl.nn.pytorch.GATConv(in_feat, out_feat, num_heads=num_heads, bias=bias, allow_zero_in_degree=True) for rel in rel_names})

    def forward(self, g, h_dict):
        h_dict = self.conv(g, h_dict)
        out_put = {}
        for n_type, h in h_dict.items():
            if self.last_layer_flag:
                h = h.mean(1)
            else:
                h = h.flatten(1)
            out_put[n_type] = h.squeeze()
        return out_put


class RGAT(BaseModel):

    @classmethod
    def build_model_from_args(cls, args, hg):
        return cls(in_dim=args.in_dim, out_dim=args.out_dim, h_dim=args.hidden_dim, etypes=hg.etypes, num_heads=args.num_heads, num_hidden_layers=args.num_layers - 2, dropout=args.dropout)

    def __init__(self, in_dim, out_dim, h_dim, etypes, num_heads, num_hidden_layers=1, dropout=0):
        super(RGAT, self).__init__()
        self.rel_names = etypes
        self.layers = nn.ModuleList()
        self.layers.append(RGATLayer(in_dim, h_dim, num_heads, self.rel_names, activation=F.relu, dropout=dropout, last_layer_flag=False))
        for i in range(num_hidden_layers):
            self.layers.append(RGATLayer(h_dim * num_heads, h_dim, num_heads, self.rel_names, activation=F.relu, dropout=dropout, last_layer_flag=False))
        self.layers.append(RGATLayer(h_dim * num_heads, out_dim, num_heads, self.rel_names, activation=None, last_layer_flag=True))
        return

    def forward(self, hg, h_dict=None):
        if hasattr(hg, 'ntypes'):
            for layer in self.layers:
                h_dict = layer(hg, h_dict)
        else:
            for layer, block in zip(self.layers, hg):
                h_dict = layer(block, h_dict)
        return h_dict


class RelationCrossing(nn.Module):

    def __init__(self, in_feats: 'int', out_feats: 'int', num_heads: 'int', dropout: 'float'=0.0, negative_slope: 'float'=0.2):
        """
        Relation crossing layer

        Parameters
        ----------
        in_feats : pair of ints
            input feature size
        out_feats : int
            output feature size
        num_heads : int
            number of heads in Multi-Head Attention
        dropout : float
            optional, dropout rate, defaults: 0.0
        negative_slope : float
            optional, negative slope rate, defaults: 0.2
        """
        super(RelationCrossing, self).__init__()
        self._in_feats = in_feats
        self._out_feats = out_feats
        self._num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(negative_slope)

    def forward(self, dsttype_node_features: 'torch.Tensor', relations_crossing_attention_weight: 'nn.Parameter'):
        """
        Parameters
        ----------
        dsttype_node_features:
            a tensor of (dsttype_node_relations_num, num_dst_nodes, n_heads * hidden_dim)
        relations_crossing_attention_weight:
            Parameter the shape is (n_heads, hidden_dim)
        Returns:
        ----------
        output_features: Tensor

        """
        if len(dsttype_node_features) == 1:
            dsttype_node_features = dsttype_node_features.squeeze(dim=0)
        else:
            dsttype_node_features = dsttype_node_features.reshape(dsttype_node_features.shape[0], -1, self._num_heads, self._out_feats)
            dsttype_node_relation_attention = (dsttype_node_features * relations_crossing_attention_weight).sum(dim=-1, keepdim=True)
            dsttype_node_relation_attention = F.softmax(self.leaky_relu(dsttype_node_relation_attention), dim=0)
            dsttype_node_features = (dsttype_node_features * dsttype_node_relation_attention).sum(dim=0)
            dsttype_node_features = self.dropout(dsttype_node_features)
            dsttype_node_features = dsttype_node_features.reshape(-1, self._num_heads * self._out_feats)
        return dsttype_node_features


class RelationGraphConv(nn.Module):

    def __init__(self, in_feats: 'tuple', out_feats: 'int', num_heads: 'int', dropout: 'float'=0.0, negative_slope: 'float'=0.2):
        """
        Relation graph convolution layer

        Parameters
        ----------
        in_feats : pair of ints
            input feature size
        out_feats : int
            output feature size
        num_heads : int
            number of heads in Multi-Head Attention
        dropout : float
            optional, dropout rate, defaults: 0
        negative_slope : float
            optional, negative slope rate, defaults: 0.2
        """
        super(RelationGraphConv, self).__init__()
        self._in_src_feats, self._in_dst_feats = in_feats[0], in_feats[1]
        self._out_feats = out_feats
        self._num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.relu = nn.ReLU()

    def forward(self, graph: 'dgl.DGLHeteroGraph', feat: 'tuple', dst_node_transformation_weight: 'nn.Parameter', src_node_transformation_weight: 'nn.Parameter', relation_embedding: 'torch.Tensor', relation_transformation_weight: 'nn.Parameter'):
        """

        Parameters
        ----------
        graph : specific relational DGLHeteroGraph
        feat : pair of torch.Tensor
            e.g The pair contains two tensors of shape (N_{in}, D_{in_{src}})` and (N_{out}, D_{in_{dst}}).
        dst_node_transformation_weight:
            e.g Parameter (input_dst_dim, n_heads * hidden_dim)
        src_node_transformation_weight:
            e.g Parameter (input_src_dim, n_heads * hidden_dim)
        relation_embedding: torch.Tensor
            e.g (relation_input_dim)
        relation_transformation_weight:
            e,g Parameter (relation_input_dim, n_heads * 2 * hidden_dim)

        Returns
        -------
        dst_features: torch.Tensor
            shape (N, H, D_out)` where H is the number of heads, and D_out is size of output feature.
        """
        graph = graph.local_var()
        feat_src = self.dropout(feat[0])
        feat_dst = self.dropout(feat[1])
        feat_src = torch.matmul(feat_src, src_node_transformation_weight).view(-1, self._num_heads, self._out_feats)
        feat_dst = torch.matmul(feat_dst, dst_node_transformation_weight).view(-1, self._num_heads, self._out_feats)
        relation_attention_weight = torch.matmul(relation_embedding.unsqueeze(dim=0), relation_transformation_weight).view(self._num_heads, 2 * self._out_feats)
        e_dst = (feat_dst * relation_attention_weight[:, :self._out_feats]).sum(dim=-1, keepdim=True)
        e_src = (feat_src * relation_attention_weight[:, self._out_feats:]).sum(dim=-1, keepdim=True)
        graph.srcdata.update({'ft': feat_src, 'e_src': e_src})
        graph.dstdata.update({'e_dst': e_dst})
        graph.apply_edges(fn.u_add_v('e_src', 'e_dst', 'e'))
        e = self.leaky_relu(graph.edata.pop('e'))
        graph.edata['a'] = edge_softmax(graph, e)
        graph.update_all(fn.u_mul_e('ft', 'a', 'msg'), fn.sum('msg', 'feat'))
        dst_features = graph.dstdata.pop('feat').reshape(-1, self._num_heads * self._out_feats)
        dst_features = self.relu(dst_features)
        return dst_features


class R_HGNN_Layer(nn.Module):

    def __init__(self, graph, input_dim: 'int', hidden_dim: 'int', relation_input_dim: 'int', relation_hidden_dim: 'int', n_heads: 'int'=8, dropout: 'float'=0.2, negative_slope: 'float'=0.2, residual: 'bool'=True, norm: 'bool'=False):
        """
        Parameters
        ----------
        graph:
            a heterogeneous graph
        input_dim: int
            node input dimension
        hidden_dim: int
            node hidden dimension
        relation_input_dim: int
            relation input dimension
        relation_hidden_dim: int
            relation hidden dimension
        n_heads: int
            number of attention heads
        dropout: float
            dropout rate
        negative_slope: float
            negative slope
        residual: boolean
            residual connections or not
        norm: boolean
            layer normalization or not
        """
        super(R_HGNN_Layer, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.relation_input_dim = relation_input_dim
        self.relation_hidden_dim = relation_hidden_dim
        self.n_heads = n_heads
        self.dropout = dropout
        self.negative_slope = negative_slope
        self.residual = residual
        self.norm = norm
        self.node_transformation_weight = nn.ParameterDict({ntype: nn.Parameter(torch.randn(input_dim, n_heads * hidden_dim)) for ntype in graph.ntypes})
        self.relation_transformation_weight = nn.ParameterDict({etype: nn.Parameter(torch.randn(relation_input_dim, n_heads * 2 * hidden_dim)) for etype in graph.etypes})
        self.relation_propagation_layer = nn.ModuleDict({etype: nn.Linear(relation_input_dim, n_heads * relation_hidden_dim) for etype in graph.etypes})
        self.hetero_conv = HeteroGraphConv({etype: RelationGraphConv(in_feats=(input_dim, input_dim), out_feats=hidden_dim, num_heads=n_heads, dropout=dropout, negative_slope=negative_slope) for etype in graph.etypes})
        if self.residual:
            self.res_fc = nn.ModuleDict()
            self.residual_weight = nn.ParameterDict()
            for ntype in graph.ntypes:
                self.res_fc[ntype] = nn.Linear(input_dim, n_heads * hidden_dim)
                self.residual_weight[ntype] = nn.Parameter(torch.randn(1))
        if self.norm:
            self.layer_norm = nn.ModuleDict({ntype: nn.LayerNorm(n_heads * hidden_dim) for ntype in graph.ntypes})
        self.relations_crossing_attention_weight = nn.ParameterDict({etype: nn.Parameter(torch.randn(n_heads, hidden_dim)) for etype in graph.etypes})
        self.relations_crossing_layer = RelationCrossing(in_feats=n_heads * hidden_dim, out_feats=hidden_dim, num_heads=n_heads, dropout=dropout, negative_slope=negative_slope)
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')
        for weight in self.node_transformation_weight:
            nn.init.xavier_normal_(self.node_transformation_weight[weight], gain=gain)
        for weight in self.relation_transformation_weight:
            nn.init.xavier_normal_(self.relation_transformation_weight[weight], gain=gain)
        for etype in self.relation_propagation_layer:
            nn.init.xavier_normal_(self.relation_propagation_layer[etype].weight, gain=gain)
        if self.residual:
            for ntype in self.res_fc:
                nn.init.xavier_normal_(self.res_fc[ntype].weight, gain=gain)
        for weight in self.relations_crossing_attention_weight:
            nn.init.xavier_normal_(self.relations_crossing_attention_weight[weight], gain=gain)

    def forward(self, graph: 'dgl.DGLHeteroGraph', relation_target_node_features: 'dict', relation_embedding: 'dict'):
        """

        :param graph: dgl.DGLHeteroGraph
        :param relation_target_node_features: dict, {relation_type: target_node_features shape (N_nodes, input_dim)},
               each value in relation_target_node_features represents the representation of target node features
        :param relation_embedding: embedding for each relation, dict, {etype: feature}
        :return: output_features: dict, {relation_type: target_node_features}
        """
        input_src = relation_target_node_features
        if graph.is_block:
            input_dst = {}
            for srctype, etypye, dsttype in relation_target_node_features:
                input_dst[srctype, etypye, dsttype] = relation_target_node_features[srctype, etypye, dsttype][:graph.number_of_dst_nodes(dsttype)]
        else:
            input_dst = relation_target_node_features
        output_features = self.hetero_conv(graph, input_src, input_dst, relation_embedding, self.node_transformation_weight, self.relation_transformation_weight)
        if self.residual:
            for srctype, etype, dsttype in output_features:
                alpha = torch.sigmoid(self.residual_weight[dsttype])
                output_features[srctype, etype, dsttype] = output_features[srctype, etype, dsttype] * alpha + self.res_fc[dsttype](input_dst[srctype, etype, dsttype]) * (1 - alpha)
        output_features_dict = {}
        for srctype, etype, dsttype in output_features:
            dst_node_relations_features = torch.stack([output_features[stype, reltype, dtype] for stype, reltype, dtype in output_features if dtype == dsttype], dim=0)
            output_features_dict[srctype, etype, dsttype] = self.relations_crossing_layer(dst_node_relations_features, self.relations_crossing_attention_weight[etype])
        if self.norm:
            for srctype, etype, dsttype in output_features_dict:
                output_features_dict[srctype, etype, dsttype] = self.layer_norm[dsttype](output_features_dict[srctype, etype, dsttype])
        relation_embedding_dict = {}
        for etype in relation_embedding:
            relation_embedding_dict[etype] = self.relation_propagation_layer[etype](relation_embedding[etype])
        return output_features_dict, relation_embedding_dict


class RelationFusing(nn.Module):

    def __init__(self, node_hidden_dim: 'int', relation_hidden_dim: 'int', num_heads: 'int', dropout: 'float'=0.0, negative_slope: 'float'=0.2):
        """

        Parameters
        ----------
        node_hidden_dim: int
            node hidden feature size
        relation_hidden_dim: int
            relation hidden feature size
        num_heads: int
            number of heads in Multi-Head Attention
        dropout: float
            dropout rate, defaults: 0.0
        negative_slope: float
            negative slope, defaults: 0.2
        """
        super(RelationFusing, self).__init__()
        self.node_hidden_dim = node_hidden_dim
        self.relation_hidden_dim = relation_hidden_dim
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(negative_slope)

    def forward(self, dst_node_features: 'list', dst_relation_embeddings: 'list', dst_node_feature_transformation_weight: 'list', dst_relation_embedding_transformation_weight: 'list'):
        """
        Parameters
        ----------
        dst_node_features: list
            e.g [each shape is (num_dst_nodes, n_heads * node_hidden_dim)]
        dst_relation_embeddings: list
            e.g [each shape is (n_heads * relation_hidden_dim)]
        dst_node_feature_transformation_weight: list
            e.g [each shape is (n_heads, node_hidden_dim, node_hidden_dim)]
        dst_relation_embedding_transformation_weight:  list
            e.g [each shape is (n_heads, relation_hidden_dim, relation_hidden_dim)]

        Returns
        ----------
        dst_node_relation_fusion_feature: Tensor
            the target node representation after relation-aware representations fusion
        """
        if len(dst_node_features) == 1:
            dst_node_relation_fusion_feature = dst_node_features[0]
        else:
            dst_node_features = torch.stack(dst_node_features, dim=0).reshape(len(dst_node_features), -1, self.num_heads, self.node_hidden_dim)
            dst_relation_embeddings = torch.stack(dst_relation_embeddings, dim=0).reshape(len(dst_node_features), self.num_heads, self.relation_hidden_dim)
            dst_node_feature_transformation_weight = torch.stack(dst_node_feature_transformation_weight, dim=0).reshape(len(dst_node_features), self.num_heads, self.node_hidden_dim, self.node_hidden_dim)
            dst_relation_embedding_transformation_weight = torch.stack(dst_relation_embedding_transformation_weight, dim=0).reshape(len(dst_node_features), self.num_heads, self.relation_hidden_dim, self.node_hidden_dim)
            dst_node_features = torch.einsum('abcd,acde->abce', dst_node_features, dst_node_feature_transformation_weight)
            dst_relation_embeddings = torch.einsum('abc,abcd->abd', dst_relation_embeddings, dst_relation_embedding_transformation_weight)
            attention_scores = (dst_node_features * dst_relation_embeddings.unsqueeze(dim=1)).sum(dim=-1, keepdim=True)
            attention_scores = F.softmax(self.leaky_relu(attention_scores), dim=0)
            dst_node_relation_fusion_feature = (dst_node_features * attention_scores).sum(dim=0)
            dst_node_relation_fusion_feature = self.dropout(dst_node_relation_fusion_feature)
            dst_node_relation_fusion_feature = dst_node_relation_fusion_feature.reshape(-1, self.num_heads * self.node_hidden_dim)
        return dst_node_relation_fusion_feature


class RHGNN(BaseModel):
    """
    This is the main method of model RHGNN

    Parameters
    ----------
    graph: dgl.DGLHeteroGraph
        a heterogeneous graph
    input_dim_dict: dict
        node input dimension dictionary
    hidden_dim: int
        node hidden dimension
    relation_input_dim: int
        relation input dimension
    relation_hidden_dim: int
        relation hidden dimension
    num_layers: int
        number of stacked layers
    n_heads: int
        number of attention heads
    dropout: float
        dropout rate
    negative_slope: float
        negative slope
    residual: boolean
        residual connections or not
    norm: boolean
        layer normalization or not
    """

    @classmethod
    def build_model_from_args(cls, args, hg):
        input_dim_dict = {ntype: hg.nodes[ntype].data['h'].shape[1] for ntype in hg.ntypes}
        return cls(graph=hg, input_dim_dict=input_dim_dict, hidden_dim=args.hidden_dim, relation_input_dim=args.relation_hidden_units, relation_hidden_dim=args.relation_hidden_units, num_layers=args.num_layers, category=args.category, out_dim=args.out_dim)

    def __init__(self, graph: 'dgl.DGLHeteroGraph', input_dim_dict, hidden_dim: 'int', relation_input_dim: 'int', relation_hidden_dim: 'int', num_layers: 'int', category, out_dim, n_heads: 'int'=4, dropout: 'float'=0.2, negative_slope: 'float'=0.2, residual: 'bool'=True, norm: 'bool'=True):
        super(RHGNN, self).__init__()
        self.category = category
        self.input_dim_dict = input_dim_dict
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.relation_input_dim = relation_input_dim
        self.relation_hidden_dim = relation_input_dim
        self.n_heads = n_heads
        self.dropout = dropout
        self.negative_slope = negative_slope
        self.residual = residual
        self.out_dim = out_dim
        self.norm = norm
        None
        self.relation_embedding = nn.ParameterDict({etype: nn.Parameter(torch.randn(relation_input_dim, 1)) for etype in graph.etypes})
        self.projection_layer = nn.ModuleDict({ntype: nn.Linear(self.input_dim_dict[ntype], hidden_dim * n_heads) for ntype in input_dim_dict})
        self.layers = nn.ModuleList()
        self.layers.append(R_HGNN_Layer(graph, hidden_dim * n_heads, hidden_dim, relation_input_dim, relation_hidden_dim, n_heads, dropout, negative_slope, residual, norm))
        for _ in range(1, self.num_layers):
            self.layers.append(R_HGNN_Layer(graph, hidden_dim * n_heads, hidden_dim, relation_hidden_dim * n_heads, relation_hidden_dim, n_heads, dropout, negative_slope, residual, norm))
        self.node_transformation_weight = nn.ParameterDict({etype: nn.Parameter(torch.randn(n_heads, hidden_dim, hidden_dim)) for etype in graph.etypes})
        self.relation_transformation_weight = nn.ParameterDict({etype: nn.Parameter(torch.randn(n_heads, relation_hidden_dim, hidden_dim)) for etype in graph.etypes})
        self.relation_fusing = RelationFusing(node_hidden_dim=hidden_dim, relation_hidden_dim=relation_hidden_dim, num_heads=n_heads, dropout=dropout, negative_slope=negative_slope)
        self.classifier = nn.Linear(self.hidden_dim * self.n_heads, self.out_dim)
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')
        for etype in self.relation_embedding:
            nn.init.xavier_normal_(self.relation_embedding[etype], gain=gain)
        for ntype in self.projection_layer:
            nn.init.xavier_normal_(self.projection_layer[ntype].weight, gain=gain)
        for etype in self.node_transformation_weight:
            nn.init.xavier_normal_(self.node_transformation_weight[etype], gain=gain)
        for etype in self.relation_transformation_weight:
            nn.init.xavier_normal_(self.relation_transformation_weight[etype], gain=gain)

    def forward(self, blocks: 'list', relation_target_node_features=None, relation_embedding: 'dict'=None):
        """

        Parameters
        ----------
        blocks: list
            list of sampled dgl.DGLHeteroGraph
        relation_target_node_features: dict
            target node features under each relation, e.g {(srctype, etype, dsttype): features}
        relation_embedding: dict
            embedding for each relation, e.g {etype: feature} or None

        """
        relation_target_node_features = {}
        for stype, etype, dtype in blocks[0].canonical_etypes:
            relation_target_node_features[stype, etype, dtype] = blocks[0].srcnodes[dtype].data.get('h')
        for stype, reltype, dtype in relation_target_node_features:
            relation_target_node_features[stype, reltype, dtype] = self.projection_layer[dtype](relation_target_node_features[stype, reltype, dtype])
        if relation_embedding is None:
            relation_embedding = {}
            for etype in self.relation_embedding:
                relation_embedding[etype] = self.relation_embedding[etype].flatten()
        for block, layer in zip(blocks, self.layers):
            relation_target_node_features, relation_embedding = layer(block, relation_target_node_features, relation_embedding)
        relation_fusion_embedding_dict = {}
        for dsttype in set([dtype for _, _, dtype in relation_target_node_features]):
            relation_target_node_features_dict = {etype: relation_target_node_features[stype, etype, dtype] for stype, etype, dtype in relation_target_node_features}
            etypes = [etype for stype, etype, dtype in relation_target_node_features if dtype == dsttype]
            dst_node_features = [relation_target_node_features_dict[etype] for etype in etypes]
            dst_relation_embeddings = [relation_embedding[etype] for etype in etypes]
            dst_node_feature_transformation_weight = [self.node_transformation_weight[etype] for etype in etypes]
            dst_relation_embedding_transformation_weight = [self.relation_transformation_weight[etype] for etype in etypes]
            dst_node_relation_fusion_feature = self.relation_fusing(dst_node_features, dst_relation_embeddings, dst_node_feature_transformation_weight, dst_relation_embedding_transformation_weight)
            relation_fusion_embedding_dict[dsttype] = dst_node_relation_fusion_feature
        classifier_result = self.classifier(relation_fusion_embedding_dict[self.category])
        return {self.category: classifier_result}

    def inference(self, graph: 'dgl.DGLHeteroGraph', relation_target_node_features: 'dict', relation_embedding: 'dict'=None, device: 'str'='cuda:0'):
        """
        mini-batch inference of final representation over all node types. Outer loop: Interate the layers, Inner loop: Interate the batches

        Parameters
        ----------
        graph: dgl.DGLHeteroGraph
            The whole relational graphs
        relation_target_node_features:  dict
            target node features under each relation, e.g {(srctype, etype, dsttype): features}
        relation_embedding: dict
            embedding for each relation, e.g {etype: feature} or None
        device: str
            device

        """
        with torch.no_grad():
            if relation_embedding is None:
                relation_embedding = {}
                for etype in self.relation_embedding:
                    relation_embedding[etype] = self.relation_embedding[etype].flatten()
            for index, layer in enumerate(self.layers):
                y = {(stype, etype, dtype): torch.zeros(graph.number_of_nodes(dtype), self.hidden_dim * self.n_heads) for stype, etype, dtype in graph.canonical_etypes}
                sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
                dataloader = dgl.dataloading.NodeDataLoader(graph, {ntype: torch.arange(graph.number_of_nodes(ntype)) for ntype in graph.ntypes}, sampler, batch_size=1280, shuffle=True, drop_last=False, num_workers=4)
                tqdm_dataloader = tqdm(dataloader, ncols=120)
                for batch, (input_nodes, output_nodes, blocks) in enumerate(tqdm_dataloader):
                    block = blocks[0]
                    if len(set(blocks[0].ntypes)) == 1:
                        input_nodes = {blocks[0].ntypes[0]: input_nodes}
                        output_nodes = {blocks[0].ntypes[0]: output_nodes}
                    input_features = {(stype, etype, dtype): relation_target_node_features[stype, etype, dtype][input_nodes[dtype]] for stype, etype, dtype in relation_target_node_features.keys()}
                    input_relation_features = relation_embedding
                    if index == 0:
                        for stype, reltype, dtype in input_features:
                            input_features[stype, reltype, dtype] = self.projection_layer[dtype](input_features[stype, reltype, dtype])
                    h, input_relation_features = layer(block, input_features, input_relation_features)
                    for stype, reltype, dtype in h.keys():
                        y[stype, reltype, dtype][output_nodes[dtype]] = h[stype, reltype, dtype].cpu()
                    tqdm_dataloader.set_description(f'inference for the {batch}-th batch in model {index}-th layer')
                relation_target_node_features = y
                relation_embedding = input_relation_features
            for stype, etype, dtype in relation_target_node_features:
                relation_target_node_features[stype, etype, dtype] = relation_target_node_features[stype, etype, dtype]
            relation_fusion_embedding_dict = {}
            for dsttype in set([dtype for _, _, dtype in relation_target_node_features]):
                relation_target_node_features_dict = {etype: relation_target_node_features[stype, etype, dtype] for stype, etype, dtype in relation_target_node_features}
                etypes = [etype for stype, etype, dtype in relation_target_node_features if dtype == dsttype]
                dst_node_features = [relation_target_node_features_dict[etype] for etype in etypes]
                dst_relation_embeddings = [relation_embedding[etype] for etype in etypes]
                dst_node_feature_transformation_weight = [self.node_transformation_weight[etype] for etype in etypes]
                dst_relation_embedding_transformation_weight = [self.relation_transformation_weight[etype] for etype in etypes]
                relation_fusion_embedding = []
                index = 0
                batch_size = 2560
                while index < dst_node_features[0].shape[0]:
                    relation_fusion_embedding.append(self.relation_fusing([dst_node_feature[index:index + batch_size, :] for dst_node_feature in dst_node_features], dst_relation_embeddings, dst_node_feature_transformation_weight, dst_relation_embedding_transformation_weight))
                    index += batch_size
                relation_fusion_embedding_dict[dsttype] = torch.cat(relation_fusion_embedding, dim=0)
            return relation_fusion_embedding_dict, relation_target_node_features


class ARLayer(nn.Module):

    def __init__(self, ent_emb, rel_emb, score_dim, device):
        super(ARLayer, self).__init__()
        self.ent_embeddings = ent_emb
        self.rel_embeddings = rel_emb
        self.score_dim = score_dim
        self.device = device

    def trans_dist(self, edges):
        return {'ar_dis_score': torch.sum(edges._src_data['h_emb'] + edges.data['r_emb'] - edges._dst_data['t_emb'], 1)}

    def forward(self, hg):
        with hg.local_scope():
            for n in hg.ntypes:
                hg.srcnodes[n].data['h_emb'] = hg.dstnodes[n].data['h_emb'] = self.ent_embeddings(hg.nodes[n].data['_ID'])
                hg.srcnodes[n].data['t_emb'] = hg.dstnodes[n].data['t_emb'] = self.ent_embeddings(hg.nodes[n].data['_ID'])
            for e in hg.etypes:
                hg.edges[e].data['r_emb'] = self.rel_embeddings(hg.edges[e].data['_ID'])
            scores = []
            for rel in hg.etypes:
                hg.apply_edges(self.trans_dist, etype=rel)
                score = hg.edges[rel].data['ar_dis_score']
                if score.shape[0] < self.score_dim:
                    score = torch.cat([score, torch.zeros(self.score_dim - score.shape[0])])
                scores.append(score)
            return torch.cat(scores)


class IRLayer(nn.Module):

    def __init__(self, ent_emb, score_dim, device):
        super(IRLayer, self).__init__()
        self.ent_embeddings = ent_emb
        self.score_dim = score_dim
        self.device = device

    def eur_dist(self, edges):
        return {'ir_dis_score': torch.sum(torch.pow(edges.src['h_emb'] - edges.dst['t_emb'], 2), 1)}

    def forward(self, hg):
        with hg.local_scope():
            for n in hg.ntypes:
                hg.srcnodes[n].data['h_emb'] = hg.dstnodes[n].data['h_emb'] = self.ent_embeddings(hg.nodes[n].data['_ID'])
                hg.srcnodes[n].data['t_emb'] = hg.dstnodes[n].data['t_emb'] = self.ent_embeddings(hg.nodes[n].data['_ID'])
            scores = []
            for rel in hg.etypes:
                hg.apply_edges(self.eur_dist, etype=rel)
                score = hg.edges[rel].data['ir_dis_score']
                if score.shape[0] < self.score_dim:
                    score = torch.cat([score, torch.zeros(self.score_dim - score.shape[0])])
                scores.append(score)
            return torch.cat(scores)


class RHINE(BaseModel):

    @classmethod
    def build_model_from_args(cls, args, hg):
        total_nodes = hg.num_nodes()
        total_IRs = args.total_IRs
        ARs = args.ARs
        IRs = args.IRs
        score_dim = args.batch_size
        device = args.device
        return cls(total_nodes, total_IRs, ARs, IRs, args.emb_dim, score_dim, device, args.hidden_dim, args.out_dim)

    def __init__(self, total_nodes, total_IRs, ARs, IRs, emb_dim, score_dim, device, hid_dim=100, out_dim=4):
        super(RHINE, self).__init__()
        self.DisLayer = nn.ModuleDict()
        self.device = device
        self.ent_embeddings = nn.Embedding(total_nodes, emb_dim)
        self.rel_embeddings = nn.Embedding(total_IRs, emb_dim)
        for AR in ARs:
            self.DisLayer[AR] = ARLayer(self.ent_embeddings, self.rel_embeddings, score_dim, device)
        for IR in IRs:
            self.DisLayer[IR] = IRLayer(self.ent_embeddings, score_dim, device)
        self.predictor = nn.ModuleList([nn.Linear(emb_dim, hid_dim), nn.ReLU(), nn.Linear(hid_dim, out_dim)])

    def forward(self, hg_dict, category=None, mod='train'):
        if mod == 'train':
            scores = {}
            for mp, hg in hg_dict.items():
                score = self.DisLayer[mp](hg)
                scores[mp] = score
            scores = torch.stack(list(scores.values()))
            return scores
        else:
            assert category is not None
            h = self.ent_embeddings(hg_dict.nodes[category].data['h'])
            for layer in self.predictor:
                h = layer(h)
            return h.squeeze(1)


class AGNNConv(nn.Module):

    def __init__(self, eps=0.0, train_eps=False, learn_beta=True):
        super(AGNNConv, self).__init__()
        self.initial_eps = eps
        if learn_beta:
            self.beta = nn.Parameter(th.Tensor(1))
        else:
            self.register_buffer('beta', th.Tensor(1))
        self.learn_beta = learn_beta
        if train_eps:
            self.eps = th.nn.Parameter(th.ones([eps]))
        else:
            self.register_buffer('eps', th.Tensor([eps]))
        self.reset_parameters()

    def reset_parameters(self):
        self.eps.data.fill_(self.initial_eps)
        if self.learn_beta:
            self.beta.data.fill_(1)

    def forward(self, graph, feat, edge_weight):
        with graph.local_scope():
            feat_src, feat_dst = expand_as_pair(feat, graph)
            graph.srcdata['norm_h'] = F.normalize(feat_src, p=2, dim=-1)
            e = self.beta * edge_weight
            graph.edata['p'] = edge_softmax(graph, e, norm_by='src')
            graph.update_all(fn.u_mul_e('norm_h', 'p', 'm'), fn.sum('m', 'h'))
            rst = graph.dstdata.pop('h')
            rst = (1 + self.eps) * feat + rst
            return rst


def give_one_hot_feats(g, ntype='h'):
    num_nodes = g.num_nodes()
    g.ndata[ntype] = th.eye(num_nodes)
    return g


class coarsened_line_graph:

    def __init__(self, rw_len, batch_size, n_dataset, symmetric=True):
        self.rw_len = rw_len
        self.batch_size = batch_size
        self.n_dataset = n_dataset
        self.symmetric = symmetric
        return

    def get_cl_graph(self, hg):
        fname = './openhgnn/output/RSHN/{}_cl_graoh_{}_{}.bin'.format(self.n_dataset, self.rw_len, self.batch_size)
        if os.path.exists(fname):
            g, _ = load_graphs(fname)
            return g[0]
        else:
            g = self.build_cl_graph(hg)
            save_graphs(fname, g)
            return g

    def init_cl_graph(self, cl_graph):
        cl_graph = give_one_hot_feats(cl_graph, 'h')
        cl_graph = dgl.remove_self_loop(cl_graph)
        edge_attr = cl_graph.edata['w'].type(th.FloatTensor)
        row, col = cl_graph.edges()
        for i in range(cl_graph.num_nodes()):
            mask = th.eq(row, i)
            edge_attr[mask] = th.nn.functional.normalize(edge_attr[mask], p=2, dim=0)
        cl_graph = dgl.add_self_loop(cl_graph)
        edge_attr = th.cat([edge_attr, th.ones(cl_graph.num_nodes(), device=edge_attr.device)], dim=0)
        cl_graph.edata['w'] = edge_attr
        return cl_graph

    def build_cl_graph(self, hg):
        if not hg.is_homogeneous:
            self.num_edge_type = len(hg.etypes)
            g = dgl.to_homogeneous(hg)
        traces = self.random_walks(g)
        edge_batch = self.rw_map_edge_type(g, traces)
        cl_graph = self.edge2graph(edge_batch)
        return cl_graph

    def random_walks(self, g):
        source_nodes = th.randint(0, g.number_of_nodes(), (self.batch_size,))
        traces, _ = dgl.sampling.random_walk(g, source_nodes, length=self.rw_len - 1)
        return traces

    def rw_map_edge_type(self, g, traces):
        edge_type = g.edata[dgl.ETYPE].long()
        edge_batch = []
        first_flag = True
        for t in traces:
            u = t[:-1]
            v = t[1:]
            edge_path = edge_type[g.edge_ids(u, v)].unsqueeze(0)
            if first_flag == True:
                edge_batch = edge_path
                first_flag = False
            else:
                edge_batch = th.cat((edge_batch, edge_path), dim=0)
        return edge_batch

    def edge2graph(self, edge_batch):
        u = edge_batch[:, :-1].reshape(-1)
        v = edge_batch[:, 1:].reshape(-1)
        if self.symmetric:
            tmp = u
            u = th.cat((u, v), dim=0)
            v = th.cat((v, tmp), dim=0)
        g = dgl.graph((u, v))
        sg = dgl.to_simple(g, return_counts='w')
        return sg


class RSHN(BaseModel):
    """
    Relation structure-aware heterogeneous graph neural network (RSHN) builds coarsened line graph to obtain edge features first,
    then uses a novel Message Passing Neural Network (MPNN) to propagate node and edge features.

    We implement a API build a coarsened line graph.

    Attributes
    -----------
    edge_layers : AGNNConv
        Applied in Edge Layer.
    coarsened line graph : dgl.DGLGraph
        Propagate edge features.
    """

    @classmethod
    def build_model_from_args(cls, args, hg):
        rshn = cls(dim=args.hidden_dim, out_dim=args.out_dim, num_node_layer=args.num_node_layer, num_edge_layer=args.num_edge_layer, dropout=args.dropout)
        cl = coarsened_line_graph(rw_len=args.rw_len, batch_size=args.batch_size, n_dataset=args.dataset, symmetric=True)
        cl_graph = cl.get_cl_graph(hg)
        cl_graph = cl.init_cl_graph(cl_graph)
        rshn.cl_graph = cl_graph
        linear_e1 = nn.Linear(in_features=cl_graph.num_nodes(), out_features=args.hidden_dim, bias=False)
        nn.init.xavier_uniform_(linear_e1.weight)
        rshn.linear_e1 = linear_e1
        return rshn

    def __init__(self, dim, out_dim, num_node_layer, num_edge_layer, dropout):
        super(RSHN, self).__init__()
        self.num_node_layer = num_node_layer
        self.edge_layers = nn.ModuleList()
        for i in range(num_edge_layer):
            self.edge_layers.append(AGNNConv())
        self.node_layers = nn.ModuleList()
        for i in range(num_node_layer):
            self.node_layers.append(GraphConv(in_feats=dim, out_feats=dim, dropout=dropout, activation=th.tanh))
        self.linear = nn.Linear(in_features=dim, out_features=out_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.init_para()

    def init_para(self):
        return

    def forward(self, hg, n_feats, *args, **kwargs):
        """
        First, apply edge_layer in cl_graph to get edge embedding.
        Then, propagate node and edge features through GraphConv.
        """
        h = self.cl_graph.ndata['h']
        h_e = self.cl_graph.edata['w']
        for layer in self.edge_layers:
            h = th.relu(layer(self.cl_graph, h, h_e))
            h = self.dropout(h)
        h = self.linear_e1(h)
        edge_weight = {}
        for i, e in enumerate(hg.canonical_etypes):
            edge_weight[e] = h[i].expand(hg.num_edges(e), -1)
        if hasattr(hg, 'ntypes'):
            for layer in self.node_layers:
                n_feats = layer(hg, n_feats, edge_weight)
        else:
            pass
        for n in n_feats:
            n_feats[n] = self.linear(n_feats[n])
        return n_feats


class RedGNNLayer(torch.nn.Module):

    def __init__(self, in_dim, out_dim, attn_dim, n_rel, act=lambda x: x):
        super(RedGNNLayer, self).__init__()
        self.n_rel = n_rel
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.attn_dim = attn_dim
        self.act = act
        self.rela_embed = nn.Embedding(2 * n_rel + 1, in_dim)
        self.Ws_attn = nn.Linear(in_dim, attn_dim, bias=False)
        self.Wr_attn = nn.Linear(in_dim, attn_dim, bias=False)
        self.Wqr_attn = nn.Linear(in_dim, attn_dim)
        self.w_alpha = nn.Linear(attn_dim, 1)
        self.W_h = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, q_sub, q_rel, hidden, edges, n_node, old_nodes_new_idx):
        sub = edges[:, 4]
        rel = edges[:, 2]
        obj = edges[:, 5]
        hs = hidden[sub]
        hr = self.rela_embed(rel)
        r_idx = edges[:, 0]
        h_qr = self.rela_embed(q_rel)[r_idx]
        message = hs + hr
        alpha = torch.sigmoid(self.w_alpha(nn.ReLU()(self.Ws_attn(hs) + self.Wr_attn(hr) + self.Wqr_attn(h_qr))))
        message = alpha * message
        message_agg = scatter(message, index=obj, dim=0, dim_size=n_node, reduce='sum')
        hidden_new = self.act(self.W_h(message_agg))
        return hidden_new


class RedGNN(BaseModel):

    @classmethod
    def build_model_from_args(cls, args, loader):
        return cls(args, loader)

    def __init__(self, args, loader):
        super(RedGNN, self).__init__()
        self.device = args.device
        self.hidden_dim = args.hidden_dim
        self.attn_dim = args.attn_dim
        self.n_layer = args.n_layer
        self.loader = loader
        self.n_rel = self.loader.n_rel
        acts = {'relu': nn.ReLU(), 'tanh': torch.tanh, 'idd': lambda x: x}
        act = acts[args.act]
        self.act = act
        self.gnn_layers = []
        for i in range(self.n_layer):
            self.gnn_layers.append(RedGNNLayer(self.hidden_dim, self.hidden_dim, self.attn_dim, self.n_rel, act=act))
        self.gnn_layers = nn.ModuleList(self.gnn_layers)
        self.dropout = nn.Dropout(args.dropout)
        self.W_final = nn.Linear(self.hidden_dim, 1, bias=False)
        self.gate = nn.GRU(self.hidden_dim, self.hidden_dim)

    def forward(self, subs, rels, mode='transductive'):
        n = len(subs)
        n_ent = self.loader.n_ent if mode == 'transductive' else self.loader.n_ent_ind
        q_sub = torch.LongTensor(subs)
        q_rel = torch.LongTensor(rels)
        h0 = torch.zeros((1, n, self.hidden_dim))
        nodes = torch.cat([torch.arange(n).unsqueeze(1), q_sub.unsqueeze(1)], 1)
        hidden = torch.zeros(n, self.hidden_dim)
        for i in range(self.n_layer):
            nodes, edges, old_nodes_new_idx = self.loader.get_neighbors(nodes.data.cpu().numpy(), mode=mode)
            edges = edges
            old_nodes_new_idx = old_nodes_new_idx
            hidden = self.gnn_layers[i](q_sub, q_rel, hidden, edges, nodes.size(0), old_nodes_new_idx)
            h0 = torch.zeros(1, nodes.size(0), hidden.size(1)).index_copy_(1, old_nodes_new_idx, h0)
            hidden = self.dropout(hidden)
            hidden, h0 = self.gate(hidden.unsqueeze(0), h0)
            hidden = hidden.squeeze(0)
        scores = self.W_final(hidden).squeeze(-1)
        scores_all = torch.zeros((n, n_ent))
        scores_all[[nodes[:, 0], nodes[:, 1]]] = scores
        return scores_all


class RedGNNT(BaseModel):

    @classmethod
    def build_model_from_args(cls, args, loader):
        return cls(args, loader)

    def __init__(self, args, loader):
        super(RedGNNT, self).__init__()
        self.device = args.device
        self.hidden_dim = args.hidden_dim
        self.attn_dim = args.attn_dim
        self.n_layer = args.n_layer
        self.loader = loader
        self.n_rel = self.loader.n_rel
        acts = {'relu': nn.ReLU(), 'tanh': torch.tanh, 'idd': lambda x: x}
        act = acts[args.act]
        self.act = act
        self.gnn_layers = []
        for i in range(self.n_layer):
            self.gnn_layers.append(RedGNNLayer(self.hidden_dim, self.hidden_dim, self.attn_dim, self.n_rel, act=act))
        self.gnn_layers = nn.ModuleList(self.gnn_layers)
        self.dropout = nn.Dropout(args.dropout)
        self.W_final = nn.Linear(self.hidden_dim, 1, bias=False)
        self.gate = nn.GRU(self.hidden_dim, self.hidden_dim)

    def forward(self, subs, rels, mode='train'):
        n = len(subs)
        q_sub = torch.LongTensor(subs)
        q_rel = torch.LongTensor(rels)
        h0 = torch.zeros((1, n, self.hidden_dim))
        nodes = torch.cat([torch.arange(n).unsqueeze(1), q_sub.unsqueeze(1)], 1)
        hidden = torch.zeros(n, self.hidden_dim)
        for i in range(self.n_layer):
            nodes, edges, old_nodes_new_idx = self.loader.get_neighbors(nodes.data.cpu().numpy(), mode=mode)
            edges = edges
            old_nodes_new_idx = old_nodes_new_idx
            hidden = self.gnn_layers[i](q_sub, q_rel, hidden, edges, nodes.size(0), old_nodes_new_idx)
            h0 = torch.zeros(1, nodes.size(0), hidden.size(1)).index_copy_(1, old_nodes_new_idx, h0)
            hidden = self.dropout(hidden)
            hidden, h0 = self.gate(hidden.unsqueeze(0), h0)
            hidden = hidden.squeeze(0)
        scores = self.W_final(hidden).squeeze(-1)
        scores_all = torch.zeros((n, self.loader.n_ent))
        scores_all[[nodes[:, 0], nodes[:, 1]]] = scores
        return scores_all


class RoHe(BaseModel):
    """
    RoHe model:  ”Robust Heterogeneous Graph Neural Networks against Adversarial Attacks“ (AAAI2022)
    RoHe model shows an example of using HAN, called RoHe-HAN. Most of the settings remain consistent with HAN,
    with partial modifications made in forward function, specifically replacing a portion of GATConv with RoHeGATConv.
    HAN model shows an example of using dgl.metapath_reachable_graph on the original heterogeneous
    graph HAN from paper `Heterogeneous Graph Attention Network <https://arxiv.org/pdf/1903.07293.pdf>`__..
    Because the original HAN implementation only gives the preprocessed homogeneous graph, this model
    could not reproduce the result in HAN as they did not provide the preprocessing code, and we
    constructed another dataset from ACM with a different set of papers, connections, features and
    labels.


    .. math::
        \\mathbf{h}_{i}^{\\prime}=\\mathbf{M}_{\\phi_{i}} \\cdot \\mathbf{h}_{i}

    where :math:`h_i` and :math:`h'_i` are the original and projected feature of node :math:`i`

    .. math::
        e_{i j}^{\\Phi}=a t t_{\\text {node }}\\left(\\mathbf{h}_{i}^{\\prime}, \\mathbf{h}_{j}^{\\prime} ; \\Phi\\right)

    where :math:`{att}_{node}` denotes the deep neural network.

    .. math::
        \\alpha_{i j}^{\\Phi}=\\operatorname{softmax}_{j}\\left(e_{i j}^{\\Phi}\\right)=\\frac{\\exp \\left(\\sigma\\left(\\mathbf{a}_{\\Phi}^{\\mathrm{T}} \\cdot\\left[\\mathbf{h}_{i}^{\\prime} \\| \\mathbf{h}_{j}^{\\prime}\\right]\\right)\\right)}{\\sum_{k \\in \\mathcal{N}_{i}^{\\Phi}} \\exp \\left(\\sigma\\left(\\mathbf{a}_{\\Phi}^{\\mathrm{T}} \\cdot\\left[\\mathbf{h}_{i}^{\\prime} \\| \\mathbf{h}_{k}^{\\prime}\\right]\\right)\\right)}

    where :math:`\\sigma` denotes the activation function, || denotes the concatenate
    operation and :math:`a_{\\Phi}` is the node-level attention vector for meta-path :math:`\\Phi`.

    .. math::
        \\mathbf{z}_{i}^{\\Phi}=\\prod_{k=1}^{K} \\sigma\\left(\\sum_{j \\in \\mathcal{N}_{i}^{\\Phi}} \\alpha_{i j}^{\\Phi} \\cdot \\mathbf{h}_{j}^{\\prime}\\right)

    where :math:`z^{\\Phi}_i` is the learned embedding of node i for the meta-path :math:`\\Phi`.
    Given the meta-path set {:math:`\\Phi_0 ,\\Phi_1,...,\\Phi_P`},after feeding node features into node-level attentionwe can obtain P groups of
    semantic-specific node embeddings, denotes as {:math:`Z_0 ,Z_1,...,Z_P`}.
    We use MetapathConv to finish Node-level Attention and Semantic-level Attention.


    Parameters
    ------------
    ntype_meta_paths_dict : dict[str, dict[str, list[etype]]]
        Dict from node type to dict from meta path name to meta path. For node classification, there is only one node type.
        For link prediction, there can be multiple node types which are source and destination node types of target links.
    in_dim : int
        Input feature dimension.
    hidden_dim : int
        Hidden layer dimension.
    out_dim : int
        Output feature dimension.
    num_heads : list[int]
        Number of attention heads.
    dropout : float
        Dropout probability.
    """

    @classmethod
    def build_model_from_args(cls, args, hg):
        ntypes = set()
        if hasattr(args, 'target_link'):
            ntypes = get_ntypes_from_canonical_etypes(args.target_link)
        elif hasattr(args, 'category'):
            ntypes.add(args.category)
        else:
            raise ValueError
        ntype_meta_paths_dict = {}
        for ntype in ntypes:
            ntype_meta_paths_dict[ntype] = {}
            for meta_path_name, meta_path in args.meta_paths_dict.items():
                if meta_path[0][0] == ntype:
                    ntype_meta_paths_dict[ntype][meta_path_name] = meta_path
        for ntype, meta_paths_dict in ntype_meta_paths_dict.items():
            if len(meta_paths_dict) == 0:
                ntype_meta_paths_dict[ntype] = extract_metapaths(ntype, hg.canonical_etypes)
        return cls(ntype_meta_paths_dict=ntype_meta_paths_dict, in_dim=args.hidden_dim, hidden_dim=args.hidden_dim, out_dim=args.out_dim, num_heads=args.num_heads, dropout=args.dropout, settings=args.settings)

    def __init__(self, ntype_meta_paths_dict, in_dim, hidden_dim, out_dim, num_heads, dropout, settings):
        super(RoHe, self).__init__()
        self.mod_dict = nn.ModuleDict()
        for ntype, meta_paths_dict in ntype_meta_paths_dict.items():
            self.mod_dict[ntype] = _HAN(meta_paths_dict, in_dim, hidden_dim, out_dim, num_heads, dropout, settings)

    def forward(self, g, h_dict):
        """
        Parameters
        -----------
        g : DGLHeteroGraph or dict[str, dict[str, DGLBlock]]
            For full batch, it is a heterogeneous graph. For mini batch, it is a dict from node type to dict from
            mata path name to DGLBlock.
        h_dict : dict[str, Tensor] or dict[str, dict[str, dict[str, Tensor]]]
            The input features. For full batch, it is a dict from node type to node features. For mini batch, it is
            a dict from node type to dict from meta path name to dict from node type to node features.

        Returns
        --------
        out_dict : dict[str, Tensor]
            The output features. Dict from node type to node features.
        """
        out_dict = {}
        for ntype, han in self.mod_dict.items():
            if isinstance(g, dict):
                if ntype not in g:
                    continue
                _g = g[ntype]
                _in_h = h_dict[ntype]
            else:
                _g = g
                _in_h = h_dict
            _out_h = han(_g, _in_h)
            for ntype, h in _out_h.items():
                out_dict[ntype] = h
        return out_dict


class RoHeGATConv(nn.Module):
    """like Graph attention layer from `Graph Attention Network
        <https://arxiv.org/pdf/1710.10903.pdf>`, but modifying the computation of \\alpha_{ij}.

        .. math::
            h_i^{(l+1)} = \\sum_{j\\in \\mathcal{N}(i)} \\alpha_{i,j} W^{(l)} h_j^{(l)}

        where :math:`\\alpha_{ij}` is the attention score bewteen node :math:`i` and
        node :math:`j`:

        .. math::
            \\alpha_{ij}^{l} &= \\mathrm{softmax_i} (e_{ij}^{l} + m_{ij}^{l})

            e_{ij}^{l} &= \\mathrm{LeakyReLU}\\left(\\vec{a}^T [W h_{i} \\| W h_{j}]\\right)

        Parameters
        ----------
        in_feats : int, or pair of ints
            Input feature size; i.e, the number of dimensions of :math:`h_i^{(l)}`.
            GATConv can be applied on homogeneous graph and unidirectional
            `bipartite graph <https://docs.dgl.ai/generated/dgl.bipartite.html?highlight=bipartite>`__.
            If the layer is to be applied to a unidirectional bipartite graph, ``in_feats``
            specifies the input feature size on both the source and destination nodes.  If
            a scalar is given, the source and destination node feature size would take the
            same value.
        out_feats : int
            Output feature size; i.e, the number of dimensions of :math:`h_i^{(l+1)}`.
        num_heads : int
            Number of heads in Multi-Head Attention.
        feat_drop : float, optional
            Dropout rate on feature. Defaults: ``0``.
        attn_drop : float, optional
            Dropout rate on attention weight. Defaults: ``0``.
        negative_slope : float, optional
            LeakyReLU angle of negative slope. Defaults: ``0.2``.
        residual : bool, optional
            If True, use residual connection. Defaults: ``False``.
        activation : callable activation function/layer or None, optional.
            If not None, applies an activation function to the updated node features.
            Default: ``None``.
    """

    def __init__(self, in_feats, out_feats, num_heads, feat_drop=0.0, attn_drop=0.0, negative_slope=0.2, residual=False, activation=None, settings={'K': 10, 'P': 0.6, 'tau': 0.1, 'Flag': 'None'}):
        super(RoHeGATConv, self).__init__()
        self._num_heads = num_heads
        self.settings = settings
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        if isinstance(in_feats, tuple):
            self.fc_src = nn.Linear(self._in_src_feats, out_feats * num_heads, bias=False)
            self.fc_dst = nn.Linear(self._in_dst_feats, out_feats * num_heads, bias=False)
        else:
            self.fc = nn.Linear(self._in_src_feats, out_feats * num_heads, bias=False)
        self.attn_l = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
        self.attn_r = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(0.0)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        if residual:
            if self._in_dst_feats != out_feats:
                self.res_fc = nn.Linear(self._in_dst_feats, num_heads * out_feats, bias=False)
            else:
                self.res_fc = Identity()
        else:
            self.register_buffer('res_fc', None)
        self.reset_parameters()
        self.activation = activation

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        if hasattr(self, 'fc'):
            nn.init.xavier_normal_(self.fc.weight, gain=gain)
        else:
            nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)

    def mask(self, attM):
        T = self.settings['T']
        indices_to_remove = attM < torch.clamp(torch.topk(attM, T)[0][..., -1, None], min=0)
        attM[indices_to_remove] = -9000000000000000.0
        return attM

    def forward(self, graph, feat):
        graph = graph.local_var()
        if isinstance(feat, tuple):
            h_src = self.feat_drop(feat[0])
            h_dst = self.feat_drop(feat[1])
            feat_src = self.fc_src(h_src).view(-1, self._num_heads, self._out_feats)
            feat_dst = self.fc_dst(h_dst).view(-1, self._num_heads, self._out_feats)
        else:
            h_src = h_dst = self.feat_drop(feat)
            feat_src = feat_dst = self.fc(h_src).view(-1, self._num_heads, self._out_feats)
        N = graph.nodes().shape[0]
        N_e = graph.edges()[0].shape[0]
        graph.srcdata.update({'ft': feat_src})
        e_trans = torch.FloatTensor(self.settings['TransM'].data).view(N_e, 1)
        e_trans = e_trans.repeat(1, 8).resize_(N_e, 8, 1)
        e = torch.cat([torch.matmul(feat_src[:, i, :].view(N, self._out_feats), feat_src[:, i, :].t().view(self._out_feats, N))[graph.edges()[0], graph.edges()[1]].view(N_e, 1) for i in range(self._num_heads)], dim=1).view(N_e, 8, 1)
        total_edge = torch.cat((graph.edges()[0].view(1, N_e), graph.edges()[1].view(1, N_e)), 0)
        attn = torch.sparse.FloatTensor(total_edge.to(self.settings['device']), torch.squeeze((e.to(self.settings['device']) * e_trans.to(self.settings['device'])).sum(-2)), torch.Size([N, N]))
        attn = self.mask(attn.to_dense()).t()
        e[attn[graph.edges()[0], graph.edges()[1]].view(N_e, 1).repeat(1, 8).view(N_e, 8, 1) < -100] = -9000000000000000.0
        graph.edata['a'] = edge_softmax(graph, e)
        graph.update_all(fn.u_mul_e('ft', 'a', 'm'), fn.sum('m', 'ft'))
        rst = graph.dstdata['ft']
        if self.res_fc is not None:
            resval = self.res_fc(h_dst).view(h_dst.shape[0], -1, self._out_feats)
            rst = rst + resval
        if self.activation:
            rst = self.activation(rst)
        return rst


class RsageLayer(nn.Module):

    def __init__(self, in_feat, out_feat, aggregator_type, rel_names, activation=None, dropout=0.0, bias=True):
        super(RsageLayer, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.aggregator_type = aggregator_type
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        self.conv = dglnn.HeteroGraphConv({rel: dgl.nn.pytorch.SAGEConv(in_feat, out_feat, aggregator_type, bias=bias) for rel in rel_names})

    def forward(self, g, h_dict):
        h_dict = self.conv(g, h_dict)
        out_put = {}
        for n_type, h in h_dict.items():
            out_put[n_type] = h.squeeze()
        return out_put


class Rsage(BaseModel):

    @classmethod
    def build_model_from_args(cls, args, hg):
        return cls(in_dim=args.in_dim, out_dim=args.out_dim, h_dim=args.hidden_dim, etypes=hg.etypes, aggregator_type=args.aggregator_type, num_hidden_layers=args.num_layers - 2, dropout=args.dropout)

    def __init__(self, in_dim, out_dim, h_dim, etypes, aggregator_type, num_hidden_layers=1, dropout=0):
        super(Rsage, self).__init__()
        self.rel_names = etypes
        self.layers = nn.ModuleList()
        self.layers.append(RsageLayer(in_dim, h_dim, aggregator_type, self.rel_names, activation=F.relu, dropout=dropout))
        for i in range(num_hidden_layers):
            self.layers.append(RsageLayer(h_dim, h_dim, aggregator_type, self.rel_names, activation=F.relu, dropout=dropout))
        self.layers.append(RsageLayer(h_dim, out_dim, aggregator_type, self.rel_names, activation=None))
        return

    def forward(self, hg, h_dict=None):
        if hasattr(hg, 'ntypes'):
            for layer in self.layers:
                h_dict = layer(hg, h_dict)
        else:
            for layer, block in zip(self.layers, hg):
                h_dict = layer(block, h_dict)
        return h_dict


class WGCN_Base(torch.nn.Module):

    def __init__(self, args):
        super(WGCN_Base, self).__init__()
        num_entities = args.num_entities
        num_relations = args.num_relations
        self.rat = args.rat
        self.wni = args.wni
        self.fa = args.final_act
        self.fb = args.final_bn
        self.fd = args.final_drop
        self.decoder_name = args.decoder
        self.num_layers = args.n_layer
        self.emb_e = torch.nn.Embedding(num_entities, args.init_emb_size, padding_idx=0)
        self.emb_rel = torch.nn.Embedding(num_relations, args.embedding_dim, padding_idx=0)
        nn.init.xavier_normal_(self.emb_e.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(self.emb_rel.weight, gain=nn.init.calculate_gain('relu'))
        if self.num_layers == 3:
            self.gc1 = GraphConvolution(args.init_emb_size, args.gc1_emb_size, num_relations, wsi=args.wsi)
            self.gc2 = GraphConvolution(args.gc1_emb_size, args.gc1_emb_size, num_relations, wsi=args.wsi)
            self.gc3 = GraphConvolution(args.gc1_emb_size, args.embedding_dim, num_relations, wsi=args.wsi)
        elif self.num_layers == 2:
            self.gc2 = GraphConvolution(args.init_emb_size, args.gc1_emb_size, num_relations, wsi=args.wsi)
            self.gc3 = GraphConvolution(args.gc1_emb_size, args.embedding_dim, num_relations, wsi=args.wsi)
        else:
            self.gc3 = GraphConvolution(args.init_emb_size, args.embedding_dim, num_relations, wsi=args.wsi)
        self.inp_drop = torch.nn.Dropout(args.input_dropout)
        self.hidden_drop = torch.nn.Dropout(args.dropout_rate)
        self.feature_map_drop = torch.nn.Dropout(args.dropout_rate)
        self.loss = torch.nn.BCELoss()
        self.conv1 = nn.Conv1d(2, args.channels, args.kernel_size, stride=1, padding=int(math.floor(args.kernel_size / 2)))
        self.bn0 = torch.nn.BatchNorm1d(2)
        self.bn1 = torch.nn.BatchNorm1d(args.channels)
        self.bn2 = torch.nn.BatchNorm1d(args.embedding_dim)
        self.register_parameter('b', nn.Parameter(torch.zeros(num_entities)))
        self.fc = torch.nn.Linear(args.embedding_dim * args.channels, args.embedding_dim)
        self.bn3 = torch.nn.BatchNorm1d(args.gc1_emb_size)
        self.bn4 = torch.nn.BatchNorm1d(args.embedding_dim)
        self.bn5 = torch.nn.BatchNorm1d(args.gc1_emb_size)
        self.bn_init = torch.nn.BatchNorm1d(args.init_emb_size)
        self.args = args
        None
        if args.decoder == 'transe':
            self.decoder = self.transe
            self.gamma = args.gamma
        elif args.decoder == 'distmult':
            self.decoder = self.distmult
            self.bias = nn.Parameter(torch.zeros(num_entities))
        elif args.decoder == 'conve':
            self.decoder = self.conve
        else:
            raise NotImplementedError

    def conve(self, e1_embedded, rel_embedded, e1_embedded_all):
        stacked_inputs = torch.cat([e1_embedded, rel_embedded], 1)
        stacked_inputs = self.bn0(stacked_inputs)
        x = self.inp_drop(stacked_inputs)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_map_drop(x)
        x = x.view(e1_embedded.shape[0], -1)
        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = torch.mm(x, e1_embedded_all.transpose(1, 0))
        pred = torch.sigmoid(x)
        return pred

    def transe(self, e1_embedded, rel_embedded, e1_embedded_all):
        obj_emb = e1_embedded + rel_embedded
        x = self.gamma - torch.norm(obj_emb - e1_embedded_all.unsqueeze(0), p=1, dim=2)
        pred = torch.sigmoid(x)
        return pred

    def distmult(self, e1_embedded, rel_embedded, e1_embedded_all):
        obj_emb = e1_embedded * rel_embedded
        x = torch.mm(obj_emb.squeeze(1), e1_embedded_all.transpose(1, 0))
        x += self.bias.expand_as(x)
        pred = torch.sigmoid(x)
        return pred

    def init(self):
        xavier_normal_(self.emb_e.weight.data)
        xavier_normal_(self.emb_rel.weight.data)
        xavier_normal_(self.gc1.weight.data)
        xavier_normal_(self.gc2.weight.data)
        xavier_normal_(self.gc3.weight.data)

    def forward(self, g, all_edge, e1, rel, entity_id):
        emb_initial = self.emb_e(entity_id)
        if self.num_layers == 3:
            x = self.gc1(g, all_edge, emb_initial)
            x = self.bn5(x)
            x = torch.tanh(x)
            x = F.dropout(x, self.args.dropout_rate, training=self.training)
        else:
            x = emb_initial
        if self.num_layers >= 2:
            x = self.gc2(g, all_edge, x)
            x = self.bn3(x)
            x = torch.tanh(x)
            x = F.dropout(x, self.args.dropout_rate, training=self.training)
        if self.num_layers >= 1:
            x = self.gc3(g, all_edge, x)
        if self.fb:
            x = self.bn4(x)
        if self.fa:
            x = torch.tanh(x)
        if self.fd:
            x = F.dropout(x, self.args.dropout_rate, training=self.training)
        e1_embedded_all = x
        e1_embedded = e1_embedded_all[e1]
        rel_embedded = self.emb_rel(rel)
        pred = self.decoder(e1_embedded, rel_embedded, e1_embedded_all)
        return pred


class SACN(BaseModel):

    @classmethod
    def build_model_from_args(cls, config):
        return cls(config)

    def __init__(self, config):
        super().__init__()
        self.model = WGCN_Base(config)

    def forward(self, *args):
        return self.model(*args)

    def extra_loss(self):
        pass


class FeatureAgg(nn.Module):

    def __init__(self, nodes_embeddings, neighbors_embeddings, nodes_profiles, neighbors_profiles, nodes_neighbors, neighbor_types, emb_size, device, fusion, att):
        super(FeatureAgg, self).__init__()
        self.emb_size = emb_size
        self.device = device
        self.nodes_embeddings = nodes_embeddings
        self.neighbors_embeddings_dict = neighbors_embeddings
        self.nodes_profiles = nodes_profiles
        self.neighbors_profiles_dict = neighbors_profiles
        self.nodes_neighbors_dict = nodes_neighbors
        self.neighbor_types_set = set(neighbor_types)
        self.num_neigh_type = len(self.neighbor_types_set)
        self.linear_1 = nn.Linear(self.emb_size, self.emb_size)
        self.linear_2 = nn.Linear(self.emb_size, self.emb_size)
        self.linear = nn.Linear(self.emb_size * 2, self.emb_size)
        self.fusion = fusion
        self.att = att
        self.w_type_att = nn.Linear(self.emb_size * self.num_neigh_type, self.num_neigh_type, bias=False)

    def forward(self, nodes):
        batch_size = len(nodes)
        nodes_emb = self.nodes_embeddings.weight[nodes]
        nodes = nodes
        nodes_profile = self.nodes_profiles[nodes]
        nodes_fusion = self.fusion(nodes_emb, nodes_profile)
        neigh_type_agg = []
        for idx, neigh_type in enumerate(self.neighbor_types_set):
            node_neighs = self.nodes_neighbors_dict[neigh_type]
            neighs = list(map(lambda y: list(y), map(lambda x: node_neighs[int(x)], nodes)))
            neigh_agg = self.aggregation(nodes_fusion, neighs, neigh_type, batch_size)
            neigh_type_agg.append(neigh_agg)
        type_agg = torch.Tensor(reduce(lambda x, y: torch.cat((x, y), 1), neigh_type_agg).cpu())
        map_type_agg = self.w_type_att(type_agg)
        att = F.softmax(map_type_agg, dim=1).view(batch_size, self.num_neigh_type, 1)
        neigh_agg_final = torch.matmul(torch.transpose(type_agg.view(batch_size, -1, self.emb_size), dim0=1, dim1=2), att).squeeze()
        neigh_agg_final = F.relu(self.linear_2(neigh_agg_final))
        combined_feature = torch.cat([nodes_fusion, neigh_agg_final], dim=1)
        combined_feature = F.relu(self.linear(combined_feature))
        return combined_feature, att

    def aggregation(self, nodes_fusion, nodes_neighbors, neigh_type, batch_size):
        neighbors_embeddings = self.neighbors_embeddings_dict[neigh_type]
        neighbors_profiles = self.neighbors_profiles_dict[neigh_type]
        neighs_fusion = list(map(lambda x: self.fusion(neighbors_embeddings.weight[x], neighbors_profiles[x]), nodes_neighbors))
        attention_list = list(map(lambda idx: self.att(nodes_fusion[idx], neighs_fusion[idx], len(nodes_neighbors[idx])), range(batch_size)))
        neigh_feature_matrix = torch.stack([torch.mm(neighs_fusion[idx].t().to(self.device), attention_list[idx].to(self.device)) for idx in range(batch_size)]).reshape(batch_size, self.emb_size)
        combined_feature = F.relu(self.linear_1(neigh_feature_matrix))
        return combined_feature


class Fusion(nn.Module):

    def __init__(self, embedding_dims, profile_dim):
        super(Fusion, self).__init__()
        self.embed_dim = embedding_dims
        self.profile_dim = profile_dim
        self.w_1 = nn.Linear(self.embed_dim * 2, self.embed_dim)
        self.w_2 = nn.Linear(self.embed_dim, self.embed_dim)
        self.linear = nn.Linear(profile_dim, self.embed_dim)

    def forward(self, embedding, profile):
        return embedding


class InfluenceProp(nn.Module):

    def __init__(self, social_rel, emb_size, user_embs, user_profiles, fusion, att):
        super(InfluenceProp, self).__init__()
        self.social_rel = social_rel
        self.emb_size = emb_size
        self.user_embs = user_embs
        self.user_profiles = user_profiles
        self.linear = nn.Linear(2 * self.emb_size, self.emb_size)
        self.fusion = fusion
        self.att = att
        self.w_c1 = nn.Linear(self.emb_size * 2, self.emb_size)
        self.w_c2 = nn.Linear(self.emb_size, self.emb_size)

    def forward(self, users, u_embs, items, i_embs, act_users):
        batch_size = len(users)
        act_u_fusion = list(map(lambda x: self.fusion(self.user_embs.weight[x], self.user_profiles[x]), act_users))
        coupling_fea = list(map(lambda idx: torch.cat((act_u_fusion[idx], i_embs[idx].repeat(len(act_u_fusion[idx]), 1)), 1), range(batch_size)))
        coupling_fea = list(map(lambda idx: F.relu(self.w_c1(coupling_fea[idx])), range(batch_size)))
        coupling_fea = list(map(lambda idx: F.relu(self.w_c2(coupling_fea[idx])), range(batch_size)))
        attention_list = list(map(lambda idx: self.att(u_embs[idx], coupling_fea[idx], len(act_users[idx])), range(batch_size)))
        neigh_feature_matrix_coupling = torch.stack([torch.mm(coupling_fea[idx].t(), attention_list[idx]) for idx in range(batch_size)]).reshape(batch_size, self.emb_size)
        combined_feature = neigh_feature_matrix_coupling
        return combined_feature, attention_list


class SIAN(BaseModel):

    @classmethod
    def build_model_from_args(cls, args, dataset):
        return SIAN(args.user_num, args.item_num, dataset.user_profile, dataset.item_profile, dataset.u_items, dataset.i_users, dataset.social_relation, args.emb_size, args.profile_size, args.device)

    def __init__(self, user_num, item_num, user_profile, item_profile, user_items, item_users, social_rel, emb_size=64, profile_size=64, device='cpu'):
        super(SIAN, self).__init__()
        self.emb_size = emb_size
        self.profile_size = profile_size
        self.device = device
        self.user_embedding = nn.Embedding(user_num, self.emb_size)
        self.item_embedding = nn.Embedding(item_num, self.emb_size)
        self.user_profile = user_profile
        self.item_profile = item_profile
        self.user_items = user_items
        self.item_users = item_users
        self.social_rel = social_rel
        self.fusion = Fusion(self.emb_size, self.profile_size)
        self.att = Attention(self.emb_size)
        self.item_feat_agg = FeatureAgg(self.item_embedding, {'user': self.user_embedding}, self.item_profile, {'user': self.user_profile}, {'user': self.item_users}, ['user'], self.emb_size, self.device, self.fusion, self.att)
        self.user_feat_social_agg = FeatureAgg(self.user_embedding, {'item': self.item_embedding, 'user': self.user_embedding}, self.user_profile, {'item': self.item_profile, 'user': self.user_profile}, {'item': self.user_items, 'user': self.social_rel}, ['item', 'user'], self.emb_size, self.device, self.fusion, self.att)
        self.social_inf_prop = InfluenceProp(self.social_rel, self.emb_size, self.user_embedding, self.user_profile, self.fusion, self.att)
        self.w_u1 = nn.Linear(self.emb_size, self.emb_size)
        self.w_u2 = nn.Linear(self.emb_size, self.emb_size)
        self.w_f1 = nn.Linear(self.emb_size, self.emb_size)
        self.w_f2 = nn.Linear(self.emb_size, self.emb_size)
        self.w_i1 = nn.Linear(self.emb_size, self.emb_size)
        self.w_i2 = nn.Linear(self.emb_size, self.emb_size)
        self.w_ui1 = nn.Linear(self.emb_size * 2, self.emb_size)
        self.w_ui2 = nn.Linear(self.emb_size, 16)
        self.w_ui3 = nn.Linear(16, 1)
        self.w_ufi1 = nn.Linear(self.emb_size * 3, self.emb_size)
        self.w_ufi2 = nn.Linear(self.emb_size, 16)
        self.w_ufi3 = nn.Linear(16, 1)
        self.bn1 = nn.BatchNorm1d(self.emb_size, momentum=0.5)
        self.bn2 = nn.BatchNorm1d(self.emb_size, momentum=0.5)
        self.bn3 = nn.BatchNorm1d(self.emb_size, momentum=0.5)
        self.bnf = nn.BatchNorm1d(self.emb_size, momentum=0.5)
        self.bn4 = nn.BatchNorm1d(16, momentum=0.5)
        self.item_fea_att_analysis = []
        self.user_fea_att_analysis = []
        self.inf_att_analysis = []

    def forward(self, users, items, act_users):
        i_embs, item_fea_att_list = self.item_feat_agg(items)
        u_embs, user_fea_att_list = self.user_feat_social_agg(users)
        u_inf, inf_att_list = self.social_inf_prop(users, u_embs, items, i_embs, act_users)
        x_u = F.relu(self.bn1(self.w_u1(u_embs)))
        x_u = F.dropout(x_u, training=self.training)
        x_u = self.w_u2(x_u)
        x_f = F.relu(self.bnf(self.w_f1(u_inf)))
        x_f = F.dropout(x_f, training=self.training)
        x_f = self.w_f2(x_f)
        x_i = F.relu(self.bn2(self.w_i1(i_embs)))
        x_i = F.dropout(x_i, training=self.training)
        x_i = self.w_i2(x_i)
        x_ufi = torch.cat((x_u, x_f, x_i), 1)
        x = F.relu(self.bn3(self.w_ufi1(x_ufi)))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.bn4(self.w_ufi2(x)))
        x = F.dropout(x, training=self.training)
        scores = torch.sigmoid(self.w_ufi3(x))
        return scores.squeeze()


class NodeEncoder(torch.nn.Module):

    def __init__(self, base_embedding_dim, num_nodes, pretrained_node_embedding_tensor, is_pre_trained):
        super().__init__()
        self.pretrained_node_embedding_tensor = pretrained_node_embedding_tensor
        self.base_embedding_dim = base_embedding_dim
        if not is_pre_trained:
            self.base_embedding_layer = torch.nn.Embedding(num_nodes, base_embedding_dim)
            self.base_embedding_layer.weight.data.uniform_(-1, 1)
        else:
            self.base_embedding_layer = torch.nn.Embedding.from_pretrained(pretrained_node_embedding_tensor)

    def forward(self, node_id):
        node_id = torch.LongTensor([int(node_id)])
        x_base = self.base_embedding_layer(node_id)
        return x_base


class GCNGraphEncoder(torch.nn.Module):

    def __init__(self, G, pretrained_node_embedding_tensor, is_pre_trained, base_embedding_dim, max_length):
        super().__init__()
        self.g = G
        self.base_embedding_dim = base_embedding_dim
        self.max_length = max_length
        self.no_nodes = self.g.num_nodes()
        self.no_relations = self.g.num_edges()
        self.node_embedding = NodeEncoder(base_embedding_dim, self.no_nodes, pretrained_node_embedding_tensor, is_pre_trained)
        self.special_tokens = {'[PAD]': 0, '[MASK]': 1}
        self.special_embed = torch.nn.Embedding(len(self.special_tokens), base_embedding_dim)
        self.special_embed.weight.data.uniform_(-1, 1)

    def forward(self, subgraphs_list, masked_nodes):
        num_subgraphs = len(subgraphs_list)
        node_emb = torch.zeros(num_subgraphs, self.max_length + 1, self.base_embedding_dim)
        for ii, subgraph in enumerate(subgraphs_list):
            masked_set = masked_nodes[ii]
            for node in subgraph.nodes():
                node_id = subgraph.ndata[dgl.NID][int(node)]
                if node_id not in masked_set:
                    node_emb[ii][node] = self.node_embedding(int(node_id))
        special_tokens_embed = {}
        for token in self.special_tokens:
            node_id = Variable(torch.LongTensor([self.special_tokens[token]]))
            tmp_embed = self.special_embed(node_id)
            special_tokens_embed[self.special_tokens[token] + self.no_nodes] = {'token': token, 'embed': tmp_embed}
        return node_emb


class ScaledDotProductAttention(torch.nn.Module):

    def __init__(self, d_k):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k

    def forward(self, Q, K, V, attn_mask=None):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k)
        scores.masked_fill_(attn_mask == True, -1000000000.0)
        attn = torch.nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context, attn


class MultiHeadAttention(torch.nn.Module):

    def __init__(self, d_model, d_k, d_v, n_heads):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model
        self.W_Q = torch.nn.Linear(d_model, d_k * n_heads)
        self.W_K = torch.nn.Linear(d_model, d_k * n_heads)
        self.W_V = torch.nn.Linear(d_model, d_v * n_heads)
        self.scaled_dot_prod_attn = ScaledDotProductAttention(d_k)
        self.wrap = torch.nn.Linear(self.n_heads * self.d_v, self.d_model)
        self.layerNorm = torch.nn.LayerNorm(self.d_model)

    def forward(self, Q, K, V, attn_mask=None):
        residual, batch_size = Q, Q.size(0)
        q_s = self.W_Q(Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k_s = self.W_K(K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v_s = self.W_V(V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1, 2)
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)
        context, attn = self.scaled_dot_prod_attn(q_s, k_s, v_s, attn_mask=attn_mask)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_v)
        output = self.wrap(context)
        return self.layerNorm(output + residual), attn


def gelu(x):
    """"Implementation of the gelu activation function by Hugging Face."""
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class PoswiseFeedForwardNet(torch.nn.Module):

    def __init__(self, d_model, d_ff):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc1 = torch.nn.Linear(d_model, d_ff)
        self.fc2 = torch.nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.fc2(gelu(self.fc1(x)))


class EncoderLayer(torch.nn.Module):

    def __init__(self, d_model, d_k, d_v, d_ff, n_heads):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention(d_model, d_k, d_v, n_heads)
        self.pos_ffn = PoswiseFeedForwardNet(d_model, d_ff)

    def forward(self, enc_inputs, enc_self_attn_mask):
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask)
        enc_outputs = self.pos_ffn(enc_outputs)
        return enc_outputs, attn


def get_attn_pad_mask(subgraph_list, pad_id, max_len):
    batch_size = len(subgraph_list)
    len_q = max_len
    pad_attn_mask = []
    for itm in subgraph_list:
        tmp_mask = []
        for sub in itm.ndata[dgl.NID]:
            if sub == pad_id:
                tmp_mask.append(True)
            else:
                tmp_mask.append(False)
        if len(tmp_mask) < max_len:
            tmp_mask = tmp_mask + [True] * (max_len - len(tmp_mask))
        pad_attn_mask.append(tmp_mask)
    pad_attn_mask = Variable(torch.ByteTensor(pad_attn_mask)).unsqueeze(1)
    pad_attn_mask = pad_attn_mask
    return pad_attn_mask.expand(batch_size, len_q, len_q)


class SLiCE(BaseModel):

    @classmethod
    def build_model_from_args(cls, args, hg):
        return cls(G=hg, pretrained_node_embedding_tensor=None, args=args)

    def load_pretrained_node2vec(self, filename, base_emb_dim):
        """
        loads embeddings from node2vec style file, where each line is
        nodeid node_embedding
        returns tensor containing node_embeddings
        for graph nodes 0 to n-1
        """
        node_embeddings = torch.empty(self.g.num_nodes(), 100)
        with open(filename, 'r') as f:
            header = f.readline()
            emb_dim = int(header.strip().split()[1])
            for line in f:
                arr = line.strip().split()
                graph_node_id = arr[0]
                node_emb = [float(x) for x in arr[1:]]
                vocab_id = int(graph_node_id)
                if vocab_id >= 0:
                    node_embeddings[vocab_id] = torch.tensor(node_emb)
        out = node_embeddings
        None
        return out

    def __init__(self, G, args, pretrained_node_embedding_tensor, num_layers=6, d_model=200, d_k=64, d_v=64, d_ff=200 * 4, n_heads=4, is_pre_trained=False, base_embedding_dim=200, max_length=6, num_gcn_layers=2, node_edge_composition_func='mult', get_embeddings=False, fine_tuning_layer=False):
        super().__init__()
        self.g = G
        self.num_layers = num_layers
        self.d_model = d_model
        self.max_length = max_length
        self.get_embeddings = get_embeddings
        self.node_edge_composition_func = node_edge_composition_func
        self.fine_tuning_layer = fine_tuning_layer
        self.no_nodes = G.num_nodes()
        self.n_pred = args.n_pred
        if not os.path.exists(args.pretrained_embeddings):
            None
            walks = []
            for _ in range(10):
                nodes = list(G.nodes())
                random.shuffle(nodes)
                walk = dgl.sampling.node2vec_random_walk(G, torch.tensor(nodes), 1, 1, walk_length=80 - 1).tolist()
                walks.extend(walk)
            walks = [list(map(str, walk)) for walk in walks]
            model = Word2Vec(walks, window=10, min_count=0, sg=1, workers=8)
            model.wv.save_word2vec_format(args.pretrained_embeddings)
        pretrained_node_embedding_tensor = self.load_pretrained_node2vec(args.pretrained_embeddings, base_embedding_dim)
        self.gcn_graph_encoder = GCNGraphEncoder(G, pretrained_node_embedding_tensor, is_pre_trained, base_embedding_dim, max_length)
        self.layers = torch.nn.ModuleList([EncoderLayer(d_model, d_k, d_v, d_ff, n_heads) for _ in range(num_layers)])
        self.linear = torch.nn.Linear(d_model, d_model)
        self.norm = torch.nn.LayerNorm(d_model)
        self.decoder = torch.nn.Linear(self.d_model, self.no_nodes)

    def set_fine_tuning(self):
        self.fine_tuning_layer = True

    def GCN_MaskGeneration(self, subgraph_sequences):
        n_pred = self.n_pred
        masked_nodes = []
        masked_position = []
        for subgraph in subgraph_sequences:
            num_nodes = subgraph.num_nodes()
            mask_index = random.sample(range(num_nodes), n_pred)
            subgraph_masked_nodes = []
            subgraph_masked_position = []
            for i in range(num_nodes):
                if i in mask_index:
                    subgraph_masked_nodes.append(subgraph.ndata[dgl.NID][i])
                    subgraph_masked_position.append(i)
            masked_nodes.append(subgraph_masked_nodes)
            masked_position.append(subgraph_masked_position)
        return torch.tensor(masked_nodes), torch.tensor(masked_position)

    def forward(self, subgraph_list):
        if self.fine_tuning_layer:
            masked_nodes = Variable(torch.LongTensor([[] for ii in range(len(subgraph_list))]))
        else:
            masked_nodes, masked_pos = self.GCN_MaskGeneration(subgraph_list)
        node_emb = self.gcn_graph_encoder(subgraph_list, masked_nodes)
        output = node_emb
        enc_self_attn_mask = get_attn_pad_mask(subgraph_list, self.no_nodes, self.max_length + 1)
        for layer in self.layers:
            output, enc_self_attn = layer(output, enc_self_attn_mask)
            try:
                layer_output = torch.cat((layer_output, output.unsqueeze(1)), 1)
            except NameError:
                layer_output = output.unsqueeze(1)
            if self.fine_tuning_layer:
                try:
                    att_output = torch.cat((att_output, enc_self_attn.unsqueeze(0)), 0)
                except NameError:
                    att_output = enc_self_attn.unsqueeze(0)
        if self.num_layers == 0:
            layer_output = output.unsqueeze(1)
            att_output = 'NA'
        if self.fine_tuning_layer:
            return output, layer_output, att_output
        else:
            masked_pos = masked_pos[:, :, None].expand(-1, -1, output.size(-1))
            h_masked = torch.gather(output, 1, masked_pos)
            h_masked = self.norm(gelu(self.linear(h_masked)))
            pred_score = self.decoder(h_masked)
            if self.get_embeddings:
                return pred_score, masked_nodes, output
            else:
                return pred_score, masked_nodes


class SLiCEFinetuneLayer(torch.nn.Module):

    @classmethod
    def build_model_from_args(cls, args):
        return cls(d_model=args.d_model, ft_d_ff=args.ft_d_ff, ft_layer=args.ft_layer, ft_drop_rate=args.ft_drop_rate, ft_input_option=args.ft_input_option, num_layers=args.num_layers)

    def __init__(self, d_model, ft_d_ff, ft_layer, ft_drop_rate, ft_input_option, num_layers):
        super().__init__()
        self.d_model = d_model
        self.ft_layer = ft_layer
        self.ft_input_option = ft_input_option
        self.num_layers = num_layers
        if ft_input_option in ['last', 'last4_sum']:
            cnt_layers = 1
        elif ft_input_option in ['last4_cat']:
            cnt_layers = 4
        if self.num_layers == 0:
            cnt_layers = 1
        if self.ft_layer == 'linear':
            self.ft_decoder = torch.nn.Linear(d_model * cnt_layers, d_model)
        elif self.ft_layer == 'ffn':
            self.ffn1 = torch.nn.Linear(d_model * cnt_layers, ft_d_ff)
            None
            self.dropout = torch.nn.Dropout(ft_drop_rate)
            self.ffn2 = torch.nn.Linear(ft_d_ff, d_model)

    def forward(self, graphbert_layer_output):
        """
        graphbert_output = batch_sz * [CLS, source, target, relation, SEP] *
        [emb_size]
        """
        if self.ft_input_option == 'last':
            graphbert_output = graphbert_layer_output[:, -1, :, :].squeeze(1)
            source_embedding = graphbert_output[:, 0, :].unsqueeze(1)
            destination_embedding = graphbert_output[:, 1, :].unsqueeze(1)
        else:
            no_layers = graphbert_layer_output.size(1)
            if no_layers == 1:
                start_layer = 0
            else:
                start_layer = no_layers - 4
            for ii in range(start_layer, no_layers):
                source_embed = graphbert_layer_output[:, ii, 0, :].unsqueeze(1)
                destination_embed = graphbert_layer_output[:, ii, 1, :].unsqueeze(1)
                if self.ft_input_option == 'last4_cat':
                    try:
                        source_embedding = torch.cat((source_embedding, source_embed), 2)
                        destination_embedding = torch.cat((destination_embedding, destination_embed), 2)
                    except:
                        source_embedding = source_embed
                        destination_embedding = destination_embed
                elif self.ft_input_option == 'last4_sum':
                    try:
                        source_embedding = torch.add(source_embedding, 1, source_embed)
                        destination_embedding = torch.add(destination_embedding, 1, destination_embed)
                    except:
                        source_embedding = source_embed
                        destination_embedding = destination_embed
        if self.ft_layer == 'linear':
            src_embedding = self.ft_decoder(source_embedding)
            dst_embedding = self.ft_decoder(destination_embedding)
        elif self.ft_layer == 'ffn':
            src_embedding = torch.relu(self.dropout(self.ffn1(source_embedding)))
            src_embedding = self.ffn2(src_embedding)
            dst_embedding = torch.relu(self.dropout(self.ffn1(destination_embedding)))
            dst_embedding = self.ffn2(dst_embedding)
        dst_embedding = dst_embedding.transpose(1, 2)
        pred_score = torch.bmm(src_embedding, dst_embedding).squeeze(1)
        pred_score = torch.sigmoid(pred_score)
        return pred_score, src_embedding, dst_embedding.transpose(1, 2)


class Transformer(nn.Module):

    def __init__(self, n_channels, att_drop=0.0, act='none', num_heads=1):
        super(Transformer, self).__init__()
        self.n_channels = n_channels
        self.num_heads = num_heads
        assert self.n_channels % (self.num_heads * 4) == 0
        self.query = nn.Linear(self.n_channels, self.n_channels // 4)
        self.key = nn.Linear(self.n_channels, self.n_channels // 4)
        self.value = nn.Linear(self.n_channels, self.n_channels)
        self.gamma = nn.Parameter(torch.tensor([0.0]))
        self.att_drop = nn.Dropout(att_drop)
        if act == 'sigmoid':
            self.act = torch.nn.Sigmoid()
        elif act == 'relu':
            self.act = torch.nn.ReLU()
        elif act == 'leaky_relu':
            self.act = torch.nn.LeakyReLU(0.2)
        elif act == 'none':
            self.act = lambda x: x
        else:
            assert 0, f'Unrecognized activation function {act} for class Transformer'

    def reset_parameters(self):

        def xavier_uniform_(tensor, gain=1.0):
            fan_in, fan_out = tensor.size()[-2:]
            std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
            a = math.sqrt(3.0) * std
            return torch.nn.init._no_grad_uniform_(tensor, -a, a)
        gain = nn.init.calculate_gain('relu')
        xavier_uniform_(self.query.weight, gain=gain)
        xavier_uniform_(self.key.weight, gain=gain)
        xavier_uniform_(self.value.weight, gain=gain)
        nn.init.zeros_(self.query.bias)
        nn.init.zeros_(self.key.bias)
        nn.init.zeros_(self.value.bias)

    def forward(self, x, mask=None):
        B, M, C = x.size()
        H = self.num_heads
        if mask is not None:
            assert mask.size() == torch.Size((B, M))
        f = self.query(x).view(B, M, H, -1).permute(0, 2, 1, 3)
        g = self.key(x).view(B, M, H, -1).permute(0, 2, 3, 1)
        h = self.value(x).view(B, M, H, -1).permute(0, 2, 1, 3)
        beta = F.softmax(self.act(f @ g / math.sqrt(f.size(-1))), dim=-1)
        beta = self.att_drop(beta)
        if mask is not None:
            beta = beta * mask.view(B, 1, 1, M)
            beta = beta / (beta.sum(-1, keepdim=True) + 1e-12)
        o = self.gamma * (beta @ h)
        return o.permute(0, 2, 1, 3).reshape((B, M, C)) + x


class Conv1d1x1(nn.Module):

    def __init__(self, cin, cout, groups, bias=True, cformat='channel-first'):
        super(Conv1d1x1, self).__init__()
        self.cin = cin
        self.cout = cout
        self.groups = groups
        self.cformat = cformat
        if not bias:
            self.bias = None
        if self.groups == 1:
            self.W = nn.Parameter(torch.randn(self.cin, self.cout))
            if bias:
                self.bias = nn.Parameter(torch.zeros(1, self.cout))
        else:
            self.W = nn.Parameter(torch.randn(self.groups, self.cin, self.cout))
            if bias:
                self.bias = nn.Parameter(torch.zeros(self.groups, self.cout))

    def reset_parameters(self):

        def xavier_uniform_(tensor, gain=1.0):
            fan_in, fan_out = tensor.size()[-2:]
            std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
            a = math.sqrt(3.0) * std
            return torch.nn.init._no_grad_uniform_(tensor, -a, a)
        gain = nn.init.calculate_gain('relu')
        xavier_uniform_(self.W, gain=gain)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x):
        if self.groups == 1:
            if self.cformat == 'channel-first':
                return torch.einsum('bcm,mn->bcn', x, self.W) + self.bias
            elif self.cformat == 'channel-last':
                return torch.einsum('bmc,mn->bnc', x, self.W) + self.bias.T
            else:
                assert False
        elif self.cformat == 'channel-first':
            return torch.einsum('bcm,cmn->bcn', x, self.W) + self.bias
        elif self.cformat == 'channel-last':
            return torch.einsum('bmc,cmn->bnc', x, self.W) + self.bias.T
        else:
            assert False


class L2Norm(nn.Module):

    def __init__(self, dim):
        super(L2Norm, self).__init__()
        self.dim = dim

    def forward(self, x):
        return F.normalize(x, p=2, dim=self.dim)


class SeHGNN(BaseModel):
    """
    This is a model SimpleHGN from `Simple and Efficient Heterogeneous Graph Neural Network
    <https://doi.org/10.48550/arXiv.2207.02547>`__
    This model is a metapath-based model. It put the neighbor aggregation in the preprocessing step, and using
    the single-layer structure and long metapaths. It performed over the state-of-the-arts on both accuracy and training speed.

    the neighbor aggregation

    .. math::
        \\mathrm{X}^{P} = \\hat{A}_{c,c_{1}}\\hat{A}_{c_{1},c_{2}}...\\hat{A}_{c_{l-1},c_{l}} \\mathrm{X}^{c_{l}}

    feature projection
    
    .. math::
        {\\mathrm{H}^{'}}^{P} = MLP_{P}(\\mathrm{X}^{P})

    semantic fusion (transformer):

    .. math::
        q^{\\mathcal{P}_{i}}=W_{Q} h^{\\prime \\mathcal{P}_{i}}, k^{\\mathcal{P}_{i}}=W_{K} h^{\\prime \\mathcal{P}_{i}}, v^{\\mathcal{P}_{i}}=W_{V} h^{\\prime \\mathcal{P}_{i}}, \\mathcal{P}_{i} \\in \\Phi_{X} \\\\

    .. math::
        \\alpha_{\\left(\\mathcal{P}_{i}, \\mathcal{P}_{j}\\right)}=\\frac{\\exp \\left(q^{\\mathcal{P}_{i}} \\cdot k^{{\\mathcal{P}_{j}}^{T}}\\right)}{\\sum_{\\mathcal{P}_{t} \\in \\Phi_{X}} \\exp \\left(q^{\\mathcal{P}_{i}} \\cdot k^{{\\mathcal{P}_{t}}^{T}}\\right)}

    .. math::
        h^{\\mathcal{P}_{i}}=\\beta \\sum_{\\mathcal{P}_{j} \\in \\Phi_{X}} \\alpha_{\\left(\\mathcal{P}_{i}, \\mathcal{P}_{j}\\right)} v^{\\mathcal{P}_{j}}+h^{\\prime \\mathcal{P}_{i}}
    """

    @classmethod
    def build_model_from_args(cls, args):
        return cls(args)

    def __init__(self, args):
        super(SeHGNN, self).__init__()
        self.data_size = args.data_size
        self.nfeat = args.nfeat
        self.hidden = args.hidden
        self.nclass = args.nclass
        self.num_feats = args.num_feats
        self.num_label_feats = args.num_label_feats
        self.dropout = args.dropout
        self.input_drop = args.input_drop
        self.att_drop = args.att_drop
        self.label_drop = args.label_drop
        self.n_layers_1 = args.n_layers_1
        self.n_layers_2 = args.n_layers_2
        self.n_layers_3 = args.n_layers_3
        self.act = args.act
        self.residual = args.residual
        self.bns = args.bns
        self.label_bns = args.label_bns
        self.label_residual = args.label_residual
        self.dataset = args.dataset
        self.tgt_key = args.tgt_key
        if any([(v != self.nfeat) for k, v in self.data_size.items()]):
            self.embedings = nn.ParameterDict({})
            for k, v in self.data_size.items():
                if v != self.nfeat:
                    self.embedings[k] = nn.Parameter(torch.Tensor(v, self.nfeat).uniform_(-0.5, 0.5))
        else:
            self.embedings = None
        self.feat_project_layers = nn.Sequential(Conv1d1x1(self.nfeat, self.hidden, self.num_feats, bias=True, cformat='channel-first'), nn.LayerNorm([self.num_feats, self.hidden]), nn.PReLU(), nn.Dropout(self.dropout), Conv1d1x1(self.hidden, self.hidden, self.num_feats, bias=True, cformat='channel-first'), nn.LayerNorm([self.num_feats, self.hidden]), nn.PReLU(), nn.Dropout(self.dropout))
        if self.num_label_feats > 0:
            self.label_feat_project_layers = nn.Sequential(Conv1d1x1(self.nclass, self.hidden, self.num_label_feats, bias=True, cformat='channel-first'), nn.LayerNorm([self.num_label_feats, self.hidden]), nn.PReLU(), nn.Dropout(self.dropout), Conv1d1x1(self.hidden, self.hidden, self.num_label_feats, bias=True, cformat='channel-first'), nn.LayerNorm([self.num_label_feats, self.hidden]), nn.PReLU(), nn.Dropout(self.dropout))
        else:
            self.label_feat_project_layers = None
        self.semantic_aggr_layers = Transformer(self.hidden, self.att_drop, self.act)
        self.concat_project_layer = nn.Linear((self.num_feats + self.num_label_feats) * self.hidden, self.hidden)
        if self.residual:
            self.res_fc = nn.Linear(self.nfeat, self.hidden, bias=False)

        def add_nonlinear_layers(nfeats, dropout, bns=False):
            if bns:
                return [nn.BatchNorm1d(self.hidden), nn.PReLU(), nn.Dropout(dropout)]
            else:
                return [nn.PReLU(), nn.Dropout(dropout)]
        lr_output_layers = [([nn.Linear(self.hidden, self.hidden, bias=not self.bns)] + add_nonlinear_layers(self.hidden, self.dropout, self.bns)) for _ in range(self.n_layers_2 - 1)]
        self.lr_output = nn.Sequential(*([ele for li in lr_output_layers for ele in li] + [nn.Linear(self.hidden, self.nclass, bias=False), nn.BatchNorm1d(self.nclass)]))
        if self.label_residual:
            label_fc_layers = [([nn.Linear(self.hidden, self.hidden, bias=not self.bns)] + add_nonlinear_layers(self.hidden, self.dropout, self.bns)) for _ in range(self.n_layers_3 - 2)]
            self.label_fc = nn.Sequential(*([nn.Linear(self.nclass, self.hidden, bias=not self.bns)] + add_nonlinear_layers(self.hidden, self.dropout, self.bns) + [ele for li in label_fc_layers for ele in li] + [nn.Linear(self.hidden, self.nclass, bias=True)]))
            self.label_drop = nn.Dropout(self.label_drop)
        self.prelu = nn.PReLU()
        self.dropout = nn.Dropout(self.dropout)
        self.input_drop = nn.Dropout(self.input_drop)
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        for layer in self.feat_project_layers:
            if isinstance(layer, Conv1d1x1):
                layer.reset_parameters()
        if self.label_feat_project_layers is not None:
            for layer in self.label_feat_project_layers:
                if isinstance(layer, Conv1d1x1):
                    layer.reset_parameters()
        if self.dataset != 'products':
            nn.init.xavier_uniform_(self.concat_project_layer.weight, gain=gain)
            nn.init.zeros_(self.concat_project_layer.bias)
        if self.residual:
            nn.init.xavier_uniform_(self.res_fc.weight, gain=gain)
        for layer in self.lr_output:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=gain)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
        if self.label_residual:
            for layer in self.label_fc:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight, gain=gain)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

    def forward(self, fk):
        """

        Parameters
        ----------
        fk = feats_dict, layer_feats_dict, label_emb

        """
        feats_dict, layer_feats_dict, label_emb = fk['0'], fk['1'], fk['2']
        if self.embedings is not None:
            for k, v in feats_dict.items():
                if k in self.embedings:
                    feats_dict[k] = v @ self.embedings[k]
        tgt_feat = self.input_drop(feats_dict[self.tgt_key])
        B = num_node = tgt_feat.size(0)
        x = self.input_drop(torch.stack(list(feats_dict.values()), dim=1))
        x = self.feat_project_layers(x)
        if self.label_feat_project_layers is not None:
            label_feats = self.input_drop(torch.stack(list(layer_feats_dict.values()), dim=1))
            label_feats = self.label_feat_project_layers(label_feats)
            x = torch.cat((x, label_feats), dim=1)
        x = self.semantic_aggr_layers(x)
        if self.dataset == 'products':
            x = x[:, :, 0].contiguous()
        else:
            x = self.concat_project_layer(x.reshape(B, -1))
        if self.residual:
            x = x + self.res_fc(tgt_feat)
        x = self.dropout(self.prelu(x))
        x = self.lr_output(x)
        if self.label_residual:
            x = x + self.label_fc(self.label_drop(label_emb))
        return x


class SimpleHGN(BaseModel):
    """
    This is a model SimpleHGN from `Are we really making much progress? Revisiting, benchmarking, and
    refining heterogeneous graph neural networks
    <https://dl.acm.org/doi/pdf/10.1145/3447548.3467350>`__

    The model extend the original graph attention mechanism in GAT by including edge type information into attention calculation.

    Calculating the coefficient:
    
    .. math::
        \\alpha_{ij} = \\frac{exp(LeakyReLU(a^T[Wh_i||Wh_j||W_r r_{\\psi(<i,j>)}]))}{\\Sigma_{k\\in\\mathcal{E}}{exp(LeakyReLU(a^T[Wh_i||Wh_k||W_r r_{\\psi(<i,k>)}]))}} \\quad (1)
    
    Residual connection including Node residual:
    
    .. math::
        h_i^{(l)} = \\sigma(\\Sigma_{j\\in \\mathcal{N}_i} {\\alpha_{ij}^{(l)}W^{(l)}h_j^{(l-1)}} + h_i^{(l-1)}) \\quad (2)
    
    and Edge residual:
        
    .. math::
        \\alpha_{ij}^{(l)} = (1-\\beta)\\alpha_{ij}^{(l)}+\\beta\\alpha_{ij}^{(l-1)} \\quad (3)
        
    Multi-heads:
    
    .. math::
        h^{(l+1)}_j = \\parallel^M_{m = 1}h^{(l + 1, m)}_j \\quad (4)
    
    Residual:
    
        .. math::
            h^{(l+1)}_j = h^{(l)}_j + \\parallel^M_{m = 1}h^{(l + 1, m)}_j \\quad (5)
    
    Parameters
    ----------
    edge_dim: int
        the edge dimension
    num_etypes: int
        the number of the edge type
    in_dim: int
        the input dimension
    hidden_dim: int
        the output dimension
    num_classes: int
        the number of the output classes
    num_layers: int
        the number of layers we used in the computing
    heads: list
        the list of the number of heads in each layer
    feat_drop: float
        the feature drop rate
    negative_slope: float
        the negative slope used in the LeakyReLU
    residual: boolean
        if we need the residual operation
    beta: float
        the hyperparameter used in edge residual
    ntypes: list
        the list of node type
    """

    @classmethod
    def build_model_from_args(cls, args, hg):
        heads = [args.num_heads] * args.num_layers + [1]
        return cls(args.edge_dim, len(hg.etypes), [args.hidden_dim], args.hidden_dim // args.num_heads, args.out_dim, args.num_layers, heads, args.feats_drop_rate, args.slope, True, args.beta, hg.ntypes)

    def __init__(self, edge_dim, num_etypes, in_dim, hidden_dim, num_classes, num_layers, heads, feat_drop, negative_slope, residual, beta, ntypes):
        super(SimpleHGN, self).__init__()
        self.ntypes = ntypes
        self.num_layers = num_layers
        self.hgn_layers = nn.ModuleList()
        self.activation = F.elu
        self.hgn_layers.append(SimpleHGNConv(edge_dim, in_dim[0], hidden_dim, heads[0], num_etypes, feat_drop, negative_slope, False, self.activation, beta=beta))
        for l in range(1, num_layers - 1):
            self.hgn_layers.append(SimpleHGNConv(edge_dim, hidden_dim * heads[l - 1], hidden_dim, heads[l], num_etypes, feat_drop, negative_slope, residual, self.activation, beta=beta))
        self.hgn_layers.append(SimpleHGNConv(edge_dim, hidden_dim * heads[-2], num_classes, heads[-1], num_etypes, feat_drop, negative_slope, residual, None, beta=beta))

    def forward(self, hg, h_dict):
        """
        The forward part of the SimpleHGN.
        
        Parameters
        ----------
        hg : object
            the dgl heterogeneous graph
        h_dict: dict
            the feature dict of different node types
            
        Returns
        -------
        dict
            The embeddings after the output projection.
        """
        if hasattr(hg, 'ntypes'):
            with hg.local_scope():
                hg.ndata['h'] = h_dict
                g = dgl.to_homogeneous(hg, ndata='h')
                h = g.ndata['h']
                for l in range(self.num_layers):
                    h = self.hgn_layers[l](g, h, g.ndata['_TYPE'], g.edata['_TYPE'], True)
                    h = h.flatten(1)
            h_dict = to_hetero_feat(h, g.ndata['_TYPE'], hg.ntypes)
        else:
            h = h_dict
            for layer, block in zip(self.hgn_layers, hg):
                h = layer(block, h, block.ndata['_TYPE']['_N'], block.edata['_TYPE'], presorted=False)
            h_dict = to_hetero_feat(h, block.ndata['_TYPE']['_N'][:block.num_dst_nodes()], self.ntypes)
        return h_dict

    @property
    def to_homo_flag(self):
        return True


class SkipGram(BaseModel):

    @classmethod
    def build_model_from_args(cls, args, hg):
        return cls(hg.num_nodes(), args.dim)

    def __init__(self, num_nodes, dim):
        super(SkipGram, self).__init__()
        self.embedding_dim = dim
        self.u_embeddings = nn.Embedding(num_nodes, self.embedding_dim, sparse=True)
        self.v_embeddings = nn.Embedding(num_nodes, self.embedding_dim, sparse=True)
        initrange = 1.0 / self.embedding_dim
        nn.init.uniform_(self.u_embeddings.weight.data, -initrange, initrange)
        nn.init.constant_(self.v_embeddings.weight.data, 0)

    def forward(self, pos_u, pos_v, neg_v):
        emb_u = self.u_embeddings(pos_u)
        emb_v = self.v_embeddings(pos_v)
        emb_neg_v = self.v_embeddings(neg_v)
        score = torch.sum(torch.mul(emb_u, emb_v), dim=1)
        score = torch.clamp(score, max=10, min=-10)
        score = -F.logsigmoid(score)
        neg_score = torch.bmm(emb_neg_v, emb_u.unsqueeze(2)).squeeze()
        neg_score = torch.clamp(neg_score, max=10, min=-10)
        neg_score = -torch.sum(F.logsigmoid(-neg_score), dim=1)
        return torch.mean(score + neg_score)

    def save_embedding(self, file_name):
        numpy.save(file_name, self.u_embeddings.weight.cpu().data.numpy())


class TransD(BaseModel):

    @classmethod
    def build_model_from_args(cls, args, hg):
        return cls(args, hg)

    def __init__(self, args, hg):
        super(TransD, self).__init__()
        self.device = args.device
        self.ent_num = hg.num_nodes()
        self.rel_num = len(hg.canonical_etypes)
        self.ent_dim = args.ent_dim
        self.rel_dim = args.rel_dim
        self.margin = args.margin
        self.dis_norm = args.dis_norm
        self.n_emb = nn.Embedding(self.ent_num, self.ent_dim)
        self.r_emb = nn.Embedding(self.rel_num, self.rel_dim)
        self.n_emb_p = nn.Embedding(self.ent_num, self.ent_dim)
        self.r_emb_p = nn.Embedding(self.rel_num, self.rel_dim)
        nn.init.xavier_uniform_(self.n_emb.weight.data)
        nn.init.xavier_uniform_(self.r_emb.weight.data)
        nn.init.xavier_uniform_(self.n_emb_p.weight.data)
        nn.init.xavier_uniform_(self.r_emb_p.weight.data)

    def _resize(self, tensor, axis, size):
        shape = tensor.size()
        axis = len(shape) + axis
        osize = shape[axis]
        if osize == size:
            return tensor
        elif osize > size:
            return th.narrow(tensor, axis, 0, size)
        else:
            for i in range(len(shape)):
                if i == axis:
                    paddings = [0, size - osize] + paddings
                else:
                    paddings = [0, 0] + paddings
            return F.pad(tensor, pad=paddings, mode='constant', value=0)

    def _transfer(self, n, n_emb_p, r_emb_p):
        if n.shape[0] != r_emb_p.shape[0]:
            n = n.view(-1, r_emb_p.shape[0], n.shape[-1])
            n_emb_p = n_emb_p.view(-1, r_emb_p.shape[0], n_emb_p.shape[-1])
            r_emb_p = r_emb_p.view(-1, r_emb_p.shape[0], r_emb_p.shape[-1])
            n = F.normalize(self._resize(n, -1, r_emb_p.size()[-1]) + th.sum(n * n_emb_p, -1, True) * r_emb_p, p=2, dim=-1)
            return n.view(-1, n.shape[-1])
        else:
            return F.normalize(self._resize(n, -1, r_emb_p.shape[-1]) + th.sum(n * n_emb_p, -1, True) * r_emb_p, p=2, dim=-1)

    def forward(self, h, r, t):
        if self.training:
            self.n_emb.weight.data = F.normalize(self.n_emb.weight.data, p=2.0, dim=-1)
            self.r_emb.weight.data = F.normalize(self.r_emb.weight.data, p=2.0, dim=-1)
            self.n_emb_p.weight.data = F.normalize(self.n_emb_p.weight.data, p=2.0, dim=-1)
            self.r_emb_p.weight.data = F.normalize(self.r_emb_p.weight.data, p=2.0, dim=-1)
        if h.shape == th.Size([]):
            h = h.view(1)
        if r.shape == th.Size([]):
            r = r.view(1)
        if t.shape == th.Size([]):
            t = t.view(1)
        h = h
        r = r
        t = t
        h_emb = self.n_emb(h)
        r_emb = self.r_emb(r)
        t_emb = self.n_emb(t)
        h_emb_p = self.n_emb_p(h)
        r_emb_p = self.r_emb_p(r)
        t_emb_p = self.n_emb_p(t)
        h_emb = self._transfer(h_emb, h_emb_p, r_emb_p)
        t_emb = self._transfer(t_emb, t_emb_p, r_emb_p)
        h_emb = F.normalize(h_emb, 2.0, -1)
        r_emb = F.normalize(r_emb, 2.0, -1)
        t_emb = F.normalize(t_emb, 2.0, -1)
        score = th.norm(h_emb + r_emb - t_emb, self.dis_norm, dim=-1)
        return score


class TransH(BaseModel):

    @classmethod
    def build_model_from_args(cls, args, hg):
        return cls(args, hg)

    def __init__(self, args, hg):
        super(TransH, self).__init__()
        self.device = args.device
        self.ent_num = hg.num_nodes()
        self.rel_num = len(hg.canonical_etypes)
        self.dim = args.hidden_dim
        self.margin = args.margin
        self.dis_norm = args.dis_norm
        self.n_emb = nn.Embedding(self.ent_num, self.dim)
        self.r_emb = nn.Embedding(self.rel_num, self.dim)
        self.norm_vector = nn.Embedding(self.rel_num, self.dim)
        nn.init.xavier_uniform_(self.n_emb.weight.data)
        nn.init.xavier_uniform_(self.r_emb.weight.data)
        nn.init.xavier_uniform_(self.norm_vector.weight.data)

    def _transfer(self, n_emb, r_norm):
        if n_emb.shape[0] != r_norm.shape[0]:
            n_emb = n_emb.view(-1, r_norm.shape[0], n_emb.shape[-1])
            r_norm = r_norm.view(-1, r_norm.shape[0], r_norm.shape[-1])
            n_emb = n_emb - th.sum(n_emb * r_norm, -1, True) * r_norm
            return n_emb.view(-1, n_emb.shape[-1])
        else:
            return n_emb - th.sum(n_emb * r_norm, -1, True) * r_norm

    def forward(self, h, r, t):
        if self.training:
            self.n_emb.weight.data = F.normalize(self.n_emb.weight.data, p=2.0, dim=-1)
            self.r_emb.weight.data = F.normalize(self.r_emb.weight.data, p=2.0, dim=-1)
            self.norm_vector.weight.data = F.normalize(self.norm_vector.weight.data, p=2.0, dim=-1)
        if h.shape == th.Size([]):
            h = h.view(1)
        if r.shape == th.Size([]):
            r = r.view(1)
        if t.shape == th.Size([]):
            t = t.view(1)
        h_emb = self.n_emb(h)
        r_emb = self.r_emb(r)
        t_emb = self.n_emb(t)
        r_norm = self.norm_vector(r)
        h_emb = self._transfer(h_emb, r_norm)
        t_emb = self._transfer(t_emb, r_norm)
        h_emb = F.normalize(h_emb, 2.0, -1)
        r_emb = F.normalize(r_emb, 2.0, -1)
        t_emb = F.normalize(t_emb, 2.0, -1)
        score = th.norm(h_emb + r_emb - t_emb, self.dis_norm, dim=-1)
        return score


class TransR(BaseModel):

    @classmethod
    def build_model_from_args(cls, args, hg):
        return cls(args, hg)

    def __init__(self, args, hg):
        super(TransR, self).__init__()
        self.device = args.device
        self.ent_num = hg.num_nodes()
        self.rel_num = len(hg.canonical_etypes)
        self.ent_dim = args.ent_dim
        self.rel_dim = args.rel_dim
        self.margin = args.margin
        self.dis_norm = args.dis_norm
        self.n_emb = nn.Embedding(self.ent_num, self.ent_dim)
        self.r_emb = nn.Embedding(self.rel_num, self.rel_dim)
        self.r_emb_p = nn.Embedding(self.rel_num, self.ent_dim * self.rel_dim)
        nn.init.xavier_uniform_(self.n_emb.weight.data)
        nn.init.xavier_uniform_(self.r_emb.weight.data)
        nn.init.xavier_uniform_(self.r_emb_p.weight.data)

    def _transfer(self, n, r_emb_p):
        r_emb_p = r_emb_p.view(-1, self.ent_dim, self.rel_dim)
        if n.shape[0] != r_emb_p.shape[0]:
            n = n.view(-1, r_emb_p.shape[0], self.ent_dim).permute(1, 0, 2)
            n = th.matmul(n, r_emb_p).permute(1, 0, 2)
        else:
            n = n.view(-1, 1, self.ent_dim)
            n = th.matmul(n, r_emb_p)
        return n.view(-1, self.rel_dim)

    def forward(self, h, r, t):
        if self.training:
            self.n_emb.weight.data = F.normalize(self.n_emb.weight.data, p=2.0, dim=-1)
            self.r_emb.weight.data = F.normalize(self.r_emb.weight.data, p=2.0, dim=-1)
            self.r_emb_p.weight.data = F.normalize(self.r_emb_p.weight.data, p=2.0, dim=-1)
        if h.shape == th.Size([]):
            h = h.view(1)
        if r.shape == th.Size([]):
            r = r.view(1)
        if t.shape == th.Size([]):
            t = t.view(1)
        r = r
        h_emb = self.n_emb(h)
        r_emb = self.r_emb(r)
        t_emb = self.n_emb(t)
        r_emb_p = self.r_emb_p(r)
        h_emb = self._transfer(h_emb, r_emb_p)
        t_emb = self._transfer(t_emb, r_emb_p)
        h_emb = F.normalize(h_emb, 2.0, -1)
        r_emb = F.normalize(r_emb, 2.0, -1)
        t_emb = F.normalize(t_emb, 2.0, -1)
        score = th.norm(h_emb + r_emb - t_emb, self.dis_norm, dim=-1)
        return score


def adam(grad, state_sum, nodes, lr, device, only_gpu):
    """ calculate gradients according to adam """
    grad_sum = (grad * grad).mean(1)
    if not only_gpu:
        grad_sum = grad_sum.cpu()
    state_sum.index_add_(0, nodes, grad_sum)
    std = state_sum[nodes]
    std_values = std.sqrt_().add_(1e-10).unsqueeze(1)
    grad = lr * grad / std_values
    return grad


def async_update(num_threads, model, queue):
    """ asynchronous embedding update """
    torch.set_num_threads(num_threads)
    while True:
        grad_u, grad_v, grad_v_neg, nodes, neg_nodes = queue.get()
        if grad_u is None:
            return
        with torch.no_grad():
            model.u_embeddings.weight.data.index_add_(0, nodes.view(-1), grad_u)
            model.v_embeddings.weight.data.index_add_(0, nodes.view(-1), grad_v)
            if neg_nodes is not None:
                model.v_embeddings.weight.data.index_add_(0, neg_nodes.view(-1), grad_v_neg)


def init_emb2neg_index(walk_length, window_size, negative, batch_size):
    """select embedding of negative nodes from a batch of node embeddings
    for fast negative sampling

    Return
    ------
    index_emb_negu torch.LongTensor : the indices of u_embeddings
    index_emb_negv torch.LongTensor : the indices of v_embeddings
    Usage
    -----
    # emb_u.shape: [batch_size * walk_length, dim]
    batch_emb2negu = torch.index_select(emb_u, 0, index_emb_negu)
    """
    idx_list_u = []
    for b in range(batch_size):
        for i in range(walk_length):
            for j in range(i - window_size, i):
                if j >= 0:
                    idx_list_u += [i + b * walk_length] * negative
            for j in range(i + 1, i + 1 + window_size):
                if j < walk_length:
                    idx_list_u += [i + b * walk_length] * negative
    idx_list_v = list(range(batch_size * walk_length)) * negative * window_size * 2
    random.shuffle(idx_list_v)
    idx_list_v = idx_list_v[:len(idx_list_u)]
    index_emb_negu = torch.LongTensor(idx_list_u)
    index_emb_negv = torch.LongTensor(idx_list_v)
    return index_emb_negu, index_emb_negv


def init_emb2pos_index(walk_length, window_size, batch_size):
    """ select embedding of positive nodes from a batch of node embeddings

    Return
    ------
    index_emb_posu torch.LongTensor : the indices of u_embeddings
    index_emb_posv torch.LongTensor : the indices of v_embeddings
    Usage
    -----
    # emb_u.shape: [batch_size * walk_length, dim]
    batch_emb2posu = torch.index_select(emb_u, 0, index_emb_posu)
    """
    idx_list_u = []
    idx_list_v = []
    for b in range(batch_size):
        for i in range(walk_length):
            for j in range(i - window_size, i):
                if j >= 0:
                    idx_list_u.append(j + b * walk_length)
                    idx_list_v.append(i + b * walk_length)
            for j in range(i + 1, i + 1 + window_size):
                if j < walk_length:
                    idx_list_u.append(j + b * walk_length)
                    idx_list_v.append(i + b * walk_length)
    index_emb_posu = torch.LongTensor(idx_list_u)
    index_emb_posv = torch.LongTensor(idx_list_v)
    return index_emb_posu, index_emb_posv


def init_empty_grad(emb_dimension, walk_length, batch_size):
    """ initialize gradient matrix """
    grad_u = torch.zeros((batch_size * walk_length, emb_dimension))
    grad_v = torch.zeros((batch_size * walk_length, emb_dimension))
    return grad_u, grad_v


def init_weight(walk_length, window_size, batch_size):
    """ init context weight """
    weight = []
    for b in range(batch_size):
        for i in range(walk_length):
            for j in range(i - window_size, i):
                if j >= 0:
                    weight.append(1.0 - float(i - j - 1) / float(window_size))
            for j in range(i + 1, i + 1 + window_size):
                if j < walk_length:
                    weight.append(1.0 - float(j - i - 1) / float(window_size))
    return torch.Tensor(weight).unsqueeze(1)


class SkipGramModel(nn.Module):
    """ Negative sampling based skip-gram """

    def __init__(self, emb_size, emb_dimension, walk_length, window_size, batch_size, only_cpu, only_gpu, mix, neg_weight, negative, lr, lap_norm, fast_neg, record_loss, norm, use_context_weight, async_update, num_threads):
        """ initialize embedding on CPU
        Paremeters
        ----------
        emb_size int : number of nodes
        emb_dimension int : embedding dimension
        walk_length int : number of nodes in a sequence
        window_size int : context window size
        batch_size int : number of node sequences in each batch
        only_cpu bool : training with CPU
        only_gpu bool : training with GPU
        mix bool : mixed training with CPU and GPU
        negative int : negative samples for each positve node pair
        neg_weight float : negative weight
        lr float : initial learning rate
        lap_norm float : weight of laplacian normalization
        fast_neg bool : do negative sampling inside a batch
        record_loss bool : print the loss during training
        norm bool : do normalizatin on the embedding after training
        use_context_weight : give different weights to the nodes in a context window
        async_update : asynchronous training
        """
        super(SkipGramModel, self).__init__()
        self.emb_size = emb_size
        self.emb_dimension = emb_dimension
        self.walk_length = walk_length
        self.window_size = window_size
        self.batch_size = batch_size
        self.only_cpu = only_cpu
        self.only_gpu = only_gpu
        self.mixed_train = mix
        self.neg_weight = neg_weight
        self.negative = negative
        self.lr = lr
        self.lap_norm = lap_norm
        self.fast_neg = fast_neg
        self.record_loss = record_loss
        self.norm = norm
        self.use_context_weight = use_context_weight
        self.async_update = async_update
        self.num_threads = num_threads
        self.device = torch.device('cpu')
        self.u_embeddings = nn.Embedding(self.emb_size, self.emb_dimension, sparse=True)
        self.v_embeddings = nn.Embedding(self.emb_size, self.emb_dimension, sparse=True)
        initrange = 1.0 / self.emb_dimension
        init.uniform_(self.u_embeddings.weight.data, -initrange, initrange)
        init.constant_(self.v_embeddings.weight.data, 0)
        self.lookup_table = torch.sigmoid(torch.arange(-6.01, 6.01, 0.01))
        self.lookup_table[0] = 0.0
        self.lookup_table[-1] = 1.0
        if self.record_loss:
            self.logsigmoid_table = torch.log(torch.sigmoid(torch.arange(-6.01, 6.01, 0.01)))
            self.loss = []
        self.index_emb_posu, self.index_emb_posv = init_emb2pos_index(self.walk_length, self.window_size, self.batch_size)
        self.index_emb_negu, self.index_emb_negv = init_emb2neg_index(self.walk_length, self.window_size, self.negative, self.batch_size)
        if self.use_context_weight:
            self.context_weight = init_weight(self.walk_length, self.window_size, self.batch_size)
        self.state_sum_u = torch.zeros(self.emb_size)
        self.state_sum_v = torch.zeros(self.emb_size)
        self.grad_u, self.grad_v = init_empty_grad(self.emb_dimension, self.walk_length, self.batch_size)

    def create_async_update(self):
        """ Set up the async update subprocess.
        """
        self.async_q = Queue(1)
        self.async_p = mp.Process(target=async_update, args=(self.num_threads, self, self.async_q))
        self.async_p.start()

    def finish_async_update(self):
        """ Notify the async update subprocess to quit.
        """
        self.async_q.put((None, None, None, None, None))
        self.async_p.join()

    def share_memory(self):
        """ share the parameters across subprocesses """
        self.u_embeddings.weight.share_memory_()
        self.v_embeddings.weight.share_memory_()
        self.state_sum_u.share_memory_()
        self.state_sum_v.share_memory_()

    def set_device(self, gpu_id):
        """ set gpu device """
        self.device = torch.device('cuda:%d' % gpu_id)
        None
        self.lookup_table = self.lookup_table
        if self.record_loss:
            self.logsigmoid_table = self.logsigmoid_table
        self.index_emb_posu = self.index_emb_posu
        self.index_emb_posv = self.index_emb_posv
        self.index_emb_negu = self.index_emb_negu
        self.index_emb_negv = self.index_emb_negv
        self.grad_u = self.grad_u
        self.grad_v = self.grad_v
        if self.use_context_weight:
            self.context_weight = self.context_weight

    def all_to_device(self, gpu_id):
        """ move all of the parameters to a single GPU """
        self.device = torch.device('cuda:%d' % gpu_id)
        self.set_device(gpu_id)
        self.u_embeddings = self.u_embeddings
        self.v_embeddings = self.v_embeddings
        self.state_sum_u = self.state_sum_u
        self.state_sum_v = self.state_sum_v

    def fast_sigmoid(self, score):
        """ do fast sigmoid by looking up in a pre-defined table """
        idx = torch.floor((score + 6.01) / 0.01).long()
        return self.lookup_table[idx]

    def fast_logsigmoid(self, score):
        """ do fast logsigmoid by looking up in a pre-defined table """
        idx = torch.floor((score + 6.01) / 0.01).long()
        return self.logsigmoid_table[idx]

    def fast_learn(self, batch_walks, neg_nodes=None):
        """ Learn a batch of random walks in a fast way. It has the following features:
            1. It calculating the gradients directly without the forward operation.
            2. It does sigmoid by a looking up table.
        Specifically, for each positive/negative node pair (i,j), the updating procedure is as following:
            score = self.fast_sigmoid(u_embedding[i].dot(v_embedding[j]))
            # label = 1 for positive samples; label = 0 for negative samples.
            u_embedding[i] += (label - score) * v_embedding[j]
            v_embedding[i] += (label - score) * u_embedding[j]
        Parameters
        ----------
        batch_walks list : a list of node sequnces
        lr float : current learning rate
        neg_nodes torch.LongTensor : a long tensor of sampled true negative nodes. If neg_nodes is None,
            then do negative sampling randomly from the nodes in batch_walks as an alternative.
        Usage example
        -------------
        batch_walks = [torch.LongTensor([1,2,3,4]),
                       torch.LongTensor([2,3,4,2])])
        lr = 0.01
        neg_nodes = None
        """
        lr = self.lr
        if isinstance(batch_walks, list):
            nodes = torch.stack(batch_walks)
        elif isinstance(batch_walks, torch.LongTensor):
            nodes = batch_walks
        if self.only_gpu:
            nodes = nodes
            if neg_nodes is not None:
                neg_nodes = neg_nodes
        emb_u = self.u_embeddings(nodes).view(-1, self.emb_dimension)
        emb_v = self.v_embeddings(nodes).view(-1, self.emb_dimension)
        bs = len(batch_walks)
        if bs < self.batch_size:
            index_emb_posu, index_emb_posv = init_emb2pos_index(self.walk_length, self.window_size, bs)
            index_emb_posu = index_emb_posu
            index_emb_posv = index_emb_posv
        else:
            index_emb_posu = self.index_emb_posu
            index_emb_posv = self.index_emb_posv
        emb_pos_u = torch.index_select(emb_u, 0, index_emb_posu)
        emb_pos_v = torch.index_select(emb_v, 0, index_emb_posv)
        pos_score = torch.sum(torch.mul(emb_pos_u, emb_pos_v), dim=1)
        pos_score = torch.clamp(pos_score, max=6, min=-6)
        score = (1 - self.fast_sigmoid(pos_score)).unsqueeze(1)
        if self.record_loss:
            self.loss.append(torch.mean(self.fast_logsigmoid(pos_score)).item())
        if self.lap_norm > 0:
            grad_u_pos = score * emb_pos_v + self.lap_norm * (emb_pos_v - emb_pos_u)
            grad_v_pos = score * emb_pos_u + self.lap_norm * (emb_pos_u - emb_pos_v)
        else:
            grad_u_pos = score * emb_pos_v
            grad_v_pos = score * emb_pos_u
        if self.use_context_weight:
            if bs < self.batch_size:
                context_weight = init_weight(self.walk_length, self.window_size, bs)
            else:
                context_weight = self.context_weight
            grad_u_pos *= context_weight
            grad_v_pos *= context_weight
        if bs < self.batch_size:
            grad_u, grad_v = init_empty_grad(self.emb_dimension, self.walk_length, bs)
            grad_u = grad_u
            grad_v = grad_v
        else:
            self.grad_u = self.grad_u
            self.grad_u.zero_()
            self.grad_v = self.grad_v
            self.grad_v.zero_()
            grad_u = self.grad_u
            grad_v = self.grad_v
        grad_u.index_add_(0, index_emb_posu, grad_u_pos)
        grad_v.index_add_(0, index_emb_posv, grad_v_pos)
        if bs < self.batch_size:
            index_emb_negu, index_emb_negv = init_emb2neg_index(self.walk_length, self.window_size, self.negative, bs)
            index_emb_negu = index_emb_negu
            index_emb_negv = index_emb_negv
        else:
            index_emb_negu = self.index_emb_negu
            index_emb_negv = self.index_emb_negv
        emb_neg_u = torch.index_select(emb_u, 0, index_emb_negu)
        if neg_nodes is None:
            emb_neg_v = torch.index_select(emb_v, 0, index_emb_negv)
        else:
            emb_neg_v = self.v_embeddings.weight[neg_nodes]
        neg_score = torch.sum(torch.mul(emb_neg_u, emb_neg_v), dim=1)
        neg_score = torch.clamp(neg_score, max=6, min=-6)
        score = -self.fast_sigmoid(neg_score).unsqueeze(1)
        if self.record_loss:
            self.loss.append(self.negative * self.neg_weight * torch.mean(self.fast_logsigmoid(-neg_score)).item())
        grad_u_neg = self.neg_weight * score * emb_neg_v
        grad_v_neg = self.neg_weight * score * emb_neg_u
        grad_u.index_add_(0, index_emb_negu, grad_u_neg)
        if neg_nodes is None:
            grad_v.index_add_(0, index_emb_negv, grad_v_neg)
        nodes = nodes.view(-1)
        grad_u = adam(grad_u, self.state_sum_u, nodes, lr, self.device, self.only_gpu)
        grad_v = adam(grad_v, self.state_sum_v, nodes, lr, self.device, self.only_gpu)
        if neg_nodes is not None:
            grad_v_neg = adam(grad_v_neg, self.state_sum_v, neg_nodes, lr, self.device, self.only_gpu)
        if self.mixed_train:
            grad_u = grad_u.cpu()
            grad_v = grad_v.cpu()
            if neg_nodes is not None:
                grad_v_neg = grad_v_neg.cpu()
            else:
                grad_v_neg = None
            if self.async_update:
                grad_u.share_memory_()
                grad_v.share_memory_()
                nodes.share_memory_()
                if neg_nodes is not None:
                    neg_nodes.share_memory_()
                    grad_v_neg.share_memory_()
                self.async_q.put((grad_u, grad_v, grad_v_neg, nodes, neg_nodes))
        if not self.async_update:
            self.u_embeddings.weight.data.index_add_(0, nodes.view(-1), grad_u)
            self.v_embeddings.weight.data.index_add_(0, nodes.view(-1), grad_v)
            if neg_nodes is not None:
                self.v_embeddings.weight.data.index_add_(0, neg_nodes.view(-1), grad_v_neg)
        return

    def forward(self, pos_u, pos_v, neg_v):
        """ Do forward and backward. It is designed for future use. """
        emb_u = self.u_embeddings(pos_u)
        emb_v = self.v_embeddings(pos_v)
        emb_neg_v = self.v_embeddings(neg_v)
        score = torch.sum(torch.mul(emb_u, emb_v), dim=1)
        score = torch.clamp(score, max=6, min=-6)
        score = -F.logsigmoid(score)
        neg_score = torch.bmm(emb_neg_v, emb_u.unsqueeze(2)).squeeze()
        neg_score = torch.clamp(neg_score, max=6, min=-6)
        neg_score = -torch.sum(F.logsigmoid(-neg_score), dim=1)
        return torch.sum(score), torch.sum(neg_score)

    def save_embedding(self, dataset, file_name):
        """ Write embedding to local file. Only used when node ids are numbers.
        Parameter
        ---------
        dataset DeepwalkDataset : the dataset
        file_name str : the file name
        """
        embedding = self.u_embeddings.weight.cpu().data.numpy()
        if self.norm:
            embedding /= np.sqrt(np.sum(embedding * embedding, 1)).reshape(-1, 1)
        np.save(file_name, embedding)

    def save_embedding_pt(self, dataset, file_name):
        """ For ogb leaderboard.
        """
        try:
            max_node_id = max(dataset.node2id.keys())
            if max_node_id + 1 != self.emb_size:
                None
            embedding = torch.zeros(max_node_id + 1, self.emb_dimension)
            index = torch.LongTensor(list(map(lambda id: dataset.id2node[id], list(range(self.emb_size)))))
            embedding.index_add_(0, index, self.u_embeddings.weight.cpu().data)
            if self.norm:
                embedding /= torch.sqrt(torch.sum(embedding.mul(embedding), 1) + 1e-06).unsqueeze(1)
            torch.save(embedding, file_name)
        except:
            self.save_embedding_pt_dgl_graph(dataset, file_name)

    def save_embedding_pt_dgl_graph(self, dataset, file_name):
        """ For ogb leaderboard """
        embedding = torch.zeros_like(self.u_embeddings.weight.cpu().data)
        valid_seeds = torch.LongTensor(dataset.valid_seeds)
        valid_embedding = self.u_embeddings.weight.cpu().data.index_select(0, valid_seeds)
        embedding.index_add_(0, valid_seeds, valid_embedding)
        if self.norm:
            embedding /= torch.sqrt(torch.sum(embedding.mul(embedding), 1) + 1e-06).unsqueeze(1)
        torch.save(embedding, file_name)

    def save_embedding_txt(self, dataset, file_name):
        """ Write embedding to local file. For future use.
        Parameter
        ---------
        dataset DeepwalkDataset : the dataset
        file_name str : the file name
        """
        embedding = self.u_embeddings.weight.cpu().data.numpy()
        if self.norm:
            embedding /= np.sqrt(np.sum(embedding * embedding, 1)).reshape(-1, 1)
        with open(file_name, 'w') as f:
            f.write('%d %d\n' % (self.emb_size, self.emb_dimension))
            for wid in range(self.emb_size):
                e = ' '.join(map(lambda x: str(x), embedding[wid]))
                f.write('%s %s\n' % (str(dataset.id2node[wid]), e))


class fastGTN(BaseModel):
    """
        fastGTN from paper `Graph Transformer Networks: Learning Meta-path Graphs to Improve GNNs
        <https://arxiv.org/abs/2106.06218>`__.
        It is the extension paper  of GTN.
        `Code from author <https://github.com/seongjunyun/Graph_Transformer_Networks>`__.

        Given a heterogeneous graph :math:`G` and its edge relation type set :math:`\\mathcal{R}`.Then we extract
        the single relation adjacency matrix list. In that, we can generate combination adjacency matrix by conv
        the single relation adjacency matrix list. We can generate :math:'l-length' meta-path adjacency matrix
        by multiplying combination adjacency matrix. Then we can generate node representation using a GCN layer.

        Parameters
        ----------
        num_edge_type : int
            Number of relations.
        num_channels : int
            Number of conv channels.
        in_dim : int
            The dimension of input feature.
        hidden_dim : int
            The dimension of hidden layer.
        num_class : int
            Number of classification type.
        num_layers : int
            Length of hybrid metapath.
        category : string
            Type of predicted nodes.
        norm : bool
            If True, the adjacency matrix will be normalized.
        identity : bool
            If True, the identity matrix will be added to relation matrix set.

    """

    @classmethod
    def build_model_from_args(cls, args, hg):
        if args.identity:
            num_edge_type = len(hg.canonical_etypes) + 1
        else:
            num_edge_type = len(hg.canonical_etypes)
        return cls(num_edge_type=num_edge_type, num_channels=args.num_channels, in_dim=args.hidden_dim, hidden_dim=args.hidden_dim, num_class=args.out_dim, num_layers=args.num_layers, category=args.category, norm=args.norm_emd_flag, identity=args.identity)

    def __init__(self, num_edge_type, num_channels, in_dim, hidden_dim, num_class, num_layers, category, norm, identity):
        super(fastGTN, self).__init__()
        self.num_edge_type = num_edge_type
        self.num_channels = num_channels
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_class = num_class
        self.num_layers = num_layers
        self.is_norm = norm
        self.category = category
        self.identity = identity
        layers = []
        for i in range(num_layers):
            layers.append(GTConv(num_edge_type, num_channels))
        self.params = nn.ParameterList()
        for i in range(num_channels):
            self.params.append(nn.Parameter(th.Tensor(in_dim, hidden_dim)))
        self.layers = nn.ModuleList(layers)
        self.gcn = GCNConv()
        self.norm = EdgeWeightNorm(norm='right')
        self.linear1 = nn.Linear(self.hidden_dim * self.num_channels, self.hidden_dim)
        self.linear2 = nn.Linear(self.hidden_dim, self.num_class)
        self.category_idx = None
        self.A = None
        self.h = None
        self.reset_parameters()

    def reset_parameters(self):
        if self.params is not None:
            for para in self.params:
                nn.init.xavier_uniform_(para)

    def normalization(self, H):
        norm_H = []
        for i in range(self.num_channels):
            g = H[i]
            g = dgl.remove_self_loop(g)
            g.edata['w_sum'] = self.norm(g, g.edata['w_sum'])
            norm_H.append(g)
        return norm_H

    def forward(self, hg, h):
        with hg.local_scope():
            hg.ndata['h'] = h
            if self.category_idx is None:
                self.A, h, self.category_idx = transform_relation_graph_list(hg, category=self.category, identity=self.identity)
            else:
                g = dgl.to_homogeneous(hg, ndata='h')
                h = g.ndata['h']
            A = self.A
            H = []
            for n_c in range(self.num_channels):
                H.append(th.matmul(h, self.params[n_c]))
            for i in range(self.num_layers):
                hat_A = self.layers[i](A)
                for n_c in range(self.num_channels):
                    edge_weight = self.norm(hat_A[n_c], hat_A[n_c].edata['w_sum'])
                    H[n_c] = self.gcn(hat_A[n_c], H[n_c], edge_weight=edge_weight)
            X_ = self.linear1(th.cat(H, dim=1))
            X_ = F.relu(X_)
            y = self.linear2(X_)
            return {self.category: y[self.category_idx]}


def sce_loss(x, y, gamma=3):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)
    loss = (1 - (x * y).sum(dim=-1)).pow_(gamma)
    loss = loss.mean()
    return loss


class HGMAE(BaseModel):
    """
    **Title:** Heterogeneous Graph Masked Autoencoders

    **Authors:** Yijun Tian, Kaiwen Dong, Chunhui Zhang, Chuxu Zhang, Nitesh V. Chawla

    HGMAE was introduced in `[paper] <https://arxiv.org/abs/2208.09957>`_
    and parameters are defined as follows:

    Parameter
    ----------
    metapaths_dict: dict[str, list[etype]]
        Dict from meta path name to meta path.
    category : string
        The category of the nodes to be classificated.
    in_dim : int
        Dim of input feats
    hidden_dim : int
        Dim of encoded embedding.
    num_layers : int
        Number of layers of HAN encoder and decoder.
    num_heads : int
        Number of attention heads of hidden layers in HAN encoder and decoder.
    num_out_heads : int
        Number of attention heads of output projection in HAN encoder and decoder.
    feat_drop : float, optional
        Dropout rate on feature. Default: ``0``
    attn_drop : float, optional
        Dropout rate on attention weight. Default: ``0``
    negative_slope : float, optional
        LeakyReLU angle of negative slope. Defaults: ``0.2``.
    residual : bool, optional
        If True, use residual connection. Defaults: ``False``.

    mp_edge_recon_loss_weight : float
        Trade-off weights for balancing mp_edge_recon_loss. Defaults: ``1.0``
    mp_edge_mask_rate : float
        Metapath-based edge masking rate. Defaults: ``0.6``
    mp_edge_gamma : float
        Scaling factor of mp_edge_recon_loss when using ``sce`` as loss function. Defaults: ``3.0``

    node_mask_rate : str
        Linearly increasing attribute mask rate to sample a subset of nodes, in the format of 'min,delta,max'. Defaults: ``'0.5,0.005,0.8'``
    attr_restore_loss_weight : float
        Trade-off weights for balancing attr_restore_loss. Defaults: ``1.0``
    attr_restore_gamma : float
        Scaling factor of att_restore_loss when using ``sce`` as loss function. Defaults: ``1.0``
    attr_replace_rate : float
        Replacing a percentage of mask tokens by random tokens, with the attr_replace_rate. Defaults: ``0.3``
    attr_unchanged_rate : float
        Leaving a percentage of nodes unchanged by utilizing the origin attribute, with the attr_unchanged_rate. Defaults: ``0.2``
    mp2vec_window_size : int
        In a random walk :attr:`w`, a node :attr:`w[j]` is considered close to a node :attr:`w[i]` if :attr:`i - window_size <= j <= i + window_size`. Defaults: ``3``
    mp2vec_rw_length : int
        The length of each random walk. Defaults: ``10``
    mp2vec_walks_per_node=args.mp2vec_walks_per_node,
        The number of walks to sample for each node. Defaults: ``2``

    mp2vec_negative_size: int
        Number of negative samples to use for each positive sample. Default: ``5``
    mp2vec_batch_size : int
        How many samples per batch to load when training mp2vec_feat. Defaults: ``128``
    mp2vec_train_epoch : int
        The training epochs of MetaPath2Vec model. Default: ``20``
    mp2vec_train_lr : float
        The training learning rate of MetaPath2Vec model. Default: ``0.001``
    mp2vec_feat_dim : int
        The feature dimension of MetaPath2Vec model. Defaults: ``128``
    mp2vec_feat_pred_loss_weight : float
        Trade-off weights for balancing mp2vec_feat_pred_loss. Defaults: ``0.1``
    mp2vec_feat_gamma: flaot
        Scaling factor of mp2vec_feat_pred_loss when using ``sce`` as loss function. Defaults: ``2.0``
    mp2vec_feat_drop: float
        The dropout rate of self.enc_out_to_mp2vec_feat_mapping. Defaults: ``0.2``
    """

    @classmethod
    def build_model_from_args(cls, args, hg, metapaths_dict: 'dict'):
        return cls(hg=hg, metapaths_dict=metapaths_dict, category=args.category, in_dim=args.in_dim, hidden_dim=args.hidden_dim, num_layers=args.num_layers, num_heads=args.num_heads, num_out_heads=args.num_out_heads, feat_drop=args.feat_drop, attn_drop=args.attn_drop, negative_slope=args.negative_slope, residual=args.residual, mp_edge_recon_loss_weight=args.mp_edge_recon_loss_weight, mp_edge_mask_rate=args.mp_edge_mask_rate, mp_edge_gamma=args.mp_edge_gamma, node_mask_rate=args.node_mask_rate, attr_restore_gamma=args.attr_restore_gamma, attr_restore_loss_weight=args.attr_restore_loss_weight, attr_replace_rate=args.attr_replace_rate, attr_unchanged_rate=args.attr_unchanged_rate, mp2vec_negative_size=args.mp2vec_negative_size, mp2vec_window_size=args.mp2vec_window_size, mp2vec_rw_length=args.mp2vec_rw_length, mp2vec_walks_per_node=args.mp2vec_walks_per_node, mp2vec_batch_size=args.mp2vec_batch_size, mp2vec_train_epoch=args.mp2vec_train_epoch, mp2vec_trian_lr=args.mp2vec_train_lr, mp2vec_feat_dim=args.mp2vec_feat_dim, mp2vec_feat_pred_loss_weight=args.mp2vec_feat_pred_loss_weight, mp2vec_feat_gamma=args.mp2vec_feat_gamma, mp2vec_feat_drop=args.mp2vec_feat_drop)

    def __init__(self, hg, metapaths_dict, category, in_dim, hidden_dim, num_layers, num_heads, num_out_heads, feat_drop=0, attn_drop=0, negative_slope=0.2, residual=False, mp_edge_recon_loss_weight=1, mp_edge_mask_rate=0.6, mp_edge_gamma=3, attr_restore_loss_weight=1, attr_restore_gamma=1, node_mask_rate='0.5,0.005,0.8', attr_replace_rate=0.3, attr_unchanged_rate=0.2, mp2vec_window_size=3, mp2vec_negative_size=5, mp2vec_rw_length=10, mp2vec_walks_per_node=2, mp2vec_feat_dim=128, mp2vec_feat_drop=0.2, mp2vec_train_epoch=20, mp2vec_batch_size=128, mp2vec_trian_lr=0.001, mp2vec_feat_pred_loss_weight=0.1, mp2vec_feat_gamma=2):
        super(HGMAE, self).__init__()
        self.metapaths_dict = metapaths_dict
        self.num_metapaths = len(metapaths_dict)
        self.category = category
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_out_heads = num_out_heads
        self.feat_drop = feat_drop
        self.attn_drop = attn_drop
        self.negative_slope = negative_slope
        self.residual = residual
        assert self.hidden_dim % self.num_heads == 0
        assert self.hidden_dim % self.num_out_heads == 0
        self.enc_dec_input_dim = self.in_dim
        enc_hidden_dim = self.hidden_dim // self.num_heads
        enc_num_heads = self.num_heads
        dec_hidden_dim = self.hidden_dim // self.num_out_heads
        dec_num_heads = self.num_out_heads
        dec_in_dim = self.hidden_dim
        self.encoder = HAN(num_metapaths=self.num_metapaths, in_dim=self.in_dim, hidden_dim=enc_hidden_dim, out_dim=enc_hidden_dim, num_layers=self.num_layers, num_heads=enc_num_heads, num_out_heads=enc_num_heads, feat_drop=self.feat_drop, attn_drop=self.attn_drop, negative_slope=self.negative_slope, residual=self.residual, norm=nn.BatchNorm1d, activation=nn.PReLU(), encoding=True)
        self.decoder = HAN(num_metapaths=self.num_metapaths, in_dim=dec_in_dim, hidden_dim=dec_hidden_dim, out_dim=self.enc_dec_input_dim, num_layers=1, num_heads=dec_num_heads, num_out_heads=dec_num_heads, feat_drop=self.feat_drop, attn_drop=self.attn_drop, negative_slope=self.negative_slope, residual=self.residual, norm=nn.BatchNorm1d, activation=nn.PReLU(), encoding=False)
        self.__cached_gs = None
        self.__cached_mps = None
        self.mp_edge_recon_loss_weight = mp_edge_recon_loss_weight
        self.mp_edge_mask_rate = mp_edge_mask_rate
        self.mp_edge_gamma = mp_edge_gamma
        self.mp_edge_recon_loss = partial(sce_loss, gamma=mp_edge_gamma)
        self.encoder_to_decoder_edge_recon = nn.Linear(dec_in_dim, dec_in_dim, bias=False)
        self.attr_restore_gamma = attr_restore_gamma
        self.enc_mask_token = nn.Parameter(torch.zeros(1, self.enc_dec_input_dim))
        self.encoder_to_decoder_attr_restore = nn.Linear(dec_in_dim, dec_in_dim, bias=False)
        self.attr_restore_loss = partial(sce_loss, gamma=attr_restore_gamma)
        self.attr_restore_loss_weight = attr_restore_loss_weight
        self.node_mask_rate = node_mask_rate
        assert attr_replace_rate + attr_unchanged_rate < 1, 'attr_replace_rate + attr_unchanged_rate must be smaller than 1 '
        self.attr_unchanged_rate = attr_unchanged_rate
        self.attr_replace_rate = attr_replace_rate
        self.mp2vec_feat_dim = mp2vec_feat_dim
        self.mp2vec_window_size = mp2vec_window_size
        self.mp2vec_negative_size = mp2vec_negative_size
        self.mp2vec_batch_size = mp2vec_batch_size
        self.mp2vec_train_lr = mp2vec_trian_lr
        self.mp2vec_train_epoch = mp2vec_train_epoch
        self.mp2vec_walks_per_node = mp2vec_walks_per_node
        self.mp2vec_rw_length = mp2vec_rw_length
        self.mp2vec_feat = None
        self.mp2vec_feat_pred_loss_weight = mp2vec_feat_pred_loss_weight
        self.mp2vec_feat_drop = mp2vec_feat_drop
        self.mp2vec_feat_gamma = mp2vec_feat_gamma
        self.mp2vec_feat_pred_loss = partial(sce_loss, gamma=self.mp2vec_feat_gamma)
        self.enc_out_to_mp2vec_feat_mapping = nn.Sequential(nn.Linear(dec_in_dim, self.mp2vec_feat_dim), nn.PReLU(), nn.Dropout(self.mp2vec_feat_drop), nn.Linear(self.mp2vec_feat_dim, self.mp2vec_feat_dim), nn.PReLU(), nn.Dropout(self.mp2vec_feat_drop), nn.Linear(self.mp2vec_feat_dim, self.mp2vec_feat_dim))

    def train_mp2vec(self, hg):
        device = hg.device
        num_nodes = hg.num_nodes(self.category)
        Mp4Mp2Vec = []
        mp_nodes_seq = []
        for mp_name, mp in self.metapaths_dict.items():
            Mp4Mp2Vec += mp
            assert mp[0][0] == mp[-1][-1], 'The start node type and the end one in metapath should be the same.'
        x = max(self.mp2vec_rw_length // (len(Mp4Mp2Vec) + 1), 1)
        Mp4Mp2Vec *= x
        for mp in Mp4Mp2Vec:
            mp_nodes_seq.append(mp[0])
        mp_nodes_seq.append(mp[-1])
        assert mp_nodes_seq[0] == mp_nodes_seq[-1], 'The start node type and the end one in metapath should be the same.'
        None
        m2v_model = MetaPath2Vec(hg, Mp4Mp2Vec, self.mp2vec_window_size, self.mp2vec_feat_dim, self.mp2vec_negative_size)
        m2v_model.train()
        dataloader = DataLoader(list(range(num_nodes)) * self.mp2vec_walks_per_node, batch_size=self.mp2vec_batch_size, shuffle=True, collate_fn=m2v_model.sample)
        optimizer = SparseAdam(m2v_model.parameters(), lr=self.mp2vec_train_lr)
        scheduler = CosineAnnealingLR(optimizer, len(dataloader))
        for _ in tqdm(range(self.mp2vec_train_epoch)):
            for pos_u, pos_v, neg_v in dataloader:
                optimizer.zero_grad()
                loss = m2v_model(pos_u, pos_v, neg_v)
                loss.backward()
                optimizer.step()
                scheduler.step()
        m2v_model.eval()
        nids = torch.LongTensor(m2v_model.local_to_global_nid[self.category])
        emb = m2v_model.node_embed(nids)
        del m2v_model, nids, pos_u, pos_v, neg_v
        if device == 'cuda':
            torch.cuda.empty_cache()
        return emb.detach()

    def get_mask_rate(self, input_mask_rate, get_min=False, epoch=None):
        try:
            return float(input_mask_rate)
        except ValueError:
            if ',' in input_mask_rate:
                mask_rate = [float(i) for i in input_mask_rate.split(',')]
                assert len(mask_rate) == 3, "input_mask_rate should be a float number (0-1), or in the format of 'min,delta,max', '0.6,-0.1,0.4', for example "
                start = mask_rate[0]
                step = mask_rate[1]
                end = mask_rate[2]
                if get_min:
                    return min(start, end)
                else:
                    cur_mask_rate = start + epoch * step
                    if cur_mask_rate < min(start, end) or cur_mask_rate > max(start, end):
                        return end
                    return cur_mask_rate
            else:
                raise NotImplementedError("input_mask_rate should be a float number (0-1), or in the format of 'min,delta,max', '0.6,-0.1,0.4', for example")

    def normalize_feat(self, feat):
        rowsum = torch.sum(feat, dim=1).reshape(-1, 1)
        r_inv = torch.pow(rowsum, -1)
        r_inv = torch.where(torch.isinf(r_inv), 0, r_inv)
        return feat * r_inv

    def normalize_adj(self, adj):
        rowsum = torch.sum(adj, dim=1).reshape(-1, 1)
        d_inv_sqrt = torch.pow(rowsum, -0.5)
        d_inv_sqrt = torch.where(torch.isinf(d_inv_sqrt), 0, d_inv_sqrt)
        return d_inv_sqrt * adj * d_inv_sqrt.T

    def get_mps(self, hg: 'dgl.DGLHeteroGraph'):
        if self.__cached_mps is None:
            self.__cached_mps = []
            mps = []
            for mp in self.metapaths_dict.values():
                adj = dgl.metapath_reachable_graph(hg, mp).adjacency_matrix()
                adj = self.normalize_adj(adj.to_dense()).to_sparse()
                mps.append(adj)
            self.__cached_mps = mps
        return self.__cached_mps.copy()

    def mps_to_gs(self, mps: 'list'):
        if self.__cached_gs is None:
            gs = []
            for mp in mps:
                indices = mp.indices()
                cur_graph = dgl.graph((indices[0], indices[1]))
                cur_graph = dgl.add_self_loop(cur_graph)
                gs.append(cur_graph)
            self.__cached_gs = gs
        return self.__cached_gs.copy()

    def mask_mp_edge_reconstruction(self, mps, feat, epoch):
        masked_gs = self.mps_to_gs(mps)
        cur_mp_edge_mask_rate = self.get_mask_rate(self.mp_edge_mask_rate, epoch=epoch)
        drop_edge = DropEdge(p=cur_mp_edge_mask_rate)
        for i in range(len(masked_gs)):
            masked_gs[i] = drop_edge(masked_gs[i])
            masked_gs[i] = dgl.add_self_loop(masked_gs[i])
        enc_emb, _ = self.encoder(masked_gs, feat)
        emb_mapped = self.encoder_to_decoder_edge_recon(enc_emb)
        feat_recon, att_mp = self.decoder(masked_gs, emb_mapped)
        gs_recon = torch.mm(feat_recon, feat_recon.T)
        loss = None
        for i in range(len(mps)):
            if loss is None:
                loss = att_mp[i] * self.mp_edge_recon_loss(gs_recon, mps[i].to_dense())
            else:
                loss += att_mp[i] * self.mp_edge_recon_loss(gs_recon, mps[i].to_dense())
        return loss

    def encoding_mask_noise(self, feat, node_mask_rate=0.3):
        num_nodes = feat.shape[0]
        all_indices = torch.randperm(num_nodes, device=feat.device)
        num_mask_nodes = int(node_mask_rate * num_nodes)
        mask_indices = all_indices[:num_mask_nodes]
        keep_indices = all_indices[num_mask_nodes:]
        num_unchanged_nodes = int(self.attr_unchanged_rate * num_mask_nodes)
        num_noise_nodes = int(self.attr_replace_rate * num_mask_nodes)
        num_real_mask_nodes = num_mask_nodes - num_unchanged_nodes - num_noise_nodes
        perm_mask = torch.randperm(num_mask_nodes, device=feat.device)
        token_nodes = mask_indices[perm_mask[:num_real_mask_nodes]]
        noise_nodes = mask_indices[perm_mask[-num_noise_nodes:]]
        nodes_as_noise = torch.randperm(num_nodes, device=feat.device)[:num_noise_nodes]
        out_feat = feat.clone()
        out_feat[token_nodes] = 0.0
        out_feat[token_nodes] += self.enc_mask_token
        if num_nodes > 0:
            out_feat[noise_nodes] = feat[nodes_as_noise]
        return out_feat, (mask_indices, keep_indices)

    def mask_attr_restoration(self, gs, feat, epoch):
        cur_node_mask_rate = self.get_mask_rate(self.node_mask_rate, epoch=epoch)
        use_feat, (mask_nodes, keep_nodes) = self.encoding_mask_noise(feat, cur_node_mask_rate)
        enc_emb, _ = self.encoder(gs, use_feat)
        emb_mapped = self.encoder_to_decoder_attr_restore(enc_emb)
        emb_mapped[mask_nodes] = 0.0
        feat_recon, att_mp = self.decoder(gs, emb_mapped)
        feat_before_mask = feat[mask_nodes]
        feat_after_mask = feat_recon[mask_nodes]
        loss = self.attr_restore_loss(feat_before_mask, feat_after_mask)
        return loss, enc_emb

    def forward(self, hg: 'dgl.heterograph', h_dict, trained_mp2vec_feat_dict=None, epoch=None):
        assert epoch is not None, 'epoch should be a positive integer'
        if trained_mp2vec_feat_dict is None:
            if self.mp2vec_feat is None:
                None
                self.mp2vec_feat = self.train_mp2vec(hg)
                self.mp2vec_feat = self.normalize_feat(self.mp2vec_feat)
            mp2vec_feat = self.mp2vec_feat
        else:
            mp2vec_feat = trained_mp2vec_feat_dict[self.category]
        mp2vec_feat = mp2vec_feat
        feat = h_dict[self.category]
        feat = self.normalize_feat(feat)
        mps = self.get_mps(hg)
        gs = self.mps_to_gs(mps)
        attr_restore_loss, enc_emb = self.mask_attr_restoration(gs, feat, epoch)
        loss = attr_restore_loss * self.attr_restore_loss_weight
        mp_edge_recon_loss = self.mp_edge_recon_loss_weight * self.mask_mp_edge_reconstruction(mps, feat, epoch)
        loss += mp_edge_recon_loss
        mp2vec_feat_pred = self.enc_out_to_mp2vec_feat_mapping(enc_emb)
        mp2vec_feat_pred_loss = self.mp2vec_feat_pred_loss_weight * self.mp2vec_feat_pred_loss(mp2vec_feat_pred, mp2vec_feat)
        loss += mp2vec_feat_pred_loss
        return loss

    def get_mp2vec_feat(self):
        return self.mp2vec_feat.detach()

    def get_embeds(self, hg, h_dict):
        feat = h_dict[self.category]
        mps = self.get_mps(hg)
        gs = self.mps_to_gs(mps)
        emb, _ = self.encoder(gs, feat)
        return emb.detach()


class ieHGCNConv(nn.Module):
    """
    The ieHGCN convolution layer.

    Parameters
    ----------
    in_size: int
        the input dimension
    out_size: int
        the output dimension
    attn_size: int
        the dimension of attention vector
    ntypes: list
        the node type list of a heterogeneous graph
    etypes: list
        the edge type list of a heterogeneous graph
    activation: str
        the activation function
    bias: boolean
        whether we need bias vector
    batchnorm: boolean
        whether we need batchnorm
    dropout: float
        the drop out rate
    """

    def __init__(self, in_size, out_size, attn_size, ntypes, etypes, activation=F.elu, bias=False, batchnorm=False, dropout=0.0):
        super(ieHGCNConv, self).__init__()
        self.bias = bias
        self.batchnorm = batchnorm
        self.dropout = dropout
        node_size = {}
        for ntype in ntypes:
            node_size[ntype] = in_size
        attn_vector = {}
        for ntype in ntypes:
            attn_vector[ntype] = attn_size
        self.W_self = dglnn.HeteroLinear(node_size, out_size)
        self.W_al = dglnn.HeteroLinear(attn_vector, 1)
        self.W_ar = dglnn.HeteroLinear(attn_vector, 1)
        self.in_size = in_size
        self.out_size = out_size
        self.attn_size = attn_size
        mods = {etype: dglnn.GraphConv(in_size, out_size, norm='right', weight=True, bias=True, allow_zero_in_degree=True) for etype in etypes}
        self.mods = nn.ModuleDict(mods)
        self.linear_q = nn.ModuleDict({ntype: nn.Linear(out_size, attn_size) for ntype in ntypes})
        self.linear_k = nn.ModuleDict({ntype: nn.Linear(out_size, attn_size) for ntype in ntypes})
        self.activation = activation
        if batchnorm:
            self.bn = nn.BatchNorm1d(out_size)
        if bias:
            self.h_bias = nn.Parameter(torch.Tensor(out_size))
            nn.init.zeros_(self.h_bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hg, h_dict):
        """
        The forward part of the ieHGCNConv.
        
        Parameters
        ----------
        hg : object or list[block]
            the dgl heterogeneous graph or the list of blocks
        h_dict: dict
            the feature dict of different node types
            
        Returns
        -------
        dict
            The embeddings after final aggregation.
        """
        outputs = {ntype: [] for ntype in hg.dsttypes}
        if hg.is_block:
            src_inputs = h_dict
            dst_inputs = {k: v[:hg.number_of_dst_nodes(k)] for k, v in h_dict.items()}
        else:
            src_inputs = h_dict
            dst_inputs = h_dict
        with hg.local_scope():
            hg.ndata['h'] = h_dict
            dst_inputs = self.W_self(dst_inputs)
            query = {}
            key = {}
            attn = {}
            attention = {}
            for ntype in hg.dsttypes:
                query[ntype] = self.linear_q[ntype](dst_inputs[ntype])
                key[ntype] = self.linear_k[ntype](dst_inputs[ntype])
            h_l = self.W_al(key)
            h_r = self.W_ar(query)
            for ntype in hg.dsttypes:
                attention[ntype] = F.elu(h_l[ntype] + h_r[ntype])
                attention[ntype] = attention[ntype].unsqueeze(0)
            for srctype, etype, dsttype in hg.canonical_etypes:
                rel_graph = hg[srctype, etype, dsttype]
                if srctype not in h_dict:
                    continue
                dstdata = self.mods[etype](rel_graph, (src_inputs[srctype], dst_inputs[dsttype]))
                outputs[dsttype].append(dstdata)
                attn[dsttype] = self.linear_k[dsttype](dstdata)
                h_attn = self.W_al(attn)
                attn.clear()
                edge_attention = F.elu(h_attn[dsttype] + h_r[dsttype])
                attention[dsttype] = torch.cat((attention[dsttype], edge_attention.unsqueeze(0)))
            for ntype in hg.dsttypes:
                attention[ntype] = F.softmax(attention[ntype], dim=0)
            rst = {ntype: (0) for ntype in hg.dsttypes}
            for ntype, data in outputs.items():
                data = [dst_inputs[ntype]] + data
                if len(data) != 0:
                    for i in range(len(data)):
                        aggregation = torch.mul(data[i], attention[ntype][i])
                        rst[ntype] = aggregation + rst[ntype]

        def _apply(ntype, h):
            if self.bias:
                h = h + self.h_bias
            if self.activation:
                h = self.activation(h)
            if self.batchnorm:
                h = self.bn(h)
            return self.dropout(h)
        return {ntype: _apply(ntype, feat) for ntype, feat in rst.items()}


class ieHGCN(BaseModel):
    """
    ie-HGCN from paper `Interpretable and Efficient Heterogeneous Graph Convolutional Network
    <https://arxiv.org/pdf/2005.13183.pdf>`__.

    `Source Code Link <https://github.com/kepsail/ie-HGCN>`_
    
    The core part of ie-HGCN, the calculating flow of projection, object-level aggregation and type-level aggregation in
    a specific type block.

    Projection
    
    .. math::
        Y^{Self-\\Omega }=H^{\\Omega} \\cdot W^{Self-\\Omega} \\quad (1)-1

        Y^{\\Gamma - \\Omega}=H^{\\Gamma} \\cdot W^{\\Gamma - \\Omega} , \\Gamma \\in N_{\\Omega} \\quad (1)-2

    Object-level Aggregation
    
    .. math::
        Z^{ Self - \\Omega } = Y^{ Self - \\Omega}=H^{\\Omega} \\cdot W^{Self - \\Omega} \\quad (2)-1

        Z^{\\Gamma - \\Omega}=\\hat{A}^{\\Omega-\\Gamma} \\cdot Y^{\\Gamma - \\Omega} = \\hat{A}^{\\Omega-\\Gamma} \\cdot H^{\\Gamma} \\cdot W^{\\Gamma - \\Omega} \\quad (2)-2

    Type-level Aggregation
    
    .. math::
        Q^{\\Omega}=Z^{Self-\\Omega} \\cdot W_q^{\\Omega} \\quad (3)-1

        K^{Self-\\Omega}=Z^{Self -\\Omega} \\cdot W_{k}^{\\Omega} \\quad (3)-2

        K^{\\Gamma - \\Omega}=Z^{\\Gamma - \\Omega} \\cdot W_{k}^{\\Omega}, \\quad \\Gamma \\in N_{\\Omega} \\quad (3)-3

    .. math::
        e^{Self-\\Omega}={ELU} ([K^{ Self-\\Omega} \\| Q^{\\Omega}] \\cdot w_{a}^{\\Omega}) \\quad (4)-1

        e^{\\Gamma - \\Omega}={ELU} ([K^{\\Gamma - \\Omega} \\| Q^{\\Omega}] \\cdot w_{a}^{\\Omega}), \\Gamma \\in N_{\\Omega} \\quad (4)-2

    .. math::
        [a^{Self-\\Omega}\\|a^{1 - \\Omega}\\| \\ldots . a^{\\Gamma - \\Omega}\\|\\ldots\\| a^{|N_{\\Omega}| - \\Omega}] \\\\
        = {softmax}([e^{Self - \\Omega}\\|e^{1 - \\Omega}\\| \\ldots\\|e^{\\Gamma - \\Omega}\\| \\ldots \\| e^{|N_{\\Omega}| - \\Omega}]) \\quad (5)

    .. math::
        H_{i,:}^{\\Omega \\prime}=\\sigma(a_{i}^{Self-\\Omega} \\cdot Z_{i,:}^{Self-\\Omega}+\\sum_{\\Gamma \\in N_{\\Omega}} a_{i}^{\\Gamma - \\Omega} \\cdot Z_{i,:}^{\\Gamma - \\Omega}) \\quad (6)
    
    Parameters
    ----------
    num_layers: int
        the number of layers
    in_dim: int
        the input dimension
    hidden_dim: int
        the hidden dimension
    out_dim: int
        the output dimension
    attn_dim: int
        the dimension of attention vector
    ntypes: list
        the node type of a heterogeneous graph
    etypes: list
        the edge type of a heterogeneous graph
    bias: boolean
        whether we need bias vector
    batchnorm: boolean
        whether we need batchnorm
    dropout: float
        the drop out rate
    """

    @classmethod
    def build_model_from_args(cls, args, hg: 'dgl.DGLGraph'):
        return cls(args.num_layers, args.hidden_dim, args.out_dim, args.attn_dim, hg.ntypes, hg.etypes, args.bias, args.batchnorm, args.dropout)

    def __init__(self, num_layers, hidden_dim, out_dim, attn_dim, ntypes, etypes, bias, batchnorm, dropout):
        super(ieHGCN, self).__init__()
        self.num_layers = num_layers
        self.activation = F.elu
        self.hgcn_layers = nn.ModuleList()
        for i in range(0, num_layers - 1):
            self.hgcn_layers.append(ieHGCNConv(hidden_dim, hidden_dim, attn_dim, ntypes, etypes, self.activation, bias, batchnorm, dropout))
        self.hgcn_layers.append(ieHGCNConv(hidden_dim, out_dim, attn_dim, ntypes, etypes, None, False, False, 0.0))

    def forward(self, hg, h_dict):
        """
        The forward part of the ieHGCN.
        
        Parameters
        ----------
        hg : object
            the dgl heterogeneous graph
        h_dict: dict
            the feature dict of different node types
            
        Returns
        -------
        dict
            The embeddings after the output projection.
        """
        if hasattr(hg, 'ntypes'):
            for l in range(self.num_layers):
                h_dict = self.hgcn_layers[l](hg, h_dict)
        else:
            for layer, block in zip(self.hgcn_layers, hg):
                h_dict = layer(block, h_dict)
        return h_dict


class lightGCN(BaseModel):
    """
        This module lightGCN was introduced in `lightGCN <https://dl.acm.org/doi/abs/10.1145/3397271.3401063>`__.

        The difference with GCN is that aggregate the entity representation and its neighborhood representation into the entity's embedding, but don't use feature transformation and nonlinear
        activation.
        The message function is defined as follow:

        :math:`\\mathbf{e}_u^{(k+1)}=\\operatorname{AGG}\\left(\\mathbf{e}_u^{(k)},\\left\\{\\mathbf{e}_i^{(k)}: i \\in \\mathcal{N}_u\\right\\}\\right)`

        The AGG is an aggregation function — the core of graph convolution — that considers the k-th layer’s representation of the target node and its neighbor nodes.


        In LightGCN, we adopt the simple weighted sum aggregator and abandon the use of feature transformation and nonlinear activation.
        :math:`\\mathbf{e}_u^{(k+1)}=\\sum_{i \\in \\mathcal{N}_u} \\frac{1}{\\sqrt{\\left|\\mathcal{N}_u\\right|} \\sqrt{\\left|\\mathcal{N}_i\\right|}}`
        :math:`\\mathbf{e}_i^{(k)}, \\\\ & \\mathbf{e}_i^{(k+1)}=\\sum_{u \\in \\mathcal{N}_i} \\frac{1}{\\sqrt{\\left|\\mathcal{N}_i\\right|} \\sqrt{\\left|\\mathcal{N}_u\\right|}} \\mathbf{e}_u^{(k)}`

        In the above equations, :math:`\\sigma` is the nonlinear function and
        :math:`\\mathrm{W}` and :math:`\\mathrm{b}` are transformation weight and bias.
        the representation of an item is bound up with its neighbors by aggregation

        The model prediction is defined as the inner product of user and
        item final representations:

        :math:`\\hat{y}_{u i}=\\mathbf{e}_u^T \\mathbf{e}_i`

        Parameters
        ----------
        g : DGLGraph
            A knowledge Graph preserves relationships between entities
        args : Config
            Model's config
        """

    @classmethod
    def build_model_from_args(cls, args, g):
        return cls(g, args)

    def __init__(self, g, args, **kwargs):
        super(lightGCN, self).__init__()
        self.g = g['g']
        self.num_nodes = self.g.shape[0]
        self.num_user = g['user_num']
        self.num_item = g['item_num']
        self.embedding_dim = args.embedding_size
        self.num_layers = args.num_layers
        self.alpha = 1.0 / (self.num_layers + 1)
        if isinstance(self.alpha, Tensor):
            assert self.alpha.size(0) == self.num_layers + 1
        else:
            self.alpha = th.tensor([self.alpha] * (self.num_layers + 1))
        self.embedding = Embedding(self.num_nodes, self.embedding_dim)
        self.embedding_user = th.nn.Embedding(num_embeddings=self.num_user, embedding_dim=self.embedding_dim)
        self.embedding_item = th.nn.Embedding(num_embeddings=self.num_item, embedding_dim=self.embedding_dim)
        nn.init.normal_(self.embedding_user.weight, std=0.1)
        nn.init.normal_(self.embedding_item.weight, std=0.1)
        self.f = nn.Sigmoid()
        self.reset_parameters()

    def reset_parameters(self):
        """Resets all learnable parameters of the module."""
        th.nn.init.normal_(self.embedding.weight, std=0.1)

    def computer(self):
        """
        propagate methods for lightGCN
        """
        all_emb = self.embedding.weight
        embs = [all_emb]
        g_droped = self.g
        for layer in range(self.num_layers):
            all_emb = th.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        embs = th.stack(embs, dim=1)
        light_out = th.mean(embs, dim=1)
        users, items = th.split(light_out, [self.num_user, self.num_item])
        return users, items

    def getUsersRating(self, users):
        all_users, all_items = self.computer()
        users_emb = all_users[users.long()]
        items_emb = all_items
        rating = self.f(th.matmul(users_emb, items_emb.t()))
        return rating

    def getEmbedding(self, users, pos_items, neg_items):
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego

    def bpr_loss(self, users, pos, neg):
        users_emb, pos_emb, neg_emb, userEmb0, posEmb0, negEmb0 = self.getEmbedding(users.long(), pos.long(), neg.long())
        reg_loss = 1 / 2 * (userEmb0.norm(2).pow(2) + posEmb0.norm(2).pow(2) + negEmb0.norm(2).pow(2)) / float(len(users))
        pos_scores = th.mul(users_emb, pos_emb)
        pos_scores = th.sum(pos_scores, dim=1)
        neg_scores = th.mul(users_emb, neg_emb)
        neg_scores = th.sum(neg_scores, dim=1)
        loss = th.mean(th.nn.functional.softplus(neg_scores - pos_scores))
        return loss, reg_loss


def mrr(y_pred, y_true):
    sorted_indices = th.argsort(y_pred, dim=1, descending=True)
    sorted_true = th.gather(y_true, 1, sorted_indices)
    first_hit = (sorted_true > 0) & (sorted_true.cumsum(dim=1) == sorted_true)
    ranks = th.nonzero(first_hit, as_tuple=False)[:, 1] + 1
    return th.mean(1.0 / ranks.float()).item()


def ndcg(pred_labels, true_relevance):
    k = true_relevance.shape[1]
    indices = pred_labels.argsort(descending=True, dim=1)
    sorted_true_relevance = torch.gather(true_relevance, 1, indices)
    discounts = torch.log2(torch.arange(k, device=true_relevance.device).float() + 2.0)
    dcg = (sorted_true_relevance[:, :k] / discounts).sum(dim=1)
    true_indices = true_relevance.argsort(descending=True, dim=1)
    ideal_sorted_relevance = torch.gather(true_relevance, 1, true_indices)
    idcg = (ideal_sorted_relevance[:, :k] / discounts).sum(dim=1)
    idcg[idcg == 0] = 1
    ndcg = dcg / idcg
    return ndcg.mean().item()


class Classifier(nn.Module):

    def __init__(self, n_in, n_out):
        super(Classifier, self).__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.linear = nn.Linear(n_in, n_out, bias=False)
        self.criterion = torch.nn.KLDivLoss(reduction='batchmean')
        nn.init.xavier_uniform_(self.linear.weight)

    def get_parameters(self):
        ml = list()
        ml.append({'params': self.linear.parameters()})
        return ml

    def forward(self, x):
        y = self.linear(x)
        return torch.log_softmax(y, dim=-1)

    def calc_loss(self, y_pred, y_true):
        return self.criterion(y_pred, y_true)

    def calc_acc(self, y_pred, y_true, metric):
        if metric not in ['ndcg', 'mrr']:
            raise 'metric not supported'
        elif metric == 'ndcg':
            return ndcg(y_pred, y_true)
        elif metric == 'mrr':
            return mrr(y_pred, y_true)


class HeteroDotProductPredictor(th.nn.Module):
    """
    References: `documentation of dgl <https://docs.dgl.ai/guide/training-link.html#heterogeneous-graphs>_`

    """

    def forward(self, edge_subgraph, x, *args, **kwargs):
        """
        Parameters
        ----------
        edge_subgraph: dgl.Heterograph
            the prediction graph only contains the edges of the target link
        x: dict[str: th.Tensor]
            the embedding dict. The key only contains the nodes involving with the target link.

        Returns
        -------
        score: th.Tensor
            the prediction of the edges in edge_subgraph
        """
        with edge_subgraph.local_scope():
            for ntype in edge_subgraph.ntypes:
                edge_subgraph.nodes[ntype].data['x'] = x[ntype]
                for etype in edge_subgraph.canonical_etypes:
                    edge_subgraph.apply_edges(dgl.function.u_dot_v('x', 'x', 'score'), etype=etype)
            score = edge_subgraph.edata['score']
            if isinstance(score, dict):
                result = []
                for _, value in score.items():
                    result.append(value)
                score = th.cat(result)
            return score.squeeze()


class HeteroDistMultPredictor(th.nn.Module):

    def forward(self, edge_subgraph, x, r_embedding, *args, **kwargs):
        """
        DistMult factorization (Yang et al. 2014) as the scoring function,
        which is known to perform well on standard link prediction benchmarks when used on its own.

        In DistMult, every relation r is associated with a diagonal matrix :math:`R_{r} \\in \\mathbb{R}^{d 	imes d}`
        and a triple (s, r, o) is scored as

        .. math::
            f(s, r, o)=e_{s}^{T} R_{r} e_{o}

        Parameters
        ----------
        edge_subgraph: dgl.Heterograph
            the prediction graph only contains the edges of the target link
        x: dict[str: th.Tensor]
            the node embedding dict. The key only contains the nodes involving with the target link.
        r_embedding: th.Tensor
            the all relation types embedding

        Returns
        -------
        score: th.Tensor
            the prediction of the edges in edge_subgraph
        """
        with edge_subgraph.local_scope():
            for ntype in edge_subgraph.ntypes:
                edge_subgraph.nodes[ntype].data['x'] = x[ntype]
            for etype in edge_subgraph.canonical_etypes:
                e = r_embedding[etype[1]]
                n = edge_subgraph.num_edges(etype)
                if 1 == len(edge_subgraph.canonical_etypes):
                    edge_subgraph.edata['e'] = e.expand(n, -1)
                else:
                    edge_subgraph.edata['e'] = {etype: e.expand(n, -1)}
                edge_subgraph.apply_edges(dgl.function.u_mul_e('x', 'e', 's'), etype=etype)
                edge_subgraph.apply_edges(dgl.function.e_mul_v('s', 'x', 'score'), etype=etype)
            score = edge_subgraph.edata['score']
            if isinstance(score, dict):
                result = []
                for _, value in score.items():
                    result.append(th.sum(value, dim=1))
                score = th.cat(result)
            else:
                score = th.sum(score, dim=1)
            return score


def eps2p(epsilon, n=2):
    return np.e ** epsilon / (np.e ** epsilon + n - 1)


def perturbation_test(value, perturbed_value, epsilon):
    value = np.array(value)
    perturbed_value = np.array(perturbed_value)
    per_eps = epsilon
    rnd = np.random.random(value.shape)
    p = eps2p(per_eps)
    return np.where(rnd < p, value, np.ones(value.shape) * perturbed_value)


class Client(nn.Module):

    def __init__(self, user_id, item_id, args):
        super().__init__()
        self.device = args.device
        self.user_id = user_id
        self.item_id = item_id

    def negative_sample(self, total_item_num):
        """生成item负样本集合"""
        item_neg_ind = []
        for _ in self.item_id:
            neg_item = np.random.randint(1, total_item_num)
            while neg_item in self.item_id:
                neg_item = np.random.randint(1, total_item_num)
            item_neg_ind.append(neg_item)
        """生成item负样本集合end"""
        return item_neg_ind

    def negative_sample_with_augment(self, total_item_num, sampled_items):
        item_set = self.item_id + sampled_items
        """生成item负样本集合"""
        item_neg_ind = []
        for _ in item_set:
            neg_item = np.random.randint(1, total_item_num)
            while neg_item in item_set:
                neg_item = np.random.randint(1, total_item_num)
            item_neg_ind.append(neg_item)
        """生成item负样本集合end"""
        return item_neg_ind

    def sample_item_augment(self, item_num):
        ls = [i for i in range(item_num) if i not in self.item_id]
        sampled_items = sample(ls, 5)
        return sampled_items

    def perturb_adj(self, value, label_author, author_label, label_count, shared_knowledge_rep, eps1, eps2):
        groups = {}
        for item in self.item_id:
            group = author_label[item]
            if group not in groups.keys():
                groups[group] = [item]
            else:
                groups[group].append(item)
        """step1:EM"""
        num_groups = len(groups)
        quality = np.array([0.0] * len(label_author))
        G_s_u = groups.keys()
        if len(G_s_u) == 0:
            for group in label_author.keys():
                quality[group] = 1
            num_groups = 1
        else:
            for group in label_author.keys():
                qua = max([((cosine_similarity(shared_knowledge_rep[g].reshape(1, -1), shared_knowledge_rep[group].reshape(1, -1)) + 1) / 2.0) for g in G_s_u])
                quality[group] = qua
        EM_eps = eps1 / num_groups
        EM_p = EM_eps * quality / 2
        EM_p = softmax(EM_p)
        select_group_keys = np.random.choice(range(len(label_author)), size=len(groups), replace=False, p=EM_p)
        select_group_keys_temp = list(select_group_keys)
        degree_list = [len(v) for _, v in groups.items()]
        new_groups = {}
        for key in select_group_keys:
            key_temp = key
            if key_temp in groups.keys():
                new_groups[key_temp] = groups[key_temp]
                degree_list.remove(len(groups[key_temp]))
                select_group_keys_temp.remove(key_temp)
        for key in select_group_keys_temp:
            key_temp = key
            cur_degree = degree_list[0]
            if len(label_author[key_temp]) >= cur_degree:
                new_groups[key_temp] = random.sample(label_author[key_temp], cur_degree)
            else:
                new_groups[key_temp] = label_author[key_temp]
            degree_list.remove(cur_degree)
        groups = new_groups
        value = np.zeros_like(value)
        for group_id, items in groups.items():
            value[:, items] = 1
        """pure em"""
        """step2:rr"""
        all_items = set(range(len(author_label)))
        select_items = []
        for group_id, items in groups.items():
            select_items.extend(label_author[group_id])
        mask_rr = list(all_items - set(select_items))
        """rr"""
        value_rr = perturbation_test(value, 1 - value, eps2)
        value_rr[:, mask_rr] = 0
        """dprr"""
        for group_id, items in groups.items():
            degree = len(items)
            n = len(label_author[group_id])
            p = eps2p(eps2)
            q = degree / (degree * (2 * p - 1) + n * (1 - p))
            rnd = np.random.random(value_rr.shape)
            dprr_results = np.where(rnd < q, value_rr, np.zeros(value_rr.shape))
            value_rr[:, label_author[group_id]] = dprr_results[:, label_author[group_id]]
        return value_rr

    def update(self, model_user, model_item):
        self.model_user = copy.deepcopy(model_user)
        self.model_item = copy.deepcopy(model_item)

    def train_(self, hg, user_emb, item_emb):
        total_item_num = item_emb.weight.shape[0]
        user_emb = torch.clone(user_emb.weight).detach()
        item_emb = torch.clone(item_emb.weight).detach()
        user_emb.requires_grad = True
        item_emb.requires_grad = True
        user_emb.grad = torch.zeros_like(user_emb)
        item_emb.grad = torch.zeros_like(item_emb)
        self.model_user.train()
        self.model_item.train()
        sampled_item = self.sample_item_augment(total_item_num)
        item_neg_id = self.negative_sample_with_augment(total_item_num, sampled_item)
        logits_user = self.model_user(hg, user_emb)
        logits_item = self.model_item(hg, item_emb)
        cur_user = logits_user[self.user_id]
        cur_item_pos = logits_item[self.item_id + sampled_item]
        cur_item_neg = logits_item[item_neg_id]
        pos_scores = torch.sum(cur_user * cur_item_pos, dim=-1)
        neg_scores = torch.sum(cur_user * cur_item_neg, dim=-1)
        loss = -(pos_scores - neg_scores).sigmoid().log().sum()
        self.model_user.zero_grad()
        self.model_item.zero_grad()
        loss.backward()
        model_grad_user = []
        model_grad_item = []
        for param in list(self.model_user.parameters()):
            grad = param.grad
            model_grad_user.append(grad)
        for param in list(self.model_item.parameters()):
            grad = param.grad
            model_grad_item.append(grad)
        mask_item = item_emb.grad.sum(-1) != 0
        updated_items = np.array(range(item_emb.shape[0]))[mask_item.cpu()]
        item_grad = item_emb.grad[updated_items, :]
        mask_user = user_emb.grad.sum(-1) != 0
        updated_users = np.array(range(user_emb.shape[0]))[mask_user.cpu()]
        user_grad = user_emb.grad[updated_users, :]
        return {'user': (user_grad, updated_users), 'item': (item_grad, updated_items), 'model': (model_grad_user, model_grad_item)}, loss.detach()


def evaluate_ndcg(rating, ground_truth, top_k):
    _, rating_k = torch.topk(rating, top_k)
    rating_k = rating_k.cpu().tolist()
    dcg, idcg = 0.0, 0.0
    for i, v in enumerate(rating_k):
        if i < len(ground_truth):
            idcg += 1 / np.log2(2 + i)
        if v in ground_truth:
            dcg += 1 / np.log2(2 + i)
    ndcg = dcg / idcg
    return ndcg


def evaluate_recall(rating, ground_truth, top_k):
    _, rating_k = torch.topk(rating, top_k)
    rating_k = rating_k.cpu().tolist()
    hit = 0
    for i, v in enumerate(rating_k):
        if v in ground_truth:
            hit += 1
    recall = hit / len(ground_truth)
    return recall


class Server(nn.Module):

    def __init__(self, client_list, model, hg, features, args):
        super().__init__()
        self.device = args.device
        self.hg = hg
        self.client_list = client_list
        self.features = features
        self.model_user = model[0]
        self.model_item = model[1]
        self.user_emb = nn.Embedding(features[0].shape[0], features[0].shape[1])
        self.item_emb = nn.Embedding(features[1].shape[0], features[1].shape[1])
        self.user_emb.weight.data = Parameter(torch.Tensor(features[0]))
        self.item_emb.weight.data = Parameter(torch.Tensor(features[1]))
        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.logger = args.logger

    def aggregate(self, param_list):
        flag = False
        number = 0
        gradient_item = torch.zeros_like(self.item_emb.weight)
        gradient_user = torch.zeros_like(self.user_emb.weight)
        item_count = torch.zeros(self.item_emb.weight.shape[0])
        user_count = torch.zeros(self.user_emb.weight.shape[0])
        for parameter in param_list:
            model_grad_user, model_grad_item = parameter['model']
            item_grad, returned_items = parameter['item']
            user_grad, returned_users = parameter['user']
            num = len(returned_items)
            item_count[returned_items] += 1
            user_count[returned_users] += num
            number += num
            if not flag:
                flag = True
                gradient_model_user = []
                gradient_model_item = []
                gradient_item[returned_items, :] += item_grad * num
                gradient_user[returned_users, :] += user_grad * num
                for i in range(len(model_grad_user)):
                    gradient_model_user.append(model_grad_user[i] * num)
                for i in range(len(model_grad_item)):
                    gradient_model_item.append(model_grad_item[i] * num)
            else:
                gradient_item[returned_items, :] += item_grad * num
                gradient_user[returned_users, :] += user_grad * num
                for i in range(len(model_grad_user)):
                    gradient_model_user[i] += model_grad_user[i] * num
                for i in range(len(model_grad_item)):
                    gradient_model_item[i] += model_grad_item[i] * num
        item_count[item_count == 0] = 1
        user_count[user_count == 0] = 1
        gradient_item /= item_count.unsqueeze(1)
        gradient_user /= user_count.unsqueeze(1)
        for i in range(len(gradient_model_user)):
            gradient_model_user[i] = gradient_model_user[i] / number
        for i in range(len(gradient_model_item)):
            gradient_model_item[i] = gradient_model_item[i] / number
        ls_model_param_user = list(self.model_user.parameters())
        ls_model_param_item = list(self.model_item.parameters())
        for i in range(len(ls_model_param_user)):
            ls_model_param_user[i].data = ls_model_param_user[i].data - self.lr * gradient_model_user[i] - self.weight_decay * ls_model_param_user[i].data
        for i in range(len(ls_model_param_item)):
            ls_model_param_item[i].data = ls_model_param_item[i].data - self.lr * gradient_model_item[i] - self.weight_decay * ls_model_param_item[i].data
        item_index = gradient_item.sum(dim=-1) != 0
        user_index = gradient_user.sum(dim=-1) != 0
        with torch.no_grad():
            self.item_emb.weight[item_index] = self.item_emb.weight[item_index] - self.lr * gradient_item[item_index] - self.weight_decay * self.item_emb.weight[item_index]
            self.user_emb.weight[user_index] = self.user_emb.weight[user_index] - self.lr * gradient_user[user_index] - self.weight_decay * self.user_emb.weight[user_index]

    def distribute(self, client_list):
        for client in client_list:
            client.update(self.model_user, self.model_item)

    def predict(self, test_dataloader, epoch):
        hit_at_5 = []
        hit_at_10 = []
        ndcg_at_5 = []
        ndcg_at_10 = []
        self.model_item.eval()
        self.model_user.eval()
        logits_user = self.model_user(self.hg, self.user_emb.weight)
        logits_item = self.model_item(self.hg, self.item_emb.weight)
        for u, i, neg_i in test_dataloader:
            cur_user = logits_user[u]
            cur_item = logits_item[i]
            rating = torch.sum(cur_user * cur_item, dim=-1)
            for eva_idx, eva in enumerate(rating):
                cur_neg = logits_item[neg_i[eva_idx]]
                cur_rating_neg = torch.sum(cur_user[eva_idx] * cur_neg, dim=-1)
                cur_eva = torch.cat([cur_rating_neg, torch.unsqueeze(rating[eva_idx], 0)], dim=0)
                hit_at_5_ = evaluate_recall(cur_eva, [99], 5)
                hit_at_10_ = evaluate_recall(cur_eva, [99], 10)
                ndcg_at_5_ = evaluate_ndcg(cur_eva, [99], 5)
                ndcg_at_10_ = evaluate_ndcg(cur_eva, [99], 10)
                hit_at_5.append(hit_at_5_)
                hit_at_10.append(hit_at_10_)
                ndcg_at_5.append(ndcg_at_5_)
                ndcg_at_10.append(ndcg_at_10_)
        hit_at_5 = np.mean(np.array(hit_at_5)).item()
        hit_at_10 = np.mean(np.array(hit_at_10)).item()
        ndcg_at_5 = np.mean(np.array(ndcg_at_5)).item()
        ndcg_at_10 = np.mean(np.array(ndcg_at_10)).item()
        self.logger.info('Epoch: %d, hit_at_5 = %.4f, hit_at_10 = %.4f, ndcg_at_5 = %.4f, ndcg_at_10 = %.4f' % (epoch, hit_at_5, hit_at_10, ndcg_at_5, ndcg_at_10))
        return hit_at_5, hit_at_10, ndcg_at_5, ndcg_at_10


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (AttentionLayer,
     lambda: ([], {'in_dim': 4, 'hidden_dim': 4, 'dropout': 0.5, 'activation': torch.nn.ReLU()}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4]), torch.rand([4, 4]), torch.rand([4, 4])], {})),
    (AvgReadout,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Base_model,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (BatchNorm1dNode,
     lambda: ([], {'dim_in': 4}),
     lambda: ([torch.rand([4, 4, 4])], {})),
    (CLUBSample,
     lambda: ([], {'x_dim': 4, 'y_dim': 4, 'hidden_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (Classifier,
     lambda: ([], {'n_in': 4, 'n_out': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Contrast,
     lambda: ([], {'hidden_dim': 4, 'tau': 4, 'lam': 4}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (Conv1d1x1,
     lambda: ([], {'cin': 4, 'cout': 4, 'groups': 1}),
     lambda: ([torch.rand([4, 4, 4])], {})),
    (EncoderLayer,
     lambda: ([], {'d_model': 4, 'd_k': 4, 'd_v': 4, 'd_ff': 4, 'n_heads': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {})),
    (FeedForwardNet,
     lambda: ([], {'in_feats': 4, 'hidden': 4, 'out_feats': 4, 'num_layers': 1, 'dropout': 0.5}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Fusion,
     lambda: ([], {'embedding_dims': 4, 'profile_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (GCN_layer,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4]), torch.ones([4], dtype=torch.int64)], {})),
    (GeneralLinear,
     lambda: ([], {'in_features': 4, 'out_features': 4}),
     lambda: ([torch.rand([4, 4, 4])], {})),
    (GraphGenerator,
     lambda: ([], {'dim': 4}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4])], {})),
    (L2Norm,
     lambda: ([], {'dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4, 4])], {})),
    (Linear,
     lambda: ([], {'dim_in': 4, 'dim_out': 4, 'dropout': 0.5}),
     lambda: ([torch.rand([4, 4, 4])], {})),
    (LogReg,
     lambda: ([], {'ft_in': 4, 'nb_classes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (LogisticRegression,
     lambda: ([], {'num_dim': 4, 'num_class': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (MLP,
     lambda: ([], {'input_dim': 4, 'feature_dim': 4, 'hidden_dim': 4, 'output_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (MetapathLearner,
     lambda: ([], {'config': SimpleNamespace(item_embedding_dim=4)}),
     lambda: ([torch.rand([4]), torch.rand([4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (MetricCalcLayer,
     lambda: ([], {'nhid': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (MultiLinearLayer,
     lambda: ([], {'linear_list': [4, 4], 'dropout': 0.5}),
     lambda: ([torch.rand([4, 4, 4])], {})),
    (Multi_level,
     lambda: ([], {}),
     lambda: ([], {})),
    (PoswiseFeedForwardNet,
     lambda: ([], {'d_model': 4, 'd_ff': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (RelationCrossing,
     lambda: ([], {'in_feats': 4, 'out_feats': 4, 'num_heads': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (SIGN,
     lambda: ([], {'in_feats': 4, 'hidden': 4, 'out_feats': 4, 'num_hops': 4, 'num_layers': 1, 'dropout': 0.5, 'input_drop': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (SelfAttention,
     lambda: ([], {'hidden_dim': 4, 'attn_drop': 0.5, 'txt': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (SemanticAttention,
     lambda: ([], {'in_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (SparseInputLinear,
     lambda: ([], {'inp_dim': 4, 'out_dim': 4}),
     lambda: ([torch.rand([4, 4])], {})),
    (Transformer,
     lambda: ([], {'n_channels': 4}),
     lambda: ([torch.rand([4, 4, 4])], {})),
    (WeightedAggregator,
     lambda: ([], {'num_feats': 4, 'in_feats': 4, 'num_hops': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
]

