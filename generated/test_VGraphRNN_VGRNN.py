
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


import numpy as np


import torch


import torch.nn as nn


import torch.utils


import torch.utils.data


from torchvision import datasets


from torchvision import transforms


from torch.autograd import Variable


import matplotlib.pyplot as plt


from scipy.ndimage import rotate


from torch.distributions.uniform import Uniform


from torch.distributions.normal import Normal


import scipy.sparse as sp


from scipy.linalg import block_diag


from torch.nn.parameter import Parameter


from torch.nn.modules.module import Module


import torch.nn.functional as F


import copy


import time


import scipy.io as sio


import inspect


from sklearn.metrics import roc_auc_score


from sklearn.metrics import average_precision_score


def scatter_(name, src, index, dim_size=None):
    """Aggregates all values from the :attr:`src` tensor at the indices
    specified in the :attr:`index` tensor along the first dimension.
    If multiple indices reference the same location, their contributions
    are aggregated according to :attr:`name` (either :obj:`"add"`,
    :obj:`"mean"` or :obj:`"max"`).
    Args:
        name (string): The aggregation to use (:obj:`"add"`, :obj:`"mean"`,
            :obj:`"max"`).
        src (Tensor): The source tensor.
        index (LongTensor): The indices of elements to scatter.
        dim_size (int, optional): Automatically create output tensor with size
            :attr:`dim_size` in the first dimension. If set to :attr:`None`, a
            minimal sized output tensor is returned. (default: :obj:`None`)
    :rtype: :class:`Tensor`
    """
    assert name in ['add', 'mean', 'max']
    op = getattr(torch_scatter, 'scatter_{}'.format(name))
    fill_value = -1e+38 if name is 'max' else 0
    out = op(src, index, 0, None, dim_size, fill_value)
    if isinstance(out, tuple):
        out = out[0]
    if name is 'max':
        out[out == fill_value] = 0
    return out


class MessagePassing(torch.nn.Module):
    """Base class for creating message passing layers
    .. math::
        \\mathbf{x}_i^{\\prime} = \\gamma_{\\mathbf{\\Theta}} \\left( \\mathbf{x}_i,
        \\square_{j \\in \\mathcal{N}(i)} \\, \\phi_{\\mathbf{\\Theta}}
        \\left(\\mathbf{x}_i, \\mathbf{x}_j,\\mathbf{e}_{i,j}\\right) \\right),
    where :math:`\\square` denotes a differentiable, permutation invariant
    function, *e.g.*, sum, mean or max, and :math:`\\gamma_{\\mathbf{\\Theta}}`
    and :math:`\\phi_{\\mathbf{\\Theta}}` denote differentiable functions such as
    MLPs.
    See `here <https://rusty1s.github.io/pytorch_geometric/build/html/notes/
    create_gnn.html>`__ for the accompanying tutorial.
    """

    def __init__(self, aggr='add'):
        super(MessagePassing, self).__init__()
        self.message_args = inspect.getargspec(self.message)[0][1:]
        self.update_args = inspect.getargspec(self.update)[0][2:]

    def propagate(self, aggr, edge_index, **kwargs):
        """The initial call to start propagating messages.
        Takes in an aggregation scheme (:obj:`"add"`, :obj:`"mean"` or
        :obj:`"max"`), the edge indices, and all additional data which is
        needed to construct messages and to update node embeddings."""
        assert aggr in ['add', 'mean', 'max']
        kwargs['edge_index'] = edge_index
        size = None
        message_args = []
        for arg in self.message_args:
            if arg[-2:] == '_i':
                tmp = kwargs[arg[:-2]]
                size = tmp.size(0)
                message_args.append(tmp[edge_index[0]])
            elif arg[-2:] == '_j':
                tmp = kwargs[arg[:-2]]
                size = tmp.size(0)
                message_args.append(tmp[edge_index[1]])
            else:
                message_args.append(kwargs[arg])
        update_args = [kwargs[arg] for arg in self.update_args]
        out = self.message(*message_args)
        out = scatter_(aggr, out, edge_index[0], dim_size=size)
        out = self.update(out, *update_args)
        return out

    def message(self, x_j):
        """Constructs messages in analogy to :math:`\\phi_{\\mathbf{\\Theta}}`
        for each edge in :math:`(i,j) \\in \\mathcal{E}`.
        Can take any argument which was initially passed to :meth:`propagate`.
        In addition, features can be lifted to the source node :math:`i` and
        target node :math:`j` by appending :obj:`_i` or :obj:`_j` to the
        variable name, *.e.g.* :obj:`x_i` and :obj:`x_j`."""
        return x_j

    def update(self, aggr_out):
        """Updates node embeddings in analogy to
        :math:`\\gamma_{\\mathbf{\\Theta}}` for each node
        :math:`i \\in \\mathcal{V}`.
        Takes in the output of aggregation as first argument and any argument
        which was initially passed to :meth:`propagate`."""
        return aggr_out


def glorot(tensor):
    stdv = math.sqrt(6.0 / (tensor.size(0) + tensor.size(1)))
    if tensor is not None:
        tensor.data.uniform_(-stdv, stdv)


def zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0)


class GCNConv(MessagePassing):

    def __init__(self, in_channels, out_channels, act=F.relu, improved=True, bias=False):
        super(GCNConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.act = act
        self.weight = Parameter(torch.Tensor(in_channels, out_channels))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        zeros(self.bias)

    def forward(self, x, edge_index, edge_weight=None):
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1),), dtype=x.dtype, device=x.device)
        edge_weight = edge_weight.view(-1)
        assert edge_weight.size(0) == edge_index.size(1)
        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))
        loop_weight = torch.full((x.size(0),), 1 if not self.improved else 2, dtype=x.dtype, device=x.device)
        edge_weight = torch.cat([edge_weight, loop_weight], dim=0)
        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=x.size(0))
        deg_inv = deg.pow(-0.5)
        deg_inv[deg_inv == float('inf')] = 0
        norm = deg_inv[row] * edge_weight * deg_inv[col]
        x = torch.matmul(x, self.weight)
        out = self.propagate('add', edge_index, x=x, norm=norm)
        return self.act(out)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def update(self, aggr_out):
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels, self.out_channels)


def uniform(size, tensor):
    stdv = 1.0 / math.sqrt(size)
    if tensor is not None:
        tensor.data.uniform_(-stdv, stdv)


class SAGEConv(torch.nn.Module):

    def __init__(self, in_channels, out_channels, pool='mean', act=F.relu, normalize=False, bias=False):
        super(SAGEConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize
        self.weight = Parameter(torch.Tensor(self.in_channels, out_channels))
        self.act = act
        self.pool = pool
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        size = self.weight.size(0)
        uniform(size, self.weight)
        uniform(size, self.bias)

    def forward(self, x, edge_index):
        edge_index, _ = remove_self_loops(edge_index)
        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        row, col = edge_index
        if self.pool == 'mean':
            out = torch.matmul(x, self.weight)
            if self.bias is not None:
                out = out + self.bias
            out = self.act(out)
            out = scatter_mean(out[col], row, dim=0, dim_size=out.size(0))
        elif self.pool == 'max':
            out = torch.matmul(x, self.weight)
            if self.bias is not None:
                out = out + self.bias
            out = self.act(out)
            out, _ = scatter_max(out[col], row, dim=0, dim_size=out.size(0))
        elif self.pool == 'add':
            x = torch.matmul(x, self.weight)
            if self.bias is not None:
                out = out + self.bias
            out = self.act(out)
            out = scatter_add(x[col], row, dim=0, dim_size=x.size(0))
        else:
            None
        if self.normalize:
            out = F.normalize(out, p=2, dim=-1)
        return out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels, self.out_channels)


def reset(nn):

    def _reset(item):
        if hasattr(item, 'reset_parameters'):
            item.reset_parameters()
    if nn is not None:
        if hasattr(nn, 'children') and len(list(nn.children())) > 0:
            for item in nn.children():
                _reset(item)
        else:
            _reset(nn)


class GINConv(torch.nn.Module):

    def __init__(self, nn, eps=0, train_eps=False):
        super(GINConv, self).__init__()
        self.nn = nn
        self.initial_eps = eps
        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.nn)
        self.eps.data.fill_(self.initial_eps)

    def forward(self, x, edge_index):
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        edge_index, _ = remove_self_loops(edge_index)
        row, col = edge_index
        out = scatter_add(x[col], row, dim=0, dim_size=x.size(0))
        out = (1 + self.eps) * x + out
        out = self.nn(out)
        return out

    def __repr__(self):
        return '{}(nn={})'.format(self.__class__.__name__, self.nn)


class graph_gru_sage(nn.Module):

    def __init__(self, input_size, hidden_size, n_layer, bias=True):
        super(graph_gru_sage, self).__init__()
        self.hidden_size = hidden_size
        self.n_layer = n_layer
        self.weight_xz = []
        self.weight_hz = []
        self.weight_xr = []
        self.weight_hr = []
        self.weight_xh = []
        self.weight_hh = []
        for i in range(self.n_layer):
            if i == 0:
                self.weight_xz.append(SAGEConv(input_size, hidden_size, act=lambda x: x, bias=bias))
                self.weight_hz.append(SAGEConv(hidden_size, hidden_size, act=lambda x: x, bias=bias))
                self.weight_xr.append(SAGEConv(input_size, hidden_size, act=lambda x: x, bias=bias))
                self.weight_hr.append(SAGEConv(hidden_size, hidden_size, act=lambda x: x, bias=bias))
                self.weight_xh.append(SAGEConv(input_size, hidden_size, act=lambda x: x, bias=bias))
                self.weight_hh.append(SAGEConv(hidden_size, hidden_size, act=lambda x: x, bias=bias))
            else:
                self.weight_xz.append(SAGEConv(hidden_size, hidden_size, act=lambda x: x, bias=bias))
                self.weight_hz.append(SAGEConv(hidden_size, hidden_size, act=lambda x: x, bias=bias))
                self.weight_xr.append(SAGEConv(hidden_size, hidden_size, act=lambda x: x, bias=bias))
                self.weight_hr.append(SAGEConv(hidden_size, hidden_size, act=lambda x: x, bias=bias))
                self.weight_xh.append(SAGEConv(hidden_size, hidden_size, act=lambda x: x, bias=bias))
                self.weight_hh.append(SAGEConv(hidden_size, hidden_size, act=lambda x: x, bias=bias))

    def forward(self, inp, edgidx, h):
        h_out = torch.zeros(h.size())
        for i in range(self.n_layer):
            if i == 0:
                z_g = torch.sigmoid(self.weight_xz[i](inp, edgidx) + self.weight_hz[i](h[i], edgidx))
                r_g = torch.sigmoid(self.weight_xr[i](inp, edgidx) + self.weight_hr[i](h[i], edgidx))
                h_tilde_g = torch.tanh(self.weight_xh[i](inp, edgidx) + self.weight_hh[i](r_g * h[i], edgidx))
                h_out[i] = z_g * h[i] + (1 - z_g) * h_tilde_g
            else:
                z_g = torch.sigmoid(self.weight_xz[i](h_out[i - 1], edgidx) + self.weight_hz[i](h[i], edgidx))
                r_g = torch.sigmoid(self.weight_xr[i](h_out[i - 1], edgidx) + self.weight_hr[i](h[i], edgidx))
                h_tilde_g = torch.tanh(self.weight_xh[i](h_out[i - 1], edgidx) + self.weight_hh[i](r_g * h[i], edgidx))
                h_out[i] = z_g * h[i] + (1 - z_g) * h_tilde_g
        out = h_out
        return out, h_out


class graph_gru_gcn(nn.Module):

    def __init__(self, input_size, hidden_size, n_layer, bias=True):
        super(graph_gru_gcn, self).__init__()
        self.hidden_size = hidden_size
        self.n_layer = n_layer
        self.weight_xz = []
        self.weight_hz = []
        self.weight_xr = []
        self.weight_hr = []
        self.weight_xh = []
        self.weight_hh = []
        for i in range(self.n_layer):
            if i == 0:
                self.weight_xz.append(GCNConv(input_size, hidden_size, act=lambda x: x, bias=bias))
                self.weight_hz.append(GCNConv(hidden_size, hidden_size, act=lambda x: x, bias=bias))
                self.weight_xr.append(GCNConv(input_size, hidden_size, act=lambda x: x, bias=bias))
                self.weight_hr.append(GCNConv(hidden_size, hidden_size, act=lambda x: x, bias=bias))
                self.weight_xh.append(GCNConv(input_size, hidden_size, act=lambda x: x, bias=bias))
                self.weight_hh.append(GCNConv(hidden_size, hidden_size, act=lambda x: x, bias=bias))
            else:
                self.weight_xz.append(GCNConv(hidden_size, hidden_size, act=lambda x: x, bias=bias))
                self.weight_hz.append(GCNConv(hidden_size, hidden_size, act=lambda x: x, bias=bias))
                self.weight_xr.append(GCNConv(hidden_size, hidden_size, act=lambda x: x, bias=bias))
                self.weight_hr.append(GCNConv(hidden_size, hidden_size, act=lambda x: x, bias=bias))
                self.weight_xh.append(GCNConv(hidden_size, hidden_size, act=lambda x: x, bias=bias))
                self.weight_hh.append(GCNConv(hidden_size, hidden_size, act=lambda x: x, bias=bias))

    def forward(self, inp, edgidx, h):
        h_out = torch.zeros(h.size())
        for i in range(self.n_layer):
            if i == 0:
                z_g = torch.sigmoid(self.weight_xz[i](inp, edgidx) + self.weight_hz[i](h[i], edgidx))
                r_g = torch.sigmoid(self.weight_xr[i](inp, edgidx) + self.weight_hr[i](h[i], edgidx))
                h_tilde_g = torch.tanh(self.weight_xh[i](inp, edgidx) + self.weight_hh[i](r_g * h[i], edgidx))
                h_out[i] = z_g * h[i] + (1 - z_g) * h_tilde_g
            else:
                z_g = torch.sigmoid(self.weight_xz[i](h_out[i - 1], edgidx) + self.weight_hz[i](h[i], edgidx))
                r_g = torch.sigmoid(self.weight_xr[i](h_out[i - 1], edgidx) + self.weight_hr[i](h[i], edgidx))
                h_tilde_g = torch.tanh(self.weight_xh[i](h_out[i - 1], edgidx) + self.weight_hh[i](r_g * h[i], edgidx))
                h_out[i] = z_g * h[i] + (1 - z_g) * h_tilde_g
        out = h_out
        return out, h_out


class InnerProductDecoder(nn.Module):

    def __init__(self, act=torch.sigmoid, dropout=0.0):
        super(InnerProductDecoder, self).__init__()
        self.act = act
        self.dropout = dropout

    def forward(self, inp):
        inp = F.dropout(inp, self.dropout, training=self.training)
        x = torch.transpose(inp, dim0=0, dim1=1)
        x = torch.mm(inp, x)
        return self.act(x)


class VGRNN(nn.Module):

    def __init__(self, x_dim, h_dim, z_dim, n_layers, eps, conv='GCN', bias=False):
        super(VGRNN, self).__init__()
        self.x_dim = x_dim
        self.eps = eps
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.n_layers = n_layers
        if conv == 'GCN':
            self.phi_x = nn.Sequential(nn.Linear(x_dim, h_dim), nn.ReLU())
            self.phi_z = nn.Sequential(nn.Linear(z_dim, h_dim), nn.ReLU())
            self.enc = GCNConv(h_dim + h_dim, h_dim)
            self.enc_mean = GCNConv(h_dim, z_dim, act=lambda x: x)
            self.enc_std = GCNConv(h_dim, z_dim, act=F.softplus)
            self.prior = nn.Sequential(nn.Linear(h_dim, h_dim), nn.ReLU())
            self.prior_mean = nn.Sequential(nn.Linear(h_dim, z_dim))
            self.prior_std = nn.Sequential(nn.Linear(h_dim, z_dim), nn.Softplus())
            self.rnn = graph_gru_gcn(h_dim + h_dim, h_dim, n_layers, bias)
        elif conv == 'SAGE':
            self.phi_x = nn.Sequential(nn.Linear(x_dim, h_dim), nn.ReLU())
            self.phi_z = nn.Sequential(nn.Linear(z_dim, h_dim), nn.ReLU())
            self.enc = SAGEConv(h_dim + h_dim, h_dim)
            self.enc_mean = SAGEConv(h_dim, z_dim, act=lambda x: x)
            self.enc_std = SAGEConv(h_dim, z_dim, act=F.softplus)
            self.prior = nn.Sequential(nn.Linear(h_dim, h_dim), nn.ReLU())
            self.prior_mean = nn.Sequential(nn.Linear(h_dim, z_dim))
            self.prior_std = nn.Sequential(nn.Linear(h_dim, z_dim), nn.Softplus())
            self.rnn = graph_gru_sage(h_dim + h_dim, h_dim, n_layers, bias)
        elif conv == 'GIN':
            self.phi_x = nn.Sequential(nn.Linear(x_dim, h_dim), nn.ReLU())
            self.phi_z = nn.Sequential(nn.Linear(z_dim, h_dim), nn.ReLU())
            self.enc = GINConv(nn.Sequential(nn.Linear(h_dim + h_dim, h_dim), nn.ReLU()))
            self.enc_mean = GINConv(nn.Sequential(nn.Linear(h_dim, z_dim)))
            self.enc_std = GINConv(nn.Sequential(nn.Linear(h_dim, z_dim), nn.Softplus()))
            self.prior = nn.Sequential(nn.Linear(h_dim, h_dim), nn.ReLU())
            self.prior_mean = nn.Sequential(nn.Linear(h_dim, z_dim))
            self.prior_std = nn.Sequential(nn.Linear(h_dim, z_dim), nn.Softplus())
            self.rnn = graph_gru_gcn(h_dim + h_dim, h_dim, n_layers, bias)

    def forward(self, x, edge_idx_list, adj_orig_dense_list, hidden_in=None):
        assert len(adj_orig_dense_list) == len(edge_idx_list)
        kld_loss = 0
        nll_loss = 0
        all_enc_mean, all_enc_std = [], []
        all_prior_mean, all_prior_std = [], []
        all_dec_t, all_z_t = [], []
        if hidden_in is None:
            h = Variable(torch.zeros(self.n_layers, x.size(1), self.h_dim))
        else:
            h = Variable(hidden_in)
        for t in range(x.size(0)):
            phi_x_t = self.phi_x(x[t])
            enc_t = self.enc(torch.cat([phi_x_t, h[-1]], 1), edge_idx_list[t])
            enc_mean_t = self.enc_mean(enc_t, edge_idx_list[t])
            enc_std_t = self.enc_std(enc_t, edge_idx_list[t])
            prior_t = self.prior(h[-1])
            prior_mean_t = self.prior_mean(prior_t)
            prior_std_t = self.prior_std(prior_t)
            z_t = self._reparameterized_sample(enc_mean_t, enc_std_t)
            phi_z_t = self.phi_z(z_t)
            dec_t = self.dec(z_t)
            _, h = self.rnn(torch.cat([phi_x_t, phi_z_t], 1), edge_idx_list[t], h)
            nnodes = adj_orig_dense_list[t].size()[0]
            enc_mean_t_sl = enc_mean_t[0:nnodes, :]
            enc_std_t_sl = enc_std_t[0:nnodes, :]
            prior_mean_t_sl = prior_mean_t[0:nnodes, :]
            prior_std_t_sl = prior_std_t[0:nnodes, :]
            dec_t_sl = dec_t[0:nnodes, 0:nnodes]
            kld_loss += self._kld_gauss(enc_mean_t_sl, enc_std_t_sl, prior_mean_t_sl, prior_std_t_sl)
            nll_loss += self._nll_bernoulli(dec_t_sl, adj_orig_dense_list[t])
            all_enc_std.append(enc_std_t_sl)
            all_enc_mean.append(enc_mean_t_sl)
            all_prior_mean.append(prior_mean_t_sl)
            all_prior_std.append(prior_std_t_sl)
            all_dec_t.append(dec_t_sl)
            all_z_t.append(z_t)
        return kld_loss, nll_loss, all_enc_mean, all_prior_mean, h

    def dec(self, z):
        outputs = InnerProductDecoder(act=lambda x: x)(z)
        return outputs

    def reset_parameters(self, stdv=0.1):
        for weight in self.parameters():
            weight.data.normal_(0, stdv)

    def _init_weights(self, stdv):
        pass

    def _reparameterized_sample(self, mean, std):
        eps1 = torch.FloatTensor(std.size()).normal_()
        eps1 = Variable(eps1)
        return eps1.mul(std).add_(mean)

    def _kld_gauss(self, mean_1, std_1, mean_2, std_2):
        num_nodes = mean_1.size()[0]
        kld_element = 2 * torch.log(std_2 + self.eps) - 2 * torch.log(std_1 + self.eps) + (torch.pow(std_1 + self.eps, 2) + torch.pow(mean_1 - mean_2, 2)) / torch.pow(std_2 + self.eps, 2) - 1
        return 0.5 / num_nodes * torch.mean(torch.sum(kld_element, dim=1), dim=0)

    def _kld_gauss_zu(self, mean_in, std_in):
        num_nodes = mean_in.size()[0]
        std_log = torch.log(std_in + self.eps)
        kld_element = torch.mean(torch.sum(1 + 2 * std_log - mean_in.pow(2) - torch.pow(torch.exp(std_log), 2), 1))
        return -0.5 / num_nodes * kld_element

    def _nll_bernoulli(self, logits, target_adj_dense):
        temp_size = target_adj_dense.size()[0]
        temp_sum = target_adj_dense.sum()
        posw = float(temp_size * temp_size - temp_sum) / temp_sum
        norm = temp_size * temp_size / float((temp_size * temp_size - temp_sum) * 2)
        nll_loss_mat = F.binary_cross_entropy_with_logits(input=logits, target=target_adj_dense, pos_weight=posw, reduction='none')
        nll_loss = -1 * norm * torch.mean(nll_loss_mat, dim=[0, 1])
        return -nll_loss


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (InnerProductDecoder,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4])], {})),
]

