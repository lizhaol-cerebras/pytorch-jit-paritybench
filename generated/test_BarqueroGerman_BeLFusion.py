
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


from torch.utils.data import DataLoader


from torch.utils.data.dataloader import default_collate


from torch.utils.data.sampler import SubsetRandomSampler


from torch.utils.data import Dataset


import torch


import torch.nn as nn


from abc import abstractmethod


from numpy import inf


import time


import pandas as pd


from scipy.spatial.transform import Rotation as R


import matplotlib.pyplot as plt


from scipy.spatial.distance import pdist


from scipy.spatial.distance import squareform


from scipy import linalg


from scipy.stats import gaussian_kde


import torch.nn.functional as F


import random


from torch import nn


from torch.nn.utils import weight_norm


import enum


import math


import torch as th


from abc import ABC


import torch.distributed as dist


from torch.nn import Module


from torch.nn import Sequential


from torch.nn import Linear


from torch.nn.parameter import Parameter


from torch.nn import ModuleList


from torch.nn import ModuleDict


from torch.nn import GELU


from torch.nn import Tanh


from torch.nn import BatchNorm1d


import logging


from functools import reduce


from functools import partial


import pandas


from torch.optim import lr_scheduler


from itertools import repeat


from collections import OrderedDict


class BaseModel(nn.Module):

    def __init__(self, n_features, n_landmarks, obs_length, pred_length):
        super(BaseModel, self).__init__()
        self.n_features = n_features
        self.n_landmarks = n_landmarks
        self.obs_length = obs_length
        self.pred_length = pred_length
        self.total_length = obs_length + pred_length
    """
    Base class for all models
    """

    @abstractmethod
    def forward(self, *inputs):
        """
        Forward pass logic
        :return: Model output
        """
        raise NotImplementedError

    def get_params(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return params

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)


class BaseAutoencoder(BaseModel):

    def __init__(self, n_features, n_landmarks, obs_length, pred_length):
        super(BaseAutoencoder, self).__init__(n_features, n_landmarks, obs_length, pred_length)

    @abstractmethod
    def encode(self, x):
        raise NotImplementedError

    @abstractmethod
    def decode(self, x_start, h_y):
        raise NotImplementedError


def init_weights(module):
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight.data, mode='fan_out')
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.constant_(module.weight, 1.0)
        nn.init.constant_(module.bias, 0.0)
    elif isinstance(module, nn.Linear):
        for name, param in module.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_uniform_(param)
    elif isinstance(module, nn.LSTM) or isinstance(module, nn.RNN) or isinstance(module, nn.LSTMCell) or isinstance(module, nn.RNNCell) or isinstance(module, nn.GRU) or isinstance(module, nn.GRUCell):
        DIV = 3 if isinstance(module, nn.GRU) or isinstance(module, nn.GRUCell) else 4
        for name, param in module.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
                if isinstance(module, nn.LSTMCell) or isinstance(module, nn.LSTM):
                    n = param.size(0)
                    start, end = n // DIV, n // 2
                    param.data[start:end].fill_(1.0)
                elif isinstance(module, nn.GRU) or isinstance(module, nn.GRUCell):
                    end = param.size(0) // DIV
                    param.data[:end].fill_(-1.0)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)
                if isinstance(module, nn.LSTMCell) or isinstance(module, nn.LSTM) or isinstance(module, nn.GRU) or isinstance(module, nn.GRUCell):
                    if 'weight_ih' in name:
                        mul = param.shape[0] // DIV
                        for idx in range(DIV):
                            nn.init.xavier_uniform_(param[idx * mul:(idx + 1) * mul])
                    elif 'weight_hh' in name:
                        mul = param.shape[0] // DIV
                        for idx in range(DIV):
                            nn.init.orthogonal_(param[idx * mul:(idx + 1) * mul])
    else:
        None


nl = {'tanh': nn.Tanh, 'relu': nn.ReLU, 'sigmoid': nn.Sigmoid, 'elu': nn.ELU, 'selu': nn.SELU, 'softplus': nn.Softplus, 'softsign': nn.Softsign, 'leaky_relu': nn.LeakyReLU, 'none': lambda x: x}


class BasicMLP(nn.Module):

    def __init__(self, input_dim, output_dim, hidden_dims=[], dropout=0.5, non_linearities='relu'):
        super(BasicMLP, self).__init__()
        self.non_linearities = non_linearities
        self.dropout = nn.Dropout(dropout)
        self.nl = nl[non_linearities]()
        self.denses = None
        hidden_dims = hidden_dims + [output_dim]
        seqs = []
        for i in range(len(hidden_dims)):
            linear = nn.Linear(input_dim if i == 0 else hidden_dims[i - 1], hidden_dims[i])
            init_weights(linear)
            seqs.append(nn.Sequential(self.dropout, linear, self.nl))
        self.denses = nn.Sequential(*seqs)

    def forward(self, x):
        return self.denses(x) if self.denses is not None else x


class MLP(nn.Module):

    def __init__(self, input_dim, hidden_dims=(128, 128), activation='tanh'):
        super(MLP, self).__init__()
        if activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'relu':
            self.activation = torch.relu
        elif activation == 'sigmoid':
            self.activation = torch.sigmoid
        self.out_dim = hidden_dims[-1]
        self.affine_layers = nn.ModuleList()
        last_dim = input_dim
        for nh in hidden_dims:
            self.affine_layers.append(nn.Linear(last_dim, nh))
            last_dim = nh

    def forward(self, x):
        for affine in self.affine_layers:
            x = self.activation(affine(x))
        return x


def batch_to(dst, *args):
    return [(x if x is not None else None) for x in args]


zeros = torch.zeros


class RNN(nn.Module):

    def __init__(self, input_dim, out_dim, cell_type='lstm', bi_dir=False):
        super(RNN, self).__init__()
        self.input_dim = input_dim
        self.out_dim = out_dim
        self.cell_type = cell_type
        self.bi_dir = bi_dir
        self.mode = 'batch'
        rnn_cls = nn.LSTMCell if cell_type == 'lstm' else nn.GRUCell
        hidden_dim = out_dim // 2 if bi_dir else out_dim
        self.rnn_f = rnn_cls(self.input_dim, hidden_dim)
        if bi_dir:
            self.rnn_b = rnn_cls(self.input_dim, hidden_dim)
        self.hx, self.cx = None, None

    def set_mode(self, mode):
        self.mode = mode

    def initialize(self, batch_size=1, hx=None, cx=None):
        if self.mode == 'step':
            self.hx = zeros((batch_size, self.rnn_f.hidden_size)) if hx is None else hx
            if self.cell_type == 'lstm':
                self.cx = zeros((batch_size, self.rnn_f.hidden_size)) if cx is None else cx

    def forward(self, x):
        if self.mode == 'step':
            self.hx, self.cx = batch_to(x.device, self.hx, self.cx)
            if self.cell_type == 'lstm':
                self.hx, self.cx = self.rnn_f(x, (self.hx, self.cx))
            else:
                self.hx = self.rnn_f(x, self.hx)
            rnn_out = self.hx
        else:
            rnn_out_f = self.batch_forward(x)
            if not self.bi_dir:
                return rnn_out_f
            rnn_out_b = self.batch_forward(x, reverse=True)
            rnn_out = torch.cat((rnn_out_f, rnn_out_b), 2)
        return rnn_out

    def batch_forward(self, x, reverse=False):
        rnn = self.rnn_b if reverse else self.rnn_f
        rnn_out = []
        hx = zeros((x.size(1), rnn.hidden_size), device=x.device)
        if self.cell_type == 'lstm':
            cx = zeros((x.size(1), rnn.hidden_size), device=x.device)
        ind = reversed(range(x.size(0))) if reverse else range(x.size(0))
        for t in ind:
            if self.cell_type == 'lstm':
                hx, cx = rnn(x[t, ...], (hx, cx))
            else:
                hx = rnn(x[t, ...], hx)
            rnn_out.append(hx.unsqueeze(0))
        if reverse:
            rnn_out.reverse()
        rnn_out = torch.cat(rnn_out, 0)
        return rnn_out


class NormConv2d(nn.Module):
    """
    Convolutional layer with l2 weight normalization and learned scaling parameters
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.beta = nn.Parameter(torch.zeros([1, out_channels, 1, 1], dtype=torch.float32))
        self.gamma = nn.Parameter(torch.ones([1, out_channels, 1, 1], dtype=torch.float32))
        self.conv = weight_norm(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding), name='weight')

    def forward(self, x):
        out = self.conv(x)
        out = self.gamma * out + self.beta
        return out


def reparametrize_logstd(mu, logstd):
    std = torch.exp(logstd)
    eps = torch.randn_like(std)
    return eps.mul(std) + mu


class BEncoder(nn.Module):

    def __init__(self, n_in, n_layers, dim_hidden, use_linear, dim_linear):
        super(BEncoder, self).__init__()
        self.rnn = nn.LSTM(input_size=n_in, hidden_size=dim_hidden, num_layers=n_layers, batch_first=True)
        self.n_layer = n_layers
        self.dim_hidden = dim_hidden
        self.hidden = self.init_hidden(bs=1, device='cpu')
        self.use_linear = use_linear
        self.linear = None
        if self.use_linear:
            self.linear = nn.Linear(self.dim_hidden, dim_linear)
        self.mu_fn = NormConv2d(self.dim_hidden, self.dim_hidden, 1)
        self.std_fn = NormConv2d(self.dim_hidden, self.dim_hidden, 1)

    def init_hidden(self, bs, device):
        self.hidden = torch.zeros((self.n_layer, bs, self.dim_hidden), device=device), torch.zeros((self.n_layer, bs, self.dim_hidden), device=device)

    def set_hidden(self, bs, device, hidden=None, cell=None):
        if hidden is None and cell is None:
            self.init_hidden(bs, device)
        elif hidden is None:
            self.hidden = torch.zeros_like(cell), cell
        elif cell is None:
            self.hidden = hidden, torch.zeros_like(hidden)
        else:
            self.hidden = hidden, cell

    def forward(self, x, sample=False):
        out, self.hidden = self.rnn(x, self.hidden)
        pre = self.hidden[0][-1]
        mu = self.mu_fn(pre.unsqueeze(dim=-1).unsqueeze(dim=-1)).squeeze(dim=-1).squeeze(dim=-1)
        logstd = self.std_fn(pre.unsqueeze(dim=-1).unsqueeze(dim=-1)).squeeze(dim=-1).squeeze(dim=-1)
        if sample:
            out = sample(mu)
        else:
            out = reparametrize_logstd(mu, logstd)
        return out, mu, logstd, pre


class ResidualRNNDecoder(nn.Module):

    def __init__(self, n_in, n_out, n_hidden, rnn_type='lstm', use_nin=False):
        super().__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.n_hidden = n_hidden
        self.rnn_type = rnn_type
        self.use_nin = use_nin
        if self.rnn_type == 'gru':
            self.rnn = nn.GRUCell(self.n_in, self.n_hidden)
            self.n_out = nn.Linear(self.n_hidden, self.n_out)
        else:
            self.rnn = nn.LSTMCell(self.n_in, self.n_hidden)
            self.n_out = nn.Linear(self.n_hidden, self.n_out)
        if self.use_nin:
            self.n_in = nn.Linear(self.n_in, self.n_in)
        self.init_hidden(bs=1, device='cpu')

    def forward(self, x):
        if len(x.shape) == 3:
            x = x.squeeze(dim=1)
        elif len(x.shape) != 2:
            raise TypeError('invalid shape of tensor.')
        res = x
        if self.use_nin:
            x = self.n_in(x)
        if self.rnn_type == 'lstm':
            self.hidden = self.rnn(x, self.hidden)
            out_rnn = self.hidden[0]
        else:
            self.hidden = self.rnn(x, self.hidden)
            out_rnn = self.hidden
        out = self.n_out(out_rnn)
        return out + res, res

    def forward_noresidual(self, x):
        if len(x.shape) == 3:
            x = x.squeeze(dim=1)
        elif len(x.shape) != 2:
            raise TypeError('invalid shape of tensor.')
        if self.use_nin:
            x = self.n_in(x)
        if self.rnn_type == 'lstm':
            self.hidden = self.rnn(x, self.hidden)
            out_rnn = self.hidden[0]
        else:
            self.hidden = self.rnn(x, self.hidden)
            out_rnn = self.hidden
        return self.n_out(out_rnn)

    def init_hidden(self, bs, device):
        if self.rnn_type == 'lstm':
            self.hidden = torch.zeros((bs, self.n_hidden), device=device), torch.zeros((bs, self.n_hidden), device=device)
        elif self.rnn_type == 'gru':
            self.hidden = torch.zeros((bs, self.n_hidden), device=device)

    def set_hidden(self, bs, device, hidden=None, cell=None):
        if self.rnn_type == 'lstm':
            if hidden is None and cell is None:
                self.init_hidden(bs, device)
            elif hidden is None:
                self.hidden = torch.zeros_like(cell), cell
            elif cell is None:
                self.hidden = hidden, torch.zeros_like(hidden)
            else:
                self.hidden = hidden, cell
        elif self.rnn_type == 'gru':
            if hidden is None:
                self.init_hidden(bs, device)
            else:
                self.hidden = hidden


def rc(x_start, pred, batch_first=True):
    if batch_first:
        x_start = x_start.unsqueeze(1)
        shapes = [(1) for s in x_start.shape]
        shapes[1] = pred.shape[1]
        x_start = x_start.repeat(shapes)
    else:
        x_start = x_start.unsqueeze(0)
        shapes = [(1) for s in x_start.shape]
        shapes[0] = pred.shape[0]
        x_start = x_start.repeat(shapes)
    return x_start + pred


def rc_recurrent(x_start, pred, batch_first=True):
    if batch_first:
        pred[:, 0] = x_start + pred[:, 0]
        for i in range(1, pred.shape[1]):
            pred[:, i] = pred[:, i - 1] + pred[:, i]
    else:
        pred[0] = x_start + pred[0]
        for i in range(1, pred.shape[0]):
            pred[i] = pred[i - 1] + pred[i]
    return pred


class ResidualBehaviorNet(BaseModel):

    def __init__(self, n_features, n_landmarks, obs_length, pred_length, dim_hidden_b=128, dim_decoder_state=128, context_length=3, decoder_arch='lstm', linear_nin_decoder=False, dim_linear_enc=128, residual=False, recurrent=False):
        super().__init__(n_features, n_landmarks, obs_length, pred_length)
        assert obs_length == pred_length, "obs_length != pred_length. Autoencoder can't be used for this case."
        n_kps = n_features * n_landmarks
        self.context_length = context_length
        self.residual = residual
        self.recurrent = recurrent
        self.dec_type = decoder_arch
        self.use_nin_dec = linear_nin_decoder
        self.dim_hidden_b = dim_hidden_b
        self.b_enc = BEncoder(n_kps, 1, self.dim_hidden_b, use_linear=False, dim_linear=1)
        self.decoder = ResidualRNNDecoder(n_in=dim_hidden_b + dim_linear_enc, n_out=n_features * n_landmarks, n_hidden=dim_decoder_state, rnn_type=self.dec_type, use_nin=self.use_nin_dec)
        self.context_encoder = nn.Linear(context_length * n_features * n_landmarks, dim_linear_enc)

    def forward(self, x, y):
        batch_size, seq_len, num_agents, num_landmarks, num_features = x.shape
        x = torch.flatten(x, start_dim=3)[:, :, 0]
        b, mu, logstd, pre = self.infer_b(x)
        xs, _ = self.generate_seq(b, x_context=x[:, :self.context_length])
        xs = xs.reshape((batch_size, self.obs_length, num_agents, num_landmarks, num_features))
        return xs, b, mu, logstd, pre

    def encode(self, x):
        batch_size, seq_len, num_agents, num_landmarks, num_features = x.shape
        x = torch.flatten(x, start_dim=3)[:, :, 0]
        b, mu, logstd, pre = self.infer_b(x)
        return b

    def infer_b(self, s):
        """
        :param s: The input sequence from which b is inferred
        :return:
        """
        bs = s.shape[0]
        self.b_enc.init_hidden(bs, device=s.device)
        outs = self.b_enc(s)
        return outs

    def decode(self, obs, b):
        batch_size = obs.shape[0]
        obs = torch.flatten(obs, start_dim=3)[:, :, 0]
        xs, _ = self.generate_seq(b, x_context=obs[:, -self.context_length:])
        xs = xs.reshape((batch_size, self.obs_length, 1, self.n_landmarks, self.n_features))[:, self.context_length:]
        return xs

    def generate_seq(self, b, x_context):
        batch_size = b.shape[0]
        xs = torch.zeros([batch_size, self.obs_length, x_context.shape[-1]], device=b.device)
        self.decoder.init_hidden(batch_size, device=b.device)
        context_length = x_context.shape[1]
        xs[:, :context_length] = x_context[:, :context_length]
        context_enc = self.context_encoder(x_context.view(batch_size, -1))
        hs = torch.cat([b, context_enc], dim=1)
        for i in range(context_length, self.obs_length):
            x = self.decoder.forward_noresidual(hs)
            xs[:, i] = x
        last_obs = x_context[:, -1]
        if self.residual and self.recurrent:
            xs[:, context_length:] = rc_recurrent(last_obs, xs[:, context_length:], batch_first=True)
        elif self.residual:
            xs[:, context_length:] = rc(last_obs, xs[:, context_length:], batch_first=True)
        return xs, b


class DecoderBehaviorNet(BaseModel):

    def __init__(self, n_features, n_landmarks, obs_length, pred_length, dim_hidden_b=1024, dim_hidden_state=128, decoder_arch='lstm', linear_nin_decoder=False):
        super().__init__(n_features, n_landmarks, obs_length, pred_length)
        assert obs_length == pred_length, "obs_length != pred_length. Autoencoder can't be used for this case."
        n_kps = n_features * n_landmarks
        self.dec_type = decoder_arch
        self.use_nin_dec = linear_nin_decoder
        self.dim_hidden_b = dim_hidden_b
        self.dim_hidden_state = dim_hidden_state
        self.decoder = ResidualRNNDecoder(n_in=self.dim_hidden_b, n_out=n_kps, n_hidden=self.dim_hidden_state, rnn_type=self.dec_type, use_nin=self.use_nin_dec)

    def forward(self, hs):
        batch_size = hs.shape[0]
        xs = torch.zeros([batch_size, self.obs_length, self.n_features * self.n_landmarks], device=hs.device)
        self.decoder.init_hidden(batch_size, device=hs.device)
        for i in range(0, self.obs_length):
            xs[:, i] = self.decoder.forward_noresidual(hs)
        xs = xs.reshape((batch_size, self.obs_length, 1, self.n_landmarks, self.n_features))
        return xs


def count_flops_attn(model, _x, y):
    """
    A counter for the `thop` package to count the operations in an
    attention operation.
    Meant to be used like:
        macs, params = thop.profile(
            model,
            inputs=(inputs, timestamps),
            custom_ops={QKVAttention: QKVAttention.count_flops},
        )
    """
    b, c, *spatial = y[0].shape
    num_spatial = int(np.prod(spatial))
    matmul_ops = 2 * b * num_spatial ** 2 * c
    model.total_ops += th.DoubleTensor([matmul_ops])


class QKVAttention(nn.Module):
    """
    A module which performs QKV attention and splits in a different order.
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.
        :param qkv: an [N x (3 * H * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.chunk(3, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum('bct,bcs->bts', (q * scale).view(bs * self.n_heads, ch, length), (k * scale).view(bs * self.n_heads, ch, length))
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum('bts,bcs->bct', weight, v.reshape(bs * self.n_heads, ch, length))
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f'unsupported dimensions: {dims}')


class AttentionPool2d(nn.Module):
    """
    Adapted from CLIP: https://github.com/openai/CLIP/blob/main/clip/model.py
    """

    def __init__(self, spacial_dim: 'int', embed_dim: 'int', num_heads_channels: 'int', output_dim: 'int'=None):
        super().__init__()
        self.positional_embedding = nn.Parameter(th.randn(embed_dim, spacial_dim ** 2 + 1) / embed_dim ** 0.5)
        self.qkv_proj = conv_nd(1, embed_dim, 3 * embed_dim, 1)
        self.c_proj = conv_nd(1, embed_dim, output_dim or embed_dim, 1)
        self.num_heads = embed_dim // num_heads_channels
        self.attention = QKVAttention(self.num_heads)

    def forward(self, x):
        b, c, *_spatial = x.shape
        x = x.reshape(b, c, -1)
        x = th.cat([x.mean(dim=-1, keepdim=True), x], dim=-1)
        x = x + self.positional_embedding[None, :, :]
        x = self.qkv_proj(x)
        x = self.attention(x)
        x = self.c_proj(x)
        return x[:, :, 0]


class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=1)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode='nearest')
        else:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        if self.use_conv:
            x = self.conv(x)
        return x


def avg_pool_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D average pooling module.
    """
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f'unsupported dimensions: {dims}')


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(dims, self.channels, self.out_channels, 3, stride=stride, padding=1)
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class CheckpointFunction(th.autograd.Function):

    @staticmethod
    def forward(ctx, run_function, length, *args):
        ctx.run_function = run_function
        ctx.input_tensors = list(args[:length])
        ctx.input_params = list(args[length:])
        with th.no_grad():
            output_tensors = ctx.run_function(*ctx.input_tensors)
        return output_tensors

    @staticmethod
    def backward(ctx, *output_grads):
        ctx.input_tensors = [x.detach().requires_grad_(True) for x in ctx.input_tensors]
        with th.enable_grad():
            shallow_copies = [x.view_as(x) for x in ctx.input_tensors]
            output_tensors = ctx.run_function(*shallow_copies)
        input_grads = th.autograd.grad(output_tensors, ctx.input_tensors + ctx.input_params, output_grads, allow_unused=True)
        del ctx.input_tensors
        del ctx.input_params
        del output_tensors
        return (None, None) + input_grads


def checkpoint(func, inputs, params, flag):
    """
    Evaluate a function without caching intermediate activations, allowing for
    reduced memory at the expense of extra compute in the backward pass.

    :param func: the function to evaluate.
    :param inputs: the argument sequence to pass to `func`.
    :param params: a sequence of parameters `func` depends on but does not
                   explicitly take as arguments.
    :param flag: if False, disable gradient checkpointing.
    """
    if flag:
        args = tuple(inputs) + tuple(params)
        return CheckpointFunction.apply(func, len(inputs), *args)
    else:
        return func(*inputs)


def linear(*args, **kwargs):
    """
    Create a linear module.
    """
    return nn.Linear(*args, **kwargs)


class GroupNorm32(nn.GroupNorm):

    def forward(self, x):
        return super().forward(x)


def normalization(channels):
    """
    Make a standard normalization layer.

    :param channels: number of input channels.
    :return: an nn.Module for normalization.
    """
    return GroupNorm32(32, channels)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(self, channels, emb_channels, dropout, out_channels=None, use_conv=False, use_scale_shift_norm=False, dims=2, use_checkpoint=False, up=False, down=False):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm
        self.in_layers = nn.Sequential(normalization(channels), nn.SiLU(), conv_nd(dims, channels, self.out_channels, 3, padding=1))
        self.updown = up or down
        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()
        self.emb_layers = nn.Sequential(nn.SiLU(), linear(emb_channels, 2 * self.out_channels if use_scale_shift_norm else self.out_channels))
        self.out_layers = nn.Sequential(normalization(self.out_channels), nn.SiLU(), nn.Dropout(p=dropout), zero_module(conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)))
        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 3, padding=1)
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.
        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return checkpoint(self._forward, (x, emb), self.parameters(), self.use_checkpoint)

    def _forward(self, x, emb):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = th.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h


class QKVAttentionLegacy(nn.Module):
    """
    A module which performs QKV attention. Matches legacy QKVAttention + input/ouput heads shaping
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.
        :param qkv: an [N x (H * 3 * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum('bct,bcs->bts', q * scale, k * scale)
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum('bts,bcs->bct', weight, v)
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.
    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(self, channels, num_heads=1, num_head_channels=-1, use_checkpoint=False, use_new_attention_order=False):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert channels % num_head_channels == 0, f'q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}'
            self.num_heads = channels // num_head_channels
        self.use_checkpoint = use_checkpoint
        self.norm = normalization(channels)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        if use_new_attention_order:
            self.attention = QKVAttention(self.num_heads)
        else:
            self.attention = QKVAttentionLegacy(self.num_heads)
        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x):
        return checkpoint(self._forward, (x,), self.parameters(), True)

    def _forward(self, x):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        h = self.attention(qkv)
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)


DTYPES = {'float16': th.float16, 'float32': th.float32, 'float64': th.float64}


def timestep_embedding(timesteps, dim, max_period=10000, dtype=th.float32):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = th.exp(-math.log(max_period) * th.arange(start=0, end=half, dtype=dtype) / half)
    args = timesteps[:, None].type(dtype) * freqs[None]
    embedding = th.cat([th.cos(args), th.sin(args)], dim=-1)
    if dim % 2:
        embedding = th.cat([embedding, th.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class GDUnet_Latent(nn.Module):
    """
    The full UNet model with attention and timestep embedding.
    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param num_heads_upsample: works with num_heads to set a different number
                               of heads for upsampling. Deprecated.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    :param use_new_attention_order: use a different attention pattern for potentially
                                    increased efficiency.
    """

    def __init__(self, in_channels=1, out_channels=1, cond_embed_dim=128, model_channels=128, num_res_blocks=2, attention_resolutions=(1, 2, 3, 4), dropout=0, channel_mult=(1, 2, 4, 8), conv_resample=True, dims=2, use_checkpoint=False, dtype='float32', num_heads=1, num_head_channels=-1, num_heads_upsample=-1, use_scale_shift_norm=False, resblock_updown=False, use_new_attention_order=False):
        super(GDUnet_Latent, self).__init__()
        if num_heads_upsample == -1:
            num_heads_upsample = num_heads
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.use_checkpoint = use_checkpoint
        self.dtype = DTYPES[dtype]
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(linear(model_channels, time_embed_dim), nn.SiLU(), linear(time_embed_dim, time_embed_dim))
        ch = input_ch = int(channel_mult[0] * model_channels)
        self.input_blocks = nn.ModuleList([TimestepEmbedSequential(conv_nd(dims, in_channels, ch, 3, padding=1))])
        self._feature_size = ch
        input_block_chans = [ch]
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [ResBlock(ch, time_embed_dim + cond_embed_dim, dropout, out_channels=int(mult * model_channels), dims=dims, use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm)]
                ch = int(mult * model_channels)
                if ds in attention_resolutions:
                    layers.append(AttentionBlock(ch, use_checkpoint=use_checkpoint, num_heads=num_heads, num_head_channels=num_head_channels, use_new_attention_order=use_new_attention_order))
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(TimestepEmbedSequential(ResBlock(ch, time_embed_dim + cond_embed_dim, dropout, out_channels=out_ch, dims=dims, use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm, down=True) if resblock_updown else Downsample(ch, conv_resample, dims=dims, out_channels=out_ch)))
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch
        self.middle_block = TimestepEmbedSequential(ResBlock(ch, time_embed_dim + cond_embed_dim, dropout, dims=dims, use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm), AttentionBlock(ch, use_checkpoint=use_checkpoint, num_heads=num_heads, num_head_channels=num_head_channels, use_new_attention_order=use_new_attention_order), ResBlock(ch, time_embed_dim + cond_embed_dim, dropout, dims=dims, use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm))
        self._feature_size += ch
        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [ResBlock(ch + ich, time_embed_dim + cond_embed_dim, dropout, out_channels=int(model_channels * mult), dims=dims, use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm)]
                ch = int(model_channels * mult)
                if ds in attention_resolutions:
                    layers.append(AttentionBlock(ch, use_checkpoint=use_checkpoint, num_heads=num_heads_upsample, num_head_channels=num_head_channels, use_new_attention_order=use_new_attention_order))
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(ResBlock(ch, time_embed_dim + cond_embed_dim, dropout, out_channels=out_ch, dims=dims, use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm, up=True) if resblock_updown else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch))
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
        self.out = nn.Sequential(normalization(ch), nn.SiLU(), zero_module(conv_nd(dims, input_ch, out_channels, 3, padding=1)))

    def forward(self, x, timesteps, cond_emb):
        """
        IMPORTANT: modified to condition on embedding (which is concatenated to the time one)
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param cond_emb: an [N x C] Tensor of embeddings to be conditioned on
        :return: an [N x C x ...] Tensor of outputs.
        """
        hs = []
        time_emb = self.time_embed(timestep_embedding(timesteps, self.model_channels, dtype=self.dtype))
        emb = torch.cat([time_emb, cond_emb], axis=1)
        h = x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)
        h = self.middle_block(h, emb)
        for module in self.output_blocks:
            h = th.cat([h, hs.pop()], dim=1)
            h = module(h, emb)
        h = h.type(x.dtype)
        return self.out(h)


class GDUnet_Latent_vSum(nn.Module):
    """
    The full UNet model with attention and timestep embedding.
    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param num_heads_upsample: works with num_heads to set a different number
                               of heads for upsampling. Deprecated.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    :param use_new_attention_order: use a different attention pattern for potentially
                                    increased efficiency.
    """

    def __init__(self, in_channels=1, out_channels=1, model_channels=128, conditioning_dim=128, num_res_blocks=2, attention_resolutions=(1, 2, 3, 4), dropout=0, channel_mult=(1, 2, 4, 8), conv_resample=True, dims=2, use_checkpoint=False, dtype='float32', num_heads=1, num_head_channels=-1, num_heads_upsample=-1, use_scale_shift_norm=False, resblock_updown=False, use_new_attention_order=False):
        super(GDUnet_Latent_vSum, self).__init__()
        if num_heads_upsample == -1:
            num_heads_upsample = num_heads
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.use_checkpoint = use_checkpoint
        self.dtype = DTYPES[dtype]
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        time_embed_dim = model_channels * 4
        cond_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(linear(model_channels, time_embed_dim), nn.SiLU(), linear(time_embed_dim, time_embed_dim))
        self.cond_embed = nn.Sequential(linear(conditioning_dim, cond_embed_dim), nn.SiLU(), linear(cond_embed_dim, cond_embed_dim))
        ch = input_ch = int(channel_mult[0] * model_channels)
        self.input_blocks = nn.ModuleList([TimestepEmbedSequential(conv_nd(dims, in_channels, ch, 3, padding=1))])
        self._feature_size = ch
        input_block_chans = [ch]
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [ResBlock(ch, time_embed_dim, dropout, out_channels=int(mult * model_channels), dims=dims, use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm)]
                ch = int(mult * model_channels)
                if ds in attention_resolutions:
                    layers.append(AttentionBlock(ch, use_checkpoint=use_checkpoint, num_heads=num_heads, num_head_channels=num_head_channels, use_new_attention_order=use_new_attention_order))
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(TimestepEmbedSequential(ResBlock(ch, time_embed_dim, dropout, out_channels=out_ch, dims=dims, use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm, down=True) if resblock_updown else Downsample(ch, conv_resample, dims=dims, out_channels=out_ch)))
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch
        self.middle_block = TimestepEmbedSequential(ResBlock(ch, time_embed_dim, dropout, dims=dims, use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm), AttentionBlock(ch, use_checkpoint=use_checkpoint, num_heads=num_heads, num_head_channels=num_head_channels, use_new_attention_order=use_new_attention_order), ResBlock(ch, time_embed_dim, dropout, dims=dims, use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm))
        self._feature_size += ch
        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [ResBlock(ch + ich, time_embed_dim, dropout, out_channels=int(model_channels * mult), dims=dims, use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm)]
                ch = int(model_channels * mult)
                if ds in attention_resolutions:
                    layers.append(AttentionBlock(ch, use_checkpoint=use_checkpoint, num_heads=num_heads_upsample, num_head_channels=num_head_channels, use_new_attention_order=use_new_attention_order))
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(ResBlock(ch, time_embed_dim, dropout, out_channels=out_ch, dims=dims, use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm, up=True) if resblock_updown else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch))
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
        self.out = nn.Sequential(normalization(ch), nn.SiLU(), zero_module(conv_nd(dims, input_ch, out_channels, 3, padding=1)))

    def forward(self, x, timesteps, cond_emb):
        """
        IMPORTANT: modified to condition on embedding (which is concatenated to the time one)
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param cond_emb: an [N x C] Tensor of embeddings to be conditioned on
        :return: an [N x C x ...] Tensor of outputs.
        """
        hs = []
        time_emb = self.time_embed(timestep_embedding(timesteps, self.model_channels, dtype=self.dtype))
        cond_emb = self.cond_embed(cond_emb)
        emb = time_emb + cond_emb
        h = x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)
        h = self.middle_block(h, emb)
        for module in self.output_blocks:
            h = th.cat([h, hs.pop()], dim=1)
            h = module(h, emb)
        h = h.type(x.dtype)
        return self.out(h)


def _get_opt_name(flags):
    for flg in flags:
        if flg.startswith('--'):
            return flg.replace('--', '')
    return flags[0].replace('--', '')


def _get_by_path(tree, keys):
    """Access a nested object in tree by sequence of keys."""
    return reduce(getitem, keys, tree)


def _set_by_path(tree, keys, value):
    """Set a value in a nested object in tree by sequence of keys."""
    keys = keys.split(';')
    _get_by_path(tree, keys[:-1])[keys[-1]] = value


def _update_config(config, modification):
    if modification is None:
        return config
    for k, v in modification.items():
        if v is not None:
            _set_by_path(config, k, v)
    return config


def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)


def setup_logging(save_dir, log_config='logger/logger_config.json', default_level=logging.INFO):
    """
    Setup logging configuration
    """
    log_config = Path(log_config)
    if log_config.is_file():
        config = read_json(log_config)
        for _, handler in config['handlers'].items():
            if 'filename' in handler:
                handler['filename'] = str(save_dir / handler['filename'])
        logging.config.dictConfig(config)
    else:
        None
        logging.basicConfig(level=default_level)


class ConfigParser:

    def __init__(self, config, resume=None, modification=None, run_id=None, save=True):
        """
        class to parse configuration json file. Handles hyperparameters for training, initializations of modules, checkpoint saving
        and logging module.
        :param config: Dict containing configurations, hyperparameters for training. contents of `config.json` file for example.
        :param resume: String, path to the checkpoint being loaded.
        :param modification: Dict keychain:value, specifying position values to be replaced from config dict.
        :param run_id: Unique Identifier for training processes. Used to save checkpoints and training log. Timestamp is being used as default
        """
        config = self.custom_modifications(config)
        self._config = _update_config(config, modification)
        self.resume = resume
        if save:
            assert 'config_path' in self.config, 'config_path not in config'
            save_dir = os.path.dirname(self.config['config_path'])
            exper_name = self.config['name']
            if run_id is None:
                run_id = datetime.now().strftime('%y%m%d_%H%M%S') + f'_{random.randint(0, 1000):03d}'
            if resume is not None and 'models' in resume:
                resumed_folder = os.path.dirname(resume)
                self._save_dir = Path(resumed_folder)
                self._log_dir = self._save_dir
            else:
                self._save_dir = Path(save_dir)
                self._log_dir = Path(save_dir)
            self.save_dir.mkdir(parents=True, exist_ok=True)
            self.log_dir.mkdir(parents=True, exist_ok=True)
            if not resume:
                config['unique_id'] = run_id
            setup_logging(self.log_dir)
        self.log_levels = {(0): logging.WARNING, (1): logging.INFO, (2): logging.DEBUG}

    def custom_modifications(self, config_dict):
        dtype = torch.float64
        if 'dtype' in config_dict:
            val = config_dict['dtype'].lower()
            assert val in ['float32', 'float64'], 'Project can only work with either float32 or float64 dtypes.'
            dtype = torch.float32 if config_dict['dtype'].lower() == 'float32' else torch.float64
        else:
            config_dict['dtype'] = 'float64'
        torch.set_default_dtype(dtype)
        for dl in ['data_loader_training', 'data_loader_validation', 'data_loader_test']:
            if dl not in config_dict:
                continue
            config_dict[dl]['args']['normalize_data'] = config_dict['normalize_data'] if 'normalize_data' in config_dict else True
            config_dict[dl]['args']['normalize_type'] = config_dict['normalize_type'] if 'normalize_type' in config_dict else 'standardize'
            config_dict[dl]['args']['precomputed_folder'] = config_dict['precomputed_folder']
            config_dict[dl]['args']['obs_length'] = config_dict['obs_length']
            config_dict[dl]['args']['pred_length'] = config_dict['pred_length']
            if 'trainer' in config_dict:
                config_dict[dl]['args']['batch_size'] = config_dict['trainer']['batch_size']
                config_dict[dl]['args']['num_workers'] = config_dict['trainer']['num_workers']
            config_dict[dl]['args']['seed'] = config_dict['seed']
            config_dict[dl]['args']['dtype'] = config_dict['dtype']
        for prefix in ['', 'aux_']:
            if prefix + 'arch' in config_dict:
                config_dict[prefix + 'arch']['args']['n_landmarks'] = config_dict['landmarks']
                config_dict[prefix + 'arch']['args']['n_features'] = config_dict['dims']
                config_dict[prefix + 'arch']['args']['obs_length'] = config_dict['obs_length']
                config_dict[prefix + 'arch']['args']['pred_length'] = config_dict['pred_length']
            if prefix + 'loss' in config_dict:
                config_dict[prefix + 'loss']['args']['n_dims'] = config_dict['eval_dims']
            if prefix + 'metrics' in config_dict:
                for met in config_dict[prefix + 'metrics']:
                    if 'args' not in met:
                        met['args'] = {}
                    met['args']['n_dims'] = config_dict['eval_dims']
        for key, suffix in zip(['generator', 'discriminator'], ['_G', '_D']):
            if key in config_dict:
                config_dict[key]['args']['n_landmarks'] = config_dict['landmarks']
                config_dict[key]['args']['n_features'] = config_dict['dims']
                config_dict[key]['args']['obs_length'] = config_dict['obs_length']
                config_dict[key]['args']['pred_length'] = config_dict['pred_length']
            loss_key = 'loss' + suffix
            if loss_key in config_dict:
                config_dict[loss_key]['args']['n_dims'] = config_dict['eval_dims']
            met_key = 'metrics' + suffix
            if met_key in config_dict:
                for met in config_dict[met_key]:
                    if 'args' not in met:
                        met['args'] = {}
                    met['args']['n_dims'] = config_dict['eval_dims']
        return config_dict

    @classmethod
    def from_args(cls, args, options='', save=True):
        """
        Initialize this class from some cli arguments. Used in train, test.
        """
        for opt in options:
            args.add_argument(*opt.flags, default=None, type=opt.type)
        if not isinstance(args, tuple):
            args = args.parse_args()
        if args.device is not None:
            os.environ['CUDA_VISIBLE_DEVICES'] = args.device
        if args.resume is not None and args.config is None:
            resume = Path(args.resume)
            cfg_fname = resume.parent / 'config.json'
        else:
            msg_no_cfg = "Configuration file need to be specified. Add '-c config.json', for example."
            assert args.config is not None, msg_no_cfg
            resume = args.resume if args.resume is not None else None
            cfg_fname = Path(args.config)
        config = read_json(cfg_fname)
        if args.config and resume:
            config.update(read_json(args.config))
        modification = {opt.target: getattr(args, _get_opt_name(opt.flags)) for opt in options}
        return cls(config, resume, modification, save=save)

    def init_obj(self, name, module, *args, **kwargs):
        """
        Finds a function handle with the name given as 'type' in config, and returns the
        instance initialized with corresponding arguments given.

        `object = config.init_obj('name', module, a, b=1)`
        is equivalent to
        `object = module.name(a, b=1)`
        """
        module_name = self[name]['type']
        module_args = dict(self[name]['args'])
        assert all([(k not in module_args) for k in kwargs]), 'Overwriting kwargs given in config file is not allowed'
        module_args.update(kwargs)
        return getattr(module, module_name)(*args, **module_args)

    def init_ftn(self, name, module, *args, **kwargs):
        """
        Finds a function handle with the name given as 'type' in config, and returns the
        function with given arguments fixed with functools.partial.

        `function = config.init_ftn('name', module, a, b=1)`
        is equivalent to
        `function = lambda *args, **kwargs: module.name(a, *args, b=1, **kwargs)`.
        """
        module_name = self[name]['type']
        module_args = dict(self[name]['args'])
        assert all([(k not in module_args) for k in kwargs]), 'Overwriting kwargs given in config file is not allowed'
        module_args.update(kwargs)
        return partial(getattr(module, module_name), *args, **module_args)

    def __getitem__(self, name):
        """Access items like ordinary dict."""
        return self.config[name]

    def get_logger(self, name, verbosity=2):
        msg_verbosity = 'verbosity option {} is invalid. Valid options are {}.'.format(verbosity, self.log_levels.keys())
        assert verbosity in self.log_levels, msg_verbosity
        logger = logging.getLogger(name)
        logger.setLevel(self.log_levels[verbosity])
        return logger

    @property
    def config(self):
        return self._config

    @property
    def save_dir(self):
        return self._save_dir

    @property
    def log_dir(self):
        return self._log_dir


class BaseLatentModel(BaseModel):

    def __init__(self, n_features, n_landmarks, obs_length, pred_length, embedder_obs_path, embedder_pred_path, emb_size=None, emb_preprocessing='none', freeze_obs_encoder=True, freeze_pred_decoder=True, load_obs_encoder=True):
        super(BaseLatentModel, self).__init__(n_features, n_landmarks, obs_length, pred_length)
        self.emb_preprocessing = emb_preprocessing
        assert emb_size is not None, 'emb_size must be specified.'
        self.emb_size = emb_size
        def_dtype = torch.get_default_dtype()
        self.embedder_obs_path = embedder_obs_path
        self.embedder_pred_path = embedder_pred_path
        models = []
        models_stats = []
        for path, load_checkpoint in zip((embedder_obs_path, embedder_pred_path), (True, load_obs_encoder)):
            configpath = os.path.join(os.path.dirname(path), 'config.json')
            assert os.path.exists(path) and os.path.exists(configpath), f"Missing checkpoint/config file for auxiliary model: '{path}'"
            config = read_json(configpath)
            config = ConfigParser(config, save=False)
            model = config.init_obj('arch', module_arch)
            if load_checkpoint:
                checkpoint = torch.load(path, map_location='cpu')
                assert 'statistics' in checkpoint or emb_preprocessing.lower() == 'none', "Model statistics are not available in its checkpoint. Can't apply embeddings preprocessing."
                stats = checkpoint['statistics'] if 'statistics' in checkpoint else None
                models_stats.append(stats)
                state_dict = checkpoint['state_dict']
                model.load_state_dict(state_dict)
            else:
                models_stats.append(None)
            models.append(model)
        self.embed_obs, self.embed_obs_stats = models[0], models_stats[0]
        self.embed_pred, self.embed_pred_stats = models[1], models_stats[1]
        if freeze_obs_encoder:
            for para in self.embed_obs.parameters():
                para.requires_grad = False
        if freeze_pred_decoder:
            for para in self.embed_pred.parameters():
                para.requires_grad = False
        torch.set_default_dtype(def_dtype)
        self.init_params = None

    def deepcopy(self):
        assert self.init_params is not None, 'Cannot deepcopy LatentUNetMatcher if init_params is None.'
        model_copy = self.__class__(**self.init_params)
        weights_path = f'weights_temp_{id(model_copy)}.pt'
        torch.save(self.state_dict(), weights_path)
        model_copy.load_state_dict(torch.load(weights_path))
        os.remove(weights_path)
        return model_copy

    def to(self, device):
        self.embed_obs
        self.embed_pred
        if self.embed_obs_stats is not None:
            for key in self.embed_obs_stats:
                self.embed_obs_stats[key] = self.embed_obs_stats[key]
        if self.embed_pred_stats is not None:
            for key in self.embed_pred_stats:
                self.embed_pred_stats[key] = self.embed_pred_stats[key]
        super()
        return self

    def preprocess(self, emb, stats, is_prediction=False):
        if stats is None:
            return emb
        if 'standardize' in self.emb_preprocessing:
            return (emb - stats['mean']) / torch.sqrt(stats['var'])
        elif 'normalize' in self.emb_preprocessing:
            return 2 * (emb - stats['min']) / (stats['max'] - stats['min']) - 1
        elif 'none' in self.emb_preprocessing.lower():
            return emb
        else:
            raise NotImplementedError(f"Error on the embedding preprocessing value: '{self.emb_preprocessing}'")

    def undo_preprocess(self, emb, stats, is_prediction=False):
        if stats is None:
            return emb
        if 'standardize' in self.emb_preprocessing:
            return torch.sqrt(stats['var']) * emb + stats['mean']
        elif 'normalize' in self.emb_preprocessing:
            return (emb + 1) * (stats['max'] - stats['min']) / 2 + stats['min']
        elif 'none' in self.emb_preprocessing.lower():
            return emb
        else:
            raise NotImplementedError(f"Error on the embedding preprocessing value: '{self.emb_preprocessing}'")

    def encode_obs(self, obs):
        return self.preprocess(self.embed_obs.encode(obs), self.embed_obs_stats, is_prediction=False)

    def encode_pred(self, pred, obs):
        pred = torch.cat((obs[:, -self.embed_pred.context_length:], pred), dim=1)
        return self.preprocess(self.embed_pred.encode(pred), self.embed_pred_stats, is_prediction=True)

    def decode_pred(self, obs, pred_emb):
        return self.embed_pred.decode(obs, self.undo_preprocess(pred_emb, self.embed_pred_stats, is_prediction=True))

    def get_emb_size(self):
        return self.emb_size

    def forward(self, pred, timesteps, obs):
        raise NotImplementedError('This is an abstract class.')


def compute_idces_order(ref, height, width):
    ref = ref.cpu().numpy().reshape((height, width))
    idces = np.array(range(ref.reshape(-1).shape[0])).reshape([height, width])
    for j in range(ref.shape[1]):
        order = np.argsort(ref[:, j])
        idces[:, j] = idces[order, j]
    for i in range(ref.shape[0]):
        order = np.argsort(ref[i])
        idces[i] = idces[i, order]
    idces_inv = np.array(range(ref.reshape(-1).shape[0]))
    idces_inv = idces_inv[np.argsort(idces.reshape(-1))]
    return idces.reshape(-1), idces_inv


class LatentUNetMatcher(BaseLatentModel):

    def __init__(self, n_features, n_landmarks, obs_length, pred_length, embedder_obs_path, embedder_pred_path, freeze_obs_encoder=True, freeze_pred_decoder=True, load_obs_encoder=True, emb_preprocessing='none', emb_height=None, emb_width=None, cond_embed_dim=128, model_channels=128, num_res_blocks=2, attention_resolutions=(1, 2, 3, 4), dropout=0, channel_mult=(1, 2, 4, 8), conv_resample=True, dims=2, use_checkpoint=False, dtype='float32', num_heads=1, num_head_channels=-1, num_heads_upsample=-1, use_scale_shift_norm=False, resblock_updown=False, use_new_attention_order=False):
        super(LatentUNetMatcher, self).__init__(n_features, n_landmarks, obs_length, pred_length, embedder_obs_path, embedder_pred_path, emb_size=emb_height * emb_width, emb_preprocessing=emb_preprocessing, freeze_obs_encoder=freeze_obs_encoder, freeze_pred_decoder=freeze_pred_decoder, load_obs_encoder=load_obs_encoder)
        assert emb_height is not None and emb_width is not None, 'Embedding height and width must be specified.'
        self.emb_height = emb_height
        self.emb_width = emb_width
        self.unet = GDUnet_Latent(in_channels=1, out_channels=1, cond_embed_dim=cond_embed_dim, model_channels=model_channels, num_res_blocks=num_res_blocks, attention_resolutions=attention_resolutions, dropout=dropout, channel_mult=channel_mult, conv_resample=conv_resample, dims=dims, use_checkpoint=use_checkpoint, num_heads=num_heads, num_head_channels=num_head_channels, num_heads_upsample=num_heads_upsample, use_scale_shift_norm=use_scale_shift_norm, resblock_updown=resblock_updown, use_new_attention_order=use_new_attention_order, dtype=dtype)
        self.init_params = {'n_features': n_features, 'n_landmarks': n_landmarks, 'obs_length': obs_length, 'pred_length': pred_length, 'embedder_obs_path': embedder_obs_path, 'embedder_pred_path': embedder_pred_path, 'freeze_obs_encoder': freeze_obs_encoder, 'freeze_pred_decoder': freeze_pred_decoder, 'load_obs_encoder': load_obs_encoder, 'emb_preprocessing': emb_preprocessing, 'emb_height': emb_height, 'emb_width': emb_width, 'cond_embed_dim': cond_embed_dim, 'model_channels': model_channels, 'num_res_blocks': num_res_blocks, 'attention_resolutions': attention_resolutions, 'dropout': dropout, 'channel_mult': channel_mult, 'conv_resample': conv_resample, 'dims': dims, 'use_checkpoint': use_checkpoint, 'dtype': dtype, 'num_heads': num_heads, 'num_head_channels': num_head_channels, 'num_heads_upsample': num_heads_upsample, 'use_scale_shift_norm': use_scale_shift_norm, 'resblock_updown': resblock_updown, 'use_new_attention_order': use_new_attention_order}

    def preprocess(self, emb, stats, is_prediction=False):
        emb = super().preprocess(emb, stats)
        if is_prediction and 'meanstd' in self.emb_preprocessing:
            if 'order_meanstd' not in stats or 'order_inv_meanstd' not in stats:
                order, order_inv = compute_idces_order(stats['mean'] * stats['std'], self.emb_height, self.emb_width)
                stats['order_meanstd'] = order
                stats['order_inv_meanstd'] = order_inv
            bs = emb.shape[0]
            return torch.gather(emb, 1, torch.repeat_interleave(torch.tensor(stats['order_meanstd'], device=emb.device).unsqueeze(0), bs, axis=0))
        else:
            return emb

    def undo_preprocess(self, emb, stats, is_prediction=False):
        emb = super().undo_preprocess(emb, stats)
        if is_prediction and 'meanstd' in self.emb_preprocessing:
            if 'order_meanstd' not in stats or 'order_inv_meanstd' not in stats:
                order, order_inv = compute_idces_order(stats['mean'] * stats['std'], self.emb_height, self.emb_width)
                stats['order_meanstd'] = order
                stats['order_inv_meanstd'] = order_inv
            bs = emb.shape[0]
            return torch.gather(emb, 1, torch.repeat_interleave(torch.tensor(stats['order_inv_meanstd'], device=emb.device).unsqueeze(0), bs, axis=0))
        else:
            return emb

    def forward(self, pred, timesteps, obs):
        pred_emb_reshaped = pred.reshape((-1, 1, self.emb_height, self.emb_width))
        out = self.unet(pred_emb_reshaped, timesteps, obs)
        out = out.reshape((out.shape[0], -1))
        return out


class SiLU(nn.Module):

    def forward(self, x):
        return x * th.sigmoid(x)


class ClassifierForFID(nn.Module):

    def __init__(self, input_size=48, hidden_size=128, hidden_layer=2, output_size=15, device='', use_noise=None):
        super(ClassifierForFID, self).__init__()
        self.device = device
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.hidden_layer = hidden_layer
        self.use_noise = use_noise
        self.recurrent = nn.GRU(input_size, hidden_size, hidden_layer)
        self.linear1 = nn.Linear(hidden_size, 30)
        self.linear2 = nn.Linear(30, output_size)

    def forward(self, motion_sequence, hidden_unit=None):
        """
        motion_sequence: b, 48, 100
        hidden_unit:
        """
        motion_sequence = motion_sequence.permute(2, 0, 1).contiguous()
        if hidden_unit is None:
            hidden_unit = self.initHidden(motion_sequence.size(1), self.hidden_layer)
        gru_o, _ = self.recurrent(motion_sequence.float(), hidden_unit)
        lin1 = self.linear1(gru_o[-1, :, :])
        lin1 = torch.tanh(lin1)
        lin2 = self.linear2(lin1)
        return lin2

    def get_fid_features(self, motion_sequence, hidden_unit=None):
        """
        motion_sequence: b, 48, 100
        hidden_unit:
        """
        motion_sequence = motion_sequence.permute(2, 0, 1).contiguous()
        if hidden_unit is None:
            hidden_unit = self.initHidden(motion_sequence.size(1), self.hidden_layer)
        gru_o, _ = self.recurrent(motion_sequence.float(), hidden_unit)
        lin1 = self.linear1(gru_o[-1, :, :])
        lin1 = torch.tanh(lin1)
        return lin1

    def initHidden(self, num_samples, layer):
        return torch.randn(layer, num_samples, self.hidden_size, device=self.device, requires_grad=False)


class Seq2Seq_Auto(BaseAutoencoder):

    def __init__(self, n_features, n_landmarks, obs_length, pred_length, nz=None, nh_mlp=[300, 200], nh_rnn=128, use_drnn_mlp=False, rnn_type='lstm', encoding_length=None, dropout=0, noise_augmentation=0, recurrent=True, residual=True):
        super(Seq2Seq_Auto, self).__init__(n_features, n_landmarks, obs_length, pred_length)
        assert obs_length == pred_length, 'For autoencoders, obs_length and pred_length must be equal'
        self.encoding_length = encoding_length
        self.nx = n_features * n_landmarks
        self.ny = self.nx
        self.seq_length = pred_length - 1
        self.rnn_type = rnn_type
        self.use_drnn_mlp = use_drnn_mlp
        self.nh_rnn = nh_rnn
        self.nh_mlp = nh_mlp
        self.recurrent = recurrent
        self.residual = residual
        self.x_rnn = RNN(self.nx, nh_rnn, cell_type=rnn_type)
        if self.use_drnn_mlp:
            self.drnn_mlp = MLP(nh_rnn, nh_mlp + [nh_rnn], activation='tanh')
        self.d_rnn = RNN(self.ny + nh_rnn, nh_rnn, cell_type=rnn_type)
        self.d_mlp = MLP(nh_rnn, nh_mlp)
        self.d_out = nn.Linear(self.d_mlp.out_dim, self.ny)
        self.d_rnn.set_mode('step')
        self.dropout = nn.Dropout(dropout)
        self.noise_augmentation = noise_augmentation

    def _encode(self, x):
        assert x.shape[0] == self.seq_length
        return self.x_rnn(x)[-1]

    def encode(self, x):
        tf = 'b s p l f  -> s (b p) (l f)'
        x = rearrange(x, tf)
        return self.x_rnn(x)[-1]

    def _decode(self, x_start, h_y):
        h_y = self.dropout(h_y)
        if self.noise_augmentation != 0:
            h_y = h_y + torch.randn_like(h_y) * self.noise_augmentation
        if self.use_drnn_mlp:
            h_d = self.drnn_mlp(h_y)
            self.d_rnn.initialize(batch_size=x_start.shape[0], hx=h_d)
        else:
            self.d_rnn.initialize(batch_size=x_start.shape[0])
        y = []
        for i in range(self.seq_length):
            y_p = x_start if i == 0 else y_i
            rnn_in = torch.cat([h_y, y_p], dim=1)
            h = self.d_rnn(rnn_in)
            h = self.d_mlp(h)
            y_i = self.d_out(h)
            y.append(y_i)
        y = torch.stack(y)
        if self.residual and self.recurrent:
            y = rc_recurrent(x_start, y, batch_first=False)
        elif self.residual:
            y = rc(x_start, y, batch_first=False)
        return y

    def decode(self, obs, h_y):
        x_start = obs[:, -1]
        n_agents, n_landmarks, n_features = x_start.shape[-3:]
        tf = 'b p l f  -> (b p) (l f)'
        x_start = rearrange(x_start, tf)
        y = self._decode(x_start, h_y)
        y = rearrange(y, 's (b p) (n f) -> b s p n f', n=n_landmarks, f=n_features, p=n_agents)
        return y

    def autoencode(self, x):
        x_start = x[0]
        h_x = self._encode(x[1:])
        return torch.cat([x[0][None, ...], self._decode(x_start, h_x)], axis=0)

    def forward(self, x, y):
        n_agents, n_landmarks, n_features = x.shape[-3:]
        tf = 'b s p l f  -> s (b p) (l f)'
        x = rearrange(x, tf)
        Y_r = self.autoencode(x)
        Y_r = rearrange(Y_r, 's (b p) (n f) -> b s p n f', n=n_landmarks, f=n_features, p=n_agents)
        return Y_r


class GraphConv(nn.Module):
    """
        adapted from : https://github.com/tkipf/gcn/blob/92600c39797c2bfb61a508e52b88fb554df30177/gcn/layers.py#L132
        """

    def __init__(self, in_len, out_len, in_node_n=66, out_node_n=66, bias=True):
        super(GraphConv, self).__init__()
        self.in_len = in_len
        self.out_len = out_len
        self.in_node_n = in_node_n
        self.out_node_n = out_node_n
        self.weight = Parameter(torch.FloatTensor(in_len, out_len))
        self.att = Parameter(torch.FloatTensor(in_node_n, out_node_n))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_len))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.att.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        """
        b, cv, t
        """
        if input.device != self.weight.device:
            self.weight = self.weight
            self.att = self.att
            if self.bias is not None:
                self.bias = self.bias
        features = torch.matmul(input, self.weight)
        output = torch.matmul(features.permute(0, 2, 1).contiguous(), self.att).permute(0, 2, 1).contiguous()
        if self.bias is not None:
            output = output + self.bias
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_len) + ' -> ' + str(self.out_len) + ')' + ' (' + str(self.in_node_n) + ' -> ' + str(self.out_node_n) + ')'


class GraphConvBlock(nn.Module):

    def __init__(self, in_len, out_len, in_node_n, out_node_n, dropout_rate=0, leaky=0.1, bias=True, residual=False):
        super(GraphConvBlock, self).__init__()
        self.dropout_rate = dropout_rate
        self.resual = residual
        self.out_len = out_len
        self.gcn = GraphConv(in_len, out_len, in_node_n=in_node_n, out_node_n=out_node_n, bias=bias)
        self.bn = nn.BatchNorm1d(out_node_n * out_len)
        self.act = nn.Tanh()
        if self.dropout_rate > 0:
            self.drop = nn.Dropout(dropout_rate)

    def forward(self, input):
        """
        Args:
            input: b, cv, t
        Returns:
        """
        x = self.gcn(input)
        b, vc, t = x.shape
        x = self.bn(x.view(b, -1)).view(b, vc, t)
        x = self.act(x)
        if self.dropout_rate > 0:
            x = self.drop(x)
        if self.resual:
            return x + input
        else:
            return x


class ResGCB(nn.Module):

    def __init__(self, in_len, out_len, in_node_n, out_node_n, dropout_rate=0, leaky=0.1, bias=True, residual=False):
        super(ResGCB, self).__init__()
        self.resual = residual
        self.gcb1 = GraphConvBlock(in_len, in_len, in_node_n=in_node_n, out_node_n=in_node_n, dropout_rate=dropout_rate, bias=bias, residual=False)
        self.gcb2 = GraphConvBlock(in_len, out_len, in_node_n=in_node_n, out_node_n=out_node_n, dropout_rate=dropout_rate, bias=bias, residual=False)

    def forward(self, input):
        """
        Args:
            x: B,CV,T
        Returns:
        """
        x = self.gcb1(input)
        x = self.gcb2(x)
        if self.resual:
            return x + input
        else:
            return x


class CVAE(Module):

    def __init__(self, node_n=16, hidden_dim=256, z_dim=64, dct_n=10, dropout_rate=0):
        super(CVAE, self).__init__()
        self.node_n = node_n
        self.dct_n = dct_n
        self.z_dim = z_dim
        self.enc = Sequential(GraphConvBlock(in_len=3 * dct_n * 2, out_len=hidden_dim, in_node_n=node_n, out_node_n=node_n, dropout_rate=dropout_rate, bias=True, residual=False), ResGCB(in_len=hidden_dim, out_len=hidden_dim, in_node_n=node_n, out_node_n=node_n, dropout_rate=dropout_rate, bias=True, residual=True), ResGCB(in_len=hidden_dim, out_len=hidden_dim, in_node_n=node_n, out_node_n=node_n, dropout_rate=dropout_rate, bias=True, residual=True), ResGCB(in_len=hidden_dim, out_len=hidden_dim, in_node_n=node_n, out_node_n=node_n, dropout_rate=dropout_rate, bias=True, residual=True), ResGCB(in_len=hidden_dim, out_len=hidden_dim, in_node_n=node_n, out_node_n=node_n, dropout_rate=dropout_rate, bias=True, residual=True))
        self.mean = Linear(hidden_dim * node_n, z_dim)
        self.logvar = Linear(hidden_dim * node_n, z_dim)
        self.dec = Sequential(GraphConvBlock(in_len=3 * dct_n + z_dim, out_len=hidden_dim, in_node_n=node_n, out_node_n=node_n, dropout_rate=dropout_rate, bias=True, residual=False), ResGCB(in_len=hidden_dim, out_len=hidden_dim, in_node_n=node_n, out_node_n=node_n, dropout_rate=dropout_rate, bias=True, residual=True), ResGCB(in_len=hidden_dim, out_len=hidden_dim, in_node_n=node_n, out_node_n=node_n, dropout_rate=dropout_rate, bias=True, residual=True), ResGCB(in_len=hidden_dim, out_len=hidden_dim, in_node_n=node_n, out_node_n=node_n, dropout_rate=dropout_rate, bias=True, residual=True), ResGCB(in_len=hidden_dim, out_len=hidden_dim, in_node_n=node_n, out_node_n=node_n, dropout_rate=dropout_rate, bias=True, residual=True), GraphConv(in_len=hidden_dim, out_len=3 * dct_n, in_node_n=node_n, out_node_n=node_n, bias=True))

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)

    def _sample(self, mean, logvar):
        """Returns a sample from a Gaussian with diagonal covariance."""
        return torch.exp(0.5 * logvar) * torch.randn_like(logvar) + mean

    def forward(self, condition, data):
        """
        Args:
            condition: [b, 48, 25] / [b, 16, 30]
            data: [b, 48, 100] / [b, 16, 30]
        Returns:
        """
        b, v, ct = condition.shape
        feature = self.enc(torch.cat((condition, data), dim=-1))
        mean = self.mean(feature.view(b, -1))
        logvar = self.logvar(feature.view(b, -1))
        z = self._sample(mean, logvar)
        out = self.dec(torch.cat((condition, z.unsqueeze(dim=1).repeat([1, self.node_n, 1])), dim=-1))
        out = out + condition
        return out, mean, logvar

    def inference(self, condition, z):
        """
        Args:
            condition: [b, 48, 25] / [b, 16, 30]
            z: b, 64
        Returns:
        """
        out = self.dec(torch.cat((condition, z.unsqueeze(dim=1).repeat([1, self.node_n, 1])), dim=-1))
        out = out + condition
        return out


class DiverseSampling(Module):

    def __init__(self, node_n=16, hidden_dim=256, base_dim=64, z_dim=64, dct_n=10, base_num_p1=10, dropout_rate=0):
        super(DiverseSampling, self).__init__()
        self.z_dim = z_dim
        self.base_dim = base_dim
        self.base_num_p1 = base_num_p1
        self.condition_enc = Sequential(GraphConvBlock(in_len=3 * dct_n, out_len=hidden_dim, in_node_n=node_n, out_node_n=node_n, dropout_rate=dropout_rate, bias=True, residual=False), ResGCB(in_len=hidden_dim, out_len=hidden_dim, in_node_n=node_n, out_node_n=node_n, dropout_rate=dropout_rate, bias=True, residual=True), ResGCB(in_len=hidden_dim, out_len=hidden_dim, in_node_n=node_n, out_node_n=node_n, dropout_rate=dropout_rate, bias=True, residual=True))
        self.bases_p1 = Sequential(Linear(node_n * hidden_dim, self.base_num_p1 * self.base_dim), BatchNorm1d(self.base_num_p1 * self.base_dim), Tanh())
        self.mean_p1 = Sequential(Linear(self.base_dim, 64), BatchNorm1d(64), Tanh(), Linear(64, self.z_dim))
        self.logvar_p1 = Sequential(Linear(self.base_dim, 64), BatchNorm1d(64), Tanh(), Linear(64, self.z_dim))

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)

    def forward(self, condition, repeated_eps=None, many_weights=None, multi_modal_head=10):
        """
        Args:
            condition: [b, 48, 25] / [b, 16, 3*10]
            repeated_eps: b*50, 64
        Returns:
        """
        b, v, ct = condition.shape
        condition_enced = self.condition_enc(condition)
        bases = self.bases_p1(condition_enced.view(b, -1)).view(b, self.base_num_p1, self.base_dim)
        repeat_many_bases = torch.repeat_interleave(bases, repeats=multi_modal_head, dim=0)
        many_bases_blending = torch.matmul(many_weights, repeat_many_bases).squeeze(dim=1).view(-1, self.base_dim)
        all_mean = self.mean_p1(many_bases_blending)
        all_logvar = self.logvar_p1(many_bases_blending)
        all_z = torch.exp(0.5 * all_logvar) * repeated_eps + all_mean
        return all_z, all_mean, all_logvar


def _sample_weight_gumbel_softmax(logits, temperature=1, eps=1e-20):
    assert temperature > 0, 'temperature must be greater than 0 !'
    U = torch.rand(logits.shape, device=logits.device)
    g = -torch.log(-torch.log(U + eps) + eps)
    y = logits + g
    y = y / temperature
    y = torch.softmax(y, dim=-1)
    return y


def dct_transform_torch(data, dct_m, dct_n):
    """
    B, 60, 35
    """
    batch_size, features, seq_len = data.shape
    data = data.contiguous().view(-1, seq_len)
    data = data.permute(1, 0)
    out_data = torch.matmul(dct_m[:dct_n, :], data)
    out_data = out_data.permute(1, 0).contiguous().view(-1, features, dct_n)
    return out_data


torch_to_numpy_dtype_dict = {torch.float32: np.float32, torch.float64: np.float64}


def get_dct_matrix(N):
    dtype = torch_to_numpy_dtype_dict[torch.get_default_dtype()]
    dct_m = np.eye(N, dtype=dtype)
    for k in np.arange(N):
        for i in np.arange(N):
            w = np.sqrt(2 / N)
            if k == 0:
                w = np.sqrt(1 / N)
            dct_m[k, i] = w * np.cos(np.pi * (i + 1 / 2) * k / N)
    idct_m = np.linalg.inv(dct_m)
    return dct_m, idct_m


def reverse_dct_torch(dct_data, idct_m, seq_len):
    """
    B, 60, 35
    """
    batch_size, features, dct_n = dct_data.shape
    dct_data = dct_data.permute(2, 0, 1).contiguous().view(dct_n, -1)
    out_data = torch.matmul(idct_m[:, :dct_n], dct_data).contiguous().view(seq_len, batch_size, -1).permute(1, 2, 0)
    return out_data


class DiverseSamplingWrapper(Module):

    def __init__(self, n_features, n_landmarks, obs_length, pred_length, model_path_t1, model_path_t2, node_n=16, hidden_dim=256, base_dim=128, z_dim=64, dct_n=10, base_num_p1=40, temperature_p1=0.85, dropout_rate=0, only_vae=False):
        super(DiverseSamplingWrapper, self).__init__()
        self.model_t1 = CVAE(node_n=node_n, hidden_dim=hidden_dim, z_dim=z_dim, dct_n=dct_n, dropout_rate=dropout_rate)
        self.model = DiverseSampling(node_n=node_n, hidden_dim=hidden_dim, base_dim=base_dim, z_dim=z_dim, dct_n=dct_n, base_num_p1=base_num_p1, dropout_rate=dropout_rate)
        self.only_vae = only_vae
        assert n_landmarks == node_n, 'n_landmarks should be equal to node_n'
        self.z_dim = z_dim
        self.n_features = n_features
        self.n_landmarks = n_landmarks
        self.obs_length = obs_length
        self.pred_length = pred_length
        self.t_length = obs_length + pred_length
        self.dct_n = dct_n
        self.dct_m, self.i_dct_m = get_dct_matrix(self.t_length)
        self.dct_m = torch.from_numpy(self.dct_m)
        self.i_dct_m = torch.from_numpy(self.i_dct_m)
        self.temperature_p1 = temperature_p1
        self.base_num_p1 = base_num_p1
        self.load_model(model_path_t1, model_path_t2)

    def load_model(self, model_path_t1, model_path_t2):
        self.model_t1.load_state_dict(torch.load(model_path_t1)['model'])
        if not self.only_vae:
            self.model.load_state_dict(torch.load(model_path_t2)['model'])

    def to(self, device):
        self.model = self.model
        self.model_t1 = self.model_t1
        self.dct_m = self.dct_m
        self.i_dct_m = self.i_dct_m
        return self

    def eval(self):
        self.model.eval()
        self.model_t1.eval()

    def forward(self, x, y, nk):
        """
        Args:
            x: batch_size, obs_length, participants, n_landmarks, n_features
            y: batch_size, pred_length, participants, n_landmarks, n_features
        Returns:
        """
        batch_size = x.shape[0]
        participants = x.shape[2]
        device = x.device
        x = x.reshape((batch_size, self.obs_length, -1)).permute(0, 2, 1)
        with torch.no_grad():
            padded_inputs = x[:, :, list(range(self.obs_length)) + [self.obs_length - 1] * self.pred_length]
            padded_inputs_dct = dct_transform_torch(padded_inputs, self.dct_m, dct_n=self.dct_n)
            padded_inputs_dct = padded_inputs_dct.view(batch_size, -1, 3 * self.dct_n)
        logtics = torch.ones((batch_size * nk, 1, self.base_num_p1), device=device) / self.base_num_p1
        many_weights = _sample_weight_gumbel_softmax(logtics, temperature=self.temperature_p1)
        if not self.only_vae:
            eps = torch.randn((batch_size, self.z_dim), device=device)
            repeated_eps = torch.repeat_interleave(eps, repeats=nk, dim=0)
            all_z, all_mean, all_logvar = self.model(padded_inputs_dct, repeated_eps, many_weights, nk)
        else:
            all_z = torch.randn(batch_size * nk, self.z_dim, device=device)
        all_outs_dct = self.model_t1.inference(condition=torch.repeat_interleave(padded_inputs_dct, repeats=nk, dim=0), z=all_z)
        all_outs_dct = all_outs_dct.reshape(batch_size * nk, -1, self.dct_n)
        outputs = reverse_dct_torch(all_outs_dct, self.i_dct_m, self.t_length)
        outputs = outputs.view(batch_size, nk, -1, self.t_length)
        outputs = outputs.permute(0, 1, 3, 2).reshape(batch_size, nk, self.t_length, participants, self.n_landmarks, self.n_features)
        return outputs[:, :, -self.pred_length:]


class VAE(BaseModel):

    def __init__(self, n_features, n_landmarks, nz, obs_length, pred_length, nh_mlp=[300, 200], nh_rnn=128, use_drnn_mlp=False, rnn_type='lstm'):
        super(VAE, self).__init__(n_features, n_landmarks, obs_length, pred_length)
        self.nx = n_features * n_landmarks
        self.ny = self.nx
        self.nz = nz
        self.horizon = pred_length
        self.rnn_type = rnn_type
        self.use_drnn_mlp = use_drnn_mlp
        self.nh_rnn = nh_rnn
        self.nh_mlp = nh_mlp
        self.x_rnn = RNN(self.nx, nh_rnn, cell_type=rnn_type)
        self.e_rnn = RNN(self.ny, nh_rnn, cell_type=rnn_type)
        self.e_mlp = MLP(2 * nh_rnn, nh_mlp)
        self.e_mu = nn.Linear(self.e_mlp.out_dim, nz)
        self.e_logvar = nn.Linear(self.e_mlp.out_dim, nz)
        if self.use_drnn_mlp:
            self.drnn_mlp = MLP(nh_rnn, nh_mlp + [nh_rnn], activation='tanh')
        self.d_rnn = RNN(self.ny + nz + nh_rnn, nh_rnn, cell_type=rnn_type)
        self.d_mlp = MLP(nh_rnn, nh_mlp)
        self.d_out = nn.Linear(self.d_mlp.out_dim, self.ny)
        self.d_rnn.set_mode('step')

    def encode_x(self, x):
        h_x = self.x_rnn(x)[-1]
        return h_x

    def encode_y(self, y):
        return self.e_rnn(y)[-1]

    def encode(self, x, y):
        h_x = self.encode_x(x)
        h_y = self.encode_y(y)
        h = torch.cat((h_x, h_y), dim=1)
        h = self.e_mlp(h)
        return self.e_mu(h), self.e_logvar(h)

    def decode(self, x, z):
        h_x = self.encode_x(x)
        if self.use_drnn_mlp:
            h_d = self.drnn_mlp(h_x)
            self.d_rnn.initialize(batch_size=z.shape[0], hx=h_d)
        else:
            self.d_rnn.initialize(batch_size=z.shape[0])
        y = []
        for i in range(self.horizon):
            y_p = x[-1] if i == 0 else y_i
            rnn_in = torch.cat([h_x, z, y_p], dim=1)
            h = self.d_rnn(rnn_in)
            h = self.d_mlp(h)
            y_i = self.d_out(h)
            y.append(y_i)
        y = torch.stack(y)
        return y


class DLow(BaseModel):

    def __init__(self, n_features, n_landmarks, obs_length, pred_length, nk, nz=128, nh_mlp=[300, 200], nh_rnn=128, rnn_type='gru'):
        super(DLow, self).__init__(n_features, n_landmarks, obs_length, pred_length)
        self.nx = n_features * n_landmarks
        self.ny = nz
        self.nk = nk
        self.nh = nh_mlp
        self.nh_rnn = nh_rnn
        self.rnn_type = rnn_type
        self.nac = nac = nk
        self.x_rnn = RNN(self.nx, nh_rnn, cell_type=rnn_type)
        self.mlp = MLP(nh_rnn, nh_mlp)
        self.head_A = nn.Linear(nh_mlp[-1], self.ny * nac)
        self.head_b = nn.Linear(nh_mlp[-1], self.ny * nac)

    def encode_x(self, x):
        return self.x_rnn(x)[-1]

    def encode(self, x, y):
        h_x = self.encode_x(x)
        h = self.mlp(h_x)
        a = self.head_A(h).view(-1, self.nk, self.ny)[:, 0, :]
        b = self.head_b(h).view(-1, self.nk, self.ny)[:, 0, :]
        z = (y - b) / a
        return z

    def forward(self, x, z=None):
        h_x = self.encode_x(x)
        if z is None:
            z = torch.randn((h_x.shape[0], self.ny), device=x.device)
        z = z.repeat_interleave(self.nk, dim=0)
        h = self.mlp(h_x)
        a = self.head_A(h).view(-1, self.ny)
        b = self.head_b(h).view(-1, self.ny)
        y = a * z + b
        return y, a, b

    def sample(self, x, z=None):
        return self.forward(x, z)[0]

    def get_kl(self, a, b):
        var = a ** 2
        KLD = -0.5 * torch.sum(1 + var.log() - b.pow(2) - var)
        return


class DLowWrapper(Module):

    def __init__(self, n_features, n_landmarks, obs_length, pred_length, model_path_vae, model_path_dlow, nh_mlp=[300, 200], nh_rnn=128, use_drnn_mlp=False, rnn_type='lstm', nz=128, nh_mlp_dlow=[300, 200], nk=10, nh_rnn_dlow=128, rnn_type_dlow='gru', only_vae=False):
        super(DLowWrapper, self).__init__()
        self.model_vae = VAE(n_features, n_landmarks, nz, obs_length, pred_length, nh_mlp=nh_mlp, nh_rnn=nh_rnn, use_drnn_mlp=use_drnn_mlp, rnn_type=rnn_type)
        self.model_dlow = DLow(n_features, n_landmarks, obs_length, pred_length, nk, nz=nh_rnn, nh_mlp=nh_mlp_dlow, nh_rnn=nh_rnn_dlow, rnn_type=rnn_type_dlow)
        self.only_vae = only_vae
        self.z_dim = nz
        self.n_features = n_features
        self.n_landmarks = n_landmarks
        self.obs_length = obs_length
        self.pred_length = pred_length
        self.t_length = obs_length + pred_length
        self.load_model(model_path_vae, model_path_dlow)

    def load_model(self, model_path_vae, model_path_dlow):
        self.model_vae.load_state_dict(pickle.load(open(model_path_vae, 'rb'))['model_dict'])
        if not self.only_vae:
            self.model_dlow.load_state_dict(pickle.load(open(model_path_dlow, 'rb'))['model_dict'])

    def to(self, device):
        self.model_vae = self.model_vae
        self.model_dlow = self.model_dlow
        return self

    def eval(self):
        self.model_vae.eval()
        self.model_dlow.eval()

    def forward(self, x, y, nk):
        """
        Args:
            x: batch_size, obs_length, participants, n_landmarks, n_features
            y: batch_size, pred_length, participants, n_landmarks, n_features
        Returns:
        """
        batch_size = x.shape[0]
        participants = x.shape[2]
        device = x.device
        x = x.reshape((batch_size, self.obs_length, -1)).permute(1, 0, 2)
        if not self.only_vae:
            Z_g = self.model_dlow.sample(x)
        else:
            Z_g = torch.randn(batch_size * nk, self.z_dim, device=device)
        x = x.repeat_interleave(nk, dim=1)
        X = x
        Y = self.model_vae.decode(X, Z_g)
        return Y.permute(1, 0, 2).reshape((batch_size, nk, self.pred_length, participants, self.n_landmarks, self.n_features))


class GraphConvolution(nn.Module):
    """
    adapted from : https://github.com/tkipf/gcn/blob/92600c39797c2bfb61a508e52b88fb554df30177/gcn/layers.py#L132
    """

    def __init__(self, in_features, out_features, bias=True, node_n=48):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.att = Parameter(torch.FloatTensor(node_n, node_n))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.att.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(self.att, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GC_Block(nn.Module):

    def __init__(self, in_features, node_n=48, act_f=nn.Tanh(), p_dropout=0, is_bn=False):
        """
        Define a residual block of GCN
        """
        super(GC_Block, self).__init__()
        self.in_features = in_features
        self.out_features = in_features
        self.is_bn = is_bn
        self.gc1 = GraphConvolution(in_features, in_features, node_n=node_n, bias=not is_bn)
        if is_bn:
            self.bn1 = nn.BatchNorm1d(node_n * in_features)
        self.gc2 = GraphConvolution(in_features, in_features, node_n=node_n, bias=not is_bn)
        if is_bn:
            self.bn2 = nn.BatchNorm1d(node_n * in_features)
        self.act_f = act_f

    def forward(self, x):
        y = self.gc1(x)
        if self.is_bn:
            b, n, f = y.shape
            y = self.bn1(y.view(b, -1)).view(b, n, f)
        y = self.act_f(y)
        y = self.gc2(y)
        if self.is_bn:
            b, n, f = y.shape
            y = self.bn2(y.view(b, -1)).view(b, n, f)
        y = self.act_f(y)
        return y + x

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GCN(nn.Module):

    def __init__(self, input_feature, hidden_feature, p_dropout=0, num_stage=1, node_n=48, is_bn=False, act_f=nn.Tanh()):
        """
        :param input_feature: num of input feature
        :param hidden_feature: num of hidden feature
        :param p_dropout: drop out prob.
        :param num_stage: number of residual blocks
        :param node_n: number of nodes in graph
        """
        super(GCN, self).__init__()
        self.num_stage = num_stage
        self.is_bn = is_bn
        self.gc1 = GraphConvolution(input_feature, hidden_feature, node_n=node_n)
        if is_bn:
            self.bn1 = nn.BatchNorm1d(node_n * hidden_feature)
        gcbs = []
        for i in range(num_stage):
            gcbs.append(GC_Block(hidden_feature, node_n=node_n, is_bn=is_bn, act_f=act_f))
        self.gcbs = nn.Sequential(*gcbs)
        self.gc7 = GraphConvolution(hidden_feature, input_feature, node_n=node_n)
        self.act_f = act_f

    def forward(self, x):
        y = self.gc1(x)
        if self.is_bn:
            b, n, f = y.shape
            y = self.bn1(y.view(b, -1)).view(b, n, f)
        y = self.act_f(y)
        y = self.gcbs(y)
        y = self.gc7(y)
        y = y + x
        return y


class GCNParts(nn.Module):

    def __init__(self, input_feature, hidden_feature, parts=[[0, 1, 2], [3, 4, 5], [6, 7, 8, 9], [10, 11, 12], [13, 14, 15]], p_dropout=0, num_stage=1, node_n=48, is_bn=False, act_f=nn.Tanh()):
        """
        :param input_feature: num of input feature
        :param hidden_feature: num of hidden feature
        :param p_dropout: drop out prob.
        :param num_stage: number of residual blocks
        :param node_n: number of nodes in graph
        """
        super(GCNParts, self).__init__()
        self.num_stage = num_stage
        self.is_bn = is_bn
        self.parts = parts
        self.node_n = node_n
        gcns = []
        pall = []
        for p in parts:
            gcns.append(GCN(input_feature, hidden_feature, num_stage=num_stage, node_n=node_n, is_bn=is_bn, act_f=act_f))
        self.gcns = nn.ModuleList(gcns)

    def forward(self, x, z):
        """
        x: bs, node_n, feat
        z: bs, parts_n, feat
        """
        y = x.clone()
        pall = []
        for i, p in enumerate(self.parts):
            zt = z[:, i:i + 1].repeat([1, self.node_n, 1])
            xt = torch.cat([y, zt], dim=-1)
            yt = self.gcns[i](xt)
            y[:, p] = yt[:, p, :x.shape[2]]
        return y


class GSPSWrapper(Module):

    def __init__(self, n_features, n_landmarks, obs_length, pred_length, model_path, parts, num_stage=4, is_bn=True, node_n=16, hidden_dim=256, n_pre=10, nz=64, p_dropout=0, act_f=nn.Tanh()):
        super(GSPSWrapper, self).__init__()
        self.model = GCNParts(n_pre * 3 + nz, hidden_dim, parts=parts, p_dropout=p_dropout, num_stage=num_stage, node_n=node_n, is_bn=is_bn, act_f=act_f)
        self.z_dim = nz
        self.n_features = n_features
        self.n_landmarks = n_landmarks
        self.obs_length = obs_length
        self.pred_length = pred_length
        self.t_length = obs_length + pred_length
        self.parts = parts
        self.dct_n = n_pre
        self.dct_m, self.i_dct_m = get_dct_matrix(self.t_length)
        self.dct_m = torch.from_numpy(self.dct_m)
        self.i_dct_m = torch.from_numpy(self.i_dct_m)
        self.load_model(model_path)

    def load_model(self, model_path):
        model_cp = pickle.load(open(model_path, 'rb'))['model_dict']
        self.model.load_state_dict(model_cp)

    def to(self, device):
        self.model = self.model
        self.dct_m = self.dct_m
        self.i_dct_m = self.i_dct_m
        return self

    def eval(self):
        self.model.eval()

    def forward(self, x, y, nk):
        """
        Args:
            x: batch_size, obs_length, participants, n_landmarks, n_features
            y: batch_size, pred_length, participants, n_landmarks, n_features
        Returns:
        """
        batch_size = x.shape[0]
        participants = x.shape[2]
        device = x.device
        x = x.reshape((batch_size, self.obs_length, -1)).permute(0, 2, 1)
        with torch.no_grad():
            padded_inputs = x[:, :, list(range(self.obs_length)) + [self.obs_length - 1] * self.pred_length]
            padded_inputs_dct = dct_transform_torch(padded_inputs, self.dct_m, dct_n=self.dct_n)
            padded_inputs_dct = padded_inputs_dct.view(batch_size, -1, 3 * self.dct_n)
        all_z = torch.randn(batch_size * nk, len(self.parts), self.z_dim, device=device)
        all_outs_dct = self.model(torch.repeat_interleave(padded_inputs_dct, repeats=nk, dim=0), z=all_z)
        all_outs_dct = all_outs_dct.reshape(batch_size * nk, -1, self.dct_n)
        outputs = reverse_dct_torch(all_outs_dct, self.i_dct_m, self.t_length)
        outputs = outputs.view(batch_size, nk, -1, self.t_length)
        outputs = outputs.permute(0, 1, 3, 2).reshape(batch_size, nk, self.t_length, participants, self.n_landmarks, self.n_features)
        return outputs[:, :, -self.pred_length:]


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (AttentionPool2d,
     lambda: ([], {'spacial_dim': 4, 'embed_dim': 4, 'num_heads_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (BEncoder,
     lambda: ([], {'n_in': 4, 'n_layers': 1, 'dim_hidden': 4, 'use_linear': 4, 'dim_linear': 4}),
     lambda: ([torch.rand([4, 4])], {})),
    (BasicMLP,
     lambda: ([], {'input_dim': 4, 'output_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Downsample,
     lambda: ([], {'channels': 4, 'use_conv': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (GraphConvBlock,
     lambda: ([], {'in_len': 4, 'out_len': 4, 'in_node_n': 4, 'out_node_n': 4}),
     lambda: ([torch.rand([4, 4, 4])], {})),
    (GroupNorm32,
     lambda: ([], {'num_groups': 1, 'num_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (MLP,
     lambda: ([], {'input_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (NormConv2d,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (RNN,
     lambda: ([], {'input_dim': 4, 'out_dim': 4}),
     lambda: ([torch.rand([4, 4, 4])], {})),
    (ResGCB,
     lambda: ([], {'in_len': 4, 'out_len': 4, 'in_node_n': 4, 'out_node_n': 4}),
     lambda: ([torch.rand([4, 4, 4])], {})),
    (SiLU,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (TimestepBlock,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (TimestepEmbedSequential,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (Upsample,
     lambda: ([], {'channels': 4, 'use_conv': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
]

