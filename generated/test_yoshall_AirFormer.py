
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


import numpy as np


import time


import scipy.sparse as sp


from scipy.sparse import linalg


import torch.nn as nn


from abc import abstractmethod


import logging


from typing import Optional


from typing import List


from typing import Union


from torch import nn


from torch import Tensor


from torch.optim.lr_scheduler import MultiStepLR


from torch.optim import Adam


import pandas as pd


import torch.nn.functional as F


from torch.utils.data import DataLoader


from torch.utils.data import TensorDataset


class BaseModel(nn.Module):
    """
    Base class for all models
    """

    def __init__(self, name, dataset, device, num_nodes, seq_len, horizon, input_dim, output_dim):
        super(BaseModel, self).__init__()
        self.name = name
        self.dataset = dataset
        self.device = device
        self.num_nodes = num_nodes
        self.seq_len = seq_len
        self.horizon = horizon
        self.input_dim = input_dim
        self.output_dim = output_dim

    @abstractmethod
    def forward(self):
        """
        Forward pass logic
        :return: Model output
        """
        raise NotImplementedError

    def param_num(self, str):
        return sum([param.nelement() for param in self.parameters()])

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)


class AirEmbedding(nn.Module):
    """
    Embed catagorical variables.
    """

    def __init__(self):
        super(AirEmbedding, self).__init__()
        self.embed_wdir = nn.Embedding(11, 3)
        self.embed_weather = nn.Embedding(18, 4)
        self.embed_day = nn.Embedding(24, 3)
        self.embed_hour = nn.Embedding(7, 5)

    def forward(self, x):
        x_wdir = self.embed_wdir(x[..., 0])
        x_weather = self.embed_weather(x[..., 1])
        x_day = self.embed_day(x[..., 2])
        x_hour = self.embed_hour(x[..., 3])
        out = torch.cat((x_wdir, x_weather, x_day, x_hour), -1)
        return out


class LatentLayer(nn.Module):
    """
    The latent layer to compute mean and std
    """

    def __init__(self, dm_dim, latent_dim_in, latent_dim_out, hidden_dim, num_layers=2):
        super(LatentLayer, self).__init__()
        self.num_layers = num_layers
        self.enc_in = nn.Sequential(nn.Conv2d(dm_dim + latent_dim_in, hidden_dim, 1))
        layers = []
        for _ in range(num_layers):
            layers.append(nn.Conv2d(hidden_dim, hidden_dim, 1))
            layers.append(nn.ReLU(inplace=True))
        self.enc_hidden = nn.Sequential(*layers)
        self.enc_out_1 = nn.Conv2d(hidden_dim, latent_dim_out, 1)
        self.enc_out_2 = nn.Conv2d(hidden_dim, latent_dim_out, 1)

    def forward(self, x):
        h = self.enc_in(x)
        for i in range(self.num_layers):
            h = self.enc_hidden[i](h)
        mu = torch.minimum(self.enc_out_1(h), torch.ones_like(h) * 10)
        sigma = torch.minimum(self.enc_out_2(h), torch.ones_like(h) * 10)
        return mu, sigma


class StochasticModel(nn.Module):
    """
    The generative model.
    The inference model can also use this implementation, while the input should be shifted
    """

    def __init__(self, dm_dim, latent_dim, num_blocks=4):
        super(StochasticModel, self).__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_blocks - 1):
            self.layers.append(LatentLayer(dm_dim, latent_dim, latent_dim, latent_dim, 2))
        self.layers.append(LatentLayer(dm_dim, 0, latent_dim, latent_dim, 2))

    def reparameterize(self, mu, sigma):
        eps = torch.randn_like(sigma, requires_grad=False)
        return mu + eps * sigma

    def forward(self, d):
        _mu, _logsigma = self.layers[-1](d[-1])
        _sigma = torch.exp(_logsigma) + 0.001
        mus = [_mu]
        sigmas = [_sigma]
        z = [self.reparameterize(_mu, _sigma)]
        for i in reversed(range(len(self.layers) - 1)):
            _mu, _logsigma = self.layers[i](torch.cat((d[i], z[-1]), dim=1))
            _sigma = torch.exp(_logsigma) + 0.001
            mus.append(_mu)
            sigmas.append(_sigma)
            z.append(self.reparameterize(_mu, _sigma))
        z = torch.stack(z)
        mus = torch.stack(mus)
        sigmas = torch.stack(sigmas)
        return z, mus, sigmas


class FeedForward(nn.Module):

    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, dim), nn.Dropout(dropout))

    def forward(self, x):
        return self.net(x)


class PreNorm(nn.Module):

    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class TemporalAttention(nn.Module):

    def __init__(self, dim, heads=2, window_size=1, qkv_bias=False, qk_scale=None, dropout=0.0, causal=True, device=None):
        super().__init__()
        assert dim % heads == 0, f'dim {dim} should be divided by num_heads {heads}.'
        self.dim = dim
        self.num_heads = heads
        self.causal = causal
        head_dim = dim // heads
        self.scale = qk_scale or head_dim ** -0.5
        self.window_size = window_size
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)
        self.mask = torch.tril(torch.ones(window_size, window_size))

    def forward(self, x):
        B_prev, T_prev, C_prev = x.shape
        if self.window_size > 0:
            x = x.reshape(-1, self.window_size, C_prev)
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, -1, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = q @ k.transpose(-2, -1) * self.scale
        if self.causal:
            attn = attn.masked_fill_(self.mask == 0, float('-inf'))
        x = (attn.softmax(dim=-1) @ v).transpose(1, 2).reshape(B, T, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        if self.window_size > 0:
            x = x.reshape(B_prev, T_prev, C_prev)
        return x


class CT_MSA(nn.Module):

    def __init__(self, dim, depth, heads, window_size, mlp_dim, num_time, dropout=0.0, device=None):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, num_time, dim))
        self.layers = nn.ModuleList([])
        for i in range(depth):
            self.layers.append(nn.ModuleList([TemporalAttention(dim=dim, heads=heads, window_size=window_size, dropout=dropout, device=device), PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))]))

    def forward(self, x):
        b, c, n, t = x.shape
        x = x.permute(0, 2, 3, 1).reshape(b * n, t, c)
        x = x + self.pos_embedding
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        x = x.reshape(b, n, t, c).permute(0, 3, 1, 2)
        return x


class SpatialAttention(nn.Module):

    def __init__(self, dim, heads=4, qkv_bias=False, qk_scale=None, dropout=0.0, num_sectors=17, assignment=None, mask=None):
        super().__init__()
        assert dim % heads == 0, f'dim {dim} should be divided by num_heads {heads}.'
        self.dim = dim
        self.num_heads = heads
        head_dim = dim // heads
        self.scale = qk_scale or head_dim ** -0.5
        self.num_sector = num_sectors
        self.assignment = assignment
        self.mask = mask
        self.q_linear = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv_linear = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.relative_bias = nn.Parameter(torch.randn(heads, 1, num_sectors))
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x):
        B, N, C = x.shape
        pre_kv = torch.einsum('bnc,mnr->bmrc', x, self.assignment)
        pre_kv = pre_kv.reshape(-1, self.num_sector, C)
        pre_q = x.reshape(-1, 1, C)
        q = self.q_linear(pre_q).reshape(B * N, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        kv = self.kv_linear(pre_kv).reshape(B * N, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        attn = q @ k.transpose(-2, -1) * self.scale
        attn = attn.reshape(B, N, self.num_heads, 1, self.num_sector) + self.relative_bias
        mask = self.mask.reshape(1, N, 1, 1, self.num_sector)
        attn = attn.masked_fill_(mask, float('-inf')).reshape(B * N, self.num_heads, 1, self.num_sector).softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class DS_MSA(nn.Module):

    def __init__(self, dim, depth, heads, mlp_dim, assignment, mask, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([])
        for i in range(depth):
            self.layers.append(nn.ModuleList([SpatialAttention(dim, heads=heads, dropout=dropout, assignment=assignment, mask=mask, num_sectors=assignment.shape[-1]), PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))]))

    def forward(self, x):
        b, c, n, t = x.shape
        x = x.permute(0, 3, 2, 1).reshape(b * t, n, c)
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        x = x.reshape(b, t, n, c).permute(0, 3, 2, 1)
        return x


dartboard_map = {(0): '50-200', (1): '50-200-500', (2): '50', (3): '25-100-250'}


class AirFormer(BaseModel):
    """
    the AirFormer model
    """

    def __init__(self, dropout=0.3, spatial_flag=True, stochastic_flag=True, hidden_channels=32, end_channels=512, blocks=4, mlp_expansion=2, num_heads=2, dartboard=0, **args):
        super(AirFormer, self).__init__(**args)
        self.dropout = dropout
        self.blocks = blocks
        self.spatial_flag = spatial_flag
        self.stochastic_flag = stochastic_flag
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.s_modules = nn.ModuleList()
        self.t_modules = nn.ModuleList()
        self.embedding_air = AirEmbedding()
        self.alpha = 10
        self.get_dartboard_info(dartboard)
        self.start_conv = nn.Conv2d(in_channels=self.input_dim, out_channels=hidden_channels, kernel_size=(1, 1))
        for b in range(blocks):
            window_size = self.seq_len // 2 ** (blocks - b - 1)
            self.t_modules.append(CT_MSA(hidden_channels, depth=1, heads=num_heads, window_size=window_size, mlp_dim=hidden_channels * mlp_expansion, num_time=self.seq_len, device=self.device))
            if self.spatial_flag:
                self.s_modules.append(DS_MSA(hidden_channels, depth=1, heads=num_heads, mlp_dim=hidden_channels * mlp_expansion, assignment=self.assignment, mask=self.mask, dropout=dropout))
            else:
                self.residual_convs.append(nn.Conv1d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=(1, 1)))
            self.bn.append(nn.BatchNorm2d(hidden_channels))
        if stochastic_flag:
            self.generative_model = StochasticModel(hidden_channels, hidden_channels, blocks)
            self.inference_model = StochasticModel(hidden_channels, hidden_channels, blocks)
            self.reconstruction_model = nn.Sequential(nn.Conv2d(in_channels=hidden_channels * blocks, out_channels=end_channels, kernel_size=(1, 1), bias=True), nn.ReLU(inplace=True), nn.Conv2d(in_channels=end_channels, out_channels=self.input_dim, kernel_size=(1, 1), bias=True))
        if self.stochastic_flag:
            self.end_conv_1 = nn.Conv2d(in_channels=hidden_channels * blocks * 2, out_channels=end_channels, kernel_size=(1, 1), bias=True)
        else:
            self.end_conv_1 = nn.Conv2d(in_channels=hidden_channels * blocks, out_channels=end_channels, kernel_size=(1, 1), bias=True)
        self.end_conv_2 = nn.Conv2d(in_channels=end_channels, out_channels=self.horizon * self.output_dim, kernel_size=(1, 1), bias=True)

    def get_dartboard_info(self, dartboard):
        """
        get dartboard-related attributes
        """
        path_assignment = 'data/local_partition/' + dartboard_map[dartboard] + '/assignment.npy'
        path_mask = 'data/local_partition/' + dartboard_map[dartboard] + '/mask.npy'
        None
        self.assignment = torch.from_numpy(np.load(path_assignment)).float()
        self.mask = torch.from_numpy(np.load(path_mask)).bool()

    def forward(self, inputs, supports=None):
        """
        inputs: the historical data
        supports: adjacency matrix (actually our method doesn't use it)
                Including adj here is for consistency with GNN-based methods
        """
        x_embed = self.embedding_air(inputs[..., 11:15].long())
        x = torch.cat((inputs[..., :11], x_embed, inputs[..., 15:]), -1)
        x = x.permute(0, 3, 2, 1)
        x = self.start_conv(x)
        d = []
        for i in range(self.blocks):
            if self.spatial_flag:
                x = self.s_modules[i](x)
            else:
                x = self.residual_convs[i](x)
            x = self.t_modules[i](x)
            x = self.bn[i](x)
            d.append(x)
        d = torch.stack(d)
        if self.stochastic_flag:
            d_shift = [nn.functional.pad(d[i], pad=(1, 0))[..., :-1] for i in range(len(d))]
            d_shift = torch.stack(d_shift)
            z_p, mu_p, sigma_p = self.generative_model(d_shift)
            z_q, mu_q, sigma_q = self.inference_model(d)
            p = torch.distributions.Normal(mu_p, sigma_p)
            q = torch.distributions.Normal(mu_q, sigma_q)
            kl_loss = torch.distributions.kl_divergence(q, p).mean() * self.alpha
            num_blocks, B, C, N, T = d.shape
            z_p = z_p.permute(1, 0, 2, 3, 4).reshape(B, -1, N, T)
            z_q = z_q.permute(1, 0, 2, 3, 4).reshape(B, -1, N, T)
            x_rec = self.reconstruction_model(z_p)
            x_rec = x_rec.permute(0, 3, 2, 1)
            num_blocks, B, C, N, T = d.shape
            d = d.permute(1, 0, 2, 3, 4).reshape(B, -1, N, T)
            x_hat = torch.cat([d[..., -1:], z_q[..., -1:]], dim=1)
            x_hat = F.relu(self.end_conv_1(x_hat))
            x_hat = self.end_conv_2(x_hat)
            return x_hat, x_rec, kl_loss
        else:
            num_blocks, B, C, N, T = d.shape
            d = d.permute(1, 0, 2, 3, 4).reshape(B, -1, N, T)
            x_hat = F.relu(d[..., -1:])
            x_hat = F.relu(self.end_conv_1(x_hat))
            x_hat = self.end_conv_2(x_hat)
            return x_hat


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (CT_MSA,
     lambda: ([], {'dim': 4, 'depth': 1, 'heads': 4, 'window_size': 4, 'mlp_dim': 4, 'num_time': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (FeedForward,
     lambda: ([], {'dim': 4, 'hidden_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (LatentLayer,
     lambda: ([], {'dm_dim': 4, 'latent_dim_in': 4, 'latent_dim_out': 4, 'hidden_dim': 4}),
     lambda: ([torch.rand([4, 8, 64, 64])], {})),
    (PreNorm,
     lambda: ([], {'dim': 4, 'fn': torch.nn.ReLU()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (TemporalAttention,
     lambda: ([], {'dim': 4}),
     lambda: ([torch.rand([4, 4, 4])], {})),
]

