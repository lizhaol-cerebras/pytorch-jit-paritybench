
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


import math


from typing import Union


from torch import nn


import torch.nn.functional as F


class RMSNorm(nn.Module):

    def __init__(self, d_model: 'int', eps: 'float'=1e-05):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight
        return output


class PScan(torch.autograd.Function):

    @staticmethod
    def pscan(A, X):
        B, D, L, _ = A.size()
        num_steps = int(math.log2(L))
        Aa = A
        Xa = X
        for k in range(num_steps):
            T = 2 * (Xa.size(2) // 2)
            Aa = Aa[:, :, :T].view(B, D, T // 2, 2, -1)
            Xa = Xa[:, :, :T].view(B, D, T // 2, 2, -1)
            Xa[:, :, :, 1].add_(Aa[:, :, :, 1].mul(Xa[:, :, :, 0]))
            Aa[:, :, :, 1].mul_(Aa[:, :, :, 0])
            Aa = Aa[:, :, :, 1]
            Xa = Xa[:, :, :, 1]
        for k in range(num_steps - 1, -1, -1):
            Aa = A[:, :, 2 ** k - 1:L:2 ** k]
            Xa = X[:, :, 2 ** k - 1:L:2 ** k]
            T = 2 * (Xa.size(2) // 2)
            if T < Xa.size(2):
                Xa[:, :, -1].add_(Aa[:, :, -1].mul(Xa[:, :, -2]))
                Aa[:, :, -1].mul_(Aa[:, :, -2])
            Aa = Aa[:, :, :T].view(B, D, T // 2, 2, -1)
            Xa = Xa[:, :, :T].view(B, D, T // 2, 2, -1)
            Xa[:, :, 1:, 0].add_(Aa[:, :, 1:, 0].mul(Xa[:, :, :-1, 1]))
            Aa[:, :, 1:, 0].mul_(Aa[:, :, :-1, 1])

    @staticmethod
    def forward(ctx, A_in, X_in):
        """
        Applies the parallel scan operation, as defined above. Returns a new tensor.

        Args:
            A_in : (B, L, D, N)
            X_in : (B, L, D, N)

        Returns:
            H : (B, L, D, N)
        """
        A = A_in.clone()
        X = X_in.clone()
        A = A.transpose(2, 1)
        X = X.transpose(2, 1)
        PScan.pscan(A, X)
        ctx.save_for_backward(A_in, X)
        return X.transpose(2, 1)

    @staticmethod
    def backward(ctx, grad_output_in):
        """
        Flows the gradient from the output to the input. Returns two new tensors.

        Args:
            ctx : A_in : (B, L, D, N), X : (B, D, L, N)
            grad_output_in : (B, L, D, N)

        Returns:
            gradA : (B, L, D, N), gradX : (B, L, D, N)
        """
        A_in, X = ctx.saved_tensors
        A = A_in.clone()
        A = A.transpose(2, 1)
        A = torch.cat((A[:, :, :1], A[:, :, 1:].flip(2)), dim=2)
        grad_output_b = grad_output_in.transpose(2, 1)
        grad_output_b = grad_output_b.flip(2)
        PScan.pscan(A, grad_output_b)
        grad_output_b = grad_output_b.flip(2)
        Q = torch.zeros_like(X)
        Q[:, :, 1:].add_(X[:, :, :-1] * grad_output_b[:, :, 1:])
        return Q.transpose(2, 1), grad_output_b.transpose(2, 1)


pscan = PScan.apply


class MambaBlock(nn.Module):

    def __init__(self, config: 'MambaConfig'):
        super().__init__()
        self.config = config
        self.in_proj = nn.Linear(config.dim, 2 * config.d_inner, bias=config.bias)
        self.conv1d = nn.Conv1d(in_channels=config.d_inner, out_channels=config.d_inner, kernel_size=config.d_conv, bias=config.conv_bias, groups=config.d_inner, padding=config.d_conv - 1)
        self.x_proj = nn.Linear(config.d_inner, config.dt_rank + 2 * config.d_state, bias=False)
        self.dt_proj = nn.Linear(config.dt_rank, config.d_inner, bias=True)
        dt_init_std = config.dt_rank ** -0.5 * config.dt_scale
        if config.dt_init == 'constant':
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif config.dt_init == 'random':
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        dt = torch.exp(torch.rand(config.d_inner) * (math.log(config.dt_max) - math.log(config.dt_min)) + math.log(config.dt_min)).clamp(min=config.dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        A = torch.arange(1, config.d_state + 1, dtype=torch.float32).repeat(config.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(config.d_inner))
        self.out_proj = nn.Linear(config.d_inner, config.dim, bias=config.bias)

    def forward(self, x):
        _, L, _ = x.shape
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        x = x.transpose(1, 2)
        x = self.conv1d(x)[:, :, :L]
        x = x.transpose(1, 2)
        x = F.silu(x)
        y = self.ssm(x)
        z = F.silu(z)
        output = y * z
        output = self.out_proj(output)
        return output

    def ssm(self, x):
        A = -torch.exp(self.A_log.float())
        D = self.D.float()
        deltaBC = self.x_proj(x)
        delta, B, C = torch.split(deltaBC, [self.config.dt_rank, self.config.d_state, self.config.d_state], dim=-1)
        delta = F.softplus(self.dt_proj(delta))
        if self.config.pscan:
            y = self.selective_scan(x, delta, A, B, C, D)
        else:
            y = self.selective_scan_seq(x, delta, A, B, C, D)
        return y

    def selective_scan(self, x, delta, A, B, C, D):
        deltaA = torch.exp(delta.unsqueeze(-1) * A)
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(2)
        BX = deltaB * x.unsqueeze(-1)
        hs = pscan(deltaA, BX)
        y = (hs @ C.unsqueeze(-1)).squeeze(3)
        y = y + D * x
        return y

    def selective_scan_seq(self, x, delta, A, B, C, D):
        _, L, _ = x.shape
        deltaA = torch.exp(delta.unsqueeze(-1) * A)
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(2)
        BX = deltaB * x.unsqueeze(-1)
        h = torch.zeros(x.size(0), self.config.d_inner, self.config.d_state, device=deltaA.device)
        hs = []
        for t in range(0, L):
            h = deltaA[:, t] * h + BX[:, t]
            hs.append(h)
        hs = torch.stack(hs, dim=1)
        y = (hs @ C.unsqueeze(-1)).squeeze(3)
        y = y + D * x
        return y
    """
    Concerning auto-regressive inference

    The cool part of using Mamba : inference is constant wrt to sequence length
    We just have to keep in cache, for each layer, two things :
    - the hidden state h (which is (B, ED, N)), as you typically would when doing inference with a RNN
    - the last d_conv-1 inputs of the layer, to be able to compute the 1D conv which is a convolution over the time dimension
      (d_conv is fixed so this doesn't incur a growing cache as we progress on generating the sequence)
      (and d_conv is usually very small, like 4, so we just have to "remember" the last 3 inputs)

    Concretely, these two quantities are put inside a cache tuple, and are named h and inputs respectively.
    h is (B, ED, N), and inputs is (B, ED, d_conv-1)
    The MambaBlock.step() receives this cache, and, along with outputing the output, alos outputs the updated cache for the next call.

    The cache object is initialized as follows : (None, torch.zeros()).
    When h is None, the selective scan function detects it and start with h=0.
    The torch.zeros() isn't a problem (it's same as just feeding the input, because the conv1d is padded)

    As we need one such cache variable per layer, we store a caches object, which is simply a list of cache object. (See mamba_lm.py)
    """

    def step(self, x, cache):
        h, inputs = cache
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=1)
        x_cache = x.unsqueeze(2)
        x = self.conv1d(torch.cat([inputs, x_cache], dim=2))[:, :, self.config.d_conv - 1]
        x = F.silu(x)
        y, h = self.ssm_step(x, h)
        z = F.silu(z)
        output = y * z
        output = self.out_proj(output)
        inputs = torch.cat([inputs[:, :, 1:], x_cache], dim=2)
        cache = h, inputs
        return output, cache

    def ssm_step(self, x, h):
        A = -torch.exp(self.A_log.float())
        D = self.D.float()
        deltaBC = self.x_proj(x)
        delta, B, C = torch.split(deltaBC, [self.config.dt_rank, self.config.d_state, self.config.d_state], dim=-1)
        delta = F.softplus(self.dt_proj(delta))
        deltaA = torch.exp(delta.unsqueeze(-1) * A)
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(1)
        BX = deltaB * x.unsqueeze(-1)
        if h is None:
            h = torch.zeros(x.size(0), self.config.d_inner, self.config.d_state, device=deltaA.device)
        h = deltaA * h + BX
        y = (h @ C.unsqueeze(-1)).squeeze(2)
        y = y + D * x
        return y, h.squeeze(1)


class ResidualBlock(nn.Module):

    def __init__(self, config: 'MambaConfig'):
        super().__init__()
        self.mixer = MambaBlock(config)
        self.norm = RMSNorm(config.dim)

    def forward(self, x):
        output = self.mixer(self.norm(x)) + x
        return output

    def step(self, x, cache):
        output, cache = self.mixer.step(self.norm(x), cache)
        output = output + x
        return output, cache


class Mamba(nn.Module):

    def __init__(self, config: 'MambaConfig'):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([ResidualBlock(config) for _ in range(config.depth)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def step(self, x, caches):
        for i, layer in enumerate(self.layers):
            x, caches[i] = layer.step(x, caches[i])
        return x, caches


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (RMSNorm,
     lambda: ([], {'d_model': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
]

