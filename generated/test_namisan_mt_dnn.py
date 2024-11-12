
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


import random


import torch


import numpy


import collections


import logging


from typing import Optional


from typing import Tuple


import numpy as np


from torch.utils.data import DataLoader


import numpy.ma as ma


import math


from torch.nn.init import uniform


from torch.nn.init import normal


from torch.nn.init import eye


from torch.nn.init import xavier_uniform


from torch.nn.init import xavier_normal


from torch.nn.init import kaiming_uniform


from torch.nn.init import kaiming_normal


from torch.nn.init import orthogonal


import torch.nn as nn


import torch.nn.functional as F


from copy import deepcopy


from torch.nn import Parameter


from functools import wraps


from functools import partial


from torch.nn.utils import weight_norm


from torch.nn.parameter import Parameter


import copy


import warnings


from typing import Union


from typing import Dict


from typing import List


from torch import nn


from torch import Tensor


from torch.nn import CrossEntropyLoss


from torch.utils.checkpoint import checkpoint


from torch.nn.modules.normalization import LayerNorm as T5LayerNorm


from collections import Sequence


from torch.utils.data import Dataset


from torch.utils.data import BatchSampler


from torch.utils.data import Sampler


import enum


from torch.nn.modules.loss import _Loss


from enum import IntEnum


from torch.nn.modules.normalization import LayerNorm


import torch.optim as optim


from torch.optim.lr_scheduler import *


from torch.optim import Optimizer


class DropoutWrapper(nn.Module):
    """
    This is a dropout wrapper which supports the fix mask dropout
    """

    def __init__(self, dropout_p=0, enable_vbp=True):
        super(DropoutWrapper, self).__init__()
        """variational dropout means fix dropout mask
        ref: https://discuss.pytorch.org/t/dropout-for-rnns/633/11
        """
        self.enable_variational_dropout = enable_vbp
        self.dropout_p = dropout_p

    def forward(self, x):
        """
        :param x: batch * len * input_size
        """
        if self.training == False or self.dropout_p == 0:
            return x
        if len(x.size()) == 3:
            mask = 1.0 / (1 - self.dropout_p) * torch.bernoulli((1 - self.dropout_p) * (x.data.new(x.size(0), x.size(2)).zero_() + 1))
            mask.requires_grad = False
            return mask.unsqueeze(1).expand_as(x) * x
        else:
            return F.dropout(x, p=self.dropout_p, training=self.training)


def _dummy(*args, **kwargs):
    return


def _norm(p, dim):
    """Computes the norm over all dimensions except dim"""
    if dim is None:
        return p.norm()
    elif dim == 0:
        output_size = (p.size(0),) + (1,) * (p.dim() - 1)
        return p.contiguous().view(p.size(0), -1).norm(dim=1).view(*output_size)
    elif dim == p.dim() - 1:
        output_size = (1,) * (p.dim() - 1) + (p.size(-1),)
        return p.contiguous().view(-1, p.size(-1)).norm(dim=0).view(*output_size)
    else:
        return _norm(p.transpose(0, dim), 0).transpose(0, dim)


class WeightNorm(torch.nn.Module):

    def __init__(self, weights, dim):
        super(WeightNorm, self).__init__()
        self.weights = weights
        self.dim = dim

    def compute_weight(self, module, name):
        g = getattr(module, name + '_g')
        v = getattr(module, name + '_v')
        return v * (g / _norm(v, self.dim))

    @staticmethod
    def apply(module, weights, dim):
        if issubclass(type(module), torch.nn.RNNBase):
            module.flatten_parameters = _dummy
        if weights is None:
            weights = [w for w in module._parameters.keys() if 'weight' in w]
        fn = WeightNorm(weights, dim)
        for name in weights:
            if hasattr(module, name):
                None
                weight = getattr(module, name)
                del module._parameters[name]
                module.register_parameter(name + '_g', Parameter(_norm(weight, dim).data))
                module.register_parameter(name + '_v', Parameter(weight.data))
                setattr(module, name, fn.compute_weight(module, name))
        module.register_forward_pre_hook(fn)
        return fn

    def remove(self, module):
        for name in self.weights:
            weight = self.compute_weight(module)
            delattr(module, name)
            del module._parameters[name + '_g']
            del module._parameters[name + '_v']
            module.register_parameter(name, Parameter(weight.data))

    def __call__(self, module, inputs):
        for name in self.weights:
            setattr(module, name, self.compute_weight(module, name))


def linear(x):
    return x


def activation(func_a):
    """Activation function wrapper"""
    try:
        f = eval('nn.{}'.format(func_a))
    except:
        f = linear
    return f


class Pooler(nn.Module):

    def __init__(self, hidden_size, dropout_p=0.1, actf='tanh'):
        super(Pooler, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = activation(actf)
        self.dropout = DropoutWrapper(dropout_p=dropout_p)

    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, 0]
        first_token_tensor = self.dropout(first_token_tensor)
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class DropoutNd(nn.Module):

    def __init__(self, p: 'float'=0.5, tie=True, transposed=True):
        """
        tie: tie dropout mask across sequence lengths (Dropout1d/2d/3d)
        """
        super().__init__()
        if p < 0 or p >= 1:
            raise ValueError('dropout probability has to be in [0, 1), but got {}'.format(p))
        self.p = p
        self.tie = tie
        self.transposed = transposed
        self.binomial = torch.distributions.binomial.Binomial(probs=1 - self.p)

    def forward(self, X):
        """ X: (batch, dim, lengths...) """
        if self.training:
            if not self.transposed:
                X = rearrange(X, 'b d ... -> b ... d')
            mask_shape = X.shape[:2] + (1,) * (X.ndim - 2) if self.tie else X.shape
            mask = torch.rand(*mask_shape, device=X.device) < 1.0 - self.p
            X = X * mask * (1.0 / (1 - self.p))
            if not self.transposed:
                X = rearrange(X, 'b ... d -> b d ...')
            return X
        return X


class OptimModule(nn.Module):
    """ Interface for Module that allows registering buffers/parameters with configurable optimizer hyperparameters """

    def register(self, name, tensor, lr=None):
        """Register a tensor with a configurable learning rate and 0 weight decay"""
        if lr == 0.0:
            self.register_buffer(name, tensor)
        else:
            self.register_parameter(name, nn.Parameter(tensor))
            optim = {'weight_decay': 0.0}
            if lr is not None:
                optim['lr'] = lr
            setattr(getattr(self, name), '_s4_optimizer', optim)


_c2r = torch.view_as_real


_conj = lambda x: torch.cat([x, x.conj()], dim=-1)


_r2c = torch.view_as_complex


has_cauchy_extension = False


def create_logger(name, silent=False, to_disk=False, log_file=None):
    """Logger wrapper"""
    log = logging.getLogger(name)
    log.setLevel(logging.DEBUG)
    log.propagate = False
    formatter = logging.Formatter(fmt='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S')
    if not silent:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        log.addHandler(ch)
    if to_disk:
        log_file = log_file if log_file is not None else strftime('%Y-%m-%d-%H-%M-%S.log', gmtime())
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        log.addHandler(fh)
    return log


def train_config(parser):
    parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available(), help='whether to use GPU acceleration.')
    parser.add_argument('--log_per_updates', type=int, default=500)
    parser.add_argument('--save_per_updates', type=int, default=10000)
    parser.add_argument('--save_per_updates_on', action='store_true')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--batch_size_eval', type=int, default=8)
    parser.add_argument('--optimizer', default='adamax', help='supported optimizer: adamax, sgd, adadelta, adam')
    parser.add_argument('--grad_clipping', type=float, default=0)
    parser.add_argument('--global_grad_clipping', type=float, default=1.0)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--learning_rate', type=float, default=5e-05)
    parser.add_argument('--momentum', type=float, default=0)
    parser.add_argument('--warmup', type=float, default=0.1)
    parser.add_argument('--warmup_schedule', type=str, default='warmup_linear')
    parser.add_argument('--adam_eps', type=float, default=1e-06)
    parser.add_argument('--vb_dropout', action='store_false')
    parser.add_argument('--dropout_p', type=float, default=0.1)
    parser.add_argument('--dropout_w', type=float, default=0.0)
    parser.add_argument('--bert_dropout_p', type=float, default=0.1)
    parser.add_argument('--model_ckpt', default='checkpoints/model_0.pt', type=str)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--scheduler_type', type=int, default=0, help='0: linear, 1: cosine, 2 constant')
    parser.add_argument('--output_dir', default='checkpoint')
    parser.add_argument('--seed', type=int, default=2018, help='random seed for data shuffling, embedding init, etc.')
    parser.add_argument('--grad_accumulation_step', type=int, default=1)
    parser.add_argument('--fp16', action='store_true', help='Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit')
    parser.add_argument('--fp16_opt_level', type=str, default='O1', help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3'].See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument('--adv_train', action='store_true')
    parser.add_argument('--adv_opt', default=0, type=int)
    parser.add_argument('--adv_norm_level', default=0, type=int)
    parser.add_argument('--adv_p_norm', default='inf', type=str)
    parser.add_argument('--adv_alpha', default=1, type=float)
    parser.add_argument('--adv_k', default=1, type=int)
    parser.add_argument('--adv_step_size', default=1e-05, type=float)
    parser.add_argument('--adv_noise_var', default=1e-05, type=float)
    parser.add_argument('--adv_epsilon', default=1e-06, type=float)
    parser.add_argument('--encode_mode', action='store_true', help='only encode test data')
    parser.add_argument('--debug', action='store_true', help='print debug info')
    parser.add_argument('--transformer_cache', default='.cache', type=str)
    return parser


def power(L, A, v=None):
    """ Compute A^L and the scan sum_i A^i v_i

    A: (..., N, N)
    v: (..., N, L)
    """
    I = torch.eye(A.shape[-1])
    powers = [A]
    l = 1
    while True:
        if L % 2 == 1:
            I = powers[-1] @ I
        L //= 2
        if L == 0:
            break
        l *= 2
        powers.append(powers[-1] @ powers[-1])
    if v is None:
        return I
    k = v.size(-1) - l
    v_ = powers.pop() @ v[..., l:]
    v = v[..., :l]
    v[..., :k] = v[..., :k] + v_
    while v.size(-1) > 1:
        v = rearrange(v, '... (z l) -> ... z l', z=2)
        v = v[..., 0, :] + powers.pop() @ v[..., 1, :]
    return I, v.squeeze(-1)


class SSKernelNPLR(OptimModule):
    """ Stores a representation of and computes the SSKernel function K_L(A^dt, B^dt, C) corresponding to a discretized state space, where A is Normal + Low Rank (NPLR)
    """

    @torch.no_grad()
    def _setup_C(self, L):
        """ Construct C~ from C

        Two modes are supported: go directly to length L if self.L is 1, or length is doubled
        """
        if self.L.item() == 0:
            if self.verbose:
                logger.info(f'S4: Initializing kernel to length {L}')
            double_length = False
        elif L > self.L.item():
            if self.verbose:
                logger.info(f'S4: Doubling length from L = {self.L.item()} to {2 * self.L.item()}')
            double_length = True
            L = self.L.item()
        else:
            return
        C = _r2c(self.C)
        dA, _ = self._setup_state()
        dA_L = power(L, dA)
        C_ = _conj(C)
        prod = contract('h m n, c h n -> c h m', dA_L.transpose(-1, -2), C_)
        if double_length:
            prod = -prod
        C_ = C_ - prod
        C_ = C_[..., :self.N]
        self.C.copy_(_c2r(C_))
        self.L = 2 * self.L if double_length else self.L + L

    def _omega(self, L, dtype, device, cache=True):
        """ Calculate (and cache) FFT nodes and their "unprocessed" version with the bilinear transform
        This should be called everytime the internal length self.L changes """
        if cache and hasattr(self, 'omega') and self.omega.size(-1) == L // 2 + 1:
            return self.omega, self.z
        omega = torch.tensor(np.exp(-2.0j * np.pi / L), dtype=dtype, device=device)
        omega = omega ** torch.arange(0, L // 2 + 1, device=device)
        z = 2 * (1 - omega) / (1 + omega)
        if cache:
            self.omega = omega
            self.z = z
        return omega, z

    def __init__(self, w, P, B, C, log_dt, L=None, lr=None, verbose=False, keops=False, real_type='exp', real_tolerance=0.001, bandlimit=None):
        """
        L: Maximum length; this module computes an SSM kernel of length L
        A is represented by diag(w) - PP^*
        w: (S, N) diagonal part
        P: (R, S, N) low-rank part

        B: (S, N)
        C: (C, H, N)
        dt: (H) timescale per feature
        lr: [dict | float | None] hook to set lr of special parameters (A, B, dt)

        Dimensions:
        N (or d_state): state size
        H (or d_model): total SSM copies
        S (or n_ssm): number of trainable copies of (A, B, dt); must divide H
        R (or rank): rank of low-rank part
        C (or channels): system is 1-dim to C-dim

        The forward pass of this Module returns a tensor of shape (C, H, L)

        Note: tensor shape N here denotes half the true state size, because of conjugate symmetry
        """
        super().__init__()
        self.verbose = verbose
        self.keops = keops
        self.bandlimit = bandlimit
        self.real_type = real_type
        self.real_tolerance = real_tolerance
        self.rank = P.shape[-3]
        assert w.size(-1) == P.size(-1) == B.size(-1) == C.size(-1)
        self.H = log_dt.size(-1)
        self.N = w.size(-1)
        assert w.size(-2) == P.size(-2) == B.size(-2)
        assert self.H % w.size(0) == 0
        self.n_ssm = w.size(0)
        self.broadcast = self.H // w.size(0)
        C = C.expand(torch.broadcast_shapes(C.shape, (1, self.H, self.N)))
        B = B.unsqueeze(0)
        self.C = nn.Parameter(_c2r(_resolve_conj(C)))
        if lr is None or isinstance(lr, float):
            lr_dict = {}
        else:
            lr_dict, lr = lr, None
        self.register('log_dt', log_dt, lr_dict.get('dt', lr))
        self.register('B', _c2r(B), lr_dict.get('B', lr))
        self.register('P', _c2r(P), lr_dict.get('A', lr))
        self.register('inv_w_real', self._w_init(w.real), lr_dict.get('A', lr))
        self.register('w_imag', w.imag, lr_dict.get('A', lr))
        self.l_max = L
        self.register_buffer('L', torch.tensor(0))

    def _w_init(self, w_real):
        w_real = torch.clamp(w_real, max=-self.real_tolerance)
        if self.real_type == 'none':
            return -w_real
        elif self.real_type == 'exp':
            return torch.log(-w_real)
        elif self.real_type == 'relu':
            return -w_real
        elif self.real_type == 'sigmoid':
            return torch.logit(-w_real)
        elif self.real_type == 'softplus':
            return torch.log(torch.exp(-w_real) - 1)
        else:
            raise NotImplementedError

    def _w(self):
        if self.real_type == 'none':
            w_real = -self.inv_w_real
        elif self.real_type == 'exp':
            w_real = -torch.exp(self.inv_w_real)
        elif self.real_type == 'relu':
            w_real = -F.relu(self.inv_w_real)
        elif self.real_type == 'sigmoid':
            w_real = -F.sigmoid(self.inv_w_real)
        elif self.real_type == 'softplus':
            w_real = -F.softplus(self.inv_w_real)
        else:
            raise NotImplementedError
        w = w_real + 1.0j * self.w_imag
        return w

    def forward(self, state=None, rate=1.0, L=None):
        """
        state: (B, H, N) initial state
        rate: sampling rate factor
        L: target length

        returns:
        (C, H, L) convolution kernel (generally C=1)
        (B, H, L) output from initial state
        """
        if self.L.item() == 0 and self.l_max is not None and self.l_max > 0:
            self._setup_C(self.l_max)
        if L is None:
            L = round(self.L.item() / rate)
        continuous_L = round(rate * L)
        while continuous_L > self.L.item():
            self._setup_C(continuous_L)
        discrete_L = round(self.L.item() / rate)
        dt = torch.exp(self.log_dt) * rate
        B = _r2c(self.B)
        C = _r2c(self.C)
        P = _r2c(self.P)
        Q = P.conj()
        w = self._w()
        if self.bandlimit is not None:
            freqs = w.imag.abs() / (2 * math.pi)
            freqs = dt[:, None] / rate * freqs
            mask = torch.where(freqs < self.bandlimit * 0.5, 1, 0)
            C = C * mask
        omega, z = self._omega(discrete_L, dtype=w.dtype, device=w.device, cache=rate == 1.0)
        B = repeat(B, '1 t n -> 1 (v t) n', v=self.broadcast)
        P = repeat(P, 'r t n -> r (v t) n', v=self.broadcast)
        Q = repeat(Q, 'r t n -> r (v t) n', v=self.broadcast)
        w = repeat(w, 't n -> (v t) n', v=self.broadcast)
        if state is not None:
            s = _conj(state) if state.size(-1) == self.N else state
            sA = s * _conj(w) - contract('bhm, rhm, rhn -> bhn', s, _conj(Q), _conj(P))
            s = s / dt.unsqueeze(-1) + sA / 2
            s = s[..., :self.N]
            B = torch.cat([s, B], dim=-3)
        w = w * dt.unsqueeze(-1)
        B = torch.cat([B, P], dim=-3)
        C = torch.cat([C, Q], dim=-3)
        v = B.unsqueeze(-3) * C.unsqueeze(-4)
        if has_cauchy_extension and z.dtype == torch.cfloat and not self.keops:
            r = cauchy_mult(v, z, w, symmetric=True)
        elif has_pykeops:
            r = cauchy_conj(v, z, w)
        else:
            r = cauchy_naive(v, z, w)
        r = r * dt[None, None, :, None]
        if self.rank == 1:
            k_f = r[:-1, :-1, :, :] - r[:-1, -1:, :, :] * r[-1:, :-1, :, :] / (1 + r[-1:, -1:, :, :])
        elif self.rank == 2:
            r00 = r[:-self.rank, :-self.rank, :, :]
            r01 = r[:-self.rank, -self.rank:, :, :]
            r10 = r[-self.rank:, :-self.rank, :, :]
            r11 = r[-self.rank:, -self.rank:, :, :]
            det = (1 + r11[:1, :1, :, :]) * (1 + r11[1:, 1:, :, :]) - r11[:1, 1:, :, :] * r11[1:, :1, :, :]
            s = r01[:, :1, :, :] * (1 + r11[1:, 1:, :, :]) * r10[:1, :, :, :] + r01[:, 1:, :, :] * (1 + r11[:1, :1, :, :]) * r10[1:, :, :, :] - r01[:, :1, :, :] * r11[:1, 1:, :, :] * r10[1:, :, :, :] - r01[:, 1:, :, :] * r11[1:, :1, :, :] * r10[:1, :, :, :]
            s = s / det
            k_f = r00 - s
        else:
            r00 = r[:-self.rank, :-self.rank, :, :]
            r01 = r[:-self.rank, -self.rank:, :, :]
            r10 = r[-self.rank:, :-self.rank, :, :]
            r11 = r[-self.rank:, -self.rank:, :, :]
            r11 = rearrange(r11, 'a b h n -> h n a b')
            r11 = torch.linalg.inv(torch.eye(self.rank, device=r.device) + r11)
            r11 = rearrange(r11, 'h n a b -> a b h n')
            k_f = r00 - torch.einsum('i j h n, j k h n, k l h n -> i l h n', r01, r11, r10)
        k_f = k_f * 2 / (1 + omega)
        k = torch.fft.irfft(k_f, n=discrete_L)
        k = k[..., :L]
        if state is not None:
            k_state = k[:-1, :, :, :]
        else:
            k_state = None
        k_B = k[-1, :, :, :]
        return k_B, k_state

    @torch.no_grad()
    def _setup_linear(self):
        """ Create parameters that allow fast linear stepping of state """
        w = self._w()
        B = _r2c(self.B)
        P = _r2c(self.P)
        Q = P.conj()
        B = repeat(B, '1 t n -> 1 (v t) n', v=self.broadcast)
        P = repeat(P, 'r t n -> r (v t) n', v=self.broadcast)
        Q = repeat(Q, 'r t n -> r (v t) n', v=self.broadcast)
        w = repeat(w, 't n -> (v t) n', v=self.broadcast)
        dt = torch.exp(self.log_dt)
        D = (2.0 / dt.unsqueeze(-1) - w).reciprocal()
        R = torch.eye(self.rank, dtype=w.dtype, device=w.device) + 2 * contract('r h n, h n, s h n -> h r s', Q, D, P).real
        Q_D = rearrange(Q * D, 'r h n -> h r n')
        try:
            R = torch.linalg.solve(R, Q_D)
        except:
            R = torch.tensor(np.linalg.solve(R.to(Q_D).contiguous().detach().cpu(), Q_D.contiguous().detach().cpu()))
        R = rearrange(R, 'h r n -> r h n')
        self.step_params = {'D': D, 'R': R, 'P': P, 'Q': Q, 'B': B, 'E': 2.0 / dt.unsqueeze(-1) + w}

    def _step_state_linear(self, u=None, state=None):
        """
        Version of the step function that has time O(N) instead of O(N^2) per step, which takes advantage of the DPLR form and bilinear discretization.

        Unfortunately, as currently implemented it's about 2x slower because it calls several sequential operations. Perhaps a fused CUDA kernel implementation would be much faster

        u: (H) input
        state: (H, N/2) state with conjugate pairs
          Optionally, the state can have last dimension N
        Returns: same shape as state
        """
        C = _r2c(self.C)
        if u is None:
            u = torch.zeros(self.H, dtype=C.dtype, device=C.device)
        if state is None:
            state = torch.zeros(self.H, self.N, dtype=C.dtype, device=C.device)
        step_params = self.step_params.copy()
        if state.size(-1) == self.N:
            contract_fn = lambda p, x, y: contract('r h n, r h m, ... h m -> ... h n', _conj(p), _conj(x), _conj(y))[..., :self.N]
        else:
            assert state.size(-1) == 2 * self.N
            step_params = {k: _conj(v) for k, v in step_params.items()}
            contract_fn = lambda p, x, y: contract('r h n, r h m, ... h m -> ... h n', p, x, y)
        D = step_params['D']
        E = step_params['E']
        R = step_params['R']
        P = step_params['P']
        Q = step_params['Q']
        B = step_params['B']
        new_state = E * state - contract_fn(P, Q, state)
        new_state = new_state + 2.0 * B * u.unsqueeze(-1)
        new_state = D * (new_state - contract_fn(P, R, new_state))
        return new_state

    def _setup_state(self):
        """ Construct dA and dB for discretized state equation """
        self._setup_linear()
        C = _r2c(self.C)
        state = torch.eye(2 * self.N, dtype=C.dtype, device=C.device).unsqueeze(-2)
        dA = self._step_state_linear(state=state)
        dA = rearrange(dA, 'n h m -> h m n')
        u = C.new_ones(self.H)
        dB = self._step_state_linear(u=u)
        dB = _conj(dB)
        dB = rearrange(dB, '1 h n -> h n')
        return dA, dB

    def _step_state(self, u, state):
        """ Must be called after self.default_state() is used to construct an initial state!  """
        next_state = self.state_contraction(self.dA, state) + self.input_contraction(self.dB, u)
        return next_state

    def _setup_step(self, mode='dense'):
        """ Set up dA, dB, dC discretized parameters for stepping """
        self.dA, self.dB = self._setup_state()
        C = _conj(_r2c(self.C))
        if self.L.item() == 0:
            dC = C
        else:
            dA_L = power(self.L.item(), self.dA)
            I = torch.eye(self.dA.size(-1))
            dC = torch.linalg.solve(I - dA_L.transpose(-1, -2), C.unsqueeze(-1)).squeeze(-1)
        self.dC = dC
        self._step_mode = mode
        if mode == 'linear':
            self.dC = 2 * self.dC[:, :, :self.N]
        elif mode == 'diagonal':
            L, V = torch.linalg.eig(self.dA)
            V_inv = torch.linalg.inv(V)
            if self.verbose:
                None
            self.dA = L
            self.dB = contract('h n m, h m -> h n', V_inv, self.dB)
            self.dC = contract('h n m, c h n -> c h m', V, self.dC)
        elif mode == 'dense':
            pass
        else:
            raise NotImplementedError("NPLR Kernel step mode must be {'dense' | 'linear' | 'diagonal'}")

    def default_state(self, *batch_shape):
        C = _r2c(self.C)
        N = C.size(-1)
        H = C.size(-2)
        step_mode = getattr(self, '_step_mode', 'dense')
        if step_mode != 'linear':
            N *= 2
            if step_mode == 'diagonal':
                self.state_contraction = contract_expression('h n, ... h n -> ... h n', (H, N), batch_shape + (H, N))
            else:
                self.state_contraction = contract_expression('h m n, ... h n -> ... h m', (H, N, N), batch_shape + (H, N))
            self.input_contraction = contract_expression('h n, ... h -> ... h n', (H, N), batch_shape + (H,))
        self.output_contraction = contract_expression('c h n, ... h n -> ... c h', (C.shape[0], H, N), batch_shape + (H, N))
        state = torch.zeros(*batch_shape, H, N, dtype=C.dtype, device=C.device)
        return state

    def step(self, u, state):
        """ Must have called self._setup_step() and created state with self.default_state() before calling this """
        if self._step_mode == 'linear':
            new_state = self._step_state_linear(u, state)
        else:
            new_state = self._step_state(u, state)
        y = self.output_contraction(self.dC, new_state)
        return y.real, new_state


class SSKernelDiag(OptimModule):
    """Version using (complex) diagonal state matrix (S4D)"""

    def __init__(self, A, B, C, log_dt, L=None, disc='bilinear', real_type='exp', lr=None, bandlimit=None):
        super().__init__()
        self.L = L
        self.disc = disc
        self.bandlimit = bandlimit
        self.real_type = real_type
        assert A.size(-1) == C.size(-1)
        self.H = log_dt.size(-1)
        self.N = A.size(-1)
        assert A.size(-2) == B.size(-2)
        assert self.H % A.size(-2) == 0
        self.n_ssm = A.size(-2)
        self.repeat = self.H // A.size(0)
        self.channels = C.shape[0]
        self.C = nn.Parameter(_c2r(_resolve_conj(C)))
        if lr is None or isinstance(lr, float):
            lr_dict = {}
        else:
            lr_dict, lr = lr, None
        self.register('log_dt', log_dt, lr_dict.get('dt', lr))
        self.register('A', _c2r(A), lr_dict.get('A', lr))
        self.register('B', _c2r(B), lr_dict.get('B', lr))
        self.register('inv_A_real', self._A_init(A.real), lr_dict.get('A', lr))
        self.register('A_imag', A.imag, lr_dict.get('A', lr))

    def _A_init(self, A_real):
        A_real = torch.clamp(A_real, max=-0.0001)
        if self.real_type == 'none':
            return -A_real
        elif self.real_type == 'exp':
            return torch.log(-A_real)
        elif self.real_type == 'relu':
            return -A_real
        elif self.real_type == 'sigmoid':
            return torch.logit(-A_real)
        elif self.real_type == 'softplus':
            return torch.log(torch.exp(-A_real) - 1)
        else:
            raise NotImplementedError

    def _A(self):
        if self.real_type == 'none':
            A_real = -self.inv_A_real
        elif self.real_type == 'exp':
            A_real = -torch.exp(self.inv_A_real)
        elif self.real_type == 'relu':
            A_real = -F.relu(self.inv_A_real) - 0.0001
        elif self.real_type == 'sigmoid':
            A_real = -F.sigmoid(self.inv_A_real)
        elif self.real_type == 'softplus':
            A_real = -F.softplus(self.inv_A_real)
        else:
            raise NotImplementedError
        A = A_real + 1.0j * self.A_imag
        return A

    def forward(self, L, state=None, rate=1.0, u=None):
        """
        state: (B, H, N) initial state
        rate: sampling rate factor
        L: target length

        returns:
        (C, H, L) convolution kernel (generally C=1)
        (B, H, L) output from initial state
        """
        dt = torch.exp(self.log_dt) * rate
        C = _r2c(self.C)
        A = self._A()
        B = _r2c(self.B)
        B = repeat(B, 't n -> 1 (v t) n', v=self.repeat)
        if self.bandlimit is not None:
            freqs = dt[:, None] / rate * A.imag.abs() / (2 * math.pi)
            mask = torch.where(freqs < self.bandlimit * 0.5, 1, 0)
            C = C * mask
        A = repeat(A, 't n -> (v t) n', v=self.repeat)
        dtA = A * dt.unsqueeze(-1)
        if state is not None:
            s = state / dt.unsqueeze(-1)
            if self.disc == 'bilinear':
                s = s * (1.0 + dtA / 2)
            elif self.disc == 'zoh':
                s = s * dtA * dtA.exp() / (dtA.exp() - 1.0)
            B = torch.cat([s, B], dim=-3)
        C = (B[:, None, :, :] * C).view(-1, self.H, self.N)
        if self.disc == 'zoh':
            C = C * (torch.exp(dtA) - 1.0) / A
            K = log_vandermonde(C, dtA, L)
        elif self.disc == 'bilinear':
            C = C * (1.0 - dtA / 2).reciprocal() * dt.unsqueeze(-1)
            dA = (1.0 + dtA / 2) / (1.0 - dtA / 2)
            K = log_vandermonde(C, dA.log(), L)
        elif self.disc == 'dss':
            P = dtA.unsqueeze(-1) * torch.arange(L, device=C.device)
            A_gt_0 = A.real > 0
            if A_gt_0.any():
                with torch.no_grad():
                    P_max = dtA * (A_gt_0 * (L - 1))
                P = P - P_max.unsqueeze(-1)
            S = P.exp()
            dtA_neg = dtA * (1 - 2 * A_gt_0)
            num = dtA_neg.exp() - 1
            den = (dtA_neg * L).exp() - 1
            x = den * A
            x_conj = _resolve_conj(x)
            r = x_conj / (x * x_conj + 1e-07)
            C = C * num * r
            K = contract('chn,hnl->chl', C, S).float()
        else:
            assert False, f'{self.disc} not supported'
        K = K.view(-1, self.channels, self.H, L)
        if state is not None:
            K_state = K[:-1, :, :, :]
        else:
            K_state = None
        K = K[-1, :, :, :]
        return K, K_state

    def _setup_step(self):
        dt = torch.exp(self.log_dt)
        B = _r2c(self.B)
        C = _r2c(self.C)
        self.dC = C
        A = self._A()
        dtA = A * dt.unsqueeze(-1)
        if self.disc == 'zoh':
            self.dA = torch.exp(dtA)
            self.dB = B * (torch.exp(dtA) - 1.0) / A
        elif self.disc == 'bilinear':
            self.dA = (1.0 + dtA / 2) / (1.0 - dtA / 2)
            self.dB = B * (1.0 - dtA / 2).reciprocal() * dt.unsqueeze(-1)

    def default_state(self, *batch_shape):
        C = _r2c(self.C)
        state = torch.zeros(*batch_shape, self.H, self.N, dtype=C.dtype, device=C.device)
        return state

    def step(self, u, state):
        next_state = contract('h n, b h n -> b h n', self.dA, state) + contract('h n, b h -> b h n', self.dB, u)
        y = contract('c h n, b h n -> b c h', self.dC, next_state)
        return 2 * y.real, next_state

    def forward_state(self, u, state):
        self._setup_step()
        AL = self.dA ** u.size(-1)
        u = u.flip(-1).contiguous()
        v = log_vandermonde_transpose(u, self.dB, self.dA.log(), u.size(-1))
        next_state = AL * state + v
        return next_state


combinations = {'hippo': ['legs', 'fourier'], 'diag': ['diag-inv', 'diag-lin'], 'all': ['legs', 'fourier', 'diag-inv', 'diag-lin']}


def rank_correction(measure, N, rank=1, dtype=torch.float):
    """ Return low-rank matrix L such that A + L is normal """
    if measure == 'legs':
        assert rank >= 1
        P = torch.sqrt(0.5 + torch.arange(N, dtype=dtype)).unsqueeze(0)
    elif measure == 'legt':
        assert rank >= 2
        P = torch.sqrt(1 + 2 * torch.arange(N, dtype=dtype))
        P0 = P.clone()
        P0[0::2] = 0.0
        P1 = P.clone()
        P1[1::2] = 0.0
        P = torch.stack([P0, P1], dim=0)
        P *= 2 ** -0.5
    elif measure in ['fourier', 'fout']:
        P = torch.zeros(N)
        P[0::2] = 2 ** 0.5
        P[0] = 1
        P = P.unsqueeze(0)
    elif measure in ['fourier_diag', 'foud', 'legsd']:
        P = torch.zeros(1, N, dtype=dtype)
    else:
        raise NotImplementedError
    d = P.size(0)
    if rank > d:
        P = torch.cat([P, torch.zeros(rank - d, N, dtype=dtype)], dim=0)
    return P


def transition(measure, N):
    """ A, B transition matrices for different measures """
    if measure == 'legt':
        Q = np.arange(N, dtype=np.float64)
        R = (2 * Q + 1) ** 0.5
        j, i = np.meshgrid(Q, Q)
        A = R[:, None] * np.where(i < j, (-1.0) ** (i - j), 1) * R[None, :]
        B = R[:, None]
        A = -A
        A *= 0.5
        B *= 0.5
    elif measure == 'legs':
        q = np.arange(N, dtype=np.float64)
        col, row = np.meshgrid(q, q)
        r = 2 * q + 1
        M = -(np.where(row >= col, r, 0) - np.diag(q))
        T = np.sqrt(np.diag(2 * q + 1))
        A = T @ M @ np.linalg.inv(T)
        B = np.diag(T)[:, None]
        B = B.copy()
    elif measure == 'legsd':
        q = np.arange(N, dtype=np.float64)
        col, row = np.meshgrid(q, q)
        r = 2 * q + 1
        M = -(np.where(row >= col, r, 0) - np.diag(q))
        T = np.sqrt(np.diag(2 * q + 1))
        A = T @ M @ np.linalg.inv(T)
        B = np.diag(T)[:, None]
        B = B.copy()
        A += 0.5 * B * B[None, :, 0]
        B = B / 2.0
    elif measure in ['fourier_diag', 'foud']:
        freqs = np.arange(N // 2)
        d = np.stack([freqs, np.zeros(N // 2)], axis=-1).reshape(-1)[:-1]
        A = 2 * np.pi * (-np.diag(d, 1) + np.diag(d, -1))
        A = A - 0.5 * np.eye(N)
        B = np.zeros(N)
        B[0::2] = 2 ** 0.5
        B[0] = 1
        B = B[:, None]
    elif measure in ['fourier', 'fout']:
        freqs = np.arange(N // 2)
        d = np.stack([np.zeros(N // 2), freqs], axis=-1).reshape(-1)[1:]
        A = np.pi * (-np.diag(d, 1) + np.diag(d, -1))
        B = np.zeros(N)
        B[0::2] = 2 ** 0.5
        B[0] = 1
        A = A - B[:, None] * B[None, :]
        B = B[:, None]
    else:
        raise NotImplementedError
    return A, B


def nplr(measure, N, rank=1, dtype=torch.float, diagonalize_precision=True):
    """ Return w, p, q, V, B such that
    (w - p q^*, B) is unitarily equivalent to the original HiPPO A, B by the matrix V
    i.e. A = V[w - p q^*]V^*, B = V B
    """
    assert dtype == torch.float or torch.double
    cdtype = torch.cfloat if dtype == torch.float else torch.cdouble
    A, B = transition(measure, N)
    A = torch.as_tensor(A, dtype=dtype)
    B = torch.as_tensor(B, dtype=dtype)[:, 0]
    P = rank_correction(measure, N, rank=rank, dtype=dtype)
    AP = A + torch.sum(P.unsqueeze(-2) * P.unsqueeze(-1), dim=-3)
    _A = AP + AP.transpose(-1, -2)
    if (err := torch.sum((_A - _A[0, 0] * torch.eye(N)) ** 2) / N) > 1e-05:
        None
    w_re = torch.mean(torch.diagonal(AP), -1, keepdim=True)
    if diagonalize_precision:
        AP = AP
    w_im, V = torch.linalg.eigh(AP * -1.0j)
    if diagonalize_precision:
        w_im, V = w_im, V
    w = w_re + 1.0j * w_im
    _, idx = torch.sort(w.imag)
    w_sorted = w[idx]
    V_sorted = V[:, idx]
    V = V_sorted[:, :N // 2]
    w = w_sorted[:N // 2]
    assert w[-2].abs() > 0.0001, 'Only 1 zero eigenvalue allowed in diagonal part of A'
    if w[-1].abs() < 0.0001:
        V[:, -1] = 0.0
        V[0, -1] = 2 ** -0.5
        V[1, -1] = 2 ** -0.5 * 1.0j
    _AP = V @ torch.diag_embed(w) @ V.conj().transpose(-1, -2)
    if (err := torch.sum((2 * _AP.real - AP) ** 2) / N) > 1e-05:
        None
    V_inv = V.conj().transpose(-1, -2)
    B = contract('ij, j -> i', V_inv, B)
    P = contract('ij, ...j -> ...i', V_inv, P)
    return w, P, B, V


def dplr(scaling, N, rank=1, H=1, dtype=torch.float, real_scale=1.0, imag_scale=1.0, random_real=False, random_imag=False, normalize=False, diagonal=True, random_B=False):
    assert dtype == torch.float or torch.double
    dtype = torch.cfloat if dtype == torch.float else torch.cdouble
    pi = torch.tensor(math.pi)
    if random_real:
        real_part = torch.rand(H, N // 2)
    else:
        real_part = 0.5 * torch.ones(H, N // 2)
    if random_imag:
        imag_part = N // 2 * torch.rand(H, N // 2)
    else:
        imag_part = repeat(torch.arange(N // 2), 'n -> h n', h=H)
    real_part = real_scale * real_part
    if scaling == 'random':
        imag_part = torch.randn(H, N // 2)
    elif scaling == 'real':
        imag_part = 0 * imag_part
        real_part = 1 + repeat(torch.arange(N // 2), 'n -> h n', h=H)
    elif scaling in ['linear', 'lin']:
        imag_part = pi * imag_part
    elif scaling in ['inverse', 'inv']:
        imag_part = 1 / pi * N * (N / (1 + 2 * imag_part) - 1)
    elif scaling in ['inverse2', 'inv2']:
        imag_part = 1 / pi * N * (N / (1 + imag_part) - 1)
    elif scaling in ['quadratic', 'quad']:
        imag_part = 1 / pi * (1 + 2 * imag_part) ** 2
    elif scaling in ['legs', 'hippo']:
        w, _, _, _ = nplr('legsd', N)
        imag_part = w.imag
    else:
        raise NotImplementedError
    imag_part = imag_scale * imag_part
    w = -real_part + 1.0j * imag_part
    if random_B:
        B = torch.randn(H, N // 2, dtype=dtype)
    else:
        B = torch.ones(H, N // 2, dtype=dtype)
    if normalize:
        norm = -B / w
        zeta = 2 * torch.sum(torch.abs(norm) ** 2, dim=-1, keepdim=True)
        B = B / zeta ** 0.5
    P = torch.randn(rank, H, N // 2, dtype=dtype)
    if diagonal:
        P = P * 0.0
    V = torch.eye(N, dtype=dtype)[::N // 2]
    V = repeat(V, 'n m -> h n m', h=H)
    return w, P, B, V


def ssm(measure, N, R, H, **ssm_args):
    """Dispatcher to create single SSM initialization

    N: state size
    R: rank (for DPLR parameterization)
    H: number of independent SSM copies
    """
    if measure == 'dplr':
        w, P, B, V = dplr(N=N, rank=R, H=H, **ssm_args)
    elif measure.startswith('diag'):
        args = measure.split('-')
        assert args[0] == 'diag' and len(args) > 1
        scaling = args[1]
        w, P, B, V = dplr(scaling=scaling, N=N, rank=R, H=H, diagonal=True, **ssm_args)
    else:
        w, P, B, V = nplr(measure, N, R, **ssm_args)
        w = repeat(w, 'n -> s n', s=H)
        P = repeat(P, 'r n -> r s n', s=H)
        B = repeat(B, 'n -> s n', s=H)
        V = repeat(V, 'n m -> s n m', s=H)
    return w, P, B, V


def combination(measures, N, R, S, **ssm_args):
    if isinstance(measures, str):
        measures = combinations[measures] if measures in combinations else [measures]
    assert S % len(measures) == 0, f'{S} independent trainable SSM copies must be multiple of {len(measures)} different measures'
    w, P, B, V = zip(*[ssm(measure, N, R, S // len(measures), **ssm_args) for measure in measures])
    w = torch.cat(w, dim=0)
    P = torch.cat(P, dim=1)
    B = torch.cat(B, dim=0)
    V = torch.cat(V, dim=0)
    return w, P, B, V


class SSKernel(nn.Module):
    """Wrapper around SSKernel parameterizations.

    The SSKernel is expected to support the interface
    forward()
    default_state()
    _setup_step()
    step()
    """

    def __init__(self, H, N=64, L=None, measure='legs', rank=1, channels=1, dt_min=0.001, dt_max=0.1, deterministic=False, lr=None, mode='nplr', n_ssm=None, verbose=False, measure_args={}, **kernel_args):
        """State Space Kernel which computes the convolution kernel $\\bar{K}$

        H: Number of independent SSM copies; controls the size of the model. Also called d_model in the config.
        N: State size (dimensionality of parameters A, B, C). Also called d_state in the config. Generally shouldn't need to be adjusted and doens't affect speed much.
        L: Maximum length of convolution kernel, if known. Should work in the majority of cases even if not known.
        measure: Options for initialization of (A, B). For NPLR mode, recommendations are "legs", "fout", "hippo" (combination of both). For Diag mode, recommendations are "diag-inv", "diag-lin", "diag-legs", and "diag" (combination of diag-inv and diag-lin)
        rank: Rank of low-rank correction for NPLR mode. Needs to be increased for measure "legt"
        channels: C channels turns the SSM from a 1-dim to C-dim map; can think of it having C separate "heads" per SSM. This was partly a feature to make it easier to implement bidirectionality; it is recommended to set channels=1 and adjust H to control parameters instead
        dt_min, dt_max: min and max values for the step size dt (\\Delta)
        mode: Which kernel algorithm to use. 'nplr' is the full S4 model; 'diag' is the simpler S4D; 'slow' is a dense version for testing
        n_ssm: Number of independent trainable (A, B) SSMs, e.g. n_ssm=1 means all A/B parameters are tied across the H different instantiations of C. n_ssm=None means all H SSMs are completely independent. Generally, changing this option can save parameters but doesn't affect performance or speed much. This parameter must divide H
        lr: Passing in a number (e.g. 0.001) sets attributes of SSM parameers (A, B, dt). A custom optimizer hook is needed to configure the optimizer to set the learning rates appropriately for these parameters.
        """
        super().__init__()
        self.N = N
        self.H = H
        dtype, cdtype = torch.float, torch.cfloat
        self.channels = channels
        self.n_ssm = n_ssm if n_ssm is not None else H
        self.mode = mode
        self.verbose = verbose
        self.kernel_args = kernel_args
        if deterministic:
            log_dt = torch.exp(torch.linspace(math.log(dt_min), math.log(dt_max), H))
        else:
            log_dt = torch.rand(self.H, dtype=dtype) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        w, P, B, V = combination(measure, self.N, rank, self.n_ssm, **measure_args)
        if deterministic:
            C = torch.zeros(channels, self.H, self.N, dtype=cdtype)
            C[:, :, :1] = 1.0
            C = contract('hmn, chn -> chm', V.conj().transpose(-1, -2), C)
        else:
            C = torch.randn(channels, self.H, self.N // 2, dtype=cdtype)
        assert self.n_ssm % B.size(-2) == 0 and self.n_ssm % P.size(-2) == 0 and self.n_ssm % w.size(-2) == 0
        B = repeat(B, 't n -> (v t) n', v=self.n_ssm // B.size(-2)).clone().contiguous()
        P = repeat(P, 'r t n -> r (v t) n', v=self.n_ssm // P.size(-2)).clone().contiguous()
        w = repeat(w, 't n -> (v t) n', v=self.n_ssm // w.size(-2)).clone().contiguous()
        C = C.contiguous()
        if mode == 'nplr':
            self.kernel = SSKernelNPLR(w, P, B, C, log_dt, L=L, lr=lr, verbose=verbose, **kernel_args)
        elif mode == 'diag':
            if not measure.startswith('diag'):
                logger.info("Diagonal kernel (S4D) activated but initialization is not intended for S4D. Set `measure` to 'diag-lin', 'diag-inv', or 'diag-legs' for the main variants, or 'diag' for a combination of S4D-Lin and S4D-Inv.")
            C = C * repeat(B, 't n -> (v t) n', v=H // self.n_ssm)
            self.kernel = SSKernelDiag(w, B, C, log_dt, L=L, lr=lr, **kernel_args)
        else:
            raise NotImplementedError(f'mode={mode!r} is not valid')

    def forward(self, state=None, L=None, rate=None):
        return self.kernel(state=state, L=L, rate=rate)

    @torch.no_grad()
    def forward_state(self, u, state):
        """ Forward the state through a sequence, i.e. computes the state after passing chunk through SSM

        state: (B, H, N)
        u: (B, H, L)

        Returns: (B, H, N)
        """
        if hasattr(self.kernel, 'forward_state'):
            return self.kernel.forward_state(u, state)
        dA, dB = self.kernel._setup_state()
        conj = state.size(-1) != dA.size(-1)
        if conj:
            state = _conj(state)
        v = contract('h n, b h l -> b h n l', dB, u.flip(-1))
        AL, v = power(u.size(-1), dA, v)
        next_state = contract('h m n, b h n -> b h m', AL, state)
        next_state = next_state + v
        if conj:
            next_state = next_state[..., :next_state.size(-1) // 2]
        return next_state

    def _setup_step(self, **kwargs):
        self.kernel._setup_step(**kwargs)

    def step(self, u, state, **kwargs):
        y, state = self.kernel.step(u, state, **kwargs)
        return y, state

    def default_state(self, *args, **kwargs):
        return self.kernel.default_state(*args, **kwargs)


def Activation(activation=None, dim=-1):
    if activation in [None, 'id', 'identity', 'linear']:
        return nn.Identity()
    elif activation == 'tanh':
        return nn.Tanh()
    elif activation == 'relu':
        return nn.ReLU()
    elif activation == 'gelu':
        return nn.GELU()
    elif activation in ['swish', 'silu']:
        return nn.SiLU()
    elif activation == 'glu':
        return nn.GLU(dim=dim)
    elif activation == 'sigmoid':
        return nn.Sigmoid()
    else:
        raise NotImplementedError("hidden activation '{}' is not implemented".format(activation))


def LinearActivation(d_input, d_output, bias=True, transposed=False, activation=None, activate=False, **kwargs):
    """ Returns a linear nn.Module with control over axes order, initialization, and activation """
    linear_cls = partial(nn.Conv1d, kernel_size=1) if transposed else nn.Linear
    if activation == 'glu':
        d_output *= 2
    linear = linear_cls(d_input, d_output, bias=bias, **kwargs)
    if activate and activation is not None:
        activation = Activation(activation, dim=-2 if transposed else -1)
        linear = nn.Sequential(linear, activation)
    return linear


class S4(nn.Module):

    def __init__(self, d_model, d_state=64, l_max=None, channels=1, bidirectional=False, activation='gelu', postact='glu', hyper_act=None, dropout=0.0, tie_dropout=False, bottleneck=None, gate=None, transposed=True, verbose=False, use_pointwise=False, **kernel_args):
        """
        d_state: the dimension of the state, also denoted by N
        l_max: the maximum kernel length, also denoted by L. Set l_max=None to always use a global kernel
        channels: can be interpreted as a number of "heads"; the SSM is a map from a 1-dim to C-dim sequence. It's not recommended to change this unless desperate for things to tune; instead, increase d_model for larger models
        bidirectional: if True, convolution kernel will be two-sided

        Position-wise feedforward components:
        --------------------
        activation: activation in between SS and FF
        postact: activation after FF
        hyper_act: use a "hypernetwork" multiplication (experimental)
        dropout: standard dropout argument. tie_dropout=True ties the dropout mask across the sequence length, emulating nn.Dropout1d

        Other arguments:
        --------------------
        transposed: choose backbone axis ordering of (B, L, H) (if False) or (B, H, L) (if True) [B=batch size, L=sequence length, H=hidden dimension]
        gate: add gated activation (GSS)
        bottleneck: reduce SSM dimension (GSS)

        See the class SSKernel for the kernel constructor which accepts kernel_args. Relevant options that are worth considering and tuning include "mode" + "measure", "dt_min", "dt_max", "lr"

        Other options are all experimental and should not need to be configured
        """
        super().__init__()
        if verbose:
            logger.info(f'Constructing S4 (H, N, L) = ({d_model}, {d_state}, {l_max})')
        self.d_model = d_model
        self.H = d_model
        self.N = d_state
        self.L = l_max
        self.bidirectional = bidirectional
        self.channels = channels
        self.transposed = transposed
        self.gate = gate
        self.bottleneck = bottleneck
        if bottleneck is not None:
            self.H = self.H // bottleneck
            self.input_linear = LinearActivation(self.d_model, self.H, transposed=self.transposed, activation=activation, activate=True)
        if gate is not None:
            self.input_gate = LinearActivation(self.d_model, self.d_model * gate, transposed=self.transposed, activation=activation, activate=True)
            self.output_gate = LinearActivation(self.d_model * gate, self.d_model, transposed=self.transposed, activation=None, activate=False)
        self.hyper = hyper_act is not None
        if self.hyper:
            channels *= 2
            self.hyper_activation = Activation(hyper_act)
        self.D = nn.Parameter(torch.randn(channels, self.H))
        if self.bidirectional:
            channels *= 2
        self.kernel = SSKernel(self.H, N=self.N, L=self.L, channels=channels, verbose=verbose, **kernel_args)
        self.use_pointwise = use_pointwise
        if self.use_pointwise:
            self.activation = Activation(activation)
            dropout_fn = DropoutNd if tie_dropout else nn.Dropout
            self.dropout = dropout_fn(dropout) if dropout > 0.0 else nn.Identity()
            self.output_linear = LinearActivation(self.H * self.channels, self.d_model * (1 if self.gate is None else self.gate), transposed=self.transposed, activation=postact, activate=True)

    def forward(self, u, state=None, rate=1.0, lengths=None, **kwargs):
        """
        u: (B H L) if self.transposed else (B L H)
        state: (H N) never needed unless you know what you're doing

        Returns: same shape as u
        """
        if not self.transposed:
            u = u.transpose(-1, -2)
        L = u.size(-1)
        if isinstance(lengths, int):
            if lengths != L:
                lengths = torch.tensor(lengths, dtype=torch.long, device=u.device)
            else:
                lengths = None
        if lengths is not None:
            assert isinstance(lengths, torch.Tensor) and lengths.ndim == 1 and lengths.size(0) in [1, u.size(0)]
            mask = torch.where(torch.arange(L, device=lengths.device) < lengths[:, None, None], 1.0, 0.0)
            u = u * mask
        if self.gate is not None:
            v = self.input_gate(u)
        if self.bottleneck is not None:
            u = self.input_linear(u)
        L_kernel = L if self.L is None else min(L, round(self.L / rate))
        k, k_state = self.kernel(L=L_kernel, rate=rate, state=state)
        if self.bidirectional:
            k0, k1 = rearrange(k, '(s c) h l -> s c h l', s=2)
            k = F.pad(k0, (0, L)) + F.pad(k1.flip(-1), (L, 0))
        k_f = torch.fft.rfft(k, n=L_kernel + L)
        u_f = torch.fft.rfft(u, n=L_kernel + L)
        y_f = contract('bhl,chl->bchl', u_f, k_f)
        y = torch.fft.irfft(y_f, n=L_kernel + L)[..., :L]
        y = y + contract('bhl,ch->bchl', u, self.D)
        if state is not None:
            assert not self.bidirectional, 'Bidirectional not supported with state forwarding'
            y = y + k_state
            next_state = self.kernel.forward_state(u, state)
        else:
            next_state = None
        if self.hyper:
            y, yh = rearrange(y, 'b (s c) h l -> s b c h l', s=2)
            y = self.hyper_activation(yh) * y
        y = rearrange(y, '... c h l -> ... (c h) l')
        if self.use_pointwise:
            y = self.dropout(self.activation(y))
        if not self.transposed:
            y = y.transpose(-1, -2)
        if self.use_pointwise:
            y = self.output_linear(y)
        if self.gate is not None:
            y = self.output_gate(y * v)
        return y

    def setup_step(self, **kwargs):
        self.kernel._setup_step(**kwargs)

    def step(self, u, state):
        """ Step one time step as a recurrent model. Intended to be used during validation.

        u: (B H)
        state: (B H N)
        Returns: output (B H), state (B H N)
        """
        assert not self.training
        y, next_state = self.kernel.step(u, state)
        y = y + u.unsqueeze(-2) * self.D
        y = rearrange(y, 'b c h -> b (c h)')
        y = self.activation(y)
        if self.transposed:
            y = self.output_linear(y.unsqueeze(-1)).squeeze(-1)
        else:
            y = self.output_linear(y)
        return y, next_state

    def default_state(self, *batch_shape, device=None):
        return self.kernel.default_state(*batch_shape)

    @property
    def d_output(self):
        return self.d_model


class S4Module(nn.Module):

    def __init__(self, cfg, is_decoder=False):
        super().__init__()
        self.use_residual_dropout = cfg.s4_use_residual_dropout
        if self.use_residual_dropout:
            self.dropout_module = nn.Dropout(cfg.dropout)
        embed_dim = cfg.d_model
        l_max = cfg.max_position_embeddings if getattr(cfg, 'max_seq_length') else cfg.max_position_embeddings
        s4_configs = {'d_model': embed_dim, 'd_state': cfg.s4_state_dim, 'l_max': l_max, 'channels': cfg.s4_channels, 'bidirectional': not is_decoder, 'verbose': True, 'dt_min': cfg.s4_dt_min, 'dt_max': cfg.s4_dt_max, 'lr': cfg.s4_lr, 'n_ssm': cfg.s4_n_ssm}
        self.s4 = S4(**s4_configs)

    def residual_connection(self, x, residual):
        return residual + x

    def forward(self, x, padding_mask):
        with torch.no_grad():
            x = x.transpose(0, 1)
            residual = x if self.use_residual_dropout else None
            padding_mask = padding_mask.squeeze(1).squeeze(1) == -10000
            if padding_mask is not None:
                padding_mask = padding_mask.transpose(0, 1).unsqueeze(-1)
                x = x.masked_fill(padding_mask, 0.0)
            x = x.permute(1, 2, 0)
            dtype = x.dtype
            device = x.device.type
            with torch.autocast(enabled=False, device_type=device):
                self.s4
                x = self.s4(x.float())
                x = x
            x = x.permute(0, 2, 1)
            if self.use_residual_dropout:
                x = self.dropout_module(x)
                x = self.residual_connection(x, residual)
            x = x
            return x


class Classifier(nn.Module):

    def __init__(self, x_size, y_size, opt, prefix='decoder', dropout=None):
        super(Classifier, self).__init__()
        self.opt = opt
        if dropout is None:
            self.dropout = DropoutWrapper(opt.get('{}_dropout_p'.format(prefix), 0))
        else:
            self.dropout = dropout
        self.merge_opt = opt.get('{}_merge_opt'.format(prefix), 0)
        self.weight_norm_on = opt.get('{}_weight_norm_on'.format(prefix), False)
        if self.merge_opt == 1:
            self.proj = nn.Linear(x_size * 4, y_size)
        else:
            self.proj = nn.Linear(x_size * 2, y_size)
        if self.weight_norm_on:
            self.proj = weight_norm(self.proj)

    def forward(self, x1, x2, mask=None):
        if self.merge_opt == 1:
            x = torch.cat([x1, x2, (x1 - x2).abs(), x1 * x2], 1)
        else:
            x = torch.cat([x1, x2], 1)
        x = self.dropout(x)
        scores = self.proj(x)
        return scores


class BilinearFlatSim(nn.Module):
    """A bilinear attention layer over a sequence X w.r.t y:
    * o_i = x_i'Wy for x_i in X.
    """

    def __init__(self, x_size, y_size, opt={}, prefix='seqatt', dropout=None):
        super(BilinearFlatSim, self).__init__()
        self.opt = opt
        self.weight_norm_on = opt.get('{}_weight_norm_on'.format(prefix), False)
        self.linear = nn.Linear(y_size, x_size)
        if self.weight_norm_on:
            self.linear = weight_norm(self.linear)
        if dropout is None:
            self.dropout = DropoutWrapper(opt.get('{}_dropout_p'.format(self.prefix), 0))
        else:
            self.dropout = dropout

    def forward(self, x, y, x_mask):
        """
        x = batch * len * h1
        y = batch * h2
        x_mask = batch * len
        """
        x = self.dropout(x)
        y = self.dropout(y)
        Wy = self.linear(y)
        xWy = x.bmm(Wy.unsqueeze(2)).squeeze(2)
        xWy.data.masked_fill_(x_mask.data, -float('inf'))
        return xWy


class FlatSim(nn.Module):

    def __init__(self, x_size, y_size, opt={}, prefix='seqatt', dropout=None):
        super(FlatSim, self).__init__()
        assert x_size == y_size
        self.opt = opt
        self.weight_norm_on = opt.get('{}_weight_norm_on'.format(prefix), False)
        self.linear = nn.Linear(x_size * 3, 1)
        if self.weight_norm_on:
            self.linear = weight_norm(self.linear)
        if dropout is None:
            self.dropout = DropoutWrapper(opt.get('{}_dropout_p'.format(self.prefix), 0))
        else:
            self.dropout = dropout

    def forward(self, x, y, x_mask):
        """
        x = batch * len * h1
        y = batch * h2
        x_mask = batch * len
        """
        x = self.dropout(x)
        y = self.dropout(y)
        y = y.unsqueeze(1).expand_as(x)
        flat_x = torch.cat([x, y, x * y], 2).contiguous().view(x.size(0) * x.size(1), -1)
        flat_scores = self.linear(flat_x)
        scores = flat_scores.contiguous().view(x.size(0), -1)
        scores.data.masked_fill_(x_mask.data, -float('inf'))
        return scores


class FlatSimV2(nn.Module):

    def __init__(self, x_size, y_size, opt={}, prefix='seqatt', dropout=None):
        super(FlatSimV2, self).__init__()
        assert x_size == y_size
        self.opt = opt
        self.weight_norm_on = opt.get('{}_weight_norm_on'.format(prefix), False)
        self.linear = nn.Linear(x_size * 4, 1)
        if self.weight_norm_on:
            self.linear = weight_norm(self.linear)
        if dropout is None:
            self.dropout = DropoutWrapper(opt.get('{}_dropout_p'.format(self.prefix), 0))
        else:
            self.dropout = dropout

    def forward(self, x, y, x_mask):
        """
        x = batch * len * h1
        y = batch * h2
        x_mask = batch * len
        """
        x = self.dropout(x)
        y = self.dropout(y)
        y = y.unsqueeze(1).expand_as(x)
        flat_x = torch.cat([x, y, x * y, torch.abs(x - y)], 2).contiguous().view(x.size(0) * x.size(1), -1)
        flat_scores = self.linear(flat_x)
        scores = flat_scores.contiguous().view(x.size(0), -1)
        scores.data.masked_fill_(x_mask.data, -float('inf'))
        return scores


class SimpleFlatSim(nn.Module):

    def __init__(self, x_size, y_size, opt={}, prefix='seqatt', dropout=None):
        super(SimpleFlatSim, self).__init__()
        self.opt = opt
        self.weight_norm_on = opt.get('{}_norm_on'.format(prefix), False)
        self.linear = nn.Linear(y_size + x_size, 1)
        if self.weight_norm_on:
            self.linear = weight_norm(self.linear)
        if dropout is None:
            self.dropout = DropoutWrapper(opt.get('{}_dropout_p'.format(self.prefix), 0))
        else:
            self.dropout = dropout

    def forward(self, x, y, x_mask):
        """
        x = batch * len * h1
        y = batch * h2
        x_mask = batch * len
        """
        x = self.dropout(x)
        y = self.dropout(y)
        y = y.unsqueeze(1).expand_as(x)
        flat_x = torch.cat([x, y], 2).contiguous().view(x.size(0) * x.size(1), -1)
        flat_scores = self.linear(flat_x)
        scores = flat_scores.contiguous().view(x.size(0), -1)
        scores.data.masked_fill_(x_mask.data, -float('inf'))
        return scores


class FlatSimilarityWrapper(nn.Module):

    def __init__(self, x1_dim, x2_dim, prefix='attention', opt={}, dropout=None):
        super(FlatSimilarityWrapper, self).__init__()
        self.score_func_str = opt.get('{}_att_type'.format(prefix), 'none').lower()
        self.att_dropout = DropoutWrapper(opt.get('{}_att_dropout'.format(prefix), 0))
        self.score_func = None
        if self.score_func_str == 'bilinear':
            self.score_func = BilinearFlatSim(x1_dim, x2_dim, prefix=prefix, opt=opt, dropout=dropout)
        elif self.score_func_str == 'simple':
            self.score_func = SimpleFlatSim(x1_dim, x2_dim, prefix=prefix, opt=opt, dropout=dropout)
        elif self.score_func_str == 'flatsim':
            self.score_func = FlatSim(x1_dim, x2_dim, prefix=prefix, opt=opt, dropout=dropout)
        else:
            self.score_func = FlatSimV2(x1_dim, x2_dim, prefix=prefix, opt=opt, dropout=dropout)

    def forward(self, x1, x2, mask):
        scores = self.score_func(x1, x2, mask)
        return scores


class LinearSelfAttn(nn.Module):
    """Self attention over a sequence:
    * o_i = softmax(Wx_i) for x_i in X.
    """

    def __init__(self, input_size, dropout=None):
        super(LinearSelfAttn, self).__init__()
        self.linear = nn.Linear(input_size, 1)
        self.dropout = dropout

    def forward(self, x, x_mask):
        x = self.dropout(x)
        x_flat = x.contiguous().view(-1, x.size(-1))
        scores = self.linear(x_flat).view(x.size(0), x.size(1))
        scores.data.masked_fill_(x_mask.data, -float('inf'))
        alpha = F.softmax(scores, 1)
        return alpha.unsqueeze(1).bmm(x).squeeze(1)


class MLPSelfAttn(nn.Module):

    def __init__(self, input_size, opt={}, prefix='attn_sum', dropout=None):
        super(MLPSelfAttn, self).__init__()
        self.prefix = prefix
        self.FC = nn.Linear(input_size, input_size)
        self.linear = nn.Linear(input_size, 1)
        self.layer_norm_on = opt.get('{}_norm_on'.format(self.prefix), False)
        self.f = activation(opt.get('{}_activation'.format(self.prefix), 'relu'))
        if dropout is None:
            self.dropout = DropoutWrapper(opt.get('{}_dropout_p'.format(self.prefix), 0))
        else:
            self.dropout = dropout
        if self.layer_norm_on:
            self.FC = weight_norm(self.FC)

    def forward(self, x, x_mask):
        x = self.dropout(x)
        x_flat = x.contiguous().view(-1, x.size(-1))
        scores = self.linear(self.f(self.FC(x_flat))).view(x.size(0), x.size(1))
        scores.data.masked_fill_(x_mask.data, -float('inf'))
        alpha = F.softmax(scores)
        return alpha.unsqueeze(1).bmm(x).squeeze(1)


class SelfAttnWrapper(nn.Module):

    def __init__(self, input_size, prefix='attn_sum', opt={}, dropout=None):
        super(SelfAttnWrapper, self).__init__()
        """
        Self att wrapper, support linear and MLP
        """
        attn_type = opt.get('{}_type'.format(prefix), 'linear')
        if attn_type == 'mlp':
            self.att = MLPSelfAttn(input_size, prefix, opt, dropout)
        else:
            self.att = LinearSelfAttn(input_size, dropout)

    def forward(self, x, x_mask):
        return self.att(x, x_mask)


def generate_mask(new_data, dropout_p=0.0, is_training=False):
    if not is_training:
        dropout_p = 0.0
    new_data = (1 - dropout_p) * (new_data.zero_() + 1)
    for i in range(new_data.size(0)):
        one = random.randint(0, new_data.size(1) - 1)
        new_data[i][one] = 1
    mask = 1.0 / (1 - dropout_p) * torch.bernoulli(new_data)
    mask.requires_grad = False
    return mask


class SANClassifier(nn.Module):
    """Implementation of Stochastic Answer Networks for Natural Language Inference, Xiaodong Liu, Kevin Duh and Jianfeng Gao
    https://arxiv.org/abs/1804.07888
    """

    def __init__(self, x_size, h_size, label_size, opt={}, prefix='decoder', dropout=None):
        super(SANClassifier, self).__init__()
        if dropout is None:
            self.dropout = DropoutWrapper(opt.get('{}_dropout_p'.format(self.prefix), 0))
        else:
            self.dropout = dropout
        self.prefix = prefix
        self.query_wsum = SelfAttnWrapper(x_size, prefix='mem_cum', opt=opt, dropout=self.dropout)
        self.attn = FlatSimilarityWrapper(x_size, h_size, prefix, opt, self.dropout)
        self.rnn_type = '{}{}'.format(opt.get('{}_rnn_type'.format(prefix), 'gru').upper(), 'Cell')
        self.rnn = getattr(nn, self.rnn_type)(x_size, h_size)
        self.num_turn = opt.get('{}_num_turn'.format(prefix), 5)
        self.opt = opt
        self.mem_random_drop = opt.get('{}_mem_drop_p'.format(prefix), 0)
        self.mem_type = opt.get('{}_mem_type'.format(prefix), 0)
        self.weight_norm_on = opt.get('{}_weight_norm_on'.format(prefix), False)
        self.label_size = label_size
        self.dump_state = opt.get('dump_state_on', False)
        self.alpha = Parameter(torch.zeros(1, 1), requires_grad=False)
        if self.weight_norm_on:
            self.rnn = WN(self.rnn)
        self.classifier = Classifier(x_size, self.label_size, opt, prefix=prefix, dropout=self.dropout)

    def forward(self, x, h0, x_mask=None, h_mask=None):
        h0 = self.query_wsum(h0, h_mask)
        if type(self.rnn) is nn.LSTMCell:
            c0 = h0.new(h0.size()).zero_()
        scores_list = []
        for turn in range(self.num_turn):
            att_scores = self.attn(x, h0, x_mask)
            x_sum = torch.bmm(F.softmax(att_scores, 1).unsqueeze(1), x).squeeze(1)
            scores = self.classifier(x_sum, h0)
            scores_list.append(scores)
            if self.rnn is not None:
                h0 = self.dropout(h0)
                if type(self.rnn) is nn.LSTMCell:
                    h0, c0 = self.rnn(x_sum, (h0, c0))
                else:
                    h0 = self.rnn(x_sum, h0)
        if self.mem_type == 1:
            mask = generate_mask(self.alpha.data.new(x.size(0), self.num_turn), self.mem_random_drop, self.training)
            mask = [m.contiguous() for m in torch.unbind(mask, 1)]
            tmp_scores_list = [(mask[idx].view(x.size(0), 1).expand_as(inp) * F.softmax(inp, 1)) for idx, inp in enumerate(scores_list)]
            scores = torch.stack(tmp_scores_list, 2)
            scores = torch.mean(scores, 2)
            scores = torch.log(scores)
        else:
            scores = scores_list[-1]
        if self.dump_state:
            return scores, scores_list
        else:
            return scores


class MaskLmHeader(nn.Module):
    """Mask LM"""

    def __init__(self, embedding_weights=None, bias=False):
        super(MaskLmHeader, self).__init__()
        self.decoder = nn.Linear(embedding_weights.size(1), embedding_weights.size(0), bias=bias)
        self.decoder.weight = embedding_weights
        self.nsp = nn.Linear(embedding_weights.size(1), 2)

    def forward(self, hidden_states):
        mlm_out = self.decoder(hidden_states)
        nsp_out = self.nsp(hidden_states[:, 0, :])
        return mlm_out, nsp_out


class SanLayer(nn.Module):

    def __init__(self, num_hid, bidirect, dropout, rnn_type):
        super().__init__()
        assert isinstance(rnn_type, str)
        rnn_type = rnn_type.upper()
        assert rnn_type == 'LSTM' or rnn_type == 'GRU'
        rnn_cls = getattr(nn, rnn_type)
        self._rnn = rnn_cls(num_hid, num_hid, 1, bidirectional=bidirect, dropout=dropout, batch_first=True)
        self._layer_norm = nn.LayerNorm(num_hid, eps=1e-12)
        self.rnn_type = rnn_type
        self.num_hid = num_hid
        self.ndirections = 1 + int(bidirect)

    def init_hidden(self, batch):
        weight = next(self.parameters()).data
        hid_shape = self.ndirections, batch, self.num_hid
        if self.rnn_type == 'LSTM':
            return weight.new(*hid_shape).zero_(), weight.new(*hid_shape).zero_()
        else:
            return weight.new(*hid_shape).zero_()

    def forward(self, x, attention_mask):
        self._rnn.flatten_parameters()
        batch = x.size(0)
        hidden0 = self.init_hidden(batch)
        tmp_output = self._rnn(x, hidden0)[0]
        if self.ndirections > 1:
            size = tmp_output.shape
            tmp_output = tmp_output.view(size[0], size[1], self.num_hid, 2).max(-1)[0]
        output = self._layer_norm(x + tmp_output)
        return output


class SanEncoder(nn.Module):

    def __init__(self, num_hid, nlayers, bidirect, dropout, rnn_type='LSTM'):
        super().__init__()
        layer = SanLayer(num_hid, bidirect, dropout, rnn_type)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(nlayers)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers


class SanPooler(nn.Module):

    def __init__(self, hidden_size, dropout_p):
        super().__init__()
        my_dropout = DropoutWrapper(dropout_p, False)
        self.self_att = SelfAttnWrapper(hidden_size, dropout=my_dropout)
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states, attention_mask):
        """
        Arguments:
            hidden_states {FloatTensor} -- shape (batch, seq_len, hidden_size)
            attention_mask {ByteTensor} -- 1 indicates padded token
        """
        first_token_tensor = self.self_att(hidden_states, attention_mask)
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class SanModel(nn.Module):

    def __init__(self, config: 'BertConfig'):
        super().__init__()
        self.embeddings = BertEmbeddings(config)
        self.encoder = SanEncoder(config.hidden_size, config.num_hidden_layers, True, config.hidden_dropout_prob)
        self.pooler = SanPooler(config.hidden_size, config.hidden_dropout_prob)
        self.config = config

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, output_all_encoded_layers=True):
        """[summary]
        Arguments:
            input_ids {LongTensor} -- shape [batch_size, seq_len]
        Keyword Arguments:
            token_type_ids {LongTensor} -- shape [batch_size, seq_len]
            attention_mask {LongTensor} -- 0 indicates padding tokens
        Returns: Tuple of (sequence_output, pooled_output)
        """
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        embedding_output = self.embeddings(input_ids, token_type_ids)
        encoded_layers = self.encoder(embedding_output, attention_mask, output_all_encoded_layers=output_all_encoded_layers)
        sequence_output = encoded_layers[-1]
        pooled_output = self.pooler(sequence_output, attention_mask == 0)
        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]
        return encoded_layers, pooled_output


class DotProduct(nn.Module):

    def __init__(self, x1_dim, x2_dim, prefix='sim', opt={}, dropout=None):
        super(DotProduct, self).__init__()
        assert x1_dim == x2_dim
        self.opt = opt
        self.prefix = prefix
        self.scale_on = opt.get('{}_scale'.format(self.prefix), False)
        self.scalor = 1.0 / numpy.power(x2_dim, 0.5)

    def forward(self, x1, x2):
        assert x1.size(2) == x2.size(2)
        scores = x1.bmm(x2.transpose(1, 2))
        if self.scale_on:
            scores *= self.scalor
        return scores


class DotProductProject(nn.Module):

    def __init__(self, x1_dim, x2_dim, prefix='sim', opt={}, dropout=None):
        super(DotProductProject, self).__init__()
        self.prefix = prefix
        self.opt = opt
        self.hidden_size = opt.get('{}_hidden_size'.format(self.prefix), 64)
        self.residual_on = opt.get('{}_residual_on'.format(self.prefix), False)
        self.layer_norm_on = opt.get('{}_norm_on'.format(self.prefix), False)
        self.share = opt.get('{}_share'.format(self.prefix), False)
        self.f = activation(opt.get('{}_activation'.format(self.prefix), 'relu'))
        self.scale_on = opt.get('{}_scale_on'.format(self.prefix), False)
        self.dropout = dropout
        x1_in_dim = x1_dim
        x2_in_dim = x2_dim
        out_dim = self.hidden_size
        self.proj_1 = nn.Linear(x1_in_dim, out_dim, bias=False)
        if self.layer_norm_on:
            self.proj_1 = weight_norm(self.proj_1)
        if self.share and x1_in_dim == x2_in_dim:
            self.proj_2 = self.proj_1
        else:
            self.proj_2 = nn.Linear(x2_in_dim, out_dim)
            if self.layer_norm_on:
                self.proj_2 = weight_norm(self.proj_2)
        if self.scale_on:
            self.scalar = Parameter(torch.ones(1, 1, 1) / self.hidden_size ** 0.5, requires_grad=False)
        else:
            self.sclalar = Parameter(torch.ones(1, 1, self.hidden_size), requires_grad=True)

    def forward(self, x1, x2):
        assert x1.size(2) == x2.size(2)
        if self.dropout:
            x1 = self.dropout(x1)
            x2 = self.dropout(x2)
        x1_flat = x1.contiguous().view(-1, x1.size(2))
        x2_flat = x2.contiguous().view(-1, x2.size(2))
        x1_o = self.f(self.proj_1(x1_flat)).view(x1.size(0), x1.size(1), -1)
        x2_o = self.f(self.proj_2(x2_flat)).view(x2.size(0), x2.size(1), -1)
        if self.scale_on:
            scalar = self.scalar.expand_as(x2_o)
            x2_o = scalar * x2_o
        scores = x1_o.bmm(x2_o.transpose(1, 2))
        return scores


class Bilinear(nn.Module):

    def __init__(self, x1_dim, x2_dim, prefix='sim', opt={}, dropout=None):
        super(Bilinear, self).__init__()
        self.opt = opt
        self.layer_norm_on = opt.get('{}_norm_on'.format(self.prefix), False)
        self.transform_on = opt.get('{}_proj_on'.format(self.prefix), False)
        self.dropout = dropout
        if self.transform_on:
            self.proj = nn.Linear(x1_dim, x2_dim)
            if self.layer_norm_on:
                self.proj = weight_norm(self.proj)

    def forward(self, x, y):
        """
        x = batch * len * h1
        y = batch * h2
        x_mask = batch * len
        """
        if self.dropout:
            x = self.dropout(x)
            y = self.dropout(y)
        proj = self.proj(y) if self.transform_on else y
        if self.dropout:
            proj = self.dropout(proj)
        scores = x.bmm(proj.unsqueeze(2)).squeeze(2)
        return scores


def init_wrapper(init='xavier_uniform'):
    return eval(init)


class BilinearSum(nn.Module):

    def __init__(self, x1_dim, x2_dim, prefix='sim', opt={}, dropout=None):
        super(BilinearSum, self).__init__()
        self.x_linear = nn.Linear(x1_dim, 1, bias=False)
        self.y_linear = nn.Linear(x2_dim, 1, bias=False)
        self.layer_norm_on = opt.get('{}_norm_on'.format(self.prefix), False)
        self.init = init_wrapper(opt.get('{}_init'.format(self.prefix), False))
        if self.layer_norm_on:
            self.x_linear = weight_norm(self.x_linear)
            self.y_linear = weight_norm(self.y_linear)
        self.init(self.x_linear.weight)
        self.init(self.y_linear.weight)
        self.dropout = dropout

    def forward(self, x1, x2):
        """
        x1: batch * len1 * input_size
        x2: batch * len2 * input_size
        score: batch * len1 * len2
        """
        if self.dropout:
            x1 = self.dropout(x1)
            x2 = self.dropout(x2)
        x1_logits = self.x_linear(x1.contiguous().view(-1, x1.size(-1))).view(x1.size(0), -1, 1)
        x2_logits = self.y_linear(x2.contiguous().view(-1, x2.size(-1))).view(x2.size(0), 1, -1)
        shape = x1.size(0), x1.size(1), x2.size()
        scores = x1_logits.expand_as(shape) + x2_logits.expand_as(shape)
        return scores


class Trilinear(nn.Module):
    """Function used in BiDAF"""

    def __init__(self, x1_dim, x2_dim, prefix='sim', opt={}, dropout=None):
        super(Trilinear, self).__init__()
        self.prefix = prefix
        self.x_linear = nn.Linear(x1_dim, 1, bias=False)
        self.x_dot_linear = nn.Linear(x1_dim, 1, bias=False)
        self.y_linear = nn.Linear(x2_dim, 1, bias=False)
        self.layer_norm_on = opt.get('{}_norm_on'.format(self.prefix), False)
        self.init = init_wrapper(opt.get('{}_init'.format(self.prefix), 'xavier_uniform'))
        if self.layer_norm_on:
            self.x_linear = weight_norm(self.x_linear)
            self.x_dot_linear = weight_norm(self.x_dot_linear)
            self.y_linear = weight_norm(self.y_linear)
        self.init(self.x_linear.weight)
        self.init(self.x_dot_linear.weight)
        self.init(self.y_linear.weight)
        self.dropout = dropout

    def forward(self, x1, x2):
        """
        x1: batch * len1 * input_size
        x2: batch * len2 * input_size
        score: batch * len1 * len2
        """
        if self.dropout:
            x1 = self.dropout(x1)
            x2 = self.dropout(x2)
        x1_logits = self.x_linear(x1.contiguous().view(-1, x1.size(-1))).view(x1.size(0), -1, 1)
        x2_logits = self.y_linear(x2.contiguous().view(-1, x2.size(-1))).view(x2.size(0), 1, -1)
        x1_dot = self.x_dot_linear(x1.contiguous().view(-1, x1.size(-1))).view(x1.size(0), -1, 1).expand_as(x1)
        x1_dot = x1 * x1_dot
        scores = x1_dot.bmm(x2.transpose(1, 2))
        scores += x1_logits.expand_as(scores) + x2_logits.expand_as(scores)
        return scores


class SimilarityWrapper(nn.Module):

    def __init__(self, x1_dim, x2_dim, prefix='attention', opt={}, dropout=None):
        super(SimilarityWrapper, self).__init__()
        self.score_func_str = opt.get('{}_sim_func'.format(prefix), 'dotproductproject').lower()
        self.score_func = None
        if self.score_func_str == 'dotproduct':
            self.score_func = DotProduct(x1_dim, x2_dim, prefix=prefix, opt=opt, dropout=dropout)
        elif self.score_func_str == 'dotproductproject':
            self.score_func = DotProductProject(x1_dim, x2_dim, prefix=prefix, opt=opt, dropout=dropout)
        elif self.score_func_str == 'bilinear':
            self.score_func = Bilinear(x1_dim, x2_dim, prefix=prefix, opt=opt, dropout=dropout)
        elif self.score_func_str == 'bilinearsum':
            self.score_func = BilinearSum(x1_dim, x2_dim, prefix=prefix, opt=opt, dropout=dropout)
        elif self.score_func_str == 'trilinear':
            self.score_func = Trilinear(x1_dim, x2_dim, prefix=prefix, opt=opt, dropout=dropout)
        else:
            raise NotImplementedError

    def forward(self, x1, x2):
        scores = self.score_func(x1, x2)
        return scores


class AttentionWrapper(nn.Module):

    def __init__(self, x1_dim, x2_dim, x3_dim=None, prefix='attention', opt={}, dropout=None):
        super(AttentionWrapper, self).__init__()
        self.prefix = prefix
        self.att_dropout = opt.get('{}_att_dropout'.format(self.prefix), 0)
        self.score_func = SimilarityWrapper(x1_dim, x2_dim, prefix=prefix, opt=opt, dropout=dropout)
        self.drop_diagonal = opt.get('{}_drop_diagonal'.format(self.prefix), False)
        self.output_size = x2_dim if x3_dim is None else x3_dim

    def forward(self, query, key, value, key_padding_mask=None, return_scores=False):
        logits = self.score_func(query, key)
        key_mask = key_padding_mask.unsqueeze(1).expand_as(logits)
        logits.data.masked_fill_(key_mask.data, -float('inf'))
        if self.drop_diagonal:
            assert logits.size(1) == logits.size(2)
            diag_mask = torch.diag(logits.data.new(logits.size(1)).zero_() + 1).byte().unsqueeze(0).expand_as(logits)
            logits.data.masked_fill_(diag_mask, -float('inf'))
        prob = F.softmax(logits.view(-1, key.size(1)), 1)
        prob = prob.view(-1, query.size(1), key.size(1))
        if self.att_dropout > 0:
            prob = self.dropout(prob)
        if value is None:
            value = key
        attn = prob.bmm(value)
        if return_scores:
            return attn, prob, logits
        else:
            return attn


class MultiheadAttentionWrapper(nn.Module):
    """Multi-headed attention.
    See "Attention Is All You Need" for more details.
    """

    def __init__(self, query_dim, key_dim, value_dim, prefix='attention', opt={}, dropout=None):
        super().__init__()
        self.prefix = prefix
        self.num_heads = opt.get('{}_head'.format(self.prefix), 1)
        self.dropout = DropoutWrapper(opt.get('{}_dropout'.format(self.prefix), 0)) if dropout is None else dropout
        self.qkv_dim = [query_dim, key_dim, value_dim]
        assert query_dim == key_dim, 'query dim must equal with key dim'
        self.hidden_size = opt.get('{}_hidden_size'.format(self.prefix), 64)
        self.proj_on = opt.get('{}_proj_on'.format(prefix), False)
        self.share = opt.get('{}_share'.format(self.prefix), False)
        self.layer_norm_on = opt.get('{}_norm_on'.format(self.prefix), False)
        self.scale_on = opt.get('{}_scale_on'.format(self.prefix), False)
        if self.proj_on:
            self.proj_modules = nn.ModuleList([nn.Linear(dim, self.hidden_size) for dim in self.qkv_dim[0:2]])
            if self.layer_norm_on:
                for proj in self.proj_modules:
                    proj = weight_norm(proj)
            if self.share and self.qkv_dim[0] == self.qkv_dim[1]:
                self.proj_modules[1] = self.proj_modules[0]
            self.f = activation(opt.get('{}_activation'.format(self.prefix), 'relu'))
            self.qkv_head_dim = [self.hidden_size // self.num_heads] * 3
            self.qkv_head_dim[2] = value_dim // self.num_heads
            assert self.qkv_head_dim[0] * self.num_heads == self.hidden_size, 'hidden size must be divisible by num_heads'
            assert self.qkv_head_dim[2] * self.num_heads == value_dim, 'value size must be divisible by num_heads'
        else:
            self.qkv_head_dim = [(emb // self.num_heads) for emb in self.qkv_dim]
            assert self.qkv_head_dim[0] * self.num_heads == self.qkv_dim[0], 'query size must be divisible by num_heads'
            assert self.qkv_head_dim[1] * self.num_heads == self.qkv_dim[1], 'key size must be divisible by num_heads'
            assert self.qkv_head_dim[2] * self.num_heads == self.qkv_dim[2], 'value size must be divisible by num_heads'
        if self.scale_on:
            self.scaling = self.qkv_head_dim[0] ** -0.5
        self.drop_diagonal = opt.get('{}_drop_diagonal'.format(self.prefix), False)
        self.output_size = self.qkv_dim[2]

    def forward(self, query, key, value, key_padding_mask=None):
        query = query.transpose(0, 1)
        key = key.transpose(0, 1)
        value = value.transpose(0, 1)
        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.qkv_dim[0]
        q, k, v = query, key, value
        if self.proj_on:
            if self.dropout:
                q, k = self.dropout(q), self.dropout(k)
            q, k = [self.f(proj(input)) for input, proj in zip([query, key], self.proj_modules)]
        src_len = k.size(0)
        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len
        if self.scale_on:
            q *= self.scaling
        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.qkv_head_dim[0]).transpose(0, 1)
        k = k.contiguous().view(src_len, bsz * self.num_heads, self.qkv_head_dim[1]).transpose(0, 1)
        v = v.contiguous().view(src_len, bsz * self.num_heads, self.qkv_head_dim[2]).transpose(0, 1)
        attn_weights = torch.bmm(q, k.transpose(1, 2))
        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]
        if key_padding_mask is not None:
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.float().masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf')).type_as(attn_weights)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        if self.drop_diagonal:
            assert attn_weights.size(1) == attn_weights.size(2)
            diag_mask = torch.diag(attn_weights.data.new(attn_weights.size(1)).zero_() + 1).byte().unsqueeze(0).expand_as(attn_weights)
            attn_weights.data.masked_fill_(diag_mask, -float('inf'))
        attn_weights = F.softmax(attn_weights.float(), dim=-1).type_as(attn_weights)
        attn_weights = self.dropout(attn_weights)
        attn = torch.bmm(attn_weights, v)
        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.qkv_head_dim[2]]
        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, -1)
        attn = attn.transpose(0, 1)
        return attn


class DeepAttentionWrapper(nn.Module):

    def __init__(self, x1_dim, x2_dim, x3_dims, att_cnt, prefix='deep_att', opt=None, dropout=None):
        super(DeepAttentionWrapper, self).__init__()
        self.opt = {} if opt is None else opt
        self.prefix = prefix
        self.x1_dim = x1_dim
        self.x2_dim = x2_dim
        self.x3_dims = x3_dims
        if dropout is None:
            self.dropout = DropoutWrapper(opt.get('{}_dropout_p'.format(self.prefix), 0))
        else:
            self.dropout = dropout
        self.attn_list = nn.ModuleList()
        for i in range(0, att_cnt):
            if opt['multihead_on']:
                attention = MultiheadAttentionWrapper(self.x1_dim, self.x2_dim, self.x3_dims[i], prefix, opt, dropout=dropout)
            else:
                attention = AttentionWrapper(self.x1_dim, self.x2_dim, self.x3_dims[i], prefix, opt, self.dropout)
            self.attn_list.append(attention)

    def forward(self, x1, x2, x3, x2_mask):
        rvl = []
        for i in range(0, len(x3)):
            hiddens = self.attn_list[i](x1, x2, x3[i], x2_mask)
            rvl.append(hiddens)
        return torch.cat(rvl, 2)


class LayerNorm(nn.Module):

    def __init__(self, hidden_size, eps=0.0001):
        super(LayerNorm, self).__init__()
        self.alpha = Parameter(torch.ones(1, 1, hidden_size))
        self.beta = Parameter(torch.zeros(1, 1, hidden_size))
        self.eps = eps

    def forward(self, x):
        """
        Args:
            :param x: batch * len * input_size

        Returns:
            normalized x
        """
        mu = torch.mean(x, 2, keepdim=True).expand_as(x)
        sigma = torch.std(x, 2, keepdim=True).expand_as(x)
        return (x - mu) / (sigma + self.eps) * self.alpha.expand_as(x) + self.beta.expand_as(x)


def make_positions(tensor, padding_idx: 'int', onnx_trace: 'bool'=False):
    """Replace non-padding symbols with their position numbers.
    Position numbers begin at padding_idx+1. Padding symbols are ignored.
    """
    mask = tensor.ne(padding_idx).int()
    return (torch.cumsum(mask, dim=1).type_as(mask) * mask).long() + padding_idx


class LearnedPositionalEmbedding(nn.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    Padding ids are ignored by either offsetting based on padding_idx
    or by setting padding_idx to None and ensuring that the appropriate
    position ids are passed to the forward function.
    """

    def __init__(self, num_embeddings: 'int', embedding_dim: 'int', padding_idx: 'int'):
        super().__init__(num_embeddings, embedding_dim, padding_idx)
        self.onnx_trace = False
        if self.padding_idx is not None:
            self.max_positions = self.num_embeddings - self.padding_idx - 1
        else:
            self.max_positions = self.num_embeddings

    def forward(self, input: 'Tensor', incremental_state: 'Optional[Dict[str, Dict[str, Optional[Tensor]]]]'=None, positions: 'Optional[Tensor]'=None, offset: 'Optional[int]'=0):
        """Input is expected to be of size [bsz x seqlen]."""
        assert positions is None or self.padding_idx is None, 'If positions is pre-computed then padding_idx should not be set.'
        if positions is None:
            if incremental_state is not None:
                positions = torch.zeros((1, 1), device=input.device, dtype=input.dtype).fill_(int(self.padding_idx + input.size(1)))
            else:
                positions = make_positions(input, self.padding_idx, onnx_trace=self.onnx_trace)
        if offset > 0 and positions.size(1) == 1:
            positions = positions + offset
        return nn.functional.embedding(positions, self.weight, self.padding_idx, self.max_norm, self.norm_type, self.scale_grad_by_freq, self.sparse)


class T5DenseActDense(nn.Module):

    def __init__(self, config: 'MSRT5Config'):
        super().__init__()
        have_bias = getattr(config, 'have_bias', False)
        self.wi = nn.Linear(config.d_model, config.d_ff, bias=have_bias)
        self.wo = nn.Linear(config.d_ff, config.d_model, bias=have_bias)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.act = ACT2FN[config.dense_act_fn]

    def forward(self, hidden_states):
        hidden_states = self.wi(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.wo(hidden_states)
        return hidden_states


class T5DenseGatedActDense(nn.Module):

    def __init__(self, config: 'MSRT5Config'):
        super().__init__()
        have_bias = getattr(config, 'have_bias', False)
        self.wi_0 = nn.Linear(config.d_model, config.d_ff, bias=have_bias)
        self.wi_1 = nn.Linear(config.d_model, config.d_ff, bias=have_bias)
        self.wo = nn.Linear(config.d_ff, config.d_model, bias=have_bias)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.act = ACT2FN[config.dense_act_fn]

    def forward(self, hidden_states):
        hidden_gelu = self.act(self.wi_0(hidden_states))
        hidden_linear = self.wi_1(hidden_states)
        hidden_states = hidden_gelu * hidden_linear
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.wo(hidden_states)
        return hidden_states


class T5LayerFF(nn.Module):

    def __init__(self, config: 'MSRT5Config'):
        super().__init__()
        if config.is_gated_act:
            self.DenseReluDense = T5DenseGatedActDense(config)
        else:
            self.DenseReluDense = T5DenseActDense(config)
        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.normalize_before = getattr(config, 'normalize_before', False)

    def forward(self, hidden_states):
        if self.normalize_before:
            forwarded_states = self.layer_norm(hidden_states)
        else:
            forwarded_states = hidden_states
        forwarded_states = self.DenseReluDense(forwarded_states)
        hidden_states = hidden_states + self.dropout(forwarded_states)
        if not self.normalize_before:
            hidden_states = self.layer_norm(hidden_states)
        return hidden_states


def _concatenate_3_blocks(x: 'torch.Tensor', block_dim: 'int', sequence_dim: 'int', pad_value: 'int'=0) ->torch.Tensor:
    """Concatenate three consecutive blocks for each input block for local attentiont.

    For more information, see: https://arxiv.org/pdf/2112.07916.pdf.
    """
    num_blocks = x.shape[block_dim]
    pad = [(0, 0)] * x.ndim
    pad[block_dim] = 1, 1
    pad = sum(pad[::-1], ())
    x = F.pad(x, pad=pad, mode='constant', value=pad_value)
    blocks_list: 'List[torch.Tensor]' = []
    for i in range(3):
        indices = [slice(0, None)] * x.ndim
        indices[block_dim] = slice(i, i + num_blocks)
        indices = tuple(indices)
        blocks_list.append(x[indices])
    return torch.cat(blocks_list, dim=sequence_dim)


def _pad_to_multiple(x: 'torch.Tensor', block_len: 'int', dim: 'int', pad_value: 'int'=0) ->torch.Tensor:
    """Pad a tensor so that a sequence length will be a multiple of `block_len`"""
    pad_len = -x.shape[dim] % block_len
    if not all(x.shape):
        new_shape = list(x.shape)
        new_shape[dim] += pad_len
        return torch.zeros(new_shape, dtype=x.dtype)
    pad = [(0, 0)] * x.ndim
    pad[dim] = 0, pad_len
    pad = sum(pad[::-1], ())
    x = F.pad(x, pad=pad, mode='constant', value=pad_value)
    return x


def _split_into_blocks(x: 'torch.Tensor', block_len: 'int', dim: 'int') ->torch.Tensor:
    """Split an input tensor into blocks of a given `block_len` along the given `dim`. If the dimension length
    is not a multiple of `block_len`, it will be padded first with selected `pad_value`.
    """
    if x.shape[dim] % block_len != 0:
        x = _pad_to_multiple(x, block_len, dim, pad_value=0)
    num_blocks = x.shape[dim] // block_len
    output_shape = x.shape[:dim] + (num_blocks, block_len) + x.shape[dim + 1:]
    if 0 in output_shape:
        return torch.empty(output_shape, dtype=x.dtype, device=x.device)
    return x.reshape(output_shape)


def make_3block_relative_position_ids(block_len: 'int') ->torch.Tensor:
    """Makes 3-blocked relative position ids for local attention."""
    position_ids = torch.arange(3 * block_len, dtype=torch.int32)
    center_position_ids = position_ids[block_len:-block_len]
    relative_position_ids = position_ids.unsqueeze(0) - center_position_ids.unsqueeze(1)
    return relative_position_ids


def _mask_local_attention_mask(local_attention_mask: 'torch.Tensor', block_len: 'int') ->torch.Tensor:
    """Mask local attention mask to enforce that tokens are not allowed to attend tokens farther than ``local_radius."""
    relative_position_ids = make_3block_relative_position_ids(block_len)
    locality_mask = torch.abs(relative_position_ids) < block_len
    locality_mask = locality_mask[None, None, :, :]
    locality_mask = locality_mask
    return torch.logical_and(local_attention_mask, locality_mask)


def get_local_attention_mask(attention_mask: 'torch.Tensor', block_len: 'int') ->torch.Tensor:
    """Prepare attention mask to be applied for a local attention."""
    _blocked_attention_mask = _split_into_blocks(attention_mask, block_len, dim=1)
    _3blocked_attention_mask = _concatenate_3_blocks(_blocked_attention_mask, block_dim=1, sequence_dim=2)
    _blocked_attention_mask = _blocked_attention_mask.unsqueeze(-1)
    _3blocked_attention_mask = _3blocked_attention_mask.unsqueeze(-2)
    local_attention_mask = torch.logical_and(_blocked_attention_mask, _3blocked_attention_mask)
    local_attention_mask = _mask_local_attention_mask(local_attention_mask, block_len)
    return local_attention_mask.unsqueeze(1)


class T5LocalAttention(nn.Module):

    def __init__(self, config: 'MSRLongT5Config', has_relative_attention_bias=False):
        super().__init__()
        self.is_decoder = config.is_decoder
        self.has_relative_attention_bias = has_relative_attention_bias
        self.relative_attention_num_buckets = config.relative_attention_num_buckets
        self.relative_attention_max_distance = config.relative_attention_max_distance
        self.d_model = config.d_model
        self.key_value_proj_dim = config.d_kv
        self.n_heads = config.num_heads
        self.dropout = config.dropout_rate
        self.inner_dim = self.n_heads * self.key_value_proj_dim
        have_bias = getattr(config, 'have_bias', False)
        self.q = nn.Linear(self.d_model, self.inner_dim, bias=have_bias)
        self.k = nn.Linear(self.d_model, self.inner_dim, bias=have_bias)
        self.v = nn.Linear(self.d_model, self.inner_dim, bias=have_bias)
        self.o = nn.Linear(self.inner_dim, self.d_model, bias=have_bias)
        self.local_radius = config.local_radius
        self.block_len = self.local_radius + 1
        if self.has_relative_attention_bias:
            self.relative_attention_bias = nn.Embedding(self.relative_attention_num_buckets, self.n_heads)
            self.relative_position_bucket_local = self._cache_relative_position_bucket_local(self.block_len)
        self.head_dim = self.d_model // self.n_heads
        self.scaling = self.head_dim ** -0.5
        self.pruned_heads = set()
        self.gradient_checkpointing = False

    def _cache_relative_position_bucket_local(self, block_length: 'int'):
        memory_position = torch.arange(3 * block_length, dtype=torch.long)
        context_position = memory_position[block_length:-block_length]
        relative_position = memory_position[None, :] - context_position[:, None]
        relative_position_bucket = self._relative_position_bucket(relative_position, bidirectional=not self.is_decoder, num_buckets=self.relative_attention_num_buckets, max_distance=self.relative_attention_max_distance)
        return relative_position_bucket

    @staticmethod
    def _relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
        """
        Adapted from Mesh Tensorflow:
        https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593

        Translate relative position to a bucket number for relative attention. The relative position is defined as
        memory_position - query_position, i.e. the distance in tokens from the attending position to the attended-to
        position. If bidirectional=False, then positive relative positions are invalid. We use smaller buckets for
        small absolute relative_position and larger buckets for larger absolute relative_positions. All relative
        positions >=max_distance map to the same bucket. All relative positions <=-max_distance map to the same bucket.
        This should allow for more graceful generalization to longer sequences than the model has been trained on

        Args:
            relative_position: an int32 Tensor
            bidirectional: a boolean - whether the attention is bidirectional
            num_buckets: an integer
            max_distance: an integer

        Returns:
            a Tensor with the same shape as relative_position, containing int32 values in the range [0, num_buckets)
        """
        relative_buckets = 0
        if bidirectional:
            num_buckets //= 2
            relative_buckets += (relative_position > 0) * num_buckets
            relative_position = torch.abs(relative_position)
        else:
            relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact
        relative_position_if_large = max_exact + torch.log(relative_position.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
        relative_position_if_large = torch.min(relative_position_if_large, torch.full_like(relative_position_if_large, num_buckets - 1))
        relative_buckets += torch.where(is_small, relative_position, relative_position_if_large)
        return relative_buckets

    def compute_bias_local_attention(self, device: 'torch.device'):
        """Compute binned relative position bias"""
        relative_position_bucket = self.relative_position_bucket_local
        values = self.relative_attention_bias(relative_position_bucket)
        values = values.permute([2, 0, 1]).unsqueeze(0).unsqueeze(0)
        return values

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(heads, self.n_heads, self.key_value_proj_dim, self.pruned_heads)
        self.q = prune_linear_layer(self.q, index)
        self.k = prune_linear_layer(self.k, index)
        self.v = prune_linear_layer(self.v, index)
        self.o = prune_linear_layer(self.o, index, dim=1)
        self.n_heads = self.n_heads - len(heads)
        self.inner_dim = self.key_value_proj_dim * self.n_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(self, hidden_states, mask=None, key_value_states=None, position_bias=None, past_key_value=None, layer_head_mask=None, query_length=None, use_cache=False, output_attentions=False, key_padding_mask=None):
        """
        Self-attention (if key_value_states is None) or attention over source sentence (provided by key_value_states).
        """
        batch_size, seq_length = hidden_states.shape[:2]

        def shape(states):
            """projection"""
            return states.view(batch_size, -1, self.n_heads, self.head_dim)

        def unshape(states):
            """reshape"""
            return states.contiguous().view(batch_size, -1, self.inner_dim)
        mask = mask.squeeze(1).squeeze(1)
        mask = get_local_attention_mask(mask != -10000, self.block_len)
        query_states = shape(self.q(hidden_states))
        key_states = shape(self.k(hidden_states))
        value_states = shape(self.v(hidden_states))
        query_states *= self.scaling
        query_states = _split_into_blocks(query_states, self.block_len, dim=1)
        key_states = _split_into_blocks(key_states, self.block_len, dim=1)
        value_states = _split_into_blocks(value_states, self.block_len, dim=1)
        key_states = _concatenate_3_blocks(key_states, block_dim=1, sequence_dim=2)
        value_states = _concatenate_3_blocks(value_states, block_dim=1, sequence_dim=2)
        scores = torch.einsum('...qhd,...khd->...hqk', query_states, key_states)
        if not self.has_relative_attention_bias:
            position_bias = torch.zeros((1, 1, self.num_heads, self.block_len, 3 * self.block_len), device=scores.device, dtype=scores.dtype)
        else:
            position_bias = self.compute_bias_local_attention(scores.device)
        if mask is not None:
            mask = torch.where(mask > 0, 0.0, -10000000000.0)
            position_bias = position_bias + mask.transpose(1, 2)
        scores += position_bias
        attn_weights = F.softmax(scores.float(), dim=-1).type_as(scores)
        attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        attn_weights = attn_weights.type(value_states.dtype)
        attn_output = unshape(torch.einsum('...hqk,...khd->...qhd', attn_weights, value_states))
        attn_output = attn_output[:, :seq_length, :]
        attn_output = self.o(attn_output)
        present_key_value_state = (key_states, value_states) if self.is_decoder and use_cache else None
        outputs = (attn_output,) + (present_key_value_state,) + (position_bias,)
        if output_attentions:
            outputs = outputs + (attn_weights,)
        return outputs


class T5LayerLocalSelfAttention(nn.Module):

    def __init__(self, config, has_relative_attention_bias=False):
        super().__init__()
        self.SelfAttention = T5LocalAttention(config, has_relative_attention_bias=has_relative_attention_bias)
        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.normalize_before = getattr(config, 'normalize_before', False)

    def forward(self, hidden_states, attention_mask=None, position_bias=None, layer_head_mask=None, past_key_value=None, use_cache=False, output_attentions=False, key_padding_mask=None):
        if self.normalize_before:
            normed_hidden_states = self.layer_norm(hidden_states)
        else:
            normed_hidden_states = hidden_states
        attention_output = self.SelfAttention(normed_hidden_states, mask=attention_mask, position_bias=position_bias, layer_head_mask=layer_head_mask, past_key_value=past_key_value, use_cache=use_cache, output_attentions=output_attentions, key_padding_mask=key_padding_mask)
        hidden_states = hidden_states + self.dropout(attention_output[0])
        if not self.normalize_before:
            hidden_states = self.layer_norm(hidden_states)
        outputs = (hidden_states,) + attention_output[1:]
        return outputs


class T5LayerLocalGlobalSelfAttention(nn.Module):

    def __init__(self, config, has_relative_attention_bias=False):
        super().__init__()
        self.SelfAttention = T5LocalAttention(config, has_relative_attention_bias=has_relative_attention_bias)
        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.normalize_before = getattr(config, 'normalize_before', False)
        self.norm_global = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.norm_local = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.aggregate = nn.Linear(2 * config.d_model, config.d_model)
        self.s4 = S4Module(config, is_decoder=False)

    def forward(self, hidden_states, attention_mask=None, position_bias=None, layer_head_mask=None, past_key_value=None, use_cache=False, output_attentions=False, key_padding_mask=None):
        if self.normalize_before:
            normed_hidden_states = self.layer_norm(hidden_states)
        else:
            normed_hidden_states = hidden_states
        global_hidden_states = self.norm_global(self.s4(normed_hidden_states, attention_mask))
        attention_output = self.SelfAttention(normed_hidden_states, mask=attention_mask, position_bias=position_bias, layer_head_mask=layer_head_mask, past_key_value=past_key_value, use_cache=use_cache, output_attentions=output_attentions, key_padding_mask=key_padding_mask)
        local_hidden_states = self.norm_local(attention_output[0])
        global_local_hidden_states = torch.concat((global_hidden_states, local_hidden_states), dim=-1)
        global_local_hidden_states = self.aggregate(global_local_hidden_states)
        hidden_states = hidden_states + self.dropout(global_local_hidden_states)
        if not self.normalize_before:
            hidden_states = self.layer_norm(hidden_states)
        outputs = (hidden_states,) + attention_output[1:]
        return outputs


class T5Attention(nn.Module):

    def __init__(self, config: 'MSRT5Config', has_relative_attention_bias=False):
        super().__init__()
        self.is_decoder = config.is_decoder
        self.has_relative_attention_bias = has_relative_attention_bias
        self.relative_attention_num_buckets = config.relative_attention_num_buckets
        self.relative_attention_max_distance = config.relative_attention_max_distance
        self.d_model = config.d_model
        self.key_value_proj_dim = config.d_kv
        self.n_heads = config.num_heads
        self.dropout = config.dropout_rate
        self.inner_dim = self.n_heads * self.key_value_proj_dim
        have_bias = getattr(config, 'have_bias', False)
        self.q = nn.Linear(self.d_model, self.inner_dim, bias=have_bias)
        self.k = nn.Linear(self.d_model, self.inner_dim, bias=have_bias)
        self.v = nn.Linear(self.d_model, self.inner_dim, bias=have_bias)
        self.o = nn.Linear(self.inner_dim, self.d_model, bias=have_bias)
        if self.has_relative_attention_bias:
            self.relative_attention_bias = nn.Embedding(self.relative_attention_num_buckets, self.n_heads)
        self.pruned_heads = set()
        self.gradient_checkpointing = False
        relative_position = torch.arange(config.max_position_embeddings, dtype=torch.long)[None, :] - torch.arange(config.max_position_embeddings, dtype=torch.long)[:, None]
        self.rp_bucket = self.relative_position_bucket(relative_position, num_buckets=self.relative_attention_num_buckets, max_distance=self.relative_attention_max_distance)
        self.rp_bucket -= self.rp_bucket.min()
        self.head_dim = self.d_model // self.n_heads
        self.scaling = self.head_dim ** -0.5

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(heads, self.n_heads, self.key_value_proj_dim, self.pruned_heads)
        self.q = prune_linear_layer(self.q, index)
        self.k = prune_linear_layer(self.k, index)
        self.v = prune_linear_layer(self.v, index)
        self.o = prune_linear_layer(self.o, index, dim=1)
        self.n_heads = self.n_heads - len(heads)
        self.inner_dim = self.key_value_proj_dim * self.n_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    @staticmethod
    def relative_position_bucket(relative_position, num_buckets=32, max_distance=128):
        sign = torch.sign(relative_position)
        num_buckets //= 2
        n = torch.abs(relative_position)
        max_exact = num_buckets // 2
        is_small = n < max_exact
        max_bucket_val = num_buckets - 1 - max_exact
        val_if_large = max_exact + torch.ceil(torch.log(n.float() / max_exact) / math.log((max_distance - 1) / max_exact) * max_bucket_val).long()
        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))
        ret = torch.where(is_small, n, val_if_large) * sign
        return ret

    def compute_bias(self, query_length, key_length, device=None):
        """Compute binned relative position bias"""
        if device is None:
            device = self.relative_attention_bias.weight.device
        relative_position_bucket = self.rp_bucket[:query_length, :key_length]
        relative_position_bucket = relative_position_bucket
        values = self.relative_attention_bias(relative_position_bucket)
        values = values.permute([2, 0, 1]).unsqueeze(0)
        return values

    def forward(self, hidden_states, mask=None, key_value_states=None, position_bias=None, past_key_value=None, layer_head_mask=None, query_length=None, use_cache=False, output_attentions=False, key_padding_mask=None):
        """
        Self-attention (if key_value_states is None) or attention over source sentence (provided by key_value_states).
        """
        batch_size, seq_length = hidden_states.shape[:2]
        real_seq_length = seq_length
        if past_key_value is not None:
            assert len(past_key_value) == 2, f'past_key_value should have 2 past states: keys and values. Got {len(past_key_value)} past states'
            real_seq_length += past_key_value[0].shape[2] if query_length is None else query_length
        key_length = real_seq_length if key_value_states is None else key_value_states.shape[1]

        def shape(states):
            """projection"""
            return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)

        def unshape(states):
            """reshape"""
            return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)

        def project(hidden_states, proj_layer, key_value_states, past_key_value):
            """projects hidden states correctly to key/query states"""
            if key_value_states is None:
                hidden_states = shape(proj_layer(hidden_states))
            elif past_key_value is None:
                hidden_states = shape(proj_layer(key_value_states))
            if past_key_value is not None:
                if key_value_states is None:
                    hidden_states = torch.cat([past_key_value, hidden_states], dim=2)
                else:
                    hidden_states = past_key_value
            return hidden_states
        query_states = shape(self.q(hidden_states)) * self.scaling
        key_states = project(hidden_states, self.k, key_value_states, past_key_value[0] if past_key_value is not None else None)
        value_states = project(hidden_states, self.v, key_value_states, past_key_value[1] if past_key_value is not None else None)
        scores = torch.matmul(query_states, key_states.transpose(3, 2))
        if position_bias is None:
            if not self.has_relative_attention_bias:
                position_bias = torch.zeros((1, self.n_heads, real_seq_length, key_length), device=scores.device, dtype=scores.dtype)
                if self.gradient_checkpointing and self.training:
                    position_bias.requires_grad = True
            else:
                position_bias = self.compute_bias(real_seq_length, key_length, device=scores.device)
            if past_key_value is not None:
                position_bias = position_bias[:, :, -hidden_states.size(1):, :]
            if mask is not None:
                position_bias = position_bias + mask
        scores += position_bias
        if key_padding_mask is not None:
            scores = scores.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))
        attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        if layer_head_mask is not None:
            attn_weights = attn_weights * layer_head_mask
        attn_output = unshape(torch.matmul(attn_weights, value_states))
        attn_output = self.o(attn_output)
        present_key_value_state = (key_states, value_states) if self.is_decoder and use_cache else None
        outputs = (attn_output,) + (present_key_value_state,) + (position_bias,)
        if output_attentions:
            outputs = outputs + (attn_weights,)
        return outputs


class T5LayerSelfAttention(nn.Module):

    def __init__(self, config, has_relative_attention_bias=False):
        super().__init__()
        self.SelfAttention = T5Attention(config, has_relative_attention_bias=has_relative_attention_bias)
        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.normalize_before = getattr(config, 'normalize_before', False)

    def forward(self, hidden_states, attention_mask=None, position_bias=None, layer_head_mask=None, past_key_value=None, use_cache=False, output_attentions=False, key_padding_mask=None):
        if self.normalize_before:
            normed_hidden_states = self.layer_norm(hidden_states)
        else:
            normed_hidden_states = hidden_states
        attention_output = self.SelfAttention(normed_hidden_states, mask=attention_mask, position_bias=position_bias, layer_head_mask=layer_head_mask, past_key_value=past_key_value, use_cache=use_cache, output_attentions=output_attentions, key_padding_mask=key_padding_mask)
        hidden_states = hidden_states + self.dropout(attention_output[0])
        if not self.normalize_before:
            hidden_states = self.layer_norm(hidden_states)
        outputs = (hidden_states,) + attention_output[1:]
        return outputs


class T5LayerCrossAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.EncDecAttention = T5Attention(config, has_relative_attention_bias=False)
        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.normalize_before = getattr(config, 'normalize_before', False)

    def forward(self, hidden_states, key_value_states, attention_mask=None, position_bias=None, layer_head_mask=None, past_key_value=None, use_cache=False, query_length=None, output_attentions=False):
        if self.normalize_before:
            normed_hidden_states = self.layer_norm(hidden_states)
        else:
            normed_hidden_states = hidden_states
        attention_output = self.EncDecAttention(normed_hidden_states, mask=attention_mask, key_value_states=key_value_states, position_bias=position_bias, layer_head_mask=layer_head_mask, past_key_value=past_key_value, use_cache=use_cache, query_length=query_length, output_attentions=output_attentions)
        layer_output = hidden_states + self.dropout(attention_output[0])
        if not self.normalize_before:
            layer_output = self.layer_norm(layer_output)
        outputs = (layer_output,) + attention_output[1:]
        return outputs


class T5Block(nn.Module):

    def __init__(self, config, has_relative_attention_bias=False):
        super().__init__()
        self.is_decoder = config.is_decoder
        self.layer = nn.ModuleList()
        self.layer.append(T5LayerSelfAttention(config, has_relative_attention_bias=has_relative_attention_bias))
        if self.is_decoder:
            self.layer.append(T5LayerCrossAttention(config))
        self.layer.append(T5LayerFF(config))

    def forward(self, hidden_states, attention_mask=None, position_bias=None, encoder_hidden_states=None, encoder_attention_mask=None, encoder_decoder_position_bias=None, layer_head_mask=None, cross_attn_layer_head_mask=None, past_key_value=None, use_cache=False, output_attentions=False, return_dict=True, key_padding_mask=None):
        if past_key_value is not None:
            if not self.is_decoder:
                logger.warning('`past_key_values` is passed to the encoder. Please make sure this is intended.')
            expected_num_past_key_values = 2 if encoder_hidden_states is None else 4
            if len(past_key_value) != expected_num_past_key_values:
                raise ValueError(f"There should be {expected_num_past_key_values} past states. {'2 (past / key) for cross attention. ' if expected_num_past_key_values == 4 else ''}Got {len(past_key_value)} past key / value states")
            self_attn_past_key_value = past_key_value[:2]
            cross_attn_past_key_value = past_key_value[2:]
        else:
            self_attn_past_key_value, cross_attn_past_key_value = None, None
        self_attention_outputs = self.layer[0](hidden_states, attention_mask=attention_mask, position_bias=position_bias, layer_head_mask=layer_head_mask, past_key_value=self_attn_past_key_value, use_cache=use_cache, output_attentions=output_attentions, key_padding_mask=key_padding_mask)
        hidden_states, present_key_value_state = self_attention_outputs[:2]
        attention_outputs = self_attention_outputs[2:]
        if hidden_states.dtype == torch.float16 and torch.isinf(hidden_states).any():
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        do_cross_attention = self.is_decoder and encoder_hidden_states is not None
        if do_cross_attention:
            if present_key_value_state is not None:
                query_length = present_key_value_state[0].shape[2]
            else:
                query_length = None
            cross_attention_outputs = self.layer[1](hidden_states, key_value_states=encoder_hidden_states, attention_mask=encoder_attention_mask, position_bias=encoder_decoder_position_bias, layer_head_mask=cross_attn_layer_head_mask, past_key_value=cross_attn_past_key_value, query_length=query_length, use_cache=use_cache, output_attentions=output_attentions)
            hidden_states = cross_attention_outputs[0]
            if hidden_states.dtype == torch.float16 and torch.isinf(hidden_states).any():
                clamp_value = torch.finfo(hidden_states.dtype).max - 1000
                hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
            if present_key_value_state is not None:
                present_key_value_state = present_key_value_state + cross_attention_outputs[1]
            attention_outputs = attention_outputs + cross_attention_outputs[2:]
        hidden_states = self.layer[-1](hidden_states)
        if hidden_states.dtype == torch.float16 and torch.isinf(hidden_states).any():
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        outputs = hidden_states,
        if use_cache:
            outputs = outputs + (present_key_value_state,) + attention_outputs
        else:
            outputs = outputs + attention_outputs
        return outputs


class RobertaLMHead(nn.Module):
    """Head for masked language modeling."""

    def __init__(self, embed_dim, output_dim, weight=None):
        super().__init__()
        self.dense = nn.Linear(embed_dim, embed_dim)
        self.activation_fn = torch.nn.functional.relu
        self.layer_norm = T5LayerNorm(embed_dim)
        if weight is None:
            weight = nn.Linear(embed_dim, output_dim, bias=False).weight
        self.weight = weight
        self.bias = nn.Parameter(torch.zeros(output_dim))

    def forward(self, features, masked_tokens=None, **kwargs):
        if masked_tokens is not None:
            features = features[masked_tokens, :]
        x = self.dense(features)
        x = self.activation_fn(x)
        x = self.layer_norm(x)
        x = torch.nn.functional.linear(x, self.weight) + self.bias
        return x


class Criterion(_Loss):

    def __init__(self, alpha=1.0, name='criterion'):
        super().__init__()
        """Alpha is used to weight each loss term
        """
        self.alpha = alpha
        self.name = name

    def forward(self, input, target, weight=None, ignore_index=-1):
        """weight: sample weight"""
        return


class CeCriterion(Criterion):

    def __init__(self, alpha=1.0, name='Cross Entropy Criterion'):
        super().__init__()
        self.alpha = alpha
        self.name = name

    def forward(self, input, target, weight=None, ignore_index=-1):
        """weight: sample weight"""
        if weight is not None:
            loss = torch.sum(F.cross_entropy(input, target, reduce=False, reduction='none', ignore_index=ignore_index) * weight)
        else:
            loss = F.cross_entropy(input, target, ignore_index=ignore_index)
        loss = loss * self.alpha
        return loss


class SeqCeCriterion(CeCriterion):

    def __init__(self, alpha=1.0, name='Seq Cross Entropy Criterion'):
        super().__init__(alpha, name)

    def forward(self, input, target, weight=None, ignore_index=-1):
        target = target.view(-1)
        if weight:
            loss = torch.mean(F.cross_entropy(input, target, reduce=False, ignore_index=ignore_index) * weight)
        else:
            loss = F.cross_entropy(input, target, ignore_index=ignore_index)
        loss = loss * self.alpha
        return loss


class MseCriterion(Criterion):

    def __init__(self, alpha=1.0, name='MSE Regression Criterion'):
        super().__init__()
        self.alpha = alpha
        self.name = name

    def forward(self, input, target, weight=None, ignore_index=-1):
        """weight: sample weight"""
        if weight:
            loss = torch.mean(F.mse_loss(input.squeeze(), target, reduce=False) * weight.reshape((target.shape[0], 1)))
        else:
            loss = F.mse_loss(input.squeeze(), target)
        loss = loss * self.alpha
        return loss


class KlCriterion(Criterion):

    def __init__(self, alpha=1.0, name='KL Div Criterion'):
        super().__init__()
        self.alpha = alpha
        self.name = name

    def forward(self, input, target, weight=None, ignore_index=-1):
        """input/target: logits"""
        input = input.float()
        target = target.float()
        loss = F.kl_div(F.log_softmax(input, dim=-1, dtype=torch.float32), F.softmax(target, dim=-1, dtype=torch.float32), reduction='batchmean')
        loss = loss * self.alpha
        return loss


def stable_kl(logit, target, epsilon=1e-06, reduce=True):
    logit = logit.view(-1, logit.size(-1)).float()
    target = target.view(-1, target.size(-1)).float()
    bs = logit.size(0)
    p = F.log_softmax(logit, 1).exp()
    y = F.log_softmax(target, 1).exp()
    rp = -(1.0 / (p + epsilon) - 1 + epsilon).detach().log()
    ry = -(1.0 / (y + epsilon) - 1 + epsilon).detach().log()
    if reduce:
        return (p * (rp - ry) * 2).sum() / bs
    else:
        return (p * (rp - ry) * 2).sum()


class NsKlCriterion(Criterion):

    def __init__(self, alpha=1.0, name='KL Div Criterion'):
        super().__init__()
        self.alpha = alpha
        self.name = name

    def forward(self, input, target, weight=None, ignore_index=-1):
        """input/target: logits"""
        input = input.float()
        target = target.float()
        loss = stable_kl(input, target.detach())
        loss = loss * self.alpha
        return loss


class SymKlCriterion(Criterion):

    def __init__(self, alpha=1.0, name='KL Div Criterion'):
        super().__init__()
        self.alpha = alpha
        self.name = name

    def forward(self, input, target, weight=None, ignore_index=-1, reduction='batchmean'):
        """input/target: logits"""
        input = input.float()
        target = target.float()
        loss = F.kl_div(F.log_softmax(input, dim=-1, dtype=torch.float32), F.softmax(target.detach(), dim=-1, dtype=torch.float32), reduction=reduction) + F.kl_div(F.log_softmax(target, dim=-1, dtype=torch.float32), F.softmax(input.detach(), dim=-1, dtype=torch.float32), reduction=reduction)
        loss = loss * self.alpha
        return loss


class NsSymKlCriterion(Criterion):

    def __init__(self, alpha=1.0, name='KL Div Criterion'):
        super().__init__()
        self.alpha = alpha
        self.name = name

    def forward(self, input, target, weight=None, ignore_index=-1):
        """input/target: logits"""
        input = input.float()
        target = target.float()
        loss = stable_kl(input, target.detach()) + stable_kl(target, input.detach())
        loss = loss * self.alpha
        return loss


class JSCriterion(Criterion):

    def __init__(self, alpha=1.0, name='JS Div Criterion'):
        super().__init__()
        self.alpha = alpha
        self.name = name

    def forward(self, input, target, weight=None, ignore_index=-1, reduction='batchmean'):
        """input/target: logits"""
        input = input.float()
        target = target.float()
        m = F.softmax(target.detach(), dim=-1, dtype=torch.float32) + F.softmax(input.detach(), dim=-1, dtype=torch.float32)
        m = 0.5 * m
        loss = F.kl_div(F.log_softmax(input, dim=-1, dtype=torch.float32), m, reduction=reduction) + F.kl_div(F.log_softmax(target, dim=-1, dtype=torch.float32), m, reduction=reduction)
        loss = loss * self.alpha
        return loss


class HLCriterion(Criterion):

    def __init__(self, alpha=1.0, name='Hellinger Criterion'):
        super().__init__()
        self.alpha = alpha
        self.name = name

    def forward(self, input, target, weight=None, ignore_index=-1, reduction='batchmean'):
        """input/target: logits"""
        input = input.float()
        target = target.float()
        si = F.softmax(target.detach(), dim=-1, dtype=torch.float32).sqrt_()
        st = F.softmax(input.detach(), dim=-1, dtype=torch.float32).sqrt_()
        loss = F.mse_loss(si, st)
        loss = loss * self.alpha
        return loss


class RankCeCriterion(Criterion):

    def __init__(self, alpha=1.0, name='Cross Entropy Criterion'):
        super().__init__()
        self.alpha = alpha
        self.name = name

    def forward(self, input, target, weight=None, ignore_index=-1, pairwise_size=1):
        input = input.view(-1, pairwise_size)
        target = target.contiguous().view(-1, pairwise_size)[:, 0]
        if weight:
            loss = torch.mean(F.cross_entropy(input, target, reduce=False, ignore_index=ignore_index) * weight)
        else:
            loss = F.cross_entropy(input, target, ignore_index=ignore_index)
        loss = loss * self.alpha
        return loss


class SpanCeCriterion(Criterion):

    def __init__(self, alpha=1.0, name='Span Cross Entropy Criterion'):
        super().__init__()
        """This is for extractive MRC, e.g., SQuAD, ReCoRD ... etc
        """
        self.alpha = alpha
        self.name = name

    def forward(self, input, target, weight=None, ignore_index=-1):
        """weight: sample weight"""
        assert len(input) == 2
        start_input, end_input = input
        if len(target) == 3:
            start_target, end_target, _ = target
        else:
            assert len(target) == 2
            start_target, end_target = target
        if weight:
            b = torch.mean(F.cross_entropy(start_input, start_target, reduce=False, ignore_index=ignore_index) * weight)
            e = torch.mean(F.cross_entropy(end_input, end_target, reduce=False, ignore_index=ignore_index) * weight)
        else:
            b = F.cross_entropy(start_input, start_target, ignore_index=ignore_index)
            e = F.cross_entropy(end_input, end_target, ignore_index=ignore_index)
        loss = 0.5 * (b + e) * self.alpha
        return loss


class SpanYNCeCriterion(Criterion):

    def __init__(self, alpha=1.0, name='Span Cross Entropy Criterion'):
        super().__init__()
        """This is for extractive MRC, e.g., SQuAD, ReCoRD ... etc
        """
        self.alpha = alpha
        self.name = name

    def forward(self, input, target, weight=None, ignore_index=-1):
        """weight: sample weight"""
        assert len(input) == 3
        start_input, end_input, labels_input = input
        start_target, end_target, labels_target = target
        if weight:
            b = torch.mean(F.cross_entropy(start_input, start_target, reduce=False, ignore_index=ignore_index) * weight)
            e = torch.mean(F.cross_entropy(end_input, end_target, reduce=False, ignore_index=ignore_index) * weight)
            e = torch.mean(F.cross_entropy(labels_input, labels_target, reduce=False, ignore_index=ignore_index) * weight)
        else:
            b = F.cross_entropy(start_input, start_target, ignore_index=ignore_index)
            e = F.cross_entropy(end_input, end_target, ignore_index=ignore_index)
            c = F.cross_entropy(labels_input, labels_target, ignore_index=ignore_index)
        loss = 0.5 * (b + e) * self.alpha + c
        return loss


class MlmCriterion(Criterion):

    def __init__(self, alpha=1.0, name='BERT pre-train Criterion'):
        super().__init__()
        self.alpha = alpha
        self.name = name

    def forward(self, input, target, weight=None, ignore_index=-1):
        """TODO: support sample weight, xiaodl"""
        mlm_y, y = target
        mlm_p, nsp_p = input
        mlm_p = mlm_p.view(-1, mlm_p.size(-1))
        mlm_y = mlm_y.view(-1)
        mlm_loss = F.cross_entropy(mlm_p, mlm_y, ignore_index=ignore_index)
        nsp_loss = F.cross_entropy(nsp_p, y)
        loss = mlm_loss + nsp_loss
        loss = loss * self.alpha
        return loss


class EncoderModelType(IntEnum):
    BERT = 1
    ROBERTA = 2
    XLNET = 3
    SAN = 4
    XLM = 5
    DEBERTA = 6
    ELECTRA = 7
    T5 = 8
    T5G = 9
    MSRT5G = 10
    MSRT5 = 11
    MSRLONGT5G = 12
    MSRLONGT5 = 13


DEPARALLELIZE_DOCSTRING = """
    Moves the model to cpu from a model parallel state.
    Example:
    ```python
    # On a 4 GPU machine with t5-3b:
    model = T5ForConditionalGeneration.from_pretrained("t5-3b")
    device_map = {
        0: [0, 1, 2],
        1: [3, 4, 5, 6, 7, 8, 9],
        2: [10, 11, 12, 13, 14, 15, 16],
        3: [17, 18, 19, 20, 21, 22, 23],
    }
    model.parallelize(device_map)  # Splits the model across several devices
    model.deparallelize()  # Put the model back on cpu and cleans memory by calling torch.cuda.empty_cache()
    ```
"""

