
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


import re


import torch


from collections import namedtuple


from copy import deepcopy


import copy


from functools import partial


import logging


from typing import Optional


from typing import Sequence


from typing import Any


from typing import Union


from torch.utils.data import RandomSampler


import collections


from typing import Mapping


from typing import MutableMapping


import itertools


from functools import reduce


from functools import wraps


from typing import Tuple


from typing import List


from typing import Dict


import random


import torch.nn as nn


from functools import partialmethod


import math


from abc import ABC


from abc import abstractmethod


from typing import Callable


from scipy.stats import truncnorm


from typing import Text


import torch.utils.checkpoint


from collections import OrderedDict


from enum import Enum


import time


import torch.cuda.profiler as profiler


from functools import lru_cache


import numpy


from torch.utils.cpp_extension import BuildExtension


from torch.utils.cpp_extension import CppExtension


from torch.utils.cpp_extension import CUDAExtension


from torch.utils.cpp_extension import CUDA_HOME


from random import randint


from scipy.spatial.transform import Rotation


from torch.nn import functional as F


class Dropout(nn.Module):
    """
    Implementation of dropout with the ability to share the dropout mask
    along a particular dimension.

    If not in training mode, this module computes the identity function.
    """

    def __init__(self, r: 'float', batch_dim: 'Union[int, List[int]]'):
        """
        Args:
            r:
                Dropout rate
            batch_dim:
                Dimension(s) along which the dropout mask is shared
        """
        super(Dropout, self).__init__()
        self.r = r
        if type(batch_dim) == int:
            batch_dim = [batch_dim]
        self.batch_dim = batch_dim
        self.dropout = nn.Dropout(self.r)

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        """
        Args:
            x:
                Tensor to which dropout is applied. Can have any shape
                compatible with self.batch_dim
        """
        shape = list(x.shape)
        if self.batch_dim is not None:
            for bd in self.batch_dim:
                shape[bd] = 1
        mask = x.new_ones(shape)
        mask = self.dropout(mask)
        x *= mask
        return x


class DropoutRowwise(Dropout):
    """
    Convenience class for rowwise dropout as described in subsection
    1.11.6.
    """
    __init__ = partialmethod(Dropout.__init__, batch_dim=-3)


class DropoutColumnwise(Dropout):
    """
    Convenience class for columnwise dropout as described in subsection
    1.11.6.
    """
    __init__ = partialmethod(Dropout.__init__, batch_dim=-2)


def final_init_(weights):
    with torch.no_grad():
        weights.fill_(0.0)


def gating_init_(weights):
    with torch.no_grad():
        weights.fill_(0.0)


def glorot_uniform_init_(weights):
    nn.init.xavier_uniform_(weights, gain=1)


def _calculate_fan(linear_weight_shape, fan='fan_in'):
    fan_out, fan_in = linear_weight_shape
    if fan == 'fan_in':
        f = fan_in
    elif fan == 'fan_out':
        f = fan_out
    elif fan == 'fan_avg':
        f = (fan_in + fan_out) / 2
    else:
        raise ValueError('Invalid fan option')
    return f


def _prod(nums):
    out = 1
    for n in nums:
        out = out * n
    return out


def trunc_normal_init_(weights, scale=1.0, fan='fan_in'):
    shape = weights.shape
    f = _calculate_fan(shape, fan)
    scale = scale / max(1, f)
    a = -2
    b = 2
    std = math.sqrt(scale) / truncnorm.std(a=a, b=b, loc=0, scale=1)
    size = _prod(shape)
    samples = truncnorm.rvs(a=a, b=b, loc=0, scale=std, size=size)
    samples = np.reshape(samples, shape)
    with torch.no_grad():
        weights.copy_(torch.tensor(samples, device=weights.device))


def he_normal_init_(weights):
    trunc_normal_init_(weights, scale=2.0)


def lecun_normal_init_(weights):
    trunc_normal_init_(weights, scale=1.0)


def normal_init_(weights):
    torch.nn.init.kaiming_normal_(weights, nonlinearity='linear')


class Linear(nn.Linear):
    """
    A Linear layer with built-in nonstandard initializations. Called just
    like torch.nn.Linear.

    Implements the initializers in 1.11.4, plus some additional ones found
    in the code.
    """

    def __init__(self, in_dim: 'int', out_dim: 'int', bias: 'bool'=True, init: 'str'='default', init_fn: 'Optional[Callable[[torch.Tensor, torch.Tensor], None]]'=None, precision=None):
        """
        Args:
            in_dim:
                The final dimension of inputs to the layer
            out_dim:
                The final dimension of layer outputs
            bias:
                Whether to learn an additive bias. True by default
            init:
                The initializer to use. Choose from:

                "default": LeCun fan-in truncated normal initialization
                "relu": He initialization w/ truncated normal distribution
                "glorot": Fan-average Glorot uniform initialization
                "gating": Weights=0, Bias=1
                "normal": Normal initialization with std=1/sqrt(fan_in)
                "final": Weights=0, Bias=0

                Overridden by init_fn if the latter is not None.
            init_fn:
                A custom initializer taking weight and bias as inputs.
                Overrides init if not None.
        """
        super(Linear, self).__init__(in_dim, out_dim, bias=bias)
        if bias:
            with torch.no_grad():
                self.bias.fill_(0)
        with torch.no_grad():
            if init_fn is not None:
                init_fn(self.weight, self.bias)
            elif init == 'default':
                lecun_normal_init_(self.weight)
            elif init == 'relu':
                he_normal_init_(self.weight)
            elif init == 'glorot':
                glorot_uniform_init_(self.weight)
            elif init == 'gating':
                gating_init_(self.weight)
                if bias:
                    self.bias.fill_(1.0)
            elif init == 'normal':
                normal_init_(self.weight)
            elif init == 'final':
                final_init_(self.weight)
            else:
                raise ValueError('Invalid init string.')
        self.precision = precision

    def forward(self, input: 'torch.Tensor') ->torch.Tensor:
        d = input.dtype
        deepspeed_is_initialized = deepspeed_is_installed and deepspeed.comm.comm.is_initialized()
        if self.precision is not None:
            with torch.amp.autocast(enabled=False):
                bias = self.bias if self.bias is not None else None
                return nn.functional.linear(input.to(dtype=self.precision), self.weight.to(dtype=self.precision), bias)
        if d is torch.bfloat16 and not deepspeed_is_initialized:
            with torch.amp.autocast(enabled=False):
                bias = self.bias if self.bias is not None else None
                return nn.functional.linear(input, self.weight, bias)
        return nn.functional.linear(input, self.weight, self.bias)


def add(m1, m2, inplace):
    if not inplace:
        m1 = m1 + m2
    else:
        m1 += m2
    return m1


class InputEmbedder(nn.Module):
    """
    Embeds a subset of the input features.

    Implements Algorithms 3 (InputEmbedder) and 4 (relpos).
    """

    def __init__(self, tf_dim: 'int', msa_dim: 'int', c_z: 'int', c_m: 'int', relpos_k: 'int', **kwargs):
        """
        Args:
            tf_dim:
                Final dimension of the target features
            msa_dim:
                Final dimension of the MSA features
            c_z:
                Pair embedding dimension
            c_m:
                MSA embedding dimension
            relpos_k:
                Window size used in relative positional encoding
        """
        super(InputEmbedder, self).__init__()
        self.tf_dim = tf_dim
        self.msa_dim = msa_dim
        self.c_z = c_z
        self.c_m = c_m
        self.linear_tf_z_i = Linear(tf_dim, c_z)
        self.linear_tf_z_j = Linear(tf_dim, c_z)
        self.linear_tf_m = Linear(tf_dim, c_m)
        self.linear_msa_m = Linear(msa_dim, c_m)
        self.relpos_k = relpos_k
        self.no_bins = 2 * relpos_k + 1
        self.linear_relpos = Linear(self.no_bins, c_z)

    def relpos(self, ri: 'torch.Tensor'):
        """
        Computes relative positional encodings

        Implements Algorithm 4.

        Args:
            ri:
                "residue_index" features of shape [*, N]
        """
        d = ri[..., None] - ri[..., None, :]
        boundaries = torch.arange(start=-self.relpos_k, end=self.relpos_k + 1, device=d.device)
        reshaped_bins = boundaries.view((1,) * len(d.shape) + (len(boundaries),))
        d = d[..., None] - reshaped_bins
        d = torch.abs(d)
        d = torch.argmin(d, dim=-1)
        d = nn.functional.one_hot(d, num_classes=len(boundaries)).float()
        d = d
        return self.linear_relpos(d)

    def forward(self, tf: 'torch.Tensor', ri: 'torch.Tensor', msa: 'torch.Tensor', inplace_safe: 'bool'=False) ->Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            batch: Dict containing
                "target_feat":
                    Features of shape [*, N_res, tf_dim]
                "residue_index":
                    Features of shape [*, N_res]
                "msa_feat":
                    Features of shape [*, N_clust, N_res, msa_dim]
        Returns:
            msa_emb:
                [*, N_clust, N_res, C_m] MSA embedding
            pair_emb:
                [*, N_res, N_res, C_z] pair embedding

        """
        tf_emb_i = self.linear_tf_z_i(tf)
        tf_emb_j = self.linear_tf_z_j(tf)
        pair_emb = self.relpos(ri.type(tf_emb_i.dtype))
        pair_emb = add(pair_emb, tf_emb_i[..., None, :], inplace=inplace_safe)
        pair_emb = add(pair_emb, tf_emb_j[..., None, :, :], inplace=inplace_safe)
        n_clust = msa.shape[-3]
        tf_m = self.linear_tf_m(tf).unsqueeze(-3).expand((-1,) * len(tf.shape[:-2]) + (n_clust, -1, -1))
        msa_emb = self.linear_msa_m(msa) + tf_m
        return msa_emb, pair_emb


def one_hot(x, v_bins):
    reshaped_bins = v_bins.view((1,) * len(x.shape) + (len(v_bins),))
    diffs = x[..., None] - reshaped_bins
    am = torch.argmin(torch.abs(diffs), dim=-1)
    return nn.functional.one_hot(am, num_classes=len(v_bins)).float()


class InputEmbedderMultimer(nn.Module):
    """
    Embeds a subset of the input features.

    Implements Algorithms 3 (InputEmbedder) and 4 (relpos).
    """

    def __init__(self, tf_dim: 'int', msa_dim: 'int', c_z: 'int', c_m: 'int', max_relative_idx: 'int', use_chain_relative: 'bool', max_relative_chain: 'int', **kwargs):
        """
        Args:
            tf_dim:
                Final dimension of the target features
            msa_dim:
                Final dimension of the MSA features
            c_z:
                Pair embedding dimension
            c_m:
                MSA embedding dimension
            relpos_k:
                Window size used in relative positional encoding
        """
        super(InputEmbedderMultimer, self).__init__()
        self.tf_dim = tf_dim
        self.msa_dim = msa_dim
        self.c_z = c_z
        self.c_m = c_m
        self.linear_tf_z_i = Linear(tf_dim, c_z)
        self.linear_tf_z_j = Linear(tf_dim, c_z)
        self.linear_tf_m = Linear(tf_dim, c_m)
        self.linear_msa_m = Linear(msa_dim, c_m)
        self.max_relative_idx = max_relative_idx
        self.use_chain_relative = use_chain_relative
        self.max_relative_chain = max_relative_chain
        if self.use_chain_relative:
            self.no_bins = 2 * max_relative_idx + 2 + 1 + 2 * max_relative_chain + 2
        else:
            self.no_bins = 2 * max_relative_idx + 1
        self.linear_relpos = Linear(self.no_bins, c_z)

    def relpos(self, batch):
        pos = batch['residue_index']
        asym_id = batch['asym_id']
        asym_id_same = asym_id[..., None] == asym_id[..., None, :]
        offset = pos[..., None] - pos[..., None, :]
        clipped_offset = torch.clamp(offset + self.max_relative_idx, 0, 2 * self.max_relative_idx)
        rel_feats = []
        if self.use_chain_relative:
            final_offset = torch.where(asym_id_same, clipped_offset, (2 * self.max_relative_idx + 1) * torch.ones_like(clipped_offset))
            boundaries = torch.arange(start=0, end=2 * self.max_relative_idx + 2, device=final_offset.device)
            rel_pos = one_hot(final_offset, boundaries)
            rel_feats.append(rel_pos)
            entity_id = batch['entity_id']
            entity_id_same = entity_id[..., None] == entity_id[..., None, :]
            rel_feats.append(entity_id_same[..., None])
            sym_id = batch['sym_id']
            rel_sym_id = sym_id[..., None] - sym_id[..., None, :]
            max_rel_chain = self.max_relative_chain
            clipped_rel_chain = torch.clamp(rel_sym_id + max_rel_chain, 0, 2 * max_rel_chain)
            final_rel_chain = torch.where(entity_id_same, clipped_rel_chain, (2 * max_rel_chain + 1) * torch.ones_like(clipped_rel_chain))
            boundaries = torch.arange(start=0, end=2 * max_rel_chain + 2, device=final_rel_chain.device)
            rel_chain = one_hot(final_rel_chain, boundaries)
            rel_feats.append(rel_chain)
        else:
            boundaries = torch.arange(start=0, end=2 * self.max_relative_idx + 1, device=clipped_offset.device)
            rel_pos = one_hot(clipped_offset, boundaries)
            rel_feats.append(rel_pos)
        rel_feat = torch.cat(rel_feats, dim=-1)
        return self.linear_relpos(rel_feat)

    def forward(self, batch) ->Tuple[torch.Tensor, torch.Tensor]:
        tf = batch['target_feat']
        msa = batch['msa_feat']
        tf_emb_i = self.linear_tf_z_i(tf)
        tf_emb_j = self.linear_tf_z_j(tf)
        pair_emb = tf_emb_i[..., None, :] + tf_emb_j[..., None, :, :]
        pair_emb = pair_emb + self.relpos(batch)
        n_clust = msa.shape[-3]
        tf_m = self.linear_tf_m(tf).unsqueeze(-3).expand((-1,) * len(tf.shape[:-2]) + (n_clust, -1, -1))
        msa_emb = self.linear_msa_m(msa) + tf_m
        return msa_emb, pair_emb


class PreembeddingEmbedder(nn.Module):
    """
    Embeds the sequence pre-embedding passed to the model and the target_feat features.
    """

    def __init__(self, tf_dim: 'int', preembedding_dim: 'int', c_z: 'int', c_m: 'int', relpos_k: 'int', **kwargs):
        """
        Args:
            tf_dim:
                End channel dimension of the incoming target features
            preembedding_dim:
                End channel dimension of the incoming embeddings
            c_z:
                Pair embedding dimension
            c_m:
                Single-Seq embedding dimension
            relpos_k:
                Window size used in relative position encoding
        """
        super(PreembeddingEmbedder, self).__init__()
        self.tf_dim = tf_dim
        self.preembedding_dim = preembedding_dim
        self.c_z = c_z
        self.c_m = c_m
        self.linear_tf_m = Linear(tf_dim, c_m)
        self.linear_preemb_m = Linear(self.preembedding_dim, c_m)
        self.linear_preemb_z_i = Linear(self.preembedding_dim, c_z)
        self.linear_preemb_z_j = Linear(self.preembedding_dim, c_z)
        self.relpos_k = relpos_k
        self.no_bins = 2 * relpos_k + 1
        self.linear_relpos = Linear(self.no_bins, c_z)

    def relpos(self, ri: 'torch.Tensor'):
        """
        Computes relative positional encodings
        Args:
            ri:
                "residue_index" feature of shape [*, N]
        Returns:
                Relative positional encoding of protein using the
                residue_index feature
        """
        d = ri[..., None] - ri[..., None, :]
        boundaries = torch.arange(start=-self.relpos_k, end=self.relpos_k + 1, device=d.device)
        reshaped_bins = boundaries.view((1,) * len(d.shape) + (len(boundaries),))
        d = d[..., None] - reshaped_bins
        d = torch.abs(d)
        d = torch.argmin(d, dim=-1)
        d = nn.functional.one_hot(d, num_classes=len(boundaries)).float()
        d = d
        return self.linear_relpos(d)

    def forward(self, tf: 'torch.Tensor', ri: 'torch.Tensor', preemb: 'torch.Tensor', inplace_safe: 'bool'=False) ->Tuple[torch.Tensor, torch.Tensor]:
        tf_m = self.linear_tf_m(tf).unsqueeze(-3)
        preemb_emb = self.linear_preemb_m(preemb[..., None, :, :]) + tf_m
        preemb_emb_i = self.linear_preemb_z_i(preemb)
        preemb_emb_j = self.linear_preemb_z_j(preemb)
        pair_emb = self.relpos(ri.type(preemb_emb_i.dtype))
        pair_emb = add(pair_emb, preemb_emb_i[..., None, :], inplace=inplace_safe)
        pair_emb = add(pair_emb, preemb_emb_j[..., None, :, :], inplace=inplace_safe)
        return preemb_emb, pair_emb


class LayerNorm(nn.Module):

    def __init__(self, c_in, eps=1e-05):
        super(LayerNorm, self).__init__()
        self.c_in = c_in,
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(c_in))
        self.bias = nn.Parameter(torch.zeros(c_in))

    def forward(self, x):
        d = x.dtype
        deepspeed_is_initialized = deepspeed_is_installed and deepspeed.comm.comm.is_initialized()
        if d is torch.bfloat16 and not deepspeed_is_initialized:
            with torch.amp.autocast(enabled=False):
                out = nn.functional.layer_norm(x, self.c_in, self.weight, self.bias, self.eps)
        else:
            out = nn.functional.layer_norm(x, self.c_in, self.weight, self.bias, self.eps)
        return out


class RecyclingEmbedder(nn.Module):
    """
    Embeds the output of an iteration of the model for recycling.

    Implements Algorithm 32.
    """

    def __init__(self, c_m: 'int', c_z: 'int', min_bin: 'float', max_bin: 'float', no_bins: 'int', inf: 'float'=100000000.0, **kwargs):
        """
        Args:
            c_m:
                MSA channel dimension
            c_z:
                Pair embedding channel dimension
            min_bin:
                Smallest distogram bin (Angstroms)
            max_bin:
                Largest distogram bin (Angstroms)
            no_bins:
                Number of distogram bins
        """
        super(RecyclingEmbedder, self).__init__()
        self.c_m = c_m
        self.c_z = c_z
        self.min_bin = min_bin
        self.max_bin = max_bin
        self.no_bins = no_bins
        self.inf = inf
        self.linear = Linear(self.no_bins, self.c_z)
        self.layer_norm_m = LayerNorm(self.c_m)
        self.layer_norm_z = LayerNorm(self.c_z)

    def forward(self, m: 'torch.Tensor', z: 'torch.Tensor', x: 'torch.Tensor', inplace_safe: 'bool'=False) ->Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            m:
                First row of the MSA embedding. [*, N_res, C_m]
            z:
                [*, N_res, N_res, C_z] pair embedding
            x:
                [*, N_res, 3] predicted C_beta coordinates
        Returns:
            m:
                [*, N_res, C_m] MSA embedding update
            z:
                [*, N_res, N_res, C_z] pair embedding update
        """
        m_update = self.layer_norm_m(m)
        if inplace_safe:
            m.copy_(m_update)
            m_update = m
        z_update = self.layer_norm_z(z)
        if inplace_safe:
            z.copy_(z_update)
            z_update = z
        bins = torch.linspace(self.min_bin, self.max_bin, self.no_bins, dtype=x.dtype, device=x.device, requires_grad=False)
        squared_bins = bins ** 2
        upper = torch.cat([squared_bins[1:], squared_bins.new_tensor([self.inf])], dim=-1)
        d = torch.sum((x[..., None, :] - x[..., None, :, :]) ** 2, dim=-1, keepdims=True)
        d = ((d > squared_bins) * (d < upper)).type(x.dtype)
        d = self.linear(d)
        z_update = add(z_update, d, inplace_safe)
        return m_update, z_update


class TemplateSingleEmbedder(nn.Module):
    """
    Embeds the "template_angle_feat" feature.

    Implements Algorithm 2, line 7.
    """

    def __init__(self, c_in: 'int', c_out: 'int', **kwargs):
        """
        Args:
            c_in:
                Final dimension of "template_angle_feat"
            c_out:
                Output channel dimension
        """
        super(TemplateSingleEmbedder, self).__init__()
        self.c_out = c_out
        self.c_in = c_in
        self.linear_1 = Linear(self.c_in, self.c_out, init='relu')
        self.relu = nn.ReLU()
        self.linear_2 = Linear(self.c_out, self.c_out, init='relu')

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        """
        Args:
            x: [*, N_templ, N_res, c_in] "template_angle_feat" features
        Returns:
            x: [*, N_templ, N_res, C_out] embedding
        """
        x = self.linear_1(x)
        x = self.relu(x)
        x = self.linear_2(x)
        return x


class TemplatePairEmbedder(nn.Module):
    """
    Embeds "template_pair_feat" features.

    Implements Algorithm 2, line 9.
    """

    def __init__(self, c_in: 'int', c_out: 'int', **kwargs):
        """
        Args:
            c_in:

            c_out:
                Output channel dimension
        """
        super(TemplatePairEmbedder, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.linear = Linear(self.c_in, self.c_out, init='relu')

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        """
        Args:
            x:
                [*, C_in] input tensor
        Returns:
            [*, C_out] output tensor
        """
        x = self.linear(x)
        return x


class ExtraMSAEmbedder(nn.Module):
    """
    Embeds unclustered MSA sequences.

    Implements Algorithm 2, line 15
    """

    def __init__(self, c_in: 'int', c_out: 'int', **kwargs):
        """
        Args:
            c_in:
                Input channel dimension
            c_out:
                Output channel dimension
        """
        super(ExtraMSAEmbedder, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.linear = Linear(self.c_in, self.c_out)

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        """
        Args:
            x:
                [*, N_extra_seq, N_res, C_in] "extra_msa_feat" features
        Returns:
            [*, N_extra_seq, N_res, C_out] embedding
        """
        x = self.linear(x)
        return x

