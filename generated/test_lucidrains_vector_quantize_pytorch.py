
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


import torch.nn as nn


from torchvision import datasets


from torchvision import transforms


from torch.utils.data import DataLoader


import math


from math import log2


import torch.nn.functional as F


from functools import wraps


from functools import partial


from typing import List


from typing import Tuple


from torch.nn import Module


from torch import Tensor


from torch import int32


from torch.amp import autocast


from typing import Callable


from torch import nn


from torch.optim import Optimizer


from math import ceil


from functools import cache


from collections import namedtuple


import torch.distributed as dist


from torch.distributed import nn as dist_nn


from torch import einsum


import random


from torch.nn import ModuleList


from itertools import zip_longest


import torch.distributed as distributed


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def maybe(fn):

    @wraps(fn)
    def inner(x, *args, **kwargs):
        if not exists(x):
            return x
        return fn(x, *args, **kwargs)
    return inner


def pack_one(t, pattern):
    packed, ps = pack([t], pattern)

    def unpack_one(to_unpack, unpack_pattern=None):
        unpacked, = unpack(to_unpack, ps, default(unpack_pattern, pattern))
        return unpacked
    return packed, unpack_one


def round_ste(z: 'Tensor') ->Tensor:
    """Round with straight through gradients."""
    zhat = z.round()
    return z + (zhat - z).detach()


def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]


class FSQ(Module):

    def __init__(self, levels: 'List[int]', dim: 'int | None'=None, num_codebooks=1, keep_num_codebooks_dim: 'bool | None'=None, scale: 'float | None'=None, allowed_dtypes: 'Tuple[torch.dtype, ...]'=(torch.float32, torch.float64), channel_first: 'bool'=False, projection_has_bias: 'bool'=True, return_indices=True, force_quantization_f32=True):
        super().__init__()
        _levels = torch.tensor(levels, dtype=int32)
        self.register_buffer('_levels', _levels, persistent=False)
        _basis = torch.cumprod(torch.tensor([1] + levels[:-1]), dim=0, dtype=int32)
        self.register_buffer('_basis', _basis, persistent=False)
        self.scale = scale
        codebook_dim = len(levels)
        self.codebook_dim = codebook_dim
        effective_codebook_dim = codebook_dim * num_codebooks
        self.num_codebooks = num_codebooks
        self.effective_codebook_dim = effective_codebook_dim
        keep_num_codebooks_dim = default(keep_num_codebooks_dim, num_codebooks > 1)
        assert not (num_codebooks > 1 and not keep_num_codebooks_dim)
        self.keep_num_codebooks_dim = keep_num_codebooks_dim
        self.dim = default(dim, len(_levels) * num_codebooks)
        self.channel_first = channel_first
        has_projections = self.dim != effective_codebook_dim
        self.project_in = nn.Linear(self.dim, effective_codebook_dim, bias=projection_has_bias) if has_projections else nn.Identity()
        self.project_out = nn.Linear(effective_codebook_dim, self.dim, bias=projection_has_bias) if has_projections else nn.Identity()
        self.has_projections = has_projections
        self.return_indices = return_indices
        if return_indices:
            self.codebook_size = self._levels.prod().item()
            implicit_codebook = self._indices_to_codes(torch.arange(self.codebook_size))
            self.register_buffer('implicit_codebook', implicit_codebook, persistent=False)
        self.allowed_dtypes = allowed_dtypes
        self.force_quantization_f32 = force_quantization_f32

    def bound(self, z, eps: 'float'=0.001):
        """ Bound `z`, an array of shape (..., d). """
        half_l = (self._levels - 1) * (1 + eps) / 2
        offset = torch.where(self._levels % 2 == 0, 0.5, 0.0)
        shift = (offset / half_l).atanh()
        return (z + shift).tanh() * half_l - offset

    def quantize(self, z):
        """ Quantizes z, returns quantized zhat, same shape as z. """
        quantized = round_ste(self.bound(z))
        half_width = self._levels // 2
        return quantized / half_width

    def _scale_and_shift(self, zhat_normalized):
        half_width = self._levels // 2
        return zhat_normalized * half_width + half_width

    def _scale_and_shift_inverse(self, zhat):
        half_width = self._levels // 2
        return (zhat - half_width) / half_width

    def _indices_to_codes(self, indices):
        level_indices = self.indices_to_level_indices(indices)
        codes = self._scale_and_shift_inverse(level_indices)
        return codes

    def codes_to_indices(self, zhat):
        """ Converts a `code` to an index in the codebook. """
        assert zhat.shape[-1] == self.codebook_dim
        zhat = self._scale_and_shift(zhat)
        return (zhat * self._basis).sum(dim=-1)

    def indices_to_level_indices(self, indices):
        """ Converts indices to indices at each level, perhaps needed for a transformer with factorized embeddings """
        indices = rearrange(indices, '... -> ... 1')
        codes_non_centered = indices // self._basis % self._levels
        return codes_non_centered

    def indices_to_codes(self, indices):
        """ Inverse of `codes_to_indices`. """
        assert exists(indices)
        is_img_or_video = indices.ndim >= 3 + int(self.keep_num_codebooks_dim)
        codes = self._indices_to_codes(indices)
        if self.keep_num_codebooks_dim:
            codes = rearrange(codes, '... c d -> ... (c d)')
        codes = self.project_out(codes)
        if is_img_or_video or self.channel_first:
            codes = rearrange(codes, 'b ... d -> b d ...')
        return codes

    def forward(self, z):
        """
        einstein notation
        b - batch
        n - sequence (or flattened spatial dimensions)
        d - feature dimension
        c - number of codebook dim
        """
        is_img_or_video = z.ndim >= 4
        need_move_channel_last = is_img_or_video or self.channel_first
        if need_move_channel_last:
            z = rearrange(z, 'b d ... -> b ... d')
            z, ps = pack_one(z, 'b * d')
        assert z.shape[-1] == self.dim, f'expected dimension of {self.dim} but found dimension of {z.shape[-1]}'
        z = self.project_in(z)
        z = rearrange(z, 'b n (c d) -> b n c d', c=self.num_codebooks)
        force_f32 = self.force_quantization_f32
        quantization_context = partial(autocast, 'cuda', enabled=False) if force_f32 else nullcontext
        with quantization_context():
            orig_dtype = z.dtype
            if force_f32 and orig_dtype not in self.allowed_dtypes:
                z = z.float()
            codes = self.quantize(z)
            indices = None
            if self.return_indices:
                indices = self.codes_to_indices(codes)
            codes = rearrange(codes, 'b n c d -> b n (c d)')
            codes = codes.type(orig_dtype)
        out = self.project_out(codes)
        if need_move_channel_last:
            out = unpack_one(out, ps, 'b * d')
            out = rearrange(out, 'b ... d -> b d ...')
            indices = maybe(unpack_one)(indices, ps, 'b * c')
        if not self.keep_num_codebooks_dim and self.return_indices:
            indices = maybe(rearrange)(indices, '... 1 -> ...')
        return out, indices


class LatentQuantize(Module):

    def __init__(self, levels: 'List[int] | int', dim: 'int', commitment_loss_weight: 'float | None'=0.1, quantization_loss_weight: 'float | None'=0.1, num_codebooks: 'int'=1, codebook_dim: 'int'=-1, keep_num_codebooks_dim: 'bool | None'=None, optimize_values: 'bool | None'=True, in_place_codebook_optimizer: 'Callable[..., Optimizer]'=None):
        """
        Initializes the LatentQuantization module.

        Args:
            levels (List[int]|init): The number of levels per codebook.
                If an int is provided, it is used for all codebooks.
            dim (int): The dimensionality of the input tensor.
                The input tensor is expected to be of shape [B D ...]
            num_codebooks (int): The number of codebooks to use.
                (default is 1)
            codebook_dim (int): the dimension of the codebook.
                If levels is a list, codebook_dim is the length of the list.
                (default to -1) 
            keep_num_codebooks_dim (Optional[bool]): Whether to keep the number of codebooks dimension in the output tensor. If not provided, it is set to True if num_codebooks > 1, otherwise False.
            optimize_values (Optional[bool]): Whether to optimize the values of the codebook. If not provided, it is set to True.
        """
        super().__init__()
        self.dim = dim
        self.in_place_codebook_optimizer = in_place_codebook_optimizer
        _levels = torch.tensor(levels, dtype=int32)
        if isinstance(levels, int):
            try:
                _levels = _levels.repeat(codebook_dim)
            except RuntimeError as e:
                raise e
        self.register_buffer('commitment_loss_weight', torch.tensor(commitment_loss_weight, dtype=torch.float32), persistent=False)
        self.register_buffer('quantization_loss_weight', torch.tensor(quantization_loss_weight, dtype=torch.float32), persistent=False)
        self.register_buffer('_levels', _levels, persistent=False)
        _basis = torch.cumprod(torch.concat([torch.tensor([1], dtype=int32), _levels[:-1]], dim=0), dim=0)
        self.register_buffer('_basis', _basis, persistent=False)
        self.codebook_dim = codebook_dim if codebook_dim > 0 else len(_levels)
        effective_codebook_dim = self.codebook_dim * num_codebooks
        self.num_codebooks = num_codebooks
        self.effective_codebook_dim = effective_codebook_dim
        keep_num_codebooks_dim = keep_num_codebooks_dim if keep_num_codebooks_dim else num_codebooks > 1
        assert not (num_codebooks > 1 and not keep_num_codebooks_dim)
        self.keep_num_codebooks_dim = keep_num_codebooks_dim
        has_projections = self.dim != effective_codebook_dim
        self.project_in = nn.Linear(self.dim, effective_codebook_dim) if has_projections else nn.Identity()
        self.project_out = nn.Linear(effective_codebook_dim, self.dim) if has_projections else nn.Identity()
        self.has_projections = has_projections
        self.codebook_size = self._levels.prod().item()
        implicit_codebook = self.indices_to_codes(torch.arange(self.codebook_size), project_out=False)
        self.register_buffer('implicit_codebook', implicit_codebook, persistent=False)
        values_per_latent = [(torch.linspace(-0.5, 0.5, level) if level % 2 == 1 else torch.arange(level) / level - 0.5) for level in _levels]
        if optimize_values:
            self.values_per_latent = nn.ParameterList([nn.Parameter(values) for values in values_per_latent])
            if in_place_codebook_optimizer is not None:
                self.in_place_codebook_optimizer = in_place_codebook_optimizer(self.values_per_latent)
        else:
            self.values_per_latent = values_per_latent

    def quantization_loss(self, z: 'Tensor', zhat: 'Tensor', reduce='mean') ->Tensor:
        """Computes the quantization loss."""
        return F.mse_loss(zhat.detach(), z, reduction=reduce)

    def commitment_loss(self, z: 'Tensor', zhat: 'Tensor', reduce='mean') ->Tensor:
        """Computes the commitment loss."""
        return F.mse_loss(z.detach(), zhat, reduction=reduce)

    def quantize(self, z: 'Tensor') ->Tensor:
        """Quantizes z, returns quantized zhat, same shape as z.
        The quantization is done by measuring the distance between the input and the codebook values per latent dimension
        and returning the index of the closest codebook value.
        """

        def distance(x, y):
            return torch.abs(x - y)
        index = torch.stack([torch.argmin(distance(z[..., i, None], self.values_per_latent[i]), dim=-1) for i in range(self.codebook_dim)], dim=-1)
        quantize = torch.stack([self.values_per_latent[i][index[..., i]] for i in range(self.codebook_dim)], dim=-1)
        quantize = z + (quantize - z).detach()
        return quantize

    def _scale_and_shift(self, zhat_normalized: 'Tensor') ->Tensor:
        """scale and shift zhat from [-0.5, 0.5] to [0, level_per_dim]"""
        half_width = self._levels // 2
        return zhat_normalized * 2 * half_width + half_width

    def _scale_and_shift_inverse(self, zhat: 'Tensor') ->Tensor:
        """normalize zhat to [-0.5, 0.5]"""
        half_width = self._levels // 2
        return (zhat - half_width) / half_width / 2

    def codes_to_indices(self, zhat: 'Tensor') ->Tensor:
        """Converts a `code` which contains the number per latent to an index in the codebook."""
        assert zhat.shape[-1] == self.codebook_dim
        zhat = self._scale_and_shift(zhat)
        return (zhat * self._basis).sum(dim=-1)

    def indices_to_codes(self, indices: 'Tensor', project_out=True) ->Tensor:
        """Inverse of `codes_to_indices`."""
        indices = rearrange(indices, '... -> ... 1')
        codes_non_centered = indices // self._basis % self._levels
        codes = self._scale_and_shift_inverse(codes_non_centered)
        if self.keep_num_codebooks_dim:
            codes = rearrange(codes, '... c d -> ... (c d)')
        if project_out:
            codes = self.project_out(codes)
        codes = rearrange(codes, 'b ... d -> b d ...')
        return codes

    def quantize_and_project(self, z: 'Tensor', is_img_or_video, ps) ->Tensor:
        codes = self.quantize(z)
        indices = self.codes_to_indices(codes)
        codes = rearrange(codes, 'b n c d -> b n (c d)')
        out = self.project_out(codes)
        out = unpack_one(out, ps, 'b * d')
        out = rearrange(out, 'b ... d -> b d ...')
        indices = unpack_one(indices, ps, 'b * c')
        if not self.keep_num_codebooks_dim:
            indices = rearrange(indices, '... 1 -> ...')
        return codes, out, indices

    def forward(self, z: 'Tensor') ->Tensor:
        """
        einstein notation
        b - batch
        n - sequence (or flattened spatial dimensions)
        d - feature dimension
        c - number of codebook dim
        """
        original_input = z
        should_inplace_optimize = self.in_place_codebook_optimizer is not None
        z = rearrange(z, 'b d ... -> b ... d')
        z, ps = pack_one(z, 'b * d')
        assert z.shape[-1] == self.dim, f'expected dimension of {self.dim} but found dimension of {z.shape[-1]}'
        z = self.project_in(z)
        z = rearrange(z, 'b n (c d) -> b n c d', c=self.num_codebooks)
        codes = self.quantize(z)
        indices = self.codes_to_indices(codes)
        codes = rearrange(codes, 'b n c d -> b n (c d)')
        out = self.project_out(codes)
        out = unpack_one(out, ps, 'b * d')
        out = rearrange(out, 'b ... d -> b d ...')
        indices = unpack_one(indices, ps, 'b * c')
        if not self.keep_num_codebooks_dim:
            indices = rearrange(indices, '... 1 -> ...')
        if should_inplace_optimize and self.training and not self.optimize_values:
            loss = self.commitment_loss(z, out) if self.commitment_loss_weight != 0 else torch.tensor(0.0)
            loss += self.quantization_loss(z, out) if self.quantization_loss_weight != 0 else torch.tensor(0.0)
            loss.backward()
            self.in_place_codebook_optimizer.step()
            self.in_place_codebook_optimizer.zero_grad()
            codes = self.quantize(z)
            indices = self.codes_to_indices(codes)
            codes = rearrange(codes, 'b n c d -> b n (c d)')
            out = self.project_out(codes)
            out = unpack_one(out, ps, 'b * d')
            out = rearrange(out, 'b ... d -> b d ...')
            indices = unpack_one(indices, ps, 'b * c')
            if not self.keep_num_codebooks_dim:
                indices = rearrange(indices, '... 1 -> ...')
        commitment_loss = self.commitment_loss(original_input, out) if self.training and self.commitment_loss_weight != 0 else torch.tensor(0.0)
        quantization_loss = self.quantization_loss(original_input, out) if self.training and self.quantization_loss_weight != 0 else torch.tensor(0.0)
        loss = self.commitment_loss_weight * commitment_loss + self.quantization_loss_weight * quantization_loss
        return out, indices, loss


class CosineSimLinear(Module):

    def __init__(self, dim_in, dim_out, scale=1.0):
        super().__init__()
        self.scale = scale
        self.weight = nn.Parameter(torch.randn(dim_in, dim_out))

    def forward(self, x):
        x = F.normalize(x, dim=-1)
        w = F.normalize(self.weight, dim=0)
        return x @ w * self.scale


LossBreakdown = namedtuple('LossBreakdown', ['commitment', 'codebook_diversity', 'orthogonal_reg', 'inplace_optimize'])


Return = namedtuple('Return', ['quantized', 'indices', 'entropy_aux_loss'])


def log(t, eps=1e-20):
    return torch.log(t.clamp(min=eps))


def entropy(prob, eps=1e-05):
    return (-prob * log(prob, eps=eps)).sum(dim=-1)


def identity(t):
    return t


def l2norm(t, dim=-1, eps=1e-06):
    return F.normalize(t, p=2, dim=dim, eps=eps)


@cache
def is_distributed():
    return distributed.is_initialized() and distributed.get_world_size() > 1


def maybe_distributed_mean(t):
    if not is_distributed():
        return t
    dist_nn.all_reduce(t)
    t = t / dist.get_world_size()
    return t


class LFQ(Module):

    def __init__(self, *, dim=None, codebook_size=None, entropy_loss_weight=0.1, commitment_loss_weight=0.0, diversity_gamma=1.0, straight_through_activation=nn.Identity(), num_codebooks=1, keep_num_codebooks_dim=None, codebook_scale=1.0, frac_per_sample_entropy=1.0, has_projections=None, projection_has_bias=True, soft_clamp_input_value=None, cosine_sim_project_in=False, cosine_sim_project_in_scale=None, channel_first=None, experimental_softplus_entropy_loss=False, entropy_loss_offset=5.0, spherical=False, force_quantization_f32=True):
        super().__init__()
        assert exists(dim) or exists(codebook_size), 'either dim or codebook_size must be specified for LFQ'
        assert not exists(codebook_size) or log2(codebook_size).is_integer(), f'your codebook size must be a power of 2 for lookup free quantization (suggested {2 ** ceil(log2(codebook_size))})'
        codebook_size = default(codebook_size, lambda : 2 ** dim)
        self.codebook_size = codebook_size
        codebook_dim = int(log2(codebook_size))
        codebook_dims = codebook_dim * num_codebooks
        dim = default(dim, codebook_dims)
        has_projections = default(has_projections, dim != codebook_dims)
        if cosine_sim_project_in:
            cosine_sim_project_in = default(cosine_sim_project_in_scale, codebook_scale)
            project_in_klass = partial(CosineSimLinear, scale=cosine_sim_project_in)
        else:
            project_in_klass = partial(nn.Linear, bias=projection_has_bias)
        self.project_in = project_in_klass(dim, codebook_dims) if has_projections else nn.Identity()
        self.project_out = nn.Linear(codebook_dims, dim, bias=projection_has_bias) if has_projections else nn.Identity()
        self.has_projections = has_projections
        self.dim = dim
        self.codebook_dim = codebook_dim
        self.num_codebooks = num_codebooks
        keep_num_codebooks_dim = default(keep_num_codebooks_dim, num_codebooks > 1)
        assert not (num_codebooks > 1 and not keep_num_codebooks_dim)
        self.keep_num_codebooks_dim = keep_num_codebooks_dim
        self.channel_first = channel_first
        self.activation = straight_through_activation
        self.spherical = spherical
        self.maybe_l2norm = (lambda t: l2norm(t) * self.codebook_scale) if spherical else identity
        assert 0 < frac_per_sample_entropy <= 1.0
        self.frac_per_sample_entropy = frac_per_sample_entropy
        self.diversity_gamma = diversity_gamma
        self.entropy_loss_weight = entropy_loss_weight
        self.codebook_scale = codebook_scale
        self.commitment_loss_weight = commitment_loss_weight
        self.soft_clamp_input_value = soft_clamp_input_value
        assert not exists(soft_clamp_input_value) or soft_clamp_input_value >= codebook_scale
        self.entropy_loss_offset = entropy_loss_offset
        self.experimental_softplus_entropy_loss = experimental_softplus_entropy_loss
        self.register_buffer('mask', 2 ** torch.arange(codebook_dim - 1, -1, -1))
        self.register_buffer('zero', torch.tensor(0.0), persistent=False)
        self.force_quantization_f32 = force_quantization_f32
        all_codes = torch.arange(codebook_size)
        bits = (all_codes[..., None].int() & self.mask != 0).float()
        codebook = self.bits_to_codes(bits)
        self.register_buffer('codebook', codebook.float(), persistent=False)

    def bits_to_codes(self, bits):
        return bits * self.codebook_scale * 2 - self.codebook_scale

    @property
    def dtype(self):
        return self.codebook.dtype

    def indices_to_codes(self, indices, project_out=True):
        is_img_or_video = indices.ndim >= 3 + int(self.keep_num_codebooks_dim)
        should_transpose = default(self.channel_first, is_img_or_video)
        if not self.keep_num_codebooks_dim:
            indices = rearrange(indices, '... -> ... 1')
        bits = indices[..., None].int() & self.mask != 0
        codes = self.bits_to_codes(bits)
        codes = self.maybe_l2norm(codes)
        codes = rearrange(codes, '... c d -> ... (c d)')
        if project_out:
            codes = self.project_out(codes)
        if should_transpose:
            codes = rearrange(codes, 'b ... d -> b d ...')
        return codes

    def forward(self, x, inv_temperature=100.0, return_loss_breakdown=False, mask=None):
        """
        einstein notation
        b - batch
        n - sequence (or flattened spatial dimensions)
        d - feature dimension, which is also log2(codebook size)
        c - number of codebook dim
        """
        is_img_or_video = x.ndim >= 4
        should_transpose = default(self.channel_first, is_img_or_video)
        if should_transpose:
            x = rearrange(x, 'b d ... -> b ... d')
            x, ps = pack_one(x, 'b * d')
        assert x.shape[-1] == self.dim, f'expected dimension of {self.dim} but received {x.shape[-1]}'
        x = self.project_in(x)
        if exists(self.soft_clamp_input_value):
            clamp_value = self.soft_clamp_input_value
            x = (x / clamp_value).tanh() * clamp_value
        x = rearrange(x, 'b n (c d) -> b n c d', c=self.num_codebooks)
        x = self.maybe_l2norm(x)
        force_f32 = self.force_quantization_f32
        quantization_context = partial(autocast, 'cuda', enabled=False) if force_f32 else nullcontext
        with quantization_context():
            if force_f32:
                orig_dtype = x.dtype
                x = x.float()
            original_input = x
            codebook_value = torch.ones_like(x) * self.codebook_scale
            quantized = torch.where(x > 0, codebook_value, -codebook_value)
            indices = reduce((quantized > 0).int() * self.mask.int(), 'b n c d -> b n c', 'sum')
            quantized = self.maybe_l2norm(quantized)
            if self.training:
                x = self.activation(x)
                x = x + (quantized - x).detach()
            else:
                x = quantized
            if self.training:
                if force_f32:
                    codebook = self.codebook.float()
                codebook = self.maybe_l2norm(codebook)
                distance = -2 * einsum('... i d, j d -> ... i j', original_input, codebook)
                prob = (-distance * inv_temperature).softmax(dim=-1)
                if exists(mask):
                    prob = prob[mask]
                else:
                    prob = rearrange(prob, 'b n ... -> (b n) ...')
                if self.frac_per_sample_entropy < 1.0:
                    num_tokens = prob.shape[0]
                    num_sampled_tokens = int(num_tokens * self.frac_per_sample_entropy)
                    rand_mask = torch.randn(num_tokens).argsort(dim=-1) < num_sampled_tokens
                    per_sample_probs = prob[rand_mask]
                else:
                    per_sample_probs = prob
                per_sample_entropy = entropy(per_sample_probs).mean()
                avg_prob = reduce(per_sample_probs, '... c d -> c d', 'mean')
                avg_prob = maybe_distributed_mean(avg_prob)
                codebook_entropy = entropy(avg_prob).mean()
                entropy_aux_loss = per_sample_entropy - self.diversity_gamma * codebook_entropy
            else:
                entropy_aux_loss = per_sample_entropy = codebook_entropy = self.zero
            if self.training and self.experimental_softplus_entropy_loss:
                entropy_aux_loss = F.softplus(entropy_aux_loss + self.entropy_loss_offset)
            if self.training and self.commitment_loss_weight > 0.0:
                commit_loss = F.mse_loss(original_input, quantized.detach(), reduction='none')
                if exists(mask):
                    commit_loss = commit_loss[mask]
                commit_loss = commit_loss.mean()
            else:
                commit_loss = self.zero
            if force_f32:
                x = x.type(orig_dtype)
        x = rearrange(x, 'b n c d -> b n (c d)')
        x = self.project_out(x)
        if should_transpose:
            x = unpack_one(x, ps, 'b * d')
            x = rearrange(x, 'b ... d -> b d ...')
            indices = unpack_one(indices, ps, 'b * c')
        if not self.keep_num_codebooks_dim:
            indices = rearrange(indices, '... 1 -> ...')
        aux_loss = entropy_aux_loss * self.entropy_loss_weight + commit_loss * self.commitment_loss_weight
        ret = Return(x, indices, aux_loss)
        if not return_loss_breakdown:
            return ret
        return ret, LossBreakdown(per_sample_entropy, codebook_entropy, commit_loss)


def sample_vectors(samples, num):
    num_samples, device = samples.shape[0], samples.device
    if num_samples >= num:
        indices = torch.randperm(num_samples, device=device)[:num]
    else:
        indices = torch.randint(0, num_samples, (num,), device=device)
    return samples[indices]


def batched_sample_vectors(samples, num):
    return torch.stack([sample_vectors(sample, num) for sample in samples.unbind(dim=0)], dim=0)


def ema_inplace(old, new, decay):
    is_mps = str(old.device).startswith('mps:')
    if not is_mps:
        old.lerp_(new, 1 - decay)
    else:
        old.mul_(decay).add_(new * (1 - decay))


def batched_bincount(x, *, minlength):
    batch, dtype, device = x.shape[0], x.dtype, x.device
    target = torch.zeros(batch, minlength, dtype=dtype, device=device)
    values = torch.ones_like(x)
    target.scatter_add_(-1, x, values)
    return target


def cdist(x, y):
    x2 = reduce(x ** 2, 'b n d -> b n', 'sum')
    y2 = reduce(y ** 2, 'b n d -> b n', 'sum')
    xy = einsum('b i d, b j d -> b i j', x, y) * -2
    return (rearrange(x2, 'b i -> b i 1') + rearrange(y2, 'b j -> b 1 j') + xy).clamp(min=0).sqrt()


def noop(*args, **kwargs):
    pass


def kmeans(samples, num_clusters, num_iters=10, use_cosine_sim=False, sample_fn=batched_sample_vectors, all_reduce_fn=noop):
    num_codebooks, dim, dtype, device = samples.shape[0], samples.shape[-1], samples.dtype, samples.device
    means = sample_fn(samples, num_clusters)
    for _ in range(num_iters):
        if use_cosine_sim:
            dists = samples @ rearrange(means, 'h n d -> h d n')
        else:
            dists = -cdist(samples, means)
        buckets = torch.argmax(dists, dim=-1)
        bins = batched_bincount(buckets, minlength=num_clusters)
        all_reduce_fn(bins)
        zero_mask = bins == 0
        bins_min_clamped = bins.masked_fill(zero_mask, 1)
        new_means = buckets.new_zeros(num_codebooks, num_clusters, dim, dtype=dtype)
        new_means.scatter_add_(1, repeat(buckets, 'h n -> h n d', d=dim), samples)
        new_means = new_means / rearrange(bins_min_clamped, '... -> ... 1')
        all_reduce_fn(new_means)
        if use_cosine_sim:
            new_means = l2norm(new_means)
        means = torch.where(rearrange(zero_mask, '... -> ... 1'), means, new_means)
    return means, bins


def laplace_smoothing(x, n_categories, eps=1e-05, dim=-1):
    denom = x.sum(dim=dim, keepdim=True)
    return (x + eps) / (denom + n_categories * eps)


def all_gather_sizes(x, dim):
    size = torch.tensor(x.shape[dim], dtype=torch.long, device=x.device)
    all_sizes = [torch.empty_like(size) for _ in range(distributed.get_world_size())]
    distributed.all_gather(all_sizes, size)
    return torch.stack(all_sizes)


def pad_shape(shape, size, dim=0):
    return [(size if i == dim else s) for i, s in enumerate(shape)]


def all_gather_variably_sized(x, sizes, dim=0):
    rank = distributed.get_rank()
    all_x = []
    for i, size in enumerate(sizes):
        t = x if i == rank else x.new_empty(pad_shape(x.shape, size, dim))
        distributed.broadcast(t, src=i, async_op=True)
        all_x.append(t)
    distributed.barrier()
    return all_x


def sample_multinomial(total_count, probs):
    device = probs.device
    probs = probs.cpu()
    total_count = probs.new_full((), total_count)
    remainder = probs.new_ones(())
    sample = torch.empty_like(probs, dtype=torch.long)
    for i, p in enumerate(probs):
        s = torch.binomial(total_count, p / remainder)
        sample[i] = s
        total_count -= s
        remainder -= p
    assert total_count == 0, f'invalid total count {total_count}'
    return sample


def sample_vectors_distributed(local_samples, num):
    local_samples = rearrange(local_samples, '1 ... -> ...')
    rank = distributed.get_rank()
    all_num_samples = all_gather_sizes(local_samples, dim=0)
    if rank == 0:
        samples_per_rank = sample_multinomial(num, all_num_samples / all_num_samples.sum())
    else:
        samples_per_rank = torch.empty_like(all_num_samples)
    distributed.broadcast(samples_per_rank, src=0)
    samples_per_rank = samples_per_rank.tolist()
    local_samples = sample_vectors(local_samples, samples_per_rank[rank])
    all_samples = all_gather_variably_sized(local_samples, samples_per_rank, dim=0)
    out = torch.cat(all_samples, dim=0)
    return rearrange(out, '... -> 1 ...')


def uniform_init(*shape):
    t = torch.empty(shape)
    nn.init.kaiming_uniform_(t)
    return t


def Sequential(*modules):
    modules = [*filter(exists, modules)]
    if len(modules) == 0:
        return None
    elif len(modules) == 1:
        return modules[0]
    return nn.Sequential(*modules)


def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))


def gumbel_sample(logits, temperature=1.0, stochastic=False, straight_through=False, dim=-1, training=True):
    dtype, size = logits.dtype, logits.shape[dim]
    if training and stochastic and temperature > 0:
        sampling_logits = logits / temperature + gumbel_noise(logits)
    else:
        sampling_logits = logits
    ind = sampling_logits.argmax(dim=dim)
    one_hot = F.one_hot(ind, size).type(dtype)
    if not straight_through or temperature <= 0.0 or not training:
        return ind, one_hot
    π1 = (logits / temperature).softmax(dim=dim)
    one_hot = one_hot + π1 - π1.detach()
    return ind, one_hot


def lens_to_mask(lens, max_length):
    seq = torch.arange(max_length, device=lens.device)
    return seq < lens[:, None]


def orthogonal_loss_fn(t):
    h, n = t.shape[:2]
    normed_codes = l2norm(t)
    cosine_sim = einsum('h i d, h j d -> h i j', normed_codes, normed_codes)
    return (cosine_sim ** 2).sum() / (h * n ** 2) - 1 / n


def efficient_rotation_trick_transform(u, q, e):
    """
    4.2 in https://arxiv.org/abs/2410.06424
    """
    e = rearrange(e, 'b d -> b 1 d')
    w = l2norm(u + q, dim=1).detach()
    return e - 2 * (e @ rearrange(w, 'b d -> b d 1') @ rearrange(w, 'b d -> b 1 d')) + 2 * (e @ rearrange(u, 'b d -> b d 1').detach() @ rearrange(q, 'b d -> b 1 d').detach())


def safe_div(num, den, eps=1e-06):
    return num / den.clamp(min=eps)


def rotate_from_to(src, tgt):
    tgt, inverse = pack_one(tgt, '* d')
    src, _ = pack_one(src, '* d')
    norm_tgt = tgt.norm(dim=-1, keepdim=True)
    norm_src = src.norm(dim=-1, keepdim=True)
    rotated_src = efficient_rotation_trick_transform(safe_div(tgt, norm_tgt), safe_div(src, norm_src), tgt).squeeze()
    rotated = rotated_src * safe_div(norm_src, norm_tgt).detach()
    return inverse(rotated)


class VectorQuantize(Module):

    def __init__(self, dim, codebook_size, codebook_dim=None, heads=1, separate_codebook_per_head=False, decay=0.8, eps=1e-05, freeze_codebook=False, kmeans_init=False, kmeans_iters=10, sync_kmeans=True, use_cosine_sim=False, layernorm_after_project_in=False, threshold_ema_dead_code=0, channel_last=True, accept_image_fmap=False, commitment_weight=1.0, commitment_use_cross_entropy_loss=False, orthogonal_reg_weight=0.0, orthogonal_reg_active_codes_only=False, orthogonal_reg_max_codes=None, codebook_diversity_loss_weight=0.0, codebook_diversity_temperature=100.0, stochastic_sample_codes=False, sample_codebook_temp=1.0, straight_through=False, rotation_trick=True, sync_codebook=None, sync_affine_param=False, ema_update=True, manual_ema_update=False, learnable_codebook=False, in_place_codebook_optimizer: 'Callable[..., Optimizer]'=None, manual_in_place_optimizer_update=False, affine_param=False, affine_param_batch_decay=0.99, affine_param_codebook_decay=0.9, sync_update_v=0.0, return_zeros_for_masked_padding=True):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.separate_codebook_per_head = separate_codebook_per_head
        codebook_dim = default(codebook_dim, dim)
        codebook_input_dim = codebook_dim * heads
        requires_projection = codebook_input_dim != dim
        self.project_in = Sequential(nn.Linear(dim, codebook_input_dim), nn.LayerNorm(codebook_input_dim) if layernorm_after_project_in else None) if requires_projection else nn.Identity()
        self.project_out = nn.Linear(codebook_input_dim, dim) if requires_projection else nn.Identity()
        self.has_projections = requires_projection
        self.eps = eps
        self.has_commitment_loss = commitment_weight > 0.0
        self.commitment_weight = commitment_weight
        self.commitment_use_cross_entropy_loss = commitment_use_cross_entropy_loss
        self.learnable_codebook = learnable_codebook
        has_codebook_orthogonal_loss = orthogonal_reg_weight > 0.0
        self.has_codebook_orthogonal_loss = has_codebook_orthogonal_loss
        self.orthogonal_reg_weight = orthogonal_reg_weight
        self.orthogonal_reg_active_codes_only = orthogonal_reg_active_codes_only
        self.orthogonal_reg_max_codes = orthogonal_reg_max_codes
        has_codebook_diversity_loss = codebook_diversity_loss_weight > 0.0
        self.has_codebook_diversity_loss = has_codebook_diversity_loss
        self.codebook_diversity_temperature = codebook_diversity_temperature
        self.codebook_diversity_loss_weight = codebook_diversity_loss_weight
        assert not (straight_through and rotation_trick)
        self.rotation_trick = rotation_trick
        assert not (ema_update and learnable_codebook), 'learnable codebook not compatible with EMA update'
        assert 0 <= sync_update_v <= 1.0
        assert not (sync_update_v > 0.0 and not learnable_codebook), 'learnable codebook must be turned on'
        self.sync_update_v = sync_update_v
        codebook_class = EuclideanCodebook if not use_cosine_sim else CosineSimCodebook
        gumbel_sample_fn = partial(gumbel_sample, stochastic=stochastic_sample_codes, straight_through=straight_through)
        if not exists(sync_codebook):
            sync_codebook = is_distributed()
        codebook_kwargs = dict(dim=codebook_dim, num_codebooks=heads if separate_codebook_per_head else 1, codebook_size=codebook_size, kmeans_init=kmeans_init, kmeans_iters=kmeans_iters, sync_kmeans=sync_kmeans, decay=decay, eps=eps, threshold_ema_dead_code=threshold_ema_dead_code, use_ddp=sync_codebook, learnable_codebook=has_codebook_orthogonal_loss or learnable_codebook, sample_codebook_temp=sample_codebook_temp, gumbel_sample=gumbel_sample_fn, ema_update=ema_update, manual_ema_update=manual_ema_update)
        if affine_param:
            assert not use_cosine_sim, 'affine param is only compatible with euclidean codebook'
            codebook_kwargs = dict(**codebook_kwargs, affine_param=True, sync_affine_param=sync_affine_param, affine_param_batch_decay=affine_param_batch_decay, affine_param_codebook_decay=affine_param_codebook_decay)
        self.use_cosine_sim = use_cosine_sim
        self._codebook = codebook_class(**codebook_kwargs)
        self.in_place_codebook_optimizer = in_place_codebook_optimizer(self._codebook.parameters()) if exists(in_place_codebook_optimizer) else None
        self.manual_in_place_optimizer_update = manual_in_place_optimizer_update
        self.codebook_size = codebook_size
        self.accept_image_fmap = accept_image_fmap
        self.channel_last = channel_last
        self.register_buffer('zero', torch.tensor(0.0), persistent=False)
        self.return_zeros_for_masked_padding = return_zeros_for_masked_padding

    @property
    def codebook(self):
        codebook = self._codebook.embed
        if self.separate_codebook_per_head:
            return codebook
        return rearrange(codebook, '1 ... -> ...')

    @codebook.setter
    def codebook(self, codes):
        if not self.separate_codebook_per_head:
            codes = rearrange(codes, '... -> 1 ...')
        self._codebook.embed.copy_(codes)

    def get_codes_from_indices(self, indices):
        codebook = self.codebook
        is_multiheaded = codebook.ndim > 2
        if not is_multiheaded:
            codes = codebook[indices]
        else:
            indices, unpack_one = pack_one(indices, 'b * h')
            indices = rearrange(indices, 'b n h -> b h n')
            indices = repeat(indices, 'b h n -> b h n d', d=codebook.shape[-1])
            codebook = repeat(codebook, 'h n d -> b h n d', b=indices.shape[0])
            codes = codebook.gather(2, indices)
            codes = rearrange(codes, 'b h n d -> b n (h d)')
            codes = unpack_one(codes, 'b * d')
        if not self.channel_last:
            codes = rearrange(codes, 'b ... d -> b d ...')
        return codes

    def get_output_from_indices(self, indices):
        codes = self.get_codes_from_indices(indices)
        return self.project_out(codes)

    def update_in_place_optimizer(self):
        if not exists(self.in_place_codebook_optimizer):
            return
        self.in_place_codebook_optimizer.step()
        self.in_place_codebook_optimizer.zero_grad()

    def maybe_split_heads_from_input(self, x):
        if self.heads == 1:
            return x
        ein_rhs_eq = 'h b n d' if self.separate_codebook_per_head else '1 (b h) n d'
        return rearrange(x, f'b n (h d) -> {ein_rhs_eq}', h=self.heads)

    def expire_codes_(self, x):
        x = self._codebook.transform_input(x)
        x = self.maybe_split_heads_from_input(x)
        self._codebook.expire_codes_(x)

    def forward(self, x, indices=None, mask=None, lens=None, sample_codebook_temp=None, freeze_codebook=False, return_loss_breakdown=False, codebook_transform_fn: 'Callable | None'=None):
        orig_input = x
        assert not (exists(mask) and exists(lens))
        if exists(lens):
            mask = lens_to_mask(lens, x.shape[1])
        only_one = x.ndim == 2
        if only_one:
            assert not exists(mask)
            x = rearrange(x, 'b d -> b 1 d')
        shape, device, heads, is_multiheaded, codebook_size, return_loss = x.shape, x.device, self.heads, self.heads > 1, self.codebook_size, exists(indices)
        need_transpose = not self.channel_last and not self.accept_image_fmap
        should_inplace_optimize = exists(self.in_place_codebook_optimizer)
        if self.accept_image_fmap:
            assert not exists(mask)
            height, width = x.shape[-2:]
            x = rearrange(x, 'b c h w -> b (h w) c')
        if need_transpose:
            x = rearrange(x, 'b d n -> b n d')
        x = self.project_in(x)
        x = self.maybe_split_heads_from_input(x)
        x = self._codebook.transform_input(x)
        codebook_forward_kwargs = dict(sample_codebook_temp=sample_codebook_temp, mask=mask, freeze_codebook=freeze_codebook, codebook_transform_fn=codebook_transform_fn)
        quantize, embed_ind, distances = self._codebook(x, **codebook_forward_kwargs)
        commit_loss = orthogonal_reg_loss = inplace_optimize_loss = codebook_diversity_loss = self.zero
        if should_inplace_optimize and self.training and not freeze_codebook:
            if exists(mask):
                loss = F.mse_loss(quantize, x.detach(), reduction='none')
                loss_mask = mask
                if is_multiheaded:
                    loss_mask = repeat(mask, 'b n -> c (b h) n', c=loss.shape[0], h=loss.shape[1] // mask.shape[0])
                loss = loss[loss_mask].mean()
            else:
                loss = F.mse_loss(quantize, x.detach())
            loss.backward()
            if not self.manual_in_place_optimizer_update:
                self.update_in_place_optimizer()
            inplace_optimize_loss = loss
            quantize, embed_ind, distances = self._codebook(x, **codebook_forward_kwargs)
        if self.training:
            maybe_detach = torch.detach if not self.learnable_codebook or freeze_codebook else identity
            commit_quantize = maybe_detach(quantize)
            if self.rotation_trick:
                quantize = rotate_from_to(quantize, x)
            else:
                quantize = x + (quantize - x).detach()
            if self.sync_update_v > 0.0:
                quantize = quantize + self.sync_update_v * (quantize - quantize.detach())

        def calculate_ce_loss(codes):
            if not is_multiheaded:
                dist_einops_eq = '1 b n l -> b l n'
            elif self.separate_codebook_per_head:
                dist_einops_eq = 'c b n l -> b l n c'
            else:
                dist_einops_eq = '1 (b h) n l -> b l n h'
            ce_loss = F.cross_entropy(rearrange(distances, dist_einops_eq, b=shape[0]), codes, ignore_index=-1)
            return ce_loss
        if return_loss:
            return quantize, calculate_ce_loss(indices)
        if is_multiheaded:
            if self.separate_codebook_per_head:
                embed_ind = rearrange(embed_ind, 'h b n -> b n h', h=heads)
            else:
                embed_ind = rearrange(embed_ind, '1 (b h) n -> b n h', h=heads)
        if self.accept_image_fmap:
            embed_ind = rearrange(embed_ind, 'b (h w) ... -> b h w ...', h=height, w=width)
        if only_one:
            embed_ind = rearrange(embed_ind, 'b 1 ... -> b ...')
        loss = torch.tensor([0.0], device=device, requires_grad=self.training)
        if self.training:
            if self.has_codebook_diversity_loss:
                prob = (-distances * self.codebook_diversity_temperature).softmax(dim=-1)
                avg_prob = reduce(prob, '... n l -> n l', 'mean')
                codebook_diversity_loss = -entropy(avg_prob).mean()
                loss = loss + codebook_diversity_loss * self.codebook_diversity_loss_weight
            if self.has_commitment_loss:
                if self.commitment_use_cross_entropy_loss:
                    if exists(mask):
                        ce_loss_mask = mask
                        if is_multiheaded:
                            ce_loss_mask = repeat(ce_loss_mask, 'b n -> b n h', h=heads)
                        embed_ind.masked_fill_(~ce_loss_mask, -1)
                    commit_loss = calculate_ce_loss(embed_ind)
                elif exists(mask):
                    commit_loss = F.mse_loss(commit_quantize, x, reduction='none')
                    loss_mask = mask
                    if is_multiheaded:
                        loss_mask = repeat(loss_mask, 'b n -> c (b h) n', c=commit_loss.shape[0], h=commit_loss.shape[1] // mask.shape[0])
                    commit_loss = commit_loss[loss_mask].mean()
                else:
                    commit_loss = F.mse_loss(commit_quantize, x)
                loss = loss + commit_loss * self.commitment_weight
            if self.has_codebook_orthogonal_loss:
                codebook = self._codebook.embed
                if self.orthogonal_reg_active_codes_only:
                    assert not (is_multiheaded and self.separate_codebook_per_head), 'orthogonal regularization for only active codes not compatible with multi-headed with separate codebooks yet'
                    unique_code_ids = torch.unique(embed_ind)
                    codebook = codebook[:, unique_code_ids]
                num_codes = codebook.shape[-2]
                if exists(self.orthogonal_reg_max_codes) and num_codes > self.orthogonal_reg_max_codes:
                    rand_ids = torch.randperm(num_codes, device=device)[:self.orthogonal_reg_max_codes]
                    codebook = codebook[:, rand_ids]
                orthogonal_reg_loss = orthogonal_loss_fn(codebook)
                loss = loss + orthogonal_reg_loss * self.orthogonal_reg_weight
        if is_multiheaded:
            if self.separate_codebook_per_head:
                quantize = rearrange(quantize, 'h b n d -> b n (h d)', h=heads)
            else:
                quantize = rearrange(quantize, '1 (b h) n d -> b n (h d)', h=heads)
        quantize = self.project_out(quantize)
        if need_transpose:
            quantize = rearrange(quantize, 'b n d -> b d n')
        if self.accept_image_fmap:
            quantize = rearrange(quantize, 'b (h w) c -> b c h w', h=height, w=width)
        if only_one:
            quantize = rearrange(quantize, 'b 1 d -> b d')
        if exists(mask):
            masked_out_value = orig_input
            if self.return_zeros_for_masked_padding:
                masked_out_value = torch.zeros_like(orig_input)
            quantize = einx.where('b n, b n d, b n d -> b n d', mask, quantize, masked_out_value)
            embed_ind = einx.where('b n, b n ..., -> b n ...', mask, embed_ind, -1)
        if not return_loss_breakdown:
            return quantize, embed_ind, loss
        loss_breakdown = LossBreakdown(commit_loss, codebook_diversity_loss, orthogonal_reg_loss, inplace_optimize_loss)
        return quantize, embed_ind, loss, loss_breakdown


class RandomProjectionQuantizer(nn.Module):
    """ https://arxiv.org/abs/2202.01855 """

    def __init__(self, *, dim, codebook_size, codebook_dim, num_codebooks=1, norm=True, **kwargs):
        super().__init__()
        self.num_codebooks = num_codebooks
        rand_projs = torch.empty(num_codebooks, dim, codebook_dim)
        nn.init.xavier_normal_(rand_projs)
        self.register_buffer('rand_projs', rand_projs)
        self.norm = nn.LayerNorm(dim, elementwise_affine=False) if norm else nn.Identity()
        self.vq = VectorQuantize(dim=codebook_dim * num_codebooks, heads=num_codebooks, codebook_size=codebook_size, use_cosine_sim=True, separate_codebook_per_head=True, **kwargs)

    def forward(self, x, indices=None):
        return_loss = exists(indices)
        x = self.norm(x)
        x = einsum('b n d, h d e -> b n h e', x, self.rand_projs)
        x, ps = pack([x], 'b n *')
        self.vq.eval()
        out = self.vq(x, indices=indices)
        if return_loss:
            _, ce_loss = out
            return ce_loss
        _, indices, _ = out
        return indices


def get_maybe_sync_seed(device, max_size=10000):
    rand_int = torch.randint(0, max_size, (), device=device)
    if is_distributed():
        dist.all_reduce(rand_int)
    return rand_int.item()


def round_up_multiple(num, mult):
    return ceil(num / mult) * mult


class ResidualFSQ(Module):
    """ Follows Algorithm 1. in https://arxiv.org/pdf/2107.03312.pdf """

    def __init__(self, *, levels: List[int], num_quantizers, dim=None, is_channel_first=False, quantize_dropout=False, quantize_dropout_cutoff_index=0, quantize_dropout_multiple_of=1, **kwargs):
        super().__init__()
        codebook_dim = len(levels)
        dim = default(dim, codebook_dim)
        requires_projection = codebook_dim != dim
        self.project_in = nn.Linear(dim, codebook_dim) if requires_projection else nn.Identity()
        self.project_out = nn.Linear(codebook_dim, dim) if requires_projection else nn.Identity()
        self.has_projections = requires_projection
        self.is_channel_first = is_channel_first
        self.num_quantizers = num_quantizers
        self.levels = levels
        self.layers = nn.ModuleList([])
        levels_tensor = torch.Tensor(levels)
        scales = []
        for ind in range(num_quantizers):
            scales.append((levels_tensor - 1) ** -ind)
            fsq = FSQ(levels=levels, dim=codebook_dim, **kwargs)
            self.layers.append(fsq)
        assert all([(not fsq.has_projections) for fsq in self.layers])
        self.codebook_size = self.layers[0].codebook_size
        self.register_buffer('scales', torch.stack(scales), persistent=False)
        self.quantize_dropout = quantize_dropout and num_quantizers > 1
        assert quantize_dropout_cutoff_index >= 0
        self.quantize_dropout_cutoff_index = quantize_dropout_cutoff_index
        self.quantize_dropout_multiple_of = quantize_dropout_multiple_of

    @property
    def codebooks(self):
        codebooks = [layer.implicit_codebook for layer in self.layers]
        codebooks = torch.stack(codebooks, dim=0)
        return codebooks

    def get_codes_from_indices(self, indices):
        batch, quantize_dim = indices.shape[0], indices.shape[-1]
        indices, ps = pack([indices], 'b * q')
        if quantize_dim < self.num_quantizers:
            assert self.quantize_dropout > 0.0, 'quantize dropout must be greater than 0 if you wish to reconstruct from a signal with less fine quantizations'
            indices = F.pad(indices, (0, self.num_quantizers - quantize_dim), value=-1)
        mask = indices == -1
        indices = indices.masked_fill(mask, 0)
        all_codes = get_at('q [c] d, b n q -> q b n d', self.codebooks, indices)
        all_codes = all_codes.masked_fill(rearrange(mask, 'b n q -> q b n 1'), 0.0)
        scales = rearrange(self.scales, 'q d -> q 1 1 d')
        all_codes = all_codes * scales
        all_codes, = unpack(all_codes, ps, 'q b * d')
        return all_codes

    def get_output_from_indices(self, indices):
        codes = self.get_codes_from_indices(indices)
        codes_summed = reduce(codes, 'q ... -> ...', 'sum')
        return self.project_out(codes_summed)

    def forward(self, x, return_all_codes=False, rand_quantize_dropout_fixed_seed=None):
        num_quant, quant_dropout_multiple_of, device = self.num_quantizers, self.quantize_dropout_multiple_of, x.device
        if self.is_channel_first:
            x = rearrange(x, 'b d ... -> b ... d')
            x, ps = pack([x], 'b * d')
        x = self.project_in(x)
        quantized_out = 0.0
        residual = x
        all_indices = []
        should_quantize_dropout = self.training and self.quantize_dropout
        if should_quantize_dropout:
            if not exists(rand_quantize_dropout_fixed_seed):
                rand_quantize_dropout_fixed_seed = get_maybe_sync_seed(device)
            rand = random.Random(rand_quantize_dropout_fixed_seed)
            rand_quantize_dropout_index = rand.randrange(self.quantize_dropout_cutoff_index, num_quant)
            if quant_dropout_multiple_of != 1:
                rand_quantize_dropout_index = round_up_multiple(rand_quantize_dropout_index + 1, quant_dropout_multiple_of) - 1
            null_indices = torch.full(x.shape[:2], -1.0, device=device, dtype=torch.long)
        with autocast('cuda', enabled=False):
            for quantizer_index, (layer, scale) in enumerate(zip(self.layers, self.scales)):
                if should_quantize_dropout and quantizer_index > rand_quantize_dropout_index:
                    all_indices.append(null_indices)
                    continue
                quantized, indices = layer(residual / scale)
                quantized = quantized * scale
                residual = residual - quantized.detach()
                quantized_out = quantized_out + quantized
                all_indices.append(indices)
        quantized_out = self.project_out(quantized_out)
        all_indices = torch.stack(all_indices, dim=-1)
        if self.is_channel_first:
            quantized_out, = unpack(quantized_out, ps, 'b * d')
            all_indices, = unpack(all_indices, ps, 'b * d')
            quantized_out = rearrange(quantized_out, 'b ... d -> b d ...')
            all_indices = rearrange(all_indices, 'b ... d -> b d ...')
        ret = quantized_out, all_indices
        if not return_all_codes:
            return ret
        all_codes = self.get_codes_from_indices(all_indices)
        return *ret, all_codes


class GroupedResidualFSQ(Module):

    def __init__(self, *, dim, groups=1, accept_image_fmap=False, **kwargs):
        super().__init__()
        self.dim = dim
        self.groups = groups
        assert dim % groups == 0
        dim_per_group = dim // groups
        self.accept_image_fmap = accept_image_fmap
        self.rvqs = nn.ModuleList([])
        for _ in range(groups):
            self.rvqs.append(ResidualFSQ(dim=dim_per_group, **kwargs))
        self.codebook_size = self.rvqs[0].codebook_size

    @property
    def codebooks(self):
        return torch.stack(tuple(rvq.codebooks for rvq in self.rvqs))

    @property
    def split_dim(self):
        return 1 if self.accept_image_fmap else -1

    def get_codes_from_indices(self, indices):
        codes = tuple(rvq.get_codes_from_indices(chunk_indices) for rvq, chunk_indices in zip(self.rvqs, indices))
        return torch.stack(codes)

    def get_output_from_indices(self, indices):
        outputs = tuple(rvq.get_output_from_indices(chunk_indices) for rvq, chunk_indices in zip(self.rvqs, indices))
        return torch.cat(outputs, dim=self.split_dim)

    def forward(self, x, return_all_codes=False):
        shape, split_dim, device = x.shape, self.split_dim, x.device
        assert shape[split_dim] == self.dim
        x = x.chunk(self.groups, dim=split_dim)
        forward_kwargs = dict(return_all_codes=return_all_codes, rand_quantize_dropout_fixed_seed=get_maybe_sync_seed(device) if self.training else None)
        out = tuple(rvq(chunk, **forward_kwargs) for rvq, chunk in zip(self.rvqs, x))
        out = tuple(zip(*out))
        quantized, all_indices, *maybe_all_codes = out
        quantized = torch.cat(quantized, dim=split_dim)
        all_indices = torch.stack(all_indices)
        ret = quantized, all_indices, *maybe_all_codes
        return ret


class ResidualLFQ(Module):
    """ Follows Algorithm 1. in https://arxiv.org/pdf/2107.03312.pdf """

    def __init__(self, *, dim, num_quantizers, codebook_size, quantize_dropout=False, quantize_dropout_cutoff_index=0, quantize_dropout_multiple_of=1, soft_clamp_input_value=None, **kwargs):
        super().__init__()
        codebook_dim = int(log2(codebook_size))
        requires_projection = codebook_dim != dim
        self.project_in = nn.Linear(dim, codebook_dim) if requires_projection else nn.Identity()
        self.project_out = nn.Linear(codebook_dim, dim) if requires_projection else nn.Identity()
        self.has_projections = requires_projection
        self.num_quantizers = num_quantizers
        self.layers = nn.ModuleList([])
        for ind in range(num_quantizers):
            codebook_scale = 2 ** -ind
            lfq = LFQ(dim=codebook_dim, codebook_scale=codebook_scale, soft_clamp_input_value=soft_clamp_input_value, **kwargs)
            self.layers.append(lfq)
            if exists(soft_clamp_input_value):
                soft_clamp_input_value *= 0.5
        assert all([(not lfq.has_projections) for lfq in self.layers])
        self.quantize_dropout = quantize_dropout and num_quantizers > 1
        assert quantize_dropout_cutoff_index >= 0
        self.quantize_dropout_cutoff_index = quantize_dropout_cutoff_index
        self.quantize_dropout_multiple_of = quantize_dropout_multiple_of

    @property
    def codebooks(self):
        codebooks = [layer.codebook for layer in self.layers]
        codebooks = torch.stack(codebooks, dim=0)
        return codebooks

    def get_codes_from_indices(self, indices):
        batch, quantize_dim = indices.shape[0], indices.shape[-1]
        indices, ps = pack([indices], 'b * q')
        if quantize_dim < self.num_quantizers:
            assert self.quantize_dropout > 0.0, 'quantize dropout must be greater than 0 if you wish to reconstruct from a signal with less fine quantizations'
            indices = F.pad(indices, (0, self.num_quantizers - quantize_dim), value=-1)
        mask = indices == -1.0
        indices = indices.masked_fill(mask, 0)
        all_codes = get_at('q [c] d, b n q -> q b n d', self.codebooks, indices)
        all_codes = all_codes.masked_fill(rearrange(mask, 'b n q -> q b n 1'), 0.0)
        all_codes, = unpack(all_codes, ps, 'q b * d')
        return all_codes

    def get_output_from_indices(self, indices):
        codes = self.get_codes_from_indices(indices)
        codes_summed = reduce(codes, 'q ... -> ...', 'sum')
        return self.project_out(codes_summed)

    def forward(self, x, mask=None, return_all_codes=False, rand_quantize_dropout_fixed_seed=None):
        num_quant, quant_dropout_multiple_of, device = self.num_quantizers, self.quantize_dropout_multiple_of, x.device
        x = self.project_in(x)
        quantized_out = 0.0
        residual = x
        all_losses = []
        all_indices = []
        should_quantize_dropout = self.training and self.quantize_dropout
        if should_quantize_dropout:
            if not exists(rand_quantize_dropout_fixed_seed):
                rand_quantize_dropout_fixed_seed = get_maybe_sync_seed(device)
            rand = random.Random(rand_quantize_dropout_fixed_seed)
            rand_quantize_dropout_index = rand.randrange(self.quantize_dropout_cutoff_index, num_quant)
            if quant_dropout_multiple_of != 1:
                rand_quantize_dropout_index = round_up_multiple(rand_quantize_dropout_index + 1, quant_dropout_multiple_of) - 1
            null_indices = torch.full(x.shape[:2], -1.0, device=device, dtype=torch.long)
            null_loss = torch.tensor(0.0, device=device, dtype=x.dtype)
        with autocast('cuda', enabled=False):
            for quantizer_index, layer in enumerate(self.layers):
                if should_quantize_dropout and quantizer_index > rand_quantize_dropout_index:
                    all_indices.append(null_indices)
                    all_losses.append(null_loss)
                    continue
                quantized, indices, loss = layer(residual, mask=mask)
                residual = residual - quantized.detach()
                quantized_out = quantized_out + quantized
                all_indices.append(indices)
                all_losses.append(loss)
        quantized_out = self.project_out(quantized_out)
        all_losses, all_indices = map(partial(torch.stack, dim=-1), (all_losses, all_indices))
        ret = quantized_out, all_indices, all_losses
        if not return_all_codes:
            return ret
        all_codes = self.get_codes_from_indices(all_indices)
        return *ret, all_codes


class GroupedResidualLFQ(Module):

    def __init__(self, *, dim, groups=1, accept_image_fmap=False, **kwargs):
        super().__init__()
        self.dim = dim
        self.groups = groups
        assert dim % groups == 0
        dim_per_group = dim // groups
        self.accept_image_fmap = accept_image_fmap
        self.rvqs = nn.ModuleList([])
        for _ in range(groups):
            self.rvqs.append(ResidualLFQ(dim=dim_per_group, **kwargs))

    @property
    def codebooks(self):
        return torch.stack(tuple(rvq.codebooks for rvq in self.rvqs))

    @property
    def split_dim(self):
        return 1 if self.accept_image_fmap else -1

    def get_codes_from_indices(self, indices):
        codes = tuple(rvq.get_codes_from_indices(chunk_indices) for rvq, chunk_indices in zip(self.rvqs, indices))
        return torch.stack(codes)

    def get_output_from_indices(self, indices):
        outputs = tuple(rvq.get_output_from_indices(chunk_indices) for rvq, chunk_indices in zip(self.rvqs, indices))
        return torch.cat(outputs, dim=self.split_dim)

    def forward(self, x, mask=None, return_all_codes=False):
        shape, split_dim, device = x.shape, self.split_dim, x.device
        assert shape[split_dim] == self.dim
        x = x.chunk(self.groups, dim=split_dim)
        forward_kwargs = dict(mask=mask, return_all_codes=return_all_codes, rand_quantize_dropout_fixed_seed=get_maybe_sync_seed(device) if self.training else None)
        out = tuple(rvq(chunk, **forward_kwargs) for rvq, chunk in zip(self.rvqs, x))
        out = tuple(zip(*out))
        quantized, all_indices, commit_losses, *maybe_all_codes = out
        quantized = torch.cat(quantized, dim=split_dim)
        all_indices = torch.stack(all_indices)
        commit_losses = torch.stack(commit_losses)
        ret = quantized, all_indices, commit_losses, *maybe_all_codes
        return ret


class MLP(Module):

    def __init__(self, dim, dim_hidden=None, depth=4, l2norm_output=False):
        super().__init__()
        dim_hidden = default(dim_hidden, dim)
        self.proj_in = nn.Linear(2 * dim, dim)
        layers = ModuleList([])
        for _ in range(depth):
            layers.append(nn.Sequential(nn.Linear(dim, dim_hidden), nn.SiLU(), nn.Linear(dim_hidden, dim)))
        self.layers = layers
        self.l2norm_output = l2norm_output

    def forward(self, codes, *, condition):
        one_headed = codes.ndim == 2
        if one_headed:
            codes = rearrange(codes, 'c d -> 1 c d')
        heads, num_codes, batch, seq_len = codes.shape[0], codes.shape[-2], condition.shape[0], condition.shape[-2]
        codes = repeat(codes, 'h c d -> h b n c d', n=seq_len, b=batch)
        condition = repeat(condition, 'b n d -> h b n c d', c=num_codes, h=heads)
        x = torch.cat((condition, codes), dim=-1)
        x = self.proj_in(x)
        for layer in self.layers:
            x = layer(x) + x
        if self.l2norm_output:
            x = F.normalize(x, dim=-1)
        if not one_headed:
            return x
        return rearrange(x, '1 ... -> ...')


def cast_tuple(t, length=1):
    return t if isinstance(t, tuple) else (t,) * length


def first(it):
    return it[0]


def unique(arr):
    return list({*arr})


class GroupedResidualVQ(Module):

    def __init__(self, *, dim, groups=1, accept_image_fmap=False, **kwargs):
        super().__init__()
        self.dim = dim
        self.groups = groups
        assert dim % groups == 0
        dim_per_group = dim // groups
        self.accept_image_fmap = accept_image_fmap
        self.rvqs = ModuleList([])
        for _ in range(groups):
            self.rvqs.append(ResidualVQ(dim=dim_per_group, accept_image_fmap=accept_image_fmap, **kwargs))

    @property
    def codebooks(self):
        return torch.stack(tuple(rvq.codebooks for rvq in self.rvqs))

    @property
    def split_dim(self):
        return 1 if self.accept_image_fmap else -1

    def get_codes_from_indices(self, indices):
        codes = tuple(rvq.get_codes_from_indices(chunk_indices) for rvq, chunk_indices in zip(self.rvqs, indices))
        return torch.stack(codes)

    def get_output_from_indices(self, indices):
        outputs = tuple(rvq.get_output_from_indices(chunk_indices) for rvq, chunk_indices in zip(self.rvqs, indices))
        return torch.cat(outputs, dim=self.split_dim)

    def forward(self, x, indices=None, return_all_codes=False, sample_codebook_temp=None, freeze_codebook=False, mask=None):
        shape, split_dim, device = x.shape, self.split_dim, x.device
        assert shape[split_dim] == self.dim
        x = x.chunk(self.groups, dim=split_dim)
        indices = default(indices, tuple())
        return_ce_loss = len(indices) > 0
        assert len(indices) == 0 or len(indices) == self.groups
        forward_kwargs = dict(return_all_codes=return_all_codes, sample_codebook_temp=sample_codebook_temp, mask=mask, freeze_codebook=freeze_codebook, rand_quantize_dropout_fixed_seed=get_maybe_sync_seed(device) if self.training else None)
        out = tuple(rvq(chunk, indices=chunk_indices, **forward_kwargs) for rvq, chunk, chunk_indices in zip_longest(self.rvqs, x, indices))
        out = tuple(zip(*out))
        if return_ce_loss:
            quantized, ce_losses = out
            return torch.cat(quantized, dim=split_dim), sum(ce_losses)
        quantized, all_indices, commit_losses, *maybe_all_codes = out
        quantized = torch.cat(quantized, dim=split_dim)
        all_indices = torch.stack(all_indices)
        commit_losses = torch.stack(commit_losses)
        ret = quantized, all_indices, commit_losses, *maybe_all_codes
        return ret


class EuclideanCodebook(Module):

    def __init__(self, dim, codebook_size, num_codebooks=1, kmeans_init=False, kmeans_iters=10, sync_kmeans=True, decay=0.8, eps=1e-05, threshold_ema_dead_code=2, reset_cluster_size=None, use_ddp=False, learnable_codebook=False, gumbel_sample=gumbel_sample, sample_codebook_temp=1.0, ema_update=True, manual_ema_update=False, affine_param=False, sync_affine_param=False, affine_param_batch_decay=0.99, affine_param_codebook_decay=0.9):
        super().__init__()
        self.transform_input = identity
        self.decay = decay
        self.ema_update = ema_update
        self.manual_ema_update = manual_ema_update
        init_fn = uniform_init if not kmeans_init else torch.zeros
        embed = init_fn(num_codebooks, codebook_size, dim)
        self.codebook_size = codebook_size
        self.num_codebooks = num_codebooks
        self.kmeans_iters = kmeans_iters
        self.eps = eps
        self.threshold_ema_dead_code = threshold_ema_dead_code
        self.reset_cluster_size = default(reset_cluster_size, threshold_ema_dead_code)
        assert callable(gumbel_sample)
        self.gumbel_sample = gumbel_sample
        self.sample_codebook_temp = sample_codebook_temp
        assert not (use_ddp and num_codebooks > 1 and kmeans_init), 'kmeans init is not compatible with multiple codebooks in distributed environment for now'
        self.sample_fn = sample_vectors_distributed if use_ddp and sync_kmeans else batched_sample_vectors
        self.replace_sample_fn = sample_vectors_distributed if use_ddp and sync_kmeans else batched_sample_vectors
        self.kmeans_all_reduce_fn = distributed.all_reduce if use_ddp and sync_kmeans else noop
        self.all_reduce_fn = distributed.all_reduce if use_ddp else noop
        self.register_buffer('initted', torch.Tensor([not kmeans_init]))
        self.register_buffer('cluster_size', torch.ones(num_codebooks, codebook_size))
        self.register_buffer('embed_avg', embed.clone())
        self.learnable_codebook = learnable_codebook
        if learnable_codebook:
            self.embed = nn.Parameter(embed)
        else:
            self.register_buffer('embed', embed)
        self.affine_param = affine_param
        self.sync_affine_param = sync_affine_param
        if not affine_param:
            return
        self.affine_param_batch_decay = affine_param_batch_decay
        self.affine_param_codebook_decay = affine_param_codebook_decay
        self.register_buffer('batch_mean', None)
        self.register_buffer('batch_variance', None)
        self.register_buffer('codebook_mean_needs_init', torch.Tensor([True]))
        self.register_buffer('codebook_mean', torch.empty(num_codebooks, 1, dim))
        self.register_buffer('codebook_variance_needs_init', torch.Tensor([True]))
        self.register_buffer('codebook_variance', torch.empty(num_codebooks, 1, dim))

    @torch.jit.ignore
    def init_embed_(self, data, mask=None):
        if self.initted:
            return
        if exists(mask):
            c = data.shape[0]
            data = rearrange(data[mask], '(c n) d -> c n d', c=c)
        embed, cluster_size = kmeans(data, self.codebook_size, self.kmeans_iters, sample_fn=self.sample_fn, all_reduce_fn=self.kmeans_all_reduce_fn)
        embed_sum = embed * rearrange(cluster_size, '... -> ... 1')
        self.embed.data.copy_(embed)
        self.embed_avg.data.copy_(embed_sum)
        self.cluster_size.data.copy_(cluster_size)
        self.initted.data.copy_(torch.Tensor([True]))

    @torch.jit.ignore
    def update_with_decay(self, buffer_name, new_value, decay):
        old_value = getattr(self, buffer_name)
        needs_init = getattr(self, buffer_name + '_needs_init', False)
        if needs_init:
            self.register_buffer(buffer_name + '_needs_init', torch.Tensor([False]))
        if not exists(old_value) or needs_init:
            self.register_buffer(buffer_name, new_value.detach())
            return
        value = old_value * decay + new_value.detach() * (1 - decay)
        self.register_buffer(buffer_name, value)

    @torch.jit.ignore
    def update_affine(self, data, embed, mask=None):
        assert self.affine_param
        var_fn = partial(torch.var, unbiased=False)
        embed = rearrange(embed, 'h ... d -> h (...) d')
        if self.training:
            self.update_with_decay('codebook_mean', reduce(embed, 'h n d -> h 1 d', 'mean'), self.affine_param_codebook_decay)
            self.update_with_decay('codebook_variance', reduce(embed, 'h n d -> h 1 d', var_fn), self.affine_param_codebook_decay)
        data = rearrange(data, 'h ... d -> h (...) d')
        if exists(mask):
            c = data.shape[0]
            data = rearrange(data[mask], '(c n) d -> c n d', c=c)
        if not self.sync_affine_param:
            self.update_with_decay('batch_mean', reduce(data, 'h n d -> h 1 d', 'mean'), self.affine_param_batch_decay)
            self.update_with_decay('batch_variance', reduce(data, 'h n d -> h 1 d', var_fn), self.affine_param_batch_decay)
            return
        num_vectors, device, dtype = data.shape[-2], data.device, data.dtype
        num_vectors = torch.tensor([num_vectors], device=device, dtype=dtype)
        distributed.all_reduce(num_vectors)
        batch_sum = reduce(data, 'h n d -> h 1 d', 'sum')
        distributed.all_reduce(batch_sum)
        batch_mean = batch_sum / num_vectors
        self.update_with_decay('batch_mean', batch_mean, self.affine_param_batch_decay)
        variance_numer = reduce((data - batch_mean) ** 2, 'h n d -> h 1 d', 'sum')
        distributed.all_reduce(variance_numer)
        batch_variance = variance_numer / num_vectors
        self.update_with_decay('batch_variance', batch_variance, self.affine_param_batch_decay)

    def replace(self, batch_samples, batch_mask):
        for ind, (samples, mask) in enumerate(zip(batch_samples, batch_mask)):
            sampled = self.replace_sample_fn(rearrange(samples, '... -> 1 ...'), mask.sum().item())
            sampled = rearrange(sampled, '1 ... -> ...')
            self.embed.data[ind][mask] = sampled
            self.cluster_size.data[ind][mask] = self.reset_cluster_size
            self.embed_avg.data[ind][mask] = sampled * self.reset_cluster_size

    def expire_codes_(self, batch_samples):
        if self.threshold_ema_dead_code == 0:
            return
        expired_codes = self.cluster_size < self.threshold_ema_dead_code
        if not torch.any(expired_codes):
            return
        batch_samples = rearrange(batch_samples, 'h ... d -> h (...) d')
        self.replace(batch_samples, batch_mask=expired_codes)

    def update_ema(self):
        cluster_size = laplace_smoothing(self.cluster_size, self.codebook_size, self.eps) * self.cluster_size.sum(dim=-1, keepdim=True)
        embed_normalized = self.embed_avg / rearrange(cluster_size, '... -> ... 1')
        self.embed.data.copy_(embed_normalized)

    @autocast('cuda', enabled=False)
    def forward(self, x, sample_codebook_temp=None, mask=None, freeze_codebook=False, codebook_transform_fn: 'Callable | None'=None):
        needs_codebook_dim = x.ndim < 4
        sample_codebook_temp = default(sample_codebook_temp, self.sample_codebook_temp)
        x = x.float()
        if needs_codebook_dim:
            x = rearrange(x, '... -> 1 ...')
        dtype = x.dtype
        flatten, unpack_one = pack_one(x, 'h * d')
        if exists(mask):
            mask = repeat(mask, 'b n -> c (b h n)', c=flatten.shape[0], h=flatten.shape[-2] // (mask.shape[0] * mask.shape[1]))
        self.init_embed_(flatten, mask=mask)
        if self.affine_param:
            self.update_affine(flatten, self.embed, mask=mask)
        if self.affine_param:
            codebook_std = self.codebook_variance.clamp(min=1e-05).sqrt()
            batch_std = self.batch_variance.clamp(min=1e-05).sqrt()
            embed = (embed - self.codebook_mean) * (batch_std / codebook_std) + self.batch_mean
        embed = self.embed if self.learnable_codebook else self.embed.detach()
        if exists(codebook_transform_fn):
            transformed_embed = codebook_transform_fn(embed)
            transformed_embed = rearrange(transformed_embed, 'h b n c d -> h (b n) c d')
            broadcastable_input = rearrange(flatten, '... d -> ... 1 d')
            dist = -F.pairwise_distance(broadcastable_input, transformed_embed)
        else:
            dist = -cdist(flatten, embed)
        embed_ind, embed_onehot = self.gumbel_sample(dist, dim=-1, temperature=sample_codebook_temp, training=self.training)
        embed_ind = unpack_one(embed_ind, 'h *')
        if exists(codebook_transform_fn):
            transformed_embed = unpack_one(transformed_embed, 'h * c d')
        if self.training:
            unpacked_onehot = unpack_one(embed_onehot, 'h * c')
            if exists(codebook_transform_fn):
                quantize = einsum('h b n c, h b n c d -> h b n d', unpacked_onehot, transformed_embed)
            else:
                quantize = einsum('h b n c, h c d -> h b n d', unpacked_onehot, embed)
        elif exists(codebook_transform_fn):
            quantize = einx.get_at('h b n [c] d, h b n -> h b n d', transformed_embed, embed_ind)
        else:
            quantize = einx.get_at('h [c] d, h b n -> h b n d', embed, embed_ind)
        if self.training and self.ema_update and not freeze_codebook:
            if self.affine_param:
                flatten = (flatten - self.batch_mean) * (codebook_std / batch_std) + self.codebook_mean
            if exists(mask):
                embed_onehot[~mask] = 0.0
            cluster_size = embed_onehot.sum(dim=1)
            self.all_reduce_fn(cluster_size)
            ema_inplace(self.cluster_size.data, cluster_size, self.decay)
            embed_sum = einsum('h n d, h n c -> h c d', flatten, embed_onehot)
            embed_sum = embed_sum.contiguous()
            self.all_reduce_fn(embed_sum)
            ema_inplace(self.embed_avg.data, embed_sum, self.decay)
            if not self.manual_ema_update:
                self.update_ema()
                self.expire_codes_(x)
        if needs_codebook_dim:
            quantize, embed_ind = map(lambda t: rearrange(t, '1 ... -> ...'), (quantize, embed_ind))
        dist = unpack_one(dist, 'h * d')
        return quantize, embed_ind, dist


class CosineSimCodebook(Module):

    def __init__(self, dim, codebook_size, num_codebooks=1, kmeans_init=False, kmeans_iters=10, sync_kmeans=True, decay=0.8, eps=1e-05, threshold_ema_dead_code=2, reset_cluster_size=None, use_ddp=False, learnable_codebook=False, gumbel_sample=gumbel_sample, sample_codebook_temp=1.0, ema_update=True, manual_ema_update=False):
        super().__init__()
        self.transform_input = l2norm
        self.ema_update = ema_update
        self.manual_ema_update = manual_ema_update
        self.decay = decay
        if not kmeans_init:
            embed = l2norm(uniform_init(num_codebooks, codebook_size, dim))
        else:
            embed = torch.zeros(num_codebooks, codebook_size, dim)
        self.codebook_size = codebook_size
        self.num_codebooks = num_codebooks
        self.kmeans_iters = kmeans_iters
        self.eps = eps
        self.threshold_ema_dead_code = threshold_ema_dead_code
        self.reset_cluster_size = default(reset_cluster_size, threshold_ema_dead_code)
        assert callable(gumbel_sample)
        self.gumbel_sample = gumbel_sample
        self.sample_codebook_temp = sample_codebook_temp
        self.sample_fn = sample_vectors_distributed if use_ddp and sync_kmeans else batched_sample_vectors
        self.replace_sample_fn = sample_vectors_distributed if use_ddp and sync_kmeans else batched_sample_vectors
        self.kmeans_all_reduce_fn = distributed.all_reduce if use_ddp and sync_kmeans else noop
        self.all_reduce_fn = distributed.all_reduce if use_ddp else noop
        self.register_buffer('initted', torch.Tensor([not kmeans_init]))
        self.register_buffer('cluster_size', torch.ones(num_codebooks, codebook_size))
        self.register_buffer('embed_avg', embed.clone())
        self.learnable_codebook = learnable_codebook
        if learnable_codebook:
            self.embed = nn.Parameter(embed)
        else:
            self.register_buffer('embed', embed)

    @torch.jit.ignore
    def init_embed_(self, data, mask=None):
        if self.initted:
            return
        if exists(mask):
            c = data.shape[0]
            data = rearrange(data[mask], '(c n) d -> c n d', c=c)
        embed, cluster_size = kmeans(data, self.codebook_size, self.kmeans_iters, use_cosine_sim=True, sample_fn=self.sample_fn, all_reduce_fn=self.kmeans_all_reduce_fn)
        embed_sum = embed * rearrange(cluster_size, '... -> ... 1')
        self.embed.data.copy_(embed)
        self.embed_avg.data.copy_(embed_sum)
        self.cluster_size.data.copy_(cluster_size)
        self.initted.data.copy_(torch.Tensor([True]))

    def replace(self, batch_samples, batch_mask):
        batch_samples = l2norm(batch_samples)
        for ind, (samples, mask) in enumerate(zip(batch_samples, batch_mask)):
            sampled = self.replace_sample_fn(rearrange(samples, '... -> 1 ...'), mask.sum().item())
            sampled = rearrange(sampled, '1 ... -> ...')
            self.embed.data[ind][mask] = sampled
            self.embed_avg.data[ind][mask] = sampled * self.reset_cluster_size
            self.cluster_size.data[ind][mask] = self.reset_cluster_size

    def expire_codes_(self, batch_samples):
        if self.threshold_ema_dead_code == 0:
            return
        expired_codes = self.cluster_size < self.threshold_ema_dead_code
        if not torch.any(expired_codes):
            return
        batch_samples = rearrange(batch_samples, 'h ... d -> h (...) d')
        self.replace(batch_samples, batch_mask=expired_codes)

    def update_ema(self):
        cluster_size = laplace_smoothing(self.cluster_size, self.codebook_size, self.eps) * self.cluster_size.sum(dim=-1, keepdim=True)
        embed_normalized = self.embed_avg / rearrange(cluster_size, '... -> ... 1')
        embed_normalized = l2norm(embed_normalized)
        self.embed.data.copy_(embed_normalized)

    @autocast('cuda', enabled=False)
    def forward(self, x, sample_codebook_temp=None, mask=None, freeze_codebook=False, codebook_transform_fn: 'Callable | None'=None):
        needs_codebook_dim = x.ndim < 4
        sample_codebook_temp = default(sample_codebook_temp, self.sample_codebook_temp)
        x = x.float()
        if needs_codebook_dim:
            x = rearrange(x, '... -> 1 ...')
        dtype = x.dtype
        flatten, unpack_one = pack_one(x, 'h * d')
        if exists(mask):
            mask = repeat(mask, 'b n -> c (b h n)', c=flatten.shape[0], h=flatten.shape[-2] // (mask.shape[0] * mask.shape[1]))
        self.init_embed_(flatten, mask=mask)
        embed = self.embed if self.learnable_codebook else self.embed.detach()
        if exists(codebook_transform_fn):
            transformed_embed = codebook_transform_fn(embed)
            transformed_embed = rearrange(transformed_embed, 'h b n c d -> h (b n) c d')
            transformed_embed = l2norm(transformed_embed)
            dist = einsum('h n d, h n c d -> h n c', flatten, transformed_embed)
        else:
            dist = einsum('h n d, h c d -> h n c', flatten, embed)
        embed_ind, embed_onehot = self.gumbel_sample(dist, dim=-1, temperature=sample_codebook_temp, training=self.training)
        embed_ind = unpack_one(embed_ind, 'h *')
        if exists(codebook_transform_fn):
            transformed_embed = unpack_one(transformed_embed, 'h * c d')
        if self.training:
            unpacked_onehot = unpack_one(embed_onehot, 'h * c')
            if exists(codebook_transform_fn):
                quantize = einsum('h b n c, h b n c d -> h b n d', unpacked_onehot, transformed_embed)
            else:
                quantize = einsum('h b n c, h c d -> h b n d', unpacked_onehot, embed)
        elif exists(codebook_transform_fn):
            quantize = einx.get_at('h b n [c] d, h b n -> h b n d', transformed_embed, embed_ind)
        else:
            quantize = einx.get_at('h [c] d, h b n -> h b n d', embed, embed_ind)
        if self.training and self.ema_update and not freeze_codebook:
            if exists(mask):
                embed_onehot[~mask] = 0.0
            bins = embed_onehot.sum(dim=1)
            self.all_reduce_fn(bins)
            ema_inplace(self.cluster_size.data, bins, self.decay)
            embed_sum = einsum('h n d, h n c -> h c d', flatten, embed_onehot)
            embed_sum = embed_sum.contiguous()
            self.all_reduce_fn(embed_sum)
            ema_inplace(self.embed_avg.data, embed_sum, self.decay)
            if not self.manual_ema_update:
                self.update_ema()
                self.expire_codes_(x)
        if needs_codebook_dim:
            quantize, embed_ind = map(lambda t: rearrange(t, '1 ... -> ...'), (quantize, embed_ind))
        dist = unpack_one(dist, 'h * d')
        return quantize, embed_ind, dist


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (CosineSimLinear,
     lambda: ([], {'dim_in': 4, 'dim_out': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
]

