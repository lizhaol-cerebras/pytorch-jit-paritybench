
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


import random


import re


from torch.utils.data import Dataset


import torch.nn as nn


from torch.nn import functional as F


import math


from torch import nn


from functools import partial


import torch.nn.functional as F


from torch.nn.utils import weight_norm


from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present


import time


from collections import deque


from inspect import isfunction


from torch.nn import Conv1d


from torch.nn import Mish


from typing import Optional


from torch import autocast


from torch.cuda.amp import GradScaler


from math import sqrt


import copy


from typing import Tuple


from sklearn.cluster import KMeans


from functools import reduce


from torch.nn.modules.module import _addindent


import logging


import matplotlib


import matplotlib.pyplot as plt


from torch.utils.tensorboard import SummaryWriter


from torch.nn import ConvTranspose1d


from torch.nn import AvgPool1d


from torch.nn import Conv2d


from torch.nn.utils import remove_weight_norm


from torch.nn.utils import spectral_norm


import torch.utils.data


from scipy.io.wavfile import read


import matplotlib.pylab as plt


from torch.optim import lr_scheduler


class SSSLoss(nn.Module):
    """
    Single-scale Spectral Loss. 
    """

    def __init__(self, n_fft=111, alpha=1.0, overlap=0, eps=1e-07):
        super().__init__()
        self.n_fft = n_fft
        self.alpha = alpha
        self.eps = eps
        self.hop_length = int(n_fft * (1 - overlap))
        self.spec = torchaudio.transforms.Spectrogram(n_fft=self.n_fft, hop_length=self.hop_length, power=1, normalized=True, center=False)

    def forward(self, x_true, x_pred):
        S_true = self.spec(x_true) + self.eps
        S_pred = self.spec(x_pred) + self.eps
        converge_term = torch.mean(torch.linalg.norm(S_true - S_pred, dim=(1, 2)) / torch.linalg.norm(S_true + S_pred, dim=(1, 2)))
        log_term = F.l1_loss(S_true.log(), S_pred.log())
        loss = converge_term + self.alpha * log_term
        return loss


class RSSLoss(nn.Module):
    """
    Random-scale Spectral Loss.
    """

    def __init__(self, fft_min, fft_max, n_scale, alpha=1.0, overlap=0, eps=1e-07, device='cuda'):
        super().__init__()
        self.fft_min = fft_min
        self.fft_max = fft_max
        self.n_scale = n_scale
        self.lossdict = {}
        for n_fft in range(fft_min, fft_max):
            self.lossdict[n_fft] = SSSLoss(n_fft, alpha, overlap, eps)

    def forward(self, x_pred, x_true):
        value = 0.0
        n_ffts = torch.randint(self.fft_min, self.fft_max, (self.n_scale,))
        for n_fft in n_ffts:
            loss_func = self.lossdict[int(n_fft)]
            value += loss_func(x_true, x_pred)
        return value / self.n_scale


class Transpose(nn.Module):

    def __init__(self, dims):
        super().__init__()
        assert len(dims) == 2, 'dims must be a tuple of two dimensions'
        self.dims = dims

    def forward(self, x):
        return x.transpose(*self.dims)


def calc_same_padding(kernel_size):
    pad = kernel_size // 2
    return pad, pad - (kernel_size + 1) % 2


class ConformerConvModule(nn.Module):

    def __init__(self, dim, expansion_factor=2, kernel_size=31, dropout=0.0, use_norm=False, conv_model_type='mode1'):
        super().__init__()
        inner_dim = dim * expansion_factor
        padding = calc_same_padding(kernel_size)
        if conv_model_type == 'mode1':
            self.net = nn.Sequential(nn.LayerNorm(dim) if use_norm else nn.Identity(), Transpose((1, 2)), nn.Conv1d(dim, inner_dim * 2, 1), nn.GLU(dim=1), nn.Conv1d(inner_dim, inner_dim, kernel_size=kernel_size, padding=padding[0], groups=inner_dim), nn.SiLU(), nn.Conv1d(inner_dim, dim, 1), Transpose((1, 2)), nn.Dropout(dropout))
        elif conv_model_type == 'mode2':
            raise NotImplementedError('mode2 not implemented yet')
        else:
            raise ValueError(f'{conv_model_type} is not a valid conv_model_type')

    def forward(self, x):
        return self.net(x)


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def orthogonal_matrix_chunk(cols, qr_uniform_q=False, device=None):
    unstructured_block = torch.randn((cols, cols), device=device)
    q, r = torch.linalg.qr(unstructured_block.cpu(), mode='reduced')
    q, r = map(lambda t: t, (q, r))
    if qr_uniform_q:
        d = torch.diag(r, 0)
        q *= d.sign()
    return q.t()


def gaussian_orthogonal_random_matrix(nb_rows, nb_columns, scaling=0, qr_uniform_q=False, device=None):
    nb_full_blocks = int(nb_rows / nb_columns)
    block_list = []
    for _ in range(nb_full_blocks):
        q = orthogonal_matrix_chunk(nb_columns, qr_uniform_q=qr_uniform_q, device=device)
        block_list.append(q)
    remaining_rows = nb_rows - nb_full_blocks * nb_columns
    if remaining_rows > 0:
        q = orthogonal_matrix_chunk(nb_columns, qr_uniform_q=qr_uniform_q, device=device)
        block_list.append(q[:remaining_rows])
    final_matrix = torch.cat(block_list)
    if scaling == 0:
        multiplier = torch.randn((nb_rows, nb_columns), device=device).norm(dim=1)
    elif scaling == 1:
        multiplier = math.sqrt(float(nb_columns)) * torch.ones((nb_rows,), device=device)
    else:
        raise ValueError(f'Invalid scaling {scaling}')
    return torch.diag(multiplier) @ final_matrix


def linear_attention(q, k, v):
    if v is None:
        out = torch.einsum('...ed,...nd->...ne', k, q)
        return out
    else:
        k_cumsum = k.sum(dim=-2)
        D_inv = 1.0 / (torch.einsum('...nd,...d->...n', q, k_cumsum.type_as(q)) + 1e-08)
        context = torch.einsum('...nd,...ne->...de', k, v)
        out = torch.einsum('...de,...nd,...n->...ne', context, q, D_inv)
        return out


def softmax_kernel(data, *, projection_matrix, is_query, normalize_data=True, eps=0.0001, device=None):
    b, h, *_ = data.shape
    data_normalizer = data.shape[-1] ** -0.25 if normalize_data else 1.0
    ratio = projection_matrix.shape[0] ** -0.5
    projection = repeat(projection_matrix, 'j d -> b h j d', b=b, h=h)
    projection = projection.type_as(data)
    data_dash = torch.einsum('...id,...jd->...ij', data_normalizer * data, projection)
    diag_data = data ** 2
    diag_data = torch.sum(diag_data, dim=-1)
    diag_data = diag_data / 2.0 * data_normalizer ** 2
    diag_data = diag_data.unsqueeze(dim=-1)
    if is_query:
        data_dash = ratio * (torch.exp(data_dash - diag_data - torch.max(data_dash, dim=-1, keepdim=True).values) + eps)
    else:
        data_dash = ratio * torch.exp(data_dash - diag_data + eps)
    return data_dash.type_as(data)


class FastAttention(nn.Module):

    def __init__(self, dim_heads, nb_features=None, ortho_scaling=0, causal=False, generalized_attention=False, kernel_fn=nn.ReLU(), qr_uniform_q=False, no_projection=False):
        super().__init__()
        nb_features = default(nb_features, int(dim_heads * math.log(dim_heads)))
        self.dim_heads = dim_heads
        self.nb_features = nb_features
        self.ortho_scaling = ortho_scaling
        self.create_projection = partial(gaussian_orthogonal_random_matrix, nb_rows=self.nb_features, nb_columns=dim_heads, scaling=ortho_scaling, qr_uniform_q=qr_uniform_q)
        projection_matrix = self.create_projection()
        self.register_buffer('projection_matrix', projection_matrix)
        self.generalized_attention = generalized_attention
        self.kernel_fn = kernel_fn
        self.no_projection = no_projection
        self.causal = causal
        if causal:
            try:
                self.causal_linear_fn = partial(causal_linear_attention)
            except ImportError:
                None
                self.causal_linear_fn = causal_linear_attention_noncuda

    @torch.no_grad()
    def redraw_projection_matrix(self):
        projections = self.create_projection()
        self.projection_matrix.copy_(projections)
        del projections

    def forward(self, q, k, v):
        device = q.device
        global FLAG_PCMER_NORM
        if FLAG_PCMER_NORM:
            q = q / (q.norm(dim=-1, keepdim=True) + 1e-08)
            k = k / (k.norm(dim=-1, keepdim=True) + 1e-08)
        if self.no_projection:
            q = q.softmax(dim=-1)
            k = torch.exp(k) if self.causal else k.softmax(dim=-2)
        elif self.generalized_attention:
            create_kernel = partial(generalized_kernel, kernel_fn=self.kernel_fn, projection_matrix=self.projection_matrix, device=device)
            q, k = map(create_kernel, (q, k))
        else:
            create_kernel = partial(softmax_kernel, projection_matrix=self.projection_matrix, device=device)
            q = create_kernel(q, is_query=True)
            k = create_kernel(k, is_query=False)
        attn_fn = linear_attention if not self.causal else self.causal_linear_fn
        if v is None:
            out = attn_fn(q, k, None)
            return out
        else:
            out = attn_fn(q, k, v)
            return out


def empty(tensor):
    return tensor.numel() == 0


class SelfAttention(nn.Module):

    def __init__(self, dim, causal=False, heads=8, dim_head=64, local_heads=0, local_window_size=256, nb_features=None, feature_redraw_interval=1000, generalized_attention=False, kernel_fn=nn.ReLU(), qr_uniform_q=False, dropout=0.0, no_projection=False):
        super().__init__()
        assert dim % heads == 0, 'dimension must be divisible by number of heads'
        dim_head = default(dim_head, dim // heads)
        inner_dim = dim_head * heads
        self.fast_attention = FastAttention(dim_head, nb_features, causal=causal, generalized_attention=generalized_attention, kernel_fn=kernel_fn, qr_uniform_q=qr_uniform_q, no_projection=no_projection)
        self.heads = heads
        self.global_heads = heads - local_heads
        self.local_attn = LocalAttention(window_size=local_window_size, causal=causal, autopad=True, dropout=dropout, look_forward=int(not causal), rel_pos_emb_config=(dim_head, local_heads)) if local_heads > 0 else None
        self.to_q = nn.Linear(dim, inner_dim)
        self.to_k = nn.Linear(dim, inner_dim)
        self.to_v = nn.Linear(dim, inner_dim)
        self.to_out = nn.Linear(inner_dim, dim)
        self.dropout = nn.Dropout(dropout)

    @torch.no_grad()
    def redraw_projection_matrix(self):
        self.fast_attention.redraw_projection_matrix()

    def forward(self, x, context=None, mask=None, context_mask=None, name=None, inference=False, **kwargs):
        b, n, _, h, gh = *x.shape, self.heads, self.global_heads
        cross_attend = exists(context)
        context = default(context, x)
        context_mask = default(context_mask, mask) if not cross_attend else context_mask
        q, k, v = self.to_q(x), self.to_k(context), self.to_v(context)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))
        (q, lq), (k, lk), (v, lv) = map(lambda t: (t[:, :gh], t[:, gh:]), (q, k, v))
        attn_outs = []
        if not empty(q):
            if exists(context_mask):
                global_mask = context_mask[:, None, :, None]
                v.masked_fill_(~global_mask, 0.0)
            if cross_attend:
                pass
            else:
                out = self.fast_attention(q, k, v)
            attn_outs.append(out)
        if not empty(lq):
            assert not cross_attend, 'local attention is not compatible with cross attention'
            out = self.local_attn(lq, lk, lv, input_mask=mask)
            attn_outs.append(out)
        out = torch.cat(attn_outs, dim=1)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return self.dropout(out)


class _EncoderLayer(nn.Module):
    """One layer of the encoder.
    
    Attributes:
        attn: (:class:`mha.MultiHeadAttention`): The attention mechanism that is used to read the input sequence.
        feed_forward (:class:`ffl.FeedForwardLayer`): The feed-forward layer on top of the attention mechanism.
    """

    def __init__(self, parent: 'PCmer'):
        """Creates a new instance of ``_EncoderLayer``.
        
        Args:
            parent (Encoder): The encoder that the layers is created for.
        """
        super().__init__()
        self.conformer = ConformerConvModule(parent.dim_model)
        self.norm = nn.LayerNorm(parent.dim_model)
        self.dropout = nn.Dropout(parent.residual_dropout)
        self.attn = SelfAttention(dim=parent.dim_model, heads=parent.num_heads, causal=False)

    def forward(self, phone, mask=None):
        phone = phone + self.attn(self.norm(phone), mask=mask)
        phone = phone + self.conformer(phone)
        return phone


class PCmer(nn.Module):
    """The encoder that is used in the Transformer model."""

    def __init__(self, num_layers, num_heads, dim_model, dim_keys, dim_values, residual_dropout, attention_dropout, pcmer_norm=False):
        super().__init__()
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dim_model = dim_model
        self.dim_values = dim_values
        self.dim_keys = dim_keys
        self.residual_dropout = residual_dropout
        self.attention_dropout = attention_dropout
        global FLAG_PCMER_NORM
        if pcmer_norm:
            FLAG_PCMER_NORM = True
        self._layers = nn.ModuleList([_EncoderLayer(self) for _ in range(num_layers)])

    def forward(self, phone, mask=None):
        for i, layer in enumerate(self._layers):
            phone = layer(phone, mask)
        return phone


class Swish(nn.Module):

    def forward(self, x):
        return x * x.sigmoid()


class GLU(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        out, gate = x.chunk(2, dim=self.dim)
        return out * gate.sigmoid()


class DepthWiseConv1d(nn.Module):

    def __init__(self, chan_in, chan_out, kernel_size, padding):
        super().__init__()
        self.padding = padding
        self.conv = nn.Conv1d(chan_in, chan_out, kernel_size, groups=chan_in)

    def forward(self, x):
        x = F.pad(x, self.padding)
        return self.conv(x)


class CFNEncoderLayer(nn.Module):
    """
    Conformer Naive Encoder Layer

    Args:
        dim_model (int): Dimension of model
        num_heads (int): Number of heads
        use_norm (bool): Whether to use norm for FastAttention, only True can use bf16/fp16, default False
        conv_only (bool): Whether to use only conv module without attention, default False
        conv_dropout (float): Dropout rate of conv module, default 0.1
        atten_dropout (float): Dropout rate of attention module, default 0.1
    """

    def __init__(self, dim_model: 'int', num_heads: 'int'=8, use_norm: 'bool'=False, conv_only: 'bool'=False, conv_dropout: 'float'=0.0, atten_dropout: 'float'=0.1):
        super().__init__()
        self.conformer = ConformerConvModule(dim_model, use_norm=use_norm, dropout=conv_dropout)
        self.norm = nn.LayerNorm(dim_model)
        self.dropout = nn.Dropout(0.1)
        if not conv_only:
            self.attn = nn.TransformerEncoderLayer(d_model=dim_model, nhead=num_heads, dim_feedforward=dim_model * 4, dropout=atten_dropout, activation='gelu')
        else:
            self.attn = None

    def forward(self, x, mask=None) ->torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor (#batch, length, dim_model)
            mask (torch.Tensor): Mask tensor, default None
        return:
            torch.Tensor: Output tensor (#batch, length, dim_model)
        """
        if self.attn is not None:
            x = x + self.attn(self.norm(x), mask=mask)
        x = x + self.conformer(x)
        return x


class ConformerNaiveEncoder(nn.Module):
    """
    Conformer Naive Encoder

    Args:
        dim_model (int): Dimension of model
        num_layers (int): Number of layers
        num_heads (int): Number of heads
        use_norm (bool): Whether to use norm for FastAttention, only True can use bf16/fp16, default False
        conv_only (bool): Whether to use only conv module without attention, default False
        conv_dropout (float): Dropout rate of conv module, default 0.
        atten_dropout (float): Dropout rate of attention module, default 0.
    """

    def __init__(self, num_layers: 'int', num_heads: 'int', dim_model: 'int', use_norm: 'bool'=False, conv_only: 'bool'=False, conv_dropout: 'float'=0.0, atten_dropout: 'float'=0.0):
        super().__init__()
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dim_model = dim_model
        self.use_norm = use_norm
        self.residual_dropout = 0.1
        self.attention_dropout = 0.1
        self.encoder_layers = nn.ModuleList([CFNEncoderLayer(dim_model, num_heads, use_norm, conv_only, conv_dropout, atten_dropout) for _ in range(num_layers)])

    def forward(self, x, mask=None) ->torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor (#batch, length, dim_model)
            mask (torch.Tensor): Mask tensor, default None
        return:
            torch.Tensor: Output tensor (#batch, length, dim_model)
        """
        for i, layer in enumerate(self.encoder_layers):
            x = layer(x, mask)
        return x


def split_to_dict(tensor, tensor_splits):
    """Split a tensor into a dictionary of multiple tensors."""
    labels = []
    sizes = []
    for k, v in tensor_splits.items():
        labels.append(k)
        sizes.append(v)
    tensors = torch.split(tensor, sizes, dim=-1)
    return dict(zip(labels, tensors))


class Unit2Control(nn.Module):

    def __init__(self, input_channel, n_spk, output_splits, use_pitch_aug=False, pcmer_norm=False, use_naive_v2=False, use_conv_stack=True):
        super().__init__()
        self.output_splits = output_splits
        self.f0_embed = nn.Linear(1, 256)
        self.phase_embed = nn.Linear(1, 256)
        self.volume_embed = nn.Linear(1, 256)
        self.n_spk = n_spk
        if n_spk is not None and n_spk > 1:
            self.spk_embed = nn.Embedding(n_spk, 256)
        if use_pitch_aug:
            self.aug_shift_embed = nn.Linear(1, 256, bias=False)
        else:
            self.aug_shift_embed = None
        if use_conv_stack:
            self.stack = nn.Sequential(nn.Conv1d(input_channel, 256, 3, 1, 1), nn.GroupNorm(4, 256), nn.LeakyReLU(), nn.Conv1d(256, 256, 3, 1, 1))
        else:
            self.stack = nn.Conv1d(input_channel, 256, 3, 1, 1)
        if use_naive_v2:
            self.decoder = ConformerNaiveEncoder(num_layers=3, num_heads=8, dim_model=256, use_norm=False, conv_only=True, conv_dropout=0, atten_dropout=0.1)
        else:
            self.decoder = PCmer(num_layers=3, num_heads=8, dim_model=256, dim_keys=256, dim_values=256, residual_dropout=0.1, attention_dropout=0.1, pcmer_norm=pcmer_norm)
        self.norm = nn.LayerNorm(256)
        self.n_out = sum([v for k, v in output_splits.items()])
        self.dense_out = weight_norm(nn.Linear(256, self.n_out))

    def forward(self, units, f0, phase, volume, spk_id=None, spk_mix_dict=None, aug_shift=None):
        """
        input: 
            B x n_frames x n_unit
        return: 
            dict of B x n_frames x feat
        """
        x = self.stack(units.transpose(1, 2)).transpose(1, 2)
        x = x + self.f0_embed((1 + f0 / 700).log()) + self.phase_embed(phase / np.pi) + self.volume_embed(volume)
        if self.n_spk is not None and self.n_spk > 1:
            if spk_mix_dict is not None:
                for k, v in spk_mix_dict.items():
                    spk_id_torch = torch.LongTensor(np.array([[k]]))
                    x = x + v * self.spk_embed(spk_id_torch - 1)
            else:
                x = x + self.spk_embed(spk_id - 1)
        if self.aug_shift_embed is not None and aug_shift is not None:
            x = x + self.aug_shift_embed(aug_shift / 5)
        x = self.decoder(x)
        x = self.norm(x)
        e = self.dense_out(x)
        controls = split_to_dict(e, self.output_splits)
        return controls, x


class FeatureExtractor(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv0 = nn.Conv1d(1, 512, 10, 5, bias=False)
        self.norm0 = nn.GroupNorm(512, 512)
        self.conv1 = nn.Conv1d(512, 512, 3, 2, bias=False)
        self.conv2 = nn.Conv1d(512, 512, 3, 2, bias=False)
        self.conv3 = nn.Conv1d(512, 512, 3, 2, bias=False)
        self.conv4 = nn.Conv1d(512, 512, 3, 2, bias=False)
        self.conv5 = nn.Conv1d(512, 512, 2, 2, bias=False)
        self.conv6 = nn.Conv1d(512, 512, 2, 2, bias=False)

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        x = F.gelu(self.norm0(self.conv0(x)))
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = F.gelu(self.conv3(x))
        x = F.gelu(self.conv4(x))
        x = F.gelu(self.conv5(x))
        x = F.gelu(self.conv6(x))
        return x


class FeatureProjection(nn.Module):

    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(512)
        self.projection = nn.Linear(512, 768)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        x = self.norm(x)
        x = self.projection(x)
        x = self.dropout(x)
        return x


class PositionalConvEmbedding(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(768, 768, kernel_size=128, padding=128 // 2, groups=16)
        self.conv = nn.utils.weight_norm(self.conv, name='weight', dim=2)

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        x = self.conv(x.transpose(1, 2))
        x = F.gelu(x[:, :, :-1])
        return x.transpose(1, 2)


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer: 'nn.TransformerEncoderLayer', num_layers: 'int') ->None:
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers

    def forward(self, src: 'torch.Tensor', mask: 'torch.Tensor'=None, src_key_padding_mask: 'torch.Tensor'=None, output_layer: 'Optional[int]'=None) ->torch.Tensor:
        output = src
        for layer in self.layers[:output_layer]:
            output = layer(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
        return output


def _compute_mask(shape: 'Tuple[int, int]', mask_prob: 'float', mask_length: 'int', device: 'torch.device', min_masks: 'int'=0) ->torch.Tensor:
    batch_size, sequence_length = shape
    if mask_length < 1:
        raise ValueError('`mask_length` has to be bigger than 0.')
    if mask_length > sequence_length:
        raise ValueError(f'`mask_length` has to be smaller than `sequence_length`, but got `mask_length`: {mask_length} and `sequence_length`: {sequence_length}`')
    num_masked_spans = int(mask_prob * sequence_length / mask_length + random.random())
    num_masked_spans = max(num_masked_spans, min_masks)
    if num_masked_spans * mask_length > sequence_length:
        num_masked_spans = sequence_length // mask_length
    mask = torch.zeros((batch_size, sequence_length), device=device, dtype=torch.bool)
    uniform_dist = torch.ones((batch_size, sequence_length - (mask_length - 1)), device=device)
    mask_indices = torch.multinomial(uniform_dist, num_masked_spans)
    mask_indices = mask_indices.unsqueeze(dim=-1).expand((batch_size, num_masked_spans, mask_length)).reshape(batch_size, num_masked_spans * mask_length)
    offsets = torch.arange(mask_length, device=device)[None, None, :].expand((batch_size, num_masked_spans, mask_length)).reshape(batch_size, num_masked_spans * mask_length)
    mask_idxs = mask_indices + offsets
    mask = mask.scatter(1, mask_idxs, True)
    return mask


class Hubert(nn.Module):

    def __init__(self, num_label_embeddings: 'int'=100, mask: 'bool'=True):
        super().__init__()
        self._mask = mask
        self.feature_extractor = FeatureExtractor()
        self.feature_projection = FeatureProjection()
        self.positional_embedding = PositionalConvEmbedding()
        self.norm = nn.LayerNorm(768)
        self.dropout = nn.Dropout(0.1)
        self.encoder = TransformerEncoder(nn.TransformerEncoderLayer(768, 12, 3072, activation='gelu', batch_first=True), 12)
        self.proj = nn.Linear(768, 256)
        self.masked_spec_embed = nn.Parameter(torch.FloatTensor(768).uniform_())
        self.label_embedding = nn.Embedding(num_label_embeddings, 256)

    def mask(self, x: 'torch.Tensor') ->Tuple[torch.Tensor, torch.Tensor]:
        mask = None
        if self.training and self._mask:
            mask = _compute_mask((x.size(0), x.size(1)), 0.8, 10, x.device, 2)
            x[mask] = self.masked_spec_embed
        return x, mask

    def encode(self, x: 'torch.Tensor', layer: 'Optional[int]'=None) ->Tuple[torch.Tensor, torch.Tensor]:
        x = self.feature_extractor(x)
        x = self.feature_projection(x.transpose(1, 2))
        x, mask = self.mask(x)
        x = x + self.positional_embedding(x)
        x = self.dropout(self.norm(x))
        x = self.encoder(x, output_layer=layer)
        return x, mask

    def logits(self, x: 'torch.Tensor') ->torch.Tensor:
        logits = torch.cosine_similarity(x.unsqueeze(2), self.label_embedding.weight.unsqueeze(0).unsqueeze(0), dim=-1)
        return logits / 0.1

    def forward(self, x: 'torch.Tensor') ->Tuple[torch.Tensor, torch.Tensor]:
        x, mask = self.encode(x)
        x = self.proj(x)
        logits = self.logits(x)
        return logits, mask


class HubertSoft(Hubert):

    def __init__(self):
        super().__init__()

    @torch.inference_mode()
    def units(self, wav: 'torch.Tensor') ->torch.Tensor:
        wav = F.pad(wav, ((400 - 320) // 2, (400 - 320) // 2))
        x, _ = self.encode(wav)
        return self.proj(x)


class Audio2HubertSoft(torch.nn.Module):

    def __init__(self, path, h_sample_rate=16000, h_hop_size=320):
        super().__init__()
        None
        self.hubert = HubertSoft()
        None
        checkpoint = torch.load(path)
        consume_prefix_in_state_dict_if_present(checkpoint, 'module.')
        self.hubert.load_state_dict(checkpoint)
        self.hubert.eval()

    def forward(self, audio):
        with torch.inference_mode():
            units = self.hubert.units(audio.unsqueeze(1))
            return units


class CNHubertSoftFish(torch.nn.Module):

    def __init__(self, path, h_sample_rate=16000, h_hop_size=320, device='cpu', gate_size=10):
        super().__init__()
        self.device = device
        self.gate_size = gate_size
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained('./pretrain/TencentGameMate/chinese-hubert-base')
        self.model = HubertModel.from_pretrained('./pretrain/TencentGameMate/chinese-hubert-base')
        self.proj = torch.nn.Sequential(torch.nn.Dropout(0.1), torch.nn.Linear(768, 256))
        state_dict = torch.load(path, map_location=device)
        self.load_state_dict(state_dict)

    @torch.no_grad()
    def forward(self, audio):
        input_values = self.feature_extractor(audio, sampling_rate=16000, return_tensors='pt').input_values
        input_values = input_values
        return self._forward(input_values[0])

    @torch.no_grad()
    def _forward(self, input_values):
        features = self.model(input_values)
        features = self.proj(features.last_hidden_state)
        topk, indices = torch.topk(features, self.gate_size, dim=2)
        features = torch.zeros_like(features).scatter(2, indices, topk)
        features = features / features.sum(2, keepdim=True)
        return features


def crop_and_compensate_delay(audio, audio_size, ir_size, padding='same', delay_compensation=-1):
    """Crop audio output from convolution to compensate for group delay.
  Args:
    audio: Audio after convolution. Tensor of shape [batch, time_steps].
    audio_size: Initial size of the audio before convolution.
    ir_size: Size of the convolving impulse response.
    padding: Either 'valid' or 'same'. For 'same' the final output to be the
      same size as the input audio (audio_timesteps). For 'valid' the audio is
      extended to include the tail of the impulse response (audio_timesteps +
      ir_timesteps - 1).
    delay_compensation: Samples to crop from start of output audio to compensate
      for group delay of the impulse response. If delay_compensation < 0 it
      defaults to automatically calculating a constant group delay of the
      windowed linear phase filter from frequency_impulse_response().
  Returns:
    Tensor of cropped and shifted audio.
  Raises:
    ValueError: If padding is not either 'valid' or 'same'.
  """
    if padding == 'valid':
        crop_size = ir_size + audio_size - 1
    elif padding == 'same':
        crop_size = audio_size
    else:
        raise ValueError("Padding must be 'valid' or 'same', instead of {}.".format(padding))
    total_size = int(audio.shape[-1])
    crop = total_size - crop_size
    start = ir_size // 2 if delay_compensation < 0 else delay_compensation
    end = crop - start
    return audio[:, start:-end]


def get_fft_size(frame_size: 'int', ir_size: 'int', power_of_2: 'bool'=True):
    """Calculate final size for efficient FFT.
  Args:
    frame_size: Size of the audio frame.
    ir_size: Size of the convolving impulse response.
    power_of_2: Constrain to be a power of 2. If False, allow other 5-smooth
      numbers. TPU requires power of 2, while GPU is more flexible.
  Returns:
    fft_size: Size for efficient FFT.
  """
    convolved_frame_size = ir_size + frame_size - 1
    if power_of_2:
        fft_size = int(2 ** np.ceil(np.log2(convolved_frame_size)))
    else:
        fft_size = convolved_frame_size
    return fft_size


def fft_convolve(audio, impulse_response):
    """Filter audio with frames of time-varying impulse responses.
    Time-varying filter. Given audio [batch, n_samples], and a series of impulse
    responses [batch, n_frames, n_impulse_response], splits the audio into frames,
    applies filters, and then overlap-and-adds audio back together.
    Applies non-windowed non-overlapping STFT/ISTFT to efficiently compute
    convolution for large impulse response sizes.
    Args:
        audio: Input audio. Tensor of shape [batch, audio_timesteps].
        impulse_response: Finite impulse response to convolve. Can either be a 2-D
        Tensor of shape [batch, ir_size], or a 3-D Tensor of shape [batch,
        ir_frames, ir_size]. A 2-D tensor will apply a single linear
        time-invariant filter to the audio. A 3-D Tensor will apply a linear
        time-varying filter. Automatically chops the audio into equally shaped
        blocks to match ir_frames.
    Returns:
        audio_out: Convolved audio. Tensor of shape
            [batch, audio_timesteps].
    """
    ir_shape = impulse_response.size()
    if len(ir_shape) == 2:
        impulse_response = impulse_response.unsqueeze(1)
        ir_shape = impulse_response.size()
    batch_size_ir, n_ir_frames, ir_size = ir_shape
    batch_size, audio_size = audio.size()
    if batch_size != batch_size_ir:
        raise ValueError('Batch size of audio ({}) and impulse response ({}) must be the same.'.format(batch_size, batch_size_ir))
    hop_size = int(audio_size / n_ir_frames)
    frame_size = 2 * hop_size
    audio_frames = F.pad(audio, (hop_size, hop_size)).unfold(1, frame_size, hop_size)
    window = torch.bartlett_window(frame_size)
    audio_frames = audio_frames * window
    fft_size = get_fft_size(frame_size, ir_size, power_of_2=False)
    audio_fft = torch.fft.rfft(audio_frames, fft_size)
    ir_fft = torch.fft.rfft(torch.cat((impulse_response, impulse_response[:, -1:, :]), 1), fft_size)
    audio_ir_fft = torch.multiply(audio_fft, ir_fft)
    audio_frames_out = torch.fft.irfft(audio_ir_fft, fft_size)
    batch_size, n_audio_frames, frame_size = audio_frames_out.size()
    fold = torch.nn.Fold(output_size=(1, (n_audio_frames - 1) * hop_size + frame_size), kernel_size=(1, frame_size), stride=(1, hop_size))
    output_signal = fold(audio_frames_out.transpose(1, 2)).squeeze(1).squeeze(1)
    output_signal = crop_and_compensate_delay(output_signal[:, hop_size:], audio_size, ir_size)
    return output_signal


def apply_dynamic_window_to_impulse_response(impulse_response, half_width_frames):
    ir_size = int(impulse_response.size(-1))
    window = torch.arange(-(ir_size // 2), (ir_size + 1) // 2) / half_width_frames
    window[window > 1] = 0
    window = (1 + torch.cos(np.pi * window)) / 2
    impulse_response = impulse_response.roll(ir_size // 2, -1)
    impulse_response = impulse_response * window
    return impulse_response


def apply_window_to_impulse_response(impulse_response, window_size: 'int'=0, causal: 'bool'=False):
    """Apply a window to an impulse response and put in causal form.
    Args:
        impulse_response: A series of impulse responses frames to window, of shape
        [batch, n_frames, ir_size]. ---------> ir_size means size of filter_bank ??????
        
        window_size: Size of the window to apply in the time domain. If window_size
        is less than 1, it defaults to the impulse_response size.
        causal: Impulse response input is in causal form (peak in the middle).
    Returns:
        impulse_response: Windowed impulse response in causal form, with last
        dimension cropped to window_size if window_size is greater than 0 and less
        than ir_size.
    """
    if causal:
        impulse_response = torch.fftshift(impulse_response, axes=-1)
    ir_size = int(impulse_response.size(-1))
    if window_size <= 0 or window_size > ir_size:
        window_size = ir_size
    window = nn.Parameter(torch.hann_window(window_size), requires_grad=False)
    padding = ir_size - window_size
    if padding > 0:
        half_idx = (window_size + 1) // 2
        window = torch.cat([window[half_idx:], torch.zeros([padding]), window[:half_idx]], axis=0)
    else:
        window = window.roll(window.size(-1) // 2, -1)
    window = window.unsqueeze(0)
    impulse_response = impulse_response * window
    if padding > 0:
        first_half_start = ir_size - (half_idx - 1) + 1
        second_half_end = half_idx + 1
        impulse_response = torch.cat([impulse_response[..., first_half_start:], impulse_response[..., :second_half_end]], dim=-1)
    else:
        impulse_response = impulse_response.roll(impulse_response.size(-1) // 2, -1)
    return impulse_response


def frequency_impulse_response(magnitudes, hann_window=True, half_width_frames=None):
    impulse_response = torch.fft.irfft(magnitudes)
    if hann_window:
        if half_width_frames is None:
            impulse_response = apply_window_to_impulse_response(impulse_response)
        else:
            impulse_response = apply_dynamic_window_to_impulse_response(impulse_response, half_width_frames)
    else:
        impulse_response = impulse_response.roll(impulse_response.size(-1) // 2, -1)
    return impulse_response


def frequency_filter(audio, magnitudes, hann_window=True, half_width_frames=None):
    impulse_response = frequency_impulse_response(magnitudes, hann_window, half_width_frames)
    return fft_convolve(audio, impulse_response)


def remove_above_fmax(amplitudes, pitch, fmax, level_start=1):
    n_harm = amplitudes.shape[-1]
    pitches = pitch * torch.arange(level_start, n_harm + level_start)
    aa = (pitches < fmax).float() + 1e-07
    return amplitudes * aa


def upsample(signal, factor):
    signal = signal.permute(0, 2, 1)
    signal = nn.functional.interpolate(torch.cat((signal, signal[:, :, -1:]), 2), size=signal.shape[-1] * factor + 1, mode='linear', align_corners=True)
    signal = signal[:, :, :-1]
    return signal.permute(0, 2, 1)


class Sins(torch.nn.Module):

    def __init__(self, sampling_rate, block_size, n_harmonics, n_mag_allpass, n_mag_noise, n_unit=256, n_spk=1):
        super().__init__()
        None
        self.register_buffer('sampling_rate', torch.tensor(sampling_rate))
        self.register_buffer('block_size', torch.tensor(block_size))
        split_map = {'amplitudes': n_harmonics, 'group_delay': n_mag_allpass, 'noise_magnitude': n_mag_noise}
        self.unit2ctrl = Unit2Control(n_unit, n_spk, split_map)

    def forward(self, units_frames, f0_frames, volume_frames, spk_id=None, spk_mix_dict=None, initial_phase=None, infer=True, max_upsample_dim=32):
        """
            units_frames: B x n_frames x n_unit
            f0_frames: B x n_frames x 1
            volume_frames: B x n_frames x 1
            spk_id: B x 1
        """
        f0 = upsample(f0_frames, self.block_size)
        if infer:
            x = torch.cumsum(f0.double() / self.sampling_rate, axis=1)
        else:
            x = torch.cumsum(f0 / self.sampling_rate, axis=1)
        if initial_phase is not None:
            x += initial_phase / 2 / np.pi
        x = x - torch.round(x)
        x = x
        phase = 2 * np.pi * x
        phase_frames = phase[:, ::self.block_size, :]
        ctrls, hidden = self.unit2ctrl(units_frames, f0_frames, phase_frames, volume_frames, spk_id=spk_id, spk_mix_dict=spk_mix_dict)
        amplitudes_frames = torch.exp(ctrls['amplitudes']) / 128
        group_delay = np.pi * torch.tanh(ctrls['group_delay'])
        noise_param = torch.exp(ctrls['noise_magnitude']) / 128
        amplitudes_frames = remove_above_fmax(amplitudes_frames, f0_frames, self.sampling_rate / 2, level_start=1)
        n_harmonic = amplitudes_frames.shape[-1]
        level_harmonic = torch.arange(1, n_harmonic + 1)
        sinusoids = 0.0
        for n in range((n_harmonic - 1) // max_upsample_dim + 1):
            start = n * max_upsample_dim
            end = (n + 1) * max_upsample_dim
            phases = phase * level_harmonic[start:end]
            amplitudes = upsample(amplitudes_frames[:, :, start:end], self.block_size)
            sinusoids += (torch.sin(phases) * amplitudes).sum(-1)
        harmonic = frequency_filter(sinusoids, torch.exp(1.0j * torch.cumsum(group_delay, axis=-1)), hann_window=False)
        noise = torch.rand_like(harmonic) * 2 - 1
        noise = frequency_filter(noise, torch.complex(noise_param, torch.zeros_like(noise_param)), hann_window=True)
        signal = harmonic + noise
        return signal, hidden, (harmonic, noise)


class CombSubSuperFast(torch.nn.Module):

    def __init__(self, sampling_rate, block_size, win_length, n_unit=256, n_spk=1, use_pitch_aug=False, pcmer_norm=False):
        super().__init__()
        None
        self.register_buffer('sampling_rate', torch.tensor(sampling_rate))
        self.register_buffer('block_size', torch.tensor(block_size))
        self.register_buffer('win_length', torch.tensor(win_length))
        self.register_buffer('window', torch.hann_window(win_length))
        split_map = {'harmonic_magnitude': win_length // 2 + 1, 'harmonic_phase': win_length // 2 + 1, 'noise_magnitude': win_length // 2 + 1, 'noise_phase': win_length // 2 + 1}
        self.unit2ctrl = Unit2Control(n_unit, n_spk, split_map, use_pitch_aug=use_pitch_aug, use_naive_v2=True, use_conv_stack=True)

    def fast_source_gen(self, f0_frames):
        n = torch.arange(self.block_size, device=f0_frames.device)
        s0 = f0_frames / self.sampling_rate
        ds0 = F.pad(s0[:, 1:, :] - s0[:, :-1, :], (0, 0, 0, 1))
        rad = s0 * (n + 1) + 0.5 * ds0 * n * (n + 1) / self.block_size
        s0 = s0 + ds0 * n / self.block_size
        rad2 = torch.fmod(rad[..., -1:].float() + 0.5, 1.0) - 0.5
        rad_acc = rad2.cumsum(dim=1).fmod(1.0)
        rad += F.pad(rad_acc[:, :-1, :], (0, 0, 1, 0))
        rad -= torch.round(rad)
        combtooth = torch.sinc(rad / (s0 + 1e-05)).reshape(f0_frames.shape[0], -1)
        phase_frames = 2 * np.pi * rad[:, :, :1]
        return combtooth, phase_frames

    def forward(self, units_frames, f0_frames, volume_frames, spk_id=None, spk_mix_dict=None, aug_shift=None, initial_phase=None, infer=True, **kwargs):
        """
            units_frames: B x n_frames x n_unit
            f0_frames: B x n_frames x 1
            volume_frames: B x n_frames x 1 
            spk_id: B x 1
        """
        combtooth, phase_frames = self.fast_source_gen(f0_frames)
        ctrls, hidden = self.unit2ctrl(units_frames, f0_frames, phase_frames, volume_frames, spk_id=spk_id, spk_mix_dict=spk_mix_dict, aug_shift=aug_shift)
        src_filter = torch.exp(ctrls['harmonic_magnitude'] + 1.0j * np.pi * ctrls['harmonic_phase'])
        src_filter = torch.cat((src_filter, src_filter[:, -1:, :]), 1)
        noise_filter = torch.exp(ctrls['noise_magnitude'] + 1.0j * np.pi * ctrls['noise_phase']) / 128
        noise_filter = torch.cat((noise_filter, noise_filter[:, -1:, :]), 1)
        if combtooth.shape[-1] > self.win_length // 2:
            pad_mode = 'reflect'
        else:
            pad_mode = 'constant'
        combtooth_stft = torch.stft(combtooth, n_fft=self.win_length, win_length=self.win_length, hop_length=self.block_size, window=self.window, center=True, return_complex=True, pad_mode=pad_mode)
        noise = torch.randn_like(combtooth)
        noise_stft = torch.stft(noise, n_fft=self.win_length, win_length=self.win_length, hop_length=self.block_size, window=self.window, center=True, return_complex=True, pad_mode=pad_mode)
        signal_stft = combtooth_stft * src_filter.permute(0, 2, 1) + noise_stft * noise_filter.permute(0, 2, 1)
        signal = torch.istft(signal_stft, n_fft=self.win_length, win_length=self.win_length, hop_length=self.block_size, window=self.window, center=True)
        return signal, hidden, (signal, signal)


class CombSubFast(torch.nn.Module):

    def __init__(self, sampling_rate, block_size, n_unit=256, n_spk=1, use_pitch_aug=False, pcmer_norm=False):
        super().__init__()
        None
        self.register_buffer('sampling_rate', torch.tensor(sampling_rate))
        self.register_buffer('block_size', torch.tensor(block_size))
        self.register_buffer('window', torch.sqrt(torch.hann_window(2 * block_size)))
        split_map = {'harmonic_magnitude': block_size + 1, 'harmonic_phase': block_size + 1, 'noise_magnitude': block_size + 1}
        self.unit2ctrl = Unit2Control(n_unit, n_spk, split_map, use_pitch_aug=use_pitch_aug, pcmer_norm=pcmer_norm)

    def forward(self, units_frames, f0_frames, volume_frames, spk_id=None, spk_mix_dict=None, aug_shift=None, initial_phase=None, infer=True, **kwargs):
        """
            units_frames: B x n_frames x n_unit
            f0_frames: B x n_frames x 1
            volume_frames: B x n_frames x 1 
            spk_id: B x 1
        """
        f0 = upsample(f0_frames, self.block_size)
        if infer:
            x = torch.cumsum(f0.double() / self.sampling_rate, axis=1)
        else:
            x = torch.cumsum(f0 / self.sampling_rate, axis=1)
        if initial_phase is not None:
            x += initial_phase / 2 / np.pi
        x = x - torch.round(x)
        x = x
        phase_frames = 2 * np.pi * x[:, ::self.block_size, :]
        ctrls, hidden = self.unit2ctrl(units_frames, f0_frames, phase_frames, volume_frames, spk_id=spk_id, spk_mix_dict=spk_mix_dict, aug_shift=aug_shift)
        src_filter = torch.exp(ctrls['harmonic_magnitude'] + 1.0j * np.pi * ctrls['harmonic_phase'])
        src_filter = torch.cat((src_filter, src_filter[:, -1:, :]), 1)
        noise_filter = torch.exp(ctrls['noise_magnitude']) / 128
        noise_filter = torch.cat((noise_filter, noise_filter[:, -1:, :]), 1)
        combtooth = torch.sinc(self.sampling_rate * x / (f0 + 0.001))
        combtooth = combtooth.squeeze(-1)
        combtooth_frames = F.pad(combtooth, (self.block_size, self.block_size)).unfold(1, 2 * self.block_size, self.block_size)
        combtooth_frames = combtooth_frames * self.window
        combtooth_fft = torch.fft.rfft(combtooth_frames, 2 * self.block_size)
        noise = torch.rand_like(combtooth) * 2 - 1
        noise_frames = F.pad(noise, (self.block_size, self.block_size)).unfold(1, 2 * self.block_size, self.block_size)
        noise_frames = noise_frames * self.window
        noise_fft = torch.fft.rfft(noise_frames, 2 * self.block_size)
        signal_fft = combtooth_fft * src_filter + noise_fft * noise_filter
        signal_frames_out = torch.fft.irfft(signal_fft, 2 * self.block_size) * self.window
        fold = torch.nn.Fold(output_size=(1, (signal_frames_out.size(1) + 1) * self.block_size), kernel_size=(1, 2 * self.block_size), stride=(1, self.block_size))
        signal = fold(signal_frames_out.transpose(1, 2))[:, 0, 0, self.block_size:-self.block_size]
        return signal, hidden, (signal, signal)


class CombSub(torch.nn.Module):

    def __init__(self, sampling_rate, block_size, n_mag_allpass, n_mag_harmonic, n_mag_noise, n_unit=256, n_spk=1):
        super().__init__()
        None
        self.register_buffer('sampling_rate', torch.tensor(sampling_rate))
        self.register_buffer('block_size', torch.tensor(block_size))
        split_map = {'group_delay': n_mag_allpass, 'harmonic_magnitude': n_mag_harmonic, 'noise_magnitude': n_mag_noise}
        self.unit2ctrl = Unit2Control(n_unit, n_spk, split_map)

    def forward(self, units_frames, f0_frames, volume_frames, spk_id=None, spk_mix_dict=None, initial_phase=None, infer=True, **kwargs):
        """
            units_frames: B x n_frames x n_unit
            f0_frames: B x n_frames x 1
            volume_frames: B x n_frames x 1 
            spk_id: B x 1
        """
        f0 = upsample(f0_frames, self.block_size)
        if infer:
            x = torch.cumsum(f0.double() / self.sampling_rate, axis=1)
        else:
            x = torch.cumsum(f0 / self.sampling_rate, axis=1)
        if initial_phase is not None:
            x += initial_phase / 2 / np.pi
        x = x - torch.round(x)
        x = x
        phase_frames = 2 * np.pi * x[:, ::self.block_size, :]
        ctrls, hidden = self.unit2ctrl(units_frames, f0_frames, phase_frames, volume_frames, spk_id=spk_id, spk_mix_dict=spk_mix_dict)
        group_delay = np.pi * torch.tanh(ctrls['group_delay'])
        src_param = torch.exp(ctrls['harmonic_magnitude'])
        noise_param = torch.exp(ctrls['noise_magnitude']) / 128
        combtooth = torch.sinc(self.sampling_rate * x / (f0 + 0.001))
        combtooth = combtooth.squeeze(-1)
        harmonic = frequency_filter(combtooth, torch.exp(1.0j * torch.cumsum(group_delay, axis=-1)), hann_window=False)
        harmonic = frequency_filter(harmonic, torch.complex(src_param, torch.zeros_like(src_param)), hann_window=True, half_width_frames=1.5 * self.sampling_rate / (f0_frames + 0.001))
        noise = torch.rand_like(harmonic) * 2 - 1
        noise = frequency_filter(noise, torch.complex(noise_param, torch.zeros_like(noise_param)), hann_window=True)
        signal = harmonic + noise
        return signal, hidden, (harmonic, noise)


class AfterDiffusion(nn.Module):

    def __init__(self, spec_max, spec_min, v_type='a'):
        super().__init__()
        self.spec_max = spec_max
        self.spec_min = spec_min
        self.type = v_type

    def forward(self, x):
        x = x.squeeze(1).permute(0, 2, 1)
        mel_out = (x + 1) / 2 * (self.spec_max - self.spec_min) + self.spec_min
        if self.type == 'nsf-hifigan-log10':
            mel_out = mel_out * 0.434294
        return mel_out.transpose(2, 1)


def expand_dims(v, dims):
    """
    Expand the tensor `v` to the dim `dims`.

    Args:
        `v`: a PyTorch tensor with shape [N].
        `dim`: a `int`.
    Returns:
        a PyTorch tensor with shape [N, 1, 1, ..., 1] and the total dimension is `dims`.
    """
    return v[(...,) + (None,) * (dims - 1)]


class DPM_Solver:

    def __init__(self, model_fn, noise_schedule, algorithm_type='dpmsolver++', correcting_x0_fn=None, correcting_xt_fn=None, thresholding_max_val=1.0, dynamic_thresholding_ratio=0.995):
        """Construct a DPM-Solver. 

        We support both DPM-Solver (`algorithm_type="dpmsolver"`) and DPM-Solver++ (`algorithm_type="dpmsolver++"`).

        We also support the "dynamic thresholding" method in Imagen[1]. For pixel-space diffusion models, you
        can set both `algorithm_type="dpmsolver++"` and `correcting_x0_fn="dynamic_thresholding"` to use the
        dynamic thresholding. The "dynamic thresholding" can greatly improve the sample quality for pixel-space
        DPMs with large guidance scales. Note that the thresholding method is **unsuitable** for latent-space
        DPMs (such as stable-diffusion).

        To support advanced algorithms in image-to-image applications, we also support corrector functions for
        both x0 and xt.

        Args:
            model_fn: A noise prediction model function which accepts the continuous-time input (t in [epsilon, T]):
                ``
                def model_fn(x, t_continuous):
                    return noise
                ``
                The shape of `x` is `(batch_size, **shape)`, and the shape of `t_continuous` is `(batch_size,)`.
            noise_schedule: A noise schedule object, such as NoiseScheduleVP.
            algorithm_type: A `str`. Either "dpmsolver" or "dpmsolver++".
            correcting_x0_fn: A `str` or a function with the following format:
                ```
                def correcting_x0_fn(x0, t):
                    x0_new = ...
                    return x0_new
                ```
                This function is to correct the outputs of the data prediction model at each sampling step. e.g.,
                ```
                x0_pred = data_pred_model(xt, t)
                if correcting_x0_fn is not None:
                    x0_pred = correcting_x0_fn(x0_pred, t)
                xt_1 = update(x0_pred, xt, t)
                ```
                If `correcting_x0_fn="dynamic_thresholding"`, we use the dynamic thresholding proposed in Imagen[1].
            correcting_xt_fn: A function with the following format:
                ```
                def correcting_xt_fn(xt, t, step):
                    x_new = ...
                    return x_new
                ```
                This function is to correct the intermediate samples xt at each sampling step. e.g.,
                ```
                xt = ...
                xt = correcting_xt_fn(xt, t, step)
                ```
            thresholding_max_val: A `float`. The max value for thresholding.
                Valid only when use `dpmsolver++` and `correcting_x0_fn="dynamic_thresholding"`.
            dynamic_thresholding_ratio: A `float`. The ratio for dynamic thresholding (see Imagen[1] for details).
                Valid only when use `dpmsolver++` and `correcting_x0_fn="dynamic_thresholding"`.

        [1] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily Denton, Seyed Kamyar Seyed Ghasemipour,
            Burcu Karagol Ayan, S Sara Mahdavi, Rapha Gontijo Lopes, et al. Photorealistic text-to-image diffusion models
            with deep language understanding. arXiv preprint arXiv:2205.11487, 2022b.
        """
        self.model = lambda x, t: model_fn(x, t.expand(x.shape[0]))
        self.noise_schedule = noise_schedule
        assert algorithm_type in ['dpmsolver', 'dpmsolver++']
        self.algorithm_type = algorithm_type
        if correcting_x0_fn == 'dynamic_thresholding':
            self.correcting_x0_fn = self.dynamic_thresholding_fn
        else:
            self.correcting_x0_fn = correcting_x0_fn
        self.correcting_xt_fn = correcting_xt_fn
        self.dynamic_thresholding_ratio = dynamic_thresholding_ratio
        self.thresholding_max_val = thresholding_max_val

    def dynamic_thresholding_fn(self, x0, t):
        """
        The dynamic thresholding method. 
        """
        dims = x0.dim()
        p = self.dynamic_thresholding_ratio
        s = torch.quantile(torch.abs(x0).reshape((x0.shape[0], -1)), p, dim=1)
        s = expand_dims(torch.maximum(s, self.thresholding_max_val * torch.ones_like(s)), dims)
        x0 = torch.clamp(x0, -s, s) / s
        return x0

    def noise_prediction_fn(self, x, t):
        """
        Return the noise prediction model.
        """
        return self.model(x, t)

    def data_prediction_fn(self, x, t):
        """
        Return the data prediction model (with corrector).
        """
        noise = self.noise_prediction_fn(x, t)
        alpha_t, sigma_t = self.noise_schedule.marginal_alpha(t), self.noise_schedule.marginal_std(t)
        x0 = (x - sigma_t * noise) / alpha_t
        if self.correcting_x0_fn is not None:
            x0 = self.correcting_x0_fn(x0, t)
        return x0

    def model_fn(self, x, t):
        """
        Convert the model to the noise prediction model or the data prediction model. 
        """
        if self.algorithm_type == 'dpmsolver++':
            return self.data_prediction_fn(x, t)
        else:
            return self.noise_prediction_fn(x, t)

    def get_time_steps(self, skip_type, t_T, t_0, N, device):
        """Compute the intermediate time steps for sampling.

        Args:
            skip_type: A `str`. The type for the spacing of the time steps. We support three types:
                - 'logSNR': uniform logSNR for the time steps.
                - 'time_uniform': uniform time for the time steps. (**Recommended for high-resolutional data**.)
                - 'time_quadratic': quadratic time for the time steps. (Used in DDIM for low-resolutional data.)
            t_T: A `float`. The starting time of the sampling (default is T).
            t_0: A `float`. The ending time of the sampling (default is epsilon).
            N: A `int`. The total number of the spacing of the time steps.
            device: A torch device.
        Returns:
            A pytorch tensor of the time steps, with the shape (N + 1,).
        """
        if skip_type == 'logSNR':
            lambda_T = self.noise_schedule.marginal_lambda(torch.tensor(t_T))
            lambda_0 = self.noise_schedule.marginal_lambda(torch.tensor(t_0))
            logSNR_steps = torch.linspace(lambda_T.cpu().item(), lambda_0.cpu().item(), N + 1)
            return self.noise_schedule.inverse_lambda(logSNR_steps)
        elif skip_type == 'time_uniform':
            return torch.linspace(t_T, t_0, N + 1)
        elif skip_type == 'time_quadratic':
            t_order = 2
            t = torch.linspace(t_T ** (1.0 / t_order), t_0 ** (1.0 / t_order), N + 1).pow(t_order)
            return t
        else:
            raise ValueError("Unsupported skip_type {}, need to be 'logSNR' or 'time_uniform' or 'time_quadratic'".format(skip_type))

    def get_orders_and_timesteps_for_singlestep_solver(self, steps, order, skip_type, t_T, t_0, device):
        """
        Get the order of each step for sampling by the singlestep DPM-Solver.

        We combine both DPM-Solver-1,2,3 to use all the function evaluations, which is named as "DPM-Solver-fast".
        Given a fixed number of function evaluations by `steps`, the sampling procedure by DPM-Solver-fast is:
            - If order == 1:
                We take `steps` of DPM-Solver-1 (i.e. DDIM).
            - If order == 2:
                - Denote K = (steps // 2). We take K or (K + 1) intermediate time steps for sampling.
                - If steps % 2 == 0, we use K steps of DPM-Solver-2.
                - If steps % 2 == 1, we use K steps of DPM-Solver-2 and 1 step of DPM-Solver-1.
            - If order == 3:
                - Denote K = (steps // 3 + 1). We take K intermediate time steps for sampling.
                - If steps % 3 == 0, we use (K - 2) steps of DPM-Solver-3, and 1 step of DPM-Solver-2 and 1 step of DPM-Solver-1.
                - If steps % 3 == 1, we use (K - 1) steps of DPM-Solver-3 and 1 step of DPM-Solver-1.
                - If steps % 3 == 2, we use (K - 1) steps of DPM-Solver-3 and 1 step of DPM-Solver-2.

        ============================================
        Args:
            order: A `int`. The max order for the solver (2 or 3).
            steps: A `int`. The total number of function evaluations (NFE).
            skip_type: A `str`. The type for the spacing of the time steps. We support three types:
                - 'logSNR': uniform logSNR for the time steps.
                - 'time_uniform': uniform time for the time steps. (**Recommended for high-resolutional data**.)
                - 'time_quadratic': quadratic time for the time steps. (Used in DDIM for low-resolutional data.)
            t_T: A `float`. The starting time of the sampling (default is T).
            t_0: A `float`. The ending time of the sampling (default is epsilon).
            device: A torch device.
        Returns:
            orders: A list of the solver order of each step.
        """
        if order == 3:
            K = steps // 3 + 1
            if steps % 3 == 0:
                orders = [3] * (K - 2) + [2, 1]
            elif steps % 3 == 1:
                orders = [3] * (K - 1) + [1]
            else:
                orders = [3] * (K - 1) + [2]
        elif order == 2:
            if steps % 2 == 0:
                K = steps // 2
                orders = [2] * K
            else:
                K = steps // 2 + 1
                orders = [2] * (K - 1) + [1]
        elif order == 1:
            K = 1
            orders = [1] * steps
        else:
            raise ValueError("'order' must be '1' or '2' or '3'.")
        if skip_type == 'logSNR':
            timesteps_outer = self.get_time_steps(skip_type, t_T, t_0, K, device)
        else:
            timesteps_outer = self.get_time_steps(skip_type, t_T, t_0, steps, device)[torch.cumsum(torch.tensor([0] + orders), 0)]
        return timesteps_outer, orders

    def denoise_to_zero_fn(self, x, s):
        """
        Denoise at the final step, which is equivalent to solve the ODE from lambda_s to infty by first-order discretization. 
        """
        return self.data_prediction_fn(x, s)

    def dpm_solver_first_update(self, x, s, t, model_s=None, return_intermediate=False):
        """
        DPM-Solver-1 (equivalent to DDIM) from time `s` to time `t`.

        Args:
            x: A pytorch tensor. The initial value at time `s`.
            s: A pytorch tensor. The starting time, with the shape (1,).
            t: A pytorch tensor. The ending time, with the shape (1,).
            model_s: A pytorch tensor. The model function evaluated at time `s`.
                If `model_s` is None, we evaluate the model by `x` and `s`; otherwise we directly use it.
            return_intermediate: A `bool`. If true, also return the model value at time `s`.
        Returns:
            x_t: A pytorch tensor. The approximated solution at time `t`.
        """
        ns = self.noise_schedule
        dims = x.dim()
        lambda_s, lambda_t = ns.marginal_lambda(s), ns.marginal_lambda(t)
        h = lambda_t - lambda_s
        log_alpha_s, log_alpha_t = ns.marginal_log_mean_coeff(s), ns.marginal_log_mean_coeff(t)
        sigma_s, sigma_t = ns.marginal_std(s), ns.marginal_std(t)
        alpha_t = torch.exp(log_alpha_t)
        if self.algorithm_type == 'dpmsolver++':
            phi_1 = torch.expm1(-h)
            if model_s is None:
                model_s = self.model_fn(x, s)
            x_t = sigma_t / sigma_s * x - alpha_t * phi_1 * model_s
            if return_intermediate:
                return x_t, {'model_s': model_s}
            else:
                return x_t
        else:
            phi_1 = torch.expm1(h)
            if model_s is None:
                model_s = self.model_fn(x, s)
            x_t = torch.exp(log_alpha_t - log_alpha_s) * x - sigma_t * phi_1 * model_s
            if return_intermediate:
                return x_t, {'model_s': model_s}
            else:
                return x_t

    def singlestep_dpm_solver_second_update(self, x, s, t, r1=0.5, model_s=None, return_intermediate=False, solver_type='dpmsolver'):
        """
        Singlestep solver DPM-Solver-2 from time `s` to time `t`.

        Args:
            x: A pytorch tensor. The initial value at time `s`.
            s: A pytorch tensor. The starting time, with the shape (1,).
            t: A pytorch tensor. The ending time, with the shape (1,).
            r1: A `float`. The hyperparameter of the second-order solver.
            model_s: A pytorch tensor. The model function evaluated at time `s`.
                If `model_s` is None, we evaluate the model by `x` and `s`; otherwise we directly use it.
            return_intermediate: A `bool`. If true, also return the model value at time `s` and `s1` (the intermediate time).
            solver_type: either 'dpmsolver' or 'taylor'. The type for the high-order solvers.
                The type slightly impacts the performance. We recommend to use 'dpmsolver' type.
        Returns:
            x_t: A pytorch tensor. The approximated solution at time `t`.
        """
        if solver_type not in ['dpmsolver', 'taylor']:
            raise ValueError("'solver_type' must be either 'dpmsolver' or 'taylor', got {}".format(solver_type))
        if r1 is None:
            r1 = 0.5
        ns = self.noise_schedule
        lambda_s, lambda_t = ns.marginal_lambda(s), ns.marginal_lambda(t)
        h = lambda_t - lambda_s
        lambda_s1 = lambda_s + r1 * h
        s1 = ns.inverse_lambda(lambda_s1)
        log_alpha_s, log_alpha_s1, log_alpha_t = ns.marginal_log_mean_coeff(s), ns.marginal_log_mean_coeff(s1), ns.marginal_log_mean_coeff(t)
        sigma_s, sigma_s1, sigma_t = ns.marginal_std(s), ns.marginal_std(s1), ns.marginal_std(t)
        alpha_s1, alpha_t = torch.exp(log_alpha_s1), torch.exp(log_alpha_t)
        if self.algorithm_type == 'dpmsolver++':
            phi_11 = torch.expm1(-r1 * h)
            phi_1 = torch.expm1(-h)
            if model_s is None:
                model_s = self.model_fn(x, s)
            x_s1 = sigma_s1 / sigma_s * x - alpha_s1 * phi_11 * model_s
            model_s1 = self.model_fn(x_s1, s1)
            if solver_type == 'dpmsolver':
                x_t = sigma_t / sigma_s * x - alpha_t * phi_1 * model_s - 0.5 / r1 * (alpha_t * phi_1) * (model_s1 - model_s)
            elif solver_type == 'taylor':
                x_t = sigma_t / sigma_s * x - alpha_t * phi_1 * model_s + 1.0 / r1 * (alpha_t * (phi_1 / h + 1.0)) * (model_s1 - model_s)
        else:
            phi_11 = torch.expm1(r1 * h)
            phi_1 = torch.expm1(h)
            if model_s is None:
                model_s = self.model_fn(x, s)
            x_s1 = torch.exp(log_alpha_s1 - log_alpha_s) * x - sigma_s1 * phi_11 * model_s
            model_s1 = self.model_fn(x_s1, s1)
            if solver_type == 'dpmsolver':
                x_t = torch.exp(log_alpha_t - log_alpha_s) * x - sigma_t * phi_1 * model_s - 0.5 / r1 * (sigma_t * phi_1) * (model_s1 - model_s)
            elif solver_type == 'taylor':
                x_t = torch.exp(log_alpha_t - log_alpha_s) * x - sigma_t * phi_1 * model_s - 1.0 / r1 * (sigma_t * (phi_1 / h - 1.0)) * (model_s1 - model_s)
        if return_intermediate:
            return x_t, {'model_s': model_s, 'model_s1': model_s1}
        else:
            return x_t

    def singlestep_dpm_solver_third_update(self, x, s, t, r1=1.0 / 3.0, r2=2.0 / 3.0, model_s=None, model_s1=None, return_intermediate=False, solver_type='dpmsolver'):
        """
        Singlestep solver DPM-Solver-3 from time `s` to time `t`.

        Args:
            x: A pytorch tensor. The initial value at time `s`.
            s: A pytorch tensor. The starting time, with the shape (1,).
            t: A pytorch tensor. The ending time, with the shape (1,).
            r1: A `float`. The hyperparameter of the third-order solver.
            r2: A `float`. The hyperparameter of the third-order solver.
            model_s: A pytorch tensor. The model function evaluated at time `s`.
                If `model_s` is None, we evaluate the model by `x` and `s`; otherwise we directly use it.
            model_s1: A pytorch tensor. The model function evaluated at time `s1` (the intermediate time given by `r1`).
                If `model_s1` is None, we evaluate the model at `s1`; otherwise we directly use it.
            return_intermediate: A `bool`. If true, also return the model value at time `s`, `s1` and `s2` (the intermediate times).
            solver_type: either 'dpmsolver' or 'taylor'. The type for the high-order solvers.
                The type slightly impacts the performance. We recommend to use 'dpmsolver' type.
        Returns:
            x_t: A pytorch tensor. The approximated solution at time `t`.
        """
        if solver_type not in ['dpmsolver', 'taylor']:
            raise ValueError("'solver_type' must be either 'dpmsolver' or 'taylor', got {}".format(solver_type))
        if r1 is None:
            r1 = 1.0 / 3.0
        if r2 is None:
            r2 = 2.0 / 3.0
        ns = self.noise_schedule
        lambda_s, lambda_t = ns.marginal_lambda(s), ns.marginal_lambda(t)
        h = lambda_t - lambda_s
        lambda_s1 = lambda_s + r1 * h
        lambda_s2 = lambda_s + r2 * h
        s1 = ns.inverse_lambda(lambda_s1)
        s2 = ns.inverse_lambda(lambda_s2)
        log_alpha_s, log_alpha_s1, log_alpha_s2, log_alpha_t = ns.marginal_log_mean_coeff(s), ns.marginal_log_mean_coeff(s1), ns.marginal_log_mean_coeff(s2), ns.marginal_log_mean_coeff(t)
        sigma_s, sigma_s1, sigma_s2, sigma_t = ns.marginal_std(s), ns.marginal_std(s1), ns.marginal_std(s2), ns.marginal_std(t)
        alpha_s1, alpha_s2, alpha_t = torch.exp(log_alpha_s1), torch.exp(log_alpha_s2), torch.exp(log_alpha_t)
        if self.algorithm_type == 'dpmsolver++':
            phi_11 = torch.expm1(-r1 * h)
            phi_12 = torch.expm1(-r2 * h)
            phi_1 = torch.expm1(-h)
            phi_22 = torch.expm1(-r2 * h) / (r2 * h) + 1.0
            phi_2 = phi_1 / h + 1.0
            phi_3 = phi_2 / h - 0.5
            if model_s is None:
                model_s = self.model_fn(x, s)
            if model_s1 is None:
                x_s1 = sigma_s1 / sigma_s * x - alpha_s1 * phi_11 * model_s
                model_s1 = self.model_fn(x_s1, s1)
            x_s2 = sigma_s2 / sigma_s * x - alpha_s2 * phi_12 * model_s + r2 / r1 * (alpha_s2 * phi_22) * (model_s1 - model_s)
            model_s2 = self.model_fn(x_s2, s2)
            if solver_type == 'dpmsolver':
                x_t = sigma_t / sigma_s * x - alpha_t * phi_1 * model_s + 1.0 / r2 * (alpha_t * phi_2) * (model_s2 - model_s)
            elif solver_type == 'taylor':
                D1_0 = 1.0 / r1 * (model_s1 - model_s)
                D1_1 = 1.0 / r2 * (model_s2 - model_s)
                D1 = (r2 * D1_0 - r1 * D1_1) / (r2 - r1)
                D2 = 2.0 * (D1_1 - D1_0) / (r2 - r1)
                x_t = sigma_t / sigma_s * x - alpha_t * phi_1 * model_s + alpha_t * phi_2 * D1 - alpha_t * phi_3 * D2
        else:
            phi_11 = torch.expm1(r1 * h)
            phi_12 = torch.expm1(r2 * h)
            phi_1 = torch.expm1(h)
            phi_22 = torch.expm1(r2 * h) / (r2 * h) - 1.0
            phi_2 = phi_1 / h - 1.0
            phi_3 = phi_2 / h - 0.5
            if model_s is None:
                model_s = self.model_fn(x, s)
            if model_s1 is None:
                x_s1 = torch.exp(log_alpha_s1 - log_alpha_s) * x - sigma_s1 * phi_11 * model_s
                model_s1 = self.model_fn(x_s1, s1)
            x_s2 = torch.exp(log_alpha_s2 - log_alpha_s) * x - sigma_s2 * phi_12 * model_s - r2 / r1 * (sigma_s2 * phi_22) * (model_s1 - model_s)
            model_s2 = self.model_fn(x_s2, s2)
            if solver_type == 'dpmsolver':
                x_t = torch.exp(log_alpha_t - log_alpha_s) * x - sigma_t * phi_1 * model_s - 1.0 / r2 * (sigma_t * phi_2) * (model_s2 - model_s)
            elif solver_type == 'taylor':
                D1_0 = 1.0 / r1 * (model_s1 - model_s)
                D1_1 = 1.0 / r2 * (model_s2 - model_s)
                D1 = (r2 * D1_0 - r1 * D1_1) / (r2 - r1)
                D2 = 2.0 * (D1_1 - D1_0) / (r2 - r1)
                x_t = torch.exp(log_alpha_t - log_alpha_s) * x - sigma_t * phi_1 * model_s - sigma_t * phi_2 * D1 - sigma_t * phi_3 * D2
        if return_intermediate:
            return x_t, {'model_s': model_s, 'model_s1': model_s1, 'model_s2': model_s2}
        else:
            return x_t

    def multistep_dpm_solver_second_update(self, x, model_prev_list, t_prev_list, t, solver_type='dpmsolver'):
        """
        Multistep solver DPM-Solver-2 from time `t_prev_list[-1]` to time `t`.

        Args:
            x: A pytorch tensor. The initial value at time `s`.
            model_prev_list: A list of pytorch tensor. The previous computed model values.
            t_prev_list: A list of pytorch tensor. The previous times, each time has the shape (1,)
            t: A pytorch tensor. The ending time, with the shape (1,).
            solver_type: either 'dpmsolver' or 'taylor'. The type for the high-order solvers.
                The type slightly impacts the performance. We recommend to use 'dpmsolver' type.
        Returns:
            x_t: A pytorch tensor. The approximated solution at time `t`.
        """
        if solver_type not in ['dpmsolver', 'taylor']:
            raise ValueError("'solver_type' must be either 'dpmsolver' or 'taylor', got {}".format(solver_type))
        ns = self.noise_schedule
        model_prev_1, model_prev_0 = model_prev_list[-2], model_prev_list[-1]
        t_prev_1, t_prev_0 = t_prev_list[-2], t_prev_list[-1]
        lambda_prev_1, lambda_prev_0, lambda_t = ns.marginal_lambda(t_prev_1), ns.marginal_lambda(t_prev_0), ns.marginal_lambda(t)
        log_alpha_prev_0, log_alpha_t = ns.marginal_log_mean_coeff(t_prev_0), ns.marginal_log_mean_coeff(t)
        sigma_prev_0, sigma_t = ns.marginal_std(t_prev_0), ns.marginal_std(t)
        alpha_t = torch.exp(log_alpha_t)
        h_0 = lambda_prev_0 - lambda_prev_1
        h = lambda_t - lambda_prev_0
        r0 = h_0 / h
        D1_0 = 1.0 / r0 * (model_prev_0 - model_prev_1)
        if self.algorithm_type == 'dpmsolver++':
            phi_1 = torch.expm1(-h)
            if solver_type == 'dpmsolver':
                x_t = sigma_t / sigma_prev_0 * x - alpha_t * phi_1 * model_prev_0 - 0.5 * (alpha_t * phi_1) * D1_0
            elif solver_type == 'taylor':
                x_t = sigma_t / sigma_prev_0 * x - alpha_t * phi_1 * model_prev_0 + alpha_t * (phi_1 / h + 1.0) * D1_0
        else:
            phi_1 = torch.expm1(h)
            if solver_type == 'dpmsolver':
                x_t = torch.exp(log_alpha_t - log_alpha_prev_0) * x - sigma_t * phi_1 * model_prev_0 - 0.5 * (sigma_t * phi_1) * D1_0
            elif solver_type == 'taylor':
                x_t = torch.exp(log_alpha_t - log_alpha_prev_0) * x - sigma_t * phi_1 * model_prev_0 - sigma_t * (phi_1 / h - 1.0) * D1_0
        return x_t

    def multistep_dpm_solver_third_update(self, x, model_prev_list, t_prev_list, t, solver_type='dpmsolver'):
        """
        Multistep solver DPM-Solver-3 from time `t_prev_list[-1]` to time `t`.

        Args:
            x: A pytorch tensor. The initial value at time `s`.
            model_prev_list: A list of pytorch tensor. The previous computed model values.
            t_prev_list: A list of pytorch tensor. The previous times, each time has the shape (1,)
            t: A pytorch tensor. The ending time, with the shape (1,).
            solver_type: either 'dpmsolver' or 'taylor'. The type for the high-order solvers.
                The type slightly impacts the performance. We recommend to use 'dpmsolver' type.
        Returns:
            x_t: A pytorch tensor. The approximated solution at time `t`.
        """
        ns = self.noise_schedule
        model_prev_2, model_prev_1, model_prev_0 = model_prev_list
        t_prev_2, t_prev_1, t_prev_0 = t_prev_list
        lambda_prev_2, lambda_prev_1, lambda_prev_0, lambda_t = ns.marginal_lambda(t_prev_2), ns.marginal_lambda(t_prev_1), ns.marginal_lambda(t_prev_0), ns.marginal_lambda(t)
        log_alpha_prev_0, log_alpha_t = ns.marginal_log_mean_coeff(t_prev_0), ns.marginal_log_mean_coeff(t)
        sigma_prev_0, sigma_t = ns.marginal_std(t_prev_0), ns.marginal_std(t)
        alpha_t = torch.exp(log_alpha_t)
        h_1 = lambda_prev_1 - lambda_prev_2
        h_0 = lambda_prev_0 - lambda_prev_1
        h = lambda_t - lambda_prev_0
        r0, r1 = h_0 / h, h_1 / h
        D1_0 = 1.0 / r0 * (model_prev_0 - model_prev_1)
        D1_1 = 1.0 / r1 * (model_prev_1 - model_prev_2)
        D1 = D1_0 + r0 / (r0 + r1) * (D1_0 - D1_1)
        D2 = 1.0 / (r0 + r1) * (D1_0 - D1_1)
        if self.algorithm_type == 'dpmsolver++':
            phi_1 = torch.expm1(-h)
            phi_2 = phi_1 / h + 1.0
            phi_3 = phi_2 / h - 0.5
            x_t = sigma_t / sigma_prev_0 * x - alpha_t * phi_1 * model_prev_0 + alpha_t * phi_2 * D1 - alpha_t * phi_3 * D2
        else:
            phi_1 = torch.expm1(h)
            phi_2 = phi_1 / h - 1.0
            phi_3 = phi_2 / h - 0.5
            x_t = torch.exp(log_alpha_t - log_alpha_prev_0) * x - sigma_t * phi_1 * model_prev_0 - sigma_t * phi_2 * D1 - sigma_t * phi_3 * D2
        return x_t

    def singlestep_dpm_solver_update(self, x, s, t, order, return_intermediate=False, solver_type='dpmsolver', r1=None, r2=None):
        """
        Singlestep DPM-Solver with the order `order` from time `s` to time `t`.

        Args:
            x: A pytorch tensor. The initial value at time `s`.
            s: A pytorch tensor. The starting time, with the shape (1,).
            t: A pytorch tensor. The ending time, with the shape (1,).
            order: A `int`. The order of DPM-Solver. We only support order == 1 or 2 or 3.
            return_intermediate: A `bool`. If true, also return the model value at time `s`, `s1` and `s2` (the intermediate times).
            solver_type: either 'dpmsolver' or 'taylor'. The type for the high-order solvers.
                The type slightly impacts the performance. We recommend to use 'dpmsolver' type.
            r1: A `float`. The hyperparameter of the second-order or third-order solver.
            r2: A `float`. The hyperparameter of the third-order solver.
        Returns:
            x_t: A pytorch tensor. The approximated solution at time `t`.
        """
        if order == 1:
            return self.dpm_solver_first_update(x, s, t, return_intermediate=return_intermediate)
        elif order == 2:
            return self.singlestep_dpm_solver_second_update(x, s, t, return_intermediate=return_intermediate, solver_type=solver_type, r1=r1)
        elif order == 3:
            return self.singlestep_dpm_solver_third_update(x, s, t, return_intermediate=return_intermediate, solver_type=solver_type, r1=r1, r2=r2)
        else:
            raise ValueError('Solver order must be 1 or 2 or 3, got {}'.format(order))

    def multistep_dpm_solver_update(self, x, model_prev_list, t_prev_list, t, order, solver_type='dpmsolver'):
        """
        Multistep DPM-Solver with the order `order` from time `t_prev_list[-1]` to time `t`.

        Args:
            x: A pytorch tensor. The initial value at time `s`.
            model_prev_list: A list of pytorch tensor. The previous computed model values.
            t_prev_list: A list of pytorch tensor. The previous times, each time has the shape (1,)
            t: A pytorch tensor. The ending time, with the shape (1,).
            order: A `int`. The order of DPM-Solver. We only support order == 1 or 2 or 3.
            solver_type: either 'dpmsolver' or 'taylor'. The type for the high-order solvers.
                The type slightly impacts the performance. We recommend to use 'dpmsolver' type.
        Returns:
            x_t: A pytorch tensor. The approximated solution at time `t`.
        """
        if order == 1:
            return self.dpm_solver_first_update(x, t_prev_list[-1], t, model_s=model_prev_list[-1])
        elif order == 2:
            return self.multistep_dpm_solver_second_update(x, model_prev_list, t_prev_list, t, solver_type=solver_type)
        elif order == 3:
            return self.multistep_dpm_solver_third_update(x, model_prev_list, t_prev_list, t, solver_type=solver_type)
        else:
            raise ValueError('Solver order must be 1 or 2 or 3, got {}'.format(order))

    def dpm_solver_adaptive(self, x, order, t_T, t_0, h_init=0.05, atol=0.0078, rtol=0.05, theta=0.9, t_err=1e-05, solver_type='dpmsolver'):
        """
        The adaptive step size solver based on singlestep DPM-Solver.

        Args:
            x: A pytorch tensor. The initial value at time `t_T`.
            order: A `int`. The (higher) order of the solver. We only support order == 2 or 3.
            t_T: A `float`. The starting time of the sampling (default is T).
            t_0: A `float`. The ending time of the sampling (default is epsilon).
            h_init: A `float`. The initial step size (for logSNR).
            atol: A `float`. The absolute tolerance of the solver. For image data, the default setting is 0.0078, followed [1].
            rtol: A `float`. The relative tolerance of the solver. The default setting is 0.05.
            theta: A `float`. The safety hyperparameter for adapting the step size. The default setting is 0.9, followed [1].
            t_err: A `float`. The tolerance for the time. We solve the diffusion ODE until the absolute error between the 
                current time and `t_0` is less than `t_err`. The default setting is 1e-5.
            solver_type: either 'dpmsolver' or 'taylor'. The type for the high-order solvers.
                The type slightly impacts the performance. We recommend to use 'dpmsolver' type.
        Returns:
            x_0: A pytorch tensor. The approximated solution at time `t_0`.

        [1] A. Jolicoeur-Martineau, K. Li, R. Piché-Taillefer, T. Kachman, and I. Mitliagkas, "Gotta go fast when generating data with score-based models," arXiv preprint arXiv:2105.14080, 2021.
        """
        ns = self.noise_schedule
        s = t_T * torch.ones((1,))
        lambda_s = ns.marginal_lambda(s)
        lambda_0 = ns.marginal_lambda(t_0 * torch.ones_like(s))
        h = h_init * torch.ones_like(s)
        x_prev = x
        nfe = 0
        if order == 2:
            r1 = 0.5
            lower_update = lambda x, s, t: self.dpm_solver_first_update(x, s, t, return_intermediate=True)
            higher_update = lambda x, s, t, **kwargs: self.singlestep_dpm_solver_second_update(x, s, t, r1=r1, solver_type=solver_type, **kwargs)
        elif order == 3:
            r1, r2 = 1.0 / 3.0, 2.0 / 3.0
            lower_update = lambda x, s, t: self.singlestep_dpm_solver_second_update(x, s, t, r1=r1, return_intermediate=True, solver_type=solver_type)
            higher_update = lambda x, s, t, **kwargs: self.singlestep_dpm_solver_third_update(x, s, t, r1=r1, r2=r2, solver_type=solver_type, **kwargs)
        else:
            raise ValueError('For adaptive step size solver, order must be 2 or 3, got {}'.format(order))
        while torch.abs(s - t_0).mean() > t_err:
            t = ns.inverse_lambda(lambda_s + h)
            x_lower, lower_noise_kwargs = lower_update(x, s, t)
            x_higher = higher_update(x, s, t, **lower_noise_kwargs)
            delta = torch.max(torch.ones_like(x) * atol, rtol * torch.max(torch.abs(x_lower), torch.abs(x_prev)))
            norm_fn = lambda v: torch.sqrt(torch.square(v.reshape((v.shape[0], -1))).mean(dim=-1, keepdim=True))
            E = norm_fn((x_higher - x_lower) / delta).max()
            if torch.all(E <= 1.0):
                x = x_higher
                s = t
                x_prev = x_lower
                lambda_s = ns.marginal_lambda(s)
            h = torch.min(theta * h * torch.float_power(E, -1.0 / order).float(), lambda_0 - lambda_s)
            nfe += order
        None
        return x

    def add_noise(self, x, t, noise=None):
        """
        Compute the noised input xt = alpha_t * x + sigma_t * noise. 

        Args:
            x: A `torch.Tensor` with shape `(batch_size, *shape)`.
            t: A `torch.Tensor` with shape `(t_size,)`.
        Returns:
            xt with shape `(t_size, batch_size, *shape)`.
        """
        alpha_t, sigma_t = self.noise_schedule.marginal_alpha(t), self.noise_schedule.marginal_std(t)
        if noise is None:
            noise = torch.randn((t.shape[0], *x.shape), device=x.device)
        x = x.reshape((-1, *x.shape))
        xt = expand_dims(alpha_t, x.dim()) * x + expand_dims(sigma_t, x.dim()) * noise
        if t.shape[0] == 1:
            return xt.squeeze(0)
        else:
            return xt

    def inverse(self, x, steps=20, t_start=None, t_end=None, order=2, skip_type='time_uniform', method='multistep', lower_order_final=True, denoise_to_zero=False, solver_type='dpmsolver', atol=0.0078, rtol=0.05, return_intermediate=False):
        """
        Inverse the sample `x` from time `t_start` to `t_end` by DPM-Solver.
        For discrete-time DPMs, we use `t_start=1/N`, where `N` is the total time steps during training.
        """
        t_0 = 1.0 / self.noise_schedule.total_N if t_start is None else t_start
        t_T = self.noise_schedule.T if t_end is None else t_end
        assert t_0 > 0 and t_T > 0, 'Time range needs to be greater than 0. For discrete-time DPMs, it needs to be in [1 / N, 1], where N is the length of betas array'
        return self.sample(x, steps=steps, t_start=t_0, t_end=t_T, order=order, skip_type=skip_type, method=method, lower_order_final=lower_order_final, denoise_to_zero=denoise_to_zero, solver_type=solver_type, atol=atol, rtol=rtol, return_intermediate=return_intermediate)

    def sample(self, x, steps=20, t_start=None, t_end=None, order=2, skip_type='time_uniform', method='multistep', lower_order_final=True, denoise_to_zero=False, solver_type='dpmsolver', atol=0.0078, rtol=0.05, return_intermediate=False):
        """
        Compute the sample at time `t_end` by DPM-Solver, given the initial `x` at time `t_start`.

        =====================================================

        We support the following algorithms for both noise prediction model and data prediction model:
            - 'singlestep':
                Singlestep DPM-Solver (i.e. "DPM-Solver-fast" in the paper), which combines different orders of singlestep DPM-Solver. 
                We combine all the singlestep solvers with order <= `order` to use up all the function evaluations (steps).
                The total number of function evaluations (NFE) == `steps`.
                Given a fixed NFE == `steps`, the sampling procedure is:
                    - If `order` == 1:
                        - Denote K = steps. We use K steps of DPM-Solver-1 (i.e. DDIM).
                    - If `order` == 2:
                        - Denote K = (steps // 2) + (steps % 2). We take K intermediate time steps for sampling.
                        - If steps % 2 == 0, we use K steps of singlestep DPM-Solver-2.
                        - If steps % 2 == 1, we use (K - 1) steps of singlestep DPM-Solver-2 and 1 step of DPM-Solver-1.
                    - If `order` == 3:
                        - Denote K = (steps // 3 + 1). We take K intermediate time steps for sampling.
                        - If steps % 3 == 0, we use (K - 2) steps of singlestep DPM-Solver-3, and 1 step of singlestep DPM-Solver-2 and 1 step of DPM-Solver-1.
                        - If steps % 3 == 1, we use (K - 1) steps of singlestep DPM-Solver-3 and 1 step of DPM-Solver-1.
                        - If steps % 3 == 2, we use (K - 1) steps of singlestep DPM-Solver-3 and 1 step of singlestep DPM-Solver-2.
            - 'multistep':
                Multistep DPM-Solver with the order of `order`. The total number of function evaluations (NFE) == `steps`.
                We initialize the first `order` values by lower order multistep solvers.
                Given a fixed NFE == `steps`, the sampling procedure is:
                    Denote K = steps.
                    - If `order` == 1:
                        - We use K steps of DPM-Solver-1 (i.e. DDIM).
                    - If `order` == 2:
                        - We firstly use 1 step of DPM-Solver-1, then use (K - 1) step of multistep DPM-Solver-2.
                    - If `order` == 3:
                        - We firstly use 1 step of DPM-Solver-1, then 1 step of multistep DPM-Solver-2, then (K - 2) step of multistep DPM-Solver-3.
            - 'singlestep_fixed':
                Fixed order singlestep DPM-Solver (i.e. DPM-Solver-1 or singlestep DPM-Solver-2 or singlestep DPM-Solver-3).
                We use singlestep DPM-Solver-`order` for `order`=1 or 2 or 3, with total [`steps` // `order`] * `order` NFE.
            - 'adaptive':
                Adaptive step size DPM-Solver (i.e. "DPM-Solver-12" and "DPM-Solver-23" in the paper).
                We ignore `steps` and use adaptive step size DPM-Solver with a higher order of `order`.
                You can adjust the absolute tolerance `atol` and the relative tolerance `rtol` to balance the computatation costs
                (NFE) and the sample quality.
                    - If `order` == 2, we use DPM-Solver-12 which combines DPM-Solver-1 and singlestep DPM-Solver-2.
                    - If `order` == 3, we use DPM-Solver-23 which combines singlestep DPM-Solver-2 and singlestep DPM-Solver-3.

        =====================================================

        Some advices for choosing the algorithm:
            - For **unconditional sampling** or **guided sampling with small guidance scale** by DPMs:
                Use singlestep DPM-Solver or DPM-Solver++ ("DPM-Solver-fast" in the paper) with `order = 3`.
                e.g., DPM-Solver:
                    >>> dpm_solver = DPM_Solver(model_fn, noise_schedule, algorithm_type="dpmsolver")
                    >>> x_sample = dpm_solver.sample(x, steps=steps, t_start=t_start, t_end=t_end, order=3,
                            skip_type='time_uniform', method='singlestep')
                e.g., DPM-Solver++:
                    >>> dpm_solver = DPM_Solver(model_fn, noise_schedule, algorithm_type="dpmsolver++")
                    >>> x_sample = dpm_solver.sample(x, steps=steps, t_start=t_start, t_end=t_end, order=3,
                            skip_type='time_uniform', method='singlestep')
            - For **guided sampling with large guidance scale** by DPMs:
                Use multistep DPM-Solver with `algorithm_type="dpmsolver++"` and `order = 2`.
                e.g.
                    >>> dpm_solver = DPM_Solver(model_fn, noise_schedule, algorithm_type="dpmsolver++")
                    >>> x_sample = dpm_solver.sample(x, steps=steps, t_start=t_start, t_end=t_end, order=2,
                            skip_type='time_uniform', method='multistep')

        We support three types of `skip_type`:
            - 'logSNR': uniform logSNR for the time steps. **Recommended for low-resolutional images**
            - 'time_uniform': uniform time for the time steps. **Recommended for high-resolutional images**.
            - 'time_quadratic': quadratic time for the time steps.

        =====================================================
        Args:
            x: A pytorch tensor. The initial value at time `t_start`
                e.g. if `t_start` == T, then `x` is a sample from the standard normal distribution.
            steps: A `int`. The total number of function evaluations (NFE).
            t_start: A `float`. The starting time of the sampling.
                If `T` is None, we use self.noise_schedule.T (default is 1.0).
            t_end: A `float`. The ending time of the sampling.
                If `t_end` is None, we use 1. / self.noise_schedule.total_N.
                e.g. if total_N == 1000, we have `t_end` == 1e-3.
                For discrete-time DPMs:
                    - We recommend `t_end` == 1. / self.noise_schedule.total_N.
                For continuous-time DPMs:
                    - We recommend `t_end` == 1e-3 when `steps` <= 15; and `t_end` == 1e-4 when `steps` > 15.
            order: A `int`. The order of DPM-Solver.
            skip_type: A `str`. The type for the spacing of the time steps. 'time_uniform' or 'logSNR' or 'time_quadratic'.
            method: A `str`. The method for sampling. 'singlestep' or 'multistep' or 'singlestep_fixed' or 'adaptive'.
            denoise_to_zero: A `bool`. Whether to denoise to time 0 at the final step.
                Default is `False`. If `denoise_to_zero` is `True`, the total NFE is (`steps` + 1).

                This trick is firstly proposed by DDPM (https://arxiv.org/abs/2006.11239) and
                score_sde (https://arxiv.org/abs/2011.13456). Such trick can improve the FID
                for diffusion models sampling by diffusion SDEs for low-resolutional images
                (such as CIFAR-10). However, we observed that such trick does not matter for
                high-resolutional images. As it needs an additional NFE, we do not recommend
                it for high-resolutional images.
            lower_order_final: A `bool`. Whether to use lower order solvers at the final steps.
                Only valid for `method=multistep` and `steps < 15`. We empirically find that
                this trick is a key to stabilizing the sampling by DPM-Solver with very few steps
                (especially for steps <= 10). So we recommend to set it to be `True`.
            solver_type: A `str`. The taylor expansion type for the solver. `dpmsolver` or `taylor`. We recommend `dpmsolver`.
            atol: A `float`. The absolute tolerance of the adaptive step size solver. Valid when `method` == 'adaptive'.
            rtol: A `float`. The relative tolerance of the adaptive step size solver. Valid when `method` == 'adaptive'.
            return_intermediate: A `bool`. Whether to save the xt at each step.
                When set to `True`, method returns a tuple (x0, intermediates); when set to False, method returns only x0.
        Returns:
            x_end: A pytorch tensor. The approximated solution at time `t_end`.

        """
        t_0 = 1.0 / self.noise_schedule.total_N if t_end is None else t_end
        t_T = self.noise_schedule.T if t_start is None else t_start
        assert t_0 > 0 and t_T > 0, 'Time range needs to be greater than 0. For discrete-time DPMs, it needs to be in [1 / N, 1], where N is the length of betas array'
        if return_intermediate:
            assert method in ['multistep', 'singlestep', 'singlestep_fixed'], 'Cannot use adaptive solver when saving intermediate values'
        if self.correcting_xt_fn is not None:
            assert method in ['multistep', 'singlestep', 'singlestep_fixed'], 'Cannot use adaptive solver when correcting_xt_fn is not None'
        device = x.device
        intermediates = []
        with torch.no_grad():
            if method == 'adaptive':
                x = self.dpm_solver_adaptive(x, order=order, t_T=t_T, t_0=t_0, atol=atol, rtol=rtol, solver_type=solver_type)
            elif method == 'multistep':
                assert steps >= order
                timesteps = self.get_time_steps(skip_type=skip_type, t_T=t_T, t_0=t_0, N=steps, device=device)
                assert timesteps.shape[0] - 1 == steps
                step = 0
                t = timesteps[step]
                t_prev_list = [t]
                model_prev_list = [self.model_fn(x, t)]
                if self.correcting_xt_fn is not None:
                    x = self.correcting_xt_fn(x, t, step)
                if return_intermediate:
                    intermediates.append(x)
                for step in range(1, order):
                    t = timesteps[step]
                    x = self.multistep_dpm_solver_update(x, model_prev_list, t_prev_list, t, step, solver_type=solver_type)
                    if self.correcting_xt_fn is not None:
                        x = self.correcting_xt_fn(x, t, step)
                    if return_intermediate:
                        intermediates.append(x)
                    t_prev_list.append(t)
                    model_prev_list.append(self.model_fn(x, t))
                for step in range(order, steps + 1):
                    t = timesteps[step]
                    if lower_order_final and steps < 10:
                        step_order = min(order, steps + 1 - step)
                    else:
                        step_order = order
                    x = self.multistep_dpm_solver_update(x, model_prev_list, t_prev_list, t, step_order, solver_type=solver_type)
                    if self.correcting_xt_fn is not None:
                        x = self.correcting_xt_fn(x, t, step)
                    if return_intermediate:
                        intermediates.append(x)
                    for i in range(order - 1):
                        t_prev_list[i] = t_prev_list[i + 1]
                        model_prev_list[i] = model_prev_list[i + 1]
                    t_prev_list[-1] = t
                    if step < steps:
                        model_prev_list[-1] = self.model_fn(x, t)
            elif method in ['singlestep', 'singlestep_fixed']:
                if method == 'singlestep':
                    timesteps_outer, orders = self.get_orders_and_timesteps_for_singlestep_solver(steps=steps, order=order, skip_type=skip_type, t_T=t_T, t_0=t_0, device=device)
                elif method == 'singlestep_fixed':
                    K = steps // order
                    orders = [order] * K
                    timesteps_outer = self.get_time_steps(skip_type=skip_type, t_T=t_T, t_0=t_0, N=K, device=device)
                for step, order in enumerate(orders):
                    s, t = timesteps_outer[step], timesteps_outer[step + 1]
                    timesteps_inner = self.get_time_steps(skip_type=skip_type, t_T=s.item(), t_0=t.item(), N=order, device=device)
                    lambda_inner = self.noise_schedule.marginal_lambda(timesteps_inner)
                    h = lambda_inner[-1] - lambda_inner[0]
                    r1 = None if order <= 1 else (lambda_inner[1] - lambda_inner[0]) / h
                    r2 = None if order <= 2 else (lambda_inner[2] - lambda_inner[0]) / h
                    x = self.singlestep_dpm_solver_update(x, s, t, order, solver_type=solver_type, r1=r1, r2=r2)
                    if self.correcting_xt_fn is not None:
                        x = self.correcting_xt_fn(x, t, step)
                    if return_intermediate:
                        intermediates.append(x)
            else:
                raise ValueError('Got wrong method {}'.format(method))
            if denoise_to_zero:
                t = torch.ones((1,)) * t_0
                x = self.denoise_to_zero_fn(x, t)
                if self.correcting_xt_fn is not None:
                    x = self.correcting_xt_fn(x, t, step + 1)
                if return_intermediate:
                    intermediates.append(x)
        if return_intermediate:
            return x, intermediates
        else:
            return x


class ResidualBlock(nn.Module):

    def __init__(self, encoder_hidden, residual_channels, dilation):
        super().__init__()
        self.residual_channels = residual_channels
        self.dilated_conv = nn.Conv1d(residual_channels, 2 * residual_channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.diffusion_projection = nn.Linear(residual_channels, residual_channels)
        self.conditioner_projection = nn.Conv1d(encoder_hidden, 2 * residual_channels, 1)
        self.output_projection = nn.Conv1d(residual_channels, 2 * residual_channels, 1)

    def forward(self, x, conditioner, diffusion_step):
        diffusion_step = self.diffusion_projection(diffusion_step).unsqueeze(-1)
        conditioner = self.conditioner_projection(conditioner)
        y = x + diffusion_step
        y = self.dilated_conv(y) + conditioner
        gate, filter = torch.split(y, [self.residual_channels, self.residual_channels], dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)
        y = self.output_projection(y)
        residual, skip = torch.split(y, [self.residual_channels, self.residual_channels], dim=1)
        return (x + residual) / math.sqrt(2.0), skip


class SinusoidalPosEmb(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class DiffNet(nn.Module):

    def __init__(self, in_dims, n_layers, n_chans, n_hidden):
        super().__init__()
        self.encoder_hidden = n_hidden
        self.residual_layers = n_layers
        self.residual_channels = n_chans
        self.input_projection = Conv1d(in_dims, self.residual_channels, 1)
        self.diffusion_embedding = SinusoidalPosEmb(self.residual_channels)
        dim = self.residual_channels
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), Mish(), nn.Linear(dim * 4, dim))
        self.residual_layers = nn.ModuleList([ResidualBlock(self.encoder_hidden, self.residual_channels, 1) for i in range(self.residual_layers)])
        self.skip_projection = Conv1d(self.residual_channels, self.residual_channels, 1)
        self.output_projection = Conv1d(self.residual_channels, in_dims, 1)
        nn.init.zeros_(self.output_projection.weight)

    def forward(self, spec, diffusion_step, cond):
        x = spec.squeeze(0)
        x = self.input_projection(x)
        x = F.relu(x)
        diffusion_step = diffusion_step.float()
        diffusion_step = self.diffusion_embedding(diffusion_step)
        diffusion_step = self.mlp(diffusion_step)
        x, skip = self.residual_layers[0](x, cond, diffusion_step)
        for layer in self.residual_layers[1:]:
            x, skip_connection = layer.forward(x, cond, diffusion_step)
            skip = skip + skip_connection
        x = skip / math.sqrt(len(self.residual_layers))
        x = self.skip_projection(x)
        x = F.relu(x)
        x = self.output_projection(x)
        return x.unsqueeze(1)


def interpolate_fn(x, xp, yp):
    """
    A piecewise linear function y = f(x), using xp and yp as keypoints.
    We implement f(x) in a differentiable way (i.e. applicable for autograd).
    The function f(x) is well-defined for all x-axis. (For x beyond the bounds of xp, we use the outmost points of xp to define the linear function.)

    Args:
        x: PyTorch tensor with shape [N, C], where N is the batch size, C is the number of channels (we use C = 1 for DPM-Solver).
        xp: PyTorch tensor with shape [C, K], where K is the number of keypoints.
        yp: PyTorch tensor with shape [C, K].
    Returns:
        The function values f(x), with shape [N, C].
    """
    N, K = x.shape[0], xp.shape[1]
    all_x = torch.cat([x.unsqueeze(2), xp.unsqueeze(0).repeat((N, 1, 1))], dim=2)
    sorted_all_x, x_indices = torch.sort(all_x, dim=2)
    x_idx = torch.argmin(x_indices, dim=2)
    cand_start_idx = x_idx - 1
    start_idx = torch.where(torch.eq(x_idx, 0), torch.tensor(1, device=x.device), torch.where(torch.eq(x_idx, K), torch.tensor(K - 2, device=x.device), cand_start_idx))
    end_idx = torch.where(torch.eq(start_idx, cand_start_idx), start_idx + 2, start_idx + 1)
    start_x = torch.gather(sorted_all_x, dim=2, index=start_idx.unsqueeze(2)).squeeze(2)
    end_x = torch.gather(sorted_all_x, dim=2, index=end_idx.unsqueeze(2)).squeeze(2)
    start_idx2 = torch.where(torch.eq(x_idx, 0), torch.tensor(0, device=x.device), torch.where(torch.eq(x_idx, K), torch.tensor(K - 2, device=x.device), cand_start_idx))
    y_positions_expanded = yp.unsqueeze(0).expand(N, -1, -1)
    start_y = torch.gather(y_positions_expanded, dim=2, index=start_idx2.unsqueeze(2)).squeeze(2)
    end_y = torch.gather(y_positions_expanded, dim=2, index=(start_idx2 + 1).unsqueeze(2)).squeeze(2)
    cand = start_y + (x - start_x) * (end_y - start_y) / (end_x - start_x)
    return cand


class NoiseScheduleVP:

    def __init__(self, schedule='discrete', betas=None, alphas_cumprod=None, continuous_beta_0=0.1, continuous_beta_1=20.0, dtype=torch.float32):
        """Create a wrapper class for the forward SDE (VP type).
        ***
        Update: We support discrete-time diffusion models by implementing a picewise linear interpolation for log_alpha_t.
                We recommend to use schedule='discrete' for the discrete-time diffusion models, especially for high-resolution images.
        ***
        The forward SDE ensures that the condition distribution q_{t|0}(x_t | x_0) = N ( alpha_t * x_0, sigma_t^2 * I ).
        We further define lambda_t = log(alpha_t) - log(sigma_t), which is the half-logSNR (described in the DPM-Solver paper).
        Therefore, we implement the functions for computing alpha_t, sigma_t and lambda_t. For t in [0, T], we have:
            log_alpha_t = self.marginal_log_mean_coeff(t)
            sigma_t = self.marginal_std(t)
            lambda_t = self.marginal_lambda(t)
        Moreover, as lambda(t) is an invertible function, we also support its inverse function:
            t = self.inverse_lambda(lambda_t)
        ===============================================================
        We support both discrete-time DPMs (trained on n = 0, 1, ..., N-1) and continuous-time DPMs (trained on t in [t_0, T]).
        1. For discrete-time DPMs:
            For discrete-time DPMs trained on n = 0, 1, ..., N-1, we convert the discrete steps to continuous time steps by:
                t_i = (i + 1) / N
            e.g. for N = 1000, we have t_0 = 1e-3 and T = t_{N-1} = 1.
            We solve the corresponding diffusion ODE from time T = 1 to time t_0 = 1e-3.
            Args:
                betas: A `torch.Tensor`. The beta array for the discrete-time DPM. (See the original DDPM paper for details)
                alphas_cumprod: A `torch.Tensor`. The cumprod alphas for the discrete-time DPM. (See the original DDPM paper for details)
            Note that we always have alphas_cumprod = cumprod(1 - betas). Therefore, we only need to set one of `betas` and `alphas_cumprod`.
            **Important**:  Please pay special attention for the args for `alphas_cumprod`:
                The `alphas_cumprod` is the \\hat{alpha_n} arrays in the notations of DDPM. Specifically, DDPMs assume that
                    q_{t_n | 0}(x_{t_n} | x_0) = N ( \\sqrt{\\hat{alpha_n}} * x_0, (1 - \\hat{alpha_n}) * I ).
                Therefore, the notation \\hat{alpha_n} is different from the notation alpha_t in DPM-Solver. In fact, we have
                    alpha_{t_n} = \\sqrt{\\hat{alpha_n}},
                and
                    log(alpha_{t_n}) = 0.5 * log(\\hat{alpha_n}).
        2. For continuous-time DPMs:
            We support two types of VPSDEs: linear (DDPM) and cosine (improved-DDPM). The hyperparameters for the noise
            schedule are the default settings in DDPM and improved-DDPM:
            Args:
                beta_min: A `float` number. The smallest beta for the linear schedule.
                beta_max: A `float` number. The largest beta for the linear schedule.
                cosine_s: A `float` number. The hyperparameter in the cosine schedule.
                cosine_beta_max: A `float` number. The hyperparameter in the cosine schedule.
                T: A `float` number. The ending time of the forward process.
        ===============================================================
        Args:
            schedule: A `str`. The noise schedule of the forward SDE. 'discrete' for discrete-time DPMs,
                    'linear' or 'cosine' for continuous-time DPMs.
        Returns:
            A wrapper object of the forward SDE (VP type).
        
        ===============================================================
        Example:
        # For discrete-time DPMs, given betas (the beta array for n = 0, 1, ..., N - 1):
        >>> ns = NoiseScheduleVP('discrete', betas=betas)
        # For discrete-time DPMs, given alphas_cumprod (the \\hat{alpha_n} array for n = 0, 1, ..., N - 1):
        >>> ns = NoiseScheduleVP('discrete', alphas_cumprod=alphas_cumprod)
        # For continuous-time DPMs (VPSDE), linear schedule:
        >>> ns = NoiseScheduleVP('linear', continuous_beta_0=0.1, continuous_beta_1=20.)
        """
        if schedule not in ['discrete', 'linear', 'cosine']:
            raise ValueError("Unsupported noise schedule {}. The schedule needs to be 'discrete' or 'linear' or 'cosine'".format(schedule))
        self.schedule = schedule
        if schedule == 'discrete':
            if betas is not None:
                log_alphas = 0.5 * torch.log(1 - betas).cumsum(dim=0)
            else:
                assert alphas_cumprod is not None
                log_alphas = 0.5 * torch.log(alphas_cumprod)
            self.total_N = len(log_alphas)
            self.T = 1.0
            self.t_array = torch.linspace(0.0, 1.0, self.total_N + 1)[1:].reshape((1, -1))
            self.log_alpha_array = log_alphas.reshape((1, -1))
        else:
            self.total_N = 1000
            self.beta_0 = continuous_beta_0
            self.beta_1 = continuous_beta_1
            self.cosine_s = 0.008
            self.cosine_beta_max = 999.0
            self.cosine_t_max = math.atan(self.cosine_beta_max * (1.0 + self.cosine_s) / math.pi) * 2.0 * (1.0 + self.cosine_s) / math.pi - self.cosine_s
            self.cosine_log_alpha_0 = math.log(math.cos(self.cosine_s / (1.0 + self.cosine_s) * math.pi / 2.0))
            self.schedule = schedule
            if schedule == 'cosine':
                self.T = 0.9946
            else:
                self.T = 1.0

    def marginal_log_mean_coeff(self, t):
        """
        Compute log(alpha_t) of a given continuous-time label t in [0, T].
        """
        if self.schedule == 'discrete':
            return interpolate_fn(t.reshape((-1, 1)), self.t_array, self.log_alpha_array).reshape(-1)
        elif self.schedule == 'linear':
            return -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
        elif self.schedule == 'cosine':
            log_alpha_fn = lambda s: torch.log(torch.cos((s + self.cosine_s) / (1.0 + self.cosine_s) * math.pi / 2.0))
            log_alpha_t = log_alpha_fn(t) - self.cosine_log_alpha_0
            return log_alpha_t

    def marginal_alpha(self, t):
        """
        Compute alpha_t of a given continuous-time label t in [0, T].
        """
        return torch.exp(self.marginal_log_mean_coeff(t))

    def marginal_std(self, t):
        """
        Compute sigma_t of a given continuous-time label t in [0, T].
        """
        return torch.sqrt(1.0 - torch.exp(2.0 * self.marginal_log_mean_coeff(t)))

    def marginal_lambda(self, t):
        """
        Compute lambda_t = log(alpha_t) - log(sigma_t) of a given continuous-time label t in [0, T].
        """
        log_mean_coeff = self.marginal_log_mean_coeff(t)
        log_std = 0.5 * torch.log(1.0 - torch.exp(2.0 * log_mean_coeff))
        return log_mean_coeff - log_std

    def inverse_lambda(self, lamb):
        """
        Compute the continuous-time label t in [0, T] of a given half-logSNR lambda_t.
        """
        if self.schedule == 'linear':
            tmp = 2.0 * (self.beta_1 - self.beta_0) * torch.logaddexp(-2.0 * lamb, torch.zeros((1,)))
            Delta = self.beta_0 ** 2 + tmp
            return tmp / (torch.sqrt(Delta) + self.beta_0) / (self.beta_1 - self.beta_0)
        elif self.schedule == 'discrete':
            log_alpha = -0.5 * torch.logaddexp(torch.zeros((1,)), -2.0 * lamb)
            t = interpolate_fn(log_alpha.reshape((-1, 1)), torch.flip(self.log_alpha_array, [1]), torch.flip(self.t_array, [1]))
            return t.reshape((-1,))
        else:
            log_alpha = -0.5 * torch.logaddexp(-2.0 * lamb, torch.zeros((1,)))
            t_fn = lambda log_alpha_t: torch.arccos(torch.exp(log_alpha_t + self.cosine_log_alpha_0)) * 2.0 * (1.0 + self.cosine_s) / math.pi - self.cosine_s
            t = t_fn(log_alpha)
            return t


def extract(a, t):
    return a[t].reshape((1, 1, 1, 1))


class Pred(nn.Module):

    def __init__(self, alphas_cumprod):
        super().__init__()
        self.alphas_cumprod = alphas_cumprod

    def forward(self, x_1, noise_t, t_1, t_prev):
        a_t = extract(self.alphas_cumprod, t_1).cpu()
        a_prev = extract(self.alphas_cumprod, t_prev).cpu()
        a_t_sq, a_prev_sq = a_t.sqrt().cpu(), a_prev.sqrt().cpu()
        x_delta = (a_prev - a_t) * (1 / (a_t_sq * (a_t_sq + a_prev_sq)) * x_1 - 1 / (a_t_sq * (((1 - a_prev) * a_t).sqrt() + ((1 - a_t) * a_prev).sqrt())) * noise_t)
        x_pred = x_1 + x_delta.cpu()
        return x_pred


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos((x / steps + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
    return np.clip(betas, a_min=0, a_max=0.999)


def linear_beta_schedule(timesteps, max_beta=0.02):
    """
    linear schedule
    """
    betas = np.linspace(0.0001, max_beta, timesteps)
    return betas


beta_schedule = {'cosine': cosine_beta_schedule, 'linear': linear_beta_schedule}


def model_wrapper(model, noise_schedule, model_type='noise', model_kwargs={}, guidance_type='uncond', condition=None, unconditional_condition=None, guidance_scale=1.0, classifier_fn=None, classifier_kwargs={}):
    """Create a wrapper function for the noise prediction model.
    """

    def get_model_input_time(t_continuous):
        """
        Convert the continuous-time `t_continuous` (in [epsilon, T]) to the model input time.
        For discrete-time DPMs, we convert `t_continuous` in [1 / N, 1] to `t_input` in [0, 1000 * (N - 1) / N].
        For continuous-time DPMs, we just use `t_continuous`.
        """
        if noise_schedule.schedule == 'discrete':
            return (t_continuous - 1.0 / noise_schedule.total_N) * noise_schedule.total_N
        else:
            return t_continuous

    def noise_pred_fn(x, t_continuous, cond=None):
        t_input = get_model_input_time(t_continuous)
        if cond is None:
            output = model(x, t_input, **model_kwargs)
        else:
            output = model(x, t_input, cond, **model_kwargs)
        if model_type == 'noise':
            return output
        elif model_type == 'x_start':
            alpha_t, sigma_t = noise_schedule.marginal_alpha(t_continuous), noise_schedule.marginal_std(t_continuous)
            return (x - alpha_t * output) / sigma_t
        elif model_type == 'v':
            alpha_t, sigma_t = noise_schedule.marginal_alpha(t_continuous), noise_schedule.marginal_std(t_continuous)
            return alpha_t * output + sigma_t * x
        elif model_type == 'score':
            sigma_t = noise_schedule.marginal_std(t_continuous)
            return -sigma_t * output

    def cond_grad_fn(x, t_input):
        """
        Compute the gradient of the classifier, i.e. nabla_{x} log p_t(cond | x_t).
        """
        with torch.enable_grad():
            x_in = x.detach().requires_grad_(True)
            log_prob = classifier_fn(x_in, t_input, condition, **classifier_kwargs)
            return torch.autograd.grad(log_prob.sum(), x_in)[0]

    def model_fn(x, t_continuous):
        """
        The noise predicition model function that is used for DPM-Solver.
        """
        if guidance_type == 'uncond':
            return noise_pred_fn(x, t_continuous)
        elif guidance_type == 'classifier':
            assert classifier_fn is not None
            t_input = get_model_input_time(t_continuous)
            cond_grad = cond_grad_fn(x, t_input)
            sigma_t = noise_schedule.marginal_std(t_continuous)
            noise = noise_pred_fn(x, t_continuous)
            return noise - guidance_scale * sigma_t * cond_grad
        elif guidance_type == 'classifier-free':
            if guidance_scale == 1.0 or unconditional_condition is None:
                return noise_pred_fn(x, t_continuous, cond=condition)
            else:
                x_in = torch.cat([x] * 2)
                t_in = torch.cat([t_continuous] * 2)
                c_in = torch.cat([unconditional_condition, condition])
                noise_uncond, noise = noise_pred_fn(x_in, t_in, cond=c_in).chunk(2)
                return noise_uncond + guidance_scale * (noise - noise_uncond)
    assert model_type in ['noise', 'x_start', 'v']
    assert guidance_type in ['uncond', 'classifier', 'classifier-free']
    return model_fn


def noise_like(shape, device, repeat=False):
    repeat_noise = lambda : torch.randn((1, *shape[1:]), device=device).repeat(shape[0], *((1,) * (len(shape) - 1)))
    noise = lambda : torch.randn(shape, device=device)
    return repeat_noise() if repeat else noise()


def predict_stage0(noise_pred, noise_pred_prev):
    return (noise_pred + noise_pred_prev) / 2


def predict_stage1(noise_pred, noise_list):
    return (noise_pred * 3 - noise_list[-1]) / 2


def predict_stage2(noise_pred, noise_list):
    return (noise_pred * 23 - noise_list[-1] * 16 + noise_list[-2] * 5) / 12


def predict_stage3(noise_pred, noise_list):
    return (noise_pred * 55 - noise_list[-1] * 59 + noise_list[-2] * 37 - noise_list[-3] * 9) / 24


class GaussianDiffusion(nn.Module):

    def __init__(self, out_dims=128, n_layers=20, n_chans=384, n_hidden=256, timesteps=1000, k_step=1000, max_beta=0.02, spec_min=-12, spec_max=2):
        super().__init__()
        self.denoise_fn = DiffNet(out_dims, n_layers, n_chans, n_hidden)
        self.out_dims = out_dims
        self.mel_bins = out_dims
        self.n_hidden = n_hidden
        betas = beta_schedule['linear'](timesteps, max_beta=max_beta)
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])
        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.k_step = k_step
        self.noise_list = deque(maxlen=4)
        to_torch = partial(torch.tensor, dtype=torch.float32)
        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1.0 - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1.0 - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1.0 / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1.0 / alphas_cumprod - 1)))
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer('posterior_variance', to_torch(posterior_variance))
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch((1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod)))
        self.register_buffer('spec_min', torch.FloatTensor([spec_min])[None, None, :out_dims])
        self.register_buffer('spec_max', torch.FloatTensor([spec_max])[None, None, :out_dims])
        self.ad = AfterDiffusion(self.spec_max, self.spec_min)
        self.xp = Pred(self.alphas_cumprod)

    def q_mean_variance(self, x_start, t):
        mean = extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = extract(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def predict_start_from_noise(self, x_t, t, noise):
        return extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = extract(self.posterior_mean_coef1, t, x_t.shape) * x_start + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, cond):
        noise_pred = self.denoise_fn(x, t, cond=cond)
        x_recon = self.predict_start_from_noise(x, t=t, noise=noise_pred)
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, t, cond, clip_denoised=True, repeat_noise=False):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, cond=cond)
        noise = noise_like(x.shape, device, repeat_noise)
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_plms(self, x, t, interval, cond, clip_denoised=True, repeat_noise=False):
        """
        Use the PLMS method from
        [Pseudo Numerical Methods for Diffusion Models on Manifolds](https://arxiv.org/abs/2202.09778).
        """

        def get_x_pred(x, noise_t, t):
            a_t = extract(self.alphas_cumprod, t)
            a_prev = extract(self.alphas_cumprod, torch.max(t - interval, torch.zeros_like(t)))
            a_t_sq, a_prev_sq = a_t.sqrt(), a_prev.sqrt()
            x_delta = (a_prev - a_t) * (1 / (a_t_sq * (a_t_sq + a_prev_sq)) * x - 1 / (a_t_sq * (((1 - a_prev) * a_t).sqrt() + ((1 - a_t) * a_prev).sqrt())) * noise_t)
            x_pred = x + x_delta
            return x_pred
        noise_list = self.noise_list
        noise_pred = self.denoise_fn(x, t, cond=cond)
        if len(noise_list) == 0:
            x_pred = get_x_pred(x, noise_pred, t)
            noise_pred_prev = self.denoise_fn(x_pred, max(t - interval, 0), cond=cond)
            noise_pred_prime = (noise_pred + noise_pred_prev) / 2
        elif len(noise_list) == 1:
            noise_pred_prime = (3 * noise_pred - noise_list[-1]) / 2
        elif len(noise_list) == 2:
            noise_pred_prime = (23 * noise_pred - 16 * noise_list[-1] + 5 * noise_list[-2]) / 12
        else:
            noise_pred_prime = (55 * noise_pred - 59 * noise_list[-1] + 37 * noise_list[-2] - 9 * noise_list[-3]) / 24
        x_prev = get_x_pred(x, noise_pred_prime, t)
        noise_list.append(noise_pred)
        return x_prev

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda : torch.randn_like(x_start))
        return extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise

    def p_losses(self, x_start, t, cond, noise=None, loss_type='l2'):
        noise = default(noise, lambda : torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_recon = self.denoise_fn(x_noisy, t, cond)
        if loss_type == 'l1':
            loss = (noise - x_recon).abs().mean()
        elif loss_type == 'l2':
            loss = F.mse_loss(noise, x_recon)
        else:
            raise NotImplementedError()
        return loss

    def org_forward(self, condition, init_noise=None, gt_spec=None, infer=True, infer_speedup=100, method='pndm', k_step=1000, use_tqdm=True):
        """
            conditioning diffusion, use fastspeech2 encoder output as the condition
        """
        cond = condition
        b, device = condition.shape[0], condition.device
        if not infer:
            spec = self.norm_spec(gt_spec)
            t = torch.randint(0, self.k_step, (b,), device=device).long()
            norm_spec = spec.transpose(1, 2)[:, None, :, :]
            return self.p_losses(norm_spec, t, cond=cond)
        else:
            shape = cond.shape[0], 1, self.out_dims, cond.shape[2]
            if gt_spec is None:
                t = self.k_step
                if init_noise is None:
                    x = torch.randn(shape, device=device)
                else:
                    x = init_noise
            else:
                t = k_step
                norm_spec = self.norm_spec(gt_spec)
                norm_spec = norm_spec.transpose(1, 2)[:, None, :, :]
                x = self.q_sample(x_start=norm_spec, t=torch.tensor([t - 1], device=device).long())
            if method is not None and infer_speedup > 1:
                if method == 'dpm-solver':
                    noise_schedule = NoiseScheduleVP(schedule='discrete', betas=self.betas[:t])

                    def my_wrapper(fn):

                        def wrapped(x, t, **kwargs):
                            ret = fn(x, t, **kwargs)
                            if use_tqdm:
                                self.bar.update(1)
                            return ret
                        return wrapped
                    model_fn = model_wrapper(my_wrapper(self.denoise_fn), noise_schedule, model_type='noise', model_kwargs={'cond': cond})
                    dpm_solver = DPM_Solver(model_fn, noise_schedule)
                    steps = t // infer_speedup
                    if use_tqdm:
                        self.bar = tqdm(desc='sample time step', total=steps)
                    x = dpm_solver.sample(x, steps=steps, order=3, skip_type='time_uniform', method='singlestep')
                    if use_tqdm:
                        self.bar.close()
                elif method == 'pndm':
                    self.noise_list = deque(maxlen=4)
                    if use_tqdm:
                        for i in tqdm(reversed(range(0, t, infer_speedup)), desc='sample time step', total=t // infer_speedup):
                            x = self.p_sample_plms(x, torch.full((b,), i, device=device, dtype=torch.long), infer_speedup, cond=cond)
                    else:
                        for i in reversed(range(0, t, infer_speedup)):
                            x = self.p_sample_plms(x, torch.full((b,), i, device=device, dtype=torch.long), infer_speedup, cond=cond)
                else:
                    raise NotImplementedError(method)
            elif use_tqdm:
                for i in tqdm(reversed(range(0, t)), desc='sample time step', total=t):
                    x = self.p_sample(x, torch.full((b,), i, device=device, dtype=torch.long), cond)
            else:
                for i in reversed(range(0, t)):
                    x = self.p_sample(x, torch.full((b,), i, device=device, dtype=torch.long), cond)
            x = x.squeeze(1).transpose(1, 2)
            return self.denorm_spec(x).transpose(2, 1)

    def norm_spec(self, x):
        return (x - self.spec_min) / (self.spec_max - self.spec_min) * 2 - 1

    def denorm_spec(self, x):
        return (x + 1) / 2 * (self.spec_max - self.spec_min) + self.spec_min

    def get_x_pred(self, x_1, noise_t, t_1, t_prev):
        a_t = extract(self.alphas_cumprod, t_1)
        a_prev = extract(self.alphas_cumprod, t_prev)
        a_t_sq, a_prev_sq = a_t.sqrt(), a_prev.sqrt()
        x_delta = (a_prev - a_t) * (1 / (a_t_sq * (a_t_sq + a_prev_sq)) * x_1 - 1 / (a_t_sq * (((1 - a_prev) * a_t).sqrt() + ((1 - a_t) * a_prev).sqrt())) * noise_t)
        x_pred = x_1 + x_delta
        return x_pred

    def OnnxExport(self, project_name=None, init_noise=None, hidden_channels=256, export_denoise=True, export_pred=True, export_after=True):
        cond = torch.randn([1, self.n_hidden, 10]).cpu()
        if init_noise is None:
            x = torch.randn((1, 1, self.mel_bins, cond.shape[2]), dtype=torch.float32).cpu()
        else:
            x = init_noise
        pndms = 100
        org_y_x = self.org_forward(cond, init_noise=x)
        device = cond.device
        n_frames = cond.shape[2]
        step_range = torch.arange(0, self.k_step, pndms, dtype=torch.long, device=device).flip(0)
        plms_noise_stage = torch.tensor(0, dtype=torch.long, device=device)
        noise_list = torch.zeros((0, 1, 1, self.mel_bins, n_frames), device=device)
        ot = step_range[0]
        ot_1 = torch.full((1,), ot, device=device, dtype=torch.long)
        if export_denoise:
            torch.onnx.export(self.denoise_fn, (x.cpu(), ot_1.cpu(), cond.cpu()), f'{project_name}_denoise.onnx', input_names=['noise', 'time', 'condition'], output_names=['noise_pred'], dynamic_axes={'noise': [3], 'condition': [2]}, opset_version=16)
        for t in step_range:
            t_1 = torch.full((1,), t, device=device, dtype=torch.long)
            noise_pred = self.denoise_fn(x, t_1, cond)
            t_prev = t_1 - pndms
            t_prev = t_prev * (t_prev > 0)
            if plms_noise_stage == 0:
                if export_pred:
                    torch.onnx.export(self.xp, (x.cpu(), noise_pred.cpu(), t_1.cpu(), t_prev.cpu()), f'{project_name}_pred.onnx', input_names=['noise', 'noise_pred', 'time', 'time_prev'], output_names=['noise_pred_o'], dynamic_axes={'noise': [3], 'noise_pred': [3]}, opset_version=16)
                x_pred = self.get_x_pred(x, noise_pred, t_1, t_prev)
                noise_pred_prev = self.denoise_fn(x_pred, t_prev, cond=cond)
                noise_pred_prime = predict_stage0(noise_pred, noise_pred_prev)
            elif plms_noise_stage == 1:
                noise_pred_prime = predict_stage1(noise_pred, noise_list)
            elif plms_noise_stage == 2:
                noise_pred_prime = predict_stage2(noise_pred, noise_list)
            else:
                noise_pred_prime = predict_stage3(noise_pred, noise_list)
            noise_pred = noise_pred.unsqueeze(0)
            if plms_noise_stage < 3:
                noise_list = torch.cat((noise_list, noise_pred), dim=0)
                plms_noise_stage = plms_noise_stage + 1
            else:
                noise_list = torch.cat((noise_list[-2:], noise_pred), dim=0)
            x = self.get_x_pred(x, noise_pred_prime, t_1, t_prev)
        if export_after:
            torch.onnx.export(self.ad, x.cpu(), f'{project_name}_after.onnx', input_names=['x'], output_names=['mel_out'], dynamic_axes={'x': [3]}, opset_version=16)
        x = self.ad(x)
        None
        return x

    def forward(self, condition=None, init_noise=None, pndms=None, k_step=None):
        cond = condition
        x = init_noise
        device = cond.device
        n_frames = cond.shape[2]
        step_range = torch.arange(0, k_step.item(), pndms.item(), dtype=torch.long, device=device).flip(0)
        plms_noise_stage = torch.tensor(0, dtype=torch.long, device=device)
        noise_list = torch.zeros((0, 1, 1, self.mel_bins, n_frames), device=device)
        ot = step_range[0]
        ot_1 = torch.full((1,), ot, device=device, dtype=torch.long)
        for t in step_range:
            t_1 = torch.full((1,), t, device=device, dtype=torch.long)
            noise_pred = self.denoise_fn(x, t_1, cond)
            t_prev = t_1 - pndms
            t_prev = t_prev * (t_prev > 0)
            if plms_noise_stage == 0:
                x_pred = self.get_x_pred(x, noise_pred, t_1, t_prev)
                noise_pred_prev = self.denoise_fn(x_pred, t_prev, cond=cond)
                noise_pred_prime = predict_stage0(noise_pred, noise_pred_prev)
            elif plms_noise_stage == 1:
                noise_pred_prime = predict_stage1(noise_pred, noise_list)
            elif plms_noise_stage == 2:
                noise_pred_prime = predict_stage2(noise_pred, noise_list)
            else:
                noise_pred_prime = predict_stage3(noise_pred, noise_list)
            noise_pred = noise_pred.unsqueeze(0)
            if plms_noise_stage < 3:
                noise_list = torch.cat((noise_list, noise_pred), dim=0)
                plms_noise_stage = plms_noise_stage + 1
            else:
                noise_list = torch.cat((noise_list[-2:], noise_pred), dim=0)
            x = self.get_x_pred(x, noise_pred_prime, t_1, t_prev)
        x = self.ad(x)
        return x


class DiffusionEmbedding(nn.Module):
    """Diffusion Step Embedding"""

    def __init__(self, d_denoiser):
        super(DiffusionEmbedding, self).__init__()
        self.dim = d_denoiser

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class NaiveV2DiffLayer(nn.Module):

    def __init__(self, dim_model: 'int', dim_cond: 'int', num_heads: 'int'=4, use_norm: 'bool'=False, conv_only: 'bool'=True, conv_dropout: 'float'=0.0, atten_dropout: 'float'=0.1, use_mlp=True, expansion_factor=2, kernel_size=31, wavenet_like=False, conv_model_type='mode1'):
        super().__init__()
        self.conformer = ConformerConvModule(dim_model, expansion_factor=expansion_factor, kernel_size=kernel_size, dropout=conv_dropout, use_norm=use_norm, conv_model_type=conv_model_type)
        self.norm = nn.LayerNorm(dim_model)
        self.dropout = nn.Dropout(0.1)
        if wavenet_like:
            self.wavenet_like_proj = nn.Conv1d(dim_model, 2 * dim_model, 1)
        else:
            self.wavenet_like_proj = None
        self.diffusion_step_projection = nn.Conv1d(dim_model, dim_model, 1)
        self.condition_projection = nn.Conv1d(dim_cond, dim_model, 1)
        if not conv_only:
            self.attn = nn.TransformerEncoderLayer(d_model=dim_model, nhead=num_heads, dim_feedforward=dim_model * 4, dropout=atten_dropout, activation='gelu')
        else:
            self.attn = None

    def forward(self, x, condition=None, diffusion_step=None) ->torch.Tensor:
        res_x = x.transpose(1, 2)
        x = x + self.diffusion_step_projection(diffusion_step) + self.condition_projection(condition)
        x = x.transpose(1, 2)
        if self.attn is not None:
            x = self.attn(self.norm(x))
        x = self.conformer(x)
        if self.wavenet_like_proj is not None:
            x = self.wavenet_like_proj(x.transpose(1, 2)).transpose(1, 2)
            x = F.glu(x, dim=-1)
            return ((x + res_x) / math.sqrt(2.0)).transpose(1, 2), res_x.transpose(1, 2)
        else:
            x = x + res_x
            x = x.transpose(1, 2)
            return x


class NaiveV2Diff(nn.Module):

    def __init__(self, mel_channels=128, dim=512, use_mlp=True, mlp_factor=4, condition_dim=256, num_layers=20, expansion_factor=2, kernel_size=31, conv_only=True, wavenet_like=False, use_norm=False, conv_model_type='mode1', conv_dropout=0.0, atten_dropout=0.1):
        super(NaiveV2Diff, self).__init__()
        self.wavenet_like = wavenet_like
        self.mask_cond_ratio = None
        self.input_projection = nn.Conv1d(mel_channels, dim, 1)
        self.diffusion_embedding = nn.Sequential(DiffusionEmbedding(dim), nn.Linear(dim, dim * mlp_factor), nn.GELU(), nn.Linear(dim * mlp_factor, dim))
        if use_mlp:
            self.conditioner_projection = nn.Sequential(nn.Conv1d(condition_dim, dim * mlp_factor, 1), nn.GELU(), nn.Conv1d(dim * mlp_factor, dim, 1))
        else:
            self.conditioner_projection = nn.Identity()
        self.residual_layers = nn.ModuleList([NaiveV2DiffLayer(dim_model=dim, dim_cond=dim if use_mlp else condition_dim, num_heads=8, use_norm=use_norm, conv_only=conv_only, conv_dropout=conv_dropout, atten_dropout=atten_dropout, use_mlp=use_mlp, expansion_factor=expansion_factor, kernel_size=kernel_size, wavenet_like=wavenet_like, conv_model_type=conv_model_type) for i in range(num_layers)])
        if use_mlp:
            _ = nn.Conv1d(dim * mlp_factor, mel_channels, kernel_size=1)
            nn.init.zeros_(_.weight)
            self.output_projection = nn.Sequential(nn.Conv1d(dim, dim * mlp_factor, kernel_size=1), nn.GELU(), _)
        else:
            self.output_projection = nn.Conv1d(dim, mel_channels, kernel_size=1)
            nn.init.zeros_(self.output_projection.weight)

    def forward(self, spec, diffusion_step, cond):
        x = spec
        conditioner = cond
        """

        :param x: [B, M, T]
        :param diffusion_step: [B,]
        :param conditioner: [B, M, T]
        :return:
        """
        use_4_dim = False
        if x.dim() == 4:
            x = x[:, 0]
            use_4_dim = True
        assert x.dim() == 3, f'mel must be 3 dim tensor, but got {x.dim()}'
        x = self.input_projection(x)
        x = F.gelu(x)
        diffusion_step = self.diffusion_embedding(diffusion_step).unsqueeze(-1)
        condition = self.conditioner_projection(conditioner)
        if self.wavenet_like:
            _sk = []
            for layer in self.residual_layers:
                if self.mask_cond_ratio is not None:
                    _mask_cond_ratio = random.choice([True, True, False])
                    if _mask_cond_ratio:
                        _mask_cond_ratio = random.uniform(0, self.mask_cond_ratio)
                    _conditioner = F.dropout(conditioner, _mask_cond_ratio)
                else:
                    _conditioner = conditioner
                x, sk = layer(x, _conditioner, diffusion_step)
                _sk.append(sk)
            x = torch.sum(torch.stack(_sk), dim=0) / math.sqrt(len(self.residual_layers))
        else:
            for layer in self.residual_layers:
                if self.mask_cond_ratio is not None:
                    _mask_cond_ratio = random.choice([True, True, False])
                    if _mask_cond_ratio:
                        _mask_cond_ratio = random.uniform(0, self.mask_cond_ratio)
                    _conditioner = F.dropout(conditioner, _mask_cond_ratio)
                else:
                    _conditioner = conditioner
                x = layer(x, condition, diffusion_step)
        x = self.output_projection(x)
        return x[:, None] if use_4_dim else x


class WaveNet(nn.Module):

    def __init__(self, in_dims=128, n_layers=20, n_chans=384, n_hidden=256):
        super().__init__()
        self.input_projection = Conv1d(in_dims, n_chans, 1)
        self.diffusion_embedding = SinusoidalPosEmb(n_chans)
        self.mlp = nn.Sequential(nn.Linear(n_chans, n_chans * 4), Mish(), nn.Linear(n_chans * 4, n_chans))
        self.residual_layers = nn.ModuleList([ResidualBlock(encoder_hidden=n_hidden, residual_channels=n_chans, dilation=1) for i in range(n_layers)])
        self.skip_projection = Conv1d(n_chans, n_chans, 1)
        self.output_projection = Conv1d(n_chans, in_dims, 1)
        nn.init.zeros_(self.output_projection.weight)

    def forward(self, spec, diffusion_step, cond):
        """
        :param spec: [B, 1, M, T]
        :param diffusion_step: [B, 1]
        :param cond: [B, M, T]
        :return:
        """
        x = spec.squeeze(1)
        x = self.input_projection(x)
        x = F.relu(x)
        diffusion_step = self.diffusion_embedding(diffusion_step)
        diffusion_step = self.mlp(diffusion_step)
        skip = []
        for layer in self.residual_layers:
            x, skip_connection = layer(x, cond, diffusion_step)
            skip.append(skip_connection)
        x = torch.sum(torch.stack(skip), dim=0) / sqrt(len(self.residual_layers))
        x = self.skip_projection(x)
        x = F.relu(x)
        x = self.output_projection(x)
        return x[:, None, :, :]


class Unit2Mel(nn.Module):

    def __init__(self, input_channel, n_spk, use_pitch_aug=False, out_dims=128, n_layers=20, n_chans=384, n_hidden=256):
        super().__init__()
        self.unit_embed = nn.Linear(input_channel, n_hidden)
        self.f0_embed = nn.Linear(1, n_hidden)
        self.volume_embed = nn.Linear(1, n_hidden)
        if use_pitch_aug:
            self.aug_shift_embed = nn.Linear(1, n_hidden, bias=False)
        else:
            self.aug_shift_embed = None
        self.n_spk = n_spk
        if n_spk is not None and n_spk > 1:
            self.spk_embed = nn.Embedding(n_spk, n_hidden)
        self.decoder = GaussianDiffusion(WaveNet(out_dims, n_layers, n_chans, n_hidden), out_dims=out_dims)

    def forward(self, units, f0, volume, spk_id=None, spk_mix_dict=None, aug_shift=None, vocoder=None, gt_spec=None, infer=True, infer_speedup=10, method='dpm-solver', k_step=300, use_tqdm=True):
        """
        input: 
            B x n_frames x n_unit
        return: 
            dict of B x n_frames x feat
        """
        x = self.unit_embed(units) + self.f0_embed((1 + f0 / 700).log()) + self.volume_embed(volume)
        if self.n_spk is not None and self.n_spk > 1:
            if spk_mix_dict is not None:
                for k, v in spk_mix_dict.items():
                    spk_id_torch = torch.LongTensor(np.array([[k]]))
                    x = x + v * self.spk_embed(spk_id_torch - 1)
            else:
                x = x + self.spk_embed(spk_id - 1)
        if self.aug_shift_embed is not None and aug_shift is not None:
            x = x + self.aug_shift_embed(aug_shift / 5)
        x = self.decoder(x, gt_spec=gt_spec, infer=infer, infer_speedup=infer_speedup, method=method, k_step=k_step, use_tqdm=use_tqdm)
        return x


def dynamic_range_compression_torch(x, C=1, clip_val=1e-05):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def load_wav_to_torch(full_path, target_sr=None, return_empty_on_exception=False):
    sampling_rate = None
    try:
        data, sampling_rate = sf.read(full_path, always_2d=True)
    except Exception as ex:
        None
        None
        if return_empty_on_exception:
            return [], sampling_rate or target_sr or 48000
        else:
            raise Exception(ex)
    if len(data.shape) > 1:
        data = data[:, 0]
        assert len(data) > 2
    if np.issubdtype(data.dtype, np.integer):
        max_mag = -np.iinfo(data.dtype).min
    else:
        max_mag = max(np.amax(data), -np.amin(data))
        max_mag = 2 ** 31 + 1 if max_mag > 2 ** 15 else 2 ** 15 + 1 if max_mag > 1.01 else 1.0
    data = torch.FloatTensor(data.astype(np.float32)) / max_mag
    if (torch.isinf(data) | torch.isnan(data)).any() and return_empty_on_exception:
        return [], sampling_rate or target_sr or 48000
    if target_sr is not None and sampling_rate != target_sr:
        data = torch.from_numpy(librosa.core.resample(data.numpy(), orig_sr=sampling_rate, target_sr=target_sr))
        sampling_rate = target_sr
    return data, sampling_rate


class STFT:

    def __init__(self, sr=22050, n_mels=80, n_fft=1024, win_size=1024, hop_length=256, fmin=20, fmax=11025, clip_val=1e-05):
        self.target_sr = sr
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.win_size = win_size
        self.hop_length = hop_length
        self.fmin = fmin
        self.fmax = fmax
        self.clip_val = clip_val
        self.mel_basis = {}
        self.hann_window = {}

    def get_mel(self, y, keyshift=0, speed=1, center=False):
        sampling_rate = self.target_sr
        n_mels = self.n_mels
        n_fft = self.n_fft
        win_size = self.win_size
        hop_length = self.hop_length
        fmin = self.fmin
        fmax = self.fmax
        clip_val = self.clip_val
        factor = 2 ** (keyshift / 12)
        n_fft_new = int(np.round(n_fft * factor))
        win_size_new = int(np.round(win_size * factor))
        hop_length_new = int(np.round(hop_length * speed))
        mel_basis_key = str(fmax) + '_' + str(y.device)
        if mel_basis_key not in self.mel_basis:
            mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax)
            self.mel_basis[mel_basis_key] = torch.from_numpy(mel).float()
        keyshift_key = str(keyshift) + '_' + str(y.device)
        if keyshift_key not in self.hann_window:
            self.hann_window[keyshift_key] = torch.hann_window(win_size_new)
        pad_left = (win_size_new - hop_length_new) // 2
        pad_right = max((win_size_new - hop_length_new + 1) // 2, win_size_new - y.size(-1) - pad_left)
        if pad_right < y.size(-1):
            mode = 'reflect'
        else:
            mode = 'constant'
        y = torch.nn.functional.pad(y.unsqueeze(1), (pad_left, pad_right), mode=mode)
        y = y.squeeze(1)
        spec = torch.stft(y, n_fft_new, hop_length=hop_length_new, win_length=win_size_new, window=self.hann_window[keyshift_key], center=center, pad_mode='reflect', normalized=False, onesided=True, return_complex=True)
        spec = torch.sqrt(spec.real.pow(2) + spec.imag.pow(2) + 1e-09)
        if keyshift != 0:
            size = n_fft // 2 + 1
            resize = spec.size(1)
            if resize < size:
                spec = F.pad(spec, (0, 0, 0, size - resize))
            spec = spec[:, :size, :] * win_size / win_size_new
        spec = torch.matmul(self.mel_basis[mel_basis_key], spec)
        spec = dynamic_range_compression_torch(spec, clip_val=clip_val)
        return spec

    def __call__(self, audiopath):
        audio, sr = load_wav_to_torch(audiopath, target_sr=self.target_sr)
        spect = self.get_mel(audio.unsqueeze(0)).squeeze(0)
        return spect


class AttrDict(dict):

    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def load_config(model_path):
    config_file = os.path.join(os.path.split(model_path)[0], 'config.json')
    with open(config_file) as f:
        data = f.read()
    json_config = json.loads(data)
    h = AttrDict(json_config)
    return h


LRELU_SLOPE = 0.1


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(mean, std)


class ResBlock1(torch.nn.Module):

    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3, 5)):
        super(ResBlock1, self).__init__()
        self.h = h
        self.convs1 = nn.ModuleList([weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0], padding=get_padding(kernel_size, dilation[0]))), weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1], padding=get_padding(kernel_size, dilation[1]))), weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[2], padding=get_padding(kernel_size, dilation[2])))])
        self.convs1.apply(init_weights)
        self.convs2 = nn.ModuleList([weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=get_padding(kernel_size, 1))), weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=get_padding(kernel_size, 1))), weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=get_padding(kernel_size, 1)))])
        self.convs2.apply(init_weights)

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)


class ResBlock2(torch.nn.Module):

    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3)):
        super(ResBlock2, self).__init__()
        self.h = h
        self.convs = nn.ModuleList([weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0], padding=get_padding(kernel_size, dilation[0]))), weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1], padding=get_padding(kernel_size, dilation[1])))])
        self.convs.apply(init_weights)

    def forward(self, x):
        for c in self.convs:
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs:
            remove_weight_norm(l)


class SineGen(torch.nn.Module):
    """ Definition of sine generator
    SineGen(samp_rate, harmonic_num = 0,
            sine_amp = 0.1, noise_std = 0.003,
            voiced_threshold = 0,
            flag_for_pulse=False)
    samp_rate: sampling rate in Hz
    harmonic_num: number of harmonic overtones (default 0)
    sine_amp: amplitude of sine-waveform (default 0.1)
    noise_std: std of Gaussian noise (default 0.003)
    voiced_threshold: F0 threshold for U/V classification (default 0)
    flag_for_pulse: this SinGen is used inside PulseGen (default False)
    Note: when flag_for_pulse is True, the first time step of a voiced
        segment is always sin(np.pi) or cos(0)
    """

    def __init__(self, samp_rate, harmonic_num=0, sine_amp=0.1, noise_std=0.003, voiced_threshold=0):
        super(SineGen, self).__init__()
        self.sine_amp = sine_amp
        self.noise_std = noise_std
        self.harmonic_num = harmonic_num
        self.dim = self.harmonic_num + 1
        self.sampling_rate = samp_rate
        self.voiced_threshold = voiced_threshold

    def _f02uv(self, f0):
        uv = torch.ones_like(f0)
        uv = uv * (f0 > self.voiced_threshold)
        return uv

    def _f02sine(self, f0, upp):
        """ f0: (batchsize, length, dim)
            where dim indicates fundamental tone and overtones
        """
        rad = f0 / self.sampling_rate * torch.arange(1, upp + 1, device=f0.device)
        rad2 = torch.fmod(rad[..., -1:].float() + 0.5, 1.0) - 0.5
        rad_acc = rad2.cumsum(dim=1).fmod(1.0)
        rad += F.pad(rad_acc, (0, 0, 1, -1))
        rad = rad.reshape(f0.shape[0], -1, 1)
        rad = torch.multiply(rad, torch.arange(1, self.dim + 1, device=f0.device).reshape(1, 1, -1))
        rand_ini = torch.rand(1, 1, self.dim, device=f0.device)
        rand_ini[..., 0] = 0
        rad += rand_ini
        sines = torch.sin(2 * np.pi * rad)
        return sines

    @torch.no_grad()
    def forward(self, f0, upp):
        """ sine_tensor, uv = forward(f0)
        input F0: tensor(batchsize=1, length, dim=1)
                  f0 for unvoiced steps should be 0
        output sine_tensor: tensor(batchsize=1, length, dim)
        output uv: tensor(batchsize=1, length, 1)
        """
        f0 = f0.unsqueeze(-1)
        sine_waves = self._f02sine(f0, upp) * self.sine_amp
        uv = (f0 > self.voiced_threshold).float()
        uv = F.interpolate(uv.transpose(2, 1), scale_factor=upp, mode='nearest').transpose(2, 1)
        noise_amp = uv * self.noise_std + (1 - uv) * self.sine_amp / 3
        noise = noise_amp * torch.randn_like(sine_waves)
        sine_waves = sine_waves * uv + noise
        return sine_waves


class SourceModuleHnNSF(torch.nn.Module):
    """ SourceModule for hn-nsf
    SourceModule(sampling_rate, harmonic_num=0, sine_amp=0.1,
                 add_noise_std=0.003, voiced_threshod=0)
    sampling_rate: sampling_rate in Hz
    harmonic_num: number of harmonic above F0 (default: 0)
    sine_amp: amplitude of sine source signal (default: 0.1)
    add_noise_std: std of additive Gaussian noise (default: 0.003)
        note that amplitude of noise in unvoiced is decided
        by sine_amp
    voiced_threshold: threhold to set U/V given F0 (default: 0)
    Sine_source, noise_source = SourceModuleHnNSF(F0_sampled)
    F0_sampled (batchsize, length, 1)
    Sine_source (batchsize, length, 1)
    noise_source (batchsize, length 1)
    uv (batchsize, length, 1)
    """

    def __init__(self, sampling_rate, harmonic_num=0, sine_amp=0.1, add_noise_std=0.003, voiced_threshod=0):
        super(SourceModuleHnNSF, self).__init__()
        self.sine_amp = sine_amp
        self.noise_std = add_noise_std
        self.l_sin_gen = SineGen(sampling_rate, harmonic_num, sine_amp, add_noise_std, voiced_threshod)
        self.l_linear = torch.nn.Linear(harmonic_num + 1, 1)
        self.l_tanh = torch.nn.Tanh()

    def forward(self, x, upp):
        sine_wavs = self.l_sin_gen(x, upp)
        sine_merge = self.l_tanh(self.l_linear(sine_wavs))
        return sine_merge


class Generator(torch.nn.Module):

    def __init__(self, h):
        super(Generator, self).__init__()
        self.h = h
        self.num_kernels = len(h.resblock_kernel_sizes)
        self.num_upsamples = len(h.upsample_rates)
        self.m_source = SourceModuleHnNSF(sampling_rate=h.sampling_rate, harmonic_num=8)
        self.noise_convs = nn.ModuleList()
        self.conv_pre = weight_norm(Conv1d(h.num_mels, h.upsample_initial_channel, 7, 1, padding=3))
        resblock = ResBlock1 if h.resblock == '1' else ResBlock2
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(h.upsample_rates, h.upsample_kernel_sizes)):
            c_cur = h.upsample_initial_channel // 2 ** (i + 1)
            self.ups.append(weight_norm(ConvTranspose1d(h.upsample_initial_channel // 2 ** i, h.upsample_initial_channel // 2 ** (i + 1), k, u, padding=(k - u) // 2)))
            if i + 1 < len(h.upsample_rates):
                stride_f0 = int(np.prod(h.upsample_rates[i + 1:]))
                self.noise_convs.append(Conv1d(1, c_cur, kernel_size=stride_f0 * 2, stride=stride_f0, padding=stride_f0 // 2))
            else:
                self.noise_convs.append(Conv1d(1, c_cur, kernel_size=1))
        self.resblocks = nn.ModuleList()
        ch = h.upsample_initial_channel
        for i in range(len(self.ups)):
            ch //= 2
            for j, (k, d) in enumerate(zip(h.resblock_kernel_sizes, h.resblock_dilation_sizes)):
                self.resblocks.append(resblock(h, ch, k, d))
        self.conv_post = weight_norm(Conv1d(ch, 1, 7, 1, padding=3))
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)
        self.upp = int(np.prod(h.upsample_rates))

    def forward(self, x, f0):
        har_source = self.m_source(f0, self.upp).transpose(1, 2)
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            x_source = self.noise_convs[i](har_source)
            x = x + x_source
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)
        return x

    def remove_weight_norm(self):
        None
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)


def load_model(model_path, device='cuda'):
    h = load_config(model_path)
    generator = Generator(h)
    cp_dict = torch.load(model_path, map_location=device)
    generator.load_state_dict(cp_dict['generator'])
    generator.eval()
    generator.remove_weight_norm()
    del cp_dict
    return generator, h


class NsfHifiGAN(torch.nn.Module):

    def __init__(self, model_path, device=None):
        super().__init__()
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        self.model_path = model_path
        self.model = None
        self.h = load_config(model_path)
        self.stft = STFT(self.h.sampling_rate, self.h.num_mels, self.h.n_fft, self.h.win_size, self.h.hop_size, self.h.fmin, self.h.fmax)

    def sample_rate(self):
        return self.h.sampling_rate

    def hop_size(self):
        return self.h.hop_size

    def dimension(self):
        return self.h.num_mels

    def extract(self, audio, keyshift=0):
        mel = self.stft.get_mel(audio, keyshift=keyshift).transpose(1, 2)
        return mel

    def forward(self, mel, f0):
        if self.model is None:
            None
            self.model, self.h = load_model(self.model_path, device=self.device)
        with torch.no_grad():
            c = mel.transpose(1, 2)
            audio = self.model(c, f0)
            return audio


class NsfHifiGANLog10(NsfHifiGAN):

    def forward(self, mel, f0):
        if self.model is None:
            None
            self.model, self.h = load_model(self.model_path, device=self.device)
        with torch.no_grad():
            c = 0.434294 * mel.transpose(1, 2)
            audio = self.model(c, f0)
            return audio


class RectifiedFlow(nn.Module):

    def __init__(self, velocity_fn, out_dims=128, spec_min=-12, spec_max=2):
        super().__init__()
        self.velocity_fn = velocity_fn
        self.out_dims = out_dims
        self.spec_min = spec_min
        self.spec_max = spec_max

    def reflow_loss(self, x_1, t, cond, loss_type='l2_lognorm'):
        x_0 = torch.randn_like(x_1)
        x_t = x_0 + t[:, None, None, None] * (x_1 - x_0)
        v_pred = self.velocity_fn(x_t, 1000 * t, cond)
        if loss_type == 'l1':
            loss = (x_1 - x_0 - v_pred).abs().mean()
        elif loss_type == 'l2':
            loss = F.mse_loss(x_1 - x_0, v_pred)
        elif loss_type == 'l2_lognorm':
            weights = 0.398942 / t / (1 - t) * torch.exp(-0.5 * torch.log(t / (1 - t)) ** 2)
            loss = torch.mean(weights[:, None, None, None] * F.mse_loss(x_1 - x_0, v_pred, reduction='none'))
        else:
            raise NotImplementedError()
        return loss

    def sample_euler(self, x, t, dt, cond):
        x += self.velocity_fn(x, 1000 * t, cond) * dt
        t += dt
        return x, t

    def sample_rk4(self, x, t, dt, cond):
        k_1 = self.velocity_fn(x, 1000 * t, cond)
        k_2 = self.velocity_fn(x + 0.5 * k_1 * dt, 1000 * (t + 0.5 * dt), cond)
        k_3 = self.velocity_fn(x + 0.5 * k_2 * dt, 1000 * (t + 0.5 * dt), cond)
        k_4 = self.velocity_fn(x + k_3 * dt, 1000 * (t + dt), cond)
        x += (k_1 + 2 * k_2 + 2 * k_3 + k_4) * dt / 6
        t += dt
        return x, t

    def forward(self, condition, gt_spec=None, infer=True, infer_step=10, method='euler', t_start=0.0, use_tqdm=True):
        cond = condition.transpose(1, 2)
        b, device = condition.shape[0], condition.device
        if t_start < 0.0:
            t_start = 0.0
        if not infer:
            x_1 = self.norm_spec(gt_spec)
            x_1 = x_1.transpose(1, 2)[:, None, :, :]
            t = t_start + (1.0 - t_start) * torch.rand(b, device=device)
            t = torch.clip(t, 1e-07, 1 - 1e-07)
            return self.reflow_loss(x_1, t, cond=cond)
        else:
            shape = cond.shape[0], 1, self.out_dims, cond.shape[2]
            if gt_spec is None:
                x = torch.randn(shape, device=device)
                t = torch.full((b,), 0, device=device)
                dt = 1.0 / infer_step
            else:
                norm_spec = self.norm_spec(gt_spec)
                norm_spec = norm_spec.transpose(1, 2)[:, None, :, :]
                x = t_start * norm_spec + (1 - t_start) * torch.randn(shape, device=device)
                t = torch.full((b,), t_start, device=device)
                dt = (1.0 - t_start) / infer_step
            if method == 'euler':
                if use_tqdm:
                    for i in tqdm(range(infer_step), desc='sample time step', total=infer_step):
                        x, t = self.sample_euler(x, t, dt, cond)
                else:
                    for i in range(infer_step):
                        x, t = self.sample_euler(x, t, dt, cond)
            elif method == 'rk4':
                if use_tqdm:
                    for i in tqdm(range(infer_step), desc='sample time step', total=infer_step):
                        x, t = self.sample_rk4(x, t, dt, cond)
                else:
                    for i in range(infer_step):
                        x, t = self.sample_rk4(x, t, dt, cond)
            else:
                raise NotImplementedError(method)
            x = x.squeeze(1).transpose(1, 2)
            return self.denorm_spec(x)

    def norm_spec(self, x):
        return (x - self.spec_min) / (self.spec_max - self.spec_min) * 2 - 1

    def denorm_spec(self, x):
        return (x + 1) / 2 * (self.spec_max - self.spec_min) + self.spec_min


class Unit2Wav(nn.Module):

    def __init__(self, sampling_rate, block_size, win_length, n_unit, n_spk, use_pitch_aug=False, out_dims=128, n_layers=6, n_chans=512):
        super().__init__()
        self.sampling_rate = sampling_rate
        self.block_size = block_size
        self.ddsp_model = CombSubSuperFast(sampling_rate, block_size, win_length, n_unit, n_spk, use_pitch_aug)
        self.reflow_model = RectifiedFlow(NaiveV2Diff(mel_channels=out_dims, dim=n_chans, num_layers=n_layers, condition_dim=out_dims, use_mlp=False), out_dims=out_dims)

    def forward(self, units, f0, volume, spk_id=None, spk_mix_dict=None, aug_shift=None, vocoder=None, gt_spec=None, infer=True, return_wav=False, infer_step=10, method='euler', t_start=0.0, silence_front=0, use_tqdm=True):
        """
        input: 
            B x n_frames x n_unit
        return: 
            dict of B x n_frames x feat
        """
        ddsp_wav, hidden, (_, _) = self.ddsp_model(units, f0, volume, spk_id=spk_id, spk_mix_dict=spk_mix_dict, aug_shift=aug_shift, infer=infer)
        start_frame = int(silence_front * self.sampling_rate / self.block_size)
        if vocoder is not None:
            ddsp_mel = vocoder.extract(ddsp_wav[:, start_frame * self.block_size:])
        else:
            ddsp_mel = None
        if not infer:
            ddsp_loss = F.mse_loss(ddsp_mel, gt_spec)
            reflow_loss = self.reflow_model(ddsp_mel, gt_spec=gt_spec, t_start=t_start, infer=False)
            return ddsp_loss, reflow_loss
        else:
            if gt_spec is not None and ddsp_mel is None:
                ddsp_mel = gt_spec
            if t_start < 1.0:
                mel = self.reflow_model(ddsp_mel, gt_spec=ddsp_mel, infer=True, infer_step=infer_step, method=method, t_start=t_start, use_tqdm=use_tqdm)
            else:
                mel = ddsp_mel
            if return_wav:
                return vocoder.infer(mel, f0[:, -mel.shape[1]:])
            else:
                return mel


class Unit2WavFast(nn.Module):

    def __init__(self, sampling_rate, block_size, win_length, n_unit, n_spk, use_pitch_aug=False, out_dims=128, n_layers=6, n_chans=512):
        super().__init__()
        self.ddsp_model = CombSubSuperFast(sampling_rate, block_size, win_length, n_unit, n_spk, use_pitch_aug)
        self.diff_model = GaussianDiffusion(NaiveV2Diff(mel_channels=out_dims, dim=n_chans, num_layers=n_layers, condition_dim=out_dims, use_mlp=False), out_dims=out_dims)

    def forward(self, units, f0, volume, spk_id=None, spk_mix_dict=None, aug_shift=None, vocoder=None, gt_spec=None, infer=True, return_wav=False, infer_speedup=10, method='dpm-solver', k_step=None, use_tqdm=True):
        """
        input: 
            B x n_frames x n_unit
        return: 
            dict of B x n_frames x feat
        """
        ddsp_wav, hidden, (_, _) = self.ddsp_model(units, f0, volume, spk_id=spk_id, spk_mix_dict=spk_mix_dict, aug_shift=aug_shift, infer=infer)
        if vocoder is not None:
            ddsp_mel = vocoder.extract(ddsp_wav)
        else:
            ddsp_mel = None
        if not infer:
            ddsp_loss = F.mse_loss(ddsp_mel, gt_spec)
            diff_loss = self.diff_model(ddsp_mel, gt_spec=gt_spec, k_step=k_step, infer=False)
            return ddsp_loss, diff_loss
        else:
            if gt_spec is not None and ddsp_mel is None:
                ddsp_mel = gt_spec
            if k_step > 0:
                mel = self.diff_model(ddsp_mel, gt_spec=ddsp_mel, infer=True, infer_speedup=infer_speedup, method=method, k_step=k_step, use_tqdm=use_tqdm)
            else:
                mel = ddsp_mel
            if return_wav:
                return vocoder.infer(mel, f0)
            else:
                return mel


class Conv1d(torch.nn.Conv1d):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        nn.init.kaiming_normal_(self.weight)


class HubertDiscrete(Hubert):

    def __init__(self, kmeans):
        super().__init__(504)
        self.kmeans = kmeans

    @torch.inference_mode()
    def units(self, wav: 'torch.Tensor') ->torch.LongTensor:
        wav = F.pad(wav, ((400 - 320) // 2, (400 - 320) // 2))
        x, _ = self.encode(wav, layer=7)
        x = self.kmeans.predict(x.squeeze().cpu().numpy())
        return torch.tensor(x, dtype=torch.long, device=wav.device)


class ConvBlockRes(nn.Module):

    def __init__(self, in_channels, out_channels, momentum=0.01):
        super(ConvBlockRes, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False), nn.BatchNorm2d(out_channels, momentum=momentum), nn.ReLU(), nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False), nn.BatchNorm2d(out_channels, momentum=momentum), nn.ReLU())
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, (1, 1))
            self.is_shortcut = True
        else:
            self.is_shortcut = False

    def forward(self, x):
        if self.is_shortcut:
            return self.conv(x) + self.shortcut(x)
        else:
            return self.conv(x) + x


class ResEncoderBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, n_blocks=1, momentum=0.01):
        super(ResEncoderBlock, self).__init__()
        self.n_blocks = n_blocks
        self.conv = nn.ModuleList()
        self.conv.append(ConvBlockRes(in_channels, out_channels, momentum))
        for i in range(n_blocks - 1):
            self.conv.append(ConvBlockRes(out_channels, out_channels, momentum))
        self.kernel_size = kernel_size
        if self.kernel_size is not None:
            self.pool = nn.AvgPool2d(kernel_size=kernel_size)

    def forward(self, x):
        for i in range(self.n_blocks):
            x = self.conv[i](x)
        if self.kernel_size is not None:
            return x, self.pool(x)
        else:
            return x


class ResDecoderBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride, n_blocks=1, momentum=0.01):
        super(ResDecoderBlock, self).__init__()
        out_padding = (0, 1) if stride == (1, 2) else (1, 1)
        self.n_blocks = n_blocks
        self.conv1 = nn.Sequential(nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), stride=stride, padding=(1, 1), output_padding=out_padding, bias=False), nn.BatchNorm2d(out_channels, momentum=momentum), nn.ReLU())
        self.conv2 = nn.ModuleList()
        self.conv2.append(ConvBlockRes(out_channels * 2, out_channels, momentum))
        for i in range(n_blocks - 1):
            self.conv2.append(ConvBlockRes(out_channels, out_channels, momentum))

    def forward(self, x, concat_tensor):
        x = self.conv1(x)
        x = torch.cat((x, concat_tensor), dim=1)
        for i in range(self.n_blocks):
            x = self.conv2[i](x)
        return x


class Encoder(nn.Module):

    def __init__(self, in_channels, in_size, n_encoders, kernel_size, n_blocks, out_channels=16, momentum=0.01):
        super(Encoder, self).__init__()
        self.n_encoders = n_encoders
        self.bn = nn.BatchNorm2d(in_channels, momentum=momentum)
        self.layers = nn.ModuleList()
        self.latent_channels = []
        for i in range(self.n_encoders):
            self.layers.append(ResEncoderBlock(in_channels, out_channels, kernel_size, n_blocks, momentum=momentum))
            self.latent_channels.append([out_channels, in_size])
            in_channels = out_channels
            out_channels *= 2
            in_size //= 2
        self.out_size = in_size
        self.out_channel = out_channels

    def forward(self, x):
        concat_tensors = []
        x = self.bn(x)
        for i in range(self.n_encoders):
            _, x = self.layers[i](x)
            concat_tensors.append(_)
        return x, concat_tensors


class Intermediate(nn.Module):

    def __init__(self, in_channels, out_channels, n_inters, n_blocks, momentum=0.01):
        super(Intermediate, self).__init__()
        self.n_inters = n_inters
        self.layers = nn.ModuleList()
        self.layers.append(ResEncoderBlock(in_channels, out_channels, None, n_blocks, momentum))
        for i in range(self.n_inters - 1):
            self.layers.append(ResEncoderBlock(out_channels, out_channels, None, n_blocks, momentum))

    def forward(self, x):
        for i in range(self.n_inters):
            x = self.layers[i](x)
        return x


class Decoder(nn.Module):

    def __init__(self, in_channels, n_decoders, stride, n_blocks, momentum=0.01):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList()
        self.n_decoders = n_decoders
        for i in range(self.n_decoders):
            out_channels = in_channels // 2
            self.layers.append(ResDecoderBlock(in_channels, out_channels, stride, n_blocks, momentum))
            in_channels = out_channels

    def forward(self, x, concat_tensors):
        for i in range(self.n_decoders):
            x = self.layers[i](x, concat_tensors[-1 - i])
        return x


class TimbreFilter(nn.Module):

    def __init__(self, latent_rep_channels):
        super(TimbreFilter, self).__init__()
        self.layers = nn.ModuleList()
        for latent_rep in latent_rep_channels:
            self.layers.append(ConvBlockRes(latent_rep[0], latent_rep[0]))

    def forward(self, x_tensors):
        out_tensors = []
        for i, layer in enumerate(self.layers):
            out_tensors.append(layer(x_tensors[i]))
        return out_tensors


N_MELS = 128


class DeepUnet(nn.Module):

    def __init__(self, kernel_size, n_blocks, en_de_layers=5, inter_layers=4, in_channels=1, en_out_channels=16):
        super(DeepUnet, self).__init__()
        self.encoder = Encoder(in_channels, N_MELS, en_de_layers, kernel_size, n_blocks, en_out_channels)
        self.intermediate = Intermediate(self.encoder.out_channel // 2, self.encoder.out_channel, inter_layers, n_blocks)
        self.tf = TimbreFilter(self.encoder.latent_channels)
        self.decoder = Decoder(self.encoder.out_channel, en_de_layers, kernel_size, n_blocks)

    def forward(self, x):
        x, concat_tensors = self.encoder(x)
        x = self.intermediate(x)
        concat_tensors = self.tf(concat_tensors)
        x = self.decoder(x, concat_tensors)
        return x


class DeepUnet0(nn.Module):

    def __init__(self, kernel_size, n_blocks, en_de_layers=5, inter_layers=4, in_channels=1, en_out_channels=16):
        super(DeepUnet0, self).__init__()
        self.encoder = Encoder(in_channels, N_MELS, en_de_layers, kernel_size, n_blocks, en_out_channels)
        self.intermediate = Intermediate(self.encoder.out_channel // 2, self.encoder.out_channel, inter_layers, n_blocks)
        self.tf = TimbreFilter(self.encoder.latent_channels)
        self.decoder = Decoder(self.encoder.out_channel, en_de_layers, kernel_size, n_blocks)

    def forward(self, x):
        x, concat_tensors = self.encoder(x)
        x = self.intermediate(x)
        x = self.decoder(x, concat_tensors)
        return x


class BiGRU(nn.Module):

    def __init__(self, input_features, hidden_features, num_layers):
        super(BiGRU, self).__init__()
        self.gru = nn.GRU(input_features, hidden_features, num_layers=num_layers, batch_first=True, bidirectional=True)

    def forward(self, x):
        return self.gru(x)[0]


N_CLASS = 360


class E2E(nn.Module):

    def __init__(self, n_blocks, n_gru, kernel_size, en_de_layers=5, inter_layers=4, in_channels=1, en_out_channels=16):
        super(E2E, self).__init__()
        self.unet = DeepUnet(kernel_size, n_blocks, en_de_layers, inter_layers, in_channels, en_out_channels)
        self.cnn = nn.Conv2d(en_out_channels, 3, (3, 3), padding=(1, 1))
        if n_gru:
            self.fc = nn.Sequential(BiGRU(3 * N_MELS, 256, n_gru), nn.Linear(512, N_CLASS), nn.Dropout(0.25), nn.Sigmoid())
        else:
            self.fc = nn.Sequential(nn.Linear(3 * N_MELS, N_CLASS), nn.Dropout(0.25), nn.Sigmoid())

    def forward(self, mel):
        mel = mel.transpose(-1, -2).unsqueeze(1)
        x = self.cnn(self.unet(mel)).transpose(1, 2).flatten(-2)
        x = self.fc(x)
        return x


class E2E0(nn.Module):

    def __init__(self, n_blocks, n_gru, kernel_size, en_de_layers=5, inter_layers=4, in_channels=1, en_out_channels=16):
        super(E2E0, self).__init__()
        self.unet = DeepUnet0(kernel_size, n_blocks, en_de_layers, inter_layers, in_channels, en_out_channels)
        self.cnn = nn.Conv2d(en_out_channels, 3, (3, 3), padding=(1, 1))
        if n_gru:
            self.fc = nn.Sequential(BiGRU(3 * N_MELS, 256, n_gru), nn.Linear(512, N_CLASS), nn.Dropout(0.25), nn.Sigmoid())
        else:
            self.fc = nn.Sequential(nn.Linear(3 * N_MELS, N_CLASS), nn.Dropout(0.25), nn.Sigmoid())

    def forward(self, mel):
        mel = mel.transpose(-1, -2).unsqueeze(1)
        x = self.cnn(self.unet(mel)).transpose(1, 2).flatten(-2)
        x = self.fc(x)
        return x


class BiLSTM(nn.Module):

    def __init__(self, input_features, hidden_features, num_layers):
        super(BiLSTM, self).__init__()
        self.lstm = nn.LSTM(input_features, hidden_features, num_layers=num_layers, batch_first=True, bidirectional=True)

    def forward(self, x):
        return self.lstm(x)[0]


class MelSpectrogram(torch.nn.Module):

    def __init__(self, n_mel_channels, sampling_rate, win_length, hop_length, n_fft=None, mel_fmin=0, mel_fmax=None, clamp=1e-05):
        super().__init__()
        n_fft = win_length if n_fft is None else n_fft
        self.hann_window = {}
        mel_basis = mel(sr=sampling_rate, n_fft=n_fft, n_mels=n_mel_channels, fmin=mel_fmin, fmax=mel_fmax, htk=True)
        mel_basis = torch.from_numpy(mel_basis).float()
        self.register_buffer('mel_basis', mel_basis)
        self.n_fft = win_length if n_fft is None else n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.sampling_rate = sampling_rate
        self.n_mel_channels = n_mel_channels
        self.clamp = clamp

    def forward(self, audio, keyshift=0, speed=1, center=True):
        factor = 2 ** (keyshift / 12)
        n_fft_new = int(np.round(self.n_fft * factor))
        win_length_new = int(np.round(self.win_length * factor))
        hop_length_new = int(np.round(self.hop_length * speed))
        keyshift_key = str(keyshift) + '_' + str(audio.device)
        if keyshift_key not in self.hann_window:
            self.hann_window[keyshift_key] = torch.hann_window(win_length_new)
        fft = torch.stft(audio, n_fft=n_fft_new, hop_length=hop_length_new, win_length=win_length_new, window=self.hann_window[keyshift_key], center=center, return_complex=True)
        magnitude = torch.sqrt(fft.real.pow(2) + fft.imag.pow(2))
        if keyshift != 0:
            size = self.n_fft // 2 + 1
            resize = magnitude.size(1)
            if resize < size:
                magnitude = F.pad(magnitude, (0, 0, 0, size - resize))
            magnitude = magnitude[:, :size, :] * self.win_length / win_length_new
        mel_output = torch.matmul(self.mel_basis, magnitude)
        log_mel_spec = torch.log(torch.clamp(mel_output, min=self.clamp))
        return log_mel_spec


class DiscriminatorP(torch.nn.Module):

    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
        super(DiscriminatorP, self).__init__()
        self.period = period
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList([norm_f(Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))), norm_f(Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))), norm_f(Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))), norm_f(Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))), norm_f(Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(2, 0)))])
        self.conv_post = norm_f(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        fmap = []
        b, c, t = x.shape
        if t % self.period != 0:
            n_pad = self.period - t % self.period
            x = F.pad(x, (0, n_pad), 'reflect')
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)
        return x, fmap


class MultiPeriodDiscriminator(torch.nn.Module):

    def __init__(self, periods=None):
        super(MultiPeriodDiscriminator, self).__init__()
        self.periods = periods if periods is not None else [2, 3, 5, 7, 11]
        self.discriminators = nn.ModuleList()
        for period in self.periods:
            self.discriminators.append(DiscriminatorP(period))

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)
        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class DiscriminatorS(torch.nn.Module):

    def __init__(self, use_spectral_norm=False):
        super(DiscriminatorS, self).__init__()
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList([norm_f(Conv1d(1, 128, 15, 1, padding=7)), norm_f(Conv1d(128, 128, 41, 2, groups=4, padding=20)), norm_f(Conv1d(128, 256, 41, 2, groups=16, padding=20)), norm_f(Conv1d(256, 512, 41, 4, groups=16, padding=20)), norm_f(Conv1d(512, 1024, 41, 4, groups=16, padding=20)), norm_f(Conv1d(1024, 1024, 41, 1, groups=16, padding=20)), norm_f(Conv1d(1024, 1024, 5, 1, padding=2))])
        self.conv_post = norm_f(Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        fmap = []
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)
        return x, fmap


class MultiScaleDiscriminator(torch.nn.Module):

    def __init__(self):
        super(MultiScaleDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList([DiscriminatorS(use_spectral_norm=True), DiscriminatorS(), DiscriminatorS()])
        self.meanpools = nn.ModuleList([AvgPool1d(4, 2, padding=2), AvgPool1d(4, 2, padding=2)])

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            if i != 0:
                y = self.meanpools[i - 1](y)
                y_hat = self.meanpools[i - 1](y_hat)
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)
        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (AfterDiffusion,
     lambda: ([], {'spec_max': 4, 'spec_min': 4}),
     lambda: ([torch.rand([4, 4, 4])], {})),
    (BiGRU,
     lambda: ([], {'input_features': 4, 'hidden_features': 4, 'num_layers': 1}),
     lambda: ([torch.rand([4, 4])], {})),
    (BiLSTM,
     lambda: ([], {'input_features': 4, 'hidden_features': 4, 'num_layers': 1}),
     lambda: ([torch.rand([4, 4])], {})),
    (ConformerConvModule,
     lambda: ([], {'dim': 4}),
     lambda: ([torch.rand([4, 4, 4])], {})),
    (Conv1d,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4])], {})),
    (ConvBlockRes,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (DiscriminatorP,
     lambda: ([], {'period': 4}),
     lambda: ([torch.rand([4, 1, 4])], {})),
    (DiscriminatorS,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1, 64])], {})),
    (Encoder,
     lambda: ([], {'in_channels': 4, 'in_size': 4, 'n_encoders': 4, 'kernel_size': 4, 'n_blocks': 4}),
     lambda: ([torch.rand([4, 4, 256, 256])], {})),
    (FeatureProjection,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 512, 512])], {})),
    (GLU,
     lambda: ([], {'dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4, 4])], {})),
    (Intermediate,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'n_inters': 4, 'n_blocks': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (MultiScaleDiscriminator,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1, 64]), torch.rand([4, 1, 64])], {})),
    (PositionalConvEmbedding,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 768, 768])], {})),
    (ResBlock1,
     lambda: ([], {'h': 4, 'channels': 4}),
     lambda: ([torch.rand([4, 4])], {})),
    (ResBlock2,
     lambda: ([], {'h': 4, 'channels': 4}),
     lambda: ([torch.rand([4, 4])], {})),
    (ResEncoderBlock,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ResidualBlock,
     lambda: ([], {'encoder_hidden': 4, 'residual_channels': 4, 'dilation': 1}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4]), torch.rand([4, 4])], {})),
    (Swish,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Transpose,
     lambda: ([], {'dims': [4, 4]}),
     lambda: ([torch.rand([4, 4, 4, 4, 4])], {})),
]

