
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


from torch.utils.data import DataLoader


import numpy as np


import pandas as pd


import torch


from torch.utils.data import Dataset


from sklearn.preprocessing import StandardScaler


import warnings


import torch.nn as nn


from torch import optim


import time


import matplotlib.pyplot as plt


from scipy import stats


import math


from torch.nn.functional import interpolate


import torch.nn.functional as F


from torch.nn.utils import weight_norm


from torch import Tensor


from typing import List


from typing import Tuple


from functools import partial


from torch import nn


from torch import einsum


from torch import diagonal


from math import log2


from math import ceil


from math import sqrt


from scipy.special import eval_legendre


import random


import torch.optim as optim


from torch.functional import align_tensors


from torch.nn.modules.linear import Linear


from typing import Union


from functools import lru_cache


from torch.utils.data.sampler import RandomSampler


from torch.nn.modules import loss


from sklearn.ensemble import GradientBoostingRegressor


class AutoCorrelation(nn.Module):
    """
    AutoCorrelation Mechanism with the following two phases:
    (1) period-based dependencies discovery
    (2) time delay aggregation
    This block can replace the self-attention family mechanism seamlessly.
    """

    def __init__(self, mask_flag=True, factor=1, scale=None, attention_dropout=0.1, output_attention=False):
        super(AutoCorrelation, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def time_delay_agg_training(self, values, corr):
        """
        SpeedUp version of Autocorrelation (a batch-normalization style design)
        This is for the training phase.
        """
        head = values.shape[1]
        channel = values.shape[2]
        length = values.shape[3]
        top_k = int(self.factor * math.log(length))
        mean_value = torch.mean(torch.mean(corr, dim=1), dim=1)
        index = torch.topk(torch.mean(mean_value, dim=0), top_k, dim=-1)[1]
        weights = torch.stack([mean_value[:, index[i]] for i in range(top_k)], dim=-1)
        tmp_corr = torch.softmax(weights, dim=-1)
        tmp_values = values
        delays_agg = torch.zeros_like(values).float()
        for i in range(top_k):
            pattern = torch.roll(tmp_values, -int(index[i]), -1)
            delays_agg = delays_agg + pattern * tmp_corr[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, head, channel, length)
        return delays_agg

    def time_delay_agg_inference(self, values, corr):
        """
        SpeedUp version of Autocorrelation (a batch-normalization style design)
        This is for the inference phase.
        """
        batch = values.shape[0]
        head = values.shape[1]
        channel = values.shape[2]
        length = values.shape[3]
        init_index = torch.arange(length).unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(batch, head, channel, 1)
        top_k = int(self.factor * math.log(length))
        mean_value = torch.mean(torch.mean(corr, dim=1), dim=1)
        weights = torch.topk(mean_value, top_k, dim=-1)[0]
        delay = torch.topk(mean_value, top_k, dim=-1)[1]
        tmp_corr = torch.softmax(weights, dim=-1)
        tmp_values = values.repeat(1, 1, 1, 2)
        delays_agg = torch.zeros_like(values).float()
        for i in range(top_k):
            tmp_delay = init_index + delay[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, head, channel, length)
            pattern = torch.gather(tmp_values, dim=-1, index=tmp_delay)
            delays_agg = delays_agg + pattern * tmp_corr[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, head, channel, length)
        return delays_agg

    def time_delay_agg_full(self, values, corr):
        """
        Standard version of Autocorrelation
        """
        batch = values.shape[0]
        head = values.shape[1]
        channel = values.shape[2]
        length = values.shape[3]
        init_index = torch.arange(length).unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(batch, head, channel, 1)
        top_k = int(self.factor * math.log(length))
        weights = torch.topk(corr, top_k, dim=-1)[0]
        delay = torch.topk(corr, top_k, dim=-1)[1]
        tmp_corr = torch.softmax(weights, dim=-1)
        tmp_values = values.repeat(1, 1, 1, 2)
        delays_agg = torch.zeros_like(values).float()
        for i in range(top_k):
            tmp_delay = init_index + delay[..., i].unsqueeze(-1)
            pattern = torch.gather(tmp_values, dim=-1, index=tmp_delay)
            delays_agg = delays_agg + pattern * tmp_corr[..., i].unsqueeze(-1)
        return delays_agg

    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        if L > S:
            zeros = torch.zeros_like(queries[:, :L - S, :]).float()
            values = torch.cat([values, zeros], dim=1)
            keys = torch.cat([keys, zeros], dim=1)
        else:
            values = values[:, :L, :, :]
            keys = keys[:, :L, :, :]
        q_fft = torch.fft.rfft(queries.permute(0, 2, 3, 1).contiguous(), dim=-1)
        k_fft = torch.fft.rfft(keys.permute(0, 2, 3, 1).contiguous(), dim=-1)
        res = q_fft * torch.conj(k_fft)
        corr = torch.fft.irfft(res, dim=-1)
        if self.training:
            V = self.time_delay_agg_training(values.permute(0, 2, 3, 1).contiguous(), corr).permute(0, 3, 1, 2)
        else:
            V = self.time_delay_agg_inference(values.permute(0, 2, 3, 1).contiguous(), corr).permute(0, 3, 1, 2)
        if self.output_attention:
            return V.contiguous(), corr.permute(0, 3, 1, 2)
        else:
            return V.contiguous(), None


class AutoCorrelationLayer(nn.Module):

    def __init__(self, correlation, d_model, n_heads, d_keys=None, d_values=None):
        super(AutoCorrelationLayer, self).__init__()
        d_keys = d_keys or d_model // n_heads
        d_values = d_values or d_model // n_heads
        self.inner_correlation = correlation
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)
        out, attn = self.inner_correlation(queries, keys, values, attn_mask)
        out = out.view(B, L, -1)
        return self.out_projection(out), attn


class my_Layernorm(nn.Module):
    """
    Special designed layernorm for the seasonal part
    """

    def __init__(self, channels):
        super(my_Layernorm, self).__init__()
        self.layernorm = nn.LayerNorm(channels)

    def forward(self, x):
        x_hat = self.layernorm(x)
        bias = torch.mean(x_hat, dim=1).unsqueeze(1).repeat(1, x.shape[1], 1)
        return x_hat - bias


class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """

    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class series_decomp_multi(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, kernel_size):
        super(series_decomp_multi, self).__init__()
        self.moving_avg = [moving_avg(kernel, stride=1) for kernel in kernel_size]
        self.layer = torch.nn.Linear(1, len(kernel_size))

    def forward(self, x):
        moving_mean = []
        for func in self.moving_avg:
            moving_avg = func(x)
            moving_mean.append(moving_avg.unsqueeze(-1))
        moving_mean = torch.cat(moving_mean, dim=-1)
        moving_mean = torch.sum(moving_mean * nn.Softmax(-1)(self.layer(x.unsqueeze(-1))), dim=-1)
        res = x - moving_mean
        return res, moving_mean


class FourierDecomp(nn.Module):

    def __init__(self):
        super(FourierDecomp, self).__init__()
        pass

    def forward(self, x):
        x_ft = torch.fft.rfft(x, dim=-1)


class EncoderLayer(nn.Module):

    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation='relu'):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == 'relu' else F.gelu

    def forward(self, x, attn_mask=None):
        new_x, attn = self.attention(x, x, x, attn_mask=attn_mask)
        x = x + self.dropout(new_x)
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm2(x + y), attn


class Encoder(nn.Module):

    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        attns = []
        if self.conv_layers is not None:
            for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                x, attn = attn_layer(x, attn_mask=attn_mask)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask)
                attns.append(attn)
        if self.norm is not None:
            x = self.norm(x)
        return x, attns


class DecoderLayer(nn.Module):

    def __init__(self, self_attention, cross_attention, d_model, d_ff=None, dropout=0.1, activation='relu'):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == 'relu' else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        x = x + self.dropout(self.self_attention(x, x, x, attn_mask=x_mask)[0])
        x = self.norm1(x)
        x = x + self.dropout(self.cross_attention(x, cross, cross, attn_mask=cross_mask)[0])
        y = x = self.norm2(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm3(x + y)


class Decoder(nn.Module):

    def __init__(self, layers, norm_layer=None, projection=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)
        if self.norm is not None:
            x = self.norm(x)
        if self.projection is not None:
            x = self.projection(x)
        return x


class PositionalEmbedding(nn.Module):

    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):

    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model, kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class FixedEmbedding(nn.Module):

    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()
        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False
        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()
        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)
        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):

    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()
        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13
        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
        if freq == 't':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):
        x = x.long()
        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(self, 'minute_embed') else 0.0
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])
        return hour_x + weekday_x + day_x + month_x + minute_x


class TimeFeatureEmbedding(nn.Module):

    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()
        freq_map = {'h': 4, 't': 5, 's': 6, 'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model, bias=False)

    def forward(self, x):
        return self.embed(x)


class DataEmbedding(nn.Module):

    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding, self).__init__()
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type, freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = self.value_embedding(x) + self.temporal_embedding(x_mark) + self.position_embedding(x)
        return self.dropout(x)


class DataEmbedding_onlypos(nn.Module):

    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_onlypos, self).__init__()
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x)


class DataEmbedding_wo_pos(nn.Module):

    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_wo_pos, self).__init__()
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type, freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = self.value_embedding(x) + self.temporal_embedding(x_mark)
        return self.dropout(x)


class DataEmbedding_wo_pos_temp(nn.Module):

    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_wo_pos_temp, self).__init__()
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type, freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = self.value_embedding(x)
        return self.dropout(x)


class DataEmbedding_wo_temp(nn.Module):

    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_wo_temp, self).__init__()
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type, freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x)


def get_frequency_modes(seq_len, modes=64, mode_select_method='random'):
    """
    get modes on frequency domain:
    'random' means sampling randomly;
    'else' means sampling the lowest modes;
    """
    modes = min(modes, seq_len // 2)
    if mode_select_method == 'random':
        index = list(range(0, seq_len // 2))
        np.random.shuffle(index)
        index = index[:modes]
    else:
        index = list(range(0, modes))
    index.sort()
    return index


class FourierBlock(nn.Module):

    def __init__(self, in_channels, out_channels, seq_len, modes=0, mode_select_method='random'):
        super(FourierBlock, self).__init__()
        None
        """
        1D Fourier block. It performs representation learning on frequency domain, 
        it does FFT, linear transform, and Inverse FFT.    
        """
        self.index = get_frequency_modes(seq_len, modes=modes, mode_select_method=mode_select_method)
        None
        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(self.scale * torch.rand(8, in_channels // 8, out_channels // 8, len(self.index), dtype=torch.cfloat))

    def compl_mul1d(self, input, weights):
        return torch.einsum('bhi,hio->bho', input, weights)

    def forward(self, q, k, v, mask):
        B, L, H, E = q.shape
        x = q.permute(0, 2, 3, 1)
        x_ft = torch.fft.rfft(x, dim=-1)
        out_ft = torch.zeros(B, H, E, L // 2 + 1, device=x.device, dtype=torch.cfloat)
        for wi, i in enumerate(self.index):
            if i >= x_ft.shape[3] or wi >= out_ft.shape[3]:
                continue
            out_ft[:, :, :, wi] = self.compl_mul1d(x_ft[:, :, :, i], self.weights1[:, :, :, wi])
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x, None


class FourierCrossAttention(nn.Module):

    def __init__(self, in_channels, out_channels, seq_len_q, seq_len_kv, modes=64, mode_select_method='random', activation='tanh', policy=0):
        super(FourierCrossAttention, self).__init__()
        None
        """
        1D Fourier Cross Attention layer. It does FFT, linear transform, attention mechanism and Inverse FFT.    
        """
        self.activation = activation
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.index_q = get_frequency_modes(seq_len_q, modes=modes, mode_select_method=mode_select_method)
        self.index_kv = get_frequency_modes(seq_len_kv, modes=modes, mode_select_method=mode_select_method)
        None
        None
        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(self.scale * torch.rand(8, in_channels // 8, out_channels // 8, len(self.index_q), dtype=torch.cfloat))

    def compl_mul1d(self, input, weights):
        return torch.einsum('bhi,hio->bho', input, weights)

    def forward(self, q, k, v, mask):
        B, L, H, E = q.shape
        xq = q.permute(0, 2, 3, 1)
        xk = k.permute(0, 2, 3, 1)
        xv = v.permute(0, 2, 3, 1)
        xq_ft_ = torch.zeros(B, H, E, len(self.index_q), device=xq.device, dtype=torch.cfloat)
        xq_ft = torch.fft.rfft(xq, dim=-1)
        for i, j in enumerate(self.index_q):
            if j >= xq_ft.shape[3]:
                continue
            xq_ft_[:, :, :, i] = xq_ft[:, :, :, j]
        xk_ft_ = torch.zeros(B, H, E, len(self.index_kv), device=xq.device, dtype=torch.cfloat)
        xk_ft = torch.fft.rfft(xk, dim=-1)
        for i, j in enumerate(self.index_kv):
            if j >= xk_ft.shape[3]:
                continue
            xk_ft_[:, :, :, i] = xk_ft[:, :, :, j]
        xqk_ft = torch.einsum('bhex,bhey->bhxy', xq_ft_, xk_ft_)
        if self.activation == 'tanh':
            xqk_ft = xqk_ft.tanh()
        elif self.activation == 'softmax':
            xqk_ft = torch.softmax(abs(xqk_ft), dim=-1)
            xqk_ft = torch.complex(xqk_ft, torch.zeros_like(xqk_ft))
        else:
            raise Exception('{} actiation function is not implemented'.format(self.activation))
        xqkv_ft = torch.einsum('bhxy,bhey->bhex', xqk_ft, xk_ft_)
        xqkvw = torch.einsum('bhex,heox->bhox', xqkv_ft, self.weights1)
        out_ft = torch.zeros(B, H, E, L // 2 + 1, device=xq.device, dtype=torch.cfloat)
        for i, j in enumerate(self.index_q):
            if i >= xqkvw.shape[3] or j >= out_ft.shape[3]:
                continue
            out_ft[:, :, :, j] = xqkvw[:, :, :, i]
        out = torch.fft.irfft(out_ft / self.in_channels / self.out_channels, n=xq.size(-1))
        return out, None


def phi_(phi_c, x, lb=0, ub=1):
    mask = np.logical_or(x < lb, x > ub) * 1.0
    return np.polynomial.polynomial.Polynomial(phi_c)(x) * (1 - mask)


def get_phi_psi(k, base):
    x = Symbol('x')
    phi_coeff = np.zeros((k, k))
    phi_2x_coeff = np.zeros((k, k))
    if base == 'legendre':
        for ki in range(k):
            coeff_ = Poly(legendre(ki, 2 * x - 1), x).all_coeffs()
            phi_coeff[ki, :ki + 1] = np.flip(np.sqrt(2 * ki + 1) * np.array(coeff_).astype(np.float64))
            coeff_ = Poly(legendre(ki, 4 * x - 1), x).all_coeffs()
            phi_2x_coeff[ki, :ki + 1] = np.flip(np.sqrt(2) * np.sqrt(2 * ki + 1) * np.array(coeff_).astype(np.float64))
        psi1_coeff = np.zeros((k, k))
        psi2_coeff = np.zeros((k, k))
        for ki in range(k):
            psi1_coeff[ki, :] = phi_2x_coeff[ki, :]
            for i in range(k):
                a = phi_2x_coeff[ki, :ki + 1]
                b = phi_coeff[i, :i + 1]
                prod_ = np.convolve(a, b)
                prod_[np.abs(prod_) < 1e-08] = 0
                proj_ = (prod_ * 1 / (np.arange(len(prod_)) + 1) * np.power(0.5, 1 + np.arange(len(prod_)))).sum()
                psi1_coeff[ki, :] -= proj_ * phi_coeff[i, :]
                psi2_coeff[ki, :] -= proj_ * phi_coeff[i, :]
            for j in range(ki):
                a = phi_2x_coeff[ki, :ki + 1]
                b = psi1_coeff[j, :]
                prod_ = np.convolve(a, b)
                prod_[np.abs(prod_) < 1e-08] = 0
                proj_ = (prod_ * 1 / (np.arange(len(prod_)) + 1) * np.power(0.5, 1 + np.arange(len(prod_)))).sum()
                psi1_coeff[ki, :] -= proj_ * psi1_coeff[j, :]
                psi2_coeff[ki, :] -= proj_ * psi2_coeff[j, :]
            a = psi1_coeff[ki, :]
            prod_ = np.convolve(a, a)
            prod_[np.abs(prod_) < 1e-08] = 0
            norm1 = (prod_ * 1 / (np.arange(len(prod_)) + 1) * np.power(0.5, 1 + np.arange(len(prod_)))).sum()
            a = psi2_coeff[ki, :]
            prod_ = np.convolve(a, a)
            prod_[np.abs(prod_) < 1e-08] = 0
            norm2 = (prod_ * 1 / (np.arange(len(prod_)) + 1) * (1 - np.power(0.5, 1 + np.arange(len(prod_))))).sum()
            norm_ = np.sqrt(norm1 + norm2)
            psi1_coeff[ki, :] /= norm_
            psi2_coeff[ki, :] /= norm_
            psi1_coeff[np.abs(psi1_coeff) < 1e-08] = 0
            psi2_coeff[np.abs(psi2_coeff) < 1e-08] = 0
        phi = [np.poly1d(np.flip(phi_coeff[i, :])) for i in range(k)]
        psi1 = [np.poly1d(np.flip(psi1_coeff[i, :])) for i in range(k)]
        psi2 = [np.poly1d(np.flip(psi2_coeff[i, :])) for i in range(k)]
    elif base == 'chebyshev':
        for ki in range(k):
            if ki == 0:
                phi_coeff[ki, :ki + 1] = np.sqrt(2 / np.pi)
                phi_2x_coeff[ki, :ki + 1] = np.sqrt(2 / np.pi) * np.sqrt(2)
            else:
                coeff_ = Poly(chebyshevt(ki, 2 * x - 1), x).all_coeffs()
                phi_coeff[ki, :ki + 1] = np.flip(2 / np.sqrt(np.pi) * np.array(coeff_).astype(np.float64))
                coeff_ = Poly(chebyshevt(ki, 4 * x - 1), x).all_coeffs()
                phi_2x_coeff[ki, :ki + 1] = np.flip(np.sqrt(2) * 2 / np.sqrt(np.pi) * np.array(coeff_).astype(np.float64))
        phi = [partial(phi_, phi_coeff[i, :]) for i in range(k)]
        x = Symbol('x')
        kUse = 2 * k
        roots = Poly(chebyshevt(kUse, 2 * x - 1)).all_roots()
        x_m = np.array([rt.evalf(20) for rt in roots]).astype(np.float64)
        wm = np.pi / kUse / 2
        psi1_coeff = np.zeros((k, k))
        psi2_coeff = np.zeros((k, k))
        psi1 = [[] for _ in range(k)]
        psi2 = [[] for _ in range(k)]
        for ki in range(k):
            psi1_coeff[ki, :] = phi_2x_coeff[ki, :]
            for i in range(k):
                proj_ = (wm * phi[i](x_m) * np.sqrt(2) * phi[ki](2 * x_m)).sum()
                psi1_coeff[ki, :] -= proj_ * phi_coeff[i, :]
                psi2_coeff[ki, :] -= proj_ * phi_coeff[i, :]
            for j in range(ki):
                proj_ = (wm * psi1[j](x_m) * np.sqrt(2) * phi[ki](2 * x_m)).sum()
                psi1_coeff[ki, :] -= proj_ * psi1_coeff[j, :]
                psi2_coeff[ki, :] -= proj_ * psi2_coeff[j, :]
            psi1[ki] = partial(phi_, psi1_coeff[ki, :], lb=0, ub=0.5)
            psi2[ki] = partial(phi_, psi2_coeff[ki, :], lb=0.5, ub=1)
            norm1 = (wm * psi1[ki](x_m) * psi1[ki](x_m)).sum()
            norm2 = (wm * psi2[ki](x_m) * psi2[ki](x_m)).sum()
            norm_ = np.sqrt(norm1 + norm2)
            psi1_coeff[ki, :] /= norm_
            psi2_coeff[ki, :] /= norm_
            psi1_coeff[np.abs(psi1_coeff) < 1e-08] = 0
            psi2_coeff[np.abs(psi2_coeff) < 1e-08] = 0
            psi1[ki] = partial(phi_, psi1_coeff[ki, :], lb=0, ub=0.5 + 1e-16)
            psi2[ki] = partial(phi_, psi2_coeff[ki, :], lb=0.5 + 1e-16, ub=1)
    return phi, psi1, psi2


def legendreDer(k, x):

    def _legendre(k, x):
        return (2 * k + 1) * eval_legendre(k, x)
    out = 0
    for i in np.arange(k - 1, -1, -2):
        out += _legendre(i, x)
    return out


def get_filter(base, k):

    def psi(psi1, psi2, i, inp):
        mask = (inp <= 0.5) * 1.0
        return psi1[i](inp) * mask + psi2[i](inp) * (1 - mask)
    if base not in ['legendre', 'chebyshev']:
        raise Exception('Base not supported')
    x = Symbol('x')
    H0 = np.zeros((k, k))
    H1 = np.zeros((k, k))
    G0 = np.zeros((k, k))
    G1 = np.zeros((k, k))
    PHI0 = np.zeros((k, k))
    PHI1 = np.zeros((k, k))
    phi, psi1, psi2 = get_phi_psi(k, base)
    if base == 'legendre':
        roots = Poly(legendre(k, 2 * x - 1)).all_roots()
        x_m = np.array([rt.evalf(20) for rt in roots]).astype(np.float64)
        wm = 1 / k / legendreDer(k, 2 * x_m - 1) / eval_legendre(k - 1, 2 * x_m - 1)
        for ki in range(k):
            for kpi in range(k):
                H0[ki, kpi] = 1 / np.sqrt(2) * (wm * phi[ki](x_m / 2) * phi[kpi](x_m)).sum()
                G0[ki, kpi] = 1 / np.sqrt(2) * (wm * psi(psi1, psi2, ki, x_m / 2) * phi[kpi](x_m)).sum()
                H1[ki, kpi] = 1 / np.sqrt(2) * (wm * phi[ki]((x_m + 1) / 2) * phi[kpi](x_m)).sum()
                G1[ki, kpi] = 1 / np.sqrt(2) * (wm * psi(psi1, psi2, ki, (x_m + 1) / 2) * phi[kpi](x_m)).sum()
        PHI0 = np.eye(k)
        PHI1 = np.eye(k)
    elif base == 'chebyshev':
        x = Symbol('x')
        kUse = 2 * k
        roots = Poly(chebyshevt(kUse, 2 * x - 1)).all_roots()
        x_m = np.array([rt.evalf(20) for rt in roots]).astype(np.float64)
        wm = np.pi / kUse / 2
        for ki in range(k):
            for kpi in range(k):
                H0[ki, kpi] = 1 / np.sqrt(2) * (wm * phi[ki](x_m / 2) * phi[kpi](x_m)).sum()
                G0[ki, kpi] = 1 / np.sqrt(2) * (wm * psi(psi1, psi2, ki, x_m / 2) * phi[kpi](x_m)).sum()
                H1[ki, kpi] = 1 / np.sqrt(2) * (wm * phi[ki]((x_m + 1) / 2) * phi[kpi](x_m)).sum()
                G1[ki, kpi] = 1 / np.sqrt(2) * (wm * psi(psi1, psi2, ki, (x_m + 1) / 2) * phi[kpi](x_m)).sum()
                PHI0[ki, kpi] = (wm * phi[ki](2 * x_m) * phi[kpi](2 * x_m)).sum() * 2
                PHI1[ki, kpi] = (wm * phi[ki](2 * x_m - 1) * phi[kpi](2 * x_m - 1)).sum() * 2
        PHI0[np.abs(PHI0) < 1e-08] = 0
        PHI1[np.abs(PHI1) < 1e-08] = 0
    H0[np.abs(H0) < 1e-08] = 0
    H1[np.abs(H1) < 1e-08] = 0
    G0[np.abs(G0) < 1e-08] = 0
    G1[np.abs(G1) < 1e-08] = 0
    return H0, H1, G0, G1, PHI0, PHI1


class sparseKernelFT1d(nn.Module):

    def __init__(self, k, alpha, c=1, nl=1, initializer=None, **kwargs):
        super(sparseKernelFT1d, self).__init__()
        self.modes1 = alpha
        self.scale = 1 / (c * k * c * k)
        self.weights1 = nn.Parameter(self.scale * torch.rand(c * k, c * k, self.modes1, dtype=torch.cfloat))
        self.weights1.requires_grad = True
        self.k = k

    def compl_mul1d(self, x, weights):
        return torch.einsum('bix,iox->box', x, weights)

    def forward(self, x):
        B, N, c, k = x.shape
        x = x.view(B, N, -1)
        x = x.permute(0, 2, 1)
        x_fft = torch.fft.rfft(x)
        l = min(self.modes1, N // 2 + 1)
        out_ft = torch.zeros(B, c * k, N // 2 + 1, device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :l] = self.compl_mul1d(x_fft[:, :, :l], self.weights1[:, :, :l])
        x = torch.fft.irfft(out_ft, n=N)
        x = x.permute(0, 2, 1).view(B, N, c, k)
        return x


class MWT_CZ1d(nn.Module):

    def __init__(self, k=3, alpha=64, L=0, c=1, base='legendre', initializer=None, **kwargs):
        super(MWT_CZ1d, self).__init__()
        self.k = k
        self.L = L
        H0, H1, G0, G1, PHI0, PHI1 = get_filter(base, k)
        H0r = H0 @ PHI0
        G0r = G0 @ PHI0
        H1r = H1 @ PHI1
        G1r = G1 @ PHI1
        H0r[np.abs(H0r) < 1e-08] = 0
        H1r[np.abs(H1r) < 1e-08] = 0
        G0r[np.abs(G0r) < 1e-08] = 0
        G1r[np.abs(G1r) < 1e-08] = 0
        self.max_item = 3
        self.A = sparseKernelFT1d(k, alpha, c)
        self.B = sparseKernelFT1d(k, alpha, c)
        self.C = sparseKernelFT1d(k, alpha, c)
        self.T0 = nn.Linear(k, k)
        self.register_buffer('ec_s', torch.Tensor(np.concatenate((H0.T, H1.T), axis=0)))
        self.register_buffer('ec_d', torch.Tensor(np.concatenate((G0.T, G1.T), axis=0)))
        self.register_buffer('rc_e', torch.Tensor(np.concatenate((H0r, G0r), axis=0)))
        self.register_buffer('rc_o', torch.Tensor(np.concatenate((H1r, G1r), axis=0)))

    def forward(self, x):
        B, N, c, k = x.shape
        ns = math.floor(np.log2(N))
        nl = pow(2, math.ceil(np.log2(N)))
        extra_x = x[:, 0:nl - N, :, :]
        x = torch.cat([x, extra_x], 1)
        Ud = torch.jit.annotate(List[Tensor], [])
        Us = torch.jit.annotate(List[Tensor], [])
        for i in range(ns - self.L):
            d, x = self.wavelet_transform(x)
            Ud += [self.A(d) + self.B(x)]
            Us += [self.C(d)]
        x = self.T0(x)
        for i in range(ns - 1 - self.L, -1, -1):
            x = x + Us[i]
            x = torch.cat((x, Ud[i]), -1)
            x = self.evenOdd(x)
        x = x[:, :N, :, :]
        return x

    def wavelet_transform(self, x):
        xa = torch.cat([x[:, ::2, :, :], x[:, 1::2, :, :]], -1)
        d = torch.matmul(xa, self.ec_d)
        s = torch.matmul(xa, self.ec_s)
        return d, s

    def evenOdd(self, x):
        B, N, c, ich = x.shape
        assert ich == 2 * self.k
        x_e = torch.matmul(x, self.rc_e)
        x_o = torch.matmul(x, self.rc_o)
        x = torch.zeros(B, N * 2, c, self.k, device=x.device)
        x[..., ::2, :, :] = x_e
        x[..., 1::2, :, :] = x_o
        return x


class MultiWaveletTransform(nn.Module):
    """
    1D multiwavelet block.
    """

    def __init__(self, ich=1, k=8, alpha=16, c=128, nCZ=1, L=0, base='legendre', attention_dropout=0.1):
        super(MultiWaveletTransform, self).__init__()
        None
        self.k = k
        self.c = c
        self.L = L
        self.nCZ = nCZ
        self.Lk0 = nn.Linear(ich, c * k)
        self.Lk1 = nn.Linear(c * k, ich)
        self.ich = ich
        self.MWT_CZ = nn.ModuleList(MWT_CZ1d(k, alpha, L, c, base) for i in range(nCZ))

    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        if L > S:
            zeros = torch.zeros_like(queries[:, :L - S, :]).float()
            values = torch.cat([values, zeros], dim=1)
            keys = torch.cat([keys, zeros], dim=1)
        else:
            values = values[:, :L, :, :]
            keys = keys[:, :L, :, :]
        values = values.view(B, L, -1)
        V = self.Lk0(values).view(B, L, self.c, -1)
        for i in range(self.nCZ):
            V = self.MWT_CZ[i](V)
            if i < self.nCZ - 1:
                V = F.relu(V)
        V = self.Lk1(V.view(B, L, -1))
        V = V.view(B, L, -1, D)
        return V.contiguous(), None


class FourierCrossAttentionW(nn.Module):

    def __init__(self, in_channels, out_channels, seq_len_q, seq_len_kv, modes=16, activation='tanh', mode_select_method='random'):
        super(FourierCrossAttentionW, self).__init__()
        None
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes
        self.activation = activation

    def forward(self, q, k, v, mask):
        B, L, E, H = q.shape
        xq = q.permute(0, 3, 2, 1)
        xk = k.permute(0, 3, 2, 1)
        xv = v.permute(0, 3, 2, 1)
        self.index_q = list(range(0, min(int(L // 2), self.modes1)))
        self.index_k_v = list(range(0, min(int(xv.shape[3] // 2), self.modes1)))
        xq_ft_ = torch.zeros(B, H, E, len(self.index_q), device=xq.device, dtype=torch.cfloat)
        xq_ft = torch.fft.rfft(xq, dim=-1)
        for i, j in enumerate(self.index_q):
            xq_ft_[:, :, :, i] = xq_ft[:, :, :, j]
        xk_ft_ = torch.zeros(B, H, E, len(self.index_k_v), device=xq.device, dtype=torch.cfloat)
        xk_ft = torch.fft.rfft(xk, dim=-1)
        for i, j in enumerate(self.index_k_v):
            xk_ft_[:, :, :, i] = xk_ft[:, :, :, j]
        xqk_ft = torch.einsum('bhex,bhey->bhxy', xq_ft_, xk_ft_)
        if self.activation == 'tanh':
            xqk_ft = xqk_ft.tanh()
        elif self.activation == 'softmax':
            xqk_ft = torch.softmax(abs(xqk_ft), dim=-1)
            xqk_ft = torch.complex(xqk_ft, torch.zeros_like(xqk_ft))
        else:
            raise Exception('{} actiation function is not implemented'.format(self.activation))
        xqkv_ft = torch.einsum('bhxy,bhey->bhex', xqk_ft, xk_ft_)
        xqkvw = xqkv_ft
        out_ft = torch.zeros(B, H, E, L // 2 + 1, device=xq.device, dtype=torch.cfloat)
        for i, j in enumerate(self.index_q):
            out_ft[:, :, :, j] = xqkvw[:, :, :, i]
        out = torch.fft.irfft(out_ft / self.in_channels / self.out_channels, n=xq.size(-1)).permute(0, 3, 2, 1)
        return out, None


class MultiWaveletCross(nn.Module):
    """
    1D Multiwavelet Cross Attention layer.
    """

    def __init__(self, in_channels, out_channels, seq_len_q, seq_len_kv, modes, c=64, k=8, ich=512, L=0, base='legendre', mode_select_method='random', initializer=None, activation='tanh', **kwargs):
        super(MultiWaveletCross, self).__init__()
        None
        self.c = c
        self.k = k
        self.L = L
        H0, H1, G0, G1, PHI0, PHI1 = get_filter(base, k)
        H0r = H0 @ PHI0
        G0r = G0 @ PHI0
        H1r = H1 @ PHI1
        G1r = G1 @ PHI1
        H0r[np.abs(H0r) < 1e-08] = 0
        H1r[np.abs(H1r) < 1e-08] = 0
        G0r[np.abs(G0r) < 1e-08] = 0
        G1r[np.abs(G1r) < 1e-08] = 0
        self.max_item = 3
        self.attn1 = FourierCrossAttentionW(in_channels=in_channels, out_channels=out_channels, seq_len_q=seq_len_q, seq_len_kv=seq_len_kv, modes=modes, activation=activation, mode_select_method=mode_select_method)
        self.attn2 = FourierCrossAttentionW(in_channels=in_channels, out_channels=out_channels, seq_len_q=seq_len_q, seq_len_kv=seq_len_kv, modes=modes, activation=activation, mode_select_method=mode_select_method)
        self.attn3 = FourierCrossAttentionW(in_channels=in_channels, out_channels=out_channels, seq_len_q=seq_len_q, seq_len_kv=seq_len_kv, modes=modes, activation=activation, mode_select_method=mode_select_method)
        self.attn4 = FourierCrossAttentionW(in_channels=in_channels, out_channels=out_channels, seq_len_q=seq_len_q, seq_len_kv=seq_len_kv, modes=modes, activation=activation, mode_select_method=mode_select_method)
        self.T0 = nn.Linear(k, k)
        self.register_buffer('ec_s', torch.Tensor(np.concatenate((H0.T, H1.T), axis=0)))
        self.register_buffer('ec_d', torch.Tensor(np.concatenate((G0.T, G1.T), axis=0)))
        self.register_buffer('rc_e', torch.Tensor(np.concatenate((H0r, G0r), axis=0)))
        self.register_buffer('rc_o', torch.Tensor(np.concatenate((H1r, G1r), axis=0)))
        self.Lk = nn.Linear(ich, c * k)
        self.Lq = nn.Linear(ich, c * k)
        self.Lv = nn.Linear(ich, c * k)
        self.out = nn.Linear(c * k, ich)
        self.modes1 = modes

    def forward(self, q, k, v, mask=None):
        B, N, H, E = q.shape
        _, S, _, _ = k.shape
        q = q.view(q.shape[0], q.shape[1], -1)
        k = k.view(k.shape[0], k.shape[1], -1)
        v = v.view(v.shape[0], v.shape[1], -1)
        q = self.Lq(q)
        q = q.view(q.shape[0], q.shape[1], self.c, self.k)
        k = self.Lk(k)
        k = k.view(k.shape[0], k.shape[1], self.c, self.k)
        v = self.Lv(v)
        v = v.view(v.shape[0], v.shape[1], self.c, self.k)
        if N > S:
            zeros = torch.zeros_like(q[:, :N - S, :]).float()
            v = torch.cat([v, zeros], dim=1)
            k = torch.cat([k, zeros], dim=1)
        else:
            v = v[:, :N, :, :]
            k = k[:, :N, :, :]
        ns = math.floor(np.log2(N))
        nl = pow(2, math.ceil(np.log2(N)))
        extra_q = q[:, 0:nl - N, :, :]
        extra_k = k[:, 0:nl - N, :, :]
        extra_v = v[:, 0:nl - N, :, :]
        q = torch.cat([q, extra_q], 1)
        k = torch.cat([k, extra_k], 1)
        v = torch.cat([v, extra_v], 1)
        Ud_q = torch.jit.annotate(List[Tuple[Tensor]], [])
        Ud_k = torch.jit.annotate(List[Tuple[Tensor]], [])
        Ud_v = torch.jit.annotate(List[Tuple[Tensor]], [])
        Us_q = torch.jit.annotate(List[Tensor], [])
        Us_k = torch.jit.annotate(List[Tensor], [])
        Us_v = torch.jit.annotate(List[Tensor], [])
        Ud = torch.jit.annotate(List[Tensor], [])
        Us = torch.jit.annotate(List[Tensor], [])
        for i in range(ns - self.L):
            d, q = self.wavelet_transform(q)
            Ud_q += [tuple([d, q])]
            Us_q += [d]
        for i in range(ns - self.L):
            d, k = self.wavelet_transform(k)
            Ud_k += [tuple([d, k])]
            Us_k += [d]
        for i in range(ns - self.L):
            d, v = self.wavelet_transform(v)
            Ud_v += [tuple([d, v])]
            Us_v += [d]
        for i in range(ns - self.L):
            dk, sk = Ud_k[i], Us_k[i]
            dq, sq = Ud_q[i], Us_q[i]
            dv, sv = Ud_v[i], Us_v[i]
            Ud += [self.attn1(dq[0], dk[0], dv[0], mask)[0] + self.attn2(dq[1], dk[1], dv[1], mask)[0]]
            Us += [self.attn3(sq, sk, sv, mask)[0]]
        v = self.attn4(q, k, v, mask)[0]
        for i in range(ns - 1 - self.L, -1, -1):
            v = v + Us[i]
            v = torch.cat((v, Ud[i]), -1)
            v = self.evenOdd(v)
        v = self.out(v[:, :N, :, :].contiguous().view(B, N, -1))
        return v.contiguous(), None

    def wavelet_transform(self, x):
        xa = torch.cat([x[:, ::2, :, :], x[:, 1::2, :, :]], -1)
        d = torch.matmul(xa, self.ec_d)
        s = torch.matmul(xa, self.ec_s)
        return d, s

    def evenOdd(self, x):
        B, N, c, ich = x.shape
        assert ich == 2 * self.k
        x_e = torch.matmul(x, self.rc_e)
        x_o = torch.matmul(x, self.rc_o)
        x = torch.zeros(B, N * 2, c, self.k, device=x.device)
        x[..., ::2, :, :] = x_e
        x[..., 1::2, :, :] = x_o
        return x


class TriangularCausalMask:

    def __init__(self, B, L, device='cpu'):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1)

    @property
    def mask(self):
        return self._mask


class FullAttention(nn.Module):

    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1.0 / sqrt(E)
        scores = torch.einsum('blhe,bshe->bhls', queries, keys)
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum('bhls,bshd->blhd', A, values)
        if self.output_attention:
            return V.contiguous(), A
        else:
            return V.contiguous(), None


class LocalMask:

    def __init__(self, B, L, S, device='cpu'):
        mask_shape = [B, 1, L, S]
        with torch.no_grad():
            self.len = math.ceil(np.log2(L))
            self._mask1 = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1)
            self._mask2 = ~torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=-self.len)
            self._mask = self._mask1 + self._mask2

    @property
    def mask(self):
        return self._mask


class SparseAttention(nn.Module):

    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1.0 / sqrt(E)
        scores = torch.einsum('blhe,bshe->bhls', queries, keys)
        if attn_mask is None:
            attn_mask = LocalMask(B, L, device=queries.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum('bhls,bshd->blhd', A, values)
        if self.output_attention:
            return V.contiguous(), A
        else:
            return V.contiguous(), None


class ProbMask:

    def __init__(self, B, H, L, index, scores, device='cpu'):
        _mask = torch.ones(L, scores.shape[-1], dtype=torch.bool).triu(1)
        _mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1])
        indicator = _mask_ex[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :]
        self._mask = indicator.view(scores.shape)

    @property
    def mask(self):
        return self._mask


class ProbAttention(nn.Module):

    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top):
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        index_sample = torch.randint(L_K, (L_Q, sample_k))
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze()
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]
        Q_reduce = Q[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], M_top, :]
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))
        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            V_sum = V.mean(dim=-2)
            contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()
        else:
            assert L_Q == L_V
            contex = V.cumsum(dim=-2)
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        B, H, L_V, D = V.shape
        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)
        attn = torch.softmax(scores, dim=-1)
        context_in[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :] = torch.matmul(attn, V).type_as(context_in)
        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V]) / L_V).type_as(attn)
            attns[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :] = attn
            return context_in, attns
        else:
            return context_in, None

    def forward(self, queries, keys, values, attn_mask):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape
        queries = queries.transpose(2, 1)
        keys = keys.transpose(2, 1)
        values = values.transpose(2, 1)
        U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item()
        u = self.factor * np.ceil(np.log(L_Q)).astype('int').item()
        U_part = U_part if U_part < L_K else L_K
        u = u if u < L_Q else L_Q
        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u)
        scale = self.scale or 1.0 / sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
        context = self._get_initial_context(values, L_Q)
        context, attn = self._update_context(context, values, scores_top, index, L_Q, attn_mask)
        return context.contiguous(), attn


class AttentionLayer(nn.Module):

    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None):
        super(AttentionLayer, self).__init__()
        d_keys = d_keys or d_model // n_heads
        d_values = d_values or d_model // n_heads
        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)
        out, attn = self.inner_attention(queries, keys, values, attn_mask)
        out = out.view(B, L, -1)
        return self.out_projection(out), attn


class ConvLayer(nn.Module):

    def __init__(self, c_in):
        super(ConvLayer, self).__init__()
        self.downConv = nn.Conv1d(in_channels=c_in, out_channels=c_in, kernel_size=3, padding=2, padding_mode='circular')
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.downConv(x.permute(0, 2, 1))
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)
        x = x.transpose(1, 2)
        return x


class Model(nn.Module):
    """
    Vanilla Transformer with O(L^2) complexity
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        if configs.embed_type == 0:
            self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout)
            self.dec_embedding = DataEmbedding(configs.dec_in, configs.d_model, configs.embed, configs.freq, configs.dropout)
        elif configs.embed_type == 1:
            self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout)
            self.dec_embedding = DataEmbedding(configs.dec_in, configs.d_model, configs.embed, configs.freq, configs.dropout)
        elif configs.embed_type == 2:
            self.enc_embedding = DataEmbedding_wo_pos(configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout)
            self.dec_embedding = DataEmbedding_wo_pos(configs.dec_in, configs.d_model, configs.embed, configs.freq, configs.dropout)
        elif configs.embed_type == 3:
            self.enc_embedding = DataEmbedding_wo_temp(configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout)
            self.dec_embedding = DataEmbedding_wo_temp(configs.dec_in, configs.d_model, configs.embed, configs.freq, configs.dropout)
        elif configs.embed_type == 4:
            self.enc_embedding = DataEmbedding_wo_pos_temp(configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout)
            self.dec_embedding = DataEmbedding_wo_pos_temp(configs.dec_in, configs.d_model, configs.embed, configs.freq, configs.dropout)
        self.encoder = Encoder([EncoderLayer(AttentionLayer(FullAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=configs.output_attention), configs.d_model, configs.n_heads), configs.d_model, configs.d_ff, dropout=configs.dropout, activation=configs.activation) for l in range(configs.e_layers)], norm_layer=torch.nn.LayerNorm(configs.d_model))
        self.decoder = Decoder([DecoderLayer(AttentionLayer(FullAttention(True, configs.factor, attention_dropout=configs.dropout, output_attention=False), configs.d_model, configs.n_heads), AttentionLayer(FullAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=False), configs.d_model, configs.n_heads), configs.d_model, configs.d_ff, dropout=configs.dropout, activation=configs.activation) for l in range(configs.d_layers)], norm_layer=torch.nn.LayerNorm(configs.d_model), projection=nn.Linear(configs.d_model, configs.c_out, bias=True))

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]


class Conv_Construct(nn.Module):
    """Convolution CSCM"""

    def __init__(self, d_model, window_size, d_inner):
        super(Conv_Construct, self).__init__()
        if not isinstance(window_size, list):
            self.conv_layers = nn.ModuleList([ConvLayer(d_model, window_size), ConvLayer(d_model, window_size), ConvLayer(d_model, window_size)])
        else:
            self.conv_layers = nn.ModuleList([ConvLayer(d_model, window_size[0]), ConvLayer(d_model, window_size[1]), ConvLayer(d_model, window_size[2])])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, enc_input):
        all_inputs = []
        enc_input = enc_input.permute(0, 2, 1)
        all_inputs.append(enc_input)
        for i in range(len(self.conv_layers)):
            enc_input = self.conv_layers[i](enc_input)
            all_inputs.append(enc_input)
        all_inputs = torch.cat(all_inputs, dim=2).transpose(1, 2)
        all_inputs = self.norm(all_inputs)
        return all_inputs


class Bottleneck_Construct(nn.Module):
    """Bottleneck convolution CSCM"""

    def __init__(self, d_model, window_size, d_inner):
        super(Bottleneck_Construct, self).__init__()
        if not isinstance(window_size, list):
            self.conv_layers = nn.ModuleList([ConvLayer(d_inner, window_size), ConvLayer(d_inner, window_size), ConvLayer(d_inner, window_size)])
        else:
            self.conv_layers = []
            for i in range(len(window_size)):
                self.conv_layers.append(ConvLayer(d_inner, window_size[i]))
            self.conv_layers = nn.ModuleList(self.conv_layers)
        self.up = Linear(d_inner, d_model)
        self.down = Linear(d_model, d_inner)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, enc_input):
        temp_input = self.down(enc_input).permute(0, 2, 1)
        all_inputs = []
        for i in range(len(self.conv_layers)):
            temp_input = self.conv_layers[i](temp_input)
            all_inputs.append(temp_input)
        all_inputs = torch.cat(all_inputs, dim=2).transpose(1, 2)
        all_inputs = self.up(all_inputs)
        all_inputs = torch.cat([enc_input, all_inputs], dim=1)
        all_inputs = self.norm(all_inputs)
        return all_inputs


class MaxPooling_Construct(nn.Module):
    """Max pooling CSCM"""

    def __init__(self, d_model, window_size, d_inner):
        super(MaxPooling_Construct, self).__init__()
        if not isinstance(window_size, list):
            self.pooling_layers = nn.ModuleList([nn.MaxPool1d(kernel_size=window_size), nn.MaxPool1d(kernel_size=window_size), nn.MaxPool1d(kernel_size=window_size)])
        else:
            self.pooling_layers = nn.ModuleList([nn.MaxPool1d(kernel_size=window_size[0]), nn.MaxPool1d(kernel_size=window_size[1]), nn.MaxPool1d(kernel_size=window_size[2])])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, enc_input):
        all_inputs = []
        enc_input = enc_input.transpose(1, 2).contiguous()
        all_inputs.append(enc_input)
        for layer in self.pooling_layers:
            enc_input = layer(enc_input)
            all_inputs.append(enc_input)
        all_inputs = torch.cat(all_inputs, dim=2).transpose(1, 2)
        all_inputs = self.norm(all_inputs)
        return all_inputs


class AvgPooling_Construct(nn.Module):
    """Average pooling CSCM"""

    def __init__(self, d_model, window_size, d_inner):
        super(AvgPooling_Construct, self).__init__()
        if not isinstance(window_size, list):
            self.pooling_layers = nn.ModuleList([nn.AvgPool1d(kernel_size=window_size), nn.AvgPool1d(kernel_size=window_size), nn.AvgPool1d(kernel_size=window_size)])
        else:
            self.pooling_layers = nn.ModuleList([nn.AvgPool1d(kernel_size=window_size[0]), nn.AvgPool1d(kernel_size=window_size[1]), nn.AvgPool1d(kernel_size=window_size[2])])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, enc_input):
        all_inputs = []
        enc_input = enc_input.transpose(1, 2).contiguous()
        all_inputs.append(enc_input)
        for layer in self.pooling_layers:
            enc_input = layer(enc_input)
            all_inputs.append(enc_input)
        all_inputs = torch.cat(all_inputs, dim=2).transpose(1, 2)
        all_inputs = self.norm(all_inputs)
        return all_inputs


class Predictor(nn.Module):

    def __init__(self, dim, num_types):
        super().__init__()
        self.linear = nn.Linear(dim, num_types, bias=False)
        nn.init.xavier_normal_(self.linear.weight)

    def forward(self, data):
        out = self.linear(data)
        out = out
        return out


class ScaledDotProductAttention(nn.Module):
    """ Scaled Dot-Product Attention """

    def __init__(self, temperature, attn_dropout=0.2):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))
        if mask is not None:
            attn = attn.masked_fill(mask, -1000000000.0)
        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)
        return output, attn


class PyramidalAttention(nn.Module):

    def __init__(self, n_head, d_model, d_k, d_v, dropout, normalize_before, q_k_mask, k_q_mask):
        super(PyramidalAttention, self).__init__()
        self.normalize_before = normalize_before
        self.n_head = n_head
        self.d_k = d_k
        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_k, bias=False)
        nn.init.xavier_uniform_(self.w_qs.weight)
        nn.init.xavier_uniform_(self.w_ks.weight)
        nn.init.xavier_uniform_(self.w_vs.weight)
        self.fc = nn.Linear(d_k * n_head, d_model)
        nn.init.xavier_uniform_(self.fc.weight)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-06)
        self.dropout_attn = nn.Dropout(dropout)
        self.dropout_fc = nn.Dropout(dropout)
        self.q_k_mask = q_k_mask
        self.k_q_mask = k_q_mask

    def forward(self, hidden_states):
        residual = hidden_states
        hidden_states = hidden_states
        bsz, seq_len, _ = hidden_states.size()
        q = hidden_states
        if self.normalize_before:
            q = self.layer_norm(q)
        q = self.w_qs(q)
        k = self.w_ks(hidden_states)
        v = self.w_vs(hidden_states)
        q /= math.sqrt(self.d_k)
        q = q.view(bsz, seq_len, self.n_head, self.d_k)
        k = k.view(bsz, seq_len, self.n_head, self.d_k)
        q = q.float().contiguous()
        k = k.float().contiguous()
        attn_weights = graph_mm_tvm(q, k, self.q_k_mask, self.k_q_mask, False, 0)
        attn_weights = self.dropout_attn(F.softmax(attn_weights, dim=-1))
        v = v.view(bsz, seq_len, self.n_head, self.d_k)
        v = v.float().contiguous()
        attn = graph_mm_tvm(attn_weights, v, self.q_k_mask, self.k_q_mask, True, 0)
        attn = attn.reshape(bsz, seq_len, self.n_head * self.d_k).contiguous()
        context = self.dropout_fc(self.fc(attn))
        context += residual
        if not self.normalize_before:
            context = self.layer_norm(context)
        return context


class MultiHeadAttention(nn.Module):
    """ Multi-Head Attention module """

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1, normalize_before=True):
        super().__init__()
        self.normalize_before = normalize_before
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        nn.init.xavier_uniform_(self.w_qs.weight)
        nn.init.xavier_uniform_(self.w_ks.weight)
        nn.init.xavier_uniform_(self.w_vs.weight)
        self.fc = nn.Linear(d_v * n_head, d_model)
        nn.init.xavier_uniform_(self.fc.weight)
        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5, attn_dropout=dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-06)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)
        residual = q
        if self.normalize_before:
            q = self.layer_norm(q)
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        if mask is not None:
            if len(mask.size()) == 3:
                mask = mask.unsqueeze(1)
        output, attn = self.attention(q, k, v, mask=mask)
        output = output.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        output = self.dropout(self.fc(output))
        output += residual
        if not self.normalize_before:
            output = self.layer_norm(output)
        return output, attn


class PositionwiseFeedForward(nn.Module):
    """ Two-layer position-wise feed-forward neural network. """

    def __init__(self, d_in, d_hid, dropout=0.1, normalize_before=True):
        super().__init__()
        self.normalize_before = normalize_before
        self.w_1 = nn.Linear(d_in, d_hid)
        self.w_2 = nn.Linear(d_hid, d_in)
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-06)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        if self.normalize_before:
            x = self.layer_norm(x)
        x = F.gelu(self.w_1(x))
        x = self.dropout(x)
        x = self.w_2(x)
        x = self.dropout(x)
        x = x + residual
        if not self.normalize_before:
            x = self.layer_norm(x)
        return x


class CustomEmbedding(nn.Module):

    def __init__(self, c_in, d_model, temporal_size, seq_num, dropout=0.1):
        super(CustomEmbedding, self).__init__()
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = nn.Linear(temporal_size, d_model)
        self.seqid_embedding = nn.Embedding(seq_num, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = self.value_embedding(x) + self.position_embedding(x) + self.temporal_embedding(x_mark[:, :, :-1]) + self.seqid_embedding(x_mark[:, :, -1].long())
        return self.dropout(x)


class SingleStepEmbedding(nn.Module):

    def __init__(self, cov_size, num_seq, d_model, input_size, device):
        super().__init__()
        self.cov_size = cov_size
        self.num_class = num_seq
        self.cov_emb = nn.Linear(cov_size + 1, d_model)
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.data_emb = nn.Conv1d(in_channels=1, out_channels=d_model, kernel_size=3, padding=padding, padding_mode='circular')
        self.position = torch.arange(input_size, device=device).unsqueeze(0)
        self.position_vec = torch.tensor([math.pow(10000.0, 2.0 * (i // 2) / d_model) for i in range(d_model)], device=device)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def transformer_embedding(self, position, vector):
        """
        Input: batch*seq_len.
        Output: batch*seq_len*d_model.
        """
        result = position.unsqueeze(-1) / vector
        result[:, :, 0::2] = torch.sin(result[:, :, 0::2])
        result[:, :, 1::2] = torch.cos(result[:, :, 1::2])
        return result

    def forward(self, x):
        covs = x[:, :, 1:1 + self.cov_size]
        seq_ids = (x[:, :, -1] / self.num_class - 0.5).unsqueeze(2)
        covs = torch.cat([covs, seq_ids], dim=-1)
        cov_embedding = self.cov_emb(covs)
        data_embedding = self.data_emb(x[:, :, 0].unsqueeze(2).permute(0, 2, 1)).transpose(1, 2)
        embedding = cov_embedding + data_embedding
        position = self.position.repeat(len(x), 1)
        position_emb = self.transformer_embedding(position, self.position_vec)
        embedding += position_emb
        return embedding


def get_k_q(q_k_mask):
    """Get the key-query index from query-key index for PAM-TVM"""
    k_q_mask = q_k_mask.clone()
    for i in range(len(q_k_mask)):
        for j in range(len(q_k_mask[0])):
            if q_k_mask[i, j] >= 0:
                k_q_mask[i, j] = torch.where(q_k_mask[q_k_mask[i, j]] == i)[0]
    return k_q_mask


def get_q_k(input_size, window_size, stride, device):
    """Get the query-key index for PAM-TVM"""
    second_length = input_size // stride
    second_last = input_size - (second_length - 1) * stride
    third_start = input_size + second_length
    third_length = second_length // stride
    third_last = second_length - (third_length - 1) * stride
    max_attn = max(second_last, third_last)
    fourth_start = third_start + third_length
    fourth_length = third_length // stride
    full_length = fourth_start + fourth_length
    fourth_last = third_length - (fourth_length - 1) * stride
    max_attn = max(third_last, fourth_last)
    max_attn += window_size + 1
    mask = torch.zeros(full_length, max_attn, dtype=torch.int32, device=device) - 1
    for i in range(input_size):
        mask[i, 0:window_size] = i + torch.arange(window_size) - window_size // 2
        mask[i, mask[i] > input_size - 1] = -1
        mask[i, -1] = i // stride + input_size
        mask[i][mask[i] > third_start - 1] = third_start - 1
    for i in range(second_length):
        mask[input_size + i, 0:window_size] = input_size + i + torch.arange(window_size) - window_size // 2
        mask[input_size + i, mask[input_size + i] < input_size] = -1
        mask[input_size + i, mask[input_size + i] > third_start - 1] = -1
        if i < second_length - 1:
            mask[input_size + i, window_size:window_size + stride] = torch.arange(stride) + i * stride
        else:
            mask[input_size + i, window_size:window_size + second_last] = torch.arange(second_last) + i * stride
        mask[input_size + i, -1] = i // stride + third_start
        mask[input_size + i, mask[input_size + i] > fourth_start - 1] = fourth_start - 1
    for i in range(third_length):
        mask[third_start + i, 0:window_size] = third_start + i + torch.arange(window_size) - window_size // 2
        mask[third_start + i, mask[third_start + i] < third_start] = -1
        mask[third_start + i, mask[third_start + i] > fourth_start - 1] = -1
        if i < third_length - 1:
            mask[third_start + i, window_size:window_size + stride] = input_size + torch.arange(stride) + i * stride
        else:
            mask[third_start + i, window_size:window_size + third_last] = input_size + torch.arange(third_last) + i * stride
        mask[third_start + i, -1] = i // stride + fourth_start
        mask[third_start + i, mask[third_start + i] > full_length - 1] = full_length - 1
    for i in range(fourth_length):
        mask[fourth_start + i, 0:window_size] = fourth_start + i + torch.arange(window_size) - window_size // 2
        mask[fourth_start + i, mask[fourth_start + i] < fourth_start] = -1
        mask[fourth_start + i, mask[fourth_start + i] > full_length - 1] = -1
        if i < fourth_length - 1:
            mask[fourth_start + i, window_size:window_size + stride] = third_start + torch.arange(stride) + i * stride
        else:
            mask[fourth_start + i, window_size:window_size + fourth_last] = third_start + torch.arange(fourth_last) + i * stride
    return mask


class GraphSelfAttention(nn.Module):

    def __init__(self, opt):
        super(GraphSelfAttention, self).__init__()
        self.normalize_before = opt.normalize_before
        self.n_head = opt.n_head
        self.d_k = opt.d_k
        self.w_qs = nn.Linear(opt.d_model, opt.n_head * opt.d_k, bias=False)
        self.w_ks = nn.Linear(opt.d_model, opt.n_head * opt.d_k, bias=False)
        self.w_vs = nn.Linear(opt.d_model, opt.n_head * opt.d_k, bias=False)
        nn.init.xavier_uniform_(self.w_qs.weight)
        nn.init.xavier_uniform_(self.w_ks.weight)
        nn.init.xavier_uniform_(self.w_vs.weight)
        self.fc = nn.Linear(opt.d_k * opt.n_head, opt.d_model)
        nn.init.xavier_uniform_(self.fc.weight)
        self.layer_norm = nn.LayerNorm(opt.d_model, eps=1e-06)
        self.dropout_attn = nn.Dropout(opt.dropout)
        self.dropout_fc = nn.Dropout(opt.dropout)
        self.seq_len = opt.seq_len
        self.window_size = opt.window_size
        self.stride_size = opt.stride_size
        self.q_k_mask = get_q_k(self.seq_len, self.window_size, self.stride_size, opt.device)
        self.k_q_mask = get_k_q(self.q_k_mask)

    def forward(self, hidden_states):
        residual = hidden_states
        hidden_states = hidden_states
        bsz, seq_len, _ = hidden_states.size()
        q = hidden_states
        if self.normalize_before:
            q = self.layer_norm(q)
        q = self.w_qs(q)
        k = self.w_ks(hidden_states)
        v = self.w_vs(hidden_states)
        q /= math.sqrt(self.d_k)
        q = q.view(bsz, seq_len, self.n_head, self.d_k)
        k = k.view(bsz, seq_len, self.n_head, self.d_k)
        q = q.float().contiguous()
        k = k.float().contiguous()
        attn_weights = graph_mm_tvm(q, k, self.q_k_mask, self.k_q_mask, False, 0)
        attn_weights = self.dropout_attn(F.softmax(attn_weights, dim=-1))
        v = v.view(bsz, seq_len, self.n_head, self.d_k)
        v = v.float().contiguous()
        attn = graph_mm_tvm(attn_weights, v, self.q_k_mask, self.k_q_mask, True, 0)
        attn = attn.reshape(bsz, seq_len, self.n_head * self.d_k).contiguous()
        context = self.dropout_fc(self.fc(attn))
        context += residual
        if not self.normalize_before:
            context = self.layer_norm(context)
        return context


def get_mask(input_size, window_size, inner_size, device):
    """Get the attention mask of PAM-Naive"""
    all_size = []
    all_size.append(input_size)
    second_size = math.floor(input_size / window_size)
    all_size.append(second_size)
    third_size = math.floor(second_size / window_size)
    all_size.append(third_size)
    fourth_size = math.floor(third_size / window_size)
    all_size.append(fourth_size)
    seq_length = sum(all_size)
    mask = torch.zeros(seq_length, seq_length, device=device)
    inner_window = inner_size // 2
    for i in range(input_size):
        left_side = max(i - inner_window, 0)
        right_side = min(i + inner_window + 1, input_size)
        mask[i, left_side:right_side] = 1
    start = input_size
    for i in range(start, start + second_size):
        left_side = max(i - inner_window, start)
        right_side = min(i + inner_window + 1, start + second_size)
        mask[i, left_side:right_side] = 1
    start = input_size + second_size
    for i in range(start, start + third_size):
        left_side = max(i - inner_window, start)
        right_side = min(i + inner_window + 1, start + third_size)
        mask[i, left_side:right_side] = 1
    start = input_size + second_size + third_size
    for i in range(start, start + fourth_size):
        left_side = max(i - inner_window, start)
        right_side = min(i + inner_window + 1, start + fourth_size)
        mask[i, left_side:right_side] = 1
    start = input_size
    for i in range(start, start + second_size):
        left_side = (i - input_size) * window_size
        if i == start + second_size - 1:
            right_side = start
        else:
            right_side = (i - input_size + 1) * window_size
        mask[i, left_side:right_side] = 1
        mask[left_side:right_side, i] = 1
    start = input_size + second_size
    for i in range(start, start + third_size):
        left_side = input_size + (i - start) * window_size
        if i == start + third_size - 1:
            right_side = start
        else:
            right_side = input_size + (i - start + 1) * window_size
        mask[i, left_side:right_side] = 1
        mask[left_side:right_side, i] = 1
    start = input_size + second_size + third_size
    for i in range(start, start + fourth_size):
        left_side = input_size + second_size + (i - start) * window_size
        if i == start + fourth_size - 1:
            right_side = start
        else:
            right_side = input_size + second_size + (i - start + 1) * window_size
        mask[i, left_side:right_side] = 1
        mask[left_side:right_side, i] = 1
    mask = (1 - mask).bool()
    return mask, all_size


class NormalSelfAttention(nn.Module):

    def __init__(self, opt):
        super(NormalSelfAttention, self).__init__()
        self.normalize_before = opt.normalize_before
        self.n_head = opt.n_head
        self.d_k = opt.d_k
        self.w_qs = nn.Linear(opt.d_model, opt.n_head * opt.d_k, bias=False)
        self.w_ks = nn.Linear(opt.d_model, opt.n_head * opt.d_k, bias=False)
        self.w_vs = nn.Linear(opt.d_model, opt.n_head * opt.d_k, bias=False)
        nn.init.xavier_uniform_(self.w_qs.weight)
        nn.init.xavier_uniform_(self.w_ks.weight)
        nn.init.xavier_uniform_(self.w_vs.weight)
        self.fc = nn.Linear(opt.d_k * opt.n_head, opt.d_model)
        nn.init.xavier_uniform_(self.fc.weight)
        self.layer_norm = nn.LayerNorm(opt.d_model, eps=1e-06)
        self.dropout_attn = nn.Dropout(opt.dropout)
        self.dropout_fc = nn.Dropout(opt.dropout)
        self.seq_len = opt.seq_len
        self.window_size = opt.window_size
        self.stride_size = opt.stride_size
        if opt.mask:
            self.mask, _ = get_mask(self.seq_len, self.stride_size, self.window_size, opt.device)
        else:
            self.mask = None

    def forward(self, hidden_states):
        residual = hidden_states
        hidden_states = hidden_states
        bsz, seq_len, _ = hidden_states.size()
        q = hidden_states
        if self.normalize_before:
            q = self.layer_norm(q)
        q = self.w_qs(q)
        k = self.w_ks(hidden_states)
        v = self.w_vs(hidden_states)
        q /= math.sqrt(self.d_k)
        q = q.view(bsz, seq_len, self.n_head, self.d_k).transpose(1, 2)
        k = k.view(bsz, seq_len, self.n_head, self.d_k).transpose(1, 2)
        v = v.view(bsz, seq_len, self.n_head, self.d_k).transpose(1, 2)
        q = q.float().contiguous()
        k = k.float().contiguous()
        v = v.float().contiguous()
        attn = torch.matmul(q, k.transpose(2, 3))
        if self.mask is not None:
            attn = attn.masked_fill(self.mask.unsqueeze(0).unsqueeze(1), -1000000000.0)
        attn = self.dropout_attn(F.softmax(attn, dim=-1))
        attn = torch.matmul(attn, v).transpose(1, 2).contiguous()
        attn = attn.view(bsz, seq_len, self.n_head * self.d_k)
        context = self.dropout_fc(self.fc(attn))
        context += residual
        if not self.normalize_before:
            context = self.layer_norm(context)
        return context


class ProbSparseAttention(nn.Module):

    def __init__(self, opt):
        super(ProbSparseAttention, self).__init__()
        self.normalize_before = opt.normalize_before
        self.n_head = opt.n_head
        self.d_k = opt.d_k
        self.w_qs = nn.Linear(opt.d_model, opt.n_head * opt.d_k, bias=False)
        self.w_ks = nn.Linear(opt.d_model, opt.n_head * opt.d_k, bias=False)
        self.w_vs = nn.Linear(opt.d_model, opt.n_head * opt.d_k, bias=False)
        nn.init.xavier_uniform_(self.w_qs.weight)
        nn.init.xavier_uniform_(self.w_ks.weight)
        nn.init.xavier_uniform_(self.w_vs.weight)
        self.fc = nn.Linear(opt.d_k * opt.n_head, opt.d_model)
        nn.init.xavier_uniform_(self.fc.weight)
        self.layer_norm = nn.LayerNorm(opt.d_model, eps=1e-06)
        self.dropout_attn = nn.Dropout(opt.dropout)
        self.dropout_fc = nn.Dropout(opt.dropout)
        self.seq_len = opt.seq_len
        self.factor = opt.factor

    def _prob_QK(self, Q, K, sample_k, n_top):
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        index_sample = torch.randint(L_K, (L_Q, sample_k))
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze()
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]
        Q_reduce = Q[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], M_top, :]
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))
        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        V_sum = V.mean(dim=-2)
        contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q):
        B, H, L_V, D = V.shape
        attn = torch.softmax(scores, dim=-1)
        context_in[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :] = torch.matmul(attn, V).type_as(context_in)
        return context_in

    def forward(self, hidden_states):
        residual = hidden_states
        hidden_states = hidden_states
        bsz, seq_len, _ = hidden_states.size()
        q = hidden_states
        if self.normalize_before:
            q = self.layer_norm(q)
        q = self.w_qs(q)
        k = self.w_ks(hidden_states)
        v = self.w_vs(hidden_states)
        q /= math.sqrt(self.d_k)
        q = q.view(bsz, seq_len, self.n_head, self.d_k).transpose(1, 2)
        k = k.view(bsz, seq_len, self.n_head, self.d_k).transpose(1, 2)
        v = v.view(bsz, seq_len, self.n_head, self.d_k).transpose(1, 2)
        q = q.float().contiguous()
        k = k.float().contiguous()
        v = v.float().contiguous()
        u = U_part = self.factor * np.ceil(np.log(seq_len)).astype('int').item()
        U_part = U_part if U_part < seq_len else seq_len
        u = u if u < seq_len else seq_len
        scores_top, index = self._prob_QK(q, k, sample_k=U_part, n_top=u)
        context = self._get_initial_context(v, seq_len)
        context = self._update_context(context, v, scores_top, index, seq_len).transpose(1, 2).contiguous()
        context = context.view(bsz, seq_len, self.n_head * self.d_k)
        context = self.dropout_fc(self.fc(context))
        context += residual
        if not self.normalize_before:
            context = self.layer_norm(context)
        return context


class TopkMSELoss(torch.nn.Module):

    def __init__(self, topk) ->None:
        super().__init__()
        self.topk = topk
        self.criterion = torch.nn.MSELoss(reduction='none')

    def forward(self, output, label):
        losses = self.criterion(output, label).mean(2).mean(1)
        losses = torch.topk(losses, self.topk)[0]
        return losses


class SingleStepLoss(torch.nn.Module):
    """ Compute top-k log-likelihood and mse. """

    def __init__(self, ignore_zero):
        super().__init__()
        self.ignore_zero = ignore_zero

    def forward(self, mu, sigma, labels, topk=0):
        if self.ignore_zero:
            indexes = labels != 0
        else:
            indexes = labels >= 0
        distribution = torch.distributions.normal.Normal(mu[indexes], sigma[indexes])
        likelihood = -distribution.log_prob(labels[indexes])
        diff = labels[indexes] - mu[indexes]
        se = diff * diff
        if 0 < topk < len(likelihood):
            likelihood = torch.topk(likelihood, topk)[0]
            se = torch.topk(se, topk)[0]
        return likelihood, se


class Naive_repeat(nn.Module):

    def __init__(self, configs):
        super(Naive_repeat, self).__init__()
        self.pred_len = configs.pred_len

    def forward(self, x):
        B, L, D = x.shape
        x = x[:, -1, :].reshape(B, 1, D).repeat(self.pred_len, axis=1)
        return x


def _arima(seq, pred_len, bt, i):
    model = pm.auto_arima(seq)
    forecasts = model.predict(pred_len)
    return forecasts, bt, i


class Arima(nn.Module):
    """
    Extremely slow, please sample < 0.1
    """

    def __init__(self, configs):
        super(Arima, self).__init__()
        self.pred_len = configs.pred_len

    def forward(self, x):
        result = np.zeros([x.shape[0], self.pred_len, x.shape[2]])
        threads = []
        for bt, seqs in tqdm(enumerate(x)):
            for i in range(seqs.shape[-1]):
                seq = seqs[:, i]
                one_seq = Naive_thread(func=_arima, args=(seq, self.pred_len, bt, i))
                threads.append(one_seq)
                threads[-1].start()
        for every_thread in tqdm(threads):
            forcast, bt, i = every_thread.return_result()
            result[bt, :, i] = forcast
        return result


def _sarima(season, seq, pred_len, bt, i):
    model = pm.auto_arima(seq, seasonal=True, m=season)
    forecasts = model.predict(pred_len)
    return forecasts, bt, i


class SArima(nn.Module):
    """
    Extremely extremely slow, please sample < 0.01
    """

    def __init__(self, configs):
        super(SArima, self).__init__()
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.season = 24
        if 'Ettm' in configs.data_path:
            self.season = 12
        elif 'ILI' in configs.data_path:
            self.season = 1
        if self.season >= self.seq_len:
            self.season = 1

    def forward(self, x):
        result = np.zeros([x.shape[0], self.pred_len, x.shape[2]])
        threads = []
        for bt, seqs in tqdm(enumerate(x)):
            for i in range(seqs.shape[-1]):
                seq = seqs[:, i]
                one_seq = Naive_thread(func=_sarima, args=(self.season, seq, self.pred_len, bt, i))
                threads.append(one_seq)
                threads[-1].start()
        for every_thread in tqdm(threads):
            forcast, bt, i = every_thread.return_result()
            result[bt, :, i] = forcast
        return result


def _gbrt(seq, seq_len, pred_len, bt, i):
    model = GradientBoostingRegressor()
    model.fit(np.arange(seq_len).reshape(-1, 1), seq.reshape(-1, 1))
    forecasts = model.predict(np.arange(seq_len, seq_len + pred_len).reshape(-1, 1))
    return forecasts, bt, i


class GBRT(nn.Module):

    def __init__(self, configs):
        super(GBRT, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

    def forward(self, x):
        result = np.zeros([x.shape[0], self.pred_len, x.shape[2]])
        threads = []
        for bt, seqs in tqdm(enumerate(x)):
            for i in range(seqs.shape[-1]):
                seq = seqs[:, i]
                one_seq = Naive_thread(func=_gbrt, args=(seq, self.seq_len, self.pred_len, bt, i))
                threads.append(one_seq)
                threads[-1].start()
        for every_thread in tqdm(threads):
            forcast, bt, i = every_thread.return_result()
            result[bt, :, i] = forcast
        return result


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (AutoCorrelation,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (ConvLayer,
     lambda: ([], {'c_in': 4}),
     lambda: ([torch.rand([4, 4, 4])], {})),
    (DataEmbedding,
     lambda: ([], {'c_in': 4, 'd_model': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {})),
    (DataEmbedding_onlypos,
     lambda: ([], {'c_in': 4, 'd_model': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {})),
    (DataEmbedding_wo_pos,
     lambda: ([], {'c_in': 4, 'd_model': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {})),
    (DataEmbedding_wo_pos_temp,
     lambda: ([], {'c_in': 4, 'd_model': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {})),
    (DataEmbedding_wo_temp,
     lambda: ([], {'c_in': 4, 'd_model': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {})),
    (FourierBlock,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'seq_len': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (FourierCrossAttentionW,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'seq_len_q': 4, 'seq_len_kv': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (FourierDecomp,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Model,
     lambda: ([], {'configs': SimpleNamespace(pred_len=4, output_attention=4, embed_type=4, enc_in=4, d_model=4, embed=4, freq=4, dropout=0.5, dec_in=4, e_layers=1, factor=4, n_heads=4, d_ff=4, activation=4, d_layers=1, c_out=4)}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {})),
    (MultiHeadAttention,
     lambda: ([], {'n_head': 4, 'd_model': 4, 'd_k': 4, 'd_v': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {})),
    (PositionalEmbedding,
     lambda: ([], {'d_model': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (PositionwiseFeedForward,
     lambda: ([], {'d_in': 4, 'd_hid': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Predictor,
     lambda: ([], {'dim': 4, 'num_types': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ProbAttention,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (ProbSparseAttention,
     lambda: ([], {'opt': SimpleNamespace(normalize_before=4, n_head=4, d_k=4, d_model=4, dropout=0.5, seq_len=4, factor=4)}),
     lambda: ([torch.rand([4, 4, 4])], {})),
    (ScaledDotProductAttention,
     lambda: ([], {'temperature': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (SingleStepLoss,
     lambda: ([], {'ignore_zero': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (TemporalEmbedding,
     lambda: ([], {'d_model': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (TimeFeatureEmbedding,
     lambda: ([], {'d_model': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (TokenEmbedding,
     lambda: ([], {'c_in': 4, 'd_model': 4}),
     lambda: ([torch.rand([4, 4, 4])], {})),
    (TopkMSELoss,
     lambda: ([], {'topk': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (moving_avg,
     lambda: ([], {'kernel_size': 4, 'stride': 1}),
     lambda: ([torch.rand([4, 4, 4])], {})),
    (my_Layernorm,
     lambda: ([], {'channels': 4}),
     lambda: ([torch.rand([4, 4, 4])], {})),
    (series_decomp,
     lambda: ([], {'kernel_size': 4}),
     lambda: ([torch.rand([4, 2, 4])], {})),
    (series_decomp_multi,
     lambda: ([], {'kernel_size': [4, 4]}),
     lambda: ([torch.rand([4, 2, 4])], {})),
]

