
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


import torch.nn.functional as F


import time


from time import perf_counter


import numpy as np


import random


from typing import Union


import math


import torch.nn as nn


from typing import List


import inspect


from copy import copy


from itertools import product


import pandas as pd


import matplotlib.pyplot as plt


from typing import *


from torch.utils.data import Dataset


from torch.utils.data import DataLoader


import torch.autograd.profiler as profiler


def repeat_kv(hidden_states: 'torch.Tensor', n_rep: 'int') ->torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class AttentionSDPA(nn.Module):

    def __init__(self, config: 'JambaLMConfig'):
        super().__init__()
        self.config = config
        self.hidden_size = config.d_model
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.attention_dropout = config.attention_dropout
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

    def forward(self, x, cache=None):
        B, L, _ = x.size()
        queries = self.q_proj(x)
        keys = self.k_proj(x)
        values = self.v_proj(x)
        queries = queries.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        keys = keys.view(B, L, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        values = values.view(B, L, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        if cache is not None:
            past_keys, past_values = cache
            if past_keys is not None:
                keys = torch.cat([past_keys, keys], dim=2)
                values = torch.cat([past_values, values], dim=2)
            cache = keys, values
        keys = repeat_kv(keys, self.num_key_value_groups)
        values = repeat_kv(values, self.num_key_value_groups)
        attn_output = F.scaled_dot_product_attention(queries, keys, values, dropout_p=self.attention_dropout if self.training else 0.0, is_causal=cache is None)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(B, L, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        return attn_output, cache


class RMSNorm(nn.Module):

    def __init__(self, d_model: 'int', eps: 'float'=1e-05):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight
        return output


class MLP(nn.Module):

    def __init__(self, config: 'JambaLMConfig'):
        super().__init__()
        self.hidden_dim = config.d_model
        self.ffn_dim = config.mlp_size
        self.gate_proj = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)
        self.down_proj = nn.Linear(self.ffn_dim, self.hidden_dim, bias=False)
        self.up_proj = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class SparseMoEBlock(nn.Module):

    def __init__(self, config: 'JambaLMConfig', num_experts: 'int', num_experts_per_tok: 'int'):
        super().__init__()
        self.hidden_dim = config.d_model
        self.ffn_dim = config.mlp_size
        self.num_experts = num_experts
        self.top_k = num_experts_per_tok
        if num_experts > 1:
            self.router = nn.Linear(self.hidden_dim, self.num_experts, bias=False)
        else:
            self.router = None
        self.experts = nn.ModuleList([MLP(config) for _ in range(self.num_experts)])

    def forward(self, x):
        batch_size, sequence_length, hidden_dim = x.shape
        if self.num_experts == 1:
            final_hidden_states = self.experts[0](x)
            router_logits = torch.ones((batch_size * sequence_length, 1), device=x.device, dtype=x.dtype, requires_grad=x.requires_grad)
            return final_hidden_states, router_logits
        x = x.view(-1, hidden_dim)
        router_logits = self.router(x)
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        routing_weights = routing_weights
        final_hidden_states = torch.zeros((batch_size * sequence_length, hidden_dim), dtype=x.dtype, device=x.device)
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)
        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])
            if top_x.shape[0] == 0:
                continue
            top_x_list = top_x.tolist()
            idx_list = idx.tolist()
            current_state = x[None, top_x_list].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x_list, idx_list, None]
            final_hidden_states.index_add_(0, top_x, current_hidden_states)
        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states, router_logits


class AttentionLayer(nn.Module):

    def __init__(self, config: 'JambaLMConfig', num_experts: 'int'):
        super().__init__()
        self.self_attn = AttentionSDPA(config)
        num_experts_per_tok = config.num_experts_per_tok if num_experts > 1 else 1
        self.moe = SparseMoEBlock(config, num_experts=num_experts, num_experts_per_tok=num_experts_per_tok)
        self.input_layernorm = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.pre_moe_layernorm = RMSNorm(config.d_model, eps=config.rms_norm_eps)

    def forward(self, x, cache=None):
        residual = x
        x = self.input_layernorm(x)
        x, cache = self.self_attn(x, cache)
        x = residual + x
        residual = x
        x = self.pre_moe_layernorm(x)
        x, router_logits = self.moe(x)
        x = residual + x
        outputs = x, router_logits
        return outputs, cache

    def get_empty_cache(self, batch_size, device):
        return None, None


class DepthWiseConv1d(nn.Module):

    def __init__(self, channels, kernel_size, bias, padding):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.bias = bias
        self.padding = padding
        self.conv1d = nn.Conv1d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, bias=True, padding=padding)
        indices = mx.arange(channels)
        mask = mx.zeros_like(self.conv1d.weight)
        mask[indices, :, indices] = 1
        self.conv1d.weight *= mask

    def __call__(self, x):
        return self.conv1d(x)


def clamp(x, min=None, max=None):
    if min is not None:
        mask_lower = x < min
    if max is not None:
        mask_upper = x > max
    if min is not None:
        if max is not None:
            return mx.where(mask_upper, max, mx.where(mask_lower, min, x))
        return mx.where(mask_lower, min, x)
    return mx.where(mask_upper, max, x)


def npo2(len):
    """
    Returns the next power of 2 above len
    """
    return 2 ** math.ceil(math.log2(len))


def pad_npo2(X):
    """
    Pads input length dim to the next power of 2

    Args:
        X : (B, L, D, N)

    Returns:
        Y : (B, npo2(L), D, N)
    """
    len_npo2 = npo2(X.size(1))
    pad_tuple = 0, 0, 0, 0, 0, len_npo2 - X.size(1)
    return F.pad(X, pad_tuple, 'constant', 0)


class PScan(torch.autograd.Function):

    @staticmethod
    def pscan(A, X):
        B, D, L, _ = A.size()
        num_steps = int(math.log2(L))
        Aa = A
        Xa = X
        for _ in range(num_steps - 2):
            T = Xa.size(2)
            Aa = Aa.view(B, D, T // 2, 2, -1)
            Xa = Xa.view(B, D, T // 2, 2, -1)
            Xa[:, :, :, 1].add_(Aa[:, :, :, 1].mul(Xa[:, :, :, 0]))
            Aa[:, :, :, 1].mul_(Aa[:, :, :, 0])
            Aa = Aa[:, :, :, 1]
            Xa = Xa[:, :, :, 1]
        if Xa.size(2) == 4:
            Xa[:, :, 1].add_(Aa[:, :, 1].mul(Xa[:, :, 0]))
            Aa[:, :, 1].mul_(Aa[:, :, 0])
            Xa[:, :, 3].add_(Aa[:, :, 3].mul(Xa[:, :, 2] + Aa[:, :, 2].mul(Xa[:, :, 1])))
        elif Xa.size(2) == 2:
            Xa[:, :, 1].add_(Aa[:, :, 1].mul(Xa[:, :, 0]))
            return
        else:
            return
        Aa = A[:, :, 2 ** (num_steps - 2) - 1:L:2 ** (num_steps - 2)]
        Xa = X[:, :, 2 ** (num_steps - 2) - 1:L:2 ** (num_steps - 2)]
        Xa[:, :, 2].add_(Aa[:, :, 2].mul(Xa[:, :, 1]))
        Aa[:, :, 2].mul_(Aa[:, :, 1])
        for k in range(num_steps - 3, -1, -1):
            Aa = A[:, :, 2 ** k - 1:L:2 ** k]
            Xa = X[:, :, 2 ** k - 1:L:2 ** k]
            T = Xa.size(2)
            Aa = Aa.view(B, D, T // 2, 2, -1)
            Xa = Xa.view(B, D, T // 2, 2, -1)
            Xa[:, :, 1:, 0].add_(Aa[:, :, 1:, 0].mul(Xa[:, :, :-1, 1]))
            Aa[:, :, 1:, 0].mul_(Aa[:, :, :-1, 1])

    @staticmethod
    def pscan_rev(A, X):
        B, D, L, _ = A.size()
        num_steps = int(math.log2(L))
        Aa = A
        Xa = X
        for _ in range(num_steps - 2):
            T = Xa.size(2)
            Aa = Aa.view(B, D, T // 2, 2, -1)
            Xa = Xa.view(B, D, T // 2, 2, -1)
            Xa[:, :, :, 0].add_(Aa[:, :, :, 0].mul(Xa[:, :, :, 1]))
            Aa[:, :, :, 0].mul_(Aa[:, :, :, 1])
            Aa = Aa[:, :, :, 0]
            Xa = Xa[:, :, :, 0]
        if Xa.size(2) == 4:
            Xa[:, :, 2].add_(Aa[:, :, 2].mul(Xa[:, :, 3]))
            Aa[:, :, 2].mul_(Aa[:, :, 3])
            Xa[:, :, 0].add_(Aa[:, :, 0].mul(Xa[:, :, 1].add(Aa[:, :, 1].mul(Xa[:, :, 2]))))
        elif Xa.size(2) == 2:
            Xa[:, :, 0].add_(Aa[:, :, 0].mul(Xa[:, :, 1]))
            return
        else:
            return
        Aa = A[:, :, 0:L:2 ** (num_steps - 2)]
        Xa = X[:, :, 0:L:2 ** (num_steps - 2)]
        Xa[:, :, 1].add_(Aa[:, :, 1].mul(Xa[:, :, 2]))
        Aa[:, :, 1].mul_(Aa[:, :, 2])
        for k in range(num_steps - 3, -1, -1):
            Aa = A[:, :, 0:L:2 ** k]
            Xa = X[:, :, 0:L:2 ** k]
            T = Xa.size(2)
            Aa = Aa.view(B, D, T // 2, 2, -1)
            Xa = Xa.view(B, D, T // 2, 2, -1)
            Xa[:, :, :-1, 1].add_(Aa[:, :, :-1, 1].mul(Xa[:, :, 1:, 0]))
            Aa[:, :, :-1, 1].mul_(Aa[:, :, 1:, 0])

    @staticmethod
    def forward(ctx, A_in, X_in):
        """
        Applies the parallel scan operation, as defined above. Returns a new tensor.
        If you can, privilege sequence lengths that are powers of two.

        Args:
            A_in : (B, L, D, N)
            X_in : (B, L, D, N)

        Returns:
            H : (B, L, D, N)
        """
        L = X_in.size(1)
        if L == npo2(L):
            A = A_in.clone()
            X = X_in.clone()
        else:
            A = pad_npo2(A_in)
            X = pad_npo2(X_in)
        A = A.transpose(2, 1)
        X = X.transpose(2, 1)
        PScan.pscan(A, X)
        ctx.save_for_backward(A_in, X)
        return X.transpose(2, 1)[:, :L]

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
        L = grad_output_in.size(1)
        if L == npo2(L):
            grad_output = grad_output_in.clone()
        else:
            grad_output = pad_npo2(grad_output_in)
            A_in = pad_npo2(A_in)
        grad_output = grad_output.transpose(2, 1)
        A_in = A_in.transpose(2, 1)
        A = torch.nn.functional.pad(A_in[:, :, 1:], (0, 0, 0, 1))
        PScan.pscan_rev(A, grad_output)
        Q = torch.zeros_like(X)
        Q[:, :, 1:].add_(X[:, :, :-1] * grad_output[:, :, 1:])
        return Q.transpose(2, 1)[:, :L], grad_output.transpose(2, 1)[:, :L]


pscan = PScan.apply


def softplus(x, beta=1, threshold=20):
    scaled_x = beta * x
    mask = scaled_x > threshold
    return mx.where(mask, x, 1 / beta * mx.logaddexp(0, x))


def unsqueeze(x, axis):
    """
    Same API as PyTorch.
    """
    assert axis <= len(x.shape)
    if axis >= 0:
        new_shape = x.shape[:axis] + tuple([1]) + x.shape[axis:]
    else:
        new_shape = x.shape + tuple([1])
    return x.reshape(new_shape)


class MambaBlock(nn.Module):

    def __init__(self, config: 'MambaConfig'):
        super().__init__()
        self.config = config
        self.in_proj = nn.Linear(config.d_model, 2 * config.d_inner, bias=config.bias)
        self.conv1d = DepthWiseConv1d(channels=config.d_inner, kernel_size=config.d_conv, bias=config.conv_bias, padding=config.d_conv - 1)
        self.x_proj = nn.Linear(config.d_inner, config.dt_rank + 2 * config.d_state, bias=False)
        self.dt_proj = nn.Linear(config.dt_rank, config.d_inner, bias=True)
        dt_init_std = config.dt_rank ** -0.5 * config.dt_scale
        if config.dt_init == 'constant':
            self.dt_proj.weight = dt_init_std * mx.ones_like(self.dt_proj.weight)
        elif config.dt_init == 'random':
            self.dt_proj.weight = mx.random.uniform(-dt_init_std, dt_init_std, self.dt_proj.weight.shape)
        else:
            raise NotImplementedError
        dt = clamp(mx.exp(mx.random.uniform(shape=[config.d_inner]) * (math.log(config.dt_max) - math.log(config.dt_min)) + math.log(config.dt_min)), min=config.dt_init_floor)
        inv_dt = dt + mx.log1p(-mx.exp(-dt))
        self.dt_proj.bias = inv_dt
        A = mx.repeat(mx.arange(1.0, 16 + 1.0).reshape([1, 16]), repeats=config.d_inner, axis=0)
        self.A_log = mx.log(A)
        self.D = mx.ones([config.d_inner])
        self.out_proj = nn.Linear(config.d_inner, config.d_model, bias=config.bias)

    def __call__(self, x):
        _, L, _ = x.shape
        xz = self.in_proj(x)
        x, z = xz.split(indices_or_sections=2, axis=2)
        x = self.conv1d(x)[:, :L, :]
        x = nn.silu(x)
        y = self.ssm(x)
        z = nn.silu(z)
        output = y * z
        output = self.out_proj(output)
        return output

    def ssm(self, x):
        A = -mx.exp(self.A_log)
        D = self.D
        deltaBC = self.x_proj(x)
        delta, B, C = mx.split(deltaBC, indices_or_sections=[self.config.dt_rank, self.config.dt_rank + self.config.d_state], axis=-1)
        delta = softplus(self.dt_proj(delta))
        if self.config.pscan:
            y = self.selective_scan(x, delta, A, B, C, D)
        else:
            y = self.selective_scan_seq(x, delta, A, B, C, D)
        return y

    def selective_scan(self, x, delta, A, B, C, D):
        deltaA = mx.exp(unsqueeze(delta, -1) * A)
        deltaB = unsqueeze(delta, -1) * unsqueeze(B, 2)
        BX = deltaB * unsqueeze(x, -1)
        hs = pscan(deltaA, BX)
        y = (hs @ unsqueeze(C, -1)).squeeze(3)
        y = y + D * x
        return y

    def selective_scan_seq(self, x, delta, A, B, C, D):
        _, L, _ = x.shape
        deltaA = mx.exp(unsqueeze(delta, -1) * A)
        deltaB = unsqueeze(delta, -1) * unsqueeze(B, 2)
        BX = deltaB * unsqueeze(x, -1)
        h = mx.zeros([x.shape[0], self.config.d_inner, self.config.d_state])
        hs = []
        for t in range(0, L):
            h = deltaA[:, t] * h + BX[:, t]
            hs.append(h)
        hs = mx.stack(hs, axis=1)
        y = (hs @ unsqueeze(C, -1)).squeeze(3)
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
        x, z = xz.split(indices_or_sections=2, axis=1)
        x_cache = unsqueeze(x, 1)
        x = self.conv1d(mx.concatenate([inputs, x_cache], axis=1))[:, self.config.d_conv - 1, :]
        x = nn.silu(x)
        y, h = self.ssm_step(x, h)
        z = nn.silu(z)
        output = y * z
        output = self.out_proj(output)
        inputs = mx.concatenate([inputs[:, 1:, :], x_cache], axis=1)
        cache = h, inputs
        return output, cache

    def ssm_step(self, x, h):
        A = -mx.exp(self.A_log)
        D = self.D
        deltaBC = self.x_proj(x)
        delta, B, C = mx.split(deltaBC, indices_or_sections=[self.config.dt_rank, self.config.dt_rank + self.config.d_state], axis=-1)
        delta = softplus(self.dt_proj(delta))
        deltaA = mx.exp(unsqueeze(delta, -1) * A)
        deltaB = unsqueeze(delta, -1) * unsqueeze(B, 1)
        BX = deltaB * unsqueeze(x, -1)
        if h is None:
            h = mx.zeros([x.shape[0], self.config.d_inner, self.config.d_state])
        h = deltaA * h + BX
        y = (h @ unsqueeze(C, -1)).squeeze(2)
        y = y + D * x
        return y, h


class MambaLayer(nn.Module):

    def __init__(self, config: 'JambaLMConfig', num_experts: 'int'):
        super().__init__()
        self.config = config
        self.mamba = MambaBlock(config=config.mamba_config)
        num_experts_per_tok = config.num_experts_per_tok if num_experts > 1 else 1
        self.moe = SparseMoEBlock(config, num_experts=num_experts, num_experts_per_tok=num_experts_per_tok)
        self.input_layernorm = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.pre_moe_layernorm = RMSNorm(config.d_model, eps=config.rms_norm_eps)

    def forward(self, x, cache=None):
        residual = x
        x = self.input_layernorm(x)
        if cache is None:
            x = self.mamba(x)
        else:
            x, cache = self.mamba.step(x.squeeze(1), cache)
            x = x.unsqueeze(1)
        x = residual + x
        residual = x
        x = self.pre_moe_layernorm(x)
        x, router_logits = self.moe(x)
        x = residual + x
        outputs = x, router_logits
        return outputs, cache

    def get_empty_cache(self, batch_size, device):
        return None, torch.zeros(batch_size, self.config.d_inner, self.config.d_conv - 1, device=device)


class Jamba(nn.Module):

    def __init__(self, config: 'JambaLMConfig'):
        super().__init__()
        self.config = config
        decoder_layers = []
        for i in range(config.n_layers):
            is_attn = True if (i - self.config.attn_layer_offset) % self.config.attn_layer_period == 0 else False
            is_expert = True if (i - self.config.expert_layer_offset) % self.config.expert_layer_period == 0 else False
            num_experts = self.config.num_experts if is_expert else 1
            if is_attn:
                decoder_layers.append(AttentionLayer(config, num_experts=num_experts))
            else:
                decoder_layers.append(MambaLayer(config, num_experts=num_experts))
        self.layers = nn.ModuleList(decoder_layers)

    def forward(self, x):
        router_logits = []
        for decoder_layer in self.layers:
            layer_output, _ = decoder_layer(x)
            x = layer_output[0]
            router_logits.append(layer_output[1])
        return x, router_logits

    def step(self, x, caches):
        for i, decoder_layer in enumerate(self.layers):
            layer_output, caches[i] = decoder_layer(x, caches[i])
            x = layer_output[0]
        return x, caches


class JambaLM(nn.Module):

    def __init__(self, config: 'JambaLMConfig'):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embedding = nn.Embedding(config.vocab_size, config.d_model, self.padding_idx)
        self.jamba = Jamba(config)
        self.final_layernorm = RMSNorm(config.d_model, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if self.config.tie_lm_weights:
            self.lm_head.weight = self.embedding.weight
        self.apply(self._init_weights)

    def forward(self, tokens):
        x = self.embedding(tokens)
        x, router_logits = self.jamba(x)
        x = self.final_layernorm(x)
        logits = self.lm_head(x)
        if self.config.num_experts == 1:
            return logits
        else:
            return logits, router_logits

    def step(self, tokens, caches):
        x = self.embedding(tokens)
        x, caches = self.jamba.step(x, caches)
        x = self.final_layernorm(x)
        logits = self.lm_head(x)
        return logits, caches

    def generate(self, tokenizer, prompt: 'str', max_tokens: 'int'=50, batch_size: 'int'=1, sample: 'bool'=True, top_k: 'int'=40, temperature: 'float'=1.0):
        self.eval()
        input_ids = tokenizer(prompt, return_tensors='pt').input_ids
        input_ids = input_ids.repeat(batch_size, 1)
        caches = [self.jamba.layers[i].get_empty_cache(batch_size, input_ids.device) for i in range(self.config.n_layers)]
        for i in range(input_ids.size(1) + max_tokens - 1):
            with torch.no_grad():
                next_token_logits, caches = self.step(input_ids[:, [i]], caches)
                next_token_logits = next_token_logits.squeeze(1)
            if i + 1 >= input_ids.size(1):
                probs = F.softmax(next_token_logits / temperature, dim=-1)
                if top_k is not None:
                    values, _ = torch.topk(probs, k=top_k)
                    probs[probs < values[:, -1, None]] = 0
                    probs = probs / probs.sum(axis=1, keepdims=True)
                if sample:
                    next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
                else:
                    next_token = torch.argmax(probs, dim=-1)
                input_ids = torch.cat([input_ids, next_token.unsqueeze(1)], dim=1)
                if next_token.item() == tokenizer.eos_token_id:
                    break
        outputs = [tokenizer.decode(output.tolist(), skip_special_tokens=True) for output in input_ids[:, 1:]]
        self.train()
        if batch_size == 1:
            return outputs[0]
        else:
            return outputs

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class ResidualBlock(nn.Module):

    def __init__(self, config: 'MambaConfig'):
        super().__init__()
        self.mixer = MambaBlock(config)
        self.norm = nn.RMSNorm(config.d_model)

    def __call__(self, x):
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
        self.layers = [ResidualBlock(config) for _ in range(config.n_layers)]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def step(self, x, caches):
        for i, layer in enumerate(self.layers):
            x, caches[i] = layer.step(x, caches[i])
        return x, caches


class Mamba2(nn.Module):

    def __init__(self, config: 'Mamba2Config'):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([ResidualBlock(config) for _ in range(config.n_layers)])

    def forward(self, x, caches=None):
        if caches is None:
            caches = [None] * self.config.n_layers
        for i, layer in enumerate(self.layers):
            x, caches[i] = layer(x, caches[i])
        if caches[0] == None:
            return x
        else:
            return x, caches


class LM(nn.Module):

    def __init__(self, model_config: 'Union[MambaConfig, Mamba2Config]', vocab_size: 'int', pad_vocab_size_multiple: 'int'=None):
        super().__init__()
        if pad_vocab_size_multiple != None and vocab_size % pad_vocab_size_multiple != 0:
            vocab_size += pad_vocab_size_multiple - vocab_size % pad_vocab_size_multiple
        self.config = model_config
        self.embedding = nn.Embedding(vocab_size, self.config.d_model, padding_idx=0)
        if isinstance(self.config, MambaConfig):
            self.mamba = Mamba(self.config)
        elif isinstance(self.config, Mamba2Config):
            self.mamba = Mamba2(self.config)
        else:
            raise NotImplementedError
        self.norm_f = RMSNorm(self.config.d_model, self.config.rms_norm_eps, self.config.mup)
        self.lm_head = nn.Linear(self.config.d_model, vocab_size, bias=False)
        if self.config.mup and isinstance(self.config, MambaConfig):
            for pn, p in self.named_parameters():
                if any(pn.endswith(w) for w in ['mixer.in_proj.weight', 'mixer.x_proj.weight', 'mixer.dt_proj.weight', 'mixer.out_proj.weight']):
                    std = self.config.base_std
                    if 'mixer.out_proj.weight' in pn:
                        std = std / math.sqrt(2 * self.config.n_layers)
                    if 'mixer.dt_proj.weight' in pn:
                        std = self.config.dt_rank ** -0.5 * self.config.dt_scale
                    torch.nn.init.normal_(p, mean=0.0, std=std / math.sqrt(self.config.mup_width_mult))
                elif 'mixer.conv1d.weight' in pn:
                    torch.nn.init.zeros_(p)
                elif pn == 'embedding.weight':
                    torch.nn.init.normal_(p, mean=0.0, std=self.config.base_std)
                elif pn == 'lm_head.weight':
                    torch.nn.init.zeros_(p)
                elif any(pn.endswith(w) for w in ['mixer.A_log', 'mixer.D']):
                    pass
                else:
                    assert p.dim() == 1, f'a 2d param ({pn}) has not been filtered out for init. please check.'
                    if 'in_proj.bias' in pn or 'out_proj.bias' in pn:
                        torch.nn.init.zeros_(p)
        elif self.config.mup and isinstance(self.config, Mamba2Config):
            for pn, p in self.named_parameters():
                if any(pn.endswith(w) for w in ['mixer.in_proj.weight', 'mixer.out_proj.weight']):
                    std = self.config.base_std
                    if 'mixer.out_proj.weight' in pn:
                        std = std / math.sqrt(2 * self.config.n_layers)
                    torch.nn.init.normal_(p, mean=0.0, std=std / math.sqrt(self.config.mup_width_mult))
                elif 'mixer.conv1d.weight' in pn:
                    torch.nn.init.zeros_(p)
                elif pn == 'embedding.weight':
                    torch.nn.init.normal_(p, mean=0.0, std=self.config.base_std)
                elif pn == 'lm_head.weight':
                    torch.nn.init.zeros_(p)
                elif any(pn.endswith(w) for w in ['mixer.A_log', 'mixer.D', 'mixer.dt_bias']):
                    pass
                else:
                    assert p.dim() == 1, f'a 2d param ({pn}) has not been filtered out for init. please check.'
                    if 'in_proj.bias' in pn or 'out_proj.bias' in pn:
                        torch.nn.init.zeros_(p)
        else:
            self.apply(self._init_weights)
            for pn, p in self.named_parameters():
                if pn.endswith('mixer.out_proj.weight'):
                    torch.nn.init.normal_(p, mean=0.0, std=self.config.base_std / math.sqrt(2 * self.config.n_layers))

    def forward(self, tokens):
        x = self.embedding(tokens)
        x = self.mamba(x)
        x = self.norm_f(x)
        if self.config.mup:
            x = x / self.config.mup_width_mult
        logits = self.lm_head(x)
        return logits

    def generate(self, prompt, num_tokens: 'int', sample: 'bool'=True, top_k: 'int'=None, temperature: 'float'=1.0):
        if top_k is not None:
            top_k = min(top_k, self.vocab_size)
        input_device = prompt.device
        prompt = prompt
        self.eval()
        generated = prompt.clone()
        with torch.no_grad():
            for _ in range(num_tokens):
                logits = self.forward(generated)
                next_token_logits = logits[:, -1]
                if sample:
                    probs = F.softmax(next_token_logits / temperature, dim=-1)
                    if top_k is not None:
                        values, _ = torch.topk(probs, k=top_k)
                        probs[probs < values[:, -1, None]] = 0
                        probs = probs / probs.sum(axis=1, keepdims=True)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = next_token_logits.argmax(dim=-1, keepdim=True)
                generated = torch.cat([generated, next_token], dim=1)
        self.train()
        return generated[:, -num_tokens:]

    def generate4(self, prompt, num_tokens: 'int', sample: 'bool'=True, top_k: 'int'=None, temperature: 'float'=1.0):
        assert not isinstance(self.core, Mamba), "Mamba1 doesn't support decoding with the generate4 function."
        if isinstance(self.core, Mamba2):
            assert self.config.use_mem_eff_path, 'Mamba2 should use the mem_eff_path when decoding with the generate4 function'
            assert prompt.size(1) >= self.config.d_conv
        if top_k is not None:
            top_k = min(top_k, self.vocab_size)
        input_device = prompt.device
        model_device = self.embedding.weight.device
        prompt = prompt
        self.eval()
        len_prompt = prompt.size(1)
        generated = prompt.clone()
        caches = [layer.get_empty_cache(prompt.size(0)) for layer in self.core.layers]
        with torch.no_grad():
            logits, caches = self.forward(prompt, caches)
            next_token_logits = logits[:, -1]
            for t in range(num_tokens):
                if sample:
                    probs = F.softmax(next_token_logits / temperature, dim=-1)
                    if top_k is not None:
                        values, _ = torch.topk(probs, k=top_k)
                        probs[probs < values[:, -1, None]] = 0
                        probs = probs / probs.sum(axis=1, keepdims=True)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = next_token_logits.argmax(dim=-1, keepdim=True)
                generated = torch.cat([generated, next_token], dim=1)
                next_token_logits, caches = self.forward(generated[:, [len_prompt + t]], caches, seq_pos=len_prompt + t)
                next_token_logits = next_token_logits.squeeze(1)
        self.train()
        return generated[:, -num_tokens:]
    """
    def step(self, token, caches):
        # token : (B)
        # caches : [cache(layer) for all layers], cache : (h, inputs)

        # logits : (B, vocab_size)
        # caches : [cache(layer) for all layers], cache : (h, inputs)

        x = self.embedding(token)

        x, caches = self.mamba.step(x, caches)
        x = self.norm_f(x)

        if self.config.mup:
            x = x / self.config.mup_width_mult
        
        logits = self.lm_head(x)

        return logits, caches
    
    # TODO process prompt in parallel, and pass in sequential mode when prompt is finished ?
    def generate(self, tokenizer, prompt: str, num_tokens: int = 50, batch_size: int = 1, sample: bool = True, top_k: int = 40, temperature: float = 1.0):
        self.eval()

        input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(next(self.parameters()).device) # (1, num_tokens)
        input_ids = input_ids.repeat(batch_size, 1)

        # caches is a list of cache, one per layer
        # cache is composed of : the hidden state, and the last d_conv-1 inputs
        # the hidden state because the update is like an RNN
        # the last d_conv-1 inputs because they are used in a 1d convolution (usually d_conv=4 so this is not large)
        caches = [(None, torch.zeros(batch_size, self.config.d_inner, self.config.d_conv-1, device=input_ids.device)) for _ in range(self.config.n_layers)]

        for i in range(input_ids.size(1) + num_tokens - 1):
            with torch.no_grad():
                # forward the new output, get new cache
                next_token_logits, caches = self.step(input_ids[:, i], caches) # (batch_size, vocab_size), caches

            # sample (no sampling when the prompt is being processed)
            if i+1 >= input_ids.size(1):
                probs = F.softmax(next_token_logits / temperature, dim=-1) # (batch_size, vocab_size)

                if top_k is not None:
                    values, _ = torch.topk(probs, k=top_k) # (batch_size, k) ordered from lowest to biggest
                    probs[probs < values[:, -1, None]] = 0
                    probs = probs / probs.sum(axis=1, keepdims=True)

                if sample:
                    next_token = torch.multinomial(probs, num_samples=1).squeeze(1) # (batch_size)
                else:
                    next_token = torch.argmax(probs, dim=-1) # (batch_size)

                input_ids = torch.cat([input_ids, next_token.unsqueeze(1)], dim=1)
                
        outputs = [tokenizer.decode(output.tolist()) for output in input_ids]

        self.train()

        if batch_size==1:
            return outputs[0]
        else:
            return outputs
    """

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.base_std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.base_std)

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        if self.config.mup and isinstance(self.config, MambaConfig):
            mup_params_keys = set([pn for pn in param_dict.keys() if any(pn.endswith(w) for w in ['mixer.in_proj.weight', 'mixer.x_proj.weight', 'mixer.dt_proj.weight', 'mixer.out_proj.weight'])])
            dim2_params_keys = set([pn for pn in param_dict.keys() if param_dict[pn].dim() >= 2])
            dim2_params_keys = dim2_params_keys.difference(mup_params_keys)
            mup_parameters = [p for n, p in param_dict.items() if n in mup_params_keys]
            decay_params = [p for n, p in param_dict.items() if n in dim2_params_keys]
            nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
            optim_groups = [{'params': mup_parameters, 'weight_decay': weight_decay * self.config.mup_width_mult, 'lr': learning_rate / self.config.mup_width_mult}, {'params': decay_params, 'weight_decay': weight_decay, 'lr': learning_rate}, {'params': nodecay_params, 'weight_decay': 0.0, 'lr': learning_rate}]
        elif self.config.mup and isinstance(self.config, Mamba2Config):
            mup_params_keys = set([pn for pn in param_dict.keys() if any(pn.endswith(w) for w in ['mixer.in_proj.weight', 'mixer.out_proj.weight'])])
            dim2_params_keys = set([pn for pn in param_dict.keys() if param_dict[pn].dim() >= 2])
            dim2_params_keys = dim2_params_keys.difference(mup_params_keys)
            mup_parameters = [p for n, p in param_dict.items() if n in mup_params_keys]
            decay_params = [p for n, p in param_dict.items() if n in dim2_params_keys]
            nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
            optim_groups = [{'params': mup_parameters, 'weight_decay': weight_decay * self.config.mup_width_mult, 'lr': learning_rate / self.config.mup_width_mult}, {'params': decay_params, 'weight_decay': weight_decay, 'lr': learning_rate}, {'params': nodecay_params, 'weight_decay': 0.0, 'lr': learning_rate}]
        else:
            decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
            nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
            optim_groups = [{'params': decay_params, 'weight_decay': weight_decay, 'lr': learning_rate}, {'params': nodecay_params, 'weight_decay': 0.0, 'lr': learning_rate}]
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        optimizer = torch.optim.AdamW(optim_groups, betas=betas, fused=use_fused)
        return optimizer


class Mamba2Block(nn.Module):

    def __init__(self, config: 'Mamba2Config'):
        super().__init__()
        factory_kwargs = {'device': config.device, 'dtype': config.dtype}
        self.config = config
        d_in_proj = 2 * self.config.d_inner + 2 * self.config.n_groups * self.config.d_state + self.config.n_heads
        self.in_proj = nn.Linear(self.config.d_model, d_in_proj, bias=self.config.bias, **factory_kwargs)
        conv_dim = self.config.d_inner + 2 * self.config.n_groups * self.config.d_state
        self.conv1d = nn.Conv1d(in_channels=conv_dim, out_channels=conv_dim, bias=self.config.conv_bias, kernel_size=self.config.d_conv, groups=conv_dim, padding=self.config.d_conv - 1, **factory_kwargs)
        if self.config.conv_init is not None:
            nn.init.uniform_(self.conv1d.weight, -self.config.conv_init, self.config.conv_init)
        if self.config.learnable_init_states:
            self.init_states = nn.Parameter(torch.zeros(self.config.n_heads, self.config.d_head, self.config.d_state, **factory_kwargs))
            self.init_states._no_weight_decay = True
        self.act = nn.SiLU()
        dt = torch.exp(torch.rand(self.config.n_heads, **factory_kwargs) * (math.log(self.config.dt_max) - math.log(self.config.dt_min)) + math.log(self.config.dt_min))
        dt = torch.clamp(dt, min=self.config.dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        self.dt_bias._no_weight_decay = True
        assert self.config.A_init_range[0] > 0 and self.config.A_init_range[1] >= self.config.A_init_range[0]
        A = torch.empty(self.config.n_heads, dtype=torch.float32, device=self.config.device).uniform_(*self.config.A_init_range)
        A_log = torch.log(A)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True
        self.D = nn.Parameter(torch.ones(self.config.n_heads, device=self.config.device))
        self.D._no_weight_decay = True
        assert RMSNormGated is not None
        self.norm = RMSNormGated(self.config.d_inner, eps=1e-05, norm_before_gate=False, **factory_kwargs)
        self.out_proj = nn.Linear(self.config.d_inner, self.config.d_model, bias=self.config.bias, **factory_kwargs)

    def forward(self, u, cache=None, seq_idx=None):
        """
        u: (B, L, D)
        Returns: out : same shape as u
        """
        batch, length, _ = u.shape
        return_cache = False
        if cache is not None and length > 1:
            cache = None
            return_cache = True
        if cache is not None:
            out, cache = self.step(u, cache)
            return out, cache
        zxbcdt = self.in_proj(u)
        A = -torch.exp(self.A_log)
        initial_states = repeat(self.init_states, '... -> b ...', b=batch) if self.config.learnable_init_states else None
        dt_limit_kwargs = {} if self.config.dt_limit == (0.0, float('inf')) else dict(dt_limit=self.config.dt_limit)
        if self.config.use_mem_eff_path:
            out = mamba_split_conv1d_scan_combined(zxbcdt, rearrange(self.conv1d.weight, 'd 1 w -> d w'), self.conv1d.bias, self.dt_bias, A, D=self.D, chunk_size=self.config.chunk_size, seq_idx=seq_idx, activation=self.config.activation, rmsnorm_weight=self.norm.weight, rmsnorm_eps=self.norm.eps, outproj_weight=self.out_proj.weight, outproj_bias=self.out_proj.bias, headdim=self.config.d_head, ngroups=self.config.n_groups, norm_before_gate=False, initial_states=initial_states, return_final_states=return_cache, **dt_limit_kwargs)
            if return_cache:
                out, h_cache = out
                _, xBC, _ = torch.split(zxbcdt, [self.config.d_inner, self.config.d_inner + 2 * self.config.n_groups * self.config.d_state, self.config.n_heads], dim=-1)
                conv_cache = xBC[:, -self.config.d_conv:].transpose(1, 2)
                cache = h_cache, conv_cache
        else:
            z, xBC, dt = torch.split(zxbcdt, [self.config.d_inner, self.config.d_inner + 2 * self.config.n_groups * self.config.d_state, self.config.n_heads], dim=-1)
            dt = F.softplus(dt + self.dt_bias)
            assert self.config.activation in ['silu', 'swish']
            if causal_conv1d_fn is None or self.config.activation not in ['silu', 'swish']:
                xBC = self.act(self.conv1d(xBC.transpose(1, 2)).transpose(1, 2))
            else:
                xBC = causal_conv1d_fn(x=xBC.transpose(1, 2), weight=rearrange(self.conv1d.weight, 'd 1 w -> d w'), bias=self.conv1d.bias, activation=self.config.activation).transpose(1, 2)
            x, B, C = torch.split(xBC, [self.config.d_inner, self.config.n_groups * self.config.d_state, self.config.n_groups * self.config.d_state], dim=-1)
            y = mamba_chunk_scan_combined(rearrange(x, 'b l (h p) -> b l h p', p=self.config.d_head), dt, A, rearrange(B, 'b l (g n) -> b l g n', g=self.config.n_groups), rearrange(C, 'b l (g n) -> b l g n', g=self.config.n_groups), chunk_size=self.config.chunk_size, D=self.D, z=None, seq_idx=seq_idx, initial_states=initial_states, **dt_limit_kwargs)
            y = rearrange(y, 'b l h p -> b l (h p)')
            y = self.norm(y, z)
            out = self.out_proj(y)
        return out, cache

    def step(self, u, cache):
        """
        u: (B, 1, D)
        cache: (h_cache, conv_cache)
        """
        h_cache, conv_cache = cache
        zxbcdt = self.in_proj(u.squeeze(1))
        d_mlp = (zxbcdt.shape[-1] - 2 * self.config.d_inner - 2 * self.config.n_groups * self.config.d_state - self.config.n_heads) // 2
        z0, x0, z, xBC, dt = torch.split(zxbcdt, [d_mlp, d_mlp, self.config.d_inner, self.config.d_inner + 2 * self.config.n_groups * self.config.d_state, self.config.n_heads], dim=-1)
        if causal_conv1d_update is None:
            conv_cache.copy_(torch.roll(conv_cache, shifts=-1, dims=-1))
            conv_cache[:, :, -1] = xBC
            xBC = torch.sum(conv_cache * rearrange(self.conv1d.weight, 'd 1 w -> d w'), dim=-1)
            if self.conv1d.bias is not None:
                xBC = xBC + self.conv1d.bias
            xBC = self.act(xBC)
        else:
            xBC = causal_conv1d_update(xBC, conv_cache, rearrange(self.conv1d.weight, 'd 1 w -> d w'), self.conv1d.bias, self.config.activation)
        x, B, C = torch.split(xBC, [self.config.d_inner, self.config.n_groups * self.config.d_state, self.config.n_groups * self.config.d_state], dim=-1)
        A = -torch.exp(self.A_log.float())
        if selective_state_update is None:
            assert self.config.n_groups == 1, 'Only support ngroups=1 for this inference code path'
            dt = F.softplus(dt + self.dt_bias)
            dA = torch.exp(dt * A)
            x = rearrange(x, 'b (h p) -> b h p', p=self.config.d_head)
            dBx = torch.einsum('bh,bn,bhp->bhpn', dt, B, x)
            h_cache.copy_(h_cache * rearrange(dA, 'b h -> b h 1 1') + dBx)
            y = torch.einsum('bhpn,bn->bhp', h_cache, C)
            y = y + rearrange(self.D, 'h -> h 1') * x
            y = rearrange(y, 'b h p -> b (h p)')
        else:
            A = repeat(A, 'h -> h p n', p=self.config.d_head, n=self.config.d_state)
            dt = repeat(dt, 'b h -> b h p', p=self.config.d_head)
            dt_bias = repeat(self.dt_bias, 'h -> h p', p=self.config.d_head)
            D = repeat(self.D, 'h -> h p', p=self.config.d_head)
            B = rearrange(B, 'b (g n) -> b g n', g=self.config.n_groups)
            C = rearrange(C, 'b (g n) -> b g n', g=self.config.n_groups)
            x_reshaped = rearrange(x, 'b (h p) -> b h p', p=self.config.d_head)
            y = selective_state_update(h_cache, x_reshaped, dt, A, B, C, D, z=None, dt_bias=dt_bias, dt_softplus=True)
            y = rearrange(y, 'b h p -> b (h p)')
        y = self.norm(y, z)
        if d_mlp > 0:
            y = torch.cat([F.silu(z0) * x0, y], dim=-1)
        out = self.out_proj(y)
        return out.unsqueeze(1), (h_cache, conv_cache)


def load_config_hf(model_name):
    resolved_archive_file = cached_file(model_name, CONFIG_NAME, _raise_exceptions_for_missing_entries=False)
    return json.load(open(resolved_archive_file))


def load_state_dict_hf(model_name):
    resolved_archive_file = cached_file(model_name, WEIGHTS_NAME, _raise_exceptions_for_missing_entries=False)
    return torch.load(resolved_archive_file, weights_only=True, map_location='cpu', mmap=True)


def torch_to_mlx_depthwise_weights(torch_weights):
    """
    
    A convolution is said to be "depthwise" when channel i of the output is only computed by passing the filter overing channel i of the input.
    In torch, this is done by setting groups=number of channels.
    Because it is not yet implemented in MLX, a workaround is to zero out the weights of a conv object initialized with groups=1 (groups=1 is when output channel i is computing by passing the filter over all input channels)
    To do that, we need to zero out all elements except those on the "diagonal":
    for channels=8 and kernel_size=4, the weights are (8, 4, 8).
    these are composed of 8 x (8, 4, 1) filter, each of those is used to compute one output channel.
    this (8, 4, 1) filter is composed of 8 x (1, 4, 1) filter, each of those is passed over each input channel.
    so we need to set to 0 all those 8 filters, except the one which corresponds to the output channel of these 8 filters (so that the channels don't mix)

    """
    torch_weights = torch_weights.transpose(2, 1)
    channels, kernel_size, _ = torch_weights.shape
    mlx_weights = torch.zeros(channels, kernel_size, channels)
    indices = torch.arange(channels)
    if torch_weights[:, :, 0].type() == 'torch.BFloat16Tensor':
        mlx_weights[indices, :, indices] = torch_weights[:, :, 0].float()
    else:
        mlx_weights[indices, :, indices] = torch_weights[:, :, 0]
    return mlx_weights


def map_mambapy_torch_to_mlx(torch_state_dict):
    new_state_dict = {}
    for key, value in torch_state_dict.items():
        if 'conv1d.weight' in key:
            value = torch_to_mlx_depthwise_weights(value)
        if 'conv1d' in key:
            key = key.replace('conv1d', 'conv1d.conv1d')
        if value.type() == 'torch.BFloat16Tensor':
            new_state_dict[key] = value.half().numpy()
        else:
            new_state_dict[key] = value.numpy()
    return new_state_dict


def map_mambassm_torch_to_mlx(torch_state_dict):
    new_state_dict = {}
    for key in torch_state_dict:
        if key == 'backbone.embedding.weight' or key == 'backbone.norm_f.weight':
            new_key = key.replace('backbone.', '')
        else:
            new_key = key.replace('backbone', 'mamba')
        new_state_dict[new_key] = torch_state_dict[key]
    return map_mambapy_torch_to_mlx(new_state_dict)


def topk(x, k):
    """
    Returns the top k biggest values of x along the 2nd dim.

    Args:
        x : (B, vocab_size). can be probs or logits

    Returns:
        values : (B, k). ordered from lowest to biggest val
    """
    return mx.sort(x)[:, -k:]


class MambaLM(nn.Module):

    def __init__(self, lm_config: 'MambaLMConfig'):
        super().__init__()
        self.lm_config = lm_config
        self.config = lm_config.to_mamba_config()
        self.embedding = nn.Embedding(self.lm_config.vocab_size, self.config.d_model)
        self.mamba = Mamba(self.config)
        self.norm_f = nn.RMSNorm(self.config.d_model)
        self.lm_head = nn.Linear(self.config.d_model, self.lm_config.vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight

    def __call__(self, tokens):
        x = self.embedding(tokens)
        x = self.mamba(x)
        x = self.norm_f(x)
        logits = self.lm_head(x)
        return logits

    def step(self, token, caches):
        x = self.embedding(token)
        x, caches = self.mamba.step(x, caches)
        x = self.norm_f(x)
        logits = self.lm_head(x)
        return logits, caches

    def generate(self, tokenizer, prompt: 'str', n_tokens_to_gen: 'int'=50, sample: 'bool'=True, temperature: 'float'=1.0, top_k: 'int'=None):
        self.eval()
        input_ids = mx.array(tokenizer(prompt, return_tensors='np').input_ids)
        caches = [(None, mx.zeros([1, self.config.d_conv - 1, self.config.d_inner])) for _ in range(self.config.n_layers)]
        for i in range(input_ids.shape[1] + n_tokens_to_gen - 1):
            next_token_logits, caches = self.step(input_ids[:, i], caches)
            if i + 1 >= input_ids.shape[1]:
                if top_k is not None:
                    values = topk(next_token_logits, k=top_k)
                    mask = next_token_logits < values[:, 0, None]
                    next_token_logits = mx.where(mask, -5000, next_token_logits)
                if sample and temperature > 0:
                    next_token = mx.random.categorical(next_token_logits * (1 / temperature), num_samples=1)
                else:
                    next_token = mx.argmax(next_token_logits, axis=-1)[:, None]
                input_ids = mx.concatenate([input_ids, next_token], axis=1)
        output = [tokenizer.decode(output.tolist()) for output in input_ids][0]
        self.train()
        return output

    @staticmethod
    def from_pretrained(name: 'str'):
        """
        Returns a model loaded with pretrained weights pulled from HuggingFace.

        Args:
            name: As of now, supports
                * 'state-spaces/mamba-2.8b-slimpj'
                * 'state-spaces/mamba-2.8b'
                * 'state-spaces/mamba-1.4b'
                * 'state-spaces/mamba-790m'
                * 'state-spaces/mamba-370m'
                * 'state-spaces/mamba-130m'

        Returns:
            model: a Mamba model configured with the proper parameters and initialized with the proper weights
        """
        import numpy as np
        config_data = load_config_hf(name)
        config = MambaLMConfig(d_model=config_data['d_model'], n_layers=config_data['n_layer'], vocab_size=config_data['vocab_size'])
        model = MambaLM(config)
        filename = name.split('/')[-1] + '.mlx.npz'
        if not os.path.exists(filename):
            state_dict = load_state_dict_hf(name)
            mlx_state_dict = map_mambassm_torch_to_mlx(state_dict)
            np.savez(filename, **mlx_state_dict)
        model.update(tree_unflatten(list(mx.load(filename).items())))
        return model


class VMamba(nn.Module):

    def __init__(self, config: 'MambaConfig'):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([ResidualBlock(config) for _ in range(config.n_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def step(self, x, caches):
        for i, layer in enumerate(self.layers):
            x, caches[i] = layer.step(x, caches[i])
        return x, caches


class VMambaBlock(nn.Module):

    def __init__(self, config: 'MambaConfig'):
        super().__init__()
        self.config = config
        self.in_proj = nn.Linear(config.d_model, 2 * config.d_inner, bias=config.bias)
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
        self.A_log._no_weight_decay = True
        self.D = nn.Parameter(torch.ones(config.d_inner))
        self.D._no_weight_decay = True
        if config.bidirectional:
            A_b = torch.arange(1, config.d_state + 1, dtype=torch.float32).repeat(config.d_inner, 1)
            self.A_log_b = nn.Parameter(torch.log(A_b))
            self.A_log_b._no_weight_decay = True
            self.conv1d_b = nn.Conv1d(in_channels=config.d_inner, out_channels=config.d_inner, kernel_size=config.d_conv, bias=config.conv_bias, groups=config.d_inner, padding=config.d_conv - 1)
            self.x_proj_b = nn.Linear(config.d_inner, config.dt_rank + 2 * config.d_state, bias=False)
            self.dt_proj_b = nn.Linear(config.dt_rank, config.d_inner, bias=True)
            self.D_b = nn.Parameter(torch.ones(config.d_inner))
            self.D_b._no_weight_decay = True
        self.out_proj = nn.Linear(config.d_inner, config.d_model, bias=config.bias)
        if self.config.inner_layernorms:
            self.dt_layernorm = RMSNorm(self.config.dt_rank, config.rms_norm_eps)
            self.B_layernorm = RMSNorm(self.config.d_state, config.rms_norm_eps)
            self.C_layernorm = RMSNorm(self.config.d_state, config.rms_norm_eps)
        else:
            self.dt_layernorm = None
            self.B_layernorm = None
            self.C_layernorm = None
        if self.config.use_cuda:
            try:
                self.selective_scan_cuda = selective_scan_fn
            except ImportError:
                None
                self.config.use_cuda = False

    def _apply_layernorms(self, dt, B, C):
        if self.dt_layernorm is not None:
            dt = self.dt_layernorm(dt)
        if self.B_layernorm is not None:
            B = self.B_layernorm(B)
        if self.C_layernorm is not None:
            C = self.C_layernorm(C)
        return dt, B, C

    def forward(self, x):
        _, L, _ = x.shape
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        x = x.transpose(1, 2)
        x = self.conv1d(x)[:, :, :L]
        x = x.transpose(1, 2)
        x = F.silu(x)
        y = self.ssm(x=x, z=z, A_log=self.A_log, D=self.D, x_proj=self.x_proj, dt_proj=self.dt_proj)
        if self.config.bidirectional:
            xz_b = xz.flip([1])
            x_b, z_b = xz_b.chunk(2, dim=-1)
            x_b = x_b.transpose(1, 2)
            x_b = self.conv1d_b(x_b)[:, :, :L]
            x_b = x_b.transpose(1, 2)
            x_b = F.silu(x_b)
            y_b = self.ssm(x=x_b, z=z_b, A_log=self.A_log_b, D=self.D_b, x_proj=self.x_proj_b, dt_proj=self.dt_proj_b)
        if self.config.use_cuda:
            if not self.config.bidirectional:
                return self.out_proj(y)
            elif self.config.divide_output:
                return self.out_proj((y + y_b.flip([1])) / 2)
            else:
                return self.out_proj(y + y_b.flip([1]))
        z = F.silu(z)
        y = y * z
        if not self.config.bidirectional:
            return self.out_proj(y)
        else:
            z_b = F.silu(z_b)
            y_b = y_b * z_b
            if self.config.divide_output:
                return self.out_proj((y + y_b.flip([1])) / 2)
            else:
                return self.out_proj(y + y_b.flip([1]))

    def ssm(self, x, z, A_log, D, x_proj, dt_proj):
        A = -torch.exp(A_log.float())
        D = D.float()
        deltaBC = x_proj(x)
        delta, B, C = torch.split(deltaBC, [self.config.dt_rank, self.config.d_state, self.config.d_state], dim=-1)
        delta, B, C = self._apply_layernorms(delta, B, C)
        delta = dt_proj.weight @ delta.transpose(1, 2)
        if self.config.use_cuda:
            x = x.transpose(1, 2)
            B = B.transpose(1, 2)
            C = C.transpose(1, 2)
            z = z.transpose(1, 2)
            y = self.selective_scan_cuda(x, delta, A, B, C, D, z=z, delta_softplus=True, delta_bias=dt_proj.bias.float())
            y = y.transpose(1, 2)
        else:
            delta = delta.transpose(1, 2)
            delta = F.softplus(delta + dt_proj.bias)
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
        delta, B, C = self._apply_layernorms(delta, B, C)
        delta = F.softplus(self.dt_proj(delta))
        deltaA = torch.exp(delta.unsqueeze(-1) * A)
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(1)
        BX = deltaB * x.unsqueeze(-1)
        if h is None:
            h = torch.zeros(x.size(0), self.config.d_inner, self.config.d_state, device=deltaA.device)
        h = deltaA * h + BX
        y = (h @ C.unsqueeze(-1)).squeeze(2)
        y = y + D * x
        return y, h


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (AttentionLayer,
     lambda: ([], {'config': SimpleNamespace(d_model=4, num_attention_heads=4, num_key_value_heads=4, attention_dropout=0.5, num_experts_per_tok=4, mlp_size=4, rms_norm_eps=4), 'num_experts': 4}),
     lambda: ([torch.rand([4, 4, 4])], {})),
    (AttentionSDPA,
     lambda: ([], {'config': SimpleNamespace(d_model=4, num_attention_heads=4, num_key_value_heads=4, attention_dropout=0.5)}),
     lambda: ([torch.rand([4, 4, 4])], {})),
    (Jamba,
     lambda: ([], {'config': SimpleNamespace(n_layers=1, attn_layer_offset=1, attn_layer_period=1, expert_layer_offset=1, expert_layer_period=1, num_experts=4, d_model=4, num_attention_heads=4, num_key_value_heads=4, attention_dropout=0.5, num_experts_per_tok=4, mlp_size=4, rms_norm_eps=4)}),
     lambda: ([torch.rand([4, 4, 4])], {})),
    (MLP,
     lambda: ([], {'config': SimpleNamespace(d_model=4, mlp_size=4)}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (RMSNorm,
     lambda: ([], {'d_model': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (SparseMoEBlock,
     lambda: ([], {'config': SimpleNamespace(d_model=4, mlp_size=4), 'num_experts': 4, 'num_experts_per_tok': 4}),
     lambda: ([torch.rand([4, 4, 4])], {})),
]

