
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


from collections import defaultdict


from typing import Any


from typing import Dict


from typing import Optional


from typing import Tuple


from typing import Union


import torch


import torch.nn.functional as F


import torch.utils.checkpoint


from torch import nn


from torch.nn import CrossEntropyLoss


from torch.utils._pytree import tree_map


class RMSNorm(nn.Module):

    def __init__(self, hidden_size, eps=1e-06):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states


class SwiGluMLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        if self.config.pretraining_tp > 1:
            slice = self.intermediate_size // self.config.pretraining_tp
            gate_proj_slices = self.gate_proj.weight.split(slice, dim=0)
            up_proj_slices = self.up_proj.weight.split(slice, dim=0)
            down_proj_slices = self.down_proj.weight.split(slice, dim=1)
            gate_proj = torch.cat([F.linear(x, gate_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1)
            up_proj = torch.cat([F.linear(x, up_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1)
            intermediate_states = (self.act_fn(gate_proj) * up_proj).split(slice, dim=2)
            down_proj = [F.linear(intermediate_states[i], down_proj_slices[i]) for i in range(self.config.pretraining_tp)]
            down_proj = sum(down_proj)
        else:
            down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


class RotaryEmbedding(nn.Module):

    def __init__(self, dim, max_position_embeddings=16, base=10000, device=None, scaling_factor=1.0):
        super().__init__()
        self.scaling_factor = scaling_factor
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float() / self.dim)
        self.register_buffer('inv_freq', inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != 'mps' else 'cpu'
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos, sin


class Conv(nn.Module):

    def __init__(self, config, layer_idx):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.conv = nn.Conv1d(config.hidden_size, config.hidden_size, bias=True, kernel_size=config.conv_kernel, groups=config.hidden_size, padding=config.conv_kernel - 1)

    def __call__(self, hidden_states, cache_params=None):
        seq_len = hidden_states.shape[1]
        hidden_states = self.norm(hidden_states)
        hidden_states = hidden_states.transpose(1, 2)
        if causal_conv1d_fn is None:
            if cache_params is not None:
                if cache_params.seqlen_offset > 0:
                    conv_state = cache_params.conv_states_dic['pre_conv'][self.layer_idx]
                    conv_state = torch.roll(conv_state, shifts=-1, dims=-1)
                    conv_state[:, :, -1] = hidden_states[:, :, 0]
                    cache_params.conv_states_dic['pre_conv'][self.layer_idx].copy_(conv_state)
                    hidden_states = torch.sum(conv_state * self.conv.weight[:, 0, :], dim=-1)
                    hidden_states += self.conv.bias
                    hidden_states = hidden_states.unsqueeze(-1)
                else:
                    conv_state = nn.functional.pad(hidden_states, (self.config.conv_kernel - hidden_states.shape[-1], 0))
                    cache_params.conv_states_dic['pre_conv'][self.layer_idx].copy_(conv_state)
                    hidden_states = self.conv(hidden_states)[..., :seq_len]
            else:
                hidden_states = self.conv(hidden_states)[..., :seq_len]
        else:
            conv_weights = self.conv.weight.view(self.conv.weight.size(0), self.conv.weight.size(2))
            if cache_params is not None and cache_params.seqlen_offset > 0:
                hidden_states = causal_conv1d_update(hidden_states.squeeze(-1), cache_params.conv_states_dic['pre_conv'][self.layer_idx], conv_weights, self.conv.bias, None)
                hidden_states = hidden_states.unsqueeze(-1)
            else:
                if cache_params is not None:
                    conv_states = nn.functional.pad(hidden_states, (self.config.conv_kernel - hidden_states.shape[-1], 0))
                    cache_params.conv_states_dic['pre_conv'][self.layer_idx].copy_(conv_states)
                hidden_states = causal_conv1d_fn(hidden_states, conv_weights, self.conv.bias, activation=None)
        hidden_states = hidden_states.transpose(1, 2)
        return hidden_states


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = q * cos + rotate_half(q) * sin
    k_embed = k * cos + rotate_half(k) * sin
    return q_embed, k_embed


def permute_qk(q, k):
    bsz, num_head, seq_len, head_dim = q.shape
    q = q.reshape(bsz, num_head, seq_len, head_dim // 2, 2).transpose(3, 4).reshape(bsz, num_head, seq_len, head_dim)
    k = k.reshape(bsz, num_head, seq_len, head_dim // 2, 2).transpose(3, 4).reshape(bsz, num_head, seq_len, head_dim)
    return q, k


def undo_permute_qk(q, k):
    bsz, num_head, seq_len, head_dim = q.shape
    q = q.reshape(bsz, num_head, seq_len, 2, head_dim // 2).transpose(3, 4).reshape(bsz, num_head, seq_len, head_dim)
    k = k.reshape(bsz, num_head, seq_len, 2, head_dim // 2).transpose(3, 4).reshape(bsz, num_head, seq_len, head_dim)
    return q, k


class TTTBase(nn.Module):

    def __init__(self, config: 'TTTConfig', layer_idx: 'Optional[int]'=None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(f'Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` when creating this class.')
        self.width = config.hidden_size
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.width // self.num_heads
        self.mini_batch_size = config.mini_batch_size
        token_idx = 1.0 / torch.arange(1, self.mini_batch_size + 1)
        self.register_buffer('token_idx', token_idx, persistent=False)
        self.learnable_token_idx = nn.Parameter(torch.zeros((self.mini_batch_size,)))
        self.share_qk = config.share_qk
        self.conv_kernel = config.conv_kernel
        self._init_qkvo_proj()
        self._init_rope()
        self._init_ttt_lr_gate()
        self._init_ttt_ln()
        self.use_gate = config.use_gate
        if self.use_gate:
            self.g_proj = nn.Linear(self.width, self.width, bias=False)
        self.post_norm = nn.LayerNorm(self.width, eps=1e-06)

    def _init_qkvo_proj(self):
        self.q_proj = nn.Linear(self.width, self.num_heads * self.head_dim, bias=False)
        if not self.share_qk:
            self.k_proj = nn.Linear(self.width, self.num_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.width, self.num_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.width, self.num_heads * self.head_dim, bias=False)
        if self.share_qk:
            self.conv_q = nn.Conv1d(self.hidden_size, self.hidden_size, bias=True, kernel_size=self.conv_kernel, groups=self.hidden_size, padding=self.conv_kernel - 1)
            self.conv_k = nn.Conv1d(self.hidden_size, self.hidden_size, bias=True, kernel_size=self.conv_kernel, groups=self.hidden_size, padding=self.conv_kernel - 1)

    def _init_rope(self):
        self.rope_theta = self.config.rope_theta
        self.rotary_emb = RotaryEmbedding(self.head_dim, max_position_embeddings=self.mini_batch_size, base=self.rope_theta)

    def _init_ttt_lr_gate(self):
        linear_weight_data = nn.Linear(self.width, 1, bias=True).weight.data
        self.learnable_ttt_lr_weight = nn.Parameter(torch.stack([torch.normal(0, 0.02, size=linear_weight_data.shape) for _ in range(self.num_heads)], dim=0))
        linear_bias_data = nn.Linear(self.width, 1, bias=True).bias.data
        self.learnable_ttt_lr_bias = nn.Parameter(torch.stack([torch.zeros_like(linear_bias_data) for _ in range(self.num_heads)], dim=0))

    def _init_ttt_ln(self):
        ln_weight_data = nn.LayerNorm(self.head_dim).weight.data
        self.ttt_norm_weight = nn.Parameter(torch.tile(ln_weight_data.unsqueeze(0), (self.num_heads, 1)))
        ln_bias_data = nn.LayerNorm(self.head_dim).bias.data
        self.ttt_norm_bias = nn.Parameter(torch.tile(ln_bias_data.unsqueeze(0), (self.num_heads, 1)))

    def get_qkv_projections(self, hidden_states, cache_params: 'Optional[TTTCache]'=None):
        if self.share_qk:
            xq, XV = self.q_proj(hidden_states), self.v_proj(hidden_states)
            seq_len = xq.shape[1]
            xq = xq.transpose(1, 2)
            if causal_conv1d_fn is None:
                if cache_params is not None:
                    if cache_params.seqlen_offset > 0:
                        conv_q_state = cache_params.conv_states_dic['ttt_conv_q'][self.layer_idx]
                        conv_q_state = torch.roll(conv_q_state, shifts=-1, dims=-1)
                        conv_q_state[:, :, -1] = xq[:, :, 0]
                        cache_params.conv_states_dic['ttt_conv_q'][self.layer_idx].copy_(conv_q_state)
                        XQ = torch.sum(conv_q_state * self.conv_q.weight[:, 0, :], dim=-1)
                        XQ += self.conv_q.bias
                        XQ = XQ.unsqueeze(-1)
                        conv_k_state = cache_params.conv_states_dic['ttt_conv_k'][self.layer_idx]
                        conv_k_state = torch.roll(conv_k_state, shifts=-1, dims=-1)
                        conv_k_state[:, :, -1] = xq[:, :, 0]
                        cache_params.conv_states_dic['ttt_conv_k'][self.layer_idx].copy_(conv_k_state)
                        XK = torch.sum(conv_k_state * self.conv_k.weight[:, 0, :], dim=-1)
                        XK += self.conv_k.bias
                        XK = XK.unsqueeze(-1)
                    else:
                        conv_q_state = nn.functional.pad(xq, (self.config.conv_kernel - xq.shape[-1], 0))
                        cache_params.conv_states_dic['ttt_conv_q'][self.layer_idx].copy_(conv_q_state)
                        XQ = self.conv_q(xq)[..., :seq_len]
                        conv_k_state = nn.functional.pad(xq, (self.config.conv_kernel - xq.shape[-1], 0))
                        cache_params.conv_states_dic['ttt_conv_k'][self.layer_idx].copy_(conv_k_state)
                        XK = self.conv_k(xq)[..., :seq_len]
                else:
                    XQ = self.conv_q(xq)[..., :seq_len]
                    XK = self.conv_k(xq)[..., :seq_len]
            else:
                conv_q_weights = self.conv_q.weight.view(self.conv_q.weight.size(0), self.conv_q.weight.size(2))
                conv_k_weights = self.conv_k.weight.view(self.conv_k.weight.size(0), self.conv_k.weight.size(2))
                if cache_params is not None and cache_params.seqlen_offset > 0:
                    XQ = causal_conv1d_update(xq.squeeze(-1), cache_params.conv_states_dic['ttt_conv_q'][self.layer_idx], conv_q_weights, self.conv_q.bias, None)
                    XQ = XQ.unsqueeze(-1)
                    XK = causal_conv1d_update(xq.squeeze(-1), cache_params.conv_states_dic['ttt_conv_k'][self.layer_idx], conv_k_weights, self.conv_k.bias, None)
                    XK = XK.unsqueeze(-1)
                else:
                    if cache_params is not None:
                        conv_q_states = nn.functional.pad(xq, (self.config.conv_kernel - xq.shape[-1], 0))
                        cache_params.conv_states_dic['ttt_conv_q'][self.layer_idx].copy_(conv_q_states)
                        conv_k_states = nn.functional.pad(xq, (self.config.conv_kernel - xq.shape[-1], 0))
                        cache_params.conv_states_dic['ttt_conv_k'][self.layer_idx].copy_(conv_k_states)
                    XQ = causal_conv1d_fn(xq, conv_q_weights, self.conv_q.bias, activation=None)
                    XK = causal_conv1d_fn(xq, conv_k_weights, self.conv_k.bias, activation=None)
            XQ = XQ.transpose(1, 2)
            XK = XK.transpose(1, 2)
        else:
            XQ, XK, XV = self.q_proj(hidden_states), self.k_proj(hidden_states), self.v_proj(hidden_states)
        return XQ, XK, XV

    def _split_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[:2] + (self.num_heads, self.head_dim))

    def get_eta(self, X, mini_batch_step_offset, mini_batch_size):
        ttt_lr = torch.einsum('bnkc,hdc->bhnkd', X, self.learnable_ttt_lr_weight) + self.learnable_ttt_lr_bias.reshape(1, -1, 1, 1, 1)
        ttt_lr = F.sigmoid(ttt_lr)
        ttt_lr = ttt_lr.permute(0, 1, 2, 4, 3)
        ttt_lr_eta = self.config.ttt_base_lr * ttt_lr / self.head_dim
        token_idx = self.token_idx + self.learnable_token_idx
        token_idx = token_idx[mini_batch_step_offset:mini_batch_step_offset + mini_batch_size]
        token_idx = torch.clamp_min(token_idx, 0.0)
        token_eta = torch.broadcast_to(token_idx.reshape(1, 1, 1, mini_batch_size, 1), (X.shape[0], self.num_heads, X.shape[1], mini_batch_size, 1))
        return token_eta, ttt_lr_eta

    def apply_gate(self, hidden_states, ttt_output):
        y = self.g_proj(hidden_states)
        y = F.gelu(y, approximate='tanh')
        output = y * ttt_output
        return output

    def get_ttt_inputs(self, inputs, mini_batch_size, cache_params):
        XQ = inputs['XQ']
        XK = inputs['XK']
        XV = inputs['XV']
        X = inputs['X']
        B, L, C = X.shape
        num_mini_batch = L // mini_batch_size
        X = X.reshape(B, num_mini_batch, mini_batch_size, self.width)
        XQ = XQ.reshape(B, self.num_heads, L // mini_batch_size, mini_batch_size, self.head_dim)
        XK = XK.reshape(B, self.num_heads, L // mini_batch_size, mini_batch_size, self.head_dim)
        XV = XV.reshape(B, self.num_heads, L // mini_batch_size, mini_batch_size, self.head_dim)
        if cache_params is not None:
            mini_batch_step_offset = cache_params.seqlen_offset % self.mini_batch_size
        else:
            mini_batch_step_offset = 0
        token_eta, ttt_lr_eta = self.get_eta(X, mini_batch_step_offset, mini_batch_size)
        eta = token_eta * ttt_lr_eta
        inputs = {'XQ': XQ, 'XK': XK, 'XV': XV, 'eta': eta, 'token_eta': token_eta, 'ttt_lr_eta': ttt_lr_eta}
        return inputs

    def ttt(self, inputs, mini_batch_size, last_mini_batch_params_dict, cache_params: 'Optional[TTTCache]'=None):
        raise NotImplementedError('ttt method must be implemented in TTTBase subclasses.')

    def forward(self, hidden_states: 'torch.Tensor', attention_mask: 'Optional[torch.Tensor]'=None, position_ids: 'Optional[torch.LongTensor]'=None, cache_params: 'Optional[TTTCache]'=None):
        B, L = hidden_states.shape[:2]
        reminder_len = L % self.mini_batch_size
        num_mini_batch = L // self.mini_batch_size
        last_mini_batch_params_dict = None
        XQ, XK, XV = self.get_qkv_projections(hidden_states, cache_params=cache_params)
        XQ = XQ.reshape(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        XK = XK.reshape(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        XV = XV.reshape(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        cos, sin = self.rotary_emb(XV, position_ids % self.mini_batch_size)
        XQ, XK = permute_qk(XQ, XK)
        XQ, XK = apply_rotary_pos_emb(XQ, XK, cos, sin)
        XQ, XK = undo_permute_qk(XQ, XK)
        output_hidden_states = []
        if num_mini_batch > 0:
            inputs = {'XQ': XQ[:, :, :num_mini_batch * self.mini_batch_size], 'XK': XK[:, :, :num_mini_batch * self.mini_batch_size], 'XV': XV[:, :, :num_mini_batch * self.mini_batch_size], 'X': hidden_states[:, :num_mini_batch * self.mini_batch_size]}
            output_mod, last_mini_batch_params_dict = self.ttt(self.get_ttt_inputs(inputs, self.mini_batch_size, cache_params), mini_batch_size=self.mini_batch_size, last_mini_batch_params_dict=last_mini_batch_params_dict, cache_params=cache_params)
            output_hidden_states.append(output_mod)
        if reminder_len > 0:
            inputs = {'XQ': XQ[:, :, -reminder_len:], 'XK': XK[:, :, -reminder_len:], 'XV': XV[:, :, -reminder_len:], 'X': hidden_states[:, -reminder_len:]}
            output_reminder, _ = self.ttt(self.get_ttt_inputs(inputs, reminder_len, cache_params), mini_batch_size=reminder_len, last_mini_batch_params_dict=last_mini_batch_params_dict, cache_params=cache_params)
            output_hidden_states.append(output_reminder)
        output_hidden_states = torch.cat(output_hidden_states, dim=1)
        output_hidden_states = self.post_norm(output_hidden_states)
        if self.use_gate:
            output_hidden_states = self.apply_gate(hidden_states, output_hidden_states)
        output_hidden_states = self.o_proj(output_hidden_states)
        return output_hidden_states


def ln_fused_l2_bwd(x, l2_target, gamma, beta, eps=1e-06):
    """Batch backward for LayerNorm fused with L2 loss."""
    D = x.shape[-1]
    mu = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    std = torch.sqrt(var + eps)
    x_hat = (x - mu) / std
    y = gamma * x_hat + beta
    grad_output = y - l2_target
    grad_x_hat = grad_output * gamma
    z = 1.0 / D * (D * grad_x_hat - grad_x_hat.sum(dim=-1, keepdim=True) - x_hat * (grad_x_hat * x_hat).sum(dim=-1, keepdim=True)) / std
    return z


def ln_fwd(x, gamma, beta, eps=1e-06):
    """Batch forward for LayerNorm."""
    mu = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    std = torch.sqrt(var + eps)
    x_hat = (x - mu) / std
    y = gamma * x_hat + beta
    return y


def scan(f, init, xs, out, checkpoint_group=0):
    """Minic jax.lax.scan function."""
    carry = init
    if isinstance(xs, dict):
        num_items = len(next(iter(xs.values())))
    else:
        num_items = len(xs[0])

    def scan_fn(carry, i_start, i_end):
        for i in range(i_start, i_end):
            if isinstance(xs, dict):
                x = {key: tensor[i] for key, tensor in xs.items()}
            else:
                x = [x[i] for x in xs]
            carry, y = f(carry, x)
            out[i] = y
        return carry
    if checkpoint_group > 0:
        ckpt_every_n = num_items // checkpoint_group
        for k in range(0, num_items, ckpt_every_n):
            carry = torch.utils.checkpoint.checkpoint(scan_fn, carry, k, min(k + ckpt_every_n, num_items), use_reentrant=False)
    else:
        carry = scan_fn(carry, 0, num_items)
    return carry, out


class TTTLinear(TTTBase):

    def __init__(self, config: 'TTTConfig', layer_idx: 'Optional[int]'=None):
        super().__init__(config, layer_idx)
        self.W1 = nn.Parameter(torch.normal(0, 0.02, size=(self.num_heads, self.head_dim, self.head_dim)))
        self.b1 = nn.Parameter(torch.zeros(self.num_heads, 1, self.head_dim))

    def ttt(self, inputs, mini_batch_size, last_mini_batch_params_dict, cache_params: 'Optional[TTTCache]'=None):
        if mini_batch_size is None:
            mini_batch_size = self.mini_batch_size
        if last_mini_batch_params_dict is None and cache_params is not None:
            last_mini_batch_params_dict = cache_params.ttt_params_to_dict(self.layer_idx)
        B = inputs['XV'].shape[0]
        num_mini_batch = inputs['XV'].shape[2]
        L = inputs['XV'].shape[2] * inputs['XV'].shape[3]
        device = inputs['XV'].device
        dtype = inputs['XV'].dtype
        use_dual_form = cache_params is None or mini_batch_size % self.mini_batch_size == 0

        def compute_mini_batch(params_dict, inputs):
            W1_init = params_dict['W1_states']
            b1_init = params_dict['b1_states']
            XQ_mini_batch = inputs['XQ']
            XV_mini_batch = inputs['XV']
            XK_mini_batch = inputs['XK']
            eta_mini_batch = inputs['eta']
            token_eta_mini_batch = inputs['token_eta']
            ttt_lr_eta_mini_batch = inputs['ttt_lr_eta']
            X1 = XK_mini_batch
            Z1 = X1 @ W1_init + b1_init
            reconstruction_target = XV_mini_batch - XK_mini_batch
            ln_weight = self.ttt_norm_weight.reshape(self.num_heads, 1, self.head_dim)
            ln_bias = self.ttt_norm_bias.reshape(self.num_heads, 1, self.head_dim)
            grad_l_wrt_Z1 = ln_fused_l2_bwd(Z1, reconstruction_target, ln_weight, ln_bias)
            if use_dual_form:
                Attn1 = torch.tril(XQ_mini_batch @ X1.transpose(-2, -1))
                b1_bar = b1_init - torch.tril(eta_mini_batch) @ grad_l_wrt_Z1
                Z1_bar = XQ_mini_batch @ W1_init - eta_mini_batch * Attn1 @ grad_l_wrt_Z1 + b1_bar
                last_eta_mini_batch = eta_mini_batch[:, :, -1, :, None]
                W1_last = W1_init - (last_eta_mini_batch * X1).transpose(-1, -2) @ grad_l_wrt_Z1
                b1_last = b1_init - torch.sum(last_eta_mini_batch * grad_l_wrt_Z1, dim=-2, keepdim=True)
                grad_W1_last = torch.zeros_like(W1_last)
                grad_b1_last = torch.zeros_like(b1_last)
            else:
                ttt_lr_eta_mini_batch = torch.broadcast_to(ttt_lr_eta_mini_batch, (*ttt_lr_eta_mini_batch.shape[:2], mini_batch_size, mini_batch_size))
                grad_W1 = torch.einsum('bhki,bhkj->bhkij', X1, grad_l_wrt_Z1)
                grad_W1 = torch.einsum('bhnk,bhkij->bhnij', torch.tril(ttt_lr_eta_mini_batch), grad_W1)
                grad_W1 = grad_W1 + params_dict['W1_grad'].unsqueeze(2)
                grad_b1 = torch.einsum('bhnk,bhki->bhni', torch.tril(ttt_lr_eta_mini_batch), grad_l_wrt_Z1)
                grad_b1 = grad_b1 + params_dict['b1_grad']
                W1_bar = W1_init.unsqueeze(2) - grad_W1 * token_eta_mini_batch.unsqueeze(-1)
                b1_bar = b1_init - grad_b1 * token_eta_mini_batch
                Z1_bar = (XQ_mini_batch.unsqueeze(3) @ W1_bar).squeeze(3) + b1_bar
                W1_last = W1_bar[:, :, -1]
                b1_last = b1_bar[:, :, -1:]
                grad_W1_last = grad_W1[:, :, -1]
                grad_b1_last = grad_b1[:, :, -1:]
            Z1_bar = ln_fwd(Z1_bar, ln_weight, ln_bias)
            XQW_mini_batch = XQ_mini_batch + Z1_bar
            last_param_dict = {'W1_states': W1_last, 'b1_states': b1_last, 'W1_grad': grad_W1_last, 'b1_grad': grad_b1_last}
            return last_param_dict, XQW_mini_batch
        if last_mini_batch_params_dict is not None:
            init_params_dict = last_mini_batch_params_dict
        else:
            init_params_dict = {'W1_states': torch.tile(self.W1.unsqueeze(0), dims=(B, 1, 1, 1)), 'b1_states': torch.tile(self.b1.unsqueeze(0), dims=(B, 1, 1, 1))}
            init_params_dict.update(W1_grad=torch.zeros_like(init_params_dict['W1_states']))
            init_params_dict.update(b1_grad=torch.zeros_like(init_params_dict['b1_states']))
        inputs = tree_map(lambda x: x.permute(2, 0, 1, 3, 4), inputs)
        XQW_batch = torch.empty((num_mini_batch, B, self.num_heads, mini_batch_size, self.head_dim), device=device, dtype=dtype)
        batch_params_dict, XQW_batch = scan(compute_mini_batch, init_params_dict, inputs, XQW_batch, self.config.scan_checkpoint_group_size if self.training else 0)
        if cache_params is not None:
            cache_params.update(batch_params_dict, self.layer_idx, L)
        XQW_batch = XQW_batch.permute(1, 0, 3, 2, 4)
        XQW_batch = XQW_batch.reshape(B, L, self.width)
        return XQW_batch, batch_params_dict


def gelu_bwd(x):
    tanh_out = torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x))
    ff = 0.5 * x * ((1 - tanh_out * tanh_out) * (0.79788456 + 0.1070322243 * x * x)) + 0.5 * (1 + tanh_out)
    return ff


class TTTMLP(TTTBase):

    def __init__(self, config: 'TTTConfig', layer_idx: 'Optional[int]'=None):
        super().__init__(config, layer_idx)
        self.W1 = nn.Parameter(torch.normal(0, 0.02, size=(self.num_heads, self.head_dim, 4 * self.head_dim)))
        self.b1 = nn.Parameter(torch.zeros(self.num_heads, 1, 4 * self.head_dim))
        self.W2 = nn.Parameter(torch.normal(0, 0.02, size=(self.num_heads, 4 * self.head_dim, self.head_dim)))
        self.b2 = nn.Parameter(torch.zeros(self.num_heads, 1, self.head_dim))

    def ttt(self, inputs, mini_batch_size, last_mini_batch_params_dict, cache_params: 'Optional[TTTCache]'=None):
        if mini_batch_size is None:
            mini_batch_size = self.mini_batch_size
        if last_mini_batch_params_dict is None and cache_params is not None:
            last_mini_batch_params_dict = cache_params.ttt_params_to_dict(self.layer_idx)
        B = inputs['XV'].shape[0]
        num_mini_batch = inputs['XV'].shape[2]
        L = inputs['XV'].shape[2] * inputs['XV'].shape[3]
        device = inputs['XV'].device
        dtype = inputs['XV'].dtype
        use_dual_form = cache_params is None or mini_batch_size % self.mini_batch_size == 0

        def compute_mini_batch(params_dict, inputs):
            W1_init = params_dict['W1_states']
            b1_init = params_dict['b1_states']
            W2_init = params_dict['W2_states']
            b2_init = params_dict['b2_states']
            XQ_mini_batch = inputs['XQ']
            XV_mini_batch = inputs['XV']
            XK_mini_batch = inputs['XK']
            eta_mini_batch = inputs['eta']
            token_eta_mini_batch = inputs['token_eta']
            ttt_lr_eta_mini_batch = inputs['ttt_lr_eta']
            X1 = XK_mini_batch
            Z1 = X1 @ W1_init + b1_init
            X2 = F.gelu(Z1, approximate='tanh')
            Z2 = X2 @ W2_init + b2_init
            reconstruction_target = XV_mini_batch - XK_mini_batch
            ln_weight = self.ttt_norm_weight.reshape(self.num_heads, 1, self.head_dim)
            ln_bias = self.ttt_norm_bias.reshape(self.num_heads, 1, self.head_dim)
            grad_l_wrt_Z2 = ln_fused_l2_bwd(Z2, reconstruction_target, ln_weight, ln_bias)
            grad_l_wrt_Z1 = grad_l_wrt_Z2 @ W2_init.transpose(-2, -1) * gelu_bwd(Z1)
            if use_dual_form:
                Attn1 = torch.tril(XQ_mini_batch @ X1.transpose(-2, -1))
                b1_bar = b1_init - torch.tril(eta_mini_batch) @ grad_l_wrt_Z1
                Z1_bar = XQ_mini_batch @ W1_init - eta_mini_batch * Attn1 @ grad_l_wrt_Z1 + b1_bar
                X2_bar = F.gelu(Z1_bar, approximate='tanh')
                Attn2 = torch.tril(X2_bar @ X2.transpose(-2, -1))
                b2_bar = b2_init - torch.tril(eta_mini_batch) @ grad_l_wrt_Z2
                Z2_bar = X2_bar @ W2_init - eta_mini_batch * Attn2 @ grad_l_wrt_Z2 + b2_bar
                last_eta_mini_batch = eta_mini_batch[:, :, -1, :, None]
                W1_last = W1_init - (last_eta_mini_batch * X1).transpose(-1, -2) @ grad_l_wrt_Z1
                b1_last = b1_init - torch.sum(last_eta_mini_batch * grad_l_wrt_Z1, dim=-2, keepdim=True)
                W2_last = W2_init - (last_eta_mini_batch * X2).transpose(-1, -2) @ grad_l_wrt_Z2
                b2_last = b2_init - torch.sum(last_eta_mini_batch * grad_l_wrt_Z2, dim=-2, keepdim=True)
                grad_W1_last = torch.zeros_like(W1_last)
                grad_b1_last = torch.zeros_like(b1_last)
                grad_W2_last = torch.zeros_like(W2_last)
                grad_b2_last = torch.zeros_like(b2_last)
            else:
                ttt_lr_eta_mini_batch = torch.broadcast_to(ttt_lr_eta_mini_batch, (*ttt_lr_eta_mini_batch.shape[:2], mini_batch_size, mini_batch_size))
                grad_W2 = torch.einsum('bhki,bhkj->bhkij', X2, grad_l_wrt_Z2)
                grad_W2 = torch.einsum('bhnk,bhkij->bhnij', torch.tril(ttt_lr_eta_mini_batch), grad_W2)
                grad_W2 = grad_W2 + params_dict['W2_grad'].unsqueeze(2)
                grad_b2 = torch.einsum('bhnk,bhki->bhni', torch.tril(ttt_lr_eta_mini_batch), grad_l_wrt_Z2)
                grad_b2 = grad_b2 + params_dict['b2_grad']
                grad_W1 = torch.einsum('bhki,bhkj->bhkij', X1, grad_l_wrt_Z1)
                grad_W1 = torch.einsum('bhnk,bhkij->bhnij', torch.tril(ttt_lr_eta_mini_batch), grad_W1)
                grad_W1 = grad_W1 + params_dict['W1_grad'].unsqueeze(2)
                grad_b1 = torch.einsum('bhnk,bhki->bhni', torch.tril(ttt_lr_eta_mini_batch), grad_l_wrt_Z1)
                grad_b1 = grad_b1 + params_dict['b1_grad']
                W1_bar = W1_init.unsqueeze(2) - grad_W1 * token_eta_mini_batch.unsqueeze(-1)
                b1_bar = b1_init - grad_b1 * token_eta_mini_batch
                W2_bar = W2_init.unsqueeze(2) - grad_W2 * token_eta_mini_batch.unsqueeze(-1)
                b2_bar = b2_init - grad_b2 * token_eta_mini_batch
                Z1_bar = (XQ_mini_batch.unsqueeze(3) @ W1_bar).squeeze(3) + b1_bar
                X2_bar = F.gelu(Z1_bar, approximate='tanh')
                Z2_bar = (X2_bar.unsqueeze(3) @ W2_bar).squeeze(3) + b2_bar
                W1_last = W1_bar[:, :, -1]
                b1_last = b1_bar[:, :, -1:]
                W2_last = W2_bar[:, :, -1]
                b2_last = b2_bar[:, :, -1:]
                grad_W1_last = grad_W1[:, :, -1]
                grad_b1_last = grad_b1[:, :, -1:]
                grad_W2_last = grad_W2[:, :, -1]
                grad_b2_last = grad_b2[:, :, -1:]
            Z2_bar = ln_fwd(Z2_bar, ln_weight, ln_bias)
            XQW_mini_batch = XQ_mini_batch + Z2_bar
            last_param_dict = {'W1_states': W1_last, 'b1_states': b1_last, 'W2_states': W2_last, 'b2_states': b2_last, 'W1_grad': grad_W1_last, 'b1_grad': grad_b1_last, 'W2_grad': grad_W2_last, 'b2_grad': grad_b2_last}
            return last_param_dict, XQW_mini_batch
        if last_mini_batch_params_dict is not None:
            init_params_dict = last_mini_batch_params_dict
        else:
            init_params_dict = {'W1_states': torch.tile(self.W1.unsqueeze(0), dims=(B, 1, 1, 1)), 'b1_states': torch.tile(self.b1.unsqueeze(0), dims=(B, 1, 1, 1)), 'W2_states': torch.tile(self.W2.unsqueeze(0), dims=(B, 1, 1, 1)), 'b2_states': torch.tile(self.b2.unsqueeze(0), dims=(B, 1, 1, 1))}
            init_params_dict.update(W1_grad=torch.zeros_like(init_params_dict['W1_states']))
            init_params_dict.update(b1_grad=torch.zeros_like(init_params_dict['b1_states']))
            init_params_dict.update(W2_grad=torch.zeros_like(init_params_dict['W2_states']))
            init_params_dict.update(b2_grad=torch.zeros_like(init_params_dict['b2_states']))
        inputs = tree_map(lambda x: x.permute(2, 0, 1, 3, 4), inputs)
        XQW_batch = torch.empty((num_mini_batch, B, self.num_heads, mini_batch_size, self.head_dim), device=device, dtype=dtype)
        batch_params_dict, XQW_batch = scan(compute_mini_batch, init_params_dict, inputs, XQW_batch, self.config.scan_checkpoint_group_size if self.training else 0)
        if cache_params is not None:
            cache_params.update(batch_params_dict, self.layer_idx, L)
        XQW_batch = XQW_batch.permute(1, 0, 3, 2, 4)
        XQW_batch = XQW_batch.reshape(B, L, self.width)
        return XQW_batch, batch_params_dict


class Block(nn.Module):

    def __init__(self, config: 'TTTConfig', layer_idx: 'int'):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.pre_conv = config.pre_conv
        if config.ttt_layer_type == 'linear':
            ttt_layer = TTTLinear
        elif config.ttt_layer_type == 'mlp':
            ttt_layer = TTTMLP
        else:
            raise ValueError(f'Invalid ttt_layer_type: {config.ttt_layer_type}')
        self.seq_modeling_block = ttt_layer(config=config, layer_idx=layer_idx)
        self.mlp = SwiGluMLP(config)
        if self.pre_conv:
            self.conv = Conv(config, layer_idx)
        self.seq_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.ffn_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.layer_idx = layer_idx

    def forward(self, hidden_states: 'torch.Tensor', attention_mask: 'Optional[torch.Tensor]'=None, position_ids: 'Optional[torch.LongTensor]'=None, cache_params: 'Optional[TTTCache]'=None):
        if self.pre_conv:
            residual = hidden_states
            hidden_states = self.conv(hidden_states, cache_params=cache_params)
            hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.seq_norm(hidden_states)
        hidden_states = self.seq_modeling_block(hidden_states=hidden_states, attention_mask=attention_mask, position_ids=position_ids, cache_params=cache_params)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.ffn_norm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (RMSNorm,
     lambda: ([], {'hidden_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (RotaryEmbedding,
     lambda: ([], {'dim': 4}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4])], {})),
]

