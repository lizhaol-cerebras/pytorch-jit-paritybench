
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


from torch import nn


from typing import Optional


from typing import Tuple


from torch import Tensor


from typing import Callable


import math


from typing import Union


import numpy as np


import re


import warnings


from torch.utils.cpp_extension import CUDA_HOME


from torch.utils.cpp_extension import BuildExtension


from torch.utils.cpp_extension import CUDAExtension


from torch.nn import functional as F


import random


from torch.utils.data import DataLoader


from torch.utils.data import Dataset


def eval_decorator(fn):

    def inner(model, *args, **kwargs):
        was_training = model.training
        model.eval()
        out = fn(model, *args, **kwargs)
        model.train(was_training)
        return out
    return inner


def exists(val):
    return val is not None


def top_k(logits, thres=0.9):
    k = int((1 - thres) * logits.shape[-1])
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(1, ind, val)
    return probs


class AutoregressiveWrapper(nn.Module):
    """
    AutoregressiveWrapper is a wrapper class that adds autoregressive generation functionality to a given neural network.

    Args:
        net (nn.Module): The neural network model.
        max_seq_len (int): The maximum sequence length for generation. Defaults to 2048.
        pad_value (int): The padding value for generated sequences. Defaults to 0.
    """

    def __init__(self, net, max_seq_len=2048, pad_value=0):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.pad_value = pad_value
        self.net = net

    @torch.no_grad()
    @eval_decorator
    def generate(self, start_tokens, seq_len, eos_token=None, temperature=1.0, filter_thres=0.9, **kwargs):
        """
        Generates autoregressive sequences based on the given start tokens.

        Args:
            start_tokens (torch.Tensor): The initial tokens to start the generation.
            seq_len (int): The length of the generated sequence.
            eos_token (int, optional): The end-of-sequence token. If provided, generation will stop when this token is generated. Defaults to None.
            temperature (float, optional): The temperature value for controlling the randomness of the generation. Higher values result in more randomness. Defaults to 1.0.
            filter_thres (float, optional): The threshold value for filtering logits during generation. Only logits above this threshold will be considered. Defaults to 0.9.
            **kwargs: Additional keyword arguments to be passed to the underlying network.

        Returns:
            torch.Tensor: The generated sequence.
        """
        b, t, device = *start_tokens.shape, start_tokens.device
        out = start_tokens
        for _ in range(seq_len):
            logits = self.net(out, **kwargs)[:, -1, :]
            filtered_logits = top_k(logits, thres=filter_thres)
            probs = F.softmax(filtered_logits / temperature, dim=-1)
            sample = torch.multinomial(probs, 1)
            out = torch.cat((out, sample), dim=-1)
            if exists(eos_token):
                is_eos_token = out == eos_token
                if is_eos_token.any(dim=-1).all():
                    shifted_is_eos_tokens = F.pad(is_eos_token, (1, -1))
                    mask = shifted_is_eos_tokens.float().cumsum(dim=-1) >= 1
                    out = out.masked_fill(mask, self.pad_value)
                    break
        out = out[:, t:]
        return out

    def forward(self, x, **kwargs):
        x_inp, x_labels = x[:, :-1], x[:, 1:]
        logits = self.net(x_inp, **kwargs)
        return F.cross_entropy(rearrange(logits, 'b c n -> b n c'), x_labels)


def activation_quant(x: 'Tensor'):
    """Per token quantization to 8bits. No grouping is needed for quantization

    Args:
        x (Tensor): _description_

    Returns:
        _type_: _description_
    """
    scale = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-05)
    y = (x * scale).round().clamp_(-128, 127) / scale
    return y


def weight_quant(w: 'Tensor'):
    scale = w.abs().mean()
    e = w.mean()
    u = (w - e).sign() * scale
    return u


class BitLinear(nn.Linear):
    """
    Custom linear layer with bit quantization.

    Args:
        dim (int): The input dimension of the layer.
        training (bool, optional): Whether the layer is in training mode or not. Defaults to False.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Attributes:
        dim (int): The input dimension of the layer.

    """

    def forward(self, x: 'Tensor') ->Tensor:
        """
        Forward pass of the BitLinear layer.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor.

        """
        w = self.weight
        x_norm = SimpleRMSNorm(self.in_features)(x)
        x_quant = x_norm + (activation_quant(x_norm) - x_norm).detach()
        w_quant = w + (weight_quant(w) - w).detach()
        y = F.linear(x_quant, w_quant)
        return y


def scaled_dot_product_gqa(query: 'Tensor', key: 'Tensor', value: 'Tensor', dropout: 'float'=0.0, scale: 'Optional[float]'=None, mask: 'Optional[Tensor]'=None, is_causal: 'Optional[bool]'=None, need_weights: 'bool'=False, average_attn_weights: 'bool'=False, force_grouped: 'bool'=False):
    """Scaled dot product attention with support for grouped queries.

    Einstein notation:
    - b: batch size
    - n / s: sequence length
    - h: number of heads
    - g: number of groups
    - d: dimension of query/key/value

    Args:
        query: Query tensor of shape (b, n, h, d)
        key: Key tensor of shape (b, s, h, d)
        value: Value tensor of shape (b, s, h, d)
        dropout: Dropout probability (default: 0.0)
        scale: Scale factor for query (default: d_query ** 0.5)
        mask: Mask tensor of shape (b, n, s) or (b, s). If 'ndim == 2', the mask is
            applied to all 'n' rows of the attention matrix. (default: None)
        force_grouped: If True, apply grouped-query attention even if the number of
            heads is equal for query, key, and value. (default: False)

    Returns:
        2-tuple of:
        - Attention output with shape (b, n, h, d)
        - (Optional) Attention weights with shape (b, h, n, s). Only returned if
          'need_weights' is True.
    """
    if mask is not None and is_causal is not None:
        raise ValueError("Only one of 'mask' and 'is_causal' should be provided, but got both.")
    elif not query.ndim == key.ndim == value.ndim == 4:
        raise ValueError(f'Expected query, key, and value to be 4-dimensional, but got shapes {query.shape}, {key.shape}, and {value.shape}.')
    query = rearrange(query, 'b n h d -> b h n d')
    key = rearrange(key, 'b s h d -> b h s d')
    value = rearrange(value, 'b s h d -> b h s d')
    bq, hq, nq, dq = query.shape
    bk, hk, nk, dk = key.shape
    bv, hv, nv, dv = value.shape
    if not (bq == bk == bv and dq == dk == dv):
        raise ValueError(f'Expected query, key, and value to have the same batch size (dim=0) and embedding dimension (dim=3), but got query: {query.shape}, key: {key.shape}, and value: {value.shape}.')
    elif hk != hv or nk != nv:
        raise ValueError(f'Expected key and value to have the same size in dimensions 1 and 2, but got key: {key.shape} and value: {value.shape}.')
    elif hq % hk != 0:
        raise ValueError(f'Expected query heads to be a multiple of key/value heads, but got query: {query.shape} and key/value: {key.shape}.')
    if scale is None:
        scale = query.size(-1) ** 0.5
    query = query / scale
    num_head_groups = hq // hk
    if num_head_groups > 1 or force_grouped:
        query = rearrange(query, 'b (h g) n d -> b g h n d', g=num_head_groups)
        similarity = einsum(query, key, 'b g h n d, b h s d -> b h n s')
    else:
        similarity = einsum(query, key, 'b h n d, b h s d -> b h n s')
    if is_causal:
        mask = torch.ones((bq, nq, nk), device=query.device, dtype=torch.bool).tril_()
    if mask is not None:
        if mask.ndim == 2:
            mask = rearrange(mask, 'b s -> b () () s')
        elif mask.ndim == 3:
            mask = rearrange(mask, 'b n s -> b () n s')
        similarity.masked_fill_(~mask, torch.finfo(similarity.dtype).min)
    attention = F.softmax(similarity / scale, dim=-1)
    if dropout > 0.0:
        attention = F.dropout(attention, p=dropout)
    out = einsum(attention, value, 'b h n s, b h s d -> b h n d')
    out = rearrange(out, 'b h n d -> b n h d')
    attn_weights: 'Optional[Tensor]' = None
    if need_weights:
        attn_weights = rearrange(attention, 'b h n s -> b n s h')
        if average_attn_weights:
            attn_weights = attn_weights.mean(dim=1)
    return out, attn_weights


class BitMGQA(nn.Module):
    """Multi-head grouped query attention (GQA) layer.

    Reference:
        "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints"
        https://arxiv.org/pdf/2305.13245v1.pdf

    GQA is a variant of multihead attention (MHA) that uses fewer write heads
    (key / value) than query heads.  GQA can be viewed as a generalization of
    multi-query attention (MQA), which uses a single write head. GQA and MQA give
    significant speedups over standard MHA in decoder layers, with minimal loss in
    accuracy. In the paper, GQA is shown to be more accurate than MQA, while still
    having a significant speedup over MHA.

    NOTE: The original authors only benchmark GQA by adapting the T5 (XL or XXL) model
    from MHA to GQA.  As a result, they do not mention parameter initialization or
    layer normalization strategies.  I follow the best practices laid out in the
    MAGNETO paper, which improves Transformer performance through better parameter
    initialization and layer norm placement.  See:
        https://arxiv.org/pdf/2210.06423.pdf, Fig. 2
    """

    def __init__(self, embed_dim: 'int', query_heads: 'int'=8, kv_heads: 'int'=4, dropout: 'float'=0.1, bias: 'bool'=True, layer_norm: 'bool'=True, layer_norm_eps: 'float'=1e-05, gamma_init: 'float'=1.0, linear_groups: 'int'=1, *args, **kwargs):
        super().__init__()
        self.query_heads = query_heads
        self.kv_heads = kv_heads
        self.dropout = dropout
        self.layer_norm = layer_norm
        self.gamma_init = gamma_init
        if self.query_heads % self.kv_heads != 0:
            raise ValueError(f'query_heads ({query_heads}) must be divisible by kv_heads ({kv_heads})')
        elif embed_dim % self.query_heads != 0 or embed_dim % self.kv_heads != 0:
            raise ValueError(f'embed_dim ({embed_dim}) must be divisible by query_heads ({query_heads}) and kv_heads ({kv_heads})')
        head_dim = embed_dim // query_heads
        if not head_dim % 8 == 0:
            raise ValueError(f'head_dim (embed_dim / num_heads = {head_dim}) must be divisible by 8')
        if not head_dim <= 128:
            raise ValueError(f'head_dim (embed_dim / num_heads = {head_dim}) must be <= 128')
        self.q_proj = BitLinear(embed_dim, embed_dim, *args, bias=bias, **kwargs)
        kv_embed_dim = embed_dim // query_heads * kv_heads
        self.k_proj = BitLinear(embed_dim, kv_embed_dim, *args, bias=bias, **kwargs)
        self.v_proj = BitLinear(embed_dim, kv_embed_dim, *args, bias=bias, **kwargs)
        self.norm: 'Optional[nn.LayerNorm]' = None
        if layer_norm:
            self.norm = nn.LayerNorm(kv_embed_dim, eps=layer_norm_eps)
        self.out_proj = BitLinear(kv_embed_dim, embed_dim, bias=bias)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_normal_(self.q_proj.weight)
        if self.q_proj.bias is not None:
            nn.init.constant_(self.q_proj.bias, 0)
        nn.init.xavier_normal_(self.k_proj.weight)
        if self.k_proj.bias is not None:
            nn.init.constant_(self.k_proj.bias, 0)
        nn.init.xavier_normal_(self.v_proj.weight, gain=self.gamma_init)
        if self.v_proj.bias is not None:
            nn.init.constant_(self.v_proj.bias, 0)
        nn.init.xavier_normal_(self.out_proj.weight, gain=self.gamma_init)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0)

    def forward(self, query: 'Tensor', key: 'Tensor', value: 'Tensor', need_weights: 'bool'=False, is_causal: 'bool'=False, average_attn_weights: 'bool'=False) ->Tuple[Tensor, Optional[Tensor]]:
        q: 'Tensor' = self.q_proj(query)
        k: 'Tensor' = self.k_proj(key)
        v: 'Tensor' = self.v_proj(value)
        q = rearrange(q, 'b n (h d) -> b n h d', h=self.query_heads)
        k = rearrange(k, 'b n (h d) -> b n h d', h=self.kv_heads)
        v = rearrange(v, 'b n (h d) -> b n h d', h=self.kv_heads)
        x, attn_weights = scaled_dot_product_gqa(query=q, key=k, value=v, is_causal=is_causal, need_weights=need_weights, average_attn_weights=average_attn_weights, force_grouped=False)
        x = rearrange(x, 'b n h d -> b n (h d)')
        if self.layer_norm:
            assert self.norm is not None
            x = self.norm(x)
        x = self.out_proj(x)
        return x, attn_weights


class GLU(nn.Module):
    """
    Gated Linear Unit (GLU) module.

    Args:
        dim_in (int): Input dimension.
        dim_out (int): Output dimension.
        activation (Callable): Activation function to be applied to the gate.
        mult_bias (bool, optional): Whether to multiply the bias term. Defaults to False.
        linear (Callable, optional): Linear function to be used for projection. Defaults to False.
    """

    def __init__(self, dim_in: 'int', dim_out: 'int', activation: 'Callable', mult_bias: 'bool'=False, linear: 'Callable'=False, *args, **kwargs):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.activation = activation
        self.mult_bias = mult_bias
        if linear:
            self.proj = linear(dim_in, dim_out * 2)
        else:
            self.proj = BitLinear(dim_in, dim_out * 4, *args, **kwargs)
        self.mult_bias = nn.Parameter(torch.ones(dim_out)) if mult_bias else 1.0

    def forward(self, x: 'Tensor'):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * self.activation(gate) * self.mult_bias


def default(val, d):
    return val if val is not None else d


def init_zero_(tensor):
    nn.init.constant_(tensor, 0.0)


class BitFeedForward(nn.Module):
    """
    BitFeedForward module performs feed-forward operations on the input tensor.

    Args:
        dim (int): The input dimension.
        dim_out (int, optional): The output dimension. If not provided, it is set to the input dimension.
        mult (int, optional): The multiplier for the inner dimension. Default is 4.
        glu (bool, optional): Whether to use Gated Linear Unit (GLU) activation. Default is False.
        glu_mult_bias (bool, optional): Whether to apply bias to the GLU activation. Default is False.
        swish (bool, optional): Whether to use Swish activation. Default is False.
        relu_squared (bool, optional): Whether to use squared ReLU activation. Default is False.
        post_act_ln (bool, optional): Whether to apply Layer Normalization after activation. Default is False.
        dropout (float, optional): The dropout probability. Default is 0.0.
        no_bias (bool, optional): Whether to exclude bias in linear layers. Default is False.
        zero_init_output (bool, optional): Whether to initialize the last linear layer to 0. Default is False.
    """

    def __init__(self, dim: 'int', dim_out: 'Optional[int]'=None, mult: 'int'=4, glu: 'bool'=False, glu_mult_bias: 'bool'=False, swish: 'bool'=False, post_act_ln: 'bool'=False, dropout: 'float'=0.0, no_bias: 'bool'=False, zero_init_output: 'bool'=False, *args, **kwargs):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        if swish:
            activation = nn.SiLU()
        else:
            activation = nn.GELU()
        if glu:
            project_in = GLU(dim, inner_dim, activation, mult_bias=glu_mult_bias)
        else:
            project_in = nn.Sequential(BitLinear(dim, inner_dim, *args, bias=not no_bias, **kwargs), activation)
        if post_act_ln:
            self.ff = nn.Sequential(project_in, nn.LayerNorm(inner_dim), nn.Dropout(dropout), BitLinear(inner_dim, dim_out, *args, bias=not no_bias, **kwargs))
        else:
            self.ff = nn.Sequential(project_in, nn.Dropout(dropout), BitLinear(inner_dim, dim_out, *args, bias=not no_bias, **kwargs))
        if zero_init_output:
            init_zero_(self.ff[-1])

    def forward(self, x):
        """
        Forward pass of the BitFeedForward module.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        return self.ff(x)


class BitLinearNew(nn.Linear):
    """
    Custom linear layer with bit quantization.

    Args:
        dim (int): The input dimension of the layer.
        training (bool, optional): Whether the layer is in training mode or not. Defaults to False.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Attributes:
        dim (int): The input dimension of the layer.

    """

    def forward(self, x: 'Tensor') ->Tensor:
        """
        Forward pass of the BitLinear layer.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor.

        """
        w = self.weight
        x_norm = SimpleRMSNorm(self.in_features)(x)
        x_quant = x_norm + (activation_quant(x_norm) - x_norm).detach()
        w_quant = w + (weight_quant(w) - w).detach()
        y = F.linear(x_quant, w_quant)
        return y


class RMSNorm(nn.Module):

    def __init__(self, heads, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(heads, 1, dim) / self.scale)

    def forward(self, x):
        normed = F.normalize(x, dim=-1)
        return normed * self.scale * self.gamma


def reshape_for_broadcast(freqs_cis: 'torch.Tensor', x: 'torch.Tensor'):
    """
    Reshape frequency tensor for broadcasting it with another tensor.

    This function reshapes the frequency tensor to have the same shape as the target tensor 'x'
    for the purpose of broadcasting the frequency tensor during element-wise operations.

    Args:
        freqs_cis (torch.Tensor): Frequency tensor to be reshaped.
        x (torch.Tensor): Target tensor for broadcasting compatibility.

    Returns:
        torch.Tensor: Reshaped frequency tensor.

    Raises:
        AssertionError: If the frequency tensor doesn't match the expected shape.
        AssertionError: If the target tensor 'x' doesn't have the expected number of dimensions.
    """
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [(d if i == 1 or i == ndim - 1 else 1) for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(xq: 'torch.Tensor', xk: 'torch.Tensor', freqs_cis: 'torch.Tensor') ->Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor.

    This function applies rotary embeddings to the given query 'xq' and key 'xk' tensors using the provided
    frequency tensor 'freqs_cis'. The input tensors are reshaped as complex numbers, and the frequency tensor
    is reshaped for broadcasting compatibility. The resulting tensors contain rotary embeddings and are
    returned as real tensors.

    Args:
        xq (torch.Tensor): Query tensor to apply rotary embeddings.
        xk (torch.Tensor): Key tensor to apply rotary embeddings.
        freqs_cis (torch.Tensor): Precomputed frequency tensor for complex exponentials.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.



    """
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: 'torch.Tensor', n_rep: 'int') ->torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return x[:, :, :, None, :].expand(bs, slen, n_kv_heads, n_rep, head_dim).reshape(bs, slen, n_kv_heads * n_rep, head_dim)


class Attention(nn.Module):
    """Multi-head attention module."""

    def __init__(self, args: 'ModelArgs'):
        """
        Initialize the Attention module.

        Args:
            args (ModelArgs): Model configuration parameters.

        Attributes:
            n_kv_heads (int): Number of key and value heads.
            n_local_heads (int): Number of local query heads.
            n_local_kv_heads (int): Number of local key and value heads.
            n_rep (int): Number of repetitions for local heads.
            head_dim (int): Dimension size of each attention head.
            wq (ColumnParallelLinear): Linear transformation for queries.
            wk (ColumnParallelLinear): Linear transformation for keys.
            wv (ColumnParallelLinear): Linear transformation for values.
            wo (RowParallelLinear): Linear transformation for output.
            cache_k (torch.Tensor): Cached keys for attention.
            cache_v (torch.Tensor): Cached values for attention.

        """
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        model_parallel_size = fs_init.get_model_parallel_world_size()
        self.n_local_heads = args.n_heads // model_parallel_size
        self.n_local_kv_heads = self.n_kv_heads // model_parallel_size
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads
        self.wq = ColumnParallelLinear(args.dim, args.n_heads * self.head_dim, bias=False, gather_output=False, init_method=lambda x: x)
        self.wk = ColumnParallelLinear(args.dim, self.n_kv_heads * self.head_dim, bias=False, gather_output=False, init_method=lambda x: x)
        self.wv = ColumnParallelLinear(args.dim, self.n_kv_heads * self.head_dim, bias=False, gather_output=False, init_method=lambda x: x)
        self.wo = RowParallelLinear(args.n_heads * self.head_dim, args.dim, bias=False, input_is_parallel=True, init_method=lambda x: x)
        self.cache_k = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_local_kv_heads, self.head_dim))
        self.cache_v = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_local_kv_heads, self.head_dim))

    def forward(self, x: 'torch.Tensor', start_pos: 'int', freqs_cis: 'torch.Tensor', mask: 'Optional[torch.Tensor]'):
        """
        Forward pass of the attention module.

        Args:
            x (torch.Tensor): Input tensor.
            start_pos (int): Starting position for caching.
            freqs_cis (torch.Tensor): Precomputed frequency tensor.
            mask (torch.Tensor, optional): Attention mask tensor.

        Returns:
            torch.Tensor: Output tensor after attention.

        """
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)
        self.cache_k = self.cache_k
        self.cache_v = self.cache_v
        self.cache_k[:bsz, start_pos:start_pos + seqlen] = xk
        self.cache_v[:bsz, start_pos:start_pos + seqlen] = xv
        keys = self.cache_k[:bsz, :start_pos + seqlen]
        values = self.cache_v[:bsz, :start_pos + seqlen]
        keys = repeat_kv(keys, self.n_rep)
        values = repeat_kv(values, self.n_rep)
        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, values)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output)


class FeedForward(nn.Module):

    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(dim), BitLinear(dim, hidden_dim), nn.GELU(), BitLinear(hidden_dim, dim))

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):

    def __init__(self, layer_id: 'int', args: 'ModelArgs'):
        """
        Initialize a TransformerBlock.

        Args:
            layer_id (int): Identifier for the layer.
            args (ModelArgs): Model configuration parameters.

        Attributes:
            n_heads (int): Number of attention heads.
            dim (int): Dimension size of the model.
            head_dim (int): Dimension size of each attention head.
            attention (Attention): Attention module.
            feed_forward (FeedForward): FeedForward module.
            layer_id (int): Identifier for the layer.
            attention_norm (RMSNorm): Layer normalization for attention output.
            ffn_norm (RMSNorm): Layer normalization for feedforward output.

        """
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = FeedForward(dim=args.dim, hidden_dim=4 * args.dim, multiple_of=args.multiple_of, ffn_dim_multiplier=args.ffn_dim_multiplier)
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x: 'torch.Tensor', start_pos: 'int', freqs_cis: 'torch.Tensor', mask: 'Optional[torch.Tensor]'):
        """
        Perform a forward pass through the TransformerBlock.

        Args:
            x (torch.Tensor): Input tensor.
            start_pos (int): Starting position for attention caching.
            freqs_cis (torch.Tensor): Precomputed cosine and sine frequencies.
            mask (torch.Tensor, optional): Masking tensor for attention. Defaults to None.

        Returns:
            torch.Tensor: Output tensor after applying attention and feedforward layers.

        """
        h = x + self.attention.forward(self.attention_norm(x), start_pos, freqs_cis, mask)
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out


class Transformer(nn.Module):

    def __init__(self, dim, depth, heads, dim_head, mlp_dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([MultiQueryAttention(dim, heads), FeedForward(dim, mlp_dim)]))

    def forward(self, x):
        for attn, ff in self.layers:
            x, _, _ = attn(x)
            x = self.norm(x) + x
            x = ff(x) + x
        return self.norm(x)


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
        self.in_proj = BitLinear(config.dim, 2 * config.d_inner, bias=config.bias)
        self.conv1d = nn.Conv1d(in_channels=config.d_inner, out_channels=config.d_inner, kernel_size=config.d_conv, bias=config.conv_bias, groups=config.d_inner, padding=config.d_conv - 1)
        self.x_proj = BitLinear(config.d_inner, config.dt_rank + 2 * config.d_state, bias=False)
        self.dt_proj = BitLinear(config.dt_rank, config.d_inner, bias=True)
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
        self.out_proj = BitLinear(config.d_inner, config.dim, bias=config.bias)

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

    def __init__(self, num_tokens: 'int', sequence_length: 'int', config: 'MambaConfig', return_embeddings: 'bool'=True, return_tokens: 'bool'=True):
        super().__init__()
        self.num_tokens = num_tokens
        self.sequence_length = sequence_length
        self.config = config
        self.return_embeddings = return_embeddings
        self.return_tokens = return_tokens
        self.token_embed = nn.Embedding(num_tokens, config.dim)
        self.norm = nn.LayerNorm(config.dim)
        self.layers = nn.ModuleList([ResidualBlock(config) for _ in range(config.depth)])

    def forward(self, x):
        x = self.token_embed(x)
        x = self.norm(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        if self.return_tokens:
            x = OutputHead(self.config.dim, -1)(x)
            return x
        else:
            return x

    def step(self, x, caches):
        for i, layer in enumerate(self.layers):
            x, caches[i] = layer.step(x, caches[i])
        return x, caches


class BitMamba(nn.Module):
    """
    BitMamba module for performing computations using the BitNet architecture.

    Args:
        dim (int): The input dimension (D).
        depth (int): The depth of the BitNet architecture.
        dt_rank (Union[int, str], optional): The rank of the time step tensor. Defaults to "auto".
        d_state (int, optional): The dimension of the state tensor (N in paper/comments). Defaults to 16.
        expand_factor (int, optional): The expansion factor for the inner dimension (E in paper/comments). Defaults to 2.
        d_conv (int, optional): The dimension of the convolutional filters. Defaults to 4.
        dt_min (float, optional): The minimum value for the time step. Defaults to 0.001.
        dt_max (float, optional): The maximum value for the time step. Defaults to 0.1.
        dt_init (str, optional): The initialization method for the time step. Can be "random" or "constant". Defaults to "random".
        dt_scale (float, optional): The scaling factor for the time step. Defaults to 1.0.
        dt_init_floor (float, optional): The floor value for the initialized time step. Defaults to 1e-4.
        bias (bool, optional): Whether to include bias terms. Defaults to False.
        conv_bias (bool, optional): Whether to include bias terms in the convolutional layers. Defaults to True.
        pscan (bool, optional): Whether to use parallel scan mode or sequential mode when training. Defaults to True.
    """

    def __init__(self, dim: 'int', num_tokens: 'int', sequence_length: 'int', depth: 'int', dt_rank: 'Union[int, str]'='auto', d_state: 'int'=16, expand_factor: 'int'=2, d_conv: 'int'=4, dt_min: 'float'=0.001, dt_max: 'float'=0.1, dt_init: 'str'='random', dt_scale: 'float'=1.0, dt_init_floor=0.0001, bias: 'bool'=False, conv_bias: 'bool'=True, pscan: 'bool'=True, return_embeddings: 'bool'=True, return_tokens: 'bool'=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dim = dim
        self.num_token = num_tokens
        self.sequence_length = sequence_length
        self.depth = depth
        self.dt_rank = dt_rank
        self.d_state = d_state
        self.expand_factor = expand_factor
        self.d_conv = d_conv
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.dt_init = dt_init
        self.dt_scale = dt_scale
        self.dt_init_floor = dt_init_floor
        self.bias = bias
        self.conv_bias = conv_bias
        self.pscan = pscan
        self.return_embeddings = return_embeddings
        self.return_tokens = return_tokens
        self.d_inner = self.expand_factor * self.dim
        if self.dt_rank == 'auto':
            self.dt_rank = math.ceil(self.dim / 16)
        config = MambaConfig(dim=self.dim, depth=self.depth, dt_rank=self.dt_rank, d_state=self.d_state, expand_factor=self.expand_factor, d_conv=self.d_conv, dt_min=self.dt_min, dt_max=self.dt_max, dt_init=self.dt_init, dt_scale=self.dt_scale, bias=self.bias, conv_bias=self.conv_bias, pscan=self.pscan)
        self.mamba = Mamba(num_tokens=self.num_token, sequence_length=self.sequence_length, config=config, return_embeddings=self.return_embeddings, return_tokens=self.return_tokens)

    def forward(self, x):
        return self.mamba(x)


class Expert(nn.Module):
    """An MLP is a simple linear layer followed by a non-linearity i.e. each Expert

    Args:
        dim (int): The input dimension of the linear layer.
        dropout (float, optional): The dropout probability. Defaults to 0.1.

    Attributes:
        net (nn.Sequential): The sequential network consisting of linear layers, ReLU activation, and dropout.

    """

    def __init__(self, dim: 'int', dropout: 'int'=0.1):
        super().__init__()
        self.net = nn.Sequential(BitLinear(dim, 4 * dim), nn.ReLU(), BitLinear(4 * dim, dim), nn.Dropout(dropout))

    def forward(self, x):
        return self.net(x)


class NoisyTopkRouter(nn.Module):
    """
    A class representing a Noisy Top-k Router module.

    This module takes the output tensor from a multihead self attention block and performs routing
    by selecting the top-k experts based on the logits. It adds scaled unit Gaussian noise to the logits
    and applies softmax to obtain the final router output.

    Args:
        dim (int): The input dimension of the tensor.
        num_experts (int): The number of experts.
        top_k (int): The number of experts to select.

    Attributes:
        top_k (int): The number of experts to select.
        topkroute_linear (BitLinear): The linear layer for router logits.
        noise_linear (BitLinear): The linear layer for noise logits.
    """

    def __init__(self, dim, num_experts, top_k):
        super(NoisyTopkRouter, self).__init__()
        self.top_k = top_k
        self.topkroute_linear = BitLinear(dim, num_experts)
        self.noise_linear = BitLinear(dim, num_experts)

    def forward(self, mh_output):
        """
        Forward pass of the NoisyTopkRouter module.

        Args:
            mh_output (torch.Tensor): The output tensor from the multihead self attention block.

        Returns:
            tuple: A tuple containing the router output tensor and the indices of the selected experts.
        """
        logits = self.topkroute_linear(mh_output)
        noise_logits = self.noise_linear(mh_output)
        noise = torch.randn_like(logits) * F.softplus(noise_logits)
        noisy_logits = logits + noise
        top_k_logits, indices = noisy_logits.topk(self.top_k, dim=-1)
        zeros = torch.full_like(noisy_logits, float('-inf'))
        sparse_logits = zeros.scatter(-1, indices, top_k_logits)
        router_output = F.softmax(sparse_logits, dim=-1)
        return router_output, indices


class BitMoE(nn.Module):
    """
    BitMoE (Bitwise Mixture of Experts) module.

    Args:
        dim (int): The input dimension.
        num_experts (int): The number of experts in the mixture.
        top_k (int, optional): The number of experts to select for each input. Defaults to 2.
    """

    def __init__(self, dim: 'int', num_experts: 'int', top_k: 'int'=2):
        super(BitMoE, self).__init__()
        self.router = NoisyTopkRouter(dim, num_experts, top_k)
        self.experts = nn.ModuleList([Expert(dim) for _ in range(num_experts)])
        self.top_k = top_k

    def forward(self, x):
        gating_output, indices = self.router(x)
        final_output = torch.zeros_like(x)
        flat_x = x.view(-1, x.size(-1))
        flat_gating_output = gating_output.view(-1, gating_output.size(-1))
        for i, expert in enumerate(self.experts):
            expert_mask = (indices == i).any(dim=-1)
            flat_mask = expert_mask.view(-1)
            if flat_mask.any():
                expert_input = flat_x[flat_mask]
                expert_output = expert(expert_input)
                gating_scores = flat_gating_output[flat_mask, i].unsqueeze(1)
                weighted_output = expert_output * gating_scores
                final_output[expert_mask] += weighted_output.squeeze(1)
        return final_output


class BitNetTransformer(nn.Module):
    """
    BitNetTransformer is a transformer-based model for BitNet.

    Args:
        dim (int): The dimension of the token embeddings.
        depth (int): The number of transformer layers.
        num_tokens (int): The number of tokens in the vocabulary.
        heads (int, optional): The number of attention heads in the transformer. Defaults to 8.
        ff_mult (int, optional): The multiplier for the feed-forward layer dimension. Defaults to 4.

    Examples:
    >>> import torch
    >>> from bitnet import BitNetTransformer
    >>> x = torch.randint(0, 20000, (1, 1024))
    >>> bitnet = BitNetTransformer(
    ...     num_tokens=20000,
    ...     dim=1024,
    ...     depth=6,
    ...     heads=8,
    ...     ff_mult=4,
    ... )
    >>> logits = bitnet(x)
    >>> print(logits)
    """

    def __init__(self, dim: 'int', depth: 'int', num_tokens: 'int', heads: 'int'=8, ff_mult: 'int'=4):
        super().__init__()
        self.emb = nn.Embedding(num_tokens, dim)
        self.transformer = Transformer(dim=dim, depth=depth, heads=heads, ff_mult=ff_mult)
        self.to_logits = OutputHead(dim, vocab_size=num_tokens)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = self.emb(x)
        x = self.norm(x)
        x = self.transformer(x)
        return self.to_logits(x)


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


def posemb_sincos_2d(h, w, dim, temperature: 'int'=10000, dtype=torch.float32):
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
    assert dim % 4 == 0, 'feature dimension must be multiple of 4 for sincos emb'
    omega = torch.arange(dim // 4) / (dim // 4 - 1)
    omega = 1.0 / temperature ** omega
    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    return pe.type(dtype)


class OneBitViT(nn.Module):
    """
    OneBitViT is a vision transformer model for image classification tasks.

    Args:
        image_size (int or tuple): The size of the input image. If an integer is provided, it is assumed to be a square image.
        patch_size (int or tuple): The size of each patch in the image. If an integer is provided, it is assumed to be a square patch.
        num_classes (int): The number of output classes.
        dim (int): The dimensionality of the token embeddings and the positional embeddings.
        depth (int): The number of transformer layers.
        heads (int): The number of attention heads in the transformer.
        mlp_dim (int): The dimensionality of the feed-forward network in the transformer.
        channels (int): The number of input channels in the image. Default is 3.
        dim_head (int): The dimensionality of each attention head. Default is 64.

    Attributes:
        to_patch_embedding (nn.Sequential): Sequential module for converting image patches to embeddings.
        pos_embedding (torch.Tensor): Positional embeddings for the patches.
        transformer (Transformer): Transformer module for processing the embeddings.
        pool (str): Pooling method used to aggregate the patch embeddings. Default is "mean".
        to_latent (nn.Identity): Identity module for converting the transformer output to the final latent representation.
        linear_head (nn.LayerNorm): Layer normalization module for the final linear projection.

    Methods:
        forward(img): Performs a forward pass through the OneBitViT model.

    """

    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels=3, dim_head=64):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        patch_dim = channels * patch_height * patch_width
        self.to_patch_embedding = nn.Sequential(Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width), nn.LayerNorm(patch_dim), BitLinear(patch_dim, dim), nn.LayerNorm(dim))
        self.pos_embedding = posemb_sincos_2d(h=image_height // patch_height, w=image_width // patch_width, dim=dim)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim)
        self.pool = 'mean'
        self.to_latent = nn.Identity()
        self.linear_head = nn.LayerNorm(dim)

    def forward(self, img):
        device = img.device
        x = self.to_patch_embedding(img)
        x += self.pos_embedding
        x = self.transformer(x)
        x = x.mean(dim=1)
        x = self.to_latent(x)
        return self.linear_head(x)


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (RMSNorm,
     lambda: ([], {'heads': 4, 'dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
]

