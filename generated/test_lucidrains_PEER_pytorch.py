
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


from torch.nn import Module


from functools import partial


from torch.utils.checkpoint import checkpoint


from math import sqrt


from torch import nn


import torch.nn.functional as F


from torch.nn import ModuleList


class ChunkedPEER(Module):

    def __init__(self, peer: 'PEER | PEERLora', seq_chunk_size: 'int'=128):
        super().__init__()
        self.peer = peer
        self.seq_chunk_size = seq_chunk_size

    def forward(self, x):
        peer = self.peer
        if self.training and x.requires_grad:
            peer = partial(checkpoint, peer)
        out = []
        for chunk in x.split(self.seq_chunk_size, dim=1):
            chunk_out = peer(chunk)
            out.append(chunk_out)
        return torch.cat(out, dim=1)


class RMSNorm(Module):

    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        return F.normalize(x, dim=-1) * self.scale * (self.gamma + 1)


def exists(v):
    return v is not None


def default(v, d):
    return v if exists(v) else d


class PEER(Module):
    """
    following Algorithm 1 in the paper
    """

    def __init__(self, dim, *, heads=8, num_experts=1000000, num_experts_per_head=16, activation=nn.GELU, dim_key=None, product_key_topk=None, separate_embed_per_head=False, pre_rmsnorm=False, non_competing_scores=True, dropout=0.0):
        """
        einops notation
        b - batch
        n - sequence
        d - dimension
        h - heads
        p - 2 for product key
        k - number of keys
        """
        super().__init__()
        self.norm = RMSNorm(dim) if pre_rmsnorm else nn.Identity()
        num_expert_sets = 1 if not separate_embed_per_head else heads
        self.heads = heads
        self.separate_embed_per_head = separate_embed_per_head
        self.num_experts = num_experts
        self.weight_down_embed = nn.Embedding(num_experts * num_expert_sets, dim)
        self.weight_up_embed = nn.Embedding(num_experts * num_expert_sets, dim)
        self.activation = activation()
        assert sqrt(num_experts).is_integer(), '`num_experts` needs to be a square'
        assert dim % 2 == 0, 'feature dimension should be divisible by 2'
        dim_key = default(dim_key, dim // 2)
        self.num_keys = int(sqrt(num_experts))
        self.to_queries = nn.Sequential(nn.Linear(dim, dim_key * heads * 2, bias=False), Rearrange('b n (p h d) -> p b n h d', p=2, h=heads))
        self.product_key_topk = default(product_key_topk, num_experts_per_head)
        self.num_experts_per_head_topk = num_experts_per_head if not non_competing_scores else 1
        self.keys = nn.Parameter(torch.zeros(heads, self.num_keys, 2, dim_key))
        nn.init.normal_(self.keys, std=0.02)
        self.dropout = nn.Dropout(dropout)
        self.score_activation = nn.Softmax(dim=-1) if not non_competing_scores else nn.ReLU()

    def forward(self, x):
        x = self.norm(x)
        queries = self.to_queries(x)
        sim = einsum(queries, self.keys, 'p b n h d, h k p d -> p b n h k')
        (scores_x, scores_y), (indices_x, indices_y) = sim.topk(self.product_key_topk, dim=-1)
        all_scores = einx.add('... i, ... j -> ... (i j)', scores_x, scores_y)
        all_indices = einx.add('... i, ... j -> ... (i j)', indices_x * self.num_keys, indices_y)
        scores, pk_indices = all_scores.topk(self.num_experts_per_head_topk, dim=-1)
        indices = all_indices.gather(-1, pk_indices)
        if self.separate_embed_per_head:
            head_expert_offsets = torch.arange(self.heads, device=x.device) * self.num_experts
            indices = einx.add('b n h k, h -> b n h k', indices, head_expert_offsets)
        weights_down = self.weight_down_embed(indices)
        weights_up = self.weight_up_embed(indices)
        x = einsum(x, weights_down, 'b n d, b n h k d -> b n h k')
        x = self.activation(x)
        x = self.dropout(x)
        x = x * self.score_activation(scores)
        x = einsum(x, weights_up, 'b n h k, b n h k d -> b n d')
        return x


class PEERLora(Module):
    """
    Same as PEER, except it retrieves LORA weights and adds them to a usual feedforward weight1 and weight2 matrices
    """

    def __init__(self, dim, *, expansion_factor=2.0, num_experts=1000000, heads=4, num_experts_per_head=4, activation=nn.GELU, dim_key=None, product_key_topk=None, pre_rmsnorm=False, non_competing_scores=True, dropout=0.0):
        """
        einops notation
        b - batch
        n - sequence
        d - dimension
        h - heads
        p - 2 for product key
        k - number of keys
        """
        super().__init__()
        dim_inner = int(dim * expansion_factor)
        self.norm = RMSNorm(dim) if pre_rmsnorm else nn.Identity()
        self.heads = heads
        self.num_experts_per_head = num_experts_per_head
        self.num_experts = num_experts
        self.proj_in = nn.Linear(dim, dim_inner, bias=False)
        self.proj_out = nn.Linear(dim_inner, dim, bias=False)
        self.proj_in_lora_a = nn.Embedding(num_experts, dim)
        self.proj_in_lora_b = nn.Embedding(num_experts, dim_inner)
        self.proj_out_lora_a = nn.Embedding(num_experts, dim_inner)
        self.proj_out_lora_b = nn.Embedding(num_experts, dim)
        self.activation = activation()
        assert sqrt(num_experts).is_integer(), '`num_experts` needs to be a square'
        assert dim % 2 == 0, 'feature dimension should be divisible by 2'
        dim_key = default(dim_key, dim // 2)
        self.num_keys = int(sqrt(num_experts))
        self.to_queries = nn.Sequential(nn.Linear(dim, dim_key * heads * 2, bias=False), Rearrange('b n (p h d) -> p b n h d', p=2, h=heads))
        self.product_key_topk = default(product_key_topk, num_experts_per_head)
        self.num_experts_per_head_topk = num_experts_per_head if not non_competing_scores else 1
        self.keys = nn.Parameter(torch.zeros(heads, self.num_keys, 2, dim_key))
        nn.init.normal_(self.keys, std=0.02)
        self.dropout = nn.Dropout(dropout)
        self.score_activation = nn.Softmax(dim=-1) if not non_competing_scores else nn.ReLU()
        nn.init.normal_(self.proj_in_lora_a.weight, std=0.02)
        nn.init.normal_(self.proj_out_lora_b.weight, std=0.02)
        nn.init.normal_(self.proj_in_lora_b.weight, std=0.02)
        nn.init.normal_(self.proj_out_lora_a.weight, std=0.02)

    @property
    def lora_k(self):
        return self.heads * self.num_experts_per_head

    def forward(self, x):
        x = self.norm(x)
        queries = self.to_queries(x)
        sim = einsum(queries, self.keys, 'p b n h d, h k p d -> p b n h k')
        (scores_x, scores_y), (indices_x, indices_y) = sim.topk(self.product_key_topk, dim=-1)
        all_scores = einx.add('... i, ... j -> ... (i j)', scores_x, scores_y)
        all_indices = einx.add('... i, ... j -> ... (i j)', indices_x * self.num_keys, indices_y)
        scores, pk_indices = all_scores.topk(self.num_experts_per_head_topk, dim=-1)
        indices = all_indices.gather(-1, pk_indices)
        proj_in_lora_a = self.proj_in_lora_a(indices)
        proj_in_lora_b = self.proj_in_lora_b(indices)
        proj_out_lora_a = self.proj_out_lora_a(indices)
        proj_out_lora_b = self.proj_out_lora_b(indices)
        hidden = self.proj_in(x)
        lora_in_hidden = einsum(x, proj_in_lora_a, 'b n d, b n h k d -> b n h k')
        lora_in_hidden = lora_in_hidden * self.score_activation(scores)
        lora_in_hidden = einsum(lora_in_hidden, proj_in_lora_b, 'b n h k, b n h k d -> b n d')
        hidden = hidden + lora_in_hidden
        hidden = self.activation(hidden)
        hidden = self.dropout(hidden)
        out = self.proj_out(hidden)
        lora_out_hidden = einsum(hidden, proj_out_lora_a, 'b n d, b n h k d -> b n h k')
        lora_out_hidden = lora_out_hidden * self.score_activation(scores)
        lora_out_hidden = einsum(lora_out_hidden, proj_out_lora_b, 'b n h k, b n h k d -> b n d')
        out = out + lora_out_hidden
        return out


class PK(Module):

    def __init__(self, dim, *, heads=8, dim_key=None, num_keys=1000, product_keys=2, product_key_topk=None, final_topk=16, num_experts_per_head=16):
        """
        einops notation
        b - batch
        n - sequence
        d - dimension
        h - heads
        p - product keys
        k - number of keys
        """
        super().__init__()
        assert dim % 2 == 0
        dim_key = default(dim_key, dim // 2)
        self.to_queries = nn.Sequential(nn.Linear(dim, dim_key * product_keys * heads, bias=False), Rearrange('b n (p h d) -> p b n h d', h=heads, p=product_keys))
        self.num_keys = num_keys
        self.product_keys = product_keys
        self.keys = nn.Parameter(torch.zeros(product_keys, num_keys, heads, dim_key))
        nn.init.normal_(self.keys, std=0.02)
        product_key_topk = default(product_key_topk, final_topk)
        assert final_topk <= product_key_topk ** product_keys
        self.topk = product_key_topk
        self.final_topk = final_topk
        self.max_index = int(num_keys ** product_keys)

    def forward(self, x, softmax_scores=False):
        queries = self.to_queries(x)
        sim = einsum(queries, self.keys, 'p b n h d, p k h d -> p b n h k')
        scores, indices = sim.topk(self.topk, dim=-1)
        strides = self.num_keys ** torch.arange(self.product_keys, device=x.device)
        indices = einx.multiply('p ..., p -> p ...', indices, strides)
        index, *rest_indices = indices
        for rest_index in rest_indices:
            index = einx.add('... i, ... j -> ... (i j)', index, rest_index)
        score, *rest_scores = scores
        for rest_score in rest_scores:
            score = einx.add('... i, ... j -> ... (i j)', score, rest_score)
        final_scores, final_indices = score, index
        final_scores, pk_indices = final_scores.topk(self.final_topk, dim=-1)
        final_indices = final_indices.gather(-1, pk_indices)
        if softmax_scores:
            final_scores = final_scores.softmax(dim=-1)
        return final_scores, final_indices


def pack_one(t, pattern):
    return pack([t], pattern)


def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]


class PKAttention(Module):

    def __init__(self, dim, *, causal=True, heads=8, num_key_values=1000000, key_value_pk_topk=16, dim_key=None, product_keys=2, pre_rmsnorm=False, dropout=0.0):
        """
        einops notation
        b - batch
        n - sequence
        d - dimension
        h - heads
        p - 2 for product key
        k - number of keys
        """
        super().__init__()
        self.causal = causal
        self.heads = heads
        self.num_key_values = num_key_values
        self.norm = RMSNorm(dim) if pre_rmsnorm else nn.Identity()
        self.to_queries = nn.Sequential(nn.Linear(dim, dim * heads, bias=False), Rearrange('b n (h d) -> b n h d', h=heads))
        self.keys = nn.EmbeddingBag(num_key_values * heads, dim, mode='sum')
        self.values = nn.EmbeddingBag(num_key_values * heads, dim, mode='sum')
        assert sqrt(num_key_values).is_integer(), '`num_key_values` needs to be a square'
        assert dim % 2 == 0, 'feature dimension should be divisible by 2'
        self.to_kv_pk_indices = PK(dim=dim, num_keys=int(sqrt(num_key_values)), final_topk=key_value_pk_topk, product_keys=product_keys)
        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Sequential(Rearrange('b h n d -> b n (h d)'), nn.Linear(dim * heads, dim, bias=False))

    def forward(self, x, mask=None):
        device = x.device
        x = self.norm(x)
        q = self.to_queries(x)
        q = q * q.shape[-1] ** -0.5
        kv_scores, indices = self.to_kv_pk_indices(x, softmax_scores=True)
        offsets = torch.arange(self.heads, device=device) * self.num_key_values
        indices = einx.add('b n h k, h -> b n h k', indices, offsets)
        indices, packed_shape = pack_one(indices, '* k')
        kv_scores, _ = pack_one(kv_scores, '* k')
        k, v = self.keys(indices, per_sample_weights=kv_scores), self.values(indices, per_sample_weights=kv_scores)
        k = unpack_one(k, packed_shape, '* d')
        v = unpack_one(v, packed_shape, '* d')
        sim = einsum(q, k, 'b i h d, b j h d -> b h i j')
        if self.causal:
            assert not exists(mask)
            i, j, device = *sim.shape[-2:], x.device
            causal_mask = torch.ones((i, j), device=device, dtype=torch.bool).triu(j - i + 1)
            sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)
        elif exists(mask):
            sim = einx.where('b j, b h i j, -> b h i j', mask, sim, -torch.finfo(sim.dtype).max)
        attn = sim.softmax(dim=-1)
        attn = self.dropout(attn)
        out = einsum(attn, v, 'b h i j, b j h d -> b h i d')
        return self.to_out(out)


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (RMSNorm,
     lambda: ([], {'dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
]

