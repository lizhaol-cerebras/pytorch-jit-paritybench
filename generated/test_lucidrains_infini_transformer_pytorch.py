
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


from typing import Tuple


from typing import List


from typing import NamedTuple


import torch


from torch import nn


from torch import Tensor


import torch.nn.functional as F


from torch.nn import Module


from torch.nn import ModuleList


from math import ceil


from typing import Callable


import numpy as np


from torch.optim import Adam


from torch.utils.data import DataLoader


from torch.utils.data import Dataset


class RMSNorm(Module):

    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.normalize(x, dim=-1) * self.scale * self.gamma


class FeedForward(Module):

    def __init__(self, dim, mult=4, dropout=0.0):
        super().__init__()
        dim_inner = int(mult * dim * 2 / 3)
        self.norm = RMSNorm(dim)
        self.proj_in = nn.Linear(dim, dim_inner * 2)
        self.proj_out = nn.Linear(dim_inner, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.norm(x)
        x, gates = self.proj_in(x).chunk(2, dim=-1)
        x = F.gelu(gates) * x
        x = self.dropout(x)
        return self.proj_out(x)


class Memories(NamedTuple):
    kv_mem: 'Tensor'
    k_norm: 'Tensor'


def exists(v):
    return v is not None


def retrieve_from_kv_memories(t, past_memories: 'Memories', eps=1e-10):
    past_memories_kv, past_memories_norm = past_memories
    numer = einsum(t, past_memories_kv, 'b h n dk, b h dk dv -> b h n dv')
    denom = einsum(t, past_memories_norm, 'b h n d, b h d -> b h n')
    denom = rearrange(denom, '... -> ... 1')
    return numer / denom.clamp(min=eps)


class FastweightMemory(Module):

    def __init__(self, heads: 'int', head_gate_init_value=10.0, use_mem_delta_rule=False):
        super().__init__()
        self.use_mem_delta_rule = use_mem_delta_rule
        self.head_gates = nn.Parameter(torch.ones(heads) * head_gate_init_value)

    def create_new_memories(self, keys: 'Tensor', values: 'Tensor', past_memories: 'Memories | None'=None) ->Memories:
        keys = F.elu(keys) + 1
        if exists(past_memories) and self.use_mem_delta_rule:
            delta_v = retrieve_from_kv_memories(keys, past_memories)
            values = values - delta_v
        new_memories_kv = einsum(keys, values, '... n dk, ... n dv -> ... dk dv')
        new_memories_norm = reduce(keys, 'b h n d -> b h d', 'sum')
        if exists(past_memories):
            past_memories_kv, past_memories_norm = past_memories
            new_memories_kv = new_memories_kv + past_memories_kv
            new_memories_norm = new_memories_norm + past_memories_norm
        return Memories(new_memories_kv, new_memories_norm)

    def retrieve_and_add_to_output(self, out: 'Tensor', queries: 'Tensor', past_memories: 'Memories | None'=None) ->Tensor:
        if not exists(past_memories):
            return out
        queries = F.elu(queries) + 1
        mem_out = retrieve_from_kv_memories(queries, past_memories)
        gates = rearrange(self.head_gates, 'h -> h 1 1')
        gates = gates.sigmoid()
        out = out * gates + mem_out * (1.0 - gates)
        return out


class CausalAttention(Module):

    def __init__(self, dim, *, dim_head=128, heads=8, dropout=0.0, head_gate_init_value=10.0, use_mem_delta_rule=False):
        super().__init__()
        dim_inner = dim_head * heads
        self.scale = dim_head ** -0.5
        self.norm = RMSNorm(dim)
        self.rotary_emb = RotaryEmbedding(dim_head)
        self.to_qkv = nn.Linear(dim, dim_inner * 3, bias=False)
        self.to_out = nn.Linear(dim_inner, dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.split_heads = Rearrange('b n (qkv h d) -> qkv b h n d', qkv=3, h=heads)
        self.merge_heads = Rearrange('b h n d -> b n (h d)')
        self.fastweight_mem = FastweightMemory(heads=heads, head_gate_init_value=head_gate_init_value, use_mem_delta_rule=use_mem_delta_rule)

    def forward(self, x, cached_kv: 'Tensor | None'=None, past_memories: 'Memories | None'=None, return_new_memories=False, eps=1e-10) ->Tuple[Tensor, Tensor, Memories]:
        """
        ein notation:

        b - batch
        h - heads
        n - sequence
        i - source sequence (q)
        j - target sequence (kv)
        d - feature dimension
        dk - feature dimension keys (and queries)
        dv - feature dimension of values
        """
        x = self.norm(x)
        x = self.to_qkv(x)
        q, k, v = self.split_heads(x)
        if exists(cached_kv):
            cached_k, cached_v = cached_kv
            k = torch.cat((cached_k, k), dim=-2)
            v = torch.cat((cached_v, v), dim=-2)
        q_scaled = q * self.scale
        q_rotated, k_rotated = self.rotary_emb.rotate_queries_with_cached_keys(q_scaled, k)
        sim = einsum(q_rotated, k_rotated, '... i d, ... j d -> ... i j')
        i, j = sim.shape[-2:]
        causal_mask = torch.ones((i, j), device=sim.device, dtype=torch.bool).triu(j - i + 1)
        sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)
        attn = sim.softmax(dim=-1)
        attn = self.dropout(attn)
        out = einsum(attn, v, '... i j, ... j d -> ... i d')
        out = self.fastweight_mem.retrieve_and_add_to_output(out, q, past_memories)
        out = self.merge_heads(out)
        out = self.to_out(out)
        if not return_new_memories:
            cached_kv = torch.stack((k, v))
            return out, cached_kv, past_memories
        new_memories = self.fastweight_mem.create_new_memories(k, v, past_memories)
        return out, None, new_memories


class TransformerReturn(NamedTuple):
    logits: 'Tensor'
    cached_kvs: 'List[Tensor] | None'
    past_memories: 'List[Memories] | None'


def default(v, d):
    return v if exists(v) else d


def detach_cached_kv_(cached_kvs: 'List[Tensor]'):
    for cached_kv in cached_kvs:
        cached_kv.detach_()


def detach_memories_(memories: 'List[Memories]'):
    for mem_kv, mem_norm in memories:
        mem_kv.detach_()
        mem_norm.detach_()


class InfiniTransformer(Module):

    def __init__(self, *, num_tokens, dim, depth, dim_head=128, heads=8, attn_dropout=0.0, ff_mult=4, ff_dropout=0.0, use_mem_delta_rule=False):
        super().__init__()
        self.token_emb = nn.Embedding(num_tokens, dim)
        self.layers = ModuleList([])
        for _ in range(depth):
            attn = CausalAttention(dim=dim, dim_head=dim_head, heads=heads, use_mem_delta_rule=use_mem_delta_rule, dropout=attn_dropout)
            ff = FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout)
            self.layers.append(ModuleList([attn, ff]))
        self.norm = RMSNorm(dim)
        self.to_logits = nn.Linear(dim, num_tokens)

    def forward(self, x, past_memories: 'List[Memories] | None'=None, cached_kv: 'List[Tensor] | None'=None, return_new_memories=False, detach_memories=False) ->TransformerReturn:
        x = self.token_emb(x)
        if exists(cached_kv):
            x = x[:, -1:]
        new_cached_kv = []
        cached_kv_iter = iter(default(cached_kv, []))
        new_memories = []
        past_memories_iter = iter(default(past_memories, []))
        for attn, ff in self.layers:
            attn_out, layer_cached_kv, layer_new_memories = attn(x, cached_kv=next(cached_kv_iter, None), past_memories=next(past_memories_iter, None), return_new_memories=return_new_memories)
            x = attn_out + x
            x = ff(x) + x
            new_cached_kv.append(layer_cached_kv)
            new_memories.append(layer_new_memories)
        embed = self.norm(x)
        logits = self.to_logits(embed)
        if detach_memories:
            detach_cached_kv_(new_cached_kv)
        if not return_new_memories:
            return TransformerReturn(logits, new_cached_kv, past_memories)
        if detach_memories:
            detach_memories_(new_memories)
        return TransformerReturn(logits, None, new_memories)


def divisible_by(num, den):
    return num % den == 0


def log(t, eps=1e-20):
    return torch.log(t.clamp(min=eps))


def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))


def gumbel_sample(t, temperature=1.0, dim=-1, keepdim=True, eps=1e-10):
    return (t / max(temperature, eps) + gumbel_noise(t)).argmax(dim=dim, keepdim=keepdim)


def round_down_multiple(n, mult):
    return n // mult * mult


def top_p(logits, thres=0.9):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cum_probs > thres
    sorted_indices_to_remove = F.pad(sorted_indices_to_remove, (1, -1), value=False)
    sorted_logits[sorted_indices_to_remove] = float('-inf')
    return sorted_logits.scatter(1, sorted_indices, sorted_logits)


class InfiniTransformerWrapper(Module):

    def __init__(self, model: 'InfiniTransformer', segment_length=512, detach_mems_every_num_segments=2, ignore_index=-1):
        super().__init__()
        self.model = model
        self.segment_length = segment_length
        self.detach_mems_every_num_segments = detach_mems_every_num_segments
        self.ignore_index = ignore_index

    @property
    def device(self):
        return next(self.model.parameters()).device

    @torch.no_grad()
    def generate(self, *, seq_len, prompt=None, batch_size=1, temperature=1.0, filter_fn: Callable=top_p, filter_kwargs: dict=dict(thres=0.9), exclude_prompt=True, segment_length=None):
        segment_length = default(segment_length, self.segment_length)
        device, train_state = self.device, self.training
        self.eval()
        out = default(prompt, torch.empty((batch_size, 0), device=device, dtype=torch.long))
        init_len = out.shape[-1]
        cached_kv = None
        past_memories = None
        for curr_len in tqdm(range(init_len, seq_len)):
            start_ind = round_down_multiple(curr_len - 1, segment_length)
            model_input = out[:, start_ind:]
            logits, cached_kv, past_memories = self.model(model_input, cached_kv=cached_kv, past_memories=past_memories, return_new_memories=divisible_by(curr_len, segment_length))
            logits = logits[:, -1]
            filtered_logits = filter_fn(logits, **filter_kwargs)
            sampled = gumbel_sample(filtered_logits, temperature=temperature)
            out, _ = pack((out, sampled), 'b *')
        if exclude_prompt:
            out = out[:, init_len:]
        self.train(train_state)
        return out

    def forward(self, seq, segment_length=None, backward=False, grad_accum_scale=1.0):
        segment_length = default(segment_length, self.segment_length)
        seq, label = seq[:, :-1], seq[:, 1:]
        if backward:
            self.model.train()
        total_tokens = (label != self.ignore_index).sum().item()
        split_seq = seq.split(segment_length, dim=-1)
        split_label = label.split(segment_length, dim=-1)
        num_segments = len(split_seq)
        total_loss = 0.0
        past_memories = None
        running_loss = 0.0
        for ind, (segment_seq, segment_label) in enumerate(zip(split_seq, split_label)):
            segment_num = ind + 1
            is_last = segment_num == num_segments
            should_detach_memories = divisible_by(segment_num, self.detach_mems_every_num_segments)
            should_backward = backward and (is_last or should_detach_memories)
            logits, _, past_memories = self.model(segment_seq, past_memories=past_memories, return_new_memories=True)
            segment_loss = F.cross_entropy(rearrange(logits, 'b n c -> b c n'), segment_label, reduction='none')
            segment_mask = segment_label != self.ignore_index
            num_segment_tokens = segment_mask.sum()
            frac_tokens = num_segment_tokens / total_tokens
            segment_loss = segment_loss[segment_mask]
            segment_scaled_loss = segment_loss.mean() * frac_tokens
            total_loss = total_loss + segment_scaled_loss
            running_loss = running_loss + segment_scaled_loss
            if should_backward:
                (running_loss / grad_accum_scale).backward()
                running_loss = 0.0
            if should_detach_memories and not is_last:
                detach_memories_(past_memories)
        return total_loss


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (FeedForward,
     lambda: ([], {'dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (RMSNorm,
     lambda: ([], {'dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
]

