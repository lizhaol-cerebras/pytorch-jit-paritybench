
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


from torch.optim import AdamW


from torch.optim import Adam


import torch


from torch import nn


from torch import einsum


import re


from functools import partial


from functools import wraps


from collections import namedtuple


import torch.nn.functional as F


from torch.utils.data import Dataset


from torch.utils.data import DataLoader


from torch.nn.utils.rnn import pad_sequence


class RMSNorm(nn.Module):

    def __init__(self, dim, eps=1e-08):
        super().__init__()
        self.scale = dim ** -0.5
        self.eps = eps
        self.g = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = torch.norm(x, dim=-1, keepdim=True) * self.scale
        return x / norm.clamp(min=self.eps) * self.g


class RotaryEmbedding(nn.Module):

    def __init__(self, dim):
        super().__init__()
        inv_freq = 1.0 / 10000 ** (torch.arange(0, dim, 2).float() / dim)
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, max_seq_len, *, device):
        seq = torch.arange(max_seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = einsum('i , j -> i j', seq, self.inv_freq)
        return torch.cat((freqs, freqs), dim=-1)


def rotate_half(x):
    x = rearrange(x, '... (j d) -> ... j d', j=2)
    x1, x2 = x.unbind(dim=-2)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(pos, t):
    return t * pos.cos() + rotate_half(t) * pos.sin()


class ParallelTransformerBlock(nn.Module):

    def __init__(self, dim, dim_head=64, heads=8, ff_mult=4):
        super().__init__()
        self.norm = RMSNorm(dim)
        attn_inner_dim = dim_head * heads
        ff_inner_dim = dim * ff_mult
        self.fused_dims = attn_inner_dim, dim_head, dim_head, ff_inner_dim
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.rotary_emb = RotaryEmbedding(dim_head)
        self.fused_attn_ff_proj = nn.Linear(dim, sum(self.fused_dims), bias=False)
        self.attn_out = nn.Linear(attn_inner_dim, dim, bias=False)
        self.ff_out = nn.Sequential(nn.GELU(), nn.Linear(ff_inner_dim, dim, bias=False))
        self.register_buffer('mask', None, persistent=False)
        self.register_buffer('pos_emb', None, persistent=False)

    def get_mask(self, n, device):
        if self.mask is not None and self.mask.shape[-1] >= n:
            return self.mask[:n, :n]
        mask = torch.ones((n, n), device=device, dtype=torch.bool).triu(1)
        self.register_buffer('mask', mask, persistent=False)
        return mask

    def get_rotary_embedding(self, n, device):
        if self.pos_emb is not None and self.pos_emb.shape[-2] >= n:
            return self.pos_emb[:n]
        pos_emb = self.rotary_emb(n, device=device)
        self.register_buffer('pos_emb', pos_emb, persistent=False)
        return pos_emb

    def forward(self, x):
        """
        einstein notation
        b - batch
        h - heads
        n, i, j - sequence length (base sequence length, source, target)
        d - feature dimension
        """
        n, device, h = x.shape[1], x.device, self.heads
        x = self.norm(x)
        q, k, v, ff = self.fused_attn_ff_proj(x).split(self.fused_dims, dim=-1)
        q = rearrange(q, 'b n (h d) -> b h n d', h=h)
        positions = self.get_rotary_embedding(n, device)
        q, k = map(lambda t: apply_rotary_pos_emb(positions, t), (q, k))
        q = q * self.scale
        sim = einsum('b h i d, b j d -> b h i j', q, k)
        causal_mask = self.get_mask(n, device)
        sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)
        attn = sim.softmax(dim=-1)
        out = einsum('b h i j, b j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.attn_out(out) + self.ff_out(ff)


class Transformer(nn.Module):

    def __init__(self, dim, depth, heads, dim_head, ff_mult=4):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(ParallelTransformerBlock(dim, dim_head, heads, ff_mult))

    def forward(self, x):
        for block in self.layers:
            x = block(x) + x
        return x


DEFAULT_PROMPT_INPUT_TAG = '[input]'


FilteredResults = namedtuple('FilteredResults', ['num_passed', 'num_failed', 'selected_indices', 'selected_mask', 'filtered_tokens', 'filtered_tokens_without_api_response', 'filtered_tokens_with_api_response'])


def FinetuneDataloader(ds: 'Dataset', *args, padding_value=0, **kwargs):
    return DataLoader(ds, *args, collate_fn=partial(pad_sequence, padding_value=padding_value), **kwargs)


class FinetuneDataset(Dataset):

    def __init__(self, tokens: 'torch.Tensor'):
        self.tokens = tokens

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        return self.tokens[idx]


def prompt_collate_fn(data, padding_value=0):
    prompts, prompt_lengths = zip(*data)
    prompts = pad_sequence(prompts, padding_value=padding_value)
    return prompts, torch.stack(prompt_lengths)


def PromptDataloader(ds: 'Dataset', *args, padding_value=0, **kwargs):
    collate_fn = partial(prompt_collate_fn, padding_value=padding_value)
    return DataLoader(ds, *args, collate_fn=collate_fn, **kwargs)


def exists(val):
    return val is not None


def all_contains_id(t: 'torch.Tensor', token_id: 'int'):
    mask = t == token_id
    return mask.any(dim=-1).all()


def default_weight_fn(t):
    return (1.0 - t * 0.2).clamp(min=0.0)


def get_pred_prob(token_ids, logits):
    logits = logits[:, :-1]
    token_ids = token_ids[:, 1:]
    token_ids = rearrange(token_ids, 'b n -> b n 1')
    probs = logits.softmax(dim=-1)
    correct_token_id_pred_prob = probs.gather(-1, token_ids)
    return rearrange(correct_token_id_pred_prob, 'b n 1 -> b n')


def log(t, eps=1e-20):
    return t.clamp(min=eps).log()


def get_arange_start_at_token_id(token_ids: 'torch.Tensor', token_id: 'int', pad_id=-1):
    is_token_id_mask = token_ids == token_id
    arange = (is_token_id_mask.cumsum(dim=-1) > 0).cumsum(dim=-1)
    before_token_mask = arange == 0
    arange = arange - 1
    arange = arange.masked_fill(before_token_mask, pad_id)
    return arange


def weight_and_mask(token_ids: 'torch.Tensor', token_id: 'int', pad_id=-1, weighting_fn: 'Callable'=default_weight_fn):
    t = get_arange_start_at_token_id(token_ids, token_id, pad_id)
    weights = weighting_fn(t)
    return weights.masked_fill(t == pad_id, 0.0)


def separate_weight_decayable_params(params):
    wd_params, no_wd_params = [], []
    for param in params:
        param_list = no_wd_params if param.ndim < 2 else wd_params
        param_list.append(param)
    return wd_params, no_wd_params


def get_optimizer(params, lr=0.0001, wd=0.01, betas=(0.9, 0.99), eps=1e-08, filter_by_requires_grad=False, group_wd_params=True, **kwargs):
    has_weight_decay = wd > 0
    if filter_by_requires_grad:
        params = list(filter(lambda t: t.requires_grad, params))
    if group_wd_params and has_weight_decay:
        wd_params, no_wd_params = separate_weight_decayable_params(params)
        params = [{'params': wd_params}, {'params': no_wd_params, 'weight_decay': 0}]
    adam_kwargs = dict(lr=lr, betas=betas, eps=eps)
    if not has_weight_decay:
        return Adam(params, **adam_kwargs)
    return AdamW(params, weight_decay=wd, **adam_kwargs)


def create_function_regex(api_start=' [', api_stop=']'):
    api_start_regex, api_stop_regex = map(re.escape, (api_start, api_stop))
    return f'({api_start_regex}(\\w+)\\(([^)]*)\\))({api_stop_regex})'


def has_api_calls(text, api_start=' [', api_stop=']'):
    regex = create_function_regex(api_start, api_stop)
    matches = re.findall(regex, text)
    return len(matches) > 0


def identity(t):
    return t


def always(val):

    def inner(*args, **kwargs):
        return val
    return inner


def is_valid_float(s):
    return exists(re.fullmatch('[+-]?\\d+(\\.\\d+)?', s))


def is_valid_integer(s):
    return exists(re.fullmatch('[+-]?\\d+', s))


def is_valid_string(s):
    return exists(re.fullmatch('\'[^\']*\'|\\"[^\\"]*\\"', s))


def try_except(fn, callback=identity):

    @wraps(fn)
    def inner(*args):
        try:
            return fn(*args)
        except Exception as e:
            return callback(e)
    return inner


def invoke_tools(registry: 'dict[str, Callable]', text: 'str', delimiter: 'str'='→', api_start=' [', api_stop=' ]') ->str:
    regex = create_function_regex(api_start, api_stop)
    replace_ = partial(replace_fn, registry, delimiter=delimiter)
    return re.sub(regex, replace_, text)


def num_matches(substr: 'str', text: 'str'):
    return len(re.findall(re.escape(substr), text))


def replace_all_but_first(text: 'str', api_start=' [', api_stop=']') ->str:
    regex = create_function_regex(api_start, api_stop)
    count = 0

    def replace_(matches):
        orig_text = matches.group(0)
        nonlocal count
        count += 1
        if count > 1:
            return ''
        return orig_text
    return re.sub(regex, replace_, text)


def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))


def gumbel_sample(t, temperature=1.0, dim=-1, eps=1e-10):
    if temperature == 0:
        return t.argmax(dim=dim)
    return (t / max(temperature, eps) + gumbel_noise(t)).argmax(dim=dim)


def find_indices_of(t: 'torch.Tensor', token_id: 'int', occurrence=1):
    assert occurrence > 0
    mask = t == token_id
    has_occurred = mask.cumsum(dim=-1)
    has_occurred = F.pad(has_occurred, (1, 0), value=0.0)
    return (has_occurred < occurrence).sum(dim=-1).long()


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (RMSNorm,
     lambda: ([], {'dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
]

