
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


import math


from torchvision import transforms


from torch.nn import functional as F


from torch import nn


from torch import optim


import torchvision.transforms as transformsVision


import matplotlib.pyplot as plt


import pandas as pd


import torch as t


from torch.utils.data.distributed import DistributedSampler


from torch.utils.data import DataLoader


from torch.utils.data import Dataset


from torch.utils.data import BatchSampler


from torch.utils.data import RandomSampler


import torch.nn as nn


import torch.nn.functional as F


from enum import Enum


import time


import warnings


from torch.nn.parallel import DistributedDataParallel


import functools


import torch.distributed as dist


from time import sleep


from torch._utils import _flatten_dense_tensors


from torch.optim import Optimizer


def get_normal(*shape, std=0.01):
    w = t.empty(shape)
    nn.init.normal_(w, std=std)
    return w


class PositionEmbedding(nn.Module):

    def __init__(self, input_shape, width, init_scale=1.0, pos_init=False):
        super().__init__()
        self.input_shape = input_shape
        self.input_dims = input_dims = np.prod(input_shape)
        self.pos_init = pos_init
        if pos_init:
            self.register_buffer('pos', t.tensor(get_pos_idx(input_shape)).long())
            self._pos_embs = nn.ModuleList()
            for i in range(len(input_shape)):
                emb = nn.Embedding(input_shape[i], width)
                nn.init.normal_(emb.weight, std=0.02)
                self._pos_embs.append(emb)
        else:
            self.pos_emb = nn.Parameter(get_normal(input_dims, width, std=0.01 * init_scale))

    def forward(self):
        if self.pos_init:
            pos_emb = sum([self._pos_embs[i](self.pos[:, i]) for i in range(len(self.input_shape))])
        else:
            pos_emb = self.pos_emb
        return pos_emb


class Conv1D(nn.Module):

    def __init__(self, n_in, n_out, zero_out=False, init_scale=1.0):
        super(Conv1D, self).__init__()
        self.n_in = n_in
        self.n_out = n_out
        if zero_out:
            w = t.zeros(n_in, n_out)
        else:
            w = t.empty(n_in, n_out)
            nn.init.normal_(w, std=0.02 * init_scale)
        b = t.zeros(n_out)
        self.w = nn.Parameter(w)
        self.b = nn.Parameter(b)

    def forward(self, x):
        size_out = *x.size()[:-1], self.n_out
        x = t.addmm(self.b.type_as(x), x.view(-1, x.size(-1)), self.w.type_as(x))
        x = x.view(*size_out)
        return x


class CheckpointFunction(t.autograd.Function):

    @staticmethod
    def forward(ctx, run_function, length, *args):
        ctx.run_function = run_function
        ctx.input_tensors = list(args[:length])
        ctx.input_params = list(args[length:])
        with t.no_grad():
            output_tensors = ctx.run_function(*ctx.input_tensors)
        return output_tensors

    @staticmethod
    def backward(ctx, *output_grads):
        for i in range(len(ctx.input_tensors)):
            temp = ctx.input_tensors[i]
            ctx.input_tensors[i] = temp.detach()
            ctx.input_tensors[i].requires_grad = temp.requires_grad
        with t.enable_grad():
            output_tensors = ctx.run_function(*ctx.input_tensors)
        input_grads = t.autograd.grad(output_tensors, ctx.input_tensors + ctx.input_params, output_grads, allow_unused=True)
        del ctx.input_tensors
        del output_tensors
        return (None, None) + input_grads


def checkpoint(func, inputs, params, flag):
    if flag:
        args = inputs + tuple(params)
        return CheckpointFunction.apply(func, len(inputs), *args)
    else:
        return func(*inputs)


def get_mask(mask, q_l, kv_l, blocks, spread, device, sample, sample_t):
    if mask is None or q_l == 1:
        return None
    offset = sample_t - q_l if sample else max(kv_l - q_l, 0)
    if mask == 'autoregressive':
        mask = t.ones(q_l, kv_l, device=device).tril(offset)
    elif mask == 'summary':
        mask = t.nn.functional.pad(t.ones(q_l, q_l, device=device).tril().view(q_l, blocks, q_l // blocks)[:, :-1, -kv_l // blocks:], (0, 0, 1, 0), value=1).contiguous().view(q_l, kv_l)
    elif mask == 'prime':
        mask = t.ones(q_l, kv_l, device=device).tril(offset)
    return mask.view(1, 1, q_l, kv_l)


class FactoredAttention(nn.Module):

    def __init__(self, n_in, n_ctx, n_state, n_head, attn_dropout=0.0, resid_dropout=0.0, scale=True, mask=False, zero_out=False, init_scale=1.0, checkpoint_attn=0, attn_func=0, blocks=None, spread=None, encoder_dims=None, prime_len=None):
        super().__init__()
        self.n_in = n_in
        self.n_ctx = n_ctx
        self.n_state = n_state
        assert n_state % n_head == 0
        self.n_head = n_head
        self.scale = scale
        self.mask = mask
        if attn_func == 6:
            self.c_attn = Conv1D(n_in, n_state, init_scale=init_scale)
            self.c_enc_kv = Conv1D(n_in, n_state * 2, init_scale=init_scale)
        else:
            self.c_attn = Conv1D(n_in, n_state * 3, init_scale=init_scale)
        self.c_proj = Conv1D(n_state, n_in, zero_out, init_scale=init_scale)
        self.attn_dropout = nn.Dropout(attn_dropout) if attn_dropout > 0.0 else lambda x: x
        self.resid_dropout = nn.Dropout(resid_dropout) if resid_dropout > 0.0 else lambda x: x
        self.attn_func = attn_func
        self.qkv, self.attn, self.attn_mask = {(0): (self.factored_qkv, self.dense_attn, 'autoregressive'), (1): (self.factored_qkv, self.block_attn, 'autoregressive'), (2): (self.factored_qkv, self.transpose_block_attn, 'autoregressive'), (3): (self.factored_qkv, self.prev_block_attn, None), (4): (self.factored_qkv, self.summary_attn, 'summary'), (5): (self.factored_qkv, self.summary_spread_attn, 'summary'), (6): (self.decode_qkv, self.decode_attn, None), (7): (self.prime_qkv, self.prime_attn, 'prime')}[attn_func]
        self.blocks = blocks
        self.spread = spread
        if blocks is not None:
            assert n_ctx % blocks == 0
            self.block_ctx = n_ctx // blocks
        self.checkpoint_attn = checkpoint_attn
        self.sample_t = 0
        self.cache = {}
        self.encoder_dims = encoder_dims
        self.prime_len = prime_len
        self.record_attn = False
        self.w = None

    def _attn(self, q, k, v, sample):
        scale = 1.0 / math.sqrt(math.sqrt(self.n_state // self.n_head))
        if self.training:
            w = t.matmul(q * scale, k * scale)
        else:
            w = t.matmul(q, k)
            w.mul_(scale * scale)
        wtype = w.dtype
        w = w.float()
        if self.mask:
            mask = get_mask(self.attn_mask, q.size(-2), k.size(-1), self.blocks, self.spread, w.device, sample, self.sample_t)
            if mask is not None:
                w = w * mask + -1000000000.0 * (1 - mask)
            w = F.softmax(w, dim=-1).type(wtype)
        else:
            w = F.softmax(w, dim=-1).type(wtype)
        if self.record_attn:
            self.w = w
            if self.attn_func == 7:
                self.w = self.w[:, :, self.prime_len:, :self.prime_len]
        w = self.attn_dropout(w)
        a = t.matmul(w, v)
        return a

    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = *x.size()[:-2], x.size(-2) * x.size(-1)
        return x.view(*new_x_shape)

    def split_heads(self, x, k=False):
        new_x_shape = *x.size()[:-1], self.n_head, x.size(-1) // self.n_head
        x = x.view(*new_x_shape)
        if k:
            return x.permute(0, 2, 3, 1)
        else:
            return x.permute(0, 2, 1, 3)

    def dense_attn(self, query, key, value, sample):
        query = self.split_heads(query)
        key = self.split_heads(key, k=True)
        value = self.split_heads(value)
        if self.checkpoint_attn == 1 and not sample:
            a = checkpoint(lambda q, k, v, s=sample: self._attn(q, k, v, s), (query, key, value), (), True)
        else:
            a = self._attn(query, key, value, sample)
        a = self.merge_heads(a)
        return a

    def block_attn(self, q, k, v, sample):
        blocks, block_ctx = self.blocks, self.block_ctx
        bs, l, d = v.shape
        if sample:
            assert l == self._suff_cache_len(), f'{l} != {self._suff_cache_len()}'
            return self.dense_attn(q, k, v, sample).view(bs, 1, d)
        else:
            ql = q.shape[1]
            q = q.view(bs * ql // block_ctx, block_ctx, d)
            if ql < l:
                l = ql
                k = k[:, -l:].contiguous()
                v = v[:, -l:].contiguous()
            k = k.view(bs * l // block_ctx, block_ctx, d)
            v = v.view(bs * l // block_ctx, block_ctx, d)
            return self.dense_attn(q, k, v, sample).view(bs, l, d)

    def transpose_block_attn(self, q, k, v, sample):
        blocks, block_ctx = self.blocks, self.block_ctx
        bs, l, d = v.shape
        if sample:
            block_l = (l - 1) % block_ctx
            k = k[:, block_l::block_ctx, :]
            v = v[:, block_l::block_ctx, :]
            return self.dense_attn(q, k, v, sample).view(bs, 1, d)
        else:
            ql = q.shape[1]
            q = q.view(bs, ql // block_ctx, block_ctx, d).transpose(1, 2).contiguous().view(bs * block_ctx, ql // block_ctx, d)
            k = k.view(bs, l // block_ctx, block_ctx, d).transpose(1, 2).contiguous().view(bs * block_ctx, l // block_ctx, d)
            v = v.view(bs, l // block_ctx, block_ctx, d).transpose(1, 2).contiguous().view(bs * block_ctx, l // block_ctx, d)
            return self.dense_attn(q, k, v, sample).view(bs, block_ctx, ql // block_ctx, d).transpose(1, 2).contiguous().view(bs, ql, d)

    def prev_block_attn(self, q, k, v, sample):
        blocks, block_ctx = self.blocks, self.block_ctx
        bs, l, d = v.shape
        if sample:
            assert l == self._suff_cache_len(), f'{l} != {self._suff_cache_len()}'
            block = (l - 1) // block_ctx
            prev_l = (block - 1) * block_ctx
            if block > 0:
                assert prev_l == 0
                k = k[:, prev_l:prev_l + block_ctx, :]
                v = v[:, prev_l:prev_l + block_ctx, :]
            else:
                k = t.zeros(bs, block_ctx, d, device=q.device, dtype=q.dtype)
                v = t.zeros(bs, block_ctx, d, device=q.device, dtype=q.dtype)
            return self.dense_attn(q, k, v, sample).view(bs, 1, d)
        else:
            ql = q.shape[1]
            q = q.view(bs * ql // block_ctx, block_ctx, d)
            k = t.nn.functional.pad(k.view(bs, l // block_ctx, block_ctx, d)[:, :-1, :, :], (0, 0, 0, 0, 1, 0)).view(bs * l // block_ctx, block_ctx, d)
            v = t.nn.functional.pad(v.view(bs, l // block_ctx, block_ctx, d)[:, :-1, :, :], (0, 0, 0, 0, 1, 0)).view(bs * l // block_ctx, block_ctx, d)
            if ql < l:
                qb = ql // block_ctx
                kb = l // block_ctx
                l = ql
                k = k.view(bs, kb, block_ctx, d)[:, -qb:].contiguous().view(bs * qb, block_ctx, d)
                v = v.view(bs, kb, block_ctx, d)[:, -qb:].contiguous().view(bs * qb, block_ctx, d)
            return self.dense_attn(q, k, v, sample).view(bs, l, d)

    def summary_attn(self, q, k, v, sample):
        blocks, block_ctx = self.blocks, self.block_ctx
        bs, l, d = v.shape
        if sample:
            k = t.nn.functional.pad(k[:, block_ctx - 1:blocks * block_ctx - 1:block_ctx, :], (0, 0, 1, 0))
            v = t.nn.functional.pad(v[:, block_ctx - 1:blocks * block_ctx - 1:block_ctx, :], (0, 0, 1, 0))
            return self.dense_attn(q, k, v, sample).view(bs, 1, d)
        else:
            k = t.nn.functional.pad(k.view(bs, blocks, l // blocks, d)[:, :-1, -1, :], (0, 0, 1, 0))
            v = t.nn.functional.pad(v.view(bs, blocks, l // blocks, d)[:, :-1, -1, :], (0, 0, 1, 0))
            return self.dense_attn(q, k, v, sample).view(bs, l, d)

    def summary_spread_attn(self, q, k, v, sample):
        blocks, block_ctx, spread = self.blocks, self.block_ctx, self.spread
        bs, l, d = v.shape
        if sample:
            assert False, 'Not yet implemented'
        else:
            k = t.nn.functional.pad(k.view(bs, blocks, l // blocks, d)[:, :-1, -spread:, :], (0, 0, 0, 0, 1, 0)).contiguous().view(bs, blocks * spread, d)
            v = t.nn.functional.pad(v.view(bs, blocks, l // blocks, d)[:, :-1, -spread:, :], (0, 0, 0, 0, 1, 0)).contiguous().view(bs, blocks * spread, d)
            return self.dense_attn(q, k, v, sample).view(bs, l, d)

    def prime_attn(self, q, k, v, sample):
        prime_len = self._prime_len
        k = k[:, :prime_len]
        v = v[:, :prime_len]
        return self.dense_attn(q, k, v, sample)

    def decode_attn(self, q, k, v, sample):
        assert k.shape[1] == v.shape[1] == self.encoder_dims, f'k: {k.shape}, v: {v.shape}, enc_dims: {self.encoder_dims}'
        return self.dense_attn(q, k, v, sample)

    def factored_qkv(self, x, encoder_kv=None, sample=False):
        curr_ctx = x.shape[1]
        assert encoder_kv is None
        query, key, value = x.chunk(3, dim=2)
        if sample:
            self.sample_t += curr_ctx
            key, value = self._append_cache(key, value)
            l_cache = self._suff_cache_len()
            if self._cache_len() > l_cache:
                self._slice_cache(-l_cache)
            if curr_ctx > 1:
                if self.attn_func != 0:
                    query = self._pad_to_block_ctx(query, query=True)
                    key = self._pad_to_block_ctx(key)
                    value = self._pad_to_block_ctx(value)
                    assert key.shape[1] % self.block_ctx == 0
                    assert query.shape[1] % self.block_ctx == 0
                assert key.shape[1] == value.shape[1]
                assert query.shape[1] <= key.shape[1]
                sample = False
            else:
                key = self.cache['key']
                value = self.cache['value']
        return query, key, value, sample

    def prime_qkv(self, x, encoder_kv=None, sample=False):
        curr_ctx = x.shape[1]
        assert encoder_kv is None
        query, key, value = x.chunk(3, dim=2)
        if sample:
            if self._cache_len() < self._prime_len:
                self._append_cache(key, value)
            if self._cache_len() > self._prime_len:
                self._slice_cache(0, self._prime_len)
            key, value = self.cache['key'], self.cache['value']
            self.sample_t += curr_ctx
            assert key.shape[1] == value.shape[1] == self._suff_cache_len(), f'k: {key.shape}, v: {value.shape}, prime_dims: {self._suff_cache_len()}'
        else:
            assert key.shape[1] == value.shape[1] == self.n_ctx, f'k: {key.shape}, v: {value.shape}, prime_dims: {self.n_ctx}'
        assert key.shape[0] == value.shape[0] == query.shape[0], f'k: {key.shape}, v: {value.shape}, q: {query.shape}'
        assert key.shape[2] == value.shape[2] == query.shape[2], f'k: {key.shape}, v: {value.shape}, q: {query.shape}'
        return query, key, value, sample

    def decode_qkv(self, x, encoder_kv=None, sample=False):
        curr_ctx = x.shape[1]
        assert encoder_kv is not None
        query = x
        if sample:
            if self.sample_t == 0:
                self.cache['key'], self.cache['value'] = self.c_enc_kv(encoder_kv.type_as(x)).chunk(2, dim=2)
            key, value = self.cache['key'], self.cache['value']
            self.sample_t += curr_ctx
        else:
            key, value = self.c_enc_kv(encoder_kv.type_as(x)).chunk(2, dim=2)
        assert key.shape[0] == value.shape[0] == query.shape[0], f'k: {key.shape}, v: {value.shape}, q: {query.shape}'
        assert key.shape[1] == value.shape[1] == self.encoder_dims, f'k: {key.shape}, v: {value.shape}, enc_dims: {self.encoder_dims}'
        assert key.shape[2] == value.shape[2] == query.shape[2], f'k: {key.shape}, v: {value.shape}, q: {query.shape}'
        return query, key, value, sample

    def forward(self, x, encoder_kv=None, sample=False):
        curr_ctx = x.shape[1]
        x = self.c_attn(x)
        query, key, value, sample = self.qkv(x, encoder_kv=encoder_kv, sample=sample)
        if self.checkpoint_attn == 2 and not sample:
            a = checkpoint(lambda q, k, v, s=sample: self.attn(q, k, v, s), (query, key, value), (), True)
        else:
            a = self.attn(query, key, value, sample)
        if a.shape[1] != curr_ctx:
            offset = self._offset(curr_ctx)
            a = a[:, offset:offset + curr_ctx, :].contiguous()
        a = self.c_proj(a)
        return self.resid_dropout(a)

    @property
    def _prime_len(self):
        prime_len = self.prime_len
        assert prime_len is not None
        prime_blocks = prime_len // self.blocks + 1
        return prime_blocks * self.blocks

    def _offset(self, curr_ctx):
        if self.attn_func == 0:
            return 0
        return (self.sample_t - curr_ctx) % self.block_ctx

    def _pad_to_block_ctx(self, x, query=False):
        l = x.shape[1]
        offset = self._offset(l) if query else 0
        n_blocks = (l + offset + self.block_ctx - 1) // self.block_ctx
        pad = n_blocks * self.block_ctx - l - offset
        if pad == 0 and offset == 0:
            return x
        else:
            return F.pad(x, (0, 0, offset, pad))

    def _cache_len(self):
        return 0 if 'key' not in self.cache else self.cache['key'].shape[1]

    def _suff_cache_len(self):
        """
        Precondition:
            key and value are appended with the current context and
            self.sample_t reflects the 1-indexed sample location in the
            context.
        """
        if self.attn_func == 0:
            return self.sample_t
        elif self.attn_func == 1:
            return (self.sample_t - 1) % self.block_ctx + 1
        elif self.attn_func == 2:
            return self.sample_t
        elif self.attn_func == 3:
            if self.sample_t <= self.block_ctx:
                return self.sample_t
            else:
                curr_block = (self.sample_t - 1) % self.block_ctx + 1
                prev_block = self.block_ctx
                return curr_block + prev_block
        elif self.attn_func == 6:
            return self.encoder_dims
        elif self.attn_func == 7:
            return min(self.sample_t, self._prime_len)
        else:
            raise NotImplementedError()

    def _slice_cache(self, start, end=None):
        self.cache['key'] = self.cache['key'][:, start:end]
        self.cache['value'] = self.cache['value'][:, start:end]

    def _append_cache(self, key, value):
        if 'key' not in self.cache:
            self.cache['key'] = key
            self.cache['value'] = value
        else:
            old_key, old_value = key, value
            key = t.cat([self.cache['key'], key], dim=1)
            value = t.cat([self.cache['value'], value], dim=1)
            del self.cache['key']
            del self.cache['value']
            del old_key
            del old_value
            self.cache['key'] = key
            self.cache['value'] = value
        return self.cache['key'], self.cache['value']

    def del_cache(self):
        self.sample_t = 0
        if 'key' in self.cache:
            del self.cache['key']
        if 'value' in self.cache:
            del self.cache['value']
        self.cache = {}

    def check(self):
        blocks = self.blocks or 1
        spread = self.spread or 1
        bs, l, d = 4, self.n_ctx, self.n_in
        x = t.randn(bs, l, d)
        x.requires_grad = True
        x_out = self.forward(x)
        loss = x_out.mean(dim=-1)
        pos = 60
        grad = t.autograd.grad(loss[2, pos], x)[0]
        assert grad.shape == (bs, l, d)
        assert (grad[:2] == 0).all()
        assert (grad[3:] == 0).all()
        assert (grad[2, pos + 1:] == 0).all()
        pos_grad = (t.sum(grad[2] ** 2, dim=-1) > 0).nonzero().view(-1).cpu()
        block_pos = pos - pos % (l // blocks)
        exp_pos_grad = {(0): t.arange(pos), (1): t.arange(block_pos, pos), (2): t.arange(pos % (l // blocks), pos, l // blocks), (3): t.arange(block_pos - l // blocks, block_pos), (4): t.arange(l // blocks - 1, pos, l // blocks), (5): ((t.arange(pos) % (l // blocks) >= l // blocks - spread) & (t.arange(pos) < block_pos)).nonzero().view(-1)}[self.attn_func]
        exp_pos_grad = t.cat([exp_pos_grad, t.tensor([pos])], dim=-1)
        assert len(pos_grad) == len(exp_pos_grad) and (pos_grad == exp_pos_grad).all(), f'Expected pos grad {exp_pos_grad} got {pos_grad} for attn_func {self.attn_func} pos {pos} l {l} blocks {blocks}'

    def check_cache(self, n_samples, sample_t, fp16):
        assert self.sample_t == sample_t, f'{self.sample_t} != {sample_t}'
        if sample_t == 0:
            assert self.cache == {}
        else:
            dtype = {(True): t.float16, (False): t.float32}[fp16]
            l_cache = self._suff_cache_len()
            assert self.cache['key'].shape == (n_samples, l_cache, self.n_state)
            assert self.cache['value'].shape == (n_samples, l_cache, self.n_state)
            assert self.cache['key'].dtype == dtype, f"Expected {dtype}, got {self.cache['key'].dtype}"
            assert self.cache['value'].dtype == dtype, f"Expected {dtype}, got {self.cache['value'].dtype}"

    def check_sample(self):
        t.manual_seed(42)
        bs, l, d = 4, self.n_ctx, self.n_in
        prime = 5
        x = t.randn(bs, l, d)
        xs = t.chunk(x, l, dim=1)
        assert self.sample_t == 0
        assert self.cache == {}
        with t.no_grad():
            enc_l = self.encoder_dims
            encoder_kv = None
            if self.attn_func == 6:
                encoder_kv = t.randn(bs, enc_l, d)
            x_out_normal = self.forward(x, encoder_kv=encoder_kv)
            x_out_sample = t.cat([self.forward(xs[i], encoder_kv=encoder_kv, sample=True) for i in range(l)], dim=1)
        max_err = t.max(t.abs(x_out_sample - x_out_normal))
        assert max_err < 1e-08, f'Max sampling err is {max_err} {[i for i in range(l) if t.max(t.abs(x_out_sample - x_out_normal)[:, i, :]) > 1e-08]}'
        with t.no_grad():
            x_out_normal = x_out_normal[:, :prime, :]
            self.del_cache()
            x_out_sample = self.forward(x[:, :prime, :].contiguous(), encoder_kv=encoder_kv, sample=True)
            self.check_cache(bs, prime, False)
        max_err = t.max(t.abs(x_out_sample - x_out_normal))
        assert max_err < 1e-08, f'Max prime sampling err is {max_err} {[i for i in range(prime) if t.max(t.abs(x_out_sample - x_out_normal)[:, i, :]) > 1e-08]}'

    def check_chunks(self, chunk_size):
        t.manual_seed(42)
        bs, l, d = 4, self.n_ctx, self.n_in
        enc_l = self.encoder_dims
        assert l % chunk_size == 0
        n_chunks = l // chunk_size
        with t.no_grad():
            encoder_kv = None
            x = t.randn(bs, l, d)
            if self.attn_func == 6:
                encoder_kv = t.randn(bs, enc_l, d)
            self.del_cache()
            y_forw = self.forward(x, encoder_kv=encoder_kv, sample=False)
            self.del_cache()
            y_forw_sample = self.forward(x, encoder_kv=encoder_kv, sample=True)
            max_err = t.max(t.abs(y_forw - y_forw_sample))
            assert max_err <= 1e-06, f'Max err is {max_err} {[i for i in range(l) if t.max(t.abs(y_forw - y_forw_sample)[:, i, :]) > 1e-06]}'
            self.del_cache()
            x_chunks = t.chunk(x, n_chunks, dim=1)
            y_chunks = []
            total_len = 0
            for x_chunk in x_chunks:
                y_chunk = self.forward(x_chunk.contiguous(), encoder_kv=encoder_kv, sample=True)
                total_len += x_chunk.shape[1]
                self.check_cache(bs, total_len, False)
                y_chunks.append(y_chunk)
            y_forw_in_chunks = t.cat(y_chunks, dim=1)
            max_err = t.max(t.abs(y_forw - y_forw_in_chunks))
            assert max_err <= 1e-06, f'Max err is {max_err} {[i for i in range(l) if t.max(t.abs(y_forw - y_forw_in_chunks)[:, i, :]) > 1e-06]}'


def gelu(x):
    return 0.5 * x * (1 + t.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * t.pow(x, 3))))


def identity(x):
    return x


@t.jit.script
def quick_gelu(x):
    return x * t.sigmoid(1.702 * x)


@t.jit.script
def quick_gelu_bwd(x, grad_output):
    sig = t.sigmoid(1.702 * x)
    return grad_output * sig * (1.702 * x * (1 - sig) + 1.0)


class QuickGelu(t.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return quick_gelu(x)

    @staticmethod
    def backward(ctx, grad_output):
        return quick_gelu_bwd(ctx.saved_tensors[0], grad_output)


def memory_efficient_quick_gelu(x):
    return QuickGelu.apply(x)


def swish(x):
    return x * t.sigmoid(x)


ACT_FNS = {'relu': t.nn.functional.relu, 'swish': swish, 'gelu': gelu, 'quick_gelu': memory_efficient_quick_gelu, 'identity': identity}


use_cuda = torch.cuda.is_available()


class MLP(nn.Module):

    def __init__(self, n_in, n_state, resid_dropout=0.0, afn='quick_gelu', zero_out=False, init_scale=1.0):
        super().__init__()
        self.c_fc = Conv1D(n_in, n_state, init_scale=init_scale)
        self.c_proj = Conv1D(n_state, n_in, zero_out, init_scale=init_scale)
        if use_cuda:
            self.act = ACT_FNS[afn]
        else:
            self.act = ACT_FNS['identity']
        self.resid_dropout = nn.Dropout(resid_dropout) if resid_dropout > 0.0 else lambda x: x

    def forward(self, x):
        m = self.act(self.c_fc(x))
        m = self.c_proj(m)
        return self.resid_dropout(m)


class ResAttnBlock(nn.Module):

    def __init__(self, n_in, n_ctx, n_head, attn_dropout=0.0, resid_dropout=0.0, afn='quick_gelu', scale=True, mask=False, zero_out=False, init_scale=1.0, res_scale=1.0, m_attn=0.25, m_mlp=1.0, checkpoint_attn=0, checkpoint_mlp=0, attn_func=0, blocks=None, spread=None, encoder_dims=None, prime_len=None):
        super().__init__()
        self.attn = FactoredAttention(n_in=n_in, n_ctx=n_ctx, n_state=int(m_attn * n_in), n_head=n_head, attn_dropout=attn_dropout, resid_dropout=resid_dropout, scale=scale, mask=mask, zero_out=zero_out, init_scale=init_scale, checkpoint_attn=checkpoint_attn, attn_func=attn_func, blocks=blocks, spread=spread, encoder_dims=encoder_dims, prime_len=prime_len)
        self.ln_0 = LayerNorm(n_in)
        self.mlp = MLP(n_in=n_in, n_state=int(m_mlp * n_in), resid_dropout=resid_dropout, afn=afn, zero_out=zero_out, init_scale=init_scale)
        self.ln_1 = LayerNorm(n_in)
        self.res_scale = res_scale
        self.checkpoint_attn = checkpoint_attn
        self.checkpoint_mlp = checkpoint_mlp
        self.n_in = n_in
        self.attn_func = attn_func

    def forward(self, x, encoder_kv, sample=False):
        if sample:
            a = self.attn(self.ln_0(x), encoder_kv, sample)
            m = self.mlp(self.ln_1(x + a))
        else:
            if self.attn_func == 6:
                assert encoder_kv is not None
                a = checkpoint(lambda _x, _enc_kv, _s=sample: self.attn(self.ln_0(_x), _enc_kv, _s), (x, encoder_kv), (*self.attn.parameters(), *self.ln_0.parameters()), self.checkpoint_attn == 3)
            else:
                assert encoder_kv is None
                a = checkpoint(lambda _x, _enc_kv=None, _s=sample: self.attn(self.ln_0(_x), _enc_kv, _s), (x,), (*self.attn.parameters(), *self.ln_0.parameters()), self.checkpoint_attn == 3)
            m = checkpoint(lambda _x: self.mlp(self.ln_1(_x)), (x + a,), (*self.mlp.parameters(), *self.ln_1.parameters()), self.checkpoint_mlp == 1)
        if self.res_scale == 1.0:
            h = x + a + m
        else:
            h = x + self.res_scale * (a + m)
        return h


class Transformer(nn.Module):

    def __init__(self, n_in, n_ctx, n_head, n_depth, attn_dropout=0.0, resid_dropout=0.0, afn='quick_gelu', scale=True, mask=False, zero_out=False, init_scale=1.0, res_scale=False, m_attn=0.25, m_mlp=1.0, checkpoint_attn=0, checkpoint_mlp=0, checkpoint_res=0, attn_order=0, blocks=None, spread=None, encoder_dims=None, prime_len=None):
        super().__init__()
        self.n_in = n_in
        self.n_ctx = n_ctx
        self.encoder_dims = encoder_dims
        self.blocks = blocks
        if blocks is not None:
            assert n_ctx % blocks == 0
            self.block_ctx = n_ctx // blocks
        self.prime_len = prime_len
        self.n_head = n_head
        res_scale = 1.0 / n_depth if res_scale else 1.0
        attn_func = {(0): lambda d: 0, (1): lambda d: [1, 2][d % 2], (2): lambda d: [1, 2, 3][d % 3], (3): lambda d: [1, 4][d % 2], (4): lambda d: [1, 5][d % 2], (5): lambda d: [1, 4, 1, 1][d % 4], (6): lambda d: [1, 2, 3, 6][d % 4], (7): lambda d: [*([1, 2, 3] * 5), 6][d % 16], (8): lambda d: [1, 2, 3, 1, 2, 3, 1, 2, 3, 6][d % 10], (9): lambda d: [1, 2, 3, 0][d % 4], (10): lambda d: [*[1, 2, 3, 1, 2, 3, 1, 2, 3], *([1, 2, 3, 1, 2, 3, 1, 2, 3, 6] * 7)][d % 79], (11): lambda d: [6, 6, 0][d % 3] if d % 16 == 15 else [1, 2, 3][d % 3], (12): lambda d: [7, 7, 0][d % 3] if d % 16 == 15 else [1, 2, 3][d % 3]}[attn_order]
        attn_cycle = {(0): 1, (1): 2, (2): 3, (3): 2, (4): 2, (5): 4, (6): 4, (7): 16, (8): 10, (9): 4, (10): 79, (11): 16, (12): 16}[attn_order]
        attn_block = lambda d: ResAttnBlock(n_in=n_in, n_ctx=n_ctx, n_head=n_head, attn_dropout=attn_dropout, resid_dropout=resid_dropout, afn=afn, scale=scale, mask=mask, zero_out=zero_out if attn_func(d) != 6 else True, init_scale=init_scale, res_scale=res_scale, m_attn=m_attn, m_mlp=m_mlp, checkpoint_attn=checkpoint_attn, checkpoint_mlp=checkpoint_mlp, attn_func=attn_func(d), blocks=blocks, spread=spread, encoder_dims=encoder_dims, prime_len=prime_len)
        self.checkpoint_res = checkpoint_res
        self._attn_mods = nn.ModuleList()
        for d in range(n_depth):
            self._attn_mods.append(attn_block(d))
        self.ws = []

    def set_record_attn(self, record_attn):
        """
        Arguments:
            record_attn (bool or set): Makes forward prop dump self-attention
                softmaxes to self.ws. Either a set of layer indices indicating
                which layers to store, or a boolean value indicating whether to
                dump all.
        """

        def _should_record_attn(layer_idx):
            if isinstance(record_attn, bool):
                return record_attn
            return layer_idx in record_attn
        for i, l in enumerate(self._attn_mods):
            l.attn.record_attn = _should_record_attn(i)
        if record_attn:
            assert self.ws == []
            for l in self._attn_mods:
                assert l.attn.w == None
        else:
            self.ws = []
            for l in self._attn_mods:
                l.attn.w = None

    def forward(self, x, encoder_kv=None, sample=False, fp16=False, fp16_out=False):
        if fp16:
            x = x.half()
        for i, l in enumerate(self._attn_mods):
            if self.checkpoint_res == 1 and not sample:
                if l.attn_func == 6:
                    assert encoder_kv is not None
                    f = functools.partial(l, sample=sample)
                    x = checkpoint(f, (x, encoder_kv), l.parameters(), True)
                else:
                    f = functools.partial(l, encoder_kv=None, sample=sample)
                    x = checkpoint(f, (x,), l.parameters(), True)
            elif l.attn_func == 6:
                x = l(x, encoder_kv=encoder_kv, sample=sample)
            else:
                x = l(x, encoder_kv=None, sample=sample)
            if l.attn.record_attn:
                self.ws.append(l.attn.w)
        if not fp16_out:
            x = x.float()
        return x

    def check_cache(self, n_samples, sample_t, fp16):
        for l in self._attn_mods:
            l.attn.check_cache(n_samples, sample_t, fp16)

    def del_cache(self):
        for l in self._attn_mods:
            l.attn.del_cache()

    def check_sample(self):
        bs, l, s, d = 4, self.n_ctx, self.encoder_dims, self.n_in
        prime = 5
        with t.no_grad():
            encoder_kv = t.randn(bs, s, d)
            x = t.randn(bs, l, d)
            y_forw = self.forward(x, encoder_kv=encoder_kv, sample=True)
            self.del_cache()
            x_chunks = t.chunk(x, 4, dim=1)
            y_chunks = []
            n = 0
            for x_chunk in x_chunks:
                self.check_cache(bs, n, False)
                y_chunk = self.forward(x_chunk, encoder_kv=encoder_kv, sample=True)
                y_chunks.append(y_chunk)
                n += x_chunk.shape[1]
            self.check_cache(bs, n, False)
            y_forw_in_chunks = t.cat(y_chunks, dim=1)
            max_err = t.max(t.abs(y_forw - y_forw_in_chunks))
            assert max_err <= 1e-06, f'Max err is {max_err} {[i for i in range(l) if t.max(t.abs(y_forw - y_forw_in_chunks)[:, i, :]) > 1e-06]}'


def empty_cache():
    gc.collect()
    t.cuda.empty_cache()


def filter_logits(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k >0: keep only top k tokens with highest probability (top-k filtering).
            top_p >0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
    """
    logits = logits.clone()
    top_k = min(top_k, logits.size(-1))
    assert top_k == 0 or top_p == 0.0
    if top_k > 0:
        indices_to_remove = logits < t.topk(logits, top_k, dim=-1)[0][..., -1:]
        logits[indices_to_remove] = filter_value
    if top_p > 0.0:
        sorted_logits, sorted_indices = t.sort(logits, descending=True, dim=-1)
        cumulative_probs = t.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = t.zeros_like(logits, dtype=sorted_indices_to_remove.dtype).scatter_(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits


def def_tqdm(x, desc=None):
    return tqdm(x, desc=desc, leave=True, file=sys.stdout, bar_format='{n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}] {desc}')


def get_range(x, desc=None):
    if dist.get_rank() == 0:
        return def_tqdm(x, desc=desc)
    else:
        return x


def roll(x, n):
    return t.cat((x[:, -n:], x[:, :-n]), dim=1)


def split_chunks(length, chunk_size):
    n_passes = (length + chunk_size - 1) // chunk_size
    chunk_sizes = [*([chunk_size] * (n_passes - 1)), (length - 1) % chunk_size + 1]
    assert sum(chunk_sizes) == length
    return chunk_sizes


class ConditionalAutoregressive2D(nn.Module):

    def __init__(self, input_shape, bins, width=128, depth=2, heads=1, attn_dropout=0.0, resid_dropout=0.0, emb_dropout=0.0, mask=True, zero_out=False, init_scale=1.0, res_scale=False, pos_init=False, m_attn=0.25, m_mlp=1, checkpoint_res=0, checkpoint_attn=0, checkpoint_mlp=0, attn_order=0, blocks=None, spread=None, x_cond=False, y_cond=False, encoder_dims=0, only_encode=False, merged_decoder=False, prime_len=None):
        super().__init__()
        self.input_shape = input_shape
        self.input_dims = input_dims = np.prod(input_shape)
        self.encoder_dims = encoder_dims
        self.bins = bins
        self.width = width
        self.depth = depth
        self.x_emb = nn.Embedding(bins, width)
        nn.init.normal_(self.x_emb.weight, std=0.02 * init_scale)
        self.x_emb_dropout = nn.Dropout(emb_dropout)
        self.y_cond = y_cond
        self.x_cond = x_cond
        if not y_cond:
            self.start_token = nn.Parameter(get_normal(1, width, std=0.01 * init_scale))
        self.pos_emb = PositionEmbedding(input_shape=input_shape, width=width, init_scale=init_scale, pos_init=pos_init)
        self.pos_emb_dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(n_in=width, n_ctx=input_dims, n_head=heads, n_depth=depth, attn_dropout=attn_dropout, resid_dropout=resid_dropout, afn='quick_gelu', scale=True, mask=mask, zero_out=zero_out, init_scale=init_scale, res_scale=res_scale, m_attn=m_attn, m_mlp=m_mlp, checkpoint_attn=checkpoint_attn, checkpoint_mlp=checkpoint_mlp, checkpoint_res=checkpoint_res, attn_order=attn_order, blocks=blocks, spread=spread, encoder_dims=encoder_dims, prime_len=prime_len)
        self.only_encode = only_encode
        self.prime_len = prime_len
        if merged_decoder:
            self.add_cond_after_transformer = False
            self.share_x_emb_x_out = False
        else:
            self.add_cond_after_transformer = True
            self.share_x_emb_x_out = True
        if not only_encode:
            self.x_out = nn.Linear(width, bins, bias=False)
            if self.share_x_emb_x_out:
                self.x_out.weight = self.x_emb.weight
            self.loss = t.nn.CrossEntropyLoss()

    def preprocess(self, x):
        N = x.shape[0]
        return x.view(N, -1).long()

    def postprocess(self, x, sample_tokens=None):
        N = x.shape[0]
        assert (0 <= x).all() and (x < self.bins).all()
        if sample_tokens is None or sample_tokens == self.input_dims:
            return x.view(N, *self.input_shape)
        else:
            return x.view(N, -1)

    def forward(self, x, x_cond=None, y_cond=None, encoder_kv=None, fp16=False, loss_full=False, encode=False, get_preds=False, get_acts=False, get_sep_loss=False):
        with t.no_grad():
            x = self.preprocess(x)
        N, D = x.shape
        if use_cuda:
            assert isinstance(x, t.cuda.LongTensor)
        else:
            assert isinstance(x, t.LongTensor)
        assert (0 <= x).all() and (x < self.bins).all()
        if self.y_cond:
            assert y_cond is not None
            assert y_cond.shape == (N, 1, self.width)
        else:
            assert y_cond is None
        if self.x_cond:
            assert x_cond is not None
            assert x_cond.shape == (N, D, self.width) or x_cond.shape == (N, 1, self.width), f'{x_cond.shape} != {N, D, self.width} nor {N, 1, self.width}. Did you pass the correct --sample_length?'
        else:
            assert x_cond is None
            x_cond = t.zeros((N, 1, self.width), device=x.device, dtype=t.float)
        x_t = x
        x = self.x_emb(x)
        x = roll(x, 1)
        if self.y_cond:
            x[:, 0] = y_cond.view(N, self.width)
        else:
            x[:, 0] = self.start_token
        x = self.x_emb_dropout(x) + self.pos_emb_dropout(self.pos_emb()) + x_cond
        x = self.transformer(x, encoder_kv=encoder_kv, fp16=fp16)
        if self.add_cond_after_transformer:
            x = x + x_cond
        acts = x
        if self.only_encode:
            return x
        x = self.x_out(x)
        if get_sep_loss:
            assert self.prime_len is not None
            x_prime = x[:, :self.prime_len].reshape(-1, self.bins)
            x_gen = x[:, self.prime_len:].reshape(-1, self.bins)
            prime_loss = F.cross_entropy(x_prime, x_t[:, :self.prime_len].reshape(-1)) / np.log(2.0)
            gen_loss = F.cross_entropy(x_gen, x_t[:, self.prime_len:].reshape(-1)) / np.log(2.0)
            loss = prime_loss, gen_loss
        else:
            loss = F.cross_entropy(x.view(-1, self.bins), x_t.view(-1)) / np.log(2.0)
        if get_preds:
            return loss, x
        elif get_acts:
            return loss, acts
        else:
            return loss, None

    def get_emb(self, sample_t, n_samples, x, x_cond, y_cond):
        N, D = n_samples, self.input_dims
        if sample_t == 0:
            x = t.empty(n_samples, 1, self.width)
            if use_cuda:
                x = x
            if self.y_cond:
                x[:, 0] = y_cond.view(N, self.width)
            else:
                x[:, 0] = self.start_token
        else:
            if use_cuda:
                assert isinstance(x, t.cuda.LongTensor)
            else:
                assert isinstance(x, t.LongTensor)
            assert (0 <= x).all() and (x < self.bins).all()
            x = self.x_emb(x)
        assert x.shape == (n_samples, 1, self.width)
        if x_cond.shape == (N, D, self.width):
            cond = x_cond[:, sample_t:sample_t + 1, :]
        else:
            cond = x_cond
        x = x + self.pos_emb()[sample_t:sample_t + 1] + cond
        assert x.shape == (n_samples, 1, self.width)
        return x, cond

    def get_logits(self, x_cond, y_cond, sample_t, n_samples, x, fp16, encoder_kv, top_k, top_p, temp, i=0):
        x, cond = self.get_emb(sample_t, n_samples, x, x_cond, y_cond)
        if i > 0:
            self.transformers[1].check_cache(n_samples, sample_t, fp16)
            x = self.transformers[1](x, encoder_kv=encoder_kv, sample=True, fp16=fp16)
        else:
            self.transformer.check_cache(n_samples, sample_t, fp16)
            x = self.transformer(x, encoder_kv=encoder_kv, sample=True, fp16=fp16)
        if self.add_cond_after_transformer:
            x = x + cond
        assert x.shape == (n_samples, 1, self.width)
        x = self.x_out(x)
        pred = x.clone()
        return x, pred

    def sample(self, n_samples, x_cond=None, y_cond=None, encoder_kv=None, fp16=False, temp=1.0, top_k=0, top_p=0.0, get_preds=False, sample_tokens=None, cfg_s=0):
        assert self.training == False
        if sample_tokens is None:
            sample_tokens = self.input_dims
        N, D = n_samples, self.input_dims
        if self.y_cond:
            assert y_cond is not None
            if isinstance(y_cond, list):
                for y in y_cond:
                    assert y.shape == (N, 1, self.width)
            else:
                assert y_cond.shape == (N, 1, self.width)
        else:
            assert y_cond is None
        if self.x_cond:
            assert x_cond is not None
            if isinstance(x_cond, list):
                for x in x_cond:
                    assert x.shape == (N, D, self.width) or x_cond.shape == (N, 1, self.width), f'Got {x_cond.shape}, expected ({N}, {D}/{1}, {self.width})'
            else:
                assert x_cond.shape == (N, D, self.width) or x_cond.shape == (N, 1, self.width), f'Got {x_cond.shape}, expected ({N}, {D}/{1}, {self.width})'
        else:
            assert x_cond is None
            x_cond = t.zeros((N, 1, self.width), dtype=t.float)
            if use_cuda:
                x_cond = x_cond
        with t.no_grad():
            xs, x = [], None
            if get_preds:
                preds = []
            if isinstance(y_cond, list):
                import copy
                self.transformers = [self.transformer, copy.deepcopy(self.transformer)]
            for sample_t in get_range(range(0, sample_tokens)):
                if isinstance(y_cond, list):
                    pred_cur, prob_cur, logits_cur = [], [], []
                    for i in range(len(y_cond)):
                        logits, pred = self.get_logits(x_cond[i], y_cond[i], sample_t, n_samples, x, fp16, encoder_kv, top_k, top_p, temp, i)
                        logits_cur.append(logits)
                        pred_cur.append(pred)
                    if get_preds:
                        preds.append(pred_cur)
                    logits_cfg = cfg_s * logits_cur[0] + (1 - cfg_s) * logits_cur[1]
                    logits_cfg = logits_cfg / temp
                    logits_cfg = filter_logits(logits_cfg, top_k=top_k, top_p=top_p)
                    x = t.distributions.Categorical(logits=logits_cfg).sample()
                else:
                    logits, pred = self.get_logits(x_cond, y_cond, sample_t, n_samples, x, fp16, encoder_kv, top_k, top_p, temp)
                    if get_preds:
                        preds.append(pred)
                    logits = logits / temp
                    logits = filter_logits(logits, top_k=top_k, top_p=top_p)
                    x = t.distributions.Categorical(logits=logits).sample()
                assert x.shape == (n_samples, 1)
                xs.append(x.clone())
            del x
            if isinstance(y_cond, list):
                for i in range(len(y_cond)):
                    self.transformers[i].del_cache()
            else:
                self.transformer.del_cache()
            x = t.cat(xs, dim=1)
            if get_preds:
                preds = t.cat(preds, dim=1)
            x = self.postprocess(x, sample_tokens)
        if get_preds:
            return x, preds
        else:
            return x

    def cache_prime(self, x_cond, y_cond, n_samples, chunk_sizes, transformer, fp16, encoder_kv, xs, get_preds, x_primes, preds):
        start = 0
        x = None
        for current_chunk_size in get_range(chunk_sizes, desc='Fill up key/value cache for past context by chunks'):
            xs_prime, conds_prime = [], []
            for sample_t in range(start, start + current_chunk_size):
                x_prime, cond_prime = self.get_emb(sample_t, n_samples, x, x_cond, y_cond)
                x = xs[sample_t]
                xs_prime.append(x_prime)
                conds_prime.append(cond_prime)
            start = start + current_chunk_size
            x_prime, cond_prime = t.cat(xs_prime, dim=1), t.cat(conds_prime, dim=1)
            assert x_prime.shape == (n_samples, current_chunk_size, self.width)
            assert cond_prime.shape == (n_samples, current_chunk_size, self.width)
            del xs_prime
            del conds_prime
            if not get_preds:
                del cond_prime
            None
            x_prime = transformer(x_prime, encoder_kv=encoder_kv, sample=True, fp16=fp16)
            if get_preds:
                if self.add_cond_after_transformer:
                    x_prime = x_prime + cond_prime
                assert x_prime.shape == (n_samples, current_chunk_size, self.width)
                del cond_prime
                x_primes.append(x_prime)
            else:
                del x_prime
        if get_preds:
            x_prime = t.cat(x_primes, dim=1)
            assert x_prime.shape == (n_samples, len(xs), self.width)
            x_prime = self.x_out(x_prime)
            preds.append(x_prime)
        empty_cache()
        transformer.check_cache(n_samples, len(xs), fp16)

    def primed_sample(self, n_samples, x, x_cond=None, y_cond=None, encoder_kv=None, fp16=False, temp=1.0, top_k=0, top_p=0.0, get_preds=False, chunk_size=None, sample_tokens=None, cfg_s=0):
        assert self.training == False
        if sample_tokens is None:
            sample_tokens = self.input_dims
        with t.no_grad():
            x = self.preprocess(x)
        if use_cuda:
            assert isinstance(x, t.cuda.LongTensor)
        else:
            assert isinstance(x, t.LongTensor)
        assert (0 <= x).all() and (x < self.bins).all()
        assert x.shape[0] == n_samples
        xs = t.split(x, 1, dim=1)
        xs = list(xs)
        assert len(xs) < sample_tokens
        N, D = n_samples, self.input_dims
        if self.y_cond:
            assert y_cond is not None
            if isinstance(y_cond, list):
                for y in y_cond:
                    assert y.shape == (N, 1, self.width)
        else:
            assert y_cond is None
        if self.x_cond:
            assert x_cond is not None
            if isinstance(x_cond, list):
                for x in x_cond:
                    assert x.shape == (N, D, self.width) or x_cond.shape == (N, 1, self.width), f'Got {x_cond.shape}, expected ({N}, {D}/{1}, {self.width})'
            else:
                assert x_cond.shape == (N, D, self.width) or x_cond.shape == (N, 1, self.width), f'Got {x_cond.shape}, expected ({N}, {D}/{1}, {self.width})'
        else:
            assert x_cond is None
            x_cond = t.zeros((N, 1, self.width), dtype=t.float)
        with t.no_grad():
            if get_preds:
                preds = []
            else:
                preds = None
            if chunk_size is None:
                chunk_size = len(xs)
            chunk_sizes = split_chunks(len(xs), chunk_size)
            x_primes = []
            None
            if isinstance(y_cond, list):
                import copy
                self.transformers = [self.transformer, copy.deepcopy(self.transformer)]
                for i in range(len(y_cond)):
                    self.cache_prime(x_cond[i], y_cond[i], n_samples, chunk_sizes, self.transformers[i], fp16, encoder_kv, xs, get_preds, x_primes, preds)
            else:
                self.cache_prime(x_cond, y_cond, n_samples, chunk_sizes, self.transformer, fp16, encoder_kv, xs, get_preds, x_primes, preds)
            x = xs[-1]
            assert x.shape == (n_samples, 1)
            empty_cache()
            for sample_t in get_range(range(len(xs), sample_tokens)):
                if isinstance(y_cond, list):
                    pred_cur, prob_cur, logits_cur = [], [], []
                    for i in range(len(y_cond)):
                        logits, pred = self.get_logits(x_cond[i], y_cond[i], sample_t, n_samples, x, fp16, encoder_kv, top_k, top_p, temp, i)
                        logits_cur.append(logits)
                        pred_cur.append(pred)
                    if get_preds:
                        preds.append(pred_cur)
                    logits_cfg = cfg_s * logits_cur[0] + (1 - cfg_s) * logits_cur[1]
                    logits_cfg = logits_cfg / temp
                    logits_cfg = filter_logits(logits_cfg, top_k=top_k, top_p=top_p)
                    x = t.distributions.Categorical(logits=logits_cfg).sample()
                else:
                    logits, pred = self.get_logits(x_cond, y_cond, sample_t, n_samples, x, fp16, encoder_kv, top_k, top_p, temp)
                    if get_preds:
                        preds.append(pred)
                    logits = logits / temp
                    logits = filter_logits(logits, top_k=top_k, top_p=top_p)
                    x = t.distributions.Categorical(logits=logits).sample()
                assert x.shape == (n_samples, 1)
                xs.append(x.clone())
            del x
            if isinstance(y_cond, list):
                for i in range(len(y_cond)):
                    self.transformers[i].del_cache()
            else:
                self.transformer.del_cache()
            x = t.cat(xs, dim=1)
            if get_preds:
                preds = t.cat(preds, dim=1)
            x = self.postprocess(x, sample_tokens)
        if get_preds:
            return x, preds
        else:
            return x

    def check_sample(self, chunk_size):
        bs, l, d = 4, self.input_dims, self.width
        prime = int(self.input_dims // 8 * 7)
        enc_l = self.encoder_dims
        with t.no_grad():
            y_cond = t.randn(bs, 1, d) if self.y_cond else None
            x_cond = t.randn(bs, l, d) if self.x_cond else None
            encoder_kv = t.randn(bs, enc_l, d)
            x, preds_sample = self.sample(bs, x_cond, y_cond, encoder_kv, get_preds=True)
            loss, preds_forw = self.forward(x, x_cond, y_cond, encoder_kv, get_preds=True)
            max_err = t.max(t.abs(preds_sample - preds_forw))
            assert max_err <= 1e-06, f'Max err is {max_err} {[i for i in range(l) if t.max(t.abs(preds_sample - preds_forw)[:, i, :]) > 1e-06]}'
            x_prime = x.view(bs, -1)[:, :prime]
            x, preds_sample = self.primed_sample(bs, x_prime.clone(), x_cond, y_cond, encoder_kv, get_preds=True)
            assert (x.view(bs, -1)[:, :prime] == x_prime).all(), "Priming samples don't match"
            loss, preds_forw = self.forward(x, x_cond, y_cond, encoder_kv, get_preds=True)
            max_err = t.max(t.abs(preds_sample - preds_forw))
            assert max_err <= 1e-06, f'Max err is {max_err} {[i for i in range(l) if t.max(t.abs(preds_sample - preds_forw)[:, i, :]) > 1e-06]}'
            x, preds_sample = self.primed_sample(bs, x_prime.clone(), x_cond, y_cond, encoder_kv, get_preds=True, chunk_size=chunk_size)
            assert (x.view(bs, -1)[:, :prime] == x_prime).all(), "Priming samples don't match"
            loss, preds_forw = self.forward(x, x_cond, y_cond, encoder_kv, get_preds=True)
            max_err = t.max(t.abs(preds_sample - preds_forw))
            assert max_err <= 1e-06, f'Max err is {max_err} {[i for i in range(l) if t.max(t.abs(preds_sample - preds_forw)[:, i, :]) > 1e-06]}'


class ResConv1DBlock(nn.Module):

    def __init__(self, n_in, n_state, dilation=1, zero_out=False, res_scale=1.0):
        super().__init__()
        padding = dilation
        self.model = nn.Sequential(nn.ReLU(), nn.Conv1d(n_in, n_state, 3, 1, padding, dilation), nn.ReLU(), nn.Conv1d(n_state, n_in, 1, 1, 0))
        if zero_out:
            out = self.model[-1]
            nn.init.zeros_(out.weight)
            nn.init.zeros_(out.bias)
        self.res_scale = res_scale

    def forward(self, x):
        return x + self.res_scale * self.model(x)


class Resnet1D(nn.Module):

    def __init__(self, n_in, n_depth, m_conv=1.0, dilation_growth_rate=1, dilation_cycle=None, zero_out=False, res_scale=False, reverse_dilation=False, checkpoint_res=False):
        super().__init__()

        def _get_depth(depth):
            if dilation_cycle is None:
                return depth
            else:
                return depth % dilation_cycle
        blocks = [ResConv1DBlock(n_in, int(m_conv * n_in), dilation=dilation_growth_rate ** _get_depth(depth), zero_out=zero_out, res_scale=1.0 if not res_scale else 1.0 / math.sqrt(n_depth)) for depth in range(n_depth)]
        if reverse_dilation:
            blocks = blocks[::-1]
        self.checkpoint_res = checkpoint_res
        if self.checkpoint_res == 1:
            if dist.get_rank() == 0:
                None
            self.blocks = nn.ModuleList(blocks)
        else:
            self.model = nn.Sequential(*blocks)

    def forward(self, x):
        if self.checkpoint_res == 1:
            for block in self.blocks:
                x = checkpoint(block, (x,), block.parameters(), True)
            return x
        else:
            return self.model(x)


class DecoderConvBock(nn.Module):

    def __init__(self, input_emb_width, output_emb_width, down_t, stride_t, width, depth, m_conv, dilation_growth_rate=1, dilation_cycle=None, zero_out=False, res_scale=False, reverse_decoder_dilation=False, checkpoint_res=False):
        super().__init__()
        blocks = []
        if down_t > 0:
            filter_t, pad_t = stride_t * 2, stride_t // 2
            block = nn.Conv1d(output_emb_width, width, 3, 1, 1)
            blocks.append(block)
            for i in range(down_t):
                block = nn.Sequential(Resnet1D(width, depth, m_conv, dilation_growth_rate, dilation_cycle, zero_out=zero_out, res_scale=res_scale, reverse_dilation=reverse_decoder_dilation, checkpoint_res=checkpoint_res), nn.ConvTranspose1d(width, input_emb_width if i == down_t - 1 else width, filter_t, stride_t, pad_t))
                blocks.append(block)
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)


def assert_shape(x, exp_shape):
    assert x.shape == exp_shape, f'Expected {exp_shape} got {x.shape}'


class Conditioner(nn.Module):

    def __init__(self, input_shape, bins, down_t, stride_t, out_width, init_scale, zero_out, res_scale, **block_kwargs):
        super().__init__()
        self.x_shape = input_shape
        self.width = out_width
        self.x_emb = nn.Embedding(bins, out_width)
        nn.init.normal_(self.x_emb.weight, std=0.02 * init_scale)
        self.cond = DecoderConvBock(self.width, self.width, down_t, stride_t, **block_kwargs, zero_out=zero_out, res_scale=res_scale)
        self.ln = LayerNorm(self.width)

    def preprocess(self, x):
        x = x.permute(0, 2, 1)
        return x

    def postprocess(self, x):
        x = x.permute(0, 2, 1)
        return x

    def forward(self, x, x_cond=None):
        N = x.shape[0]
        assert_shape(x, (N, *self.x_shape))
        if x_cond is not None:
            assert_shape(x_cond, (N, *self.x_shape, self.width))
        else:
            x_cond = 0.0
        x = x.long()
        x = self.x_emb(x)
        assert_shape(x, (N, *self.x_shape, self.width))
        x = x + x_cond
        x = self.preprocess(x)
        x = self.cond(x)
        x = self.postprocess(x)
        x = self.ln(x)
        return x


class SimpleEmbedding(nn.Module):

    def __init__(self, bins, out_width, init_scale):
        super().__init__()
        self.bins = bins
        self.emb = nn.Embedding(bins, out_width)
        nn.init.normal_(self.emb.weight, std=0.01 * init_scale)

    def forward(self, y):
        assert len(y.shape) == 2, f'Expected shape with 2 dims, got {y.shape}'
        assert (0 <= y).all() and (y < self.bins).all(), f'Bins {self.bins}, got label {y}'
        return self.emb(y)


class MLPLayers(nn.Module):

    def __init__(self, units=[512, 512, 512], nonlin=nn.ReLU(), dropout=0.1):
        super(MLPLayers, self).__init__()
        self.nonlin = nonlin
        self.dropout = dropout
        sequence = []
        for u0, u1 in zip(units[:-1], units[1:]):
            sequence.append(nn.Linear(u0, u1))
            sequence.append(self.nonlin)
            sequence.append(nn.Dropout(self.dropout))
        sequence = sequence[:-2]
        self.sequential = nn.Sequential(*sequence)

    def forward(self, X):
        X = self.sequential(X)
        return X


class RangeEmbedding(nn.Module):

    def __init__(self, n_time, bins, range, out_width, init_scale, clamp=False):
        super().__init__()
        self.n_time = n_time
        self.bins = bins
        self.emb = nn.Embedding(bins, out_width)
        nn.init.normal_(self.emb.weight, std=0.01 * init_scale)
        self.pos_min, self.pos_max = range
        self.clamp = clamp

    def forward(self, pos_start, pos_end=None):
        assert len(pos_start.shape) == 2, f'Expected shape with 2 dims, got {pos_start.shape}'
        assert (self.pos_min <= pos_start).all() and (pos_start < self.pos_max).all(), f'Range is [{self.pos_min},{self.pos_max}), got {pos_start}'
        pos_start = pos_start.float()
        if pos_end is not None:
            assert len(pos_end.shape) == 2, f'Expected shape with 2 dims, got {pos_end.shape}'
            if self.clamp:
                pos_end = pos_end.clamp(self.pos_min, self.pos_max)
            assert (self.pos_min <= pos_end).all() and (pos_end <= self.pos_max).all(), f'Range is [{self.pos_min},{self.pos_max}), got {pos_end}'
            pos_end = pos_end.float()
        n_time = self.n_time
        if n_time != 1:
            assert pos_end is not None
            if use_cuda:
                interpolation = t.arange(0, n_time, dtype=t.float, device='cuda').view(1, n_time) / n_time
            else:
                interpolation = t.arange(0, n_time, dtype=t.float).view(1, n_time) / n_time
            position = pos_start + (pos_end - pos_start) * interpolation
        else:
            position = pos_start
        normalised_position = (position - self.pos_min) / (self.pos_max - self.pos_min)
        bins = (self.bins * normalised_position).floor().long().detach()
        return self.emb(bins)


class Condition_Modes(Enum):
    label = 1
    null_label = 2
    default = 3


class LabelConditioner(nn.Module):

    def __init__(self, y_bins, t_bins, sr, min_duration, max_duration, n_time, out_width, init_scale, max_bow_genre_size, include_time_signal, clip_emb, video_clip_emb, class_free_guidance_prob):
        super().__init__()
        self.n_time = n_time
        self.clip_emb = clip_emb
        self.video_clip_emb = video_clip_emb
        self.out_width = out_width
        self.class_free_guidance_prob = class_free_guidance_prob
        if self.clip_emb:
            self.clip_map = MLPLayers([512, 512, out_width])
            if self.class_free_guidance_prob >= 0:
                self.null_CLIP = nn.Parameter(get_normal(self.out_width, std=0.01 * init_scale))
            if self.video_clip_emb:
                self.video_map = MLPLayers([512, 512, out_width])
                if self.class_free_guidance_prob >= 0:
                    self.null_video_CLIP = nn.Parameter(get_normal(self.out_width, std=0.01 * init_scale))
        else:
            assert len(y_bins) == 2, f'Expecting (genre, artist) bins, got {y_bins}'
            bow_genre_bins, artist_bins = y_bins
            self.max_bow_genre_size = max_bow_genre_size
            self.bow_genre_emb = SimpleEmbedding(bow_genre_bins, out_width, init_scale)
            self.artist_emb = SimpleEmbedding(artist_bins, out_width, init_scale)
        self.include_time_signal = include_time_signal
        if self.include_time_signal:
            t_ranges = (min_duration * sr, max_duration * sr), (0.0, max_duration * sr), (0.0, 1.0)
            assert len(t_ranges) == 3, f'Expecting (total, absolute, relative) ranges, got {t_ranges}'
            total_length_range, absolute_pos_range, relative_pos_range = t_ranges
            self.total_length_emb = RangeEmbedding(1, t_bins, total_length_range, out_width, init_scale)
            self.absolute_pos_emb = RangeEmbedding(n_time, t_bins, absolute_pos_range, out_width, init_scale)
            self.relative_pos_emb = RangeEmbedding(n_time, t_bins, relative_pos_range, out_width, init_scale, clamp=True)

    def forward(self, y, mode=Condition_Modes.default):
        if self.video_clip_emb:
            assert len(y.shape) == 3, f'Expected shape with 3 dims, got {y.shape}'
        else:
            assert len(y.shape) == 2, f'Expected shape with 2 dims, got {y.shape}'
        if self.clip_emb:
            assert y.shape[-1] == 3 + 512, f'Expected shape (N,{3 + 512}), got {y.shape}'
        else:
            assert y.shape[-1] == 4 + self.max_bow_genre_size, f'Expected shape (N,{4 + self.max_bow_genre_size}), got {y.shape}'
        N = y.shape[0]
        video_emb = None
        if self.clip_emb:
            if self.video_clip_emb:
                total_length, offset, length, clip = y[:, 0, 0:1], y[:, 0, 1:2], y[:, 0, 2:3], y[:, :, 3:]
                mean_clip = t.mean(clip, dim=1)
                start_emb = self.clip_map(mean_clip)
                video_emb = self.video_map(clip)
            else:
                total_length, offset, length, clip = y[:, 0:1], y[:, 1:2], y[:, 2:3], y[:, 3:]
                start_emb = self.clip_map(clip)
            start_emb = start_emb[:, None, :]
        else:
            total_length, offset, length, artist, genre = y[:, 0:1], y[:, 1:2], y[:, 2:3], y[:, 3:4], y[:, 4:]
            artist_emb = self.artist_emb(artist)
            mask = (genre >= 0).float().unsqueeze(2)
            genre_emb = (self.bow_genre_emb(genre.clamp(0)) * mask).sum(dim=1, keepdim=True)
            start_emb = genre_emb + artist_emb
            assert_shape(start_emb, (N, 1, self.out_width))
        if self.include_time_signal:
            start, end = offset, offset + length
            total_length, start, end = total_length.float(), start.float(), end.float()
            pos_emb = self.total_length_emb(total_length) + self.absolute_pos_emb(start, end) + self.relative_pos_emb(start / total_length, end / total_length)
            assert_shape(pos_emb, (N, self.n_time, self.out_width))
        else:
            pos_emb = None
        if mode == Condition_Modes.default:
            cfg_prob = self.class_free_guidance_prob
        elif mode == Condition_Modes.label:
            cfg_prob = -1
        elif mode == Condition_Modes.null_label:
            cfg_prob = 2
        if self.class_free_guidance_prob >= 0:
            if use_cuda:
                mask = t.cuda.FloatTensor(start_emb.shape[0]).uniform_() < cfg_prob
            else:
                mask = t.FloatTensor(start_emb.shape[0]).uniform_() < cfg_prob
            start_emb[mask] = self.null_CLIP
            if self.video_clip_emb:
                video_emb[mask, :] = self.null_video_CLIP
        return start_emb, pos_emb, video_emb


def calculate_strides(strides, downs):
    return [(stride ** down) for stride, down in zip(strides, downs)]


def print_once(msg):
    if not dist.is_available() or dist.get_rank() == 0:
        None


class SimplePrior(nn.Module):

    def __init__(self, z_shapes, l_bins, encoder, decoder, level, downs_t, strides_t, labels, prior_kwargs, x_cond_kwargs, y_cond_kwargs, prime_kwargs, copy_input, labels_v3=False, merged_decoder=False, single_enc_dec=False, clip_emb=False, video_clip_emb=False):
        super().__init__()
        self.use_tokens = prime_kwargs.pop('use_tokens')
        self.n_tokens = prime_kwargs.pop('n_tokens')
        self.prime_loss_fraction = prime_kwargs.pop('prime_loss_fraction')
        self.copy_input = copy_input
        self.clip_emb = clip_emb
        self.video_clip_emb = video_clip_emb
        if self.copy_input:
            prime_kwargs['bins'] = l_bins
        self.z_shapes = z_shapes
        self.levels = len(self.z_shapes)
        self.z_shape = self.z_shapes[level]
        self.level = level
        assert level < self.levels, f'Total levels {self.levels}, got level {level}'
        self.l_bins = l_bins
        self.encoder = encoder
        self.decoder = decoder
        self.x_cond = level != self.levels - 1
        self.cond_level = level + 1
        self.y_cond = labels
        self.single_enc_dec = single_enc_dec
        if self.x_cond:
            self.conditioner_blocks = nn.ModuleList()
            conditioner_block = lambda _level: Conditioner(input_shape=z_shapes[_level], bins=l_bins, down_t=downs_t[_level], stride_t=strides_t[_level], **x_cond_kwargs)
            if dist.get_rank() == 0:
                None
            self.conditioner_blocks.append(conditioner_block(self.cond_level))
        if self.y_cond:
            self.n_time = self.z_shape[0]
            self.y_emb = LabelConditioner(n_time=self.n_time, include_time_signal=not self.x_cond, **y_cond_kwargs)
        if single_enc_dec:
            self.prior_shapes = [(self.n_tokens,), prior_kwargs.pop('input_shape')]
            self.prior_bins = [prime_kwargs['bins'], prior_kwargs.pop('bins')]
            self.prior_dims = [np.prod(shape) for shape in self.prior_shapes]
            self.prior_bins_shift = np.cumsum([0, *self.prior_bins])[:-1]
            self.prior_width = prior_kwargs['width']
            print_once(f'Creating cond. autoregress with prior bins {self.prior_bins}, ')
            print_once(f'dims {self.prior_dims}, ')
            print_once(f'shift {self.prior_bins_shift}')
            print_once(f'input shape {sum(self.prior_dims)}')
            print_once(f'input bins {sum(self.prior_bins)}')
            print_once(f'Self copy is {self.copy_input}')
            self.prime_loss_dims, self.gen_loss_dims = self.prior_dims[0], self.prior_dims[1]
            self.total_loss_dims = self.prime_loss_dims + self.gen_loss_dims
            self.prior = ConditionalAutoregressive2D(input_shape=(sum(self.prior_dims),), bins=sum(self.prior_bins), x_cond=self.x_cond or self.y_cond, y_cond=True, prime_len=self.prime_loss_dims, **prior_kwargs)
        else:
            if self.n_tokens != 0 and self.use_tokens:
                prime_input_shape = self.n_tokens,
                self.prime_loss_dims = np.prod(prime_input_shape)
                self.prime_acts_width, self.prime_state_width = prime_kwargs['width'], prior_kwargs['width']
                self.prime_prior = ConditionalAutoregressive2D(input_shape=prime_input_shape, x_cond=False, y_cond=False, only_encode=True, **prime_kwargs)
                self.prime_state_proj = Conv1D(self.prime_acts_width, self.prime_state_width, init_scale=prime_kwargs['init_scale'])
                self.prime_state_ln = LayerNorm(self.prime_state_width)
                self.prime_bins = prime_kwargs['bins']
                self.prime_x_out = nn.Linear(self.prime_state_width, self.prime_bins, bias=False)
                nn.init.normal_(self.prime_x_out.weight, std=0.02 * prior_kwargs['init_scale'])
            else:
                self.prime_loss_dims = 0
            self.gen_loss_dims = np.prod(self.z_shape)
            self.total_loss_dims = self.prime_loss_dims + self.gen_loss_dims
            self.prior = ConditionalAutoregressive2D(x_cond=self.x_cond or self.y_cond, y_cond=self.y_cond, encoder_dims=self.prime_loss_dims, merged_decoder=merged_decoder, **prior_kwargs)
        self.n_ctx = self.gen_loss_dims
        self.downsamples = calculate_strides(strides_t, downs_t)
        self.cond_downsample = self.downsamples[level + 1] if level != self.levels - 1 else None
        self.raw_to_tokens = np.prod(self.downsamples[:level + 1])
        self.sample_length = self.n_ctx * self.raw_to_tokens
        None

    def get_z_conds(self, zs, start, end):
        if self.level != self.levels - 1:
            assert start % self.cond_downsample == end % self.cond_downsample == 0
            z_cond = zs[self.level + 1][:, start // self.cond_downsample:end // self.cond_downsample]
            assert z_cond.shape[1] == self.n_ctx // self.cond_downsample
            z_conds = [z_cond]
        else:
            z_conds = None
        return z_conds

    def prior_preprocess(self, xs, conds):
        N = xs[0].shape[0]
        for i in range(len(xs)):
            x, shape, dims = xs[i], self.prior_shapes[i], self.prior_dims[i]
            bins, bins_shift = int(self.prior_bins[i]), int(self.prior_bins_shift[i])
            assert isinstance(x, t.cuda.LongTensor), x
            assert (0 <= x).all() and (x < bins).all()
            xs[i] = (xs[i] + bins_shift).view(N, -1)
        for i in range(len(conds)):
            cond, shape, dims = conds[i], self.prior_shapes[i], self.prior_dims[i]
            if cond is not None:
                assert_shape(cond, (N, dims, self.prior_width))
            else:
                conds[i] = t.zeros((N, dims, self.prior_width), dtype=t.float, device='cuda')
        return t.cat(xs, dim=1), t.cat(conds, dim=1)

    def prior_postprocess(self, z):
        N = z.shape[0]
        dims = self.prior_dims[0], z.shape[1] - self.prior_dims[0]
        xs = list(t.split(z, dims, dim=1))
        for i in range(len(xs)):
            shape = self.prior_shapes[i]
            bins, bins_shift = int(self.prior_bins[i]), int(self.prior_bins_shift[i])
            xs[i] = (xs[i] - bins_shift).view(N, -1, *shape[1:])
            xs[i] = t.clamp(xs[i], min=0)
            assert (xs[i] < bins).all(), f'rank: {dist.get_rank()}, bins: {bins}, dims {dims}, shape {shape}, prior_shape {self.prior_shapes}, bins_shift {bins_shift}, xs[i]: {xs[i]}'
        return xs[-1]

    def x_emb(self, z_conds):
        z_conds = z_conds[:self.cond_level - self.level]
        assert len(z_conds) == len(self.conditioner_blocks) == self.cond_level - self.level, f'Expected {len(z_conds)} == {len(self.conditioner_blocks)} == {self.cond_level} - {self.level}'
        x_cond = None
        for z_cond, conditioner_block in reversed(list(zip(z_conds, self.conditioner_blocks))):
            x_cond = conditioner_block(z_cond, x_cond)
        return x_cond

    def encode(self, x, start_level=None, end_level=None, bs_chunks=1):
        if start_level == None:
            start_level = self.level
        if end_level == None:
            end_level = self.levels
        with t.no_grad():
            zs = self.encoder(x, start_level=start_level, end_level=end_level, bs_chunks=bs_chunks)
        return zs

    def decode(self, zs, start_level=None, end_level=None, bs_chunks=1):
        if start_level == None:
            start_level = self.level
        if end_level == None:
            end_level = self.levels
        assert len(zs) == end_level - start_level
        with t.no_grad():
            x_out = self.decoder(zs, start_level=start_level, end_level=end_level, bs_chunks=bs_chunks)
        return x_out

    def get_x_cond(self, z_conds, y_pos, y_video):
        if self.x_cond:
            x_cond = self.x_emb(z_conds)
        else:
            x_cond = y_pos
            if self.video_clip_emb:
                frames_num = y_video.shape[1]
                token2frame = np.linspace(0, 1, num=self.n_time) * (frames_num - 1)
                token2frame = np.round(token2frame)
                y_video_cond = y_video[:, token2frame]
                x_cond += y_video_cond
        return x_cond

    def get_cond(self, z_conds, y, inference=False):
        if y is not None:
            if self.clip_emb:
                if self.video_clip_emb:
                    assert y.shape[2] == 3 + 512 + self.n_tokens, f'Expected {3} + {512} + {self.n_tokens}, got {y.shape[1]}'
                else:
                    assert y.shape[1] == 3 + 512 + self.n_tokens, f'Expected {3} + {512} + {self.n_tokens}, got {y.shape[1]}'
            else:
                assert y.shape[1] == 4 + self.y_emb.max_bow_genre_size + self.n_tokens, f'Expected {4} + {self.y_emb.max_bow_genre_size} + {self.n_tokens}, got {y.shape[1]}'
            n_labels = y.shape[1] - self.n_tokens
            y, prime = y[:, :n_labels], y[:, n_labels:]
        else:
            y, prime = None, None
        if inference and self.y_emb.class_free_guidance_prob >= 0:
            y_cond_label, y_pos_label, y_video_label = self.y_emb(y, mode=Condition_Modes.label) if self.y_cond else (None, None, None)
            y_cond_null_label, y_pos_null_label, y_video_null_label = self.y_emb(y, mode=Condition_Modes.null_label) if self.y_cond else (None, None, None)
            y_cond, y_pos, y_video = [y_cond_label, y_cond_null_label], [y_pos_label, y_pos_null_label], [y_video_label, y_video_null_label]
            x_cond = [self.get_x_cond(z_conds, y_pos[i], y_video[i]) for i in range(2)]
        else:
            y_cond, y_pos, y_video = self.y_emb(y) if self.y_cond else (None, None, None)
            x_cond = self.get_x_cond(z_conds, y_pos, y_video)
        return x_cond, y_cond, prime

    def sample(self, n_samples, z=None, z_conds=None, y=None, fp16=False, temp=1.0, top_k=0, top_p=0.0, chunk_size=None, sample_tokens=None, cfg_s=0):
        N = n_samples
        if z is not None:
            assert z.shape[0] == N, f'Expected shape ({N},**), got shape {z.shape}'
        if y is not None:
            assert y.shape[0] == N, f'Expected shape ({N},**), got shape {y.shape}'
        if z_conds is not None:
            for z_cond in z_conds:
                assert z_cond.shape[0] == N, f'Expected shape ({N},**), got shape {z_cond.shape}'
        no_past_context = z is None or z.shape[1] == 0
        if dist.get_rank() == 0:
            name = {(True): 'Ancestral', (False): 'Primed'}[no_past_context]
            None
        with t.no_grad():
            x_cond, y_cond, prime = self.get_cond(z_conds, y, inference=True)
            if self.single_enc_dec:
                if no_past_context:
                    z, x_cond = self.prior_preprocess([prime], [None, x_cond])
                else:
                    z, x_cond = self.prior_preprocess([prime, z], [None, x_cond])
                if sample_tokens is not None:
                    sample_tokens += self.n_tokens
                z = self.prior.primed_sample(n_samples, z, x_cond, y_cond, fp16=fp16, temp=temp, top_k=top_k, top_p=top_p, chunk_size=chunk_size, sample_tokens=sample_tokens)
                z = self.prior_postprocess(z)
            else:
                encoder_kv = self.get_encoder_kv(prime, fp16=fp16, sample=True)
                if no_past_context:
                    z = self.prior.sample(n_samples, x_cond, y_cond, encoder_kv, fp16=fp16, temp=temp, top_k=top_k, top_p=top_p, sample_tokens=sample_tokens, cfg_s=cfg_s)
                else:
                    z = self.prior.primed_sample(n_samples, z, x_cond, y_cond, encoder_kv, fp16=fp16, temp=temp, top_k=top_k, top_p=top_p, chunk_size=chunk_size, sample_tokens=sample_tokens, cfg_s=cfg_s)
            if sample_tokens is None:
                assert_shape(z, (N, *self.z_shape))
        return z

    def get_encoder_kv(self, prime, fp16=False, sample=False):
        if self.n_tokens != 0 and self.use_tokens:
            if sample:
                self.prime_prior
            N = prime.shape[0]
            prime_acts = self.prime_prior(prime, None, None, None, fp16=fp16)
            assert_shape(prime_acts, (N, self.prime_loss_dims, self.prime_acts_width))
            assert prime_acts.dtype == t.float, f'Expected t.float, got {prime_acts.dtype}'
            encoder_kv = self.prime_state_ln(self.prime_state_proj(prime_acts))
            assert encoder_kv.dtype == t.float, f'Expected t.float, got {encoder_kv.dtype}'
            if sample:
                self.prime_prior.cpu()
                if fp16:
                    encoder_kv = encoder_kv.half()
        else:
            encoder_kv = None
        return encoder_kv

    def get_prime_loss(self, encoder_kv, prime_t):
        if self.use_tokens:
            encoder_kv = encoder_kv.float()
            encoder_kv = self.prime_x_out(encoder_kv)
            prime_loss = nn.functional.cross_entropy(encoder_kv.view(-1, self.prime_bins), prime_t.view(-1)) / np.log(2.0)
        elif use_cuda:
            prime_loss = t.tensor(0.0, device='cuda')
        else:
            prime_loss = t.tensor(0.0)
        return prime_loss

    def z_forward(self, z, z_conds=[], y=None, fp16=False, get_preds=False, get_attn_weights=False):
        """
        Arguments:
            get_attn_weights (bool or set): Makes forward prop dump
                self-attention softmaxes to self.prior.transformer.ws. Either a
                set of layer indices indicating which layers to store, or a
                boolean value indicating whether to dump all.
        """
        assert isinstance(get_attn_weights, (bool, set))
        if get_attn_weights:
            self.prior.transformer.set_record_attn(get_attn_weights)
        x_cond, y_cond, prime = self.get_cond(z_conds, y)
        if self.copy_input:
            prime = z[:, :self.n_tokens]
        if self.single_enc_dec:
            z, x_cond = self.prior_preprocess([prime, z], [None, x_cond])
            (prime_loss, gen_loss), preds = self.prior(z, x_cond, y_cond, fp16=fp16, get_sep_loss=True, get_preds=get_preds)
        else:
            encoder_kv = self.get_encoder_kv(prime, fp16=fp16)
            prime_loss = self.get_prime_loss(encoder_kv, prime)
            gen_loss, preds = self.prior(z, x_cond, y_cond, encoder_kv, fp16=fp16, get_preds=get_preds)
        loss = self.prime_loss_fraction * prime_loss * self.prime_loss_dims / self.total_loss_dims + gen_loss * self.gen_loss_dims / self.total_loss_dims
        metrics = dict(bpd=gen_loss.clone().detach(), prime_loss=prime_loss.clone().detach(), gen_loss=gen_loss.clone().detach())
        if get_preds:
            metrics['preds'] = preds.clone().detach()
        if get_attn_weights:
            ws = self.prior.transformer.ws
            self.prior.transformer.set_record_attn(False)
            return ws
        else:
            return loss, metrics

    def forward(self, x, y=None, fp16=False, decode=False, get_preds=False):
        bs = x.shape[0]
        z, *z_conds = self.encode(x, bs_chunks=bs)
        loss, metrics = self.z_forward(z=z, z_conds=z_conds, y=y, fp16=fp16, get_preds=get_preds)
        if decode:
            x_out = self.decode([z, *z_conds])
        else:
            x_out = None
        return x_out, loss, metrics


class Mask(nn.Module):

    def __init__(self, n_ctx):
        super().__init__()
        self.register_buffer('b', t.tril(t.ones(n_ctx, n_ctx)).view(1, 1, n_ctx, n_ctx))

    def forward(self, w):
        w = w * self.b + -1000000000.0 * (1 - self.b)
        return w


class BottleneckBlock(nn.Module):

    def __init__(self, k_bins, emb_width, mu):
        super().__init__()
        self.k_bins = k_bins
        self.emb_width = emb_width
        self.mu = mu
        self.reset_k()
        self.threshold = 1.0

    def reset_k(self):
        self.init = False
        self.k_sum = None
        self.k_elem = None
        if use_cuda:
            self.register_buffer('k', t.zeros(self.k_bins, self.emb_width))
        else:
            self.register_buffer('k', t.zeros(self.k_bins, self.emb_width))

    def _tile(self, x):
        d, ew = x.shape
        if d < self.k_bins:
            n_repeats = (self.k_bins + d - 1) // d
            std = 0.01 / np.sqrt(ew)
            x = x.repeat(n_repeats, 1)
            x = x + t.randn_like(x) * std
        return x

    def init_k(self, x):
        mu, emb_width, k_bins = self.mu, self.emb_width, self.k_bins
        self.init = True
        y = self._tile(x)
        _k_rand = y[t.randperm(y.shape[0])][:k_bins]
        dist.broadcast(_k_rand, 0)
        self.k = _k_rand
        assert self.k.shape == (k_bins, emb_width)
        self.k_sum = self.k
        self.k_elem = t.ones(k_bins, device=self.k.device)

    def restore_k(self, num_tokens=None, threshold=1.0):
        mu, emb_width, k_bins = self.mu, self.emb_width, self.k_bins
        self.init = True
        assert self.k.shape == (k_bins, emb_width)
        self.k_sum = self.k.clone()
        self.k_elem = t.ones(k_bins, device=self.k.device)
        if num_tokens is not None:
            expected_usage = num_tokens / k_bins
            self.k_elem.data.mul_(expected_usage)
            self.k_sum.data.mul_(expected_usage)
        self.threshold = threshold

    def update_k(self, x, x_l):
        mu, emb_width, k_bins = self.mu, self.emb_width, self.k_bins
        with t.no_grad():
            x_l_onehot = t.zeros(k_bins, x.shape[0], device=x.device)
            x_l_onehot.scatter_(0, x_l.view(1, x.shape[0]), 1)
            _k_sum = t.matmul(x_l_onehot, x)
            _k_elem = x_l_onehot.sum(dim=-1)
            y = self._tile(x)
            _k_rand = y[t.randperm(y.shape[0])][:k_bins]
            dist.broadcast(_k_rand, 0)
            dist.all_reduce(_k_sum)
            dist.all_reduce(_k_elem)
            old_k = self.k
            self.k_sum = mu * self.k_sum + (1.0 - mu) * _k_sum
            self.k_elem = mu * self.k_elem + (1.0 - mu) * _k_elem
            usage = (self.k_elem.view(k_bins, 1) >= self.threshold).float()
            self.k = usage * (self.k_sum.view(k_bins, emb_width) / self.k_elem.view(k_bins, 1)) + (1 - usage) * _k_rand
            _k_prob = _k_elem / t.sum(_k_elem)
            entropy = -t.sum(_k_prob * t.log(_k_prob + 1e-08))
            used_curr = (_k_elem >= self.threshold).sum()
            usage = t.sum(usage)
            dk = t.norm(self.k - old_k) / np.sqrt(np.prod(old_k.shape))
        return dict(entropy=entropy, used_curr=used_curr, usage=usage, dk=dk)

    def preprocess(self, x):
        x = x.permute(0, 2, 1).contiguous()
        x = x.view(-1, x.shape[-1])
        if x.shape[-1] == self.emb_width:
            prenorm = t.norm(x - t.mean(x)) / np.sqrt(np.prod(x.shape))
        elif x.shape[-1] == 2 * self.emb_width:
            x1, x2 = x[..., :self.emb_width], x[..., self.emb_width:]
            prenorm = t.norm(x1 - t.mean(x1)) / np.sqrt(np.prod(x1.shape)) + t.norm(x2 - t.mean(x2)) / np.sqrt(np.prod(x2.shape))
            x = x1 + x2
        else:
            assert False, f'Expected {x.shape[-1]} to be (1 or 2) * {self.emb_width}'
        return x, prenorm

    def postprocess(self, x_l, x_d, x_shape):
        N, T = x_shape
        x_d = x_d.view(N, T, -1).permute(0, 2, 1).contiguous()
        x_l = x_l.view(N, T)
        return x_l, x_d

    def quantise(self, x):
        k_w = self.k.t()
        distance = t.sum(x ** 2, dim=-1, keepdim=True) - 2 * t.matmul(x, k_w) + t.sum(k_w ** 2, dim=0, keepdim=True)
        min_distance, x_l = t.min(distance, dim=-1)
        fit = t.mean(min_distance)
        return x_l, fit

    def dequantise(self, x_l):
        x = F.embedding(x_l, self.k)
        return x

    def encode(self, x):
        N, width, T = x.shape
        x, prenorm = self.preprocess(x)
        x_l, fit = self.quantise(x)
        x_l = x_l.view(N, T)
        return x_l

    def decode(self, x_l):
        N, T = x_l.shape
        width = self.emb_width
        x_d = self.dequantise(x_l)
        x_d = x_d.view(N, T, width).permute(0, 2, 1).contiguous()
        return x_d

    def forward(self, x, update_k=True):
        N, width, T = x.shape
        x, prenorm = self.preprocess(x)
        if update_k and not self.init:
            self.init_k(x)
        x_l, fit = self.quantise(x)
        x_d = self.dequantise(x_l)
        if update_k:
            update_metrics = self.update_k(x, x_l)
        else:
            update_metrics = {}
        commit_loss = t.norm(x_d.detach() - x) ** 2 / np.prod(x.shape)
        x_d = x + (x_d - x).detach()
        x_l, x_d = self.postprocess(x_l, x_d, (N, T))
        return x_l, x_d, commit_loss, dict(fit=fit, pn=prenorm, **update_metrics)


class Bottleneck(nn.Module):

    def __init__(self, l_bins, emb_width, mu, levels):
        super().__init__()
        self.levels = levels
        level_block = lambda level: BottleneckBlock(l_bins, emb_width, mu)
        self.level_blocks = nn.ModuleList()
        for level in range(self.levels):
            self.level_blocks.append(level_block(level))

    def encode(self, xs):
        zs = [level_block.encode(x) for level_block, x in zip(self.level_blocks, xs)]
        return zs

    def decode(self, zs, start_level=0, end_level=None):
        if end_level is None:
            end_level = self.levels
        xs_quantised = [level_block.decode(z) for level_block, z in zip(self.level_blocks[start_level:end_level], zs)]
        return xs_quantised

    def forward(self, xs):
        zs, xs_quantised, commit_losses, metrics = [], [], [], []
        for level in range(self.levels):
            level_block = self.level_blocks[level]
            x = xs[level]
            z, x_quantised, commit_loss, metric = level_block(x, update_k=self.training)
            zs.append(z)
            if not self.training:
                x_quantised = x_quantised.detach()
            xs_quantised.append(x_quantised)
            commit_losses.append(commit_loss)
            if self.training:
                metrics.append(metric)
        return zs, xs_quantised, commit_losses, metrics


class NoBottleneckBlock(nn.Module):

    def restore_k(self):
        pass


class NoBottleneck(nn.Module):

    def __init__(self, levels):
        super().__init__()
        self.level_blocks = nn.ModuleList()
        self.levels = levels
        for level in range(levels):
            self.level_blocks.append(NoBottleneckBlock())

    def encode(self, xs):
        return xs

    def decode(self, zs, start_level=0, end_level=None):
        if end_level is None:
            end_level = self.levels
        return zs

    def forward(self, xs):
        zero = t.zeros(())
        commit_losses = [zero for _ in range(self.levels)]
        metrics = [dict(entropy=zero, usage=zero, used_curr=zero, pn=zero, dk=zero) for _ in range(self.levels)]
        return xs, xs, commit_losses, metrics


class EncoderConvBlock(nn.Module):

    def __init__(self, input_emb_width, output_emb_width, down_t, stride_t, width, depth, m_conv, dilation_growth_rate=1, dilation_cycle=None, zero_out=False, res_scale=False):
        super().__init__()
        blocks = []
        filter_t, pad_t = stride_t * 2, stride_t // 2
        if down_t > 0:
            for i in range(down_t):
                block = nn.Sequential(nn.Conv1d(input_emb_width if i == 0 else width, width, filter_t, stride_t, pad_t), Resnet1D(width, depth, m_conv, dilation_growth_rate, dilation_cycle, zero_out, res_scale))
                blocks.append(block)
            block = nn.Conv1d(width, output_emb_width, 3, 1, 1)
            blocks.append(block)
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)


class Encoder(nn.Module):

    def __init__(self, input_emb_width, output_emb_width, levels, downs_t, strides_t, **block_kwargs):
        super().__init__()
        self.input_emb_width = input_emb_width
        self.output_emb_width = output_emb_width
        self.levels = levels
        self.downs_t = downs_t
        self.strides_t = strides_t
        block_kwargs_copy = dict(**block_kwargs)
        if 'reverse_decoder_dilation' in block_kwargs_copy:
            del block_kwargs_copy['reverse_decoder_dilation']
        level_block = lambda level, down_t, stride_t: EncoderConvBlock(input_emb_width if level == 0 else output_emb_width, output_emb_width, down_t, stride_t, **block_kwargs_copy)
        self.level_blocks = nn.ModuleList()
        iterator = zip(list(range(self.levels)), downs_t, strides_t)
        for level, down_t, stride_t in iterator:
            self.level_blocks.append(level_block(level, down_t, stride_t))

    def forward(self, x):
        N, T = x.shape[0], x.shape[-1]
        emb = self.input_emb_width
        assert_shape(x, (N, emb, T))
        xs = []
        iterator = zip(list(range(self.levels)), self.downs_t, self.strides_t)
        for level, down_t, stride_t in iterator:
            level_block = self.level_blocks[level]
            x = level_block(x)
            emb, T = self.output_emb_width, T // stride_t ** down_t
            assert_shape(x, (N, emb, T))
            xs.append(x)
        return xs


class Decoder(nn.Module):

    def __init__(self, input_emb_width, output_emb_width, levels, downs_t, strides_t, **block_kwargs):
        super().__init__()
        self.input_emb_width = input_emb_width
        self.output_emb_width = output_emb_width
        self.levels = levels
        self.downs_t = downs_t
        self.strides_t = strides_t
        level_block = lambda level, down_t, stride_t: DecoderConvBock(output_emb_width, output_emb_width, down_t, stride_t, **block_kwargs)
        self.level_blocks = nn.ModuleList()
        iterator = zip(list(range(self.levels)), downs_t, strides_t)
        for level, down_t, stride_t in iterator:
            self.level_blocks.append(level_block(level, down_t, stride_t))
        self.out = nn.Conv1d(output_emb_width, input_emb_width, 3, 1, 1)

    def forward(self, xs, all_levels=True):
        if all_levels:
            assert len(xs) == self.levels
        else:
            assert len(xs) == 1
        x = xs[-1]
        N, T = x.shape[0], x.shape[-1]
        emb = self.output_emb_width
        assert_shape(x, (N, emb, T))
        iterator = reversed(list(zip(list(range(self.levels)), self.downs_t, self.strides_t)))
        for level, down_t, stride_t in iterator:
            level_block = self.level_blocks[level]
            x = level_block(x)
            emb, T = self.output_emb_width, T * stride_t ** down_t
            assert_shape(x, (N, emb, T))
            if level != 0 and all_levels:
                x = x + xs[level - 1]
        x = self.out(x)
        return x


class ResConvBlock(nn.Module):

    def __init__(self, n_in, n_state):
        super().__init__()
        self.model = nn.Sequential(nn.ReLU(), nn.Conv2d(n_in, n_state, 3, 1, 1), nn.ReLU(), nn.Conv2d(n_state, n_in, 1, 1, 0))

    def forward(self, x):
        return x + self.model(x)


class Resnet(nn.Module):

    def __init__(self, n_in, n_depth, m_conv=1.0):
        super().__init__()
        self.model = nn.Sequential(*[ResConvBlock(n_in, int(m_conv * n_in)) for _ in range(n_depth)])

    def forward(self, x):
        return self.model(x)


def _loss_fn(loss_fn, x_target, x_pred, hps):
    if loss_fn == 'l1':
        return t.mean(t.abs(x_pred - x_target)) / hps.bandwidth['l1']
    elif loss_fn == 'l2':
        return t.mean((x_pred - x_target) ** 2) / hps.bandwidth['l2']
    elif loss_fn == 'linf':
        residual = ((x_pred - x_target) ** 2).reshape(x_target.shape[0], -1)
        values, _ = t.topk(residual, hps.linf_k, dim=1)
        return t.mean(values) / hps.bandwidth['l2']
    elif loss_fn == 'lmix':
        loss = 0.0
        if hps.lmix_l1:
            loss += hps.lmix_l1 * _loss_fn('l1', x_target, x_pred, hps)
        if hps.lmix_l2:
            loss += hps.lmix_l2 * _loss_fn('l2', x_target, x_pred, hps)
        if hps.lmix_linf:
            loss += hps.lmix_linf * _loss_fn('linf', x_target, x_pred, hps)
        return loss
    else:
        assert False, f'Unknown loss_fn {loss_fn}'


def audio_postprocess(x, hps):
    return x


def average_metrics(_metrics):
    metrics = {}
    for _metric in _metrics:
        for key, val in _metric.items():
            if key not in metrics:
                metrics[key] = []
            metrics[key].append(val)
    return {key: (sum(vals) / len(vals)) for key, vals in metrics.items()}


class STFTValues:

    def __init__(self, hps, n_fft, hop_length, window_size):
        self.sr = hps.sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.window_size = window_size


def norm(x):
    return (x.view(x.shape[0], -1) ** 2).sum(dim=-1).sqrt()


def stft(sig, hps):
    return t.stft(sig, hps.n_fft, hps.hop_length, win_length=hps.window_size, window=t.hann_window(hps.window_size, device=sig.device))


def spec(x, hps):
    return t.norm(stft(x, hps), p=2, dim=-1)


def squeeze(x):
    if len(x.shape) == 3:
        assert x.shape[-1] in [1, 2]
        x = t.mean(x, -1)
    if len(x.shape) != 2:
        raise ValueError(f'Unknown input shape {x.shape}')
    return x


def multispectral_loss(x_in, x_out, hps):
    losses = []
    assert len(hps.multispec_loss_n_fft) == len(hps.multispec_loss_hop_length) == len(hps.multispec_loss_window_size)
    args = [hps.multispec_loss_n_fft, hps.multispec_loss_hop_length, hps.multispec_loss_window_size]
    for n_fft, hop_length, window_size in zip(*args):
        hps = STFTValues(hps, n_fft, hop_length, window_size)
        spec_in = spec(squeeze(x_in.float()), hps)
        spec_out = spec(squeeze(x_out.float()), hps)
        losses.append(norm(spec_in - spec_out))
    return sum(losses) / len(losses)


class DefaultSTFTValues:

    def __init__(self, hps):
        self.sr = hps.sr
        self.n_fft = 2048
        self.hop_length = 256
        self.window_size = 6 * self.hop_length


def spectral_convergence(x_in, x_out, hps, epsilon=0.002):
    hps = DefaultSTFTValues(hps)
    spec_in = spec(squeeze(x_in.float()), hps)
    spec_out = spec(squeeze(x_out.float()), hps)
    gt_norm = norm(spec_in)
    residual_norm = norm(spec_in - spec_out)
    mask = (gt_norm > epsilon).float()
    return residual_norm * mask / t.clamp(gt_norm, min=epsilon)


def spectral_loss(x_in, x_out, hps):
    hps = DefaultSTFTValues(hps)
    spec_in = spec(squeeze(x_in.float()), hps)
    spec_out = spec(squeeze(x_out.float()), hps)
    return norm(spec_in - spec_out)


class VQVAE(nn.Module):

    def __init__(self, input_shape, levels, downs_t, strides_t, emb_width, l_bins, mu, commit, spectral, multispectral, multipliers=None, use_bottleneck=True, **block_kwargs):
        super().__init__()
        self.sample_length = input_shape[0]
        x_shape, x_channels = input_shape[:-1], input_shape[-1]
        self.x_shape = x_shape
        self.downsamples = calculate_strides(strides_t, downs_t)
        self.hop_lengths = np.cumprod(self.downsamples)
        self.z_shapes = z_shapes = [(x_shape[0] // self.hop_lengths[level],) for level in range(levels)]
        self.levels = levels
        self.emb_width = emb_width
        if multipliers is None:
            self.multipliers = [1] * levels
        else:
            assert len(multipliers) == levels, 'Invalid number of multipliers'
            self.multipliers = multipliers

        def _block_kwargs(level):
            this_block_kwargs = dict(block_kwargs)
            this_block_kwargs['width'] *= self.multipliers[level]
            this_block_kwargs['depth'] *= self.multipliers[level]
            return this_block_kwargs
        encoder = lambda level: Encoder(x_channels, emb_width, level + 1, downs_t[:level + 1], strides_t[:level + 1], **_block_kwargs(level))
        decoder = lambda level: Decoder(x_channels, emb_width, level + 1, downs_t[:level + 1], strides_t[:level + 1], **_block_kwargs(level))
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for level in range(levels):
            self.encoders.append(encoder(level))
            self.decoders.append(decoder(level))
        if use_bottleneck:
            self.bottleneck = Bottleneck(l_bins, emb_width, mu, levels)
        else:
            self.bottleneck = NoBottleneck(levels)
        self.downs_t = downs_t
        self.strides_t = strides_t
        self.l_bins = l_bins
        self.commit = commit
        self.spectral = spectral
        self.multispectral = multispectral

    def preprocess(self, x):
        assert len(x.shape) == 3
        x = x.permute(0, 2, 1).float()
        return x

    def postprocess(self, x):
        x = x.permute(0, 2, 1)
        return x

    def _decode(self, zs, start_level=0, end_level=None):
        if end_level is None:
            end_level = self.levels
        assert len(zs) == end_level - start_level
        xs_quantised = self.bottleneck.decode(zs, start_level=start_level, end_level=end_level)
        assert len(xs_quantised) == end_level - start_level
        decoder, x_quantised = self.decoders[start_level], xs_quantised[0:1]
        x_out = decoder(x_quantised, all_levels=False)
        x_out = self.postprocess(x_out)
        return x_out

    def decode(self, zs, start_level=0, end_level=None, bs_chunks=1):
        z_chunks = [t.chunk(z, bs_chunks, dim=0) for z in zs]
        x_outs = []
        for i in range(bs_chunks):
            zs_i = [z_chunk[i] for z_chunk in z_chunks]
            x_out = self._decode(zs_i, start_level=start_level, end_level=end_level)
            x_outs.append(x_out)
        return t.cat(x_outs, dim=0)

    def _encode_noBottleneck(self, x, start_level=0, end_level=None):
        if end_level is None:
            end_level = self.levels
        x_in = self.preprocess(x)
        xs = []
        for level in range(self.levels):
            encoder = self.encoders[level]
            x_out = encoder(x_in)
            xs.append(x_out[-1])
        return xs

    def _encode(self, x, start_level=0, end_level=None):
        if end_level is None:
            end_level = self.levels
        x_in = self.preprocess(x)
        xs = []
        for level in range(self.levels):
            encoder = self.encoders[level]
            x_out = encoder(x_in)
            xs.append(x_out[-1])
        zs = self.bottleneck.encode(xs)
        return zs[start_level:end_level]

    def encode(self, x, start_level=0, end_level=None, bs_chunks=1):
        x_chunks = t.chunk(x, bs_chunks, dim=0)
        zs_list = []
        for x_i in x_chunks:
            zs_i = self._encode(x_i, start_level=start_level, end_level=end_level)
            zs_list.append(zs_i)
        zs = [t.cat(zs_level_list, dim=0) for zs_level_list in zip(*zs_list)]
        return zs

    def sample(self, n_samples):
        if use_cuda:
            zs = [t.randint(0, self.l_bins, size=(n_samples, *z_shape), device='cuda') for z_shape in self.z_shapes]
        else:
            zs = [t.randint(0, self.l_bins, size=(n_samples, *z_shape)) for z_shape in self.z_shapes]
        return self.decode(zs)

    def forward(self, x, hps, loss_fn='l1'):
        metrics = {}
        N = x.shape[0]
        x_in = self.preprocess(x)
        xs = []
        for level in range(self.levels):
            encoder = self.encoders[level]
            x_out = encoder(x_in)
            xs.append(x_out[-1])
        zs, xs_quantised, commit_losses, quantiser_metrics = self.bottleneck(xs)
        x_outs = []
        for level in range(self.levels):
            decoder = self.decoders[level]
            x_out = decoder(xs_quantised[level:level + 1], all_levels=False)
            assert_shape(x_out, x_in.shape)
            x_outs.append(x_out)

        def _spectral_loss(x_target, x_out, hps):
            if hps.use_nonrelative_specloss:
                sl = spectral_loss(x_target, x_out, hps) / hps.bandwidth['spec']
            else:
                sl = spectral_convergence(x_target, x_out, hps)
            sl = t.mean(sl)
            return sl

        def _multispectral_loss(x_target, x_out, hps):
            sl = multispectral_loss(x_target, x_out, hps) / hps.bandwidth['spec']
            sl = t.mean(sl)
            return sl
        recons_loss = t.zeros(())
        spec_loss = t.zeros(())
        multispec_loss = t.zeros(())
        x_target = audio_postprocess(x.float(), hps)
        for level in reversed(range(self.levels)):
            x_out = self.postprocess(x_outs[level])
            x_out = audio_postprocess(x_out, hps)
            this_recons_loss = _loss_fn(loss_fn, x_target, x_out, hps)
            this_spec_loss = _spectral_loss(x_target, x_out, hps)
            this_multispec_loss = _multispectral_loss(x_target, x_out, hps)
            metrics[f'recons_loss_l{level + 1}'] = this_recons_loss
            metrics[f'spectral_loss_l{level + 1}'] = this_spec_loss
            metrics[f'multispectral_loss_l{level + 1}'] = this_multispec_loss
            recons_loss += this_recons_loss
            spec_loss += this_spec_loss
            multispec_loss += this_multispec_loss
        commit_loss = sum(commit_losses)
        loss = recons_loss + self.spectral * spec_loss + self.multispectral * multispec_loss + self.commit * commit_loss
        with t.no_grad():
            sc = t.mean(spectral_convergence(x_target, x_out, hps))
            l2_loss = _loss_fn('l2', x_target, x_out, hps)
            l1_loss = _loss_fn('l1', x_target, x_out, hps)
            linf_loss = _loss_fn('linf', x_target, x_out, hps)
        quantiser_metrics = average_metrics(quantiser_metrics)
        metrics.update(dict(recons_loss=recons_loss, spectral_loss=spec_loss, multispectral_loss=multispec_loss, spectral_convergence=sc, l2_loss=l2_loss, l1_loss=l1_loss, linf_loss=linf_loss, commit_loss=commit_loss, **quantiser_metrics))
        for key, val in metrics.items():
            metrics[key] = val.detach()
        return x_out, loss, metrics


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (Bottleneck,
     lambda: ([], {'l_bins': 4, 'emb_width': 4, 'mu': 4, 'levels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Conv1D,
     lambda: ([], {'n_in': 4, 'n_out': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (DecoderConvBock,
     lambda: ([], {'input_emb_width': 4, 'output_emb_width': 4, 'down_t': 4, 'stride_t': 1, 'width': 4, 'depth': 1, 'm_conv': 4}),
     lambda: ([torch.rand([4, 4])], {})),
    (FactoredAttention,
     lambda: ([], {'n_in': 4, 'n_ctx': 4, 'n_state': 4, 'n_head': 4}),
     lambda: ([torch.rand([4, 4, 4])], {})),
    (MLP,
     lambda: ([], {'n_in': 4, 'n_state': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Mask,
     lambda: ([], {'n_ctx': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (NoBottleneck,
     lambda: ([], {'levels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (PositionEmbedding,
     lambda: ([], {'input_shape': 4, 'width': 4}),
     lambda: ([], {})),
    (ResConv1DBlock,
     lambda: ([], {'n_in': 4, 'n_state': 4}),
     lambda: ([torch.rand([4, 4])], {})),
    (ResConvBlock,
     lambda: ([], {'n_in': 4, 'n_state': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Resnet,
     lambda: ([], {'n_in': 4, 'n_depth': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Resnet1D,
     lambda: ([], {'n_in': 4, 'n_depth': 1}),
     lambda: ([torch.rand([4, 4])], {})),
]

