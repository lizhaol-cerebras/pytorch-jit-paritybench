
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


import uuid


import torch


import numpy as np


import torch.nn.functional as F


import copy


import torch.nn as nn


import math


import time


from random import shuffle


class AttentionError(Exception):
    pass


def d(tensor=None):
    if tensor is None:
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    return 'cuda' if tensor.is_cuda else 'cpu'


class MultiheadedAttention(nn.Module):
    """
    Narrow multiheaded attention. Each attention head inspects a 
    fraction of the embedding space and expresses attention vectors for each sequence position as a weighted average of all (earlier) positions.
    """

    def __init__(self, d_model, heads=8, dropout=0.1, relative_pos=True):
        super().__init__()
        if d_model % heads != 0:
            raise AttentionError('Number of heads does not divide model dimension')
        self.d_model = d_model
        self.heads = heads
        s = d_model // heads
        self.linears = torch.nn.ModuleList([nn.Linear(s, s, bias=False) for i in range(3)])
        self.recombine_heads = nn.Linear(heads * s, d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.max_length = 1024
        self.relative_pos = relative_pos
        if relative_pos:
            self.Er = torch.randn([heads, self.max_length, s], device=d())
        else:
            self.Er = None

    def forward(self, x, mask):
        b, t, e = x.size()
        h = self.heads
        s = e // h
        embedding_start = self.max_length - t
        x = x.view(b, t, h, s)
        queries, keys, values = [w(x).transpose(1, 2) for w, x in zip(self.linears, (x, x, x))]
        if self.relative_pos:
            Er = self.Er[:, embedding_start:, :].unsqueeze(0)
            QEr = torch.matmul(queries, Er.transpose(-1, -2))
            QEr = self._mask_positions(QEr)
            SRel = self._skew(QEr).contiguous().view(b * h, t, t)
        else:
            SRel = torch.zeros([b * h, t, t], device=d())
        queries, keys, values = map(lambda x: x.contiguous().view(b * h, t, s), (queries, keys, values))
        queries = queries / e ** (1 / 4)
        keys = keys / e ** (1 / 4)
        scores = torch.bmm(queries, keys.transpose(1, 2))
        scores = scores + SRel
        subsequent_mask = torch.triu(torch.ones(1, t, t, device=d()), 1)
        scores = scores.masked_fill(subsequent_mask == 1, -1000000000.0)
        if mask is not None:
            mask = mask.repeat_interleave(h, 0)
            wtf = (mask == 0).nonzero().transpose(0, 1)
            scores[wtf[0], wtf[1], :] = -1000000000.0
        attn_probs = F.softmax(scores, dim=2)
        attn_probs = self.dropout(attn_probs)
        out = torch.bmm(attn_probs, values).view(b, h, t, s)
        out = out.transpose(1, 2).contiguous().view(b, t, s * h)
        return self.recombine_heads(out)

    def _mask_positions(self, qe):
        L = qe.shape[-1]
        mask = torch.triu(torch.ones(L, L, device=d()), 1).flip(1)
        return qe.masked_fill(mask == 1, 0)

    def _skew(self, qe):
        padded_qe = F.pad(qe, [1, 0])
        s = padded_qe.shape
        padded_qe = padded_qe.view(s[0], s[1], s[3], s[2])
        return padded_qe[:, :, 1:, :]


class PositionwiseFeedForward(nn.Module):

    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class DecoderLayer(nn.Module):

    def __init__(self, size, n_heads, d_feedforward, dropout, relative_pos):
        super().__init__()
        self.self_attn = MultiheadedAttention(size, n_heads, dropout, relative_pos)
        self.feed_forward = PositionwiseFeedForward(size, d_feedforward, dropout)
        self.size = size
        self.norm1 = nn.LayerNorm(size)
        self.norm2 = nn.LayerNorm(size)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask):
        attn = self.self_attn(x, mask)
        x = x + self.dropout1(attn)
        x = self.norm1(x)
        ff = self.feed_forward(x)
        x = x + self.dropout2(ff)
        x = self.norm2(x)
        return x


class MusicTransformerError(Exception):
    pass


class SequenceEmbedding(nn.Module):
    """
    Standard embedding, scaled by the sqrt of model's hidden state size
    """

    def __init__(self, vocab_size, model_size):
        super().__init__()
        self.d_model = model_size
        self.emb = nn.Embedding(vocab_size, model_size)

    def forward(self, x):
        return self.emb(x) * math.sqrt(self.d_model)


def clones(module, N):
    """Clone N identical layers of a module"""
    return torch.nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class MusicTransformer(nn.Module):
    """Generative, autoregressive transformer model. Train on a 
    dataset of encoded musical sequences."""

    def __init__(self, n_tokens, seq_length=None, d_model=64, n_heads=4, depth=2, d_feedforward=512, dropout=0.1, positional_encoding=False, relative_pos=True):
        """
        Args:
            n_tokens: number of commands/states in encoded musical sequence
            seq_length: length of (padded) input/target sequences
            d_model: dimensionality of embedded sequences
            n_heads: number of attention heads
            depth: number of stacked transformer layers
            d_feedforward: dimensionality of dense sublayer 
            dropout: probability of dropout in dropout sublayer
            relative_pos: (bool) if True, use relative positional embeddings
        """
        super().__init__()
        self.n_tokens = n_tokens
        self.d_model = d_model
        self.embed = SequenceEmbedding(n_tokens, d_model)
        self.positional_encoding = positional_encoding
        if self.positional_encoding:
            pos = torch.zeros(5000, d_model)
            position = torch.arange(5000).unsqueeze(1)
            div_term = torch.exp(torch.arange(0.0, d_model, 2) * -(math.log(10000.0) / d_model))
            pos[0:, 0::2] = torch.sin(position * div_term)
            pos[0:, 1::2] = torch.cos(position * div_term)
            pos = pos.unsqueeze(0)
            pos = pos
            self.register_buffer('pos', pos)
        else:
            if seq_length == None:
                raise MusicTransformerError('seq_length not provided for positional embeddings')
            self.pos = nn.Embedding(seq_length, d_model)
        self.to_scores = nn.Linear(d_model, n_tokens)
        self.layers = clones(DecoderLayer(d_model, n_heads, d_feedforward, dropout, relative_pos), depth)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        x = self.embed(x)
        b, t, e = x.size()
        if self.positional_encoding:
            positions = self.pos[:, :t, :]
        else:
            positions = self.pos(torch.arange(t, device=d()))[None, :, :].expand(b, t, e)
        x = x + positions
        for layer in self.layers:
            x = layer(x, mask)
        z = self.norm(x)
        return self.to_scores(z)


class Accuracy(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, prediction, target, mask=None, token_dim=-1, sequence_dim=-2):
        prediction = F.softmax(prediction, token_dim).argmax(sequence_dim)
        scores = prediction == target
        n_padded = 0
        if mask is not None:
            n_padded = (mask == 0).sum()
        return scores.sum() / float(scores.numel() - n_padded)


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (Accuracy,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (DecoderLayer,
     lambda: ([], {'size': 4, 'n_heads': 4, 'd_feedforward': 4, 'dropout': 0.5, 'relative_pos': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (PositionwiseFeedForward,
     lambda: ([], {'d_model': 4, 'd_ff': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
]

