
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


import logging as L


import torch


from torch.utils.tensorboard import SummaryWriter


import torch.nn as nn


from torch.nn import functional as F


import time


from uuid import uuid4


from typing import Any


from typing import Union


from typing import TYPE_CHECKING


from typing import Iterator


from typing import Tuple


import logging


import re


import math


from collections import defaultdict


from typing import Optional


from typing import Protocol


from typing import cast


import random


import uuid


from typing import List


from typing import Dict


from typing import Generator


class MultiHeadFast(nn.Module):
    """Multihead self-attention."""

    def __init__(self, hparams, input_size):
        super().__init__()
        self.input_size = input_size
        self.head_size = input_size // hparams.n_heads
        self.n_heads = hparams.n_heads
        self.dropout = hparams.dropout
        self.qkv_proj = nn.Linear(input_size, 3 * input_size, bias=False)
        self.use_flash_attention = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        self.register_buffer('tril', torch.tril(torch.ones(hparams.context_size, hparams.context_size).view(1, 1, hparams.context_size, hparams.context_size)))
        self.head_dropout = nn.Dropout(hparams.dropout)
        self.proj = nn.Linear(input_size, input_size)
        self.out_dropout = nn.Dropout(hparams.dropout)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(self.input_size, dim=-1)
        q = q.view(B, T, self.n_heads, -1).transpose(1, 2)
        k = k.view(B, T, self.n_heads, -1).transpose(1, 2)
        v = v.view(B, T, self.n_heads, -1).transpose(1, 2)
        if self.use_flash_attention:
            heads_out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout, is_causal=True)
        else:
            weight = torch.einsum('bnth,bnuh->bntu', q, k)
            weight /= torch.sqrt(self.head_size)
            weight = weight.masked_fill(self.tril[:, :, :T, :T] == 0, float('-inf'))
            dist = F.softmax(weight, dim=-1)
            dist = self.head_dropout(dist)
            heads_out = torch.einsum('bntu,bnuh->bnth', dist, v)
        multi_head_out = heads_out.transpose(1, 2).reshape(B, T, C)
        return self.out_dropout(self.proj(multi_head_out))


class MLP(nn.Module):
    """Multi-Layer Perception (last ff ops of each block)."""

    def __init__(self, hparams, input_size):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_size, 4 * input_size), nn.ReLU(), nn.Linear(4 * input_size, input_size), nn.Dropout(hparams.dropout))

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Transformer block."""

    def __init__(self, hparams):
        super().__init__()
        self.sa_heads = MultiHeadFast(hparams, hparams.n_embed)
        self.mlp = MLP(hparams, hparams.n_embed)
        self.ln1 = nn.LayerNorm(hparams.n_embed)
        self.ln2 = nn.LayerNorm(hparams.n_embed)

    def forward(self, x):
        x = x + self.sa_heads(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class AttentionModel(nn.Module):

    def __init__(self, vocab_size, hparams, device):
        super().__init__()
        self.context_size = hparams.context_size
        self.device = device
        assert hparams.n_embed % hparams.n_heads == 0, 'n_embed must be divisible by n_heads'
        self.token_embedding_table = nn.Embedding(vocab_size, hparams.n_embed, device=device)
        self.pos_embedding_table = nn.Embedding(hparams.context_size, hparams.n_embed)
        self.blocks = nn.Sequential(*[Block(hparams) for _ in range(hparams.n_blocks)])
        self.ln_f = nn.LayerNorm(hparams.n_embed)
        self.lm_head = nn.Linear(hparams.n_embed, vocab_size)

    def forward(self, input_tokens, targets=None):
        B, T = input_tokens.shape
        token_embedding = self.token_embedding_table(input_tokens)
        position_embedding = self.pos_embedding_table(torch.arange(T, device=self.device))
        embedding = token_embedding + position_embedding
        x = self.blocks(embedding)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        if targets is not None:
            xlogits = logits.view(logits.shape[0] * logits.shape[1], -1)
            xtargets = targets.view(-1)
            loss = F.cross_entropy(xlogits, xtargets)
        else:
            loss = None
        return logits, loss

    @torch.no_grad()
    def generate(self, input_tokens, max_new_tokens):
        for i in range(max_new_tokens):
            logits = self(input_tokens[:, -self.context_size:])[0]
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_tokens = torch.cat([input_tokens, next_token], axis=1)
        return input_tokens

    @torch.no_grad()
    def generate_from_text(self, tokenizer, text, max_new_tokens):
        encoded_prompt = tokenizer.encode(text)
        torch_input = torch.tensor(encoded_prompt, dtype=torch.long)
        torch_input = torch_input.view(1, len(torch_input))
        torch_input = torch_input
        tokens = self.generate(torch_input, max_new_tokens)[0]
        chars = tokenizer.decode([x for x in tokens.tolist()])
        chars_out = chars[len(text):]
        return ''.join(chars_out)

