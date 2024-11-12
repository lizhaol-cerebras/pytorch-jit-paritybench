
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


from torch.utils import data


from sklearn.metrics import recall_score


import numpy as np


from torch import nn


import torch.nn.functional as F


import torch.nn as nn


import math


from torch.utils.tensorboard import SummaryWriter


def attention(q, k, v, d_k, mask=None, dropout=None):
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1000000000.0)
    scores = F.softmax(scores, dim=-1)
    if dropout is not None:
        scores = dropout(scores)
    output = torch.matmul(scores, v)
    return output


class MultiHeadAttention(nn.Module):

    def __init__(self, heads, d_model, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads
        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        bs = q.size(0)
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)
        scores = attention(q, k, v, self.d_k, mask, self.dropout)
        concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.d_model)
        output = self.out(concat)
        return output


class NeighbourEmbedding(nn.Module):

    def __init__(self, vocab_size, dimension):
        super().__init__()
        self.word_embed = nn.Embedding(vocab_size, dimension)
        self.linear1 = nn.Linear(2, 128)
        self.linear2 = nn.Linear(128, dimension)
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)

    def forward(self, words, positions):
        embedding = self.word_embed(words)
        pos = F.relu(self.linear1(positions))
        pos = self.dropout1(pos)
        pos = F.relu(self.linear2(pos))
        pos = self.dropout1(pos)
        neighbour_embedding = torch.cat((embedding, pos), dim=2)
        return neighbour_embedding


class Model(nn.Module):

    def __init__(self, vocab_size, embedding_dim, neighbours, heads):
        super().__init__()
        self.cand_embed = nn.Linear(2, 128)
        self.field_embed = nn.Linear(3, embedding_dim)
        self.embedding_dimension = embedding_dim
        self.neighbour_embeddings = NeighbourEmbedding(vocab_size, embedding_dim)
        self.attention_encodings = MultiHeadAttention(heads, embedding_dim * 2)
        self.linear_projection = nn.Linear(neighbours * embedding_dim * 2, 4 * embedding_dim * 2)
        self.linear_projection_2 = nn.Linear(128 + 2 * embedding_dim, embedding_dim)
        self.cos_sim = nn.CosineSimilarity(dim=1, eps=1e-06)

    def forward(self, field_id, candidate, neighbour_words, neighbour_positions, masks):
        id_embed = self.field_embed(field_id)
        cand_embed = self.cand_embed(candidate)
        neighbour_embeds = self.neighbour_embeddings(neighbour_words, neighbour_positions)
        self_attention = self.attention_encodings(neighbour_embeds, neighbour_embeds, neighbour_embeds, mask=masks)
        bs = self_attention.size(0)
        self_attention = self_attention.view(bs, -1)
        linear_proj = F.relu(self.linear_projection(self_attention))
        linear_proj = linear_proj.view(bs, 4, -1)
        pooled_attention = F.max_pool2d(linear_proj, 2, 2)
        unrolled_attention = pooled_attention.view(bs, -1)
        concat = torch.cat((cand_embed, unrolled_attention), dim=1)
        projected_candidate_encoding = F.relu(self.linear_projection_2(concat))
        similarity = self.cos_sim(id_embed, projected_candidate_encoding).view(bs, -1)
        scores = (similarity + 1) / 2
        return scores


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (MultiHeadAttention,
     lambda: ([], {'heads': 4, 'd_model': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
]

