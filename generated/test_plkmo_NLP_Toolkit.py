
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


import pandas as pd


import numpy as np


import torch


from torch.utils.data import DataLoader


import time


import logging


import torch.nn as nn


import torch.optim as optim


import torch.nn.functional as F


from torch.autograd import Variable


import math


import copy


import re


from torch.utils.data import Dataset


from sklearn.preprocessing import MinMaxScaler


from sklearn.preprocessing import StandardScaler


from torch.nn.utils import clip_grad_norm_


import matplotlib.pyplot as plt


from string import ascii_lowercase


from torch.nn import CrossEntropyLoss


from torch.nn import MSELoss


from functools import partial


from functools import wraps


from typing import Optional


from torch import nn


from torch.nn import functional as F


import itertools


from sklearn.feature_extraction.text import TfidfVectorizer


import scipy.sparse as sp


from collections import OrderedDict


from itertools import combinations


from sklearn.cluster import KMeans


from sklearn.cluster import AgglomerativeClustering


from sklearn.metrics import silhouette_score


from sklearn.metrics import davies_bouldin_score


from sklearn.metrics import calinski_harabasz_score


from sklearn.decomposition import PCA


from sklearn.manifold import TSNE


from copy import deepcopy


from typing import Dict


from time import time


from typing import List


from typing import Any


import numpy


from torch.nn.modules.linear import Linear


from typing import Tuple


from typing import Union


from typing import Iterable


import torch.optim.lr_scheduler


from random import seed


import collections


from torch.nn.parameter import Parameter


import warnings


import inspect


from typing import Callable


from torch import Tensor


from torch import device


from torch import dtype


import functools


from collections import UserDict


from collections import defaultdict


from enum import Enum


from typing import MutableMapping


from typing import NamedTuple


from typing import Sequence


from torch.utils.data import RandomSampler


from torch.utils.data import SequentialSampler


from torch.utils.data import TensorDataset


from torch.utils.data.distributed import DistributedSampler


from torch.optim import Optimizer


from torch.optim.lr_scheduler import LambdaLR


from torch.nn.utils.rnn import pad_sequence


from torch import optim


from collections import namedtuple


class pBLSTMLayer(nn.Module):

    def __init__(self, input_size, lstm_hidden_size, num_layers=1):
        super(pBLSTMLayer, self).__init__()
        self.lstm = nn.LSTM(input_size=2 * input_size, hidden_size=lstm_hidden_size, num_layers=num_layers, dropout=0.1, batch_first=True, bidirectional=True)

    def forward(self, x):
        x = x.contiguous().view(x.shape[0], int(x.shape[1] / 2), 2 * x.shape[2])
        out, hidden = self.lstm(x)
        return out, hidden


class Listener(nn.Module):

    def __init__(self, input_size, hidden_size):
        super(Listener, self).__init__()
        self.layer1 = pBLSTMLayer(input_size, hidden_size)
        self.layer2 = pBLSTMLayer(2 * hidden_size, hidden_size)
        self.layer3 = pBLSTMLayer(2 * hidden_size, hidden_size)

    def forward(self, x):
        output, _ = self.layer1(x)
        output, _ = self.layer2(output)
        output, _ = self.layer3(output)
        return output

    def flatten_parameters(self):
        self.layer1.lstm.flatten_parameters()
        self.layer2.lstm.flatten_parameters()
        self.layer3.lstm.flatten_parameters()


def Attention(q, k, v, dh, mask=None, dropout=None):
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(dh)
    if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -50000.0)
    scores = torch.softmax(scores, dim=-1)
    if dropout is not None:
        scores = dropout(scores)
    output = torch.matmul(scores, v)
    return output


class AttentionBlock(nn.Module):

    def __init__(self, hidden_size, features_size, d_model, droprate=0.1):
        super(AttentionBlock, self).__init__()
        self.d_model = d_model
        self.s_matrix = nn.Linear(hidden_size, d_model)
        self.h_matrix = nn.Linear(features_size, d_model)
        self.fc1 = nn.Linear(d_model, d_model)

    def forward(self, s_i, h):
        s_i = self.s_matrix(s_i)
        h = self.h_matrix(h)
        context, attention_distribution = Attention(s_i, h)
        context = self.fc1(context)
        return context, attention_distribution


class Speller(nn.Module):

    def __init__(self, embed_dim, hidden_size, features_size, max_label_len, output_class_dim, d_model=64, num_layers=2):
        super(Speller, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embed = nn.Embedding(output_class_dim, embed_dim)
        self.rnn = nn.ModuleList()
        self.rnn += [nn.LSTMCell(input_size=embed_dim + d_model, hidden_size=hidden_size)]
        self.rnn += [nn.LSTMCell(hidden_size, hidden_size) for _ in range(1, num_layers)]
        self.attention = AttentionBlock(hidden_size, features_size, d_model)
        self.max_label_len = max_label_len
        self.label_dim = output_class_dim
        self.character_distribution = nn.Linear(hidden_size + d_model, output_class_dim)
        self.teacher_forcing = True

    def forward(self, listener_feature, trg_input, infer=False):
        """If infer=False, performs a forward pass and returns logits, else if 
        infer=True, performs forward passes with initial trg_input=<sos>, whose output is appended to generate the next sequence until
        <eos> or max_length is reached, returns the logits sequence"""
        pred_y = []
        if not self.teacher_forcing:
            y = [self.embed(torch.FloatTensor(np.zeros((listener_feature.shape[0], 1))))]
        else:
            y = self.embed(trg_input)
        hidden_states = [torch.zeros((listener_feature.shape[0], self.hidden_size)) for _ in range(self.num_layers)]
        c_states = [torch.zeros((listener_feature.shape[0], self.hidden_size)) for _ in range(self.num_layers)]
        if listener_feature.is_cuda:
            hidden_states = [hidden_state for hidden_state in hidden_states]
            c_states = [c_state for c_state in c_states]
        if not infer:
            for step in range(len(y[0, :, 0])):
                context, attention_score = self.attention(hidden_states[-1].unsqueeze(1), listener_feature)
                rnn_input = torch.cat([y[:, step, :], context.squeeze(1)], dim=1)
                for layer in range(self.num_layers):
                    if layer == 0:
                        hidden_states[0], c_states[0] = self.rnn[layer](rnn_input, (hidden_states[0], c_states[0]))
                    else:
                        hidden_states[layer], c_states[layer] = self.rnn[layer](hidden_states[layer - 1], (hidden_states[layer], c_states[layer]))
                logits = self.character_distribution(torch.cat([hidden_states[-1], context.squeeze(1)], dim=1))
                if not self.teacher_forcing:
                    y.append(self.embed(torch.softmax(logits, dim=1).max(1)[1]))
                pred_y.append(logits)
            pred_y = torch.stack(pred_y, dim=1)
        else:
            for step in range(self.max_label_len):
                context, attention_score = self.attention(hidden_states[-1].unsqueeze(1), listener_feature)
                rnn_input = torch.cat([y[:, step, :], context.squeeze(1)], dim=1)
                for layer in range(self.num_layers):
                    if layer == 0:
                        hidden_states[0], c_states[0] = self.rnn[layer](rnn_input, (hidden_states[0], c_states[0]))
                    else:
                        hidden_states[layer], c_states[layer] = self.rnn[layer](hidden_states[layer - 1], (hidden_states[layer], c_states[layer]))
                logits = self.character_distribution(torch.cat([hidden_states[-1], context.squeeze(1)], dim=1))
                pred_y.append(logits)
                pred_token = torch.softmax(logits, dim=1).max(1)[1]
                y = torch.cat([y, self.embed(pred_token).unsqueeze(1)], dim=1)
                if pred_token.item() == 2:
                    break
            pred_y = torch.stack(pred_y, dim=1)
            pred_y = torch.softmax(pred_y, dim=2).max(2)[1]
        return pred_y

    def flatten_parameters(self):
        for l in range(len(self.rnn)):
            self.rnn[l].flatten_parameters()


class LAS(nn.Module):

    def __init__(self, vocab_size, listener_embed_size, listener_hidden_size, output_class_dim, max_label_len=100):
        super(LAS, self).__init__()
        self.vocab_size = vocab_size
        self.listener_embed_size = listener_embed_size
        self.listener_hidden_size = listener_hidden_size
        self.output_class_dim = output_class_dim
        self.max_label_len = max_label_len
        self.embed = nn.Embedding(vocab_size, listener_embed_size)
        self.listener = Listener(listener_embed_size, listener_hidden_size)
        self.speller = Speller(listener_hidden_size, listener_hidden_size, 2 * listener_hidden_size, max_label_len, output_class_dim, num_layers=2)

    def forward(self, x, trg_input, infer=False):
        x = self.embed(x)
        x = self.listener(x)
        x = self.speller(x, trg_input, infer)
        return x

    @classmethod
    def load_model(cls, path, args, cuda=True, amp=None):
        checkpoint = torch.load(path)
        model = cls(vocab_size=checkpoint['vocab_size'], listener_embed_size=checkpoint['listener_embed_size'], listener_hidden_size=checkpoint['listener_hidden_size'], output_class_dim=checkpoint['output_class_dim'], max_label_len=checkpoint['max_label_len'])
        model.listener.flatten_parameters()
        if cuda:
            model
        if amp is not None:
            optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-09)
            model, optimizer = amp.initialize(model, optimizer, opt_level='O2')
            amp.load_state_dict(checkpoint['amp'])
            None
        else:
            optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-09)
            optimizer.load_state_dict(checkpoint['optimizer'])
        model.load_state_dict(checkpoint['state_dict'])
        return model, optimizer

    def save_state(self, epoch, optimizer, scheduler, best_acc, path, amp=None):
        state = {'epoch': epoch + 1, 'state_dict': self.state_dict(), 'best_acc': best_acc, 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), 'vocab_size': self.vocab_size, 'listener_embed_size': self.listener_embed_size, 'listener_hidden_size': self.listener_hidden_size, 'output_class_dim': self.output_class_dim, 'max_label_len': self.max_label_len, 'amp': amp.state_dict() if amp != None else None}
        torch.save(state, path)


class Conv2dBlock(nn.Module):

    def __init__(self, c_in, c_out=64):
        super(Conv2dBlock, self).__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv2 = nn.Conv2d(c_out, c_out, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c_out)
        self.bn2 = nn.BatchNorm2d(c_out)
        self.drop1 = nn.Dropout(p=0.1)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.drop1(x)
        x = torch.relu(self.bn2(self.conv2(x)))
        return x


class pyTransformer(nn.Module):

    def __init__(self, src_vocab, trg_vocab, d_model, ff_dim, num, n_heads, max_encoder_len=80, max_decoder_len=80):
        super(pyTransformer, self).__init__()
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab
        self.d_model = d_model
        self.ff_dim = ff_dim
        self.num = num
        self.n_heads = n_heads
        self.max_encoder_len = max_encoder_len
        self.max_decoder_len = max_decoder_len
        self.embed1 = nn.Embedding(src_vocab, d_model)
        self.embed2 = nn.Embedding(trg_vocab, d_model)
        self.transformer = nn.Transformer(d_model=d_model, nhead=n_heads, num_encoder_layers=num, num_decoder_layers=num, dim_feedforward=ff_dim, dropout=0.1)
        self.fc1 = nn.Linear(d_model, trg_vocab)

    def forward(self, src, trg, src_mask, trg_mask=None, infer=False, trg_vocab_obj=None):
        src = self.embed1(src)
        trg = self.embed2(trg)
        src = src.permute(1, 0, 2)
        trg = trg.permute(1, 0, 2)
        out = self.transformer(src, trg, src_key_padding_mask=src_mask, tgt_key_padding_mask=trg_mask)
        out = out.permute(1, 0, 2)
        out = self.fc1(out)
        return out

    @classmethod
    def load_model(cls, path):
        checkpoint = torch.load(path)
        model = cls(src_vocab=checkpoint['src_vocab'], trg_vocab=checkpoint['trg_vocab'], d_model=checkpoint['d_model'], ff_dim=checkpoint['ff_dim'], num=checkpoint['num'], n_heads=checkpoint['n_heads'], max_encoder_len=checkpoint['max_encoder_len'], max_decoder_len=checkpoint['max_decoder_len'])
        model.load_state_dict(checkpoint['state_dict'])
        return model

    def save_state(self, epoch, optimizer, scheduler, best_acc, path):
        state = {'epoch': epoch + 1, 'state_dict': self.state_dict(), 'best_acc': best_acc, 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), 'src_vocab': self.src_vocab, 'trg_vocab': self.trg_vocab, 'd_model': self.d_model, 'ff_dim': self.ff_dim, 'num': self.num, 'n_heads': self.n_heads, 'max_encoder_len': self.max_encoder_len, 'max_decoder_len': self.max_decoder_len}
        torch.save(state, path)


class Pos_Encoder(nn.Module):

    def __init__(self, d_model, max_len):
        super(Pos_Encoder, self).__init__()
        self.d_model = d_model
        pe = torch.zeros(max_len, d_model)
        for pos in range(max_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / 10000 ** (i / d_model))
                pe[pos, i + 1] = math.cos(pos / 10000 ** (i / d_model))
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return x


class MHAttention(nn.Module):

    def __init__(self, d_model, n_heads, droprate=0.1):
        super(MHAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.dh = d_model // n_heads
        self.q_matrix = nn.Linear(d_model, d_model)
        self.k_matrix = nn.Linear(d_model, d_model)
        self.v_matrix = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(droprate)
        self.fc1 = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        q = self.q_matrix(q)
        q = q.view(q.size(0), self.n_heads, -1, self.dh)
        k = self.k_matrix(k)
        k = k.view(k.size(0), self.n_heads, -1, self.dh)
        v = self.v_matrix(v)
        v = v.view(v.size(0), self.n_heads, -1, self.dh)
        scores = Attention(q, k, v, self.dh, mask, self.dropout)
        scores = scores.reshape(q.size(0), -1, self.d_model)
        output = self.fc1(scores)
        return output


class FeedForward(nn.Module):

    def __init__(self, d_model, hidden_size=2048, droprate=0.1):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, hidden_size)
        self.dropout = nn.Dropout(droprate)
        self.fc2 = nn.Linear(hidden_size, d_model)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class LayerNorm(nn.Module):

    def __init__(self, d_model):
        super(LayerNorm, self).__init__()
        self.d_model = d_model
        self.alpha = nn.Parameter(torch.ones(self.d_model))
        self.bias = nn.Parameter(torch.zeros(self.d_model))

    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + 1e-07) + self.bias
        return norm


class EncoderLayer(nn.Module):

    def __init__(self, d_model, n_heads, ff_dim, droprate=0.1):
        super(EncoderLayer, self).__init__()
        self.norm1 = LayerNorm(d_model)
        self.attn = MHAttention(d_model=d_model, n_heads=n_heads)
        self.dropout1 = nn.Dropout(droprate)
        self.norm2 = LayerNorm(d_model)
        self.fc1 = FeedForward(d_model=d_model, hidden_size=ff_dim)
        self.dropout2 = nn.Dropout(droprate)

    def forward(self, x, mask):
        x1 = self.norm1(x)
        x = x + self.dropout1(self.attn(x1, x1, x1, mask))
        x1 = self.norm2(x)
        x = x + self.dropout2(self.fc1(x1))
        return x


class DecoderLayer(nn.Module):

    def __init__(self, d_model, n_heads, ff_dim, droprate=0.1):
        super(DecoderLayer, self).__init__()
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)
        self.dropout1 = nn.Dropout(droprate)
        self.dropout2 = nn.Dropout(droprate)
        self.dropout3 = nn.Dropout(droprate)
        self.attn1 = MHAttention(d_model=d_model, n_heads=n_heads)
        self.attn2 = MHAttention(d_model=d_model, n_heads=n_heads)
        self.fc1 = FeedForward(d_model=d_model, hidden_size=ff_dim)

    def forward(self, x, e_out, src_mask, trg_mask):
        x1 = self.norm1(x)
        x = x + self.dropout1(self.attn1(x1, x1, x1, trg_mask))
        x1 = self.norm2(x)
        x = x + self.dropout2(self.attn2(x1, e_out, e_out, src_mask))
        x1 = self.norm3(x)
        x = x + self.dropout3(self.fc1(x1))
        return x


class Conv1dBlock(nn.Module):

    def __init__(self, c_in, c_out=64):
        super(Conv1dBlock, self).__init__()
        self.conv1 = nn.Conv1d(c_in, c_out, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv2 = nn.Conv1d(c_out, c_out, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv3 = nn.Conv1d(c_out, c_out, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(c_out)
        self.bn2 = nn.BatchNorm1d(c_out)
        self.bn3 = nn.BatchNorm1d(c_out)
        self.drop1 = nn.Dropout(p=0.05)
        self.drop2 = nn.Dropout(p=0.05)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.drop1(x)
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.drop2(x)
        x = torch.relu(self.bn3(self.conv3(x)))
        return x


def clone_layers(module, num):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(num)])


class EncoderBlock(nn.Module):

    def __init__(self, vocab_size, d_model, ff_dim, num, n_heads, max_encoder_len):
        super(EncoderBlock, self).__init__()
        self.num = num
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pe = Pos_Encoder(d_model, max_len=max_encoder_len)
        self.layers = clone_layers(EncoderLayer(d_model, n_heads, ff_dim), num)
        self.norm = LayerNorm(d_model)

    def forward(self, src, mask):
        x = self.embed(src)
        x = self.pe(x)
        for i in range(self.num):
            x = self.layers[i](x, mask)
        x = self.norm(x)
        return x


class DecoderBlock(nn.Module):

    def __init__(self, vocab_size, d_model, ff_dim, num, n_heads, max_decoder_len):
        super(DecoderBlock, self).__init__()
        self.num = num
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pe = Pos_Encoder(d_model, max_len=max_decoder_len)
        self.layers = clone_layers(DecoderLayer(d_model, n_heads, ff_dim), num)
        self.norm = LayerNorm(d_model)

    def forward(self, trg, e_out, src_mask, trg_mask):
        x = self.embed(trg)
        x = self.pe(x)
        for i in range(self.num):
            x = self.layers[i](x, e_out, src_mask, trg_mask)
        x = self.norm(x)
        return x


def create_trg_mask(trg, cuda):
    trg_mask = (trg == 1).unsqueeze(-2).bool()
    trg_mask = trg_mask[:, 0, :]
    return trg_mask


class SpeechTransformer(nn.Module):

    def __init__(self, src_vocab, trg_vocab, d_model, ff_dim, num, n_heads, max_encoder_len, max_decoder_len, use_conv=1):
        super(SpeechTransformer, self).__init__()
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab
        self.d_model = d_model
        self.ff_dim = ff_dim
        self.num = num
        self.n_heads = n_heads
        self.max_encoder_len = max_encoder_len
        self.max_decoder_len = max_decoder_len
        if use_conv == 1:
            self.conv = Conv2dBlock(3, 64)
        self.encoder = EncoderBlock(vocab_size=int(64 * src_vocab / 12), d_model=d_model, ff_dim=ff_dim, num=num, n_heads=n_heads, max_len=max_encoder_len)
        self.decoder = DecoderBlock(vocab_size=trg_vocab, d_model=d_model, ff_dim=ff_dim, num=num, n_heads=n_heads, max_len=max_decoder_len)
        self.fc1 = nn.Linear(d_model, trg_vocab)

    def forward(self, src, trg, src_mask, trg_mask=None, g_mask1=None, g_mask2=None, infer=False):
        """Runs a forward pass if infer=False.
        If infer=True (evaluation mode), generate text sequence given a sequence of features and returns the generated sequence indexes"""
        src = self.conv(src)
        src = src.reshape(src.shape[0], src.shape[-1], -1)
        e_out = self.encoder(src, src_mask, g_mask1)
        if not infer:
            d_out = self.decoder(trg, e_out, src_mask, trg_mask, g_mask2)
            x = self.fc1(d_out)
            return x
        else:
            for i in range(2, self.max_decoder_len):
                trg_mask = create_trg_mask(trg, src.is_cuda)
                x = self.fc1(self.decoder(trg, e_out, src_mask, trg_mask, g_mask2=None))
                o_labels = torch.softmax(x, dim=2).max(2)[1]
                trg = torch.cat((trg, o_labels[:, -1:]), dim=1)
                if o_labels[0, -1].item() == 2:
                    break
            return trg

    @classmethod
    def load_model(cls, path, args, cuda=True, amp=None):
        checkpoint = torch.load(path)
        model = cls(src_vocab=checkpoint['src_vocab'], trg_vocab=checkpoint['trg_vocab'], d_model=checkpoint['d_model'], ff_dim=checkpoint['ff_dim'], num=checkpoint['num'], n_heads=checkpoint['n_heads'], max_encoder_len=checkpoint['max_encoder_len'], max_decoder_len=checkpoint['max_decoder_len'], use_conv=True)
        if cuda:
            model
        if amp is not None:
            optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-09)
            model, optimizer = amp.initialize(model, optimizer, opt_level='O2')
            amp.load_state_dict(checkpoint['amp'])
            None
        else:
            optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-09)
            optimizer.load_state_dict(checkpoint['optimizer'])
        model.load_state_dict(checkpoint['state_dict'])
        return model, optimizer

    def save_state(self, epoch, optimizer, scheduler, best_acc, path, amp=None):
        state = {'epoch': epoch + 1, 'state_dict': self.state_dict(), 'best_acc': best_acc, 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), 'src_vocab': self.src_vocab, 'trg_vocab': self.trg_vocab, 'd_model': self.d_model, 'ff_dim': self.ff_dim, 'num': self.num, 'n_heads': self.n_heads, 'max_encoder_len': self.max_encoder_len, 'max_decoder_len': self.max_decoder_len, 'amp': amp.state_dict()}
        torch.save(state, path)


def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {'gelu': gelu, 'relu': torch.nn.functional.relu, 'swish': swish}


class BertSelfAttention(nn.Module):

    def __init__(self, config):
        super(BertSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError('The hidden size (%d) is not a multiple of the number of attention heads (%d)' % (config.hidden_size, config.num_attention_heads))
        self.output_attentions = config.output_attentions
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask, head_mask=None):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_scores = attention_scores + attention_mask
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)
        if head_mask is not None:
            attention_probs = attention_probs * head_mask
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        outputs = (context_layer, attention_probs) if self.output_attentions else (context_layer,)
        return outputs


def prune_linear_layer(layer, index, dim=0):
    """ Prune a linear layer (a model parameters) to keep only entries in index.
        Return the pruned layer as a new layer with requires_grad=True.
        Used to remove heads.
    """
    index = index
    W = layer.weight.index_select(dim, index).clone().detach()
    if layer.bias is not None:
        if dim == 1:
            b = layer.bias.clone().detach()
        else:
            b = layer.bias[index].clone().detach()
    new_size = list(layer.weight.size())
    new_size[dim] = len(index)
    new_layer = nn.Linear(new_size[1], new_size[0], bias=layer.bias is not None)
    new_layer.weight.requires_grad = False
    new_layer.weight.copy_(W.contiguous())
    new_layer.weight.requires_grad = True
    if layer.bias is not None:
        new_layer.bias.requires_grad = False
        new_layer.bias.copy_(b.contiguous())
        new_layer.bias.requires_grad = True
    return new_layer


class AlbertAttention(BertSelfAttention):

    def __init__(self, config):
        super().__init__(config)
        self.output_attentions = config.output_attentions
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        mask = torch.ones(self.num_attention_heads, self.attention_head_size)
        heads = set(heads) - self.pruned_heads
        for head in heads:
            head = head - sum(1 if h < head else 0 for h in self.pruned_heads)
            mask[head] = 0
        mask = mask.view(-1).contiguous().eq(1)
        index = torch.arange(len(mask))[mask].long()
        self.query = prune_linear_layer(self.query, index)
        self.key = prune_linear_layer(self.key, index)
        self.value = prune_linear_layer(self.value, index)
        self.dense = prune_linear_layer(self.dense, index, dim=1)
        self.num_attention_heads = self.num_attention_heads - len(heads)
        self.all_head_size = self.attention_head_size * self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(self, input_ids, attention_mask=None, head_mask=None):
        mixed_query_layer = self.query(input_ids)
        mixed_key_layer = self.key(input_ids)
        mixed_value_layer = self.value(input_ids)
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)
        if head_mask is not None:
            attention_probs = attention_probs * head_mask
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        w = self.dense.weight.t().view(self.num_attention_heads, self.attention_head_size, self.hidden_size)
        b = self.dense.bias
        projected_context_layer = torch.einsum('bfnd,ndh->bfh', context_layer, w) + b
        projected_context_layer_dropout = self.dropout(projected_context_layer)
        layernormed_context_layer = self.LayerNorm(input_ids + projected_context_layer_dropout)
        return (layernormed_context_layer, attention_probs) if self.output_attentions else (layernormed_context_layer,)


class AlbertLayer(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.full_layer_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attention = AlbertAttention(config)
        self.ffn = nn.Linear(config.hidden_size, config.intermediate_size)
        self.ffn_output = nn.Linear(config.intermediate_size, config.hidden_size)
        self.activation = ACT2FN[config.hidden_act]

    def forward(self, hidden_states, attention_mask=None, head_mask=None):
        attention_output = self.attention(hidden_states, attention_mask, head_mask)
        ffn_output = self.ffn(attention_output[0])
        ffn_output = self.activation(ffn_output)
        ffn_output = self.ffn_output(ffn_output)
        hidden_states = self.full_layer_layer_norm(ffn_output + attention_output[0])
        return (hidden_states,) + attention_output[1:]


class AlbertLayerGroup(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.albert_layers = nn.ModuleList([AlbertLayer(config) for _ in range(config.inner_group_num)])

    def forward(self, hidden_states, attention_mask=None, head_mask=None):
        layer_hidden_states = ()
        layer_attentions = ()
        for layer_index, albert_layer in enumerate(self.albert_layers):
            layer_output = albert_layer(hidden_states, attention_mask, head_mask[layer_index])
            hidden_states = layer_output[0]
            if self.output_attentions:
                layer_attentions = layer_attentions + (layer_output[1],)
            if self.output_hidden_states:
                layer_hidden_states = layer_hidden_states + (hidden_states,)
        outputs = hidden_states,
        if self.output_hidden_states:
            outputs = outputs + (layer_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (layer_attentions,)
        return outputs


class AlbertTransformer(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.embedding_hidden_mapping_in = nn.Linear(config.embedding_size, config.hidden_size)
        self.albert_layer_groups = nn.ModuleList([AlbertLayerGroup(config) for _ in range(config.num_hidden_groups)])

    def forward(self, hidden_states, attention_mask=None, head_mask=None):
        hidden_states = self.embedding_hidden_mapping_in(hidden_states)
        all_attentions = ()
        if self.output_hidden_states:
            all_hidden_states = hidden_states,
        for i in range(self.config.num_hidden_layers):
            layers_per_group = int(self.config.num_hidden_layers / self.config.num_hidden_groups)
            group_idx = int(i / (self.config.num_hidden_layers / self.config.num_hidden_groups))
            layer_group_output = self.albert_layer_groups[group_idx](hidden_states, attention_mask, head_mask[group_idx * layers_per_group:(group_idx + 1) * layers_per_group])
            hidden_states = layer_group_output[0]
            if self.output_attentions:
                all_attentions = all_attentions + layer_group_output[-1]
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
        outputs = hidden_states,
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs


class AlbertMLMHead(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.LayerNorm = nn.LayerNorm(config.embedding_size)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.dense = nn.Linear(config.hidden_size, config.embedding_size)
        self.decoder = nn.Linear(config.embedding_size, config.vocab_size)
        self.activation = ACT2FN[config.hidden_act]
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        hidden_states = self.decoder(hidden_states)
        prediction_scores = hidden_states + self.bias
        return prediction_scores


BertLayerNorm = torch.nn.LayerNorm


class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """

    def __init__(self, config):
        super(BertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None, position_ids=None):
        seq_length = input_ids.size(1)
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertSelfOutput(nn.Module):

    def __init__(self, config):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):

    def __init__(self, config):
        super(BertAttention, self).__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        mask = torch.ones(self.self.num_attention_heads, self.self.attention_head_size)
        heads = set(heads) - self.pruned_heads
        for head in heads:
            head = head - sum(1 if h < head else 0 for h in self.pruned_heads)
            mask[head] = 0
        mask = mask.view(-1).contiguous().eq(1)
        index = torch.arange(len(mask))[mask].long()
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(self, input_tensor, attention_mask, head_mask=None):
        self_outputs = self.self(input_tensor, attention_mask, head_mask)
        attention_output = self.output(self_outputs[0], input_tensor)
        outputs = (attention_output,) + self_outputs[1:]
        return outputs


class BertIntermediate(nn.Module):

    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str) or sys.version_info[0] == 2 and isinstance(config.hidden_act, unicode):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):

    def __init__(self, config):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):

    def __init__(self, config):
        super(BertLayer, self).__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask, head_mask=None):
        attention_outputs = self.attention(hidden_states, attention_mask, head_mask)
        attention_output = attention_outputs[0]
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output,) + attention_outputs[1:]
        return outputs


class BertEncoder(nn.Module):

    def __init__(self, config):
        super(BertEncoder, self).__init__()
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask, head_mask=None):
        all_hidden_states = ()
        all_attentions = ()
        for i, layer_module in enumerate(self.layer):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            layer_outputs = layer_module(hidden_states, attention_mask, head_mask[i])
            hidden_states = layer_outputs[0]
            if self.output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        outputs = hidden_states,
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs


class BertPooler(nn.Module):

    def __init__(self, config):
        super(BertPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertPredictionHeadTransform(nn.Module):

    def __init__(self, config):
        super(BertPredictionHeadTransform, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str) or sys.version_info[0] == 2 and isinstance(config.hidden_act, unicode):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertLMPredictionHead(nn.Module):

    def __init__(self, config):
        super(BertLMPredictionHead, self).__init__()
        self.transform = BertPredictionHeadTransform(config)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states) + self.bias
        return hidden_states


class BertOnlyMLMHead(nn.Module):

    def __init__(self, config):
        super(BertOnlyMLMHead, self).__init__()
        self.predictions = BertLMPredictionHead(config)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class BertOnlyNSPHead(nn.Module):

    def __init__(self, config):
        super(BertOnlyNSPHead, self).__init__()
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, pooled_output):
        seq_relationship_score = self.seq_relationship(pooled_output)
        return seq_relationship_score


class BertPreTrainingHeads(nn.Module):

    def __init__(self, config):
        super(BertPreTrainingHeads, self).__init__()
        self.predictions = BertLMPredictionHead(config)
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, sequence_output, pooled_output):
        prediction_scores = self.predictions(sequence_output)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score


CONFIG_NAME = 'config.json'


def http_get(url, temp_file, proxies=None):
    req = requests.get(url, stream=True, proxies=proxies)
    content_length = req.headers.get('Content-Length')
    total = int(content_length) if content_length is not None else None
    progress = tqdm(unit='B', total=total)
    for chunk in req.iter_content(chunk_size=1024):
        if chunk:
            progress.update(len(chunk))
            temp_file.write(chunk)
    progress.close()


logger = logging.getLogger('__file__')


class Config(object):

    def __init__(self, task):
        if task == 'classification':
            self.train_data = './data/train.csv'
            self.infer_data = './data/infer.csv'
            self.max_vocab_len = 7000
            self.hidden_size_1 = 330
            self.hidden_size_2 = 130
            self.tokens_length = 200
            self.num_classes = 5
            self.train_test_split = 0
            self.test_ratio = 0.1
            self.batch_size = 32
            self.gradient_acc_steps = 1
            self.max_norm = 1.0
            self.num_epochs = 40
            self.lr = 0.001
            self.model_no = 2
            self.train = 1
            self.infer = 0
        elif task == 'translation':
            self.src_path = './data/translation//eng_zh/news-commentary-v13.zh-en.en'
            self.trg_path = './data/translation/eng_zh/news-commentary-v13.zh-en.zh'
            self.src_lang = 'en'
            self.trg_lang = 'zh'
            self.batch_size = 16
            self.d_model = 512
            self.ff_dim = 2048
            self.num = 6
            self.n_heads = 8
            self.max_encoder_len = 200
            self.max_decoder_len = 200
            self.fp16 = 1
            self.num_epochs = 500
            self.lr = 0.0001
            self.gradient_acc_steps = 2
            self.max_norm = 1.0
            self.T_max = 5000
            self.model_no = 0
            self.train = 1
            self.evaluate = 0
            self.infer = 0
        elif task == 'punctuation_restoration':
            self.data_path = './data/train.tags.en-fr.en'
            self.level = 'bpe'
            self.bpe_word_ratio = 0.7
            self.bpe_vocab_size = 7000
            self.batch_size = 32
            self.d_model = 512
            self.ff_dim = 2048
            self.num = 6
            self.n_heads = 8
            self.max_encoder_len = 80
            self.max_decoder_len = 80
            self.LAS_embed_dim = 512
            self.LAS_hidden_size = 512
            self.num_epochs = 127
            self.lr = 3e-05
            self.gradient_acc_steps = 2
            self.max_norm = 1.0
            self.T_max = 5000
            self.model_no = 1
            self.train = 1
            self.infer = 0
        elif task == 'summarization':
            self.data_path = './data/train.tags.en-fr.en'
            self.level = 'bpe'
            self.bpe_word_ratio = 0.7
            self.bpe_vocab_size = 7000
            self.max_features_length = 200
            self.d_model = 512
            self.ff_dim = 2048
            self.num = 6
            self.n_heads = 8
            self.max_encoder_len = 200
            self.max_decoder_len = 200
            self.LAS_embed_dim = 512
            self.LAS_hidden_size = 512
            self.batch_size = 32
            self.fp16 = 1
            self.num_epochs = 500
            self.lr = 0.0003
            self.gradient_acc_steps = 2
            self.max_norm = 1.0
            self.T_max = 5000
            self.model_no = 0
            self.train = 1
            self.infer = 0
        elif task == 'ner':
            self.train_path = './data/ner/conll2003/eng.train.txt'
            self.test_path = './data/ner/conll2003/eng.testa.txt'
            self.num_classes = 9
            self.batch_size = 8
            self.tokens_length = 128
            self.max_steps = -1
            self.warmup_steps = 0
            self.weight_decay = 0.0
            self.adam_epsilon = 1e-08
            self.gradient_acc_steps = 1
            self.max_norm = 1.0
            self.num_epochs = 7
            self.lr = 5e-05
            self.model_no = 0
            self.model_type = 'bert'
            self.train = 0
            self.infer = 1
        elif task == 'pos':
            self.train_path = './data/ner/conll2003/eng.train.txt'
            self.test_path = './data/ner/conll2003/eng.testa.txt'
            self.num_classes = 45
            self.batch_size = 16
            self.tokens_length = 100
            self.max_steps = -1
            self.warmup_steps = 0
            self.weight_decay = 0.0
            self.adam_epsilon = 1e-08
            self.gradient_acc_steps = 2
            self.max_norm = 1.0
            self.num_epochs = 7
            self.lr = 5e-05
            self.model_no = 0
            self.model_type = 'bert'
            self.train = 1
            self.infer = 1
        elif task == 'ASR':
            self.folder = 'train-clean-5'
            self.level = 'word'
            self.use_lg_mels = 1
            self.use_conv = 1
            self.n_mels = 80
            self.n_mfcc = 13
            self.n_fft = 25
            self.hop_length = 10
            self.max_frame_len = 1000
            self.d_model = 128
            self.ff_dim = 128
            self.num = 6
            self.n_heads = 4
            self.batch_size = 32
            self.fp16 = 1
            self.num_epochs = 9000
            self.lr = 0.0003
            self.gradient_acc_steps = 4
            self.max_norm = 1.0
            self.T_max = 5000
            self.model_no = 0
            self.train = 1
            self.infer = 0
        elif task == 'generation':
            self.model_no = 0
        elif task == 'style_transfer':
            cls = style_config()


def s3_request(func):
    """
    Wrapper function for s3 requests in order to create more helpful error
    messages.
    """

    @wraps(func)
    def wrapper(url, *args, **kwargs):
        try:
            return func(url, *args, **kwargs)
        except ClientError as exc:
            if int(exc.response['Error']['Code']) == 404:
                raise EnvironmentError('file {} not found'.format(url))
            else:
                raise
    return wrapper


def split_s3_path(url):
    """Split a full s3 path into the bucket name and path."""
    parsed = urlparse(url)
    if not parsed.netloc or not parsed.path:
        raise ValueError('bad s3 path {}'.format(url))
    bucket_name = parsed.netloc
    s3_path = parsed.path
    if s3_path.startswith('/'):
        s3_path = s3_path[1:]
    return bucket_name, s3_path


@s3_request
def s3_etag(url, proxies=None):
    """Check ETag on S3 object."""
    s3_resource = boto3.resource('s3', config=Config(proxies=proxies))
    bucket_name, s3_path = split_s3_path(url)
    s3_object = s3_resource.Object(bucket_name, s3_path)
    return s3_object.e_tag


@s3_request
def s3_get(url, temp_file, proxies=None):
    """Pull a file directly from S3."""
    s3_resource = boto3.resource('s3', config=Config(proxies=proxies))
    bucket_name, s3_path = split_s3_path(url)
    s3_resource.Bucket(bucket_name).download_fileobj(s3_path, temp_file)


def url_to_filename(url, etag=None):
    """
    Convert `url` into a hashed filename in a repeatable way.
    If `etag` is specified, append its hash to the url's, delimited
    by a period.
    """
    url_bytes = url.encode('utf-8')
    url_hash = sha256(url_bytes)
    filename = url_hash.hexdigest()
    if etag:
        etag_bytes = etag.encode('utf-8')
        etag_hash = sha256(etag_bytes)
        filename += '.' + etag_hash.hexdigest()
    return filename


def get_from_cache(url, cache_dir=None, force_download=False, proxies=None):
    """
    Given a URL, look for the corresponding dataset in the local cache.
    If it's not there, download it. Then return the path to the cached file.
    """
    if cache_dir is None:
        cache_dir = PYTORCH_TRANSFORMERS_CACHE
    if sys.version_info[0] == 3 and isinstance(cache_dir, Path):
        cache_dir = str(cache_dir)
    if sys.version_info[0] == 2 and not isinstance(cache_dir, str):
        cache_dir = str(cache_dir)
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    if url.startswith('s3://'):
        etag = s3_etag(url, proxies=proxies)
    else:
        try:
            response = requests.head(url, allow_redirects=True, proxies=proxies)
            if response.status_code != 200:
                etag = None
            else:
                etag = response.headers.get('ETag')
        except EnvironmentError:
            etag = None
    if sys.version_info[0] == 2 and etag is not None:
        etag = etag.decode('utf-8')
    filename = url_to_filename(url, etag)
    cache_path = os.path.join(cache_dir, filename)
    if not os.path.exists(cache_path) and etag is None:
        matching_files = fnmatch.filter(os.listdir(cache_dir), filename + '.*')
        matching_files = list(filter(lambda s: not s.endswith('.json'), matching_files))
        if matching_files:
            cache_path = os.path.join(cache_dir, matching_files[-1])
    if not os.path.exists(cache_path) or force_download:
        with tempfile.NamedTemporaryFile() as temp_file:
            logger.info('%s not found in cache or force_download set to True, downloading to %s', url, temp_file.name)
            if url.startswith('s3://'):
                s3_get(url, temp_file, proxies=proxies)
            else:
                http_get(url, temp_file, proxies=proxies)
            temp_file.flush()
            temp_file.seek(0)
            logger.info('copying %s to cache at %s', temp_file.name, cache_path)
            with open(cache_path, 'wb') as cache_file:
                shutil.copyfileobj(temp_file, cache_file)
            logger.info('creating metadata file for %s', cache_path)
            meta = {'url': url, 'etag': etag}
            meta_path = cache_path + '.json'
            with open(meta_path, 'w') as meta_file:
                output_string = json.dumps(meta)
                if sys.version_info[0] == 2 and isinstance(output_string, str):
                    output_string = unicode(output_string, 'utf-8')
                meta_file.write(output_string)
            logger.info('removing temp file %s', temp_file.name)
    return cache_path


def cached_path(url_or_filename, cache_dir=None, force_download=False, proxies=None):
    """
    Given something that might be a URL (or might be a local path),
    determine which. If it's a URL, download the file and cache it, and
    return the path to the cached file. If it's already a local path,
    make sure the file exists and then return the path.
    Args:
        cache_dir: specify a cache directory to save the file to (overwrite the default cache dir).
        force_download: if True, re-dowload the file even if it's already cached in the cache dir.
    """
    if cache_dir is None:
        cache_dir = PYTORCH_TRANSFORMERS_CACHE
    if sys.version_info[0] == 3 and isinstance(url_or_filename, Path):
        url_or_filename = str(url_or_filename)
    if sys.version_info[0] == 3 and isinstance(cache_dir, Path):
        cache_dir = str(cache_dir)
    parsed = urlparse(url_or_filename)
    if parsed.scheme in ('http', 'https', 's3'):
        return get_from_cache(url_or_filename, cache_dir=cache_dir, force_download=force_download, proxies=proxies)
    elif os.path.exists(url_or_filename):
        return url_or_filename
    elif parsed.scheme == '':
        raise EnvironmentError('file {} not found'.format(url_or_filename))
    else:
        raise ValueError('unable to parse {} as a URL or as a local path'.format(url_or_filename))


class PretrainedConfig(object):
    """ Base class for all configuration classes.
        Handle a few common parameters and methods for loading/downloading/saving configurations.
    """
    pretrained_config_archive_map = {}

    def __init__(self, **kwargs):
        self.finetuning_task = kwargs.pop('finetuning_task', None)
        self.num_labels = kwargs.pop('num_labels', 2)
        self.output_attentions = kwargs.pop('output_attentions', False)
        self.output_hidden_states = kwargs.pop('output_hidden_states', False)
        self.torchscript = kwargs.pop('torchscript', False)

    def save_pretrained(self, save_directory):
        """ Save a configuration object to a directory, so that it
            can be re-loaded using the `from_pretrained(save_directory)` class method.
        """
        assert os.path.isdir(save_directory), 'Saving path should be a directory where the model and configuration can be saved'
        output_config_file = os.path.join(save_directory, CONFIG_NAME)
        self.to_json_file(output_config_file)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *input, **kwargs):
        """ Instantiate a PretrainedConfig from a pre-trained model configuration.

        Params:
            **pretrained_model_name_or_path**: either:
                - a string with the `shortcut name` of a pre-trained model configuration to load from cache
                    or download and cache if not already stored in cache (e.g. 'bert-base-uncased').
                - a path to a `directory` containing a configuration file saved
                    using the `save_pretrained(save_directory)` method.
                - a path or url to a saved configuration `file`.
            **cache_dir**: (`optional`) string:
                Path to a directory in which a downloaded pre-trained model
                configuration should be cached if the standard cache should not be used.
            **kwargs**: (`optional`) dict:
                Dictionnary of key, values to update the configuration object after loading.
                Can be used to override selected configuration parameters.

        Examples::

            >>> config = BertConfig.from_pretrained('bert-base-uncased')    # Download configuration from S3 and cache.
            >>> config = BertConfig.from_pretrained('./test/saved_model/')  # E.g. config (or model) was saved using `save_pretrained('./test/saved_model/')`
            >>> config = BertConfig.from_pretrained('./test/saved_model/my_configuration.json')
            >>> config = BertConfig.from_pretrained('bert-base-uncased', output_attention=True)
            >>> assert config.output_attention == True

        """
        cache_dir = kwargs.pop('cache_dir', None)
        if pretrained_model_name_or_path in cls.pretrained_config_archive_map:
            config_file = cls.pretrained_config_archive_map[pretrained_model_name_or_path]
        elif os.path.isdir(pretrained_model_name_or_path):
            config_file = os.path.join(pretrained_model_name_or_path, CONFIG_NAME)
        else:
            config_file = pretrained_model_name_or_path
        try:
            resolved_config_file = cached_path(config_file, cache_dir=cache_dir)
        except EnvironmentError:
            if pretrained_model_name_or_path in cls.pretrained_config_archive_map:
                logger.error("Couldn't reach server at '{}' to download pretrained model configuration file.".format(config_file))
            else:
                logger.error("Model name '{}' was not found in model name list ({}). We assumed '{}' was a path or url but couldn't find any file associated to this path or url.".format(pretrained_model_name_or_path, ', '.join(cls.pretrained_config_archive_map.keys()), config_file))
            return None
        if resolved_config_file == config_file:
            logger.info('loading configuration file {}'.format(config_file))
        else:
            logger.info('loading configuration file {} from cache at {}'.format(config_file, resolved_config_file))
        config = cls.from_json_file(resolved_config_file)
        to_remove = []
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
                to_remove.append(key)
        for key in to_remove:
            kwargs.pop(key, None)
        logger.info('Model config %s', config)
        return config

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `Config` from a Python dictionary of parameters."""
        config = cls(vocab_size_or_config_json_file=-1)
        for key, value in json_object.items():
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `BertConfig` from a json file of parameters."""
        with open(json_file, 'r', encoding='utf-8') as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + '\n'

    def to_json_file(self, json_file_path):
        """ Save this instance to a json file."""
        with open(json_file_path, 'w', encoding='utf-8') as writer:
            writer.write(self.to_json_string())


TF_WEIGHTS_NAME = 'model.ckpt'


WEIGHTS_NAME = 'pytorch_model.bin'


class PreTrainedModel(nn.Module):
    """ Base class for all models.

        :class:`~pytorch_transformers.PreTrainedModel` takes care of storing the configuration of the models and handles methods for loading/downloading/saving models
        as well as a few methods commons to all models to (i) resize the input embeddings and (ii) prune heads in the self-attention heads.

        Class attributes (overridden by derived classes):
            - ``config_class``: a class derived from :class:`~pytorch_transformers.PretrainedConfig` to use as configuration class for this model architecture.
            - ``pretrained_model_archive_map``: a python ``dict`` of with `short-cut-names` (string) as keys and `url` (string) of associated pretrained weights as values.
            - ``load_tf_weights``: a python ``method`` for loading a TensorFlow checkpoint in a PyTorch model, taking as arguments:

                - ``model``: an instance of the relevant subclass of :class:`~pytorch_transformers.PreTrainedModel`,
                - ``config``: an instance of the relevant subclass of :class:`~pytorch_transformers.PretrainedConfig`,
                - ``path``: a path (string) to the TensorFlow checkpoint.

            - ``base_model_prefix``: a string indicating the attribute associated to the base model in derived classes of the same architecture adding modules on top of the base model.
    """
    config_class = None
    pretrained_model_archive_map = {}
    load_tf_weights = lambda model, config, path: None
    base_model_prefix = ''

    def __init__(self, config, *inputs, **kwargs):
        super(PreTrainedModel, self).__init__()
        if not isinstance(config, PretrainedConfig):
            raise ValueError('Parameter config in `{}(config)` should be an instance of class `PretrainedConfig`. To create a model from a pretrained model use `model = {}.from_pretrained(PRETRAINED_MODEL_NAME)`'.format(self.__class__.__name__, self.__class__.__name__))
        self.config = config

    def _get_resized_embeddings(self, old_embeddings, new_num_tokens=None):
        """ Build a resized Embedding Module from a provided token Embedding Module.
            Increasing the size will add newly initialized vectors at the end
            Reducing the size will remove vectors from the end

        Args:
            new_num_tokens: (`optional`) int
                New number of tokens in the embedding matrix.
                Increasing the size will add newly initialized vectors at the end
                Reducing the size will remove vectors from the end
                If not provided or None: return the provided token Embedding Module.
        Return: ``torch.nn.Embeddings``
            Pointer to the resized Embedding Module or the old Embedding Module if new_num_tokens is None
        """
        if new_num_tokens is None:
            return old_embeddings
        old_num_tokens, old_embedding_dim = old_embeddings.weight.size()
        if old_num_tokens == new_num_tokens:
            return old_embeddings
        new_embeddings = nn.Embedding(new_num_tokens, old_embedding_dim)
        new_embeddings
        self._init_weights(new_embeddings)
        num_tokens_to_copy = min(old_num_tokens, new_num_tokens)
        new_embeddings.weight.data[:num_tokens_to_copy, :] = old_embeddings.weight.data[:num_tokens_to_copy, :]
        return new_embeddings

    def _tie_or_clone_weights(self, first_module, second_module):
        """ Tie or clone module weights depending of weither we are using TorchScript or not
        """
        if self.config.torchscript:
            first_module.weight = nn.Parameter(second_module.weight.clone())
        else:
            first_module.weight = second_module.weight
        if hasattr(first_module, 'bias') and first_module.bias is not None:
            first_module.bias.data = torch.nn.functional.pad(first_module.bias.data, (0, first_module.weight.shape[0] - first_module.bias.shape[0]), 'constant', 0)

    def resize_token_embeddings(self, new_num_tokens=None):
        """ Resize input token embeddings matrix of the model if new_num_tokens != config.vocab_size.
        Take care of tying weights embeddings afterwards if the model class has a `tie_weights()` method.

        Arguments:

            new_num_tokens: (`optional`) int:
                New number of tokens in the embedding matrix. Increasing the size will add newly initialized vectors at the end. Reducing the size will remove vectors from the end.
                If not provided or None: does nothing and just returns a pointer to the input tokens ``torch.nn.Embeddings`` Module of the model.

        Return: ``torch.nn.Embeddings``
            Pointer to the input tokens Embeddings Module of the model
        """
        base_model = getattr(self, self.base_model_prefix, self)
        model_embeds = base_model._resize_token_embeddings(new_num_tokens)
        if new_num_tokens is None:
            return model_embeds
        self.config.vocab_size = new_num_tokens
        base_model.vocab_size = new_num_tokens
        if hasattr(self, 'tie_weights'):
            self.tie_weights()
        return model_embeds

    def init_weights(self):
        """ Initialize and prunes weights if needed. """
        self.apply(self._init_weights)
        if self.config.pruned_heads:
            self.prune_heads(self.config.pruned_heads)

    def prune_heads(self, heads_to_prune):
        """ Prunes heads of the base model.

            Arguments:

                heads_to_prune: dict with keys being selected layer indices (`int`) and associated values being the list of heads to prune in said layer (list of `int`).
                E.g. {1: [0, 2], 2: [2, 3]} will prune heads 0 and 2 on layer 1 and heads 2 and 3 on layer 2.
        """
        base_model = getattr(self, self.base_model_prefix, self)
        for layer, heads in heads_to_prune.items():
            union_heads = set(self.config.pruned_heads.get(layer, [])) | set(heads)
            self.config.pruned_heads[layer] = list(union_heads)
        base_model._prune_heads(heads_to_prune)

    def save_pretrained(self, save_directory):
        """ Save a model and its configuration file to a directory, so that it
            can be re-loaded using the `:func:`~pytorch_transformers.PreTrainedModel.from_pretrained`` class method.
        """
        assert os.path.isdir(save_directory), 'Saving path should be a directory where the model and configuration can be saved'
        model_to_save = self.module if hasattr(self, 'module') else self
        model_to_save.config.save_pretrained(save_directory)
        output_model_file = os.path.join(save_directory, WEIGHTS_NAME)
        torch.save(model_to_save.state_dict(), output_model_file)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """Instantiate a pretrained pytorch model from a pre-trained model configuration.

        The model is set in evaluation mode by default using ``model.eval()`` (Dropout modules are deactivated)
        To train the model, you should first set it back in training mode with ``model.train()``

        The warning ``Weights from XXX not initialized from pretrained model`` means that the weights of XXX do not come pre-trained with the rest of the model.
        It is up to you to train those weights with a downstream fine-tuning task.

        The warning ``Weights from XXX not used in YYY`` means that the layer XXX is not used by YYY, therefore those weights are discarded.

        Parameters:
            pretrained_model_name_or_path: either:

                - a string with the `shortcut name` of a pre-trained model to load from cache or download, e.g.: ``bert-base-uncased``.
                - a path to a `directory` containing model weights saved using :func:`~pytorch_transformers.PreTrainedModel.save_pretrained`, e.g.: ``./my_model_directory/``.
                - a path or url to a `tensorflow index checkpoint file` (e.g. `./tf_model/model.ckpt.index`). In this case, ``from_tf`` should be set to True and a configuration object should be provided as ``config`` argument. This loading path is slower than converting the TensorFlow checkpoint in a PyTorch model using the provided conversion scripts and loading the PyTorch model afterwards.

            model_args: (`optional`) Sequence of positional arguments:
                All remaning positional arguments will be passed to the underlying model's ``__init__`` method

            config: (`optional`) instance of a class derived from :class:`~pytorch_transformers.PretrainedConfig`:
                Configuration for the model to use instead of an automatically loaded configuation. Configuration can be automatically loaded when:

                - the model is a model provided by the library (loaded with the ``shortcut-name`` string of a pretrained model), or
                - the model was saved using :func:`~pytorch_transformers.PreTrainedModel.save_pretrained` and is reloaded by suppling the save directory.
                - the model is loaded by suppling a local directory as ``pretrained_model_name_or_path`` and a configuration JSON file named `config.json` is found in the directory.

            state_dict: (`optional`) dict:
                an optional state dictionnary for the model to use instead of a state dictionary loaded from saved weights file.
                This option can be used if you want to create a model from a pretrained configuration but load your own weights.
                In this case though, you should check if using :func:`~pytorch_transformers.PreTrainedModel.save_pretrained` and :func:`~pytorch_transformers.PreTrainedModel.from_pretrained` is not a simpler option.

            cache_dir: (`optional`) string:
                Path to a directory in which a downloaded pre-trained model
                configuration should be cached if the standard cache should not be used.

            force_download: (`optional`) boolean, default False:
                Force to (re-)download the model weights and configuration files and override the cached versions if they exists.

            proxies: (`optional`) dict, default None:
                A dictionary of proxy servers to use by protocol or endpoint, e.g.: {'http': 'foo.bar:3128', 'http://hostname': 'foo.bar:4012'}.
                The proxies are used on each request.

            output_loading_info: (`optional`) boolean:
                Set to ``True`` to also return a dictionnary containing missing keys, unexpected keys and error messages.

            kwargs: (`optional`) Remaining dictionary of keyword arguments:
                Can be used to update the configuration object (after it being loaded) and initiate the model. (e.g. ``output_attention=True``). Behave differently depending on whether a `config` is provided or automatically loaded:

                - If a configuration is provided with ``config``, ``**kwargs`` will be directly passed to the underlying model's ``__init__`` method (we assume all relevant updates to the configuration have already been done)
                - If a configuration is not provided, ``kwargs`` will be first passed to the configuration class initialization function (:func:`~pytorch_transformers.PretrainedConfig.from_pretrained`). Each key of ``kwargs`` that corresponds to a configuration attribute will be used to override said attribute with the supplied ``kwargs`` value. Remaining keys that do not correspond to any configuration attribute will be passed to the underlying model's ``__init__`` function.

        Examples::

            model = BertModel.from_pretrained('bert-base-uncased')    # Download model and configuration from S3 and cache.
            model = BertModel.from_pretrained('./test/saved_model/')  # E.g. model was saved using `save_pretrained('./test/saved_model/')`
            model = BertModel.from_pretrained('bert-base-uncased', output_attention=True)  # Update configuration during loading
            assert model.config.output_attention == True
            # Loading from a TF checkpoint file instead of a PyTorch model (slower)
            config = BertConfig.from_json_file('./tf_model/my_tf_model_config.json')
            model = BertModel.from_pretrained('./tf_model/my_tf_checkpoint.ckpt.index', from_tf=True, config=config)

        """
        config = kwargs.pop('config', None)
        state_dict = kwargs.pop('state_dict', None)
        cache_dir = kwargs.pop('cache_dir', None)
        from_tf = kwargs.pop('from_tf', False)
        force_download = kwargs.pop('force_download', False)
        proxies = kwargs.pop('proxies', None)
        output_loading_info = kwargs.pop('output_loading_info', False)
        if config is None:
            config, model_kwargs = cls.config_class.from_pretrained(pretrained_model_name_or_path, *model_args, cache_dir=cache_dir, return_unused_kwargs=True, force_download=force_download, **kwargs)
        else:
            model_kwargs = kwargs
        if pretrained_model_name_or_path in cls.pretrained_model_archive_map:
            archive_file = cls.pretrained_model_archive_map[pretrained_model_name_or_path]
        elif os.path.isdir(pretrained_model_name_or_path):
            if from_tf:
                archive_file = os.path.join(pretrained_model_name_or_path, TF_WEIGHTS_NAME + '.index')
            else:
                archive_file = os.path.join(pretrained_model_name_or_path, WEIGHTS_NAME)
        elif from_tf:
            archive_file = pretrained_model_name_or_path + '.index'
        else:
            archive_file = pretrained_model_name_or_path
        try:
            resolved_archive_file = cached_path(archive_file, cache_dir=cache_dir, force_download=force_download, proxies=proxies)
        except EnvironmentError as e:
            if pretrained_model_name_or_path in cls.pretrained_model_archive_map:
                logger.error("Couldn't reach server at '{}' to download pretrained weights.".format(archive_file))
            else:
                logger.error("Model name '{}' was not found in model name list ({}). We assumed '{}' was a path or url but couldn't find any file associated to this path or url.".format(pretrained_model_name_or_path, ', '.join(cls.pretrained_model_archive_map.keys()), archive_file))
            raise e
        if resolved_archive_file == archive_file:
            logger.info('loading weights file {}'.format(archive_file))
        else:
            logger.info('loading weights file {} from cache at {}'.format(archive_file, resolved_archive_file))
        model = cls(config, *model_args, **model_kwargs)
        if state_dict is None and not from_tf:
            state_dict = torch.load(resolved_archive_file, map_location='cpu')
        if from_tf:
            return cls.load_tf_weights(model, config, resolved_archive_file[:-6])
        old_keys = []
        new_keys = []
        for key in state_dict.keys():
            new_key = None
            if 'gamma' in key:
                new_key = key.replace('gamma', 'weight')
            if 'beta' in key:
                new_key = key.replace('beta', 'bias')
            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)
        for old_key, new_key in zip(old_keys, new_keys):
            state_dict[new_key] = state_dict.pop(old_key)
        missing_keys = []
        unexpected_keys = []
        error_msgs = []
        metadata = getattr(state_dict, '_metadata', None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        def load(module, prefix=''):
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            module._load_from_state_dict(state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + '.')
        start_prefix = ''
        model_to_load = model
        if not hasattr(model, cls.base_model_prefix) and any(s.startswith(cls.base_model_prefix) for s in state_dict.keys()):
            start_prefix = cls.base_model_prefix + '.'
        if hasattr(model, cls.base_model_prefix) and not any(s.startswith(cls.base_model_prefix) for s in state_dict.keys()):
            model_to_load = getattr(model, cls.base_model_prefix)
        load(model_to_load, prefix=start_prefix)
        if len(missing_keys) > 0:
            logger.info('Weights of {} not initialized from pretrained model: {}'.format(model.__class__.__name__, missing_keys))
        if len(unexpected_keys) > 0:
            logger.info('Weights from pretrained model not used in {}: {}'.format(model.__class__.__name__, unexpected_keys))
        if len(error_msgs) > 0:
            raise RuntimeError('Error(s) in loading state_dict for {}:\n\t{}'.format(model.__class__.__name__, '\n\t'.join(error_msgs)))
        if hasattr(model, 'tie_weights'):
            model.tie_weights()
        model.eval()
        if output_loading_info:
            loading_info = {'missing_keys': missing_keys, 'unexpected_keys': unexpected_keys, 'error_msgs': error_msgs}
            return model, loading_info
        return model


class Conv1D(nn.Module):

    def __init__(self, nf, nx):
        """ Conv1D layer as defined by Radford et al. for OpenAI GPT (and also used in GPT-2)
            Basically works like a Linear layer but the weights are transposed
        """
        super(Conv1D, self).__init__()
        self.nf = nf
        w = torch.empty(nx, nf)
        nn.init.normal_(w, std=0.02)
        self.weight = nn.Parameter(w)
        self.bias = nn.Parameter(torch.zeros(nf))

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        x = x.view(*size_out)
        return x


class PoolerStartLogits(nn.Module):
    """ Compute SQuAD start_logits from sequence hidden states. """

    def __init__(self, config):
        super(PoolerStartLogits, self).__init__()
        self.dense = nn.Linear(config.hidden_size, 1)

    def forward(self, hidden_states, p_mask=None):
        """ Args:
            **p_mask**: (`optional`) ``torch.FloatTensor`` of shape `(batch_size, seq_len)`
                invalid position mask such as query and special symbols (PAD, SEP, CLS)
                1.0 means token should be masked.
        """
        x = self.dense(hidden_states).squeeze(-1)
        if p_mask is not None:
            if next(self.parameters()).dtype == torch.float16:
                x = x * (1 - p_mask) - 65500 * p_mask
            else:
                x = x * (1 - p_mask) - 1e+30 * p_mask
        return x


class PoolerEndLogits(nn.Module):
    """ Compute SQuAD end_logits from sequence hidden states and start token hidden state.
    """

    def __init__(self, config):
        super(PoolerEndLogits, self).__init__()
        self.dense_0 = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.activation = nn.Tanh()
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dense_1 = nn.Linear(config.hidden_size, 1)

    def forward(self, hidden_states, start_states=None, start_positions=None, p_mask=None):
        """ Args:
            One of ``start_states``, ``start_positions`` should be not None.
            If both are set, ``start_positions`` overrides ``start_states``.

            **start_states**: ``torch.LongTensor`` of shape identical to hidden_states
                hidden states of the first tokens for the labeled span.
            **start_positions**: ``torch.LongTensor`` of shape ``(batch_size,)``
                position of the first token for the labeled span:
            **p_mask**: (`optional`) ``torch.FloatTensor`` of shape ``(batch_size, seq_len)``
                Mask of invalid position such as query and special symbols (PAD, SEP, CLS)
                1.0 means token should be masked.
        """
        assert start_states is not None or start_positions is not None, 'One of start_states, start_positions should be not None'
        if start_positions is not None:
            slen, hsz = hidden_states.shape[-2:]
            start_positions = start_positions[:, None, None].expand(-1, -1, hsz)
            start_states = hidden_states.gather(-2, start_positions)
            start_states = start_states.expand(-1, slen, -1)
        x = self.dense_0(torch.cat([hidden_states, start_states], dim=-1))
        x = self.activation(x)
        x = self.LayerNorm(x)
        x = self.dense_1(x).squeeze(-1)
        if p_mask is not None:
            x = x * (1 - p_mask) - 1e+30 * p_mask
        return x


class PoolerAnswerClass(nn.Module):
    """ Compute SQuAD 2.0 answer class from classification and start tokens hidden states. """

    def __init__(self, config):
        super(PoolerAnswerClass, self).__init__()
        self.dense_0 = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.activation = nn.Tanh()
        self.dense_1 = nn.Linear(config.hidden_size, 1, bias=False)

    def forward(self, hidden_states, start_states=None, start_positions=None, cls_index=None):
        """
        Args:
            One of ``start_states``, ``start_positions`` should be not None.
            If both are set, ``start_positions`` overrides ``start_states``.

            **start_states**: ``torch.LongTensor`` of shape identical to ``hidden_states``.
                hidden states of the first tokens for the labeled span.
            **start_positions**: ``torch.LongTensor`` of shape ``(batch_size,)``
                position of the first token for the labeled span.
            **cls_index**: torch.LongTensor of shape ``(batch_size,)``
                position of the CLS token. If None, take the last token.

            note(Original repo):
                no dependency on end_feature so that we can obtain one single `cls_logits`
                for each sample
        """
        hsz = hidden_states.shape[-1]
        assert start_states is not None or start_positions is not None, 'One of start_states, start_positions should be not None'
        if start_positions is not None:
            start_positions = start_positions[:, None, None].expand(-1, -1, hsz)
            start_states = hidden_states.gather(-2, start_positions).squeeze(-2)
        if cls_index is not None:
            cls_index = cls_index[:, None, None].expand(-1, -1, hsz)
            cls_token_state = hidden_states.gather(-2, cls_index).squeeze(-2)
        else:
            cls_token_state = hidden_states[:, -1, :]
        x = self.dense_0(torch.cat([start_states, cls_token_state], dim=-1))
        x = self.activation(x)
        x = self.dense_1(x).squeeze(-1)
        return x


class SQuADHead(nn.Module):
    """ A SQuAD head inspired by XLNet.

    Parameters:
        config (:class:`~pytorch_transformers.XLNetConfig`): Model configuration class with all the parameters of the model.

    Inputs:
        **hidden_states**: ``torch.FloatTensor`` of shape ``(batch_size, seq_len, hidden_size)``
            hidden states of sequence tokens
        **start_positions**: ``torch.LongTensor`` of shape ``(batch_size,)``
            position of the first token for the labeled span.
        **end_positions**: ``torch.LongTensor`` of shape ``(batch_size,)``
            position of the last token for the labeled span.
        **cls_index**: torch.LongTensor of shape ``(batch_size,)``
            position of the CLS token. If None, take the last token.
        **is_impossible**: ``torch.LongTensor`` of shape ``(batch_size,)``
            Whether the question has a possible answer in the paragraph or not.
        **p_mask**: (`optional`) ``torch.FloatTensor`` of shape ``(batch_size, seq_len)``
            Mask of invalid position such as query and special symbols (PAD, SEP, CLS)
            1.0 means token should be masked.

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned if both ``start_positions`` and ``end_positions`` are provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Classification loss as the sum of start token, end token (and is_impossible if provided) classification losses.
        **start_top_log_probs**: (`optional`, returned if ``start_positions`` or ``end_positions`` is not provided)
            ``torch.FloatTensor`` of shape ``(batch_size, config.start_n_top)``
            Log probabilities for the top config.start_n_top start token possibilities (beam-search).
        **start_top_index**: (`optional`, returned if ``start_positions`` or ``end_positions`` is not provided)
            ``torch.LongTensor`` of shape ``(batch_size, config.start_n_top)``
            Indices for the top config.start_n_top start token possibilities (beam-search).
        **end_top_log_probs**: (`optional`, returned if ``start_positions`` or ``end_positions`` is not provided)
            ``torch.FloatTensor`` of shape ``(batch_size, config.start_n_top * config.end_n_top)``
            Log probabilities for the top ``config.start_n_top * config.end_n_top`` end token possibilities (beam-search).
        **end_top_index**: (`optional`, returned if ``start_positions`` or ``end_positions`` is not provided)
            ``torch.LongTensor`` of shape ``(batch_size, config.start_n_top * config.end_n_top)``
            Indices for the top ``config.start_n_top * config.end_n_top`` end token possibilities (beam-search).
        **cls_logits**: (`optional`, returned if ``start_positions`` or ``end_positions`` is not provided)
            ``torch.FloatTensor`` of shape ``(batch_size,)``
            Log probabilities for the ``is_impossible`` label of the answers.
    """

    def __init__(self, config):
        super(SQuADHead, self).__init__()
        self.start_n_top = config.start_n_top
        self.end_n_top = config.end_n_top
        self.start_logits = PoolerStartLogits(config)
        self.end_logits = PoolerEndLogits(config)
        self.answer_class = PoolerAnswerClass(config)

    def forward(self, hidden_states, start_positions=None, end_positions=None, cls_index=None, is_impossible=None, p_mask=None):
        outputs = ()
        start_logits = self.start_logits(hidden_states, p_mask=p_mask)
        if start_positions is not None and end_positions is not None:
            for x in (start_positions, end_positions, cls_index, is_impossible):
                if x is not None and x.dim() > 1:
                    x.squeeze_(-1)
            end_logits = self.end_logits(hidden_states, start_positions=start_positions, p_mask=p_mask)
            loss_fct = CrossEntropyLoss()
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            if cls_index is not None and is_impossible is not None:
                cls_logits = self.answer_class(hidden_states, start_positions=start_positions, cls_index=cls_index)
                loss_fct_cls = nn.BCEWithLogitsLoss()
                cls_loss = loss_fct_cls(cls_logits, is_impossible)
                total_loss += cls_loss * 0.5
            outputs = (total_loss,) + outputs
        else:
            bsz, slen, hsz = hidden_states.size()
            start_log_probs = F.softmax(start_logits, dim=-1)
            start_top_log_probs, start_top_index = torch.topk(start_log_probs, self.start_n_top, dim=-1)
            start_top_index_exp = start_top_index.unsqueeze(-1).expand(-1, -1, hsz)
            start_states = torch.gather(hidden_states, -2, start_top_index_exp)
            start_states = start_states.unsqueeze(1).expand(-1, slen, -1, -1)
            hidden_states_expanded = hidden_states.unsqueeze(2).expand_as(start_states)
            p_mask = p_mask.unsqueeze(-1) if p_mask is not None else None
            end_logits = self.end_logits(hidden_states_expanded, start_states=start_states, p_mask=p_mask)
            end_log_probs = F.softmax(end_logits, dim=1)
            end_top_log_probs, end_top_index = torch.topk(end_log_probs, self.end_n_top, dim=1)
            end_top_log_probs = end_top_log_probs.view(-1, self.start_n_top * self.end_n_top)
            end_top_index = end_top_index.view(-1, self.start_n_top * self.end_n_top)
            start_states = torch.einsum('blh,bl->bh', hidden_states, start_log_probs)
            cls_logits = self.answer_class(hidden_states, start_states=start_states, cls_index=cls_index)
            outputs = (start_top_log_probs, start_top_index, end_top_log_probs, end_top_index, cls_logits) + outputs
        return outputs


class SequenceSummary(nn.Module):
    """ Compute a single vector summary of a sequence hidden states according to various possibilities:
        Args of the config class:
            summary_type:
                - 'last' => [default] take the last token hidden state (like XLNet)
                - 'first' => take the first token hidden state (like Bert)
                - 'mean' => take the mean of all tokens hidden states
                - 'cls_index' => supply a Tensor of classification token position (GPT/GPT-2)
                - 'attn' => Not implemented now, use multi-head attention
            summary_use_proj: Add a projection after the vector extraction
            summary_proj_to_labels: If True, the projection outputs to config.num_labels classes (otherwise to hidden_size). Default: False.
            summary_activation: 'tanh' => add a tanh activation to the output, Other => no activation. Default
            summary_first_dropout: Add a dropout before the projection and activation
            summary_last_dropout: Add a dropout after the projection and activation
    """

    def __init__(self, config):
        super(SequenceSummary, self).__init__()
        self.summary_type = config.summary_type if hasattr(config, 'summary_use_proj') else 'last'
        if self.summary_type == 'attn':
            raise NotImplementedError
        self.summary = Identity()
        if hasattr(config, 'summary_use_proj') and config.summary_use_proj:
            if hasattr(config, 'summary_proj_to_labels') and config.summary_proj_to_labels and config.num_labels > 0:
                num_classes = config.num_labels
            else:
                num_classes = config.hidden_size
            self.summary = nn.Linear(config.hidden_size, num_classes)
        self.activation = Identity()
        if hasattr(config, 'summary_activation') and config.summary_activation == 'tanh':
            self.activation = nn.Tanh()
        self.first_dropout = Identity()
        if hasattr(config, 'summary_first_dropout') and config.summary_first_dropout > 0:
            self.first_dropout = nn.Dropout(config.summary_first_dropout)
        self.last_dropout = Identity()
        if hasattr(config, 'summary_last_dropout') and config.summary_last_dropout > 0:
            self.last_dropout = nn.Dropout(config.summary_last_dropout)

    def forward(self, hidden_states, cls_index=None):
        """ hidden_states: float Tensor in shape [bsz, seq_len, hidden_size], the hidden-states of the last layer.
            cls_index: [optional] position of the classification token if summary_type == 'cls_index',
                shape (bsz,) or more generally (bsz, ...) where ... are optional leading dimensions of hidden_states.
                if summary_type == 'cls_index' and cls_index is None:
                    we take the last token of the sequence as classification token
        """
        if self.summary_type == 'last':
            output = hidden_states[:, -1]
        elif self.summary_type == 'first':
            output = hidden_states[:, 0]
        elif self.summary_type == 'mean':
            output = hidden_states.mean(dim=1)
        elif self.summary_type == 'cls_index':
            if cls_index is None:
                cls_index = torch.full_like(hidden_states[..., :1, :], hidden_states.shape[-2] - 1, dtype=torch.long)
            else:
                cls_index = cls_index.unsqueeze(-1).unsqueeze(-1)
                cls_index = cls_index.expand((-1,) * (cls_index.dim() - 1) + (hidden_states.size(-1),))
            output = hidden_states.gather(-2, cls_index).squeeze(-2)
        elif self.summary_type == 'attn':
            raise NotImplementedError
        output = self.first_dropout(output)
        output = self.summary(output)
        output = self.activation(output)
        output = self.last_dropout(output)
        return output


BERT_PRETRAINED_MODEL_ARCHIVE_MAP = {'bert-base-uncased': 'https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-pytorch_model.bin', 'bert-large-uncased': 'https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-pytorch_model.bin', 'bert-base-cased': 'https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased-pytorch_model.bin', 'bert-large-cased': 'https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-pytorch_model.bin', 'bert-base-multilingual-uncased': 'https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-uncased-pytorch_model.bin', 'bert-base-multilingual-cased': 'https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-cased-pytorch_model.bin', 'bert-base-chinese': 'https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese-pytorch_model.bin', 'bert-base-german-cased': 'https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-german-cased-pytorch_model.bin', 'bert-large-uncased-whole-word-masking': 'https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-whole-word-masking-pytorch_model.bin', 'bert-large-cased-whole-word-masking': 'https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-whole-word-masking-pytorch_model.bin', 'bert-large-uncased-whole-word-masking-finetuned-squad': 'https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-whole-word-masking-finetuned-squad-pytorch_model.bin', 'bert-large-cased-whole-word-masking-finetuned-squad': 'https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-whole-word-masking-finetuned-squad-pytorch_model.bin', 'bert-base-cased-finetuned-mrpc': 'https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased-finetuned-mrpc-pytorch_model.bin'}


BERT_PRETRAINED_CONFIG_ARCHIVE_MAP = {'bert-base-uncased': 'https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-config.json', 'bert-large-uncased': 'https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-config.json', 'bert-base-cased': 'https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased-config.json', 'bert-large-cased': 'https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-config.json', 'bert-base-multilingual-uncased': 'https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-uncased-config.json', 'bert-base-multilingual-cased': 'https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-cased-config.json', 'bert-base-chinese': 'https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese-config.json', 'bert-base-german-cased': 'https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-german-cased-config.json', 'bert-large-uncased-whole-word-masking': 'https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-whole-word-masking-config.json', 'bert-large-cased-whole-word-masking': 'https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-whole-word-masking-config.json', 'bert-large-uncased-whole-word-masking-finetuned-squad': 'https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-whole-word-masking-finetuned-squad-config.json', 'bert-large-cased-whole-word-masking-finetuned-squad': 'https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-whole-word-masking-finetuned-squad-config.json', 'bert-base-cased-finetuned-mrpc': 'https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased-finetuned-mrpc-config.json'}


class BertConfig(PretrainedConfig):
    """
        :class:`~pytorch_transformers.BertConfig` is the configuration class to store the configuration of a
        `BertModel`.


        Arguments:
            vocab_size_or_config_json_file: Vocabulary size of `inputs_ids` in `BertModel`.
            hidden_size: Size of the encoder layers and the pooler layer.
            num_hidden_layers: Number of hidden layers in the Transformer encoder.
            num_attention_heads: Number of attention heads for each attention layer in
                the Transformer encoder.
            intermediate_size: The size of the "intermediate" (i.e., feed-forward)
                layer in the Transformer encoder.
            hidden_act: The non-linear activation function (function or string) in the
                encoder and pooler. If string, "gelu", "relu" and "swish" are supported.
            hidden_dropout_prob: The dropout probabilitiy for all fully connected
                layers in the embeddings, encoder, and pooler.
            attention_probs_dropout_prob: The dropout ratio for the attention
                probabilities.
            max_position_embeddings: The maximum sequence length that this model might
                ever be used with. Typically set this to something large just in case
                (e.g., 512 or 1024 or 2048).
            type_vocab_size: The vocabulary size of the `token_type_ids` passed into
                `BertModel`.
            initializer_range: The sttdev of the truncated_normal_initializer for
                initializing all weight matrices.
            layer_norm_eps: The epsilon used by LayerNorm.
    """
    pretrained_config_archive_map = BERT_PRETRAINED_CONFIG_ARCHIVE_MAP

    def __init__(self, vocab_size_or_config_json_file=30522, hidden_size=768, num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072, hidden_act='gelu', hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1, max_position_embeddings=512, type_vocab_size=2, initializer_range=0.02, layer_norm_eps=1e-12, **kwargs):
        super(BertConfig, self).__init__(**kwargs)
        if isinstance(vocab_size_or_config_json_file, str) or sys.version_info[0] == 2 and isinstance(vocab_size_or_config_json_file, unicode):
            with open(vocab_size_or_config_json_file, 'r', encoding='utf-8') as reader:
                json_config = json.loads(reader.read())
            for key, value in json_config.items():
                self.__dict__[key] = value
        elif isinstance(vocab_size_or_config_json_file, int):
            self.vocab_size = vocab_size_or_config_json_file
            self.hidden_size = hidden_size
            self.num_hidden_layers = num_hidden_layers
            self.num_attention_heads = num_attention_heads
            self.hidden_act = hidden_act
            self.intermediate_size = intermediate_size
            self.hidden_dropout_prob = hidden_dropout_prob
            self.attention_probs_dropout_prob = attention_probs_dropout_prob
            self.max_position_embeddings = max_position_embeddings
            self.type_vocab_size = type_vocab_size
            self.initializer_range = initializer_range
            self.layer_norm_eps = layer_norm_eps
        else:
            raise ValueError('First argument must be either a vocabulary size (int)or the path to a pretrained model config file (str)')


def load_tf_weights_in_bert(model, config, tf_checkpoint_path):
    """ Load tf checkpoints in a pytorch model.
    """
    try:
        import re
        import numpy as np
        import tensorflow as tf
    except ImportError:
        logger.error('Loading a TensorFlow model in PyTorch, requires TensorFlow to be installed. Please see https://www.tensorflow.org/install/ for installation instructions.')
        raise
    tf_path = os.path.abspath(tf_checkpoint_path)
    logger.info('Converting TensorFlow checkpoint from {}'.format(tf_path))
    init_vars = tf.train.list_variables(tf_path)
    names = []
    arrays = []
    for name, shape in init_vars:
        logger.info('Loading TF weight {} with shape {}'.format(name, shape))
        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        arrays.append(array)
    for name, array in zip(names, arrays):
        name = name.split('/')
        if any(n in ['adam_v', 'adam_m', 'global_step'] for n in name):
            logger.info('Skipping {}'.format('/'.join(name)))
            continue
        pointer = model
        for m_name in name:
            if re.fullmatch('[A-Za-z]+_\\d+', m_name):
                l = re.split('_(\\d+)', m_name)
            else:
                l = [m_name]
            if l[0] == 'kernel' or l[0] == 'gamma':
                pointer = getattr(pointer, 'weight')
            elif l[0] == 'output_bias' or l[0] == 'beta':
                pointer = getattr(pointer, 'bias')
            elif l[0] == 'output_weights':
                pointer = getattr(pointer, 'weight')
            elif l[0] == 'squad':
                pointer = getattr(pointer, 'classifier')
            else:
                try:
                    pointer = getattr(pointer, l[0])
                except AttributeError:
                    logger.info('Skipping {}'.format('/'.join(name)))
                    continue
            if len(l) >= 2:
                num = int(l[1])
                pointer = pointer[num]
        if m_name[-11:] == '_embeddings':
            pointer = getattr(pointer, 'weight')
        elif m_name == 'kernel':
            array = np.transpose(array)
        try:
            assert pointer.shape == array.shape
        except AssertionError as e:
            e.args += pointer.shape, array.shape
            raise
        logger.info('Initialize PyTorch weight {}'.format(name))
        pointer.data = torch.from_numpy(array)
    return model


class BertPreTrainedModel(PreTrainedModel):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """
    config_class = BertConfig
    pretrained_model_archive_map = BERT_PRETRAINED_MODEL_ARCHIVE_MAP
    load_tf_weights = load_tf_weights_in_bert
    base_model_prefix = 'bert'

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


BERT_INPUTS_DOCSTRING = """
    Inputs:
        **input_ids**: ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            Indices of input sequence tokens in the vocabulary.
            To match pre-training, BERT input sequence should be formatted with [CLS] and [SEP] tokens as follows:

            (a) For sequence pairs:

                ``tokens:         [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]``
                
                ``token_type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1``

            (b) For single sequences:

                ``tokens:         [CLS] the dog is hairy . [SEP]``
                
                ``token_type_ids:   0   0   0   0  0     0   0``

            Bert is a model with absolute position embeddings so it's usually advised to pad the inputs on
            the right rather than the left.

            Indices can be obtained using :class:`pytorch_transformers.BertTokenizer`.
            See :func:`pytorch_transformers.PreTrainedTokenizer.encode` and
            :func:`pytorch_transformers.PreTrainedTokenizer.convert_tokens_to_ids` for details.
        **attention_mask**: (`optional`) ``torch.FloatTensor`` of shape ``(batch_size, sequence_length)``:
            Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            ``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens.
        **token_type_ids**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            Segment token indices to indicate first and second portions of the inputs.
            Indices are selected in ``[0, 1]``: ``0`` corresponds to a `sentence A` token, ``1``
            corresponds to a `sentence B` token
            (see `BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding`_ for more details).
        **position_ids**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            Indices of positions of each input sequence tokens in the position embeddings.
            Selected in the range ``[0, config.max_position_embeddings - 1]``.
        **head_mask**: (`optional`) ``torch.FloatTensor`` of shape ``(num_heads,)`` or ``(num_layers, num_heads)``:
            Mask to nullify selected heads of the self-attention modules.
            Mask values selected in ``[0, 1]``:
            ``1`` indicates the head is **not masked**, ``0`` indicates the head is **masked**.
"""


BERT_START_DOCSTRING = """    The BERT model was proposed in
    `BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding`_
    by Jacob Devlin, Ming-Wei Chang, Kenton Lee and Kristina Toutanova. It's a bidirectional transformer
    pre-trained using a combination of masked language modeling objective and next sentence prediction
    on a large corpus comprising the Toronto Book Corpus and Wikipedia.

    This model is a PyTorch `torch.nn.Module`_ sub-class. Use it as a regular PyTorch Module and
    refer to the PyTorch documentation for all matter related to general usage and behavior.

    .. _`BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding`:
        https://arxiv.org/abs/1810.04805

    .. _`torch.nn.Module`:
        https://pytorch.org/docs/stable/nn.html#module

    Parameters:
        config (:class:`~pytorch_transformers.BertConfig`): Model configuration class with all the parameters of the model. 
            Initializing with a config file does not load the weights associated with the model, only the configuration.
            Check out the :meth:`~pytorch_transformers.PreTrainedModel.from_pretrained` method to load the model weights.
"""


def add_start_docstrings(*docstr):

    def docstring_decorator(fn):
        fn.__doc__ = ''.join(docstr) + (fn.__doc__ if fn.__doc__ is not None else '')
        return fn
    return docstring_decorator


@add_start_docstrings('The bare Bert Model transformer outputing raw hidden-states without any specific head on top.', BERT_START_DOCSTRING, BERT_INPUTS_DOCSTRING)
class BertModel(BertPreTrainedModel):
    """
    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **last_hidden_state**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, hidden_size)``
            Sequence of hidden-states at the output of the last layer of the model.
        **pooler_output**: ``torch.FloatTensor`` of shape ``(batch_size, hidden_size)``
            Last layer hidden-state of the first token of the sequence (classification token)
            further processed by a Linear layer and a Tanh activation function. The Linear
            layer weights are trained from the next sentence prediction (classification)
            objective during Bert pretraining. This output is usually *not* a good summary
            of the semantic content of the input, you're often better with averaging or pooling
            the sequence of hidden-states for the whole input sequence.
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained('bert-base-uncased')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids)
        last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple

    """

    def __init__(self, config):
        super(BertModel, self).__init__(config)
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)
        self.init_weights()

    def _resize_token_embeddings(self, new_num_tokens):
        old_embeddings = self.embeddings.word_embeddings
        new_embeddings = self._get_resized_embeddings(old_embeddings, new_num_tokens)
        self.embeddings.word_embeddings = new_embeddings
        return self.embeddings.word_embeddings

    def _prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
            See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
            head_mask = head_mask
        else:
            head_mask = [None] * self.config.num_hidden_layers
        embedding_output = self.embeddings(input_ids, position_ids=position_ids, token_type_ids=token_type_ids)
        encoder_outputs = self.encoder(embedding_output, extended_attention_mask, head_mask=head_mask)
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)
        outputs = (sequence_output, pooled_output) + encoder_outputs[1:]
        return outputs


@add_start_docstrings("""Bert Model with two heads on top as done during the pre-training:
    a `masked language modeling` head and a `next sentence prediction (classification)` head. """, BERT_START_DOCSTRING, BERT_INPUTS_DOCSTRING)
class BertForPreTraining(BertPreTrainedModel):
    """
        **masked_lm_labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            Labels for computing the masked language modeling loss.
            Indices should be in ``[-1, 0, ..., config.vocab_size]`` (see ``input_ids`` docstring)
            Tokens with indices set to ``-1`` are ignored (masked), the loss is only computed for the tokens with labels
            in ``[0, ..., config.vocab_size]``
        **next_sentence_label**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for computing the next sequence prediction (classification) loss. Input should be a sequence pair (see ``input_ids`` docstring)
            Indices should be in ``[0, 1]``.
            ``0`` indicates sequence B is a continuation of sequence A,
            ``1`` indicates sequence B is a random sequence.

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when both ``masked_lm_labels`` and ``next_sentence_label`` are provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Total loss as the sum of the masked language modeling loss and the next sequence prediction (classification) loss.
        **prediction_scores**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, config.vocab_size)``
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        **seq_relationship_scores**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, 2)``
            Prediction scores of the next sequence prediction (classification) head (scores of True/False continuation before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForPreTraining.from_pretrained('bert-base-uncased')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids)
        prediction_scores, seq_relationship_scores = outputs[:2]

    """

    def __init__(self, config):
        super(BertForPreTraining, self).__init__(config)
        self.bert = BertModel(config)
        self.cls = BertPreTrainingHeads(config)
        self.init_weights()
        self.tie_weights()

    def tie_weights(self):
        """ Make sure we are sharing the input and output embeddings.
            Export to TorchScript can't handle parameter sharing so we are cloning them instead.
        """
        self._tie_or_clone_weights(self.cls.predictions.decoder, self.bert.embeddings.word_embeddings)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, masked_lm_labels=None, next_sentence_label=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask)
        sequence_output, pooled_output = outputs[:2]
        prediction_scores, seq_relationship_score = self.cls(sequence_output, pooled_output)
        outputs = (prediction_scores, seq_relationship_score) + outputs[2:]
        if masked_lm_labels is not None and next_sentence_label is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
            next_sentence_loss = loss_fct(seq_relationship_score.view(-1, 2), next_sentence_label.view(-1))
            total_loss = masked_lm_loss + next_sentence_loss
            outputs = (total_loss,) + outputs
        return outputs


@add_start_docstrings('Bert Model with a `language modeling` head on top. ', BERT_START_DOCSTRING, BERT_INPUTS_DOCSTRING)
class BertForMaskedLM(BertPreTrainedModel):
    """
        **masked_lm_labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            Labels for computing the masked language modeling loss.
            Indices should be in ``[-1, 0, ..., config.vocab_size]`` (see ``input_ids`` docstring)
            Tokens with indices set to ``-1`` are ignored (masked), the loss is only computed for the tokens with labels
            in ``[0, ..., config.vocab_size]``

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``masked_lm_labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Masked language modeling loss.
        **prediction_scores**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, config.vocab_size)``
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForMaskedLM.from_pretrained('bert-base-uncased')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, masked_lm_labels=input_ids)
        loss, prediction_scores = outputs[:2]

    """

    def __init__(self, config):
        super(BertForMaskedLM, self).__init__(config)
        self.bert = BertModel(config)
        self.cls = BertOnlyMLMHead(config)
        self.init_weights()
        self.tie_weights()

    def tie_weights(self):
        """ Make sure we are sharing the input and output embeddings.
            Export to TorchScript can't handle parameter sharing so we are cloning them instead.
        """
        self._tie_or_clone_weights(self.cls.predictions.decoder, self.bert.embeddings.word_embeddings)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, masked_lm_labels=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask)
        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)
        outputs = (prediction_scores,) + outputs[2:]
        if masked_lm_labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
            outputs = (masked_lm_loss,) + outputs
        return outputs


@add_start_docstrings('Bert Model with a `next sentence prediction (classification)` head on top. ', BERT_START_DOCSTRING, BERT_INPUTS_DOCSTRING)
class BertForNextSentencePrediction(BertPreTrainedModel):
    """
        **next_sentence_label**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for computing the next sequence prediction (classification) loss. Input should be a sequence pair (see ``input_ids`` docstring)
            Indices should be in ``[0, 1]``.
            ``0`` indicates sequence B is a continuation of sequence A,
            ``1`` indicates sequence B is a random sequence.

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``next_sentence_label`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Next sequence prediction (classification) loss.
        **seq_relationship_scores**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, 2)``
            Prediction scores of the next sequence prediction (classification) head (scores of True/False continuation before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForNextSentencePrediction.from_pretrained('bert-base-uncased')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids)
        seq_relationship_scores = outputs[0]

    """

    def __init__(self, config):
        super(BertForNextSentencePrediction, self).__init__(config)
        self.bert = BertModel(config)
        self.cls = BertOnlyNSPHead(config)
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, next_sentence_label=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask)
        pooled_output = outputs[1]
        seq_relationship_score = self.cls(pooled_output)
        outputs = (seq_relationship_score,) + outputs[2:]
        if next_sentence_label is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            next_sentence_loss = loss_fct(seq_relationship_score.view(-1, 2), next_sentence_label.view(-1))
            outputs = (next_sentence_loss,) + outputs
        return outputs


@add_start_docstrings("""Bert Model transformer with a sequence classification/regression head on top (a linear layer on top of
    the pooled output) e.g. for GLUE tasks. """, BERT_START_DOCSTRING, BERT_INPUTS_DOCSTRING)
class BertForSequenceClassification(BertPreTrainedModel):
    """
        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for computing the sequence classification/regression loss.
            Indices should be in ``[0, ..., config.num_labels - 1]``.
            If ``config.num_labels == 1`` a regression loss is computed (Mean-Square loss),
            If ``config.num_labels > 1`` a classification loss is computed (Cross-Entropy).

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Classification (or regression if config.num_labels==1) loss.
        **logits**: ``torch.FloatTensor`` of shape ``(batch_size, config.num_labels)``
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, logits = outputs[:2]

    """

    def __init__(self, config):
        super(BertForSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, labels=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        outputs = (logits,) + outputs[2:]
        if labels is not None:
            if self.num_labels == 1:
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs
        return outputs


@add_start_docstrings("""Bert Model with a multiple choice classification head on top (a linear layer on top of
    the pooled output and a softmax) e.g. for RocStories/SWAG tasks. """, BERT_START_DOCSTRING, BERT_INPUTS_DOCSTRING)
class BertForMultipleChoice(BertPreTrainedModel):
    """
        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for computing the multiple choice classification loss.
            Indices should be in ``[0, ..., num_choices]`` where `num_choices` is the size of the second dimension
            of the input tensors. (see `input_ids` above)

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Classification loss.
        **classification_scores**: ``torch.FloatTensor`` of shape ``(batch_size, num_choices)`` where `num_choices` is the size of the second dimension
            of the input tensors. (see `input_ids` above).
            Classification scores (before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForMultipleChoice.from_pretrained('bert-base-uncased')
        choices = ["Hello, my dog is cute", "Hello, my cat is amazing"]
        input_ids = torch.tensor([tokenizer.encode(s) for s in choices]).unsqueeze(0)  # Batch size 1, 2 choices
        labels = torch.tensor(1).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, classification_scores = outputs[:2]

    """

    def __init__(self, config):
        super(BertForMultipleChoice, self).__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, labels=None):
        num_choices = input_ids.shape[1]
        input_ids = input_ids.view(-1, input_ids.size(-1))
        attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, num_choices)
        outputs = (reshaped_logits,) + outputs[2:]
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)
            outputs = (loss,) + outputs
        return outputs


@add_start_docstrings("""Bert Model with a token classification head on top (a linear layer on top of
    the hidden-states output) e.g. for Named-Entity-Recognition (NER) tasks. """, BERT_START_DOCSTRING, BERT_INPUTS_DOCSTRING)
class BertForTokenClassification(BertPreTrainedModel):
    """
        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            Labels for computing the token classification loss.
            Indices should be in ``[0, ..., config.num_labels - 1]``.

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Classification loss.
        **scores**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, config.num_labels)``
            Classification scores (before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForTokenClassification.from_pretrained('bert-base-uncased')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1] * input_ids.size(1)).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, scores = outputs[:2]

    """

    def __init__(self, config):
        super(BertForTokenClassification, self).__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, labels=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        outputs = (logits,) + outputs[2:]
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs
        return outputs


@add_start_docstrings("""Bert Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear layers on top of
    the hidden-states output to compute `span start logits` and `span end logits`). """, BERT_START_DOCSTRING, BERT_INPUTS_DOCSTRING)
class BertForQuestionAnswering(BertPreTrainedModel):
    """
        **start_positions**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`).
            Position outside of the sequence are not taken into account for computing the loss.
        **end_positions**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`).
            Position outside of the sequence are not taken into account for computing the loss.

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Total span extraction loss is the sum of a Cross-Entropy for the start and end positions.
        **start_scores**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length,)``
            Span-start scores (before SoftMax).
        **end_scores**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length,)``
            Span-end scores (before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForQuestionAnswering.from_pretrained('bert-base-uncased')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        start_positions = torch.tensor([1])
        end_positions = torch.tensor([3])
        outputs = model(input_ids, start_positions=start_positions, end_positions=end_positions)
        loss, start_scores, end_scores = outputs[:2]

    """

    def __init__(self, config):
        super(BertForQuestionAnswering, self).__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, start_positions=None, end_positions=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask)
        sequence_output = outputs[0]
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        outputs = (start_logits, end_logits) + outputs[2:]
        if start_positions is not None and end_positions is not None:
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)
            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            outputs = (total_loss,) + outputs
        return outputs


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        var = 2.0 / (self.W.size(1) + self.W.size(0))
        self.W.data.normal_(0, var)
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        var2 = 2.0 / (self.a.size(1) + self.a.size(0))
        self.a.data.normal_(0, var2)
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):
        h = torch.mm(input, self.W)
        N = h.size()[0]
        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))
        zero_vec = -9000000000000000.0 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GAT(nn.Module):

    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout
        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim=1)


class SpecialSpmmFunction(torch.autograd.Function):
    """Special function for only sparse region backpropataion layer."""

    @staticmethod
    def forward(ctx, indices, values, shape, b):
        assert indices.requires_grad == False
        a = torch.sparse_coo_tensor(indices, values, shape)
        ctx.save_for_backward(a, b)
        ctx.N = shape[0]
        return torch.matmul(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_values = grad_b = None
        if ctx.needs_input_grad[1]:
            grad_a_dense = grad_output.matmul(b.t())
            edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :]
            grad_values = grad_a_dense.view(-1)[edge_idx]
        if ctx.needs_input_grad[3]:
            grad_b = a.t().matmul(grad_output)
        return None, grad_values, None, grad_b


class SpecialSpmm(nn.Module):

    def forward(self, indices, values, shape, b):
        return SpecialSpmmFunction.apply(indices, values, shape, b)


class SpGraphAttentionLayer(nn.Module):
    """
    Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(SpGraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        var = 2.0 / (self.W.size(1) + self.W.size(0))
        self.W.data.normal_(0, var)
        self.a = nn.Parameter(torch.zeros(size=(1, 2 * out_features)))
        var2 = 2.0 / (self.a.size(1) + self.a.size(0))
        self.a.data.normal_(0, var2)
        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.special_spmm = SpecialSpmm()

    def forward(self, input, adj):
        dv = 'cuda' if input.is_cuda else 'cpu'
        N = input.size()[0]
        edge = adj.nonzero().t()
        h = torch.mm(input, self.W)
        assert not torch.isnan(h).any()
        edge_h = torch.cat((h[edge[0, :], :], h[edge[1, :], :]), dim=1).t()
        edge_e = torch.exp(-self.leakyrelu(self.a.mm(edge_h).squeeze()))
        assert not torch.isnan(edge_e).any()
        e_rowsum = self.special_spmm(edge, edge_e, torch.Size([N, N]), torch.ones(size=(N, 1), device=dv))
        edge_e = self.dropout(edge_e)
        h_prime = self.special_spmm(edge, edge_e, torch.Size([N, N]), h)
        assert not torch.isnan(h_prime).any()
        h_prime = h_prime.div(e_rowsum)
        assert not torch.isnan(h_prime).any()
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class SpGAT(nn.Module):

    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Sparse version of GAT."""
        super(SpGAT, self).__init__()
        self.dropout = dropout
        self.attentions = [SpGraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        self.out_att = SpGraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim=1)


class gcn(nn.Module):

    def __init__(self, X_size, A_hat, cuda, args, bias=True):
        super(gcn, self).__init__()
        self.A_hat = torch.tensor(A_hat, requires_grad=False).float()
        if cuda:
            self.A_hat = self.A_hat
        self.weight = nn.parameter.Parameter(torch.zeros(size=(X_size, args.hidden_size_1)))
        var = 2.0 / (self.weight.size(1) + self.weight.size(0))
        self.weight.data.normal_(0, var)
        self.weight2 = nn.parameter.Parameter(torch.zeros(size=(args.hidden_size_1, args.hidden_size_2)))
        var2 = 2.0 / (self.weight2.size(1) + self.weight2.size(0))
        self.weight2.data.normal_(0, var2)
        if bias:
            self.bias = nn.parameter.Parameter(torch.zeros(args.hidden_size_1))
            self.bias.data.normal_(0, var)
            self.bias2 = nn.parameter.Parameter(torch.zeros(args.hidden_size_2))
            self.bias2.data.normal_(0, var2)
        else:
            self.register_parameter('bias', None)
        self.fc1 = nn.Linear(args.hidden_size_2, args.num_classes)

    def forward(self, X):
        X = torch.mm(X, self.weight)
        if self.bias is not None:
            X = X + self.bias
        X = F.relu(torch.mm(self.A_hat, X))
        X = torch.mm(X, self.weight2)
        if self.bias2 is not None:
            X = X + self.bias2
        X = F.relu(torch.mm(self.A_hat, X))
        return self.fc1(X)


class gcn_batched(nn.Module):

    def __init__(self, X_size, args, bias=True):
        super(gcn_batched, self).__init__()
        self.weight = nn.parameter.Parameter(torch.zeros(size=(X_size, args.hidden_size_1)))
        var = 2.0 / (self.weight.size(1) + self.weight.size(0))
        self.weight.data.normal_(0, var)
        self.weight2 = nn.parameter.Parameter(torch.zeros(size=(args.hidden_size_1, args.hidden_size_2)))
        var2 = 2.0 / (self.weight2.size(1) + self.weight2.size(0))
        self.weight2.data.normal_(0, var2)
        if bias:
            self.bias = nn.parameter.Parameter(torch.zeros(args.hidden_size_1))
            self.bias.data.normal_(0, var)
            self.bias2 = nn.parameter.Parameter(torch.zeros(args.hidden_size_2))
            self.bias2.data.normal_(0, var2)
        else:
            self.register_parameter('bias', None)
        self.fc1 = nn.Linear(args.hidden_size_2, args.num_classes)

    def forward(self, X, A_hat):
        X = torch.mm(X, self.weight)
        if self.bias is not None:
            X = X + self.bias
        X = F.relu(torch.mm(A_hat, X))
        X = torch.mm(X, self.weight2)
        if self.bias2 is not None:
            X = X + self.bias2
        X = F.relu(torch.mm(A_hat, X))
        return self.fc1(X)


class GIN(nn.Module):

    def __init__(self, X_size, n_nodes, A_hat, cuda, args, bias=True):
        super(GIN, self).__init__()
        self.X_size = X_size
        self.n_nodes = n_nodes
        self.dum = torch.ones(size=(1, self.n_nodes))
        self.I = torch.eye(self.n_nodes, self.n_nodes)
        self.A_hat = torch.tensor(A_hat, requires_grad=False).float()
        if cuda:
            self.A_hat = self.A_hat
            self.I = self.I
            self.dum = self.dum
        self.diag_A = torch.diag(self.A_hat.diag())
        self.off_diag_A = self.A_hat - self.diag_A
        self.e1 = nn.parameter.Parameter(torch.zeros(size=(1, 1)))
        var = 2.0 / (self.e1.size(1) + self.e1.size(0))
        self.e1.data.normal_(0, var)
        self.mlp1_fc1 = nn.Linear(self.X_size, self.X_size, bias=bias)
        self.mlp1_fc2 = nn.Linear(self.X_size, args.hidden_size_1, bias=bias)
        """
        self.e2 = nn.parameter.Parameter(torch.zeros(size=(1, 1))) # size=(self.n_nodes, 1)
        var2 = 2./(self.e2.size(1) + self.e2.size(0))
        self.e2.data.normal_(0, var2)
        self.mlp2 = nn.Linear(self.X_size, args.hidden_size_1, bias=bias)
        """

    def forward(self, X, A_hat=None):
        hv = torch.mm(self.diag_A, X)
        hu = torch.mm(self.off_diag_A, X)
        X = torch.mm(torch.diag((self.dum * self.e1).squeeze()) + self.I, hv) + hu
        X = self.mlp1_fc2(F.relu(self.mlp1_fc1(X)))
        """
        hv = torch.mm(self.diag_A, X)
        hu = torch.mm(self.off_diag_A, X)
        X = torch.mm((torch.diag((self.dum*self.e2).squeeze()) + self.I), hv) + hu
        X = F.relu(self.mlp1_fc2(F.relu(self.mlp1_fc1(X))))
        """
        return X


class GIN_batched(nn.Module):

    def __init__(self, X_size, args, bias=True):
        super(GIN_batched, self).__init__()
        self.X_size = X_size
        self.e1 = nn.parameter.Parameter(torch.zeros(size=(1, 1)))
        var = 2.0 / (self.e1.size(1) + self.e1.size(0))
        self.e1.data.normal_(0, var)
        self.mlp1_fc1 = nn.Linear(self.X_size, self.X_size, bias=bias)
        self.mlp1_fc2 = nn.Linear(self.X_size, args.hidden_size_1, bias=bias)
        """
        self.e2 = nn.parameter.Parameter(torch.zeros(size=(1, 1))) # size=(self.n_nodes, 1)
        var2 = 2./(self.e2.size(1) + self.e2.size(0))
        self.e2.data.normal_(0, var2)
        self.mlp2 = nn.Linear(self.X_size, args.hidden_size_1, bias=bias)
        """

    def forward(self, X, A_hat):
        n_nodes = X.shape[0]
        dum = torch.ones(size=(1, n_nodes))
        I = torch.eye(n_nodes, n_nodes)
        if X.is_cuda:
            I = I
            dum = dum
        diag_A = torch.diag(A_hat.diag())
        off_diag_A = A_hat - diag_A
        hv = torch.mm(diag_A, X)
        hu = torch.mm(off_diag_A, X)
        X = torch.mm(torch.diag((dum * self.e1).squeeze()) + I, hv) + hu
        X = self.mlp1_fc2(F.relu(self.mlp1_fc1(X)))
        """
        hv = torch.mm(diag_A, X)
        hu = torch.mm(off_diag_A, X)
        X = torch.mm((torch.diag((dum*self.e2).squeeze()) + I), hv) + hu
        X = self.mlp2(X)
        """
        return X


class RobertaEmbeddings(BertEmbeddings):
    """
    Same as BertEmbeddings with a tiny tweak for positional embeddings indexing.
    """

    def __init__(self, config):
        super().__init__(config)
        self.padding_idx = 1
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=self.padding_idx)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size, padding_idx=self.padding_idx)

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None):
        if position_ids is None:
            if input_ids is not None:
                position_ids = self.create_position_ids_from_input_ids(input_ids)
            else:
                position_ids = self.create_position_ids_from_inputs_embeds(inputs_embeds)
        return super().forward(input_ids, token_type_ids=token_type_ids, position_ids=position_ids, inputs_embeds=inputs_embeds)

    def create_position_ids_from_input_ids(self, x):
        """ Replace non-padding symbols with their position numbers. Position numbers begin at
        padding_idx+1. Padding symbols are ignored. This is modified from fairseq's
        `utils.make_positions`.

        :param torch.Tensor x:
        :return torch.Tensor:
        """
        mask = x.ne(self.padding_idx).long()
        incremental_indicies = torch.cumsum(mask, dim=1) * mask
        return incremental_indicies + self.padding_idx

    def create_position_ids_from_inputs_embeds(self, inputs_embeds):
        """ We are provided embeddings directly. We cannot infer which are padded so just generate
        sequential position ids.

        :param torch.Tensor inputs_embeds:
        :return torch.Tensor:
        """
        input_shape = inputs_embeds.size()[:-1]
        sequence_length = input_shape[1]
        position_ids = torch.arange(self.padding_idx + 1, sequence_length + self.padding_idx + 1, dtype=torch.long, device=inputs_embeds.device)
        return position_ids.unsqueeze(0).expand(input_shape)


ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP = {'roberta-base': 'https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-pytorch_model.bin', 'roberta-large': 'https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-pytorch_model.bin', 'roberta-large-mnli': 'https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-mnli-pytorch_model.bin', 'distilroberta-base': 'https://s3.amazonaws.com/models.huggingface.co/bert/distilroberta-base-pytorch_model.bin', 'roberta-base-openai-detector': 'https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-openai-detector-pytorch_model.bin', 'roberta-large-openai-detector': 'https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-openai-detector-pytorch_model.bin'}


ROBERTA_START_DOCSTRING = """

    This model is a PyTorch `torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`_ sub-class.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general
    usage and behavior.

    Parameters:
        config (:class:`~transformers.RobertaConfig`): Model configuration class with all the parameters of the
            model. Initializing with a config file does not load the weights associated with the model, only the configuration.
            Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model weights.
"""


ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP = {'roberta-base': 'https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-config.json', 'roberta-large': 'https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-config.json', 'roberta-large-mnli': 'https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-mnli-config.json', 'distilroberta-base': 'https://s3.amazonaws.com/models.huggingface.co/bert/distilroberta-base-config.json', 'roberta-base-openai-detector': 'https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-openai-detector-config.json', 'roberta-large-openai-detector': 'https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-openai-detector-config.json'}


class RobertaConfig(BertConfig):
    """
        This is the configuration class to store the configuration of an :class:`~transformers.RobertaModel`.
        It is used to instantiate an RoBERTa model according to the specified arguments, defining the model
        architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of
        the BERT `bert-base-uncased <https://huggingface.co/bert-base-uncased>`__ architecture.

        Configuration objects inherit from  :class:`~transformers.PretrainedConfig` and can be used
        to control the model outputs. Read the documentation from  :class:`~transformers.PretrainedConfig`
        for more information.

        The :class:`~transformers.RobertaConfig` class directly inherits :class:`~transformers.BertConfig`.
        It reuses the same defaults. Please check the parent class for more information.

        Example::

            from transformers import RobertaConfig, RobertaModel

            # Initializing a RoBERTa configuration
            configuration = RobertaConfig()

            # Initializing a model from the configuration
            model = RobertaModel(configuration)

            # Accessing the model configuration
            configuration = model.config

        Attributes:
            pretrained_config_archive_map (Dict[str, str]):
                A dictionary containing all the available pre-trained checkpoints.
    """
    pretrained_config_archive_map = ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP
    model_type = 'roberta'


@add_start_docstrings('The bare RoBERTa Model transformer outputting raw hidden-states without any specific head on top.', ROBERTA_START_DOCSTRING)
class RobertaModel(BertModel):
    """
    This class overrides :class:`~transformers.BertModel`. Please check the
    superclass for the appropriate documentation alongside usage examples.
    """
    config_class = RobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = 'roberta'

    def __init__(self, config):
        super().__init__(config)
        self.embeddings = RobertaEmbeddings(config)
        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value


ROBERTA_INPUTS_DOCSTRING = """
    Args:
        input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using :class:`transformers.RobertaTokenizer`.
            See :func:`transformers.PreTrainedTokenizer.encode` and
            :func:`transformers.PreTrainedTokenizer.encode_plus` for details.

            `What are input IDs? <../glossary.html#input-ids>`__
        attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            ``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens.

            `What are attention masks? <../glossary.html#attention-mask>`__
        token_type_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Segment token indices to indicate first and second portions of the inputs.
            Indices are selected in ``[0, 1]``: ``0`` corresponds to a `sentence A` token, ``1``
            corresponds to a `sentence B` token

            `What are token type IDs? <../glossary.html#token-type-ids>`_
        position_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Indices of positions of each input sequence tokens in the position embeddings.
            Selected in the range ``[0, config.max_position_embeddings - 1]``.

            `What are position IDs? <../glossary.html#position-ids>`_
        head_mask (:obj:`torch.FloatTensor` of shape :obj:`(num_heads,)` or :obj:`(num_layers, num_heads)`, `optional`, defaults to :obj:`None`):
            Mask to nullify selected heads of the self-attention modules.
            Mask values selected in ``[0, 1]``:
            :obj:`1` indicates the head is **not masked**, :obj:`0` indicates the head is **masked**.
        inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`, defaults to :obj:`None`):
            Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded representation.
            This is useful if you want more control over how to convert `input_ids` indices into associated vectors
            than the model's internal embedding lookup matrix.
"""


class RobertaLMHead(nn.Module):
    """Roberta Head for masked language modeling."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.decoder.bias = self.bias

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = gelu(x)
        x = self.layer_norm(x)
        x = self.decoder(x) + self.bias
        return x


def add_start_docstrings_to_callable(*docstr):

    def docstring_decorator(fn):
        class_name = ':class:`~transformers.{}`'.format(fn.__qualname__.split('.')[0])
        intro = '   The {} forward method, overrides the :func:`__call__` special method.'.format(class_name)
        note = '\n\n    .. note::\n        Although the recipe for forward pass needs to be defined within\n        this function, one should call the :class:`Module` instance afterwards\n        instead of this since the former takes care of running the\n        pre and post processing steps while the latter silently ignores them.\n        '
        fn.__doc__ = intro + note + ''.join(docstr) + (fn.__doc__ if fn.__doc__ is not None else '')
        return fn
    return docstring_decorator


@add_start_docstrings('RoBERTa Model with a `language modeling` head on top. ', ROBERTA_START_DOCSTRING)
class RobertaForMaskedLM(BertPreTrainedModel):
    config_class = RobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = 'roberta'

    def __init__(self, config):
        super().__init__(config)
        self.roberta = RobertaModel(config)
        self.lm_head = RobertaLMHead(config)
        self.init_weights()

    def get_output_embeddings(self):
        return self.lm_head.decoder

    @add_start_docstrings_to_callable(ROBERTA_INPUTS_DOCSTRING)
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None, masked_lm_labels=None):
        """
        masked_lm_labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Labels for computing the masked language modeling loss.
            Indices should be in ``[-100, 0, ..., config.vocab_size]`` (see ``input_ids`` docstring)
            Tokens with indices set to ``-100`` are ignored (masked), the loss is only computed for the tokens with labels
            in ``[0, ..., config.vocab_size]``

    Returns:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.RobertaConfig`) and inputs:
        masked_lm_loss (`optional`, returned when ``masked_lm_labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Masked language modeling loss.
        prediction_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, config.vocab_size)`)
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.

    Examples::

        from transformers import RobertaTokenizer, RobertaForMaskedLM
        import torch

        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        model = RobertaForMaskedLM.from_pretrained('roberta-base')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, masked_lm_labels=input_ids)
        loss, prediction_scores = outputs[:2]

        """
        outputs = self.roberta(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds)
        sequence_output = outputs[0]
        prediction_scores = self.lm_head(sequence_output)
        outputs = (prediction_scores,) + outputs[2:]
        if masked_lm_labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
            outputs = (masked_lm_loss,) + outputs
        return outputs


class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


@add_start_docstrings("""RoBERTa Model transformer with a sequence classification/regression head on top (a linear layer
    on top of the pooled output) e.g. for GLUE tasks. """, ROBERTA_START_DOCSTRING)
class RobertaForSequenceClassification(BertPreTrainedModel):
    config_class = RobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = 'roberta'

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.roberta = RobertaModel(config)
        self.classifier = RobertaClassificationHead(config)

    @add_start_docstrings_to_callable(ROBERTA_INPUTS_DOCSTRING)
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None, labels=None):
        """
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
            Labels for computing the sequence classification/regression loss.
            Indices should be in :obj:`[0, ..., config.num_labels - 1]`.
            If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).

    Returns:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.RobertaConfig`) and inputs:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`label` is provided):
            Classification (or regression if config.num_labels==1) loss.
        logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, config.num_labels)`):
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.

    Examples::

        from transformers import RobertaTokenizer, RobertaForSequenceClassification
        import torch

        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        model = RobertaForSequenceClassification.from_pretrained('roberta-base')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, logits = outputs[:2]

        """
        outputs = self.roberta(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds)
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)
        outputs = (logits,) + outputs[2:]
        if labels is not None:
            if self.num_labels == 1:
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs
        return outputs


@add_start_docstrings("""Roberta Model with a multiple choice classification head on top (a linear layer on top of
    the pooled output and a softmax) e.g. for RocStories/SWAG tasks. """, ROBERTA_START_DOCSTRING)
class RobertaForMultipleChoice(BertPreTrainedModel):
    config_class = RobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = 'roberta'

    def __init__(self, config):
        super().__init__(config)
        self.roberta = RobertaModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.init_weights()

    @add_start_docstrings_to_callable(ROBERTA_INPUTS_DOCSTRING)
    def forward(self, input_ids=None, token_type_ids=None, attention_mask=None, labels=None, position_ids=None, head_mask=None, inputs_embeds=None):
        """
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
            Labels for computing the multiple choice classification loss.
            Indices should be in ``[0, ..., num_choices]`` where `num_choices` is the size of the second dimension
            of the input tensors. (see `input_ids` above)

    Returns:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.RobertaConfig`) and inputs:
        loss (:obj:`torch.FloatTensor`` of shape ``(1,)`, `optional`, returned when :obj:`labels` is provided):
            Classification loss.
        classification_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, num_choices)`):
            `num_choices` is the second dimension of the input tensors. (see `input_ids` above).

            Classification scores (before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.

    Examples::

        from transformers import RobertaTokenizer, RobertaForMultipleChoice
        import torch

        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        model = RobertaForMultipleChoice.from_pretrained('roberta-base')
        choices = ["Hello, my dog is cute", "Hello, my cat is amazing"]
        input_ids = torch.tensor([tokenizer.encode(s, add_special_tokens=True) for s in choices]).unsqueeze(0)  # Batch size 1, 2 choices
        labels = torch.tensor(1).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, classification_scores = outputs[:2]

        """
        num_choices = input_ids.shape[1]
        flat_input_ids = input_ids.view(-1, input_ids.size(-1))
        flat_position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        outputs = self.roberta(flat_input_ids, position_ids=flat_position_ids, token_type_ids=flat_token_type_ids, attention_mask=flat_attention_mask, head_mask=head_mask)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, num_choices)
        outputs = (reshaped_logits,) + outputs[2:]
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)
            outputs = (loss,) + outputs
        return outputs


@add_start_docstrings("""Roberta Model with a token classification head on top (a linear layer on top of
    the hidden-states output) e.g. for Named-Entity-Recognition (NER) tasks. """, ROBERTA_START_DOCSTRING)
class RobertaForTokenClassification(BertPreTrainedModel):
    config_class = RobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = 'roberta'

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.roberta = RobertaModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.init_weights()

    @add_start_docstrings_to_callable(ROBERTA_INPUTS_DOCSTRING)
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None, labels=None):
        """
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Labels for computing the token classification loss.
            Indices should be in ``[0, ..., config.num_labels - 1]``.

    Returns:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.RobertaConfig`) and inputs:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when ``labels`` is provided) :
            Classification loss.
        scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, config.num_labels)`)
            Classification scores (before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.

    Examples::

        from transformers import RobertaTokenizer, RobertaForTokenClassification
        import torch

        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        model = RobertaForTokenClassification.from_pretrained('roberta-base')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1] * input_ids.size(1)).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, scores = outputs[:2]

        """
        outputs = self.roberta(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        outputs = (logits,) + outputs[2:]
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs
        return outputs


@add_start_docstrings("""Roberta Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear layers on top of
    the hidden-states output to compute `span start logits` and `span end logits`). """, ROBERTA_START_DOCSTRING)
class RobertaForQuestionAnswering(BertPreTrainedModel):
    config_class = RobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = 'roberta'

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.roberta = RobertaModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)
        self.init_weights()

    @add_start_docstrings_to_callable(ROBERTA_INPUTS_DOCSTRING)
    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None, start_positions=None, end_positions=None):
        """
        start_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`).
            Position outside of the sequence are not taken into account for computing the loss.
        end_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`).
            Position outside of the sequence are not taken into account for computing the loss.

    Returns:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.RobertaConfig`) and inputs:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`labels` is provided):
            Total span extraction loss is the sum of a Cross-Entropy for the start and end positions.
        start_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length,)`):
            Span-start scores (before SoftMax).
        end_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length,)`):
            Span-end scores (before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.

    Examples::

        # The checkpoint roberta-large is not fine-tuned for question answering. Please see the
        # examples/run_squad.py example to see how to fine-tune a model to a question answering task.

        from transformers import RobertaTokenizer, RobertaForQuestionAnswering
        import torch

        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        model = RobertaForQuestionAnswering.from_pretrained('roberta-base')

        question, text = "Who was Jim Henson?", "Jim Henson was a nice puppet"
        input_ids = tokenizer.encode(question, text)
        start_scores, end_scores = model(torch.tensor([input_ids]))

        all_tokens = tokenizer.convert_ids_to_tokens(input_ids)
        answer = ' '.join(all_tokens[torch.argmax(start_scores) : torch.argmax(end_scores)+1])

        """
        outputs = self.roberta(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds)
        sequence_output = outputs[0]
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        outputs = (start_logits, end_logits) + outputs[2:]
        if start_positions is not None and end_positions is not None:
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)
            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            outputs = (total_loss,) + outputs
        return outputs


class XLNetRelativeAttention(nn.Module):

    def __init__(self, config):
        super(XLNetRelativeAttention, self).__init__()
        self.output_attentions = config.output_attentions
        if config.d_model % config.n_head != 0:
            raise ValueError('The hidden size (%d) is not a multiple of the number of attention heads (%d)' % (config.d_model, config.n_head))
        self.n_head = config.n_head
        self.d_head = config.d_head
        self.d_model = config.d_model
        self.scale = 1 / config.d_head ** 0.5
        self.q = nn.Parameter(torch.Tensor(config.d_model, self.n_head, self.d_head))
        self.k = nn.Parameter(torch.Tensor(config.d_model, self.n_head, self.d_head))
        self.v = nn.Parameter(torch.Tensor(config.d_model, self.n_head, self.d_head))
        self.o = nn.Parameter(torch.Tensor(config.d_model, self.n_head, self.d_head))
        self.r = nn.Parameter(torch.Tensor(config.d_model, self.n_head, self.d_head))
        self.r_r_bias = nn.Parameter(torch.Tensor(self.n_head, self.d_head))
        self.r_s_bias = nn.Parameter(torch.Tensor(self.n_head, self.d_head))
        self.r_w_bias = nn.Parameter(torch.Tensor(self.n_head, self.d_head))
        self.seg_embed = nn.Parameter(torch.Tensor(2, self.n_head, self.d_head))
        self.layer_norm = XLNetLayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.dropout)

    def prune_heads(self, heads):
        raise NotImplementedError

    @staticmethod
    def rel_shift(x, klen=-1):
        """perform relative shift to form the relative attention score."""
        x_size = x.shape
        x = x.reshape(x_size[1], x_size[0], x_size[2], x_size[3])
        x = x[1:, ...]
        x = x.reshape(x_size[0], x_size[1] - 1, x_size[2], x_size[3])
        x = torch.index_select(x, 1, torch.arange(klen, device=x.device, dtype=torch.long))
        return x

    def rel_attn_core(self, q_head, k_head_h, v_head_h, k_head_r, seg_mat=None, attn_mask=None, head_mask=None):
        """Core relative positional attention operations."""
        ac = torch.einsum('ibnd,jbnd->ijbn', q_head + self.r_w_bias, k_head_h)
        bd = torch.einsum('ibnd,jbnd->ijbn', q_head + self.r_r_bias, k_head_r)
        bd = self.rel_shift(bd, klen=ac.shape[1])
        if seg_mat is None:
            ef = 0
        else:
            ef = torch.einsum('ibnd,snd->ibns', q_head + self.r_s_bias, self.seg_embed)
            ef = torch.einsum('ijbs,ibns->ijbn', seg_mat, ef)
        attn_score = (ac + bd + ef) * self.scale
        if attn_mask is not None:
            attn_score = attn_score - 1e+30 * attn_mask
        attn_prob = F.softmax(attn_score, dim=1)
        attn_prob = self.dropout(attn_prob)
        if head_mask is not None:
            attn_prob = attn_prob * head_mask
        attn_vec = torch.einsum('ijbn,jbnd->ibnd', attn_prob, v_head_h)
        if self.output_attentions:
            return attn_vec, attn_prob
        return attn_vec

    def post_attention(self, h, attn_vec, residual=True):
        """Post-attention processing."""
        attn_out = torch.einsum('ibnd,hnd->ibh', attn_vec, self.o)
        attn_out = self.dropout(attn_out)
        if residual:
            attn_out = attn_out + h
        output = self.layer_norm(attn_out)
        return output

    def forward(self, h, g, attn_mask_h, attn_mask_g, r, seg_mat, mems=None, target_mapping=None, head_mask=None):
        if g is not None:
            if mems is not None and mems.dim() > 1:
                cat = torch.cat([mems, h], dim=0)
            else:
                cat = h
            k_head_h = torch.einsum('ibh,hnd->ibnd', cat, self.k)
            v_head_h = torch.einsum('ibh,hnd->ibnd', cat, self.v)
            k_head_r = torch.einsum('ibh,hnd->ibnd', r, self.r)
            q_head_h = torch.einsum('ibh,hnd->ibnd', h, self.q)
            attn_vec_h = self.rel_attn_core(q_head_h, k_head_h, v_head_h, k_head_r, seg_mat=seg_mat, attn_mask=attn_mask_h, head_mask=head_mask)
            if self.output_attentions:
                attn_vec_h, attn_prob_h = attn_vec_h
            output_h = self.post_attention(h, attn_vec_h)
            q_head_g = torch.einsum('ibh,hnd->ibnd', g, self.q)
            if target_mapping is not None:
                q_head_g = torch.einsum('mbnd,mlb->lbnd', q_head_g, target_mapping)
                attn_vec_g = self.rel_attn_core(q_head_g, k_head_h, v_head_h, k_head_r, seg_mat=seg_mat, attn_mask=attn_mask_g, head_mask=head_mask)
                if self.output_attentions:
                    attn_vec_g, attn_prob_g = attn_vec_g
                attn_vec_g = torch.einsum('lbnd,mlb->mbnd', attn_vec_g, target_mapping)
            else:
                attn_vec_g = self.rel_attn_core(q_head_g, k_head_h, v_head_h, k_head_r, seg_mat=seg_mat, attn_mask=attn_mask_g, head_mask=head_mask)
                if self.output_attentions:
                    attn_vec_g, attn_prob_g = attn_vec_g
            output_g = self.post_attention(g, attn_vec_g)
            if self.output_attentions:
                attn_prob = attn_prob_h, attn_prob_g
        else:
            if mems is not None and mems.dim() > 1:
                cat = torch.cat([mems, h], dim=0)
            else:
                cat = h
            q_head_h = torch.einsum('ibh,hnd->ibnd', h, self.q)
            k_head_h = torch.einsum('ibh,hnd->ibnd', cat, self.k)
            v_head_h = torch.einsum('ibh,hnd->ibnd', cat, self.v)
            k_head_r = torch.einsum('ibh,hnd->ibnd', r, self.r)
            attn_vec = self.rel_attn_core(q_head_h, k_head_h, v_head_h, k_head_r, seg_mat=seg_mat, attn_mask=attn_mask_h, head_mask=head_mask)
            if self.output_attentions:
                attn_vec, attn_prob = attn_vec
            output_h = self.post_attention(h, attn_vec)
            output_g = None
        outputs = output_h, output_g
        if self.output_attentions:
            outputs = outputs + (attn_prob,)
        return outputs


class XLNetFeedForward(nn.Module):

    def __init__(self, config):
        super(XLNetFeedForward, self).__init__()
        self.layer_norm = XLNetLayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.layer_1 = nn.Linear(config.d_model, config.d_inner)
        self.layer_2 = nn.Linear(config.d_inner, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        if isinstance(config.ff_activation, str) or sys.version_info[0] == 2 and isinstance(config.ff_activation, unicode):
            self.activation_function = ACT2FN[config.ff_activation]
        else:
            self.activation_function = config.ff_activation

    def forward(self, inp):
        output = inp
        output = self.layer_1(output)
        output = self.activation_function(output)
        output = self.dropout(output)
        output = self.layer_2(output)
        output = self.dropout(output)
        output = self.layer_norm(output + inp)
        return output


class XLNetLayer(nn.Module):

    def __init__(self, config):
        super(XLNetLayer, self).__init__()
        self.rel_attn = XLNetRelativeAttention(config)
        self.ff = XLNetFeedForward(config)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, output_h, output_g, attn_mask_h, attn_mask_g, r, seg_mat, mems=None, target_mapping=None, head_mask=None):
        outputs = self.rel_attn(output_h, output_g, attn_mask_h, attn_mask_g, r, seg_mat, mems=mems, target_mapping=target_mapping, head_mask=head_mask)
        output_h, output_g = outputs[:2]
        if output_g is not None:
            output_g = self.ff(output_g)
        output_h = self.ff(output_h)
        outputs = (output_h, output_g) + outputs[2:]
        return outputs


XLNET_PRETRAINED_MODEL_ARCHIVE_MAP = {'xlnet-base-cased': 'https://s3.amazonaws.com/models.huggingface.co/bert/xlnet-base-cased-pytorch_model.bin', 'xlnet-large-cased': 'https://s3.amazonaws.com/models.huggingface.co/bert/xlnet-large-cased-pytorch_model.bin'}


XLNET_PRETRAINED_CONFIG_ARCHIVE_MAP = {'xlnet-base-cased': 'https://s3.amazonaws.com/models.huggingface.co/bert/xlnet-base-cased-config.json', 'xlnet-large-cased': 'https://s3.amazonaws.com/models.huggingface.co/bert/xlnet-large-cased-config.json'}


class XLNetConfig(PretrainedConfig):
    """Configuration class to store the configuration of a ``XLNetModel``.

    Args:
        vocab_size_or_config_json_file: Vocabulary size of ``inputs_ids`` in ``XLNetModel``.
        d_model: Size of the encoder layers and the pooler layer.
        n_layer: Number of hidden layers in the Transformer encoder.
        n_head: Number of attention heads for each attention layer in
            the Transformer encoder.
        d_inner: The size of the "intermediate" (i.e., feed-forward)
            layer in the Transformer encoder.
        ff_activation: The non-linear activation function (function or string) in the
            encoder and pooler. If string, "gelu", "relu" and "swish" are supported.
        untie_r: untie relative position biases
        attn_type: 'bi' for XLNet, 'uni' for Transformer-XL

        dropout: The dropout probabilitiy for all fully connected
            layers in the embeddings, encoder, and pooler.
        dropatt: The dropout ratio for the attention
            probabilities.
        initializer_range: The sttdev of the truncated_normal_initializer for
            initializing all weight matrices.
        layer_norm_eps: The epsilon used by LayerNorm.

        dropout: float, dropout rate.
        dropatt: float, dropout rate on attention probabilities.
        init: str, the initialization scheme, either "normal" or "uniform".
        init_range: float, initialize the parameters with a uniform distribution
            in [-init_range, init_range]. Only effective when init="uniform".
        init_std: float, initialize the parameters with a normal distribution
            with mean 0 and stddev init_std. Only effective when init="normal".
        mem_len: int, the number of tokens to cache.
        reuse_len: int, the number of tokens in the currect batch to be cached
            and reused in the future.
        bi_data: bool, whether to use bidirectional input pipeline.
            Usually set to True during pretraining and False during finetuning.
        clamp_len: int, clamp all relative distances larger than clamp_len.
            -1 means no clamping.
        same_length: bool, whether to use the same attention length for each token.
        finetuning_task: name of the glue task on which the model was fine-tuned if any
    """
    pretrained_config_archive_map = XLNET_PRETRAINED_CONFIG_ARCHIVE_MAP

    def __init__(self, vocab_size_or_config_json_file=32000, d_model=1024, n_layer=24, n_head=16, d_inner=4096, ff_activation='gelu', untie_r=True, attn_type='bi', initializer_range=0.02, layer_norm_eps=1e-12, dropout=0.1, mem_len=None, reuse_len=None, bi_data=False, clamp_len=-1, same_length=False, finetuning_task=None, num_labels=2, summary_type='last', summary_use_proj=True, summary_activation='tanh', summary_last_dropout=0.1, start_n_top=5, end_n_top=5, **kwargs):
        """Constructs XLNetConfig.
        """
        super(XLNetConfig, self).__init__(**kwargs)
        if isinstance(vocab_size_or_config_json_file, str) or sys.version_info[0] == 2 and isinstance(vocab_size_or_config_json_file, unicode):
            with open(vocab_size_or_config_json_file, 'r', encoding='utf-8') as reader:
                json_config = json.loads(reader.read())
            for key, value in json_config.items():
                self.__dict__[key] = value
        elif isinstance(vocab_size_or_config_json_file, int):
            self.n_token = vocab_size_or_config_json_file
            self.d_model = d_model
            self.n_layer = n_layer
            self.n_head = n_head
            assert d_model % n_head == 0
            self.d_head = d_model // n_head
            self.ff_activation = ff_activation
            self.d_inner = d_inner
            self.untie_r = untie_r
            self.attn_type = attn_type
            self.initializer_range = initializer_range
            self.layer_norm_eps = layer_norm_eps
            self.dropout = dropout
            self.mem_len = mem_len
            self.reuse_len = reuse_len
            self.bi_data = bi_data
            self.clamp_len = clamp_len
            self.same_length = same_length
            self.finetuning_task = finetuning_task
            self.num_labels = num_labels
            self.summary_type = summary_type
            self.summary_use_proj = summary_use_proj
            self.summary_activation = summary_activation
            self.summary_last_dropout = summary_last_dropout
            self.start_n_top = start_n_top
            self.end_n_top = end_n_top
        else:
            raise ValueError('First argument must be either a vocabulary size (int)or the path to a pretrained model config file (str)')

    @property
    def max_position_embeddings(self):
        return -1

    @property
    def vocab_size(self):
        return self.n_token

    @vocab_size.setter
    def vocab_size(self, value):
        self.n_token = value

    @property
    def hidden_size(self):
        return self.d_model

    @property
    def num_attention_heads(self):
        return self.n_head

    @property
    def num_hidden_layers(self):
        return self.n_layer


XLNET_INPUTS_DOCSTRING = """
    Inputs:
        **input_ids**: ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            Indices of input sequence tokens in the vocabulary.
            Indices can be obtained using :class:`pytorch_transformers.XLNetTokenizer`.
            See :func:`pytorch_transformers.PreTrainedTokenizer.encode` and
            :func:`pytorch_transformers.PreTrainedTokenizer.convert_tokens_to_ids` for details.
        **token_type_ids**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            A parallel sequence of tokens (can be used to indicate various portions of the inputs).
            The embeddings from these tokens will be summed with the respective token embeddings.
            Indices are selected in the vocabulary (unlike BERT which has a specific vocabulary for segment indices).
        **attention_mask**: (`optional`) ``torch.Tensor`` of shape ``(batch_size, sequence_length)``:
            Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            ``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens.
        **input_mask**: (`optional`) ``torch.Tensor`` of shape ``(batch_size, sequence_length)``:
            Mask to avoid performing attention on padding token indices.
            Negative of `attention_mask`, i.e. with 0 for real tokens and 1 for padding.
            Kept for compatibility with the original code base.
            You can only uses one of `input_mask` and `attention_mask`
            Mask values selected in ``[0, 1]``:
            ``1`` for tokens that are MASKED, ``0`` for tokens that are NOT MASKED.
        **mems**: (`optional`)
            list of ``torch.FloatTensor`` (one for each layer):
            that contains pre-computed hidden-states (key and values in the attention blocks) as computed by the model
            (see `mems` output below). Can be used to speed up sequential decoding and attend to longer context.
        **perm_mask**: (`optional`) ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, sequence_length)``:
            Mask to indicate the attention pattern for each input token with values selected in ``[0, 1]``:
            If ``perm_mask[k, i, j] = 0``, i attend to j in batch k;
            if ``perm_mask[k, i, j] = 1``, i does not attend to j in batch k.
            If None, each token attends to all the others (full bidirectional attention).
            Only used during pretraining (to define factorization order) or for sequential decoding (generation).
        **target_mapping**: (`optional`) ``torch.FloatTensor`` of shape ``(batch_size, num_predict, sequence_length)``:
            Mask to indicate the output tokens to use.
            If ``target_mapping[k, i, j] = 1``, the i-th predict in batch k is on the j-th token.
            Only used during pretraining for partial prediction or for sequential decoding (generation).
        **head_mask**: (`optional`) ``torch.Tensor`` of shape ``(num_heads,)`` or ``(num_layers, num_heads)``:
            Mask to nullify selected heads of the self-attention modules.
            Mask values selected in ``[0, 1]``:
            ``1`` indicates the head is **not masked**, ``0`` indicates the head is **masked**.
"""


XLNET_START_DOCSTRING = """    The XLNet model was proposed in
    `XLNet: Generalized Autoregressive Pretraining for Language Understanding`_
    by Zhilin Yang*, Zihang Dai*, Yiming Yang, Jaime Carbonell, Ruslan Salakhutdinov, Quoc V. Le.
    XLnet is an extension of the Transformer-XL model pre-trained using an autoregressive method
    to learn bidirectional contexts by maximizing the expected likelihood over all permutations
    of the input sequence factorization order.

    The specific attention pattern can be controlled at training and test time using the `perm_mask` input.

    Do to the difficulty of training a fully auto-regressive model over various factorization order,
    XLNet is pretrained using only a sub-set of the output tokens as target which are selected
    with the `target_mapping` input.

    To use XLNet for sequential decoding (i.e. not in fully bi-directional setting), use the `perm_mask` and
    `target_mapping` inputs to control the attention span and outputs (see examples in `examples/run_generation.py`)

    This model is a PyTorch `torch.nn.Module`_ sub-class. Use it as a regular PyTorch Module and
    refer to the PyTorch documentation for all matter related to general usage and behavior.

    .. _`XLNet: Generalized Autoregressive Pretraining for Language Understanding`:
        http://arxiv.org/abs/1906.08237

    .. _`torch.nn.Module`:
        https://pytorch.org/docs/stable/nn.html#module

    Parameters:
        config (:class:`~pytorch_transformers.XLNetConfig`): Model configuration class with all the parameters of the model.
"""

