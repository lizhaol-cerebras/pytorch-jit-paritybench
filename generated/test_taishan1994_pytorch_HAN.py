
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


import torch.nn as nn


import numpy as np


import torch.optim as optim


class Attn_head(nn.Module):

    def __init__(self, in_channel, out_sz, bias_mat, in_drop=0.0, coef_drop=0.0, activation=None, residual=False, return_coef=False):
        super(Attn_head, self).__init__()
        self.bias_mat = bias_mat
        self.in_drop = in_drop
        self.coef_drop = coef_drop
        self.return_coef = return_coef
        self.conv1 = nn.Conv1d(in_channel, out_sz, 1, bias=False)
        self.conv2_1 = nn.Conv1d(out_sz, 1, 1, bias=False)
        self.conv2_2 = nn.Conv1d(out_sz, 1, 1, bias=False)
        self.leakyrelu = nn.LeakyReLU()
        self.softmax = nn.Softmax(dim=1)
        self.in_dropout = nn.Dropout(in_drop)
        self.coef_dropout = nn.Dropout(coef_drop)
        self.activation = activation

    def forward(self, x):
        seq = x.float()
        if self.in_drop != 0.0:
            seq = self.in_dropout(x)
            seq = seq.float()
        seq_fts = self.conv1(seq)
        f_1 = self.conv2_1(seq_fts)
        f_2 = self.conv2_1(seq_fts)
        logits = f_1 + torch.transpose(f_2, 2, 1)
        logits = self.leakyrelu(logits)
        coefs = self.softmax(logits + self.bias_mat.float())
        if self.coef_drop != 0.0:
            coefs = self.coef_dropout(coefs)
        if self.in_drop != 0.0:
            seq_fts = self.in_dropout(seq_fts)
        ret = torch.matmul(coefs, torch.transpose(seq_fts, 2, 1))
        ret = torch.transpose(ret, 2, 1)
        if self.return_coef:
            return self.activation(ret), coefs
        else:
            return self.activation(ret)


class SimpleAttLayer(nn.Module):

    def __init__(self, inputs, attention_size, time_major=False, return_alphas=False):
        super(SimpleAttLayer, self).__init__()
        self.hidden_size = inputs
        self.return_alphas = return_alphas
        self.time_major = time_major
        self.w_omega = nn.Parameter(torch.Tensor(self.hidden_size, attention_size))
        self.b_omega = nn.Parameter(torch.Tensor(attention_size))
        self.u_omega = nn.Parameter(torch.Tensor(attention_size, 1))
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.w_omega)
        nn.init.zeros_(self.b_omega)
        nn.init.xavier_uniform_(self.u_omega)

    def forward(self, x):
        v = self.tanh(torch.matmul(x, self.w_omega) + self.b_omega)
        vu = torch.matmul(v, self.u_omega)
        alphas = self.softmax(vu)
        output = torch.sum(x * alphas.reshape(alphas.shape[0], -1, 1), dim=1)
        if not self.return_alphas:
            return ouput
        else:
            return output, alphas


class HeteGAT_multi(nn.Module):

    def __init__(self, inputs_list, nb_classes, nb_nodes, attn_drop, ffd_drop, bias_mat_list, hid_units, n_heads, activation=nn.ELU(), residual=False):
        super(HeteGAT_multi, self).__init__()
        self.inputs_list = inputs_list
        self.nb_classes = nb_classes
        self.nb_nodes = nb_nodes
        self.attn_drop = attn_drop
        self.ffd_drop = ffd_drop
        self.bias_mat_list = bias_mat_list
        self.hid_units = hid_units
        self.n_heads = n_heads
        self.activation = activation
        self.residual = residual
        self.mp_att_size = 128
        self.layers = self._make_attn_head()
        self.simpleAttLayer = SimpleAttLayer(64, self.mp_att_size, time_major=False, return_alphas=True)
        self.fc = nn.Linear(64, self.nb_classes)

    def _make_attn_head(self):
        layers = []
        for inputs, biases in zip(self.inputs_list, self.bias_mat_list):
            layers.append(Attn_head(in_channel=inputs.shape[1], out_sz=self.hid_units[0], bias_mat=biases, in_drop=self.ffd_drop, coef_drop=self.attn_drop, activation=self.activation, residual=self.residual))
        return nn.Sequential(*list(m for m in layers))

    def forward(self, x):
        embed_list = []
        for i, (inputs, biases) in enumerate(zip(x, self.bias_mat_list)):
            attns = []
            jhy_embeds = []
            for _ in range(self.n_heads[0]):
                attns.append(self.layers[i](inputs))
            h_1 = torch.cat(attns, dim=1)
            embed_list.append(torch.squeeze(h_1).reshape(h_1.shape[-1], 1, -1))
        multi_embed = torch.cat(embed_list, dim=1)
        final_embed, att_val = self.simpleAttLayer(multi_embed)
        out = []
        for i in range(self.n_heads[-1]):
            out.append(self.fc(final_embed))
        return out[0]

