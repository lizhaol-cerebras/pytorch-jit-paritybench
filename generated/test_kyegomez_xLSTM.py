
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


class sLSTMCell(nn.Module):

    def __init__(self, input_size, hidden_size):
        super(sLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.w_i = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.w_f = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.w_o = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.w_z = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.r_i = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.r_f = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.r_o = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.r_z = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_i = nn.Parameter(torch.Tensor(hidden_size))
        self.b_f = nn.Parameter(torch.Tensor(hidden_size))
        self.b_o = nn.Parameter(torch.Tensor(hidden_size))
        self.b_z = nn.Parameter(torch.Tensor(hidden_size))
        self.sigmoid = nn.Sigmoid()
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.w_i)
        nn.init.xavier_uniform_(self.w_f)
        nn.init.xavier_uniform_(self.w_o)
        nn.init.xavier_uniform_(self.w_z)
        nn.init.orthogonal_(self.r_i)
        nn.init.orthogonal_(self.r_f)
        nn.init.orthogonal_(self.r_o)
        nn.init.orthogonal_(self.r_z)
        nn.init.zeros_(self.b_i)
        nn.init.zeros_(self.b_f)
        nn.init.zeros_(self.b_o)
        nn.init.zeros_(self.b_z)

    def forward(self, x, states):
        h_prev, c_prev, n_prev, m_prev = states
        i_tilda = torch.matmul(self.w_i, x) + torch.matmul(self.r_i, h_prev) + self.b_i
        f_tilda = torch.matmul(self.w_f, x) + torch.matmul(self.r_f, h_prev) + self.b_f
        o_tilda = torch.matmul(self.w_o, x) + torch.matmul(self.r_o, h_prev) + self.b_o
        z_tilda = torch.matmul(self.w_z, x) + torch.matmul(self.r_z, h_prev) + self.b_z
        i_t = torch.exp(i_tilda)
        f_t = self.sigmoid(f_tilda)
        m_t = torch.max(torch.log(f_t) + m_prev, torch.log(i_t))
        i_prime = torch.exp(torch.log(i_t) - m_t)
        f_prime = torch.exp(torch.log(f_t) + m_prev - m_t)
        c_t = f_prime * c_prev + i_prime * torch.tanh(z_tilda)
        n_t = f_prime * n_prev + i_prime
        c_hat = c_t / n_t
        h_t = self.sigmoid(o_tilda) * torch.tanh(c_hat)
        return h_t, (h_t, c_t, n_t, m_t)


class sLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers):
        super(sLSTM, self).__init__()
        self.layers = nn.ModuleList([sLSTMCell(input_size if i == 0 else hidden_size, hidden_size) for i in range(num_layers)])

    def forward(self, x, initial_states=None):
        batch_size, seq_len, _ = x.size()
        if initial_states is None:
            initial_states = [(torch.zeros(batch_size, self.layers[0].hidden_size), torch.zeros(batch_size, self.layers[0].hidden_size), torch.zeros(batch_size, self.layers[0].hidden_size), torch.zeros(batch_size, self.layers[0].hidden_size)) for _ in self.layers]
        outputs = []
        current_states = initial_states
        for t in range(seq_len):
            x_t = x[:, t, :]
            new_states = []
            for layer, state in zip(self.layers, current_states):
                h_t, new_state = layer(x_t, state)
                new_states.append(new_state)
                x_t = h_t
            outputs.append(h_t.unsqueeze(1))
            current_states = new_states
        outputs = torch.cat(outputs, dim=1)
        return outputs, current_states


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (sLSTM,
     lambda: ([], {'input_size': 4, 'hidden_size': 4, 'num_layers': 1}),
     lambda: ([torch.rand([4, 4, 4])], {})),
    (sLSTMCell,
     lambda: ([], {'input_size': 4, 'hidden_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
]

