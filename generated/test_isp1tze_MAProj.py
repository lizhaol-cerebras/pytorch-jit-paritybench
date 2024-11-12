
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


import random


import torch.nn as nn


from copy import deepcopy


from torch.optim import Adam


from torch.autograd import Variable


import torch as th


import torch.nn.functional as F


class Actor(nn.Module):

    def __init__(self, dim_observation, dim_action):
        super(Actor, self).__init__()
        self.FC1 = nn.Linear(dim_observation, 500)
        self.FC2 = nn.Linear(500, 128)
        self.FC3 = nn.Linear(128, dim_action)

    def forward(self, obs):
        result = F.relu(self.FC1(obs))
        result = F.relu(self.FC2(result))
        result = F.tanh(self.FC3(result))
        return result


class Critic(nn.Module):

    def __init__(self, n_agent, dim_observation, dim_action):
        super(Critic, self).__init__()
        self.n_agent = n_agent
        self.dim_observation = dim_observation
        self.dim_action = dim_action
        obs_dim = self.dim_observation * n_agent
        act_dim = self.dim_action * n_agent
        self.FC1 = nn.Linear(obs_dim, 1024)
        self.FC2 = nn.Linear(1024 + act_dim, 512)
        self.FC3 = nn.Linear(512, 300)
        self.FC4 = nn.Linear(300, 1)

    def forward(self, obs, acts):
        result = F.relu(self.FC1(obs))
        combined = th.cat([result, acts], dim=1)
        result = F.relu(self.FC2(combined))
        return self.FC4(F.relu(self.FC3(result)))


def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1.0 / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)


class DenseNet(nn.Module):

    def __init__(self, s_dim, hidden_dim, a_dim, norm_in=False, hidden_activation=nn.ReLU, output_activation=None):
        super(DenseNet, self).__init__()
        self._norm_in = norm_in
        if self._norm_in:
            self.norm1 = nn.BatchNorm1d(s_dim)
            self.norm2 = nn.BatchNorm1d(hidden_dim)
            self.norm3 = nn.BatchNorm1d(hidden_dim)
            self.norm4 = nn.BatchNorm1d(hidden_dim)
        self.dense1 = nn.Linear(s_dim, hidden_dim)
        self.dense1.weight.data = fanin_init(self.dense1.weight.data.size())
        self.dense2 = nn.Linear(hidden_dim, hidden_dim)
        self.dense2.weight.data = fanin_init(self.dense2.weight.data.size())
        self.dense3 = nn.Linear(hidden_dim, hidden_dim)
        self.dense3.weight.data.uniform_(-0.003, 0.003)
        self.dense4 = nn.Linear(hidden_dim, a_dim)
        if hidden_activation:
            self.hidden_activation = hidden_activation()
        else:
            self.hidden_activation = lambda x: x
        if output_activation:
            self.output_activation = output_activation()
        else:
            self.output_activation = lambda x: x

    def forward(self, x):
        use_norm = True if self._norm_in and x.shape[0] != 1 else False
        if use_norm:
            x = self.norm1(x)
        x = self.hidden_activation(self.dense1(x))
        if use_norm:
            x = self.norm2(x)
        x = self.hidden_activation(self.dense2(x))
        if use_norm:
            x = self.norm3(x)
        x = self.hidden_activation(self.dense3(x))
        if use_norm:
            x = self.norm4(x)
        x = self.output_activation(self.dense4(x))
        return x


class LSTMNet(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, batch_fisrt=True, bidirectional=True):
        super(LSTMNet, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, bias=bias, batch_first=batch_fisrt, bidirectional=bidirectional)

    def forward(self, input, wh=None, wc=None):
        output, (hidden, cell) = self.lstm(input)
        return output


class CommNetWork_Actor(nn.Module):

    def __init__(self, input_shape, batch_size, action_dim):
        super(CommNetWork_Actor, self).__init__()
        self.rnn_hidden_dim = 256
        self.n_agents = 3
        self.n_actions = action_dim
        self.cuda = True
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.input_size = input_shape * self.n_agents
        self.encoding = nn.Linear(self.input_shape, self.rnn_hidden_dim)
        self.f_obs = nn.Linear(self.rnn_hidden_dim, self.rnn_hidden_dim)
        self.f_comm = nn.GRUCell(self.rnn_hidden_dim, self.rnn_hidden_dim)
        self.decoding0 = nn.Linear(self.rnn_hidden_dim, self.rnn_hidden_dim)
        self.decoding = nn.Linear(self.rnn_hidden_dim, self.n_actions)

    def forward(self, obs):
        size = obs.view(-1, self.n_agents, self.input_shape).shape
        size0 = size[0]
        obs_encoding = torch.relu(self.encoding(obs.view(size0 * self.n_agents, self.input_shape)))
        h_out = self.f_obs(obs_encoding)
        for k in range(2):
            if k == 0:
                h = h_out
                c = torch.zeros_like(h)
            else:
                h = h.reshape(-1, self.n_agents, self.rnn_hidden_dim)
                c = h.reshape(-1, 1, self.n_agents * self.rnn_hidden_dim)
                c = c.repeat(1, self.n_agents, 1)
                mask = 1 - torch.eye(self.n_agents)
                mask = mask.view(-1, 1).repeat(1, self.rnn_hidden_dim).view(self.n_agents, -1)
                if self.cuda:
                    mask = mask
                c = c * mask.unsqueeze(0)
                c = c.reshape(-1, self.n_agents, self.n_agents, self.rnn_hidden_dim)
                c = c.mean(dim=-2)
                h = h.reshape(-1, self.rnn_hidden_dim)
                c = c.reshape(-1, self.rnn_hidden_dim)
            h = self.f_comm(c, h)
        weights = torch.relu(self.decoding0(h))
        weights = torch.tanh(self.decoding(weights))
        return weights


class CommNetWork_Critic(nn.Module):

    def __init__(self, input_shape, batch_size, action_dim):
        super(CommNetWork_Critic, self).__init__()
        self.rnn_hidden_dim = 256
        self.n_agents = 3
        self.n_actions = action_dim
        self.cuda = True
        self.batch_size = batch_size
        self.input_shape = input_shape + self.n_actions
        self.input_size = input_shape * self.n_agents
        self.encoding = nn.Linear(self.input_shape, self.rnn_hidden_dim)
        self.f_obs = nn.Linear(self.rnn_hidden_dim, self.rnn_hidden_dim)
        self.f_comm = nn.GRUCell(self.rnn_hidden_dim, self.rnn_hidden_dim)
        self.decoding = nn.Linear(self.rnn_hidden_dim, 1)

    def forward(self, obs, act):
        obs = obs.view(-1, self.n_agents, self.input_shape - self.n_actions)
        size0 = obs.shape[0]
        act = act.view(-1, self.n_agents, self.n_actions)
        x = torch.cat((obs, act), dim=-1)
        obs_encoding = torch.relu(self.encoding(x.view(size0 * self.n_agents, self.input_shape)))
        h_out = self.f_obs(obs_encoding)
        for k in range(2):
            if k == 0:
                h = h_out
                c = torch.zeros_like(h)
            else:
                h = h.reshape(-1, self.n_agents, self.rnn_hidden_dim)
                c = h.reshape(-1, 1, self.n_agents * self.rnn_hidden_dim)
                c = c.repeat(1, self.n_agents, 1)
                mask = 1 - torch.eye(self.n_agents)
                mask = mask.view(-1, 1).repeat(1, self.rnn_hidden_dim).view(self.n_agents, -1)
                if self.cuda:
                    mask = mask
                c = c * mask.unsqueeze(0)
                c = c.reshape(-1, self.n_agents, self.n_agents, self.rnn_hidden_dim)
                c = c.mean(dim=-2)
                h = h.reshape(-1, self.rnn_hidden_dim)
                c = c.reshape(-1, self.rnn_hidden_dim)
            h = self.f_comm(c, h)
        weights = self.decoding(h).view(size0, self.n_agents, -1)
        return weights


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (Actor,
     lambda: ([], {'dim_observation': 4, 'dim_action': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (CommNetWork_Actor,
     lambda: ([], {'input_shape': 4, 'batch_size': 4, 'action_dim': 4}),
     lambda: ([torch.rand([4, 3, 4])], {})),
    (CommNetWork_Critic,
     lambda: ([], {'input_shape': 4, 'batch_size': 4, 'action_dim': 4}),
     lambda: ([torch.rand([4, 3, 4]), torch.rand([4, 3, 4])], {})),
    (DenseNet,
     lambda: ([], {'s_dim': 4, 'hidden_dim': 4, 'a_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (LSTMNet,
     lambda: ([], {'input_size': 4, 'hidden_size': 4}),
     lambda: ([torch.rand([4, 4])], {})),
]

