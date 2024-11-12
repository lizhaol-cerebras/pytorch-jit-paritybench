
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


import copy


import torch


import torch.nn as nn


import torch.nn.functional as F


from torch.distributions import MultivariateNormal


import numpy as np


class Actor(nn.Module):

    def __init__(self, state_dim, action_dim, hidden_dim, max_action, is_recurrent=False):
        super(Actor, self).__init__()
        self.recurrent = is_recurrent
        if self.recurrent:
            self.l1 = nn.LSTM(state_dim, hidden_dim, batch_first=True)
        else:
            self.l1 = nn.Linear(state_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, action_dim)
        self.max_action = max_action

    def forward(self, state, hidden):
        if self.recurrent:
            self.l1.flatten_parameters()
            a, h = self.l1(state, hidden)
        else:
            a, h = F.relu(self.l1(state)), None
        a = F.relu(self.l2(a))
        a = torch.tanh(self.l3(a))
        return self.max_action * a, h


class Critic(nn.Module):

    def __init__(self, state_dim, action_dim, hidden_dim, is_recurrent=False):
        super(Critic, self).__init__()
        self.recurrent = is_recurrent
        if self.recurrent:
            self.l1 = nn.LSTM(state_dim + action_dim, hidden_dim, batch_first=True)
            self.l4 = nn.LSTM(state_dim + action_dim, hidden_dim, batch_first=True)
        else:
            self.l1 = nn.Linear(state_dim + action_dim, hidden_dim)
            self.l4 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, 1)
        self.l5 = nn.Linear(hidden_dim, hidden_dim)
        self.l6 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action, hidden1, hidden2):
        sa = torch.cat([state, action], -1)
        if self.recurrent:
            self.l1.flatten_parameters()
            self.l4.flatten_parameters()
            q1, hidden1 = self.l1(sa, hidden1)
            q2, hidden2 = self.l4(sa, hidden2)
        else:
            q1, hidden1 = F.relu(self.l1(sa)), None
            q2, hidden2 = F.relu(self.l4(sa)), None
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

    def Q1(self, state, action, hidden1):
        sa = torch.cat([state, action], -1)
        if self.recurrent:
            self.l1.flatten_parameters()
            q1, hidden1 = self.l1(sa, hidden1)
        else:
            q1, hidden1 = F.relu(self.l1(sa)), None
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1


class ActorCritic(nn.Module):

    def __init__(self, state_dim, action_dim, hidden_dim, max_action, policy_noise, is_recurrent=True):
        super(ActorCritic, self).__init__()
        self.recurrent = is_recurrent
        self.action_dim = action_dim
        if self.recurrent:
            self.l1 = nn.LSTM(state_dim, hidden_dim, batch_first=True)
        else:
            self.l1 = nn.Linear(state_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.actor = nn.Linear(hidden_dim, action_dim)
        self.critic = nn.Linear(hidden_dim, 1)
        self.max_action = max_action
        self.policy_noise = policy_noise

    def forward(self, state, hidden):
        if self.recurrent:
            self.l1.flatten_parameters()
            p, h = self.l1(state, hidden)
        else:
            p, h = torch.tanh(self.l1(state)), None
        p = torch.tanh(self.l2(p.data))
        return p, h

    def act(self, state, hidden):
        p, h = self.forward(state, hidden)
        action = torch.tanh(self.actor(p))
        return action * self.max_action, h

    def evaluate(self, state, action, hidden):
        p, h = self.forward(state, hidden)
        action_mean, _ = self.act(state, hidden)
        cov_mat = torch.eye(self.action_dim) * self.policy_noise
        dist = MultivariateNormal(action_mean, cov_mat)
        _ = dist.sample()
        action_logprob = dist.log_prob(action)
        entropy = dist.entropy()
        values = self.critic(p)
        if self.recurrent:
            values = values[..., 0]
        else:
            action_logprob = action_logprob[..., None]
        return values, action_logprob, entropy


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (Actor,
     lambda: ([], {'state_dim': 4, 'action_dim': 4, 'hidden_dim': 4, 'max_action': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (Critic,
     lambda: ([], {'state_dim': 4, 'action_dim': 4, 'hidden_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
]

