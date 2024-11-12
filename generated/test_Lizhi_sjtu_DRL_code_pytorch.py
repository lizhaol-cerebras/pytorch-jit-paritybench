
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


import torch.nn.functional as F


import numpy as np


from torch.utils.tensorboard import SummaryWriter


import math


import copy


from collections import deque


from torch.utils.data.sampler import BatchSampler


from torch.utils.data.sampler import SubsetRandomSampler


from torch.distributions import Categorical


from torch.distributions import Beta


from torch.distributions import Normal


from torch.utils.data.sampler import SequentialSampler


class Policy(nn.Module):

    def __init__(self, state_dim, action_dim, hidden_width):
        super(Policy, self).__init__()
        self.l1 = nn.Linear(state_dim, hidden_width)
        self.l2 = nn.Linear(hidden_width, action_dim)

    def forward(self, s):
        s = F.relu(self.l1(s))
        a_prob = F.softmax(self.l2(s), dim=1)
        return a_prob


class Value(nn.Module):

    def __init__(self, state_dim, hidden_width):
        super(Value, self).__init__()
        self.l1 = nn.Linear(state_dim, hidden_width)
        self.l2 = nn.Linear(hidden_width, 1)

    def forward(self, s):
        s = F.relu(self.l1(s))
        v_s = self.l2(s)
        return v_s


class Actor(nn.Module):

    def __init__(self, state_dim, action_dim, hidden_width, max_action):
        super(Actor, self).__init__()
        self.max_action = max_action
        self.l1 = nn.Linear(state_dim, hidden_width)
        self.l2 = nn.Linear(hidden_width, hidden_width)
        self.mean_layer = nn.Linear(hidden_width, action_dim)
        self.log_std_layer = nn.Linear(hidden_width, action_dim)

    def forward(self, x, deterministic=False, with_logprob=True):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, -20, 2)
        std = torch.exp(log_std)
        dist = Normal(mean, std)
        if deterministic:
            a = mean
        else:
            a = dist.rsample()
        if with_logprob:
            log_pi = dist.log_prob(a).sum(dim=1, keepdim=True)
            log_pi -= (2 * (np.log(2) - a - F.softplus(-2 * a))).sum(dim=1, keepdim=True)
        else:
            log_pi = None
        a = self.max_action * torch.tanh(a)
        return a, log_pi


class Critic(nn.Module):

    def __init__(self, state_dim, action_dim, hidden_width):
        super(Critic, self).__init__()
        self.l1 = nn.Linear(state_dim + action_dim, hidden_width)
        self.l2 = nn.Linear(hidden_width, hidden_width)
        self.l3 = nn.Linear(hidden_width, 1)
        self.l4 = nn.Linear(state_dim + action_dim, hidden_width)
        self.l5 = nn.Linear(hidden_width, hidden_width)
        self.l6 = nn.Linear(hidden_width, 1)

    def forward(self, s, a):
        s_a = torch.cat([s, a], 1)
        q1 = F.relu(self.l1(s_a))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        q2 = F.relu(self.l4(s_a))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2


class NoisyLinear(nn.Module):

    def __init__(self, in_features, out_features, sigma_init=0.5):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma_init = sigma_init
        self.weight_mu = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.FloatTensor(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))
        self.register_buffer('bias_epsilon', torch.FloatTensor(out_features))
        self.reset_parameters()
        self.reset_noise()

    def forward(self, x):
        if self.training:
            self.reset_noise()
            weight = self.weight_mu + self.weight_sigma.mul(self.weight_epsilon)
            bias = self.bias_mu + self.bias_sigma.mul(self.bias_epsilon)
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma_init / math.sqrt(self.in_features))
        self.bias_sigma.data.fill_(self.sigma_init / math.sqrt(self.out_features))

    def reset_noise(self):
        epsilon_i = self.scale_noise(self.in_features)
        epsilon_j = self.scale_noise(self.out_features)
        self.weight_epsilon.copy_(torch.ger(epsilon_j, epsilon_i))
        self.bias_epsilon.copy_(epsilon_j)

    def scale_noise(self, size):
        x = torch.randn(size)
        x = x.sign().mul(x.abs().sqrt())
        return x


class Dueling_Net(nn.Module):

    def __init__(self, args):
        super(Dueling_Net, self).__init__()
        self.fc1 = nn.Linear(args.state_dim, args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, args.hidden_dim)
        if args.use_noisy:
            self.V = NoisyLinear(args.hidden_dim, 1)
            self.A = NoisyLinear(args.hidden_dim, args.action_dim)
        else:
            self.V = nn.Linear(args.hidden_dim, 1)
            self.A = nn.Linear(args.hidden_dim, args.action_dim)

    def forward(self, s):
        s = torch.relu(self.fc1(s))
        s = torch.relu(self.fc2(s))
        V = self.V(s)
        A = self.A(s)
        Q = V + (A - torch.mean(A, dim=-1, keepdim=True))
        return Q


class Net(nn.Module):

    def __init__(self, args):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(args.state_dim, args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, args.hidden_dim)
        if args.use_noisy:
            self.fc3 = NoisyLinear(args.hidden_dim, args.action_dim)
        else:
            self.fc3 = nn.Linear(args.hidden_dim, args.action_dim)

    def forward(self, s):
        s = torch.relu(self.fc1(s))
        s = torch.relu(self.fc2(s))
        Q = self.fc3(s)
        return Q


def orthogonal_init(layer, gain=np.sqrt(2)):
    for name, param in layer.named_parameters():
        if 'bias' in name:
            nn.init.constant_(param, 0)
        elif 'weight' in name:
            nn.init.orthogonal_(param, gain=gain)
    return layer


class Actor_Beta(nn.Module):

    def __init__(self, args):
        super(Actor_Beta, self).__init__()
        self.fc1 = nn.Linear(args.state_dim, args.hidden_width)
        self.fc2 = nn.Linear(args.hidden_width, args.hidden_width)
        self.alpha_layer = nn.Linear(args.hidden_width, args.action_dim)
        self.beta_layer = nn.Linear(args.hidden_width, args.action_dim)
        self.activate_func = [nn.ReLU(), nn.Tanh()][args.use_tanh]
        if args.use_orthogonal_init:
            None
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.alpha_layer, gain=0.01)
            orthogonal_init(self.beta_layer, gain=0.01)

    def forward(self, s):
        s = self.activate_func(self.fc1(s))
        s = self.activate_func(self.fc2(s))
        alpha = F.softplus(self.alpha_layer(s)) + 1.0
        beta = F.softplus(self.beta_layer(s)) + 1.0
        return alpha, beta

    def get_dist(self, s):
        alpha, beta = self.forward(s)
        dist = Beta(alpha, beta)
        return dist

    def mean(self, s):
        alpha, beta = self.forward(s)
        mean = alpha / (alpha + beta)
        return mean


class Actor_Gaussian(nn.Module):

    def __init__(self, args):
        super(Actor_Gaussian, self).__init__()
        self.max_action = args.max_action
        self.fc1 = nn.Linear(args.state_dim, args.hidden_width)
        self.fc2 = nn.Linear(args.hidden_width, args.hidden_width)
        self.mean_layer = nn.Linear(args.hidden_width, args.action_dim)
        self.log_std = nn.Parameter(torch.zeros(1, args.action_dim))
        self.activate_func = [nn.ReLU(), nn.Tanh()][args.use_tanh]
        if args.use_orthogonal_init:
            None
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.mean_layer, gain=0.01)

    def forward(self, s):
        s = self.activate_func(self.fc1(s))
        s = self.activate_func(self.fc2(s))
        mean = self.max_action * torch.tanh(self.mean_layer(s))
        return mean

    def get_dist(self, s):
        mean = self.forward(s)
        log_std = self.log_std.expand_as(mean)
        std = torch.exp(log_std)
        dist = Normal(mean, std)
        return dist


class Actor_Critic_RNN(nn.Module):

    def __init__(self, args):
        super(Actor_Critic_RNN, self).__init__()
        self.use_gru = args.use_gru
        self.activate_func = [nn.ReLU(), nn.Tanh()][args.use_tanh]
        self.actor_rnn_hidden = None
        self.actor_fc1 = nn.Linear(args.state_dim, args.hidden_dim)
        if args.use_gru:
            None
            self.actor_rnn = nn.GRU(args.hidden_dim, args.hidden_dim, batch_first=True)
        else:
            None
            self.actor_rnn = nn.LSTM(args.hidden_dim, args.hidden_dim, batch_first=True)
        self.actor_fc2 = nn.Linear(args.hidden_dim, args.action_dim)
        self.critic_rnn_hidden = None
        self.critic_fc1 = nn.Linear(args.state_dim, args.hidden_dim)
        if args.use_gru:
            self.critic_rnn = nn.GRU(args.hidden_dim, args.hidden_dim, batch_first=True)
        else:
            self.critic_rnn = nn.LSTM(args.hidden_dim, args.hidden_dim, batch_first=True)
        self.critic_fc2 = nn.Linear(args.hidden_dim, 1)
        if args.use_orthogonal_init:
            None
            orthogonal_init(self.actor_fc1)
            orthogonal_init(self.actor_rnn)
            orthogonal_init(self.actor_fc2, gain=0.01)
            orthogonal_init(self.critic_fc1)
            orthogonal_init(self.critic_rnn)
            orthogonal_init(self.critic_fc2)

    def actor(self, s):
        s = self.activate_func(self.actor_fc1(s))
        output, self.actor_rnn_hidden = self.actor_rnn(s, self.actor_rnn_hidden)
        logit = self.actor_fc2(output)
        return logit

    def critic(self, s):
        s = self.activate_func(self.critic_fc1(s))
        output, self.critic_rnn_hidden = self.critic_rnn(s, self.critic_rnn_hidden)
        value = self.critic_fc2(output)
        return value


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (Actor,
     lambda: ([], {'state_dim': 4, 'action_dim': 4, 'hidden_width': 4, 'max_action': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Critic,
     lambda: ([], {'state_dim': 4, 'action_dim': 4, 'hidden_width': 4}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4])], {})),
    (Dueling_Net,
     lambda: ([], {'args': SimpleNamespace(state_dim=4, hidden_dim=4, use_noisy=4, action_dim=4)}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Net,
     lambda: ([], {'args': SimpleNamespace(state_dim=4, hidden_dim=4, use_noisy=4, action_dim=4)}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (NoisyLinear,
     lambda: ([], {'in_features': 4, 'out_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Policy,
     lambda: ([], {'state_dim': 4, 'action_dim': 4, 'hidden_width': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Value,
     lambda: ([], {'state_dim': 4, 'hidden_width': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
]

