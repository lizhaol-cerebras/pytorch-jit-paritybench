
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


from torch.utils.tensorboard import SummaryWriter


import numpy as np


import torch.nn.functional as F


import torch.nn as nn


import torch


import copy


from collections import deque


from torch import nn


import math


from torch.distributions import Normal


import time


from matplotlib import pyplot as plt


from torch.distributions import Categorical


from torch.distributions import Beta


from torch.distributions.categorical import Categorical


from copy import deepcopy


import torch.multiprocessing as mp


class Q_Net(nn.Module):

    def __init__(self, action_dim, hidden):
        super(Q_Net, self).__init__()
        self.net = nn.Sequential(nn.Conv2d(4, 32, 8, stride=4), nn.ReLU(), nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(), nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(), nn.Flatten(), nn.Linear(64 * 7 * 7, hidden), nn.ReLU(), nn.Linear(hidden, action_dim))

    def forward(self, obs):
        s = obs.float() / 255
        q = self.net(s)
        return q

    def orthogonal_init(self, layer, gain=1.4142):
        for name, param in layer.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name:
                nn.init.orthogonal_(param, gain=gain)
        return layer


class NoisyLinear(nn.Module):
    """From https://github.com/Lizhi-sjtu/DRL-code-pytorch/blob/main/3.Rainbow_DQN/network.py"""

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


class Duel_Q_Net(nn.Module):

    def __init__(self, opt):
        super(Duel_Q_Net, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(4, 32, 8, stride=4), nn.ReLU(), nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(), nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(), nn.Flatten())
        if opt.Noisy:
            self.fc = NoisyLinear(64 * 7 * 7, opt.fc_width)
            self.A = NoisyLinear(opt.fc_width, opt.action_dim)
            self.V = NoisyLinear(opt.fc_width, 1)
        else:
            self.fc = nn.Linear(64 * 7 * 7, opt.fc_width)
            self.A = nn.Linear(opt.fc_width, opt.action_dim)
            self.V = nn.Linear(opt.fc_width, 1)

    def forward(self, obs):
        s = obs.float() / 255
        s = self.conv(s)
        s = torch.relu(self.fc(s))
        Adv = self.A(s)
        V = self.V(s)
        Q = V + (Adv - torch.mean(Adv, dim=-1, keepdim=True))
        return Q


def build_net(layer_shape, hidden_activation, output_activation):
    """Build net with for loop"""
    layers = []
    for j in range(len(layer_shape) - 1):
        act = hidden_activation if j < len(layer_shape) - 2 else output_activation
        layers += [nn.Linear(layer_shape[j], layer_shape[j + 1]), act()]
    return nn.Sequential(*layers)


class Categorical_Q_Net(nn.Module):

    def __init__(self, state_dim, action_dim, hid_shape, atoms):
        super(Categorical_Q_Net, self).__init__()
        self.atoms = atoms
        self.n_atoms = len(atoms)
        self.action_dim = action_dim
        layers = [state_dim] + list(hid_shape) + [action_dim * self.n_atoms]
        self.net = build_net(layers, nn.ReLU, nn.Identity)

    def _predict(self, state):
        logits = self.net(state)
        distributions = torch.softmax(logits.view(len(state), self.action_dim, self.n_atoms), dim=2)
        q_values = (distributions * self.atoms).sum(2)
        return distributions, q_values

    def forward(self, state, action=None):
        distributions, q_values = self._predict(state)
        if action is None:
            action = torch.argmax(q_values, dim=1)
        return action, distributions[torch.arange(len(state)), action]


class Noisy_Q_Net(nn.Module):

    def __init__(self, state_dim, action_dim, hid_shape):
        super(Noisy_Q_Net, self).__init__()
        layers = [state_dim] + list(hid_shape) + [action_dim]
        self.Q = build_net(layers, nn.ReLU, nn.Identity)

    def forward(self, s):
        q = self.Q(s)
        return q


class LinearSchedule(object):

    def __init__(self, schedule_timesteps, initial_p, final_p):
        """Linear interpolation between initial_p and final_p over
		schedule_timesteps. After this many timesteps pass final_p is
		returned.
		Parameters
		----------
		schedule_timesteps: int
			Number of timesteps for which to linearly anneal initial_p
			to final_p
		initial_p: float
			initial output value
		final_p: float
			final output value
		"""
        self.schedule_timesteps = schedule_timesteps
        self.initial_p = initial_p
        self.final_p = final_p

    def value(self, t):
        fraction = min(float(t) / self.schedule_timesteps, 1.0)
        return self.initial_p + fraction * (self.final_p - self.initial_p)


class Actor:

    def __init__(self, opt, shared_data):
        self.shared_data = shared_data
        self.device = torch.device(opt.A_dvc)
        self.max_train_steps = opt.max_train_steps
        self.train_envs = opt.train_envs
        self.action_dim = opt.action_dim
        self.explore_steps = opt.explore_steps
        self.time_feedback = opt.time_feedback
        self.explore_frac_scheduler = LinearSchedule(opt.decay_step, opt.init_explore_frac, opt.end_explore_frac)
        self.p = torch.zeros(opt.train_envs)
        self.min_eps = opt.min_eps
        self.envs = envpool.make_gym(opt.ExpEnvName, num_envs=opt.train_envs, seed=opt.seed, max_episode_steps=int(50000.0 / 4), episodic_life=True, reward_clip=True)
        self.actor_net = Q_Net(opt.action_dim, opt.fc_width)
        self.step_counter = 0

    def run(self):
        ct = np.ones(self.train_envs, dtype=np.bool_)
        s, info = self.envs.reset()
        mean_t, c = 0, 0
        while True:
            if self.step_counter > self.max_train_steps:
                break
            random_phase = self.step_counter < self.explore_steps
            if random_phase:
                a = np.random.randint(0, self.action_dim, self.train_envs)
            else:
                t0 = time.time()
                a = self.select_action(s)
            s_next, r, dw, tr, info = self.envs.step(a)
            self.shared_data.add(s, a, r, dw, ct)
            ct = ~(dw + tr)
            s = s_next
            self.step_counter += self.train_envs
            self.shared_data.set_total_steps(self.step_counter)
            if not random_phase:
                if self.step_counter % (5 * self.train_envs) == 0:
                    if self.shared_data.get_should_download():
                        self.shared_data.set_should_download(False)
                        self.download_model()
                if self.step_counter % (10 * self.train_envs) == 0:
                    self.fresh_explore_prob(self.step_counter - self.explore_steps)
                if self.step_counter % (100 * self.train_envs) == 0:
                    None
                if self.time_feedback:
                    c += 1
                    current_t = time.time() - t0
                    mean_t = mean_t + (current_t - mean_t) / c
                    self.shared_data.set_t(mean_t, 0)
                    t = self.shared_data.get_t()
                    if t[0] < t[1]:
                        hold_time = t[1] - t[0]
                        if hold_time > 1:
                            hold_time = 1
                        time.sleep(hold_time)

    def fresh_explore_prob(self, steps):
        explore_frac = self.explore_frac_scheduler.value(steps)
        i = int(explore_frac * self.train_envs)
        explore = torch.arange(i) / (1.25 * i)
        self.p *= 0
        self.p[self.train_envs - i:] = explore
        self.p += self.min_eps

    def select_action(self, s):
        """For envpool, the input is [n,4,84,84], npdarray"""
        with torch.no_grad():
            s = torch.from_numpy(s)
            a = self.actor_net(s).argmax(dim=-1).cpu()
            replace = torch.rand(self.train_envs) < self.p
            rd_a = torch.randint(0, self.action_dim, (self.train_envs,))
            a[replace] = rd_a[replace]
            return a.numpy()

    def download_model(self):
        self.actor_net.load_state_dict(self.shared_data.get_net_param())
        for actor_param in self.actor_net.parameters():
            actor_param.requires_grad = False


class Critic(nn.Module):

    def __init__(self, state_dim, net_width):
        super(Critic, self).__init__()
        self.C1 = nn.Linear(state_dim, net_width)
        self.C2 = nn.Linear(net_width, net_width)
        self.C3 = nn.Linear(net_width, 1)

    def forward(self, state):
        v = torch.tanh(self.C1(state))
        v = torch.tanh(self.C2(v))
        v = self.C3(v)
        return v


class BetaActor(nn.Module):

    def __init__(self, state_dim, action_dim, net_width):
        super(BetaActor, self).__init__()
        self.l1 = nn.Linear(state_dim, net_width)
        self.l2 = nn.Linear(net_width, net_width)
        self.alpha_head = nn.Linear(net_width, action_dim)
        self.beta_head = nn.Linear(net_width, action_dim)

    def forward(self, state):
        a = torch.tanh(self.l1(state))
        a = torch.tanh(self.l2(a))
        alpha = F.softplus(self.alpha_head(a)) + 1.0
        beta = F.softplus(self.beta_head(a)) + 1.0
        return alpha, beta

    def get_dist(self, state):
        alpha, beta = self.forward(state)
        dist = Beta(alpha, beta)
        return dist

    def deterministic_act(self, state):
        alpha, beta = self.forward(state)
        mode = alpha / (alpha + beta)
        return mode


class GaussianActor_musigma(nn.Module):

    def __init__(self, state_dim, action_dim, net_width):
        super(GaussianActor_musigma, self).__init__()
        self.l1 = nn.Linear(state_dim, net_width)
        self.l2 = nn.Linear(net_width, net_width)
        self.mu_head = nn.Linear(net_width, action_dim)
        self.sigma_head = nn.Linear(net_width, action_dim)

    def forward(self, state):
        a = torch.tanh(self.l1(state))
        a = torch.tanh(self.l2(a))
        mu = torch.sigmoid(self.mu_head(a))
        sigma = F.softplus(self.sigma_head(a))
        return mu, sigma

    def get_dist(self, state):
        mu, sigma = self.forward(state)
        dist = Normal(mu, sigma)
        return dist

    def deterministic_act(self, state):
        mu, sigma = self.forward(state)
        return mu


class GaussianActor_mu(nn.Module):

    def __init__(self, state_dim, action_dim, net_width, log_std=0):
        super(GaussianActor_mu, self).__init__()
        self.l1 = nn.Linear(state_dim, net_width)
        self.l2 = nn.Linear(net_width, net_width)
        self.mu_head = nn.Linear(net_width, action_dim)
        self.mu_head.weight.data.mul_(0.1)
        self.mu_head.bias.data.mul_(0.0)
        self.action_log_std = nn.Parameter(torch.ones(1, action_dim) * log_std)

    def forward(self, state):
        a = torch.relu(self.l1(state))
        a = torch.relu(self.l2(a))
        mu = torch.sigmoid(self.mu_head(a))
        return mu

    def get_dist(self, state):
        mu = self.forward(state)
        action_log_std = self.action_log_std.expand_as(mu)
        action_std = torch.exp(action_log_std)
        dist = Normal(mu, action_std)
        return dist

    def deterministic_act(self, state):
        return self.forward(state)


class Q_Critic(nn.Module):

    def __init__(self, state_dim, action_dim, net_width):
        super(Q_Critic, self).__init__()
        self.l1 = nn.Linear(state_dim + action_dim, net_width)
        self.l2 = nn.Linear(net_width, 300)
        self.l3 = nn.Linear(300, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        q = F.relu(self.l1(sa))
        q = F.relu(self.l2(q))
        q = self.l3(q)
        return q


class Double_Q_Critic(nn.Module):

    def __init__(self, state_dim, action_dim, hid_shape):
        super(Double_Q_Critic, self).__init__()
        layers = [state_dim + action_dim] + list(hid_shape) + [1]
        self.Q_1 = build_net(layers, nn.ReLU, nn.Identity)
        self.Q_2 = build_net(layers, nn.ReLU, nn.Identity)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        q1 = self.Q_1(sa)
        q2 = self.Q_2(sa)
        return q1, q2


class Double_Q_Net(nn.Module):

    def __init__(self, state_dim, action_dim, hid_shape):
        super(Double_Q_Net, self).__init__()
        layers = [state_dim] + list(hid_shape) + [action_dim]
        self.Q1 = build_net(layers, nn.ReLU, nn.Identity)
        self.Q2 = build_net(layers, nn.ReLU, nn.Identity)

    def forward(self, s):
        q1 = self.Q1(s)
        q2 = self.Q2(s)
        return q1, q2


class Policy_Net(nn.Module):

    def __init__(self, state_dim, action_dim, hid_shape):
        super(Policy_Net, self).__init__()
        layers = [state_dim] + list(hid_shape) + [action_dim]
        self.P = build_net(layers, nn.ReLU, nn.Identity)

    def forward(self, s):
        logits = self.P(s)
        probs = F.softmax(logits, dim=1)
        return probs


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (BetaActor,
     lambda: ([], {'state_dim': 4, 'action_dim': 4, 'net_width': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Critic,
     lambda: ([], {'state_dim': 4, 'net_width': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (GaussianActor_mu,
     lambda: ([], {'state_dim': 4, 'action_dim': 4, 'net_width': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (GaussianActor_musigma,
     lambda: ([], {'state_dim': 4, 'action_dim': 4, 'net_width': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (NoisyLinear,
     lambda: ([], {'in_features': 4, 'out_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Q_Critic,
     lambda: ([], {'state_dim': 4, 'action_dim': 4, 'net_width': 4}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4])], {})),
]

