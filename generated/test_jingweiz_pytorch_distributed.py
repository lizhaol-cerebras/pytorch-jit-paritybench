
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


import numpy as np


from collections import deque


import random


import torch


import torch.multiprocessing as mp


import torch.nn as nn


import torch.nn.functional as F


import time


from collections import namedtuple


class Model(nn.Module):

    def __init__(self, args, norm_val, input_dims, output_dims, action_dims):
        super(Model, self).__init__()
        self.norm_val = norm_val
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.action_dims = action_dims

    def _init_weights(self):
        raise NotImplementedError('not implemented in base calss')

    def _reset(self):
        self._init_weights()

    def forward(self, input):
        raise NotImplementedError('not implemented in base calss')

    def get_action(self, input):
        raise NotImplementedError('not implemented in base calss')


class DdpgMlpModel(Model):

    def __init__(self, args, norm_val, input_dims, output_dims, action_dims):
        super(DdpgMlpModel, self).__init__(args, norm_val, input_dims, output_dims, action_dims)
        self.actor = nn.Sequential(nn.Linear(self.input_dims[0] * self.input_dims[1] * self.input_dims[2], 300), nn.Tanh(), nn.Linear(300, 200), nn.Tanh(), nn.Linear(200, self.output_dims), nn.Tanh())
        self.critic = nn.ModuleList()
        self.critic.append(nn.Sequential(nn.Linear(self.input_dims[0] * self.input_dims[1] * self.input_dims[2], 400), nn.Tanh()))
        self.critic.append(nn.Sequential(nn.Linear(400 + self.output_dims, 300), nn.Tanh(), nn.Linear(300, 1)))
        self._reset()

    def _init_weights(self):
        bound = 0.003
        nn.init.xavier_uniform_(self.actor[0].weight.data)
        nn.init.constant_(self.actor[0].bias.data, 0)
        nn.init.xavier_uniform_(self.actor[2].weight.data)
        nn.init.constant_(self.actor[2].bias.data, 0)
        nn.init.uniform_(self.actor[4].weight.data, -bound, bound)
        nn.init.constant_(self.actor[4].bias.data, 0)
        nn.init.xavier_uniform_(self.critic[0][0].weight.data)
        nn.init.constant_(self.critic[0][0].bias.data, 0)
        nn.init.xavier_uniform_(self.critic[1][0].weight.data)
        nn.init.constant_(self.critic[1][0].bias.data, 0)
        nn.init.uniform_(self.critic[1][2].weight.data, -bound, bound)
        nn.init.constant_(self.critic[1][2].bias.data, 0)

    def forward_actor(self, input):
        input = input.view(input.size(0), -1)
        action = self.actor(input)
        return action

    def forward_critic(self, input, action):
        input = input.view(input.size(0), -1)
        qx = self.critic[0](input)
        qvalue = self.critic[1](torch.cat((qx, action), 1))
        return qvalue

    def forward(self, input):
        action = self.forward_actor(input)
        qvalue = self.forward_critic(input, action)
        return action, qvalue

    def get_action(self, input, noise=0.0, device=torch.device('cpu')):
        input = torch.FloatTensor(input).unsqueeze(0)
        action = self.forward_actor(input)
        action = np.array([[action.item()]])
        return action + noise, 0.0, 0.0


class DqnCnnModel(Model):

    def __init__(self, args, norm_val, input_dims, output_dims, action_dims):
        super(DqnCnnModel, self).__init__(args, norm_val, input_dims, output_dims, action_dims)
        self.critic = nn.ModuleList()
        self.critic.append(nn.Sequential(nn.Conv2d(self.input_dims[0], 32, kernel_size=8, stride=4), nn.ReLU(), nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(), nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU()))
        _conv_out_size = self._get_conv_out_size(self.input_dims)
        self.critic.append(nn.Sequential(nn.Linear(_conv_out_size, 512), nn.ReLU(), nn.Linear(512, self.output_dims)))

    def _get_conv_out_size(self, input_dims):
        out = self.critic[0](torch.zeros(input_dims).unsqueeze(0))
        return int(np.prod(out.size()))

    def _init_weights(self):
        relu_gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(self.critic[0][0].weight.data, relu_gain)
        nn.init.constant_(self.critic[0][0].bias.data, 0)
        nn.init.orthogonal_(self.critic[0][2].weight.data, relu_gain)
        nn.init.constant_(self.critic[0][2].bias.data, 0)
        nn.init.orthogonal_(self.critic[0][4].weight.data, relu_gain)
        nn.init.constant_(self.critic[0][4].bias.data, 0)
        nn.init.orthogonal_(self.critic[1][0].weight.data, relu_gain)
        nn.init.constant_(self.critic[1][0].bias.data, 0)
        nn.init.orthogonal_(self.critic[1][2].weight.data, relu_gain)
        nn.init.constant_(self.critic[1][2].bias.data, 0)

    def forward(self, input):
        qvalue = self.critic[1](self.critic[0](input / self.norm_val).view(input.size(0), -1))
        return qvalue

    def get_action(self, input, enable_per=False, eps=0.0, device=torch.device('cpu')):
        forward_flag = True
        action, qvalue, max_qvalue = None, None, None
        input = torch.FloatTensor(input).unsqueeze(0)
        if eps > 0.0 and np.random.uniform() < eps:
            action = np.random.randint(self.output_dims, size=(input.size(0), self.action_dims))
            if not enable_per:
                forward_flag = False
        if forward_flag:
            qvalues = self.forward(input)
            max_qvalue, max_action = qvalues.max(dim=1, keepdim=True)
            max_qvalue = max_qvalue.item()
            max_action = max_action.item()
            if action is None:
                qvalue, action = max_qvalue, max_action
                action = np.array([[action]])
            elif enable_per:
                qvalue = qvalues[0][action[0][0]].item()
        return action, qvalue, max_qvalue


class DqnMlpModel(Model):

    def __init__(self, args, norm_val, input_dims, output_dims, action_dims):
        super(DqnMlpModel, self).__init__(args, norm_val, input_dims, output_dims, action_dims)
        self.hidden_dims = 256
        self.critic = nn.Sequential(nn.Linear(self.input_dims[0] * self.input_dims[1] * self.input_dims[2], self.hidden_dims), nn.ReLU(), nn.Linear(self.hidden_dims, self.hidden_dims), nn.ReLU(), nn.Linear(self.hidden_dims, self.hidden_dims), nn.ReLU(), nn.Linear(self.hidden_dims, self.output_dims))
        self._reset()

    def _init_weights(self):
        pass

    def forward(self, input):
        input = input.view(input.size(0), -1)
        qvalue = self.critic(input)
        return qvalue

    def get_action(self, input, eps=0.0):
        input = torch.FloatTensor(input).unsqueeze(0)
        if eps > 0.0 and np.random.uniform() < eps:
            action = np.random.randint(self.output_dims, size=(input.size(0), self.action_dims))
        else:
            qvalue = self.forward(input)
            _, action = qvalue.max(dim=1, keepdim=True)
            action = action.numpy()
        return action


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (DdpgMlpModel,
     lambda: ([], {'args': SimpleNamespace(), 'norm_val': 4, 'input_dims': [4, 4, 4], 'output_dims': 4, 'action_dims': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (DqnMlpModel,
     lambda: ([], {'args': SimpleNamespace(), 'norm_val': 4, 'input_dims': [4, 4, 4], 'output_dims': 4, 'action_dims': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
]

