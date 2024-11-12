
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


import time


from copy import deepcopy


from typing import Tuple


import torch


import torch.nn as nn


from torch.distributions import Normal


from torch.optim import Adam


from typing import Dict


import numpy as np


import warnings


from torch.utils.tensorboard import SummaryWriter


import logging


from typing import Optional


from typing import Any


from torch import Tensor


import random


import copy


import matplotlib.pyplot as plt


import pandas as pd


from copy import copy


def formatter(src: 'str', firstUpper: 'bool'=True):
    arr = src.split('_')
    res = ''
    for i in arr:
        res = res + i[0].upper() + i[1:]
    if not firstUpper:
        res = res[0].lower() + res[1:]
    return res


def create_apprfunc(**kwargs):
    apprfunc_name = kwargs['apprfunc']
    apprfunc_file_name = apprfunc_name.lower()
    try:
        file = importlib.import_module('networks.' + apprfunc_file_name)
    except NotImplementedError:
        raise NotImplementedError('This apprfunc does not exist')
    name = formatter(kwargs['name'])
    if hasattr(file, name):
        apprfunc_cls = getattr(file, name)
        apprfunc = apprfunc_cls(**kwargs)
    else:
        raise NotImplementedError('This apprfunc is not properly defined')
    return apprfunc


def get_apprfunc_dict(key: 'str', type=None, **kwargs):
    var = dict()
    var['apprfunc'] = kwargs[key + '_func_type']
    var['name'] = kwargs[key + '_func_name']
    var['obs_dim'] = kwargs['obsv_dim']
    var['min_log_std'] = kwargs.get(key + '_min_log_std', float('-20'))
    var['max_log_std'] = kwargs.get(key + '_max_log_std', float('2'))
    var['std_type'] = kwargs.get(key + '_std_type', 'mlp_shared')
    var['norm_matrix'] = kwargs.get('norm_matrix', None)
    apprfunc_type = kwargs[key + '_func_type']
    if apprfunc_type == 'MLP':
        var['hidden_sizes'] = kwargs[key + '_hidden_sizes']
        var['hidden_activation'] = kwargs[key + '_hidden_activation']
        var['output_activation'] = kwargs[key + '_output_activation']
    elif apprfunc_type == 'CNN':
        var['hidden_activation'] = kwargs[key + '_hidden_activation']
        var['output_activation'] = kwargs[key + '_output_activation']
        var['conv_type'] = kwargs[key + '_conv_type']
    elif apprfunc_type == 'CNN_SHARED':
        if key == 'feature':
            var['conv_type'] = kwargs['conv_type']
        else:
            var['feature_net'] = kwargs['feature_net']
            var['hidden_activation'] = kwargs[key + '_hidden_activation']
            var['output_activation'] = kwargs[key + '_output_activation']
    else:
        raise NotImplementedError
    if kwargs['action_type'] == 'continu':
        var['act_high_lim'] = np.array(kwargs['action_high_limit'])
        var['act_low_lim'] = np.array(kwargs['action_low_limit'])
        var['act_dim'] = kwargs['action_dim']
    else:
        raise NotImplementedError("DSAC don't support discrete action space!")
    var['action_distribution_cls'] = getattr(sys.modules[__name__], kwargs['policy_act_distribution'])
    return var


class ApproxContainer(torch.nn.Module):
    """Approximate function container for DSAC.

    Contains one policy and one action value.
    """

    def __init__(self, **kwargs):
        super().__init__()
        q_args = get_apprfunc_dict('value', kwargs['value_func_type'], **kwargs)
        self.q: 'nn.Module' = create_apprfunc(**q_args)
        self.q_target = deepcopy(self.q)
        policy_args = get_apprfunc_dict('policy', kwargs['policy_func_type'], **kwargs)
        self.policy: 'nn.Module' = create_apprfunc(**policy_args)
        self.policy_target = deepcopy(self.policy)
        for p in self.policy_target.parameters():
            p.requires_grad = False
        for p in self.q_target.parameters():
            p.requires_grad = False
        self.log_alpha = nn.Parameter(torch.tensor(1, dtype=torch.float32))
        self.q_optimizer = Adam(self.q.parameters(), lr=kwargs['value_learning_rate'])
        self.policy_optimizer = Adam(self.policy.parameters(), lr=kwargs['policy_learning_rate'])
        self.alpha_optimizer = Adam([self.log_alpha], lr=kwargs['alpha_learning_rate'])

    def create_action_distributions(self, logits):
        return self.policy.get_act_dist(logits)


def CNN(kernel_sizes, channels, strides, activation, input_channel):
    """Implementation of CNN.
    :param list kernel_sizes: list of kernel_size,
    :param list channels: list of channels,
    :param list strides: list of stride,
    :param activation: activation function,
    :param int input_channel: number of channels of input image.
    Return CNN.
    Input shape for CNN: (batch_size, channel_num, height, width).
    """
    layers = []
    for j in range(len(kernel_sizes)):
        act = activation
        if j == 0:
            layers += [nn.Conv2d(input_channel, channels[j], kernel_sizes[j], strides[j]), act()]
        else:
            layers += [nn.Conv2d(channels[j - 1], channels[j], kernel_sizes[j], strides[j]), act()]
    return nn.Sequential(*layers)


class Feature(nn.Module):
    """
    CNN for extracting features from picture.
    """

    def __init__(self, **kwargs):
        super(Feature, self).__init__()
        obs_dim = kwargs['obs_dim']
        conv_type = kwargs['conv_type']
        if conv_type == 'type_1':
            conv_kernel_sizes = [8, 4, 3]
            conv_channels = [32, 64, 64]
            conv_strides = [4, 2, 1]
            conv_activation = nn.ReLU
            conv_input_channel = obs_dim[0]
        elif conv_type == 'type_2':
            conv_kernel_sizes = [4, 3, 3, 3, 3, 3]
            conv_channels = [8, 16, 32, 64, 128, 256]
            conv_strides = [2, 2, 2, 2, 1, 1]
            conv_activation = nn.ReLU
            conv_input_channel = obs_dim[0]
        else:
            raise NotImplementedError
        self.conv = CNN(conv_kernel_sizes, conv_channels, conv_strides, conv_activation, conv_input_channel)


class Action_Distribution:

    def __init__(self):
        super().__init__()

    def get_act_dist(self, logits):
        act_dist_cls = getattr(self, 'action_distribution_cls')
        has_act_lim = hasattr(self, 'act_high_lim')
        act_dist = act_dist_cls(logits)
        if has_act_lim:
            act_dist.act_high_lim = getattr(self, 'act_high_lim')
            act_dist.act_low_lim = getattr(self, 'act_low_lim')
        return act_dist


def get_activation_func(key: 'str'):
    assert isinstance(key, str)
    activation_func = None
    if key == 'relu':
        activation_func = nn.ReLU
    elif key == 'elu':
        activation_func = nn.ELU
    elif key == 'gelu':
        activation_func = nn.GELU
    elif key == 'selu':
        activation_func = nn.SELU
    elif key == 'sigmoid':
        activation_func = nn.Sigmoid
    elif key == 'tanh':
        activation_func = nn.Tanh
    elif key == 'linear':
        activation_func = nn.Identity
    if activation_func is None:
        None
        raise RuntimeError
    return activation_func


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


class StochaPolicy(nn.Module, Action_Distribution):
    """
    Approximated function of stochastic policy.
    Input: observation.
    Output: parameters of action distribution.
    """

    def __init__(self, **kwargs):
        super().__init__()
        obs_dim = kwargs['obs_dim']
        act_dim = kwargs['act_dim']
        hidden_sizes = kwargs['hidden_sizes']
        self.std_type = kwargs['std_type']
        if self.std_type == 'mlp_separated':
            pi_sizes = [obs_dim] + list(hidden_sizes) + [act_dim]
            self.mean = mlp(pi_sizes, get_activation_func(kwargs['hidden_activation']), get_activation_func(kwargs['output_activation']))
            self.log_std = mlp(pi_sizes, get_activation_func(kwargs['hidden_activation']), get_activation_func(kwargs['output_activation']))
        elif self.std_type == 'mlp_shared':
            pi_sizes = [obs_dim] + list(hidden_sizes) + [act_dim * 2]
            self.policy = mlp(pi_sizes, get_activation_func(kwargs['hidden_activation']), get_activation_func(kwargs['output_activation']))
        elif self.std_type == 'parameter':
            pi_sizes = [obs_dim] + list(hidden_sizes) + [act_dim]
            self.mean = mlp(pi_sizes, get_activation_func(kwargs['hidden_activation']), get_activation_func(kwargs['output_activation']))
            self.log_std = nn.Parameter(-0.5 * torch.ones(1, act_dim))
        self.min_log_std = kwargs['min_log_std']
        self.max_log_std = kwargs['max_log_std']
        self.register_buffer('act_high_lim', torch.from_numpy(kwargs['act_high_lim']))
        self.register_buffer('act_low_lim', torch.from_numpy(kwargs['act_low_lim']))
        self.action_distribution_cls = kwargs['action_distribution_cls']

    def forward(self, obs):
        if self.std_type == 'mlp_separated':
            action_mean = self.mean(obs)
            action_std = torch.clamp(self.log_std(obs), self.min_log_std, self.max_log_std).exp()
        elif self.std_type == 'mlp_shared':
            logits = self.policy(obs)
            action_mean, action_log_std = torch.chunk(logits, chunks=2, dim=-1)
            action_std = torch.clamp(action_log_std, self.min_log_std, self.max_log_std).exp()
        elif self.std_type == 'parameter':
            action_mean = self.mean(obs)
            action_log_std = self.log_std + torch.zeros_like(action_mean)
            action_std = torch.clamp(action_log_std, self.min_log_std, self.max_log_std).exp()
        return torch.cat((action_mean, action_std), dim=-1)


class ActionValueDistri(nn.Module):
    """
    Approximated function of distributed action-value function.
    Input: observation.
    Output: parameters of action-value distribution.
    """

    def __init__(self, **kwargs):
        super().__init__()
        obs_dim = kwargs['obs_dim']
        act_dim = kwargs['act_dim']
        hidden_sizes = kwargs['hidden_sizes']
        self.q = mlp([obs_dim + act_dim] + list(hidden_sizes) + [2], get_activation_func(kwargs['hidden_activation']), get_activation_func(kwargs['output_activation']))
        if 'min_log_std' in kwargs or 'max_log_std' in kwargs:
            warnings.warn('min_log_std and max_log_std are deprecated in ActionValueDistri.')

    def forward(self, obs, act):
        logits = self.q(torch.cat([obs, act], dim=-1))
        value_mean, value_std = torch.chunk(logits, chunks=2, dim=-1)
        value_log_std = torch.nn.functional.softplus(value_std)
        return torch.cat((value_mean, value_log_std), dim=-1)

