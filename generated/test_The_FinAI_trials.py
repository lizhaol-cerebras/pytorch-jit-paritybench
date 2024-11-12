
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


from torch import nn


import math


from typing import List


from typing import Optional


from typing import Tuple


import numpy as np


import pandas as pd


import torch


import torch as th


from torch.nn.functional import one_hot


from torch.nn.utils.rnn import pack_padded_sequence


from torch.nn.utils.rnn import pad_packed_sequence


from typing import Callable


from typing import Dict


from typing import Type


from typing import Union


class TemporalAttention(nn.Module):
    """The attention mechanism for temporal information"""

    def __init__(self, hidden_dim, asset_num, **kwargs):
        super(TemporalAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.asset_num = asset_num
        self.fnn = nn.Sequential(nn.Linear(4 * self.hidden_dim, self.hidden_dim), nn.LeakyReLU(), nn.LayerNorm(self.hidden_dim))

    def forward(self, batch_size, present_output, history):
        """Attention over history with present output

        :params batch_size: int
        :params present_output: (B x N x 1 x 2H)
        :params history: (B x N x (T - 1) x 2H)
        """
        attention_scores = th.bmm(present_output, history.permute(0, 2, 1))
        attention_scores = attention_scores / np.sqrt(present_output.shape[-1])
        attention_output = th.bmm(attention_scores, history)
        attention_output = th.cat([present_output, attention_output], -1).reshape(batch_size, self.asset_num, 4 * self.hidden_dim)
        attention_output = self.fnn(attention_output)
        return attention_output, attention_scores


class ParallelSelection(nn.Module):
    """Parallel selection policy network

    :param feature_dim: dimension of the features, which is N x M
    :param hidden_dim: dimension of the hidden representations as M
    :param asset_num: number of assets as N
    :param last_layer_dim_pi: (int) number of units of the policy network
    """

    def __init__(self, feature_dim: 'int', hidden_dim: 'int', asset_num: 'int', last_layer_dim_pi: 'int'=60, **kwargs):
        super(ParallelSelection, self).__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.asset_num = asset_num
        self.last_layer_dim_pi = last_layer_dim_pi

    def forward(self, features: 'th.Tensor') ->th.Tensor:
        """Calculate logits and reverse them to select two assets

        :return: (th.Tensor) latent_policy
        """
        features = features.reshape(-1, self.asset_num, self.hidden_dim)
        features_self_attention = th.bmm(features, features.permute(0, 2, 1))
        triu_indices = th.triu_indices(self.asset_num, self.asset_num, 1)
        triu_vector = features_self_attention[:, triu_indices[0], triu_indices[1]]
        return triu_vector


class SimpleSerialSelection(nn.Module):
    """Simple serial selection policy network

    :param feature_dim: dimension of the features, which is N x M
    :param hidden_dim: dimension of the hidden representations as M
    :param asset_num: number of assets as N
    :param last_layer_dim_pi: (int) number of units of the policy network
    """

    def __init__(self, feature_dim: 'int', hidden_dim: 'int', asset_num: 'int', last_layer_dim_pi: 'int'=60, **kwargs):
        super(SimpleSerialSelection, self).__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.asset_num = asset_num
        self.last_layer_dim_pi = last_layer_dim_pi
        self.selection_vector = nn.Parameter(th.Tensor(self.hidden_dim).unsqueeze(0))

    def forward(self, features: 'th.Tensor') ->th.Tensor:
        """Calculate logits and reverse them to select two assets

        :return: (th.Tensor) latent_policy
        """
        features = features.reshape(-1, self.asset_num, self.hidden_dim)
        batch_size = features.size(0)
        forward_logits = th.bmm(features, self.selection_vector.unsqueeze(-1).repeat(batch_size, 1, 1)).squeeze(-1)
        backward_logits = -forward_logits
        return th.cat([forward_logits, backward_logits], dim=1)


class SerialSelection(nn.Module):
    """Serial selection policy network based on attentions

    :param feature_dim: dimension of the features, which is N x M
    :param hidden_dim: dimension of the hidden representations as M
    :param asset_num: number of assets as N
    :param last_layer_dim_pi: (int) number of units of the policy network
    """

    def __init__(self, feature_dim: 'int', hidden_dim: 'int', asset_num: 'int', num_heads: 'int', last_layer_dim_pi: 'int'=60, **kwargs):
        super(SerialSelection, self).__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.asset_num = asset_num
        self.num_heads = num_heads
        self.last_layer_dim_pi = last_layer_dim_pi
        self.selection_vector = nn.Parameter(th.ones(self.hidden_dim).unsqueeze(0))
        self.first_cross_attention = nn.MultiheadAttention(self.hidden_dim, self.num_heads, batch_first=True)
        self.second_cross_attention = nn.MultiheadAttention(self.hidden_dim, self.num_heads, batch_first=True)

    def forward(self, features: 'th.Tensor') ->th.Tensor:
        """Calculate logits and reverse them to select two assets

        :return: (th.Tensor) latent_policy
        """
        features = features.reshape(-1, self.asset_num, self.hidden_dim)
        batch_size = features.size(0)
        first_logits = th.nn.Softmax(dim=1)(th.bmm(features, self.selection_vector.unsqueeze(-1).repeat(batch_size, 1, 1)))
        output = th.bmm(first_logits.permute(0, 2, 1), features)
        second_logits = th.nn.Softmax(dim=-1)(th.bmm(output, features.permute(0, 2, 1)))
        pair_logits = th.bmm(first_logits, second_logits)
        triu_indices = th.triu_indices(self.asset_num, self.asset_num, 1)
        triu_vector = pair_logits[:, triu_indices[0], triu_indices[1]]
        return triu_vector


POLICY_NETWORKS = {'simple_serial_selection': SimpleSerialSelection, 'serial_selection': SerialSelection, 'parallel_selection': ParallelSelection}


class PairSelectionNetwork(nn.Module):
    """Pair selection policy and mlp value function.
    It receives as input the features extracted by the feature extractor.

    :param feature_dim: dimension of the features.
    :param hidden_dim: dimension of the hidden representations as M
    :param asset_num: number of assets as N
    :param last_layer_dim_pi: (int) number of units of the policy network
    :param last_layer_dim_vf: (int) number of units of the value network
    """

    def __init__(self, policy: 'str', feature_dim: 'int', hidden_dim: 'int', asset_num: 'int', num_heads: 'int', last_layer_dim_pi: 'int'=60, last_layer_dim_vf: 'int'=64):
        super(PairSelectionNetwork, self).__init__()
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf
        self.hidden_dim = hidden_dim
        self.asset_num = asset_num
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.policy_net = POLICY_NETWORKS[policy](feature_dim, hidden_dim, asset_num, last_layer_dim_pi=last_layer_dim_pi, num_heads=num_heads)
        self.value_net = nn.Sequential(nn.Linear(hidden_dim * asset_num, last_layer_dim_vf), nn.ReLU())

    def forward(self, features: 'th.Tensor') ->Tuple[th.Tensor, th.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value
            If all layers are shared, then ``latent_policy == latent_value``
        """
        batch_size = features.shape[0]
        return self.policy_net(features), self.value_net(features.reshape(batch_size, -1))

    def forward_actor(self, features: 'th.Tensor') ->th.Tensor:
        return self.policy_net(features)

    def forward_critic(self, features: 'th.Tensor') ->th.Tensor:
        batch_size = features.shape[0]
        return self.value_net(features.reshape(batch_size, -1))


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (ParallelSelection,
     lambda: ([], {'feature_dim': 4, 'hidden_dim': 4, 'asset_num': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (SerialSelection,
     lambda: ([], {'feature_dim': 4, 'hidden_dim': 4, 'asset_num': 4, 'num_heads': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (SimpleSerialSelection,
     lambda: ([], {'feature_dim': 4, 'hidden_dim': 4, 'asset_num': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
]

