
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


import torch


import torch.nn as nn


import torch.optim as optim


import torch.nn.functional as F


from torch import optim


from torch import nn


from copy import deepcopy


from itertools import chain


from torch.nn.parameter import Parameter


from sklearn.exceptions import NotFittedError


from collections import deque


from collections import defaultdict


import warnings


from collections import UserDict


import numbers


from sklearn.ensemble import ExtraTreesRegressor


import itertools


import numpy.random


class CriticNetwork(nn.Module):

    def __init__(self, input_shape, output_shape, **kwargs):
        super().__init__()
        n_input = input_shape[-1]
        n_output = output_shape[0]
        self._h = nn.Linear(n_input, n_output)
        nn.init.xavier_uniform_(self._h.weight, gain=nn.init.calculate_gain('relu'))

    def forward(self, state, action):
        state_action = torch.cat((state.float(), action.float()), dim=1)
        q = F.relu(self._h(state_action))
        return torch.squeeze(q)


class ActorNetwork(nn.Module):

    def __init__(self, input_shape, output_shape, **kwargs):
        super(ActorNetwork, self).__init__()
        n_input = input_shape[-1]
        n_output = output_shape[0]
        self._h = nn.Linear(n_input, n_output)
        nn.init.xavier_uniform_(self._h.weight, gain=nn.init.calculate_gain('relu'))

    def forward(self, state):
        return F.relu(self._h(torch.squeeze(state, 1).float()))


class Network(nn.Module):

    def __init__(self, input_shape, output_shape, n_features, **kwargs):
        super(Network, self).__init__()
        n_input = input_shape[-1]
        n_output = output_shape[0]
        self._h1 = nn.Linear(n_input, n_features)
        self._h2 = nn.Linear(n_features, n_features)
        self._h3 = nn.Linear(n_features, n_output)
        nn.init.xavier_uniform_(self._h1.weight, gain=nn.init.calculate_gain('tanh'))
        nn.init.xavier_uniform_(self._h2.weight, gain=nn.init.calculate_gain('tanh'))
        nn.init.xavier_uniform_(self._h3.weight, gain=nn.init.calculate_gain('linear'))

    def forward(self, state, **kwargs):
        features1 = torch.tanh(self._h1(torch.squeeze(state, -1).float()))
        features2 = torch.tanh(self._h2(features1))
        a = self._h3(features2)
        return a


class FeatureNetwork(nn.Module):

    def __init__(self, input_shape, output_shape, **kwargs):
        super().__init__()

    def forward(self, state, action=None):
        return torch.squeeze(state, 1).float()


def get_recurrent_network(rnn_type):
    if rnn_type == 'vanilla':
        return torch.nn.RNN
    elif rnn_type == 'gru':
        return torch.nn.GRU
    else:
        raise ValueError('Unknown RNN type %s.' % rnn_type)


class PPOCriticBPTTNetwork(torch.nn.Module):

    def __init__(self, input_shape, output_shape, dim_env_state, dim_action, rnn_type, n_hidden_features=128, n_features=128, num_hidden_layers=1, hidden_state_treatment='zero_initial', **kwargs):
        super().__init__()
        assert hidden_state_treatment in ['zero_initial', 'use_policy_hidden_state']
        self._input_shape = input_shape
        self._output_shape = output_shape
        self._dim_env_state = dim_env_state
        self._dim_action = dim_action
        self._use_policy_hidden_states = True if hidden_state_treatment == 'use_policy_hidden_state' else False
        rnn = get_recurrent_network(rnn_type)
        self._h1_o = torch.nn.Linear(dim_env_state, n_features)
        self._h1_o_post_rnn = torch.nn.Linear(dim_env_state, n_features)
        self._rnn = rnn(input_size=n_features, hidden_size=n_hidden_features, num_layers=num_hidden_layers, batch_first=True)
        self._hq_1 = torch.nn.Linear(n_hidden_features + n_features, n_features)
        self._hq_2 = torch.nn.Linear(n_features, 1)
        self._act_func = torch.nn.ReLU()
        torch.nn.init.xavier_uniform_(self._h1_o.weight, gain=torch.nn.init.calculate_gain('relu'))
        torch.nn.init.xavier_uniform_(self._h1_o_post_rnn.weight, gain=torch.nn.init.calculate_gain('relu'))
        torch.nn.init.xavier_uniform_(self._hq_1.weight, gain=torch.nn.init.calculate_gain('relu'))
        torch.nn.init.xavier_uniform_(self._hq_2.weight, gain=torch.nn.init.calculate_gain('relu'))

    def forward(self, state, policy_state, lengths):
        input_rnn = self._act_func(self._h1_o(state))
        packed_seq = torch.nn.utils.rnn.pack_padded_sequence(input_rnn, lengths, enforce_sorted=False, batch_first=True)
        if self._use_policy_hidden_states:
            policy_state_reshaped = policy_state.view(-1, self._num_hidden_layers, self._n_hidden_features)
            policy_state_reshaped = torch.swapaxes(policy_state_reshaped, 0, 1)
            out_rnn, _ = self._rnn(packed_seq, policy_state_reshaped)
        else:
            out_rnn, _ = self._rnn(packed_seq)
        features_rnn, _ = torch.nn.utils.rnn.pad_packed_sequence(out_rnn, batch_first=True)
        rel_indices = lengths.view(-1, 1, 1) - 1
        features_rnn = torch.squeeze(torch.take_along_dim(features_rnn, rel_indices, dim=1), dim=1)
        last_state = torch.squeeze(torch.take_along_dim(state, rel_indices, dim=1), dim=1)
        feature_s = self._act_func(self._h1_o_post_rnn(last_state))
        input_last_layer = torch.concat([feature_s, features_rnn], dim=1)
        q = self._hq_2(self._act_func(self._hq_1(input_last_layer)))
        return torch.squeeze(q)


class PPOActorBPTTNetwork(torch.nn.Module):

    def __init__(self, input_shape, output_shape, n_features, dim_env_state, rnn_type, n_hidden_features, num_hidden_layers=1, **kwargs):
        super().__init__()
        dim_state = input_shape[0]
        dim_action = output_shape[0]
        self._dim_env_state = dim_env_state
        self._num_hidden_layers = num_hidden_layers
        self._n_hidden_features = n_hidden_features
        rnn = get_recurrent_network(rnn_type)
        self._h1_o = torch.nn.Linear(dim_env_state, n_features)
        self._h1_o_post_rnn = torch.nn.Linear(dim_env_state, n_features)
        self._rnn = rnn(input_size=n_features, hidden_size=n_hidden_features, num_layers=num_hidden_layers, batch_first=True)
        self._h3 = torch.nn.Linear(n_hidden_features + n_features, dim_action)
        self._act_func = torch.nn.ReLU()
        self._tanh = torch.nn.Tanh()
        torch.nn.init.xavier_uniform_(self._h1_o.weight, gain=torch.nn.init.calculate_gain('relu') * 0.05)
        torch.nn.init.xavier_uniform_(self._h1_o_post_rnn.weight, gain=torch.nn.init.calculate_gain('relu') * 0.05)
        torch.nn.init.xavier_uniform_(self._h3.weight, gain=torch.nn.init.calculate_gain('relu') * 0.05)

    def forward(self, state, policy_state, lengths):
        input_rnn = self._act_func(self._h1_o(state))
        packed_seq = torch.nn.utils.rnn.pack_padded_sequence(input_rnn, lengths, enforce_sorted=False, batch_first=True)
        policy_state_reshaped = policy_state.view(-1, self._num_hidden_layers, self._n_hidden_features)
        policy_state_reshaped = torch.swapaxes(policy_state_reshaped, 0, 1)
        out_rnn, next_hidden = self._rnn(packed_seq, policy_state_reshaped)
        features_rnn, _ = torch.nn.utils.rnn.pad_packed_sequence(out_rnn, batch_first=True)
        rel_indices = lengths.view(-1, 1, 1) - 1
        features_rnn = torch.squeeze(torch.take_along_dim(features_rnn, rel_indices, dim=1), dim=1)
        last_state = torch.squeeze(torch.take_along_dim(state, rel_indices, dim=1), dim=1)
        feature_sa = self._act_func(self._h1_o_post_rnn(last_state))
        input_last_layer = torch.concat([feature_sa, features_rnn], dim=1)
        a = self._h3(input_last_layer)
        return a, torch.swapaxes(next_hidden, 0, 1)


class LinearNetwork(nn.Module):

    def __init__(self, input_shape, output_shape, **kwargs):
        super().__init__()
        n_input = input_shape[-1]
        n_output = output_shape[0]
        self._h1 = nn.Linear(n_input, n_output)
        nn.init.xavier_uniform_(self._h1.weight, gain=nn.init.calculate_gain('relu'))

    def forward(self, state, action=None):
        q = F.relu(self._h1(torch.squeeze(state, 1).float()))
        if action is None:
            return q
        else:
            action = action.long()
            q_acted = torch.squeeze(q.gather(1, action))
            return q_acted


class StateEmbedding(nn.Module):

    def __init__(self, input_shape):
        super().__init__()
        self._obs_shape = input_shape
        n_input = input_shape[0]
        self._h1 = nn.Conv2d(n_input, 32, kernel_size=8, stride=3)
        self._h2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self._h3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        dummy_obs = torch.zeros(1, *input_shape)
        self._output_shape = np.prod(self._h3(self._h2(self._h1(dummy_obs))).shape),
        nn.init.xavier_uniform_(self._h1.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h2.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h3.weight, gain=nn.init.calculate_gain('relu'))

    def forward(self, state):
        h = state.view(-1, *self._obs_shape).float() / 255.0
        h = F.relu(self._h1(h))
        h = F.relu(self._h2(h))
        h = F.relu(self._h3(h))
        return h


class TorchUtils(object):
    _default_device = 'cpu'

    @classmethod
    def set_default_device(cls, device):
        cls._default_device = device

    @classmethod
    def get_device(cls, device=None):
        return cls._default_device if device is None else device

    @classmethod
    def set_weights(cls, parameters, weights, device=None):
        """
        Function used to set the value of a set of torch parameters given a
        vector of values.

        Args:
            parameters (list): list of parameters to be considered;
            weights (numpy.ndarray): array of the new values for
                the parameters;
            device (str, None): device to use to store the tensor.

        """
        idx = 0
        for p in parameters:
            shape = p.data.shape
            c = 1
            for s in shape:
                c *= s
            w = weights[idx:idx + c].reshape(shape)
            w_tensor = torch.as_tensor(w, device=cls.get_device(device)).type(p.data.dtype)
            p.data = w_tensor
            idx += c

    @staticmethod
    def get_weights(parameters):
        """
        Function used to get the value of a set of torch parameters as
        a single vector of values.

        Args:
            parameters (list): list of parameters to be considered.

        Returns:
            A numpy vector consisting of all the values of the vectors.

        """
        weights = list()
        for p in parameters:
            w = p.data.detach()
            weights.append(w.flatten())
        weights = torch.concatenate(weights)
        return weights

    @staticmethod
    def zero_grad(parameters):
        """
        Function used to set to zero the value of the gradient of a set
        of torch parameters.

        Args:
            parameters (list): list of parameters to be considered.

        """
        for p in parameters:
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()

    @staticmethod
    def get_gradient(params):
        """
        Function used to get the value of the gradient of a set of
        torch parameters.

        Args:
            parameters (list): list of parameters to be considered.

        """
        views = []
        for p in params:
            if p.grad is None:
                view = p.new(p.numel()).zero_()
            else:
                view = p.grad.view(-1)
            views.append(view)
        return torch.cat(views, 0)

    @classmethod
    def to_float_tensor(cls, x, device=None):
        """
        Function used to convert a numpy array to a float torch tensor.

        Args:
            x (np.ndarray): numpy array to be converted as torch tensor;
            device (str, None): device to use to store the tensor.

        Returns:
            A float tensor build from the values contained in the input array.

        """
        return torch.as_tensor(x, device=cls.get_device(device), dtype=torch.float)

    @classmethod
    def to_int_tensor(cls, x, device=None):
        """
        Function used to convert a numpy array to a float torch tensor.

        Args:
            x (np.ndarray): numpy array to be converted as torch tensor;
            device (str, None): device to use to store the tensor.

        Returns:
            A float tensor build from the values contained in the input array.

        """
        return torch.as_tensor(x, device=cls.get_device(device), dtype=torch.int)

    @staticmethod
    def update_optimizer_parameters(optimizer, new_parameters):
        if len(optimizer.state) > 0:
            for p_old, p_new in zip(optimizer.param_groups[0]['params'], new_parameters):
                data = optimizer.state[p_old]
                del optimizer.state[p_old]
                optimizer.state[p_new] = data
        optimizer.param_groups[0]['params'] = new_parameters


eps = torch.finfo(torch.float32).eps


class CategoricalNetwork(nn.Module):

    def __init__(self, input_shape, output_shape, features_network, n_atoms, v_min, v_max, n_features, **kwargs):
        super().__init__()
        self._n_output = output_shape[0]
        self._phi = features_network(input_shape, (n_features,), n_features=n_features, **kwargs)
        self._n_atoms = n_atoms
        self._v_min = v_min
        self._v_max = v_max
        delta = (self._v_max - self._v_min) / (self._n_atoms - 1)
        self._a_values = torch.arange(self._v_min, self._v_max + eps, delta, device=TorchUtils.get_device())
        self._p = nn.ModuleList([nn.Linear(n_features, n_atoms) for _ in range(self._n_output)])
        for i in range(self._n_output):
            nn.init.xavier_uniform_(self._p[i].weight, gain=nn.init.calculate_gain('linear'))

    def forward(self, state, action=None, get_distribution=False):
        features = self._phi(state)
        a_p = [F.softmax(self._p[i](features), -1) for i in range(self._n_output)]
        a_p = torch.stack(a_p, dim=1)
        if not get_distribution:
            q = torch.empty(a_p.shape[:-1])
            for i in range(a_p.shape[0]):
                q[i] = a_p[i] @ self._a_values
            if action is not None:
                return torch.squeeze(q.gather(1, action))
            else:
                return q
        elif action is not None:
            action = torch.unsqueeze(action.long(), 2).repeat(1, 1, self._n_atoms)
            return torch.squeeze(a_p.gather(1, action))
        else:
            return a_p


class DuelingNetwork(nn.Module):

    def __init__(self, input_shape, output_shape, features_network, n_features, avg_advantage, **kwargs):
        super().__init__()
        self._avg_advantage = avg_advantage
        self._n_output = output_shape[0]
        self._phi = features_network(input_shape, (n_features,), n_features=n_features, **kwargs)
        self._A = nn.Linear(n_features, self._n_output)
        self._V = nn.Linear(n_features, 1)
        nn.init.xavier_uniform_(self._A.weight, gain=nn.init.calculate_gain('linear'))
        nn.init.xavier_uniform_(self._V.weight, gain=nn.init.calculate_gain('linear'))

    def forward(self, state, action=None):
        features = self._phi(state)
        advantage = self._A(features)
        value = self._V(features)
        q = value + advantage
        if self._avg_advantage:
            q -= advantage.mean(1).reshape(-1, 1)
        else:
            q -= advantage.max(1).reshape(-1, 1)
        if action is None:
            return q
        else:
            q_acted = torch.squeeze(q.gather(1, action.long()))
            return q_acted


class NoisyNetwork(nn.Module):


    class NoisyLinear(nn.Module):
        __constants__ = ['in_features', 'out_features']

        def __init__(self, in_features, out_features, sigma_coeff=0.5, bias=True):
            super().__init__()
            self.in_features = in_features
            self.out_features = out_features
            self.mu_weight = Parameter(torch.Tensor(out_features, in_features))
            self.sigma_weight = Parameter(torch.Tensor(out_features, in_features))
            if bias:
                self.mu_bias = Parameter(torch.Tensor(out_features))
                self.sigma_bias = Parameter(torch.Tensor(out_features))
            else:
                self.register_parameter('bias', None)
            self._sigma_coeff = sigma_coeff
            self.reset_parameters()

        def reset_parameters(self):
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.mu_weight)
            fan_in **= 0.5
            bound_weight = 1 / fan_in
            bound_sigma = self._sigma_coeff / fan_in
            nn.init.uniform_(self.mu_weight, -bound_weight, bound_weight)
            nn.init.constant_(self.sigma_weight, bound_sigma)
            if hasattr(self, 'mu_bias'):
                nn.init.uniform_(self.mu_bias, -bound_weight, bound_weight)
                nn.init.constant_(self.sigma_bias, bound_sigma)

        def forward(self, input):
            eps_output = torch.rand(self.mu_weight.shape[0], 1, device=TorchUtils.get_device())
            eps_input = torch.rand(1, self.mu_weight.shape[1], device=TorchUtils.get_device())
            eps_dot = torch.matmul(self._noise(eps_output), self._noise(eps_input))
            weight = self.mu_weight + self.sigma_weight * eps_dot
            if hasattr(self, 'mu_bias'):
                self.bias = self.mu_bias + self.sigma_bias * self._noise(eps_output[:, 0])
            return F.linear(input, weight, self.bias)

        @staticmethod
        def _noise(x):
            return torch.sign(x) * torch.sqrt(torch.abs(x))

        def extra_repr(self):
            return 'in_features={}, out_features={}, mu_bias={}, sigma_bias={}'.format(self.in_features, self.out_features, self.mu_bias, self.sigma_bias is not None)

    def __init__(self, input_shape, output_shape, features_network, n_features, **kwargs):
        super().__init__()
        self._n_output = output_shape[0]
        self._phi = features_network(input_shape, (n_features,), n_features=n_features, **kwargs)
        self._Q = self.NoisyLinear(n_features, self._n_output)

    def forward(self, state, action=None):
        features = self._phi(state)
        q = self._Q(features)
        if action is None:
            return q
        else:
            q_acted = torch.squeeze(q.gather(1, action.long()))
            return q_acted


class QuantileNetwork(nn.Module):

    def __init__(self, input_shape, output_shape, features_network, n_quantiles, n_features, **kwargs):
        super().__init__()
        self._n_output = output_shape[0]
        self._phi = features_network(input_shape, (n_features,), n_features=n_features, **kwargs)
        self._n_quantiles = n_quantiles
        self._quant = nn.ModuleList([nn.Linear(n_features, n_quantiles) for _ in range(self._n_output)])
        for i in range(self._n_output):
            nn.init.xavier_uniform_(self._quant[i].weight, gain=nn.init.calculate_gain('linear'))

    def forward(self, state, action=None, get_quantiles=False):
        features = self._phi(state)
        a_quant = [self._quant[i](features) for i in range(self._n_output)]
        a_quant = torch.stack(a_quant, dim=1)
        if not get_quantiles:
            quant = a_quant.mean(-1)
            if action is not None:
                return torch.squeeze(quant.gather(1, action))
            else:
                return quant
        elif action is not None:
            action = torch.unsqueeze(action.long(), 2).repeat(1, 1, self._n_quantiles)
            return torch.squeeze(a_quant.gather(1, action))
        else:
            return a_quant


class RainbowNetwork(nn.Module):

    def __init__(self, input_shape, output_shape, features_network, n_atoms, v_min, v_max, n_features, sigma_coeff, **kwargs):
        super().__init__()
        self._n_output = output_shape[0]
        self._phi = features_network(input_shape, (n_features,), n_features=n_features, **kwargs)
        self._n_atoms = n_atoms
        self._v_min = v_min
        self._v_max = v_max
        delta = (self._v_max - self._v_min) / (self._n_atoms - 1)
        self._a_values = torch.arange(self._v_min, self._v_max + eps, delta, device=TorchUtils.get_device())
        self._pv = NoisyNetwork.NoisyLinear(n_features, n_atoms, sigma_coeff)
        self._pa = nn.ModuleList([NoisyNetwork.NoisyLinear(n_features, n_atoms, sigma_coeff) for _ in range(self._n_output)])

    def forward(self, state, action=None, get_distribution=False):
        features = self._phi(state)
        a_pv = self._pv(features)
        a_pa = [self._pa[i](features) for i in range(self._n_output)]
        a_pa = torch.stack(a_pa, dim=1)
        a_pv = a_pv.unsqueeze(1).repeat(1, self._n_output, 1)
        mean_a_pa = a_pa.mean(1, keepdim=True).repeat(1, self._n_output, 1)
        softmax = F.softmax(a_pv + a_pa - mean_a_pa, dim=-1)
        if not get_distribution:
            q = torch.empty(softmax.shape[:-1])
            for i in range(softmax.shape[0]):
                q[i] = softmax[i] @ self._a_values
            if action is not None:
                return torch.squeeze(q.gather(1, action))
            else:
                return q
        elif action is not None:
            action = torch.unsqueeze(action.long(), 2).repeat(1, 1, self._n_atoms)
            return torch.squeeze(softmax.gather(1, action))
        else:
            return softmax


def uniform_grid(n_centers, low, high, eta=0.25, cyclic=False):
    """
    This function is used to create the parameters of uniformly spaced radial
    basis functions with `eta` of overlap. It creates a uniformly spaced grid of
    ``n_centers[i]`` points in each dimension i. Also returns a vector
    containing the appropriate width of the radial basis functions.

    Args:
         n_centers (list): number of centers of each dimension;
         low (np.ndarray): lowest value for each dimension;
         high (np.ndarray): highest value for each dimension;
         eta (float, 0.25): overlap between two radial basis functions;
         cyclic (bool, False): whether the state space is a ring or not

    Returns:
        The uniformly spaced grid and the width vector.

    """
    assert 0 < eta < 1.0
    n_features = len(low)
    w = np.zeros(n_features)
    c = list()
    tot_points = 1
    for i, n in enumerate(n_centers):
        start = low[i]
        end = high[i]
        if n == 1:
            w[i] = abs(end - start) / 2
            c_i = (start + end) / 2.0
            c.append(np.array([c_i]))
        else:
            if cyclic:
                end_new = end - abs(end - start) / n
            else:
                end_new = end
            w[i] = (1 + eta) * abs(end_new - start) / n
            c_i = np.linspace(start, end_new, n)
            c.append(c_i)
        tot_points *= n
    n_rows = 1
    n_cols = 0
    grid = np.zeros((tot_points, n_features))
    for discrete_values in c:
        i1 = 0
        dim = len(discrete_values)
        for i in range(dim):
            for r in range(n_rows):
                idx_r = r + i * n_rows
                for c in range(n_cols):
                    grid[idx_r, c] = grid[r, c]
                grid[idx_r, n_cols] = discrete_values[i1]
            i1 += 1
        n_cols += 1
        n_rows *= len(discrete_values)
    return grid, w


class GenericBasisTensor(nn.Module):
    """
    Abstract Pytorch module to implement a generic basis function.
    All the basis function generated by this module are

    """

    def __init__(self, mu, scale, dim=None, normalized=False):
        """
        Constructor.

        Args:
            mu (np.ndarray): centers of the gaussian RBFs;
            scale (np.ndarray): scales for the RBFs;
            dim (np.ndarray, None): list of dimension to be considered for the computation of the features. If None, all
                dimension are used to compute the features;
            normalized (bool, False): whether the features need to be normalized to sum to one or not;

        """
        self._mu = TorchUtils.to_float_tensor(mu)
        self._scale = TorchUtils.to_float_tensor(scale)
        if dim is not None:
            self._dim = TorchUtils.to_int_tensor(dim)
        else:
            self._dim = None
        self._normalized = normalized
        super().__init__()

    def forward(self, x):
        if self._dim is not None:
            x = torch.index_select(x, 1, self._dim)
        x = x.unsqueeze(1).repeat(1, self._mu.shape[0], 1)
        delta = x - self._mu.repeat(x.shape[0], 1, 1)
        phi = self._basis_function(delta, self._scale)
        if self._normalized:
            return self._normalize(phi).squeeze(-1)
        else:
            return phi.squeeze(-1)

    def _basis_function(self, delta, scale):
        raise NotImplementedError

    @staticmethod
    def _convert_to_scale(w):
        """
        Converts width of a basis function to scale

        Args:
            w (np.ndarray): array of widths of basis function for every dimension

        Returns:
            The array of scales for each basis function in any given dimension

        """
        raise NotImplementedError

    @staticmethod
    def _normalize(raw_phi):
        if len(raw_phi.shape) == 1:
            return torch.nan_to_num(raw_phi / torch.sum(raw_phi, -1), 0.0)
        else:
            return torch.nan_to_num(raw_phi / torch.sum(raw_phi, -1).unsqueeze(1))

    @classmethod
    def is_cyclic(cls):
        """
        Method used to change the basis generation in case of cyclic features.
        Returns:
            Whether the space we consider is cyclic or not.

        """
        return False

    @classmethod
    def generate(cls, n_centers, low, high, dimensions=None, eta=0.25, normalized=False):
        """
        Factory method that generates the list of dictionaries to build the tensors representing a set of uniformly
        spaced radial basis functions with `eta` overlap.

        Args:
            n_centers (list): list of the number of radial basis functions to be used for each dimension;
            low (np.ndarray): lowest value for each dimension;
            high (np.ndarray): highest value for each dimension;
            dimensions (list, None): list of the dimensions of the input to be considered by the feature. The number of
                dimensions must match the number of elements in ``high`` and ``low``;
            eta (float, 0.25): percentage of overlap between the features;
            normalized (bool, False): whether the features need to be normalized to sum to one or not.

        Returns:
            The tensor list.

        """
        n_features = len(low)
        assert len(n_centers) == n_features
        assert len(low) == len(high)
        assert dimensions is None or n_features == len(dimensions)
        mu, w = uniform_grid(n_centers, low, high, eta, cls.is_cyclic())
        scale = cls._convert_to_scale(w)
        tensor_list = [cls(mu, scale, dimensions, normalized)]
        return tensor_list

    @property
    def size(self):
        return self._mu.shape[0]


class GaussianRBFTensor(GenericBasisTensor):

    def _basis_function(self, delta, scale):
        return torch.exp(-torch.sum(delta ** 2 / scale, -1))

    @staticmethod
    def _convert_to_scale(w):
        return 2 * (w / 3) ** 2


class VonMisesBFTensor(GenericBasisTensor):

    def _basis_function(self, delta, scale):
        return torch.exp(torch.sum(torch.cos(2 * np.pi * delta) / scale, -1) - torch.sum(1 / scale))

    @classmethod
    def is_cyclic(cls):
        return True

    @staticmethod
    def _convert_to_scale(w):
        return w


class ConstantTensor(nn.Module):
    """
    Pytorch module to implement a constant function (always one).

    """

    def forward(self, x):
        return torch.ones(x.shape[0], 1)

    @property
    def size(self):
        return 1


class RandomFourierBasis(nn.Module):
    """
    Class implementing Random Fourier basis functions. The value of the feature is computed using the formula:

    .. math::
        \\sin{\\dfrac{PX}{\\nu}+\\varphi}


    where :math:`X` is the input, :math:`P` is a random weights matrix, :math:`\\nu` is the bandwidth parameter and
    :math:`\\varphi` is a bias vector.

    These features have been presented in:

    "Towards generalization and simplicity in continuous control". Rajeswaran A. et Al.. 2017.

    """

    def __init__(self, P, phi, nu):
        """
        Constructor.

        Args:
            P (np.ndarray): weights matrix, every weight should be drawn from a normal distribution;
            phi (np.ndarray): bias vector, every weight should be drawn from a uniform distribution in the interval
                :math: `[-\\pi, \\pi)`;
            nu (float):  bandwidth parameter, it should be chosen approximately as the average pairwise distances
                between different observation vectors.

        """
        self._P = TorchUtils.to_float_tensor(P)
        self._phi = TorchUtils.to_float_tensor(phi)
        self._nu = nu
        super().__init__()

    def forward(self, x):
        return torch.sin(x @ self._P / self._nu + self._phi)

    def __str__(self):
        return str(self._P) + ' ' + str(self._phi)

    @staticmethod
    def generate(nu, n_output, input_size, use_bias=True):
        """
        Factory method to build random fourier basis. Includes a constant tensor into the output.

        Args:
            nu (float):  bandwidth parameter, it should be chosen approximately as the average pairwise distances
                between different observation vectors.
            n_output (int): number of basis to use;
            input_size (int): size of the input.

        Returns:
            The list of the generated fourier basis functions.

        """
        if use_bias:
            n_output -= 1
        P = np.random.randn(input_size, n_output)
        phi = np.random.uniform(-np.pi, np.pi, n_output)
        tensor_list = [RandomFourierBasis(P, phi, nu)]
        if use_bias:
            tensor_list.append(ConstantTensor())
        return tensor_list

    @property
    def size(self):
        return self._phi.shape[0]


class ExampleNet(nn.Module):

    def __init__(self, input_shape, output_shape, **kwargs):
        super(ExampleNet, self).__init__()
        self._q = nn.Linear(input_shape[0], output_shape[0])
        nn.init.xavier_uniform_(self._q.weight, gain=nn.init.calculate_gain('linear'))

    def forward(self, x, a=None):
        x = x.float()
        q = self._q(x)
        if a is None:
            return q
        else:
            action = a.long()
            q_acted = torch.squeeze(q.gather(1, action))
            return q_acted


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (ActorNetwork,
     lambda: ([], {'input_shape': [4, 4], 'output_shape': [4, 4]}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ConstantTensor,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (CriticNetwork,
     lambda: ([], {'input_shape': [4, 4], 'output_shape': [4, 4]}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (ExampleNet,
     lambda: ([], {'input_shape': [4, 4], 'output_shape': [4, 4]}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (FeatureNetwork,
     lambda: ([], {'input_shape': 4, 'output_shape': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (LinearNetwork,
     lambda: ([], {'input_shape': [4, 4], 'output_shape': [4, 4]}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Network,
     lambda: ([], {'input_shape': [4, 4], 'output_shape': [4, 4], 'n_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
]

