
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


import torch.nn as nn


import math


import torch.nn.functional as F


import copy


from collections import defaultdict


import random


import inspect


import torch.distributed as dist


from torch.autograd import Variable


class FixedBernoulli(torch.distributions.Bernoulli):

    def log_probs(self, actions):
        return super.log_prob(actions).view(actions.size(0), -1).sum(-1).unsqueeze(-1)

    def entropy(self):
        return super().entropy().sum(-1)

    def mode(self):
        return torch.gt(self.probs, 0.5).float()


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


class Bernoulli(nn.Module):

    def __init__(self, num_inputs, num_outputs, use_orthogonal=True, gain=0.01):
        super(Bernoulli, self).__init__()
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain)
        self.linear = init_(nn.Linear(num_inputs, num_outputs))

    def forward(self, x):
        x = self.linear(x)
        return FixedBernoulli(logits=x)


class FixedCategorical(torch.distributions.Categorical):

    def sample(self):
        return super().sample()

    def log_probs(self, actions):
        return super().log_prob(actions.squeeze(-1)).view(actions.size(0), -1).sum(-1).unsqueeze(-1)

    def mode(self):
        return self.probs.argmax(dim=-1, keepdim=True)


class Categorical(nn.Module):

    def __init__(self, num_inputs, num_outputs, use_orthogonal=True, gain=0.01):
        super(Categorical, self).__init__()
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain)
        self.linear = init_(nn.Linear(num_inputs, num_outputs))

    def forward(self, x, available_actions=None):
        x = self.linear(x)
        if available_actions is not None:
            x[available_actions == 0] = -10000000000.0
        return FixedCategorical(logits=x)


class AddBias(nn.Module):

    def __init__(self, bias):
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))

    def forward(self, x):
        if x.dim() == 2:
            bias = self._bias.t().view(1, -1)
        else:
            bias = self._bias.t().view(1, -1, 1, 1)
        return x + bias


class FixedNormal(torch.distributions.Normal):

    def log_probs(self, actions):
        return super().log_prob(actions).sum(-1, keepdim=True)

    def entrop(self):
        return super.entropy().sum(-1)

    def mode(self):
        return self.mean


class DiagGaussian(nn.Module):

    def __init__(self, num_inputs, num_outputs, use_orthogonal=True, gain=0.01):
        super(DiagGaussian, self).__init__()
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain)
        self.fc_mean = init_(nn.Linear(num_inputs, num_outputs))
        self.logstd = AddBias(torch.zeros(num_outputs))

    def forward(self, x):
        action_mean = self.fc_mean(x)
        zeros = torch.zeros(action_mean.size())
        if x.is_cuda:
            zeros = zeros
        action_logstd = self.logstd(zeros)
        return FixedNormal(action_mean, action_logstd.exp())


class ACTLayer(nn.Module):

    def __init__(self, action_space, inputs_dim, use_orthogonal, gain):
        super(ACTLayer, self).__init__()
        self.multidiscrete_action = False
        self.continuous_action = False
        self.mixed_action = False
        if action_space.__class__.__name__ == 'Discrete':
            action_dim = action_space.n
            self.action_out = Categorical(inputs_dim, action_dim, use_orthogonal, gain)
        elif action_space.__class__.__name__ == 'Box':
            self.continuous_action = True
            action_dim = action_space.shape[0]
            self.action_out = DiagGaussian(inputs_dim, action_dim, use_orthogonal, gain)
        elif action_space.__class__.__name__ == 'MultiBinary':
            action_dim = action_space.shape[0]
            self.action_out = Bernoulli(inputs_dim, action_dim, use_orthogonal, gain)
        elif action_space.__class__.__name__ == 'MultiDiscrete':
            self.multidiscrete_action = True
            action_dims = action_space.high - action_space.low + 1
            self.action_outs = []
            for action_dim in action_dims:
                self.action_outs.append(Categorical(inputs_dim, action_dim, use_orthogonal, gain))
            self.action_outs = nn.ModuleList(self.action_outs)
        else:
            self.mixed_action = True
            continous_dim = action_space[0].shape[0]
            discrete_dim = action_space[1].n
            self.action_outs = nn.ModuleList([DiagGaussian(inputs_dim, continous_dim, use_orthogonal, gain), Categorical(inputs_dim, discrete_dim, use_orthogonal, gain)])

    def forward(self, x, available_actions=None, deterministic=False):
        if self.mixed_action:
            actions = []
            action_log_probs = []
            for action_out in self.action_outs:
                action_logit = action_out(x)
                action = action_logit.mode() if deterministic else action_logit.sample()
                action_log_prob = action_logit.log_probs(action)
                actions.append(action.float())
                action_log_probs.append(action_log_prob)
            actions = torch.cat(actions, -1)
            action_log_probs = torch.sum(torch.cat(action_log_probs, -1), -1, keepdim=True)
        elif self.multidiscrete_action:
            actions = []
            action_log_probs = []
            for action_out in self.action_outs:
                action_logit = action_out(x)
                action = action_logit.mode() if deterministic else action_logit.sample()
                action_log_prob = action_logit.log_probs(action)
                actions.append(action)
                action_log_probs.append(action_log_prob)
            actions = torch.cat(actions, -1)
            action_log_probs = torch.cat(action_log_probs, -1)
        elif self.continuous_action:
            action_logits = self.action_out(x)
            actions = action_logits.mode() if deterministic else action_logits.sample()
            action_log_probs = action_logits.log_probs(actions)
        else:
            action_logits = self.action_out(x, available_actions)
            actions = action_logits.mode() if deterministic else action_logits.sample()
            action_log_probs = action_logits.log_probs(actions)
        return actions, action_log_probs

    def get_probs(self, x, available_actions=None):
        if self.mixed_action or self.multidiscrete_action:
            action_probs = []
            for action_out in self.action_outs:
                action_logit = action_out(x)
                action_prob = action_logit.probs
                action_probs.append(action_prob)
            action_probs = torch.cat(action_probs, -1)
        elif self.continuous_action:
            action_logits = self.action_out(x)
            action_probs = action_logits.probs
        else:
            action_logits = self.action_out(x, available_actions)
            action_probs = action_logits.probs
        return action_probs

    def get_log_1mp(self, x, action, available_actions=None, active_masks=None):
        action_logits = self.action_out(x, available_actions)
        action_prob = torch.gather(action_logits.probs, 1, action.long())
        action_prob = torch.clamp(action_prob, 0, 1 - 1e-06)
        action_log_1mp = torch.log(1 - action_prob)
        return action_log_1mp

    def evaluate_actions(self, x, action, available_actions=None, active_masks=None):
        if self.mixed_action:
            a, b = action.split((2, 1), -1)
            b = b.long()
            action = [a, b]
            action_log_probs = []
            dist_entropy = []
            for action_out, act in zip(self.action_outs, action):
                action_logit = action_out(x)
                action_log_probs.append(action_logit.log_probs(act))
                if active_masks is not None:
                    if len(action_logit.entropy().shape) == len(active_masks.shape):
                        dist_entropy.append((action_logit.entropy() * active_masks).sum() / active_masks.sum())
                    else:
                        dist_entropy.append((action_logit.entropy() * active_masks.squeeze(-1)).sum() / active_masks.sum())
                else:
                    dist_entropy.append(action_logit.entropy().mean())
            action_log_probs = torch.sum(torch.cat(action_log_probs, -1), -1, keepdim=True)
            dist_entropy = dist_entropy[0] * 0.0025 + dist_entropy[1] * 0.01
        elif self.multidiscrete_action:
            action = torch.transpose(action, 0, 1)
            action_log_probs = []
            dist_entropy = []
            for action_out, act in zip(self.action_outs, action):
                action_logit = action_out(x)
                action_log_probs.append(action_logit.log_probs(act))
                if active_masks is not None:
                    dist_entropy.append((action_logit.entropy() * active_masks.squeeze(-1)).sum() / active_masks.sum())
                else:
                    dist_entropy.append(action_logit.entropy().mean())
            action_log_probs = torch.cat(action_log_probs, -1)
            dist_entropy = torch.tensor(dist_entropy).mean()
        elif self.continuous_action:
            action_logits = self.action_out(x)
            action_log_probs = action_logits.log_probs(action)
            if active_masks is not None:
                dist_entropy = (action_logits.entropy() * active_masks).sum() / active_masks.sum()
            else:
                dist_entropy = action_logits.entropy().mean()
        else:
            action_logits = self.action_out(x, available_actions)
            action_log_probs = action_logits.log_probs(action)
            if active_masks is not None:
                dist_entropy = (action_logits.entropy() * active_masks.squeeze(-1)).sum() / active_masks.sum()
            else:
                dist_entropy = action_logits.entropy().mean()
        return action_log_probs, dist_entropy


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class MLPLayer(nn.Module):

    def __init__(self, input_dim, hidden_size, layer_N, use_orthogonal, activation_id):
        super(MLPLayer, self).__init__()
        self._layer_N = layer_N
        active_func = [nn.Tanh(), nn.ReLU(), nn.LeakyReLU(), nn.ELU()][activation_id]
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        gain = nn.init.calculate_gain(['tanh', 'relu', 'leaky_relu', 'leaky_relu'][activation_id])

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)
        self.fc1 = nn.Sequential(init_(nn.Linear(input_dim, hidden_size)), active_func, nn.LayerNorm(hidden_size))
        self.fc_h = nn.Sequential(init_(nn.Linear(hidden_size, hidden_size)), active_func, nn.LayerNorm(hidden_size))
        self.fc2 = get_clones(self.fc_h, self._layer_N)

    def forward(self, x):
        x = self.fc1(x)
        for i in range(self._layer_N):
            x = self.fc2[i](x)
        return x


class MLPBase(nn.Module):

    def __init__(self, args, obs_shape, use_attn_internal=False, use_cat_self=True):
        super(MLPBase, self).__init__()
        self._use_feature_normalization = args.use_feature_normalization
        self._use_orthogonal = args.use_orthogonal
        self._activation_id = args.activation_id
        self._use_conv1d = args.use_conv1d
        self._stacked_frames = args.stacked_frames
        self._layer_N = args.layer_N
        self.hidden_size = args.hidden_size
        obs_dim = obs_shape[0]
        inputs_dim = obs_dim
        if self._use_feature_normalization:
            self.feature_norm = nn.LayerNorm(obs_dim)
        self.mlp = MLPLayer(inputs_dim, self.hidden_size, self._layer_N, self._use_orthogonal, self._activation_id)

    def forward(self, x):
        if self._use_feature_normalization:
            x = self.feature_norm(x)
        x = self.mlp(x)
        return x

    @property
    def output_size(self):
        return self.hidden_size


class PopArt(torch.nn.Module):

    def __init__(self, input_shape, output_shape, norm_axes=1, beta=0.99999, epsilon=1e-05, device=torch.device('cpu')):
        super(PopArt, self).__init__()
        self.beta = beta
        self.epsilon = epsilon
        self.norm_axes = norm_axes
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.weight = nn.Parameter(torch.Tensor(output_shape, input_shape))
        self.bias = nn.Parameter(torch.Tensor(output_shape))
        self.stddev = nn.Parameter(torch.ones(output_shape), requires_grad=False)
        self.mean = nn.Parameter(torch.zeros(output_shape), requires_grad=False)
        self.mean_sq = nn.Parameter(torch.zeros(output_shape), requires_grad=False)
        self.debiasing_term = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)
        self.mean.zero_()
        self.mean_sq.zero_()
        self.debiasing_term.zero_()

    def forward(self, input_vector):
        if type(input_vector) == np.ndarray:
            input_vector = torch.from_numpy(input_vector)
        input_vector = input_vector
        return F.linear(input_vector, self.weight, self.bias)

    @torch.no_grad()
    def update(self, input_vector):
        if type(input_vector) == np.ndarray:
            input_vector = torch.from_numpy(input_vector)
        input_vector = input_vector
        old_mean, old_stddev = self.mean, self.stddev
        batch_mean = input_vector.mean(dim=tuple(range(self.norm_axes)))
        batch_sq_mean = (input_vector ** 2).mean(dim=tuple(range(self.norm_axes)))
        self.mean.mul_(self.beta).add_(batch_mean * (1.0 - self.beta))
        self.mean_sq.mul_(self.beta).add_(batch_sq_mean * (1.0 - self.beta))
        self.debiasing_term.mul_(self.beta).add_(1.0 * (1.0 - self.beta))
        self.stddev = (self.mean_sq - self.mean ** 2).sqrt().clamp(min=0.0001)
        self.weight = self.weight * old_stddev / self.stddev
        self.bias = (old_stddev * self.bias + old_mean - self.mean) / self.stddev

    def debiased_mean_var(self):
        debiased_mean = self.mean / self.debiasing_term.clamp(min=self.epsilon)
        debiased_mean_sq = self.mean_sq / self.debiasing_term.clamp(min=self.epsilon)
        debiased_var = (debiased_mean_sq - debiased_mean ** 2).clamp(min=0.01)
        return debiased_mean, debiased_var

    def normalize(self, input_vector):
        if type(input_vector) == np.ndarray:
            input_vector = torch.from_numpy(input_vector)
        input_vector = input_vector
        mean, var = self.debiased_mean_var()
        out = (input_vector - mean[(None,) * self.norm_axes]) / torch.sqrt(var)[(None,) * self.norm_axes]
        return out

    def denormalize(self, input_vector):
        if type(input_vector) == np.ndarray:
            input_vector = torch.from_numpy(input_vector)
        input_vector = input_vector
        mean, var = self.debiased_mean_var()
        out = input_vector * torch.sqrt(var)[(None,) * self.norm_axes] + mean[(None,) * self.norm_axes]
        out = out.cpu().numpy()
        return out


class RNNLayer(nn.Module):

    def __init__(self, inputs_dim, outputs_dim, recurrent_N, use_orthogonal):
        super(RNNLayer, self).__init__()
        self._recurrent_N = recurrent_N
        self._use_orthogonal = use_orthogonal
        self.rnn = nn.GRU(inputs_dim, outputs_dim, num_layers=self._recurrent_N)
        for name, param in self.rnn.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name:
                if self._use_orthogonal:
                    nn.init.orthogonal_(param)
                else:
                    nn.init.xavier_uniform_(param)
        self.norm = nn.LayerNorm(outputs_dim)

    def forward(self, x, hxs, masks):
        if x.size(0) == hxs.size(0):
            x, hxs = self.rnn(x.unsqueeze(0), (hxs * masks.repeat(1, self._recurrent_N).unsqueeze(-1)).transpose(0, 1).contiguous())
            x = x.squeeze(0)
            hxs = hxs.transpose(0, 1)
        else:
            N = hxs.size(0)
            T = int(x.size(0) / N)
            x = x.view(T, N, x.size(1))
            masks = masks.view(T, N)
            has_zeros = (masks[1:] == 0.0).any(dim=-1).nonzero().squeeze().cpu()
            if has_zeros.dim() == 0:
                has_zeros = [has_zeros.item() + 1]
            else:
                has_zeros = (has_zeros + 1).numpy().tolist()
            has_zeros = [0] + has_zeros + [T]
            hxs = hxs.transpose(0, 1)
            outputs = []
            for i in range(len(has_zeros) - 1):
                start_idx = has_zeros[i]
                end_idx = has_zeros[i + 1]
                temp = (hxs * masks[start_idx].view(1, -1, 1).repeat(self._recurrent_N, 1, 1)).contiguous()
                rnn_scores, hxs = self.rnn(x[start_idx:end_idx], temp)
                outputs.append(rnn_scores)
            x = torch.cat(outputs, dim=0)
            x = x.reshape(T * N, -1)
            hxs = hxs.transpose(0, 1)
        x = self.norm(x)
        return x, hxs


def check(input):
    if type(input) == np.ndarray:
        return torch.from_numpy(input)


def get_shape_from_obs_space(obs_space):
    if obs_space.__class__.__name__ == 'Box':
        obs_shape = obs_space.shape
    elif obs_space.__class__.__name__ == 'list':
        obs_shape = obs_space
    elif obs_space.__class__.__name__ == 'Dict':
        obs_shape = obs_space.spaces
    else:
        raise NotImplementedError
    return obs_shape


class PolicyNetwork(nn.Module):

    def __init__(self, args, obs_space, action_space, device=torch.device('cpu')):
        super(PolicyNetwork, self).__init__()
        self.hidden_size = args.hidden_size
        self._gain = args.gain
        self._use_orthogonal = args.use_orthogonal
        self._activation_id = args.activation_id
        self._use_policy_active_masks = args.use_policy_active_masks
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._use_recurrent_policy = args.use_recurrent_policy
        self._use_influence_policy = args.use_influence_policy
        self._influence_layer_N = args.influence_layer_N
        self._use_policy_vhead = args.use_policy_vhead
        self._recurrent_N = args.recurrent_N
        self.tpdv = dict(dtype=torch.float32, device=device)
        obs_shape = get_shape_from_obs_space(obs_space)
        self._mixed_obs = False
        self.base = MLPBase(args, obs_shape, use_attn_internal=False, use_cat_self=True)
        input_size = self.base.output_size
        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            self.rnn = RNNLayer(input_size, self.hidden_size, self._recurrent_N, self._use_orthogonal)
            input_size = self.hidden_size
        if self._use_influence_policy:
            self.mlp = MLPLayer(obs_shape[0], self.hidden_size, self._influence_layer_N, self._use_orthogonal, self._activation_id)
            input_size += self.hidden_size
        self.act = ACTLayer(action_space, input_size, self._use_orthogonal, self._gain)
        if self._use_policy_vhead:
            init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][self._use_orthogonal]

            def init_(m):
                return init(m, init_method, lambda x: nn.init.constant_(x, 0))
            if self._use_popart:
                self.v_out = init_(PopArt(input_size, 1, device=device))
            else:
                self.v_out = init_(nn.Linear(input_size, 1))
        self

    def forward(self, obs, rnn_states, masks, available_actions=None, deterministic=False):
        if self._mixed_obs:
            for key in obs.keys():
                obs[key] = check(obs[key])
        else:
            obs = check(obs)
        rnn_states = check(rnn_states)
        masks = check(masks)
        if available_actions is not None:
            available_actions = check(available_actions)
        actor_features = self.base(obs)
        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)
        if self._use_influence_policy:
            mlp_obs = self.mlp(obs)
            actor_features = torch.cat([actor_features, mlp_obs], dim=1)
        actions, action_log_probs = self.act(actor_features, available_actions, deterministic)
        return actions, action_log_probs, rnn_states

    def evaluate_actions(self, obs, rnn_states, action, masks, available_actions=None, active_masks=None):
        if self._mixed_obs:
            for key in obs.keys():
                obs[key] = check(obs[key])
        else:
            obs = check(obs)
        rnn_states = check(rnn_states)
        action = check(action)
        masks = check(masks)
        if available_actions is not None:
            available_actions = check(available_actions)
        if active_masks is not None:
            active_masks = check(active_masks)
        actor_features = self.base(obs)
        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)
        if self._use_influence_policy:
            mlp_obs = self.mlp(obs)
            actor_features = torch.cat([actor_features, mlp_obs], dim=1)
        action_log_probs, dist_entropy = self.act.evaluate_actions(actor_features, action, available_actions, active_masks=active_masks if self._use_policy_active_masks else None)
        values = self.v_out(actor_features) if self._use_policy_vhead else None
        return action_log_probs, dist_entropy, values

    def get_policy_values(self, obs, rnn_states, masks):
        if self._mixed_obs:
            for key in obs.keys():
                obs[key] = check(obs[key])
        else:
            obs = check(obs)
        rnn_states = check(rnn_states)
        masks = check(masks)
        actor_features = self.base(obs)
        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)
        if self._use_influence_policy:
            mlp_obs = self.mlp(obs)
            actor_features = torch.cat([actor_features, mlp_obs], dim=1)
        values = self.v_out(actor_features)
        return values


class ValueNorm(nn.Module):
    """ Normalize a vector of observations - across the first norm_axes dimensions"""

    def __init__(self, input_shape, norm_axes=1, beta=0.99999, per_element_update=False, epsilon=1e-05, device=torch.device('cpu')):
        super(ValueNorm, self).__init__()
        self.input_shape = input_shape
        self.norm_axes = norm_axes
        self.epsilon = epsilon
        self.beta = beta
        self.per_element_update = per_element_update
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.running_mean = nn.Parameter(torch.zeros(input_shape), requires_grad=False)
        self.running_mean_sq = nn.Parameter(torch.zeros(input_shape), requires_grad=False)
        self.debiasing_term = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        self.reset_parameters()

    def reset_parameters(self):
        self.running_mean.zero_()
        self.running_mean_sq.zero_()
        self.debiasing_term.zero_()

    def running_mean_var(self):
        debiased_mean = self.running_mean / self.debiasing_term.clamp(min=self.epsilon)
        debiased_mean_sq = self.running_mean_sq / self.debiasing_term.clamp(min=self.epsilon)
        debiased_var = (debiased_mean_sq - debiased_mean ** 2).clamp(min=0.01)
        return debiased_mean, debiased_var

    @torch.no_grad()
    def update(self, input_vector):
        if type(input_vector) == np.ndarray:
            input_vector = torch.from_numpy(input_vector)
        input_vector = input_vector
        batch_mean = input_vector.mean(dim=tuple(range(self.norm_axes)))
        batch_sq_mean = (input_vector ** 2).mean(dim=tuple(range(self.norm_axes)))
        if self.per_element_update:
            batch_size = np.prod(input_vector.size()[:self.norm_axes])
            weight = self.beta ** batch_size
        else:
            weight = self.beta
        self.running_mean.mul_(weight).add_(batch_mean * (1.0 - weight))
        self.running_mean_sq.mul_(weight).add_(batch_sq_mean * (1.0 - weight))
        self.debiasing_term.mul_(weight).add_(1.0 * (1.0 - weight))

    def normalize(self, input_vector):
        if type(input_vector) == np.ndarray:
            input_vector = torch.from_numpy(input_vector)
        input_vector = input_vector
        mean, var = self.running_mean_var()
        out = (input_vector - mean[(None,) * self.norm_axes]) / torch.sqrt(var)[(None,) * self.norm_axes]
        return out

    def denormalize(self, input_vector):
        """ Transform normalized data back into original distribution """
        if type(input_vector) == np.ndarray:
            input_vector = torch.from_numpy(input_vector)
        input_vector = input_vector
        mean, var = self.running_mean_var()
        out = input_vector * torch.sqrt(var)[(None,) * self.norm_axes] + mean[(None,) * self.norm_axes]
        out = out.cpu().numpy()
        return out


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (Bernoulli,
     lambda: ([], {'num_inputs': 4, 'num_outputs': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Categorical,
     lambda: ([], {'num_inputs': 4, 'num_outputs': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (DiagGaussian,
     lambda: ([], {'num_inputs': 4, 'num_outputs': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (PopArt,
     lambda: ([], {'input_shape': 4, 'output_shape': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
]

