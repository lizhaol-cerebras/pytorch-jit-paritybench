
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


import numpy as np


from torch.distributions import TanhTransform


import torch.nn.functional as F


from torch.distributions import Normal


from torch.utils.tensorboard import SummaryWriter


def initialize_weights(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.kaiming_uniform_(m.weight.data, nonlinearity='relu')
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight.data)
        nn.init.constant_(m.bias.data, 0)


def build_network(input_size, hidden_size, num_layers, activation, output_size):
    assert num_layers >= 2, 'num_layers must be at least 2'
    activation = getattr(nn, activation)()
    layers = []
    layers.append(nn.Linear(input_size, hidden_size))
    layers.append(activation)
    for i in range(num_layers - 2):
        layers.append(nn.Linear(hidden_size, hidden_size))
        layers.append(activation)
    layers.append(nn.Linear(hidden_size, output_size))
    network = nn.Sequential(*layers)
    network.apply(initialize_weights)
    return network


def create_normal_dist(x, std=None, mean_scale=1, init_std=0, min_std=0.1, activation=None, event_shape=None):
    if std == None:
        mean, std = torch.chunk(x, 2, -1)
        mean = mean / mean_scale
        if activation:
            mean = activation(mean)
        mean = mean_scale * mean
        std = F.softplus(std + init_std) + min_std
    else:
        mean = x
    dist = torch.distributions.Normal(mean, std)
    if event_shape:
        dist = torch.distributions.Independent(dist, event_shape)
    return dist


class Actor(nn.Module):

    def __init__(self, discrete_action_bool, action_size, config):
        super().__init__()
        self.config = config.parameters.dreamer.agent.actor
        self.discrete_action_bool = discrete_action_bool
        self.stochastic_size = config.parameters.dreamer.stochastic_size
        self.deterministic_size = config.parameters.dreamer.deterministic_size
        action_size = action_size if discrete_action_bool else 2 * action_size
        self.network = build_network(self.stochastic_size + self.deterministic_size, self.config.hidden_size, self.config.num_layers, self.config.activation, action_size)

    def forward(self, posterior, deterministic):
        x = torch.cat((posterior, deterministic), -1)
        x = self.network(x)
        if self.discrete_action_bool:
            dist = torch.distributions.OneHotCategorical(logits=x)
            action = dist.sample() + dist.probs - dist.probs.detach()
        else:
            dist = create_normal_dist(x, mean_scale=self.config.mean_scale, init_std=self.config.init_std, min_std=self.config.min_std, activation=torch.tanh)
            dist = torch.distributions.TransformedDistribution(dist, TanhTransform())
            action = torch.distributions.Independent(dist, 1).rsample()
        return action


def horizontal_forward(network, x, y=None, input_shape=(-1,), output_shape=(-1,)):
    batch_with_horizon_shape = x.shape[:-len(input_shape)]
    if not batch_with_horizon_shape:
        batch_with_horizon_shape = 1,
    if y is not None:
        x = torch.cat((x, y), -1)
        input_shape = x.shape[-1],
    x = x.reshape(-1, *input_shape)
    x = network(x)
    x = x.reshape(*batch_with_horizon_shape, *output_shape)
    return x


class Critic(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config.parameters.dreamer.agent.critic
        self.stochastic_size = config.parameters.dreamer.stochastic_size
        self.deterministic_size = config.parameters.dreamer.deterministic_size
        self.network = build_network(self.stochastic_size + self.deterministic_size, self.config.hidden_size, self.config.num_layers, self.config.activation, 1)

    def forward(self, posterior, deterministic):
        x = horizontal_forward(self.network, posterior, deterministic, output_shape=(1,))
        dist = create_normal_dist(x, std=1, event_shape=1)
        return dist


class Decoder(nn.Module):

    def __init__(self, observation_shape, config):
        super().__init__()
        self.config = config.parameters.dreamer.decoder
        self.stochastic_size = config.parameters.dreamer.stochastic_size
        self.deterministic_size = config.parameters.dreamer.deterministic_size
        activation = getattr(nn, self.config.activation)()
        self.observation_shape = observation_shape
        self.network = nn.Sequential(nn.Linear(self.deterministic_size + self.stochastic_size, self.config.depth * 32), nn.Unflatten(1, (self.config.depth * 32, 1)), nn.Unflatten(2, (1, 1)), nn.ConvTranspose2d(self.config.depth * 32, self.config.depth * 4, self.config.kernel_size, self.config.stride), activation, nn.ConvTranspose2d(self.config.depth * 4, self.config.depth * 2, self.config.kernel_size, self.config.stride), activation, nn.ConvTranspose2d(self.config.depth * 2, self.config.depth * 1, self.config.kernel_size + 1, self.config.stride), activation, nn.ConvTranspose2d(self.config.depth * 1, self.observation_shape[0], self.config.kernel_size + 1, self.config.stride))
        self.network.apply(initialize_weights)

    def forward(self, posterior, deterministic):
        x = horizontal_forward(self.network, posterior, deterministic, output_shape=self.observation_shape)
        dist = create_normal_dist(x, std=1, event_shape=len(self.observation_shape))
        return dist


class Encoder(nn.Module):

    def __init__(self, observation_shape, config):
        super().__init__()
        self.config = config.parameters.dreamer.encoder
        activation = getattr(nn, self.config.activation)()
        self.observation_shape = observation_shape
        self.network = nn.Sequential(nn.Conv2d(self.observation_shape[0], self.config.depth * 1, self.config.kernel_size, self.config.stride), activation, nn.Conv2d(self.config.depth * 1, self.config.depth * 2, self.config.kernel_size, self.config.stride), activation, nn.Conv2d(self.config.depth * 2, self.config.depth * 4, self.config.kernel_size, self.config.stride), activation, nn.Conv2d(self.config.depth * 4, self.config.depth * 8, self.config.kernel_size, self.config.stride), activation)
        self.network.apply(initialize_weights)

    def forward(self, x):
        return horizontal_forward(self.network, x, input_shape=self.observation_shape)


class RecurrentModel(nn.Module):

    def __init__(self, action_size, config):
        super().__init__()
        self.config = config.parameters.dreamer.rssm.recurrent_model
        self.device = config.operation.device
        self.stochastic_size = config.parameters.dreamer.stochastic_size
        self.deterministic_size = config.parameters.dreamer.deterministic_size
        self.activation = getattr(nn, self.config.activation)()
        self.linear = nn.Linear(self.stochastic_size + action_size, self.config.hidden_size)
        self.recurrent = nn.GRUCell(self.config.hidden_size, self.deterministic_size)

    def forward(self, embedded_state, action, deterministic):
        x = torch.cat((embedded_state, action), 1)
        x = self.activation(self.linear(x))
        x = self.recurrent(x, deterministic)
        return x

    def input_init(self, batch_size):
        return torch.zeros(batch_size, self.deterministic_size)


class RepresentationModel(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config.parameters.dreamer.rssm.representation_model
        self.embedded_state_size = config.parameters.dreamer.embedded_state_size
        self.stochastic_size = config.parameters.dreamer.stochastic_size
        self.deterministic_size = config.parameters.dreamer.deterministic_size
        self.network = build_network(self.embedded_state_size + self.deterministic_size, self.config.hidden_size, self.config.num_layers, self.config.activation, self.stochastic_size * 2)

    def forward(self, embedded_observation, deterministic):
        x = self.network(torch.cat((embedded_observation, deterministic), 1))
        posterior_dist = create_normal_dist(x, min_std=self.config.min_std)
        posterior = posterior_dist.rsample()
        return posterior_dist, posterior


class TransitionModel(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config.parameters.dreamer.rssm.transition_model
        self.device = config.operation.device
        self.stochastic_size = config.parameters.dreamer.stochastic_size
        self.deterministic_size = config.parameters.dreamer.deterministic_size
        self.network = build_network(self.deterministic_size, self.config.hidden_size, self.config.num_layers, self.config.activation, self.stochastic_size * 2)

    def forward(self, x):
        x = self.network(x)
        prior_dist = create_normal_dist(x, min_std=self.config.min_std)
        prior = prior_dist.rsample()
        return prior_dist, prior

    def input_init(self, batch_size):
        return torch.zeros(batch_size, self.stochastic_size)


class RSSM(nn.Module):

    def __init__(self, action_size, config):
        super().__init__()
        self.config = config.parameters.dreamer.rssm
        self.recurrent_model = RecurrentModel(action_size, config)
        self.transition_model = TransitionModel(config)
        self.representation_model = RepresentationModel(config)

    def recurrent_model_input_init(self, batch_size):
        return self.transition_model.input_init(batch_size), self.recurrent_model.input_init(batch_size)


class RewardModel(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config.parameters.dreamer.reward
        self.stochastic_size = config.parameters.dreamer.stochastic_size
        self.deterministic_size = config.parameters.dreamer.deterministic_size
        self.network = build_network(self.stochastic_size + self.deterministic_size, self.config.hidden_size, self.config.num_layers, self.config.activation, 1)

    def forward(self, posterior, deterministic):
        x = horizontal_forward(self.network, posterior, deterministic, output_shape=(1,))
        dist = create_normal_dist(x, std=1, event_shape=1)
        return dist


class ContinueModel(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config.parameters.dreamer.continue_
        self.stochastic_size = config.parameters.dreamer.stochastic_size
        self.deterministic_size = config.parameters.dreamer.deterministic_size
        self.network = build_network(self.stochastic_size + self.deterministic_size, self.config.hidden_size, self.config.num_layers, self.config.activation, 1)

    def forward(self, posterior, deterministic):
        x = horizontal_forward(self.network, posterior, deterministic, output_shape=(1,))
        dist = torch.distributions.Bernoulli(logits=x)
        return dist


class OneStepModel(nn.Module):

    def __init__(self, action_size, config):
        """
        For plan2explore
        There are several variations, but in our implementation,
        we use stochastic and deterministic actions as input and embedded observations as output
        """
        super().__init__()
        self.config = config.parameters.plan2explore.one_step_model
        self.embedded_state_size = config.parameters.dreamer.embedded_state_size
        self.stochastic_size = config.parameters.dreamer.stochastic_size
        self.deterministic_size = config.parameters.dreamer.deterministic_size
        self.action_size = action_size
        self.network = build_network(self.deterministic_size + self.stochastic_size + action_size, self.config.hidden_size, self.config.num_layers, self.config.activation, self.embedded_state_size)

    def forward(self, action, stochastic, deterministic):
        stoch_deter = torch.concat((stochastic, deterministic), axis=-1)
        x = horizontal_forward(self.network, action, stoch_deter, output_shape=(self.embedded_state_size,))
        dist = create_normal_dist(x, std=1, event_shape=1)
        return dist

