
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


import random


import numpy as np


import torch


from abc import abstractmethod


from collections import defaultdict


from typing import Callable


from typing import Generator


from typing import Generic


from typing import Optional


from typing import Sequence


from typing import TypeVar


from torch import nn


import math


from sklearn.neighbors import NearestNeighbors


import torch.nn.functional as F


from abc import ABCMeta


from typing import Union


from torch.optim import Optimizer


from typing import cast


from torch.distributions import Categorical


from typing import Protocol


from collections import deque


from typing import BinaryIO


from typing import Any


import torch.distributed as dist


import time


from typing import Iterator


from torch.distributions import Normal


from torch.distributions.kl import kl_divergence


from typing import NoReturn


from typing import NamedTuple


from torch.optim.lr_scheduler import CosineAnnealingLR


from torch.optim.lr_scheduler import LambdaLR


from torch.optim.lr_scheduler import LRScheduler


from typing import Iterable


from typing import Mapping


from torch.optim import SGD


from torch.optim import Adam


from torch.optim import AdamW


from torch.optim import RMSprop


import collections


from typing import overload


from torch.cuda import CUDAGraph


from torch.nn.parallel import DistributedDataParallel as DDP


from typing import runtime_checkable


import numpy.typing as npt


import copy


class Encoder(nn.Module, metaclass=ABCMeta):

    @abstractmethod
    def forward(self, x: 'TorchObservation') ->torch.Tensor:
        pass

    def __call__(self, x: 'TorchObservation') ->torch.Tensor:
        return super().__call__(x)


class EncoderWithAction(nn.Module, metaclass=ABCMeta):

    @abstractmethod
    def forward(self, x: 'TorchObservation', action: 'torch.Tensor') ->torch.Tensor:
        pass

    def __call__(self, x: 'TorchObservation', action: 'torch.Tensor') ->torch.Tensor:
        return super().__call__(x, action)


class PixelEncoder(Encoder):
    _cnn_layers: 'nn.Module'
    _last_layers: 'nn.Module'

    def __init__(self, observation_shape: 'Sequence[int]', filters: 'Optional[list[list[int]]]'=None, feature_size: 'int'=512, use_batch_norm: 'bool'=False, dropout_rate: 'Optional[float]'=False, activation: 'nn.Module'=nn.ReLU(), exclude_last_activation: 'bool'=False, last_activation: 'Optional[nn.Module]'=None):
        super().__init__()
        if filters is None:
            filters = [[32, 8, 4], [64, 4, 2], [64, 3, 1]]
        if feature_size is None:
            feature_size = 512
        cnn_layers = []
        in_channels = [observation_shape[0]] + [f[0] for f in filters[:-1]]
        for in_channel, f in zip(in_channels, filters):
            out_channel, kernel_size, stride = f
            conv = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride)
            cnn_layers.append(conv)
            cnn_layers.append(activation)
            if use_batch_norm:
                cnn_layers.append(nn.BatchNorm2d(out_channel))
            if dropout_rate is not None:
                cnn_layers.append(nn.Dropout2d(dropout_rate))
        self._cnn_layers = nn.Sequential(*cnn_layers)
        x = torch.rand((1,) + tuple(observation_shape))
        with torch.no_grad():
            cnn_output_size = self._cnn_layers(x).view(1, -1).shape[1]
        layers: 'list[nn.Module]' = []
        layers.append(nn.Linear(cnn_output_size, feature_size))
        if not exclude_last_activation:
            layers.append(last_activation if last_activation else activation)
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(feature_size))
        if dropout_rate is not None:
            layers.append(nn.Dropout(dropout_rate))
        self._last_layers = nn.Sequential(*layers)

    def forward(self, x: 'TorchObservation') ->torch.Tensor:
        assert isinstance(x, torch.Tensor)
        h = self._cnn_layers(x)
        return self._last_layers(h.reshape(x.shape[0], -1))


class PixelEncoderWithAction(EncoderWithAction):
    _cnn_layers: 'nn.Module'
    _last_layers: 'nn.Module'
    _discrete_action: 'bool'
    _action_size: 'int'

    def __init__(self, observation_shape: 'Sequence[int]', action_size: 'int', filters: 'Optional[list[list[int]]]'=None, feature_size: 'int'=512, use_batch_norm: 'bool'=False, dropout_rate: 'Optional[float]'=False, discrete_action: 'bool'=False, activation: 'nn.Module'=nn.ReLU(), exclude_last_activation: 'bool'=False, last_activation: 'Optional[nn.Module]'=None):
        super().__init__()
        self._discrete_action = discrete_action
        self._action_size = action_size
        if filters is None:
            filters = [[32, 8, 4], [64, 4, 2], [64, 3, 1]]
        if feature_size is None:
            feature_size = 512
        cnn_layers = []
        in_channels = [observation_shape[0]] + [f[0] for f in filters[:-1]]
        for in_channel, f in zip(in_channels, filters):
            out_channel, kernel_size, stride = f
            conv = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride)
            cnn_layers.append(conv)
            cnn_layers.append(activation)
            if use_batch_norm:
                cnn_layers.append(nn.BatchNorm2d(out_channel))
            if dropout_rate is not None:
                cnn_layers.append(nn.Dropout2d(dropout_rate))
        self._cnn_layers = nn.Sequential(*cnn_layers)
        x = torch.rand((1,) + tuple(observation_shape))
        with torch.no_grad():
            cnn_output_size = self._cnn_layers(x).view(1, -1).shape[1]
        layers: 'list[nn.Module]' = []
        layers.append(nn.Linear(cnn_output_size + action_size, feature_size))
        if not exclude_last_activation:
            layers.append(last_activation if last_activation else activation)
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(feature_size))
        if dropout_rate is not None:
            layers.append(nn.Dropout(dropout_rate))
        self._last_layers = nn.Sequential(*layers)

    def forward(self, x: 'TorchObservation', action: 'torch.Tensor') ->torch.Tensor:
        assert isinstance(x, torch.Tensor)
        h = self._cnn_layers(x)
        if self._discrete_action:
            action = F.one_hot(action.view(-1).long(), num_classes=self._action_size).float()
        h = torch.cat([h.reshape(h.shape[0], -1), action], dim=1)
        return self._last_layers(h)


T = TypeVar('T')


def last_flag(iterator: 'Iterable[T]') ->Iterator[tuple[bool, T]]:
    items = list(iterator)
    for i, item in enumerate(items):
        yield i == len(items) - 1, item


class VectorEncoder(Encoder):
    _layers: 'nn.Module'

    def __init__(self, observation_shape: 'Sequence[int]', hidden_units: 'Optional[Sequence[int]]'=None, use_batch_norm: 'bool'=False, use_layer_norm: 'bool'=False, dropout_rate: 'Optional[float]'=None, activation: 'nn.Module'=nn.ReLU(), exclude_last_activation: 'bool'=False, last_activation: 'Optional[nn.Module]'=None):
        super().__init__()
        if hidden_units is None:
            hidden_units = [256, 256]
        layers = []
        in_units = [observation_shape[0]] + list(hidden_units[:-1])
        for is_last, (in_unit, out_unit) in last_flag(zip(in_units, hidden_units)):
            layers.append(nn.Linear(in_unit, out_unit))
            if not is_last or not exclude_last_activation:
                if is_last and last_activation:
                    layers.append(last_activation)
                else:
                    layers.append(activation)
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(out_unit))
            if use_layer_norm:
                layers.append(nn.LayerNorm(out_unit))
            if dropout_rate is not None:
                layers.append(nn.Dropout(dropout_rate))
        self._layers = nn.Sequential(*layers)

    def forward(self, x: 'TorchObservation') ->torch.Tensor:
        assert isinstance(x, torch.Tensor)
        return self._layers(x)


class VectorEncoderWithAction(EncoderWithAction):
    _layers: 'nn.Module'
    _action_size: 'int'
    _discrete_action: 'bool'

    def __init__(self, observation_shape: 'Sequence[int]', action_size: 'int', hidden_units: 'Optional[Sequence[int]]'=None, use_batch_norm: 'bool'=False, use_layer_norm: 'bool'=False, dropout_rate: 'Optional[float]'=None, discrete_action: 'bool'=False, activation: 'nn.Module'=nn.ReLU(), exclude_last_activation: 'bool'=False, last_activation: 'Optional[nn.Module]'=None):
        super().__init__()
        self._action_size = action_size
        self._discrete_action = discrete_action
        if hidden_units is None:
            hidden_units = [256, 256]
        layers = []
        in_units = [observation_shape[0] + action_size] + list(hidden_units[:-1])
        for is_last, (in_unit, out_unit) in last_flag(zip(in_units, hidden_units)):
            layers.append(nn.Linear(in_unit, out_unit))
            if not is_last or not exclude_last_activation:
                if is_last and last_activation:
                    layers.append(last_activation)
                else:
                    layers.append(activation)
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(out_unit))
            if use_layer_norm:
                layers.append(nn.LayerNorm(out_unit))
            if dropout_rate is not None:
                layers.append(nn.Dropout(dropout_rate))
        self._layers = nn.Sequential(*layers)

    def forward(self, x: 'TorchObservation', action: 'torch.Tensor') ->torch.Tensor:
        assert isinstance(x, torch.Tensor)
        if self._discrete_action:
            action = F.one_hot(action.view(-1).long(), num_classes=self._action_size).float()
        x = torch.cat([x, action], dim=1)
        return self._layers(x)


class SimBaBlock(nn.Module):

    def __init__(self, input_size: 'int', hidden_size: 'int', out_size: 'int'):
        super().__init__()
        layers = [nn.LayerNorm(input_size), nn.Linear(input_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, out_size)]
        self._layers = nn.Sequential(*layers)

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        return x + self._layers(x)


class SimBaEncoder(Encoder):

    def __init__(self, observation_shape: 'Sequence[int]', hidden_size: 'int', output_size: 'int', n_blocks: 'int'):
        super().__init__()
        layers = [nn.Linear(observation_shape[0], output_size), *[SimBaBlock(output_size, hidden_size, output_size) for _ in range(n_blocks)], nn.LayerNorm(output_size)]
        self._layers = nn.Sequential(*layers)

    def forward(self, x: 'TorchObservation') ->torch.Tensor:
        assert isinstance(x, torch.Tensor)
        return self._layers(x)


class SimBaEncoderWithAction(EncoderWithAction):

    def __init__(self, observation_shape: 'Sequence[int]', action_size: 'int', hidden_size: 'int', output_size: 'int', n_blocks: 'int', discrete_action: 'bool'):
        super().__init__()
        layers = [nn.Linear(observation_shape[0] + action_size, output_size), *[SimBaBlock(output_size, hidden_size, output_size) for _ in range(n_blocks)], nn.LayerNorm(output_size)]
        self._layers = nn.Sequential(*layers)
        self._action_size = action_size
        self._discrete_action = discrete_action

    def forward(self, x: 'TorchObservation', action: 'torch.Tensor') ->torch.Tensor:
        assert isinstance(x, torch.Tensor)
        if self._discrete_action:
            action = F.one_hot(action.view(-1).long(), num_classes=self._action_size).float()
        h = torch.cat([x, action], dim=1)
        return self._layers(h)


class VAEEncoder(nn.Module):
    _encoder: 'EncoderWithAction'
    _mu: 'nn.Module'
    _logstd: 'nn.Module'
    _min_logstd: 'float'
    _max_logstd: 'float'
    _latent_size: 'int'

    def __init__(self, encoder: 'EncoderWithAction', hidden_size: 'int', latent_size: 'int', min_logstd: 'float'=-20.0, max_logstd: 'float'=2.0):
        super().__init__()
        self._encoder = encoder
        self._mu = nn.Linear(hidden_size, latent_size)
        self._logstd = nn.Linear(hidden_size, latent_size)
        self._min_logstd = min_logstd
        self._max_logstd = max_logstd
        self._latent_size = latent_size

    def forward(self, x: 'TorchObservation', action: 'torch.Tensor') ->Normal:
        h = self._encoder(x, action)
        mu = self._mu(h)
        logstd = self._logstd(h)
        clipped_logstd = logstd.clamp(self._min_logstd, self._max_logstd)
        return Normal(mu, clipped_logstd.exp())

    def __call__(self, x: 'TorchObservation', action: 'torch.Tensor') ->Normal:
        return super().__call__(x, action)

    @property
    def latent_size(self) ->int:
        return self._latent_size


class VAEDecoder(nn.Module):
    _encoder: 'EncoderWithAction'
    _fc: 'nn.Linear'
    _action_size: 'int'

    def __init__(self, encoder: 'EncoderWithAction', hidden_size: 'int', action_size: 'int'):
        super().__init__()
        self._encoder = encoder
        self._fc = nn.Linear(hidden_size, action_size)
        self._action_size = action_size

    def forward(self, x: 'TorchObservation', latent: 'torch.Tensor', with_squash: 'bool') ->torch.Tensor:
        h = self._encoder(x, latent)
        if with_squash:
            return self._fc(h)
        return torch.tanh(self._fc(h))

    def __call__(self, x: 'TorchObservation', latent: 'torch.Tensor', with_squash: 'bool'=True) ->torch.Tensor:
        return super().__call__(x, latent, with_squash)

    @property
    def action_size(self) ->int:
        return self._action_size


class Parameter(nn.Module):
    _parameter: 'nn.Parameter'

    def __init__(self, data: 'torch.Tensor'):
        super().__init__()
        self._parameter = nn.Parameter(data)

    def forward(self) ->NoReturn:
        raise NotImplementedError('Parameter does not support __call__. Use parameter property instead.')

    def __call__(self) ->NoReturn:
        raise NotImplementedError('Parameter does not support __call__. Use parameter property instead.')


class ActionOutput(NamedTuple):
    mu: 'torch.Tensor'
    squashed_mu: 'torch.Tensor'
    logstd: 'Optional[torch.Tensor]'

    def copy_(self, src: "'ActionOutput'") ->None:
        self.mu.copy_(src.mu)
        self.squashed_mu.copy_(src.squashed_mu)
        if self.logstd:
            assert src.logstd is not None
            self.logstd.copy_(src.logstd)


class Policy(nn.Module, metaclass=ABCMeta):

    @abstractmethod
    def forward(self, x: 'TorchObservation', *args: Any) ->ActionOutput:
        pass

    def __call__(self, x: 'TorchObservation', *args: Any) ->ActionOutput:
        return super().__call__(x, *args)


class DeterministicPolicy(Policy):
    _encoder: 'Encoder'
    _fc: 'nn.Linear'

    def __init__(self, encoder: 'Encoder', hidden_size: 'int', action_size: 'int'):
        super().__init__()
        self._encoder = encoder
        self._fc = nn.Linear(hidden_size, action_size)

    def forward(self, x: 'TorchObservation', *args: Any) ->ActionOutput:
        h = self._encoder(x)
        mu = self._fc(h)
        return ActionOutput(mu, torch.tanh(mu), logstd=None)


class DeterministicResidualPolicy(Policy):
    _encoder: 'EncoderWithAction'
    _scale: 'float'
    _fc: 'nn.Linear'

    def __init__(self, encoder: 'EncoderWithAction', hidden_size: 'int', action_size: 'int', scale: 'float'):
        super().__init__()
        self._scale = scale
        self._encoder = encoder
        self._fc = nn.Linear(hidden_size, action_size)

    def forward(self, x: 'TorchObservation', *args: Any) ->ActionOutput:
        action = args[0]
        h = self._encoder(x, action)
        residual_action = self._scale * torch.tanh(self._fc(h))
        action = (action + residual_action).clamp(-1.0, 1.0)
        return ActionOutput(mu=action, squashed_mu=action, logstd=None)


class NormalPolicy(Policy):
    _encoder: 'Encoder'
    _action_size: 'int'
    _min_logstd: 'float'
    _max_logstd: 'float'
    _use_std_parameter: 'bool'
    _mu: 'nn.Linear'
    _logstd: 'Union[nn.Linear, nn.Parameter]'

    def __init__(self, encoder: 'Encoder', hidden_size: 'int', action_size: 'int', min_logstd: 'float', max_logstd: 'float', use_std_parameter: 'bool'):
        super().__init__()
        self._action_size = action_size
        self._encoder = encoder
        self._min_logstd = min_logstd
        self._max_logstd = max_logstd
        self._use_std_parameter = use_std_parameter
        self._mu = nn.Linear(hidden_size, action_size)
        if use_std_parameter:
            initial_logstd = torch.zeros(1, action_size, dtype=torch.float32)
            self._logstd = nn.Parameter(initial_logstd)
        else:
            self._logstd = nn.Linear(hidden_size, action_size)

    def forward(self, x: 'TorchObservation', *args: Any) ->ActionOutput:
        h = self._encoder(x)
        mu = self._mu(h)
        if self._use_std_parameter:
            assert isinstance(self._logstd, nn.Parameter)
            logstd = torch.sigmoid(self._logstd)
            base_logstd = self._max_logstd - self._min_logstd
            clipped_logstd = self._min_logstd + logstd * base_logstd
        else:
            assert isinstance(self._logstd, nn.Linear)
            logstd = self._logstd(h)
            clipped_logstd = logstd.clamp(self._min_logstd, self._max_logstd)
        return ActionOutput(mu, torch.tanh(mu), clipped_logstd)


class CategoricalPolicy(nn.Module):
    _encoder: 'Encoder'
    _fc: 'nn.Linear'

    def __init__(self, encoder: 'Encoder', hidden_size: 'int', action_size: 'int'):
        super().__init__()
        self._encoder = encoder
        self._fc = nn.Linear(hidden_size, action_size)

    def forward(self, x: 'TorchObservation') ->Categorical:
        return Categorical(logits=self._fc(self._encoder(x)))

    def __call__(self, x: 'TorchObservation') ->Categorical:
        return super().__call__(x)


class QFunctionOutput(NamedTuple):
    q_value: 'torch.Tensor'
    quantiles: 'Optional[torch.Tensor]'
    taus: 'Optional[torch.Tensor]'


class ContinuousQFunction(nn.Module, metaclass=ABCMeta):

    @abstractmethod
    def forward(self, x: 'TorchObservation', action: 'torch.Tensor') ->QFunctionOutput:
        pass

    def __call__(self, x: 'TorchObservation', action: 'torch.Tensor') ->QFunctionOutput:
        return super().__call__(x, action)

    @property
    @abstractmethod
    def encoder(self) ->EncoderWithAction:
        pass


class DiscreteQFunction(nn.Module, metaclass=ABCMeta):

    @abstractmethod
    def forward(self, x: 'TorchObservation') ->QFunctionOutput:
        pass

    def __call__(self, x: 'TorchObservation') ->QFunctionOutput:
        return super().__call__(x)

    @property
    @abstractmethod
    def encoder(self) ->Encoder:
        pass


def _make_taus(n_quantiles: 'int', device: 'torch.device') ->torch.Tensor:
    steps = torch.arange(n_quantiles, dtype=torch.float32, device=device)
    taus = ((steps + 1).float() / n_quantiles).view(1, -1)
    taus_dot = (steps.float() / n_quantiles).view(1, -1)
    return (taus + taus_dot) / 2.0


def compute_iqn_feature(h: 'torch.Tensor', taus: 'torch.Tensor', embed: 'nn.Linear', embed_size: 'int') ->torch.Tensor:
    steps = torch.arange(embed_size, device=h.device).float() + 1
    expanded_taus = taus.view(h.shape[0], -1, 1)
    prior = torch.cos(math.pi * steps.view(1, 1, -1) * expanded_taus)
    phi = torch.relu(embed(prior))
    return h.view(h.shape[0], 1, -1) * phi


def get_batch_size(x: 'Union[torch.Tensor, Sequence[torch.Tensor]]') ->int:
    if isinstance(x, torch.Tensor):
        return int(x.shape[0])
    else:
        return int(x[0].shape[0])


def get_device(x: 'Union[torch.Tensor, Sequence[torch.Tensor]]') ->str:
    if isinstance(x, torch.Tensor):
        return str(x.device)
    else:
        return str(x[0].device)


class DiscreteIQNQFunction(DiscreteQFunction):
    _action_size: 'int'
    _encoder: 'Encoder'
    _fc: 'nn.Linear'
    _n_quantiles: 'int'
    _n_greedy_quantiles: 'int'
    _embed_size: 'int'
    _embed: 'nn.Linear'

    def __init__(self, encoder: 'Encoder', hidden_size: 'int', action_size: 'int', n_quantiles: 'int', n_greedy_quantiles: 'int', embed_size: 'int'):
        super().__init__()
        self._encoder = encoder
        self._action_size = action_size
        self._fc = nn.Linear(hidden_size, self._action_size)
        self._n_quantiles = n_quantiles
        self._n_greedy_quantiles = n_greedy_quantiles
        self._embed_size = embed_size
        self._embed = nn.Linear(embed_size, hidden_size)

    def forward(self, x: 'TorchObservation') ->QFunctionOutput:
        h = self._encoder(x)
        if self.training:
            n_quantiles = self._n_quantiles
        else:
            n_quantiles = self._n_greedy_quantiles
        taus = _make_taus(batch_size=get_batch_size(x), n_quantiles=n_quantiles, training=self.training, device=torch.device(get_device(x)))
        prod = compute_iqn_feature(h, taus, self._embed, self._embed_size)
        quantiles = self._fc(prod).transpose(1, 2)
        return QFunctionOutput(q_value=quantiles.mean(dim=2), quantiles=quantiles, taus=taus)

    @property
    def encoder(self) ->Encoder:
        return self._encoder


class ContinuousIQNQFunction(ContinuousQFunction, nn.Module):
    _encoder: 'EncoderWithAction'
    _fc: 'nn.Linear'
    _n_quantiles: 'int'
    _n_greedy_quantiles: 'int'
    _embed_size: 'int'
    _embed: 'nn.Linear'

    def __init__(self, encoder: 'EncoderWithAction', hidden_size: 'int', n_quantiles: 'int', n_greedy_quantiles: 'int', embed_size: 'int'):
        super().__init__()
        self._encoder = encoder
        self._fc = nn.Linear(hidden_size, 1)
        self._n_quantiles = n_quantiles
        self._n_greedy_quantiles = n_greedy_quantiles
        self._embed_size = embed_size
        self._embed = nn.Linear(embed_size, hidden_size)

    def forward(self, x: 'TorchObservation', action: 'torch.Tensor') ->QFunctionOutput:
        h = self._encoder(x, action)
        if self.training:
            n_quantiles = self._n_quantiles
        else:
            n_quantiles = self._n_greedy_quantiles
        taus = _make_taus(batch_size=get_batch_size(x), n_quantiles=n_quantiles, training=self.training, device=torch.device(get_device(x)))
        prod = compute_iqn_feature(h, taus, self._embed, self._embed_size)
        quantiles = self._fc(prod).view(h.shape[0], -1)
        return QFunctionOutput(q_value=quantiles.mean(dim=1, keepdim=True), quantiles=quantiles, taus=taus)

    @property
    def encoder(self) ->EncoderWithAction:
        return self._encoder


class DiscreteMeanQFunction(DiscreteQFunction):
    _encoder: 'Encoder'
    _fc: 'nn.Linear'

    def __init__(self, encoder: 'Encoder', hidden_size: 'int', action_size: 'int'):
        super().__init__()
        self._encoder = encoder
        self._fc = nn.Linear(hidden_size, action_size)

    def forward(self, x: 'TorchObservation') ->QFunctionOutput:
        return QFunctionOutput(q_value=self._fc(self._encoder(x)), quantiles=None, taus=None)

    @property
    def encoder(self) ->Encoder:
        return self._encoder


class ContinuousMeanQFunction(ContinuousQFunction):
    _encoder: 'EncoderWithAction'
    _fc: 'nn.Linear'

    def __init__(self, encoder: 'EncoderWithAction', hidden_size: 'int'):
        super().__init__()
        self._encoder = encoder
        self._fc = nn.Linear(hidden_size, 1)

    def forward(self, x: 'TorchObservation', action: 'torch.Tensor') ->QFunctionOutput:
        return QFunctionOutput(q_value=self._fc(self._encoder(x, action)), quantiles=None, taus=None)

    @property
    def encoder(self) ->EncoderWithAction:
        return self._encoder


class DiscreteQRQFunction(DiscreteQFunction):
    _action_size: 'int'
    _encoder: 'Encoder'
    _n_quantiles: 'int'
    _fc: 'nn.Linear'

    def __init__(self, encoder: 'Encoder', hidden_size: 'int', action_size: 'int', n_quantiles: 'int'):
        super().__init__()
        self._encoder = encoder
        self._action_size = action_size
        self._n_quantiles = n_quantiles
        self._fc = nn.Linear(hidden_size, action_size * n_quantiles)

    def forward(self, x: 'TorchObservation') ->QFunctionOutput:
        quantiles = self._fc(self._encoder(x))
        quantiles = quantiles.view(-1, self._action_size, self._n_quantiles)
        return QFunctionOutput(q_value=quantiles.mean(dim=2), quantiles=quantiles, taus=_make_taus(self._n_quantiles, device=get_device(x)))

    @property
    def encoder(self) ->Encoder:
        return self._encoder


class ContinuousQRQFunction(ContinuousQFunction):
    _encoder: 'EncoderWithAction'
    _fc: 'nn.Linear'
    _n_quantiles: 'int'

    def __init__(self, encoder: 'EncoderWithAction', hidden_size: 'int', n_quantiles: 'int'):
        super().__init__()
        self._encoder = encoder
        self._fc = nn.Linear(hidden_size, n_quantiles)
        self._n_quantiles = n_quantiles

    def forward(self, x: 'TorchObservation', action: 'torch.Tensor') ->QFunctionOutput:
        quantiles = self._fc(self._encoder(x, action))
        return QFunctionOutput(q_value=quantiles.mean(dim=1, keepdim=True), quantiles=quantiles, taus=_make_taus(self._n_quantiles, device=get_device(x)))

    @property
    def encoder(self) ->EncoderWithAction:
        return self._encoder


def create_attention_mask(context_size: 'int') ->torch.Tensor:
    mask = torch.ones(context_size, context_size, dtype=torch.float32)
    return torch.tril(mask).view(1, 1, context_size, context_size)


class CausalSelfAttention(nn.Module):
    _num_heads: 'int'
    _context_size: 'int'
    _k: 'nn.Linear'
    _q: 'nn.Linear'
    _v: 'nn.Linear'
    _proj: 'nn.Linear'
    _attn_dropout: 'nn.Dropout'
    _proj_dropout: 'nn.Dropout'
    _mask: 'torch.Tensor'

    def __init__(self, embed_size: 'int', num_heads: 'int', context_size: 'int', attn_dropout: 'float', resid_dropout: 'float'):
        super().__init__()
        self._num_heads = num_heads
        self._context_size = context_size
        self._k = nn.Linear(embed_size, embed_size)
        self._q = nn.Linear(embed_size, embed_size)
        self._v = nn.Linear(embed_size, embed_size)
        self._proj = nn.Linear(embed_size, embed_size)
        self._attn_dropout = nn.Dropout(attn_dropout)
        self._proj_dropout = nn.Dropout(resid_dropout)
        mask = create_attention_mask(context_size)
        self.register_buffer('_mask', mask)

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        assert x.dim() == 3, f'Expects (B, T, N), but got {x.shape}'
        batch_size, context_size, _ = x.shape
        assert context_size <= self._context_size, 'Exceeds context_size'
        shape = batch_size, context_size, self._num_heads, -1
        k = self._k(x).view(shape).transpose(1, 2)
        q = self._q(x).view(shape).transpose(1, 2)
        v = self._v(x).view(shape).transpose(1, 2)
        qkT = torch.matmul(q, k.transpose(2, 3))
        attention = qkT / math.sqrt(k.shape[-1])
        attention = attention.masked_fill(self._mask[..., :context_size, :context_size] == 0, float('-inf'))
        attention = F.softmax(attention, dim=-1)
        attention = self._attn_dropout(attention)
        output = torch.matmul(attention, v)
        output = output.transpose(1, 2).reshape(batch_size, context_size, -1)
        return self._proj_dropout(self._proj(output))


class MLP(nn.Module):
    _l1: 'nn.Linear'
    _l2: 'nn.Linear'
    _dropout: 'nn.Dropout'
    _activation: 'nn.Module'

    def __init__(self, in_size: 'int', out_size: 'int', pre_activation_hidden_size: 'int', post_activation_hidden_size: 'int', dropout: 'float', activation: 'nn.Module'):
        super().__init__()
        self._l1 = nn.Linear(in_size, pre_activation_hidden_size)
        self._l2 = nn.Linear(post_activation_hidden_size, out_size)
        self._dropout = nn.Dropout(dropout)
        self._activation = activation

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        h = self._activation(self._l1(x))
        h = self._dropout(self._l2(h))
        return h


class Block(nn.Module):
    _attention: 'CausalSelfAttention'
    _mlp: 'MLP'
    _layer_norm1: 'nn.LayerNorm'
    _layer_norm2: 'nn.LayerNorm'

    def __init__(self, layer_width: 'int', pre_activation_ff_hidden_size: 'int', post_activation_ff_hidden_size: 'int', num_heads: 'int', context_size: 'int', attn_dropout: 'float', resid_dropout: 'float', activation: 'nn.Module'):
        super().__init__()
        self._attention = CausalSelfAttention(embed_size=layer_width, num_heads=num_heads, context_size=context_size, attn_dropout=attn_dropout, resid_dropout=resid_dropout)
        self._mlp = MLP(in_size=layer_width, out_size=layer_width, pre_activation_hidden_size=pre_activation_ff_hidden_size, post_activation_hidden_size=post_activation_ff_hidden_size, dropout=resid_dropout, activation=activation)
        self._layer_norm1 = nn.LayerNorm(layer_width, eps=0.003)
        self._layer_norm2 = nn.LayerNorm(layer_width, eps=0.003)

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        norm_x = self._layer_norm1(x)
        x = x + self._attention(norm_x)
        norm_x = self._layer_norm2(x)
        x = x + self._mlp(norm_x)
        return x


class PositionEncoding(nn.Module, metaclass=ABCMeta):

    @abstractmethod
    def forward(self, t: 'torch.Tensor') ->torch.Tensor:
        raise NotImplementedError


class SimplePositionEncoding(PositionEncoding):

    def __init__(self, embed_dim: 'int', max_timestep: 'int'):
        super().__init__()
        self._embed = nn.Embedding(max_timestep, embed_dim)

    def forward(self, t: 'torch.Tensor') ->torch.Tensor:
        assert t.dim() == 2, 'Expects (B, T)'
        return self._embed(t)


def get_parameter(parameter: 'Parameter') ->nn.Parameter:
    return next(parameter.parameters())


class GlobalPositionEncoding(PositionEncoding):

    def __init__(self, embed_dim: 'int', max_timestep: 'int', context_size: 'int'):
        super().__init__()
        self._embed_dim = embed_dim
        self._global_position_embedding = nn.Embedding(max_timestep, embed_dim)
        self._block_position_embedding = Parameter(torch.zeros(1, 3 * context_size, embed_dim, dtype=torch.float32))

    def forward(self, t: 'torch.Tensor') ->torch.Tensor:
        assert t.dim() == 2, 'Expects (B, T)'
        _, context_size = t.shape
        global_embedding = self._global_position_embedding(t[:, -1:])
        block_embedding = get_parameter(self._block_position_embedding)[:, :context_size, :]
        return global_embedding + block_embedding


class GPT2(nn.Module):
    _transformer: 'nn.Sequential'
    _layer_norm: 'nn.LayerNorm'
    _dropout: 'nn.Dropout'

    def __init__(self, layer_width: 'int', pre_activation_ff_hidden_size: 'int', post_activation_ff_hidden_size: 'int', num_heads: 'int', context_size: 'int', num_layers: 'int', attn_dropout: 'float', resid_dropout: 'float', embed_dropout: 'float', activation: 'nn.Module'):
        super().__init__()
        blocks = [Block(layer_width=layer_width, pre_activation_ff_hidden_size=pre_activation_ff_hidden_size, post_activation_ff_hidden_size=post_activation_ff_hidden_size, num_heads=num_heads, context_size=context_size, attn_dropout=attn_dropout, resid_dropout=resid_dropout, activation=activation) for _ in range(num_layers)]
        self._transformer = nn.Sequential(*blocks)
        self._layer_norm = nn.LayerNorm(layer_width, eps=0.003)
        self._dropout = nn.Dropout(embed_dropout)

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        h = self._dropout(x)
        h = self._transformer(h)
        h = self._layer_norm(h)
        return h


def _init_weights(module: 'nn.Module') ->None:
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)


class ContinuousDecisionTransformer(nn.Module):
    _encoder: 'Encoder'
    _position_encoding: 'PositionEncoding'
    _action_embed: 'nn.Linear'
    _rtg_embed: 'nn.Linear'
    _gpt2: 'GPT2'
    _output: 'nn.Linear'

    def __init__(self, encoder: 'Encoder', embed_size: 'int', position_encoding: 'PositionEncoding', action_size: 'int', num_heads: 'int', context_size: 'int', num_layers: 'int', attn_dropout: 'float', resid_dropout: 'float', embed_dropout: 'float', activation: 'nn.Module'):
        super().__init__()
        self._position_encoding = position_encoding
        self._embed_ln = nn.LayerNorm(embed_size)
        self._gpt2 = GPT2(layer_width=embed_size, pre_activation_ff_hidden_size=4 * embed_size, post_activation_ff_hidden_size=4 * embed_size, num_heads=num_heads, context_size=3 * context_size, num_layers=num_layers, attn_dropout=attn_dropout, resid_dropout=resid_dropout, embed_dropout=embed_dropout, activation=activation)
        self.apply(_init_weights)
        self._encoder = encoder
        self._rtg_embed = nn.Linear(1, embed_size)
        self._action_embed = nn.Linear(action_size, embed_size)
        self._output = nn.Linear(embed_size, action_size)

    def forward(self, x: 'TorchObservation', action: 'torch.Tensor', return_to_go: 'torch.Tensor', timesteps: 'torch.Tensor') ->torch.Tensor:
        batch_size, context_size, _ = return_to_go.shape
        position_embedding = self._position_encoding(timesteps)
        if isinstance(x, torch.Tensor):
            flat_x = x.view(-1, *x.shape[2:])
        else:
            flat_x = [_x.view(-1, *_x.shape[2:]) for _x in x]
        flat_state_embedding = self._encoder(flat_x)
        state_embedding = flat_state_embedding.view(batch_size, context_size, -1)
        state_embedding = state_embedding + position_embedding
        action_embedding = self._action_embed(action) + position_embedding
        rtg_embedding = self._rtg_embed(return_to_go) + position_embedding
        h = torch.stack([rtg_embedding, state_embedding, action_embedding], dim=1)
        h = h.transpose(1, 2).reshape(batch_size, 3 * context_size, -1)
        if not self.training:
            h = h[:, :-1, :]
        h = self._gpt2(self._embed_ln(h))
        return torch.tanh(self._output(h[:, 1::3, :]))


class DiscreteDecisionTransformer(nn.Module):
    _encoder: 'Encoder'
    _position_encoding: 'PositionEncoding'
    _action_embed: 'nn.Embedding'
    _rtg_embed: 'nn.Linear'
    _gpt2: 'GPT2'
    _output: 'nn.Linear'
    _embed_activation: 'nn.Module'

    def __init__(self, encoder: 'Encoder', embed_size: 'int', position_encoding: 'PositionEncoding', action_size: 'int', num_heads: 'int', context_size: 'int', num_layers: 'int', attn_dropout: 'float', resid_dropout: 'float', embed_dropout: 'float', activation: 'nn.Module', embed_activation: 'nn.Module'):
        super().__init__()
        self._position_encoding = position_encoding
        self._gpt2 = GPT2(layer_width=embed_size, pre_activation_ff_hidden_size=4 * embed_size, post_activation_ff_hidden_size=4 * embed_size, num_heads=num_heads, context_size=3 * context_size, num_layers=num_layers, attn_dropout=attn_dropout, resid_dropout=resid_dropout, embed_dropout=embed_dropout, activation=activation)
        self._output = nn.Linear(embed_size, action_size, bias=False)
        self._action_embed = nn.Embedding(action_size, embed_size)
        self.apply(_init_weights)
        self._encoder = encoder
        self._rtg_embed = nn.Linear(1, embed_size)
        self._embed_activation = embed_activation

    def forward(self, x: 'TorchObservation', action: 'torch.Tensor', return_to_go: 'torch.Tensor', timesteps: 'torch.Tensor') ->tuple[torch.Tensor, torch.Tensor]:
        batch_size, context_size, _ = return_to_go.shape
        position_embedding = self._position_encoding(timesteps)
        if isinstance(x, torch.Tensor):
            flat_x = x.reshape(-1, *x.shape[2:])
        else:
            flat_x = [_x.reshape(-1, *_x.shape[2:]) for _x in x]
        flat_state_embedding = self._encoder(flat_x)
        state_embedding = flat_state_embedding.view(batch_size, context_size, -1)
        flat_action = action.view(batch_size, context_size).long()
        action_embedding = self._action_embed(flat_action)
        rtg_embedding = self._rtg_embed(return_to_go)
        h = torch.stack([rtg_embedding, state_embedding, action_embedding], dim=1)
        h = self._embed_activation(h)
        h = h + position_embedding.view(batch_size, 1, context_size, -1)
        h = h.transpose(1, 2).reshape(batch_size, 3 * context_size, -1)
        if not self.training:
            h = h[:, :-1, :]
        h = self._gpt2(h)
        logits = self._output(h[:, 1::3, :])
        return F.softmax(logits, dim=-1), logits


class GEGLU(nn.Module):

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        assert x.shape[-1] % 2 == 0
        a, b = x.chunk(2, dim=-1)
        return a * F.gelu(b)


class GatoTransformer(nn.Module):
    _gpt2: 'GPT2'
    _token_embed: 'nn.Embedding'
    _observation_pos_embed: 'nn.Embedding'
    _action_pos_embed: 'Parameter'
    _output: 'nn.Linear'
    _embed_activation: 'nn.Module'

    def __init__(self, layer_width: 'int', max_observation_length: 'int', vocab_size: 'int', num_heads: 'int', context_size: 'int', num_layers: 'int', attn_dropout: 'float', resid_dropout: 'float', embed_dropout: 'float', embed_activation: 'nn.Module'):
        super().__init__()
        self._gpt2 = GPT2(layer_width=layer_width, pre_activation_ff_hidden_size=2 * 4 * layer_width, post_activation_ff_hidden_size=4 * layer_width, num_heads=num_heads, context_size=context_size, num_layers=num_layers, attn_dropout=attn_dropout, resid_dropout=resid_dropout, embed_dropout=embed_dropout, activation=GEGLU())
        self._output = nn.Linear(layer_width, vocab_size, bias=False)
        self._token_embed = nn.Embedding(vocab_size + 1, layer_width)
        self._observation_pos_embed = nn.Embedding(max_observation_length, layer_width)
        self._action_pos_embed = Parameter(torch.zeros(1, 1, layer_width, dtype=torch.float32))
        self.apply(_init_weights)
        self._embed_activation = embed_activation

    def forward(self, tokens: 'torch.Tensor', observation_masks: 'torch.Tensor', observation_positions: 'torch.Tensor', action_masks: 'torch.Tensor') ->tuple[torch.Tensor, torch.Tensor]:
        assert tokens.ndim == 2
        batch_size, context_size = tokens.shape
        assert observation_masks.shape == (batch_size, context_size, 1)
        assert observation_positions.shape == (batch_size, context_size)
        assert action_masks.shape == (batch_size, context_size, 1)
        embeddings = self._embed_activation(self._token_embed(tokens))
        embeddings = embeddings + observation_masks * self._observation_pos_embed(observation_positions)
        embeddings = embeddings + action_masks * get_parameter(self._action_pos_embed)
        h = self._gpt2(embeddings)
        logits = self._output(h)
        return F.softmax(logits, dim=-1), logits


class ValueFunction(nn.Module):
    _encoder: 'Encoder'
    _fc: 'nn.Linear'

    def __init__(self, encoder: 'Encoder', hidden_size: 'int'):
        super().__init__()
        self._encoder = encoder
        self._fc = nn.Linear(hidden_size, 1)

    def forward(self, x: 'TorchObservation') ->torch.Tensor:
        h = self._encoder(x)
        return cast(torch.Tensor, self._fc(h))

    def __call__(self, x: 'TorchObservation') ->torch.Tensor:
        return cast(torch.Tensor, super().__call__(x))


class View(nn.Module):
    _shape: 'Sequence[int]'

    def __init__(self, shape: 'Sequence[int]'):
        super().__init__()
        self._shape = shape

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        return x.view(self._shape)


class Swish(nn.Module):

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        return x * torch.sigmoid(x)


class QFunction(nn.Module):

    def __init__(self, observation_shape: 'Sequence[int]', action_size: 'int'):
        super().__init__()
        self._fc1 = nn.Linear(observation_shape[0], 256)
        self._fc2 = nn.Linear(256, 256)
        self._fc3 = nn.Linear(256, action_size)

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        h = torch.relu(self._fc1(x))
        h = torch.relu(self._fc2(h))
        return self._fc3(h)


class TupleEncoder(Encoder):

    def __init__(self, observation_shape: 'Shape'):
        super().__init__()
        shape1, shape2 = observation_shape
        assert isinstance(shape1, (tuple, list))
        assert isinstance(shape2, (tuple, list))
        self.fc1 = nn.Linear(shape1[0], 256)
        self.fc2 = nn.Linear(shape2[0], 256)
        self.shared = nn.Linear(256 * 2, 256)

    def forward(self, x: 'TorchObservation') ->torch.Tensor:
        h1 = self.fc1(x[0])
        h2 = self.fc2(x[1])
        return self.shared(torch.cat([h1, h2], dim=1))


class TupleEncoderWithAction(EncoderWithAction):

    def __init__(self, observation_shape: 'Shape', action_size: 'int'):
        super().__init__()
        shape1, shape2 = observation_shape
        assert isinstance(shape1, (tuple, list))
        assert isinstance(shape2, (tuple, list))
        self.fc1 = nn.Linear(shape1[0], 256)
        self.fc2 = nn.Linear(shape2[0], 256)
        self.shared = nn.Linear(256 * 2 + action_size, 256)

    def forward(self, x: 'TorchObservation', action: 'torch.Tensor') ->torch.Tensor:
        h1 = self.fc1(x[0])
        h2 = self.fc2(x[1])
        return self.shared(torch.cat([h1, h2, action], dim=1))


class DummyEncoder(Encoder):

    def __init__(self, input_shape: 'Shape'):
        super().__init__()
        self.input_shape = input_shape
        self.dummy_parameter = torch.nn.Parameter(torch.rand(1, self.get_feature_size()))

    def forward(self, x: 'TorchObservation') ->torch.Tensor:
        if isinstance(x, torch.Tensor):
            y = x.view(x.shape[0], -1)
        else:
            batch_size = x[0].shape[0]
            y = torch.cat([_x.view(batch_size, -1) for _x in x], dim=-1)
        return y + self.dummy_parameter

    def get_feature_size(self) ->int:
        if isinstance(self.input_shape[0], int):
            return int(np.cumprod(self.input_shape)[-1])
        else:
            return sum([np.cumprod(shape)[-1] for shape in self.input_shape])


class DummyEncoderWithAction(EncoderWithAction):

    def __init__(self, input_shape: 'Shape', action_size: 'int'):
        super().__init__()
        self.input_shape = input_shape
        self._action_size = action_size
        self.dummy_parameter = torch.nn.Parameter(torch.rand(1, self.get_feature_size()))

    def forward(self, x: 'TorchObservation', action: 'torch.Tensor') ->torch.Tensor:
        if isinstance(x, torch.Tensor):
            y = x.view(x.shape[0], -1)
        else:
            batch_size = x[0].shape[0]
            y = torch.cat([_x.view(batch_size, -1) for _x in x], dim=-1)
        return torch.cat([y, action], dim=-1) + self.dummy_parameter

    def get_feature_size(self) ->int:
        if isinstance(self.input_shape[0], int):
            feature_size = int(np.cumprod(self.input_shape)[-1])
        else:
            feature_size = sum([np.cumprod(shape)[-1] for shape in self.input_shape])
        return feature_size + self._action_size


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (CategoricalPolicy,
     lambda: ([], {'encoder': torch.nn.ReLU(), 'hidden_size': 4, 'action_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (CausalSelfAttention,
     lambda: ([], {'embed_size': 4, 'num_heads': 4, 'context_size': 4, 'attn_dropout': 0.5, 'resid_dropout': 0.5}),
     lambda: ([torch.rand([4, 4, 4])], {})),
    (DeterministicPolicy,
     lambda: ([], {'encoder': torch.nn.ReLU(), 'hidden_size': 4, 'action_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (DiscreteMeanQFunction,
     lambda: ([], {'encoder': torch.nn.ReLU(), 'hidden_size': 4, 'action_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (DiscreteQRQFunction,
     lambda: ([], {'encoder': torch.nn.ReLU(), 'hidden_size': 4, 'action_size': 4, 'n_quantiles': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (DummyEncoder,
     lambda: ([], {'input_shape': [4, 4]}),
     lambda: ([torch.rand([4, 16])], {})),
    (DummyEncoderWithAction,
     lambda: ([], {'input_shape': [4, 4], 'action_size': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4])], {})),
    (GEGLU,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (QFunction,
     lambda: ([], {'observation_shape': [4, 4], 'action_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (SimBaBlock,
     lambda: ([], {'input_size': 4, 'hidden_size': 4, 'out_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (SimBaEncoder,
     lambda: ([], {'observation_shape': [4, 4], 'hidden_size': 4, 'output_size': 4, 'n_blocks': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Swish,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ValueFunction,
     lambda: ([], {'encoder': torch.nn.ReLU(), 'hidden_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (VectorEncoder,
     lambda: ([], {'observation_shape': [4, 4]}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (VectorEncoderWithAction,
     lambda: ([], {'observation_shape': [4, 4], 'action_size': 4}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4])], {})),
    (View,
     lambda: ([], {'shape': 4}),
     lambda: ([torch.rand([4])], {})),
]

