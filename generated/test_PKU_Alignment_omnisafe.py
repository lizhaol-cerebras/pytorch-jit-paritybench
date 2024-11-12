
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


import warnings


import torch


import random


from typing import Any


from typing import ClassVar


import time


from typing import Callable


import numpy as np


from abc import ABC


from abc import abstractmethod


import itertools


from collections import defaultdict


from functools import partial


import torch.nn as nn


import torch.nn.functional as F


from torch import nn


from torch import optim


from torch.nn.utils.clip_grad import clip_grad_norm_


from matplotlib import pylab


from copy import deepcopy


from typing import Dict


from typing import Tuple


from torch.utils.data import DataLoader


from torch.utils.data import TensorDataset


from torch.distributions import Distribution


from torch.distributions import Normal


import abc


from typing import List


from torch.nn.functional import relu


from torch.nn.functional import softplus


from torch import load


from collections import deque


from typing import TextIO


from torch.utils.tensorboard.writer import SummaryWriter


from typing import Mapping


from torch.utils.data import Dataset


import inspect


import logging


from typing import Optional


from torch.optim.lr_scheduler import ConstantLR


from torch.optim.lr_scheduler import LinearLR


from typing import Literal


from typing import NamedTuple


from typing import Sequence


from typing import TypeVar


from typing import Union


from torch.types import Device


import torch.distributed as dist


from torch.distributed import ReduceOp


from torch.distributions import TanhTransform


from torch.distributions import TransformedDistribution


from torch.distributions import constraints


import torch.backends.cudnn


import torch.types


class EnsembleFC(nn.Module):
    """Ensemble fully connected network.

    A fully connected network with ensemble_size models.

    Args:
        in_features (int): The number of input features.
        out_features (int): The number of output features.
        ensemble_size (int): The number of models in the ensemble.
        weight_decay (float): The decaying factor.
        bias (bool): Whether to use bias.

    Attributes:
        in_features (int): The number of input features.
        out_features (int): The number of output features.
        ensemble_size (int): The number of models in the ensemble.
        weight (nn.Parameter): The weight of the network.
        bias (nn.Parameter): The bias of the network.
    """
    _constants_: 'list[str]'
    in_features: 'int'
    out_features: 'int'
    ensemble_size: 'int'
    weight: 'nn.Parameter'

    def __init__(self, in_features: 'int', out_features: 'int', ensemble_size: 'int', weight_decay: 'float'=0.0, bias: 'bool'=True) ->None:
        """Initialize an instance of fully connected network."""
        super().__init__()
        self._constants_ = ['in_features', 'out_features']
        self.in_features = in_features
        self.out_features = out_features
        self.ensemble_size = ensemble_size
        self.weight = nn.Parameter(torch.Tensor(ensemble_size, in_features, out_features))
        self.weight_decay = weight_decay
        if bias:
            self.bias = nn.Parameter(torch.Tensor(ensemble_size, out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, input_data: 'torch.Tensor') ->torch.Tensor:
        """Forward pass.

        Args:
            input_data (torch.Tensor): The input data.

        Returns:
            The forward output of the network.
        """
        w_times_x = torch.bmm(input_data, self.weight)
        if self.bias is not None:
            w_times_x = torch.add(w_times_x, self.bias[:, None, :])
        return w_times_x


class MultiLayerPerceptron(nn.Sequential):
    """Multi-layer perceptron.

    Args:
        n_units (List[int]): The number of units in each layer.
        activation (nn.Module, optional): The activation function. Defaults to nn.ReLU.
        auto_squeeze (bool, optional): Whether to auto-squeeze the output. Defaults to True.
        output_activation ([type], optional): The output activation function. Defaults to None.
    """

    def __init__(self, n_units, activation=nn.ReLU, auto_squeeze=True, output_activation=None) ->None:
        """Initialize the multi-layer perceptron."""
        layers = []
        for in_features, out_features in zip(n_units[:-1], n_units[1:]):
            if layers:
                layers.append(activation())
            layers.append(nn.Linear(in_features, out_features))
        if output_activation:
            layers.append(output_activation())
        super().__init__(*layers)
        self._n_units = n_units
        self._auto_squeeze = auto_squeeze
        self._activation = [activation]

    def forward(self, *inputs):
        """Forward pass of the MLP.

        Args:
            *inputs: The input tensors.
        """
        inputs = inputs[0] if len(inputs) == 1 else torch.cat(inputs, dim=-1)
        outputs = inputs
        for layer in self:
            outputs = layer(outputs)
        if self._auto_squeeze and outputs.shape[-1] == 1:
            outputs = outputs.squeeze(-1)
        return outputs

    def copy(self):
        """Copy the MLP.

        Returns:
            MultiLayerPerceptron: The copied MLP.
        """
        return MultiLayerPerceptron(self._n_units, self._activation[0], self._auto_squeeze)

    def extra_repr(self):
        """Extra representation of the MLP.

        Returns:
            str: The extra representation.
        """
        return f'activation = {self._activation}, # units = {self._n_units}, squeeze = {self._auto_squeeze}'


class CrabsCore(torch.nn.Module):
    """Core class for CRABS.

    It encapsulates the core process of barrier function.
    For more details, you can refer to the paper: https://arxiv.org/abs/2108.01846

    Args:
        h: The barrier function.
        model: The ensemble model for transition dynamics.
        policy: The policy.
    """

    def __init__(self, h, model: 'EnsembleModel', policy: 'ConstraintActorQCritic', cfgs) ->None:
        """Initialize the CRABS core."""
        super().__init__()
        self.h = h
        self.policy = policy
        self.model = model
        self.init_cfgs(cfgs)

    def init_cfgs(self, cfgs):
        """Initialize the configuration.

        Args:
            cfgs: The configurations.
        """
        self.eps = cfgs.obj.eps
        self.neg_coef = cfgs.obj.neg_coef

    def u(self, states, actions=None):
        """Compute the value of the barrier function.

        Args:
            states (torch.Tensor): The states.
            actions (torch.Tensor, optional): The actions. Defaults to None.
        """
        if actions is None:
            actions = self.policy(states)
        next_states = [self.model.models[idx](states, actions) for idx in self.model.elites]
        all_next_states = torch.stack(next_states)
        all_nh = self.h(all_next_states)
        return all_nh.max(dim=0).values

    def obj_eval(self, s):
        """Short cut for barrier function.

        Args:
            s: The states.

        Returns:
            dict: The results of the barrier function.
        """
        h = self.h(s)
        u = self.u(s)
        eps = self.eps
        obj = u + eps
        mask = (h < 0) & (u + eps > 0)
        return {'h': h, 'u': u, 's': s, 'obj': obj, 'constraint': h, 'mask': mask, 'max_obj': (obj * mask).max(), 'hard_obj': torch.where(h < 0, u + eps, -h - 1000)}


class BasePolicy(abc.ABC):
    """Base class for policy."""

    @abc.abstractmethod
    def get_actions(self, states):
        """Sample the actions.

        Args:
            states (torch.Tensor): The states.
        """


class SafeTanhTransformer(TanhTransform):
    """Safe Tanh Transformer.

    This transformer is used to avoid the error caused by the input of tanh function being too large
    or too small.
    """

    def __call__(self, x: 'torch.Tensor') ->torch.Tensor:
        """Apply the transform to the input."""
        return torch.clamp(torch.tanh(x), min=-0.999999, max=0.999999)

    def _inverse(self, y: 'torch.Tensor') ->torch.Tensor:
        if y.dtype.is_floating_point:
            eps = torch.finfo(y.dtype).eps
        else:
            raise ValueError('Expected floating point type')
        y = y.clamp(min=-1 + eps, max=1 - eps)
        return super()._inverse(y)


class ExplorationPolicy(nn.Module, BasePolicy):
    """Exploration policy for CRABS.

    Args:
        policy (BasePolicy): The policy.
        core (CrabsCore): The CRABS core.
    """

    def __init__(self, policy, core: 'CrabsCore') ->None:
        """Initialize the exploration policy."""
        super().__init__()
        self.policy = policy
        self.crabs = core
        self.last_h = 0
        self.last_u = 0

    @torch.no_grad()
    def forward(self, states: 'torch.Tensor'):
        """Safe exploration policy.

        Certify the safety of the action by the barrier function.

        Args:
            states (torch.Tensor): The states.
        """
        device = states.device
        assert len(states) == 1
        dist = self.policy(states)
        if isinstance(dist, TanhNormal):
            mean, std = dist.mean, dist.stddev
            n = 100
            states = states.repeat([n, 1])
            decay = torch.logspace(0, -3, n, base=10.0, device=device)
            actions = (mean + torch.randn([n, *mean.shape[1:]], device=device) * std * decay[:, None]).tanh()
        else:
            mean = dist
            n = 100
            states = states.repeat([n, 1])
            decay = torch.logspace(0, -3, n, base=10.0, device=device)
            actions = mean + torch.randn([n, *mean.shape[1:]], device=device) * decay[:, None]
        all_u = self.crabs.u(states, actions).detach().cpu().numpy()
        if np.min(all_u) <= 0:
            index = np.min(np.where(all_u <= 0)[0])
            action = actions[index]
        else:
            action = self.crabs.policy(states[0])
        return action[None]

    def step(self, obs: 'torch.Tensor', deterministic: 'bool'=False) ->torch.Tensor:
        """Choose the action based on the observation. used in rollout without gradient.

        Args:
            obs (torch.tensor): The observation from environments.
            deterministic (bool, optional): Whether to use deterministic action. Defaults to False.

        Returns:
            The deterministic action if deterministic is True.
            Action with noise other wise.
        """
        with torch.no_grad():
            return self.forward(obs)

    def get_actions(self, states):
        """Sample the actions.

        Args:
            states (torch.Tensor): The states.

        Returns:
            torch.Tensor: The sampled actions.
        """
        return self(states)


class NetPolicy(nn.Module, BasePolicy):
    """Base class for policy."""

    def get_actions(self, states):
        """Sample the actions.

        Args:
            states (torch.Tensor): The states.

        Returns:
            torch.Tensor: The sampled actions.
        """
        return self(states).sample()


class DetNetPolicy(NetPolicy):
    """Deterministic policy for CRABS."""

    def get_actions(self, states):
        """Get the deterministic actions.

        Args:
            states (torch.Tensor): The states.

        Returns:
            torch.Tensor: The deterministic actions.
        """
        return self(states)


class MeanPolicy(DetNetPolicy):
    """Mean policy for CRABS.

    Args:
        policy (NetPolicy): The policy.
    """

    def __init__(self, policy) ->None:
        """Initialize the mean policy."""
        super().__init__()
        self.policy = policy

    def forward(self, states):
        """Forward pass of the mean policy.

        Args:
            states (torch.Tensor): The states.

        Returns:
            torch.Tensor: The mean of the policy.
        """
        return self.policy(states).mean

    def step(self, obs: 'torch.Tensor', deterministic: 'bool'=False) ->torch.Tensor:
        """Choose the action based on the observation. used in rollout without gradient.

        Args:
            obs (torch.tensor): The observation from environments.
            deterministic (bool, optional): Whether to use deterministic action. Defaults to False.

        Returns:
            The deterministic action if deterministic is True.
            Action with noise other wise.
        """
        with torch.no_grad():
            return self.policy(obs).mean


class AddGaussianNoise(NetPolicy):
    """Add Gaussian noise to the actions.

    Args:
        policy (NetPolicy): The policy.
        mean: The mean of the noise.
        std: The standard deviation of the noise.
    """

    def __init__(self, policy: 'NetPolicy', mean, std) ->None:
        """Initialize the policy with Gaussian noise."""
        super().__init__()
        self.policy = policy
        self.mean = mean
        self.std = std

    def forward(self, states):
        """Forward pass of the policy.

        Args:
            states (torch.Tensor): The states.

        Returns:
            torch.Tensor: The actions.
        """
        actions = self.policy(states)
        if isinstance(actions, TanhNormal):
            return TanhNormal(actions.mean + self.mean, actions.stddev * self.std)
        noises = torch.randn(*actions.shape, device=states.device) * self.std + self.mean
        return actions + noises


class UniformPolicy(NetPolicy):
    """Uniform policy for CRABS.

    Args:
        dim_action (int): The dimension of the action.
    """

    def __init__(self, dim_action) ->None:
        """Initialize the uniform policy."""
        super().__init__()
        self.dim_action = dim_action

    def forward(self, states):
        """Forward pass of the policy.

        Args:
            states (torch.Tensor): The states.

        Returns:
            torch.Tensor: The actions.
        """
        return torch.rand(states.shape[:-1] + (self.dim_action,), device=states.device) * 2 - 1


class Barrier(nn.Module):
    """Barrier function for the environment.

    This is corresponding to the function h(x) in the paper.

    Args:
        net (nn.Module): Neural network that represents the barrier function.
        env_barrier_fn (Callable): Barrier function for the environment.
        s0 (torch.Tensor): Initial state.
    """

    def __init__(self, net, env_barrier_fn, s0, cfgs) ->None:
        """Initialize the barrier function."""
        super().__init__()
        self.net = net
        self.env_barrier_fn = env_barrier_fn
        self.s0 = s0
        self.ell = softplus
        self.ell_coef = cfgs.ell_coef
        self.barrier_coef = cfgs.barrier_coef

    def forward(self, states: 'torch.Tensor') ->torch.Tensor:
        """Forward pass of the barrier function.

        Args:
            states (torch.Tensor): States to evaluate the barrier function.

        Returns:
            torch.Tensor: Barrier function values.
        """
        return self.ell(self.net(states) - self.net(self.s0[None])) * self.ell_coef + self.env_barrier_fn(states) * self.barrier_coef - 1


class SLangevinOptimizer(nn.Module):
    """Stochastic Langevin optimizer for the s*.

    This class is used to optimize the s* in the paper.

    Args:
        core (CrabsCore): Core model for the optimization.
        state_box (StateBox): State box for the optimization.
        cfgs: Configuration for the optimization.
        logger: Logger for the optimization.
    """

    def __init__(self, core: 'CrabsCore', state_box: 'StateBox', device, cfgs, logger) ->None:
        """Initialize the optimizer."""
        super().__init__()
        self.core = core
        self.state_box = state_box
        self._cfgs = cfgs
        self._logger = logger
        self.init_cfgs(cfgs)
        self.temperature = self.temperature.max
        self.z = nn.Parameter(torch.zeros(self.batch_size, *state_box.shape, device=device), requires_grad=True)
        self.tau = nn.Parameter(torch.full([self.batch_size, 1], 0.01), requires_grad=False)
        self.alpha = nn.Parameter(torch.full([self.batch_size], 3.0), requires_grad=False)
        self.opt = torch.optim.Adam([self.z])
        self.max_s = torch.zeros(state_box.shape, device=device)
        self.min_s = torch.zeros(state_box.shape, device=device)
        self.mask = torch.tensor([0], dtype=torch.int64)
        self.n_failure = torch.zeros(self.batch_size, dtype=torch.int64, device=device)
        self.n_resampled = 0
        self.adam = torch.optim.Adam([self.z], betas=(0, 0.999), lr=0.001)
        self.since_last_reset = 0
        self.reinit()

    def init_cfgs(self, cfgs):
        """Initialize the configuration.

        Args:
            cfgs: Configuration for the optimization.
        """
        self.temperature = cfgs.temperature
        self.filter = cfgs.filter
        self.n_steps = cfgs.n_steps
        self.method = cfgs.method
        self.lr = cfgs.lr
        self.batch_size = cfgs.batch_size
        self.extend_region = cfgs.extend_region
        self.barrier_coef = cfgs.barrier_coef
        self.L_neg_coef = cfgs.L_neg_coef
        self.is_resample = cfgs.resample
        self.n_proj_iters = cfgs.n_proj_iters
        self.precond = cfgs.precond

    @property
    def s(self):
        """Decoded state from the state box.

        Returns:
            torch.Tensor: Decoded state.
        """
        return self.state_box.decode(self.z)

    def reinit(self):
        """Reinitialize the optimizer."""
        nn.init.uniform_(self.z, -1.0, 1.0)
        nn.init.constant_(self.tau, 0.01)
        nn.init.constant_(self.alpha, 3.0)
        self.since_last_reset = 0

    def set_temperature(self, p):
        """Set the temperature for the optimizer.

        Args:
            p (float): Temperature parameter.
        """
        max = self.temperature.max
        min = self.temperature.min
        self.temperature = np.exp(np.log(max) * (1 - p) + np.log(min) * p)

    def pdf(self, z):
        """Probability density function for the optimizer.

        Args:
            z (torch.Tensor): State tensor.

        Returns:
            Tuple[torch.Tensor, dict]: Probability density function and the result.
        """
        s = self.state_box.decode(z)
        result = self.core.obj_eval(s)
        return result['hard_obj'] / self.temperature, result

    def project_back(self):
        """Use gradient descent to project s back to the set C_h."""
        for _ in range(self.n_proj_iters):
            with torch.enable_grad():
                h = self.core.h(self.s)
                loss = relu(h - 0.03)
                if (h > 0.03).sum() < 1000:
                    break
                self.adam.zero_grad()
                loss.sum().backward()
                self.adam.step()

    @torch.no_grad()
    def resample(self, f: 'torch.Tensor', idx):
        """Resample the states.

        Args:
            f (torch.Tensor): Probability density function.
            idx: Index of the states to resample.
        """
        if len(idx) == 0:
            return
        new_idx = f.softmax(0).multinomial(len(idx), replacement=True)
        self.z[idx] = self.z[new_idx]
        self.tau[idx] = self.tau[new_idx]
        self.n_failure[idx] = 0
        self.n_resampled += len(idx)

    def step(self):
        """One step of the optimizer."""
        self.since_last_reset += 1
        self.project_back()
        tau = self.tau
        a = self.z
        f_a, a_info = self.pdf(a)
        grad_a = torch.autograd.grad(f_a.sum(), a)[0]
        w = torch.randn_like(a)
        b = a + tau * grad_a + (tau * 2).sqrt() * w
        b = b.detach().requires_grad_()
        f_b, b_info = self.pdf(b)
        grad_b = torch.autograd.grad(f_b.sum(), b)[0]
        (a_info['h'] < 0) & (b_info['h'] > 0)
        with torch.no_grad():
            log_p_a_to_b = -w.norm(dim=-1) ** 2
            log_p_b_to_a = -((a - b - tau * grad_b) ** 2).sum(dim=-1) / tau[:, 0] / 4
            log_ratio = f_b + log_p_b_to_a - (f_a + log_p_a_to_b)
            ratio = log_ratio.clamp(max=0).exp()[:, None]
            sampling = torch.rand_like(ratio) < ratio
            b = torch.where(sampling.squeeze((0, 1))[:, None] & (b_info['h'][:, None].squeeze((0, 1))[:, None] < 0), b, a)
            new_f_b = torch.where(sampling[:, 0], f_b, f_a)
            self.mask = torch.nonzero(new_f_b >= 0)[:, 0]
            if len(self.mask) == 0:
                self.mask = torch.tensor([0], dtype=torch.int64)
            self.z.set_(b)
            self.tau.mul_(self.lr * (ratio.squeeze()[:, None] - 0.574) + 1)
            if self.is_resample:
                self.n_failure[new_f_b >= -100] = 0
                self.n_failure += 1
                self.resample(new_f_b, torch.nonzero(self.n_failure > 1000)[:, 0])
        return {'optimal': a_info['hard_obj'].max().item()}

    @torch.no_grad()
    def debug(self, *, step=0):
        """Debug."""
        result = self.core.obj_eval(self.s)
        h = result['h']
        result['hard_obj'].max().item()
        inside = (result['constraint'] <= 0).sum().item()
        result['mask'].sum().item()
        self.tau.log().mean().exp().item()
        self.tau.max().item()
        h_inside = h.cpu().numpy()
        h_inside = h_inside[np.where(result['constraint'].cpu() <= 0)]
        np.percentile(h_inside, [25, 50, 75]) if len(h_inside) else []
        self.n_resampled = 0
        return {'inside': inside}


class SSampleOptimizer(nn.Module):
    """Sample optimizer for the s*.

    Args:
        obj_eval (Callable): Objective evaluation function.
        state_box (StateBox): State box.
        logger: Logger for the optimizer.
    """

    def __init__(self, obj_eval: 'Callable[[torch.Tensor], dict]', state_box: 'StateBox', logger=None) ->None:
        """Initialize the optimizer."""
        super().__init__()
        self.obj_eval = obj_eval
        self.s = nn.Parameter(torch.randn(100000, *state_box.shape), requires_grad=False)
        self.state_box = state_box
        self._logger = logger

    @torch.no_grad()
    def debug(self, *, step):
        """Debug."""
        self.state_box.fill_(self.s)
        s = self.s
        result = self.obj_eval(s)
        result['hard_obj'].max().item()
        (result['h'] <= 0).sum().item()


class SGradOptimizer(nn.Module):
    """Gradient optimizer for the s*.

    Args:
        obj_eval (Callable): Objective evaluation function.
        state_box (StateBox): State box.
        logger: Logger for the optimizer.
    """

    def __init__(self, obj_eval: 'Callable[[torch.Tensor], dict]', state_box: 'StateBox', logger=None) ->None:
        """Initialize the optimizer."""
        super().__init__()
        self.obj_eval = obj_eval
        self.z = nn.Parameter(torch.randn(10000, *state_box.shape), requires_grad=True)
        self.opt = torch.optim.Adam([self.z], lr=0.001)
        self.state_box = state_box
        self._logger = logger

    @property
    def s(self):
        """Decoded state from the state box.

        Returns:
            torch.Tensor: Decoded state.
        """
        return self.state_box.decode(self.z)

    def step(self):
        """One step of the optimizer.

        Returns:
            torch.Tensor: Loss.
        """
        result = self.obj_eval(self.s)
        obj = result['hard_obj']
        loss = (-obj).mean()
        self.opt.zero_grad()
        loss.mean().backward()
        self.opt.step()
        return loss

    @torch.no_grad()
    def reinit(self):
        """Reinitialize the optimizer."""
        nn.init.uniform_(self.z, -1.0, 1.0)

    def debug(self, *, step):
        """Debug."""
        result = self.obj_eval(self.s)
        hardD = result['hard_obj']
        result['constraint']
        result['obj']
        hardD.argmax()
        hardD.max().item()
        return {'optimal': hardD.max().item()}


class Normalizer(nn.Module):
    """Calculate normalized raw_data from running mean and std.

    References:
        - Title: Updating Formulae and a Pairwise Algorithm for Computing Sample Variances
        - Author: Tony F. Chan, Gene H. Golub, Randall J. LeVeque
        - URL: `Normalizer <http://i.stanford.edu/pub/cstr/reports/cs/tr/79/773/CS-TR-79-773.pdf>`_
    """
    _mean: 'torch.Tensor'
    _sumsq: 'torch.Tensor'
    _var: 'torch.Tensor'
    _std: 'torch.Tensor'
    _count: 'torch.Tensor'
    _clip: 'torch.Tensor'

    def __init__(self, shape: 'tuple[int, ...]', clip: 'float'=1000000.0) ->None:
        """Initialize an instance of :class:`Normalizer`."""
        super().__init__()
        if shape == ():
            self.register_buffer('_mean', torch.tensor(0.0))
            self.register_buffer('_sumsq', torch.tensor(0.0))
            self.register_buffer('_var', torch.tensor(0.0))
            self.register_buffer('_std', torch.tensor(0.0))
            self.register_buffer('_count', torch.tensor(0))
            self.register_buffer('_clip', clip * torch.tensor(1.0))
        else:
            self.register_buffer('_mean', torch.zeros(*shape))
            self.register_buffer('_sumsq', torch.zeros(*shape))
            self.register_buffer('_var', torch.zeros(*shape))
            self.register_buffer('_std', torch.zeros(*shape))
            self.register_buffer('_count', torch.tensor(0))
            self.register_buffer('_clip', clip * torch.ones(*shape))
        self._shape: 'tuple[int, ...]' = shape
        self._first: 'bool' = True

    @property
    def shape(self) ->tuple[int, ...]:
        """Return the shape of the normalize."""
        return self._shape

    @property
    def mean(self) ->torch.Tensor:
        """Return the mean of the normalize."""
        return self._mean

    @property
    def std(self) ->torch.Tensor:
        """Return the std of the normalize."""
        return self._std

    def forward(self, data: 'torch.Tensor') ->torch.Tensor:
        """Normalize the data.

        Args:
            data (torch.Tensor): The raw data to be normalized.

        Returns:
            The normalized data.
        """
        return self.normalize(data)

    def normalize(self, data: 'torch.Tensor') ->torch.Tensor:
        """Normalize the data.

        .. hint::
            - If the data is the first data, the data will be used to initialize the mean and std.
            - If the data is not the first data, the data will be normalized by the mean and std.
            - Update the mean and std by the data.

        Args:
            data (torch.Tensor): The raw data to be normalized.

        Returns:
            The normalized data.
        """
        data = data
        self._push(data)
        if self._count <= 1:
            return data
        output = (data - self._mean) / self._std
        return torch.clamp(output, -self._clip, self._clip)

    def _push(self, raw_data: 'torch.Tensor') ->None:
        """Update the mean and std by the raw_data.

        Args:
            raw_data (torch.Tensor): The raw data to be normalized.
        """
        if raw_data.shape == self._shape:
            raw_data = raw_data.unsqueeze(0)
        assert raw_data.shape[1:] == self._shape, 'data shape must be equal to (batch_size, *shape)'
        if self._first:
            self._mean = torch.mean(raw_data, dim=0)
            self._sumsq = torch.sum((raw_data - self._mean) ** 2, dim=0)
            self._count = torch.tensor(raw_data.shape[0], dtype=self._count.dtype, device=self._count.device)
            self._first = False
        else:
            count_raw = raw_data.shape[0]
            count = self._count + count_raw
            mean_raw = torch.mean(raw_data, dim=0)
            delta = mean_raw - self._mean
            self._mean += delta * count_raw / count
            sumq_raw = torch.sum((raw_data - mean_raw) ** 2, dim=0)
            self._sumsq += sumq_raw + delta ** 2 * self._count * count_raw / count
            self._count = count
        self._var = self._sumsq / (self._count - 1)
        self._std = torch.sqrt(self._var)
        self._std = torch.max(self._std, 0.01 * torch.ones_like(self._std))

    def load_state_dict(self, state_dict: 'Mapping[str, Any]', strict: 'bool'=True, assign: 'bool'=False) ->Any:
        """Load the state_dict to the normalizer.

        Args:
            state_dict (Mapping[str, Any]): The state_dict to be loaded.
            strict (bool, optional): Whether to strictly enforce that the keys in :attr:`state_dict`.
                Defaults to True.

        Returns:
            The loaded normalizer.
        """
        self._first = False
        return super().load_state_dict(state_dict, strict, assign)


class Actor(nn.Module, ABC):
    """An abstract class for actor.

    An actor approximates the policy function that maps observations to actions. Actor is
    parameterized by a neural network that takes observations as input, and outputs the mean and
    standard deviation of the action distribution.

    .. note::
        You can use this class to implement your own actor by inheriting it.

    Args:
        obs_space (OmnisafeSpace): observation space.
        act_space (OmnisafeSpace): action space.
        hidden_sizes (list of int): List of hidden layer sizes.
        activation (Activation, optional): Activation function. Defaults to ``'relu'``.
        weight_initialization_mode (InitFunction, optional): Weight initialization mode. Defaults to
            ``'kaiming_uniform'``.
    """

    def __init__(self, obs_space: 'OmnisafeSpace', act_space: 'OmnisafeSpace', hidden_sizes: 'list[int]', activation: 'Activation'='relu', weight_initialization_mode: 'InitFunction'='kaiming_uniform') ->None:
        """Initialize an instance of :class:`Actor`."""
        nn.Module.__init__(self)
        self._obs_space: 'OmnisafeSpace' = obs_space
        self._act_space: 'OmnisafeSpace' = act_space
        self._weight_initialization_mode: 'InitFunction' = weight_initialization_mode
        self._activation: 'Activation' = activation
        self._hidden_sizes: 'list[int]' = hidden_sizes
        self._after_inference: 'bool' = False
        if isinstance(self._obs_space, spaces.Box) and len(self._obs_space.shape) == 1:
            self._obs_dim: 'int' = self._obs_space.shape[0]
        else:
            raise NotImplementedError
        if isinstance(self._act_space, spaces.Box) and len(self._act_space.shape) == 1:
            self._act_dim: 'int' = self._act_space.shape[0]
        else:
            raise NotImplementedError

    @abstractmethod
    def _distribution(self, obs: 'torch.Tensor') ->Distribution:
        """Return the distribution of action.

        An actor generates a distribution, which is used to sample actions during training. When
        training, the mean and the variance of the distribution are used to calculate the loss. When
        testing, the mean of the distribution is used directly as actions.

        For example, if the action is continuous, the actor can generate a Gaussian distribution.

        .. math::

            p (a | s) = N (\\mu (s), \\sigma (s))

        where :math:`\\mu (s)` and :math:`\\sigma (s)` are the mean and standard deviation of the
        distribution.

        .. warning::
            The distribution is a private method, which is only used to sample actions during
            training. You should not use it directly in your code, instead, you should use the
            public method :meth:`predict` to sample actions.

        Args:
            obs (torch.Tensor): Observation from environments.

        Returns:
            The distribution of action.
        """

    @abstractmethod
    def forward(self, obs: 'torch.Tensor') ->Distribution:
        """Return the distribution of action.

        Args:
            obs (torch.Tensor): Observation from environments.
        """

    @abstractmethod
    def predict(self, obs: 'torch.Tensor', deterministic: 'bool'=False) ->torch.Tensor:
        """Predict deterministic or stochastic action based on observation.

        - ``deterministic`` = ``True`` or ``False``

        When training the actor, one important trick to avoid local minimum is to use stochastic
        actions, which can simply be achieved by sampling actions from the distribution (set
        ``deterministic=False``).

        When testing the actor, we want to know the actual action that the agent will take, so we
        should use deterministic actions (set ``deterministic=True``).

        .. math::

            L = -\\underset{s \\sim p(s)}{\\mathbb{E}}[ \\log p (a | s) A^R (s, a) ]

        where :math:`p (s)` is the distribution of observation, :math:`p (a | s)` is the
        distribution of action, and :math:`\\log p (a | s)` is the log probability of action under
        the distribution., and :math:`A^R (s, a)` is the advantage function.

        Args:
            obs (torch.Tensor): Observation from environments.
            deterministic (bool, optional): Whether to predict deterministic action. Defaults to False.
        """

    @abstractmethod
    def log_prob(self, act: 'torch.Tensor') ->torch.Tensor:
        """Return the log probability of action under the distribution.

        :meth:`log_prob` only can be called after calling :meth:`predict` or :meth:`forward`.

        Args:
            act (torch.Tensor): The action.

        Returns:
            The log probability of action under the distribution.
        """


class GaussianActor(Actor, ABC):
    """An abstract class for normal distribution actor.

    A NormalActor inherits from Actor and use Normal distribution to approximate the policy function.

    .. note::
        You can use this class to implement your own actor by inheriting it.
    """

    @property
    @abstractmethod
    def std(self) ->float:
        """Get the standard deviation of the normal distribution."""

    @std.setter
    @abstractmethod
    def std(self, std: 'float') ->None:
        """Set the standard deviation of the normal distribution."""


def initialize_layer(init_function: 'InitFunction', layer: 'nn.Linear') ->None:
    """Initialize the layer with the given initialization function.

    The ``init_function`` can be chosen from: ``kaiming_uniform``, ``xavier_normal``, ``glorot``,
    ``xavier_uniform``, ``orthogonal``.

    Args:
        init_function (InitFunction): The initialization function.
        layer (nn.Linear): The layer to be initialized.
    """
    if init_function == 'kaiming_uniform':
        nn.init.kaiming_uniform_(layer.weight, a=np.sqrt(5))
    elif init_function == 'xavier_normal':
        nn.init.xavier_normal_(layer.weight)
    elif init_function in ['glorot', 'xavier_uniform']:
        nn.init.xavier_uniform_(layer.weight)
    elif init_function == 'orthogonal':
        nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
    else:
        raise TypeError(f'Invalid initialization function: {init_function}')


def build_mlp_network(sizes: 'list[int]', activation: 'Activation', output_activation: 'Activation'='identity', weight_initialization_mode: 'InitFunction'='kaiming_uniform') ->nn.Module:
    """Build the MLP network.

    Examples:
        >>> build_mlp_network([64, 64, 64], 'relu', 'tanh')
        Sequential(
            (0): Linear(in_features=64, out_features=64, bias=True)
            (1): ReLU()
            (2): Linear(in_features=64, out_features=64, bias=True)
            (3): ReLU()
            (4): Linear(in_features=64, out_features=64, bias=True)
            (5): Tanh()
        )

    Args:
        sizes (list of int): The sizes of the layers.
        activation (Activation): The activation function.
        output_activation (Activation, optional): The output activation function. Defaults to
            ``identity``.
        weight_initialization_mode (InitFunction, optional): Weight initialization mode. Defaults to
            ``'kaiming_uniform'``.

    Returns:
        The MLP network.
    """
    activation_fn = get_activation(activation)
    output_activation_fn = get_activation(output_activation)
    layers = []
    for j in range(len(sizes) - 1):
        act_fn = activation_fn if j < len(sizes) - 2 else output_activation_fn
        affine_layer = nn.Linear(sizes[j], sizes[j + 1])
        initialize_layer(weight_initialization_mode, affine_layer)
        layers += [affine_layer, act_fn()]
    return nn.Sequential(*layers)


class GaussianLearningActor(GaussianActor):
    """Implementation of GaussianLearningActor.

    GaussianLearningActor is a Gaussian actor with a learnable standard deviation. It is used in
    on-policy algorithms such as ``PPO``, ``TRPO`` and so on.

    Args:
        obs_space (OmnisafeSpace): Observation space.
        act_space (OmnisafeSpace): Action space.
        hidden_sizes (list of int): List of hidden layer sizes.
        activation (Activation, optional): Activation function. Defaults to ``'relu'``.
        weight_initialization_mode (InitFunction, optional): Weight initialization mode. Defaults to
            ``'kaiming_uniform'``.
    """
    _current_dist: 'Normal'

    def __init__(self, obs_space: 'OmnisafeSpace', act_space: 'OmnisafeSpace', hidden_sizes: 'list[int]', activation: 'Activation'='relu', weight_initialization_mode: 'InitFunction'='kaiming_uniform') ->None:
        """Initialize an instance of :class:`GaussianLearningActor`."""
        super().__init__(obs_space, act_space, hidden_sizes, activation, weight_initialization_mode)
        self.mean: 'nn.Module' = build_mlp_network(sizes=[self._obs_dim, *self._hidden_sizes, self._act_dim], activation=activation, weight_initialization_mode=weight_initialization_mode)
        self.log_std: 'nn.Parameter' = nn.Parameter(torch.zeros(self._act_dim), requires_grad=True)

    def _distribution(self, obs: 'torch.Tensor') ->Normal:
        """Get the distribution of the actor.

        .. warning::
            This method is not supposed to be called by users. You should call :meth:`forward`
            instead.

        Args:
            obs (torch.Tensor): Observation from environments.

        Returns:
            The normal distribution of the mean and standard deviation from the actor.
        """
        mean = self.mean(obs)
        std = torch.exp(self.log_std)
        return Normal(mean, std)

    def predict(self, obs: 'torch.Tensor', deterministic: 'bool'=False) ->torch.Tensor:
        """Predict the action given observation.

        The predicted action depends on the ``deterministic`` flag.

        - If ``deterministic`` is ``True``, the predicted action is the mean of the distribution.
        - If ``deterministic`` is ``False``, the predicted action is sampled from the distribution.

        Args:
            obs (torch.Tensor): Observation from environments.
            deterministic (bool, optional): Whether to use deterministic policy. Defaults to False.

        Returns:
            The mean of the distribution if deterministic is True, otherwise the sampled action.
        """
        self._current_dist = self._distribution(obs)
        self._after_inference = True
        if deterministic:
            return self._current_dist.mean
        return self._current_dist.rsample()

    def forward(self, obs: 'torch.Tensor') ->Distribution:
        """Forward method.

        Args:
            obs (torch.Tensor): Observation from environments.

        Returns:
            The current distribution.
        """
        self._current_dist = self._distribution(obs)
        self._after_inference = True
        return self._current_dist

    def log_prob(self, act: 'torch.Tensor') ->torch.Tensor:
        """Compute the log probability of the action given the current distribution.

        .. warning::
            You must call :meth:`forward` or :meth:`predict` before calling this method.

        Args:
            act (torch.Tensor): Action from :meth:`predict` or :meth:`forward` .

        Returns:
            Log probability of the action.
        """
        assert self._after_inference, 'log_prob() should be called after predict() or forward()'
        self._after_inference = False
        return self._current_dist.log_prob(act).sum(axis=-1)

    @property
    def std(self) ->float:
        """Standard deviation of the distribution."""
        return torch.exp(self.log_std).mean().item()

    @std.setter
    def std(self, std: 'float') ->None:
        device = self.log_std.device
        self.log_std.data.fill_(torch.log(torch.tensor(std, device=device)))


class MLPActor(Actor):
    """Implementation of MLPActor.

    MLPActor is a Gaussian actor with a learnable mean value. It is used in off-policy algorithms
    such as ``DDPG``, ``TD3`` and so on.

    Args:
        obs_space (OmnisafeSpace): Observation space.
        act_space (OmnisafeSpace): Action space.
        hidden_sizes (list of int): List of hidden layer sizes.
        activation (Activation, optional): Activation function. Defaults to ``'relu'``.
        output_activation (Activation, optional): Output activation function. Defaults to ``'tanh'``.
        weight_initialization_mode (InitFunction, optional): Weight initialization mode. Defaults to
            ``'kaiming_uniform'``.
    """

    def __init__(self, obs_space: 'OmnisafeSpace', act_space: 'OmnisafeSpace', hidden_sizes: 'list[int]', activation: 'Activation'='relu', output_activation: 'Activation'='tanh', weight_initialization_mode: 'InitFunction'='kaiming_uniform') ->None:
        """Initialize an instance of :class:`MLPActor`."""
        super().__init__(obs_space, act_space, hidden_sizes, activation, weight_initialization_mode)
        self.net: 'torch.nn.Module' = build_mlp_network(sizes=[self._obs_dim, *self._hidden_sizes, self._act_dim], activation=activation, output_activation=output_activation, weight_initialization_mode=weight_initialization_mode)
        self._noise: 'float' = 0.1

    def predict(self, obs: 'torch.Tensor', deterministic: 'bool'=True) ->torch.Tensor:
        """Predict the action given observation.

        The predicted action depends on the ``deterministic`` flag.

        - If ``deterministic`` is ``True``, the predicted action is the mean of the distribution.
        - If ``deterministic`` is ``False``, the predicted action is sampled from the distribution.

        Args:
            obs (torch.Tensor): Observation from environments.
            deterministic (bool, optional): Whether to use deterministic policy. Defaults to True.
        """
        action = self.net(obs)
        if deterministic:
            return action
        with torch.no_grad():
            noise = torch.normal(0, self._noise * torch.ones_like(action))
            return torch.clamp(action + noise, -1, 1)

    @property
    def noise(self) ->float:
        """Noise of the action."""
        return self._noise

    @noise.setter
    def noise(self, noise: 'float') ->None:
        """Set the action noise."""
        assert noise >= 0, 'Noise should be non-negative.'
        self._noise = noise

    def _distribution(self, obs: 'torch.Tensor') ->Distribution:
        raise NotImplementedError

    def forward(self, obs: 'torch.Tensor') ->Distribution:
        """Forward method implementation.

        Args:
            obs (torch.Tensor): Observation from environments.

        Returns:
            The distribution of the action.
        """
        return self._distribution(obs)

    def log_prob(self, act: 'torch.Tensor') ->torch.Tensor:
        """Log probability of the action.

        Args:
            act (torch.Tensor): Action from :meth:`predict` or :meth:`forward`  tensor.

        Raises:
            NotImplementedError: The method is not implemented.
        """
        raise NotImplementedError

    @property
    def std(self) ->float:
        """Standard deviation of the distribution."""
        return self._noise


class VAE(Actor):
    """Class for VAE.

    VAE is a variational auto-encoder. It is used in offline algorithms such as ``BCQ`` and so on.

    Args:
        obs_space (OmnisafeSpace): Observation space.
        act_space (OmnisafeSpace): Action space.
        hidden_sizes (list): List of hidden layer sizes.
        latent_dim (Optional[int]): Latent dimension, if None, latent_dim = act_dim * 2.
        activation (Activation): Activation function.
        weight_initialization_mode (InitFunction, optional): Weight initialization mode. Defaults to
            ``'kaiming_uniform'``.
    """

    def __init__(self, obs_space: 'OmnisafeSpace', act_space: 'OmnisafeSpace', hidden_sizes: 'List[int]', activation: 'Activation'='relu', weight_initialization_mode: 'InitFunction'='kaiming_uniform') ->None:
        """Initialize an instance of :class:`VAE`."""
        super().__init__(obs_space, act_space, hidden_sizes, activation, weight_initialization_mode)
        self._latent_dim = self._act_dim * 2
        self._encoder = build_mlp_network(sizes=[self._obs_dim + self._act_dim, *hidden_sizes, self._latent_dim * 2], activation=activation, weight_initialization_mode=weight_initialization_mode)
        self._decoder = build_mlp_network(sizes=[self._obs_dim + self._latent_dim, *hidden_sizes, self._act_dim], activation=activation, weight_initialization_mode=weight_initialization_mode)
        self.add_module('encoder', self._encoder)
        self.add_module('decoder', self._decoder)

    def encode(self, obs: 'torch.Tensor', act: 'torch.Tensor') ->Normal:
        """Encode observation to latent distribution.

        Args:
            obs (torch.Tensor): Observation.
            act (torch.Tensor): Action from :meth:`predict` or :meth:`forward` .

        Returns:
            Normal: Latent distribution.
        """
        latent = self._encoder(torch.cat([obs, act], dim=-1))
        mean, log_std = torch.chunk(latent, 2, dim=-1)
        log_std = torch.clamp(log_std, min=-20, max=2)
        return Normal(mean, log_std.exp())

    def decode(self, obs: 'torch.Tensor', latent: 'Optional[torch.Tensor]'=None) ->torch.Tensor:
        """Decode latent vector to action.

        When ``latent`` is None, sample latent vector from standard normal distribution.

        Args:
            obs (torch.Tensor): Observation.
            latent (Optional[torch.Tensor], optional): Latent vector. Defaults to None.

        Returns:
            torch.Tensor: Action.
        """
        if latent is None:
            latent = Normal(0, 1).sample([obs.shape[0], self._latent_dim])
        return self._decoder(torch.cat([obs, latent], dim=-1))

    def loss(self, obs: 'torch.Tensor', act: 'torch.Tensor') ->Tuple[torch.Tensor, torch.Tensor]:
        """Compute loss for VAE.

        Args:
            obs (torch.Tensor): Observation.
            act (torch.Tensor): Action from :meth:`predict` or :meth:`forward` .
        """
        dist = self.encode(obs, act)
        latent = dist.rsample()
        pred_act = self.decode(obs, latent)
        recon_loss = nn.functional.mse_loss(pred_act, act)
        kl_loss = torch.distributions.kl.kl_divergence(dist, Normal(0, 1)).mean()
        return recon_loss, kl_loss

    def _distribution(self, obs: 'torch.Tensor') ->Distribution:
        raise NotImplementedError

    def forward(self, obs: 'torch.Tensor') ->Distribution:
        """Forward is not used in this method, it is just for compatibility."""
        raise NotImplementedError

    def predict(self, obs: 'torch.Tensor', deterministic: 'bool'=False) ->torch.Tensor:
        """Predict the action given observation.

        deterministic if not used in VAE model. VAE actor's default behavior is stochastic,
        sampling from the latent standard normal distribution.

        Args:
            obs (torch.Tensor): Observation from environments.
            deterministic (bool, optional): Whether to use deterministic policy. Defaults to False.

        Returns:
            torch.Tensor: Predicted action.
        """
        return self.decode(obs)

    def log_prob(self, act: 'torch.Tensor') ->torch.Tensor:
        """log_prob is not used in this method, it is just for compatibility."""
        raise NotImplementedError


class PerturbationActor(Actor):
    """Class for Perturbation Actor.

    Perturbation Actor is used in offline algorithms such as ``BCQ`` and so on.
    Perturbation Actor is a combination of VAE and a perturbation network,
    algorithm BCQ uses the perturbation network to perturb the action predicted by VAE,
    which trained like behavior cloning.

    Args:
        obs_space (OmnisafeSpace): Observation space.
        act_space (OmnisafeSpace): Action space.
        hidden_sizes (list): List of hidden layer sizes.
        latent_dim (Optional[int]): Latent dimension, if None, latent_dim = act_dim * 2.
        activation (Activation): Activation function.
        weight_initialization_mode (InitFunction, optional): Weight initialization mode. Defaults to
            ``'kaiming_uniform'``.
    """

    def __init__(self, obs_space: 'OmnisafeSpace', act_space: 'OmnisafeSpace', hidden_sizes: 'List[int]', activation: 'Activation'='relu', weight_initialization_mode: 'InitFunction'='kaiming_uniform') ->None:
        """Initialize an instance of :class:`PerturbationActor`."""
        super().__init__(obs_space, act_space, hidden_sizes, activation, weight_initialization_mode)
        self.vae = VAE(obs_space, act_space, hidden_sizes, activation, weight_initialization_mode)
        self.perturbation = build_mlp_network(sizes=[self._obs_dim + self._act_dim, *hidden_sizes, self._act_dim], activation=activation, output_activation='tanh', weight_initialization_mode=weight_initialization_mode)
        self._phi = torch.nn.Parameter(torch.tensor(0.05))

    @property
    def phi(self) ->float:
        """Return phi, which is the maximum perturbation."""
        return self._phi.item()

    @phi.setter
    def phi(self, phi: 'float') ->None:
        """Set phi. which is the maximum perturbation."""
        self._phi = torch.nn.Parameter(torch.tensor(phi, device=self._phi.device))

    def predict(self, obs: 'torch.Tensor', deterministic: 'bool'=False) ->torch.Tensor:
        """Predict action from observation.

        deterministic is not used in this method, it is just for compatibility.

        Args:
            obs (torch.Tensor): Observation.
            deterministic (bool, optional): Whether to return deterministic action. Defaults to False.

        Returns:
            torch.Tensor: Action.
        """
        act = self.vae.predict(obs, deterministic)
        perturbation = self.perturbation(torch.cat([obs, act], dim=-1))
        return act + self._phi * perturbation

    def _distribution(self, obs: 'torch.Tensor') ->Distribution:
        raise NotImplementedError

    def forward(self, obs: 'torch.Tensor') ->Distribution:
        """Forward is not used in this method, it is just for compatibility."""
        raise NotImplementedError

    def log_prob(self, act: 'torch.Tensor') ->torch.Tensor:
        """log_prob is not used in this method, it is just for compatibility."""
        raise NotImplementedError


class ActorBuilder:
    """Class for building actor networks.

    Args:
        obs_space (OmnisafeSpace): Observation space.
        act_space (OmnisafeSpace): Action space.
        hidden_sizes (list of int): List of hidden layer sizes.
        activation (Activation, optional): Activation function. Defaults to ``'relu'``.
        weight_initialization_mode (InitFunction, optional): Weight initialization mode. Defaults to
            ``'kaiming_uniform'``.
    """

    def __init__(self, obs_space: 'OmnisafeSpace', act_space: 'OmnisafeSpace', hidden_sizes: 'list[int]', activation: 'Activation'='relu', weight_initialization_mode: 'InitFunction'='kaiming_uniform') ->None:
        """Initialize an instance of :class:`ActorBuilder`."""
        self._obs_space: 'OmnisafeSpace' = obs_space
        self._act_space: 'OmnisafeSpace' = act_space
        self._weight_initialization_mode: 'InitFunction' = weight_initialization_mode
        self._activation: 'Activation' = activation
        self._hidden_sizes: 'list[int]' = hidden_sizes

    def build_actor(self, actor_type: 'ActorType') ->Actor:
        """Build actor network.

        Currently, we support the following actor types:
            - ``gaussian_learning``: Gaussian actor with learnable standard deviation parameters.
            - ``gaussian_sac``: Gaussian actor with learnable standard deviation network.
            - ``mlp``: Multi-layer perceptron actor, used in ``DDPG`` and ``TD3``.

        Args:
            actor_type (ActorType): Type of actor network, e.g. ``gaussian_learning``.

        Returns:
            Actor network, ranging from GaussianLearningActor, GaussianSACActor to MLPActor.

        Raises:
            NotImplementedError: If the actor type is not implemented.
        """
        if actor_type == 'gaussian_learning':
            return GaussianLearningActor(self._obs_space, self._act_space, self._hidden_sizes, activation=self._activation, weight_initialization_mode=self._weight_initialization_mode)
        if actor_type == 'gaussian_sac':
            return GaussianSACActor(self._obs_space, self._act_space, self._hidden_sizes, activation=self._activation, weight_initialization_mode=self._weight_initialization_mode)
        if actor_type == 'mlp':
            return MLPActor(self._obs_space, self._act_space, self._hidden_sizes, activation=self._activation, weight_initialization_mode=self._weight_initialization_mode)
        if actor_type == 'vae':
            return VAE(self._obs_space, self._act_space, self._hidden_sizes, activation=self._activation, weight_initialization_mode=self._weight_initialization_mode)
        if actor_type == 'perturbation':
            return PerturbationActor(self._obs_space, self._act_space, self._hidden_sizes, activation=self._activation, weight_initialization_mode=self._weight_initialization_mode)
        raise NotImplementedError(f'Actor type {actor_type} is not implemented! Available actor types are: gaussian_learning, gaussian_sac, mlp, vae, perturbation.')


class Critic(nn.Module, ABC):
    """An abstract class for critic.

    A critic approximates the value function that maps observations to values. Critic is
    parameterized by a neural network that takes observations as input, (Q critic also takes actions
    as input) and outputs the value estimated.

    .. note::
        OmniSafe provides two types of critic:
        Q critic (Input = ``observation`` + ``action`` , Output = ``value``),
        and V critic (Input = ``observation`` , Output = ``value``).
        You can also use this class to implement your own actor by inheriting it.

    Args:
        obs_space (OmnisafeSpace): Observation space.
        act_space (OmnisafeSpace): Action space.
        hidden_sizes (list of int): List of hidden layer sizes.
        activation (Activation, optional): Activation function. Defaults to ``'relu'``.
        weight_initialization_mode (InitFunction, optional): Weight initialization mode. Defaults to
            ``'kaiming_uniform'``.
        num_critics (int, optional): Number of critics. Defaults to 1.
        use_obs_encoder (bool, optional): Whether to use observation encoder, only used in q critic.
            Defaults to False.
    """

    def __init__(self, obs_space: 'OmnisafeSpace', act_space: 'OmnisafeSpace', hidden_sizes: 'list[int]', activation: 'Activation'='relu', weight_initialization_mode: 'InitFunction'='kaiming_uniform', num_critics: 'int'=1, use_obs_encoder: 'bool'=False) ->None:
        """Initialize an instance of :class:`Critic`."""
        nn.Module.__init__(self)
        self._obs_space: 'OmnisafeSpace' = obs_space
        self._act_space: 'OmnisafeSpace' = act_space
        self._weight_initialization_mode: 'InitFunction' = weight_initialization_mode
        self._activation: 'Activation' = activation
        self._hidden_sizes: 'list[int]' = hidden_sizes
        self._num_critics: 'int' = num_critics
        self._use_obs_encoder: 'bool' = use_obs_encoder
        if isinstance(self._obs_space, spaces.Box) and len(self._obs_space.shape) == 1:
            self._obs_dim = self._obs_space.shape[0]
        else:
            raise NotImplementedError
        if isinstance(self._act_space, spaces.Box) and len(self._act_space.shape) == 1:
            self._act_dim = self._act_space.shape[0]
        else:
            raise NotImplementedError


class QCritic(Critic):
    """Implementation of Q Critic.

    A Q-function approximator that uses a multi-layer perceptron (MLP) to map observation-action
    pairs to Q-values. This class is an inherit class of :class:`Critic`. You can design your own
    Q-function approximator by inheriting this class or :class:`Critic`.

    The Q critic network has two modes:

    .. hint::
        - ``use_obs_encoder = False``: The input of the network is the concatenation of the
            observation and action.
        - ``use_obs_encoder = True``: The input of the network is the concatenation of the output of
            the observation encoder and action.

    For example, in :class:`DDPG`, the action is not directly concatenated with the observation, but
    is concatenated with the output of the observation encoder.

    .. note::
        The Q critic network contains multiple critics, and the output of the network :meth`forward`
        is a list of Q-values. If you want to get the single Q-value of a specific critic, you need
        to use the index to get it.

    Args:
        obs_space (OmnisafeSpace): observation space.
        act_space (OmnisafeSpace): action space.
        hidden_sizes (list of int): List of hidden layer sizes.
        activation (Activation, optional): Activation function. Defaults to ``'relu'``.
        weight_initialization_mode (InitFunction, optional): Weight initialization mode. Defaults to
            ``'kaiming_uniform'``.
        num_critics (int, optional): Number of critics. Defaults to 1.
        use_obs_encoder (bool, optional): Whether to use observation encoder, only used in q critic.
            Defaults to False.
    """

    def __init__(self, obs_space: 'OmnisafeSpace', act_space: 'OmnisafeSpace', hidden_sizes: 'list[int]', activation: 'Activation'='relu', weight_initialization_mode: 'InitFunction'='kaiming_uniform', num_critics: 'int'=1, use_obs_encoder: 'bool'=False) ->None:
        """Initialize an instance of :class:`QCritic`."""
        super().__init__(obs_space, act_space, hidden_sizes, activation, weight_initialization_mode, num_critics, use_obs_encoder)
        self.net_lst: 'list[nn.Sequential]' = []
        for idx in range(self._num_critics):
            if self._use_obs_encoder:
                obs_encoder = build_mlp_network([self._obs_dim, hidden_sizes[0]], activation=activation, output_activation=activation, weight_initialization_mode=weight_initialization_mode)
                net = build_mlp_network([hidden_sizes[0] + self._act_dim] + hidden_sizes[1:] + [1], activation=activation, weight_initialization_mode=weight_initialization_mode)
                critic = nn.Sequential(obs_encoder, net)
            else:
                net = build_mlp_network([self._obs_dim + self._act_dim, *hidden_sizes, 1], activation=activation, weight_initialization_mode=weight_initialization_mode)
                critic = nn.Sequential(net)
            self.net_lst.append(critic)
            self.add_module(f'critic_{idx}', critic)

    def forward(self, obs: 'torch.Tensor', act: 'torch.Tensor') ->list[torch.Tensor]:
        """Forward function.

        As a multi-critic network, the output of the network is a list of Q-values. If you want to
        use it as a single-critic network, you only need to set the ``num_critics`` parameter to 1
        when initializing the network, and then use the index 0 to get the Q-value.

        Args:
            obs (torch.Tensor): Observation from environments.
            act (torch.Tensor): Action from actor .

        Returns:
            A list of Q critic values of action and observation pair.
        """
        res = []
        for critic in self.net_lst:
            if self._use_obs_encoder:
                obs_encode = critic[0](obs)
                res.append(torch.squeeze(critic[1](torch.cat([obs_encode, act], dim=-1)), -1))
            else:
                res.append(torch.squeeze(critic(torch.cat([obs, act], dim=-1)), -1))
        return res


class VCritic(Critic):
    """Implementation of VCritic.

    A V-function approximator that uses a multi-layer perceptron (MLP) to map observations to V-values.
    This class is an inherit class of :class:`Critic`.
    You can design your own V-function approximator by inheriting this class or :class:`Critic`.

    Args:
        obs_dim (int): Observation dimension.
        act_dim (int): Action dimension.
        hidden_sizes (list of int): List of hidden layer sizes.
        activation (Activation, optional): Activation function. Defaults to ``'relu'``.
        weight_initialization_mode (InitFunction, optional): Weight initialization mode. Defaults to
            ``'kaiming_uniform'``.
        num_critics (int, optional): Number of critics. Defaults to 1.
    """

    def __init__(self, obs_space: 'OmnisafeSpace', act_space: 'OmnisafeSpace', hidden_sizes: 'list[int]', activation: 'Activation'='relu', weight_initialization_mode: 'InitFunction'='kaiming_uniform', num_critics: 'int'=1) ->None:
        """Initialize an instance of :class:`VCritic`."""
        super().__init__(obs_space, act_space, hidden_sizes, activation, weight_initialization_mode, num_critics, use_obs_encoder=False)
        self.net_lst: 'list[nn.Module]'
        self.net_lst = []
        for idx in range(self._num_critics):
            net = build_mlp_network(sizes=[self._obs_dim, *self._hidden_sizes, 1], activation=self._activation, weight_initialization_mode=self._weight_initialization_mode)
            self.net_lst.append(net)
            self.add_module(f'critic_{idx}', net)

    def forward(self, obs: 'torch.Tensor') ->list[torch.Tensor]:
        """Forward function.

        Specifically, V function approximator maps observations to V-values.

        Args:
            obs (torch.Tensor): Observations from environments.

        Returns:
            The V critic value of observation.
        """
        res = []
        for critic in self.net_lst:
            res.append(torch.squeeze(critic(obs), -1))
        return res


class CriticBuilder:
    """Implementation of CriticBuilder.

    .. note::
        A :class:`CriticBuilder` is a class for building a critic network. In OmniSafe, instead of
        building the critic network directly, we build it by integrating various types of critic
        networks into the :class:`CriticBuilder`. The advantage of this is that each type of critic
        has a uniform way of passing parameters. This makes it easy for users to use existing
        critics, and also facilitates the extension of new critic types.

    Args:
        obs_space (OmnisafeSpace): Observation space.
        act_space (OmnisafeSpace): Action space.
        hidden_sizes (list of int): List of hidden layer sizes.
        activation (Activation, optional): Activation function. Defaults to ``'relu'``.
        weight_initialization_mode (InitFunction, optional): Weight initialization mode. Defaults to
            ``'kaiming_uniform'``.
        num_critics (int, optional): Number of critics. Defaults to 1.
        use_obs_encoder (bool, optional): Whether to use observation encoder, only used in q critic.
            Defaults to False.
    """

    def __init__(self, obs_space: 'OmnisafeSpace', act_space: 'OmnisafeSpace', hidden_sizes: 'list[int]', activation: 'Activation'='relu', weight_initialization_mode: 'InitFunction'='kaiming_uniform', num_critics: 'int'=1, use_obs_encoder: 'bool'=False) ->None:
        """Initialize an instance of :class:`CriticBuilder`."""
        self._obs_space: 'OmnisafeSpace' = obs_space
        self._act_space: 'OmnisafeSpace' = act_space
        self._weight_initialization_mode: 'InitFunction' = weight_initialization_mode
        self._activation: 'Activation' = activation
        self._hidden_sizes: 'list[int]' = hidden_sizes
        self._num_critics: 'int' = num_critics
        self._use_obs_encoder: 'bool' = use_obs_encoder

    def build_critic(self, critic_type: 'CriticType') ->Critic:
        """Build critic.

        Currently, we support two types of critics: ``q`` and ``v``.
        If you want to add a new critic type, you can simply add it here.

        Args:
            critic_type (str): Critic type.

        Returns:
            An instance of V-Critic or Q-Critic

        Raises:
            NotImplementedError: If the critic type is not ``q`` or ``v``.
        """
        if critic_type == 'q':
            return QCritic(obs_space=self._obs_space, act_space=self._act_space, hidden_sizes=self._hidden_sizes, activation=self._activation, weight_initialization_mode=self._weight_initialization_mode, num_critics=self._num_critics, use_obs_encoder=self._use_obs_encoder)
        if critic_type == 'v':
            return VCritic(obs_space=self._obs_space, act_space=self._act_space, hidden_sizes=self._hidden_sizes, activation=self._activation, weight_initialization_mode=self._weight_initialization_mode, num_critics=self._num_critics)
        raise NotImplementedError(f'critic_type "{critic_type}" is not implemented.Available critic types are: "q", "v".')


class Schedule(ABC):
    """Schedule for a value based on the step."""

    @abstractmethod
    def value(self, time: 'float') ->float:
        """Value at time t."""


def _linear_interpolation(left: 'float', right: 'float', alpha: 'float') ->float:
    return left + alpha * (right - left)


class PiecewiseSchedule(Schedule):
    """Piece-wise schedule for a value based on the step, from OpenAI baselines.

    Args:
        endpoints (list[tuple[int, float]]): List of pairs `(time, value)` meaning that schedule
            will output `value` when `t==time`. All the values for time must be sorted in an
            increasing order. When t is between two times, e.g. `(time_a, value_a)` and
            `(time_b, value_b)`, such that `time_a <= t < time_b` then value outputs is interpolated
            linearly between `value_a` and `value_b`.
        outside_value (int or float): Value to use if `t` is before the first time in `endpoints` or
            after the last one.
    """

    def __init__(self, endpoints: 'list[tuple[int, float]]', outside_value: 'float') ->None:
        """Initialize an instance of :class:`PiecewiseSchedule`."""
        idxes = [e[0] for e in endpoints]
        assert idxes == sorted(idxes)
        self._interpolation: 'Callable[[float, float, float], float]' = _linear_interpolation
        self._outside_value: 'float' = outside_value
        self._endpoints: 'list[tuple[int, float]]' = endpoints

    def value(self, time: 'float') ->float:
        """Value at time t.

        Args:
            time (int or float): Current time step.

        Returns:
            The interpolation value at time t or outside_value if t is before the first time in
            endpoints of after the last one.

        Raises:
            AssertionError: If the time is not in the endpoints.
        """
        for (left_t, left), (right_t, right) in zip(self._endpoints[:-1], self._endpoints[1:]):
            if left_t <= time < right_t:
                alpha = float(time - left_t) / (right_t - left_t)
                return self._interpolation(left, right, alpha)
        assert self._outside_value is not None
        return self._outside_value


class ActorCritic(nn.Module):
    """Class for ActorCritic.

    In OmniSafe, we combine the actor and critic into one this class.

    +-----------------+-----------------------------------------------+
    | Model           | Description                                   |
    +=================+===============================================+
    | Actor           | Input is observation. Output is action.       |
    +-----------------+-----------------------------------------------+
    | Reward V Critic | Input is observation. Output is reward value. |
    +-----------------+-----------------------------------------------+

    Args:
        obs_space (OmnisafeSpace): The observation space.
        act_space (OmnisafeSpace): The action space.
        model_cfgs (ModelConfig): The model configurations.
        epochs (int): The number of epochs.

    Attributes:
        actor (Actor): The actor network.
        reward_critic (Critic): The critic network.
        std_schedule (Schedule): The schedule for the standard deviation of the Gaussian distribution.
    """
    std_schedule: 'Schedule'

    def __init__(self, obs_space: 'OmnisafeSpace', act_space: 'OmnisafeSpace', model_cfgs: 'ModelConfig', epochs: 'int') ->None:
        """Initialize an instance of :class:`ActorCritic`."""
        super().__init__()
        self.actor: 'Actor' = ActorBuilder(obs_space=obs_space, act_space=act_space, hidden_sizes=model_cfgs.actor.hidden_sizes, activation=model_cfgs.actor.activation, weight_initialization_mode=model_cfgs.weight_initialization_mode).build_actor(actor_type=model_cfgs.actor_type)
        self.reward_critic: 'Critic' = CriticBuilder(obs_space=obs_space, act_space=act_space, hidden_sizes=model_cfgs.critic.hidden_sizes, activation=model_cfgs.critic.activation, weight_initialization_mode=model_cfgs.weight_initialization_mode, num_critics=1, use_obs_encoder=False).build_critic(critic_type='v')
        self.add_module('actor', self.actor)
        self.add_module('reward_critic', self.reward_critic)
        if model_cfgs.actor.lr is not None:
            self.actor_optimizer: 'optim.Optimizer'
            self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=model_cfgs.actor.lr)
        if model_cfgs.critic.lr is not None:
            self.reward_critic_optimizer: 'optim.Optimizer' = optim.Adam(self.reward_critic.parameters(), lr=model_cfgs.critic.lr)
        if model_cfgs.actor.lr is not None:
            self.actor_scheduler: 'LinearLR | ConstantLR'
            if model_cfgs.linear_lr_decay:
                self.actor_scheduler = LinearLR(self.actor_optimizer, start_factor=1.0, end_factor=0.0, total_iters=epochs)
            else:
                self.actor_scheduler = ConstantLR(self.actor_optimizer, factor=1.0, total_iters=epochs)

    def step(self, obs: 'torch.Tensor', deterministic: 'bool'=False) ->tuple[torch.Tensor, ...]:
        """Choose the action based on the observation. used in rollout without gradient.

        Args:
            obs (torch.tensor): The observation from environments.
            deterministic (bool, optional): Whether to use deterministic action. Defaults to False.

        Returns:
            action: The deterministic action if ``deterministic`` is True, otherwise the action with
                Gaussian noise.
            value_r: The reward value of the observation.
            log_prob: The log probability of the action.
        """
        with torch.no_grad():
            value_r = self.reward_critic(obs)
            act = self.actor.predict(obs, deterministic=deterministic)
            log_prob = self.actor.log_prob(act)
        return act, value_r[0], log_prob

    def forward(self, obs: 'torch.Tensor', deterministic: 'bool'=False) ->tuple[torch.Tensor, ...]:
        """Choose the action based on the observation. used in training with gradient.

        Args:
            obs (torch.tensor): The observation from environments.
            deterministic (bool, optional): Whether to use deterministic action. Defaults to False.

        Returns:
            action: The deterministic action if ``deterministic`` is True, otherwise the action with
                Gaussian noise.
            value_r: The reward value of the observation.
            log_prob: The log probability of the action.
        """
        return self.step(obs, deterministic=deterministic)

    def set_annealing(self, epochs: 'list[int]', std: 'list[float]') ->None:
        """Set the annealing mode for the actor.

        Args:
            epochs (list of int): The list of epochs.
            std (list of float): The list of standard deviation.
        """
        assert isinstance(self.actor, GaussianLearningActor), 'Only GaussianLearningActor support annealing.'
        self.std_schedule = PiecewiseSchedule(endpoints=list(zip(epochs, std)), outside_value=std[-1])

    def annealing(self, epoch: 'int') ->None:
        """Set the annealing mode for the actor.

        Args:
            epoch (int): The current epoch.
        """
        assert isinstance(self.actor, GaussianLearningActor), 'Only GaussianLearningActor support annealing.'
        self.actor.std = self.std_schedule.value(epoch)


class ActorQCritic(nn.Module):
    """Class for ActorQCritic.

    In OmniSafe, we combine the actor and critic into one this class.

    +-----------------+---------------------------------------------------+
    | Model           | Description                                       |
    +=================+===================================================+
    | Actor           | Input is observation. Output is action.           |
    +-----------------+---------------------------------------------------+
    | Reward Q Critic | Input is obs-action pair. Output is reward value. |
    +-----------------+---------------------------------------------------+

    Args:
        obs_space (OmnisafeSpace): The observation space.
        act_space (OmnisafeSpace): The action space.
        model_cfgs (ModelConfig): The model configurations.
        epochs (int): The number of epochs.

    Attributes:
        actor (Actor): The actor network.
        target_actor (Actor): The target actor network.
        reward_critic (Critic): The critic network.
        target_reward_critic (Critic): The target critic network.
        actor_optimizer (Optimizer): The optimizer for the actor network.
        reward_critic_optimizer (Optimizer): The optimizer for the critic network.
        std_schedule (Schedule): The schedule for the standard deviation of the Gaussian distribution.
    """

    def __init__(self, obs_space: 'OmnisafeSpace', act_space: 'OmnisafeSpace', model_cfgs: 'ModelConfig', epochs: 'int') ->None:
        """Initialize an instance of :class:`ActorQCritic`."""
        super().__init__()
        self.actor: 'Actor' = ActorBuilder(obs_space=obs_space, act_space=act_space, hidden_sizes=model_cfgs.actor.hidden_sizes, activation=model_cfgs.actor.activation, weight_initialization_mode=model_cfgs.weight_initialization_mode).build_actor(actor_type=model_cfgs.actor_type)
        self.reward_critic: 'Critic' = CriticBuilder(obs_space=obs_space, act_space=act_space, hidden_sizes=model_cfgs.critic.hidden_sizes, activation=model_cfgs.critic.activation, weight_initialization_mode=model_cfgs.weight_initialization_mode, num_critics=model_cfgs.critic.num_critics, use_obs_encoder=False).build_critic(critic_type='q')
        self.target_reward_critic: 'Critic' = deepcopy(self.reward_critic)
        for param in self.target_reward_critic.parameters():
            param.requires_grad = False
        self.target_actor: 'Actor' = deepcopy(self.actor)
        for param in self.target_actor.parameters():
            param.requires_grad = False
        self.add_module('actor', self.actor)
        self.add_module('reward_critic', self.reward_critic)
        if model_cfgs.actor.lr is not None:
            self.actor_optimizer: 'optim.Optimizer'
            self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=model_cfgs.actor.lr)
        if model_cfgs.critic.lr is not None:
            self.reward_critic_optimizer: 'optim.Optimizer'
            self.reward_critic_optimizer = optim.Adam(self.reward_critic.parameters(), lr=model_cfgs.critic.lr)
        self.actor_scheduler: 'LinearLR | ConstantLR'
        if model_cfgs.linear_lr_decay:
            self.actor_scheduler = LinearLR(self.actor_optimizer, start_factor=1.0, end_factor=0.0, total_iters=epochs)
        else:
            self.actor_scheduler = ConstantLR(self.actor_optimizer, factor=1.0, total_iters=epochs)

    def step(self, obs: 'torch.Tensor', deterministic: 'bool'=False) ->torch.Tensor:
        """Choose the action based on the observation. used in rollout without gradient.

        Args:
            obs (torch.tensor): The observation from environments.
            deterministic (bool, optional): Whether to use deterministic action. Defaults to False.

        Returns:
            The deterministic action if deterministic is True.
            Action with noise other wise.
        """
        with torch.no_grad():
            return self.actor.predict(obs, deterministic=deterministic)

    def forward(self, obs: 'torch.Tensor', deterministic: 'bool'=False) ->torch.Tensor:
        """Choose the action based on the observation. used in training with gradient.

        Args:
            obs (torch.tensor): The observation from environments.
            deterministic (bool, optional): Whether to use deterministic action. Defaults to False.

        Returns:
            The deterministic action if deterministic is True.
            Action with noise other wise.
        """
        return self.step(obs, deterministic=deterministic)

    def polyak_update(self, tau: 'float') ->None:
        """Update the target network with polyak averaging.

        Args:
            tau (float): The polyak averaging factor.
        """
        for param, target_param in zip(self.reward_critic.parameters(), self.target_reward_critic.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        for param, target_param in zip(self.actor.parameters(), self.target_actor.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


class ConstraintActorCritic(ActorCritic):
    """ConstraintActorCritic is a wrapper around ActorCritic that adds a cost critic to the model.

    In OmniSafe, we combine the actor and critic into one this class.

    +-----------------+-----------------------------------------------+
    | Model           | Description                                   |
    +=================+===============================================+
    | Actor           | Input is observation. Output is action.       |
    +-----------------+-----------------------------------------------+
    | Reward V Critic | Input is observation. Output is reward value. |
    +-----------------+-----------------------------------------------+
    | Cost V Critic   | Input is observation. Output is cost value.   |
    +-----------------+-----------------------------------------------+

    Args:
        obs_space (OmnisafeSpace): The observation space.
        act_space (OmnisafeSpace): The action space.
        model_cfgs (ModelConfig): The model configurations.
        epochs (int): The number of epochs.

    Attributes:
        actor (Actor): The actor network.
        reward_critic (Critic): The critic network.
        cost_critic (Critic): The critic network.
        std_schedule (Schedule): The schedule for the standard deviation of the Gaussian distribution.
    """

    def __init__(self, obs_space: 'OmnisafeSpace', act_space: 'OmnisafeSpace', model_cfgs: 'ModelConfig', epochs: 'int') ->None:
        """Initialize an instance of :class:`ConstraintActorCritic`."""
        super().__init__(obs_space, act_space, model_cfgs, epochs)
        self.cost_critic: 'Critic' = CriticBuilder(obs_space=obs_space, act_space=act_space, hidden_sizes=model_cfgs.critic.hidden_sizes, activation=model_cfgs.critic.activation, weight_initialization_mode=model_cfgs.weight_initialization_mode, num_critics=1, use_obs_encoder=False).build_critic('v')
        self.add_module('cost_critic', self.cost_critic)
        if model_cfgs.critic.lr is not None:
            self.cost_critic_optimizer: 'optim.Optimizer'
            self.cost_critic_optimizer = optim.Adam(self.cost_critic.parameters(), lr=model_cfgs.critic.lr)

    def step(self, obs: 'torch.Tensor', deterministic: 'bool'=False) ->tuple[torch.Tensor, ...]:
        """Choose action based on observation.

        Args:
            obs (torch.Tensor): Observation from environments.
            deterministic (bool, optional): Whether to use deterministic policy. Defaults to False.

        Returns:
            action: The deterministic action if ``deterministic`` is True, otherwise the action with
                Gaussian noise.
            value_r: The reward value of the observation.
            value_c: The cost value of the observation.
            log_prob: The log probability of the action.
        """
        with torch.no_grad():
            value_r = self.reward_critic(obs)
            value_c = self.cost_critic(obs)
            action = self.actor.predict(obs, deterministic=deterministic)
            log_prob = self.actor.log_prob(action)
        return action, value_r[0], value_c[0], log_prob

    def forward(self, obs: 'torch.Tensor', deterministic: 'bool'=False) ->tuple[torch.Tensor, ...]:
        """Choose action based on observation.

        Args:
            obs (torch.Tensor): Observation from environments.
            deterministic (bool, optional): Whether to use deterministic policy. Defaults to False.

        Returns:
            action: The deterministic action if ``deterministic`` is True, otherwise the action with
                Gaussian noise.
            value_r: The reward value of the observation.
            value_c: The cost value of the observation.
            log_prob: The log probability of the action.
        """
        return self.step(obs, deterministic=deterministic)


class ConstraintActorQCritic(ActorQCritic):
    """ConstraintActorQCritic is a wrapper around ActorCritic that adds a cost critic to the model.

    In OmniSafe, we combine the actor and critic into one this class.

    +-----------------+---------------------------------------------------+
    | Model           | Description                                       |
    +=================+===================================================+
    | Actor           | Input is observation. Output is action.           |
    +-----------------+---------------------------------------------------+
    | Reward Q Critic | Input is obs-action pair, Output is reward value. |
    +-----------------+---------------------------------------------------+
    | Cost Q Critic   | Input is obs-action pair. Output is cost value.   |
    +-----------------+---------------------------------------------------+

    Args:
        obs_space (OmnisafeSpace): The observation space.
        act_space (OmnisafeSpace): The action space.
        model_cfgs (ModelConfig): The model configurations.
        epochs (int): The number of epochs.

    Attributes:
        actor (Actor): The actor network.
        target_actor (Actor): The target actor network.
        reward_critic (Critic): The critic network.
        target_reward_critic (Critic): The target critic network.
        cost_critic (Critic): The critic network.
        target_cost_critic (Critic): The target critic network.
        actor_optimizer (Optimizer): The optimizer for the actor network.
        reward_critic_optimizer (Optimizer): The optimizer for the critic network.
        std_schedule (Schedule): The schedule for the standard deviation of the Gaussian distribution.
    """

    def __init__(self, obs_space: 'OmnisafeSpace', act_space: 'OmnisafeSpace', model_cfgs: 'ModelConfig', epochs: 'int') ->None:
        """Initialize an instance of :class:`ConstraintActorQCritic`."""
        super().__init__(obs_space, act_space, model_cfgs, epochs)
        self.cost_critic: 'Critic' = CriticBuilder(obs_space=obs_space, act_space=act_space, hidden_sizes=model_cfgs.critic.hidden_sizes, activation=model_cfgs.critic.activation, weight_initialization_mode=model_cfgs.weight_initialization_mode, num_critics=1, use_obs_encoder=False).build_critic('q')
        self.target_cost_critic: 'Critic' = deepcopy(self.cost_critic)
        for param in self.target_cost_critic.parameters():
            param.requires_grad = False
        self.add_module('cost_critic', self.cost_critic)
        if model_cfgs.critic.lr is not None:
            self.cost_critic_optimizer: 'optim.Optimizer'
            self.cost_critic_optimizer = optim.Adam(self.cost_critic.parameters(), lr=model_cfgs.critic.lr)

    def polyak_update(self, tau: 'float') ->None:
        """Update the target network with polyak averaging.

        Args:
            tau (float): The polyak averaging factor.
        """
        super().polyak_update(tau)
        for target_param, param in zip(self.target_cost_critic.parameters(), self.cost_critic.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


class ObsEncoder(nn.Module):
    """Implementation of observation encoder.

    Observation encoder is used to encode observation into a latent vector.
    It is similar to the QCritic, but the output dimension is not limited to 1.
    DICE-based algorithms often use the network like this to encode observation.

    Args:
        obs_space (OmnisafeSpace): observation space.
        act_space (OmnisafeSpace): action space.
        hidden_sizes (list of int): List of hidden layer sizes.
        activation (Activation, optional): Activation function. Defaults to ``'relu'``.
        weight_initialization_mode (InitFunction, optional): Weight initialization mode. Defaults to
            ``'kaiming_uniform'``.
        out_dim (int, optional): Output dimension. Defaults to 1.
    """

    def __init__(self, obs_space: 'OmnisafeSpace', act_space: 'OmnisafeSpace', hidden_sizes: 'list[int]', activation: 'Activation'='relu', weight_initialization_mode: 'InitFunction'='kaiming_uniform', out_dim: 'int'=1) ->None:
        """Initialize an instance of :class:`ObsEncoder`."""
        nn.Module.__init__(self)
        if isinstance(obs_space, spaces.Box) and len(obs_space.shape) == 1:
            self._obs_dim = obs_space.shape[0]
        else:
            raise NotImplementedError
        if isinstance(act_space, spaces.Box) and len(act_space.shape) == 1:
            self._act_dim = act_space.shape[0]
        else:
            raise NotImplementedError
        self._out_dim = out_dim
        self.weight_initialization_mode = weight_initialization_mode
        self.activation = activation
        self.hidden_sizes = hidden_sizes
        self.net = build_mlp_network([self._obs_dim, *list(hidden_sizes), self._out_dim], activation=activation, weight_initialization_mode=weight_initialization_mode)

    def forward(self, obs: 'torch.Tensor') ->torch.Tensor:
        """Forward function.

        When ``out_dim`` is 1, the output is squeezed to remove the last dimension.

        Args:
            obs (torch.Tensor): Observation.
        """
        if self._out_dim == 1:
            return self.net(obs).squeeze(-1)
        return self.net(obs)


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (EnsembleFC,
     lambda: ([], {'in_features': 4, 'out_features': 4, 'ensemble_size': 4}),
     lambda: ([torch.rand([4, 4, 4])], {})),
    (MeanPolicy,
     lambda: ([], {'policy': torch.nn.ReLU()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (UniformPolicy,
     lambda: ([], {'dim_action': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
]

