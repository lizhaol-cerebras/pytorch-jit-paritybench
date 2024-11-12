
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


import torch.utils.data


from torch import nn


from torch import optim


from torch.distributions.normal import Normal


from torch.nn import functional as F


from torchvision import datasets


from torchvision import transforms


import pandas as pd


import random


import torch.nn as nn


import numpy as np


from torch.testing import assert_close


from torch.nn import init


import warnings


import math


from torch.distributions.multivariate_normal import MultivariateNormal


import torch.nn.init as init


import torch.nn.functional as F


import collections.abc as container_abcs


from itertools import repeat


class VAE(nn.Module):

    def __init__(self):
        super().__init__()
        self.encode = nn.Sequential(nn.Linear(img_dim ** 2, 512), nn.ReLU(True), nn.Linear(512, 256), nn.ReLU(True))
        self.f1 = nn.Linear(256, args.latent_size)
        self.f2 = nn.Linear(256, args.latent_size)
        self.decode = nn.Sequential(nn.Linear(args.latent_size, 256), nn.ReLU(True), nn.Linear(256, 512), nn.ReLU(True), nn.Linear(512, img_dim ** 2))

    def forward(self, x):
        mu, log_var = self.f1(self.encode(x.view(x.size(0) * x.size(1), img_dim ** 2))), self.f2(self.encode(x.view(x.size(0) * x.size(1), img_dim ** 2)))
        std = torch.exp(0.5 * log_var)
        norm_scale = torch.randn_like(std)
        z_ = mu + norm_scale * std
        q0 = Normal(mu, torch.exp(0.5 * log_var))
        p = Normal(0.0, 1.0)
        z_ = z_.view(z_.size(0), args.latent_size)
        zD = self.decode(z_)
        out = torch.sigmoid(zD)
        return out, mu, log_var


class SimpleFlowModel(nn.Module):

    def __init__(self, flows):
        super().__init__()
        self.flows = nn.ModuleList(flows)

    def forward(self, z):
        ld = 0.0
        for flow in self.flows:
            z, ld_ = flow(z)
            ld += ld_
        return z, ld


class FlowVAE(nn.Module):

    def __init__(self, flows):
        super().__init__()
        self.encode = nn.Sequential(nn.Linear(img_dim ** 2, 512), nn.ReLU(True), nn.Linear(512, 256), nn.ReLU(True))
        self.f1 = nn.Linear(256, args.latent_size)
        self.f2 = nn.Linear(256, args.latent_size)
        self.decode = nn.Sequential(nn.Linear(args.latent_size, 256), nn.ReLU(True), nn.Linear(256, 512), nn.ReLU(True), nn.Linear(512, img_dim ** 2))
        self.flows = flows

    def forward(self, x):
        mu, log_var = self.f1(self.encode(x.view(x.size(0) * x.size(1), img_dim ** 2))), self.f2(self.encode(x.view(x.size(0) * x.size(1), img_dim ** 2)))
        std = torch.exp(0.5 * log_var)
        norm_scale = torch.randn_like(std)
        z_0 = mu + norm_scale * std
        z_, log_det = self.flows(z_0)
        z_ = z_.squeeze()
        q0 = Normal(mu, torch.exp(0.5 * log_var))
        p = Normal(0.0, 1.0)
        kld = -torch.sum(p.log_prob(z_), -1) + torch.sum(q0.log_prob(z_0), -1) - log_det.view(-1)
        self.test_params = [torch.mean(-torch.sum(p.log_prob(z_), -1)), torch.mean(torch.sum(q0.log_prob(z_0), -1)), torch.mean(log_det.view(-1)), torch.mean(kld)]
        z_ = z_.view(z_.size(0), args.latent_size)
        zD = self.decode(z_)
        out = torch.sigmoid(zD)
        return out, kld


class NormalizingFlow(nn.Module):
    """
    Normalizing Flow model to approximate target distribution
    """

    def __init__(self, q0, flows, p=None):
        """Constructor

        Args:
          q0: Base distribution
          flows: List of flows
          p: Target distribution
        """
        super().__init__()
        self.q0 = q0
        self.flows = nn.ModuleList(flows)
        self.p = p

    def forward(self, z):
        """Transforms latent variable z to the flow variable x

        Args:
          z: Batch in the latent space

        Returns:
          Batch in the space of the target distribution
        """
        for flow in self.flows:
            z, _ = flow(z)
        return z

    def forward_and_log_det(self, z):
        """Transforms latent variable z to the flow variable x and
        computes log determinant of the Jacobian

        Args:
          z: Batch in the latent space

        Returns:
          Batch in the space of the target distribution,
          log determinant of the Jacobian
        """
        log_det = torch.zeros(len(z), device=z.device)
        for flow in self.flows:
            z, log_d = flow(z)
            log_det += log_d
        return z, log_det

    def inverse(self, x):
        """Transforms flow variable x to the latent variable z

        Args:
          x: Batch in the space of the target distribution

        Returns:
          Batch in the latent space
        """
        for i in range(len(self.flows) - 1, -1, -1):
            x, _ = self.flows[i].inverse(x)
        return x

    def inverse_and_log_det(self, x):
        """Transforms flow variable x to the latent variable z and
        computes log determinant of the Jacobian

        Args:
          x: Batch in the space of the target distribution

        Returns:
          Batch in the latent space, log determinant of the
          Jacobian
        """
        log_det = torch.zeros(len(x), device=x.device)
        for i in range(len(self.flows) - 1, -1, -1):
            x, log_d = self.flows[i].inverse(x)
            log_det += log_d
        return x, log_det

    def forward_kld(self, x):
        """Estimates forward KL divergence, see [arXiv 1912.02762](https://arxiv.org/abs/1912.02762)

        Args:
          x: Batch sampled from target distribution

        Returns:
          Estimate of forward KL divergence averaged over batch
        """
        log_q = torch.zeros(len(x), device=x.device)
        z = x
        for i in range(len(self.flows) - 1, -1, -1):
            z, log_det = self.flows[i].inverse(z)
            log_q += log_det
        log_q += self.q0.log_prob(z)
        return -torch.mean(log_q)

    def reverse_kld(self, num_samples=1, beta=1.0, score_fn=True):
        """Estimates reverse KL divergence, see [arXiv 1912.02762](https://arxiv.org/abs/1912.02762)

        Args:
          num_samples: Number of samples to draw from base distribution
          beta: Annealing parameter, see [arXiv 1505.05770](https://arxiv.org/abs/1505.05770)
          score_fn: Flag whether to include score function in gradient, see [arXiv 1703.09194](https://arxiv.org/abs/1703.09194)

        Returns:
          Estimate of the reverse KL divergence averaged over latent samples
        """
        z, log_q_ = self.q0(num_samples)
        log_q = torch.zeros_like(log_q_)
        log_q += log_q_
        for flow in self.flows:
            z, log_det = flow(z)
            log_q -= log_det
        if not score_fn:
            z_ = z
            log_q = torch.zeros(len(z_), device=z_.device)
            utils.set_requires_grad(self, False)
            for i in range(len(self.flows) - 1, -1, -1):
                z_, log_det = self.flows[i].inverse(z_)
                log_q += log_det
            log_q += self.q0.log_prob(z_)
            utils.set_requires_grad(self, True)
        log_p = self.p.log_prob(z)
        return torch.mean(log_q) - beta * torch.mean(log_p)

    def reverse_alpha_div(self, num_samples=1, alpha=1, dreg=False):
        """Alpha divergence when sampling from q

        Args:
          num_samples: Number of samples to draw
          dreg: Flag whether to use Double Reparametrized Gradient estimator, see [arXiv 1810.04152](https://arxiv.org/abs/1810.04152)

        Returns:
          Alpha divergence
        """
        z, log_q = self.q0(num_samples)
        for flow in self.flows:
            z, log_det = flow(z)
            log_q -= log_det
        log_p = self.p.log_prob(z)
        if dreg:
            w_const = torch.exp(log_p - log_q).detach()
            z_ = z
            log_q = torch.zeros(len(z_), device=z_.device)
            utils.set_requires_grad(self, False)
            for i in range(len(self.flows) - 1, -1, -1):
                z_, log_det = self.flows[i].inverse(z_)
                log_q += log_det
            log_q += self.q0.log_prob(z_)
            utils.set_requires_grad(self, True)
            w = torch.exp(log_p - log_q)
            w_alpha = w_const ** alpha
            w_alpha = w_alpha / torch.mean(w_alpha)
            weights = (1 - alpha) * w_alpha + alpha * w_alpha ** 2
            loss = -alpha * torch.mean(weights * torch.log(w))
        else:
            loss = np.sign(alpha - 1) * torch.logsumexp(alpha * (log_p - log_q), 0)
        return loss

    def sample(self, num_samples=1):
        """Samples from flow-based approximate distribution

        Args:
          num_samples: Number of samples to draw

        Returns:
          Samples, log probability
        """
        z, log_q = self.q0(num_samples)
        for flow in self.flows:
            z, log_det = flow(z)
            log_q -= log_det
        return z, log_q

    def log_prob(self, x):
        """Get log probability for batch

        Args:
          x: Batch

        Returns:
          log probability
        """
        log_q = torch.zeros(len(x), dtype=x.dtype, device=x.device)
        z = x
        for i in range(len(self.flows) - 1, -1, -1):
            z, log_det = self.flows[i].inverse(z)
            log_q += log_det
        log_q += self.q0.log_prob(z)
        return log_q

    def save(self, path):
        """Save state dict of model

        Args:
          path: Path including filename where to save model
        """
        torch.save(self.state_dict(), path)

    def load(self, path):
        """Load model from state dict

        Args:
          path: Path including filename where to load model from
        """
        self.load_state_dict(torch.load(path))


class ConditionalNormalizingFlow(NormalizingFlow):
    """
    Conditional normalizing flow model, providing condition,
    which is also called context, to both the base distribution
    and the flow layers
    """

    def forward(self, z, context=None):
        """Transforms latent variable z to the flow variable x

        Args:
          z: Batch in the latent space
          context: Batch of conditions/context

        Returns:
          Batch in the space of the target distribution
        """
        for flow in self.flows:
            z, _ = flow(z, context=context)
        return z

    def forward_and_log_det(self, z, context=None):
        """Transforms latent variable z to the flow variable x and
        computes log determinant of the Jacobian

        Args:
          z: Batch in the latent space
          context: Batch of conditions/context

        Returns:
          Batch in the space of the target distribution,
          log determinant of the Jacobian
        """
        log_det = torch.zeros(len(z), device=z.device)
        for flow in self.flows:
            z, log_d = flow(z, context=context)
            log_det += log_d
        return z, log_det

    def inverse(self, x, context=None):
        """Transforms flow variable x to the latent variable z

        Args:
          x: Batch in the space of the target distribution
          context: Batch of conditions/context

        Returns:
          Batch in the latent space
        """
        for i in range(len(self.flows) - 1, -1, -1):
            x, _ = self.flows[i].inverse(x, context=context)
        return x

    def inverse_and_log_det(self, x, context=None):
        """Transforms flow variable x to the latent variable z and
        computes log determinant of the Jacobian

        Args:
          x: Batch in the space of the target distribution
          context: Batch of conditions/context

        Returns:
          Batch in the latent space, log determinant of the
          Jacobian
        """
        log_det = torch.zeros(len(x), device=x.device)
        for i in range(len(self.flows) - 1, -1, -1):
            x, log_d = self.flows[i].inverse(x, context=context)
            log_det += log_d
        return x, log_det

    def sample(self, num_samples=1, context=None):
        """Samples from flow-based approximate distribution

        Args:
          num_samples: Number of samples to draw
          context: Batch of conditions/context

        Returns:
          Samples, log probability
        """
        z, log_q = self.q0(num_samples, context=context)
        for flow in self.flows:
            z, log_det = flow(z, context=context)
            log_q -= log_det
        return z, log_q

    def log_prob(self, x, context=None):
        """Get log probability for batch

        Args:
          x: Batch
          context: Batch of conditions/context

        Returns:
          log probability
        """
        log_q = torch.zeros(len(x), dtype=x.dtype, device=x.device)
        z = x
        for i in range(len(self.flows) - 1, -1, -1):
            z, log_det = self.flows[i].inverse(z, context=context)
            log_q += log_det
        log_q += self.q0.log_prob(z, context=context)
        return log_q

    def forward_kld(self, x, context=None):
        """Estimates forward KL divergence, see [arXiv 1912.02762](https://arxiv.org/abs/1912.02762)

        Args:
          x: Batch sampled from target distribution
          context: Batch of conditions/context

        Returns:
          Estimate of forward KL divergence averaged over batch
        """
        log_q = torch.zeros(len(x), device=x.device)
        z = x
        for i in range(len(self.flows) - 1, -1, -1):
            z, log_det = self.flows[i].inverse(z, context=context)
            log_q += log_det
        log_q += self.q0.log_prob(z, context=context)
        return -torch.mean(log_q)

    def reverse_kld(self, num_samples=1, context=None, beta=1.0, score_fn=True):
        """Estimates reverse KL divergence, see [arXiv 1912.02762](https://arxiv.org/abs/1912.02762)

        Args:
          num_samples: Number of samples to draw from base distribution
          context: Batch of conditions/context
          beta: Annealing parameter, see [arXiv 1505.05770](https://arxiv.org/abs/1505.05770)
          score_fn: Flag whether to include score function in gradient, see [arXiv 1703.09194](https://arxiv.org/abs/1703.09194)

        Returns:
          Estimate of the reverse KL divergence averaged over latent samples
        """
        z, log_q_ = self.q0(num_samples, context=context)
        log_q = torch.zeros_like(log_q_)
        log_q += log_q_
        for flow in self.flows:
            z, log_det = flow(z, context=context)
            log_q -= log_det
        if not score_fn:
            z_ = z
            log_q = torch.zeros(len(z_), device=z_.device)
            utils.set_requires_grad(self, False)
            for i in range(len(self.flows) - 1, -1, -1):
                z_, log_det = self.flows[i].inverse(z_, context=context)
                log_q += log_det
            log_q += self.q0.log_prob(z_, context=context)
            utils.set_requires_grad(self, True)
        log_p = self.p.log_prob(z, context=context)
        return torch.mean(log_q) - beta * torch.mean(log_p)


class ClassCondFlow(nn.Module):
    """
    Class conditional normalizing Flow model, providing the
    class to be conditioned on only to the base distribution,
    as done e.g. in [Glow](https://arxiv.org/abs/1807.03039)
    """

    def __init__(self, q0, flows):
        """Constructor

        Args:
          q0: Base distribution
          flows: List of flows
        """
        super().__init__()
        self.q0 = q0
        self.flows = nn.ModuleList(flows)

    def forward_kld(self, x, y):
        """Estimates forward KL divergence, see [arXiv 1912.02762](https://arxiv.org/abs/1912.02762)

        Args:
          x: Batch sampled from target distribution

        Returns:
          Estimate of forward KL divergence averaged over batch
        """
        log_q = torch.zeros(len(x), dtype=x.dtype, device=x.device)
        z = x
        for i in range(len(self.flows) - 1, -1, -1):
            z, log_det = self.flows[i].inverse(z)
            log_q += log_det
        log_q += self.q0.log_prob(z, y)
        return -torch.mean(log_q)

    def sample(self, num_samples=1, y=None):
        """Samples from flow-based approximate distribution

        Args:
          num_samples: Number of samples to draw
          y: Classes to sample from, will be sampled uniformly if None

        Returns:
          Samples, log probability
        """
        z, log_q = self.q0(num_samples, y)
        for flow in self.flows:
            z, log_det = flow(z)
            log_q -= log_det
        return z, log_q

    def log_prob(self, x, y):
        """Get log probability for batch

        Args:
          x: Batch
          y: Classes of x

        Returns:
          log probability
        """
        log_q = torch.zeros(len(x), dtype=x.dtype, device=x.device)
        z = x
        for i in range(len(self.flows) - 1, -1, -1):
            z, log_det = self.flows[i].inverse(z)
            log_q += log_det
        log_q += self.q0.log_prob(z, y)
        return log_q

    def save(self, path):
        """Save state dict of model

        Args:
         param path: Path including filename where to save model
        """
        torch.save(self.state_dict(), path)

    def load(self, path):
        """Load model from state dict

        Args:
          path: Path including filename where to load model from
        """
        self.load_state_dict(torch.load(path))


class MultiscaleFlow(nn.Module):
    """
    Normalizing Flow model with multiscale architecture, see RealNVP or Glow paper
    """

    def __init__(self, q0, flows, merges, transform=None, class_cond=True):
        """Constructor

        Args:

          q0: List of base distribution
          flows: List of flows for each level
          merges: List of merge/split operations (forward pass must do merge)
          transform: Initial transformation of inputs
          class_cond: Flag, indicated whether model has class conditional
        base distributions
        """
        super().__init__()
        self.q0 = nn.ModuleList(q0)
        self.num_levels = len(self.q0)
        self.flows = torch.nn.ModuleList([nn.ModuleList(flow) for flow in flows])
        self.merges = torch.nn.ModuleList(merges)
        self.transform = transform
        self.class_cond = class_cond

    def forward_kld(self, x, y=None):
        """Estimates forward KL divergence, see [arXiv 1912.02762](https://arxiv.org/abs/1912.02762)

        Args:
          x: Batch sampled from target distribution
          y: Batch of classes to condition on, if applicable

        Returns:
          Estimate of forward KL divergence averaged over batch
        """
        return -torch.mean(self.log_prob(x, y))

    def forward(self, x, y=None):
        """Get negative log-likelihood for maximum likelihood training

        Args:
          x: Batch of data
          y: Batch of classes to condition on, if applicable

        Returns:
            Negative log-likelihood of the batch
        """
        return -self.log_prob(x, y)

    def forward_and_log_det(self, z):
        """Get observed variable x from list of latent variables z

        Args:
            z: List of latent variables

        Returns:
            Observed variable x, log determinant of Jacobian
        """
        log_det = torch.zeros(len(z[0]), dtype=z[0].dtype, device=z[0].device)
        for i in range(len(self.q0)):
            if i == 0:
                z_ = z[0]
            else:
                z_, log_det_ = self.merges[i - 1]([z_, z[i]])
                log_det += log_det_
            for flow in self.flows[i]:
                z_, log_det_ = flow(z_)
                log_det += log_det_
        if self.transform is not None:
            z_, log_det_ = self.transform(z_)
            log_det += log_det_
        return z_, log_det

    def inverse_and_log_det(self, x):
        """Get latent variable z from observed variable x

        Args:
            x: Observed variable

        Returns:
            List of latent variables z, log determinant of Jacobian
        """
        log_det = torch.zeros(len(x), dtype=x.dtype, device=x.device)
        if self.transform is not None:
            x, log_det_ = self.transform.inverse(x)
            log_det += log_det_
        z = [None] * len(self.q0)
        for i in range(len(self.q0) - 1, -1, -1):
            for flow in reversed(self.flows[i]):
                x, log_det_ = flow.inverse(x)
                log_det += log_det_
            if i == 0:
                z[i] = x
            else:
                [x, z[i]], log_det_ = self.merges[i - 1].inverse(x)
                log_det += log_det_
        return z, log_det

    def sample(self, num_samples=1, y=None, temperature=None):
        """Samples from flow-based approximate distribution

        Args:
          num_samples: Number of samples to draw
          y: Classes to sample from, will be sampled uniformly if None
          temperature: Temperature parameter for temp annealed sampling

        Returns:
          Samples, log probability
        """
        if temperature is not None:
            self.set_temperature(temperature)
        for i in range(len(self.q0)):
            if self.class_cond:
                z_, log_q_ = self.q0[i](num_samples, y)
            else:
                z_, log_q_ = self.q0[i](num_samples)
            if i == 0:
                log_q = log_q_
                z = z_
            else:
                log_q += log_q_
                z, log_det = self.merges[i - 1]([z, z_])
                log_q -= log_det
            for flow in self.flows[i]:
                z, log_det = flow(z)
                log_q -= log_det
        if self.transform is not None:
            z, log_det = self.transform(z)
            log_q -= log_det
        if temperature is not None:
            self.reset_temperature()
        return z, log_q

    def log_prob(self, x, y=None):
        """Get log probability for batch

        Args:
          x: Batch
          y: Classes of x. Must be passed in if `class_cond` is True.

        Returns:
          log probability
        """
        log_q = 0
        z = x
        if self.transform is not None:
            z, log_det = self.transform.inverse(z)
            log_q += log_det
        for i in range(len(self.q0) - 1, -1, -1):
            for j in range(len(self.flows[i]) - 1, -1, -1):
                z, log_det = self.flows[i][j].inverse(z)
                log_q += log_det
            if i > 0:
                [z, z_], log_det = self.merges[i - 1].inverse(z)
                log_q += log_det
            else:
                z_ = z
            if self.class_cond:
                log_q += self.q0[i].log_prob(z_, y)
            else:
                log_q += self.q0[i].log_prob(z_)
        return log_q

    def save(self, path):
        """Save state dict of model

        Args:
          path: Path including filename where to save model
        """
        torch.save(self.state_dict(), path)

    def load(self, path):
        """Load model from state dict

        Args:
          path: Path including filename where to load model from
        """
        self.load_state_dict(torch.load(path))

    def set_temperature(self, temperature):
        """Set temperature for temperature a annealed sampling

        Args:
          temperature: Temperature parameter
        """
        for q0 in self.q0:
            if hasattr(q0, 'temperature'):
                q0.temperature = temperature
            else:
                raise NotImplementedError('One base function does not support temperature annealed sampling')

    def reset_temperature(self):
        """
        Set temperature values of base distributions back to None
        """
        self.set_temperature(None)


class BaseDistribution(nn.Module):
    """
    Base distribution of a flow-based model
    Parameters do not depend of target variable (as is the case for a VAE encoder)
    """

    def __init__(self):
        super().__init__()

    def forward(self, num_samples=1):
        """Samples from base distribution and calculates log probability

        Args:
          num_samples: Number of samples to draw from the distriubtion

        Returns:
          Samples drawn from the distribution, log probability
        """
        raise NotImplementedError

    def log_prob(self, z):
        """Calculate log probability of batch of samples

        Args:
          z: Batch of random variables to determine log probability for

        Returns:
          log probability for each batch element
        """
        raise NotImplementedError

    def sample(self, num_samples=1, **kwargs):
        """Samples from base distribution

        Args:
          num_samples: Number of samples to draw from the distriubtion

        Returns:
          Samples drawn from the distribution
        """
        z, _ = self.forward(num_samples, **kwargs)
        return z


class DiagGaussian(BaseDistribution):
    """
    Multivariate Gaussian distribution with diagonal covariance matrix
    """

    def __init__(self, shape, trainable=True):
        """Constructor

        Args:
          shape: Tuple with shape of data, if int shape has one dimension
          trainable: Flag whether to use trainable or fixed parameters
        """
        super().__init__()
        if isinstance(shape, int):
            shape = shape,
        if isinstance(shape, list):
            shape = tuple(shape)
        self.shape = shape
        self.n_dim = len(shape)
        self.d = np.prod(shape)
        if trainable:
            self.loc = nn.Parameter(torch.zeros(1, *self.shape))
            self.log_scale = nn.Parameter(torch.zeros(1, *self.shape))
        else:
            self.register_buffer('loc', torch.zeros(1, *self.shape))
            self.register_buffer('log_scale', torch.zeros(1, *self.shape))
        self.temperature = None

    def forward(self, num_samples=1, context=None):
        eps = torch.randn((num_samples,) + self.shape, dtype=self.loc.dtype, device=self.loc.device)
        if self.temperature is None:
            log_scale = self.log_scale
        else:
            log_scale = self.log_scale + np.log(self.temperature)
        z = self.loc + torch.exp(log_scale) * eps
        log_p = -0.5 * self.d * np.log(2 * np.pi) - torch.sum(log_scale + 0.5 * torch.pow(eps, 2), list(range(1, self.n_dim + 1)))
        return z, log_p

    def log_prob(self, z, context=None):
        if self.temperature is None:
            log_scale = self.log_scale
        else:
            log_scale = self.log_scale + np.log(self.temperature)
        log_p = -0.5 * self.d * np.log(2 * np.pi) - torch.sum(log_scale + 0.5 * torch.pow((z - self.loc) / torch.exp(log_scale), 2), list(range(1, self.n_dim + 1)))
        return log_p


class Target(nn.Module):
    """
    Sample target distributions to test models
    """

    def __init__(self, prop_scale=torch.tensor(6.0), prop_shift=torch.tensor(-3.0)):
        """Constructor

        Args:
          prop_scale: Scale for the uniform proposal
          prop_shift: Shift for the uniform proposal
        """
        super().__init__()
        self.register_buffer('prop_scale', prop_scale)
        self.register_buffer('prop_shift', prop_shift)

    def log_prob(self, z):
        """
        Args:
          z: value or batch of latent variable

        Returns:
          log probability of the distribution for z
        """
        raise NotImplementedError('The log probability is not implemented yet.')

    def rejection_sampling(self, num_steps=1):
        """Perform rejection sampling on image distribution

        Args:
          num_steps: Number of rejection sampling steps to perform

        Returns:
          Accepted samples
        """
        eps = torch.rand((num_steps, self.n_dims), dtype=self.prop_scale.dtype, device=self.prop_scale.device)
        z_ = self.prop_scale * eps + self.prop_shift
        prob = torch.rand(num_steps, dtype=self.prop_scale.dtype, device=self.prop_scale.device)
        prob_ = torch.exp(self.log_prob(z_) - self.max_log_prob)
        accept = prob_ > prob
        z = z_[accept, :]
        return z

    def sample(self, num_samples=1):
        """Sample from image distribution through rejection sampling

        Args:
          num_samples: Number of samples to draw

        Returns:
          Samples
        """
        z = torch.zeros((0, self.n_dims), dtype=self.prop_scale.dtype, device=self.prop_scale.device)
        while len(z) < num_samples:
            z_ = self.rejection_sampling(num_samples)
            ind = np.min([len(z_), num_samples - len(z)])
            z = torch.cat([z, z_[:ind, :]], 0)
        return z


class ConditionalDiagGaussian(Target):
    """
    Gaussian distribution conditioned on its mean and standard
    deviation

    The first half of the entries of the condition, also called context,
    are the mean, while the second half are the standard deviation.
    """

    def log_prob(self, z, context=None):
        d = z.shape[-1]
        loc = context[:, :d]
        scale = context[:, d:]
        log_p = -0.5 * d * np.log(2 * np.pi) - torch.sum(torch.log(scale) + 0.5 * torch.pow((z - loc) / scale, 2), dim=-1)
        return log_p

    def sample(self, num_samples=1, context=None):
        d = context.shape[-1] // 2
        loc = context[:, :d]
        scale = context[:, d:]
        eps = torch.randn((num_samples, d), dtype=context.dtype, device=context.device)
        z = loc + scale * eps
        return z


class BaseEncoder(nn.Module):
    """
    Base distribution of a flow-based variational autoencoder
    Parameters of the distribution depend of the target variable x
    """

    def __init__(self):
        super().__init__()

    def forward(self, x, num_samples=1):
        """
        Args:
          x: Variable to condition on, first dimension is batch size
          num_samples: number of samples to draw per element of mini-batch

        Returns
          sample of z for x, log probability for sample
        """
        raise NotImplementedError

    def log_prob(self, z, x):
        """

        Args:
          z: Primary random variable, first dimension is batch size
          x: Variable to condition on, first dimension is batch size

        Returns:
          log probability of z given x
        """
        raise NotImplementedError


class Uniform(BaseEncoder):

    def __init__(self, zmin=0.0, zmax=1.0):
        super().__init__()
        self.zmin = zmin
        self.zmax = zmax
        self.log_q = -np.log(zmax - zmin)

    def forward(self, x, num_samples=1):
        z = x.unsqueeze(1).repeat(1, num_samples, 1).uniform_(self.zmin, self.zmax)
        log_q = torch.zeros(z.size()[0:2]).fill_(self.log_q)
        return z, log_q

    def log_prob(self, z, x):
        log_q = torch.zeros(z.size()[0:2]).fill_(self.log_q)
        return log_q


class UniformGaussian(BaseDistribution):
    """
    Distribution of a 1D random variable with some entries having a uniform and
    others a Gaussian distribution
    """

    def __init__(self, ndim, ind, scale=None):
        """Constructor

        Args:
          ndim: Int, number of dimensions
          ind: Iterable, indices of uniformly distributed entries
          scale: Iterable, standard deviation of Gaussian or width of uniform distribution
        """
        super().__init__()
        self.ndim = ndim
        if isinstance(ind, int):
            ind = [ind]
        self.ndim = ndim
        if torch.is_tensor(ind):
            self.register_buffer('ind', torch._cast_Long(ind))
        else:
            self.register_buffer('ind', torch.tensor(ind, dtype=torch.long))
        ind_ = []
        for i in range(self.ndim):
            if not i in self.ind:
                ind_ += [i]
        self.register_buffer('ind_', torch.tensor(ind_, dtype=torch.long))
        perm_ = torch.cat((self.ind, self.ind_))
        inv_perm_ = torch.zeros_like(perm_)
        for i in range(self.ndim):
            inv_perm_[perm_[i]] = i
        self.register_buffer('inv_perm', inv_perm_)
        if scale is None:
            self.register_buffer('scale', torch.ones(self.ndim))
        else:
            self.register_buffer('scale', scale)

    def forward(self, num_samples=1, context=None):
        z = self.sample(num_samples)
        return z, self.log_prob(z)

    def sample(self, num_samples=1, context=None):
        eps_u = torch.rand((num_samples, len(self.ind)), dtype=self.scale.dtype, device=self.scale.device) - 0.5
        eps_g = torch.randn((num_samples, len(self.ind_)), dtype=self.scale.dtype, device=self.scale.device)
        z = torch.cat((eps_u, eps_g), -1)
        z = z[..., self.inv_perm]
        return self.scale * z

    def log_prob(self, z, context=None):
        log_p_u = torch.broadcast_to(-torch.log(self.scale[self.ind]), (len(z), -1))
        log_p_g = -0.5 * np.log(2 * np.pi) - torch.log(self.scale[self.ind_]) - 0.5 * torch.pow(z[..., self.ind_] / self.scale[self.ind_], 2)
        return torch.sum(log_p_u, -1) + torch.sum(log_p_g, -1)


class ClassCondDiagGaussian(BaseDistribution):
    """
    Class conditional multivariate Gaussian distribution with diagonal covariance matrix
    """

    def __init__(self, shape, num_classes):
        """Constructor

        Args:
          shape: Tuple with shape of data, if int shape has one dimension
          num_classes: Number of classes to condition on
        """
        super().__init__()
        if isinstance(shape, int):
            shape = shape,
        if isinstance(shape, list):
            shape = tuple(shape)
        self.shape = shape
        self.n_dim = len(shape)
        self.perm = [self.n_dim] + list(range(self.n_dim))
        self.d = np.prod(shape)
        self.num_classes = num_classes
        self.loc = nn.Parameter(torch.zeros(*self.shape, num_classes))
        self.log_scale = nn.Parameter(torch.zeros(*self.shape, num_classes))
        self.temperature = None

    def forward(self, num_samples=1, y=None):
        if y is not None:
            num_samples = len(y)
        else:
            y = torch.randint(self.num_classes, (num_samples,), device=self.loc.device)
        if y.dim() == 1:
            y_onehot = torch.zeros((self.num_classes, num_samples), dtype=self.loc.dtype, device=self.loc.device)
            y_onehot.scatter_(0, y[None], 1)
            y = y_onehot
        else:
            y = y.t()
        eps = torch.randn((num_samples,) + self.shape, dtype=self.loc.dtype, device=self.loc.device)
        loc = (self.loc @ y).permute(*self.perm)
        log_scale = (self.log_scale @ y).permute(*self.perm)
        if self.temperature is not None:
            log_scale = np.log(self.temperature) + log_scale
        z = loc + torch.exp(log_scale) * eps
        log_p = -0.5 * self.d * np.log(2 * np.pi) - torch.sum(log_scale + 0.5 * torch.pow(eps, 2), list(range(1, self.n_dim + 1)))
        return z, log_p

    def log_prob(self, z, y):
        if y.dim() == 1:
            y_onehot = torch.zeros((self.num_classes, len(y)), dtype=self.loc.dtype, device=self.loc.device)
            y_onehot.scatter_(0, y[None], 1)
            y = y_onehot
        else:
            y = y.t()
        loc = (self.loc @ y).permute(*self.perm)
        log_scale = (self.log_scale @ y).permute(*self.perm)
        if self.temperature is not None:
            log_scale = np.log(self.temperature) + log_scale
        log_p = -0.5 * self.d * np.log(2 * np.pi) - torch.sum(log_scale + 0.5 * torch.pow((z - loc) / torch.exp(log_scale), 2), list(range(1, self.n_dim + 1)))
        return log_p


class GlowBase(BaseDistribution):
    """
    Base distribution of the Glow model, i.e. Diagonal Gaussian with one mean and
    log scale for each channel
    """

    def __init__(self, shape, num_classes=None, logscale_factor=3.0):
        """Constructor

        Args:
          shape: Shape of the variables
          num_classes: Number of classes if the base is class conditional, None otherwise
          logscale_factor: Scaling factor for mean and log variance
        """
        super().__init__()
        if isinstance(shape, int):
            shape = shape,
        if isinstance(shape, list):
            shape = tuple(shape)
        self.shape = shape
        self.n_dim = len(shape)
        self.num_pix = np.prod(shape[1:])
        self.d = np.prod(shape)
        self.sum_dim = list(range(1, self.n_dim + 1))
        self.num_classes = num_classes
        self.class_cond = num_classes is not None
        self.logscale_factor = logscale_factor
        self.loc = nn.Parameter(torch.zeros(1, self.shape[0], *((self.n_dim - 1) * [1])))
        self.loc_logs = nn.Parameter(torch.zeros(1, self.shape[0], *((self.n_dim - 1) * [1])))
        self.log_scale = nn.Parameter(torch.zeros(1, self.shape[0], *((self.n_dim - 1) * [1])))
        self.log_scale_logs = nn.Parameter(torch.zeros(1, self.shape[0], *((self.n_dim - 1) * [1])))
        if self.class_cond:
            self.loc_cc = nn.Parameter(torch.zeros(self.num_classes, self.shape[0]))
            self.log_scale_cc = nn.Parameter(torch.zeros(self.num_classes, self.shape[0]))
        self.temperature = None

    def forward(self, num_samples=1, y=None):
        loc = self.loc * torch.exp(self.loc_logs * self.logscale_factor)
        log_scale = self.log_scale * torch.exp(self.log_scale_logs * self.logscale_factor)
        if self.class_cond:
            if y is not None:
                num_samples = len(y)
            else:
                y = torch.randint(self.num_classes, (num_samples,), device=self.loc.device)
            if y.dim() == 1:
                y_onehot = torch.zeros((len(y), self.num_classes), dtype=self.loc.dtype, device=self.loc.device)
                y_onehot.scatter_(1, y[:, None], 1)
                y = y_onehot
            loc = loc + (y @ self.loc_cc).view(y.size(0), self.shape[0], *((self.n_dim - 1) * [1]))
            log_scale = log_scale + (y @ self.log_scale_cc).view(y.size(0), self.shape[0], *((self.n_dim - 1) * [1]))
        if self.temperature is not None:
            log_scale = log_scale + np.log(self.temperature)
        eps = torch.randn((num_samples,) + self.shape, dtype=self.loc.dtype, device=self.loc.device)
        z = loc + torch.exp(log_scale) * eps
        log_p = -0.5 * self.d * np.log(2 * np.pi) - self.num_pix * torch.sum(log_scale, dim=self.sum_dim) - 0.5 * torch.sum(torch.pow(eps, 2), dim=self.sum_dim)
        return z, log_p

    def log_prob(self, z, y=None):
        loc = self.loc * torch.exp(self.loc_logs * self.logscale_factor)
        log_scale = self.log_scale * torch.exp(self.log_scale_logs * self.logscale_factor)
        if self.class_cond:
            if y.dim() == 1:
                y_onehot = torch.zeros((len(y), self.num_classes), dtype=self.loc.dtype, device=self.loc.device)
                y_onehot.scatter_(1, y[:, None], 1)
                y = y_onehot
            loc = loc + (y @ self.loc_cc).view(y.size(0), self.shape[0], *((self.n_dim - 1) * [1]))
            log_scale = log_scale + (y @ self.log_scale_cc).view(y.size(0), self.shape[0], *((self.n_dim - 1) * [1]))
        if self.temperature is not None:
            log_scale = log_scale + np.log(self.temperature)
        log_p = -0.5 * self.d * np.log(2 * np.pi) - self.num_pix * torch.sum(log_scale, dim=self.sum_dim) - 0.5 * torch.sum(torch.pow((z - loc) / torch.exp(log_scale), 2), dim=self.sum_dim)
        return log_p


class AffineGaussian(BaseDistribution):
    """
    Diagonal Gaussian an affine constant transformation applied to it,
    can be class conditional or not
    """

    def __init__(self, shape, affine_shape, num_classes=None):
        """Constructor

        Args:
          shape: Shape of the variables
          affine_shape: Shape of the parameters in the affine transformation
          num_classes: Number of classes if the base is class conditional, None otherwise
        """
        super().__init__()
        if isinstance(shape, int):
            shape = shape,
        if isinstance(shape, list):
            shape = tuple(shape)
        self.shape = shape
        self.n_dim = len(shape)
        self.d = np.prod(shape)
        self.sum_dim = list(range(1, self.n_dim + 1))
        self.affine_shape = affine_shape
        self.num_classes = num_classes
        self.class_cond = num_classes is not None
        if self.class_cond:
            self.transform = flows.CCAffineConst(self.affine_shape, self.num_classes)
        else:
            self.transform = flows.AffineConstFlow(self.affine_shape)
        self.temperature = None

    def forward(self, num_samples=1, y=None):
        dtype = self.transform.s.dtype
        device = self.transform.s.device
        if self.class_cond:
            if y is not None:
                num_samples = len(y)
            else:
                y = torch.randint(self.num_classes, (num_samples,), device=device)
            if y.dim() == 1:
                y_onehot = torch.zeros((len(y), self.num_classes), dtype=dtype, device=device)
                y_onehot.scatter_(1, y[:, None], 1)
                y = y_onehot
        if self.temperature is not None:
            log_scale = np.log(self.temperature)
        else:
            log_scale = 0.0
        eps = torch.randn((num_samples,) + self.shape, dtype=dtype, device=device)
        z = np.exp(log_scale) * eps
        log_p = -0.5 * self.d * np.log(2 * np.pi) - self.d * log_scale - 0.5 * torch.sum(torch.pow(eps, 2), dim=self.sum_dim)
        if self.class_cond:
            z, log_det = self.transform(z, y)
        else:
            z, log_det = self.transform(z)
        log_p -= log_det
        return z, log_p

    def log_prob(self, z, y=None):
        if self.class_cond:
            if y.dim() == 1:
                y_onehot = torch.zeros((len(y), self.num_classes), dtype=self.transform.s.dtype, device=self.transform.s.device)
                y_onehot.scatter_(1, y[:, None], 1)
                y = y_onehot
        if self.temperature is not None:
            log_scale = np.log(self.temperature)
        else:
            log_scale = 0.0
        if self.class_cond:
            z, log_p = self.transform.inverse(z, y)
        else:
            z, log_p = self.transform.inverse(z)
        z = z / np.exp(log_scale)
        log_p = log_p - self.d * log_scale - 0.5 * self.d * np.log(2 * np.pi) - 0.5 * torch.sum(torch.pow(z, 2), dim=self.sum_dim)
        return log_p


class GaussianMixture(BaseDistribution):
    """
    Mixture of Gaussians with diagonal covariance matrix
    """

    def __init__(self, n_modes, dim, loc=None, scale=None, weights=None, trainable=True):
        """Constructor

        Args:
          n_modes: Number of modes of the mixture model
          dim: Number of dimensions of each Gaussian
          loc: List of mean values
          scale: List of diagonals of the covariance matrices
          weights: List of mode probabilities
          trainable: Flag, if true parameters will be optimized during training
        """
        super().__init__()
        self.n_modes = n_modes
        self.dim = dim
        if loc is None:
            loc = np.random.randn(self.n_modes, self.dim)
        loc = np.array(loc)[None, ...]
        if scale is None:
            scale = np.ones((self.n_modes, self.dim))
        scale = np.array(scale)[None, ...]
        if weights is None:
            weights = np.ones(self.n_modes)
        weights = np.array(weights)[None, ...]
        weights /= weights.sum(1)
        if trainable:
            self.loc = nn.Parameter(torch.tensor(1.0 * loc))
            self.log_scale = nn.Parameter(torch.tensor(np.log(1.0 * scale)))
            self.weight_scores = nn.Parameter(torch.tensor(np.log(1.0 * weights)))
        else:
            self.register_buffer('loc', torch.tensor(1.0 * loc))
            self.register_buffer('log_scale', torch.tensor(np.log(1.0 * scale)))
            self.register_buffer('weight_scores', torch.tensor(np.log(1.0 * weights)))

    def forward(self, num_samples=1):
        weights = torch.softmax(self.weight_scores, 1)
        mode = torch.multinomial(weights[0, :], num_samples, replacement=True)
        mode_1h = nn.functional.one_hot(mode, self.n_modes)
        mode_1h = mode_1h[..., None]
        eps_ = torch.randn(num_samples, self.dim, dtype=self.loc.dtype, device=self.loc.device)
        scale_sample = torch.sum(torch.exp(self.log_scale) * mode_1h, 1)
        loc_sample = torch.sum(self.loc * mode_1h, 1)
        z = eps_ * scale_sample + loc_sample
        eps = (z[:, None, :] - self.loc) / torch.exp(self.log_scale)
        log_p = -0.5 * self.dim * np.log(2 * np.pi) + torch.log(weights) - 0.5 * torch.sum(torch.pow(eps, 2), 2) - torch.sum(self.log_scale, 2)
        log_p = torch.logsumexp(log_p, 1)
        return z, log_p

    def log_prob(self, z):
        weights = torch.softmax(self.weight_scores, 1)
        eps = (z[:, None, :] - self.loc) / torch.exp(self.log_scale)
        log_p = -0.5 * self.dim * np.log(2 * np.pi) + torch.log(weights) - 0.5 * torch.sum(torch.pow(eps, 2), 2) - torch.sum(self.log_scale, 2)
        log_p = torch.logsumexp(log_p, 1)
        return log_p


class GaussianPCA(BaseDistribution):
    """
    Gaussian distribution resulting from linearly mapping a normal distributed latent
    variable describing the "content of the target"
    """

    def __init__(self, dim, latent_dim=None, sigma=0.1):
        """Constructor

        Args:
          dim: Number of dimensions of the flow variables
          latent_dim: Number of dimensions of the latent "content" variable;
                           if None it is set equal to dim
          sigma: Noise level
        """
        super().__init__()
        self.dim = dim
        if latent_dim is None:
            self.latent_dim = dim
        else:
            self.latent_dim = latent_dim
        self.loc = nn.Parameter(torch.zeros(1, dim))
        self.W = nn.Parameter(torch.randn(latent_dim, dim))
        self.log_sigma = nn.Parameter(torch.tensor(np.log(sigma)))

    def forward(self, num_samples=1):
        eps = torch.randn(num_samples, self.latent_dim, dtype=self.loc.dtype, device=self.loc.device)
        z_ = torch.matmul(eps, self.W)
        z = z_ + self.loc
        Sig = torch.matmul(self.W.T, self.W) + torch.exp(self.log_sigma * 2) * torch.eye(self.dim, dtype=self.loc.dtype, device=self.loc.device)
        log_p = self.dim / 2 * np.log(2 * np.pi) - 0.5 * torch.det(Sig) - 0.5 * torch.sum(z_ * torch.matmul(z_, torch.inverse(Sig)), 1)
        return z, log_p

    def log_prob(self, z):
        z_ = z - self.loc
        Sig = torch.matmul(self.W.T, self.W) + torch.exp(self.log_sigma * 2) * torch.eye(self.dim, dtype=self.loc.dtype, device=self.loc.device)
        log_p = self.dim / 2 * np.log(2 * np.pi) - 0.5 * torch.det(Sig) - 0.5 * torch.sum(z_ * torch.matmul(z_, torch.inverse(Sig)), 1)
        return log_p


class BaseDecoder(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, z):
        """Decodes z to x

        Args:
          z: latent variable

        Returns:
          x, std of x
        """
        raise NotImplementedError

    def log_prob(self, x, z):
        """Log probability

        Args:
          x: observable
          z: latent variable

        Returns:
          log(p) of x given z
        """
        raise NotImplementedError


class NNDiagGaussianDecoder(BaseDecoder):
    """
    BaseDecoder representing a diagonal Gaussian distribution with mean and std parametrized by a NN
    """

    def __init__(self, net):
        """Constructor

        Args:
          net: neural network parametrizing mean and standard deviation of diagonal Gaussian
        """
        super().__init__()
        self.net = net

    def forward(self, z):
        mean_std = self.net(z)
        n_hidden = mean_std.size()[1] // 2
        mean = mean_std[:, :n_hidden, ...]
        std = torch.exp(0.5 * mean_std[:, n_hidden:, ...])
        return mean, std

    def log_prob(self, x, z):
        mean_std = self.net(z)
        n_hidden = mean_std.size()[1] // 2
        mean = mean_std[:, :n_hidden, ...]
        var = torch.exp(mean_std[:, n_hidden:, ...])
        if len(z) > len(x):
            x = x.unsqueeze(1)
            x = x.repeat(1, z.size()[0] // x.size()[0], *((x.dim() - 2) * [1])).view(-1, *x.size()[2:])
        log_p = -0.5 * torch.prod(torch.tensor(z.size()[1:])) * np.log(2 * np.pi) - 0.5 * torch.sum(torch.log(var) + (x - mean) ** 2 / var, list(range(1, z.dim())))
        return log_p


class NNBernoulliDecoder(BaseDecoder):
    """
    BaseDecoder representing a Bernoulli distribution with mean parametrized by a NN
    """

    def __init__(self, net):
        """Constructor

        Args:
          net: neural network parametrizing mean Bernoulli (mean = sigmoid(nn_out)
        """
        super().__init__()
        self.net = net

    def forward(self, z):
        mean = torch.sigmoid(self.net(z))
        return mean

    def log_prob(self, x, z):
        score = self.net(z)
        if len(z) > len(x):
            x = x.unsqueeze(1)
            x = x.repeat(1, z.size()[0] // x.size()[0], *((x.dim() - 2) * [1])).view(-1, *x.size()[2:])
        log_sig = lambda a: -torch.relu(-a) - torch.log(1 + torch.exp(-torch.abs(a)))
        log_p = torch.sum(x * log_sig(score) + (1 - x) * log_sig(-score), list(range(1, x.dim())))
        return log_p


class Dirac(BaseEncoder):

    def __init__(self):
        super().__init__()

    def forward(self, x, num_samples=1):
        z = x.unsqueeze(1).repeat(1, num_samples, 1)
        log_q = torch.zeros(z.size()[0:2])
        return z, log_q

    def log_prob(self, z, x):
        log_q = torch.zeros(z.size()[0:2])
        return log_q


class ConstDiagGaussian(BaseEncoder):

    def __init__(self, loc, scale):
        """Multivariate Gaussian distribution with diagonal covariance and parameters being constant wrt x

        Args:
          loc: mean vector of the distribution
          scale: vector of the standard deviations on the diagonal of the covariance matrix
        """
        super().__init__()
        self.d = len(loc)
        if not torch.is_tensor(loc):
            loc = torch.tensor(loc)
        if not torch.is_tensor(scale):
            scale = torch.tensor(scale)
        self.loc = nn.Parameter(loc.reshape((1, 1, self.d)))
        self.scale = nn.Parameter(scale)

    def forward(self, x=None, num_samples=1):
        """
        Args:
          x: Variable to condition on, will only be used to determine the batch size
          num_samples: number of samples to draw per element of mini-batch

        Returns:
          sample of z for x, log probability for sample
        """
        if x is not None:
            batch_size = len(x)
        else:
            batch_size = 1
        eps = torch.randn((batch_size, num_samples, self.d), device=x.device)
        z = self.loc + self.scale * eps
        log_q = -0.5 * self.d * np.log(2 * np.pi) - torch.sum(torch.log(self.scale) + 0.5 * torch.pow(eps, 2), 2)
        return z, log_q

    def log_prob(self, z, x):
        """
        Args:
          z: Primary random variable, first dimension is batch dimension
          x: Variable to condition on, first dimension is batch dimension

        Returns:
          log probability of z given x
        """
        if z.dim() == 1:
            z = z.unsqueeze(0)
        if z.dim() == 2:
            z = z.unsqueeze(0)
        log_q = -0.5 * self.d * np.log(2 * np.pi) - torch.sum(torch.log(self.scale) + 0.5 * ((z - self.loc) / self.scale) ** 2, 2)
        return log_q


class NNDiagGaussian(BaseEncoder):
    """
    Diagonal Gaussian distribution with mean and variance determined by a neural network
    """

    def __init__(self, net):
        """Construtor

        Args:
          net: net computing mean (first n / 2 outputs), standard deviation (second n / 2 outputs)
        """
        super().__init__()
        self.net = net

    def forward(self, x, num_samples=1):
        """
        Args:
          x: Variable to condition on
          num_samples: number of samples to draw per element of mini-batch

        Returns:
          sample of z for x, log probability for sample
        """
        batch_size = len(x)
        mean_std = self.net(x)
        n_hidden = mean_std.size()[1] // 2
        mean = mean_std[:, :n_hidden, ...].unsqueeze(1)
        std = torch.exp(0.5 * mean_std[:, n_hidden:2 * n_hidden, ...].unsqueeze(1))
        eps = torch.randn((batch_size, num_samples) + tuple(mean.size()[2:]), device=x.device)
        z = mean + std * eps
        log_q = -0.5 * torch.prod(torch.tensor(z.size()[2:])) * np.log(2 * np.pi) - torch.sum(torch.log(std) + 0.5 * torch.pow(eps, 2), list(range(2, z.dim())))
        return z, log_q

    def log_prob(self, z, x):
        """

        Args:
          z: Primary random variable, first dimension is batch dimension
          x: Variable to condition on, first dimension is batch dimension

        Returns:
          log probability of z given x
        """
        if z.dim() == 1:
            z = z.unsqueeze(0)
        if z.dim() == 2:
            z = z.unsqueeze(0)
        mean_std = self.net(x)
        n_hidden = mean_std.size()[1] // 2
        mean = mean_std[:, :n_hidden, ...].unsqueeze(1)
        var = torch.exp(mean_std[:, n_hidden:2 * n_hidden, ...].unsqueeze(1))
        log_q = -0.5 * torch.prod(torch.tensor(z.size()[2:])) * np.log(2 * np.pi) - 0.5 * torch.sum(torch.log(var) + (z - mean) ** 2 / var, 2)
        return log_q


class MHProposal(nn.Module):
    """
    Proposal distribution for the Metropolis Hastings algorithm
    """

    def __init__(self):
        super().__init__()

    def sample(self, z):
        """
        Sample new value based on previous z
        """
        raise NotImplementedError

    def log_prob(self, z_, z):
        """
        Args:
          z_: Potential new sample
          z: Previous sample

        Returns:
          Log probability of proposal distribution
        """
        raise NotImplementedError

    def forward(self, z):
        """Draw samples given z and compute log probability difference

        ```
        log(p(z | z_new)) - log(p(z_new | z))
        ```

        Args:
          z: Previous samples

        Returns:
          Proposal, difference of log probability ratio
        """
        raise NotImplementedError


class DiagGaussianProposal(MHProposal):
    """
    Diagonal Gaussian distribution with previous value as mean
    as a proposal for Metropolis Hastings algorithm
    """

    def __init__(self, shape, scale):
        """Constructor

        Args:
          shape: Shape of variables to sample
          scale: Standard deviation of distribution
        """
        super().__init__()
        self.shape = shape
        self.scale_cpu = torch.tensor(scale)
        self.register_buffer('scale', self.scale_cpu.unsqueeze(0))

    def sample(self, z):
        num_samples = len(z)
        eps = torch.randn((num_samples,) + self.shape, dtype=z.dtype, device=z.device)
        z_ = eps * self.scale + z
        return z_

    def log_prob(self, z_, z):
        log_p = -0.5 * np.prod(self.shape) * np.log(2 * np.pi) - torch.sum(torch.log(self.scale) + 0.5 * torch.pow((z_ - z) / self.scale, 2), list(range(1, z.dim())))
        return log_p

    def forward(self, z):
        num_samples = len(z)
        eps = torch.randn((num_samples,) + self.shape, dtype=z.dtype, device=z.device)
        z_ = eps * self.scale + z
        log_p_diff = torch.zeros(num_samples, dtype=z.dtype, device=z.device)
        return z_, log_p_diff


class ImagePrior(nn.Module):
    """
    Intensities of an image determine probability density of prior
    """

    def __init__(self, image, x_range=[-3, 3], y_range=[-3, 3], eps=1e-10):
        """Constructor

        Args:
          image: image as np matrix
          x_range: x range to position image at
          y_range: y range to position image at
          eps: small value to add to image to avoid log(0) problems
        """
        super().__init__()
        image_ = np.flip(image, 0).transpose() + eps
        self.image_cpu = torch.tensor(image_ / np.max(image_))
        self.image_size_cpu = self.image_cpu.size()
        self.x_range = torch.tensor(x_range)
        self.y_range = torch.tensor(y_range)
        self.register_buffer('image', self.image_cpu)
        self.register_buffer('image_size', torch.tensor(self.image_size_cpu).unsqueeze(0))
        self.register_buffer('density', torch.log(self.image_cpu / torch.sum(self.image_cpu)))
        self.register_buffer('scale', torch.tensor([[self.x_range[1] - self.x_range[0], self.y_range[1] - self.y_range[0]]]))
        self.register_buffer('shift', torch.tensor([[self.x_range[0], self.y_range[0]]]))

    def log_prob(self, z):
        """
        Args:
          z: value or batch of latent variable

        Returns:
          log probability of the distribution for z
        """
        z_ = torch.clamp((z - self.shift) / self.scale, max=1, min=0)
        ind = (z_ * (self.image_size - 1)).long()
        return self.density[ind[:, 0], ind[:, 1]]

    def rejection_sampling(self, num_steps=1):
        """Perform rejection sampling on image distribution

        Args:
         num_steps: Number of rejection sampling steps to perform

        Returns:
          Accepted samples
        """
        z_ = torch.rand((num_steps, 2), dtype=self.image.dtype, device=self.image.device)
        prob = torch.rand(num_steps, dtype=self.image.dtype, device=self.image.device)
        ind = (z_ * (self.image_size - 1)).long()
        intensity = self.image[ind[:, 0], ind[:, 1]]
        accept = intensity > prob
        z = z_[accept, :] * self.scale + self.shift
        return z

    def sample(self, num_samples=1):
        """Sample from image distribution through rejection sampling

        Args:
          num_samples: Number of samples to draw

        Returns:
          Samples
        """
        z = torch.ones((0, 2), dtype=self.image.dtype, device=self.image.device)
        while len(z) < num_samples:
            z_ = self.rejection_sampling(num_samples)
            ind = np.min([len(z_), num_samples - len(z)])
            z = torch.cat([z, z_[:ind, :]], 0)
        return z


class Flow(nn.Module):
    """
    Generic class for flow functions
    """

    def __init__(self):
        super().__init__()

    def forward(self, z):
        """
        Args:
          z: input variable, first dimension is batch dim

        Returns:
          transformed z and log of absolute determinant
        """
        raise NotImplementedError('Forward pass has not been implemented.')

    def inverse(self, z):
        raise NotImplementedError('This flow has no algebraic inverse.')


class Split(Flow):
    """
    Split features into two sets
    """

    def __init__(self, mode='channel'):
        """Constructor

        The splitting mode can be:

        - channel: Splits first feature dimension, usually channels, into two halfs
        - channel_inv: Same as channel, but with z1 and z2 flipped
        - checkerboard: Splits features using a checkerboard pattern (last feature dimension must be even)
        - checkerboard_inv: Same as checkerboard, but with inverted coloring

        Args:
         mode: splitting mode
        """
        super().__init__()
        self.mode = mode

    def forward(self, z):
        if self.mode == 'channel':
            z1, z2 = z.chunk(2, dim=1)
        elif self.mode == 'channel_inv':
            z2, z1 = z.chunk(2, dim=1)
        elif 'checkerboard' in self.mode:
            n_dims = z.dim()
            cb0 = 0
            cb1 = 1
            for i in range(1, n_dims):
                cb0_ = cb0
                cb1_ = cb1
                cb0 = [(cb0_ if j % 2 == 0 else cb1_) for j in range(z.size(n_dims - i))]
                cb1 = [(cb1_ if j % 2 == 0 else cb0_) for j in range(z.size(n_dims - i))]
            cb = cb1 if 'inv' in self.mode else cb0
            cb = torch.tensor(cb)[None].repeat(len(z), *((n_dims - 1) * [1]))
            cb = cb
            z_size = z.size()
            z1 = z.reshape(-1)[torch.nonzero(cb.view(-1), as_tuple=False)].view(*z_size[:-1], -1)
            z2 = z.reshape(-1)[torch.nonzero((1 - cb).view(-1), as_tuple=False)].view(*z_size[:-1], -1)
        else:
            raise NotImplementedError('Mode ' + self.mode + ' is not implemented.')
        log_det = 0
        return [z1, z2], log_det

    def inverse(self, z):
        z1, z2 = z
        if self.mode == 'channel':
            z = torch.cat([z1, z2], 1)
        elif self.mode == 'channel_inv':
            z = torch.cat([z2, z1], 1)
        elif 'checkerboard' in self.mode:
            n_dims = z1.dim()
            z_size = list(z1.size())
            z_size[-1] *= 2
            cb0 = 0
            cb1 = 1
            for i in range(1, n_dims):
                cb0_ = cb0
                cb1_ = cb1
                cb0 = [(cb0_ if j % 2 == 0 else cb1_) for j in range(z_size[n_dims - i])]
                cb1 = [(cb1_ if j % 2 == 0 else cb0_) for j in range(z_size[n_dims - i])]
            cb = cb1 if 'inv' in self.mode else cb0
            cb = torch.tensor(cb)[None].repeat(z_size[0], *((n_dims - 1) * [1]))
            cb = cb
            z1 = z1[..., None].repeat(*(n_dims * [1]), 2).view(*z_size[:-1], -1)
            z2 = z2[..., None].repeat(*(n_dims * [1]), 2).view(*z_size[:-1], -1)
            z = cb * z1 + (1 - cb) * z2
        else:
            raise NotImplementedError('Mode ' + self.mode + ' is not implemented.')
        log_det = 0
        return z, log_det


class TwoIndependent(Target):
    """
    Target distribution that combines two independent distributions of equal
    size into one distribution. This is needed for Augmented Normalizing Flows,
    see https://arxiv.org/abs/2002.07101
    """

    def __init__(self, target1, target2):
        super().__init__()
        self.target1 = target1
        self.target2 = target2
        self.split = Split(mode='channel')

    def log_prob(self, z):
        z1, z2 = self.split(z)[0]
        return self.target1.log_prob(z1) + self.target2.log_prob(z2)

    def sample(self, num_samples=1):
        z1 = self.target1.sample(num_samples)
        z2 = self.target2.sample(num_samples)
        return self.split.inverse([z1, z2])[0]


class TwoMoons(Target):
    """
    Bimodal two-dimensional distribution
    """

    def __init__(self):
        super().__init__()
        self.n_dims = 2
        self.max_log_prob = 0.0

    def log_prob(self, z):
        """
        ```
        log(p) = - 1/2 * ((norm(z) - 2) / 0.2) ** 2
                 + log(  exp(-1/2 * ((z[0] - 2) / 0.3) ** 2)
                       + exp(-1/2 * ((z[0] + 2) / 0.3) ** 2))
        ```

        Args:
          z: value or batch of latent variable

        Returns:
          log probability of the distribution for z
        """
        a = torch.abs(z[:, 0])
        log_prob = -0.5 * ((torch.norm(z, dim=1) - 2) / 0.2) ** 2 - 0.5 * ((a - 2) / 0.3) ** 2 + torch.log(1 + torch.exp(-4 * a / 0.09))
        return log_prob


class CircularGaussianMixture(nn.Module):
    """
    Two-dimensional Gaussian mixture arranged in a circle
    """

    def __init__(self, n_modes=8):
        """Constructor

        Args:
          n_modes: Number of modes
        """
        super(CircularGaussianMixture, self).__init__()
        self.n_modes = n_modes
        self.register_buffer('scale', torch.tensor(2 / 3 * np.sin(np.pi / self.n_modes)).float())

    def log_prob(self, z):
        d = torch.zeros((len(z), 0), dtype=z.dtype, device=z.device)
        for i in range(self.n_modes):
            d_ = ((z[:, 0] - 2 * np.sin(2 * np.pi / self.n_modes * i)) ** 2 + (z[:, 1] - 2 * np.cos(2 * np.pi / self.n_modes * i)) ** 2) / (2 * self.scale ** 2)
            d = torch.cat((d, d_[:, None]), 1)
        log_p = -torch.log(2 * np.pi * self.scale ** 2 * self.n_modes) + torch.logsumexp(-d, 1)
        return log_p

    def sample(self, num_samples=1):
        eps = torch.randn((num_samples, 2), dtype=self.scale.dtype, device=self.scale.device)
        phi = 2 * np.pi / self.n_modes * torch.randint(0, self.n_modes, (num_samples,), device=self.scale.device)
        loc = torch.stack((2 * torch.sin(phi), 2 * torch.cos(phi)), 1).type(eps.dtype)
        return eps * self.scale + loc


class RingMixture(Target):
    """
    Mixture of ring distributions in two dimensions
    """

    def __init__(self, n_rings=2):
        super().__init__()
        self.n_dims = 2
        self.max_log_prob = 0.0
        self.n_rings = n_rings
        self.scale = 1 / 4 / self.n_rings

    def log_prob(self, z):
        d = torch.zeros((len(z), 0), dtype=z.dtype, device=z.device)
        for i in range(self.n_rings):
            d_ = (torch.norm(z, dim=1) - 2 / self.n_rings * (i + 1)) ** 2 / (2 * self.scale ** 2)
            d = torch.cat((d, d_[:, None]), 1)
        return torch.logsumexp(-d, 1)


class Reverse(Flow):
    """
    Switches the forward transform of a flow layer with its inverse and vice versa
    """

    def __init__(self, flow):
        """Constructor

        Args:
          flow: Flow layer to be reversed
        """
        super().__init__()
        self.flow = flow

    def forward(self, z):
        return self.flow.inverse(z)

    def inverse(self, z):
        return self.flow.forward(z)


class Composite(Flow):
    """
    Composes several flows into one, in the order they are given.
    """

    def __init__(self, flows):
        """Constructor

        Args:
          flows: Iterable of flows to composite
        """
        super().__init__()
        self._flows = nn.ModuleList(flows)

    @staticmethod
    def _cascade(inputs, funcs):
        batch_size = inputs.shape[0]
        outputs = inputs
        total_logabsdet = torch.zeros(batch_size)
        for func in funcs:
            outputs, logabsdet = func(outputs)
            total_logabsdet += logabsdet
        return outputs, total_logabsdet

    def forward(self, inputs):
        funcs = self._flows
        return self._cascade(inputs, funcs)

    def inverse(self, inputs):
        funcs = (flow.inverse for flow in self._flows[::-1])
        return self._cascade(inputs, funcs)


class Permute(Flow):
    """
    Permutation features along the channel dimension
    """

    def __init__(self, num_channels, mode='shuffle'):
        """Constructor

        Args:
          num_channel: Number of channels
          mode: Mode of permuting features, can be shuffle for random permutation or swap for interchanging upper and lower part
        """
        super().__init__()
        self.mode = mode
        self.num_channels = num_channels
        if self.mode == 'shuffle':
            perm = torch.randperm(self.num_channels)
            inv_perm = torch.empty_like(perm).scatter_(dim=0, index=perm, src=torch.arange(self.num_channels))
            self.register_buffer('perm', perm)
            self.register_buffer('inv_perm', inv_perm)

    def forward(self, z, context=None):
        if self.mode == 'shuffle':
            z = z[:, self.perm, ...]
        elif self.mode == 'swap':
            z1 = z[:, :self.num_channels // 2, ...]
            z2 = z[:, self.num_channels // 2:, ...]
            z = torch.cat([z2, z1], dim=1)
        else:
            raise NotImplementedError('The mode ' + self.mode + ' is not implemented.')
        log_det = torch.zeros(len(z), device=z.device)
        return z, log_det

    def inverse(self, z, context=None):
        if self.mode == 'shuffle':
            z = z[:, self.inv_perm, ...]
        elif self.mode == 'swap':
            z1 = z[:, :(self.num_channels + 1) // 2, ...]
            z2 = z[:, (self.num_channels + 1) // 2:, ...]
            z = torch.cat([z2, z1], dim=1)
        else:
            raise NotImplementedError('The mode ' + self.mode + ' is not implemented.')
        log_det = torch.zeros(len(z), device=z.device)
        return z, log_det


class Invertible1x1Conv(Flow):
    """
    Invertible 1x1 convolution introduced in the Glow paper
    Assumes 4d input/output tensors of the form NCHW
    """

    def __init__(self, num_channels, use_lu=False):
        """Constructor

        Args:
          num_channels: Number of channels of the data
          use_lu: Flag whether to parametrize weights through the LU decomposition
        """
        super().__init__()
        self.num_channels = num_channels
        self.use_lu = use_lu
        Q, _ = torch.linalg.qr(torch.randn(self.num_channels, self.num_channels))
        if use_lu:
            P, L, U = torch.lu_unpack(*Q.lu())
            self.register_buffer('P', P)
            self.L = nn.Parameter(L)
            S = U.diag()
            self.register_buffer('sign_S', torch.sign(S))
            self.log_S = nn.Parameter(torch.log(torch.abs(S)))
            self.U = nn.Parameter(torch.triu(U, diagonal=1))
            self.register_buffer('eye', torch.diag(torch.ones(self.num_channels)))
        else:
            self.W = nn.Parameter(Q)

    def _assemble_W(self, inverse=False):
        L = torch.tril(self.L, diagonal=-1) + self.eye
        U = torch.triu(self.U, diagonal=1) + torch.diag(self.sign_S * torch.exp(self.log_S))
        if inverse:
            if self.log_S.dtype == torch.float64:
                L_inv = torch.inverse(L)
                U_inv = torch.inverse(U)
            else:
                L_inv = torch.inverse(L.double()).type(self.log_S.dtype)
                U_inv = torch.inverse(U.double()).type(self.log_S.dtype)
            W = U_inv @ L_inv @ self.P.t()
        else:
            W = self.P @ L @ U
        return W

    def forward(self, z):
        if self.use_lu:
            W = self._assemble_W(inverse=True)
            log_det = -torch.sum(self.log_S)
        else:
            W_dtype = self.W.dtype
            if W_dtype == torch.float64:
                W = torch.inverse(self.W)
            else:
                W = torch.inverse(self.W.double()).type(W_dtype)
            W = W.view(*W.size(), 1, 1)
            log_det = -torch.slogdet(self.W)[1]
        W = W.view(self.num_channels, self.num_channels, 1, 1)
        z_ = torch.nn.functional.conv2d(z, W)
        log_det = log_det * z.size(2) * z.size(3)
        return z_, log_det

    def inverse(self, z):
        if self.use_lu:
            W = self._assemble_W()
            log_det = torch.sum(self.log_S)
        else:
            W = self.W
            log_det = torch.slogdet(self.W)[1]
        W = W.view(self.num_channels, self.num_channels, 1, 1)
        z_ = torch.nn.functional.conv2d(z, W)
        log_det = log_det * z.size(2) * z.size(3)
        return z_, log_det


class InvertibleAffine(Flow):
    """
    Invertible affine transformation without shift, i.e. one-dimensional
    version of the invertible 1x1 convolutions
    """

    def __init__(self, num_channels, use_lu=True):
        """Constructor

        Args:
          num_channels: Number of channels of the data
          use_lu: Flag whether to parametrize weights through the LU decomposition
        """
        super().__init__()
        self.num_channels = num_channels
        self.use_lu = use_lu
        Q, _ = torch.linalg.qr(torch.randn(self.num_channels, self.num_channels))
        if use_lu:
            P, L, U = torch.lu_unpack(*Q.lu())
            self.register_buffer('P', P)
            self.L = nn.Parameter(L)
            S = U.diag()
            self.register_buffer('sign_S', torch.sign(S))
            self.log_S = nn.Parameter(torch.log(torch.abs(S)))
            self.U = nn.Parameter(torch.triu(U, diagonal=1))
            self.register_buffer('eye', torch.diag(torch.ones(self.num_channels)))
        else:
            self.W = nn.Parameter(Q)

    def _assemble_W(self, inverse=False):
        L = torch.tril(self.L, diagonal=-1) + self.eye
        U = torch.triu(self.U, diagonal=1) + torch.diag(self.sign_S * torch.exp(self.log_S))
        if inverse:
            if self.log_S.dtype == torch.float64:
                L_inv = torch.inverse(L)
                U_inv = torch.inverse(U)
            else:
                L_inv = torch.inverse(L.double()).type(self.log_S.dtype)
                U_inv = torch.inverse(U.double()).type(self.log_S.dtype)
            W = U_inv @ L_inv @ self.P.t()
        else:
            W = self.P @ L @ U
        return W

    def forward(self, z, context=None):
        if self.use_lu:
            W = self._assemble_W(inverse=True)
            log_det = -torch.sum(self.log_S)
        else:
            W_dtype = self.W.dtype
            if W_dtype == torch.float64:
                W = torch.inverse(self.W)
            else:
                W = torch.inverse(self.W.double()).type(W_dtype)
            log_det = -torch.slogdet(self.W)[1]
        z_ = z @ W
        return z_, log_det

    def inverse(self, z, context=None):
        if self.use_lu:
            W = self._assemble_W()
            log_det = torch.sum(self.log_S)
        else:
            W = self.W
            log_det = torch.slogdet(self.W)[1]
        z_ = z @ W
        return z_, log_det


class _Permutation(Flow):
    """Permutes inputs on a given dimension using a given permutation."""

    def __init__(self, permutation, dim=1):
        if permutation.ndimension() != 1:
            raise ValueError('Permutation must be a 1D tensor.')
        super().__init__()
        self._dim = dim
        self.register_buffer('_permutation', permutation)

    @property
    def _inverse_permutation(self):
        return torch.argsort(self._permutation)

    @staticmethod
    def _permute(inputs, permutation, dim):
        if dim >= inputs.ndimension():
            raise ValueError('No dimension {} in inputs.'.format(dim))
        if inputs.shape[dim] != len(permutation):
            raise ValueError('Dimension {} in inputs must be of size {}.'.format(dim, len(permutation)))
        batch_size = inputs.shape[0]
        outputs = torch.index_select(inputs, dim, permutation)
        logabsdet = torch.zeros(batch_size)
        return outputs, logabsdet

    def forward(self, inputs, context=None):
        return self._permute(inputs, self._permutation, self._dim)

    def inverse(self, inputs, context=None):
        return self._permute(inputs, self._inverse_permutation, self._dim)


class _RandomPermutation(_Permutation):
    """Permutes using a random, but fixed, permutation. Only works with 1D inputs."""

    def __init__(self, features, dim=1):
        super().__init__(torch.randperm(features), dim)


class _LinearCache(object):
    """Helper class to store the cache of a linear transform.

    The cache consists of: the weight matrix, its inverse and its log absolute determinant.
    """

    def __init__(self):
        self.weight = None
        self.inverse = None
        self.logabsdet = None

    def invalidate(self):
        self.weight = None
        self.inverse = None
        self.logabsdet = None


class _Linear(Flow):
    """Abstract base class for linear transforms that parameterize a weight matrix."""

    def __init__(self, features, using_cache=False):
        super().__init__()
        self.features = features
        self.bias = nn.Parameter(torch.zeros(features))
        self.using_cache = using_cache
        self.cache = _LinearCache()

    def forward(self, inputs, context=None):
        if not self.training and self.using_cache:
            self._check_forward_cache()
            outputs = F.linear(inputs, self.cache.weight, self.bias)
            logabsdet = self.cache.logabsdet * torch.ones(outputs.shape[0])
            return outputs, logabsdet
        else:
            return self.forward_no_cache(inputs)

    def _check_forward_cache(self):
        if self.cache.weight is None and self.cache.logabsdet is None:
            self.cache.weight, self.cache.logabsdet = self.weight_and_logabsdet()
        elif self.cache.weight is None:
            self.cache.weight = self.weight()
        elif self.cache.logabsdet is None:
            self.cache.logabsdet = self.logabsdet()

    def inverse(self, inputs, context=None):
        if not self.training and self.using_cache:
            self._check_inverse_cache()
            outputs = F.linear(inputs - self.bias, self.cache.inverse)
            logabsdet = -self.cache.logabsdet * torch.ones(outputs.shape[0])
            return outputs, logabsdet
        else:
            return self.inverse_no_cache(inputs)

    def _check_inverse_cache(self):
        if self.cache.inverse is None and self.cache.logabsdet is None:
            self.cache.inverse, self.cache.logabsdet = self.weight_inverse_and_logabsdet()
        elif self.cache.inverse is None:
            self.cache.inverse = self.weight_inverse()
        elif self.cache.logabsdet is None:
            self.cache.logabsdet = self.logabsdet()

    def train(self, mode=True):
        if mode:
            self.cache.invalidate()
        return super().train(mode)

    def use_cache(self, mode=True):
        self.using_cache = mode

    def weight_and_logabsdet(self):
        return self.weight(), self.logabsdet()

    def weight_inverse_and_logabsdet(self):
        return self.weight_inverse(), self.logabsdet()

    def forward_no_cache(self, inputs):
        """Applies `forward` method without using the cache."""
        raise NotImplementedError()

    def inverse_no_cache(self, inputs):
        """Applies `inverse` method without using the cache."""
        raise NotImplementedError()

    def weight(self):
        """Returns the weight matrix."""
        raise NotImplementedError()

    def weight_inverse(self):
        """Returns the inverse weight matrix."""
        raise NotImplementedError()

    def logabsdet(self):
        """Returns the log absolute determinant of the weight matrix."""
        raise NotImplementedError()


class _LULinear(_Linear):
    """A linear transform where we parameterize the LU decomposition of the weights."""

    def __init__(self, features, using_cache=False, identity_init=True, eps=0.001):
        super().__init__(features, using_cache)
        self.eps = eps
        self.lower_indices = np.tril_indices(features, k=-1)
        self.upper_indices = np.triu_indices(features, k=1)
        self.diag_indices = np.diag_indices(features)
        n_triangular_entries = (features - 1) * features // 2
        self.lower_entries = nn.Parameter(torch.zeros(n_triangular_entries))
        self.upper_entries = nn.Parameter(torch.zeros(n_triangular_entries))
        self.unconstrained_upper_diag = nn.Parameter(torch.zeros(features))
        self._initialize(identity_init)

    def _initialize(self, identity_init):
        init.zeros_(self.bias)
        if identity_init:
            init.zeros_(self.lower_entries)
            init.zeros_(self.upper_entries)
            constant = np.log(np.exp(1 - self.eps) - 1)
            init.constant_(self.unconstrained_upper_diag, constant)
        else:
            stdv = 1.0 / np.sqrt(self.features)
            init.uniform_(self.lower_entries, -stdv, stdv)
            init.uniform_(self.upper_entries, -stdv, stdv)
            init.uniform_(self.unconstrained_upper_diag, -stdv, stdv)

    def _create_lower_upper(self):
        lower = self.lower_entries.new_zeros(self.features, self.features)
        lower[self.lower_indices[0], self.lower_indices[1]] = self.lower_entries
        lower[self.diag_indices[0], self.diag_indices[1]] = 1.0
        upper = self.upper_entries.new_zeros(self.features, self.features)
        upper[self.upper_indices[0], self.upper_indices[1]] = self.upper_entries
        upper[self.diag_indices[0], self.diag_indices[1]] = self.upper_diag
        return lower, upper

    def forward_no_cache(self, inputs):
        """
        Cost:

        ```
            output = O(D^2N)
            logabsdet = O(D)
        ```

        where:

        ```
            D = num of features
            N = num of inputs
        ```
        """
        lower, upper = self._create_lower_upper()
        outputs = F.linear(inputs, upper)
        outputs = F.linear(outputs, lower, self.bias)
        logabsdet = self.logabsdet() * inputs.new_ones(outputs.shape[0])
        return outputs, logabsdet

    def inverse_no_cache(self, inputs):
        """
        Cost:

        ```
            output = O(D^2N)
            logabsdet = O(D)
        ```

        where:

        ```
            D = num of features
            N = num of inputs
        ```
        """
        lower, upper = self._create_lower_upper()
        outputs = inputs - self.bias
        try:
            outputs = torch.linalg.solve_triangular(lower, outputs.t(), upper=False, unitriangular=True)
            outputs = torch.linalg.solve_triangular(upper, outputs, upper=True, unitriangular=False)
        except:
            outputs, _ = torch.triangular_solve(outputs.t(), lower, upper=False, unitriangular=True)
            outputs, _ = torch.triangular_solve(outputs, upper, upper=True, unitriangular=False)
        outputs = outputs.t()
        logabsdet = -self.logabsdet()
        logabsdet = logabsdet * inputs.new_ones(outputs.shape[0])
        return outputs, logabsdet

    def weight(self):
        """
        Cost:

        ```
            weight = O(D^3)
        ```

        where:

        ```
            D = num of features
        ```
        """
        lower, upper = self._create_lower_upper()
        return lower @ upper

    def weight_inverse(self):
        """
        Cost:

        ```
            inverse = O(D^3)
        ```

        where:

        ```
            D = num of features
        ```
        """
        lower, upper = self._create_lower_upper()
        identity = torch.eye(self.features, self.features)
        lower_inverse = torch.linalg.solve_triangular(lower, identity, upper=False, unitriangular=True)
        weight_inverse = torch.linalg.solve_triangular(upper, lower_inverse, upper=True, unitriangular=False)
        return weight_inverse

    @property
    def upper_diag(self):
        return F.softplus(self.unconstrained_upper_diag) + self.eps

    def logabsdet(self):
        """
        Cost:

        ```
            logabsdet = O(D)
        ```

        where:

        ```
            D = num of features
        ```
        """
        return torch.sum(torch.log(self.upper_diag))


class LULinearPermute(Flow):
    """
    Fixed permutation combined with a linear transformation parametrized
    using the LU decomposition, used in https://arxiv.org/abs/1906.04032
    """

    def __init__(self, num_channels, identity_init=True):
        """Constructor

        Args:
          num_channels: Number of dimensions of the data
          identity_init: Flag, whether to initialize linear transform as identity matrix
        """
        super().__init__()
        self.permutation = _RandomPermutation(num_channels)
        self.linear = _LULinear(num_channels, identity_init=identity_init)

    def forward(self, z, context=None):
        z, log_det = self.linear.inverse(z, context=context)
        z, _ = self.permutation.inverse(z, context=context)
        return z, log_det.view(-1)

    def inverse(self, z, context=None):
        z, _ = self.permutation(z, context=context)
        z, log_det = self.linear(z, context=context)
        return z, log_det.view(-1)


class Coupling(Flow):
    """A base class for coupling layers. Supports 2D inputs (NxD), as well as 4D inputs for
    images (NxCxHxW). For images the splitting is done on the channel dimension, using the
    provided 1D mask."""

    def __init__(self, mask, transform_net_create_fn, unconditional_transform=None):
        """Constructor.

        mask: a 1-dim tensor, tuple or list. It indexes inputs as follows:

        - if `mask[i] > 0`, `input[i]` will be transformed.
        - if `mask[i] <= 0`, `input[i]` will be passed unchanged.

        Args:
          mask
        """
        mask = torch.as_tensor(mask)
        if mask.dim() != 1:
            raise ValueError('Mask must be a 1-dim tensor.')
        if mask.numel() <= 0:
            raise ValueError("Mask can't be empty.")
        super().__init__()
        self.features = len(mask)
        features_vector = torch.arange(self.features)
        self.register_buffer('identity_features', features_vector.masked_select(mask <= 0))
        self.register_buffer('transform_features', features_vector.masked_select(mask > 0))
        assert self.num_identity_features + self.num_transform_features == self.features
        self.transform_net = transform_net_create_fn(self.num_identity_features, self.num_transform_features * self._transform_dim_multiplier())
        if unconditional_transform is None:
            self.unconditional_transform = None
        else:
            self.unconditional_transform = unconditional_transform(features=self.num_identity_features)

    @property
    def num_identity_features(self):
        return len(self.identity_features)

    @property
    def num_transform_features(self):
        return len(self.transform_features)

    def forward(self, inputs, context=None):
        if inputs.dim() not in [2, 4]:
            raise ValueError('Inputs must be a 2D or a 4D tensor.')
        if inputs.shape[1] != self.features:
            raise ValueError('Expected features = {}, got {}.'.format(self.features, inputs.shape[1]))
        identity_split = inputs[:, self.identity_features, ...]
        transform_split = inputs[:, self.transform_features, ...]
        transform_params = self.transform_net(identity_split, context)
        transform_split, logabsdet = self._coupling_transform_forward(inputs=transform_split, transform_params=transform_params)
        if self.unconditional_transform is not None:
            identity_split, logabsdet_identity = self.unconditional_transform(identity_split, context)
            logabsdet += logabsdet_identity
        outputs = torch.empty_like(inputs)
        outputs[:, self.identity_features, ...] = identity_split
        outputs[:, self.transform_features, ...] = transform_split
        return outputs, logabsdet

    def inverse(self, inputs, context=None):
        if inputs.dim() not in [2, 4]:
            raise ValueError('Inputs must be a 2D or a 4D tensor.')
        if inputs.shape[1] != self.features:
            raise ValueError('Expected features = {}, got {}.'.format(self.features, inputs.shape[1]))
        identity_split = inputs[:, self.identity_features, ...]
        transform_split = inputs[:, self.transform_features, ...]
        logabsdet = 0.0
        if self.unconditional_transform is not None:
            identity_split, logabsdet = self.unconditional_transform.inverse(identity_split, context)
        transform_params = self.transform_net(identity_split, context)
        transform_split, logabsdet_split = self._coupling_transform_inverse(inputs=transform_split, transform_params=transform_params)
        logabsdet += logabsdet_split
        outputs = torch.empty_like(inputs)
        outputs[:, self.identity_features] = identity_split
        outputs[:, self.transform_features] = transform_split
        return outputs, logabsdet

    def _transform_dim_multiplier(self):
        """Number of features to output for each transform dimension."""
        raise NotImplementedError()

    def _coupling_transform_forward(self, inputs, transform_params):
        """Forward pass of the coupling transform."""
        raise NotImplementedError()

    def _coupling_transform_inverse(self, inputs, transform_params):
        """Inverse of the coupling transform."""
        raise NotImplementedError()


class PiecewiseCoupling(Coupling):

    def _coupling_transform_forward(self, inputs, transform_params):
        return self._coupling_transform(inputs, transform_params, inverse=False)

    def _coupling_transform_inverse(self, inputs, transform_params):
        return self._coupling_transform(inputs, transform_params, inverse=True)

    def _coupling_transform(self, inputs, transform_params, inverse=False):
        if inputs.dim() == 4:
            b, c, h, w = inputs.shape
            transform_params = transform_params.reshape(b, c, -1, h, w).permute(0, 1, 3, 4, 2)
        elif inputs.dim() == 2:
            b, d = inputs.shape
            transform_params = transform_params.reshape(b, d, -1)
        outputs, logabsdet = self._piecewise_cdf(inputs, transform_params, inverse)
        return outputs, utils.sum_except_batch(logabsdet)

    def _piecewise_cdf(self, inputs, transform_params, inverse=False):
        raise NotImplementedError()


DEFAULT_MIN_DERIVATIVE = 0.001


class ResidualBlock(nn.Module):
    """A general-purpose residual block. Works only with 1-dim inputs."""

    def __init__(self, features, context_features, activation=F.relu, dropout_probability=0.0, use_batch_norm=False, zero_initialization=True):
        super().__init__()
        self.activation = activation
        self.use_batch_norm = use_batch_norm
        if use_batch_norm:
            self.batch_norm_layers = nn.ModuleList([nn.BatchNorm1d(features, eps=0.001) for _ in range(2)])
        if context_features is not None:
            self.context_layer = nn.Linear(context_features, features)
        self.linear_layers = nn.ModuleList([nn.Linear(features, features) for _ in range(2)])
        self.dropout = nn.Dropout(p=dropout_probability)
        if zero_initialization:
            init.uniform_(self.linear_layers[-1].weight, -0.001, 0.001)
            init.uniform_(self.linear_layers[-1].bias, -0.001, 0.001)

    def forward(self, inputs, context=None):
        temps = inputs
        if self.use_batch_norm:
            temps = self.batch_norm_layers[0](temps)
        temps = self.activation(temps)
        temps = self.linear_layers[0](temps)
        if self.use_batch_norm:
            temps = self.batch_norm_layers[1](temps)
        temps = self.activation(temps)
        temps = self.dropout(temps)
        temps = self.linear_layers[1](temps)
        if context is not None:
            temps = F.glu(torch.cat((temps, self.context_layer(context)), dim=1), dim=1)
        return inputs + temps


class ResidualNet(nn.Module):
    """A general-purpose residual network. Works only with 1-dim inputs."""

    def __init__(self, in_features, out_features, hidden_features, context_features=None, num_blocks=2, activation=F.relu, dropout_probability=0.0, use_batch_norm=False, preprocessing=None):
        super().__init__()
        self.hidden_features = hidden_features
        self.context_features = context_features
        self.preprocessing = preprocessing
        if context_features is not None:
            self.initial_layer = nn.Linear(in_features + context_features, hidden_features)
        else:
            self.initial_layer = nn.Linear(in_features, hidden_features)
        self.blocks = nn.ModuleList([ResidualBlock(features=hidden_features, context_features=context_features, activation=activation, dropout_probability=dropout_probability, use_batch_norm=use_batch_norm) for _ in range(num_blocks)])
        self.final_layer = nn.Linear(hidden_features, out_features)

    def forward(self, inputs, context=None):
        if self.preprocessing is None:
            temps = inputs
        else:
            temps = self.preprocessing(inputs)
        if context is None:
            temps = self.initial_layer(temps)
        else:
            temps = self.initial_layer(torch.cat((temps, context), dim=1))
        for block in self.blocks:
            temps = block(temps, context=context)
        outputs = self.final_layer(temps)
        return outputs


def create_alternating_binary_mask(features, even=True):
    """Creates a binary mask of a given dimension which alternates its masking.

    Args:
      features: Dimension of mask.
      even: If True, even values are assigned 1s, odd 0s. If False, vice versa.

    Returns:
      Alternating binary mask of type torch.Tensor.
    """
    mask = torch.zeros(features).byte()
    start = 0 if even else 1
    mask[start::2] += 1
    return mask


class CoupledRationalQuadraticSpline(Flow):
    """
    Neural spline flow coupling layer, wrapper for the implementation
    of Durkan et al., see [source](https://github.com/bayesiains/nsf)
    """

    def __init__(self, num_input_channels, num_blocks, num_hidden_channels, num_context_channels=None, num_bins=8, tails='linear', tail_bound=3.0, activation=nn.ReLU, dropout_probability=0.0, reverse_mask=False, init_identity=True):
        """Constructor

        Args:
          num_input_channels (int): Flow dimension
          num_blocks (int): Number of residual blocks of the parameter NN
          num_hidden_channels (int): Number of hidden units of the NN
          num_context_channels (int): Number of context/conditional channels
          num_bins (int): Number of bins
          tails (str): Behaviour of the tails of the distribution, can be linear, circular for periodic distribution, or None for distribution on the compact interval
          tail_bound (float): Bound of the spline tails
          activation (torch module): Activation function
          dropout_probability (float): Dropout probability of the NN
          reverse_mask (bool): Flag whether the reverse mask should be used
          init_identity (bool): Flag, initialize transform as identity
        """
        super().__init__()

        def transform_net_create_fn(in_features, out_features):
            net = ResidualNet(in_features=in_features, out_features=out_features, context_features=num_context_channels, hidden_features=num_hidden_channels, num_blocks=num_blocks, activation=activation(), dropout_probability=dropout_probability, use_batch_norm=False)
            if init_identity:
                torch.nn.init.constant_(net.final_layer.weight, 0.0)
                torch.nn.init.constant_(net.final_layer.bias, np.log(np.exp(1 - DEFAULT_MIN_DERIVATIVE) - 1))
            return net
        self.prqct = PiecewiseRationalQuadraticCoupling(mask=create_alternating_binary_mask(num_input_channels, even=reverse_mask), transform_net_create_fn=transform_net_create_fn, num_bins=num_bins, tails=tails, tail_bound=tail_bound, apply_unconditional_transform=True)

    def forward(self, z, context=None):
        z, log_det = self.prqct.inverse(z, context)
        return z, log_det.view(-1)

    def inverse(self, z, context=None):
        z, log_det = self.prqct(z, context)
        return z, log_det.view(-1)


class PeriodicFeaturesElementwise(nn.Module):
    """
    Converts a specified part of the input to periodic features by
    replacing those features f with
    w1 * sin(scale * f) + w2 * cos(scale * f).

    Note that this operation is done elementwise and, therefore,
    some information about the feature can be lost.
    """

    def __init__(self, ndim, ind, scale=1.0, bias=False, activation=None):
        """Constructor

        Args:
          ndim (int): number of dimensions
          ind (iterable): indices of input elements to convert to periodic features
          scale: Scalar or iterable, used to scale inputs before converting them to periodic features
          bias: Flag, whether to add a bias
          activation: Function or None, activation function to be applied
        """
        super(PeriodicFeaturesElementwise, self).__init__()
        self.ndim = ndim
        if torch.is_tensor(ind):
            self.register_buffer('ind', torch._cast_Long(ind))
        else:
            self.register_buffer('ind', torch.tensor(ind, dtype=torch.long))
        ind_ = []
        for i in range(self.ndim):
            if not i in self.ind:
                ind_ += [i]
        self.register_buffer('ind_', torch.tensor(ind_, dtype=torch.long))
        perm_ = torch.cat((self.ind, self.ind_))
        inv_perm_ = torch.zeros_like(perm_)
        for i in range(self.ndim):
            inv_perm_[perm_[i]] = i
        self.register_buffer('inv_perm', inv_perm_)
        self.weights = nn.Parameter(torch.ones(len(self.ind), 2))
        if torch.is_tensor(scale):
            self.register_buffer('scale', scale)
        else:
            self.scale = scale
        self.apply_bias = bias
        if self.apply_bias:
            self.bias = nn.Parameter(torch.zeros(len(self.ind)))
        if activation is None:
            self.activation = torch.nn.Identity()
        else:
            self.activation = activation

    def forward(self, inputs):
        inputs_ = inputs[..., self.ind]
        inputs_ = self.scale * inputs_
        inputs_ = self.weights[:, 0] * torch.sin(inputs_) + self.weights[:, 1] * torch.cos(inputs_)
        if self.apply_bias:
            inputs_ = inputs_ + self.bias
        inputs_ = self.activation(inputs_)
        out = torch.cat((inputs_, inputs[..., self.ind_]), -1)
        return out[..., self.inv_perm]


class CircularCoupledRationalQuadraticSpline(Flow):
    """
    Neural spline flow coupling layer with circular coordinates
    """

    def __init__(self, num_input_channels, num_blocks, num_hidden_channels, ind_circ, num_context_channels=None, num_bins=8, tail_bound=3.0, activation=nn.ReLU, dropout_probability=0.0, reverse_mask=False, mask=None, init_identity=True):
        """Constructor

        Args:
          num_input_channels (int): Flow dimension
          num_blocks (int): Number of residual blocks of the parameter NN
          num_hidden_channels (int): Number of hidden units of the NN
          num_context_channels (int): Number of context/conditional channels
          ind_circ (Iterable): Indices of the circular coordinates
          num_bins (int): Number of bins
          tail_bound (float or Iterable): Bound of the spline tails
          activation (torch module): Activation function
          dropout_probability (float): Dropout probability of the NN
          reverse_mask (bool): Flag whether the reverse mask should be used
          mask (torch tensor): Mask to be used, alternating masked generated is None
          init_identity (bool): Flag, initialize transform as identity
        """
        super().__init__()
        if mask is None:
            mask = create_alternating_binary_mask(num_input_channels, even=reverse_mask)
        features_vector = torch.arange(num_input_channels)
        identity_features = features_vector.masked_select(mask <= 0)
        ind_circ = torch.tensor(ind_circ)
        ind_circ_id = []
        for i, id in enumerate(identity_features):
            if id in ind_circ:
                ind_circ_id += [i]
        if torch.is_tensor(tail_bound):
            scale_pf = np.pi / tail_bound[ind_circ_id]
        else:
            scale_pf = np.pi / tail_bound

        def transform_net_create_fn(in_features, out_features):
            if len(ind_circ_id) > 0:
                pf = PeriodicFeaturesElementwise(in_features, ind_circ_id, scale_pf)
            else:
                pf = None
            net = ResidualNet(in_features=in_features, out_features=out_features, context_features=num_context_channels, hidden_features=num_hidden_channels, num_blocks=num_blocks, activation=activation(), dropout_probability=dropout_probability, use_batch_norm=False, preprocessing=pf)
            if init_identity:
                torch.nn.init.constant_(net.final_layer.weight, 0.0)
                torch.nn.init.constant_(net.final_layer.bias, np.log(np.exp(1 - DEFAULT_MIN_DERIVATIVE) - 1))
            return net
        tails = [('circular' if i in ind_circ else 'linear') for i in range(num_input_channels)]
        self.prqct = PiecewiseRationalQuadraticCoupling(mask=mask, transform_net_create_fn=transform_net_create_fn, num_bins=num_bins, tails=tails, tail_bound=tail_bound, apply_unconditional_transform=True)

    def forward(self, z, context=None):
        z, log_det = self.prqct.inverse(z, context)
        return z, log_det.view(-1)

    def inverse(self, z, context=None):
        z, log_det = self.prqct(z, context)
        return z, log_det.view(-1)


class Autoregressive(Flow):
    """Transforms each input variable with an invertible elementwise transformation.

    The parameters of each invertible elementwise transformation can be functions of previous input
    variables, but they must not depend on the current or any following input variables.

    **NOTE** Calculating the inverse transform is D times slower than calculating the
    forward transform, where D is the dimensionality of the input to the transform.
    """

    def __init__(self, autoregressive_net):
        super(Autoregressive, self).__init__()
        self.autoregressive_net = autoregressive_net

    def forward(self, inputs, context=None):
        autoregressive_params = self.autoregressive_net(inputs, context)
        outputs, logabsdet = self._elementwise_forward(inputs, autoregressive_params)
        return outputs, logabsdet

    def inverse(self, inputs, context=None):
        num_inputs = np.prod(inputs.shape[1:])
        outputs = torch.zeros_like(inputs)
        logabsdet = None
        for _ in range(num_inputs):
            autoregressive_params = self.autoregressive_net(outputs, context)
            outputs, logabsdet = self._elementwise_inverse(inputs, autoregressive_params)
        return outputs, logabsdet

    def _output_dim_multiplier(self):
        raise NotImplementedError()

    def _elementwise_forward(self, inputs, autoregressive_params):
        raise NotImplementedError()

    def _elementwise_inverse(self, inputs, autoregressive_params):
        raise NotImplementedError()


class AutoregressiveRationalQuadraticSpline(Flow):
    """
    Neural spline flow coupling layer, wrapper for the implementation
    of Durkan et al., see [sources](https://github.com/bayesiains/nsf)
    """

    def __init__(self, num_input_channels, num_blocks, num_hidden_channels, num_context_channels=None, num_bins=8, tail_bound=3, activation=nn.ReLU, dropout_probability=0.0, permute_mask=False, init_identity=True):
        """Constructor

        Args:
          num_input_channels (int): Flow dimension
          num_blocks (int): Number of residual blocks of the parameter NN
          num_hidden_channels (int): Number of hidden units of the NN
          num_context_channels (int): Number of context/conditional channels
          num_bins (int): Number of bins
          tail_bound (int): Bound of the spline tails
          activation (torch.nn.Module): Activation function
          dropout_probability (float): Dropout probability of the NN
          permute_mask (bool): Flag, permutes the mask of the NN
          init_identity (bool): Flag, initialize transform as identity
        """
        super().__init__()
        self.mprqat = MaskedPiecewiseRationalQuadraticAutoregressive(features=num_input_channels, hidden_features=num_hidden_channels, context_features=num_context_channels, num_bins=num_bins, tails='linear', tail_bound=tail_bound, num_blocks=num_blocks, use_residual_blocks=True, random_mask=False, permute_mask=permute_mask, activation=activation(), dropout_probability=dropout_probability, use_batch_norm=False, init_identity=init_identity)

    def forward(self, z, context=None):
        z, log_det = self.mprqat.inverse(z, context=context)
        return z, log_det.view(-1)

    def inverse(self, z, context=None):
        z, log_det = self.mprqat(z, context=context)
        return z, log_det.view(-1)


class CircularAutoregressiveRationalQuadraticSpline(Flow):
    """
    Neural spline flow coupling layer, wrapper for the implementation
    of Durkan et al., see [sources](https://github.com/bayesiains/nsf)
    """

    def __init__(self, num_input_channels, num_blocks, num_hidden_channels, ind_circ, num_context_channels=None, num_bins=8, tail_bound=3, activation=nn.ReLU, dropout_probability=0.0, permute_mask=True, init_identity=True):
        """Constructor

        Args:
          num_input_channels (int): Flow dimension
          num_blocks (int): Number of residual blocks of the parameter NN
          num_hidden_channels (int): Number of hidden units of the NN
          ind_circ (Iterable): Indices of the circular coordinates
          num_context_channels (int): Number of context/conditional channels
          num_bins (int): Number of bins
          tail_bound (int): Bound of the spline tails
          activation (torch module): Activation function
          dropout_probability (float): Dropout probability of the NN
          permute_mask (bool): Flag, permutes the mask of the NN
          init_identity (bool): Flag, initialize transform as identity
        """
        super().__init__()
        tails = [('circular' if i in ind_circ else 'linear') for i in range(num_input_channels)]
        self.mprqat = MaskedPiecewiseRationalQuadraticAutoregressive(features=num_input_channels, hidden_features=num_hidden_channels, context_features=num_context_channels, num_bins=num_bins, tails=tails, tail_bound=tail_bound, num_blocks=num_blocks, use_residual_blocks=True, random_mask=False, permute_mask=permute_mask, activation=activation(), dropout_probability=dropout_probability, use_batch_norm=False, init_identity=init_identity)

    def forward(self, z, context=None):
        z, log_det = self.mprqat.inverse(z, context=context)
        return z, log_det.view(-1)

    def inverse(self, z, context=None):
        z, log_det = self.mprqat(z, context=context)
        return z, log_det.view(-1)


class BatchNorm(Flow):
    """
    Batch Normalization with out considering the derivatives of the batch statistics, see [arXiv: 1605.08803](https://arxiv.org/abs/1605.08803)
    """

    def __init__(self, eps=1e-10):
        super().__init__()
        self.eps_cpu = torch.tensor(eps)
        self.register_buffer('eps', self.eps_cpu)

    def forward(self, z):
        """
        Do batch norm over batch and sample dimension
        """
        mean = torch.mean(z, dim=0, keepdims=True)
        std = torch.std(z, dim=0, keepdims=True)
        z_ = (z - mean) / torch.sqrt(std ** 2 + self.eps)
        log_det = torch.log(1 / torch.prod(torch.sqrt(std ** 2 + self.eps))).repeat(z.size()[0])
        return z_, log_det


class PeriodicWrap(Flow):
    """
    Map periodic coordinates to fixed interval
    """

    def __init__(self, ind, bound=1.0):
        """Constructor

        ind: Iterable, indices of coordinates to be mapped
        bound: Float or iterable, bound of interval
        """
        super().__init__()
        self.ind = ind
        if torch.is_tensor(bound):
            self.register_buffer('bound', bound)
        else:
            self.bound = bound

    def forward(self, z):
        return z, torch.zeros(len(z), dtype=z.dtype, device=z.device)

    def inverse(self, z):
        z_ = z.clone()
        z_[..., self.ind] = torch.remainder(z_[..., self.ind] + self.bound, 2 * self.bound) - self.bound
        return z_, torch.zeros(len(z), dtype=z.dtype, device=z.device)


class PeriodicShift(Flow):
    """
    Shift and wrap periodic coordinates
    """

    def __init__(self, ind, bound=1.0, shift=0.0):
        """Constructor

        Args:
          ind: Iterable, indices of coordinates to be mapped
          bound: Float or iterable, bound of interval
          shift: Tensor, shift to be applied
        """
        super().__init__()
        self.ind = ind
        if torch.is_tensor(bound):
            self.register_buffer('bound', bound)
        else:
            self.bound = bound
        if torch.is_tensor(shift):
            self.register_buffer('shift', shift)
        else:
            self.shift = shift

    def forward(self, z):
        z_ = z.clone()
        z_[..., self.ind] = torch.remainder(z_[..., self.ind] + self.shift + self.bound, 2 * self.bound) - self.bound
        return z_, torch.zeros(len(z), dtype=z.dtype, device=z.device)

    def inverse(self, z):
        z_ = z.clone()
        z_[..., self.ind] = torch.remainder(z_[..., self.ind] - self.shift + self.bound, 2 * self.bound) - self.bound
        return z_, torch.zeros(len(z), dtype=z.dtype, device=z.device)


class Planar(Flow):
    """Planar flow as introduced in [arXiv: 1505.05770](https://arxiv.org/abs/1505.05770)

    ```
        f(z) = z + u * h(w * z + b)
    ```
    """

    def __init__(self, shape, act='tanh', u=None, w=None, b=None):
        """Constructor of the planar flow

        Args:
          shape: shape of the latent variable z
          h: nonlinear function h of the planar flow (see definition of f above)
          u,w,b: optional initialization for parameters
        """
        super().__init__()
        lim_w = np.sqrt(2.0 / np.prod(shape))
        lim_u = np.sqrt(2)
        if u is not None:
            self.u = nn.Parameter(u)
        else:
            self.u = nn.Parameter(torch.empty(shape)[None])
            nn.init.uniform_(self.u, -lim_u, lim_u)
        if w is not None:
            self.w = nn.Parameter(w)
        else:
            self.w = nn.Parameter(torch.empty(shape)[None])
            nn.init.uniform_(self.w, -lim_w, lim_w)
        if b is not None:
            self.b = nn.Parameter(b)
        else:
            self.b = nn.Parameter(torch.zeros(1))
        self.act = act
        if act == 'tanh':
            self.h = torch.tanh
        elif act == 'leaky_relu':
            self.h = torch.nn.LeakyReLU(negative_slope=0.2)
        else:
            raise NotImplementedError('Nonlinearity is not implemented.')

    def forward(self, z):
        lin = torch.sum(self.w * z, list(range(1, self.w.dim())), keepdim=True) + self.b
        inner = torch.sum(self.w * self.u)
        u = self.u + (torch.log(1 + torch.exp(inner)) - 1 - inner) * self.w / torch.sum(self.w ** 2)
        if self.act == 'tanh':
            h_ = lambda x: 1 / torch.cosh(x) ** 2
        elif self.act == 'leaky_relu':
            h_ = lambda x: (x < 0) * (self.h.negative_slope - 1.0) + 1.0
        z_ = z + u * self.h(lin)
        log_det = torch.log(torch.abs(1 + torch.sum(self.w * u) * h_(lin.reshape(-1))))
        return z_, log_det

    def inverse(self, z):
        if self.act != 'leaky_relu':
            raise NotImplementedError('This flow has no algebraic inverse.')
        lin = torch.sum(self.w * z, list(range(1, self.w.dim()))) + self.b
        a = (lin < 0) * (self.h.negative_slope - 1.0) + 1.0
        inner = torch.sum(self.w * self.u)
        u = self.u + (torch.log(1 + torch.exp(inner)) - 1 - inner) * self.w / torch.sum(self.w ** 2)
        dims = [-1] + (u.dim() - 1) * [1]
        u = a.reshape(*dims) * u
        inner_ = torch.sum(self.w * u, list(range(1, self.w.dim())))
        z_ = z - u * (lin / (1 + inner_)).reshape(*dims)
        log_det = -torch.log(torch.abs(1 + inner_))
        return z_, log_det


class Radial(Flow):
    """Radial flow as introduced in [arXiv: 1505.05770](https://arxiv.org/abs/1505.05770)

    ```
        f(z) = z + beta * h(alpha, r) * (z - z_0)
    ```
    """

    def __init__(self, shape, z_0=None):
        """Constructor of the radial flow

        Args:
          shape: shape of the latent variable z
          z_0: parameter of the radial flow
        """
        super().__init__()
        self.d_cpu = torch.prod(torch.tensor(shape))
        self.register_buffer('d', self.d_cpu)
        self.beta = nn.Parameter(torch.empty(1))
        lim = 1.0 / np.prod(shape)
        nn.init.uniform_(self.beta, -lim - 1.0, lim - 1.0)
        self.alpha = nn.Parameter(torch.empty(1))
        nn.init.uniform_(self.alpha, -lim, lim)
        if z_0 is not None:
            self.z_0 = nn.Parameter(z_0)
        else:
            self.z_0 = nn.Parameter(torch.randn(shape)[None])

    def forward(self, z):
        beta = torch.log(1 + torch.exp(self.beta)) - torch.abs(self.alpha)
        dz = z - self.z_0
        r = torch.linalg.vector_norm(dz, dim=list(range(1, self.z_0.dim())), keepdim=True)
        h_arr = beta / (torch.abs(self.alpha) + r)
        h_arr_ = -beta * r / (torch.abs(self.alpha) + r) ** 2
        z_ = z + h_arr * dz
        log_det = (self.d - 1) * torch.log(1 + h_arr) + torch.log(1 + h_arr + h_arr_)
        log_det = log_det.reshape(-1)
        return z_, log_det


class Merge(Split):
    """
    Same as Split but with forward and backward pass interchanged
    """

    def __init__(self, mode='channel'):
        super().__init__(mode)

    def forward(self, z):
        return super().inverse(z)

    def inverse(self, z):
        return super().forward(z)


class Squeeze(Flow):
    """
    Squeeze operation of multi-scale architecture, RealNVP or Glow paper
    """

    def __init__(self):
        """
        Constructor
        """
        super().__init__()

    def forward(self, z):
        log_det = 0
        s = z.size()
        z = z.view(s[0], s[1] // 4, 2, 2, s[2], s[3])
        z = z.permute(0, 1, 4, 2, 5, 3).contiguous()
        z = z.view(s[0], s[1] // 4, 2 * s[2], 2 * s[3])
        return z, log_det

    def inverse(self, z):
        log_det = 0
        s = z.size()
        z = z.view(*s[:2], s[2] // 2, 2, s[3] // 2, 2)
        z = z.permute(0, 1, 3, 5, 2, 4).contiguous()
        z = z.view(s[0], 4 * s[1], s[2] // 2, s[3] // 2)
        return z, log_det


def basic_logdet_estimator(g, x, n_power_series, vareps, coeff_fn, training):
    vjp = vareps
    logdetgrad = torch.tensor(0.0)
    for k in range(1, n_power_series + 1):
        vjp = torch.autograd.grad(g, x, vjp, create_graph=training, retain_graph=True)[0]
        tr = torch.sum(vjp.view(x.shape[0], -1) * vareps.view(x.shape[0], -1), 1)
        delta = (-1) ** (k + 1) / k * coeff_fn(k) * tr
        logdetgrad = logdetgrad + delta
    return logdetgrad


def batch_jacobian(g, x):
    jac = []
    for d in range(g.shape[1]):
        jac.append(torch.autograd.grad(torch.sum(g[:, d]), x, create_graph=True)[0].view(x.shape[0], 1, x.shape[1]))
    return torch.cat(jac, 1)


def batch_trace(M):
    return M.view(M.shape[0], -1)[:, ::M.shape[1] + 1].sum(1)


def geometric_1mcdf(p, k, offset):
    if k <= offset:
        return 1.0
    else:
        k = k - offset
    """P(n >= k)"""
    return (1 - p) ** max(k - 1, 0)


def geometric_sample(p, n_samples):
    return np.random.geometric(p, n_samples)


def safe_detach(tensor):
    return tensor.detach().requires_grad_(tensor.requires_grad)


class MemoryEfficientLogDetEstimator(torch.autograd.Function):

    @staticmethod
    def forward(ctx, estimator_fn, gnet, x, n_power_series, vareps, coeff_fn, training, *g_params):
        ctx.training = training
        with torch.enable_grad():
            x = x.detach().requires_grad_(True)
            g = gnet(x)
            ctx.g = g
            ctx.x = x
            logdetgrad = estimator_fn(g, x, n_power_series, vareps, coeff_fn, training)
            if training:
                grad_x, *grad_params = torch.autograd.grad(logdetgrad.sum(), (x,) + g_params, retain_graph=True, allow_unused=True)
                if grad_x is None:
                    grad_x = torch.zeros_like(x)
                ctx.save_for_backward(grad_x, *g_params, *grad_params)
        return safe_detach(g), safe_detach(logdetgrad)

    @staticmethod
    def backward(ctx, grad_g, grad_logdetgrad):
        training = ctx.training
        if not training:
            raise ValueError('Provide training=True if using backward.')
        with torch.enable_grad():
            grad_x, *params_and_grad = ctx.saved_tensors
            g, x = ctx.g, ctx.x
            g_params = params_and_grad[:len(params_and_grad) // 2]
            grad_params = params_and_grad[len(params_and_grad) // 2:]
            dg_x, *dg_params = torch.autograd.grad(g, [x] + g_params, grad_g, allow_unused=True)
        dL = grad_logdetgrad[0].detach()
        with torch.no_grad():
            grad_x.mul_(dL)
            grad_params = tuple([(g.mul_(dL) if g is not None else None) for g in grad_params])
        with torch.no_grad():
            grad_x.add_(dg_x)
            grad_params = tuple([(dg.add_(djac) if djac is not None else dg) for dg, djac in zip(dg_params, grad_params)])
        return (None, None, grad_x, None, None, None, None) + grad_params


def mem_eff_wrapper(estimator_fn, gnet, x, n_power_series, vareps, coeff_fn, training):
    if not isinstance(gnet, nn.Module):
        raise ValueError('g is required to be an instance of nn.Module.')
    return MemoryEfficientLogDetEstimator.apply(estimator_fn, gnet, x, n_power_series, vareps, coeff_fn, training, *list(gnet.parameters()))


def neumann_logdet_estimator(g, x, n_power_series, vareps, coeff_fn, training):
    vjp = vareps
    neumann_vjp = vareps
    with torch.no_grad():
        for k in range(1, n_power_series + 1):
            vjp = torch.autograd.grad(g, x, vjp, retain_graph=True)[0]
            neumann_vjp = neumann_vjp + (-1) ** k * coeff_fn(k) * vjp
    vjp_jac = torch.autograd.grad(g, x, neumann_vjp, create_graph=training)[0]
    logdetgrad = torch.sum(vjp_jac.view(x.shape[0], -1) * vareps.view(x.shape[0], -1), 1)
    return logdetgrad


def poisson_1mcdf(lamb, k, offset):
    if k <= offset:
        return 1.0
    else:
        k = k - offset
    """P(n >= k)"""
    sum = 1.0
    for i in range(1, k):
        sum += lamb ** i / math.factorial(i)
    return 1 - np.exp(-lamb) * sum


def poisson_sample(lamb, n_samples):
    return np.random.poisson(lamb, n_samples)


class iResBlock(nn.Module):

    def __init__(self, nnet, geom_p=0.5, lamb=2.0, n_power_series=None, exact_trace=False, brute_force=False, n_samples=1, n_exact_terms=2, n_dist='geometric', neumann_grad=True, grad_in_forward=False):
        """
        Args:
            nnet: a nn.Module
            n_power_series: number of power series. If not None, uses a biased approximation to logdet.
            exact_trace: if False, uses a Hutchinson trace estimator. Otherwise computes the exact full Jacobian.
            brute_force: Computes the exact logdet. Only available for 2D inputs.
        """
        nn.Module.__init__(self)
        self.nnet = nnet
        self.n_dist = n_dist
        self.geom_p = nn.Parameter(torch.tensor(np.log(geom_p) - np.log(1.0 - geom_p)))
        self.lamb = nn.Parameter(torch.tensor(lamb))
        self.n_samples = n_samples
        self.n_power_series = n_power_series
        self.exact_trace = exact_trace
        self.brute_force = brute_force
        self.n_exact_terms = n_exact_terms
        self.grad_in_forward = grad_in_forward
        self.neumann_grad = neumann_grad
        self.register_buffer('last_n_samples', torch.zeros(self.n_samples))
        self.register_buffer('last_firmom', torch.zeros(1))
        self.register_buffer('last_secmom', torch.zeros(1))

    def forward(self, x, logpx=None):
        if logpx is None:
            y = x + self.nnet(x)
            return y
        else:
            g, logdetgrad = self._logdetgrad(x)
            return x + g, logpx - logdetgrad

    def inverse(self, y, logpy=None):
        x = self._inverse_fixed_point(y)
        if logpy is None:
            return x
        else:
            return x, logpy + self._logdetgrad(x)[1]

    def _inverse_fixed_point(self, y, atol=1e-05, rtol=1e-05):
        x, x_prev = y - self.nnet(y), y
        i = 0
        tol = atol + y.abs() * rtol
        while not torch.all((x - x_prev) ** 2 / tol < 1):
            x, x_prev = y - self.nnet(x), x
            i += 1
            if i > 1000:
                break
        return x

    def _logdetgrad(self, x):
        """Returns g(x) and ```logdet|d(x+g(x))/dx|```"""
        with torch.enable_grad():
            if (self.brute_force or not self.training) and (x.ndimension() == 2 and x.shape[1] == 2):
                x = x.requires_grad_(True)
                g = self.nnet(x)
                jac = batch_jacobian(g, x)
                batch_dets = (jac[:, 0, 0] + 1) * (jac[:, 1, 1] + 1) - jac[:, 0, 1] * jac[:, 1, 0]
                return g, torch.log(torch.abs(batch_dets)).view(-1, 1)
            if self.n_dist == 'geometric':
                geom_p = torch.sigmoid(self.geom_p).item()
                sample_fn = lambda m: geometric_sample(geom_p, m)
                rcdf_fn = lambda k, offset: geometric_1mcdf(geom_p, k, offset)
            elif self.n_dist == 'poisson':
                lamb = self.lamb.item()
                sample_fn = lambda m: poisson_sample(lamb, m)
                rcdf_fn = lambda k, offset: poisson_1mcdf(lamb, k, offset)
            if self.training:
                if self.n_power_series is None:
                    lamb = self.lamb.item()
                    n_samples = sample_fn(self.n_samples)
                    n_power_series = max(n_samples) + self.n_exact_terms
                    coeff_fn = lambda k: 1 / rcdf_fn(k, self.n_exact_terms) * sum(n_samples >= k - self.n_exact_terms) / len(n_samples)
                else:
                    n_power_series = self.n_power_series
                    coeff_fn = lambda k: 1.0
            else:
                lamb = self.lamb.item()
                n_samples = sample_fn(self.n_samples)
                n_power_series = max(n_samples) + 20
                coeff_fn = lambda k: 1 / rcdf_fn(k, 20) * sum(n_samples >= k - 20) / len(n_samples)
            if not self.exact_trace:
                vareps = torch.randn_like(x)
                if self.training and self.neumann_grad:
                    estimator_fn = neumann_logdet_estimator
                else:
                    estimator_fn = basic_logdet_estimator
                if self.training and self.grad_in_forward:
                    g, logdetgrad = mem_eff_wrapper(estimator_fn, self.nnet, x, n_power_series, vareps, coeff_fn, self.training)
                else:
                    x = x.requires_grad_(True)
                    g = self.nnet(x)
                    logdetgrad = estimator_fn(g, x, n_power_series, vareps, coeff_fn, self.training)
            else:
                x = x.requires_grad_(True)
                g = self.nnet(x)
                jac = batch_jacobian(g, x)
                logdetgrad = batch_trace(jac)
                jac_k = jac
                for k in range(2, n_power_series + 1):
                    jac_k = torch.bmm(jac, jac_k)
                    logdetgrad = logdetgrad + (-1) ** (k + 1) / k * coeff_fn(k) * batch_trace(jac_k)
            if self.training and self.n_power_series is None:
                self.last_n_samples.copy_(torch.tensor(n_samples))
                estimator = logdetgrad.detach()
                self.last_firmom.copy_(torch.mean(estimator))
                self.last_secmom.copy_(torch.mean(estimator ** 2))
            return g, logdetgrad.view(-1, 1)

    def extra_repr(self):
        return 'dist={}, n_samples={}, n_power_series={}, neumann_grad={}, exact_trace={}, brute_force={}'.format(self.n_dist, self.n_samples, self.n_power_series, self.neumann_grad, self.exact_trace, self.brute_force)


class Residual(Flow):
    """
    Invertible residual net block, wrapper to the implementation of Chen et al.,
    see [sources](https://github.com/rtqichen/residual-flows)
    """

    def __init__(self, net, reverse=True, reduce_memory=True, geom_p=0.5, lamb=2.0, n_power_series=None, exact_trace=False, brute_force=False, n_samples=1, n_exact_terms=2, n_dist='geometric'):
        """Constructor

        Args:
          net: Neural network, must be Lipschitz continuous with L < 1
          reverse: Flag, if true the map ```f(x) = x + net(x)``` is applied in the inverse pass, otherwise it is done in forward
          reduce_memory: Flag, if true Neumann series and precomputations, for backward pass in forward pass are done
          geom_p: Parameter of the geometric distribution used for the Neumann series
          lamb: Parameter of the geometric distribution used for the Neumann series
          n_power_series: Number of terms in the Neumann series
          exact_trace: Flag, if true the trace of the Jacobian is computed exactly
          brute_force: Flag, if true the Jacobian is computed exactly in 2D
          n_samples: Number of samples used to estimate power series
          n_exact_terms: Number of terms always included in the power series
          n_dist: Distribution used for the power series, either "geometric" or "poisson"
        """
        super().__init__()
        self.reverse = reverse
        self.iresblock = iResBlock(net, n_samples=n_samples, n_exact_terms=n_exact_terms, neumann_grad=reduce_memory, grad_in_forward=reduce_memory, exact_trace=exact_trace, geom_p=geom_p, lamb=lamb, n_power_series=n_power_series, brute_force=brute_force, n_dist=n_dist)

    def forward(self, z):
        if self.reverse:
            z, log_det = self.iresblock.inverse(z, 0)
        else:
            z, log_det = self.iresblock.forward(z, 0)
        return z, -log_det.view(-1)

    def inverse(self, z):
        if self.reverse:
            z, log_det = self.iresblock.forward(z, 0)
        else:
            z, log_det = self.iresblock.inverse(z, 0)
        return z, -log_det.view(-1)


class MetropolisHastings(Flow):
    """Sampling through Metropolis Hastings in Stochastic Normalizing Flow

    See [arXiv: 2002.06707](https://arxiv.org/abs/2002.06707)
    """

    def __init__(self, target, proposal, steps):
        """Constructor

        Args:
          target: The stationary distribution of this Markov transition, i.e. the target distribution to sample from.
          proposal: Proposal distribution
          steps: Number of MCMC steps to perform
        """
        super().__init__()
        self.target = target
        self.proposal = proposal
        self.steps = steps

    def forward(self, z):
        num_samples = len(z)
        log_det = torch.zeros(num_samples, dtype=z.dtype, device=z.device)
        log_p = self.target.log_prob(z)
        for i in range(self.steps):
            z_, log_p_diff = self.proposal(z)
            log_p_ = self.target.log_prob(z_)
            w = torch.rand(num_samples, dtype=z.dtype, device=z.device)
            log_w_accept = log_p_ - log_p + log_p_diff
            w_accept = torch.clamp(torch.exp(log_w_accept), max=1)
            accept = w <= w_accept
            z = torch.where(accept.unsqueeze(1), z_, z)
            log_det_ = log_p - log_p_
            log_det = torch.where(accept, log_det + log_det_, log_det)
            log_p = torch.where(accept, log_p_, log_p)
        return z, log_det

    def inverse(self, z):
        return self.forward(z)


class HamiltonianMonteCarlo(Flow):
    """Flow layer using the HMC proposal in Stochastic Normalising Flows

    See [arXiv: 2002.06707](https://arxiv.org/abs/2002.06707)
    """

    def __init__(self, target, steps, log_step_size, log_mass, max_abs_grad=None):
        """Constructor

        Args:
          target: The stationary distribution of this Markov transition, i.e. the target distribution to sample from.
          steps: The number of leapfrog steps
          log_step_size: The log step size used in the leapfrog integrator. shape (dim)
          log_mass: The log_mass determining the variance of the momentum samples. shape (dim)
          max_abs_grad: Maximum absolute value of the gradient of the target distribution's log probability. If set to None then no gradient clipping is applied. Useful for improving numerical stability."""
        super().__init__()
        self.target = target
        self.steps = steps
        self.register_parameter('log_step_size', torch.nn.Parameter(log_step_size))
        self.register_parameter('log_mass', torch.nn.Parameter(log_mass))
        self.max_abs_grad = max_abs_grad

    def forward(self, z):
        p = torch.randn_like(z) * torch.exp(0.5 * self.log_mass)
        z_new = z.clone()
        p_new = p.clone()
        step_size = torch.exp(self.log_step_size)
        for i in range(self.steps):
            p_half = p_new - step_size / 2.0 * -self.gradlogP(z_new)
            z_new = z_new + step_size * (p_half / torch.exp(self.log_mass))
            p_new = p_half - step_size / 2.0 * -self.gradlogP(z_new)
        probabilities = torch.exp(self.target.log_prob(z_new) - self.target.log_prob(z) - 0.5 * torch.sum(p_new ** 2 / torch.exp(self.log_mass), 1) + 0.5 * torch.sum(p ** 2 / torch.exp(self.log_mass), 1))
        uniforms = torch.rand_like(probabilities)
        mask = uniforms < probabilities
        z_out = torch.where(mask.unsqueeze(1), z_new, z)
        return z_out, self.target.log_prob(z) - self.target.log_prob(z_out)

    def inverse(self, z):
        return self.forward(z)

    def gradlogP(self, z):
        z_ = z.detach().requires_grad_()
        logp = self.target.log_prob(z_)
        grad = torch.autograd.grad(logp, z_, grad_outputs=torch.ones_like(logp))[0]
        if self.max_abs_grad:
            grad = torch.clamp(grad, max=self.max_abs_grad, min=-self.max_abs_grad)
        return grad


class ConvNet2d(nn.Module):
    """
    Convolutional Neural Network with leaky ReLU nonlinearities
    """

    def __init__(self, channels, kernel_size, leaky=0.0, init_zeros=True, actnorm=False, weight_std=None):
        """Constructor

        Args:
          channels: List of channels of conv layers, first entry is in_channels
          kernel_size: List of kernel sizes, same for height and width
          leaky: Leaky part of ReLU
          init_zeros: Flag whether last layer shall be initialized with zeros
          scale_output: Flag whether to scale output with a log scale parameter
          logscale_factor: Constant factor to be multiplied to log scaling
          actnorm: Flag whether activation normalization shall be done after each conv layer except output
          weight_std: Fixed std used to initialize every layer
        """
        super().__init__()
        net = nn.ModuleList([])
        for i in range(len(kernel_size) - 1):
            conv = nn.Conv2d(channels[i], channels[i + 1], kernel_size[i], padding=kernel_size[i] // 2, bias=not actnorm)
            if weight_std is not None:
                conv.weight.data.normal_(mean=0.0, std=weight_std)
            net.append(conv)
            if actnorm:
                net.append(utils.ActNorm((channels[i + 1],) + (1, 1)))
            net.append(nn.LeakyReLU(leaky))
        i = len(kernel_size)
        net.append(nn.Conv2d(channels[i - 1], channels[i], kernel_size[i - 1], padding=kernel_size[i - 1] // 2))
        if init_zeros:
            nn.init.zeros_(net[-1].weight)
            nn.init.zeros_(net[-1].bias)
        self.net = nn.Sequential(*net)

    def forward(self, x):
        return self.net(x)


def leaky_elu(x, a=0.3):
    return a * x + (1 - a) * F.elu(x)


def asym_squash(x):
    return torch.tanh(-leaky_elu(-x + 0.5493061829986572)) * 2 + 3


def projmax_(v):
    """Inplace argmax on absolute value."""
    ind = torch.argmax(torch.abs(v))
    v.zero_()
    v[ind] = 1
    return v


def vector_norm(x, p):
    x = x.view(-1)
    return torch.sum(x ** p) ** (1 / p)


def normalize_u(u, codomain, out=None):
    if not torch.is_tensor(codomain) and codomain == 2:
        u = F.normalize(u, p=2, dim=0, out=out)
    elif codomain == float('inf'):
        u = projmax_(u)
    else:
        uabs = torch.abs(u)
        uph = u / uabs
        uph[torch.isnan(uph)] = 1
        uabs = uabs / torch.max(uabs)
        uabs = uabs ** (codomain - 1)
        if codomain == 1:
            u = uph * uabs / vector_norm(uabs, float('inf'))
        else:
            u = uph * uabs / vector_norm(uabs, codomain / (codomain - 1))
    return u


def normalize_v(v, domain, out=None):
    if not torch.is_tensor(domain) and domain == 2:
        v = F.normalize(v, p=2, dim=0, out=out)
    elif domain == 1:
        v = projmax_(v)
    else:
        vabs = torch.abs(v)
        vph = v / vabs
        vph[torch.isnan(vph)] = 1
        vabs = vabs / torch.max(vabs)
        vabs = vabs ** (1 / (domain - 1))
        v = vph * vabs / vector_norm(vabs, domain)
    return v


class InducedNormLinear(nn.Module):

    def __init__(self, in_features, out_features, bias=True, coeff=0.97, domain=2, codomain=2, n_iterations=None, atol=None, rtol=None, zero_init=False, **unused_kwargs):
        del unused_kwargs
        super(InducedNormLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.coeff = coeff
        self.n_iterations = n_iterations
        self.atol = atol
        self.rtol = rtol
        self.domain = domain
        self.codomain = codomain
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters(zero_init)
        with torch.no_grad():
            domain, codomain = self.compute_domain_codomain()
        h, w = self.weight.shape
        self.register_buffer('scale', torch.tensor(0.0))
        self.register_buffer('u', normalize_u(self.weight.new_empty(h).normal_(0, 1), codomain))
        self.register_buffer('v', normalize_v(self.weight.new_empty(w).normal_(0, 1), domain))
        with torch.no_grad():
            self.compute_weight(True, n_iterations=200, atol=None, rtol=None)
            best_scale = self.scale.clone()
            best_u, best_v = self.u.clone(), self.v.clone()
            if not (domain == 2 and codomain == 2):
                for _ in range(10):
                    self.register_buffer('u', normalize_u(self.weight.new_empty(h).normal_(0, 1), codomain))
                    self.register_buffer('v', normalize_v(self.weight.new_empty(w).normal_(0, 1), domain))
                    self.compute_weight(True, n_iterations=200)
                    if self.scale > best_scale:
                        best_u, best_v = self.u.clone(), self.v.clone()
            self.u.copy_(best_u)
            self.v.copy_(best_v)

    def reset_parameters(self, zero_init=False):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if zero_init:
            self.weight.data.div_(1000)
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def compute_domain_codomain(self):
        if torch.is_tensor(self.domain):
            domain = asym_squash(self.domain)
            codomain = asym_squash(self.codomain)
        else:
            domain, codomain = self.domain, self.codomain
        return domain, codomain

    def compute_one_iter(self):
        domain, codomain = self.compute_domain_codomain()
        u = self.u.detach()
        v = self.v.detach()
        weight = self.weight.detach()
        u = normalize_u(torch.mv(weight, v), codomain)
        v = normalize_v(torch.mv(weight.t(), u), domain)
        return torch.dot(u, torch.mv(weight, v))

    def compute_weight(self, update=True, n_iterations=None, atol=None, rtol=None):
        u = self.u
        v = self.v
        weight = self.weight
        if update:
            n_iterations = self.n_iterations if n_iterations is None else n_iterations
            atol = self.atol if atol is None else atol
            rtol = self.rtol if rtol is None else atol
            if n_iterations is None and (atol is None or rtol is None):
                raise ValueError('Need one of n_iteration or (atol, rtol).')
            max_itrs = 200
            if n_iterations is not None:
                max_itrs = n_iterations
            with torch.no_grad():
                domain, codomain = self.compute_domain_codomain()
                for _ in range(max_itrs):
                    if n_iterations is None and atol is not None and rtol is not None:
                        old_v = v.clone()
                        old_u = u.clone()
                    u = normalize_u(torch.mv(weight, v), codomain, out=u)
                    v = normalize_v(torch.mv(weight.t(), u), domain, out=v)
                    if n_iterations is None and atol is not None and rtol is not None:
                        err_u = torch.norm(u - old_u) / u.nelement() ** 0.5
                        err_v = torch.norm(v - old_v) / v.nelement() ** 0.5
                        tol_u = atol + rtol * torch.max(u)
                        tol_v = atol + rtol * torch.max(v)
                        if err_u < tol_u and err_v < tol_v:
                            break
                self.v.copy_(v)
                self.u.copy_(u)
                u = u.clone()
                v = v.clone()
        sigma = torch.dot(u, torch.mv(weight, v))
        with torch.no_grad():
            self.scale.copy_(sigma)
        factor = torch.max(torch.ones(1), sigma / self.coeff)
        weight = weight / factor
        return weight

    def forward(self, input):
        weight = self.compute_weight(update=False)
        return F.linear(input, weight, self.bias)

    def extra_repr(self):
        domain, codomain = self.compute_domain_codomain()
        return 'in_features={}, out_features={}, bias={}, coeff={}, domain={:.2f}, codomain={:.2f}, n_iters={}, atol={}, rtol={}, learnable_ord={}'.format(self.in_features, self.out_features, self.bias is not None, self.coeff, domain, codomain, self.n_iterations, self.atol, self.rtol, torch.is_tensor(self.domain))


class Swish(nn.Module):

    def __init__(self):
        super(Swish, self).__init__()
        self.beta = nn.Parameter(torch.tensor([0.5]))

    def forward(self, x):
        return (x * torch.sigmoid_(x * F.softplus(self.beta))).div_(1.1)


class LipschitzMLP(nn.Module):
    """Fully connected neural net which is Lipschitz continuou with Lipschitz constant L < 1"""

    def __init__(self, channels, lipschitz_const=0.97, max_lipschitz_iter=5, lipschitz_tolerance=None, init_zeros=True):
        """
        Constructor
          channels: Integer list with the number of channels of
        the layers
          lipschitz_const: Maximum Lipschitz constant of each layer
          max_lipschitz_iter: Maximum number of iterations used to
        ensure that layers are Lipschitz continuous with L smaller than
        set maximum; if None, tolerance is used
          lipschitz_tolerance: Float, tolerance used to ensure
        Lipschitz continuity if max_lipschitz_iter is None, typically 1e-3
          init_zeros: Flag, whether to initialize last layer
        approximately with zeros
        """
        super().__init__()
        self.n_layers = len(channels) - 1
        self.channels = channels
        self.lipschitz_const = lipschitz_const
        self.max_lipschitz_iter = max_lipschitz_iter
        self.lipschitz_tolerance = lipschitz_tolerance
        self.init_zeros = init_zeros
        layers = []
        for i in range(self.n_layers):
            layers += [Swish(), InducedNormLinear(in_features=channels[i], out_features=channels[i + 1], coeff=lipschitz_const, domain=2, codomain=2, n_iterations=max_lipschitz_iter, atol=lipschitz_tolerance, rtol=lipschitz_tolerance, zero_init=init_zeros if i == self.n_layers - 1 else False)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def _ntuple(n):

    def parse(x):
        if isinstance(x, container_abcs.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse


_pair = _ntuple(2)


class InducedNormConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True, coeff=0.97, domain=2, codomain=2, n_iterations=None, atol=None, rtol=None, zero_init=False, **unused_kwargs):
        del unused_kwargs
        super(InducedNormConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.coeff = coeff
        self.n_iterations = n_iterations
        self.domain = domain
        self.codomain = codomain
        self.atol = atol
        self.rtol = rtol
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, *self.kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters(zero_init)
        self.register_buffer('initialized', torch.tensor(0))
        self.register_buffer('spatial_dims', torch.tensor([1.0, 1.0]))
        self.register_buffer('scale', torch.tensor(0.0))
        self.register_buffer('u', self.weight.new_empty(self.out_channels))
        self.register_buffer('v', self.weight.new_empty(self.in_channels))

    def compute_domain_codomain(self):
        if torch.is_tensor(self.domain):
            domain = asym_squash(self.domain)
            codomain = asym_squash(self.codomain)
        else:
            domain, codomain = self.domain, self.codomain
        return domain, codomain

    def reset_parameters(self, zero_init=False):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if zero_init:
            self.weight.data.div_(1000)
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def _initialize_u_v(self):
        with torch.no_grad():
            domain, codomain = self.compute_domain_codomain()
            if self.kernel_size == (1, 1):
                self.u.resize_(self.out_channels).normal_(0, 1)
                self.u.copy_(normalize_u(self.u, codomain))
                self.v.resize_(self.in_channels).normal_(0, 1)
                self.v.copy_(normalize_v(self.v, domain))
            else:
                c, h, w = self.in_channels, int(self.spatial_dims[0].item()), int(self.spatial_dims[1].item())
                with torch.no_grad():
                    num_input_dim = c * h * w
                    self.v.resize_(num_input_dim).normal_(0, 1)
                    self.v.copy_(normalize_v(self.v, domain))
                    u = F.conv2d(self.v.view(1, c, h, w), self.weight, stride=self.stride, padding=self.padding, bias=None)
                    num_output_dim = u.shape[0] * u.shape[1] * u.shape[2] * u.shape[3]
                    self.u.resize_(num_output_dim).normal_(0, 1)
                    self.u.copy_(normalize_u(self.u, codomain))
            self.initialized.fill_(1)
            self.compute_weight(True)
            best_scale = self.scale.clone()
            best_u, best_v = self.u.clone(), self.v.clone()
            if not (domain == 2 and codomain == 2):
                for _ in range(10):
                    if self.kernel_size == (1, 1):
                        self.u.copy_(normalize_u(self.weight.new_empty(self.out_channels).normal_(0, 1), codomain))
                        self.v.copy_(normalize_v(self.weight.new_empty(self.in_channels).normal_(0, 1), domain))
                    else:
                        self.u.copy_(normalize_u(torch.randn(num_output_dim), codomain))
                        self.v.copy_(normalize_v(torch.randn(num_input_dim), domain))
                    self.compute_weight(True, n_iterations=200)
                    if self.scale > best_scale:
                        best_u, best_v = self.u.clone(), self.v.clone()
            self.u.copy_(best_u)
            self.v.copy_(best_v)

    def compute_one_iter(self):
        if not self.initialized:
            raise ValueError('Layer needs to be initialized first.')
        domain, codomain = self.compute_domain_codomain()
        if self.kernel_size == (1, 1):
            u = self.u.detach()
            v = self.v.detach()
            weight = self.weight.detach().view(self.out_channels, self.in_channels)
            u = normalize_u(torch.mv(weight, v), codomain)
            v = normalize_v(torch.mv(weight.t(), u), domain)
            return torch.dot(u, torch.mv(weight, v))
        else:
            u = self.u.detach()
            v = self.v.detach()
            weight = self.weight.detach()
            c, h, w = self.in_channels, int(self.spatial_dims[0].item()), int(self.spatial_dims[1].item())
            u_s = F.conv2d(v.view(1, c, h, w), weight, stride=self.stride, padding=self.padding, bias=None)
            out_shape = u_s.shape
            u = normalize_u(u_s.view(-1), codomain)
            v_s = F.conv_transpose2d(u.view(out_shape), weight, stride=self.stride, padding=self.padding, output_padding=0)
            v = normalize_v(v_s.view(-1), domain)
            weight_v = F.conv2d(v.view(1, c, h, w), weight, stride=self.stride, padding=self.padding, bias=None)
            return torch.dot(u.view(-1), weight_v.view(-1))

    def compute_weight(self, update=True, n_iterations=None, atol=None, rtol=None):
        if not self.initialized:
            self._initialize_u_v()
        if self.kernel_size == (1, 1):
            return self._compute_weight_1x1(update, n_iterations, atol, rtol)
        else:
            return self._compute_weight_kxk(update, n_iterations, atol, rtol)

    def _compute_weight_1x1(self, update=True, n_iterations=None, atol=None, rtol=None):
        n_iterations = self.n_iterations if n_iterations is None else n_iterations
        atol = self.atol if atol is None else atol
        rtol = self.rtol if rtol is None else atol
        if n_iterations is None and (atol is None or rtol is None):
            raise ValueError('Need one of n_iteration or (atol, rtol).')
        max_itrs = 200
        if n_iterations is not None:
            max_itrs = n_iterations
        u = self.u
        v = self.v
        weight = self.weight.view(self.out_channels, self.in_channels)
        if update:
            with torch.no_grad():
                domain, codomain = self.compute_domain_codomain()
                itrs_used = 0
                for _ in range(max_itrs):
                    old_v = v.clone()
                    old_u = u.clone()
                    u = normalize_u(torch.mv(weight, v), codomain, out=u)
                    v = normalize_v(torch.mv(weight.t(), u), domain, out=v)
                    itrs_used = itrs_used + 1
                    if n_iterations is None and atol is not None and rtol is not None:
                        err_u = torch.norm(u - old_u) / u.nelement() ** 0.5
                        err_v = torch.norm(v - old_v) / v.nelement() ** 0.5
                        tol_u = atol + rtol * torch.max(u)
                        tol_v = atol + rtol * torch.max(v)
                        if err_u < tol_u and err_v < tol_v:
                            break
                if itrs_used > 0:
                    if domain != 1 and domain != 2:
                        self.v.copy_(v)
                    if codomain != 2 and codomain != float('inf'):
                        self.u.copy_(u)
                    u = u.clone()
                    v = v.clone()
        sigma = torch.dot(u, torch.mv(weight, v))
        with torch.no_grad():
            self.scale.copy_(sigma)
        factor = torch.max(torch.ones(1), sigma / self.coeff)
        weight = weight / factor
        return weight.view(self.out_channels, self.in_channels, 1, 1)

    def _compute_weight_kxk(self, update=True, n_iterations=None, atol=None, rtol=None):
        n_iterations = self.n_iterations if n_iterations is None else n_iterations
        atol = self.atol if atol is None else atol
        rtol = self.rtol if rtol is None else atol
        if n_iterations is None and (atol is None or rtol is None):
            raise ValueError('Need one of n_iteration or (atol, rtol).')
        max_itrs = 200
        if n_iterations is not None:
            max_itrs = n_iterations
        u = self.u
        v = self.v
        weight = self.weight
        c, h, w = self.in_channels, int(self.spatial_dims[0].item()), int(self.spatial_dims[1].item())
        if update:
            with torch.no_grad():
                domain, codomain = self.compute_domain_codomain()
                itrs_used = 0
                for _ in range(max_itrs):
                    old_u = u.clone()
                    old_v = v.clone()
                    u_s = F.conv2d(v.view(1, c, h, w), weight, stride=self.stride, padding=self.padding, bias=None)
                    out_shape = u_s.shape
                    u = normalize_u(u_s.view(-1), codomain, out=u)
                    v_s = F.conv_transpose2d(u.view(out_shape), weight, stride=self.stride, padding=self.padding, output_padding=0)
                    v = normalize_v(v_s.view(-1), domain, out=v)
                    itrs_used = itrs_used + 1
                    if n_iterations is None and atol is not None and rtol is not None:
                        err_u = torch.norm(u - old_u) / u.nelement() ** 0.5
                        err_v = torch.norm(v - old_v) / v.nelement() ** 0.5
                        tol_u = atol + rtol * torch.max(u)
                        tol_v = atol + rtol * torch.max(v)
                        if err_u < tol_u and err_v < tol_v:
                            break
                if itrs_used > 0:
                    if domain != 2:
                        self.v.copy_(v)
                    if codomain != 2:
                        self.u.copy_(u)
                    v = v.clone()
                    u = u.clone()
        weight_v = F.conv2d(v.view(1, c, h, w), weight, stride=self.stride, padding=self.padding, bias=None)
        weight_v = weight_v.view(-1)
        sigma = torch.dot(u.view(-1), weight_v)
        with torch.no_grad():
            self.scale.copy_(sigma)
        factor = torch.max(torch.ones(1), sigma / self.coeff)
        weight = weight / factor
        return weight

    def forward(self, input):
        if not self.initialized:
            self.spatial_dims.copy_(torch.tensor(input.shape[2:4]))
        weight = self.compute_weight(update=False)
        return F.conv2d(input, weight, self.bias, self.stride, self.padding, 1, 1)

    def extra_repr(self):
        domain, codomain = self.compute_domain_codomain()
        s = '{in_channels}, {out_channels}, kernel_size={kernel_size}, stride={stride}'
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.bias is None:
            s += ', bias=False'
        s += ', coeff={}, domain={:.2f}, codomain={:.2f}, n_iters={}, atol={}, rtol={}, learnable_ord={}'.format(self.coeff, domain, codomain, self.n_iterations, self.atol, self.rtol, torch.is_tensor(self.domain))
        return s.format(**self.__dict__)


class LipschitzCNN(nn.Module):
    """
    Convolutional neural network which is Lipschitz continuous
    with Lipschitz constant L < 1
    """

    def __init__(self, channels, kernel_size, lipschitz_const=0.97, max_lipschitz_iter=5, lipschitz_tolerance=None, init_zeros=True):
        """Constructor

        Args:
          channels: Integer list with the number of channels of the layers
          kernel_size: Integer list of kernel sizes of the layers
          lipschitz_const: Maximum Lipschitz constant of each layer
          max_lipschitz_iter: Maximum number of iterations used to ensure that layers are Lipschitz continuous with L smaller than set maximum; if None, tolerance is used
          lipschitz_tolerance: Float, tolerance used to ensure Lipschitz continuity if max_lipschitz_iter is None, typically 1e-3
          init_zeros: Flag, whether to initialize last layer approximately with zeros
        """
        super().__init__()
        self.n_layers = len(kernel_size)
        self.channels = channels
        self.kernel_size = kernel_size
        self.lipschitz_const = lipschitz_const
        self.max_lipschitz_iter = max_lipschitz_iter
        self.lipschitz_tolerance = lipschitz_tolerance
        self.init_zeros = init_zeros
        layers = []
        for i in range(self.n_layers):
            layers += [Swish(), InducedNormConv2d(in_channels=channels[i], out_channels=channels[i + 1], kernel_size=kernel_size[i], stride=1, padding=kernel_size[i] // 2, bias=True, coeff=lipschitz_const, domain=2, codomain=2, n_iterations=max_lipschitz_iter, atol=lipschitz_tolerance, rtol=lipschitz_tolerance, zero_init=init_zeros if i == self.n_layers - 1 else False)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def _get_input_degrees(in_features):
    """Returns the degrees an input to MADE should have."""
    return torch.arange(1, in_features + 1)


def tile(x, n):
    x_ = x.reshape(-1)
    x_ = x_.repeat(n)
    x_ = x_.reshape(n, -1)
    x_ = x_.transpose(1, 0)
    x_ = x_.reshape(-1)
    return x_


class MaskedLinear(nn.Linear):
    """A linear module with a masked weight matrix."""

    def __init__(self, in_degrees, out_features, autoregressive_features, random_mask, is_output, bias=True, out_degrees_=None):
        super().__init__(in_features=len(in_degrees), out_features=out_features, bias=bias)
        mask, degrees = self._get_mask_and_degrees(in_degrees=in_degrees, out_features=out_features, autoregressive_features=autoregressive_features, random_mask=random_mask, is_output=is_output, out_degrees_=out_degrees_)
        self.register_buffer('mask', mask)
        self.register_buffer('degrees', degrees)

    @classmethod
    def _get_mask_and_degrees(cls, in_degrees, out_features, autoregressive_features, random_mask, is_output, out_degrees_=None):
        if is_output:
            if out_degrees_ is None:
                out_degrees_ = _get_input_degrees(autoregressive_features)
            out_degrees = tile(out_degrees_, out_features // autoregressive_features)
            mask = (out_degrees[..., None] > in_degrees).float()
        else:
            if random_mask:
                min_in_degree = torch.min(in_degrees).item()
                min_in_degree = min(min_in_degree, autoregressive_features - 1)
                out_degrees = torch.randint(low=min_in_degree, high=autoregressive_features, size=[out_features], dtype=torch.long)
            else:
                max_ = max(1, autoregressive_features - 1)
                min_ = min(1, autoregressive_features - 1)
                out_degrees = torch.arange(out_features) % max_ + min_
            mask = (out_degrees[..., None] >= in_degrees).float()
        return mask, out_degrees

    def forward(self, x):
        return F.linear(x, self.weight * self.mask, self.bias)


class MaskedFeedforwardBlock(nn.Module):
    """A feedforward block based on a masked linear module.

    **NOTE** In this implementation, the number of output features is taken to be equal to the number of input features.
    """

    def __init__(self, in_degrees, autoregressive_features, context_features=None, random_mask=False, activation=F.relu, dropout_probability=0.0, use_batch_norm=False):
        super().__init__()
        features = len(in_degrees)
        if use_batch_norm:
            self.batch_norm = nn.BatchNorm1d(features, eps=0.001)
        else:
            self.batch_norm = None
        if context_features is not None:
            raise NotImplementedError()
        self.linear = MaskedLinear(in_degrees=in_degrees, out_features=features, autoregressive_features=autoregressive_features, random_mask=random_mask, is_output=False)
        self.degrees = self.linear.degrees
        self.activation = activation
        self.dropout = nn.Dropout(p=dropout_probability)

    def forward(self, inputs, context=None):
        if context is not None:
            raise NotImplementedError()
        if self.batch_norm:
            outputs = self.batch_norm(inputs)
        else:
            outputs = inputs
        outputs = self.linear(outputs)
        outputs = self.activation(outputs)
        outputs = self.dropout(outputs)
        return outputs


class MaskedResidualBlock(nn.Module):
    """A residual block containing masked linear modules."""

    def __init__(self, in_degrees, autoregressive_features, context_features=None, random_mask=False, activation=F.relu, dropout_probability=0.0, use_batch_norm=False, zero_initialization=True):
        if random_mask:
            raise ValueError("Masked residual block can't be used with random masks.")
        super().__init__()
        features = len(in_degrees)
        if context_features is not None:
            self.context_layer = nn.Linear(context_features, features)
        self.use_batch_norm = use_batch_norm
        if use_batch_norm:
            self.batch_norm_layers = nn.ModuleList([nn.BatchNorm1d(features, eps=0.001) for _ in range(2)])
        linear_0 = MaskedLinear(in_degrees=in_degrees, out_features=features, autoregressive_features=autoregressive_features, random_mask=False, is_output=False)
        linear_1 = MaskedLinear(in_degrees=linear_0.degrees, out_features=features, autoregressive_features=autoregressive_features, random_mask=False, is_output=False)
        self.linear_layers = nn.ModuleList([linear_0, linear_1])
        self.degrees = linear_1.degrees
        if torch.all(self.degrees >= in_degrees).item() != 1:
            raise RuntimeError("In a masked residual block, the output degrees can't be less than the corresponding input degrees.")
        self.activation = activation
        self.dropout = nn.Dropout(p=dropout_probability)
        if zero_initialization:
            init.uniform_(self.linear_layers[-1].weight, a=-0.001, b=0.001)
            init.uniform_(self.linear_layers[-1].bias, a=-0.001, b=0.001)

    def forward(self, inputs, context=None):
        temps = inputs
        if self.use_batch_norm:
            temps = self.batch_norm_layers[0](temps)
        temps = self.activation(temps)
        temps = self.linear_layers[0](temps)
        if self.use_batch_norm:
            temps = self.batch_norm_layers[1](temps)
        temps = self.activation(temps)
        temps = self.dropout(temps)
        temps = self.linear_layers[1](temps)
        if context is not None:
            temps = F.glu(torch.cat((temps, self.context_layer(context)), dim=1), dim=1)
        return inputs + temps


class MADE(nn.Module):
    """Implementation of MADE.

    It can use either feedforward blocks or residual blocks (default is residual).
    Optionally, it can use batch norm or dropout within blocks (default is no).
    """

    def __init__(self, features, hidden_features, context_features=None, num_blocks=2, output_multiplier=1, use_residual_blocks=True, random_mask=False, permute_mask=False, activation=F.relu, dropout_probability=0.0, use_batch_norm=False, preprocessing=None):
        if use_residual_blocks and random_mask:
            raise ValueError("Residual blocks can't be used with random masks.")
        super().__init__()
        if preprocessing is None:
            self.preprocessing = torch.nn.Identity()
        else:
            self.preprocessing = preprocessing
        input_degrees_ = _get_input_degrees(features)
        if permute_mask:
            input_degrees_ = input_degrees_[torch.randperm(features)]
        self.initial_layer = MaskedLinear(in_degrees=input_degrees_, out_features=hidden_features, autoregressive_features=features, random_mask=random_mask, is_output=False)
        if context_features is not None:
            self.context_layer = nn.Linear(context_features, hidden_features)
        blocks = []
        if use_residual_blocks:
            block_constructor = MaskedResidualBlock
        else:
            block_constructor = MaskedFeedforwardBlock
        prev_out_degrees = self.initial_layer.degrees
        for _ in range(num_blocks):
            blocks.append(block_constructor(in_degrees=prev_out_degrees, autoregressive_features=features, context_features=context_features, random_mask=random_mask, activation=activation, dropout_probability=dropout_probability, use_batch_norm=use_batch_norm))
            prev_out_degrees = blocks[-1].degrees
        self.blocks = nn.ModuleList(blocks)
        self.final_layer = MaskedLinear(in_degrees=prev_out_degrees, out_features=features * output_multiplier, autoregressive_features=features, random_mask=random_mask, is_output=True, out_degrees_=input_degrees_)

    def forward(self, inputs, context=None):
        outputs = self.preprocessing(inputs)
        outputs = self.initial_layer(outputs)
        if context is not None:
            outputs += self.context_layer(context)
        for block in self.blocks:
            outputs = block(outputs, context)
        outputs = self.final_layer(outputs)
        return outputs


class MLP(nn.Module):
    """
    A multilayer perceptron with Leaky ReLU nonlinearities
    """

    def __init__(self, layers, leaky=0.0, score_scale=None, output_fn=None, output_scale=None, init_zeros=False, dropout=None):
        """
        layers: list of layer sizes from start to end
        leaky: slope of the leaky part of the ReLU, if 0.0, standard ReLU is used
        score_scale: Factor to apply to the scores, i.e. output before output_fn.
        output_fn: String, function to be applied to the output, either None, "sigmoid", "relu", "tanh", or "clampexp"
        output_scale: Rescale outputs if output_fn is specified, i.e. ```scale * output_fn(out / scale)```
        init_zeros: Flag, if true, weights and biases of last layer are initialized with zeros (helpful for deep models, see [arXiv 1807.03039](https://arxiv.org/abs/1807.03039))
        dropout: Float, if specified, dropout is done before last layer; if None, no dropout is done
        """
        super().__init__()
        net = nn.ModuleList([])
        for k in range(len(layers) - 2):
            net.append(nn.Linear(layers[k], layers[k + 1]))
            net.append(nn.LeakyReLU(leaky))
        if dropout is not None:
            net.append(nn.Dropout(p=dropout))
        net.append(nn.Linear(layers[-2], layers[-1]))
        if init_zeros:
            nn.init.zeros_(net[-1].weight)
            nn.init.zeros_(net[-1].bias)
        if output_fn is not None:
            if score_scale is not None:
                net.append(utils.ConstScaleLayer(score_scale))
            if output_fn == 'sigmoid':
                net.append(nn.Sigmoid())
            elif output_fn == 'relu':
                net.append(nn.ReLU())
            elif output_fn == 'tanh':
                net.append(nn.Tanh())
            elif output_fn == 'clampexp':
                net.append(utils.ClampExp())
            else:
                NotImplementedError('This output function is not implemented.')
            if output_scale is not None:
                net.append(utils.ConstScaleLayer(output_scale))
        self.net = nn.Sequential(*net)

    def forward(self, x):
        return self.net(x)


class ConvResidualBlock(nn.Module):

    def __init__(self, channels, context_channels=None, activation=F.relu, dropout_probability=0.0, use_batch_norm=False, zero_initialization=True):
        super().__init__()
        self.activation = activation
        if context_channels is not None:
            self.context_layer = nn.Conv2d(in_channels=context_channels, out_channels=channels, kernel_size=1, padding=0)
        self.use_batch_norm = use_batch_norm
        if use_batch_norm:
            self.batch_norm_layers = nn.ModuleList([nn.BatchNorm2d(channels, eps=0.001) for _ in range(2)])
        self.conv_layers = nn.ModuleList([nn.Conv2d(channels, channels, kernel_size=3, padding=1) for _ in range(2)])
        self.dropout = nn.Dropout(p=dropout_probability)
        if zero_initialization:
            init.uniform_(self.conv_layers[-1].weight, -0.001, 0.001)
            init.uniform_(self.conv_layers[-1].bias, -0.001, 0.001)

    def forward(self, inputs, context=None):
        temps = inputs
        if self.use_batch_norm:
            temps = self.batch_norm_layers[0](temps)
        temps = self.activation(temps)
        temps = self.conv_layers[0](temps)
        if self.use_batch_norm:
            temps = self.batch_norm_layers[1](temps)
        temps = self.activation(temps)
        temps = self.dropout(temps)
        temps = self.conv_layers[1](temps)
        if context is not None:
            temps = F.glu(torch.cat((temps, self.context_layer(context)), dim=1), dim=1)
        return inputs + temps


class ConvResidualNet(nn.Module):

    def __init__(self, in_channels, out_channels, hidden_channels, context_channels=None, num_blocks=2, activation=F.relu, dropout_probability=0.0, use_batch_norm=False):
        super().__init__()
        self.context_channels = context_channels
        self.hidden_channels = hidden_channels
        if context_channels is not None:
            self.initial_layer = nn.Conv2d(in_channels=in_channels + context_channels, out_channels=hidden_channels, kernel_size=1, padding=0)
        else:
            self.initial_layer = nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=1, padding=0)
        self.blocks = nn.ModuleList([ConvResidualBlock(channels=hidden_channels, context_channels=context_channels, activation=activation, dropout_probability=dropout_probability, use_batch_norm=use_batch_norm) for _ in range(num_blocks)])
        self.final_layer = nn.Conv2d(hidden_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, inputs, context=None):
        if context is None:
            temps = self.initial_layer(inputs)
        else:
            temps = self.initial_layer(torch.cat((inputs, context), dim=1))
        for block in self.blocks:
            temps = block(temps, context)
        outputs = self.final_layer(temps)
        return outputs


class Logit:
    """Transform for dataloader

    ```
    logit(alpha + (1 - alpha) * x) where logit(x) = log(x / (1 - x))
    ```
    """

    def __init__(self, alpha=0):
        """Constructor

        Args:
          alpha: see above
        """
        self.alpha = alpha

    def __call__(self, x):
        x_ = self.alpha + (1 - self.alpha) * x
        return torch.log(x_ / (1 - x_))

    def inverse(self, x):
        return (torch.sigmoid(x) - self.alpha) / (1 - self.alpha)


class ConstScaleLayer(nn.Module):
    """
    Scaling features by a fixed factor
    """

    def __init__(self, scale=1.0):
        """Constructor

        Args:
          scale: Scale to apply to features
        """
        super().__init__()
        self.scale_cpu = torch.tensor(scale)
        self.register_buffer('scale', self.scale_cpu)

    def forward(self, input):
        return input * self.scale


class ActNorm(nn.Module):
    """
    ActNorm layer with just one forward pass
    """

    def __init__(self, shape):
        """Constructor

        Args:
          shape: Same as shape in flows.ActNorm
          logscale_factor: Same as shape in flows.ActNorm

        """
        super().__init__()
        self.actNorm = flows.ActNorm(shape)

    def forward(self, input):
        out, _ = self.actNorm(input)
        return out


class ClampExp(nn.Module):
    """
    Nonlinearity min(exp(lam * x), 1)
    """

    def __init__(self):
        """Constructor

        Args:
          lam: Lambda parameter
        """
        super(ClampExp, self).__init__()

    def forward(self, x):
        one = torch.tensor(1.0, device=x.device, dtype=x.dtype)
        return torch.min(torch.exp(x), one)


class PeriodicFeaturesCat(nn.Module):
    """
    Converts a specified part of the input to periodic features by
    replacing those features f with [sin(scale * f), cos(scale * f)].

    Note that this decreases the number of features and their order
    is changed.
    """

    def __init__(self, ndim, ind, scale=1.0):
        """
        Constructor
        :param ndim: Int, number of dimensions
        :param ind: Iterable, indices of input elements to convert to
        periodic features
        :param scale: Scalar or iterable, used to scale inputs before
        converting them to periodic features
        """
        super(PeriodicFeaturesCat, self).__init__()
        self.ndim = ndim
        if torch.is_tensor(ind):
            self.register_buffer('ind', torch._cast_Long(ind))
        else:
            self.register_buffer('ind', torch.tensor(ind, dtype=torch.long))
        ind_ = []
        for i in range(self.ndim):
            if not i in self.ind:
                ind_ += [i]
        self.register_buffer('ind_', torch.tensor(ind_, dtype=torch.long))
        if torch.is_tensor(scale):
            self.register_buffer('scale', scale)
        else:
            self.scale = scale

    def forward(self, inputs):
        inputs_ = inputs[..., self.ind]
        inputs_ = self.scale * inputs_
        inputs_sin = torch.sin(inputs_)
        inputs_cos = torch.cos(inputs_)
        out = torch.cat((inputs_sin, inputs_cos, inputs[..., self.ind_]), -1)
        return out


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (BatchNorm,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ClampExp,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ClassCondDiagGaussian,
     lambda: ([], {'shape': 4, 'num_classes': 4}),
     lambda: ([], {})),
    (ConstScaleLayer,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ConvNet2d,
     lambda: ([], {'channels': [4, 4, 4], 'kernel_size': [4, 4]}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ConvResidualBlock,
     lambda: ([], {'channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ConvResidualNet,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'hidden_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (DiagGaussian,
     lambda: ([], {'shape': 4}),
     lambda: ([], {})),
    (Dirac,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4])], {})),
    (GaussianMixture,
     lambda: ([], {'n_modes': 4, 'dim': 4}),
     lambda: ([], {})),
    (GlowBase,
     lambda: ([], {'shape': 4}),
     lambda: ([], {})),
    (InducedNormLinear,
     lambda: ([], {'in_features': 4, 'out_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Invertible1x1Conv,
     lambda: ([], {'num_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (InvertibleAffine,
     lambda: ([], {'num_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (LULinearPermute,
     lambda: ([], {'num_channels': 4}),
     lambda: ([torch.rand([4, 4])], {})),
    (LipschitzCNN,
     lambda: ([], {'channels': [4, 4, 4], 'kernel_size': [4, 4]}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (LipschitzMLP,
     lambda: ([], {'channels': [4, 4]}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (MADE,
     lambda: ([], {'features': 4, 'hidden_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (MLP,
     lambda: ([], {'layers': [4, 4]}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Merge,
     lambda: ([], {}),
     lambda: ([(torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]))], {})),
    (NNBernoulliDecoder,
     lambda: ([], {'net': torch.nn.ReLU()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (NNDiagGaussian,
     lambda: ([], {'net': torch.nn.ReLU()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (NNDiagGaussianDecoder,
     lambda: ([], {'net': torch.nn.ReLU()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (PeriodicShift,
     lambda: ([], {'ind': 4}),
     lambda: ([torch.rand([4, 5])], {})),
    (PeriodicWrap,
     lambda: ([], {'ind': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Permute,
     lambda: ([], {'num_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Planar,
     lambda: ([], {'shape': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Radial,
     lambda: ([], {'shape': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ResidualBlock,
     lambda: ([], {'features': 4, 'context_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ResidualNet,
     lambda: ([], {'in_features': 4, 'out_features': 4, 'hidden_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Split,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Squeeze,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Swish,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Uniform,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4])], {})),
    (_LULinear,
     lambda: ([], {'features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (_RandomPermutation,
     lambda: ([], {'features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (iResBlock,
     lambda: ([], {'nnet': torch.nn.ReLU()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
]

