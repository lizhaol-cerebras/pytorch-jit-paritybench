
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


import time


from torch.utils.data import DataLoader


from torchvision import datasets


from torchvision import transforms


import torch.nn.functional as F


import logging


from enum import Enum


from typing import List


import numpy as np


from sklearn.metrics import accuracy_score


from torch import nn


from warnings import warn


from typing import Optional


from typing import Tuple


from typing import Union


from torch._C import Value


from typing import Callable


from typing import Any


import math


import pandas as pd


import matplotlib.pyplot as plt


from matplotlib.gridspec import GridSpec


from typing import Dict


from collections import defaultdict


from matplotlib.figure import Figure


from matplotlib.artist import Artist


from matplotlib.animation import ArtistAnimation


from torchvision.transforms import Compose


from torchvision.transforms import Normalize


from torchvision.transforms import Lambda


import torch.utils.data as data


import torch._dynamo as dynamo


class Net(nn.Module):

    def __init__(self):
        super().__init__()
        snn.LIF.clear_instances()
        self.fc1 = nn.Linear(1, 1)
        self.lif1 = snn.Synaptic(alpha=0.5, beta=0.5, num_inputs=1, batch_size=1, init_hidden=True)
        self.lif2 = snn.Alpha(alpha=0.6, beta=0.5, num_inputs=1, batch_size=1, hidden_init=True)

    def forward(self, x):
        cur1 = self.fc1(x)
        self.lif1.spk1, self.lif1.syn1, self.lif1.mem1 = self.lif1(cur1, self.lif1.syn, self.lif1.mem)
        return self.lif1.spk, self.lif1.mem


batch_size = 128


beta = 0.9


num_hidden = 1000


num_inputs = 784


num_output = 10


num_steps = 25


class Model(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(num_inputs, num_hidden)
        self.lif1 = snn.Leaky(beta=beta)
        self.fc2 = nn.Linear(num_hidden, num_output)
        self.lif2 = snn.Leaky(beta=beta)
        self.loss_fn = SF.ce_count_loss()

    def forward(self, x, labels=None):
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        spk2_rec = []
        mem2_rec = []
        for step in range(num_steps):
            cur1 = self.fc1(x.view(batch_size, -1))
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            spk2_rec.append(spk2)
            mem2_rec.append(mem2)
        spk2_rec = torch.stack(spk2_rec)
        mem2_rec = torch.stack(mem2_rec)
        if self.training:
            return spk2_rec, poptorch.identity_loss(self.loss_fn(mem2_rec, labels), 'none')
        return spk2_rec


class GradedSpikes(torch.nn.Module):
    """Learnable spiking magnitude for spiking layers."""

    def __init__(self, size, constant_factor):
        """

        :param size: The input size of the layer. Must be equal to the
            number of neurons in the preceeding layer.
        :type size: int
        :param constant_factor: If provided, the weights will be
            initialized with ones times the contant_factor. If
            not provided, the weights will be initialized using a uniform
            distribution U(0.5, 1.5).
        :type constant_factor: float

        """
        super().__init__()
        self.size = size
        if constant_factor:
            weights = torch.ones(size=[size, 1]) * constant_factor
            self.weights = torch.nn.Parameter(weights)
        else:
            weights = torch.rand(size=[size, 1]) + 0.5
            self.weights = torch.nn.Parameter(weights)

    def forward(self, x):
        """Forward pass is simply: spikes 'x' * weights."""
        return torch.multiply(input=x, other=self.weights)


class LeakyKernel(nn.Module):
    """
    A parallel implementation of the Leaky neuron with a fused input linear layer.
    All time steps are passed to the input at once.
    This implementation uses `torch.nn.RNN` to accelerate the implementation.

    First-order leaky integrate-and-fire neuron model.
    Input is assumed to be a current injection.
    Membrane potential decays exponentially with rate beta.
    For :math:`U[T] > U_{\\rm thr} ⇒ S[T+1] = 1`.

    .. math::

            U[t+1] = βU[t] + I_{\\rm in}[t+1]


    * :math:`I_{\\rm in}` - Input current
    * :math:`U` - Membrane potential
    * :math:`U_{\\rm thr}` - Membrane threshold
    * :math:`β` - Membrane potential decay rate

    Several differences between `LeakyParallel` and `Leaky` include:

    * Negative hidden states are clipped due to the forced ReLU operation in RNN
    * Linear weights are included in addition to recurrent weights
    * `beta` is clipped between [0,1] and cloned to `weight_hh_l` only upon layer initialization. It is unused otherwise
    * There is no explicit reset mechanism
    * Several functions such as `init_hidden`, `output`, `inhibition`, and `state_quant` are unavailable in `LeakyParallel`
    * Only the output spike is returned. Membrane potential is not accessible by default
    * RNN uses a hidden matrix of size (num_hidden, num_hidden) to transform the hidden state vector. This would 'leak' the membrane potential between LIF neurons, and so the hidden matrix is forced to a diagonal matrix by default. This can be disabled by setting `weight_hh_enable=True`.

    Example::

        import torch
        import torch.nn as nn
        import snntorch as snn

        beta = 0.5
        num_inputs = 784
        num_hidden = 128
        num_outputs = 10
        batch_size = 128
        x = torch.rand((num_steps, batch_size, num_inputs))

        # Define Network
        class Net(nn.Module):
            def __init__(self):
                super().__init__()

                # initialize layers
                self.lif1 = snn.LeakyParallel(input_size=num_inputs, hidden_size=num_hidden) # randomly initialize recurrent weights
                self.lif2 = snn.LeakyParallel(input_size=num_hidden, hidden_size=num_outputs, beta=beta, learn_beta=True) # learnable recurrent weights initialized at beta

            def forward(self, x):
                spk1 = self.lif1(x)
                spk2 = self.lif2(spk1)
                return spk2


    :param input_size: The number of expected features in the input `x`
    :type input_size: int

    :param hidden_size: The number of features in the hidden state `h`
    :type hidden_size: int

    :param beta: membrane potential decay rate. Clipped between 0 and 1
        during the forward-pass. May be a single-valued tensor (i.e., equal
        decay rate for all neurons in a layer), or multi-valued (one weight per
        neuron). If left unspecified, then the decay rates will be randomly initialized based on PyTorch's initialization for RNN. Defaults to None
    :type beta: float or torch.tensor, optional

    :param bias: If `False`, then the layer does not use bias weights `b_ih` and `b_hh`. Defaults to True
    :type bias: Bool, optional

    :param threshold: Threshold for :math:`mem` to reach in order to
        generate a spike `S=1`. Defaults to 1
    :type threshold: float, optional

    :param dropout: If non-zero, introduces a Dropout layer on the RNN output with dropout probability equal to dropout. Defaults to 0
    :type dropout: float, optional

    :param spike_grad: Surrogate gradient for the term dS/dU. Defaults to
        None (corresponds to ATan surrogate gradient. See
        `snntorch.surrogate` for more options)
    :type spike_grad: surrogate gradient function from snntorch.surrogate,
        optional

    :param surrogate_disable: Disables surrogate gradients regardless of
        `spike_grad` argument. Useful for ONNX compatibility. Defaults
        to False
    :type surrogate_disable: bool, Optional

    :param learn_beta: Option to enable learnable beta. Defaults to False
    :type learn_beta: bool, optional

    :param learn_threshold: Option to enable learnable threshold. Defaults
        to False
    :type learn_threshold: bool, optional

    :param weight_hh_enable: Option to set the hidden matrix to be dense or
        diagonal. Diagonal (i.e., False) adheres to how a LIF neuron works.
        Dense (True) would allow the membrane potential of one LIF neuron to
        influence all others, and follow the RNN default implementation. Defaults to False
    :type weight_hh_enable: bool, optional


    Inputs: \\input_
        - **input_** of shape of  shape `(L, H_{in})` for unbatched input,
            or `(L, N, H_{in})` containing the features of the input sequence.

    Outputs: spk
        - **spk** of shape `(L, batch, input_size)`: tensor containing the
            output spikes.

    where:

    `L = sequence length`

    `N = batch size`

    `H_{in} = input_size`

    `H_{out} = hidden_size`

    Learnable Parameters:
        - **rnn.weight_ih_l** (torch.Tensor) - the learnable input-hidden weights of shape (hidden_size, input_size)
        - **rnn.weight_hh_l** (torch.Tensor) - the learnable hidden-hidden weights of the k-th layer which are sampled from `beta` of shape (hidden_size, hidden_size)
        - **bias_ih_l** - the learnable input-hidden bias of the k-th layer, of shape (hidden_size)
        - **bias_hh_l** - the learnable hidden-hidden bias of the k-th layer, of shape (hidden_size)
        - **threshold** (torch.Tensor) - optional learnable thresholds
            must be manually passed in, of shape `1` or`` (input_size).
        - **graded_spikes_factor** (torch.Tensor) - optional learnable graded spike factor

    """

    def __init__(self, input_size, hidden_size, beta=None, bias=True, threshold=1.0, dropout=0.0, spike_grad=None, surrogate_disable=False, learn_beta=False, learn_threshold=False, graded_spikes_factor=1.0, learn_graded_spikes_factor=False, weight_hh_enable=False, device=None, dtype=None):
        super(LeakyKernel, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers=1, nonlinearity='relu', bias=bias, batch_first=False, dropout=dropout, device=device, dtype=dtype)
        self._beta_buffer(beta, learn_beta)
        self.hidden_size = hidden_size
        if self.beta is not None:
            self.beta = self.beta.clamp(0, 1)
        if spike_grad is None:
            self.spike_grad = self.ATan.apply
        else:
            self.spike_grad = spike_grad
        self._beta_to_weight_hh()
        if weight_hh_enable is False:
            self.weight_hh_enable()
            if learn_beta:
                self.rnn.weight_hh_l0.register_hook(self.grad_hook)
        if not learn_beta:
            self.rnn.weight_hh_l0.requires_grad_(False)
        self._threshold_buffer(threshold, learn_threshold)
        self._graded_spikes_buffer(graded_spikes_factor, learn_graded_spikes_factor)
        self.surrogate_disable = surrogate_disable
        if self.surrogate_disable:
            self.spike_grad = self._surrogate_bypass

    def forward(self, input_):
        mem = self.rnn(input_)
        mem_shift = mem[0] - self.threshold
        spk = self.spike_grad(mem_shift)
        spk = spk * self.graded_spikes_factor
        return spk

    @staticmethod
    def _surrogate_bypass(input_):
        return (input_ > 0).float()


    @staticmethod
    class ATan(torch.autograd.Function):
        """
        Surrogate gradient of the Heaviside step function.

        **Forward pass:** Heaviside step function shifted.

            .. math::

                S=\\begin{cases} 1 & \\text{if U ≥ U$_{\\rm thr}$} \\\\
                0 & \\text{if U < U$_{\\rm thr}$}
                \\end{cases}

        **Backward pass:** Gradient of shifted arc-tan function.

            .. math::

                    S&≈\\frac{1}{π}\\text{arctan}(πU \\frac{α}{2}) \\\\
                    \\frac{∂S}{∂U}&=\\frac{1}{π}                    \\frac{1}{(1+(πU\\frac{α}{2})^2)}


        :math:`alpha` defaults to 2, and can be modified by calling
        ``surrogate.atan(alpha=2)``.

        Adapted from:

        *W. Fang, Z. Yu, Y. Chen, T. Masquelier, T. Huang, Y. Tian (2021)
        Incorporating Learnable Membrane Time Constants to Enhance Learning
        of Spiking Neural Networks. Proc. IEEE/CVF Int. Conf. Computer
        Vision (ICCV), pp. 2661-2671.*"""

        @staticmethod
        def forward(ctx, input_, alpha=2.0):
            ctx.save_for_backward(input_)
            ctx.alpha = alpha
            out = (input_ > 0).float()
            return out

        @staticmethod
        def backward(ctx, grad_output):
            input_, = ctx.saved_tensors
            grad_input = grad_output.clone()
            grad = ctx.alpha / 2 / (1 + (torch.pi / 2 * ctx.alpha * input_).pow_(2)) * grad_input
            return grad, None

    def weight_hh_enable(self):
        mask = torch.eye(self.hidden_size, self.hidden_size)
        self.rnn.weight_hh_l0.data = self.rnn.weight_hh_l0.data * mask

    def grad_hook(self, grad):
        device = grad.device
        mask = torch.eye(self.hidden_size, self.hidden_size, device=device)
        return grad * mask

    def _beta_to_weight_hh(self):
        with torch.no_grad():
            if self.beta is not None:
                if isinstance(self.beta, float) or isinstance(self.beta, int):
                    self.rnn.weight_hh_l0.fill_(self.beta)
                elif isinstance(self.beta, torch.Tensor) or isinstance(self.beta, torch.FloatTensor):
                    if len(self.beta) == 1:
                        self.rnn.weight_hh_l0.fill_(self.beta[0])
                elif len(self.beta) == self.hidden_size:
                    for i in range(self.hidden_size):
                        self.rnn.weight_hh_l0.data[i].fill_(self.beta[i])
                else:
                    raise ValueError("Beta must be either a single value or of length 'hidden_size'.")

    def _beta_buffer(self, beta, learn_beta):
        if not isinstance(beta, torch.Tensor):
            if beta is not None:
                beta = torch.as_tensor([beta])
        self.register_buffer('beta', beta)

    def _graded_spikes_buffer(self, graded_spikes_factor, learn_graded_spikes_factor):
        if not isinstance(graded_spikes_factor, torch.Tensor):
            graded_spikes_factor = torch.as_tensor(graded_spikes_factor)
        if learn_graded_spikes_factor:
            self.graded_spikes_factor = nn.Parameter(graded_spikes_factor)
        else:
            self.register_buffer('graded_spikes_factor', graded_spikes_factor)

    def _threshold_buffer(self, threshold, learn_threshold):
        if not isinstance(threshold, torch.Tensor):
            threshold = torch.as_tensor(threshold)
        if learn_threshold:
            self.threshold = nn.Parameter(threshold)
        else:
            self.register_buffer('threshold', threshold)


class LeakyParallel(nn.Module):
    """
    A parallel implementation of the Leaky neuron intended to handle arbitrary input dimensions.
    All time steps are passed to the input at once.
    This implementation uses `torch.nn.RNN` to accelerate the implementation.

    First-order leaky integrate-and-fire neuron model.
    Input is assumed to be a current injection.
    Membrane potential decays exponentially with rate beta.
    For :math:`U[T] > U_{\\rm thr} ⇒ S[T+1] = 1`.

    .. math::

            U[t+1] = βU[t] + I_{\\rm in}[t+1]


    * :math:`I_{\\rm in}` - Input current
    * :math:`U` - Membrane potential
    * :math:`U_{\\rm thr}` - Membrane threshold
    * :math:`β` - Membrane potential decay rate

    Several differences between `LeakyParallel` and `Leaky` include:

    * Negative hidden states are clipped due to the forced ReLU operation in RNN
    * Linear weights are included in addition to recurrent weights
    * `beta` is clipped between [0,1] and cloned to `weight_hh_l` only upon layer initialization. It is unused otherwise
    * There is no explicit reset mechanism
    * Several functions such as `init_hidden`, `output`, `inhibition`, and `state_quant` are unavailable in `LeakyParallel`
    * Only the output spike is returned. Membrane potential is not accessible by default
    * RNN uses a hidden matrix of size (num_hidden, num_hidden) to transform the hidden state vector. This would 'leak' the membrane potential between LIF neurons, and so the hidden matrix is forced to a diagonal matrix by default. This can be disabled by setting `weight_hh_enable=True`.

    Example::

        import torch
        import torch.nn as nn
        import snntorch as snn

        beta = 0.5
        num_inputs = 784
        num_hidden = 128
        num_outputs = 10
        batch_size = 128
        x = torch.rand((num_steps, batch_size, num_inputs))

        # Define Network
        class Net(nn.Module):
            def __init__(self):
                super().__init__()

                # initialize layers
                self.lif1 = snn.LeakyParallel(input_size=num_inputs, hidden_size=num_hidden) # randomly initialize recurrent weights
                self.lif2 = snn.LeakyParallel(input_size=num_hidden, hidden_size=num_outputs, beta=beta, learn_beta=True) # learnable recurrent weights initialized at beta

            def forward(self, x):
                spk1 = self.lif1(x)
                spk2 = self.lif2(spk1)
                return spk2


    :param input_size: The number of expected features in the input `x`. The output of a linear layer should be an int, whereas the output of a 2D-convolution should be a tuple with 3 int values
    :type input_size: int or tuple

    :param hidden_size: The number of features in the hidden state `h`
    :type hidden_size: int

    :param beta: membrane potential decay rate. Clipped between 0 and 1
        during the forward-pass. May be a single-valued tensor (i.e., equal
        decay rate for all neurons in a layer), or multi-valued (one weight per
        neuron). If left unspecified, then the decay rates will be randomly initialized based on PyTorch's initialization for RNN. Defaults to None
    :type beta: float or torch.tensor, optional

    :param bias: If `False`, then the layer does not use bias weights `b_ih` and `b_hh`. Defaults to True
    :type bias: Bool, optional

    :param threshold: Threshold for :math:`mem` to reach in order to
        generate a spike `S=1`. Defaults to 1
    :type threshold: float, optional

    :param dropout: If non-zero, introduces a Dropout layer on the RNN output with dropout probability equal to dropout. Defaults to 0
    :type dropout: float, optional

    :param spike_grad: Surrogate gradient for the term dS/dU. Defaults to
        None (corresponds to ATan surrogate gradient. See
        `snntorch.surrogate` for more options)
    :type spike_grad: surrogate gradient function from snntorch.surrogate,
        optional

    :param surrogate_disable: Disables surrogate gradients regardless of
        `spike_grad` argument. Useful for ONNX compatibility. Defaults
        to False
    :type surrogate_disable: bool, Optional

    :param learn_beta: Option to enable learnable beta. Defaults to False
    :type learn_beta: bool, optional

    :param learn_threshold: Option to enable learnable threshold. Defaults
        to False
    :type learn_threshold: bool, optional

    :param weight_hh_enable: Option to set the hidden matrix to be dense or
        diagonal. Diagonal (i.e., False) adheres to how a LIF neuron works.
        Dense (True) would allow the membrane potential of one LIF neuron to
        influence all others, and follow the RNN default implementation. Defaults to False
    :type weight_hh_enable: bool, optional


    Inputs: \\input_
        - **input_** of shape of  shape `(L, H_{in})` for unbatched input,
            or `(L, N, H_{in})` containing the features of the input sequence.

    Outputs: spk
        - **spk** of shape `(L, batch, input_size)`: tensor containing the
            output spikes.

    where:

    `L = sequence length`

    `N = batch size`

    `H_{in} = input_size`

    `H_{out} = hidden_size`

    Learnable Parameters:
        - **rnn.weight_ih_l** (torch.Tensor) - the learnable input-hidden weights of shape (hidden_size, input_size)
        - **rnn.weight_hh_l** (torch.Tensor) - the learnable hidden-hidden weights of the k-th layer which are sampled from `beta` of shape (hidden_size, hidden_size)
        - **bias_ih_l** - the learnable input-hidden bias of the k-th layer, of shape (hidden_size)
        - **bias_hh_l** - the learnable hidden-hidden bias of the k-th layer, of shape (hidden_size)
        - **threshold** (torch.Tensor) - optional learnable thresholds
            must be manually passed in, of shape `1` or`` (input_size).
        - **graded_spikes_factor** (torch.Tensor) - optional learnable graded spike factor

    """

    def __init__(self, input_size, beta=None, bias=True, threshold=1.0, dropout=0.0, spike_grad=None, surrogate_disable=False, learn_beta=False, learn_threshold=False, graded_spikes_factor=1.0, learn_graded_spikes_factor=False, weight_hh_enable=False, device=None, dtype=None):
        super(LeakyParallel, self).__init__()
        self.input_size = input_size
        unrolled_input_size = self._process_input()
        self.rnn = nn.RNN(unrolled_input_size, unrolled_input_size, num_layers=1, nonlinearity='relu', bias=bias, batch_first=False, dropout=dropout, device=device, dtype=dtype)
        self._beta_buffer(beta, learn_beta)
        self.hidden_size = unrolled_input_size
        if self.beta is not None:
            self.beta = self.beta.clamp(0, 1)
        if spike_grad is None:
            self.spike_grad = self.ATan.apply
        else:
            self.spike_grad = spike_grad
        self.weight_ih_disable()
        self._beta_to_weight_hh()
        if weight_hh_enable is False:
            self.weight_hh_enable()
            if learn_beta:
                self.rnn.weight_hh_l0.register_hook(self.grad_hook)
        if not learn_beta:
            self.rnn.weight_hh_l0.requires_grad_(False)
        self._threshold_buffer(threshold, learn_threshold)
        self._graded_spikes_buffer(graded_spikes_factor, learn_graded_spikes_factor)
        self.surrogate_disable = surrogate_disable
        if self.surrogate_disable:
            self.spike_grad = self._surrogate_bypass

    def forward(self, input_):
        input_ = self.process_tensor(input_)
        mem = self.rnn(input_)
        mem_shift = mem[0] - self.threshold
        spk = self.spike_grad(mem_shift)
        spk = spk * self.graded_spikes_factor
        spk = self.unprocess_tensor(self, spk)
        return spk

    @staticmethod
    def _surrogate_bypass(input_):
        return (input_ > 0).float()


    @staticmethod
    class ATan(torch.autograd.Function):
        """
        Surrogate gradient of the Heaviside step function.

        **Forward pass:** Heaviside step function shifted.

            .. math::

                S=\\begin{cases} 1 & \\text{if U ≥ U$_{\\rm thr}$} \\\\
                0 & \\text{if U < U$_{\\rm thr}$}
                \\end{cases}

        **Backward pass:** Gradient of shifted arc-tan function.

            .. math::

                    S&≈\\frac{1}{π}\\text{arctan}(πU \\frac{α}{2}) \\\\
                    \\frac{∂S}{∂U}&=\\frac{1}{π}                    \\frac{1}{(1+(πU\\frac{α}{2})^2)}


        :math:`alpha` defaults to 2, and can be modified by calling
        ``surrogate.atan(alpha=2)``.

        Adapted from:

        *W. Fang, Z. Yu, Y. Chen, T. Masquelier, T. Huang, Y. Tian (2021)
        Incorporating Learnable Membrane Time Constants to Enhance Learning
        of Spiking Neural Networks. Proc. IEEE/CVF Int. Conf. Computer
        Vision (ICCV), pp. 2661-2671.*"""

        @staticmethod
        def forward(ctx, input_, alpha=2.0):
            ctx.save_for_backward(input_)
            ctx.alpha = alpha
            out = (input_ > 0).float()
            return out

        @staticmethod
        def backward(ctx, grad_output):
            input_, = ctx.saved_tensors
            grad_input = grad_output.clone()
            grad = ctx.alpha / 2 / (1 + (torch.pi / 2 * ctx.alpha * input_).pow_(2)) * grad_input
            return grad, None

    def _process_input(self):
        if isinstance(self.input_size, int):
            return self.input_size
        elif isinstance(self.input_size, tuple):
            product = 1
            for item in self.input_size:
                if not isinstance(item, int):
                    raise ValueError('All elements in the tuple must be integers')
                product *= item
            return product
        else:
            raise TypeError('Input must be an integer or a tuple of integers')

    def weight_hh_enable(self):
        mask = torch.eye(self.hidden_size, self.hidden_size)
        self.rnn.weight_hh_l0.data = self.rnn.weight_hh_l0.data * mask

    def weight_ih_disable(self):
        with torch.no_grad():
            mask = torch.eye(self.input_size, self.input_size)
            self.rnn.weight_ih_l0.data = mask
            self.rnn.weight_ih_l0.requires_grad_(False)

    def process_tensor(self, input_):
        if isinstance(self.input_size, int):
            return input_
        elif isinstance(self.input_size, tuple):
            return input_.flatten(2)
        else:
            raise ValueError('input_size must be either an int or a tuple')

    def unprocess_tensor(self, input_):
        if isinstance(self.input_size, int):
            return input_
        elif isinstance(self.input_size, tuple):
            return input_.unflatten(2, self.input_size)
        else:
            raise ValueError('input_size must be either an int or a tuple')

    def grad_hook(self, grad):
        device = grad.device
        mask = torch.eye(self.hidden_size, self.hidden_size, device=device)
        return grad * mask

    def _beta_to_weight_hh(self):
        with torch.no_grad():
            if self.beta is not None:
                if isinstance(self.beta, float) or isinstance(self.beta, int):
                    self.rnn.weight_hh_l0.fill_(self.beta)
                elif isinstance(self.beta, torch.Tensor) or isinstance(self.beta, torch.FloatTensor):
                    if len(self.beta) == 1:
                        self.rnn.weight_hh_l0.fill_(self.beta[0])
                elif len(self.beta) == self.hidden_size:
                    for i in range(self.hidden_size):
                        self.rnn.weight_hh_l0.data[i].fill_(self.beta[i])
                else:
                    raise ValueError("Beta must be either a single value or of length 'hidden_size'.")

    def _beta_buffer(self, beta, learn_beta):
        if not isinstance(beta, torch.Tensor):
            if beta is not None:
                beta = torch.as_tensor([beta])
        self.register_buffer('beta', beta)

    def _graded_spikes_buffer(self, graded_spikes_factor, learn_graded_spikes_factor):
        if not isinstance(graded_spikes_factor, torch.Tensor):
            graded_spikes_factor = torch.as_tensor(graded_spikes_factor)
        if learn_graded_spikes_factor:
            self.graded_spikes_factor = nn.Parameter(graded_spikes_factor)
        else:
            self.register_buffer('graded_spikes_factor', graded_spikes_factor)

    def _threshold_buffer(self, threshold, learn_threshold):
        if not isinstance(threshold, torch.Tensor):
            threshold = torch.as_tensor(threshold)
        if learn_threshold:
            self.threshold = nn.Parameter(threshold)
        else:
            self.register_buffer('threshold', threshold)


class ATan(torch.autograd.Function):
    """
    Surrogate gradient of the Heaviside step function.

    **Forward pass:** Heaviside step function shifted.

        .. math::

            S=\\begin{cases} 1 & \\text{if U ≥ U$_{\\rm thr}$} \\\\
            0 & \\text{if U < U$_{\\rm thr}$}
            \\end{cases}

    **Backward pass:** Gradient of shifted arc-tan function.

        .. math::

                S&≈\\frac{1}{π}\\text{arctan}(πU \\frac{α}{2}) \\\\
                \\frac{∂S}{∂U}&=\\frac{1}{π}\\frac{1}{(1+(πU\\frac{α}{2})^2)}


    α defaults to 2, and can be modified by calling         ``surrogate.atan(alpha=2)``.

    Adapted from:

    *W. Fang, Z. Yu, Y. Chen, T. Masquelier, T. Huang,
    Y. Tian (2021) Incorporating Learnable Membrane Time Constants
    to Enhance Learning of Spiking Neural Networks. Proc. IEEE/CVF
    Int. Conf. Computer Vision (ICCV), pp. 2661-2671.*"""

    @staticmethod
    def forward(ctx, input_, alpha):
        ctx.save_for_backward(input_)
        ctx.alpha = alpha
        out = (input_ > 0).float()
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input_, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = ctx.alpha / 2 / (1 + (torch.pi / 2 * ctx.alpha * input_).pow_(2)) * grad_input
        return grad, None


def atan(alpha=2.0):
    """ArcTan surrogate gradient enclosed with a parameterized slope."""
    alpha = alpha

    def inner(x):
        return ATan.apply(x, alpha)
    return inner


class SpikingNeuron(nn.Module):
    """Parent class for spiking neuron models."""
    instances = []
    """Each :mod:`snntorch.SpikingNeuron` neuron
    (e.g., :mod:`snntorch.Synaptic`) will populate the
    :mod:`snntorch.SpikingNeuron.instances` list with a new entry.
    The list is used to initialize and clear neuron states when the
    argument `init_hidden=True`."""
    reset_dict = {'subtract': 0, 'zero': 1, 'none': 2}

    def __init__(self, threshold=1.0, spike_grad=None, surrogate_disable=False, init_hidden=False, inhibition=False, learn_threshold=False, reset_mechanism='subtract', state_quant=False, output=False, graded_spikes_factor=1.0, learn_graded_spikes_factor=False):
        super().__init__()
        SpikingNeuron.instances.append(self)
        if surrogate_disable:
            self.spike_grad = self._surrogate_bypass
        elif spike_grad == None:
            self.spike_grad = atan()
        else:
            self.spike_grad = spike_grad
        self.init_hidden = init_hidden
        self.inhibition = inhibition
        self.output = output
        self.surrogate_disable = surrogate_disable
        self._snn_cases(reset_mechanism, inhibition)
        self._snn_register_buffer(threshold=threshold, learn_threshold=learn_threshold, reset_mechanism=reset_mechanism, graded_spikes_factor=graded_spikes_factor, learn_graded_spikes_factor=learn_graded_spikes_factor)
        self._reset_mechanism = reset_mechanism
        self.state_quant = state_quant

    def fire(self, mem):
        """Generates spike if mem > threshold.
        Returns spk."""
        if self.state_quant:
            mem = self.state_quant(mem)
        mem_shift = mem - self.threshold
        spk = self.spike_grad(mem_shift)
        spk = spk * self.graded_spikes_factor
        return spk

    def fire_inhibition(self, batch_size, mem):
        """Generates spike if mem > threshold, only for the largest membrane.
        All others neurons will be inhibited for that time step.
        Returns spk."""
        mem_shift = mem - self.threshold
        index = torch.argmax(mem_shift, dim=1)
        spk_tmp = self.spike_grad(mem_shift)
        mask_spk1 = torch.zeros_like(spk_tmp)
        mask_spk1[torch.arange(batch_size), index] = 1
        spk = spk_tmp * mask_spk1
        return spk

    def mem_reset(self, mem):
        """Generates detached reset signal if mem > threshold.
        Returns reset."""
        mem_shift = mem - self.threshold
        reset = self.spike_grad(mem_shift).clone().detach()
        return reset

    def _snn_cases(self, reset_mechanism, inhibition):
        self._reset_cases(reset_mechanism)
        if inhibition:
            warn('Inhibition is an unstable feature that has only been tested for dense (fully-connected) layers. Use with caution!', UserWarning)

    def _reset_cases(self, reset_mechanism):
        if reset_mechanism != 'subtract' and reset_mechanism != 'zero' and reset_mechanism != 'none':
            raise ValueError("reset_mechanism must be set to either 'subtract', 'zero', or 'none'.")

    def _snn_register_buffer(self, threshold, learn_threshold, reset_mechanism, graded_spikes_factor, learn_graded_spikes_factor):
        """Set variables as learnable parameters else register them in the
        buffer."""
        self._threshold_buffer(threshold, learn_threshold)
        self._graded_spikes_buffer(graded_spikes_factor, learn_graded_spikes_factor)
        try:
            if torch.is_tensor(self.reset_mechanism_val):
                self.reset_mechanism = list(SpikingNeuron.reset_dict)[self.reset_mechanism_val]
        except AttributeError:
            self._reset_mechanism_buffer(reset_mechanism)

    def _graded_spikes_buffer(self, graded_spikes_factor, learn_graded_spikes_factor):
        if not isinstance(graded_spikes_factor, torch.Tensor):
            graded_spikes_factor = torch.as_tensor(graded_spikes_factor)
        if learn_graded_spikes_factor:
            self.graded_spikes_factor = nn.Parameter(graded_spikes_factor)
        else:
            self.register_buffer('graded_spikes_factor', graded_spikes_factor)

    def _threshold_buffer(self, threshold, learn_threshold):
        if not isinstance(threshold, torch.Tensor):
            threshold = torch.as_tensor(threshold)
        if learn_threshold:
            self.threshold = nn.Parameter(threshold)
        else:
            self.register_buffer('threshold', threshold)

    def _reset_mechanism_buffer(self, reset_mechanism):
        """Assign mapping to each reset mechanism state.
        Must be of type tensor to store in register buffer. See reset_dict
        for mapping."""
        reset_mechanism_val = torch.as_tensor(SpikingNeuron.reset_dict[reset_mechanism])
        self.register_buffer('reset_mechanism_val', reset_mechanism_val)

    def _V_register_buffer(self, V, learn_V):
        if not isinstance(V, torch.Tensor):
            V = torch.as_tensor(V)
        if learn_V:
            self.V = nn.Parameter(V)
        else:
            self.register_buffer('V', V)

    @property
    def reset_mechanism(self):
        """If reset_mechanism is modified, reset_mechanism_val is triggered
        to update.
        0: subtract, 1: zero, 2: none."""
        return self._reset_mechanism

    @reset_mechanism.setter
    def reset_mechanism(self, new_reset_mechanism):
        self._reset_cases(new_reset_mechanism)
        self.reset_mechanism_val = torch.as_tensor(SpikingNeuron.reset_dict[new_reset_mechanism])
        self._reset_mechanism = new_reset_mechanism

    @classmethod
    def init(cls):
        """Removes all items from :mod:`snntorch.SpikingNeuron.instances`
        when called."""
        cls.instances = []

    @staticmethod
    def detach(*args):
        """Used to detach input arguments from the current graph.
        Intended for use in truncated backpropagation through time where
        hidden state variables are global variables."""
        for state in args:
            state.detach_()

    @staticmethod
    def zeros(*args):
        """Used to clear hidden state variables to zero.
        Intended for use where hidden state variables are global variables."""
        for state in args:
            state = torch.zeros_like(state)

    @staticmethod
    def _surrogate_bypass(input_):
        return (input_ > 0).float()


class LIF(SpikingNeuron):
    """Parent class for leaky integrate and fire neuron models."""

    def __init__(self, beta, threshold=1.0, spike_grad=None, surrogate_disable=False, init_hidden=False, inhibition=False, learn_beta=False, learn_threshold=False, reset_mechanism='subtract', state_quant=False, output=False, graded_spikes_factor=1.0, learn_graded_spikes_factor=False):
        super().__init__(threshold, spike_grad, surrogate_disable, init_hidden, inhibition, learn_threshold, reset_mechanism, state_quant, output, graded_spikes_factor, learn_graded_spikes_factor)
        self._lif_register_buffer(beta, learn_beta)
        self._reset_mechanism = reset_mechanism

    def _lif_register_buffer(self, beta, learn_beta):
        """Set variables as learnable parameters else register them in the
        buffer."""
        self._beta_buffer(beta, learn_beta)

    def _beta_buffer(self, beta, learn_beta):
        if not isinstance(beta, torch.Tensor):
            beta = torch.as_tensor(beta)
        if learn_beta:
            self.beta = nn.Parameter(beta)
        else:
            self.register_buffer('beta', beta)

    def _V_register_buffer(self, V, learn_V):
        if V is not None:
            if not isinstance(V, torch.Tensor):
                V = torch.as_tensor(V)
        if learn_V:
            self.V = nn.Parameter(V)
        else:
            self.register_buffer('V', V)


class RecurrentOneToOne(nn.Module):

    def __init__(self, V):
        super().__init__()
        self.V = V

    def forward(self, x):
        return x * self.V


class RLeaky(LIF):
    """
    First-order recurrent leaky integrate-and-fire neuron model.
    Input is assumed to be a current injection appended to the voltage
    spike output.
    Membrane potential decays exponentially with rate beta.
    For :math:`U[T] > U_{\\rm thr} ⇒ S[T+1] = 1`.

    If `reset_mechanism = "subtract"`, then :math:`U[t+1]` will have
    `threshold` subtracted from it whenever the neuron emits a spike:

    .. math::

            U[t+1] = βU[t] + I_{\\rm in}[t+1] + V(S_{\\rm out}[t]) -
            RU_{\\rm thr}

    Where :math:`V(\\cdot)` acts either as a linear layer, a convolutional
    operator, or elementwise product on :math:`S_{\\rm out}`.

    * If `all_to_all = "True"` and `linear_features` is specified, then         :math:`V(\\cdot)` acts as a recurrent linear layer of the         same size as :math:`S_{\\rm out}`.
    * If `all_to_all = "True"` and `conv2d_channels` and `kernel_size` are         specified, then :math:`V(\\cdot)` acts as a recurrent convlutional         layer         with padding to ensure the output matches the size of the input.
    * If `all_to_all = "False"`, then :math:`V(\\cdot)` acts as an         elementwise multiplier with :math:`V`.

    * If `reset_mechanism = "zero"`, then :math:`U[t+1]` will be set to `0`         whenever the neuron emits a spike:

    .. math::
            U[t+1] = βU[t] + I_{\\rm in}[t+1] +  V(S_{\\rm out}[t]) -
            R(βU[t] + I_{\\rm in}[t+1] +  V(S_{\\rm out}[t]))

    * :math:`I_{\\rm in}` - Input current
    * :math:`U` - Membrane potential
    * :math:`U_{\\rm thr}` - Membrane threshold
    * :math:`S_{\\rm out}` - Output spike
    * :math:`R` - Reset mechanism: if active, :math:`R = 1`, otherwise         :math:`R = 0`
    * :math:`β` - Membrane potential decay rate
    * :math:`V` - Explicit recurrent weight when `all_to_all=False`

    Example::

        import torch
        import torch.nn as nn
        import snntorch as snn

        beta = 0.5 # decay rate
        V1 = 0.5 # shared recurrent connection
        V2 = torch.rand(num_outputs) # unshared recurrent connections

        # Define Network
        class Net(nn.Module):
            def __init__(self):
                super().__init__()

                # initialize layers
                self.fc1 = nn.Linear(num_inputs, num_hidden)

                # Default RLeaky Layer where recurrent connections
                # are initialized using PyTorch defaults in nn.Linear.
                self.lif1 = snn.RLeaky(beta=beta,
                            linear_features=num_hidden)

                self.fc2 = nn.Linear(num_hidden, num_outputs)

                # each neuron has a single connection back to itself
                # where the output spike is scaled by V.
                # For `all_to_all = False`, V can be shared between
                # neurons (e.g., V1) or unique / unshared between
                # neurons (e.g., V2).
                # V is learnable by default.
                self.lif2 = snn.RLeaky(beta=beta, all_to_all=False, V=V1)

            def forward(self, x):
                # Initialize hidden states at t=0
                spk1, mem1 = self.lif1.init_rleaky()
                spk2, mem2 = self.lif2.init_rleaky()

                # Record output layer spikes and membrane
                spk2_rec = []
                mem2_rec = []

                # time-loop
                for step in range(num_steps):
                    cur1 = self.fc1(x)
                    spk1, mem1 = self.lif1(cur1, spk1, mem1)
                    cur2 = self.fc2(spk1)
                    spk2, mem2 = self.lif2(cur2, spk2, mem2)

                    spk2_rec.append(spk2)
                    mem2_rec.append(mem2)

                # convert lists to tensors
                spk2_rec = torch.stack(spk2_rec)
                mem2_rec = torch.stack(mem2_rec)

                return spk2_rec, mem2_rec

    :param beta: membrane potential decay rate. Clipped between 0 and 1
        during the forward-pass. May be a single-valued tensor (i.e., equal
        decay rate for all neurons in a layer), or multi-valued
        (one weight per neuron).
    :type beta: float or torch.tensor

    :param V: Recurrent weights to scale output spikes, only used when
        `all_to_all=False`. Defaults to 1.
    :type V: float or torch.tensor

    :param all_to_all: Enables output spikes to be connected in dense or
        convolutional recurrent structures instead of 1-to-1 connections.
        Defaults to True.
    :type all_to_all: bool, optional

    :param linear_features: Size of each output sample. Must be specified
        if `all_to_all=True` and the input data is 1D. Defaults to None
    :type linear_features: int, optional

    :param conv2d_channels: Number of channels in each output sample. Must
        be specified if `all_to_all=True` and the input data is 3D.
        Defaults to None
    :type conv2d_channels: int, optional

    :param kernel_size:  Size of the convolving kernel. Must be
        specified if `all_to_all=True` and the input data is 3D.
        Defaults to None
    :type kernel_size: int or tuple

    :param threshold: Threshold for :math:`mem` to reach in order to
        generate a spike `S=1`. Defaults to 1 :type threshold: float,
        optional

    :param spike_grad: Surrogate gradient for the term dS/dU. Defaults
        to None (corresponds to ATan surrogate gradient. See
        `snntorch.surrogate` for more options)
    :type spike_grad: surrogate gradient function from snntorch.surrogate,
        optional

    :param surrogate_disable: Disables surrogate gradients regardless of
        `spike_grad` argument. Useful for ONNX compatibility. Defaults
        to False
    :type surrogate_disable: bool, Optional

    :param init_hidden: Instantiates state variables as instance variables.
        Defaults to False :type init_hidden: bool, optional

    :param inhibition: If `True`, suppresses all spiking other than the
        neuron with the highest state. Defaults to False :type inhibition:
        bool, optional

    :param learn_beta: Option to enable learnable beta. Defaults to False
    :type learn_beta: bool, optional

    :param learn_recurrent: Option to enable learnable recurrent weights.
        Defaults to True
    :type learn_recurrent: bool, optional

    :param learn_threshold: Option to enable learnable threshold.
        Defaults to False
    :type learn_threshold: bool, optional

    :param reset_mechanism: Defines the reset mechanism applied to
        :math:`mem` each time the threshold is met.
        Reset-by-subtraction: "subtract", reset-to-zero: "zero",
        none: "none". Defaults to "subtract"
    :type reset_mechanism: str, optional

    :param state_quant: If specified, hidden state :math:`mem` is
        quantized to a valid state for the forward pass. Defaults to False
    :type state_quant: quantization function from snntorch.quant, optional

    :param output: If `True` as well as `init_hidden=True`, states are
        returned when neuron is called. Defaults to False :type output:
        bool, optional




    Inputs: \\input_, spk_0, mem_0
        - **input_** of shape `(batch, input_size)`: tensor containing
        input
          features
        - **spk_0** of shape `(batch, input_size)`: tensor containing
        output
          spike features
        - **mem_0** of shape `(batch, input_size)`: tensor containing the
          initial membrane potential for each element in the batch.

    Outputs: spk_1, mem_1
        - **spk_1** of shape `(batch, input_size)`: tensor containing the
        output
          spikes.
        - **mem_1** of shape `(batch, input_size)`: tensor containing
        the next
          membrane potential for each element in the batch

    Learnable Parameters:
        - **RLeaky.beta** (torch.Tensor) - optional learnable weights
        must be
          manually passed in, of shape `1` or (input_size).
        - **RLeaky.recurrent.weight** (torch.Tensor) - optional learnable
          weights are automatically generated if `all_to_all=True`.
          `RLeaky.recurrent` stores a `nn.Linear` or `nn.Conv2d` layer
          depending on input arguments provided.
        - **RLeaky.V** (torch.Tensor) - optional learnable weights must be
          manually passed in, of shape `1` or (input_size). It is only used
          where `all_to_all=False` for 1-to-1 recurrent connections.
        - **RLeaky.threshold** (torch.Tensor) - optional learnable
            thresholds must be manually passed in, of shape `1` or``
            (input_size).

    """

    def __init__(self, beta, V=1.0, all_to_all=True, linear_features=None, conv2d_channels=None, kernel_size=None, threshold=1.0, spike_grad=None, surrogate_disable=False, init_hidden=False, inhibition=False, learn_beta=False, learn_threshold=False, learn_recurrent=True, reset_mechanism='subtract', state_quant=False, output=False, reset_delay=True):
        super().__init__(beta, threshold, spike_grad, surrogate_disable, init_hidden, inhibition, learn_beta, learn_threshold, reset_mechanism, state_quant, output)
        self.all_to_all = all_to_all
        self.learn_recurrent = learn_recurrent
        self.linear_features = linear_features
        self.kernel_size = kernel_size
        self.conv2d_channels = conv2d_channels
        self._rleaky_init_cases()
        if self.all_to_all:
            self._init_recurrent_net()
        else:
            self._V_register_buffer(V, learn_recurrent)
            self._init_recurrent_one_to_one()
        if not learn_recurrent:
            self._disable_recurrent_grad()
        self._init_mem()
        if self.reset_mechanism_val == 0:
            self.state_function = self._base_sub
        elif self.reset_mechanism_val == 1:
            self.state_function = self._base_zero
        elif self.reset_mechanism_val == 2:
            self.state_function = self._base_int
        self.reset_delay = reset_delay

    def _init_mem(self):
        spk = torch.zeros(0)
        mem = torch.zeros(0)
        self.register_buffer('spk', spk, False)
        self.register_buffer('mem', mem, False)

    def reset_mem(self):
        self.spk = torch.zeros_like(self.spk, device=self.spk.device)
        self.mem = torch.zeros_like(self.mem, device=self.mem.device)
        return self.spk, self.mem

    def init_rleaky(self):
        """Deprecated, use :class:`RLeaky.reset_mem` instead"""
        return self.reset_mem()

    def forward(self, input_, spk=None, mem=None):
        if not spk == None:
            self.spk = spk
        if not mem == None:
            self.mem = mem
        if self.init_hidden and (not mem == None or not spk == None):
            raise TypeError('When `init_hidden=True`,RLeaky expects 1 input argument.')
        if not self.spk.shape == input_.shape:
            self.spk = torch.zeros_like(input_, device=self.spk.device)
        if not self.mem.shape == input_.shape:
            self.mem = torch.zeros_like(input_, device=self.mem.device)
        self.reset = self.mem_reset(self.mem)
        self.mem = self.state_function(input_)
        if self.state_quant:
            self.mem = self.state_quant(self.mem)
        if self.inhibition:
            self.spk = self.fire_inhibition(self.mem.size(0), self.mem)
        else:
            self.spk = self.fire(self.mem)
        if not self.reset_delay:
            do_reset = self.spk / self.graded_spikes_factor - self.reset
            if self.reset_mechanism_val == 0:
                self.mem = self.mem - do_reset * self.threshold
            elif self.reset_mechanism_val == 1:
                self.mem = self.mem - do_reset * self.mem
        if self.output:
            return self.spk, self.mem
        elif self.init_hidden:
            return self.spk
        else:
            return self.spk, self.mem

    def _init_recurrent_net(self):
        if self.all_to_all:
            if self.linear_features:
                self._init_recurrent_linear()
            elif self.kernel_size is not None:
                self._init_recurrent_conv2d()
        else:
            self._init_recurrent_one_to_one()

    def _init_recurrent_linear(self):
        self.recurrent = nn.Linear(self.linear_features, self.linear_features)

    def _init_recurrent_conv2d(self):
        self._init_padding()
        self.recurrent = nn.Conv2d(in_channels=self.conv2d_channels, out_channels=self.conv2d_channels, kernel_size=self.kernel_size, padding=self.padding)

    def _init_padding(self):
        if type(self.kernel_size) is int:
            self.padding = self.kernel_size // 2, self.kernel_size // 2
        else:
            self.padding = self.kernel_size[0] // 2, self.kernel_size[1] // 2

    def _init_recurrent_one_to_one(self):
        self.recurrent = RecurrentOneToOne(self.V)

    def _disable_recurrent_grad(self):
        for param in self.recurrent.parameters():
            param.requires_grad = False

    def _base_state_function(self, input_):
        base_fn = self.beta.clamp(0, 1) * self.mem + input_ + self.recurrent(self.spk)
        return base_fn

    def _base_sub(self, input_):
        return self._base_state_function(input_) - self.reset * self.threshold

    def _base_zero(self, input_):
        return self._base_state_function(input_) - self.reset * self._base_state_function(input_)

    def _base_int(self, input_):
        return self._base_state_function(input_)

    def _rleaky_init_cases(self):
        all_to_all_bool = bool(self.all_to_all)
        linear_features_bool = self.linear_features
        conv2d_channels_bool = bool(self.conv2d_channels)
        kernel_size_bool = bool(self.kernel_size)
        if all_to_all_bool:
            if not linear_features_bool:
                if not (conv2d_channels_bool or kernel_size_bool):
                    raise TypeError('When `all_to_all=True`, RLeaky requires either`linear_features` or (`conv2d_channels` and `kernel_size`) to be specified. The shape should match the shape of the output spike of the layer.')
                elif conv2d_channels_bool ^ kernel_size_bool:
                    raise TypeError('`conv2d_channels` and `kernel_size` must both bespecified. The shape of `conv2d_channels` should match the shape of the outputspikes.')
            elif linear_features_bool and kernel_size_bool or linear_features_bool and conv2d_channels_bool:
                raise TypeError('`linear_features` cannot be specified at the same time as`conv2d_channels` or `kernel_size`. A linear layer and conv2d layer cannot bothbe specified at the same time.')
        elif linear_features_bool or conv2d_channels_bool or kernel_size_bool:
            raise TypeError('When `all_to_all`=False, none of `linear_features`,`conv2d_channels`, or `kernel_size` should be specified. The weight `V` is usedinstead.')

    @classmethod
    def detach_hidden(cls):
        """Returns the hidden states, detached from the current graph.
        Intended
        for use in truncated backpropagation through time where hidden state
        variables
        are instance variables."""
        for layer in range(len(cls.instances)):
            if isinstance(cls.instances[layer], RLeaky):
                cls.instances[layer].mem.detach_()
                cls.instances[layer].spk.detach_()

    @classmethod
    def reset_hidden(cls):
        """Used to clear hidden state variables to zero.
        Intended for use where hidden state variables are instance variables.
        Assumes hidden states have a batch dimension already."""
        for layer in range(len(cls.instances)):
            if isinstance(cls.instances[layer], RLeaky):
                cls.instances[layer].spk, cls.instances[layer].mem = cls.instances[layer].init_rleaky()


class RSynaptic(LIF):
    """
    2nd order recurrent leaky integrate and fire neuron model accounting for
    synaptic conductance.
    The synaptic current jumps upon spike arrival, which causes a jump in
    membrane potential.
    Synaptic current and membrane potential decay exponentially with rates
    of alpha and beta, respectively.
    For :math:`U[T] > U_{\\rm thr} ⇒ S[T+1] = 1`.

    If `reset_mechanism = "subtract"`, then :math:`U[t+1]` will have
    `threshold` subtracted from it whenever the neuron emits a spike:

    .. math::

            I_{\\rm syn}[t+1] = αI_{\\rm syn}[t] + V(S_{\\rm out}[t]
            + I_{\\rm in}[t+1] \\\\
            U[t+1] = βU[t] + I_{\\rm syn}[t+1] - RU_{\\rm thr}

    Where :math:`V(\\cdot)` acts either as a linear layer, a convolutional
    operator, or elementwise product on :math:`S_{\\rm out}`.

    * If `all_to_all = "True"` and `linear_features` is specified, then         :math:`V(\\cdot)` acts as a recurrent linear layer of the same size         as :math:`S_{\\rm out}`.
    * If `all_to_all = "True"` and `conv2d_channels` and `kernel_size` are         specified, then :math:`V(\\cdot)` acts as a recurrent convlutional         layer with padding to ensure the output matches the size of the input.
    * If `all_to_all = "False"`, then :math:`V(\\cdot)` acts as an         elementwise multiplier with :math:`V`.

    * If `reset_mechanism = "zero"`, then :math:`U[t+1]` will be set to `0`
        whenever the neuron emits a spike:

    .. math::

            I_{\\rm syn}[t+1] = αI_{\\rm syn}[t] + VS_{\\rm out}[t]
            + I_{\\rm in}[t+1] \\\\
            U[t+1] = βU[t] + I_{\\rm syn}[t+1] - R(βU[t] + I_{\\rm syn}[t+1])

    * :math:`I_{\\rm syn}` - Synaptic current
    * :math:`I_{\\rm in}` - Input current
    * :math:`U` - Membrane potential
    * :math:`U_{\\rm thr}` - Membrane threshold
    * :math:`S_{\\rm out}` - Output spike
    * :math:`R` - Reset mechanism: if active, :math:`R = 1`, otherwise         :math:`R = 0`
    * :math:`α` - Synaptic current decay rate
    * :math:`β` - Membrane potential decay rate
    * :math:`V` - Explicit recurrent weight when `all_to_all=False`

    Example::

        import torch
        import torch.nn as nn
        import snntorch as snn

        beta = 0.5 # decay rate
        V1 = 0.5 # shared recurrent connection
        V2 = torch.rand(num_outputs) # unshared recurrent connections

        # Define Network
        class Net(nn.Module):
            def __init__(self):
                super().__init__()

                # initialize layers
                self.fc1 = nn.Linear(num_inputs, num_hidden)

                # Default RLeaky Layer where recurrent connections
                # are initialized using PyTorch defaults in nn.Linear.
                self.lif1 = snn.RLeaky(beta=beta,
                            linear_features=num_hidden)

                self.fc2 = nn.Linear(num_hidden, num_outputs)

                # each neuron has a single connection back to itself
                # where the output spike is scaled by V.
                # For `all_to_all = True`, V can be shared between
                # neurons (e.g., V1) or unique / unshared between
                # neurons (e.g., V2).
                # V is learnable by default.
                self.lif2 = snn.RLeaky(beta=beta, all_to_all=False, V=V1)

            def forward(self, x):
                # Initialize hidden states at t=0
                spk1, syn1, mem1 = self.lif1.init_rsynaptic()
                spk2, syn2, mem2 = self.lif2.init_rsynaptic()

                # Record output layer spikes and membrane
                spk2_rec = []
                mem2_rec = []

                # time-loop
                for step in range(num_steps):
                    cur1 = self.fc1(x)
                    spk1, syn1, mem1 = self.lif1(cur1, spk1, syn1, mem1)
                    cur2 = self.fc2(spk1)
                    spk2, syn2, mem2 = self.lif2(cur2, spk2, syn2, mem2)

                    spk2_rec.append(spk2)
                    mem2_rec.append(mem2)

                # convert lists to tensors
                spk2_rec = torch.stack(spk2_rec)
                mem2_rec = torch.stack(mem2_rec)

                return spk2_rec, mem2_rec

    :param alpha: synaptic current decay rate. Clipped between 0 and 1
        during the forward-pass. May be a single-valued tensor (i.e., equal
        decay rate for all neurons in a layer), or multi-valued (one weight per
        neuron).
    :type alpha: float or torch.tensor

    :param beta: membrane potential decay rate. Clipped between 0 and 1
        during the forward-pass. May be a single-valued tensor (i.e., equal
        decay rate for all neurons in a layer), or multi-valued (one weight per
        neuron).
    :type beta: float or torch.tensor

    :param V: Recurrent weights to scale output spikes, only used when
        `all_to_all=False`. Defaults to 1.
    :type V: float or torch.tensor

    :param all_to_all: Enables output spikes to be connected in dense or
        convolutional recurrent structures instead of 1-to-1 connections.
        Defaults to True.
    :type all_to_all: bool, optional

    :param linear_features: Size of each output sample. Must be specified if
        `all_to_all=True` and the input data is 1D. Defaults to None
    :type linear_features: int, optional

    :param conv2d_channels: Number of channels in each output sample. Must
        be specified if `all_to_all=True` and the input data is 3D. Defaults to
        None
    :type conv2d_channels: int, optional

    :param kernel_size:  Size of the convolving kernel. Must be specified if
        `all_to_all=True` and the input data is 3D. Defaults to None
    :type kernel_size: int or tuple

    :param threshold: Threshold for :math:`mem` to reach in order to
        generate a spike `S=1`. Defaults to 1
    :type threshold: float, optional

    :param spike_grad: Surrogate gradient for the term dS/dU. Defaults to
        None (corresponds to ATan surrogate gradient. See
        `snntorch.surrogate` for more options)
    :type spike_grad: surrogate gradient function from snntorch.surrogate,
        optional

    :param surrogate_disable: Disables surrogate gradients regardless of
        `spike_grad` argument. Useful for ONNX compatibility. Defaults
        to False
    :type surrogate_disable: bool, Optional

    :param init_hidden: Instantiates state variables as instance variables.
        Defaults to False
    :type init_hidden: bool, optional

    :param inhibition: If `True`, suppresses all spiking other than the
        neuron with the highest state. Defaults to False
    :type inhibition: bool, optional

    :param learn_alpha: Option to enable learnable alpha. Defaults to False
    :type learn_alpha: bool, optional

    :param learn_beta: Option to enable learnable beta. Defaults to False
    :type learn_beta: bool, optional

    :param learn_recurrent: Option to enable learnable recurrent weights.
        Defaults to True
    :type learn_recurrent: bool, optional

    :param learn_threshold: Option to enable learnable threshold.
        Defaults to False
    :type learn_threshold: bool, optional

    :param reset_mechanism: Defines the reset mechanism applied to     :math:`mem` each time the threshold is met. Reset-by-subtraction:
        "subtract", reset-to-zero: "zero, none: "none". Defaults to "subtract"
    :type reset_mechanism: str, optional

    :param state_quant: If specified, hidden states :math:`mem` and     :math:`syn` are quantized to a valid state for the forward pass.         Defaults to False
    :type state_quant: quantization function from snntorch.quant, optional

    :param output: If `True` as well as `init_hidden=True`, states are
        returned when neuron is called. Defaults to False
    :type output: bool, optional


    Inputs: \\input_, spk_0, syn_0, mem_0
        - **input_** of shape `(batch, input_size)`: tensor containing input         features
        - **spk_0** of shape `(batch, input_size)`: tensor containing output         spike features
        - **syn_0** of shape `(batch, input_size)`: tensor containing input         features
        - **mem_0** of shape `(batch, input_size)`: tensor containing the         initial membrane potential for each element in the batch.

    Outputs: spk_1, syn_1, mem_1
        - **spk_1** of shape `(batch, input_size)`: tensor containing the         output spikes.
        - **syn_1** of shape `(batch, input_size)`: tensor containing the         next synaptic current for each element in the batch
        - **mem_1** of shape `(batch, input_size)`: tensor containing the         next membrane potential for each element in the batch

    Learnable Parameters:
        - **RSynaptic.alpha** (torch.Tensor) - optional learnable weights          must be manually passed in, of shape `1` or (input_size).
        - **RSynaptic.beta** (torch.Tensor) - optional learnable weights         must be manually passed in, of shape `1` or (input_size).
        - **RSynaptic.recurrent.weight** (torch.Tensor) - optional learnable         weights are automatically generated if `all_to_all=True`.         `RSynaptic.recurrent` stores a `nn.Linear` or `nn.Conv2d` layer         depending on input arguments provided.
        - **RSynaptic.V** (torch.Tensor) - optional learnable weights must         be manually passed in, of shape `1` or (input_size). It is only used         where `all_to_all=False` for 1-to-1 recurrent connections.
        - **RSynaptic.threshold** (torch.Tensor) - optional learnable         thresholds must be manually passed in, of shape `1` or`` (input_size).

"""

    def __init__(self, alpha, beta, V=1.0, all_to_all=True, linear_features=None, conv2d_channels=None, kernel_size=None, threshold=1.0, spike_grad=None, surrogate_disable=False, init_hidden=False, inhibition=False, learn_alpha=False, learn_beta=False, learn_threshold=False, learn_recurrent=True, reset_mechanism='subtract', state_quant=False, output=False, reset_delay=True):
        super().__init__(beta, threshold, spike_grad, surrogate_disable, init_hidden, inhibition, learn_beta, learn_threshold, reset_mechanism, state_quant, output)
        self.all_to_all = all_to_all
        self.learn_recurrent = learn_recurrent
        self.linear_features = linear_features
        self.kernel_size = kernel_size
        self.conv2d_channels = conv2d_channels
        self._rsynaptic_init_cases()
        if self.all_to_all:
            self._init_recurrent_net()
        else:
            self._V_register_buffer(V, learn_recurrent)
            self._init_recurrent_one_to_one()
        if not learn_recurrent:
            self._disable_recurrent_grad()
        self._alpha_register_buffer(alpha, learn_alpha)
        self._init_mem()
        if self.reset_mechanism_val == 0:
            self.state_function = self._base_sub
        elif self.reset_mechanism_val == 1:
            self.state_function = self._base_zero
        elif self.reset_mechanism_val == 2:
            self.state_function = self._base_int
        self.reset_delay = reset_delay

    def _init_mem(self):
        spk = torch.zeros(0)
        syn = torch.zeros(0)
        mem = torch.zeros(0)
        self.register_buffer('spk', spk, False)
        self.register_buffer('syn', syn, False)
        self.register_buffer('mem', mem, False)

    def reset_mem(self):
        self.spk = torch.zeros_like(self.spk, device=self.spk.device)
        self.syn = torch.zeros_like(self.syn, device=self.syn.device)
        self.mem = torch.zeros_like(self.mem, device=self.mem.device)
        return self.spk, self.syn, self.mem

    def init_rsynaptic(self):
        """Deprecated, use :class:`RSynaptic.reset_mem` instead"""
        return self.reset_mem()

    def forward(self, input_, spk=None, syn=None, mem=None):
        if not spk == None:
            self.spk = spk
        if not syn == None:
            self.syn = syn
        if not mem == None:
            self.mem = mem
        if self.init_hidden and (not spk == None or not syn == None or not mem == None):
            raise TypeError('When `init_hidden=True`, RSynaptic expects 1 input argument.')
        if not self.spk.shape == input_.shape:
            self.spk = torch.zeros_like(input_, device=self.spk.device)
        if not self.syn.shape == input_.shape:
            self.syn = torch.zeros_like(input_, device=self.syn.device)
        if not self.mem.shape == input_.shape:
            self.mem = torch.zeros_like(input_, device=self.mem.device)
        self.reset = self.mem_reset(self.mem)
        self.syn, self.mem = self.state_function(input_)
        if self.state_quant:
            self.syn = self.state_quant(self.syn)
            self.mem = self.state_quant(self.mem)
        if self.inhibition:
            self.spk = self.fire_inhibition(self.mem.size(0), self.mem)
        else:
            self.spk = self.fire(self.mem)
        if not self.reset_delay:
            do_reset = spk / self.graded_spikes_factor - self.reset
            if self.reset_mechanism_val == 0:
                mem = mem - do_reset * self.threshold
            elif self.reset_mechanism_val == 1:
                mem = mem - do_reset * mem
        if self.output:
            return self.spk, self.syn, self.mem
        elif self.init_hidden:
            return self.spk
        else:
            return self.spk, self.syn, self.mem

    def _init_recurrent_net(self):
        if self.all_to_all:
            if self.linear_features:
                self._init_recurrent_linear()
            elif self.kernel_size is not None:
                self._init_recurrent_conv2d()
        else:
            self._init_recurrent_one_to_one()

    def _init_recurrent_linear(self):
        self.recurrent = nn.Linear(self.linear_features, self.linear_features)

    def _init_recurrent_conv2d(self):
        self._init_padding()
        self.recurrent = nn.Conv2d(in_channels=self.conv2d_channels, out_channels=self.conv2d_channels, kernel_size=self.kernel_size, padding=self.padding)

    def _init_padding(self):
        if type(self.kernel_size) is int:
            self.padding = self.kernel_size // 2, self.kernel_size // 2
        else:
            self.padding = self.kernel_size[0] // 2, self.kernel_size[1] // 2

    def _init_recurrent_one_to_one(self):
        self.recurrent = RecurrentOneToOne(self.V)

    def _disable_recurrent_grad(self):
        for param in self.recurrent.parameters():
            param.requires_grad = False

    def _base_state_function(self, input_):
        base_fn_syn = self.alpha.clamp(0, 1) * self.syn + input_ + self.recurrent(self.spk)
        base_fn_mem = self.beta.clamp(0, 1) * self.mem + base_fn_syn
        return base_fn_syn, base_fn_mem

    def _base_state_reset_zero(self, input_):
        base_fn_syn = self.alpha.clamp(0, 1) * self.syn + input_ + self.recurrent(self.spk)
        base_fn_mem = self.beta.clamp(0, 1) * self.mem + base_fn_syn
        return 0, base_fn_mem

    def _base_sub(self, input_):
        syn, mem = self._base_state_function(input_)
        mem -= self.reset * self.threshold
        return syn, mem

    def _base_zero(self, input_):
        syn, mem = self._base_state_function(input_)
        syn2, mem2 = self._base_state_reset_zero(input_)
        syn2 *= self.reset
        mem2 *= self.reset
        syn -= syn2
        mem -= mem2
        return syn, mem

    def _base_int(self, input_):
        return self._base_state_function(input_)

    def _alpha_register_buffer(self, alpha, learn_alpha):
        if not isinstance(alpha, torch.Tensor):
            alpha = torch.as_tensor(alpha)
        if learn_alpha:
            self.alpha = nn.Parameter(alpha)
        else:
            self.register_buffer('alpha', alpha)

    def _rsynaptic_init_cases(self):
        all_to_all_bool = bool(self.all_to_all)
        linear_features_bool = self.linear_features
        conv2d_channels_bool = bool(self.conv2d_channels)
        kernel_size_bool = bool(self.kernel_size)
        if all_to_all_bool:
            if not linear_features_bool:
                if not (conv2d_channels_bool or kernel_size_bool):
                    raise TypeError('When `all_to_all=True`, RSynaptic requires either `linear_features` or (`conv2d_channels` and `kernel_size`) to be specified. The shape should match the shape of the output spike of the layer.')
                elif conv2d_channels_bool ^ kernel_size_bool:
                    raise TypeError('`conv2d_channels` and `kernel_size` must both be specified. The shape of `conv2d_channels` should match the shape of the output spikes.')
            elif linear_features_bool and kernel_size_bool or linear_features_bool and conv2d_channels_bool:
                raise TypeError('`linear_features` cannot be specified at the same time as `conv2d_channels` or `kernel_size`. A linear layer and conv2d layer cannot both be specified at the same time.')
        elif linear_features_bool or conv2d_channels_bool or kernel_size_bool:
            raise TypeError('When `all_to_all`=False, none of `linear_features`, `conv2d_channels`, or `kernel_size` should be specified. The weight `V` is used instead.')

    @classmethod
    def detach_hidden(cls):
        """Returns the hidden states, detached from the current graph.
        Intended for use in truncated backpropagation through time where
        hidden state variables are instance variables."""
        for layer in range(len(cls.instances)):
            if isinstance(cls.instances[layer], RSynaptic):
                cls.instances[layer].syn.detach_()
                cls.instances[layer].mem.detach_()

    @classmethod
    def reset_hidden(cls):
        """Used to clear hidden state variables to zero.
        Intended for use where hidden state variables are instance
        variables."""
        for layer in range(len(cls.instances)):
            if isinstance(cls.instances[layer], RSynaptic):
                cls.instances[layer].syn = torch.zeros_like(cls.instances[layer].syn, device=cls.instances[layer].syn.device)
                cls.instances[layer].mem = torch.zeros_like(cls.instances[layer].mem, device=cls.instances[layer].mem.device)


class SConv2dLSTM(SpikingNeuron):
    """
    A spiking 2d convolutional long short-term memory cell.
    Hidden states are membrane potential and synaptic current
    :math:`mem, syn`, which correspond to the hidden and cell states
    :math:`h, c` in the original LSTM formulation.

    The input is expected to be of size :math:`(N, C_{in}, H_{in}, W_{in})`
    where :math:`N` is the batch size.

    Unlike the LSTM module in PyTorch, only one time step is simulated each
    time the cell is called.

    .. math::
            \\begin{array}{ll} \\\\
            i_t = \\sigma(W_{ii} ⋆ x_t + b_{ii} + W_{hi} ⋆ mem_{t-1} + b_{hi})
            \\\\
            f_t = \\sigma(W_{if} ⋆ x_t + b_{if} + W_{hf} mem_{t-1} + b_{hf})
            \\\\
            g_t = \\tanh(W_{ig} ⋆ x_t + b_{ig} + W_{hg} ⋆ mem_{t-1} + b_{hg})
            \\\\
            o_t = \\sigma(W_{io} ⋆ x_t + b_{io} + W_{ho} ⋆ mem_{t-1} + b_{ho})
            \\\\
            syn_t = f_t ∗  c_{t-1} + i_t ∗  g_t \\\\
            mem_t = o_t ∗  \\tanh(syn_t) \\\\
        \\end{array}

    where :math:`\\sigma` is the sigmoid function, ⋆ is the 2D
    cross-correlation operator and ∗ is the Hadamard product.
    The output state :math:`mem_{t+1}` is thresholded to determine whether
    an output spike is generated.
    To conform to standard LSTM state behavior, the default reset mechanism
    is set to `reset="none"`, i.e., no reset is applied. If this is changed,
    the reset is only applied to :math:`mem_t`.

    Options to apply max-pooling or average-pooling to the state
    :math:`mem_t` are also enabled. Note that it is preferable to apply
    pooling to the state rather than the spike, as it does not make sense
    to apply pooling to activations of 1's and 0's which may lead to random
    tie-breaking.

    Padding is automatically applied to ensure consistent sizes for
    hidden states from one time step to the next.

    At the moment, stride != 1 is not supported.

    Example::

        import torch
        import torch.nn as nn
        import snntorch as snn


        # Define Network
        class Net(nn.Module):
            def __init__(self):
                super().__init__()

                in_channels = 1
                out_channels = 8
                kernel_size = 3
                max_pool = 2
                avg_pool = 2
                flattened_input = 49 * 16
                num_outputs = 10
                beta = 0.5

                spike_grad_lstm = snn.surrogate.straight_through_estimator()
                spike_grad_fc = snn.surrogate.fast_sigmoid(slope=5)

                # initialize layers
                self.sclstm1 = snn.SConv2dLSTM(
                    in_channels,
                    out_channels,
                    kernel_size,
                    max_pool=max_pool,
                    spike_grad=spike_grad_lstm,
                )
                self.sclstm2 = snn.SConv2dLSTM(
                    out_channels,
                    out_channels,
                    kernel_size,
                    avg_pool=avg_pool,
                    spike_grad=spike_grad_lstm,
                )
                self.fc1 = nn.Linear(flattened_input, num_outputs)
                self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad_fc)

            def forward(self, x):
                # Initialize hidden states and outputs at t=0
                syn1, mem1 = self.lif1.init_sconv2dlstm()
                syn2, mem2 = self.lif1.init_sconv2dlstm()
                mem3 = self.lif3.init_leaky()

                # Record the final layer
                spk3_rec = []
                mem3_rec = []

                # Number of steps assuming x is [N, T, C, H, W] with
                # N = Batches, T = Time steps, C = Channels,
                # H = Height, W = Width
                num_steps = x.size()[1]

                for step in range(num_steps):
                    x_step = x[:, step, :, :, :]
                    spk1, syn1, mem1 = self.sclstm1(x_step, syn1, mem1)
                    spk2, syn2, mem2 = self.sclstm2(spk1, syn2, mem2)
                    cur = self.fc1(spk2.flatten(1))
                    spk3, mem3 = self.lif1(cur, mem3)

                    spk3_rec.append(spk3)
                    mem3_rec.append(mem3)

                return torch.stack(spk3_rec), torch.stack(mem3_rec)


    :param in_channels: number of input channels
    :type in_channels: int

    :param kernel_size: Size of the convolving kernel
    :type kernel_size: int, tuple, or list

    :param bias: If `True`, adds a learnable bias to the output. Defaults to
        `True`
    :type bias: bool, optional

    :param max_pool: Applies max-pooling to the hidden state :math:`mem`
        prior to thresholding if specified. Defaults to 0
    :type max_pool: int, tuple, or list, optional

    :param avg_pool: Applies average-pooling to the hidden state :math:`mem`
        prior to thresholding if specified. Defaults to 0
    :type avg_pool: int, tuple, or list, optional

    :param threshold: Threshold for :math:`mem` to reach in order to
        generate a spike `S=1`. Defaults to 1
    :type threshold: float, optional

    :param spike_grad: Surrogate gradient for the term dS/dU. Defaults to
        ATan surrogate gradient
    :type spike_grad: surrogate gradient function from snntorch.surrogate,
        optional

    :param surrogate_disable: Disables surrogate gradients regardless of
        `spike_grad` argument. Useful for ONNX compatibility. Defaults
        to False
    :type surrogate_disable: bool, Optional

    :param learn_threshold: Option to enable learnable threshold. Defaults
        to False
    :type learn_threshold: bool, optional

    :param init_hidden: Instantiates state variables as instance variables.
        Defaults to False
    :type init_hidden: bool, optional

    :param inhibition: If `True`, suppresses all spiking other than the
        neuron with the highest state. Defaults to False
    :type inhibition: bool, optional

    :param reset_mechanism: Defines the reset mechanism applied to     :math:`mem` each time the threshold is met. Reset-by-subtraction:         "subtract", reset-to-zero: "zero, none: "none". Defaults to "none"
    :type reset_mechanism: str, optional

    :param state_quant: If specified, hidden states :math:`mem` and     :math:`syn` are quantized to a valid state for the forward pass.         Defaults to False
    :type state_quant: quantization function from snntorch.quant, optional

    :param output: If `True` as well as `init_hidden=True`, states are
        returned when neuron is called. Defaults to False
    :type output: bool, optional


    Inputs: \\input_, syn_0, mem_0
        - **input_** of shape `(batch, in_channels, H, W)`: tensor         containing input features
        - **syn_0** of shape `(batch, out_channels, H, W)`: tensor         containing the initial synaptic current (or cell state) for each         element in the batch.
        - **mem_0** of shape `(batch, out_channels, H, W)`: tensor         containing the initial membrane potential (or hidden state) for each         element in the batch.

    Outputs: spk, syn_1, mem_1
        - **spk** of shape `(batch, out_channels, H/pool, W/pool)`: tensor         containing the output spike (avg_pool and max_pool scale if greater         than 0.)
        - **syn_1** of shape `(batch, out_channels, H, W)`: tensor         containing the next synaptic current (or cell state) for each element         in the batch
        - **mem_1** of shape `(batch, out_channels, H, W)`: tensor         containing the next membrane potential (or hidden state) for each         element in the batch

    Learnable Parameters:
        - **SConv2dLSTM.conv.weight** (torch.Tensor) - the learnable         weights, of shape ((in_channels + out_channels), 4*out_channels,         kernel_size).

    """

    def __init__(self, in_channels, out_channels, kernel_size, bias=True, max_pool=0, avg_pool=0, threshold=1.0, spike_grad=None, surrogate_disable=False, init_hidden=False, inhibition=False, learn_threshold=False, reset_mechanism='none', state_quant=False, output=False):
        super().__init__(threshold, spike_grad, surrogate_disable, init_hidden, inhibition, learn_threshold, reset_mechanism, state_quant, output)
        self._init_mem()
        if self.reset_mechanism_val == 0:
            self.state_function = self._base_sub
        elif self.reset_mechanism_val == 1:
            self.state_function = self._base_zero
        elif self.reset_mechanism_val == 2:
            self.state_function = self._base_int
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.max_pool = max_pool
        self.avg_pool = avg_pool
        self.bias = bias
        self._sconv2dlstm_cases()
        if type(self.kernel_size) is int:
            self.padding = kernel_size // 2, kernel_size // 2
        else:
            self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.conv = nn.Conv2d(in_channels=self.in_channels + self.out_channels, out_channels=4 * self.out_channels, kernel_size=self.kernel_size, padding=self.padding, bias=self.bias)

    def _init_mem(self):
        syn = torch.zeros(0)
        mem = torch.zeros(0)
        self.register_buffer('syn', syn, False)
        self.register_buffer('mem', mem, False)

    def reset_mem(self):
        self.syn = torch.zeros_like(self.syn, device=self.syn.device)
        self.mem = torch.zeros_like(self.mem, device=self.mem.device)
        return self.syn, self.mem

    def init_sconv2dlstm(self):
        """Deprecated, use :class:`SConv2dLSTM.reset_mem` instead"""
        return self.reset_mem()

    def forward(self, input_, syn=None, mem=None):
        if not syn == None:
            self.syn = syn
        if not mem == None:
            self.mem = mem
        if self.init_hidden and (not mem == None or not syn == None):
            raise TypeError('`mem` or `syn` should not be passed as an argument while `init_hidden=True`')
        size = input_.size()
        correct_shape = size[0], self.out_channels, size[2], size[3]
        if not self.syn.shape == correct_shape:
            self.syn = torch.zeros(correct_shape, device=self.syn.device)
        if not self.mem.shape == correct_shape:
            self.mem = torch.zeros(correct_shape, device=self.mem.device)
        self.reset = self.mem_reset(self.mem)
        self.syn, self.mem = self.state_function(input_)
        if self.state_quant:
            self.syn = self.state_quant(self.syn)
            self.mem = self.state_quant(self.mem)
        if self.max_pool:
            self.spk = self.fire(F.max_pool2d(self.mem, self.max_pool))
        elif self.avg_pool:
            self.spk = self.fire(F.avg_pool2d(self.mem, self.avg_pool))
        else:
            self.spk = self.fire(self.mem)
        if self.output:
            return self.spk, self.syn, self.mem
        elif self.init_hidden:
            return self.spk
        else:
            return self.spk, self.syn, self.mem

    def _base_state_function(self, input_):
        combined = torch.cat([input_, self.mem], dim=1)
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.out_channels, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        base_fn_syn = f * self.syn + i * g
        base_fn_mem = o * torch.tanh(base_fn_syn)
        return base_fn_syn, base_fn_mem

    def _base_state_reset_zero(self, input_):
        combined = torch.cat([input_, self.mem], dim=1)
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.out_channels, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        base_fn_syn = f * self.syn + i * g
        base_fn_mem = o * torch.tanh(base_fn_syn)
        return 0, base_fn_mem

    def _base_sub(self, input_):
        syn, mem = self._base_state_function(input_)
        mem -= self.reset * self.threshold
        return syn, mem

    def _base_zero(self, input_):
        syn, mem = self._base_state_function(input_)
        syn2, mem2 = self._base_state_reset_zero(input_)
        syn2 *= self.reset
        mem2 *= self.reset
        syn -= syn2
        mem -= mem2
        return syn, mem

    def _base_int(self, input_):
        return self._base_state_function(input_)

    def _sconv2dlstm_cases(self):
        if self.max_pool and self.avg_pool:
            raise ValueError('Only one of either `max_pool` or `avg_pool` may be specified, not both.')

    @classmethod
    def detach_hidden(cls):
        """Returns the hidden states, detached from the current graph.
        Intended for use in truncated backpropagation through time where
        hidden state variables are instance variables."""
        for layer in range(len(cls.instances)):
            if isinstance(cls.instances[layer], SConv2dLSTM):
                cls.instances[layer].syn.detach_()
                cls.instances[layer].mem.detach_()

    @classmethod
    def reset_hidden(cls):
        """Used to clear hidden state variables to zero.
        Intended for use where hidden state variables are
        instance variables."""
        for layer in range(len(cls.instances)):
            if isinstance(cls.instances[layer], SConv2dLSTM):
                cls.instances[layer].syn = torch.zeros_like(cls.instances[layer].syn, device=cls.instances[layer].syn.device)
                cls.instances[layer].mem = torch.zeros_like(cls.instances[layer].mem, device=cls.instances[layer].mem.device)


class SLSTM(SpikingNeuron):
    """
    A spiking long short-term memory cell.
    Hidden states are membrane potential and synaptic current
    :math:`mem, syn`, which correspond to the hidden and cell
    states :math:`h, c` in the original LSTM formulation.

    The input is expected to be of size :math:`(N, X)` where
    :math:`N` is the batch size.

    Unlike the LSTM module in PyTorch, only one time step is
    simulated each time the cell is called.

    .. math::
            \\begin{array}{ll} \\\\
            i_t = \\sigma(W_{ii} x_t + b_{ii} + W_{hi} mem_{t-1} + b_{hi}) \\\\
            f_t = \\sigma(W_{if} x_t + b_{if} + W_{hf} mem_{t-1} + b_{hf}) \\\\
            g_t = \\tanh(W_{ig} x_t + b_{ig} + W_{hg} mem_{t-1} + b_{hg}) \\\\
            o_t = \\sigma(W_{io} x_t + b_{io} + W_{ho} mem_{t-1} + b_{ho}) \\\\
            syn_t = f_t ∗  syn_{t-1} + i_t ∗  g_t \\\\
            mem_t = o_t ∗  \\tanh(syn_t) \\\\
        \\end{array}

    where :math:`\\sigma` is the sigmoid function and ∗ is the
    Hadamard product.
    The output state :math:`mem_{t+1}` is thresholded to determine whether
    an output spike is generated.
    To conform to standard LSTM state behavior, the default reset mechanism
    is set to `reset="none"`, i.e., no reset is applied. If this is changed,
    the reset is only applied to :math:`h_t`.

    Example::

        import torch
        import torch.nn as nn
        import snntorch as snn

        beta = 0.5

        # Define Network
        class Net(nn.Module):
            def __init__(self):
                super().__init__()

                num_inputs = 784
                num_hidden1 = 1000
                num_hidden2 = 10

                spike_grad_lstm = surrogate.straight_through_estimator()

                # initialize layers
                self.slstm1 = snn.SLSTM(num_inputs, num_hidden1,
                spike_grad=spike_grad_lstm)
                self.slstm2 = snn.SLSTM(num_hidden1, num_hidden2,
                spike_grad=spike_grad_lstm)

            def forward(self, x):
                # Initialize hidden states and outputs at t=0
                syn1, mem1 = self.slstm1.init_slstm()
                syn2, mem2 = self.slstm2.init_slstm()

                # Record the final layer
                spk2_rec = []
                mem2_rec = []

                for step in range(num_steps):
                    spk1, syn1, mem1 = self.slstm1(x.flatten(1), syn1, mem1)
                    spk2, syn2, mem2 = self.slstm2(spk1, syn2, mem2)

                    spk2_rec.append(spk2)
                    mem2_rec.append(mem2)

                return torch.stack(spk2_rec), torch.stack(mem2_rec)

    :param input_size: number of expected features in the input :math:`x`
    :type input_size: int

    :param hidden_size: the number of features in the hidden state :math:`mem`
    :type hidden_size: int

    :param bias: If `True`, adds a learnable bias to the output.
        Defaults to `True`
    :type bias: bool, optional

    :param threshold: Threshold for :math:`h` to reach in order to generate
        a spike `S=1`. Defaults to 1
    :type threshold: float, optional

    :param spike_grad: Surrogate gradient for the term dS/dU. Defaults to
        ATan surrogate gradient
    :type spike_grad: surrogate gradient function from snntorch.surrogate,
        optional

    :param surrogate_disable: Disables surrogate gradients regardless of
        `spike_grad` argument. Useful for ONNX compatibility. Defaults
        to False
    :type surrogate_disable: bool, Optional

    :param learn_threshold: Option to enable learnable threshold. Defaults
        to False
    :type learn_threshold: bool, optional

    :param init_hidden: Instantiates state variables as instance variables.
        Defaults to False
    :type init_hidden: bool, optional

    :param inhibition: If `True`, suppresses all spiking other than the
        neuron with the highest state. Defaults to False
    :type inhibition: bool, optional

    :param reset_mechanism: Defines the reset mechanism applied to     :math:`mem` each time the threshold is met. Reset-by-subtraction:         "subtract", reset-to-zero: "zero, none: "none". Defaults to "none"
    :type reset_mechanism: str, optional

    :param state_quant: If specified, hidden states :math:`mem` and     :math:`syn` are quantized to a valid state for the forward pass.         Defaults to False
    :type state_quant: quantization function from snntorch.quant, optional

    :param output: If `True` as well as `init_hidden=True`, states are
        returned when neuron is called. Defaults to False
    :type output: bool, optional


    Inputs: \\input_, syn_0, mem_0
        - **input_** of shape `(batch, input_size)`: tensor containing input         features
        - **syn_0** of shape `(batch, hidden_size)`: tensor containing the         initial synaptic current (or cell state) for each element in the batch.
        - **mem_0** of shape `(batch, hidden_size)`: tensor containing the         initial membrane potential (or hidden state) for each element in the         batch.

    Outputs: spk, syn_1, mem_1
        - **spk** of shape `(batch, hidden_size)`: tensor containing the         output spike
        - **syn_1** of shape `(batch, hidden_size)`: tensor containing the         next synaptic current (or cell state) for each element in the batch
        - **mem_1** of shape `(batch, hidden_size)`: tensor containing the         next membrane potential (or hidden state) for each element in the batch

    Learnable Parameters:
        - **SLSTM.lstm_cell.weight_ih** (torch.Tensor) - the learnable         input-hidden weights, of shape (4*hidden_size, input_size)
        - **SLSTM.lstm_cell.weight_ih** (torch.Tensor) – the learnable         hidden-hidden weights, of shape (4*hidden_size, hidden_size)
        - **SLSTM.lstm_cell.bias_ih** – the learnable input-hidden bias, of         shape (4*hidden_size)
        - **SLSTM.lstm_cell.bias_hh** – the learnable hidden-hidden bias, of         shape (4*hidden_size)

    """

    def __init__(self, input_size, hidden_size, bias=True, threshold=1.0, spike_grad=None, surrogate_disable=False, init_hidden=False, inhibition=False, learn_threshold=False, reset_mechanism='none', state_quant=False, output=False):
        super().__init__(threshold, spike_grad, surrogate_disable, init_hidden, inhibition, learn_threshold, reset_mechanism, state_quant, output)
        self._init_mem()
        if self.reset_mechanism_val == 0:
            self.state_function = self._base_sub
        elif self.reset_mechanism_val == 1:
            self.state_function = self._base_zero
        elif self.reset_mechanism_val == 2:
            self.state_function = self._base_int
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.lstm_cell = nn.LSTMCell(self.input_size, self.hidden_size, bias=self.bias)

    def _init_mem(self):
        syn = torch.zeros(0)
        mem = torch.zeros(0)
        self.register_buffer('syn', syn, False)
        self.register_buffer('mem', mem, False)

    def reset_mem(self):
        self.syn = torch.zeros_like(self.syn, device=self.syn.device)
        self.mem = torch.zeros_like(self.mem, device=self.mem.device)
        return self.syn, self.mem

    def init_slstm(self):
        """Deprecated, use :class:`SLSTM.reset_mem` instead"""
        return self.reset_mem()

    def forward(self, input_, syn=None, mem=None):
        if not syn == None:
            self.syn = syn
        if not mem == None:
            self.mem = mem
        if self.init_hidden and (not mem == None or not syn == None):
            raise TypeError('`mem` or `syn` should not be passed as an argument while `init_hidden=True`')
        size = input_.size()
        correct_shape = size[0], self.hidden_size
        if not self.syn.shape == input_.shape:
            self.syn = torch.zeros(correct_shape, device=self.syn.device)
        if not self.mem.shape == input_.shape:
            self.mem = torch.zeros(correct_shape, device=self.mem.device)
        self.reset = self.mem_reset(self.mem)
        self.syn, self.mem = self.state_function(input_)
        if self.state_quant:
            self.syn = self.state_quant(self.syn)
            self.mem = self.state_quant(self.mem)
        self.spk = self.fire(self.mem)
        if self.output:
            return self.spk, self.syn, self.mem
        elif self.init_hidden:
            return self.spk
        else:
            return self.spk, self.syn, self.mem

    def _base_state_function(self, input_):
        base_fn_mem, base_fn_syn = self.lstm_cell(input_, (self.mem, self.syn))
        return base_fn_syn, base_fn_mem

    def _base_state_reset_zero(self, input_):
        base_fn_mem, _ = self.lstm_cell(input_, (self.mem, self.syn))
        return 0, base_fn_mem

    def _base_sub(self, input_):
        syn, mem = self._base_state_function(input_)
        mem -= self.reset * self.threshold
        return syn, mem

    def _base_zero(self, input_):
        syn, mem = self._base_state_function(input_)
        syn2, mem2 = self._base_state_reset_zero(input_)
        syn2 *= self.reset
        mem2 *= self.reset
        syn -= syn2
        mem -= mem2
        return syn, mem

    def _base_int(self, input_):
        return self._base_state_function(input_)

    @classmethod
    def detach_hidden(cls):
        """Returns the hidden states, detached from the current graph.
        Intended for use in truncated backpropagation through time where
        hidden state variables are instance variables."""
        for layer in range(len(cls.instances)):
            if isinstance(cls.instances[layer], SLSTM):
                cls.instances[layer].syn.detach_()
                cls.instances[layer].mem.detach_()

    @classmethod
    def reset_hidden(cls):
        """Used to clear hidden state variables to zero.
        Intended for use where hidden state variables are instance
        variables."""
        for layer in range(len(cls.instances)):
            if isinstance(cls.instances[layer], SLSTM):
                cls.instances[layer].syn = torch.zeros_like(cls.instances[layer].syn, device=cls.instances[layer].syn.device)
                cls.instances[layer].mem = torch.zeros_like(cls.instances[layer].mem, device=cls.instances[layer].mem.device)


class Synaptic(LIF):
    """
    2nd order leaky integrate and fire neuron model accounting for synaptic
    conductance.
    The synaptic current jumps upon spike arrival, which causes a jump in
    membrane potential.
    Synaptic current and membrane potential decay exponentially with rates
    of alpha and beta, respectively.
    For :math:`U[T] > U_{\\rm thr} ⇒ S[T+1] = 1`.

    If `reset_mechanism = "subtract"`, then :math:`U[t+1]` will have
    `threshold` subtracted from it whenever the neuron emits a spike:

    .. math::

            I_{\\rm syn}[t+1] = αI_{\\rm syn}[t] + I_{\\rm in}[t+1] \\\\
            U[t+1] = βU[t] + I_{\\rm syn}[t+1] - RU_{\\rm thr}

    If `reset_mechanism = "zero"`, then :math:`U[t+1]` will be set to `0`
    whenever the neuron emits a spike:

    .. math::

            I_{\\rm syn}[t+1] = αI_{\\rm syn}[t] + I_{\\rm in}[t+1] \\\\
            U[t+1] = βU[t] + I_{\\rm syn}[t+1] - R(βU[t] + I_{\\rm syn}[t+1])

    * :math:`I_{\\rm syn}` - Synaptic current
    * :math:`I_{\\rm in}` - Input current
    * :math:`U` - Membrane potential
    * :math:`U_{\\rm thr}` - Membrane threshold
    * :math:`R` - Reset mechanism: if active, :math:`R = 1`, otherwise         :math:`R = 0`
    * :math:`α` - Synaptic current decay rate
    * :math:`β` - Membrane potential decay rate

    Example::

        import torch
        import torch.nn as nn
        import snntorch as snn

        alpha = 0.9
        beta = 0.5

        # Define Network
        class Net(nn.Module):
            def __init__(self, num_inputs, num_hidden, num_outputs):
                super().__init__()

                # initialize layers
                self.fc1 = nn.Linear(num_inputs, num_hidden)
                self.lif1 = snn.Synaptic(alpha=alpha, beta=beta)
                self.fc2 = nn.Linear(num_hidden, num_outputs)
                self.lif2 = snn.Synaptic(alpha=alpha, beta=beta)

            def forward(self, x, syn1, mem1, spk1, syn2, mem2):
                cur1 = self.fc1(x)
                spk1, syn1, mem1 = self.lif1(cur1, syn1, mem1)
                cur2 = self.fc2(spk1)
                spk2, syn2, mem2 = self.lif2(cur2, syn2, mem2)
                return syn1, mem1, spk1, syn2, mem2, spk2



    :param alpha: synaptic current decay rate. Clipped between 0 and 1
        during the forward-pass. May be a single-valued tensor (i.e.,
        equal decay rate for all neurons in a layer), or multi-valued
        (one weight per neuron).
    :type alpha: float or torch.tensor

    :param beta: membrane potential decay rate. Clipped between 0 and 1
        during the forward-pass. May be a single-valued tensor (i.e., equal
        decay rate for all neurons in a layer), or multi-valued (one weight
        per neuron).
    :type beta: float or torch.tensor

    :param threshold: Threshold for :math:`mem` to reach in order to generate
        a spike `S=1`. Defaults to 1
    :type threshold: float, optional

    :param spike_grad: Surrogate gradient for the term dS/dU. Defaults to None
        (corresponds to Heaviside surrogate gradient. See `snntorch.surrogate`
        for more options)
    :type spike_grad: surrogate gradient function from snntorch.surrogate,
        optional

    :param surrogate_disable: Disables surrogate gradients regardless of
        `spike_grad` argument. Useful for ONNX compatibility. Defaults
        to False
    :type surrogate_disable: bool, Optional

    :param init_hidden: Instantiates state variables as instance variables.
        Defaults to False
    :type init_hidden: bool, optional

    :param inhibition: If `True`, suppresses all spiking other than the
        neuron with the highest state. Defaults to False
    :type inhibition: bool, optional

    :param learn_alpha: Option to enable learnable alpha. Defaults to False
    :type learn_alpha: bool, optional

    :param learn_beta: Option to enable learnable beta. Defaults to False
    :type learn_beta: bool, optional

    :param learn_threshold: Option to enable learnable threshold. Defaults
        to False
    :type learn_threshold: bool, optional

    :param reset_mechanism: Defines the reset mechanism applied to :math:`mem`
        each time the threshold is met. Reset-by-subtraction: "subtract",
        reset-to-zero: "zero", none: "none". Defaults to "subtract"
    :type reset_mechanism: str, optional

    :param state_quant: If specified, hidden states :math:`mem` and     :math:`syn` are quantized to a valid state for the forward pass.         Defaults to False
    :type state_quant: quantization function from snntorch.quant, optional

    :param output: If `True` as well as `init_hidden=True`, states are
        returned when neuron is called. Defaults to False
    :type output: bool, optional


    Inputs: \\input_, syn_0, mem_0
        - **input_** of shape `(batch, input_size)`: tensor containing         input features
        - **syn_0** of shape `(batch, input_size)`: tensor containing         input features
        - **mem_0** of shape `(batch, input_size)`: tensor containing         the initial membrane potential for each element in the batch.

    Outputs: spk, syn_1, mem_1
        - **spk** of shape `(batch, input_size)`: tensor containing the         output spikes.
        - **syn_1** of shape `(batch, input_size)`: tensor containing the         next synaptic current for each element in the batch
        - **mem_1** of shape `(batch, input_size)`: tensor containing the         next membrane potential for each element in the batch

    Learnable Parameters:
        - **Synaptic.alpha** (torch.Tensor) - optional learnable weights         must be manually passed in, of shape `1` or (input_size).
        - **Synaptic.beta** (torch.Tensor) - optional learnable weights must         be manually passed in, of shape `1` or (input_size).
        - **Synaptic.threshold** (torch.Tensor) - optional learnable         thresholds must be manually passed in, of shape `1` or`` (input_size).

    """

    def __init__(self, alpha, beta, threshold=1.0, spike_grad=None, surrogate_disable=False, init_hidden=False, inhibition=False, learn_alpha=False, learn_beta=False, learn_threshold=False, reset_mechanism='subtract', state_quant=False, output=False, reset_delay=True):
        super().__init__(beta, threshold, spike_grad, surrogate_disable, init_hidden, inhibition, learn_beta, learn_threshold, reset_mechanism, state_quant, output)
        self._alpha_register_buffer(alpha, learn_alpha)
        self._init_mem()
        if self.reset_mechanism_val == 0:
            self.state_function = self._base_sub
        elif self.reset_mechanism_val == 1:
            self.state_function = self._base_zero
        elif self.reset_mechanism_val == 2:
            self.state_function = self._base_int
        self.reset_delay = reset_delay

    def _init_mem(self):
        syn = torch.zeros(0)
        mem = torch.zeros(0)
        self.register_buffer('syn', syn, False)
        self.register_buffer('mem', mem, False)

    def reset_mem(self):
        self.syn = torch.zeros_like(self.syn, device=self.syn.device)
        self.mem = torch.zeros_like(self.mem, device=self.mem.device)
        return self.syn, self.mem

    def init_synaptic(self):
        """Deprecated, use :class:`Synaptic.reset_mem` instead"""
        return self.reset_mem()

    def forward(self, input_, syn=None, mem=None):
        if not syn == None:
            self.syn = mem
        if not mem == None:
            self.mem = mem
        if self.init_hidden and (not mem == None or not syn == None):
            raise TypeError('`mem` or `syn` should not be passed as an argument while `init_hidden=True`')
        if not self.syn.shape == input_.shape:
            self.syn = torch.zeros_like(input_, device=self.syn.device)
        if not self.mem.shape == input_.shape:
            self.mem = torch.zeros_like(input_, device=self.mem.device)
        self.reset = self.mem_reset(self.mem)
        self.syn, self.mem = self.state_function(input_)
        if self.state_quant:
            self.mem = self.state_quant(self.mem)
            self.syn = self.state_quant(self.syn)
        if self.inhibition:
            spk = self.fire_inhibition(self.mem.size(0), self.mem)
        else:
            spk = self.fire(self.mem)
        if not self.reset_delay:
            do_reset = spk / self.graded_spikes_factor - self.reset
            if self.reset_mechanism_val == 0:
                mem = mem - do_reset * self.threshold
            elif self.reset_mechanism_val == 1:
                mem = mem - do_reset * mem
        if self.output:
            return spk, self.syn, self.mem
        elif self.init_hidden:
            return spk
        else:
            return spk, self.syn, self.mem

    def _base_state_function(self, input_):
        base_fn_syn = self.alpha.clamp(0, 1) * self.syn + input_
        base_fn_mem = self.beta.clamp(0, 1) * self.mem + base_fn_syn
        return base_fn_syn, base_fn_mem

    def _base_state_reset_zero(self, input_):
        base_fn_syn = self.alpha.clamp(0, 1) * self.syn + input_
        base_fn_mem = self.beta.clamp(0, 1) * self.mem + base_fn_syn
        return 0, base_fn_mem

    def _base_sub(self, input_):
        syn, mem = self._base_state_function(input_)
        mem = mem - self.reset * self.threshold
        return syn, mem

    def _base_zero(self, input_):
        syn, mem = self._base_state_function(input_)
        syn2, mem2 = self._base_state_reset_zero(input_)
        syn -= syn2 * self.reset
        mem -= mem2 * self.reset
        return syn, mem

    def _base_int(self, input_):
        return self._base_state_function(input_)

    def _alpha_register_buffer(self, alpha, learn_alpha):
        if not isinstance(alpha, torch.Tensor):
            alpha = torch.as_tensor(alpha)
        if learn_alpha:
            self.alpha = nn.Parameter(alpha)
        else:
            self.register_buffer('alpha', alpha)

    @classmethod
    def detach_hidden(cls):
        """Returns the hidden states, detached from the current graph.
        Intended for use in truncated backpropagation through time where
        hidden state variables are instance variables."""
        for layer in range(len(cls.instances)):
            if isinstance(cls.instances[layer], Synaptic):
                cls.instances[layer].syn.detach_()
                cls.instances[layer].mem.detach_()

    @classmethod
    def reset_hidden(cls):
        """Used to clear hidden state variables to zero.
        Intended for use where hidden state variables are
        instance variables."""
        for layer in range(len(cls.instances)):
            if isinstance(cls.instances[layer], Synaptic):
                cls.instances[layer].syn = torch.zeros_like(cls.instances[layer].syn, device=cls.instances[layer].syn.device)
                cls.instances[layer].mem = torch.zeros_like(cls.instances[layer].mem, device=cls.instances[layer].mem.device)


class SpikeTime(nn.Module):
    """Used by ce_temporal_loss and mse_temporal_loss to convert spike
    outputs into spike times."""

    def __init__(self, target_is_time=False, on_target=0, off_target=-1, tolerance=0, multi_spike=False):
        super().__init__()
        self.target_is_time = target_is_time
        self.tolerance = tolerance
        self.tolerance_fn = self.Tolerance.apply
        self.multi_spike = multi_spike
        if not self.target_is_time:
            self.on_target = on_target
            self.off_target = off_target
        if self.multi_spike:
            self.first_spike_fn = self.MultiSpike.apply
        else:
            self.first_spike_fn = self.FirstSpike.apply

    def forward(self, spk_out, targets):
        self.device, num_steps, num_outputs = self._prediction_check(spk_out)
        if not self.target_is_time:
            targets = self.labels_to_spike_times(targets, num_outputs)
        targets[targets < 0] = spk_out.size(0) + targets[targets < 0]
        if self.multi_spike:
            self.spike_count = targets.size(0)
            spk_time_final = self.first_spike_fn(spk_out, self.spike_count, self.device)
        else:
            spk_time_final = self.first_spike_fn(spk_out, self.device)
        if self.tolerance:
            spk_time_final = self.tolerance_fn(spk_time_final, targets, self.tolerance)
        return spk_time_final, targets

    def _prediction_check(self, spk_out):
        device = spk_out.device
        num_steps = spk_out.size(0)
        num_outputs = spk_out.size(-1)
        return device, num_steps, num_outputs


    @staticmethod
    class FirstSpike(torch.autograd.Function):
        """Convert spk_rec of 1/0s [TxBxN] --> first spike time [BxN].
        Linearize df/dS=-1 if spike, 0 if no spike."""

        @staticmethod
        def forward(ctx, spk_rec, device='cpu'):
            """Convert spk_rec of 1/0s [TxBxN] --> spk_time [TxBxN].
            0's indicate no spike --> +1 is first time step.
            Transpose accounts for broadcasting along final dimension
            (i.e., multiply along T)."""
            spk_time = (spk_rec.transpose(0, -1) * (torch.arange(0, spk_rec.size(0)).detach() + 1)).transpose(0, -1)
            """extact first spike time. Will be used to pass into loss
            function."""
            first_spike_time = torch.zeros_like(spk_time[0])
            for step in range(spk_time.size(0)):
                first_spike_time += spk_time[step] * ~first_spike_time.bool()
            """override element 0 (no spike) with shadow spike @ final time
            step, then offset by -1
            s.t. first_spike is at t=0."""
            first_spike_time += ~first_spike_time.bool() * spk_time.size(0)
            first_spike_time -= 1
            ctx.save_for_backward(first_spike_time, spk_rec)
            return first_spike_time

        @staticmethod
        def backward(ctx, grad_output):
            first_spike_time, spk_rec = ctx.saved_tensors
            spk_time_grad = torch.zeros_like(spk_rec)
            """spike extraction step/indexing @ each step is
            non-differentiable.
            Apply sign estimator by substituting gradient for -1 ONLY at
            first spike time."""
            for i in range(first_spike_time.size(0)):
                for j in range(first_spike_time.size(1)):
                    spk_time_grad[first_spike_time[i, j].long(), i, j] = 1.0
            grad = -grad_output * spk_time_grad
            return grad, None


    @staticmethod
    class MultiSpike(torch.autograd.Function):
        """Convert spk_rec of 1/0s [TxBxN] --> first F spike times [FxBxN].
        Linearize df/dS=-1 if spike, 0 if no spike."""

        @staticmethod
        def forward(ctx, spk_rec, spk_count, device='cpu'):
            spk_rec_tmp = spk_rec.clone()
            spk_time_rec = []
            for step in range(spk_count):
                """Convert spk_rec of 1/0s [TxBxN] --> spk_time [TxBxN].
                0's indicate no spike --> +1 is first time step.
                Transpose accounts for broadcasting along final dimension
                (i.e., multiply along T)."""
                spk_time = (spk_rec_tmp.transpose(0, -1) * (torch.arange(0, spk_rec_tmp.size(0)).detach() + 1)).transpose(0, -1)
                """extact n-th spike time (n=step) up to F."""
                nth_spike_time = torch.zeros_like(spk_time[0])
                for step in range(spk_time.size(0)):
                    nth_spike_time += spk_time[step] * ~nth_spike_time.bool()
                """override element 0 (no spike) with shadow spike @ final
                time step, then offset by -1
                s.t. first_spike is at t=0."""
                nth_spike_time += ~nth_spike_time.bool() * spk_time.size(0)
                nth_spike_time -= 1
                spk_time_rec.append(nth_spike_time)
                """before looping, eliminate n-th spike. this avoids double
                counting spikes."""
                spk_rec_tmp[nth_spike_time.long()] = 0
            """Pass this into loss function."""
            spk_time_rec = torch.stack(spk_time_rec)
            ctx.save_for_backward(spk_time_rec, spk_rec)
            return spk_time_rec

        @staticmethod
        def backward(ctx, grad_output):
            spk_time_final, spk_rec = ctx.saved_tensors
            spk_time_grad = torch.zeros_like(spk_rec)
            """spike extraction step/indexing @ each step is
            non-differentiable.
            Apply sign estimator by substituting gradient for -1 ONLY at
            F-th spike time."""
            for i in range(spk_time_final.size(0)):
                for j in range(spk_time_final.size(1)):
                    for k in range(spk_time_final.size(2)):
                        spk_time_grad[spk_time_final[i, j, k].long(), j, k] = -grad_output[i, j, k]
            grad = spk_time_grad
            return grad, None, None


    @staticmethod
    class Tolerance(torch.autograd.Function):
        """If spike time is 'close enough' to target spike within tolerance,
        set the time to target for loss calc only."""

        @staticmethod
        def forward(ctx, spk_time, target, tolerance):
            spk_time_clone = spk_time.clone()
            spk_time_clone[torch.abs(spk_time - target) < tolerance] = (torch.ones_like(spk_time) * target)[torch.abs(spk_time - target) < tolerance]
            return spk_time_clone

        @staticmethod
        def backward(ctx, grad_output):
            grad = grad_output
            return grad, None, None

    def labels_to_spike_times(self, targets, num_outputs):
        """Convert index labels [B] into spike times."""
        if not self.multi_spike:
            targets = self.label_to_single_spike(targets, num_outputs)
        else:
            targets = self.label_to_multi_spike(targets, num_outputs)
        return targets

    def label_to_single_spike(self, targets, num_outputs):
        """Convert labels from neuron index (dim: B) to first spike time
        (dim: B x N)."""
        targets = spikegen.targets_convert(targets, num_classes=num_outputs, on_target=self.on_target, off_target=self.off_target)
        return targets

    def label_to_multi_spike(self, targets, num_outputs):
        """Convert labels from neuron index (dim: B) to multiple spike times
        (dim: F x B x N).
        F is the number of spikes per neuron. Assumes target is iterable
        along F."""
        num_spikes_on = len(self.on_target)
        num_spikes_off = len(self.off_target)
        if num_spikes_on != num_spikes_off:
            raise IndexError(f'`on_target` (length: {num_spikes_on}) must have the same length as `off_target` (length: {num_spikes_off}.')
        targets_rec = []
        for step in range(num_spikes_on):
            target_step = spikegen.targets_convert(targets, num_classes=num_outputs, on_target=self.on_target[step], off_target=self.off_target[step])
            targets_rec.append(target_step)
        targets_rec = torch.stack(targets_rec)
        return targets_rec


def stdp_conv1d_single_step(conv: 'nn.Conv1d', in_spike: 'torch.Tensor', out_spike: 'torch.Tensor', trace_pre: 'Union[torch.Tensor, None]', trace_post: 'Union[torch.Tensor, None]', tau_pre: 'float', tau_post: 'float', f_pre: 'Callable'=lambda x: x, f_post: 'Callable'=lambda x: x):
    if conv.dilation != (1,):
        raise NotImplementedError('STDP with dilation != 1 for Conv1d has not been implemented!')
    if conv.groups != 1:
        raise NotImplementedError('STDP with groups != 1 for Conv1d has not been implemented!')
    stride_l = conv.stride[0]
    if conv.padding == (0,):
        pass
    else:
        pL = conv.padding[0]
        if conv.padding_mode != 'zeros':
            in_spike = F.pad(in_spike, conv._reversed_padding_repeated_twice, mode=conv.padding_mode)
        else:
            in_spike = F.pad(in_spike, pad=(pL, pL))
    if trace_pre is None:
        trace_pre = torch.zeros_like(in_spike, device=in_spike.device, dtype=in_spike.dtype)
    if trace_post is None:
        trace_post = torch.zeros_like(out_spike, device=in_spike.device, dtype=in_spike.dtype)
    trace_pre = trace_pre - trace_pre / tau_pre + in_spike
    trace_post = trace_post - trace_post / tau_post + out_spike
    delta_w = torch.zeros_like(conv.weight.data)
    for l in range(conv.weight.shape[2]):
        l_end = in_spike.shape[2] - conv.weight.shape[2] + 1 + l
        pre_spike = in_spike[:, :, l:l_end:stride_l]
        post_spike = out_spike
        weight = conv.weight.data[:, :, l]
        tr_pre = trace_pre[:, :, l:l_end:stride_l]
        tr_post = trace_post
        delta_w_pre = -(f_pre(weight) * (tr_post.unsqueeze(2) * pre_spike.unsqueeze(1)).permute([1, 2, 0, 3]).sum(dim=[2, 3]))
        delta_w_post = f_post(weight) * (tr_pre.unsqueeze(1) * post_spike.unsqueeze(2)).permute([1, 2, 0, 3]).sum(dim=[2, 3])
        delta_w[:, :, l] += delta_w_pre + delta_w_post
    return trace_pre, trace_post, delta_w


def stdp_conv2d_single_step(conv: 'nn.Conv2d', in_spike: 'torch.Tensor', out_spike: 'torch.Tensor', trace_pre: 'Union[torch.Tensor, None]', trace_post: 'Union[torch.Tensor, None]', tau_pre: 'float', tau_post: 'float', f_pre: 'Callable'=lambda x: x, f_post: 'Callable'=lambda x: x):
    if conv.dilation != (1, 1):
        raise NotImplementedError('STDP with dilation != 1 for Conv2d has not been implemented!')
    if conv.groups != 1:
        raise NotImplementedError('STDP with groups != 1 for Conv2d has not been implemented!')
    stride_h = conv.stride[0]
    stride_w = conv.stride[1]
    if conv.padding == (0, 0):
        pass
    else:
        pH = conv.padding[0]
        pW = conv.padding[1]
        if conv.padding_mode != 'zeros':
            in_spike = F.pad(in_spike, conv._reversed_padding_repeated_twice, mode=conv.padding_mode)
        else:
            in_spike = F.pad(in_spike, pad=(pW, pW, pH, pH))
    if trace_pre is None:
        trace_pre = torch.zeros_like(in_spike, device=in_spike.device, dtype=in_spike.dtype)
    if trace_post is None:
        trace_post = torch.zeros_like(out_spike, device=in_spike.device, dtype=in_spike.dtype)
    trace_pre = trace_pre - trace_pre / tau_pre + in_spike
    trace_post = trace_post - trace_post / tau_post + out_spike
    delta_w = torch.zeros_like(conv.weight.data)
    for h in range(conv.weight.shape[2]):
        for w in range(conv.weight.shape[3]):
            h_end = in_spike.shape[2] - conv.weight.shape[2] + 1 + h
            w_end = in_spike.shape[3] - conv.weight.shape[3] + 1 + w
            pre_spike = in_spike[:, :, h:h_end:stride_h, w:w_end:stride_w]
            post_spike = out_spike
            weight = conv.weight.data[:, :, h, w]
            tr_pre = trace_pre[:, :, h:h_end:stride_h, w:w_end:stride_w]
            tr_post = trace_post
            delta_w_pre = -(f_pre(weight) * (tr_post.unsqueeze(2) * pre_spike.unsqueeze(1)).permute([1, 2, 0, 3, 4]).sum(dim=[2, 3, 4]))
            delta_w_post = f_post(weight) * (tr_pre.unsqueeze(1) * post_spike.unsqueeze(2)).permute([1, 2, 0, 3, 4]).sum(dim=[2, 3, 4])
            delta_w[:, :, h, w] += delta_w_pre + delta_w_post
    return trace_pre, trace_post, delta_w


def stdp_linear_single_step(fc: 'nn.Linear', in_spike: 'torch.Tensor', out_spike: 'torch.Tensor', trace_pre: 'Union[float, torch.Tensor, None]', trace_post: 'Union[float, torch.Tensor, None]', tau_pre: 'float', tau_post: 'float', f_pre: 'Callable'=lambda x: x, f_post: 'Callable'=lambda x: x):
    if trace_pre is None:
        trace_pre = 0.0
    if trace_post is None:
        trace_post = 0.0
    weight = fc.weight.data
    trace_pre = trace_pre - trace_pre / tau_pre + in_spike
    trace_post = trace_post - trace_post / tau_post + out_spike
    delta_w_pre = -f_pre(weight) * (trace_post.unsqueeze(2) * in_spike.unsqueeze(1)).sum(0)
    delta_w_post = f_post(weight) * (trace_pre.unsqueeze(1) * out_spike.unsqueeze(2)).sum(0)
    return trace_pre, trace_post, delta_w_pre + delta_w_post


class STDPLearner(nn.Module):

    def __init__(self, synapse: 'Union[nn.Conv2d, nn.Linear]', sn, tau_pre: 'float', tau_post: 'float', f_pre: 'Callable'=lambda x: x, f_post: 'Callable'=lambda x: x):
        super().__init__()
        self.tau_pre = tau_pre
        self.tau_post = tau_post
        self.f_pre = f_pre
        self.f_post = f_post
        self.synapse = synapse
        self.in_spike_monitor = probe.InputMonitor(synapse)
        self.out_spike_monitor = probe.OutputMonitor(sn)
        self.trace_pre = None
        self.trace_post = None

    def reset(self):
        super(STDPLearner, self).reset()
        self.in_spike_monitor.clear_recorded_data()
        self.out_spike_monitor.clear_recorded_data()

    def disable(self):
        self.in_spike_monitor.disable()
        self.out_spike_monitor.disable()

    def enable(self):
        self.in_spike_monitor.enable()
        self.out_spike_monitor.enable()

    def step(self, on_grad: 'bool'=True, scale: 'float'=1.0):
        length = self.in_spike_monitor.records.__len__()
        delta_w = None
        if isinstance(self.synapse, nn.Linear):
            stdp_f = stdp_linear_single_step
        elif isinstance(self.synapse, nn.Conv2d):
            stdp_f = stdp_conv2d_single_step
        elif isinstance(self.synapse, nn.Conv1d):
            stdp_f = stdp_conv1d_single_step
        else:
            raise NotImplementedError(self.synapse)
        for _ in range(length):
            in_spike = self.in_spike_monitor.records.pop(0)
            out_spike = self.out_spike_monitor.records.pop(0)
            self.trace_pre, self.trace_post, dw = stdp_f(self.synapse, in_spike, out_spike, self.trace_pre, self.trace_post, self.tau_pre, self.tau_post, self.f_pre, self.f_post)
            if scale != 1.0:
                dw *= scale
            delta_w = dw if delta_w is None else delta_w + dw
        if on_grad:
            if self.synapse.weight.grad is None:
                self.synapse.weight.grad = -delta_w
            else:
                self.synapse.weight.grad = self.synapse.weight.grad - delta_w
        else:
            return delta_w


class NetWithAvgPool(torch.nn.Module):

    def __init__(self):
        super(NetWithAvgPool, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.lif1 = snn.Leaky(beta=0.9, init_hidden=True)
        self.fc1 = torch.nn.Linear(28 * 28 * 16 // 4, 500)
        self.lif2 = snn.Leaky(beta=0.9, init_hidden=True, output=True)

    def forward(self, x):
        x = torch.nn.functional.avg_pool2d(self.conv1(x), kernel_size=2, stride=2)
        x = x.view(-1, 28 * 28 * 16 // 4)
        x = self.lif1(x)
        x = self.fc1(x)
        x = self.lif2(x)
        return x


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (GradedSpikes,
     lambda: ([], {'size': 4, 'constant_factor': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (LeakyKernel,
     lambda: ([], {'input_size': 4, 'hidden_size': 4}),
     lambda: ([torch.rand([4, 4])], {})),
    (RecurrentOneToOne,
     lambda: ([], {'V': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
]

