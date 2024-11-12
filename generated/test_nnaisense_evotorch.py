
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


from time import sleep


from typing import Optional


from typing import Union


import numpy as np


from typing import Tuple


import math


from collections.abc import Mapping


from copy import copy


from copy import deepcopy


from typing import Type


from typing import NamedTuple


from typing import Iterable


from typing import Callable


import itertools


from typing import Any


from typing import List


import logging


import random


from collections.abc import Sequence


from numbers import Number


from torch.func import vmap


from torch import nn


import torch.nn.functional as nnf


from collections import namedtuple


from typing import Sequence


from torch.nn import utils as nnu


from warnings import warn


from torch.utils.data import DataLoader


from torch.utils.data import Dataset


from numbers import Real


from collections import OrderedDict


from collections.abc import Iterable


from collections.abc import Set


import functools


import inspect


from numbers import Integral


from typing import Dict


from typing import Mapping


from itertools import product


from torch.utils.data import TensorDataset


from torch.func import grad


from torch import FloatTensor


class Clip(nn.Module):
    """A small torch module for clipping the values of tensors"""

    def __init__(self, lb: 'float', ub: 'float'):
        """`__init__(...)`: Initialize the Clip operator.

        Args:
            lb: Lower bound. Values less than this will be clipped.
            ub: Upper bound. Values greater than this will be clipped.
        """
        nn.Module.__init__(self)
        self._lb = float(lb)
        self._ub = float(ub)

    def forward(self, x: 'torch.Tensor'):
        return x.clamp(self._lb, self._ub)

    def extra_repr(self):
        return 'lb={}, ub={}'.format(self._lb, self._ub)


class Bin(nn.Module):
    """A small torch module for binning the values of tensors.

    In more details, considering a lower bound value lb,
    an upper bound value ub, and an input tensor x,
    each value within x closer to lb will be converted to lb
    and each value within x closer to ub will be converted to ub.
    """

    def __init__(self, lb: 'float', ub: 'float'):
        """`__init__(...)`: Initialize the Clip operator.

        Args:
            lb: Lower bound
            ub: Upper bound
        """
        nn.Module.__init__(self)
        self._lb = float(lb)
        self._ub = float(ub)
        self._interval_size = self._ub - self._lb
        self._shrink_amount = self._interval_size / 2.0
        self._shift_amount = (self._ub + self._lb) / 2.0

    def forward(self, x: 'torch.Tensor'):
        x = x - self._shift_amount
        x = x / self._shrink_amount
        x = torch.sign(x)
        x = x * self._shrink_amount
        x = x + self._shift_amount
        return x

    def extra_repr(self):
        return 'lb={}, ub={}'.format(self._lb, self._ub)


class Slice(nn.Module):
    """A small torch module for getting the slice of an input tensor"""

    def __init__(self, from_index: 'int', to_index: 'int'):
        """`__init__(...)`: Initialize the Slice operator.

        Args:
            from_index: The index from which the slice begins.
            to_index: The exclusive index at which the slice ends.
        """
        nn.Module.__init__(self)
        self._from_index = from_index
        self._to_index = to_index

    def forward(self, x):
        return x[self._from_index:self._to_index]

    def extra_repr(self):
        return 'from_index={}, to_index={}'.format(self._from_index, self._to_index)


class Round(nn.Module):
    """A small torch module for rounding the values of an input tensor"""

    def __init__(self, ndigits: 'int'=0):
        nn.Module.__init__(self)
        self._ndigits = int(ndigits)
        self._q = 10.0 ** self._ndigits

    def forward(self, x):
        x = x * self._q
        x = torch.round(x)
        x = x / self._q
        return x

    def extra_repr(self):
        return 'ndigits=' + str(self._ndigits)


class Apply(nn.Module):
    """A torch module for applying an arithmetic operator on an input tensor"""

    def __init__(self, operator: 'str', argument: 'float'):
        """`__init__(...)`: Initialize the Apply module.

        Args:
            operator: Must be '+', '-', '*', '/', or '**'.
                Indicates which operation will be done
                on the input tensor.
            argument: Expected as a float, represents
                the right-argument of the operation
                (the left-argument being the input
                tensor).
        """
        nn.Module.__init__(self)
        self._operator = str(operator)
        assert self._operator in ('+', '-', '*', '/', '**')
        self._argument = float(argument)

    def forward(self, x):
        op = self._operator
        arg = self._argument
        if op == '+':
            return x + arg
        elif op == '-':
            return x - arg
        elif op == '*':
            return x * arg
        elif op == '/':
            return x / arg
        elif op == '**':
            return x ** arg
        else:
            raise ValueError('Unknown operator:' + repr(op))

    def extra_repr(self):
        return 'operator={}, argument={}'.format(repr(self._operator), self._argument)


class RNN(nn.Module):

    def __init__(self, input_size: 'int', hidden_size: 'int', nonlinearity: 'str'='tanh', *, dtype: torch.dtype=torch.float32, device: Union[str, torch.device]='cpu'):
        super().__init__()
        input_size = int(input_size)
        hidden_size = int(hidden_size)
        nonlinearity = str(nonlinearity)
        self.W1 = nn.Parameter(torch.randn(hidden_size, input_size, dtype=dtype, device=device))
        self.W2 = nn.Parameter(torch.randn(hidden_size, hidden_size, dtype=dtype, device=device))
        self.b1 = nn.Parameter(torch.zeros(hidden_size, dtype=dtype, device=device))
        self.b2 = nn.Parameter(torch.zeros(hidden_size, dtype=dtype, device=device))
        if nonlinearity == 'tanh':
            self.actfunc = torch.tanh
        else:
            self.actfunc = getattr(nnf, nonlinearity)
        self.nonlinearity = nonlinearity
        self.input_size = input_size
        self.hidden_size = hidden_size

    def forward(self, x: 'torch.Tensor', h: 'Optional[torch.Tensor]'=None) ->tuple:
        if h is None:
            h = torch.zeros(self.hidden_size, dtype=x.dtype, device=x.device)
        act = self.actfunc
        W1 = self.W1
        W2 = self.W2
        b1 = self.b1.unsqueeze(-1)
        b2 = self.b2.unsqueeze(-1)
        x = x.unsqueeze(-1)
        h = h.unsqueeze(-1)
        y = act(W1 @ x + b1 + (W2 @ h + b2))
        y = y.squeeze(-1)
        return y, y

    def __repr__(self) ->str:
        clsname = type(self).__name__
        return f'{clsname}(input_size={self.input_size}, hidden_size={self.hidden_size}, nonlinearity={repr(self.nonlinearity)})'


class LSTM(nn.Module):

    def __init__(self, input_size: 'int', hidden_size: 'int', *, dtype: torch.dtype=torch.float32, device: Union[str, torch.device]='cpu'):
        super().__init__()
        input_size = int(input_size)
        hidden_size = int(hidden_size)
        self.input_size = input_size
        self.hidden_size = hidden_size

        def input_weight():
            return nn.Parameter(torch.randn(self.hidden_size, self.input_size, dtype=dtype, device=device))

        def weight():
            return nn.Parameter(torch.randn(self.hidden_size, self.hidden_size, dtype=dtype, device=device))

        def bias():
            return nn.Parameter(torch.zeros(self.hidden_size, dtype=dtype, device=device))
        self.W_ii = input_weight()
        self.W_if = input_weight()
        self.W_ig = input_weight()
        self.W_io = input_weight()
        self.W_hi = weight()
        self.W_hf = weight()
        self.W_hg = weight()
        self.W_ho = weight()
        self.b_ii = bias()
        self.b_if = bias()
        self.b_ig = bias()
        self.b_io = bias()
        self.b_hi = bias()
        self.b_hf = bias()
        self.b_hg = bias()
        self.b_ho = bias()

    def forward(self, x: 'torch.Tensor', hidden=None) ->tuple:
        sigm = torch.sigmoid
        tanh = torch.tanh
        if hidden is None:
            h_prev = torch.zeros(self.hidden_size, dtype=x.dtype, device=x.device)
            c_prev = torch.zeros(self.hidden_size, dtype=x.dtype, device=x.device)
        else:
            h_prev, c_prev = hidden
        i_t = sigm(self.W_ii @ x + self.b_ii + self.W_hi @ h_prev + self.b_hi)
        f_t = sigm(self.W_if @ x + self.b_if + self.W_hf @ h_prev + self.b_hf)
        g_t = tanh(self.W_ig @ x + self.b_ig + self.W_hg @ h_prev + self.b_hg)
        o_t = sigm(self.W_io @ x + self.b_io + self.W_ho @ h_prev + self.b_ho)
        c_t = f_t * c_prev + i_t * g_t
        h_t = o_t * tanh(c_t)
        return h_t, (h_t, c_t)

    def __repr__(self) ->str:
        clsname = type(self).__name__
        return f'{clsname}(input_size={self.input_size}, hidden_size={self.hidden_size})'


class FeedForwardNet(nn.Module):
    """
    Representation of a feed forward neural network as a torch Module.

    An example initialization of a FeedForwardNet is as follows:

        net = drt.FeedForwardNet(4, [(8, 'tanh'), (6, 'tanh')])

    which means that we would like to have a network which expects an input
    vector of length 4 and passes its input through 2 tanh-activated hidden
    layers (with neurons count 8 and 6, respectively).
    The output of the last hidden layer (of length 6) is the final
    output vector.

    The string representation of the module obtained via the example above
    is:

        FeedForwardNet(
          (layer_0): Linear(in_features=4, out_features=8, bias=True)
          (actfunc_0): Tanh()
          (layer_1): Linear(in_features=8, out_features=6, bias=True)
          (actfunc_1): Tanh()
        )
    """
    LengthActTuple = Tuple[int, Union[str, Callable]]
    LengthActBiasTuple = Tuple[int, Union[str, Callable], Union[bool]]

    def __init__(self, input_size: 'int', layers: 'List[Union[LengthActTuple, LengthActBiasTuple]]'):
        """`__init__(...)`: Initialize the FeedForward network.

        Args:
            input_size: Input size of the network, expected as an int.
            layers: Expected as a list of tuples,
                where each tuple is either of the form
                `(layer_size, activation_function)`
                or of the form
                `(layer_size, activation_function, bias)`
                in which
                (i) `layer_size` is an int, specifying the number of neurons;
                (ii) `activation_function` is None, or a callable object,
                or a string containing the name of the activation function
                ('relu', 'selu', 'elu', 'tanh', 'hardtanh', or 'sigmoid');
                (iii) `bias` is a boolean, specifying whether the layer
                is to have a bias or not.
                When omitted, bias is set to True.
        """
        nn.Module.__init__(self)
        for i, layer in enumerate(layers):
            if len(layer) == 2:
                size, actfunc = layer
                bias = True
            elif len(layer) == 3:
                size, actfunc, bias = layer
            else:
                assert False, 'A layer tuple of invalid size is encountered'
            setattr(self, 'layer_' + str(i), nn.Linear(input_size, size, bias=bias))
            if isinstance(actfunc, str):
                if actfunc == 'relu':
                    actfunc = nn.ReLU()
                elif actfunc == 'selu':
                    actfunc = nn.SELU()
                elif actfunc == 'elu':
                    actfunc = nn.ELU()
                elif actfunc == 'tanh':
                    actfunc = nn.Tanh()
                elif actfunc == 'hardtanh':
                    actfunc = nn.Hardtanh()
                elif actfunc == 'sigmoid':
                    actfunc = nn.Sigmoid()
                elif actfunc == 'round':
                    actfunc = Round()
                else:
                    raise ValueError('Unknown activation function: ' + repr(actfunc))
            setattr(self, 'actfunc_' + str(i), actfunc)
            input_size = size

    def forward(self, x):
        i = 0
        while hasattr(self, 'layer_' + str(i)):
            x = getattr(self, 'layer_' + str(i))(x)
            f = getattr(self, 'actfunc_' + str(i))
            if f is not None:
                x = f(x)
            i += 1
        return x


class StructuredControlNet(nn.Module):
    """Structured Control Net.

    This is a control network consisting of two components:
    (i) a non-linear component, which is a feed-forward network; and
    (ii) a linear component, which is a linear layer.
    Both components take the input vector provided to the
    structured control network.
    The final output is the sum of the outputs of both components.

    Reference:
        Mario Srouji, Jian Zhang, Ruslan Salakhutdinov (2018).
        Structured Control Nets for Deep Reinforcement Learning.
    """

    def __init__(self, *, in_features: int, out_features: int, num_layers: int, hidden_size: int, bias: bool=True, nonlinearity: Union[str, Callable]='tanh'):
        """`__init__(...)`: Initialize the structured control net.

        Args:
            in_features: Length of the input vector
            out_features: Length of the output vector
            num_layers: Number of hidden layers for the non-linear component
            hidden_size: Number of neurons in a hidden layer of the
                non-linear component
            bias: Whether or not the linear component is to have bias
            nonlinearity: Activation function
        """
        nn.Module.__init__(self)
        self._in_features = in_features
        self._out_features = out_features
        self._num_layers = num_layers
        self._hidden_size = hidden_size
        self._bias = bias
        self._nonlinearity = nonlinearity
        self._linear_component = nn.Linear(in_features=self._in_features, out_features=self._out_features, bias=self._bias)
        self._nonlinear_component = FeedForwardNet(input_size=self._in_features, layers=list((self._hidden_size, self._nonlinearity) for _ in range(self._num_layers)) + [(self._out_features, self._nonlinearity)])

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        """TODO: documentation"""
        return self._linear_component(x) + self._nonlinear_component(x)

    @property
    def in_features(self):
        """TODO: documentation"""
        return self._in_features

    @property
    def out_features(self):
        """TODO: documentation"""
        return self._out_features

    @property
    def num_layers(self):
        """TODO: documentation"""
        return self._num_layers

    @property
    def hidden_size(self):
        """TODO: documentation"""
        return self._hidden_size

    @property
    def bias(self):
        """TODO: documentation"""
        return self._bias

    @property
    def nonlinearity(self):
        """TODO: documentation"""
        return self._nonlinearity


class LocomotorNet(nn.Module):
    """LocomotorNet: A locomotion-specific structured control net.

    This is a control network which consists of two components:
    one linear, and one non-linear. The non-linear component
    is an input-independent set of sinusoidals waves whose
    amplitudes, frequencies and phases are trainable.
    Upon execution of a forward pass, the output of the non-linear
    component is the sum of all these sinusoidal waves.
    The linear component is a linear layer (optionally with bias)
    whose weights (and biases) are trainable.
    The final output of the LocomotorNet at the end of a forward pass
    is the sum of the linear and the non-linear components.

    Note that this is a stateful network, where the only state
    is the timestep t, which starts from 0 and gets incremented by 1
    at the end of each forward pass. The `reset()` method resets
    t back to 0.

    Reference:
        Mario Srouji, Jian Zhang, Ruslan Salakhutdinov (2018).
        Structured Control Nets for Deep Reinforcement Learning.
    """

    def __init__(self, *, in_features: int, out_features: int, bias: bool=True, num_sinusoids=16):
        """`__init__(...)`: Initialize the LocomotorNet.

        Args:
            in_features: Length of the input vector
            out_features: Length of the output vector
            bias: Whether or not the linear component is to have a bias
            num_sinusoids: Number of sinusoidal waves
        """
        nn.Module.__init__(self)
        self._in_features = in_features
        self._out_features = out_features
        self._bias = bias
        self._num_sinusoids = num_sinusoids
        self._linear_component = nn.Linear(in_features=self._in_features, out_features=self._out_features, bias=self._bias)
        self._amplitudes = nn.ParameterList()
        self._frequencies = nn.ParameterList()
        self._phases = nn.ParameterList()
        for _ in range(self._num_sinusoids):
            for paramlist in (self._amplitudes, self._frequencies, self._phases):
                paramlist.append(nn.Parameter(torch.randn(self._out_features, dtype=torch.float32)))
        self.reset()

    def reset(self):
        """Set the timestep t to 0"""
        self._t = 0

    @property
    def t(self) ->int:
        """The current timestep t"""
        return self._t

    @property
    def in_features(self) ->int:
        """Get the length of the input vector"""
        return self._in_features

    @property
    def out_features(self) ->int:
        """Get the length of the output vector"""
        return self._out_features

    @property
    def num_sinusoids(self) ->int:
        """Get the number of sinusoidal waves of the non-linear component"""
        return self._num_sinusoids

    @property
    def bias(self) ->bool:
        """Get whether or not the linear component has bias"""
        return self._bias

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        """Execute a forward pass"""
        u_linear = self._linear_component(x)
        t = self._t
        u_nonlinear = torch.zeros(self._out_features)
        for i in range(self._num_sinusoids):
            A = self._amplitudes[i]
            w = self._frequencies[i]
            phi = self._phases[i]
            u_nonlinear = u_nonlinear + A * torch.sin(w * t + phi)
        self._t += 1
        return u_linear + u_nonlinear


class MultiLayered(nn.Module):

    def __init__(self, *layers: nn.Module):
        super().__init__()
        self._submodules = nn.ModuleList(layers)

    def forward(self, x: 'torch.Tensor', h: 'Optional[dict]'=None):
        if h is None:
            h = {}
        new_h = {}
        for i, layer in enumerate(self._submodules):
            layer_h = h.get(i, None)
            if layer_h is None:
                layer_result = layer(x)
            else:
                layer_result = layer(x, h[i])
            if isinstance(layer_result, tuple):
                if len(layer_result) == 2:
                    x, layer_new_h = layer_result
                else:
                    raise ValueError(f'The layer number {i} returned a tuple of length {len(layer_result)}. A tensor or a tuple of two elements was expected.')
            elif isinstance(layer_result, torch.Tensor):
                x = layer_result
                layer_new_h = None
            else:
                raise TypeError(f'The layer number {i} returned an object of type {type(layer_result)}. A tensor or a tuple of two elements was expected.')
            if layer_new_h is not None:
                new_h[i] = layer_new_h
        if len(new_h) == 0:
            return x
        else:
            return x, new_h

    def __iter__(self):
        return self._submodules.__iter__()

    def __getitem__(self, i):
        return self._submodules[i]

    def __len__(self):
        return len(self._submodules)

    def append(self, module: 'nn.Module'):
        self._submodules.append(module)


def device_of_module(m: 'nn.Module', default: 'Optional[Union[str, torch.device]]'=None) ->torch.device:
    """
    Get the device in which the module exists.

    This function looks at the first parameter of the module, and returns
    its device. This function is not meant to be used on modules whose
    parameters exist on different devices.

    Args:
        m: The module whose device is being queried.
        default: The fallback device to return if the module has no
            parameters. If this is left as None, the fallback device
            is assumed to be "cpu".
    Returns:
        The device of the module, determined from its first parameter.
    """
    if default is None:
        default = torch.device('cpu')
    device = default
    for p in m.parameters():
        device = p.device
        break
    return device


class ActClipWrapperModule(nn.Module):

    def __init__(self, wrapped_module: 'nn.Module', obs_space: 'Box'):
        super().__init__()
        device = device_of_module(wrapped_module)
        if not isinstance(obs_space, Box):
            raise TypeError(f'Unrecognized observation space: {obs_space}')
        self.wrapped_module = wrapped_module
        self.register_buffer('_low', torch.from_numpy(obs_space.low))
        self.register_buffer('_high', torch.from_numpy(obs_space.high))

    def forward(self, x: 'torch.Tensor', h: 'Any'=None) ->Union[torch.Tensor, tuple]:
        if h is None:
            result = self.wrapped_module(x)
        else:
            result = self.wrapped_module(x, h)
        if isinstance(result, tuple):
            x, h = result
            got_h = True
        else:
            x = result
            h = None
            got_h = False
        x = torch.max(x, self._low)
        x = torch.min(x, self._high)
        if got_h:
            return x, h
        else:
            return x


class ObsNormWrapperModule(nn.Module):

    def __init__(self, wrapped_module: 'nn.Module', rn: 'Union[RunningStat, RunningNorm]'):
        super().__init__()
        device = device_of_module(wrapped_module)
        self.wrapped_module = wrapped_module
        with torch.no_grad():
            normalizer = deepcopy(rn.to_layer())
        self.normalizer = normalizer

    def forward(self, x: 'torch.Tensor', h: 'Any'=None) ->Union[torch.Tensor, tuple]:
        x = self.normalizer(x)
        if h is None:
            result = self.wrapped_module(x)
        else:
            result = self.wrapped_module(x, h)
        if isinstance(result, tuple):
            x, h = result
            got_h = True
        else:
            x = result
            h = None
            got_h = False
        if got_h:
            return x, h
        else:
            return x


def _clamp(x: 'torch.Tensor', min: 'Optional[float]', max: 'Optional[float]') ->torch.Tensor:
    """
    Clamp the tensor x according to the given min and max values.
    Unlike PyTorch's clamp, this function allows both min and max
    to be None, in which case no clamping will be done.

    Args:
        x: The tensor subject to the clamp operation.
        min: The minimum value.
        max: The maximum value.
    Returns:
        The result of the clamp operation, as a tensor.
        If both min and max were None, the returned object is x itself.
    """
    if min is None and max is None:
        return x
    else:
        return torch.clamp(x, min, max)


class ObsNormLayer(nn.Module):
    """
    An observation normalizer which behaves as a PyTorch Module.
    """

    def __init__(self, mean: 'torch.Tensor', stdev: 'torch.Tensor', low: 'Optional[float]'=None, high: 'Optional[float]'=None) ->None:
        """
        `__init__(...)`: Initialize the ObsNormLayer.

        Args:
            mean: The mean according to which the observations are to be
                normalized.
            stdev: The standard deviation according to which the observations
                are to be normalized.
            low: Optionally a real number if the result of the normalization
                is to be clipped. Represents the lower bound for the clipping
                operation.
            high: Optionally a real number if the result of the normalization
                is to be clipped. Represents the upper bound for the clipping
                operation.
        """
        super().__init__()
        self.register_buffer('_mean', mean)
        self.register_buffer('_stdev', stdev)
        self._lb = None if low is None else float(low)
        self._ub = None if high is None else float(high)

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        """
        Normalize an observation or a batch of observations.

        Args:
            x: The observation(s).
        Returns:
            The normalized counterpart of the observation(s).
        """
        return _clamp((x - self._mean) / self._stdev, self._lb, self._ub)


class StatefulModule(nn.Module):
    """
    A wrapper that provides a stateful interface for recurrent torch modules.

    If the torch module to be wrapped is non-recurrent and its forward method
    has a single input (the input tensor) and a single output (the output
    tensor), then this wrapper module acts as a no-op wrapper.

    If the torch module to be wrapped is recurrent and its forward method has
    two inputs (the input tensor and an optional second argument for the hidden
    state) and two outputs (the output tensor and the new hidden state), then
    this wrapper brings a new forward-passing interface. In this new interface,
    the forward method has a single input (the input tensor) and a single
    output (the output tensor). The hidden states, instead of being
    explicitly requested via a second argument and returned as a second
    result, are stored and used by the wrapper.
    When a new series of inputs is to be used, one has to call the `reset()`
    method of this wrapper.
    """

    def __init__(self, wrapped_module: 'nn.Module'):
        """
        `__init__(...)`: Initialize the StatefulModule.

        Args:
            wrapped_module: The `torch.nn.Module` instance to wrap.
        """
        super().__init__()
        self._hidden: 'Any' = None
        self.wrapped_module = wrapped_module

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        if self._hidden is None:
            out = self.wrapped_module(x)
        else:
            out = self.wrapped_module(x, self._hidden)
        if isinstance(out, tuple):
            y, self._hidden = out
        else:
            y = out
            self._hidden = None
        return y

    def reset(self):
        """
        Reset the hidden state, if any.
        """
        self._hidden = None


class DummyRecurrentNet(nn.Module):

    def __init__(self, first_value: 'int'=1):
        super().__init__()
        self.first_value = int(first_value)

    def forward(self, x: 'torch.Tensor', h: 'Optional[torch.Tensor]'=None) ->tuple:
        if h is None:
            h = torch.tensor(self.first_value, dtype=torch.int64, device=x.device)
        return x * torch.as_tensor(h, dtype=x.dtype, device=x.device), h + 1


class Unbatched:


    class LSTM(nn.Module):

        def __init__(self, *args, **kwargs):
            super().__init__()
            self.lstm = nn.LSTM(*args, **kwargs)

        def forward(self, x: 'torch.Tensor', h: 'Optional[tuple]'=None) ->tuple:
            if h is not None:
                a, b = h
                a = a.reshape(1, 1, -1)
                b = b.reshape(1, 1, -1)
                h = a, b
            x = x.reshape(1, 1, -1)
            x, h = self.lstm(x, h)
            x = x.reshape(-1)
            a, b = h
            a = a.reshape(-1)
            b = b.reshape(-1)
            h = a, b
            return x, h


    class RNN(nn.Module):

        def __init__(self, *args, **kwargs):
            super().__init__()
            self.rnn = nn.RNN(*args, **kwargs)

        def forward(self, x: 'torch.Tensor', h: 'Optional[torch.Tensor]'=None) ->tuple:
            if h is not None:
                h = h.reshape(1, 1, -1)
            x = x.reshape(1, 1, -1)
            x, h = self.rnn(x, h)
            x = x.reshape(-1)
            h = h.reshape(-1)
            return x, h


class DummyComposedRecurrent(nn.Module):

    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([Unbatched.RNN(3, 5), nn.Linear(5, 8), Unbatched.LSTM(8, 2)])

    def forward(self, x: 'torch.Tensor', h: 'Optional[dict]'=None) ->tuple:
        if h is None:
            h = {(0): None, (2): None}
        x, h[0] = self.layers[0](x, h[0])
        x = self.layers[1](x)
        x = torch.tanh(x)
        x, h[2] = self.layers[2](x, h[2])
        return x, h


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (Apply,
     lambda: ([], {'operator': '+', 'argument': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Bin,
     lambda: ([], {'lb': 4, 'ub': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Clip,
     lambda: ([], {'lb': 4, 'ub': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (DummyRecurrentNet,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (LSTM,
     lambda: ([], {'input_size': 4, 'hidden_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (LocomotorNet,
     lambda: ([], {'in_features': 4, 'out_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (MultiLayered,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (RNN,
     lambda: ([], {'input_size': 4, 'hidden_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Round,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Slice,
     lambda: ([], {'from_index': 4, 'to_index': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (StatefulModule,
     lambda: ([], {'wrapped_module': torch.nn.ReLU()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (StructuredControlNet,
     lambda: ([], {'in_features': 4, 'out_features': 4, 'num_layers': 1, 'hidden_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
]

