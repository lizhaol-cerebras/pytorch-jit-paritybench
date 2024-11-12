
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


from torch.nn import functional as F


from collections import defaultdict


import time


import numpy as np


import math


from torch import nn


import random


import warnings


import abc


import copy


import torch.nn.functional as F


import itertools


import collections


from collections import deque


from torch.distributions import Normal


from torch.distributions.independent import Independent


import torch.nn as nn


from torch.optim import Optimizer


import tensorflow as tf


from functools import partial


from torch import tensor


class NonLinearity(nn.Module):
    """Wrapper class for non linear function or module.

    Args:
        non_linear (callable or type): Non-linear function or type to be
            wrapped.

    """

    def __init__(self, non_linear):
        super().__init__()
        if isinstance(non_linear, type):
            self.module = non_linear()
        elif callable(non_linear):
            self.module = copy.deepcopy(non_linear)
        else:
            raise ValueError('Non linear function {} is not supported'.format(non_linear))

    def forward(self, input_value):
        """Forward method.

        Args:
            input_value (torch.Tensor): Input values

        Returns:
            torch.Tensor: Output value

        """
        return self.module(input_value)

    def __repr__(self):
        """object representation method."""
        return repr(self.module)


@dataclass(frozen=True)
class InOutSpec:
    """Describes the input and output spaces of a primitive or module."""
    input_space: 'akro.Space'
    output_space: 'akro.Space'


def _check_spec(spec, image_format):
    """Check that an InOutSpec is suitable for a CNNModule.

    Args:
        spec (garage.InOutSpec): Specification of inputs and outputs.  The
            input should be in 'NCHW' format: [batch_size, channel, height,
            width].  Will print a warning if the channel size is not 1 or 3.
            If output_space is specified, then a final linear layer will be
            inserted to map to that dimensionality.  If output_space is None,
            it will be filled in with the computed output space.
        image_format (str): Either 'NCHW' or 'NHWC'. Should match the input
            specification. Gym uses NHWC by default, but PyTorch uses NCHW by
            default.

    Returns:
        tuple[int, int, int]: The input channels, height, and width.

    Raises:
        ValueError: If spec isn't suitable for a CNNModule.

    """
    input_space = spec.input_space
    output_space = spec.output_space
    if getattr(input_space, 'shape', None) is None:
        raise ValueError(f'input_space to CNNModule is {input_space}, but should be an akro.Box or akro.Image')
    elif len(input_space.shape) != 3:
        raise ValueError(f'Input to CNNModule is {input_space}, but should have three dimensions.')
    if output_space is not None and not (hasattr(output_space, 'shape') and len(output_space.shape) == 1):
        raise ValueError(f'output_space to CNNModule is {output_space}, but should be an akro.Box with a single dimension or None')
    if image_format == 'NCHW':
        in_channels = spec.input_space.shape[0]
        height = spec.input_space.shape[1]
        width = spec.input_space.shape[2]
    elif image_format == 'NHWC':
        height = spec.input_space.shape[0]
        width = spec.input_space.shape[1]
        in_channels = spec.input_space.shape[2]
    else:
        raise ValueError(f"image_format has value {image_format!r}, but must be either 'NCHW' or 'NHWC'")
    if in_channels not in (1, 3):
        warnings.warn(f'CNNModule input has {in_channels} channels, but 1 or 3 channels are typical. Consider changing the CNN image_format.')
    return in_channels, height, width


def _echo_run_names(header, d):
    """Echo run names to the command line.

    Args:
        header (str): The header name.
        d (dict): The dict containing benchmark options.

    """
    click.echo('-----' + header + '-----')
    for name in d:
        click.echo(name)
    click.echo()


def _get_runs_dict(module):
    """Return a dict containing benchmark options of the module.

    Dict of (str: obj) representing benchmark name and its function object.

    Args:
        module (object): Module object.

    Returns:
        dict: Benchmark options of the module.

    """
    d = {}
    for name, obj in inspect.getmembers(module):
        if inspect.isfunction(obj) and name.endswith('benchmarks'):
            d[name] = obj
    return d


def expand_var(name, item, n_expected, reference):
    """Expand a variable to an expected length.

    This is used to handle arguments to primitives that can all be reasonably
    set to the same value, or multiple different values.

    Args:
        name (str): Name of variable being expanded.
        item (any): Element being expanded.
        n_expected (int): Number of elements expected.
        reference (str): Source of n_expected.

    Returns:
        list: List of references to item or item itself.

    Raises:
        ValueError: If the variable is a sequence but length of the variable
            is not 1 or n_expected.

    """
    if n_expected == 1:
        warnings.warn(f'Providing a {reference} of length 1 prevents {name} from being expanded.')
    if isinstance(item, (list, tuple)):
        if len(item) == n_expected:
            return item
        elif len(item) == 1:
            return list(item) * n_expected
        else:
            raise ValueError(f'{name} is length {len(item)} but should be length {n_expected} to match {reference}')
    else:
        return [item] * n_expected


def _value_at_axis(value, axis):
    """Get the value for a particular axis.

    Args:
        value (tuple or list or int): Possible tuple of per-axis values.
        axis (int): Axis to get value for.

    Returns:
        int: the value at the available axis.

    """
    if not isinstance(value, (list, tuple)):
        return value
    if len(value) == 1:
        return value[0]
    else:
        return value[axis]


def output_height_2d(layer, height):
    """Compute the output height of a torch.nn.Conv2d, assuming NCHW format.

    This requires knowing the input height. Because NCHW format makes this very
    easy to mix up, this is a seperate function from conv2d_output_height.

    It also works on torch.nn.MaxPool2d.

    This function implements the formula described in the torch.nn.Conv2d
    documentation:
    https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html

    Args:
        layer (torch.nn.Conv2d): The layer to compute output size for.
        height (int): The height of the input image.

    Returns:
        int: The height of the output image.

    """
    assert isinstance(layer, (torch.nn.Conv2d, torch.nn.MaxPool2d))
    padding = _value_at_axis(layer.padding, 0)
    dilation = _value_at_axis(layer.dilation, 0)
    kernel_size = _value_at_axis(layer.kernel_size, 0)
    stride = _value_at_axis(layer.stride, 0)
    return math.floor((height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1)


def output_width_2d(layer, width):
    """Compute the output width of a torch.nn.Conv2d, assuming NCHW format.

    This requires knowing the input width. Because NCHW format makes this very
    easy to mix up, this is a seperate function from conv2d_output_height.

    It also works on torch.nn.MaxPool2d.

    This function implements the formula described in the torch.nn.Conv2d
    documentation:
    https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html

    Args:
        layer (torch.nn.Conv2d): The layer to compute output size for.
        width (int): The width of the input image.

    Returns:
        int: The width of the output image.

    """
    assert isinstance(layer, (torch.nn.Conv2d, torch.nn.MaxPool2d))
    padding = _value_at_axis(layer.padding, 1)
    dilation = _value_at_axis(layer.dilation, 1)
    kernel_size = _value_at_axis(layer.kernel_size, 1)
    stride = _value_at_axis(layer.stride, 1)
    return math.floor((width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1)


class CNNModule(nn.Module):
    """Convolutional neural network (CNN) model in pytorch.

    Args:
        spec (garage.InOutSpec): Specification of inputs and outputs.
            The input should be in 'NCHW' format: [batch_size, channel, height,
            width]. Will print a warning if the channel size is not 1 or 3.
            If output_space is specified, then a final linear layer will be
            inserted to map to that dimensionality.
            If output_space is None, it will be filled in with the computed
            output space.
        image_format (str): Either 'NCHW' or 'NHWC'. Should match the input
            specification. Gym uses NHWC by default, but PyTorch uses NCHW by
            default.
        hidden_channels (tuple[int]): Number of output channels for CNN.
            For example, (3, 32) means there are two convolutional layers.
            The filter for the first conv layer outputs 3 channels
            and the second one outputs 32 channels.
        kernel_sizes (tuple[int]): Dimension of the conv filters.
            For example, (3, 5) means there are two convolutional layers.
            The filter for first layer is of dimension (3 x 3)
            and the second one is of dimension (5 x 5).
        strides (tuple[int]): The stride of the sliding window. For example,
            (1, 2) means there are two convolutional layers. The stride of the
            filter for first layer is 1 and that of the second layer is 2.
        paddings (tuple[int]): Amount of zero-padding added to both sides of
            the input of a conv layer.
        padding_mode (str): The type of padding algorithm to use, i.e.
            'constant', 'reflect', 'replicate' or 'circular' and
            by default is 'zeros'.
        hidden_nonlinearity (callable or torch.nn.Module):
            Activation function for intermediate dense layer(s).
            It should return a torch.Tensor. Set it to None to maintain a
            linear activation.
        hidden_b_init (callable): Initializer function for the bias
            of intermediate dense layer(s). The function should return a
            torch.Tensor.
        max_pool (bool): Bool for using max-pooling or not.
        pool_shape (tuple[int]): Dimension of the pooling layer(s). For
            example, (2, 2) means that all pooling layers are of the same
            shape (2, 2).
        pool_stride (tuple[int]): The strides of the pooling layer(s). For
            example, (2, 2) means that all the pooling layers have
            strides (2, 2).
        layer_normalization (bool): Bool for using layer normalization or not.
        hidden_w_init (callable): Initializer function for the weight
            of intermediate dense layer(s). The function should return a
            torch.Tensor.
        enable_cudnn_benchmarks (bool): Whether to enable cudnn benchmarks
            in `torch`. If enabled, the backend selects the CNN benchamark
            algorithm with the best performance.
    """

    def __init__(self, spec, image_format, hidden_channels, *, kernel_sizes, strides, paddings=0, padding_mode='zeros', hidden_nonlinearity=nn.ReLU, hidden_w_init=nn.init.xavier_uniform_, hidden_b_init=nn.init.zeros_, max_pool=False, pool_shape=None, pool_stride=1, layer_normalization=False, enable_cudnn_benchmarks=True):
        super().__init__()
        assert len(hidden_channels) > 0
        in_channels, height, width = _check_spec(spec, image_format)
        self._format = image_format
        kernel_sizes = expand_var('kernel_sizes', kernel_sizes, len(hidden_channels), 'hidden_channels')
        strides = expand_var('strides', strides, len(hidden_channels), 'hidden_channels')
        paddings = expand_var('paddings', paddings, len(hidden_channels), 'hidden_channels')
        pool_shape = expand_var('pool_shape', pool_shape, len(hidden_channels), 'hidden_channels')
        pool_stride = expand_var('pool_stride', pool_stride, len(hidden_channels), 'hidden_channels')
        self._cnn_layers = nn.Sequential()
        torch.backends.cudnn.benchmark = enable_cudnn_benchmarks
        out_channels = in_channels
        for i, out_channels in enumerate(hidden_channels):
            conv_layer = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_sizes[i], stride=strides[i], padding=paddings[i], padding_mode=padding_mode)
            height = output_height_2d(conv_layer, height)
            width = output_width_2d(conv_layer, width)
            hidden_w_init(conv_layer.weight)
            hidden_b_init(conv_layer.bias)
            self._cnn_layers.add_module(f'conv_{i}', conv_layer)
            if layer_normalization:
                self._cnn_layers.add_module(f'layer_norm_{i}', nn.LayerNorm((out_channels, height, width)))
            if hidden_nonlinearity:
                self._cnn_layers.add_module(f'non_linearity_{i}', NonLinearity(hidden_nonlinearity))
            if max_pool:
                pool = nn.MaxPool2d(kernel_size=pool_shape[i], stride=pool_stride[i])
                height = output_height_2d(pool, height)
                width = output_width_2d(pool, width)
                self._cnn_layers.add_module(f'max_pooling_{i}', pool)
            in_channels = out_channels
        output_dims = out_channels * height * width
        if spec.output_space is None:
            final_spec = InOutSpec(spec.input_space, akro.Box(low=-np.inf, high=np.inf, shape=(output_dims,)))
            self._final_layer = None
        else:
            final_spec = spec
            self._final_layer = nn.Linear(output_dims, spec.output_space.shape[0])
        self.spec = final_spec

    def forward(self, x):
        """Forward method.

        Args:
            x (torch.Tensor): Input values. Should match image_format
                specified at construction (either NCHW or NCWH).

        Returns:
            List[torch.Tensor]: Output values

        """
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        if isinstance(self.spec.input_space, akro.Image):
            x = torch.div(x, 255.0)
        assert len(x.shape) == 4
        if self._format == 'NHWC':
            x = x.permute((0, 3, 1, 2))
        for layer in self._cnn_layers:
            x = layer(x)
        if self._format == 'NHWC':
            x = x.permute((0, 2, 3, 1))
        x = x.reshape(x.shape[0], -1)
        if self._final_layer is not None:
            x = self._final_layer(x)
        return x


class MultiHeadedMLPModule(nn.Module):
    """MultiHeadedMLPModule Model.

    A PyTorch module composed only of a multi-layer perceptron (MLP) with
    multiple parallel output layers which maps real-valued inputs to
    real-valued outputs. The length of outputs is n_heads and shape of each
    output element is depend on each output dimension

    Args:
        n_heads (int): Number of different output layers
        input_dim (int): Dimension of the network input.
        output_dims (int or list or tuple): Dimension of the network output.
        hidden_sizes (list[int]): Output dimension of dense layer(s).
            For example, (32, 32) means this MLP consists of two
            hidden layers, each with 32 hidden units.
        hidden_nonlinearity (callable or torch.nn.Module or list or tuple):
            Activation function for intermediate dense layer(s).
            It should return a torch.Tensor. Set it to None to maintain a
            linear activation.
        hidden_w_init (callable): Initializer function for the weight
            of intermediate dense layer(s). The function should return a
            torch.Tensor.
        hidden_b_init (callable): Initializer function for the bias
            of intermediate dense layer(s). The function should return a
            torch.Tensor.
        output_nonlinearities (callable or torch.nn.Module or list or tuple):
            Activation function for output dense layer. It should return a
            torch.Tensor. Set it to None to maintain a linear activation.
            Size of the parameter should be 1 or equal to n_head
        output_w_inits (callable or list or tuple): Initializer function for
            the weight of output dense layer(s). The function should return a
            torch.Tensor. Size of the parameter should be 1 or equal to n_head
        output_b_inits (callable or list or tuple): Initializer function for
            the bias of output dense layer(s). The function should return a
            torch.Tensor. Size of the parameter should be 1 or equal to n_head
        layer_normalization (bool): Bool for using layer normalization or not.

    """

    def __init__(self, n_heads, input_dim, output_dims, hidden_sizes, hidden_nonlinearity=torch.relu, hidden_w_init=nn.init.xavier_normal_, hidden_b_init=nn.init.zeros_, output_nonlinearities=None, output_w_inits=nn.init.xavier_normal_, output_b_inits=nn.init.zeros_, layer_normalization=False):
        super().__init__()
        self._layers = nn.ModuleList()
        output_dims = self._check_parameter_for_output_layer('output_dims', output_dims, n_heads)
        output_w_inits = self._check_parameter_for_output_layer('output_w_inits', output_w_inits, n_heads)
        output_b_inits = self._check_parameter_for_output_layer('output_b_inits', output_b_inits, n_heads)
        output_nonlinearities = self._check_parameter_for_output_layer('output_nonlinearities', output_nonlinearities, n_heads)
        self._layers = nn.ModuleList()
        prev_size = input_dim
        for size in hidden_sizes:
            hidden_layers = nn.Sequential()
            if layer_normalization:
                hidden_layers.add_module('layer_normalization', nn.LayerNorm(prev_size))
            linear_layer = nn.Linear(prev_size, size)
            hidden_w_init(linear_layer.weight)
            hidden_b_init(linear_layer.bias)
            hidden_layers.add_module('linear', linear_layer)
            if hidden_nonlinearity:
                hidden_layers.add_module('non_linearity', NonLinearity(hidden_nonlinearity))
            self._layers.append(hidden_layers)
            prev_size = size
        self._output_layers = nn.ModuleList()
        for i in range(n_heads):
            output_layer = nn.Sequential()
            linear_layer = nn.Linear(prev_size, output_dims[i])
            output_w_inits[i](linear_layer.weight)
            output_b_inits[i](linear_layer.bias)
            output_layer.add_module('linear', linear_layer)
            if output_nonlinearities[i]:
                output_layer.add_module('non_linearity', NonLinearity(output_nonlinearities[i]))
            self._output_layers.append(output_layer)

    @classmethod
    def _check_parameter_for_output_layer(cls, var_name, var, n_heads):
        """Check input parameters for output layer are valid.

        Args:
            var_name (str): variable name
            var (any): variable to be checked
            n_heads (int): number of head

        Returns:
            list: list of variables (length of n_heads)

        Raises:
            ValueError: if the variable is a list but length of the variable
                is not equal to n_heads

        """
        if isinstance(var, (list, tuple)):
            if len(var) == 1:
                return list(var) * n_heads
            if len(var) == n_heads:
                return var
            msg = '{} should be either an integer or a collection of length n_heads ({}), but {} provided.'
            raise ValueError(msg.format(var_name, n_heads, var))
        return [copy.deepcopy(var) for _ in range(n_heads)]

    def forward(self, input_val):
        """Forward method.

        Args:
            input_val (torch.Tensor): Input values with (N, *, input_dim)
                shape.

        Returns:
            List[torch.Tensor]: Output values

        """
        x = input_val
        for layer in self._layers:
            x = layer(x)
        return [output_layer(x) for output_layer in self._output_layers]


class MLPModule(MultiHeadedMLPModule):
    """MLP Model.

    A Pytorch module composed only of a multi-layer perceptron (MLP), which
    maps real-valued inputs to real-valued outputs.

    Args:
        input_dim (int) : Dimension of the network input.
        output_dim (int): Dimension of the network output.
        hidden_sizes (list[int]): Output dimension of dense layer(s).
            For example, (32, 32) means this MLP consists of two
            hidden layers, each with 32 hidden units.
        hidden_nonlinearity (callable or torch.nn.Module): Activation function
            for intermediate dense layer(s). It should return a torch.Tensor.
            Set it to None to maintain a linear activation.
        hidden_w_init (callable): Initializer function for the weight
            of intermediate dense layer(s). The function should return a
            torch.Tensor.
        hidden_b_init (callable): Initializer function for the bias
            of intermediate dense layer(s). The function should return a
            torch.Tensor.
        output_nonlinearity (callable or torch.nn.Module): Activation function
            for output dense layer. It should return a torch.Tensor.
            Set it to None to maintain a linear activation.
        output_w_init (callable): Initializer function for the weight
            of output dense layer(s). The function should return a
            torch.Tensor.
        output_b_init (callable): Initializer function for the bias
            of output dense layer(s). The function should return a
            torch.Tensor.
        layer_normalization (bool): Bool for using layer normalization or not.

    """

    def __init__(self, input_dim, output_dim, hidden_sizes, hidden_nonlinearity=F.relu, hidden_w_init=nn.init.xavier_normal_, hidden_b_init=nn.init.zeros_, output_nonlinearity=None, output_w_init=nn.init.xavier_normal_, output_b_init=nn.init.zeros_, layer_normalization=False):
        super().__init__(1, input_dim, output_dim, hidden_sizes, hidden_nonlinearity, hidden_w_init, hidden_b_init, output_nonlinearity, output_w_init, output_b_init, layer_normalization)
        self._output_dim = output_dim

    def forward(self, input_value):
        """Forward method.

        Args:
            input_value (torch.Tensor): Input values with (N, *, input_dim)
                shape.

        Returns:
            torch.Tensor: Output value

        """
        return super().forward(input_value)[0]

    @property
    def output_dim(self):
        """Return output dimension of network.

        Returns:
            int: Output dimension of network.

        """
        return self._output_dim


class DiscreteCNNModule(nn.Module):
    """Discrete CNN Module.

    A CNN followed by one or more fully connected layers with a set number
    of discrete outputs.

    Args:
        spec (garage.InOutSpec): Specification of inputs and outputs.
            The input should be in 'NCHW' format: [batch_size, channel, height,
            width]. Will print a warning if the channel size is not 1 or 3.
            The output space will be flattened.
        image_format (str): Either 'NCHW' or 'NHWC'. Should match the input
            specification. Gym uses NHWC by default, but PyTorch uses NCHW by
            default.
        kernel_sizes (tuple[int]): Dimension of the conv filters.
            For example, (3, 5) means there are two convolutional layers.
            The filter for first layer is of dimension (3 x 3)
            and the second one is of dimension (5 x 5).
        strides (tuple[int]): The stride of the sliding window. For example,
            (1, 2) means there are two convolutional layers. The stride of the
            filter for first layer is 1 and that of the second layer is 2.
        hidden_channels (tuple[int]): Number of output channels for CNN.
            For example, (3, 32) means there are two convolutional layers.
            The filter for the first conv layer outputs 3 channels
            and the second one outputs 32 channels.
        hidden_sizes (list[int]): Output dimension of dense layer(s) for
            the MLP for mean. For example, (32, 32) means the MLP consists
            of two hidden layers, each with 32 hidden units.
        mlp_hidden_nonlinearity (callable): Activation function for
            intermediate dense layer(s) in the MLP. It should return
            a torch.Tensor. Set it to None to maintain a linear activation.
        cnn_hidden_nonlinearity (callable): Activation function for
            intermediate CNN layer(s). It should return a torch.Tensor.
            Set it to None to maintain a linear activation.
        hidden_w_init (callable): Initializer function for the weight
            of intermediate dense layer(s). The function should return a
            torch.Tensor.
        hidden_b_init (callable): Initializer function for the bias
            of intermediate dense layer(s). The function should return a
            torch.Tensor.
        paddings (tuple[int]):  Zero-padding added to both sides of the input
        padding_mode (str): The type of padding algorithm to use,
            either 'SAME' or 'VALID'.
        max_pool (bool): Bool for using max-pooling or not.
        pool_shape (tuple[int]): Dimension of the pooling layer(s). For
            example, (2, 2) means that all the pooling layers are of the same
            shape (2, 2).
        pool_stride (tuple[int]): The strides of the pooling layer(s). For
            example, (2, 2) means that all the pooling layers have
            strides (2, 2).
        output_nonlinearity (callable): Activation function for output dense
            layer. It should return a torch.Tensor. Set it to None to
            maintain a linear activation.
        output_w_init (callable): Initializer function for the weight
            of output dense layer(s). The function should return a
            torch.Tensor.
        output_b_init (callable): Initializer function for the bias
            of output dense layer(s). The function should return a
            torch.Tensor.
        layer_normalization (bool): Bool for using layer normalization or not.
    """

    def __init__(self, spec, image_format, *, kernel_sizes, hidden_channels, strides, hidden_sizes=(32, 32), cnn_hidden_nonlinearity=nn.ReLU, mlp_hidden_nonlinearity=nn.ReLU, hidden_w_init=nn.init.xavier_uniform_, hidden_b_init=nn.init.zeros_, paddings=0, padding_mode='zeros', max_pool=False, pool_shape=None, pool_stride=1, output_nonlinearity=None, output_w_init=nn.init.xavier_uniform_, output_b_init=nn.init.zeros_, layer_normalization=False):
        super().__init__()
        cnn_spec = InOutSpec(input_space=spec.input_space, output_space=None)
        cnn_module = CNNModule(spec=cnn_spec, image_format=image_format, kernel_sizes=kernel_sizes, strides=strides, hidden_w_init=hidden_w_init, hidden_b_init=hidden_b_init, hidden_channels=hidden_channels, hidden_nonlinearity=cnn_hidden_nonlinearity, paddings=paddings, padding_mode=padding_mode, max_pool=max_pool, layer_normalization=layer_normalization, pool_shape=pool_shape, pool_stride=pool_stride)
        flat_dim = cnn_module.spec.output_space.flat_dim
        output_dim = spec.output_space.flat_dim
        mlp_module = MLPModule(flat_dim, output_dim, hidden_sizes, hidden_nonlinearity=mlp_hidden_nonlinearity, hidden_w_init=hidden_w_init, hidden_b_init=hidden_b_init, output_nonlinearity=output_nonlinearity, output_w_init=output_w_init, output_b_init=output_b_init, layer_normalization=layer_normalization)
        if mlp_hidden_nonlinearity is None:
            self._module = nn.Sequential(cnn_module, nn.Flatten(), mlp_module)
        else:
            self._module = nn.Sequential(cnn_module, mlp_hidden_nonlinearity(), nn.Flatten(), mlp_module)

    def forward(self, inputs):
        """Forward method.

        Args:
            inputs (torch.Tensor): Inputs to the model of shape
                (input_shape*).

        Returns:
            torch.Tensor: Output tensor of shape :math:`(N, output_dim)`.

        """
        return self._module(inputs)


class TanhNormal(torch.distributions.Distribution):
    """A distribution induced by applying a tanh transformation to a Gaussian random variable.

    Algorithms like SAC and Pearl use this transformed distribution.
    It can be thought of as a distribution of X where
        :math:`Y ~ \\mathcal{N}(\\mu, \\sigma)`
        :math:`X = tanh(Y)`

    Args:
        loc (torch.Tensor): The mean of this distribution.
        scale (torch.Tensor): The stdev of this distribution.

    """

    def __init__(self, loc, scale):
        self._normal = Independent(Normal(loc, scale), 1)
        super().__init__()

    def log_prob(self, value, pre_tanh_value=None, epsilon=1e-06):
        """The log likelihood of a sample on the this Tanh Distribution.

        Args:
            value (torch.Tensor): The sample whose loglikelihood is being
                computed.
            pre_tanh_value (torch.Tensor): The value prior to having the tanh
                function applied to it but after it has been sampled from the
                normal distribution.
            epsilon (float): Regularization constant. Making this value larger
                makes the computation more stable but less precise.

        Note:
              when pre_tanh_value is None, an estimate is made of what the
              value is. This leads to a worse estimation of the log_prob.
              If the value being used is collected from functions like
              `sample` and `rsample`, one can instead use functions like
              `sample_return_pre_tanh_value` or
              `rsample_return_pre_tanh_value`


        Returns:
            torch.Tensor: The log likelihood of value on the distribution.

        """
        if pre_tanh_value is None:
            pre_tanh_value = torch.log((1 + epsilon + value) / (1 + epsilon - value)) / 2
        norm_lp = self._normal.log_prob(pre_tanh_value)
        ret = norm_lp - torch.sum(torch.log(self._clip_but_pass_gradient(1.0 - value ** 2) + epsilon), axis=-1)
        return ret

    def sample(self, sample_shape=torch.Size()):
        """Return a sample, sampled from this TanhNormal Distribution.

        Args:
            sample_shape (list): Shape of the returned value.

        Note:
            Gradients `do not` pass through this operation.

        Returns:
            torch.Tensor: Sample from this TanhNormal distribution.

        """
        with torch.no_grad():
            return self.rsample(sample_shape=sample_shape)

    def rsample(self, sample_shape=torch.Size()):
        """Return a sample, sampled from this TanhNormal Distribution.

        Args:
            sample_shape (list): Shape of the returned value.

        Note:
            Gradients pass through this operation.

        Returns:
            torch.Tensor: Sample from this TanhNormal distribution.

        """
        z = self._normal.rsample(sample_shape)
        return torch.tanh(z)

    def rsample_with_pre_tanh_value(self, sample_shape=torch.Size()):
        """Return a sample, sampled from this TanhNormal distribution.

        Returns the sampled value before the tanh transform is applied and the
        sampled value with the tanh transform applied to it.

        Args:
            sample_shape (list): shape of the return.

        Note:
            Gradients pass through this operation.

        Returns:
            torch.Tensor: Samples from this distribution.
            torch.Tensor: Samples from the underlying
                :obj:`torch.distributions.Normal` distribution, prior to being
                transformed with `tanh`.

        """
        z = self._normal.rsample(sample_shape)
        return z, torch.tanh(z)

    def cdf(self, value):
        """Returns the CDF at the value.

        Returns the cumulative density/mass function evaluated at
        `value` on the underlying normal distribution.

        Args:
            value (torch.Tensor): The element where the cdf is being evaluated
                at.

        Returns:
            torch.Tensor: the result of the cdf being computed.

        """
        return self._normal.cdf(value)

    def icdf(self, value):
        """Returns the icdf function evaluated at `value`.

        Returns the icdf function evaluated at `value` on the underlying
        normal distribution.

        Args:
            value (torch.Tensor): The element where the cdf is being evaluated
                at.

        Returns:
            torch.Tensor: the result of the cdf being computed.

        """
        return self._normal.icdf(value)

    @classmethod
    def _from_distribution(cls, new_normal):
        """Construct a new TanhNormal distribution from a normal distribution.

        Args:
            new_normal (Independent(Normal)): underlying normal dist for
                the new TanhNormal distribution.

        Returns:
            TanhNormal: A new distribution whose underlying normal dist
                is new_normal.

        """
        new = cls(torch.zeros(1), torch.zeros(1))
        new._normal = new_normal
        return new

    def expand(self, batch_shape, _instance=None):
        """Returns a new TanhNormal distribution.

        (or populates an existing instance provided by a derived class) with
        batch dimensions expanded to `batch_shape`. This method calls
        :class:`~torch.Tensor.expand` on the distribution's parameters. As
        such, this does not allocate new memory for the expanded distribution
        instance. Additionally, this does not repeat any args checking or
        parameter broadcasting in `__init__.py`, when an instance is first
        created.

        Args:
            batch_shape (torch.Size): the desired expanded size.
            _instance(instance): new instance provided by subclasses that
                need to override `.expand`.

        Returns:
            Instance: New distribution instance with batch dimensions expanded
            to `batch_size`.

        """
        new_normal = self._normal.expand(batch_shape, _instance)
        new = self._from_distribution(new_normal)
        return new

    def enumerate_support(self, expand=True):
        """Returns tensor containing all values supported by a discrete dist.

        The result will enumerate over dimension 0, so the shape
        of the result will be `(cardinality,) + batch_shape + event_shape`
        (where `event_shape = ()` for univariate distributions).

        Note that this enumerates over all batched tensors in lock-step
        `[[0, 0], [1, 1], ...]`. With `expand=False`, enumeration happens
        along dim 0, but with the remaining batch dimensions being
        singleton dimensions, `[[0], [1], ..`.

        To iterate over the full Cartesian product use
        `itertools.product(m.enumerate_support())`.

        Args:
            expand (bool): whether to expand the support over the
                batch dims to match the distribution's `batch_shape`.

        Note:
            Calls the enumerate_support function of the underlying normal
            distribution.

        Returns:
            torch.Tensor: Tensor iterating over dimension 0.

        """
        return self._normal.enumerate_support(expand)

    @property
    def mean(self):
        """torch.Tensor: mean of the distribution."""
        return torch.tanh(self._normal.mean)

    @property
    def variance(self):
        """torch.Tensor: variance of the underlying normal distribution."""
        return self._normal.variance

    def entropy(self):
        """Returns entropy of the underlying normal distribution.

        Returns:
            torch.Tensor: entropy of the underlying normal distribution.

        """
        return self._normal.entropy()

    @staticmethod
    def _clip_but_pass_gradient(x, lower=0.0, upper=1.0):
        """Clipping function that allows for gradients to flow through.

        Args:
            x (torch.Tensor): value to be clipped
            lower (float): lower bound of clipping
            upper (float): upper bound of clipping

        Returns:
            torch.Tensor: x clipped between lower and upper.

        """
        clip_up = (x > upper).float()
        clip_low = (x < lower).float()
        with torch.no_grad():
            clip = (upper - x) * clip_up + (lower - x) * clip_low
        return x + clip

    def __repr__(self):
        """Returns the parameterization of the distribution.

        Returns:
            str: The parameterization of the distribution and underlying
                distribution.

        """
        return self.__class__.__name__


class GaussianMLPBaseModule(nn.Module):
    """Base of GaussianMLPModel.

    Args:
        input_dim (int): Input dimension of the model.
        output_dim (int): Output dimension of the model.
        hidden_sizes (list[int]): Output dimension of dense layer(s) for
            the MLP for mean. For example, (32, 32) means the MLP consists
            of two hidden layers, each with 32 hidden units.
        hidden_nonlinearity (callable): Activation function for intermediate
            dense layer(s). It should return a torch.Tensor. Set it to
            None to maintain a linear activation.
        hidden_w_init (callable): Initializer function for the weight
            of intermediate dense layer(s). The function should return a
            torch.Tensor.
        hidden_b_init (callable): Initializer function for the bias
            of intermediate dense layer(s). The function should return a
            torch.Tensor.
        output_nonlinearity (callable): Activation function for output dense
            layer. It should return a torch.Tensor. Set it to None to
            maintain a linear activation.
        output_w_init (callable): Initializer function for the weight
            of output dense layer(s). The function should return a
            torch.Tensor.
        output_b_init (callable): Initializer function for the bias
            of output dense layer(s). The function should return a
            torch.Tensor.
        learn_std (bool): Is std trainable.
        init_std (float): Initial value for std.
            (plain value - not log or exponentiated).
        std_hidden_sizes (list[int]): Output dimension of dense layer(s) for
            the MLP for std. For example, (32, 32) means the MLP consists
            of two hidden layers, each with 32 hidden units.
        min_std (float): If not None, the std is at least the value of min_std,
            to avoid numerical issues (plain value - not log or exponentiated).
        max_std (float): If not None, the std is at most the value of max_std,
            to avoid numerical issues (plain value - not log or exponentiated).
        std_hidden_nonlinearity (callable): Nonlinearity for each hidden layer
            in the std network.
        std_hidden_w_init (callable):  Initializer function for the weight
            of hidden layer (s).
        std_hidden_b_init (callable): Initializer function for the bias
            of intermediate dense layer(s).
        std_output_nonlinearity (callable): Activation function for output
            dense layer in the std network. It should return a torch.Tensor.
            Set it to None to maintain a linear activation.
        std_output_w_init (callable): Initializer function for the weight
            of output dense layer(s) in the std network.
        std_parameterization (str): How the std should be parametrized. There
            are two options:
            - exp: the logarithm of the std will be stored, and applied a
               exponential transformation.
            - softplus: the std will be computed as log(1+exp(x)).
        layer_normalization (bool): Bool for using layer normalization or not.
        normal_distribution_cls (torch.distribution): normal distribution class
            to be constructed and returned by a call to forward. By default, is
            `torch.distributions.Normal`.

    """

    def __init__(self, input_dim, output_dim, hidden_sizes=(32, 32), *, hidden_nonlinearity=torch.tanh, hidden_w_init=nn.init.xavier_uniform_, hidden_b_init=nn.init.zeros_, output_nonlinearity=None, output_w_init=nn.init.xavier_uniform_, output_b_init=nn.init.zeros_, learn_std=True, init_std=1.0, min_std=1e-06, max_std=None, std_hidden_sizes=(32, 32), std_hidden_nonlinearity=torch.tanh, std_hidden_w_init=nn.init.xavier_uniform_, std_hidden_b_init=nn.init.zeros_, std_output_nonlinearity=None, std_output_w_init=nn.init.xavier_uniform_, std_parameterization='exp', layer_normalization=False, normal_distribution_cls=Normal):
        super().__init__()
        self._input_dim = input_dim
        self._hidden_sizes = hidden_sizes
        self._action_dim = output_dim
        self._learn_std = learn_std
        self._std_hidden_sizes = std_hidden_sizes
        self._min_std = min_std
        self._max_std = max_std
        self._std_hidden_nonlinearity = std_hidden_nonlinearity
        self._std_hidden_w_init = std_hidden_w_init
        self._std_hidden_b_init = std_hidden_b_init
        self._std_output_nonlinearity = std_output_nonlinearity
        self._std_output_w_init = std_output_w_init
        self._std_parameterization = std_parameterization
        self._hidden_nonlinearity = hidden_nonlinearity
        self._hidden_w_init = hidden_w_init
        self._hidden_b_init = hidden_b_init
        self._output_nonlinearity = output_nonlinearity
        self._output_w_init = output_w_init
        self._output_b_init = output_b_init
        self._layer_normalization = layer_normalization
        self._norm_dist_class = normal_distribution_cls
        if self._std_parameterization not in ('exp', 'softplus'):
            raise NotImplementedError
        self._init_std = torch.Tensor([init_std]).log()
        log_std = torch.Tensor([init_std] * output_dim).log()
        if self._learn_std:
            self._log_std = torch.nn.Parameter(log_std)
        else:
            self._log_std = log_std
            self.register_buffer('log_std', self._log_std)
        self._min_std_param = self._max_std_param = None
        if min_std is not None:
            self._min_std_param = torch.Tensor([min_std]).log()
            self.register_buffer('min_std_param', self._min_std_param)
        if max_std is not None:
            self._max_std_param = torch.Tensor([max_std]).log()
            self.register_buffer('max_std_param', self._max_std_param)

    def to(self, *args, **kwargs):
        """Move the module to the specified device.

        Args:
            *args: args to pytorch to function.
            **kwargs: keyword args to pytorch to function.

        """
        super()
        buffers = dict(self.named_buffers())
        if not isinstance(self._log_std, torch.nn.Parameter):
            self._log_std = buffers['log_std']
        self._min_std_param = buffers['min_std_param']
        self._max_std_param = buffers['max_std_param']

    @abc.abstractmethod
    def _get_mean_and_log_std(self, *inputs):
        pass

    def forward(self, *inputs):
        """Forward method.

        Args:
            *inputs: Input to the module.

        Returns:
            torch.distributions.independent.Independent: Independent
                distribution.

        """
        mean, log_std_uncentered = self._get_mean_and_log_std(*inputs)
        if self._min_std_param or self._max_std_param:
            log_std_uncentered = log_std_uncentered.clamp(min=None if self._min_std_param is None else self._min_std_param.item(), max=None if self._max_std_param is None else self._max_std_param.item())
        if self._std_parameterization == 'exp':
            std = log_std_uncentered.exp()
        else:
            std = log_std_uncentered.exp().exp().add(1.0).log()
        dist = self._norm_dist_class(mean, std)
        if not isinstance(dist, TanhNormal):
            dist = Independent(dist, 1)
        return dist


class GaussianMLPModule(GaussianMLPBaseModule):
    """GaussianMLPModule that mean and std share the same network.

    Args:
        input_dim (int): Input dimension of the model.
        output_dim (int): Output dimension of the model.
        hidden_sizes (list[int]): Output dimension of dense layer(s) for
            the MLP for mean. For example, (32, 32) means the MLP consists
            of two hidden layers, each with 32 hidden units.
        hidden_nonlinearity (callable): Activation function for intermediate
            dense layer(s). It should return a torch.Tensor. Set it to
            None to maintain a linear activation.
        hidden_w_init (callable): Initializer function for the weight
            of intermediate dense layer(s). The function should return a
            torch.Tensor.
        hidden_b_init (callable): Initializer function for the bias
            of intermediate dense layer(s). The function should return a
            torch.Tensor.
        output_nonlinearity (callable): Activation function for output dense
            layer. It should return a torch.Tensor. Set it to None to
            maintain a linear activation.
        output_w_init (callable): Initializer function for the weight
            of output dense layer(s). The function should return a
            torch.Tensor.
        output_b_init (callable): Initializer function for the bias
            of output dense layer(s). The function should return a
            torch.Tensor.
        learn_std (bool): Is std trainable.
        init_std (float): Initial value for std.
            (plain value - not log or exponentiated).
        min_std (float): If not None, the std is at least the value of min_std,
            to avoid numerical issues (plain value - not log or exponentiated).
        max_std (float): If not None, the std is at most the value of max_std,
            to avoid numerical issues (plain value - not log or exponentiated).
        std_parameterization (str): How the std should be parametrized. There
            are two options:
            - exp: the logarithm of the std will be stored, and applied a
               exponential transformation
            - softplus: the std will be computed as log(1+exp(x))
        layer_normalization (bool): Bool for using layer normalization or not.
        normal_distribution_cls (torch.distribution): normal distribution class
            to be constructed and returned by a call to forward. By default, is
            `torch.distributions.Normal`.

    """

    def __init__(self, input_dim, output_dim, hidden_sizes=(32, 32), *, hidden_nonlinearity=torch.tanh, hidden_w_init=nn.init.xavier_uniform_, hidden_b_init=nn.init.zeros_, output_nonlinearity=None, output_w_init=nn.init.xavier_uniform_, output_b_init=nn.init.zeros_, learn_std=True, init_std=1.0, min_std=1e-06, max_std=None, std_parameterization='exp', layer_normalization=False, normal_distribution_cls=Normal):
        super().__init__(input_dim=input_dim, output_dim=output_dim, hidden_sizes=hidden_sizes, hidden_nonlinearity=hidden_nonlinearity, hidden_w_init=hidden_w_init, hidden_b_init=hidden_b_init, output_nonlinearity=output_nonlinearity, output_w_init=output_w_init, output_b_init=output_b_init, learn_std=learn_std, init_std=init_std, min_std=min_std, max_std=max_std, std_parameterization=std_parameterization, layer_normalization=layer_normalization, normal_distribution_cls=normal_distribution_cls)
        self._mean_module = MLPModule(input_dim=self._input_dim, output_dim=self._action_dim, hidden_sizes=self._hidden_sizes, hidden_nonlinearity=self._hidden_nonlinearity, hidden_w_init=self._hidden_w_init, hidden_b_init=self._hidden_b_init, output_nonlinearity=self._output_nonlinearity, output_w_init=self._output_w_init, output_b_init=self._output_b_init, layer_normalization=self._layer_normalization)

    def _get_mean_and_log_std(self, x):
        """Get mean and std of Gaussian distribution given inputs.

        Args:
            x: Input to the module.

        Returns:
            torch.Tensor: The mean of Gaussian distribution.
            torch.Tensor: The variance of Gaussian distribution.

        """
        mean = self._mean_module(x)
        return mean, self._log_std


class GaussianMLPIndependentStdModule(GaussianMLPBaseModule):
    """GaussianMLPModule which has two different mean and std network.

    Args:
        input_dim (int): Input dimension of the model.
        output_dim (int): Output dimension of the model.
        hidden_sizes (list[int]): Output dimension of dense layer(s) for
            the MLP for mean. For example, (32, 32) means the MLP consists
            of two hidden layers, each with 32 hidden units.
        hidden_nonlinearity (callable): Activation function for intermediate
            dense layer(s). It should return a torch.Tensor. Set it to
            None to maintain a linear activation.
        hidden_w_init (callable): Initializer function for the weight
            of intermediate dense layer(s). The function should return a
            torch.Tensor.
        hidden_b_init (callable): Initializer function for the bias
            of intermediate dense layer(s). The function should return a
            torch.Tensor.
        output_nonlinearity (callable): Activation function for output dense
            layer. It should return a torch.Tensor. Set it to None to
            maintain a linear activation.
        output_w_init (callable): Initializer function for the weight
            of output dense layer(s). The function should return a
            torch.Tensor.
        output_b_init (callable): Initializer function for the bias
            of output dense layer(s). The function should return a
            torch.Tensor.
        learn_std (bool): Is std trainable.
        init_std (float): Initial value for std.
            (plain value - not log or exponentiated).
        min_std (float): If not None, the std is at least the value of min_std,
            to avoid numerical issues (plain value - not log or exponentiated).
        max_std (float): If not None, the std is at most the value of max_std,
            to avoid numerical issues (plain value - not log or exponentiated).
        std_hidden_sizes (list[int]): Output dimension of dense layer(s) for
            the MLP for std. For example, (32, 32) means the MLP consists
            of two hidden layers, each with 32 hidden units.
        std_hidden_nonlinearity (callable): Nonlinearity for each hidden layer
            in the std network.
        std_hidden_w_init (callable):  Initializer function for the weight
            of hidden layer (s).
        std_hidden_b_init (callable): Initializer function for the bias
            of intermediate dense layer(s).
        std_output_nonlinearity (callable): Activation function for output
            dense layer in the std network. It should return a torch.Tensor.
            Set it to None to maintain a linear activation.
        std_output_w_init (callable): Initializer function for the weight
            of output dense layer(s) in the std network.
        std_parameterization (str): How the std should be parametrized. There
            are two options:
            - exp: the logarithm of the std will be stored, and applied a
               exponential transformation
            - softplus: the std will be computed as log(1+exp(x))
        layer_normalization (bool): Bool for using layer normalization or not.
        normal_distribution_cls (torch.distribution): normal distribution class
            to be constructed and returned by a call to forward. By default, is
            `torch.distributions.Normal`.

    """

    def __init__(self, input_dim, output_dim, hidden_sizes=(32, 32), *, hidden_nonlinearity=torch.tanh, hidden_w_init=nn.init.xavier_uniform_, hidden_b_init=nn.init.zeros_, output_nonlinearity=None, output_w_init=nn.init.xavier_uniform_, output_b_init=nn.init.zeros_, learn_std=True, init_std=1.0, min_std=1e-06, max_std=None, std_hidden_sizes=(32, 32), std_hidden_nonlinearity=torch.tanh, std_hidden_w_init=nn.init.xavier_uniform_, std_hidden_b_init=nn.init.zeros_, std_output_nonlinearity=None, std_output_w_init=nn.init.xavier_uniform_, std_parameterization='exp', layer_normalization=False, normal_distribution_cls=Normal):
        super().__init__(input_dim=input_dim, output_dim=output_dim, hidden_sizes=hidden_sizes, hidden_nonlinearity=hidden_nonlinearity, hidden_w_init=hidden_w_init, hidden_b_init=hidden_b_init, output_nonlinearity=output_nonlinearity, output_w_init=output_w_init, output_b_init=output_b_init, learn_std=learn_std, init_std=init_std, min_std=min_std, max_std=max_std, std_hidden_sizes=std_hidden_sizes, std_hidden_nonlinearity=std_hidden_nonlinearity, std_hidden_w_init=std_hidden_w_init, std_hidden_b_init=std_hidden_b_init, std_output_nonlinearity=std_output_nonlinearity, std_output_w_init=std_output_w_init, std_parameterization=std_parameterization, layer_normalization=layer_normalization, normal_distribution_cls=normal_distribution_cls)
        self._mean_module = MLPModule(input_dim=self._input_dim, output_dim=self._action_dim, hidden_sizes=self._hidden_sizes, hidden_nonlinearity=self._hidden_nonlinearity, hidden_w_init=self._hidden_w_init, hidden_b_init=self._hidden_b_init, output_nonlinearity=self._output_nonlinearity, output_w_init=self._output_w_init, output_b_init=self._output_b_init, layer_normalization=self._layer_normalization)
        self._log_std_module = MLPModule(input_dim=self._input_dim, output_dim=self._action_dim, hidden_sizes=self._std_hidden_sizes, hidden_nonlinearity=self._std_hidden_nonlinearity, hidden_w_init=self._std_hidden_w_init, hidden_b_init=self._std_hidden_b_init, output_nonlinearity=self._std_output_nonlinearity, output_w_init=self._std_output_w_init, output_b_init=self._init_std_b, layer_normalization=self._layer_normalization)

    def _init_std_b(self, b):
        """Default bias initialization function.

        Args:
            b (torch.Tensor): The bias tensor.

        Returns:
            torch.Tensor: The bias tensor itself.

        """
        return nn.init.constant_(b, self._init_std.item())

    def _get_mean_and_log_std(self, *inputs):
        """Get mean and std of Gaussian distribution given inputs.

        Args:
            *inputs: Input to the module.

        Returns:
            torch.Tensor: The mean of Gaussian distribution.
            torch.Tensor: The variance of Gaussian distribution.

        """
        return self._mean_module(*inputs), self._log_std_module(*inputs)


class GaussianMLPTwoHeadedModule(GaussianMLPBaseModule):
    """GaussianMLPModule which has only one mean network.

    Args:
        input_dim (int): Input dimension of the model.
        output_dim (int): Output dimension of the model.
        hidden_sizes (list[int]): Output dimension of dense layer(s) for
            the MLP for mean. For example, (32, 32) means the MLP consists
            of two hidden layers, each with 32 hidden units.
        hidden_nonlinearity (callable): Activation function for intermediate
            dense layer(s). It should return a torch.Tensor. Set it to
            None to maintain a linear activation.
        hidden_w_init (callable): Initializer function for the weight
            of intermediate dense layer(s). The function should return a
            torch.Tensor.
        hidden_b_init (callable): Initializer function for the bias
            of intermediate dense layer(s). The function should return a
            torch.Tensor.
        output_nonlinearity (callable): Activation function for output dense
            layer. It should return a torch.Tensor. Set it to None to
            maintain a linear activation.
        output_w_init (callable): Initializer function for the weight
            of output dense layer(s). The function should return a
            torch.Tensor.
        output_b_init (callable): Initializer function for the bias
            of output dense layer(s). The function should return a
            torch.Tensor.
        learn_std (bool): Is std trainable.
        init_std (float): Initial value for std.
            (plain value - not log or exponentiated).
        min_std (float): If not None, the std is at least the value of min_std,
            to avoid numerical issues (plain value - not log or exponentiated).
        max_std (float): If not None, the std is at most the value of max_std,
            to avoid numerical issues (plain value - not log or exponentiated).
        std_parameterization (str): How the std should be parametrized. There
            are two options:
            - exp: the logarithm of the std will be stored, and applied a
               exponential transformation
            - softplus: the std will be computed as log(1+exp(x))
        layer_normalization (bool): Bool for using layer normalization or not.
        normal_distribution_cls (torch.distribution): normal distribution class
            to be constructed and returned by a call to forward. By default, is
            `torch.distributions.Normal`.

    """

    def __init__(self, input_dim, output_dim, hidden_sizes=(32, 32), *, hidden_nonlinearity=torch.tanh, hidden_w_init=nn.init.xavier_uniform_, hidden_b_init=nn.init.zeros_, output_nonlinearity=None, output_w_init=nn.init.xavier_uniform_, output_b_init=nn.init.zeros_, learn_std=True, init_std=1.0, min_std=1e-06, max_std=None, std_parameterization='exp', layer_normalization=False, normal_distribution_cls=Normal):
        super().__init__(input_dim=input_dim, output_dim=output_dim, hidden_sizes=hidden_sizes, hidden_nonlinearity=hidden_nonlinearity, hidden_w_init=hidden_w_init, hidden_b_init=hidden_b_init, output_nonlinearity=output_nonlinearity, output_w_init=output_w_init, output_b_init=output_b_init, learn_std=learn_std, init_std=init_std, min_std=min_std, max_std=max_std, std_parameterization=std_parameterization, layer_normalization=layer_normalization, normal_distribution_cls=normal_distribution_cls)
        self._shared_mean_log_std_network = MultiHeadedMLPModule(n_heads=2, input_dim=self._input_dim, output_dims=self._action_dim, hidden_sizes=self._hidden_sizes, hidden_nonlinearity=self._hidden_nonlinearity, hidden_w_init=self._hidden_w_init, hidden_b_init=self._hidden_b_init, output_nonlinearities=self._output_nonlinearity, output_w_inits=self._output_w_init, output_b_inits=[nn.init.zeros_, lambda x: nn.init.constant_(x, self._init_std.item())], layer_normalization=self._layer_normalization)

    def _get_mean_and_log_std(self, *inputs):
        """Get mean and std of Gaussian distribution given inputs.

        Args:
            *inputs: Input to the module.

        Returns:
            torch.Tensor: The mean of Gaussian distribution.
            torch.Tensor: The variance of Gaussian distribution.

        """
        return self._shared_mean_log_std_network(*inputs)


def global_device():
    """Returns the global device that torch.Tensors should be placed on.

    Note: The global device is set by using the function
        `garage.torch._functions.set_gpu_mode.`
        If this functions is never called
        `garage.torch._functions.device()` returns None.

    Returns:
        `torch.Device`: The global device that newly created torch.Tensors
            should be placed on.

    """
    global _DEVICE
    return _DEVICE


def product_of_gaussians(mus, sigmas_squared):
    """Compute mu, sigma of product of gaussians.

    Args:
        mus (torch.Tensor): Means, with shape :math:`(N, M)`. M is the number
            of mean values.
        sigmas_squared (torch.Tensor): Variances, with shape :math:`(N, V)`. V
            is the number of variance values.

    Returns:
        torch.Tensor: Mu of product of gaussians, with shape :math:`(N, 1)`.
        torch.Tensor: Sigma of product of gaussians, with shape :math:`(N, 1)`.

    """
    sigmas_squared = torch.clamp(sigmas_squared, min=1e-07)
    sigma_squared = 1.0 / torch.sum(torch.reciprocal(sigmas_squared), dim=0)
    mu = sigma_squared * torch.sum(mus / sigmas_squared, dim=0)
    return mu, sigma_squared


class ContextConditionedPolicy(nn.Module):
    """A policy that outputs actions based on observation and latent context.

    In PEARL, policies are conditioned on current state and a latent context
    (adaptation data) variable Z. This inference network estimates the
    posterior probability of z given past transitions. It uses context
    information stored in the encoder to infer the probabilistic value of z and
    samples from a policy conditioned on z.

    Args:
        latent_dim (int): Latent context variable dimension.
        context_encoder (garage.torch.embeddings.ContextEncoder): Recurrent or
            permutation-invariant context encoder.
        policy (garage.torch.policies.Policy): Policy used to train the
            network.
        use_information_bottleneck (bool): True if latent context is not
            deterministic; false otherwise.
        use_next_obs (bool): True if next observation is used in context
            for distinguishing tasks; false otherwise.

    """

    def __init__(self, latent_dim, context_encoder, policy, use_information_bottleneck, use_next_obs):
        super().__init__()
        self._latent_dim = latent_dim
        self._context_encoder = context_encoder
        self._policy = policy
        self._use_information_bottleneck = use_information_bottleneck
        self._use_next_obs = use_next_obs
        self.register_buffer('z', torch.zeros(1, latent_dim))
        self.register_buffer('z_means', torch.zeros(1, latent_dim))
        self.register_buffer('z_vars', torch.zeros(1, latent_dim))
        self.reset_belief()

    def reset_belief(self, num_tasks=1):
        """Reset :math:`q(z \\| c)` to the prior and sample a new z from the prior.

        Args:
            num_tasks (int): Number of tasks.

        """
        mu = torch.zeros(num_tasks, self._latent_dim)
        if self._use_information_bottleneck:
            var = torch.ones(num_tasks, self._latent_dim)
        else:
            var = torch.zeros(num_tasks, self._latent_dim)
        self.z_means = mu
        self.z_vars = var
        self.sample_from_belief()
        self._context = None
        self._context_encoder.reset()

    def sample_from_belief(self):
        """Sample z using distributions from current means and variances."""
        if self._use_information_bottleneck:
            posteriors = [torch.distributions.Normal(m, torch.sqrt(s)) for m, s in zip(torch.unbind(self.z_means), torch.unbind(self.z_vars))]
            z = [d.rsample() for d in posteriors]
            self.z = torch.stack(z)
        else:
            self.z = self.z_means

    def update_context(self, timestep):
        """Append single transition to the current context.

        Args:
            timestep (garage._dtypes.TimeStep): Timestep containing transition
                information to be added to context.

        """
        o = torch.as_tensor(timestep.observation[None, None, ...], device=global_device()).float()
        a = torch.as_tensor(timestep.action[None, None, ...], device=global_device()).float()
        r = torch.as_tensor(np.array([timestep.reward])[None, None, ...], device=global_device()).float()
        no = torch.as_tensor(timestep.next_observation[None, None, ...], device=global_device()).float()
        if self._use_next_obs:
            data = torch.cat([o, a, r, no], dim=2)
        else:
            data = torch.cat([o, a, r], dim=2)
        if self._context is None:
            self._context = data
        else:
            self._context = torch.cat([self._context, data], dim=1)

    def infer_posterior(self, context):
        """Compute :math:`q(z \\| c)` as a function of input context and sample new z.

        Args:
            context (torch.Tensor): Context values, with shape
                :math:`(X, N, C)`. X is the number of tasks. N is batch size. C
                is the combined size of observation, action, reward, and next
                observation if next observation is used in context. Otherwise,
                C is the combined size of observation, action, and reward.

        """
        params = self._context_encoder.forward(context)
        params = params.view(context.size(0), -1, self._context_encoder.output_dim)
        if self._use_information_bottleneck:
            mu = params[..., :self._latent_dim]
            sigma_squared = F.softplus(params[..., self._latent_dim:])
            z_params = [product_of_gaussians(m, s) for m, s in zip(torch.unbind(mu), torch.unbind(sigma_squared))]
            self.z_means = torch.stack([p[0] for p in z_params])
            self.z_vars = torch.stack([p[1] for p in z_params])
        else:
            self.z_means = torch.mean(params, dim=1)
        self.sample_from_belief()

    def forward(self, obs, context):
        """Given observations and context, get actions and probs from policy.

        Args:
            obs (torch.Tensor): Observation values, with shape
                :math:`(X, N, O)`. X is the number of tasks. N is batch size. O
                 is the size of the flattened observation space.
            context (torch.Tensor): Context values, with shape
                :math:`(X, N, C)`. X is the number of tasks. N is batch size. C
                is the combined size of observation, action, reward, and next
                observation if next observation is used in context. Otherwise,
                C is the combined size of observation, action, and reward.

        Returns:
            tuple:
                * torch.Tensor: Predicted action values.
                * np.ndarray: Mean of distribution.
                * np.ndarray: Log std of distribution.
                * torch.Tensor: Log likelihood of distribution.
                * torch.Tensor: Sampled values from distribution before
                    applying tanh transformation.
            torch.Tensor: z values, with shape :math:`(N, L)`. N is batch size.
                L is the latent dimension.

        """
        self.infer_posterior(context)
        self.sample_from_belief()
        task_z = self.z
        t, b, _ = obs.size()
        obs = obs.view(t * b, -1)
        task_z = [z.repeat(b, 1) for z in task_z]
        task_z = torch.cat(task_z, dim=0)
        obs_z = torch.cat([obs, task_z.detach()], dim=1)
        dist = self._policy(obs_z)[0]
        pre_tanh, actions = dist.rsample_with_pre_tanh_value()
        log_pi = dist.log_prob(value=actions, pre_tanh_value=pre_tanh)
        log_pi = log_pi.unsqueeze(1)
        mean = dist.mean.detach().numpy()
        log_std = (dist.variance ** 0.5).log().detach().numpy()
        return (actions, mean, log_std, log_pi, pre_tanh), task_z

    def get_action(self, obs):
        """Sample action from the policy, conditioned on the task embedding.

        Args:
            obs (torch.Tensor): Observation values, with shape :math:`(1, O)`.
                O is the size of the flattened observation space.

        Returns:
            torch.Tensor: Output action value, with shape :math:`(1, A)`.
                A is the size of the flattened action space.
            dict:
                * np.ndarray[float]: Mean of the distribution.
                * np.ndarray[float]: Standard deviation of logarithmic values
                    of the distribution.

        """
        z = self.z
        obs = torch.as_tensor(obs[None], device=global_device()).float()
        obs_in = torch.cat([obs, z], dim=1)
        action, info = self._policy.get_action(obs_in)
        return action, info

    def compute_kl_div(self):
        """Compute :math:`KL(q(z|c) \\| p(z))`.

        Returns:
            float: :math:`KL(q(z|c) \\| p(z))`.

        """
        prior = torch.distributions.Normal(torch.zeros(self._latent_dim), torch.ones(self._latent_dim))
        posteriors = [torch.distributions.Normal(mu, torch.sqrt(var)) for mu, var in zip(torch.unbind(self.z_means), torch.unbind(self.z_vars))]
        kl_divs = [torch.distributions.kl.kl_divergence(post, prior) for post in posteriors]
        kl_div_sum = torch.sum(torch.stack(kl_divs))
        return kl_div_sum

    @property
    def networks(self):
        """Return context_encoder and policy.

        Returns:
            list: Encoder and policy networks.

        """
        return [self._context_encoder, self._policy]

    @property
    def context(self):
        """Return context.

        Returns:
            torch.Tensor: Context values, with shape :math:`(X, N, C)`.
                X is the number of tasks. N is batch size. C is the combined
                size of observation, action, reward, and next observation if
                next observation is used in context. Otherwise, C is the
                combined size of observation, action, and reward.

        """
        return self._context


def list_to_tensor(data):
    """Convert a list to a PyTorch tensor.

    Args:
        data (list): Data to convert to tensor

    Returns:
        torch.Tensor: A float tensor
    """
    return torch.as_tensor(data, dtype=torch.float32, device=global_device())


def np_to_torch(array):
    """Numpy arrays to PyTorch tensors.

    Args:
        array (np.ndarray): Data in numpy array.

    Returns:
        torch.Tensor: float tensor on the global device.

    """
    tensor = torch.from_numpy(array)
    if tensor.dtype != torch.float32:
        tensor = tensor.float()
    return tensor


class DiscreteCNNQFunction(nn.Module):
    """Discrete CNN Q Function.

    A Q network that estimates Q values of all possible discrete actions.
    It is constructed using a CNN followed by one or more fully-connected
    layers.

    Args:
        env_spec (EnvSpec): Environment specification.
        image_format (str): Either 'NCHW' or 'NHWC'. Should match the input
            specification. Gym uses NHWC by default, but PyTorch uses NCHW by
            default.
        kernel_sizes (tuple[int]): Dimension of the conv filters.
            For example, (3, 5) means there are two convolutional layers.
            The filter for first layer is of dimension (3 x 3)
            and the second one is of dimension (5 x 5).
        strides (tuple[int]): The stride of the sliding window. For example,
            (1, 2) means there are two convolutional layers. The stride of the
            filter for first layer is 1 and that of the second layer is 2.
        hidden_channels (tuple[int]): Number of output channels for CNN.
            For example, (3, 32) means there are two convolutional layers.
            The filter for the first conv layer outputs 3 channels
            and the second one outputs 32 channels.
        hidden_sizes (list[int]): Output dimension of dense layer(s) for
            the MLP for mean. For example, (32, 32) means the MLP consists
            of two hidden layers, each with 32 hidden units.
        mlp_hidden_nonlinearity (callable): Activation function for
            intermediate dense layer(s) in the MLP. It should return
            a torch.Tensor. Set it to None to maintain a linear activation.
        cnn_hidden_nonlinearity (callable): Activation function for
            intermediate CNN layer(s). It should return a torch.Tensor.
            Set it to None to maintain a linear activation.
        hidden_w_init (callable): Initializer function for the weight
            of intermediate dense layer(s). The function should return a
            torch.Tensor.
        hidden_b_init (callable): Initializer function for the bias
            of intermediate dense layer(s). The function should return a
            torch.Tensor.
        paddings (tuple[int]):  Zero-padding added to both sides of the input
        padding_mode (str): The type of padding algorithm to use,
            either 'SAME' or 'VALID'.
        max_pool (bool): Bool for using max-pooling or not.
        pool_shape (tuple[int]): Dimension of the pooling layer(s). For
            example, (2, 2) means that all the pooling layers are of the same
            shape (2, 2).
        pool_stride (tuple[int]): The strides of the pooling layer(s). For
            example, (2, 2) means that all the pooling layers have
            strides (2, 2).
        output_nonlinearity (callable): Activation function for output dense
            layer. It should return a torch.Tensor. Set it to None to
            maintain a linear activation.
        output_w_init (callable): Initializer function for the weight
            of output dense layer(s). The function should return a
            torch.Tensor.
        output_b_init (callable): Initializer function for the bias
            of output dense layer(s). The function should return a
            torch.Tensor.
        layer_normalization (bool): Bool for using layer normalization or not.
    """

    def __init__(self, env_spec, image_format, *, kernel_sizes, hidden_channels, strides, hidden_sizes=(32, 32), cnn_hidden_nonlinearity=torch.nn.ReLU, mlp_hidden_nonlinearity=torch.nn.ReLU, hidden_w_init=nn.init.xavier_uniform_, hidden_b_init=nn.init.zeros_, paddings=0, padding_mode='zeros', max_pool=False, pool_shape=None, pool_stride=1, output_nonlinearity=None, output_w_init=nn.init.xavier_uniform_, output_b_init=nn.init.zeros_, layer_normalization=False):
        super().__init__()
        self._env_spec = env_spec
        self._cnn_module = DiscreteCNNModule(spec=InOutSpec(input_space=env_spec.observation_space, output_space=env_spec.action_space), image_format=image_format, kernel_sizes=kernel_sizes, hidden_channels=hidden_channels, strides=strides, hidden_sizes=hidden_sizes, cnn_hidden_nonlinearity=cnn_hidden_nonlinearity, mlp_hidden_nonlinearity=mlp_hidden_nonlinearity, hidden_w_init=hidden_w_init, hidden_b_init=hidden_b_init, paddings=paddings, padding_mode=padding_mode, max_pool=max_pool, pool_shape=pool_shape, pool_stride=pool_stride, output_nonlinearity=output_nonlinearity, output_w_init=output_w_init, output_b_init=output_b_init, layer_normalization=layer_normalization)

    def forward(self, observations):
        """Return Q-value(s).

        Args:
            observations (np.ndarray): observations of shape :math: `(N, O*)`.

        Returns:
            torch.Tensor: Output value
        """
        observations = observations.reshape(-1, *self._env_spec.observation_space.shape)
        return self._cnn_module(observations)


class DiscreteDuelingCNNQFunction(nn.Module):
    """Discrete Dueling CNN Q Function.

    A dueling Q network that estimates Q values of all possible discrete
    actions. It is constructed using a CNN followed by one or more
    fully-connected layers for each the value portion and the advantage
    portion of the fully-connected layers.

    Args:
        env_spec (EnvSpec): Environment specification.
        image_format (str): Either 'NCHW' or 'NHWC'. Should match the input
            specification. Gym uses NHWC by default, but PyTorch uses NCHW by
            default.
        kernel_sizes (tuple[int]): Dimension of the conv filters.
            For example, (3, 5) means there are two convolutional layers.
            The filter for first layer is of dimension (3 x 3)
            and the second one is of dimension (5 x 5).
        strides (tuple[int]): The stride of the sliding window. For example,
            (1, 2) means there are two convolutional layers. The stride of the
            filter for first layer is 1 and that of the second layer is 2.
        hidden_channels (tuple[int]): Number of output channels for CNN.
            For example, (3, 32) means there are two convolutional layers.
            The filter for the first conv layer outputs 3 channels
            and the second one outputs 32 channels.
        hidden_sizes (list[int]): Output dimension of dense layer(s) for
            the MLP for mean. For example, (32, 32) means the MLP consists
            of two hidden layers, each with 32 hidden units.
        mlp_hidden_nonlinearity (callable): Activation function for
            intermediate dense layer(s) in the MLP. It should return
            a torch.Tensor. Set it to None to maintain a linear activation.
        cnn_hidden_nonlinearity (callable): Activation function for
            intermediate CNN layer(s). It should return a torch.Tensor.
            Set it to None to maintain a linear activation.
        hidden_w_init (callable): Initializer function for the weight
            of intermediate dense layer(s). The function should return a
            torch.Tensor.
        hidden_b_init (callable): Initializer function for the bias
            of intermediate dense layer(s). The function should return a
            torch.Tensor.
        paddings (tuple[int]):  Zero-padding added to both sides of the input
        padding_mode (str): The type of padding algorithm to use,
            either 'SAME' or 'VALID'.
        max_pool (bool): Bool for using max-pooling or not.
        pool_shape (tuple[int]): Dimension of the pooling layer(s). For
            example, (2, 2) means that all the pooling layers are of the same
            shape (2, 2).
        pool_stride (tuple[int]): The strides of the pooling layer(s). For
            example, (2, 2) means that all the pooling layers have
            strides (2, 2).
        output_nonlinearity (callable): Activation function for output dense
            layer. It should return a torch.Tensor. Set it to None to
            maintain a linear activation.
        output_w_init (callable): Initializer function for the weight
            of output dense layer(s). The function should return a
            torch.Tensor.
        output_b_init (callable): Initializer function for the bias
            of output dense layer(s). The function should return a
            torch.Tensor.
        layer_normalization (bool): Bool for using layer normalization or not.
    """

    def __init__(self, env_spec, image_format, *, kernel_sizes, hidden_channels, strides, hidden_sizes=(32, 32), cnn_hidden_nonlinearity=torch.nn.ReLU, mlp_hidden_nonlinearity=torch.nn.ReLU, hidden_w_init=nn.init.xavier_uniform_, hidden_b_init=nn.init.zeros_, paddings=0, padding_mode='zeros', max_pool=False, pool_shape=None, pool_stride=1, output_nonlinearity=None, output_w_init=nn.init.xavier_uniform_, output_b_init=nn.init.zeros_, layer_normalization=False):
        super().__init__()
        self._env_spec = env_spec
        cnn_spec = InOutSpec(input_space=env_spec.observation_space, output_space=None)
        cnn_module = CNNModule(spec=cnn_spec, image_format=image_format, kernel_sizes=kernel_sizes, strides=strides, hidden_w_init=hidden_w_init, hidden_b_init=hidden_b_init, hidden_channels=hidden_channels, hidden_nonlinearity=cnn_hidden_nonlinearity, paddings=paddings, padding_mode=padding_mode, max_pool=max_pool, layer_normalization=layer_normalization, pool_shape=pool_shape, pool_stride=pool_stride)
        flat_dim = cnn_module.spec.output_space.flat_dim
        self._val = MLPModule(flat_dim, 1, hidden_sizes, hidden_nonlinearity=mlp_hidden_nonlinearity, hidden_w_init=hidden_w_init, hidden_b_init=hidden_b_init, output_nonlinearity=output_nonlinearity, output_w_init=output_w_init, output_b_init=output_b_init, layer_normalization=layer_normalization)
        self._act = MLPModule(flat_dim, env_spec.action_space.flat_dim, hidden_sizes, hidden_nonlinearity=mlp_hidden_nonlinearity, hidden_w_init=hidden_w_init, hidden_b_init=hidden_b_init, output_nonlinearity=output_nonlinearity, output_w_init=output_w_init, output_b_init=output_b_init, layer_normalization=layer_normalization)
        if mlp_hidden_nonlinearity is None:
            self._module = nn.Sequential(cnn_module, nn.Flatten())
        else:
            self._module = nn.Sequential(cnn_module, mlp_hidden_nonlinearity(), nn.Flatten())

    def forward(self, observations):
        """Return Q-value(s).

        Args:
            observations (np.ndarray): observations of shape :math: `(N, O*)`.

        Returns:
            torch.Tensor: Output value
        """
        observations = observations.reshape(-1, *self._env_spec.observation_space.shape)
        out = self._module(observations)
        val = self._val(out)
        act = self._act(out)
        act = act - act.mean(1).unsqueeze(1)
        return val + act


class ValueFunction(abc.ABC, nn.Module):
    """Base class for all baselines.

    Args:
        env_spec (EnvSpec): Environment specification.
        name (str): Value function name, also the variable scope.

    """

    def __init__(self, env_spec, name):
        super(ValueFunction, self).__init__()
        self._mdp_spec = env_spec
        self.name = name

    @abc.abstractmethod
    def compute_loss(self, obs, returns):
        """Compute mean value of loss.

        Args:
            obs (torch.Tensor): Observation from the environment
                with shape :math:`(N \\dot [T], O*)`.
            returns (torch.Tensor): Acquired returns with shape :math:`(N, )`.

        Returns:
            torch.Tensor: Calculated negative mean scalar value of
                objective (float).

        """


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (MLPModule,
     lambda: ([], {'input_dim': 4, 'output_dim': 4, 'hidden_sizes': [4, 4]}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (MultiHeadedMLPModule,
     lambda: ([], {'n_heads': 4, 'input_dim': 4, 'output_dims': 4, 'hidden_sizes': [4, 4]}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
]

