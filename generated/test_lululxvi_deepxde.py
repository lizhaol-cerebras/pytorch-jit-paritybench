
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


import time


import numpy as np


import random


import abc


from typing import Literal


from typing import Union


from scipy import spatial


import itertools


from scipy import stats


from sklearn import preprocessing


from abc import ABC


from abc import abstractmethod


from collections import OrderedDict


import math


from typing import Tuple


import matplotlib.pyplot as plt


from scipy.io import loadmat


from scipy import integrate


def sum(input_tensor, dim, keepdims=False):
    return torch.sum(input_tensor, dim, keepdim=keepdims)


class NN(torch.nn.Module):
    """Base class for all neural network modules."""

    def __init__(self):
        super().__init__()
        self.regularizer = None
        self._auxiliary_vars = None
        self._input_transform = None
        self._output_transform = None

    @property
    def auxiliary_vars(self):
        """Tensors: Any additional variables needed."""
        return self._auxiliary_vars

    @auxiliary_vars.setter
    def auxiliary_vars(self, value):
        self._auxiliary_vars = value

    def apply_feature_transform(self, transform):
        """Compute the features by appling a transform to the network inputs, i.e.,
        features = transform(inputs). Then, outputs = network(features).
        """
        self._input_transform = transform

    def apply_output_transform(self, transform):
        """Apply a transform to the network outputs, i.e.,
        outputs = transform(inputs, outputs).
        """
        self._output_transform = transform

    def num_trainable_parameters(self):
        """Evaluate the number of trainable parameters for the NN."""
        return sum(v.numel() for v in self.parameters() if v.requires_grad)


class FNN(NN):
    """Fully-connected neural network."""

    def __init__(self, layer_sizes, activation, kernel_initializer, regularization=None):
        super().__init__()
        if isinstance(activation, list):
            if not len(layer_sizes) - 1 == len(activation):
                raise ValueError('Total number of activation functions do not match with sum of hidden layers and output layer!')
            self.activation = list(map(activations.get, activation))
        else:
            self.activation = activations.get(activation)
        initializer = initializers.get(kernel_initializer)
        initializer_zero = initializers.get('zeros')
        self.regularizer = regularization
        self.linears = torch.nn.ModuleList()
        for i in range(1, len(layer_sizes)):
            self.linears.append(torch.nn.Linear(layer_sizes[i - 1], layer_sizes[i], dtype=config.real(torch)))
            initializer(self.linears[-1].weight)
            initializer_zero(self.linears[-1].bias)

    def forward(self, inputs):
        x = inputs
        if self._input_transform is not None:
            x = self._input_transform(x)
        for j, linear in enumerate(self.linears[:-1]):
            x = self.activation[j](linear(x)) if isinstance(self.activation, list) else self.activation(linear(x))
        x = self.linears[-1](x)
        if self._output_transform is not None:
            x = self._output_transform(inputs, x)
        return x


class DeepONetStrategy(ABC):
    """DeepONet building strategy.

    See the section 3.1.6. in
    L. Lu, X. Meng, S. Cai, Z. Mao, S. Goswami, Z. Zhang, & G. Karniadakis.
    A comprehensive and fair comparison of two neural operators
    (with practical extensions) based on FAIR data.
    Computer Methods in Applied Mechanics and Engineering, 393, 114778, 2022.
    """

    def __init__(self, net):
        self.net = net

    @abstractmethod
    def build(self, layer_sizes_branch, layer_sizes_trunk):
        """Build branch and trunk nets."""

    @abstractmethod
    def call(self, x_func, x_loc):
        """Forward pass."""


class SingleOutputStrategy(DeepONetStrategy):
    """Single output build strategy is the standard build method."""

    def build(self, layer_sizes_branch, layer_sizes_trunk):
        branch = self.net.build_branch_net(layer_sizes_branch)
        trunk = self.net.build_trunk_net(layer_sizes_trunk)
        return branch, trunk

    def call(self, x_func, x_loc):
        x_func = self.net.branch(x_func)
        x_loc = self.net.activation_trunk(self.net.trunk(x_loc))
        if x_func.shape[-1] != x_loc.shape[-1]:
            raise AssertionError('Output sizes of branch net and trunk net do not match.')
        x = self.net.merge_branch_trunk(x_func, x_loc, 0)
        return x


class IndependentStrategy(DeepONetStrategy):
    """Directly use n independent DeepONets,
    and each DeepONet outputs only one function.
    """

    def build(self, layer_sizes_branch, layer_sizes_trunk):
        single_output_strategy = SingleOutputStrategy(self.net)
        branch, trunk = [], []
        for _ in range(self.net.num_outputs):
            branch_, trunk_ = single_output_strategy.build(layer_sizes_branch, layer_sizes_trunk)
            branch.append(branch_)
            trunk.append(trunk_)
        return branch, trunk

    def call(self, x_func, x_loc):
        xs = []
        for i in range(self.net.num_outputs):
            x_func_ = self.net.branch[i](x_func)
            x_loc_ = self.net.activation_trunk(self.net.trunk[i](x_loc))
            x = self.net.merge_branch_trunk(x_func_, x_loc_, i)
            xs.append(x)
        return self.net.concatenate_outputs(xs)


class SplitBothStrategy(DeepONetStrategy):
    """Split the outputs of both the branch net and the trunk net into n groups,
    and then the kth group outputs the kth solution.

    For example, if n = 2 and both the branch and trunk nets have 100 output neurons,
    then the dot product between the first 50 neurons of
    the branch and trunk nets generates the first function,
    and the remaining 50 neurons generate the second function.
    """

    def build(self, layer_sizes_branch, layer_sizes_trunk):
        if layer_sizes_branch[-1] != layer_sizes_trunk[-1]:
            raise AssertionError('Output sizes of branch net and trunk net do not match.')
        if layer_sizes_branch[-1] % self.net.num_outputs != 0:
            raise AssertionError(f'Output size of the branch net is not evenly divisible by {self.net.num_outputs}.')
        single_output_strategy = SingleOutputStrategy(self.net)
        return single_output_strategy.build(layer_sizes_branch, layer_sizes_trunk)

    def call(self, x_func, x_loc):
        x_func = self.net.branch(x_func)
        x_loc = self.net.activation_trunk(self.net.trunk(x_loc))
        shift = 0
        size = x_func.shape[1] // self.net.num_outputs
        xs = []
        for i in range(self.net.num_outputs):
            x_func_ = x_func[:, shift:shift + size]
            x_loc_ = x_loc[:, shift:shift + size]
            x = self.net.merge_branch_trunk(x_func_, x_loc_, i)
            xs.append(x)
            shift += size
        return self.net.concatenate_outputs(xs)


class SplitBranchStrategy(DeepONetStrategy):
    """Split the branch net and share the trunk net."""

    def build(self, layer_sizes_branch, layer_sizes_trunk):
        if layer_sizes_branch[-1] % self.net.num_outputs != 0:
            raise AssertionError(f'Output size of the branch net is not evenly divisible by {self.net.num_outputs}.')
        if layer_sizes_branch[-1] / self.net.num_outputs != layer_sizes_trunk[-1]:
            raise AssertionError(f'Output size of the trunk net does not equal to {layer_sizes_branch[-1] // self.net.num_outputs}.')
        return self.net.build_branch_net(layer_sizes_branch), self.net.build_trunk_net(layer_sizes_trunk)

    def call(self, x_func, x_loc):
        x_func = self.net.branch(x_func)
        x_loc = self.net.activation_trunk(self.net.trunk(x_loc))
        shift = 0
        size = x_loc.shape[1]
        xs = []
        for i in range(self.net.num_outputs):
            x_func_ = x_func[:, shift:shift + size]
            x = self.net.merge_branch_trunk(x_func_, x_loc, i)
            xs.append(x)
            shift += size
        return self.net.concatenate_outputs(xs)


class SplitTrunkStrategy(DeepONetStrategy):
    """Split the trunk net and share the branch net."""

    def build(self, layer_sizes_branch, layer_sizes_trunk):
        if layer_sizes_trunk[-1] % self.net.num_outputs != 0:
            raise AssertionError(f'Output size of the trunk net is not evenly divisible by {self.net.num_outputs}.')
        if layer_sizes_trunk[-1] / self.net.num_outputs != layer_sizes_branch[-1]:
            raise AssertionError(f'Output size of the branch net does not equal to {layer_sizes_trunk[-1] // self.net.num_outputs}.')
        return self.net.build_branch_net(layer_sizes_branch), self.net.build_trunk_net(layer_sizes_trunk)

    def call(self, x_func, x_loc):
        x_func = self.net.branch(x_func)
        x_loc = self.net.activation_trunk(self.net.trunk(x_loc))
        shift = 0
        size = x_func.shape[1]
        xs = []
        for i in range(self.net.num_outputs):
            x_loc_ = x_loc[:, shift:shift + size]
            x = self.net.merge_branch_trunk(x_func, x_loc_, i)
            xs.append(x)
            shift += size
        return self.net.concatenate_outputs(xs)


class DeepONet(NN):
    """Deep operator network.

    `Lu et al. Learning nonlinear operators via DeepONet based on the universal
    approximation theorem of operators. Nat Mach Intell, 2021.
    <https://doi.org/10.1038/s42256-021-00302-5>`_

    Args:
        layer_sizes_branch: A list of integers as the width of a fully connected network,
            or `(dim, f)` where `dim` is the input dimension and `f` is a network
            function. The width of the last layer in the branch and trunk net
            should be the same for all strategies except "split_branch" and "split_trunk".
        layer_sizes_trunk (list): A list of integers as the width of a fully connected
            network.
        activation: If `activation` is a ``string``, then the same activation is used in
            both trunk and branch nets. If `activation` is a ``dict``, then the trunk
            net uses the activation `activation["trunk"]`, and the branch net uses
            `activation["branch"]`.
        num_outputs (integer): Number of outputs. In case of multiple outputs, i.e., `num_outputs` > 1,
            `multi_output_strategy` below should be set.
        multi_output_strategy (str or None): ``None``, "independent", "split_both", "split_branch" or
            "split_trunk". It makes sense to set in case of multiple outputs.

            - None
            Classical implementation of DeepONet with a single output.
            Cannot be used with `num_outputs` > 1.

            - independent
            Use `num_outputs` independent DeepONets, and each DeepONet outputs only
            one function.

            - split_both
            Split the outputs of both the branch net and the trunk net into `num_outputs`
            groups, and then the kth group outputs the kth solution.

            - split_branch
            Split the branch net and share the trunk net. The width of the last layer
            in the branch net should be equal to the one in the trunk net multiplied
            by the number of outputs.

            - split_trunk
            Split the trunk net and share the branch net. The width of the last layer
            in the trunk net should be equal to the one in the branch net multiplied
            by the number of outputs.
    """

    def __init__(self, layer_sizes_branch, layer_sizes_trunk, activation, kernel_initializer, num_outputs=1, multi_output_strategy=None):
        super().__init__()
        if isinstance(activation, dict):
            self.activation_branch = activation['branch']
            self.activation_trunk = activations.get(activation['trunk'])
        else:
            self.activation_branch = self.activation_trunk = activations.get(activation)
        self.kernel_initializer = kernel_initializer
        self.num_outputs = num_outputs
        if self.num_outputs == 1:
            if multi_output_strategy is not None:
                raise ValueError('num_outputs is set to 1, but multi_output_strategy is not None.')
        elif multi_output_strategy is None:
            multi_output_strategy = 'independent'
            None
        self.multi_output_strategy = {None: SingleOutputStrategy, 'independent': IndependentStrategy, 'split_both': SplitBothStrategy, 'split_branch': SplitBranchStrategy, 'split_trunk': SplitTrunkStrategy}[multi_output_strategy](self)
        self.branch, self.trunk = self.multi_output_strategy.build(layer_sizes_branch, layer_sizes_trunk)
        if isinstance(self.branch, list):
            self.branch = torch.nn.ModuleList(self.branch)
        if isinstance(self.trunk, list):
            self.trunk = torch.nn.ModuleList(self.trunk)
        self.b = torch.nn.ParameterList([torch.nn.Parameter(torch.tensor(0.0)) for _ in range(self.num_outputs)])

    def build_branch_net(self, layer_sizes_branch):
        if callable(layer_sizes_branch[1]):
            return layer_sizes_branch[1]
        return FNN(layer_sizes_branch, self.activation_branch, self.kernel_initializer)

    def build_trunk_net(self, layer_sizes_trunk):
        return FNN(layer_sizes_trunk, self.activation_trunk, self.kernel_initializer)

    def merge_branch_trunk(self, x_func, x_loc, index):
        y = torch.einsum('bi,bi->b', x_func, x_loc)
        y = torch.unsqueeze(y, dim=1)
        y += self.b[index]
        return y

    @staticmethod
    def concatenate_outputs(ys):
        return torch.concat(ys, dim=1)

    def forward(self, inputs):
        x_func = inputs[0]
        x_loc = inputs[1]
        if self._input_transform is not None:
            x_loc = self._input_transform(x_loc)
        x = self.multi_output_strategy.call(x_func, x_loc)
        if self._output_transform is not None:
            x = self._output_transform(inputs, x)
        return x


class DeepONetCartesianProd(NN):
    """Deep operator network for dataset in the format of Cartesian product.

    Args:
        layer_sizes_branch: A list of integers as the width of a fully connected network,
            or `(dim, f)` where `dim` is the input dimension and `f` is a network
            function. The width of the last layer in the branch and trunk net
            should be the same for all strategies except "split_branch" and "split_trunk".
        layer_sizes_trunk (list): A list of integers as the width of a fully connected
            network.
        activation: If `activation` is a ``string``, then the same activation is used in
            both trunk and branch nets. If `activation` is a ``dict``, then the trunk
            net uses the activation `activation["trunk"]`, and the branch net uses
            `activation["branch"]`.
        num_outputs (integer): Number of outputs. In case of multiple outputs, i.e., `num_outputs` > 1,
            `multi_output_strategy` below should be set.
        multi_output_strategy (str or None): ``None``, "independent", "split_both", "split_branch" or
            "split_trunk". It makes sense to set in case of multiple outputs.

            - None
            Classical implementation of DeepONet with a single output.
            Cannot be used with `num_outputs` > 1.

            - independent
            Use `num_outputs` independent DeepONets, and each DeepONet outputs only
            one function.

            - split_both
            Split the outputs of both the branch net and the trunk net into `num_outputs`
            groups, and then the kth group outputs the kth solution.

            - split_branch
            Split the branch net and share the trunk net. The width of the last layer
            in the branch net should be equal to the one in the trunk net multiplied
            by the number of outputs.

            - split_trunk
            Split the trunk net and share the branch net. The width of the last layer
            in the trunk net should be equal to the one in the branch net multiplied
            by the number of outputs.
    """

    def __init__(self, layer_sizes_branch, layer_sizes_trunk, activation, kernel_initializer, num_outputs=1, multi_output_strategy=None):
        super().__init__()
        if isinstance(activation, dict):
            self.activation_branch = activation['branch']
            self.activation_trunk = activations.get(activation['trunk'])
        else:
            self.activation_branch = self.activation_trunk = activations.get(activation)
        self.kernel_initializer = kernel_initializer
        self.num_outputs = num_outputs
        if self.num_outputs == 1:
            if multi_output_strategy is not None:
                raise ValueError('num_outputs is set to 1, but multi_output_strategy is not None.')
        elif multi_output_strategy is None:
            multi_output_strategy = 'independent'
            None
        self.multi_output_strategy = {None: SingleOutputStrategy, 'independent': IndependentStrategy, 'split_both': SplitBothStrategy, 'split_branch': SplitBranchStrategy, 'split_trunk': SplitTrunkStrategy}[multi_output_strategy](self)
        self.branch, self.trunk = self.multi_output_strategy.build(layer_sizes_branch, layer_sizes_trunk)
        if isinstance(self.branch, list):
            self.branch = torch.nn.ModuleList(self.branch)
        if isinstance(self.trunk, list):
            self.trunk = torch.nn.ModuleList(self.trunk)
        self.b = torch.nn.ParameterList([torch.nn.Parameter(torch.tensor(0.0)) for _ in range(self.num_outputs)])

    def build_branch_net(self, layer_sizes_branch):
        if callable(layer_sizes_branch[1]):
            return layer_sizes_branch[1]
        return FNN(layer_sizes_branch, self.activation_branch, self.kernel_initializer)

    def build_trunk_net(self, layer_sizes_trunk):
        return FNN(layer_sizes_trunk, self.activation_trunk, self.kernel_initializer)

    def merge_branch_trunk(self, x_func, x_loc, index):
        y = torch.einsum('bi,ni->bn', x_func, x_loc)
        y += self.b[index]
        return y

    @staticmethod
    def concatenate_outputs(ys):
        return torch.stack(ys, dim=2)

    def forward(self, inputs):
        x_func = inputs[0]
        x_loc = inputs[1]
        if self._input_transform is not None:
            x_loc = self._input_transform(x_loc)
        x = self.multi_output_strategy.call(x_func, x_loc)
        if self._output_transform is not None:
            x = self._output_transform(inputs, x)
        return x


class PODDeepONet(NN):
    """Deep operator network with proper orthogonal decomposition (POD) for dataset in
    the format of Cartesian product.

    Args:
        pod_basis: POD basis used in the trunk net.
        layer_sizes_branch: A list of integers as the width of a fully connected network,
            or `(dim, f)` where `dim` is the input dimension and `f` is a network
            function. The width of the last layer in the branch and trunk net should be
            equal.
        activation: If `activation` is a ``string``, then the same activation is used in
            both trunk and branch nets. If `activation` is a ``dict``, then the trunk
            net uses the activation `activation["trunk"]`, and the branch net uses
            `activation["branch"]`.
        layer_sizes_trunk (list): A list of integers as the width of a fully connected
            network. If ``None``, then only use POD basis as the trunk net.

    References:
        `L. Lu, X. Meng, S. Cai, Z. Mao, S. Goswami, Z. Zhang, & G. E. Karniadakis. A
        comprehensive and fair comparison of two neural operators (with practical
        extensions) based on FAIR data. arXiv preprint arXiv:2111.05512, 2021
        <https://arxiv.org/abs/2111.05512>`_.
    """

    def __init__(self, pod_basis, layer_sizes_branch, activation, kernel_initializer, layer_sizes_trunk=None, regularization=None):
        super().__init__()
        self.regularization = regularization
        self.pod_basis = torch.as_tensor(pod_basis, dtype=torch.float32)
        if isinstance(activation, dict):
            activation_branch = activation['branch']
            self.activation_trunk = activations.get(activation['trunk'])
        else:
            activation_branch = self.activation_trunk = activations.get(activation)
        if callable(layer_sizes_branch[1]):
            self.branch = layer_sizes_branch[1]
        else:
            self.branch = FNN(layer_sizes_branch, activation_branch, kernel_initializer)
        self.trunk = None
        if layer_sizes_trunk is not None:
            self.trunk = FNN(layer_sizes_trunk, self.activation_trunk, kernel_initializer)
            self.b = torch.nn.parameter.Parameter(torch.tensor(0.0))

    def forward(self, inputs):
        x_func = inputs[0]
        x_loc = inputs[1]
        x_func = self.branch(x_func)
        if self.trunk is None:
            x = torch.einsum('bi,ni->bn', x_func, self.pod_basis)
        else:
            x_loc = self.activation_trunk(self.trunk(x_loc))
            x = torch.einsum('bi,ni->bn', x_func, torch.concat((self.pod_basis, x_loc), 1))
            x += self.b
        if self._output_transform is not None:
            x = self._output_transform(inputs, x)
        return x


class PFNN(NN):
    """Parallel fully-connected network that uses independent sub-networks for each
    network output.

    Args:
        layer_sizes: A nested list that defines the architecture of the neural network
            (how the layers are connected). If `layer_sizes[i]` is an int, it represents
            one layer shared by all the outputs; if `layer_sizes[i]` is a list, it
            represents `len(layer_sizes[i])` sub-layers, each of which is exclusively
            used by one output. Note that `len(layer_sizes[i])` should equal the number
            of outputs. Every number specifies the number of neurons in that layer.
    """

    def __init__(self, layer_sizes, activation, kernel_initializer):
        super().__init__()
        self.activation = activations.get(activation)
        initializer = initializers.get(kernel_initializer)
        initializer_zero = initializers.get('zeros')
        if len(layer_sizes) <= 1:
            raise ValueError('must specify input and output sizes')
        if not isinstance(layer_sizes[0], int):
            raise ValueError('input size must be integer')
        if not isinstance(layer_sizes[-1], int):
            raise ValueError('output size must be integer')
        n_output = layer_sizes[-1]

        def make_linear(n_input, n_output):
            linear = torch.nn.Linear(n_input, n_output, dtype=config.real(torch))
            initializer(linear.weight)
            initializer_zero(linear.bias)
            return linear
        self.layers = torch.nn.ModuleList()
        for i in range(1, len(layer_sizes) - 1):
            prev_layer_size = layer_sizes[i - 1]
            curr_layer_size = layer_sizes[i]
            if isinstance(curr_layer_size, (list, tuple)):
                if len(curr_layer_size) != n_output:
                    raise ValueError('number of sub-layers should equal number of network outputs')
                if isinstance(prev_layer_size, (list, tuple)):
                    self.layers.append(torch.nn.ModuleList([make_linear(prev_layer_size[j], curr_layer_size[j]) for j in range(n_output)]))
                else:
                    self.layers.append(torch.nn.ModuleList([make_linear(prev_layer_size, curr_layer_size[j]) for j in range(n_output)]))
            else:
                if not isinstance(prev_layer_size, int):
                    raise ValueError('cannot rejoin parallel subnetworks after splitting')
                self.layers.append(make_linear(prev_layer_size, curr_layer_size))
        if isinstance(layer_sizes[-2], (list, tuple)):
            self.layers.append(torch.nn.ModuleList([make_linear(layer_sizes[-2][j], 1) for j in range(n_output)]))
        else:
            self.layers.append(make_linear(layer_sizes[-2], n_output))

    def forward(self, inputs):
        x = inputs
        if self._input_transform is not None:
            x = self._input_transform(x)
        for layer in self.layers[:-1]:
            if isinstance(layer, torch.nn.ModuleList):
                if isinstance(x, list):
                    x = [self.activation(f(x_)) for f, x_ in zip(layer, x)]
                else:
                    x = [self.activation(f(x)) for f in layer]
            else:
                x = self.activation(layer(x))
        if isinstance(x, list):
            x = torch.cat([f(x_) for f, x_ in zip(self.layers[-1], x)], dim=1)
        else:
            x = self.layers[-1](x)
        if self._output_transform is not None:
            x = self._output_transform(inputs, x)
        return x


class MIONetCartesianProd(NN):
    """MIONet with two input functions for Cartesian product format."""

    def __init__(self, layer_sizes_branch1, layer_sizes_branch2, layer_sizes_trunk, activation, kernel_initializer, regularization=None, trunk_last_activation=False, merge_operation='mul', layer_sizes_merger=None, output_merge_operation='mul', layer_sizes_output_merger=None):
        super().__init__()
        if isinstance(activation, dict):
            self.activation_branch1 = activations.get(activation['branch1'])
            self.activation_branch2 = activations.get(activation['branch2'])
            self.activation_trunk = activations.get(activation['trunk'])
        else:
            self.activation_branch1 = self.activation_branch2 = self.activation_trunk = activations.get(activation)
        if callable(layer_sizes_branch1[1]):
            self.branch1 = layer_sizes_branch1[1]
        else:
            self.branch1 = FNN(layer_sizes_branch1, self.activation_branch1, kernel_initializer)
        if callable(layer_sizes_branch2[1]):
            self.branch2 = layer_sizes_branch2[1]
        else:
            self.branch2 = FNN(layer_sizes_branch2, self.activation_branch2, kernel_initializer)
        if layer_sizes_merger is not None:
            self.activation_merger = activations.get(activation['merger'])
            if callable(layer_sizes_merger[1]):
                self.merger = layer_sizes_merger[1]
            else:
                self.merger = FNN(layer_sizes_merger, self.activation_merger, kernel_initializer)
        else:
            self.merger = None
        if layer_sizes_output_merger is not None:
            self.activation_output_merger = activations.get(activation['output merger'])
            if callable(layer_sizes_output_merger[1]):
                self.output_merger = layer_sizes_output_merger[1]
            else:
                self.output_merger = FNN(layer_sizes_output_merger, self.activation_output_merger, kernel_initializer)
        else:
            self.output_merger = None
        self.trunk = FNN(layer_sizes_trunk, self.activation_trunk, kernel_initializer)
        self.b = torch.nn.parameter.Parameter(torch.tensor(0.0))
        self.regularizer = regularization
        self.trunk_last_activation = trunk_last_activation
        self.merge_operation = merge_operation
        self.output_merge_operation = output_merge_operation

    def forward(self, inputs):
        x_func1 = inputs[0]
        x_func2 = inputs[1]
        x_loc = inputs[2]
        y_func1 = self.branch1(x_func1)
        y_func2 = self.branch2(x_func2)
        if self.merge_operation == 'cat':
            x_merger = torch.cat((y_func1, y_func2), 1)
        else:
            if y_func1.shape[-1] != y_func2.shape[-1]:
                raise AssertionError('Output sizes of branch1 net and branch2 net do not match.')
            if self.merge_operation == 'add':
                x_merger = y_func1 + y_func2
            elif self.merge_operation == 'mul':
                x_merger = torch.mul(y_func1, y_func2)
            else:
                raise NotImplementedError(f'{self.merge_operation} operation to be implimented')
        if self.merger is not None:
            y_func = self.merger(x_merger)
        else:
            y_func = x_merger
        if self._input_transform is not None:
            x_loc = self._input_transform(x_loc)
        y_loc = self.trunk(x_loc)
        if self.trunk_last_activation:
            y_loc = self.activation_trunk(y_loc)
        if y_func.shape[-1] != y_loc.shape[-1]:
            raise AssertionError('Output sizes of merger net and trunk net do not match.')
        if self.output_merger is None:
            y = torch.einsum('ip,jp->ij', y_func, y_loc)
        else:
            y_func = y_func[:, None, :]
            y_loc = y_loc[None, :]
            if self.output_merge_operation == 'mul':
                y = torch.mul(y_func, y_loc)
            elif self.output_merge_operation == 'add':
                y = y_func + y_loc
            elif self.output_merge_operation == 'cat':
                y_func = y_func.repeat(1, y_loc.shape[1], 1)
                y_loc = y_loc.repeat(y_func.shape[0], 1, 1)
                y = torch.cat((y_func, y_loc), dim=2)
            shape0 = y.shape[0]
            shape1 = y.shape[1]
            y = y.reshape(shape0 * shape1, -1)
            y = self.output_merger(y)
            y = y.reshape(shape0, shape1)
        y += self.b
        if self._output_transform is not None:
            y = self._output_transform(inputs, y)
        return y


class PODMIONet(NN):
    """MIONet with two input functions and proper orthogonal decomposition (POD)
    for Cartesian product format."""

    def __init__(self, pod_basis, layer_sizes_branch1, layer_sizes_branch2, activation, kernel_initializer, layer_sizes_trunk=None, regularization=None, trunk_last_activation=False, merge_operation='mul', layer_sizes_merger=None):
        super().__init__()
        if isinstance(activation, dict):
            self.activation_branch1 = activations.get(activation['branch1'])
            self.activation_branch2 = activations.get(activation['branch2'])
            self.activation_trunk = activations.get(activation['trunk'])
            self.activation_merger = activations.get(activation['merger'])
        else:
            self.activation_branch1 = self.activation_branch2 = self.activation_trunk = activations.get(activation)
        self.pod_basis = torch.as_tensor(pod_basis, dtype=torch.float32)
        if callable(layer_sizes_branch1[1]):
            self.branch1 = layer_sizes_branch1[1]
        else:
            self.branch1 = FNN(layer_sizes_branch1, self.activation_branch1, kernel_initializer)
        if callable(layer_sizes_branch2[1]):
            self.branch2 = layer_sizes_branch2[1]
        else:
            self.branch2 = FNN(layer_sizes_branch2, self.activation_branch2, kernel_initializer)
        if layer_sizes_merger is not None:
            if callable(layer_sizes_merger[1]):
                self.merger = layer_sizes_merger[1]
            else:
                self.merger = FNN(layer_sizes_merger, self.activation_merger, kernel_initializer)
        else:
            self.merger = None
        self.trunk = None
        if layer_sizes_trunk is not None:
            self.trunk = FNN(layer_sizes_trunk, self.activation_trunk, kernel_initializer)
            self.b = torch.nn.parameter.Parameter(torch.tensor(0.0))
        self.regularizer = regularization
        self.trunk_last_activation = trunk_last_activation
        self.merge_operation = merge_operation

    def forward(self, inputs):
        x_func1 = inputs[0]
        x_func2 = inputs[1]
        x_loc = inputs[2]
        y_func1 = self.branch1(x_func1)
        y_func2 = self.branch2(x_func2)
        if self.merge_operation == 'cat':
            x_merger = torch.cat((y_func1, y_func2), 1)
        else:
            if y_func1.shape[-1] != y_func2.shape[-1]:
                raise AssertionError('Output sizes of branch1 net and branch2 net do not match.')
            if self.merge_operation == 'add':
                x_merger = y_func1 + y_func2
            elif self.merge_operation == 'mul':
                x_merger = torch.mul(y_func1, y_func2)
            else:
                raise NotImplementedError(f'{self.merge_operation} operation to be implimented')
        if self.merger is not None:
            y_func = self.merger(x_merger)
        else:
            y_func = x_merger
        if self.trunk is None:
            y = torch.einsum('bi,ni->bn', y_func, self.pod_basis)
        else:
            y_loc = self.trunk(x_loc)
            if self.trunk_last_activation:
                y_loc = self.activation_trunk(y_loc)
            y = torch.einsum('bi,ni->bn', y_func, torch.cat((self.pod_basis, y_loc), 1))
            y += self.b
        if self._output_transform is not None:
            y = self._output_transform(inputs, y)
        return y

