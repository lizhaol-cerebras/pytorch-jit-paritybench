
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


from typing import Any


from typing import Callable


from typing import Dict


from typing import List


from typing import Optional


from typing import Tuple


from typing import cast


import torch


from torch import Tensor


from torch import nn


import torch.nn.functional as F


from torch.optim import RMSprop


import time


from torch.optim import SGD


import torch.utils.data


from collections import OrderedDict


from typing import TYPE_CHECKING


from typing import Iterator


from typing import Union


from typing import Generator


from torch.optim.lr_scheduler import LambdaLR


from torch.utils.data import DataLoader


import torchvision


from collections import deque


from typing import Deque


import copy


from queue import Queue


from functools import partial


import torch.cuda


from copy import deepcopy


from itertools import chain


from torch import optim


import torch.nn as nn


from typing import TypeVar


from torch.nn.modules.batchnorm import _BatchNorm


from torch import ByteTensor


import torch.autograd


from typing import Iterable


import typing


import torch.cuda.comm


from types import TracebackType


from typing import Type


from typing import ClassVar


from typing import FrozenSet


from typing import Set


class Classify(nn.Module):

    def __init__(self, channels_prev: 'int', num_classes: 'int'):
        super().__init__()
        self.pool = nn.AvgPool2d(7)
        self.flat = nn.Flatten()
        self.fc = nn.Linear(channels_prev, num_classes)

    def forward(self, states: 'Tuple[Tensor, Tensor]') ->Tensor:
        x, _ = states
        x = self.pool(x)
        x = self.flat(x)
        x = self.fc(x)
        return x


class Stem(nn.Module):

    def __init__(self, channels: 'int') ->None:
        super().__init__()
        self.relu = nn.ReLU(inplace=False)
        self.conv = nn.Conv2d(3, channels, 3, stride=2, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, input: 'Tensor') ->Tensor:
        x = input
        x = self.relu(x)
        x = self.conv(x)
        x = self.bn(x)
        return x


class FactorizedReduce(nn.Module):

    def __init__(self, in_channels: 'int', out_channels: 'int'):
        super().__init__()
        self.relu = nn.ReLU(inplace=False)
        self.pad = nn.ZeroPad2d((0, 1, 0, 1))
        self.conv1 = nn.Conv2d(in_channels, out_channels // 2, kernel_size=1, stride=2, bias=False)
        self.conv2 = nn.Conv2d(in_channels, out_channels // 2, kernel_size=1, stride=2, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, input: 'Tensor') ->Tensor:
        x = input
        x = self.relu(x)
        x = torch.cat([self.conv1(x), self.conv2(self.pad(x[:, :, 1:, 1:]))], dim=1)
        x = self.bn(x)
        return x


NORMAL_CONCAT = [0, 3, 4, 6]


class Operation(nn.Module):
    """Includes the operation name into the representation string for
    debugging.
    """

    def __init__(self, name: 'str', module: 'nn.Module'):
        super().__init__()
        self.name = name
        self.module = module

    def __repr__(self) ->str:
        return f'{self.__class__.__name__}[{self.name}]'

    def forward(self, *args: Any) ->Any:
        return self.module(*args)


def avg_pool_3x3(channels: 'int', stride: 'int') ->Operation:
    module = nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False)
    return Operation('avg_pool_3x3', module)


def conv_1x1(channels: 'int', stride: 'int') ->Operation:
    c = channels
    module = nn.Sequential(nn.ReLU(inplace=False), nn.Conv2d(c, c, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(c))
    return Operation('conv_1x1', module)


def conv_1x7_7x1(channels: 'int', stride: 'int') ->Operation:
    c = channels
    module = nn.Sequential(nn.ReLU(inplace=False), nn.Conv2d(c, c // 4, kernel_size=1, stride=1, padding=0, bias=False), nn.BatchNorm2d(c // 4), nn.ReLU(inplace=False), nn.Conv2d(c // 4, c // 4, kernel_size=(1, 7), stride=(1, stride), padding=(0, 3), bias=False), nn.BatchNorm2d(c // 4), nn.ReLU(inplace=False), nn.Conv2d(c // 4, c // 4, kernel_size=(7, 1), stride=(stride, 1), padding=(3, 0), bias=False), nn.BatchNorm2d(c // 4), nn.ReLU(inplace=False), nn.Conv2d(c // 4, c, kernel_size=1, stride=1, padding=0, bias=False), nn.BatchNorm2d(c))
    return Operation('conv_1x7_7x1', module)


def max_pool_3x3(channels: 'int', stride: 'int') ->Operation:
    module = nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False)
    return Operation('max_pool_3x3', module)


def none(channels: 'int', stride: 'int') ->Operation:
    module: 'nn.Module'
    if stride == 1:
        module = nn.Identity()
    else:
        module = FactorizedReduce(channels, channels)
    return Operation('none', module)


NORMAL_OPERATIONS = [(1, conv_1x1), (1, max_pool_3x3), (1, none), (0, conv_1x7_7x1), (0, conv_1x1), (0, conv_1x7_7x1), (2, max_pool_3x3), (2, none), (1, avg_pool_3x3), (5, conv_1x1)]


REDUCTION_CONCAT = [4, 5, 6]


def conv_3x3(channels: 'int', stride: 'int') ->Operation:
    c = channels
    module = nn.Sequential(nn.ReLU(inplace=False), nn.Conv2d(c, c // 4, kernel_size=1, bias=False), nn.BatchNorm2d(c // 4), nn.ReLU(inplace=False), nn.Conv2d(c // 4, c // 4, kernel_size=3, stride=stride, padding=1, bias=False), nn.BatchNorm2d(c // 4), nn.ReLU(inplace=False), nn.Conv2d(c // 4, c, kernel_size=1, bias=False), nn.BatchNorm2d(c))
    return Operation('conv_3x3', module)


def max_pool_2x2(channels: 'int', stride: 'int') ->Operation:
    module = nn.MaxPool2d(2, stride=stride, padding=0)
    return Operation('max_pool_2x2', module)


REDUCTION_OPERATIONS = [(0, max_pool_2x2), (0, max_pool_3x3), (2, none), (1, conv_3x3), (2, conv_1x7_7x1), (2, max_pool_3x3), (3, none), (1, max_pool_2x2), (2, avg_pool_3x3), (3, conv_1x1)]


def relu_conv_bn(in_channels: 'int', out_channels: 'int', kernel_size: 'int'=1, stride: 'int'=1, padding: 'int'=0) ->nn.Module:
    return nn.Sequential(nn.ReLU(inplace=False), nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False), nn.BatchNorm2d(out_channels))


class Cell(nn.Module):

    def __init__(self, channels_prev_prev: 'int', channels_prev: 'int', channels: 'int', reduction: 'bool', reduction_prev: 'bool') ->None:
        super().__init__()
        self.reduce1 = relu_conv_bn(in_channels=channels_prev, out_channels=channels)
        self.reduce2: 'nn.Module' = nn.Identity()
        if reduction_prev:
            self.reduce2 = FactorizedReduce(channels_prev_prev, channels)
        elif channels_prev_prev != channels:
            self.reduce2 = relu_conv_bn(in_channels=channels_prev_prev, out_channels=channels)
        if reduction:
            self.indices, op_classes = zip(*REDUCTION_OPERATIONS)
            self.concat = REDUCTION_CONCAT
        else:
            self.indices, op_classes = zip(*NORMAL_OPERATIONS)
            self.concat = NORMAL_CONCAT
        self.operations = nn.ModuleList()
        for i, op_class in zip(self.indices, op_classes):
            if reduction and i < 2:
                stride = 2
            else:
                stride = 1
            op = op_class(channels, stride)
            self.operations.append(op)

    def extra_repr(self) ->str:
        return f'indices: {self.indices}'

    def forward(self, input_or_states: 'Union[Tensor, Tuple[Tensor, Tensor]]') ->Tuple[Tensor, Tensor]:
        if isinstance(input_or_states, tuple):
            s1, s2 = input_or_states
        else:
            s1 = s2 = input_or_states
        skip = s1
        s1 = self.reduce1(s1)
        s2 = self.reduce2(s2)
        _states = [s1, s2]
        operations = cast(nn.ModuleList, self.operations)
        indices = cast(List[int], self.indices)
        for i in range(0, len(operations), 2):
            h1 = _states[indices[i]]
            h2 = _states[indices[i + 1]]
            op1 = operations[i]
            op2 = operations[i + 1]
            h1 = op1(h1)
            h2 = op2(h2)
            s = h1 + h2
            _states.append(s)
        return torch.cat([_states[i] for i in self.concat], dim=1), skip


Tensors = Tuple[Tensor, ...]


TensorOrTensors = Union[Tensor, Tensors]


class Batch:
    """An abstraction of an atomic tensor or a tuple of tensors. This
    eliminates every boilerplate code to classify an atomic tensor or a tuple
    of tensors.
    ::

        x = generate_tensor_or_tensors()
        x = Batch(x)

        # in-place update
        x[0] = F.apply(x[0])
        x[:] = F.apply(*x)

        # f(x) if x is a tensor.
        # f(*x) if x is a tuple of tensors.
        # y is also a batch.
        y = x.call(f)

    """

    def __init__(self, value: 'TensorOrTensors') ->None:
        self.value = value
        self.atomic = torch.is_tensor(value)

    @property
    def tensor(self) ->Tensor:
        """Retrieves the underlying tensor."""
        if not self.atomic:
            raise AttributeError('not atomic batch')
        return cast(Tensor, self.value)

    @property
    def tensors(self) ->Tensors:
        """Retrieves the underlying tensors."""
        if self.atomic:
            raise AttributeError('batch is atomic')
        return cast(Tensors, self.value)

    @property
    def tensor_or_tensors(self) ->TensorOrTensors:
        """Retrieves the underlying tensor or tensors regardless of type."""
        return self.value

    def call(self, function: 'Function') ->'Batch':
        """Calls a function by the underlying tensor or tensors. It also wraps
        the output with :class:`Batch`.
        """
        return Batch(function(self.value))

    def __repr__(self) ->str:
        return f'Batch[atomic={self.atomic!r}]({self.value!r})'

    def __iter__(self) ->Iterator[Tensor]:
        if self.atomic:
            yield self.tensor
        else:
            yield from self.tensors

    def __len__(self) ->int:
        return 1 if self.atomic else len(self.tensors)

    def __getitem__(self, index: 'int') ->Tensor:
        if not self.atomic:
            return self.tensors[index]
        if index != 0:
            raise IndexError('atomic batch allows index 0 only')
        return self.tensor

    @typing.overload
    def __setitem__(self, index: 'int', value: 'Tensor') ->None:
        ...

    @typing.overload
    def __setitem__(self, index: 'slice', value: 'Tensors') ->None:
        ...

    def __setitem__(self, index: 'Union[int, slice]', value: 'TensorOrTensors') ->None:
        if isinstance(index, int):
            value = cast(Tensor, value)
            self._setitem_by_index(index, value)
        else:
            value = cast(Tensors, value)
            self._setitem_by_slice(index, value)

    def _setitem_by_index(self, index: 'int', value: 'Tensor') ->None:
        if not self.atomic:
            i = index
            self.value = self.value[:i] + (value,) + self.value[i + 1:]
            return
        if index != 0:
            raise IndexError('atomic batch allows index 0 only')
        self.value = value

    def _setitem_by_slice(self, index: 'slice', value: 'Tensors') ->None:
        if not index.start is index.stop is index.step is None:
            raise NotImplementedError('only slice [:] supported')
        if not self.atomic:
            self.value = value
            return
        if len(value) != 1:
            raise IndexError('atomic batch cannot be replaced with multiple tensors')
        self.value = value[0]


T = TypeVar('T', bound='Skippable')


class SkipTracker:
    """Tracks saved skip tensors.

    It will update the given micro-batch in place. This is because when it
    manipulates the underlying skip tensors, the current micro-batch also has
    to be connected with the skip tensors.

    One thread has one skip tracker. Call :func:`current_skip_tracker` to get
    the skip tracker on the current thread.

    """

    def __init__(self) ->None:
        self.tensors: 'Dict[Tuple[Namespace, str], Optional[Tensor]]' = {}

    def save(self, batch: 'Batch', ns: 'Namespace', name: 'str', tensor: 'Optional[Tensor]') ->None:
        self.tensors[ns, name] = tensor

    def load(self, batch: 'Batch', ns: 'Namespace', name: 'str') ->Optional[Tensor]:
        return self.tensors.pop((ns, name))

    def copy(self, batch: 'Batch', prev_stream: 'AbstractStream', next_stream: 'AbstractStream', ns: 'Namespace', name: 'str') ->None:
        raise TypeError('copy is not supported for non-portal skip tensors')


def current_skip_tracker() ->SkipTracker:
    """Gets the skip tracker on the current thread."""
    skip_tracker = thread_local.skip_tracker
    if skip_tracker is None:
        skip_tracker = SkipTracker()
        thread_local.skip_tracker = skip_tracker
    return skip_tracker


class pop:
    """The command to pop a skip tensor.

    ::

        def forward(self, input):
            skip = yield pop('name')
            return f(input) + skip

    Args:
        name (str): name of skip tensor

    Returns:
        the skip tensor previously stashed by another layer under the same name

    """
    __slots__ = 'name',

    def __init__(self, name: 'str') ->None:
        self.name = name


class stash:
    """The command to stash a skip tensor.

    ::

        def forward(self, input):
            yield stash('name', input)
            return f(input)

    Args:
        name (str): name of skip tensor
        input (torch.Tensor or None): tensor to pass to the skip connection

    """
    __slots__ = 'name', 'tensor'

    def __init__(self, name: 'str', tensor: 'Optional[Tensor]') ->None:
        self.name = name
        self.tensor = tensor


def is_checkpointing() ->bool:
    """Whether the current forward propagation is under checkpointing.

    Returns:
        bool: :data:`True` if it's under checkpointing.

    """
    return thread_local.is_checkpointing


def is_recomputing() ->bool:
    """Whether the current forward propagation is under checkpoint
    recomputation. Use this to prevent duplicated side-effects at forward
    propagation::

        class Counter(nn.Module):
            def __init__(self):
                super().__init__()
                self.counter = 0

            def forward(self, input):
                if not is_recomputing():
                    self.counter += 1
                return input

    Returns:
        bool: :data:`True` if it's under checkpoint recomputation.

    .. seealso:: :ref:`Detecting Recomputation`

    """
    return thread_local.is_recomputing


class SkipsAsTuple(nn.Module):
    """The base module for old-fashioned skip connections. It handles arguments
    including the input and the skips.
    """

    def __init__(self, num_skips: 'int', unpack_input: 'Deque[bool]', unpack_output: 'Deque[bool]') ->None:
        super().__init__()
        self.num_skips = num_skips
        self.unpack_input = unpack_input
        self.unpack_input_for_recomputing: 'List[bool]' = []
        self.unpack_output = unpack_output

    def forward(self, input_skips: 'TensorOrTensors') ->TensorOrTensors:
        input: 'TensorOrTensors' = input_skips
        skips: 'Tensors' = ()
        if self.num_skips:
            assert isinstance(input_skips, tuple)
            input = input_skips[:-self.num_skips]
            skips = input_skips[-self.num_skips:]
            if is_recomputing():
                unpack_input = self.unpack_input_for_recomputing.pop()
            else:
                unpack_input = self.unpack_input.popleft()
                if is_checkpointing():
                    self.unpack_input_for_recomputing.append(unpack_input)
            if unpack_input:
                input = input[0]
        output, skips = self._forward(input, skips)
        unpack_output = torch.is_tensor(output)
        self.unpack_output.append(unpack_output)
        if not skips:
            return output
        if unpack_output:
            return cast(Tensor, output), *skips
        else:
            return output + skips

    def _forward(self, input: 'TensorOrTensors', skips: 'Tensors') ->Tuple[TensorOrTensors, Tensors]:
        raise NotImplementedError


class Gutter(SkipsAsTuple):
    """Just passes the incoming skips."""

    def __init__(self, module: 'nn.Module', num_skips: 'int', unpack_input: 'Deque[bool]', unpack_output: 'Deque[bool]') ->None:
        super().__init__(num_skips, unpack_input, unpack_output)
        self.module = module

    def _forward(self, input: 'TensorOrTensors', skips: 'Tensors') ->Tuple[TensorOrTensors, Tensors]:
        output = self.module(input)
        return output, skips


class Branch(SkipsAsTuple):
    """Stashes or pops skips."""

    def __init__(self, skippable: 'Skippable', prev_seats: 'Seats', next_seats: 'Seats', num_skips: 'int', unpack_input: 'Deque[bool]', unpack_output: 'Deque[bool]') ->None:
        super().__init__(num_skips, unpack_input, unpack_output)
        self.skippable = skippable
        self.prev_seats = prev_seats
        self.next_seats = next_seats

    def _forward(self, input: 'TensorOrTensors', skips: 'Tensors') ->Tuple[TensorOrTensors, Tensors]:
        stashed = {}

        def handle_stash(name: 'str', tensor: 'Optional[Tensor]') ->None:
            ns, name = self.skippable.namespaced(name)
            stashed[ns, name] = tensor

        def handle_pop(name: 'str') ->Optional[Tensor]:
            ns, name = self.skippable.namespaced(name)
            i = self.prev_seats[ns, name]
            return skips[i]
        output = self.skippable.dispatch(input, handle_stash, handle_pop)
        next_skips = []
        for ns, name in self.next_seats:
            if (ns, name) in stashed:
                skip = stashed[ns, name]
            else:
                i = self.prev_seats[ns, name]
                skip = skips[i]
            assert skip is not None
            next_skips.append(skip)
        return output, tuple(next_skips)


class Pass(nn.Module):

    def forward(self, input):
        return input


TModule = TypeVar('TModule', bound=nn.Module)


class DeferredBatchNorm(_BatchNorm):
    """A BatchNorm layer tracks multiple micro-batches to update running
    statistics per mini-batch.
    """
    sum: 'Tensor'
    sum_squares: 'Tensor'

    def __init__(self, num_features: 'int', eps: 'float'=1e-05, momentum: 'Optional[float]'=0.1, affine: 'bool'=True, chunks: 'int'=1) ->None:
        super().__init__(num_features, eps, momentum, affine, track_running_stats=True)
        self.register_buffer('sum', torch.zeros_like(self.running_mean))
        self.register_buffer('sum_squares', torch.zeros_like(self.running_var))
        self.counter = 0
        self.tracked = 0
        self.chunks = chunks

    def _check_input_dim(self, input: 'Tensor') ->None:
        if input.dim() <= 2:
            raise ValueError('expected at least 3D input (got %dD input)' % input.dim())

    def _track(self, input: 'Tensor') ->bool:
        """Tracks statistics of a micro-batch."""
        dim = [0]
        dim.extend(range(2, input.dim()))
        with torch.no_grad():
            self.sum += input.sum(dim)
            self.sum_squares += (input ** 2).sum(dim)
        size = input.size().numel() // input.size(1)
        self.counter += size
        self.tracked += 1
        return self.tracked == self.chunks

    def _commit(self) ->None:
        """Updates the running statistics of a mini-batch."""
        exponential_average_factor = 0.0
        self.num_batches_tracked += 1
        if self.momentum is None:
            exponential_average_factor = 1.0 / float(self.num_batches_tracked)
        else:
            exponential_average_factor = self.momentum
        mean = self.sum / self.counter
        var = self.sum_squares / self.counter - mean ** 2
        m = exponential_average_factor
        self.running_mean *= 1 - m
        self.running_mean += mean * m
        self.running_var *= 1 - m
        self.running_var += var * m
        self.sum.zero_()
        self.sum_squares.zero_()
        self.counter = 0
        self.tracked = 0

    def forward(self, input: 'Tensor') ->Tensor:
        if not self.training:
            return F.batch_norm(input, running_mean=self.running_mean, running_var=self.running_var, weight=self.weight, bias=self.bias, training=False, momentum=0.0, eps=self.eps)
        if not is_recomputing():
            tracked_enough = self._track(input)
            if tracked_enough:
                self._commit()
        return F.batch_norm(input, running_mean=None, running_var=None, weight=self.weight, bias=self.bias, training=True, momentum=0.0, eps=self.eps)

    @classmethod
    def convert_deferred_batch_norm(cls, module: 'TModule', chunks: 'int'=1) ->TModule:
        """Converts a :class:`nn.BatchNorm` or underlying
        :class:`nn.BatchNorm`s into :class:`DeferredBatchNorm`::

            from torchvision.models.resnet import resnet101
            from torchgpipe.batchnorm import DeferredBatchNorm
            model = resnet101()
            model = DeferredBatchNorm.convert_deferred_batch_norm(model)

        """
        if isinstance(module, DeferredBatchNorm) and module.chunks is chunks:
            return cast(TModule, module)
        module_output: 'nn.Module' = module
        if isinstance(module, _BatchNorm) and module.track_running_stats:
            module_output = DeferredBatchNorm(module.num_features, module.eps, module.momentum, module.affine, chunks)
            if module.affine:
                module_output.register_parameter('weight', module.weight)
                module_output.register_parameter('bias', module.bias)
            module_output.register_buffer('running_mean', module.running_mean)
            module_output.register_buffer('running_var', module.running_var)
            module_output.register_buffer('num_batches_tracked', module.num_batches_tracked)
        for name, child in module.named_children():
            module_output.add_module(name, cls.convert_deferred_batch_norm(child, chunks))
        return cast(TModule, module_output)


class CPUStreamType:
    pass


AbstractStream = Union[torch.Stream, CPUStreamType]


class BalanceError(ValueError):
    pass


Device = Union[torch.device, int, str]


Devices = Union[Iterable[Device], List[Device]]


MOVING_DENIED = TypeError('denied to move parameters and buffers, because GPipe should manage device placement')


def save_rng_states(device: 'torch.device', rng_states: 'Deque[RNGStates]') ->None:
    """:meth:`Checkpoint.forward` captures the current PyTorch's random number
    generator states at CPU and GPU to reuse in :meth:`Recompute.backward`.

    .. seealso:: :ref:`Referential Transparency`

    """
    cpu_rng_state = torch.get_rng_state()
    gpu_rng_state: 'Optional[ByteTensor]'
    if device.type == 'cuda':
        gpu_rng_state = torch.get_rng_state(device)
    else:
        gpu_rng_state = None
    rng_states.append((cpu_rng_state, gpu_rng_state))


class Checkpoint(torch.autograd.Function):

    @staticmethod
    def forward(ctx: 'Context', phony: 'Tensor', recomputed: 'Deque[Recomputed]', rng_states: 'Deque[RNGStates]', function: 'Function', input_atomic: 'bool', *input: Tensor) ->TensorOrTensors:
        ctx.recomputed = recomputed
        ctx.rng_states = rng_states
        save_rng_states(input[0].device, ctx.rng_states)
        ctx.function = function
        ctx.input_atomic = input_atomic
        ctx.save_for_backward(*input)
        with torch.no_grad(), enable_checkpointing():
            output = function(input[0] if input_atomic else input)
        return output

    @staticmethod
    def backward(ctx: 'Context', *grad_output: Tensor) ->Tuple[Optional[Tensor], ...]:
        output, input_leaf = ctx.recomputed.pop()
        if isinstance(output, tuple):
            tensors = output
        else:
            tensors = output,
        if any(y.requires_grad for y in tensors):
            torch.autograd.backward(tensors, grad_output)
        grad_input: 'List[Optional[Tensor]]' = [None, None, None, None, None]
        grad_input.extend(x.grad for x in input_leaf)
        return tuple(grad_input)


class Recompute(torch.autograd.Function):

    @staticmethod
    def forward(ctx: 'Context', phony: 'Tensor', recomputed: 'Deque[Recomputed]', rng_states: 'Deque[RNGStates]', function: 'Function', input_atomic: 'bool', *input: Tensor) ->Tensor:
        ctx.recomputed = recomputed
        ctx.rng_states = rng_states
        ctx.function = function
        ctx.input_atomic = input_atomic
        ctx.save_for_backward(*input)
        return phony

    @staticmethod
    def backward(ctx: 'Context', *grad_output: Tensor) ->Tuple[None, ...]:
        input = ctx.saved_tensors
        input_leaf = tuple(x.detach().requires_grad_(x.requires_grad) for x in input)
        with restore_rng_states(input[0].device, ctx.rng_states):
            with torch.enable_grad(), enable_recomputing():
                output = ctx.function(input_leaf[0] if ctx.input_atomic else input_leaf)
        ctx.recomputed.append((output, input_leaf))
        grad_input: 'List[None]' = [None, None, None, None, None]
        grad_input.extend(None for _ in ctx.saved_tensors)
        return tuple(grad_input)


CPUStream = CPUStreamType()


def default_stream(device: 'torch.device') ->AbstractStream:
    """:func:`torch.cuda.default_stream` for either CPU or CUDA device."""
    if device.type != 'cuda':
        return CPUStream
    return torch.cuda.default_stream(device)


def as_cuda(stream: 'AbstractStream') ->torch.Stream:
    """Casts the given stream as :class:`torch.cuda.Stream`."""
    return cast(torch.Stream, stream)


def is_cuda(stream: 'AbstractStream') ->bool:
    """Returns ``True`` if the given stream is a valid CUDA stream."""
    return stream is not CPUStream


def get_phony(device: 'torch.device', *, requires_grad: bool) ->Tensor:
    """Gets a phony. Phony is tensor without space. It is useful to make
    arbitrary dependency in a autograd graph because it doesn't require any
    gradient accumulation.

    .. note::

        Phonies for each device are cached. If an autograd function gets a phony
        internally, the phony must be detached to be returned. Otherwise, the
        autograd engine will mutate the cached phony in-place::

            class Phonify(torch.autograd.Function):
                @staticmethod
                def forward(ctx, input):
                    phony = get_phony(input.device, requires_grad=False)
                    return phony.detach()  # detach() is necessary.

    """
    key = device, requires_grad
    try:
        phony = _phonies[key]
    except KeyError:
        with use_stream(default_stream(device)):
            phony = torch.empty(0, device=device, requires_grad=requires_grad)
        _phonies[key] = phony
    return phony


class Fork(torch.autograd.Function):

    @staticmethod
    def forward(ctx: "'Fork'", input: 'Tensor') ->Tuple[Tensor, Tensor]:
        phony = get_phony(input.device, requires_grad=False)
        return input.detach(), phony.detach()

    @staticmethod
    def backward(ctx: "'Fork'", grad_input: 'Tensor', grad_grad: 'Tensor') ->Tensor:
        return grad_input


def fork(input: 'Tensor') ->Tuple[Tensor, Tensor]:
    """Branches out from an autograd lane of the given tensor."""
    if torch.is_grad_enabled() and input.requires_grad:
        input, phony = Fork.apply(input)
    else:
        phony = get_phony(input.device, requires_grad=False)
    return input, phony


class Join(torch.autograd.Function):

    @staticmethod
    def forward(ctx: "'Join'", input: 'Tensor', phony: 'Tensor') ->Tensor:
        return input.detach()

    @staticmethod
    def backward(ctx: "'Join'", grad_input: 'Tensor') ->Tuple[Tensor, None]:
        return grad_input, None


def join(input: 'Tensor', phony: 'Tensor') ->Tensor:
    """Merges two autograd lanes."""
    if torch.is_grad_enabled() and (input.requires_grad or phony.requires_grad):
        input = Join.apply(input, phony)
    return input


class Checkpointing:
    """Generates a pair of :class:`Checkpoint` and :class:`Recompute`."""

    def __init__(self, function: 'Function', batch: 'Batch') ->None:
        self.function = function
        self.batch = batch
        self.recomputed: 'Deque[Recomputed]' = deque(maxlen=1)
        self.rng_states: 'Deque[RNGStates]' = deque(maxlen=1)

    def checkpoint(self) ->Batch:
        """Returns a batch applied by :class:`Checkpoint`."""
        input_atomic = self.batch.atomic
        input = tuple(self.batch)
        phony = get_phony(self.batch[0].device, requires_grad=True)
        output = Checkpoint.apply(phony, self.recomputed, self.rng_states, self.function, input_atomic, *input)
        return Batch(output)

    def recompute(self, batch: 'Batch') ->None:
        """Applies :class:`Recompute` to the batch in place."""
        input_atomic = self.batch.atomic
        input = tuple(self.batch)
        batch[0], phony = fork(batch[0])
        phony = Recompute.apply(phony, self.recomputed, self.rng_states, self.function, input_atomic, *input)
        batch[0] = join(batch[0], phony)


ExcInfo = Tuple[Type[BaseException], BaseException, TracebackType]


class PortalBlue(torch.autograd.Function):
    """Hides a tensor from the autograd engine by a :class:`Portal`."""

    @staticmethod
    def forward(ctx: 'Context', portal: 'Portal', tensor: 'Tensor') ->Tensor:
        ctx.portal = portal
        phony = get_phony(tensor.device, requires_grad=False)
        return phony.detach()

    @staticmethod
    def backward(ctx: 'Context', grad_phony: 'Tensor') ->Tuple[None, Tensor]:
        grad = ctx.portal.use_grad()
        return None, grad


def current_stream(device: 'torch.device') ->AbstractStream:
    """:func:`torch.cuda.current_stream` for either CPU or CUDA device."""
    if device.type != 'cuda':
        return CPUStream
    return torch.cuda.current_stream(device)


def get_device(stream: 'AbstractStream') ->torch.device:
    """Gets the device from CPU or CUDA stream."""
    if is_cuda(stream):
        return as_cuda(stream).device
    return torch.device('cpu')


def record_stream(tensor: 'torch.Tensor', stream: 'AbstractStream') ->None:
    """:meth:`torch.Tensor.record_stream` for either CPU or CUDA stream."""
    if is_cuda(stream):
        tensor = tensor.new_empty([0]).set_(tensor.storage())
        tensor.record_stream(as_cuda(stream))


class Copy(torch.autograd.Function):
    """Copies tensors on specific streams."""

    @staticmethod
    def forward(ctx: 'Context', prev_stream: 'AbstractStream', next_stream: 'AbstractStream', *input: Tensor) ->Tensors:
        ctx.prev_stream = prev_stream
        ctx.next_stream = next_stream
        output = []
        output_stream = current_stream(get_device(next_stream))
        with use_stream(prev_stream), use_stream(next_stream):
            for x in input:
                y = x
                output.append(y)
                record_stream(x, prev_stream)
                record_stream(y, output_stream)
        return tuple(output)

    @staticmethod
    def backward(ctx: 'Context', *grad_output: Tensor) ->Tuple[Optional[Tensor], ...]:
        prev_stream = ctx.prev_stream
        next_stream = ctx.next_stream
        grad_input: 'Deque[Tensor]' = deque(maxlen=len(grad_output))
        input_stream = current_stream(get_device(prev_stream))
        with use_stream(prev_stream), use_stream(next_stream):
            for x in reversed(grad_output):
                y = x
                grad_input.appendleft(y)
                record_stream(x, next_stream)
                record_stream(y, input_stream)
        grad_streams: 'Tuple[Optional[Tensor], ...]' = (None, None)
        return grad_streams + tuple(grad_input)


class PortalCopy(torch.autograd.Function):
    """Copies the hidden tensor in a :class:`Portal`. It replaces the hidden
    tensor with copied one.
    """

    @staticmethod
    def forward(ctx: 'Context', portal: 'Portal', prev_stream: 'AbstractStream', next_stream: 'AbstractStream', phony: 'Tensor') ->Tensor:
        ctx.portal = portal
        assert portal.tensor is not None
        portal.tensor, = Copy.forward(ctx, prev_stream, next_stream, portal.tensor)
        phony = get_phony(get_device(next_stream), requires_grad=False)
        return phony.detach()

    @staticmethod
    def backward(ctx: 'Context', grad_phony: 'Tensor') ->Tuple[None, None, None, None]:
        portal = ctx.portal
        assert portal.grad is not None
        _, _, portal.grad = Copy.backward(ctx, portal.grad)
        return None, None, None, None


class PortalOrange(torch.autograd.Function):
    """Retrieves the hidden tensor from a :class:`Portal`."""

    @staticmethod
    def forward(ctx: 'Context', portal: 'Portal', phony: 'Tensor') ->Tensor:
        ctx.portal = portal
        tensor = portal.use_tensor()
        assert tensor is not None
        return tensor.detach()

    @staticmethod
    def backward(ctx: 'Context', grad: 'Tensor') ->Tuple[None, None]:
        ctx.portal.put_grad(grad)
        return None, None


class Portal:
    """A portal for a tensor."""

    def __init__(self, tensor: 'Optional[Tensor]', tensor_life: 'int') ->None:
        self.put_tensor(tensor, tensor_life)
        self.grad: 'Optional[Tensor]' = None

    def blue(self) ->Tensor:
        """Creates a :class:`PortalBlue` which hides the underlying tensor from
        the autograd engine.

        Join the returning phony to the main lane of the autograd graph to
        assure the correct backpropagation::

            PortalBlue --+
                         |
            ---------- Join --

        """
        tensor = self.use_tensor()
        if tensor is None:
            return get_phony(torch.device('cpu'), requires_grad=False)
        return PortalBlue.apply(self, tensor)

    def orange(self, phony: 'Tensor') ->Optional[Tensor]:
        """Creates a :class:`PortalOrange` which retrieves the hidden tensor
        without losing ability of backpropagation.

        Give a phony forked from the main lane of an autograd graph::

                +-- PortalOrange --+
                |                  |
            -- Fork --------- f(a, b) --

        """
        self.check_tensor_life()
        if self.tensor is None:
            return self.use_tensor()
        return PortalOrange.apply(self, phony)

    def copy(self, prev_stream: 'AbstractStream', next_stream: 'AbstractStream', phony: 'Tensor') ->Tensor:
        """Copies the hidden tensor by a :class:`PortalCopy`.

        Give a phony and use the returning phony to keep backpropagation::

                +-- PortalCopy --+
                |                |
            -- Fork ---------- Join --

        """
        if self.tensor is None:
            return get_phony(torch.device('cpu'), requires_grad=False)
        return PortalCopy.apply(self, prev_stream, next_stream, phony)

    def check_tensor_life(self) ->None:
        if self.tensor_life <= 0:
            raise RuntimeError('tensor in portal has been removed')

    def put_tensor(self, tensor: 'Optional[Tensor]', tensor_life: 'int') ->None:
        """Stores a tensor into this portal."""
        self.tensor_life = tensor_life
        if tensor_life > 0:
            self.tensor = tensor
        else:
            self.tensor = None

    def use_tensor(self) ->Optional[Tensor]:
        """Retrieves the underlying tensor and decreases the tensor  life. When
        the life becomes 0, it the tensor will be removed.
        """
        self.check_tensor_life()
        tensor = self.tensor
        self.tensor_life -= 1
        if self.tensor_life <= 0:
            self.tensor = None
        return tensor

    def put_grad(self, grad: 'Tensor') ->None:
        """Stores a gradient into this portal."""
        self.grad = grad

    def use_grad(self) ->Tensor:
        """Retrieves and removes the underlying gradient. The gradient is
        always ephemeral.
        """
        if self.grad is None:
            raise RuntimeError('grad in portal has been removed or never set')
        grad = self.grad
        self.grad = None
        return grad


class SkipTrackerThroughPotals(SkipTracker):
    """Tracks saved skip tensors through portals. The skip tensors will be
    hidden in portals so that the autograd engine does not need to track them.

    This tracker is only used when the training or evaluating module is wrapped
    with :class:`torchgpipe.GPipe`.

    """

    def __init__(self, skip_layout: 'SkipLayout') ->None:
        super().__init__()
        self.skip_layout = skip_layout
        self.portals: 'Dict[Tuple[Namespace, str], Portal]' = {}

    def save(self, batch: 'Batch', ns: 'Namespace', name: 'str', tensor: 'Optional[Tensor]') ->None:
        """Saves the stashed skip tensor in a portal. The portal is then
        connected to the given micro-batch with :class:`Join`.
        """
        if not self.skip_layout.requires_copy(ns, name):
            super().save(batch, ns, name, tensor)
            return
        if (ns, name) not in self.portals:
            if is_checkpointing():
                tensor_life = 3
            else:
                tensor_life = 2
            portal = Portal(tensor, tensor_life)
            self.portals[ns, name] = portal
        else:
            portal = self.portals[ns, name]
            tensor_life = 1
            portal.put_tensor(tensor, tensor_life)
        phony = portal.blue()
        batch[0] = join(batch[0], phony)

    def load(self, batch: 'Batch', ns: 'Namespace', name: 'str') ->Optional[Tensor]:
        """Loads a skip tensor from the corresponding portal to pop. The given
        micro-batch is connected to the portal with :class:`Fork`.
        """
        if not self.skip_layout.requires_copy(ns, name):
            tensor = super().load(batch, ns, name)
            return tensor
        portal = self.portals[ns, name]
        batch[0], phony = fork(batch[0])
        tensor = portal.orange(phony)
        return tensor

    def copy(self, batch: 'Batch', prev_stream: 'AbstractStream', next_stream: 'AbstractStream', ns: 'Namespace', name: 'str') ->None:
        """Copies the skip tensor in the corresponding portal. The given
        micro-batch and the portal will be tied with :class:`Fork` and
        :class:`Join`.
        """
        assert self.skip_layout.requires_copy(ns, name)
        batch[0], phony = fork(batch[0])
        portal = self.portals[ns, name]
        phony = portal.copy(prev_stream, next_stream, phony)
        batch[0] = join(batch[0], phony)


class Task:
    """A task represents how to compute a micro-batch on a partition.

    It consists of two parts: :meth:`compute` and :meth:`finalize`.
    :meth:`compute` should be executed in worker threads concurrently.
    :meth:`finalize` should be executed after when worker threads complete to
    execute :meth:`compute`.

    :meth:`compute` might be boosted by worker threads. Because it produces
    several CUDA API calls by user code. In PyTorch, parallel CUDA API calls
    are not serialized through GIL. So more than one CUDA API call can be
    produced at the same time.

    """

    def __init__(self, stream: 'AbstractStream', *, compute: Callable[[], Batch], finalize: Optional[Callable[[Batch], None]]) ->None:
        self.stream = stream
        self._compute = compute
        self._finalize = finalize

    def compute(self) ->Batch:
        with use_stream(self.stream):
            return self._compute()

    def finalize(self, batch: 'Batch') ->None:
        if self._finalize is None:
            return
        with use_stream(self.stream):
            self._finalize(batch)


def clock_cycles(m: 'int', n: 'int') ->Iterable[List[Tuple[int, int]]]:
    """Generates schedules for each clock cycle."""
    for k in range(m + n - 1):
        yield [(k - j, j) for j in range(max(1 + k - m, 0), min(1 + k, n))]


def depend(fork_from: 'Batch', join_to: 'Batch') ->None:
    fork_from[0], phony = fork(fork_from[0])
    join_to[0] = join(join_to[0], phony)


def worker(in_queue: 'InQueue', out_queue: 'OutQueue', device: 'torch.device', grad_mode: 'bool') ->None:
    """The main loop of a worker thread."""
    torch.set_grad_enabled(grad_mode)
    with use_device(device):
        while True:
            task = in_queue.get()
            if task is None:
                break
            try:
                batch = task.compute()
            except Exception:
                exc_info = cast(ExcInfo, sys.exc_info())
                out_queue.put((False, exc_info))
                continue
            out_queue.put((True, (task, batch)))
    done = False, None
    out_queue.put(done)


def wait_stream(source: 'AbstractStream', target: 'AbstractStream') ->None:
    """:meth:`torch.cuda.Stream.wait_stream` for either CPU or CUDA stream. It
    makes the source stream wait until the target stream completes work queued.
    """
    if is_cuda(target):
        if is_cuda(source):
            as_cuda(source).wait_stream(as_cuda(target))
        else:
            as_cuda(target).synchronize()


class Wait(torch.autograd.Function):
    """Synchronizes a stream to another stream.

    Place it just before you want to start an operation on the next stream,
    provided that all operations on the previous stream are done.

    """

    @staticmethod
    def forward(ctx: 'Context', prev_stream: 'AbstractStream', next_stream: 'AbstractStream', *input: Tensor) ->Tensors:
        ctx.prev_stream = prev_stream
        ctx.next_stream = next_stream
        wait_stream(next_stream, prev_stream)
        return tuple(x.detach() for x in input)

    @staticmethod
    def backward(ctx: 'Context', *grad_input: Tensor) ->Tuple[Optional[Tensor], ...]:
        prev_stream = ctx.prev_stream
        next_stream = ctx.next_stream
        wait_stream(prev_stream, next_stream)
        grad_streams: 'Tuple[Optional[Tensor], ...]' = (None, None)
        return grad_streams + grad_input


def wait(batch: 'Batch', prev_stream: 'AbstractStream', next_stream: 'AbstractStream') ->None:
    batch[:] = Wait.apply(prev_stream, next_stream, *batch)


class Pipeline:
    """The pipeline parallelism for GPipe."""

    def __init__(self, batches: 'List[Batch]', partitions: 'List[nn.Sequential]', devices: 'Optional[List[torch.device]]'=None, copy_streams: 'Optional[List[List[AbstractStream]]]'=None, skip_layout: 'Optional[SkipLayout]'=None, checkpoint_stop: 'int'=0) ->None:
        self.batches = batches
        self.partitions = partitions
        if devices is None:
            devices = [torch.device('cpu') for _ in partitions]
        self.devices = devices
        if copy_streams is None:
            copy_streams = [([current_stream(d)] * len(batches)) for d in devices]
        self.copy_streams = copy_streams
        if skip_layout is None:
            skip_layout = inspect_skip_layout(partitions)
        self.skip_layout = skip_layout
        self.checkpoint_stop = checkpoint_stop

    def run(self) ->None:
        """Runs pipeline parallelism.

        It modifies the given batches in place.

        """
        batches = self.batches
        partitions = self.partitions
        devices = self.devices
        skip_layout = self.skip_layout
        m = len(batches)
        n = len(partitions)
        skip_trackers = [SkipTrackerThroughPotals(skip_layout) for _ in batches]
        with spawn_workers(devices) as (in_queues, out_queues):
            for schedule in clock_cycles(m, n):
                self.fence(schedule, skip_trackers)
                self.compute(schedule, skip_trackers, in_queues, out_queues)

    def fence(self, schedule: 'List[Tuple[int, int]]', skip_trackers: 'List[SkipTrackerThroughPotals]') ->None:
        """Copies micro-batches after computation for the previous
        micro-batches.
        """
        batches = self.batches
        copy_streams = self.copy_streams
        skip_layout = self.skip_layout
        for i, j in schedule:
            if i != 0:
                depend(batches[i - 1], batches[i])
            next_stream = copy_streams[j][i]
            for prev_j, ns, name in skip_layout.copy_policy(j):
                prev_stream = copy_streams[prev_j][i]
                skip_trackers[i].copy(batches[i], prev_stream, next_stream, ns, name)
            if j != 0:
                prev_stream = copy_streams[j - 1][i]
                copy(batches[i], prev_stream, next_stream)

    def compute(self, schedule: 'List[Tuple[int, int]]', skip_trackers: 'List[SkipTrackerThroughPotals]', in_queues: 'List[InQueue]', out_queues: 'List[OutQueue]') ->None:
        """Runs tasks with synchronization to copy streams."""
        batches = self.batches
        partitions = self.partitions
        devices = self.devices
        copy_streams = self.copy_streams
        checkpoint_stop = self.checkpoint_stop
        n = len(partitions)
        streams = [current_stream(d) for d in devices]
        exc_info: 'Optional[ExcInfo]' = None
        for i, j in schedule:
            batch = batches[i]
            partition = partitions[j]
            if j != 0:
                wait(batch, copy_streams[j][i], streams[j])
            checkpoint = i < checkpoint_stop
            if checkpoint:

                def function(input: 'TensorOrTensors', partition: 'nn.Sequential'=partition, skip_tracker: 'SkipTrackerThroughPotals'=skip_trackers[i]) ->TensorOrTensors:
                    with use_skip_tracker(skip_tracker):
                        return partition(input)
                chk = Checkpointing(function, batch)
                task = Task(streams[j], compute=chk.checkpoint, finalize=chk.recompute)
                del function, chk
            else:

                def compute(batch: 'Batch'=batch, partition: 'nn.Sequential'=partition, skip_tracker: 'SkipTrackerThroughPotals'=skip_trackers[i]) ->Batch:
                    with use_skip_tracker(skip_tracker):
                        return batch.call(partition)
                task = Task(streams[j], compute=compute, finalize=None)
                del compute
            in_queues[j].put(task)
        for i, j in schedule:
            ok, payload = out_queues[j].get()
            if exc_info is not None:
                continue
            elif not ok:
                exc_info = cast(ExcInfo, payload)
                continue
            task, batch = cast(Tuple[Task, Batch], payload)
            if j != n - 1:
                wait(batch, streams[j], copy_streams[j][i])
            with use_device(devices[j]):
                task.finalize(batch)
            batches[i] = batch
        if exc_info is not None:
            raise exc_info[0].with_traceback(exc_info[1], exc_info[2])


def new_stream(device: 'torch.device') ->AbstractStream:
    """Creates a new stream for either CPU or CUDA device."""
    if device.type != 'cuda':
        return CPUStream
    return torch.Stream(device)


def recommend_auto_balance(message: 'str') ->str:
    """Expands a message with recommendation to :mod:`torchgpipe.balance`."""
    return f"{message}\n\nIf your model is still under development, its optimal balance would change\nfrequently. In this case, we highly recommend 'torchgpipe.balance' for naive\nautomatic balancing:\n\n  from torchgpipe import GPipe\n  from torchgpipe.balance import balance_by_time\n\n  partitions = torch.cuda.device_count()\n  sample = torch.empty(...)\n  balance = balance_by_time(partitions, model, sample)\n\n  model = GPipe(model, balance, ...)\n"


def split_module(module: 'nn.Sequential', balance: 'Iterable[int]', devices: 'List[torch.device]') ->Tuple[List[nn.Sequential], List[int], List[torch.device]]:
    """Splits a module into multiple partitions.

    Returns:
        A tuple of (partitions, balance, devices).

        Partitions are represented as a :class:`~torch.nn.ModuleList` whose
        item is a partition. All layers in a partition are placed in the
        same device.

    Raises:
        BalanceError:
            wrong balance
        IndexError:
            the number of devices is fewer than the number of partitions.

    """
    balance = list(balance)
    if len(module) != sum(balance):
        raise BalanceError(f'module and sum of balance have different length (module: {len(module)}, sum of balance: {sum(balance)})')
    if any(x <= 0 for x in balance):
        raise BalanceError(f'all balance numbers must be positive integer (balance: {balance})')
    if len(balance) > len(devices):
        raise IndexError(f'too few devices to hold given partitions (devices: {len(devices)}, partitions: {len(balance)})')
    j = 0
    partitions = []
    layers: 'NamedModules' = OrderedDict()
    for name, layer in module.named_children():
        layers[name] = layer
        if len(layers) == balance[j]:
            partition = nn.Sequential(layers)
            device = devices[j]
            partition
            partitions.append(partition)
            layers.clear()
            j += 1
    partitions = cast(List[nn.Sequential], nn.ModuleList(partitions))
    del devices[j:]
    return partitions, balance, devices


def verify_module(module: 'nn.Sequential') ->None:
    if not isinstance(module, nn.Sequential):
        raise TypeError('module must be nn.Sequential to be partitioned')
    named_children = list(module.named_children())
    if len(named_children) != len(module):
        raise ValueError('module with duplicate children is not supported')
    num_parameters = len(list(module.parameters()))
    num_child_parameters = sum(len(list(child.parameters())) for child in module.children())
    if num_parameters != num_child_parameters:
        raise ValueError('module with duplicate parameters in distinct children is not supported')


def verify_skippables(module: 'nn.Sequential') ->None:
    """Verifies if the underlying skippable modules satisfy integrity.

    Every skip tensor must have only one pair of `stash` and `pop`. If there
    are one or more unmatched pairs, it will raise :exc:`TypeError` with the
    detailed messages.

    Here are a few failure cases. :func:`verify_skippables` will report failure
    for these cases::

        # Layer1 stashes "1to3".
        # Layer3 pops "1to3".

        nn.Sequential(Layer1(), Layer2())
        #               └──── ?

        nn.Sequential(Layer2(), Layer3())
        #                   ? ────┘

        nn.Sequential(Layer1(), Layer2(), Layer3(), Layer3())
        #               └───────────────────┘       ^^^^^^

        nn.Sequential(Layer1(), Layer1(), Layer2(), Layer3())
        #             ^^^^^^      └───────────────────┘

    To use the same name for multiple skip tensors, they must be isolated by
    different namespaces. See :meth:`isolate()
    <torchgpipe.skip.skippable.Skippable.isolate>`.

    Raises:
        TypeError:
            one or more pairs of `stash` and `pop` are not matched.

    """
    stashed: 'Set[Tuple[Namespace, str]]' = set()
    popped: 'Set[Tuple[Namespace, str]]' = set()
    msgs: 'List[str]' = []
    for layer_name, layer in module.named_children():
        if not isinstance(layer, Skippable):
            continue
        for name in (layer.stashable_names & layer.poppable_names):
            msg = f"'{layer_name}' declared '{name}' both as stashable and as poppable"
            msgs.append(msg)
        for ns, name in layer.stashable():
            if name in layer.poppable_names:
                continue
            if (ns, name) in stashed:
                msg = f"'{layer_name}' redeclared '{name}' as stashable but not isolated by namespace"
                msgs.append(msg)
                continue
            stashed.add((ns, name))
        for ns, name in layer.poppable():
            if name in layer.stashable_names:
                continue
            if (ns, name) in popped:
                msg = f"'{layer_name}' redeclared '{name}' as poppable but not isolated by namespace"
                msgs.append(msg)
                continue
            if (ns, name) not in stashed:
                msg = f"'{layer_name}' declared '{name}' as poppable but it was not stashed"
                msgs.append(msg)
                continue
            popped.add((ns, name))
    for _, name in (stashed - popped):
        msg = f"no module declared '{name}' as poppable but stashed"
        msgs.append(msg)
    if msgs:
        raise TypeError('one or more pairs of stash and pop do not match:\n\n%s' % '\n'.join('* %s' % x for x in msgs))


class GPipe(Module):
    """Wraps an arbitrary :class:`nn.Sequential <torch.nn.Sequential>` module
    to train on GPipe_. If the module requires lots of memory, GPipe will be
    very efficient.
    ::

        model = nn.Sequential(a, b, c, d)
        model = GPipe(model, balance=[1, 1, 1, 1], chunks=8)
        output = model(input)

    .. _GPipe: https://arxiv.org/abs/1811.06965

    GPipe combines pipeline parallelism with checkpointing to reduce peak
    memory required to train while minimizing device under-utilization.

    You should determine the balance when defining a :class:`GPipe` module, as
    balancing will not be done automatically. The module will be partitioned
    into multiple devices according to the given balance. You may rely on
    heuristics to find your own optimal configuration.

    Args:
        module (torch.nn.Sequential):
            sequential module to be parallelized
        balance (ints):
            list of number of layers in each partition

    Keyword Args:
        devices (iterable of devices):
            devices to use (default: all CUDA devices)
        chunks (int):
            number of micro-batches (default: ``1``)
        checkpoint (str):
            when to enable checkpointing, one of ``'always'``,
            ``'except_last'``, or ``'never'`` (default: ``'except_last'``)
        deferred_batch_norm (bool):
            whether to use deferred BatchNorm moving statistics (default:
            :data:`False`, see :ref:`Deferred Batch Normalization` for more
            details)

    Raises:
        TypeError:
            the module is not a :class:`nn.Sequential <torch.nn.Sequential>`.
        ValueError:
            invalid arguments, or wrong balance
        IndexError:
            the number of devices is fewer than the number of partitions.

    """
    balance: 'List[int]' = []
    devices: 'List[torch.device]' = []
    chunks: 'int' = 1
    checkpoint: 'str' = 'except_last'

    def __init__(self, module: 'nn.Sequential', balance: 'Optional[Iterable[int]]'=None, *, devices: Optional[Devices]=None, chunks: int=chunks, checkpoint: str=checkpoint, deferred_batch_norm: bool=False) ->None:
        super().__init__()
        chunks = int(chunks)
        checkpoint = str(checkpoint)
        if balance is None:
            raise ValueError(recommend_auto_balance('balance is required'))
        if chunks <= 0:
            raise ValueError('number of chunks must be positive integer')
        if checkpoint not in ['always', 'except_last', 'never']:
            raise ValueError("checkpoint is not one of 'always', 'except_last', or 'never'")
        verify_module(module)
        verify_skippables(module)
        self.chunks = chunks
        self.checkpoint = checkpoint
        if deferred_batch_norm:
            module = DeferredBatchNorm.convert_deferred_batch_norm(module, chunks)
        if devices is None:
            devices = range(torch.cuda.device_count())
        devices = [torch.device(d) for d in devices]
        devices = cast(List[torch.device], devices)
        try:
            self.partitions, self.balance, self.devices = split_module(module, balance, devices)
        except BalanceError as exc:
            raise ValueError(recommend_auto_balance(str(exc)))
        self._copy_streams: 'List[List[AbstractStream]]' = []
        self._skip_layout = inspect_skip_layout(self.partitions)

    def __len__(self) ->int:
        """Counts the length of the underlying sequential module."""
        return sum(len(p) for p in self.partitions)

    def __getitem__(self, index: 'int') ->nn.Module:
        """Gets a layer in the underlying sequential module."""
        partitions = self.partitions
        if index < 0:
            partitions = partitions[::-1]
        for partition in partitions:
            try:
                return partition[index]
            except IndexError:
                pass
            shift = len(partition)
            if index < 0:
                index += shift
            else:
                index -= shift
        raise IndexError

    def __iter__(self) ->Iterable[nn.Module]:
        """Iterates over children of the underlying sequential module."""
        for partition in self.partitions:
            yield from partition

    def cuda(self, device: 'Optional[Device]'=None) ->'GPipe':
        raise MOVING_DENIED

    def cpu(self) ->'GPipe':
        raise MOVING_DENIED

    def to(self, *args: Any, **kwargs: Any) ->'GPipe':
        if 'device' in kwargs or 'tensor' in kwargs:
            raise MOVING_DENIED
        if args:
            if isinstance(args[0], (torch.device, int, str)):
                raise MOVING_DENIED
            if torch.is_tensor(args[0]):
                raise MOVING_DENIED
        return super()

    def _ensure_copy_streams(self) ->List[List[AbstractStream]]:
        """Ensures that :class:`GPipe` caches CUDA streams for copy.

        It's worth to cache CUDA streams although PyTorch already manages a
        pool of pre-allocated CUDA streams, because it may reduce GPU memory
        fragementation when the number of micro-batches is small.

        """
        if not self._copy_streams:
            for device in self.devices:
                self._copy_streams.append([new_stream(device) for _ in range(self.chunks)])
        return self._copy_streams

    def forward(self, input: 'TensorOrTensors') ->TensorOrTensors:
        """:class:`GPipe` is a fairly transparent module wrapper. It doesn't
        modify the input and output signature of the underlying module. But
        there's type restriction. Input and output have to be a
        :class:`~torch.Tensor` or a tuple of tensors. This restriction is
        applied at partition boundaries too.

        Args:
            input (torch.Tensor or tensors): input mini-batch

        Returns:
            tensor or tensors: output mini-batch

        Raises:
            TypeError: input is not a tensor or tensors.

        """
        microbatch.check(input)
        if not self.devices:
            return input
        batches = microbatch.scatter(input, self.chunks)
        copy_streams = self._ensure_copy_streams()
        if self.training:
            checkpoint_stop = {'always': self.chunks, 'except_last': self.chunks - 1, 'never': 0}[self.checkpoint]
        else:
            checkpoint_stop = 0
        pipeline = Pipeline(batches, self.partitions, self.devices, copy_streams, self._skip_layout, checkpoint_stop)
        pipeline.run()
        output = microbatch.gather(batches)
        return output


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (Cell,
     lambda: ([], {'channels_prev_prev': 4, 'channels_prev': 4, 'channels': 4, 'reduction': 4, 'reduction_prev': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (DeferredBatchNorm,
     lambda: ([], {'num_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (FactorizedReduce,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Pass,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Stem,
     lambda: ([], {'channels': 4}),
     lambda: ([torch.rand([4, 3, 64, 64])], {})),
]

