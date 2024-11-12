
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


import torch.nn.functional as F


import inspect


import time


import logging


import math


import random


import torch.utils.checkpoint


from torchvision import transforms


from torch.utils.cpp_extension import CUDA_HOME


from torch.utils.cpp_extension import CUDNN_HOME


from torch.utils.cpp_extension import CppExtension


from torch.utils.cpp_extension import CUDAExtension


import functools


from torch._dynamo.backends.registry import register_backend


from torch._dynamo.backends.common import aot_autograd


from torch._dynamo.backends.common import fake_tensor_unsupported


from torch.overrides import TorchFunctionMode


from typing import List


from typing import Optional


import torch.nn as nn


from torch._prims_common import suggest_memory_format


from itertools import product


import itertools


from typing import Sequence


from torch.utils._python_dispatch import TorchDispatchMode


import copy


from torch.hub import download_url_to_file


from torch.hub import get_dir


class TracedPosArgOnlyModuleWrapper(torch.nn.Module):

    def __init__(self, module, *, training=None):
        super().__init__()
        self.module = module
        if training is None:
            training = getattr(module, 'training', False) if isinstance(module, torch.nn.Module) else False
        self.train(training)

    def forward(self, *args, **kwargs):
        outputs = self.module(*self.convert_inputs(args, kwargs))
        unflat_outputs = flat_tensors.unflattern(outputs)
        return unflat_outputs

    def convert_inputs(self, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        return flat_tensors.flattern((args, kwargs))


class TraceablePosArgOnlyModuleWrapper(torch.nn.Module):

    def __init__(self, module):
        super().__init__()
        self.module = module
        training = getattr(module, 'training', False) if isinstance(module, torch.nn.Module) else False
        self.train(training)

    def forward(self, *args):
        orig_args, orig_kwargs = flat_tensors.unflattern(args)
        outputs = self.module(*orig_args, **orig_kwargs)
        flat_outputs = flat_tensors.flattern(outputs)
        return flat_outputs


class TritonLoRACompatibleConv(nn.Module):

    def __init__(self, module):
        super().__init__()
        self.module = module
        self.training = module.training

    def set_lora_layer(self, lora_layer):
        if hasattr(self.module, 'set_lora_layer'):
            self.module.set_lora_layer(lora_layer)

    def forward(self, x, *args, **kwargs):
        weight = self.module.weight
        x = TTO.contiguous(x, memory_format=suggest_memory_format(weight))
        return self.module(x, *args, **kwargs)


class TritonLoRACompatibleLinear(nn.Module):

    def __init__(self, module):
        super().__init__()
        self.module = module
        self.training = module.training

    def set_lora_layer(self, lora_layer):
        if hasattr(self.module, 'set_lora_layer'):
            self.module.set_lora_layer(lora_layer)

    def forward(self, x, *args, **kwargs):
        weight = self.module.weight
        x = TTO.contiguous(x, memory_format=suggest_memory_format(weight))
        return self.module(x, *args, **kwargs)


class TritonConv2D(nn.Module):

    def __init__(self, module):
        super().__init__()
        self.module = module
        self.training = module.training

    def forward(self, x, *args, **kwargs):
        weight = self.module.weight
        x = TTO.contiguous(x, memory_format=suggest_memory_format(weight))
        return self.module(x, *args, **kwargs)


class TritonLinear(nn.Module):

    def __init__(self, module):
        super().__init__()
        self.module = module
        self.training = module.training

    def forward(self, x, *args, **kwargs):
        weight = self.module.weight
        x = TTO.contiguous(x, memory_format=suggest_memory_format(weight))
        return self.module(x, *args, **kwargs)


class TritonGroupNorm(nn.Module):

    def __init__(self, module):
        super().__init__()
        self.module = module
        self.training = module.training

    def forward(self, x, *args, **kwargs):
        module = self.module
        return TTO.group_norm(x, module.num_groups, module.weight, module.bias, module.eps)


class TritonGroupNormSiLU(nn.Module):

    def __init__(self, module):
        super().__init__()
        self.module = module
        self.training = module.training

    def forward(self, x, *args, **kwargs):
        module = self.module
        return TTO.group_norm_silu(x, module.num_groups, module.weight, module.bias, module.eps)


class ConvBiasAddActivation(torch.nn.Module):

    def __init__(self, bias=True, activation_cls=None):
        super().__init__()
        self.conv = torch.nn.Conv2d(2, 2, 3, bias=bias)
        self.act = activation_cls() if activation_cls is not None else torch.nn.Identity()

    def forward(self, x, y=None, alpha=1.0):
        x = self.conv(x)
        if y is not None:
            x = x.add(y, alpha=alpha)
        x = self.act(x)
        return x


class FusedConvBiasAddActivation(torch.nn.Module):

    def __init__(self, m):
        super().__init__()
        self.conv = m.conv
        self.act = m.act
        self.train(m.training)

    def forward(self, x, y=None, alpha=1.0):
        raise NotImplementedError()


class GEGLU(nn.Module):
    """
    A [variant](https://arxiv.org/abs/2002.05202) of the gated linear unit activation function.

    Parameters:
        dim_in (`int`): The number of channels in the input.
        dim_out (`int`): The number of channels in the output.
    """

    def __init__(self, dim_in: 'int', dim_out: 'int', bias=True):
        super().__init__()
        linear_cls = nn.Linear
        self.proj = linear_cls(dim_in, dim_out * 2, bias=bias)

    def gelu(self, gate: 'torch.Tensor') ->torch.Tensor:
        if gate.device.type != 'mps':
            return F.gelu(gate)
        return F.gelu(gate.to(dtype=torch.float32))

    def forward(self, hidden_states, enable_opt=False):
        if enable_opt:
            return torch.ops.sfast.cutlass_linear_geglu_unified(hidden_states, self.proj.weight, self.proj.bias)
        else:
            hidden_states, gate = self.proj(hidden_states).chunk(2, dim=-1)
            return hidden_states * self.gelu(gate)


class LinearModule(torch.nn.Module):

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.linear = torch.nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x):
        return self.linear(x)


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (ConvBiasAddActivation,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 2, 64, 64])], {})),
    (GEGLU,
     lambda: ([], {'dim_in': 4, 'dim_out': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (LinearModule,
     lambda: ([], {'in_features': 4, 'out_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
]

