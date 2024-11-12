
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


import matplotlib.pyplot as plt


import numpy as np


import time


from typing import Optional


from functools import partial


import torch.utils.benchmark as benchmark


import torch.nn.functional as F


from torchvision import datasets


from torchvision import transforms


import functools


import torch.nn as nn


from torch.nn.modules.module import register_module_forward_hook


from torch.nn.modules.module import register_module_forward_pre_hook


from torch.overrides import TorchFunctionMode


import warnings


from typing import List


from torch.utils.cpp_extension import load


from typing import Union


from collections.abc import Mapping


from typing import Any


from typing import Dict


from abc import ABC


from typing import TYPE_CHECKING


from torch.autograd import Function


import numbers


from typing import Callable


import math


from typing import Tuple


from torch.utils import _pytree as pytree


from copy import copy


from enum import Enum


import uuid


import torch.utils.checkpoint


from torch import nn


class Optimizer(ABC):

    def __call__(self, base: 'torch.Tensor', bits: 'int', axis: 'int', group_size: 'Optional[int]'=None) ->Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        raise NotImplementedError


class SymmetricOptimizer(Optimizer):

    def __call__(self, base: 'torch.Tensor', qtype: 'qtype', axis: 'Optional[int]'=None) ->torch.Tensor:
        if axis not in [None, 0, -1]:
            raise ValueError('axis parameter must be None, 0 (first axis) or -1 (last axis)')
        if axis is not None and base.shape[axis] == 1:
            axis = None
        scale = self.optimize(base, qtype, axis)
        assert scale.dtype == base.dtype
        return scale

    def optimize(self, base: 'torch.Tensor', qmax: 'float', axis: 'Optional[int]'=None) ->torch.Tensor:
        raise NotImplementedError


class AbsmaxOptimizer(SymmetricOptimizer):

    def optimize(self, base: 'torch.Tensor', qtype: 'qtype', axis: 'Optional[int]'=None) ->Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        base = torch.abs(base)
        if axis is None:
            rmax = torch.max(base)
        else:
            dim = list(range(1, base.ndim)) if axis == 0 else list(range(0, base.ndim - 1))
            rmax = torch.amax(torch.abs(base), dim=dim, keepdim=True)
        return rmax / qtype.qmax

