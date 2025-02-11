
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


import time


from typing import Optional


import torch


import logging


from typing import List


import numpy


from typing import Callable


from typing import Dict


from torch import Tensor


from torch.nn import Module


from abc import ABC


from abc import abstractmethod


from torch.nn import Sigmoid


from torch.nn import Sequential


from torch.nn import Tanh


import math


from torch.nn import ModuleList


from torch.nn import Upsample


from typing import Union


from torch.nn import Conv2d


from torch.nn import ConvTranspose2d


from matplotlib import pyplot


from torch.nn.functional import affine_grid


from torch.nn.functional import grid_sample


from torch import zero_


from torch.nn.init import kaiming_normal_


from torch.nn.init import xavier_normal_


from torch.nn.init import normal_


from torch.nn import ReLU


from torch.nn import LeakyReLU


from torch.nn import ELU


from torch.nn import ReLU6


from torch.nn import Hardswish


from torch.nn import SiLU


from torch import layer_norm


from torch.nn import BatchNorm2d


from torch.nn import InstanceNorm2d


from torch.nn import Parameter


from torch.nn.init import constant_


from torch.nn.utils import spectral_norm


from typing import Tuple


from enum import Enum


from torch.nn.functional import interpolate


from matplotlib import cm


def create_init_function(method: 'str'='none') ->Callable[[Module], Module]:

    def init(module: 'Module'):
        if method == 'none':
            return module
        elif method == 'he':
            kaiming_normal_(module.weight)
            return module
        elif method == 'xavier':
            xavier_normal_(module.weight)
            return module
        elif method == 'dcgan':
            normal_(module.weight, 0.0, 0.02)
            return module
        elif method == 'dcgan_001':
            normal_(module.weight, 0.0, 0.01)
            return module
        elif method == 'zero':
            with torch.no_grad():
                zero_(module.weight)
            return module
        else:
            raise ('Invalid initialization method %s' % method)
    return init


class ModuleFactory(ABC):

    @abstractmethod
    def create(self) ->Module:
        pass


class ReLUFactory(ModuleFactory):

    def __init__(self, inplace: 'bool'=False):
        self.inplace = inplace

    def create(self) ->Module:
        return ReLU(self.inplace)


def resolve_nonlinearity_factory(nonlinearity_fatory: 'Optional[ModuleFactory]') ->ModuleFactory:
    if nonlinearity_fatory is None:
        return ReLUFactory(inplace=False)
    else:
        return nonlinearity_fatory


def apply_spectral_norm(module: 'Module', use_spectrial_norm: 'bool'=False) ->Module:
    if use_spectrial_norm:
        return spectral_norm(module)
    else:
        return module


def wrap_conv_or_linear_module(module: 'Module', initialization_method: 'Union[str, Callable[[Module], Module]]', use_spectral_norm: 'bool'):
    if isinstance(initialization_method, str):
        init = create_init_function(initialization_method)
    else:
        init = initialization_method
    return apply_spectral_norm(init(module), use_spectral_norm)


class BlockArgs:

    def __init__(self, initialization_method: 'Union[str, Callable[[Module], Module]]'='he', use_spectral_norm: 'bool'=False, normalization_layer_factory: 'Optional[NormalizationLayerFactory]'=None, nonlinearity_factory: 'Optional[ModuleFactory]'=None):
        self.nonlinearity_factory = resolve_nonlinearity_factory(nonlinearity_factory)
        self.normalization_layer_factory = normalization_layer_factory
        self.use_spectral_norm = use_spectral_norm
        self.initialization_method = initialization_method

    def wrap_module(self, module: 'Module') ->Module:
        return wrap_conv_or_linear_module(module, self.get_init_func(), self.use_spectral_norm)

    def get_init_func(self) ->Callable[[Module], Module]:
        if isinstance(self.initialization_method, str):
            return create_init_function(self.initialization_method)
        else:
            return self.initialization_method

