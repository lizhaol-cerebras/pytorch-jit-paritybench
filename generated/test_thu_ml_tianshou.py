
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


import numpy as np


import torch


from collections.abc import Callable


from collections.abc import Sequence


from typing import Any


from torch import nn


from torch.distributions import Categorical


from torch.optim.lr_scheduler import LambdaLR


from torch.utils.tensorboard import SummaryWriter


from typing import SupportsFloat


from typing import cast


from torch.distributions import Distribution


from torch.distributions import Independent


from torch.distributions import Normal


from typing import Literal


import copy


from itertools import starmap


from torch.distributions.categorical import Categorical


import numpy.typing as npt


from collections import Counter


from collections.abc import Iterator


from torch.utils.data import DataLoader


from torch.utils.data import Dataset


from torch.utils.data import DistributedSampler


import torch.distributions as dist


import torch.nn as nn


import warnings


from copy import deepcopy


from functools import partial


from collections.abc import Collection


from collections.abc import Iterable


from collections.abc import KeysView


from numbers import Number


from typing import Protocol


from typing import TypeVar


from typing import Union


from typing import overload


from typing import runtime_checkable


import pandas as pd


import logging


import time


from abc import ABC


from abc import abstractmethod


from copy import copy


from typing import Generic


from typing import Optional


from typing import TypedDict


from typing import no_type_check


from typing import TYPE_CHECKING


from torch.optim import Adam


from torch.optim import RMSprop


from torch.optim.lr_scheduler import LRScheduler


from numpy.typing import ArrayLike


import torch.nn.functional as F


from torch.nn.utils import clip_grad_norm_


import math


from torch.distributions import kl_divergence


from matplotlib.figure import Figure


class ScaledObsInputModule(torch.nn.Module):

    def __init__(self, module: 'NetBase', denom: 'float'=255.0) ->None:
        super().__init__()
        self.module = module
        self.denom = denom
        self.output_dim = module.output_dim

    def forward(self, obs: 'np.ndarray | torch.Tensor', state: 'Any | None'=None, info: 'dict[str, Any] | None'=None) ->tuple[torch.Tensor, Any]:
        if info is None:
            info = {}
        return self.module.forward(obs / self.denom, state, info)


class DQN(nn.Module):
    """Reference: Human-level control through deep reinforcement learning.

    For advanced usage (how to customize the network), please refer to
    :ref:`build_the_network`.
    """

    def __init__(self, c: 'int', h: 'int', w: 'int', device: 'str | int | torch.device'='cpu') ->None:
        super().__init__()
        self.device = device
        self.c = c
        self.h = h
        self.w = w
        self.net = nn.Sequential(nn.Conv2d(c, 32, kernel_size=8, stride=4), nn.ReLU(inplace=True), nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(inplace=True), nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(inplace=True), nn.Flatten())
        with torch.no_grad():
            self.output_dim = np.prod(self.net(torch.zeros(1, c, h, w)).shape[1:])

    def forward(self, x: 'np.ndarray | torch.Tensor', state: 'Any | None'=None, info: 'dict[str, Any] | None'=None) ->tuple[torch.Tensor, Any]:
        """Mapping: x -> Q(x, \\*)."""
        if info is None:
            info = {}
        x = torch.as_tensor(x, device=self.device, dtype=torch.float32)
        return self.net(x.reshape(-1, self.c, self.w, self.h)), state


class ProtocolCalledException(Exception):
    """The methods of a Protocol should never be called.

    Currently, no static type checker actually verifies that a class that inherits
    from a Protocol does in fact provide the correct interface. Thus, it may happen
    that a method of the protocol is called accidentally (this is an
    implementation error). The normal error for that is a somewhat cryptic
    AttributeError, wherefore we instead raise this custom exception in the
    BatchProtocol.

    Finally and importantly: using this in BatchProtocol makes mypy verify the fields
    in the various sub-protocols and thus renders is MUCH more useful!
    """


TBatch = TypeVar('TBatch', bound='BatchProtocol')


def _assert_type_keys(keys: 'Iterable[str]') ->None:
    assert all(isinstance(key, str) for key in keys), f'keys should all be string, but got {keys}'

