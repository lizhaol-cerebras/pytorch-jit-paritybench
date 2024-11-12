
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


from torch.nn import Module


from torch.nn import Linear


from math import sqrt


import torch.autograd


from torch.nn import Dropout


from torch.nn.init import normal_


from torch.nn import functional as F


from torch.nn import LayerNorm


import numpy as np


from functools import partial


from math import log


import warnings


from torch.nn import ModuleList


import torch.nn.functional as F


import re


from functools import lru_cache


from itertools import dropwhile


import time


import torch.nn as nn


class Event(object):
    """The Event is the base class for all events that are dispatched from any
    transformer module.

    This class defines only the basic attributes of an event without any
    payload.

    Arguments
    ---------
        source: torch.nn.Module instance that dispatched this event
    """

    def __init__(self, source):
        self.source = source

