import sys
_module = sys.modules[__name__]
del sys
setup_actions = _module
test_activeloop_code_analysis = _module
test_activeloop_deeplake = _module
test_activeloop_deeplake_selfquery = _module
test_activeloop_semanitic_search = _module
test_activeloop_twitter = _module
test_code_analysis_deeplake = _module
test_semanitic_search = _module
test_twitter = _module
deeplake = _module
_tensorflow = _module
_torch = _module
core = _module
formats = _module
schemas = _module
storage = _module
tql = _module
types = _module

from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, matplotlib, numbers, numpy, pandas, queue, random, re, scipy, sklearn, string, tensorflow, time, torch, torchaudio, torchvision, types, typing, uuid, warnings
import operator as op
from dataclasses import dataclass
import numpy as np
from torch import Tensor
__version__ = '1.0.0'
xrange = range
wraps = functools.wraps

