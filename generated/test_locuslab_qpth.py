import sys
_module = sys.modules[__name__]
del sys
qpth = _module
qp = _module
solvers = _module
cvxpy = _module
pdipm = _module
batch = _module
single = _module
spbatch = _module
util = _module
setup = _module
test = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, matplotlib, numbers, numpy, pandas, queue, random, re, scipy, sklearn, string, tensorflow, time, torch, torchaudio, torchvision, types, typing, uuid, warnings
import numpy as np
from torch import Tensor
patch_functional()
open = mock_open()
yaml = logging = sys = argparse = MagicMock()
ArgumentParser = argparse.ArgumentParser
_global_config = args = argv = cfg = config = params = _mock_config()
argparse.ArgumentParser.return_value.parse_args.return_value = _global_config
yaml.load.return_value = _global_config
sys.argv = _global_config
__version__ = '1.0.0'
xrange = range
wraps = functools.wraps


import numpy as np


import numpy.random as npr


import itertools


import time


import torch


from torch import nn


from torch.autograd import Variable


from torch.autograd import Function


from enum import Enum


import numpy.testing as npt


from numpy.testing import dec

