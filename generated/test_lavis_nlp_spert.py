import sys
_module = sys.modules[__name__]
del sys
master = _module
args = _module
config_reader = _module
convert_ade = _module
convert_conll04 = _module
convert_scierc = _module
spert = _module
entities = _module
evaluator = _module
input_reader = _module
loss = _module
models = _module
opt = _module
prediction = _module
sampling = _module
spert_trainer = _module
trainer = _module
util = _module

from paritybench._paritybench_helpers import _mock_config, patch_functional
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


from collections import OrderedDict


from typing import List


from torch.utils.data import Dataset as TorchDataset


import warnings


from typing import Tuple


from typing import Dict


import torch


from sklearn.metrics import precision_recall_fscore_support as prfs


from abc import ABC


from torch import nn as nn


import random


import math


from typing import Type


from torch.nn import DataParallel


from torch.optim import Optimizer


from torch.utils.data import DataLoader


import logging


import numpy as np

