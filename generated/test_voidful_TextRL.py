import sys
_module = sys.modules[__name__]
del sys
setup = _module
textrl = _module
actor = _module
dump = _module
environment = _module

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


import copy


import torch


import torch.nn.functional as F


import random


import numpy


class SoftmaxCategoricalHead(torch.nn.Module):

    def forward(self, logits, temperature=0.1):
        softmax = torch.nn.Softmax(dim=1)
        return torch.distributions.Categorical(probs=softmax(logits / temperature))


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (SoftmaxCategoricalHead,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_voidful_TextRL(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

