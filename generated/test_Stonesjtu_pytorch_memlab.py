import sys
_module = sys.modules[__name__]
del sys
pytorch_memlab = _module
courtesy = _module
line_profiler = _module
extension = _module
line_profiler = _module
line_records = _module
profile = _module
mem_reporter = _module
utils = _module
setup = _module
test = _module
test_courtesy = _module
test_line_profiler = _module
test_mem_reporter = _module

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


import torch


import inspect


from types import FrameType


import warnings


from typing import Any


from typing import Callable


from typing import Optional


from typing import Tuple


from typing import List


from typing import Dict


import pandas as pd


from functools import wraps


import math


from collections import defaultdict


import numpy as np

