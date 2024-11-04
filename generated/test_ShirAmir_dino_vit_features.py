import sys
_module = sys.modules[__name__]
del sys
correspondences = _module
cosegmentation = _module
extractor = _module
inspect_similarity = _module
part_cosegmentation = _module
pca = _module

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


import torch


import numpy as np


from sklearn.cluster import KMeans


import matplotlib.pyplot as plt


from matplotlib.colors import ListedColormap


from typing import List


from typing import Tuple


from torchvision import transforms


import torchvision.transforms


from torch import nn


import torch.nn.modules.utils as nn_utils


import math


import types


from typing import Union


import numpy


from sklearn.decomposition import PCA

