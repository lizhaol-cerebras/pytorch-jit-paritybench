
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


from functools import partial


from time import perf_counter


from typing import List


import torch


from typing import Tuple


from typing import Literal


from typing import Callable


from typing import Generator


import re


import pandas as pd


from functools import wraps


import numpy as np


from typing import Dict


from typing import Optional


from typing import Union


import torch.nn.functional as F


from torch import nn


from torchvision.transforms import CenterCrop


from torchvision.transforms import Compose


from torchvision.transforms import InterpolationMode


from torchvision.transforms import Normalize


from torchvision.transforms import RandomResizedCrop


from torchvision.transforms import Resize


from torchvision.transforms import ToTensor


from typing import Mapping


from typing import Any


import torch.nn as nn


from torch import Tensor


from typing import Sequence


class ImageFeaturesPooler(nn.Module):

    def __init__(self, input_size, hidden_size, num_attn_heads, intermediate_size, num_latents, initializer_range):
        super().__init__()
        self.projection = nn.Linear(input_size, hidden_size)
        self.pooler = nn.TransformerDecoderLayer(hidden_size, num_attn_heads, intermediate_size, activation=nn.functional.silu, batch_first=True, norm_first=True)
        self.image_latents = nn.Parameter(torch.randn(1, num_latents, hidden_size) * initializer_range ** 0.5)

    def forward(self, features):
        features = self.projection(features)
        return self.pooler(self.image_latents.expand(features.shape[0], -1, -1), features)


def _is_on_gpu(model: 'nn.Module') ->bool:
    try:
        return next(model.parameters()).device.type == 'cuda'
    except StopIteration:
        return False


def read_config(path_or_object: 'ConfigOrPath') ->object:
    if isinstance(path_or_object, (PathLike, str)):
        with open(path_or_object, 'r') as f:
            return json.load(f)
    else:
        return path_or_object


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (ImageFeaturesPooler,
     lambda: ([], {'input_size': 4, 'hidden_size': 4, 'num_attn_heads': 4, 'intermediate_size': 4, 'num_latents': 4, 'initializer_range': 4}),
     lambda: ([torch.rand([4, 4, 4])], {})),
]

