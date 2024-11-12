
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


import random


import torch


import torch.nn as nn


import numpy as np


import torch.distributed as dist


from itertools import product


import time


import re


import logging


import collections


import torchvision.datasets as dest


from torchvision import transforms


from torchvision.io import read_image


from torchvision.io.image import ImageReadMode


import torch.nn.functional as F


from scipy.stats import spearmanr


import math


import scipy.stats


import tensorflow as tf


from tensorflow.python import pywrap_tensorflow


import tensorflow.keras.backend as K


import copy


import torch.nn.init as init


from torch.nn.parameter import Parameter


from torch import _C


from torch.cuda import _lazy_call


from torch.cuda import device as device_ctx_manager


from torch.utils.checkpoint import detach_variable


import torch.multiprocessing as mp


from torch.nn.parallel import DistributedDataParallel


from torchvision.utils import make_grid


from torchvision.utils import save_image


from math import sqrt


from math import log


from typing import Optional


from typing import List


from typing import Dict


from typing import Callable


from typing import Iterable


from typing import Tuple


from torch.optim import Optimizer


from torch.optim.lr_scheduler import LambdaLR


class LayerNorm(nn.Module):
    """
    Layer Normalization.
    https://arxiv.org/abs/1607.06450
    """

    def __init__(self, hidden_size, eps=1e-06, eps_inside=False):
        super(LayerNorm, self).__init__()
        self.eps = eps
        self.eps_inside = eps_inside
        self.gamma = nn.Parameter(torch.ones(hidden_size))
        self.beta = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        if self.eps_inside:
            std = torch.sqrt(x.var(-1, keepdim=True) + self.eps)
        else:
            std = x.std(-1, keepdim=True) + self.eps
        hidden_states = self.gamma * (x - mean) / std
        return hidden_states + self.beta


class Embedding(nn.Module):

    def __init__(self, args):
        super(Embedding, self).__init__()
        self.embedding_name_list = []
        self.dropout = nn.Dropout(args.dropout)
        self.remove_embedding_layernorm = args.remove_embedding_layernorm
        if not self.remove_embedding_layernorm and 'dual' not in args.embedding:
            self.layer_norm = LayerNorm(args.emb_size)

    def update(self, embedding, embedding_name):
        setattr(self, embedding_name, embedding)
        self.embedding_name_list.append(embedding_name)

    def forward(self, src, seg):
        if self.embedding_name_list[0] == 'dual':
            return self.dual(src, seg)
        for i, embedding_name in enumerate(self.embedding_name_list):
            embedding = getattr(self, embedding_name)
            if i == 0:
                emb = embedding(src, seg)
            else:
                emb = embedding(src, seg) + emb.clone()
        if not self.remove_embedding_layernorm:
            emb = self.layer_norm(emb)
        emb = self.dropout(emb)
        return emb

