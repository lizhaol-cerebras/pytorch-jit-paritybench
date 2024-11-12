
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


from typing import Iterable


from typing import Optional


from typing import Union


import numpy as np


from scipy.sparse import coo_matrix


import torch


from abc import ABCMeta


from abc import abstractmethod


import collections


import random


from typing import Any


from typing import List


from typing import Tuple


import warnings


import pandas as pd


from scipy.sparse import dok_matrix


import math


from typing import Dict


from typing import Callable


from scipy.sparse import csr_matrix


from collections.abc import Iterable


from collections import OrderedDict


from functools import reduce


from functools import partial


from torch import nn


from torch.optim.lr_scheduler import ReduceLROnPlateau


import torch.nn.functional as F


import copy


import inspect


import re


import time


from numpy.testing import assert_almost_equal


from numpy.testing import assert_array_equal


from sklearn.metrics import roc_auc_score


from torch.optim.lr_scheduler import StepLR


class ScaledEmbedding(torch.nn.Embedding):
    """Embedding layer that initializes its values to use a truncated normal distribution."""

    def reset_parameters(self) ->None:
        """Overriding default ``reset_parameters`` method."""
        self.weight.data.normal_(0, 1.0 / (self.embedding_dim * 2.5))


class ZeroEmbedding(torch.nn.Embedding):
    """Embedding layer with weights zeroed-out."""

    def reset_parameters(self) ->None:
        """Overriding default ``reset_parameters`` method."""
        self.weight.data.zero_()

