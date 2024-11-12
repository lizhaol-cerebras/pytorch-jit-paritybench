
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


import math


import numpy as np


import logging


import re


import time


import torch


import pandas as pd


import torch.utils.data


import torchvision.transforms as transforms


import torchvision.datasets as datasets


import warnings


import random


import torch.nn as nn


import torch.backends.cudnn as cudnn


from torch.utils.data import DataLoader


from torch.utils.data.sampler import WeightedRandomSampler


import torch.utils.data as data


from torchvision.datasets import CIFAR100


from torchvision.datasets import CIFAR10


import copy


from torch.nn import functional as F


from torchvision.datasets import ImageFolder


from torchvision.datasets import SVHN


from collections import OrderedDict


from torch.autograd import Variable


from tensorflow.keras.preprocessing.image import ImageDataGenerator


from tensorflow.keras.preprocessing.image import DirectoryIterator


import tensorflow as tf


import tensorflow.keras.layers as nn


class PytorchModel(torch.nn.Module):

    def __init__(self):
        super(PytorchModel, self).__init__()
        self.dense = torch.nn.Linear(in_features=1024, out_features=1000, bias=False)

    def forward(self, x):
        x = self.dense(x)
        return x

