
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


import time


import torch


import collections


import logging


import math


from torch.nn.modules import Module


from torch.cuda._utils import _get_device_index


import numpy as np


import torch.backends.cudnn as cudnn


import torch.nn.functional as F


import torch.optim as optim


import torch.utils.data.distributed


from torchvision import models


import torch.multiprocessing as mp


import torchvision


import torchvision.transforms as transforms


import torch.nn as nn


import torch.distributed as dist


from torch.nn.parallel import DistributedDataParallel as DDP


from torchvision import datasets


from torchvision import transforms


import random


import warnings


import torch.nn.parallel


import torch.optim


import torch.utils.data


import torchvision.datasets as datasets


import torchvision.models as models


import re


class Compressor(object):
    """Interface for compressing and decompressing a given tensor."""

    @staticmethod
    def compress(tensor):
        """Compresses a tensor and returns it with the context needed to decompress it."""
        pass

    @staticmethod
    def decompress(tensor, ctx):
        """Decompress the tensor with the given context."""
        pass


class FP16Compressor(Compressor):
    """Compress all floating point gradients to 16-bit."""

    @staticmethod
    def compress(tensor):
        """Downcasts the tensor to 16-bit."""
        tensor_compressed = tensor
        if tensor.dtype.is_floating_point:
            tensor_compressed = tensor.type(torch.float16)
        return tensor_compressed, tensor.dtype

    @staticmethod
    def decompress(tensor, ctx):
        """Upcasts the tensor to the initialization dtype."""
        tensor_decompressed = tensor
        dtype = ctx
        if dtype.is_floating_point:
            tensor_decompressed = tensor.type(dtype)
        return tensor_decompressed


class NoneCompressor(Compressor):
    """Default no-op compression."""

    @staticmethod
    def compress(tensor):
        """Returns the tensor unmodified."""
        return tensor, None

    @staticmethod
    def decompress(tensor, ctx):
        """Returns the tensor unmodified."""
        return tensor


class Compression(object):
    """Optional gradient compression algorithm used during push_pull."""
    """Do not compress the gradients. This is the default."""
    none = NoneCompressor
    """Compress all floating point gradients to 16-bit."""
    fp16 = FP16Compressor


def get_ext_suffix():
    """Determine library extension for various versions of Python."""
    ext_suffix = sysconfig.get_config_var('EXT_SUFFIX')
    if ext_suffix:
        return ext_suffix
    ext_suffix = sysconfig.get_config_var('SO')
    if ext_suffix:
        return ext_suffix
    return '.so'


def byteps_push_pull(tensor, version=0, priority=0, name=None, is_average=True):
    """
    A function that performs pushing and pulling tensors

    The operation is keyed by the name. If name is not provided, an
    incremented auto-generated name is used. The tensor type and shape must be
    the same on all BytePS processes for a given name. The reduction will not
    start until all processes are ready to send and receive the tensor.

    This acts as a thin wrapper around an autograd function.  If your input
    tensor requires tensors, then callings this function will allow tensors
    to be computed and backpropagated.

    Arguments:
        tensor: A tensor to average and sum.
        average: A flag indicating whether to compute average or summation,
                 defaults to average.
        name: A name of the reduction operation.

    Returns:
        None
    """
    c_in = tensor.handle
    if isinstance(name, string_types):
        check_call(MXNET_LIB_CTYPES.byteps_mxnet_push_pull_async(c_in, c_str(name), ctypes.c_int(version), ctypes.c_int(priority), ctypes.c_bool(is_average)))
    else:
        check_call(MXNET_LIB_CTYPES.byteps_mxnet_push_pull_async(c_in, name, ctypes.c_int(version), ctypes.c_int(priority), ctypes.c_bool(is_average)))
    return


def byteps_torch_set_num_grads(num_grads_):
    c_lib.byteps_torch_set_num_grads(num_grads_)
    return 0


def declare(name):
    c_lib.byteps_torch_declare_tensor(name.encode())
    return 0


class ConvNet(nn.Module):

    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(7 * 7 * 32, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

