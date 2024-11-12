
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


import torch.nn as nn


from torch.nn import Module


from torch.nn import Parameter


from torch.nn import functional as F


from torch.autograd import Function


from torch.utils.cpp_extension import BuildExtension


from torch.utils.cpp_extension import CppExtension


from torch.utils.cpp_extension import CUDAExtension


from torch.utils.data import TensorDataset


from torch.autograd import Variable


from torch.autograd import gradcheck


import torch.distributed as dist


from math import ceil


from random import Random


from torch.multiprocessing import Process


from torchvision import datasets


from torchvision import transforms


class DenseFunction(Function):

    @staticmethod
    def forward(ctx, input, weight, bias=None):
        output = input.mm(weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        output = torch.sigmoid(output)
        ctx.save_for_backward(input, weight, bias, output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias, output = ctx.saved_tensors
        grad_sigmoid = (1.0 - output) * output
        grad_output = grad_sigmoid * grad_output
        grad_input = grad_weight = grad_bias = None
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0).squeeze(0)
        return grad_input, grad_weight, grad_bias


class Dense(Module):

    def __init__(self, input_features, output_features, bias=True):
        super(Dense, self).__init__()
        self.input_features = input_features
        self.output_features = output_features
        self.weight = Parameter(torch.Tensor(output_features, input_features))
        if bias:
            self.bias = Parameter(torch.Tensor(output_features))
        else:
            self.register_parameter('bias', None)
        self.weight.data.uniform_(-0.1, 0.1)
        if bias is not None:
            self.bias.data.uniform_(-0.1, 0.1)

    def forward(self, input):
        return DenseFunction.apply(input, self.weight, self.bias)


class Model(Module):

    def __init__(self):
        super(Model, self).__init__()
        self.dense1 = Dense(256, 64)
        self.dense2 = Dense(64, 16)
        self.dense3 = Dense(16, 10)

    def forward(self, x):
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        return F.log_softmax(x, dim=1)


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (Dense,
     lambda: ([], {'input_features': 4, 'output_features': 4}),
     lambda: ([torch.rand([4, 4])], {})),
]

