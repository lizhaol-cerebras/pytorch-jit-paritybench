
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


import numpy as np


import re


from torch.utils.data import Dataset


from torch.utils.data import DataLoader


import torch


import torch.nn as nn


import math


from time import time


from torch.nn.init import xavier_normal


import warnings


from torch.nn import Module


from torch.nn import Parameter


import torch.nn.functional as F


from torch.autograd import Variable


from torch.nn.modules import Module


from torch.nn.parameter import Parameter


from torch.nn.utils.rnn import PackedSequence


from torch.nn import init


def conv(batch_norm, c_in, c_out, ks=3, sd=1):
    if batch_norm:
        return nn.Sequential(nn.Conv2d(c_in, c_out, kernel_size=ks, stride=sd, padding=(ks - 1) // 2, bias=False), nn.BatchNorm2d(c_out), nn.ReLU())
    else:
        return nn.Sequential(nn.Conv2d(c_in, c_out, kernel_size=ks, stride=sd, padding=(ks - 1) // 2, bias=True), nn.ReLU())


def fc(c_in, c_out, activation=False):
    if activation:
        return nn.Sequential(nn.Linear(c_in, c_out), nn.ReLU())
    else:
        return nn.Linear(c_in, c_out)


class Net(nn.Module):

    def __init__(self, batch_norm=False):
        super(Net, self).__init__()
        self.batch_norm = batch_norm
        self.conv1 = conv(self.batch_norm, 6, 64, ks=7, sd=2)
        self.conv2 = conv(self.batch_norm, 64, 128, ks=5, sd=2)
        self.conv3 = conv(self.batch_norm, 128, 256, ks=5, sd=2)
        self.conv3_1 = conv(self.batch_norm, 256, 256)
        self.conv4 = conv(self.batch_norm, 256, 512, sd=2)
        self.conv4_1 = conv(self.batch_norm, 512, 512)
        self.conv5 = conv(self.batch_norm, 512, 512, sd=2)
        self.conv5_1 = conv(self.batch_norm, 512, 512)
        self.conv6 = conv(self.batch_norm, 512, 1024, sd=2)
        self.conv6_1 = conv(self.batch_norm, 1024, 1024)
        self.pool_1 = nn.MaxPool2d(2, stride=2)
        self.dropout1 = nn.Dropout(0.5)
        self.fc1 = fc(1024 * 3 * 10, 4096, activation=True)
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = fc(4096, 1024, activation=True)
        self.fc3 = fc(1024, 128, activation=True)
        self.fc4 = fc(128, 3)
        self.dropout2_1 = nn.Dropout(0.5)
        self.fc2_1 = fc(4096, 1024, activation=True)
        self.fc3_1 = fc(1024, 128, activation=True)
        self.fc4_1 = fc(128, 3)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_normal(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv2(self.conv1(x))
        x = self.conv3_1(self.conv3(x))
        x = self.conv4_1(self.conv4(x))
        x = self.conv5_1(self.conv5(x))
        x = self.conv6_1(self.conv6(x))
        x = self.pool_1(x)
        x = x.view(x.size(0), -1)
        x = self.dropout1(x)
        x = self.fc1(x)
        x_1 = self.dropout2(x)
        x_1 = self.fc2(x_1)
        x_1 = self.fc3(x_1)
        x_1 = self.fc4(x_1)
        x_2 = self.dropout2_1(x)
        x_2 = self.fc2_1(x_2)
        x_2 = self.fc3_1(x_2)
        x_2 = self.fc4_1(x_2)
        x = torch.cat((x_1, x_2), dim=1)
        return x

    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if 'weight' in name]

    def bias_parameters(self):
        return [param for name, param in self.named_parameters() if 'bias' in name]


class RNNCellBase(Module):

    def __repr__(self):
        s = '{name}({input_size}, {hidden_size}'
        if 'bias' in self.__dict__ and self.bias is not True:
            s += ', bias={bias}'
        if 'nonlinearity' in self.__dict__ and self.nonlinearity != 'tanh':
            s += ', nonlinearity={nonlinearity}'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)


def clip_grad(v, min, max):
    v.register_hook(lambda g: g.clamp(min, max))
    return v


class RNNCell(RNNCellBase):

    def __init__(self, input_size, hidden_size, bias=True, grad_clip=None):
        super(RNNCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.grad_clip = grad_clip
        self.weight_ih = Parameter(torch.Tensor(hidden_size, input_size))
        self.weight_hh = Parameter(torch.Tensor(hidden_size, hidden_size))
        if bias:
            self.bias = Parameter(torch.Tensor(hidden_size))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, input, h):
        output = F.linear(input, self.weight_ih, self.bias) + F.linear(h, self.weight_hh)
        if self.grad_clip:
            output = clip_grad(output, -self.grad_clip, self.grad_clip)
        output = F.relu(output)
        return output


class GRUCell(RNNCellBase):

    def __init__(self, input_size, hidden_size, bias=True, grad_clip=None):
        super(GRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.grad_clip = grad_clip
        self.weight_ih = Parameter(torch.Tensor(3 * hidden_size, input_size))
        self.weight_hh_rz = Parameter(torch.Tensor(2 * hidden_size, hidden_size))
        self.weight_hh = Parameter(torch.Tensor(hidden_size, hidden_size))
        if bias:
            self.bias = Parameter(torch.Tensor(3 * hidden_size))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, input, h):
        ih = F.linear(input, self.weight_ih, self.bias)
        hh_rz = F.linear(h, self.weight_hh_rz)
        if self.grad_clip:
            ih = clip_grad(ih, -self.grad_clip, self.grad_clip)
            hh_rz = clip_grad(hh_rz, -self.grad_clip, self.grad_clip)
        r = F.sigmoid(ih[:, :self.hidden_size] + hh_rz[:, :self.hidden_size])
        i = F.sigmoid(ih[:, self.hidden_size:self.hidden_size * 2] + hh_rz[:, self.hidden_size:])
        hhr = F.linear(h * r, self.weight_hh)
        if self.grad_clip:
            hhr = clip_grad(hhr, -self.grad_clip, self.grad_clip)
        n = F.relu(ih[:, self.hidden_size * 2:] + hhr)
        h = (1 - i) * n + i * h
        return h


class LSTMCell(RNNCellBase):

    def __init__(self, input_size, hidden_size, bias=True, grad_clip=None):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.grad_clip = grad_clip
        self.weight_ih = Parameter(torch.Tensor(4 * hidden_size, input_size))
        self.weight_hh = Parameter(torch.Tensor(4 * hidden_size, hidden_size))
        if bias:
            self.bias = Parameter(torch.Tensor(4 * hidden_size))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, input, hx):
        h, c = hx
        pre = F.linear(input, self.weight_ih, self.bias) + F.linear(h, self.weight_hh)
        if self.grad_clip:
            pre = clip_grad(pre, -self.grad_clip, self.grad_clip)
        i = F.sigmoid(pre[:, :self.hidden_size])
        f = F.sigmoid(pre[:, self.hidden_size:self.hidden_size * 2])
        g = F.relu(pre[:, self.hidden_size * 2:self.hidden_size * 3])
        o = F.sigmoid(pre[:, self.hidden_size * 3:])
        c = f * c + i * g
        h = o * F.relu(c)
        return h, c


class LSTMPCell(RNNCellBase):

    def __init__(self, input_size, hidden_size, recurrent_size, bias=True, grad_clip=None):
        super(LSTMPCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.recurrent_size = recurrent_size
        self.grad_clip = grad_clip
        self.weight_ih = Parameter(torch.Tensor(4 * hidden_size, input_size))
        self.weight_hh = Parameter(torch.Tensor(4 * hidden_size, recurrent_size))
        self.weight_rec = Parameter(torch.Tensor(recurrent_size, hidden_size))
        if bias:
            self.bias = Parameter(torch.Tensor(4 * hidden_size))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, input, hx):
        h, c = hx
        pre = F.linear(input, self.weight_ih, self.bias) + F.linear(h, self.weight_hh)
        if self.grad_clip:
            pre = clip_grad(pre, -self.grad_clip, self.grad_clip)
        i = F.sigmoid(pre[:, :self.hidden_size])
        f = F.sigmoid(pre[:, self.hidden_size:self.hidden_size * 2])
        g = F.tanh(pre[:, self.hidden_size * 2:self.hidden_size * 3])
        o = F.sigmoid(pre[:, self.hidden_size * 3:])
        c = f * c + i * g
        h = o * F.tanh(c)
        h = F.linear(h, self.weight_rec)
        return h, c


class MGRUCell(RNNCellBase):
    """Minimal GRU
    Reference:
    Ravanelli et al. [Improving speech recognition by revising gated recurrent units](https://arxiv.org/abs/1710.00641).
    """

    def __init__(self, input_size, hidden_size, bias=True, grad_clip=None):
        super(MGRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.grad_clip = grad_clip
        self.weight_ih = Parameter(torch.Tensor(2 * hidden_size, input_size))
        self.weight_hh = Parameter(torch.Tensor(2 * hidden_size, hidden_size))
        if bias:
            self.bias = Parameter(torch.Tensor(2 * hidden_size))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, input, h):
        ih = F.linear(input, self.weight_ih, self.bias)
        hh = F.linear(h, self.weight_hh)
        if self.grad_clip:
            ih = clip_grad(ih, -self.grad_clip, self.grad_clip)
            hh = clip_grad(hh, -self.grad_clip, self.grad_clip)
        z = F.sigmoid(ih[:, :self.hidden_size] + hh[:, :self.hidden_size])
        n = F.relu(ih[:, self.hidden_size:] + hh[:, self.hidden_size:])
        h = (1 - z) * n + z * h
        return h


class RNNBase(Module):

    def __init__(self, mode, input_size, hidden_size, num_layers=1, bias=True, batch_first=False, dropout=0, bidirectional=False):
        super(RNNBase, self).__init__()
        self.mode = mode
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.dropout_state = {}
        self.bidirectional = bidirectional
        num_directions = 2 if bidirectional else 1
        if mode == 'LSTM':
            gate_size = 4 * hidden_size
        elif mode == 'GRU':
            gate_size = 3 * hidden_size
        else:
            gate_size = hidden_size
        self._all_weights = []
        for layer in range(num_layers):
            for direction in range(num_directions):
                layer_input_size = input_size if layer == 0 else hidden_size * num_directions
                w_ih = Parameter(torch.Tensor(gate_size, layer_input_size))
                w_hh = Parameter(torch.Tensor(gate_size, hidden_size))
                b_ih = Parameter(torch.Tensor(gate_size))
                b_hh = Parameter(torch.Tensor(gate_size))
                layer_params = w_ih, w_hh, b_ih, b_hh
                suffix = '_reverse' if direction == 1 else ''
                param_names = ['weight_ih_l{}{}', 'weight_hh_l{}{}']
                if bias:
                    param_names += ['bias_ih_l{}{}', 'bias_hh_l{}{}']
                param_names = [x.format(layer, suffix) for x in param_names]
                for name, param in zip(param_names, layer_params):
                    setattr(self, name, param)
                self._all_weights.append(param_names)
        self.flatten_parameters()
        self.reset_parameters()

    def flatten_parameters(self):
        """Resets parameter data pointer so that they can use faster code paths.

        Right now, this works only if the module is on the GPU and cuDNN is enabled.
        Otherwise, it's a no-op.
        """
        any_param = next(self.parameters()).data
        if not any_param.is_cuda or not torch.backends.cudnn.is_acceptable(any_param):
            self._data_ptrs = []
            return
        unique_data_ptrs = set(p.data_ptr() for l in self.all_weights for p in l)
        if len(unique_data_ptrs) != sum(len(l) for l in self.all_weights):
            self._data_ptrs = []
            return
        with torch.cuda.device_of(any_param):
            from torch.backends.cudnn import rnn
            from torch.backends import cudnn
            from torch.nn._functions.rnn import CudnnRNN
            handle = cudnn.get_handle()
            with warnings.catch_warnings(record=True):
                fn = CudnnRNN(self.mode, self.input_size, self.hidden_size, num_layers=self.num_layers, batch_first=self.batch_first, dropout=self.dropout, train=self.training, bidirectional=self.bidirectional, dropout_state=self.dropout_state)
            fn.datatype = cudnn._typemap[any_param.type()]
            fn.x_descs = cudnn.descriptor(any_param.new(1, self.input_size), 1)
            fn.rnn_desc = rnn.init_rnn_descriptor(fn, handle)
            self._param_buf_size = rnn.get_num_weights(handle, fn.rnn_desc, fn.x_descs[0], fn.datatype)
            fn.weight_buf = any_param.new(self._param_buf_size).zero_()
            fn.w_desc = rnn.init_weight_descriptor(fn, fn.weight_buf)
            all_weights = [[p.data for p in l] for l in self.all_weights]
            params = rnn.get_parameters(fn, handle, fn.weight_buf)
            rnn._copyParams(all_weights, params)
            for orig_layer_param, new_layer_param in zip(all_weights, params):
                for orig_param, new_param in zip(orig_layer_param, new_layer_param):
                    orig_param.set_(new_param.view_as(orig_param))
            self._data_ptrs = list(p.data.data_ptr() for p in self.parameters())

    def _apply(self, fn):
        ret = super(RNNBase, self)._apply(fn)
        self.flatten_parameters()
        return ret

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, input, hx=None):
        is_packed = isinstance(input, PackedSequence)
        if is_packed:
            input, batch_sizes = input
            max_batch_size = batch_sizes[0]
        else:
            batch_sizes = None
            max_batch_size = input.size(0) if self.batch_first else input.size(1)
        if hx is None:
            num_directions = 2 if self.bidirectional else 1
            hx = torch.autograd.Variable(input.data.new(self.num_layers * num_directions, max_batch_size, self.hidden_size).zero_(), requires_grad=False)
            if self.mode == 'LSTM':
                hx = hx, hx
        has_flat_weights = list(p.data.data_ptr() for p in self.parameters()) == self._data_ptrs
        if has_flat_weights:
            first_data = next(self.parameters()).data
            assert first_data.storage().size() == self._param_buf_size
            flat_weight = first_data.new().set_(first_data.storage(), 0, torch.Size([self._param_buf_size]))
        else:
            flat_weight = None
        h, c = hx
        pre = F.linear(input, self.w_ih, self.bias) + F.linear(h, self.weight_hh)
        i = F.sigmoid(pre[:, :self.hidden_size])
        f = F.sigmoid(pre[:, self.hidden_size:self.hidden_size * 2])
        g = F.relu(pre[:, self.hidden_size * 2:self.hidden_size * 3])
        o = F.sigmoid(pre[:, self.hidden_size * 3:])
        c = f * c + i * g
        h = o * F.relu(c)
        return h, c

    def __repr__(self):
        s = '{name}({input_size}, {hidden_size}'
        if self.num_layers != 1:
            s += ', num_layers={num_layers}'
        if self.bias is not True:
            s += ', bias={bias}'
        if self.batch_first is not False:
            s += ', batch_first={batch_first}'
        if self.dropout != 0:
            s += ', dropout={dropout}'
        if self.bidirectional is not False:
            s += ', bidirectional={bidirectional}'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)

    def __setstate__(self, d):
        super(RNNBase, self).__setstate__(d)
        self.__dict__.setdefault('_data_ptrs', [])
        if 'all_weights' in d:
            self._all_weights = d['all_weights']
        if isinstance(self._all_weights[0][0], str):
            return
        num_layers = self.num_layers
        num_directions = 2 if self.bidirectional else 1
        self._all_weights = []
        for layer in range(num_layers):
            for direction in range(num_directions):
                suffix = '_reverse' if direction == 1 else ''
                weights = ['weight_ih_l{}{}', 'weight_hh_l{}{}', 'bias_ih_l{}{}', 'bias_hh_l{}{}']
                weights = [x.format(layer, suffix) for x in weights]
                if self.bias:
                    self._all_weights += [weights]
                else:
                    self._all_weights += [weights[:2]]

    @property
    def all_weights(self):
        return [[getattr(self, weight) for weight in weights] for weights in self._all_weights]


class RNN(RNNBase):

    def __init__(self, *args, **kwargs):
        super(RNN, self).__init__('RNN', *args, **kwargs)


class GRU(RNNBase):

    def __init__(self, *args, **kwargs):
        super(GRU, self).__init__('GRU', *args, **kwargs)


class MGRU(RNNBase):

    def __init__(self, *args, **kwargs):
        super(MGRU, self).__init__('MGRU', *args, **kwargs)


class LSTM(RNNBase):
    """Applies a multi-layer long short-term memory (LSTM) RNN to an input
    sequence.
    """

    def __init__(self, *args, **kwargs):
        super(LSTM, self).__init__('LSTM', *args, **kwargs)


class LSTMP(RNNBase):

    def __init__(self, *args, **kwargs):
        super(LSTMP, self).__init__('LSTMP', *args, **kwargs)


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (GRUCell,
     lambda: ([], {'input_size': 4, 'hidden_size': 4}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4])], {})),
    (MGRUCell,
     lambda: ([], {'input_size': 4, 'hidden_size': 4}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4])], {})),
    (RNNCell,
     lambda: ([], {'input_size': 4, 'hidden_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
]

