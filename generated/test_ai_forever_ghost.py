
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


import torch


import torch.nn as nn


from torch.autograd import Variable


import torch.nn.functional as F


import math


import warnings


import functools


import numpy as np


from types import MethodType


import types


import itertools


from collections import OrderedDict


import torch.nn.functional


from itertools import product


from torch import nn


from torch.nn.modules.batchnorm import _BatchNorm


from torch.nn import init


from torch.nn import Parameter


import torch.distributed.distributed_c10d as c10d


from torch.optim.optimizer import Optimizer


from torch.optim.optimizer import required


import collections


from itertools import permutations


import random


import time


from torch.nn.parameter import Parameter


from torch._utils import _flatten_dense_tensors


from torch._utils import _unflatten_dense_tensors


from copy import copy


import numbers


from torch.nn import functional as F


import torch.distributed as dist


from torch.nn.modules import Module


from itertools import chain


import copy


from torch.autograd.function import Function


import torch.cuda.profiler as profiler


import torchvision.models as models


import torch.optim as optim


import inspect


import torch.cuda.nvtx as nvtx


import numpy


import inspect as ins


from abc import ABC


from abc import abstractmethod


import torch.nn.parallel


import torch.backends.cudnn as cudnn


import torch.utils.data


import torchvision.datasets as dset


import torchvision.transforms as transforms


import torchvision.utils as vutils


import torch.optim


import torch.utils.data.distributed


import torchvision.datasets as datasets


from torch.utils import cpp_extension


import functools as ft


import itertools as it


from math import floor


from time import time


from torch.nn.parallel import DistributedDataParallel as DDP


from torch.optim import Optimizer


from torch.nn import Module


import torch.nn.init as init


from torch.nn.parallel.data_parallel import DataParallel


import torchvision


import torch.nn.utils.spectral_norm as spectral_norm


import re


from torch.utils.data import DataLoader


import torch.optim.lr_scheduler as scheduler


from typing import List


from typing import Tuple


from typing import Callable


from typing import Any


from matplotlib import pyplot as plt


from scipy.spatial import distance


from torch.utils.data import Dataset


from torch.utils.data import TensorDataset


class RNNCell(nn.Module):
    """ 
    RNNCell 
    gate_multiplier is related to the architecture you're working with
    For LSTM-like it will be 4 and GRU-like will be 3.
    Always assumes input is NOT batch_first.
    Output size that's not hidden size will use output projection
    Hidden_states is number of hidden states that are needed for cell
    if one will go directly to cell as tensor, if more will go as list
    """

    def __init__(self, gate_multiplier, input_size, hidden_size, cell, n_hidden_states=2, bias=False, output_size=None):
        super(RNNCell, self).__init__()
        self.gate_multiplier = gate_multiplier
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.cell = cell
        self.bias = bias
        self.output_size = output_size
        if output_size is None:
            self.output_size = hidden_size
        self.gate_size = gate_multiplier * self.hidden_size
        self.n_hidden_states = n_hidden_states
        self.w_ih = nn.Parameter(torch.Tensor(self.gate_size, self.input_size))
        self.w_hh = nn.Parameter(torch.Tensor(self.gate_size, self.output_size))
        if self.output_size != self.hidden_size:
            self.w_ho = nn.Parameter(torch.Tensor(self.output_size, self.hidden_size))
        self.b_ih = self.b_hh = None
        if self.bias:
            self.b_ih = nn.Parameter(torch.Tensor(self.gate_size))
            self.b_hh = nn.Parameter(torch.Tensor(self.gate_size))
        self.hidden = [None for states in range(self.n_hidden_states)]
        self.reset_parameters()

    def new_like(self, new_input_size=None):
        """
        new_like()
        """
        if new_input_size is None:
            new_input_size = self.input_size
        return type(self)(self.gate_multiplier, new_input_size, self.hidden_size, self.cell, self.n_hidden_states, self.bias, self.output_size)

    def reset_parameters(self, gain=1):
        """
        reset_parameters()
        """
        stdev = 1.0 / math.sqrt(self.hidden_size)
        for param in self.parameters():
            param.data.uniform_(-stdev, stdev)
    """
    Xavier reset:
    def reset_parameters(self, gain=1):
        stdv = 1.0 / math.sqrt(self.gate_size)

        for param in self.parameters():
            if (param.dim() > 1):
                torch.nn.init.xavier_normal(param, gain)
            else:
                param.data.uniform_(-stdv, stdv)
    """

    def init_hidden(self, bsz):
        """
        init_hidden()
        """
        for param in self.parameters():
            if param is not None:
                a_param = param
                break
        for i, _ in enumerate(self.hidden):
            if self.hidden[i] is None or self.hidden[i].data.size()[0] != bsz:
                if i == 0:
                    hidden_size = self.output_size
                else:
                    hidden_size = self.hidden_size
                tens = a_param.data.new(bsz, hidden_size).zero_()
                self.hidden[i] = Variable(tens, requires_grad=False)

    def reset_hidden(self, bsz):
        """
        reset_hidden()
        """
        for i, _ in enumerate(self.hidden):
            self.hidden[i] = None
        self.init_hidden(bsz)

    def detach_hidden(self):
        """
        detach_hidden()
        """
        for i, _ in enumerate(self.hidden):
            if self.hidden[i] is None:
                raise RuntimeError('Must initialize hidden state before you can detach it')
        for i, _ in enumerate(self.hidden):
            self.hidden[i] = self.hidden[i].detach()

    def forward(self, input):
        """
        forward()
        if not inited or bsz has changed this will create hidden states
        """
        self.init_hidden(input.size()[0])
        hidden_state = self.hidden[0] if self.n_hidden_states == 1 else self.hidden
        self.hidden = self.cell(input, hidden_state, self.w_ih, self.w_hh, b_ih=self.b_ih, b_hh=self.b_hh)
        if self.n_hidden_states > 1:
            self.hidden = list(self.hidden)
        else:
            self.hidden = [self.hidden]
        if self.output_size != self.hidden_size:
            self.hidden[0] = F.linear(self.hidden[0], self.w_ho)
        return tuple(self.hidden)


def is_iterable(maybe_iterable):
    return isinstance(maybe_iterable, list) or isinstance(maybe_iterable, tuple)


def flatten_list(tens_list):
    """
    flatten_list
    """
    if not is_iterable(tens_list):
        return tens_list
    return torch.cat(tens_list, dim=0).view(len(tens_list), *tens_list[0].size())


class stackedRNN(nn.Module):
    """
    stackedRNN
    """

    def __init__(self, inputRNN, num_layers=1, dropout=0):
        super(stackedRNN, self).__init__()
        self.dropout = dropout
        if isinstance(inputRNN, RNNCell):
            self.rnns = [inputRNN]
            for i in range(num_layers - 1):
                self.rnns.append(inputRNN.new_like(inputRNN.output_size))
        elif isinstance(inputRNN, list):
            assert len(inputRNN) == num_layers, 'RNN list length must be equal to num_layers'
            self.rnns = inputRNN
        else:
            raise RuntimeError()
        self.nLayers = len(self.rnns)
        self.rnns = nn.ModuleList(self.rnns)
    """
    Returns output as hidden_state[0] Tensor([sequence steps][batch size][features])
    If collect hidden will also return Tuple(
        [n_hidden_states][sequence steps] Tensor([layer][batch size][features])
    )
    If not collect hidden will also return Tuple(
        [n_hidden_states] Tensor([layer][batch size][features])
    """

    def forward(self, input, collect_hidden=False, reverse=False):
        """
        forward()
        """
        seq_len = input.size(0)
        bsz = input.size(1)
        inp_iter = reversed(range(seq_len)) if reverse else range(seq_len)
        hidden_states = [[] for i in range(self.nLayers)]
        outputs = []
        for seq in inp_iter:
            for layer in range(self.nLayers):
                if layer == 0:
                    prev_out = input[seq]
                outs = self.rnns[layer](prev_out)
                if collect_hidden:
                    hidden_states[layer].append(outs)
                elif seq == seq_len - 1:
                    hidden_states[layer].append(outs)
                prev_out = outs[0]
            outputs.append(prev_out)
        if reverse:
            outputs = list(reversed(outputs))
        """
        At this point outputs is in format:
        list( [seq_length] x Tensor([bsz][features]) )
        need to convert it to:
        list( Tensor([seq_length][bsz][features]) )
        """
        output = flatten_list(outputs)
        """
        hidden_states at this point is in format:
        list( [layer][seq_length][hidden_states] x Tensor([bsz][features]) )
        need to convert it to:
          For not collect hidden:
            list( [hidden_states] x Tensor([layer][bsz][features]) )
          For collect hidden:
            list( [hidden_states][seq_length] x Tensor([layer][bsz][features]) )
        """
        if not collect_hidden:
            seq_len = 1
        n_hid = self.rnns[0].n_hidden_states
        new_hidden = [[[None for k in range(self.nLayers)] for j in range(seq_len)] for i in range(n_hid)]
        for i in range(n_hid):
            for j in range(seq_len):
                for k in range(self.nLayers):
                    new_hidden[i][j][k] = hidden_states[k][j][i]
        hidden_states = new_hidden
        if reverse:
            hidden_states = list(list(reversed(list(entry))) for entry in hidden_states)
        hiddens = list(list(flatten_list(seq) for seq in hidden) for hidden in hidden_states)
        if not collect_hidden:
            hidden_states = list(entry[0] for entry in hidden_states)
        return output, hidden_states

    def reset_parameters(self):
        """
        reset_parameters()
        """
        for rnn in self.rnns:
            rnn.reset_parameters()

    def init_hidden(self, bsz):
        """
        init_hidden()
        """
        for rnn in self.rnns:
            rnn.init_hidden(bsz)

    def detach_hidden(self):
        """
        detach_hidden()
        """
        for rnn in self.rnns:
            rnn.detach_hidden()

    def reset_hidden(self, bsz):
        """
        reset_hidden()
        """
        for rnn in self.rnns:
            rnn.reset_hidden(bsz)

    def init_inference(self, bsz):
        """ 
        init_inference()
        """
        for rnn in self.rnns:
            rnn.init_inference(bsz)


class bidirectionalRNN(nn.Module):
    """
    bidirectionalRNN
    """

    def __init__(self, inputRNN, num_layers=1, dropout=0):
        super(bidirectionalRNN, self).__init__()
        self.dropout = dropout
        self.fwd = stackedRNN(inputRNN, num_layers=num_layers, dropout=dropout)
        self.bckwrd = stackedRNN(inputRNN.new_like(), num_layers=num_layers, dropout=dropout)
        self.rnns = nn.ModuleList([self.fwd, self.bckwrd])

    def forward(self, input, collect_hidden=False):
        """
        forward()
        """
        seq_len = input.size(0)
        bsz = input.size(1)
        fwd_out, fwd_hiddens = list(self.fwd(input, collect_hidden=collect_hidden))
        bckwrd_out, bckwrd_hiddens = list(self.bckwrd(input, reverse=True, collect_hidden=collect_hidden))
        output = torch.cat([fwd_out, bckwrd_out], -1)
        hiddens = tuple(torch.cat(hidden, -1) for hidden in zip(fwd_hiddens, bckwrd_hiddens))
        return output, hiddens

    def reset_parameters(self):
        """
        reset_parameters()
        """
        for rnn in self.rnns:
            rnn.reset_parameters()

    def init_hidden(self, bsz):
        """
        init_hidden()
        """
        for rnn in self.rnns:
            rnn.init_hidden(bsz)

    def detach_hidden(self):
        """
        detach_hidden()
        """
        for rnn in self.rnns:
            rnn.detachHidden()

    def reset_hidden(self, bsz):
        """
        reset_hidden()
        """
        for rnn in self.rnns:
            rnn.reset_hidden(bsz)

    def init_inference(self, bsz):
        """
        init_inference()
        """
        for rnn in self.rnns:
            rnn.init_inference(bsz)


def mLSTMCell(input, hidden, w_ih, w_hh, w_mih, w_mhh, b_ih=None, b_hh=None):
    """
    mLSTMCell
    """
    if input.is_cuda:
        igates = F.linear(input, w_ih)
        m = F.linear(input, w_mih) * F.linear(hidden[0], w_mhh)
        hgates = F.linear(m, w_hh)
        state = fusedBackend.LSTMFused.apply
        return state(igates, hgates, hidden[1], b_ih, b_hh)
    hx, cx = hidden
    m = F.linear(input, w_mih) * F.linear(hidden[0], w_mhh)
    gates = F.linear(input, w_ih, b_ih) + F.linear(m, w_hh, b_hh)
    ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)
    ingate = F.sigmoid(ingate)
    forgetgate = F.sigmoid(forgetgate)
    cellgate = F.tanh(cellgate)
    outgate = F.sigmoid(outgate)
    cy = forgetgate * cx + ingate * cellgate
    hy = outgate * F.tanh(cy)
    return hy, cy


class mLSTMRNNCell(RNNCell):
    """
    mLSTMRNNCell
    """

    def __init__(self, input_size, hidden_size, bias=False, output_size=None):
        gate_multiplier = 4
        super(mLSTMRNNCell, self).__init__(gate_multiplier, input_size, hidden_size, mLSTMCell, n_hidden_states=2, bias=bias, output_size=output_size)
        self.w_mih = nn.Parameter(torch.Tensor(self.output_size, self.input_size))
        self.w_mhh = nn.Parameter(torch.Tensor(self.output_size, self.output_size))
        self.reset_parameters()

    def forward(self, input):
        """
        mLSTMRNNCell.forward()
        """
        self.init_hidden(input.size()[0])
        hidden_state = self.hidden[0] if self.n_hidden_states == 1 else self.hidden
        self.hidden = list(self.cell(input, hidden_state, self.w_ih, self.w_hh, self.w_mih, self.w_mhh, b_ih=self.b_ih, b_hh=self.b_hh))
        if self.output_size != self.hidden_size:
            self.hidden[0] = F.linear(self.hidden[0], self.w_ho)
        return tuple(self.hidden)

    def new_like(self, new_input_size=None):
        if new_input_size is None:
            new_input_size = self.input_size
        return type(self)(new_input_size, self.hidden_size, self.bias, self.output_size)


class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer('weight', torch.ones(n))
        self.register_buffer('bias', torch.zeros(n))
        self.register_buffer('running_mean', torch.zeros(n))
        self.register_buffer('running_var', torch.ones(n))

    def get_scale_bias(self, nhwc=False):
        scale = self.weight * self.running_var.rsqrt()
        bias = self.bias - self.running_mean * scale
        if nhwc:
            scale = scale.reshape(1, 1, 1, -1)
            bias = bias.reshape(1, 1, 1, -1)
        else:
            scale = scale.reshape(1, -1, 1, 1)
            bias = bias.reshape(1, -1, 1, 1)
        return scale, bias

    def forward(self, x):
        scale, bias = self.get_scale_bias()
        return x * scale + bias


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class FMHAFun(torch.autograd.Function):

    @staticmethod
    def forward(ctx, qkv, cu_seqlens, seqlens, p_dropout, max_s, is_training):
        context, S_dmask = mha.fwd(qkv, cu_seqlens, seqlens, p_dropout, max_s, is_training, None)
        ctx.save_for_backward(qkv, S_dmask)
        ctx.cu_seqlens = cu_seqlens
        ctx.seqlens = seqlens
        ctx.p_dropout = p_dropout
        ctx.max_s = max_s
        return context

    @staticmethod
    def backward(ctx, dout):
        qkv, S_dmask = ctx.saved_tensors
        dqkv, dp = mha.bwd(dout, qkv, S_dmask, ctx.cu_seqlens, ctx.seqlens, ctx.p_dropout, ctx.max_s)
        return dqkv, None, None, None, None, None, None


class FMHA(torch.nn.Module):

    def __init__(self, config):
        super(FMHA, self).__init__()
        self.p_dropout = config.attention_probs_dropout_prob
        self.h = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.d = self.hidden_size // self.h
        assert self.d * self.h == self.hidden_size, 'Invalid hidden size/num_heads'

    def forward(self, qkv, cu_seqlens, seqlens, max_s, is_training=True):
        ctx = FMHAFun.apply(qkv.view(-1, 3, self.h, self.d), cu_seqlens, seqlens, self.p_dropout, max_s, is_training)
        return ctx.view(-1, self.hidden_size)


class bn_NHWC_impl(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, s, b, rm, riv, mini_m, mini_riv, ret_cta, mom, epsilon, fuse_relu, is_train, bn_group, my_data, pair_data, magic, pair_data2, pair_data3, fwd_occup, fwd_grid_x, bwd_occup, bwd_grid_x, multi_stream):
        if is_train:
            ctx.save_for_backward(x, s, b, rm, riv, mini_m, mini_riv)
            ctx.epsilon = epsilon
            ctx.momentum = mom
            ctx.ret_cta = ret_cta
            ctx.fuse_relu = fuse_relu
            ctx.my_data = my_data
            ctx.pair_data = pair_data
            ctx.magic = magic
            ctx.pair_data2 = pair_data2
            ctx.pair_data3 = pair_data3
            ctx.bn_group = bn_group
            ctx.bwd_occup = bwd_occup
            ctx.bwd_grid_x = bwd_grid_x
            ctx.multi_stream = multi_stream
            res = bnp.bn_fwd_nhwc(x, s, b, rm, riv, mini_m, mini_riv, ret_cta, mom, epsilon, fuse_relu, my_data, pair_data, pair_data2, pair_data3, bn_group, magic, fwd_occup, fwd_grid_x, multi_stream)
            return res
        else:
            return bnp.bn_fwd_eval_nhwc(x, s, b, rm, riv, ret_cta, bn_group, mom, epsilon, fuse_relu)

    @staticmethod
    def backward(ctx, grad_y):
        x, s, b, rm, riv, mini_m, mini_riv = ctx.saved_variables
        epsilon = ctx.epsilon
        mom = ctx.momentum
        ret_cta = ctx.ret_cta
        fuse_relu = ctx.fuse_relu
        my_data = ctx.my_data
        pair_data = ctx.pair_data
        magic = ctx.magic
        pair_data2 = ctx.pair_data2
        pair_data3 = ctx.pair_data3
        bn_group = ctx.bn_group
        bwd_occup = ctx.bwd_occup
        bwd_grid_x = ctx.bwd_grid_x
        multi_stream = ctx.multi_stream
        dx, dscale, dbias = bnp.bn_bwd_nhwc(x, grad_y, s, b, rm, riv, mini_m, mini_riv, ret_cta, mom, epsilon, fuse_relu, my_data, pair_data, pair_data2, pair_data3, bn_group, magic, bwd_occup, bwd_grid_x, multi_stream)
        return dx, dscale, dbias, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None


class bn_addrelu_NHWC_impl(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, z, s, b, rm, riv, mini_m, mini_riv, grid_dim_y, ret_cta, mom, epsilon, is_train, bn_group, my_data, pair_data, magic, pair_data2, pair_data3, fwd_occup, fwd_grid_x, bwd_occup, bwd_grid_x, multi_stream):
        if is_train:
            bitmask = torch.IntTensor((x.numel() + 31) // 32 * 2 * grid_dim_y)
            ctx.save_for_backward(x, s, b, rm, riv, mini_m, mini_riv, bitmask)
            ctx.epsilon = epsilon
            ctx.momentum = mom
            ctx.ret_cta = ret_cta
            ctx.my_data = my_data
            ctx.pair_data = pair_data
            ctx.magic = magic
            ctx.pair_data2 = pair_data2
            ctx.pair_data3 = pair_data3
            ctx.bn_group = bn_group
            ctx.bwd_occup = bwd_occup
            ctx.bwd_grid_x = bwd_grid_x
            ctx.multi_stream = multi_stream
            res = bnp.bn_addrelu_fwd_nhwc(x, z, s, b, rm, riv, mini_m, mini_riv, bitmask, ret_cta, mom, epsilon, my_data, pair_data, pair_data2, pair_data3, bn_group, magic, fwd_occup, fwd_grid_x, multi_stream)
            return res
        else:
            return bnp.bn_addrelu_fwd_eval_nhwc(x, z, s, b, rm, riv, ret_cta, bn_group, mom, epsilon)

    @staticmethod
    def backward(ctx, grad_y):
        x, s, b, rm, riv, mini_m, mini_riv, bitmask = ctx.saved_variables
        epsilon = ctx.epsilon
        mom = ctx.momentum
        ret_cta = ctx.ret_cta
        my_data = ctx.my_data
        pair_data = ctx.pair_data
        magic = ctx.magic
        pair_data2 = ctx.pair_data2
        pair_data3 = ctx.pair_data3
        bn_group = ctx.bn_group
        bwd_occup = ctx.bwd_occup
        bwd_grid_x = ctx.bwd_grid_x
        multi_stream = ctx.multi_stream
        dx, dz, dscale, dbias = bnp.bn_addrelu_bwd_nhwc(x, grad_y, s, b, rm, riv, mini_m, mini_riv, bitmask, ret_cta, mom, epsilon, my_data, pair_data, pair_data2, pair_data3, bn_group, magic, bwd_occup, bwd_grid_x, multi_stream)
        return dx, dz, dscale, dbias, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None


class BatchNorm2d_NHWC(_BatchNorm):

    def __init__(self, num_features, fuse_relu=False, bn_group=1, max_cta_per_sm=2, cta_launch_margin=12, multi_stream=False):
        super(BatchNorm2d_NHWC, self).__init__(num_features)
        self.fuse_relu = fuse_relu
        self.multi_stream = multi_stream
        self.minibatch_mean = torch.FloatTensor(num_features)
        self.minibatch_riv = torch.FloatTensor(num_features)
        self.bn_group = bn_group
        self.max_cta_per_sm = max_cta_per_sm
        self.cta_launch_margin = cta_launch_margin
        self.my_data = None
        self.pair_data = None
        self.pair_data2 = None
        self.pair_data3 = None
        self.local_rank = 0
        self.magic = torch.IntTensor([0])
        assert max_cta_per_sm > 0
        self.fwd_occupancy = min(bnp.bn_fwd_nhwc_occupancy(), max_cta_per_sm)
        self.bwd_occupancy = min(bnp.bn_bwd_nhwc_occupancy(), max_cta_per_sm)
        self.addrelu_fwd_occupancy = min(bnp.bn_addrelu_fwd_nhwc_occupancy(), max_cta_per_sm)
        self.addrelu_bwd_occupancy = min(bnp.bn_addrelu_bwd_nhwc_occupancy(), max_cta_per_sm)
        mp_count = torch.cuda.get_device_properties(None).multi_processor_count
        self.fwd_grid_dim_x = max(mp_count * self.fwd_occupancy - cta_launch_margin, 1)
        self.bwd_grid_dim_x = max(mp_count * self.bwd_occupancy - cta_launch_margin, 1)
        self.addrelu_fwd_grid_dim_x = max(mp_count * self.addrelu_fwd_occupancy - cta_launch_margin, 1)
        self.addrelu_bwd_grid_dim_x = max(mp_count * self.addrelu_bwd_occupancy - cta_launch_margin, 1)
        self.grid_dim_y = (num_features + 63) // 64
        self.ret_cta = torch.ByteTensor(8192).fill_(0)
        if bn_group > 1:
            local_rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
            assert world_size >= bn_group
            assert world_size % bn_group == 0
            bn_sync_steps = 1
            if bn_group == 4:
                bn_sync_steps = 2
            if bn_group == 8:
                bn_sync_steps = 3
            self.ipc_buffer = torch.ByteTensor(bnp.get_buffer_size(bn_sync_steps))
            self.my_data = bnp.get_data_ptr(self.ipc_buffer)
            self.storage = self.ipc_buffer.storage()
            self.share_cuda = self.storage._share_cuda_()
            internal_cuda_mem = self.share_cuda
            my_handle = torch.ByteTensor(np.frombuffer(internal_cuda_mem[1], dtype=np.uint8))
            my_offset = torch.IntTensor([internal_cuda_mem[3]])
            handles_all = torch.empty(world_size, my_handle.size(0), dtype=my_handle.dtype, device=my_handle.device)
            handles_l = list(handles_all.unbind(0))
            torch.distributed.all_gather(handles_l, my_handle)
            offsets_all = torch.empty(world_size, my_offset.size(0), dtype=my_offset.dtype, device=my_offset.device)
            offsets_l = list(offsets_all.unbind(0))
            torch.distributed.all_gather(offsets_l, my_offset)
            self.pair_handle = handles_l[local_rank ^ 1].cpu().contiguous()
            pair_offset = offsets_l[local_rank ^ 1].cpu()
            self.pair_data = bnp.get_remote_data_ptr(self.pair_handle, pair_offset)
            if bn_group > 2:
                self.pair_handle2 = handles_l[local_rank ^ 2].cpu().contiguous()
                pair_offset2 = offsets_l[local_rank ^ 2].cpu()
                self.pair_data2 = bnp.get_remote_data_ptr(self.pair_handle2, pair_offset2)
            if bn_group > 4:
                self.pair_handle3 = handles_l[local_rank ^ 4].cpu().contiguous()
                pair_offset3 = offsets_l[local_rank ^ 4].cpu()
                self.pair_data3 = bnp.get_remote_data_ptr(self.pair_handle3, pair_offset3)
            self.magic = torch.IntTensor([2])
            self.local_rank = local_rank

    def forward(self, x, z=None):
        if z is not None:
            assert self.fuse_relu == True
            return bn_addrelu_NHWC_impl.apply(x, z, self.weight, self.bias, self.running_mean, self.running_var, self.minibatch_mean, self.minibatch_riv, self.grid_dim_y, self.ret_cta, self.momentum, self.eps, self.training, self.bn_group, self.my_data, self.pair_data, self.magic, self.pair_data2, self.pair_data3, self.addrelu_fwd_occupancy, self.addrelu_fwd_grid_dim_x, self.addrelu_bwd_occupancy, self.addrelu_bwd_grid_dim_x, self.multi_stream)
        else:
            return bn_NHWC_impl.apply(x, self.weight, self.bias, self.running_mean, self.running_var, self.minibatch_mean, self.minibatch_riv, self.ret_cta, self.momentum, self.eps, self.fuse_relu, self.training, self.bn_group, self.my_data, self.pair_data, self.magic, self.pair_data2, self.pair_data3, self.fwd_occupancy, self.fwd_grid_dim_x, self.bwd_occupancy, self.bwd_grid_dim_x, self.multi_stream)

    def __del__(self):
        if self.bn_group > 1:
            bnp.close_remote_data(self.pair_handle)
            if self.bn_group > 2:
                bnp.close_remote_data(self.pair_handle2)
                if self.bn_group > 4:
                    bnp.close_remote_data(self.pair_handle3)


class FastLayerNormFN(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, gamma, beta, epsilon):
        x = x.contiguous()
        gamma = gamma.contiguous()
        beta = beta.contiguous()
        hidden_size = gamma.numel()
        xmat = x.view((-1, hidden_size))
        ymat, mu, rsigma = fast_layer_norm.ln_fwd(xmat, gamma, beta, epsilon)
        ctx.save_for_backward(x, gamma, mu, rsigma)
        return ymat.view(x.shape)

    @staticmethod
    def backward(ctx, dy):
        dy = dy.contiguous()
        x, gamma, mu, rsigma = ctx.saved_tensors
        hidden_size = gamma.numel()
        xmat = x.view((-1, hidden_size))
        dymat = dy.view(xmat.shape)
        dxmat, dgamma, dbeta = fast_layer_norm.ln_bwd(dymat, xmat, mu, rsigma, gamma)
        dx = dxmat.view(x.shape)
        return dx, dgamma, dbeta, None


class FastLayerNorm(torch.nn.Module):

    def __init__(self, hidden_size, eps=1e-05):
        super(FastLayerNorm, self).__init__()
        self.epsilon = eps
        self.weight = torch.nn.Parameter(torch.Tensor(hidden_size))
        self.bias = torch.nn.Parameter(torch.Tensor(hidden_size))
        self.reset_parameters()

    def reset_parameters(self):
        init.ones_(self.weight)
        init.zeros_(self.bias)

    def forward(self, x):
        return FastLayerNormFN.apply(x, self.weight, self.bias, self.epsilon)


class FusedLayerNormAffineFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, weight, bias, normalized_shape, eps):
        global fused_layer_norm_cuda
        if fused_layer_norm_cuda is None:
            fused_layer_norm_cuda = importlib.import_module('fused_layer_norm_cuda')
        ctx.normalized_shape = normalized_shape
        ctx.eps = eps
        input_ = input.contiguous()
        weight_ = weight.contiguous()
        bias_ = bias.contiguous()
        output, mean, invvar = fused_layer_norm_cuda.forward_affine(input_, ctx.normalized_shape, weight_, bias_, ctx.eps)
        ctx.save_for_backward(input_, weight_, bias_, mean, invvar)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input_, weight_, bias_, mean, invvar = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None
        grad_input, grad_weight, grad_bias = fused_layer_norm_cuda.backward_affine(grad_output.contiguous(), mean, invvar, input_, ctx.normalized_shape, weight_, bias_, ctx.eps)
        return grad_input, grad_weight, grad_bias, None, None


class FusedLayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, normalized_shape, eps):
        global fused_layer_norm_cuda
        if fused_layer_norm_cuda is None:
            fused_layer_norm_cuda = importlib.import_module('fused_layer_norm_cuda')
        ctx.normalized_shape = normalized_shape
        ctx.eps = eps
        input_ = input.contiguous()
        output, mean, invvar = fused_layer_norm_cuda.forward(input_, ctx.normalized_shape, ctx.eps)
        ctx.save_for_backward(input_, mean, invvar)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input_, mean, invvar = ctx.saved_tensors
        grad_input = None
        grad_input = fused_layer_norm_cuda.backward(grad_output.contiguous(), mean, invvar, input_, ctx.normalized_shape, ctx.eps)
        return grad_input, None, None


class FusedLayerNorm(torch.nn.Module):
    """Applies Layer Normalization over a mini-batch of inputs as described in
    the paper `Layer Normalization`_ .

    Currently only runs on cuda() tensors.

    .. math::
        y = \\frac{x - \\mathrm{E}[x]}{ \\sqrt{\\mathrm{Var}[x] + \\epsilon}} * \\gamma + \\beta

    The mean and standard-deviation are calculated separately over the last
    certain number dimensions which have to be of the shape specified by
    :attr:`normalized_shape`.
    :math:`\\gamma` and :math:`\\beta` are learnable affine transform parameters of
    :attr:`normalized_shape` if :attr:`elementwise_affine` is ``True``.

    .. note::
        Unlike Batch Normalization and Instance Normalization, which applies
        scalar scale and bias for each entire channel/plane with the
        :attr:`affine` option, Layer Normalization applies per-element scale and
        bias with :attr:`elementwise_affine`.

    This layer uses statistics computed from input data in both training and
    evaluation modes.

    Args:
        normalized_shape (int or list or torch.Size): input shape from an expected input
            of size

            .. math::
                [* \\times \\text{normalized}\\_\\text{shape}[0] \\times \\text{normalized}\\_\\text{shape}[1]
                    \\times \\ldots \\times \\text{normalized}\\_\\text{shape}[-1]]

            If a single integer is used, it is treated as a singleton list, and this module will
            normalize over the last dimension which is expected to be of that specific size.
        eps: a value added to the denominator for numerical stability. Default: 1e-5
        elementwise_affine: a boolean value that when set to ``True``, this module
            has learnable per-element affine parameters initialized to ones (for weights)
            and zeros (for biases). Default: ``True``.

    Shape:
        - Input: :math:`(N, *)`
        - Output: :math:`(N, *)` (same shape as input)

    Examples::

        >>> input = torch.randn(20, 5, 10, 10)
        >>> # With Learnable Parameters
        >>> m = apex.normalization.FusedLayerNorm(input.size()[1:])
        >>> # Without Learnable Parameters
        >>> m = apex.normalization.FusedLayerNorm(input.size()[1:], elementwise_affine=False)
        >>> # Normalize over last two dimensions
        >>> m = apex.normalization.FusedLayerNorm([10, 10])
        >>> # Normalize over last dimension of size 10
        >>> m = apex.normalization.FusedLayerNorm(10)
        >>> # Activating the module
        >>> output = m(input)

    .. _`Layer Normalization`: https://arxiv.org/abs/1607.06450
    """

    def __init__(self, normalized_shape, eps=1e-05, elementwise_affine=True):
        super(FusedLayerNorm, self).__init__()
        global fused_layer_norm_cuda
        fused_layer_norm_cuda = importlib.import_module('fused_layer_norm_cuda')
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = normalized_shape,
        self.normalized_shape = torch.Size(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = Parameter(torch.Tensor(*normalized_shape))
            self.bias = Parameter(torch.Tensor(*normalized_shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.elementwise_affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)

    def forward(self, input):
        if not input.is_cuda:
            return F.layer_norm(input, self.normalized_shape, self.weight, self.bias, self.eps)
        if self.elementwise_affine:
            return FusedLayerNormAffineFunction.apply(input, self.weight, self.bias, self.normalized_shape, self.eps)
        else:
            return FusedLayerNormFunction.apply(input, self.normalized_shape, self.eps)

    def extra_repr(self):
        return '{normalized_shape}, eps={eps}, elementwise_affine={elementwise_affine}'.format(**self.__dict__)


class EncdecAttnFunc(torch.autograd.Function):

    @staticmethod
    def forward(ctx, use_time_mask, is_training, heads, scale, inputs_q, inputs_kv, input_weights_q, input_weights_kv, output_weights, input_biases_q, input_biases_kv, output_biases, mask, dropout_prob):
        use_biases_t = torch.tensor([input_biases_q is not None])
        heads_t = torch.tensor([heads])
        scale_t = torch.tensor([scale])
        dropout_prob_t = torch.tensor([dropout_prob])
        null_tensor = torch.tensor([])
        head_dim = inputs_q.size(2) // heads
        if use_biases_t[0]:
            input_lin_q_results = torch.addmm(input_biases_q, inputs_q.view(inputs_q.size(0) * inputs_q.size(1), inputs_q.size(2)), input_weights_q.transpose(0, 1), beta=1.0, alpha=1.0)
        else:
            input_lin_q_results = torch.mm(inputs_q.view(inputs_q.size(0) * inputs_q.size(1), inputs_q.size(2)), input_weights_q.transpose(0, 1))
        input_lin_q_results = input_lin_q_results.view(inputs_q.size(0), inputs_q.size(1), input_weights_q.size(0))
        if use_biases_t[0]:
            input_lin_kv_results = torch.addmm(input_biases_kv, inputs_kv.view(inputs_kv.size(0) * inputs_kv.size(1), inputs_kv.size(2)), input_weights_kv.transpose(0, 1), beta=1.0, alpha=1.0)
        else:
            input_lin_kv_results = torch.mm(inputs_kv.view(inputs_kv.size(0) * inputs_kv.size(1), inputs_kv.size(2)), input_weights_kv.transpose(0, 1))
        input_lin_kv_results = input_lin_kv_results.view(inputs_kv.size(0), inputs_kv.size(1), input_weights_kv.size(0))
        queries = input_lin_q_results.view(inputs_q.size(0), inputs_q.size(1) * heads, head_dim)
        input_lin_kv_results = input_lin_kv_results.view(inputs_kv.size(0), inputs_kv.size(1) * heads, 2, head_dim)
        keys = input_lin_kv_results[:, :, 0, :]
        values = input_lin_kv_results[:, :, 1, :]
        matmul1_results = torch.empty((queries.size(1), queries.size(0), keys.size(0)), dtype=queries.dtype, device=torch.device('cuda'))
        matmul1_results = torch.baddbmm(matmul1_results, queries.transpose(0, 1), keys.transpose(0, 1).transpose(1, 2), out=matmul1_results, beta=0.0, alpha=scale_t[0])
        if mask is not None:
            if use_time_mask:
                assert len(mask.size()) == 2, 'Timing mask is not 2D!'
                assert mask.size(0) == mask.size(1), 'Sequence length should match!'
                mask = mask
                matmul1_results = matmul1_results.masked_fill_(mask, float('-inf'))
            else:
                batches, seql_q, seql_k = matmul1_results.size()
                seqs = int(batches / heads)
                matmul1_results = matmul1_results.view(seqs, heads, seql_q, seql_k)
                mask = mask
                matmul1_results = matmul1_results.masked_fill_(mask.unsqueeze(1).unsqueeze(2), float('-inf'))
                matmul1_results = matmul1_results.view(seqs * heads, seql_q, seql_k)
        softmax_results = F.softmax(matmul1_results, dim=-1)
        if is_training:
            dropout_results, dropout_mask = torch._fused_dropout(softmax_results, p=1.0 - dropout_prob_t[0])
        else:
            dropout_results = softmax_results
            dropout_mask = null_tensor
        matmul2_results = torch.empty((dropout_results.size(1), dropout_results.size(0), values.size(2)), dtype=dropout_results.dtype, device=torch.device('cuda')).transpose(1, 0)
        matmul2_results = torch.bmm(dropout_results, values.transpose(0, 1), out=matmul2_results)
        matmul2_results = matmul2_results.transpose(0, 1).contiguous().view(inputs_q.size(0), inputs_q.size(1), inputs_q.size(2))
        if use_biases_t[0]:
            outputs = torch.addmm(output_biases, matmul2_results.view(inputs_q.size(0) * inputs_q.size(1), inputs_q.size(2)), output_weights.transpose(0, 1), beta=1.0, alpha=1.0)
        else:
            outputs = torch.mm(matmul2_results.view(inputs_q.size(0) * inputs_q.size(1), inputs_q.size(2)), output_weights.transpose(0, 1))
        outputs = outputs.view(inputs_q.size(0), inputs_q.size(1), output_weights.size(0))
        ctx.save_for_backward(use_biases_t, heads_t, scale_t, matmul2_results, dropout_results, softmax_results, input_lin_q_results, input_lin_kv_results, inputs_q, inputs_kv, input_weights_q, input_weights_kv, output_weights, dropout_mask, dropout_prob_t)
        return outputs.detach()

    @staticmethod
    def backward(ctx, output_grads):
        use_biases_t, heads_t, scale_t, matmul2_results, dropout_results, softmax_results, input_lin_q_results, input_lin_kv_results, inputs_q, inputs_kv, input_weights_q, input_weights_kv, output_weights, dropout_mask, dropout_prob_t = ctx.saved_tensors
        head_dim = inputs_q.size(2) // heads_t[0]
        queries = input_lin_q_results.view(inputs_q.size(0), inputs_q.size(1) * heads_t[0], head_dim)
        input_lin_kv_results = input_lin_kv_results.view(inputs_kv.size(0), inputs_kv.size(1) * heads_t[0], 2, head_dim)
        keys = input_lin_kv_results[:, :, 0, :]
        values = input_lin_kv_results[:, :, 1, :]
        input_lin_kv_results_grads = torch.empty_like(input_lin_kv_results)
        queries_grads = torch.empty_like(queries)
        keys_grads = input_lin_kv_results_grads[:, :, 0, :]
        values_grads = input_lin_kv_results_grads[:, :, 1, :]
        output_lin_grads = torch.mm(output_grads.view(output_grads.size(0) * output_grads.size(1), output_grads.size(2)), output_weights)
        output_lin_grads = output_lin_grads.view(output_grads.size(0), output_grads.size(1), output_weights.size(1))
        output_weight_grads = torch.mm(output_grads.view(output_grads.size(0) * output_grads.size(1), output_grads.size(2)).transpose(0, 1), matmul2_results.view(matmul2_results.size(0) * matmul2_results.size(1), matmul2_results.size(2)))
        output_lin_grads = output_lin_grads.view(output_grads.size(0), output_grads.size(1) * heads_t[0], head_dim).transpose(0, 1)
        if use_biases_t[0]:
            output_bias_grads = torch.sum(output_grads.view(output_grads.size(0) * output_grads.size(1), output_grads.size(2)), 0)
        else:
            output_bias_grads = None
        matmul2_dgrad1 = torch.bmm(output_lin_grads, values.transpose(0, 1).transpose(1, 2))
        values_grads = torch.bmm(dropout_results.transpose(1, 2), output_lin_grads, out=values_grads.transpose(0, 1))
        dropout_grads = torch._masked_scale(matmul2_dgrad1, dropout_mask, 1.0 / (1.0 - dropout_prob_t[0]))
        softmax_grads = torch._softmax_backward_data(dropout_grads, softmax_results, -1, softmax_results)
        queries_grads = torch.baddbmm(queries_grads.transpose(0, 1), softmax_grads, keys.transpose(0, 1), out=queries_grads.transpose(0, 1), beta=0.0, alpha=scale_t[0])
        keys_grads = torch.baddbmm(keys_grads.transpose(0, 1), softmax_grads.transpose(1, 2), queries.transpose(0, 1), out=keys_grads.transpose(0, 1), beta=0.0, alpha=scale_t[0])
        queries_grads = queries_grads.transpose(0, 1).view(inputs_q.size(0) * inputs_q.size(1), heads_t[0] * head_dim)
        input_q_grads = torch.mm(queries_grads, input_weights_q)
        input_q_grads = input_q_grads.view(inputs_q.size(0), inputs_q.size(1), inputs_q.size(2))
        input_lin_kv_results_grads = input_lin_kv_results_grads.view(inputs_kv.size(0) * inputs_kv.size(1), heads_t[0] * 2 * head_dim)
        input_kv_grads = torch.mm(input_lin_kv_results_grads, input_weights_kv)
        input_kv_grads = input_kv_grads.view(inputs_kv.size(0), inputs_kv.size(1), inputs_kv.size(2))
        input_weight_q_grads = torch.mm(queries_grads.transpose(0, 1), inputs_q.view(inputs_q.size(0) * inputs_q.size(1), inputs_q.size(2)))
        input_weight_kv_grads = torch.mm(input_lin_kv_results_grads.transpose(0, 1), inputs_kv.view(inputs_kv.size(0) * inputs_kv.size(1), inputs_kv.size(2)))
        if use_biases_t[0]:
            input_bias_grads_q = torch.sum(queries_grads, 0)
            input_bias_grads_kv = torch.sum(input_lin_kv_results_grads, 0)
        else:
            input_bias_grads_q = None
            input_bias_grads_kv = None
        return None, None, None, None, input_q_grads, input_kv_grads, input_weight_q_grads, input_weight_kv_grads, output_weight_grads, input_bias_grads_q, input_bias_grads_kv, output_bias_grads, None, None


encdec_attn_func = EncdecAttnFunc.apply


class FastEncdecAttnFunc(torch.autograd.Function):

    @staticmethod
    def forward(ctx, use_time_mask, is_training, heads, inputs_q, inputs_kv, input_weights_q, input_weights_kv, output_weights, pad_mask, dropout_prob):
        heads_t = torch.tensor([heads])
        dropout_prob_t = torch.tensor([dropout_prob])
        null_tensor = torch.tensor([])
        use_mask = pad_mask is not None
        input_lin_q_results, input_lin_kv_results, softmax_results, dropout_results, dropout_mask, matmul2_results, outputs = fast_encdec_multihead_attn.forward(use_mask, use_time_mask, is_training, heads, inputs_q, inputs_kv, input_weights_q, input_weights_kv, output_weights, pad_mask if use_mask else null_tensor, dropout_prob)
        ctx.save_for_backward(heads_t, matmul2_results, dropout_results, softmax_results, input_lin_q_results, input_lin_kv_results, inputs_q, inputs_kv, input_weights_q, input_weights_kv, output_weights, dropout_mask, dropout_prob_t)
        return outputs.detach()

    @staticmethod
    def backward(ctx, output_grads):
        heads_t, matmul2_results, dropout_results, softmax_results, input_lin_q_results, input_lin_kv_results, inputs_q, inputs_kv, input_weights_q, input_weights_kv, output_weights, dropout_mask, dropout_prob_t = ctx.saved_tensors
        input_q_grads, input_kv_grads, input_weight_q_grads, input_weight_kv_grads, output_weight_grads = fast_encdec_multihead_attn.backward(heads_t[0], output_grads, matmul2_results, dropout_results, softmax_results, input_lin_q_results, input_lin_kv_results, inputs_q, inputs_kv, input_weights_q, input_weights_kv, output_weights, dropout_mask, dropout_prob_t[0])
        return None, None, None, input_q_grads, input_kv_grads, input_weight_q_grads, input_weight_kv_grads, output_weight_grads, None, None


fast_encdec_attn_func = FastEncdecAttnFunc.apply


class FastEncdecAttnNormAddFunc(torch.autograd.Function):

    @staticmethod
    def forward(ctx, use_time_mask, is_training, heads, inputs_q, inputs_kv, lyr_nrm_gamma_weights, lyr_nrm_beta_weights, input_weights_q, input_weights_kv, output_weights, pad_mask, dropout_prob):
        heads_t = torch.tensor([heads])
        dropout_prob_t = torch.tensor([dropout_prob])
        null_tensor = torch.tensor([])
        use_mask = pad_mask is not None
        lyr_nrm_results, lyr_nrm_mean, lyr_nrm_invvar, input_lin_q_results, input_lin_kv_results, softmax_results, dropout_results, dropout_mask, matmul2_results, dropout_add_mask, outputs = fast_encdec_multihead_attn_norm_add.forward(use_mask, use_time_mask, is_training, heads, inputs_q, inputs_kv, lyr_nrm_gamma_weights, lyr_nrm_beta_weights, input_weights_q, input_weights_kv, output_weights, pad_mask if use_mask else null_tensor, dropout_prob)
        ctx.save_for_backward(heads_t, matmul2_results, dropout_results, softmax_results, input_lin_q_results, input_lin_kv_results, lyr_nrm_results, lyr_nrm_mean, lyr_nrm_invvar, inputs_q, inputs_kv, lyr_nrm_gamma_weights, lyr_nrm_beta_weights, input_weights_q, input_weights_kv, output_weights, dropout_mask, dropout_add_mask, dropout_prob_t)
        return outputs.detach()

    @staticmethod
    def backward(ctx, output_grads):
        heads_t, matmul2_results, dropout_results, softmax_results, input_lin_q_results, input_lin_kv_results, lyr_nrm_results, lyr_nrm_mean, lyr_nrm_invvar, inputs_q, inputs_kv, lyr_nrm_gamma_weights, lyr_nrm_beta_weights, input_weights_q, input_weights_kv, output_weights, dropout_mask, dropout_add_mask, dropout_prob_t = ctx.saved_tensors
        input_q_grads, input_kv_grads, lyr_nrm_gamma_grads, lyr_nrm_beta_grads, input_weight_q_grads, input_weight_kv_grads, output_weight_grads = fast_encdec_multihead_attn_norm_add.backward(heads_t[0], output_grads, matmul2_results, dropout_results, softmax_results, input_lin_q_results, input_lin_kv_results, lyr_nrm_results, lyr_nrm_mean, lyr_nrm_invvar, inputs_q, inputs_kv, lyr_nrm_gamma_weights, lyr_nrm_beta_weights, input_weights_q, input_weights_kv, output_weights, dropout_mask, dropout_add_mask, dropout_prob_t[0])
        return None, None, None, input_q_grads, input_kv_grads, lyr_nrm_gamma_grads, lyr_nrm_beta_grads, input_weight_q_grads, input_weight_kv_grads, output_weight_grads, None, None


fast_encdec_attn_norm_add_func = FastEncdecAttnNormAddFunc.apply


@torch.jit.script
def jit_dropout_add(x, residual, prob, is_training):
    out = F.dropout(x, p=prob, training=True)
    out = residual + out
    return out


class EncdecMultiheadAttn(nn.Module):
    """Multi-headed attention.

    See "Attention Is All You Need" for more details.
    """

    def __init__(self, embed_dim, num_heads, dropout=0.0, bias=False, include_norm_add=False, impl='fast'):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, 'embed_dim must be divisible by num_heads'
        self.bias = bias
        self.include_norm_add = include_norm_add
        self.impl = impl
        self.scaling = self.head_dim ** -0.5
        self.in_proj_weight_q = Parameter(torch.Tensor(embed_dim, embed_dim))
        self.in_proj_weight_kv = Parameter(torch.Tensor(2 * embed_dim, embed_dim))
        self.out_proj_weight = Parameter(torch.Tensor(embed_dim, embed_dim))
        if self.bias:
            assert impl != 'fast', 'ERROR! The Fast implementation does not support biases!'
            self.in_proj_bias_q = Parameter(torch.Tensor(embed_dim))
            self.in_proj_bias_kv = Parameter(torch.Tensor(2 * embed_dim))
            self.out_proj_bias = Parameter(torch.Tensor(embed_dim))
        else:
            self.register_parameter('in_proj_bias_q', None)
            self.register_parameter('in_proj_bias_kv', None)
            self.in_proj_bias_q = None
            self.in_proj_bias_kv = None
            self.out_proj_bias = None
        if self.include_norm_add:
            if impl == 'fast':
                self.lyr_nrm_gamma_weights = Parameter(torch.Tensor(embed_dim))
                self.lyr_nrm_beta_weights = Parameter(torch.Tensor(embed_dim))
                self.lyr_nrm = None
            else:
                self.register_parameter('lyr_norm_gamma_weights', None)
                self.register_parameter('lyr_norm_beta_weights', None)
                self.lyr_nrm_gamma_weights = None
                self.lyr_nrm_beta_weights = None
                self.lyr_nrm = FusedLayerNorm(embed_dim)
        self.reset_parameters()
        if self.include_norm_add:
            if impl == 'fast':
                self.attn_func = fast_encdec_attn_norm_add_func
            elif impl == 'default':
                self.attn_func = encdec_attn_func
            else:
                assert False, 'Unsupported impl: {} !'.format(impl)
        elif impl == 'fast':
            self.attn_func = fast_encdec_attn_func
        elif impl == 'default':
            self.attn_func = encdec_attn_func
        else:
            assert False, 'Unsupported impl: {} !'.format(impl)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.in_proj_weight_q)
        nn.init.xavier_uniform_(self.in_proj_weight_kv, gain=math.sqrt(1.5))
        nn.init.xavier_uniform_(self.out_proj_weight)
        if self.bias:
            nn.init.constant_(self.in_proj_bias_q, 0.0)
            nn.init.constant_(self.in_proj_bias_kv, 0.0)
            nn.init.constant_(self.out_proj_bias, 0.0)
        if self.include_norm_add:
            if self.impl == 'fast':
                nn.init.ones_(self.lyr_nrm_gamma_weights)
                nn.init.zeros_(self.lyr_nrm_beta_weights)
            else:
                self.lyr_nrm.reset_parameters()

    def forward(self, query, key, value, key_padding_mask=None, need_weights=False, attn_mask=None, is_training=True):
        """Input shape: Time x Batch x Channel

        Self-attention can be implemented by passing in the same arguments for
        query, key and value. Future timesteps can be masked with the
        `mask_future_timesteps` argument. Padding elements can be excluded from
        the key by passing a binary ByteTensor (`key_padding_mask`) with shape:
        batch x src_len, where padding elements are indicated by 1s.
        """
        if key_padding_mask is not None:
            assert attn_mask is None, 'ERROR attn_mask and key_padding_mask should not be both defined!'
            mask = key_padding_mask
        elif attn_mask is not None:
            mask = attn_mask
        else:
            mask = None
        if self.include_norm_add:
            if self.impl == 'fast':
                outputs = self.attn_func(attn_mask is not None, is_training, self.num_heads, query, key, self.lyr_nrm_gamma_weights, self.lyr_nrm_beta_weights, self.in_proj_weight_q, self.in_proj_weight_kv, self.out_proj_weight, mask, self.dropout)
            else:
                lyr_nrm_results = self.lyr_nrm(query)
                outputs = self.attn_func(attn_mask is not None, is_training, self.num_heads, self.scaling, lyr_nrm_results, key, self.in_proj_weight_q, self.in_proj_weight_kv, self.out_proj_weight, self.in_proj_bias_q, self.in_proj_bias_kv, self.out_proj_bias, mask, self.dropout)
                if is_training:
                    outputs = jit_dropout_add(outputs, query, self.dropout, is_training)
                else:
                    outputs = outputs + query
        elif self.impl == 'fast':
            outputs = self.attn_func(attn_mask is not None, is_training, self.num_heads, query, key, self.in_proj_weight_q, self.in_proj_weight_kv, self.out_proj_weight, mask, self.dropout)
        else:
            outputs = self.attn_func(attn_mask is not None, is_training, self.num_heads, self.scaling, query, key, self.in_proj_weight_q, self.in_proj_weight_kv, self.out_proj_weight, self.in_proj_bias_q, self.in_proj_bias_kv, self.out_proj_bias, mask, self.dropout)
        return outputs, None


class FastSelfAttnFunc(torch.autograd.Function):

    @staticmethod
    def forward(ctx, use_time_mask, is_training, heads, inputs, input_weights, output_weights, input_biases, output_biases, pad_mask, mask_additive, dropout_prob):
        use_biases_t = torch.tensor([input_biases is not None])
        heads_t = torch.tensor([heads])
        dropout_prob_t = torch.tensor([dropout_prob])
        null_tensor = torch.tensor([])
        use_mask = pad_mask is not None
        mask_additive_t = torch.tensor([mask_additive])
        if use_biases_t[0]:
            if not mask_additive:
                input_lin_results, softmax_results, dropout_results, dropout_mask, matmul2_results, outputs = fast_self_multihead_attn_bias.forward(use_mask, use_time_mask, is_training, heads, inputs, input_weights, output_weights, input_biases, output_biases, pad_mask if use_mask else null_tensor, dropout_prob)
                ctx.save_for_backward(use_biases_t, heads_t, matmul2_results, dropout_results, softmax_results, null_tensor, null_tensor, mask_additive_t, input_lin_results, inputs, input_weights, output_weights, dropout_mask, dropout_prob_t)
            else:
                input_lin_results, bmm1_results, dropout_results, dropout_mask, matmul2_results, outputs = fast_self_multihead_attn_bias_additive_mask.forward(use_mask, use_time_mask, is_training, heads, inputs, input_weights, output_weights, input_biases, output_biases, pad_mask if use_mask else null_tensor, dropout_prob)
                ctx.save_for_backward(use_biases_t, heads_t, matmul2_results, dropout_results, null_tensor, bmm1_results, pad_mask, mask_additive_t, input_lin_results, inputs, input_weights, output_weights, dropout_mask, dropout_prob_t)
        else:
            input_lin_results, softmax_results, dropout_results, dropout_mask, matmul2_results, outputs = fast_self_multihead_attn.forward(use_mask, use_time_mask, is_training, heads, inputs, input_weights, output_weights, pad_mask if use_mask else null_tensor, dropout_prob)
            ctx.save_for_backward(use_biases_t, heads_t, matmul2_results, dropout_results, softmax_results, null_tensor, null_tensor, mask_additive_t, input_lin_results, inputs, input_weights, output_weights, dropout_mask, dropout_prob_t)
        return outputs.detach()

    @staticmethod
    def backward(ctx, output_grads):
        use_biases_t, heads_t, matmul2_results, dropout_results, softmax_results, bmm1_results, pad_mask, mask_additive_t, input_lin_results, inputs, input_weights, output_weights, dropout_mask, dropout_prob_t = ctx.saved_tensors
        if use_biases_t[0]:
            if not mask_additive_t[0]:
                input_grads, input_weight_grads, output_weight_grads, input_bias_grads, output_bias_grads = fast_self_multihead_attn_bias.backward(heads_t[0], output_grads, matmul2_results, dropout_results, softmax_results, input_lin_results, inputs, input_weights, output_weights, dropout_mask, dropout_prob_t[0])
            else:
                input_grads, input_weight_grads, output_weight_grads, input_bias_grads, output_bias_grads = fast_self_multihead_attn_bias_additive_mask.backward(heads_t[0], output_grads, matmul2_results, dropout_results, bmm1_results, pad_mask, input_lin_results, inputs, input_weights, output_weights, dropout_mask, dropout_prob_t[0])
        else:
            input_bias_grads = None
            output_bias_grads = None
            input_grads, input_weight_grads, output_weight_grads = fast_self_multihead_attn.backward(heads_t[0], output_grads, matmul2_results, dropout_results, softmax_results, input_lin_results, inputs, input_weights, output_weights, dropout_mask, dropout_prob_t[0])
        return None, None, None, input_grads, input_weight_grads, output_weight_grads, input_bias_grads, output_bias_grads, None, None, None


fast_self_attn_func = FastSelfAttnFunc.apply


class FastSelfAttnNormAddFunc(torch.autograd.Function):

    @staticmethod
    def forward(ctx, use_time_mask, is_training, heads, inputs, lyr_nrm_gamma_weights, lyr_nrm_beta_weights, input_weights, output_weights, pad_mask, dropout_prob):
        heads_t = torch.tensor([heads])
        dropout_prob_t = torch.tensor([dropout_prob])
        null_tensor = torch.tensor([])
        use_mask = pad_mask is not None
        lyr_nrm_results, lyr_nrm_mean, lyr_nrm_invvar, input_lin_results, softmax_results, dropout_results, dropout_mask, matmul2_results, dropout_add_mask, outputs = fast_self_multihead_attn_norm_add.forward(use_mask, use_time_mask, is_training, heads, inputs, lyr_nrm_gamma_weights, lyr_nrm_beta_weights, input_weights, output_weights, pad_mask if use_mask else null_tensor, dropout_prob)
        ctx.save_for_backward(heads_t, matmul2_results, dropout_results, softmax_results, input_lin_results, lyr_nrm_results, lyr_nrm_mean, lyr_nrm_invvar, inputs, lyr_nrm_gamma_weights, lyr_nrm_beta_weights, input_weights, output_weights, dropout_mask, dropout_add_mask, dropout_prob_t)
        return outputs.detach()

    @staticmethod
    def backward(ctx, output_grads):
        heads_t, matmul2_results, dropout_results, softmax_results, input_lin_results, lyr_nrm_results, lyr_nrm_mean, lyr_nrm_invvar, inputs, lyr_nrm_gamma_weights, lyr_nrm_beta_weights, input_weights, output_weights, dropout_mask, dropout_add_mask, dropout_prob_t = ctx.saved_tensors
        input_grads, lyr_nrm_gamma_grads, lyr_nrm_beta_grads, input_weight_grads, output_weight_grads = fast_self_multihead_attn_norm_add.backward(heads_t[0], output_grads, matmul2_results, dropout_results, softmax_results, input_lin_results, lyr_nrm_results, lyr_nrm_mean, lyr_nrm_invvar, inputs, lyr_nrm_gamma_weights, lyr_nrm_beta_weights, input_weights, output_weights, dropout_mask, dropout_add_mask, dropout_prob_t[0])
        return None, None, None, input_grads, lyr_nrm_gamma_grads, lyr_nrm_beta_grads, input_weight_grads, output_weight_grads, None, None


fast_self_attn_norm_add_func = FastSelfAttnNormAddFunc.apply


class SelfAttnFunc(torch.autograd.Function):

    @staticmethod
    def forward(ctx, use_time_mask, is_training, heads, scale, inputs, input_weights, output_weights, input_biases, output_biases, mask, is_additive_mask, dropout_prob):
        use_biases_t = torch.tensor([input_biases is not None])
        heads_t = torch.tensor([heads])
        scale_t = torch.tensor([scale])
        dropout_prob_t = torch.tensor([dropout_prob])
        null_tensor = torch.tensor([])
        head_dim = inputs.size(2) // heads
        if use_biases_t[0]:
            input_lin_results = torch.addmm(input_biases, inputs.view(inputs.size(0) * inputs.size(1), inputs.size(2)), input_weights.transpose(0, 1), beta=1.0, alpha=1.0)
        else:
            input_lin_results = torch.mm(inputs.view(inputs.size(0) * inputs.size(1), inputs.size(2)), input_weights.transpose(0, 1))
        input_lin_results = input_lin_results.view(inputs.size(0), inputs.size(1), input_weights.size(0))
        input_lin_results = input_lin_results.view(inputs.size(0), inputs.size(1) * heads, 3, head_dim)
        queries = input_lin_results[:, :, 0, :]
        keys = input_lin_results[:, :, 1, :]
        values = input_lin_results[:, :, 2, :]
        matmul1_results = torch.empty((queries.size(1), queries.size(0), keys.size(0)), dtype=queries.dtype, device=torch.device('cuda'))
        matmul1_results = torch.baddbmm(matmul1_results, queries.transpose(0, 1), keys.transpose(0, 1).transpose(1, 2), out=matmul1_results, beta=0.0, alpha=scale_t[0])
        if mask is not None:
            if use_time_mask:
                assert len(mask.size()) == 2, 'Timing mask is not 2D!'
                assert mask.size(0) == mask.size(1), 'Sequence length should match!'
                mask = mask
                matmul1_results = matmul1_results.masked_fill_(mask, float('-inf'))
            else:
                batches, seql_q, seql_k = matmul1_results.size()
                seqs = int(batches / heads)
                matmul1_results = matmul1_results.view(seqs, heads, seql_q, seql_k)
                if is_additive_mask:
                    matmul1_results = matmul1_results + mask.unsqueeze(1).unsqueeze(2)
                else:
                    mask = mask
                    matmul1_results = matmul1_results.masked_fill_(mask.unsqueeze(1).unsqueeze(2), float('-inf'))
                matmul1_results = matmul1_results.view(seqs * heads, seql_q, seql_k)
        softmax_results = F.softmax(matmul1_results, dim=-1)
        if is_training:
            dropout_results, dropout_mask = torch._fused_dropout(softmax_results, p=1.0 - dropout_prob_t[0])
        else:
            dropout_results = softmax_results
            dropout_mask = null_tensor
        matmul2_results = torch.empty((dropout_results.size(1), dropout_results.size(0), values.size(2)), dtype=dropout_results.dtype, device=torch.device('cuda')).transpose(1, 0)
        matmul2_results = torch.bmm(dropout_results, values.transpose(0, 1), out=matmul2_results)
        matmul2_results = matmul2_results.transpose(0, 1).contiguous().view(inputs.size(0), inputs.size(1), inputs.size(2))
        if use_biases_t[0]:
            outputs = torch.addmm(output_biases, matmul2_results.view(inputs.size(0) * inputs.size(1), inputs.size(2)), output_weights.transpose(0, 1), beta=1.0, alpha=1.0)
        else:
            outputs = torch.mm(matmul2_results.view(inputs.size(0) * inputs.size(1), inputs.size(2)), output_weights.transpose(0, 1))
        outputs = outputs.view(inputs.size(0), inputs.size(1), output_weights.size(0))
        ctx.save_for_backward(use_biases_t, heads_t, scale_t, matmul2_results, dropout_results, softmax_results, input_lin_results, inputs, input_weights, output_weights, dropout_mask, dropout_prob_t)
        return outputs.detach()

    @staticmethod
    def backward(ctx, output_grads):
        use_biases_t, heads_t, scale_t, matmul2_results, dropout_results, softmax_results, input_lin_results, inputs, input_weights, output_weights, dropout_mask, dropout_prob_t = ctx.saved_tensors
        head_dim = inputs.size(2) // heads_t[0]
        input_lin_results = input_lin_results.view(inputs.size(0), inputs.size(1) * heads_t[0], 3, head_dim)
        queries = input_lin_results[:, :, 0, :]
        keys = input_lin_results[:, :, 1, :]
        values = input_lin_results[:, :, 2, :]
        input_lin_results_grads = torch.empty_like(input_lin_results)
        queries_grads = input_lin_results_grads[:, :, 0, :]
        keys_grads = input_lin_results_grads[:, :, 1, :]
        values_grads = input_lin_results_grads[:, :, 2, :]
        output_lin_grads = torch.mm(output_grads.view(output_grads.size(0) * output_grads.size(1), output_grads.size(2)), output_weights)
        output_lin_grads = output_lin_grads.view(output_grads.size(0), output_grads.size(1), output_weights.size(1))
        output_weight_grads = torch.mm(output_grads.view(output_grads.size(0) * output_grads.size(1), output_grads.size(2)).transpose(0, 1), matmul2_results.view(matmul2_results.size(0) * matmul2_results.size(1), matmul2_results.size(2)))
        output_lin_grads = output_lin_grads.view(inputs.size(0), inputs.size(1) * heads_t[0], head_dim).transpose(0, 1)
        if use_biases_t[0]:
            output_bias_grads = torch.sum(output_grads.view(output_grads.size(0) * output_grads.size(1), output_grads.size(2)), 0)
        else:
            output_bias_grads = None
        matmul2_dgrad1 = torch.bmm(output_lin_grads, values.transpose(0, 1).transpose(1, 2))
        values_grads = torch.bmm(dropout_results.transpose(1, 2), output_lin_grads, out=values_grads.transpose(0, 1))
        dropout_grads = torch._masked_scale(matmul2_dgrad1, dropout_mask, 1.0 / (1.0 - dropout_prob_t[0]))
        softmax_grads = torch._softmax_backward_data(dropout_grads, softmax_results, -1, softmax_results)
        queries_grads = torch.baddbmm(queries_grads.transpose(0, 1), softmax_grads, keys.transpose(0, 1), out=queries_grads.transpose(0, 1), beta=0.0, alpha=scale_t[0])
        keys_grads = torch.baddbmm(keys_grads.transpose(0, 1), softmax_grads.transpose(1, 2), queries.transpose(0, 1), out=keys_grads.transpose(0, 1), beta=0.0, alpha=scale_t[0])
        input_lin_results_grads = input_lin_results_grads.view(inputs.size(0) * inputs.size(1), heads_t[0] * 3 * head_dim)
        input_grads = torch.mm(input_lin_results_grads, input_weights)
        input_grads = input_grads.view(inputs.size(0), inputs.size(1), inputs.size(2))
        input_weight_grads = torch.mm(input_lin_results_grads.transpose(0, 1), inputs.view(inputs.size(0) * inputs.size(1), inputs.size(2)))
        if use_biases_t[0]:
            input_bias_grads = torch.sum(input_lin_results_grads, 0)
        else:
            input_bias_grads = None
        return None, None, None, None, input_grads, input_weight_grads, output_weight_grads, input_bias_grads, output_bias_grads, None, None


self_attn_func = SelfAttnFunc.apply


class SelfMultiheadAttn(nn.Module):
    """Multi-headed attention.

    See "Attention Is All You Need" for more details.
    """

    def __init__(self, embed_dim, num_heads, dropout=0.0, bias=False, include_norm_add=False, impl='fast', separate_qkv_params=False, mask_additive=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, 'embed_dim must be divisible by num_heads'
        self.bias = bias
        self.include_norm_add = include_norm_add
        self.impl = impl
        self.scaling = self.head_dim ** -0.5
        self.separate_qkv_params = separate_qkv_params
        self.mask_additive = mask_additive
        if mask_additive:
            assert self.include_norm_add == False, 'additive mask not supported with layer norm'
            assert impl == 'default' or impl == 'fast' and bias, 'additive mask not supported for fast mode without bias'
        if separate_qkv_params:
            self.q_weight = Parameter(torch.Tensor(embed_dim, embed_dim))
            self.k_weight = Parameter(torch.Tensor(embed_dim, embed_dim))
            self.v_weight = Parameter(torch.Tensor(embed_dim, embed_dim))
        else:
            self.in_proj_weight = Parameter(torch.Tensor(3 * embed_dim, embed_dim))
        self.out_proj_weight = Parameter(torch.Tensor(embed_dim, embed_dim))
        if self.bias:
            if separate_qkv_params:
                self.q_bias = Parameter(torch.Tensor(embed_dim))
                self.k_bias = Parameter(torch.Tensor(embed_dim))
                self.v_bias = Parameter(torch.Tensor(embed_dim))
            else:
                self.in_proj_bias = Parameter(torch.Tensor(3 * embed_dim))
            self.out_proj_bias = Parameter(torch.Tensor(embed_dim))
        else:
            if separate_qkv_params:
                self.register_parameter('q_bias', None)
                self.register_parameter('k_bias', None)
                self.register_parameter('v_bias', None)
                self.q_bias = None
                self.k_bias = None
                self.v_bias = None
            else:
                self.register_parameter('in_proj_bias', None)
                self.in_proj_bias = None
            self.register_parameter('out_proj_bias', None)
            self.out_proj_bias = None
        if self.include_norm_add:
            if impl == 'fast':
                self.lyr_nrm_gamma_weights = Parameter(torch.Tensor(embed_dim))
                self.lyr_nrm_beta_weights = Parameter(torch.Tensor(embed_dim))
                self.lyr_nrm = None
            else:
                self.register_parameter('lyr_norm_gamma_weights', None)
                self.register_parameter('lyr_norm_beta_weights', None)
                self.lyr_nrm_gamma_weights = None
                self.lyr_nrm_beta_weights = None
                self.lyr_nrm = FusedLayerNorm(embed_dim)
        self.reset_parameters()
        if self.include_norm_add:
            if impl == 'fast':
                self.attn_func = fast_self_attn_norm_add_func
            elif impl == 'default':
                self.attn_func = self_attn_func
            else:
                assert False, 'Unsupported impl: {} !'.format(impl)
        elif impl == 'fast':
            self.attn_func = fast_self_attn_func
        elif impl == 'default':
            self.attn_func = self_attn_func
        else:
            assert False, 'Unsupported impl: {} !'.format(impl)

    def reset_parameters(self):
        if self.separate_qkv_params:
            nn.init.xavier_uniform_(self.q_weight)
            nn.init.xavier_uniform_(self.k_weight)
            nn.init.xavier_uniform_(self.v_weight)
        else:
            nn.init.xavier_uniform_(self.in_proj_weight, gain=math.sqrt(2))
        nn.init.xavier_uniform_(self.out_proj_weight)
        if self.bias:
            if self.separate_qkv_params:
                nn.init.constant_(self.q_bias, 0.0)
                nn.init.constant_(self.k_bias, 0.0)
                nn.init.constant_(self.v_bias, 0.0)
            else:
                nn.init.constant_(self.in_proj_bias, 0.0)
            nn.init.constant_(self.out_proj_bias, 0.0)
        if self.include_norm_add:
            if self.impl == 'fast':
                nn.init.ones_(self.lyr_nrm_gamma_weights)
                nn.init.zeros_(self.lyr_nrm_beta_weights)
            else:
                self.lyr_nrm.reset_parameters()

    def forward(self, query, key, value, key_padding_mask=None, need_weights=False, attn_mask=None, is_training=True):
        """Input shape: Time x Batch x Channel

        Self-attention can be implemented by passing in the same arguments for
        query, key and value. Future timesteps can be masked with the
        `mask_future_timesteps` argument. Padding elements can be excluded from
        the key by passing a binary ByteTensor (`key_padding_mask`) with shape:
        batch x src_len, where padding elements are indicated by 1s.
        """
        if self.separate_qkv_params:
            input_weights = torch.cat([self.q_weight.view(self.num_heads, 1, self.head_dim, self.embed_dim), self.k_weight.view(self.num_heads, 1, self.head_dim, self.embed_dim), self.v_weight.view(self.num_heads, 1, self.head_dim, self.embed_dim)], dim=1).reshape(3 * self.embed_dim, self.embed_dim).contiguous()
        else:
            input_weights = self.in_proj_weight
        if self.bias:
            if self.separate_qkv_params:
                input_bias = torch.cat([self.q_bias.view(self.num_heads, 1, self.head_dim), self.k_bias.view(self.num_heads, 1, self.head_dim), self.v_bias.view(self.num_heads, 1, self.head_dim)], dim=1).reshape(3 * self.embed_dim).contiguous()
            else:
                input_bias = self.in_proj_bias
        else:
            input_bias = None
        if key_padding_mask is not None:
            assert attn_mask is None, 'ERROR attn_mask and key_padding_mask should not be both defined!'
            mask = key_padding_mask
        elif attn_mask is not None:
            assert self.mask_additive == False, 'additive mask not supported for time mask'
            mask = attn_mask
        else:
            mask = None
        if self.include_norm_add:
            if self.impl == 'fast':
                outputs = self.attn_func(attn_mask is not None, is_training, self.num_heads, query, self.lyr_nrm_gamma_weights, self.lyr_nrm_beta_weights, input_weights, self.out_proj_weight, mask, self.dropout)
            else:
                lyr_nrm_results = self.lyr_nrm(query)
                outputs = self.attn_func(attn_mask is not None, is_training, self.num_heads, self.scaling, lyr_nrm_results, input_weights, self.out_proj_weight, input_bias, self.out_proj_bias, mask, self.dropout)
                if is_training:
                    outputs = jit_dropout_add(outputs, query, self.dropout, is_training)
                else:
                    outputs = outputs + query
        elif self.impl == 'fast':
            outputs = self.attn_func(attn_mask is not None, is_training, self.num_heads, query, input_weights, self.out_proj_weight, input_bias, self.out_proj_bias, mask, self.mask_additive, self.dropout)
        else:
            outputs = self.attn_func(attn_mask is not None, is_training, self.num_heads, self.scaling, query, input_weights, self.out_proj_weight, input_bias, self.out_proj_bias, mask, self.mask_additive, self.dropout)
        return outputs, None


class TransducerJointFunc(torch.autograd.Function):

    @staticmethod
    def forward(ctx, f, g, f_len, g_len, pack_output, batch_offset, packed_batch, opt, fwd_tile_size):
        h = transducer_joint_cuda.forward(f, g, f_len, g_len, batch_offset, packed_batch, opt, pack_output, fwd_tile_size)
        ctx.save_for_backward(f_len, g_len, batch_offset)
        ctx.pack_output = pack_output
        ctx.max_f_len = f.size(1)
        ctx.max_g_len = g.size(1)
        return h

    @staticmethod
    def backward(ctx, loss_grad):
        f_len, g_len, batch_offset = ctx.saved_tensors
        f_grad, g_grad = transducer_joint_cuda.backward(loss_grad, f_len, g_len, batch_offset, ctx.max_f_len, ctx.max_g_len, ctx.pack_output)
        return f_grad, g_grad, None, None, None, None, None, None, None, None, None, None


class TransducerJoint(torch.nn.Module):
    """Transducer joint
    Detail of this loss function can be found in: Sequence Transduction with Recurrent Neural 
    Networks

    Arguments:
        pack_output (bool, optional): whether to pack the output in a compact form with don't-care 
        data being removed. (default: False)
        opt (int, optional): pick the optimization level in [0, 1]. opt=1 picks a tiled algorithm. 
            (default: 1)
        fwd_tile_size (int, optional): tile size used in forward operation. This argument will be 
        ignored if opt != 1. (default: 4) 
    """

    def __init__(self, pack_output=False, opt=1, fwd_tile_size=4):
        super(TransducerJoint, self).__init__()
        self.pack_output = pack_output
        self.opt = opt
        self.fwd_tile_size = fwd_tile_size
        self.dummy_batch_offset = torch.empty(0)

    def forward(self, f, g, f_len, g_len, batch_offset=None, packed_batch=0):
        """Forward operation of transducer joint

        Arguments:
            f (tensor): transcription vector from encode block of shape (B, T, H).
            g (tensor): prediction vector form predict block of shape (B, U, H).
            f_len (tensor): length of transcription vector for each batch.
            g_len (tensor): length of prediction vector minus 1 for each batch.
            batch_offset (tensor, optional): tensor containing the offset of each batch
                in the results. For example, batch offset can be obtained from: 
                batch_offset = torch.cumsum(f_len*g_len, dim=0)
                This argument is required if pack_output == True, and is ignored if 
                pack_output == False. (default: None)
            packed_batch (int, optional): the batch size after packing. This argument is 
                ignored if pack_output == False. (default: 0)
        """
        my_batch_offset = batch_offset if self.pack_output else self.dummy_batch_offset
        if self.pack_output and (batch_offset is None or packed_batch == 0):
            raise Exception('Please specify batch_offset and packed_batch when packing is enabled')
        return TransducerJointFunc.apply(f, g, f_len, g_len, self.pack_output, my_batch_offset, packed_batch, self.opt, self.fwd_tile_size)


class TransducerLossFunc(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, label, f_len, y_len, batch_offset, max_f_len, blank_idx, fuse_softmax_backward, debug_list, opt, packed_input):
        if fuse_softmax_backward == False:
            with torch.enable_grad():
                x = torch.nn.functional.log_softmax(x, dim=-1)
        else:
            x = torch.nn.functional.log_softmax(x, dim=-1)
        alpha, beta, loss = transducer_loss_cuda.forward(x, label, f_len, y_len, batch_offset, max_f_len, blank_idx, opt, packed_input)
        if debug_list == []:
            debug_list += [alpha, beta]
        ctx.save_for_backward(x, alpha, beta, f_len, y_len, label, batch_offset)
        ctx.blank_idx = blank_idx
        ctx.fuse_softmax_backward = fuse_softmax_backward
        ctx.opt = opt
        ctx.packed_input = packed_input
        ctx.max_f_len = max_f_len
        return loss

    @staticmethod
    def backward(ctx, loss_grad):
        x, alpha, beta, f_len, y_len, label, batch_offset = ctx.saved_tensors
        x_grad = transducer_loss_cuda.backward(x, loss_grad, alpha, beta, f_len, y_len, label, batch_offset, ctx.max_f_len, ctx.blank_idx, ctx.opt, ctx.fuse_softmax_backward, ctx.packed_input)
        if ctx.fuse_softmax_backward == False:
            x_grad = x.backward(x_grad)
        return x_grad, None, None, None, None, None, None, None, None, None, None


class TransducerLoss(torch.nn.Module):
    """Transducer loss
    Detail of this loss function can be found in: Sequence Transduction with Recurrent Neural 
    Networks

    Arguments:
        fuse_softmax_backward (bool, optional) whether to fuse the backward of transducer loss with
            softmax. (default: True)
        opt (int, optional): pick the optimization level in [0, 1]. opt=1 picks a more optimized 
            algorithm. In some cases, opt=1 might fall back to opt=0. (default: 1)
        packed_input (bool, optional): whether to pack the output in a compact form with don't-care 
        data being removed. (default: False)
    """

    def __init__(self, fuse_softmax_backward=True, opt=1, packed_input=False):
        super(TransducerLoss, self).__init__()
        self.fuse_softmax_backward = fuse_softmax_backward
        self.opt = opt
        self.packed_input = packed_input
        self.dummy_batch_offset = torch.empty(0)

    def forward(self, x, label, f_len, y_len, blank_idx, batch_offset=None, max_f_len=None, debug_list=None):
        """Forward operation of transducer joint

        Arguments:
            x (tensor): input tensor to the loss function with a shape of (B, T, U, H).
            label (tensor): labels for the input data.
            f_len (tensor): lengths of the inputs in the time dimension for each batch.
            y_len (tensor): lengths of the labels for each batch.
            blank_idx (int): index for the null symbol.
            batch_offset (tensor, optional): tensor containing the offset of each batch
                in the input. For example, batch offset can be obtained from: 
                batch_offset = torch.cumsum(f_len*(y_len+1), dim=0)
                This argument is required if packed_input == True, and is ignored if 
                packed_input == False. (default: None)
            max_f_len (int, optional): maximum length of the input in the time dimension.
                For example, it can be obtained as 
                max_f_len = max(f_len)
                This argument is required if packed_input == True, and is ignored if 
                packed_input == False. (default: None)
                (default: None)
            debug_list (list, optional): when an empty list is supplied, Alpha and Beta generated 
                in the forward operation will be attached to this list for debug purpose. 
                (default: None)
        """
        if self.packed_input:
            if batch_offset is None or max_f_len is None:
                raise Exception('Please specify batch_offset and max_f_len when packing is                                     enabled')
            my_batch_offset = batch_offset
            my_max_f_len = max_f_len
        else:
            my_batch_offset = self.dummy_batch_offset
            my_max_f_len = x.size(1)
        return TransducerLossFunc.apply(x, label, f_len, y_len, my_batch_offset, my_max_f_len, blank_idx, self.fuse_softmax_backward, debug_list, self.opt, self.packed_input)


class tofp16(nn.Module):
    """
    Utility module that implements::

        def forward(self, input):
            return input.half()
    """

    def __init__(self):
        super(tofp16, self).__init__()

    def forward(self, input):
        return input.half()


def convert_module(module, dtype):
    """
    Converts a module's immediate parameters and buffers to dtype.
    """
    for param in module.parameters(recurse=False):
        if param is not None:
            if param.data.dtype.is_floating_point:
                param.data = param.data
            if param._grad is not None and param._grad.data.dtype.is_floating_point:
                param._grad.data = param._grad.data
    for buf in module.buffers(recurse=False):
        if buf is not None and buf.data.dtype.is_floating_point:
            buf.data = buf.data


def convert_network(network, dtype):
    """
    Converts a network's parameters and buffers to dtype.
    """
    for module in network.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm) and module.affine is True:
            continue
        convert_module(module, dtype)
        if isinstance(module, torch.nn.RNNBase) or isinstance(module, torch.nn.modules.rnn.RNNBase):
            module.flatten_parameters()
    return network


class FP16Model(nn.Module):
    """
    Convert model to half precision in a batchnorm-safe way.
    """

    def __init__(self, network):
        super(FP16Model, self).__init__()
        self.network = convert_network(network, dtype=torch.half)

    def forward(self, *inputs):
        inputs = tuple(t.half() for t in inputs)
        return self.network(*inputs)


class MlpFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, bias, activation, *args):
        output = mlp_cuda.forward(bias, activation, args)
        ctx.save_for_backward(*args)
        ctx.outputs = output
        ctx.bias = bias
        ctx.activation = activation
        return output[0]

    @staticmethod
    def backward(ctx, grad_o):
        grads = mlp_cuda.backward(ctx.bias, ctx.activation, grad_o, ctx.outputs, ctx.saved_tensors)
        del ctx.outputs
        return None, None, *grads


class MLP(torch.nn.Module):
    """Launch MLP in C++

    Args:
        mlp_sizes (list of int): MLP sizes. Example: [1024,1024,1024] will create 2 MLP layers with shape 1024x1024
        bias (bool): Default True:
        relu (bool): Default True
    """

    def __init__(self, mlp_sizes, bias=True, activation='relu'):
        super(MLP, self).__init__()
        self.num_layers = len(mlp_sizes) - 1
        self.mlp_sizes = copy(mlp_sizes)
        self.bias = 1 if bias else 0
        if activation is 'none':
            self.activation = 0
        elif activation is 'relu':
            self.activation = 1
        elif activation is 'sigmoid':
            self.activation = 2
        else:
            raise TypeError('activation must be relu or none.')
        self.weights = []
        self.biases = []
        for i in range(self.num_layers):
            w = torch.nn.Parameter(torch.empty(mlp_sizes[i + 1], mlp_sizes[i]))
            self.weights.append(w)
            name = 'weight_{}'.format(i)
            setattr(self, name, w)
            if self.bias:
                b = torch.nn.Parameter(torch.empty(mlp_sizes[i + 1]))
                self.biases.append(b)
                name = 'bias_{}'.format(i)
                setattr(self, name, b)
        self.reset_parameters()

    def reset_parameters(self):
        for weight in self.weights:
            dimsum = weight.size(0) + weight.size(1)
            std = math.sqrt(2.0 / float(dimsum))
            nn.init.normal_(weight, 0.0, std)
        if self.bias:
            for bias in self.biases:
                std = math.sqrt(1.0 / float(bias.size(0)))
                nn.init.normal_(bias, 0.0, std)

    def forward(self, input):
        return mlp_function(self.bias, self.activation, input, *self.weights, *self.biases)

    def extra_repr(self):
        s = f'MLP sizes: {self.mlp_sizes}, Bias={self.bias}, activation={self.activation}'
        return s


def import_flatten_impl():
    global flatten_impl, unflatten_impl, imported_flatten_impl
    try:
        flatten_impl = apex_C.flatten
        unflatten_impl = apex_C.unflatten
    except ImportError:
        None
        flatten_impl = torch._utils._flatten_dense_tensors
        unflatten_impl = torch._utils._unflatten_dense_tensors
    imported_flatten_impl = True


imported_flatten_impl = False


def flatten(bucket):
    if not imported_flatten_impl:
        import_flatten_impl()
    return flatten_impl(bucket)


def unflatten(coalesced, bucket):
    if not imported_flatten_impl:
        import_flatten_impl()
    return unflatten_impl(coalesced, bucket)


def apply_flat_dist_call(bucket, call, extra_args=None):
    coalesced = flatten(bucket)
    if extra_args is not None:
        call(coalesced, *extra_args)
    else:
        call(coalesced)
    if call is dist.all_reduce:
        coalesced /= dist.get_world_size()
    for buf, synced in zip(bucket, unflatten(coalesced, bucket)):
        buf.copy_(synced)


def split_by_type(tensors):
    buckets = OrderedDict()
    for tensor in tensors:
        tp = tensor.type()
        if tp not in buckets:
            buckets[tp] = []
        buckets[tp].append(tensor)
    return buckets


def flat_dist_call(tensors, call, extra_args=None):
    buckets = split_by_type(tensors)
    for tp in buckets:
        bucket = buckets[tp]
        apply_flat_dist_call(bucket, call, extra_args)


class MultiTensorApply(object):
    available = False
    warned = False

    def __init__(self, chunk_size):
        try:
            MultiTensorApply.available = True
            self.chunk_size = chunk_size
        except ImportError as err:
            MultiTensorApply.available = False
            MultiTensorApply.import_err = err

    def check_avail(self):
        if MultiTensorApply.available == False:
            raise RuntimeError('Attempted to call MultiTensorApply method, but MultiTensorApply is not available, possibly because Apex was installed without --cpp_ext --cuda_ext.  Original import error message:', MultiTensorApply.import_err)

    def __call__(self, op, noop_flag_buffer, tensor_lists, *args):
        self.check_avail()
        return op(self.chunk_size, noop_flag_buffer, tensor_lists, *args)


multi_tensor_applier = MultiTensorApply(2048 * 32)


def split_half_float_double(tensors):
    dtypes = ['torch.cuda.HalfTensor', 'torch.cuda.FloatTensor', 'torch.cuda.DoubleTensor']
    buckets = []
    for i, dtype in enumerate(dtypes):
        bucket = [t for t in tensors if t.type() == dtype]
        if bucket:
            buckets.append(bucket)
    return buckets


class SyncBatchnormFunction(Function):

    @staticmethod
    def forward(ctx, input, weight, bias, running_mean, running_variance, eps, process_group, world_size):
        torch.cuda.nvtx.range_push('sync_BN_fw')
        c_last_input = input.transpose(1, -1).contiguous().clone()
        ctx.save_for_backward(c_last_input, weight, bias, running_mean, running_variance)
        ctx.eps = eps
        ctx.process_group = process_group
        ctx.world_size = world_size
        c_last_input = (c_last_input - running_mean) / torch.sqrt(running_variance + eps)
        if weight is not None:
            c_last_input = c_last_input * weight
        if bias is not None:
            c_last_input = c_last_input + bias
        torch.cuda.nvtx.range_pop()
        return c_last_input.transpose(1, -1).contiguous().clone()

    @staticmethod
    def backward(ctx, grad_output):
        torch.cuda.nvtx.range_push('sync_BN_bw')
        c_last_input, weight, bias, running_mean, running_variance = ctx.saved_tensors
        eps = ctx.eps
        process_group = ctx.process_group
        world_size = ctx.world_size
        grad_input = grad_weight = grad_bias = None
        num_features = running_mean.size()[0]
        torch.cuda.nvtx.range_push('carilli field')
        c_last_grad = grad_output.transpose(1, -1).contiguous()
        c_grad = c_last_grad.view(-1, num_features).contiguous()
        torch.cuda.nvtx.range_pop()
        if ctx.needs_input_grad[0]:
            mean_dy = c_grad.mean(0)
            mean_dy_xmu = (c_last_grad * (c_last_input - running_mean)).view(-1, num_features).mean(0)
            if torch.distributed.is_initialized():
                torch.distributed.all_reduce(mean_dy, ReduceOp.SUM, process_group)
                mean_dy = mean_dy / world_size
                torch.distributed.all_reduce(mean_dy_xmu, ReduceOp.SUM, process_group)
                mean_dy_xmu = mean_dy_xmu / world_size
            c_last_grad_input = (c_last_grad - mean_dy - (c_last_input - running_mean) / (running_variance + eps) * mean_dy_xmu) / torch.sqrt(running_variance + eps)
            if weight is not None:
                c_last_grad_input.mul_(weight)
            grad_input = c_last_grad_input.transpose(1, -1).contiguous()
        grad_weight = None
        if weight is not None and ctx.needs_input_grad[1]:
            grad_weight = ((c_last_input - running_mean) / torch.sqrt(running_variance + eps) * c_last_grad).view(-1, num_features).sum(0)
        grad_bias = None
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = c_grad.sum(0)
        torch.cuda.nvtx.range_pop()
        return grad_input, grad_weight, grad_bias, None, None, None, None, None


class SyncBatchNorm(_BatchNorm):
    """
    synchronized batch normalization module extented from ``torch.nn.BatchNormNd``
    with the added stats reduction across multiple processes.
    :class:`apex.parallel.SyncBatchNorm` is designed to work with
    ``DistributedDataParallel``.

    When running in training mode, the layer reduces stats across all processes
    to increase the effective batchsize for normalization layer. This is useful
    in applications where batch size is small on a given process that would
    diminish converged accuracy of the model. The model uses collective
    communication package from ``torch.distributed``.

    When running in evaluation mode, the layer falls back to
    ``torch.nn.functional.batch_norm``.

    Args:
        num_features: :math:`C` from an expected input of size
            :math:`(N, C, L)` or :math:`L` from input of size :math:`(N, L)`
        eps: a value added to the denominator for numerical stability.
            Default: 1e-5
        momentum: the value used for the running_mean and running_var
            computation. Can be set to ``None`` for cumulative moving average
            (i.e. simple average). Default: 0.1
        affine: a boolean value that when set to ``True``, this module has
            learnable affine parameters. Default: ``True``
        track_running_stats: a boolean value that when set to ``True``, this
            module tracks the running mean and variance, and when set to ``False``,
            this module does not track such statistics and always uses batch
            statistics in both training and eval modes. Default: ``True``

    Example::

        >>> sbn = apex.parallel.SyncBatchNorm(100).cuda()
        >>> inp = torch.randn(10, 100, 14, 14).cuda()
        >>> out = sbn(inp)
        >>> inp = torch.randn(3, 100, 20).cuda()
        >>> out = sbn(inp)
    """
    warned = False

    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, process_group=None, channel_last=False):
        if channel_last == True:
            raise AttributeError('channel_last is not supported by primitive SyncBatchNorm implementation. Try install apex with `--cuda_ext` if channel_last is desired.')
        if not SyncBatchNorm.warned:
            if hasattr(self, 'syncbn_import_error'):
                None
            else:
                None
            SyncBatchNorm.warned = True
        super(SyncBatchNorm, self).__init__(num_features, eps=eps, momentum=momentum, affine=affine, track_running_stats=track_running_stats)
        self.process_group = process_group

    def _specify_process_group(self, process_group):
        self.process_group = process_group

    def forward(self, input):
        torch.cuda.nvtx.range_push('sync_bn_fw_with_mean_var')
        mean = None
        var = None
        cast = None
        out = None
        if self.running_mean is not None:
            if self.running_mean.dtype != input.dtype:
                input = input
                cast = input.dtype
        elif self.weight is not None:
            if self.weight.dtype != input.dtype:
                input = input
                cast = input.dtype
        if not self.training and self.track_running_stats:
            torch.cuda.nvtx.range_pop()
            out = F.batch_norm(input, self.running_mean, self.running_var, self.weight, self.bias, False, 0.0, self.eps)
        else:
            process_group = self.process_group
            world_size = 1
            if not self.process_group:
                process_group = torch.distributed.group.WORLD
            self.num_batches_tracked += 1
            with torch.no_grad():
                channel_first_input = input.transpose(0, 1).contiguous()
                squashed_input_tensor_view = channel_first_input.view(channel_first_input.size(0), -1)
                m = None
                local_m = float(squashed_input_tensor_view.size()[1])
                local_mean = torch.mean(squashed_input_tensor_view, 1)
                local_sqr_mean = torch.pow(squashed_input_tensor_view, 2).mean(1)
                if torch.distributed.is_initialized():
                    world_size = torch.distributed.get_world_size(process_group)
                    torch.distributed.all_reduce(local_mean, ReduceOp.SUM, process_group)
                    mean = local_mean / world_size
                    torch.distributed.all_reduce(local_sqr_mean, ReduceOp.SUM, process_group)
                    sqr_mean = local_sqr_mean / world_size
                    m = local_m * world_size
                else:
                    m = local_m
                    mean = local_mean
                    sqr_mean = local_sqr_mean
                var = sqr_mean - mean.pow(2)
                if self.running_mean is not None:
                    self.running_mean = self.momentum * mean + (1 - self.momentum) * self.running_mean
                if self.running_var is not None:
                    self.running_var = m / (m - 1) * self.momentum * var + (1 - self.momentum) * self.running_var
            torch.cuda.nvtx.range_pop()
            out = SyncBatchnormFunction.apply(input, self.weight, self.bias, mean, var, self.eps, process_group, world_size)
        return out


class Foo(torch.nn.Module):

    def __init__(self, size):
        super(Foo, self).__init__()
        self.n = torch.nn.Parameter(torch.ones(size))
        self.m = torch.nn.Parameter(torch.ones(size))

    def forward(self, input):
        return self.n * input + self.m


class LeNet5(nn.Module):

    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, include_top=True):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.include_top = include_top
        self.conv0 = nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn0 = nn.BatchNorm2d(64)
        self.relu0 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(64, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 32, layers[0], stride=2)
        self.layer2 = self._make_layer(block, 64, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 128, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 256, layers[3], stride=2)
        self.layer5 = self._make_layer(block, 512, layers[4], stride=2)
        self.layer6 = self._make_layer(block, 256, layers[5], stride=2)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x0 = self.conv0(x)
        x0 = self.bn0(x0)
        x0 = self.relu0(x0)
        x1 = self.conv1(x0)
        x1 = self.bn1(x1)
        x1 = self.relu(x1)
        x2 = self.layer1(x1)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)
        x6 = self.layer5(x5)
        x7 = self.layer6(x6)
        return x7, x6, x5, x4, x3, x2, x1, x0


class Generator(nn.Module):

    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False), nn.BatchNorm2d(ngf * 8), nn.ReLU(True), nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False), nn.BatchNorm2d(ngf * 4), nn.ReLU(True), nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False), nn.BatchNorm2d(ngf * 2), nn.ReLU(True), nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False), nn.BatchNorm2d(ngf), nn.ReLU(True), nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False), nn.Tanh())

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output


class Discriminator(nn.Module):

    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(nn.Conv2d(nc, ndf, 4, 2, 1, bias=False), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False), nn.BatchNorm2d(ndf * 2), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False), nn.BatchNorm2d(ndf * 4), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False), nn.BatchNorm2d(ndf * 8), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False))

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output.view(-1, 1).squeeze(1)


class MyModel(torch.nn.Module):

    def __init__(self, unique):
        super(MyModel, self).__init__()
        self.weight0 = Parameter(unique + torch.arange(2, device='cuda', dtype=torch.float32))
        self.weight1 = Parameter(1.0 + unique + torch.arange(2, device='cuda', dtype=torch.float16))

    @staticmethod
    def ops(input, weight0, weight1):
        return (input * weight0.float() * weight1.float()).sum()

    def forward(self, input):
        return self.ops(input, self.weight0, self.weight1)


class WhitelistModule(torch.nn.Module):

    def __init__(self, dtype):
        super(WhitelistModule, self).__init__()
        self.weight = torch.nn.Parameter(torch.arange(8 * 8, device='cuda', dtype=dtype).view(8, 8))

    @staticmethod
    def ops(input, weight):
        return input.mm(weight).mm(weight).sum()

    def forward(self, input):
        return self.ops(input, self.weight)


class BlacklistModule(torch.nn.Module):

    def __init__(self, dtype):
        super(BlacklistModule, self).__init__()
        self.weight = torch.nn.Parameter(torch.arange(2 * 8, device='cuda', dtype=dtype).view(2, 8))

    @staticmethod
    def ops(input, weight):
        return (input + torch.pow(weight, 2) + torch.pow(weight, 2)).sum()

    def forward(self, input):
        return self.ops(input, self.weight)


class PromoteModule(torch.nn.Module):

    def __init__(self, dtype):
        super(PromoteModule, self).__init__()
        self.weight = torch.nn.Parameter(torch.arange(2 * 8, device='cuda', dtype=dtype).view(2, 8))

    @staticmethod
    def ops(input, weight):
        return (input * weight * weight).sum()

    def forward(self, input):
        return self.ops(input, self.weight)


class DummyBlock(nn.Module):

    def __init__(self):
        super(DummyBlock, self).__init__()
        self.conv = nn.Conv2d(10, 10, 2)
        self.bn = nn.BatchNorm2d(10, affine=True)

    def forward(self, x):
        return self.conv(self.bn(x))


class DummyNet(nn.Module):

    def __init__(self):
        super(DummyNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, 2)
        self.bn1 = nn.BatchNorm2d(10, affine=False)
        self.db1 = DummyBlock()
        self.db2 = DummyBlock()

    def forward(self, x):
        out = x
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.db1(out)
        out = self.db2(out)
        return out


class DummyNetWrapper(nn.Module):

    def __init__(self):
        super(DummyNetWrapper, self).__init__()
        self.bn = nn.BatchNorm2d(3, affine=True)
        self.dn = DummyNet()

    def forward(self, x):
        return self.dn(self.bn(x))


class TestModel(torch.nn.Module):

    def __init__(self, args):
        super(TestModel, self).__init__()
        self.linear = torch.nn.Sequential(*[torch.nn.Linear(args.dim, args.dim, bias=args.bias) for _ in range(args.layers)])

    def forward(self, x):
        return self.linear(x)


class Model(Module):

    def __init__(self):
        super(Model, self).__init__()
        self.a = Parameter(torch.FloatTensor(4096 * 4096).fill_(1.0))
        self.b = Parameter(torch.FloatTensor(4096 * 4096).fill_(2.0))

    def forward(self, input):
        return input * self.a * self.b


class SEBlock(nn.Module):

    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(channel, channel // reduction), nn.PReLU(), nn.Linear(channel // reduction, channel), nn.Sigmoid())

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class IRBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_se=True):
        super(IRBlock, self).__init__()
        self.bn0 = nn.BatchNorm2d(inplanes)
        self.conv1 = conv3x3(inplanes, inplanes)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.prelu = nn.PReLU()
        self.conv2 = conv3x3(inplanes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.use_se = use_se
        if self.use_se:
            self.se = SEBlock(planes)

    def forward(self, x):
        residual = x
        out = self.bn0(x)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.use_se:
            out = self.se(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.prelu(out)
        return out


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


num_classes = 93431


class ArcMarginModel(nn.Module):

    def __init__(self, args):
        super(ArcMarginModel, self).__init__()
        self.weight = Parameter(torch.FloatTensor(num_classes, args.emb_size))
        nn.init.xavier_uniform_(self.weight)
        self.easy_margin = args.easy_margin
        self.m = args.margin_m
        self.s = args.margin_s
        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)
        self.th = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - self.m) * self.m

    def forward(self, input, label):
        x = F.normalize(input)
        W = F.normalize(self.weight)
        cosine = F.linear(x, W)
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        one_hot = torch.zeros(cosine.size(), device=device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = one_hot * phi + (1.0 - one_hot) * cosine
        output *= self.s
        return output


class FutureResult(object):
    """A thread-safe future implementation. Used only as one-to-one pipe."""

    def __init__(self):
        self._result = None
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)

    def put(self, result):
        with self._lock:
            assert self._result is None, "Previous result has't been fetched."
            self._result = result
            self._cond.notify()

    def get(self):
        with self._lock:
            if self._result is None:
                self._cond.wait()
            res = self._result
            self._result = None
            return res


_SlavePipeBase = collections.namedtuple('_SlavePipeBase', ['identifier', 'queue', 'result'])


class SlavePipe(_SlavePipeBase):
    """Pipe for master-slave communication."""

    def run_slave(self, msg):
        self.queue.put((self.identifier, msg))
        ret = self.result.get()
        self.queue.put(True)
        return ret


_MasterRegistry = collections.namedtuple('MasterRegistry', ['result'])


class SyncMaster(object):
    """An abstract `SyncMaster` object.

    - During the replication, as the data parallel will trigger an callback of each module, all slave devices should
    call `register(id)` and obtain an `SlavePipe` to communicate with the master.
    - During the forward pass, master device invokes `run_master`, all messages from slave devices will be collected,
    and passed to a registered callback.
    - After receiving the messages, the master device should gather the information and determine to message passed
    back to each slave devices.
    """

    def __init__(self, master_callback):
        """

        Args:
            master_callback: a callback to be invoked after having collected messages from slave devices.
        """
        self._master_callback = master_callback
        self._queue = queue.Queue()
        self._registry = collections.OrderedDict()
        self._activated = False

    def __getstate__(self):
        return {'master_callback': self._master_callback}

    def __setstate__(self, state):
        self.__init__(state['master_callback'])

    def register_slave(self, identifier):
        """
        Register an slave device.

        Args:
            identifier: an identifier, usually is the device id.

        Returns: a `SlavePipe` object which can be used to communicate with the master device.

        """
        if self._activated:
            assert self._queue.empty(), 'Queue is not clean before next initialization.'
            self._activated = False
            self._registry.clear()
        future = FutureResult()
        self._registry[identifier] = _MasterRegistry(future)
        return SlavePipe(identifier, self._queue, future)

    def run_master(self, master_msg):
        """
        Main entry for the master device in each forward pass.
        The messages were first collected from each devices (including the master device), and then
        an callback will be invoked to compute the message to be sent back to each devices
        (including the master device).

        Args:
            master_msg: the message that the master want to send to itself. This will be placed as the first
            message when calling `master_callback`. For detailed usage, see `_SynchronizedBatchNorm` for an example.

        Returns: the message to be sent back to the master device.

        """
        self._activated = True
        intermediates = [(0, master_msg)]
        for i in range(self.nr_slaves):
            intermediates.append(self._queue.get())
        results = self._master_callback(intermediates)
        assert results[0][0] == 0, 'The first result should belongs to the master.'
        for i, res in results:
            if i == 0:
                continue
            self._registry[i].result.put(res)
        for i in range(self.nr_slaves):
            assert self._queue.get() is True
        return results[0][1]

    @property
    def nr_slaves(self):
        return len(self._registry)


_ChildMessage = collections.namedtuple('_ChildMessage', ['sum', 'ssum', 'sum_size'])


_MasterMessage = collections.namedtuple('_MasterMessage', ['sum', 'inv_std'])


def _sum_ft(tensor):
    """sum over the first and last dimention"""
    return tensor.sum(dim=0).sum(dim=-1)


def _unsqueeze_ft(tensor):
    """add new dimensions at the front and the tail"""
    return tensor.unsqueeze(0).unsqueeze(-1)


class _SynchronizedBatchNorm(_BatchNorm):

    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True):
        assert ReduceAddCoalesced is not None, 'Can not use Synchronized Batch Normalization without CUDA support.'
        super(_SynchronizedBatchNorm, self).__init__(num_features, eps=eps, momentum=momentum, affine=affine)
        self._sync_master = SyncMaster(self._data_parallel_master)
        self._is_parallel = False
        self._parallel_id = None
        self._slave_pipe = None

    def forward(self, input):
        if not (self._is_parallel and self.training):
            return F.batch_norm(input, self.running_mean, self.running_var, self.weight, self.bias, self.training, self.momentum, self.eps)
        input_shape = input.size()
        input = input.view(input.size(0), self.num_features, -1)
        sum_size = input.size(0) * input.size(2)
        input_sum = _sum_ft(input)
        input_ssum = _sum_ft(input ** 2)
        if self._parallel_id == 0:
            mean, inv_std = self._sync_master.run_master(_ChildMessage(input_sum, input_ssum, sum_size))
        else:
            mean, inv_std = self._slave_pipe.run_slave(_ChildMessage(input_sum, input_ssum, sum_size))
        if self.affine:
            output = (input - _unsqueeze_ft(mean)) * _unsqueeze_ft(inv_std * self.weight) + _unsqueeze_ft(self.bias)
        else:
            output = (input - _unsqueeze_ft(mean)) * _unsqueeze_ft(inv_std)
        return output.view(input_shape)

    def __data_parallel_replicate__(self, ctx, copy_id):
        self._is_parallel = True
        self._parallel_id = copy_id
        if self._parallel_id == 0:
            ctx.sync_master = self._sync_master
        else:
            self._slave_pipe = ctx.sync_master.register_slave(copy_id)

    def _data_parallel_master(self, intermediates):
        """Reduce the sum and square-sum, compute the statistics, and broadcast it."""
        intermediates = sorted(intermediates, key=lambda i: i[1].sum.get_device())
        to_reduce = [i[1][:2] for i in intermediates]
        to_reduce = [j for i in to_reduce for j in i]
        target_gpus = [i[1].sum.get_device() for i in intermediates]
        sum_size = sum([i[1].sum_size for i in intermediates])
        sum_, ssum = ReduceAddCoalesced.apply(target_gpus[0], 2, *to_reduce)
        mean, inv_std = self._compute_mean_std(sum_, ssum, sum_size)
        broadcasted = Broadcast.apply(target_gpus, mean, inv_std)
        outputs = []
        for i, rec in enumerate(intermediates):
            outputs.append((rec[0], _MasterMessage(*broadcasted[i * 2:i * 2 + 2])))
        return outputs

    def _compute_mean_std(self, sum_, ssum, size):
        """Compute the mean and standard-deviation with sum and square-sum. This method
        also maintains the moving average on the master device."""
        assert size > 1, 'BatchNorm computes unbiased standard-deviation, which requires size > 1.'
        mean = sum_ / size
        sumvar = ssum - sum_ * mean
        unbias_var = sumvar / (size - 1)
        bias_var = sumvar / size
        if hasattr(torch, 'no_grad'):
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean.data
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * unbias_var.data
        else:
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean.data
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * unbias_var.data
        return mean, bias_var.clamp(self.eps) ** -0.5


class SynchronizedBatchNorm1d(_SynchronizedBatchNorm):
    """Applies Synchronized Batch Normalization over a 2d or 3d input that is seen as a
    mini-batch.

    .. math::

        y = \\frac{x - mean[x]}{ \\sqrt{Var[x] + \\epsilon}} * gamma + beta

    This module differs from the built-in PyTorch BatchNorm1d as the mean and
    standard-deviation are reduced across all devices during training.

    For example, when one uses `nn.DataParallel` to wrap the network during
    training, PyTorch's implementation normalize the tensor on each device using
    the statistics only on that device, which accelerated the computation and
    is also easy to implement, but the statistics might be inaccurate.
    Instead, in this synchronized version, the statistics will be computed
    over all training samples distributed on multiple devices.

    Note that, for one-GPU or CPU-only case, this module behaves exactly same
    as the built-in PyTorch implementation.

    The mean and standard-deviation are calculated per-dimension over
    the mini-batches and gamma and beta are learnable parameter vectors
    of size C (where C is the input size).

    During training, this layer keeps a running estimate of its computed mean
    and variance. The running sum is kept with a default momentum of 0.1.

    During evaluation, this running mean/variance is used for normalization.

    Because the BatchNorm is done over the `C` dimension, computing statistics
    on `(N, L)` slices, it's common terminology to call this Temporal BatchNorm

    Args:
        num_features: num_features from an expected input of size
            `batch_size x num_features [x width]`
        eps: a value added to the denominator for numerical stability.
            Default: 1e-5
        momentum: the value used for the running_mean and running_var
            computation. Default: 0.1
        affine: a boolean value that when set to ``True``, gives the layer learnable
            affine parameters. Default: ``True``

    Shape::
        - Input: :math:`(N, C)` or :math:`(N, C, L)`
        - Output: :math:`(N, C)` or :math:`(N, C, L)` (same shape as input)

    Examples:
        >>> # With Learnable Parameters
        >>> m = SynchronizedBatchNorm1d(100)
        >>> # Without Learnable Parameters
        >>> m = SynchronizedBatchNorm1d(100, affine=False)
        >>> input = torch.autograd.Variable(torch.randn(20, 100))
        >>> output = m(input)
    """

    def _check_input_dim(self, input):
        if input.dim() != 2 and input.dim() != 3:
            raise ValueError('expected 2D or 3D input (got {}D input)'.format(input.dim()))
        super(SynchronizedBatchNorm1d, self)._check_input_dim(input)


class SynchronizedBatchNorm2d(_SynchronizedBatchNorm):
    """Applies Batch Normalization over a 4d input that is seen as a mini-batch
    of 3d inputs

    .. math::

        y = \\frac{x - mean[x]}{ \\sqrt{Var[x] + \\epsilon}} * gamma + beta

    This module differs from the built-in PyTorch BatchNorm2d as the mean and
    standard-deviation are reduced across all devices during training.

    For example, when one uses `nn.DataParallel` to wrap the network during
    training, PyTorch's implementation normalize the tensor on each device using
    the statistics only on that device, which accelerated the computation and
    is also easy to implement, but the statistics might be inaccurate.
    Instead, in this synchronized version, the statistics will be computed
    over all training samples distributed on multiple devices.

    Note that, for one-GPU or CPU-only case, this module behaves exactly same
    as the built-in PyTorch implementation.

    The mean and standard-deviation are calculated per-dimension over
    the mini-batches and gamma and beta are learnable parameter vectors
    of size C (where C is the input size).

    During training, this layer keeps a running estimate of its computed mean
    and variance. The running sum is kept with a default momentum of 0.1.

    During evaluation, this running mean/variance is used for normalization.

    Because the BatchNorm is done over the `C` dimension, computing statistics
    on `(N, H, W)` slices, it's common terminology to call this Spatial BatchNorm

    Args:
        num_features: num_features from an expected input of
            size batch_size x num_features x height x width
        eps: a value added to the denominator for numerical stability.
            Default: 1e-5
        momentum: the value used for the running_mean and running_var
            computation. Default: 0.1
        affine: a boolean value that when set to ``True``, gives the layer learnable
            affine parameters. Default: ``True``

    Shape::
        - Input: :math:`(N, C, H, W)`
        - Output: :math:`(N, C, H, W)` (same shape as input)

    Examples:
        >>> # With Learnable Parameters
        >>> m = SynchronizedBatchNorm2d(100)
        >>> # Without Learnable Parameters
        >>> m = SynchronizedBatchNorm2d(100, affine=False)
        >>> input = torch.autograd.Variable(torch.randn(20, 100, 35, 45))
        >>> output = m(input)
    """

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'.format(input.dim()))
        super(SynchronizedBatchNorm2d, self)._check_input_dim(input)


class SynchronizedBatchNorm3d(_SynchronizedBatchNorm):
    """Applies Batch Normalization over a 5d input that is seen as a mini-batch
    of 4d inputs

    .. math::

        y = \\frac{x - mean[x]}{ \\sqrt{Var[x] + \\epsilon}} * gamma + beta

    This module differs from the built-in PyTorch BatchNorm3d as the mean and
    standard-deviation are reduced across all devices during training.

    For example, when one uses `nn.DataParallel` to wrap the network during
    training, PyTorch's implementation normalize the tensor on each device using
    the statistics only on that device, which accelerated the computation and
    is also easy to implement, but the statistics might be inaccurate.
    Instead, in this synchronized version, the statistics will be computed
    over all training samples distributed on multiple devices.

    Note that, for one-GPU or CPU-only case, this module behaves exactly same
    as the built-in PyTorch implementation.

    The mean and standard-deviation are calculated per-dimension over
    the mini-batches and gamma and beta are learnable parameter vectors
    of size C (where C is the input size).

    During training, this layer keeps a running estimate of its computed mean
    and variance. The running sum is kept with a default momentum of 0.1.

    During evaluation, this running mean/variance is used for normalization.

    Because the BatchNorm is done over the `C` dimension, computing statistics
    on `(N, D, H, W)` slices, it's common terminology to call this Volumetric BatchNorm
    or Spatio-temporal BatchNorm

    Args:
        num_features: num_features from an expected input of
            size batch_size x num_features x depth x height x width
        eps: a value added to the denominator for numerical stability.
            Default: 1e-5
        momentum: the value used for the running_mean and running_var
            computation. Default: 0.1
        affine: a boolean value that when set to ``True``, gives the layer learnable
            affine parameters. Default: ``True``

    Shape::
        - Input: :math:`(N, C, D, H, W)`
        - Output: :math:`(N, C, D, H, W)` (same shape as input)

    Examples:
        >>> # With Learnable Parameters
        >>> m = SynchronizedBatchNorm3d(100)
        >>> # Without Learnable Parameters
        >>> m = SynchronizedBatchNorm3d(100, affine=False)
        >>> input = torch.autograd.Variable(torch.randn(20, 100, 35, 45, 10))
        >>> output = m(input)
    """

    def _check_input_dim(self, input):
        if input.dim() != 5:
            raise ValueError('expected 5D input (got {}D input)'.format(input.dim()))
        super(SynchronizedBatchNorm3d, self)._check_input_dim(input)


class BatchNorm2dReimpl(nn.Module):
    """
    A re-implementation of batch normalization, used for testing the numerical
    stability.

    Author: acgtyrant
    See also:
    https://github.com/vacancy/Synchronized-BatchNorm-PyTorch/issues/14
    """

    def __init__(self, num_features, eps=1e-05, momentum=0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.weight = nn.Parameter(torch.empty(num_features))
        self.bias = nn.Parameter(torch.empty(num_features))
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.reset_parameters()

    def reset_running_stats(self):
        self.running_mean.zero_()
        self.running_var.fill_(1)

    def reset_parameters(self):
        self.reset_running_stats()
        init.uniform_(self.weight)
        init.zeros_(self.bias)

    def forward(self, input_):
        batchsize, channels, height, width = input_.size()
        numel = batchsize * height * width
        input_ = input_.permute(1, 0, 2, 3).contiguous().view(channels, numel)
        sum_ = input_.sum(1)
        sum_of_square = input_.pow(2).sum(1)
        mean = sum_ / numel
        sumvar = sum_of_square - sum_ * mean
        self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean.detach()
        unbias_var = sumvar / (numel - 1)
        self.running_var = (1 - self.momentum) * self.running_var + self.momentum * unbias_var.detach()
        bias_var = sumvar / numel
        inv_std = 1 / (bias_var + self.eps).pow(0.5)
        output = (input_ - mean.unsqueeze(1)) * inv_std.unsqueeze(1) * self.weight.unsqueeze(1) + self.bias.unsqueeze(1)
        return output.view(channels, batchsize, height, width).permute(1, 0, 2, 3).contiguous()


class CallbackContext(object):
    pass


def execute_replication_callbacks(modules):
    """
    Execute an replication callback `__data_parallel_replicate__` on each module created by original replication.

    The callback will be invoked with arguments `__data_parallel_replicate__(ctx, copy_id)`

    Note that, as all modules are isomorphism, we assign each sub-module with a context
    (shared among multiple copies of this module on different devices).
    Through this context, different copies can share some information.

    We guarantee that the callback on the master copy (the first copy) will be called ahead of calling the callback
    of any slave copies.
    """
    master_copy = modules[0]
    nr_modules = len(list(master_copy.modules()))
    ctxs = [CallbackContext() for _ in range(nr_modules)]
    for i, module in enumerate(modules):
        for j, m in enumerate(module.modules()):
            if hasattr(m, '__data_parallel_replicate__'):
                m.__data_parallel_replicate__(ctxs[j], i)


class DataParallelWithCallback(DataParallel):
    """
    Data Parallel with a replication callback.

    An replication callback `__data_parallel_replicate__` of each module will be invoked after being created by
    original `replicate` function.
    The callback will be invoked with arguments `__data_parallel_replicate__(ctx, copy_id)`

    Examples:
        > sync_bn = SynchronizedBatchNorm1d(10, eps=1e-5, affine=False)
        > sync_bn = DataParallelWithCallback(sync_bn, device_ids=[0, 1])
        # sync_bn.__data_parallel_replicate__ will be invoked.
    """

    def replicate(self, module, device_ids):
        modules = super(DataParallelWithCallback, self).replicate(module, device_ids)
        execute_replication_callbacks(modules)
        return modules


class InstanceNorm2d(nn.Module):

    def __init__(self, epsilon=1e-08, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon

    def forward(self, x):
        tmp = torch.mul(x, x)
        tmp = torch.rsqrt(torch.mean(tmp, (2, 3), True) + self.epsilon)
        return x * tmp


class SPADE(nn.Module):

    def __init__(self, config_text, norm_nc, label_nc):
        super().__init__()
        assert config_text.startswith('spade')
        parsed = re.search('spade(\\D+)(\\d)x\\d', config_text)
        param_free_norm_type = str(parsed.group(1))
        ks = int(parsed.group(2))
        if param_free_norm_type == 'instance':
            self.param_free_norm = InstanceNorm2d(norm_nc)
        elif param_free_norm_type == 'syncbatch':
            self.param_free_norm = SynchronizedBatchNorm2d(norm_nc, affine=False)
        elif param_free_norm_type == 'batch':
            self.param_free_norm = nn.BatchNorm2d(norm_nc, affine=False)
        else:
            raise ValueError('%s is not a recognized param-free norm type in SPADE' % param_free_norm_type)
        nhidden = 128 if norm_nc > 128 else norm_nc
        pw = ks // 2
        self.mlp_shared = nn.Sequential(nn.Conv2d(label_nc, nhidden, kernel_size=ks, padding=pw), nn.ReLU())
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw, bias=False)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw, bias=False)

    def forward(self, x, segmap):
        normalized = self.param_free_norm(x)
        segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')
        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)
        out = normalized * gamma + beta
        return out


class SPADEResnetBlock(nn.Module):

    def __init__(self, fin, fout, opt, semantic_nc=None):
        super().__init__()
        self.learned_shortcut = fin != fout
        fmiddle = min(fin, fout)
        if semantic_nc is None:
            semantic_nc = opt.semantic_nc
        self.conv_0 = nn.Conv2d(fin, fmiddle, kernel_size=3, padding=1)
        self.conv_1 = nn.Conv2d(fmiddle, fout, kernel_size=3, padding=1)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(fin, fout, kernel_size=1, bias=False)
        if 'spectral' in opt.norm_G:
            self.conv_0 = spectral_norm(self.conv_0)
            self.conv_1 = spectral_norm(self.conv_1)
            if self.learned_shortcut:
                self.conv_s = spectral_norm(self.conv_s)
        spade_config_str = opt.norm_G.replace('spectral', '')
        self.norm_0 = SPADE(spade_config_str, fin, semantic_nc)
        self.norm_1 = SPADE(spade_config_str, fmiddle, semantic_nc)
        if self.learned_shortcut:
            self.norm_s = SPADE(spade_config_str, fin, semantic_nc)

    def forward(self, x, seg):
        x_s = self.shortcut(x, seg)
        dx = self.conv_0(self.actvn(self.norm_0(x, seg)))
        dx = self.conv_1(self.actvn(self.norm_1(dx, seg)))
        out = x_s + dx
        return out

    def shortcut(self, x, seg):
        if self.learned_shortcut:
            x_s = self.conv_s(self.norm_s(x, seg))
        else:
            x_s = x
        return x_s

    def actvn(self, x):
        return F.leaky_relu(x, 0.2)


class ResnetBlock(nn.Module):

    def __init__(self, dim, norm_layer, activation=nn.ReLU(False), kernel_size=3):
        super().__init__()
        pw = (kernel_size - 1) // 2
        self.conv_block = nn.Sequential(nn.ReflectionPad2d(pw), norm_layer(nn.Conv2d(dim, dim, kernel_size=kernel_size)), activation, nn.ReflectionPad2d(pw), norm_layer(nn.Conv2d(dim, dim, kernel_size=kernel_size)))

    def forward(self, x):
        y = self.conv_block(x)
        out = x + y
        return out


class VGG19(torch.nn.Module):

    def __init__(self, requires_grad=False):
        super().__init__()
        vgg_pretrained_features = torchvision.models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


class BaseNetwork(nn.Module):

    def __init__(self):
        super(BaseNetwork, self).__init__()

    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def print_network(self):
        if isinstance(self, list):
            self = self[0]
        num_params = 0
        for param in self.parameters():
            num_params += param.numel()
        None

    def init_weights(self, init_type='normal', gain=0.02):

        def init_func(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm2d') != -1:
                if hasattr(m, 'weight') and m.weight is not None:
                    init.normal_(m.weight.data, 1.0, gain)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)
            elif hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'xavier_uniform':
                    init.xavier_uniform_(m.weight.data, gain=1.0)
                elif init_type == 'kaiming':
                    init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    init.orthogonal_(m.weight.data, gain=gain)
                elif init_type == 'none':
                    m.reset_parameters()
                else:
                    raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)
        self.apply(init_func)
        for m in self.children():
            if hasattr(m, 'init_weights'):
                m.init_weights(init_type, gain)


class NLayerDiscriminator(nn.Module):

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, getIntermFeat=False):
        super(NLayerDiscriminator, self).__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers
        kw = 4
        padw = int(np.ceil((kw - 1.0) / 2))
        sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]]
        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw), norm_layer(nf), nn.LeakyReLU(0.2, True)]]
        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw), norm_layer(nf), nn.LeakyReLU(0.2, True)]]
        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]
        if use_sigmoid:
            sequence += [[nn.Sigmoid()]]
        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, 'model' + str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)

    def forward(self, input):
        if self.getIntermFeat:
            res = [input]
            for n in range(self.n_layers + 2):
                model = getattr(self, 'model' + str(n))
                res.append(model(res[-1]))
            return res[1:]
        else:
            return self.model(input)


class MultiscaleDiscriminator(nn.Module):

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, num_D=3, getIntermFeat=False):
        super(MultiscaleDiscriminator, self).__init__()
        self.num_D = num_D
        self.n_layers = n_layers
        self.getIntermFeat = getIntermFeat
        for i in range(num_D):
            netD = NLayerDiscriminator(input_nc, ndf, n_layers, norm_layer, use_sigmoid, getIntermFeat)
            if getIntermFeat:
                for j in range(n_layers + 2):
                    setattr(self, 'scale' + str(i) + '_layer' + str(j), getattr(netD, 'model' + str(j)))
            else:
                setattr(self, 'layer' + str(i), netD.model)
        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def singleD_forward(self, model, input):
        if self.getIntermFeat:
            result = [input]
            for i in range(len(model)):
                result.append(model[i](result[-1]))
            return result[1:]
        else:
            return [model(input)]

    def forward(self, input):
        num_D = self.num_D
        result = []
        input_downsampled = input
        for i in range(num_D):
            if self.getIntermFeat:
                model = [getattr(self, 'scale' + str(num_D - 1 - i) + '_layer' + str(j)) for j in range(self.n_layers + 2)]
            else:
                model = getattr(self, 'layer' + str(num_D - 1 - i))
            result.append(self.singleD_forward(model, input_downsampled))
            if i != num_D - 1:
                input_downsampled = self.downsample(input_downsampled)
        return result


class ScaledLeakyReLU(nn.Module):

    def __init__(self, negative_slope=0.2):
        super().__init__()
        self.negative_slope = negative_slope

    def forward(self, input):
        out = F.leaky_relu(input, negative_slope=self.negative_slope)
        return out * math.sqrt(2)


def make_kernel(k):
    k = torch.tensor(k, dtype=torch.float32)
    if k.ndim == 1:
        k = k[None, :] * k[:, None]
    k /= k.sum()
    return k


class Blur(nn.Module):

    def __init__(self, kernel, pad, upsample_factor=1):
        super().__init__()
        kernel = make_kernel(kernel)
        if upsample_factor > 1:
            kernel = kernel * upsample_factor ** 2
        self.register_buffer('kernel', kernel)
        self.pad = pad

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, pad=self.pad)
        return out


class EqualConv2d(nn.Module):

    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_channel, in_channel, kernel_size, kernel_size))
        self.scale = 1 / math.sqrt(in_channel * kernel_size ** 2)
        self.stride = stride
        self.padding = padding
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channel))
        else:
            self.bias = None

    def forward(self, input):
        out = F.conv2d(input, self.weight * self.scale, bias=self.bias, stride=self.stride, padding=self.padding)
        return out


class ConvLayer(nn.Sequential):

    def __init__(self, in_channel, out_channel, kernel_size, downsample=False, blur_kernel=[1, 3, 3, 1], bias=True, activate=True):
        layers = []
        if downsample:
            factor = 2
            p = len(blur_kernel) - factor + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2
            layers.append(Blur(blur_kernel, pad=(pad0, pad1)))
            stride = 2
            self.padding = 0
        else:
            stride = 1
            self.padding = kernel_size // 2
        layers.append(EqualConv2d(in_channel, out_channel, kernel_size, padding=self.padding, stride=stride, bias=bias and not activate))
        if activate:
            if bias:
                layers.append(FusedLeakyReLU(out_channel))
            else:
                layers.append(ScaledLeakyReLU(0.2))
        super().__init__(*layers)


def get_nonspade_norm_layer(opt, norm_type='instance'):

    def get_out_channel(layer):
        if hasattr(layer, 'out_channels'):
            return getattr(layer, 'out_channels')
        return layer.weight.size(0)

    def add_norm_layer(layer):
        nonlocal norm_type
        if norm_type.startswith('spectral'):
            layer = spectral_norm(layer)
            subnorm_type = norm_type[len('spectral'):]
        if subnorm_type == 'none' or len(subnorm_type) == 0:
            return layer
        if getattr(layer, 'bias', None) is not None:
            delattr(layer, 'bias')
            layer.register_parameter('bias', None)
        if subnorm_type == 'batch':
            norm_layer = nn.BatchNorm2d(get_out_channel(layer), affine=True)
        elif subnorm_type == 'sync_batch':
            norm_layer = SynchronizedBatchNorm2d(get_out_channel(layer), affine=True)
        elif subnorm_type == 'instance':
            norm_layer = nn.InstanceNorm2d(get_out_channel(layer), affine=False)
        else:
            raise ValueError('normalization layer %s is not recognized' % subnorm_type)
        return nn.Sequential(layer, norm_layer)
    return add_norm_layer


class ConvEncoder(BaseNetwork):
    """ Same architecture as the image discriminator """

    def __init__(self, opt):
        super().__init__()
        kw = 3
        pw = int(np.ceil((kw - 1.0) / 2))
        ndf = opt.ngf
        norm_layer = get_nonspade_norm_layer(opt, opt.norm_E)
        self.layer1 = norm_layer(nn.Conv2d(3, ndf, kw, stride=2, padding=pw))
        self.layer2 = norm_layer(nn.Conv2d(ndf * 1, ndf * 2, kw, stride=2, padding=pw))
        self.layer3 = norm_layer(nn.Conv2d(ndf * 2, ndf * 4, kw, stride=2, padding=pw))
        self.layer4 = norm_layer(nn.Conv2d(ndf * 4, ndf * 8, kw, stride=2, padding=pw))
        self.layer5 = norm_layer(nn.Conv2d(ndf * 8, ndf * 8, kw, stride=2, padding=pw))
        if opt.crop_size >= 256:
            self.layer6 = norm_layer(nn.Conv2d(ndf * 8, ndf * 8, kw, stride=2, padding=pw))
        self.so = s0 = 4
        self.fc_mu = nn.Linear(ndf * 8 * s0 * s0, 256)
        self.fc_var = nn.Linear(ndf * 8 * s0 * s0, 256)
        self.actvn = nn.LeakyReLU(0.2, False)
        self.opt = opt

    def forward(self, x):
        if x.size(2) != 256 or x.size(3) != 256:
            x = F.interpolate(x, size=(256, 256), mode='bilinear')
        x = self.layer1(x)
        x = self.layer2(self.actvn(x))
        x = self.layer3(self.actvn(x))
        x = self.layer4(self.actvn(x))
        x = self.layer5(self.actvn(x))
        if self.opt.crop_size >= 256:
            x = self.layer6(self.actvn(x))
        x = self.actvn(x)
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_var(x)
        return mu, logvar


class SPADEGenerator(BaseNetwork):

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(norm_G='spectralspadesyncbatch3x3')
        parser.add_argument('--num_upsampling_layers', choices=('normal', 'more', 'most'), default='normal', help="If 'more', adds upsampling layer between the two middle resnet blocks. If 'most', also add one more upsampling + resnet layer at the end of the generator")
        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        nf = opt.ngf
        self.sw, self.sh, self.scale_ratio = self.compute_latent_vector_size(opt)
        if opt.use_vae:
            self.fc = nn.Linear(opt.z_dim, 16 * nf * self.sw * self.sh)
        else:
            self.fc = nn.Conv2d(self.opt.semantic_nc, 16 * nf, 3, padding=1)
        self.head_0 = SPADEResnetBlock(16 * nf, 16 * nf, opt)
        self.G_middle_0 = SPADEResnetBlock(16 * nf, 16 * nf, opt)
        self.G_middle_1 = SPADEResnetBlock(16 * nf, 16 * nf, opt)
        self.ups = nn.ModuleList([SPADEResnetBlock(16 * nf, 8 * nf, opt), SPADEResnetBlock(8 * nf, 4 * nf, opt), SPADEResnetBlock(4 * nf, 2 * nf, opt), SPADEResnetBlock(2 * nf, 1 * nf, opt)])
        self.to_rgbs = nn.ModuleList([nn.Conv2d(8 * nf, 3, 3, padding=1), nn.Conv2d(4 * nf, 3, 3, padding=1), nn.Conv2d(2 * nf, 3, 3, padding=1), nn.Conv2d(1 * nf, 3, 3, padding=1)])
        self.up = nn.Upsample(scale_factor=2)

    def encode(self, input):
        h, w = input.size()[-2:]
        sh, sw = h // 2 ** self.scale_ratio, w // 2 ** self.scale_ratio
        x = F.interpolate(input, size=(sh, sw))
        return self.fc(x)

    def compute_latent_vector_size(self, opt):
        if opt.num_upsampling_layers == 'normal':
            num_up_layers = 5
        elif opt.num_upsampling_layers == 'more':
            num_up_layers = 6
        elif opt.num_upsampling_layers == 'most':
            num_up_layers = 7
        else:
            raise ValueError('opt.num_upsampling_layers [%s] not recognized' % opt.num_upsampling_layers)
        scale_ratio = num_up_layers
        sw = opt.crop_size // 2 ** num_up_layers
        sh = round(sw / opt.aspect_ratio)
        return sw, sh, scale_ratio

    def forward(self, input, seg=None):
        """
            20200307: Dangerous Change
            Add separable forward to allow different 
            input and segmentation maps...
            
            To return to original, simply add
            seg = input at begining, and disable the seg parameter.
            
            20200308: A more elegant solution:
            @ Allow forward to take default parameters.
            @ Allow customizable input encoding
            
            20200310: Merge fc into encode, since encoder directly outputs processed feature map.
            
            [TODO] @ Allow customizable segmap encoding?
        """
        if seg is None:
            seg = input
        x = self.encode(input)
        x = self.head_0(x, seg)
        x = self.up(x)
        x = self.G_middle_0(x, seg)
        x = self.G_middle_1(x, seg)
        if self.opt.is_test:
            phase = len(self.to_rgbs)
        else:
            phase = self.opt.train_phase + 1
        for i in range(phase):
            x = self.up(x)
            x = self.ups[i](x, seg)
        x = self.to_rgbs[phase - 1](F.leaky_relu(x, 0.2))
        x = torch.tanh(x)
        return x

    def mixed_guidance_forward(self, input, seg=None, n=0, mode='progressive'):
        """
            mixed_forward: input and seg are different images
            For the first n levels (including encoder)
            we use input, for the rest we use seg.
            
            If mode = 'progressive', the output's like: AAABBB
            If mode = 'one_plug', the output's like:    AAABAA
            If mode = 'one_ablate', the output's like:  BBBABB
        """
        if seg is None:
            return self.forward(input)
        if self.opt.is_test:
            phase = len(self.to_rgbs)
        else:
            phase = self.opt.train_phase + 1
        if mode == 'progressive':
            n = max(min(n, 4 + phase), 0)
            guide_list = [input] * n + [seg] * (4 + phase - n)
        elif mode == 'one_plug':
            n = max(min(n, 4 + phase - 1), 0)
            guide_list = [seg] * (4 + phase)
            guide_list[n] = input
        elif mode == 'one_ablate':
            if n > 3 + phase:
                return self.forward(input)
            guide_list = [input] * (4 + phase)
            guide_list[n] = seg
        x = self.encode(guide_list[0])
        x = self.head_0(x, guide_list[1])
        x = self.up(x)
        x = self.G_middle_0(x, guide_list[2])
        x = self.G_middle_1(x, guide_list[3])
        for i in range(phase):
            x = self.up(x)
            x = self.ups[i](x, guide_list[4 + i])
        x = self.to_rgbs[phase - 1](F.leaky_relu(x, 0.2))
        x = torch.tanh(x)
        return x


class SoftGate(nn.Module):
    COEFF = 12.0

    def __init__(self):
        super(SoftGate, self).__init__()

    def forward(self, x):
        return torch.sigmoid(x).mul(self.COEFF)


def lip2d(x, logit, kernel=3, stride=2, padding=1):
    weight = logit.exp()
    return F.avg_pool2d(x * weight, kernel, stride, padding) / F.avg_pool2d(weight, kernel, stride, padding)


class SimplifiedLIP(nn.Module):

    def __init__(self, channels):
        super(SimplifiedLIP, self).__init__()
        rp = channels
        self.logit = nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1, bias=False), nn.InstanceNorm2d(channels, affine=True), SoftGate())
        """
        OrderedDict((
            ('conv', nn.Conv2d(channels, channels, 3, padding=1, bias=False)),
            ('bn', nn.InstanceNorm2d(channels, affine=True)),
            ('gate', SoftGate()),
        ))
        """

    def init_layer(self):
        self.logit[0].weight.data.fill_(0.0)

    def forward(self, x):
        frac = lip2d(x, self.logit(x))
        return frac


class ContentAdaptiveSuppresor(BaseNetwork):

    def __init__(self, opt, sw, sh, n_2xdown, norm_layer=nn.InstanceNorm2d):
        super().__init__()
        self.sw = sw
        self.sh = sh
        self.max_ratio = 16
        self.n_2xdown = n_2xdown
        ngf = opt.ngf
        kw = 3
        pw = (kw - 1) // 2
        self.head = nn.Sequential(nn.Conv2d(opt.semantic_nc, ngf, kw, stride=1, padding=pw, bias=False), norm_layer(ngf), nn.ReLU())
        cur_ratio = 1
        for i in range(n_2xdown):
            next_ratio = min(cur_ratio * 2, self.max_ratio)
            model = [SimplifiedLIP(ngf * cur_ratio), nn.Conv2d(ngf * cur_ratio, ngf * next_ratio, kw, stride=1, padding=pw), norm_layer(ngf * next_ratio)]
            cur_ratio = next_ratio
            if i < n_2xdown - 1:
                model += [nn.ReLU(inplace=True)]
            setattr(self, 'encoder_%d' % i, nn.Sequential(*model))

    def forward(self, x):
        x = [self.head(x)]
        for i in range(self.n_2xdown):
            net = getattr(self, 'encoder_%d' % i)
            x = [net(x[0])] + x
        return x


class HiFaceGANGenerator(SPADEGenerator):

    def __init__(self, opt):
        super().__init__(opt)
        self.opt = opt
        nf = opt.ngf
        self.sw, self.sh, self.scale_ratio = self.compute_latent_vector_size(opt)
        if opt.use_vae:
            self.fc = nn.Linear(opt.z_dim, 16 * nf * self.sw * self.sh)
        else:
            self.fc = nn.Conv2d(self.opt.semantic_nc, 16 * nf, 3, padding=1)
        self.head_0 = SPADEResnetBlock(16 * nf, 16 * nf, opt, 16 * nf)
        self.G_middle_0 = SPADEResnetBlock(16 * nf, 16 * nf, opt, 16 * nf)
        self.G_middle_1 = SPADEResnetBlock(16 * nf, 16 * nf, opt, 16 * nf)
        self.ups = nn.ModuleList([SPADEResnetBlock(16 * nf, 8 * nf, opt, 8 * nf), SPADEResnetBlock(8 * nf, 4 * nf, opt, 4 * nf), SPADEResnetBlock(4 * nf, 2 * nf, opt, 2 * nf), SPADEResnetBlock(2 * nf, 1 * nf, opt, 1 * nf)])
        self.to_rgbs = nn.ModuleList([nn.Conv2d(8 * nf, 3, 3, padding=1), nn.Conv2d(4 * nf, 3, 3, padding=1), nn.Conv2d(2 * nf, 3, 3, padding=1), nn.Conv2d(1 * nf, 3, 3, padding=1)])
        self.up = nn.Upsample(scale_factor=2)
        self.encoder = ContentAdaptiveSuppresor(opt, self.sw, self.sh, self.scale_ratio)

    def nested_encode(self, x):
        return self.encoder(x)

    def forward(self, input):
        xs = self.nested_encode(input)
        x = self.encode(input)
        """
        print([_x.shape for _x in xs])
        print(x.shape)
        print(self.head_0)
        """
        x = self.head_0(x, xs[0])
        x = self.up(x)
        x = self.G_middle_0(x, xs[1])
        x = self.G_middle_1(x, xs[1])
        if self.opt.is_test:
            phase = len(self.to_rgbs)
        else:
            phase = self.opt.train_phase + 1
        for i in range(phase):
            x = self.up(x)
            x = self.ups[i](x, xs[i + 2])
        x = self.to_rgbs[phase - 1](F.leaky_relu(x, 0.2))
        x = torch.tanh(x)
        return x


class LIPEncoder(BaseNetwork):

    def __init__(self, opt, sw, sh, n_2xdown, norm_layer=nn.InstanceNorm2d):
        super().__init__()
        self.sw = sw
        self.sh = sh
        self.max_ratio = 16
        ngf = opt.ngf
        kw = 3
        pw = (kw - 1) // 2
        model = [nn.Conv2d(opt.semantic_nc, ngf, kw, stride=1, padding=pw, bias=False), norm_layer(ngf), nn.ReLU()]
        cur_ratio = 1
        for i in range(n_2xdown):
            next_ratio = min(cur_ratio * 2, self.max_ratio)
            model += [SimplifiedLIP(ngf * cur_ratio), nn.Conv2d(ngf * cur_ratio, ngf * next_ratio, kw, stride=1, padding=pw), norm_layer(ngf * next_ratio)]
            cur_ratio = next_ratio
            if i < n_2xdown - 1:
                model += [nn.ReLU(inplace=True)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class LIPSPADEGenerator(SPADEGenerator):
    """
        20200309: SPADEGenerator with a learnable feature encoder
        Encoder design: Local Importance-based Pooling (Ziteng Gao et.al.,ICCV 2019)
    """

    def __init__(self, opt):
        super().__init__(opt)
        self.lip_encoder = LIPEncoder(opt, self.sw, self.sh, self.scale_ratio)

    def encode(self, x):
        return self.lip_encoder(x)


class NoiseClassPredictor(nn.Module):
    """
        Input: nc*sw*sw tensor, either from clean or corrupted images
        Output: n-dim tensor indicating the loss type (or intensity?)
    """

    def __init__(self, opt, sw, nc, outdim):
        super().__init__()
        nbottleneck = 256
        middim = 256
        conv = [nn.Conv2d(nc, nbottleneck, 1, stride=1), nn.InstanceNorm2d(nbottleneck), nn.LeakyReLU(0.2, inplace=True)]
        while sw > 4:
            sw = sw // 2
            conv += [nn.Conv2d(nbottleneck, nbottleneck, 3, stride=2, padding=1), nn.InstanceNorm2d(nbottleneck), nn.LeakyReLU(0.2, inplace=True)]
        self.conv = nn.Sequential(*conv)
        indim = sw * sw * nbottleneck
        self.fc = nn.Sequential(nn.Linear(indim, middim), nn.BatchNorm1d(middim), nn.LeakyReLU(0.2, inplace=True), nn.Linear(middim, outdim))

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.shape[0], -1)
        return self.fc(x)


class NoiseIntensityPredictor(nn.Module):
    """
        Input: nc*sw*sw tensor, either from clean or corrupted images
        Output: 1-dim tensor indicating the loss intensity
    """

    def __init__(self, opt, sw, nc, outdim):
        super().__init__()
        nbottleneck = 256
        middim = 256
        conv = [nn.Conv2d(nc, nbottleneck, 1, stride=1), nn.BatchNorm2d(nbottleneck), nn.LeakyReLU(0.2, inplace=True)]
        while sw > 4:
            sw = sw // 2
            conv += [nn.Conv2d(nbottleneck, nbottleneck, 3, stride=2, padding=1), nn.BatchNorm2d(nbottleneck), nn.LeakyReLU(0.2, inplace=True)]
        self.conv = nn.Sequential(*conv)
        indim = sw * sw * nbottleneck
        self.fc = nn.Sequential(nn.Linear(indim, middim), nn.BatchNorm1d(middim), nn.LeakyReLU(0.2, inplace=True), nn.Linear(middim, outdim))

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x.squeeze()


data = torch.rand((8, 4))


class SubAddGenerator(SPADEGenerator):
    """
        20200311: 
        This generator contains a complete set
        of self-supervised training scheme
        that requires a separate dataloader.
        
        The self-supervised pre-training is 
        implemented as a clean interface.
        SubAddGenerator::train_E(self, dataloader, epochs)
        
        For the upperlevel Pix2Pix_Model, 
        two things to be done:
        A) Run the pretrain script
        B) Save encoder and adjust learning rate.
        
        -----------------------------------------
        20200312:
        Pre-test problem: The discriminator is hard to test real vs fake
        Also, using residual of feature maps ain't work...
        
        Cause: The feature map is too close to be properly separated.
        
        Try to test on one single reduction: arbitrary ratio of downsampling
        try to estimate the reduction ratio?
    """

    def __init__(self, opt):
        super().__init__(opt)
        self.encoder = LIPEncoder(opt, self.sw, self.sh, self.scale_ratio)
        self.dis_nc = self.opt.ngf * min(16, 2 ** self.scale_ratio)
        self.discriminator = NoiseIntensityPredictor(opt, self.sw, self.dis_nc, 1)
        if opt.isTrain:
            self.attach_dataloader(opt)
            self.noise_dim = opt.noise_dim
            self.l1_loss = nn.L1Loss()
            self.gan_loss = nn.MSELoss()
            beta1, beta2 = opt.beta1, opt.beta2
            if opt.no_TTUR:
                G_lr, D_lr = opt.lr, opt.lr
            else:
                G_lr, D_lr = opt.lr / 2, opt.lr * 2
            self.optimizer_E = torch.optim.Adam(self.encoder.parameters(), lr=G_lr, betas=(beta1, beta2))
            self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=D_lr / 2, betas=(beta1, beta2))

    def _create_auxiliary_opt(self, opt):
        """
            Create auxiliary options
            change necessary params
            --- dataroot
            --- dataset_mode
        """
        aux_opt = copy.deepcopy(opt)
        aux_opt.dataroot = opt.dataroot_assist
        aux_opt.dataset_mode = 'assist'
        aux_opt.batchSize = 4
        aux_opt.nThreads = 4
        return aux_opt

    def attach_dataloader(self, opt):
        aux_opt = self._create_auxiliary_opt(opt)
        self.loader = data.create_dataloader(aux_opt)

    def encode(self, x):
        return self.encoder(x)

    def process_input(self, data):
        self.clean = data['clean']
        self.noisy = data['noisy']
        self.noise_label = data['label']
        self.clean_label = torch.ones_like(self.noise_label)

    def update_E(self):
        bundle_in = torch.cat((self.clean, self.noisy), dim=0)
        bundle_out = self.encode(bundle_in)
        nb = bundle_in.shape[0] // 2
        F_real, F_fake = bundle_out[:nb], bundle_out[nb:]
        pred_fake = self.discriminator(F_fake)
        loss_l1 = self.l1_loss(F_fake, F_real)
        loss_gan = self.gan_loss(pred_fake, self.clean_label)
        loss_sum = loss_l1 * 10 + loss_gan
        self.optimizer_E.zero_grad()
        loss_sum.backward()
        self.optimizer_E.step()
        self.loss_l1 = loss_l1.item()
        self.loss_gan_E = loss_gan.item()
        self.loss_sum = loss_sum.item()

    def update_D(self):
        with torch.no_grad():
            bundle_in = torch.cat((self.clean, self.noisy), dim=0)
            bundle_out = self.encode(bundle_in)
            nb = bundle_in.shape[0] // 2
            F_fake = bundle_out[nb:] / (bundle_out[:nb] + 1e-06)
            F_real = torch.ones_like(F_fake, requires_grad=False)
        pred_real = self.discriminator(F_real)
        loss_real = self.gan_loss(pred_real, self.clean_label)
        pred_fake = self.discriminator(F_fake.detach())
        loss_fake = self.gan_loss(pred_fake, self.noise_label)
        loss_sum = (loss_real + loss_fake * self.opt.noise_dim) / 2
        self.optimizer_D.zero_grad()
        loss_sum.backward()
        self.optimizer_D.step()
        self.loss_gan_D_real = loss_real.item()
        self.loss_gan_D_fake = loss_fake.item()

    def debug_D(self):
        with torch.no_grad():
            bundle_in = torch.cat((self.clean, self.noisy), dim=0)
            bundle_out = self.encode(bundle_in)
            nb = bundle_in.shape[0] // 2
            F_real, F_fake = bundle_out[:nb], bundle_out[nb:]
            F_res = F_fake - F_real
        pred_real = self.discriminator(F_real)
        pred_fake = self.discriminator(F_res.detach())
        None
        real_acc = (pred_real == 0).sum().item() / pred_real.shape[0]
        fake_acc = (pred_fake == self.noise_label).sum().item() / pred_fake.shape[0]
        None

    def log(self, epoch, i):
        logstring = '   Epoch [%d] iter [%d]: ' % (epoch, i)
        logstring += 'l1: %.4f ' % self.loss_l1
        logstring += 'gen: %.4f ' % self.loss_gan_E
        logstring += 'E_sum: %.4f ' % self.loss_sum
        logstring += 'dis_real: %.4f ' % self.loss_gan_D_real
        logstring += 'dis_fake: %.4f' % self.loss_gan_D_fake
        None

    def train_E(self, epochs):
        pretrained_ckpt_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name, 'pretrained_net_E_%d.pth' % epochs)
        None
        None
        if os.path.isfile(pretrained_ckpt_dir):
            state_dict_E = torch.load(pretrained_ckpt_dir)
            self.encoder.load_state_dict(state_dict_E)
            None
        else:
            None
            for epoch in range(1, epochs + 1):
                for i, data in enumerate(self.loader):
                    self.process_input(data)
                    self.update_E()
                    self.update_D()
                    if i % 10 == 0:
                        self.log(epoch, i)
                None
                torch.save(self.encoder.state_dict(), os.path.join(self.opt.checkpoints_dir, self.opt.name, 'pretrained_net_E_%d.pth' % epoch))


class ContrasiveGenerator(SPADEGenerator):

    def __init__(self, opt):
        super().__init__(opt)
        self.encoder = LIPEncoder(opt, self.sw, self.sh, self.scale_ratio)
        if opt.isTrain:
            self.attach_dataloader(opt)
            self.noise_dim = opt.noise_dim
            self.l1_loss = nn.L1Loss()
            beta1, beta2 = opt.beta1, opt.beta2
            self.optimizer_E = torch.optim.Adam(self.encoder.parameters(), lr=opt.lr, betas=(beta1, beta2))

    def _create_auxiliary_opt(self, opt):
        """
            Create auxiliary options
            change necessary params
            --- dataroot
            --- dataset_mode
        """
        aux_opt = copy.deepcopy(opt)
        aux_opt.dataroot = opt.dataroot_assist
        aux_opt.dataset_mode = 'assist'
        aux_opt.batchSize = 8
        aux_opt.nThreads = 4
        return aux_opt

    def attach_dataloader(self, opt):
        aux_opt = self._create_auxiliary_opt(opt)
        self.loader = data.create_dataloader(aux_opt)

    def encode(self, x):
        return self.encoder(x)

    def process_input(self, data):
        self.clean = data['clean']
        self.noisy = data['noisy']

    def update_E(self):
        bundle_in = torch.cat((self.clean, self.noisy), dim=0)
        bundle_out = self.encode(bundle_in)
        nb = bundle_in.shape[0] // 2
        F_real, F_fake = bundle_out[:nb], bundle_out[nb:]
        loss_l1 = self.l1_loss(F_fake, F_real)
        self.optimizer_E.zero_grad()
        loss_l1.backward()
        self.optimizer_E.step()
        self.loss_l1 = loss_l1.item()

    def log(self, epoch, i):
        logstring = '   Epoch [%d] iter [%d]: ' % (epoch, i)
        logstring += 'l1: %.4f ' % self.loss_l1
        None

    def train_E(self, epochs):
        pretrained_ckpt_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name, 'pretrained_net_E_%d.pth' % epochs)
        None
        None
        if os.path.isfile(pretrained_ckpt_dir):
            state_dict_E = torch.load(pretrained_ckpt_dir)
            self.encoder.load_state_dict(state_dict_E)
            None
        else:
            None
            for epoch in range(1, epochs + 1):
                for i, data in enumerate(self.loader):
                    self.process_input(data)
                    self.update_E()
                    if i % 10 == 0:
                        self.log(epoch, i)
                None
                torch.save(self.encoder.state_dict(), os.path.join(self.opt.checkpoints_dir, self.opt.name, 'pretrained_net_E_%d.pth' % epoch))


class GANLoss(nn.Module):

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0, tensor=torch.FloatTensor, opt=None):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_tensor = None
        self.fake_label_tensor = None
        self.zero_tensor = None
        self.Tensor = tensor
        self.gan_mode = gan_mode
        self.opt = opt
        if gan_mode == 'ls':
            pass
        elif gan_mode == 'original':
            pass
        elif gan_mode == 'w':
            pass
        elif gan_mode == 'hinge':
            pass
        else:
            raise ValueError('Unexpected gan_mode {}'.format(gan_mode))

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            if self.real_label_tensor is None:
                self.real_label_tensor = self.Tensor(1).fill_(self.real_label)
                self.real_label_tensor.requires_grad_(False)
            return self.real_label_tensor.expand_as(input)
        else:
            if self.fake_label_tensor is None:
                self.fake_label_tensor = self.Tensor(1).fill_(self.fake_label)
                self.fake_label_tensor.requires_grad_(False)
            return self.fake_label_tensor.expand_as(input)

    def get_zero_tensor(self, input):
        if self.zero_tensor is None:
            self.zero_tensor = self.Tensor(1).fill_(0)
            self.zero_tensor.requires_grad_(False)
        return self.zero_tensor.expand_as(input)

    def loss(self, input, target_is_real, for_discriminator=True):
        if self.gan_mode == 'original':
            target_tensor = self.get_target_tensor(input, target_is_real)
            loss = F.binary_cross_entropy_with_logits(input, target_tensor)
            return loss
        elif self.gan_mode == 'ls':
            target_tensor = self.get_target_tensor(input, target_is_real)
            return F.mse_loss(input, target_tensor)
        elif self.gan_mode == 'hinge':
            if for_discriminator:
                if target_is_real:
                    minval = torch.min(input - 1, self.get_zero_tensor(input))
                    loss = -torch.mean(minval)
                else:
                    minval = torch.min(-input - 1, self.get_zero_tensor(input))
                    loss = -torch.mean(minval)
            else:
                assert target_is_real, "The generator's hinge loss must be aiming for real"
                loss = -torch.mean(input)
            return loss
        elif target_is_real:
            return -input.mean()
        else:
            return input.mean()

    def __call__(self, input, target_is_real, for_discriminator=True):
        if isinstance(input, list):
            loss = 0
            for pred_i in input:
                if isinstance(pred_i, list):
                    pred_i = pred_i[-1]
                loss_tensor = self.loss(pred_i, target_is_real, for_discriminator)
                bs = 1 if len(loss_tensor.size()) == 0 else loss_tensor.size(0)
                new_loss = torch.mean(loss_tensor.view(bs, -1), dim=1)
                loss += new_loss
            return loss / len(input)
        else:
            return self.loss(input, target_is_real, for_discriminator)


class VGGLoss(nn.Module):

    def __init__(self, gpu_ids):
        super(VGGLoss, self).__init__()
        self.vgg = VGG19() if len(gpu_ids) > 0 else VGG19()
        self.criterion = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

    def forward(self, x, y):
        if x.shape[-2:] != y.shape[-2:]:
            y = torch.nn.functional.interpolate(y, tuple(x.shape[-2:]))
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss


class KLDLoss(nn.Module):

    def forward(self, mu, logvar):
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())


class Pix2PixModel(torch.nn.Module):

    @staticmethod
    def modify_commandline_options(parser, is_train):
        networks.modify_commandline_options(parser, is_train)
        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.FloatTensor = torch.FloatTensor if self.use_gpu() else torch.FloatTensor
        self.ByteTensor = torch.ByteTensor if self.use_gpu() else torch.ByteTensor
        self.netG, self.netD, self.netE = self.initialize_networks(opt)
        if opt.isTrain:
            self.criterionGAN = networks.GANLoss(opt.gan_mode, tensor=self.FloatTensor, opt=self.opt)
            self.criterionFeat = torch.nn.L1Loss()
            if not opt.no_vgg_loss:
                self.criterionVGG = networks.VGGLoss(self.opt.gpu_ids)
            if opt.use_vae:
                self.KLDLoss = networks.KLDLoss()

    def forward(self, data, mode):
        input_semantics, real_image = self.preprocess_input(data)
        if mode == 'generator':
            g_loss, generated = self.compute_generator_loss(input_semantics, real_image)
            return g_loss, generated
        elif mode == 'discriminator':
            d_loss = self.compute_discriminator_loss(input_semantics, real_image)
            return d_loss
        elif mode == 'inference':
            with torch.no_grad():
                fake_image = self.generate_fake(input_semantics)
            return fake_image
        elif mode == 'inference2':
            with torch.no_grad():
                fake_image = self.netG(input_semantics)
                return fake_image
        else:
            raise ValueError('|mode| is invalid')

    def preprocess_input(self, data):
        if self.use_gpu():
            data['label'] = data['label']
            data['image'] = data['image']
        return data['label'], data['image']

    def compute_generator_loss(self, input_semantics, real_image):
        G_losses = {}
        fake_image = self.generate_fake(input_semantics)
        pred_fake, pred_real = self.discriminate(input_semantics, fake_image, real_image)
        G_losses['GAN'] = self.criterionGAN(pred_fake, True, for_discriminator=False)
        if not self.opt.no_ganFeat_loss:
            num_D = len(pred_fake)
            GAN_Feat_loss = self.FloatTensor(1).fill_(0)
            for i in range(num_D):
                num_intermediate_outputs = len(pred_fake[i]) - 1
                for j in range(num_intermediate_outputs):
                    unweighted_loss = self.criterionFeat(pred_fake[i][j], pred_real[i][j].detach())
                    GAN_Feat_loss += unweighted_loss * self.opt.lambda_feat / num_D
            G_losses['GAN_Feat'] = GAN_Feat_loss
        h, w = fake_image.shape[-2:]
        if not self.opt.no_vgg_loss and min(w, h) >= 64:
            G_losses['VGG'] = self.criterionVGG(fake_image, real_image) * self.opt.lambda_vgg
        return G_losses, fake_image

    def compute_discriminator_loss(self, input_semantics, real_image):
        D_losses = {}
        with torch.no_grad():
            fake_image = self.generate_fake(input_semantics)
            fake_image = fake_image.detach()
            fake_image.requires_grad_()
        pred_fake, pred_real = self.discriminate(input_semantics, fake_image, real_image)
        D_losses['D_Fake'] = self.criterionGAN(pred_fake, False, for_discriminator=True)
        D_losses['D_real'] = self.criterionGAN(pred_real, True, for_discriminator=True)
        return D_losses

    def generate_fake(self, input_semantics):
        fake_image = self.netG(input_semantics)
        return fake_image

    def discriminate(self, input_semantics, fake_image, real_image):
        h, w = fake_image.shape[-2:]
        if fake_image.shape[-2:] != input_semantics.shape[-2:]:
            semantics = torch.nn.functional.interpolate(input_semantics, (h, w))
            real = torch.nn.functional.interpolate(real_image, (h, w))
            fake_concat = torch.cat([semantics, fake_image], dim=1)
            real_concat = torch.cat([semantics, real], dim=1)
        else:
            fake_concat = torch.cat([input_semantics, fake_image], dim=1)
            real_concat = torch.cat([input_semantics, real_image], dim=1)
        fake_and_real = torch.cat([fake_concat, real_concat], dim=0)
        discriminator_out = self.netD(fake_and_real)
        pred_fake, pred_real = self.divide_pred(discriminator_out)
        return pred_fake, pred_real

    def encode_z(self, real_image):
        mu, logvar = self.netE(real_image)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def create_optimizers(self, opt):
        G_params = list(self.netG.parameters())
        if opt.use_vae:
            G_params += list(self.netE.parameters())
        if opt.isTrain:
            D_params = list(self.netD.parameters())
        beta1, beta2 = opt.beta1, opt.beta2
        if opt.no_TTUR:
            G_lr, D_lr = opt.lr, opt.lr
        else:
            G_lr, D_lr = opt.lr / 2, opt.lr * 2
        optimizer_G = torch.optim.Adam(G_params, lr=G_lr, betas=(beta1, beta2))
        optimizer_D = torch.optim.Adam(D_params, lr=D_lr, betas=(beta1, beta2))
        return optimizer_G, optimizer_D

    def save(self, epoch):
        util.save_network(self.netG, 'G', epoch, self.opt)
        util.save_network(self.netD, 'D', epoch, self.opt)
        if self.opt.use_vae:
            util.save_network(self.netE, 'E', epoch, self.opt)

    def initialize_networks(self, opt):
        netG = networks.define_G(opt)
        netD = networks.define_D(opt) if opt.isTrain else None
        netE = networks.define_E(opt) if opt.use_vae else None
        if not opt.isTrain or opt.continue_train:
            netG = util.load_network(netG, 'G', opt.which_epoch, opt)
            if opt.isTrain:
                netD = util.load_network(netD, 'D', opt.which_epoch, opt)
            if opt.use_vae:
                netE = util.load_network(netE, 'E', opt.which_epoch, opt)
        return netG, netD, netE

    def divide_pred(self, pred):
        if type(pred) == list:
            fake = []
            real = []
            for p in pred:
                fake.append([tensor[:tensor.size(0) // 2] for tensor in p])
                real.append([tensor[tensor.size(0) // 2:] for tensor in p])
        else:
            fake = pred[:pred.size(0) // 2]
            real = pred[pred.size(0) // 2:]
        return fake, real

    def get_edges(self, t):
        edge = self.ByteTensor(t.size()).zero_()
        edge[:, :, :, 1:] = edge[:, :, :, 1:] | (t[:, :, :, 1:] != t[:, :, :, :-1])
        edge[:, :, :, :-1] = edge[:, :, :, :-1] | (t[:, :, :, 1:] != t[:, :, :, :-1])
        edge[:, :, 1:, :] = edge[:, :, 1:, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
        edge[:, :, :-1, :] = edge[:, :, :-1, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
        return edge.float()

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std) + mu

    def use_gpu(self):
        return len(self.opt.gpu_ids) > 0


class AADLayer(nn.Module):

    def __init__(self, c_x, attr_c, c_id):
        super(AADLayer, self).__init__()
        self.attr_c = attr_c
        self.c_id = c_id
        self.c_x = c_x
        self.conv1 = nn.Conv2d(attr_c, c_x, kernel_size=1, stride=1, padding=0, bias=True)
        self.conv2 = nn.Conv2d(attr_c, c_x, kernel_size=1, stride=1, padding=0, bias=True)
        self.fc1 = nn.Linear(c_id, c_x)
        self.fc2 = nn.Linear(c_id, c_x)
        self.norm = nn.InstanceNorm2d(c_x, affine=False)
        self.conv_h = nn.Conv2d(c_x, 1, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, h_in, z_attr, z_id):
        h = self.norm(h_in)
        gamma_attr = self.conv1(z_attr)
        beta_attr = self.conv2(z_attr)
        gamma_id = self.fc1(z_id)
        beta_id = self.fc2(z_id)
        A = gamma_attr * h + beta_attr
        gamma_id = gamma_id.reshape(h.shape[0], self.c_x, 1, 1).expand_as(h)
        beta_id = beta_id.reshape(h.shape[0], self.c_x, 1, 1).expand_as(h)
        I = gamma_id * h + beta_id
        M = torch.sigmoid(self.conv_h(h))
        out = (torch.ones_like(M) - M) * A + M * I
        return out


class AddBlocksSequential(nn.Sequential):

    def forward(self, *inputs):
        h, z_attr, z_id = inputs
        for i, module in enumerate(self._modules.values()):
            if i % 3 == 0 and i > 0:
                inputs = inputs, z_attr, z_id
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs


class AAD_ResBlk(nn.Module):

    def __init__(self, cin, cout, c_attr, c_id, num_blocks):
        super(AAD_ResBlk, self).__init__()
        self.cin = cin
        self.cout = cout
        add_blocks = []
        for i in range(num_blocks):
            out = cin if i < num_blocks - 1 else cout
            add_blocks.extend([AADLayer(cin, c_attr, c_id), nn.ReLU(inplace=True), nn.Conv2d(cin, out, kernel_size=3, stride=1, padding=1, bias=False)])
        self.add_blocks = AddBlocksSequential(*add_blocks)
        if cin != cout:
            last_add_block = [AADLayer(cin, c_attr, c_id), nn.ReLU(inplace=True), nn.Conv2d(cin, cout, kernel_size=3, stride=1, padding=1, bias=False)]
            self.last_add_block = AddBlocksSequential(*last_add_block)

    def forward(self, h, z_attr, z_id):
        x = self.add_blocks(h, z_attr, z_id)
        if self.cin != self.cout:
            h = self.last_add_block(h, z_attr, z_id)
        x = x + h
        return x


class deconv4x4(nn.Module):

    def __init__(self, in_c, out_c, norm=nn.BatchNorm2d):
        super(deconv4x4, self).__init__()
        self.deconv = nn.ConvTranspose2d(in_channels=in_c, out_channels=out_c, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn = norm(out_c)
        self.lrelu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, input, skip, backbone):
        x = self.deconv(input)
        x = self.bn(x)
        x = self.lrelu(x)
        if backbone == 'linknet':
            return x + skip
        else:
            return torch.cat((x, skip), dim=1)


def conv4x4(in_c, out_c, norm=nn.BatchNorm2d):
    return nn.Sequential(nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=4, stride=2, padding=1, bias=False), norm(out_c), nn.LeakyReLU(0.1, inplace=True))


def weight_init(m):
    if isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 0.001)
        m.bias.data.zero_()
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight.data)
    if isinstance(m, nn.ConvTranspose2d):
        nn.init.xavier_normal_(m.weight.data)


class MLAttrEncoder(nn.Module):

    def __init__(self, backbone):
        super(MLAttrEncoder, self).__init__()
        self.backbone = backbone
        self.conv1 = conv4x4(3, 32)
        self.conv2 = conv4x4(32, 64)
        self.conv3 = conv4x4(64, 128)
        self.conv4 = conv4x4(128, 256)
        self.conv5 = conv4x4(256, 512)
        self.conv6 = conv4x4(512, 1024)
        self.conv7 = conv4x4(1024, 1024)
        if backbone == 'unet':
            self.deconv1 = deconv4x4(1024, 1024)
            self.deconv2 = deconv4x4(2048, 512)
            self.deconv3 = deconv4x4(1024, 256)
            self.deconv4 = deconv4x4(512, 128)
            self.deconv5 = deconv4x4(256, 64)
            self.deconv6 = deconv4x4(128, 32)
        elif backbone == 'linknet':
            self.deconv1 = deconv4x4(1024, 1024)
            self.deconv2 = deconv4x4(1024, 512)
            self.deconv3 = deconv4x4(512, 256)
            self.deconv4 = deconv4x4(256, 128)
            self.deconv5 = deconv4x4(128, 64)
            self.deconv6 = deconv4x4(64, 32)
        self.apply(weight_init)

    def forward(self, Xt):
        feat1 = self.conv1(Xt)
        feat2 = self.conv2(feat1)
        feat3 = self.conv3(feat2)
        feat4 = self.conv4(feat3)
        feat5 = self.conv5(feat4)
        feat6 = self.conv6(feat5)
        z_attr1 = self.conv7(feat6)
        z_attr2 = self.deconv1(z_attr1, feat6, self.backbone)
        z_attr3 = self.deconv2(z_attr2, feat5, self.backbone)
        z_attr4 = self.deconv3(z_attr3, feat4, self.backbone)
        z_attr5 = self.deconv4(z_attr4, feat3, self.backbone)
        z_attr6 = self.deconv5(z_attr5, feat2, self.backbone)
        z_attr7 = self.deconv6(z_attr6, feat1, self.backbone)
        z_attr8 = F.interpolate(z_attr7, scale_factor=2, mode='bilinear', align_corners=True)
        return z_attr1, z_attr2, z_attr3, z_attr4, z_attr5, z_attr6, z_attr7, z_attr8


class AADGenerator(nn.Module):

    def __init__(self, backbone, c_id=256, num_blocks=2):
        super(AADGenerator, self).__init__()
        self.up1 = nn.ConvTranspose2d(c_id, 1024, kernel_size=2, stride=1, padding=0)
        self.AADBlk1 = AAD_ResBlk(1024, 1024, 1024, c_id, num_blocks)
        if backbone == 'linknet':
            self.AADBlk2 = AAD_ResBlk(1024, 1024, 1024, c_id, num_blocks)
            self.AADBlk3 = AAD_ResBlk(1024, 1024, 512, c_id, num_blocks)
            self.AADBlk4 = AAD_ResBlk(1024, 512, 256, c_id, num_blocks)
            self.AADBlk5 = AAD_ResBlk(512, 256, 128, c_id, num_blocks)
            self.AADBlk6 = AAD_ResBlk(256, 128, 64, c_id, num_blocks)
            self.AADBlk7 = AAD_ResBlk(128, 64, 32, c_id, num_blocks)
            self.AADBlk8 = AAD_ResBlk(64, 3, 32, c_id, num_blocks)
        else:
            self.AADBlk2 = AAD_ResBlk(1024, 1024, 2048, c_id, num_blocks)
            self.AADBlk3 = AAD_ResBlk(1024, 1024, 1024, c_id, num_blocks)
            self.AADBlk4 = AAD_ResBlk(1024, 512, 512, c_id, num_blocks)
            self.AADBlk5 = AAD_ResBlk(512, 256, 256, c_id, num_blocks)
            self.AADBlk6 = AAD_ResBlk(256, 128, 128, c_id, num_blocks)
            self.AADBlk7 = AAD_ResBlk(128, 64, 64, c_id, num_blocks)
            self.AADBlk8 = AAD_ResBlk(64, 3, 64, c_id, num_blocks)
        self.apply(weight_init)

    def forward(self, z_attr, z_id):
        m = self.up1(z_id.reshape(z_id.shape[0], -1, 1, 1))
        m2 = F.interpolate(self.AADBlk1(m, z_attr[0], z_id), scale_factor=2, mode='bilinear', align_corners=True)
        m3 = F.interpolate(self.AADBlk2(m2, z_attr[1], z_id), scale_factor=2, mode='bilinear', align_corners=True)
        m4 = F.interpolate(self.AADBlk3(m3, z_attr[2], z_id), scale_factor=2, mode='bilinear', align_corners=True)
        m5 = F.interpolate(self.AADBlk4(m4, z_attr[3], z_id), scale_factor=2, mode='bilinear', align_corners=True)
        m6 = F.interpolate(self.AADBlk5(m5, z_attr[4], z_id), scale_factor=2, mode='bilinear', align_corners=True)
        m7 = F.interpolate(self.AADBlk6(m6, z_attr[5], z_id), scale_factor=2, mode='bilinear', align_corners=True)
        m8 = F.interpolate(self.AADBlk7(m7, z_attr[6], z_id), scale_factor=2, mode='bilinear', align_corners=True)
        y = self.AADBlk8(m8, z_attr[7], z_id)
        return torch.tanh(y)


def MLAttrEncoderResnet(**kwargs):
    model = ResNet(Bottleneck, [2, 2, 2, 2, 2, 2], **kwargs)
    return model


class AEI_Net(nn.Module):

    def __init__(self, backbone, num_blocks=2, c_id=256):
        super(AEI_Net, self).__init__()
        if backbone in ['unet', 'linknet']:
            self.encoder = MLAttrEncoder(backbone)
        elif backbone == 'resnet':
            self.encoder = MLAttrEncoderResnet()
        self.generator = AADGenerator(backbone, c_id, num_blocks)

    def forward(self, Xt, z_id):
        attr = self.encoder(Xt)
        Y = self.generator(attr, z_id)
        return Y, attr

    def get_attr(self, X):
        return self.encoder(X)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (BasicBlock,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (BatchNorm2dReimpl,
     lambda: ([], {'num_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (DataParallelWithCallback,
     lambda: ([], {'module': torch.nn.ReLU()}),
     lambda: ([], {'input': torch.rand([4, 4])})),
    (DummyNet,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {})),
    (EqualConv2d,
     lambda: ([], {'in_channel': 4, 'out_channel': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Foo,
     lambda: ([], {'size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (FrozenBatchNorm2d,
     lambda: ([], {'n': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (IRBlock,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (InstanceNorm2d,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (KLDLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (MultiscaleDiscriminator,
     lambda: ([], {'input_nc': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (NLayerDiscriminator,
     lambda: ([], {'input_nc': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (NoiseClassPredictor,
     lambda: ([], {'opt': SimpleNamespace(), 'sw': 4, 'nc': 4, 'outdim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (NoiseIntensityPredictor,
     lambda: ([], {'opt': SimpleNamespace(), 'sw': 4, 'nc': 4, 'outdim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (SEBlock,
     lambda: ([], {'channel': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ScaledLeakyReLU,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (SimplifiedLIP,
     lambda: ([], {'channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (SoftGate,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (SyncBatchNorm,
     lambda: ([], {'num_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (TestModel,
     lambda: ([], {'args': SimpleNamespace(layers=1, dim=4, bias=4)}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (VGG19,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {})),
    (mLSTMRNNCell,
     lambda: ([], {'input_size': 4, 'hidden_size': 4}),
     lambda: ([torch.rand([4, 4])], {})),
    (tofp16,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
]

