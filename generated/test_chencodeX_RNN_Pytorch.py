
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


from torch import nn


import torch.nn.functional as f


from torch.autograd import Variable


import torch.nn as nn


from torch import optim


import numpy as np


class BMSELoss(torch.nn.Module):

    def __init__(self):
        super(BMSELoss, self).__init__()
        self.w_l = [1, 2, 5, 10, 30]
        self.y_l = [0.283, 0.353, 0.424, 0.565, 1]

    def forward(self, x, y):
        w = y.clone()
        for i in range(len(self.w_l)):
            w[w < self.y_l[i]] = self.w_l[i]
        return torch.mean(w * (y - x) ** 2)


class ConvGRUCell(nn.Module):

    def __init__(self, input_size, hidden_size, kernel_size):
        super(ConvGRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.kernel_size = kernel_size
        self.dropout = nn.Dropout(p=0.5)
        self.ConvGates = nn.Conv2d(self.input_size + self.hidden_size, 2 * self.hidden_size, self.kernel_size, padding=self.kernel_size // 2)
        self.Conv_ct = nn.Conv2d(self.input_size + self.hidden_size, self.hidden_size, self.kernel_size, padding=self.kernel_size // 2)
        dtype = torch.FloatTensor

    def forward(self, input, hidden):
        if hidden is None:
            size_h = [input.data.size()[0], self.hidden_size] + list(input.data.size()[2:])
            hidden = Variable(torch.zeros(size_h))
        if input is None:
            size_h = [hidden.data.size()[0], self.input_size] + list(hidden.data.size()[2:])
            input = Variable(torch.zeros(size_h))
        c1 = self.ConvGates(torch.cat((input, hidden), 1))
        rt, ut = c1.chunk(2, 1)
        reset_gate = self.dropout(f.sigmoid(rt))
        update_gate = self.dropout(f.sigmoid(ut))
        gated_hidden = torch.mul(reset_gate, hidden)
        p1 = self.Conv_ct(torch.cat((input, gated_hidden), 1))
        ct = f.tanh(p1)
        next_h = torch.mul(update_gate, hidden) + (1 - update_gate) * ct
        return next_h


class CLSTM_cell(nn.Module):
    """Initialize a basic Conv LSTM cell.
    Args:
      shape: int tuple thats the height and width of the hidden states h and c()
      filter_size: int that is the height and width of the filters
      num_features: int thats the num of channels of the states, like hidden_size

    """

    def __init__(self, shape, input_chans, filter_size, num_features):
        super(CLSTM_cell, self).__init__()
        self.shape = shape
        self.input_chans = input_chans
        self.filter_size = filter_size
        self.num_features = num_features
        self.dropout = nn.Dropout(p=0.5)
        self.padding = (filter_size - 1) / 2
        self.conv = nn.Conv2d(self.input_chans + self.num_features, 4 * self.num_features, self.filter_size, 1, self.padding)

    def forward(self, input, hidden_state):
        hidden, c = hidden_state
        combined = torch.cat((input, hidden), 1)
        A = self.conv(combined)
        ai, af, ao, ag = torch.split(A, self.num_features, dim=1)
        i = torch.sigmoid(ai)
        i = self.dropout(i)
        f = torch.sigmoid(af)
        f = self.dropout(f)
        o = torch.sigmoid(ao)
        o = self.dropout(o)
        g = torch.tanh(ag)
        g = self.dropout(g)
        next_c = f * c + i * g
        next_h = o * torch.tanh(next_c)
        next_h = self.dropout(next_h)
        return next_h, (next_h, next_c)

    def init_hidden(self, batch_size):
        return Variable(torch.zeros(batch_size, self.num_features, self.shape[0], self.shape[1])), Variable(torch.zeros(batch_size, self.num_features, self.shape[0], self.shape[1]))


class CLSTM_all_cell(nn.Module):
    """Initialize a basic Conv LSTM cell.
    Args:
      shape: int tuple thats the height and width of the hidden states h and c()
      filter_size: int that is the height and width of the filters
      num_features: int thats the num of channels of the states, like hidden_size

    """

    def __init__(self, shape, input_chans, filter_size, num_features):
        super(CLSTM_all_cell, self).__init__()
        self.shape = shape
        self.input_channels = input_chans
        self.kernel_size = filter_size
        self.hidden_channels = num_features
        self.dropout = nn.Dropout(p=0.5)
        self.padding = (filter_size - 1) / 2
        self.Wxi = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whi = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxf = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whf = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxc = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whc = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxo = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Who = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wci = None
        self.Wcf = None
        self.Wco = None

    def forward(self, input, hidden_state):
        hidden, c = hidden_state
        ci = torch.sigmoid(self.Wxi(input) + self.Whi(hidden) + c * self.Wci)
        cf = torch.sigmoid(self.Wxf(input) + self.Whf(hidden) + c * self.Wcf)
        co = torch.sigmoid(self.Wxo(input) + self.Who(hidden) + c * self.Wco)
        new_c = cf * c + ci * torch.tanh(self.Wxc(input) + self.Whc(hidden))
        new_h = co * torch.tanh(new_c)
        return new_h, (new_h, new_c)

    def init_hidden(self, batch_size):
        self.Wci = Variable(torch.zeros(1, self.hidden_channels, self.shape[0], self.shape[1]))
        self.Wcf = Variable(torch.zeros(1, self.hidden_channels, self.shape[0], self.shape[1]))
        self.Wco = Variable(torch.zeros(1, self.hidden_channels, self.shape[0], self.shape[1]))
        return Variable(torch.zeros(batch_size, self.hidden_channels, self.shape[0], self.shape[1])), Variable(torch.zeros(batch_size, self.hidden_channels, self.shape[0], self.shape[1]))


class MultiConvRNNCell(nn.Module):

    def __init__(self, cells, state_is_tuple=True):
        super(MultiConvRNNCell, self).__init__()
        self._cells = cells
        self._state_is_tuple = state_is_tuple

    def init_hidden(self, batch_size):
        init_states = []
        for i in range(len(self._cells)):
            init_states.append(self._cells[i].init_hidden(batch_size))
        return init_states

    def forward(self, input, hidden_state):
        cur_inp = input
        new_states = []
        for i, cell in enumerate(self._cells):
            cur_state = hidden_state[i]
            cur_inp, new_state = cell(cur_inp, cur_state)
            new_states.append(new_state)
        new_states = tuple(new_states)
        return cur_inp, new_states


class CLSTM(nn.Module):
    """Initialize a basic Conv LSTM cell.
    Args:
      shape: int tuple thats the height and width of the hidden states h and c()
      filter_size: int that is the height and width of the filters
      num_features: int thats the num of channels of the states, like hidden_size

    """

    def __init__(self, shape, input_chans, filter_size, num_features, num_layers):
        super(CLSTM, self).__init__()
        self.shape = shape
        self.input_chans = input_chans
        self.filter_size = filter_size
        self.num_features = num_features
        self.num_layers = num_layers
        cell_list = []
        cell_list.append(CLSTM_cell(self.shape, self.input_chans, self.filter_size, self.num_features))
        for idcell in range(1, self.num_layers):
            cell_list.append(CLSTM_cell(self.shape, self.num_features, self.filter_size, self.num_features))
        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input, hidden_state):
        """
        args:
            hidden_state:list of tuples, one for every layer, each tuple should be hidden_layer_i,c_layer_i
            input is the tensor of shape seq_len,Batch,Chans,H,W
        """
        current_input = input
        next_hidden = []
        seq_len = current_input.size(0)
        for idlayer in range(self.num_layers):
            hidden_c = hidden_state[idlayer]
            all_output = []
            output_inner = []
            for t in range(seq_len):
                hidden_c = self.cell_list[idlayer](current_input[t, ...], hidden_c)
                output_inner.append(hidden_c[0])
            next_hidden.append(hidden_c)
            None
            current_input = torch.cat(output_inner, 0).view(current_input.size(0), *output_inner[0].size())
            None
        return next_hidden, current_input

    def init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size))
        return init_states


def conv2_act(inplanes, out_channels=8, kernel_size=7, stride=5, padding=1, bias=True):
    layers = []
    layers += [nn.Conv2d(inplanes, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)]
    layers += [nn.BatchNorm2d(out_channels)]
    layers += [nn.LeakyReLU(negative_slope=0.2)]
    return nn.Sequential(*layers)


def downsmaple(inplanes, out_channels=8, kernel_size=7, stride=5, padding=1, bias=True):
    ret = conv2_act(inplanes, out_channels, kernel_size, stride, padding, bias)
    return ret


class Encoder(nn.Module):

    def __init__(self, inplanes, num_seqs):
        super(Encoder, self).__init__()
        self.num_seqs = num_seqs
        self.conv1_act = conv2_act(inplanes, out_channels=16, kernel_size=3, stride=1, padding=1, bias=True)
        num_filter = [64, 192, 192]
        kernel_size_l = [7, 7, 5]
        rnn_block_num = len(num_filter)
        stack_num = [2, 3, 3]
        encoder_rnn_block_states = []
        self.rnn1_1 = ConvGRUCell(input_size=16, hidden_size=num_filter[0], kernel_size=kernel_size_l[0])
        self.rnn1_1_h = None
        self.rnn1_2 = ConvGRUCell(input_size=num_filter[0], hidden_size=num_filter[0], kernel_size=kernel_size_l[0])
        self.rnn1_2_h = None
        self.downsample1 = downsmaple(inplanes=num_filter[0], out_channels=num_filter[1], kernel_size=4, stride=2, padding=1)
        self.rnn2_1 = ConvGRUCell(input_size=num_filter[1], hidden_size=num_filter[1], kernel_size=kernel_size_l[1])
        self.rnn2_1_h = None
        self.rnn2_2 = ConvGRUCell(input_size=num_filter[1], hidden_size=num_filter[1], kernel_size=kernel_size_l[1])
        self.rnn2_2_h = None
        self.rnn2_3 = ConvGRUCell(input_size=num_filter[1], hidden_size=num_filter[1], kernel_size=kernel_size_l[1])
        self.rnn2_3_h = None
        self.downsample2 = downsmaple(inplanes=num_filter[1], out_channels=num_filter[2], kernel_size=5, stride=3, padding=1)
        self.rnn3_1 = ConvGRUCell(input_size=num_filter[2], hidden_size=num_filter[2], kernel_size=kernel_size_l[2])
        self.rnn3_1_h = None
        self.rnn3_2 = ConvGRUCell(input_size=num_filter[2], hidden_size=num_filter[2], kernel_size=kernel_size_l[2])
        self.rnn3_2_h = None
        self.rnn3_3 = ConvGRUCell(input_size=num_filter[2], hidden_size=num_filter[2], kernel_size=kernel_size_l[2])
        self.rnn3_3_h = None

    def init_h0(self):
        self.rnn1_1_h = None
        self.rnn1_2_h = None
        self.rnn2_1_h = None
        self.rnn2_2_h = None
        self.rnn2_3_h = None
        self.rnn3_1_h = None
        self.rnn3_2_h = None
        self.rnn3_3_h = None

    def forward(self, data):
        data = self.conv1_act(data)
        self.rnn1_1_h = self.rnn1_1(data, self.rnn1_1_h)
        self.rnn1_2_h = self.rnn1_2(self.rnn1_1_h, self.rnn1_2_h)
        data = self.downsample1(self.rnn1_2_h)
        self.rnn2_1_h = self.rnn2_1(data, self.rnn2_1_h)
        self.rnn2_2_h = self.rnn2_2(self.rnn2_1_h, self.rnn2_2_h)
        self.rnn2_3_h = self.rnn2_3(self.rnn2_2_h, self.rnn2_3_h)
        data = self.downsample2(self.rnn2_3_h)
        self.rnn3_1_h = self.rnn3_1(data, self.rnn3_1_h)
        self.rnn3_2_h = self.rnn3_2(self.rnn3_1_h, self.rnn3_2_h)
        self.rnn3_3_h = self.rnn3_3(self.rnn3_2_h, self.rnn3_3_h)
        return self.rnn2_3_h


def deconv2_act(inplanes, out_channels=8, kernel_size=7, stride=5, padding=1, bias=True):
    layers = []
    layers += [nn.ConvTranspose2d(inplanes, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)]
    layers += [nn.BatchNorm2d(out_channels)]
    layers += [nn.LeakyReLU(negative_slope=0.2)]
    return nn.Sequential(*layers)


def upsmaple(inplanes, out_channels=8, kernel_size=7, stride=5, padding=1, bias=True):
    ret = deconv2_act(inplanes, out_channels, kernel_size, stride, padding, bias)
    return ret


class Forecaster(nn.Module):

    def __init__(self, num_seqs):
        super(Forecaster, self).__init__()
        num_filter = [64, 192, 192]
        kernel_size_l = [7, 7, 5]
        self.rnn1_1 = ConvGRUCell(input_size=num_filter[2], hidden_size=num_filter[2], kernel_size=kernel_size_l[2])
        self.rnn1_1_h = None
        self.rnn1_2 = ConvGRUCell(input_size=num_filter[2], hidden_size=num_filter[2], kernel_size=kernel_size_l[2])
        self.rnn1_2_h = None
        self.rnn1_3 = ConvGRUCell(input_size=num_filter[2], hidden_size=num_filter[2], kernel_size=kernel_size_l[2])
        self.rnn1_3_h = None
        self.upsample1 = upsmaple(inplanes=num_filter[2], out_channels=num_filter[2], kernel_size=5, stride=3, padding=1)
        self.rnn2_1 = ConvGRUCell(input_size=num_filter[2], hidden_size=num_filter[1], kernel_size=kernel_size_l[1])
        self.rnn2_1_h = None
        self.rnn2_2 = ConvGRUCell(input_size=num_filter[1], hidden_size=num_filter[1], kernel_size=kernel_size_l[1])
        self.rnn2_2_h = None
        self.rnn2_3 = ConvGRUCell(input_size=num_filter[1], hidden_size=num_filter[1], kernel_size=kernel_size_l[1])
        self.rnn2_3_h = None
        self.upsample2 = upsmaple(inplanes=num_filter[1], out_channels=num_filter[1], kernel_size=4, stride=2, padding=1)
        self.rnn3_1 = ConvGRUCell(input_size=num_filter[1], hidden_size=num_filter[0], kernel_size=kernel_size_l[0])
        self.rnn3_1_h = None
        self.rnn3_2 = ConvGRUCell(input_size=num_filter[0], hidden_size=num_filter[0], kernel_size=kernel_size_l[0])
        self.rnn3_2_h = None
        self.deconv1 = deconv2_act(inplanes=num_filter[0], out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv_final = conv2_act(inplanes=16, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.conv_pre = nn.Conv2d(in_channels=8, out_channels=1, kernel_size=1)

    def set_h0(self, encoder):
        self.rnn1_1_h = encoder.rnn3_3_h
        self.rnn1_2_h = encoder.rnn3_2_h
        self.rnn1_3_h = encoder.rnn3_1_h
        self.rnn2_1_h = encoder.rnn2_3_h
        self.rnn2_2_h = encoder.rnn2_2_h
        self.rnn2_3_h = encoder.rnn2_1_h
        self.rnn3_1_h = encoder.rnn1_2_h
        self.rnn3_2_h = encoder.rnn1_1_h

    def forward(self, data):
        self.rnn1_1_h = self.rnn1_1(data, self.rnn1_1_h)
        self.rnn1_2_h = self.rnn1_1(self.rnn1_1_h, self.rnn1_2_h)
        self.rnn1_3_h = self.rnn1_1(self.rnn1_2_h, self.rnn1_3_h)
        data = self.upsample1(self.rnn1_3_h)
        self.rnn2_1_h = self.rnn2_1(data, self.rnn2_1_h)
        self.rnn2_2_h = self.rnn2_2(self.rnn2_1_h, self.rnn2_2_h)
        self.rnn2_3_h = self.rnn2_3(self.rnn2_2_h, self.rnn2_3_h)
        data = self.upsample2(self.rnn2_3_h)
        self.rnn3_1_h = self.rnn3_1(data, self.rnn3_1_h)
        self.rnn3_2_h = self.rnn3_2(self.rnn3_1_h, self.rnn3_2_h)
        data = self.deconv1(self.rnn3_2_h)
        data = self.conv_final(data)
        pre_data = self.conv_pre(data)
        return pre_data


class HKOModel(nn.Module):

    def __init__(self, inplanes, input_num_seqs, output_num_seqs):
        super(HKOModel, self).__init__()
        self.input_num_seqs = input_num_seqs
        self.output_num_seqs = output_num_seqs
        self.encoder = Encoder(inplanes=inplanes, num_seqs=input_num_seqs)
        self.forecaster = Forecaster(num_seqs=output_num_seqs)

    def forward(self, data):
        self.encoder.init_h0()
        for time in range(self.input_num_seqs):
            self.encoder(data[time])
        all_pre_data = []
        self.forecaster.set_h0(self.encoder)
        for time in range(self.output_num_seqs):
            pre_data = self.forecaster(None)
            all_pre_data.append(pre_data)
        return all_pre_data


class RNNCovnGRU(nn.Module):

    def __init__(self, inplanes, input_num_seqs, output_num_seqs):
        super(RNNCovnGRU, self).__init__()
        self.inplanes = inplanes
        self.input_num_seqs = input_num_seqs
        self.output_num_seqs = output_num_seqs
        num_filter = 70
        kernel_size = 7
        self.rnn1_1 = ConvGRUCell(input_size=2, hidden_size=num_filter, kernel_size=kernel_size)
        self.rnn1_1_h = None
        self.rnn1_2 = ConvGRUCell(input_size=num_filter, hidden_size=num_filter, kernel_size=kernel_size)
        self.rnn1_2_h = None
        self.deconv1 = nn.ConvTranspose2d(num_filter, out_channels=1, kernel_size=3, stride=1, padding=1, bias=True)

    def init_h0(self):
        self.rnn1_1_h = None
        self.rnn1_2_h = None

    def forward(self, data):
        self.rnn1_1_h = self.rnn1_1(data, self.rnn1_1_h)
        self.rnn1_2_h = self.rnn1_2(self.rnn1_1_h, self.rnn1_2_h)
        data = self.deconv1(self.rnn1_2_h)
        return data


class RNNConvLSTM(nn.Module):

    def __init__(self, inplanes, input_num_seqs, output_num_seqs, shape):
        super(RNNConvLSTM, self).__init__()
        self.inplanes = inplanes
        self.input_num_seqs = input_num_seqs
        self.output_num_seqs = output_num_seqs
        self.shape = shape, shape
        num_filter = 84
        kernel_size = 7
        self.cell1 = CLSTM_cell(self.shape, self.inplanes, kernel_size, num_filter)
        self.cell2 = CLSTM_cell(self.shape, num_filter, kernel_size, num_filter)
        self.stacked_lstm = MultiConvRNNCell([self.cell1, self.cell2])
        self.deconv1 = nn.ConvTranspose2d(num_filter, out_channels=1, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, data):
        new_state = self.stacked_lstm.init_hidden(data.size()[1])
        x_unwrap = []
        for i in xrange(self.input_num_seqs + self.output_num_seqs):
            if i < self.input_num_seqs:
                y_1, new_state = self.stacked_lstm(data[i], new_state)
            else:
                y_1, new_state = self.stacked_lstm(x_1, new_state)
            x_1 = self.deconv1(y_1)
            if i >= self.input_num_seqs:
                x_unwrap.append(x_1)
        return x_unwrap


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (BMSELoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (RNNCovnGRU,
     lambda: ([], {'inplanes': 4, 'input_num_seqs': 4, 'output_num_seqs': 4}),
     lambda: ([torch.rand([4, 2, 4, 4])], {})),
]

