
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


import numpy as np


from torch.autograd import Variable


from sklearn import metrics


import math


import torch.backends.cudnn as cudnn


import scipy.sparse as sp


import scipy


from torch.nn import functional as F


from torch.nn import Parameter


from torch.nn.init import xavier_normal_


from torch.nn.init import xavier_uniform_


from torch.nn.utils.rnn import pack_padded_sequence


from torch.nn.utils.rnn import pad_packed_sequence


from torch.nn.parameter import Parameter


from torch.nn.modules.module import Module


import torch.nn as nn


import torch.nn.init as init


import random


from torch.nn.modules.rnn import LSTM


from itertools import chain


from torch.nn import LSTM


import torch.nn.functional as F


import scipy.stats


from collections import namedtuple


import time


import queue


from torch.cuda import Event


from scipy.sparse import csr_matrix


from scipy.sparse import spmatrix


class Backends:
    TORCH = 'pytorch'
    TENSORFLOW = 'tensorflow'
    TEST = 'test'
    CNTK = 'cntk'


alias2params = {}


def get_home_path():
    return os.environ['HOME']


def get_logger_path():
    return join(get_home_path(), '.data', 'log_files')


class GlobalLogger:
    timestr = None
    global_logger_path = None
    f_global_logger = None

    @staticmethod
    def init():
        GlobalLogger.timestr = time.strftime('%Y%m%d-%H%M%S')
        if not os.path.exists(join(get_logger_path(), 'full_logs')):
            os.mkdir(join(get_logger_path(), 'full_logs'))
        GlobalLogger.global_logger_path = join(get_logger_path(), 'full_logs', GlobalLogger.timestr + '.txt')
        GlobalLogger.f_global_logger = open(GlobalLogger.global_logger_path, 'w')

    @staticmethod
    def flush():
        GlobalLogger.f_global_logger.close()
        GlobalLogger.f_global_logger = open(GlobalLogger.global_logger_path, 'a')

    def __del__(self):
        GlobalLogger.f_global_logger.close()


def make_dirs_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)


params2field = {}


params2type = {}


class Config:
    dropout_rate = 0.25
    channels = 200
    kernel_size = 5
    init_emb_size = 100
    gc1_emb_size = 150
    embedding_dim = 200
    learning_rate = 0.002
    batch_size = 128
    dropout = 0.2
    backend = Backends.TORCH
    L2 = 0.0
    cuda = False
    init_embedding_dim = 100
    gc1_emb_size = 200
    gc2_emb_size = 100
    hidden_size = 256
    input_dropout = 0.0
    feature_map_dropout = 0.2
    convs = '80.1_20.3'
    sim_use_relu = True
    use_conv_transpose = False
    use_bias = True
    optimizer = 'adam'
    learning_rate_decay = 1.0
    label_smoothing_epsilon = 0.1
    epochs = 1000
    dataset = None
    process = False
    model_name = None
    save_model_dir = None
    load_model = False

    @staticmethod
    def parse_argv(argv):
        file_name = argv[0]
        args = argv[1:]
        assert len(args) % 2 == 0, 'Global parser expects an even number of arguments.'
        values = []
        names = []
        for i, token in enumerate(args):
            if i % 2 == 0:
                names.append(token)
            else:
                values.append(token)
        for i in range(len(names)):
            if names[i] in alias2params:
                log.debug('Replaced parameters alias {0} with name {1}', names[i], alias2params[names[i]])
                names[i] = alias2params[names[i]]
        for i in range(len(names)):
            name = names[i]
            if name[:2] == '--':
                continueparams2field
            if name not in params2type:
                log.info('List of possible parameters: {0}', params2type.keys())
                log.error('Parameter {0} does not exist. Prefix your custom parameters with -- to skip parsing for global config', name)
            values[i] = params2type[name](values[i])
        for name, value in zip(names, values):
            if name[:2] == '--':
                continue
            params2field[name](value)
            log.info('Set parameter {0} to {1}', name, value)
    use_transposed_convolutions = False


class Complex(torch.nn.Module):

    def __init__(self, num_entities, num_relations):
        super(Complex, self).__init__()
        self.num_entities = num_entities
        self.emb_e_real = torch.nn.Embedding(num_entities, Config.embedding_dim, padding_idx=0)
        self.emb_e_img = torch.nn.Embedding(num_entities, Config.embedding_dim, padding_idx=0)
        self.emb_rel_real = torch.nn.Embedding(num_relations, Config.embedding_dim, padding_idx=0)
        self.emb_rel_img = torch.nn.Embedding(num_relations, Config.embedding_dim, padding_idx=0)
        self.inp_drop = torch.nn.Dropout(Config.input_dropout)
        self.loss = torch.nn.BCELoss()

    def init(self):
        xavier_normal_(self.emb_e_real.weight.data)
        xavier_normal_(self.emb_e_img.weight.data)
        xavier_normal_(self.emb_rel_real.weight.data)
        xavier_normal_(self.emb_rel_img.weight.data)

    def forward(self, e1, rel, X, A):
        e1_embedded_real = self.emb_e_real(e1).squeeze()
        rel_embedded_real = self.emb_rel_real(rel).squeeze()
        e1_embedded_img = self.emb_e_img(e1).squeeze()
        rel_embedded_img = self.emb_rel_img(rel).squeeze()
        e1_embedded_real = self.inp_drop(e1_embedded_real)
        rel_embedded_real = self.inp_drop(rel_embedded_real)
        e1_embedded_img = self.inp_drop(e1_embedded_img)
        rel_embedded_img = self.inp_drop(rel_embedded_img)
        realrealreal = torch.mm(e1_embedded_real * rel_embedded_real, self.emb_e_real.weight.transpose(1, 0))
        realimgimg = torch.mm(e1_embedded_real * rel_embedded_img, self.emb_e_img.weight.transpose(1, 0))
        imgrealimg = torch.mm(e1_embedded_img * rel_embedded_real, self.emb_e_img.weight.transpose(1, 0))
        imgimgreal = torch.mm(e1_embedded_img * rel_embedded_img, self.emb_e_real.weight.transpose(1, 0))
        pred = realrealreal + realimgimg + imgrealimg - imgimgreal
        pred = F.sigmoid(pred)
        return pred


class DistMult(torch.nn.Module):

    def __init__(self, num_entities, num_relations):
        super(DistMult, self).__init__()
        self.emb_e = torch.nn.Embedding(num_entities, Config.embedding_dim, padding_idx=0)
        self.emb_rel = torch.nn.Embedding(num_relations, Config.embedding_dim, padding_idx=0)
        self.inp_drop = torch.nn.Dropout(Config.input_dropout)
        self.loss = torch.nn.BCELoss()

    def init(self):
        xavier_normal_(self.emb_e.weight.data)
        xavier_normal_(self.emb_rel.weight.data)

    def forward(self, e1, rel, X, A):
        e1_embedded = self.emb_e(e1)
        rel_embedded = self.emb_rel(rel)
        e1_embedded = e1_embedded.squeeze()
        rel_embedded = rel_embedded.squeeze()
        e1_embedded = self.inp_drop(e1_embedded)
        rel_embedded = self.inp_drop(rel_embedded)
        pred = torch.mm(e1_embedded * rel_embedded, self.emb_e.weight.transpose(1, 0))
        pred = F.sigmoid(pred)
        return pred


class ConvE(torch.nn.Module):

    def __init__(self, num_entities, num_relations):
        super(ConvE, self).__init__()
        self.emb_e = torch.nn.Embedding(num_entities, Config.embedding_dim, padding_idx=0)
        self.emb_rel = torch.nn.Embedding(num_relations, Config.embedding_dim, padding_idx=0)
        self.inp_drop = torch.nn.Dropout(Config.input_dropout)
        self.hidden_drop = torch.nn.Dropout(Config.dropout)
        self.feature_map_drop = torch.nn.Dropout2d(Config.feature_map_dropout)
        self.loss = torch.nn.BCELoss()
        self.conv1 = torch.nn.Conv2d(1, 32, (3, 3), 1, 0, bias=Config.use_bias)
        self.bn0 = torch.nn.BatchNorm2d(1)
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.bn2 = torch.nn.BatchNorm1d(Config.embedding_dim)
        self.register_parameter('b', Parameter(torch.zeros(num_entities)))
        self.fc = torch.nn.Linear(10368, Config.embedding_dim)
        None

    def init(self):
        xavier_normal_(self.emb_e.weight.data)
        xavier_normal_(self.emb_rel.weight.data)

    def forward(self, e1, rel, X, A):
        e1_embedded = self.emb_e(e1).view(-1, 1, 10, 20)
        rel_embedded = self.emb_rel(rel).view(-1, 1, 10, 20)
        stacked_inputs = torch.cat([e1_embedded, rel_embedded], 2)
        stacked_inputs = self.bn0(stacked_inputs)
        x = self.inp_drop(stacked_inputs)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_map_drop(x)
        x = x.view(Config.batch_size, -1)
        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = torch.mm(x, self.emb_e.weight.transpose(1, 0))
        x += self.b.expand_as(x)
        pred = F.sigmoid(x)
        return pred


class GraphConvolution(torch.nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, num_relations, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.num_relations = num_relations
        self.alpha = torch.nn.Embedding(num_relations + 1, 1, padding_idx=0)
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        alp = self.alpha(adj[1]).t()[0]
        A = torch.sparse_coo_tensor(adj[0], alp, torch.Size([adj[2], adj[2]]), requires_grad=True)
        A = A + A.transpose(0, 1)
        support = torch.mm(input, self.weight)
        output = torch.sparse.mm(A, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class ConvTransE(torch.nn.Module):

    def __init__(self, num_entities, num_relations):
        super(ConvTransE, self).__init__()
        self.emb_e = torch.nn.Embedding(num_entities, Config.init_emb_size, padding_idx=0)
        self.emb_rel = torch.nn.Embedding(num_relations, Config.init_emb_size, padding_idx=0)
        self.inp_drop = torch.nn.Dropout(Config.input_dropout)
        self.hidden_drop = torch.nn.Dropout(Config.dropout_rate)
        self.feature_map_drop = torch.nn.Dropout(Config.dropout_rate)
        self.loss = torch.nn.BCELoss()
        self.conv1 = nn.Conv1d(2, Config.channels, Config.kernel_size, stride=1, padding=int(math.floor(Config.kernel_size / 2)))
        self.bn0 = torch.nn.BatchNorm1d(2)
        self.bn1 = torch.nn.BatchNorm1d(Config.channels)
        self.bn2 = torch.nn.BatchNorm1d(Config.init_emb_size)
        self.register_parameter('b', Parameter(torch.zeros(num_entities)))
        self.fc = torch.nn.Linear(Config.init_emb_size * Config.channels, Config.init_emb_size)
        self.bn_init = torch.nn.BatchNorm1d(Config.init_emb_size)
        None

    def init(self):
        xavier_normal_(self.emb_e.weight.data)
        xavier_normal_(self.emb_rel.weight.data)

    def forward(self, e1, rel, X, A):
        emb_initial = self.emb_e(X)
        e1_embedded_all = self.bn_init(emb_initial)
        e1_embedded = e1_embedded_all[e1]
        rel_embedded = self.emb_rel(rel)
        stacked_inputs = torch.cat([e1_embedded, rel_embedded], 1)
        stacked_inputs = self.bn0(stacked_inputs)
        x = self.inp_drop(stacked_inputs)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_map_drop(x)
        x = x.view(Config.batch_size, -1)
        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = torch.mm(x, e1_embedded_all.transpose(1, 0))
        pred = F.sigmoid(x)
        return pred


class SACN(torch.nn.Module):

    def __init__(self, num_entities, num_relations):
        super(SACN, self).__init__()
        self.emb_e = torch.nn.Embedding(num_entities, Config.init_emb_size, padding_idx=0)
        self.gc1 = GraphConvolution(Config.init_emb_size, Config.gc1_emb_size, num_relations)
        self.gc2 = GraphConvolution(Config.gc1_emb_size, Config.embedding_dim, num_relations)
        self.emb_rel = torch.nn.Embedding(num_relations, Config.embedding_dim, padding_idx=0)
        self.inp_drop = torch.nn.Dropout(Config.input_dropout)
        self.hidden_drop = torch.nn.Dropout(Config.dropout_rate)
        self.feature_map_drop = torch.nn.Dropout(Config.dropout_rate)
        self.loss = torch.nn.BCELoss()
        self.conv1 = nn.Conv1d(2, Config.channels, Config.kernel_size, stride=1, padding=int(math.floor(Config.kernel_size / 2)))
        self.bn0 = torch.nn.BatchNorm1d(2)
        self.bn1 = torch.nn.BatchNorm1d(Config.channels)
        self.bn2 = torch.nn.BatchNorm1d(Config.embedding_dim)
        self.register_parameter('b', Parameter(torch.zeros(num_entities)))
        self.fc = torch.nn.Linear(Config.embedding_dim * Config.channels, Config.embedding_dim)
        self.bn3 = torch.nn.BatchNorm1d(Config.gc1_emb_size)
        self.bn4 = torch.nn.BatchNorm1d(Config.embedding_dim)
        self.bn_init = torch.nn.BatchNorm1d(Config.init_emb_size)
        None

    def init(self):
        xavier_normal_(self.emb_e.weight.data)
        xavier_normal_(self.emb_rel.weight.data)
        xavier_normal_(self.gc1.weight.data)
        xavier_normal_(self.gc2.weight.data)

    def forward(self, e1, rel, X, A):
        emb_initial = self.emb_e(X)
        x = self.gc1(emb_initial, A)
        x = self.bn3(x)
        x = F.tanh(x)
        x = F.dropout(x, Config.dropout_rate, training=self.training)
        x = self.bn4(self.gc2(x, A))
        e1_embedded_all = F.tanh(x)
        e1_embedded_all = F.dropout(e1_embedded_all, Config.dropout_rate, training=self.training)
        e1_embedded = e1_embedded_all[e1]
        rel_embedded = self.emb_rel(rel)
        stacked_inputs = torch.cat([e1_embedded, rel_embedded], 1)
        stacked_inputs = self.bn0(stacked_inputs)
        x = self.inp_drop(stacked_inputs)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_map_drop(x)
        x = x.view(Config.batch_size, -1)
        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = torch.mm(x, e1_embedded_all.transpose(1, 0))
        pred = F.sigmoid(x)
        return pred


class Net(torch.nn.Module):

    def __init__(self, num_embeddings, num_labels):
        super(Net, self).__init__()
        self.emb = torch.nn.Embedding(num_embeddings, Config.embedding_dim, padding_idx=0)
        self.lstm1 = LSTM(Config.embedding_dim, Config.hidden_size, num_layers=1, batch_first=True, bias=True, dropout=Config.dropout, bidirectional=True)
        self.lstm2 = LSTM(Config.embedding_dim, Config.hidden_size, num_layers=1, batch_first=True, bias=True, dropout=Config.dropout, bidirectional=True)
        self.linear = torch.nn.Linear(Config.hidden_size * 4, num_labels)
        self.loss = torch.nn.CrossEntropyLoss()
        self.pred = torch.nn.Softmax()
        self.h0 = Variable(torch.zeros(2, Config.batch_size, Config.hidden_size))
        self.c0 = Variable(torch.zeros(2, Config.batch_size, Config.hidden_size))
        self.h1 = Variable(torch.zeros(2, Config.batch_size, Config.hidden_size))
        self.c1 = Variable(torch.zeros(2, Config.batch_size, Config.hidden_size))
        if Config.cuda:
            self.h0 = self.h0
            self.c0 = self.c0
            self.h1 = self.h1
            self.c1 = self.c1

    def forward(self, str2var):
        inp = str2var['input']
        sup = str2var['support']
        l1 = str2var['input_length']
        l2 = str2var['support_length']
        t = str2var['target']
        self.h0.data.zero_()
        self.c0.data.zero_()
        inp_seq = self.emb(inp)
        sup_seq = self.emb(sup)
        out1, hid1 = self.lstm1(inp_seq, (self.h0, self.c0))
        out2, hid2 = self.lstm2(sup_seq, hid1)
        outs1 = []
        outs2 = []
        for i in range(Config.batch_size):
            outs1.append(out1[i, l1.data[i] - 1, :])
            outs2.append(out2[i, l2.data[i] - 1, :])
        out1_stacked = torch.stack(outs1, 0)
        out2_stacked = torch.stack(outs2, 0)
        out = torch.cat([out1_stacked, out2_stacked], 1)
        projected = self.linear(out)
        loss = self.loss(projected, t)
        max_values, argmax = torch.topk(self.pred(projected), 1)
        return loss, argmax


class AbstractModel(object):

    def __init__(self):
        super(AbstractModel, self).__init__()
        self.input_str_args = None
        self.output_str_args = None
        self.used_keys = None

    def forward(self, str2var, *args):
        raise NotImplementedError('Classes that inherit from AbstractModel need to implement the forward method.')

    @property
    def modules(self):
        raise NotImplementedError('Classes that inherit from AbstractModel need to overrite the modules property.')

    def expected_str2var_keys(self, str2var, keys):
        self.used_keys = keys
        for key in keys:
            if key not in str2var:
                log.error('Variable with name {0} expected, but not found in str2variable dict with keys {1}'.format(key, str2var.keys()))

    def expected_str2var_keys_oneof(self, str2var, keys):
        self.used_keys = keys
        one_exists = False
        for key in keys:
            if key in str2var:
                one_exists = True
        if not one_exists:
            log.error('At least one of these variable was expected: {0}. But str2var only has these variables: {1}.', keys, str2var.keys())

    def expected_args(self, str_arg_names, str_arg_description):
        log.debug_once('Expected args {0}'.format(str_arg_names))
        log.debug_once('Info for the expected arguments: {0}'.format(str_arg_description))
        self.input_str_args = str_arg_names

    def generated_outputs(self, str_output_names, str_output_description):
        log.debug_once('Generated outputs: {0}'.format(str_output_names))
        log.debug_once('Info for the provided outputs: {0}'.format(str_output_description))
        self.output_str_args = str_output_names
        self.used_keys
        self.input_str_args
        self.output_str_args
        message = '{0} + {1} -> {2}'.format(self.used_keys, self.input_str_args, self.output_str_args)
        log.info_once(message)


class TorchEmbedding(torch.nn.Module, AbstractModel):

    def __init__(self, embedding_size, num_embeddings):
        super(TorchEmbedding, self).__init__()
        self.emb = torch.nn.Embedding(num_embeddings, embedding_size, padding_idx=0)

    def forward(self, str2var, *args):
        self.expected_str2var_keys_oneof(str2var, ['input', 'support'])
        self.expected_args('None', 'None')
        self.generated_outputs('input idx, support idx', 'both sequences have shape = [batch, timesteps, embedding dim]')
        embedded_results = []
        if 'input' in str2var:
            embedded_results.append(self.emb(str2var['input']))
        if 'support' in str2var:
            embedded_results.append(self.emb(str2var['support']))
        return embedded_results


class TorchBiDirectionalLSTM(torch.nn.Module, AbstractModel):

    def __init__(self, input_size, hidden_size, dropout=0.0, layers=1, bidirectional=True, to_cuda=False, conditional_encoding=True):
        super(TorchBiDirectionalLSTM, self).__init__()
        use_bias = True
        num_directions = 1 if not bidirectional else 2
        self.lstm = LSTM(input_size, hidden_size, layers, use_bias, True, 0.2, bidirectional)
        self.h0 = None
        self.c0 = None
        self.h0 = Variable(torch.FloatTensor(num_directions * layers, Config.batch_size, hidden_size))
        self.c0 = Variable(torch.FloatTensor(num_directions * layers, Config.batch_size, hidden_size))
        if Config.cuda:
            self.h0 = self.h0
            self.c0 = self.c0

    def forward(self, str2var, *args):
        self.expected_str2var_keys(str2var, [])
        self.expected_args('embedded seq', 'size [batch, time steps, embedding dim]')
        self.generated_outputs('LSTM output seq', 'size [batch, time steps, 2x hidden dim]')
        seq = args
        self.h0.data.zero_()
        self.c0.data.zero_()
        out, hid = self.lstm(seq, (self.h0, self.c0))
        return [out, hid]


class TorchPairedBiDirectionalLSTM(torch.nn.Module, AbstractModel):

    def __init__(self, input_size, hidden_size, dropout=0.0, layers=1, bidirectional=True, to_cuda=False, conditional_encoding=True):
        super(TorchPairedBiDirectionalLSTM, self).__init__()
        self.conditional_encoding = conditional_encoding
        use_bias = True
        num_directions = 1 if not bidirectional else 2
        self.conditional_encoding = conditional_encoding
        self.lstm1 = LSTM(input_size, hidden_size, layers, use_bias, True, Config.dropout, bidirectional)
        self.lstm2 = LSTM(input_size, hidden_size, layers, use_bias, True, Config.dropout, bidirectional)
        self.h01 = None
        self.c01 = None
        self.h02 = None
        self.c02 = None
        self.h01 = Variable(torch.FloatTensor(num_directions * layers, Config.batch_size, hidden_size))
        self.c01 = Variable(torch.FloatTensor(num_directions * layers, Config.batch_size, hidden_size))
        if Config.cuda:
            self.h01 = self.h01
            self.c01 = self.c01
        if not self.conditional_encoding:
            self.h02 = Variable(torch.FloatTensor(num_directions * layers, Config.batch_size, hidden_size))
            self.c02 = Variable(torch.FloatTensor(num_directions * layers, Config.batch_size, hidden_size))
            if Config.cuda:
                self.h02 = self.h02
                self.c02 = self.c02

    def forward(self, str2var, *args):
        self.expected_str2var_keys(str2var, [])
        self.expected_args('embedded input seq, embedded seq support', 'both of size [batch, time steps, embedding dim]')
        self.generated_outputs('LSTM output seq inputs, LSTM output seq support', 'both of size [batch, time steps, 2x hidden dim]')
        seq1, seq2 = args
        if self.conditional_encoding:
            self.h01.data.zero_()
            self.c01.data.zero_()
            out1, hid1 = self.lstm1(seq1, (self.h01, self.c01))
            out2, hid2 = self.lstm2(seq2, hid1)
        else:
            self.h01.data.zero_()
            self.c01.data.zero_()
            self.h02.data.zero_()
            self.c02.data.zero_()
            out1, hid1 = self.lstm1(seq1, (self.h01, self.c01))
            out2, hid2 = self.lstm2(seq2, (self.h02, self.c02))
        return [out1, out2]


class TorchVariableLengthOutputSelection(torch.nn.Module, AbstractModel):

    def __init__(self):
        super(TorchVariableLengthOutputSelection, self).__init__()
        self.b1 = None
        self.b2 = None

    def forward(self, str2var, *args):
        self.expected_str2var_keys(str2var, ['input_length', 'support_length'])
        self.expected_args('LSTM output sequence input , LSTM output sequence support', 'dimension of both: [batch, time steps, 2x LSTM hidden size]')
        self.generated_outputs('stacked bidirectional outputs of last timestep', 'dim is [batch_size, 4x hidden size]')
        output_lstm1, output_lstm2 = args
        l1, l2 = str2var['input_length'], str2var['support_length']
        if self.b1 == None:
            b1 = torch.ByteTensor(output_lstm1.size())
            b2 = torch.ByteTensor(output_lstm2.size())
            if Config.cuda:
                b1 = b1
                b2 = b2
        b1.fill_(0)
        for i, num in enumerate(l1.data):
            b1[i, num - 1, :] = 1
        out1 = output_lstm1[b1].view(Config.batch_size, -1)
        b2.fill_(0)
        for i, num in enumerate(l2.data):
            b2[i, num - 1, :] = 1
        out2 = output_lstm2[b2].view(Config.batch_size, -1)
        out = torch.cat([out1, out2], 1)
        return [out]


class TorchSoftmaxCrossEntropy(torch.nn.Module, AbstractModel):

    def __init__(self, input_dim, num_labels):
        super(TorchSoftmaxCrossEntropy, self).__init__()
        self.num_labels = num_labels
        self.projection_to_labels = torch.nn.Linear(input_dim, num_labels)

    def forward(self, str2var, *args):
        self.expected_str2var_keys(str2var, ['target'])
        self.expected_args('some inputs', 'dimension: [batch, any]')
        self.generated_outputs('logits, loss, argmax', 'dimensions: logits = [batch, labels], loss = 1x1, argmax = [batch, 1]')
        outputs_prev_layer = args[0]
        t = str2var['target']
        logits = self.projection_to_labels(outputs_prev_layer)
        out = F.log_softmax(logits)
        loss = F.nll_loss(out, t)
        maximum, argmax = torch.topk(out.data, 1)
        return [logits, loss, argmax]

