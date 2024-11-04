import sys
_module = sys.modules[__name__]
del sys
Constants = _module
Modules = _module
Model = _module
train = _module
utils = _module

from paritybench._paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, matplotlib, numbers, numpy, pandas, queue, random, re, scipy, sklearn, string, tensorflow, time, torch, torchaudio, torchvision, types, typing, uuid, warnings
import numpy as np
from torch import Tensor
patch_functional()
open = mock_open()
yaml = logging = sys = argparse = MagicMock()
ArgumentParser = argparse.ArgumentParser
_global_config = args = argv = cfg = config = params = _mock_config()
argparse.ArgumentParser.return_value.parse_args.return_value = _global_config
yaml.load.return_value = _global_config
sys.argv = _global_config
__version__ = '1.0.0'
xrange = range
wraps = functools.wraps


import torch


import torch.nn as nn


from torch.autograd import Variable


import torch.nn.functional as F


import numpy as np


from torch.optim import Adam


from torch import cuda


def check_cuda(torch_var, use_cuda=False):
    if use_cuda and cuda.is_available():
        return torch_var
    else:
        return torch_var


class Encoder(nn.Module):
    """A LSTM encoder to encode a sentence into a latent vector z."""

    def __init__(self, n_src_vocab, n_layers=1, d_word_vec=150, d_inner_hid=300, dropout=0.1, d_out_hid=300, use_cuda=False):
        super(Encoder, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.src_word_emb = nn.Embedding(n_src_vocab, d_word_vec, padding_idx=Constants.PAD)
        self.n_layers = n_layers
        self.d_inner_hid = d_inner_hid
        self.use_cuda = use_cuda
        self.rnn = nn.LSTM(d_word_vec, d_inner_hid, n_layers, dropout=dropout)
        self._enc_mu = nn.Linear(d_inner_hid, d_out_hid)
        self._enc_log_sigma = nn.Linear(d_inner_hid, d_out_hid)
        self.init_weights()

    def _sample_latent(self, enc_hidden):
        mu = self._enc_mu(enc_hidden)
        log_sigma = self._enc_log_sigma(enc_hidden)
        sigma = torch.exp(log_sigma)
        std_z = torch.from_numpy(np.random.normal(0, 1, size=sigma.size())).float()
        self.z_mean = mu
        self.z_sigma = sigma
        std_z_var = Variable(std_z, requires_grad=False)
        std_z_var = check_cuda(std_z_var, self.use_cuda)
        return mu + sigma * std_z_var

    def forward(self, src_seq, hidden, dont_pass_emb=False):
        if dont_pass_emb:
            enc_input = self.drop(src_seq)
        else:
            enc_input = self.drop(self.src_word_emb(src_seq))
        enc_input = enc_input.permute(1, 0, 2)
        _, hidden = self.rnn(enc_input, hidden)
        hidden = self._sample_latent(hidden[0]), hidden[1]
        return hidden

    def init_hidden(self, batch_size):
        hidden = [Variable(torch.zeros(self.n_layers, batch_size, self.d_inner_hid)), Variable(torch.zeros(self.n_layers, batch_size, self.d_inner_hid))]
        hidden[0] = check_cuda(hidden[0], self.use_cuda)
        hidden[1] = check_cuda(hidden[1], self.use_cuda)
        return hidden

    def init_weights(self):
        initrange = 0.1
        self.src_word_emb.weight.data.uniform_(-initrange, initrange)
        self._enc_mu.weight.data.uniform_(-initrange, initrange)
        self._enc_log_sigma.weight.data.uniform_(-initrange, initrange)


class Generator(nn.Module):
    """A LSTM generator to synthesis a sentence with input (z, c)
       where z is a latent vector from encoder and c is attribute code.
    """

    def __init__(self, n_target_vocab, n_layers=1, d_word_vec=150, d_inner_hid=300, c_dim=1, dropout=0.1, use_cuda=False):
        super(Generator, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.d_inner_hid = d_inner_hid
        self.c_dim = c_dim
        self.n_layers = n_layers
        self.use_cuda = use_cuda
        self.target_word_emb = nn.Embedding(n_target_vocab, d_word_vec, padding_idx=Constants.PAD)
        self.rnn = nn.LSTM(d_word_vec, d_inner_hid + c_dim, n_layers, dropout=dropout)
        self.to_word_emb = nn.Sequential(nn.Linear(d_inner_hid + c_dim, d_word_vec), nn.ReLU())
        self.linear = nn.Linear(d_word_vec, n_target_vocab)
        self.one_hot_to_word_emb = nn.Linear(n_target_vocab, d_word_vec)
        self.linear.weight = self.target_word_emb.weight
        self.one_hot_to_word_emb.weight = torch.nn.Parameter(self.linear.weight.permute(1, 0).data)
        self.softmax = nn.Softmax()
        self.init_weights()

    def forward(self, target_word, hidden, low_temp=False, one_hot_input=False):
        """ hidden is composed of z and c """
        """ input is word-by-word in Generator """
        if one_hot_input:
            dec_input = self.drop(self.one_hot_to_word_emb(target_word)).unsqueeze(0)
        else:
            dec_input = self.drop(self.target_word_emb(target_word)).unsqueeze(0)
        output, hidden = self.rnn(dec_input, hidden)
        output = self.to_word_emb(output)
        output = self.linear(output)
        if low_temp:
            pre_soft = output[0]
            lowed_output = pre_soft / 0.001
            output = self.softmax(lowed_output)
            return output, hidden, pre_soft
        return output, hidden

    def init_hidden_c_for_lstm(self, batch_size):
        hidden = Variable(torch.zeros(self.n_layers, batch_size, self.d_inner_hid))
        hidden = check_cuda(hidden, self.use_cuda)
        return hidden

    def init_weights(self):
        initrange = 0.1
        self.target_word_emb.weight.data.uniform_(-initrange, initrange)
        self.to_word_emb[0].weight.data.uniform_(-initrange, initrange)
        self.linear.weight.data.uniform_(-initrange, initrange)


class Discriminator(nn.Module):
    """A CNN discriminator to classify the attributes given a sentence."""

    def __init__(self, n_src_vocab, maxlen, d_word_vec=150, dropout=0.1, use_cuda=False):
        super(Discriminator, self).__init__()
        self.use_cuda = use_cuda
        self.src_word_emb = nn.Embedding(n_src_vocab, d_word_vec, padding_idx=Constants.PAD)
        self.drop = nn.Dropout(dropout)
        self.conv1 = nn.Conv1d(maxlen, 128, kernel_size=5)
        self.conv2 = nn.Conv1d(128, 128, kernel_size=5)
        self.conv3 = nn.Conv1d(128, 128, kernel_size=5)
        self.softmax = nn.LogSoftmax()

    def forward(self, input_sentence, is_softmax=False, dont_pass_emb=False):
        if dont_pass_emb:
            emb_sentence = input_sentence
        else:
            emb_sentence = self.src_word_emb(input_sentence)
        relu1 = F.relu(self.conv1(emb_sentence))
        layer1 = F.max_pool1d(relu1, 3)
        relu2 = F.relu(self.conv2(layer1))
        layer2 = F.max_pool1d(relu2, 3)
        layer3 = F.max_pool1d(F.relu(self.conv2(layer2)), 10)
        flatten = self.drop(layer2.view(layer3.size()[0], -1))
        if not hasattr(self, 'linear'):
            self.linear = nn.Linear(flatten.size()[1], 2)
            self.linear = check_cuda(self.linear, self.use_cuda)
        logit = self.linear(flatten)
        if is_softmax:
            logit = self.softmax(logit)
        return logit

