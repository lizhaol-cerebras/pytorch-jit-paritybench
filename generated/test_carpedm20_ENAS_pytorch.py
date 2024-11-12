
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


import torch as t


import torchvision.datasets as datasets


import torchvision.transforms as transforms


import collections


import torch


import torch.nn.functional as F


import numpy as np


from collections import defaultdict


from collections import deque


from torch import nn


from torch.autograd import Variable


import math


import scipy.signal


import torch.nn.parallel


import logging


Node = collections.namedtuple('Node', ['id', 'name'])


def _construct_dags(prev_nodes, activations, func_names, num_blocks):
    """Constructs a set of DAGs based on the actions, i.e., previous nodes and
    activation functions, sampled from the controller/policy pi.

    Args:
        prev_nodes: Previous node actions from the policy.
        activations: Activations sampled from the policy.
        func_names: Mapping from activation function names to functions.
        num_blocks: Number of blocks in the target RNN cell.

    Returns:
        A list of DAGs defined by the inputs.

    RNN cell DAGs are represented in the following way:

    1. Each element (node) in a DAG is a list of `Node`s.

    2. The `Node`s in the list dag[i] correspond to the subsequent nodes
       that take the output from node i as their own input.

    3. dag[-1] is the node that takes input from x^{(t)} and h^{(t - 1)}.
       dag[-1] always feeds dag[0].
       dag[-1] acts as if `w_xc`, `w_hc`, `w_xh` and `w_hh` are its
       weights.

    4. dag[N - 1] is the node that produces the hidden state passed to
       the next timestep. dag[N - 1] is also always a leaf node, and therefore
       is always averaged with the other leaf nodes and fed to the output
       decoder.
    """
    dags = []
    for nodes, func_ids in zip(prev_nodes, activations):
        dag = collections.defaultdict(list)
        dag[-1] = [Node(0, func_names[func_ids[0]])]
        dag[-2] = [Node(0, func_names[func_ids[0]])]
        for jdx, (idx, func_id) in enumerate(zip(nodes, func_ids[1:])):
            dag[utils.to_item(idx)].append(Node(jdx + 1, func_names[func_id]))
        leaf_nodes = set(range(num_blocks)) - dag.keys()
        for idx in leaf_nodes:
            dag[idx] = [Node(num_blocks, 'avg')]
        last_node = Node(num_blocks + 1, 'h[t]')
        dag[num_blocks] = [last_node]
        dags.append(dag)
    return dags


class Controller(torch.nn.Module):
    """Based on
    https://github.com/pytorch/examples/blob/master/word_language_model/model.py

    TODO(brendan): RL controllers do not necessarily have much to do with
    language models.

    Base the controller RNN on the GRU from:
    https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/model.py
    """

    def __init__(self, args):
        torch.nn.Module.__init__(self)
        self.args = args
        if self.args.network_type == 'rnn':
            self.num_tokens = [len(args.shared_rnn_activations)]
            for idx in range(self.args.num_blocks):
                self.num_tokens += [idx + 1, len(args.shared_rnn_activations)]
            self.func_names = args.shared_rnn_activations
        elif self.args.network_type == 'cnn':
            self.num_tokens = [len(args.shared_cnn_types), self.args.num_blocks]
            self.func_names = args.shared_cnn_types
        num_total_tokens = sum(self.num_tokens)
        self.encoder = torch.nn.Embedding(num_total_tokens, args.controller_hid)
        self.lstm = torch.nn.LSTMCell(args.controller_hid, args.controller_hid)
        self.decoders = []
        for idx, size in enumerate(self.num_tokens):
            decoder = torch.nn.Linear(args.controller_hid, size)
            self.decoders.append(decoder)
        self._decoders = torch.nn.ModuleList(self.decoders)
        self.reset_parameters()
        self.static_init_hidden = utils.keydefaultdict(self.init_hidden)

        def _get_default_hidden(key):
            return utils.get_variable(torch.zeros(key, self.args.controller_hid), self.args.cuda, requires_grad=False)
        self.static_inputs = utils.keydefaultdict(_get_default_hidden)

    def reset_parameters(self):
        init_range = 0.1
        for param in self.parameters():
            param.data.uniform_(-init_range, init_range)
        for decoder in self.decoders:
            decoder.bias.data.fill_(0)

    def forward(self, inputs, hidden, block_idx, is_embed):
        if not is_embed:
            embed = self.encoder(inputs)
        else:
            embed = inputs
        hx, cx = self.lstm(embed, hidden)
        logits = self.decoders[block_idx](hx)
        logits /= self.args.softmax_temperature
        if self.args.mode == 'train':
            logits = self.args.tanh_c * F.tanh(logits)
        return logits, (hx, cx)

    def sample(self, batch_size=1, with_details=False, save_dir=None):
        """Samples a set of `args.num_blocks` many computational nodes from the
        controller, where each node is made up of an activation function, and
        each node except the last also includes a previous node.
        """
        if batch_size < 1:
            raise Exception(f'Wrong batch_size: {batch_size} < 1')
        inputs = self.static_inputs[batch_size]
        hidden = self.static_init_hidden[batch_size]
        activations = []
        entropies = []
        log_probs = []
        prev_nodes = []
        for block_idx in range(2 * (self.args.num_blocks - 1) + 1):
            logits, hidden = self.forward(inputs, hidden, block_idx, is_embed=block_idx == 0)
            probs = F.softmax(logits, dim=-1)
            log_prob = F.log_softmax(logits, dim=-1)
            entropy = -(log_prob * probs).sum(1, keepdim=False)
            action = probs.multinomial(num_samples=1).data
            selected_log_prob = log_prob.gather(1, utils.get_variable(action, requires_grad=False))
            entropies.append(entropy)
            log_probs.append(selected_log_prob[:, 0])
            mode = block_idx % 2
            inputs = utils.get_variable(action[:, 0] + sum(self.num_tokens[:mode]), requires_grad=False)
            if mode == 0:
                activations.append(action[:, 0])
            elif mode == 1:
                prev_nodes.append(action[:, 0])
        prev_nodes = torch.stack(prev_nodes).transpose(0, 1)
        activations = torch.stack(activations).transpose(0, 1)
        dags = _construct_dags(prev_nodes, activations, self.func_names, self.args.num_blocks)
        if save_dir is not None:
            for idx, dag in enumerate(dags):
                utils.draw_network(dag, os.path.join(save_dir, f'graph{idx}.png'))
        if with_details:
            return dags, torch.cat(log_probs), torch.cat(entropies)
        return dags

    def init_hidden(self, batch_size):
        zeros = torch.zeros(batch_size, self.args.controller_hid)
        return utils.get_variable(zeros, self.args.cuda, requires_grad=False), utils.get_variable(zeros.clone(), self.args.cuda, requires_grad=False)


def size(p):
    return np.prod(p.size())


class SharedModel(torch.nn.Module):

    def __init__(self):
        torch.nn.Module.__init__(self)

    @property
    def num_parameters(self):
        return sum([size(param) for param in self.parameters()])

    def get_f(self, name):
        raise NotImplementedError()

    def get_num_cell_parameters(self, dag):
        raise NotImplementedError()

    def reset_parameters(self):
        raise NotImplementedError()


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv5x5(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=5, stride=stride, padding=1, bias=False)


def conv(kernel, planes):
    if kernel == 3:
        _conv = conv3x3
    elif kernel == 5:
        _conv = conv5x5
    else:
        raise NotImplemented(f'Unkown kernel size: {kernel}')
    return nn.Sequential(nn.ReLU(inplace=True), _conv(planes, planes), nn.BatchNorm2d(planes))


class CNN(SharedModel):

    def __init__(self, args, images):
        super(CNN, self).__init__()
        self.args = args
        self.images = images
        self.w_c, self.w_h = defaultdict(dict), defaultdict(dict)
        self.reset_parameters()
        self.conv = defaultdict(dict)
        for idx in range(args.num_blocks):
            for jdx in range(idx + 1, args.num_blocks):
                self.conv[idx][jdx] = conv()
        raise NotImplemented('In progress...')

    def forward(self, inputs, dag):
        pass

    def get_f(self, name):
        name = name.lower()
        return f

    def get_num_cell_parameters(self, dag):
        pass

    def reset_parameters(self):
        pass


class EmbeddingDropout(torch.nn.Embedding):
    """Class for dropping out embeddings by zero'ing out parameters in the
    embedding matrix.

    This is equivalent to dropping out particular words, e.g., in the sentence
    'the quick brown fox jumps over the lazy dog', dropping out 'the' would
    lead to the sentence '### quick brown fox jumps over ### lazy dog' (in the
    embedding vector space).

    See 'A Theoretically Grounded Application of Dropout in Recurrent Neural
    Networks', (Gal and Ghahramani, 2016).
    """

    def __init__(self, num_embeddings, embedding_dim, max_norm=None, norm_type=2, scale_grad_by_freq=False, sparse=False, dropout=0.1, scale=None):
        """Embedding constructor.

        Args:
            dropout: Dropout probability.
            scale: Used to scale parameters of embedding weight matrix that are
                not dropped out. Note that this is _in addition_ to the
                `1/(1 - dropout)` scaling.

        See `torch.nn.Embedding` for remaining arguments.
        """
        torch.nn.Embedding.__init__(self, num_embeddings=num_embeddings, embedding_dim=embedding_dim, max_norm=max_norm, norm_type=norm_type, scale_grad_by_freq=scale_grad_by_freq, sparse=sparse)
        self.dropout = dropout
        assert dropout >= 0.0 and dropout < 1.0, 'Dropout must be >= 0.0 and < 1.0'
        self.scale = scale

    def forward(self, inputs):
        """Embeds `inputs` with the dropped out embedding weight matrix."""
        if self.training:
            dropout = self.dropout
        else:
            dropout = 0
        if dropout:
            mask = self.weight.data.new(self.weight.size(0), 1)
            mask.bernoulli_(1 - dropout)
            mask = mask.expand_as(self.weight)
            mask = mask / (1 - dropout)
            masked_weight = self.weight * Variable(mask)
        else:
            masked_weight = self.weight
        if self.scale and self.scale != 1:
            masked_weight = masked_weight * self.scale
        return F.embedding(inputs, masked_weight, max_norm=self.max_norm, norm_type=self.norm_type, scale_grad_by_freq=self.scale_grad_by_freq, sparse=self.sparse)


class LockedDropout(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x, dropout=0.5):
        if not self.training or not dropout:
            return x
        m = x.data.new(1, x.size(1), x.size(2)).bernoulli_(1 - dropout)
        mask = Variable(m, requires_grad=False) / (1 - dropout)
        mask = mask.expand_as(x)
        return mask * x


def _get_dropped_weights(w_raw, dropout_p, is_training):
    """Drops out weights to implement DropConnect.

    Args:
        w_raw: Full, pre-dropout, weights to be dropped out.
        dropout_p: Proportion of weights to drop out.
        is_training: True iff _shared_ model is training.

    Returns:
        The dropped weights.

    TODO(brendan): Why does torch.nn.functional.dropout() return:
    1. `torch.autograd.Variable()` on the training loop
    2. `torch.nn.Parameter()` on the controller or eval loop, when
    training = False...

    Even though the call to `_setweights` in the Smerity repo's
    `weight_drop.py` does not have this behaviour, and `F.dropout` always
    returns `torch.autograd.Variable` there, even when `training=False`?

    The above TODO is the reason for the hacky check for `torch.nn.Parameter`.
    """
    dropped_w = F.dropout(w_raw, p=dropout_p, training=is_training)
    if isinstance(dropped_w, torch.nn.Parameter):
        dropped_w = dropped_w.clone()
    return dropped_w


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (LockedDropout,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
]

