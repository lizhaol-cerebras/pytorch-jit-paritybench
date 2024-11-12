
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


import copy


import numpy as np


import pandas as pd


import torch


import torch.optim as optim


from torch.utils.data import DataLoader


from torch.utils.data import Subset


from torch.utils.data import SubsetRandomSampler


from torch.utils.data import Dataset


from torch.nn import Linear


from torch.nn import Bilinear


from torch.nn import Sigmoid


from torch.nn import Softplus


from torch.nn import ELU


from torch.nn import ReLU


from torch.nn import SELU


from torch.nn import CELU


from torch.nn import BatchNorm1d


from torch.nn import ModuleList


from torch.nn import Sequential


from torch.nn.modules.module import Module


import torch.nn.functional as F


import re


import torch.nn as nn


from torch.nn.utils import clip_grad_value_


class SSP(Module):
    """Applies element-wise :math:`\\text{SSP}(x)=\\text{Softplus}(x)-\\text{Softplus}(0)`

    Shifted SoftPlus (SSP)

    Args:
        beta: the :math:`\\beta` value for the Softplus formulation. Default: 1
        threshold: values above this revert to a linear function. Default: 20

    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(N, *)`, same shape as the input
    """

    def __init__(self, beta=1, threshold=20):
        super(SSP, self).__init__()
        self.beta = beta
        self.threshold = threshold

    def forward(self, input):
        sp0 = F.softplus(torch.Tensor([0]), self.beta, self.threshold).item()
        return F.softplus(input, self.beta, self.threshold) - sp0

    def extra_repr(self):
        return 'beta={}, threshold={}'.format(self.beta, self.threshold)


class BatchNormBilinear(Module):
    """
    Batch Norm Bilinear layer
    """

    def __init__(self, bilinear):
        super(BatchNormBilinear, self).__init__()
        self.in1_features = bilinear.in1_features
        self.in2_features = bilinear.in2_features
        self.out_features = bilinear.out_features
        self.bn = BatchNorm1d(self.out_features)
        self.bilinear = bilinear

    def forward(self, input1, input2):
        output = self.bn(self.bilinear(input1, input2))
        return output


class NodeEmbedding(Module):
    """
    Node Embedding layer
    """

    def __init__(self, in_features, out_features):
        super(NodeEmbedding, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.linear = Linear(in_features, out_features, bias=False)

    def forward(self, input):
        output = self.linear(input)
        return output


class EdgeNetwork(Module):
    """
    Edge Network layer
    """

    def __init__(self, in_features, out_features, n_layers, activation=ELU(), use_batch_norm=False, bias=False, use_shortcut=False):
        super(EdgeNetwork, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bilinears = [Bilinear(in_features, in_features, out_features, bias=not use_batch_norm and bias)]
        self.bilinears += [Bilinear(in_features, out_features, out_features, bias=not use_batch_norm and bias) for _ in range(n_layers - 1)]
        if use_shortcut:
            self.shortcuts = [Linear(in_features, out_features, bias=False)]
            self.shortcuts += [Linear(out_features, out_features, bias=False) for _ in range(n_layers - 1)]
        else:
            self.shortcuts = [None for _ in range(n_layers)]
        if use_batch_norm:
            self.bilinears = [BatchNormBilinear(layer) for layer in self.bilinears]
        self.bilinears = ModuleList(self.bilinears)
        self.shortcuts = ModuleList(self.shortcuts)
        self.activation = activation

    def forward(self, input, edge_sources, e):
        h = input[edge_sources]
        h = h.contiguous()
        z = e.contiguous()
        for bilinear, shortcut in zip(self.bilinears, self.shortcuts):
            if shortcut is not None:
                sc = shortcut(z)
            z = self.activation(bilinear(h, z))
            if shortcut is not None:
                z = z + sc
        return z


def _add_bn(layer):
    return Sequential(layer, BatchNorm1d(layer.out_features))


class FastEdgeNetwork(Module):
    """
    Fast Edge Network layer
    """

    def __init__(self, in_features, out_features, n_layers, activation=ELU(), net_type=0, use_batch_norm=False, bias=False, use_shortcut=False):
        super(FastEdgeNetwork, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.node_linears = [Linear(in_features, out_features, bias=not use_batch_norm and bias) for _ in range(n_layers)]
        self.edge_linears = [Linear(in_features, out_features, bias=not use_batch_norm and bias)]
        self.edge_linears += [Linear(out_features, out_features, bias=not use_batch_norm and bias) for _ in range(n_layers - 1)]
        if use_shortcut:
            self.shortcuts = [Linear(in_features, out_features, bias=False)]
            self.shortcuts += [Linear(out_features, out_features, bias=False) for _ in range(n_layers - 1)]
        else:
            self.shortcuts = [None for _ in range(n_layers)]
        if use_batch_norm:
            self.node_linears = [_add_bn(layer) for layer in self.node_linears]
            self.edge_linears = [_add_bn(layer) for layer in self.edge_linears]
        self.node_linears = ModuleList(self.node_linears)
        self.edge_linears = ModuleList(self.edge_linears)
        self.shortcuts = ModuleList(self.shortcuts)
        self.activation = activation
        self.net_type = net_type

    def forward(self, input, edge_sources, e):
        z = e
        for node_linear, edge_linear, shortcut in zip(self.node_linears, self.edge_linears, self.shortcuts):
            if shortcut is not None:
                sc = shortcut(z)
            z = edge_linear(z)
            if self.net_type == 0:
                h = node_linear(input.clone())[edge_sources]
                z = self.activation(h * z)
            else:
                h = self.activation(node_linear(input.clone()))[edge_sources]
                z = h * self.activation(z)
            if shortcut is not None:
                z += sc
        return z


class AggregatedBilinear(Module):
    """
    Aggregated Bilinear layer
    """

    def __init__(self, in1_features, in2_features, out_features, cardinality=32, width=4, activation=ELU(), use_batch_norm=False, bias=False):
        super(AggregatedBilinear, self).__init__()
        self.in1_features = in1_features
        self.in2_features = in2_features
        self.out_features = out_features
        self.cardinality = cardinality
        self.width = width
        self.interal_features = cardinality * width
        self.fc_in1 = Linear(in1_features, self.interal_features, bias=not use_batch_norm and bias)
        self.fc_in2 = Linear(in2_features, self.interal_features, bias=not use_batch_norm and bias)
        self.fc_out = Linear(self.interal_features, out_features, bias=not use_batch_norm and bias)
        self.bilinears = [Bilinear(width, width, width, bias=not use_batch_norm and bias) for _ in range(cardinality)]
        if use_batch_norm:
            self.fc_in1 = _add_bn(self.fc_in1)
            self.fc_in2 = _add_bn(self.fc_in2)
            self.bilinears = [BatchNormBilinear(layer) for layer in self.bilinears]
        self.bilinears = ModuleList(self.bilinears)
        self.activation = activation

    def forward(self, input1, input2):
        x1 = self.activation(self.fc_in1(input1))
        x2 = self.activation(self.fc_in2(input2))
        x1 = torch.chunk(x1, self.cardinality, dim=1)
        x2 = torch.chunk(x2, self.cardinality, dim=1)
        output = [bl(c1, c2) for bl, c1, c2 in zip(self.bilinears, x1, x2)]
        output = self.activation(torch.cat(output, dim=1))
        output = self.fc_out(output)
        return output


class AggregatedEdgeNetwork(Module):
    """
    Aggregated Edge Network layer
    """

    def __init__(self, in_features, out_features, n_layers, cardinality=32, width=4, activation=ELU(), use_batch_norm=False, bias=False, use_shortcut=False):
        super(AggregatedEdgeNetwork, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bilinears = [AggregatedBilinear(in_features, in_features, out_features, cardinality=cardinality, width=width, bias=not use_batch_norm and bias)]
        self.bilinears += [AggregatedBilinear(in_features, out_features, out_features, cardinality=cardinality, width=width, bias=not use_batch_norm and bias) for _ in range(n_layers - 1)]
        if use_shortcut:
            self.shortcuts = [Linear(in_features, out_features, bias=False)]
            self.shortcuts += [Linear(out_features, out_features, bias=False) for _ in range(n_layers - 1)]
        else:
            self.shortcuts = [None for _ in range(n_layers)]
        if use_batch_norm:
            self.bilinears = [BatchNormBilinear(layer) for layer in self.bilinears]
        self.bilinears = ModuleList(self.bilinears)
        self.shortcuts = ModuleList(self.shortcuts)
        self.activation = activation

    def forward(self, input, edge_sources, e):
        h = input[edge_sources]
        z = e
        for bilinear, shortcut in zip(self.bilinears, self.shortcuts):
            if shortcut is not None:
                sc = shortcut(z)
            z = self.activation(bilinear(h, z))
            if shortcut is not None:
                z = z + sc
        return z


class PostconvolutionNetwork(Module):
    """
    Postconvolution Network layer
    """

    def __init__(self, in_features, out_features, n_layers, activation=ELU(), use_batch_norm=False, bias=False):
        super(PostconvolutionNetwork, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.linears = [Linear(in_features, out_features, bias=not use_batch_norm and bias)]
        self.linears += [Linear(out_features, out_features, bias=not use_batch_norm and bias) for _ in range(n_layers - 1)]
        if use_batch_norm:
            self.linears = [_add_bn(layer) for layer in self.linears]
        self.linears = ModuleList(self.linears)
        self.activation = activation

    def forward(self, x):
        for linear in self.linears:
            x = self.activation(linear(x))
        return x


def _bn_act(num_features, activation, use_batch_norm=False):
    if use_batch_norm:
        if activation is None:
            return BatchNorm1d(num_features)
        else:
            return Sequential(BatchNorm1d(num_features), activation)
    else:
        return activation


class GatedGraphConvolution(Module):
    """
    Gated Graph Convolution layer
    """

    def __init__(self, in_features, out_features, gate_activation=Sigmoid(), node_activation=None, edge_activation=None, edge_network=None, use_node_batch_norm=False, use_edge_batch_norm=False, bias=False, postconv_network=None, conv_type=0):
        super(GatedGraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.linear_h = None
        if edge_network is None:
            self.linear = Linear(in_features, 2 * out_features, bias=not use_edge_batch_norm and bias)
        else:
            if conv_type > 0:
                linear_out_features = out_features
                self.linear_h = Linear(in_features, out_features, bias=not use_edge_batch_norm and bias)
            else:
                linear_out_features = 2 * out_features
            self.linear = Linear(edge_network.out_features, linear_out_features, bias=not use_edge_batch_norm and bias)
        self.gate_activation = _bn_act(out_features, gate_activation, use_edge_batch_norm)
        self.node_activation = _bn_act(out_features, node_activation, use_node_batch_norm)
        self.edge_activation = _bn_act(out_features, edge_activation, use_edge_batch_norm)
        self.edge_network = edge_network
        self.postconv_network = postconv_network

    def forward(self, input, edge_sources, edge_targets):
        h = input[edge_targets]
        if self.edge_network is None:
            e = h
        else:
            e = self.edge_network(input, edge_sources, h)
        if self.linear_h is None:
            e = self.linear(e)
            g, e = torch.chunk(e, 2, dim=1)
        else:
            g = self.linear(e)
            e = self.linear_h(h)
        g = self.gate_activation(g)
        if self.edge_activation is not None:
            e = self.edge_activation(e)
        if self.postconv_network is None:
            output = input.clone()
            output.index_add_(0, edge_sources, g * e)
        else:
            output = torch.zeros_like(input)
            output.index_add_(0, edge_sources, g * e)
            output = input + self.postconv_network(output)
        if self.node_activation is not None:
            output = self.node_activation(output)
        return output


class LinearPooling(Module):
    """
    Linear Pooling layer
    """

    def __init__(self, num_features):
        super(LinearPooling, self).__init__()
        self.num_features = num_features

    def forward(self, input, graph_indices, node_counts):
        graph_count = node_counts.size(0)
        g = torch.zeros(graph_count, self.num_features)
        g.index_add_(0, graph_indices, input)
        n = torch.unsqueeze(node_counts, 1)
        output = g / n
        return output


class GatedPooling(Module):
    """
    Gated Pooling layer
    """

    def __init__(self, num_features, gate_activation=Sigmoid(), bias=True):
        super(GatedPooling, self).__init__()
        self.num_features = num_features
        self.linear = Linear(num_features, num_features, bias=bias)
        self.gate_activation = gate_activation
        self.pooling = LinearPooling(num_features)

    def forward(self, input, graph_indices, node_counts):
        input = self.gate_activation(self.linear(input.clone())) * input.clone()
        output = self.pooling(input, graph_indices, node_counts)
        return output


class GraphPooling(Module):
    """
    Graph Pooling layer
    """

    def __init__(self, num_features, n_steps, activation=Softplus(), use_batch_norm=False):
        super(GraphPooling, self).__init__()
        self.num_features = num_features
        if n_steps > 1:
            self.linears = [Linear(num_features, num_features, bias=False) for _ in range(n_steps)]
            self.linears = ModuleList(self.linears)
        else:
            self.linears = None
        if use_batch_norm:
            self.activation = Sequential(BatchNorm1d(num_features), activation)
        else:
            self.activation = activation

    def forward(self, input):
        if self.linears is not None:
            input = [linear(m) for m, linear in zip(input, self.linears)]
            input = torch.sum(torch.stack(input), 0)
        else:
            input = input[0]
        output = self.activation(input)
        return output


class FullConnection(Module):
    """
    Full Connection layer
    """

    def __init__(self, in_features, out_features, activation=Softplus(), use_batch_norm=False, bias=True):
        super(FullConnection, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        if use_batch_norm:
            self.linear = Linear(in_features, out_features, bias=False)
            self.activation = Sequential(BatchNorm1d(out_features), activation)
        else:
            self.linear = Linear(in_features, out_features, bias=bias)
            self.activation = activation

    def forward(self, input):
        output = self.activation(self.linear(input))
        return output


class LinearRegression(Module):
    """
    Linear Regression layer
    """

    def __init__(self, in_features, out_features=1):
        super(LinearRegression, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.linear = Linear(in_features, out_features)

    def forward(self, input):
        output = self.linear(input)
        return torch.squeeze(output)


class Extension(Module):
    """
    Extension layer
    """

    def __init__(self, num_features=1):
        super(Extension, self).__init__()
        self.num_features = num_features

    def forward(self, input, node_counts):
        if self.num_features > 1:
            n = torch.unsqueeze(node_counts, 1)
        else:
            n = node_counts
        output = input * n
        return output


def get_activation(name):
    act_name = name.lower()
    m = re.match('(\\w+)\\((\\d+\\.\\d+)\\)', act_name)
    if m is not None:
        act_name, alpha = m.groups()
        alpha = float(alpha)
        None
    else:
        alpha = 1.0
    if act_name == 'softplus':
        return Softplus()
    elif act_name == 'ssp':
        return SSP()
    elif act_name == 'elu':
        return ELU(alpha)
    elif act_name == 'relu':
        return ReLU()
    elif act_name == 'selu':
        return SELU()
    elif act_name == 'celu':
        return CELU(alpha)
    else:
        raise NameError('Not supported activation: {}'.format(name))


class GGNN(nn.Module):
    """
    Gated Graph Neural Networks

    Nodes -> Embedding -> Gated Convolutions -> Graph Pooling -> Full Connections -> Linear Regression

    * Yujia Li, et al., "Gated graph sequence neural networks", https://arxiv.org/abs/1511.05493
    * Justin Gilmer, et al., "Neural Message Passing for Quantum Chemistry", https://arxiv.org/abs/1704.01212
    * Tian Xie, et al., "Crystal Graph Convolutional Neural Networks for an Accurate and
                         Interpretable Prediction of Material Properties", https://arxiv.org/abs/1710.10324
    """

    def __init__(self, n_node_feat, n_hidden_feat, n_graph_feat, n_conv, n_fc, activation, use_batch_norm, node_activation, use_node_batch_norm, edge_activation, use_edge_batch_norm, n_edge_net_feat, n_edge_net_layers, edge_net_activation, use_edge_net_batch_norm, use_fast_edge_network, fast_edge_network_type, use_aggregated_edge_network, edge_net_cardinality, edge_net_width, use_edge_net_shortcut, n_postconv_net_layers, postconv_net_activation, use_postconv_net_batch_norm, conv_type, conv_bias=False, edge_net_bias=False, postconv_net_bias=False, full_pooling=False, gated_pooling=False, use_extension=False):
        super(GGNN, self).__init__()
        act_fn = get_activation(activation)
        if node_activation is not None:
            node_act_fn = get_activation(node_activation)
        else:
            node_act_fn = None
        if edge_activation is not None:
            edge_act_fn = get_activation(edge_activation)
        else:
            edge_act_fn = None
        postconv_net_act_fn = get_activation(postconv_net_activation)
        if n_edge_net_layers < 1:
            edge_nets = [None for i in range(n_conv)]
        else:
            edge_net_act_fn = get_activation(edge_net_activation)
            if use_aggregated_edge_network:
                edge_nets = [AggregatedEdgeNetwork(n_hidden_feat, n_edge_net_feat, n_edge_net_layers, cardinality=edge_net_cardinality, width=edge_net_width, activation=edge_net_act_fn, use_batch_norm=use_edge_net_batch_norm, bias=edge_net_bias, use_shortcut=use_edge_net_shortcut) for i in range(n_conv)]
            elif use_fast_edge_network:
                edge_nets = [FastEdgeNetwork(n_hidden_feat, n_edge_net_feat, n_edge_net_layers, activation=edge_net_act_fn, net_type=fast_edge_network_type, use_batch_norm=use_edge_net_batch_norm, bias=edge_net_bias, use_shortcut=use_edge_net_shortcut) for i in range(n_conv)]
            else:
                edge_nets = [EdgeNetwork(n_hidden_feat, n_edge_net_feat, n_edge_net_layers, activation=edge_net_act_fn, use_batch_norm=use_edge_net_batch_norm, bias=edge_net_bias, use_shortcut=use_edge_net_shortcut) for i in range(n_conv)]
        if n_postconv_net_layers < 1:
            postconv_nets = [None for i in range(n_conv)]
        else:
            postconv_nets = [PostconvolutionNetwork(n_hidden_feat, n_hidden_feat, n_postconv_net_layers, activation=postconv_net_act_fn, use_batch_norm=use_postconv_net_batch_norm, bias=postconv_net_bias) for i in range(n_conv)]
        self.embedding = NodeEmbedding(n_node_feat, n_hidden_feat)
        self.convs = [GatedGraphConvolution(n_hidden_feat, n_hidden_feat, node_activation=node_act_fn, edge_activation=edge_act_fn, use_node_batch_norm=use_node_batch_norm, use_edge_batch_norm=use_edge_batch_norm, bias=conv_bias, conv_type=conv_type, edge_network=edge_nets[i], postconv_network=postconv_nets[i]) for i in range(n_conv)]
        self.convs = nn.ModuleList(self.convs)
        if full_pooling:
            n_steps = n_conv
            if gated_pooling:
                self.pre_poolings = [GatedPooling(n_hidden_feat) for _ in range(n_conv)]
            else:
                self.pre_poolings = [LinearPooling(n_hidden_feat) for _ in range(n_conv)]
        else:
            n_steps = 1
            self.pre_poolings = [None for _ in range(n_conv - 1)]
            if gated_pooling:
                self.pre_poolings.append(GatedPooling(n_hidden_feat))
            else:
                self.pre_poolings.append(LinearPooling(n_hidden_feat))
        self.pre_poolings = nn.ModuleList(self.pre_poolings)
        self.pooling = GraphPooling(n_hidden_feat, n_steps, activation=act_fn, use_batch_norm=use_batch_norm)
        self.fcs = [FullConnection(n_hidden_feat, n_graph_feat, activation=act_fn, use_batch_norm=use_batch_norm)]
        self.fcs += [FullConnection(n_graph_feat, n_graph_feat, activation=act_fn, use_batch_norm=use_batch_norm) for i in range(n_fc - 1)]
        self.fcs = nn.ModuleList(self.fcs)
        self.regression = LinearRegression(n_graph_feat)
        if use_extension:
            self.extension = Extension()
        else:
            self.extension = None

    def forward(self, input):
        x = self.embedding(input.nodes)
        y = []
        for conv, pre_pooling in zip(self.convs, self.pre_poolings):
            x = conv(x, input.edge_sources, input.edge_targets)
            if pre_pooling is not None:
                y.append(pre_pooling(x, input.graph_indices, input.node_counts))
        x = self.pooling(y)
        for fc in self.fcs:
            x = fc(x)
        x = self.regression(x)
        if self.extension is not None:
            x = self.extension(x, input.node_counts)
        return x


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (AggregatedBilinear,
     lambda: ([], {'in1_features': 4, 'in2_features': 4, 'out_features': 4}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4])], {})),
    (Extension,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (FullConnection,
     lambda: ([], {'in_features': 4, 'out_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (GraphPooling,
     lambda: ([], {'num_features': 4, 'n_steps': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (LinearRegression,
     lambda: ([], {'in_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (NodeEmbedding,
     lambda: ([], {'in_features': 4, 'out_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (PostconvolutionNetwork,
     lambda: ([], {'in_features': 4, 'out_features': 4, 'n_layers': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (SSP,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
]

