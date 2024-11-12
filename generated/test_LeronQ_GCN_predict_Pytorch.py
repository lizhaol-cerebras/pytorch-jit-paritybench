
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


import torch.nn.init as init


import torch.nn.functional as F


import numpy as np


from torch.utils.data import Dataset


import time


import torch.optim as optim


from torch.utils.data import DataLoader


import warnings


class ChebConv(nn.Module):
    """
    The ChebNet convolution operation.

    :param in_c: int, number of input channels.
    :param out_c: int, number of output channels.
    :param K: int, the order of Chebyshev Polynomial.
    """

    def __init__(self, in_c, out_c, K, bias=True, normalize=True):
        super(ChebConv, self).__init__()
        self.normalize = normalize
        self.weight = nn.Parameter(torch.Tensor(K + 1, 1, in_c, out_c))
        init.xavier_normal_(self.weight)
        if bias:
            self.bias = nn.Parameter(torch.Tensor(1, 1, out_c))
            init.zeros_(self.bias)
        else:
            self.register_parameter('bias', None)
        self.K = K + 1

    def forward(self, inputs, graph):
        """
        :param inputs: the input data, [B, N, C]
        :param graph: the graph structure, [N, N]
        :return: convolution result, [B, N, D]
        """
        L = ChebConv.get_laplacian(graph, self.normalize)
        mul_L = self.cheb_polynomial(L).unsqueeze(1)
        result = torch.matmul(mul_L, inputs)
        result = torch.matmul(result, self.weight)
        result = torch.sum(result, dim=0) + self.bias
        return result

    def cheb_polynomial(self, laplacian):
        """
        Compute the Chebyshev Polynomial, according to the graph laplacian.

        :param laplacian: the graph laplacian, [N, N].
        :return: the multi order Chebyshev laplacian, [K, N, N].
        """
        N = laplacian.size(0)
        multi_order_laplacian = torch.zeros([self.K, N, N], device=laplacian.device, dtype=torch.float)
        multi_order_laplacian[0] = torch.eye(N, device=laplacian.device, dtype=torch.float)
        if self.K == 1:
            return multi_order_laplacian
        else:
            multi_order_laplacian[1] = laplacian
            if self.K == 2:
                return multi_order_laplacian
            else:
                for k in range(2, self.K):
                    multi_order_laplacian[k] = 2 * torch.mm(laplacian, multi_order_laplacian[k - 1]) - multi_order_laplacian[k - 2]
        return multi_order_laplacian

    @staticmethod
    def get_laplacian(graph, normalize):
        """
        return the laplacian of the graph.

        :param graph: the graph structure without self loop, [N, N].
        :param normalize: whether to used the normalized laplacian.
        :return: graph laplacian.
        """
        if normalize:
            D = torch.diag(torch.sum(graph, dim=-1) ** (-1 / 2))
            L = torch.eye(graph.size(0), device=graph.device, dtype=graph.dtype) - torch.mm(torch.mm(D, graph), D)
        else:
            D = torch.diag(torch.sum(graph, dim=-1))
            L = D - graph
        return L


class ChebNet(nn.Module):

    def __init__(self, in_c, hid_c, out_c, K):
        """
        :param in_c: int, number of input channels.
        :param hid_c: int, number of hidden channels.class
        :param out_c: int, number of output channels.
        :param K:
        """
        super(ChebNet, self).__init__()
        self.conv1 = ChebConv(in_c=in_c, out_c=hid_c, K=K)
        self.conv2 = ChebConv(in_c=hid_c, out_c=out_c, K=K)
        self.act = nn.ReLU()

    def forward(self, data, device):
        graph_data = data['graph'][0]
        flow_x = data['flow_x']
        B, N = flow_x.size(0), flow_x.size(1)
        flow_x = flow_x.view(B, N, -1)
        output_1 = self.act(self.conv1(flow_x, graph_data))
        output_2 = self.act(self.conv2(output_1, graph_data))
        return output_2.unsqueeze(2)


class GraphAttentionLayer(nn.Module):

    def __init__(self, in_c, out_c):
        super(GraphAttentionLayer, self).__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.F = F.softmax
        self.W = nn.Linear(in_c, out_c, bias=False)
        self.b = nn.Parameter(torch.Tensor(out_c))
        nn.init.normal_(self.W.weight)
        nn.init.normal_(self.b)

    def forward(self, inputs, graph):
        """
        :param inputs: input features, [B, N, C].
        :param graph: graph structure, [N, N].
        :return:
            output features, [B, N, D].
        """
        h = self.W(inputs)
        outputs = torch.bmm(h, h.transpose(1, 2)) * graph.unsqueeze(0)
        outputs.data.masked_fill_(torch.eq(outputs, 0), -float(1e+16))
        attention = self.F(outputs, dim=2)
        return torch.bmm(attention, h) + self.b


class GATSubNet(nn.Module):

    def __init__(self, in_c, hid_c, out_c, n_heads):
        super(GATSubNet, self).__init__()
        self.attention_module = nn.ModuleList([GraphAttentionLayer(in_c, hid_c) for _ in range(n_heads)])
        self.out_att = GraphAttentionLayer(hid_c * n_heads, out_c)
        self.act = nn.LeakyReLU()

    def forward(self, inputs, graph):
        """
        :param inputs: [B, N, C]
        :param graph: [N, N]
        :return:
        """
        outputs = torch.cat([attn(inputs, graph) for attn in self.attention_module], dim=-1)
        outputs = self.act(outputs)
        outputs = self.out_att(outputs, graph)
        return self.act(outputs)


class GATNet(nn.Module):

    def __init__(self, in_c, hid_c, out_c, n_heads):
        super(GATNet, self).__init__()
        self.subnet = GATSubNet(in_c, hid_c, out_c, n_heads)

    def forward(self, data, device):
        graph = data['graph'][0]
        flow = data['flow_x']
        flow = flow
        B, N = flow.size(0), flow.size(1)
        flow = flow.view(B, N, -1)
        """
       上面是将这一段的时间的特征数据摊平做为特征，这种做法实际上忽略了时序上的连续性
       这种做法可行，但是比较粗糙，当然也可以这么做：
       flow[:, :, 0] ... flow[:, :, T-1]   则就有T个[B, N, C]这样的张量，也就是 [B, N, C]*T
       每一个张量都用一个SubNet来表示，则一共有T个SubNet，初始化定义　self.subnet = [GATSubNet(...) for _ in range(T)]
       然后用nn.ModuleList将SubNet分别拎出来处理，参考多头注意力的处理，同理

       """
        prediction = self.subnet(flow, graph).unsqueeze(2)
        return prediction


class GCN(nn.Module):

    def __init__(self, in_c, hid_c, out_c):
        super(GCN, self).__init__()
        self.linear_1 = nn.Linear(in_c, hid_c)
        self.linear_2 = nn.Linear(hid_c, out_c)
        self.act = nn.ReLU()

    def forward(self, data, device):
        graph_data = data['graph'][0]
        graph_data = GCN.process_graph(graph_data)
        flow_x = data['flow_x']
        B, N = flow_x.size(0), flow_x.size(1)
        flow_x = flow_x.view(B, N, -1)
        output_1 = self.linear_1(flow_x)
        output_1 = self.act(torch.matmul(graph_data, output_1))
        output_2 = self.linear_2(output_1)
        output_2 = self.act(torch.matmul(graph_data, output_2))
        return output_2.unsqueeze(2)

    @staticmethod
    def process_graph(graph_data):
        N = graph_data.size(0)
        matrix_i = torch.eye(N, dtype=torch.float, device=graph_data.device)
        graph_data += matrix_i
        degree_matrix = torch.sum(graph_data, dim=1, keepdim=False)
        degree_matrix = degree_matrix.pow(-1)
        degree_matrix[degree_matrix == float('inf')] = 0.0
        degree_matrix = torch.diag(degree_matrix)
        return torch.mm(degree_matrix, graph_data)


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (ChebConv,
     lambda: ([], {'in_c': 4, 'out_c': 4, 'K': 4}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4])], {})),
]

