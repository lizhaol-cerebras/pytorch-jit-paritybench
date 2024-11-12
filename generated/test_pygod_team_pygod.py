
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


import warnings


import time


from torch import nn


import torch.nn.functional as F


from inspect import signature


from abc import ABC


from abc import abstractmethod


import numpy as np


from scipy.stats import binom


from scipy.special import erf


from copy import deepcopy


import math


from sklearn.metrics import roc_auc_score


from sklearn.metrics import average_precision_score


from sklearn.metrics import f1_score


import torch.nn as nn


from torch.nn.functional import binary_cross_entropy_with_logits


from sklearn.cluster import KMeans


from scipy.linalg import sqrtm


import random


import torch.multiprocessing as mp


from numpy.testing import assert_equal


from numpy.testing import assert_warns


from numpy.testing import assert_raises


from torch.testing import assert_close


from numpy.testing import assert_allclose


from sklearn.metrics import precision_score


from sklearn.metrics import recall_score


import copy as cp


import numbers


class ANOMALOUSBase(nn.Module):

    def __init__(self, w, r):
        super(ANOMALOUSBase, self).__init__()
        self.w = nn.Parameter(w)
        self.r = nn.Parameter(r)

    def forward(self, x):
        return x @ self.w @ x, self.r


class ONEBase(torch.nn.Module):

    def __init__(self, g, h, u, v, w, alpha=1.0, beta=1.0, gamma=1.0):
        super(ONEBase, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.g = torch.nn.Parameter(g)
        self.h = torch.nn.Parameter(h)
        self.u = torch.nn.Parameter(u)
        self.v = torch.nn.Parameter(v)
        self.w = torch.nn.Parameter(w)

    def forward(self):
        x_ = self.u @ self.v
        s_ = self.g @ self.h
        diff = self.g - self.u @ self.w
        return x_, s_, diff

    def loss_func(self, x, x_, s, s_, diff):
        dx = torch.sum(torch.pow(x - x_, 2), 1)
        o1 = dx / torch.sum(dx)
        loss_a = torch.mean(torch.log(torch.pow(o1, -1)) * dx)
        ds = torch.sum(torch.pow(s - s_, 2), 1)
        o2 = ds / torch.sum(ds)
        loss_s = torch.mean(torch.log(torch.pow(o2, -1)) * ds)
        dc = torch.sum(torch.pow(diff, 2), 1)
        o3 = dc / torch.sum(dc)
        loss_c = torch.mean(torch.log(torch.pow(o3, -1)) * dc)
        loss = self.alpha * loss_a + self.beta * loss_s + self.gamma * loss_c
        return loss, o1, o2, o3


class RadarBase(torch.nn.Module):

    def __init__(self, w, r):
        super(RadarBase, self).__init__()
        self.w = torch.nn.Parameter(w)
        self.r = torch.nn.Parameter(r)

    def forward(self, x):
        return self.w @ x, self.r


class DONEBase(nn.Module):
    """
    Deep Outlier Aware Attributed Network Embedding

    DONE consists of an attribute autoencoder and a structure
    autoencoder. It estimates five losses to optimize the model,
    including an attribute proximity loss, an attribute homophily loss,
    a structure proximity loss, a structure homophily loss, and a
    combination loss. It calculates three outlier scores, and averages
    them as an overall scores. This model is transductive only.

    See :cite:`bandyopadhyay2020outlier` for details.

    Parameters
    ----------
    x_dim : int
        Input dimension of attribute.
    s_dim : int
        Input dimension of structure.
    hid_dim :  int, optional
        Hidden dimension of model. Default: ``64``.
    num_layers : int, optional
        Total number of layers in model. A half (floor) of the layers
        are for the encoder, the other half (ceil) of the layers are for
        decoders. Default: ``4``.
    dropout : float, optional
        Dropout rate. Default: ``0.``.
    weight_decay : float, optional
        Weight decay (L2 penalty). Default: ``0.``.
    act : callable activation function or None, optional
        Activation function if not None.
        Default: ``torch.nn.functional.relu``.
    w1 : float, optional
        Weight of structure proximity loss. Default: ``0.2``.
    w2 : float, optional
        Weight of structure homophily loss. Default: ``0.2``.
    w3 : float, optional
        Weight of attribute proximity loss. Default: ``0.2``.
    w4 : float, optional
        Weight of attribute homophily loss. Default: ``0.2``.
    w5 : float, optional
        Weight of combination loss. Default: ``0.2``.
    **kwargs
        Other parameters for the backbone.
    """

    def __init__(self, x_dim, s_dim, hid_dim=64, num_layers=4, dropout=0.0, act=torch.nn.functional.relu, w1=0.2, w2=0.2, w3=0.2, w4=0.2, w5=0.2, **kwargs):
        super(DONEBase, self).__init__()
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.w4 = w4
        self.w5 = w5
        assert num_layers >= 2, 'Number of layers must be greater than or equal to 2.'
        encoder_layers = math.floor(num_layers / 2)
        decoder_layers = math.ceil(num_layers / 2)
        self.attr_encoder = MLP(in_channels=x_dim, hidden_channels=hid_dim, out_channels=hid_dim, num_layers=encoder_layers, dropout=dropout, act=act, **kwargs)
        self.attr_decoder = MLP(in_channels=hid_dim, hidden_channels=hid_dim, out_channels=x_dim, num_layers=decoder_layers, dropout=dropout, act=act, **kwargs)
        self.struct_encoder = MLP(in_channels=s_dim, hidden_channels=hid_dim, out_channels=hid_dim, num_layers=encoder_layers, dropout=dropout, act=act, **kwargs)
        self.struct_decoder = MLP(in_channels=hid_dim, hidden_channels=hid_dim, out_channels=s_dim, num_layers=decoder_layers, dropout=dropout, act=act, **kwargs)
        self.neigh_diff = NeighDiff()
        self.emb = None

    def forward(self, x, s, edge_index):
        """
        Forward computation.

        Parameters
        ----------
        x : torch.Tensor
            Input attribute embeddings.
        s : torch.Tensor
            Input structure embeddings.
        edge_index : torch.Tensor
            Edge index.

        Returns
        -------
        x_ : torch.Tensor
            Reconstructed attribute embeddings.
        s_ : torch.Tensor
            Reconstructed structure embeddings.
        h_a : torch.Tensor
            Attribute hidden embeddings.
        h_s : torch.Tensor
            Structure hidden embeddings.
        dna : torch.Tensor
            Attribute neighbor distance.
        dns : torch.Tensor
            Structure neighbor distance.
        """
        h_a = self.attr_encoder(x)
        x_ = self.attr_decoder(h_a)
        dna = self.neigh_diff(h_a, edge_index).squeeze()
        h_s = self.struct_encoder(s)
        s_ = self.struct_decoder(h_s)
        dns = self.neigh_diff(h_s, edge_index).squeeze()
        self.emb = h_a, h_s
        return x_, s_, h_a, h_s, dna, dns

    def loss_func(self, x, x_, s, s_, h_a, h_s, dna, dns):
        """
        Loss function for DONE.

        Parameters
        ----------
        x : torch.Tensor
            Input attribute embeddings.
        x_ : torch.Tensor
            Reconstructed attribute embeddings.
        s : torch.Tensor
            Input structure embeddings.
        s_ : torch.Tensor
            Reconstructed structure embeddings.
        h_a : torch.Tensor
            Attribute hidden embeddings.
        h_s : torch.Tensor
            Structure hidden embeddings.
        dna : torch.Tensor
            Attribute neighbor distance.
        dns : torch.Tensor
            Structure neighbor distance.

        Returns
        -------
        loss : torch.Tensor
            Loss value.
        oa : torch.Tensor
            Attribute outlier scores.
        os : torch.Tensor
            Structure outlier scores.
        oc : torch.Tensor
            Combined outlier scores.
        """
        dx = torch.sum(torch.pow(x - x_, 2), 1)
        tmp = self.w3 * dx + self.w4 * dna
        oa = tmp / torch.sum(tmp)
        ds = torch.sum(torch.pow(s - s_, 2), 1)
        tmp = self.w1 * ds + self.w2 * dns
        os = tmp / torch.sum(tmp)
        dc = torch.sum(torch.pow(h_a - h_s, 2), 1)
        oc = dc / torch.sum(dc)
        loss_prox_a = torch.mean(torch.log(torch.pow(oa, -1)) * dx)
        loss_hom_a = torch.mean(torch.log(torch.pow(oa, -1)) * dna)
        loss_prox_s = torch.mean(torch.log(torch.pow(os, -1)) * ds)
        loss_hom_s = torch.mean(torch.log(torch.pow(os, -1)) * dns)
        loss_c = torch.mean(torch.log(torch.pow(oc, -1)) * dc)
        loss = self.w3 * loss_prox_a + self.w4 * loss_hom_a + self.w1 * loss_prox_s + self.w2 * loss_hom_s + self.w5 * loss_c
        return loss, oa, os, oc

    @staticmethod
    def process_graph(data):
        """
        Obtain the dense adjacency matrix of the graph.

        Parameters
        ----------
        data : torch_geometric.data.Data
            Input graph.
        """
        data.s = to_dense_adj(data.edge_index)[0]


class AdONEBase(torch.nn.Module):
    """
    Adversarial Outlier Aware Attributed Network Embedding

    AdONE consists of an attribute autoencoder and a structure
    autoencoder. It estimates five loss to optimize the model,
    including an attribute proximity loss, an attribute homophily loss,
    a structure proximity loss, a structure homophily loss, and an
    alignment loss. It calculates three outlier scores, and averages
    them as an overall score. This model is transductive only.

    See :cite:`bandyopadhyay2020outlier` for details.

    Parameters
    ----------
    x_dim : int
        Input dimension of attribute.
    s_dim : int
        Input dimension of structure.
    hid_dim :  int, optional
        Hidden dimension of model. Default: ``64``.
    num_layers : int, optional
        Total number of layers in model. A half (floor) of the layers
        are for the encoder, the other half (ceil) of the layers are for
        decoders. Default: ``4``.
    dropout : float, optional
        Dropout rate. Default: ``0.``.
    weight_decay : float, optional
        Weight decay (L2 penalty). Default: ``0.``.
    act : callable activation function or None, optional
        Activation function if not None.
        Default: ``torch.nn.functional.relu``.
    w1 : float, optional
        Weight of structure proximity loss. Default: ``0.2``.
    w2 : float, optional
        Weight of structure homophily loss. Default: ``0.2``.
    w3 : float, optional
        Weight of attribute proximity loss. Default: ``0.2``.
    w4 : float, optional
        Weight of attribute homophily loss. Default: ``0.2``.
    w5 : float, optional
        Weight of alignment loss. Default: ``0.2``.
    **kwargs
        Other parameters for the backbone.
    """

    def __init__(self, x_dim, s_dim, hid_dim=64, num_layers=4, dropout=0.0, act=torch.nn.functional.relu, w1=0.2, w2=0.2, w3=0.2, w4=0.2, w5=0.2, **kwargs):
        super(AdONEBase, self).__init__()
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.w4 = w4
        self.w5 = w5
        self.generator = DONEBase(x_dim=x_dim, s_dim=s_dim, hid_dim=hid_dim, num_layers=num_layers, dropout=dropout, act=act, w1=self.w1, w2=self.w2, w3=self.w3, w4=self.w4, w5=self.w5, **kwargs)
        self.discriminator = MLP(in_channels=hid_dim, hidden_channels=int(hid_dim / 2), out_channels=1, num_layers=2, dropout=dropout, act=torch.tanh)
        self.emb = None
        self.inner = self.discriminator
        self.outer = self.generator

    def forward(self, x, s, edge_index):
        """
        Forward computation.

        Parameters
        ----------
        x : torch.Tensor
            Input attribute embeddings.
        s : torch.Tensor
            Input structure embeddings.
        edge_index : torch.Tensor
            Edge index.

        Returns
        -------
        x_ : torch.Tensor
            Reconstructed attribute embeddings.
        s_ : torch.Tensor
            Reconstructed structure embeddings.
        h_a : torch.Tensor
            Attribute hidden embeddings.
        h_s : torch.Tensor
            Structure hidden embeddings.
        dna : torch.Tensor
            Attribute neighbor distance.
        dns : torch.Tensor
            Structure neighbor distance.
        dis_a : torch.Tensor
            Attribute discriminator score.
        dis_s : torch.Tensor
            Structure discriminator score.
        """
        x_, s_, h_a, h_s, dna, dns = self.generator(x, s, edge_index)
        self.emb = h_a, h_s
        return x_, s_, h_a, h_s, dna, dns

    def loss_func_g(self, x, x_, s, s_, h_a, h_s, dna, dns):
        """
        Generator loss function for AdONE.

        Parameters
        ----------
        x : torch.Tensor
            Input attribute embeddings.
        x_ : torch.Tensor
            Reconstructed attribute embeddings.
        s : torch.Tensor
            Input structure embeddings.
        s_ : torch.Tensor
            Reconstructed structure embeddings.
        h_a : torch.Tensor
            Attribute hidden embeddings.
        h_s : torch.Tensor
            Structure hidden embeddings.
        dna : torch.Tensor
            Attribute neighbor distance.
        dns : torch.Tensor
            Structure neighbor distance.

        Returns
        -------
        loss : torch.Tensor
            Loss value.
        oa : torch.Tensor
            Attribute outlier scores.
        os : torch.Tensor
            Structure outlier scores.
        oc : torch.Tensor
            Combined outlier scores.
        """
        dx = torch.sum(torch.pow(x - x_, 2), 1)
        tmp = self.w3 * dx + self.w4 * dna
        oa = tmp / torch.sum(tmp)
        ds = torch.sum(torch.pow(s - s_, 2), 1)
        tmp = self.w1 * ds + self.w2 * dns
        os = tmp / torch.sum(tmp)
        dc = torch.sum(torch.pow(h_a - h_s, 2), 1)
        oc = dc / torch.sum(dc)
        loss_prox_a = torch.mean(torch.log(torch.pow(oa, -1)) * dx)
        loss_hom_a = torch.mean(torch.log(torch.pow(oa, -1)) * dna)
        loss_prox_s = torch.mean(torch.log(torch.pow(os, -1)) * ds)
        loss_hom_s = torch.mean(torch.log(torch.pow(os, -1)) * dns)
        dis_a = torch.sigmoid(self.discriminator(h_a))
        dis_s = torch.sigmoid(self.discriminator(h_s))
        loss_alg = torch.mean(torch.log(torch.pow(oc, -1)) * (torch.log(1 - dis_a) + torch.log(dis_s)))
        loss = self.w3 * loss_prox_a + self.w4 * loss_hom_a + self.w1 * loss_prox_s + self.w2 * loss_hom_s + self.w5 * loss_alg
        return loss, oa, os, oc

    def loss_func_d(self, h_a, h_s):
        """
        Discriminator loss function for AdONE.

        Parameters
        ----------
        h_a : torch.Tensor
            Attribute hidden embeddings.
        h_s : torch.Tensor
            Structure hidden embeddings.

        Returns
        -------
        loss : torch.Tensor
            Loss value.
        """
        dis_a = torch.sigmoid(self.discriminator(h_a))
        dis_s = torch.sigmoid(self.discriminator(h_s))
        loss = -torch.mean(torch.log(1 - dis_a) + torch.log(dis_s))
        return loss

    @staticmethod
    def process_graph(data):
        """
        Obtain the dense adjacency matrix of the graph.

        Parameters
        ----------
        data : torch_geometric.data.Data
            Input graph.
        """
        data.s = to_dense_adj(data.edge_index)[0]


def double_recon_loss(x, x_, s, s_, weight=0.5, pos_weight_a=0.5, pos_weight_s=0.5, bce_s=False):
    """
    Double reconstruction loss function for feature and structure.
    The loss function is defined as :math:`\\alpha \\symbf{E_a} +
    (1-\\alpha) \\symbf{E_s}`, where :math:`\\alpha` is the weight between
    0 and 1 inclusive, and :math:`\\symbf{E_a}` and :math:`\\symbf{E_s}`
    are the reconstruction loss for feature and structure, respectively.
    The first dimension is kept for outlier scores of each node.

    For feature reconstruction, we use mean squared error loss:
    :math:`\\symbf{E_a} = \\|\\symbf{X}-\\symbf{X}'\\|\\odot H`,
    where :math:`H=\\begin{cases}1 - \\eta &
    \\text{if }x_{ij}=0\\\\ \\eta & \\text{if }x_{ij}>0\\end{cases}`, and
    :math:`\\eta` is the positive weight for feature.

    For structure reconstruction, we use mean squared error loss by
    default: :math:`\\symbf{E_s} = \\|\\symbf{S}-\\symbf{S}'\\|\\odot
    \\Theta`, where :math:`\\Theta=\\begin{cases}1 -
    \\theta & \\text{if }s_{ij}=0\\\\ \\theta & \\text{if }s_{ij}>0
    \\end{cases}`, and :math:`\\theta` is the positive weight for
    structure. Alternatively, we can use binary cross entropy loss
    for structure reconstruction: :math:`\\symbf{E_s} =
    \\text{BCE}(\\symbf{S}, \\symbf{S}' \\odot \\Theta)`.

    Parameters
    ----------
    x : torch.Tensor
        Ground truth node feature
    x_ : torch.Tensor
        Reconstructed node feature
    s : torch.Tensor
        Ground truth node structure
    s_ : torch.Tensor
        Reconstructed node structure
    weight : float, optional
        Balancing weight :math:`\\alpha` between 0 and 1 inclusive between node feature
        and graph structure. Default: ``0.5``.
    pos_weight_a : float, optional
        Positive weight for feature :math:`\\eta`. Default: ``0.5``.
    pos_weight_s : float, optional
        Positive weight for structure :math:`\\theta`. Default: ``0.5``.
    bce_s : bool, optional
        Use binary cross entropy for structure reconstruction loss.

    Returns
    -------
    score : torch.tensor
        Outlier scores of shape :math:`N` with gradients.
    """
    assert 0 <= weight <= 1, 'weight must be a float between 0 and 1.'
    assert 0 <= pos_weight_a <= 1 and 0 <= pos_weight_s <= 1, 'positive weight must be a float between 0 and 1.'
    diff_attr = torch.pow(x - x_, 2)
    if pos_weight_a != 0.5:
        diff_attr = torch.where(x > 0, diff_attr * pos_weight_a, diff_attr * (1 - pos_weight_a))
    attr_error = torch.sqrt(torch.sum(diff_attr, 1))
    if bce_s:
        diff_stru = F.binary_cross_entropy(s_, s, reduction='none')
    else:
        diff_stru = torch.pow(s - s_, 2)
    if pos_weight_s != 0.5:
        diff_stru = torch.where(s > 0, diff_stru * pos_weight_s, diff_stru * (1 - pos_weight_s))
    stru_error = torch.sqrt(torch.sum(diff_stru, 1))
    score = weight * attr_error + (1 - weight) * stru_error
    return score


class AnomalyDAEBase(nn.Module):
    """
    Dual Autoencoder for Anomaly Detection on Attributed Networks

    AnomalyDAE is an anomaly detector that consists of a structure
    autoencoder and an attribute autoencoder to learn both node
    embedding and attribute embedding jointly in latent space. The
    structural autoencoder uses Graph Attention layers. The
    reconstruction mean square error of the decoders are defined as
    structure anomaly score and attribute anomaly score, respectively,
    with two additional penalties on the reconstructed adj matrix and
    node attributes (force entries to be nonzero).

    See :cite:`fan2020anomalydae` for details.

    Parameters
    ----------
    in_dim : int
         Input dimension of model.
    num_nodes: int
         Number of input nodes or batch size in minibatch training.
    emb_dim:: int
         Embedding dimension of model. Default: ``64``.
    hid_dim : int
         Hidden dimension of model. Default: ``64``.
    dropout : float, optional
        Dropout rate. Default: ``0.``.
    act : callable activation function or None, optional
        Activation function if not None.
        Default: ``torch.nn.functional.relu``.
    **kwargs : optional
        Other parameters of ``torch_geometric.nn.GATConv``.
    """

    def __init__(self, in_dim, num_nodes, emb_dim=64, hid_dim=64, dropout=0.0, act=F.relu, **kwargs):
        super(AnomalyDAEBase, self).__init__()
        self.num_nodes = num_nodes
        self.dense_stru = nn.Linear(in_dim, emb_dim)
        self.gat_layer = GATConv(emb_dim, hid_dim, **kwargs)
        self.dense_attr_1 = nn.Linear(self.num_nodes, emb_dim)
        self.dense_attr_2 = nn.Linear(emb_dim, hid_dim)
        self.dropout = dropout
        self.act = act
        self.loss_func = double_recon_loss
        self.emb = None

    def forward(self, x, edge_index, batch_size):
        """
        Forward computation.

        Parameters
        ----------
        x : torch.Tensor
            Input attribute embeddings.
        edge_index : torch.Tensor
            Edge index.
        batch_size : int
            Batch size.

        Returns
        -------
        x_ : torch.Tensor
            Reconstructed attribute embeddings.
        s_ : torch.Tensor
            Reconstructed adjacency matrix.
        """
        h = self.dense_stru(x)
        if self.act is not None:
            h = self.act(h)
        h = F.dropout(h, self.dropout)
        self.emb = self.gat_layer(h, edge_index)
        s_ = torch.sigmoid(self.emb @ self.emb.T)
        if batch_size < self.num_nodes:
            x = F.pad(x, (0, 0, 0, self.num_nodes - batch_size))
        x = self.dense_attr_1(x[:self.num_nodes].T)
        if self.act is not None:
            x = self.act(x)
        x = F.dropout(x, self.dropout)
        x = self.dense_attr_2(x)
        x = F.dropout(x, self.dropout)
        x_ = self.emb @ x.T
        return x_, s_

    @staticmethod
    def process_graph(data):
        """
        Obtain the dense adjacency matrix of the graph.

        Parameters
        ----------
        data : torch_geometric.data.Data
            Input graph.
        """
        data.s = to_dense_adj(data.edge_index)[0]


class GNA(torch.nn.Module):
    """
    Graph Node Attention Network (GNA). See :cite:`yuan2021higher` for
    more details.
    """

    def __init__(self, in_channels, hidden_channels, num_layers, out_channels, dropout, act):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        self.layers.append(GNAConv(in_channels, hidden_channels))
        for layer in range(num_layers - 2):
            self.layers.append(GNAConv(hidden_channels, hidden_channels))
        self.layers.append(GNAConv(hidden_channels, out_channels))
        self.dropout = dropout
        self.act = act

    def forward(self, s, edge_index):
        """
        Forward computation.

        Parameters
        ----------
        s : torch.Tensor
            Input node embeddings.
        edge_index : torch.Tensor
            Edge index.

        Returns
        -------
        s : torch.Tensor
            Updated node embeddings.
        """
        for layer in self.layers:
            s = layer(s, edge_index)
            s = F.dropout(s, self.dropout, training=self.training)
            if self.act is not None:
                s = self.act(s)
        return s


class GAANBase(torch.nn.Module):
    """
    Generative Adversarial Attributed Network Anomaly Detection

    GAAN is a generative adversarial attribute network anomaly
    detection framework, including a generator module, an encoder
    module, a discriminator module, and uses anomaly evaluation
    measures that consider sample reconstruction error and real sample
    recognition confidence to make predictions. This model is
    transductive only.

    See :cite:`chen2020generative` for details.

    Parameters
    ----------
    in_dim : int
        Input dimension of the node features.
    noise_dim :  int, optional
        Input dimension of the Gaussian random noise. Defaults: ``16``.
    hid_dim :  int, optional
        Hidden dimension of model. Default: ``64``.
    num_layers : int, optional
       Total number of layers in model. A half (floor) of the layers
       are for the generator, the other half (ceil) of the layers are
       for encoder. Default: ``4``.
    dropout : float, optional
        Dropout rate. Default: ``0.``.
    act : callable activation function or None, optional
        Activation function if not None.
        Default: ``torch.nn.functional.relu``.
    **kwargs
        Other parameters for the backbone.
    """

    def __init__(self, in_dim, noise_dim, hid_dim=64, num_layers=4, dropout=0.0, act=torch.nn.functional.relu, **kwargs):
        super(GAANBase, self).__init__()
        assert num_layers >= 2, 'Number of layers must be greater than or equal to 2.'
        generator_layers = math.floor(num_layers / 2)
        encoder_layers = math.ceil(num_layers / 2)
        self.generator = MLP(in_channels=noise_dim, hidden_channels=hid_dim, out_channels=in_dim, num_layers=generator_layers, dropout=dropout, act=act, **kwargs)
        self.discriminator = MLP(in_channels=in_dim, hidden_channels=hid_dim, out_channels=hid_dim, num_layers=encoder_layers, dropout=dropout, act=act, **kwargs)
        self.emb = None
        self.score_func = double_recon_loss
        self.inner = self.generator
        self.outer = self.discriminator

    def forward(self, x, noise):
        """
        Forward computation.

        Parameters
        ----------
        x : torch.Tensor
            Input attribute embeddings.
        noise : torch.Tensor
            Input noise.

        Returns
        -------
        x_ : torch.Tensor
            Reconstructed node features.
        a : torch.Tensor
            Reconstructed adjacency matrix from real samples.
        a_ : torch.Tensor
            Reconstructed adjacency matrix from fake samples.
        """
        x_ = self.generator(noise)
        self.emb = self.discriminator(x)
        z_ = self.discriminator(x_)
        a = torch.sigmoid(self.emb @ self.emb.T)
        a_ = torch.sigmoid(z_ @ z_.T)
        return x_, a, a_

    @staticmethod
    def loss_func_g(a_):
        loss_g = F.binary_cross_entropy(a_, torch.ones_like(a_))
        return loss_g

    @staticmethod
    def loss_func_ed(a, a_):
        loss_r = F.binary_cross_entropy(a, torch.ones_like(a))
        loss_f = F.binary_cross_entropy(a_, torch.zeros_like(a_))
        return (loss_f + loss_r) / 2

    @staticmethod
    def process_graph(data):
        """
        Obtain the dense adjacency matrix of the graph.

        Parameters
        ----------
        data : torch_geometric.data.Data
            Input graph.
        """
        data.s = to_dense_adj(data.edge_index)[0]


class MLP_GAD_NR(torch.nn.Module):
    """
    The personalized MLP module used by GAD_NR
    Source: https://github.com/Graph-COM/GAD-NR/blob/master/GAD-NR.ipynb
    
    Parameters
    ----------
    in_dim : int
        Input dimension of the embedding.
    hid_dim :  int
        Hidden dimension of model.
    out_dim : int
        Output dimension.
    num_layers : int
        Number of layers in the decoder.
    act : callable activation function or None, optional
        Activation function if not None.
        Default: ``torch.nn.functional.relu``. 
    """

    def __init__(self, in_dim, hid_dim, out_dim, num_layers, act=torch.nn.functional.relu):
        super(MLP_GAD_NR, self).__init__()
        self.linear_or_not = True
        self.num_layers = num_layers
        self.act = act
        if num_layers < 1:
            raise ValueError('number of layers should be positive!')
        elif num_layers == 1:
            self.linear = nn.Linear(in_dim, out_dim)
        else:
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            self.batch_norms = torch.nn.ModuleList()
            self.linears.append(nn.Linear(in_dim, hid_dim))
            for _ in range(num_layers - 2):
                self.linears.append(nn.Linear(hid_dim, hid_dim))
            self.linears.append(nn.Linear(hid_dim, out_dim))
            for _ in range(num_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d(hid_dim))

    def forward(self, x):
        """
        Forward computation.

        Parameters
        ----------
        x : torch.Tensor
            Input embedding.

        Returns
        -------
        h : torch.Tensor
            Transformed embeddings.
        """
        if self.linear_or_not:
            return self.linear(x)
        else:
            h = x
            for layer in range(self.num_layers - 1):
                h = self.act(self.batch_norms[layer](self.linears[layer](h)))
            return self.linears[self.num_layers - 1](h)


class FNN_GAD_NR(nn.Module):
    """
    The personalized FNN module used by GAD_NR
    Source: https://github.com/Graph-COM/GAD-NR/blob/master/GAD-NR.ipynb
    
    Parameters
    ----------
    in_dim : int
        Input dimension of the embedding.
    hid_dim :  int
        Hidden dimension of model.
    out_dim : int
        Output dimension.
    num_layers : int
        Number of layers in the decoder.
    act : callable activation function or None, optional
        Activation function if not None.
        Default: ``torch.nn.functional.relu``.
    """

    def __init__(self, in_dim, hid_dim, out_dim, num_layers, act=torch.nn.functional.relu):
        super(FNN_GAD_NR, self).__init__()
        self.act = act
        self.linear1 = MLP_GAD_NR(in_dim, hid_dim, out_dim, num_layers)
        self.linear2 = nn.Linear(out_dim, out_dim)

    def forward(self, emb):
        """
        Forward computation.

        Parameters
        ----------
        emb : torch.Tensor
            Input embedding.

        Returns
        -------
        x_ : torch.Tensor
            Output embedding.
        """
        emb = self.linear1(emb)
        emb = self.linear2(self.act(emb))
        x_ = self.act(emb)
        return x_


def KL_neighbor_loss(predictions, targets, mask_len, device):
    """
    The local neighor distribution KL divergence loss used in GAD-NR.
    Source:
    https://github.com/Graph-COM/GAD-NR/blob/master/GAD-NR_inj_cora.ipynb
    """
    x1 = predictions.squeeze().cpu().detach()[:mask_len, :]
    x2 = targets.squeeze().cpu().detach()[:mask_len, :]
    mean_x1 = x1.mean(0)
    mean_x2 = x2.mean(0)
    nn = x1.shape[0]
    h_dim = x1.shape[1]
    cov_x1 = (x1 - mean_x1).transpose(1, 0).matmul(x1 - mean_x1) / max(nn - 1, 1)
    cov_x2 = (x2 - mean_x2).transpose(1, 0).matmul(x2 - mean_x2) / max(nn - 1, 1)
    eye = torch.eye(h_dim)
    cov_x1 = cov_x1 + eye
    cov_x2 = cov_x2 + eye
    KL_loss = 0.5 * (math.log(torch.det(cov_x1) / torch.det(cov_x2)) - h_dim + torch.trace(torch.inverse(cov_x2).matmul(cov_x1)) + (mean_x2 - mean_x1).reshape(1, -1).matmul(torch.inverse(cov_x2)).matmul(mean_x2 - mean_x1))
    KL_loss = KL_loss
    return KL_loss


class MLP_generator(nn.Module):
    """
    The personalized MLP module used by GAD_NR
    Source: https://github.com/Graph-COM/GAD-NR/blob/master/GAD-NR.ipynb
    
    Parameters
    ----------
    in_dim : int
        Input dimension of the embedding.
    out_dim : int
        Output dimension.
    act : callable activation function or None, optional
        Activation function if not None.
        Default: ``torch.nn.functional.relu``. 
    """

    def __init__(self, in_dim, out_dim, act=torch.nn.functional.relu):
        super(MLP_generator, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.linear2 = nn.Linear(out_dim, out_dim)
        self.linear3 = nn.Linear(out_dim, out_dim)
        self.linear4 = nn.Linear(out_dim, out_dim)
        self.act = act

    def forward(self, emb):
        """
        Forward computation.

        Parameters
        ----------
        emb : torch.Tensor
            Input embedding.

        Returns
        -------
        neighbor_emb : torch.Tensor
            Output neighbor embedding.
        """
        neighbor_emb = self.act(self.linear(emb))
        neighbor_emb = self.act(self.linear2(neighbor_emb))
        neighbor_emb = self.act(self.linear3(neighbor_emb))
        neighbor_emb = self.linear4(neighbor_emb)
        return neighbor_emb


def W2_neighbor_loss(predictions, targets, mask_len, device):
    """
    The local neighor distribution W2 loss used in GAD-NR.
    Source:
    https://github.com/Graph-COM/GAD-NR/blob/master/GAD-NR_inj_cora.ipynb
    """
    x1 = predictions.squeeze().cpu().detach()[:mask_len, :]
    x2 = targets.squeeze().cpu().detach()[:mask_len, :]
    mean_x1 = x1.mean(0)
    mean_x2 = x2.mean(0)
    nn = x1.shape[0]
    cov_x1 = (x1 - mean_x1).transpose(1, 0).matmul(x1 - mean_x1) / max(nn - 1, 1)
    cov_x2 = (x2 - mean_x2).transpose(1, 0).matmul(x2 - mean_x2) / max(nn - 1, 1)
    W2_loss = torch.square(mean_x1 - mean_x2).sum()
    +torch.trace(cov_x1 + cov_x2 + 2 * sqrtm(sqrtm(cov_x1) @ cov_x2.numpy() @ sqrtm(cov_x1)))
    W2_loss = W2_loss
    return W2_loss


class GUIDEBase(torch.nn.Module):
    """
    Higher-order Structure based Anomaly Detection on Attributed
    Networks

    GUIDE is an anomaly detector consisting of an attribute graph
    convolutional autoencoder, and a structure graph attentive
    autoencoder (not the same as the graph attention networks). Instead
    of the adjacency matrix, node motif degree is used as input of
    structure autoencoder. The reconstruction mean square error of the
    autoencoders are defined as structure anomaly score and attribute
    anomaly score, respectively.

    Note: The calculation of node motif degree in preprocessing has
    high time complexity. It may take longer than you expect.

    See :cite:`yuan2021higher` for details.

    Parameters
    ----------
    dim_a : int
        Input dimension for attribute.
    dim_s : int
        Input dimension for structure.
    hid_a : int, optional
        Hidden dimension for attribute. Default: ``64``.
    hid_s : int, optional
        Hidden dimension for structure. Default: ``4``.
    num_layers : int, optional
        Total number of layers in model. Default: ``4``.
    dropout : float, optional
        Dropout rate. Default: ``0.``.
    act : callable activation function or None, optional
        Activation function if not None.
        Default: ``torch.nn.functional.relu``.
    **kwargs
        Other parameters for GCN.
    """

    def __init__(self, dim_a, dim_s, hid_a=64, hid_s=4, num_layers=4, dropout=0.0, act=torch.nn.functional.relu, **kwargs):
        super(GUIDEBase, self).__init__()
        assert num_layers >= 2, 'Number of layers must be greater than or equal to 2.'
        encoder_layers = math.floor(num_layers / 2)
        decoder_layers = math.ceil(num_layers / 2)
        self.attr_encoder = GCN(in_channels=dim_a, hidden_channels=hid_a, num_layers=encoder_layers, out_channels=hid_a, dropout=dropout, act=act, **kwargs)
        self.attr_decoder = GCN(in_channels=hid_a, hidden_channels=hid_a, num_layers=decoder_layers, out_channels=dim_a, dropout=dropout, act=act, **kwargs)
        self.stru_encoder = GNA(in_channels=dim_s, hidden_channels=hid_s, num_layers=encoder_layers, out_channels=hid_s, dropout=dropout, act=act)
        self.stru_decoder = GNA(in_channels=hid_s, hidden_channels=hid_s, num_layers=decoder_layers, out_channels=dim_s, dropout=dropout, act=act)
        self.loss_func = double_recon_loss
        self.emb = None

    def forward(self, x, s, edge_index):
        """
        Forward computation of GUIDE.

        Parameters
        ----------
        x : torch.Tensor
            Input attribute embeddings.
        s : torch.Tensor
            Input structure embeddings.
        edge_index : torch.Tensor
            Edge index.

        Returns
        -------
        x_ : torch.Tensor
            Reconstructed attribute embeddings.
        s_ : torch.Tensor
            Reconstructed structure embeddings.
        """
        h_x = self.attr_encoder(x, edge_index)
        x_ = self.attr_decoder(h_x, edge_index)
        h_s = self.stru_encoder(s, edge_index)
        s_ = self.stru_decoder(h_s, edge_index)
        self.emb = h_x, h_s
        return x_, s_

    @staticmethod
    def calc_gdd(data, cache_dir=None, graphlet_size=4, selected_motif=True):
        """
        Calculation of Node Motif Degree / Graphlet Degree
        Distribution. Part of this function is adapted
        from https://github.com/benedekrozemberczki/OrbitalFeatures.

        Parameters
        ----------
        data : PyTorch Geometric Data instance (torch_geometric.data.Data)
            The input data.
        cache_dir : str
            The directory for the node motif degree caching.
        graphlet_size : int, optional
            The maximum size of the graphlet. Default: 4.
        selected_motif : bool, optional
            Whether to use the selected motif or not. Default: True.

        Returns
        -------
        s : torch.Tensor
            Structure matrix (node motif degree/graphlet degree)
        """
        if cache_dir is None:
            cache_dir = os.path.join(os.path.expanduser('~'), '.pygod')
        hash_func = hashlib.sha1()
        hash_func.update(str(data).encode('utf-8'))
        file_name = 'nmd_' + str(hash_func.hexdigest()[:8]) + str(graphlet_size) + str(selected_motif)[0] + '.pt'
        file_path = os.path.join(cache_dir, file_name)
        if os.path.exists(file_path):
            s = torch.load(file_path)
        else:
            edge_index = data.edge_index
            g = nx.from_edgelist(edge_index.T.tolist())
            edge_subsets = dict()
            subsets = [[edge[0], edge[1]] for edge in g.edges()]
            edge_subsets[2] = subsets
            unique_subsets = dict()
            for i in range(3, graphlet_size + 1):
                for subset in subsets:
                    for node in subset:
                        for neb in g.neighbors(node):
                            new_subset = subset + [neb]
                            if len(set(new_subset)) == i:
                                new_subset.sort()
                                unique_subsets[tuple(new_subset)] = 1
                subsets = [list(k) for k, v in unique_subsets.items()]
                edge_subsets[i] = subsets
                unique_subsets = dict()
            graphs = graph_atlas_g()
            interesting_graphs = {i: [] for i in range(2, graphlet_size + 1)}
            for graph in graphs:
                if 1 < graph.number_of_nodes() < graphlet_size + 1:
                    if nx.is_connected(graph):
                        interesting_graphs[graph.number_of_nodes()].append(graph)
            main_index = 0
            categories = dict()
            for size, graphs in interesting_graphs.items():
                categories[size] = dict()
                for index, graph in enumerate(graphs):
                    categories[size][index] = dict()
                    degrees = list(set([graph.degree(node) for node in graph.nodes()]))
                    for degree in degrees:
                        categories[size][index][degree] = main_index
                        main_index += 1
            unique_motif_count = main_index
            features = {node: {i: (0) for i in range(unique_motif_count)} for node in g.nodes()}
            for size, node_lists in edge_subsets.items():
                graphs = interesting_graphs[size]
                for nodes in node_lists:
                    sub_gr = g.subgraph(nodes)
                    for index, graph in enumerate(graphs):
                        if nx.is_isomorphic(sub_gr, graph):
                            for node in sub_gr.nodes():
                                features[node][categories[size][index][sub_gr.degree(node)]] += 1
                            break
            motifs = [([n] + [features[n][i] for i in range(unique_motif_count)]) for n in g.nodes()]
            motifs = torch.Tensor(motifs)
            motifs = motifs[torch.sort(motifs[:, 0]).indices, 1:]
            if selected_motif:
                s = torch.zeros((data.x.shape[0], 6))
                s[:, 0] = motifs[:, 3]
                s[:, 1] = motifs[:, 1] + motifs[:, 2]
                s[:, 2] = motifs[:, 14]
                s[:, 3] = motifs[:, 12] + motifs[:, 13]
                s[:, 4] = motifs[:, 11]
                s[:, 5] = motifs[:, 0]
            else:
                s = motifs
            if not os.path.exists(cache_dir):
                os.makedirs(cache_dir)
            torch.save(s, file_path)
        return s


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (FNN_GAD_NR,
     lambda: ([], {'in_dim': 4, 'hid_dim': 4, 'out_dim': 4, 'num_layers': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (MLP_GAD_NR,
     lambda: ([], {'in_dim': 4, 'hid_dim': 4, 'out_dim': 4, 'num_layers': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (MLP_generator,
     lambda: ([], {'in_dim': 4, 'out_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
]

