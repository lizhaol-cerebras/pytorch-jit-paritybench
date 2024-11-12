
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


from torch.types import Device


import math


from typing import Iterable


from typing import Iterator


from typing import Optional


from typing import Sequence


import numpy as np


import pandas as pd


from collections import UserDict


from typing import Mapping


from abc import ABC


from abc import abstractmethod


from functools import lru_cache


from itertools import chain


from typing import ClassVar


from typing import Dict


from typing import Tuple


from typing import cast


from typing import Union


from typing import TypeVar


from torch.nn.modules.loss import _Loss


from torch import nn


from typing import List


from torch.nn.functional import normalize


from torch.fft import fft


from torch.fft import ifft


from torch.nn import functional as F


import functools


import torch.nn as nn


from typing import Any


import torch.nn.functional


from torch.nn import LayerNorm


from torch.nn.modules.container import ModuleList


import collections.abc


import time


from typing import Type


from sklearn.metrics import mean_absolute_error


from sklearn.metrics import mean_squared_error


from sklearn.metrics import roc_auc_score


from torch.optim.optimizer import Optimizer


import logging


import torch.cuda


import inspect


class CASTERSupervisedLoss(_Loss):
    """An implementation of the custom loss function for the supervised learning stage of the CASTER algorithm.

    The algorithm is described in [huang2020]_. The loss function combines three separate loss functions on
    different model outputs: class prediction loss, input reconstruction loss, and dictionary projection loss.

    .. [huang2020] Huang, K., *et al.* (2020). `CASTER: Predicting drug interactions
       with chemical substructure representation <https://doi.org/10.1609/aaai.v34i01.5412>`_.
       *AAAI 2020 - 34th AAAI Conference on Artificial Intelligence*, 702–709.
    """

    def __init__(self, recon_loss_coeff: 'float'=0.1, proj_coeff: 'float'=0.1, lambda1: 'float'=0.01, lambda2: 'float'=0.1):
        """
        Initialize the custom loss function for the supervised learning stage of the CASTER algorithm.

        :param recon_loss_coeff: coefficient for the reconstruction loss
        :param proj_coeff: coefficient for the projection loss
        :param lambda1: regularization coefficient for the projection loss
        :param lambda2: regularization coefficient for the augmented projection loss
        """
        super().__init__(reduction='none')
        self.recon_loss_coeff = recon_loss_coeff
        self.proj_coeff = proj_coeff
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.loss = torch.nn.BCELoss()

    def forward(self, x: 'Tuple[torch.FloatTensor, ...]', target: 'torch.Tensor') ->torch.FloatTensor:
        """Perform a forward pass of the loss calculation for the supervised learning stage of the CASTER algorithm.

        :param x: a tuple of tensors returned by the model forward pass (see CASTER.forward() method)
        :param target: target labels
        :return: combined loss value
        """
        score, recon, code, dictionary_features_latent, drug_pair_features_latent, drug_pair_features = x
        batch_size, _ = drug_pair_features.shape
        loss_prediction = self.loss(score, target.float())
        loss_reconstruction = self.recon_loss_coeff * self.loss(recon, drug_pair_features)
        loss_projection = self.proj_coeff * (torch.norm(drug_pair_features_latent - torch.matmul(code, dictionary_features_latent)) + self.lambda1 * torch.sum(torch.abs(code)) / batch_size + self.lambda2 * torch.norm(dictionary_features_latent, p='fro') / batch_size)
        loss = loss_prediction + loss_reconstruction + loss_projection
        return loss


class Model(nn.Module, ABC):
    """The base class for ChemicalX models."""

    @abstractmethod
    def unpack(self, batch: 'DrugPairBatch'):
        """Unpack a batch into a tuple of the features needed for forward.

        :param batch: A batch object
        :returns: A tuple that will be used as the positional arguments
            in this model's :func:`forward` method.
        """


class CASTER(Model):
    """An implementation of the CASTER model from [huang2020]_.

    .. seealso:: This model was suggested in https://github.com/AstraZeneca/chemicalx/issues/17

    .. [huang2020] Huang, K., *et al.* (2020). `CASTER: Predicting drug interactions
       with chemical substructure representation <https://doi.org/10.1609/aaai.v34i01.5412>`_.
       *AAAI 2020 - 34th AAAI Conference on Artificial Intelligence*, 702–709.
    """

    def __init__(self, *, drug_channels: int, encoder_hidden_channels: int=32, encoder_output_channels: int=32, decoder_hidden_channels: int=32, hidden_channels: int=32, out_hidden_channels: int=32, out_channels: int=1, lambda3: float=1e-05, magnifying_factor: int=100):
        """Instantiate the CASTER model.

        :param drug_channels: The number of drug features (recognised frequent substructures).
            The original implementation recognised 1722 basis substructures in the BIOSNAP experiment.
        :param encoder_hidden_channels: The number of hidden layer neurons in the encoder module.
        :param encoder_output_channels: The number of output layer neurons in the encoder module.
        :param decoder_hidden_channels: The number of hidden layer neurons in the decoder module.
        :param hidden_channels: The number of hidden layer neurons in the predictor module.
        :param out_hidden_channels: The last hidden layer channels before output.
        :param out_channels: The number of output channels.
        :param lambda3: regularisation coefficient in the dictionary encoder module.
        :param magnifying_factor: The magnifying factor coefficient applied to the predictor module input.
        """
        super().__init__()
        self.lambda3 = lambda3
        self.magnifying_factor = magnifying_factor
        self.drug_channels = drug_channels
        self.encoder = torch.nn.Sequential(torch.nn.Linear(self.drug_channels, encoder_hidden_channels), torch.nn.ReLU(True), torch.nn.Linear(encoder_hidden_channels, encoder_output_channels))
        self.decoder = torch.nn.Sequential(torch.nn.Linear(encoder_output_channels, decoder_hidden_channels), torch.nn.ReLU(True), torch.nn.Linear(decoder_hidden_channels, drug_channels))
        predictor_layers = []
        predictor_layers.append(torch.nn.Linear(self.drug_channels, hidden_channels))
        predictor_layers.append(torch.nn.ReLU(True))
        for i in range(1, 6):
            predictor_layers.append(torch.nn.BatchNorm1d(hidden_channels))
            if i < 5:
                predictor_layers.append(torch.nn.Linear(hidden_channels, hidden_channels))
            else:
                predictor_layers.append(torch.nn.Linear(hidden_channels, out_hidden_channels))
            predictor_layers.append(torch.nn.ReLU(True))
        predictor_layers.append(torch.nn.Linear(out_hidden_channels, out_channels))
        predictor_layers.append(torch.nn.Sigmoid())
        self.predictor = torch.nn.Sequential(*predictor_layers)

    def unpack(self, batch: 'DrugPairBatch') ->Tuple[torch.FloatTensor]:
        """Return the "functional representation" of drug pairs, as defined in the original implementation.

        :param batch: batch of drug pairs
        :return: each pair is represented as a single vector with x^i = 1 if either x_1^i >= 1 or x_2^i >= 1
        """
        pair_representation = (torch.maximum(batch.drug_features_left, batch.drug_features_right) >= 1.0).float()
        return pair_representation,

    def dictionary_encoder(self, drug_pair_features_latent: 'torch.FloatTensor', dictionary_features_latent: 'torch.FloatTensor') ->torch.FloatTensor:
        """Perform a forward pass of the dictionary encoder submodule.

        :param drug_pair_features_latent: encoder output for the input drug_pair_features
            (batch_size x encoder_output_channels)
        :param dictionary_features_latent: projection of the drug_pair_features using the dictionary basis
            (encoder_output_channels x drug_channels)
        :return: sparse code X_o: (batch_size x drug_channels)
        """
        dict_feat_squared = torch.matmul(dictionary_features_latent, dictionary_features_latent.transpose(2, 1))
        dict_feat_squared_inv = torch.inverse(dict_feat_squared + self.lambda3 * torch.eye(self.drug_channels))
        dict_feat_closed_form = torch.matmul(dict_feat_squared_inv, dictionary_features_latent)
        r = drug_pair_features_latent[:, None, :].matmul(dict_feat_closed_form.transpose(2, 1)).squeeze(1)
        return r

    def forward(self, drug_pair_features: 'torch.FloatTensor') ->Tuple[torch.FloatTensor, ...]:
        """Run a forward pass of the CASTER model.

        :param drug_pair_features: functional representation of each drug pair (see unpack method)
        :return: (Tuple[torch.FloatTensor): a tuple of tensors including:
                prediction_scores: predicted target scores for each drug pair
                reconstructed: input drug pair vectors reconstructed by the encoder-decoder chain
                dictionary_encoded: drug pair features encoded by the dictionary encoder submodule
                dictionary_features_latent: projection of the encoded drug pair features using the dictionary basis
                drug_pair_features_latent: encoder output for the input drug_pair_features
                drug_pair_features: a copy of the input unpacked drug_pair_features (needed for loss calculation)
        """
        drug_pair_features_latent = self.encoder(drug_pair_features)
        dictionary_features_latent = self.encoder(torch.eye(self.drug_channels))
        dictionary_features_latent = dictionary_features_latent.mul(drug_pair_features[:, :, None])
        drug_pair_features_reconstructed = self.decoder(drug_pair_features_latent)
        reconstructed = torch.sigmoid(drug_pair_features_reconstructed)
        dictionary_encoded = self.dictionary_encoder(drug_pair_features_latent, dictionary_features_latent)
        prediction_scores = self.predictor(self.magnifying_factor * dictionary_encoded)
        return prediction_scores, reconstructed, dictionary_encoded, dictionary_features_latent, drug_pair_features_latent, drug_pair_features


class DeepDDI(Model):
    """An implementation of the DeepDDI model from [ryu2018]_.

    .. seealso:: This model was suggested in https://github.com/AstraZeneca/chemicalx/issues/2

    .. [ryu2018] Ryu, J. Y., *et al.* (2018). `Deep learning improves prediction
       of drug–drug and drug–food interactions <https://doi.org/10.1073/pnas.1803294115>`_.
       *Proceedings of the National Academy of Sciences*, 115(18), E4304–E4311.
    """

    def __init__(self, *, drug_channels: int, hidden_channels: int=2048, hidden_layers_num: int=9, out_channels: int=1):
        """Instantiate the DeepDDI model.

        :param drug_channels: The number of drug features.
        :param hidden_channels: The number of hidden layer neurons.
        :param hidden_layers_num: The number of hidden layers.
        :param out_channels: The number of output channels.
        """
        super().__init__()
        assert hidden_layers_num > 1
        layers = [nn.Linear(drug_channels * 2, hidden_channels), nn.ReLU(), nn.BatchNorm1d(num_features=hidden_channels, affine=True, momentum=None), nn.ReLU()]
        for _ in range(hidden_layers_num - 1):
            layers.extend([nn.Linear(hidden_channels, hidden_channels), nn.ReLU(), nn.BatchNorm1d(num_features=hidden_channels, affine=True, momentum=None), nn.ReLU()])
        layers.extend([nn.Linear(hidden_channels, out_channels), nn.Sigmoid()])
        self.final = nn.Sequential(*layers)

    def unpack(self, batch: 'DrugPairBatch'):
        """Return the context features, left drug features and right drug features."""
        return batch.drug_features_left, batch.drug_features_right

    def _combine_sides(self, left: 'torch.FloatTensor', right: 'torch.FloatTensor') ->torch.FloatTensor:
        return torch.cat([left, right], dim=1)

    def forward(self, drug_features_left: 'torch.FloatTensor', drug_features_right: 'torch.FloatTensor') ->torch.FloatTensor:
        """Run a forward pass of the DeepDDI model.

        :param drug_features_left: A matrix of head drug features.
        :param drug_features_right: A matrix of tail drug features.
        :returns: A column vector of predicted interaction scores.
        """
        hidden = self._combine_sides(drug_features_left, drug_features_right)
        return self.final(hidden)


class DeepSynergy(Model):
    """An implementation of the DeepSynergy model from [preuer2018]_.

    .. seealso:: This model was suggested in https://github.com/AstraZeneca/chemicalx/issues/16

    .. [preuer2018] Preuer, K., *et al.* (2018). `DeepSynergy: predicting anti-cancer drug synergy
       with Deep Learning <https://doi.org/10.1093/bioinformatics/btx806>`_. *Bioinformatics*, 34(9), 1538–1546.
    """

    def __init__(self, *, context_channels: int, drug_channels: int, input_hidden_channels: int=32, middle_hidden_channels: int=32, final_hidden_channels: int=32, out_channels: int=1, dropout_rate: float=0.5):
        """Instantiate the DeepSynergy model.

        :param context_channels: The number of context features.
        :param drug_channels: The number of drug features.
        :param input_hidden_channels: The number of hidden layer neurons in the input layer.
        :param middle_hidden_channels: The number of hidden layer neurons in the middle layer.
        :param final_hidden_channels: The number of hidden layer neurons in the final layer.
        :param out_channels: The number of output channels.
        :param dropout_rate: The rate of dropout before the scoring head is used.
        """
        super().__init__()
        self.final = nn.Sequential(nn.Linear(drug_channels + drug_channels + context_channels, input_hidden_channels), nn.ReLU(), nn.Linear(input_hidden_channels, middle_hidden_channels), nn.ReLU(), nn.Linear(middle_hidden_channels, final_hidden_channels), nn.ReLU(), nn.Dropout(dropout_rate), nn.Linear(final_hidden_channels, out_channels), nn.Sigmoid())

    def unpack(self, batch: 'DrugPairBatch'):
        """Return the context features, left drug features, and right drug features."""
        return batch.context_features, batch.drug_features_left, batch.drug_features_right

    def forward(self, context_features: 'torch.FloatTensor', drug_features_left: 'torch.FloatTensor', drug_features_right: 'torch.FloatTensor') ->torch.FloatTensor:
        """Run a forward pass of the DeepSynergy model.

        :param context_features: A matrix of biological context features.
        :param drug_features_left: A matrix of head drug features.
        :param drug_features_right: A matrix of tail drug features.
        :returns: A column vector of predicted synergy scores.
        """
        hidden = torch.cat([context_features, drug_features_left, drug_features_right], dim=1)
        return self.final(hidden)


class Highway(nn.Module):
    """The Highway update layer from [srivastava2015]_.

    .. [srivastava2015] Srivastava, R. K., *et al.* (2015).
       `Highway Networks <http://arxiv.org/abs/1505.00387>`_.
       *arXiv*, 1505.00387.
    """

    def __init__(self, input_size: 'int', prev_input_size: 'int'):
        """Instantiate the Highway update layer.

        :param input_size: Current representation size.
        :param prev_input_size: Size of the representation obtained by the previous convolutional layer.
        """
        super().__init__()
        total_size = input_size + prev_input_size
        self.proj = nn.Linear(total_size, input_size)
        self.transform = nn.Linear(total_size, input_size)
        self.transform.bias.data.fill_(-2.0)

    def forward(self, current: 'torch.Tensor', previous: 'torch.Tensor') ->torch.Tensor:
        """Compute the gated update.

        :param current: Current layer node representations.
        :param previous: Previous layer node representations.
        :returns: The highway-updated inputs.
        """
        concat_inputs = torch.cat((current, previous), 1)
        proj_result = F.relu(self.proj(concat_inputs))
        proj_gate = F.sigmoid(self.transform(concat_inputs))
        gated = proj_gate * proj_result + (1 - proj_gate) * current
        return gated


class AttentionPooling(nn.Module):
    """The attention pooling layer from [chen2020]_."""

    def __init__(self, molecule_channels: 'int', hidden_channels: 'int'):
        """Instantiate the attention pooling layer.

        :param molecule_channels: Input node features.
        :param hidden_channels: Final node representation.
        """
        super(AttentionPooling, self).__init__()
        total_features_channels = molecule_channels + hidden_channels
        self.lin = nn.Linear(total_features_channels, hidden_channels)
        self.last_rep = nn.Linear(hidden_channels, hidden_channels)

    def forward(self, input_rep: 'torch.Tensor', final_rep: 'torch.Tensor', graph_index: 'torch.Tensor') ->torch.Tensor:
        """
        Compute an attention-based readout using the input and output layers of the RGCN encoder for one molecule.

        :param input_rep: Input nodes representations.
        :param final_rep: Final nodes representations.
        :param graph_index: Node to graph readout index.
        :returns: Graph-level representation.
        """
        att = torch.sigmoid(self.lin(torch.cat((input_rep, final_rep), dim=1)))
        g = att.mul(self.last_rep(final_rep))
        g = scatter_add(g, graph_index, dim=0)
        return g


def circular_correlation(left: 'torch.FloatTensor', right: 'torch.FloatTensor') ->torch.FloatTensor:
    """Compute the circular correlation of two vectors ``left`` and ``right`` via their Fast Fourier Transforms.

    :param left: the left vector
    :param right: the right vector
    :returns: Joint representation by circular correlation.
    """
    left_x_cfft = torch.conj(fft(left))
    right_x_fft = fft(right)
    circ_corr = ifft(torch.mul(left_x_cfft, right_x_fft))
    return circ_corr.real


class MatchMaker(Model):
    """An implementation of the MatchMaker model from [kuru2021]_.

    .. seealso:: This model was suggested in https://github.com/AstraZeneca/chemicalx/issues/23

    .. [kuru2021] Kuru, H. I., *et al.* (2021). `MatchMaker: A Deep Learning Framework
       for Drug Synergy Prediction <https://doi.org/10.1109/TCBB.2021.3086702>`_.
       *IEEE/ACM Transactions on Computational Biology and Bioinformatics*, 1–1.
    """

    def __init__(self, *, context_channels: int, drug_channels: int, input_hidden_channels: int=32, middle_hidden_channels: int=32, final_hidden_channels: int=32, out_channels: int=1, dropout_rate: float=0.5):
        """Instantiate the MatchMaker model.

        :param context_channels: The number of context features.
        :param drug_channels: The number of drug features.
        :param input_hidden_channels: The number of hidden layer neurons in the input layer.
        :param middle_hidden_channels: The number of hidden layer neurons in the middle layer.
        :param final_hidden_channels: The number of hidden layer neurons in the final layer.
        :param out_channels: The number of output channels.
        :param dropout_rate: The rate of dropout before the scoring head is used.
        """
        super().__init__()
        self.drug_context_layer = nn.Sequential(nn.Linear(drug_channels + context_channels, input_hidden_channels), nn.ReLU(), nn.Dropout(dropout_rate), nn.Linear(input_hidden_channels, middle_hidden_channels), nn.ReLU(), nn.Dropout(dropout_rate), nn.Linear(middle_hidden_channels, middle_hidden_channels))
        self.final = nn.Sequential(nn.Linear(2 * middle_hidden_channels, final_hidden_channels), nn.ReLU(), nn.Dropout(dropout_rate), nn.Linear(final_hidden_channels, out_channels), nn.Sigmoid())

    def unpack(self, batch: 'DrugPairBatch'):
        """Return the context features, left drug features, and right drug features."""
        return batch.context_features, batch.drug_features_left, batch.drug_features_right

    def _combine_sides(self, left: 'torch.FloatTensor', right: 'torch.FloatTensor') ->torch.FloatTensor:
        return torch.cat([left, right], dim=1)

    def forward(self, context_features: 'torch.FloatTensor', drug_features_left: 'torch.FloatTensor', drug_features_right: 'torch.FloatTensor') ->torch.FloatTensor:
        """Run a forward pass of the MatchMaker model.

        :param context_features: A matrix of biological context features.
        :param drug_features_left: A matrix of head drug features.
        :param drug_features_right: A matrix of tail drug features.
        :returns: A column vector of predicted synergy scores.
        """
        hidden_left = torch.cat([context_features, drug_features_left], dim=1)
        hidden_left = self.drug_context_layer(hidden_left)
        hidden_right = torch.cat([context_features, drug_features_right], dim=1)
        hidden_right = self.drug_context_layer(hidden_right)
        hidden = self._combine_sides(hidden_left, hidden_right)
        return self.final(hidden)


class MessagePassing(nn.Module):
    """A network for creating node representations based on internal message passing."""

    def __init__(self, node_channels: 'int', edge_channels: 'int', hidden_channels: 'int', dropout: 'float'=0.5):
        """Instantiate the MessagePassing network.

        :param node_channels: Dimension of node features
        :param edge_channels: Dimension of edge features
        :param hidden_channels: Dimension of hidden layer
        :param dropout: Dropout probability
        """
        super().__init__()
        self.node_projection = nn.Sequential(nn.Linear(node_channels, hidden_channels, bias=False), nn.Dropout(dropout))
        self.edge_projection = nn.Sequential(nn.Linear(edge_channels, hidden_channels), nn.LeakyReLU(), nn.Dropout(dropout), nn.Linear(hidden_channels, hidden_channels), nn.LeakyReLU(), nn.Dropout(dropout))

    def forward(self, nodes: 'torch.FloatTensor', edges: 'torch.FloatTensor', segmentation_index: 'torch.LongTensor', index: 'torch.LongTensor') ->torch.FloatTensor:
        """Calculate forward pass of message passing network.

        :param nodes: Node feature matrix.
        :param edges: Edge feature matrix.
        :param segmentation_index: List of node indices from where edges in the molecular graph start.
        :param index: List of node indices from where edges in the molecular graph end.
        :returns: Messages between nodes.
        """
        edges = self.edge_projection(edges)
        messages = self.node_projection(nodes)
        messages = self.message_composing(messages, edges, index)
        messages = self.message_aggregation(nodes, messages, segmentation_index)
        return messages

    def message_composing(self, messages: 'torch.FloatTensor', edges: 'torch.FloatTensor', index: 'torch.LongTensor') ->torch.FloatTensor:
        """Compose message based by elementwise multiplication of edge and node projections.

        :param messages: Message matrix.
        :param edges: Edge feature matrix.
        :param index: Global node indexing.
        :returns: Composed messages.
        """
        messages = messages.index_select(0, index)
        messages = messages * edges
        return messages

    def message_aggregation(self, nodes: 'torch.FloatTensor', messages: 'torch.FloatTensor', segmentation_index: 'torch.LongTensor') ->torch.FloatTensor:
        """Aggregate the messages.

        :param nodes: Node feature matrix.
        :param messages: Message feature matrix.
        :param segmentation_index: List of node indices from where edges in the molecular graph start.
        :returns: Messages between nodes.
        """
        messages = torch.zeros_like(nodes).index_add(0, segmentation_index, messages)
        return messages


def segment_max(logit: 'torch.FloatTensor', number_of_segments: 'torch.LongTensor', segmentation_index: 'torch.LongTensor', index: 'torch.LongTensor'):
    """Segmentation maximal index finder.

    :param logit: Logit vector.
    :param number_of_segments: Segment numbers.
    :param segmentation_index: Index of segments.
    :param index: Global index
    :returns: Largest index in each segmentation.
    """
    max_number_of_segments = index.max().item() + 1
    segmentation_max = logit.new_full((number_of_segments, max_number_of_segments), -np.inf)
    segmentation_max = segmentation_max.index_put_((segmentation_index, index), logit).max(dim=1)[0]
    return segmentation_max[segmentation_index]


def segment_sum(logit: 'torch.FloatTensor', number_of_segments: 'torch.LongTensor', segmentation_index: 'torch.LongTensor'):
    """Segmentation sum calculation.

    :param logit: Logit vector.
    :param number_of_segments: Segment numbers.
    :param segmentation_index: Index of segments.
    :returns: Sum of logits on segments.
    """
    norm = logit.new_zeros(number_of_segments).index_add(0, segmentation_index, logit)
    return norm[segmentation_index]


def segment_softmax(logit: 'torch.FloatTensor', number_of_segments: 'torch.LongTensor', segmentation_index: 'torch.LongTensor', index: 'torch.LongTensor', temperature: 'torch.FloatTensor'):
    """Segmentation softmax calculation.

    :param logit: Logit vector.
    :param number_of_segments: Segment numbers.
    :param segmentation_index: Index of segmentation.
    :param index: Global index.
    :param temperature: Normalization values.
    :returns: Probability scores for attention.
    """
    logit_max = segment_max(logit, number_of_segments, segmentation_index, index).detach()
    logit = torch.exp((logit - logit_max) / temperature)
    logit_norm = segment_sum(logit, number_of_segments, segmentation_index)
    prob = logit / (logit_norm + torch.finfo(logit_norm.dtype).eps)
    return prob


class CoAttention(nn.Module):
    """The co-attention network for MHCADDI model."""

    def __init__(self, input_channels: 'int', output_channels: 'int', dropout: 'float'=0.1):
        """Instantiate the co-attention network.

        :param input_channels: The number of atom features.
        :param output_channels: The number of output features.
        :param dropout: Dropout probability.
        """
        super().__init__()
        self.temperature = np.sqrt(input_channels)
        self.key_projection = nn.Linear(input_channels, input_channels, bias=False)
        self.value_projection = nn.Linear(input_channels, input_channels, bias=False)
        nn.init.xavier_normal_(self.key_projection.weight)
        nn.init.xavier_normal_(self.value_projection.weight)
        self.attention_dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=1)
        self.out_projection = nn.Sequential(nn.Linear(input_channels, output_channels), nn.LeakyReLU(), nn.Dropout(dropout))

    def _calculate_message(self, translation: 'torch.Tensor', segmentation_number: 'torch.Tensor', segmentation_index: 'torch.Tensor', index: 'torch.Tensor', node: 'torch.Tensor', node_hidden_channels: 'torch.Tensor', node_neighbor: 'torch.Tensor'):
        """Calculate the outer message."""
        node_edge = self.attention_dropout(segment_softmax(translation, segmentation_number, segmentation_index, index, self.temperature))
        node_edge = node_edge.view(-1, 1)
        message = node.new_zeros((segmentation_number, node_hidden_channels)).index_add(0, segmentation_index, node_edge * node_neighbor)
        message_graph = self.out_projection(message)
        return message_graph

    def forward(self, node_left: 'torch.FloatTensor', segmentation_index_left: 'torch.LongTensor', index_left: 'torch.LongTensor', node_right: 'torch.FloatTensor', segmentation_index_right: 'torch.LongTensor', index_right: 'torch.LongTensor'):
        """Forward pass with the segmentation indices and node features.

        :param node_left: Left side node features.
        :param segmentation_index_left: Left side segmentation index.
        :param index_left: Left side indices.
        :param node_right: Right side node features.
        :param segmentation_index_right: Right side segmentation index.
        :param index_right: Right side indices.
        :returns: Left and right side messages and edge indices.
        """
        node_left_hidden_channels = node_left.size(1)
        node_right_hidden_channels = node_right.size(1)
        segmentation_number_left = node_left.size(0)
        segmentation_number_right = node_right.size(0)
        node_left_center = self.key_projection(node_left).index_select(0, segmentation_index_left)
        node_right_center = self.key_projection(node_right).index_select(0, segmentation_index_right)
        node_left_neighbor = self.value_projection(node_right).index_select(0, segmentation_index_right)
        node_right_neighbor = self.value_projection(node_left).index_select(0, segmentation_index_left)
        translation = (node_left_center * node_right_center).sum(1)
        message_graph_left = self._calculate_message(translation, segmentation_number_left, segmentation_index_left, index_left, node_left, node_left_hidden_channels, node_left_neighbor)
        message_graph_right = self._calculate_message(translation, segmentation_number_right, segmentation_index_right, index_right, node_right, node_right_hidden_channels, node_right_neighbor)
        return message_graph_left, message_graph_right


class CoAttentionMessagePassingNetwork(nn.Module):
    """Coattention message passing layer."""

    def __init__(self, hidden_channels: 'int', readout_channels: 'int', dropout: 'float'=0.5):
        """Initialize a co-attention message passing network.

        :param hidden_channels: Input channel number.
        :param readout_channels: Readout channel number.
        :param dropout: Rate of dropout.
        """
        super().__init__()
        self.message_passing = MessagePassing(node_channels=hidden_channels, edge_channels=hidden_channels, hidden_channels=hidden_channels, dropout=dropout)
        self.co_attention = CoAttention(input_channels=hidden_channels, output_channels=hidden_channels, dropout=dropout)
        self.linear = nn.LayerNorm(hidden_channels)
        self.leaky_relu = nn.LeakyReLU()
        self.prediction_readout_projection = nn.Linear(hidden_channels, readout_channels)

    def _get_graph_features(self, atom_features: 'torch.Tensor', inner_message: 'torch.Tensor', outer_message: 'torch.Tensor', segmentation_molecule: 'torch.Tensor'):
        """Get the graph representations."""
        message = atom_features + inner_message + outer_message
        message = self.linear(message)
        graph_features = self.readout(message, segmentation_molecule)
        return graph_features

    def forward(self, segmentation_molecule_left: 'torch.Tensor', atom_left: 'torch.Tensor', bond_left: 'torch.Tensor', inner_segmentation_index_left: 'torch.Tensor', inner_index_left: 'torch.Tensor', outer_segmentation_index_left: 'torch.Tensor', outer_index_left: 'torch.Tensor', segmentation_molecule_right: 'torch.Tensor', atom_right: 'torch.Tensor', bond_right: 'torch.Tensor', inner_segmentation_index_right: 'torch.Tensor', inner_index_right: 'torch.Tensor', outer_segmentation_index_right: 'torch.Tensor', outer_index_right: 'torch.Tensor'):
        """Make a forward pass with the data.

        :param segmentation_molecule_left: Mapping from node id to graph id for the left drugs.
        :param atom_left: Atom features on the left-hand side.
        :param bond_left: Bond features on the left-hand side.
        :param inner_segmentation_index_left: Heads of edges connecting atoms within the left drug molecules.
        :param inner_index_left: Tails of edges connecting atoms within the left drug molecules.
        :param outer_segmentation_index_left: Heads of edges connecting atoms between left and right drug molecules
        :param outer_index_left: Tails of edges connecting atoms between left and right drug molecules.
        :param segmentation_molecule_right:  Mapping from node id to graph id for the right drugs.
        :param atom_right: Atom features on the right-hand side.
        :param bond_right: Bond features on the right-hand side.
        :param inner_segmentation_index_right: Heads of edges connecting atoms within the right drug molecules.
        :param inner_index_right: Tails of edges connecting atoms within the right drug molecules.
        :param outer_segmentation_index_right: Heads of edges connecting atoms between right and left drug molecules
        :param outer_index_right: Heads of edges connecting atoms between right and left drug molecules
        :returns: Graph level representations.
        """
        outer_message_left, outer_message_right = self.co_attention(atom_left, outer_segmentation_index_left, outer_index_left, atom_right, outer_segmentation_index_right, outer_index_right)
        inner_message_left = self.message_passing(atom_left, bond_left, inner_segmentation_index_left, inner_index_left)
        inner_message_right = self.message_passing(atom_right, bond_right, inner_segmentation_index_right, inner_index_right)
        graph_left = self._get_graph_features(atom_left, inner_message_left, outer_message_left, segmentation_molecule_left)
        graph_right = self._get_graph_features(atom_right, inner_message_right, outer_message_right, segmentation_molecule_right)
        return graph_left, graph_right

    def readout(self, atom_features: 'torch.Tensor', segmentation_molecule: 'torch.Tensor'):
        """Aggregate node features.

        :param atom_features: Atom embeddings.
        :param segmentation_molecule: Molecular segmentation index.
        :returns: Graph readout vectors.
        """
        segmentation_max = segmentation_molecule.max() + 1
        atom_features = self.leaky_relu(self.prediction_readout_projection(atom_features))
        hidden_channels = atom_features.size(1)
        readout_vectors = atom_features.new_zeros((segmentation_max, hidden_channels)).index_add(0, segmentation_molecule, atom_features)
        return readout_vectors


class MHCADDI(Model):
    """An implementation of the MHCADDI model from [deac2019]_.

    .. seealso:: This model was suggested in https://github.com/AstraZeneca/chemicalx/issues/13

    .. [deac2019] Deac, A., *et al.* (2019). `Drug-Drug Adverse Effect Prediction with
       Graph Co-Attention <http://arxiv.org/abs/1905.00534>`_. *arXiv*, 1905.00534.
    """

    def __init__(self, *, atom_feature_channels: int=16, atom_type_channels: int=16, bond_type_channels: int=16, node_channels: int=16, edge_channels: int=16, hidden_channels: int=16, readout_channels: int=16, output_channels: int=1, dropout: float=0.5):
        """Instantiate the MHCADDI network.

        :param atom_feature_channels: Number of atom features.
        :param atom_type_channels: Number of atom types.
        :param bond_type_channels: Number of bonds.
        :param node_channels: Node feature number.
        :param edge_channels: Edge feature number.
        :param hidden_channels: Number of hidden layers.
        :param readout_channels: Readout dimensions.
        :param output_channels: Number of labels.
        :param dropout: Dropout rate.
        """
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.atom_projection = nn.Linear(node_channels + atom_feature_channels, node_channels)
        self.atom_embedding = nn.Embedding(atom_type_channels, node_channels, padding_idx=0)
        self.bond_embedding = nn.Embedding(bond_type_channels, edge_channels, padding_idx=0)
        nn.init.xavier_normal_(self.atom_embedding.weight)
        nn.init.xavier_normal_(self.bond_embedding.weight)
        self.encoder = CoAttentionMessagePassingNetwork(hidden_channels=hidden_channels, readout_channels=readout_channels, dropout=dropout)
        self.head_layer = nn.Linear(readout_channels * 2, output_channels)

    def _get_molecule_features(self, drug_molecules_primary: 'PackedGraph', drug_molecules_secondary: 'PackedGraph'):
        outer_segmentation_index, outer_index = self.generate_outer_segmentation(drug_molecules_primary.num_nodes, drug_molecules_secondary.num_nodes)
        atom = self.dropout(self.atom_comp(drug_molecules_primary.node_feature, drug_molecules_primary.atom_type))
        bond = self.dropout(self.bond_embedding(drug_molecules_primary.bond_type))
        return outer_segmentation_index, outer_index, atom, bond

    def unpack(self, batch: 'DrugPairBatch'):
        """Adjust drug pair batch to model design.

        :param batch: Molecular data in a drug pair batch.
        :returns: Tuple of data.
        """
        return batch.drug_molecules_left, batch.drug_molecules_right

    def forward(self, drug_molecules_left: 'PackedGraph', drug_molecules_right: 'PackedGraph') ->torch.FloatTensor:
        """Forward pass with the data."""
        outer_segmentation_index_left, outer_index_left, atom_left, bond_left = self._get_molecule_features(drug_molecules_left, drug_molecules_right)
        outer_segmentation_index_right, outer_index_right, atom_right, bond_right = self._get_molecule_features(drug_molecules_right, drug_molecules_left)
        drug_left, drug_right = self.encoder(drug_molecules_left.node2graph, atom_left, bond_left, drug_molecules_left.edge_list[:, 0], drug_molecules_left.edge_list[:, 1], outer_segmentation_index_left, outer_index_left, drug_molecules_right.node2graph, atom_right, bond_right, drug_molecules_right.edge_list[:, 0], drug_molecules_right.edge_list[:, 1], outer_segmentation_index_right, outer_index_right)
        prediction_left = self.head_layer(torch.cat([drug_left, drug_right], dim=1))
        prediction_right = self.head_layer(torch.cat([drug_right, drug_left], dim=1))
        prediction_mean = (prediction_left + prediction_right) / 2
        return torch.sigmoid(prediction_mean)

    def atom_comp(self, atom_features: 'torch.Tensor', atom_index: 'torch.Tensor'):
        """Compute atom projection, a linear transformation of a learned atom embedding and the atom features.

        :param atom_features: Atom input features
        :param atom_index: Index of atom type
        :returns: Node index.
        """
        atom_embedding = self.atom_embedding(atom_index)
        node_embedding = self.atom_projection(torch.cat([atom_embedding, atom_features], -1))
        return node_embedding

    def generate_outer_segmentation(self, graph_sizes_left: 'torch.LongTensor', graph_sizes_right: 'torch.LongTensor'):
        """Calculate all pairwise edges between the atoms in a set of drug pairs.

        Example: Given two sets of drug sizes:

        graph_sizes_left = torch.tensor([1, 2])
        graph_sizes_right = torch.tensor([3, 4])

        Here the drug pairs have sizes (1,3) and (2,4)

        This results in:

        outer_segmentation_index = tensor([0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
        outer_index = tensor([0, 1, 2, 3, 4, 5, 6, 3, 4, 5, 6])

        :param graph_sizes_left: List of graph sizes in the left drug batch.
        :param graph_sizes_right: List of graph sizes in the right drug batch.
        :returns: Edge indices.
        """
        interactions = graph_sizes_left * graph_sizes_right
        left_shifted_graph_size_cum_sum = torch.cumsum(graph_sizes_left, 0) - graph_sizes_left
        shift_sums_left = torch.repeat_interleave(left_shifted_graph_size_cum_sum, interactions)
        outer_segmentation_index = [np.repeat(np.array(range(0, left_graph_size)), right_graph_size) for left_graph_size, right_graph_size in zip(graph_sizes_left, graph_sizes_right)]
        outer_segmentation_index = functools.reduce(operator.iconcat, outer_segmentation_index, [])
        outer_segmentation_index = torch.tensor(outer_segmentation_index) + shift_sums_left
        right_shifted_graph_size_cum_sum = torch.cumsum(graph_sizes_right, 0) - graph_sizes_right
        shift_sums_right = torch.repeat_interleave(right_shifted_graph_size_cum_sum, interactions)
        outer_index = [(list(range(0, right_graph_size)) * left_graph_size) for left_graph_size, right_graph_size in zip(graph_sizes_left, graph_sizes_right)]
        outer_index = functools.reduce(operator.iconcat, outer_index, [])
        outer_index = torch.tensor(outer_index) + shift_sums_right
        return outer_segmentation_index, outer_index


class EmbeddingLayer(torch.nn.Module):
    """Attention layer."""

    def __init__(self, feature_number: 'int'):
        """Initialize the relational embedding layer.

        :param feature_number: Number of features.
        """
        super().__init__()
        self.weights = torch.nn.Parameter(torch.zeros(feature_number, feature_number))
        torch.nn.init.xavier_uniform_(self.weights)

    def forward(self, left_representations: 'torch.FloatTensor', right_representations: 'torch.FloatTensor', alpha_scores: 'torch.FloatTensor'):
        """
        Make a forward pass with the drug representations.

        :param left_representations: Left side drug representations.
        :param right_representations: Right side drug representations.
        :param alpha_scores: Attention scores.
        :returns: Positive label scores vector.
        """
        attention = torch.nn.functional.normalize(self.weights, dim=-1)
        left_representations = torch.nn.functional.normalize(left_representations, dim=-1)
        right_representations = torch.nn.functional.normalize(right_representations, dim=-1)
        attention = attention.view(-1, self.weights.shape[0], self.weights.shape[1])
        scores = alpha_scores * (left_representations @ attention @ right_representations.transpose(-2, -1))
        scores = scores.sum(dim=(-2, -1)).view(-1, 1)
        return scores


class DrugDrugAttentionLayer(torch.nn.Module):
    """Co-attention layer for drug pairs."""

    def __init__(self, feature_number: 'int'):
        """Initialize the co-attention layer.

        :param feature_number: Number of input features.
        """
        super().__init__()
        self.weight_query = torch.nn.Parameter(torch.zeros(feature_number, feature_number // 2))
        self.weight_key = torch.nn.Parameter(torch.zeros(feature_number, feature_number // 2))
        self.bias = torch.nn.Parameter(torch.zeros(feature_number // 2))
        self.attention = torch.nn.Parameter(torch.zeros(feature_number // 2))
        self.tanh = torch.nn.Tanh()
        torch.nn.init.xavier_uniform_(self.weight_query)
        torch.nn.init.xavier_uniform_(self.weight_key)
        torch.nn.init.xavier_uniform_(self.bias.view(*self.bias.shape, -1))
        torch.nn.init.xavier_uniform_(self.attention.view(*self.attention.shape, -1))

    def forward(self, left_representations: 'torch.Tensor', right_representations: 'torch.Tensor'):
        """Make a forward pass with the co-attention calculation.

        :param left_representations: Matrix of left hand side representations.
        :param right_representations: Matrix of right hand side representations.
        :returns: Attention scores.
        """
        keys = left_representations @ self.weight_key
        queries = right_representations @ self.weight_query
        e_activations = queries.unsqueeze(-3) + keys.unsqueeze(-2) + self.bias
        attentions = self.tanh(e_activations) @ self.attention
        return attentions


class SSIDDIBlock(torch.nn.Module):
    """SSIDDI Block with convolution and pooling."""

    def __init__(self, head_number: 'int', in_channels: 'int', out_channels: 'int'):
        """Initialize an SSI-DDI Block.

        :param head_number: Number of attention heads.
        :param in_channels: Number of input channels.
        :param out_channels: Number of convolutional filters.
        """
        super().__init__()
        self.conv = GraphAttentionConv(input_dim=in_channels, output_dim=out_channels, num_head=head_number)
        self.readout = MeanReadout()

    def forward(self, molecules: 'PackedGraph'):
        """Make a forward pass.

        :param molecules: A batch of graphs.
        :returns: The molecules with updated atom states and the pooled representations.
        """
        molecules.node_feature = self.conv(molecules, molecules.node_feature)
        h_graphs = self.readout(molecules, molecules.node_feature)
        return molecules, h_graphs


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (CASTER,
     lambda: ([], {'drug_channels': 4}),
     lambda: ([torch.rand([4, 4])], {})),
    (DeepDDI,
     lambda: ([], {'drug_channels': 4}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4])], {})),
    (DeepSynergy,
     lambda: ([], {'context_channels': 4, 'drug_channels': 4}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4]), torch.rand([4, 4])], {})),
    (DrugDrugAttentionLayer,
     lambda: ([], {'feature_number': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (EmbeddingLayer,
     lambda: ([], {'feature_number': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (Highway,
     lambda: ([], {'input_size': 4, 'prev_input_size': 4}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4])], {})),
    (MatchMaker,
     lambda: ([], {'context_channels': 4, 'drug_channels': 4}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4]), torch.rand([4, 4])], {})),
]

