
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


import uuid


from abc import ABCMeta


from typing import Any


from typing import Dict


from typing import List


from typing import Optional


from typing import Sequence


from typing import Tuple


from typing import Union


from typing import cast


import numpy as np


from scipy.sparse import issparse


from sklearn.utils.multiclass import type_of_target


from torch.utils.data import Dataset


from torch.utils.data import Subset


import torchvision


import torch


from torch.utils.data import TensorDataset


import torchvision.transforms


from torchvision.transforms import functional as TF


import copy


import warnings


import pandas as pd


from pandas._libs.tslibs.np_datetime import OutOfBoundsDatetime


from torch.utils.data.dataset import ConcatDataset


from torch.utils.data.dataset import Dataset


from collections import Counter


from sklearn.pipeline import Pipeline


from sklearn.utils.validation import check_random_state


from scipy.sparse import spmatrix


from sklearn.base import BaseEstimator


from sklearn.compose import ColumnTransformer


from sklearn.pipeline import make_pipeline


import torch.optim.lr_scheduler


from torch.optim import Optimizer


from torch.optim.lr_scheduler import _LRScheduler


from torch import nn


from abc import abstractmethod


from torch.distributions import AffineTransform


from torch.distributions import TransformedDistribution


from typing import Iterable


import math


from collections import OrderedDict


from torch.nn import functional as F


from typing import Callable


from typing import NamedTuple


from enum import Enum


from torch.nn.utils import weight_norm


from torch.autograd import Function


import torch.nn as nn


from typing import Type


import torch.nn.functional as F


from torch.distributions import Beta


from torch.distributions import Distribution


from torch.distributions import Gamma


from torch.distributions import Normal


from torch.distributions import Poisson


from torch.distributions import StudentT


from torch.optim import Adam


from torch.optim import AdamW


from torch.optim import RMSprop


from torch.optim import SGD


import logging.handlers


from sklearn.utils import check_random_state


from sklearn.utils import check_array


from functools import partial


from typing import Iterator


import collections


from typing import Mapping


from typing import Sized


from torch.utils.data._utils.collate import default_collate


from torch.utils.data._utils.collate import default_collate_err_msg_format


from torch.utils.data._utils.collate import np_str_obj_array_pattern


from torch.utils.data.sampler import SequentialSampler


from torch.utils.data.sampler import SubsetRandomSampler


from torch.nn.modules.loss import BCEWithLogitsLoss


from torch.nn.modules.loss import CrossEntropyLoss


from torch.nn.modules.loss import L1Loss


from torch.nn.modules.loss import MSELoss


from torch.nn.modules.loss import _Loss as Loss


import time


from torch.utils.tensorboard.writer import SummaryWriter


from abc import ABC


from sklearn.base import ClassifierMixin


from sklearn.base import RegressorMixin


from torch.utils.data.dataloader import default_collate


from scipy import sparse


from sklearn.base import TransformerMixin


import random


import sklearn.datasets


import re


from sklearn.datasets import fetch_openml


from sklearn.datasets import make_classification


from sklearn.datasets import make_regression


import itertools


from itertools import product


from sklearn.base import clone


import logging


from sklearn.preprocessing import StandardScaler


ALL_NET_OUTPUT = Union[torch.Tensor, List[torch.Tensor], torch.distributions.Distribution]


class AddLayer(nn.Module):

    def __init__(self, input_size: 'int', skip_size: 'int'):
        super().__init__()
        if input_size == skip_size:
            self.fc = nn.Linear(skip_size, input_size)
        self.norm = nn.LayerNorm(input_size)

    def forward(self, input: 'torch.Tensor', skip: 'torch.Tensor') ->torch.Tensor:
        if hasattr(self, 'fc'):
            return self.norm(input + self.fc(skip))
        else:
            return self.norm(input)


class StackedDecoder(nn.Module):
    """
    Decoder network that is stacked by several decoders. Skip-connections can be applied to each stack. It decodes the
    encoded features (encoder2decoder) from each corresponding stacks and known_future_features to generate the decoded
    output features that will be further fed to the network decoder.
    """

    def __init__(self, network_structure: 'NetworkStructure', encoder: 'nn.ModuleDict', encoder_info: 'Dict[str, EncoderBlockInfo]', decoder_info: 'Dict[str, DecoderBlockInfo]'):
        super().__init__()
        self.num_blocks = network_structure.num_blocks
        self.first_block = -1
        self.skip_connection = network_structure.skip_connection
        self.decoder_has_hidden_states = []
        decoder = nn.ModuleDict()
        for i in range(1, self.num_blocks + 1):
            block_id = f'block_{i}'
            if block_id in decoder_info:
                self.first_block = i if self.first_block == -1 else self.first_block
                decoder[block_id] = decoder_info[block_id].decoder
                if decoder_info[block_id].decoder_properties.has_hidden_states:
                    self.decoder_has_hidden_states.append(True)
                else:
                    self.decoder_has_hidden_states.append(False)
                if self.skip_connection:
                    input_size_encoder = encoder_info[block_id].encoder_output_shape[-1]
                    skip_size_encoder = encoder_info[block_id].encoder_input_shape[-1]
                    input_size_decoder = decoder_info[block_id].decoder_output_shape[-1]
                    skip_size_decoder = decoder_info[block_id].decoder_input_shape[-1]
                    if skip_size_decoder > 0:
                        if input_size_encoder == input_size_decoder and skip_size_encoder == skip_size_decoder:
                            decoder[f'skip_connection_{i}'] = encoder[f'skip_connection_{i}']
                        elif network_structure.skip_connection_type == 'add':
                            decoder[f'skip_connection_{i}'] = AddLayer(input_size_decoder, skip_size_decoder)
                        elif network_structure.skip_connection_type == 'gate_add_norm':
                            decoder[f'skip_connection_{i}'] = GateAddNorm(input_size_decoder, hidden_size=input_size_decoder, skip_size=skip_size_decoder, dropout=network_structure.grn_dropout_rate)
        self.cached_intermediate_state = [torch.empty(0) for _ in range(self.num_blocks + 1 - self.first_block)]
        self.decoder = decoder

    def forward(self, x_future: 'Optional[torch.Tensor]', encoder_output: 'List[torch.Tensor]', pos_idx: 'Optional[Tuple[int]]'=None, cache_intermediate_state: 'bool'=False, incremental_update: 'bool'=False) ->torch.Tensor:
        """
        A forward pass through the decoder

        Args:
            x_future (Optional[torch.Tensor]):
                known future features
            encoder_output (List[torch.Tensor])
                encoded features, stored as List, whereas each element in the list indicates encoded features from an
                encoder stack
            pos_idx (int)
                position index of the current x_future. This is applied to transformer decoder
            cache_intermediate_state (bool):
                if the intermediate values are cached
            incremental_update (bool):
                if an incremental update is applied, this is normally applied for auto-regressive model

        Returns:
            decoder_output (torch.Tensor):
                decoder output that will be passed to the network head
        """
        x = x_future
        for i, block_id in enumerate(range(self.first_block, self.num_blocks + 1)):
            decoder_i = self.decoder[f'block_{block_id}']
            if self.decoder_has_hidden_states[i]:
                if incremental_update:
                    hx = self.cached_intermediate_state[i]
                    fx, hx = decoder_i(x_future=x, encoder_output=hx, pos_idx=pos_idx)
                else:
                    fx, hx = decoder_i(x_future=x, encoder_output=encoder_output[i], pos_idx=pos_idx)
            elif incremental_update:
                fx = decoder_i(x, encoder_output=encoder_output[i], pos_idx=pos_idx)
            else:
                fx = decoder_i(x, encoder_output=encoder_output[i], pos_idx=pos_idx)
            skip_id = f'skip_connection_{block_id}'
            if self.skip_connection and skip_id in self.decoder and x is not None:
                fx = self.decoder[skip_id](fx, x)
            if cache_intermediate_state:
                if self.decoder_has_hidden_states[i]:
                    self.cached_intermediate_state[i] = hx
            x = fx
        return x


class EncoderOutputForm(Enum):
    NoOutput = 0
    HiddenStates = 1
    Sequence = 2
    SequenceLast = 3


class StackedEncoder(nn.Module):
    """
    Encoder network that is stacked by several encoders. Skip-connections can be applied to each stack. Each stack
    needs to generate a sequence of encoded features passed to the next stack and the
    corresponding decoder (encoder2decoder) that is located at the same layer.Additionally, if temporal fusion
    transformer is applied, the last encoder also needs to output the full encoded feature sequence
    """

    def __init__(self, network_structure: 'NetworkStructure', has_temporal_fusion: 'bool', encoder_info: 'Dict[str, EncoderBlockInfo]', decoder_info: 'Dict[str, DecoderBlockInfo]'):
        super().__init__()
        self.num_blocks = network_structure.num_blocks
        self.skip_connection = network_structure.skip_connection
        self.has_temporal_fusion = has_temporal_fusion
        self.encoder_output_type = [EncoderOutputForm.NoOutput] * self.num_blocks
        self.encoder_has_hidden_states = [False] * self.num_blocks
        len_cached_intermediate_states = self.num_blocks + 1 if self.has_temporal_fusion else self.num_blocks
        self.cached_intermediate_state = [torch.empty(0) for _ in range(len_cached_intermediate_states)]
        self.encoder_num_hidden_states = [0] * self.num_blocks
        encoder = nn.ModuleDict()
        for i, block_idx in enumerate(range(1, self.num_blocks + 1)):
            block_id = f'block_{block_idx}'
            encoder[block_id] = encoder_info[block_id].encoder
            if self.skip_connection:
                input_size = encoder_info[block_id].encoder_output_shape[-1]
                skip_size = encoder_info[block_id].encoder_input_shape[-1]
                if network_structure.skip_connection_type == 'add':
                    encoder[f'skip_connection_{block_idx}'] = AddLayer(input_size, skip_size)
                elif network_structure.skip_connection_type == 'gate_add_norm':
                    encoder[f'skip_connection_{block_idx}'] = GateAddNorm(input_size, hidden_size=input_size, skip_size=skip_size, dropout=network_structure.grn_dropout_rate)
            if block_id in decoder_info:
                if decoder_info[block_id].decoder_properties.recurrent:
                    if decoder_info[block_id].decoder_properties.has_hidden_states:
                        self.encoder_output_type[i] = EncoderOutputForm.HiddenStates
                    else:
                        self.encoder_output_type[i] = EncoderOutputForm.Sequence
                else:
                    self.encoder_output_type[i] = EncoderOutputForm.SequenceLast
            if encoder_info[block_id].encoder_properties.has_hidden_states:
                self.encoder_has_hidden_states[i] = True
                self.encoder_num_hidden_states[i] = encoder_info[block_id].n_hidden_states
            else:
                self.encoder_has_hidden_states[i] = False
        self.encoder = encoder

    def forward(self, encoder_input: 'torch.Tensor', additional_input: 'List[Optional[torch.Tensor]]', output_seq: 'bool'=False, cache_intermediate_state: 'bool'=False, incremental_update: 'bool'=False) ->Tuple[List[torch.Tensor], Optional[torch.Tensor]]:
        """
        A forward pass through the encoder

        Args:
            encoder_input (torch.Tensor):
                encoder input
            additional_input (List[Optional[torch.Tensor]])
                additional input to the encoder, e.g., initial hidden states
            output_seq (bool)
                if the encoder want to generate a sequence of multiple time steps or a single time step
            cache_intermediate_state (bool):
                if the intermediate values are cached
            incremental_update (bool):
                if an incremental update is applied, this is normally applied for
                auto-regressive model, however, ony deepAR requires incremental update in encoder

        Returns:
            encoder2decoder ([List[torch.Tensor]]):
                encoder output that will be passed to decoders
            encoder_output (torch.Tensor):
                full sequential encoded features from the last encoder layer. Applied to temporal transformer
        """
        encoder2decoder = []
        x = encoder_input
        for i, block_id in enumerate(range(1, self.num_blocks + 1)):
            output_seq_i = output_seq or self.has_temporal_fusion or block_id < self.num_blocks
            encoder_i = self.encoder[f'block_{block_id}']
            if self.encoder_has_hidden_states[i]:
                if incremental_update:
                    hx = self.cached_intermediate_state[i]
                    fx, hx = encoder_i(x, output_seq=False, hx=hx)
                else:
                    rnn_num_layers = encoder_i.config['num_layers']
                    hx = additional_input[i]
                    if hx is None:
                        fx, hx = encoder_i(x, output_seq=output_seq_i, hx=hx)
                    elif self.encoder_num_hidden_states[i] == 1:
                        fx, hx = encoder_i(x, output_seq=output_seq_i, hx=hx[0].expand((rnn_num_layers, -1, -1)).contiguous())
                    else:
                        hx = tuple(hx_i.expand(rnn_num_layers, -1, -1).contiguous() for hx_i in hx)
                        fx, hx = encoder_i(x, output_seq=output_seq_i, hx=hx)
            elif incremental_update:
                x_all = torch.cat([self.cached_intermediate_state[i], x], dim=1)
                fx = encoder_i(x_all, output_seq=False)
            else:
                fx = encoder_i(x, output_seq=output_seq_i)
            if self.skip_connection:
                if output_seq_i:
                    fx = self.encoder[f'skip_connection_{block_id}'](fx, x)
                else:
                    fx = self.encoder[f'skip_connection_{block_id}'](fx, x[:, -1:])
            if self.encoder_output_type[i] == EncoderOutputForm.HiddenStates:
                encoder2decoder.append(hx)
            elif self.encoder_output_type[i] == EncoderOutputForm.Sequence:
                encoder2decoder.append(fx)
            elif self.encoder_output_type[i] == EncoderOutputForm.SequenceLast:
                if output_seq_i and not output_seq:
                    encoder2decoder.append(encoder_i.get_last_seq_value(fx).squeeze(1))
                else:
                    encoder2decoder.append(fx)
            else:
                raise NotImplementedError
            if cache_intermediate_state:
                if self.encoder_has_hidden_states[i]:
                    self.cached_intermediate_state[i] = hx
                elif incremental_update:
                    self.cached_intermediate_state[i] = x_all
                else:
                    self.cached_intermediate_state[i] = x
            x = fx
        if self.has_temporal_fusion:
            if incremental_update:
                self.cached_intermediate_state[i + 1] = torch.cat([self.cached_intermediate_state[i + 1], x], dim=1)
            else:
                self.cached_intermediate_state[i + 1] = x
            return encoder2decoder, x
        else:
            return encoder2decoder, None


class TransformedDistribution_(TransformedDistribution):
    """
    We implement the mean function such that we do not need to enquire base mean every time
    """

    @property
    def mean(self) ->torch.Tensor:
        mean = self.base_dist.mean
        for transform in self.transforms:
            mean = transform(mean)
        return mean


class VariableSelector(nn.Module):

    def __init__(self, network_structure: 'NetworkStructure', dataset_properties: 'Dict[str, Any]', network_encoder: 'Dict[str, EncoderBlockInfo]', auto_regressive: 'bool'=False, feature_names: 'Union[Tuple[str], Tuple[()]]'=(), known_future_features: 'Union[Tuple[str], Tuple[()]]'=(), feature_shapes: 'Dict[str, int]'={}, static_features: 'Union[Tuple[Union[str, int]], Tuple[()]]'=(), time_feature_names: 'Union[Tuple[str], Tuple[()]]'=()):
        """
        Variable Selector. This models follows the implementation from
        pytorch_forecasting.models.temporal_fusion_transformer.sub_modules.VariableSelectionNetwork
        However, we adjust the structure to fit the data extracted from our dataloader: we record the feature index from
        each feature names and break the input features on the fly.

        The order of the input variables is as follows:
        [features (from the dataset), time_features (from time feature transformers), targets]
        Args:
            network_structure (NetworkStructure):
                contains the information of the overall architecture information
            dataset_properties (Dict):
                dataset properties
            network_encoder(Dict[str, EncoderBlockInfo]):
                Network encoders
            auto_regressive (bool):
                if it belongs to an auto-regressive model
            feature_names (Tuple[str]):
                feature names, used to construct the selection network
            known_future_features (Tuple[str]):
                known future features
            feature_shapes (Dict[str, int]):
                shapes of each features
            time_feature_names (Tuple[str]):
                time feature names, used to complement feature_shapes
        """
        super().__init__()
        first_encoder_output_shape = network_encoder['block_1'].encoder_output_shape[-1]
        self.hidden_size = first_encoder_output_shape
        assert set(feature_names) == set(feature_shapes.keys()), f'feature_names and feature_shapes must have the same variable names but they are differentat {set(feature_names) ^ set(feature_shapes.keys())}'
        pre_scalar = {'past_targets': nn.Linear(dataset_properties['output_shape'][-1], self.hidden_size)}
        encoder_input_sizes = {'past_targets': self.hidden_size}
        decoder_input_sizes = {}
        future_feature_name2tensor_idx = {}
        feature_names2tensor_idx = {}
        idx_tracker = 0
        idx_tracker_future = 0
        static_features = set(static_features)
        static_features_input_size = {}
        known_future_features = tuple(known_future_features)
        feature_names = tuple(feature_names)
        time_feature_names = tuple(time_feature_names)
        if feature_names:
            for name in feature_names:
                feature_shape = feature_shapes[name]
                feature_names2tensor_idx[name] = [idx_tracker, idx_tracker + feature_shape]
                idx_tracker += feature_shape
                pre_scalar[name] = nn.Linear(feature_shape, self.hidden_size)
                if name in static_features:
                    static_features_input_size[name] = self.hidden_size
                else:
                    encoder_input_sizes[name] = self.hidden_size
                    if name in known_future_features:
                        decoder_input_sizes[name] = self.hidden_size
        for future_name in known_future_features:
            feature_shape = feature_shapes[future_name]
            future_feature_name2tensor_idx[future_name] = [idx_tracker_future, idx_tracker_future + feature_shape]
            idx_tracker_future += feature_shape
        if time_feature_names:
            for name in time_feature_names:
                feature_names2tensor_idx[name] = [idx_tracker, idx_tracker + 1]
                future_feature_name2tensor_idx[name] = [idx_tracker_future, idx_tracker_future + 1]
                idx_tracker += 1
                idx_tracker_future += 1
                pre_scalar[name] = nn.Linear(1, self.hidden_size)
                encoder_input_sizes[name] = self.hidden_size
                decoder_input_sizes[name] = self.hidden_size
        if not feature_names or not known_future_features:
            placeholder_features = 'placeholder_features'
            i = 0
            self.placeholder_features: 'List[str]' = []
            while placeholder_features in feature_names or placeholder_features in self.placeholder_features:
                i += 1
                placeholder_features = f'placeholder_features_{i}'
                if i == 5000:
                    raise RuntimeError('Cannot assign name to placeholder features, please considering rename your features')
            name = placeholder_features
            pre_scalar[name] = nn.Linear(1, self.hidden_size)
            encoder_input_sizes[name] = self.hidden_size
            decoder_input_sizes[name] = self.hidden_size
            self.placeholder_features.append(placeholder_features)
        feature_names = time_feature_names + feature_names
        known_future_features = time_feature_names + known_future_features
        self.feature_names = feature_names
        self.feature_names2tensor_idx = feature_names2tensor_idx
        self.future_feature_name2tensor_idx = future_feature_name2tensor_idx
        self.known_future_features = known_future_features
        if auto_regressive:
            pre_scalar.update({'future_prediction': nn.Linear(dataset_properties['output_shape'][-1], self.hidden_size)})
            decoder_input_sizes.update({'future_prediction': self.hidden_size})
        self.pre_scalars = nn.ModuleDict(pre_scalar)
        self._device = torch.device('cpu')
        if not dataset_properties['uni_variant']:
            self.static_variable_selection = VariableSelectionNetwork(input_sizes=static_features_input_size, hidden_size=self.hidden_size, input_embedding_flags={}, dropout=network_structure.grn_dropout_rate, prescalers=self.pre_scalars)
        self.static_input_sizes = static_features_input_size
        self.static_features = static_features
        self.auto_regressive = auto_regressive
        if network_structure.share_single_variable_networks:
            self.shared_single_variable_grns = nn.ModuleDict()
            for name, input_size in encoder_input_sizes.items():
                self.shared_single_variable_grns[name] = GatedResidualNetwork(input_size, min(input_size, self.hidden_size), self.hidden_size, network_structure.grn_dropout_rate)
            for name, input_size in decoder_input_sizes.items():
                if name not in self.shared_single_variable_grns:
                    self.shared_single_variable_grns[name] = GatedResidualNetwork(input_size, min(input_size, self.hidden_size), self.hidden_size, network_structure.grn_dropout_rate)
        self.encoder_variable_selection = VariableSelectionNetwork(input_sizes=encoder_input_sizes, hidden_size=self.hidden_size, input_embedding_flags={}, dropout=network_structure.grn_dropout_rate, context_size=self.hidden_size, single_variable_grns={} if not network_structure.share_single_variable_networks else self.shared_single_variable_grns, prescalers=self.pre_scalars)
        self.decoder_variable_selection = VariableSelectionNetwork(input_sizes=decoder_input_sizes, hidden_size=self.hidden_size, input_embedding_flags={}, dropout=network_structure.grn_dropout_rate, context_size=self.hidden_size, single_variable_grns={} if not network_structure.share_single_variable_networks else self.shared_single_variable_grns, prescalers=self.pre_scalars)
        self.static_context_variable_selection = GatedResidualNetwork(input_size=self.hidden_size, hidden_size=self.hidden_size, output_size=self.hidden_size, dropout=network_structure.grn_dropout_rate)
        n_hidden_states = 0
        if network_encoder['block_1'].encoder_properties.has_hidden_states:
            n_hidden_states = network_encoder['block_1'].n_hidden_states
        static_context_initial_hidden = [GatedResidualNetwork(input_size=self.hidden_size, hidden_size=self.hidden_size, output_size=self.hidden_size, dropout=network_structure.grn_dropout_rate) for _ in range(n_hidden_states)]
        self.static_context_initial_hidden = nn.ModuleList(static_context_initial_hidden)
        self.cached_static_contex: 'Optional[torch.Tensor]' = None
        self.cached_static_embedding: 'Optional[torch.Tensor]' = None

    @property
    def device(self) ->torch.device:
        return self._device

    @device.setter
    def device(self, device: 'torch.device') ->None:
        self
        self._device = device

    def forward(self, x_past: 'Optional[Dict[str, torch.Tensor]]', x_future: 'Optional[Dict[str, torch.Tensor]]', x_static: 'Optional[Dict[str, torch.Tensor]]', length_past: 'int'=0, length_future: 'int'=0, batch_size: 'int'=0, cache_static_contex: 'bool'=False, use_cached_static_contex: 'bool'=False) ->Tuple[Optional[torch.Tensor], Optional[torch.Tensor], torch.Tensor, Optional[torch.Tensor]]:
        if x_past is None and x_future is None:
            raise ValueError('Either past input or future inputs need to be given!')
        if length_past == 0 and length_future == 0:
            raise ValueError('Either length_past or length_future must be given!')
        timesteps = length_past + length_future
        if not use_cached_static_contex:
            if len(self.static_input_sizes) > 0:
                static_embedding, _ = self.static_variable_selection(x_static)
            else:
                if length_past > 0:
                    assert x_past is not None, 'x_past must be given when length_past is greater than 0!'
                    model_dtype = next(iter(x_past.values())).dtype
                else:
                    assert x_future is not None, 'x_future must be given when length_future is greater than 0!'
                    model_dtype = next(iter(x_future.values())).dtype
                static_embedding = torch.zeros((batch_size, self.hidden_size), dtype=model_dtype, device=self.device)
            static_context_variable_selection = self.static_context_variable_selection(static_embedding)[:, None]
            static_context_initial_hidden: 'Optional[Tuple[torch.Tensor, ...]]' = tuple(init_hidden(static_embedding) for init_hidden in self.static_context_initial_hidden)
            if cache_static_contex:
                self.cached_static_contex = static_context_variable_selection
                self.cached_static_embedding = static_embedding
        else:
            static_embedding = self.cached_static_embedding
            static_context_initial_hidden = None
            static_context_variable_selection = self.cached_static_contex
        static_context_variable_selection = static_context_variable_selection.expand(-1, timesteps, -1)
        if x_past is not None:
            embeddings_varying_encoder, _ = self.encoder_variable_selection(x_past, static_context_variable_selection[:, :length_past])
        else:
            embeddings_varying_encoder = None
        if x_future is not None:
            embeddings_varying_decoder, _ = self.decoder_variable_selection(x_future, static_context_variable_selection[:, length_past:])
        else:
            embeddings_varying_decoder = None
        return embeddings_varying_encoder, embeddings_varying_decoder, static_embedding, static_context_initial_hidden


class _NoEmbedding(nn.Module):

    def get_partial_models(self, subset_features: 'List[int]') ->'_NoEmbedding':
        return self

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        return x


class AbstractForecastingNet(nn.Module):
    """
    This is a basic forecasting network. It is only composed of a embedding net, an encoder and a head (including
    MLP decoder and the final head).

    This structure is active when the decoder is a MLP with auto_regressive set as false

    Attributes:
        network_structure (NetworkStructure):
            network structure information
        network_embedding (nn.Module):
            network embedding
        network_encoder (Dict[str, EncoderBlockInfo]):
            Encoder network, could be selected to return a sequence or a 2D Matrix
        network_decoder (Dict[str, DecoderBlockInfo]):
            network decoder
        temporal_fusion Optional[TemporalFusionLayer]:
            Temporal Fusion Layer
        network_head (nn.Module):
            network head, maps the output of decoder to the final output
        dataset_properties (Dict):
            dataset properties
        auto_regressive (bool):
            if the model is auto-regressive model
        output_type (str):
            the form that the network outputs. It could be regression, distribution or quantile
        forecast_strategy (str):
            only valid if output_type is distribution or quantile, how the network transforms
            its output to predicted values, could be mean or sample
        num_samples (int):
            only valid if output_type is not regression and forecast_strategy is sample. This indicates the
            number of the points to sample when doing prediction
        aggregation (str):
            only valid if output_type is not regression and forecast_strategy is sample. The way that the samples
            are aggregated. We could take their mean or median values.
    """
    future_target_required = False
    dtype = torch.float

    def __init__(self, network_structure: 'NetworkStructure', network_embedding: 'nn.Module', network_encoder: 'Dict[str, EncoderBlockInfo]', network_decoder: 'Dict[str, DecoderBlockInfo]', temporal_fusion: 'Optional[TemporalFusionLayer]', network_head: 'nn.Module', window_size: 'int', target_scaler: 'BaseTargetScaler', dataset_properties: 'Dict', auto_regressive: 'bool', feature_names: 'Union[Tuple[str], Tuple[()]]'=(), known_future_features: 'Union[Tuple[str], Tuple[()]]'=(), feature_shapes: 'Dict[str, int]'={}, static_features: 'Union[Tuple[str], Tuple[()]]'=(), time_feature_names: 'Union[Tuple[str], Tuple[()]]'=(), output_type: 'str'='regression', forecast_strategy: 'Optional[str]'='mean', num_samples: 'int'=50, aggregation: 'str'='mean'):
        super().__init__()
        self.network_structure = network_structure
        self.embedding = network_embedding
        if len(known_future_features) > 0:
            known_future_features_idx = [feature_names.index(kff) for kff in known_future_features]
            self.decoder_embedding = self.embedding.get_partial_models(known_future_features_idx)
        else:
            self.decoder_embedding = _NoEmbedding()
        self.lazy_modules = []
        if network_structure.variable_selection:
            self.variable_selector = VariableSelector(network_structure=network_structure, dataset_properties=dataset_properties, network_encoder=network_encoder, auto_regressive=auto_regressive, feature_names=feature_names, known_future_features=known_future_features, feature_shapes=feature_shapes, static_features=static_features, time_feature_names=time_feature_names)
            self.lazy_modules.append(self.variable_selector)
        has_temporal_fusion = network_structure.use_temporal_fusion
        self.encoder = StackedEncoder(network_structure=network_structure, has_temporal_fusion=has_temporal_fusion, encoder_info=network_encoder, decoder_info=network_decoder)
        self.decoder = StackedDecoder(network_structure=network_structure, encoder=self.encoder.encoder, encoder_info=network_encoder, decoder_info=network_decoder)
        if has_temporal_fusion:
            if temporal_fusion is None:
                raise ValueError('When the network structure uses temporal fusion layer, temporal_fusion must be given!')
            self.temporal_fusion = temporal_fusion
            self.lazy_modules.append(self.temporal_fusion)
        self.has_temporal_fusion = has_temporal_fusion
        self.head = network_head
        first_decoder = 'block_0'
        for i in range(1, network_structure.num_blocks + 1):
            block_number = f'block_{i}'
            if block_number in network_decoder:
                if first_decoder == 'block_0':
                    first_decoder = block_number
        if first_decoder == 0:
            raise ValueError('At least one decoder must be specified!')
        self.target_scaler = target_scaler
        self.n_prediction_steps = dataset_properties['n_prediction_steps']
        self.window_size = window_size
        self.output_type = output_type
        self.forecast_strategy = forecast_strategy
        self.num_samples = num_samples
        self.aggregation = aggregation
        self._device = torch.device('cpu')
        if not network_structure.variable_selection:
            self.encoder_lagged_input = network_encoder['block_1'].encoder_properties.lagged_input
            self.decoder_lagged_input = network_decoder[first_decoder].decoder_properties.lagged_input
        else:
            self.encoder_lagged_input = False
            self.decoder_lagged_input = False
        if self.encoder_lagged_input:
            self.cached_lag_mask_encoder = None
            self.encoder_lagged_value = network_encoder['block_1'].encoder.lagged_value
        if self.decoder_lagged_input:
            self.cached_lag_mask_decoder = None
            self.decoder_lagged_value = network_decoder[first_decoder].decoder.lagged_value

    @property
    def device(self) ->torch.device:
        return self._device

    @device.setter
    def device(self, device: 'torch.device') ->None:
        self
        self._device = device
        for model in self.lazy_modules:
            model.device = device

    def rescale_output(self, outputs: 'ALL_NET_OUTPUT', loc: 'Optional[torch.Tensor]', scale: 'Optional[torch.Tensor]', device: 'torch.device'=torch.device('cpu')) ->ALL_NET_OUTPUT:
        """
        rescale the network output to its raw scale

        Args:
            outputs (ALL_NET_OUTPUT):
                network head output
            loc (Optional[torch.Tensor]):
                scaling location value
            scale (Optional[torch.Tensor]):
                scaling scale value
            device (torch.device):
                which device the output is stored

        Return:
            ALL_NET_OUTPUT:
                rescaleed network output
        """
        if isinstance(outputs, List):
            return [self.rescale_output(output, loc, scale, device) for output in outputs]
        if loc is not None or scale is not None:
            if isinstance(outputs, torch.distributions.Distribution):
                transform = AffineTransform(loc=0.0 if loc is None else loc, scale=1.0 if scale is None else scale)
                outputs = TransformedDistribution_(outputs, [transform])
            elif loc is None:
                outputs = outputs * scale
            elif scale is None:
                outputs = outputs + loc
            else:
                outputs = outputs * scale + loc
        return outputs

    def scale_value(self, raw_value: 'torch.Tensor', loc: 'Optional[torch.Tensor]', scale: 'Optional[torch.Tensor]', device: 'torch.device'=torch.device('cpu')) ->torch.Tensor:
        """
        scale the outputs

        Args:
            raw_value (torch.Tensor):
                network head output
            loc (Optional[torch.Tensor]):
                scaling location value
            scale (Optional[torch.Tensor]):
                scaling scale value
            device (torch.device):
                which device the output is stored

        Return:
            torch.Tensor:
                scaled input value
        """
        if loc is not None or scale is not None:
            if loc is None:
                outputs = raw_value / scale
            elif scale is None:
                outputs = raw_value - loc
            else:
                outputs = (raw_value - loc) / scale
            return outputs
        else:
            return raw_value

    @abstractmethod
    def forward(self, past_targets: 'torch.Tensor', future_targets: 'Optional[torch.Tensor]'=None, past_features: 'Optional[torch.Tensor]'=None, future_features: 'Optional[torch.Tensor]'=None, past_observed_targets: 'Optional[torch.BoolTensor]'=None, decoder_observed_values: 'Optional[torch.Tensor]'=None) ->ALL_NET_OUTPUT:
        raise NotImplementedError

    @abstractmethod
    def pred_from_net_output(self, net_output: 'ALL_NET_OUTPUT') ->torch.Tensor:
        """
        This function is applied to transform the network head output to torch tensor to create the point prediction

        Args:
            net_output (ALL_NET_OUTPUT):
                network head output

        Return:
            torch.Tensor:
                point prediction
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, past_targets: 'torch.Tensor', past_features: 'Optional[torch.Tensor]'=None, future_features: 'Optional[torch.Tensor]'=None, past_observed_targets: 'Optional[torch.Tensor]'=None) ->torch.Tensor:
        raise NotImplementedError

    def repeat_intermediate_values(self, intermediate_values: 'List[Optional[Union[torch.Tensor, Tuple[torch.Tensor]]]]', is_hidden_states: 'List[bool]', repeats: 'int') ->List[Optional[Union[torch.Tensor, Tuple[torch.Tensor]]]]:
        """
        This function is often applied for auto-regressive model where we sample multiple points to form several
        trajectories and we need to repeat the intermediate values to ensure that the batch sizes match

        Args:
             intermediate_values (List[Optional[Union[torch.Tensor, Tuple[torch.Tensor]]]])
                a list of intermediate values to be repeated
             is_hidden_states  (List[bool]):
                if the intermediate_value is hidden states in RNN-form network, we need to consider the
                hidden states differently
            repeats (int):
                number of repeats

        Return:
            List[Optional[Union[torch.Tensor, Tuple[torch.Tensor]]]]:
                repeated values
        """
        for i, (is_hx, inter_value) in enumerate(zip(is_hidden_states, intermediate_values)):
            if isinstance(inter_value, torch.Tensor):
                repeated_value = inter_value.repeat_interleave(repeats=repeats, dim=1 if is_hx else 0)
                intermediate_values[i] = repeated_value
            elif isinstance(inter_value, tuple):
                dim = 1 if is_hx else 0
                repeated_value = tuple(hx.repeat_interleave(repeats=repeats, dim=dim) for hx in inter_value)
                intermediate_values[i] = repeated_value
        return intermediate_values

    def pad_tensor(self, tensor_to_be_padded: 'torch.Tensor', target_length: 'int') ->torch.Tensor:
        """
        pad tensor to meet the required length

        Args:
             tensor_to_be_padded (torch.Tensor)
                tensor to be padded
             target_length  (int):
                target length

        Return:
            torch.Tensor:
                padded tensors
        """
        tensor_shape = tensor_to_be_padded.shape
        padding_size = [tensor_shape[0], target_length - tensor_shape[1], tensor_shape[-1]]
        tensor_to_be_padded = torch.cat([tensor_to_be_padded.new_zeros(padding_size), tensor_to_be_padded], dim=1)
        return tensor_to_be_padded


def get_lagged_subsequences(sequence: 'torch.Tensor', subsequences_length: 'int', lags_seq: 'Optional[List[int]]'=None, mask: 'Optional[torch.Tensor]'=None) ->Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Returns lagged subsequences of a given sequence, this allows the model to receive the input from the past targets
    outside the sliding windows. This implementation is similar to gluonTS's implementation
     the only difference is that we pad the sequence that is not long enough

    Args:
        sequence (torch.Tensor):
            the sequence from which lagged subsequences should be extracted, Shape: (N, T, C).
        subsequences_length (int):
            length of the subsequences to be extracted.
        lags_seq (Optional[List[int]]):
            lags of the sequence, indicating the sequence that needs to be extracted
        mask (Optional[torch.Tensor]):
            a mask tensor indicating, it is a cached mask tensor that allows the model to quickly extract the desired
            lagged values

    Returns:
        lagged (Tensor)
            A tensor of shape (N, S, I * C), where S = subsequences_length and I = len(indices),
             containing lagged subsequences.
        mask (torch.Tensor):
            cached mask
    """
    batch_size = sequence.shape[0]
    num_features = sequence.shape[2]
    if mask is None:
        if lags_seq is None:
            warnings.warn('Neither lag_mask or lags_seq is given, we simply return the input value')
            return sequence, None
        num_lags = len(lags_seq)
        mask_length = max(lags_seq) + subsequences_length
        mask = torch.zeros((num_lags, mask_length), dtype=torch.bool)
        for i, lag_index in enumerate(lags_seq):
            begin_index = -lag_index - subsequences_length
            end_index = -lag_index if lag_index > 0 else None
            mask[i, begin_index:end_index] = True
    else:
        num_lags = mask.shape[0]
        mask_length = mask.shape[1]
    mask_extend = mask.clone()
    if mask_length > sequence.shape[1]:
        sequence = torch.cat([sequence.new_zeros([batch_size, mask_length - sequence.shape[1], num_features]), sequence], dim=1)
    elif mask_length < sequence.shape[1]:
        mask_extend = torch.cat([mask.new_zeros([num_lags, sequence.shape[1] - mask_length]), mask_extend], dim=1)
    sequence = sequence.unsqueeze(1)
    mask_extend = mask_extend.unsqueeze(-1)
    lagged_seq = torch.masked_select(sequence, mask_extend).reshape(batch_size, num_lags, subsequences_length, -1)
    lagged_seq = torch.transpose(lagged_seq, 1, 2).reshape(batch_size, subsequences_length, -1)
    return lagged_seq, mask


class ForecastingNet(AbstractForecastingNet):

    def pre_processing(self, past_targets: 'torch.Tensor', past_observed_targets: 'torch.BoolTensor', past_features: 'Optional[torch.Tensor]'=None, future_features: 'Optional[torch.Tensor]'=None, length_past: 'int'=0, length_future: 'int'=0, variable_selector_kwargs: 'Dict'={}) ->Tuple[torch.Tensor, ...]:
        if self.encoder_lagged_input:
            if self.window_size < past_targets.shape[1]:
                past_targets[:, -self.window_size:], _, loc, scale = self.target_scaler(past_targets[:, -self.window_size:], past_observed_targets[:, -self.window_size:])
                past_targets[:, :-self.window_size] = torch.where(past_observed_targets[:, :-self.window_size], self.scale_value(past_targets[:, :-self.window_size], loc, scale), past_targets[:, :-self.window_size])
            else:
                past_targets, _, loc, scale = self.target_scaler(past_targets, past_observed_targets)
            truncated_past_targets, self.cached_lag_mask_encoder = get_lagged_subsequences(past_targets, self.window_size, self.encoder_lagged_value, self.cached_lag_mask_encoder)
        else:
            if self.window_size < past_targets.shape[1]:
                past_targets = past_targets[:, -self.window_size:]
                past_observed_targets = past_observed_targets[:, -self.window_size:]
            past_targets, _, loc, scale = self.target_scaler(past_targets, past_observed_targets)
            truncated_past_targets = past_targets
        if past_features is not None:
            if self.window_size <= past_features.shape[1]:
                past_features = past_features[:, -self.window_size:]
            elif self.encoder_lagged_input:
                past_features = self.pad_tensor(past_features, self.window_size)
        if self.network_structure.variable_selection:
            batch_size = truncated_past_targets.shape[0]
            feat_dict_static = {}
            if length_past > 0:
                if past_features is not None:
                    past_features = self.embedding(past_features)
                feat_dict_past = {'past_targets': truncated_past_targets}
                if past_features is not None:
                    for feature_name in self.variable_selector.feature_names:
                        tensor_idx = self.variable_selector.feature_names2tensor_idx[feature_name]
                        if feature_name not in self.variable_selector.static_features:
                            feat_dict_past[feature_name] = past_features[:, :, tensor_idx[0]:tensor_idx[1]]
                        else:
                            static_feature = past_features[:, 0, tensor_idx[0]:tensor_idx[1]]
                            feat_dict_static[feature_name] = static_feature
                if hasattr(self.variable_selector, 'placeholder_features'):
                    for placehold in self.variable_selector.placeholder_features:
                        feat_dict_past[placehold] = torch.zeros((batch_size, length_past, 1), dtype=past_targets.dtype, device=self.device)
            else:
                feat_dict_past = None
            if length_future > 0:
                if future_features is not None:
                    future_features = self.decoder_embedding(future_features)
                feat_dict_future = {}
                if hasattr(self.variable_selector, 'placeholder_features'):
                    for placehold in self.variable_selector.placeholder_features:
                        feat_dict_future[placehold] = torch.zeros((batch_size, length_future, 1), dtype=past_targets.dtype, device=self.device)
                if future_features is not None:
                    for feature_name in self.variable_selector.known_future_features:
                        tensor_idx = self.variable_selector.future_feature_name2tensor_idx[feature_name]
                        if feature_name not in self.variable_selector.static_features:
                            feat_dict_future[feature_name] = future_features[:, :, tensor_idx[0]:tensor_idx[1]]
                        elif length_past == 0:
                            static_feature = future_features[:, 0, tensor_idx[0]:tensor_idx[1]]
                            feat_dict_static[feature_name] = static_feature
            else:
                feat_dict_future = None
            x_past, x_future, x_static, static_context_initial_hidden = self.variable_selector(x_past=feat_dict_past, x_future=feat_dict_future, x_static=feat_dict_static, batch_size=batch_size, length_past=length_past, length_future=length_future, **variable_selector_kwargs)
            return x_past, x_future, x_static, loc, scale, static_context_initial_hidden, past_targets
        else:
            if past_features is not None:
                x_past = torch.cat([truncated_past_targets, past_features], dim=-1)
                x_past = self.embedding(x_past)
            else:
                x_past = self.embedding(truncated_past_targets)
            if future_features is not None and length_future > 0:
                future_features = self.decoder_embedding(future_features)
            return x_past, future_features, None, loc, scale, None, past_targets

    def forward(self, past_targets: 'torch.Tensor', future_targets: 'Optional[torch.Tensor]'=None, past_features: 'Optional[torch.Tensor]'=None, future_features: 'Optional[torch.Tensor]'=None, past_observed_targets: 'Optional[torch.BoolTensor]'=None, decoder_observed_values: 'Optional[torch.Tensor]'=None) ->ALL_NET_OUTPUT:
        x_past, x_future, x_static, loc, scale, static_context_initial_hidden, _ = self.pre_processing(past_targets=past_targets, past_observed_targets=past_observed_targets, past_features=past_features, future_features=future_features, length_past=min(self.window_size, past_targets.shape[1]), length_future=self.n_prediction_steps)
        encoder_additional = [static_context_initial_hidden]
        encoder_additional.extend([None] * (self.network_structure.num_blocks - 1))
        encoder2decoder, encoder_output = self.encoder(encoder_input=x_past, additional_input=encoder_additional)
        decoder_output = self.decoder(x_future=x_future, encoder_output=encoder2decoder, pos_idx=(x_past.shape[1], x_past.shape[1] + self.n_prediction_steps))
        if self.has_temporal_fusion:
            decoder_output = self.temporal_fusion(encoder_output=encoder_output, decoder_output=decoder_output, past_observed_targets=past_observed_targets, decoder_length=self.n_prediction_steps, static_embedding=x_static)
        output = self.head(decoder_output)
        return self.rescale_output(output, loc, scale, self.device)

    def pred_from_net_output(self, net_output: 'ALL_NET_OUTPUT') ->torch.Tensor:
        if self.output_type == 'regression':
            return net_output
        elif self.output_type == 'quantile':
            return net_output[0]
        elif self.output_type == 'distribution':
            if self.forecast_strategy == 'mean':
                if isinstance(net_output, list):
                    return torch.cat([dist.mean for dist in net_output], dim=-2)
                else:
                    return net_output.mean
            elif self.forecast_strategy == 'sample':
                if isinstance(net_output, list):
                    samples = torch.cat([dist.sample((self.num_samples,)) for dist in net_output], dim=-2)
                else:
                    samples = net_output.sample((self.num_samples,))
                if self.aggregation == 'mean':
                    return torch.mean(samples, dim=0)
                elif self.aggregation == 'median':
                    return torch.median(samples, 0)[0]
                else:
                    raise NotImplementedError(f'Unknown aggregation: {self.aggregation}')
            else:
                raise NotImplementedError(f'Unknown forecast_strategy: {self.forecast_strategy}')
        else:
            raise NotImplementedError(f'Unknown output_type: {self.output_type}')

    def predict(self, past_targets: 'torch.Tensor', past_features: 'Optional[torch.Tensor]'=None, future_features: 'Optional[torch.Tensor]'=None, past_observed_targets: 'Optional[torch.BoolTensor]'=None) ->torch.Tensor:
        net_output = self(past_targets=past_targets, past_features=past_features, future_features=future_features, past_observed_targets=past_observed_targets)
        return self.pred_from_net_output(net_output)


def get_lagged_subsequences_inference(sequence: 'torch.Tensor', subsequences_length: 'int', lags_seq: 'List[int]') ->torch.Tensor:
    """
    this function works exactly the same as get_lagged_subsequences. However, this implementation is faster when no
    cached value is available, thus it is applied during inference times.

    Args:
        sequence (torch.Tensor):
            the sequence from which lagged subsequences should be extracted, Shape: (N, T, C).
        subsequences_length (int):
            length of the subsequences to be extracted.
        lags_seq (Optional[List[int]]):
            lags of the sequence, indicating the sequence that needs to be extracted

    Returns:
        lagged (Tensor)
            A tensor of shape (N, S, I * C), where S = subsequences_length and I = len(indices),
             containing lagged subsequences.
    """
    sequence_length = sequence.shape[1]
    batch_size = sequence.shape[0]
    lagged_values = []
    for lag_index in lags_seq:
        begin_index = -lag_index - subsequences_length
        end_index = -lag_index if lag_index > 0 else None
        if end_index is not None and end_index < -sequence_length:
            lagged_values.append(torch.zeros([batch_size, subsequences_length, *sequence.shape[2:]]))
            continue
        if begin_index < -sequence_length:
            if end_index is not None:
                pad_shape = [batch_size, subsequences_length - sequence_length - end_index, *sequence.shape[2:]]
                lagged_values.append(torch.cat([torch.zeros(pad_shape), sequence[:, :end_index, ...]], dim=1))
            else:
                pad_shape = [batch_size, subsequences_length - sequence_length, *sequence.shape[2:]]
                lagged_values.append(torch.cat([torch.zeros(pad_shape), sequence], dim=1))
            continue
        else:
            lagged_values.append(sequence[:, begin_index:end_index, ...])
    lagged_seq = torch.stack(lagged_values, -1).transpose(-1, -2).reshape(batch_size, subsequences_length, -1)
    return lagged_seq


class ForecastingSeq2SeqNet(ForecastingNet):
    future_target_required = True
    """
    Forecasting network with Seq2Seq structure, Encoder/ Decoder need to be the same recurrent models while

    This structure is activate when the decoder is recurrent (RNN or transformer).
    We train the network with teacher forcing, thus
    future_targets is required for the network. To train the network, past targets and past features are fed to the
    encoder to obtain the hidden states whereas future targets and future features.
    When the output type is distribution and forecast_strategy is sampling,
    this model is equivalent to a deepAR model during inference.
    """

    def decoder_select_variable(self, future_targets: 'torch.tensor', future_features: 'Optional[torch.Tensor]') ->torch.Tensor:
        batch_size = future_targets.shape[0]
        length_future = future_targets.shape[1]
        future_targets = future_targets
        if future_features is not None:
            future_features = self.decoder_embedding(future_features)
        feat_dict_future = {}
        if hasattr(self.variable_selector, 'placeholder_features'):
            for placeholder in self.variable_selector.placeholder_features:
                feat_dict_future[placeholder] = torch.zeros((batch_size, length_future, 1), dtype=future_targets.dtype, device=self.device)
        for feature_name in self.variable_selector.known_future_features:
            tensor_idx = self.variable_selector.future_feature_name2tensor_idx[feature_name]
            if feature_name not in self.variable_selector.static_features:
                feat_dict_future[feature_name] = future_features[:, :, tensor_idx[0]:tensor_idx[1]]
        feat_dict_future['future_prediction'] = future_targets
        _, x_future, _, _ = self.variable_selector(x_past=None, x_future=feat_dict_future, x_static=None, length_past=0, length_future=length_future, batch_size=batch_size, use_cached_static_contex=True)
        return x_future

    def forward(self, past_targets: 'torch.Tensor', future_targets: 'Optional[torch.Tensor]'=None, past_features: 'Optional[torch.Tensor]'=None, future_features: 'Optional[torch.Tensor]'=None, past_observed_targets: 'Optional[torch.BoolTensor]'=None, decoder_observed_values: 'Optional[torch.Tensor]'=None) ->ALL_NET_OUTPUT:
        x_past, _, x_static, loc, scale, static_context_initial_hidden, past_targets = self.pre_processing(past_targets=past_targets, past_observed_targets=past_observed_targets, past_features=past_features, future_features=future_features, length_past=min(self.window_size, past_targets.shape[1]), length_future=0, variable_selector_kwargs={'cache_static_contex': True})
        encoder_additional = [static_context_initial_hidden]
        encoder_additional.extend([None] * (self.network_structure.num_blocks - 1))
        if self.training:
            future_targets = self.scale_value(future_targets, loc, scale)
            if self.decoder_lagged_input:
                future_targets = torch.cat([past_targets, future_targets[:, :-1, :]], dim=1)
                future_targets, self.cached_lag_mask_decoder = get_lagged_subsequences(future_targets, self.n_prediction_steps, self.decoder_lagged_value, self.cached_lag_mask_decoder)
            else:
                future_targets = torch.cat([past_targets[:, [-1], :], future_targets[:, :-1, :]], dim=1)
            if self.network_structure.variable_selection:
                decoder_input = self.decoder_select_variable(future_targets, future_features)
            else:
                decoder_input = future_targets if future_features is None else torch.cat([future_features, future_targets], dim=-1)
                decoder_input = decoder_input
                decoder_input = self.decoder_embedding(decoder_input)
            encoder2decoder, encoder_output = self.encoder(encoder_input=x_past, additional_input=encoder_additional)
            decoder_output = self.decoder(x_future=decoder_input, encoder_output=encoder2decoder, pos_idx=(x_past.shape[1], x_past.shape[1] + self.n_prediction_steps))
            if self.has_temporal_fusion:
                decoder_output = self.temporal_fusion(encoder_output=encoder_output, decoder_output=decoder_output, past_observed_targets=past_observed_targets, decoder_length=self.n_prediction_steps, static_embedding=x_static)
            net_output = self.head(decoder_output)
            return self.rescale_output(net_output, loc, scale, self.device)
        else:
            encoder2decoder, encoder_output = self.encoder(encoder_input=x_past, additional_input=encoder_additional)
            if self.has_temporal_fusion:
                decoder_output_all: 'Optional[torch.Tensor]' = None
            if self.forecast_strategy != 'sample':
                all_predictions = []
                predicted_target = past_targets[:, [-1]]
                past_targets = past_targets[:, :-1]
                for idx_pred in range(self.n_prediction_steps):
                    predicted_target = predicted_target.cpu()
                    if self.decoder_lagged_input:
                        past_targets = torch.cat([past_targets, predicted_target], dim=1)
                        ar_future_target = get_lagged_subsequences_inference(past_targets, 1, self.decoder_lagged_value)
                    else:
                        ar_future_target = predicted_target[:, [-1]]
                    if self.network_structure.variable_selection:
                        decoder_input = self.decoder_select_variable(future_targets=predicted_target[:, -1:], future_features=future_features[:, [idx_pred]] if future_features is not None else None)
                    else:
                        decoder_input = ar_future_target if future_features is None else torch.cat([future_features[:, [idx_pred]], ar_future_target], dim=-1)
                        decoder_input = decoder_input
                        decoder_input = self.decoder_embedding(decoder_input)
                    decoder_output = self.decoder(decoder_input, encoder_output=encoder2decoder, pos_idx=(x_past.shape[1] + idx_pred, x_past.shape[1] + idx_pred + 1), cache_intermediate_state=True, incremental_update=idx_pred > 0)
                    if self.has_temporal_fusion:
                        if decoder_output_all is not None:
                            decoder_output_all = torch.cat([decoder_output_all, decoder_output], dim=1)
                        else:
                            decoder_output_all = decoder_output
                        decoder_output = self.temporal_fusion(encoder_output=encoder_output, decoder_output=decoder_output_all, past_observed_targets=past_observed_targets, decoder_length=idx_pred + 1, static_embedding=x_static)[:, -1:]
                    net_output = self.head(decoder_output)
                    predicted_target = torch.cat([predicted_target, self.pred_from_net_output(net_output).cpu()], dim=1)
                    all_predictions.append(net_output)
                if self.output_type == 'regression':
                    all_predictions = torch.cat(all_predictions, dim=1)
                elif self.output_type == 'quantile':
                    all_predictions = torch.cat([self.pred_from_net_output(pred) for pred in all_predictions], dim=1)
                else:
                    all_predictions = self.pred_from_net_output(all_predictions)
                return self.rescale_output(all_predictions, loc, scale, self.device)
            else:
                batch_size = past_targets.shape[0]
                encoder2decoder = self.repeat_intermediate_values(encoder2decoder, is_hidden_states=self.encoder.encoder_has_hidden_states, repeats=self.num_samples)
                if self.has_temporal_fusion:
                    intermediate_values = self.repeat_intermediate_values([encoder_output, past_observed_targets], is_hidden_states=[False, False], repeats=self.num_samples)
                    encoder_output = intermediate_values[0]
                    past_observed_targets = intermediate_values[1]
                if self.decoder_lagged_input:
                    max_lag_seq_length = max(self.decoder_lagged_value) + 1
                else:
                    max_lag_seq_length = 1 + self.window_size
                repeated_past_target = past_targets[:, -max_lag_seq_length:].repeat_interleave(repeats=self.num_samples, dim=0).squeeze(1)
                repeated_predicted_target = repeated_past_target[:, [-1]]
                repeated_past_target = repeated_past_target[:, :-1]
                repeated_x_static = x_static.repeat_interleave(repeats=self.num_samples, dim=0) if x_static is not None else None
                repeated_future_features = future_features.repeat_interleave(repeats=self.num_samples, dim=0) if future_features is not None else None
                if self.network_structure.variable_selection:
                    self.variable_selector.cached_static_contex = self.repeat_intermediate_values([self.variable_selector.cached_static_contex], is_hidden_states=[False], repeats=self.num_samples)[0]
                for idx_pred in range(self.n_prediction_steps):
                    if self.decoder_lagged_input:
                        ar_future_target = torch.cat([repeated_past_target, repeated_predicted_target.cpu()], dim=1)
                        ar_future_target = get_lagged_subsequences_inference(ar_future_target, 1, self.decoder_lagged_value)
                    else:
                        ar_future_target = repeated_predicted_target[:, [-1]]
                    if self.network_structure.variable_selection:
                        decoder_input = self.decoder_select_variable(future_targets=ar_future_target, future_features=None if repeated_future_features is None else repeated_future_features[:, [idx_pred]])
                    else:
                        decoder_input = ar_future_target if repeated_future_features is None else torch.cat([repeated_future_features[:, [idx_pred], :], ar_future_target], dim=-1)
                        decoder_input = decoder_input
                        decoder_input = self.decoder_embedding(decoder_input)
                    decoder_output = self.decoder(decoder_input, encoder_output=encoder2decoder, pos_idx=(x_past.shape[1] + idx_pred, x_past.shape[1] + idx_pred + 1), cache_intermediate_state=True, incremental_update=idx_pred > 0)
                    if self.has_temporal_fusion:
                        if decoder_output_all is not None:
                            decoder_output_all = torch.cat([decoder_output_all, decoder_output], dim=1)
                        else:
                            decoder_output_all = decoder_output
                        decoder_output = self.temporal_fusion(encoder_output=encoder_output, decoder_output=decoder_output_all, past_observed_targets=past_observed_targets, decoder_length=idx_pred + 1, static_embedding=repeated_x_static)[:, -1:]
                    net_output = self.head(decoder_output)
                    samples = net_output.sample().cpu()
                    repeated_predicted_target = torch.cat([repeated_predicted_target, samples], dim=1)
                all_predictions = repeated_predicted_target[:, 1:].unflatten(0, (batch_size, self.num_samples))
                if self.aggregation == 'mean':
                    return self.rescale_output(torch.mean(all_predictions, dim=1), loc, scale)
                elif self.aggregation == 'median':
                    return self.rescale_output(torch.median(all_predictions, dim=1)[0], loc, scale)
                else:
                    raise ValueError(f'Unknown aggregation: {self.aggregation}')

    def predict(self, past_targets: 'torch.Tensor', past_features: 'Optional[torch.Tensor]'=None, future_features: 'Optional[torch.Tensor]'=None, past_observed_targets: 'Optional[torch.BoolTensor]'=None) ->torch.Tensor:
        net_output = self(past_targets=past_targets, past_features=past_features, future_features=future_features, past_observed_targets=past_observed_targets)
        if self.output_type == 'regression':
            return self.pred_from_net_output(net_output)
        else:
            return net_output


class ForecastingDeepARNet(ForecastingSeq2SeqNet):
    future_target_required = True

    def __init__(self, **kwargs: Any):
        """
        Forecasting network with DeepAR structure.

        This structure is activate when the decoder is not recurrent (MLP) and its hyperparameter "auto_regressive" is
        set  as True. We train the network to let it do a one-step prediction. This structure is compatible with any
         sorts of encoder (except MLP).
        """
        super(ForecastingDeepARNet, self).__init__(**kwargs)
        self.encoder_bijective_seq_output = kwargs['network_encoder']['block_1'].encoder_properties.bijective_seq_output
        self.cached_lag_mask_encoder_test = None
        self.only_generate_future_dist = False

    def train(self, mode: 'bool'=True) ->nn.Module:
        self.only_generate_future_dist = False
        return super().train(mode=mode)

    def encoder_select_variable(self, past_targets: 'torch.tensor', past_features: 'Optional[torch.Tensor]', length_past: 'int', **variable_selector_kwargs: Any) ->Tuple[torch.Tensor, Tuple[torch.Tensor]]:
        batch_size = past_targets.shape[0]
        past_targets = past_targets
        if past_features is not None:
            past_features = past_features
            past_features = self.embedding(past_features)
        feat_dict_past = {'past_targets': past_targets}
        feat_dict_static = {}
        if hasattr(self.variable_selector, 'placeholder_features'):
            for placehold in self.variable_selector.placeholder_features:
                feat_dict_past[placehold] = torch.zeros((batch_size, length_past, 1), dtype=past_targets.dtype, device=self.device)
        for feature_name in self.variable_selector.feature_names:
            tensor_idx = self.variable_selector.feature_names2tensor_idx[feature_name]
            if feature_name not in self.variable_selector.static_features:
                feat_dict_past[feature_name] = past_features[:, :, tensor_idx[0]:tensor_idx[1]]
            else:
                static_feature = past_features[:, 0, tensor_idx[0]:tensor_idx[1]]
                feat_dict_static[feature_name] = static_feature
        x_past, _, _, static_context_initial_hidden = self.variable_selector(x_past=feat_dict_past, x_future=None, x_static=feat_dict_static, length_past=length_past, length_future=0, batch_size=batch_size, **variable_selector_kwargs)
        return x_past, static_context_initial_hidden

    def forward(self, past_targets: 'torch.Tensor', future_targets: 'Optional[torch.Tensor]'=None, past_features: 'Optional[torch.Tensor]'=None, future_features: 'Optional[torch.Tensor]'=None, past_observed_targets: 'Optional[torch.BoolTensor]'=None, decoder_observed_values: 'Optional[torch.Tensor]'=None) ->ALL_NET_OUTPUT:
        encode_length = min(self.window_size, past_targets.shape[1])
        if past_observed_targets is None:
            past_observed_targets = torch.ones_like(past_targets, dtype=torch.bool)
        if self.training:
            if self.encoder_lagged_input:
                if self.window_size < past_targets.shape[1]:
                    past_targets[:, -self.window_size:], _, loc, scale = self.target_scaler(past_targets[:, -self.window_size:], past_observed_targets[:, -self.window_size:])
                    past_targets[:, :-self.window_size] = torch.where(past_observed_targets[:, :-self.window_size], self.scale_value(past_targets[:, :-self.window_size], loc, scale), past_targets[:, :-self.window_size])
                else:
                    past_targets, _, loc, scale = self.target_scaler(past_targets, past_observed_targets)
                future_targets = self.scale_value(future_targets, loc, scale)
                targets_all = torch.cat([past_targets, future_targets[:, :-1]], dim=1)
                seq_length = self.window_size + self.n_prediction_steps
                targets_all, self.cached_lag_mask_encoder = get_lagged_subsequences(targets_all, seq_length - 1, self.encoder_lagged_value, self.cached_lag_mask_encoder)
                targets_all = targets_all[:, -(encode_length + self.n_prediction_steps - 1):]
            else:
                if self.window_size < past_targets.shape[1]:
                    past_targets = past_targets[:, -self.window_size:]
                    past_observed_targets = past_observed_targets[:, -self.window_size:]
                past_targets, _, loc, scale = self.target_scaler(past_targets, past_observed_targets)
                future_targets = self.scale_value(future_targets, loc, scale)
                targets_all = torch.cat([past_targets, future_targets[:, :-1]], dim=1)
            if self.network_structure.variable_selection:
                if past_features is not None:
                    assert future_features is not None
                    past_features = past_features[:, -self.window_size:]
                    features_all = torch.cat([past_features, future_features[:, :-1]], dim=1)
                else:
                    features_all = None
                length_past = min(self.window_size, past_targets.shape[1]) + self.n_prediction_steps - 1
                encoder_input, static_context_initial_hidden = self.encoder_select_variable(targets_all, past_features=features_all, length_past=length_past)
            else:
                if past_features is not None:
                    assert future_features is not None
                    if self.window_size <= past_features.shape[1]:
                        past_features = past_features[:, -self.window_size:]
                    features_all = torch.cat([past_features, future_features[:, :-1]], dim=1)
                    encoder_input = torch.cat([features_all, targets_all], dim=-1)
                else:
                    encoder_input = targets_all
                encoder_input = encoder_input
                encoder_input = self.embedding(encoder_input)
                static_context_initial_hidden = None
            encoder_additional: 'List[Optional[torch.Tensor]]' = [static_context_initial_hidden]
            encoder_additional.extend([None] * (self.network_structure.num_blocks - 1))
            encoder2decoder, encoder_output = self.encoder(encoder_input=encoder_input, additional_input=encoder_additional, output_seq=True)
            if self.only_generate_future_dist:
                encoder2decoder = [encoder2decoder[-1][:, -self.n_prediction_steps:]]
            net_output = self.head(self.decoder(x_future=None, encoder_output=encoder2decoder))
            return self.rescale_output(net_output, loc, scale, self.device)
        else:
            if self.encoder_lagged_input:
                if self.window_size < past_targets.shape[1]:
                    past_targets[:, -self.window_size:], _, loc, scale = self.target_scaler(past_targets[:, -self.window_size:], past_observed_targets[:, -self.window_size:])
                    past_targets[:, :-self.window_size] = torch.where(past_observed_targets[:, :-self.window_size], self.scale_value(past_targets[:, :-self.window_size], loc, scale), past_targets[:, :-self.window_size])
                else:
                    past_targets, _, loc, scale = self.target_scaler(past_targets, past_observed_targets)
                truncated_past_targets, self.cached_lag_mask_encoder_test = get_lagged_subsequences(past_targets, self.window_size, self.encoder_lagged_value, self.cached_lag_mask_encoder_test)
                truncated_past_targets = truncated_past_targets[:, -encode_length:]
            else:
                if self.window_size < past_targets.shape[1]:
                    past_targets = past_targets[:, -self.window_size:]
                    past_observed_targets = past_observed_targets[:, -self.window_size]
                past_targets, _, loc, scale = self.target_scaler(past_targets, past_observed_targets)
                truncated_past_targets = past_targets
            if self.network_structure.variable_selection:
                if past_features is not None:
                    features_all = past_features[:, -self.window_size:]
                else:
                    features_all = None
                variable_selector_kwargs = dict(cache_static_contex=True, use_cached_static_contex=False)
                encoder_input, static_context_initial_hidden = self.encoder_select_variable(truncated_past_targets, past_features=features_all, length_past=encode_length, **variable_selector_kwargs)
            else:
                if past_features is not None:
                    assert future_features is not None
                    features_all = torch.cat([past_features[:, -encode_length:], future_features[:, :-1]], dim=1)
                else:
                    features_all = None
                encoder_input = truncated_past_targets if features_all is None else torch.cat([features_all[:, :encode_length], truncated_past_targets], dim=-1)
                encoder_input = encoder_input
                encoder_input = self.embedding(encoder_input)
                static_context_initial_hidden = None
            all_samples = []
            batch_size: 'int' = past_targets.shape[0]
            encoder_additional: 'List[Optional[torch.Tensor]]' = [static_context_initial_hidden]
            encoder_additional.extend([None] * (self.network_structure.num_blocks - 1))
            encoder2decoder, encoder_output = self.encoder(encoder_input=encoder_input, additional_input=encoder_additional, cache_intermediate_state=True)
            self.encoder.cached_intermediate_state = self.repeat_intermediate_values(self.encoder.cached_intermediate_state, is_hidden_states=self.encoder.encoder_has_hidden_states, repeats=self.num_samples)
            if self.network_structure.variable_selection:
                self.variable_selector.cached_static_contex = self.repeat_intermediate_values([self.variable_selector.cached_static_contex], is_hidden_states=[False], repeats=self.num_samples)[0]
            if self.encoder_lagged_input:
                max_lag_seq_length = max(max(self.encoder_lagged_value), encode_length)
            else:
                max_lag_seq_length = encode_length
            net_output = self.head(self.decoder(x_future=None, encoder_output=encoder2decoder))
            next_sample = net_output.sample(sample_shape=(self.num_samples,))
            next_sample = next_sample.transpose(0, 1).reshape((next_sample.shape[0] * next_sample.shape[1], 1, -1)).cpu()
            all_samples.append(next_sample)
            if self.n_prediction_steps > 1:
                repeated_past_target = past_targets[:, -max_lag_seq_length:].repeat_interleave(repeats=self.num_samples, dim=0).squeeze(1)
                if future_features is not None:
                    future_features = future_features[:, 1:]
                else:
                    future_features = None
                repeated_future_features = future_features.repeat_interleave(repeats=self.num_samples, dim=0) if future_features is not None else None
            for k in range(1, self.n_prediction_steps):
                if self.encoder_lagged_input:
                    repeated_past_target = torch.cat([repeated_past_target, all_samples[-1]], dim=1)
                    ar_future_target = get_lagged_subsequences_inference(repeated_past_target, 1, self.encoder_lagged_value)
                else:
                    ar_future_target = next_sample
                if self.network_structure.variable_selection:
                    length_past = 1
                    variable_selector_kwargs = dict(use_cached_static_contex=True)
                    if repeated_future_features is not None:
                        feature_next = repeated_future_features[:, [k - 1]]
                    else:
                        feature_next = None
                    encoder_input, _ = self.encoder_select_variable(ar_future_target, past_features=feature_next, length_past=1, **variable_selector_kwargs)
                else:
                    if repeated_future_features is not None:
                        encoder_input = torch.cat([repeated_future_features[:, [k - 1]], ar_future_target], dim=-1)
                    else:
                        encoder_input = ar_future_target
                    encoder_input = encoder_input
                    encoder_input = self.embedding(encoder_input)
                encoder2decoder, _ = self.encoder(encoder_input=encoder_input, additional_input=[None] * self.network_structure.num_blocks, output_seq=False, cache_intermediate_state=True, incremental_update=True)
                net_output = self.head(self.decoder(x_future=None, encoder_output=encoder2decoder))
                next_sample = net_output.sample().cpu()
                all_samples.append(next_sample)
            all_predictions = torch.cat(all_samples, dim=1).unflatten(0, (batch_size, self.num_samples))
            if not self.output_type == 'distribution' and self.forecast_strategy == 'sample':
                raise ValueError(f'A DeepAR network must have output type as Distribution and forecast_strategy as sample,but this network has {self.output_type} and {self.forecast_strategy}')
            if self.aggregation == 'mean':
                return self.rescale_output(torch.mean(all_predictions, dim=1), loc, scale)
            elif self.aggregation == 'median':
                return self.rescale_output(torch.median(all_predictions, dim=1)[0], loc, scale)
            else:
                raise ValueError(f'Unknown aggregation: {self.aggregation}')

    def predict(self, past_targets: 'torch.Tensor', past_features: 'Optional[torch.Tensor]'=None, future_features: 'Optional[torch.Tensor]'=None, past_observed_targets: 'Optional[torch.BoolTensor]'=None) ->torch.Tensor:
        net_output = self(past_targets=past_targets, past_features=past_features, future_features=future_features, past_observed_targets=past_observed_targets)
        return net_output


class NBEATSNet(ForecastingNet):
    future_target_required = False

    def forward(self, past_targets: 'torch.Tensor', future_targets: 'Optional[torch.Tensor]'=None, past_features: 'Optional[torch.Tensor]'=None, future_features: 'Optional[torch.Tensor]'=None, past_observed_targets: 'Optional[torch.BoolTensor]'=None, decoder_observed_values: 'Optional[torch.Tensor]'=None) ->Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if past_observed_targets is None:
            past_observed_targets = torch.ones_like(past_targets, dtype=torch.bool)
        if self.window_size <= past_targets.shape[1]:
            past_targets = past_targets[:, -self.window_size:]
            past_observed_targets = past_observed_targets[:, -self.window_size:]
        else:
            past_targets = self.pad_tensor(past_targets, self.window_size)
        past_targets, _, loc, scale = self.target_scaler(past_targets, past_observed_targets)
        past_targets = past_targets
        batch_size = past_targets.shape[0]
        output_shape = past_targets.shape[2:]
        forcast_shape = [batch_size, self.n_prediction_steps, *output_shape]
        forecast = torch.zeros(forcast_shape).flatten(1)
        backcast, _ = self.encoder(past_targets, [None])
        backcast = backcast[0]
        for block in self.decoder.decoder['block_1']:
            backcast_block, forecast_block = block([None], backcast)
            backcast = backcast - backcast_block
            forecast = forecast + forecast_block
        backcast = backcast.reshape(past_targets.shape)
        forecast = forecast.reshape(forcast_shape)
        forecast = self.rescale_output(forecast, loc, scale, self.device)
        if self.training:
            backcast = self.rescale_output(backcast, loc, scale, self.device)
            return backcast, forecast
        else:
            return forecast

    def pred_from_net_output(self, net_output: 'torch.Tensor') ->torch.Tensor:
        return net_output


_activations = {'relu': torch.nn.ReLU, 'tanh': torch.nn.Tanh, 'sigmoid': torch.nn.Sigmoid}


class _DenseLayer(nn.Sequential):

    def __init__(self, num_input_features: 'int', activation: 'str', growth_rate: 'int', bn_size: 'int', drop_rate: 'float', bn_args: 'Dict[str, Any]'):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features, **bn_args)),
        self.add_module('relu1', _activations[activation]()),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size * growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate, **bn_args)),
        self.add_module('relu2', _activations[activation]()),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):

    def __init__(self, num_layers: 'int', num_input_features: 'int', activation: 'str', bn_size: 'int', growth_rate: 'int', drop_rate: 'float', bn_args: 'Dict[str, Any]'):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features=num_input_features + i * growth_rate, activation=activation, growth_rate=growth_rate, bn_size=bn_size, drop_rate=drop_rate, bn_args=bn_args)
            self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):

    def __init__(self, num_input_features: 'int', activation: 'str', num_output_features: 'int', pool_size: 'int', bn_args: 'Dict[str, Any]'):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features, **bn_args))
        self.add_module('relu', _activations[activation]())
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features, kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=pool_size, stride=pool_size))


class ShakeDropFunction(Function):
    """
    References:
        Title: ShakeDrop Regularization for Deep Residual Learning
        Authors: Yoshihiro Yamada et. al.
        URL: https://arxiv.org/pdf/1802.02375.pdf
        Title: ShakeDrop Regularization
        Authors: Yoshihiro Yamada et. al.
        URL: https://openreview.net/pdf?id=S1NHaMW0b
        Github URL: https://github.com/owruby/shake-drop_pytorch/blob/master/models/shakedrop.py
    """

    @staticmethod
    def forward(ctx: 'Any', x: 'torch.Tensor', alpha: 'torch.Tensor', beta: 'torch.Tensor', bl: 'torch.Tensor') ->torch.Tensor:
        ctx.save_for_backward(x, alpha, beta, bl)
        y = (bl + alpha - bl * alpha) * x
        return y

    @staticmethod
    def backward(ctx: 'Any', grad_output: 'torch.Tensor') ->Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x, alpha, beta, bl = ctx.saved_tensors
        grad_x = grad_alpha = grad_beta = grad_bl = None
        if ctx.needs_input_grad[0]:
            grad_x = grad_output * (bl + beta - bl * beta)
        return grad_x, grad_alpha, grad_beta, grad_bl


shake_drop = ShakeDropFunction.apply


def shake_drop_get_bl(block_index: 'int', min_prob_no_shake: 'float', num_blocks: 'int', is_training: 'bool', is_cuda: 'bool') ->torch.Tensor:
    """
    The sampling of Bernoulli random variable
    based on Eq. (4) in the paper

    Args:
        block_index (int): The index of the block from the input layer
        min_prob_no_shake (float): The initial shake probability
        num_blocks (int): The total number of building blocks
        is_training (bool): Whether it is training
        is_cuda (bool): Whether the tensor is on CUDA

    Returns:
        bl (torch.Tensor): a Bernoulli random variable in {0, 1}

    Reference:
        ShakeDrop Regularization for Deep Residual Learning
        Yoshihiro Yamada et. al. (2020)
        paper: https://arxiv.org/pdf/1802.02375.pdf
        implementation: https://github.com/imenurok/ShakeDrop
    """
    pl = 1 - (block_index + 1) / num_blocks * (1 - min_prob_no_shake)
    if is_training:
        bl = torch.as_tensor(1.0) if torch.rand(1) <= pl else torch.as_tensor(0.0)
    else:
        bl = torch.as_tensor(pl)
    if is_cuda:
        bl = bl
    return bl


def shake_get_alpha_beta(is_training: 'bool', is_cuda: 'bool') ->Tuple[torch.Tensor, torch.Tensor]:
    """
    The methods used in this function have been introduced in 'ShakeShake Regularisation'
    Currently, this function supports `shake-shake`.

    Args:
        is_training (bool): Whether the computation for the training
        is_cuda (bool): Whether the tensor is on CUDA

    Returns:
        alpha, beta (Tuple[float, float]):
            alpha (in [0, 1]) is the weight coefficient  for the forward pass
            beta (in [0, 1]) is the weight coefficient for the backward pass

    Reference:
        Title: Shake-shake regularization
        Author: Xavier Gastaldi
        URL: https://arxiv.org/abs/1705.07485

    Note:
        The names have been taken from the paper as well.
        Currently, this function supports `shake-shake`.
    """
    if not is_training:
        result = torch.FloatTensor([0.5]), torch.FloatTensor([0.5])
        return result if not is_cuda else (result[0], result[1])
    alpha = torch.rand(1)
    beta = torch.rand(1)
    if is_cuda:
        alpha = alpha
        beta = beta
    return alpha, beta


class ShakeShakeFunction(Function):
    """
    References:
        Title: Shake-Shake regularization
        Authors: Xavier Gastaldi
        URL: https://arxiv.org/pdf/1705.07485.pdf
        Github URL: https://github.com/hysts/pytorch_shake_shake/blob/master/functions/shake_shake_function.py
    """

    @staticmethod
    def forward(ctx: 'Any', x1: 'torch.Tensor', x2: 'torch.Tensor', alpha: 'torch.Tensor', beta: 'torch.Tensor') ->torch.Tensor:
        ctx.save_for_backward(x1, x2, alpha, beta)
        y = x1 * alpha + x2 * (1 - alpha)
        return y

    @staticmethod
    def backward(ctx: 'Any', grad_output: 'torch.Tensor') ->Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x1, x2, alpha, beta = ctx.saved_tensors
        grad_x1 = grad_x2 = grad_alpha = grad_beta = None
        if ctx.needs_input_grad[0]:
            grad_x1 = grad_output * beta
        if ctx.needs_input_grad[1]:
            grad_x2 = grad_output * (1 - beta)
        return grad_x1, grad_x2, grad_alpha, grad_beta


shake_shake = ShakeShakeFunction.apply


class ResBlock(nn.Module):
    """
    __author__ = "Max Dippel, Michael Burkart and Matthias Urban"
    """

    def __init__(self, config: 'Dict[str, Any]', in_features: 'int', out_features: 'int', blocks_per_group: 'int', block_index: 'int', dropout: 'Optional[float]', activation: 'nn.Module'):
        super(ResBlock, self).__init__()
        self.config = config
        self.dropout = dropout
        self.activation = activation
        self.shortcut = None
        self.start_norm: 'Optional[Callable]' = None
        if in_features != out_features:
            self.shortcut = nn.Linear(in_features, out_features)
            self.start_norm = nn.Sequential(nn.BatchNorm1d(in_features), self.activation())
        self.block_index = block_index
        self.num_blocks = blocks_per_group * self.config['num_groups']
        self.layers = self._build_block(in_features, out_features)
        if config['use_shake_shake']:
            self.shake_shake_layers = self._build_block(in_features, out_features)

    def _build_block(self, in_features: 'int', out_features: 'int') ->nn.Module:
        layers = list()
        if self.start_norm is None:
            layers.append(nn.BatchNorm1d(in_features))
            layers.append(self.activation())
        layers.append(nn.Linear(in_features, out_features))
        layers.append(nn.BatchNorm1d(out_features))
        layers.append(self.activation())
        if self.config['use_dropout']:
            layers.append(nn.Dropout(self.dropout))
        layers.append(nn.Linear(out_features, out_features))
        return nn.Sequential(*layers)

    def forward(self, x: 'torch.FloatTensor') ->torch.FloatTensor:
        residual = x
        if self.shortcut is not None and self.start_norm is not None:
            x = self.start_norm(x)
            residual = self.shortcut(x)
        if self.config['use_shake_shake']:
            x1 = self.layers(x)
            x2 = self.shake_shake_layers(x)
            alpha, beta = shake_get_alpha_beta(self.training, x.is_cuda)
            x = shake_shake(x1, x2, alpha, beta)
        else:
            x = self.layers(x)
        if self.config['use_shake_drop']:
            alpha, beta = shake_get_alpha_beta(self.training, x.is_cuda)
            bl = shake_drop_get_bl(self.block_index, 1 - self.config['max_shake_drop_probability'], self.num_blocks, self.training, x.is_cuda)
            x = shake_drop(x, alpha, beta, bl)
        x = x + residual
        return x


class TemporalFusionLayer(nn.Module):
    """
    (Lim et al.
    Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting,
    https://arxiv.org/abs/1912.09363)
    we follow the implementation from pytorch forecasting:
    https://github.com/jdb78/pytorch-forecasting/blob/master/pytorch_forecasting/models/temporal_fusion_transformer/__init__.py
    """

    def __init__(self, window_size: 'int', network_structure: 'NetworkStructure', network_encoder: 'Dict[str, EncoderBlockInfo]', n_decoder_output_features: 'int', d_model: 'int', n_head: 'int', dropout: 'Optional[float]'=None):
        super().__init__()
        num_blocks = network_structure.num_blocks
        last_block = f'block_{num_blocks}'
        n_encoder_output = network_encoder[last_block].encoder_output_shape[-1]
        self.window_size = window_size
        if n_decoder_output_features != n_encoder_output:
            self.decoder_proj_layer = nn.Linear(n_decoder_output_features, n_encoder_output, bias=False)
        else:
            self.decoder_proj_layer = None
        if network_structure.variable_selection:
            if network_structure.skip_connection:
                n_encoder_output_first = network_encoder['block_1'].encoder_output_shape[-1]
                self.static_context_enrichment = GatedResidualNetwork(n_encoder_output_first, n_encoder_output_first, n_encoder_output_first, dropout)
                self.enrichment = GatedResidualNetwork(input_size=n_encoder_output, hidden_size=n_encoder_output, output_size=d_model, dropout=dropout, context_size=n_encoder_output_first, residual=False)
                self.enrich_with_static = True
        if not hasattr(self, 'enrichment'):
            self.enrichment = GatedResidualNetwork(input_size=n_encoder_output, hidden_size=n_encoder_output, output_size=d_model, dropout=dropout, residual=False)
            self.enrich_with_static = False
        self.attention_fusion = InterpretableMultiHeadAttention(d_model=d_model, n_head=n_head, dropout=dropout or 0.0)
        self.post_attn_gate_norm = GateAddNorm(d_model, dropout=dropout, trainable_add=False)
        self.pos_wise_ff = GatedResidualNetwork(input_size=d_model, hidden_size=d_model, output_size=d_model, dropout=dropout)
        self.network_structure = network_structure
        if network_structure.skip_connection:
            if network_structure.skip_connection_type == 'add':
                self.residual_connection = AddLayer(d_model, n_encoder_output)
            elif network_structure.skip_connection_type == 'gate_add_norm':
                self.residual_connection = GateAddNorm(d_model, skip_size=n_encoder_output, dropout=None, trainable_add=False)
        self._device = 'cpu'

    def forward(self, encoder_output: 'torch.Tensor', decoder_output: 'torch.Tensor', past_observed_targets: 'torch.BoolTensor', decoder_length: 'int', static_embedding: 'Optional[torch.Tensor]'=None) ->torch.Tensor:
        """
        Args:
            encoder_output (torch.Tensor):
                the output of the last layer of encoder network
            decoder_output (torch.Tensor):
                the output of the last layer of decoder network
            past_observed_targets (torch.BoolTensor):
                observed values in the past
            decoder_length (int):
                length of decoder network
            static_embedding Optional[torch.Tensor]:
                embeddings of static features  (if available)
        """
        if self.decoder_proj_layer is not None:
            decoder_output = self.decoder_proj_layer(decoder_output)
        network_output = torch.cat([encoder_output, decoder_output], dim=1)
        if self.enrich_with_static and static_embedding is not None:
            static_context_enrichment = self.static_context_enrichment(static_embedding)
            attn_input = self.enrichment(network_output, static_context_enrichment[:, None].expand(-1, network_output.shape[1], -1))
        else:
            attn_input = self.enrichment(network_output)
        encoder_out_length = encoder_output.shape[1]
        past_observed_targets = past_observed_targets[:, -encoder_out_length:]
        past_observed_targets = past_observed_targets
        mask = self.get_attention_mask(past_observed_targets=past_observed_targets, decoder_length=decoder_length)
        if mask.shape[-1] < attn_input.shape[1]:
            mask = torch.cat([mask.new_full((*mask.shape[:-1], attn_input.shape[1] - mask.shape[-1]), True), mask], dim=-1)
        attn_output, attn_output_weights = self.attention_fusion(q=attn_input[:, -decoder_length:], k=attn_input, v=attn_input, mask=mask)
        attn_output = self.post_attn_gate_norm(attn_output, attn_input[:, -decoder_length:])
        output = self.pos_wise_ff(attn_output)
        if self.network_structure.skip_connection:
            return self.residual_connection(output, decoder_output)
        else:
            return output

    @property
    def device(self) ->torch.device:
        return self._device

    @device.setter
    def device(self, device: 'torch.device') ->None:
        self
        self._device = device

    def get_attention_mask(self, past_observed_targets: 'torch.BoolTensor', decoder_length: 'int') ->torch.Tensor:
        """
        https://github.com/jdb78/pytorch-forecasting/blob/master/pytorch_forecasting/models/
        temporal_fusion_transformer/__init__.py
        """
        attend_step = torch.arange(decoder_length, device=self.device)
        predict_step = torch.arange(0, decoder_length, device=self.device)[:, None]
        decoder_mask = attend_step >= predict_step
        encoder_mask = ~past_observed_targets.squeeze(-1)
        mask = torch.cat((encoder_mask.unsqueeze(1).expand(-1, decoder_length, -1), decoder_mask.unsqueeze(0).expand(encoder_mask.size(0), -1, -1)), dim=2)
        return mask


class PositionalEncoding(nn.Module):
    """https://github.com/pytorch/examples/blob/master/word_language_model/model.py

        NOTE: different from the raw implementation, this model is designed for the batch_first inputs!
        Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \\text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \\text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \\text{where pos is the word position and i is the embed idx)
    Args:
        d_model (int):
            the embed dim (required).
        dropout(float):
            the dropout value (default=0.1).
        max_len(int):
            the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model: 'int', dropout: 'float'=0.1, max_len: 'int'=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: 'torch.Tensor', pos_idx: 'Optional[Tuple[int]]'=None) ->torch.Tensor:
        """Inputs of forward function
        Args:
            x (torch.Tensor(B, L, N)):
                the sequence fed to the positional encoder model (required).
            pos_idx (Tuple[int]):
                position idx indicating the start (first) and end (last) time index of x in a sequence

        Examples:
            >>> output = pos_encoder(x)
        """
        if pos_idx is None:
            x = x + self.pe[:, :x.size(1), :]
        else:
            x = x + self.pe[:, pos_idx[0]:pos_idx[1], :]
        return self.dropout(x)


class DecoderNetwork(nn.Module):

    def forward(self, x_future: 'torch.Tensor', encoder_output: 'torch.Tensor', pos_idx: 'Optional[Tuple[int]]'=None) ->torch.Tensor:
        """
        Base forecasting Decoder Network, its output needs to be a 3-d Tensor:


        Args:
            x_future: torch.Tensor(B, L_future, N_out), the future features
            encoder_output: torch.Tensor(B, L_encoder, N), output of the encoder network, or the hidden states
            pos_idx: positional index, indicating the position of the forecasted tensor, used for transformer
        Returns:
            net_output: torch.Tensor with shape either (B, L_future, N)

        """
        raise NotImplementedError


class EncoderNetwork(nn.Module):

    def forward(self, x: 'torch.Tensor', output_seq: 'bool'=False) ->torch.Tensor:
        """
        Base forecasting network, its output needs to be a 2-d or 3-d Tensor:
        When the decoder is an auto-regressive model, then it needs to output a 3-d Tensor, in which case, output_seq
         needs to be set as True
        When the decoder is a seq2seq model, the network needs to output a 2-d Tensor (B, N), in which case,
        output_seq needs to be set as False

        Args:
            x: torch.Tensor(B, L_in, N)
                input data
            output_seq (bool): if the network outputs a sequence tensor. If it is set True,
                output will be a 3-d Tensor (B, L_out, N). L_out = L_in if encoder_properties['recurrent'] is True.
                If this value is set as False, the network only returns the last item of the sequence.
            hx (Optional[torch.Tensor]): addational input to the network, this could be a hidden states or a sequence
                from previous inputs

        Returns:
            net_output: torch.Tensor with shape either (B, N) or (B, L_out, N)

        """
        raise NotImplementedError

    def get_last_seq_value(self, x: 'torch.Tensor') ->torch.Tensor:
        """
        get the last value of the sequential output
        Args:
            x (torch.Tensor(B, L, N)):
                a sequential value output by the network, usually this value needs to be fed to the decoder
                (or a 2D tensor for a flat encoder)
        Returns:
            output (torch.Tensor(B, 1, M)):
                last element of the sequential value (or a 2D tensor for flat encoder)

        """
        raise NotImplementedError


class TimeSeriesMLP(EncoderNetwork):

    def __init__(self, window_size: 'int', network: 'Optional[nn.Module]'=None):
        """
        Transform the input features (B, T, N) to fit the requirement of MLP
        Args:
            window_size (int): T
            fill_lower_resolution_seq: if sequence with lower resolution needs to be filled with 0
        (for multi-fidelity problems with resolution as fidelity)
        """
        super().__init__()
        self.window_size = window_size
        self.network = network

    def forward(self, x: 'torch.Tensor', output_seq: 'bool'=False) ->torch.Tensor:
        """

        Args:
            x: torch.Tensor(B, L_in, N)
            output_seq (bool), if the MLP outputs a squence, in which case, the input will be rolled to fit the size of
            the network. For Instance if self.window_size = 3, and we obtain a squence with [1, 2, 3, 4, 5]
            the input of this mlp is rolled as :
            [[1, 2, 3]
            [2, 3, 4]
            [3, 4 ,5]]

        Returns:

        """
        if output_seq:
            x = x.unfold(1, self.window_size, 1).transpose(-1, -2)
        elif x.shape[1] > self.window_size:
            x = x[:, -self.window_size:]
        x = x.flatten(-2)
        return x if self.network is None else self.network(x)

    def get_last_seq_value(self, x: 'torch.Tensor') ->torch.Tensor:
        return x


class _InceptionBlock(nn.Module):

    def __init__(self, n_inputs: 'int', n_filters: 'int', kernel_size: 'int', bottleneck: 'int'=None):
        super(_InceptionBlock, self).__init__()
        self.n_filters = n_filters
        self.bottleneck = None if bottleneck is None else nn.Conv1d(n_inputs, bottleneck, kernel_size=1)
        kernel_sizes = [(kernel_size // 2 ** i) for i in range(3)]
        n_inputs = n_inputs if bottleneck is None else bottleneck
        self.pad1 = nn.ConstantPad1d(padding=self._padding(kernel_sizes[0]), value=0)
        self.conv1 = nn.Conv1d(n_inputs, n_filters, kernel_sizes[0])
        self.pad2 = nn.ConstantPad1d(padding=self._padding(kernel_sizes[1]), value=0)
        self.conv2 = nn.Conv1d(n_inputs, n_filters, kernel_sizes[1])
        self.pad3 = nn.ConstantPad1d(padding=self._padding(kernel_sizes[2]), value=0)
        self.conv3 = nn.Conv1d(n_inputs, n_filters, kernel_sizes[2])
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self.convpool = nn.Conv1d(n_inputs, n_filters, 1)
        self.bn = nn.BatchNorm1d(4 * n_filters)

    def _padding(self, kernel_size: 'int') ->Tuple[int, int]:
        if kernel_size % 2 == 0:
            return kernel_size // 2, kernel_size // 2 - 1
        else:
            return kernel_size // 2, kernel_size // 2

    def get_n_outputs(self) ->int:
        return 4 * self.n_filters

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        if self.bottleneck is not None:
            x = self.bottleneck(x)
        x1 = self.conv1(self.pad1(x))
        x2 = self.conv2(self.pad2(x))
        x3 = self.conv3(self.pad3(x))
        x4 = self.convpool(self.maxpool(x))
        x = torch.cat([x1, x2, x3, x4], dim=1)
        x = self.bn(x)
        return torch.relu(x)


class _ResidualBlock(nn.Module):

    def __init__(self, n_res_inputs: 'int', n_outputs: 'int'):
        super(_ResidualBlock, self).__init__()
        self.shortcut = nn.Conv1d(n_res_inputs, n_outputs, 1, bias=False)
        self.bn = nn.BatchNorm1d(n_outputs)

    def forward(self, x: 'torch.Tensor', res: 'torch.Tensor') ->torch.Tensor:
        shortcut = self.shortcut(res)
        shortcut = self.bn(shortcut)
        x = x + shortcut
        return torch.relu(x)


class _InceptionTime(nn.Module):

    def __init__(self, in_features: 'int', config: 'Dict[str, Any]') ->None:
        super().__init__()
        self.config = config
        n_inputs = in_features
        n_filters = self.config['num_filters']
        bottleneck_size = self.config['bottleneck_size']
        kernel_size = self.config['kernel_size']
        n_res_inputs = in_features
        receptive_field = 1
        for i in range(self.config['num_blocks']):
            block = _InceptionBlock(n_inputs=n_inputs, n_filters=n_filters, bottleneck=bottleneck_size, kernel_size=kernel_size)
            receptive_field += max(kernel_size, 3) - 1
            self.__setattr__(f'inception_block_{i}', block)
            if i % 3 == 2:
                n_res_outputs = block.get_n_outputs()
                self.__setattr__(f'residual_block_{i}', _ResidualBlock(n_res_inputs=n_res_inputs, n_outputs=n_res_outputs))
                n_res_inputs = n_res_outputs
            n_inputs = block.get_n_outputs()
        self.receptive_field = receptive_field

    def forward(self, x: 'torch.Tensor', output_seq: 'bool'=False) ->torch.Tensor:
        x = x.transpose(1, 2).contiguous()
        res = x
        for i in range(self.config['num_blocks']):
            x = self.__getattr__(f'inception_block_{i}')(x)
            if i % 3 == 2:
                x = self.__getattr__(f'residual_block_{i}')(x, res)
                res = x
        x = x.transpose(1, 2).contiguous()
        if output_seq:
            return x
        else:
            return self.get_last_seq_value(x)

    def get_last_seq_value(self, x: 'torch.Tensor') ->torch.Tensor:
        return x[:, -1:, :]


class _RNN(EncoderNetwork):

    def __init__(self, in_features: 'int', config: 'Dict[str, Any]', lagged_value: 'Optional[List[int]]'=None):
        super().__init__()
        if lagged_value is None:
            self.lagged_value = [0]
        else:
            self.lagged_value = lagged_value
        self.config = config
        if config['cell_type'] == 'lstm':
            cell_type = nn.LSTM
        else:
            cell_type = nn.GRU
        self.lstm = cell_type(input_size=in_features, hidden_size=config['hidden_size'], num_layers=config['num_layers'], dropout=config.get('dropout', 0.0), bidirectional=config['bidirectional'], batch_first=True)
        self.cell_type = config['cell_type']

    def forward(self, x: 'torch.Tensor', output_seq: 'bool'=False, hx: 'Optional[Tuple[torch.Tensor, torch.Tensor]]'=None) ->Tuple[torch.Tensor, ...]:
        B, T, _ = x.shape
        x, hidden_state = self.lstm(x, hx)
        if output_seq:
            return x, hidden_state
        else:
            return self.get_last_seq_value(x), hidden_state

    def get_last_seq_value(self, x: 'torch.Tensor') ->torch.Tensor:
        B, T, _ = x.shape
        if not self.config['bidirectional']:
            return x[:, -1:]
        else:
            x_by_direction = x.view(B, T, 2, self.config['hidden_size'])
            x = torch.cat([x_by_direction[:, -1, [0], :], x_by_direction[:, 0, [1], :]], dim=-1)
            return x


class _Chomp1d(nn.Module):

    def __init__(self, chomp_size: 'int'):
        super(_Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        return x[:, :, :-self.chomp_size].contiguous()


class _TemporalBlock(nn.Module):

    def __init__(self, n_inputs: 'int', n_outputs: 'int', kernel_size: 'int', stride: 'int', dilation: 'int', padding: 'int', dropout: 'float'=0.2):
        super(_TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = _Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = _Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1, self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def init_weights(self) ->None:
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class _TemporalConvNet(EncoderNetwork):

    def __init__(self, num_inputs: 'int', num_channels: 'List[int]', kernel_size: 'List[int]', dropout: 'float'=0.2):
        super(_TemporalConvNet, self).__init__()
        layers: 'List[Any]' = []
        num_levels = len(num_channels)
        receptive_field = 1
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            stride = 1
            layers += [_TemporalBlock(in_channels, out_channels, kernel_size[i], stride=stride, dilation=dilation_size, padding=(kernel_size[i] - 1) * dilation_size, dropout=dropout)]
            receptive_field_block = 1 + 2 * (kernel_size[i] - 1) * dilation_size
            receptive_field += receptive_field_block
        self.receptive_field = receptive_field
        self.network = nn.Sequential(*layers)

    def forward(self, x: 'torch.Tensor', output_seq: 'bool'=False) ->torch.Tensor:
        x = x.transpose(1, 2).contiguous()
        x = self.network(x)
        x = x.transpose(1, 2).contiguous()
        if output_seq:
            return x
        else:
            return self.get_last_seq_value(x)

    def get_last_seq_value(self, x: 'torch.Tensor') ->torch.Tensor:
        return x[:, -1:]


class _TransformerEncoder(EncoderNetwork):

    def __init__(self, in_features: 'int', d_model: 'int', num_layers: 'int', transformer_encoder_layers: 'nn.Module', use_positional_encoder: 'bool', use_layer_norm_output: 'bool', dropout_pe: 'float'=0.0, layer_norm_eps_output: 'Optional[float]'=None, lagged_value: 'Optional[List[int]]'=None):
        super().__init__()
        if lagged_value is None:
            self.lagged_value = [0]
        else:
            self.lagged_value = lagged_value
        if in_features != d_model:
            input_layer = [nn.Linear(in_features, d_model, bias=False)]
        else:
            input_layer = []
        if use_positional_encoder:
            input_layer.append(PositionalEncoding(d_model, dropout_pe))
        self.input_layer = nn.Sequential(*input_layer)
        self.use_layer_norm_output = use_layer_norm_output
        if use_layer_norm_output:
            norm = nn.LayerNorm(d_model, eps=layer_norm_eps_output)
        else:
            norm = None
        self.transformer_encoder_layers = nn.TransformerEncoder(encoder_layer=transformer_encoder_layers, num_layers=num_layers, norm=norm)

    def forward(self, x: 'torch.Tensor', output_seq: 'bool'=False, mask: 'Optional[torch.Tensor]'=None, src_key_padding_mask: 'Optional[torch.Tensor]'=None) ->torch.Tensor:
        x = self.input_layer(x)
        x = self.transformer_encoder_layers(x)
        if output_seq:
            return x
        else:
            return self.get_last_seq_value(x)

    def get_last_seq_value(self, x: 'torch.Tensor') ->torch.Tensor:
        return x[:, -1:]

