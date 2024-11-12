
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


import functools


import logging


from typing import Any


import torch


import torch.nn.functional as F


from torch.utils.tensorboard import SummaryWriter


from typing import Dict


from typing import Tuple


from torch import nn


from torch.nn import functional as F


from typing import List


from typing import Optional


from typing import Union


import random


import numpy as np


import torch.nn as nn


import re


import copy


import math


import time


from torch import Tensor


from torch.optim.lr_scheduler import ExponentialLR


from typing import TypeVar


import collections


from typing import Callable


from typing import no_type_check


from collections import namedtuple


import torch.distributions as tds


import numpy


from abc import ABC


from typing import NamedTuple


from torch.autograd import Function


from typing import Generic


from typing import Type


from typing import Iterable


from torch.nn.utils import clip_grad_norm_


from torch.optim.lr_scheduler import _LRScheduler


from torch.optim.optimizer import Optimizer


from torch.distributions import Normal


from torch.distributions import Independent


import warnings


from typing import Iterator


from typing import Sequence


from torch.utils.data import Dataset


from typing import Mapping


from torch import __version__ as _torch_version


from torch.utils.data import DataLoader


from torch.utils.data.dataloader import _BaseDataLoaderIter


from torch.utils.data.dataloader import _SingleProcessDataLoaderIter


from torch.utils.data.dataloader import _MultiProcessingDataLoaderIter


from enum import Enum


from typing import ItemsView


from typing import KeysView


from typing import ValuesView


from torch.utils.data import sampler as torch_sampler


from typing import IO


from typing import overload


from collections import defaultdict


from typing import DefaultDict


from torch.optim.lr_scheduler import _LRScheduler as LRScheduler


from typing import Counter


from torch.distributions.distribution import Distribution


from abc import abstractmethod


from torch.distributions import Categorical


from torch.distributions import Gumbel


from typing import Set


import itertools


import types


from enum import auto


from time import time as time_now


from collections import OrderedDict


from collections import Counter


from typing import Counter as CounterType


from typing import TYPE_CHECKING


from collections import deque


from typing import Deque


import inspect


from functools import lru_cache


from typing import Collection


from typing import MutableMapping


from typing import cast


from torch.nn.modules.conv import _ConvNd


class ModelWrapper(nn.Module):

    def __init__(self, model: 'Transformer', beam_width: 'int'):
        super().__init__()
        self.model = model
        self.beam_width = beam_width

    def forward(self, batch: 'tx.data.Batch') ->Dict[str, torch.Tensor]:
        loss = self.model(encoder_input=batch.source, decoder_input=batch.target_input, labels=batch.target_output)
        return {'loss': loss}

    def predict(self, batch: 'tx.data.Batch') ->Dict[str, torch.Tensor]:
        predictions = self.model(encoder_input=batch.source, beam_width=self.beam_width)
        if self.beam_width == 1:
            decoded_ids = predictions[0].sample_id
        else:
            decoded_ids = predictions['sample_id'][:, :, 0]
        return {'preds': decoded_ids}


class SentenceClassifier(nn.Module):

    def __init__(self, vocab_size: 'int', max_seq_length: 'int', emb_dim: 'int', hparams: 'Dict[str, Any]'):
        super().__init__()
        self.embedder = tx.modules.WordEmbedder(vocab_size=vocab_size, hparams=hparams['embedder'])
        self.classifier = tx.modules.Conv1DClassifier(in_channels=max_seq_length, in_features=emb_dim, hparams=hparams['classifier'])

    def forward(self, batch: 'tx.data.Batch') ->Tuple[torch.Tensor, torch.Tensor]:
        logits, pred = self.classifier(self.embedder(batch['sentence_text_ids']))
        loss = F.cross_entropy(logits, batch['label'])
        return pred, loss


class Seq2SeqAttn(nn.Module):

    def __init__(self, train_data):
        super().__init__()
        self.source_vocab_size = train_data.source_vocab.size
        self.target_vocab_size = train_data.target_vocab.size
        self.bos_token_id = train_data.target_vocab.bos_token_id
        self.eos_token_id = train_data.target_vocab.eos_token_id
        self.source_embedder = tx.modules.WordEmbedder(vocab_size=self.source_vocab_size, hparams=config_model.embedder)
        self.target_embedder = tx.modules.WordEmbedder(vocab_size=self.target_vocab_size, hparams=config_model.embedder)
        self.encoder = tx.modules.BidirectionalRNNEncoder(input_size=self.source_embedder.dim, hparams=config_model.encoder)
        self.decoder = tx.modules.AttentionRNNDecoder(token_embedder=self.target_embedder, encoder_output_size=self.encoder.cell_fw.hidden_size + self.encoder.cell_bw.hidden_size, input_size=self.target_embedder.dim, vocab_size=self.target_vocab_size, hparams=config_model.decoder)

    def forward(self, batch, mode):
        enc_outputs, _ = self.encoder(inputs=self.source_embedder(batch['source_text_ids']), sequence_length=batch['source_length'])
        memory = torch.cat(enc_outputs, dim=2)
        if mode == 'train':
            helper_train = self.decoder.create_helper(decoding_strategy='train_greedy')
            training_outputs, _, _ = self.decoder(memory=memory, memory_sequence_length=batch['source_length'], helper=helper_train, inputs=batch['target_text_ids'][:, :-1], sequence_length=batch['target_length'] - 1)
            mle_loss = tx.losses.sequence_sparse_softmax_cross_entropy(labels=batch['target_text_ids'][:, 1:], logits=training_outputs.logits, sequence_length=batch['target_length'] - 1)
            return mle_loss
        else:
            start_tokens = memory.new_full(batch['target_length'].size(), self.bos_token_id, dtype=torch.int64)
            infer_outputs = self.decoder(start_tokens=start_tokens, end_token=self.eos_token_id, memory=memory, memory_sequence_length=batch['source_length'], beam_width=config_model.beam_width)
            return infer_outputs


class LabelSmoothingLoss(nn.Module):
    """With label smoothing,
    KL-divergence between q_{smoothed ground truth prob.}(w)
    and p_{prob. computed by model}(w) is minimized.

    Args:
        label_confidence: the confidence weight on the ground truth label.
        tgt_vocab_size: the size of the final classification.
        ignore_index: The index in the vocabulary to ignore weight.
    """
    one_hot: 'torch.Tensor'

    def __init__(self, label_confidence, tgt_vocab_size, ignore_index=0):
        super().__init__()
        self.ignore_index = ignore_index
        self.tgt_vocab_size = tgt_vocab_size
        label_smoothing = 1 - label_confidence
        assert 0.0 < label_smoothing <= 1.0
        smoothing_value = label_smoothing / (tgt_vocab_size - 2)
        one_hot = torch.full((tgt_vocab_size,), smoothing_value)
        one_hot[self.ignore_index] = 0
        self.register_buffer('one_hot', one_hot.unsqueeze(0))
        self.confidence = label_confidence

    def forward(self, output: 'torch.Tensor', target: 'torch.Tensor', label_lengths: 'torch.LongTensor') ->torch.Tensor:
        """Compute the label smoothing loss.

        Args:
            output (FloatTensor): batch_size x seq_length * n_classes
            target (LongTensor): batch_size * seq_length, specify the label
                target
            label_lengths(torch.LongTensor): specify the length of the labels
        """
        orig_shapes = output.size(), target.size()
        output = output.view(-1, self.tgt_vocab_size)
        target = target.view(-1)
        model_prob = self.one_hot.repeat(target.size(0), 1)
        model_prob = model_prob
        model_prob.scatter_(1, target.unsqueeze(1), self.confidence)
        model_prob.masked_fill_((target == self.ignore_index).unsqueeze(1), 0)
        output = output.view(orig_shapes[0])
        model_prob = model_prob.view(orig_shapes[0])
        return tx.losses.sequence_softmax_cross_entropy(labels=model_prob, logits=output, sequence_length=label_lengths, average_across_batch=False, sum_over_timesteps=False)


class Transformer(nn.Module):
    """A standalone sequence-to-sequence Transformer model, from "Attention
    Is All You Need". The Transformer model consists of the word embedding
    layer, position embedding layer, an encoder and a decoder. Both encoder
    and decoder are stacks of self-attention layers followed by feed-forward
    layers. See "Attention Is All You Need" (https://arxiv.org/abs/1706.03762)
    for the full description of the model.
    """

    def __init__(self, model_config, data_config, vocab: 'tx.data.Vocab'):
        super().__init__()
        self.config_model = model_config
        self.config_data = data_config
        self.vocab = vocab
        self.vocab_size = vocab.size
        self.word_embedder = tx.modules.WordEmbedder(vocab_size=self.vocab_size, hparams=self.config_model.emb)
        self.pos_embedder = tx.modules.SinusoidsPositionEmbedder(position_size=self.config_data.max_decoding_length, hparams=self.config_model.position_embedder_hparams)
        self.encoder = tx.modules.TransformerEncoder(hparams=self.config_model.encoder)
        self.decoder = tx.modules.TransformerDecoder(token_pos_embedder=self._embedding_fn, vocab_size=self.vocab_size, output_layer=self.word_embedder.embedding, hparams=self.config_model.decoder)
        self.smoothed_loss_func = LabelSmoothingLoss(label_confidence=self.config_model.loss_label_confidence, tgt_vocab_size=self.vocab_size, ignore_index=0)

    def _embedding_fn(self, tokens: 'torch.LongTensor', positions: 'torch.LongTensor') ->torch.Tensor:
        word_embed = self.word_embedder(tokens)
        scale = self.config_model.hidden_dim ** 0.5
        pos_embed = self.pos_embedder(positions)
        return word_embed * scale + pos_embed

    def forward(self, encoder_input: 'torch.Tensor', decoder_input: 'Optional[torch.LongTensor]'=None, labels: 'Optional[torch.LongTensor]'=None, beam_width: 'Optional[int]'=None):
        """Compute the maximum likelihood loss or perform decoding, depending
        on arguments.

        Args:
            encoder_input: the source sentence embedding, with the shape of
                `[batch_size, source_seq_length, input_dim]`.
            decoder_input: the target sentence embedding, with the shape of
                `[batch_size, target_seq_length, input_dim]`.
            labels: the target sentence labels, with the shape of
                `[batch_size, target_seq_length]`.
            beam_width: Used in beam search.

        :returns:
            - If both :attr:`decoder_input` and :attr:`labels` are both
              provided, the function enters training logic and returns the
              maximum likelihood loss.
            - Otherwise the function enters inference logic and returns the
              decoded sequence.
            - If `beam_width` > 1, beam search decoding is performed. Please
              refer to :meth:`texar.modules.TransformerDecoder.forward` for
              details on return types.
        """
        batch_size = encoder_input.size(0)
        encoder_input_length = (encoder_input != 0).int().sum(dim=1)
        positions = torch.arange(encoder_input_length.max(), dtype=torch.long, device=encoder_input.device).unsqueeze(0).expand(batch_size, -1)
        src_input_embedding = self._embedding_fn(encoder_input, positions)
        encoder_output = self.encoder(inputs=src_input_embedding, sequence_length=encoder_input_length)
        if decoder_input is not None and labels is not None:
            outputs = self.decoder(memory=encoder_output, memory_sequence_length=encoder_input_length, inputs=decoder_input, decoding_strategy='train_greedy')
            label_lengths = (labels != 0).long().sum(dim=1)
            is_target = (labels != 0).float()
            mle_loss = self.smoothed_loss_func(outputs.logits, labels, label_lengths)
            mle_loss = (mle_loss * is_target).sum() / is_target.sum()
            return mle_loss
        else:
            start_tokens = encoder_input.new_full((batch_size,), self.vocab.bos_token_id)
            predictions = self.decoder(memory=encoder_output, memory_sequence_length=encoder_input_length, beam_width=beam_width, length_penalty=self.config_model.length_penalty, start_tokens=start_tokens, end_token=self.vocab.eos_token_id, max_decoding_length=self.config_data.max_decoding_length, decoding_strategy='infer_greedy')
            return predictions


def MultivariateNormalDiag(loc, scale_diag):
    if loc.dim() < 1:
        raise ValueError('loc must be at least one-dimensional.')
    return Independent(Normal(loc, scale_diag), 1)


def kl_divergence(means: 'Tensor', logvars: 'Tensor') ->Tensor:
    """Compute the KL divergence between Gaussian distribution
    """
    kl_cost = -0.5 * (logvars - means ** 2 - torch.exp(logvars) + 1.0)
    kl_cost = torch.mean(kl_cost, 0)
    return torch.sum(kl_cost)


class DummyClassifier(nn.Module):

    def __init__(self, vocab_size: 'int', n_classes: 'int'):
        super().__init__()
        self.embedder = tx.modules.WordEmbedder(vocab_size=vocab_size, hparams={'dim': 10})
        self.encoder = tx.modules.BidirectionalRNNEncoder(input_size=10, hparams={'rnn_cell_fw': {'kwargs': {'num_units': 256}}})
        self.linear = nn.Linear(sum(self.encoder.output_size), n_classes)

    def _compute_logits(self, tokens: 'torch.LongTensor') ->torch.Tensor:
        embeds = self.embedder(tokens)
        fw_state, bw_state = self.encoder(embeds)[1]
        state = torch.cat([fw_state[0], bw_state[0]], dim=1)
        logits = self.linear(state)
        return logits

    def forward(self, batch: 'tx.data.Batch') ->Dict[str, torch.Tensor]:
        logits = self._compute_logits(batch.tokens)
        loss = F.cross_entropy(logits, batch.label)
        preds = torch.argmax(logits, dim=1)
        return {'loss': loss, 'preds': preds}

    def predict(self, batch: 'tx.data.Batch') ->Dict[str, torch.Tensor]:
        logits = self._compute_logits(batch.tokens)
        preds = torch.argmax(logits, dim=1)
        return {'preds': preds}


State = TypeVar('State')


class RNNCellBase(nn.Module, Generic[State]):
    """The base class for RNN cells in our framework. Major differences over
    :torch_nn:`RNNCell` are two-fold:

    1. Holds an :torch_nn:`Module` which could either be a built-in
       RNN cell or a wrapped cell instance. This design allows
       :class:`RNNCellBase` to serve as the base class for both vanilla
       cells and wrapped cells.

    2. Adds :meth:`zero_state` method for initialization of hidden states,
       which can also be used to implement batch-specific initialization
       routines.
    """

    def __init__(self, cell: "Union[nn.RNNCellBase, 'RNNCellBase']"):
        super().__init__()
        if not isinstance(cell, nn.Module):
            raise ValueError("Type of parameter 'cell' must be derived fromnn.Module, and has 'input_size' and 'hidden_size'attributes.")
        self._cell = cell

    @property
    def input_size(self) ->int:
        """The number of expected features in the input."""
        return self._cell.input_size

    @property
    def hidden_size(self) ->int:
        """The number of features in the hidden state."""
        return self._cell.hidden_size

    @property
    def _param(self) ->nn.Parameter:
        """Convenience method to access a parameter under the module. Useful
        when creating tensors of the same attributes using `param.new_*`.
        """
        return next(self.parameters())

    def init_batch(self):
        """Perform batch-specific initialization routines. For most cells this
        is a no-op.
        """
        pass

    def zero_state(self, batch_size: 'int') ->State:
        """Return zero-filled state tensor(s).

        Args:
            batch_size: int, the batch size.

        Returns:
            State tensor(s) initialized to zeros. Note that different subclasses
            might return tensors of different shapes and structures.
        """
        self.init_batch()
        if isinstance(self._cell, nn.RNNCellBase):
            state = self._param.new_zeros(batch_size, self.hidden_size, requires_grad=False)
        else:
            state = self._cell.zero_state(batch_size)
        return state

    def forward(self, input: 'torch.Tensor', state: 'Optional[State]'=None) ->Tuple[torch.Tensor, State]:
        """
        Returns:
            A tuple of (output, state). For single layer RNNs, output is
            the same as state.
        """
        if state is None:
            batch_size = input.size(0)
            state = self.zero_state(batch_size)
        return self._cell(input, state)


class MaxReducePool1d(nn.Module):
    """A subclass of :torch_nn:`Module`.
    Max Pool layer for 1D inputs. The same as :torch_nn:`MaxPool1d` except that
    the pooling dimension is entirely reduced (i.e., `pool_size=input_length`).
    """

    def forward(self, input: 'torch.Tensor') ->torch.Tensor:
        output, _ = torch.max(input, dim=2)
        return output


class AvgReducePool1d(nn.Module):
    """A subclass of :torch_nn:`Module`.
    Avg Pool layer for 1D inputs. The same as :torch_nn:`AvgPool1d` except that
    the pooling dimension is entirely reduced (i.e., `pool_size=input_length`).
    """

    def forward(self, input: 'torch.Tensor') ->torch.Tensor:
        return torch.mean(input, dim=2)


def is_str(x):
    """Returns `True` if :attr:`x` is either a str or unicode.
    Returns `False` otherwise.
    """
    return isinstance(x, str)


def get_layer(hparams: 'Union[HParams, Dict[str, Any]]') ->nn.Module:
    """Makes a layer instance.

    The layer must be an instance of :torch_nn:`Module`.

    Args:
        hparams (dict or HParams): Hyperparameters of the layer, with
            structure:

            .. code-block:: python

                {
                    "type": "LayerClass",
                    "kwargs": {
                        # Keyword arguments of the layer class
                        # ...
                    }
                }

            Here:

            `"type"`: str or layer class or layer instance
                The layer type. This can be

                - The string name or full module path of a layer class. If
                  the class name is provided, the class must be in module
                  :torch_nn:`Module`, :mod:`texar.torch.core`, or
                  :mod:`texar.torch.custom`.
                - A layer class.
                - An instance of a layer class.

                For example

                .. code-block:: python

                    "type": "Conv1D"                               # class name
                    "type": "texar.torch.core.MaxReducePooling1D"  # module path
                    "type": "my_module.MyLayer"                    # module path
                    "type": torch.nn.Module.Linear                 # class
                    "type": Conv1D(filters=10, kernel_size=2)  # cell instance
                    "type": MyLayer(...)                       # cell instance

            `"kwargs"`: dict
                A dictionary of keyword arguments for constructor of the
                layer class. Ignored if :attr:`"type"` is a layer instance.

                - Arguments named "activation" can be a callable, or a `str` of
                  the name or module path to the activation function.
                - Arguments named "\\*_regularizer" and "\\*_initializer" can be a
                  class instance, or a `dict` of hyperparameters of respective
                  regularizers and initializers. See
                - Arguments named "\\*_constraint" can be a callable, or a `str`
                  of the name or full path to the constraint function.

    Returns:
        A layer instance. If ``hparams["type"]`` is a layer instance, returns it
        directly.

    Raises:
        ValueError: If :attr:`hparams` is `None`.
        ValueError: If the resulting layer is not an instance of
            :torch_nn:`Module`.
    """
    if hparams is None:
        raise ValueError('`hparams` must not be `None`.')
    layer_type = hparams['type']
    if not is_str(layer_type) and not isinstance(layer_type, type):
        layer = layer_type
    else:
        layer_modules = ['torch.nn', 'texar.torch.core', 'texar.torch.custom']
        layer_class: 'Type[nn.Module]' = utils.check_or_get_class(layer_type, layer_modules)
        if isinstance(hparams, dict):
            if layer_class.__name__ == 'Linear' and 'in_features' not in hparams['kwargs']:
                raise ValueError('"in_features" should be specified for "torch.nn.{}"'.format(layer_class.__name__))
            elif layer_class.__name__ in ['Conv1d', 'Conv2d', 'Conv3d'] and 'in_channels' not in hparams['kwargs']:
                raise ValueError('"in_channels" should be specified for "torch.nn.{}"'.format(layer_class.__name__))
            default_kwargs: 'Dict[str, Any]' = {}
            default_hparams = {'type': layer_type, 'kwargs': default_kwargs}
            hparams = HParams(hparams, default_hparams)
        if layer_type == 'Sequential':
            names: 'List[str]' = []
            layer = nn.Sequential()
            sub_hparams = hparams.kwargs.layers
            for hparam in sub_hparams:
                sub_layer = get_layer(hparam)
                name = utils.uniquify_str(sub_layer.__class__.__name__, names)
                names.append(name)
                layer.add_module(name=name, module=sub_layer)
        else:
            layer = utils.get_instance(layer_type, hparams.kwargs.todict(), layer_modules)
    if not isinstance(layer, nn.Module):
        raise ValueError('layer must be an instance of `torch.nn.Module`.')
    return layer


class MergeLayer(nn.Module):
    """A subclass of :torch_nn:`Module`.
    A layer that consists of multiple layers in parallel. Input is fed to
    each of the parallel layers, and the outputs are merged with a
    specified mode.

    Args:
        layers (list, optional): A list of :torch_docs:`torch.nn.Module
            <nn.html#module>` instances, or a list of hyperparameter
            dictionaries each of which specifies `"type"` and `"kwargs"` of each
            layer (see the `hparams` argument of :func:`get_layer`).

            If `None`, this layer degenerates to a merging operator that merges
            inputs directly.
        mode (str): Mode of the merge op. This can be:

            - :attr:`'concat'`: Concatenates layer outputs along one dim.
              Tensors must have the same shape except for the dimension
              specified in `dim`, which can have different sizes.
            - :attr:`'elemwise_sum'`: Outputs element-wise sum.
            - :attr:`'elemwise_mul'`: Outputs element-wise product.
            - :attr:`'sum'`: Computes the sum of layer outputs along the
              dimension given by `dim`. For example, given `dim=1`,
              two tensors of shape `[a, b]` and `[a, c]` respectively
              will result in a merged tensor of shape `[a]`.
            - :attr:`'mean'`: Computes the mean of layer outputs along the
              dimension given in `dim`.
            - :attr:`'prod'`: Computes the product of layer outputs along the
              dimension given in `dim`.
            - :attr:`'max'`: Computes the maximum of layer outputs along the
              dimension given in `dim`.
            - :attr:`'min'`: Computes the minimum of layer outputs along the
              dimension given in `dim`.
            - :attr:`'and'`: Computes the `logical and` of layer outputs along
              the dimension given in `dim`.
            - :attr:`'or'`: Computes the `logical or` of layer outputs along
              the dimension given in `dim`.
            - :attr:`'logsumexp'`: Computes
              log(sum(exp(elements across the dimension of layer outputs)))
        dim (int): The dim to use in merging. Ignored in modes
            :attr:`'elemwise_sum'` and :attr:`'elemwise_mul'`.
    """
    _functions: 'Dict[str, Callable[[torch.Tensor, int], torch.Tensor]]' = {'sum': torch.sum, 'mean': torch.mean, 'prod': torch.prod, 'max': lambda tensors, dim: torch.max(tensors, dim)[0], 'min': lambda tensors, dim: torch.min(tensors, dim)[0], 'and': torch.all, 'or': torch.any, 'logsumexp': torch.logsumexp}

    def __init__(self, layers: 'Optional[List[nn.Module]]'=None, mode: 'str'='concat', dim: 'Optional[int]'=None):
        super().__init__()
        self._mode = mode
        self._dim = dim
        self._layers: 'Optional[nn.ModuleList]' = None
        if layers is not None:
            if len(layers) == 0:
                raise ValueError("'layers' must be either None or a non-empty list.")
            self._layers = nn.ModuleList()
            for layer in layers:
                if isinstance(layer, nn.Module):
                    self._layers.append(layer)
                else:
                    self._layers.append(get_layer(hparams=layer))

    def forward(self, input: 'torch.Tensor') ->torch.Tensor:
        """Feed input to every containing layer and merge the outputs.

        Args:
            input: The input tensor.

        Returns:
            The merged tensor.
        """
        layer_outputs: 'List[torch.Tensor]'
        if self._layers is None:
            layer_outputs = input
            if not isinstance(layer_outputs, (list, tuple)):
                layer_outputs = [layer_outputs]
        else:
            layer_outputs = []
            for layer in self._layers:
                layer_output = layer(input)
                layer_outputs.append(layer_output)
        dim = self._dim if self._dim is not None else -1
        if self._mode == 'concat':
            outputs = torch.cat(tensors=layer_outputs, dim=dim)
        elif self._mode == 'elemwise_sum':
            outputs = layer_outputs[0]
            for i in range(1, len(layer_outputs)):
                outputs = torch.add(outputs, layer_outputs[i])
        elif self._mode == 'elemwise_mul':
            outputs = layer_outputs[0]
            for i in range(1, len(layer_outputs)):
                outputs = torch.mul(outputs, layer_outputs[i])
        elif self._mode in self._functions:
            _concat = torch.cat(tensors=layer_outputs, dim=dim)
            outputs = self._functions[self._mode](_concat, dim)
        else:
            raise ValueError("Unknown merge mode: '%s'" % self._mode)
        return outputs

    @property
    def layers(self) ->Optional[nn.ModuleList]:
        """The list of parallel layers.
        """
        return self._layers


class Flatten(nn.Module):
    """Flatten layer to flatten a tensor after convolution."""

    def forward(self, input: 'torch.Tensor') ->torch.Tensor:
        return input.view(input.size()[0], -1)


class Identity(nn.Module):
    """Identity activation layer."""

    def forward(self, input: 'torch.Tensor') ->torch.Tensor:
        return input


class BertGELU(nn.Module):
    """Bert uses GELU as the activation function for the position-wise network.
    """

    def forward(self, x):
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class GPTGELU(nn.Module):
    """For information: OpenAI GPT's GELU is slightly different (and gives
    slightly different results).
    """

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


def uniquify_str(str_: 'str', str_set: 'Collection[str]') ->str:
    """Uniquifies :attr:`str_` if :attr:`str_` is included in :attr:`str_set`.

    This is done by appending a number to :attr:`str_`. Returns
    :attr:`str_` directly if it is not included in :attr:`str_set`.

    Args:
        str\\_ (string): A string to uniquify.
        str_set (set, dict, or list): A collection of strings. The returned
            string is guaranteed to be different from the elements in the
            collection.

    Returns:
        The uniquified string. Returns :attr:`str_` directly if it is
        already unique.

    Example:

        .. code-block:: python

            print(uniquify_str('name', ['name', 'name_1']))
            # 'name_2'

    """
    if str_ not in str_set:
        return str_
    else:
        for i in range(1, len(str_set) + 1):
            unique_str = str_ + '_%d' % i
            if unique_str not in str_set:
                return unique_str
    raise ValueError('Failed to uniquify string: ' + str_)


def _to_list(value: 'Union[Dict[str, Any], List, Tuple, int]', name=None, list_length=None):
    """Converts `hparams` value into a list.

    If :attr:`list_length` is given, then the canonicalized :attr:`value`
    must be of length :attr:`list_length`.
    """
    if not isinstance(value, (list, tuple)):
        if list_length is not None:
            value = [value] * list_length
        else:
            value = [value]
    if list_length is not None and len(value) != list_length:
        name = '' if name is None else name
        raise ValueError("hparams '%s' must be a list of length %d" % (name, list_length))
    return value


_POOLING_TO_REDUCE = {'MaxPool1d': 'MaxReducePool1d', 'AvgPool1d': 'AvgReducePool1d', torch.nn.MaxPool1d: MaxReducePool1d, torch.nn.AvgPool1d: AvgReducePool1d}


def get_pooling_layer_hparams(hparams: 'Union[HParams, Dict[str, Any]]') ->Dict[str, Any]:
    """Creates pooling layer hyperparameters `dict` for :func:`get_layer`.

    If the :attr:`hparams` sets `'pool_size'` to `None`, the layer will be
    changed to the respective reduce-pooling layer. For example,
    :torch_docs:`torch.conv.MaxPool1d <nn.html#torch.nn.Conv1d>` is replaced
    with :class:`~texar.torch.core.MaxReducePool1d`.
    """
    if isinstance(hparams, HParams):
        hparams = hparams.todict()
    new_hparams = copy.copy(hparams)
    kwargs = new_hparams.get('kwargs', None)
    if kwargs and kwargs.get('kernel_size', None) is None:
        pool_type = hparams['type']
        new_hparams['type'] = _POOLING_TO_REDUCE.get(pool_type, pool_type)
        kwargs.pop('kernel_size', None)
        kwargs.pop('stride', None)
        kwargs.pop('padding', None)
    return new_hparams


def transpose_batch_time(inputs: 'torch.Tensor') ->torch.Tensor:
    """Transposes inputs between time-major and batch-major.

    Args:
        inputs: A Tensor of shape ``[batch_size, max_time, ...]`` (batch-major)
            or ``[max_time, batch_size, ...]`` (time-major), or a (possibly
            nested) tuple of such elements.

    Returns:
        A (possibly nested tuple of) Tensor with transposed batch and
        time dimensions of inputs.
    """
    return inputs.transpose(0, 1)


def mask_sequences(sequence: 'Union[torch.Tensor, List[int]]', sequence_length: 'Union[torch.LongTensor, List[int]]', dtype: 'Optional[torch.dtype]'=None, time_major: 'bool'=False) ->torch.Tensor:
    """Masks out sequence entries that are beyond the respective sequence
    lengths. Masks along the time dimension.

    :attr:`sequence` and :attr:`sequence_length` can either be python
    arrays or Tensors, respectively. If both are Python arrays (or None), the
    return will be a Python array as well.

    Args:
        sequence: A Tensor or Python array of sequence values.
            If ``time_major==False`` (default), this must be a Tensor of shape
            ``[batch_size, max_time, ...]``. The batch and time dimension is
            exchanged if ``time_major==True``.
        sequence_length: A Tensor or python array of shape ``[batch_size]``.
            Time steps beyond the respective sequence lengths will be
            made zero.
        dtype (dtype): Type of :attr:`sequence`. If `None`, infer from
            :attr:`sequence` automatically.
        time_major (bool): The shape format of the inputs. If `True`,
            :attr:`sequence` must have shape
            ``[max_time, batch_size, ...]``.
            If `False` (default), :attr:`sequence` must have
            shape ``[batch_size, max_time, ...]``.

    Returns:
        The masked sequence, i.e., a Tensor or python array of the same shape
        as :attr:`sequence` but with masked-out entries (set to zero).

        If both :attr:`sequence` and :attr:`sequence_length` are python
        arrays, the returned value is a python array as well.
    """
    if not torch.is_tensor(sequence):
        sequence = torch.tensor(sequence, dtype=dtype)
    sequence: 'torch.Tensor'
    rank = sequence.dim()
    if rank < 2:
        raise ValueError('`sequence` must be 2D or higher order.')
    if time_major:
        sequence = transpose_batch_time(sequence)
    max_time = sequence.size(1)
    if dtype is None:
        dtype = sequence.dtype
    mask = utils.sequence_mask(sequence_length, max_time, dtype=dtype)
    mask = mask.view(*mask.size(), *([1] * (rank - 2)))
    sequence = sequence * mask
    if time_major:
        sequence = transpose_batch_time(sequence)
    return sequence


def _extract_google_drive_file_id(url: 'str') ->str:
    url_suffix = url[url.find('/d/') + 3:]
    if url_suffix.find('/') == -1:
        return url_suffix
    file_id = url_suffix[:url_suffix.find('/')]
    return file_id


def get_filename(url: 'str') ->str:
    """Extracts the filename of the downloaded checkpoint file from the URL.
    """
    if 'drive.google.com' in url:
        return _extract_google_drive_file_id(url)
    url, filename = os.path.split(url)
    return filename or os.path.basename(url)


def _download(url: 'str', filename: 'str', path: 'str') ->str:

    def _progress_hook(count, block_size, total_size):
        percent = float(count * block_size) / float(total_size) * 100.0
        sys.stdout.write(f'\r>> Downloading {filename} {percent:.1f}%')
        sys.stdout.flush()
    filepath = os.path.join(path, filename)
    filepath, _ = urllib.request.urlretrieve(url, filepath, _progress_hook)
    None
    statinfo = os.stat(filepath)
    None
    return filepath


def maybe_download(urls, path, filenames=None, extract=False):
    """Downloads a set of files.

    Args:
        urls: A (list of) URLs to download files.
        path (str): The destination path to save the files.
        filenames: A (list of) strings of the file names. If given,
            must have the same length with :attr:`urls`. If `None`,
            filenames are extracted from :attr:`urls`.
        extract (bool): Whether to extract compressed files.

    Returns:
        A list of paths to the downloaded files.
    """
    utils_io.maybe_create_dir(path)
    if not isinstance(urls, (list, tuple)):
        is_list = False
        urls = [urls]
    else:
        is_list = True
    if filenames is not None:
        if not isinstance(filenames, (list, tuple)):
            filenames = [filenames]
        if len(urls) != len(filenames):
            raise ValueError('`filenames` must have the same number of elements as `urls`.')
    result = []
    for i, url in enumerate(urls):
        if filenames is not None:
            filename = filenames[i]
        elif 'drive.google.com' in url:
            filename = _extract_google_drive_file_id(url)
        else:
            filename = url.split('/')[-1]
            if filename.endswith('?raw=true'):
                filename = filename[:-9]
        filepath = os.path.join(path, filename)
        result.append(filepath)
        if not os.path.exists(filepath):
            if 'drive.google.com' in url:
                filepath = _download_from_google_drive(url, filename, path)
            else:
                filepath = _download(url, filename, path)
            if extract:
                logging.info('Extract %s', filepath)
                if tarfile.is_tarfile(filepath):
                    tarfile.open(filepath, 'r').extractall(path)
                elif zipfile.is_zipfile(filepath):
                    with zipfile.ZipFile(filepath) as zfile:
                        zfile.extractall(path)
                else:
                    logging.info('Unknown compression type. Only .tar.gz.tar.bz2, .tar, and .zip are supported')
    if not is_list:
        return result[0]
    return result


_CHECKPOINT_FILES = ['checkpoint', 'encoder.json', 'hparams.json', 'vocab.bpe', 'model.ckpt.data-00000-of-00001', 'model.ckpt.index', 'model.ckpt.meta']


_GPT2_PATH = 'https://openaipublic.blob.core.windows.net/gpt-2/models/'


Type_size_keeper = [nn.ELU, nn.Hardshrink, nn.Hardtanh, nn.LeakyReLU, nn.LogSigmoid, nn.PReLU, nn.ReLU, nn.RReLU, nn.SELU, nn.CELU, nn.Sigmoid, nn.Softplus, nn.Softshrink, nn.Softsign, nn.Tanh, nn.Tanhshrink, nn.Threshold, nn.Softmin, nn.Softmax, nn.LogSoftmax, nn.Dropout, nn.AlphaDropout]


Type_size_lambda_map = {nn.Linear: lambda x: x.out_features, nn.Bilinear: lambda x: x.out_features, _ConvNd: lambda x: x.out_channels * len(x.kernel_size), nn.Embedding: lambda x: x.embedding_dim, nn.EmbeddingBag: lambda x: x.embedding_dim, nn.RNNCellBase: lambda x: x.hidden_size}


def get_output_size(input_instance: 'nn.Module') ->Optional[int]:
    """Return the final dimension size of :attr:`input_instance` output.

    If type of :attr:`input_instance` is among the common types, the final
    dimension size will be computed.

    Args:
        input_instance: A :class:`~torch.nn.Module` instance from
            which to compute the final dimension size.

    Returns:
        int (optional): The final dimension size of the output.
            If output size is determined by input, returns ``-1``,
            otherwise if output size is not computable, return `None`.
    """
    for t, l in Type_size_lambda_map.items():
        if isinstance(input_instance, t):
            return l(input_instance)
    for t in Type_size_keeper:
        if isinstance(input_instance, t):
            return -1
    return None


def default_transformer_poswise_net_hparams(input_dim: 'int', output_dim: 'int'=512) ->Dict[str, Any]:
    """Returns default hyperparameters of a
    :class:`~texar.torch.modules.FeedForwardNetwork` as a position-wise network
    used in :class:`~texar.torch.modules.TransformerEncoder` and
    :class:`~texar.torch.modules.TransformerDecoder`.
    This is a 2-layer dense network with dropout in-between.

    .. code-block:: python

        {
            "layers": [
                {
                    "type": "Linear",
                    "kwargs": {
                        "in_features": input_dim,
                        "out_features": output_dim * 4,
                        "bias": True,
                    }
                },
                {
                    "type": "nn.ReLU",
                    "kwargs": {
                        "inplace": True
                    }
                },
                {
                    "type": "Dropout",
                    "kwargs": {
                        "p": 0.1,
                    }
                },
                {
                    "type": "Linear",
                    "kwargs": {
                        "in_features": output_dim * 4,
                        "out_features": output_dim,
                        "bias": True,
                    }
                }
            ],
            "name": "ffn"
        }

    Args:
        input_dim (int): The size of dense layer input.
        output_dim (int): The size of dense layer output.
    """
    return {'layers': [{'type': 'Linear', 'kwargs': {'in_features': input_dim, 'out_features': output_dim * 4, 'bias': True}}, {'type': 'ReLU', 'kwargs': {'inplace': True}}, {'type': 'Dropout', 'kwargs': {'p': 0.1}}, {'type': 'Linear', 'kwargs': {'in_features': output_dim * 4, 'out_features': output_dim, 'bias': True}}], 'name': 'ffn'}


def sequence_mask(lengths: 'Union[torch.LongTensor, List[int]]', max_len: 'Optional[int]'=None, dtype: 'Optional[torch.dtype]'=None, device: 'Optional[torch.device]'=None) ->torch.ByteTensor:
    """Return a mask tensor representing the first N positions of each cell.

    If ``lengths`` has shape ``[d_1, d_2, ..., d_n]`` the resulting tensor
    ``mask`` has dtype ``dtype`` and shape ``[d_1, d_2, ..., d_n, maxlen]``,
    with

    ```
    mask[i_1, i_2, ..., i_n, j] = (j < lengths[i_1, i_2, ..., i_n])
    ```

    Examples:

    ```python
    sequence_mask([1, 3, 2], 5)  # [[True, False, False, False, False],
                                 #  [True,  True,  True, False, False],
                                 #  [True,  True, False, False, False]]

    sequence_mask([[1, 3],[2,0]])  # [[[ True, False, False],
                                   #   [ True,  True,  True]],
                                   #  [[ True,  True, False],
                                   #   [False, False, False]]]
    ```

    Args:
        lengths: integer tensor or list of int, all its values <= max_len.
        max_len: scalar integer tensor, size of last dimension of returned
            tensor. Default is the maximum value in ``lengths``.
        dtype: the desired data type of returned tensor. Default: if None,
            returns :torch:`ByteTensor`.
        device: the desired device of returned tensor. Default: if None, uses
            the current device for the default tensor type.
    Returns:
        A mask tensor of shape :python:`lengths.shape + (max_len,)`, cast to
        specified dtype.
    Raises:
        ValueError: if ``max_len`` is not a scalar.
    """
    if not isinstance(lengths, torch.Tensor):
        lengths = torch.tensor(lengths, device=device)
    elif device is None:
        device = lengths.device
    lengths: 'torch.LongTensor'
    if max_len is None:
        max_len = torch.max(lengths).item()
    size = lengths.size()
    row_vector = torch.arange(max_len, device=device, dtype=lengths.dtype).view(*([1] * len(size)), -1).expand(*size, max_len)
    mask = row_vector < lengths.unsqueeze(-1)
    if dtype is not None:
        mask = mask
    return mask


AnyDict = MutableMapping[str, Any]


def dict_fetch(src_dict: 'Optional[ParamDict]', tgt_dict_or_keys: 'Union[ParamDict, List[str]]') ->Optional[AnyDict]:
    """Fetches a sub-dictionary of :attr:`src_dict` with the keys in
    :attr:`tgt_dict_or_keys`.

    Args:
        src_dict: A dictionary or instance of :class:`~texar.torch.HParams`.
            The source dictionary to fetch values from.
        tgt_dict_or_keys: A dictionary, instance of
            :class:`~texar.torch.HParams`, or a list (or a
            ``dict_keys``/``KeysView``) of keys to be included in the output
            dictionary.

    Returns:
        A new dictionary that is a sub-dictionary of :attr:`src_dict`.
    """
    if src_dict is None:
        return src_dict
    if isinstance(tgt_dict_or_keys, HParams):
        tgt_dict_or_keys = tgt_dict_or_keys.todict()
    if isinstance(tgt_dict_or_keys, MutableMapping):
        tgt_dict_or_keys = tgt_dict_or_keys.keys()
    keys = list(tgt_dict_or_keys)
    if isinstance(src_dict, HParams):
        src_dict = src_dict.todict()
    return {k: src_dict[k] for k in keys if k in src_dict}


def get_initializer(hparams=None) ->Optional[Callable[[torch.Tensor], torch.Tensor]]:
    """Returns an initializer instance.

    Args:
        hparams (dict or HParams, optional): Hyperparameters with the structure

            .. code-block:: python

                {
                    "type": "initializer_class_or_function",
                    "kwargs": {
                        # ...
                    }
                }

            The `"type"` field can be a function name or module path. If name is
            provided, it be must be from one the following modules:
            :torch_docs:`torch.nn.init <nn.html#torch-nn-init>` and
            :mod:`texar.torch.custom`.

            Besides, the `"type"` field can also be an initialization function
            called with :python:`initialization_fn(**kwargs)`. In this case
            `"type"` can be the function, or its name or module path. If no
            keyword argument is required, `"kwargs"` can be omitted.

    Returns:
        An initializer instance. `None` if :attr:`hparams` is `None`.
    """
    if hparams is None:
        return None
    kwargs = hparams.get('kwargs', {})
    if isinstance(kwargs, HParams):
        kwargs = kwargs.todict()
    modules = ['torch.nn.init', 'torch', 'texar.torch.custom']
    initializer_fn = utils.get_function(hparams['type'], modules)
    initializer = functools.partial(initializer_fn, **kwargs)
    return initializer


class BuiltinCellWrapper(RNNCellBase[State]):
    """Base class for wrappers over built-in :torch_nn:`RNNCellBase`
    RNN cells.
    """

    def forward(self, input: 'torch.Tensor', state: 'Optional[State]'=None) ->Tuple[torch.Tensor, State]:
        if state is None:
            batch_size = input.size(0)
            state = self.zero_state(batch_size)
        new_state = self._cell(input, state)
        return new_state, new_state


LSTMState = Tuple[torch.Tensor, torch.Tensor]


class LSTMCell(BuiltinCellWrapper[LSTMState]):
    """A wrapper over :torch_nn:`LSTMCell`, additionally providing the
    option to initialize the forget-gate bias to a constant value.
    """

    def __init__(self, input_size, hidden_size, bias=True, forget_bias: 'Optional[float]'=None):
        if forget_bias is not None and not bias:
            raise ValueError("Parameter 'forget_bias' must be set to None when'bias' is set to False.")
        cell = nn.LSTMCell(input_size, hidden_size, bias=bias)
        if forget_bias is not None:
            with torch.no_grad():
                cell.bias_ih[hidden_size:2 * hidden_size].fill_(forget_bias)
                cell.bias_hh[hidden_size:2 * hidden_size].fill_(forget_bias)
        super().__init__(cell)

    def zero_state(self, batch_size: 'int') ->LSTMState:
        """Returns the zero state for LSTMs as (h, c)."""
        state = self._param.new_zeros(batch_size, self.hidden_size, requires_grad=False)
        return state, state

    def forward(self, input: 'torch.Tensor', state: 'Optional[LSTMState]'=None) ->Tuple[torch.Tensor, LSTMState]:
        if state is None:
            batch_size = input.size(0)
            state = self.zero_state(batch_size)
        new_state = self._cell(input, state)
        return new_state[0], new_state


def _build_dense_output_layer(cell_output_size: 'int', hparams: 'HParams') ->Optional[nn.Sequential]:
    """Build the output layers.

    Args:
        cell_output_size: The output size of the rnn cell.
        hparams (dict or HParams): Hyperparameters. Missing hyperparameters
            will be set to default values. See
            :meth:`default_hparams` for the hyperparameter structure and
            default values.

    Returns:
        A :torch_nn:`Sequential` module containing the output layers.
    """
    nlayers = hparams.num_layers
    if nlayers <= 0:
        return None
    layer_size = _to_list(hparams.layer_size, 'output_layer.layer_size', nlayers)
    dropout_layer_ids = _to_list(hparams.dropout_layer_ids)
    other_kwargs = hparams.other_dense_kwargs or {}
    if isinstance(other_kwargs, HParams):
        other_kwargs = other_kwargs.todict()
    if not isinstance(other_kwargs, dict):
        raise ValueError("hparams 'output_layer.other_dense_kwargs' must be a dict.")
    output_layers: 'List[nn.Module]' = []
    for i in range(nlayers):
        if i in dropout_layer_ids:
            output_layers.append(nn.Dropout(p=hparams.dropout_rate))
        dense_layer = nn.Linear(in_features=cell_output_size if i == 0 else layer_size[i - 1], out_features=layer_size[i], **other_kwargs)
        output_layers.append(dense_layer)
        if i == nlayers - 1:
            activation = hparams.final_layer_activation
        else:
            activation = hparams.activation
        if activation is not None:
            layer_hparams = {'type': activation, 'kwargs': {}}
            activation_layer = layers.get_layer(hparams=layer_hparams)
            output_layers.append(activation_layer)
    if nlayers in dropout_layer_ids:
        output_layers.append(nn.Dropout(p=hparams.dropout_rate))
    return nn.Sequential(*output_layers)


def _default_output_layer_hparams() ->Dict[str, Any]:
    return {'num_layers': 0, 'layer_size': 128, 'activation': 'Identity', 'final_layer_activation': None, 'other_dense_kwargs': None, 'dropout_layer_ids': [], 'dropout_rate': 0.5, 'variational_dropout': False, '@no_typecheck': ['activation', 'final_layer_activation', 'layer_size', 'dropout_layer_ids']}


def _forward_output_layers(inputs: 'torch.Tensor', output_layer: 'Optional[nn.Module]', time_major: 'bool', sequence_length: 'Optional[Union[torch.LongTensor, List[int]]]'=None) ->Tuple[torch.Tensor, int]:
    """Forwards inputs through the output layers.

    Args:
        inputs: A Tensor of shape ``[batch_size, max_time] + input_size`` if
            :attr:`time_major` is `False`, or shape
            ``[max_time, batch_size] + input_size`` if :attr:`time_major` is
            `True`.
        output_layer (optional): :torch_nn:`Sequential` or :torch_nn:`Module`
            of output layers.
        time_major (bool): The shape format of the :attr:`inputs` and
            :attr:`outputs` Tensors. If `True`, these tensors are of shape
            `[max_time, batch_size, input_size]`. If `False` (default),
            these tensors are of shape `[batch_size, max_time, input_size]`.
        sequence_length (optional): A 1D :tensor:`LongTensor` of shape
            ``[batch_size]``. Sequence lengths of the batch inputs. Used to
            copy-through state and zero-out outputs when past a batch element's
            sequence length.

    Returns:
        A pair :attr:`(outputs, outputs_size), where

        - :attr:`outputs`: A Tensor of shape
        `[batch_size, max_time] + outputs_size`.

        - :attr:`outputs_size`: An `int` representing the output size.
    """
    if output_layer is None:
        return inputs, inputs.shape[-1]
    output = output_layer(inputs)
    if sequence_length is not None:
        output = mask_sequences(output, sequence_length, time_major=time_major)
    output_size = output.shape[-1]
    return output, output_size


R = TypeVar('R')


@no_type_check
def map_structure(fn: 'Callable[[T], R]', obj: 'Collection[T]') ->Collection[R]:
    """Map a function over all elements in a (possibly nested) collection.

    Args:
        fn (callable): The function to call on elements.
        obj: The collection to map function over.

    Returns:
        The collection in the same structure, with elements mapped.
    """
    if hasattr(obj, '--no-map--'):
        return fn(obj)
    if isinstance(obj, list):
        return [map_structure(fn, x) for x in obj]
    if isinstance(obj, tuple):
        if isinstance(obj, torch.Size):
            return fn(obj)
        if hasattr(obj, '_fields'):
            return type(obj)(*[map_structure(fn, x) for x in obj])
        else:
            return tuple(map_structure(fn, x) for x in obj)
    if isinstance(obj, dict):
        return {k: map_structure(fn, v) for k, v in obj.items()}
    if isinstance(obj, set):
        return {map_structure(fn, x) for x in obj}
    return fn(obj)


@no_type_check
def map_structure_zip(fn: 'Callable[..., R]', objs: 'Sequence[Collection[T]]') ->Collection[R]:
    """Map a function over tuples formed by taking one elements from each
    (possibly nested) collection. Each collection must have identical
    structures.

    .. note::
        Although identical structures are required, it is not enforced by
        assertions. The structure of the first collection is assumed to be
        the structure for all collections.

        For rare cases where collections need to have different structures,
        refer to :meth:`no_map`.

    Args:
        fn (callable): The function to call on elements.
        objs: The list of collections to map function over.

    Returns:
        A collection with the same structure, with elements mapped.
    """
    obj = objs[0]
    if hasattr(obj, '--no-map--'):
        return fn(*objs)
    if isinstance(obj, list):
        return [map_structure_zip(fn, xs) for xs in zip(*objs)]
    if isinstance(obj, tuple):
        if isinstance(obj, torch.Size):
            return fn(obj)
        if hasattr(obj, '_fields'):
            return type(obj)(*[map_structure_zip(fn, xs) for xs in zip(*objs)])
        else:
            return tuple(map_structure_zip(fn, xs) for xs in zip(*objs))
    if isinstance(obj, dict):
        return {k: map_structure_zip(fn, [o[k] for o in objs]) for k in obj.keys()}
    if isinstance(obj, set):
        return {map_structure_zip(fn, xs) for xs in zip(*objs)}
    return fn(*objs)


T = TypeVar('T')


@lru_cache(maxsize=None)
def _no_map_type(container_type: 'Type[T]') ->Type[T]:
    new_type = type('_no_map' + container_type.__name__, (container_type,), {'--no-map--': True})
    return new_type


def no_map(container_type: 'Type[T]', *args, **kwargs) ->T:
    """Create a "`non-mappable`" container type, i.e. it will be treated as a
    singleton object in :meth:`map_structure` and :meth:`map_structure_zip`,
    its contents will not be traversed.

    This is implemented by dynamically creating a subclass of the required type,
    and overriding the :attr:`__subclasscheck__` class method to always return
    `False`.

    Args:
        container_type: The type of the container to create,
            e.g. `list`, `dict`.
        args: Arguments to the constructor.
        kwargs: Keyword arguments to the constructor

    Returns:
        The `non-mappable` container type.
    """
    return _no_map_type(container_type)(*args, **kwargs)


def _dynamic_rnn_loop(cell: 'RNNCellBase[State]', inputs: 'torch.Tensor', initial_state: 'State', sequence_length: 'torch.LongTensor') ->Tuple[torch.Tensor, State]:
    """Internal implementation of Dynamic RNN.

    Args:
        cell: An instance of RNNCell.
        inputs: A ``Tensor`` of shape ``[time, batch_size, input_size]``,
            or a nested tuple of such elements.
        initial_state: A ``Tensor`` of shape ``[batch_size, state_size]``,
            or if ``cell.state_size`` is a tuple, then this should be a tuple
            of tensors having shapes ``[batch_size, s]`` for ``s`` in
            ``cell.state_size``.
        sequence_length: (optional) An ``int32`` ``Tensor``
            of shape ``[batch_size]``.

    Returns:
        Tuple ``(final_outputs, final_state)``.
        final_outputs:
            A ``Tensor`` of shape ``[time, batch_size, cell.output_size]``. If
            ``cell.output_size`` is a (possibly nested) tuple of ints or
            ``torch.Size`` objects, then this returns a
            (possibly nested) tuple of Tensors matching the corresponding
            shapes.
        final_state:
            A ``Tensor``, or possibly nested tuple of Tensors, matching
            in length and shapes to ``initial_state``.
    """
    state = initial_state
    time_steps = inputs.shape[0]
    all_outputs = []
    all_state = map_structure(lambda _: no_map(list), state)
    for i in range(time_steps):
        output, state = cell(inputs[i], state)
        all_outputs.append(output)
        map_structure_zip(lambda xs, x: xs.append(x), (all_state, state))
    final_outputs = torch.stack(all_outputs, dim=0)
    final_outputs = mask_sequences(final_outputs, sequence_length=sequence_length, time_major=True)
    final_state = map_structure(lambda _: no_map(list), state)
    for batch_idx, time_idx in enumerate(sequence_length.tolist()):
        if time_idx > 0:
            map_structure_zip(lambda xs, x: xs.append(x[time_idx - 1][batch_idx]), (final_state, all_state))
        else:
            map_structure_zip(lambda xs, x: xs.append(x[batch_idx]), (final_state, initial_state))
    final_state = map_structure(lambda x: torch.stack(x, dim=0), final_state)
    return final_outputs, final_state


def dynamic_rnn(cell: 'RNNCellBase[State]', inputs: 'torch.Tensor', sequence_length: 'Optional[Union[torch.LongTensor, List[int]]]'=None, initial_state: 'Optional[State]'=None, time_major: 'bool'=False) ->Tuple[torch.Tensor, State]:
    """Creates a recurrent neural network specified by RNNCell ``cell``.

    Performs fully dynamic unrolling of ``inputs``.

    Args:
        cell: An instance of RNNCell.
        inputs: The RNN inputs.
            If ``time_major == False`` (default), this must be a ``Tensor``
            of shape: ``[batch_size, max_time, ...]``, or a nested
            tuple of such elements.
            If ``time_major == True``, this must be a ``Tensor`` of shape:
            ``[max_time, batch_size, ...]``, or a nested tuple of such
            elements.
            This may also be a (possibly nested) tuple of Tensors satisfying
            this property.  The first two dimensions must match across all the
            inputs, but otherwise the ranks and other shape components
            may differ. In this case, input to ``cell`` at each time-step
            will replicate the structure of these tuples, except for the
            time dimension (from which the time is taken).
            The input to ``cell`` at each time step will be a
            ``Tensor`` or (possibly nested) tuple of Tensors each with
            dimensions ``[batch_size, ...]``.
        sequence_length: (optional) An int32/int64 tensor sized
            ``[batch_size]``. Used to copy-through state and
            zero-out outputs when past a batch element's sequence length.
            So it's more for performance than correctness.
        initial_state: (optional) An initial state for the RNN.
            If ``cell.state_size`` is an integer, this must be
            a ``Tensor`` of appropriate type and shape
            ``[batch_size, cell.state_size]``. If ``cell.state_size`` is
            a tuple, this should be a tuple of tensors having shapes
            ``[batch_size, s]`` for ``s`` in ``cell.state_size``.
        time_major: The shape format of the ``inputs`` and ``outputs`` Tensors.
            If true, these ``Tensors`` must be shaped
            ``[max_time, batch_size, depth]``. If false, these ``Tensors``
            must be shaped ``[batch_size, max_time, depth]``.
            Using ``time_major = True`` is a bit more efficient because
            it avoids transposes at the beginning and end of the
            RNN calculation. However, most TensorFlow data is batch-major,
            so by default this function accepts input and emits output in
            batch-major form.

    Returns:
        A pair (outputs, state) where:

        outputs: The RNN output ``Tensor``.

            If time_major == False (default), this will be a ``Tensor`` shaped:
            ``[batch_size, max_time, cell.output_size]``.

            If time_major == True, this will be a ``Tensor`` shaped:
            ``[max_time, batch_size, cell.output_size]``.

            Note, if ``cell.output_size`` is a (possibly nested) tuple of
            integers or ``torch.Size`` objects, then ``outputs``
            will be a tuple having the same structure as ``cell.output_size``,
            containing Tensors having shapes corresponding to the shape
            data in ``cell.output_size``.

        state: The final state.  If ``cell.state_size`` is an int, this
            will be shaped ``[batch_size, cell.state_size]``.  If it is a
            ``torch.Size``, this will be shaped
            ``[batch_size] + cell.state_size``.
            If it is a (possibly nested) tuple of ints or ``torch.Size``,
            this will be a tuple having the corresponding shapes.
            If cells are ``LSTMCells``, ``state`` will be a tuple containing
            a ``LSTMStateTuple`` for each cell.

    Raises:
        TypeError: If ``cell`` is not an instance of RNNCell.
        ValueError: If inputs is None or an empty list.
    """
    if not time_major:
        inputs = inputs.permute(1, 0, 2)
    time_steps = inputs.shape[0]
    batch_size = inputs.shape[1]
    if sequence_length is not None:
        if not isinstance(sequence_length, torch.Tensor):
            sequence_length = torch.tensor(sequence_length, dtype=torch.int32, device=inputs.device)
        if sequence_length.dim() != 1:
            raise ValueError('sequence_length must be a vector of length batch_size, but saw shape: %s' % sequence_length.shape)
        if sequence_length.shape != torch.Size([batch_size]):
            raise ValueError('Expected shape for Tensor sequence_length is %s' % batch_size, ' but saw shape: %s' % sequence_length.shape)
    else:
        sequence_length = torch.tensor([time_steps] * batch_size, dtype=torch.int32, device=inputs.device)
    if initial_state is not None:
        state = initial_state
    else:
        state = cell.zero_state(batch_size=batch_size)
    outputs, final_state = _dynamic_rnn_loop(cell, inputs, state, sequence_length=sequence_length)
    if not time_major:
        outputs = outputs.permute(1, 0, 2)
    return outputs, final_state


_XLNET_PATH = 'https://storage.googleapis.com/xlnet/released_models/'


def init_weights(module: 'nn.Module'):
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, 0.0, 0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, 0.0, 0.02)


class PositionalEmbedding(nn.Module):
    inv_freq: 'torch.Tensor'

    def __init__(self, embed_dim: 'int'):
        super().__init__()
        freq_seq = torch.arange(0.0, embed_dim, 2.0)
        inv_freq = 1 / 10000 ** (freq_seq / embed_dim)
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, pos_seq: 'torch.Tensor') ->torch.Tensor:
        sinusoid = torch.ger(pos_seq, self.inv_freq)
        pos_embed = torch.cat([sinusoid.sin(), sinusoid.cos()], dim=-1)
        return pos_embed


def params_except_in(module: 'nn.Module', except_names: 'List[str]') ->Iterable[nn.Parameter]:
    return itertools.chain.from_iterable(child.parameters() for name, child in module.named_children() if name not in except_names)


def sum_tensors(xs: 'List[Optional[torch.Tensor]]') ->Optional[torch.Tensor]:
    """Sum a list of tensors with possible `None` values.

    Args:
        xs: A list of tensors.

    Returns:
        The summation of all the elements in the list.
    """
    idx = next((idx for idx, tensor in enumerate(xs) if tensor is not None), -1)
    if idx == -1:
        return None
    ret = xs[idx]
    for tensor in xs[idx + 1:]:
        if tensor is not None:
            ret = ret + tensor
    return ret


MaybeTuple = Union[T, Tuple[T, ...]]


OutputSize = MaybeTuple[Union[int, torch.Size]]


def _get_sizes(sizes: 'List[Any]') ->List[int]:
    """

    Args:
        sizes: A list of ``int`` or ``torch.Size``. If each element is of type
            ``torch.Size``, the size is computed by taking the product of the
            shape.

    Returns:
        A list of sizes with ``torch.Size`` replaced by product of its
        individual dimensions
    """
    if isinstance(sizes[0], torch.Size):
        size_list = [np.prod(shape) for shape in sizes]
    else:
        size_list = sizes
    return size_list


def _mlp_transform(inputs: 'TensorStruct', output_size: 'OutputSize', linear_layer: 'Optional[LinearLayer]'=None, activation_fn: 'Optional[ActivationFn]'=None) ->Any:
    """Transforms inputs through a fully-connected layer that creates
    the output with specified size.

    Args:
        inputs: A Tensor of shape `[batch_size, d1, ..., dn]`, or a (nested)
            tuple of such elements. The dimensions `d1, ..., dn` will be flatten
            and transformed by a dense layer.
        output_size: Can be an ``int``, a ``torch.Size``, or a (nested)
            tuple of ``int`` or ``torch.Size``.
        activation_fn: Activation function applied to the output.

    :returns:
        If :attr:`output_size` is an ``int`` or a ``torch.Size``,
        returns a tensor of shape ``[batch_size, *, output_size]``.
        If :attr:`output_size` is a tuple of ``int`` or ``torch.Size``,
        returns a tuple having the same structure as :attr:`output_size`,
        where each element has the same size as defined in :attr:`output_size`.
    """
    flat_input = nest.flatten(inputs)
    flat_input = [x.view(-1, x.size(-1)) for x in flat_input]
    concat_input = torch.cat(flat_input, 1)
    flat_output_size = nest.flatten(output_size)
    size_list = _get_sizes(flat_output_size)
    fc_output = concat_input
    if linear_layer is not None:
        fc_output = linear_layer(fc_output)
    if activation_fn is not None:
        fc_output = activation_fn(fc_output)
    flat_output = torch.split(fc_output, size_list, dim=1)
    flat_output = list(flat_output)
    if isinstance(flat_output_size[0], torch.Size):
        flat_output = [torch.reshape(output, (-1,) + shape) for output, shape in zip(flat_output, flat_output_size)]
    output = nest.pack_sequence_as(structure=output_size, flat_sequence=flat_output)
    return output


def _sum_output_size(output_size: 'OutputSize') ->int:
    """Return sum of all dim values in :attr:`output_size`

    Args:
        output_size: Can be an ``int``, a ``torch.Size``, or a (nested)
            tuple of ``int`` or ``torch.Size``.
    """
    flat_output_size = nest.flatten(output_size)
    size_list = _get_sizes(flat_output_size)
    ret = sum(size_list)
    return ret


def get_activation_fn(fn_name: 'Optional[Union[str, Callable[[torch.Tensor], torch.Tensor]]]'=None, kwargs: 'Union[HParams, Dict, None]'=None) ->Optional[Callable[[torch.Tensor], torch.Tensor]]:
    """Returns an activation function `fn` with the signature
    `output = fn(input)`.

    If the function specified by :attr:`fn_name` has more than one arguments
    without default values, then all these arguments except the input feature
    argument must be specified in :attr:`kwargs`. Arguments with default values
    can also be specified in :attr:`kwargs` to take values other than the
    defaults. In this case a partial function is returned with the above
    signature.

    Args:
        fn_name (str or callable): An activation function, or its name or
            module path. The function can be:

            - Built-in function defined in
              :torch_docs:`torch.nn.functional<nn.html#torch-nn-functional>`
            - User-defined activation functions in module
              :mod:`texar.torch.custom`.
            - External activation functions. Must provide the full module path,
              e.g., ``"my_module.my_activation_fn"``.

        kwargs (optional): A `dict` or instance of :class:`~texar.torch.HParams`
            containing the keyword arguments of the activation function.

    Returns:
        An activation function. `None` if :attr:`fn_name` is `None`.
    """
    if fn_name is None:
        return None
    fn_modules = ['torch', 'torch.nn.functional', 'texar.torch.custom', 'texar.torch.core.layers']
    activation_fn_ = utils.get_function(fn_name, fn_modules)
    activation_fn = activation_fn_
    if kwargs is not None:
        if isinstance(kwargs, HParams):
            kwargs = kwargs.todict()

        def _partial_fn(features):
            return activation_fn_(features, **kwargs)
        activation_fn = _partial_fn
    return activation_fn


def _assert_same_size(outputs: 'TensorStruct', output_size: 'OutputSize'):
    """Check if outputs match output_size

    Args:
        outputs: A tensor or a (nested) tuple of tensors
        output_size: Can be an ``int``, a ``torch.Size``, or a (nested)
            tuple of ``int`` or ``torch.Size``.
    """
    flat_output_size = nest.flatten(output_size)
    flat_output = nest.flatten(outputs)
    for output, size in zip(flat_output, flat_output_size):
        if isinstance(size, torch.Size):
            if output[0].size() != size:
                raise ValueError('The output size does not matchthe required output_size')
        elif output[0].size()[-1] != size:
            raise ValueError('The output size does not match the required output_size')


HelperInitTuple = Tuple[torch.ByteTensor, torch.Tensor]


IDType = TypeVar('IDType', bound=torch.Tensor)


NextInputTuple = Tuple[torch.ByteTensor, torch.Tensor]


class Helper(Generic[IDType], ABC):
    """Interface for implementing sampling in seq2seq decoders.

    Please refer to the documentation for the TensorFlow counterpart
    `tf.contrib.seq2seq.Helper
    <https://www.tensorflow.org/api_docs/python/tf/contrib/seq2seq/Helper>`_.
    """

    def initialize(self, embedding_fn: 'EmbeddingFn', inputs: 'Optional[torch.Tensor]', sequence_length: 'Optional[torch.LongTensor]') ->HelperInitTuple:
        """Initialize the current batch.

        Args:
            embedding_fn: A function taking input tokens and timestamps,
                returning embedding tensors.
            inputs: Input tensors.
            sequence_length: An int32 vector tensor.

        Returns:
            ``(initial_finished, initial_inputs)``.
        """
        raise NotImplementedError

    def sample(self, time: 'int', outputs: 'torch.Tensor') ->IDType:
        """Returns ``sample_ids``.
        """
        raise NotImplementedError

    def next_inputs(self, embedding_fn: 'EmbeddingFn', time: 'int', outputs: 'torch.Tensor', sample_ids: 'IDType') ->NextInputTuple:
        """Returns ``(finished, next_inputs, next_state)``.
        """
        raise NotImplementedError


class XLNetDecoderOutput(NamedTuple):
    """The output of :class:`XLNetDecoder`.
    """
    logits: 'torch.Tensor'
    """A :tensor:`Tensor` of shape ``[batch_size, max_time, vocab_size]``
    containing the logits."""
    sample_id: 'torch.LongTensor'
    """A :tensor:`LongTensor` of shape ``[batch_size, max_time]``
    (or ``[batch_size, max_time, vocab_size]``) containing the sampled token
    indices. Note that the shape of ``sample_id`` is different for different
    decoding strategy or helper. Please refer to
    :class:`~texar.torch.modules.Helper` for the detailed information."""


Output = XLNetDecoderOutput


torch_bool = (torch.empty(()) < 0).dtype


_CHECKPOINT_FILES_GEN_MAP = {'small': (1000000, 16), 'base': (999900, 16), 'large': (1000700, 8), 'B': (1000000, 64)}


_T5_PATH = 'https://storage.googleapis.com/t5-data/pretrained_models/'


_T5_VOCAB_PATH = 'https://storage.googleapis.com/t5-data/vocabs/cc_all.32000/'


def _generate_t5_file_list(ckpt_tuple: 'tuple') ->List[str]:
    """ Helper function to generate file list given a tuple of model_id and
    partition size.

    Args:
        ckpt_tuple: A tuple of model_id and number of partitions

    """
    ckpt_id = ckpt_tuple[0]
    ckpt_parts = ckpt_tuple[1]
    return ['checkpoint', *[f'model.ckpt-{ckpt_id}.data-{idx:05d}-of-{ckpt_parts:05d}' for idx in range(ckpt_parts)], f'model.ckpt-{ckpt_id}.index', f'model.ckpt-{ckpt_id}.meta', 'operative_config.gin']


IMPORTANT_PARAMS = 'd_ff', 'd_kv', 'd_model', 'dropout', 'num_heads', 'num_layers', 'inputs_length'


def read_t5_gin_config_file(config_file_path: 'str') ->Dict:
    """Simple helper function to read a gin file
    and get hyperparameters for T5.

    Args:
        config_file_path: path of config.gin file as a string.

    Returns:
        A dictionary with important parameters for loading T5.

    """
    config = {}
    with open(config_file_path, 'r') as gin_file:
        for line in gin_file:
            if line.startswith(IMPORTANT_PARAMS):
                assignment = line.strip().split()
                assert len(assignment) == 3
                arg_name, _, value = assignment
                config[arg_name] = ast.literal_eval(value)
    return config


class T5LayerNorm(nn.Module):
    """ Custom LayerNorm for T5 with no mean subtraction and no bias.
    """

    def __init__(self, input_size: 'int', eps: 'float'=1e-05):
        super().__init__()
        self.w = nn.Parameter(torch.ones(input_size))
        self.eps = eps

    def forward(self, x: 'torch.Tensor'):
        x = x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.w * x


class EmbeddingHelper(Helper[IDType], ABC):
    """A generic helper for use during inference.

    Uses output logits for sampling, and passes the result through an embedding
    layer to get the next input.

    Args:
        start_tokens: 1D :tensor:`LongTensor` shaped ``[batch_size]``,
            representing the start tokens for each sequence in batch.
        end_token: Python int or scalar :tensor:`LongTensor`, denoting the
            token that marks end of decoding.

    Raises:
        ValueError: if :attr:`start_tokens` is not a 1D tensor or
            :attr:`end_token` is not a scalar.
    """
    _start_inputs: 'torch.Tensor'

    def __init__(self, start_tokens: 'torch.LongTensor', end_token: 'Union[int, torch.LongTensor]'):
        if start_tokens.dim() != 1:
            raise ValueError('start_tokens must be a vector')
        if not isinstance(end_token, int) and end_token.dim() != 0:
            raise ValueError('end_token must be a scalar')
        self._start_tokens = start_tokens
        self._batch_size = start_tokens.size(0)
        if isinstance(end_token, int):
            self._end_token = start_tokens.new_tensor(end_token)
        else:
            self._end_token = end_token

    @property
    def batch_size(self) ->int:
        return self._batch_size

    def initialize(self, embedding_fn: 'EmbeddingFn', inputs: 'Optional[torch.Tensor]', sequence_length: 'Optional[torch.LongTensor]') ->HelperInitTuple:
        del inputs, sequence_length
        times = torch.zeros_like(self._start_tokens)
        self._start_inputs = embedding_fn(self._start_tokens, times)
        finished = torch.zeros_like(self._start_tokens, dtype=torch_bool)
        return finished, self._start_inputs


class TransformerDecoderOutput(NamedTuple):
    """The output of :class:`TransformerDecoder`.
    """
    logits: 'torch.Tensor'
    """A :tensor:`Tensor` of shape ``[batch_size, max_time, vocab_size]``
    containing the logits."""
    sample_id: 'torch.LongTensor'
    """A :tensor:`LongTensor` of shape ``[batch_size, max_time]``
    (or ``[batch_size, max_time, vocab_size]``) containing the sampled
    token indices. Note that the shape of ``sample_id`` is different for
    different decoding strategy or helper. Please refer to
    :class:`~texar.torch.modules.Helper` for the detailed information."""


def identity(inputs: 'torch.Tensor'):
    """Returns a tensor with the same content as the input tensor.

    Arguments:
        inputs: The input tensor.

    Returns:
        A tensor of the same shape, type, and content.
    """
    return inputs


def _make_output_layer(layer: 'Optional[Union[nn.Module, torch.Tensor]]', vocab_size: 'Optional[int]', output_size: 'int', bias: 'bool') ->Tuple[nn.Module, Optional[int]]:
    """Construct the output layer for decoders. Based on the input, multiple
    types of output layers could be constructed:

    - If ``layer`` is a :torch_nn:`Module`, then the layer is returned as is.
    - If ``layer`` is `None`, then a :torch_nn:`Linear` layer is constructed
      with ``output_size`` and ``vocab_size`` as input and output dimensions.
    - If ``layer`` is a :tensor:`Tensor`, then a :torch_nn:`Linear` layer is
      constructed with the provided tensor as parameters. Note that this tensor
      should have transposed shape, i.e. shape of ``[vocab_size, output_size]``.
      Also, if the provided tensor is not an instance of :torch_nn:`Parameter`,
      it will **not** accumulate gradients.
    - If ``layer`` is :method:`texar.torch.core.identity`, identity function is
      used as the output layer.
    """
    if isinstance(layer, nn.Module):
        output_layer = layer
    elif layer is None:
        if vocab_size is None:
            raise ValueError('Either `output_layer` or `vocab_size` must be provided. Set `output_layer=tx.core.identity` if no output layer is wanted.')
        output_layer = nn.Linear(output_size, vocab_size, bias)
    elif torch.is_tensor(layer):
        vocab_size = layer.size(0)
        output_layer = nn.Linear(layer.size(1), vocab_size, bias)
        if not isinstance(layer, nn.Parameter):
            layer = nn.Parameter(layer, requires_grad=False)
        output_layer.weight = layer
    elif layer is identity:
        output_layer = Identity()
    else:
        raise ValueError(f'output_layer should be an instance of `nn.Module`, a tensor,or None. Unsupported type: {type(layer)}')
    return output_layer, vocab_size


INF = 1.0 * 10000000.0


def _expand_to_beam_size(tensor: 'Any', beam_size: 'int') ->Any:
    """Tiles a given tensor by :attr:`beam_size`.

    Args:
        tensor: tensor to tile. Shape: `[batch_size, ...]`.
        beam_size: How much to tile the tensor by.

    Returns:
        Tiled tensor of shape `[batch_size, beam_size, ...]`.
    """
    if not isinstance(tensor, torch.Tensor):
        return tensor
    tensor = torch.unsqueeze(tensor, dim=1)
    tile_dims = [1] * len(tensor.size())
    tile_dims[1] = beam_size
    return tensor.repeat(tuple(tile_dims))


def _merge_beam_dim(tensor: 'Any') ->Any:
    """Reshapes first two dimensions in to single dimension.

    Args:
        tensor: Tensor to reshape of shape `[A, B, ...]`.

    Returns:
        Reshaped tensor of shape `[A * B, ...]`.
    """
    if not isinstance(tensor, torch.Tensor):
        return tensor
    shape = list(tensor.size())
    shape[0] *= shape[1]
    shape.pop(1)
    return tensor.view(tuple(shape))


def _unmerge_beam_dim(tensor: 'Any', batch_size: 'int', beam_size: 'int') ->Any:
    """Reshapes first dimension back to `[batch_size, beam_size]`.

    Args:
        tensor: Tensor to reshape of shape `[batch_size * beam_size, ...]`.
        batch_size: int, original batch size.
        beam_size: int, original beam size.

    Returns:
        Reshaped tensor of shape `[batch_size, beam_size, ...]`.
    """
    if not isinstance(tensor, torch.Tensor):
        return tensor
    shape = list(tensor.size())
    new_shape = [batch_size] + [beam_size] + shape[1:]
    return tensor.view(tuple(new_shape))


def compute_batch_indices(batch_size: 'int', beam_size: 'int') ->torch.LongTensor:
    """Computes the i-th coordinate that contains the batch index for
    gathers.

    The batch index tensor is a tensor like `[[0,0,0,0,],[1,1,1,1],..]`.
    It says which batch the beam item is in. This will create the first
    dimension of the 2D coordinates needed for the gather.

    Args:
        batch_size: Batch size
        beam_size: Size of the beam.

    Returns:
        `[batch_size, beam_size]` tensor of ids.
    """
    batch_pos = torch.arange(batch_size)
    batch_pos = batch_pos.view(-1, 1).expand(batch_size, beam_size)
    return batch_pos


def gather_nd(params: 'Any', indices: 'torch.Tensor') ->Any:
    if not isinstance(params, torch.Tensor):
        return params
    assert len(indices.size()) == 3
    orig_size = params.size()
    index = indices[:, :, 1].view(-1) + indices[:, :, 0].view(-1) * orig_size[1]
    ret = torch.index_select(params.view(-1, *params.size()[2:]), dim=0, index=index)
    ret = ret.view(orig_size[0], indices.size(1), *orig_size[2:])
    return ret


def compute_topk_scores_and_seq(sequences: 'torch.LongTensor', scores: 'torch.Tensor', scores_to_gather: 'torch.Tensor', flags: 'torch.ByteTensor', beam_size: 'int', batch_size: 'int', states_to_gather: 'Optional[State]'=None) ->Tuple[torch.LongTensor, torch.Tensor, torch.ByteTensor, Optional[State]]:
    """Given sequences and scores, will gather the top-k (`k = beam`) size
    sequences.

    This function is used to grow alive, and finished. It takes sequences,
    scores, and flags, and returns the top k from sequence
    :attr:`scores_to_gather`, and flags based on the values in scores.

    Args:
        sequences: Tensor of sequences that we need to gather from.
            Shape: `[batch_size, beam_size, seq_length]`.
        scores: Tensor of scores for each sequence in sequences. We will use
            these to compute the top-k. Shape: `[batch_size, beam_size]`.
        scores_to_gather: Tensor of scores for each sequence in sequences.
            Shape: `[batch_size, beam_size]`.
            We will return the gathered scores from here.
            Scores to gather is different from scores because for
            grow_alive, we will need to return log-probabilities, while for
            grow_finished, we will need to return the length penalized
            scores.
        flags: Tensor of booleans for sequences that say whether a sequence
            has reached `EOS`.
        beam_size: int
        batch_size: int
        states_to_gather: (possibly nested structure of) decoding states.

    :returns: Tuple of:

        - `topk_seq`: `[batch_size, beam_size, decode_length]`.
        - `topk_gathered_scores`: `[batch_size, beam_size]`.
        - `topk_finished_flags`: `[batch_size, beam_size]`.
    """
    _, topk_indexes = torch.topk(scores, k=beam_size)
    batch_pos = compute_batch_indices(batch_size, beam_size)
    batch_pos = batch_pos
    top_coordinates = torch.stack([batch_pos, topk_indexes], dim=2)
    topk_seq = gather_nd(sequences, top_coordinates)
    topk_flags = gather_nd(flags, top_coordinates)
    topk_gathered_scores = gather_nd(scores_to_gather, top_coordinates)
    if states_to_gather is not None:
        topk_gathered_states = map_structure(lambda state: gather_nd(state, top_coordinates), states_to_gather)
    else:
        topk_gathered_states = states_to_gather
    return topk_seq, topk_gathered_scores, topk_flags, topk_gathered_states


def log_prob_from_logits(logits: 'torch.Tensor') ->torch.Tensor:
    return logits - torch.logsumexp(logits, dim=-1, keepdim=True)


def beam_search(symbols_to_logits_fn, initial_ids, beam_size, decode_length, vocab_size, alpha, eos_id, states=None, stop_early=True):
    """Beam search with length penalties.

    Requires a function that can take the currently decoded symbols and
    return the logits for the next symbol. The implementation is inspired
    by https://arxiv.org/abs/1609.08144.

    Variables used within this function follow the naming pattern:
    `(alive|finished)_topk_(seq,scores)`.

    Variables marked `alive` represent the new beam sequences that will be
    processed in the next step.    Variables marked `finished` represent
    the completed beam sequences, which may be padded with 0 if no beams
    finished.

    Variables marked `seq` store the full beam sequence for the time step.
    Variables marked `scores` store the sequence's final log scores.

    The beam search steps will be processed sequentially in order, so when
    capturing observed from these operations, tensors, clients can make
    assumptions about which step is being recorded.

    Args:
        symbols_to_logits_fn: Interface to the model, to provide logits.
            Should take `[batch_size, decoded_ids]` and return
            `[batch_size, vocab_size]`.
        initial_ids: LongTensor of shape `[batch_size]`. IDs to start off the
            decoding, this will be the first thing handed to
            :attr:`symbols_to_logits_fn` (after expanding to beam size).
        beam_size: Size of the beam.
        decode_length: Number of steps to decode for.
        vocab_size: Size of the vocab, must equal the size of the logits
            returned by :attr:`symbols_to_logits_fn`.
        alpha: alpha for length penalty.
        eos_id: ID for end of sentence.
        states: (possibly nested structure of) decoding states.
        stop_early: a boolean - stop once best sequence is provably
            determined.

    Returns:
        Tuple of

        - decoded beams (shape: `[batch_size, beam_size, decode_length]`)
        - decoding probabilities (shape: `[batch_size, beam_size]`)
    """
    batch_size = initial_ids.size()[0]
    initial_log_probs = torch.Tensor([[0.0] + [-float('inf')] * (beam_size - 1)])
    initial_log_probs = initial_log_probs
    alive_log_probs = initial_log_probs.repeat((batch_size, 1))
    alive_seq = _expand_to_beam_size(initial_ids, beam_size)
    alive_seq = torch.unsqueeze(alive_seq, dim=2)
    if states is not None:
        states = map_structure(lambda state: _expand_to_beam_size(state, beam_size), states)
    finished_seq = torch.zeros(alive_seq.size(), dtype=torch.long)
    finished_scores = torch.full((batch_size, beam_size), -INF)
    finished_flags = torch.zeros((batch_size, beam_size), dtype=torch_bool)
    finished_seq = finished_seq
    finished_scores = finished_scores
    finished_flags = finished_flags

    def grow_finished(finished_seq: 'torch.LongTensor', finished_scores: 'torch.Tensor', finished_flags: 'torch.ByteTensor', curr_seq: 'torch.LongTensor', curr_scores: 'torch.Tensor', curr_finished: 'torch.ByteTensor') ->Tuple[torch.LongTensor, torch.Tensor, torch.ByteTensor]:
        """Given sequences and scores, will gather the top-k (`k = beam`) size
        sequences.

        Args:
            finished_seq: Finished sequences.
                Shape: `[batch_size, beam_size, current_decoded_length]`.
            finished_scores: Scores for each finished sequences.
                Shape: `[batch_size, beam_size]`.
            finished_flags: Finished flags for each of these sequences.
                Shape: `[batch_size, beam_size]`
            curr_seq: Top-k sequences that has been grown by one
                position.
                Shape: `[batch_size, beam_size, current_decoded_length]`.
            curr_scores: Scores for each of the top-k sequences.
                Shape: `[batch_size, beam_size]`.
            curr_finished: Finished flags for each of the top-k sequences.
                Shape: `[batch_size, beam_size]`.

        Returns:
            Tuple of

            - Top-k sequences based on scores.
            - Log-probabilities of these sequences.
            - Finished flags of these sequences.
        """
        _appended = torch.zeros(batch_size, beam_size, 1, dtype=torch.long)
        _appended = _appended
        finished_seq = torch.cat([finished_seq, _appended], dim=2)
        curr_scores = curr_scores + (1.0 - curr_finished.float()) * -INF
        curr_finished_seq = torch.cat([finished_seq, curr_seq], dim=1)
        curr_finished_scores = torch.cat([finished_scores, curr_scores], dim=1)
        curr_finished_flags = torch.cat([finished_flags, curr_finished], dim=1)
        next_seq, next_scores, next_flags, _ = compute_topk_scores_and_seq(curr_finished_seq, curr_finished_scores, curr_finished_scores, curr_finished_flags, beam_size, batch_size)
        return next_seq, next_scores, next_flags

    def grow_alive(curr_seq: 'torch.LongTensor', curr_scores: 'torch.Tensor', curr_log_probs: 'torch.Tensor', curr_finished: 'torch.ByteTensor', states: 'Optional[State]') ->Tuple[torch.LongTensor, torch.Tensor, torch.ByteTensor, Optional[State]]:
        """Given sequences and scores, will gather the top k=beam size
        sequences.

        Args:
            curr_seq: Current top-k sequences that has been grown by one
                position.
                Shape: `[batch_size, beam_size, i + 1]`.
            curr_scores: Scores for each of these sequences.
                Shape: `[batch_size, beam_size]`.
            curr_log_probs: Log-probabilities for each of these sequences.
                Shape: `[batch_size, beam_size]`.
            curr_finished: Finished flags for each of these sequences.
                Shape: `[batch_size, beam_size]`.
            states: (possibly nested structure of) decoding states.

        :returns: Tuple of:

            - Top-k sequences based on scores.
            - Log-probabilities of these sequences.
            - Finished flags of these sequences.
            - Decoding states for these sequences.
        """
        curr_scores = curr_scores + curr_finished.float() * -INF
        return compute_topk_scores_and_seq(curr_seq, curr_scores, curr_log_probs, curr_finished, beam_size, batch_size, states)

    def grow_topk(i: 'int', alive_seq: 'torch.LongTensor', alive_log_probs: 'torch.Tensor', states: 'Optional[State]') ->Tuple[torch.LongTensor, torch.Tensor, torch.Tensor, torch.ByteTensor, Optional[State]]:
        """Inner beam search loop.

        This function takes the current alive sequences, and grows them to
        top-k sequences where `k = 2 * beam`. We use `2 * beam` because we could
        have `beam_size` number of sequences that might hit `<EOS>` and there
        will be no alive sequences to continue. With `2 * beam_size`, this
        will not happen. This relies on the assumption the vocab size is >
        beam size. If this is true, we'll have at least `beam_size` non-`<EOS>`
        extensions if we extract the next top `2 * beam` words.
        Length penalty is given by :math:`(5+len(decode)/6) ^ -\\alpha`.

        Please refer to https://arxiv.org/abs/1609.08144.

        Args:
            i: loop index
            alive_seq: Top-k sequences decoded so far.
                Shape: `[batch_size, beam_size, i + 1]`.
            alive_log_probs: Log-probabilities of these sequences.
                Shape: `[batch_size, beam_size]`
            states: (possibly nested structure of) decoding states.

        :returns: Tuple of:

            - Top-k sequences extended by the next word.
            - Log-probabilities of these sequences,
            - The scores with length penalty of these sequences,
            - Flags indicating which of these sequences have finished
              decoding.
            - Transformed decoding states with same structure as :attr:`state`.
        """
        flat_ids = alive_seq.view(batch_size * beam_size, -1)
        if states is not None:
            flat_states = map_structure(_merge_beam_dim, states)
            flat_logits, flat_states = symbols_to_logits_fn(flat_ids, flat_states)
            states = map_structure(lambda t: _unmerge_beam_dim(t, batch_size, beam_size), flat_states)
        else:
            flat_logits = symbols_to_logits_fn(flat_ids)
        logits = flat_logits.view(batch_size, beam_size, -1)
        candidate_log_probs = log_prob_from_logits(logits)
        log_probs = candidate_log_probs + alive_log_probs.unsqueeze(dim=2)
        length_penalty = ((5.0 + float(i + 1)) / 6.0) ** alpha
        curr_scores = log_probs / length_penalty
        flat_curr_scores = curr_scores.view(-1, beam_size * vocab_size)
        topk_scores, topk_ids = torch.topk(flat_curr_scores, k=beam_size * 2)
        topk_log_probs = topk_scores * length_penalty
        topk_beam_index = topk_ids // vocab_size
        topk_ids %= vocab_size
        batch_pos = compute_batch_indices(batch_size, beam_size * 2)
        batch_pos = batch_pos
        topk_coordinates = torch.stack([batch_pos, topk_beam_index], dim=2)
        topk_seq = gather_nd(alive_seq, topk_coordinates)
        if states is not None:
            states = map_structure(lambda state: gather_nd(state, topk_coordinates), states)
        topk_seq = torch.cat([topk_seq, topk_ids.unsqueeze(dim=2)], dim=2)
        topk_finished = topk_ids == eos_id
        return topk_seq, topk_log_probs, topk_scores, topk_finished, states

    def inner_loop(i: 'int', alive_seq: 'torch.LongTensor', alive_log_probs: 'torch.Tensor', finished_seq: 'torch.LongTensor', finished_scores: 'torch.Tensor', finished_flags: 'torch.ByteTensor', states: 'Optional[State]') ->Tuple[int, torch.LongTensor, torch.Tensor, torch.LongTensor, torch.Tensor, torch.ByteTensor, Optional[State]]:
        """Inner beam search loop.

        There are three groups of tensors: `alive`, `finished`, and `top-k`.

        - The `alive` group contains information about the current alive
          sequences.
        - The `top-k` group contains information about `alive + top_k`
          current decoded words.
        - The `finished` group contains information about finished sentences,
          that is, the ones that have decoded to `<EOS>`. These are what we
          return.

        The general beam search algorithm is as follows:

            While not terminated (please refer to termination condition):

            1. Grow the current `alive` to get `beam * 2` top-k sequences.
            2. Among the `top-k`, move the top `beam_size` ones that haven't
               reached `EOS` into `alive`.
            3. Among the `top-k`, move the top `beam_size` ones have reached
               `EOS` into `finished`.

            Repeat

        To make things simple with using fixed size tensors, we will end
        up inserting unfinished sequences into finished in the beginning.
        To prevent that we add `-INF` to the score of the unfinished
        sequence so that when a true finished sequence does appear, it
        will have a higher score than all the unfinished ones.

        Args:
            i: Loop index
            alive_seq: Topk sequences decoded so far
                Shape: `[batch_size, beam_size, i + 1]`.
            alive_log_probs: Log-probabilities of the beams.
                Shape: `[batch_size, beam_size]`
            finished_seq: Current finished sequences.
                Shape: `[batch_size, beam_size, i+1]`.
            finished_scores: Scores for each of these sequences.
                Shape: `[batch_size, beam_size]`.
            finished_flags: Finished flags for each of these sequences.
                Shape: `[batch_size, beam_size]`
            states: (possibly nested structure of) decoding states.

        :returns: Tuple of:

            - Incremented loop index.
            - New `alive` sequences.
            - Log-probabilities of the `alive` sequences.
            - New `finished` sequences.
            - Scores of the `finished` sequences.
            - Flags indicating which sequences in `finished` has reached `EOS`.
            - Final decoding states with same structure as :attr:`state`.
        """
        topk_seq, topk_log_probs, topk_scores, topk_finished, states = grow_topk(i, alive_seq, alive_log_probs, states)
        alive_seq, alive_log_probs, _, states = grow_alive(topk_seq, topk_scores, topk_log_probs, topk_finished, states)
        finished_seq, finished_scores, finished_flags = grow_finished(finished_seq, finished_scores, finished_flags, topk_seq, topk_scores, topk_finished)
        return i + 1, alive_seq, alive_log_probs, finished_seq, finished_scores, finished_flags, states

    def _is_finished(i: 'int', alive_log_probs: 'torch.Tensor', finished_scores: 'torch.Tensor') ->bool:
        """Check termination condition.

        We terminate when we decoded up to `decode_length` or the lowest
        scoring item in finished has a greater score that the highest probable
        item in alive divided by the max length penalty.

        Args:
            i: Loop index
            alive_log_probs: Log-probabilities of the beams.
                Shape: `[batch_size, beam_size]`.
            finished_scores: Scores for each of these sequences.
                Shape: `[batch_size, beam_size]`.

        Returns:
            Bool.
        """
        max_length_penalty = ((5.0 + float(decode_length)) / 6.0) ** alpha
        lower_bound_alive_scores = alive_log_probs[:, 0] / max_length_penalty
        if not stop_early:
            lowest_score_of_finished_in_finished = torch.min(finished_scores)
        else:
            lowest_score_of_finished_in_finished, _ = torch.max(finished_scores, dim=1)
        bound_is_met = (lowest_score_of_finished_in_finished > lower_bound_alive_scores).all().item()
        ret = (i < decode_length) & ~bound_is_met
        return ret
    step = 0
    while _is_finished(step, alive_log_probs, finished_scores):
        step, alive_seq, alive_log_probs, finished_seq, finished_scores, finished_flags, states = inner_loop(step, alive_seq, alive_log_probs, finished_seq, finished_scores, finished_flags, states)
    ret_seq, ret_scores = [], []
    for idx, flag_per_instance in enumerate(finished_flags.any(dim=1).tolist()):
        if flag_per_instance:
            ret_seq.append(finished_seq[idx])
            ret_scores.append(finished_scores[idx])
        else:
            ret_seq.append(alive_seq[idx])
            ret_scores.append(alive_log_probs[idx])
    ret_seq = torch.stack(ret_seq, dim=0)
    ret_scores = torch.stack(ret_scores, dim=0)
    return ret_seq, ret_scores


def reverse_sequence(inputs: 'torch.Tensor', seq_lengths: 'Union[torch.LongTensor, List[int]]', time_major: 'bool') ->torch.Tensor:
    """Reverses variable length slices.

    This op first slices input along the dimension batch_axis, and for each
    slice i, reverses the first seq_lengths[i] elements along the dimension
    seq_axis.

    The elements of seq_lengths must obey seq_lengths[i] <=
    input.dims[seq_dim], and seq_lengths must be a vector of length
    input.dims[batch_dim].

    The output slice i along dimension batch_axis is then given by input slice
    i, with the first seq_lengths[i] slices along dimension seq_axis reversed.

    Args:
        inputs: A Tensor. The input to reverse.
        seq_lengths: A Tensor. Must be one of the following types: int32,
            int64. 1-D with length input.dims(batch_dim) and
            max(seq_lengths) <= input.dims(seq_dim)
        time_major: The shape format of the ``inputs`` and ``outputs`` Tensors.
            If true, these ``Tensors`` must be shaped
            ``[max_time, batch_size, depth]``. If false, these ``Tensors`` must
            be shaped ``[batch_size, max_time, depth]``.
            Using ``time_major = True`` is a bit more efficient because it
            avoids transposes at the beginning and end of the RNN calculation.
            However, most TensorFlow data is batch-major, so by
            default this functionb accepts input and emits output
            in batch-major form.

    Returns:
        A ``Tensor``. Has the same type as input.
    """
    if time_major:
        inputs = inputs.permute(1, 0, 2)
    batch_size = inputs.shape[0]
    outputs = inputs.clone()
    for i in range(batch_size):
        outputs[i][0:seq_lengths[i]] = torch.flip(inputs[i][0:seq_lengths[i]], dims=(0,))
    if time_major:
        outputs = outputs.permute(1, 0, 2)
    return outputs


def bidirectional_dynamic_rnn(cell_fw: 'RNNCellBase[State]', cell_bw: 'RNNCellBase[State]', inputs: 'torch.Tensor', sequence_length: 'Optional[Union[torch.LongTensor, List[int]]]'=None, initial_state_fw: 'Optional[State]'=None, initial_state_bw: 'Optional[State]'=None, time_major: 'bool'=False) ->Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[State, State]]:
    """Creates a dynamic version of bidirectional recurrent neural network.

    Takes input and builds independent forward and backward RNNs. The
    input_size of forward and backward cell must match. The initial state for
    both directions is zero by default (but can be set optionally) and no
    intermediate states are ever returned -- the network is fully unrolled
    for the given (passed in) length(s) of the sequence(s) or completely
    unrolled if length(s) is not given.

    Args:
        cell_fw: An instance of RNNCell, to be used for forward direction.
        cell_bw: An instance of RNNCell, to be used for backward direction.
        inputs: The RNN inputs.
            If time_major == False (default), this must be a tensor of shape:
            ``[batch_size, max_time, ...]``, or a nested tuple of such elements.
            If time_major == True, this must be a tensor of shape:
            ``[max_time, batch_size, ...]``, or a nested tuple of such elements.
        sequence_length: (optional) An int32/int64 tensor, size
            ``[batch_size]``, containing the actual lengths for each of the
            sequences in
            the batch. If not provided, all batch entries are assumed
            to be full sequences; and time reversal is applied from time
            ``0`` to ``max_time`` for each sequence.
        initial_state_fw: (optional) An initial state for the forward RNN.
            This must be a tensor of appropriate type and shape
            ``[batch_size, cell_fw.state_size]``.
            If ``cell_fw.state_size`` is a tuple, this should be a tuple of
            tensors having shapes ``[batch_size, s]``
            for ``s`` in ``cell_fw.state_size``.
        initial_state_bw: (optional) Same as for ``initial_state_fw``, but using
            the corresponding properties of ``cell_bw``.
        time_major: The shape format of the ``inputs`` and ``outputs`` Tensors.
            If true, these ``Tensors`` must be shaped
            ``[max_time, batch_size, depth]``.
            If false, these ``Tensors`` must be shaped
            ``[batch_size, max_time, depth]``.
            Using ``time_major = True`` is a bit more efficient because it
            avoids transposes at the beginning and end of the RNN calculation.
            However, most TensorFlow data is batch-major, so by
            default this function accepts input and emits output
            in batch-major form.

    Returns:
        A tuple (outputs, output_states) where:

        outputs: A tuple (output_fw, output_bw) containing the forward and
            the backward rnn output ``Tensor``.
            If time_major == False (default),
                output_fw will be a ``Tensor`` shaped:
                ``[batch_size, max_time, cell_fw.output_size]``
                and output_bw will be a ``Tensor`` shaped:
                ``[batch_size, max_time, cell_bw.output_size]``.
            If time_major == True,
                output_fw will be a ``Tensor`` shaped:
                ``[max_time, batch_size, cell_fw.output_size]``
                and output_bw will be a ``Tensor`` shaped:
                ``[max_time, batch_size, cell_bw.output_size]``.
            It returns a tuple instead of a single concatenated ``Tensor``,
            unlike in the ``bidirectional_rnn``. If the concatenated
            one is preferred, the forward and backward outputs can
            be concatenated as ``tf.concat(outputs, 2)``.
        output_states: A tuple (output_state_fw, output_state_bw) containing
            the forward and the backward final states of bidirectional rnn.
    """
    output_fw, output_state_fw = dynamic_rnn(cell=cell_fw, inputs=inputs, sequence_length=sequence_length, initial_state=initial_state_fw, time_major=time_major)
    if time_major:
        time_steps = inputs.shape[0]
        batch_size = inputs.shape[1]
    else:
        time_steps = inputs.shape[1]
        batch_size = inputs.shape[0]
    if sequence_length is None:
        sequence_length = torch.tensor([time_steps] * batch_size, dtype=torch.int32, device=inputs.device)
    inputs_reverse = reverse_sequence(inputs=inputs, seq_lengths=sequence_length, time_major=time_major)
    tmp, output_state_bw = dynamic_rnn(cell=cell_bw, inputs=inputs_reverse, sequence_length=sequence_length, initial_state=initial_state_bw, time_major=time_major)
    output_bw = reverse_sequence(inputs=tmp, seq_lengths=sequence_length, time_major=time_major)
    outputs = output_fw, output_bw
    output_states = output_state_fw, output_state_bw
    return outputs, output_states


_ROBERTA_PATH = 'https://dl.fbaipublicfiles.com/fairseq/models/'


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (AvgReducePool1d,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (BertGELU,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Flatten,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (GPTGELU,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Identity,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (LSTMCell,
     lambda: ([], {'input_size': 4, 'hidden_size': 4}),
     lambda: ([torch.rand([4, 4])], {})),
    (MaxReducePool1d,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (MergeLayer,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (T5LayerNorm,
     lambda: ([], {'input_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
]

