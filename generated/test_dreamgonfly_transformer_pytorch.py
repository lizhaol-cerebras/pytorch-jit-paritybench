
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


import math


import numpy as np


from collections import defaultdict


from torch.optim import Adam


from torch.utils.data import DataLoader


import random


class PositionalEncoding(nn.Module):
    """
    Implements the sinusoidal positional encoding for
    non-recurrent neural networks.

    Implementation based on "Attention Is All You Need"
    :cite:`DBLP:journals/corr/VaswaniSPUJGKP17`

    Args:
       dropout_prob (float): dropout parameter
       dim (int): embedding size
    """

    def __init__(self, num_embeddings, embedding_dim, dim, dropout_prob=0.0, padding_idx=0, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, dim, 2) * -(math.log(10000.0) / dim)).float())
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(0)
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.embbedding = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
        self.weight = self.embbedding.weight
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.dim = dim

    def forward(self, x, step=None):
        x = self.embbedding(x)
        x = x * math.sqrt(self.dim)
        if step is None:
            x = x + self.pe[:, :x.size(1)]
        else:
            x = x + self.pe[:, step]
        x = self.dropout(x)
        return x


class TokenCrossEntropyLoss(nn.Module):

    def __init__(self, pad_index=0):
        super(TokenCrossEntropyLoss, self).__init__()
        self.pad_index = pad_index
        self.base_loss_function = nn.CrossEntropyLoss(reduction='sum', ignore_index=pad_index)

    def forward(self, outputs, targets):
        batch_size, seq_len, vocabulary_size = outputs.size()
        outputs_flat = outputs.view(batch_size * seq_len, vocabulary_size)
        targets_flat = targets.view(batch_size * seq_len)
        batch_loss = self.base_loss_function(outputs_flat, targets_flat)
        count = (targets != self.pad_index).sum().item()
        return batch_loss, count


class LabelSmoothingLoss(nn.Module):
    """
    With label smoothing,
    KL-divergence between q_{smoothed ground truth prob.}(w)
    and p_{prob. computed by model}(w) is minimized.
    """

    def __init__(self, label_smoothing, vocabulary_size, pad_index=0):
        assert 0.0 < label_smoothing <= 1.0
        super(LabelSmoothingLoss, self).__init__()
        self.pad_index = pad_index
        self.log_softmax = nn.LogSoftmax(dim=-1)
        self.criterion = nn.KLDivLoss(reduction='sum')
        smoothing_value = label_smoothing / (vocabulary_size - 2)
        smoothed_targets = torch.full((vocabulary_size,), smoothing_value)
        smoothed_targets[self.pad_index] = 0
        self.register_buffer('smoothed_targets', smoothed_targets.unsqueeze(0))
        self.confidence = 1.0 - label_smoothing

    def forward(self, outputs, targets):
        """
        outputs (FloatTensor): (batch_size, seq_len, vocabulary_size)
        targets (LongTensor): (batch_size, seq_len)
        """
        batch_size, seq_len, vocabulary_size = outputs.size()
        outputs_log_softmax = self.log_softmax(outputs)
        outputs_flat = outputs_log_softmax.view(batch_size * seq_len, vocabulary_size)
        targets_flat = targets.view(batch_size * seq_len)
        smoothed_targets = self.smoothed_targets.repeat(targets_flat.size(0), 1)
        smoothed_targets.scatter_(1, targets_flat.unsqueeze(1), self.confidence)
        smoothed_targets.masked_fill_((targets_flat == self.pad_index).unsqueeze(1), 0)
        loss = self.criterion(outputs_flat, smoothed_targets)
        count = (targets != self.pad_index).sum().item()
        return loss, count


class AccuracyMetric(nn.Module):

    def __init__(self, pad_index=0):
        super(AccuracyMetric, self).__init__()
        self.pad_index = pad_index

    def forward(self, outputs, targets):
        batch_size, seq_len, vocabulary_size = outputs.size()
        outputs = outputs.view(batch_size * seq_len, vocabulary_size)
        targets = targets.view(batch_size * seq_len)
        predicts = outputs.argmax(dim=1)
        corrects = predicts == targets
        corrects.masked_fill_(targets == self.pad_index, 0)
        correct_count = corrects.sum().item()
        count = (targets != self.pad_index).sum().item()
        return correct_count, count


PAD_TOKEN_INDEX = 0


def pad_masking(x, target_len):
    batch_size, seq_len = x.size()
    padded_positions = x == PAD_TOKEN_INDEX
    pad_mask = padded_positions.unsqueeze(1).expand(batch_size, target_len, seq_len)
    return pad_mask


def subsequent_masking(x):
    batch_size, seq_len = x.size()
    subsequent_mask = np.triu(np.ones(shape=(seq_len, seq_len)), k=1).astype('uint8')
    subsequent_mask = torch.tensor(subsequent_mask)
    subsequent_mask = subsequent_mask.unsqueeze(0).expand(batch_size, seq_len, seq_len)
    return subsequent_mask


class Transformer(nn.Module):

    def __init__(self, encoder, decoder):
        super(Transformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, sources, inputs):
        batch_size, sources_len = sources.size()
        batch_size, inputs_len = inputs.size()
        sources_mask = pad_masking(sources, sources_len)
        memory_mask = pad_masking(sources, inputs_len)
        inputs_mask = subsequent_masking(inputs) | pad_masking(inputs, inputs_len)
        memory = self.encoder(sources, sources_mask)
        outputs, state = self.decoder(inputs, memory, memory_mask, inputs_mask)
        return outputs


class MultiHeadAttention(nn.Module):

    def __init__(self, heads_count, d_model, dropout_prob, mode='self-attention'):
        super(MultiHeadAttention, self).__init__()
        assert d_model % heads_count == 0
        assert mode in ('self-attention', 'memory-attention')
        self.d_head = d_model // heads_count
        self.heads_count = heads_count
        self.mode = mode
        self.query_projection = nn.Linear(d_model, heads_count * self.d_head)
        self.key_projection = nn.Linear(d_model, heads_count * self.d_head)
        self.value_projection = nn.Linear(d_model, heads_count * self.d_head)
        self.final_projection = nn.Linear(d_model, heads_count * self.d_head)
        self.dropout = nn.Dropout(dropout_prob)
        self.softmax = nn.Softmax(dim=3)
        self.attention = None
        self.key_projected = None
        self.value_projected = None

    def forward(self, query, key, value, mask=None, layer_cache=None):
        """

        Args:
            query: (batch_size, query_len, model_dim)
            key: (batch_size, key_len, model_dim)
            value: (batch_size, value_len, model_dim)
            mask: (batch_size, query_len, key_len)
            state: DecoderState
        """
        batch_size, query_len, d_model = query.size()
        d_head = d_model // self.heads_count
        query_projected = self.query_projection(query)
        if layer_cache is None or layer_cache[self.mode] is None:
            key_projected = self.key_projection(key)
            value_projected = self.value_projection(value)
        elif self.mode == 'self-attention':
            key_projected = self.key_projection(key)
            value_projected = self.value_projection(value)
            key_projected = torch.cat([key_projected, layer_cache[self.mode]['key_projected']], dim=1)
            value_projected = torch.cat([value_projected, layer_cache[self.mode]['value_projected']], dim=1)
        elif self.mode == 'memory-attention':
            key_projected = layer_cache[self.mode]['key_projected']
            value_projected = layer_cache[self.mode]['value_projected']
        self.key_projected = key_projected
        self.value_projected = value_projected
        batch_size, key_len, d_model = key_projected.size()
        batch_size, value_len, d_model = value_projected.size()
        query_heads = query_projected.view(batch_size, query_len, self.heads_count, d_head).transpose(1, 2)
        key_heads = key_projected.view(batch_size, key_len, self.heads_count, d_head).transpose(1, 2)
        value_heads = value_projected.view(batch_size, value_len, self.heads_count, d_head).transpose(1, 2)
        attention_weights = self.scaled_dot_product(query_heads, key_heads)
        if mask is not None:
            mask_expanded = mask.unsqueeze(1).expand_as(attention_weights)
            attention_weights = attention_weights.masked_fill(mask_expanded, -1e+18)
        self.attention = self.softmax(attention_weights)
        attention_dropped = self.dropout(self.attention)
        context_heads = torch.matmul(attention_dropped, value_heads)
        context_sequence = context_heads.transpose(1, 2).contiguous()
        context = context_sequence.view(batch_size, query_len, d_model)
        final_output = self.final_projection(context)
        return final_output

    def scaled_dot_product(self, query_heads, key_heads):
        """

        Args:
             query_heads: (batch_size, heads_count, query_len, d_head)
             key_heads: (batch_size, heads_count, key_len, d_head)
        """
        key_heads_transposed = key_heads.transpose(2, 3)
        dot_product = torch.matmul(query_heads, key_heads_transposed)
        attention_weights = dot_product / np.sqrt(self.d_head)
        return attention_weights


class PointwiseFeedForwardNetwork(nn.Module):

    def __init__(self, d_ff, d_model, dropout_prob):
        super(PointwiseFeedForwardNetwork, self).__init__()
        self.feed_forward = nn.Sequential(nn.Linear(d_model, d_ff), nn.Dropout(dropout_prob), nn.ReLU(), nn.Linear(d_ff, d_model), nn.Dropout(dropout_prob))

    def forward(self, x):
        """

        Args:
             x: (batch_size, seq_len, d_model)
        """
        return self.feed_forward(x)


class LayerNormalization(nn.Module):

    def __init__(self, features_count, epsilon=1e-06):
        super(LayerNormalization, self).__init__()
        self.gain = nn.Parameter(torch.ones(features_count))
        self.bias = nn.Parameter(torch.zeros(features_count))
        self.epsilon = epsilon

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.gain * (x - mean) / (std + self.epsilon) + self.bias


class Sublayer(nn.Module):

    def __init__(self, sublayer, d_model):
        super(Sublayer, self).__init__()
        self.sublayer = sublayer
        self.layer_normalization = LayerNormalization(d_model)

    def forward(self, *args):
        x = args[0]
        x = self.sublayer(*args) + x
        return self.layer_normalization(x)


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, heads_count, d_ff, dropout_prob):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attention_layer = Sublayer(MultiHeadAttention(heads_count, d_model, dropout_prob), d_model)
        self.pointwise_feedforward_layer = Sublayer(PointwiseFeedForwardNetwork(d_ff, d_model, dropout_prob), d_model)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, sources, sources_mask):
        sources = self.self_attention_layer(sources, sources, sources, sources_mask)
        sources = self.dropout(sources)
        sources = self.pointwise_feedforward_layer(sources)
        return sources


class TransformerEncoder(nn.Module):

    def __init__(self, layers_count, d_model, heads_count, d_ff, dropout_prob, embedding):
        super(TransformerEncoder, self).__init__()
        self.d_model = d_model
        self.embedding = embedding
        self.encoder_layers = nn.ModuleList([TransformerEncoderLayer(d_model, heads_count, d_ff, dropout_prob) for _ in range(layers_count)])

    def forward(self, sources, mask):
        """

        args:
           sources: embedded_sequence, (batch_size, seq_len, embed_size)
        """
        sources = self.embedding(sources)
        for encoder_layer in self.encoder_layers:
            sources = encoder_layer(sources, mask)
        return sources


class DecoderState:

    def __init__(self):
        self.previous_inputs = torch.tensor([])
        self.layer_caches = defaultdict(lambda : {'self-attention': None, 'memory-attention': None})

    def update_state(self, layer_index, layer_mode, key_projected, value_projected):
        self.layer_caches[layer_index][layer_mode] = {'key_projected': key_projected, 'value_projected': value_projected}

    def beam_update(self, positions):
        for layer_index in self.layer_caches:
            for mode in ('self-attention', 'memory-attention'):
                if self.layer_caches[layer_index][mode] is not None:
                    for projection in self.layer_caches[layer_index][mode]:
                        cache = self.layer_caches[layer_index][mode][projection]
                        if cache is not None:
                            cache.data.copy_(cache.data.index_select(0, positions))


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, heads_count, d_ff, dropout_prob):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attention_layer = Sublayer(MultiHeadAttention(heads_count, d_model, dropout_prob, mode='self-attention'), d_model)
        self.memory_attention_layer = Sublayer(MultiHeadAttention(heads_count, d_model, dropout_prob, mode='memory-attention'), d_model)
        self.pointwise_feedforward_layer = Sublayer(PointwiseFeedForwardNetwork(d_ff, d_model, dropout_prob), d_model)

    def forward(self, inputs, memory, memory_mask, inputs_mask, layer_cache=None):
        inputs = self.self_attention_layer(inputs, inputs, inputs, inputs_mask, layer_cache)
        inputs = self.memory_attention_layer(inputs, memory, memory, memory_mask, layer_cache)
        inputs = self.pointwise_feedforward_layer(inputs)
        return inputs


class TransformerDecoder(nn.Module):

    def __init__(self, layers_count, d_model, heads_count, d_ff, dropout_prob, embedding):
        super(TransformerDecoder, self).__init__()
        self.d_model = d_model
        self.embedding = embedding
        self.decoder_layers = nn.ModuleList([TransformerDecoderLayer(d_model, heads_count, d_ff, dropout_prob) for _ in range(layers_count)])
        self.generator = nn.Linear(embedding.embedding_dim, embedding.num_embeddings)
        self.generator.weight = self.embedding.weight

    def forward(self, inputs, memory, memory_mask, inputs_mask=None, state=None):
        inputs = self.embedding(inputs)
        for layer_index, decoder_layer in enumerate(self.decoder_layers):
            if state is None:
                inputs = decoder_layer(inputs, memory, memory_mask, inputs_mask)
            else:
                layer_cache = state.layer_caches[layer_index]
                inputs = decoder_layer(inputs, memory, memory_mask, inputs_mask, layer_cache)
                state.update_state(layer_index=layer_index, layer_mode='self-attention', key_projected=decoder_layer.self_attention_layer.sublayer.key_projected, value_projected=decoder_layer.self_attention_layer.sublayer.value_projected)
                state.update_state(layer_index=layer_index, layer_mode='memory-attention', key_projected=decoder_layer.memory_attention_layer.sublayer.key_projected, value_projected=decoder_layer.memory_attention_layer.sublayer.value_projected)
        generated = self.generator(inputs)
        return generated, state

    def init_decoder_state(self, **args):
        return DecoderState()


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (AccuracyMetric,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([16])], {})),
    (LayerNormalization,
     lambda: ([], {'features_count': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (MultiHeadAttention,
     lambda: ([], {'heads_count': 4, 'd_model': 4, 'dropout_prob': 0.5}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {})),
    (PointwiseFeedForwardNetwork,
     lambda: ([], {'d_ff': 4, 'd_model': 4, 'dropout_prob': 0.5}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
]

