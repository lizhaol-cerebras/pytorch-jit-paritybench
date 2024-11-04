import sys
_module = sys.modules[__name__]
del sys
allrank = _module
click_models = _module
base = _module
cascade_models = _module
click_utils = _module
duplicate_aware = _module
config = _module
data = _module
dataset_loading = _module
dataset_saving = _module
generate_dummy_data = _module
inference = _module
inference_utils = _module
main = _module
models = _module
losses = _module
approxNDCG = _module
bce = _module
binary_listNet = _module
lambdaLoss = _module
listMLE = _module
listNet = _module
loss_utils = _module
neuralNDCG = _module
ordinal = _module
pointwise = _module
rankNet = _module
metrics = _module
model = _module
model_utils = _module
positional = _module
transformer = _module
rank_and_click = _module
training = _module
early_stop = _module
train_utils = _module
utils = _module
args_utils = _module
command_executor = _module
config_utils = _module
experiments = _module
file_utils = _module
ltr_logging = _module
python_utils = _module
tensorboard_utils = _module
normalize_features = _module
setup = _module
tests = _module
click_models = _module
test_alternative_click_models = _module
test_apply_click_model = _module
test_base_cascade_model = _module
test_diverse_clicks_model = _module
test_duplicate_click_model = _module
test_feature_click_model = _module
test_fixed_click_model = _module
test_masked_click_model = _module
test_random_click_model = _module
test_approxndcg = _module
test_binary_listnet = _module
test_lambdaloss = _module
test_listmle = _module
test_listnet = _module
test_loss_ordinal = _module
test_loss_pointwise = _module
test_mrr = _module
test_ndcg = _module
test_neuralndcg = _module
test_ranknet = _module
utils = _module
test_rank_slates = _module

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


import math


from abc import ABC


from abc import abstractmethod


from typing import List


from typing import Tuple


from typing import Callable


import numpy as np


import torch


from scipy.spatial.distance import cdist


from typing import Union


from sklearn.datasets import load_svmlight_file


from torch.utils.data import DataLoader


from torch.utils.data import Dataset


from torchvision import transforms


from torchvision.transforms import Compose


from typing import Dict


from typing import Generator


from torch.utils.data.dataloader import DataLoader


from functools import partial


from torch import optim


from torch.nn import BCELoss


import torch.nn.functional as F


from itertools import product


from torch.nn import BCEWithLogitsLoss


import torch.nn as nn


from typing import Any


from typing import Optional


import copy


import pandas as pd


from torch.nn.utils import clip_grad_norm_


from scipy.special import softmax


def instantiate_class(full_name: str, **kwargs):
    module_name, class_name = full_name.rsplit('.', 1)
    module = importlib.import_module(module_name)
    class_ = getattr(module, class_name)
    return class_(**kwargs)


class FCModel(nn.Module):
    """
    This class represents a fully connected neural network model with given layer sizes and activation function.
    """

    def __init__(self, sizes, input_norm, activation, dropout, n_features):
        """
        :param sizes: list of layer sizes (excluding the input layer size which is given by n_features parameter)
        :param input_norm: flag indicating whether to perform layer normalization on the input
        :param activation: name of the PyTorch activation function, e.g. Sigmoid or Tanh
        :param dropout: dropout probability
        :param n_features: number of input features
        """
        super(FCModel, self).__init__()
        sizes.insert(0, n_features)
        layers = [nn.Linear(size_in, size_out) for size_in, size_out in zip(sizes[:-1], sizes[1:])]
        self.input_norm = nn.LayerNorm(n_features) if input_norm else nn.Identity()
        self.activation = nn.Identity() if activation is None else instantiate_class('torch.nn.modules.activation', activation)
        self.dropout = nn.Dropout(dropout or 0.0)
        self.output_size = sizes[-1]
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        """
        Forward pass through the FCModel.
        :param x: input of shape [batch_size, slate_length, self.layers[0].in_features]
        :return: output of shape [batch_size, slate_length, self.output_size]
        """
        x = self.input_norm(x)
        for layer in self.layers:
            x = self.dropout(self.activation(layer(x)))
        return x


def first_arg_id(x, *y):
    return x


class LTRModel(nn.Module):
    """
    This class represents a full neural Learning to Rank model with a given encoder model.
    """

    def __init__(self, input_layer, encoder, output_layer):
        """
        :param input_layer: the input block (e.g. FCModel)
        :param encoder: the encoding block (e.g. transformer.Encoder)
        :param output_layer: the output block (e.g. OutputLayer)
        """
        super(LTRModel, self).__init__()
        self.input_layer = input_layer if input_layer else nn.Identity()
        self.encoder = encoder if encoder else first_arg_id
        self.output_layer = output_layer

    def prepare_for_output(self, x, mask, indices):
        """
        Forward pass through the input layer and encoder.
        :param x: input of shape [batch_size, slate_length, input_dim]
        :param mask: padding mask of shape [batch_size, slate_length]
        :param indices: original item ranks used in positional encoding, shape [batch_size, slate_length]
        :return: encoder output of shape [batch_size, slate_length, encoder_output_dim]
        """
        return self.encoder(self.input_layer(x), mask, indices)

    def forward(self, x, mask, indices):
        """
        Forward pass through the whole LTRModel.
        :param x: input of shape [batch_size, slate_length, input_dim]
        :param mask: padding mask of shape [batch_size, slate_length]
        :param indices: original item ranks used in positional encoding, shape [batch_size, slate_length]
        :return: model output of shape [batch_size, slate_length, output_dim]
        """
        return self.output_layer(self.prepare_for_output(x, mask, indices))

    def score(self, x, mask, indices):
        """
        Forward pass through the whole LTRModel and item scoring.

        Used when evaluating listwise metrics in the training loop.
        :param x: input of shape [batch_size, slate_length, input_dim]
        :param mask: padding mask of shape [batch_size, slate_length]
        :param indices: original item ranks used in positional encoding, shape [batch_size, slate_length]
        :return: scores of shape [batch_size, slate_length]
        """
        return self.output_layer.score(self.prepare_for_output(x, mask, indices))


class OutputLayer(nn.Module):
    """
    This class represents an output block reducing the output dimensionality to d_output.
    """

    def __init__(self, d_model, d_output, output_activation=None):
        """
        :param d_model: dimensionality of the output layer input
        :param d_output: dimensionality of the output layer output
        :param output_activation: name of the PyTorch activation function used before scoring, e.g. Sigmoid or Tanh
        """
        super(OutputLayer, self).__init__()
        self.activation = nn.Identity() if output_activation is None else instantiate_class('torch.nn.modules.activation', output_activation)
        self.d_output = d_output
        self.w_1 = nn.Linear(d_model, d_output)

    def forward(self, x):
        """
        Forward pass through the OutputLayer.
        :param x: input of shape [batch_size, slate_length, self.d_model]
        :return: output of shape [batch_size, slate_length, self.d_output]
        """
        return self.activation(self.w_1(x).squeeze(dim=2))

    def score(self, x):
        """
        Forward pass through the OutputLayer and item scoring by summing the individual outputs if d_output > 1.
        :param x: input of shape [batch_size, slate_length, self.d_model]
        :return: output of shape [batch_size, slate_length]
        """
        if self.d_output > 1:
            return self.forward(x).sum(-1)
        else:
            return self.forward(x)


class CustomDataParallel(nn.DataParallel):
    """
    Wrapper for scoring with nn.DataParallel object containing LTRModel.
    """

    def score(self, x, mask, indices):
        """
        Wrapper function for a forward pass through the whole LTRModel and item scoring.
        :param x: input of shape [batch_size, slate_length, input_dim]
        :param mask: padding mask of shape [batch_size, slate_length]
        :param indices: original item ranks used in positional encoding, shape [batch_size, slate_length]
        :return: scores of shape [batch_size, slate_length]
        """
        return self.module.score(x, mask, indices)


class FixedPositionalEncoding(nn.Module):
    """
    Class implementing fixed positional encodings.

    Fixed positional encodings up to max_len position are computed once during object construction.
    """

    def __init__(self, d_model: int, max_len=5000):
        """
        :param d_model: dimensionality of the embeddings
        :param max_len: maximum length of the sequence
        """
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0.0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0.0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = torch.cat((pe, torch.zeros([1, d_model])))
        self.padding_idx = pe.size()[0] - 1
        self.register_buffer('pe', pe)

    def forward(self, x, mask, indices):
        """
        Forward pass through the FixedPositionalEncoding.
        :param x: input of shape [batch_size, slate_length, d_model]
        :param mask: padding mask of shape [batch_size, slate_length]
        :param indices: original item ranks used in positional encoding, shape [batch_size, slate_length]
        :return: output of shape [batch_size, slate_length, d_model]
        """
        padded_indices = indices.masked_fill(mask, self.padding_idx)
        padded_indices[padded_indices > self.padding_idx] = self.padding_idx
        x = math.sqrt(self.pe.shape[1]) * x + self.pe[padded_indices, :]
        return x


class LearnedPositionalEncoding(nn.Module):
    """
    Class implementing learnable positional encodings.
    """

    def __init__(self, d_model, max_len=5000):
        """
        :param d_model: dimensionality of the embeddings
        :param max_len: maximum length of the sequence
        """
        super().__init__()
        self.pe = nn.Embedding(max_len + 1, d_model, padding_idx=-1)

    def forward(self, x, mask, indices):
        """
        Forward pass through the LearnedPositionalEncoding.
        :param x: input of shape [batch_size, slate_length, d_model]
        :param mask: padding mask of shape [batch_size, slate_length]
        :param indices: original item ranks used in positional encoding, shape [batch_size, slate_length]
        :return: output of shape [batch_size, slate_length, d_model]
        """
        padded_indices = indices.masked_fill(mask, self.pe.padding_idx)
        padded_indices[padded_indices > self.pe.padding_idx] = self.pe.padding_idx
        x = math.sqrt(self.pe.embedding_dim) * x + self.pe(padded_indices)
        return x


class LayerNorm(nn.Module):
    """
    Layer normalization module.
    """

    def __init__(self, features, eps=1e-06):
        """
        :param features: shape of normalized features
        :param eps: epsilon used for standard deviation
        """
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        """
        Forward pass through the layer normalization.
        :param x: input of shape [batch_size, slate_length, input_dim]
        :return: normalized input of shape [batch_size, slate_length, output_dim]
        """
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


def clones(module, N):
    """
    Creation of N identical layers.
    :param module: module to clone
    :param N: number of copies
    :return: nn.ModuleList of module copies
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Encoder(nn.Module):
    """
    Stack of Transformer encoder blocks with positional encoding.
    """

    def __init__(self, layer, N, position):
        """
        :param layer: single building block to clone
        :param N: number of copies
        :param position: positional encoding module
        """
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        self.position = position

    def forward(self, x, mask, indices):
        """
        Forward pass through each block of the Transformer.
        :param x: input of shape [batch_size, slate_length, input_dim]
        :param mask: padding mask of shape [batch_size, slate_length]
        :param indices: original item ranks used in positional encoding, shape [batch_size, slate_length]
        :return: output of shape [batch_size, slate_length, output_dim]
        """
        if self.position:
            x = self.position(x, mask, indices)
        mask = mask.unsqueeze(-2)
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class SublayerConnection(nn.Module):
    """
    Residual connection followed by layer normalization.
    Please not that for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        """
        :param size: number of input/output features
        :param dropout: dropout probability
        """
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        """
        Forward pass through the sublayer connection module, applying the residual connection to any sublayer with the same size.
        :param x: input of shape [batch_size, slate_length, input_dim]
        :param sublayer: layer through which to pass the input prior to applying the sum
        :return: output of shape [batch_size, slate_length, output_dim]
        """
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    """
    Single Transformer encoder block made of self-attention and feed-forward layers with residual connections.
    """

    def __init__(self, size, self_attn, feed_forward, dropout):
        """
        :param size: input/output size of the encoder block
        :param self_attn: self-attention layer
        :param feed_forward: feed-forward layer
        :param dropout: dropout probability
        """
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        """
        Forward pass through the encoder block.
        :param x: input of shape [batch_size, slate_length, self.size]
        :param mask: padding mask of shape [batch_size, slate_length]
        :return: output of shape [batch_size, slate_length, self.size]
        """
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


def attention(query, key, value, mask=None, dropout=None):
    """
    Basic function for "Scaled Dot Product Attention" computation.
    :param query: query set of shape [batch_size, slate_size, n_attention_heads, attention_dim]
    :param key: key set of shape [batch_size, slate_size, n_attention_heads, attention_dim]
    :param value: value set of shape [batch_size, slate_size, n_attention_heads, attention_dim]
    :param mask: padding mask of shape [batch_size, slate_length]
    :param dropout: dropout probability
    :return: attention scores of shape [batch_size, slate_size, n_attention_heads, attention_dim]
    """
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 1, float('-inf'))
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    """
    Multi-headed attention block.
    """

    def __init__(self, h, d_model, dropout=0.1):
        """
        :param h: number of attention heads
        :param d_model: input/output dimensionality
        :param dropout: dropout probability
        """
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        """
        Forward pass through the multi-head attention block.
        :param query: query set of shape [batch_size, slate_size, self.d_model]
        :param key: key set of shape [batch_size, slate_size, self.d_model]
        :param value: value set of shape [batch_size, slate_size, self.d_model]
        :param mask: padding mask of shape [batch_size, slate_length]
        :return: output of shape [batch_size, slate_size, self.d_model]
        """
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        query, key, value = [linear(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2) for linear, x in zip(self.linears, (query, key, value))]
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    """
    Feed-forward block.
    """

    def __init__(self, d_model, d_ff, dropout=0.1):
        """
        :param d_model: input/output dimensionality
        :param d_ff: hidden dimensionality
        :param dropout: dropout probability
        """
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Forward pass through the feed-forward block.
        :param x: input of shape [batch_size, slate_size, self.d_model]
        :return: output of shape [batch_size, slate_size, self.d_model]
        """
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


import torch
from torch.nn import MSELoss, ReLU
from paritybench._paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (LayerNorm,
     lambda: ([], {'features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (MultiHeadedAttention,
     lambda: ([], {'h': 4, 'd_model': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (OutputLayer,
     lambda: ([], {'d_model': 4, 'd_output': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (PositionwiseFeedForward,
     lambda: ([], {'d_model': 4, 'd_ff': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SublayerConnection,
     lambda: ([], {'size': 4, 'dropout': 0.5}),
     lambda: ([torch.rand([4, 4, 4, 4]), _mock_layer()], {}),
     False),
]

class Test_allegro_allRank(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

    def test_003(self):
        self._check(*TESTCASES[3])

    def test_004(self):
        self._check(*TESTCASES[4])

