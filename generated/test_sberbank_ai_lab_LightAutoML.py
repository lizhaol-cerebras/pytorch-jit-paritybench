
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


from collections import Counter


from collections import defaultdict


from random import shuffle


from typing import Any


from typing import Callable


from typing import Dict


from typing import List


from typing import Optional


from typing import Tuple


import numpy as np


import pandas as pd


import torch


from torch import nn


from torch.nn.utils.rnn import pad_sequence


from torch.utils.data import DataLoader


from torch.utils.data import Dataset


from torch.utils.data import Sampler


import logging


from numbers import Number


from typing import Type


from typing import Union


import torch.nn as nn


from torch.optim import Adam


from torch.optim.lr_scheduler import ReduceLROnPlateau


import torch.nn.functional as F


from torch.distributions.utils import clamp_probs


import itertools


import matplotlib.pyplot as plt


from matplotlib.colors import Colormap


from typing import Sequence


from typing import Iterable


from pandas import DataFrame


from copy import copy


from copy import deepcopy


from typing import cast


from sklearn.base import TransformerMixin


import uuid


from torch.optim import lr_scheduler


from sklearn.linear_model import ElasticNet


from sklearn.linear_model import Lasso


from sklearn.linear_model import LogisticRegression


from scipy import sparse


from torch import optim


import inspect


from functools import partial


from typing import TYPE_CHECKING


from itertools import chain


from torch._utils import ExceptionWrapper


from torch.cuda._utils import _get_device_index


import random


from sklearn.utils.murmurhash import murmurhash3_32


from sklearn.feature_extraction.text import TfidfVectorizer


from sklearn.linear_model import SGDClassifier


from sklearn.linear_model import SGDRegressor


def create_emb_layer(weights_matrix=None, voc_size=None, embed_dim=None, trainable_embeds=True) ->torch.nn.Embedding:
    """Create initialized embedding layer.

    Args:
        weights_matrix: Weights of embedding layer.
        voc_size: Size of vocabulary.
        embed_dim: Size of embeddings.
        trainable_embeds: To optimize layer when training model.

    Returns:
        Initialized embedding layer.

    """
    assert weights_matrix is not None or voc_size is not None and embed_dim is not None, 'Please define anything: weights_matrix or voc_size & embed_dim'
    if weights_matrix is not None:
        voc_size, embed_dim = weights_matrix.size()
    emb_layer = nn.Embedding(voc_size, embed_dim)
    if weights_matrix is not None:
        emb_layer.load_state_dict({'weight': weights_matrix})
    if not trainable_embeds:
        emb_layer.weight.requires_grad = False
    return emb_layer


class TIModel(nn.Module):

    def __init__(self, voc_size: 'int', embed_dim: 'int'=50, conv_filters: 'int'=100, conv_ksize: 'int'=3, drop_rate: 'float'=0.2, hidden_dim: 'int'=100, weights_matrix: 'Optional[torch.FloatTensor]'=None, trainable_embeds: 'bool'=False):
        super(TIModel, self).__init__()
        self.lookup = create_emb_layer(weights_matrix, voc_size, embed_dim, trainable_embeds)
        embed_dim = self.lookup.embedding_dim
        self.drop1 = nn.Dropout(p=drop_rate)
        self.conv1 = nn.Conv1d(embed_dim, conv_filters, conv_ksize, padding=1)
        self.act1 = nn.ReLU()
        self.pool1 = nn.AdaptiveAvgPool1d(output_size=1)
        self.global_info = nn.Linear(conv_filters, hidden_dim)
        self.global_act = nn.ReLU()
        self.conv2 = nn.Conv1d(conv_filters, hidden_dim, conv_ksize, padding=1)
        self.act2 = nn.ReLU()
        self.local_info = nn.Conv1d(hidden_dim, hidden_dim, conv_ksize, padding=1)
        self.local_act = nn.ReLU()
        self.drop3 = nn.Dropout(p=drop_rate)
        self.conv3 = nn.Conv1d(2 * hidden_dim, conv_filters, 1)
        self.act3 = nn.ReLU()
        self.conv4 = nn.Conv1d(conv_filters, 1, 1)

    def get_embedding(self, input):
        return self.lookup(input)

    def get_embedding_layer(self):
        return self.lookup

    def freeze_embedding(self):
        embeds = self.get_embedding_layer()
        embeds.weight.requires_grad_(False)

    def unfreeze_embedding(self):
        embeds = self.get_embedding_layer()
        embeds.weight.requires_grad_(True)

    def predict(self, x):
        x = x.transpose(1, 2)
        x = self.act1(self.conv1(self.drop1(x)))
        global_info = self.global_act(self.global_info(self.pool1(x).squeeze(2)))
        local_info = self.local_act(self.local_info(self.act2(self.conv2(x))))
        global_info = global_info.unsqueeze(-1).expand_as(local_info)
        z = torch.cat([global_info, local_info], dim=1)
        z = self.act3(self.conv3(self.drop3(z)))
        logits = self.conv4(z)
        return logits

    def forward(self, x):
        embed = self.get_embedding(x)
        logits = self.predict(embed)
        return logits


class GumbelTopKSampler(nn.Module):

    def __init__(self, T, k):
        super(GumbelTopKSampler, self).__init__()
        self.T = nn.Parameter(torch.tensor(T, dtype=torch.float32), requires_grad=False)
        self.k = nn.Parameter(torch.tensor(k, dtype=torch.int32), requires_grad=False)

    def sample_continous(self, logits):
        l_shape = logits.shape[0], self.k, logits.shape[2]
        u = clamp_probs(torch.rand(l_shape, device=logits.device))
        gumbel = -torch.log(-torch.log(u))
        noisy_logits = (gumbel + logits) / self.T
        samples = F.softmax(noisy_logits, dim=-1)
        samples = torch.max(samples, dim=1)[0]
        return samples

    def sample_discrete(self, logits):
        threshold = torch.topk(logits, self.k, sorted=True)[0][..., -1]
        samples = torch.ge(logits.squeeze(1), threshold).float()
        return samples

    def forward(self, logits):
        csamples = None
        if self.training:
            csamples = self.sample_continous(logits)
        dsamples = self.sample_discrete(logits)
        return dsamples, csamples


class SoftSubSampler(nn.Module):

    def __init__(self, T, k):
        super(SoftSubSampler, self).__init__()
        self.T = nn.Parameter(torch.tensor(T, dtype=torch.float32), requires_grad=False)
        self.k = nn.Parameter(torch.tensor(k, dtype=torch.int32), requires_grad=False)

    def inject_noise(self, logits):
        u = clamp_probs(torch.rand_like(logits))
        z = -torch.log(-torch.log(u))
        noisy_logits = logits + z
        return noisy_logits

    def continuous_topk(self, w, separate=False):
        khot_list = []
        onehot_approx = torch.zeros_like(w, dtype=torch.float32)
        for _ in range(self.k):
            khot_mask = clamp_probs(1.0 - onehot_approx)
            w += torch.log(khot_mask)
            onehot_approx = F.softmax(w / self.T, dim=-1)
            khot_list.append(onehot_approx)
        if separate:
            return khot_list
        else:
            return torch.stack(khot_list, dim=-1).sum(-1).squeeze(1)

    def sample_continous(self, logits):
        return self.continuous_topk(self.inject_noise(logits))

    def sample_discrete(self, logits):
        threshold = torch.topk(logits, self.k, sorted=True)[0][..., -1]
        samples = torch.ge(logits.squeeze(1), threshold).float()
        return samples

    def forward(self, logits):
        csamples = None
        if self.training:
            csamples = self.sample_continous(logits)
        dsamples = self.sample_discrete(logits)
        return dsamples, csamples


class DistilPredictor(nn.Module):

    def __init__(self, task_name: 'str', n_outs: 'int', voc_size: 'int', embed_dim: 'int'=300, hidden_dim: 'int'=100, weights_matrix: 'Optional[torch.FloatTensor]'=None, trainable_embeds: 'bool'=False):
        super(DistilPredictor, self).__init__()
        self.lookup = create_emb_layer(weights_matrix, voc_size, embed_dim, trainable_embeds)
        embed_dim = self.lookup.embedding_dim
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.act = nn.ReLU()
        if task_name == 'reg':
            self.head = nn.Linear(hidden_dim, n_outs)
        elif task_name == 'binary':
            self.head = nn.Sequential(nn.Linear(hidden_dim, n_outs), nn.Sigmoid())
        elif task_name == 'multiclass':
            self.head = nn.Sequential(nn.Linear(hidden_dim, n_outs), nn.Softmax(dim=-1))

    def get_embedding(self, input):
        return self.lookup(input)

    def get_embedding_layer(self):
        return self.lookup

    def freeze_embedding(self):
        embeds = self.get_embedding_layer()
        embeds.weight.requires_grad_(False)

    def unfreeze_embedding(self):
        embeds = self.get_embedding_layer()
        embeds.weight.requires_grad_(True)

    def predict(self, embed, T):
        out = torch.mean(embed * T.unsqueeze(2), axis=1)
        out = self.act(self.fc1(out))
        out = self.head(out)
        return out

    def forward(self, x, T):
        embed = self.get_embedding(x)
        out = self.predict(embed, T)
        return out


class L2XModel(nn.Module):

    def __init__(self, task_name: 'str', n_outs: 'int', voc_size: 'int'=1000, embed_dim: 'int'=100, conv_filters: 'int'=100, conv_ksize: 'int'=3, drop_rate: 'float'=0.2, hidden_dim: 'int'=100, T: 'float'=0.3, k: 'int'=5, weights_matrix: 'Optional[torch.FloatTensor]'=None, trainable_embeds: 'bool'=False, sampler: 'str'='gumbeltopk', anneal_factor: 'float'=1.0):
        super(L2XModel, self).__init__()
        self.ti_model = TIModel(voc_size, embed_dim, conv_filters, conv_ksize, drop_rate, hidden_dim, weights_matrix, trainable_embeds)
        self.T = T
        self.anneal_factor = anneal_factor
        if sampler == 'gumbeltopk':
            self.sampler = GumbelTopKSampler(T, k)
        else:
            self.sampler = SoftSubSampler(T, k)
        self.distil_model = DistilPredictor(task_name, n_outs, voc_size, embed_dim, hidden_dim, weights_matrix, trainable_embeds)

    def forward(self, x):
        """Forward pass."""
        logits = self.ti_model(x)
        dsamples, csamples = self.sampler(logits)
        if self.training:
            T = csamples
        else:
            T = dsamples
        out = self.distil_model(x, T)
        return out, T

    def anneal(self):
        """Temperature annealing."""
        self.sampler.T *= self.anneal_factor


class EffNetImageEmbedder(nn.Module):
    """Class to compute EfficientNet embeddings."""

    def __init__(self, model_name: 'str'='efficientnet-b0', weights_path: 'Optional[str]'=None, is_advprop: 'bool'=True, device=torch.device('cuda:0')):
        """Pytorch module for image embeddings based on efficient-net model.

        Args:
            model_name: Name of effnet model.
            weights_path: Path to saved weights.
            is_advprop: Use adversarial training.
            device: Device to use.

        """
        super(EffNetImageEmbedder, self).__init__()
        self.device = device
        self.model = EfficientNet.from_pretrained(model_name, weights_path=weights_path, advprop=is_advprop, include_top=False).eval()
        self.feature_shape = self.get_shape()
        self.is_advprop = is_advprop
        self.model_name = model_name

    @torch.no_grad()
    def get_shape(self) ->int:
        """Calculate output embedding shape.

        Returns:
            Shape of embedding.

        """
        return self.model(torch.randn(1, 3, 224, 224)).squeeze().shape[0]

    def forward(self, x) ->torch.Tensor:
        """Forward pass."""
        out = self.model(x)
        return out[:, :, 0, 0]


class CatLinear(nn.Module):
    """Simple linear model to handle numeric and categorical features.

    Args:
        numeric_size: Number of numeric features.
        embed_sizes: Embedding sizes.
        output_size: Size of output layer.

    """

    def __init__(self, numeric_size: 'int'=0, embed_sizes: 'Sequence[int]'=(), output_size: 'int'=1):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(output_size))
        self.linear = None
        if numeric_size > 0:
            self.linear = nn.Linear(in_features=numeric_size, out_features=output_size, bias=False)
            nn.init.zeros_(self.linear.weight)
        self.cat_params = None
        if len(embed_sizes) > 0:
            self.cat_params = nn.Parameter(torch.zeros(sum(embed_sizes), output_size))
            self.embed_idx = torch.LongTensor(embed_sizes).cumsum(dim=0) - torch.LongTensor(embed_sizes)

    def forward(self, numbers: 'Optional[torch.Tensor]'=None, categories: 'Optional[torch.Tensor]'=None):
        """Forward-pass.

        Args:
            numbers: Input numeric features.
            categories: Input categorical features.

        Returns:
            Linear prediction.

        """
        x = self.bias
        if self.linear is not None:
            x = x + self.linear(numbers)
        if self.cat_params is not None:
            x = x + self.cat_params[categories + self.embed_idx].sum(dim=1)
        return x


class CatLogisticRegression(CatLinear):
    """Realisation of torch-based logistic regression."""

    def __init__(self, numeric_size: 'int', embed_sizes: 'Sequence[int]'=(), output_size: 'int'=1):
        super().__init__(numeric_size, embed_sizes=embed_sizes, output_size=output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, numbers: 'Optional[torch.Tensor]'=None, categories: 'Optional[torch.Tensor]'=None):
        """Forward-pass. Sigmoid func at the end of linear layer.

        Args:
            numbers: Input numeric features.
            categories: Input categorical features.

        Returns:
            Probabilitics.

        """
        x = super().forward(numbers, categories)
        x = torch.clamp(x, -50, 50)
        x = self.sigmoid(x)
        return x


class CatRegression(CatLinear):
    """Realisation of torch-based linear regreession."""

    def __init__(self, numeric_size: 'int', embed_sizes: 'Sequence[int]'=(), output_size: 'int'=1):
        super().__init__(numeric_size, embed_sizes=embed_sizes, output_size=output_size)


class CatMulticlass(CatLinear):
    """Realisation of multi-class linear classifier."""

    def __init__(self, numeric_size: 'int', embed_sizes: 'Sequence[int]'=(), output_size: 'int'=1):
        super().__init__(numeric_size, embed_sizes=embed_sizes, output_size=output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, numbers: 'Optional[torch.Tensor]'=None, categories: 'Optional[torch.Tensor]'=None):
        """Forward-pass.

        Args:
            numbers: Input numeric features.
            categories: Input categorical features.

        Returns:
            Linear prediction.

        """
        x = super().forward(numbers, categories)
        x = torch.clamp(x, -50, 50)
        x = self.softmax(x)
        return x


class TorchLossWrapper(nn.Module):
    """Customize PyTorch-based loss.

    Args:
        func: loss to customize. Example: `torch.nn.MSELoss`.
        **kwargs: additional parameters.

    Returns:
        callable loss, uses format (y_true, y_pred, sample_weight).

    """

    def __init__(self, func: 'Callable', flatten=False, log=False, **kwargs: Any):
        super(TorchLossWrapper, self).__init__()
        self.base_loss = func(reduction='none', **kwargs)
        self.flatten = flatten
        self.log = log

    def forward(self, y_true: 'torch.Tensor', y_pred: 'torch.Tensor', sample_weight: 'Optional[torch.Tensor]'=None):
        """Forward-pass."""
        if self.flatten:
            y_true = y_true[:, 0].type(torch.int64)
        if self.log:
            y_pred = torch.log(y_pred)
        outp = self.base_loss(y_pred, y_true)
        if len(outp.shape) == 2:
            outp = outp.sum(dim=1)
        if sample_weight is not None:
            outp = outp * sample_weight
            return outp.mean() / sample_weight.mean()
        return outp.mean()


class SequenceAbstractPooler(nn.Module):
    """Abstract pooling class."""

    def __init__(self):
        super(SequenceAbstractPooler, self).__init__()

    def forward(self, x: 'torch.Tensor', x_mask: 'torch.Tensor') ->torch.Tensor:
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class SequenceAvgPooler(SequenceAbstractPooler):
    """Mean value pooling."""

    def __init__(self):
        super(SequenceAvgPooler, self).__init__()

    def forward(self, x: 'torch.Tensor', x_mask: 'torch.Tensor') ->torch.Tensor:
        x = x.masked_fill(~x_mask, 0)
        x_active = torch.sum(x_mask, dim=-2)
        x_active = x_active.masked_fill(x_active == 0, 1)
        values = torch.sum(x, dim=-2) / x_active.data
        return values


class SequenceClsPooler(SequenceAbstractPooler):
    """CLS token pooling."""

    def __init__(self):
        super(SequenceClsPooler, self).__init__()

    def forward(self, x: 'torch.Tensor', x_mask: 'torch.Tensor') ->torch.Tensor:
        return x[..., 0, :]


class SequenceIndentityPooler(SequenceAbstractPooler):
    """Identity pooling."""

    def __init__(self):
        super(SequenceIndentityPooler, self).__init__()

    def forward(self, x: 'torch.Tensor', x_mask: 'torch.Tensor') ->torch.Tensor:
        return x


class SequenceMaxPooler(SequenceAbstractPooler):
    """Max value pooling."""

    def __init__(self):
        super(SequenceMaxPooler, self).__init__()

    def forward(self, x: 'torch.Tensor', x_mask: 'torch.Tensor') ->torch.Tensor:
        x = x.masked_fill(~x_mask, -float('inf'))
        values, _ = torch.max(x, dim=-2)
        return values


class SequenceSumPooler(SequenceAbstractPooler):
    """Sum value pooling."""

    def __init__(self):
        super(SequenceSumPooler, self).__init__()

    def forward(self, x: 'torch.Tensor', x_mask: 'torch.Tensor') ->torch.Tensor:
        x = x.masked_fill(~x_mask, 0)
        values = torch.sum(x, dim=-2)
        return values


pooling_by_name = {'mean': SequenceAvgPooler, 'sum': SequenceSumPooler, 'max': SequenceMaxPooler, 'cls': SequenceClsPooler, 'none': SequenceIndentityPooler}


def position_encoding_init(n_pos: 'int', embed_size: 'int') ->torch.Tensor:
    """Compute positional embedding matrix.

    Args:
        n_pos: Len of sequence.
        embed_size: Size of output sentence embedding.

    Returns:
        Torch tensor with all positional embeddings.

    """
    position_enc = np.array([([(pos / np.power(10000, 2 * (j // 2) / embed_size)) for j in range(embed_size)] if pos != 0 else np.zeros(embed_size)) for pos in range(n_pos)])
    position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2])
    position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2])
    return torch.from_numpy(position_enc).float()


def seed_everything(seed: 'int'=42, deterministic: 'bool'=True):
    """Set random seed and cudnn params.

    Args:
        seed: Random state.
        deterministic: cudnn backend.

    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.manual_seed(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True


class BOREP(nn.Module):
    """Class to compute Bag of Random Embedding Projections sentence embeddings from words embeddings.

    Bag of Random Embedding Projections sentence embeddings.

    Args:
        embed_size: Size of word embeddings.
        proj_size: Size of output sentence embedding.
        pooling: Pooling type.
        max_length: Maximum length of sentence.
        init: Type of weight initialization.
        pos_encoding: Add positional embedding.
        **kwargs: Ignored params.

    Note:
        There are several pooling types:

            - `'max'`: Maximum on seq_len dimension for non masked inputs.
            - `'mean'`: Mean on seq_len dimension for non masked inputs.
            - `'sum'`: Sum on seq_len dimension for non masked inputs.

        For init parameter there are several options:

            - `'orthogonal'`: Orthogonal init.
            - `'normal'`: Normal with std 0.1.
            - `'uniform'`: Uniform from -0.1 to 0.1.
            - `'kaiming'`: Uniform kaiming init.
            - `'xavier'`: Uniform xavier init.

    """
    name = 'BOREP'
    _poolers = {'max', 'mean', 'sum'}

    def __init__(self, embed_size: 'int'=300, proj_size: 'int'=300, pooling: 'str'='mean', max_length: 'int'=200, init: 'str'='orthogonal', pos_encoding: 'bool'=False, **kwargs: Any):
        super(BOREP, self).__init__()
        self.embed_size = embed_size
        self.proj_size = proj_size
        self.pos_encoding = pos_encoding
        seed_everything(42)
        if self.pos_encoding:
            self.pos_code = position_encoding_init(max_length, self.embed_size).view(1, max_length, self.embed_size)
        self.pooling = pooling_by_name[pooling]()
        self.proj = nn.Linear(self.embed_size, self.proj_size, bias=False)
        if init == 'orthogonal':
            nn.init.orthogonal_(self.proj.weight)
        elif init == 'normal':
            nn.init.normal_(self.proj.weight, std=0.1)
        elif init == 'uniform':
            nn.init.uniform_(self.proj.weight, a=-0.1, b=0.1)
        elif init == 'kaiming':
            nn.init.kaiming_uniform_(self.proj.weight)
        elif init == 'xavier':
            nn.init.xavier_uniform_(self.proj.weight)

    def get_out_shape(self) ->int:
        """Output shape.

        Returns:
            Int with module output shape.

        """
        return self.proj_size

    def get_name(self) ->str:
        """Module name.

        Returns:
            String with module name.

        """
        return self.name

    @torch.no_grad()
    def forward(self, inp: 'Dict[str, torch.Tensor]') ->torch.Tensor:
        """Forward-pass."""
        x = inp['text']
        batch_size, batch_max_length = x.shape[0], x.shape[1]
        if self.pos_encoding:
            x = x + self.pos_code[:, :batch_max_length, :]
        x = x.contiguous().view(batch_size * batch_max_length, -1)
        x = self.proj(x)
        out = x.contiguous().view(batch_size, batch_max_length, -1)
        x_length = (torch.arange(out.shape[1])[None, :] < inp['length'][:, None])[:, :, None]
        out = self.pooling(out, x_length)
        return out


class RandomLSTM(nn.Module):
    """Class to compute Random LSTM sentence embeddings from words embeddings.

    Args:
        embed_size: Size of word embeddings.
        hidden_size: Size of hidden dimensions of LSTM.
        pooling: Pooling type.
        num_layers: Number of lstm layers.
        **kwargs: Ignored params.

    Note:
        There are several pooling types:

            - `'max'`: Maximum on seq_len dimension for non masked inputs.
            - `'mean'`: Mean on seq_len dimension for non masked inputs.
            - `'sum'`: Sum on seq_len dimension for non masked inputs.

    """
    name = 'RandomLSTM'
    _poolers = 'max', 'mean', 'sum'

    def __init__(self, embed_size: 'int'=300, hidden_size: 'int'=256, pooling: 'str'='mean', num_layers: 'int'=1, **kwargs: Any):
        super(RandomLSTM, self).__init__()
        if pooling not in self._poolers:
            raise ValueError('pooling - {} - not in the list of available types {}'.format(pooling, self._poolers))
        seed_everything(42)
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers=num_layers, bidirectional=True, batch_first=True)
        self.pooling = pooling_by_name[pooling]()

    def get_out_shape(self) ->int:
        """Output shape.

        Returns:
            Int with module output shape.

        """
        return self.hidden_size * 2

    def get_name(self) ->str:
        """Module name.

        Returns:
            String with module name.

        """
        return self.name

    @torch.no_grad()
    def forward(self, inp: 'Dict[str, torch.Tensor]') ->torch.Tensor:
        """Forward-pass."""
        out, _ = self.lstm(inp['text'])
        x_length = (torch.arange(out.shape[1])[None, :] < inp['length'][:, None])[:, :, None]
        out = self.pooling(out, x_length)
        return out


def single_text_hash(x: 'str') ->str:
    """Get text hash.

    Args:
        x: Text.

    Returns:
        String text hash.

    """
    numhash = murmurhash3_32(x, seed=13)
    texthash = str(numhash) if numhash > 0 else 'm' + str(abs(numhash))
    return texthash


class BertEmbedder(nn.Module):
    """Class to compute `HuggingFace <https://huggingface.co>`_ transformers words or sentence embeddings.

    Bert sentence or word embeddings.

    Args:
        model_name: Name of transformers model.
        pooling: Pooling type.
        **kwargs: Ignored params.

    Note:
        There are several pooling types:

            - `'cls'`: Use CLS token for sentence embedding
                from last hidden state.
            - `'max'`: Maximum on seq_len dimension
                for non masked inputs from last hidden state.
            - `'mean'`: Mean on seq_len dimension for non masked
                inputs from last hidden state.
            - `'sum'`: Sum on seq_len dimension for non masked inputs
                from last hidden state.
            - `'none'`: Don't use pooling (for RandomLSTM pooling strategy).

    """
    name = 'BertEmb'
    _poolers = {'cls', 'max', 'mean', 'sum', 'none'}

    def __init__(self, model_name: 'str', pooling: 'str'='none', **kwargs: Any):
        super(BertEmbedder, self).__init__()
        if pooling not in self._poolers:
            raise ValueError('pooling - {} - not in the list of available types {}'.format(pooling, self._poolers))
        self.pooling = pooling_by_name[pooling]()
        self.model_name = model_name
        self.transformer = AutoModel.from_pretrained(model_name)

    def forward(self, inp: 'Dict[str, torch.Tensor]') ->torch.Tensor:
        """Forward-pass."""
        encoded_layers, _ = self.transformer(input_ids=inp['input_ids'], attention_mask=inp['attention_mask'], token_type_ids=inp.get('token_type_ids'), return_dict=False)
        encoded_layers = self.pooling(encoded_layers, inp['attention_mask'].unsqueeze(-1).bool())
        return encoded_layers

    def freeze(self):
        """Freeze module parameters."""
        for param in self.transformer.parameters():
            param.requires_grad = False

    def get_name(self) ->str:
        """Module name.

        Returns:
            String with module name.

        """
        return self.name + single_text_hash(self.model_name)

    def get_out_shape(self) ->int:
        """Output shape.

        Returns:
            Int with module output shape.

        """
        return self.transformer.config.hidden_size


class CustomDataParallel(nn.DataParallel):
    """Extension for nn.DataParallel for supporting predict method of DL model."""

    def __init__(self, module: 'nn.Module', device_ids: 'Optional[List[int]]'=None, output_device: 'Optional[torch.device]'=None, dim: 'Optional[int]'=0):
        super(CustomDataParallel, self).__init__(module, device_ids, output_device, dim)
        try:
            self.n_out = module.n_out
        except:
            pass

    def predict(self, *inputs, **kwargs):
        """Predict."""
        if not self.device_ids:
            return self.module(*inputs, **kwargs)
        for t in chain(self.module.parameters(), self.module.buffers()):
            if t.device != self.src_device_obj:
                raise RuntimeError('module must have its parameters and buffers on device {} (device_ids[0]) but found one of them on device: {}'.format(self.src_device_obj, t.device))
        inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids)
        if len(self.device_ids) == 1:
            return self.module.predict(*inputs[0], **kwargs[0])
        replicas = self.replicate(self.module, self.device_ids[:len(inputs)])
        outputs = self.parallel_apply_predict(replicas, inputs, kwargs)
        return self.gather(outputs, self.output_device)

    def parallel_apply_predict(self, replicas, inputs, kwargs):
        """Parrallel prediction."""
        return parallel_apply_predict(replicas, inputs, kwargs, self.device_ids[:len(replicas)])


class Clump(nn.Module):
    """Clipping input tensor.

    Args:
        min_v: Min value.
        max_v: Max value.

    """

    def __init__(self, min_v: 'int'=-50, max_v: 'int'=50):
        super(Clump, self).__init__()
        self.min_v = min_v
        self.max_v = max_v

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        """Forward-pass."""
        x = torch.clamp(x, self.min_v, self.max_v)
        return x


class TextBert(nn.Module):
    """Text data model.

    Class for working with text data based on HuggingFace transformers.

    Args:
        model_name: Transformers model name.
        pooling: Pooling type.

    Note:
        There are different pooling types:

            - cls: Use CLS token for sentence embedding
                from last hidden state.
            - max: Maximum on seq_len dimension for non masked
                inputs from last hidden state.
            - mean: Mean on seq_len dimension for non masked
                inputs from last hidden state.
            - sum: Sum on seq_len dimension for non masked
                inputs from last hidden state.
            - none: Without pooling for seq2seq models.

    """
    _poolers = {'cls', 'max', 'mean', 'sum', 'none'}

    def __init__(self, model_name: 'str'='bert-base-uncased', pooling: 'str'='cls'):
        super(TextBert, self).__init__()
        if pooling not in self._poolers:
            raise ValueError('pooling - {} - not in the list of available types {}'.format(pooling, self._poolers))
        self.transformer = AutoModel.from_pretrained(model_name)
        self.n_out = self.transformer.config.hidden_size
        self.dropout = torch.nn.Dropout(0.2)
        self.activation = torch.nn.ReLU(inplace=True)
        self.pooling = pooling_by_name[pooling]()

    def get_out_shape(self) ->int:
        """Output shape.

        Returns:
            int with module output shape.

        """
        return self.n_out

    def forward(self, inp: 'Dict[str, torch.Tensor]') ->torch.Tensor:
        """Forward-pass."""
        encoded_layers, _ = self.transformer(input_ids=inp['input_ids'], attention_mask=inp['attention_mask'], token_type_ids=inp.get('token_type_ids'), return_dict=False)
        encoded_layers = self.pooling(encoded_layers, inp['attention_mask'].unsqueeze(-1).bool())
        mean_last_hidden_state = self.activation(encoded_layers)
        mean_last_hidden_state = self.dropout(mean_last_hidden_state)
        return mean_last_hidden_state


class CatEmbedder(nn.Module):
    """Category data model.

    Args:
        cat_dims: Sequence with number of unique categories
            for category features.
        emb_dropout: Dropout probability.
        emb_ratio: Ratio for embedding size = (x + 1) // emb_ratio.
        max_emb_size: Max embedding size.

    """

    def __init__(self, cat_dims: 'Sequence[int]', emb_dropout: 'bool'=0.1, emb_ratio: 'int'=3, max_emb_size: 'int'=50):
        super(CatEmbedder, self).__init__()
        emb_dims = [(int(x), int(min(max_emb_size, max(1, (x + 1) // emb_ratio)))) for x in cat_dims]
        self.no_of_embs = sum([y for x, y in emb_dims])
        assert self.no_of_embs != 0, 'The input is empty.'
        self.emb_layers = nn.ModuleList([nn.Embedding(x, y) for x, y in emb_dims])
        self.emb_dropout_layer = nn.Dropout(emb_dropout)

    def get_out_shape(self) ->int:
        """Output shape.

        Returns:
            Int with module output shape.

        """
        return self.no_of_embs

    def forward(self, inp: 'Dict[str, torch.Tensor]') ->torch.Tensor:
        """Forward-pass."""
        output = torch.cat([emb_layer(inp['cat'][:, i]) for i, emb_layer in enumerate(self.emb_layers)], dim=1)
        output = self.emb_dropout_layer(output)
        return output


class ContEmbedder(nn.Module):
    """Numeric data model.

    Class for working with numeric data.

    Args:
        num_dims: Sequence with number of numeric features.
        input_bn: Use 1d batch norm for input data.

    """

    def __init__(self, num_dims: 'int', input_bn: 'bool'=True):
        super(ContEmbedder, self).__init__()
        self.n_out = num_dims
        self.bn = None
        if input_bn:
            self.bn = nn.BatchNorm1d(num_dims)
        assert num_dims != 0, 'The input is empty.'

    def get_out_shape(self) ->int:
        """Output shape.

        Returns:
            int with module output shape.

        """
        return self.n_out

    def forward(self, inp: 'Dict[str, torch.Tensor]') ->torch.Tensor:
        """Forward-pass."""
        output = inp['cont']
        if self.bn is not None:
            output = self.bn(output)
        return output


class TorchUniversalModel(nn.Module):
    """Mixed data model.

    Class for preparing input for DL model with mixed data.

    Args:
        loss: Callable torch loss with order of arguments (y_true, y_pred).
        task: Task object.
        n_out: Number of output dimensions.
        cont_embedder: Torch module for numeric data.
        cont_params: Dict with numeric model params.
        cat_embedder: Torch module for category data.
        cat_params: Dict with category model params.
        text_embedder: Torch module for text data.
        text_params: Dict with text model params.
        bias: Array with last hidden linear layer bias.

    """

    def __init__(self, loss: 'Callable', task: 'Task', n_out: 'int'=1, cont_embedder: 'Optional[Any]'=None, cont_params: 'Optional[Dict]'=None, cat_embedder: 'Optional[Any]'=None, cat_params: 'Optional[Dict]'=None, text_embedder: 'Optional[Any]'=None, text_params: 'Optional[Dict]'=None, bias: 'Optional[Sequence]'=None):
        super(TorchUniversalModel, self).__init__()
        self.n_out = n_out
        self.loss = loss
        self.task = task
        self.cont_embedder = None
        self.cat_embedder = None
        self.text_embedder = None
        n_in = 0
        if cont_embedder is not None:
            self.cont_embedder = cont_embedder(**cont_params)
            n_in += self.cont_embedder.get_out_shape()
        if cat_embedder is not None:
            self.cat_embedder = cat_embedder(**cat_params)
            n_in += self.cat_embedder.get_out_shape()
        if text_embedder is not None:
            self.text_embedder = text_embedder(**text_params)
            n_in += self.text_embedder.get_out_shape()
        self.bn = nn.BatchNorm1d(n_in)
        self.fc = torch.nn.Linear(n_in, self.n_out)
        if bias is not None:
            bias = torch.Tensor(bias)
            self.fc.bias.data = nn.Parameter(bias)
            self.fc.weight.data = nn.Parameter(torch.zeros(self.n_out, n_in))
        if self.task.name == 'binary' or self.task.name == 'multilabel':
            self.fc = nn.Sequential(self.fc, Clump(), nn.Sigmoid())
        elif self.task.name == 'multiclass':
            self.fc = nn.Sequential(self.fc, Clump(), nn.Softmax(dim=1))

    def forward(self, inp: 'Dict[str, torch.Tensor]') ->torch.Tensor:
        """Forward-pass."""
        x = self.predict(inp)
        loss = self.loss(inp['label'].view(inp['label'].shape[0], -1), x, inp.get('weight', None))
        return loss

    def predict(self, inp: 'Dict[str, torch.Tensor]') ->torch.Tensor:
        """Prediction."""
        outputs = []
        if self.cont_embedder is not None:
            outputs.append(self.cont_embedder(inp))
        if self.cat_embedder is not None:
            outputs.append(self.cat_embedder(inp))
        if self.text_embedder is not None:
            outputs.append(self.text_embedder(inp))
        if len(outputs) > 1:
            output = torch.cat(outputs, dim=1)
        else:
            output = outputs[0]
        logits = self.fc(output)
        return logits.view(logits.shape[0], -1)


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (CatLinear,
     lambda: ([], {}),
     lambda: ([], {})),
    (Clump,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (CustomDataParallel,
     lambda: ([], {'module': torch.nn.ReLU()}),
     lambda: ([], {'input': torch.rand([4, 4])})),
    (GumbelTopKSampler,
     lambda: ([], {'T': 4, 'k': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (SequenceClsPooler,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (SequenceIndentityPooler,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (SoftSubSampler,
     lambda: ([], {'T': 4, 'k': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
]

