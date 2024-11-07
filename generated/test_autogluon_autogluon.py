
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


import logging


import pandas as pd


from pandas import DataFrame


from typing import Union


from typing import Tuple


from types import ModuleType


from typing import Any


from typing import Dict


from typing import Optional


import numpy as np


from sklearn.neighbors import NearestNeighbors


import copy


import time


import typing


from collections import defaultdict


from typing import List


import random


import torch as th


from sklearn.model_selection import train_test_split


import warnings


from sklearn.model_selection import StratifiedKFold


from sklearn.metrics import mean_squared_error


import torch


import torch.nn as nn


import torch.nn.functional as F


from sklearn.pipeline import make_pipeline


from sklearn.preprocessing import StandardScaler


from sklearn.svm import SVC


from torch.utils.data import Dataset as TorchDataset


from sklearn.metrics.pairwise import paired_cosine_distances


from scipy.stats import pearsonr


from scipy.stats import spearmanr


from torch import tensor


import math


from torch.utils.data import DataLoader


from torch.utils.data import Dataset


import collections


from typing import Sequence


from numpy import random


from torch import nn


import re


from torchvision import transforms


from typing import Callable


from copy import deepcopy


from sklearn.pipeline import Pipeline


from typing import Iterable


import matplotlib.pyplot as plt


from scipy.special import softmax


from torch.distributions.normal import Normal


import torch.utils.checkpoint


from torch import Tensor


import enum


from typing import cast


from abc import ABC


from abc import abstractclassmethod


from abc import abstractmethod


from functools import lru_cache


import torch._dynamo


from torch.nn.modules.loss import _Loss


from typing import Generator


from typing import Mapping


import functools


from torch.optim.lr_scheduler import LambdaLR


from scipy.ndimage import convolve


from scipy.ndimage import distance_transform_edt as bwdist


from torch import optim


from torch.nn import functional as F


import uuid


from typing import IO


from scipy.special import expit


from sklearn.metrics import f1_score


from sklearn.metrics import log_loss


import numpy.testing as npt


from sklearn.isotonic import IsotonicRegression


from functools import partial


import sklearn


from torch.nn import Module


from torch.nn import init


from torch.nn.functional import dropout


from torch.nn.functional import linear


from torch.nn.functional import softmax


from torch.nn.init import constant_


from torch.nn.init import xavier_normal_


from torch.nn.init import xavier_uniform_


from torch.nn.modules.dropout import Dropout


from torch.nn.modules.normalization import LayerNorm


from torch.nn.parameter import Parameter


from collections import Counter


from sklearn.feature_extraction.text import TfidfVectorizer


from sklearn.preprocessing import KBinsDiscretizer


from sklearn.preprocessing import PowerTransformer


from sklearn.preprocessing import QuantileTransformer


from sklearn.preprocessing import RobustScaler


from typing import Literal


from typing import Iterator


from typing import Type


from sklearn.compose import ColumnTransformer


from uuid import uuid4


class AutoMMMemoryBank(nn.Module):
    """
    The model to generate few shot predict probability with 
    features extracted by AutoGluon MultiModal Predictor.
    """

    def __init__(self, bank_keys, bank_labels, hidden_size, num_classes, clip_weights=None, model_head_type='linear'):
        """
        Create the model head and the memory bank.

        Parameters
        ----------
        bank_keys
            The content of bank composed of features in the training set.
        bank_labels 
            The labels of corresponding bank_keys.
        hidden_size
            The size of features.
        num_classes
            The classes of the dataset.
        clip_weights
            The clip embedding of the semantic text that describes the labels.
        model_head_type 
            The type of the few-shot classification head.
        """
        super(AutoMMMemoryBank, self).__init__()
        self.bank_keys = bank_keys
        self.bank_values = F.one_hot(bank_labels).float()
        self.adapter = nn.Linear(bank_keys.shape[0], bank_keys.shape[1], bias=False)
        self.adapter.weight = nn.Parameter(bank_keys.t())
        self.clip_weights = clip_weights
        self.model_head_type = model_head_type
        if clip_weights is None:
            if model_head_type == 'SVM':
                self.model_head = make_pipeline(StandardScaler(), SVC(gamma='auto', probability=True))
                self.model_head.fit(bank_keys.t().cpu(), bank_labels.cpu())
            else:
                self.model_head = nn.Linear(hidden_size, num_classes, bias=True) if clip_weights is None else None
        else:
            self.model_head = None

    def adapt_logits(self, affinity, pure_logits, alpha, beta):
        """
        Generate logits with memory bank based on pure_logits and bank output.

        Parameters
        ----------
        affinity
            The result of bank similarity. It is based on cosine similarity or a projector initialized with bank_keys.
        pure_logits
            The predict probability of the classifier.
        alpha
            The hyper-parameters of bank model.
        beta
            The hyper-parameters of bank model.
        
        Return
        ------
            The logits with memory bank.
        """
        bank_logits = (-1 * (beta - beta * affinity)).exp() @ self.bank_values
        logits = pure_logits + bank_logits * alpha
        return logits

    def change_head_state(self, grad_state):
        """
        Change the training state of the model head.

        Parameters
        ----------
        grad_state
            The training state of the model head. If "True", the model head is trainable. If "False", the model head is freezed.
        """
        if self.model_head is not None and self.model_head_type == 'linear':
            for param in self.model_head.parameters():
                param.requires_grad = grad_state

    def change_adapter_state(self, grad_state):
        """
        Change the training state of the memory bank.

        Parameters
        ----------
        grad_state
            The training state of the memory bank. If "True", the memory bank is trainable. If "False", the memory bank is freezed.
        """
        for param in self.adapter.parameters():
            param.requires_grad = grad_state

    def forward(self, x, alpha=1, beta=1, pure_logits=None):
        """
        Generate three types of logits with features.

        Parameters
        ----------
        x
            The image/text features generated by AutoGluon Multimodal Predictor.
        alpha
            The hyper-parameters of memory bank model.
        beta
            The hyper-parameters of memory bank model.
        
        Return
        ------
        The predict probability of the feature.
            - "pure_logits"
                The predict probability of classifier.
            - "adapted_logits"
                The predict probability composed of classifier and memory bank similarity result.
            - "adapted_logits_with_finetuning"
                The predict probability composed of classifier and fine-tuned memory bank result.
        """
        if pure_logits is None:
            if self.clip_weights is not None:
                pure_logits = 100.0 * x @ self.clip_weights
            elif self.model_head_type == 'SVM':
                pure_logits = torch.tensor(self.model_head.predict_proba(x.cpu()))
            else:
                pure_logits = self.model_head(x)
        affinity = x @ self.bank_keys
        adapted_logits = self.adapt_logits(affinity, pure_logits, alpha, beta)
        finetuned_affinity = self.adapter(x)
        adapted_logits_with_finetuning = self.adapt_logits(finetuned_affinity, pure_logits, alpha, beta)
        return {'pure_logits': pure_logits, 'adapted_logits': adapted_logits, 'adapted_logits_with_finetuning': adapted_logits_with_finetuning}


def identity(x):
    return x


class LoRALayer:
    """
    Abstract LoRA Layer.

    Parameters
    ----------
    r
        rank r of the low-rank decomposition.
    lora_alpha
        Scaling factor. Can be simply set to same value as r as initialization is scaled already.
    merge_weights
        Merging weights during inference to reduce latency.

    References
    ----------
    1. Edward J. Hu*, Yelong Shen*, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen,
    "LoRA: Low-Rank Adaptation of Large Language Models", 2021
    https://arxiv.org/abs/2106.09685
    2. Code: https://github.com/microsoft/LoRA
    """

    def __init__(self, r: 'int', lora_alpha: 'int', lora_dropout: 'float', merge_weights: 'bool'):
        self.r = r
        self.lora_alpha = lora_alpha
        if lora_dropout > 0.0:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = identity
        self.merged = False
        self.merge_weights = merge_weights


class IA3LoRALinear(nn.Linear, LoRALayer):
    """
    LoRA (low-rank adaptation) followed by (IA)^3 (weight rescaling) incorporated in a Linear Layer. Weights of Linear layer are set to be frozen per default.

    Parameters
    ----------
    in_features
        input dimension, set to the original linear layer input dimension LoRA is replacing.
    out_features
        output dimension, set to the original linear layer output dimension LoRA is replacing.
    r
        rank r of the low-rank decomposition.
    lora_alpha
        Scaling factor. Can be simply set to same value as r as initialization is scaled already.
    lora_dropout
        Dropout probability.
    fan_in_fan_out
        Set this to True if the layer to replace stores weight like (fan_in, fan_out).
    merge_weights
        Merging weights during inference to reduce latency.
    """

    def __init__(self, in_features: 'int', out_features: 'int', r=8, lora_alpha=8, lora_dropout: 'float'=0.0, fan_in_fan_out=False, merge_weights=False, **kwargs):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights)
        self.lora_b = nn.Parameter(torch.ones(out_features, 1))
        self.lora_A = nn.Parameter(self.weight.new_zeros((r, in_features)))
        self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, r)))
        self.fan_in_fan_out = fan_in_fan_out
        self.weight.requires_grad = False
        self.scaling = self.lora_alpha / self.r
        self.reset_parameters()

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def T(self, w):
        return w.T if self.fan_in_fan_out else w

    def forward(self, x: 'torch.Tensor'):
        result = F.linear(x, self.T(self.weight), bias=self.bias)
        if self.r > 0:
            result += self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T * self.scaling
        hidden = result * self.lora_b.flatten()
        return hidden

    def train(self, mode: 'bool'=True):
        nn.Linear.train(self, mode)
        if self.merge_weights and self.merged:
            if self.r > 0:
                self.weight.data /= self.lora_b.flatten()
                self.weight.data -= self.T(self.lora_B @ self.lora_A) * self.scaling
            self.merged = False

    def eval(self):
        nn.Linear.eval(self)
        if self.merge_weights and not self.merged:
            if self.r > 0:
                self.weight.data += self.T(self.lora_B @ self.lora_A) * self.scaling
                self.weight.data *= self.lora_b.flatten()
            self.merged = True
        return hidden

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(self.in_features, self.out_features, self.bias is not None)


class IA3Linear(nn.Linear, LoRALayer):
    """
    (IA)^3 incorporated in a Linear Layer. Weights of Linear layer are set to be frozen per default.

    Parameters
    ----------
    in_features
        input dimension, set to the original linear layer input dimension LoRA is replacing.
    out_features
        output dimension, set to the original linear layer output dimension LoRA is replacing.
    scaling_rank
        Merging weights during inference to reduce latency.

    References
    ----------
    1. Liu, Haokun and Tam, Derek and Muqeeth, Mohammed and Mohta, Jay and Huang, Tenghao and Bansal, Mohit and Raffel, Colin,
    "Few-Shot Parameter-Efficient Fine-Tuning is Better and Cheaper than In-Context Learning", 2022
    https://arxiv.org/pdf/2205.05638.pdf
    2. Code: https://github.com/r-three/t-few
    """

    def __init__(self, in_features: 'int', out_features: 'int', merge_weights: 'False', **kwargs):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=4, lora_alpha=4, lora_dropout=0.0, merge_weights=merge_weights)
        self.lora_b = nn.Parameter(torch.ones(out_features, 1))
        self.weight.requires_grad = False

    def forward(self, x: 'torch.Tensor'):
        hidden = F.linear(x, self.weight, self.bias)
        hidden = hidden * self.lora_b.flatten()
        return hidden

    def train(self, mode: 'bool'=True):
        nn.Linear.train(self, mode)
        if self.merge_weights and self.merged:
            if self.r > 0:
                self.weight.data /= self.lora_b.flatten()
            self.merged = False

    def eval(self):
        nn.Linear.eval(self)
        if self.merge_weights and not self.merged:
            if self.r > 0:
                self.weight.data *= self.lora_b.flatten()
            self.merged = True
        return hidden

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(self.in_features, self.out_features, self.bias is not None)


class LoRALinear(nn.Linear, LoRALayer):
    """
    LoRA incorporated in Linear Layer. Weights of linear layer are set to be frozen per default.

    Parameters
    ----------
    in_features
        input dimension, set to the original linear layer input dimension LoRA is replacing.
    out_features
        output dimension, set to the original linear layer output dimension LoRA is replacing.
    r
        rank r of the low-rank decomposition.
    lora_alpha
        Scaling factor. Can be simply set to same value as r as initialization is scaled already.
    lora_dropout
        Dropout probability.
    fan_in_fan_out
        Set this to True if the layer to replace stores weight like (fan_in, fan_out).
    merge_weights
        Merging weights during inference to reduce latency.

    References
    ----------
    1. Edward J. Hu*, Yelong Shen*, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen,
    "LoRA: Low-Rank Adaptation of Large Language Models", 2021
    https://arxiv.org/abs/2106.09685
    2. Code: https://github.com/microsoft/LoRA
    """

    def __init__(self, in_features: 'int', out_features: 'int', r: 'int'=0, lora_alpha: 'int'=1, lora_dropout: 'float'=0.0, fan_in_fan_out: 'bool'=False, merge_weights: 'bool'=True, **kwargs):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights)
        self.fan_in_fan_out = fan_in_fan_out
        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, in_features)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, r)))
            self.scaling = self.lora_alpha / self.r
            self.weight.requires_grad = False
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def T(self, w):
        return w.T if self.fan_in_fan_out else w

    def train(self, mode: 'bool'=True):
        nn.Linear.train(self, mode)
        if self.merge_weights and self.merged:
            if self.r > 0:
                self.weight.data -= self.T(self.lora_B @ self.lora_A) * self.scaling
            self.merged = False

    def eval(self):
        nn.Linear.eval(self)
        if self.merge_weights and not self.merged:
            if self.r > 0:
                self.weight.data += self.T(self.lora_B @ self.lora_A) * self.scaling
            self.merged = True

    def forward(self, x: 'torch.Tensor'):
        if self.r > 0 and not self.merged:
            result = F.linear(x, self.T(self.weight), bias=self.bias)
            if self.r > 0:
                result += self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T * self.scaling
            return result
        else:
            return F.linear(x, self.T(self.weight), bias=self.bias)


class LoRAEmbedding(nn.Embedding, LoRALayer):
    """
    LoRA incorporated in Embedding Layer. Weights of embedding layer are set to be frozen per default.

    Parameters
    ----------
    num_embeddings
        size of the dictionary of embeddings.
    embedding_dim
         the size of each embedding vector.
    r
        rank r of the low-rank decomposition.
    lora_alpha
        Scaling factor. Can be simply set to same value as r as initialization is scaled already.
    merge_weights
        Merging weights during inference to reduce latency.

    References
    ----------
    1. Edward J. Hu*, Yelong Shen*, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen,
    "LoRA: Low-Rank Adaptation of Large Language Models", 2021
    https://arxiv.org/abs/2106.09685
    2. Code: https://github.com/microsoft/LoRA
    """

    def __init__(self, num_embeddings: 'int', embedding_dim: 'int', r: 'int'=0, lora_alpha: 'int'=1, merge_weights: 'bool'=True, **kwargs):
        nn.Embedding.__init__(self, num_embeddings, embedding_dim, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=0, merge_weights=merge_weights)
        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, num_embeddings)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((embedding_dim, r)))
            self.scaling = self.lora_alpha / self.r
            self.weight.requires_grad = False
        self.reset_parameters()

    def reset_parameters(self):
        nn.Embedding.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            nn.init.zeros_(self.lora_A)
            nn.init.normal_(self.lora_B)

    def train(self, mode: 'bool'=True):
        nn.Embedding.train(self, mode)
        if self.merge_weights and self.merged:
            if self.r > 0:
                self.weight.data -= (self.lora_B @ self.lora_A).T * self.scaling
            self.merged = False

    def eval(self):
        nn.Linear.eval(self)
        if self.merge_weights and not self.merged:
            if self.r > 0:
                self.weight.data += self.lora_B @ self.lora_A * self.scaling
            self.merged = True

    def forward(self, x: 'torch.Tensor'):
        if self.r > 0 and not self.merged:
            result = nn.Embedding.forward(self, x)
            if self.r > 0:
                after_A = F.embedding(x, self.lora_A.T, self.padding_idx, self.max_norm, self.norm_type, self.scale_grad_by_freq, self.sparse)
                result += after_A @ self.lora_B.T * self.scaling
            return result
        else:
            return nn.Embedding.forward(self, x)


class LoRAMergedLinear(nn.Linear, LoRALayer):
    """
    LoRA where single nn.Linear represents more than one layer (used in some implementations of attention query/key/value projections). Weights of linear layer are set to be frozen per default.

    Parameters
    ----------
    in_features
        input dimension, set to the original linear layer input dimension LoRA is replacing
    out_features
        output dimension, set to the original linear layer output dimension LoRA is replacing
    r
        rank r of the low-rank decomposition
    lora_alpha
        Scaling factor. Can be simply set to same value as r as initialization is scaled already.
    lora_dropout
        Dropout rate for LoRA
    fan_in_fan_out
        Set this to True if the layer to replace stores weight like (fan_in, fan_out)
    merge_weights
        Merging weights during inference to reduce latency

    References
    ----------
    1. Edward J. Hu*, Yelong Shen*, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen,
    "LoRA: Low-Rank Adaptation of Large Language Models", 2021
    https://arxiv.org/abs/2106.09685
    2. Code: https://github.com/microsoft/LoRA
    """

    def __init__(self, in_features: 'int', out_features: 'int', r: 'int'=0, lora_alpha: 'int'=1, lora_dropout: 'float'=0.0, enable_lora: 'List[bool]'=[False], fan_in_fan_out: 'bool'=False, merge_weights: 'bool'=True, **kwargs):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights)
        assert out_features % len(enable_lora) == 0, 'The length of enable_lora must divide out_features'
        self.enable_lora = enable_lora
        self.fan_in_fan_out = fan_in_fan_out
        if r > 0 and any(enable_lora):
            self.lora_A = nn.Parameter(self.weight.new_zeros((r * sum(enable_lora), in_features)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((out_features // len(enable_lora) * sum(enable_lora), r)))
            self.scaling = self.lora_alpha / self.r
            self.weight.requires_grad = False
            self.lora_ind = self.weight.new_zeros((out_features,), dtype=torch.bool).view(len(enable_lora), -1)
            self.lora_ind[enable_lora, :] = True
            self.lora_ind = self.lora_ind.view(-1)
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def zero_pad(self, x):
        result = x.new_zeros((*x.shape[:-1], self.out_features))
        result = result.view(-1, self.out_features)
        result[:, self.lora_ind] = x.reshape(-1, self.out_features // len(self.enable_lora) * sum(self.enable_lora))
        return result.view((*x.shape[:-1], self.out_features))

    def train(self, mode: 'bool'=True):

        def T(w):
            return w.T if self.fan_in_fan_out else w
        nn.Linear.train(self, mode)
        if self.merge_weights and self.merged:
            if self.r > 0 and any(self.enable_lora):
                delta_w = F.conv1d(self.lora_A.data.unsqueeze(0), self.lora_B.data.unsqueeze(-1), groups=sum(self.enable_lora)).squeeze(0)
                self.weight.data -= self.zero_pad(T(delta_w * self.scaling))
            self.merged = False

    def eval(self):

        def T(w):
            return w.T if self.fan_in_fan_out else w
        nn.Linear.eval(self)
        if self.merge_weights and not self.merged:
            if self.r > 0 and any(self.enable_lora):
                delta_w = F.conv1d(self.lora_A.data.unsqueeze(0), self.lora_B.data.unsqueeze(-1), groups=sum(self.enable_lora)).squeeze(0)
                self.weight.data += self.zero_pad(T(delta_w * self.scaling))
            self.merged = True

    def forward(self, x: 'torch.Tensor'):

        def T(w):
            return w.T if self.fan_in_fan_out else w
        if self.merged:
            return F.linear(x, T(self.weight), bias=self.bias)
        else:
            result = F.linear(x, T(self.weight), bias=self.bias)
            if self.r > 0:
                after_A = F.linear(self.lora_dropout(x), self.lora_A)
                after_B = F.conv1d(after_A.transpose(-2, -1), self.lora_B.unsqueeze(-1), groups=sum(self.enable_lora)).transpose(-2, -1)
                result += self.zero_pad(after_B) * self.scaling
            return result


class LoRAConv2d(nn.Conv2d, LoRALayer):
    """
    LoRA incorporated in 2d-Convolutional Layer. Weights of convolutional layer are set to be frozen per default.

    Parameters
    ----------
    in_channels
         Number of channels in the input image.
    out_channels
        Number of channels produced by the convolution.
    kernel_size
        Size of the convolving kernel.
    r
        rank r of the low-rank decomposition.
    lora_alpha
        Scaling factor. Can be simply set to same value as r as initialization is scaled already.
    lora_dropout
        Adding dropout to LoRA.
    merge_weights
        Merging weights during inference to reduce latency.

    References
    ----------
    1. Edward J. Hu*, Yelong Shen*, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen,
    "LoRA: Low-Rank Adaptation of Large Language Models", 2021
    https://arxiv.org/abs/2106.09685
    2. Code: https://github.com/microsoft/LoRA
    """

    def __init__(self, in_channels: 'int', out_channels: 'int', kernel_size: 'int', r: 'int'=0, lora_alpha: 'int'=1, lora_dropout: 'float'=0.0, merge_weights: 'bool'=True, **kwargs):
        nn.Conv2d.__init__(self, in_channels, out_channels, kernel_size, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights)
        assert type(kernel_size) is int
        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((r * kernel_size, in_channels * kernel_size)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((out_channels * kernel_size, r * kernel_size)))
            self.scaling = self.lora_alpha / self.r
            self.weight.requires_grad = False
        self.reset_parameters()

    def reset_parameters(self):
        nn.Conv2d.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def train(self, mode: 'bool'=True):
        nn.Conv2d.train(self, mode)
        if self.merge_weights and self.merged:
            self.weight.data -= (self.lora_B @ self.lora_A).view(self.weight.shape) * self.scaling
            self.merged = False

    def eval(self):
        nn.Conv2d.eval(self)
        if self.merge_weights and not self.merged:
            self.weight.data += (self.lora_B @ self.lora_A).view(self.weight.shape) * self.scaling
            self.merged = True

    def forward(self, x: 'torch.Tensor'):
        if self.r > 0 and not self.merged:
            return F.conv2d(x, self.weight + (self.lora_B @ self.lora_A).view(self.weight.shape) * self.scaling, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return nn.Conv2d.forward(self, x)


class MoEGate(nn.Module):

    def __init__(self, d, M=4, K=1, noisy_gating=True):
        """Constructor
        Args:
            d: input channel dimensionality.
            M: the number of experts.
            K: the number of chosen experts for each forward pass.
        """
        super(MoEGate, self).__init__()
        self.M = M
        self.k = K
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.noisy_gating = noisy_gating
        self.w_gate = nn.Parameter(torch.zeros(d, M), requires_grad=True)
        self.w_noise = nn.Parameter(torch.zeros(d, M), requires_grad=True)
        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)
        self.register_buffer('mean', torch.tensor([0.0]))
        self.register_buffer('std', torch.tensor([1.0]))
        assert self.k <= self.M

    def forward(self, feats, loss_coef=0.01, noise_epsilon=0.01):
        batch_size = feats.shape[0]
        feats_S = self.gap(feats).view(batch_size, -1)
        clean_logits = feats_S @ self.w_gate
        if self.noisy_gating and self.training:
            raw_noise_stddev = feats_S @ self.w_noise
            noise_stddev = self.softplus(raw_noise_stddev) + noise_epsilon
            noisy_logits = clean_logits + torch.randn_like(clean_logits) * noise_stddev
            logits = noisy_logits
        else:
            logits = clean_logits
        top_logits, top_indices = logits.topk(min(self.k + 1, self.M), dim=1)
        top_k_logits = top_logits[:, :self.k]
        top_k_indices = top_indices[:, :self.k]
        top_k_gates = self.softmax(top_k_logits)
        zeros = torch.zeros_like(logits, requires_grad=True).float()
        gates = zeros.scatter(1, top_k_indices, top_k_gates)
        if self.noisy_gating and self.k < self.M and self.training:
            load = self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits).sum(0)
        else:
            load = self._gates_to_load(gates)
        importance = gates.sum(0)
        loss = self.cv_squared(importance) + self.cv_squared(load)
        loss *= loss_coef
        return gates, loss

    def _gates_to_load(self, gates):
        """Compute the true load per expert, given the gates.
        The load is the number of examples for which the corresponding gate is >0.
        Args:
        gates: a `Tensor` of shape [batch_size, n]
        Returns:
        a float32 `Tensor` of shape [n]
        """
        return (gates > 0).sum(0)

    def cv_squared(self, x):
        """The squared coefficient of variation of a sample.
        Useful as a loss to encourage a positive distribution to be more uniform.
        Epsilons added for numerical stability.
        Returns 0 for an empty Tensor.
        Args:
        x: a `Tensor`.
        Returns:
        a `Scalar`.
        """
        eps = 1e-10
        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean() ** 2 + eps)

    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        """Helper function to NoisyTopKGating.
        Computes the probability that value is in top k, given different random noise.
        This gives us a way of backpropagating from a loss that balances the number
        of times each expert is in the top k experts per example.
        In the case of no noise, pass in None for noise_stddev, and the result will
        not be differentiable.
        Args:
        clean_values: a `Tensor` of shape [batch, n].
        noisy_values: a `Tensor` of shape [batch, n].  Equal to clean values plus
          normally distributed noise with standard deviation noise_stddev.
        noise_stddev: a `Tensor` of shape [batch, n], or None
        noisy_top_values: a `Tensor` of shape [batch, m].
           "values" Output of tf.top_k(noisy_top_values, m).  m >= k+1
        Returns:
        a `Tensor` of shape [batch, n].
        """
        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()
        threshold_positions_if_in = torch.arange(batch, device=clean_values.device) * m + self.k
        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)
        normal = Normal(self.mean, self.std)
        prob_if_in = normal.cdf((clean_values - threshold_if_in) / noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out) / noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob


class SparseDispatcher(object):
    """Helper for implementing a mixture of experts.
    The purpose of this class is to create input minibatches for the
    experts and to combine the results of the experts to form a unified
    output tensor.
    There are two functions:
    dispatch - take an input Tensor and create input Tensors for each expert.
    combine - take output Tensors from each expert and form a combined output
      Tensor.  Outputs from different experts for the same batch element are
      summed together, weighted by the provided "gates".
    The class is initialized with a "gates" Tensor, which specifies which
    batch elements go to which experts, and the weights to use when combining
    the outputs.  Batch element b is sent to expert e iff gates[b, e] != 0.
    The inputs and outputs are all two-dimensional [batch, depth].
    Caller is responsible for collapsing additional dimensions prior to
    calling this class and reshaping the output to the original shape.
    See common_layers.reshape_like().
    Example use:
    gates: a float32 `Tensor` with shape `[batch_size, num_experts]`
    inputs: a float32 `Tensor` with shape `[batch_size, input_size]`
    experts: a list of length `num_experts` containing sub-networks.
    dispatcher = SparseDispatcher(num_experts, gates)
    expert_inputs = dispatcher.dispatch(inputs)
    expert_outputs = [experts[i](expert_inputs[i]) for i in range(num_experts)]
    outputs = dispatcher.combine(expert_outputs)
    The preceding code sets the output for a particular example b to:
    output[b] = Sum_i(gates[b, i] * experts[i](inputs[b]))
    This class takes advantage of sparsity in the gate matrix by including in the
    `Tensor`s for expert i only the batch elements for which `gates[b, i] > 0`.

    References
    ----------
    1. Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc Le, Geoffrey Hinton, Jeff Dean,
    "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer", 2017
    https://arxiv.org/abs/1701.06538
    2. Code: https://github.com/davidmrau/mixture-of-experts/blob/master/moe.py
    """

    def __init__(self, num_experts, gates):
        """Create a SparseDispatcher."""
        self._gates = gates
        self._num_experts = num_experts
        sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(0)
        _, self._expert_index = sorted_experts.split(1, dim=1)
        self._batch_index = torch.nonzero(gates)[index_sorted_experts[:, 1], 0]
        self._part_sizes = (gates > 0).sum(0).tolist()
        gates_exp = gates[self._batch_index.flatten()]
        self._nonzero_gates = torch.gather(gates_exp, 1, self._expert_index)

    def dispatch(self, inp):
        """Create one input Tensor for each expert.
        The `Tensor` for a expert `i` contains the slices of `inp` corresponding
        to the batch elements `b` where `gates[b, i] > 0`.
        Args:
          inp: a `Tensor` of shape "[batch_size, <extra_input_dims>]`
        Returns:
          a list of `num_experts` `Tensor`s with shapes
            `[expert_batch_size_i, <extra_input_dims>]`.
        """
        inp_exp = inp[self._batch_index].squeeze(1)
        return torch.split(inp_exp, self._part_sizes, dim=0)

    def combine(self, expert_out, multiply_by_gates=True):
        """Sum together the expert output, weighted by the gates.
        The slice corresponding to a particular batch element `b` is computed
        as the sum over all experts `i` of the expert output, weighted by the
        corresponding gate values.  If `multiply_by_gates` is set to False, the
        gate values are ignored.
        Args:
          expert_out: a list of `num_experts` `Tensor`s, each with shape
            `[expert_batch_size_i, <extra_output_dims>]`.
          multiply_by_gates: a boolean
        Returns:
          a `Tensor` with shape `[batch_size, <extra_output_dims>]`.
        """
        stitched = torch.cat(expert_out, 0).exp()
        if multiply_by_gates:
            stitched = stitched.mul(self._nonzero_gates.unsqueeze(-1).unsqueeze(-1))
        zeros = torch.zeros(self._gates.size(0), expert_out[-1].size()[1], expert_out[-1].size()[2], expert_out[-1].size()[3], requires_grad=True, device=stitched.device)
        combined = zeros.index_add(0, self._batch_index, stitched.float())
        combined[combined == 0] = np.finfo(float).eps
        return combined.log()

    def expert_to_gates(self):
        """Gate values corresponding to the examples in the per-expert `Tensor`s.
        Returns:
          a list of `num_experts` one-dimensional `Tensor`s with type `tf.float32`
              and shapes `[expert_batch_size_i]`
        """
        return torch.split(self._nonzero_gates, self._part_sizes, dim=0)


class ConvLoRALinear(nn.Linear, LoRALayer):
    """
    Conv-LoRA incorporated in Linear Layer. Weights of linear layer are set to be frozen per default.

    Parameters
    ----------
    in_features
        input dimension, set to the original linear layer input dimension LoRA is replacing.
    out_features
        output dimension, set to the original linear layer output dimension LoRA is replacing.
    r
        rank r of the low-rank decomposition.
    lora_alpha
        Scaling factor. Can be simply set to same value as r as initialization is scaled already.
    lora_dropout
        Dropout probability.
    fan_in_fan_out
        Set this to True if the layer to replace stores weight like (fan_in, fan_out).
    merge_weights
        Merging weights during inference to reduce latency.
    conv_lora_expert_num
        The number of experts in MoE-Conv.

    References
    ----------
    1. Zihan Zhong, Zhiqiang Tang, Tong He, Haoyang Fang, Chun Yuan,
    "Convolution Meets LoRA: Parameter Efficient Finetuning for Segment Anything Model", 2024
    https://arxiv.org/abs/2401.17868
    """

    def __init__(self, in_features: 'int', out_features: 'int', r: 'int'=0, lora_alpha: 'int'=1, lora_dropout: 'float'=0.0, fan_in_fan_out: 'bool'=False, merge_weights: 'bool'=False, conv_lora_expert_num: 'Optional[int]'=None, **kwargs):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights)
        self.fan_in_fan_out = fan_in_fan_out
        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, in_features)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, r)))
            self.scaling = self.lora_alpha / self.r
            self.weight.requires_grad = False
            topk = 1
            self.lora_moe_gating = MoEGate(M=conv_lora_expert_num, d=self.r, K=topk)
            self.lora_moe_experts = nn.ModuleList([])
            self.upsample_ratios = list(range(1, conv_lora_expert_num + 1))
            for upsample_ratio in self.upsample_ratios:
                expert = nn.Conv2d(in_channels=r, out_channels=r, kernel_size=3, stride=1, padding=1, bias=True)
                expert.bias.data.zero_()
                self.lora_moe_experts.append(nn.Sequential(expert, nn.GELU()))
            self.num_experts = conv_lora_expert_num
            self.multiply_by_gates = False
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def T(self, w):
        return w.T if self.fan_in_fan_out else w

    def forward(self, x: 'torch.Tensor'):
        result = F.linear(x, self.T(self.weight), bias=self.bias)
        if self.r > 0:
            lora_res = self.lora_dropout(x) @ self.lora_A.T
            dim = lora_res.dim()
            if dim == 3:
                B, L, C = lora_res.size()
                H = W = int(math.sqrt(L))
                lora_res = lora_res.reshape(B, H, W, C)
            else:
                H, W = lora_res.size()[1:3]
            lora_res = lora_res.permute(0, 3, 1, 2).contiguous()
            gates, moe_loss = self.lora_moe_gating(lora_res)
            dispatcher = SparseDispatcher(self.num_experts, gates)
            expert_inputs = dispatcher.dispatch(lora_res)
            expert_outputs = []
            for i in range(self.num_experts):
                if len(expert_inputs[i]) == 0:
                    continue
                upsample_ratio = self.upsample_ratios[i]
                cur_res = expert_inputs[i]
                if upsample_ratio != 1:
                    cur_res = F.interpolate(cur_res, scale_factor=upsample_ratio, mode='bicubic')
                cur_res = self.lora_moe_experts[i](cur_res)
                if upsample_ratio != 1:
                    cur_res = F.interpolate(cur_res, size=(int(H), int(W)), mode='bicubic')
                expert_outputs.append(cur_res)
            temp_lora_res = dispatcher.combine(expert_outputs, multiply_by_gates=self.multiply_by_gates)
            lora_res = lora_res + temp_lora_res
            lora_res = lora_res.permute(0, 2, 3, 1).contiguous()
            if dim == 3:
                lora_res = lora_res.reshape(B, L, C)
            result += lora_res @ self.lora_B.T * self.scaling
        return result, moe_loss


CATEGORICAL = 'categorical'


FEATURES = 'features'


LABEL = '__label__'


LOGITS = 'logits'


ALL_ACT_LAYERS = {'leaky_relu': nn.LeakyReLU, 'gelu': nn.GELU, 'relu': nn.ReLU}


class GhostBatchNorm(nn.Module):
    """
    Ghost Batch Normalization.
    It allows the use of large batch sizes,
    but with batch normalization parameters calculated on smaller sub-batches.

    [1] Train longer, generalize better: closing the generalization gap in large batch training of neural networks : https://arxiv.org/abs/1705.08741
    [2] Simple Modifications to Improve Tabular Neural Networks: https://arxiv.org/pdf/2108.03214
    """

    def __init__(self, input_dim: 'int', virtual_batch_size: 'Optional[int]'=64, momentum: 'Optional[float]'=0.01):
        super(GhostBatchNorm, self).__init__()
        self.input_dim = input_dim
        self.virtual_batch_size = virtual_batch_size
        self.bn = nn.BatchNorm1d(self.input_dim, momentum=momentum)

    def forward(self, x):
        chunks = x.chunk(int(np.ceil(x.shape[0] / self.virtual_batch_size)), 0)
        res = [self.bn(x_) for x_ in chunks]
        return torch.cat(res, dim=0)


class Unit(nn.Module):
    """
    One MLP layer. It orders the operations as: norm -> fc -> act_fn -> dropout
    """

    def __init__(self, normalization: 'str', in_features: 'int', out_features: 'int', activation: 'str', dropout_prob: 'float'):
        """
        Parameters
        ----------
        normalization
            Name of activation function.
        in_features
            Dimension of input features.
        out_features
            Dimension of output features.
        activation
            Name of activation function.
        dropout_prob
            Dropout probability.
        """
        super().__init__()
        if normalization == 'layer_norm':
            self.norm = nn.LayerNorm(in_features)
        elif normalization == 'batch_norm':
            self.norm = nn.BatchNorm1d(in_features)
        elif normalization == 'ghost_batch_norm':
            self.norm = GhostBatchNorm(in_features)
        else:
            raise ValueError(f'unknown normalization: {normalization}')
        self.fc = nn.Linear(in_features, out_features)
        self.act_fn = ALL_ACT_LAYERS[activation]()
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        x = self.norm(x)
        x = self.fc(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        return x


class MLP(nn.Module):
    """
    Multi-layer perceptron (MLP). If the hidden or output feature dimension is
    not provided, we assign it the input feature dimension.
    """

    def __init__(self, in_features: 'int', hidden_features: 'Optional[int]'=None, out_features: 'Optional[int]'=None, num_layers: 'Optional[int]'=1, activation: 'Optional[str]'='gelu', dropout_prob: 'Optional[float]'=0.5, normalization: 'Optional[str]'='layer_norm'):
        """
        Parameters
        ----------
        in_features
            Dimension of input features.
        hidden_features
            Dimension of hidden features.
        out_features
            Dimension of output features.
        num_layers
            Number of layers.
        activation
            Name of activation function.
        dropout_prob
            Dropout probability.
        normalization
            Name of normalization function.
        """
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        layers = []
        for _ in range(num_layers):
            per_unit = Unit(normalization=normalization, in_features=in_features, out_features=hidden_features, activation=activation, dropout_prob=dropout_prob)
            in_features = hidden_features
            layers.append(per_unit)
        if out_features != hidden_features:
            self.fc_out = nn.Linear(hidden_features, out_features)
        else:
            self.fc_out = None
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)
        if self.fc_out is not None:
            return self.fc_out(x)
        else:
            return x


def init_weights(module: 'nn.Module'):
    """
    Initialize one module. It uses xavier_norm to initialize nn.Embedding
    and xavier_uniform to initialize nn.Linear's weight.

    Parameters
    ----------
    module
        A Pytorch nn.Module.
    """
    if isinstance(module, nn.Embedding):
        nn.init.xavier_normal_(module.weight)
    elif isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)


class CategoricalMLP(nn.Module):
    """
    MLP for categorical input. The input dimension is automatically computed based on
    the number of categories in each categorical column.
    """

    def __init__(self, prefix: 'str', num_categories: 'List[int]', out_features: 'Optional[int]'=None, num_layers: 'Optional[int]'=1, activation: 'Optional[str]'='gelu', dropout_prob: 'Optional[float]'=0.5, normalization: 'Optional[str]'='layer_norm', num_classes: 'Optional[int]'=0):
        """
        Parameters
        ----------
        prefix
            The model prefix.
        num_categories
            A list of integers. Each one is the number of categories in one categorical column.
        out_features
            Dimension of output features.
        num_layers
            Number of MLP layers.
        activation
            Name of activation function.
        dropout_prob
            Dropout probability.
        normalization
            Name of normalization function.
        num_classes
            Number of classes. 1 for a regression task.
        """
        super().__init__()
        self.out_features = out_features
        max_embedding_dim = 100
        embed_exponent = 0.56
        size_factor = 1.0
        self.column_embeddings = nn.ModuleList()
        self.column_mlps = nn.ModuleList()
        assert isinstance(num_categories, list)
        for num_categories_per_col in num_categories:
            embedding_dim_per_col = int(size_factor * max(2, min(max_embedding_dim, 1.6 * num_categories_per_col ** embed_exponent)))
            self.column_embeddings.append(nn.Embedding(num_embeddings=num_categories_per_col, embedding_dim=embedding_dim_per_col))
            self.column_mlps.append(MLP(in_features=embedding_dim_per_col, hidden_features=out_features, out_features=out_features, num_layers=num_layers, activation=activation, dropout_prob=dropout_prob, normalization=normalization))
        self.aggregator_mlp = MLP(in_features=out_features * len(num_categories), hidden_features=out_features * len(num_categories), out_features=out_features, num_layers=num_layers, activation=activation, dropout_prob=dropout_prob, normalization=normalization)
        self.head = nn.Linear(out_features, num_classes) if num_classes > 0 else nn.Identity()
        self.apply(init_weights)
        self.prefix = prefix
        self.name_to_id = self.get_layer_ids()
        self.head_layer_names = [n for n, layer_id in self.name_to_id.items() if layer_id == 0]

    @property
    def categorical_key(self):
        return f'{self.prefix}_{CATEGORICAL}'

    @property
    def input_keys(self):
        return [self.categorical_key]

    @property
    def label_key(self):
        return f'{self.prefix}_{LABEL}'

    def forward(self, batch: 'dict'):
        """

        Parameters
        ----------
        batch
            A dictionary containing the input mini-batch data.
            We need to use the keys with the model prefix to index required data.

        Returns
        -------
            A dictionary with logits and features.
        """
        assert len(batch[self.categorical_key]) == len(self.column_embeddings)
        features = []
        for categorical_id, embed, mlp in zip(batch[self.categorical_key], self.column_embeddings, self.column_mlps):
            features.append(mlp(embed(categorical_id)))
        cat_features = torch.cat(features, dim=1)
        features = self.aggregator_mlp(cat_features)
        logits = self.head(features)
        return {self.prefix: {LOGITS: logits, FEATURES: features}}

    def get_layer_ids(self):
        """
        All layers have the same id 0 since there is no pre-trained models used here.

        Returns
        -------
        A dictionary mapping the layer names (keys) to their ids (values).
        """
        name_to_id = {}
        for n, _ in self.named_parameters():
            name_to_id[n] = 0
        return name_to_id


COLUMN = 'column'


COLUMN_FEATURES = 'column_features'


IMAGE = 'image'


IMAGE_VALID_NUM = 'image_valid_num'


LOGIT_SCALE = 'logit_scale'


MASKS = 'masks'


TEXT_TOKEN_IDS = 'text_token_ids'


TEXT_VALID_LENGTH = 'text_valid_length'


def assign_encoder_layer_ids(encoder_names: 'List[List[str]]'):
    """
    Assign ids to encoder layers. The encoder may contain several blocks e.g., block1 and block2.
    This function iterates through all the layers of each block from the input end towards the output end.
    It increases 1 on the layer id when the detected digit in a layer name changes.

    Parameters
    ----------
    encoder_names
        Encoder layer names.

    Returns
    -------
    name_to_id
        The encoder layer-to-id mapping.
    encoder_layer_num
        The encoder layer number.
    """
    name_to_id = {}
    cur_id = 0
    for i, group_names in enumerate(encoder_names):
        last_inferred_id = -1
        for n in group_names:
            detect_id = False
            n_splits = n.split('.')
            for split in n_splits:
                if split.isdigit():
                    inferred_id = int(split)
                    if inferred_id != last_inferred_id:
                        cur_id += 1
                        last_inferred_id = inferred_id
                    name_to_id[n] = cur_id
                    detect_id = True
                    break
            if detect_id is False:
                raise ValueError(f'parameter name: {n} not has no id inside')
    if len(name_to_id) > 0:
        encoder_layer_num = max(name_to_id.values())
    else:
        encoder_layer_num = 0
    return name_to_id, encoder_layer_num


def assign_non_encoder_layer_ids(non_encoder_names: 'List[str]', layer_id: 'int'):
    """
    Assign the provided id to non-encoder layers.

    Parameters
    ----------
    non_encoder_names
        Names layers not belonging to an encoder.
    layer_id
        provided id.

    Returns
    -------
    A dictionary mapping the layer names (keys) to their ids (values).
    """
    name_to_id = {}
    for n in non_encoder_names:
        name_to_id[n] = layer_id
    return name_to_id


def split_encoder_non_encoder(names: 'List[str]', post_encoder_patterns: 'Tuple[str, ...]'):
    """
    Group layer names into two types: encoder and non-encoder.
    A layer belongs to encoder if its name contains at least one digit.
    It uses this rule since a model's encoder in Pytorch's implementation
    is generally wrapped by nn.Sequential() or nn.ModuleList(),
    which produce digits in layer names.

    Parameters
    ----------
    names
        Model layer names.
    Returns
    -------
    encoder_names
        A list of encoder layer names.
    non_encoder_names
        A list of non-encoder layer names.
    """
    encoder_names = []
    non_encoder_names = []
    for n in names:
        is_encoder = False
        if any(p in n for p in post_encoder_patterns):
            non_encoder_names.append(n)
            continue
        for i in n.split('.'):
            if i.isdigit():
                encoder_names.append(n)
                is_encoder = True
                break
        if not is_encoder:
            non_encoder_names.append(n)
    return encoder_names, non_encoder_names


def group_param_names(names: 'List[str]', pre_encoder_patterns: 'Tuple[str, ...]', post_encoder_patterns: 'Tuple[str, ...]', model_prefix: 'Optional[str]'=None):
    """
    Group layer names into three types: pre-encoder, encoder, and post-encoder.
    If "model_prefix" is provided, the selected layer names must start with it.
    In this case, the left names will be returned for the next-time processing.
    This function first extracts the first-level children modules' names and
    classify them into encoder and non-encoder layers. Note that an encoder may
    consist of several manually named children modules, e.g., block1 and block2.
    The non-encoder layers are further subdivided into pre-encoder and post-encoder.

    Parameters
    ----------
    names
        Model layer names
    pre_encoder_patterns
        Patterns to identify a layer as a pre-encoder layer. If a layer name contains one pattern,
        the layer will be grouped into pre-encoder layers.
    post_encoder_patterns
        Patterns to identify a layer as a post-encoder layer. If a layer name contains one pattern,
        the layer will be grouped into post-encoder layers.
    model_prefix
        A prefix to filter layer names. Only layer names starting with it will be selected.
    Returns
    -------
    left_names
        The layer names left for the next-time processing.
    encoder_names_grouped
        Encoder layer names.
    pre_encoder_names
        Names of layers before the encoder.
    post_encoder_names
        Names of layers after the encoder.
    """
    assert all(pre_p not in post_encoder_patterns for pre_p in pre_encoder_patterns)
    left_names = []
    selected_names = []
    for n in names:
        if model_prefix is not None and not n.startswith(model_prefix):
            left_names.append(n)
        else:
            selected_names.append(n)
    children_prefix = []
    for n in selected_names:
        if model_prefix is not None:
            child_name = n[len(model_prefix) + 1:].split('.')[0]
            child_prefix = f'{model_prefix}.{child_name}'
        else:
            child_prefix = n.split('.')[0]
        if child_prefix not in children_prefix:
            children_prefix.append(child_prefix)
    encoder_names_grouped = []
    non_encoder_names = []
    for child_prefix in children_prefix:
        per_names_group = [n for n in selected_names if n.startswith(child_prefix)]
        per_encoder_names, per_non_encoder_names = split_encoder_non_encoder(per_names_group, post_encoder_patterns)
        encoder_names_grouped.append(per_encoder_names)
        non_encoder_names.extend(per_non_encoder_names)
    pre_encoder_names = []
    post_encoder_names = []
    for n in non_encoder_names:
        if any(p in n for p in pre_encoder_patterns):
            pre_encoder_names.append(n)
        elif any(p in n for p in post_encoder_patterns):
            post_encoder_names.append(n)
        else:
            raise ValueError(f'parameter name: {n} belong to neither pre or post encoder names')
    return left_names, encoder_names_grouped, pre_encoder_names, post_encoder_names


logger = logging.getLogger(__name__)


def reverse_layer_ids(encoder_name_to_id: 'dict', pre_enocder_name_to_id: 'dict', post_enocder_name_to_id: 'dict'):
    """
    The layer ids need to increase when going from the output end to the input end.
    We need to reverse the ids which were originally assigned in a decreasing order.

    Parameters
    ----------
    encoder_name_to_id
        The layer-to-id mapping of encoder layers.
    pre_enocder_name_to_id
        The layer-to-id mapping of pre-encoder layers.
    post_enocder_name_to_id
        The layer-to-id mapping of post-encoder layers.

    Returns
    -------
    The layer-to-id mapping of all layers with layer ids reversed.
    """
    name_to_id = {**pre_enocder_name_to_id, **encoder_name_to_id, **post_enocder_name_to_id}
    if len(name_to_id) > 0:
        layer_num = max(name_to_id.values())
        if len(post_enocder_name_to_id) == 0:
            layer_num += 1
    for n, layer_id in name_to_id.items():
        name_to_id[n] = layer_num - layer_id
    return name_to_id


def assign_layer_ids(names: 'List[str]', pre_encoder_patterns: 'Tuple[str, ...]', post_encoder_patterns: 'Tuple[str, ...]', model_pre: 'Optional[str]'=None):
    """
    Assign ids to all layers. It splits a model into three parts: pre-encoder, encoder, and post-encoder.
    Encoder is generally a stack of multiple similar layers, such as transformer layers. Since encoder is
    generally wrapped by nn.Sequential() or nn.ModuleList(), its inside layer names contain digits.
    It sets 0 as the ids of all post-encoder layers and a maximum id (layer_num) for the all the pre-encoder
    layers. The encoder layers have decreasing ids from the input to the output ends.

    Parameters
    ----------
    names
        model layer names.
    pre_encoder_patterns
        Patterns to identify a layer as a pre-encoder layer. If a layer name contains one pattern,
        the layer will be grouped into pre-encoder layers.
    post_encoder_patterns
        Patterns to identify a layer as a post-encoder layer. If a layer name contains one pattern,
        the layer will be grouped into post-encoder layers.
    model_pre
        The layer names' prefix. Only the layer names with this prefix will be assigned ids. The left
        layer names will be returned.

    Returns
    -------
    name_to_id
        A dictionary mapping the layer names (keys) to their ids (values).
    left_names
        The layer names not starting with the "model_pre".
    """
    try:
        left_names, encoder_names, pre_encoder_names, post_encoder_names = group_param_names(names=names, pre_encoder_patterns=pre_encoder_patterns, post_encoder_patterns=post_encoder_patterns, model_prefix=model_pre)
        if len(encoder_names) == 0 and len(pre_encoder_names) != 0:
            raise ValueError(f'encoder_names is empty, but pre_encoder_names has values: {pre_encoder_names}')
        encoder_name_to_id, encoder_layer_num = assign_encoder_layer_ids(encoder_names=encoder_names)
        pre_encoder_name_to_id = assign_non_encoder_layer_ids(non_encoder_names=pre_encoder_names, layer_id=0)
        post_encoder_name_to_id = assign_non_encoder_layer_ids(non_encoder_names=post_encoder_names, layer_id=encoder_layer_num + 1)
        name_to_id = reverse_layer_ids(encoder_name_to_id=encoder_name_to_id, pre_enocder_name_to_id=pre_encoder_name_to_id, post_enocder_name_to_id=post_encoder_name_to_id)
    except Exception as e:
        logger.debug(f'When calling assign_layer_ids(), it catches exception: {e}. All the layers will use the same layer_id.')
        name_to_id = dict()
        left_names = names
    return name_to_id, left_names


def get_column_features(batch: 'Dict[str, torch.Tensor]', column_name_prefix: 'str', features: 'torch.Tensor', valid_lengths: 'torch.Tensor', cls_feature: 'Optional[torch.Tensor]'=None):
    """
    Index the features of one column defined by `column_name_prefix`.
    This function can be used to index both image and text features.
    The features have shape (b, n, d), where n can be the image number or
    text token number. One column corresponds to a subset of
    the n images or text tokens. One column name can only appear once in the return.

    Parameters
    ----------
    batch
        The batch input containing the feature column information, i.e., indexes.
    column_name_prefix
        The column name prefix of one modality (image or text).
    features
        The features of columns whose names starts with column_name_prefix.
    valid_lengths
        The valid image number or text token number of each sample in a batch.
    cls_feature
        The cls feature containing information from all feature columns.

    Returns
    -------
    The column features with masks. If the column has no valid features, its
    mask is 0.
    """
    column_features = {}
    feature_masks = {}
    cut_idx = len(column_name_prefix) + 1
    if cls_feature is not None:
        all_column_names = []
        joint_mask = torch.zeros(features.shape[0])
    for key in batch:
        if key.startswith(column_name_prefix):
            per_col_features = []
            per_col_masks = torch.zeros(features.shape[0])
            assert batch[key].ndim == 2 and batch[key].shape[1] == 2
            for i, per_sample_col_idx in enumerate(batch[key]):
                start_idx = per_sample_col_idx[0]
                end_idx = per_sample_col_idx[1]
                if start_idx < end_idx:
                    assert end_idx <= valid_lengths[i]
                    per_col_features.append(features[i, start_idx:end_idx].mean(dim=0))
                    per_col_masks[i] = 1
                else:
                    per_col_features.append(torch.zeros_like(features[0, 0]))
                    per_col_masks[i] = 0
            column_name = key[cut_idx:]
            column_features[column_name] = torch.stack(per_col_features, dim=0)
            feature_masks[column_name] = per_col_masks
            if cls_feature is not None:
                all_column_names.append(column_name)
                joint_mask = torch.logical_or(joint_mask, per_col_masks)
    if cls_feature is not None and len(all_column_names) > 0:
        for column_name in all_column_names:
            column_features.pop(column_name)
            feature_masks.pop(column_name)
        joint_column_name = '_'.join(all_column_names)
        column_features[joint_column_name] = cls_feature
        feature_masks[joint_column_name] = joint_mask
    return column_features, feature_masks


def get_hf_config_and_model(checkpoint_name: 'str', pretrained: 'Optional[bool]'=True, low_cpu_mem_usage: 'Optional[bool]'=False):
    """
    Get a Huggingface config and model based on a checkpoint name.

    Parameters
    ----------
    checkpoint_name
        A model checkpoint name or a local path that saves a custom checkpoint.
    pretrained
         Whether using the pretrained weights. If pretrained=True, download the pretrained model.
    low_cpu_mem_usage
        Whether to turn on the optimization of reducing the peak CPU memory usage when loading the pretrained model.

    Returns
    -------
    A Huggingface config and model.
    """
    config = AutoConfig.from_pretrained(checkpoint_name)
    if pretrained:
        model = AutoModel.from_pretrained(checkpoint_name, low_cpu_mem_usage=low_cpu_mem_usage)
    else:
        model = AutoModel.from_config(config)
    return config, model


def get_pretrained_tokenizer(tokenizer_name: 'str', checkpoint_name: 'str', use_fast: 'Optional[bool]'=True, add_prefix_space: 'Optional[bool]'=None):
    """
    Load the tokenizer for a pre-trained huggingface checkpoint.

    Parameters
    ----------
    tokenizer_name
        The tokenizer type, e.g., "bert", "clip", "electra", and "hf_auto".
    checkpoint_name
        Name of a pre-trained checkpoint.
    use_fast
        Use a fast Rust-based tokenizer if it is supported for a given model.
        If a fast tokenizer is not available for a given model, a normal Python-based tokenizer is returned instead.
        See: https://huggingface.co/docs/transformers/model_doc/auto#transformers.AutoTokenizer.from_pretrained.use_fast

    Returns
    -------
    A tokenizer instance.
    """
    try:
        tokenizer_class = ALL_TOKENIZERS[tokenizer_name]
        if add_prefix_space is None:
            return tokenizer_class.from_pretrained(checkpoint_name, use_fast=use_fast)
        else:
            return tokenizer_class.from_pretrained(checkpoint_name, use_fast=use_fast, add_prefix_space=add_prefix_space)
    except TypeError as e:
        try:
            tokenizer = BertTokenizer.from_pretrained(checkpoint_name)
            warnings.warn(f'Current checkpoint {checkpoint_name} does not support AutoTokenizer. Switch to BertTokenizer instead.', UserWarning)
            return tokenizer
        except:
            raise e


class CLIPForImageText(nn.Module):
    """
    Support the CLIP model.
    Refer to https://huggingface.co/docs/transformers/model_doc/clip
    """

    def __init__(self, prefix: 'str', checkpoint_name: 'str', num_classes: 'Optional[int]'=None, pretrained: 'Optional[bool]'=True, tokenizer_name: 'Optional[str]'='clip'):
        """
        Load the pretrained CLIP from huggingface transformers.

        Parameters
        ----------
        prefix
            The model prefix.
        checkpoint_name
            Name of the checkpoint.
        num_classes
            The number of classes. 1 for a regression task.
        pretrained
            Whether using the pretrained weights. If pretrained=True, download the pretrained model.
        tokenizer_name
            Name of the huggingface tokenizer type.
        """
        super().__init__()
        logger.debug(f'initializing {checkpoint_name}')
        self.checkpoint_name = checkpoint_name
        self.num_classes = num_classes
        self.config, self.model = get_hf_config_and_model(checkpoint_name=checkpoint_name, pretrained=pretrained)
        self.tokenizer_name = tokenizer_name
        self.tokenizer = get_pretrained_tokenizer(tokenizer_name=self.tokenizer_name, checkpoint_name=self.checkpoint_name)
        self.out_features = self.model.config.projection_dim
        self.head = nn.Linear(self.out_features, num_classes) if num_classes else nn.Identity()
        self.head.apply(init_weights)
        self.prefix = prefix
        self.name_to_id = self.get_layer_ids()
        self.head_layer_names = [n for n, layer_id in self.name_to_id.items() if layer_id == 0]

    @property
    def text_token_ids_key(self):
        return f'{self.prefix}_{TEXT_TOKEN_IDS}'

    @property
    def text_valid_length_key(self):
        return f'{self.prefix}_{TEXT_VALID_LENGTH}'

    @property
    def image_key(self):
        return f'{self.prefix}_{IMAGE}'

    @property
    def image_valid_num_key(self):
        return f'{self.prefix}_{IMAGE_VALID_NUM}'

    @property
    def label_key(self):
        return f'{self.prefix}_{LABEL}'

    @property
    def text_column_prefix(self):
        return f'{self.text_token_ids_key}_{COLUMN}'

    @property
    def image_column_prefix(self):
        return f'{self.image_key}_{COLUMN}'

    @property
    def text_feature_dim(self):
        return self.model.config.text_config.hidden_size

    @property
    def image_feature_dim(self):
        return self.model.config.vision_config.hidden_size

    def forward(self, batch: 'dict'):
        """
        Parameters
        ----------
        batch
            A dictionary containing the input mini-batch data.
            We need to use the keys with the model prefix to index required data.

        Returns
        -------
            A dictionary with logits and features.
        """
        has_image = self.image_key in batch
        has_text = self.text_token_ids_key in batch
        ret = {COLUMN_FEATURES: {FEATURES: {}, MASKS: {}}}
        if has_image:
            images = batch[self.image_key]
            image_valid_num = batch[self.image_valid_num_key]
            assert images.dim() == 5
            b, n, c, h, w = images.shape
            vision_outputs = self.model.vision_model(pixel_values=images.reshape((b * n, c, h, w)), output_attentions=True, output_hidden_states=True, return_dict=True)
            image_features = self.model.visual_projection(vision_outputs.pooler_output)
            steps = torch.arange(0, n).type_as(image_valid_num)
            image_masks = (steps.reshape((1, -1)) < image_valid_num.reshape((-1, 1))).type_as(image_features)
            image_features = image_features.reshape((b, n, -1)) * image_masks[:, :, None]
            image_features = image_features / torch.clamp(image_features.norm(dim=-1, keepdim=True), min=1e-06)
            image_column_features, image_column_feature_masks = get_column_features(batch=batch, column_name_prefix=self.image_column_prefix, features=image_features, valid_lengths=image_valid_num)
            ret[COLUMN_FEATURES][FEATURES].update(image_column_features)
            ret[COLUMN_FEATURES][MASKS].update(image_column_feature_masks)
            image_features = image_features.mean(dim=1)
            ret[FEATURES] = image_features
        if has_text:
            text_token_ids = batch[self.text_token_ids_key]
            text_valid_length = batch[self.text_valid_length_key]
            steps = torch.arange(0, text_token_ids.shape[1]).type_as(text_valid_length)
            text_masks = (steps.reshape((1, -1)) < text_valid_length.reshape((-1, 1))).type_as(text_token_ids)
            assert torch.equal(text_valid_length, text_masks.sum(dim=-1))
            text_outputs = self.model.text_model(input_ids=text_token_ids, attention_mask=text_masks, output_attentions=True, output_hidden_states=True, return_dict=True)
            text_features = self.model.text_projection(text_outputs.pooler_output)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            text_column_features, text_column_feature_masks = get_column_features(batch=batch, column_name_prefix=self.text_column_prefix, features=self.model.text_projection(text_outputs.last_hidden_state), valid_lengths=text_valid_length, cls_feature=text_features)
            ret[COLUMN_FEATURES][FEATURES].update(text_column_features)
            ret[COLUMN_FEATURES][MASKS].update(text_column_feature_masks)
            ret[FEATURES] = text_features
        if has_image and has_text:
            if self.num_classes:
                features = image_features + text_features
                logits = self.head(features)
                ret[FEATURES] = features
            else:
                logits = torch.sum(image_features * text_features, dim=-1)
            ret[LOGITS] = logits
        ret[LOGIT_SCALE] = self.model.logit_scale.exp()
        return {self.prefix: ret}

    def get_layer_ids(self):
        """
        Assign an id to each layer. Layer ids will be used in layer-wise lr decay.
        Basically, id gradually increases when going from the output end to
        the input end. The layers defined in this class, e.g., head, have id 0.

        Returns
        -------
        A dictionary mapping the layer names (keys) to their ids (values).
        """
        model_prefixes = ['model.text_model', 'model.vision_model', 'model']
        for i, model_pre in enumerate(model_prefixes):
            for model_pre2 in model_prefixes[i + 1:]:
                if model_pre2.startswith(model_pre):
                    raise ValueError(f'{model_pre} is a substring of {model_pre2}. Need to swap them in {model_prefixes}.')
        pre_encoder_patterns = 'embeddings', 'pre'
        post_encoder_patterns = 'head', 'final', 'post', 'logit', 'project'
        names = [n for n, _ in self.named_parameters()]
        name_to_id = {}
        for per_prefix in model_prefixes:
            per_model_name_to_id, names = assign_layer_ids(names=names, pre_encoder_patterns=pre_encoder_patterns, post_encoder_patterns=post_encoder_patterns, model_pre=per_prefix)
            name_to_id.update(per_model_name_to_id)
        if len(names) > 0:
            logger.debug(f'outer layers are treated as head: {names}')
        for n in names:
            assert n not in name_to_id
            name_to_id[n] = 0
        return name_to_id


class SamPatchEmbeddings(nn.Module):
    """
    This class turns `pixel_values` of shape `(batch_size, num_channels, height, width)` into the initial
    `hidden_states` (patch embeddings) of shape `(batch_size, seq_length, hidden_size)` to be consumed by a
    Transformer.
    """

    def __init__(self, config):
        super().__init__()
        image_size, patch_size = config.image_size, config.patch_size
        num_channels, hidden_size = config.num_channels, config.hidden_size
        image_size = image_size if isinstance(image_size, collections.abc.Iterable) else (image_size, image_size)
        patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)
        num_patches = image_size[1] // patch_size[1] * (image_size[0] // patch_size[0])
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_patches = num_patches
        self.projection = nn.Conv2d(num_channels, hidden_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, pixel_values):
        batch_size, num_channels, height, width = pixel_values.shape
        if num_channels != self.num_channels:
            raise ValueError('Make sure that the channel dimension of the pixel values match with the one set in the configuration.')
        if height != self.image_size[0] or width != self.image_size[1]:
            raise ValueError(f"Input image size ({height}*{width}) doesn't match model ({self.image_size[0]}*{self.image_size[1]}).")
        embeddings = self.projection(pixel_values).permute(0, 2, 3, 1)
        return embeddings


class SamMLPBlock(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.lin1 = nn.Linear(config.hidden_size, config.mlp_dim)
        self.lin2 = nn.Linear(config.mlp_dim, config.hidden_size)
        self.act = ACT2FN[config.hidden_act]

    def forward(self, hidden_states: 'torch.Tensor') ->torch.Tensor:
        hidden_states = self.lin1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.lin2(hidden_states)
        return hidden_states


class SamLayerNorm(nn.Module):
    """LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with shape (batch_size, height,
    width, channels) while channels_first corresponds to inputs with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-06, data_format='channels_last'):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ['channels_last', 'channels_first']:
            raise NotImplementedError(f'Unsupported data format: {self.data_format}')
        self.normalized_shape = normalized_shape,

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        if self.data_format == 'channels_last':
            x = torch.nn.functional.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == 'channels_first':
            input_dtype = x.dtype
            x = x.float()
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = x
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class SamAttention(nn.Module):
    """
    SAM's attention layer that allows for downscaling the size of the embedding after projection to queries, keys, and
    values.
    """

    def __init__(self, config, downsample_rate=None):
        super().__init__()
        self.hidden_size = config.hidden_size
        downsample_rate = config.attention_downsample_rate if downsample_rate is None else downsample_rate
        self.internal_dim = config.hidden_size // downsample_rate
        self.num_attention_heads = config.num_attention_heads
        if self.internal_dim % config.num_attention_heads != 0:
            raise ValueError('num_attention_heads must divide hidden_size.')
        self.q_proj = nn.Linear(self.hidden_size, self.internal_dim)
        self.k_proj = nn.Linear(self.hidden_size, self.internal_dim)
        self.v_proj = nn.Linear(self.hidden_size, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, self.hidden_size)

    def _separate_heads(self, hidden_states: 'Tensor', num_attention_heads: 'int') ->Tensor:
        batch, point_batch_size, n_tokens, channel = hidden_states.shape
        c_per_head = channel // num_attention_heads
        hidden_states = hidden_states.reshape(batch * point_batch_size, n_tokens, num_attention_heads, c_per_head)
        return hidden_states.transpose(1, 2)

    def _recombine_heads(self, hidden_states: 'Tensor', point_batch_size: 'int') ->Tensor:
        batch, n_heads, n_tokens, c_per_head = hidden_states.shape
        hidden_states = hidden_states.transpose(1, 2)
        return hidden_states.reshape(batch // point_batch_size, point_batch_size, n_tokens, n_heads * c_per_head)

    def forward(self, query: 'Tensor', key: 'Tensor', value: 'Tensor', attention_similarity: 'Tensor'=None) ->Tensor:
        query = self.q_proj(query)
        key = self.k_proj(key)
        value = self.v_proj(value)
        point_batch_size = query.shape[1]
        query = self._separate_heads(query, self.num_attention_heads)
        key = self._separate_heads(key, self.num_attention_heads)
        value = self._separate_heads(value, self.num_attention_heads)
        _, _, _, c_per_head = query.shape
        attn = query @ key.permute(0, 1, 3, 2)
        attn = attn / math.sqrt(c_per_head)
        attn = torch.softmax(attn, dim=-1)
        if attention_similarity is not None:
            attn = attn + attention_similarity
            attn = torch.softmax(attn, dim=-1)
        out = attn @ value
        out = self._recombine_heads(out, point_batch_size)
        out = self.out_proj(out)
        return out


class SamTwoWayAttentionBlock(nn.Module):

    def __init__(self, config, attention_downsample_rate: 'int'=2, skip_first_layer_pe: 'bool'=False):
        """
        A transformer block with four layers:
            (1) self-attention of sparse inputs (2) cross attention of sparse inputs -> dense inputs (3) mlp block on
            sparse inputs (4) cross attention of dense inputs -> sparse inputs

        Arguments:
            config (`SamMaskDecoderConfig`):
                The configuration file used to instantiate the block
            attention_downsample_rate (*optionalk*, int, defaults to 2):
                The downsample ratio of the block used to reduce the inner dim of the attention.
            skip_first_layer_pe (*optional*, bool, defaults to `False`):
                Whether or not to skip the addition of the query_point_embedding on the first layer.
        """
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_norm_eps = config.layer_norm_eps
        self.self_attn = SamAttention(config, downsample_rate=1)
        self.layer_norm1 = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.cross_attn_token_to_image = SamAttention(config, downsample_rate=attention_downsample_rate)
        self.layer_norm2 = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.mlp = SamMLPBlock(config)
        self.layer_norm3 = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.layer_norm4 = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.cross_attn_image_to_token = SamAttention(config, downsample_rate=attention_downsample_rate)
        self.skip_first_layer_pe = skip_first_layer_pe

    def forward(self, queries: 'Tensor', keys: 'Tensor', query_point_embedding: 'Tensor', key_point_embedding: 'Tensor', attention_similarity: 'Tensor', output_attentions: 'bool'=False):
        if self.skip_first_layer_pe:
            queries = self.self_attn(query=queries, key=queries, value=queries)
        else:
            query = queries + query_point_embedding
            attn_out = self.self_attn(query=query, key=query, value=queries)
            queries = queries + attn_out
        queries = self.layer_norm1(queries)
        query = queries + query_point_embedding
        key = keys + key_point_embedding
        attn_out = self.cross_attn_token_to_image(query=query, key=key, value=keys, attention_similarity=attention_similarity)
        queries = queries + attn_out
        queries = self.layer_norm2(queries)
        mlp_out = self.mlp(queries)
        queries = queries + mlp_out
        queries = self.layer_norm3(queries)
        query = queries + query_point_embedding
        key = keys + key_point_embedding
        attn_out = self.cross_attn_image_to_token(query=key, key=query, value=queries)
        keys = keys + attn_out
        keys = self.layer_norm4(keys)
        outputs = queries, keys
        if output_attentions:
            outputs = outputs + (attn_out,)
        else:
            outputs = outputs + (None,)
        return outputs


class SamFeedForward(nn.Module):

    def __init__(self, input_dim: 'int', hidden_dim: 'int', output_dim: 'int', num_layers: 'int', sigmoid_output: 'bool'=False):
        super().__init__()
        self.num_layers = num_layers
        self.activation = nn.ReLU()
        self.proj_in = nn.Linear(input_dim, hidden_dim)
        self.proj_out = nn.Linear(hidden_dim, output_dim)
        self.layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers - 2)])
        self.sigmoid_output = sigmoid_output

    def forward(self, hidden_states):
        hidden_states = self.proj_in(hidden_states)
        hidden_states = self.activation(hidden_states)
        for layer in self.layers:
            hidden_states = self.activation(layer(hidden_states))
        hidden_states = self.proj_out(hidden_states)
        if self.sigmoid_output:
            hidden_states = F.sigmoid(hidden_states)
        return hidden_states


class SamMaskDecoder(nn.Module):

    def __init__(self, config: 'SamMaskDecoderConfig'):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_multimask_outputs = config.num_multimask_outputs
        self.num_mask_tokens = config.num_multimask_outputs + 1
        self.iou_token = nn.Embedding(1, self.hidden_size)
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, self.hidden_size)
        self.transformer = SamTwoWayTransformer(config)
        self.upscale_conv1 = nn.ConvTranspose2d(self.hidden_size, self.hidden_size // 4, kernel_size=2, stride=2)
        self.upscale_conv2 = nn.ConvTranspose2d(self.hidden_size // 4, self.hidden_size // 8, kernel_size=2, stride=2)
        self.upscale_layer_norm = SamLayerNorm(self.hidden_size // 4, data_format='channels_first')
        self.activation = nn.GELU()
        mlps_list = []
        for _ in range(self.num_mask_tokens):
            mlps_list += [SamFeedForward(self.hidden_size, self.hidden_size, self.hidden_size // 8, 3)]
        self.output_hypernetworks_mlps = nn.ModuleList(mlps_list)
        self.iou_prediction_head = SamFeedForward(self.hidden_size, config.iou_head_hidden_dim, self.num_mask_tokens, config.iou_head_depth)

    def forward(self, image_embeddings: 'torch.Tensor', image_positional_embeddings: 'torch.Tensor', sparse_prompt_embeddings: 'torch.Tensor', dense_prompt_embeddings: 'torch.Tensor', multimask_output: 'bool', output_attentions: 'Optional[bool]'=None, attention_similarity: 'torch.Tensor'=None, target_embedding: 'torch.Tensor'=None) ->Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict masks given image and prompt embeddings.

        Args:
            image_embeddings (`torch.Tensor`):
                the embeddings from the image encoder
            image_positional_embedding (`torch.Tensor`):
                positional encoding with the shape of image_embeddings
            sparse_prompt_embeddings (`torch.Tensor`):
                The embeddings of the points and boxes
            dense_prompt_embeddings (`torch.Tensor`):
                the embeddings of the mask inputs
            multimask_output (bool):
                Whether to return multiple masks or a single mask.
            output_attentions (bool, *optional*):
                Whether or not to return the attentions tensors of all attention layers.
        """
        batch_size, num_channels, height, width = image_embeddings.shape
        point_batch_size = sparse_prompt_embeddings.shape[1]
        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)
        output_tokens = output_tokens.repeat(batch_size, point_batch_size, 1, 1)
        if sparse_prompt_embeddings.sum().item() != 0:
            tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=2)
        else:
            tokens = output_tokens
        point_embeddings = tokens
        image_embeddings = image_embeddings + dense_prompt_embeddings
        image_embeddings = image_embeddings.repeat(point_batch_size, 1, 1, 1)
        image_positional_embeddings = image_positional_embeddings.repeat(point_batch_size, 1, 1, 1)
        point_embedding, image_embeddings, attentions = self.transformer(point_embeddings=point_embeddings, image_embeddings=image_embeddings, image_positional_embeddings=image_positional_embeddings, attention_similarity=attention_similarity, target_embedding=target_embedding, output_attentions=output_attentions)
        iou_token_out = point_embedding[:, :, 0, :]
        mask_tokens_out = point_embedding[:, :, 1:1 + self.num_mask_tokens, :]
        image_embeddings = image_embeddings.transpose(2, 3).reshape(batch_size * point_batch_size, num_channels, height, width)
        upscaled_embedding = self.upscale_conv1(image_embeddings)
        upscaled_embedding = self.activation(self.upscale_layer_norm(upscaled_embedding))
        upscaled_embedding = self.activation(self.upscale_conv2(upscaled_embedding))
        hyper_in_list = []
        for i in range(self.num_mask_tokens):
            current_mlp = self.output_hypernetworks_mlps[i]
            hyper_in_list += [current_mlp(mask_tokens_out[:, :, i, :])]
        hyper_in = torch.stack(hyper_in_list, dim=2)
        _, num_channels, height, width = upscaled_embedding.shape
        upscaled_embedding = upscaled_embedding.reshape(batch_size, point_batch_size, num_channels, height * width)
        masks = (hyper_in @ upscaled_embedding).reshape(batch_size, point_batch_size, -1, height, width)
        iou_pred = self.iou_prediction_head(iou_token_out)
        if multimask_output:
            mask_slice = slice(1, None)
        else:
            mask_slice = slice(0, 1)
        masks = masks[:, :, mask_slice, :, :]
        iou_pred = iou_pred[:, :, mask_slice]
        outputs = masks, iou_pred
        if output_attentions:
            outputs = outputs + (attentions,)
        else:
            outputs = outputs + (None,)
        return outputs


class SamPositionalEmbedding(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.scale = config.hidden_size // 2
        self.register_buffer('positional_embedding', self.scale * torch.randn((2, config.num_pos_feats)))

    def forward(self, input_coords, input_shape=None):
        """Positionally encode points that are normalized to [0,1]."""
        coordinates = input_coords.clone()
        if input_shape is not None:
            coordinates[:, :, :, 0] = coordinates[:, :, :, 0] / input_shape[1]
            coordinates[:, :, :, 1] = coordinates[:, :, :, 1] / input_shape[0]
        coordinates = 2 * coordinates - 1
        coordinates = coordinates
        coordinates = coordinates @ self.positional_embedding
        coordinates = 2 * np.pi * coordinates
        return torch.cat([torch.sin(coordinates), torch.cos(coordinates)], dim=-1)


class SamMaskEmbedding(nn.Module):

    def __init__(self, config: 'SamPromptEncoderConfig'):
        super().__init__()
        self.mask_input_channels = config.mask_input_channels // 4
        self.activation = ACT2FN[config.hidden_act]
        self.conv1 = nn.Conv2d(1, self.mask_input_channels, kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(self.mask_input_channels, config.mask_input_channels, kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(config.mask_input_channels, config.hidden_size, kernel_size=1)
        self.layer_norm1 = SamLayerNorm(self.mask_input_channels, eps=config.layer_norm_eps, data_format='channels_first')
        self.layer_norm2 = SamLayerNorm(self.mask_input_channels * 4, eps=config.layer_norm_eps, data_format='channels_first')

    def forward(self, masks):
        hidden_states = self.conv1(masks)
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.conv2(hidden_states)
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.activation(hidden_states)
        dense_embeddings = self.conv3(hidden_states)
        return dense_embeddings


class SamPromptEncoder(nn.Module):

    def __init__(self, config: 'SamPromptEncoderConfig', shared_patch_embedding):
        super().__init__()
        self.shared_embedding = shared_patch_embedding
        self.mask_embed = SamMaskEmbedding(config)
        self.no_mask_embed = nn.Embedding(1, config.hidden_size)
        self.image_embedding_size = config.image_embedding_size, config.image_embedding_size
        self.input_image_size = config.image_size
        self.point_embed = nn.ModuleList([nn.Embedding(1, config.hidden_size) for i in range(config.num_point_embeddings)])
        self.hidden_size = config.hidden_size
        self.not_a_point_embed = nn.Embedding(1, config.hidden_size)

    def _embed_points(self, points: 'torch.Tensor', labels: 'torch.Tensor', pad: 'bool') ->torch.Tensor:
        """Embeds point prompts."""
        points = points + 0.5
        if pad:
            target_point_shape = points.shape[0], points.shape[1], 1, points.shape[-1]
            target_labels_shape = points.shape[0], points.shape[1], 1
            padding_point = torch.zeros(target_point_shape, device=points.device)
            padding_label = -torch.ones(target_labels_shape, device=labels.device)
            points = torch.cat([points, padding_point], dim=2)
            labels = torch.cat([labels, padding_label], dim=2)
        input_shape = self.input_image_size, self.input_image_size
        point_embedding = self.shared_embedding(points, input_shape)
        point_embedding = torch.where(labels[..., None] == -1, self.not_a_point_embed.weight, point_embedding)
        point_embedding = torch.where(labels[..., None] != -10, point_embedding, torch.tensor(0.0, dtype=point_embedding.dtype, device=point_embedding.device))
        point_embedding = torch.where((labels == 0)[:, :, :, None], point_embedding + self.point_embed[0].weight[None, None, :, :], point_embedding)
        point_embedding = torch.where((labels == 1)[:, :, :, None], point_embedding + self.point_embed[1].weight[None, None, :, :], point_embedding)
        return point_embedding

    def _embed_boxes(self, boxes: 'torch.Tensor') ->torch.Tensor:
        """Embeds box prompts."""
        boxes = boxes + 0.5
        batch_size, nb_boxes = boxes.shape[:2]
        coords = boxes.reshape(batch_size, nb_boxes, 2, 2)
        input_shape = self.input_image_size, self.input_image_size
        corner_embedding = self.shared_embedding(coords, input_shape)
        corner_embedding[:, :, 0, :] += self.point_embed[2].weight
        corner_embedding[:, :, 1, :] += self.point_embed[3].weight
        return corner_embedding

    def forward(self, input_points: 'Optional[Tuple[torch.Tensor, torch.Tensor]]', input_labels: 'Optional[torch.Tensor]', input_boxes: 'Optional[torch.Tensor]', input_masks: 'Optional[torch.Tensor]') ->Tuple[torch.Tensor, torch.Tensor]:
        """
        Embeds different types of prompts, returning both sparse and dense embeddings.

        Args:
            points (`torch.Tensor`, *optional*):
                point coordinates and labels to embed.
            boxes (`torch.Tensor`, *optional*):
                boxes to embed
            masks (`torch.Tensor`, *optional*):
                masks to embed
        """
        sparse_embeddings = None
        batch_size = 1
        target_device = self.shared_embedding.positional_embedding.device
        if input_points is not None:
            batch_size, point_batch_size = input_points.shape[:2]
            if input_labels is None:
                raise ValueError('If points are provided, labels must also be provided.')
            point_embeddings = self._embed_points(input_points, input_labels, pad=input_boxes is None)
            sparse_embeddings = point_embeddings
        if input_boxes is not None:
            batch_size = input_boxes.shape[0]
            box_embeddings = self._embed_boxes(input_boxes)
            if sparse_embeddings is None:
                sparse_embeddings = box_embeddings
            else:
                sparse_embeddings = torch.cat([sparse_embeddings, box_embeddings], dim=2)
        if input_masks is not None:
            dense_embeddings = self.mask_embed(input_masks)
        else:
            dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(batch_size, -1, self.image_embedding_size[0], self.image_embedding_size[1])
        if sparse_embeddings is None:
            sparse_embeddings = torch.zeros((batch_size, 1, 1, self.hidden_size), device=target_device)
        return sparse_embeddings, dense_embeddings


class SamVisionAttention(nn.Module):
    """Multi-head Attention block with relative position embeddings."""

    def __init__(self, config, window_size):
        super().__init__()
        input_size = (config.image_size // config.patch_size, config.image_size // config.patch_size) if window_size == 0 else (window_size, window_size)
        self.num_attention_heads = config.num_attention_heads
        head_dim = config.hidden_size // config.num_attention_heads
        self.scale = head_dim ** -0.5
        self.dropout = config.attention_dropout
        self.qkv = nn.Linear(config.hidden_size, config.hidden_size * 3, bias=config.qkv_bias)
        self.proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.use_rel_pos = config.use_rel_pos
        if self.use_rel_pos:
            if input_size is None:
                raise ValueError('Input size must be provided if using relative positional encoding.')
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, head_dim))

    def get_rel_pos(self, q_size: 'int', k_size: 'int', rel_pos: 'torch.Tensor') ->torch.Tensor:
        """
        Get relative positional embeddings according to the relative positions of
            query and key sizes.

        Args:
            q_size (int):
                size of the query.
            k_size (int):
                size of key k.
            rel_pos (`torch.Tensor`):
                relative position embeddings (L, channel).

        Returns:
            Extracted positional embeddings according to relative positions.
        """
        max_rel_dist = int(2 * max(q_size, k_size) - 1)
        rel_pos_resized = F.interpolate(rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1), size=max_rel_dist, mode='linear')
        rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)
        q_coords = torch.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
        k_coords = torch.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
        relative_coords = q_coords - k_coords + (k_size - 1) * max(q_size / k_size, 1.0)
        return rel_pos_resized[relative_coords.long()]

    def add_decomposed_rel_pos(self, attn: 'torch.Tensor', query: 'torch.Tensor', rel_pos_h: 'torch.Tensor', rel_pos_w: 'torch.Tensor', q_size: 'Tuple[int, int]', k_size: 'Tuple[int, int]') ->torch.Tensor:
        """
        Calculate decomposed Relative Positional Embeddings from :paper:`mvitv2`.
        https://github.com/facebookresearch/mvit/blob/19786631e330df9f3622e5402b4a419a263a2c80/mvit/models/attention.py

        Args:
            attn (`torch.Tensor`):
                attention map.
            query (`torch.Tensor`):
                query q in the attention layer with shape (batch_size, query_height * query_width, channel).
            rel_pos_h (`torch.Tensor`):
                relative position embeddings (Lh, channel) for height axis.
            rel_pos_w (`torch.Tensor`):
                relative position embeddings (Lw, channel) for width axis.
            q_size (tuple):
                spatial sequence size of query q with (query_height, query_width).
            k_size (tuple):
                spatial sequence size of key k with (key_height, key_width).

        Returns:
            attn (`torch.Tensor`):
                attention map with added relative positional embeddings.
        """
        query_height, query_width = q_size
        key_height, key_width = k_size
        relative_position_height = self.get_rel_pos(query_height, key_height, rel_pos_h)
        relative_position_width = self.get_rel_pos(query_width, key_width, rel_pos_w)
        batch_size, _, dim = query.shape
        reshaped_query = query.reshape(batch_size, query_height, query_width, dim)
        rel_h = torch.einsum('bhwc,hkc->bhwk', reshaped_query, relative_position_height)
        rel_w = torch.einsum('bhwc,wkc->bhwk', reshaped_query, relative_position_width)
        attn = attn.reshape(batch_size, query_height, query_width, key_height, key_width)
        attn = attn + rel_h[:, :, :, :, None] + rel_w[:, :, :, None, :]
        attn = attn.reshape(batch_size, query_height * query_width, key_height * key_width)
        return attn

    def forward(self, hidden_states: 'torch.Tensor', output_attentions=False, output_moe_loss=False) ->torch.Tensor:
        batch_size, height, width, _ = hidden_states.shape
        qkv = self.qkv(hidden_states)
        if output_moe_loss:
            qkv, moe_loss = qkv
        qkv = qkv.reshape(batch_size, height * width, 3, self.num_attention_heads, -1).permute(2, 0, 3, 1, 4)
        query, key, value = qkv.reshape(3, batch_size * self.num_attention_heads, height * width, -1).unbind(0)
        attn_weights = query * self.scale @ key.transpose(-2, -1)
        if self.use_rel_pos:
            attn_weights = self.add_decomposed_rel_pos(attn_weights, query, self.rel_pos_h, self.rel_pos_w, (height, width), (height, width))
        attn_weights = torch.nn.functional.softmax(attn_weights, dtype=torch.float32, dim=-1)
        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        attn_output = (attn_probs @ value).reshape(batch_size, self.num_attention_heads, height, width, -1)
        attn_output = attn_output.permute(0, 2, 3, 1, 4).reshape(batch_size, height, width, -1)
        attn_output = self.proj(attn_output)
        if output_attentions:
            outputs = attn_output, attn_weights
        else:
            outputs = attn_output, None
        if output_moe_loss:
            outputs += moe_loss,
        return outputs


class SamVisionLayer(nn.Module):

    def __init__(self, config, window_size):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attn = SamVisionAttention(config, window_size)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = SamMLPBlock(config)
        self.window_size = window_size

    def window_partition(self, hidden_states: 'torch.Tensor', window_size: 'int') ->Tuple[torch.Tensor, Tuple[int, int]]:
        """
        Args:
        Partition into non-overlapping windows with padding if needed.
            hidden_states (tensor): input tokens with [batch_size, height, width, channel]. window_size (int): window
            size.

        Returns:
            windows: windows after partition with [batch_size * num_windows, window_size, window_size, channel].
            (pad_height, pad_width): padded height and width before partition
        """
        batch_size, height, width, channel = hidden_states.shape
        pad_h = (window_size - height % window_size) % window_size
        pad_w = (window_size - width % window_size) % window_size
        hidden_states = F.pad(hidden_states, (0, 0, 0, pad_w, 0, pad_h))
        pad_height, pad_width = height + pad_h, width + pad_w
        hidden_states = hidden_states.reshape(batch_size, pad_height // window_size, window_size, pad_width // window_size, window_size, channel)
        windows = hidden_states.permute(0, 1, 3, 2, 4, 5).contiguous().reshape(-1, window_size, window_size, channel)
        return windows, (pad_height, pad_width)

    def window_unpartition(self, windows: 'torch.Tensor', window_size: 'int', padding_shape: 'Tuple[int, int]', original_shape: 'Tuple[int, int]') ->torch.Tensor:
        """
        Args:
        Window unpartition into original sequences and removing padding.
            hidden_states (tensor):
                input tokens with [batch_size * num_windows, window_size, window_size, channel].
            window_size (int):
                window size.
            padding_shape (Tuple):
                padded height and width (pad_height, pad_width).
            original_shape (Tuple): original height and width (height, width) before padding.

        Returns:
            hidden_states: unpartitioned sequences with [batch_size, height, width, channel].
        """
        pad_height, pad_width = padding_shape
        height, width = original_shape
        batch_size = windows.shape[0] // (pad_height * pad_width // window_size // window_size)
        hidden_states = windows.reshape(batch_size, pad_height // window_size, pad_width // window_size, window_size, window_size, -1)
        hidden_states = hidden_states.permute(0, 1, 3, 2, 4, 5).contiguous().reshape(batch_size, pad_height, pad_width, -1)
        hidden_states = hidden_states[:, :height, :width, :].contiguous()
        return hidden_states

    def forward(self, hidden_states: 'torch.Tensor', output_attentions: 'Optional[bool]'=False, output_moe_loss: 'Optional[bool]'=False) ->Tuple[torch.FloatTensor]:
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        if self.window_size > 0:
            height, width = hidden_states.shape[1], hidden_states.shape[2]
            hidden_states, padding_shape = self.window_partition(hidden_states, self.window_size)
        attn_outputs = self.attn(hidden_states=hidden_states, output_attentions=output_attentions, output_moe_loss=output_moe_loss)
        if output_moe_loss:
            hidden_states, attn_weights, moe_loss = attn_outputs
        else:
            hidden_states, attn_weights = attn_outputs
        if self.window_size > 0:
            hidden_states = self.window_unpartition(hidden_states, self.window_size, padding_shape, (height, width))
        hidden_states = residual + hidden_states
        layernorm_output = self.layer_norm2(hidden_states)
        hidden_states = hidden_states + self.mlp(layernorm_output)
        outputs = hidden_states,
        if output_attentions:
            outputs += attn_weights,
        else:
            outputs += None,
        if output_moe_loss:
            outputs += moe_loss,
        return outputs


class SamVisionNeck(nn.Module):

    def __init__(self, config: 'SamVisionConfig'):
        super().__init__()
        self.config = config
        self.conv1 = nn.Conv2d(config.hidden_size, config.output_channels, kernel_size=1, bias=False)
        self.layer_norm1 = SamLayerNorm(config.output_channels, data_format='channels_first')
        self.conv2 = nn.Conv2d(config.output_channels, config.output_channels, kernel_size=3, padding=1, bias=False)
        self.layer_norm2 = SamLayerNorm(config.output_channels, data_format='channels_first')

    def forward(self, hidden_states):
        hidden_states = hidden_states.permute(0, 3, 1, 2)
        hidden_states = self.conv1(hidden_states)
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.conv2(hidden_states)
        hidden_states = self.layer_norm2(hidden_states)
        return hidden_states


def reglu(x: 'Tensor') ->Tensor:
    """The ReGLU activation function from [1].

    References:
    ----------
    [1] Noam Shazeer, "GLU Variants Improve Transformer", 2020
    """
    assert x.shape[-1] % 2 == 0
    a, b = x.chunk(2, dim=-1)
    return a * F.relu(b)


class ReGLU(nn.Module):
    """
    The ReGLU activation function from [1].

    References:
    ----------
    [1] Noam Shazeer, "GLU Variants Improve Transformer", 2020
    """

    def forward(self, x: 'Tensor') ->Tensor:
        return reglu(x)


def geglu(x: 'Tensor') ->Tensor:
    """The GEGLU activation function from [1].

    References:
    ----------
    [1] Noam Shazeer, "GLU Variants Improve Transformer", 2020
    """
    assert x.shape[-1] % 2 == 0
    a, b = x.chunk(2, dim=-1)
    return a * F.gelu(b)


class GEGLU(nn.Module):
    """
    The GEGLU activation function from [1].

    References:
    ----------
    [1] Noam Shazeer, "GLU Variants Improve Transformer", 2020
    """

    def forward(self, x: 'Tensor') ->Tensor:
        return geglu(x)


class _TokenInitialization(enum.Enum):
    UNIFORM = 'uniform'
    NORMAL = 'normal'

    @classmethod
    def from_str(cls, initialization: 'str') ->'_TokenInitialization':
        try:
            return cls(initialization)
        except ValueError:
            valid_values = [x.value for x in _TokenInitialization]
            raise ValueError(f'initialization must be one of {valid_values}')

    def apply(self, x: 'Tensor', d: 'int') ->None:
        d_sqrt_inv = 1 / math.sqrt(d)
        if self == _TokenInitialization.UNIFORM:
            nn.init.uniform_(x, a=-d_sqrt_inv, b=d_sqrt_inv)
        elif self == _TokenInitialization.NORMAL:
            nn.init.normal_(x, std=d_sqrt_inv)


class CLSToken(nn.Module):
    """[CLS]-token for BERT-like inference.

    To learn about the [CLS]-based inference, see [1].

    When used as a module, the [CLS]-token is appended **to the end** of each item in
    the batch.

    References:
    ----------
    [1] Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding" 2018
    """

    def __init__(self, d_token: 'int', initialization: 'str') ->None:
        """
        Args:
            d_token: the size of token
            initialization: initialization policy for parameters. Must be one of
                :code:`['uniform', 'normal']`. Let :code:`s = d ** -0.5`. Then, the
                corresponding distributions are :code:`Uniform(-s, s)` and :code:`Normal(0, s)`. In
                the paper [gorishniy2021revisiting], the 'uniform' initialization was
                used.

        References:
            * [gorishniy2021revisiting] Yury Gorishniy, Ivan Rubachev, Valentin Khrulkov, Artem Babenko "Revisiting Deep Learning Models for Tabular Data", 2021
        """
        super().__init__()
        initialization_ = _TokenInitialization.from_str(initialization)
        self.weight = nn.Parameter(Tensor(d_token))
        initialization_.apply(self.weight, d_token)

    def expand(self, *leading_dimensions: int) ->Tensor:
        """Expand (repeat) the underlying [CLS]-token to a tensor with the given leading dimensions.

        A possible use case is building a batch of [CLS]-tokens. See `_CLSToken` for
        examples of usage.

        Note:
            Under the hood, the `torch.Tensor.expand` method is applied to the
            underlying :code:`weight` parameter, so gradients will be propagated as
            expected.

        Args:
            leading_dimensions: the additional new dimensions

        Returns:
            tensor of the shape :code:`(*leading_dimensions, len(self.weight))`
        """
        if not leading_dimensions:
            return self.weight
        new_dims = (1,) * (len(leading_dimensions) - 1)
        return self.weight.view(*new_dims, -1).expand(*leading_dimensions, -1)

    def forward(self, x: 'Tensor') ->Tensor:
        """Append self **to the end** of each item in the batch (see `_CLSToken`)."""
        return torch.cat([x, self.expand(len(x), 1)], dim=1)


class Linear(Module):
    """Applies a linear transformation to the incoming data: :math:`y = xA^T + b`
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``
    Shape:
        - Input: :math:`(N, *, H_{in})` where :math:`*` means any number of
          additional dimensions and :math:`H_{in} = \\text{in\\_features}`
        - Output: :math:`(N, *, H_{out})` where all but the last dimension
          are the same shape as the input and :math:`H_{out} = \\text{out\\_features}`.
    Attributes:
        weight: the learnable weights of the module of shape
            :math:`(\\text{out\\_features}, \\text{in\\_features})`. The values are
            initialized from :math:`\\mathcal{U}(-\\sqrt{k}, \\sqrt{k})`, where
            :math:`k = \\frac{1}{\\text{in\\_features}}`
        bias:   the learnable bias of the module of shape :math:`(\\text{out\\_features})`.
                If :attr:`bias` is ``True``, the values are initialized from
                :math:`\\mathcal{U}(-\\sqrt{k}, \\sqrt{k})` where
                :math:`k = \\frac{1}{\\text{in\\_features}}`
    Examples::
        >>> m = nn.Linear(20, 30)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 30])
    """
    __constants__ = ['in_features', 'out_features']
    in_features: 'int'
    out_features: 'int'
    weight: 'torch.Tensor'

    def __init__(self, in_features: 'int', out_features: 'int', bias: 'bool'=True) ->None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) ->None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input: 'torch.Tensor') ->torch.Tensor:
        return F.linear(input, self.weight, self.bias)

    def extra_repr(self) ->str:
        return 'in_features={}, out_features={}, bias={}'.format(self.in_features, self.out_features, self.bias is not None)


class _LinearWithBias(Linear):
    bias: 'torch.Tensor'

    def __init__(self, in_features: 'int', out_features: 'int') ->None:
        super().__init__(in_features, out_features, bias=True)


def multi_head_attention_forward(self, query, key, value, embed_dim_to_check, num_heads, in_proj_weight, in_proj_bias, bias_k, bias_v, add_zero_attn, dropout_p, out_proj_weight, out_proj_bias, fixed_k=None, fixed_q=None, training=True, key_padding_mask=None, need_weights=True, attn_mask=None, use_separate_proj_weight=False, q_proj_weight=None, k_proj_weight=None, v_proj_weight=None, static_k=None, static_v=None):
    """
    Args:
        query, key, value: map a query and a set of key-value pairs to an output.
            See "Attention Is All You Need" for more details.
        embed_dim_to_check: total dimension of the model.
        num_heads: parallel attention heads.
        in_proj_weight, in_proj_bias: input projection weight and bias.
        bias_k, bias_v: bias of the key and value sequences to be added at dim=0.
        add_zero_attn: add a new batch of zeros to the key and
                       value sequences at dim=1.
        dropout_p: probability of an element to be zeroed.
        out_proj_weight, out_proj_bias: the output projection weight and bias.
        training: apply dropout if is ``True``.
        key_padding_mask: if provided, specified padding elements in the key will
            be ignored by the attention. This is an binary mask. When the value is True,
            the corresponding value on the attention layer will be filled with -inf.
        need_weights: output attn_output_weights.
        attn_mask: 2D or 3D mask that prevents attention to certain positions. A 2D mask will be broadcasted for all
            the batches while a 3D mask allows to specify a different mask for the entries of each batch.
        use_separate_proj_weight: the function accept the proj. weights for query, key,
            and value in different forms. If false, in_proj_weight will be used, which is
            a combination of q_proj_weight, k_proj_weight, v_proj_weight.
        q_proj_weight, k_proj_weight, v_proj_weight, in_proj_bias: input projection weight and bias.
        static_k, static_v: static key and value used for attention operators.
    Shape:
        Inputs:
        - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
          the embedding dimension.
        - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - key_padding_mask: :math:`(N, S)` where N is the batch size, S is the source sequence length.
          If a ByteTensor is provided, the non-zero positions will be ignored while the zero positions
          will be unchanged. If a BoolTensor is provided, the positions with the
          value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.
        - attn_mask: 2D mask :math:`(L, S)` where L is the target sequence length, S is the source sequence length.
          3D mask :math:`(N*num_heads, L, S)` where N is the batch size, L is the target sequence length,
          S is the source sequence length. attn_mask ensures that position i is allowed to attend the unmasked
          positions. If a ByteTensor is provided, the non-zero positions are not allowed to attend
          while the zero positions will be unchanged. If a BoolTensor is provided, positions with ``True``
          are not allowed to attend while ``False`` values will be unchanged. If a FloatTensor
          is provided, it will be added to the attention weight.
        - static_k: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
          N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.
        - static_v: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
          N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.
        Outputs:
        - attn_output: :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
          E is the embedding dimension.
        - attn_output_weights: :math:`(N, L, S)` where N is the batch size,
          L is the target sequence length, S is the source sequence length.
    """
    if not torch.jit.is_scripting():
        tens_ops = query, key, value, in_proj_weight, in_proj_bias, bias_k, bias_v, out_proj_weight, out_proj_bias
        if any([(type(t) is not torch.Tensor) for t in tens_ops]) and has_torch_function(tens_ops):
            return handle_torch_function(self.multi_head_attention_forward, tens_ops, query, key, value, embed_dim_to_check, num_heads, in_proj_weight, in_proj_bias, bias_k, bias_v, add_zero_attn, dropout_p, out_proj_weight, out_proj_bias, training=training, key_padding_mask=key_padding_mask, need_weights=need_weights, attn_mask=attn_mask, use_separate_proj_weight=use_separate_proj_weight, q_proj_weight=q_proj_weight, k_proj_weight=k_proj_weight, v_proj_weight=v_proj_weight, static_k=static_k, static_v=static_v)
    tgt_len, bsz, embed_dim = query.size()
    assert embed_dim == embed_dim_to_check
    assert key.size(0) == value.size(0) and key.size(1) == value.size(1)
    head_dim = embed_dim // num_heads
    assert head_dim * num_heads == embed_dim, 'embed_dim must be divisible by num_heads'
    scaling = float(head_dim) ** -0.5
    v = linear(query, in_proj_weight, in_proj_bias)
    k = torch.cat([fixed_k.unsqueeze(1) for _ in range(key.shape[1])], dim=1)
    q = torch.cat([fixed_q.unsqueeze(1) for _ in range(key.shape[1])], dim=1)
    q = q * scaling
    q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    src_len = k.size(1)
    attn_output_weights = torch.bmm(q, k.transpose(1, 2))
    assert list(attn_output_weights.size()) == [bsz * num_heads, tgt_len, src_len]
    attn_output_weights = softmax(attn_output_weights, dim=-1)
    attn_output_weights = dropout(attn_output_weights, p=dropout_p, training=training)
    attn_output = torch.bmm(attn_output_weights, v)
    assert list(attn_output.size()) == [bsz * num_heads, tgt_len, head_dim]
    attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
    attn_output = linear(attn_output, out_proj_weight, out_proj_bias)
    if need_weights:
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        return attn_output, attn_output_weights.sum(dim=1) / num_heads
    else:
        return attn_output, None


class MultiheadAttention(Module):
    """Allows the model to jointly attend to information
    from different representation subspaces.
    See reference: Attention Is All You Need

    .. math::
        text{MultiHead}(Q, K, V) = text{Concat}(head_1,dots,head_h)W^O
        text{where} head_i = text{Attention}(QW_i^Q, KW_i^K, VW_i^V)

    Args:
        embed_dim: total dimension of the model.
        num_heads: parallel attention heads.
        dropout: a Dropout layer on attn_output_weights. Default: 0.0.
        bias: add bias as module parameter. Default: True.
        add_bias_kv: add bias to the key and value sequences at dim=0.
        add_zero_attn: add a new batch of zeros to the key and
                       value sequences at dim=1.
        kdim: total number of features in key. Default: None.
        vdim: total number of features in value. Default: None.

        Note: if kdim and vdim are None, they will be set to embed_dim such that
        query, key, and value have the same number of features.

    Examples::

        >>> multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)
        >>> attn_output, attn_output_weights = multihead_attn(query, key, value)
    """
    __annotations__ = {'bias_k': torch._jit_internal.Optional[torch.Tensor], 'bias_v': torch._jit_internal.Optional[torch.Tensor]}

    def __init__(self, embed_dim, n_cat_embeddings, num_heads, dropout=0.0, bias=True, add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None):
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, 'embed_dim must be divisible by num_heads'
        if self._qkv_same_embed_dim is False:
            self.q_proj_weight = Parameter(torch.Tensor(embed_dim, embed_dim))
            self.k_proj_weight = Parameter(torch.Tensor(embed_dim, self.kdim))
            self.v_proj_weight = Parameter(torch.Tensor(embed_dim, self.vdim))
            self.register_parameter('in_proj_weight', None)
            self.register_parameter('fixed_k', None)
        else:
            self.in_proj_weight = Parameter(torch.empty(embed_dim, embed_dim))
            self.fixed_k = Parameter(torch.empty(n_cat_embeddings, embed_dim))
            self.fixed_q = Parameter(torch.empty(n_cat_embeddings, embed_dim))
            self.register_parameter('q_proj_weight', None)
            self.register_parameter('k_proj_weight', None)
            self.register_parameter('v_proj_weight', None)
        if bias:
            self.in_proj_bias = Parameter(torch.empty(embed_dim))
        else:
            self.register_parameter('in_proj_bias', None)
        self.out_proj = _LinearWithBias(embed_dim, embed_dim)
        if add_bias_kv:
            self.bias_k = Parameter(torch.empty(1, 1, embed_dim))
            self.bias_v = Parameter(torch.empty(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None
        self.add_zero_attn = add_zero_attn
        self._reset_parameters()

    def _reset_parameters(self):
        if self._qkv_same_embed_dim:
            xavier_uniform_(self.in_proj_weight)
            xavier_uniform_(self.fixed_k)
            xavier_uniform_(self.fixed_q)
        else:
            xavier_uniform_(self.q_proj_weight)
            xavier_uniform_(self.k_proj_weight)
            xavier_uniform_(self.v_proj_weight)
        if self.in_proj_bias is not None:
            constant_(self.in_proj_bias, 0.0)
            constant_(self.out_proj.bias, 0.0)
        if self.bias_k is not None:
            xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            xavier_normal_(self.bias_v)

    def __setstate__(self, state):
        if '_qkv_same_embed_dim' not in state:
            state['_qkv_same_embed_dim'] = True
        super().__setstate__(state)

    def forward(self, query, key, value, key_padding_mask=None, need_weights=True, attn_mask=None):
        """
        Args:
            query, key, value: map a query and a set of key-value pairs to an output.
                See "Attention Is All You Need" for more details.
            key_padding_mask: if provided, specified padding elements in the key will
                be ignored by the attention. When given a binary mask and a value is True,
                the corresponding value on the attention layer will be ignored. When given
                a byte mask and a value is non-zero, the corresponding value on the attention
                layer will be ignored
            need_weights: output attn_output_weights.
            attn_mask: 2D or 3D mask that prevents attention to certain positions. A 2D mask will be broadcasted for all
                the batches while a 3D mask allows to specify a different mask for the entries of each batch.

        Shape:
            - Inputs:
            - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
              the embedding dimension.
            - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
              the embedding dimension.
            - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
              the embedding dimension.
            - key_padding_mask: :math:`(N, S)` where N is the batch size, S is the source sequence length.
              If a ByteTensor is provided, the non-zero positions will be ignored while the position
              with the zero positions will be unchanged. If a BoolTensor is provided, the positions with the
              value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.
            - attn_mask: 2D mask :math:`(L, S)` where L is the target sequence length, S is the source sequence length.
              3D mask :math:`(N*num_heads, L, S)` where N is the batch size, L is the target sequence length,
              S is the source sequence length. attn_mask ensure that position i is allowed to attend the unmasked
              positions. If a ByteTensor is provided, the non-zero positions are not allowed to attend
              while the zero positions will be unchanged. If a BoolTensor is provided, positions with ``True``
              is not allowed to attend while ``False`` values will be unchanged. If a FloatTensor
              is provided, it will be added to the attention weight.

            - Outputs:
            - attn_output: :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
              E is the embedding dimension.
            - attn_output_weights: :math:`(N, L, S)` where N is the batch size,
              L is the target sequence length, S is the source sequence length.
        """
        return multi_head_attention_forward(self, query=query, key=key, value=value, embed_dim_to_check=self.embed_dim, num_heads=self.num_heads, in_proj_weight=self.in_proj_weight, in_proj_bias=self.in_proj_bias, bias_k=self.bias_k, bias_v=self.bias_v, add_zero_attn=self.add_zero_attn, dropout_p=self.dropout, out_proj_weight=self.out_proj.weight, out_proj_bias=self.out_proj.bias, fixed_k=self.fixed_k, fixed_q=self.fixed_q, training=self.training, key_padding_mask=key_padding_mask, need_weights=need_weights, attn_mask=attn_mask)


class AdditiveAttention(nn.Module):
    """Additive Attention with linear complexity to input sequence length.

    Additive attention was proposed and used in FastFormer.
    See Ref. [1] for details.
    This implementation is motivated by: https://github.com/jrzaurin/pytorch-widedeep.git

    References:
    ----------
    [1] Wu, Chuhan, et al. "Fastformer: Additive attention can be all you need." arXiv preprint arXiv:2108.09084 (2021).
    """

    def __init__(self, *, d_token: int, n_heads: int, dropout: float, bias: bool, share_qv_weights: bool, initialization: str) ->None:
        """
        Parameters
        ----------
        d_token:
            the token size. Must be a multiple of :code:`n_heads`.
        n_heads:
            the number of heads. If greater than 1, then the module will have
            an addition output layer (so called "mixing" layer).
        dropout:
            dropout rate for the attention map. The dropout is applied to
            *probabilities* and do not affect logits.
        bias:
            if `True`, then input (and output, if presented) layers also have bias.
            `True` is a reasonable default choice.
        share_qv_weights:
            if 'True', then value and query transformation parameters are shared.
        initialization:
            initialization for input projection layers. Must be one of
            :code:`['kaiming', 'xavier']`. `kaiming` is a reasonable default choice.
        """
        super().__init__()
        assert d_token % n_heads == 0, 'd_token must be a multiple of n_heads'
        assert initialization in ['kaiming', 'xavier']
        self.head_dim = d_token // n_heads
        self.n_heads = n_heads
        self.share_qv_weights = share_qv_weights
        self.dropout = nn.Dropout(dropout)
        trainable = []
        if share_qv_weights:
            self.qv_proj = nn.Linear(d_token, d_token, bias=bias)
            trainable.extend([self.qv_proj])
        else:
            self.q_proj = nn.Linear(d_token, d_token, bias=bias)
            self.v_proj = nn.Linear(d_token, d_token, bias=bias)
            trainable.extend([self.q_proj, self.v_proj])
        self.k_proj = nn.Linear(d_token, d_token, bias=bias)
        self.W_q = nn.Linear(d_token, n_heads)
        self.W_k = nn.Linear(d_token, n_heads)
        self.r_out = nn.Linear(d_token, d_token)
        trainable.extend([self.k_proj, self.W_q, self.W_k, self.r_out])
        if initialization == 'xavier':
            self.apply(init_weights)
        else:
            for m in trainable:
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x_q: 'Tensor', x_kv: 'Tensor', *args) ->Tuple[Tensor, Dict[str, Tensor]]:
        batch_size, n_q_tokens, d_token = x_q.shape
        batch_size, n_k_tokens, d_token = x_kv.shape
        q = self.qv_proj(x_q) if self.share_qv_weights else self.q_proj(x_q)
        v = self.qv_proj(x_kv) if self.share_qv_weights else self.v_proj(x_kv)
        k = self.k_proj(x_kv)
        alphas = (self.W_q(q) / math.sqrt(self.head_dim)).softmax(dim=1)
        q_r = q.reshape(batch_size, n_q_tokens, self.n_heads, self.head_dim)
        global_query = torch.einsum(' b s h, b s h d -> b h d', alphas, q_r)
        global_query = global_query.reshape(batch_size, self.n_heads * self.head_dim).unsqueeze(1)
        p = k * global_query
        betas = (self.W_k(p) / math.sqrt(self.head_dim)).softmax(dim=1)
        p_r = p.reshape(batch_size, n_k_tokens, self.n_heads, self.head_dim)
        global_key = torch.einsum(' b s h, b s h d -> b h d', betas, p_r)
        global_key = global_key.reshape(batch_size, self.n_heads * self.head_dim).unsqueeze(1)
        u = v * global_key
        output = q + self.dropout(self.r_out(u))
        return output, {'query_weight': alphas, 'key_weight': betas}


_INTERNAL_ERROR_MESSAGE = 'Internal error. Please, open an issue.'


def _all_or_none(values):
    return all(x is None for x in values) or all(x is not None for x in values)


def _is_glu_activation(activation: 'ModuleType'):
    return isinstance(activation, str) and activation.endswith('glu') or activation in [ReGLU, GEGLU]


def _make_nn_module(module_type: 'ModuleType', *args) ->nn.Module:
    if isinstance(module_type, str):
        if module_type == 'reglu':
            return ReGLU()
        elif module_type == 'geglu':
            return GEGLU()
        elif module_type == 'gelu':
            return nn.GELU()
        elif module_type == 'relu':
            return nn.ReLU()
        elif module_type == 'leaky_relu':
            return nn.LeakyReLU()
        elif module_type == 'layer_norm':
            return nn.LayerNorm(*args)
        else:
            try:
                cls = getattr(nn, module_type)
            except AttributeError as err:
                raise ValueError(f'Failed to construct the module {module_type} with the arguments {args}') from err
            return cls(*args)
    else:
        return module_type(*args)


class Custom_Transformer(nn.Module):
    """Transformer with extra features.

    This module is the backbone of `FTTransformer`."""
    WARNINGS = {'first_prenormalization': True, 'prenormalization': True}


    class FFN(nn.Module):
        """The Feed-Forward Network module used in every `Transformer` block."""

        def __init__(self, *, d_token: int, d_hidden: int, bias_first: bool, bias_second: bool, dropout: float, activation: ModuleType):
            super().__init__()
            self.linear_first = nn.Linear(d_token, d_hidden * (2 if _is_glu_activation(activation) else 1), bias_first)
            self.activation = _make_nn_module(activation)
            self.dropout = nn.Dropout(dropout)
            self.linear_second = nn.Linear(d_hidden, d_token, bias_second)

        def forward(self, x: 'Tensor') ->Tensor:
            x = self.linear_first(x)
            x = self.activation(x)
            x = self.dropout(x)
            x = self.linear_second(x)
            return x


    class Head(nn.Module):
        """The final module of the `Transformer` that performs BERT-like inference."""

        def __init__(self, *, d_in: int, bias: bool, activation: ModuleType, normalization: ModuleType, d_out: int):
            super().__init__()
            self.normalization = _make_nn_module(normalization, d_in)
            self.activation = _make_nn_module(activation)
            self.linear = nn.Linear(d_in, d_out, bias)

        def forward(self, x: 'Tensor') ->Tensor:
            x = x[:, -1]
            x = self.normalization(x)
            x = self.activation(x)
            x = self.linear(x)
            return x

    def __init__(self, *, d_token: int, n_blocks: int, attention_n_heads: int, attention_dropout: float, attention_initialization: str, attention_normalization: str, ffn_d_hidden: int, ffn_dropout: float, ffn_activation: str, ffn_normalization: str, residual_dropout: float, prenormalization: bool, first_prenormalization: bool, last_layer_query_idx: Union[None, List[int], slice], n_tokens: Optional[int], kv_compression_ratio: Optional[float], kv_compression_sharing: Optional[str], head_activation: ModuleType, head_normalization: ModuleType, d_out: int, projection: Optional[bool]=False, additive_attention: Optional[bool]=False, share_qv_weights: Optional[bool]=False) ->None:
        """
        Parameters
        ----------
        d_token
            The size of one token for `_CategoricalFeatureTokenizer`.
        n_blocks
            Number of the `FT_Transformer` blocks, which should be non-negative.
        attention_n_heads
            Number of attention heads in each `FT_Transformer` block, which should be positive.
        attention_dropout
            Dropout ratio for the Multi Headed Attention module.
        attention_initialization
            Weights initialization scheme for Multi Headed Attention module.
        attention_normalization
            Normalization policy for attention layers. "layer_norm" is a good default.
        ffn_d_hidden
            Number of the hidden nodes of the linear layers in the Feed-Forward Network module.
        ffn_dropout
            Dropout ratio of the hidden nodes of the linear layers in the Feed-Forward Network module.
        ffn_activation
            Activation function type for the Feed-Forward Network module.
        ffn_normalization
            Normalization scheme of the Feed-Forward Network module.
        residual_dropout
            Dropout ratio for the linear layers in FT_Transformer block.
        prenormalization, first_prenormalization
            Prenormalization to stabilize the training.
        n_tokens
            Number of tokens of the input sequence.
        kv_compression_ratio
            The compression ration to reduce the input sequence length.
        kv_compression_sharing
            If `true` the projections will share weights.
        head_activation
            Activation function type of the MLP layer.
        head_normalization
            Normalization scheme of the MLP layer.
        d_out
            Output dimension.
        projection
            Whether to use a project head.
        additive_attention
            If 'true' the transformer will use additive attention with linear complexity to sequence length.
        share_qv_weights
            if 'true', then value and query transformation parameters are shared in additive attention.
        """
        super().__init__()
        if isinstance(last_layer_query_idx, int):
            raise ValueError(f'last_layer_query_idx must be None, list[int] or slice. Do you mean last_layer_query_idx=[{last_layer_query_idx}] ?')
        if not prenormalization:
            assert not first_prenormalization, 'If `prenormalization` is False, then `first_prenormalization` must be False'
        assert _all_or_none([n_tokens, kv_compression_ratio, kv_compression_sharing]), 'If any of the following arguments is (not) None, then all of them must (not) be None: n_tokens, kv_compression_ratio, kv_compression_sharing'
        assert additive_attention or not share_qv_weights, 'If `share_qv_weights` is True, then `additive_attention` must be True'
        assert kv_compression_sharing in [None, 'headwise', 'key-value', 'layerwise']
        if not prenormalization:
            if self.WARNINGS['prenormalization']:
                warnings.warn('prenormalization is set to False. Are you sure about this? The training can become less stable. You can turn off this warning by tweaking the rtdl.Transformer.WARNINGS dictionary.', UserWarning)
            assert not first_prenormalization, 'If prenormalization is False, then first_prenormalization is ignored and must be set to False'
        if prenormalization and first_prenormalization and self.WARNINGS['first_prenormalization']:
            warnings.warn('first_prenormalization is set to True. Are you sure about this? For example, the vanilla FTTransformer with first_prenormalization=True performs SIGNIFICANTLY worse. You can turn off this warning by tweaking the rtdl.Transformer.WARNINGS dictionary.', UserWarning)

        def make_kv_compression():
            assert n_tokens and kv_compression_ratio, _INTERNAL_ERROR_MESSAGE
            return nn.Linear(n_tokens, int(n_tokens * kv_compression_ratio), bias=False)
        self.shared_kv_compression = make_kv_compression() if kv_compression_ratio and kv_compression_sharing == 'layerwise' else None
        self.prenormalization = prenormalization
        self.last_layer_query_idx = last_layer_query_idx
        self.blocks = nn.ModuleList([])
        for layer_idx in range(n_blocks):
            layer = nn.ModuleDict({'attention': AdditiveAttention(d_token=d_token, n_heads=attention_n_heads, dropout=attention_dropout, bias=True, share_qv_weights=share_qv_weights, initialization=attention_initialization) if additive_attention else MultiheadAttention(d_token=d_token, n_heads=attention_n_heads, dropout=attention_dropout, bias=True, initialization=attention_initialization), 'ffn': Custom_Transformer.FFN(d_token=d_token, d_hidden=ffn_d_hidden, bias_first=True, bias_second=True, dropout=ffn_dropout, activation=ffn_activation), 'attention_residual_dropout': nn.Dropout(residual_dropout), 'ffn_residual_dropout': nn.Dropout(residual_dropout), 'output': nn.Identity()})
            if layer_idx or not prenormalization or first_prenormalization:
                layer['attention_normalization'] = _make_nn_module(attention_normalization, d_token)
            layer['ffn_normalization'] = _make_nn_module(ffn_normalization, d_token)
            if kv_compression_ratio and self.shared_kv_compression is None:
                layer['key_compression'] = make_kv_compression()
                if kv_compression_sharing == 'headwise':
                    layer['value_compression'] = make_kv_compression()
                else:
                    assert kv_compression_sharing == 'key-value', _INTERNAL_ERROR_MESSAGE
            self.blocks.append(layer)
        self.head = Custom_Transformer.Head(d_in=d_token, d_out=d_out, bias=True, activation=head_activation, normalization=head_normalization if prenormalization else 'Identity') if projection else nn.Identity()

    def _get_kv_compressions(self, layer):
        return (self.shared_kv_compression, self.shared_kv_compression) if self.shared_kv_compression is not None else (layer['key_compression'], layer['value_compression']) if 'key_compression' in layer and 'value_compression' in layer else (layer['key_compression'], layer['key_compression']) if 'key_compression' in layer else (None, None)

    def _start_residual(self, layer, stage, x):
        assert stage in ['attention', 'ffn'], _INTERNAL_ERROR_MESSAGE
        x_residual = x
        if self.prenormalization:
            norm_key = f'{stage}_normalization'
            if norm_key in layer:
                x_residual = layer[norm_key](x_residual)
        return x_residual

    def _end_residual(self, layer, stage, x, x_residual):
        assert stage in ['attention', 'ffn'], _INTERNAL_ERROR_MESSAGE
        x_residual = layer[f'{stage}_residual_dropout'](x_residual)
        x = x + x_residual
        if not self.prenormalization:
            x = layer[f'{stage}_normalization'](x)
        return x

    def forward(self, x: 'Tensor') ->Tensor:
        assert x.ndim == 3, 'The input must have 3 dimensions: (n_objects, n_tokens, d_token)'
        for layer_idx, layer in enumerate(self.blocks):
            layer = cast(nn.ModuleDict, layer)
            query_idx = self.last_layer_query_idx if layer_idx + 1 == len(self.blocks) else None
            x_residual = self._start_residual(layer, 'attention', x)
            x_residual, _ = layer['attention'](x_residual if query_idx is None else x_residual[:, query_idx], x_residual, *self._get_kv_compressions(layer))
            if query_idx is not None:
                x = x[:, query_idx]
            x = self._end_residual(layer, 'attention', x, x_residual)
            x_residual = self._start_residual(layer, 'ffn', x)
            x_residual = layer['ffn'](x_residual)
            x = self._end_residual(layer, 'ffn', x, x_residual)
            x = layer['output'](x)
        x = self.head(x)
        return x


class CategoricalFeatureTokenizer(nn.Module):
    """
    Feature tokenizer for categorical features in tabular data.
    It transforms the input categorical features to tokens (embeddings).

    The categorical features usually refers to discrete features.
    """

    def __init__(self, num_categories: 'List[int]', d_token: 'int', bias: 'Optional[bool]'=True, initialization: 'Optional[str]'='normal') ->None:
        """
        Parameters
        ----------
        num_categories:
            A list of integers. Each one is the number of categories in one categorical column.
        d_token:
            The size of one token.
        bias:
            If `True`, for each feature, an additional trainable vector will be added to the
            embedding regardless of feature value. Notablly, the bias are not shared between features.
        initialization:
            Initialization policy for parameters. Must be one of `['uniform', 'normal']`.

        References
        ----------
        1. Yury Gorishniy, Ivan Rubachev, Valentin Khrulkov, Artem Babenko,
        "Revisiting Deep Learning Models for Tabular Data", NeurIPS 2021
        https://arxiv.org/pdf/2106.11959.pdf
        2. Code: https://github.com/Yura52/tabular-dl-revisiting-models
        """
        super().__init__()
        self.num_categories = num_categories
        category_offsets = torch.tensor([0] + num_categories[:-1]).cumsum(0)
        self.register_buffer('category_offsets', category_offsets, persistent=False)
        self.embeddings = nn.Embedding(sum(num_categories), d_token)
        self.bias = nn.Parameter(Tensor(len(num_categories), d_token)) if bias else None
        initialization_ = _TokenInitialization.from_str(initialization)
        for parameter in [self.embeddings.weight, self.bias]:
            if parameter is not None:
                initialization_.apply(parameter, d_token)

    @property
    def n_tokens(self) ->int:
        """The number of tokens."""
        return len(self.num_categories)

    @property
    def d_token(self) ->int:
        """The size of one token."""
        return self.embeddings.embedding_dim

    def forward(self, x: 'Tensor') ->Tensor:
        x = self.embeddings(x + self.category_offsets[None])
        if self.bias is not None:
            x = x + self.bias[None]
        return x


class Periodic(nn.Module):

    def __init__(self, in_features: 'int', d_embedding: 'int', trainable: 'Optional[bool]'=True, initialization: 'Optional[str]'='normal', sigma: 'Optional[float]'=1.0):
        """
        Parameters
        ----------
        in_features
            Input feature size.
        d_embedding
            Output feature size, should be an even number.
        trainable
            Determine whether the coefficients needed to be updated.
        initialization
            Initialization scheme.
        sigma
            Standard deviation used for initialization='normal'

        Reference:
        ----------
        1. Code: https://github.com/Yura52/tabular-dl-num-embeddings
        2. Paper: On Embeddings for Numerical Features in Tabular Deep Learning, https://arxiv.org/abs/2203.05556
        """
        super().__init__()
        assert d_embedding % 2 == 0, 'd_embedding mod 2 should be 0, current d_embedding is {}'.format(d_embedding)
        if initialization == 'log-linear':
            coefficients = sigma ** (torch.arange(d_embedding // 2) / (d_embedding // 2))
            coefficients = coefficients[None].repeat(in_features, 1)
        elif initialization == 'normal':
            coefficients = torch.normal(0.0, sigma, (in_features, d_embedding // 2))
        if trainable:
            self.coefficients = nn.Parameter(coefficients)
        else:
            self.register_buffer('coefficients', coefficients)

    def cos_sin(self, x: 'Tensor'):
        return torch.cat([torch.cos(x), torch.sin(x)], -1)

    def forward(self, x: 'Tensor'):
        assert x.ndim == 2, 'Periodic should only be applied to first layer i.e. ndim==2'
        return self.cos_sin(2 * torch.pi * self.coefficients[None] * x[..., None])


class NLinear(nn.Module):

    def __init__(self, n: 'int', d_in: 'int', d_out: 'int', bias: 'bool'=True):
        super().__init__()
        self.weight = nn.Parameter(Tensor(n, d_in, d_out))
        self.bias = nn.Parameter(Tensor(n, d_out)) if bias else None
        with torch.no_grad():
            for i in range(n):
                layer = nn.Linear(d_in, d_out)
                self.weight[i] = layer.weight.T
                if self.bias is not None:
                    self.bias[i] = layer.bias

    def forward(self, x):
        assert x.ndim == 3, 'Error input dimension, should be 3, but given {}'.format(x.ndim)
        x = x[..., None] * self.weight[None]
        x = x.sum(-2)
        if self.bias is not None:
            x = x + self.bias[None]
        return x


class NLinearMemoryEfficient(nn.Module):

    def __init__(self, n: 'int', d_in: 'int', d_out: 'int'):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(d_in, d_out) for _ in range(n)])

    def forward(self, x):
        return torch.stack([l(x[:, i]) for i, l in enumerate(self.layers)], 1)


class NLayerNorm(nn.Module):

    def __init__(self, n_features: 'int', d: 'int'):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(n_features, d))
        self.bias = nn.Parameter(torch.zeros(n_features, d))

    def forward(self, x: 'Tensor'):
        assert x.ndim == 3
        x = (x - x.mean(-1, keepdim=True)) / x.std(-1, keepdim=True)
        x = self.weight * x + self.bias
        return x


class NumericalFeatureTokenizer(nn.Module):
    """
    Numerical tokenizer for numerical features in tabular data.
    It transforms the input numerical features to tokens (embeddings).

    The numerical features usually refers to continuous features.

    It consists of two steps:
        1. each feature is multiplied by a trainable vector i.e., weights,
        2. another trainable vector is added i.e., bias.

    Note that each feature has its separate pair of trainable vectors,
    i.e. the vectors are not shared between features.
    """

    def __init__(self, in_features: 'int', d_token: 'int', bias: 'Optional[bool]'=True, initialization: 'Optional[str]'='normal'):
        """
        Parameters
        ----------
        in_features:
            Dimension of input features i.e. the number of continuous (scalar) features
        d_token:
            The size of one token.
        bias:
            If `True`, for each feature, an additional trainable vector will be added to the
            embedding regardless of feature value. Notablly, the bias are not shared between features.
        initialization:
            Initialization policy for parameters. Must be one of `['uniform', 'normal']`.

        References
        ----------
        Yury Gorishniy, Ivan Rubachev, Valentin Khrulkov, Artem Babenko,
        "Revisiting Deep Learning Models for Tabular Data", 2021
        https://arxiv.org/pdf/2106.11959.pdf
        """
        super().__init__()
        initialization_ = _TokenInitialization.from_str(initialization)
        self.weight = nn.Parameter(Tensor(in_features, d_token))
        self.bias = nn.Parameter(Tensor(in_features, d_token)) if bias else None
        for parameter in [self.weight, self.bias]:
            if parameter is not None:
                initialization_.apply(parameter, d_token)

    @property
    def n_tokens(self) ->int:
        """The number of tokens."""
        return len(self.weight)

    @property
    def d_token(self) ->int:
        """The size of one token."""
        return self.weight.shape[1]

    def forward(self, x: 'Tensor') ->Tensor:
        x = self.weight[None] * x[..., None]
        if self.bias is not None:
            x = x + self.bias[None]
        return x


class AutoDis(nn.Module):
    """
    Paper (the version is important): https://arxiv.org/abs/2012.08986v2
    Code: https://github.com/mindspore-ai/models/tree/bdf2d8bcf11fe28e4ad3060cf2ddc818eacd8597/research/recommend/autodis
    We borrow the implementations from: https://github.com/Yura52/tabular-dl-num-embeddings/blob/main/bin/train4.py
    The paper is significantly different from the code (it looks like the code
    implements the first version of the paper). We implement the second version
    here. Not all technical details are given for the second version, so what we do
    here can be different from what authors actually did.
    Anyway, AutoDis (v2) is essentially the following sequence of layers (applied from
    left to right): [Linear(no bias), LeakyReLU, Linear(no bias), Softmax, Linear]
    """

    def __init__(self, in_features: 'int', d_embedding: 'int', n_meta_embeddings: 'int', temperature: 'Optional[float]'=3.0):
        super().__init__()
        self.first_layer = NumericalFeatureTokenizer(in_features=in_features, d_token=n_meta_embeddings, bias=False, initialization='uniform')
        self.leaky_relu = nn.LeakyReLU()
        self.second_layer = NLinear(in_features, n_meta_embeddings, n_meta_embeddings, False)
        self.softmax = nn.Softmax(-1)
        self.temperature = temperature
        self.third_layer = NLinear(in_features, n_meta_embeddings, d_embedding, False)
        nn.init.uniform_(self.third_layer.weight, 0.01)

    def forward(self, x: 'Tensor'):
        x = self.first_layer(x)
        x = self.leaky_relu(x)
        x = self.second_layer(x)
        x = self.softmax(x / self.temperature)
        x = self.third_layer(x)
        return x


class NumEmbeddings(nn.Module):

    def __init__(self, in_features: 'int', embedding_arch: 'List[str]', d_embedding: 'Optional[int]'=None, memory_efficient: 'Optional[bool]'=False):
        """
        Parameters
        ----------
        in_features
            Input feature size.
        embedding_arch
            A list containing the names of embedding layers.
            Currently support:
                {'linear', 'shared_linear', 'autodis', 'positional', 'relu', 'leaky_relu', 'layernorm'}
            To use the embedding schemes summarized in Table 3 of 'On Embeddings for Numerical Features in Tabular Deep Learning' (https://arxiv.org/abs/2203.05556)
            By setting the embedding_arch as follows:
                1. `L`: ['linear']
                2. `LR`: ['linear', 'relu']
                3. `LRLR`: ['linear', 'relu', 'linear', 'relu']
                4. `P`: ['positional']
                5. `PL`: ['positional', 'linear']
                6. `PLR`: ['positional', 'linear', 'relu']
                7. `PLRLR`: ['positional', 'linear', 'relu', 'linear', 'relu']
                8. `AutoDis`: ['autodis']
                9. `Leaky Gates` in [ref.3]: ['linear', 'leaky_relu']
            Notably, in `L` (i.e. embedding_arch=['linear']) for numerical transformer,
            it identical as the original feature_tokenzier in FT_Transformer (c.f. Figure 2.a in https://arxiv.org/pdf/2106.11959.pdf).
        d_embedding:
            Dimension of the embeddings.
            The output shape should be [batch_size, number_of_numerical_featurs, d_embedding]
        memory_efficient:
            Use efficient linear layer scheme if True. Default is False.

        Reference:
        ----------
        1. Code: https://github.com/Yura52/tabular-dl-num-embeddings
        2. Paper: On Embeddings for Numerical Features in Tabular Deep Learning, https://arxiv.org/abs/2203.05556
        3. Paper: Simple Modifications to Improve Tabular Neural Networks: https://arxiv.org/pdf/2108.03214
        """
        super().__init__()
        assert embedding_arch
        assert set(embedding_arch) <= {'linear', 'shared_linear', 'autodis', 'positional', 'relu', 'leaky_relu', 'layernorm'}
        if any(x in embedding_arch for x in ['linear', 'shared_linear', 'autodis']):
            assert d_embedding is not None
        assert embedding_arch.count('positional') <= 1
        if 'autodis' in embedding_arch:
            embedding_arch = ['autodis']
        NLinear_ = NLinearMemoryEfficient if memory_efficient else NLinear
        layers: 'list[nn.Module]' = []
        if embedding_arch[0] == 'linear':
            layers.append(NumericalFeatureTokenizer(in_features=in_features, d_token=d_embedding, bias=True, initialization='normal'))
        elif embedding_arch[0] == 'positional':
            layers.append(Periodic(in_features=in_features, d_embedding=d_embedding, trainable=True, initialization='normal', sigma=1.0))
        elif embedding_arch[0] == 'autodis':
            layers.append(AutoDis(in_features=in_features, d_embedding=d_embedding, n_meta_embeddings=d_embedding, temperature=3.0))
        else:
            layers.append(nn.Identity())
        for x in embedding_arch[1:]:
            layers.append(nn.ReLU() if x == 'relu' else nn.LeakyReLU() if x == 'leaky_relu' else NLinear_(in_features, d_embedding, d_embedding) if x == 'linear' else nn.Linear(d_embedding, d_embedding) if x == 'shared_linear' else NLayerNorm(in_features, d_embedding) if x == 'layernorm' else nn.Identity())
            assert not isinstance(layers[-1], nn.Identity)
        self.d_embedding = d_embedding
        self.in_features = in_features
        self.layers = nn.Sequential(*layers)

    @property
    def n_tokens(self) ->int:
        """The number of tokens."""
        y = self.forward(torch.ones(1, self.in_features))
        return y.shape[1]

    @property
    def d_token(self) ->int:
        """The size of one token."""
        y = self.forward(torch.ones(1, self.in_features))
        return y.shape[-1]

    def forward(self, x):
        return self.layers(x)


NUMERICAL = 'numerical'


S3_PREFIX = 's3://'


def sha1sum(filename):
    """Calculate the sha1sum of a file
    Parameters
    ----------
    filename
        Name of the file
    Returns
    -------
    ret
        The sha1sum
    """
    with open(filename, mode='rb') as f:
        d = hashlib.sha1()
        for buf in iter(functools.partial(f.read, 1024 * 100), b''):
            d.update(buf)
    return d.hexdigest()


def download(url: 'str', path: 'Optional[str]'=None, overwrite: 'Optional[bool]'=False, sha1_hash: 'Optional[str]'=None, retries: 'int'=5, verify_ssl: 'Optional[bool]'=True) ->str:
    """Download a given URL

    Parameters
    ----------
    url
        URL to download
    path
        Destination path to store downloaded file. By default stores to the
        current directory with same name as in url.
    overwrite
        Whether to overwrite destination file if already exists.
    sha1_hash
        Expected sha1 hash in hexadecimal digits. Will ignore existing file when hash is specified
        but doesn't match.
    retries
        The number of times to attempt the download in case of failure or non 200 return codes
    verify_ssl
        Verify SSL certificates.
    Returns
    -------
    fname
        The file path of the downloaded file.
    """
    is_s3 = url.startswith(S3_PREFIX)
    if is_s3:
        s3 = boto3.resource('s3')
        if boto3.session.Session().get_credentials() is None:
            s3.meta.client.meta.events.register('choose-signer.s3.*', disable_signing)
        components = url[len(S3_PREFIX):].split('/')
        if len(components) < 2:
            raise ValueError('Invalid S3 url. Received url={}'.format(url))
        s3_bucket_name = components[0]
        s3_key = '/'.join(components[1:])
    if path is None:
        fname = url.split('/')[-1]
        assert fname, "Can't construct file-name from this URL. Please set the `path` option manually."
    else:
        path = os.path.expanduser(path)
        if os.path.isdir(path):
            fname = os.path.join(path, url.split('/')[-1])
        else:
            fname = path
    assert retries >= 0, "Number of retries should be at least 0, currently it's {}".format(retries)
    if not verify_ssl:
        warnings.warn('Unverified HTTPS request is being made (verify_ssl=False). Adding certificate verification is strongly advised.')
    if overwrite or not os.path.exists(fname) or sha1_hash and not sha1sum(fname) == sha1_hash:
        dirname = os.path.dirname(os.path.abspath(os.path.expanduser(fname)))
        if not os.path.exists(dirname):
            os.makedirs(dirname, exist_ok=True)
        while retries + 1 > 0:
            try:
                None
                if is_s3:
                    response = s3.meta.client.head_object(Bucket=s3_bucket_name, Key=s3_key)
                    total_size = int(response.get('ContentLength', 0))
                    random_uuid = str(uuid.uuid4())
                    tmp_path = '{}.{}'.format(fname, random_uuid)
                    if tqdm is not None:

                        def hook(t_obj):

                            def inner(bytes_amount):
                                t_obj.update(bytes_amount)
                            return inner
                        with tqdm.tqdm(total=total_size, unit='iB', unit_scale=True) as t:
                            s3.meta.client.download_file(s3_bucket_name, s3_key, tmp_path, Callback=hook(t))
                    else:
                        s3.meta.client.download_file(s3_bucket_name, s3_key, tmp_path)
                else:
                    r = requests.get(url, stream=True, verify=verify_ssl)
                    if r.status_code != 200:
                        raise RuntimeError('Failed downloading url {}'.format(url))
                    random_uuid = str(uuid.uuid4())
                    total_size = int(r.headers.get('content-length', 0))
                    chunk_size = 1024
                    if tqdm is not None:
                        t = tqdm.tqdm(total=total_size, unit='iB', unit_scale=True)
                    with open('{}.{}'.format(fname, random_uuid), 'wb') as f:
                        for chunk in r.iter_content(chunk_size=chunk_size):
                            if chunk:
                                if tqdm is not None:
                                    t.update(len(chunk))
                                f.write(chunk)
                    if tqdm is not None:
                        t.close()
                if not os.path.exists(fname) or sha1_hash and not sha1sum(fname) == sha1_hash:
                    replace_file('{}.{}'.format(fname, random_uuid), fname)
                else:
                    try:
                        os.remove('{}.{}'.format(fname, random_uuid))
                    except OSError:
                        pass
                    finally:
                        warnings.warn('File {} exists in file system so the downloaded file is deleted'.format(fname))
                if sha1_hash and not sha1sum(fname) == sha1_hash:
                    raise UserWarning('File {} is downloaded but the content hash does not match. The repo may be outdated or download may be incomplete. If the "repo_url" is overridden, consider switching to the default repo.'.format(fname))
                break
            except Exception as e:
                retries -= 1
                if retries <= 0:
                    raise e
                None
    return fname


class FT_Transformer(nn.Module):
    """
    FT-Transformer for categorical tabular features.
    The input dimension is automatically computed based on
    the number of categories in each categorical column.
    """

    def __init__(self, prefix: 'str', num_numerical_columns: 'int', num_categories: 'List[int]', embedding_arch: 'List[str]', token_dim: 'int', hidden_size: 'Optional[int]'=192, hidden_features: 'Optional[int]'=192, num_classes: 'Optional[int]'=0, token_bias: 'Optional[bool]'=True, token_initialization: 'Optional[str]'='normal', num_blocks: 'Optional[int]'=0, attention_n_heads: 'Optional[int]'=8, attention_initialization: 'Optional[str]'='kaiming', attention_normalization: 'Optional[str]'='layer_norm', attention_dropout: 'Optional[str]'=0.2, residual_dropout: 'Optional[str]'=0.0, ffn_activation: 'Optional[str]'='reglu', ffn_normalization: 'Optional[str]'='layer_norm', ffn_hidden_size: 'Optional[str]'=6, ffn_dropout: 'Optional[str]'=0.0, prenormalization: 'Optional[bool]'=True, first_prenormalization: 'Optional[bool]'=False, kv_compression_ratio: 'Optional[float]'=None, kv_compression_sharing: 'Optional[str]'=None, head_activation: 'Optional[str]'='relu', head_normalization: 'Optional[str]'='layer_norm', additive_attention: 'Optional[bool]'=False, share_qv_weights: 'Optional[bool]'=False, pooling_mode: 'Optional[str]'='cls', checkpoint_name: 'str'=None, pretrained: 'bool'=False) ->None:
        """
        Parameters
        ----------
        prefix
            The model prefix.
        num_categories
            A list of integers. Each one is the number of categories in one categorical column.
        token_dim
            The size of one token after categorical/numerical tokenizers.
        hidden_size
            The embedding dimension of the transformer backbone.
        out_features
            Dimension of output features.
        num_classes
            Number of classes. 1 for a regression task.
        token_bias
            If `True`, for each feature, an additional trainable vector will be added in `_CategoricalFeatureTokenizer`
            to the embedding regardless of feature value. Notably, the bias are not shared between features.
        token_initialization
            Initialization policy for parameters in `_CategoricalFeatureTokenizer` and `_CLSToke`.
            Must be one of `['uniform', 'normal']`.
        num_blocks
            Number of the `FT_Transformer` blocks, which should be non-negative.
        attention_n_heads
            Number of attention heads in each `FT_Transformer` block, which should be positive.
        attention_initialization
            Weights initialization scheme for Multi Headed Attention module.
        attention_dropout
            Dropout ratio for the Multi Headed Attention module.
        residual_dropout
            Dropout ratio for the linear layers in FT_Transformer block.
        ffn_activation
            Activation function type for the Feed-Forward Network module.
        ffn_normalization
            Normalization scheme of the Feed-Forward Network module.
        ffn_hidden_size
            Number of the hidden nodes of the linear layers in the Feed-Forward Network module.
        ffn_dropout
            Dropout ratio of the hidden nodes of the linear layers in the Feed-Forward Network module.
        prenormalization, first_prenormalization
            Prenormalization to stabilize the training.
        kv_compression_ratio
            The compression ration to reduce the input sequence length.
        kv_compression_sharing
            If `true` the projections will share weights.
        head_activation
            Activation function type of the MLP layer.
        head_normalization
            Normalization scheme of the MLP layer.
        additive_attention
            If 'true' the transformer will use additive attention with linear complexity to sequence length.
        share_qv_weights
            if 'true', then value and query transformation parameters are shared in additive attention.
        pooling_mode
            The pooling mode for the Transformer. Can be "cls", or "mean"

        References
        ----------
        1. Yury Gorishniy, Ivan Rubachev, Valentin Khrulkov, Artem Babenko,
        "Revisiting Deep Learning Models for Tabular Data", 2021
        https://arxiv.org/pdf/2106.11959.pdf
        2. Code: https://github.com/Yura52/tabular-dl-revisiting-models
        """
        super().__init__()
        assert num_categories or num_numerical_columns > 0, 'there must be categorical columns or numerical columns'
        assert token_dim > 0, 'd_token must be positive'
        assert num_blocks >= 0, 'n_blocks must be non-negative'
        assert attention_n_heads > 0, 'attention_n_heads must be positive'
        assert token_initialization in ['uniform', 'normal'], 'initialization must be uniform or normal'
        self.prefix = prefix
        self.out_features = hidden_size
        self.pooling_mode = pooling_mode
        self.categorical_feature_tokenizer = None
        self.numerical_feature_tokenizer = None
        if num_categories:
            self.num_categories = num_categories
            self.categorical_feature_tokenizer = CategoricalFeatureTokenizer(num_categories=num_categories, d_token=token_dim, bias=token_bias, initialization=token_initialization)
            self.categorical_adapter = nn.Linear(token_dim, hidden_size)
        if num_numerical_columns > 0:
            self.numerical_feature_tokenizer = NumEmbeddings(in_features=num_numerical_columns, d_embedding=token_dim, embedding_arch=embedding_arch)
            self.numerical_adapter = nn.Linear(token_dim, hidden_size)
        self.transformer = Custom_Transformer(d_token=hidden_size, n_blocks=num_blocks, attention_n_heads=attention_n_heads, attention_dropout=attention_dropout, attention_initialization=attention_initialization, attention_normalization=attention_normalization, ffn_d_hidden=ffn_hidden_size, ffn_dropout=ffn_dropout, ffn_activation=ffn_activation, ffn_normalization=ffn_normalization, residual_dropout=residual_dropout, prenormalization=prenormalization, first_prenormalization=first_prenormalization, last_layer_query_idx=None, n_tokens=None, kv_compression_ratio=kv_compression_ratio, kv_compression_sharing=kv_compression_sharing, head_activation=head_activation, head_normalization=head_normalization, d_out=hidden_features, projection=False, additive_attention=additive_attention, share_qv_weights=share_qv_weights)
        self.head = Custom_Transformer.Head(d_in=hidden_size, d_out=num_classes, bias=True, activation=head_activation, normalization=head_normalization)
        self.cls_token = CLSToken(d_token=hidden_size, initialization='uniform')
        if self.numerical_feature_tokenizer:
            self.numerical_adapter.apply(init_weights)
        if self.categorical_feature_tokenizer:
            self.categorical_adapter.apply(init_weights)
        self.head.apply(init_weights)
        if pretrained and checkpoint_name:
            if os.path.exists(checkpoint_name):
                ckpt = torch.load(checkpoint_name)
            else:
                with tempfile.TemporaryDirectory() as tmpdirname:
                    checkpoint_path = os.path.join(tmpdirname, './ft_transformer_pretrained.ckpt')
                    download(checkpoint_name, checkpoint_path)
                    ckpt = torch.load(checkpoint_path)
            self.transformer.load_state_dict(ckpt['state_dict'])
        self.name_to_id = self.get_layer_ids()

    @property
    def categorical_key(self):
        return f'{self.prefix}_{CATEGORICAL}'

    @property
    def numerical_key(self):
        return f'{self.prefix}_{NUMERICAL}'

    @property
    def input_keys(self):
        input_keys = []
        if self.categorical_feature_tokenizer:
            input_keys.append(self.categorical_key)
        if self.numerical_feature_tokenizer:
            input_keys.append(self.numerical_key)
        return input_keys

    @property
    def label_key(self):
        return f'{self.prefix}_{LABEL}'

    def forward(self, batch: 'dict'):
        """

        Parameters
        ----------
        batch
            A dictionary containing the input mini-batch data.
            We need to use the keys with the model prefix to index required data.

        Returns
        -------
            A dictionary with logits and features.
        """
        multimodal_features = []
        if self.categorical_feature_tokenizer:
            categorical_inputs = []
            for categorical_input in batch[self.categorical_key]:
                categorical_inputs.append(categorical_input)
            categorical_inputs = torch.stack(categorical_inputs, dim=1)
            categorical_features = self.categorical_feature_tokenizer(categorical_inputs)
            categorical_features = self.categorical_adapter(categorical_features)
            multimodal_features.append(categorical_features)
        if self.numerical_feature_tokenizer:
            numerical_features = self.numerical_feature_tokenizer(batch[self.numerical_key])
            numerical_features = self.numerical_adapter(numerical_features)
            multimodal_features.append(numerical_features)
        multimodal_features = torch.cat(multimodal_features, dim=1)
        multimodal_features = self.cls_token(multimodal_features)
        features = self.transformer(multimodal_features)
        logits = self.head(features)
        if self.pooling_mode == 'cls':
            features = features[:, -1, :]
        elif self.pooling_mode == 'mean':
            features = features.mean(dim=1)
        else:
            raise NotImplementedError(f'Pooling mode={self.pooling_mode} is not supported.')
        output = {self.prefix: {LOGITS: logits, FEATURES: features}}
        return output

    def get_layer_ids(self):
        """
        All layers have the same id 0 since there is no pre-trained models used here.

        Returns
        -------
        A dictionary mapping the layer names (keys) to their ids (values).
        """
        name_to_id = {}
        for n, _ in self.named_parameters():
            name_to_id[n] = 0
        return name_to_id


class AbstractMultimodalFusionModel(ABC, nn.Module):
    """
    An abstract class to fuse different models' features (single-modal and multimodal).
    """

    def __init__(self, prefix: 'str', models: 'list', loss_weight: 'Optional[float]'=None):
        super().__init__()
        self.prefix = prefix
        self.loss_weight = loss_weight
        self.model = nn.ModuleList(models)

    @property
    @abstractmethod
    def label_key(self):
        pass

    @abstractmethod
    def forward(self, *args, **kwargs):
        pass

    def get_layer_ids(self):
        """
        Assign an id to each layer. Layer ids will be used in layer-wise lr decay.
        Basically, id gradually increases when going from the output end to
        the input end.

        It assumes that each individual model has the "name_to_id" attribute storing
        the already computed model's layer ids. This function only collects those layer ids.
        It also add prefixes for each model's parameter names since the fusion model wraps
        those individual models, making the name scope changed. Configuring the optimizer
        requires a full name of each parameter.

        The layers defined in this class, e.g., head, adapter,
        and, fusion_mlp, have id 0.

        Returns
        -------
        A dictionary mapping the layer names (keys) to their ids (values).
        """
        model_prefix = 'model'
        names = [n for n, _ in self.named_parameters()]
        outer_layer_names = [n for n in names if not n.startswith(model_prefix)]
        name_to_id = {}
        logger.debug(f'outer layers are treated as head: {outer_layer_names}')
        for n in outer_layer_names:
            name_to_id[n] = 0
        for i, per_model in enumerate(self.model):
            per_model_prefix = f'{model_prefix}.{i}'
            if not hasattr(per_model, 'name_to_id'):
                raise ValueError(f'name_to_id attribute is missing in model: {per_model.__class__.__name__}')
            for n, layer_id in per_model.name_to_id.items():
                full_n = f'{per_model_prefix}.{n}'
                name_to_id[full_n] = layer_id
        for n in names:
            assert n in name_to_id
        return name_to_id


WEIGHT = 'weight'


ATTENTION_MASK = 'attention_mask'


BBOX = 'bbox'


class DummyLayer(nn.Module):
    """
    DummyLayer to ensure that the gradient checkpointing will assign output layer as require_grad=True.
    Reference: https://discuss.pytorch.org/t/checkpoint-with-no-grad-requiring-inputs-problem/19117/9
    """

    def __init__(self):
        super().__init__()
        self.dummy_bias = torch.ones(1, dtype=torch.float32, requires_grad=True)

    def forward(self, x):
        return x + self.dummy_bias - self.dummy_bias


TEXT_SEGMENT_IDS = 'text_segment_ids'


class HFAutoModelForTextPrediction(nn.Module):
    """
    Support huggingface text backbones.
    Refer to https://github.com/huggingface/transformers
    """

    def __init__(self, prefix: 'str', checkpoint_name: 'str'='microsoft/deberta-v3-base', num_classes: 'Optional[int]'=0, pooling_mode: 'Optional[str]'='cls', gradient_checkpointing: 'Optional[bool]'=False, low_cpu_mem_usage: 'Optional[bool]'=False, pretrained: 'Optional[bool]'=True, tokenizer_name: 'Optional[str]'='hf_auto', use_fast: 'Optional[bool]'=True):
        """
        Load a pretrained huggingface text transformer backbone.

        Parameters
        ----------
        prefix
            The model prefix.
        checkpoint_name
            Name of the checkpoint or the local directory of a custom checkpoint.
            We support loading checkpoint from
            Huggingface Models list: https://huggingface.co/models
            For example, you may use
                English backbones:
                    - 'microsoft/deberta-v3-base'
                    - 'bert-base-uncased'
                    - 'google/electra-base-discriminator'
                    - 'distilroberta-base'
                Multilingual backbones:
                    - 'microsoft/mdeberta-v3-base'
                    - 'xlm-roberta-base'
        num_classes
            The number of classes. 1 for a regression task.
        pooling_mode
            The pooling mode for the Transformer. Can be "cls", or "mean"
        gradient_checkpointing
            Whether to enable gradient checkpointing
        low_cpu_mem_usage
            Whether to turn on the optimization of reducing the peak CPU memory usage when loading the pretrained model.
        pretrained
            Whether using the pretrained weights. If pretrained=True, download the pretrained model.
        tokenizer_name
            Name of the huggingface tokenizer type.
        use_fast
            Use a fast Rust-based tokenizer if it is supported for a given model.
            If a fast tokenizer is not available for a given model, a normal Python-based tokenizer is returned instead.
            See: https://huggingface.co/docs/transformers/model_doc/auto#transformers.AutoTokenizer.from_pretrained.use_fast
        """
        super().__init__()
        logger.debug(f'initializing {checkpoint_name}')
        self.checkpoint_name = checkpoint_name
        self.num_classes = num_classes
        self.config, self.model = get_hf_config_and_model(checkpoint_name=checkpoint_name, pretrained=pretrained, low_cpu_mem_usage=low_cpu_mem_usage)
        self.tokenizer_name = tokenizer_name
        self.tokenizer = get_pretrained_tokenizer(tokenizer_name=self.tokenizer_name, checkpoint_name=self.checkpoint_name, use_fast=use_fast)
        if isinstance(self.model, T5PreTrainedModel):
            self.is_t5 = True
            del self.model.decoder
        else:
            self.is_t5 = False
        self.gradient_checkpointing = gradient_checkpointing
        if gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
            if self.is_t5:
                self.dummy_layer = DummyLayer()
        self.out_features = self.model.config.hidden_size
        self.head = nn.Linear(self.out_features, num_classes) if num_classes else nn.Identity()
        self.head.apply(init_weights)
        self.prefix = prefix
        self.pooling_mode = pooling_mode
        self.name_to_id = self.get_layer_ids()
        self.head_layer_names = [n for n, layer_id in self.name_to_id.items() if layer_id == 0]
        if hasattr(self.model.config, 'type_vocab_size') and self.model.config.type_vocab_size <= 1:
            self.disable_seg_ids = True
        else:
            self.disable_seg_ids = False

    @property
    def text_token_ids_key(self):
        return f'{self.prefix}_{TEXT_TOKEN_IDS}'

    @property
    def text_segment_ids_key(self):
        return f'{self.prefix}_{TEXT_SEGMENT_IDS}'

    @property
    def text_valid_length_key(self):
        return f'{self.prefix}_{TEXT_VALID_LENGTH}'

    @property
    def input_keys(self):
        return [self.text_token_ids_key, self.text_segment_ids_key, self.text_valid_length_key]

    @property
    def label_key(self):
        return f'{self.prefix}_{LABEL}'

    @property
    def text_column_prefix(self):
        return f'{self.text_token_ids_key}_{COLUMN}'

    @property
    def text_feature_dim(self):
        return self.model.config.hidden_size

    def forward(self, text_token_ids: 'torch.Tensor', text_segment_ids: 'torch.Tensor', text_valid_length: 'torch.Tensor', text_column_names: 'Optional[List[str]]'=None, text_column_indices: 'Optional[List[torch.Tensor]]'=None):
        """
        Parameters
        ----------
        text_token_ids : torch.Tensor
            Indices of input sequence tokens in the vocabulary.
        text_segment_ids : torch.Tensor
            Indices of input sequence segments.
        text_valid_length : torch.Tensor
            Valid length of the input text sequence.
        text_column_names : list of str, optional
            Names of the text columns.
        text_column_indices : list of torch.Tensor, optional
            Start and stop indices of the text columns.

        Returns
        -------
            A tuple that contains (pooled_features, logits, column_features, column_feature_masks)
        """
        if self.disable_seg_ids:
            text_segment_ids = None
        steps = torch.arange(0, text_token_ids.shape[1]).type_as(text_valid_length)
        text_masks = (steps.reshape((1, -1)) < text_valid_length.reshape((-1, 1))).type_as(text_token_ids)
        if self.is_t5:
            inputs_embeds = self.model.encoder.embed_tokens(text_token_ids)
            if self.gradient_checkpointing:
                inputs_embeds = self.dummy_layer(inputs_embeds)
            outputs = self.model.encoder(inputs_embeds=inputs_embeds, attention_mask=text_masks)
        elif 'token_type_ids' in self.tokenizer.model_input_names:
            outputs = self.model(input_ids=text_token_ids, token_type_ids=text_segment_ids, attention_mask=text_masks)
        else:
            outputs = self.model(input_ids=text_token_ids, attention_mask=text_masks)
        if self.pooling_mode == 'cls':
            pooled_features = outputs.last_hidden_state[:, 0, :]
        elif self.pooling_mode == 'mean':
            pooled_features = (outputs.last_hidden_state * text_masks.unsqueeze(-1)).sum(1)
            sum_mask = text_masks.unsqueeze(-1).sum(1)
            sum_mask = torch.clamp(sum_mask, min=1e-09)
            pooled_features = pooled_features / sum_mask
        else:
            raise NotImplementedError(f'Pooling mode={self.pooling_mode} is not supported.')
        logits = self.head(pooled_features)
        last_hidden_state = outputs.last_hidden_state
        batch = {self.text_token_ids_key: text_token_ids, self.text_segment_ids_key: text_segment_ids, self.text_valid_length_key: text_valid_length}
        if text_column_names:
            assert len(text_column_names) == len(text_column_indices), 'invalid text column inputs'
            for idx, name in enumerate(text_column_names):
                batch[name] = text_column_indices[idx]
        column_features, column_feature_masks = get_column_features(batch=batch, column_name_prefix=self.text_column_prefix, features=last_hidden_state, valid_lengths=text_valid_length, cls_feature=pooled_features)
        if column_features == {} or column_feature_masks == {}:
            return pooled_features, logits
        else:
            return pooled_features, logits, column_features, column_feature_masks

    def get_output_dict(self, pooled_features: 'torch.Tensor', logits: 'torch.Tensor', column_features: 'Optional[Dict[str, torch.Tensor]]'=None, column_feature_masks: 'Optional[Dict[str, torch.Tensor]]'=None):
        ret = {COLUMN_FEATURES: {FEATURES: {}, MASKS: {}}}
        if column_features != None:
            ret[COLUMN_FEATURES][FEATURES].update(column_features)
            ret[COLUMN_FEATURES][MASKS].update(column_feature_masks)
        ret[LOGITS] = logits
        ret[FEATURES] = pooled_features
        return {self.prefix: ret}

    def get_layer_ids(self):
        """
        Assign an id to each layer. Layer ids will be used in layer-wise lr decay.
        Basically, id gradually increases when going from the output end to
        the input end. The layers defined in this class, e.g., head, have id 0.

        In the AutoModel scenario, this function may not always return the correct result.
        Thus, you can use "print(json.dumps(name_to_id, indent=2))" to manually check whether
        the layer ids are reasonable.

        Returns
        -------
        A dictionary mapping the layer names (keys) to their ids (values).
        """
        model_prefix = 'model'
        pre_encoder_patterns = 'embeddings', 'LayerNorm', 'wte', 'wpe', 'shared.weight', 'encoder.conv.conv', 'relative_attention_bias', 'dummy_layer'
        post_encoder_patterns = 'head', 'pooler', 'ln_f', 'final_layer_norm'
        names = [n for n, _ in self.named_parameters()]
        name_to_id, names = assign_layer_ids(names=names, pre_encoder_patterns=pre_encoder_patterns, post_encoder_patterns=post_encoder_patterns, model_pre=model_prefix)
        if len(names) > 0:
            logger.debug(f'outer layers are treated as head: {names}')
        for n in names:
            assert n not in name_to_id
            name_to_id[n] = 0
        return name_to_id

    def save(self, save_path: 'str'='./'):
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        logger.info(f'Model weights and tokenizer for {self.prefix} are saved to {save_path}.')


INPUT_IDS = 'input_ids'


PIXEL_VALUES = 'pixel_values'


TOKEN_TYPE_IDS = 'token_type_ids'


class DocumentTransformer(HFAutoModelForTextPrediction):
    """
    Document Classification with Huggingface backbones. Inherit from HFAutoModelForTextPrediction.
    """

    def __init__(self, prefix: 'str', checkpoint_name: 'str'='microsoft/layoutlmv3-base', num_classes: 'Optional[int]'=0, pooling_mode: 'Optional[str]'='cls', gradient_checkpointing: 'Optional[bool]'=False, low_cpu_mem_usage: 'Optional[bool]'=False, pretrained: 'Optional[bool]'=True, tokenizer_name: 'Optional[str]'='hf_auto'):
        """
        Load a pretrained huggingface layout-aware document transformer backbone.

        Parameters
        ----------
        prefix
            The model prefix.
        checkpoint_name
            Name of the checkpoint. We support loading checkpoint from
            Huggingface Models list: https://huggingface.co/models
            For example, you can use layout-aware models:
                - microsoft/layoutlmv3-base
                - microsoft/layoutlm-base-uncased
                - microsoft/xdoc-base
                - microsoft/layoutxlm-base
                - microsoft/layoutlmv2-base-uncased
            you may also use text focused transformers:
                - 'microsoft/deberta-v3-base'
                - 'bert-base-uncased'
        num_classes
            The number of classes. 1 for a regression task.
        pooling_mode
            The pooling mode for the Transformer. Can be "cls", or "mean"
        gradient_checkpointing
            Whether to enable gradient checkpointing
        low_cpu_mem_usage
            Whether to turn on the optimization of reducing the peak CPU memory usage when loading the pretrained model.
        pretrained
            Whether using the pretrained weights. If pretrained=True, download the pretrained model.
        tokenizer_name
            Name of the huggingface tokenizer type.
        """
        logger.debug(f'initializing {checkpoint_name}')
        super().__init__(prefix=prefix, checkpoint_name=checkpoint_name, num_classes=num_classes, pooling_mode=pooling_mode, gradient_checkpointing=gradient_checkpointing, low_cpu_mem_usage=low_cpu_mem_usage, pretrained=pretrained, tokenizer_name=tokenizer_name)
        self.is_text_only_flag = self.is_text_only()
        if self.is_text_only_flag:
            logger.debug(f'Checkpoint: {checkpoint_name} uses the text data only for classification.')

    def is_text_only(self):
        """
        Check the tokenizer to see if it is a text only tokenizer.

        Parameters
        ----------
        tokenizer
            The tokenizer to be used.

        Returns
        -------
        True if the tokenizer only accept text, otherwise, False.
        """
        model_args = list(inspect.signature(self.tokenizer.__call__).parameters.keys())
        if 'boxes' not in model_args:
            return True
        else:
            return False

    @property
    def text_attention_mask_key(self):
        return f'{self.prefix}_{ATTENTION_MASK}'

    @property
    def text_bbox_key(self):
        return f'{self.prefix}_{BBOX}'

    @property
    def document_pixel_value_key(self):
        return f'{self.prefix}_{PIXEL_VALUES}'

    def update_input_data(self, input_data: 'dict', batch: 'dict', keys: 'dict'):
        """
        Update the model input data based on the model argument.
        For example, microsoft/layoutlm-base-uncased has a "bbox" argument;
        microsoft/layoutlmv3-base has arguments: "bbox" and "image".
        Text only bert does not have these two arguments.

        Parameters
        ----------
        input_data
            A dictionary containing the model input data.
        batch:
            A dictionary containing the input mini-batch data.
            We need to use the keys with the model prefix to index required data.
        keys:
            A dictionary containing the model arguments and corresponding batch keys.
        """
        model_args = list(inspect.signature(self.model.forward).parameters.keys())
        for key, value in keys.items():
            if key in model_args:
                input_data.update({key: batch[value]})

    def forward(self, batch: 'dict'):
        """
        Parameters
        ----------
        batch
            A dictionary containing the input mini-batch data.
            We need to use the keys with the model prefix to index required data.

        Returns
        -------
            A dictionary with logits and features.
        """
        input_data = {}
        self.update_input_data(input_data, batch, keys={INPUT_IDS: self.text_token_ids_key, TOKEN_TYPE_IDS: self.text_segment_ids_key, ATTENTION_MASK: self.text_attention_mask_key, BBOX: self.text_bbox_key, PIXEL_VALUES: self.document_pixel_value_key, IMAGE: self.document_pixel_value_key})
        text_masks = batch[self.text_attention_mask_key]
        outputs = self.model(**input_data)
        if self.pooling_mode == 'cls':
            pooled_features = outputs.last_hidden_state[:, 0, :]
        elif self.pooling_mode == 'mean':
            pooled_features = outputs.last_hidden_state.mean(1)
        else:
            raise NotImplementedError(f'Pooling mode={self.pooling_mode} is not supported.')
        logits = self.head(pooled_features)
        ret = {COLUMN_FEATURES: {FEATURES: {}, MASKS: {}}}
        column_features, column_feature_masks = get_column_features(batch=batch, column_name_prefix=self.text_column_prefix, features=outputs.last_hidden_state, valid_lengths=sum(text_masks), cls_feature=pooled_features)
        ret[COLUMN_FEATURES][FEATURES].update(column_features)
        ret[COLUMN_FEATURES][MASKS].update(column_feature_masks)
        ret.update({LOGITS: logits, FEATURES: pooled_features})
        return {self.prefix: ret}

