import sys
_module = sys.modules[__name__]
del sys
make_ref = _module
preprocess = _module
src = _module
data = _module
dataset = _module
dictionary = _module
loader = _module
evaluation = _module
evaluator = _module
glue = _module
xnli = _module
logger = _module
model = _module
embedder = _module
memory = _module
memory = _module
query = _module
utils = _module
pretrain = _module
transformer = _module
optim = _module
slurm = _module
trainer = _module
utils = _module
lowercase_and_remove_accent = _module
segment_th = _module
train = _module
translate = _module
transformers = _module
add_umt_subqs_subas_to_q_squad_format_new = _module
convert_hotpot2questionlist_script = _module
convert_hotpot2squad_simple_script = _module
conf = _module
download_glue_data = _module
ensemble_answers_by_confidence_script = _module
dataset = _module
distiller = _module
binarized_data = _module
extract_for_distil = _module
token_counts = _module
train = _module
utils = _module
hotpot_evaluate_v1 = _module
finetune_on_pregenerated = _module
pregenerate_training_data = _module
simple_lm_finetuning = _module
run_bertology = _module
run_generation = _module
run_glue = _module
run_lm_finetuning = _module
run_squad = _module
run_openai_gpt = _module
run_swag = _module
run_transfo_xl = _module
test_examples = _module
utils_glue = _module
utils_squad = _module
utils_squad_evaluate = _module
hubconf = _module
pseudoalignment = _module
embed_questions_with_bert = _module
pseudo_decomp_bert = _module
pseudo_decomp_bert_nsp = _module
pseudo_decomp_fasttext = _module
pseudo_decomp_random = _module
pseudo_decomp_tfidf = _module
pseudo_decomp_utils = _module
pseudo_decomp_variable = _module
replace_subq_entities = _module
pytorch_transformers = _module
convert_gpt2_checkpoint_to_pytorch = _module
convert_openai_checkpoint_to_pytorch = _module
convert_pytorch_checkpoint_to_tf = _module
convert_roberta_checkpoint_to_pytorch = _module
convert_tf_checkpoint_to_pytorch = _module
convert_transfo_xl_checkpoint_to_pytorch = _module
convert_xlm_checkpoint_to_pytorch = _module
convert_xlnet_checkpoint_to_pytorch = _module
file_utils = _module
modeling_auto = _module
modeling_bert = _module
modeling_distilbert = _module
modeling_gpt2 = _module
modeling_openai = _module
modeling_roberta = _module
modeling_transfo_xl = _module
modeling_transfo_xl_utilities = _module
modeling_utils = _module
modeling_xlm = _module
modeling_xlnet = _module
optimization = _module
tests = _module
conftest = _module
modeling_auto_test = _module
modeling_bert_test = _module
modeling_common_test = _module
modeling_distilbert_test = _module
modeling_gpt2_test = _module
modeling_openai_test = _module
modeling_roberta_test = _module
modeling_transfo_xl_test = _module
modeling_xlm_test = _module
modeling_xlnet_test = _module
optimization_test = _module
tokenization_auto_test = _module
tokenization_bert_test = _module
tokenization_dilbert_test = _module
tokenization_gpt2_test = _module
tokenization_openai_test = _module
tokenization_roberta_test = _module
tokenization_tests_commons = _module
tokenization_transfo_xl_test = _module
tokenization_utils_test = _module
tokenization_xlm_test = _module
tokenization_xlnet_test = _module
tokenization_auto = _module
tokenization_bert = _module
tokenization_distilbert = _module
tokenization_gpt2 = _module
tokenization_openai = _module
tokenization_roberta = _module
tokenization_transfo_xl = _module
tokenization_utils = _module
tokenization_xlm = _module
tokenization_xlnet = _module
setup = _module
split_hotpot_dev = _module
split_umt_dev = _module
umt_gen_subqs_to_squad_format = _module

from _paritybench_helpers import _mock_config, patch_functional
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


from logging import getLogger


import math


import numpy as np


import torch


from collections import OrderedDict


import copy


import time


from torch import nn


import torch.nn.functional as F


from scipy.stats import spearmanr


from scipy.stats import pearsonr


from sklearn.metrics import f1_score


from sklearn.metrics import matthews_corrcoef


import itertools


from torch.nn import functional as F


import torch.nn as nn


import re


import inspect


from torch import optim


from torch.nn.utils import clip_grad_norm_


import random


from typing import List


from itertools import chain


from collections import Counter


import logging


from collections import namedtuple


from torch.utils.data import DataLoader


from torch.utils.data import Dataset


from torch.utils.data import RandomSampler


from torch.utils.data.distributed import DistributedSampler


from torch.utils.data import SequentialSampler


from torch.utils.data import TensorDataset


from torch.utils.data import Subset


from torch.nn import CrossEntropyLoss


from torch.nn import MSELoss


from sklearn.preprocessing import normalize


import tensorflow as tf


import numpy


from functools import wraps


import collections


from torch.nn.parameter import Parameter


from collections import defaultdict


from torch.optim import Optimizer


from torch.optim.lr_scheduler import LambdaLR


import uuid


def get_gaussian_keys(n_keys, dim, normalized, seed):
    """
    Generate random Gaussian keys.
    """
    rng = np.random.RandomState(seed)
    X = rng.randn(n_keys, dim)
    if normalized:
        X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X.astype(np.float32)


def get_uniform_keys(n_keys, dim, normalized, seed):
    """
    Generate random uniform keys (same initialization as nn.Linear).
    """
    rng = np.random.RandomState(seed)
    bound = 1 / math.sqrt(dim)
    X = rng.uniform(-bound, bound, (n_keys, dim))
    if normalized:
        X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X.astype(np.float32)


def cartesian_product(a, b):
    """
    Compute the batched cartesian product between two matrices.
    Input:
        a: Tensor(n, d1)
        b: Tensor(n, d2)
    Output:
        output: Tensor(n, d1 * d2, 2)
    """
    n1, d1 = a.shape
    n2, d2 = b.shape
    assert n1 == n2
    return torch.cat([a.unsqueeze(-1).repeat(1, 1, d2).unsqueeze(-1), b.repeat(1, d1).view(n2, d1, d2).unsqueeze(-1)], 3).view(n1, d1 * d2, 2)


def swig_ptr_from_FloatTensor(x):
    assert x.is_contiguous()
    assert x.dtype == torch.float32
    return faiss.cast_integer_to_float_ptr(x.storage().data_ptr() + x.storage_offset() * 4)


def swig_ptr_from_LongTensor(x):
    assert x.is_contiguous()
    assert x.dtype == torch.int64, 'dtype=%s' % x.dtype
    return faiss.cast_integer_to_long_ptr(x.storage().data_ptr() + x.storage_offset() * 8)


def get_knn_faiss(xb, xq, k, distance='dot_product'):
    """
    `metric` can be faiss.METRIC_INNER_PRODUCT or faiss.METRIC_L2
    https://github.com/facebookresearch/faiss/blob/master/gpu/test/test_pytorch_faiss.py
    """
    assert xb.device == xq.device
    assert distance in ['dot_product', 'l2']
    metric = faiss.METRIC_INNER_PRODUCT if distance == 'dot_product' else faiss.METRIC_L2
    xq_ptr = swig_ptr_from_FloatTensor(xq)
    xb_ptr = swig_ptr_from_FloatTensor(xb)
    nq, d1 = xq.size()
    nb, d2 = xb.size()
    assert d1 == d2
    D = torch.empty(nq, k, device=xb.device, dtype=torch.float32)
    I = torch.empty(nq, k, device=xb.device, dtype=torch.int64)
    D_ptr = swig_ptr_from_FloatTensor(D)
    I_ptr = swig_ptr_from_LongTensor(I)
    faiss.bruteForceKnn(FAISS_RES, metric, xb_ptr, nb, xq_ptr, nq, d1, k, D_ptr, I_ptr)
    return D, I


class BottleneckResidualConv2d(nn.Module):

    def __init__(self, input_channels, output_channels, kernel_size, bias=True, batchnorm=True, groups=1):
        super().__init__()
        hidden_channels = min(input_channels, output_channels)
        assert all(k % 2 == 1 for k in kernel_size)
        self.conv1 = nn.Conv2d(input_channels, hidden_channels, kernel_size, padding=[(k // 2) for k in kernel_size], bias=bias, groups=groups)
        self.conv2 = nn.Conv2d(hidden_channels, output_channels, kernel_size, padding=[(k // 2) for k in kernel_size], bias=bias, groups=groups)
        self.act = nn.ReLU()
        self.batchnorm = batchnorm
        if self.batchnorm:
            self.bn1 = nn.BatchNorm2d(hidden_channels)
            self.bn2 = nn.BatchNorm2d(output_channels)
        if input_channels == output_channels:
            self.residual = nn.Sequential()
        else:
            self.residual = nn.Conv2d(input_channels, output_channels, (1, 1), bias=False, groups=groups)

    def forward(self, input):
        x = self.conv1(input)
        x = self.bn1(x) if self.batchnorm else x
        x = self.act(x)
        x = self.conv2(x)
        x = self.bn2(x) if self.batchnorm else x
        x = self.act(x + self.residual(input))
        return x


def convs(channel_sizes, kernel_sizes, bias=True, batchnorm=True, residual=False, groups=1):
    """
    Generate a convolutional neural network.
    """
    assert len(channel_sizes) >= 2
    assert len(channel_sizes) == len(kernel_sizes) + 1
    pairs = [(channel_sizes[i], channel_sizes[i + 1]) for i in range(len(channel_sizes) - 1)]
    layers = []
    for i, (dim_in, dim_out) in enumerate(pairs):
        ks = kernel_sizes[i], kernel_sizes[i]
        in_group = 1 if i == 0 else groups
        _dim_in = dim_in * in_group
        _dim_out = dim_out * groups
        if not residual:
            layers.append(nn.Conv2d(_dim_in, _dim_out, ks, padding=[(k // 2) for k in ks], bias=bias, groups=in_group))
            if batchnorm:
                layers.append(nn.BatchNorm2d(_dim_out))
            if i < len(pairs) - 1:
                layers.append(nn.ReLU())
        else:
            layers.append(BottleneckResidualConv2d(_dim_in, _dim_out, ks, bias=bias, batchnorm=batchnorm, groups=in_group))
            if i == len(pairs) - 1:
                layers.append(nn.Conv2d(_dim_out, _dim_out, (1, 1), bias=bias))
    return nn.Sequential(*layers)


class QueryConv(nn.Module):

    def __init__(self, input_dim, heads, k_dim, product_quantization, multi_query_net, sizes, kernel_sizes, bias=True, batchnorm=True, residual=False, grouped_conv=False):
        super().__init__()
        self.input_dim = input_dim
        self.heads = heads
        self.k_dim = k_dim
        self.sizes = sizes
        self.grouped_conv = grouped_conv
        assert not multi_query_net or product_quantization or heads >= 2
        assert sizes[0] == input_dim
        assert sizes[-1] == k_dim // 2 if multi_query_net else heads * k_dim
        assert self.grouped_conv is False or len(sizes) > 2
        assert len(sizes) == len(kernel_sizes) + 1 >= 2 and all(ks % 2 == 1 for ks in kernel_sizes)
        self.groups = 2 * heads if multi_query_net else 1
        if self.grouped_conv:
            self.query_convs = convs(sizes, kernel_sizes, bias=bias, batchnorm=batchnorm, residual=residual, groups=self.groups)
        elif len(self.sizes) == 2:
            sizes_ = list(sizes)
            sizes_[-1] = sizes_[-1] * self.groups
            self.query_convs = convs(sizes_, kernel_sizes, bias=bias, batchnorm=batchnorm, residual=residual, groups=1)
        else:
            self.query_convs = nn.ModuleList([convs(sizes, kernel_sizes, bias=bias, batchnorm=batchnorm, residual=residual, groups=1) for _ in range(self.groups)])

    def forward(self, input):
        bs, nf, h, w = input.shape
        assert nf == self.input_dim
        if self.grouped_conv or len(self.sizes) == 2:
            query = self.query_convs(input)
        else:
            outputs = [m(input) for m in self.query_convs]
            query = torch.cat(outputs, 1) if len(outputs) > 1 else outputs[0]
        assert query.shape == (bs, self.heads * self.k_dim, h, w)
        query = query.transpose(1, 3).contiguous().view(bs * w * h * self.heads, self.k_dim)
        return query


def get_slices(dim, head_id):
    """
    Generate slices of hidden dimensions.
    Used when there are multiple heads and/or different set of keys,
    and that there is no query network.
    """
    if head_id == 0:
        return [(0, dim)]
    offset = dim // 2 ** (head_id + 1)
    starts = np.arange(0, dim, offset)
    slices1 = [(x, x + offset) for i, x in enumerate(starts) if i % 2 == 0]
    slices2 = [(x, x + offset) for i, x in enumerate(starts) if i % 2 == 1]
    return slices1 + slices2


class QueryIdentity(nn.Module):

    def __init__(self, input_dim, heads, shuffle_hidden):
        super().__init__()
        self.input_dim = input_dim
        self.heads = heads
        self.shuffle_query = shuffle_hidden
        assert shuffle_hidden is False or heads > 1
        assert shuffle_hidden is False or self.input_dim % 2 ** self.heads == 0
        if shuffle_hidden:
            self.slices = {head_id: get_slices(input_dim, head_id) for head_id in range(heads)}

    def forward(self, input):
        """
        Generate queries from hidden states by either
        repeating them or creating some shuffled version.
        """
        assert input.shape[-1] == self.input_dim
        input = input.contiguous().view(-1, self.input_dim) if input.dim() > 2 else input
        bs = len(input)
        if self.heads == 1:
            query = input
        elif not self.shuffle_query:
            query = input.unsqueeze(1).repeat(1, self.heads, 1)
            query = query.view(bs * self.heads, self.input_dim)
        else:
            query = torch.cat([input[:, a:b] for head_id in range(self.heads) for a, b in self.slices[head_id]], 1).view(bs * self.heads, self.input_dim)
        assert query.shape == (bs * self.heads, self.input_dim)
        return query


class GroupedLinear(nn.Module):

    def __init__(self, in_features, out_features, bias=True, groups=1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.groups = groups
        self.bias = bias
        assert groups > 1
        self.layer = nn.Conv1d(in_features, out_features, bias=bias, kernel_size=1, groups=groups)

    def forward(self, input):
        assert input.dim() == 2 and input.size(1) == self.in_features
        return self.layer(input.unsqueeze(2)).squeeze(2)

    def extra_repr(self):
        return 'in_features={}, out_features={}, groups={}, bias={}'.format(self.in_features, self.out_features, self.groups, self.bias is not None)


def mlp(sizes, bias=True, batchnorm=True, groups=1):
    """
    Generate a feedforward neural network.
    """
    assert len(sizes) >= 2
    pairs = [(sizes[i], sizes[i + 1]) for i in range(len(sizes) - 1)]
    layers = []
    for i, (dim_in, dim_out) in enumerate(pairs):
        if groups == 1 or i == 0:
            layers.append(nn.Linear(dim_in, groups * dim_out, bias=bias))
        else:
            layers.append(GroupedLinear(groups * dim_in, groups * dim_out, bias=bias, groups=groups))
        if batchnorm:
            layers.append(nn.BatchNorm1d(groups * dim_out))
        if i < len(pairs) - 1:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class QueryMLP(nn.Module):

    def __init__(self, input_dim, heads, k_dim, product_quantization, multi_query_net, sizes, bias=True, batchnorm=True, grouped_conv=False):
        super().__init__()
        self.input_dim = input_dim
        self.heads = heads
        self.k_dim = k_dim
        self.sizes = sizes
        self.grouped_conv = grouped_conv
        assert not multi_query_net or product_quantization or heads >= 2
        assert sizes[0] == input_dim
        assert sizes[-1] == k_dim // 2 if multi_query_net else heads * k_dim
        assert self.grouped_conv is False or len(sizes) > 2
        self.groups = 2 * heads if multi_query_net else 1
        if self.grouped_conv:
            self.query_mlps = mlp(sizes, bias=bias, batchnorm=batchnorm, groups=self.groups)
        elif len(self.sizes) == 2:
            sizes_ = list(sizes)
            sizes_[-1] = sizes_[-1] * self.groups
            self.query_mlps = mlp(sizes_, bias=bias, batchnorm=batchnorm, groups=1)
        else:
            self.query_mlps = nn.ModuleList([mlp(sizes, bias=bias, batchnorm=batchnorm, groups=1) for _ in range(self.groups)])

    def forward(self, input):
        """
        Compute queries using either grouped 1D convolutions or ModuleList + concat.
        """
        assert input.shape[-1] == self.input_dim
        input = input.contiguous().view(-1, self.input_dim) if input.dim() > 2 else input
        bs = len(input)
        if self.grouped_conv or len(self.sizes) == 2:
            query = self.query_mlps(input)
        else:
            outputs = [m(input) for m in self.query_mlps]
            query = torch.cat(outputs, 1) if len(outputs) > 1 else outputs[0]
        assert query.shape == (bs, self.heads * self.k_dim)
        return query.view(bs * self.heads, self.k_dim)


FALSY_STRINGS = {'off', 'false', '0'}


TRUTHY_STRINGS = {'on', 'true', '1'}


def bool_flag(s):
    """
    Parse boolean arguments from the command line.
    """
    if s.lower() in FALSY_STRINGS:
        return False
    elif s.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError('Invalid value for a boolean flag!')


logger = logging.getLogger(__name__)


class HashingMemory(nn.Module):
    MEM_VALUES_PARAMS = '.values.weight'
    VALUES = None
    EVAL_MEMORY = True
    _ids = itertools.count(0)

    def __init__(self, input_dim, output_dim, params):
        super().__init__()
        self.id = next(self._ids)
        self.input2d = params.mem_input2d
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.size = params.mem_size
        self.modulo_size = params.mem_modulo_size
        self.n_indices = params.n_indices
        self.k_dim = params.mem_k_dim
        self.v_dim = params.mem_v_dim if params.mem_v_dim > 0 else output_dim
        self.heads = params.mem_heads
        self.knn = params.mem_knn
        self.shuffle_indices = params.mem_shuffle_indices
        self.keys_normalized_init = params.mem_keys_normalized_init
        self.product_quantization = params.mem_product_quantization
        assert self.modulo_size == -1 and self.size == self.n_indices or self.n_indices > self.size == self.modulo_size >= 1
        self.keys_type = params.mem_keys_type
        self.learn_keys = params.mem_keys_learn
        self.use_different_keys = params.mem_use_different_keys
        self.query_detach_input = params.mem_query_detach_input
        self.query_net_learn = params.mem_query_net_learn
        self.multi_query_net = params.mem_multi_query_net
        self.shuffle_query = params.mem_shuffle_query
        assert self.use_different_keys is False or self.keys_type in ['gaussian', 'uniform']
        assert self.use_different_keys is False or self.heads >= 2 or self.product_quantization
        assert self.multi_query_net is False or self.heads >= 2 or self.product_quantization
        assert self.shuffle_query is False or self.heads > 1 and params.mem_query_layer_sizes == ''
        assert self.shuffle_query is False or self.input_dim % 2 ** self.heads == 0
        self.normalize_query = params.mem_normalize_query
        self.temperature = params.mem_temperature
        self.score_softmax = params.mem_score_softmax
        self.score_subtract = params.mem_score_subtract
        self.score_normalize = params.mem_score_normalize
        assert self.score_subtract in ['', 'min', 'mean', 'median']
        assert self.score_subtract == '' or self.knn >= 2
        assert not (self.score_normalize and self.score_softmax and self.score_subtract == '')
        self.input_dropout = params.mem_input_dropout
        self.query_dropout = params.mem_query_dropout
        self.value_dropout = params.mem_value_dropout
        self.init_keys()
        self.values = nn.EmbeddingBag(self.size, self.v_dim, mode='sum', sparse=params.mem_sparse)
        if params.mem_share_values:
            if HashingMemory.VALUES is None:
                HashingMemory.VALUES = self.values.weight
            else:
                self.values.weight = HashingMemory.VALUES
        if params.mem_value_zero_init:
            nn.init.zeros_(self.values.weight)
        else:
            nn.init.normal_(self.values.weight, mean=0, std=self.v_dim ** -0.5)
        if len(params.mem_query_layer_sizes) == 0:
            assert self.heads == 1 or self.use_different_keys or self.shuffle_query
            assert self.input_dim == self.k_dim
            self.query_proj = QueryIdentity(self.input_dim, self.heads, self.shuffle_query)
        if len(params.mem_query_layer_sizes) > 0:
            assert not self.shuffle_query
            l_sizes = list(params.mem_query_layer_sizes)
            assert len(l_sizes) >= 2 and l_sizes[0] == l_sizes[-1] == 0
            l_sizes[0] = self.input_dim
            l_sizes[-1] = self.k_dim // 2 if self.multi_query_net else self.heads * self.k_dim
            if self.input2d:
                self.query_proj = QueryConv(self.input_dim, self.heads, self.k_dim, self.product_quantization, self.multi_query_net, l_sizes, params.mem_query_kernel_sizes, bias=params.mem_query_bias, batchnorm=params.mem_query_batchnorm, grouped_conv=params.mem_grouped_conv)
            else:
                assert params.mem_query_kernel_sizes == ''
                assert not params.mem_query_residual
                self.query_proj = QueryMLP(self.input_dim, self.heads, self.k_dim, self.product_quantization, self.multi_query_net, l_sizes, bias=params.mem_query_bias, batchnorm=params.mem_query_batchnorm, grouped_conv=params.mem_grouped_conv)
        if self.shuffle_indices:
            head_permutations = [torch.randperm(self.n_indices).unsqueeze(0) for i in range(self.heads)]
            self.register_buffer('head_permutations', torch.cat(head_permutations, 0))
        if self.query_net_learn is False:
            for p in self.query_proj.parameters():
                p.requires_grad = False

    def forward(self, input):
        """
        Read from the memory.
        """
        if self.query_detach_input:
            input = input.detach()
        if self.input2d:
            assert input.shape[1] == self.input_dim
            n_images, _, height, width = input.shape
            prefix_shape = n_images, width, height
        else:
            assert input.shape[-1] == self.input_dim
            prefix_shape = input.shape[:-1]
        bs = np.prod(prefix_shape)
        input = F.dropout(input, p=self.input_dropout, training=self.training)
        query = self.query_proj(input)
        query = F.dropout(query, p=self.query_dropout, training=self.training)
        assert query.shape == (bs * self.heads, self.k_dim)
        scores, indices = self.get_indices(query, self.knn)
        if self.shuffle_indices:
            indices = indices.view(bs, self.heads, -1).chunk(self.heads, 1)
            indices = [p[idx] for p, idx in zip(self.head_permutations, indices)]
            indices = torch.cat(indices, 1).view(bs * self.heads, -1)
        if self.modulo_size != -1:
            indices = indices % self.modulo_size
        if self.temperature != 1:
            scores = scores / self.temperature
        if self.score_softmax:
            scores = F.softmax(scores.float(), dim=-1).type_as(scores)
        if self.score_subtract != '':
            if self.score_subtract == 'min':
                to_sub = scores.min(1, keepdim=True)[0]
            if self.score_subtract == 'mean':
                to_sub = scores.mean(1, keepdim=True)
            if self.score_subtract == 'median':
                to_sub = scores.median(1, keepdim=True)[0]
            scores = scores - to_sub
        if self.score_normalize:
            scores = scores / scores.norm(p=1, dim=1, keepdim=True)
        indices = indices.view(bs, self.heads * self.knn)
        scores = scores.view(bs, self.heads * self.knn)
        output = self.values(indices, per_sample_weights=scores.to(self.values.weight.data))
        output = F.dropout(output, p=self.value_dropout, training=self.training)
        if self.input2d:
            output = output.view(n_images, width, height, self.v_dim)
            output = output.transpose(1, 3)
        elif len(prefix_shape) >= 2:
            output = output.view(prefix_shape + (self.v_dim,))
        if not self.training and HashingMemory.EVAL_MEMORY:
            self.last_indices = indices.view(bs, self.heads, self.knn).detach().cpu()
            self.last_scores = scores.view(bs, self.heads, self.knn).detach().cpu().float()
        return output

    def init_keys(self):
        raise Exception('Not implemented!')

    def _get_indices(self, query, knn, keys):
        raise Exception('Not implemented!')

    def get_indices(self, query, knn):
        raise Exception('Not implemented!')

    @staticmethod
    def register_args(parser):
        """
        Register memory parameters
        """
        parser.add_argument('--mem_implementation', type=str, default='pq_fast', help='Memory implementation (flat, pq_default, pq_fast)')
        parser.add_argument('--mem_grouped_conv', type=bool_flag, default=False, help='Use grouped convolutions in the query network')
        parser.add_argument('--mem_values_optimizer', type=str, default='adam,lr=0.001', help='Memory values optimizer ( for the same optimizer as the rest of the model)')
        parser.add_argument('--mem_sparse', type=bool_flag, default=False, help='Perform sparse updates for the values')
        parser.add_argument('--mem_input2d', type=bool_flag, default=False, help='Convolutional query network')
        parser.add_argument('--mem_k_dim', type=int, default=256, help='Memory keys dimension')
        parser.add_argument('--mem_v_dim', type=int, default=-1, help='Memory values dimension (-1 for automatic output dimension)')
        parser.add_argument('--mem_heads', type=int, default=4, help='Number of memory reading heads')
        parser.add_argument('--mem_knn', type=int, default=32, help='Number of memory slots to read / update - k-NN to the query')
        parser.add_argument('--mem_share_values', type=bool_flag, default=False, help='Share values across memories')
        parser.add_argument('--mem_shuffle_indices', type=bool_flag, default=False, help='Shuffle indices for different heads')
        parser.add_argument('--mem_shuffle_query', type=bool_flag, default=False, help='Shuffle query dimensions (when the query network is the identity and there are multiple heads)')
        parser.add_argument('--mem_modulo_size', type=int, default=-1, help='Effective memory size: indices are taken modulo this parameter. -1 to disable.')
        parser.add_argument('--mem_keys_type', type=str, default='uniform', help='Memory keys type (binary,gaussian,uniform)')
        parser.add_argument('--mem_n_keys', type=int, default=512, help='Number of keys')
        parser.add_argument('--mem_keys_normalized_init', type=bool_flag, default=False, help='Normalize keys at initialization')
        parser.add_argument('--mem_keys_learn', type=bool_flag, default=True, help='Learn keys')
        parser.add_argument('--mem_use_different_keys', type=bool_flag, default=True, help='Use different keys for each head / product quantization')
        parser.add_argument('--mem_query_detach_input', type=bool_flag, default=False, help='Detach input')
        parser.add_argument('--mem_query_layer_sizes', type=str, default='0,0', help="Query MLP layer sizes ('', '0,0', '0,512,0')")
        parser.add_argument('--mem_query_kernel_sizes', type=str, default='', help='Query MLP kernel sizes (2D inputs only)')
        parser.add_argument('--mem_query_bias', type=bool_flag, default=True, help='Query MLP bias')
        parser.add_argument('--mem_query_batchnorm', type=bool_flag, default=False, help='Query MLP batch norm')
        parser.add_argument('--mem_query_net_learn', type=bool_flag, default=True, help='Query MLP learn')
        parser.add_argument('--mem_query_residual', type=bool_flag, default=False, help='Use a bottleneck with a residual layer in the query MLP')
        parser.add_argument('--mem_multi_query_net', type=bool_flag, default=False, help='Use multiple query MLP (one for each head)')
        parser.add_argument('--mem_value_zero_init', type=bool_flag, default=False, help='Initialize values with zeros')
        parser.add_argument('--mem_normalize_query', type=bool_flag, default=False, help='Normalize queries')
        parser.add_argument('--mem_temperature', type=float, default=1, help='Divide scores by a temperature')
        parser.add_argument('--mem_score_softmax', type=bool_flag, default=True, help='Apply softmax on scores')
        parser.add_argument('--mem_score_subtract', type=str, default='', help="Subtract scores ('', min, mean, median)")
        parser.add_argument('--mem_score_normalize', type=bool_flag, default=False, help='L1 normalization of the scores')
        parser.add_argument('--mem_input_dropout', type=float, default=0, help='Input dropout')
        parser.add_argument('--mem_query_dropout', type=float, default=0, help='Query dropout')
        parser.add_argument('--mem_value_dropout', type=float, default=0, help='Value dropout')

    @staticmethod
    def build(input_dim, output_dim, params):
        if params.mem_implementation == 'flat':
            M = HashingMemoryFlat
        elif params.mem_implementation == 'pq_default':
            M = HashingMemoryProduct
        elif params.mem_implementation == 'pq_fast':
            M = HashingMemoryProductFast
        else:
            raise Exception('Unknown memory implementation!')
        return M(input_dim, output_dim, params)

    @staticmethod
    def check_params(params):
        """
        Check and initialize memory parameters.
        """
        assert params.mem_implementation in ['flat', 'pq_default', 'pq_fast']
        params.mem_product_quantization = params.mem_implementation != 'flat'
        assert params.mem_grouped_conv is False or params.mem_multi_query_net
        params.mem_values_optimizer = params.optimizer if params.mem_values_optimizer == '' else params.mem_values_optimizer
        params.mem_values_optimizer = params.mem_values_optimizer.replace('adam', 'sparseadam') if params.mem_sparse else params.mem_values_optimizer
        assert params.mem_k_dim >= 2
        assert params.mem_product_quantization is False or params.mem_k_dim % 2 == 0
        assert params.mem_keys_type in ['binary', 'gaussian', 'uniform']
        if params.mem_keys_type == 'binary':
            assert params.mem_keys_normalized_init is False
            assert 1 << params.mem_k_dim == params.mem_n_keys
        if params.mem_product_quantization:
            params.n_indices = params.mem_n_keys ** 2
        else:
            params.n_indices = params.mem_n_keys
        if params.mem_modulo_size == -1:
            params.mem_size = params.n_indices
        else:
            assert 1 <= params.mem_modulo_size < params.n_indices
            params.mem_size = params.mem_modulo_size
        assert not params.mem_use_different_keys or params.mem_keys_type in ['gaussian', 'uniform']
        assert not params.mem_use_different_keys or params.mem_heads >= 2 or params.mem_product_quantization
        assert not params.mem_multi_query_net or params.mem_heads >= 2 or params.mem_product_quantization
        assert not params.mem_multi_query_net or params.mem_query_layer_sizes not in ['', '0,0']
        assert not params.mem_shuffle_query or params.mem_heads > 1 and params.mem_query_layer_sizes == ''
        if params.mem_query_layer_sizes == '':
            assert params.mem_heads == 1 or params.mem_use_different_keys or params.mem_shuffle_query
        else:
            s = [int(x) for x in filter(None, params.mem_query_layer_sizes.split(','))]
            assert len(s) >= 2 and s[0] == s[-1] == 0
            params.mem_query_layer_sizes = s
            assert not params.mem_query_residual or params.mem_input2d
        if params.mem_query_kernel_sizes == '':
            assert not params.mem_input2d or params.mem_query_layer_sizes == ''
        else:
            assert params.mem_input2d
            s = [int(x) for x in filter(None, params.mem_query_kernel_sizes.split(','))]
            params.mem_query_kernel_sizes = s
            assert all(ks % 2 == 1 for ks in s)
            assert len(params.mem_query_kernel_sizes) == len(params.mem_query_layer_sizes) - 1 >= 1
        assert params.mem_score_subtract in ['', 'min', 'mean', 'median']
        assert params.mem_score_subtract == '' or params.mem_knn >= 2
        assert not (params.mem_score_normalize and params.mem_score_softmax and params.mem_score_subtract == '')
        assert 0 <= params.mem_input_dropout < 1
        assert 0 <= params.mem_query_dropout < 1
        assert 0 <= params.mem_value_dropout < 1
        if params.mem_query_batchnorm:
            logger.warning('WARNING: if you use batch normalization, be sure that you use batches of sentences with the same size at training time. Otherwise, the padding token will result in incorrect mean/variance estimations in the BatchNorm layer.')


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    return m


class PredLayer(nn.Module):
    """
    Prediction layer (cross_entropy or adaptive_softmax).
    """

    def __init__(self, params):
        super().__init__()
        self.asm = params.asm
        self.n_words = params.n_words
        self.pad_index = params.pad_index
        dim = params.emb_dim
        if params.asm is False:
            self.proj = Linear(dim, params.n_words, bias=True)
        else:
            self.proj = nn.AdaptiveLogSoftmaxWithLoss(in_features=dim, n_classes=params.n_words, cutoffs=params.asm_cutoffs, div_value=params.asm_div_value, head_bias=True)

    def forward(self, x, y, get_scores=False):
        """
        Compute the loss, and optionally the scores.
        """
        assert (y == self.pad_index).sum().item() == 0
        if self.asm is False:
            scores = self.proj(x).view(-1, self.n_words)
            loss = F.cross_entropy(scores, y, reduction='mean')
        else:
            _, loss = self.proj(x, y)
            scores = self.proj.log_prob(x) if get_scores else None
        return scores, loss

    def get_scores(self, x):
        """
        Compute scores.
        """
        assert x.dim() == 2
        return self.proj.log_prob(x) if self.asm else self.proj(x)


def prune_linear_layer(layer, index, dim=0):
    """ Prune a linear layer (a model parameters) to keep only entries in index.
        Return the pruned layer as a new layer with requires_grad=True.
        Used to remove heads.
    """
    index = index
    W = layer.weight.index_select(dim, index).clone().detach()
    if layer.bias is not None:
        if dim == 1:
            b = layer.bias.clone().detach()
        else:
            b = layer.bias[index].clone().detach()
    new_size = list(layer.weight.size())
    new_size[dim] = len(index)
    new_layer = nn.Linear(new_size[1], new_size[0], bias=layer.bias is not None)
    new_layer.weight.requires_grad = False
    new_layer.weight.copy_(W.contiguous())
    new_layer.weight.requires_grad = True
    if layer.bias is not None:
        new_layer.bias.requires_grad = False
        new_layer.bias.copy_(b.contiguous())
        new_layer.bias.requires_grad = True
    return new_layer


class MultiHeadAttention(nn.Module):
    NEW_ID = itertools.count()

    def __init__(self, n_heads, dim, config):
        super(MultiHeadAttention, self).__init__()
        self.layer_id = next(MultiHeadAttention.NEW_ID)
        self.output_attentions = config.output_attentions
        self.dim = dim
        self.n_heads = n_heads
        self.dropout = config.attention_dropout
        assert self.dim % self.n_heads == 0
        self.q_lin = nn.Linear(dim, dim)
        self.k_lin = nn.Linear(dim, dim)
        self.v_lin = nn.Linear(dim, dim)
        self.out_lin = nn.Linear(dim, dim)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        attention_head_size = self.dim // self.n_heads
        if len(heads) == 0:
            return
        mask = torch.ones(self.n_heads, attention_head_size)
        heads = set(heads) - self.pruned_heads
        for head in heads:
            head -= sum(1 if h < head else 0 for h in self.pruned_heads)
            mask[head] = 0
        mask = mask.view(-1).contiguous().eq(1)
        index = torch.arange(len(mask))[mask].long()
        self.q_lin = prune_linear_layer(self.q_lin, index)
        self.k_lin = prune_linear_layer(self.k_lin, index)
        self.v_lin = prune_linear_layer(self.v_lin, index)
        self.out_lin = prune_linear_layer(self.out_lin, index, dim=1)
        self.n_heads = self.n_heads - len(heads)
        self.dim = attention_head_size * self.n_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(self, input, mask, kv=None, cache=None, head_mask=None):
        """
        Self-attention (if kv is None) or attention over source sentence (provided by kv).
        """
        bs, qlen, dim = input.size()
        if kv is None:
            klen = qlen if cache is None else cache['slen'] + qlen
        else:
            klen = kv.size(1)
        n_heads = self.n_heads
        dim_per_head = self.dim // n_heads
        mask_reshape = (bs, 1, qlen, klen) if mask.dim() == 3 else (bs, 1, 1, klen)

        def shape(x):
            """  projection """
            return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)

        def unshape(x):
            """  compute context """
            return x.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * dim_per_head)
        q = shape(self.q_lin(input))
        if kv is None:
            k = shape(self.k_lin(input))
            v = shape(self.v_lin(input))
        elif cache is None or self.layer_id not in cache:
            k = v = kv
            k = shape(self.k_lin(k))
            v = shape(self.v_lin(v))
        if cache is not None:
            if self.layer_id in cache:
                if kv is None:
                    k_, v_ = cache[self.layer_id]
                    k = torch.cat([k_, k], dim=2)
                    v = torch.cat([v_, v], dim=2)
                else:
                    k, v = cache[self.layer_id]
            cache[self.layer_id] = k, v
        q = q / math.sqrt(dim_per_head)
        scores = torch.matmul(q, k.transpose(2, 3))
        mask = (mask == 0).view(mask_reshape).expand_as(scores)
        scores.masked_fill_(mask, -float('inf'))
        weights = F.softmax(scores.float(), dim=-1).type_as(scores)
        weights = F.dropout(weights, p=self.dropout, training=self.training)
        if head_mask is not None:
            weights = weights * head_mask
        context = torch.matmul(weights, v)
        context = unshape(context)
        outputs = self.out_lin(context),
        if self.output_attentions:
            outputs = outputs + (weights,)
        return outputs


def gelu(x):
    """ Implementation of the gelu activation function.
        XLNet is using OpenAI GPT's gelu (not exactly the same as BERT)
        Also see https://arxiv.org/abs/1606.08415
    """
    cdf = 0.5 * (1.0 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    return x * cdf


class TransformerFFN(nn.Module):

    def __init__(self, in_dim, dim_hidden, out_dim, config):
        super(TransformerFFN, self).__init__()
        self.dropout = config.dropout
        self.lin1 = nn.Linear(in_dim, dim_hidden)
        self.lin2 = nn.Linear(dim_hidden, out_dim)
        self.act = gelu if config.gelu_activation else F.relu

    def forward(self, input):
        x = self.lin1(input)
        x = self.act(x)
        x = self.lin2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class BeamHypotheses(object):

    def __init__(self, n_hyp, max_len, length_penalty, early_stopping):
        """
        Initialize n-best list of hypotheses.
        """
        self.max_len = max_len - 1
        self.length_penalty = length_penalty
        self.early_stopping = early_stopping
        self.n_hyp = n_hyp
        self.hyp = []
        self.worst_score = 1000000000.0

    def __len__(self):
        """
        Number of hypotheses in the list.
        """
        return len(self.hyp)

    def add(self, hyp, sum_logprobs):
        """
        Add a new hypothesis to the list.
        """
        score = sum_logprobs / len(hyp) ** self.length_penalty
        if len(self) < self.n_hyp or score > self.worst_score:
            self.hyp.append((score, hyp))
            if len(self) > self.n_hyp:
                sorted_scores = sorted([(s, idx) for idx, (s, _) in enumerate(self.hyp)])
                del self.hyp[sorted_scores[0][1]]
                self.worst_score = sorted_scores[1][0]
            else:
                self.worst_score = min(score, self.worst_score)

    def is_done(self, best_sum_logprobs):
        """
        If there are enough hypotheses and that none of the hypotheses being generated
        can become better than the worst one in the heap, then we are done with this sentence.
        """
        if len(self) < self.n_hyp:
            return False
        elif self.early_stopping:
            return True
        else:
            return self.worst_score >= best_sum_logprobs / self.max_len ** self.length_penalty


def Embedding(num_embeddings, embedding_dim, padding_idx=None):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    if padding_idx is not None:
        nn.init.constant_(m.weight[padding_idx], 0)
    return m


N_MAX_POSITIONS = 512


def create_sinusoidal_embeddings(n_pos, dim, out):
    position_enc = np.array([[(pos / np.power(10000, 2 * (j // 2) / dim)) for j in range(dim)] for pos in range(n_pos)])
    out[:, 0::2] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))
    out[:, 1::2] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))
    out.detach_()
    out.requires_grad = False


def get_masks(slen, lengths, causal, padding_mask=None):
    """
    Generate hidden states mask, and optionally an attention mask.
    """
    bs = lengths.size(0)
    if padding_mask is not None:
        mask = padding_mask
    else:
        assert lengths.max().item() <= slen
        alen = torch.arange(slen, dtype=torch.long, device=lengths.device)
        mask = alen < lengths[:, None]
    if causal:
        attn_mask = alen[None, None, :].repeat(bs, slen, 1) <= alen[None, :, None]
    else:
        attn_mask = mask
    assert mask.size() == (bs, slen)
    assert causal is False or attn_mask.size() == (bs, slen, slen)
    return mask, attn_mask


class TransformerModel(nn.Module):
    ATTRIBUTES = ['encoder', 'with_output', 'eos_index', 'pad_index', 'n_langs', 'n_words', 'dim', 'n_layers', 'n_heads', 'hidden_dim', 'dropout', 'attention_dropout', 'asm', 'asm_cutoffs', 'asm_div_value']

    def __init__(self, params, dico, is_encoder, with_output):
        """
        Transformer model (encoder or decoder).
        """
        super().__init__()
        self.is_encoder = is_encoder
        self.is_decoder = not is_encoder
        self.with_output = with_output
        self.n_langs = params.n_langs
        self.n_words = params.n_words
        self.eos_index = params.eos_index
        self.pad_index = params.pad_index
        self.dico = dico
        self.id2lang = params.id2lang
        self.lang2id = params.lang2id
        self.use_lang_emb = getattr(params, 'use_lang_emb', True)
        assert len(self.dico) == self.n_words
        assert len(self.id2lang) == len(self.lang2id) == self.n_langs
        self.dim = params.emb_dim
        self.hidden_dim = self.dim * 4
        self.n_heads = params.n_heads
        self.n_layers = params.n_layers
        self.dropout = params.dropout
        self.attention_dropout = params.attention_dropout
        assert self.dim % self.n_heads == 0, 'transformer dim must be a multiple of n_heads'
        self.position_embeddings = Embedding(N_MAX_POSITIONS, self.dim)
        if params.sinusoidal_embeddings:
            create_sinusoidal_embeddings(N_MAX_POSITIONS, self.dim, out=self.position_embeddings.weight)
        if params.n_langs > 1 and self.use_lang_emb:
            self.lang_embeddings = Embedding(self.n_langs, self.dim)
        self.embeddings = Embedding(self.n_words, self.dim, padding_idx=self.pad_index)
        self.layer_norm_emb = nn.LayerNorm(self.dim, eps=1e-12)
        self.attentions = nn.ModuleList()
        self.layer_norm1 = nn.ModuleList()
        self.ffns = nn.ModuleList()
        self.layer_norm2 = nn.ModuleList()
        if self.is_decoder:
            self.layer_norm15 = nn.ModuleList()
            self.encoder_attn = nn.ModuleList()
        self.memories = nn.ModuleDict()
        if getattr(params, 'use_memory', False):
            mem_positions = params.mem_enc_positions if is_encoder else params.mem_dec_positions
            for layer_id, pos in mem_positions:
                assert 0 <= layer_id <= params.n_layers - 1
                assert pos in ['in', 'after']
                self.memories['%i_%s' % (layer_id, pos)] = HashingMemory.build(self.dim, self.dim, params)
        for layer_id in range(self.n_layers):
            self.attentions.append(MultiHeadAttention(self.n_heads, self.dim, dropout=self.attention_dropout))
            self.layer_norm1.append(nn.LayerNorm(self.dim, eps=1e-12))
            if self.is_decoder:
                self.layer_norm15.append(nn.LayerNorm(self.dim, eps=1e-12))
                self.encoder_attn.append(MultiHeadAttention(self.n_heads, self.dim, dropout=self.attention_dropout))
            if '%i_in' % layer_id in self.memories:
                self.ffns.append(None)
            else:
                self.ffns.append(TransformerFFN(self.dim, self.hidden_dim, self.dim, dropout=self.dropout, gelu_activation=params.gelu_activation))
            self.layer_norm2.append(nn.LayerNorm(self.dim, eps=1e-12))
        if self.with_output:
            self.pred_layer = PredLayer(params)
            if params.share_inout_emb:
                self.pred_layer.proj.weight = self.embeddings.weight

    def forward(self, mode, **kwargs):
        """
        Forward function with different forward modes.
        ### Small hack to handle PyTorch distributed.
        """
        if mode == 'fwd':
            return self.fwd(**kwargs)
        elif mode == 'predict':
            return self.predict(**kwargs)
        else:
            raise Exception('Unknown mode: %s' % mode)

    def fwd(self, x, lengths, causal, src_enc=None, src_len=None, positions=None, langs=None, cache=None):
        """
        Inputs:
            `x` LongTensor(slen, bs), containing word indices
            `lengths` LongTensor(bs), containing the length of each sentence
            `causal` Boolean, if True, the attention is only done over previous hidden states
            `positions` LongTensor(slen, bs), containing word positions
            `langs` LongTensor(slen, bs), containing language IDs
        """
        slen, bs = x.size()
        assert lengths.size(0) == bs
        assert lengths.max().item() <= slen
        x = x.transpose(0, 1)
        assert (src_enc is None) == (src_len is None)
        if src_enc is not None:
            assert self.is_decoder
            assert src_enc.size(0) == bs
        mask, attn_mask = get_masks(slen, lengths, causal)
        if self.is_decoder and src_enc is not None:
            src_mask = torch.arange(src_len.max(), dtype=torch.long, device=lengths.device) < src_len[:, None]
        if positions is None:
            positions = x.new(slen).long()
            positions = torch.arange(slen, out=positions).unsqueeze(0)
        else:
            assert positions.size() == (slen, bs)
            positions = positions.transpose(0, 1)
        if langs is not None:
            assert langs.size() == (slen, bs)
            langs = langs.transpose(0, 1)
        if cache is not None:
            _slen = slen - cache['slen']
            x = x[:, -_slen:]
            positions = positions[:, -_slen:]
            if langs is not None:
                langs = langs[:, -_slen:]
            mask = mask[:, -_slen:]
            attn_mask = attn_mask[:, -_slen:]
        tensor = self.embeddings(x)
        tensor = tensor + self.position_embeddings(positions).expand_as(tensor)
        if langs is not None and self.use_lang_emb:
            tensor = tensor + self.lang_embeddings(langs)
        tensor = self.layer_norm_emb(tensor)
        tensor = F.dropout(tensor, p=self.dropout, training=self.training)
        tensor *= mask.unsqueeze(-1)
        for i in range(self.n_layers):
            attn = self.attentions[i](tensor, attn_mask, cache=cache)
            attn = F.dropout(attn, p=self.dropout, training=self.training)
            tensor = tensor + attn
            tensor = self.layer_norm1[i](tensor)
            if self.is_decoder and src_enc is not None:
                attn = self.encoder_attn[i](tensor, src_mask, kv=src_enc, cache=cache)
                attn = F.dropout(attn, p=self.dropout, training=self.training)
                tensor = tensor + attn
                tensor = self.layer_norm15[i](tensor)
            if '%i_in' % i in self.memories:
                tensor = tensor + self.memories['%i_in' % i](tensor)
            else:
                tensor = tensor + self.ffns[i](tensor)
            tensor = self.layer_norm2[i](tensor)
            if '%i_after' % i in self.memories:
                tensor = tensor + self.memories['%i_after' % i](tensor)
            tensor *= mask.unsqueeze(-1)
        if cache is not None:
            cache['slen'] += tensor.size(1)
        tensor = tensor.transpose(0, 1)
        return tensor

    def predict(self, tensor, pred_mask, y, get_scores):
        """
        Given the last hidden state, compute word scores and/or the loss.
            `pred_mask` is a ByteTensor of shape (slen, bs), filled with 1 when
                we need to predict a word
            `y` is a LongTensor of shape (pred_mask.sum(),)
            `get_scores` is a boolean specifying whether we need to return scores
        """
        masked_tensor = tensor[pred_mask.unsqueeze(-1).expand_as(tensor)].view(-1, self.dim)
        scores, loss = self.pred_layer(masked_tensor, y, get_scores)
        return scores, loss

    def generate(self, src_enc, src_len, tgt_lang_id, max_len=200, sample_temperature=None):
        """
        Decode a sentence given initial start.
        `x`:
            - LongTensor(bs, slen)
                <EOS> W1 W2 W3 <EOS> <PAD>
                <EOS> W1 W2 W3   W4  <EOS>
        `lengths`:
            - LongTensor(bs) [5, 6]
        `positions`:
            - False, for regular "arange" positions (LM)
            - True, to reset positions from the new generation (MT)
        `langs`:
            - must be None if the model only supports one language
            - lang_id if only one language is involved (LM)
            - (lang_id1, lang_id2) if two languages are involved (MT)
        """
        bs = len(src_len)
        assert src_enc.size(0) == bs
        generated = src_len.new(max_len, bs)
        generated.fill_(self.pad_index)
        generated[0].fill_(self.eos_index)
        positions = src_len.new(max_len).long()
        positions = torch.arange(max_len, out=positions).unsqueeze(1).expand(max_len, bs)
        langs = src_len.new(max_len).long().fill_(tgt_lang_id)
        langs = langs.unsqueeze(1).expand(max_len, bs)
        cur_len = 1
        gen_len = src_len.clone().fill_(1)
        unfinished_sents = src_len.clone().fill_(1)
        cache = {'slen': 0}
        while cur_len < max_len:
            tensor = self.forward('fwd', x=generated[:cur_len], lengths=gen_len, positions=positions[:cur_len], langs=langs[:cur_len], causal=True, src_enc=src_enc, src_len=src_len, cache=cache)
            assert tensor.size() == (1, bs, self.dim), (cur_len, max_len, src_enc.size(), tensor.size(), (1, bs, self.dim))
            tensor = tensor.data[-1, :, :].type_as(src_enc)
            scores = self.pred_layer.get_scores(tensor)
            if sample_temperature is None:
                next_words = torch.topk(scores, 1)[1].squeeze(1)
            else:
                next_words = torch.multinomial(F.softmax(scores / sample_temperature, dim=1), 1).squeeze(1)
            assert next_words.size() == (bs,)
            generated[cur_len] = next_words * unfinished_sents + self.pad_index * (1 - unfinished_sents)
            gen_len.add_(unfinished_sents)
            unfinished_sents.mul_(next_words.ne(self.eos_index).long())
            cur_len = cur_len + 1
            if unfinished_sents.max() == 0:
                break
        if cur_len == max_len:
            generated[-1].masked_fill_(unfinished_sents.byte(), self.eos_index)
        assert (generated == self.eos_index).sum() == 2 * bs
        return generated[:cur_len], gen_len

    def generate_beam(self, src_enc, src_len, tgt_lang_id, beam_size, length_penalty, early_stopping, max_len=200, output_all_hyps=False):
        """
        Decode a sentence given initial start.
        `x`:
            - LongTensor(bs, slen)
                <EOS> W1 W2 W3 <EOS> <PAD>
                <EOS> W1 W2 W3   W4  <EOS>
        `lengths`:
            - LongTensor(bs) [5, 6]
        `positions`:
            - False, for regular "arange" positions (LM)
            - True, to reset positions from the new generation (MT)
        `langs`:
            - must be None if the model only supports one language
            - lang_id if only one language is involved (LM)
            - (lang_id1, lang_id2) if two languages are involved (MT)
        """
        assert src_enc.size(0) == src_len.size(0)
        assert beam_size >= 1
        bs = len(src_len)
        n_words = self.n_words
        src_enc = src_enc.unsqueeze(1).expand((bs, beam_size) + src_enc.shape[1:]).contiguous().view((bs * beam_size,) + src_enc.shape[1:])
        src_len = src_len.unsqueeze(1).expand(bs, beam_size).contiguous().view(-1)
        generated = src_len.new(max_len, bs * beam_size)
        generated.fill_(self.pad_index)
        generated[0].fill_(self.eos_index)
        generated_hyps = [BeamHypotheses(beam_size, max_len, length_penalty, early_stopping) for _ in range(bs)]
        positions = src_len.new(max_len).long()
        positions = torch.arange(max_len, out=positions).unsqueeze(1).expand_as(generated)
        langs = positions.clone().fill_(tgt_lang_id)
        beam_scores = src_enc.new(bs, beam_size).fill_(0)
        beam_scores[:, 1:] = -1000000000.0
        beam_scores = beam_scores.view(-1)
        cur_len = 1
        cache = {'slen': 0}
        done = [(False) for _ in range(bs)]
        while cur_len < max_len:
            tensor = self.forward('fwd', x=generated[:cur_len], lengths=src_len.new(bs * beam_size).fill_(cur_len), positions=positions[:cur_len], langs=langs[:cur_len], causal=True, src_enc=src_enc, src_len=src_len, cache=cache)
            assert tensor.size() == (1, bs * beam_size, self.dim)
            tensor = tensor.data[-1, :, :]
            scores = self.pred_layer.get_scores(tensor)
            scores = F.log_softmax(scores, dim=-1)
            assert scores.size() == (bs * beam_size, n_words)
            _scores = scores + beam_scores[:, None].expand_as(scores)
            _scores = _scores.view(bs, beam_size * n_words)
            next_scores, next_words = torch.topk(_scores, 2 * beam_size, dim=1, largest=True, sorted=True)
            assert next_scores.size() == next_words.size() == (bs, 2 * beam_size)
            next_batch_beam = []
            for sent_id in range(bs):
                done[sent_id] = done[sent_id] or generated_hyps[sent_id].is_done(next_scores[sent_id].max().item())
                if done[sent_id]:
                    next_batch_beam.extend([(0, self.pad_index, 0)] * beam_size)
                    continue
                next_sent_beam = []
                for idx, value in zip(next_words[sent_id], next_scores[sent_id]):
                    beam_id = idx // n_words
                    word_id = idx % n_words
                    if word_id == self.eos_index or cur_len + 1 == max_len:
                        generated_hyps[sent_id].add(generated[:cur_len, sent_id * beam_size + beam_id].clone(), value.item())
                    else:
                        next_sent_beam.append((value, word_id, sent_id * beam_size + beam_id))
                    if len(next_sent_beam) == beam_size:
                        break
                assert len(next_sent_beam) == 0 if cur_len + 1 == max_len else beam_size
                if len(next_sent_beam) == 0:
                    next_sent_beam = [(0, self.pad_index, 0)] * beam_size
                next_batch_beam.extend(next_sent_beam)
                assert len(next_batch_beam) == beam_size * (sent_id + 1)
            assert len(next_batch_beam) == bs * beam_size
            beam_scores = beam_scores.new([x[0] for x in next_batch_beam])
            beam_words = generated.new([x[1] for x in next_batch_beam])
            beam_idx = src_len.new([x[2] for x in next_batch_beam])
            generated = generated[:, beam_idx]
            generated[cur_len] = beam_words
            for k in cache.keys():
                if k != 'slen':
                    cache[k] = cache[k][0][beam_idx], cache[k][1][beam_idx]
            cur_len = cur_len + 1
            if all(done):
                break
        if output_all_hyps:
            generated_hyp_strs = [[] for _ in range(bs)]
            for hyp_rank in range(bs):
                for ss, ww in sorted(generated_hyps[hyp_rank].hyp, key=lambda x: x[0], reverse=True):
                    generated_hyp_strs[hyp_rank].append(' '.join(self.dico[x] for x in ww.tolist()[1:]).replace('<unk>', '<<unk>>'))
        tgt_len = src_len.new(bs)
        best = []
        for i, hypotheses in enumerate(generated_hyps):
            best_hyp = max(hypotheses.hyp, key=lambda x: x[0])[1]
            tgt_len[i] = len(best_hyp) + 1
            best.append(best_hyp)
        decoded = src_len.new(tgt_len.max().item(), bs).fill_(self.pad_index)
        for i, hypo in enumerate(best):
            decoded[:tgt_len[i] - 1, i] = hypo
            decoded[tgt_len[i] - 1, i] = self.eos_index
        assert (decoded == self.eos_index).sum() == 2 * bs
        if output_all_hyps:
            return decoded, tgt_len, generated_hyp_strs
        else:
            return decoded, tgt_len


class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """

    def __init__(self, config):
        super(BertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None, position_ids=None):
        seq_length = input_ids.size(1)
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertSelfAttention(nn.Module):

    def __init__(self, config):
        super(BertSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError('The hidden size (%d) is not a multiple of the number of attention heads (%d)' % (config.hidden_size, config.num_attention_heads))
        self.output_attentions = config.output_attentions
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask, head_mask=None):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_scores = attention_scores + attention_mask
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)
        if head_mask is not None:
            attention_probs = attention_probs * head_mask
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        outputs = (context_layer, attention_probs) if self.output_attentions else (context_layer,)
        return outputs


class BertSelfOutput(nn.Module):

    def __init__(self, config):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):

    def __init__(self, config):
        super(BertAttention, self).__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        mask = torch.ones(self.self.num_attention_heads, self.self.attention_head_size)
        heads = set(heads) - self.pruned_heads
        for head in heads:
            head = head - sum(1 if h < head else 0 for h in self.pruned_heads)
            mask[head] = 0
        mask = mask.view(-1).contiguous().eq(1)
        index = torch.arange(len(mask))[mask].long()
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(self, input_tensor, attention_mask, head_mask=None):
        self_outputs = self.self(input_tensor, attention_mask, head_mask)
        attention_output = self.output(self_outputs[0], input_tensor)
        outputs = (attention_output,) + self_outputs[1:]
        return outputs


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {'gelu': gelu, 'relu': torch.nn.functional.relu, 'swish': swish}


class BertIntermediate(nn.Module):

    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str) or sys.version_info[0] == 2 and isinstance(config.hidden_act, unicode):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):

    def __init__(self, config):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):

    def __init__(self, config):
        super(BertLayer, self).__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask, head_mask=None):
        attention_outputs = self.attention(hidden_states, attention_mask, head_mask)
        attention_output = attention_outputs[0]
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output,) + attention_outputs[1:]
        return outputs


class BertEncoder(nn.Module):

    def __init__(self, config):
        super(BertEncoder, self).__init__()
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask, head_mask=None):
        all_hidden_states = ()
        all_attentions = ()
        for i, layer_module in enumerate(self.layer):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            layer_outputs = layer_module(hidden_states, attention_mask, head_mask[i])
            hidden_states = layer_outputs[0]
            if self.output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        outputs = hidden_states,
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs


class BertPooler(nn.Module):

    def __init__(self, config):
        super(BertPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertPredictionHeadTransform(nn.Module):

    def __init__(self, config):
        super(BertPredictionHeadTransform, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str) or sys.version_info[0] == 2 and isinstance(config.hidden_act, unicode):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertLMPredictionHead(nn.Module):

    def __init__(self, config):
        super(BertLMPredictionHead, self).__init__()
        self.transform = BertPredictionHeadTransform(config)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states) + self.bias
        return hidden_states


class BertOnlyMLMHead(nn.Module):

    def __init__(self, config):
        super(BertOnlyMLMHead, self).__init__()
        self.predictions = BertLMPredictionHead(config)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class BertOnlyNSPHead(nn.Module):

    def __init__(self, config):
        super(BertOnlyNSPHead, self).__init__()
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, pooled_output):
        seq_relationship_score = self.seq_relationship(pooled_output)
        return seq_relationship_score


class BertPreTrainingHeads(nn.Module):

    def __init__(self, config):
        super(BertPreTrainingHeads, self).__init__()
        self.predictions = BertLMPredictionHead(config)
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, sequence_output, pooled_output):
        prediction_scores = self.predictions(sequence_output)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score


class Embeddings(nn.Module):

    def __init__(self, config):
        super(Embeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.dim, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.dim)
        if config.sinusoidal_pos_embds:
            create_sinusoidal_embeddings(n_pos=config.max_position_embeddings, dim=config.dim, out=self.position_embeddings.weight)
        self.LayerNorm = nn.LayerNorm(config.dim, eps=1e-12)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, input_ids):
        """
        Parameters
        ----------
        input_ids: torch.tensor(bs, max_seq_length)
            The token ids to embed.

        Outputs
        -------
        embeddings: torch.tensor(bs, max_seq_length, dim)
            The embedded tokens (plus position embeddings, no token_type embeddings)
        """
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        word_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = word_embeddings + position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class MultiHeadSelfAttention(nn.Module):

    def __init__(self, config):
        super(MultiHeadSelfAttention, self).__init__()
        self.n_heads = config.n_heads
        self.dim = config.dim
        self.dropout = nn.Dropout(p=config.attention_dropout)
        self.output_attentions = config.output_attentions
        assert self.dim % self.n_heads == 0
        self.q_lin = nn.Linear(in_features=config.dim, out_features=config.dim)
        self.k_lin = nn.Linear(in_features=config.dim, out_features=config.dim)
        self.v_lin = nn.Linear(in_features=config.dim, out_features=config.dim)
        self.out_lin = nn.Linear(in_features=config.dim, out_features=config.dim)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        attention_head_size = self.dim // self.n_heads
        if len(heads) == 0:
            return
        mask = torch.ones(self.n_heads, attention_head_size)
        heads = set(heads) - self.pruned_heads
        for head in heads:
            head -= sum(1 if h < head else 0 for h in self.pruned_heads)
            mask[head] = 0
        mask = mask.view(-1).contiguous().eq(1)
        index = torch.arange(len(mask))[mask].long()
        self.q_lin = prune_linear_layer(self.q_lin, index)
        self.k_lin = prune_linear_layer(self.k_lin, index)
        self.v_lin = prune_linear_layer(self.v_lin, index)
        self.out_lin = prune_linear_layer(self.out_lin, index, dim=1)
        self.n_heads = self.n_heads - len(heads)
        self.dim = attention_head_size * self.n_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(self, query, key, value, mask, head_mask=None):
        """
        Parameters
        ----------
        query: torch.tensor(bs, seq_length, dim)
        key: torch.tensor(bs, seq_length, dim)
        value: torch.tensor(bs, seq_length, dim)
        mask: torch.tensor(bs, seq_length)

        Outputs
        -------
        weights: torch.tensor(bs, n_heads, seq_length, seq_length)
            Attention weights
        context: torch.tensor(bs, seq_length, dim)
            Contextualized layer. Optional: only if `output_attentions=True`
        """
        bs, q_length, dim = query.size()
        k_length = key.size(1)
        dim_per_head = self.dim // self.n_heads
        assert 2 <= mask.dim() <= 3
        causal = mask.dim() == 3
        mask_reshp = bs, 1, 1, k_length

        def shape(x):
            """ separate heads """
            return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)

        def unshape(x):
            """ group heads """
            return x.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * dim_per_head)
        q = shape(self.q_lin(query))
        k = shape(self.k_lin(key))
        v = shape(self.v_lin(value))
        q = q / math.sqrt(dim_per_head)
        scores = torch.matmul(q, k.transpose(2, 3))
        mask = (mask == 0).view(mask_reshp).expand_as(scores)
        scores.masked_fill_(mask, -float('inf'))
        weights = nn.Softmax(dim=-1)(scores)
        weights = self.dropout(weights)
        if head_mask is not None:
            weights = weights * head_mask
        context = torch.matmul(weights, v)
        context = unshape(context)
        context = self.out_lin(context)
        if self.output_attentions:
            return context, weights
        else:
            return context,


class FFN(nn.Module):

    def __init__(self, config):
        super(FFN, self).__init__()
        self.dropout = nn.Dropout(p=config.dropout)
        self.lin1 = nn.Linear(in_features=config.dim, out_features=config.hidden_dim)
        self.lin2 = nn.Linear(in_features=config.hidden_dim, out_features=config.dim)
        assert config.activation in ['relu', 'gelu'], "activation ({}) must be in ['relu', 'gelu']".format(config.activation)
        self.activation = gelu if config.activation == 'gelu' else nn.ReLU()

    def forward(self, input):
        x = self.lin1(input)
        x = self.activation(x)
        x = self.lin2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):

    def __init__(self, config):
        super(TransformerBlock, self).__init__()
        self.n_heads = config.n_heads
        self.dim = config.dim
        self.hidden_dim = config.hidden_dim
        self.dropout = nn.Dropout(p=config.dropout)
        self.activation = config.activation
        self.output_attentions = config.output_attentions
        assert config.dim % config.n_heads == 0
        self.attention = MultiHeadSelfAttention(config)
        self.sa_layer_norm = nn.LayerNorm(normalized_shape=config.dim, eps=1e-12)
        self.ffn = FFN(config)
        self.output_layer_norm = nn.LayerNorm(normalized_shape=config.dim, eps=1e-12)

    def forward(self, x, attn_mask=None, head_mask=None):
        """
        Parameters
        ----------
        x: torch.tensor(bs, seq_length, dim)
        attn_mask: torch.tensor(bs, seq_length)

        Outputs
        -------
        sa_weights: torch.tensor(bs, n_heads, seq_length, seq_length)
            The attention weights
        ffn_output: torch.tensor(bs, seq_length, dim)
            The output of the transformer block contextualization.
        """
        sa_output = self.attention(query=x, key=x, value=x, mask=attn_mask, head_mask=head_mask)
        if self.output_attentions:
            sa_output, sa_weights = sa_output
        else:
            assert type(sa_output) == tuple
            sa_output = sa_output[0]
        sa_output = self.sa_layer_norm(sa_output + x)
        ffn_output = self.ffn(sa_output)
        ffn_output = self.output_layer_norm(ffn_output + sa_output)
        output = ffn_output,
        if self.output_attentions:
            output = (sa_weights,) + output
        return output


class Transformer(nn.Module):

    def __init__(self, config):
        super(Transformer, self).__init__()
        self.n_layers = config.n_layers
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        layer = TransformerBlock(config)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(config.n_layers)])

    def forward(self, x, attn_mask=None, head_mask=None):
        """
        Parameters
        ----------
        x: torch.tensor(bs, seq_length, dim)
            Input sequence embedded.
        attn_mask: torch.tensor(bs, seq_length)
            Attention mask on the sequence.

        Outputs
        -------
        hidden_state: torch.tensor(bs, seq_length, dim)
            Sequence of hiddens states in the last (top) layer
        all_hidden_states: Tuple[torch.tensor(bs, seq_length, dim)]
            Tuple of length n_layers with the hidden states from each layer.
            Optional: only if output_hidden_states=True
        all_attentions: Tuple[torch.tensor(bs, n_heads, seq_length, seq_length)]
            Tuple of length n_layers with the attention weights from each layer
            Optional: only if output_attentions=True
        """
        all_hidden_states = ()
        all_attentions = ()
        hidden_state = x
        for i, layer_module in enumerate(self.layer):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_state,)
            layer_outputs = layer_module(x=hidden_state, attn_mask=attn_mask, head_mask=head_mask[i])
            hidden_state = layer_outputs[-1]
            if self.output_attentions:
                assert len(layer_outputs) == 2
                attentions = layer_outputs[0]
                all_attentions = all_attentions + (attentions,)
            else:
                assert len(layer_outputs) == 1
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_state,)
        outputs = hidden_state,
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs


class Conv1D(nn.Module):

    def __init__(self, nf, nx):
        """ Conv1D layer as defined by Radford et al. for OpenAI GPT (and also used in GPT-2)
            Basically works like a Linear layer but the weights are transposed
        """
        super(Conv1D, self).__init__()
        self.nf = nf
        w = torch.empty(nx, nf)
        nn.init.normal_(w, std=0.02)
        self.weight = nn.Parameter(w)
        self.bias = nn.Parameter(torch.zeros(nf))

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        x = x.view(*size_out)
        return x


def prune_conv1d_layer(layer, index, dim=1):
    """ Prune a Conv1D layer (a model parameters) to keep only entries in index.
        A Conv1D work as a Linear layer (see e.g. BERT) but the weights are transposed.
        Return the pruned layer as a new layer with requires_grad=True.
        Used to remove heads.
    """
    index = index
    W = layer.weight.index_select(dim, index).clone().detach()
    if dim == 0:
        b = layer.bias.clone().detach()
    else:
        b = layer.bias[index].clone().detach()
    new_size = list(layer.weight.size())
    new_size[dim] = len(index)
    new_layer = Conv1D(new_size[1], new_size[0])
    new_layer.weight.requires_grad = False
    new_layer.weight.copy_(W.contiguous())
    new_layer.weight.requires_grad = True
    new_layer.bias.requires_grad = False
    new_layer.bias.copy_(b.contiguous())
    new_layer.bias.requires_grad = True
    return new_layer


class Attention(nn.Module):

    def __init__(self, nx, n_ctx, config, scale=False):
        super(Attention, self).__init__()
        n_state = nx
        assert n_state % config.n_head == 0
        self.register_buffer('bias', torch.tril(torch.ones(n_ctx, n_ctx)).view(1, 1, n_ctx, n_ctx))
        self.n_head = config.n_head
        self.split_size = n_state
        self.scale = scale
        self.output_attentions = config.output_attentions
        self.c_attn = Conv1D(n_state * 3, nx)
        self.c_proj = Conv1D(n_state, nx)
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        mask = torch.ones(self.n_head, self.split_size // self.n_head)
        heads = set(heads) - self.pruned_heads
        for head in heads:
            head -= sum(1 if h < head else 0 for h in self.pruned_heads)
            mask[head] = 0
        mask = mask.view(-1).contiguous().eq(1)
        index = torch.arange(len(mask))[mask].long()
        index_attn = torch.cat([index, index + self.split_size, index + 2 * self.split_size])
        self.c_attn = prune_conv1d_layer(self.c_attn, index_attn, dim=1)
        self.c_proj = prune_conv1d_layer(self.c_proj, index, dim=0)
        self.split_size = self.split_size // self.n_head * (self.n_head - len(heads))
        self.n_head = self.n_head - len(heads)
        self.pruned_heads = self.pruned_heads.union(heads)

    def _attn(self, q, k, v, head_mask=None):
        w = torch.matmul(q, k)
        if self.scale:
            w = w / math.sqrt(v.size(-1))
        b = self.bias[:, :, :w.size(-2), :w.size(-1)]
        w = w * b + -1000000000.0 * (1 - b)
        w = nn.Softmax(dim=-1)(w)
        w = self.attn_dropout(w)
        if head_mask is not None:
            w = w * head_mask
        outputs = [torch.matmul(w, v)]
        if self.output_attentions:
            outputs.append(w)
        return outputs

    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_x_shape)

    def split_heads(self, x, k=False):
        new_x_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)
        x = x.view(*new_x_shape)
        if k:
            return x.permute(0, 2, 3, 1)
        else:
            return x.permute(0, 2, 1, 3)

    def forward(self, x, head_mask=None):
        x = self.c_attn(x)
        query, key, value = x.split(self.split_size, dim=2)
        query = self.split_heads(query)
        key = self.split_heads(key, k=True)
        value = self.split_heads(value)
        attn_outputs = self._attn(query, key, value, head_mask)
        a = attn_outputs[0]
        a = self.merge_heads(a)
        a = self.c_proj(a)
        a = self.resid_dropout(a)
        outputs = [a] + attn_outputs[1:]
        return outputs


ACT_FNS = {'relu': nn.ReLU, 'swish': swish, 'gelu': gelu}


class MLP(nn.Module):

    def __init__(self, n_state, config):
        super(MLP, self).__init__()
        nx = config.n_embd
        self.c_fc = Conv1D(n_state, nx)
        self.c_proj = Conv1D(nx, n_state)
        self.act = ACT_FNS[config.afn]
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, x):
        h = self.act(self.c_fc(x))
        h2 = self.c_proj(h)
        return self.dropout(h2)


class Block(nn.Module):

    def __init__(self, n_ctx, config, scale=False):
        super(Block, self).__init__()
        nx = config.n_embd
        self.attn = Attention(nx, n_ctx, config, scale)
        self.ln_1 = LayerNorm(nx, eps=config.layer_norm_epsilon)
        self.mlp = MLP(4 * nx, config)
        self.ln_2 = LayerNorm(nx, eps=config.layer_norm_epsilon)

    def forward(self, x, head_mask=None):
        attn_outputs = self.attn(x, head_mask=head_mask)
        a = attn_outputs[0]
        n = self.ln_1(x + a)
        m = self.mlp(n)
        h = self.ln_2(n + m)
        outputs = [h] + attn_outputs[1:]
        return outputs


class RobertaEmbeddings(BertEmbeddings):
    """
    Same as BertEmbeddings with a tiny tweak for positional embeddings indexing.
    """

    def __init__(self, config):
        super(RobertaEmbeddings, self).__init__(config)
        self.padding_idx = 1

    def forward(self, input_ids, token_type_ids=None, position_ids=None):
        seq_length = input_ids.size(1)
        if position_ids is None:
            position_ids = torch.arange(self.padding_idx + 1, seq_length + self.padding_idx + 1, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        return super(RobertaEmbeddings, self).forward(input_ids, token_type_ids=token_type_ids, position_ids=position_ids)


class RobertaLMHead(nn.Module):
    """Roberta Head for masked language modeling."""

    def __init__(self, config):
        super(RobertaLMHead, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = gelu(x)
        x = self.layer_norm(x)
        x = self.decoder(x) + self.bias
        return x


class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super(RobertaClassificationHead, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class PositionalEmbedding(nn.Module):

    def __init__(self, demb):
        super(PositionalEmbedding, self).__init__()
        self.demb = demb
        inv_freq = 1 / 10000 ** (torch.arange(0.0, demb, 2.0) / demb)
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, pos_seq, bsz=None):
        sinusoid_inp = torch.ger(pos_seq, self.inv_freq)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)
        if bsz is not None:
            return pos_emb[:, None, :].expand(-1, bsz, -1)
        else:
            return pos_emb[:, None, :]


class PositionwiseFF(nn.Module):

    def __init__(self, d_model, d_inner, dropout, pre_lnorm=False):
        super(PositionwiseFF, self).__init__()
        self.d_model = d_model
        self.d_inner = d_inner
        self.dropout = dropout
        self.CoreNet = nn.Sequential(nn.Linear(d_model, d_inner), nn.ReLU(inplace=True), nn.Dropout(dropout), nn.Linear(d_inner, d_model), nn.Dropout(dropout))
        self.layer_norm = LayerNorm(d_model)
        self.pre_lnorm = pre_lnorm

    def forward(self, inp):
        if self.pre_lnorm:
            core_out = self.CoreNet(self.layer_norm(inp))
            output = core_out + inp
        else:
            core_out = self.CoreNet(inp)
            output = self.layer_norm(inp + core_out)
        return output


class MultiHeadAttn(nn.Module):

    def __init__(self, n_head, d_model, d_head, dropout, dropatt=0, pre_lnorm=False, r_r_bias=None, r_w_bias=None, output_attentions=False):
        super(MultiHeadAttn, self).__init__()
        self.output_attentions = output_attentions
        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.dropout = dropout
        self.q_net = nn.Linear(d_model, n_head * d_head, bias=False)
        self.kv_net = nn.Linear(d_model, 2 * n_head * d_head, bias=False)
        self.drop = nn.Dropout(dropout)
        self.dropatt = nn.Dropout(dropatt)
        self.o_net = nn.Linear(n_head * d_head, d_model, bias=False)
        self.layer_norm = LayerNorm(d_model)
        self.scale = 1 / d_head ** 0.5
        self.pre_lnorm = pre_lnorm
        if r_r_bias is None or r_w_bias is None:
            self.r_r_bias = nn.Parameter(torch.FloatTensor(self.n_head, self.d_head))
            self.r_w_bias = nn.Parameter(torch.FloatTensor(self.n_head, self.d_head))
        else:
            self.r_r_bias = r_r_bias
            self.r_w_bias = r_w_bias

    def forward(self, h, attn_mask=None, mems=None, head_mask=None):
        if mems is not None:
            c = torch.cat([mems, h], 0)
        else:
            c = h
        if self.pre_lnorm:
            c = self.layer_norm(c)
        head_q = self.q_net(h)
        head_k, head_v = torch.chunk(self.kv_net(c), 2, -1)
        head_q = head_q.view(h.size(0), h.size(1), self.n_head, self.d_head)
        head_k = head_k.view(c.size(0), c.size(1), self.n_head, self.d_head)
        head_v = head_v.view(c.size(0), c.size(1), self.n_head, self.d_head)
        attn_score = torch.einsum('ibnd,jbnd->ijbn', (head_q, head_k))
        attn_score.mul_(self.scale)
        if attn_mask is not None and attn_mask.any().item():
            if attn_mask.dim() == 2:
                attn_score.masked_fill_(attn_mask[None, :, :, None], -float('inf'))
            elif attn_mask.dim() == 3:
                attn_score.masked_fill_(attn_mask[:, :, :, None], -float('inf'))
        attn_prob = F.softmax(attn_score, dim=1)
        attn_prob = self.dropatt(attn_prob)
        if head_mask is not None:
            attn_prob = attn_prob * head_mask
        attn_vec = torch.einsum('ijbn,jbnd->ibnd', (attn_prob, head_v))
        attn_vec = attn_vec.contiguous().view(attn_vec.size(0), attn_vec.size(1), self.n_head * self.d_head)
        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)
        if self.pre_lnorm:
            outputs = [h + attn_out]
        else:
            outputs = [self.layer_norm(h + attn_out)]
        if self.output_attentions:
            outputs.append(attn_prob)
        return outputs


class RelMultiHeadAttn(nn.Module):

    def __init__(self, n_head, d_model, d_head, dropout, dropatt=0, tgt_len=None, ext_len=None, mem_len=None, pre_lnorm=False, r_r_bias=None, r_w_bias=None, output_attentions=False):
        super(RelMultiHeadAttn, self).__init__()
        self.output_attentions = output_attentions
        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.dropout = dropout
        self.qkv_net = nn.Linear(d_model, 3 * n_head * d_head, bias=False)
        self.drop = nn.Dropout(dropout)
        self.dropatt = nn.Dropout(dropatt)
        self.o_net = nn.Linear(n_head * d_head, d_model, bias=False)
        self.layer_norm = LayerNorm(d_model)
        self.scale = 1 / d_head ** 0.5
        self.pre_lnorm = pre_lnorm
        if r_r_bias is None or r_w_bias is None:
            self.r_r_bias = nn.Parameter(torch.FloatTensor(self.n_head, self.d_head))
            self.r_w_bias = nn.Parameter(torch.FloatTensor(self.n_head, self.d_head))
        else:
            self.r_r_bias = r_r_bias
            self.r_w_bias = r_w_bias

    def _parallelogram_mask(self, h, w, left=False):
        mask = torch.ones((h, w)).byte()
        m = min(h, w)
        mask[:m, :m] = torch.triu(mask[:m, :m])
        mask[-m:, -m:] = torch.tril(mask[-m:, -m:])
        if left:
            return mask
        else:
            return mask.flip(0)

    def _shift(self, x, qlen, klen, mask, left=False):
        if qlen > 1:
            zero_pad = torch.zeros((x.size(0), qlen - 1, x.size(2), x.size(3)), device=x.device, dtype=x.dtype)
        else:
            zero_pad = torch.zeros(0, device=x.device, dtype=x.dtype)
        if left:
            mask = mask.flip(1)
            x_padded = torch.cat([zero_pad, x], dim=1).expand(qlen, -1, -1, -1)
        else:
            x_padded = torch.cat([x, zero_pad], dim=1).expand(qlen, -1, -1, -1)
        x = x_padded.masked_select(mask[:, :, None, None]).view(qlen, klen, x.size(2), x.size(3))
        return x

    def _rel_shift(self, x, zero_triu=False):
        zero_pad_shape = (x.size(0), 1) + x.size()[2:]
        zero_pad = torch.zeros(zero_pad_shape, device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=1)
        x_padded_shape = (x.size(1) + 1, x.size(0)) + x.size()[2:]
        x_padded = x_padded.view(*x_padded_shape)
        x = x_padded[1:].view_as(x)
        if zero_triu:
            ones = torch.ones((x.size(0), x.size(1)))
            x = x * torch.tril(ones, x.size(1) - x.size(0))[:, :, None, None]
        return x

    def forward(self, w, r, attn_mask=None, mems=None):
        raise NotImplementedError


class RelPartialLearnableMultiHeadAttn(RelMultiHeadAttn):

    def __init__(self, *args, **kwargs):
        super(RelPartialLearnableMultiHeadAttn, self).__init__(*args, **kwargs)
        self.r_net = nn.Linear(self.d_model, self.n_head * self.d_head, bias=False)

    def forward(self, w, r, attn_mask=None, mems=None, head_mask=None):
        qlen, rlen, bsz = w.size(0), r.size(0), w.size(1)
        if mems is not None:
            cat = torch.cat([mems, w], 0)
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(cat))
            else:
                w_heads = self.qkv_net(cat)
            r_head_k = self.r_net(r)
            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)
            w_head_q = w_head_q[-qlen:]
        else:
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(w))
            else:
                w_heads = self.qkv_net(w)
            r_head_k = self.r_net(r)
            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)
        klen = w_head_k.size(0)
        w_head_q = w_head_q.view(qlen, bsz, self.n_head, self.d_head)
        w_head_k = w_head_k.view(klen, bsz, self.n_head, self.d_head)
        w_head_v = w_head_v.view(klen, bsz, self.n_head, self.d_head)
        r_head_k = r_head_k.view(rlen, self.n_head, self.d_head)
        rw_head_q = w_head_q + self.r_w_bias
        AC = torch.einsum('ibnd,jbnd->ijbn', (rw_head_q, w_head_k))
        rr_head_q = w_head_q + self.r_r_bias
        BD = torch.einsum('ibnd,jnd->ijbn', (rr_head_q, r_head_k))
        BD = self._rel_shift(BD)
        attn_score = AC + BD
        attn_score.mul_(self.scale)
        if attn_mask is not None and attn_mask.any().item():
            if attn_mask.dim() == 2:
                attn_score = attn_score.float().masked_fill(attn_mask[None, :, :, None], -1e+30).type_as(attn_score)
            elif attn_mask.dim() == 3:
                attn_score = attn_score.float().masked_fill(attn_mask[:, :, :, None], -1e+30).type_as(attn_score)
        attn_prob = F.softmax(attn_score, dim=1)
        attn_prob = self.dropatt(attn_prob)
        if head_mask is not None:
            attn_prob = attn_prob * head_mask
        attn_vec = torch.einsum('ijbn,jbnd->ibnd', (attn_prob, w_head_v))
        attn_vec = attn_vec.contiguous().view(attn_vec.size(0), attn_vec.size(1), self.n_head * self.d_head)
        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)
        if self.pre_lnorm:
            outputs = [w + attn_out]
        else:
            outputs = [self.layer_norm(w + attn_out)]
        if self.output_attentions:
            outputs.append(attn_prob)
        return outputs


class RelLearnableMultiHeadAttn(RelMultiHeadAttn):

    def __init__(self, *args, **kwargs):
        super(RelLearnableMultiHeadAttn, self).__init__(*args, **kwargs)

    def forward(self, w, r_emb, r_w_bias, r_bias, attn_mask=None, mems=None, head_mask=None):
        qlen, bsz = w.size(0), w.size(1)
        if mems is not None:
            cat = torch.cat([mems, w], 0)
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(cat))
            else:
                w_heads = self.qkv_net(cat)
            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)
            w_head_q = w_head_q[-qlen:]
        else:
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(w))
            else:
                w_heads = self.qkv_net(w)
            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)
        klen = w_head_k.size(0)
        w_head_q = w_head_q.view(qlen, bsz, self.n_head, self.d_head)
        w_head_k = w_head_k.view(klen, bsz, self.n_head, self.d_head)
        w_head_v = w_head_v.view(klen, bsz, self.n_head, self.d_head)
        if klen > r_emb.size(0):
            r_emb_pad = r_emb[0:1].expand(klen - r_emb.size(0), -1, -1)
            r_emb = torch.cat([r_emb_pad, r_emb], 0)
            r_bias_pad = r_bias[0:1].expand(klen - r_bias.size(0), -1)
            r_bias = torch.cat([r_bias_pad, r_bias], 0)
        else:
            r_emb = r_emb[-klen:]
            r_bias = r_bias[-klen:]
        rw_head_q = w_head_q + r_w_bias[None]
        AC = torch.einsum('ibnd,jbnd->ijbn', (rw_head_q, w_head_k))
        B_ = torch.einsum('ibnd,jnd->ijbn', (w_head_q, r_emb))
        D_ = r_bias[None, :, None]
        BD = self._rel_shift(B_ + D_)
        attn_score = AC + BD
        attn_score.mul_(self.scale)
        if attn_mask is not None and attn_mask.any().item():
            if attn_mask.dim() == 2:
                attn_score.masked_fill_(attn_mask[None, :, :, None], -float('inf'))
            elif attn_mask.dim() == 3:
                attn_score.masked_fill_(attn_mask[:, :, :, None], -float('inf'))
        attn_prob = F.softmax(attn_score, dim=1)
        attn_prob = self.dropatt(attn_prob)
        if head_mask is not None:
            attn_prob = attn_prob * head_mask
        attn_vec = torch.einsum('ijbn,jbnd->ibnd', (attn_prob, w_head_v))
        attn_vec = attn_vec.contiguous().view(attn_vec.size(0), attn_vec.size(1), self.n_head * self.d_head)
        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)
        if self.pre_lnorm:
            outputs = [w + attn_out]
        else:
            outputs = [self.layer_norm(w + attn_out)]
        if self.output_attentions:
            outputs.append(attn_prob)
        return outputs


class DecoderLayer(nn.Module):

    def __init__(self, n_head, d_model, d_head, d_inner, dropout, **kwargs):
        super(DecoderLayer, self).__init__()
        self.dec_attn = MultiHeadAttn(n_head, d_model, d_head, dropout, **kwargs)
        self.pos_ff = PositionwiseFF(d_model, d_inner, dropout, pre_lnorm=kwargs.get('pre_lnorm'))

    def forward(self, dec_inp, dec_attn_mask=None, mems=None, head_mask=None):
        attn_outputs = self.dec_attn(dec_inp, attn_mask=dec_attn_mask, mems=mems, head_mask=head_mask)
        ff_output = self.pos_ff(attn_outputs[0])
        outputs = [ff_output] + attn_outputs[1:]
        return outputs


class RelLearnableDecoderLayer(nn.Module):

    def __init__(self, n_head, d_model, d_head, d_inner, dropout, **kwargs):
        super(RelLearnableDecoderLayer, self).__init__()
        self.dec_attn = RelLearnableMultiHeadAttn(n_head, d_model, d_head, dropout, **kwargs)
        self.pos_ff = PositionwiseFF(d_model, d_inner, dropout, pre_lnorm=kwargs.get('pre_lnorm'))

    def forward(self, dec_inp, r_emb, r_w_bias, r_bias, dec_attn_mask=None, mems=None, head_mask=None):
        attn_outputs = self.dec_attn(dec_inp, r_emb, r_w_bias, r_bias, attn_mask=dec_attn_mask, mems=mems, head_mask=head_mask)
        ff_output = self.pos_ff(attn_outputs[0])
        outputs = [ff_output] + attn_outputs[1:]
        return outputs


class RelPartialLearnableDecoderLayer(nn.Module):

    def __init__(self, n_head, d_model, d_head, d_inner, dropout, **kwargs):
        super(RelPartialLearnableDecoderLayer, self).__init__()
        self.dec_attn = RelPartialLearnableMultiHeadAttn(n_head, d_model, d_head, dropout, **kwargs)
        self.pos_ff = PositionwiseFF(d_model, d_inner, dropout, pre_lnorm=kwargs.get('pre_lnorm'))

    def forward(self, dec_inp, r, dec_attn_mask=None, mems=None, head_mask=None):
        attn_outputs = self.dec_attn(dec_inp, r, attn_mask=dec_attn_mask, mems=mems, head_mask=head_mask)
        ff_output = self.pos_ff(attn_outputs[0])
        outputs = [ff_output] + attn_outputs[1:]
        return outputs


class AdaptiveEmbedding(nn.Module):

    def __init__(self, n_token, d_embed, d_proj, cutoffs, div_val=1, sample_softmax=False):
        super(AdaptiveEmbedding, self).__init__()
        self.n_token = n_token
        self.d_embed = d_embed
        self.cutoffs = cutoffs + [n_token]
        self.div_val = div_val
        self.d_proj = d_proj
        self.emb_scale = d_proj ** 0.5
        self.cutoff_ends = [0] + self.cutoffs
        self.emb_layers = nn.ModuleList()
        self.emb_projs = nn.ParameterList()
        if div_val == 1:
            self.emb_layers.append(nn.Embedding(n_token, d_embed, sparse=sample_softmax > 0))
            if d_proj != d_embed:
                self.emb_projs.append(nn.Parameter(torch.FloatTensor(d_proj, d_embed)))
        else:
            for i in range(len(self.cutoffs)):
                l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i + 1]
                d_emb_i = d_embed // div_val ** i
                self.emb_layers.append(nn.Embedding(r_idx - l_idx, d_emb_i))
                self.emb_projs.append(nn.Parameter(torch.FloatTensor(d_proj, d_emb_i)))

    def forward(self, inp):
        if self.div_val == 1:
            embed = self.emb_layers[0](inp)
            if self.d_proj != self.d_embed:
                embed = F.linear(embed, self.emb_projs[0])
        else:
            param = next(self.parameters())
            inp_flat = inp.view(-1)
            emb_flat = torch.zeros([inp_flat.size(0), self.d_proj], dtype=param.dtype, device=param.device)
            for i in range(len(self.cutoffs)):
                l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i + 1]
                mask_i = (inp_flat >= l_idx) & (inp_flat < r_idx)
                indices_i = mask_i.nonzero().squeeze()
                if indices_i.numel() == 0:
                    continue
                inp_i = inp_flat.index_select(0, indices_i) - l_idx
                emb_i = self.emb_layers[i](inp_i)
                emb_i = F.linear(emb_i, self.emb_projs[i])
                emb_flat.index_copy_(0, indices_i, emb_i)
            embed_shape = inp.size() + (self.d_proj,)
            embed = emb_flat.view(embed_shape)
        embed.mul_(self.emb_scale)
        return embed


class ProjectedAdaptiveLogSoftmax(nn.Module):

    def __init__(self, n_token, d_embed, d_proj, cutoffs, div_val=1, keep_order=False):
        super(ProjectedAdaptiveLogSoftmax, self).__init__()
        self.n_token = n_token
        self.d_embed = d_embed
        self.d_proj = d_proj
        self.cutoffs = cutoffs + [n_token]
        self.cutoff_ends = [0] + self.cutoffs
        self.div_val = div_val
        self.shortlist_size = self.cutoffs[0]
        self.n_clusters = len(self.cutoffs) - 1
        self.head_size = self.shortlist_size + self.n_clusters
        if self.n_clusters > 0:
            self.cluster_weight = nn.Parameter(torch.zeros(self.n_clusters, self.d_embed))
            self.cluster_bias = nn.Parameter(torch.zeros(self.n_clusters))
        self.out_layers = nn.ModuleList()
        self.out_projs = nn.ParameterList()
        if div_val == 1:
            for i in range(len(self.cutoffs)):
                if d_proj != d_embed:
                    self.out_projs.append(nn.Parameter(torch.FloatTensor(d_proj, d_embed)))
                else:
                    self.out_projs.append(None)
            self.out_layers.append(nn.Linear(d_embed, n_token))
        else:
            for i in range(len(self.cutoffs)):
                l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i + 1]
                d_emb_i = d_embed // div_val ** i
                self.out_projs.append(nn.Parameter(torch.FloatTensor(d_proj, d_emb_i)))
                self.out_layers.append(nn.Linear(d_emb_i, r_idx - l_idx))
        self.keep_order = keep_order

    def _compute_logit(self, hidden, weight, bias, proj):
        if proj is None:
            logit = F.linear(hidden, weight, bias=bias)
        else:
            proj_hid = F.linear(hidden, proj.t().contiguous())
            logit = F.linear(proj_hid, weight, bias=bias)
        return logit

    def forward(self, hidden, labels=None, keep_order=False):
        """
            Params:
                hidden :: [len*bsz x d_proj]
                labels :: [len*bsz]
            Return:
                if labels is None:
                    out :: [len*bsz] Negative log likelihood
                else:
                    out :: [len*bsz x n_tokens] log probabilities of tokens over the vocabulary
            We could replace this implementation by the native PyTorch one
            if their's had an option to set bias on all clusters in the native one.
            here: https://github.com/pytorch/pytorch/blob/dbe6a7a9ff1a364a8706bf5df58a1ca96d2fd9da/torch/nn/modules/adaptive.py#L138
        """
        if labels is not None:
            labels = labels.view(-1)
            if hidden.size(0) != labels.size(0):
                raise RuntimeError('Input and labels should have the same size in the batch dimension.')
        if self.n_clusters == 0:
            logit = self._compute_logit(hidden, self.out_layers[0].weight, self.out_layers[0].bias, self.out_projs[0])
            if labels is not None:
                out = -F.log_softmax(logit, dim=-1).gather(1, labels.unsqueeze(1)).squeeze(1)
            else:
                out = F.log_softmax(logit, dim=-1)
        else:
            weights, biases = [], []
            for i in range(len(self.cutoffs)):
                if self.div_val == 1:
                    l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i + 1]
                    weight_i = self.out_layers[0].weight[l_idx:r_idx]
                    bias_i = self.out_layers[0].bias[l_idx:r_idx]
                else:
                    weight_i = self.out_layers[i].weight
                    bias_i = self.out_layers[i].bias
                if i == 0:
                    weight_i = torch.cat([weight_i, self.cluster_weight], dim=0)
                    bias_i = torch.cat([bias_i, self.cluster_bias], dim=0)
                weights.append(weight_i)
                biases.append(bias_i)
            head_weight, head_bias, head_proj = weights[0], biases[0], self.out_projs[0]
            head_logit = self._compute_logit(hidden, head_weight, head_bias, head_proj)
            head_logprob = F.log_softmax(head_logit, dim=1)
            if labels is None:
                out = hidden.new_empty((head_logit.size(0), self.n_token))
            else:
                out = torch.zeros_like(labels, dtype=hidden.dtype, device=hidden.device)
            offset = 0
            cutoff_values = [0] + self.cutoffs
            for i in range(len(cutoff_values) - 1):
                l_idx, r_idx = cutoff_values[i], cutoff_values[i + 1]
                if labels is not None:
                    mask_i = (labels >= l_idx) & (labels < r_idx)
                    indices_i = mask_i.nonzero().squeeze()
                    if indices_i.numel() == 0:
                        continue
                    target_i = labels.index_select(0, indices_i) - l_idx
                    head_logprob_i = head_logprob.index_select(0, indices_i)
                    hidden_i = hidden.index_select(0, indices_i)
                else:
                    hidden_i = hidden
                if i == 0:
                    if labels is not None:
                        logprob_i = head_logprob_i.gather(1, target_i[:, None]).squeeze(1)
                    else:
                        out[:, :self.cutoffs[0]] = head_logprob[:, :self.cutoffs[0]]
                else:
                    weight_i, bias_i, proj_i = weights[i], biases[i], self.out_projs[i]
                    tail_logit_i = self._compute_logit(hidden_i, weight_i, bias_i, proj_i)
                    tail_logprob_i = F.log_softmax(tail_logit_i, dim=1)
                    cluster_prob_idx = self.cutoffs[0] + i - 1
                    if labels is not None:
                        logprob_i = head_logprob_i[:, cluster_prob_idx] + tail_logprob_i.gather(1, target_i[:, None]).squeeze(1)
                    else:
                        logprob_i = head_logprob[:, cluster_prob_idx, None] + tail_logprob_i
                        out[:, l_idx:r_idx] = logprob_i
                if labels is not None:
                    if hasattr(self, 'keep_order') and self.keep_order or keep_order:
                        out.index_copy_(0, indices_i, -logprob_i)
                    else:
                        out[offset:offset + logprob_i.size(0)].copy_(-logprob_i)
                    offset += logprob_i.size(0)
        return out

    def log_prob(self, hidden):
        """ Computes log probabilities for all :math:`n\\_classes`
        From: https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/adaptive.py
        Args:
            hidden (Tensor): a minibatch of examples
        Returns:
            log-probabilities of for each class :math:`c`
            in range :math:`0 <= c <= n\\_classes`, where :math:`n\\_classes` is a
            parameter passed to ``AdaptiveLogSoftmaxWithLoss`` constructor.
        Shape:
            - Input: :math:`(N, in\\_features)`
            - Output: :math:`(N, n\\_classes)`
        """
        if self.n_clusters == 0:
            logit = self._compute_logit(hidden, self.out_layers[0].weight, self.out_layers[0].bias, self.out_projs[0])
            return F.log_softmax(logit, dim=-1)
        else:
            weights, biases = [], []
            for i in range(len(self.cutoffs)):
                if self.div_val == 1:
                    l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i + 1]
                    weight_i = self.out_layers[0].weight[l_idx:r_idx]
                    bias_i = self.out_layers[0].bias[l_idx:r_idx]
                else:
                    weight_i = self.out_layers[i].weight
                    bias_i = self.out_layers[i].bias
                if i == 0:
                    weight_i = torch.cat([weight_i, self.cluster_weight], dim=0)
                    bias_i = torch.cat([bias_i, self.cluster_bias], dim=0)
                weights.append(weight_i)
                biases.append(bias_i)
            head_weight, head_bias, head_proj = weights[0], biases[0], self.out_projs[0]
            head_logit = self._compute_logit(hidden, head_weight, head_bias, head_proj)
            out = hidden.new_empty((head_logit.size(0), self.n_token))
            head_logprob = F.log_softmax(head_logit, dim=1)
            cutoff_values = [0] + self.cutoffs
            for i in range(len(cutoff_values) - 1):
                start_idx, stop_idx = cutoff_values[i], cutoff_values[i + 1]
                if i == 0:
                    out[:, :self.cutoffs[0]] = head_logprob[:, :self.cutoffs[0]]
                else:
                    weight_i, bias_i, proj_i = weights[i], biases[i], self.out_projs[i]
                    tail_logit_i = self._compute_logit(hidden, weight_i, bias_i, proj_i)
                    tail_logprob_i = F.log_softmax(tail_logit_i, dim=1)
                    logprob_i = head_logprob[:, -i] + tail_logprob_i
                    out[:, start_idx, stop_idx] = logprob_i
            return out


CONFIG_NAME = 'config.json'


def http_get(url, temp_file, proxies=None):
    req = requests.get(url, stream=True, proxies=proxies)
    content_length = req.headers.get('Content-Length')
    total = int(content_length) if content_length is not None else None
    progress = tqdm(unit='B', total=total)
    for chunk in req.iter_content(chunk_size=1024):
        if chunk:
            progress.update(len(chunk))
            temp_file.write(chunk)
    progress.close()


def s3_request(func):
    """
    Wrapper function for s3 requests in order to create more helpful error
    messages.
    """

    @wraps(func)
    def wrapper(url, *args, **kwargs):
        try:
            return func(url, *args, **kwargs)
        except ClientError as exc:
            if int(exc.response['Error']['Code']) == 404:
                raise EnvironmentError('file {} not found'.format(url))
            else:
                raise
    return wrapper


def split_s3_path(url):
    """Split a full s3 path into the bucket name and path."""
    parsed = urlparse(url)
    if not parsed.netloc or not parsed.path:
        raise ValueError('bad s3 path {}'.format(url))
    bucket_name = parsed.netloc
    s3_path = parsed.path
    if s3_path.startswith('/'):
        s3_path = s3_path[1:]
    return bucket_name, s3_path


@s3_request
def s3_etag(url, proxies=None):
    """Check ETag on S3 object."""
    s3_resource = boto3.resource('s3', config=Config(proxies=proxies))
    bucket_name, s3_path = split_s3_path(url)
    s3_object = s3_resource.Object(bucket_name, s3_path)
    return s3_object.e_tag


@s3_request
def s3_get(url, temp_file, proxies=None):
    """Pull a file directly from S3."""
    s3_resource = boto3.resource('s3', config=Config(proxies=proxies))
    bucket_name, s3_path = split_s3_path(url)
    s3_resource.Bucket(bucket_name).download_fileobj(s3_path, temp_file)


def url_to_filename(url, etag=None):
    """
    Convert `url` into a hashed filename in a repeatable way.
    If `etag` is specified, append its hash to the url's, delimited
    by a period.
    """
    url_bytes = url.encode('utf-8')
    url_hash = sha256(url_bytes)
    filename = url_hash.hexdigest()
    if etag:
        etag_bytes = etag.encode('utf-8')
        etag_hash = sha256(etag_bytes)
        filename += '.' + etag_hash.hexdigest()
    return filename


def get_from_cache(url, cache_dir=None, force_download=False, proxies=None):
    """
    Given a URL, look for the corresponding dataset in the local cache.
    If it's not there, download it. Then return the path to the cached file.
    """
    if cache_dir is None:
        cache_dir = PYTORCH_TRANSFORMERS_CACHE
    if sys.version_info[0] == 3 and isinstance(cache_dir, Path):
        cache_dir = str(cache_dir)
    if sys.version_info[0] == 2 and not isinstance(cache_dir, str):
        cache_dir = str(cache_dir)
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    if url.startswith('s3://'):
        etag = s3_etag(url, proxies=proxies)
    else:
        try:
            response = requests.head(url, allow_redirects=True, proxies=proxies)
            if response.status_code != 200:
                etag = None
            else:
                etag = response.headers.get('ETag')
        except EnvironmentError:
            etag = None
    if sys.version_info[0] == 2 and etag is not None:
        etag = etag.decode('utf-8')
    filename = url_to_filename(url, etag)
    cache_path = os.path.join(cache_dir, filename)
    if not os.path.exists(cache_path) and etag is None:
        matching_files = fnmatch.filter(os.listdir(cache_dir), filename + '.*')
        matching_files = list(filter(lambda s: not s.endswith('.json'), matching_files))
        if matching_files:
            cache_path = os.path.join(cache_dir, matching_files[-1])
    if not os.path.exists(cache_path) or force_download:
        with tempfile.NamedTemporaryFile() as temp_file:
            logger.info('%s not found in cache or force_download set to True, downloading to %s', url, temp_file.name)
            if url.startswith('s3://'):
                s3_get(url, temp_file, proxies=proxies)
            else:
                http_get(url, temp_file, proxies=proxies)
            temp_file.flush()
            temp_file.seek(0)
            logger.info('copying %s to cache at %s', temp_file.name, cache_path)
            with open(cache_path, 'wb') as cache_file:
                shutil.copyfileobj(temp_file, cache_file)
            logger.info('creating metadata file for %s', cache_path)
            meta = {'url': url, 'etag': etag}
            meta_path = cache_path + '.json'
            with open(meta_path, 'w') as meta_file:
                output_string = json.dumps(meta)
                if sys.version_info[0] == 2 and isinstance(output_string, str):
                    output_string = unicode(output_string, 'utf-8')
                meta_file.write(output_string)
            logger.info('removing temp file %s', temp_file.name)
    return cache_path


def cached_path(url_or_filename, cache_dir=None, force_download=False, proxies=None):
    """
    Given something that might be a URL (or might be a local path),
    determine which. If it's a URL, download the file and cache it, and
    return the path to the cached file. If it's already a local path,
    make sure the file exists and then return the path.
    Args:
        cache_dir: specify a cache directory to save the file to (overwrite the default cache dir).
        force_download: if True, re-dowload the file even if it's already cached in the cache dir.
    """
    if cache_dir is None:
        cache_dir = PYTORCH_TRANSFORMERS_CACHE
    if sys.version_info[0] == 3 and isinstance(url_or_filename, Path):
        url_or_filename = str(url_or_filename)
    if sys.version_info[0] == 3 and isinstance(cache_dir, Path):
        cache_dir = str(cache_dir)
    parsed = urlparse(url_or_filename)
    if parsed.scheme in ('http', 'https', 's3'):
        return get_from_cache(url_or_filename, cache_dir=cache_dir, force_download=force_download, proxies=proxies)
    elif os.path.exists(url_or_filename):
        return url_or_filename
    elif parsed.scheme == '':
        raise EnvironmentError('file {} not found'.format(url_or_filename))
    else:
        raise ValueError('unable to parse {} as a URL or as a local path'.format(url_or_filename))


class PretrainedConfig(object):
    """ Base class for all configuration classes.
        Handles a few parameters common to all models' configurations as well as methods for loading/downloading/saving configurations.

        Note:
            A configuration file can be loaded and saved to disk. Loading the configuration file and using this file to initialize a model does **not** load the model weights.
            It only affects the model's configuration.

        Class attributes (overridden by derived classes):
            - ``pretrained_config_archive_map``: a python ``dict`` of with `short-cut-names` (string) as keys and `url` (string) of associated pretrained model configurations as values.

        Parameters:
            ``finetuning_task``: string, default `None`. Name of the task used to fine-tune the model. This can be used when converting from an original (TensorFlow or PyTorch) checkpoint.
            ``num_labels``: integer, default `2`. Number of classes to use when the model is a classification model (sequences/tokens)
            ``output_attentions``: boolean, default `False`. Should the model returns attentions weights.
            ``output_hidden_states``: string, default `False`. Should the model returns all hidden-states.
            ``torchscript``: string, default `False`. Is the model used with Torchscript.
    """
    pretrained_config_archive_map = {}

    def __init__(self, **kwargs):
        self.finetuning_task = kwargs.pop('finetuning_task', None)
        self.num_labels = kwargs.pop('num_labels', 2)
        self.output_attentions = kwargs.pop('output_attentions', False)
        self.output_hidden_states = kwargs.pop('output_hidden_states', False)
        self.torchscript = kwargs.pop('torchscript', False)
        self.pruned_heads = kwargs.pop('pruned_heads', {})

    def save_pretrained(self, save_directory):
        """ Save a configuration object to the directory `save_directory`, so that it
            can be re-loaded using the :func:`~pytorch_transformers.PretrainedConfig.from_pretrained` class method.
        """
        assert os.path.isdir(save_directory), 'Saving path should be a directory where the model and configuration can be saved'
        output_config_file = os.path.join(save_directory, CONFIG_NAME)
        self.to_json_file(output_config_file)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        """ Instantiate a :class:`~pytorch_transformers.PretrainedConfig` (or a derived class) from a pre-trained model configuration.

        Parameters:
            pretrained_model_name_or_path: either:

                - a string with the `shortcut name` of a pre-trained model configuration to load from cache or download, e.g.: ``bert-base-uncased``.
                - a path to a `directory` containing a configuration file saved using the :func:`~pytorch_transformers.PretrainedConfig.save_pretrained` method, e.g.: ``./my_model_directory/``.
                - a path or url to a saved configuration JSON `file`, e.g.: ``./my_model_directory/configuration.json``.

            cache_dir: (`optional`) string:
                Path to a directory in which a downloaded pre-trained model
                configuration should be cached if the standard cache should not be used.

            kwargs: (`optional`) dict: key/value pairs with which to update the configuration object after loading.

                - The values in kwargs of any keys which are configuration attributes will be used to override the loaded values.
                - Behavior concerning key/value pairs whose keys are *not* configuration attributes is controlled by the `return_unused_kwargs` keyword parameter.

            force_download: (`optional`) boolean, default False:
                Force to (re-)download the model weights and configuration files and override the cached versions if they exists.

            proxies: (`optional`) dict, default None:
                A dictionary of proxy servers to use by protocol or endpoint, e.g.: {'http': 'foo.bar:3128', 'http://hostname': 'foo.bar:4012'}.
                The proxies are used on each request.

            return_unused_kwargs: (`optional`) bool:

                - If False, then this function returns just the final configuration object.
                - If True, then this functions returns a tuple `(config, unused_kwargs)` where `unused_kwargs` is a dictionary consisting of the key/value pairs whose keys are not configuration attributes: ie the part of kwargs which has not been used to update `config` and is otherwise ignored.

        Examples::

            # We can't instantiate directly the base class `PretrainedConfig` so let's show the examples on a
            # derived class: BertConfig
            config = BertConfig.from_pretrained('bert-base-uncased')    # Download configuration from S3 and cache.
            config = BertConfig.from_pretrained('./test/saved_model/')  # E.g. config (or model) was saved using `save_pretrained('./test/saved_model/')`
            config = BertConfig.from_pretrained('./test/saved_model/my_configuration.json')
            config = BertConfig.from_pretrained('bert-base-uncased', output_attention=True, foo=False)
            assert config.output_attention == True
            config, unused_kwargs = BertConfig.from_pretrained('bert-base-uncased', output_attention=True,
                                                               foo=False, return_unused_kwargs=True)
            assert config.output_attention == True
            assert unused_kwargs == {'foo': False}

        """
        cache_dir = kwargs.pop('cache_dir', None)
        force_download = kwargs.pop('force_download', False)
        proxies = kwargs.pop('proxies', None)
        return_unused_kwargs = kwargs.pop('return_unused_kwargs', False)
        if pretrained_model_name_or_path in cls.pretrained_config_archive_map:
            config_file = cls.pretrained_config_archive_map[pretrained_model_name_or_path]
        elif os.path.isdir(pretrained_model_name_or_path):
            config_file = os.path.join(pretrained_model_name_or_path, CONFIG_NAME)
        else:
            config_file = pretrained_model_name_or_path
        try:
            resolved_config_file = cached_path(config_file, cache_dir=cache_dir, force_download=force_download, proxies=proxies)
        except EnvironmentError as e:
            if pretrained_model_name_or_path in cls.pretrained_config_archive_map:
                logger.error("Couldn't reach server at '{}' to download pretrained model configuration file.".format(config_file))
            else:
                logger.error("Model name '{}' was not found in model name list ({}). We assumed '{}' was a path or url but couldn't find any file associated to this path or url.".format(pretrained_model_name_or_path, ', '.join(cls.pretrained_config_archive_map.keys()), config_file))
            raise e
        if resolved_config_file == config_file:
            logger.info('loading configuration file {}'.format(config_file))
        else:
            logger.info('loading configuration file {} from cache at {}'.format(config_file, resolved_config_file))
        config = cls.from_json_file(resolved_config_file)
        if hasattr(config, 'pruned_heads'):
            config.pruned_heads = dict((int(key), set(value)) for key, value in config.pruned_heads.items())
        to_remove = []
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
                to_remove.append(key)
        for key in to_remove:
            kwargs.pop(key, None)
        logger.info('Model config %s', config)
        if return_unused_kwargs:
            return config, kwargs
        else:
            return config

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `Config` from a Python dictionary of parameters."""
        config = cls(vocab_size_or_config_json_file=-1)
        for key, value in json_object.items():
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `BertConfig` from a json file of parameters."""
        with open(json_file, 'r', encoding='utf-8') as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + '\n'

    def to_json_file(self, json_file_path):
        """ Save this instance to a json file."""
        with open(json_file_path, 'w', encoding='utf-8') as writer:
            writer.write(self.to_json_string())


TF_WEIGHTS_NAME = 'model.ckpt'


WEIGHTS_NAME = 'pytorch_model.bin'


class PreTrainedModel(nn.Module):
    """ Base class for all models.

        :class:`~pytorch_transformers.PreTrainedModel` takes care of storing the configuration of the models and handles methods for loading/downloading/saving models
        as well as a few methods commons to all models to (i) resize the input embeddings and (ii) prune heads in the self-attention heads.

        Class attributes (overridden by derived classes):
            - ``config_class``: a class derived from :class:`~pytorch_transformers.PretrainedConfig` to use as configuration class for this model architecture.
            - ``pretrained_model_archive_map``: a python ``dict`` of with `short-cut-names` (string) as keys and `url` (string) of associated pretrained weights as values.
            - ``load_tf_weights``: a python ``method`` for loading a TensorFlow checkpoint in a PyTorch model, taking as arguments:

                - ``model``: an instance of the relevant subclass of :class:`~pytorch_transformers.PreTrainedModel`,
                - ``config``: an instance of the relevant subclass of :class:`~pytorch_transformers.PretrainedConfig`,
                - ``path``: a path (string) to the TensorFlow checkpoint.

            - ``base_model_prefix``: a string indicating the attribute associated to the base model in derived classes of the same architecture adding modules on top of the base model.
    """
    config_class = None
    pretrained_model_archive_map = {}
    load_tf_weights = lambda model, config, path: None
    base_model_prefix = ''

    def __init__(self, config, *inputs, **kwargs):
        super(PreTrainedModel, self).__init__()
        if not isinstance(config, PretrainedConfig):
            raise ValueError('Parameter config in `{}(config)` should be an instance of class `PretrainedConfig`. To create a model from a pretrained model use `model = {}.from_pretrained(PRETRAINED_MODEL_NAME)`'.format(self.__class__.__name__, self.__class__.__name__))
        self.config = config

    def _get_resized_embeddings(self, old_embeddings, new_num_tokens=None):
        """ Build a resized Embedding Module from a provided token Embedding Module.
            Increasing the size will add newly initialized vectors at the end
            Reducing the size will remove vectors from the end

        Args:
            new_num_tokens: (`optional`) int
                New number of tokens in the embedding matrix.
                Increasing the size will add newly initialized vectors at the end
                Reducing the size will remove vectors from the end
                If not provided or None: return the provided token Embedding Module.
        Return: ``torch.nn.Embeddings``
            Pointer to the resized Embedding Module or the old Embedding Module if new_num_tokens is None
        """
        if new_num_tokens is None:
            return old_embeddings
        old_num_tokens, old_embedding_dim = old_embeddings.weight.size()
        if old_num_tokens == new_num_tokens:
            return old_embeddings
        new_embeddings = nn.Embedding(new_num_tokens, old_embedding_dim)
        new_embeddings
        self._init_weights(new_embeddings)
        num_tokens_to_copy = min(old_num_tokens, new_num_tokens)
        new_embeddings.weight.data[:num_tokens_to_copy, :] = old_embeddings.weight.data[:num_tokens_to_copy, :]
        return new_embeddings

    def _tie_or_clone_weights(self, first_module, second_module):
        """ Tie or clone module weights depending of weither we are using TorchScript or not
        """
        if self.config.torchscript:
            first_module.weight = nn.Parameter(second_module.weight.clone())
        else:
            first_module.weight = second_module.weight

    def resize_token_embeddings(self, new_num_tokens=None):
        """ Resize input token embeddings matrix of the model if new_num_tokens != config.vocab_size.
        Take care of tying weights embeddings afterwards if the model class has a `tie_weights()` method.

        Arguments:

            new_num_tokens: (`optional`) int:
                New number of tokens in the embedding matrix. Increasing the size will add newly initialized vectors at the end. Reducing the size will remove vectors from the end. 
                If not provided or None: does nothing and just returns a pointer to the input tokens ``torch.nn.Embeddings`` Module of the model.

        Return: ``torch.nn.Embeddings``
            Pointer to the input tokens Embeddings Module of the model
        """
        base_model = getattr(self, self.base_model_prefix, self)
        model_embeds = base_model._resize_token_embeddings(new_num_tokens)
        if new_num_tokens is None:
            return model_embeds
        self.config.vocab_size = new_num_tokens
        base_model.vocab_size = new_num_tokens
        if hasattr(self, 'tie_weights'):
            self.tie_weights()
        return model_embeds

    def init_weights(self):
        """ Initialize and prunes weights if needed. """
        self.apply(self._init_weights)
        if self.config.pruned_heads:
            self.prune_heads(self.config.pruned_heads)

    def prune_heads(self, heads_to_prune):
        """ Prunes heads of the base model.

            Arguments:

                heads_to_prune: dict with keys being selected layer indices (`int`) and associated values being the list of heads to prune in said layer (list of `int`).
                E.g. {1: [0, 2], 2: [2, 3]} will prune heads 0 and 2 on layer 1 and heads 2 and 3 on layer 2.
        """
        base_model = getattr(self, self.base_model_prefix, self)
        for layer, heads in heads_to_prune.items():
            union_heads = set(self.config.pruned_heads.get(layer, [])) | set(heads)
            self.config.pruned_heads[layer] = list(union_heads)
        base_model._prune_heads(heads_to_prune)

    def save_pretrained(self, save_directory):
        """ Save a model and its configuration file to a directory, so that it
            can be re-loaded using the `:func:`~pytorch_transformers.PreTrainedModel.from_pretrained`` class method.
        """
        assert os.path.isdir(save_directory), 'Saving path should be a directory where the model and configuration can be saved'
        model_to_save = self.module if hasattr(self, 'module') else self
        model_to_save.config.save_pretrained(save_directory)
        output_model_file = os.path.join(save_directory, WEIGHTS_NAME)
        torch.save(model_to_save.state_dict(), output_model_file)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """Instantiate a pretrained pytorch model from a pre-trained model configuration.

        The model is set in evaluation mode by default using ``model.eval()`` (Dropout modules are deactivated)
        To train the model, you should first set it back in training mode with ``model.train()``

        The warning ``Weights from XXX not initialized from pretrained model`` means that the weights of XXX do not come pre-trained with the rest of the model.
        It is up to you to train those weights with a downstream fine-tuning task.

        The warning ``Weights from XXX not used in YYY`` means that the layer XXX is not used by YYY, therefore those weights are discarded.

        Parameters:
            pretrained_model_name_or_path: either:

                - a string with the `shortcut name` of a pre-trained model to load from cache or download, e.g.: ``bert-base-uncased``.
                - a path to a `directory` containing model weights saved using :func:`~pytorch_transformers.PreTrainedModel.save_pretrained`, e.g.: ``./my_model_directory/``.
                - a path or url to a `tensorflow index checkpoint file` (e.g. `./tf_model/model.ckpt.index`). In this case, ``from_tf`` should be set to True and a configuration object should be provided as ``config`` argument. This loading path is slower than converting the TensorFlow checkpoint in a PyTorch model using the provided conversion scripts and loading the PyTorch model afterwards.

            model_args: (`optional`) Sequence of positional arguments:
                All remaning positional arguments will be passed to the underlying model's ``__init__`` method

            config: (`optional`) instance of a class derived from :class:`~pytorch_transformers.PretrainedConfig`:
                Configuration for the model to use instead of an automatically loaded configuation. Configuration can be automatically loaded when:

                - the model is a model provided by the library (loaded with the ``shortcut-name`` string of a pretrained model), or
                - the model was saved using :func:`~pytorch_transformers.PreTrainedModel.save_pretrained` and is reloaded by suppling the save directory.
                - the model is loaded by suppling a local directory as ``pretrained_model_name_or_path`` and a configuration JSON file named `config.json` is found in the directory.

            state_dict: (`optional`) dict:
                an optional state dictionnary for the model to use instead of a state dictionary loaded from saved weights file.
                This option can be used if you want to create a model from a pretrained configuration but load your own weights.
                In this case though, you should check if using :func:`~pytorch_transformers.PreTrainedModel.save_pretrained` and :func:`~pytorch_transformers.PreTrainedModel.from_pretrained` is not a simpler option.

            cache_dir: (`optional`) string:
                Path to a directory in which a downloaded pre-trained model
                configuration should be cached if the standard cache should not be used.

            force_download: (`optional`) boolean, default False:
                Force to (re-)download the model weights and configuration files and override the cached versions if they exists.

            proxies: (`optional`) dict, default None:
                A dictionary of proxy servers to use by protocol or endpoint, e.g.: {'http': 'foo.bar:3128', 'http://hostname': 'foo.bar:4012'}.
                The proxies are used on each request.

            output_loading_info: (`optional`) boolean:
                Set to ``True`` to also return a dictionnary containing missing keys, unexpected keys and error messages.

            kwargs: (`optional`) Remaining dictionary of keyword arguments:
                Can be used to update the configuration object (after it being loaded) and initiate the model. (e.g. ``output_attention=True``). Behave differently depending on whether a `config` is provided or automatically loaded:

                - If a configuration is provided with ``config``, ``**kwargs`` will be directly passed to the underlying model's ``__init__`` method (we assume all relevant updates to the configuration have already been done)
                - If a configuration is not provided, ``kwargs`` will be first passed to the configuration class initialization function (:func:`~pytorch_transformers.PretrainedConfig.from_pretrained`). Each key of ``kwargs`` that corresponds to a configuration attribute will be used to override said attribute with the supplied ``kwargs`` value. Remaining keys that do not correspond to any configuration attribute will be passed to the underlying model's ``__init__`` function.

        Examples::

            model = BertModel.from_pretrained('bert-base-uncased')    # Download model and configuration from S3 and cache.
            model = BertModel.from_pretrained('./test/saved_model/')  # E.g. model was saved using `save_pretrained('./test/saved_model/')`
            model = BertModel.from_pretrained('bert-base-uncased', output_attention=True)  # Update configuration during loading
            assert model.config.output_attention == True
            # Loading from a TF checkpoint file instead of a PyTorch model (slower)
            config = BertConfig.from_json_file('./tf_model/my_tf_model_config.json')
            model = BertModel.from_pretrained('./tf_model/my_tf_checkpoint.ckpt.index', from_tf=True, config=config)

        """
        config = kwargs.pop('config', None)
        state_dict = kwargs.pop('state_dict', None)
        cache_dir = kwargs.pop('cache_dir', None)
        from_tf = kwargs.pop('from_tf', False)
        force_download = kwargs.pop('force_download', False)
        proxies = kwargs.pop('proxies', None)
        output_loading_info = kwargs.pop('output_loading_info', False)
        if config is None:
            config, model_kwargs = cls.config_class.from_pretrained(pretrained_model_name_or_path, *model_args, cache_dir=cache_dir, return_unused_kwargs=True, force_download=force_download, **kwargs)
        else:
            model_kwargs = kwargs
        if pretrained_model_name_or_path in cls.pretrained_model_archive_map:
            archive_file = cls.pretrained_model_archive_map[pretrained_model_name_or_path]
        elif os.path.isdir(pretrained_model_name_or_path):
            if from_tf:
                archive_file = os.path.join(pretrained_model_name_or_path, TF_WEIGHTS_NAME + '.index')
            else:
                archive_file = os.path.join(pretrained_model_name_or_path, WEIGHTS_NAME)
        elif from_tf:
            archive_file = pretrained_model_name_or_path + '.index'
        else:
            archive_file = pretrained_model_name_or_path
        try:
            resolved_archive_file = cached_path(archive_file, cache_dir=cache_dir, force_download=force_download, proxies=proxies)
        except EnvironmentError as e:
            if pretrained_model_name_or_path in cls.pretrained_model_archive_map:
                logger.error("Couldn't reach server at '{}' to download pretrained weights.".format(archive_file))
            else:
                logger.error("Model name '{}' was not found in model name list ({}). We assumed '{}' was a path or url but couldn't find any file associated to this path or url.".format(pretrained_model_name_or_path, ', '.join(cls.pretrained_model_archive_map.keys()), archive_file))
            raise e
        if resolved_archive_file == archive_file:
            logger.info('loading weights file {}'.format(archive_file))
        else:
            logger.info('loading weights file {} from cache at {}'.format(archive_file, resolved_archive_file))
        model = cls(config, *model_args, **model_kwargs)
        if state_dict is None and not from_tf:
            state_dict = torch.load(resolved_archive_file, map_location='cpu')
        if from_tf:
            return cls.load_tf_weights(model, config, resolved_archive_file[:-6])
        old_keys = []
        new_keys = []
        for key in state_dict.keys():
            new_key = None
            if 'gamma' in key:
                new_key = key.replace('gamma', 'weight')
            if 'beta' in key:
                new_key = key.replace('beta', 'bias')
            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)
        for old_key, new_key in zip(old_keys, new_keys):
            state_dict[new_key] = state_dict.pop(old_key)
        missing_keys = []
        unexpected_keys = []
        error_msgs = []
        metadata = getattr(state_dict, '_metadata', None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        def load(module, prefix=''):
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            module._load_from_state_dict(state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + '.')
        start_prefix = ''
        model_to_load = model
        if not hasattr(model, cls.base_model_prefix) and any(s.startswith(cls.base_model_prefix) for s in state_dict.keys()):
            start_prefix = cls.base_model_prefix + '.'
        if hasattr(model, cls.base_model_prefix) and not any(s.startswith(cls.base_model_prefix) for s in state_dict.keys()):
            model_to_load = getattr(model, cls.base_model_prefix)
        load(model_to_load, prefix=start_prefix)
        if len(missing_keys) > 0:
            logger.info('Weights of {} not initialized from pretrained model: {}'.format(model.__class__.__name__, missing_keys))
        if len(unexpected_keys) > 0:
            logger.info('Weights from pretrained model not used in {}: {}'.format(model.__class__.__name__, unexpected_keys))
        if len(error_msgs) > 0:
            raise RuntimeError('Error(s) in loading state_dict for {}:\n\t{}'.format(model.__class__.__name__, '\n\t'.join(error_msgs)))
        if hasattr(model, 'tie_weights'):
            model.tie_weights()
        model.eval()
        if output_loading_info:
            loading_info = {'missing_keys': missing_keys, 'unexpected_keys': unexpected_keys, 'error_msgs': error_msgs}
            return model, loading_info
        return model


class PoolerStartLogits(nn.Module):
    """ Compute SQuAD start_logits from sequence hidden states. """

    def __init__(self, config):
        super(PoolerStartLogits, self).__init__()
        self.dense = nn.Linear(config.hidden_size, 1)

    def forward(self, hidden_states, p_mask=None):
        """ Args:
            **p_mask**: (`optional`) ``torch.FloatTensor`` of shape `(batch_size, seq_len)`
                invalid position mask such as query and special symbols (PAD, SEP, CLS)
                1.0 means token should be masked.
        """
        x = self.dense(hidden_states).squeeze(-1)
        if p_mask is not None:
            x = x * (1 - p_mask) - 1e+30 * p_mask
        return x


class PoolerEndLogits(nn.Module):
    """ Compute SQuAD end_logits from sequence hidden states and start token hidden state.
    """

    def __init__(self, config):
        super(PoolerEndLogits, self).__init__()
        self.dense_0 = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.activation = nn.Tanh()
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dense_1 = nn.Linear(config.hidden_size, 1)

    def forward(self, hidden_states, start_states=None, start_positions=None, p_mask=None):
        """ Args:
            One of ``start_states``, ``start_positions`` should be not None.
            If both are set, ``start_positions`` overrides ``start_states``.

            **start_states**: ``torch.LongTensor`` of shape identical to hidden_states
                hidden states of the first tokens for the labeled span.
            **start_positions**: ``torch.LongTensor`` of shape ``(batch_size,)``
                position of the first token for the labeled span:
            **p_mask**: (`optional`) ``torch.FloatTensor`` of shape ``(batch_size, seq_len)``
                Mask of invalid position such as query and special symbols (PAD, SEP, CLS)
                1.0 means token should be masked.
        """
        assert start_states is not None or start_positions is not None, 'One of start_states, start_positions should be not None'
        if start_positions is not None:
            slen, hsz = hidden_states.shape[-2:]
            start_positions = start_positions[:, None, None].expand(-1, -1, hsz)
            start_states = hidden_states.gather(-2, start_positions)
            start_states = start_states.expand(-1, slen, -1)
        x = self.dense_0(torch.cat([hidden_states, start_states], dim=-1))
        x = self.activation(x)
        x = self.LayerNorm(x)
        x = self.dense_1(x).squeeze(-1)
        if p_mask is not None:
            x = x * (1 - p_mask) - 1e+30 * p_mask
        return x


class PoolerAnswerClass(nn.Module):
    """ Compute SQuAD 2.0 answer class from classification and start tokens hidden states. """

    def __init__(self, config):
        super(PoolerAnswerClass, self).__init__()
        self.dense_0 = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.activation = nn.Tanh()
        self.dense_1 = nn.Linear(config.hidden_size, 1, bias=False)

    def forward(self, hidden_states, start_states=None, start_positions=None, cls_index=None):
        """
        Args:
            One of ``start_states``, ``start_positions`` should be not None.
            If both are set, ``start_positions`` overrides ``start_states``.

            **start_states**: ``torch.LongTensor`` of shape identical to ``hidden_states``.
                hidden states of the first tokens for the labeled span.
            **start_positions**: ``torch.LongTensor`` of shape ``(batch_size,)``
                position of the first token for the labeled span.
            **cls_index**: torch.LongTensor of shape ``(batch_size,)``
                position of the CLS token. If None, take the last token.

            note(Original repo):
                no dependency on end_feature so that we can obtain one single `cls_logits`
                for each sample
        """
        hsz = hidden_states.shape[-1]
        assert start_states is not None or start_positions is not None, 'One of start_states, start_positions should be not None'
        if start_positions is not None:
            start_positions = start_positions[:, None, None].expand(-1, -1, hsz)
            start_states = hidden_states.gather(-2, start_positions).squeeze(-2)
        if cls_index is not None:
            cls_index = cls_index[:, None, None].expand(-1, -1, hsz)
            cls_token_state = hidden_states.gather(-2, cls_index).squeeze(-2)
        else:
            cls_token_state = hidden_states[:, -1, :]
        x = self.dense_0(torch.cat([start_states, cls_token_state], dim=-1))
        x = self.activation(x)
        x = self.dense_1(x).squeeze(-1)
        return x


class SQuADHead(nn.Module):
    """ A SQuAD head inspired by XLNet.

    Parameters:
        config (:class:`~pytorch_transformers.XLNetConfig`): Model configuration class with all the parameters of the model.

    Inputs:
        **hidden_states**: ``torch.FloatTensor`` of shape ``(batch_size, seq_len, hidden_size)``
            hidden states of sequence tokens
        **start_positions**: ``torch.LongTensor`` of shape ``(batch_size,)``
            position of the first token for the labeled span.
        **end_positions**: ``torch.LongTensor`` of shape ``(batch_size,)``
            position of the last token for the labeled span.
        **cls_index**: torch.LongTensor of shape ``(batch_size,)``
            position of the CLS token. If None, take the last token.
        **is_impossible**: ``torch.LongTensor`` of shape ``(batch_size,)``
            Whether the question has a possible answer in the paragraph or not.
        **p_mask**: (`optional`) ``torch.FloatTensor`` of shape ``(batch_size, seq_len)``
            Mask of invalid position such as query and special symbols (PAD, SEP, CLS)
            1.0 means token should be masked.

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned if both ``start_positions`` and ``end_positions`` are provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Classification loss as the sum of start token, end token (and is_impossible if provided) classification losses.
        **start_top_log_probs**: (`optional`, returned if ``start_positions`` or ``end_positions`` is not provided)
            ``torch.FloatTensor`` of shape ``(batch_size, config.start_n_top)``
            Log probabilities for the top config.start_n_top start token possibilities (beam-search).
        **start_top_index**: (`optional`, returned if ``start_positions`` or ``end_positions`` is not provided)
            ``torch.LongTensor`` of shape ``(batch_size, config.start_n_top)``
            Indices for the top config.start_n_top start token possibilities (beam-search).
        **end_top_log_probs**: (`optional`, returned if ``start_positions`` or ``end_positions`` is not provided)
            ``torch.FloatTensor`` of shape ``(batch_size, config.start_n_top * config.end_n_top)``
            Log probabilities for the top ``config.start_n_top * config.end_n_top`` end token possibilities (beam-search).
        **end_top_index**: (`optional`, returned if ``start_positions`` or ``end_positions`` is not provided)
            ``torch.LongTensor`` of shape ``(batch_size, config.start_n_top * config.end_n_top)``
            Indices for the top ``config.start_n_top * config.end_n_top`` end token possibilities (beam-search).
        **cls_logits**: (`optional`, returned if ``start_positions`` or ``end_positions`` is not provided)
            ``torch.FloatTensor`` of shape ``(batch_size,)``
            Log probabilities for the ``is_impossible`` label of the answers.
    """

    def __init__(self, config):
        super(SQuADHead, self).__init__()
        self.start_n_top = config.start_n_top
        self.end_n_top = config.end_n_top
        self.start_logits = PoolerStartLogits(config)
        self.end_logits = PoolerEndLogits(config)
        self.answer_class = PoolerAnswerClass(config)

    def forward(self, hidden_states, start_positions=None, end_positions=None, cls_index=None, is_impossible=None, p_mask=None):
        outputs = ()
        start_logits = self.start_logits(hidden_states, p_mask=p_mask)
        if start_positions is not None and end_positions is not None:
            for x in (start_positions, end_positions, cls_index, is_impossible):
                if x is not None and x.dim() > 1:
                    x.squeeze_(-1)
            end_logits = self.end_logits(hidden_states, start_positions=start_positions, p_mask=p_mask)
            loss_fct = CrossEntropyLoss()
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            if cls_index is not None and is_impossible is not None:
                cls_logits = self.answer_class(hidden_states, start_positions=start_positions, cls_index=cls_index)
                loss_fct_cls = nn.BCEWithLogitsLoss()
                cls_loss = loss_fct_cls(cls_logits, is_impossible)
                total_loss += cls_loss * 0.5
            outputs = (total_loss,) + outputs
        else:
            bsz, slen, hsz = hidden_states.size()
            start_log_probs = F.softmax(start_logits, dim=-1)
            start_top_log_probs, start_top_index = torch.topk(start_log_probs, self.start_n_top, dim=-1)
            start_top_index_exp = start_top_index.unsqueeze(-1).expand(-1, -1, hsz)
            start_states = torch.gather(hidden_states, -2, start_top_index_exp)
            start_states = start_states.unsqueeze(1).expand(-1, slen, -1, -1)
            hidden_states_expanded = hidden_states.unsqueeze(2).expand_as(start_states)
            p_mask = p_mask.unsqueeze(-1) if p_mask is not None else None
            end_logits = self.end_logits(hidden_states_expanded, start_states=start_states, p_mask=p_mask)
            end_log_probs = F.softmax(end_logits, dim=1)
            end_top_log_probs, end_top_index = torch.topk(end_log_probs, self.end_n_top, dim=1)
            end_top_log_probs = end_top_log_probs.view(-1, self.start_n_top * self.end_n_top)
            end_top_index = end_top_index.view(-1, self.start_n_top * self.end_n_top)
            start_states = torch.einsum('blh,bl->bh', hidden_states, start_log_probs)
            cls_logits = self.answer_class(hidden_states, start_states=start_states, cls_index=cls_index)
            outputs = (start_top_log_probs, start_top_index, end_top_log_probs, end_top_index, cls_logits) + outputs
        return outputs


class SequenceSummary(nn.Module):
    """ Compute a single vector summary of a sequence hidden states according to various possibilities:
        Args of the config class:
            summary_type:
                - 'last' => [default] take the last token hidden state (like XLNet)
                - 'first' => take the first token hidden state (like Bert)
                - 'mean' => take the mean of all tokens hidden states
                - 'cls_index' => supply a Tensor of classification token position (GPT/GPT-2)
                - 'attn' => Not implemented now, use multi-head attention
            summary_use_proj: Add a projection after the vector extraction
            summary_proj_to_labels: If True, the projection outputs to config.num_labels classes (otherwise to hidden_size). Default: False.
            summary_activation: 'tanh' => add a tanh activation to the output, Other => no activation. Default
            summary_first_dropout: Add a dropout before the projection and activation
            summary_last_dropout: Add a dropout after the projection and activation
    """

    def __init__(self, config):
        super(SequenceSummary, self).__init__()
        self.summary_type = config.summary_type if hasattr(config, 'summary_use_proj') else 'last'
        if self.summary_type == 'attn':
            raise NotImplementedError
        self.summary = Identity()
        if hasattr(config, 'summary_use_proj') and config.summary_use_proj:
            if hasattr(config, 'summary_proj_to_labels') and config.summary_proj_to_labels and config.num_labels > 0:
                num_classes = config.num_labels
            else:
                num_classes = config.hidden_size
            self.summary = nn.Linear(config.hidden_size, num_classes)
        self.activation = Identity()
        if hasattr(config, 'summary_activation') and config.summary_activation == 'tanh':
            self.activation = nn.Tanh()
        self.first_dropout = Identity()
        if hasattr(config, 'summary_first_dropout') and config.summary_first_dropout > 0:
            self.first_dropout = nn.Dropout(config.summary_first_dropout)
        self.last_dropout = Identity()
        if hasattr(config, 'summary_last_dropout') and config.summary_last_dropout > 0:
            self.last_dropout = nn.Dropout(config.summary_last_dropout)

    def forward(self, hidden_states, cls_index=None):
        """ hidden_states: float Tensor in shape [bsz, seq_len, hidden_size], the hidden-states of the last layer.
            cls_index: [optional] position of the classification token if summary_type == 'cls_index',
                shape (bsz,) or more generally (bsz, ...) where ... are optional leading dimensions of hidden_states.
                if summary_type == 'cls_index' and cls_index is None:
                    we take the last token of the sequence as classification token
        """
        if self.summary_type == 'last':
            output = hidden_states[:, -1]
        elif self.summary_type == 'first':
            output = hidden_states[:, 0]
        elif self.summary_type == 'mean':
            output = hidden_states.mean(dim=1)
        elif self.summary_type == 'cls_index':
            if cls_index is None:
                cls_index = torch.full_like(hidden_states[..., :1, :], hidden_states.shape[-2] - 1, dtype=torch.long)
            else:
                cls_index = cls_index.unsqueeze(-1).unsqueeze(-1)
                cls_index = cls_index.expand((-1,) * (cls_index.dim() - 1) + (hidden_states.size(-1),))
            output = hidden_states.gather(-2, cls_index).squeeze(-2)
        elif self.summary_type == 'attn':
            raise NotImplementedError
        output = self.first_dropout(output)
        output = self.summary(output)
        output = self.activation(output)
        output = self.last_dropout(output)
        return output


XLM_PRETRAINED_CONFIG_ARCHIVE_MAP = {'xlm-mlm-en-2048': 'https://s3.amazonaws.com/models.huggingface.co/bert/xlm-mlm-en-2048-config.json', 'xlm-mlm-ende-1024': 'https://s3.amazonaws.com/models.huggingface.co/bert/xlm-mlm-ende-1024-config.json', 'xlm-mlm-enfr-1024': 'https://s3.amazonaws.com/models.huggingface.co/bert/xlm-mlm-enfr-1024-config.json', 'xlm-mlm-enro-1024': 'https://s3.amazonaws.com/models.huggingface.co/bert/xlm-mlm-enro-1024-config.json', 'xlm-mlm-tlm-xnli15-1024': 'https://s3.amazonaws.com/models.huggingface.co/bert/xlm-mlm-tlm-xnli15-1024-config.json', 'xlm-mlm-xnli15-1024': 'https://s3.amazonaws.com/models.huggingface.co/bert/xlm-mlm-xnli15-1024-config.json', 'xlm-clm-enfr-1024': 'https://s3.amazonaws.com/models.huggingface.co/bert/xlm-clm-enfr-1024-config.json', 'xlm-clm-ende-1024': 'https://s3.amazonaws.com/models.huggingface.co/bert/xlm-clm-ende-1024-config.json', 'xlm-mlm-17-1280': 'https://s3.amazonaws.com/models.huggingface.co/bert/xlm-mlm-17-1280-config.json', 'xlm-mlm-100-1280': 'https://s3.amazonaws.com/models.huggingface.co/bert/xlm-mlm-100-1280-config.json'}


class XLMConfig(PretrainedConfig):
    """Configuration class to store the configuration of a `XLMModel`.

    Args:
        vocab_size_or_config_json_file: Vocabulary size of `inputs_ids` in `XLMModel`.
        d_model: Size of the encoder layers and the pooler layer.
        n_layer: Number of hidden layers in the Transformer encoder.
        n_head: Number of attention heads for each attention layer in
            the Transformer encoder.
        d_inner: The size of the "intermediate" (i.e., feed-forward)
            layer in the Transformer encoder.
        ff_activation: The non-linear activation function (function or string) in the
            encoder and pooler. If string, "gelu", "relu" and "swish" are supported.
        untie_r: untie relative position biases
        attn_type: 'bi' for XLM, 'uni' for Transformer-XL

        dropout: The dropout probabilitiy for all fully connected
            layers in the embeddings, encoder, and pooler.
        dropatt: The dropout ratio for the attention
            probabilities.
        max_position_embeddings: The maximum sequence length that this model might
            ever be used with. Typically set this to something large just in case
            (e.g., 512 or 1024 or 2048).
        initializer_range: The sttdev of the truncated_normal_initializer for
            initializing all weight matrices.
        layer_norm_eps: The epsilon used by LayerNorm.

        dropout: float, dropout rate.
        dropatt: float, dropout rate on attention probabilities.
        init: str, the initialization scheme, either "normal" or "uniform".
        init_range: float, initialize the parameters with a uniform distribution
            in [-init_range, init_range]. Only effective when init="uniform".
        init_std: float, initialize the parameters with a normal distribution
            with mean 0 and stddev init_std. Only effective when init="normal".
        mem_len: int, the number of tokens to cache.
        reuse_len: int, the number of tokens in the currect batch to be cached
            and reused in the future.
        bi_data: bool, whether to use bidirectional input pipeline.
            Usually set to True during pretraining and False during finetuning.
        clamp_len: int, clamp all relative distances larger than clamp_len.
            -1 means no clamping.
        same_length: bool, whether to use the same attention length for each token.
    """
    pretrained_config_archive_map = XLM_PRETRAINED_CONFIG_ARCHIVE_MAP

    def __init__(self, vocab_size_or_config_json_file=30145, emb_dim=2048, n_layers=12, n_heads=16, dropout=0.1, attention_dropout=0.1, gelu_activation=True, sinusoidal_embeddings=False, causal=False, asm=False, n_langs=1, use_lang_emb=True, max_position_embeddings=512, embed_init_std=2048 ** -0.5, layer_norm_eps=1e-12, init_std=0.02, bos_index=0, eos_index=1, pad_index=2, unk_index=3, mask_index=5, is_encoder=True, finetuning_task=None, num_labels=2, summary_type='first', summary_use_proj=True, summary_activation=None, summary_proj_to_labels=True, summary_first_dropout=0.1, start_n_top=5, end_n_top=5, **kwargs):
        """Constructs XLMConfig.
        """
        super(XLMConfig, self).__init__(**kwargs)
        if isinstance(vocab_size_or_config_json_file, str) or sys.version_info[0] == 2 and isinstance(vocab_size_or_config_json_file, unicode):
            with open(vocab_size_or_config_json_file, 'r', encoding='utf-8') as reader:
                json_config = json.loads(reader.read())
            for key, value in json_config.items():
                self.__dict__[key] = value
        elif isinstance(vocab_size_or_config_json_file, int):
            self.n_words = vocab_size_or_config_json_file
            self.emb_dim = emb_dim
            self.n_layers = n_layers
            self.n_heads = n_heads
            self.dropout = dropout
            self.attention_dropout = attention_dropout
            self.gelu_activation = gelu_activation
            self.sinusoidal_embeddings = sinusoidal_embeddings
            self.causal = causal
            self.asm = asm
            self.n_langs = n_langs
            self.use_lang_emb = use_lang_emb
            self.layer_norm_eps = layer_norm_eps
            self.bos_index = bos_index
            self.eos_index = eos_index
            self.pad_index = pad_index
            self.unk_index = unk_index
            self.mask_index = mask_index
            self.is_encoder = is_encoder
            self.max_position_embeddings = max_position_embeddings
            self.embed_init_std = embed_init_std
            self.init_std = init_std
            self.finetuning_task = finetuning_task
            self.num_labels = num_labels
            self.summary_type = summary_type
            self.summary_use_proj = summary_use_proj
            self.summary_activation = summary_activation
            self.summary_proj_to_labels = summary_proj_to_labels
            self.summary_first_dropout = summary_first_dropout
            self.start_n_top = start_n_top
            self.end_n_top = end_n_top
        else:
            raise ValueError('First argument must be either a vocabulary size (int) or the path to a pretrained model config file (str)')

    @property
    def vocab_size(self):
        return self.n_words

    @vocab_size.setter
    def vocab_size(self, value):
        self.n_words = value

    @property
    def hidden_size(self):
        return self.emb_dim

    @property
    def num_attention_heads(self):
        return self.n_heads

    @property
    def num_hidden_layers(self):
        return self.n_layers


XLM_PRETRAINED_MODEL_ARCHIVE_MAP = {'xlm-mlm-en-2048': 'https://s3.amazonaws.com/models.huggingface.co/bert/xlm-mlm-en-2048-pytorch_model.bin', 'xlm-mlm-ende-1024': 'https://s3.amazonaws.com/models.huggingface.co/bert/xlm-mlm-ende-1024-pytorch_model.bin', 'xlm-mlm-enfr-1024': 'https://s3.amazonaws.com/models.huggingface.co/bert/xlm-mlm-enfr-1024-pytorch_model.bin', 'xlm-mlm-enro-1024': 'https://s3.amazonaws.com/models.huggingface.co/bert/xlm-mlm-enro-1024-pytorch_model.bin', 'xlm-mlm-tlm-xnli15-1024': 'https://s3.amazonaws.com/models.huggingface.co/bert/xlm-mlm-tlm-xnli15-1024-pytorch_model.bin', 'xlm-mlm-xnli15-1024': 'https://s3.amazonaws.com/models.huggingface.co/bert/xlm-mlm-xnli15-1024-pytorch_model.bin', 'xlm-clm-enfr-1024': 'https://s3.amazonaws.com/models.huggingface.co/bert/xlm-clm-enfr-1024-pytorch_model.bin', 'xlm-clm-ende-1024': 'https://s3.amazonaws.com/models.huggingface.co/bert/xlm-clm-ende-1024-pytorch_model.bin', 'xlm-mlm-17-1280': 'https://s3.amazonaws.com/models.huggingface.co/bert/xlm-mlm-17-1280-pytorch_model.bin', 'xlm-mlm-100-1280': 'https://s3.amazonaws.com/models.huggingface.co/bert/xlm-mlm-100-1280-pytorch_model.bin'}


class XLMPreTrainedModel(PreTrainedModel):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """
    config_class = XLMConfig
    pretrained_model_archive_map = XLM_PRETRAINED_MODEL_ARCHIVE_MAP
    load_tf_weights = None
    base_model_prefix = 'transformer'

    def __init__(self, *inputs, **kwargs):
        super(XLMPreTrainedModel, self).__init__(*inputs, **kwargs)

    def _init_weights(self, module):
        """ Initialize the weights. """
        if isinstance(module, nn.Embedding):
            if self.config is not None and self.config.embed_init_std is not None:
                nn.init.normal_(module.weight, mean=0, std=self.config.embed_init_std)
        if isinstance(module, nn.Linear):
            if self.config is not None and self.config.init_std is not None:
                nn.init.normal_(module.weight, mean=0, std=self.config.init_std)
                if hasattr(module, 'bias') and module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
        if isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


XLM_INPUTS_DOCSTRING = """
    Inputs:
        **input_ids**: ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            Indices of input sequence tokens in the vocabulary.

            XLM is a model with absolute position embeddings so it's usually advised to pad the inputs on
            the right rather than the left.

            Indices can be obtained using :class:`pytorch_transformers.XLMTokenizer`.
            See :func:`pytorch_transformers.PreTrainedTokenizer.encode` and
            :func:`pytorch_transformers.PreTrainedTokenizer.convert_tokens_to_ids` for details.
        **position_ids**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            Indices of positions of each input sequence tokens in the position embeddings.
            Selected in the range ``[0, config.max_position_embeddings - 1]``.
        **token_type_ids**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            A parallel sequence of tokens (can be used to indicate various portions of the inputs).
            The embeddings from these tokens will be summed with the respective token embeddings.
            Indices are selected in the vocabulary (unlike BERT which has a specific vocabulary for segment indices).
        **langs**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            A parallel sequence of tokens to be used to indicate the language of each token in the input.
            Indices are languages ids which can be obtained from the language names by using two conversion mappings
            provided in the configuration of the model (only provided for multilingual models).
            More precisely, the `language name -> language id` mapping is in `model.config.lang2id` (dict str -> int) and
            the `language id -> language name` mapping is `model.config.id2lang` (dict int -> str).
        **attention_mask**: (`optional`) ``torch.FloatTensor`` of shape ``(batch_size, sequence_length)``:
            Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            ``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens.
        **lengths**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Length of each sentence that can be used to avoid performing attention on padding token indices.
            You can also use `attention_mask` for the same result (see above), kept here for compatbility.
            Indices selected in ``[0, ..., input_ids.size(-1)]``:
        **cache**:
            dictionary with ``torch.FloatTensor`` that contains pre-computed
            hidden-states (key and values in the attention blocks) as computed by the model
            (see `cache` output below). Can be used to speed up sequential decoding.
            The dictionary object will be modified in-place during the forward pass to add newly computed hidden-states.
        **head_mask**: (`optional`) ``torch.FloatTensor`` of shape ``(num_heads,)`` or ``(num_layers, num_heads)``:
            Mask to nullify selected heads of the self-attention modules.
            Mask values selected in ``[0, 1]``:
            ``1`` indicates the head is **not masked**, ``0`` indicates the head is **masked**.
"""


XLM_START_DOCSTRING = """    The XLM model was proposed in
    `Cross-lingual Language Model Pretraining`_
    by Guillaume Lample*, Alexis Conneau*. It's a transformer pre-trained using one of the following objectives:

        - a causal language modeling (CLM) objective (next token prediction),
        - a masked language modeling (MLM) objective (Bert-like), or
        - a Translation Language Modeling (TLM) object (extension of Bert's MLM to multiple language inputs)

    Original code can be found `here`_.

    This model is a PyTorch `torch.nn.Module`_ sub-class. Use it as a regular PyTorch Module and
    refer to the PyTorch documentation for all matter related to general usage and behavior.

    .. _`Cross-lingual Language Model Pretraining`:
        https://arxiv.org/abs/1901.07291

    .. _`torch.nn.Module`:
        https://pytorch.org/docs/stable/nn.html#module

    .. _`here`:
        https://github.com/facebookresearch/XLM

    Parameters:
        config (:class:`~pytorch_transformers.XLMConfig`): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the configuration.
            Check out the :meth:`~pytorch_transformers.PreTrainedModel.from_pretrained` method to load the model weights.
"""


class XLMPredLayer(nn.Module):
    """
    Prediction layer (cross_entropy or adaptive_softmax).
    """

    def __init__(self, config):
        super(XLMPredLayer, self).__init__()
        self.asm = config.asm
        self.n_words = config.n_words
        self.pad_index = config.pad_index
        dim = config.emb_dim
        if config.asm is False:
            self.proj = nn.Linear(dim, config.n_words, bias=True)
        else:
            self.proj = nn.AdaptiveLogSoftmaxWithLoss(in_features=dim, n_classes=config.n_words, cutoffs=config.asm_cutoffs, div_value=config.asm_div_value, head_bias=True)

    def forward(self, x, y=None):
        """ Compute the loss, and optionally the scores.
        """
        outputs = ()
        if self.asm is False:
            scores = self.proj(x).view(-1, self.n_words)
            outputs = (scores,) + outputs
            if y is not None:
                loss = F.cross_entropy(scores, y, reduction='elementwise_mean')
                outputs = (loss,) + outputs
        else:
            scores = self.proj.log_prob(x)
            outputs = (scores,) + outputs
            if y is not None:
                _, loss = self.proj(x, y)
                outputs = (loss,) + outputs
        return outputs


class XLNetRelativeAttention(nn.Module):

    def __init__(self, config):
        super(XLNetRelativeAttention, self).__init__()
        self.output_attentions = config.output_attentions
        if config.d_model % config.n_head != 0:
            raise ValueError('The hidden size (%d) is not a multiple of the number of attention heads (%d)' % (config.d_model, config.n_head))
        self.n_head = config.n_head
        self.d_head = config.d_head
        self.d_model = config.d_model
        self.scale = 1 / config.d_head ** 0.5
        self.q = nn.Parameter(torch.FloatTensor(config.d_model, self.n_head, self.d_head))
        self.k = nn.Parameter(torch.FloatTensor(config.d_model, self.n_head, self.d_head))
        self.v = nn.Parameter(torch.FloatTensor(config.d_model, self.n_head, self.d_head))
        self.o = nn.Parameter(torch.FloatTensor(config.d_model, self.n_head, self.d_head))
        self.r = nn.Parameter(torch.FloatTensor(config.d_model, self.n_head, self.d_head))
        self.r_r_bias = nn.Parameter(torch.FloatTensor(self.n_head, self.d_head))
        self.r_s_bias = nn.Parameter(torch.FloatTensor(self.n_head, self.d_head))
        self.r_w_bias = nn.Parameter(torch.FloatTensor(self.n_head, self.d_head))
        self.seg_embed = nn.Parameter(torch.FloatTensor(2, self.n_head, self.d_head))
        self.layer_norm = XLNetLayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.dropout)

    def prune_heads(self, heads):
        raise NotImplementedError

    @staticmethod
    def rel_shift(x, klen=-1):
        """perform relative shift to form the relative attention score."""
        x_size = x.shape
        x = x.reshape(x_size[1], x_size[0], x_size[2], x_size[3])
        x = x[1:, ...]
        x = x.reshape(x_size[0], x_size[1] - 1, x_size[2], x_size[3])
        x = torch.index_select(x, 1, torch.arange(klen, device=x.device, dtype=torch.long))
        return x

    def rel_attn_core(self, q_head, k_head_h, v_head_h, k_head_r, seg_mat=None, attn_mask=None, head_mask=None):
        """Core relative positional attention operations."""
        ac = torch.einsum('ibnd,jbnd->ijbn', q_head + self.r_w_bias, k_head_h)
        bd = torch.einsum('ibnd,jbnd->ijbn', q_head + self.r_r_bias, k_head_r)
        bd = self.rel_shift(bd, klen=ac.shape[1])
        if seg_mat is None:
            ef = 0
        else:
            ef = torch.einsum('ibnd,snd->ibns', q_head + self.r_s_bias, self.seg_embed)
            ef = torch.einsum('ijbs,ibns->ijbn', seg_mat, ef)
        attn_score = (ac + bd + ef) * self.scale
        if attn_mask is not None:
            if attn_mask.dtype == torch.float16:
                attn_score = attn_score - 65500 * attn_mask
            else:
                attn_score = attn_score - 1e+30 * attn_mask
        attn_prob = F.softmax(attn_score, dim=1)
        attn_prob = self.dropout(attn_prob)
        if head_mask is not None:
            attn_prob = attn_prob * head_mask
        attn_vec = torch.einsum('ijbn,jbnd->ibnd', attn_prob, v_head_h)
        if self.output_attentions:
            return attn_vec, attn_prob
        return attn_vec

    def post_attention(self, h, attn_vec, residual=True):
        """Post-attention processing."""
        attn_out = torch.einsum('ibnd,hnd->ibh', attn_vec, self.o)
        attn_out = self.dropout(attn_out)
        if residual:
            attn_out = attn_out + h
        output = self.layer_norm(attn_out)
        return output

    def forward(self, h, g, attn_mask_h, attn_mask_g, r, seg_mat, mems=None, target_mapping=None, head_mask=None):
        if g is not None:
            if mems is not None and mems.dim() > 1:
                cat = torch.cat([mems, h], dim=0)
            else:
                cat = h
            k_head_h = torch.einsum('ibh,hnd->ibnd', cat, self.k)
            v_head_h = torch.einsum('ibh,hnd->ibnd', cat, self.v)
            k_head_r = torch.einsum('ibh,hnd->ibnd', r, self.r)
            q_head_h = torch.einsum('ibh,hnd->ibnd', h, self.q)
            attn_vec_h = self.rel_attn_core(q_head_h, k_head_h, v_head_h, k_head_r, seg_mat=seg_mat, attn_mask=attn_mask_h, head_mask=head_mask)
            if self.output_attentions:
                attn_vec_h, attn_prob_h = attn_vec_h
            output_h = self.post_attention(h, attn_vec_h)
            q_head_g = torch.einsum('ibh,hnd->ibnd', g, self.q)
            if target_mapping is not None:
                q_head_g = torch.einsum('mbnd,mlb->lbnd', q_head_g, target_mapping)
                attn_vec_g = self.rel_attn_core(q_head_g, k_head_h, v_head_h, k_head_r, seg_mat=seg_mat, attn_mask=attn_mask_g, head_mask=head_mask)
                if self.output_attentions:
                    attn_vec_g, attn_prob_g = attn_vec_g
                attn_vec_g = torch.einsum('lbnd,mlb->mbnd', attn_vec_g, target_mapping)
            else:
                attn_vec_g = self.rel_attn_core(q_head_g, k_head_h, v_head_h, k_head_r, seg_mat=seg_mat, attn_mask=attn_mask_g, head_mask=head_mask)
                if self.output_attentions:
                    attn_vec_g, attn_prob_g = attn_vec_g
            output_g = self.post_attention(g, attn_vec_g)
            if self.output_attentions:
                attn_prob = attn_prob_h, attn_prob_g
        else:
            if mems is not None and mems.dim() > 1:
                cat = torch.cat([mems, h], dim=0)
            else:
                cat = h
            q_head_h = torch.einsum('ibh,hnd->ibnd', h, self.q)
            k_head_h = torch.einsum('ibh,hnd->ibnd', cat, self.k)
            v_head_h = torch.einsum('ibh,hnd->ibnd', cat, self.v)
            k_head_r = torch.einsum('ibh,hnd->ibnd', r, self.r)
            attn_vec = self.rel_attn_core(q_head_h, k_head_h, v_head_h, k_head_r, seg_mat=seg_mat, attn_mask=attn_mask_h, head_mask=head_mask)
            if self.output_attentions:
                attn_vec, attn_prob = attn_vec
            output_h = self.post_attention(h, attn_vec)
            output_g = None
        outputs = output_h, output_g
        if self.output_attentions:
            outputs = outputs + (attn_prob,)
        return outputs


class XLNetFeedForward(nn.Module):

    def __init__(self, config):
        super(XLNetFeedForward, self).__init__()
        self.layer_norm = XLNetLayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.layer_1 = nn.Linear(config.d_model, config.d_inner)
        self.layer_2 = nn.Linear(config.d_inner, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        if isinstance(config.ff_activation, str) or sys.version_info[0] == 2 and isinstance(config.ff_activation, unicode):
            self.activation_function = ACT2FN[config.ff_activation]
        else:
            self.activation_function = config.ff_activation

    def forward(self, inp):
        output = inp
        output = self.layer_1(output)
        output = self.activation_function(output)
        output = self.dropout(output)
        output = self.layer_2(output)
        output = self.dropout(output)
        output = self.layer_norm(output + inp)
        return output


class XLNetLayer(nn.Module):

    def __init__(self, config):
        super(XLNetLayer, self).__init__()
        self.rel_attn = XLNetRelativeAttention(config)
        self.ff = XLNetFeedForward(config)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, output_h, output_g, attn_mask_h, attn_mask_g, r, seg_mat, mems=None, target_mapping=None, head_mask=None):
        outputs = self.rel_attn(output_h, output_g, attn_mask_h, attn_mask_g, r, seg_mat, mems=mems, target_mapping=target_mapping, head_mask=head_mask)
        output_h, output_g = outputs[:2]
        if output_g is not None:
            output_g = self.ff(output_g)
        output_h = self.ff(output_h)
        outputs = (output_h, output_g) + outputs[2:]
        return outputs


XLNET_PRETRAINED_MODEL_ARCHIVE_MAP = {'xlnet-base-cased': 'https://s3.amazonaws.com/models.huggingface.co/bert/xlnet-base-cased-pytorch_model.bin', 'xlnet-large-cased': 'https://s3.amazonaws.com/models.huggingface.co/bert/xlnet-large-cased-pytorch_model.bin'}


XLNET_PRETRAINED_CONFIG_ARCHIVE_MAP = {'xlnet-base-cased': 'https://s3.amazonaws.com/models.huggingface.co/bert/xlnet-base-cased-config.json', 'xlnet-large-cased': 'https://s3.amazonaws.com/models.huggingface.co/bert/xlnet-large-cased-config.json'}


class XLNetConfig(PretrainedConfig):
    """Configuration class to store the configuration of a ``XLNetModel``.

    Args:
        vocab_size_or_config_json_file: Vocabulary size of ``inputs_ids`` in ``XLNetModel``.
        d_model: Size of the encoder layers and the pooler layer.
        n_layer: Number of hidden layers in the Transformer encoder.
        n_head: Number of attention heads for each attention layer in
            the Transformer encoder.
        d_inner: The size of the "intermediate" (i.e., feed-forward)
            layer in the Transformer encoder.
        ff_activation: The non-linear activation function (function or string) in the
            encoder and pooler. If string, "gelu", "relu" and "swish" are supported.
        untie_r: untie relative position biases
        attn_type: 'bi' for XLNet, 'uni' for Transformer-XL

        dropout: The dropout probabilitiy for all fully connected
            layers in the embeddings, encoder, and pooler.
        dropatt: The dropout ratio for the attention
            probabilities.
        initializer_range: The sttdev of the truncated_normal_initializer for
            initializing all weight matrices.
        layer_norm_eps: The epsilon used by LayerNorm.

        dropout: float, dropout rate.
        dropatt: float, dropout rate on attention probabilities.
        init: str, the initialization scheme, either "normal" or "uniform".
        init_range: float, initialize the parameters with a uniform distribution
            in [-init_range, init_range]. Only effective when init="uniform".
        init_std: float, initialize the parameters with a normal distribution
            with mean 0 and stddev init_std. Only effective when init="normal".
        mem_len: int, the number of tokens to cache.
        reuse_len: int, the number of tokens in the currect batch to be cached
            and reused in the future.
        bi_data: bool, whether to use bidirectional input pipeline.
            Usually set to True during pretraining and False during finetuning.
        clamp_len: int, clamp all relative distances larger than clamp_len.
            -1 means no clamping.
        same_length: bool, whether to use the same attention length for each token.
        finetuning_task: name of the glue task on which the model was fine-tuned if any
    """
    pretrained_config_archive_map = XLNET_PRETRAINED_CONFIG_ARCHIVE_MAP

    def __init__(self, vocab_size_or_config_json_file=32000, d_model=1024, n_layer=24, n_head=16, d_inner=4096, ff_activation='gelu', untie_r=True, attn_type='bi', initializer_range=0.02, layer_norm_eps=1e-12, dropout=0.1, mem_len=None, reuse_len=None, bi_data=False, clamp_len=-1, same_length=False, finetuning_task=None, num_labels=2, summary_type='last', summary_use_proj=True, summary_activation='tanh', summary_last_dropout=0.1, start_n_top=5, end_n_top=5, **kwargs):
        """Constructs XLNetConfig.
        """
        super(XLNetConfig, self).__init__(**kwargs)
        if isinstance(vocab_size_or_config_json_file, str) or sys.version_info[0] == 2 and isinstance(vocab_size_or_config_json_file, unicode):
            with open(vocab_size_or_config_json_file, 'r', encoding='utf-8') as reader:
                json_config = json.loads(reader.read())
            for key, value in json_config.items():
                self.__dict__[key] = value
        elif isinstance(vocab_size_or_config_json_file, int):
            self.n_token = vocab_size_or_config_json_file
            self.d_model = d_model
            self.n_layer = n_layer
            self.n_head = n_head
            assert d_model % n_head == 0
            self.d_head = d_model // n_head
            self.ff_activation = ff_activation
            self.d_inner = d_inner
            self.untie_r = untie_r
            self.attn_type = attn_type
            self.initializer_range = initializer_range
            self.layer_norm_eps = layer_norm_eps
            self.dropout = dropout
            self.mem_len = mem_len
            self.reuse_len = reuse_len
            self.bi_data = bi_data
            self.clamp_len = clamp_len
            self.same_length = same_length
            self.finetuning_task = finetuning_task
            self.num_labels = num_labels
            self.summary_type = summary_type
            self.summary_use_proj = summary_use_proj
            self.summary_activation = summary_activation
            self.summary_last_dropout = summary_last_dropout
            self.start_n_top = start_n_top
            self.end_n_top = end_n_top
        else:
            raise ValueError('First argument must be either a vocabulary size (int) or the path to a pretrained model config file (str)')

    @property
    def max_position_embeddings(self):
        return -1

    @property
    def vocab_size(self):
        return self.n_token

    @vocab_size.setter
    def vocab_size(self, value):
        self.n_token = value

    @property
    def hidden_size(self):
        return self.d_model

    @property
    def num_attention_heads(self):
        return self.n_head

    @property
    def num_hidden_layers(self):
        return self.n_layer


XLNET_INPUTS_DOCSTRING = """
    Inputs:
        **input_ids**: ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            Indices of input sequence tokens in the vocabulary.
            XLNet is a model with relative position embeddings so you can either pad the inputs on
            the right or on the left.
            Indices can be obtained using :class:`pytorch_transformers.XLNetTokenizer`.
            See :func:`pytorch_transformers.PreTrainedTokenizer.encode` and
            :func:`pytorch_transformers.PreTrainedTokenizer.convert_tokens_to_ids` for details.
        **token_type_ids**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            A parallel sequence of tokens (can be used to indicate various portions of the inputs).
            The embeddings from these tokens will be summed with the respective token embeddings.
            Indices are selected in the vocabulary (unlike BERT which has a specific vocabulary for segment indices).
        **attention_mask**: (`optional`) ``torch.FloatTensor`` of shape ``(batch_size, sequence_length)``:
            Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            ``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens.
        **input_mask**: (`optional`) ``torch.FloatTensor`` of shape ``(batch_size, sequence_length)``:
            Mask to avoid performing attention on padding token indices.
            Negative of `attention_mask`, i.e. with 0 for real tokens and 1 for padding.
            Kept for compatibility with the original code base.
            You can only uses one of `input_mask` and `attention_mask`
            Mask values selected in ``[0, 1]``:
            ``1`` for tokens that are MASKED, ``0`` for tokens that are NOT MASKED.
        **mems**: (`optional`)
            list of ``torch.FloatTensor`` (one for each layer):
            that contains pre-computed hidden-states (key and values in the attention blocks) as output by the model
            (see `mems` output below). Can be used to speed up sequential decoding and attend to longer context.
            To activate mems you need to set up config.mem_len to a positive value which will be the max number of tokens in
            the memory output by the model. E.g. `model = XLNetModel.from_pretrained('xlnet-base-case, mem_len=1024)` will
            instantiate a model which can use up to 1024 tokens of memory (in addition to the input it self).
        **perm_mask**: (`optional`) ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, sequence_length)``:
            Mask to indicate the attention pattern for each input token with values selected in ``[0, 1]``:
            If ``perm_mask[k, i, j] = 0``, i attend to j in batch k;
            if ``perm_mask[k, i, j] = 1``, i does not attend to j in batch k.
            If None, each token attends to all the others (full bidirectional attention).
            Only used during pretraining (to define factorization order) or for sequential decoding (generation).
        **target_mapping**: (`optional`) ``torch.FloatTensor`` of shape ``(batch_size, num_predict, sequence_length)``:
            Mask to indicate the output tokens to use.
            If ``target_mapping[k, i, j] = 1``, the i-th predict in batch k is on the j-th token.
            Only used during pretraining for partial prediction or for sequential decoding (generation).
        **head_mask**: (`optional`) ``torch.FloatTensor`` of shape ``(num_heads,)`` or ``(num_layers, num_heads)``:
            Mask to nullify selected heads of the self-attention modules.
            Mask values selected in ``[0, 1]``:
            ``1`` indicates the head is **not masked**, ``0`` indicates the head is **masked**.
"""


XLNET_START_DOCSTRING = """    The XLNet model was proposed in
    `XLNet: Generalized Autoregressive Pretraining for Language Understanding`_
    by Zhilin Yang*, Zihang Dai*, Yiming Yang, Jaime Carbonell, Ruslan Salakhutdinov, Quoc V. Le.
    XLnet is an extension of the Transformer-XL model pre-trained using an autoregressive method
    to learn bidirectional contexts by maximizing the expected likelihood over all permutations
    of the input sequence factorization order.

    The specific attention pattern can be controlled at training and test time using the `perm_mask` input.

    Do to the difficulty of training a fully auto-regressive model over various factorization order,
    XLNet is pretrained using only a sub-set of the output tokens as target which are selected
    with the `target_mapping` input.

    To use XLNet for sequential decoding (i.e. not in fully bi-directional setting), use the `perm_mask` and
    `target_mapping` inputs to control the attention span and outputs (see examples in `examples/run_generation.py`)

    This model is a PyTorch `torch.nn.Module`_ sub-class. Use it as a regular PyTorch Module and
    refer to the PyTorch documentation for all matter related to general usage and behavior.

    .. _`XLNet: Generalized Autoregressive Pretraining for Language Understanding`:
        http://arxiv.org/abs/1906.08237

    .. _`torch.nn.Module`:
        https://pytorch.org/docs/stable/nn.html#module

    Parameters:
        config (:class:`~pytorch_transformers.XLNetConfig`): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the configuration.
            Check out the :meth:`~pytorch_transformers.PreTrainedModel.from_pretrained` method to load the model weights.
"""


def build_tf_xlnet_to_pytorch_map(model, config, tf_weights=None):
    """ A map of modules from TF to PyTorch.
        I use a map to keep the PyTorch model as
        identical to the original PyTorch model as possible.
    """
    tf_to_pt_map = {}
    if hasattr(model, 'transformer'):
        if hasattr(model, 'lm_loss'):
            tf_to_pt_map['model/lm_loss/bias'] = model.lm_loss.bias
        if hasattr(model, 'sequence_summary') and 'model/sequnece_summary/summary/kernel' in tf_weights:
            tf_to_pt_map['model/sequnece_summary/summary/kernel'] = model.sequence_summary.summary.weight
            tf_to_pt_map['model/sequnece_summary/summary/bias'] = model.sequence_summary.summary.bias
        if hasattr(model, 'logits_proj') and config.finetuning_task is not None and 'model/regression_{}/logit/kernel'.format(config.finetuning_task) in tf_weights:
            tf_to_pt_map['model/regression_{}/logit/kernel'.format(config.finetuning_task)] = model.logits_proj.weight
            tf_to_pt_map['model/regression_{}/logit/bias'.format(config.finetuning_task)] = model.logits_proj.bias
        model = model.transformer
    tf_to_pt_map.update({'model/transformer/word_embedding/lookup_table': model.word_embedding.weight, 'model/transformer/mask_emb/mask_emb': model.mask_emb})
    for i, b in enumerate(model.layer):
        layer_str = 'model/transformer/layer_%d/' % i
        tf_to_pt_map.update({(layer_str + 'rel_attn/LayerNorm/gamma'): b.rel_attn.layer_norm.weight, (layer_str + 'rel_attn/LayerNorm/beta'): b.rel_attn.layer_norm.bias, (layer_str + 'rel_attn/o/kernel'): b.rel_attn.o, (layer_str + 'rel_attn/q/kernel'): b.rel_attn.q, (layer_str + 'rel_attn/k/kernel'): b.rel_attn.k, (layer_str + 'rel_attn/r/kernel'): b.rel_attn.r, (layer_str + 'rel_attn/v/kernel'): b.rel_attn.v, (layer_str + 'ff/LayerNorm/gamma'): b.ff.layer_norm.weight, (layer_str + 'ff/LayerNorm/beta'): b.ff.layer_norm.bias, (layer_str + 'ff/layer_1/kernel'): b.ff.layer_1.weight, (layer_str + 'ff/layer_1/bias'): b.ff.layer_1.bias, (layer_str + 'ff/layer_2/kernel'): b.ff.layer_2.weight, (layer_str + 'ff/layer_2/bias'): b.ff.layer_2.bias})
    if config.untie_r:
        r_r_list = []
        r_w_list = []
        r_s_list = []
        seg_embed_list = []
        for b in model.layer:
            r_r_list.append(b.rel_attn.r_r_bias)
            r_w_list.append(b.rel_attn.r_w_bias)
            r_s_list.append(b.rel_attn.r_s_bias)
            seg_embed_list.append(b.rel_attn.seg_embed)
    else:
        r_r_list = [model.r_r_bias]
        r_w_list = [model.r_w_bias]
        r_s_list = [model.r_s_bias]
        seg_embed_list = [model.seg_embed]
    tf_to_pt_map.update({'model/transformer/r_r_bias': r_r_list, 'model/transformer/r_w_bias': r_w_list, 'model/transformer/r_s_bias': r_s_list, 'model/transformer/seg_embed': seg_embed_list})
    return tf_to_pt_map


def load_tf_weights_in_xlnet(model, config, tf_path):
    """ Load tf checkpoints in a pytorch model
    """
    try:
        import numpy as np
        import tensorflow as tf
    except ImportError:
        logger.error('Loading a TensorFlow models in PyTorch, requires TensorFlow to be installed. Please see https://www.tensorflow.org/install/ for installation instructions.')
        raise
    init_vars = tf.train.list_variables(tf_path)
    tf_weights = {}
    for name, shape in init_vars:
        logger.info('Loading TF weight {} with shape {}'.format(name, shape))
        array = tf.train.load_variable(tf_path, name)
        tf_weights[name] = array
    tf_to_pt_map = build_tf_xlnet_to_pytorch_map(model, config, tf_weights)
    for name, pointer in tf_to_pt_map.items():
        logger.info('Importing {}'.format(name))
        if name not in tf_weights:
            logger.info('{} not in tf pre-trained weights, skipping'.format(name))
            continue
        array = tf_weights[name]
        if 'kernel' in name and ('ff' in name or 'summary' in name or 'logit' in name):
            logger.info('Transposing')
            array = np.transpose(array)
        if isinstance(pointer, list):
            assert len(pointer) == array.shape[0]
            for i, p_i in enumerate(pointer):
                arr_i = array[i, ...]
                try:
                    assert p_i.shape == arr_i.shape
                except AssertionError as e:
                    e.args += p_i.shape, arr_i.shape
                    raise
                logger.info('Initialize PyTorch weight {} for layer {}'.format(name, i))
                p_i.data = torch.from_numpy(arr_i)
        else:
            try:
                assert pointer.shape == array.shape
            except AssertionError as e:
                e.args += pointer.shape, array.shape
                raise
            logger.info('Initialize PyTorch weight {}'.format(name))
            pointer.data = torch.from_numpy(array)
        tf_weights.pop(name, None)
        tf_weights.pop(name + '/Adam', None)
        tf_weights.pop(name + '/Adam_1', None)
    logger.info('Weights not copied to PyTorch model: {}'.format(', '.join(tf_weights.keys())))
    return model


class XLNetPreTrainedModel(PreTrainedModel):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """
    config_class = XLNetConfig
    pretrained_model_archive_map = XLNET_PRETRAINED_MODEL_ARCHIVE_MAP
    load_tf_weights = load_tf_weights_in_xlnet
    base_model_prefix = 'transformer'

    def _init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, XLNetLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, XLNetRelativeAttention):
            for param in [module.q, module.k, module.v, module.o, module.r, module.r_r_bias, module.r_s_bias, module.r_w_bias, module.seg_embed]:
                param.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, XLNetModel):
            module.mask_emb.data.normal_(mean=0.0, std=self.config.initializer_range)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Attention,
     lambda: ([], {'nx': 4, 'n_ctx': 4, 'config': _mock_config(n_head=4, output_attentions=4, attn_pdrop=0.5, resid_pdrop=0.5)}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (BertIntermediate,
     lambda: ([], {'config': _mock_config(hidden_size=4, intermediate_size=4, hidden_act=_mock_layer())}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (BertOnlyNSPHead,
     lambda: ([], {'config': _mock_config(hidden_size=4)}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (BertPooler,
     lambda: ([], {'config': _mock_config(hidden_size=4)}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (BertSelfAttention,
     lambda: ([], {'config': _mock_config(hidden_size=4, num_attention_heads=4, output_attentions=4, attention_probs_dropout_prob=0.5)}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {}),
     False),
    (Conv1D,
     lambda: ([], {'nf': 4, 'nx': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (FFN,
     lambda: ([], {'config': _mock_config(dropout=0.5, dim=4, hidden_dim=4, activation='relu')}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (MultiHeadAttention,
     lambda: ([], {'n_heads': 4, 'dim': 4, 'config': _mock_config(output_attentions=4, attention_dropout=0.5)}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 1, 1, 4])], {}),
     False),
    (PoolerStartLogits,
     lambda: ([], {'config': _mock_config(hidden_size=4)}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (RobertaClassificationHead,
     lambda: ([], {'config': _mock_config(hidden_size=4, hidden_dropout_prob=0.5, num_labels=4)}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (SQuADHead,
     lambda: ([], {'config': _mock_config(start_n_top=4, end_n_top=4, hidden_size=4, layer_norm_eps=1)}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (TransformerFFN,
     lambda: ([], {'in_dim': 4, 'dim_hidden': 4, 'out_dim': 4, 'config': _mock_config(dropout=0.5, gelu_activation=4)}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_facebookresearch_UnsupervisedDecomposition(_paritybench_base):
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

    def test_005(self):
        self._check(*TESTCASES[5])

    def test_006(self):
        self._check(*TESTCASES[6])

    def test_007(self):
        self._check(*TESTCASES[7])

    def test_008(self):
        self._check(*TESTCASES[8])

    def test_009(self):
        self._check(*TESTCASES[9])

    def test_010(self):
        self._check(*TESTCASES[10])

    def test_011(self):
        self._check(*TESTCASES[11])

