
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


import warnings


import time


from itertools import chain


import torch


import numpy as np


import copy


from random import randint


import re


from typing import List


from typing import Optional


from typing import Union


from torch.fx.experimental.proxy_tensor import make_fx


import tensorflow as tf


import torchvision.models as models


import logging


from torch._dynamo import register_backend


from torch._decomp import get_decompositions


from torch.nn.utils import stateless


from torch import fx


import functools


from torch._functorch.compile_utils import strip_overloads


from torch.func import functionalize


from torch.utils.cpp_extension import load_inline


from torch.utils.cpp_extension import include_paths


import torch.nn as nn


import torch.nn.functional as F


from typing import Dict


from typing import Any


from torchvision import transforms


from collections import OrderedDict


from torch import nn


from typing import Tuple


import inspect


from typing import Callable


import itertools


import math


import random


import torch.utils.checkpoint


from torch.utils.data import Dataset


import torch._dynamo as dynamo


from collections import defaultdict


import torch.fx


from torch.fx.node import Node


from torch.utils._pytree import tree_map


from typing import Iterable


from torch.nn import CrossEntropyLoss


from functools import partial


from torch.nn import functional as F


from torchvision.transforms import functional as TF


pretrained_model_name_or_path = 'stabilityai/stable-diffusion-2'


class UnetModel(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.unet = unet

    def forward(self, x, y, z):
        return self.unet.forward(x, y, z, return_dict=False)[0]


class HuggingFaceLanguage(torch.nn.Module):

    def __init__(self, hf_model_name):
        super().__init__()
        transformers_path = trf.__path__[0]
        hf_model_path = f'{transformers_path}/models/{hf_model_name}'
        self.model = AutoModelForSequenceClassification.from_pretrained(hf_model_name, num_labels=2, output_attentions=False, output_hidden_states=False, torchscript=True)

    def forward(self, tokens):
        return self.model.forward(tokens)[0]


class VisionModule(torch.nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.train(False)

    def forward(self, input):
        return self.model.forward(input)


class ResidualDenseBlock_5C(nn.Module):

    def __init__(self, nf=64, gc=32, bias=True):
        super(ResidualDenseBlock_5C, self).__init__()
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    """Residual in Residual Dense Block"""

    def __init__(self, nf, gc=32):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(nf, gc)
        self.RDB2 = ResidualDenseBlock_5C(nf, gc)
        self.RDB3 = ResidualDenseBlock_5C(nf, gc)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + x


def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)


class RRDBNet(nn.Module):

    def __init__(self, in_nc, out_nc, nf, nb, gc=32):
        super(RRDBNet, self).__init__()
        RRDB_block_f = functools.partial(RRDB, nf=nf, gc=gc)
        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.RRDB_trunk = make_layer(RRDB_block_f, nb)
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.upconv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        fea = self.conv_first(x)
        trunk = self.trunk_conv(self.RRDB_trunk(fea))
        fea = fea + trunk
        fea = self.lrelu(self.upconv1(F.interpolate(fea, scale_factor=2, mode='nearest')))
        fea = self.lrelu(self.upconv2(F.interpolate(fea, scale_factor=2, mode='nearest')))
        out = self.conv_last(self.lrelu(self.HRconv(fea)))
        return out


class AlbertModule(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.model = AutoModelForMaskedLM.from_pretrained('albert-base-v2')
        self.model.eval()

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask).logits


class MegaModel(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.model = MEGABYTE(num_tokens=16000, dim=(512, 256), max_seq_len=(1024, 4), depth=(6, 4), dim_head=64, heads=8, flash_attn=True)

    def forward(self, input):
        return self.model(input)


class MiniLMSequenceClassification(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained('microsoft/MiniLM-L12-H384-uncased', num_labels=2, output_attentions=False, output_hidden_states=False, torchscript=True)

    def forward(self, tokens):
        return self.model.forward(tokens)[0]


class ResnestModule(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.model = torch.hub.load('zhanghang1989/ResNeSt', 'resnest50', pretrained=True)
        self.model.eval()

    def forward(self, input):
        return self.model.forward(input)


class Resnet50Module(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.resnet = models.resnet50(pretrained=True)
        self.train(False)

    def forward(self, img):
        return self.resnet.forward(img)


class DLRM_Net(nn.Module):

    def create_mlp(self, ln, sigmoid_layer):
        layers = nn.ModuleList()
        for i in range(0, ln.size - 1):
            n = ln[i]
            m = ln[i + 1]
            LL = nn.Linear(int(n), int(m), bias=True)
            mean = 0.0
            std_dev = np.sqrt(2 / (m + n))
            W = np.random.normal(mean, std_dev, size=(m, n)).astype(np.float32)
            std_dev = np.sqrt(1 / m)
            bt = np.random.normal(mean, std_dev, size=m).astype(np.float32)
            LL.weight.data = torch.tensor(W, requires_grad=True)
            LL.bias.data = torch.tensor(bt, requires_grad=True)
            layers.append(LL)
            if i == sigmoid_layer:
                layers.append(nn.Sigmoid())
            else:
                layers.append(nn.ReLU())
        return torch.nn.Sequential(*layers)

    def create_emb(self, m, ln, weighted_pooling=None):
        emb_l = nn.ModuleList()
        v_W_l = []
        for i in range(0, ln.size):
            n = ln[i]
            EE = nn.EmbeddingBag(n, m, mode='sum')
            W = np.random.uniform(low=-np.sqrt(1 / n), high=np.sqrt(1 / n), size=(n, m)).astype(np.float32)
            None
            EE.weight.data = torch.tensor(W, requires_grad=True)
            if weighted_pooling is None:
                v_W_l.append(None)
            else:
                v_W_l.append(torch.ones(n, dtype=torch.float32))
            emb_l.append(EE)
        return emb_l, v_W_l

    def __init__(self, m_spa=None, ln_emb=None, ln_bot=None, ln_top=None, arch_interaction_op=None, arch_interaction_itself=False, sigmoid_bot=-1, sigmoid_top=-1, weighted_pooling=None):
        super(DLRM_Net, self).__init__()
        if m_spa is not None and ln_emb is not None and ln_bot is not None and ln_top is not None and arch_interaction_op is not None:
            self.output_d = 0
            self.arch_interaction_op = arch_interaction_op
            self.arch_interaction_itself = arch_interaction_itself
            if weighted_pooling is not None and weighted_pooling != 'fixed':
                self.weighted_pooling = 'learned'
            else:
                self.weighted_pooling = weighted_pooling
            self.emb_l, w_list = self.create_emb(m_spa, ln_emb, weighted_pooling)
            if self.weighted_pooling == 'learned':
                self.v_W_l = nn.ParameterList()
                for w in w_list:
                    self.v_W_l.append(nn.Parameter(w))
            else:
                self.v_W_l = w_list
            self.bot_l = self.create_mlp(ln_bot, sigmoid_bot)
            self.top_l = self.create_mlp(ln_top, sigmoid_top)

    def apply_mlp(self, x, layers):
        return layers(x)

    def apply_emb(self, lS_o, lS_i, emb_l, v_W_l):
        ly = []
        for k, sparse_index_group_batch in enumerate(lS_i):
            sparse_offset_group_batch = lS_o[k]
            if v_W_l[k] is not None:
                per_sample_weights = v_W_l[k].gather(0, sparse_index_group_batch)
            else:
                per_sample_weights = None
            E = emb_l[k]
            V = E(sparse_index_group_batch, sparse_offset_group_batch, per_sample_weights=per_sample_weights)
            ly.append(V)
        return ly

    def interact_features(self, x, ly):
        if self.arch_interaction_op == 'dot':
            batch_size, d = x.shape
            T = torch.cat([x] + ly, dim=1).view((batch_size, -1, d))
            Z = torch.bmm(T, torch.transpose(T, 1, 2))
            _, ni, nj = Z.shape
            offset = 1 if self.arch_interaction_itself else 0
            li = torch.tensor([i for i in range(ni) for j in range(i + offset)])
            lj = torch.tensor([j for i in range(nj) for j in range(i + offset)])
            Zflat = Z[:, li, lj]
            R = torch.cat([x] + [Zflat], dim=1)
        elif self.arch_interaction_op == 'cat':
            R = torch.cat([x] + ly, dim=1)
        else:
            sys.exit('ERROR: --arch-interaction-op=' + self.arch_interaction_op + ' is not supported')
        return R

    def forward(self, dense_x, lS_o, *lS_i):
        return self.sequential_forward(dense_x, lS_o, lS_i)

    def sequential_forward(self, dense_x, lS_o, lS_i):
        x = self.apply_mlp(dense_x, self.bot_l)
        ly = self.apply_emb(lS_o, lS_i, self.emb_l, self.v_W_l)
        z = self.interact_features(x, ly)
        p = self.apply_mlp(z, self.top_l)
        return p


class SparseArchShark(nn.Module):

    def create_emb(self, embedding_dim, num_embeddings_list):
        embedding_list = nn.ModuleList()
        for i in range(0, num_embeddings_list.size):
            num_embeddings = num_embeddings_list[i]
            EE = nn.EmbeddingBag(num_embeddings, embedding_dim, mode='sum')
            W = np.random.uniform(low=-np.sqrt(1 / num_embeddings), high=np.sqrt(1 / num_embeddings), size=(num_embeddings, embedding_dim)).astype(np.float32)
            EE.weight.data = torch.tensor(W, requires_grad=True)
            embedding_list.append(EE)
        return embedding_list

    def __init__(self, embedding_dim, total_features, num_embeddings_list):
        super(SparseArchShark, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_features = total_features
        self.embedding_list = self.create_emb(embedding_dim, num_embeddings_list)

    def forward(self, *batched_inputs):
        concatenated_list = []
        input_enum, embedding_enum = 0, 0
        for k in range(len(batched_inputs) // 3):
            values = batched_inputs[input_enum]
            input_enum += 1
            offsets = batched_inputs[input_enum]
            input_enum += 1
            embedding_pointer = int(batched_inputs[input_enum])
            input_enum += 1
            E = self.embedding_list[embedding_pointer]
            V = E(values, offsets)
            concatenated_list.append(V)
        return torch.cat(concatenated_list, dim=1).reshape(-1, self.num_features, self.embedding_dim)


class DLRMShark(nn.Module):

    def __init__(self, embedding_dim, total_features, num_embeddings_list, dense_in_features: 'int', dense_arch_layer_sizes: 'List[int]', over_arch_layer_sizes: 'List[int]') ->None:
        super().__init__()
        self.sparse_arch: 'SparseArchShark' = SparseArchShark(embedding_dim, total_features, num_embeddings_list)
        num_sparse_features: 'int' = total_features
        self.dense_arch = DenseArch(in_features=dense_in_features, layer_sizes=dense_arch_layer_sizes)
        self.inter_arch = InteractionArch(num_sparse_features=num_sparse_features)
        over_in_features: 'int' = embedding_dim + choose(num_sparse_features, 2) + num_sparse_features
        self.over_arch = OverArch(in_features=over_in_features, layer_sizes=over_arch_layer_sizes)

    def forward(self, dense_features: 'torch.Tensor', *sparse_features) ->torch.Tensor:
        embedded_dense = self.dense_arch(dense_features)
        embedded_sparse = self.sparse_arch(*sparse_features)
        concatenated_dense = self.inter_arch(dense_features=embedded_dense, sparse_features=embedded_sparse)
        logits = self.over_arch(concatenated_dense)
        return logits


class UnetModule(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet', in_channels=3, out_channels=1, init_features=32, pretrained=True)
        self.model.eval()

    def forward(self, input):
        return self.model(input)


class Foo(torch.nn.Module):

    def __init__(self):
        super(Foo, self).__init__()
        self.l1 = torch.nn.Linear(10, 16)
        self.relu = torch.nn.ReLU()
        self.l2 = torch.nn.Linear(16, 2)

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        return out


class VaeModel(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.vae = vae

    def forward(self, input):
        x = self.vae.encode(input, return_dict=False)[0]
        return x


class OPTForCausalLMModel(torch.nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        combine_input_dict = {'input_ids': input_ids, 'attention_mask': attention_mask}
        output = self.model(**combine_input_dict)
        return output.logits


class OPTLearnedPositionalEmbedding(nn.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    """

    def __init__(self, num_embeddings: 'int', embedding_dim: 'int'):
        self.offset = 2
        super().__init__(num_embeddings + self.offset, embedding_dim)

    def forward(self, attention_mask: 'torch.LongTensor', past_key_values_length: 'int'=0):
        """`input_ids_shape` is expected to be [bsz x seqlen]."""
        attention_mask = attention_mask.long()
        positions = (torch.cumsum(attention_mask, dim=1).type_as(attention_mask) * attention_mask).long() - 1
        positions = positions[:, past_key_values_length:]
        return super().forward(positions + self.offset)


class OPTAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, embed_dim: 'int', num_heads: 'int', dropout: 'float'=0.0, is_decoder: 'bool'=False, bias: 'bool'=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        if self.head_dim * num_heads != self.embed_dim:
            raise ValueError(f'embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {num_heads}).')
        self.scaling = self.head_dim ** -0.5
        self.is_decoder = is_decoder
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _shape(self, tensor: 'torch.Tensor', seq_len: 'int', bsz: 'int'):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(self, hidden_states: 'torch.Tensor', key_value_states: 'Optional[torch.Tensor]'=None, past_key_value: 'Optional[Tuple[torch.Tensor]]'=None, attention_mask: 'Optional[torch.Tensor]'=None, layer_head_mask: 'Optional[torch.Tensor]'=None, output_attentions: 'bool'=False) ->Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""
        is_cross_attention = key_value_states is not None
        bsz, tgt_len, _ = hidden_states.size()
        query_states = self.q_proj(hidden_states) * self.scaling
        if is_cross_attention and past_key_value is not None:
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
        if self.is_decoder:
            past_key_value = key_states, value_states
        proj_shape = bsz * self.num_heads, -1, self.head_dim
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)
        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(f'Attention weights should be of size {bsz * self.num_heads, tgt_len, src_len}, but is {attn_weights.size()}')
        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(f'Attention mask should be of size {bsz, 1, tgt_len, src_len}, but is {attention_mask.size()}')
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        if layer_head_mask is not None:
            if layer_head_mask.size() != (self.num_heads,):
                raise ValueError(f'Head mask for a single layer should be of size {self.num_heads,}, but is {layer_head_mask.size()}')
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        if output_attentions:
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None
        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        attn_output = torch.bmm(attn_probs, value_states)
        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(f'`attn_output` should be of size {bsz, self.num_heads, tgt_len, self.head_dim}, but is {attn_output.size()}')
        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
        attn_output = self.out_proj(attn_output)
        return attn_output, attn_weights_reshaped, past_key_value


class OPTDecoderLayer(nn.Module):

    def __init__(self, config: 'OPTConfig'):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = OPTAttention(embed_dim=self.embed_dim, num_heads=config.num_attention_heads, dropout=config.attention_dropout, is_decoder=True, bias=config.enable_bias)
        self.do_layer_norm_before = config.do_layer_norm_before
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim, elementwise_affine=config.layer_norm_elementwise_affine)
        self.fc1 = nn.Linear(self.embed_dim, config.ffn_dim)
        self.fc2 = nn.Linear(config.ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim, elementwise_affine=config.layer_norm_elementwise_affine)

    def forward(self, hidden_states: 'torch.Tensor', attention_mask: 'Optional[torch.Tensor]'=None, layer_head_mask: 'Optional[torch.Tensor]'=None, output_attentions: 'Optional[bool]'=False, use_cache: 'Optional[bool]'=False, past_key_value: 'Optional[Tuple[torch.Tensor]]'=None) ->Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states
        if self.do_layer_norm_before:
            hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states, self_attn_weights, present_key_value = self.self_attn(hidden_states=hidden_states, past_key_value=past_key_value, attention_mask=attention_mask, layer_head_mask=layer_head_mask, output_attentions=output_attentions)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        if not self.do_layer_norm_before:
            hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states_shape = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, hidden_states.size(-1))
        residual = hidden_states
        if self.do_layer_norm_before:
            hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = (residual + hidden_states).view(hidden_states_shape)
        if not self.do_layer_norm_before:
            hidden_states = self.final_layer_norm(hidden_states)
        outputs = hidden_states,
        if output_attentions:
            outputs += self_attn_weights,
        if use_cache:
            outputs += present_key_value,
        return outputs


class ResidualBlock(nn.Module):

    def __init__(self, main, skip=None):
        super().__init__()
        self.main = nn.Sequential(*main)
        self.skip = skip if skip else nn.Identity()

    def forward(self, input):
        return self.main(input) + self.skip(input)


class ResLinearBlock(ResidualBlock):

    def __init__(self, f_in, f_mid, f_out, is_last=False):
        skip = None if f_in == f_out else nn.Linear(f_in, f_out, bias=False)
        super().__init__([nn.Linear(f_in, f_mid), nn.ReLU(inplace=True), nn.Linear(f_mid, f_out), nn.ReLU(inplace=True) if not is_last else nn.Identity()], skip)


class Modulation2d(nn.Module):

    def __init__(self, state, feats_in, c_out):
        super().__init__()
        self.state = state
        self.layer = nn.Linear(feats_in, c_out * 2, bias=False)

    def forward(self, input):
        scales, shifts = self.layer(self.state['cond']).chunk(2, dim=-1)
        return torch.addcmul(shifts[..., None, None], input, scales[..., None, None] + 1)


class ResModConvBlock(ResidualBlock):

    def __init__(self, state, feats_in, c_in, c_mid, c_out, is_last=False):
        skip = None if c_in == c_out else nn.Conv2d(c_in, c_out, 1, bias=False)
        super().__init__([nn.Conv2d(c_in, c_mid, 3, padding=1), nn.GroupNorm(1, c_mid, affine=False), Modulation2d(state, feats_in, c_mid), nn.ReLU(inplace=True), nn.Conv2d(c_mid, c_out, 3, padding=1), nn.GroupNorm(1, c_out, affine=False) if not is_last else nn.Identity(), Modulation2d(state, feats_in, c_out) if not is_last else nn.Identity(), nn.ReLU(inplace=True) if not is_last else nn.Identity()], skip)


class SkipBlock(nn.Module):

    def __init__(self, main, skip=None):
        super().__init__()
        self.main = nn.Sequential(*main)
        self.skip = skip if skip else nn.Identity()

    def forward(self, input):
        return torch.cat([self.main(input), self.skip(input)], dim=1)


class FourierFeatures(nn.Module):

    def __init__(self, in_features, out_features, std=1.0):
        super().__init__()
        assert out_features % 2 == 0
        self.weight = nn.Parameter(torch.randn([out_features // 2, in_features]) * std)
        self.weight.requires_grad_(False)

    def forward(self, input):
        f = 2 * math.pi * input @ self.weight.T
        return torch.cat([f.cos(), f.sin()], dim=-1)


class SelfAttention2d(nn.Module):

    def __init__(self, c_in, n_head=1, dropout_rate=0.1):
        super().__init__()
        assert c_in % n_head == 0
        self.norm = nn.GroupNorm(1, c_in)
        self.n_head = n_head
        self.qkv_proj = nn.Conv2d(c_in, c_in * 3, 1)
        self.out_proj = nn.Conv2d(c_in, c_in, 1)
        self.dropout = nn.Identity()

    def forward(self, input):
        n, c, h, w = input.shape
        qkv = self.qkv_proj(self.norm(input))
        qkv = qkv.view([n, self.n_head * 3, c // self.n_head, h * w]).transpose(2, 3)
        q, k, v = qkv.chunk(3, dim=1)
        scale = k.shape[3] ** -0.25
        att = (q * scale @ (k.transpose(2, 3) * scale)).softmax(3)
        y = (att @ v).transpose(2, 3).contiguous().view([n, c, h, w])
        return input + self.dropout(self.out_proj(y))


class CC12M1Model(nn.Module):

    def __init__(self):
        super().__init__()
        self.shape = 3, 256, 256
        self.clip_model = 'ViT-B/16'
        self.min_t = 0.0
        self.max_t = 1.0
        c = 128
        cs = [c, c * 2, c * 2, c * 4, c * 4, c * 8, c * 8]
        self.mapping_timestep_embed = FourierFeatures(1, 128)
        self.mapping = nn.Sequential(ResLinearBlock(512 + 128, 1024, 1024), ResLinearBlock(1024, 1024, 1024, is_last=True))
        with torch.no_grad():
            for param in self.mapping.parameters():
                param *= 0.5 ** 0.5
        self.state = {}
        conv_block = partial(ResModConvBlock, self.state, 1024)
        self.timestep_embed = FourierFeatures(1, 16)
        self.down = nn.AvgPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.net = nn.Sequential(conv_block(3 + 16, cs[0], cs[0]), conv_block(cs[0], cs[0], cs[0]), conv_block(cs[0], cs[0], cs[0]), conv_block(cs[0], cs[0], cs[0]), SkipBlock([self.down, conv_block(cs[0], cs[1], cs[1]), conv_block(cs[1], cs[1], cs[1]), conv_block(cs[1], cs[1], cs[1]), conv_block(cs[1], cs[1], cs[1]), SkipBlock([self.down, conv_block(cs[1], cs[2], cs[2]), conv_block(cs[2], cs[2], cs[2]), conv_block(cs[2], cs[2], cs[2]), conv_block(cs[2], cs[2], cs[2]), SkipBlock([self.down, conv_block(cs[2], cs[3], cs[3]), conv_block(cs[3], cs[3], cs[3]), conv_block(cs[3], cs[3], cs[3]), conv_block(cs[3], cs[3], cs[3]), SkipBlock([self.down, conv_block(cs[3], cs[4], cs[4]), SelfAttention2d(cs[4], cs[4] // 64), conv_block(cs[4], cs[4], cs[4]), SelfAttention2d(cs[4], cs[4] // 64), conv_block(cs[4], cs[4], cs[4]), SelfAttention2d(cs[4], cs[4] // 64), conv_block(cs[4], cs[4], cs[4]), SelfAttention2d(cs[4], cs[4] // 64), SkipBlock([self.down, conv_block(cs[4], cs[5], cs[5]), SelfAttention2d(cs[5], cs[5] // 64), conv_block(cs[5], cs[5], cs[5]), SelfAttention2d(cs[5], cs[5] // 64), conv_block(cs[5], cs[5], cs[5]), SelfAttention2d(cs[5], cs[5] // 64), conv_block(cs[5], cs[5], cs[5]), SelfAttention2d(cs[5], cs[5] // 64), SkipBlock([self.down, conv_block(cs[5], cs[6], cs[6]), SelfAttention2d(cs[6], cs[6] // 64), conv_block(cs[6], cs[6], cs[6]), SelfAttention2d(cs[6], cs[6] // 64), conv_block(cs[6], cs[6], cs[6]), SelfAttention2d(cs[6], cs[6] // 64), conv_block(cs[6], cs[6], cs[6]), SelfAttention2d(cs[6], cs[6] // 64), conv_block(cs[6], cs[6], cs[6]), SelfAttention2d(cs[6], cs[6] // 64), conv_block(cs[6], cs[6], cs[6]), SelfAttention2d(cs[6], cs[6] // 64), conv_block(cs[6], cs[6], cs[6]), SelfAttention2d(cs[6], cs[6] // 64), conv_block(cs[6], cs[6], cs[5]), SelfAttention2d(cs[5], cs[5] // 64), self.up]), conv_block(cs[5] * 2, cs[5], cs[5]), SelfAttention2d(cs[5], cs[5] // 64), conv_block(cs[5], cs[5], cs[5]), SelfAttention2d(cs[5], cs[5] // 64), conv_block(cs[5], cs[5], cs[5]), SelfAttention2d(cs[5], cs[5] // 64), conv_block(cs[5], cs[5], cs[4]), SelfAttention2d(cs[4], cs[4] // 64), self.up]), conv_block(cs[4] * 2, cs[4], cs[4]), SelfAttention2d(cs[4], cs[4] // 64), conv_block(cs[4], cs[4], cs[4]), SelfAttention2d(cs[4], cs[4] // 64), conv_block(cs[4], cs[4], cs[4]), SelfAttention2d(cs[4], cs[4] // 64), conv_block(cs[4], cs[4], cs[3]), SelfAttention2d(cs[3], cs[3] // 64), self.up]), conv_block(cs[3] * 2, cs[3], cs[3]), conv_block(cs[3], cs[3], cs[3]), conv_block(cs[3], cs[3], cs[3]), conv_block(cs[3], cs[3], cs[2]), self.up]), conv_block(cs[2] * 2, cs[2], cs[2]), conv_block(cs[2], cs[2], cs[2]), conv_block(cs[2], cs[2], cs[2]), conv_block(cs[2], cs[2], cs[1]), self.up]), conv_block(cs[1] * 2, cs[1], cs[1]), conv_block(cs[1], cs[1], cs[1]), conv_block(cs[1], cs[1], cs[1]), conv_block(cs[1], cs[1], cs[0]), self.up]), conv_block(cs[0] * 2, cs[0], cs[0]), conv_block(cs[0], cs[0], cs[0]), conv_block(cs[0], cs[0], cs[0]), conv_block(cs[0], cs[0], 3, is_last=True))
        with torch.no_grad():
            for param in self.net.parameters():
                param *= 0.5 ** 0.5

    def forward(self, input, timestep_embed, selfcond):
        self.state['cond'] = selfcond
        out = self.net(torch.cat([input, timestep_embed], dim=1))
        self.state.clear()
        return out


class HuggingFaceImageClassification(torch.nn.Module):

    def __init__(self, hf_model_name):
        super().__init__()
        self.model = AutoModelForImageClassification.from_pretrained(hf_model_name, output_attentions=False, return_dict=False, torchscript=True)

    def forward(self, inputs):
        return self.model.forward(inputs)[0]


T5_MAX_SEQUENCE_LENGTH = 512


class HFSeq2SeqLanguageModel(torch.nn.Module):

    def __init__(self, model_name):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenization_kwargs = {'pad_to_multiple_of': T5_MAX_SEQUENCE_LENGTH, 'padding': True, 'return_tensors': 'pt'}
        self.model = T5Model.from_pretrained(model_name, return_dict=True)

    def preprocess_input(self, text):
        return self.tokenizer(text, **self.tokenization_kwargs)

    def forward(self, input_ids, decoder_input_ids):
        return self.model.forward(input_ids, decoder_input_ids=decoder_input_ids)[0]


class HFCausalLM(torch.nn.Module):

    def __init__(self, model_name: 'str'):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(model_name, num_labels=2, output_attentions=False, output_hidden_states=False, torchscript=True)
        self.model.eval()

    def forward(self, tokens):
        return self.model.forward(tokens)[0]


class BertHalfPrecisionModel(torch.nn.Module):

    def __init__(self, hf_model_name):
        super().__init__()
        self.model = AutoModelForMaskedLM.from_pretrained(hf_model_name, num_labels=2, output_attentions=False, output_hidden_states=False, torchscript=True, torch_dtype=torch.float16)

    def forward(self, tokens):
        return self.model.forward(tokens)[0]


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (FourierFeatures,
     lambda: ([], {'in_features': 4, 'out_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (OPTAttention,
     lambda: ([], {'embed_dim': 4, 'num_heads': 4}),
     lambda: ([torch.rand([4, 4, 4])], {})),
    (OPTLearnedPositionalEmbedding,
     lambda: ([], {'num_embeddings': 4, 'embedding_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (RRDB,
     lambda: ([], {'nf': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (RRDBNet,
     lambda: ([], {'in_nc': 4, 'out_nc': 4, 'nf': 4, 'nb': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ResLinearBlock,
     lambda: ([], {'f_in': 4, 'f_mid': 4, 'f_out': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ResidualDenseBlock_5C,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 64, 64, 64])], {})),
    (Resnet50Module,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {})),
    (SelfAttention2d,
     lambda: ([], {'c_in': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (UnetModule,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {})),
]

