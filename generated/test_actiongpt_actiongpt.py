
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


import torch.nn.functional as F


import matplotlib.pyplot as plt


import torch


import numpy as np


from torch.utils.data import DataLoader


from copy import deepcopy


import random


import math


from torch import nn


from torch import Tensor


import torch.nn as nn


from torch.nn.parameter import Parameter


from torch.nn.modules.module import Module


from torch.utils.tensorboard import SummaryWriter


import functools


from typing import Optional


import matplotlib


import logging


from re import A


from typing import Dict


from typing import List


from typing import Tuple


import pandas


from torch.utils.data import Dataset


from typing import Any


from typing import MutableMapping


from typing import Union


from torch.distributions.distribution import Distribution


from torch.nn.functional import smooth_l1_loss


from re import I


class MaskedNorm(nn.Module):
    """
        Original Code from:
        https://discuss.pytorch.org/t/batchnorm-for-different-sized-samples-in-batch/44251/8
    """

    def __init__(self, norm_type, num_groups, num_features):
        super().__init__()
        self.norm_type = norm_type
        if self.norm_type == 'batch':
            self.norm = nn.BatchNorm1d(num_features=num_features)
        elif self.norm_type == 'group':
            self.norm = nn.GroupNorm(num_groups=num_groups, num_channels=num_features)
        elif self.norm_type == 'layer':
            self.norm = nn.LayerNorm(normalized_shape=num_features)
        else:
            raise ValueError('Unsupported Normalization Layer')
        self.num_features = num_features

    def forward(self, x: 'Tensor', mask: 'Tensor'):
        if self.training:
            reshaped = x.reshape([-1, self.num_features])
            reshaped_mask = mask.reshape([-1, 1]) > 0
            selected = torch.masked_select(reshaped, reshaped_mask).reshape([-1, self.num_features])
            batch_normed = self.norm(selected)
            scattered = reshaped.masked_scatter(reshaped_mask, batch_normed)
            return scattered.reshape([x.shape[0], -1, self.num_features])
        else:
            reshaped = x.reshape([-1, self.num_features])
            batched_normed = self.norm(reshaped)
            return batched_normed.reshape([x.shape[0], -1, self.num_features])


def freeze_params(module: 'nn.Module') ->None:
    """
    Freeze the parameters of this module,
    i.e. do not update them during training

    :param module: freeze parameters of this module
    """
    for _, p in module.named_parameters():
        p.requires_grad = False


def get_activation(activation_type):
    if activation_type == 'relu':
        return nn.ReLU()
    elif activation_type == 'relu6':
        return nn.ReLU6()
    elif activation_type == 'prelu':
        return nn.PReLU()
    elif activation_type == 'selu':
        return nn.SELU()
    elif activation_type == 'celu':
        return nn.CELU()
    elif activation_type == 'gelu':
        return nn.GELU()
    elif activation_type == 'sigmoid':
        return nn.Sigmoid()
    elif activation_type == 'softplus':
        return nn.Softplus()
    elif activation_type == 'softshrink':
        return nn.Softshrink()
    elif activation_type == 'softsign':
        return nn.Softsign()
    elif activation_type == 'tanh':
        return nn.Tanh()
    elif activation_type == 'tanhshrink':
        return nn.Tanhshrink()
    else:
        raise ValueError('Unknown activation type {}'.format(activation_type))


class Embeddings(nn.Module):
    """
    Simple embeddings class
    """

    def __init__(self, embedding_dim: 'int'=64, num_heads: 'int'=8, scale: 'bool'=False, scale_factor: 'float'=None, norm_type: 'str'=None, activation_type: 'str'=None, vocab_size: 'int'=0, padding_idx: 'int'=1, freeze: 'bool'=False, **kwargs):
        """
        Create new embeddings for the vocabulary.
        Use scaling for the Transformer.

        :param embedding_dim:
        :param scale:
        :param vocab_size:
        :param padding_idx:
        :param freeze: freeze the embeddings during training
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.lut = nn.Embedding(vocab_size, self.embedding_dim, padding_idx=padding_idx)
        self.norm_type = norm_type
        if self.norm_type:
            self.norm = MaskedNorm(norm_type=norm_type, num_groups=num_heads, num_features=embedding_dim)
        self.activation_type = activation_type
        if self.activation_type:
            self.activation = get_activation(activation_type)
        self.scale = scale
        if self.scale:
            if scale_factor:
                self.scale_factor = scale_factor
            else:
                self.scale_factor = math.sqrt(self.embedding_dim)
        if freeze:
            freeze_params(self)

    def forward(self, x: 'Tensor', mask: 'Tensor'=None) ->Tensor:
        """
        Perform lookup for input `x` in the embedding table.

        :param mask: token masks
        :param x: index in the vocabulary
        :return: embedded representation for `x`
        """
        x = self.lut(x)
        if self.norm_type:
            x = self.norm(x, mask)
        if self.activation_type:
            x = self.activation(x)
        if self.scale:
            return x * self.scale_factor
        else:
            return x

    def __repr__(self):
        return '%s(embedding_dim=%d, vocab_size=%d)' % (self.__class__.__name__, self.embedding_dim, self.vocab_size)


class SpatialEmbeddings(nn.Module):
    """
    Simple Linear Projection Layer
    (For encoder outputs to predict glosses)
    """

    def __init__(self, embedding_dim: 'int', input_size: 'int', num_heads: 'int', freeze: 'bool'=False, norm_type: 'str'='batch', activation_type: 'str'='softsign', scale: 'bool'=False, scale_factor: 'float'=None, **kwargs):
        """
        Create new embeddings for the vocabulary.
        Use scaling for the Transformer.

        :param embedding_dim:
        :param input_size:
        :param freeze: freeze the embeddings during training
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.input_size = input_size
        self.ln = nn.Linear(self.input_size, self.embedding_dim)
        self.norm_type = norm_type
        if self.norm_type:
            self.norm = MaskedNorm(norm_type=norm_type, num_groups=num_heads, num_features=embedding_dim)
        self.activation_type = activation_type
        if self.activation_type:
            self.activation = get_activation(activation_type)
        self.scale = scale
        if self.scale:
            if scale_factor:
                self.scale_factor = scale_factor
            else:
                self.scale_factor = math.sqrt(self.embedding_dim)
        if freeze:
            freeze_params(self)

    def forward(self, x: 'Tensor', mask: 'Tensor') ->Tensor:
        """
        :param mask: frame masks
        :param x: input frame features
        :return: embedded representation for `x`
        """
        x = self.ln(x)
        if self.norm_type:
            x = self.norm(x, mask)
        if self.activation_type:
            x = self.activation(x)
        if self.scale:
            return x * self.scale_factor
        else:
            return x

    def __repr__(self):
        return '%s(embedding_dim=%d, input_size=%d)' % (self.__class__.__name__, self.embedding_dim, self.input_size)


class MultiHeadedAttention(nn.Module):
    """
    Multi-Head Attention module from "Attention is All You Need"

    Implementation modified from OpenNMT-py.
    https://github.com/OpenNMT/OpenNMT-py
    """

    def __init__(self, num_heads: 'int', size: 'int', dropout: 'float'=0.1):
        """
        Create a multi-headed attention layer.
        :param num_heads: the number of heads
        :param size: model size (must be divisible by num_heads)
        :param dropout: probability of dropping a unit
        """
        super().__init__()
        assert size % num_heads == 0
        self.head_size = head_size = size // num_heads
        self.model_size = size
        self.num_heads = num_heads
        self.k_layer = nn.Linear(size, num_heads * head_size)
        self.v_layer = nn.Linear(size, num_heads * head_size)
        self.q_layer = nn.Linear(size, num_heads * head_size)
        self.output_layer = nn.Linear(size, size)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, k: 'Tensor', v: 'Tensor', q: 'Tensor', mask: 'Tensor'=None):
        """
        Computes multi-headed attention.

        :param k: keys   [B, M, D] with M being the sentence length.
        :param v: values [B, M, D]
        :param q: query  [B, M, D]
        :param mask: optional mask [B, 1, M] or [B, M, M]
        :return:
        """
        batch_size = k.size(0)
        num_heads = self.num_heads
        k = self.k_layer(k)
        v = self.v_layer(v)
        q = self.q_layer(q)
        k = k.view(batch_size, -1, num_heads, self.head_size).transpose(1, 2)
        v = v.view(batch_size, -1, num_heads, self.head_size).transpose(1, 2)
        q = q.view(batch_size, -1, num_heads, self.head_size).transpose(1, 2)
        q = q / math.sqrt(self.head_size)
        scores = torch.matmul(q, k.transpose(2, 3))
        if mask is not None:
            scores = scores.masked_fill(~mask.unsqueeze(1), float('-inf'))
        attention = self.softmax(scores)
        attention = self.dropout(attention)
        context = torch.matmul(attention, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, num_heads * self.head_size)
        output = self.output_layer(context)
        return output


class PositionwiseFeedForward(nn.Module):
    """
    Position-wise Feed-forward layer
    Projects to ff_size and then back down to input_size.
    """

    def __init__(self, input_size, ff_size, dropout=0.1):
        """
        Initializes position-wise feed-forward layer.
        :param input_size: dimensionality of the input.
        :param ff_size: dimensionality of intermediate representation
        :param dropout:
        """
        super().__init__()
        self.layer_norm = nn.LayerNorm(input_size, eps=1e-06)
        self.pwff_layer = nn.Sequential(nn.Linear(input_size, ff_size), nn.ReLU(), nn.Dropout(dropout), nn.Linear(ff_size, input_size), nn.Dropout(dropout))

    def forward(self, x):
        x_norm = self.layer_norm(x)
        return self.pwff_layer(x_norm) + x


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000, batch_first=False, negative=False):
        super().__init__()
        self.batch_first = batch_first
        self.dropout = nn.Dropout(p=dropout)
        self.max_len = max_len
        self.negative = negative
        if negative:
            pe = torch.zeros(2 * max_len, d_model)
            position = torch.arange(-max_len, max_len, dtype=torch.float).unsqueeze(1)
        else:
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x, hist_frames=0):
        if not self.negative:
            center = 0
            assert hist_frames == 0
            first = 0
        else:
            center = self.max_len
            first = center - hist_frames
        if self.batch_first:
            last = first + x.shape[1]
            x = x + self.pe.permute(1, 0, 2)[:, first:last, :]
        else:
            last = first + x.shape[0]
            x = x + self.pe[first:last, :]
        return self.dropout(x)


class TransformerEncoderLayer(nn.Module):
    """
    One Transformer encoder layer has a Multi-head attention layer plus
    a position-wise feed-forward layer.
    """

    def __init__(self, size: 'int'=0, ff_size: 'int'=0, num_heads: 'int'=0, dropout: 'float'=0.1):
        """
        A single Transformer layer.
        :param size:
        :param ff_size:
        :param num_heads:
        :param dropout:
        """
        super().__init__()
        self.layer_norm = nn.LayerNorm(size, eps=1e-06)
        self.src_src_att = MultiHeadedAttention(num_heads, size, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(size, ff_size=ff_size, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.size = size

    def forward(self, x: 'Tensor', mask: 'Tensor') ->Tensor:
        """
        Forward pass for a single transformer encoder layer.
        First applies layer norm, then self attention,
        then dropout with residual connection (adding the input to the result),
        and then a position-wise feed-forward layer.

        :param x: layer input
        :param mask: input mask
        :return: output tensor
        """
        x_norm = self.layer_norm(x)
        h = self.src_src_att(x_norm, x_norm, x_norm, mask)
        h = self.dropout(h) + x
        o = self.feed_forward(h)
        return o


class TransformerDecoderLayer(nn.Module):
    """
    Transformer decoder layer.

    Consists of self-attention, source-attention, and feed-forward.
    """

    def __init__(self, size: 'int'=0, ff_size: 'int'=0, num_heads: 'int'=0, dropout: 'float'=0.1):
        """
        Represents a single Transformer decoder layer.

        It attends to the source representation and the previous decoder states.

        :param size: model dimensionality
        :param ff_size: size of the feed-forward intermediate layer
        :param num_heads: number of heads
        :param dropout: dropout to apply to input
        """
        super().__init__()
        self.size = size
        self.trg_trg_att = MultiHeadedAttention(num_heads, size, dropout=dropout)
        self.src_trg_att = MultiHeadedAttention(num_heads, size, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(size, ff_size=ff_size, dropout=dropout)
        self.x_layer_norm = nn.LayerNorm(size, eps=1e-06)
        self.dec_layer_norm = nn.LayerNorm(size, eps=1e-06)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: 'Tensor'=None, memory: 'Tensor'=None, src_mask: 'Tensor'=None, trg_mask: 'Tensor'=None) ->Tensor:
        """
        Forward pass of a single Transformer decoder layer.

        :param x: inputs
        :param memory: source representations
        :param src_mask: source mask
        :param trg_mask: target mask (so as to not condition on future steps)
        :return: output tensor
        """
        x_norm = self.x_layer_norm(x)
        h1 = self.trg_trg_att(x_norm, x_norm, x_norm, mask=trg_mask)
        h1 = self.dropout(h1) + x
        h1_norm = self.dec_layer_norm(h1)
        h2 = self.src_trg_att(memory, memory, h1_norm, mask=src_mask)
        o = self.feed_forward(self.dropout(h2) + h1)
        return o


class TimeEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(TimeEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask, lengths):
        time = mask * 1 / (lengths[..., None] - 1)
        time = time[:, None] * torch.arange(time.shape[1], device=x.device)[None, :]
        time = time[:, 0].T
        x = x + time[..., None]
        return self.dropout(x)


class Encoder_TRANSFORMER(nn.Module):

    def __init__(self, modeltype, njoints, nfeats, num_frames, num_classes, translation, pose_rep, glob, glob_rot, latent_dim=256, ff_size=1024, num_layers=4, num_heads=4, dropout=0.1, ablation=None, activation='gelu', **kargs):
        super().__init__()
        self.modeltype = modeltype
        self.njoints = njoints
        self.nfeats = nfeats
        self.num_frames = num_frames
        self.num_classes = num_classes
        self.pose_rep = pose_rep
        self.glob = glob
        self.glob_rot = glob_rot
        self.translation = translation
        self.latent_dim = latent_dim
        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.ablation = ablation
        self.activation = activation
        self.input_feats = self.njoints * self.nfeats
        self.normalize_output = kargs.get('normalize_encoder_output', False)
        if self.ablation == 'average_encoder':
            self.mu_layer = nn.Linear(self.latent_dim, self.latent_dim)
            self.sigma_layer = nn.Linear(self.latent_dim, self.latent_dim)
        else:
            self.muQuery = nn.Parameter(torch.randn(self.num_classes, self.latent_dim))
        self.skelEmbedding = nn.Linear(self.input_feats, self.latent_dim)
        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)
        seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim, nhead=self.num_heads, dim_feedforward=self.ff_size, dropout=self.dropout, activation=self.activation)
        self.seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer, num_layers=self.num_layers)

    def forward(self, batch):
        x, y, mask = batch['x'], batch['y'], batch['mask']
        bs, njoints, nfeats, nframes = x.shape
        x = x.permute((3, 0, 1, 2)).reshape(nframes, bs, njoints * nfeats)
        x = self.skelEmbedding(x)
        xseq = torch.cat((self.muQuery[y][None], x), axis=0)
        xseq = self.sequence_pos_encoder(xseq)
        muandsigmaMask = torch.ones((bs, 1), dtype=bool, device=x.device)
        maskseq = torch.cat((muandsigmaMask, mask), axis=1)
        final = self.seqTransEncoder(xseq, src_key_padding_mask=~maskseq)
        mu = final[0]
        if self.normalize_output:
            mu = mu / mu.norm(dim=-1, keepdim=True)
        return {'mu': mu}


class Decoder_TRANSFORMER(nn.Module):

    def __init__(self, modeltype, njoints, nfeats, num_frames, num_classes, translation, pose_rep, glob, glob_rot, latent_dim=256, ff_size=1024, num_layers=4, num_heads=4, dropout=0.1, activation='gelu', ablation=None, **kargs):
        super().__init__()
        self.modeltype = modeltype
        self.njoints = njoints
        self.nfeats = nfeats
        self.num_frames = num_frames
        self.num_classes = num_classes
        self.pose_rep = pose_rep
        self.glob = glob
        self.glob_rot = glob_rot
        self.translation = translation
        self.latent_dim = latent_dim
        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.ablation = ablation
        self.activation = activation
        self.input_feats = self.njoints * self.nfeats
        self.normalize_decoder_input = kargs.get('normalize_decoder_input', False)
        if self.ablation == 'zandtime':
            self.ztimelinear = nn.Linear(self.latent_dim + self.num_classes, self.latent_dim)
        else:
            self.actionBiases = nn.Parameter(torch.randn(self.num_classes, self.latent_dim))
        if self.ablation == 'time_encoding':
            self.sequence_pos_encoder = TimeEncoding(self.dropout)
        else:
            self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)
        seqTransDecoderLayer = nn.TransformerDecoderLayer(d_model=self.latent_dim, nhead=self.num_heads, dim_feedforward=self.ff_size, dropout=self.dropout, activation=activation)
        self.seqTransDecoder = nn.TransformerDecoder(seqTransDecoderLayer, num_layers=self.num_layers)
        self.finallayer = nn.Linear(self.latent_dim, self.input_feats)

    def forward(self, batch, use_text_emb=False):
        z, mask, lengths = batch['z'], batch['mask'], batch['lengths']
        if use_text_emb:
            z = batch['clip_text_emb']
        latent_dim = z.shape[1]
        bs, nframes = mask.shape
        njoints, nfeats = self.njoints, self.nfeats
        if self.ablation == 'zandtime':
            yoh = F.one_hot(y, self.num_classes)
            z = torch.cat((z, yoh), axis=1)
            z = self.ztimelinear(z)
            z = z[None]
        elif self.ablation == 'concat_bias':
            z = torch.stack((z, self.actionBiases[y]), axis=0)
        else:
            z = z[None]
        timequeries = torch.zeros(nframes, bs, latent_dim, device=z.device)
        if self.ablation == 'time_encoding':
            timequeries = self.sequence_pos_encoder(timequeries, mask, lengths)
        else:
            timequeries = self.sequence_pos_encoder(timequeries)
        if self.normalize_decoder_input:
            z = z / torch.norm(z, dim=-1, keepdim=True)
        output = self.seqTransDecoder(tgt=timequeries, memory=z, tgt_key_padding_mask=~mask)
        output = self.finallayer(output).reshape(nframes, bs, njoints, nfeats)
        output[~mask.T] = 0
        output = output.permute(1, 2, 3, 0)
        if use_text_emb:
            batch['txt_output'] = output
        else:
            batch['output'] = output
        return batch


JOINTSTYPES = ['a2m', 'a2mpl', 'smpl', 'vibe', 'vertices']


JOINTSTYPE_ROOT = {'a2m': 0, 'smpl': 0, 'a2mpl': 0, 'vibe': 8}


JOINT_MAP = {'OP Nose': 24, 'OP Neck': 12, 'OP RShoulder': 17, 'OP RElbow': 19, 'OP RWrist': 21, 'OP LShoulder': 16, 'OP LElbow': 18, 'OP LWrist': 20, 'OP MidHip': 0, 'OP RHip': 2, 'OP RKnee': 5, 'OP RAnkle': 8, 'OP LHip': 1, 'OP LKnee': 4, 'OP LAnkle': 7, 'OP REye': 25, 'OP LEye': 26, 'OP REar': 27, 'OP LEar': 28, 'OP LBigToe': 29, 'OP LSmallToe': 30, 'OP LHeel': 31, 'OP RBigToe': 32, 'OP RSmallToe': 33, 'OP RHeel': 34, 'Right Ankle': 8, 'Right Knee': 5, 'Right Hip': 45, 'Left Hip': 46, 'Left Knee': 4, 'Left Ankle': 7, 'Right Wrist': 21, 'Right Elbow': 19, 'Right Shoulder': 17, 'Left Shoulder': 16, 'Left Elbow': 18, 'Left Wrist': 20, 'Neck (LSP)': 47, 'Top of Head (LSP)': 48, 'Pelvis (MPII)': 49, 'Thorax (MPII)': 50, 'Spine (H36M)': 51, 'Jaw (H36M)': 52, 'Head (H36M)': 53, 'Nose': 24, 'Left Eye': 26, 'Right Eye': 25, 'Left Ear': 28, 'Right Ear': 27}


JOINT_NAMES = ['OP Nose', 'OP Neck', 'OP RShoulder', 'OP RElbow', 'OP RWrist', 'OP LShoulder', 'OP LElbow', 'OP LWrist', 'OP MidHip', 'OP RHip', 'OP RKnee', 'OP RAnkle', 'OP LHip', 'OP LKnee', 'OP LAnkle', 'OP REye', 'OP LEye', 'OP REar', 'OP LEar', 'OP LBigToe', 'OP LSmallToe', 'OP LHeel', 'OP RBigToe', 'OP RSmallToe', 'OP RHeel', 'Right Ankle', 'Right Knee', 'Right Hip', 'Left Hip', 'Left Knee', 'Left Ankle', 'Right Wrist', 'Right Elbow', 'Right Shoulder', 'Left Shoulder', 'Left Elbow', 'Left Wrist', 'Neck (LSP)', 'Top of Head (LSP)', 'Pelvis (MPII)', 'Thorax (MPII)', 'Spine (H36M)', 'Jaw (H36M)', 'Head (H36M)', 'Nose', 'Left Eye', 'Right Eye', 'Left Ear', 'Right Ear']


SMPL_DATA_PATH = './models/smpl'


action2motion_joints = [8, 1, 2, 3, 4, 5, 6, 7, 0, 9, 10, 11, 12, 13, 14, 21, 24, 38]


class Rotation2xyz:

    def __init__(self, device):
        self.device = device
        self.smpl_model = SMPL().eval()

    def __call__(self, x, mask, pose_rep, translation, glob, jointstype, vertstrans, betas=None, beta=0, glob_rot=None, get_rotations_back=False, **kwargs):
        if pose_rep == 'xyz':
            return x
        if mask is None:
            mask = torch.ones((x.shape[0], x.shape[-1]), dtype=bool, device=x.device)
        if not glob and glob_rot is None:
            raise TypeError('You must specify global rotation if glob is False')
        if jointstype not in JOINTSTYPES:
            raise NotImplementedError('This jointstype is not implemented.')
        if translation:
            x_translations = x[:, -1, :3]
            x_rotations = x[:, :-1]
        else:
            x_rotations = x
        x_rotations = x_rotations.permute(0, 3, 1, 2)
        nsamples, time, njoints, feats = x_rotations.shape
        if pose_rep == 'rotvec':
            rotations = geometry.axis_angle_to_matrix(x_rotations[mask])
        elif pose_rep == 'rotmat':
            rotations = x_rotations[mask].view(-1, njoints, 3, 3)
        elif pose_rep == 'rotquat':
            rotations = geometry.quaternion_to_matrix(x_rotations[mask])
        elif pose_rep == 'rot6d':
            rotations = geometry.rotation_6d_to_matrix(x_rotations[mask])
        else:
            raise NotImplementedError('No geometry for this one.')
        if not glob:
            global_orient = torch.tensor(glob_rot, device=x.device)
            global_orient = geometry.axis_angle_to_matrix(global_orient).view(1, 1, 3, 3)
            global_orient = global_orient.repeat(len(rotations), 1, 1, 1)
        else:
            global_orient = rotations[:, 0]
            rotations = rotations[:, 1:]
        if betas is None:
            betas = torch.zeros([rotations.shape[0], self.smpl_model.num_betas], dtype=rotations.dtype, device=rotations.device)
            betas[:, 1] = beta
        out = self.smpl_model(body_pose=rotations, global_orient=global_orient, betas=betas)
        joints = out[jointstype]
        x_xyz = torch.empty(nsamples, time, joints.shape[1], 3, device=x.device, dtype=x.dtype)
        x_xyz[~mask] = 0
        x_xyz[mask] = joints
        x_xyz = x_xyz.permute(0, 2, 3, 1).contiguous()
        if jointstype != 'vertices':
            rootindex = JOINTSTYPE_ROOT[jointstype]
            x_xyz = x_xyz - x_xyz[:, [rootindex], :, :]
        if translation and vertstrans:
            x_translations = x_translations - x_translations[:, :, [0]]
            x_xyz = x_xyz + x_translations[:, None, :, :]
        if get_rotations_back:
            return x_xyz, rotations, global_orient
        else:
            return x_xyz


cosine_sim = nn.CosineSimilarity(dim=1, eps=1e-06)


def listify(x):
    """If x is already a list, do nothing. Otherwise, wrap x in a list."""
    if isinstance(x, list):
        return x
    else:
        return [x]


def multi_layer_second_directional_derivative(G, batch, dz, G_z, epsilon, **G_kwargs):
    """Estimates the second directional derivative of G w.r.t. its input at z in the direction x"""
    batch_plus = {**batch, 'x': batch['x'] + dz}
    batch_moins = {**batch, 'x': batch['x'] - dz}
    G_to_x = G(batch_plus, **G_kwargs)
    G_from_x = G(batch_moins, **G_kwargs)
    G_to_x = listify(G_to_x)
    G_from_x = listify(G_from_x)
    G_z = listify(G_z)
    eps_sqr = epsilon ** 2
    sdd = [((G2x - 2 * G_z_base + Gfx) / eps_sqr) for G2x, G_z_base, Gfx in zip(G_to_x, G_z, G_from_x)]
    return sdd


def stack_var_and_reduce(list_of_activations, reduction=torch.max):
    """Equation (5) from the paper."""
    second_orders = torch.stack(list_of_activations)
    var_tensor = torch.var(second_orders, dim=0, unbiased=True)
    penalty = reduction(var_tensor)
    return penalty


def multi_stack_var_and_reduce(sdds, reduction=torch.max, return_separately=False):
    """Iterate over all activations to be regularized, then apply Equation (5) to each."""
    sum_of_penalties = 0 if not return_separately else []
    for activ_n in zip(*sdds):
        penalty = stack_var_and_reduce(activ_n, reduction)
        sum_of_penalties += penalty if not return_separately else [penalty]
    return sum_of_penalties


def rademacher(shape, device='cpu'):
    """Creates a random tensor of size [shape] under the Rademacher distribution (P(x=1) == P(x=-1) == 0.5)"""
    x = torch.empty(shape, device=device)
    x.random_(0, 2)
    x[x == 0] = -1
    return x


def hessian_penalty(G, batch, k=2, epsilon=0.1, reduction=torch.max, return_separately=False, G_z=None, **G_kwargs):
    """
    Official PyTorch Hessian Penalty implementation.

    Note: If you want to regularize multiple network activations simultaneously, you need to
    make sure the function G you pass to hessian_penalty returns a list of those activations when it's called with
    G(z, **G_kwargs). Otherwise, if G returns a tensor the Hessian Penalty will only be computed for the final
    output of G.

    :param G: Function that maps input z to either a tensor or a list of tensors (activations)
    :param z: Input to G that the Hessian Penalty will be computed with respect to
    :param k: Number of Hessian directions to sample (must be >= 2)
    :param epsilon: Amount to blur G before estimating Hessian (must be > 0)
    :param reduction: Many-to-one function to reduce each pixel/neuron's individual hessian penalty into a final loss
    :param return_separately: If False, hessian penalties for each activation output by G are automatically summed into
                              a final loss. If True, the hessian penalties for each layer will be returned in a list
                              instead. If G outputs a single tensor, setting this to True will produce a length-1
                              list.
    :param G_z: [Optional small speed-up] If you have already computed G(z, **G_kwargs) for the current training
                iteration, then you can provide it here to reduce the number of forward passes of this method by 1
    :param G_kwargs: Additional inputs to G besides the z vector. For example, in BigGAN you
                     would pass the class label into this function via y=<class_label_tensor>

    :return: A differentiable scalar (the hessian penalty), or a list of hessian penalties if return_separately is True
    """
    if G_z is None:
        G_z = G(batch, **G_kwargs)
    z = batch['x']
    rademacher_size = torch.Size((k, *z.size()))
    dzs = epsilon * rademacher(rademacher_size, device=z.device)
    second_orders = []
    for dz in dzs:
        central_second_order = multi_layer_second_directional_derivative(G, batch, dz, G_z, epsilon, **G_kwargs)
        second_orders.append(central_second_order)
    loss = multi_stack_var_and_reduce(second_orders, reduction, return_separately)
    return loss


def compute_hp_loss(model, batch):
    loss = hessian_penalty(model.return_latent, batch, seed=torch.random.seed())
    return loss


def compute_kernel(x, y):
    x_size = x.size(0)
    y_size = y.size(0)
    dim = x.size(1)
    x = x.unsqueeze(1)
    y = y.unsqueeze(0)
    tiled_x = x.expand(x_size, y_size, dim)
    tiled_y = y.expand(x_size, y_size, dim)
    kernel_input = (tiled_x - tiled_y).pow(2).mean(2) / float(dim)
    return torch.exp(-kernel_input)


def compute_mmd(x, y):
    x_kernel = compute_kernel(x, x)
    y_kernel = compute_kernel(y, y)
    xy_kernel = compute_kernel(x, y)
    mmd = x_kernel.mean() + y_kernel.mean() - 2 * xy_kernel.mean()
    return mmd


def compute_mmd_loss(model, batch):
    z = batch['z']
    true_samples = torch.randn(z.shape, requires_grad=False, device=model.device)
    loss = compute_mmd(true_samples, z)
    return loss


def compute_rc_loss(model, batch, use_txt_output=False):
    x = batch['x']
    output = batch['output']
    mask = batch['mask']
    if use_txt_output:
        output = batch['txt_output']
    gtmasked = x.permute(0, 3, 1, 2)[mask]
    outmasked = output.permute(0, 3, 1, 2)[mask]
    loss = F.mse_loss(gtmasked, outmasked, reduction='mean')
    return loss


def compute_rcxyz_loss(model, batch, use_txt_output=False):
    x = batch['x_xyz']
    output = batch['output_xyz']
    mask = batch['mask']
    if use_txt_output:
        output = batch['txt_output_xyz']
    gtmasked = x.permute(0, 3, 1, 2)[mask]
    outmasked = output.permute(0, 3, 1, 2)[mask]
    loss = F.mse_loss(gtmasked, outmasked, reduction='mean')
    return loss


def compute_vel_loss(model, batch, use_txt_output=False):
    x = batch['x']
    output = batch['output']
    if use_txt_output:
        output = batch['txt_output']
    gtvel = x[..., 1:] - x[..., :-1]
    outputvel = output[..., 1:] - output[..., :-1]
    mask = batch['mask'][..., 1:]
    gtvelmasked = gtvel.permute(0, 3, 1, 2)[mask]
    outvelmasked = outputvel.permute(0, 3, 1, 2)[mask]
    loss = F.mse_loss(gtvelmasked, outvelmasked, reduction='mean')
    return loss


def compute_velxyz_loss(model, batch, use_txt_output=False):
    x = batch['x_xyz']
    output = batch['output_xyz']
    if use_txt_output:
        output = batch['txt_output_xyz']
    gtvel = x[..., 1:] - x[..., :-1]
    outputvel = output[..., 1:] - output[..., :-1]
    mask = batch['mask'][..., 1:]
    gtvelmasked = gtvel.permute(0, 3, 1, 2)[mask]
    outvelmasked = outputvel.permute(0, 3, 1, 2)[mask]
    loss = F.mse_loss(gtvelmasked, outvelmasked, reduction='mean')
    return loss


_matching_ = {'rc': compute_rc_loss, 'hp': compute_hp_loss, 'mmd': compute_mmd_loss, 'rcxyz': compute_rcxyz_loss, 'vel': compute_vel_loss, 'velxyz': compute_velxyz_loss}


def get_loss_function(ltype):
    return _matching_[ltype]


loss_ce = nn.CrossEntropyLoss()


loss_mse = nn.MSELoss()


class MOTIONCLIP(nn.Module):

    def __init__(self, encoder, decoder, device, lambdas, latent_dim, outputxyz, pose_rep, glob, glob_rot, translation, jointstype, vertstrans, clip_lambdas={}, **kwargs):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.outputxyz = outputxyz
        self.lambdas = lambdas
        self.clip_lambdas = clip_lambdas
        self.latent_dim = latent_dim
        self.pose_rep = pose_rep
        self.glob = glob
        self.glob_rot = glob_rot
        self.device = device
        self.translation = translation
        self.jointstype = jointstype
        self.vertstrans = vertstrans
        self.clip_model = kwargs['clip_model']
        assert self.clip_model.training == False
        self.align_pose_frontview = kwargs.get('align_pose_frontview', False)
        self.use_generation_losses = kwargs.get('use_generation_losses', False)
        self.losses = list(self.lambdas) + ['mixed']
        self.rotation2xyz = Rotation2xyz(device=self.device)
        self.param2xyz = {'pose_rep': self.pose_rep, 'glob_rot': self.glob_rot, 'glob': self.glob, 'jointstype': self.jointstype, 'translation': self.translation, 'vertstrans': self.vertstrans}

    def rot2xyz(self, x, mask, get_rotations_back=False, **kwargs):
        kargs = self.param2xyz.copy()
        kargs.update(kwargs)
        return self.rotation2xyz(x, mask, get_rotations_back=get_rotations_back, **kargs)

    def compute_loss(self, batch):
        mixed_loss = 0.0
        losses = {}
        for ltype, lam in self.lambdas.items():
            loss_function = get_loss_function(ltype)
            loss = loss_function(self, batch)
            mixed_loss += loss * lam
            losses[ltype] = loss.item()
        mixed_clip_loss, clip_losses = self.compute_clip_losses(batch)
        mixed_loss_with_clip = mixed_loss + mixed_clip_loss
        losses.update(clip_losses)
        losses['mixed_without_clip'] = mixed_loss.item()
        losses['mixed_with_clip'] = mixed_loss_with_clip.item()
        if not isinstance(mixed_clip_loss, float):
            losses['mixed_clip_only'] = mixed_clip_loss.item()
        else:
            losses['mixed_clip_only'] = mixed_clip_loss
        if self.use_generation_losses:
            batch.update(self.decoder(batch, use_text_emb=True))
            if self.outputxyz:
                batch['txt_output_xyz'] = self.rot2xyz(batch['txt_output'], batch['mask'])
            elif self.pose_rep == 'xyz':
                batch['txt_output_xyz'] = batch['output']
            gen_mixed_loss = 0.0
            for ltype, lam in self.lambdas.items():
                loss_function = get_loss_function(ltype)
                loss = loss_function(self, batch, use_txt_output=True)
                gen_mixed_loss += loss * lam
                losses[f'gen_{ltype}'] = loss.item()
            mixed_loss_with_clip += gen_mixed_loss
        return mixed_loss_with_clip, losses

    def compute_clip_losses(self, batch):
        mixed_clip_loss = 0.0
        clip_losses = {}
        for d in self.clip_lambdas.keys():
            if len(self.clip_lambdas[d].keys()) == 0:
                continue
            with torch.no_grad():
                if d == 'image':
                    if 'clip_images_emb' in batch:
                        d_features = batch['clip_images_emb'].float()
                    else:
                        d_features = self.clip_model.encode_image(batch['clip_images']).float()
                elif d == 'text':
                    texts = clip.tokenize(batch['clip_text'], truncate=True)
                    temp_d_features = self.clip_model.encode_text(texts).float()
                    d_features = []
                    for idx in range(len(temp_d_features) // 4):
                        temp_here = (temp_d_features[4 * idx] + temp_d_features[4 * idx + 1] + temp_d_features[4 * idx + 2] + temp_d_features[4 * idx + 3]) / 4
                        if len(d_features) == 0:
                            d_features = temp_here.clone().detach()[None]
                        else:
                            d_features = torch.cat((d_features, temp_here[None]), 0)
                    batch['clip_text_emb'] = d_features
                else:
                    raise ValueError(f'Invalid clip domain [{d}]')
            motion_features = batch['z']
            d_features_norm = d_features / d_features.norm(dim=-1, keepdim=True)
            motion_features_norm = motion_features / motion_features.norm(dim=-1, keepdim=True)
            if 'ce' in self.clip_lambdas[d].keys():
                logit_scale = self.clip_model.logit_scale.exp()
                logits_per_motion = logit_scale * motion_features_norm @ d_features_norm.t()
                logits_per_d = logits_per_motion.t()
                batch_size = batch['x'].shape[0]
                ground_truth = torch.arange(batch_size, dtype=torch.long, device=self.device)
                ce_from_motion_loss = loss_ce(logits_per_motion, ground_truth)
                ce_from_d_loss = loss_ce(logits_per_d, ground_truth)
                clip_mixed_loss = (ce_from_motion_loss + ce_from_d_loss) / 2.0
                clip_losses[f'{d}_ce_from_d'] = ce_from_d_loss.item()
                clip_losses[f'{d}_ce_from_motion'] = ce_from_motion_loss.item()
                clip_losses[f'{d}_mixed_ce'] = clip_mixed_loss.item()
                mixed_clip_loss += clip_mixed_loss * self.clip_lambdas[d]['ce']
            if 'mse' in self.clip_lambdas[d].keys():
                mse_clip_loss = loss_mse(d_features, motion_features)
                clip_losses[f'{d}_mse'] = mse_clip_loss.item()
                mixed_clip_loss += mse_clip_loss * self.clip_lambdas[d]['mse']
            if 'cosine' in self.clip_lambdas[d].keys():
                cos = cosine_sim(d_features_norm, motion_features_norm)
                cosine_loss = (1 - cos).mean()
                clip_losses[f'{d}_cosine'] = cosine_loss.item()
                mixed_clip_loss += cosine_loss * self.clip_lambdas[d]['cosine']
        return mixed_clip_loss, clip_losses

    @staticmethod
    def lengths_to_mask(lengths):
        max_len = max(lengths)
        if isinstance(max_len, torch.Tensor):
            max_len = max_len.item()
        index = torch.arange(max_len, device=lengths.device).expand(len(lengths), max_len)
        mask = index < lengths.unsqueeze(1)
        return mask

    def generate(self, classes, durations, nspa=1, is_amass=False, is_clip_features=False, textual_labels=None):
        clip_dim = self.clip_model.ln_final.normalized_shape[0]
        if is_clip_features:
            assert classes.shape[-1] == clip_dim
            clip_features = classes.reshape([-1, clip_dim])
            nspa, nats = classes.shape[:2]
            y = clip_features
            if textual_labels is not None:
                y = np.array(textual_labels).reshape([-1])
        if len(durations.shape) == 1:
            lengths = durations.repeat(nspa)
        else:
            lengths = durations.reshape(clip_features.shape[0])
        mask = self.lengths_to_mask(lengths)
        batch = {'z': clip_features, 'y': y, 'mask': mask, 'lengths': lengths}
        if not is_clip_features:
            batch['y'] = y
        batch = self.decoder(batch)
        if is_amass and not self.align_pose_frontview:
            None
            batch['output'][:, 0] = torch.tensor([1, 0, 0, 0, 1, 0]).unsqueeze(0).unsqueeze(2)
        if self.outputxyz:
            batch['output_xyz'] = self.rot2xyz(batch['output'], batch['mask'])
        elif self.pose_rep == 'xyz':
            batch['output_xyz'] = batch['output']
        return batch

    def forward(self, batch):
        if self.outputxyz:
            batch['x_xyz'] = self.rot2xyz(batch['x'], batch['mask'])
        elif self.pose_rep == 'xyz':
            batch['x_xyz'] = batch['x']
        batch.update(self.encoder(batch))
        batch['z'] = batch['mu']
        batch.update(self.decoder(batch))
        if self.outputxyz:
            batch['output_xyz'] = self.rot2xyz(batch['output'], batch['mask'])
        elif self.pose_rep == 'xyz':
            batch['output_xyz'] = batch['output']
        return batch


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class AutoParams(nn.Module):

    def __init__(self, **kargs):
        try:
            for param in self.needed_params:
                if param in kargs:
                    setattr(self, param, kargs[param])
                else:
                    raise ValueError(f'{param} is needed.')
        except ModuleAttributeError:
            pass
        try:
            for param, default in self.optional_params.items():
                if param in kargs and kargs[param] is not None:
                    setattr(self, param, kargs[param])
                else:
                    setattr(self, param, default)
        except ModuleAttributeError:
            pass
        super().__init__()


class Joints2Jfeats(nn.Module):

    def __init__(self, path: 'Optional[str]'=None, normalization: 'bool'=False, eps: 'float'=1e-12, **kwargs) ->None:
        if normalization and path is None:
            raise TypeError('You should provide a path if normalization is on.')
        super().__init__()
        self.normalization = normalization
        self.eps = eps
        if path is not None:
            rel_p = path.split('/')
            rel_p = rel_p[rel_p.index('deps'):]
            rel_p = '/'.join(rel_p)
            path = hydra.utils.get_original_cwd() + '/' + rel_p
        if normalization:
            mean_path = Path(path) / 'jfeats_mean.pt'
            std_path = Path(path) / 'jfeats_std.pt'
            self.register_buffer('mean', torch.load(mean_path))
            self.register_buffer('std', torch.load(std_path))

    def normalize(self, features: 'Tensor') ->Tensor:
        if self.normalization:
            features = (features - self.mean) / (self.std + self.eps)
        return features

    def unnormalize(self, features: 'Tensor') ->Tensor:
        if self.normalization:
            features = features * self.std + self.mean
        return features


def gaussian_filter1d(_inputs, sigma, truncate=4.0):
    if len(_inputs.shape) == 2:
        inputs = _inputs[None]
    else:
        inputs = _inputs
    sd = float(sigma)
    radius = int(truncate * sd + 0.5)
    sigma2 = sigma * sigma
    x = torch.arange(-radius, radius + 1, device=inputs.device, dtype=inputs.dtype)
    phi_x = torch.exp(-0.5 / sigma2 * x ** 2)
    phi_x = phi_x / phi_x.sum()
    groups = inputs.shape[-1]
    weights = torch.tile(phi_x, (groups, 1, 1))
    inputs = inputs.transpose(-1, -2)
    outputs = F.conv1d(inputs, weights, padding='same', groups=groups).transpose(-1, -2)
    return outputs.reshape(_inputs.shape)


mmm_joints = ['root', 'BP', 'BT', 'BLN', 'BUN', 'LS', 'LE', 'LW', 'RS', 'RE', 'RW', 'LH', 'LK', 'LA', 'LMrot', 'LF', 'RH', 'RK', 'RA', 'RMrot', 'RF']


LF, RF = mmm_joints.index('LF'), mmm_joints.index('RF')


LM, RM = mmm_joints.index('LMrot'), mmm_joints.index('RMrot')


def softmax(x, softness=1.0, dim=None):
    maxi, mini = x.max(dim=dim).values, x.min(dim=dim).values
    return maxi + torch.log(softness + torch.exp(mini - maxi))


def softmin(x, softness=1.0, dim=0):
    return -softmax(-x, softness=softness, dim=dim)


def get_floor(poses, jointstype='mmm'):
    assert jointstype == 'mmm'
    ndim = len(poses.shape)
    foot_heights = poses[..., (LM, LF, RM, RF), 1].min(-1).values
    floor_height = softmin(foot_heights, softness=0.5, dim=-1)
    return floor_height[(ndim - 2) * [None]].transpose(0, -1)


LH, RH = mmm_joints.index('LH'), mmm_joints.index('RH')


LS, RS = mmm_joints.index('LS'), mmm_joints.index('RS')


def get_forward_direction(poses, jointstype='mmm'):
    assert jointstype == 'mmm'
    across = poses[..., RH, :] - poses[..., LH, :] + poses[..., RS, :] - poses[..., LS, :]
    forward = torch.stack((-across[..., 2], across[..., 0]), axis=-1)
    forward = torch.nn.functional.normalize(forward, dim=-1)
    return forward


def matrix_of_angles(cos, sin, inv=False, dim=2):
    assert dim in [2, 3]
    sin = -sin if inv else sin
    if dim == 2:
        row1 = torch.stack((cos, -sin), axis=-1)
        row2 = torch.stack((sin, cos), axis=-1)
        return torch.stack((row1, row2), axis=-2)
    elif dim == 3:
        row1 = torch.stack((cos, -sin, 0 * cos), axis=-1)
        row2 = torch.stack((sin, cos, 0 * cos), axis=-1)
        row3 = torch.stack((0 * sin, 0 * cos, 1 + 0 * cos), axis=-1)
        return torch.stack((row1, row2, row3), axis=-2)


class Rifke(Joints2Jfeats):

    def __init__(self, jointstype: 'str'='mmm', path: 'Optional[str]'=None, normalization: 'bool'=False, forward_filter: 'bool'=False, **kwargs) ->None:
        super().__init__(path=path, normalization=normalization)
        self.jointstype = jointstype
        self.forward_filter = forward_filter

    def forward(self, joints: 'Tensor') ->Tensor:
        poses = joints.clone()
        poses[..., 1] -= get_floor(poses, jointstype=self.jointstype)
        translation = poses[..., 0, :].clone()
        root_y = translation[..., 1]
        trajectory = translation[..., [0, 2]]
        poses = poses[..., 1:, :]
        poses[..., [0, 2]] -= trajectory[..., None, :]
        vel_trajectory = torch.diff(trajectory, dim=-2)
        vel_trajectory = torch.cat((0 * vel_trajectory[..., [0], :], vel_trajectory), dim=-2)
        forward = get_forward_direction(poses, jointstype=self.jointstype)
        if self.forward_filter:
            forward = gaussian_filter1d(forward, 2)
            forward = torch.nn.functional.normalize(forward, dim=-1)
        angles = torch.atan2(*forward.transpose(0, -1)).transpose(0, -1)
        vel_angles = torch.diff(angles, dim=-1)
        vel_angles = torch.cat((0 * vel_angles[..., [0]], vel_angles), dim=-1)
        sin, cos = forward[..., 0], forward[..., 1]
        rotations_inv = matrix_of_angles(cos, sin, inv=True)
        poses_local = torch.einsum('...lj,...jk->...lk', poses[..., [0, 2]], rotations_inv)
        poses_local = torch.stack((poses_local[..., 0], poses[..., 1], poses_local[..., 1]), axis=-1)
        poses_features = rearrange(poses_local, '... joints xyz -> ... (joints xyz)')
        vel_trajectory_local = torch.einsum('...j,...jk->...k', vel_trajectory, rotations_inv)
        features = torch.cat((root_y[..., None], poses_features, vel_angles[..., None], vel_trajectory_local), -1)
        features = self.normalize(features)
        return features

    def inverse(self, features: 'Tensor') ->Tensor:
        features = self.unnormalize(features)
        root_y, poses_features, vel_angles, vel_trajectory_local = self.extract(features)
        angles = torch.cumsum(vel_angles, dim=-1)
        angles = angles - angles[..., [0]]
        cos, sin = torch.cos(angles), torch.sin(angles)
        rotations = matrix_of_angles(cos, sin, inv=False)
        poses_local = rearrange(poses_features, '... (joints xyz) -> ... joints xyz', xyz=3)
        poses = torch.einsum('...lj,...jk->...lk', poses_local[..., [0, 2]], rotations)
        poses = torch.stack((poses[..., 0], poses_local[..., 1], poses[..., 1]), axis=-1)
        vel_trajectory = torch.einsum('...j,...jk->...k', vel_trajectory_local, rotations)
        trajectory = torch.cumsum(vel_trajectory, dim=-2)
        trajectory = trajectory - trajectory[..., [0], :]
        poses = torch.cat((0 * poses[..., [0], :], poses), -2)
        poses[..., 0, 1] = root_y
        poses[..., [0, 2]] += trajectory[..., None, :]
        return poses

    def extract(self, features: 'Tensor') ->tuple[Tensor]:
        root_y = features[..., 0]
        poses_features = features[..., 1:-3]
        vel_angles = features[..., -3]
        vel_trajectory_local = features[..., -2:]
        return root_y, poses_features, vel_angles, vel_trajectory_local


class Rots2Joints(nn.Module):

    def __init__(self, path: 'Optional[str]'=None, normalization: 'bool'=False, eps: 'float'=1e-12, **kwargs) ->None:
        if normalization and path is None:
            raise TypeError('You should provide a path if normalization is on.')
        super().__init__()
        self.normalization = normalization
        self.eps = eps
        if path is not None:
            rel_p = path.split('/')
            rel_p = rel_p[rel_p.index('deps'):]
            rel_p = '/'.join(rel_p)
            path = hydra.utils.get_original_cwd() + '/' + rel_p
        if normalization:
            mean_path = Path(path) / 'mean.pt'
            std_path = Path(path) / 'std.pt'
            self.register_buffer('mean', torch.load(mean_path))
            self.register_buffer('std', torch.load(std_path))

    def normalize(self, features: 'Tensor') ->Tensor:
        if self.normalization:
            features = (features - self.mean) / (self.std + self.eps)
        return features

    def unnormalize(self, features: 'Tensor') ->Tensor:
        if self.normalization:
            features = features * self.std + self.mean
        return features


def slice_or_none(data, cslice):
    if data is None:
        return data
    else:
        return data[cslice]


mmm_joints_info = {'root': mmm_joints.index('root'), 'feet': [mmm_joints.index('LMrot'), mmm_joints.index('RMrot'), mmm_joints.index('LF'), mmm_joints.index('RF')], 'shoulders': [mmm_joints.index('LS'), mmm_joints.index('RS')], 'hips': [mmm_joints.index('LH'), mmm_joints.index('RH')]}


smplh_joints = ['pelvis', 'left_hip', 'right_hip', 'spine1', 'left_knee', 'right_knee', 'spine2', 'left_ankle', 'right_ankle', 'spine3', 'left_foot', 'right_foot', 'neck', 'left_collar', 'right_collar', 'head', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_index1', 'left_index2', 'left_index3', 'left_middle1', 'left_middle2', 'left_middle3', 'left_pinky1', 'left_pinky2', 'left_pinky3', 'left_ring1', 'left_ring2', 'left_ring3', 'left_thumb1', 'left_thumb2', 'left_thumb3', 'right_index1', 'right_index2', 'right_index3', 'right_middle1', 'right_middle2', 'right_middle3', 'right_pinky1', 'right_pinky2', 'right_pinky3', 'right_ring1', 'right_ring2', 'right_ring3', 'right_thumb1', 'right_thumb2', 'right_thumb3', 'nose', 'right_eye', 'left_eye', 'right_ear', 'left_ear', 'left_big_toe', 'left_small_toe', 'left_heel', 'right_big_toe', 'right_small_toe', 'right_heel', 'left_thumb', 'left_index', 'left_middle', 'left_ring', 'left_pinky', 'right_thumb', 'right_index', 'right_middle', 'right_ring', 'right_pinky']


smplnh_joints = ['pelvis', 'left_hip', 'right_hip', 'spine1', 'left_knee', 'right_knee', 'spine2', 'left_ankle', 'right_ankle', 'spine3', 'left_foot', 'right_foot', 'neck', 'left_collar', 'right_collar', 'head', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist']


smplnh_joints_info = {'root': smplnh_joints.index('pelvis'), 'feet': [smplnh_joints.index('left_ankle'), smplnh_joints.index('right_ankle'), smplnh_joints.index('left_foot'), smplnh_joints.index('right_foot')], 'shoulders': [smplnh_joints.index('left_shoulder'), smplnh_joints.index('right_shoulder')], 'hips': [smplnh_joints.index('left_hip'), smplnh_joints.index('right_hip')]}


root_joints = {'mmm': mmm_joints_info['root'], 'mmmns': mmm_joints_info['root'], 'smplmmm': mmm_joints_info['root'], 'smplnh': smplnh_joints_info['root'], 'smplh': smplh_joints.index('pelvis')}


def get_root_idx(joinstype):
    return root_joints[joinstype]


mmm2smplh_correspondence = {'root': 'pelvis', 'BP': 'spine1', 'BT': 'spine3', 'BLN': 'neck', 'BUN': 'head', 'LS': 'left_shoulder', 'LE': 'left_elbow', 'LW': 'left_wrist', 'RS': 'right_shoulder', 'RE': 'right_elbow', 'RW': 'right_wrist', 'LH': 'left_hip', 'LK': 'left_knee', 'LA': 'left_ankle', 'LMrot': 'left_heel', 'LF': 'left_foot', 'RH': 'right_hip', 'RK': 'right_knee', 'RA': 'right_ankle', 'RMrot': 'right_heel', 'RF': 'right_foot'}


smplh2mmm_indexes = [smplh_joints.index(mmm2smplh_correspondence[x]) for x in mmm_joints]


smplnh2smplh_correspondence = {key: key for key in smplnh_joints}


smplh2smplnh_indexes = [smplh_joints.index(smplnh2smplh_correspondence[x]) for x in smplnh_joints]


smplh_to_mmm_scaling_factor = 480 / 0.75


def smplh_to(jointstype, data, trans):
    if 'mmm' in jointstype:
        indexes = smplh2mmm_indexes
        data = data[..., indexes, :]
        if jointstype == 'mmm':
            data *= smplh_to_mmm_scaling_factor
        if jointstype == 'smplmmm':
            pass
        elif jointstype in ['mmm', 'mmmns']:
            data = data[..., [1, 2, 0]]
            data[..., 2] = -data[..., 2]
    elif jointstype == 'smplnh':
        indexes = smplh2smplnh_indexes
        data = data[..., indexes, :]
    elif jointstype == 'smplh':
        pass
    elif jointstype == 'vertices':
        pass
    else:
        raise NotImplementedError(f'SMPLH to {jointstype} is not implemented.')
    if jointstype != 'vertices':
        root_joint_idx = get_root_idx(jointstype)
        shift = trans[..., 0, :] - data[..., 0, root_joint_idx, :]
        data += shift[..., None, None, :]
    return data


class SMPLH(Rots2Joints):

    def __init__(self, path: 'str', jointstype: 'str'='mmm', input_pose_rep: 'str'='matrix', batch_size: 'int'=512, gender='neutral', **kwargs) ->None:
        super().__init__(path=None, normalization=False)
        self.batch_size = batch_size
        self.input_pose_rep = input_pose_rep
        self.jointstype = jointstype
        self.training = False
        rel_p = path.split('/')
        rel_p = rel_p[rel_p.index('data'):]
        rel_p = '/'.join(rel_p)
        path = hydra.utils.get_original_cwd() + '/' + rel_p
        with contextlib.redirect_stdout(None):
            self.smplh = SMPLHLayer(path, ext='pkl', gender=gender).eval()
        self.faces = self.smplh.faces
        for p in self.parameters():
            p.requires_grad = False

    def train(self, *args, **kwargs):
        return self

    def forward(self, smpl_data: 'dict', jointstype: 'Optional[str]'=None, input_pose_rep: 'Optional[str]'=None, batch_size: 'Optional[int]'=None) ->Tensor:
        jointstype = self.jointstype if jointstype is None else jointstype
        batch_size = self.batch_size if batch_size is None else batch_size
        input_pose_rep = self.input_pose_rep if input_pose_rep is None else input_pose_rep
        if input_pose_rep == 'xyz':
            raise NotImplementedError('You should use identity pose2joints instead')
        poses = smpl_data.rots
        trans = smpl_data.trans
        from functools import reduce
        save_shape_bs_len = poses.shape[:-3]
        nposes = reduce(operator.mul, save_shape_bs_len, 1)
        if poses.shape[-3] == 52:
            nohands = False
        elif poses.shape[-3] == 22:
            nohands = True
        else:
            raise NotImplementedError('Could not parse the poses.')
        matrix_poses = poses
        matrix_poses = matrix_poses.reshape((nposes, *matrix_poses.shape[-3:]))
        global_orient = matrix_poses[:, 0]
        if trans is None:
            trans = torch.zeros((*save_shape_bs_len, 3), dtype=poses.dtype, device=poses.device)
        trans_all = trans.reshape((nposes, *trans.shape[-1:]))
        body_pose = matrix_poses[:, 1:22]
        if nohands:
            left_hand_pose = None
            right_hand_pose = None
        else:
            hand_pose = matrix_poses[:, 22:]
            left_hand_pose = hand_pose[:, :15]
            right_hand_pose = hand_pose[:, 15:]
        n = len(body_pose)
        outputs = []
        for chunk in range(int((n - 1) / batch_size) + 1):
            chunk_slice = slice(chunk * batch_size, (chunk + 1) * batch_size)
            smpl_output = self.smplh(global_orient=slice_or_none(global_orient, chunk_slice), body_pose=slice_or_none(body_pose, chunk_slice), left_hand_pose=slice_or_none(left_hand_pose, chunk_slice), right_hand_pose=slice_or_none(right_hand_pose, chunk_slice), transl=slice_or_none(trans_all, chunk_slice))
            if jointstype == 'vertices':
                output_chunk = smpl_output.vertices
            else:
                joints = smpl_output.joints
                output_chunk = joints
            outputs.append(output_chunk)
        outputs = torch.cat(outputs)
        outputs = outputs.reshape((*save_shape_bs_len, *outputs.shape[1:]))
        outputs = smplh_to(jointstype, outputs, trans)
        return outputs

    def inverse(self, joints: 'Tensor') ->Tensor:
        raise NotImplementedError('Cannot inverse SMPLH layer.')


class Rots2Rfeats(nn.Module):

    def __init__(self, path: 'Optional[str]'=None, normalization: 'bool'=True, eps: 'float'=1e-12, **kwargs) ->None:
        if normalization and path is None:
            raise TypeError('You should provide a path if normalization is on.')
        super().__init__()
        self.normalization = normalization
        self.eps = eps
        if normalization:
            rel_p = path.split('/')
            if rel_p[-1] == 'separate_pairs':
                rel_p.remove('separate_pairs')
            rel_p = rel_p[rel_p.index('deps'):]
            rel_p = '/'.join(rel_p)
            path = hydra.utils.get_original_cwd() + '/' + rel_p
            mean_path = Path(path) / 'rfeats_mean.pt'
            std_path = Path(path) / 'rfeats_std.pt'
            self.register_buffer('mean', torch.load(mean_path))
            self.register_buffer('std', torch.load(std_path))

    def normalize(self, features: 'Tensor') ->Tensor:
        if self.normalization:
            features = (features - self.mean) / (self.std + self.eps)
        return features

    def unnormalize(self, features: 'Tensor') ->Tensor:
        if self.normalization:
            features = features * self.std + self.mean
        return features


@dataclass
class Datastruct:

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        self.__dict__[key] = value

    def get(self, key, default=None):
        return getattr(self, key, default)

    def __iter__(self):
        return self.keys()

    def keys(self):
        keys = [t.name for t in fields(self)]
        return iter(keys)

    def values(self):
        values = [getattr(self, t.name) for t in fields(self)]
        return iter(values)

    def items(self):
        data = [(t.name, getattr(self, t.name)) for t in fields(self)]
        return iter(data)

    def to(self, *args, **kwargs):
        for key in self.datakeys:
            if self[key] is not None:
                self[key] = self[key]
        return self

    @property
    def device(self):
        return self[self.datakeys[0]].device

    def detach(self):

        def detach_or_none(tensor):
            if tensor is not None:
                return tensor.detach()
            return None
        kwargs = {key: detach_or_none(self[key]) for key in self.datakeys}
        return self.transforms.Datastruct(**kwargs)

