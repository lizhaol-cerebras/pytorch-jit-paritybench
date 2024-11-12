
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


import torch


from collections import OrderedDict


import numpy as np


import scipy.linalg


from sklearn.cluster._kmeans import k_means


from torch.utils.dlpack import from_dlpack


from torch.utils.dlpack import to_dlpack


from typing import List


import re


import torch.nn as nn


import copy


from torch.utils.data import DataLoader


import torch.optim as optim


import torch.distributed as dist


from torch.utils.data.distributed import DistributedSampler


import random


from torch.utils.data import IterableDataset


from torch.utils.data import Dataset


from scipy import signal


from scipy.io import wavfile


import logging


import torch.nn.functional as F


from torch import Tensor


from torch import nn


from typing import Iterable


from typing import Optional


import math


import warnings


class Fbank(torch.nn.Module):

    def __init__(self, opts):
        super(Fbank, self).__init__()
        self.fbank = kaldifeat.Fbank(opts)

    def forward(self, waves: 'List[torch.Tensor]'):
        feats = self.fbank(waves)
        B, T, F = len(feats), feats[0].size(0), feats[0].size(1)
        feats = torch.cat(feats, axis=0)
        feats = torch.reshape(feats, (B, T, F))
        feats = feats - torch.mean(feats, dim=1, keepdim=True)
        return feats


class LinearModel(nn.Module):

    def __init__(self, input_dim):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
        nn.init.constant_(self.linear.weight, 1.0 / input_dim)
        nn.init.constant_(self.linear.bias, 0)

    def forward(self, x):
        out = self.linear(x)
        return out


class S3prlFrontend(nn.Module):
    """Speech Pretrained Representation Frontend."""

    def __init__(self, upstream_args: 'dict', download_dir: 'str'='./s3prl_hub', multilayer_feature: 'bool'=True, layer: 'int'=-1, frozen: 'bool'=False, frame_shift: 'int'=20, frame_length: 'int'=20, sample_rate: 'int'=16000):
        super().__init__()
        self.multilayer_feature = multilayer_feature
        self.layer = layer
        self.frozen = frozen
        if download_dir is not None:
            s3prl.util.download.set_dir(download_dir)
        assert upstream_args.get('name', None) in S3PRLUpstream.available_names()
        self.upstream = S3PRLUpstream(upstream_args.get('name'), path_or_url=upstream_args.get('path_or_url', None), normalize=upstream_args.get('normalize', False), extra_conf=upstream_args.get('extra_conf', None))
        if getattr(self.upstream.upstream, 'model', None):
            if getattr(self.upstream.upstream.model, 'feature_grad_mult', None) is not None:
                self.upstream.upstream.model.feature_grad_mult = 1.0
        self.upstream.eval()
        if layer != -1:
            layer_selections = [layer]
            assert not multilayer_feature, 'multilayer_feature must be False if layer is specified'
        else:
            layer_selections = None
        self.featurizer = Featurizer(self.upstream, layer_selections=layer_selections)
        assert self.featurizer.downsample_rate == sample_rate * frame_shift // 1000
        if self.frozen:
            for param in self.upstream.parameters():
                param.requires_grad_(False)
        else:
            for name, param in self.upstream.named_parameters():
                if 'mask_emb' in name:
                    param.requires_grad_(False)

    def output_size(self):
        return self.featurizer.output_size

    def forward(self, input: 'torch.Tensor', input_lengths: 'torch.LongTensor'):
        with (torch.no_grad() if self.frozen else contextlib.nullcontext()):
            feats, feats_lens = self.upstream(input, input_lengths)
        if self.layer != -1:
            layer = self.layer
            feats, feats_lens = feats[layer], feats_lens[layer]
            return feats, feats_lens
        if self.multilayer_feature:
            feats, feats_lens = self.featurizer(feats, feats_lens)
        else:
            feats, feats_lens = self.featurizer(feats[-1:], feats_lens[-1:])
        return feats, feats_lens


class Linear(nn.Module):
    """
    The linear transform for simple softmax loss
    """

    def __init__(self, emb_dim=512, class_num=1000):
        super(Linear, self).__init__()
        self.trans = nn.Sequential(nn.BatchNorm1d(emb_dim), nn.ReLU(inplace=True), nn.Linear(emb_dim, class_num))

    def forward(self, input, label):
        out = self.trans(input)
        return out


class Conv1d(nn.Conv1d):

    def _conv_forward(self, x: 'Tensor', weight: 'Tensor', bias: 'Optional[Tensor]') ->Tensor:
        return super()._conv_forward(x, weight, None if bias is None else bias)


class LayerNorm(nn.Module):
    """
    LayerNorm that supports two data formats: channels_last or channels_first.
    The ordering of the dimensions in the inputs.
    channels_last corresponds to inputs with shape (batch_size, T, channels)
    while channels_first corresponds to shape (batch_size, channels, T).
    """

    def __init__(self, C, eps=1e-06, data_format='channels_last'):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(C))
        self.bias = nn.Parameter(torch.zeros(C))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ['channels_last', 'channels_first']:
            raise NotImplementedError
        self.C = C,

    def forward(self, x):
        if self.data_format == 'channels_last':
            return F.layer_norm(x, self.C, self.weight, self.bias, self.eps)
        elif self.data_format == 'channels_first':
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            w = self.weight
            b = self.bias
            for _ in range(x.ndim - 2):
                w = w.unsqueeze(-1)
                b = b.unsqueeze(-1)
            x = w * x + b
            return x

    def extra_repr(self) ->str:
        return ', '.join([f'{k}={v}' for k, v in {'C': self.C, 'data_format': self.data_format, 'eps': self.eps}.items()])


class MultiHeadAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, embed_dim, num_heads, dropout=0.0, bias=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        if self.head_dim * num_heads != self.embed_dim:
            raise ValueError(f'embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {num_heads}).')
        self.scaling = self.head_dim ** -0.5
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _shape(self, tensor: 'torch.Tensor', seq_len, bsz):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(self, hidden_states: 'torch.Tensor') ->torch.Tensor:
        """Input shape: Batch x Time x Channel"""
        bsz, tgt_len, _ = hidden_states.size()
        query_states = self.q_proj(hidden_states) * self.scaling
        key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
        proj_shape = bsz * self.num_heads, -1, self.head_dim
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_probs = F.dropout(attn_weights, p=self.dropout, training=self.training)
        attn_output = torch.bmm(attn_probs, value_states)
        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
        attn_output = self.out_proj(attn_output)
        return attn_output


class ResidualAttentionBlock(nn.Module):

    def __init__(self, n_state: 'int', n_head: 'int', cross_attention: 'bool'=False):
        super().__init__()
        self.attn = MultiHeadAttention(n_state, n_head)
        self.attn_ln = LayerNorm(n_state)
        self.cross_attn = MultiHeadAttention(n_state, n_head) if cross_attention else None
        self.cross_attn_ln = LayerNorm(n_state) if cross_attention else None
        n_mlp = n_state * 4
        self.mlp = nn.Sequential(Linear(n_state, n_mlp), nn.GELU(), Linear(n_mlp, n_state))
        self.mlp_ln = LayerNorm(n_state)

    def forward(self, x: 'Tensor', xa: 'Optional[Tensor]'=None, mask: 'Optional[Tensor]'=None, kv_cache: 'Optional[dict]'=None):
        x = x + self.attn(self.attn_ln(x), mask=mask, kv_cache=kv_cache)[0]
        if self.cross_attn:
            x = x + self.cross_attn(self.cross_attn_ln(x), xa, kv_cache=kv_cache)[0]
        x = x + self.mlp(self.mlp_ln(x))
        return x


def sinusoids(length, channels, max_timescale=10000):
    """Returns sinusoids for positional embedding"""
    assert channels % 2 == 0
    log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
    scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
    return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)


class AudioEncoder(nn.Module):

    def __init__(self, n_mels: 'int', n_ctx: 'int', n_state: 'int', n_head: 'int', n_layer: 'int', layer_st: 'int', layer_ed: 'int'):
        super().__init__()
        self.conv1 = Conv1d(n_mels, n_state, kernel_size=3, padding=1)
        self.conv2 = Conv1d(n_state, n_state, kernel_size=3, stride=2, padding=1)
        self.register_buffer('positional_embedding', sinusoids(n_ctx, n_state))
        self.blocks: 'Iterable[ResidualAttentionBlock]' = nn.ModuleList([ResidualAttentionBlock(n_state, n_head) for _ in range(n_layer)])
        self.ln_post2 = LayerNorm(n_state * (layer_ed - layer_st + 1))
        self.layer_st = layer_st
        self.layer_ed = layer_ed

    def forward(self, x: 'Tensor'):
        """
        x : torch.Tensor, shape = (batch_size, n_mels, n_ctx)
            the mel spectrogram of the audio
        """
        x = x.permute(0, 2, 1)
        x = x.squeeze(1)
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = x.permute(0, 2, 1)
        assert x.shape[2:] == self.positional_embedding.shape[1:], 'incorrect audio shape'
        if self.positional_embedding.shape[0] > x.shape[1]:
            temp_positional_embedding = self.positional_embedding[:x.shape[1], :]
        elif self.positional_embedding.shape[0] < x.shape[1]:
            x = x[:, :self.positional_embedding.shape[0], :]
            temp_positional_embedding = self.positional_embedding
        else:
            temp_positional_embedding = self.positional_embedding
        x = x + temp_positional_embedding
        out = []
        for i, block in enumerate(self.blocks):
            x = block(x)
            if self.layer_st <= i <= self.layer_ed:
                out.append(x)
        xs = torch.cat(out, dim=-1)
        xs = self.ln_post2(xs)
        return xs


class whisper_encoder(torch.nn.Module):

    def __init__(self, frozen=False, n_mels=80, num_blocks=24, output_size=1280, n_head=20, layer_st=16, layer_ed=23, model_path=None, sample_rate=16000):
        super(whisper_encoder, self).__init__()
        self.encoder = AudioEncoder(n_mels=n_mels, n_layer=num_blocks, n_state=output_size, n_ctx=1500, n_head=n_head, layer_st=layer_st, layer_ed=layer_ed)
        self.frozen = frozen
        self.single_output_size = output_size
        self.concat_layer = layer_ed - layer_st + 1
        self.n_mels = n_mels
        if model_path:
            if dist.is_initialized():
                if dist.get_rank() == 0:
                    self._download_whisper_model(model_path)
                dist.barrier()
                self._load_pretrained_weights(model_path)
            else:
                self._download_whisper_model(model_path)
                self._load_pretrained_weights(model_path)
        if self.frozen:
            for param in self.encoder.parameters():
                param.requires_grad_(False)

    def _download_whisper_model(self, model_path='whisper_hub/large-v2.pt'):
        download_dir = os.path.dirname(model_path)
        if not os.path.exists(download_dir):
            os.makedirs(download_dir)
        if not os.path.isfile(model_path):
            None
            url = 'https://openaipublic.azureedge.net/main/whisper/models/81f7c96c852ee8fc832187b0132e569d6c3065a3252ed18e56effd0b6a73e524/large-v2.pt'
            urllib.request.urlretrieve(url, model_path)
            md5 = hashlib.md5(open(model_path, 'rb').read()).hexdigest()
            if md5 != '668764447eeda98eeba5ef7bfcb4cc3d':
                None
                os.remove(model_path)
                raise ValueError('MD5 checksum does not match!')
        else:
            None

    def _load_pretrained_weights(self, model_path):
        None
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        state_dict = state_dict['model_state_dict']
        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = k.replace('encoder.', '', 1)
            new_state_dict[new_key] = v
        missing_keys, unexpected_keys = self.encoder.load_state_dict(new_state_dict, strict=False)
        None
        for key in missing_keys:
            logging.warning('missing tensor: {}'.format(key))
        for key in unexpected_keys:
            logging.warning('unexpected tensor: {}'.format(key))

    def output_size(self):
        return self.single_output_size * self.concat_layer

    def forward(self, wavs, wavs_len):
        with torch.no_grad():
            processed_feats = []
            for i in range(wavs.size(0)):
                tf_tensor = wavs[i].unsqueeze(0)
                mat = whisper.log_mel_spectrogram(tf_tensor.squeeze(), n_mels=self.n_mels)
                processed_feats.append(mat)
            feat = torch.stack(processed_feats, dim=0)
        feat = feat.transpose(1, 2)
        x = self.encoder(feat)
        return x, None


def get_nonlinear(config_str, channels):
    nonlinear = nn.Sequential()
    for name in config_str.split('-'):
        if name == 'relu':
            nonlinear.add_module('relu', nn.ReLU(inplace=True))
        elif name == 'prelu':
            nonlinear.add_module('prelu', nn.PReLU(channels))
        elif name == 'batchnorm':
            nonlinear.add_module('batchnorm', nn.BatchNorm1d(channels))
        elif name == 'batchnorm_':
            nonlinear.add_module('batchnorm', nn.BatchNorm1d(channels, affine=False))
        else:
            raise ValueError('Unexpected module ({}).'.format(name))
    return nonlinear


class TDNNLayer(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=False, config_str='batchnorm-relu'):
        super(TDNNLayer, self).__init__()
        if padding < 0:
            assert kernel_size % 2 == 1, 'Expect equal paddings,                     but got even kernel size ({})'.format(kernel_size)
            padding = (kernel_size - 1) // 2 * dilation
        self.linear = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)
        self.nonlinear = get_nonlinear(config_str, out_channels)

    def forward(self, x):
        x = self.linear(x)
        x = self.nonlinear(x)
        return x


class CAMLayer(nn.Module):

    def __init__(self, bn_channels, out_channels, kernel_size, stride, padding, dilation, bias, reduction=2):
        super(CAMLayer, self).__init__()
        self.linear_local = nn.Conv1d(bn_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)
        self.linear1 = nn.Conv1d(bn_channels, bn_channels // reduction, 1)
        self.relu = nn.ReLU(inplace=True)
        self.linear2 = nn.Conv1d(bn_channels // reduction, out_channels, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.linear_local(x)
        context = x.mean(-1, keepdim=True) + self.seg_pooling(x)
        context = self.relu(self.linear1(context))
        m = self.sigmoid(self.linear2(context))
        return y * m

    def seg_pooling(self, x, seg_len: 'int'=100, stype: 'str'='avg'):
        if stype == 'avg':
            seg = F.avg_pool1d(x, kernel_size=seg_len, stride=seg_len, ceil_mode=True)
        elif stype == 'max':
            seg = F.max_pool1d(x, kernel_size=seg_len, stride=seg_len, ceil_mode=True)
        else:
            raise ValueError('Wrong segment pooling type.')
        shape = seg.shape
        seg = seg.unsqueeze(-1).expand(shape[0], shape[1], shape[2], seg_len).reshape(shape[0], shape[1], -1)
        seg = seg[..., :x.shape[-1]]
        return seg


class CAMDenseTDNNLayer(nn.Module):

    def __init__(self, in_channels, out_channels, bn_channels, kernel_size, stride=1, dilation=1, bias=False, config_str='batchnorm-relu'):
        super(CAMDenseTDNNLayer, self).__init__()
        assert kernel_size % 2 == 1, 'Expect equal paddings,                 but got even kernel size ({})'.format(kernel_size)
        padding = (kernel_size - 1) // 2 * dilation
        self.nonlinear1 = get_nonlinear(config_str, in_channels)
        self.linear1 = nn.Conv1d(in_channels, bn_channels, 1, bias=False)
        self.nonlinear2 = get_nonlinear(config_str, bn_channels)
        self.cam_layer = CAMLayer(bn_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)

    def bn_function(self, x):
        return self.linear1(self.nonlinear1(x))

    def forward(self, x):
        x = self.bn_function(x)
        x = self.cam_layer(self.nonlinear2(x))
        return x


class CAMDenseTDNNBlock(nn.ModuleList):

    def __init__(self, num_layers, in_channels, out_channels, bn_channels, kernel_size, stride=1, dilation=1, bias=False, config_str='batchnorm-relu'):
        super(CAMDenseTDNNBlock, self).__init__()
        for i in range(num_layers):
            layer = CAMDenseTDNNLayer(in_channels=in_channels + i * out_channels, out_channels=out_channels, bn_channels=bn_channels, kernel_size=kernel_size, stride=stride, dilation=dilation, bias=bias, config_str=config_str)
            self.add_module('tdnnd%d' % (i + 1), layer)

    def forward(self, x):
        for layer in self:
            x = torch.cat([x, layer(x)], dim=1)
        return x


class TransitLayer(nn.Module):

    def __init__(self, in_channels, out_channels, bias=True, config_str='batchnorm-relu'):
        super(TransitLayer, self).__init__()
        self.nonlinear = get_nonlinear(config_str, in_channels)
        self.linear = nn.Conv1d(in_channels, out_channels, 1, bias=bias)

    def forward(self, x):
        x = self.nonlinear(x)
        x = self.linear(x)
        return x


class DenseLayer(nn.Module):

    def __init__(self, in_channels, out_channels, bias=False, config_str='batchnorm-relu'):
        super(DenseLayer, self).__init__()
        self.linear = nn.Conv1d(in_channels, out_channels, 1, bias=bias)
        self.nonlinear = get_nonlinear(config_str, out_channels)

    def forward(self, x):
        if len(x.shape) == 2:
            x = self.linear(x.unsqueeze(dim=-1)).squeeze(dim=-1)
        else:
            x = self.linear(x)
        x = self.nonlinear(x)
        return x


class BasicResBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=(stride, 1), padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=(stride, 1), bias=False), nn.BatchNorm2d(self.expansion * planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class FCM(nn.Module):

    def __init__(self, block, num_blocks, m_channels=32, feat_dim=80):
        super(FCM, self).__init__()
        self.in_planes = m_channels
        self.conv1 = nn.Conv2d(1, m_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(m_channels)
        self.layer1 = self._make_layer(block, m_channels, num_blocks[0], stride=2)
        self.layer2 = self._make_layer(block, m_channels, num_blocks[1], stride=2)
        self.conv2 = nn.Conv2d(m_channels, m_channels, kernel_size=3, stride=(2, 1), padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(m_channels)
        self.out_channels = m_channels * (feat_dim // 8)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.unsqueeze(1)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = F.relu(self.bn2(self.conv2(out)))
        shape = out.shape
        out = out.reshape(shape[0], shape[1] * shape[2], shape[3])
        return out


class CAMPPlus(nn.Module):

    def __init__(self, feat_dim=80, embed_dim=512, pooling_func='TSTP', growth_rate=32, bn_size=4, init_channels=128, config_str='batchnorm-relu'):
        super(CAMPPlus, self).__init__()
        self.head = FCM(block=BasicResBlock, num_blocks=[2, 2], feat_dim=feat_dim)
        channels = self.head.out_channels
        self.xvector = nn.Sequential(OrderedDict([('tdnn', TDNNLayer(channels, init_channels, 5, stride=2, dilation=1, padding=-1, config_str=config_str))]))
        channels = init_channels
        for i, (num_layers, kernel_size, dilation) in enumerate(zip((12, 24, 16), (3, 3, 3), (1, 2, 2))):
            block = CAMDenseTDNNBlock(num_layers=num_layers, in_channels=channels, out_channels=growth_rate, bn_channels=bn_size * growth_rate, kernel_size=kernel_size, dilation=dilation, config_str=config_str)
            self.xvector.add_module('block%d' % (i + 1), block)
            channels = channels + num_layers * growth_rate
            self.xvector.add_module('transit%d' % (i + 1), TransitLayer(channels, channels // 2, bias=False, config_str=config_str))
            channels //= 2
        self.xvector.add_module('out_nonlinear', get_nonlinear(config_str, channels))
        self.pool = getattr(pooling_layers, pooling_func)(in_dim=channels)
        self.pool_out_dim = self.pool.get_out_dim()
        self.xvector.add_module('stats', self.pool)
        self.xvector.add_module('dense', DenseLayer(self.pool_out_dim, embed_dim, config_str='batchnorm_'))
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def get_frame_level_feat(self, x):
        x = x.permute(0, 2, 1)
        x = self.head(x)
        for layer in self.xvector[:-2]:
            x = layer(x)
        out = x.permute(0, 2, 1)
        return out

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.head(x)
        x = self.xvector(x)
        return x


class Res2Conv1dReluBn(nn.Module):
    """
    in_channels == out_channels == channels
    """

    def __init__(self, channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=True, scale=4):
        super().__init__()
        assert channels % scale == 0, '{} % {} != 0'.format(channels, scale)
        self.scale = scale
        self.width = channels // scale
        self.nums = scale if scale == 1 else scale - 1
        self.convs = []
        self.bns = []
        for i in range(self.nums):
            self.convs.append(nn.Conv1d(self.width, self.width, kernel_size, stride, padding, dilation, bias=bias))
            self.bns.append(nn.BatchNorm1d(self.width))
        self.convs = nn.ModuleList(self.convs)
        self.bns = nn.ModuleList(self.bns)

    def forward(self, x):
        out = []
        spx = torch.split(x, self.width, 1)
        sp = spx[0]
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            if i >= 1:
                sp = sp + spx[i]
            sp = conv(sp)
            sp = bn(F.relu(sp))
            out.append(sp)
        if self.scale != 1:
            out.append(spx[self.nums])
        out = torch.cat(out, dim=1)
        return out


class Conv1dReluBn(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=True):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=bias)
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        return self.bn(F.relu(self.conv(x)))


class SE_Connect(nn.Module):

    def __init__(self, channels, se_bottleneck_dim=128):
        super().__init__()
        self.linear1 = nn.Linear(channels, se_bottleneck_dim)
        self.linear2 = nn.Linear(se_bottleneck_dim, channels)

    def forward(self, x):
        out = x.mean(dim=2)
        out = F.relu(self.linear1(out))
        out = torch.sigmoid(self.linear2(out))
        out = x * out.unsqueeze(2)
        return out


class SE_Res2Block(nn.Module):

    def __init__(self, channels, kernel_size, stride, padding, dilation, scale):
        super().__init__()
        self.se_res2block = nn.Sequential(Conv1dReluBn(channels, channels, kernel_size=1, stride=1, padding=0), Res2Conv1dReluBn(channels, kernel_size, stride, padding, dilation, scale=scale), Conv1dReluBn(channels, channels, kernel_size=1, stride=1, padding=0), SE_Connect(channels))

    def forward(self, x):
        return x + self.se_res2block(x)


class ECAPA_TDNN(nn.Module):

    def __init__(self, channels=512, feat_dim=80, embed_dim=192, pooling_func='ASTP', global_context_att=False, emb_bn=False):
        super().__init__()
        self.layer1 = Conv1dReluBn(feat_dim, channels, kernel_size=5, padding=2)
        self.layer2 = SE_Res2Block(channels, kernel_size=3, stride=1, padding=2, dilation=2, scale=8)
        self.layer3 = SE_Res2Block(channels, kernel_size=3, stride=1, padding=3, dilation=3, scale=8)
        self.layer4 = SE_Res2Block(channels, kernel_size=3, stride=1, padding=4, dilation=4, scale=8)
        cat_channels = channels * 3
        out_channels = 512 * 3
        self.conv = nn.Conv1d(cat_channels, out_channels, kernel_size=1)
        self.pool = getattr(pooling_layers, pooling_func)(in_dim=out_channels, global_context_att=global_context_att)
        self.pool_out_dim = self.pool.get_out_dim()
        self.bn = nn.BatchNorm1d(self.pool_out_dim)
        self.linear = nn.Linear(self.pool_out_dim, embed_dim)
        self.emb_bn = emb_bn
        if emb_bn:
            self.bn2 = nn.BatchNorm1d(embed_dim)
        else:
            self.bn2 = nn.Identity()

    def _get_frame_level_feat(self, x):
        x = x.permute(0, 2, 1)
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        out = torch.cat([out2, out3, out4], dim=1)
        out = self.conv(out)
        return out

    def get_frame_level_feat(self, x):
        out = self._get_frame_level_feat(x).permute(0, 2, 1)
        return out

    def forward(self, x):
        out = F.relu(self._get_frame_level_feat(x))
        out = self.bn(self.pool(out))
        out = self.linear(out)
        if self.emb_bn:
            out = self.bn2(out)
        return out


class ReLU(nn.Hardtanh):

    def __init__(self, inplace=False):
        super(ReLU, self).__init__(0.0, 20.0, inplace)

    def __repr__(self):
        inplace_str = 'inplace' if self.inplace else ''
        return self.__class__.__name__ + ' (' + inplace_str + ')'


class AFF(nn.Module):

    def __init__(self, channels=64, r=4):
        super(AFF, self).__init__()
        inter_channels = int(channels // r)
        self.local_att = nn.Sequential(nn.Conv2d(channels * 2, inter_channels, kernel_size=1, stride=1, padding=0), nn.BatchNorm2d(inter_channels), nn.SiLU(inplace=True), nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0), nn.BatchNorm2d(channels))

    def forward(self, x, ds_y):
        xa = torch.cat((x, ds_y), dim=1)
        x_att = self.local_att(xa)
        x_att = 1.0 + torch.tanh(x_att)
        xo = torch.mul(x, x_att) + torch.mul(ds_y, 2.0 - x_att)
        return xo


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution without padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlockERes2Net(nn.Module):

    def __init__(self, in_planes, planes, stride=1, baseWidth=32, scale=2, expansion=2):
        super(BasicBlockERes2Net, self).__init__()
        width = int(math.floor(planes * (baseWidth / 64.0)))
        self.conv1 = conv1x1(in_planes, width * scale, stride)
        self.bn1 = nn.BatchNorm2d(width * scale)
        self.nums = scale
        self.expansion = expansion
        convs = []
        bns = []
        for i in range(self.nums):
            convs.append(conv3x3(width, width))
            bns.append(nn.BatchNorm2d(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)
        self.relu = ReLU(inplace=True)
        self.conv3 = conv1x1(width * scale, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(self.expansion * planes))
        self.stride = stride
        self.width = width
        self.scale = scale

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        spx = torch.split(out, self.width, 1)
        sp = spx[0]
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            if i >= 1:
                sp = sp + spx[i]
            sp = conv(sp)
            sp = self.relu(bn(sp))
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)
        out = self.conv3(out)
        out = self.bn3(out)
        residual = self.shortcut(x)
        out += residual
        out = self.relu(out)
        return out


class BasicBlockERes2Net_diff_AFF(nn.Module):

    def __init__(self, in_planes, planes, stride=1, baseWidth=32, scale=2, expansion=2):
        super(BasicBlockERes2Net_diff_AFF, self).__init__()
        width = int(math.floor(planes * (baseWidth / 64.0)))
        self.conv1 = conv1x1(in_planes, width * scale, stride)
        self.bn1 = nn.BatchNorm2d(width * scale)
        self.nums = scale
        self.expansion = expansion
        self.conv2_1 = conv3x3(width, width)
        self.bn2_1 = nn.BatchNorm2d(width)
        convs = []
        fuse_models = []
        bns = []
        for i in range(self.nums - 1):
            convs.append(conv3x3(width, width))
            bns.append(nn.BatchNorm2d(width))
            fuse_models.append(AFF(channels=width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)
        self.fuse_models = nn.ModuleList(fuse_models)
        self.relu = ReLU(inplace=True)
        self.conv3 = conv1x1(width * scale, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(self.expansion * planes))
        self.stride = stride
        self.width = width
        self.scale = scale

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        spx = torch.split(out, self.width, 1)
        sp = spx[0]
        sp = self.conv2_1(sp)
        sp = self.relu(self.bn2_1(sp))
        out = sp
        for i, (conv, bn, fuse_model) in enumerate(zip(self.convs, self.bns, self.fuse_models), 1):
            sp = fuse_model(sp, spx[i])
            sp = conv(sp)
            sp = self.relu(bn(sp))
            out = torch.cat((out, sp), 1)
        out = self.conv3(out)
        out = self.bn3(out)
        residual = self.shortcut(x)
        out += residual
        out = self.relu(out)
        return out


class ERes2Net(nn.Module):

    def __init__(self, m_channels, num_blocks, baseWidth=32, scale=2, expansion=2, block=BasicBlockERes2Net, block_fuse=BasicBlockERes2Net_diff_AFF, feat_dim=80, embed_dim=192, pooling_func='TSTP', two_emb_layer=False):
        super(ERes2Net, self).__init__()
        self.in_planes = m_channels
        self.feat_dim = feat_dim
        self.embed_dim = embed_dim
        self.stats_dim = int(feat_dim / 8) * m_channels * 8
        self.two_emb_layer = two_emb_layer
        self.expansion = expansion
        self.conv1 = nn.Conv2d(1, m_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(m_channels)
        self.layer1 = self._make_layer(block, m_channels, num_blocks[0], stride=1, baseWidth=baseWidth, scale=scale, expansion=expansion)
        self.layer2 = self._make_layer(block, m_channels * 2, num_blocks[1], stride=2, baseWidth=baseWidth, scale=scale, expansion=expansion)
        self.layer3 = self._make_layer(block_fuse, m_channels * 4, num_blocks[2], stride=2, baseWidth=baseWidth, scale=scale, expansion=expansion)
        self.layer4 = self._make_layer(block_fuse, m_channels * 8, num_blocks[3], stride=2, baseWidth=baseWidth, scale=scale, expansion=expansion)
        self.layer1_downsample = nn.Conv2d(m_channels * expansion, m_channels * expansion * 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.layer2_downsample = nn.Conv2d(m_channels * expansion * 2, m_channels * expansion * 4, kernel_size=3, padding=1, stride=2, bias=False)
        self.layer3_downsample = nn.Conv2d(m_channels * expansion * 4, m_channels * expansion * 8, kernel_size=3, padding=1, stride=2, bias=False)
        self.fuse_mode12 = AFF(channels=m_channels * expansion * 2)
        self.fuse_mode123 = AFF(channels=m_channels * expansion * 4)
        self.fuse_mode1234 = AFF(channels=m_channels * expansion * 8)
        self.pool = getattr(pooling_layers, pooling_func)(in_dim=self.stats_dim * expansion)
        self.pool_out_dim = self.pool.get_out_dim()
        self.seg_1 = nn.Linear(self.pool_out_dim, embed_dim)
        if self.two_emb_layer:
            self.seg_bn_1 = nn.BatchNorm1d(embed_dim, affine=False)
            self.seg_2 = nn.Linear(embed_dim, embed_dim)
        else:
            self.seg_bn_1 = nn.Identity()
            self.seg_2 = nn.Identity()

    def _make_layer(self, block, planes, num_blocks, stride, baseWidth=32, scale=2, expansion=2):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, baseWidth, scale, expansion))
            self.in_planes = planes * self.expansion
        return nn.Sequential(*layers)

    def _get_frame_level_feat(self, x):
        x = x.permute(0, 2, 1)
        x = x.unsqueeze_(1)
        out = F.relu(self.bn1(self.conv1(x)))
        out1 = self.layer1(out)
        out2 = self.layer2(out1)
        out1_downsample = self.layer1_downsample(out1)
        fuse_out12 = self.fuse_mode12(out2, out1_downsample)
        out3 = self.layer3(out2)
        fuse_out12_downsample = self.layer2_downsample(fuse_out12)
        fuse_out123 = self.fuse_mode123(out3, fuse_out12_downsample)
        out4 = self.layer4(out3)
        fuse_out123_downsample = self.layer3_downsample(fuse_out123)
        fuse_out1234 = self.fuse_mode1234(out4, fuse_out123_downsample)
        return fuse_out1234

    def get_frame_level_feat(self, x):
        out = self._get_frame_level_feat(x)
        out = out.transpose(1, 3)
        out = torch.flatten(out, 2, -1)
        return out

    def forward(self, x):
        fuse_out1234 = self._get_frame_level_feat(x)
        stats = self.pool(fuse_out1234)
        embed_a = self.seg_1(stats)
        if self.two_emb_layer:
            out = F.relu(embed_a)
            out = self.seg_bn_1(out)
            embed_b = self.seg_2(out)
            return embed_b
        else:
            return embed_a


class Inverted_Bottleneck(nn.Module):

    def __init__(self, dim):
        super(Inverted_Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(dim, 4 * dim, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(4 * dim)
        self.conv2 = nn.Conv2d(4 * dim, 4 * dim, kernel_size=3, padding=1, groups=4 * dim, bias=False)
        self.bn2 = nn.BatchNorm2d(4 * dim)
        self.conv3 = nn.Conv2d(4 * dim, dim, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(dim)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += x
        out = F.relu(out)
        return out


class Gemini_DF_ResNet(nn.Module):

    def __init__(self, depths, dims, feat_dim=40, embed_dim=128, pooling_func='TSTP', two_emb_layer=False):
        super(Gemini_DF_ResNet, self).__init__()
        self.feat_dim = feat_dim
        self.embed_dim = embed_dim
        self.stats_dim = int(feat_dim / 8 / 2) * dims[-1]
        self.two_emb_layer = two_emb_layer
        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(nn.Conv2d(1, dims[0], kernel_size=3, stride=1, padding=1, bias=False), nn.BatchNorm2d(dims[0]), nn.ReLU())
        self.downsample_layers.append(stem)
        stride_f = [2, 2, 2, 2]
        stride_t = [1, 2, 1, 1]
        for i in range(4):
            downsample_layer = nn.Sequential(nn.Conv2d(dims[i], dims[i + 1], kernel_size=3, stride=(stride_f[i], stride_t[i]), padding=1, bias=False), nn.BatchNorm2d(dims[i + 1]))
            self.downsample_layers.append(downsample_layer)
        self.stages = nn.ModuleList()
        for i in range(4):
            stage = nn.Sequential(*[Inverted_Bottleneck(dim=dims[i + 1]) for _ in range(depths[i])])
            self.stages.append(stage)
        self.pool = getattr(pooling_layers, pooling_func)(in_dim=self.stats_dim)
        self.pool_out_dim = self.pool.get_out_dim()
        self.seg_1 = nn.Linear(self.pool_out_dim, embed_dim)
        if self.two_emb_layer:
            self.seg_bn_1 = nn.BatchNorm1d(embed_dim, affine=False)
            self.seg_2 = nn.Linear(embed_dim, embed_dim)
        else:
            self.seg_bn_1 = nn.Identity()
            self.seg_2 = nn.Identity()

    def _get_frame_level_feat(self, x):
        x = x.permute(0, 2, 1)
        x = x.unsqueeze_(1)
        out = self.downsample_layers[0](x)
        out = self.downsample_layers[1](out)
        out = self.stages[0](out)
        out = self.downsample_layers[2](out)
        out = self.stages[1](out)
        out = self.downsample_layers[3](out)
        out = self.stages[2](out)
        out = self.downsample_layers[4](out)
        out = self.stages[3](out)
        return out

    def get_frame_level_feat(self, x):
        out = self._get_frame_level_feat(x)
        out = out.transpose(1, 3)
        out = torch.flatten(out, 2, -1)
        return out

    def forward(self, x):
        out = self._get_frame_level_feat(x)
        stats = self.pool(out)
        embed_a = self.seg_1(stats)
        if self.two_emb_layer:
            out = F.relu(embed_a)
            out = self.seg_bn_1(out)
            embed_b = self.seg_2(out)
            return embed_a, embed_b
        else:
            return torch.tensor(0.0), embed_a


class TAP(nn.Module):
    """
    Temporal average pooling, only first-order mean is considered
    """

    def __init__(self, in_dim=0, **kwargs):
        super(TAP, self).__init__()
        self.in_dim = in_dim

    def forward(self, x):
        pooling_mean = x.mean(dim=-1)
        pooling_mean = pooling_mean.flatten(start_dim=1)
        return pooling_mean

    def get_out_dim(self):
        self.out_dim = self.in_dim
        return self.out_dim


class TSDP(nn.Module):
    """
    Temporal standard deviation pooling, only second-order std is considered
    """

    def __init__(self, in_dim=0, **kwargs):
        super(TSDP, self).__init__()
        self.in_dim = in_dim

    def forward(self, x):
        pooling_std = torch.sqrt(torch.var(x, dim=-1) + 1e-07)
        pooling_std = pooling_std.flatten(start_dim=1)
        return pooling_std

    def get_out_dim(self):
        self.out_dim = self.in_dim
        return self.out_dim


class TSTP(nn.Module):
    """
    Temporal statistics pooling, concatenate mean and std, which is used in
    x-vector
    Comment: simple concatenation can not make full use of both statistics
    """

    def __init__(self, in_dim=0, **kwargs):
        super(TSTP, self).__init__()
        self.in_dim = in_dim

    def forward(self, x):
        pooling_mean = x.mean(dim=-1)
        pooling_std = torch.sqrt(torch.var(x, dim=-1) + 1e-07)
        pooling_mean = pooling_mean.flatten(start_dim=1)
        pooling_std = pooling_std.flatten(start_dim=1)
        stats = torch.cat((pooling_mean, pooling_std), 1)
        return stats

    def get_out_dim(self):
        self.out_dim = self.in_dim * 2
        return self.out_dim


class ASTP(nn.Module):
    """ Attentive statistics pooling: Channel- and context-dependent
        statistics pooling, first used in ECAPA_TDNN.
    """

    def __init__(self, in_dim, bottleneck_dim=128, global_context_att=False, **kwargs):
        super(ASTP, self).__init__()
        self.in_dim = in_dim
        self.global_context_att = global_context_att
        if global_context_att:
            self.linear1 = nn.Conv1d(in_dim * 3, bottleneck_dim, kernel_size=1)
        else:
            self.linear1 = nn.Conv1d(in_dim, bottleneck_dim, kernel_size=1)
        self.linear2 = nn.Conv1d(bottleneck_dim, in_dim, kernel_size=1)

    def forward(self, x):
        """
        x: a 3-dimensional tensor in tdnn-based architecture (B,F,T)
            or a 4-dimensional tensor in resnet architecture (B,C,F,T)
            0-dim: batch-dimension, last-dim: time-dimension (frame-dimension)
        """
        if len(x.shape) == 4:
            x = x.reshape(x.shape[0], x.shape[1] * x.shape[2], x.shape[3])
        assert len(x.shape) == 3
        if self.global_context_att:
            context_mean = torch.mean(x, dim=-1, keepdim=True).expand_as(x)
            context_std = torch.sqrt(torch.var(x, dim=-1, keepdim=True) + 1e-07).expand_as(x)
            x_in = torch.cat((x, context_mean, context_std), dim=1)
        else:
            x_in = x
        alpha = torch.tanh(self.linear1(x_in))
        alpha = torch.softmax(self.linear2(alpha), dim=2)
        mean = torch.sum(alpha * x, dim=2)
        var = torch.sum(alpha * x ** 2, dim=2) - mean ** 2
        std = torch.sqrt(var.clamp(min=1e-07))
        return torch.cat([mean, std], dim=1)

    def get_out_dim(self):
        self.out_dim = 2 * self.in_dim
        return self.out_dim


class ASP(nn.Module):

    def __init__(self, in_planes, acoustic_dim):
        super(ASP, self).__init__()
        outmap_size = int(acoustic_dim / 8)
        self.out_dim = in_planes * 8 * outmap_size * 2
        self.attention = nn.Sequential(nn.Conv1d(in_planes * 8 * outmap_size, 128, kernel_size=1), nn.ReLU(), nn.BatchNorm1d(128), nn.Conv1d(128, in_planes * 8 * outmap_size, kernel_size=1), nn.Softmax(dim=2))

    def forward(self, x):
        x = x.reshape(x.size()[0], -1, x.size()[-1])
        w = self.attention(x)
        mu = torch.sum(x * w, dim=2)
        sg = torch.sqrt((torch.sum(x ** 2 * w, dim=2) - mu ** 2).clamp(min=1e-05))
        x = torch.cat((mu, sg), 1)
        x = x.view(x.size()[0], -1)
        return x


class MHASTP(torch.nn.Module):
    """ Multi head attentive statistics pooling
    Reference:
        Self Multi-Head Attention for Speaker Recognition
        https://arxiv.org/pdf/1906.09890.pdf
    """

    def __init__(self, in_dim, layer_num=2, head_num=2, d_s=1, bottleneck_dim=64, **kwargs):
        super(MHASTP, self).__init__()
        assert in_dim % head_num == 0
        self.in_dim = in_dim
        self.head_num = head_num
        d_model = int(in_dim / head_num)
        channel_dims = [bottleneck_dim for i in range(layer_num + 1)]
        if d_s > 1:
            d_s = d_model
        else:
            d_s = 1
        self.d_s = d_s
        channel_dims[0], channel_dims[-1] = d_model, d_s
        heads_att_trans = []
        for i in range(self.head_num):
            att_trans = nn.Sequential()
            for i in range(layer_num - 1):
                att_trans.add_module('att_' + str(i), nn.Conv1d(channel_dims[i], channel_dims[i + 1], 1, 1))
                att_trans.add_module('tanh' + str(i), nn.Tanh())
            att_trans.add_module('att_' + str(layer_num - 1), nn.Conv1d(channel_dims[layer_num - 1], channel_dims[layer_num], 1, 1))
            heads_att_trans.append(att_trans)
        self.heads_att_trans = nn.ModuleList(heads_att_trans)

    def forward(self, input):
        """
        input: a 3-dimensional tensor in xvector architecture
            or a 4-dimensional tensor in resnet architecture
            0-dim: batch-dimension, last-dim: time-dimension (frame-dimension)
        """
        if len(input.shape) == 4:
            input = input.reshape(input.shape[0], input.shape[1] * input.shape[2], input.shape[3])
        assert len(input.shape) == 3
        bs, f_dim, t_dim = input.shape
        chunks = torch.chunk(input, self.head_num, 1)
        chunks_out = []
        for i, layer in enumerate(self.heads_att_trans):
            att_score = layer(chunks[i])
            alpha = F.softmax(att_score, dim=-1)
            mean = torch.sum(alpha * chunks[i], dim=2)
            var = torch.sum(alpha * chunks[i] ** 2, dim=2) - mean ** 2
            std = torch.sqrt(var.clamp(min=1e-07))
            chunks_out.append(torch.cat((mean, std), dim=1))
        out = torch.cat(chunks_out, dim=1)
        return out

    def get_out_dim(self):
        self.out_dim = 2 * self.in_dim
        return self.out_dim


class MQMHASTP(torch.nn.Module):
    """ An attentive pooling
    Reference:
        multi query multi head attentive statistics pooling
        https://arxiv.org/pdf/2110.05042.pdf
    Args:
        in_dim: the feature dimension of input
        layer_num: the number of layer in the pooling layer
        query_num: the number of querys
        head_num: the number of heads
        bottleneck_dim: the bottleneck dimension

    SA (H = 1, Q = 1, n = 2, d_s = 1) ref:
        https://www.danielpovey.com/files/2018_interspeech_xvector_attention.pdf
    MHA (H > 1, Q = 1, n = 1, d_s = 1) ref:
        https://arxiv.org/pdf/1906.09890.pdf
    AS (H = 1, Q > 1, n = 2, d_s = 1) ref:
        https://arxiv.org/pdf/1803.10963.pdf
    VSA (H = 1, Q > 1, n = 2, d_s = d_h) ref:
        http://www.interspeech2020.org/uploadfile/pdf/Mon-2-10-5.pdf
    """

    def __init__(self, in_dim, layer_num=2, query_num=2, head_num=8, d_s=2, bottleneck_dim=64, **kwargs):
        super(MQMHASTP, self).__init__()
        self.n_query = nn.ModuleList([MHASTP(in_dim, layer_num=layer_num, head_num=head_num, d_s=d_s, bottleneck_dim=bottleneck_dim) for i in range(query_num)])
        self.query_num = query_num
        self.in_dim = in_dim

    def forward(self, input):
        """
        input: a 3-dimensional tensor in xvector architecture
            or a 4-dimensional tensor in resnet architecture
            0-dim: batch-dimension, last-dim: time-dimension (frame-dimension)
        """
        if len(input.shape) == 4:
            input = input.reshape(input.shape[0], input.shape[1] * input.shape[2], input.shape[3])
        assert len(input.shape) == 3
        res = []
        for i, layer in enumerate(self.n_query):
            res.append(layer(input))
        out = torch.cat(res, dim=-1)
        return out

    def get_out_dim(self):
        self.out_dim = self.in_dim * 2 * self.query_num
        return self.out_dim


class SphereFace2(nn.Module):
    """Implement of sphereface2 for speaker verification:
        Reference:
            [1] Exploring Binary Classification Loss for Speaker Verification
            https://ieeexplore.ieee.org/abstract/document/10094954
            [2] Sphereface2: Binary classification is all you need
            for deep face recognition
            https://arxiv.org/pdf/2108.01513
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            scale: norm of input feature
            margin: margin
            lanbuda: weight of positive and negative pairs
            t: parameter for adjust score distribution
            margin_type: A:cos(theta+margin) or C:cos(theta)-margin
        Recommend margin:
            training: 0.2 for C and 0.15 for A
            LMF: 0.3 for C and 0.25 for A
        """

    def __init__(self, in_features, out_features, scale=32.0, margin=0.2, lanbuda=0.7, t=3, margin_type='C'):
        super(SphereFace2, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scale = scale
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.bias = nn.Parameter(torch.zeros(1, 1))
        self.t = t
        self.lanbuda = lanbuda
        self.margin_type = margin_type
        self.margin = margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin)
        self.mmm = 1.0 + math.cos(math.pi - margin)

    def update(self, margin=0.2):
        self.margin = margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin)
        self.mmm = 1.0 + math.cos(math.pi - margin)

    def fun_g(self, z, t: 'int'):
        gz = 2 * torch.pow((z + 1) / 2, t) - 1
        return gz

    def forward(self, input, label):
        cos = F.linear(F.normalize(input), F.normalize(self.weight))
        if self.margin_type == 'A':
            sin = torch.sqrt(1.0 - torch.pow(cos, 2))
            cos_m_theta_p = self.scale * self.fun_g(torch.where(cos > self.th, cos * self.cos_m - sin * self.sin_m, cos - self.mmm), self.t) + self.bias[0][0]
            cos_m_theta_n = self.scale * self.fun_g(cos * self.cos_m + sin * self.sin_m, self.t) + self.bias[0][0]
            cos_p_theta = self.lanbuda * torch.log(1 + torch.exp(-1.0 * cos_m_theta_p))
            cos_n_theta = (1 - self.lanbuda) * torch.log(1 + torch.exp(cos_m_theta_n))
        else:
            cos_m_theta_p = self.scale * (self.fun_g(cos, self.t) - self.margin) + self.bias[0][0]
            cos_m_theta_n = self.scale * (self.fun_g(cos, self.t) + self.margin) + self.bias[0][0]
            cos_p_theta = self.lanbuda * torch.log(1 + torch.exp(-1.0 * cos_m_theta_p))
            cos_n_theta = (1 - self.lanbuda) * torch.log(1 + torch.exp(cos_m_theta_n))
        target_mask = input.new_zeros(cos.size())
        target_mask.scatter_(1, label.view(-1, 1).long(), 1.0)
        nontarget_mask = 1 - target_mask
        cos1 = (cos - self.margin) * target_mask + cos * nontarget_mask
        output = self.scale * cos1
        loss = (target_mask * cos_p_theta + nontarget_mask * cos_n_theta).sum(1).mean()
        return output, loss

    def extra_repr(self):
        return """in_features={}, out_features={}, scale={}, lanbuda={},
                  margin={}, t={}, margin_type={}""".format(self.in_features, self.out_features, self.scale, self.lanbuda, self.margin, self.t, self.margin_type)


class ArcMarginProduct(nn.Module):
    """Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            scale: norm of input feature
            margin: margin
            cos(theta + margin)
        """

    def __init__(self, in_features, out_features, scale=32.0, margin=0.2, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scale = scale
        self.margin = margin
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.easy_margin = easy_margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin
        self.mmm = 1.0 + math.cos(math.pi - margin)
        self.m = self.margin

    def update(self, margin=0.2):
        self.margin = margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin
        self.m = self.margin
        self.mmm = 1.0 + math.cos(math.pi - margin)

    def forward(self, input, label):
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mmm)
        one_hot = input.new_zeros(cosine.size())
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = one_hot * phi + (1.0 - one_hot) * cosine
        output *= self.scale
        return output

    def extra_repr(self):
        return """in_features={}, out_features={}, scale={},
                  margin={}, easy_margin={}""".format(self.in_features, self.out_features, self.scale, self.margin, self.easy_margin)


class ArcMarginProduct_intertopk_subcenter(nn.Module):
    """Implement of large margin arc distance with intertopk and subcenter:
        Reference:
            MULTI-QUERY MULTI-HEAD ATTENTION POOLING AND INTER-TOPK PENALTY
            FOR SPEAKER VERIFICATION.
            https://arxiv.org/pdf/2110.05042.pdf
            Sub-center ArcFace: Boosting Face Recognition by
            Large-Scale Noisy Web Faces.
            https://ibug.doc.ic.ac.uk/media/uploads/documents/eccv_1445.pdf
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            scale: norm of input feature
            margin: margin
            cos(theta + margin)
            K: number of sub-centers
            k_top: number of hard samples
            mp: margin penalty of hard samples
            do_lm: whether do large margin finetune
        """

    def __init__(self, in_features, out_features, scale=32.0, margin=0.2, easy_margin=False, K=3, mp=0.06, k_top=5, do_lm=False):
        super(ArcMarginProduct_intertopk_subcenter, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scale = scale
        self.margin = margin
        self.do_lm = do_lm
        self.K = K
        if do_lm:
            self.mp = 0.0
            self.k_top = 0
        else:
            self.mp = mp
            self.k_top = k_top
        self.weight = nn.Parameter(torch.FloatTensor(self.K * out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.easy_margin = easy_margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin
        self.mmm = 1.0 + math.cos(math.pi - margin)
        self.m = self.margin
        self.cos_mp = math.cos(0.0)
        self.sin_mp = math.sin(0.0)

    def update(self, margin=0.2):
        self.margin = margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin
        self.m = self.margin
        self.mmm = 1.0 + math.cos(math.pi - margin)
        if margin > 0.001:
            mp = self.mp * (margin / 0.2)
        else:
            mp = 0.0
        self.cos_mp = math.cos(mp)
        self.sin_mp = math.sin(mp)

    def forward(self, input, label):
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        cosine = torch.reshape(cosine, (-1, self.out_features, self.K))
        cosine, _ = torch.max(cosine, 2)
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi_mp = cosine * self.cos_mp + sine * self.sin_mp
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mmm)
        one_hot = input.new_zeros(cosine.size())
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        if self.k_top > 0:
            _, top_k_index = torch.topk(cosine - 2 * one_hot, self.k_top)
            top_k_one_hot = input.new_zeros(cosine.size()).scatter_(1, top_k_index, 1)
            output = one_hot * phi + top_k_one_hot * phi_mp + (1.0 - one_hot - top_k_one_hot) * cosine
        else:
            output = one_hot * phi + (1.0 - one_hot) * cosine
        output *= self.scale
        return output

    def extra_repr(self):
        return 'in_features={}, out_features={}, scale={}, margin={}, easy_margin={},K={}, mp={}, k_top={}, do_lm={}'.format(self.in_features, self.out_features, self.scale, self.margin, self.easy_margin, self.K, self.mp, self.k_top, self.do_lm)


class AddMarginProduct(nn.Module):
    """Implement of large margin cosine distance: :
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        scale: norm of input feature
        margin: margin
        cos(theta) - margin
    """

    def __init__(self, in_features, out_features, scale=32.0, margin=0.2):
        super(AddMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scale = scale
        self.margin = margin
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def update(self, margin):
        self.margin = margin

    def forward(self, input, label):
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        phi = cosine - self.margin
        one_hot = input.new_zeros(cosine.size())
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = one_hot * phi + (1.0 - one_hot) * cosine
        output *= self.scale
        return output

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'in_features=' + str(self.in_features) + ', out_features=' + str(self.out_features) + ', scale=' + str(self.scale) + ', margin=' + str(self.margin) + ')'


class SphereProduct(nn.Module):
    """Implement of large margin cosine distance: :
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        margin: margin
        cos(margin * theta)
    """

    def __init__(self, in_features, out_features, margin=2):
        super(SphereProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.margin = margin
        self.base = 1000.0
        self.gamma = 0.12
        self.power = 1
        self.LambdaMin = 5.0
        self.iter = 0
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform(self.weight)
        self.mlambda = [lambda x: x ** 0, lambda x: x ** 1, lambda x: 2 * x ** 2 - 1, lambda x: 4 * x ** 3 - 3 * x, lambda x: 8 * x ** 4 - 8 * x ** 2 + 1, lambda x: 16 * x ** 5 - 20 * x ** 3 + 5 * x]
        assert self.margin < 6

    def forward(self, input, label):
        self.iter += 1
        self.lamb = max(self.LambdaMin, self.base * (1 + self.gamma * self.iter) ** (-1 * self.power))
        cos_theta = F.linear(F.normalize(input), F.normalize(self.weight))
        cos_theta = cos_theta.clamp(-1, 1)
        cos_m_theta = self.mlambda[self.margin](cos_theta)
        theta = cos_theta.data.acos()
        k = (self.margin * theta / 3.14159265).floor()
        phi_theta = (-1.0) ** k * cos_m_theta - 2 * k
        NormOfFeature = torch.norm(input, 2, 1)
        one_hot = input.new_zeros(cos_theta.size())
        one_hot.scatter_(1, label.view(-1, 1), 1)
        output = one_hot * (phi_theta - cos_theta) / (1 + self.lamb) + cos_theta
        output *= NormOfFeature.view(-1, 1)
        return output

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'in_features=' + str(self.in_features) + ', out_features=' + str(self.out_features) + ', margin=' + str(self.margin) + ')'


class to1d(nn.Module):

    def forward(self, x):
        size = x.size()
        bs, c, f, t = tuple(size)
        return x.permute((0, 2, 1, 3)).reshape((bs, c * f, t))


class NewGELUActivation(nn.Module):

    def forward(self, input):
        return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))


class GRU(nn.Module):

    def __init__(self, *args, **kwargs):
        super(GRU, self).__init__()
        self.gru = nn.GRU(*args, **kwargs)

    def forward(self, x):
        return self.gru(x.permute((0, 2, 1)))[0].permute((0, 2, 1))


class PosEncConv(nn.Module):

    def __init__(self, C, ks, groups=None):
        super().__init__()
        assert ks % 2 == 1
        self.conv = nn.Conv1d(C, C, ks, padding=ks // 2, groups=C if groups is None else groups)
        self.norm = LayerNorm(C, eps=1e-06, data_format='channels_first')

    def forward(self, x):
        return x + self.norm(self.conv(x))


BatchNormNd = {(1): nn.BatchNorm1d, (2): nn.BatchNorm2d}


ConvNd = {(1): nn.Conv1d, (2): nn.Conv2d}


class ConvNeXtLikeBlock(nn.Module):

    def __init__(self, C, dim=2, kernel_sizes=((3, 3),), group_divisor=1, padding='same'):
        super().__init__()
        self.dwconvs = nn.ModuleList(modules=[ConvNd[dim](C, C, kernel_size=ks, padding=padding, groups=C // group_divisor if group_divisor is not None else 1) for ks in kernel_sizes])
        self.norm = BatchNormNd[dim](C * len(kernel_sizes))
        self.gelu = nn.GELU()
        self.pwconv1 = ConvNd[dim](C * len(kernel_sizes), C, 1)

    def forward(self, x):
        skip = x
        x = torch.cat([dwconv(x) for dwconv in self.dwconvs], dim=1)
        x = self.gelu(self.norm(x))
        x = self.pwconv1(x)
        x = skip + x
        return x


class fwSEBlock(nn.Module):
    """
    Squeeze-and-Excitation block
    link: https://arxiv.org/pdf/1709.01507.pdf
    PyTorch implementation
    """

    def __init__(self, num_freq, num_feats=64):
        super(fwSEBlock, self).__init__()
        self.squeeze = nn.Linear(num_freq, num_feats)
        self.exitation = nn.Linear(num_feats, num_freq)
        self.activation = nn.ReLU()

    def forward(self, inputs):
        x = torch.mean(inputs, dim=[1, 3])
        x = self.squeeze(x)
        x = self.activation(x)
        x = self.exitation(x)
        x = torch.sigmoid(x)
        x = x[:, None, :, None]
        x = inputs * x
        return x


class ResBasicBlock(nn.Module):

    def __init__(self, in_planes, planes, num_freq, stride=1, se_channels=64, group_divisor=4, use_fwSE=False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, in_planes if group_divisor is not None else planes, kernel_size=3, stride=stride, padding=1, bias=False, groups=in_planes // group_divisor if group_divisor is not None else 1)
        if group_divisor is not None:
            self.conv1pw = nn.Conv2d(in_planes, planes, 1)
        else:
            self.conv1pw = nn.Identity()
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False, groups=planes // group_divisor if group_divisor is not None else 1)
        if group_divisor is not None:
            self.conv2pw = nn.Conv2d(planes, planes, 1)
        else:
            self.conv2pw = nn.Identity()
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        if use_fwSE:
            self.se = fwSEBlock(num_freq, se_channels)
        else:
            self.se = nn.Identity()
        if planes != in_planes:
            self.downsample = nn.Sequential(nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes))
        else:
            self.downsample = nn.Identity()

    def forward(self, x):
        residual = x
        out = self.conv1pw(self.conv1(x))
        out = self.relu(out)
        out = self.bn1(out)
        out = self.conv2pw(self.conv2(out))
        out = self.bn2(out)
        out = self.se(out)
        out += self.downsample(residual)
        out = self.relu(out)
        return out


class ConvBlock2d(nn.Module):

    def __init__(self, c, f, block_type='convnext_like', group_divisor=1):
        super().__init__()
        if block_type == 'convnext_like':
            self.conv_block = ConvNeXtLikeBlock(c, dim=2, kernel_sizes=[(3, 3)], group_divisor=group_divisor, padding='same')
        elif block_type == 'basic_resnet':
            self.conv_block = ResBasicBlock(c, c, f, stride=1, se_channels=min(64, max(c, 32)), group_divisor=group_divisor, use_fwSE=False)
        elif block_type == 'basic_resnet_fwse':
            self.conv_block = ResBasicBlock(c, c, f, stride=1, se_channels=min(64, max(c, 32)), group_divisor=group_divisor, use_fwSE=True)
        else:
            raise NotImplementedError()

    def forward(self, x):
        return self.conv_block(x)


class FeedForward(nn.Module):

    def __init__(self, hidden_size, intermediate_size, activation_dropout=0.0, hidden_dropout=0.0):
        super().__init__()
        self.intermediate_dropout = nn.Dropout(activation_dropout)
        self.intermediate_dense = nn.Linear(hidden_size, intermediate_size)
        self.intermediate_act_fn = NewGELUActivation()
        self.output_dense = nn.Linear(intermediate_size, hidden_size)
        self.output_dropout = nn.Dropout(hidden_dropout)

    def forward(self, hidden_states):
        hidden_states = self.intermediate_dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.intermediate_dropout(hidden_states)
        hidden_states = self.output_dense(hidden_states)
        hidden_states = self.output_dropout(hidden_states)
        return hidden_states


class TransformerEncoderLayer(nn.Module):

    def __init__(self, n_state, n_mlp, n_head, channel_last=False, act_do=0.0, att_do=0.0, hid_do=0.0, ln_eps=1e-06):
        hidden_size = n_state
        num_attention_heads = n_head
        intermediate_size = n_mlp
        activation_dropout = act_do
        attention_dropout = att_do
        hidden_dropout = hid_do
        layer_norm_eps = ln_eps
        super().__init__()
        self.channel_last = channel_last
        self.attention = MultiHeadAttention(embed_dim=hidden_size, num_heads=num_attention_heads, dropout=attention_dropout)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.feed_forward = FeedForward(hidden_size=hidden_size, intermediate_size=intermediate_size, activation_dropout=activation_dropout, hidden_dropout=hidden_dropout)
        self.final_layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

    def forward(self, hidden_states):
        if not self.channel_last:
            hidden_states = hidden_states.permute(0, 2, 1)
        attn_residual = hidden_states
        hidden_states = self.attention(hidden_states)
        hidden_states = attn_residual + hidden_states
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = hidden_states + self.feed_forward(hidden_states)
        hidden_states = self.final_layer_norm(hidden_states)
        outputs = hidden_states
        if not self.channel_last:
            outputs = outputs.permute(0, 2, 1)
        return outputs


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(self.expansion * planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class TimeContextBlock1d(nn.Module):
    """ """

    def __init__(self, C, hC, pos_ker_sz=59, block_type='att'):
        super().__init__()
        assert pos_ker_sz
        self.red_dim_conv = nn.Sequential(nn.Conv1d(C, hC, 1), LayerNorm(hC, eps=1e-06, data_format='channels_first'))
        if block_type == 'fc':
            self.tcm = nn.Sequential(nn.Conv1d(hC, hC * 2, 1), LayerNorm(hC * 2, eps=1e-06, data_format='channels_first'), nn.GELU(), nn.Conv1d(hC * 2, hC, 1))
        elif block_type == 'gru':
            self.tcm = nn.Sequential(GRU(input_size=hC, hidden_size=hC, num_layers=1, bias=True, batch_first=False, dropout=0.0, bidirectional=True), nn.Conv1d(2 * hC, hC, 1))
        elif block_type == 'att':
            self.tcm = nn.Sequential(PosEncConv(hC, ks=pos_ker_sz, groups=hC), TransformerEncoderLayer(n_state=hC, n_mlp=hC * 2, n_head=4))
        elif block_type == 'conv+att':
            self.tcm = nn.Sequential(ConvNeXtLikeBlock(hC, dim=1, kernel_sizes=[7], group_divisor=1, padding='same'), ConvNeXtLikeBlock(hC, dim=1, kernel_sizes=[19], group_divisor=1, padding='same'), ConvNeXtLikeBlock(hC, dim=1, kernel_sizes=[31], group_divisor=1, padding='same'), ConvNeXtLikeBlock(hC, dim=1, kernel_sizes=[59], group_divisor=1, padding='same'), TransformerEncoderLayer(n_state=hC, n_mlp=hC, n_head=4))
        else:
            raise NotImplementedError()
        self.exp_dim_conv = nn.Conv1d(hC, C, 1)

    def forward(self, x):
        skip = x
        x = self.red_dim_conv(x)
        x = self.tcm(x)
        x = self.exp_dim_conv(x)
        return skip + x


class ReDimNetBone(nn.Module):

    def __init__(self, F=72, C=16, block_1d_type='conv+att', block_2d_type='basic_resnet', stages_setup=((1, 2, 1, [(3, 3)], None), (2, 3, 1, [(3, 3)], None), (3, 4, 1, [(3, 3)], 8), (2, 5, 1, [(3, 3)], 8), (1, 5, 1, [(7, 1)], 8), (2, 3, 1, [(3, 3)], 8)), group_divisor=1, out_channels=512):
        super().__init__()
        self.F = F
        self.C = C
        self.block_1d_type = block_1d_type
        self.block_2d_type = block_2d_type
        self.stages_setup = stages_setup
        self.build(stages_setup, group_divisor, out_channels)

    def build(self, stages_setup, group_divisor, out_channels):
        self.num_stages = len(stages_setup)
        cur_c = self.C
        cur_f = self.F
        self.inputs_weights = torch.nn.ParameterList([nn.Parameter(torch.ones(1, 1, 1, 1), requires_grad=False)] + [nn.Parameter(torch.zeros(1, num_inputs + 1, self.C * self.F, 1), requires_grad=True) for num_inputs in range(1, len(stages_setup) + 1)])
        self.stem = nn.Sequential(nn.Conv2d(1, int(cur_c), kernel_size=3, stride=1, padding='same'), LayerNorm(int(cur_c), eps=1e-06, data_format='channels_first'))
        Block1d = functools.partial(TimeContextBlock1d, block_type=self.block_1d_type)
        Block2d = functools.partial(ConvBlock2d, block_type=self.block_2d_type)
        self.stages_cfs = []
        for stage_ind, (stride, num_blocks, conv_exp, kernel_sizes, att_block_red) in enumerate(stages_setup):
            assert stride in [1, 2, 3]
            layers = [nn.Conv2d(int(cur_c), int(stride * cur_c * conv_exp), kernel_size=(stride, 1), stride=(stride, 1), padding=0, groups=1)]
            self.stages_cfs.append((cur_c, cur_f))
            cur_c = stride * cur_c
            assert cur_f % stride == 0
            cur_f = cur_f // stride
            for _ in range(num_blocks):
                layers.append(Block2d(c=int(cur_c * conv_exp), f=cur_f, group_divisor=group_divisor))
            if conv_exp != 1:
                _group_divisor = group_divisor
                layers.append(nn.Sequential(nn.Conv2d(int(cur_c * conv_exp), cur_c, kernel_size=(3, 3), stride=1, padding='same', groups=cur_c // _group_divisor if _group_divisor is not None else 1), nn.BatchNorm2d(cur_c, eps=1e-06), nn.GELU(), nn.Conv2d(cur_c, cur_c, 1)))
            layers.append(to1d())
            if att_block_red is not None:
                layers.append(Block1d(self.C * self.F, hC=self.C * self.F // att_block_red))
            setattr(self, f'stage{stage_ind}', nn.Sequential(*layers))
        if out_channels is not None:
            self.mfa = nn.Sequential(nn.Conv1d(self.F * self.C, out_channels, kernel_size=1, padding='same'), nn.BatchNorm1d(out_channels, affine=True))
        else:
            self.mfa = nn.Identity()

    def to1d(self, x):
        size = x.size()
        bs, c, f, t = tuple(size)
        return x.permute((0, 2, 1, 3)).reshape((bs, c * f, t))

    def to2d(self, x, c, f):
        size = x.size()
        bs, cf, t = tuple(size)
        return x.reshape((bs, f, c, t)).permute((0, 2, 1, 3))

    def weigth1d(self, outs_1d, i):
        xs = torch.cat([t.unsqueeze(1) for t in outs_1d], dim=1)
        w = F.softmax(self.inputs_weights[i], dim=1)
        x = (w * xs).sum(dim=1)
        return x

    def run_stage(self, prev_outs_1d, stage_ind):
        stage = getattr(self, f'stage{stage_ind}')
        c, f = self.stages_cfs[stage_ind]
        x = self.weigth1d(prev_outs_1d, stage_ind)
        x = self.to2d(x, c, f)
        x = stage(x)
        return x

    def forward(self, inp):
        x = self.stem(inp)
        outputs_1d = [self.to1d(x)]
        for stage_ind in range(self.num_stages):
            outputs_1d.append(self.run_stage(outputs_1d, stage_ind))
        x = self.weigth1d(outputs_1d, -1)
        x = self.mfa(x)
        return x


class ReDimNet(nn.Module):

    def __init__(self, feat_dim=72, C=16, block_1d_type='conv+att', block_2d_type='basic_resnet', stages_setup=((1, 2, 1, [(3, 3)], 12), (2, 2, 1, [(3, 3)], 12), (1, 3, 1, [(3, 3)], 12), (2, 4, 1, [(3, 3)], 8), (1, 4, 1, [(3, 3)], 8), (2, 4, 1, [(3, 3)], 4)), group_divisor=4, out_channels=None, embed_dim=192, pooling_func='ASTP', global_context_att=True, two_emb_layer=False):
        super().__init__()
        self.two_emb_layer = two_emb_layer
        self.backbone = ReDimNetBone(feat_dim, C, block_1d_type, block_2d_type, stages_setup, group_divisor, out_channels)
        if out_channels is None:
            out_channels = C * feat_dim
        self.pool = getattr(pooling_layers, pooling_func)(in_dim=out_channels, global_context_att=global_context_att)
        self.pool_out_dim = self.pool.get_out_dim()
        self.seg_1 = nn.Linear(self.pool_out_dim, embed_dim)
        if self.two_emb_layer:
            self.seg_bn_1 = nn.BatchNorm1d(embed_dim, affine=False)
            self.seg_2 = nn.Linear(embed_dim, embed_dim)
        else:
            self.seg_bn_1 = nn.Identity()
            self.seg_2 = nn.Identity()

    def _get_frame_level_feat(self, x):
        x = x.permute(0, 2, 1)
        x = x.unsqueeze_(1)
        out = self.backbone(x)
        return out

    def get_frame_level_feat(self, x):
        out = self._get_frame_level_feat(x).permute(0, 2, 1)
        return out

    def forward(self, x):
        out = self._get_frame_level_feat(x)
        stats = self.pool(out)
        embed_a = self.seg_1(stats)
        if self.two_emb_layer:
            out = F.relu(embed_a)
            out = self.seg_bn_1(out)
            embed_b = self.seg_2(out)
            return embed_a, embed_b
        else:
            return torch.tensor(0.0), embed_a


class SEBlock_2D(torch.nn.Module):
    """ A SE Block layer layer which can learn to use global information to
    selectively emphasise informative features and suppress less useful ones.
    This is a pytorch implementation of SE Block based on the paper:
    Squeeze-and-Excitation Networks
    by JFChou xmuspeech 2019-07-13
        leo 2020-12-20 [Check and update]
        """

    def __init__(self, in_planes, ratio=16, inplace=True):
        """
        @ratio: a reduction ratio which allows us to vary the capacity
        and computational cost of the SE blocks
        in the network.
        """
        super(SEBlock_2D, self).__init__()
        self.in_planes = in_planes
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.fc_1 = torch.nn.Linear(in_planes, in_planes // ratio)
        self.relu = torch.nn.ReLU(inplace=inplace)
        self.fc_2 = torch.nn.Linear(in_planes // ratio, in_planes)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, inputs):
        """
        @inputs: a 3-dimensional tensor (a batch),
                 including [samples-index, frames-dim-index, frames-index]
        """
        assert len(inputs.shape) == 4
        assert inputs.shape[1] == self.in_planes
        b, c, _, _ = inputs.size()
        x = self.avg_pool(inputs).view(b, c)
        x = self.fc_1(x)
        x = self.relu(x)
        x = self.fc_2(x)
        x = self.sigmoid(x)
        scale = x.view(b, c, 1, 1)
        return inputs * scale


def conv_bn(in_channels, out_channels, kernel_size, stride, padding, dilation=1, groups=1):
    result = nn.Sequential()
    result.add_module('conv', nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=False))
    result.add_module('bn', nn.BatchNorm2d(num_features=out_channels))
    return result


class RepVGGBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros', deploy=False, use_se=False):
        super(RepVGGBlock, self).__init__()
        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels
        assert kernel_size == 3
        assert padding == 1
        padding_11 = padding - kernel_size // 2
        self.nonlinearity = nn.ReLU(inplace=True)
        if use_se:
            self.se = SEBlock_2D(out_channels, 4)
        else:
            self.se = nn.Identity()
        self.rbr_reparam = None
        self.rbr_identity = None
        self.rbr_dense = None
        self.rbr_1x1 = None
        if deploy:
            self.rbr_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=True, padding_mode=padding_mode)
        else:
            self.rbr_identity = nn.BatchNorm2d(num_features=in_channels) if out_channels == in_channels and stride == 1 else None
            self.rbr_dense = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups)
            self.rbr_1x1 = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, padding=padding_11, groups=groups)

    def forward(self, inputs):
        if self.deploy and self.rbr_reparam is not None:
            return self.se(self.nonlinearity(self.rbr_reparam(inputs)))
        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)
        if self.rbr_dense is not None and self.rbr_1x1 is not None:
            return self.se(self.nonlinearity(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out))
        else:
            raise TypeError("It's a training repvgg structure but branch conv not exits.")

    def get_custom_L2(self):
        K3 = self.rbr_dense.conv.weight
        K1 = self.rbr_1x1.conv.weight
        t3 = (self.rbr_dense.bn.weight / (self.rbr_dense.bn.running_var + self.rbr_dense.bn.eps).sqrt()).reshape(-1, 1, 1, 1).detach()
        t1 = (self.rbr_1x1.bn.weight / (self.rbr_1x1.bn.running_var + self.rbr_1x1.bn.eps).sqrt()).reshape(-1, 1, 1, 1).detach()
        l2_loss_circle = (K3 ** 2).sum() - (K3[:, :, 1:2, 1:2] ** 2).sum()
        eq_kernel = K3[:, :, 1:2, 1:2] * t3 + K1 * t1
        l2_loss_eq_kernel = (eq_kernel ** 2 / (t3 ** 2 + t1 ** 2)).sum()
        return l2_loss_eq_kernel + l2_loss_circle

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros((self.in_channels, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def switch_to_deploy(self):
        if self.rbr_reparam is not None:
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv2d(in_channels=self.rbr_dense.conv.in_channels, out_channels=self.rbr_dense.conv.out_channels, kernel_size=self.rbr_dense.conv.kernel_size, stride=self.rbr_dense.conv.stride, padding=self.rbr_dense.conv.padding, dilation=self.rbr_dense.conv.dilation, groups=self.rbr_dense.conv.groups, bias=True)
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.rbr_dense = None
        self.rbr_1x1 = None
        if hasattr(self, 'rbr_identity'):
            self.rbr_identity = None
        if hasattr(self, 'id_tensor'):
            self.__delattr__('id_tensor')
        self.deploy = True


class RepSPKBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, branch_dilation=2, groups=1, padding_mode='zeros', deploy=False, use_se=False):
        super(RepSPKBlock, self).__init__()
        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels
        assert kernel_size == 3
        assert padding == 1
        assert dilation == 1
        assert branch_dilation == 2
        self.branch_dilation = branch_dilation
        self.depoly_kernel_size = (kernel_size - 1) * (branch_dilation - 1) + kernel_size
        self.nonlinearity = nn.ReLU(inplace=True)
        if use_se:
            self.se = SEBlock_2D(out_channels, 4)
        else:
            self.se = nn.Identity()
        self.rbr_reparam = None
        self.rbr_identity = None
        self.rbr_dense = None
        self.rbr_dense_dilation = None
        if deploy:
            self.rbr_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=self.depoly_kernel_size, stride=stride, padding=self.branch_dilation, groups=groups, bias=True, padding_mode=padding_mode)
        else:
            self.rbr_identity = nn.BatchNorm2d(num_features=in_channels) if out_channels == in_channels and stride == 1 else None
            self.rbr_dense = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups)
            self.rbr_dense_dilation = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=self.branch_dilation, dilation=self.branch_dilation, groups=groups)

    def forward(self, inputs):
        if self.deploy and self.rbr_reparam is not None:
            return self.se(self.nonlinearity(self.rbr_reparam(inputs)))
        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)
        if self.rbr_dense is not None and self.rbr_dense_dilation is not None:
            return self.se(self.nonlinearity(self.rbr_dense(inputs) + self.rbr_dense_dilation(inputs) + id_out))
        else:
            raise TypeError("It's a training repvgg structure but branch conv not exits.")

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel_dilation_branch, bias_dilation_branch = self._fuse_bn_tensor(self.rbr_dense_dilation)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        return self._convert_3x3_dilation_to_5x5_tensor(kernel_dilation_branch) + self._pad_3x3_to_5x5_tensor(kernel3x3) + kernelid, bias3x3 + bias_dilation_branch + biasid

    def _pad_3x3_to_5x5_tensor(self, kernel3x3):
        if kernel3x3 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel3x3, [1, 1, 1, 1])

    def _convert_3x3_dilation_to_5x5_tensor(self, kernel3x3):
        if kernel3x3 is None:
            return 0
        else:
            kernel_value = torch.zeros((kernel3x3.size(0), kernel3x3.size(1), 5, 5), dtype=kernel3x3.dtype)
            kernel_value[:, :, ::2, ::2] = kernel3x3
            return kernel_value

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros((self.in_channels, input_dim, 5, 5), dtype=np.float32)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 2, 2] = 1
                self.id_tensor = torch.from_numpy(kernel_value)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def switch_to_deploy(self):
        if self.rbr_reparam is not None:
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv2d(in_channels=self.rbr_dense.conv.in_channels, out_channels=self.rbr_dense.conv.out_channels, kernel_size=self.depoly_kernel_size, stride=self.rbr_dense.conv.stride, padding=self.branch_dilation, dilation=self.rbr_dense.conv.dilation, groups=self.rbr_dense.conv.groups, bias=True)
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.rbr_dense = None
        self.rbr_dense_dilation = None
        if hasattr(self, 'rbr_identity'):
            self.rbr_identity = None
        if hasattr(self, 'id_tensor'):
            self.__delattr__('id_tensor')
        self.deploy = True


class RepVGG(nn.Module):

    def __init__(self, head_inplanes=1, block='RepVGG', num_blocks=None, strides=None, base_width=64, width_multiplier=None, override_groups_map=None, deploy=False, use_se=False, pooling_func='ASTP', feat_dim=80, embed_dim=256):
        super(RepVGG, self).__init__()
        assert len(width_multiplier) == 4
        assert len(num_blocks) == 4
        assert len(strides) == 5
        width_multiplier = [(w * (base_width / 64.0)) for w in width_multiplier]
        self.deploy = deploy
        self.override_groups_map = override_groups_map or dict()
        self.use_se = use_se
        self.downsample_multiple = 1
        if block == 'RepVGG':
            used_block = RepVGGBlock
        elif block == 'RepSPK':
            used_block = RepSPKBlock
        else:
            raise TypeError('Do not support {} block.'.format(block))
        for s in strides:
            self.downsample_multiple *= s
        assert 0 not in self.override_groups_map
        self.in_planes = min(64, int(64 * width_multiplier[0]))
        self.stage0 = used_block(head_inplanes, out_channels=self.in_planes, kernel_size=3, stride=strides[0], padding=1, deploy=self.deploy, use_se=self.use_se)
        self.cur_layer_idx = 1
        self.stage1 = self._make_stage(used_block, int(64 * width_multiplier[0]), num_blocks[0], stride=strides[1])
        self.stage2 = self._make_stage(used_block, int(128 * width_multiplier[1]), num_blocks[1], stride=strides[2])
        self.stage3 = self._make_stage(used_block, int(256 * width_multiplier[2]), num_blocks[2], stride=strides[3])
        self.stage4 = self._make_stage(used_block, int(512 * width_multiplier[3]), num_blocks[3], stride=strides[4])
        self.output_planes = self.in_planes
        self.stats_dim = self.output_planes * int(feat_dim / 8)
        self.pool = getattr(pooling_layers, pooling_func)(in_dim=self.stats_dim)
        self.pool_out_dim = self.pool.get_out_dim()
        self.seg = nn.Linear(self.pool_out_dim, embed_dim)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.normal_(m.weight, 0.0, 0.01)
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_stage(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        blocks = []
        for stride in strides:
            cur_groups = self.override_groups_map.get(self.cur_layer_idx, 1)
            blocks.append(block(in_channels=self.in_planes, out_channels=planes, kernel_size=3, stride=stride, padding=1, groups=cur_groups, deploy=self.deploy, use_se=self.use_se))
            self.in_planes = planes
            self.cur_layer_idx += 1
        return nn.Sequential(*blocks)

    def get_downsample_multiple(self):
        return self.downsample_multiple

    def get_output_planes(self):
        return self.output_planes

    def _get_frame_level_feat(self, x):
        x = x.permute(0, 2, 1)
        x = x.unsqueeze_(1)
        x = self.stage0(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        return x

    def get_frame_level_feat(self, x):
        out = self._get_frame_level_feat(x)
        out = out.transpose(1, 3)
        out = torch.flatten(out, 2, -1)
        return out

    def forward(self, x):
        x = self._get_frame_level_feat(x)
        stats = self.pool(x)
        embed = self.seg(stats)
        return embed


class BasicBlockRes2Net(nn.Module):
    expansion = 2

    def __init__(self, in_planes, planes, stride=1, baseWidth=32, scale=2):
        super(BasicBlockRes2Net, self).__init__()
        width = int(math.floor(planes * (baseWidth / 64.0)))
        self.conv1 = conv1x1(in_planes, width * scale, stride)
        self.bn1 = nn.BatchNorm2d(width * scale)
        self.nums = scale - 1
        convs = []
        bns = []
        for i in range(self.nums):
            convs.append(conv3x3(width, width))
            bns.append(nn.BatchNorm2d(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)
        self.relu = ReLU(inplace=True)
        self.conv3 = conv1x1(width * scale, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(self.expansion * planes))
        self.stride = stride
        self.width = width
        self.scale = scale

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        spx = torch.split(out, self.width, 1)
        sp = spx[0]
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            if i >= 1:
                sp = sp + spx[i]
            sp = conv(sp)
            sp = self.relu(bn(sp))
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)
        out = torch.cat((out, spx[self.nums]), 1)
        out = self.conv3(out)
        out = self.bn3(out)
        residual = self.shortcut(x)
        out += residual
        out = self.relu(out)
        return out


class Res2Net(nn.Module):

    def __init__(self, m_channels, num_blocks, block=BasicBlockRes2Net, feat_dim=80, embed_dim=192, pooling_func='TSTP', two_emb_layer=False):
        super(Res2Net, self).__init__()
        self.in_planes = m_channels
        self.feat_dim = feat_dim
        self.embed_dim = embed_dim
        self.stats_dim = int(feat_dim / 8) * m_channels * 8
        self.two_emb_layer = two_emb_layer
        self.conv1 = nn.Conv2d(1, m_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(m_channels)
        self.layer1 = self._make_layer(block, m_channels, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, m_channels * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, m_channels * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, m_channels * 8, num_blocks[3], stride=2)
        self.pool = getattr(pooling_layers, pooling_func)(in_dim=self.stats_dim * block.expansion)
        self.pool_out_dim = self.pool.get_out_dim()
        self.seg_1 = nn.Linear(self.pool_out_dim, embed_dim)
        if self.two_emb_layer:
            self.seg_bn_1 = nn.BatchNorm1d(embed_dim, affine=False)
            self.seg_2 = nn.Linear(embed_dim, embed_dim)
        else:
            self.seg_bn_1 = nn.Identity()
            self.seg_2 = nn.Identity()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _get_frame_level_feat(self, x):
        x = x.permute(0, 2, 1)
        x = x.unsqueeze_(1)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        return out

    def get_frame_level_feat(self, x):
        out = self._get_frame_level_feat(x)
        out = out.transpose(1, 3)
        out = torch.flatten(out, 2, -1)
        return out

    def forward(self, x):
        out = self._get_frame_level_feat(x)
        stats = self.pool(out)
        embed_a = self.seg_1(stats)
        if self.two_emb_layer:
            out = F.relu(embed_a)
            out = self.seg_bn_1(out)
            embed_b = self.seg_2(out)
            return embed_b
        else:
            return embed_a


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(self.expansion * planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):

    def __init__(self, in_planes, block, num_blocks, in_ch=1, **kwargs):
        super(ResNet, self).__init__()
        self.in_planes = in_planes
        self.NormLayer = nn.BatchNorm2d
        self.ConvLayer = nn.Conv2d
        self.conv1 = self.ConvLayer(in_ch, in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = self.NormLayer(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, in_planes, num_blocks[0], stride=1, block_id=1)
        self.layer2 = self._make_layer(block, in_planes * 2, num_blocks[1], stride=2, block_id=2)
        self.layer3 = self._make_layer(block, in_planes * 4, num_blocks[2], stride=2, block_id=3)
        self.layer4 = self._make_layer(block, in_planes * 8, num_blocks[3], stride=2, block_id=4)

    def _make_layer(self, block, planes, num_blocks, stride, block_id=1):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.ConvLayer, self.NormLayer, self.in_planes, planes, stride, block_id))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


class SimAMBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, ConvLayer, NormLayer, in_planes, planes, stride=1, block_id=1):
        super(SimAMBasicBlock, self).__init__()
        self.conv1 = ConvLayer(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = NormLayer(planes)
        self.conv2 = ConvLayer(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = NormLayer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.downsample = nn.Sequential(ConvLayer(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False), NormLayer(self.expansion * planes))

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.SimAM(out)
        out += self.downsample(x)
        out = self.relu(out)
        return out

    def SimAM(self, X, lambda_p=0.0001):
        n = X.shape[2] * X.shape[3] - 1
        d = (X - X.mean(dim=[2, 3], keepdim=True)).pow(2)
        v = d.sum(dim=[2, 3], keepdim=True) / n
        E_inv = d / (4 * (v + lambda_p)) + 0.5
        return X * self.sigmoid(E_inv)


def SimAM_ResNet34(in_planes):
    return ResNet(in_planes, SimAMBasicBlock, [3, 4, 6, 3])


class SimAM_ResNet34_ASP(nn.Module):

    def __init__(self, in_planes=64, embed_dim=256, acoustic_dim=80, dropout=0):
        super(SimAM_ResNet34_ASP, self).__init__()
        self.front = SimAM_ResNet34(in_planes)
        self.pooling = pooling_layers.ASP(in_planes, acoustic_dim)
        self.bottleneck = nn.Linear(self.pooling.out_dim, embed_dim)
        self.drop = nn.Dropout(dropout) if dropout else None

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.front(x.unsqueeze(dim=1))
        x = self.pooling(x)
        if self.drop:
            x = self.drop(x)
        x = self.bottleneck(x)
        return x


def SimAM_ResNet100(in_planes):
    return ResNet(in_planes, SimAMBasicBlock, [6, 16, 24, 3])


class SimAM_ResNet100_ASP(nn.Module):

    def __init__(self, in_planes=64, embed_dim=256, acoustic_dim=80, dropout=0):
        super(SimAM_ResNet100_ASP, self).__init__()
        self.front = SimAM_ResNet100(in_planes)
        self.pooling = pooling_layers.ASP(in_planes, acoustic_dim)
        self.bottleneck = nn.Linear(self.pooling.out_dim, embed_dim)
        self.drop = nn.Dropout(dropout) if dropout else None

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.front(x.unsqueeze(dim=1))
        x = self.pooling(x)
        if self.drop:
            x = self.drop(x)
        x = self.bottleneck(x)
        return x


class TdnnLayer(nn.Module):

    def __init__(self, in_dim, out_dim, context_size, dilation=1, padding=0):
        """Define the TDNN layer, essentially 1-D convolution

        Args:
            in_dim (int): input dimension
            out_dim (int): output channels
            context_size (int): context size, essentially the filter size
            dilation (int, optional):  Defaults to 1.
            padding (int, optional):  Defaults to 0.
        """
        super(TdnnLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.context_size = context_size
        self.dilation = dilation
        self.padding = padding
        self.conv_1d = nn.Conv1d(self.in_dim, self.out_dim, self.context_size, dilation=self.dilation, padding=self.padding)
        self.bn = nn.BatchNorm1d(out_dim, affine=False)

    def forward(self, x):
        out = self.conv_1d(x)
        out = F.relu(out)
        out = self.bn(out)
        return out


class XVEC(nn.Module):

    def __init__(self, feat_dim=40, hid_dim=512, stats_dim=1500, embed_dim=512, pooling_func='TSTP'):
        """
        Implementation of Kaldi style xvec, as described in
        X-VECTORS: ROBUST DNN EMBEDDINGS FOR SPEAKER RECOGNITION
        """
        super(XVEC, self).__init__()
        self.feat_dim = feat_dim
        self.stats_dim = stats_dim
        self.embed_dim = embed_dim
        self.frame_1 = TdnnLayer(feat_dim, hid_dim, context_size=5, dilation=1)
        self.frame_2 = TdnnLayer(hid_dim, hid_dim, context_size=3, dilation=2)
        self.frame_3 = TdnnLayer(hid_dim, hid_dim, context_size=3, dilation=3)
        self.frame_4 = TdnnLayer(hid_dim, hid_dim, context_size=1, dilation=1)
        self.frame_5 = TdnnLayer(hid_dim, stats_dim, context_size=1, dilation=1)
        self.pool = getattr(pooling_layers, pooling_func)(in_dim=stats_dim)
        self.pool_out_dim = self.pool.get_out_dim()
        self.seg_1 = nn.Linear(self.pool_out_dim, embed_dim)
        self.seg_bn_1 = nn.BatchNorm1d(embed_dim, affine=False)
        self.seg_2 = nn.Linear(embed_dim, embed_dim)

    def _get_frame_level_feat(self, x):
        x = x.permute(0, 2, 1)
        out = self.frame_1(x)
        out = self.frame_2(out)
        out = self.frame_3(out)
        out = self.frame_4(out)
        out = self.frame_5(out)
        return out

    def get_frame_level_feat(self, x):
        out = self._get_frame_level_feat(x).permute(0, 2, 1)
        return out

    def forward(self, x):
        out = self._get_frame_level_feat(x)
        stats = self.pool(out)
        embed_a = self.seg_1(stats)
        out = F.relu(embed_a)
        out = self.seg_bn_1(out)
        embed_b = self.seg_2(out)
        return embed_a, embed_b


class BatchNorm1d(nn.Module):
    """Applies 1d batch normalization to the input tensor.

    Arguments
    ---------
    input_shape : tuple
        The expected shape of the input. Alternatively, use ``input_size``.
    input_size : int
        The expected size of the input. Alternatively, use ``input_shape``.
    eps : float
        This value is added to std deviation estimation to improve the numerical
        stability.
    momentum : float
        It is a value used for the running_mean and running_var computation.
    affine : bool
        When set to True, the affine parameters are learned.
    track_running_stats : bool
        When set to True, this module tracks the running mean and variance,
        and when set to False, this module does not track such statistics.
    combine_batch_time : bool
        When true, it combines batch an time axis.


    Example
    -------
    >>> input = torch.randn(100, 10)
    >>> norm = BatchNorm1d(input_shape=input.shape)
    >>> output = norm(input)
    >>> output.shape
    torch.Size([100, 10])
    """

    def __init__(self, input_shape=None, input_size=None, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, combine_batch_time=False, skip_transpose=True):
        super().__init__()
        self.combine_batch_time = combine_batch_time
        self.skip_transpose = skip_transpose
        if input_size is None and skip_transpose:
            input_size = input_shape[1]
        elif input_size is None:
            input_size = input_shape[-1]
        self.norm = nn.BatchNorm1d(input_size, eps=eps, momentum=momentum, affine=affine, track_running_stats=track_running_stats)

    def forward(self, x):
        """Returns the normalized input tensor.

        Arguments
        ---------
        x : torch.Tensor (batch, time, [channels])
            input to normalize. 2d or 3d tensors are expected in input
            4d tensors can be used when combine_dims=True.
        """
        shape_or = x.shape
        if self.combine_batch_time:
            if x.ndim == 3:
                x = x.reshape(shape_or[0] * shape_or[1], shape_or[2])
            else:
                x = x.reshape(shape_or[0] * shape_or[1], shape_or[3], shape_or[2])
        elif not self.skip_transpose:
            x = x.transpose(-1, 1)
        x_n = self.norm(x)
        if self.combine_batch_time:
            x_n = x_n.reshape(shape_or)
        elif not self.skip_transpose:
            x_n = x_n.transpose(1, -1)
        return x_n


class whisper_PMFA(torch.nn.Module):

    def __init__(self, output_size=1280, embedding_dim=192, pooling_func='ASTP', global_context_att=True):
        super(whisper_PMFA, self).__init__()
        self.pooling = getattr(pooling_layers, pooling_func)(in_dim=output_size, global_context_att=global_context_att)
        self.bn = BatchNorm1d(input_size=output_size * 2)
        self.fc = torch.nn.Linear(output_size * 2, embedding_dim)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.pooling(x)
        x = x.unsqueeze(-1)
        x = self.bn(x)
        x = x.permute(0, 2, 1)
        x = self.fc(x)
        x = x.squeeze(1)
        return x


def _no_grad_trunc_normal_(tensor, mean, std, a, b):

    def norm_cdf(x):
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0
    if mean < a - 2 * std or mean > b + 2 * std:
        warnings.warn('mean is more than 2 std from [a, b] in nn.init.trunc_normal_. The distribution of values may be incorrect.', stacklevel=2)
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


class DINOHead(nn.Module):

    def __init__(self, in_dim, out_dim, use_bn=False, norm_last_layer=True, nlayers=3, hidden_dim=2048, bottleneck_dim=256, normalize_input=False):
        super().__init__()
        self.normalize_input = normalize_input
        if nlayers == 0:
            self.mlp = nn.Identity()
        elif nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x, return_mlp=False):
        if self.normalize_input:
            x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.mlp(x)
        if return_mlp:
            return x
        x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


class DINOLoss(nn.Module):

    def __init__(self, out_dim, n_scrops, n_tcrops, warmup_teacher_temp, teacher_temp, nepochs, warmup_teacher_temp_epochs_ratio=0.2, student_temp=0.1, center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.n_scrops = n_scrops
        self.n_tcrops = n_tcrops
        self.register_buffer('center', torch.zeros(1, out_dim))
        warmup_teacher_temp_epochs = int(nepochs * warmup_teacher_temp_epochs_ratio)
        self.teacher_temp_schedule = np.concatenate((np.linspace(warmup_teacher_temp, teacher_temp, warmup_teacher_temp_epochs), np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp))
        self.student_entropy = 0.0
        self.teacher_entropy = 0.0

    def forward(self, student_output, teacher_output, epoch, mode=0):
        """
        Cross-entropy between softmax outputs of the
        teacher and student networks.
        """
        student_out = student_output / self.student_temp
        student_tmp = student_out.detach()
        student_out = student_out.chunk(self.n_scrops)
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_tmp = (teacher_output.detach() - self.center) / temp
        teacher_out = teacher_out.detach().chunk(self.n_tcrops)
        student_tmp = F.softmax(student_tmp, dim=1) + 1e-07
        teacher_tmp = F.softmax(teacher_tmp, dim=1) + 1e-07
        self.student_entropy = torch.mean(torch.sum(-student_tmp * torch.log(student_tmp), dim=1)).item()
        self.teacher_entropy = torch.mean(torch.sum(-teacher_tmp * torch.log(teacher_tmp), dim=1)).item()
        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if mode == 0:
                    if v == iq:
                        continue
                elif mode == 1:
                    if v != iq:
                        continue
                elif mode == 2:
                    if v < 2:
                        continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        self.update_center(teacher_output)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        dist.all_reduce(batch_center)
        batch_center = batch_center / (len(teacher_output) * get_world_size())
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)


class DINO(nn.Module):
    """
    https://arxiv.org/abs/2104.14294
    """

    def __init__(self, base_model, dino_head_args, dino_loss_args, sync_bn=True):
        """
        model: the student and teacher base model
        """
        super(DINO, self).__init__()
        self.s_model = base_model
        self.t_model = copy.deepcopy(base_model)
        self.s_model.add_module('projection_head', DINOHead(**dino_head_args))
        self.t_model.add_module('projection_head', DINOHead(**dino_head_args))
        self.t_model.projection_head.load_state_dict(self.s_model.projection_head.state_dict())
        if sync_bn:
            self.s_model = nn.SyncBatchNorm.convert_sync_batchnorm(self.s_model)
            self.t_model = nn.SyncBatchNorm.convert_sync_batchnorm(self.t_model)
        for p in self.t_model.parameters():
            p.requires_grad = False
        self.dino_loss_calculator = DINOLoss(**dino_loss_args)

    @torch.no_grad()
    def ema_update(self, m=0.0):
        for param_q, param_k in zip(self.s_model.parameters(), self.t_model.parameters()):
            param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

    def forward(self, local_feats, global_feats, epoch=0):
        """
        Input:
            local_feats: (chunk_num * B, T, F)
            global_feats: (chunk_num' * B, T, F)
        Output:
            loss: a scalar value
        """
        g_outputs = self.s_model(global_feats)
        l_outputs = self.s_model(local_feats)
        g_output = g_outputs[-1] if isinstance(g_outputs, tuple) else g_outputs
        l_output = l_outputs[-1] if isinstance(l_outputs, tuple) else l_outputs
        s_output = torch.cat([g_output, l_output])
        s_output = self.s_model.projection_head(s_output)
        with torch.no_grad():
            t_outputs = self.t_model(global_feats)
            t_output = t_outputs[-1] if isinstance(t_outputs, tuple) else t_outputs
            t_output = self.t_model.projection_head(t_output)
        loss = self.dino_loss_calculator(s_output, t_output, epoch)
        return loss


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
    output = torch.cat(tensors_gather, dim=0)
    return output


class MoCo(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """

    def __init__(self, encoder, embed_dim=256, K=65536, m=0.999, T=0.07, mlp=False):
        """
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MoCo, self).__init__()
        self.K = K
        self.m = m
        self.T = T
        self.encoder_q = encoder
        self.encoder_k = copy.deepcopy(encoder)
        if mlp:
            self.encoder_q.add_module('mlp', nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.ReLU(), nn.Linear(embed_dim, embed_dim)))
            self.encoder_k.add_module('mlp', nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.ReLU(), nn.Linear(embed_dim, embed_dim)))
        else:
            self.encoder_q.add_module('mlp', nn.Sequential())
            self.encoder_k.add_module('mlp', nn.Sequential())
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
        self.register_buffer('queue', torch.randn(embed_dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        keys = concat_all_gather(keys)
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K
        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]
        num_gpus = batch_size_all // batch_size_this
        idx_shuffle = torch.randperm(batch_size_all)
        torch.distributed.broadcast(idx_shuffle, src=0)
        idx_unshuffle = torch.argsort(idx_shuffle)
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]
        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]
        num_gpus = batch_size_all // batch_size_this
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]
        return x_gather[idx_this]

    def forward(self, input_q, input_k):
        """
        Input:
            input_q: a batch of query inputs
            input_k: a batch of key inputs
        Output:
            logits, targets
        """
        q = self.encoder_q(input_q)
        q = self.encoder_q.mlp(q)
        q = nn.functional.normalize(q, dim=1)
        with torch.no_grad():
            self._momentum_update_key_encoder()
            input_k, idx_unshuffle = self._batch_shuffle_ddp(input_k)
            k = self.encoder_k(input_k)
            k = self.encoder_k.mlp(k)
            k = nn.functional.normalize(k, dim=1)
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= self.T
        labels = torch.zeros(logits.shape[0], dtype=torch.long)
        self._dequeue_and_enqueue(k)
        return logits, labels


class SimCLR(nn.Module):
    """
    https://arxiv.org/pdf/2002.05709.pdf
    """

    def __init__(self, encoder, embed_dim=256, T=0.07, mlp=False, n_views=2):
        """
        T: softmax temperature (default: 0.07)
        n_views: number of views for each sample
        """
        super(SimCLR, self).__init__()
        self.T = T
        self.n_views = n_views
        self.encoder = encoder
        if mlp:
            self.encoder.add_module('mlp', nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.ReLU(), nn.Linear(embed_dim, embed_dim)))
        else:
            self.encoder.add_module('mlp', nn.Sequential())

    def prepare_for_info_nce_loss(self, features):
        """
        Input:
            features: (self.n_views * bs, embed_dim)
        Return:
            logits: (self.n_views * bs, self.n_views * bs - 1)
            labels (torch.long): (self.n_views * bs)

        """
        bs = features.shape[0] // self.n_views
        labels = torch.cat([torch.arange(bs) for _ in range(self.n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels
        features = F.normalize(features, dim=1)
        similarity_matrix = torch.matmul(features, features.T)
        mask = torch.eye(labels.shape[0], dtype=torch.bool)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)
        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long)
        logits = logits / self.T
        return logits, labels

    def forward(self, input_q, input_k):
        """
        Input:
            input_q: a batch of query inputs
            input_k: a batch of key inputs
        Output:
            logits, targets
        """
        combine_input = torch.cat((input_q, input_k), dim=0)
        features = self.encoder(combine_input)
        features = self.encoder.mlp(features)
        logits, labels = self.prepare_for_info_nce_loss(features)
        return logits, labels


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (ASTP,
     lambda: ([], {'in_dim': 4}),
     lambda: ([torch.rand([4, 1, 4, 4])], {})),
    (BasicBlock,
     lambda: ([], {'in_planes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (BasicBlockERes2Net,
     lambda: ([], {'in_planes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (BasicBlockRes2Net,
     lambda: ([], {'in_planes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (BasicResBlock,
     lambda: ([], {'in_planes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Bottleneck,
     lambda: ([], {'in_planes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Conv1d,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4])], {})),
    (Conv1dReluBn,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4])], {})),
    (ConvBlock2d,
     lambda: ([], {'c': 4, 'f': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ConvNeXtLikeBlock,
     lambda: ([], {'C': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (DINOHead,
     lambda: ([], {'in_dim': 4, 'out_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (DenseLayer,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4])], {})),
    (FeedForward,
     lambda: ([], {'hidden_size': 4, 'intermediate_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (GRU,
     lambda: ([], {'input_size': 4, 'hidden_size': 4}),
     lambda: ([torch.rand([4, 4, 4])], {})),
    (Inverted_Bottleneck,
     lambda: ([], {'dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (LayerNorm,
     lambda: ([], {'C': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (LinearModel,
     lambda: ([], {'input_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (MHASTP,
     lambda: ([], {'in_dim': 4}),
     lambda: ([torch.rand([4, 1, 4, 4])], {})),
    (MultiHeadAttention,
     lambda: ([], {'embed_dim': 4, 'num_heads': 4}),
     lambda: ([torch.rand([4, 4, 4])], {})),
    (NewGELUActivation,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ReLU,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Res2Conv1dReluBn,
     lambda: ([], {'channels': 4}),
     lambda: ([torch.rand([4, 4, 4])], {})),
    (ResBasicBlock,
     lambda: ([], {'in_planes': 4, 'planes': 4, 'num_freq': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (SEBlock_2D,
     lambda: ([], {'in_planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (SE_Connect,
     lambda: ([], {'channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (TAP,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (TDNNLayer,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4])], {})),
    (TSDP,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (TSTP,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (TdnnLayer,
     lambda: ([], {'in_dim': 4, 'out_dim': 4, 'context_size': 4}),
     lambda: ([torch.rand([4, 4, 4])], {})),
    (TimeContextBlock1d,
     lambda: ([], {'C': 4, 'hC': 4}),
     lambda: ([torch.rand([4, 4, 4])], {})),
    (TransformerEncoderLayer,
     lambda: ([], {'n_state': 4, 'n_mlp': 4, 'n_head': 4}),
     lambda: ([torch.rand([4, 4, 4])], {})),
    (TransitLayer,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4])], {})),
    (fwSEBlock,
     lambda: ([], {'num_freq': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (to1d,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
]

