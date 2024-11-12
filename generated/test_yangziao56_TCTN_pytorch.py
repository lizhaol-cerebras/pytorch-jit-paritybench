
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


import torch.nn as nn


from torch.nn import init


import math


import copy


import numpy as np


import torch.utils.data as da


from torch.nn.modules.sparse import Embedding


from torch.optim import Adam


import matplotlib as mpl


import matplotlib.pyplot as plt


import torch.nn.functional as F


class DecoderEmbedding(nn.Module):

    def __init__(self, depth=256):
        super(DecoderEmbedding, self).__init__()
        self.conv0 = nn.Sequential(nn.Conv3d(in_channels=16, out_channels=depth, kernel_size=(1, 7, 7), stride=1, padding=(0, 3, 3), bias=True), nn.LeakyReLU(0.2, inplace=True))
        self.conv1 = nn.Sequential(nn.Conv3d(in_channels=depth, out_channels=depth, kernel_size=(1, 5, 5), stride=1, padding=(0, 2, 2), bias=True), nn.LeakyReLU(0.2, inplace=True))
        self.conv2 = nn.Sequential(nn.Conv3d(in_channels=depth, out_channels=depth, kernel_size=(1, 5, 5), stride=1, padding=(0, 2, 2), bias=True), nn.LeakyReLU(0.2, inplace=True))
        self.depth = depth
        self.dropout = nn.Dropout3d(0.1)

    def forward(self, input_img):
        img_ = input_img.permute(0, 2, 1, 3, 4).clone()
        feature_0 = self.conv0(img_)
        feature_1 = self.conv1(feature_0)
        feature_1 = feature_0 + feature_1
        """
        b, c, s, h, w = feature_2.shape
        pos = positional_encoding(s, self.depth, h, w)
        pos = pos.unsqueeze(0).expand(b, -1, -1, -1, -1)
        feature_2 = feature_2 + pos.permute(0, 2, 1, 3, 4)
        """
        out = self.dropout(feature_1).permute(0, 2, 1, 3, 4)
        return out


class FeedForwardNet(nn.Module):

    def __init__(self, depth=128):
        super(FeedForwardNet, self).__init__()
        self.pad = (3 - 1) * 1
        self.conv0 = nn.Sequential(nn.Conv3d(in_channels=depth, out_channels=depth * 3, kernel_size=(3, 3, 3), stride=1, padding=(self.pad, 1, 1), bias=True), nn.LeakyReLU(0.2, inplace=True))
        self.conv1 = nn.Sequential(nn.Conv3d(in_channels=depth * 3, out_channels=depth, kernel_size=(3, 3, 3), stride=1, padding=(self.pad, 1, 1), bias=True))
        self.dropout1 = nn.Dropout3d(0.1)

    def forward(self, input_tensor):
        out = self.conv0(input_tensor.permute(0, 2, 1, 3, 4))
        out = out[:, :, :-self.pad]
        out = self.dropout1(out)
        out = self.conv1(out)
        out = out[:, :, :-self.pad].permute(0, 2, 1, 3, 4)
        return out


class QKVNet(nn.Module):

    def __init__(self, depth=32):
        super(QKVNet, self).__init__()
        self.pad = (3 - 1) * 1
        self.conv0 = nn.Sequential(nn.Conv3d(in_channels=depth, out_channels=depth * 3, kernel_size=(3, 3, 3), stride=1, padding=(self.pad, 1, 1), bias=True))

    def forward(self, input_tensor):
        qkvconcat = self.conv0(input_tensor)
        qkvconcat = qkvconcat[:, :, :-self.pad]
        return qkvconcat


class MultiHeadAttention(nn.Module):

    def __init__(self, head_depth=32, num_heads=4, with_pos=True):
        super(MultiHeadAttention, self).__init__()
        self.depth_perhead = head_depth
        self.num_heads = num_heads
        self.qkv = QKVNet(self.depth_perhead * self.num_heads)
        self.with_pos = with_pos
        self.dropout1 = nn.Dropout3d(0.1)

    def forward(self, input_tensor, pos_decoding, type=0):
        if type == 0:
            batch, seq, channel, height, width = input_tensor.shape
            input_tensor = input_tensor.permute(0, 2, 1, 3, 4)
            qkvconcat = self.qkv(input_tensor)
            q_feature, k_feature, v_feature = torch.split(qkvconcat, self.depth_perhead * self.num_heads, dim=1)
            if self.with_pos:
                q_feature = q_feature + pos_decoding.permute(0, 2, 1, 3, 4)
                k_feature = k_feature + pos_decoding.permute(0, 2, 1, 3, 4)
            q_feature = q_feature.view(batch, self.num_heads, self.depth_perhead, seq, height, width)
            k_feature = k_feature.view(batch, self.num_heads, self.depth_perhead, seq, height, width)
            v_feature = v_feature.view(batch, self.num_heads, self.depth_perhead, seq, height, width)
            q = q_feature.permute(0, 4, 5, 1, 3, 2)
            k = k_feature.permute(0, 4, 5, 1, 2, 3)
            v = v_feature.permute(0, 4, 5, 1, 3, 2)
            attention_map = torch.matmul(q, k) / math.sqrt(self.depth_perhead)
            """
            #distribution
            s_q = np.arange(seq)[:, np.newaxis]
            s_k = np.arange(seq)[np.newaxis, :]
            GD = np.exp(-(s_k-s_q)*(s_k-s_q)/2)/seq
            #GD = -(s_k-s_q)*(s_k-s_q)/(seq*seq*seq)
            GD = torch.from_numpy(GD).unsqueeze(0).expand(batch*height*width*self.num_heads, -1, -1).view(batch, height, width, self.num_heads, seq, seq)
            GD = torch.tensor(GD, dtype=torch.float32).cuda()
            attention_map = attention_map + GD
            """
            mask = 1 - torch.triu(torch.ones((seq, seq)), diagonal=1)
            mask = mask.unsqueeze(0).expand(batch * height * width * self.num_heads, -1, -1).view(batch, height, width, self.num_heads, seq, seq)
            attention_map = attention_map * mask
            attention_map = attention_map.masked_fill(attention_map == 0, -1000000000.0)
            attention_map = nn.Softmax(dim=-1)(attention_map)
            attention_map_ = attention_map.permute(0, 3, 5, 4, 1, 2).contiguous().view(batch, -1, seq, height, width)
            attention_map_ = self.dropout1(attention_map_)
            attention_map = attention_map_.view(batch, self.num_heads, seq, seq, height, width).permute(0, 4, 5, 1, 3, 2)
            attentioned_v_Feature = torch.matmul(attention_map, v).permute(0, 4, 3, 5, 1, 2).reshape(batch, seq, self.num_heads * self.depth_perhead, height, width)
        return attentioned_v_Feature


class out(nn.Module):

    def __init__(self, depth):
        super(out, self).__init__()
        self.pad = (3 - 1) * 1
        self.conv0 = nn.Sequential(nn.Conv3d(in_channels=depth, out_channels=depth, kernel_size=(3, 3, 3), stride=1, padding=(self.pad, 1, 1), bias=True))

    def forward(self, input_tensor):
        out = self.conv0(input_tensor.permute(0, 2, 1, 3, 4))
        out = out[:, :, :-self.pad].permute(0, 2, 1, 3, 4)
        return out


class DecoderLayer(nn.Module):

    def __init__(self, model_depth=128, num_heads=4, with_pos=True):
        super(DecoderLayer, self).__init__()
        self.depth = model_depth
        self.depth_perhead = int(model_depth / num_heads)
        self.attention = self.__get_clones(MultiHeadAttention(self.depth_perhead, num_heads, with_pos=with_pos), 1)
        self.out = out(self.depth)
        self.feedforward = FeedForwardNet(self.depth)
        self.GN1 = nn.GroupNorm(num_groups=1, num_channels=model_depth)
        self.GN2 = nn.GroupNorm(num_groups=1, num_channels=model_depth)
        self.dropout1 = nn.Dropout3d(0.1)
        self.dropout2 = nn.Dropout3d(0.1)

    def __get_clones(self, module, n):
        return nn.ModuleList([copy.deepcopy(module) for i in range(n)])

    def forward(self, input_tensor, pos_decoding):
        att_layer_in = self.GN1(input_tensor.permute(0, 2, 1, 3, 4)).permute(0, 2, 1, 3, 4)
        i = 0
        for layer in self.attention:
            att_out = layer(att_layer_in, pos_decoding, type=0)
        att_layer_out = self.dropout1(self.out(att_out).permute(0, 2, 1, 3, 4)).permute(0, 2, 1, 3, 4) + input_tensor
        ff_in = self.GN2(att_layer_out.permute(0, 2, 1, 3, 4)).permute(0, 2, 1, 3, 4)
        out = self.dropout2(self.feedforward(ff_in).permute(0, 2, 1, 3, 4)).permute(0, 2, 1, 3, 4) + att_layer_out
        return out


class PositionalEmbeddingLearned(nn.Module):

    def __init__(self, embedding_depth=128):
        super(PositionalEmbeddingLearned, self).__init__()
        self.depth = embedding_depth
        self.positional_embedding = nn.Embedding(10, self.depth)

    def forward(self, shape):
        b, c, h, w = shape
        index = torch.arange(b)
        position = self.positional_embedding(index)
        position = position.unsqueeze(2).repeat(1, 1, h * w).reshape(b, self.depth, h, w)
        return position


def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, 2 * (i // 2) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding(position, d_model, h=128, w=226):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis], np.arange(d_model)[np.newaxis, :], d_model)
    sines = np.sin(angle_rads[:, 0::2])
    cones = np.cos(angle_rads[:, 1::2])
    pos_encoding = np.concatenate([sines, cones], axis=-1).astype(np.float32)
    pos_embedding = torch.from_numpy(0.5 * pos_encoding)
    pos = pos_embedding.unsqueeze(2).repeat(1, 1, h * w).reshape(position, d_model, h, w)
    return pos


class Decoder(nn.Module):

    def __init__(self, num_layers=5, num_frames=1, model_depth=128, num_heads=4, with_residual=True, with_pos=True, pos_kind='sine'):
        super(Decoder, self).__init__()
        self.depth = model_depth
        self.decoderlayer = DecoderLayer(model_depth, num_heads, with_pos=with_pos)
        self.num_layers = num_layers
        self.decoder = self.__get_clones(self.decoderlayer, self.num_layers)
        self.positionnet = PositionalEmbeddingLearned(int(model_depth / num_heads))
        self.num_frames = num_frames
        self.pos_kind = pos_kind
        self.GN = nn.GroupNorm(num_groups=1, num_channels=model_depth)

    def __get_clones(self, module, n):
        return nn.ModuleList([copy.deepcopy(module) for i in range(n)])

    def forward(self, dec_init):
        b, s, c, h, w = dec_init.shape
        out = dec_init
        if self.pos_kind == 'sine':
            pos_dec = positional_encoding(s, self.depth, h, w)
            pos_dec = pos_dec.unsqueeze(0).expand(b, -1, -1, -1, -1)
        elif self.pos_kind == 'learned':
            pos_dec = self.positionnet(out.shape)
        else:
            None
            return
        for layer in self.decoder:
            out = layer(out, pos_dec)
        return self.GN(out.permute(0, 2, 1, 3, 4)).permute(0, 2, 1, 3, 4)


class TCTN(nn.Module):

    def __init__(self, num_layers, num_dec_frames, model_depth, num_heads, with_residual, with_pos, pos_kind, mode, config):
        super(TCTN, self).__init__()
        self.configs = config
        self.decoder_embedding = DecoderEmbedding(model_depth)
        self.decoder = Decoder(num_layers=config.de_layers, model_depth=model_depth, num_heads=num_heads, num_frames=num_dec_frames, with_residual=with_residual, with_pos=with_pos, pos_kind=pos_kind)
        self.conv_last = nn.Conv3d(model_depth, config.img_channel * config.patch_size * config.patch_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.task = mode
        self.num_dec_frames = num_dec_frames

    def forward(self, input_img, val_signal=1):
        if val_signal == 0:
            dec_init = self.decoder_embedding(input_img)
            decoderout = self.decoder(dec_init)
            if self.configs.w_pffn == 1:
                out = self.prediction(decoderout)
            else:
                out = self.conv_last(decoderout.permute(0, 2, 1, 3, 4)).permute(0, 2, 1, 3, 4)
        else:
            for i in range(self.configs.total_length - self.configs.input_length):
                if i == 0:
                    dec_init = self.decoder_embedding(input_img[:, 0:self.configs.input_length])
                else:
                    dec_init = torch.cat((dec_init, new_embedding), 1)
                decoderout = self.decoder(dec_init)
                if i < self.configs.total_length - self.configs.input_length - 1:
                    nex_img = decoderout[:, -1].unsqueeze(1)
                    if self.configs.w_pffn == 1:
                        img = self.prediction(nex_img)
                    else:
                        img = self.conv_last(nex_img.permute(0, 2, 1, 3, 4)).permute(0, 2, 1, 3, 4)
                    new_embedding = self.decoder_embedding(img)
                elif self.configs.w_pffn == 1:
                    out = self.prediction(decoderout)
                else:
                    out = self.conv_last(decoderout.permute(0, 2, 1, 3, 4)).permute(0, 2, 1, 3, 4)
        return out


class WeightedCrossEntropyLoss(nn.Module):

    def __init__(self, thresholds, weight=None, LAMBDA=None):
        super().__init__()
        self._weight = weight
        self._lambda = LAMBDA
        self._thresholds = thresholds

    def forward(self, inputs, targets):
        inputs = inputs.permute((0, 2, 1, 3, 4))
        targets = targets.squeeze(2)
        class_index = torch.zeros_like(targets).long()
        thresholds = self._thresholds
        class_index[...] = 0
        for i, threshold in enumerate(thresholds):
            i = i + 1
            class_index[targets >= threshold] = i
        if self._weight is not None:
            self._weight = self._weight
            error = F.cross_entropy(inputs, class_index, self._weight, reduction='none')
        else:
            error = F.cross_entropy(inputs, class_index, reduction='none')
        if self._lambda is not None:
            B, S, H, W = error.size()
            w = torch.arange(1.0, 1.0 + S * self._lambda, self._lambda)
            if torch.cuda.is_available():
                w = w
            error = (w * error.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        error = error.permute(0, 1, 2, 3).unsqueeze(2)
        return torch.mean(error.float())

