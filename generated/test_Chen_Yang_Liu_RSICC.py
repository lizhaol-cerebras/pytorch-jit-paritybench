
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


import torch.backends.cudnn as cudnn


import torch.optim


import torch.utils.data


import torchvision.transforms as transforms


import torch.nn.functional as F


import time


import numpy


import torch


from torch.utils.data import Dataset


from torch import nn


import torchvision


from torchvision.transforms import Resize


import math


from torch.nn.init import xavier_uniform_


from torch import Tensor


from typing import Optional


from torch.nn import functional as F


from torch.nn.utils.rnn import pack_padded_sequence


from torch.optim.lr_scheduler import StepLR


import numpy as np


from collections import Counter


from random import seed


from random import choice


from random import sample


import matplotlib.pyplot as plt


import matplotlib as mpl


class CNN_Encoder(nn.Module):
    """
    CNN_Encoder.
    """

    def __init__(self, NetType, method, encoded_image_size=14):
        super(CNN_Encoder, self).__init__()
        self.NetType = NetType
        self.enc_image_size = encoded_image_size
        if 'resnet' in NetType:
            cnn = getattr(torchvision.models, NetType)(pretrained=True)
            layers = [cnn.conv1, cnn.bn1, cnn.relu, cnn.maxpool]
            model_stage = 3
            for i in range(model_stage):
                name = 'layer%d' % (i + 1)
                layers.append(getattr(cnn, name))
            self.net = nn.Sequential(*layers)
        if 'vgg' in NetType:
            net = torchvision.models.vgg16(pretrained=True)
            modules = list(net.children())[:-1]
            self.net = nn.Sequential(*modules)
        if 'vit' in NetType:
            net = getattr(torchvision.models, NetType)(pretrained=True)
            self.net = net
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))
        self.fine_tune()

    def forward(self, images):
        """
        Forward propagation.
        :param images: images, a tensor of dimensions (batch_size, 3, image_size, image_size)
        :return: encoded images [batch_size, encoded_image_size=14, encoded_image_size=14, 2048]
        """
        self.NetType = 'resnet'
        if 'resnet' in self.NetType:
            out = self.net(images)
            out = self.adaptive_pool(out)
        if 'vgg' in self.NetType:
            out = self.net(images)
            out = self.adaptive_pool(out)
        if 'vit' in self.NetType:
            torch_resize = Resize([224, 224])
            images = torch_resize(images)
            x = self.net._process_input(images)
            n = x.shape[0]
            batch_class_token = self.net.class_token.expand(n, -1, -1)
            x = torch.cat([batch_class_token, x], dim=1)
            x = self.net.encoder(x)
            x = x[:, 1:, :]
            out = x
            out = out.permute(0, 2, 1).view(n, -1, 14, 14)
        return out

    def fine_tune(self, fine_tune=True):
        """
        Allow or prevent the computation of gradients for convolutional blocks 2 through 4 of the encoder.

        :param fine_tune: Allow?
        """
        for p in self.net.parameters():
            p.requires_grad = False
        for c in list(self.net.children())[5:]:
            for p in c.parameters():
                p.requires_grad = fine_tune


class CrossTransformer(nn.Module):
    """
    Cross Transformer layer
    """

    def __init__(self, dropout, d_model=512, n_head=4):
        """
        :param dropout: dropout rate
        :param d_model: dimension of hidden state
        :param n_head: number of heads in multi head attention
        """
        super(CrossTransformer, self).__init__()
        self.attention = nn.MultiheadAttention(d_model, n_head, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        self.linear1 = nn.Linear(d_model, d_model * 4)
        self.linear2 = nn.Linear(d_model * 4, d_model)

    def forward(self, input1, input2):
        dif = input2 - input1
        output_1 = self.cross(input1, dif)
        output_2 = self.cross(input2, dif)
        return output_1, output_2

    def cross(self, input, dif):
        attn_output, attn_weight = self.attention(input, dif, dif)
        output = input + self.dropout1(attn_output)
        output = self.norm1(output)
        ff_output = self.linear2(self.dropout2(self.activation(self.linear1(output))))
        output = output + self.dropout3(ff_output)
        output = self.norm2(output)
        return output


class resblock(nn.Module):
    """
    module: Residual Block
    """

    def __init__(self, inchannel, outchannel, stride=1, shortcut=None):
        super(resblock, self).__init__()
        self.left = nn.Sequential(nn.Conv2d(inchannel, int(outchannel / 2), kernel_size=1), nn.BatchNorm2d(int(outchannel / 2)), nn.ReLU(inplace=True), nn.Conv2d(int(outchannel / 2), int(outchannel / 2), kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(int(outchannel / 2)), nn.ReLU(inplace=True), nn.Conv2d(int(outchannel / 2), outchannel, kernel_size=1), nn.BatchNorm2d(outchannel))
        self.right = shortcut

    def forward(self, x):
        out = self.left(x)
        residual = x
        out += residual
        return F.relu(out)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class MCCFormers_diff_as_Q(nn.Module):
    """
    RSICCFormers_diff
    """

    def __init__(self, feature_dim, dropout, h, w, d_model=512, n_head=4, n_layers=3):
        """
        :param feature_dim: dimension of input features
        :param dropout: dropout rate
        :param d_model: dimension of hidden state
        :param n_head: number of heads in multi head attention
        :param n_layer: number of layers of transformer layer
        """
        super(MCCFormers_diff_as_Q, self).__init__()
        self.d_model = d_model
        None
        self.n_layers = n_layers
        self.w_embedding = nn.Embedding(w, int(d_model / 2))
        self.h_embedding = nn.Embedding(h, int(d_model / 2))
        self.embedding_1D = nn.Embedding(h * w, int(d_model))
        self.projection = nn.Conv2d(feature_dim, d_model, kernel_size=1)
        self.projection2 = nn.Conv2d(768, d_model, kernel_size=1)
        self.projection3 = nn.Conv2d(512, d_model, kernel_size=1)
        self.projection4 = nn.Conv2d(256, d_model, kernel_size=1)
        self.transformer = nn.ModuleList([CrossTransformer(dropout, d_model, n_head) for i in range(n_layers)])
        self.resblock = nn.ModuleList([resblock(d_model * 2, d_model * 2) for i in range(n_layers)])
        self.LN = nn.ModuleList([nn.LayerNorm(d_model * 2) for i in range(n_layers)])
        self._reset_parameters()

    def _reset_parameters(self):
        """Initiate parameters in the transformer model."""
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

    def forward(self, img_feat1, img_feat2):
        batch = img_feat1.size(0)
        feature_dim = img_feat1.size(1)
        w, h = img_feat1.size(2), img_feat1.size(3)
        if feature_dim == 1024:
            img_feat1 = self.projection(img_feat1)
            img_feat2 = self.projection(img_feat2)
        if feature_dim == 768:
            img_feat1 = self.projection2(img_feat1)
            img_feat2 = self.projection2(img_feat2)
        if feature_dim == 512:
            img_feat1 = self.projection3(img_feat1)
            img_feat2 = self.projection3(img_feat2)
        if feature_dim == 256:
            img_feat1 = self.projection4(img_feat1)
            img_feat2 = self.projection4(img_feat2)
        pos_w = torch.arange(w, device=device)
        pos_h = torch.arange(h, device=device)
        embed_w = self.w_embedding(pos_w)
        embed_h = self.h_embedding(pos_h)
        position_embedding = torch.cat([embed_w.unsqueeze(0).repeat(h, 1, 1), embed_h.unsqueeze(1).repeat(1, w, 1)], dim=-1)
        position_embedding = position_embedding.permute(2, 0, 1).unsqueeze(0).repeat(batch, 1, 1, 1)
        img_feat1 = img_feat1 + position_embedding
        img_feat2 = img_feat2 + position_embedding
        encoder_output1 = img_feat1.view(batch, self.d_model, -1).permute(2, 0, 1)
        encoder_output2 = img_feat2.view(batch, self.d_model, -1).permute(2, 0, 1)
        output1 = encoder_output1
        output2 = encoder_output2
        output1_list = list()
        output2_list = list()
        for l in self.transformer:
            output1, output2 = l(output1, output2)
            output1_list.append(output1)
            output2_list.append(output2)
        i = 0
        output = torch.zeros((196, batch, self.d_model * 2))
        for res in self.resblock:
            input = torch.cat([output1_list[i], output2_list[i]], dim=-1)
            output = output + input
            output = output.permute(1, 2, 0).view(batch, self.d_model * 2, 14, 14)
            output = res(output)
            output = output.view(batch, self.d_model * 2, -1).permute(2, 0, 1)
            output = self.LN[i](output)
            i = i + 1
        return output


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
        self.embedding_1D = nn.Embedding(52, int(d_model))

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class Mesh_TransformerDecoderLayer(nn.Module):
    __constants__ = ['batch_first', 'norm_first']

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, layer_norm_eps=1e-05, batch_first=False, norm_first=False, device=None, dtype=None) ->None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(Mesh_TransformerDecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.self_attn2 = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(int(d_model), nhead, dropout=dropout)
        self.multihead_attn2 = nn.MultiheadAttention(d_model, int(nhead), dropout=dropout)
        self.multihead_attn3 = nn.MultiheadAttention(int(d_model), int(nhead), dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.dropout4 = nn.Dropout(dropout)
        self.dropout5 = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        self.activation2 = nn.Softmax(dim=-1)
        self.fc_alpha1 = nn.Linear(d_model + d_model, d_model)
        self.fc_alpha2 = nn.Linear(d_model + d_model, d_model)
        self.fc_alpha3 = nn.Linear(d_model + d_model, d_model)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.fc_alpha1.weight)
        nn.init.xavier_uniform_(self.fc_alpha2.weight)
        nn.init.xavier_uniform_(self.fc_alpha3.weight)
        nn.init.constant_(self.fc_alpha1.bias, 0)
        nn.init.constant_(self.fc_alpha2.bias, 0)
        nn.init.constant_(self.fc_alpha3.bias, 0)

    def forward(self, tgt: 'Tensor', memory: 'Tensor', tgt_mask: 'Optional[Tensor]'=None, memory_mask: 'Optional[Tensor]'=None, tgt_key_padding_mask: 'Optional[Tensor]'=None, memory_key_padding_mask: 'Optional[Tensor]'=None) ->Tensor:
        self_att_tgt = self.norm1(tgt + self._sa_block(tgt, tgt_mask, tgt_key_padding_mask))
        enc_att, att_weight = self._mha_block2(self_att_tgt, memory, memory_mask, memory_key_padding_mask)
        x = self.norm2(self_att_tgt + enc_att)
        x = self.norm3(x + self._ff_block(x))
        return x

    def _sa_block(self, x: 'Tensor', attn_mask: 'Optional[Tensor]', key_padding_mask: 'Optional[Tensor]') ->Tensor:
        x = self.self_attn(x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=False)[0]
        return self.dropout1(x)

    def _mha_block(self, x: 'Tensor', mem: 'Tensor', attn_mask: 'Optional[Tensor]', key_padding_mask: 'Optional[Tensor]') ->Tensor:
        x, att_weight = self.multihead_attn(x, mem, mem, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=True)
        return self.dropout2(x), att_weight

    def _mha_block2(self, x: 'Tensor', mem: 'Tensor', attn_mask: 'Optional[Tensor]', key_padding_mask: 'Optional[Tensor]') ->Tensor:
        x, att_weight = self.multihead_attn2(x, mem, mem, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=True)
        return self.dropout3(x), att_weight

    def _mha_block3(self, x: 'Tensor', mem: 'Tensor', attn_mask: 'Optional[Tensor]', key_padding_mask: 'Optional[Tensor]') ->Tensor:
        x, att_weight = self.multihead_attn3(x, mem, mem, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=True)
        return self.dropout4(x), att_weight

    def _ff_block(self, x: 'Tensor') ->Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout4(x)


class DecoderTransformer(nn.Module):
    """
    Decoder with Transformer.
    """

    def __init__(self, feature_dim, vocab_size, n_head, n_layers, dropout):
        """
        :param n_head: the number of heads in Transformer
        :param n_layers: the number of layers of Transformer
        """
        super(DecoderTransformer, self).__init__()
        None
        self.feature_dim = feature_dim
        self.embed_dim = feature_dim
        self.vocab_size = vocab_size
        self.dropout = dropout
        self.vocab_embedding = nn.Embedding(vocab_size, self.embed_dim)
        decoder_layer = Mesh_TransformerDecoderLayer(feature_dim, n_head, dim_feedforward=feature_dim * 4, dropout=self.dropout)
        self.transformer = nn.TransformerDecoder(decoder_layer, n_layers)
        self.position_encoding = PositionalEncoding(feature_dim)
        self.wdc = nn.Linear(feature_dim, vocab_size)
        self.dropout = nn.Dropout(p=self.dropout)
        self.init_weights()

    def init_weights(self):
        """
        Initializes some parameters with values from the uniform distribution, for easier convergence
        """
        self.vocab_embedding.weight.data.uniform_(-0.1, 0.1)
        self.wdc.bias.data.fill_(0)
        self.wdc.weight.data.uniform_(-0.1, 0.1)

    def forward(self, memory, encoded_captions, caption_lengths):
        """
        :param memory: image feature (S, batch, feature_dim)
        :param tgt: target sequence (length, batch)
        :param sentence_index: sentence index of each token in target sequence (length, batch)
        """
        tgt = encoded_captions.permute(1, 0)
        tgt_length = tgt.size(0)
        mask = (torch.triu(torch.ones(tgt_length, tgt_length)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        mask = mask
        tgt_embedding = self.vocab_embedding(tgt)
        tgt_embedding = self.position_encoding(tgt_embedding)
        pred = self.transformer(tgt_embedding, memory, tgt_mask=mask)
        pred = self.wdc(self.dropout(pred))
        pred = pred.permute(1, 0, 2)
        caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0, descending=True)
        encoded_captions = encoded_captions[sort_ind]
        pred = pred[sort_ind]
        decode_lengths = (caption_lengths - 1).tolist()
        return pred, encoded_captions, decode_lengths, sort_ind


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (Mesh_TransformerDecoderLayer,
     lambda: ([], {'d_model': 4, 'nhead': 4}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4])], {})),
    (PositionalEncoding,
     lambda: ([], {'d_model': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (resblock,
     lambda: ([], {'inchannel': 4, 'outchannel': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
]

