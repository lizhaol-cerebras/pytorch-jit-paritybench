
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


import numpy as np


import torch.nn.functional as F


import torchvision.transforms.functional as tf


import math


import torchvision


import torch.nn as nn


from typing import Sequence


from itertools import chain


from torchvision import models


from collections import OrderedDict


from torchvision import transforms


from torch.utils.data import Dataset


from torch.utils.data import DataLoader


import random


from scipy.stats import skewnorm


import typing


from abc import ABC


from abc import ABCMeta


from abc import abstractmethod


from abc import abstractproperty


from typing import Type


from torch import nn


from typing import Tuple


from sklearn.decomposition import PCA


import torchvision.transforms


import torch.nn.modules.utils as nn_utils


import types


from typing import Union


from typing import List


import torch.optim as optim


from torch.optim.lr_scheduler import LinearLR


from torch.utils.tensorboard import SummaryWriter


import matplotlib.pyplot as plt


from typing import NamedTuple


from random import randint


import uuid


from torch.autograd import Variable


from math import exp


class LinLayers(nn.ModuleList):

    def __init__(self, n_channels_list: 'Sequence[int]'):
        super(LinLayers, self).__init__([nn.Sequential(nn.Identity(), nn.Conv2d(nc, 1, 1, 1, 0, bias=False)) for nc in n_channels_list])
        for param in self.parameters():
            param.requires_grad = False


def normalize_activation(x, eps=1e-10):
    norm_factor = torch.sqrt(torch.sum(x ** 2, dim=1, keepdim=True))
    return x / (norm_factor + eps)


class BaseNet(nn.Module):

    def __init__(self):
        super(BaseNet, self).__init__()
        self.register_buffer('mean', torch.Tensor([-0.03, -0.088, -0.188])[None, :, None, None])
        self.register_buffer('std', torch.Tensor([0.458, 0.448, 0.45])[None, :, None, None])

    def set_requires_grad(self, state: 'bool'):
        for param in chain(self.parameters(), self.buffers()):
            param.requires_grad = state

    def z_score(self, x: 'torch.Tensor'):
        return (x - self.mean) / self.std

    def forward(self, x: 'torch.Tensor'):
        x = self.z_score(x)
        output = []
        for i, (_, layer) in enumerate(self.layers._modules.items(), 1):
            x = layer(x)
            if i in self.target_layers:
                output.append(normalize_activation(x))
            if len(output) == len(self.target_layers):
                break
        return output


class AlexNet(BaseNet):

    def __init__(self):
        super(AlexNet, self).__init__()
        self.layers = models.alexnet(True).features
        self.target_layers = [2, 5, 8, 10, 12]
        self.n_channels_list = [64, 192, 384, 256, 256]
        self.set_requires_grad(False)


class SqueezeNet(BaseNet):

    def __init__(self):
        super(SqueezeNet, self).__init__()
        self.layers = models.squeezenet1_1(True).features
        self.target_layers = [2, 5, 8, 10, 11, 12, 13]
        self.n_channels_list = [64, 128, 256, 384, 384, 512, 512]
        self.set_requires_grad(False)


class VGG16(BaseNet):

    def __init__(self):
        super(VGG16, self).__init__()
        self.layers = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
        self.target_layers = [4, 9, 16, 23, 30]
        self.n_channels_list = [64, 128, 256, 512, 512]
        self.set_requires_grad(False)


def get_network(net_type: 'str'):
    if net_type == 'alex':
        return AlexNet()
    elif net_type == 'squeeze':
        return SqueezeNet()
    elif net_type == 'vgg':
        return VGG16()
    else:
        raise NotImplementedError('choose net_type from [alex, squeeze, vgg].')


def get_state_dict(net_type: 'str'='alex', version: 'str'='0.1'):
    url = 'https://raw.githubusercontent.com/richzhang/PerceptualSimilarity/' + f'master/lpips/weights/v{version}/{net_type}.pth'
    old_state_dict = torch.hub.load_state_dict_from_url(url, progress=True, map_location=None if torch.cuda.is_available() else torch.device('cpu'))
    new_state_dict = OrderedDict()
    for key, val in old_state_dict.items():
        new_key = key
        new_key = new_key.replace('lin', '')
        new_key = new_key.replace('model.', '')
        new_state_dict[new_key] = val
    return new_state_dict


class LPIPS(nn.Module):
    """Creates a criterion that measures
    Learned Perceptual Image Patch Similarity (LPIPS).

    Arguments:
        net_type (str): the network type to compare the features: 
                        'alex' | 'squeeze' | 'vgg'. Default: 'alex'.
        version (str): the version of LPIPS. Default: 0.1.
    """

    def __init__(self, net_type: 'str'='alex', version: 'str'='0.1'):
        assert version in ['0.1'], 'v0.1 is only supported now'
        super(LPIPS, self).__init__()
        self.net = get_network(net_type)
        self.lin = LinLayers(self.net.n_channels_list)
        self.lin.load_state_dict(get_state_dict(net_type, version))

    def forward(self, x: 'torch.Tensor', y: 'torch.Tensor'):
        feat_x, feat_y = self.net(x), self.net(y)
        diff = [((fx - fy) ** 2) for fx, fy in zip(feat_x, feat_y)]
        res = [l(d).mean((2, 3), True) for d, l in zip(diff, self.lin)]
        return torch.sum(torch.cat(res, 0), 0, True)


class BaseImageEncoder(nn.Module):

    @abstractproperty
    def name(self) ->str:
        """
        returns the name of the encoder
        """

    @abstractproperty
    def embedding_dim(self) ->int:
        """
        returns the dimension of the embeddings
        """

    @abstractmethod
    def encode_image(self, input: 'torch.Tensor') ->torch.Tensor:
        """
        Given a batch of input images, return their encodings
        """

    @abstractmethod
    def get_relevancy(self, embed: 'torch.Tensor', positive_id: 'int') ->torch.Tensor:
        """
        Given a batch of embeddings, return the relevancy to the given positive id
        """


class OpenCLIPNetwork(BaseImageEncoder):

    def __init__(self, config: 'OpenCLIPNetworkConfig'):
        super().__init__()
        self.config = config
        self.process = torchvision.transforms.Compose([torchvision.transforms.Resize((224, 224)), torchvision.transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])])
        model, _, _ = open_clip.create_model_and_transforms(self.config.clip_model_type, pretrained=self.config.clip_model_pretrained, precision='fp16')
        model.eval()
        self.tokenizer = open_clip.get_tokenizer(self.config.clip_model_type)
        self.model = model
        self.clip_n_dims = self.config.clip_n_dims
        self.positive_input = None
        self.positives = self.positive_input.value.split(';')
        self.negatives = self.config.negatives
        with torch.no_grad():
            tok_phrases = torch.cat([self.tokenizer(phrase) for phrase in self.positives])
            self.pos_embeds = model.encode_text(tok_phrases)
            tok_phrases = torch.cat([self.tokenizer(phrase) for phrase in self.negatives])
            self.neg_embeds = model.encode_text(tok_phrases)
        self.pos_embeds /= self.pos_embeds.norm(dim=-1, keepdim=True)
        self.neg_embeds /= self.neg_embeds.norm(dim=-1, keepdim=True)
        assert self.pos_embeds.shape[1] == self.neg_embeds.shape[1], 'Positive and negative embeddings must have the same dimensionality'
        assert self.pos_embeds.shape[1] == self.clip_n_dims, 'Embedding dimensionality must match the model dimensionality'

    @property
    def name(self) ->str:
        return 'openclip_{}_{}'.format(self.config.clip_model_type, self.config.clip_model_pretrained)

    @property
    def embedding_dim(self) ->int:
        return self.config.clip_n_dims

    def gui_cb(self, element):
        self.set_positives(element.value.split(';'))

    def set_positives(self, text_list):
        self.positives = text_list
        with torch.no_grad():
            tok_phrases = torch.cat([self.tokenizer(phrase) for phrase in self.positives])
            self.pos_embeds = self.model.encode_text(tok_phrases)
        self.pos_embeds /= self.pos_embeds.norm(dim=-1, keepdim=True)

    def get_relevancy(self, embed: 'torch.Tensor', positive_id: 'int') ->torch.Tensor:
        phrases_embeds = torch.cat([self.pos_embeds, self.neg_embeds], dim=0)
        p = phrases_embeds
        output = torch.mm(embed, p.T)
        positive_vals = output[..., positive_id:positive_id + 1]
        negative_vals = output[..., len(self.positives):]
        repeated_pos = positive_vals.repeat(1, len(self.negatives))
        sims = torch.stack((repeated_pos, negative_vals), dim=-1)
        softmax = torch.softmax(10 * sims, dim=-1)
        best_id = softmax[..., 0].argmin(dim=1)
        return torch.gather(softmax, 1, best_id[..., None, None].expand(best_id.shape[0], len(self.negatives), 2))[:, 0, :]

    def encode_image(self, input):
        processed_input = self.process(input).half()
        return self.model.encode_image(processed_input)


class VectorQuantizer(nn.Module):
    """
    Discretization bottleneck part of the VQ-VAE.

    Inputs:
    - n_e : number of embeddings
    - e_dim : dimension of embedding
    - beta : commitment cost used in loss term, beta * ||z_e(x)-sg[e]||^2
    """

    def __init__(self, n_e, e_dim, beta, device, concat=False, dino_weight=0):
        super(VectorQuantizer, self).__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)
        if device != None:
            self.device = device
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.concat = concat
        self.dino_weight = dino_weight

    def forward(self, z):
        """
        Inputs the output of the encoder network z and maps it to a discrete 
        one-hot vector that is the index of the closest embedding vector e_j

        z (continuous) -> z_q (discrete)

        z.shape = (batch, height, width, channel)

        quantization pipeline:

            1. get encoder input (B,H,W,C)
            2. flatten input to (B*H*W,C)

        """
        z_flattened = z.view(-1, self.e_dim)
        assert not torch.isnan(z_flattened).any()
        cb_normalized = self._normalize_cb(self.embedding.weight)
        d = self._d(cb_normalized, z_flattened)
        assert not torch.isnan(cb_normalized).any()
        assert not torch.isnan(d).any()
        min_encoding_indices = torch.argmax(d, dim=1).unsqueeze(1)
        encoding_indices_prob = torch.softmax(d, dim=1)
        assert not torch.isnan(min_encoding_indices).any()
        assert not torch.isnan(encoding_indices_prob).any()
        min_encodings = torch.zeros(min_encoding_indices.shape[0], self.n_e)
        min_encodings.scatter_(1, min_encoding_indices, 1)
        z_q = torch.matmul(min_encodings, cb_normalized).view(z.shape)
        assert not torch.isnan(z_q).any()
        e_mean = torch.mean(min_encodings, dim=0)
        loss_kl = -torch.sum(e_mean * torch.log(1 / self.n_e / (e_mean + 1e-06)))
        loss, constrative_loss = self._loss(cb_normalized, min_encoding_indices, z_q, z)
        z_q = z + (z_q - z).detach()
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-06)))
        return loss, constrative_loss, loss_kl, encoding_indices_prob, d, z_q, perplexity, min_encodings, min_encoding_indices

    def _d(self, cb, z_flattened):
        if self.concat:
            d_clip = self._cosine_sim(cb[:, :512], z_flattened[:, :512])
            d_dino = self._cosine_sim(cb[:, 512:], z_flattened[:, 512:])
            d = d_clip + self.dino_weight * d_dino
        else:
            d = self._cosine_sim(cb, z_flattened)
        return d

    def _loss(self, cb, min_encoding_indices, z_q, z):
        loss = 0
        constrative_loss = 0
        if self.concat:
            z_q_clip = z_q[:, :, :, :512]
            z_q_dino = z_q[:, :, :, 512:]
            loss_cos_clip = 1 - torch.mean(torch.cosine_similarity(z_q_clip, z.detach()[:, :, :, :512], dim=-1)) + self.beta * (1 - torch.mean(torch.cosine_similarity(z_q_clip.detach(), z[:, :, :, :512], dim=-1)))
            loss_cos_dino = 1 - torch.mean(torch.cosine_similarity(z_q_dino, z.detach()[:, :, :, 512:], dim=-1)) + self.beta * (1 - torch.mean(torch.cosine_similarity(z_q_dino.detach(), z[:, :, :, 512:], dim=-1)))
            loss += loss_cos_clip + self.dino_weight * loss_cos_dino
            cb_clip = cb[..., :512]
            cb_dino = cb[..., 512:]
            cb_clip_cos = torch.cosine_similarity(cb_clip.unsqueeze(0), cb_clip.unsqueeze(1), dim=-1)
            cb_dino_cos = torch.cosine_similarity(cb_dino.unsqueeze(0), cb_dino.unsqueeze(1), dim=-1)
            cb_cos = cb_clip_cos + self.dino_weight * cb_dino_cos
            cb_neg = (torch.sum(cb_cos, dim=1) - cb_cos[0][0]) / (cb_cos.shape[0] - 1)
            x = F.embedding(min_encoding_indices, cb_neg[..., None]).squeeze()
            zq_clip_cos = torch.cosine_similarity(z_q_clip, z.detach()[:, :, :, :512], dim=-1).view(-1)
            zq_dino_cos = torch.cosine_similarity(z_q_dino, z.detach()[:, :, :, 512:], dim=-1).view(-1)
            zq_cos = zq_clip_cos + self.dino_weight * zq_dino_cos
            constrative_loss += torch.mean(-1 * zq_cos + x)
        else:
            loss += 1 - torch.mean(torch.cosine_similarity(z_q, z.detach(), dim=-1)) + self.beta * (1 - torch.mean(torch.cosine_similarity(z_q.detach(), z, dim=-1)))
            constrative_loss += 0
        return loss, constrative_loss

    def _mse(self, embedding, z_flattened):
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + torch.sum(embedding ** 2, dim=1) - 2 * torch.matmul(z_flattened, embedding.t())
        return d

    def _cosine_sim(self, embedding, z_flattened):
        embedding_norm = torch.norm(embedding, dim=-1)[None, :]
        z_flattened_norm = torch.norm(z_flattened, dim=-1)[:, None]
        assert not torch.isnan(embedding).any()
        assert not torch.isnan(z_flattened).any()
        assert not torch.isnan(embedding_norm).any()
        assert not torch.isnan(z_flattened_norm).any()
        d = torch.matmul(z_flattened, embedding.t()) / (torch.matmul(z_flattened_norm, embedding_norm) + 1e-06)
        assert not torch.isnan(torch.matmul(z_flattened, embedding.t())).any()
        assert not torch.isnan(torch.matmul(z_flattened_norm, embedding_norm)).any()
        assert not torch.isnan(d).any()
        return d

    def _normalize_cb(self, cb):
        norm_cb_clip = torch.norm(cb[..., :512], p=2, dim=-1, keepdim=True)
        norm_cb_dino = torch.norm(cb[..., 512:], p=2, dim=-1, keepdim=True)
        cb_clip_normalized = cb[..., :512] / norm_cb_clip
        cb_dino_normalized = cb[..., 512:] / norm_cb_dino
        cb_normalized = torch.cat((cb_clip_normalized, cb_dino_normalized), dim=-1)
        return cb_normalized


def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan(fovY / 2)
    tanHalfFovX = math.tan(fovX / 2)
    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right
    P = torch.zeros(4, 4)
    z_sign = 1.0
    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


def getWorld2View2(R, t, translate=np.array([0.0, 0.0, 0.0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)


class Camera(nn.Module):

    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask, language_feature_indices, image_name, uid, is_novel_view=False, trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device='cuda'):
        super(Camera, self).__init__()
        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name
        self.is_novel_view = is_novel_view
        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            None
            None
            self.data_device = torch.device('cuda')
        if image is not None:
            self.original_image = image.clamp(0.0, 1.0)
            self.image_width = self.original_image.shape[2]
            self.image_height = self.original_image.shape[1]
        else:
            self.original_image = None
            self.image_width = None
            self.image_height = None
        if language_feature_indices is not None:
            self.language_feature_indices = torch.from_numpy(language_feature_indices)
        else:
            self.language_feature_indices = None
        if self.original_image is not None:
            if gt_alpha_mask is not None:
                self.original_image *= gt_alpha_mask
            else:
                self.original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)
        self.zfar = 100.0
        self.znear = 0.01
        self.trans = trans
        self.scale = scale
        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1)
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0, 1)
        self.full_proj_transform = self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0)).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]


class IndexDecoder(nn.Module):

    def __init__(self, input_channels, output_channels):
        super(IndexDecoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=128, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=1)
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=output_channels, kernel_size=1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.conv3(x)
        return x


class XyzMLP(nn.Module):

    def __init__(self, D=4, W=128, in_channels_xyz=63, out_channels_xyz=8):
        super(XyzMLP, self).__init__()
        self.D = D
        self.W = W
        self.in_channels_xyz = in_channels_xyz
        self.out_channels_xyz = out_channels_xyz
        for i in range(D):
            if i == 0:
                layer = nn.Linear(in_channels_xyz, W)
            else:
                layer = nn.Linear(W, W)
            layer = nn.Sequential(layer, nn.ReLU(True))
            setattr(self, f'xyz_encoding_{i}', layer)
        self.xyz_encoding_final = nn.Linear(W, out_channels_xyz)

    def forward(self, x):
        for i in range(self.D):
            x = getattr(self, f'xyz_encoding_{i}')(x)
        xyz_encoding_final = self.xyz_encoding_final(x)
        return xyz_encoding_final


class Embedding(nn.Module):

    def __init__(self, in_channels, N_freqs, logscale=True):
        """
        Defines a function that embeds x to (x, sin(2^k x), cos(2^k x), ...)
        in_channels: number of input channels (3 for both xyz and direction)
        """
        super(Embedding, self).__init__()
        self.N_freqs = N_freqs
        self.in_channels = in_channels
        self.funcs = [torch.sin, torch.cos]
        self.out_channels = in_channels * (len(self.funcs) * N_freqs + 1)
        if logscale:
            self.freq_bands = 2 ** torch.linspace(0, N_freqs - 1, N_freqs)
        else:
            self.freq_bands = torch.linspace(1, 2 ** (N_freqs - 1), N_freqs)

    def forward(self, x):
        """
        Embeds x to (x, sin(2^k x), cos(2^k x), ...) 
        Different from the paper, "x" is also in the output
        See https://github.com/bmild/nerf/issues/12

        Inputs:
            x: (B, self.in_channels)

        Outputs:
            out: (B, self.out_channels)
        """
        out = [x]
        for freq in self.freq_bands:
            for func in self.funcs:
                out += [func(freq * x)]
        return torch.cat(out, -1)


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (AlexNet,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {})),
    (Embedding,
     lambda: ([], {'in_channels': 4, 'N_freqs': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (IndexDecoder,
     lambda: ([], {'input_channels': 4, 'output_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (SqueezeNet,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {})),
    (VGG16,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {})),
    (VectorQuantizer,
     lambda: ([], {'n_e': 4, 'e_dim': 4, 'beta': 4, 'device': 0}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
]

