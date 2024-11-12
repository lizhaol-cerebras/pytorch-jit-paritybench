
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


from typing import Optional


from typing import Union


import numpy as np


import torch


from torch import nn


from typing import Iterator


from typing import Sized


from typing import Dict


from typing import List


import math


import numbers


import random


import warnings


from typing import Sequence


from typing import Tuple


import torchvision.transforms.functional as F


from torchvision import transforms


import torch.distributed as dist


from typing import Iterable


from torch.optim.optimizer import Optimizer


import torch.nn as nn


from torch.nn import functional as F


from functools import reduce


from torch.nn.modules.batchnorm import _BatchNorm


from sklearn.cluster import KMeans


from collections import OrderedDict


from functools import partial


import torch.nn.functional as F


from math import cos


from math import pi


from torch.utils.data import DataLoader


from typing import Any


import time


from scipy.sparse import csr_matrix


import torchvision


from torch.utils.data import Dataset


import logging


import copy


import matplotlib.pyplot as plt


from sklearn.manifold import TSNE


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        """Forward function."""
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    """A faster version of GELU."""

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        """Forward function."""
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    """Residual Attention Block (RAB).

    This module implements the same function as the MultiheadAttention in
    MMClassification, but with a different interface, which is mainly used
    in CLIP.

    Args:
        d_model (int): The feature dimension.
        n_head (int): The number of attention heads.
        attn_mask (torch.Tensor, optional): The attention mask.
            Defaults to None.
    """

    def __init__(self, d_model: 'int', n_head: 'int', attn_mask: 'Optional[torch.Tensor]'=None, return_attention: 'bool'=False) ->None:
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([('c_fc', nn.Linear(d_model, d_model * 4)), ('gelu', QuickGELU()), ('c_proj', nn.Linear(d_model * 4, d_model))]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask
        self.return_attention = return_attention

    def attention(self, x: 'torch.Tensor') ->torch.Tensor:
        """Attention function."""
        self.attn_mask = self.attn_mask if self.attn_mask is not None else None
        if self.return_attention:
            return self.attn(x, x, x, need_weights=self.return_attention, attn_mask=self.attn_mask)
        else:
            return self.attn(x, x, x, need_weights=self.return_attention, attn_mask=self.attn_mask)[0]

    def forward(self, x: 'torch.Tensor') ->Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward function."""
        if self.return_attention:
            x_, attention = self.attention(self.ln_1(x))
            x = x + x_
            x = x + self.mlp(self.ln_2(x))
            return x, attention
        else:
            x = x + self.attention(self.ln_1(x))
            x = x + self.mlp(self.ln_2(x))
            return x


class Transformer(nn.Module):
    """Transformer.

    Both visual and text branches use this transformer.

    Args:
        width (int): The feature dimension.
        layers (int): The number of layers.
        heads (int): The number of attention heads.
        attn_mask (torch.Tensor, optional): The attention mask.
    """

    def __init__(self, width: 'int', layers: 'int', heads: 'int', attn_mask: 'Optional[torch.Tensor]'=None) ->None:
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.ModuleList()
        for _ in range(layers - 1):
            self.resblocks.append(ResidualAttentionBlock(width, heads, attn_mask))
        self.resblocks.append(ResidualAttentionBlock(width, heads, attn_mask, return_attention=True))

    def forward(self, x: 'torch.Tensor') ->Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward function."""
        z = []
        for idx, blk in enumerate(self.resblocks):
            if idx < self.layers - 1:
                x = blk(x)
                z.append(x.permute(1, 0, 2))
            else:
                x, attention = blk(x)
                z.append(x.permute(1, 0, 2))
        return x, attention, z


class VisionTransformer(nn.Module):
    """Vision Transformer for CLIP.

    Args:
        input_resolution (int): The image size.
        patch_size (int): The patch size.
        width (int): The feature dimension.
        layers (int): The number of layers.
        heads (int): The number of attention heads.
        out_dim (int): The output dimension.
        fineturn (bool): Whether to fineturn the model.
        average_target (bool): Whether to average the target.
    """

    def __init__(self, input_resolution: 'int', patch_size: 'int', width: 'int', layers: 'int', heads: 'int', output_dim: 'int', finetune=False, average_targets: 'int'=1) ->None:
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)
        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)
        self.transformer = Transformer(width, layers, heads)
        self.finetune = finetune
        if finetune is False:
            self.ln_post = LayerNorm(width)
            self.proj = nn.Parameter(scale * torch.randn(width, output_dim))
        self.average_targets = average_targets

    def forward(self, x: 'torch.Tensor') ->Tuple[torch.Tensor, torch.Tensor]:
        """Forward function."""
        x = self.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        x = torch.cat([self.class_embedding + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)
        x = x + self.positional_embedding
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)
        x, attention, z = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_post(x)
        if self.proj is not None:
            x = x @ self.proj
        return x, attention


class CLIP(nn.Module):
    """CLIP.

    Args:
        embed_dim (int): The embedding dimension.
        image_resolution (int): The image size.
        vision_layers (int): The number of layers in the vision transformer.
        vision_width (int): The feature dimension in the vision transformer.
        vision_patch_size (int): The patch size in the vision transformer.
        context_length (int): The context length.
        vocab_size (int): The vocabulary size.
        transformer_width (int): The feature dimension in the text transformer.
        transformer_heads (int): The number of attention heads in the
            text transformer.
        transformer_layers (int): The number of layers in the text transformer.
        fineturn (bool): Whether to fineturn the model.
        average_target (bool): Whether to average the target.
    """

    def __init__(self, embed_dim: 'int', image_resolution: 'int', vision_layers: 'Union[Tuple[int, int, int, int], int]', vision_width: 'int', vision_patch_size: 'int', context_length: 'int', vocab_size: 'int', transformer_width: 'int', transformer_heads: 'int', transformer_layers: 'int', finetune: 'bool'=False, average_targets: 'int'=1) ->None:
        super().__init__()
        self.context_length = context_length
        vision_heads = vision_width // 64
        self.visual = VisionTransformer(input_resolution=image_resolution, patch_size=vision_patch_size, width=vision_width, layers=vision_layers, heads=vision_heads, output_dim=embed_dim, finetune=finetune, average_targets=average_targets)
        self.transformer = Transformer(width=transformer_width, layers=transformer_layers, heads=transformer_heads, attn_mask=self.build_attention_mask())
        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)
        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.initialize_parameters()

    def initialize_parameters(self) ->None:
        """Initialize the parameters.

        The pretrained weight will override the initialized parameters by this
        function.
        """
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)
        proj_std = self.transformer.width ** -0.5 * (2 * self.transformer.layers) ** -0.5
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)
        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self) ->torch.Tensor:
        """Build the attention mask."""
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float('-inf'))
        mask.triu_(1)
        return mask

    @property
    def dtype(self) ->torch.dtype:
        """Get the dtype."""
        return self.visual.conv1.weight.dtype

    def encode_image(self, image: 'torch.Tensor') ->Tuple[torch.Tensor, torch.Tensor]:
        """Encode the image.

        Get the feature and attention mask from the last layer of the visual
        branch of CLIP.

        Args:
            image (torch.Tensor): The image tensor with shape NCHW.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The feature and attention mask.
        """
        return self.visual(image.type(self.dtype))


class AvgPool2d(nn.Module):
    """The wrapper for AdaptiveAvgPool2d, which supports tuple input."""

    def __init__(self, output_size: 'int'=1) ->None:
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(output_size)

    def forward(self, x: 'Sequence[torch.Tensor]') ->List[torch.Tensor]:
        """Forward function."""
        assert len(x) == 1
        return [self.avgpool(x[-1])]


def sample_vectors(samples: 'torch.Tensor', num: 'int') ->torch.Tensor:
    """Sample vectors according to the given number."""
    num_samples, device = samples.shape[0], samples.device
    if num_samples >= num:
        indices = torch.randperm(num_samples, device=device)[:num]
    else:
        indices = torch.randint(0, num_samples, (num,), device=device)
    return samples[indices]


def kmeans(samples: 'torch.Tensor', num_clusters: 'int', num_iters: 'int'=10, use_cosine_sim: 'bool'=False) ->Tuple[torch.Tensor, torch.Tensor]:
    """Run k-means algorithm."""
    dim, dtype, _ = samples.shape[-1], samples.dtype, samples.device
    means = sample_vectors(samples, num_clusters)
    for _ in range(num_iters):
        if use_cosine_sim:
            dists = samples @ means.t()
        else:
            diffs = rearrange(samples, 'n d -> n () d') - rearrange(means, 'c d -> () c d')
            dists = -(diffs ** 2).sum(dim=-1)
        buckets = dists.max(dim=-1).indices
        bins = torch.bincount(buckets, minlength=num_clusters)
        zero_mask = bins == 0
        bins_min_clamped = bins.masked_fill(zero_mask, 1)
        new_means = buckets.new_zeros(num_clusters, dim, dtype=dtype)
        new_means.scatter_add_(0, repeat(buckets, 'n -> n d', d=dim), samples)
        new_means = new_means / bins_min_clamped[..., None]
        if use_cosine_sim:
            new_means = F.normalize(new_means, p=2, dim=-1)
        means = torch.where(zero_mask[..., None], means, new_means)
    return means, bins


class EmbeddingEMA(nn.Module):
    """The codebook of embedding vectors.

    Args:
        num_tokens (int): Number of embedding vectors in the codebook.
        codebook_dim (int) : The dimension of embedding vectors in the
            codebook.
        kmeans_init (bool): Whether to use k-means to initialize the
            VectorQuantizer. Defaults to True.
        codebook_init_path (str): The initialization checkpoint for codebook.
            Defaults to None.
    """

    def __init__(self, num_tokens: 'int', codebook_dim: 'int', kmeans_init: 'bool'=True, codebook_init_path: 'Optional[str]'=None):
        super().__init__()
        self.num_tokens = num_tokens
        self.codebook_dim = codebook_dim
        if codebook_init_path is None:
            if not kmeans_init:
                weight = torch.randn(num_tokens, codebook_dim)
                weight = F.normalize(weight, p=2, dim=-1)
            else:
                weight = torch.zeros(num_tokens, codebook_dim)
            self.register_buffer('initted', torch.Tensor([not kmeans_init]))
        else:
            None
            codebook_ckpt_weight = torch.load(codebook_init_path, map_location='cpu')
            weight = codebook_ckpt_weight.clone()
            self.register_buffer('initted', torch.Tensor([True]))
        self.weight = nn.Parameter(weight, requires_grad=False)
        self.update = True

    @torch.jit.ignore
    def init_embed_(self, data: 'torch.Tensor') ->None:
        """Initialize embedding vectors of codebook."""
        if self.initted:
            return
        None
        embed, _ = kmeans(data, self.num_tokens, 10, use_cosine_sim=True)
        self.weight.data.copy_(embed)
        self.initted.data.copy_(torch.Tensor([True]))

    def forward(self, embed_id: 'torch.Tensor') ->torch.Tensor:
        """Get embedding vectors."""
        return F.embedding(embed_id, self.weight)


def ema_inplace(moving_avg: 'torch.Tensor', new: 'torch.Tensor', decay: 'torch.Tensor') ->None:
    """Update moving average."""
    moving_avg.data.mul_(decay).add_(new, alpha=1 - decay)


def norm_ema_inplace(moving_avg: 'torch.Tensor', new: 'torch.Tensor', decay: 'torch.Tensor') ->None:
    """Update moving average with norm data."""
    moving_avg.data.mul_(decay).add_(new, alpha=1 - decay)
    moving_avg.data.copy_(F.normalize(moving_avg.data, p=2, dim=-1))


class NormEMAVectorQuantizer(nn.Module):
    """Normed EMA vector quantizer module.

    Args:
        num_embed (int): Number of embedding vectors in the codebook. Defaults
            to 8192.
        embed_dims (int) : The dimension of embedding vectors in the codebook.
            Defaults to 32.
        beta (float): The mutiplier for VectorQuantizer embedding loss.
            Defaults to 1.
        decay (float): The decay parameter of EMA. Defaults to 0.99.
        statistic_code_usage (bool): Whether to use cluster_size to record
            statistic. Defaults to True.
        kmeans_init (bool): Whether to use k-means to initialize the
            VectorQuantizer. Defaults to True.
        codebook_init_path (str): The initialization checkpoint for codebook.
            Defaults to None.
    """

    def __init__(self, num_embed: 'int', embed_dims: 'int', beta: 'float', decay: 'float'=0.99, statistic_code_usage: 'bool'=True, kmeans_init: 'bool'=True, codebook_init_path: 'Optional[str]'=None) ->None:
        super().__init__()
        self.codebook_dim = embed_dims
        self.num_tokens = num_embed
        self.beta = beta
        self.decay = decay
        self.embedding = EmbeddingEMA(num_tokens=self.num_tokens, codebook_dim=self.codebook_dim, kmeans_init=kmeans_init, codebook_init_path=codebook_init_path)
        self.statistic_code_usage = statistic_code_usage
        if statistic_code_usage:
            self.register_buffer('cluster_size', torch.zeros(num_embed))

    def reset_cluster_size(self, device):
        if self.statistic_code_usage:
            self.register_buffer('cluster_size', torch.zeros(self.num_tokens))
            self.cluster_size = self.cluster_size

    def forward(self, z):
        """Forward function."""
        z = rearrange(z, 'b c h w -> b h w c')
        z = F.normalize(z, p=2, dim=-1)
        z_flattened = z.reshape(-1, self.codebook_dim)
        self.embedding.init_embed_(z_flattened)
        d = z_flattened.pow(2).sum(dim=1, keepdim=True) + self.embedding.weight.pow(2).sum(dim=1) - 2 * torch.einsum('bd,nd->bn', z_flattened, self.embedding.weight)
        encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embedding(encoding_indices).view(z.shape)
        encodings = F.one_hot(encoding_indices, self.num_tokens).type(z.dtype)
        if not self.training:
            with torch.no_grad():
                cluster_size = encodings.sum(0)
                all_reduce(cluster_size)
                ema_inplace(self.cluster_size, cluster_size, self.decay)
        if self.training and self.embedding.update:
            bins = encodings.sum(0)
            all_reduce(bins)
            ema_inplace(self.cluster_size, bins, self.decay)
            zero_mask = bins == 0
            bins = bins.masked_fill(zero_mask, 1.0)
            embed_sum = z_flattened.t() @ encodings
            all_reduce(embed_sum)
            embed_normalized = (embed_sum / bins.unsqueeze(0)).t()
            embed_normalized = F.normalize(embed_normalized, p=2, dim=-1)
            embed_normalized = torch.where(zero_mask[..., None], self.embedding.weight, embed_normalized)
            norm_ema_inplace(self.embedding.weight, embed_normalized, self.decay)
        loss = self.beta * F.mse_loss(z_q.detach(), z)
        z_q = z + (z_q - z).detach()
        z_q = rearrange(z_q, 'b h w c -> b c h w')
        return z_q, loss, encoding_indices


class AliasMethod(nn.Module):
    """The alias method for sampling.

    From: https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/

    Args:
        probs (torch.Tensor): Sampling probabilities.
    """

    def __init__(self, probs: 'torch.Tensor') ->None:
        super().__init__()
        if probs.sum() > 1:
            probs.div_(probs.sum())
        K = len(probs)
        self.register_buffer('prob', torch.zeros(K))
        self.register_buffer('alias', torch.LongTensor([0] * K))
        smaller = []
        larger = []
        for kk, prob in enumerate(probs):
            self.prob[kk] = K * prob
            if self.prob[kk] < 1.0:
                smaller.append(kk)
            else:
                larger.append(kk)
        while len(smaller) > 0 and len(larger) > 0:
            small = smaller.pop()
            large = larger.pop()
            self.alias[small] = large
            self.prob[large] = self.prob[large] - 1.0 + self.prob[small]
            if self.prob[large] < 1.0:
                smaller.append(large)
            else:
                larger.append(large)
        for last_one in (smaller + larger):
            self.prob[last_one] = 1

    def draw(self, N: 'int') ->None:
        """Draw N samples from multinomial.

        Args:
            N (int): Number of samples.

        Returns:
            torch.Tensor: Samples.
        """
        assert N > 0
        K = self.alias.size(0)
        kk = torch.zeros(N, dtype=torch.long, device=self.prob.device).random_(0, K)
        prob = self.prob.index_select(0, kk)
        alias = self.alias.index_select(0, kk)
        b = torch.bernoulli(prob)
        oq = kk.mul(b.long())
        oj = alias.mul((1 - b).long())
        return oq + oj


class ToyViTBackbone(nn.Module):

    def __init__(self):
        super().__init__()
        self.cls_token = nn.Parameter(torch.ones(1))
        self.pos_embed = nn.Parameter(torch.ones(1))
        self.layers = nn.ModuleList()
        for _ in range(2):
            layer = nn.Conv2d(3, 3, 1)
            self.layers.append(layer)


class ToyViT(nn.Module):

    def __init__(self):
        super().__init__()
        self.backbone = ToyViTBackbone()
        self.head = nn.Linear(1, 1)


class ToySwin(nn.Module):

    def __init__(self):
        super().__init__()
        self.backbone = SwinTransformer()
        self.head = nn.Linear(1, 1)


class ToyModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.loss_lambda = 0.5
        self.linear = nn.Linear(2, 1)

    def forward(self, data_batch, return_loss=False):
        inputs, labels = [], []
        for x in data_batch:
            inputs.append(x['inputs'])
            labels.append(x['data_sample'])
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        inputs = torch.stack(inputs)
        labels = torch.stack(labels)
        outputs = self.linear(inputs)
        if return_loss:
            loss = (labels - outputs).sum()
            outputs = dict(loss=loss, log_vars=dict(loss=loss.item()))
            return outputs
        else:
            outputs = dict(log_vars=dict(a=1, b=0.5))
            return outputs


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (LayerNorm,
     lambda: ([], {'normalized_shape': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (QuickGELU,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ResidualAttentionBlock,
     lambda: ([], {'d_model': 4, 'n_head': 4}),
     lambda: ([torch.rand([4, 4])], {})),
    (Transformer,
     lambda: ([], {'width': 4, 'layers': 1, 'heads': 4}),
     lambda: ([torch.rand([4, 4, 4])], {})),
]

