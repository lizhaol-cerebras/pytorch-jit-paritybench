
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


import math


import torch


import torch.nn.functional as F


from torch import nn


from torch import einsum


from torch.nn import Module


from torch.nn import ModuleList


from typing import Tuple


import copy


from functools import wraps


from torch.autograd import grad as torch_grad


import torchvision


from math import sqrt


from random import choice


from torch.utils.data import Dataset


from torch.utils.data import DataLoader


from torch.utils.data import random_split


import torchvision.transforms as T


from torchvision.datasets import ImageFolder


from torchvision.utils import make_grid


from torchvision.utils import save_image


from functools import partial


from typing import List


import numpy as np


from torch.utils.data import DataLoader as PytorchDataLoader


from torchvision import transforms as T


from torchvision import utils


from torch.optim import AdamW


from torch.optim import Adam


import functools


from random import random


from random import choices


from collections import namedtuple


from typing import Iterable


class LayerNorm(Module):

    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer('beta', torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)


class GEGLU(Module):

    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.gelu(gate) * x


def exists(val):
    return val is not None


class AlibiPositionalBias(Module):

    def __init__(self, heads):
        super().__init__()
        self.heads = heads
        slopes = torch.Tensor(self._get_slopes(heads))
        slopes = rearrange(slopes, 'h -> h 1 1')
        self.register_buffer('slopes', slopes, persistent=False)
        self.register_buffer('bias', None, persistent=False)

    def get_bias(self, i, j, device):
        i_arange = torch.arange(j - i, j, device=device)
        j_arange = torch.arange(j, device=device)
        bias = -torch.abs(rearrange(j_arange, 'j -> 1 1 j') - rearrange(i_arange, 'i -> 1 i 1'))
        return bias

    @staticmethod
    def _get_slopes(heads):

        def get_slopes_power_of_2(n):
            start = 2 ** -2 ** -(math.log2(n) - 3)
            ratio = start
            return [(start * ratio ** i) for i in range(n)]
        if math.log2(heads).is_integer():
            return get_slopes_power_of_2(heads)
        closest_power_of_2 = 2 ** math.floor(math.log2(heads))
        return get_slopes_power_of_2(closest_power_of_2) + get_slopes_power_of_2(2 * closest_power_of_2)[0::2][:heads - closest_power_of_2]

    def forward(self, sim):
        h, i, j, device = *sim.shape[-3:], sim.device
        if exists(self.bias) and self.bias.shape[-1] >= j:
            return self.bias[..., :i, :j]
        bias = self.get_bias(i, j, device)
        bias = bias * self.slopes
        num_heads_unalibied = h - bias.shape[0]
        bias = F.pad(bias, (0, 0, 0, 0, 0, num_heads_unalibied))
        self.register_buffer('bias', bias, persistent=False)
        return self.bias


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def l2norm(t):
    return F.normalize(t, dim=-1)


class Attention(Module):

    def __init__(self, dim, dim_context=None, dim_head=64, heads=8, causal=False, num_null_kv=0, norm_context=True, dropout=0.0, scale=8):
        super().__init__()
        self.heads = heads
        self.causal = causal
        self.scale = scale
        inner_dim = dim_head * heads
        dim_context = default(dim_context, dim)
        if causal:
            self.rel_pos_bias = AlibiPositionalBias(heads=heads)
        self.attn_dropout = nn.Dropout(dropout)
        self.norm = LayerNorm(dim)
        self.context_norm = LayerNorm(dim_context) if norm_context else nn.Identity()
        self.num_null_kv = num_null_kv
        self.null_kv = nn.Parameter(torch.randn(heads, 2 * num_null_kv, dim_head))
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim_context, inner_dim * 2, bias=False)
        self.q_scale = nn.Parameter(torch.ones(dim_head))
        self.k_scale = nn.Parameter(torch.ones(dim_head))
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x, mask=None, context=None, attn_bias=None):
        batch, device, dtype = x.shape[0], x.device, x.dtype
        if exists(context):
            context = self.context_norm(context)
        kv_input = default(context, x)
        x = self.norm(x)
        q, k, v = self.to_q(x), *self.to_kv(kv_input).chunk(2, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), (q, k, v))
        nk, nv = repeat(self.null_kv, 'h (n r) d -> b h n r d', b=batch, r=2).unbind(dim=-2)
        k = torch.cat((nk, k), dim=-2)
        v = torch.cat((nv, v), dim=-2)
        q, k = map(l2norm, (q, k))
        q = q * self.q_scale
        k = k * self.k_scale
        sim = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        i, j = sim.shape[-2:]
        if exists(attn_bias):
            attn_bias = F.pad(attn_bias, (self.num_null_kv, 0), value=0.0)
            sim = sim + attn_bias
        if exists(mask):
            mask = F.pad(mask, (self.num_null_kv, 0), value=True)
            mask = rearrange(mask, 'b j -> b 1 1 j')
            sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)
        if self.causal:
            sim = sim + self.rel_pos_bias(sim)
            causal_mask = torch.ones((i, j), device=device, dtype=torch.bool).triu(j - i + 1)
            sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)
        attn = sim.softmax(dim=-1)
        attn = self.attn_dropout(attn)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


def leaky_relu(p=0.1):
    return nn.LeakyReLU(p)


class ContinuousPositionBias(Module):
    """ from https://arxiv.org/abs/2111.09883 """

    def __init__(self, *, dim, heads, num_dims=2, layers=2, log_dist=True, cache_rel_pos=False):
        super().__init__()
        self.num_dims = num_dims
        self.log_dist = log_dist
        self.net = ModuleList([])
        self.net.append(nn.Sequential(nn.Linear(self.num_dims, dim), leaky_relu()))
        for _ in range(layers - 1):
            self.net.append(nn.Sequential(nn.Linear(dim, dim), leaky_relu()))
        self.net.append(nn.Linear(dim, heads))
        self.cache_rel_pos = cache_rel_pos
        self.register_buffer('rel_pos', None, persistent=False)

    def forward(self, *dimensions, device=torch.device('cpu')):
        if not exists(self.rel_pos) or not self.cache_rel_pos:
            positions = [torch.arange(d, device=device) for d in dimensions]
            grid = torch.stack(torch.meshgrid(*positions, indexing='ij'))
            grid = rearrange(grid, 'c ... -> (...) c')
            rel_pos = rearrange(grid, 'i c -> i 1 c') - rearrange(grid, 'j c -> 1 j c')
            if self.log_dist:
                rel_pos = torch.sign(rel_pos) * torch.log(rel_pos.abs() + 1)
            self.register_buffer('rel_pos', rel_pos, persistent=False)
        rel_pos = self.rel_pos.float()
        for layer in self.net:
            rel_pos = layer(rel_pos)
        return rearrange(rel_pos, 'i j h -> h i j')


def FeedForward(dim, mult=4, dropout=0.0):
    inner_dim = int(mult * (2 / 3) * dim)
    return nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, inner_dim * 2, bias=False), GEGLU(), nn.Dropout(dropout), nn.Linear(inner_dim, dim, bias=False))


class DiscriminatorBlock(Module):

    def __init__(self, input_channels, filters, downsample=True):
        super().__init__()
        self.conv_res = nn.Conv2d(input_channels, filters, 1, stride=2 if downsample else 1)
        self.net = nn.Sequential(nn.Conv2d(input_channels, filters, 3, padding=1), leaky_relu(), nn.Conv2d(filters, filters, 3, padding=1), leaky_relu())
        self.downsample = nn.Sequential(Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w', p1=2, p2=2), nn.Conv2d(filters * 4, filters, 1)) if downsample else None

    def forward(self, x):
        res = self.conv_res(x)
        x = self.net(x)
        if exists(self.downsample):
            x = self.downsample(x)
        x = (x + res) * (1 / math.sqrt(2))
        return x


def cast_tuple(val, length=1):
    return val if isinstance(val, tuple) else (val,) * length


def pair(val):
    return val if isinstance(val, tuple) else (val, val)


class Discriminator(Module):

    def __init__(self, *, dim, image_size, channels=3, attn_res_layers=(16,), max_dim=512):
        super().__init__()
        image_size = pair(image_size)
        min_image_resolution = min(image_size)
        num_layers = int(math.log2(min_image_resolution) - 2)
        attn_res_layers = cast_tuple(attn_res_layers, num_layers)
        blocks = []
        layer_dims = [channels] + [(dim * 4 * 2 ** i) for i in range(num_layers + 1)]
        layer_dims = [min(layer_dim, max_dim) for layer_dim in layer_dims]
        layer_dims_in_out = tuple(zip(layer_dims[:-1], layer_dims[1:]))
        blocks = []
        attn_blocks = []
        image_resolution = min_image_resolution
        for ind, (in_chan, out_chan) in enumerate(layer_dims_in_out):
            num_layer = ind + 1
            is_not_last = ind != len(layer_dims_in_out) - 1
            block = DiscriminatorBlock(in_chan, out_chan, downsample=is_not_last)
            blocks.append(block)
            attn_block = None
            if image_resolution in attn_res_layers:
                attn_block = Attention(dim=out_chan)
            attn_blocks.append(attn_block)
            image_resolution //= 2
        self.blocks = ModuleList(blocks)
        self.attn_blocks = ModuleList(attn_blocks)
        dim_last = layer_dims[-1]
        downsample_factor = 2 ** num_layers
        last_fmap_size = tuple(map(lambda n: n // downsample_factor, image_size))
        latent_dim = last_fmap_size[0] * last_fmap_size[1] * dim_last
        self.to_logits = nn.Sequential(nn.Conv2d(dim_last, dim_last, 3, padding=1), leaky_relu(), Rearrange('b ... -> b (...)'), nn.Linear(latent_dim, 1), Rearrange('b 1 -> b'))

    def forward(self, x):
        for block, attn_block in zip(self.blocks, self.attn_blocks):
            x = block(x)
            if exists(attn_block):
                x, ps = pack([x], 'b c *')
                x = rearrange(x, 'b c n -> b n c')
                x = attn_block(x) + x
                x = rearrange(x, 'b n c -> b c n')
                x, = unpack(x, ps, 'b c *')
        return self.to_logits(x)


def log(t, eps=1e-10):
    return torch.log(t + eps)


def bce_discr_loss(fake, real):
    return (-log(1 - torch.sigmoid(fake)) - log(torch.sigmoid(real))).mean()


def bce_gen_loss(fake):
    return -log(torch.sigmoid(fake)).mean()


def divisible_by(numer, denom):
    return numer % denom == 0


def grad_layer_wrt_loss(loss, layer):
    return torch_grad(outputs=loss, inputs=layer, grad_outputs=torch.ones_like(loss), retain_graph=True)[0].detach()


def gradient_penalty(images, output, weight=10):
    batch_size = images.shape[0]
    gradients = torch_grad(outputs=output, inputs=images, grad_outputs=torch.ones(output.size(), device=images.device), create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = rearrange(gradients, 'b ... -> b (...)')
    return weight * ((gradients.norm(2, dim=1) - 1) ** 2).mean()


def hinge_discr_loss(fake, real):
    return (F.relu(1 + fake) + F.relu(1 - real)).mean()


def hinge_gen_loss(fake):
    return -fake.mean()


def pick_video_frame(video, frame_indices):
    batch, device = video.shape[0], video.device
    video = rearrange(video, 'b c f ... -> b f c ...')
    batch_indices = torch.arange(batch, device=device)
    batch_indices = rearrange(batch_indices, 'b -> b 1')
    images = video[batch_indices, frame_indices]
    images = rearrange(images, 'b 1 c ... -> b c ...')
    return images


def remove_vgg(fn):

    @wraps(fn)
    def inner(self, *args, **kwargs):
        has_vgg = hasattr(self, 'vgg')
        if has_vgg:
            vgg = self.vgg
            delattr(self, 'vgg')
        out = fn(self, *args, **kwargs)
        if has_vgg:
            self.vgg = vgg
        return out
    return inner


def safe_div(numer, denom, eps=1e-08):
    return numer / (denom + eps)


class CViViT(Module):

    def __init__(self, *, dim, codebook_size, image_size, patch_size, temporal_patch_size, spatial_depth, temporal_depth, discr_base_dim=16, dim_head=64, heads=8, channels=3, use_vgg_and_gan=True, vgg=None, discr_attn_res_layers=(16,), use_hinge_loss=True, attn_dropout=0.0, ff_dropout=0.0, lookup_free_quantization=True, lookup_free_quantization_kwargs: dict={}):
        """
        einstein notations:

        b - batch
        c - channels
        t - time
        d - feature dimension
        p1, p2, pt - image patch sizes and then temporal patch size
        """
        super().__init__()
        self.image_size = pair(image_size)
        self.patch_size = pair(patch_size)
        patch_height, patch_width = self.patch_size
        self.temporal_patch_size = temporal_patch_size
        self.spatial_rel_pos_bias = ContinuousPositionBias(dim=dim, heads=heads)
        image_height, image_width = self.image_size
        assert image_height % patch_height == 0 and image_width % patch_width == 0
        self.to_patch_emb_first_frame = nn.Sequential(Rearrange('b c 1 (h p1) (w p2) -> b 1 h w (c p1 p2)', p1=patch_height, p2=patch_width), nn.LayerNorm(channels * patch_width * patch_height), nn.Linear(channels * patch_width * patch_height, dim), nn.LayerNorm(dim))
        self.to_patch_emb = nn.Sequential(Rearrange('b c (t pt) (h p1) (w p2) -> b t h w (c pt p1 p2)', p1=patch_height, p2=patch_width, pt=temporal_patch_size), nn.LayerNorm(channels * patch_width * patch_height * temporal_patch_size), nn.Linear(channels * patch_width * patch_height * temporal_patch_size, dim), nn.LayerNorm(dim))
        spatial_transformer_kwargs = dict(dim=dim, dim_head=dim_head, heads=heads, attn_dropout=attn_dropout, ff_dropout=ff_dropout, causal=False, peg=False)
        temporal_transformer_kwargs = dict(dim=dim, dim_head=dim_head, heads=heads, attn_dropout=attn_dropout, ff_dropout=ff_dropout, causal=True, peg=True, peg_causal=True)
        self.enc_spatial_transformer = Transformer(depth=spatial_depth, **spatial_transformer_kwargs)
        self.enc_temporal_transformer = Transformer(depth=temporal_depth, **temporal_transformer_kwargs)
        self.lookup_free_quantization = lookup_free_quantization
        if lookup_free_quantization:
            self.vq = LFQ(dim=dim, codebook_size=codebook_size, **lookup_free_quantization_kwargs)
        else:
            self.vq = VectorQuantize(dim=dim, codebook_size=codebook_size, use_cosine_sim=True)
        self.dec_spatial_transformer = Transformer(depth=spatial_depth, **spatial_transformer_kwargs)
        self.dec_temporal_transformer = Transformer(depth=temporal_depth, **temporal_transformer_kwargs)
        self.to_pixels_first_frame = nn.Sequential(nn.Linear(dim, channels * patch_width * patch_height), Rearrange('b 1 h w (c p1 p2) -> b c 1 (h p1) (w p2)', p1=patch_height, p2=patch_width))
        self.to_pixels = nn.Sequential(nn.Linear(dim, channels * patch_width * patch_height * temporal_patch_size), Rearrange('b t h w (c pt p1 p2) -> b c (t pt) (h p1) (w p2)', p1=patch_height, p2=patch_width, pt=temporal_patch_size))
        self.vgg = None
        self.discr = None
        self.use_vgg_and_gan = use_vgg_and_gan
        if not use_vgg_and_gan:
            return
        if exists(vgg):
            self.vgg = vgg
        else:
            self.vgg = torchvision.models.vgg16(pretrained=True)
            self.vgg.classifier = nn.Sequential(*self.vgg.classifier[:-2])
        self.discr = Discriminator(image_size=self.image_size, dim=discr_base_dim, channels=channels, attn_res_layers=discr_attn_res_layers)
        self.discr_loss = hinge_discr_loss if use_hinge_loss else bce_discr_loss
        self.gen_loss = hinge_gen_loss if use_hinge_loss else bce_gen_loss

    def calculate_video_token_mask(self, videos, video_frame_mask):
        *_, h, w = videos.shape
        ph, pw = self.patch_size
        assert torch.all((video_frame_mask.sum(dim=-1) - 1) % self.temporal_patch_size == 0), 'number of frames must be divisible by temporal patch size, subtracting off the first frame'
        first_frame_mask, rest_frame_mask = video_frame_mask[:, :1], video_frame_mask[:, 1:]
        rest_vq_mask = rearrange(rest_frame_mask, 'b (f p) -> b f p', p=self.temporal_patch_size)
        video_mask = torch.cat((first_frame_mask, rest_vq_mask.any(dim=-1)), dim=-1)
        return repeat(video_mask, 'b f -> b (f hw)', hw=h // ph * (w // pw))

    def get_video_patch_shape(self, num_frames, include_first_frame=True):
        patch_frames = 0
        if include_first_frame:
            num_frames -= 1
            patch_frames += 1
        patch_frames += num_frames // self.temporal_patch_size
        return patch_frames, *self.patch_height_width

    @property
    def image_num_tokens(self):
        return int(self.image_size[0] / self.patch_size[0]) * int(self.image_size[1] / self.patch_size[1])

    def frames_per_num_tokens(self, num_tokens):
        tokens_per_frame = self.image_num_tokens
        assert num_tokens % tokens_per_frame == 0, f'number of tokens must be divisible by number of tokens per frame {tokens_per_frame}'
        assert num_tokens > 0
        pseudo_frames = num_tokens // tokens_per_frames
        return (pseudo_frames - 1) * self.temporal_patch_size + 1

    def num_tokens_per_frames(self, num_frames, include_first_frame=True):
        image_num_tokens = self.image_num_tokens
        total_tokens = 0
        if include_first_frame:
            num_frames -= 1
            total_tokens += image_num_tokens
        assert num_frames % self.temporal_patch_size == 0
        return total_tokens + int(num_frames / self.temporal_patch_size) * image_num_tokens

    def copy_for_eval(self):
        device = next(self.parameters()).device
        vae_copy = copy.deepcopy(self.cpu())
        if vae_copy.use_vgg_and_gan:
            del vae_copy.discr
            del vae_copy.vgg
        vae_copy.eval()
        return vae_copy

    @remove_vgg
    def state_dict(self, *args, **kwargs):
        return super().state_dict(*args, **kwargs)

    @remove_vgg
    def load_state_dict(self, *args, **kwargs):
        return super().load_state_dict(*args, **kwargs)

    def load(self, path):
        path = Path(path)
        assert path.exists()
        pt = torch.load(str(path))
        self.load_state_dict(pt)

    def decode_from_codebook_indices(self, indices):
        if self.lookup_free_quantization:
            codes = self.vq.indices_to_codes(indices)
        else:
            codes = self.vq.codebook[indices]
        return self.decode(codes)

    @property
    def patch_height_width(self):
        return self.image_size[0] // self.patch_size[0], self.image_size[1] // self.patch_size[1]

    def encode(self, tokens):
        b = tokens.shape[0]
        h, w = self.patch_height_width
        video_shape = tuple(tokens.shape[:-1])
        tokens = rearrange(tokens, 'b t h w d -> (b t) (h w) d')
        attn_bias = self.spatial_rel_pos_bias(h, w, device=tokens.device)
        tokens = self.enc_spatial_transformer(tokens, attn_bias=attn_bias, video_shape=video_shape)
        tokens = rearrange(tokens, '(b t) (h w) d -> b t h w d', b=b, h=h, w=w)
        tokens = rearrange(tokens, 'b t h w d -> (b h w) t d')
        tokens = self.enc_temporal_transformer(tokens, video_shape=video_shape)
        tokens = rearrange(tokens, '(b h w) t d -> b t h w d', b=b, h=h, w=w)
        return tokens

    def decode(self, tokens):
        b = tokens.shape[0]
        h, w = self.patch_height_width
        if tokens.ndim == 3:
            tokens = rearrange(tokens, 'b (t h w) d -> b t h w d', h=h, w=w)
        video_shape = tuple(tokens.shape[:-1])
        tokens = rearrange(tokens, 'b t h w d -> (b h w) t d')
        tokens = self.dec_temporal_transformer(tokens, video_shape=video_shape)
        tokens = rearrange(tokens, '(b h w) t d -> b t h w d', b=b, h=h, w=w)
        tokens = rearrange(tokens, 'b t h w d -> (b t) (h w) d')
        attn_bias = self.spatial_rel_pos_bias(h, w, device=tokens.device)
        tokens = self.dec_spatial_transformer(tokens, attn_bias=attn_bias, video_shape=video_shape)
        tokens = rearrange(tokens, '(b t) (h w) d -> b t h w d', b=b, h=h, w=w)
        first_frame_token, rest_frames_tokens = tokens[:, :1], tokens[:, 1:]
        first_frame = self.to_pixels_first_frame(first_frame_token)
        rest_frames = self.to_pixels(rest_frames_tokens)
        recon_video = torch.cat((first_frame, rest_frames), dim=2)
        return recon_video

    def forward(self, video, mask=None, return_recons=False, return_recons_only=False, return_discr_loss=False, apply_grad_penalty=True, return_only_codebook_ids=False):
        assert video.ndim in {4, 5}
        is_image = video.ndim == 4
        if is_image:
            video = rearrange(video, 'b c h w -> b c 1 h w')
            assert not exists(mask)
        b, c, f, *image_dims, device = *video.shape, video.device
        assert tuple(image_dims) == self.image_size
        assert not exists(mask) or mask.shape[-1] == f
        assert divisible_by(f - 1, self.temporal_patch_size), f'number of frames ({f}) minus one ({f - 1}) must be divisible by temporal patch size ({self.temporal_patch_size})'
        first_frame, rest_frames = video[:, :, :1], video[:, :, 1:]
        first_frame_tokens = self.to_patch_emb_first_frame(first_frame)
        rest_frames_tokens = self.to_patch_emb(rest_frames)
        tokens = torch.cat((first_frame_tokens, rest_frames_tokens), dim=1)
        shape = tokens.shape
        *_, h, w, _ = shape
        tokens = self.encode(tokens)
        tokens, packed_fhw_shape = pack([tokens], 'b * d')
        vq_mask = None
        if exists(mask):
            vq_mask = self.calculate_video_token_mask(video, mask)
        vq_kwargs = dict(mask=vq_mask) if not self.lookup_free_quantization else dict()
        tokens, indices, vq_aux_loss = self.vq(tokens, **vq_kwargs)
        if return_only_codebook_ids:
            indices, = unpack(indices, packed_fhw_shape, 'b *')
            return indices
        tokens = rearrange(tokens, 'b (t h w) d -> b t h w d', h=h, w=w)
        recon_video = self.decode(tokens)
        returned_recon = rearrange(recon_video, 'b c 1 h w -> b c h w') if is_image else recon_video.clone()
        if return_recons_only:
            return returned_recon
        if exists(mask):
            recon_loss = F.mse_loss(video, recon_video, reduction='none')
            recon_loss = recon_loss[repeat(mask, 'b t -> b c t', c=c)]
            recon_loss = recon_loss.mean()
        else:
            recon_loss = F.mse_loss(video, recon_video)
        pick_frame_logits = torch.randn(b, f)
        if exists(mask):
            mask_value = -torch.finfo(pick_frame_logits.dtype).max
            pick_frame_logits = pick_frame_logits.masked_fill(~mask, mask_value)
        frame_indices = pick_frame_logits.topk(1, dim=-1).indices
        if return_discr_loss:
            assert exists(self.discr), 'discriminator must exist to train it'
            video = pick_video_frame(video, frame_indices)
            recon_video = pick_video_frame(recon_video, frame_indices)
            recon_video = recon_video.detach()
            video.requires_grad_()
            recon_video_discr_logits, video_discr_logits = map(self.discr, (recon_video, video))
            discr_loss = self.discr_loss(recon_video_discr_logits, video_discr_logits)
            if apply_grad_penalty:
                gp = gradient_penalty(video, video_discr_logits)
                loss = discr_loss + gp
            if return_recons:
                return loss, returned_recon
            return loss
        if not self.use_vgg_and_gan:
            if return_recons:
                return recon_loss, returned_recon
            return recon_loss
        input_vgg_input = pick_video_frame(video, frame_indices)
        recon_vgg_input = pick_video_frame(recon_video, frame_indices)
        if video.shape[1] == 1:
            input_vgg_input, recon_vgg_input = map(lambda t: repeat(t, 'b 1 ... -> b c ...', c=3), (img_vgg_input, fmap_vgg_input))
        input_vgg_feats = self.vgg(input_vgg_input)
        recon_vgg_feats = self.vgg(recon_vgg_input)
        perceptual_loss = F.mse_loss(input_vgg_feats, recon_vgg_feats)
        gen_loss = self.gen_loss(self.discr(recon_vgg_input))
        last_dec_layer = self.to_pixels[0].weight
        norm_grad_wrt_gen_loss = grad_layer_wrt_loss(gen_loss, last_dec_layer).norm(p=2)
        norm_grad_wrt_perceptual_loss = grad_layer_wrt_loss(perceptual_loss, last_dec_layer).norm(p=2)
        adaptive_weight = safe_div(norm_grad_wrt_perceptual_loss, norm_grad_wrt_gen_loss)
        adaptive_weight.clamp_(max=10000.0)
        loss = recon_loss + perceptual_loss + vq_aux_loss + adaptive_weight * gen_loss
        if return_recons:
            return loss, returned_recon
        return loss


class ImageDataset(Dataset):

    def __init__(self, folder, image_size, exts=['jpg', 'jpeg', 'png']):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.paths = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]
        None
        self.transform = T.Compose([T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img), T.Resize(image_size), T.RandomHorizontalFlip(), T.CenterCrop(image_size), T.ToTensor()])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        return self.transform(img)


def cast_num_frames(t, *, frames):
    f = t.shape[1]
    if f == frames:
        return t
    if f > frames:
        return t[:, :frames]
    return F.pad(t, (0, 0, 0, 0, 0, frames - f))


CHANNELS_TO_MODE = {(1): 'L', (3): 'RGB', (4): 'RGBA'}


def seek_all_images(img, channels=3):
    assert channels in CHANNELS_TO_MODE, f'channels {channels} invalid'
    mode = CHANNELS_TO_MODE[channels]
    i = 0
    while True:
        try:
            img.seek(i)
            yield img.convert(mode)
        except EOFError:
            break
        i += 1


def gif_to_tensor(path, channels=3, transform=T.ToTensor()):
    img = Image.open(path)
    tensors = tuple(map(transform, seek_all_images(img, channels=channels)))
    return torch.stack(tensors, dim=1)


def identity(t, *args, **kwargs):
    return t


def crop_center(img, cropx, cropy) ->torch.Tensor:
    y, x, c = img.shape
    startx = x // 2 - cropx // 2
    starty = y // 2 - cropy // 2
    return img[starty:starty + cropy, startx:startx + cropx, :]


def video_to_tensor(path: 'str', num_frames=-1, crop_size=None) ->torch.Tensor:
    video = cv2.VideoCapture(path)
    frames = []
    check = True
    while check:
        check, frame = video.read()
        if not check:
            continue
        if exists(crop_size):
            frame = crop_center(frame, *pair(crop_size))
        frames.append(rearrange(frame, '... -> 1 ...'))
    frames = np.array(np.concatenate(frames[:-1], axis=0))
    frames = rearrange(frames, 'f h w c -> c f h w')
    frames_torch = torch.tensor(frames).float()
    return frames_torch[:, :num_frames, :, :]


class VideoDataset(Dataset):

    def __init__(self, folder, image_size, channels=3, num_frames=17, horizontal_flip=False, force_num_frames=True, exts=['gif', 'mp4']):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.channels = channels
        self.paths = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]
        self.transform = T.Compose([T.Resize(image_size), T.RandomHorizontalFlip() if horizontal_flip else T.Lambda(identity), T.CenterCrop(image_size), T.ToTensor()])
        self.gif_to_tensor = partial(gif_to_tensor, channels=self.channels, transform=self.transform)
        self.mp4_to_tensor = partial(video_to_tensor, crop_size=self.image_size)
        self.cast_num_frames_fn = partial(cast_num_frames, frames=num_frames) if force_num_frames else identity

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        ext = path.suffix
        if ext == '.gif':
            tensor = self.gif_to_tensor(path)
        elif ext == '.mp4':
            tensor = self.mp4_to_tensor(str(path))
        else:
            raise ValueError(f'unknown extension {ext}')
        return self.cast_num_frames_fn(tensor)


def accum_log(log, new_logs):
    for key, new_value in new_logs.items():
        old_value = log.get(key, 0.0)
        log[key] = old_value + new_value
    return log


def cycle(dl):
    while True:
        for data in dl:
            yield data


def separate_weight_decayable_params(params):
    wd_params, no_wd_params = [], []
    for param in params:
        param_list = no_wd_params if param.ndim < 2 else wd_params
        param_list.append(param)
    return wd_params, no_wd_params


def get_optimizer(params, lr=0.0001, wd=0.01, betas=(0.9, 0.99), eps=1e-08, filter_by_requires_grad=False, group_wd_params=True, **kwargs):
    if filter_by_requires_grad:
        params = list(filter(lambda t: t.requires_grad, params))
    if wd == 0:
        return Adam(params, lr=lr, betas=betas, eps=eps)
    if group_wd_params:
        wd_params, no_wd_params = separate_weight_decayable_params(params)
        params = [{'params': wd_params}, {'params': no_wd_params, 'weight_decay': 0}]
    return AdamW(params, lr=lr, weight_decay=wd, betas=betas, eps=eps)


def noop(*args, **kwargs):
    pass


def video_tensor_to_gif(tensor, path, duration=120, loop=0, optimize=True):
    images = map(T.ToPILImage(), tensor.unbind(dim=1))
    first_img, *rest_imgs = images
    first_img.save(path, save_all=True, append_images=rest_imgs, duration=duration, loop=loop, optimize=optimize)
    return images


def yes_or_no(question):
    answer = input(f'{question} (y/n) ')
    return answer.lower() in ('yes', 'y')


def prob_mask_like(shape, prob, device):
    if prob == 1:
        return torch.ones(shape, device=device, dtype=torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device=device, dtype=torch.bool)
    else:
        return torch.zeros(shape, device=device).float().uniform_(0, 1) < prob


class MaskGit(Module):

    def __init__(self, *, dim, num_tokens, max_seq_len, gradient_shrink_alpha=0.1, heads=8, dim_head=64, unconditional=False, attn_dropout=0.0, ff_dropout=0.0, **kwargs):
        super().__init__()
        self.dim = dim
        self.mask_id = num_tokens
        self.unconditional = unconditional
        self.token_emb = nn.Embedding(num_tokens + 1, dim)
        self.max_seq_len = max_seq_len
        self.pos_emb = nn.Embedding(max_seq_len, dim)
        self.gradient_shrink_alpha = gradient_shrink_alpha
        self.continuous_pos_bias = ContinuousPositionBias(dim=dim_head, heads=heads, num_dims=3)
        self.transformer = Transformer(dim=dim, attn_num_null_kv=2, has_cross_attn=not self.unconditional, dim_head=dim_head, heads=heads, attn_dropout=attn_dropout, ff_dropout=ff_dropout, peg=True, **kwargs)
        self.to_logits = nn.Linear(dim, num_tokens)

    def forward_with_cond_scale(self, *args, cond_scale=3, **kwargs):
        logits = self.forward(*args, cond_drop_prob=0.0, **kwargs)
        if cond_scale == 1:
            return logits
        null_logits = self.forward(*args, cond_drop_prob=1.0, **kwargs)
        return null_logits + (logits - null_logits) * cond_scale

    def forward(self, x, cond_drop_prob=0.0, text_mask=None, video_mask=None, video_patch_shape=None, return_embeds=False, **kwargs):
        assert x.ndim in {2, 4}, 'video token ids must be of shape (batch, seq) or (batch, frame, height, width)'
        if x.ndim == 4:
            video_patch_shape = x.shape[1:]
            x = rearrange(x, 'b ... -> b (...)')
        b, n, device = *x.shape, x.device
        if not exists(text_mask):
            text_mask = torch.ones((b, n), device=device, dtype=torch.bool)
        assert exists(video_patch_shape), 'video patch shape must be given'
        rel_pos_bias = self.continuous_pos_bias(*video_patch_shape, device=device)
        if cond_drop_prob > 0:
            keep_mask = prob_mask_like((b,), 1 - cond_drop_prob, device=device)
            text_mask = rearrange(keep_mask, 'b -> b 1') & text_mask
        video_shape = b, *video_patch_shape
        x = self.token_emb(x)
        assert n <= self.max_seq_len, f'the video token sequence length you are passing in ({n}) is greater than the `max_seq_len` ({self.max_seq_len}) set on your `MaskGit`'
        x = self.pos_emb(torch.arange(n, device=device)) + x
        x = x * self.gradient_shrink_alpha + x.detach() * (1 - self.gradient_shrink_alpha)
        x = self.transformer(x, video_shape=video_shape, attn_bias=rel_pos_bias, self_attn_mask=video_mask, cross_attn_context_mask=text_mask, **kwargs)
        if return_embeds:
            return x
        return self.to_logits(x)


class TokenCritic(Module):

    def __init__(self, *, dim, num_tokens, max_seq_len, has_cross_attn=False, attn_dropout=0.0, ff_dropout=0.0, **kwargs):
        super().__init__()
        self.has_cross_attn = has_cross_attn
        self.mask_id = num_tokens
        self.token_emb = nn.Embedding(num_tokens + 1, dim)
        self.pos_emb = nn.Embedding(max_seq_len, dim)
        self.transformer = Transformer(dim=dim, peg=True, attn_dropout=attn_dropout, ff_dropout=ff_dropout, has_cross_attn=has_cross_attn, **kwargs)
        self.to_logits = nn.Sequential(nn.Linear(dim, 1), Rearrange('... 1 -> ...'))

    def forward_with_cond_scale(self, *args, cond_scale=3, **kwargs):
        logits = self.forward(*args, cond_drop_prob=0.0, **kwargs)
        if cond_scale == 1:
            return logits
        null_logits = self.forward(*args, cond_drop_prob=1.0, **kwargs)
        return null_logits + (logits - null_logits) * cond_scale

    def forward(self, x, text_mask=None, cond_drop_prob=None, context=None, video_mask=None, video_patch_shape=None, **kwargs):
        if exists(video_patch_shape):
            video_shape = x.shape[0], *video_patch_shape
        else:
            video_shape = x.shape
        x = rearrange(x, 'b ... -> b (...)')
        b, n, device = *x.shape, x.device
        if not exists(text_mask):
            text_mask = torch.ones((b, n), device=device, dtype=torch.bool)
        if exists(context) and cond_drop_prob > 0:
            keep_mask = prob_mask_like((b,), 1 - cond_drop_prob, device=device)
            text_mask = rearrange(keep_mask, 'b -> b 1') & text_mask
        x = self.token_emb(x)
        x = self.pos_emb(torch.arange(n, device=device)) + x
        x = self.transformer(x, video_shape=video_shape, context=context, self_attn_mask=video_mask, cross_attn_context_mask=text_mask, **kwargs)
        return self.to_logits(x)


DEFAULT_T5_NAME = 'google/t5-v1_1-base'


def eval_decorator(fn):

    def inner(model, *args, **kwargs):
        was_training = model.training
        model.eval()
        out = fn(model, *args, **kwargs)
        model.train(was_training)
        return out
    return inner


T5_CONFIGS = {}


def get_encoded_dim(name):
    if name not in T5_CONFIGS:
        config = T5Config.from_pretrained(name)
        T5_CONFIGS[name] = dict(config=config)
    elif 'config' in T5_CONFIGS[name]:
        config = T5_CONFIGS[name]['config']
    elif 'model' in T5_CONFIGS[name]:
        config = T5_CONFIGS[name]['model'].config
    else:
        raise ValueError(f'unknown t5 name {name}')
    return config.d_model


def get_mask_subset_with_prob(mask, prob):
    batch, seq_len, device = *mask.shape, mask.device
    num_tokens = mask.sum(dim=-1)
    num_pads = seq_len - num_tokens
    num_masked = (prob * num_tokens).round().clamp(min=1)
    randperm_indices = torch.rand((batch, seq_len), device=device).argsort(dim=-1)
    randperm_indices -= rearrange(num_pads, 'b -> b 1')
    randperm_indices.masked_fill_(randperm_indices < 0, seq_len)
    mask_subset = randperm_indices < rearrange(num_masked, 'b -> b 1')
    return mask_subset


def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))


def gumbel_sample(t, temperature=1.0, dim=-1):
    return (t / max(temperature, 1e-10) + gumbel_noise(t)).argmax(dim=dim)


MAX_LENGTH = 256


def get_model(name):
    model = T5EncoderModel.from_pretrained(name)
    return model


def get_tokenizer(name):
    tokenizer = T5Tokenizer.from_pretrained(name)
    return tokenizer


def get_model_and_tokenizer(name):
    global T5_CONFIGS
    if name not in T5_CONFIGS:
        T5_CONFIGS[name] = dict()
    if 'model' not in T5_CONFIGS[name]:
        T5_CONFIGS[name]['model'] = get_model(name)
    if 'tokenizer' not in T5_CONFIGS[name]:
        T5_CONFIGS[name]['tokenizer'] = get_tokenizer(name)
    return T5_CONFIGS[name]['model'], T5_CONFIGS[name]['tokenizer']


def t5_encode_text(texts, name=DEFAULT_T5_NAME, output_device=None):
    t5, tokenizer = get_model_and_tokenizer(name)
    if torch.cuda.is_available():
        t5 = t5
    device = next(t5.parameters()).device
    encoded = tokenizer.batch_encode_plus(texts, return_tensors='pt', padding='longest', max_length=MAX_LENGTH, truncation=True)
    input_ids = encoded.input_ids
    attn_mask = encoded.attention_mask
    t5.eval()
    with torch.no_grad():
        output = t5(input_ids=input_ids, attention_mask=attn_mask)
        encoded_text = output.last_hidden_state.detach()
    attn_mask = attn_mask[..., None].bool()
    if not exists(output_device):
        encoded_text = encoded_text.masked_fill(~attn_mask, 0.0)
        return encoded_text
    encoded_text = encoded_text
    attn_mask = attn_mask
    encoded_text = encoded_text.masked_fill(~attn_mask, 0.0)
    return encoded_text


def uniform(shape, device):
    return torch.zeros(shape, device=device).float().uniform_(0, 1)


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (GEGLU,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (LayerNorm,
     lambda: ([], {'dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
]

