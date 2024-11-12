
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


from functools import reduce


import math


import numpy as np


from torch import nn


from torch.nn import functional as F


from torchvision import transforms


from functools import lru_cache


import torch._dynamo


from functools import update_wrapper


from typing import Union


from scipy import integrate


import time


import warnings


from torch import optim


from torch.utils import data


from torchvision.transforms import functional as TF


from copy import deepcopy


from functools import partial


from torch import distributed as dist


from torch import multiprocessing as mp


from torch.utils import flop_counter


from torchvision import datasets


from torchvision import utils


class KarrasAugmentWrapper(nn.Module):

    def __init__(self, model):
        super().__init__()
        self.inner_model = model

    def forward(self, input, sigma, aug_cond=None, mapping_cond=None, **kwargs):
        if aug_cond is None:
            aug_cond = input.new_zeros([input.shape[0], 9])
        if mapping_cond is None:
            mapping_cond = aug_cond
        else:
            mapping_cond = torch.cat([aug_cond, mapping_cond], dim=1)
        return self.inner_model(input, sigma, mapping_cond=mapping_cond, **kwargs)

    def param_groups(self, *args, **kwargs):
        return self.inner_model.param_groups(*args, **kwargs)

    def set_skip_stages(self, skip_stages):
        return self.inner_model.set_skip_stages(skip_stages)

    def set_patch_size(self, patch_size):
        return self.inner_model.set_patch_size(patch_size)


class InceptionV3FeatureExtractor(nn.Module):

    def __init__(self, device='cpu'):
        super().__init__()
        path = Path(os.environ.get('XDG_CACHE_HOME', Path.home() / '.cache')) / 'k-diffusion'
        url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/inception-2015-12-05.pt'
        digest = 'f58cb9b6ec323ed63459aa4fb441fe750cfe39fafad6da5cb504a16f19e958f4'
        utils.download_file(path / 'inception-2015-12-05.pt', url, digest)
        self.model = InceptionV3W(str(path), resize_inside=False)
        self.size = 299, 299

    def forward(self, x):
        x = F.interpolate(x, self.size, mode='bicubic', align_corners=False, antialias=True)
        if x.shape[1] == 1:
            x = torch.cat([x] * 3, dim=1)
        x = (x * 127.5 + 127.5).clamp(0, 255)
        return self.model(x)


class CLIPFeatureExtractor(nn.Module):

    def __init__(self, name='ViT-B/16', device='cpu'):
        super().__init__()
        self.model = clip.load(name, device=device)[0].eval().requires_grad_(False)
        self.normalize = transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
        self.size = self.model.visual.input_resolution, self.model.visual.input_resolution

    @classmethod
    def available_models(cls):
        return clip.available_models()

    def forward(self, x):
        x = (x + 1) / 2
        x = F.interpolate(x, self.size, mode='bicubic', align_corners=False, antialias=True)
        if x.shape[1] == 1:
            x = torch.cat([x] * 3, dim=1)
        x = self.normalize(x)
        x = self.model.encode_image(x).float()
        x = F.normalize(x) * x.shape[-1] ** 0.5
        return x


class DINOv2FeatureExtractor(nn.Module):

    def __init__(self, name='vitl14', device='cpu'):
        super().__init__()
        self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_' + name).eval().requires_grad_(False)
        self.normalize = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        self.size = 224, 224

    @classmethod
    def available_models(cls):
        return ['vits14', 'vitb14', 'vitl14', 'vitg14']

    def forward(self, x):
        x = (x + 1) / 2
        x = F.interpolate(x, self.size, mode='bicubic', align_corners=False, antialias=True)
        if x.shape[1] == 1:
            x = torch.cat([x] * 3, dim=1)
        x = self.normalize(x)
        with torch.amp.autocast(dtype=torch.float16):
            x = self.model(x).float()
        x = F.normalize(x) * x.shape[-1] ** 0.5
        return x


class VDenoiser(nn.Module):
    """A v-diffusion-pytorch model wrapper for k-diffusion."""

    def __init__(self, inner_model):
        super().__init__()
        self.inner_model = inner_model
        self.sigma_data = 1.0

    def get_scalings(self, sigma):
        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = -sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2) ** 0.5
        c_in = 1 / (sigma ** 2 + self.sigma_data ** 2) ** 0.5
        return c_skip, c_out, c_in

    def sigma_to_t(self, sigma):
        return sigma.atan() / math.pi * 2

    def t_to_sigma(self, t):
        return (t * math.pi / 2).tan()

    def loss(self, input, noise, sigma, **kwargs):
        c_skip, c_out, c_in = [utils.append_dims(x, input.ndim) for x in self.get_scalings(sigma)]
        noised_input = input + noise * utils.append_dims(sigma, input.ndim)
        model_output = self.inner_model(noised_input * c_in, self.sigma_to_t(sigma), **kwargs)
        target = (input - c_skip * noised_input) / c_out
        return (model_output - target).pow(2).flatten(1).mean(1)

    def forward(self, input, sigma, **kwargs):
        c_skip, c_out, c_in = [utils.append_dims(x, input.ndim) for x in self.get_scalings(sigma)]
        return self.inner_model(input * c_in, self.sigma_to_t(sigma), **kwargs) * c_out + input * c_skip


class DiscreteSchedule(nn.Module):
    """A mapping between continuous noise levels (sigmas) and a list of discrete noise
    levels."""

    def __init__(self, sigmas, quantize):
        super().__init__()
        self.register_buffer('sigmas', sigmas)
        self.register_buffer('log_sigmas', sigmas.log())
        self.quantize = quantize

    @property
    def sigma_min(self):
        return self.sigmas[0]

    @property
    def sigma_max(self):
        return self.sigmas[-1]

    def get_sigmas(self, n=None):
        if n is None:
            return sampling.append_zero(self.sigmas.flip(0))
        t_max = len(self.sigmas) - 1
        t = torch.linspace(t_max, 0, n, device=self.sigmas.device)
        return sampling.append_zero(self.t_to_sigma(t))

    def sigma_to_t(self, sigma, quantize=None):
        quantize = self.quantize if quantize is None else quantize
        log_sigma = sigma.log()
        dists = log_sigma - self.log_sigmas[:, None]
        if quantize:
            return dists.abs().argmin(dim=0).view(sigma.shape)
        low_idx = dists.ge(0).cumsum(dim=0).argmax(dim=0).clamp(max=self.log_sigmas.shape[0] - 2)
        high_idx = low_idx + 1
        low, high = self.log_sigmas[low_idx], self.log_sigmas[high_idx]
        w = (low - log_sigma) / (low - high)
        w = w.clamp(0, 1)
        t = (1 - w) * low_idx + w * high_idx
        return t.view(sigma.shape)

    def t_to_sigma(self, t):
        t = t.float()
        low_idx, high_idx, w = t.floor().long(), t.ceil().long(), t.frac()
        log_sigma = (1 - w) * self.log_sigmas[low_idx] + w * self.log_sigmas[high_idx]
        return log_sigma.exp()


class DiscreteEpsDDPMDenoiser(DiscreteSchedule):
    """A wrapper for discrete schedule DDPM models that output eps (the predicted
    noise)."""

    def __init__(self, model, alphas_cumprod, quantize):
        super().__init__(((1 - alphas_cumprod) / alphas_cumprod) ** 0.5, quantize)
        self.inner_model = model
        self.sigma_data = 1.0

    def get_scalings(self, sigma):
        c_out = -sigma
        c_in = 1 / (sigma ** 2 + self.sigma_data ** 2) ** 0.5
        return c_out, c_in

    def get_eps(self, *args, **kwargs):
        return self.inner_model(*args, **kwargs)

    def loss(self, input, noise, sigma, **kwargs):
        c_out, c_in = [utils.append_dims(x, input.ndim) for x in self.get_scalings(sigma)]
        noised_input = input + noise * utils.append_dims(sigma, input.ndim)
        eps = self.get_eps(noised_input * c_in, self.sigma_to_t(sigma), **kwargs)
        return (eps - noise).pow(2).flatten(1).mean(1)

    def forward(self, input, sigma, **kwargs):
        c_out, c_in = [utils.append_dims(x, input.ndim) for x in self.get_scalings(sigma)]
        eps = self.get_eps(input * c_in, self.sigma_to_t(sigma), **kwargs)
        return input + eps * c_out


class OpenAIDenoiser(DiscreteEpsDDPMDenoiser):
    """A wrapper for OpenAI diffusion models."""

    def __init__(self, model, diffusion, quantize=False, has_learned_sigmas=True, device='cpu'):
        alphas_cumprod = torch.tensor(diffusion.alphas_cumprod, device=device, dtype=torch.float32)
        super().__init__(model, alphas_cumprod, quantize=quantize)
        self.has_learned_sigmas = has_learned_sigmas

    def get_eps(self, *args, **kwargs):
        model_output = self.inner_model(*args, **kwargs)
        if self.has_learned_sigmas:
            return model_output.chunk(2, dim=1)[0]
        return model_output


class CompVisDenoiser(DiscreteEpsDDPMDenoiser):
    """A wrapper for CompVis diffusion models."""

    def __init__(self, model, quantize=False, device='cpu'):
        super().__init__(model, model.alphas_cumprod, quantize=quantize)

    def get_eps(self, *args, **kwargs):
        return self.inner_model.apply_model(*args, **kwargs)


class DiscreteVDDPMDenoiser(DiscreteSchedule):
    """A wrapper for discrete schedule DDPM models that output v."""

    def __init__(self, model, alphas_cumprod, quantize):
        super().__init__(((1 - alphas_cumprod) / alphas_cumprod) ** 0.5, quantize)
        self.inner_model = model
        self.sigma_data = 1.0

    def get_scalings(self, sigma):
        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = -sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2) ** 0.5
        c_in = 1 / (sigma ** 2 + self.sigma_data ** 2) ** 0.5
        return c_skip, c_out, c_in

    def get_v(self, *args, **kwargs):
        return self.inner_model(*args, **kwargs)

    def loss(self, input, noise, sigma, **kwargs):
        c_skip, c_out, c_in = [utils.append_dims(x, input.ndim) for x in self.get_scalings(sigma)]
        noised_input = input + noise * utils.append_dims(sigma, input.ndim)
        model_output = self.get_v(noised_input * c_in, self.sigma_to_t(sigma), **kwargs)
        target = (input - c_skip * noised_input) / c_out
        return (model_output - target).pow(2).flatten(1).mean(1)

    def forward(self, input, sigma, **kwargs):
        c_skip, c_out, c_in = [utils.append_dims(x, input.ndim) for x in self.get_scalings(sigma)]
        return self.get_v(input * c_in, self.sigma_to_t(sigma), **kwargs) * c_out + input * c_skip


class CompVisVDenoiser(DiscreteVDDPMDenoiser):
    """A wrapper for CompVis diffusion models that output v."""

    def __init__(self, model, quantize=False, device='cpu'):
        super().__init__(model, model.alphas_cumprod, quantize=quantize)

    def get_v(self, x, t, cond, **kwargs):
        return self.inner_model.apply_model(x, t, cond)


def dct(x):
    if x.ndim == 3:
        return df.dct(x)
    if x.ndim == 4:
        return df.dct2(x)
    if x.ndim == 5:
        return df.dct3(x)
    raise ValueError(f'Unsupported dimensionality {x.ndim}')


@lru_cache
def freq_weight_1d(n, scales=0, dtype=None, device=None):
    ramp = torch.linspace(0.5 / n, 0.5, n, dtype=dtype, device=device)
    weights = -torch.log2(ramp)
    if scales >= 1:
        weights = torch.clamp_max(weights, scales)
    return weights


@lru_cache
def freq_weight_nd(shape, scales=0, dtype=None, device=None):
    indexers = [[(slice(None) if i == j else None) for j in range(len(shape))] for i in range(len(shape))]
    weights = [freq_weight_1d(n, scales, dtype, device)[ix] for n, ix in zip(shape, indexers)]
    return reduce(torch.minimum, weights)


class Denoiser(nn.Module):
    """A Karras et al. preconditioner for denoising diffusion models."""

    def __init__(self, inner_model, sigma_data=1.0, weighting='karras', scales=1):
        super().__init__()
        self.inner_model = inner_model
        self.sigma_data = sigma_data
        self.scales = scales
        if callable(weighting):
            self.weighting = weighting
        if weighting == 'karras':
            self.weighting = torch.ones_like
        elif weighting == 'soft-min-snr':
            self.weighting = self._weighting_soft_min_snr
        elif weighting == 'snr':
            self.weighting = self._weighting_snr
        else:
            raise ValueError(f'Unknown weighting type {weighting}')

    def _weighting_soft_min_snr(self, sigma):
        return (sigma * self.sigma_data) ** 2 / (sigma ** 2 + self.sigma_data ** 2) ** 2

    def _weighting_snr(self, sigma):
        return self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)

    def get_scalings(self, sigma):
        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2) ** 0.5
        c_in = 1 / (sigma ** 2 + self.sigma_data ** 2) ** 0.5
        return c_skip, c_out, c_in

    def loss(self, input, noise, sigma, **kwargs):
        c_skip, c_out, c_in = [utils.append_dims(x, input.ndim) for x in self.get_scalings(sigma)]
        c_weight = self.weighting(sigma)
        noised_input = input + noise * utils.append_dims(sigma, input.ndim)
        model_output = self.inner_model(noised_input * c_in, sigma, **kwargs)
        target = (input - c_skip * noised_input) / c_out
        if self.scales == 1:
            return ((model_output - target) ** 2).flatten(1).mean(1) * c_weight
        sq_error = dct(model_output - target) ** 2
        f_weight = freq_weight_nd(sq_error.shape[2:], self.scales, dtype=sq_error.dtype, device=sq_error.device)
        return (sq_error * f_weight).flatten(1).mean(1) * c_weight

    def forward(self, input, sigma, **kwargs):
        c_skip, c_out, c_in = [utils.append_dims(x, input.ndim) for x in self.get_scalings(sigma)]
        return self.inner_model(input * c_in, sigma, **kwargs) * c_out + input * c_skip


class DenoiserWithVariance(Denoiser):

    def loss(self, input, noise, sigma, **kwargs):
        c_skip, c_out, c_in = [utils.append_dims(x, input.ndim) for x in self.get_scalings(sigma)]
        noised_input = input + noise * utils.append_dims(sigma, input.ndim)
        model_output, logvar = self.inner_model(noised_input * c_in, sigma, return_variance=True, **kwargs)
        logvar = utils.append_dims(logvar, model_output.ndim)
        target = (input - c_skip * noised_input) / c_out
        losses = ((model_output - target) ** 2 / logvar.exp() + logvar) / 2
        return losses.flatten(1).mean(1)


class SimpleLossDenoiser(Denoiser):
    """L_simple with the Karras et al. preconditioner."""

    def loss(self, input, noise, sigma, **kwargs):
        noised_input = input + noise * utils.append_dims(sigma, input.ndim)
        denoised = self(noised_input, sigma, **kwargs)
        eps = sampling.to_d(noised_input, sigma, denoised)
        return (eps - noise).pow(2).flatten(1).mean(1)


class ResidualBlock(nn.Module):

    def __init__(self, *main, skip=None):
        super().__init__()
        self.main = nn.Sequential(*main)
        self.skip = skip if skip else nn.Identity()

    def forward(self, input):
        return self.main(input) + self.skip(input)


class ConditionedModule(nn.Module):
    pass


class UnconditionedModule(ConditionedModule):

    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, input, cond=None):
        return self.module(input)


class ConditionedSequential(nn.Sequential, ConditionedModule):

    def forward(self, input, cond):
        for module in self:
            if isinstance(module, ConditionedModule):
                input = module(input, cond)
            else:
                input = module(input)
        return input


class ConditionedResidualBlock(ConditionedModule):

    def __init__(self, *main, skip=None):
        super().__init__()
        self.main = ConditionedSequential(*main)
        self.skip = skip if skip else nn.Identity()

    def forward(self, input, cond):
        skip = self.skip(input, cond) if isinstance(self.skip, ConditionedModule) else self.skip(input)
        return self.main(input, cond) + skip


class AdaGN(ConditionedModule):

    def __init__(self, feats_in, c_out, num_groups, eps=1e-05, cond_key='cond'):
        super().__init__()
        self.num_groups = num_groups
        self.eps = eps
        self.cond_key = cond_key
        self.mapper = nn.Linear(feats_in, c_out * 2)
        nn.init.zeros_(self.mapper.weight)
        nn.init.zeros_(self.mapper.bias)

    def forward(self, input, cond):
        weight, bias = self.mapper(cond[self.cond_key]).chunk(2, dim=-1)
        input = F.group_norm(input, self.num_groups, eps=self.eps)
        return torch.addcmul(utils.append_dims(bias, input.ndim), input, utils.append_dims(weight, input.ndim) + 1)


class SelfAttention2d(ConditionedModule):

    def __init__(self, c_in, n_head, norm, dropout_rate=0.0):
        super().__init__()
        assert c_in % n_head == 0
        self.norm_in = norm(c_in)
        self.n_head = n_head
        self.qkv_proj = nn.Conv2d(c_in, c_in * 3, 1)
        self.out_proj = nn.Conv2d(c_in, c_in, 1)
        self.dropout = nn.Dropout(dropout_rate)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, input, cond):
        n, c, h, w = input.shape
        qkv = self.qkv_proj(self.norm_in(input, cond))
        qkv = qkv.view([n, self.n_head * 3, c // self.n_head, h * w]).transpose(2, 3)
        q, k, v = qkv.chunk(3, dim=1)
        y = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout.p)
        y = y.transpose(2, 3).contiguous().view([n, c, h, w])
        return input + self.out_proj(y)


class CrossAttention2d(ConditionedModule):

    def __init__(self, c_dec, c_enc, n_head, norm_dec, dropout_rate=0.0, cond_key='cross', cond_key_padding='cross_padding'):
        super().__init__()
        assert c_dec % n_head == 0
        self.cond_key = cond_key
        self.cond_key_padding = cond_key_padding
        self.norm_enc = nn.LayerNorm(c_enc)
        self.norm_dec = norm_dec(c_dec)
        self.n_head = n_head
        self.q_proj = nn.Conv2d(c_dec, c_dec, 1)
        self.kv_proj = nn.Linear(c_enc, c_dec * 2)
        self.out_proj = nn.Conv2d(c_dec, c_dec, 1)
        self.dropout = nn.Dropout(dropout_rate)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, input, cond):
        n, c, h, w = input.shape
        q = self.q_proj(self.norm_dec(input, cond))
        q = q.view([n, self.n_head, c // self.n_head, h * w]).transpose(2, 3)
        kv = self.kv_proj(self.norm_enc(cond[self.cond_key]))
        kv = kv.view([n, -1, self.n_head * 2, c // self.n_head]).transpose(1, 2)
        k, v = kv.chunk(2, dim=1)
        attn_mask = cond[self.cond_key_padding][:, None, None, :] * -10000
        y = F.scaled_dot_product_attention(q, k, v, attn_mask, dropout_p=self.dropout.p)
        y = y.transpose(2, 3).contiguous().view([n, c, h, w])
        return input + self.out_proj(y)


_kernels = {'linear': [1 / 8, 3 / 8, 3 / 8, 1 / 8], 'cubic': [-0.01171875, -0.03515625, 0.11328125, 0.43359375, 0.43359375, 0.11328125, -0.03515625, -0.01171875], 'lanczos3': [0.003689131001010537, 0.015056144446134567, -0.03399861603975296, -0.066637322306633, 0.13550527393817902, 0.44638532400131226, 0.44638532400131226, 0.13550527393817902, -0.066637322306633, -0.03399861603975296, 0.015056144446134567, 0.003689131001010537]}


class Downsample2d(nn.Module):

    def __init__(self, kernel='linear', pad_mode='reflect'):
        super().__init__()
        self.pad_mode = pad_mode
        kernel_1d = torch.tensor([_kernels[kernel]])
        self.pad = kernel_1d.shape[1] // 2 - 1
        self.register_buffer('kernel', kernel_1d.T @ kernel_1d)

    def forward(self, x):
        x = F.pad(x, (self.pad,) * 4, self.pad_mode)
        weight = x.new_zeros([x.shape[1], x.shape[1], self.kernel.shape[0], self.kernel.shape[1]])
        indices = torch.arange(x.shape[1], device=x.device)
        weight[indices, indices] = self.kernel
        return F.conv2d(x, weight, stride=2)


class Upsample2d(nn.Module):

    def __init__(self, kernel='linear', pad_mode='reflect'):
        super().__init__()
        self.pad_mode = pad_mode
        kernel_1d = torch.tensor([_kernels[kernel]]) * 2
        self.pad = kernel_1d.shape[1] // 2 - 1
        self.register_buffer('kernel', kernel_1d.T @ kernel_1d)

    def forward(self, x):
        x = F.pad(x, ((self.pad + 1) // 2,) * 4, self.pad_mode)
        weight = x.new_zeros([x.shape[1], x.shape[1], self.kernel.shape[0], self.kernel.shape[1]])
        indices = torch.arange(x.shape[1], device=x.device)
        weight[indices, indices] = self.kernel
        return F.conv_transpose2d(x, weight, stride=2, padding=self.pad * 2 + 1)


class FourierFeatures(nn.Module):

    def __init__(self, in_features, out_features, std=1.0):
        super().__init__()
        assert out_features % 2 == 0
        self.register_buffer('weight', torch.randn([out_features // 2, in_features]) * std)

    def forward(self, input):
        f = 2 * math.pi * input @ self.weight.T
        return torch.cat([f.cos(), f.sin()], dim=-1)


class UNet(ConditionedModule):

    def __init__(self, d_blocks, u_blocks, skip_stages=0):
        super().__init__()
        self.d_blocks = nn.ModuleList(d_blocks)
        self.u_blocks = nn.ModuleList(u_blocks)
        self.skip_stages = skip_stages

    def forward(self, input, cond):
        skips = []
        for block in self.d_blocks[self.skip_stages:]:
            input = block(input, cond)
            skips.append(input)
        for i, (block, skip) in enumerate(zip(self.u_blocks, reversed(skips))):
            input = block(input, cond, skip if i > 0 else None)
        return input


class AxialRoPE(nn.Module):

    def __init__(self, dim, n_heads):
        super().__init__()
        log_min = math.log(math.pi)
        log_max = math.log(10.0 * math.pi)
        freqs = torch.linspace(log_min, log_max, n_heads * dim // 4 + 1)[:-1].exp()
        self.register_buffer('freqs', freqs.view(dim // 4, n_heads).T.contiguous())

    def extra_repr(self):
        return f'dim={self.freqs.shape[1] * 4}, n_heads={self.freqs.shape[0]}'

    def forward(self, pos):
        theta_h = pos[..., None, 0:1] * self.freqs
        theta_w = pos[..., None, 1:2] * self.freqs
        return torch.cat((theta_h, theta_w), dim=-1)


class GEGLU(nn.Module):

    def forward(self, x):
        return geglu(x)


class RMSNorm(nn.Module):

    def __init__(self, shape, eps=1e-06):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(shape))

    def extra_repr(self):
        return f'shape={tuple(self.scale.shape)}, eps={self.eps}'

    def forward(self, x):
        return rms_norm(x, self.scale, self.eps)


class QKNorm(nn.Module):

    def __init__(self, n_heads, eps=1e-06, max_scale=100.0):
        super().__init__()
        self.eps = eps
        self.max_scale = math.log(max_scale)
        self.scale = nn.Parameter(torch.full((n_heads,), math.log(10.0)))
        self.proj_()

    def extra_repr(self):
        return f'n_heads={self.scale.shape[0]}, eps={self.eps}'

    @torch.no_grad()
    def proj_(self):
        """Modify the scale in-place so it doesn't get "stuck" with zero gradient if it's clamped
        to the max value."""
        self.scale.clamp_(max=self.max_scale)

    def forward(self, x):
        self.proj_()
        scale = torch.exp(0.5 * self.scale - 0.25 * math.log(x.shape[-1]))
        return rms_norm(x, scale[:, None, None], self.eps)


class Linear(nn.Linear):

    def forward(self, x):
        flops.op(flops.op_linear, x.shape, self.weight.shape)
        return super().forward(x)


def tag_param(param, tag):
    if not hasattr(param, '_tags'):
        param._tags = set([tag])
    else:
        param._tags.add(tag)
    return param


def apply_wd(module):
    for name, param in module.named_parameters():
        if name.endswith('weight'):
            tag_param(param, 'wd')
    return module


def tag_module(module, tag):
    for param in module.parameters():
        tag_param(param, tag)
    return module


def zero_init(layer):
    nn.init.zeros_(layer.weight)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)
    return layer


class AdaRMSNorm(nn.Module):

    def __init__(self, features, cond_features, eps=1e-06):
        super().__init__()
        self.eps = eps
        self.linear = apply_wd(zero_init(Linear(cond_features, features, bias=False)))
        tag_module(self.linear, 'mapping')

    def extra_repr(self):
        return f'eps={self.eps},'

    def forward(self, x, cond):
        return rms_norm(x, self.linear(cond)[:, None, None, :] + 1, self.eps)


class ApplyRotaryEmbeddingInplace(torch.autograd.Function):

    @staticmethod
    def forward(x, theta, conj):
        _apply_rotary_emb_inplace(x, theta, conj=conj)
        return x

    @staticmethod
    def setup_context(ctx, inputs, output):
        _, theta, conj = inputs
        ctx.save_for_backward(theta)
        ctx.conj = conj

    @staticmethod
    def backward(ctx, grad_output):
        theta, = ctx.saved_tensors
        _apply_rotary_emb_inplace(grad_output, theta, conj=not ctx.conj)
        return grad_output, None, None


def apply_rotary_emb_(x, theta):
    return ApplyRotaryEmbeddingInplace.apply(x, theta, False)


def use_flash_2(x):
    if not flags.get_use_flash_attention_2():
        return False
    if flash_attn is None:
        return False
    if x.device.type != 'cuda':
        return False
    if x.dtype not in (torch.float16, torch.bfloat16):
        return False
    return True


class SelfAttentionBlock(nn.Module):

    def __init__(self, d_model, d_head, cond_features, dropout=0.0):
        super().__init__()
        self.d_head = d_head
        self.n_heads = d_model // d_head
        self.norm = AdaRMSNorm(d_model, cond_features)
        self.qkv_proj = apply_wd(Linear(d_model, d_model * 3, bias=False))
        self.scale = nn.Parameter(torch.full([self.n_heads], 10.0))
        self.pos_emb = AxialRoPE(d_head // 2, self.n_heads)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = apply_wd(zero_init(Linear(d_model, d_model, bias=False)))

    def extra_repr(self):
        return f'd_head={self.d_head},'

    def forward(self, x, pos, cond):
        skip = x
        x = self.norm(x, cond)
        qkv = self.qkv_proj(x)
        pos = rearrange(pos, '... h w e -> ... (h w) e')
        theta = self.pos_emb(pos)
        if use_flash_2(qkv):
            qkv = rearrange(qkv, 'n h w (t nh e) -> n (h w) t nh e', t=3, e=self.d_head)
            qkv = scale_for_cosine_sim_qkv(qkv, self.scale, 1e-06)
            theta = torch.stack((theta, theta, torch.zeros_like(theta)), dim=-3)
            qkv = apply_rotary_emb_(qkv, theta)
            flops_shape = qkv.shape[-5], qkv.shape[-2], qkv.shape[-4], qkv.shape[-1]
            flops.op(flops.op_attention, flops_shape, flops_shape, flops_shape)
            x = flash_attn.flash_attn_qkvpacked_func(qkv, softmax_scale=1.0)
            x = rearrange(x, 'n (h w) nh e -> n h w (nh e)', h=skip.shape[-3], w=skip.shape[-2])
        else:
            q, k, v = rearrange(qkv, 'n h w (t nh e) -> t n nh (h w) e', t=3, e=self.d_head)
            q, k = scale_for_cosine_sim(q, k, self.scale[:, None, None], 1e-06)
            theta = theta.movedim(-2, -3)
            q = apply_rotary_emb_(q, theta)
            k = apply_rotary_emb_(k, theta)
            flops.op(flops.op_attention, q.shape, k.shape, v.shape)
            x = F.scaled_dot_product_attention(q, k, v, scale=1.0)
            x = rearrange(x, 'n nh (h w) e -> n h w (nh e)', h=skip.shape[-3], w=skip.shape[-2])
        x = self.dropout(x)
        x = self.out_proj(x)
        return x + skip


class LinearGEGLU(nn.Linear):

    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features * 2, bias=bias)
        self.out_features = out_features

    def forward(self, x):
        flops.op(flops.op_linear, x.shape, self.weight.shape)
        return linear_geglu(x, self.weight, self.bias)


class FeedForwardBlock(nn.Module):

    def __init__(self, d_model, d_ff, cond_features, dropout=0.0):
        super().__init__()
        self.norm = AdaRMSNorm(d_model, cond_features)
        self.up_proj = apply_wd(LinearGEGLU(d_model, d_ff, bias=False))
        self.dropout = nn.Dropout(dropout)
        self.down_proj = apply_wd(zero_init(Linear(d_ff, d_model, bias=False)))

    def forward(self, x, cond):
        skip = x
        x = self.norm(x, cond)
        x = self.up_proj(x)
        x = self.dropout(x)
        x = self.down_proj(x)
        return x + skip


def checkpoint_helper(function, *args, **kwargs):
    if flags.get_checkpointing():
        kwargs.setdefault('use_reentrant', True)
        return torch.utils.checkpoint.checkpoint(function, *args, **kwargs)
    else:
        return function(*args, **kwargs)


class TransformerBlock(nn.Module):

    def __init__(self, d_model, d_ff, d_head, dropout=0.0):
        super().__init__()
        self.self_attn = SelfAttentionBlock(d_model, d_head, dropout=dropout)
        self.ff = FeedForwardBlock(d_model, d_ff, dropout=dropout)

    def forward(self, x, pos, attn_mask, cond):
        x = checkpoint_helper(self.self_attn, x, pos, attn_mask, cond)
        x = checkpoint_helper(self.ff, x, cond)
        return x


def bounding_box(h, w, pixel_aspect_ratio=1.0):
    w_adj = w
    h_adj = h * pixel_aspect_ratio
    ar_adj = w_adj / h_adj
    y_min, y_max, x_min, x_max = -1.0, 1.0, -1.0, 1.0
    if ar_adj > 1:
        y_min, y_max = -1 / ar_adj, 1 / ar_adj
    elif ar_adj < 1:
        x_min, x_max = -ar_adj, ar_adj
    return y_min, y_max, x_min, x_max


def centers(start, stop, num, dtype=None, device=None):
    edges = torch.linspace(start, stop, num + 1, dtype=dtype, device=device)
    return (edges[:-1] + edges[1:]) / 2


def make_grid(h_pos, w_pos):
    grid = torch.stack(torch.meshgrid(h_pos, w_pos, indexing='ij'), dim=-1)
    h, w, d = grid.shape
    return grid.view(h * w, d)


def make_axial_pos(h, w, pixel_aspect_ratio=1.0, align_corners=False, dtype=None, device=None):
    y_min, y_max, x_min, x_max = bounding_box(h, w, pixel_aspect_ratio)
    if align_corners:
        h_pos = torch.linspace(y_min, y_max, h, dtype=dtype, device=device)
        w_pos = torch.linspace(x_min, x_max, w, dtype=dtype, device=device)
    else:
        h_pos = centers(y_min, y_max, h, dtype=dtype, device=device)
        w_pos = centers(x_min, x_max, w, dtype=dtype, device=device)
    return make_grid(h_pos, w_pos)


class Patching(nn.Module):

    def __init__(self, features, patch_size):
        super().__init__()
        self.features = features
        self.patch_size = patch_size
        self.d_out = features * patch_size[0] * patch_size[1]

    def extra_repr(self):
        return f'features={self.features}, patch_size={self.patch_size!r}'

    def forward(self, x, pixel_aspect_ratio=1.0):
        *_, h, w = x.shape
        h_out = h // self.patch_size[0]
        w_out = w // self.patch_size[1]
        if h % self.patch_size[0] != 0 or w % self.patch_size[1] != 0:
            raise ValueError(f'Image size {h}x{w} is not divisible by patch size {self.patch_size[0]}x{self.patch_size[1]}')
        x = rearrange(x, '... c (h i) (w j) -> ... (h w) (c i j)', i=self.patch_size[0], j=self.patch_size[1])
        pixel_aspect_ratio = pixel_aspect_ratio * self.patch_size[0] / self.patch_size[1]
        pos = make_axial_pos(h_out, w_out, pixel_aspect_ratio, device=x.device)
        return x, pos


class Unpatching(nn.Module):

    def __init__(self, features, patch_size):
        super().__init__()
        self.features = features
        self.patch_size = patch_size
        self.d_in = features * patch_size[0] * patch_size[1]

    def extra_repr(self):
        return f'features={self.features}, patch_size={self.patch_size!r}'

    def forward(self, x, h, w):
        h_in = h // self.patch_size[0]
        w_in = w // self.patch_size[1]
        x = rearrange(x, '... (h w) (c i j) -> ... c (h i) (w j)', h=h_in, w=w_in, i=self.patch_size[0], j=self.patch_size[1])
        return x


class MappingFeedForwardBlock(nn.Module):

    def __init__(self, d_model, d_ff, dropout=0.0):
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.up_proj = apply_wd(LinearGEGLU(d_model, d_ff, bias=False))
        self.dropout = nn.Dropout(dropout)
        self.down_proj = apply_wd(zero_init(Linear(d_ff, d_model, bias=False)))

    def forward(self, x):
        skip = x
        x = self.norm(x)
        x = self.up_proj(x)
        x = self.dropout(x)
        x = self.down_proj(x)
        return x + skip


class MappingNetwork(nn.Module):

    def __init__(self, n_layers, d_model, d_ff, dropout=0.0):
        super().__init__()
        self.in_norm = RMSNorm(d_model)
        self.blocks = nn.ModuleList([MappingFeedForwardBlock(d_model, d_ff, dropout=dropout) for _ in range(n_layers)])
        self.out_norm = RMSNorm(d_model)

    def forward(self, x):
        x = self.in_norm(x)
        for block in self.blocks:
            x = block(x)
        x = self.out_norm(x)
        return x


def filter_params(function, module):
    for param in module.parameters():
        tags = getattr(param, '_tags', set())
        if function(tags):
            yield param


class ImageTransformerDenoiserModelV1(nn.Module):

    def __init__(self, n_layers, d_model, d_ff, in_features, out_features, patch_size, num_classes=0, dropout=0.0, sigma_data=1.0):
        super().__init__()
        self.sigma_data = sigma_data
        self.num_classes = num_classes
        self.patch_in = Patching(in_features, patch_size)
        self.patch_out = Unpatching(out_features, patch_size)
        self.time_emb = layers.FourierFeatures(1, d_model)
        self.time_in_proj = nn.Linear(d_model, d_model, bias=False)
        self.aug_emb = layers.FourierFeatures(9, d_model)
        self.aug_in_proj = nn.Linear(d_model, d_model, bias=False)
        self.class_emb = nn.Embedding(num_classes, d_model) if num_classes else None
        self.mapping = tag_module(MappingNetwork(2, d_model, d_ff, dropout=dropout), 'mapping')
        self.in_proj = nn.Linear(self.patch_in.d_out, d_model, bias=False)
        self.blocks = nn.ModuleList([TransformerBlock(d_model, d_ff, 64, dropout=dropout) for _ in range(n_layers)])
        self.out_norm = RMSNorm(d_model)
        self.out_proj = zero_init(nn.Linear(d_model, self.patch_out.d_in, bias=False))

    def proj_(self):
        for block in self.blocks:
            block.self_attn.qk_norm.proj_()

    def param_groups(self, base_lr=0.0005, mapping_lr_scale=1 / 3):
        wd = filter_params(lambda tags: 'wd' in tags and 'mapping' not in tags, self)
        no_wd = filter_params(lambda tags: 'wd' not in tags and 'mapping' not in tags, self)
        mapping_wd = filter_params(lambda tags: 'wd' in tags and 'mapping' in tags, self)
        mapping_no_wd = filter_params(lambda tags: 'wd' not in tags and 'mapping' in tags, self)
        groups = [{'params': list(wd), 'lr': base_lr}, {'params': list(no_wd), 'lr': base_lr, 'weight_decay': 0.0}, {'params': list(mapping_wd), 'lr': base_lr * mapping_lr_scale}, {'params': list(mapping_no_wd), 'lr': base_lr * mapping_lr_scale, 'weight_decay': 0.0}]
        return groups

    def forward(self, x, sigma, aug_cond=None, class_cond=None):
        *_, h, w = x.shape
        x, pos = self.patch_in(x)
        attn_mask = None
        x = self.in_proj(x)
        if class_cond is None and self.class_emb is not None:
            raise ValueError('class_cond must be specified if num_classes > 0')
        c_noise = torch.log(sigma) / 4
        time_emb = self.time_in_proj(self.time_emb(c_noise[..., None]))
        aug_cond = x.new_zeros([x.shape[0], 9]) if aug_cond is None else aug_cond
        aug_emb = self.aug_in_proj(self.aug_emb(aug_cond))
        class_emb = self.class_emb(class_cond) if self.class_emb is not None else 0
        cond = self.mapping(time_emb + aug_emb + class_emb).unsqueeze(-2)
        for block in self.blocks:
            x = block(x, pos, attn_mask, cond)
        x = self.out_norm(x)
        x = self.out_proj(x)
        x = self.patch_out(x, h, w)
        return x


class NeighborhoodSelfAttentionBlock(nn.Module):

    def __init__(self, d_model, d_head, cond_features, kernel_size, dropout=0.0):
        super().__init__()
        self.d_head = d_head
        self.n_heads = d_model // d_head
        self.kernel_size = kernel_size
        self.norm = AdaRMSNorm(d_model, cond_features)
        self.qkv_proj = apply_wd(Linear(d_model, d_model * 3, bias=False))
        self.scale = nn.Parameter(torch.full([self.n_heads], 10.0))
        self.pos_emb = AxialRoPE(d_head // 2, self.n_heads)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = apply_wd(zero_init(Linear(d_model, d_model, bias=False)))

    def extra_repr(self):
        return f'd_head={self.d_head}, kernel_size={self.kernel_size}'

    def forward(self, x, pos, cond):
        skip = x
        x = self.norm(x, cond)
        qkv = self.qkv_proj(x)
        if natten is None:
            raise ModuleNotFoundError('natten is required for neighborhood attention')
        if natten.has_fused_na():
            q, k, v = rearrange(qkv, 'n h w (t nh e) -> t n h w nh e', t=3, e=self.d_head)
            q, k = scale_for_cosine_sim(q, k, self.scale[:, None], 1e-06)
            theta = self.pos_emb(pos)
            q = apply_rotary_emb_(q, theta)
            k = apply_rotary_emb_(k, theta)
            flops.op(flops.op_natten, q.shape, k.shape, v.shape, self.kernel_size)
            x = natten.functional.na2d(q, k, v, self.kernel_size, scale=1.0)
            x = rearrange(x, 'n h w nh e -> n h w (nh e)')
        else:
            q, k, v = rearrange(qkv, 'n h w (t nh e) -> t n nh h w e', t=3, e=self.d_head)
            q, k = scale_for_cosine_sim(q, k, self.scale[:, None, None, None], 1e-06)
            theta = self.pos_emb(pos).movedim(-2, -4)
            q = apply_rotary_emb_(q, theta)
            k = apply_rotary_emb_(k, theta)
            flops.op(flops.op_natten, q.shape, k.shape, v.shape, self.kernel_size)
            qk = natten.functional.na2d_qk(q, k, self.kernel_size)
            a = torch.softmax(qk, dim=-1)
            x = natten.functional.na2d_av(a, v, self.kernel_size)
            x = rearrange(x, 'n nh h w e -> n h w (nh e)')
        x = self.dropout(x)
        x = self.out_proj(x)
        return x + skip


@lru_cache
def make_shifted_window_masks(n_h_w, n_w_w, w_h, w_w, shift, device=None):
    ph_coords = torch.arange(n_h_w, device=device)
    pw_coords = torch.arange(n_w_w, device=device)
    h_coords = torch.arange(w_h, device=device)
    w_coords = torch.arange(w_w, device=device)
    patch_h, patch_w, q_h, q_w, k_h, k_w = torch.meshgrid(ph_coords, pw_coords, h_coords, w_coords, h_coords, w_coords, indexing='ij')
    is_top_patch = patch_h == 0
    is_left_patch = patch_w == 0
    q_above_shift = q_h < shift
    k_above_shift = k_h < shift
    q_left_of_shift = q_w < shift
    k_left_of_shift = k_w < shift
    m_corner = is_left_patch & is_top_patch & (q_left_of_shift == k_left_of_shift) & (q_above_shift == k_above_shift)
    m_left = is_left_patch & ~is_top_patch & (q_left_of_shift == k_left_of_shift)
    m_top = ~is_left_patch & is_top_patch & (q_above_shift == k_above_shift)
    m_rest = ~is_left_patch & ~is_top_patch
    m = m_corner | m_left | m_top | m_rest
    return m


def unwindow(x):
    *b, h, w, wh, ww, c = x.shape
    x = torch.permute(x, (*range(len(b)), -5, -3, -4, -2, -1))
    x = torch.reshape(x, (*b, h * wh, w * ww, c))
    return x


def shifted_unwindow(window_shift, x):
    x = unwindow(x)
    x = torch.roll(x, shifts=(-window_shift, -window_shift), dims=(-2, -3))
    return x


def window(window_size, x):
    *b, h, w, c = x.shape
    x = torch.reshape(x, (*b, h // window_size, window_size, w // window_size, window_size, c))
    x = torch.permute(x, (*range(len(b)), -5, -3, -4, -2, -1))
    return x


def shifted_window(window_size, window_shift, x):
    x = torch.roll(x, shifts=(window_shift, window_shift), dims=(-2, -3))
    windows = window(window_size, x)
    return windows


def apply_window_attention(window_size, window_shift, q, k, v, scale=None):
    q_windows = shifted_window(window_size, window_shift, q)
    k_windows = shifted_window(window_size, window_shift, k)
    v_windows = shifted_window(window_size, window_shift, v)
    b, heads, h, w, wh, ww, d_head = q_windows.shape
    mask = make_shifted_window_masks(h, w, wh, ww, window_shift, device=q.device)
    q_seqs = torch.reshape(q_windows, (b, heads, h, w, wh * ww, d_head))
    k_seqs = torch.reshape(k_windows, (b, heads, h, w, wh * ww, d_head))
    v_seqs = torch.reshape(v_windows, (b, heads, h, w, wh * ww, d_head))
    mask = torch.reshape(mask, (h, w, wh * ww, wh * ww))
    flops.op(flops.op_attention, q_seqs.shape, k_seqs.shape, v_seqs.shape)
    qkv = F.scaled_dot_product_attention(q_seqs, k_seqs, v_seqs, mask, scale=scale)
    qkv = torch.reshape(qkv, (b, heads, h, w, wh, ww, d_head))
    return shifted_unwindow(window_shift, qkv)


class ShiftedWindowSelfAttentionBlock(nn.Module):

    def __init__(self, d_model, d_head, cond_features, window_size, window_shift, dropout=0.0):
        super().__init__()
        self.d_head = d_head
        self.n_heads = d_model // d_head
        self.window_size = window_size
        self.window_shift = window_shift
        self.norm = AdaRMSNorm(d_model, cond_features)
        self.qkv_proj = apply_wd(Linear(d_model, d_model * 3, bias=False))
        self.scale = nn.Parameter(torch.full([self.n_heads], 10.0))
        self.pos_emb = AxialRoPE(d_head // 2, self.n_heads)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = apply_wd(zero_init(Linear(d_model, d_model, bias=False)))

    def extra_repr(self):
        return f'd_head={self.d_head}, window_size={self.window_size}, window_shift={self.window_shift}'

    def forward(self, x, pos, cond):
        skip = x
        x = self.norm(x, cond)
        qkv = self.qkv_proj(x)
        q, k, v = rearrange(qkv, 'n h w (t nh e) -> t n nh h w e', t=3, e=self.d_head)
        q, k = scale_for_cosine_sim(q, k, self.scale[:, None, None, None], 1e-06)
        theta = self.pos_emb(pos).movedim(-2, -4)
        q = apply_rotary_emb_(q, theta)
        k = apply_rotary_emb_(k, theta)
        x = apply_window_attention(self.window_size, self.window_shift, q, k, v, scale=1.0)
        x = rearrange(x, 'n nh h w e -> n h w (nh e)')
        x = self.dropout(x)
        x = self.out_proj(x)
        return x + skip


def checkpoint(function, *args, **kwargs):
    if flags.get_checkpointing():
        kwargs.setdefault('use_reentrant', True)
        return torch.utils.checkpoint.checkpoint(function, *args, **kwargs)
    else:
        return function(*args, **kwargs)


class GlobalTransformerLayer(nn.Module):

    def __init__(self, d_model, d_ff, d_head, cond_features, dropout=0.0):
        super().__init__()
        self.self_attn = SelfAttentionBlock(d_model, d_head, cond_features, dropout=dropout)
        self.ff = FeedForwardBlock(d_model, d_ff, cond_features, dropout=dropout)

    def forward(self, x, pos, cond):
        x = checkpoint(self.self_attn, x, pos, cond)
        x = checkpoint(self.ff, x, cond)
        return x


class NeighborhoodTransformerLayer(nn.Module):

    def __init__(self, d_model, d_ff, d_head, cond_features, kernel_size, dropout=0.0):
        super().__init__()
        self.self_attn = NeighborhoodSelfAttentionBlock(d_model, d_head, cond_features, kernel_size, dropout=dropout)
        self.ff = FeedForwardBlock(d_model, d_ff, cond_features, dropout=dropout)

    def forward(self, x, pos, cond):
        x = checkpoint(self.self_attn, x, pos, cond)
        x = checkpoint(self.ff, x, cond)
        return x


class ShiftedWindowTransformerLayer(nn.Module):

    def __init__(self, d_model, d_ff, d_head, cond_features, window_size, index, dropout=0.0):
        super().__init__()
        window_shift = window_size // 2 if index % 2 == 1 else 0
        self.self_attn = ShiftedWindowSelfAttentionBlock(d_model, d_head, cond_features, window_size, window_shift, dropout=dropout)
        self.ff = FeedForwardBlock(d_model, d_ff, cond_features, dropout=dropout)

    def forward(self, x, pos, cond):
        x = checkpoint(self.self_attn, x, pos, cond)
        x = checkpoint(self.ff, x, cond)
        return x


class NoAttentionTransformerLayer(nn.Module):

    def __init__(self, d_model, d_ff, cond_features, dropout=0.0):
        super().__init__()
        self.ff = FeedForwardBlock(d_model, d_ff, cond_features, dropout=dropout)

    def forward(self, x, pos, cond):
        x = checkpoint(self.ff, x, cond)
        return x


class Level(nn.ModuleList):

    def forward(self, x, *args, **kwargs):
        for layer in self:
            x = layer(x, *args, **kwargs)
        return x


class TokenMerge(nn.Module):

    def __init__(self, in_features, out_features, patch_size=(2, 2)):
        super().__init__()
        self.h = patch_size[0]
        self.w = patch_size[1]
        self.proj = apply_wd(Linear(in_features * self.h * self.w, out_features, bias=False))

    def forward(self, x):
        x = rearrange(x, '... (h nh) (w nw) e -> ... h w (nh nw e)', nh=self.h, nw=self.w)
        return self.proj(x)


class TokenSplitWithoutSkip(nn.Module):

    def __init__(self, in_features, out_features, patch_size=(2, 2)):
        super().__init__()
        self.h = patch_size[0]
        self.w = patch_size[1]
        self.proj = apply_wd(Linear(in_features, out_features * self.h * self.w, bias=False))

    def forward(self, x):
        x = self.proj(x)
        return rearrange(x, '... h w (nh nw e) -> ... (h nh) (w nw) e', nh=self.h, nw=self.w)


class TokenSplit(nn.Module):

    def __init__(self, in_features, out_features, patch_size=(2, 2)):
        super().__init__()
        self.h = patch_size[0]
        self.w = patch_size[1]
        self.proj = apply_wd(Linear(in_features, out_features * self.h * self.w, bias=False))
        self.fac = nn.Parameter(torch.ones(1) * 0.5)

    def forward(self, x, skip):
        x = self.proj(x)
        x = rearrange(x, '... h w (nh nw e) -> ... (h nh) (w nw) e', nh=self.h, nw=self.w)
        return torch.lerp(skip, x, self.fac)


@dataclass
class NoAttentionSpec:
    pass


def downscale_pos(pos):
    pos = rearrange(pos, '... (h nh) (w nw) e -> ... h w (nh nw) e', nh=2, nw=2)
    return torch.mean(pos, dim=-2)


class ImageTransformerDenoiserModelV2(nn.Module):

    def __init__(self, levels, mapping, in_channels, out_channels, patch_size, num_classes=0, mapping_cond_dim=0):
        super().__init__()
        self.num_classes = num_classes
        self.patch_in = TokenMerge(in_channels, levels[0].width, patch_size)
        self.time_emb = layers.FourierFeatures(1, mapping.width)
        self.time_in_proj = Linear(mapping.width, mapping.width, bias=False)
        self.aug_emb = layers.FourierFeatures(9, mapping.width)
        self.aug_in_proj = Linear(mapping.width, mapping.width, bias=False)
        self.class_emb = nn.Embedding(num_classes, mapping.width) if num_classes else None
        self.mapping_cond_in_proj = Linear(mapping_cond_dim, mapping.width, bias=False) if mapping_cond_dim else None
        self.mapping = tag_module(MappingNetwork(mapping.depth, mapping.width, mapping.d_ff, dropout=mapping.dropout), 'mapping')
        self.down_levels, self.up_levels = nn.ModuleList(), nn.ModuleList()
        for i, spec in enumerate(levels):
            if isinstance(spec.self_attn, GlobalAttentionSpec):
                layer_factory = lambda _: GlobalTransformerLayer(spec.width, spec.d_ff, spec.self_attn.d_head, mapping.width, dropout=spec.dropout)
            elif isinstance(spec.self_attn, NeighborhoodAttentionSpec):
                layer_factory = lambda _: NeighborhoodTransformerLayer(spec.width, spec.d_ff, spec.self_attn.d_head, mapping.width, spec.self_attn.kernel_size, dropout=spec.dropout)
            elif isinstance(spec.self_attn, ShiftedWindowAttentionSpec):
                layer_factory = lambda i: ShiftedWindowTransformerLayer(spec.width, spec.d_ff, spec.self_attn.d_head, mapping.width, spec.self_attn.window_size, i, dropout=spec.dropout)
            elif isinstance(spec.self_attn, NoAttentionSpec):
                layer_factory = lambda _: NoAttentionTransformerLayer(spec.width, spec.d_ff, mapping.width, dropout=spec.dropout)
            else:
                raise ValueError(f'unsupported self attention spec {spec.self_attn}')
            if i < len(levels) - 1:
                self.down_levels.append(Level([layer_factory(i) for i in range(spec.depth)]))
                self.up_levels.append(Level([layer_factory(i + spec.depth) for i in range(spec.depth)]))
            else:
                self.mid_level = Level([layer_factory(i) for i in range(spec.depth)])
        self.merges = nn.ModuleList([TokenMerge(spec_1.width, spec_2.width) for spec_1, spec_2 in zip(levels[:-1], levels[1:])])
        self.splits = nn.ModuleList([TokenSplit(spec_2.width, spec_1.width) for spec_1, spec_2 in zip(levels[:-1], levels[1:])])
        self.out_norm = RMSNorm(levels[0].width)
        self.patch_out = TokenSplitWithoutSkip(levels[0].width, out_channels, patch_size)
        nn.init.zeros_(self.patch_out.proj.weight)

    def param_groups(self, base_lr=0.0005, mapping_lr_scale=1 / 3):
        wd = filter_params(lambda tags: 'wd' in tags and 'mapping' not in tags, self)
        no_wd = filter_params(lambda tags: 'wd' not in tags and 'mapping' not in tags, self)
        mapping_wd = filter_params(lambda tags: 'wd' in tags and 'mapping' in tags, self)
        mapping_no_wd = filter_params(lambda tags: 'wd' not in tags and 'mapping' in tags, self)
        groups = [{'params': list(wd), 'lr': base_lr}, {'params': list(no_wd), 'lr': base_lr, 'weight_decay': 0.0}, {'params': list(mapping_wd), 'lr': base_lr * mapping_lr_scale}, {'params': list(mapping_no_wd), 'lr': base_lr * mapping_lr_scale, 'weight_decay': 0.0}]
        return groups

    def forward(self, x, sigma, aug_cond=None, class_cond=None, mapping_cond=None):
        x = x.movedim(-3, -1)
        x = self.patch_in(x)
        pos = make_axial_pos(x.shape[-3], x.shape[-2], device=x.device).view(x.shape[-3], x.shape[-2], 2)
        if class_cond is None and self.class_emb is not None:
            raise ValueError('class_cond must be specified if num_classes > 0')
        if mapping_cond is None and self.mapping_cond_in_proj is not None:
            raise ValueError('mapping_cond must be specified if mapping_cond_dim > 0')
        c_noise = torch.log(sigma) / 4
        time_emb = self.time_in_proj(self.time_emb(c_noise[..., None]))
        aug_cond = x.new_zeros([x.shape[0], 9]) if aug_cond is None else aug_cond
        aug_emb = self.aug_in_proj(self.aug_emb(aug_cond))
        class_emb = self.class_emb(class_cond) if self.class_emb is not None else 0
        mapping_emb = self.mapping_cond_in_proj(mapping_cond) if self.mapping_cond_in_proj is not None else 0
        cond = self.mapping(time_emb + aug_emb + class_emb + mapping_emb)
        skips, poses = [], []
        for down_level, merge in zip(self.down_levels, self.merges):
            x = down_level(x, pos, cond)
            skips.append(x)
            poses.append(pos)
            x = merge(x)
            pos = downscale_pos(pos)
        x = self.mid_level(x, pos, cond)
        for up_level, split, skip, pos in reversed(list(zip(self.up_levels, self.splits, skips, poses))):
            x = split(x, skip)
            x = up_level(x, pos, cond)
        x = self.out_norm(x)
        x = self.patch_out(x)
        x = x.movedim(-1, -3)
        return x


def orthogonal_(module):
    nn.init.orthogonal_(module.weight)
    return module


class MappingNet(nn.Sequential):

    def __init__(self, feats_in, feats_out, n_layers=2):
        layers = []
        for i in range(n_layers):
            layers.append(orthogonal_(nn.Linear(feats_in if i == 0 else feats_out, feats_out)))
            layers.append(nn.GELU())
        super().__init__(*layers)


class ImageDenoiserModelV1(nn.Module):

    def __init__(self, c_in, feats_in, depths, channels, self_attn_depths, cross_attn_depths=None, mapping_cond_dim=0, unet_cond_dim=0, cross_cond_dim=0, dropout_rate=0.0, patch_size=1, skip_stages=0, has_variance=False):
        super().__init__()
        self.c_in = c_in
        self.channels = channels
        self.unet_cond_dim = unet_cond_dim
        self.patch_size = patch_size
        self.has_variance = has_variance
        self.timestep_embed = layers.FourierFeatures(1, feats_in)
        if mapping_cond_dim > 0:
            self.mapping_cond = nn.Linear(mapping_cond_dim, feats_in, bias=False)
        self.mapping = MappingNet(feats_in, feats_in)
        self.proj_in = nn.Conv2d((c_in + unet_cond_dim) * self.patch_size ** 2, channels[max(0, skip_stages - 1)], 1)
        self.proj_out = nn.Conv2d(channels[max(0, skip_stages - 1)], c_in * self.patch_size ** 2 + (1 if self.has_variance else 0), 1)
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)
        if cross_cond_dim == 0:
            cross_attn_depths = [False] * len(self_attn_depths)
        d_blocks, u_blocks = [], []
        for i in range(len(depths)):
            my_c_in = channels[max(0, i - 1)]
            d_blocks.append(DBlock(depths[i], feats_in, my_c_in, channels[i], channels[i], downsample=i > skip_stages, self_attn=self_attn_depths[i], cross_attn=cross_attn_depths[i], c_enc=cross_cond_dim, dropout_rate=dropout_rate))
        for i in range(len(depths)):
            my_c_in = channels[i] * 2 if i < len(depths) - 1 else channels[i]
            my_c_out = channels[max(0, i - 1)]
            u_blocks.append(UBlock(depths[i], feats_in, my_c_in, channels[i], my_c_out, upsample=i > skip_stages, self_attn=self_attn_depths[i], cross_attn=cross_attn_depths[i], c_enc=cross_cond_dim, dropout_rate=dropout_rate))
        self.u_net = layers.UNet(d_blocks, reversed(u_blocks), skip_stages=skip_stages)

    def param_groups(self, base_lr=0.0002):
        wd_names = []
        for name, _ in self.named_parameters():
            if name.startswith('mapping') or name.startswith('u_net'):
                if name.endswith('.weight'):
                    wd_names.append(name)
        wd, no_wd = [], []
        for name, param in self.named_parameters():
            if name in wd_names:
                wd.append(param)
            else:
                no_wd.append(param)
        groups = [{'params': wd, 'lr': base_lr}, {'params': no_wd, 'lr': base_lr, 'weight_decay': 0.0}]
        return groups

    def forward(self, input, sigma, mapping_cond=None, unet_cond=None, cross_cond=None, cross_cond_padding=None, return_variance=False):
        c_noise = sigma.log() / 4
        timestep_embed = self.timestep_embed(utils.append_dims(c_noise, 2))
        mapping_cond_embed = torch.zeros_like(timestep_embed) if mapping_cond is None else self.mapping_cond(mapping_cond)
        mapping_out = self.mapping(timestep_embed + mapping_cond_embed)
        cond = {'cond': mapping_out}
        if unet_cond is not None:
            input = torch.cat([input, unet_cond], dim=1)
        if cross_cond is not None:
            cond['cross'] = cross_cond
            cond['cross_padding'] = cross_cond_padding
        if self.patch_size > 1:
            input = F.pixel_unshuffle(input, self.patch_size)
        input = self.proj_in(input)
        input = self.u_net(input, cond)
        input = self.proj_out(input)
        if self.has_variance:
            input, logvar = input[:, :-1], input[:, -1].flatten(1).mean(1)
        if self.patch_size > 1:
            input = F.pixel_shuffle(input, self.patch_size)
        if self.has_variance and return_variance:
            return input, logvar
        return input

    def set_skip_stages(self, skip_stages):
        self.proj_in = nn.Conv2d(self.proj_in.in_channels, self.channels[max(0, skip_stages - 1)], 1)
        self.proj_out = nn.Conv2d(self.channels[max(0, skip_stages - 1)], self.proj_out.out_channels, 1)
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)
        self.u_net.skip_stages = skip_stages
        for i, block in enumerate(self.u_net.d_blocks):
            block.set_downsample(i > skip_stages)
        for i, block in enumerate(reversed(self.u_net.u_blocks)):
            block.set_upsample(i > skip_stages)
        return self

    def set_patch_size(self, patch_size):
        self.patch_size = patch_size
        self.proj_in = nn.Conv2d((self.c_in + self.unet_cond_dim) * self.patch_size ** 2, self.channels[max(0, self.u_net.skip_stages - 1)], 1)
        self.proj_out = nn.Conv2d(self.channels[max(0, self.u_net.skip_stages - 1)], self.c_in * self.patch_size ** 2 + (1 if self.has_variance else 0), 1)
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)


class PIDStepSizeController:
    """A PID controller for ODE adaptive step size control."""

    def __init__(self, h, pcoeff, icoeff, dcoeff, order=1, accept_safety=0.81, eps=1e-08):
        self.h = h
        self.b1 = (pcoeff + icoeff + dcoeff) / order
        self.b2 = -(pcoeff + 2 * dcoeff) / order
        self.b3 = dcoeff / order
        self.accept_safety = accept_safety
        self.eps = eps
        self.errs = []

    def limiter(self, x):
        return 1 + math.atan(x - 1)

    def propose_step(self, error):
        inv_error = 1 / (float(error) + self.eps)
        if not self.errs:
            self.errs = [inv_error, inv_error, inv_error]
        self.errs[0] = inv_error
        factor = self.errs[0] ** self.b1 * self.errs[1] ** self.b2 * self.errs[2] ** self.b3
        factor = self.limiter(factor)
        accept = factor >= self.accept_safety
        if accept:
            self.errs[2] = self.errs[1]
            self.errs[1] = self.errs[0]
        self.h *= factor
        return accept


def default_noise_sampler(x):
    return lambda sigma, sigma_next: torch.randn_like(x)


def get_ancestral_step(sigma_from, sigma_to, eta=1.0):
    """Calculates the noise level (sigma_down) to step down to and the amount
    of noise to add (sigma_up) when doing an ancestral sampling step."""
    if not eta:
        return sigma_to, 0.0
    sigma_up = min(sigma_to, eta * (sigma_to ** 2 * (sigma_from ** 2 - sigma_to ** 2) / sigma_from ** 2) ** 0.5)
    sigma_down = (sigma_to ** 2 - sigma_up ** 2) ** 0.5
    return sigma_down, sigma_up


class DPMSolver(nn.Module):
    """DPM-Solver. See https://arxiv.org/abs/2206.00927."""

    def __init__(self, model, extra_args=None, eps_callback=None, info_callback=None):
        super().__init__()
        self.model = model
        self.extra_args = {} if extra_args is None else extra_args
        self.eps_callback = eps_callback
        self.info_callback = info_callback

    def t(self, sigma):
        return -sigma.log()

    def sigma(self, t):
        return t.neg().exp()

    def eps(self, eps_cache, key, x, t, *args, **kwargs):
        if key in eps_cache:
            return eps_cache[key], eps_cache
        sigma = self.sigma(t) * x.new_ones([x.shape[0]])
        eps = (x - self.model(x, sigma, *args, **self.extra_args, **kwargs)) / self.sigma(t)
        if self.eps_callback is not None:
            self.eps_callback()
        return eps, {key: eps, **eps_cache}

    def dpm_solver_1_step(self, x, t, t_next, eps_cache=None):
        eps_cache = {} if eps_cache is None else eps_cache
        h = t_next - t
        eps, eps_cache = self.eps(eps_cache, 'eps', x, t)
        x_1 = x - self.sigma(t_next) * h.expm1() * eps
        return x_1, eps_cache

    def dpm_solver_2_step(self, x, t, t_next, r1=1 / 2, eps_cache=None):
        eps_cache = {} if eps_cache is None else eps_cache
        h = t_next - t
        eps, eps_cache = self.eps(eps_cache, 'eps', x, t)
        s1 = t + r1 * h
        u1 = x - self.sigma(s1) * (r1 * h).expm1() * eps
        eps_r1, eps_cache = self.eps(eps_cache, 'eps_r1', u1, s1)
        x_2 = x - self.sigma(t_next) * h.expm1() * eps - self.sigma(t_next) / (2 * r1) * h.expm1() * (eps_r1 - eps)
        return x_2, eps_cache

    def dpm_solver_3_step(self, x, t, t_next, r1=1 / 3, r2=2 / 3, eps_cache=None):
        eps_cache = {} if eps_cache is None else eps_cache
        h = t_next - t
        eps, eps_cache = self.eps(eps_cache, 'eps', x, t)
        s1 = t + r1 * h
        s2 = t + r2 * h
        u1 = x - self.sigma(s1) * (r1 * h).expm1() * eps
        eps_r1, eps_cache = self.eps(eps_cache, 'eps_r1', u1, s1)
        u2 = x - self.sigma(s2) * (r2 * h).expm1() * eps - self.sigma(s2) * (r2 / r1) * ((r2 * h).expm1() / (r2 * h) - 1) * (eps_r1 - eps)
        eps_r2, eps_cache = self.eps(eps_cache, 'eps_r2', u2, s2)
        x_3 = x - self.sigma(t_next) * h.expm1() * eps - self.sigma(t_next) / r2 * (h.expm1() / h - 1) * (eps_r2 - eps)
        return x_3, eps_cache

    def dpm_solver_fast(self, x, t_start, t_end, nfe, eta=0.0, s_noise=1.0, noise_sampler=None):
        noise_sampler = default_noise_sampler(x) if noise_sampler is None else noise_sampler
        if not t_end > t_start and eta:
            raise ValueError('eta must be 0 for reverse sampling')
        m = math.floor(nfe / 3) + 1
        ts = torch.linspace(t_start, t_end, m + 1, device=x.device)
        if nfe % 3 == 0:
            orders = [3] * (m - 2) + [2, 1]
        else:
            orders = [3] * (m - 1) + [nfe % 3]
        for i in range(len(orders)):
            eps_cache = {}
            t, t_next = ts[i], ts[i + 1]
            if eta:
                sd, su = get_ancestral_step(self.sigma(t), self.sigma(t_next), eta)
                t_next_ = torch.minimum(t_end, self.t(sd))
                su = (self.sigma(t_next) ** 2 - self.sigma(t_next_) ** 2) ** 0.5
            else:
                t_next_, su = t_next, 0.0
            eps, eps_cache = self.eps(eps_cache, 'eps', x, t)
            denoised = x - self.sigma(t) * eps
            if self.info_callback is not None:
                self.info_callback({'x': x, 'i': i, 't': ts[i], 't_up': t, 'denoised': denoised})
            if orders[i] == 1:
                x, eps_cache = self.dpm_solver_1_step(x, t, t_next_, eps_cache=eps_cache)
            elif orders[i] == 2:
                x, eps_cache = self.dpm_solver_2_step(x, t, t_next_, eps_cache=eps_cache)
            else:
                x, eps_cache = self.dpm_solver_3_step(x, t, t_next_, eps_cache=eps_cache)
            x = x + su * s_noise * noise_sampler(self.sigma(t), self.sigma(t_next))
        return x

    def dpm_solver_adaptive(self, x, t_start, t_end, order=3, rtol=0.05, atol=0.0078, h_init=0.05, pcoeff=0.0, icoeff=1.0, dcoeff=0.0, accept_safety=0.81, eta=0.0, s_noise=1.0, noise_sampler=None):
        noise_sampler = default_noise_sampler(x) if noise_sampler is None else noise_sampler
        if order not in {2, 3}:
            raise ValueError('order should be 2 or 3')
        forward = t_end > t_start
        if not forward and eta:
            raise ValueError('eta must be 0 for reverse sampling')
        h_init = abs(h_init) * (1 if forward else -1)
        atol = torch.tensor(atol)
        rtol = torch.tensor(rtol)
        s = t_start
        x_prev = x
        accept = True
        pid = PIDStepSizeController(h_init, pcoeff, icoeff, dcoeff, 1.5 if eta else order, accept_safety)
        info = {'steps': 0, 'nfe': 0, 'n_accept': 0, 'n_reject': 0}
        while s < t_end - 1e-05 if forward else s > t_end + 1e-05:
            eps_cache = {}
            t = torch.minimum(t_end, s + pid.h) if forward else torch.maximum(t_end, s + pid.h)
            if eta:
                sd, su = get_ancestral_step(self.sigma(s), self.sigma(t), eta)
                t_ = torch.minimum(t_end, self.t(sd))
                su = (self.sigma(t) ** 2 - self.sigma(t_) ** 2) ** 0.5
            else:
                t_, su = t, 0.0
            eps, eps_cache = self.eps(eps_cache, 'eps', x, s)
            denoised = x - self.sigma(s) * eps
            if order == 2:
                x_low, eps_cache = self.dpm_solver_1_step(x, s, t_, eps_cache=eps_cache)
                x_high, eps_cache = self.dpm_solver_2_step(x, s, t_, eps_cache=eps_cache)
            else:
                x_low, eps_cache = self.dpm_solver_2_step(x, s, t_, r1=1 / 3, eps_cache=eps_cache)
                x_high, eps_cache = self.dpm_solver_3_step(x, s, t_, eps_cache=eps_cache)
            delta = torch.maximum(atol, rtol * torch.maximum(x_low.abs(), x_prev.abs()))
            error = torch.linalg.norm((x_low - x_high) / delta) / x.numel() ** 0.5
            accept = pid.propose_step(error)
            if accept:
                x_prev = x_low
                x = x_high + su * s_noise * noise_sampler(self.sigma(s), self.sigma(t))
                s = t
                info['n_accept'] += 1
            else:
                info['n_reject'] += 1
            info['nfe'] += order
            info['steps'] += 1
            if self.info_callback is not None:
                self.info_callback({'x': x, 'i': info['steps'] - 1, 't': s, 't_up': s, 'denoised': denoised, 'error': error, 'h': pid.h, **info})
        return x, info


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (AxialRoPE,
     lambda: ([], {'dim': 4, 'n_heads': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ConditionedResidualBlock,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (ConditionedSequential,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (Downsample2d,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (FourierFeatures,
     lambda: ([], {'in_features': 4, 'out_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Level,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (MappingNet,
     lambda: ([], {'feats_in': 4, 'feats_out': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ResidualBlock,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (UnconditionedModule,
     lambda: ([], {'module': torch.nn.ReLU()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Upsample2d,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
]

