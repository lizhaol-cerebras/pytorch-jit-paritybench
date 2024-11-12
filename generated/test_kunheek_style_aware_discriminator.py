
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


from torchvision import transforms


from torchvision.transforms.functional import to_pil_image


from torchvision.utils import make_grid


import torch.nn.functional as F


import torchvision.transforms.functional as VF


import torch.nn as nn


from torch.utils.data import DataLoader


import time


import math


import random


from copy import deepcopy


from torchvision.utils import save_image


from torch.autograd import Function


from torch.nn import functional as F


from torch.utils.cpp_extension import load


from collections import abc


from typing import Iterable


from typing import Sequence


import numpy as np


from torchvision.datasets import LSUNClass


import matplotlib.pyplot as plt


from sklearn.manifold import TSNE


import torch.distributed as dist


class SwappedPredictionLoss(nn.Module):

    def __init__(self, temperature, eps=0.05):
        super().__init__()
        self.register_buffer('temperature', torch.as_tensor(temperature))
        self.register_buffer('eps', torch.as_tensor(eps))
        if torch.distributed.is_initialized():
            self.world_size = torch.distributed.get_world_size()
        else:
            self.world_size = 1

    def forward(self, scores, code_ids, score_ids, bs=999, n_iters=3):
        loss = 0.0
        for s, t in zip(code_ids, score_ids):
            q = self.sinkhorn_knopp(scores[s], n_iters)[:bs]
            logp = F.log_softmax(scores[t][:bs] / self.temperature, dim=1)
            loss -= torch.sum(q * logp, dim=1).mean()
        return loss

    @torch.no_grad()
    def sinkhorn_knopp(self, scores, n_iters):
        Q = torch.exp(scores / self.eps).T
        K = Q.size(0)
        B = Q.size(1) * self.world_size
        sum_Q = torch.sum(Q)
        if self.world_size > 1:
            torch.distributed.all_reduce(sum_Q)
        Q /= sum_Q
        for _ in range(n_iters):
            sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
            if self.world_size > 1:
                torch.distributed.all_reduce(sum_of_rows)
            Q /= sum_of_rows
            Q /= K
            Q /= torch.sum(Q, dim=0, keepdim=True)
            Q /= B
        Q *= B
        return Q.T


class ReconstructionLoss(nn.Module):

    def __init__(self, modes, *lpips_args, **lpips_kwargs):
        super().__init__()
        self.modes = list(map(lambda x: x.lower(), modes))
        if 'lpips' in self.modes:
            self.lpips = lpips.LPIPS(*lpips_args, **lpips_kwargs)
            self.eval()
            self.requires_grad_(False)

    def forward(self, input, target, *args, **kwargs):
        loss = 0.0
        if 'l1' in self.modes:
            loss += F.l1_loss(input, target)
        if 'mse' in self.modes:
            loss += F.mse_loss(input, target)
        if 'lpips' in self.modes:
            loss += self.lpips(input, target).mean()
        if len(self.modes) > 1:
            loss /= len(self.modes)
        return loss


class LinearLayer(nn.Linear):

    def __init__(self, in_features, out_features, bias=True, bias_init=0.0, lr_mul=1.0, activation=True):
        assert 0.0 < lr_mul <= 1.0
        self.lr_mul = lr_mul
        self.bias_init = bias_init
        self.activation = activation
        super().__init__(in_features, out_features, bias)

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.weight)
        if hasattr(self, 'lr_mul') and self.lr_mul < 1.0:
            with torch.no_grad():
                self.weight.div_(self.lr_mul)
        if self.bias is not None:
            nn.init.constant_(self.bias, self.bias_init)

    def forward(self, input):
        if hasattr(self, 'lr_mul') and self.lr_mul < 1.0:
            weight = self.weight * self.lr_mul
            if self.bias is not None:
                bias = self.bias * self.lr_mul
        else:
            weight = self.weight
            bias = self.bias
        out = F.linear(input, weight, bias)
        if self.activation:
            out = F.leaky_relu(out, 0.2, True)
        return out


class MappingNetwork(nn.Sequential):

    def __init__(self, in_features, out_features, depth=8):
        layers = []
        for i in range(depth):
            dim_in = in_features if i == 0 else out_features
            layers.append(LinearLayer(dim_in, out_features, lr_mul=0.01))
        super().__init__(*layers)


def normalize_2nd_moment(input, dim=1, eps=1e-08):
    return input * input.square().mean(dim, keepdim=True).add(eps).rsqrt()


class PixelNorm(nn.Module):

    def __init__(self, eps=1e-08):
        super().__init__()
        self.register_buffer('eps', torch.as_tensor(float(eps)))

    def forward(self, input):
        return normalize_2nd_moment(input, dim=1, eps=self.eps)


class HighPassFilter(nn.Module):

    def __init__(self, w_hpf=1.0):
        super().__init__()
        filter = torch.as_tensor([[-1.0, -1.0, -1.0], [-1.0, 8.0, -1.0], [-1.0, -1.0, -1.0]]).unsqueeze(0).unsqueeze(0)
        self.register_buffer('filter', filter / w_hpf)

    def forward(self, x):
        filter = self.filter.expand(x.size(1), 1, 3, 3)
        return F.conv2d(x, filter, padding=1, groups=x.size(1))


class VectorNorm(nn.Module):

    def forward(self, input):
        assert input.dim() == 2
        return F.normalize(input, p=2, dim=1)


class Projector(nn.Sequential):

    def __init__(self, in_features, hidden_features, out_features):
        if isinstance(hidden_features, int):
            hidden_features = [hidden_features]
        layers = []
        for i, out_feats in enumerate(hidden_features):
            in_feats = in_features if i == 0 else out_feats
            layers.append(nn.Linear(in_feats, out_feats))
            layers.append(nn.LeakyReLU(0.2, True))
        layers.append(nn.Linear(out_feats, out_features))
        layers.append(VectorNorm())
        super().__init__(*layers)


class Discriminator(nn.Sequential):

    def __init__(self, in_features, hidden_features):
        if isinstance(hidden_features, int):
            hidden_features = [hidden_features]
        layers = []
        for i, out_feats in enumerate(hidden_features):
            in_feats = in_features if i == 0 else out_feats
            layers.append(nn.Linear(in_feats, out_feats))
            layers.append(nn.LeakyReLU(0.2, True))
        layers.append(nn.Linear(out_feats, 1))
        super().__init__(*layers)


def conv1x1(in_channels, out_channels, **kwargs):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, **kwargs)


def conv3x3(in_channels, out_channels, stride=1, **kwargs):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, **kwargs)


class DiscriminatorBlock(nn.Module):

    def __init__(self, in_channels, out_channels, downsample=False, **conv_kwargs):
        super().__init__()
        self.register_buffer('gain', torch.rsqrt(torch.as_tensor(2.0)))
        activation = nn.LeakyReLU(0.2, inplace=False)
        residual = []
        residual.append(conv3x3(in_channels, in_channels, **conv_kwargs))
        residual.append(activation)
        residual.append(conv3x3(in_channels, out_channels, **conv_kwargs))
        residual.append(activation)
        if downsample:
            residual.append(nn.AvgPool2d(kernel_size=2))
        self.residual = nn.Sequential(*residual)
        shortcut = []
        if in_channels != out_channels:
            shortcut.append(conv1x1(in_channels, out_channels, bias=False))
        if downsample:
            shortcut.append(nn.AvgPool2d(kernel_size=2))
        self.shortcut = nn.Sequential(*shortcut)

    def forward(self, x):
        x = self.shortcut(x) + self.residual(x)
        return self.gain * x


def kaiming_init(module):
    assert isinstance(module, nn.Module)
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        if module.weight.requires_grad:
            nn.init.kaiming_normal_(module.weight, a=0.2, mode='fan_in')
        if module.bias is not None and module.bias.requires_grad:
            nn.init.zeros_(module.bias)


class StyleDiscriminator(nn.Module):

    def __init__(self, image_size, latent_dim, big_disc=False, channel_multiplier=1, channel_max=512):
        super().__init__()
        channels = {(2 ** i): min(int(2 ** (14 - i) * channel_multiplier), channel_max) for i in range(2, 11)}
        activation = nn.LeakyReLU(0.2)
        encoder = [conv1x1(3, channels[image_size]), activation]
        while image_size > 4:
            in_channels = channels[image_size]
            image_size //= 2
            out_channels = channels[image_size]
            encoder.append(DiscriminatorBlock(in_channels, out_channels, downsample=True))
        self.feat_dim = out_channels
        encoder.append(conv3x3(out_channels, out_channels))
        encoder.append(activation)
        if big_disc:
            self.encoder = nn.Sequential(*encoder)
            self.projector = Projector(out_channels * 4 * 4, out_channels, latent_dim)
            self.discriminator = Discriminator(out_channels * 4 * 4, out_channels)
        else:
            encoder.append(nn.AdaptiveAvgPool2d(output_size=(1, 1)))
            self.encoder = nn.Sequential(*encoder)
            self.projector = Projector(out_channels, out_channels, latent_dim)
            self.discriminator = Discriminator(out_channels, out_channels)
        self.apply(kaiming_init)

    def forward(self, *args, command='_forward_impl', **kwargs):
        return getattr(self, command)(*args, **kwargs)

    def _forward_impl(self, input):
        features = self.encoder(input).view(input.size(0), -1)
        embedding = self.projector(features)
        logit = self.discriminator(features)
        return embedding, logit

    def discriminate(self, input):
        return self.discriminator(self.get_features(input))

    def encode(self, input):
        return self.projector(self.get_features(input))

    def get_features(self, input):
        return self.encoder(input).view(input.size(0), -1)


class Blur(nn.Module):

    def __init__(self, kernel, pad, scale_factor=1):
        super().__init__()
        k = self.make_kernel(kernel)
        if scale_factor > 1:
            kernel = kernel * scale_factor ** 2
        self.register_buffer('kernel', k)
        self.pad = pad
        self.up = 1
        self.down = 1

    def forward(self, input):
        return ops.upfirdn2d(input, self.kernel, self.up, self.down, self.pad)

    @staticmethod
    def make_kernel(k):
        k = torch.as_tensor(k, dtype=torch.float32)
        if k.dim() == 1:
            k = k[None, :] * k[:, None]
        k /= k.sum()
        return k


class Downsample(Blur):

    def __init__(self, kernel, factor=2):
        p = len(kernel) - factor
        pad0 = (p + 1) // 2
        pad1 = p // 2
        super().__init__(kernel, (pad0, pad1))
        self.down = factor


class EncodeBlock(nn.Module):

    def __init__(self, in_channels, out_channels, downsample=True, **conv_kwargs):
        super().__init__()
        self.register_buffer('gain', torch.rsqrt(torch.as_tensor(2.0)))
        activation = nn.LeakyReLU(0.2, inplace=True)
        residual = []
        residual.append(PixelNorm())
        residual.append(activation)
        residual.append(conv3x3(in_channels, in_channels, **conv_kwargs))
        residual.append(PixelNorm())
        residual.append(activation)
        residual.append(conv3x3(in_channels, out_channels, **conv_kwargs))
        if downsample:
            residual.append(Downsample(kernel=[1, 3, 3, 1]))
        self.residual = nn.Sequential(*residual)
        shortcut = []
        if in_channels != out_channels:
            shortcut.append(conv1x1(in_channels, out_channels, bias=False))
        if downsample:
            shortcut.append(Downsample(kernel=[1, 3, 3, 1]))
        self.shortcut = nn.Sequential(*shortcut)

    def forward(self, x):
        x = self.shortcut(x) + self.residual(x)
        return self.gain * x


class ModulatedConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, style_dim, demodulate=True, upsample=False, blur_kernel=[1, 3, 3, 1]):
        super().__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.demodulate = demodulate
        self.upsample = upsample
        self.weight_shape = -1, in_channels, kernel_size, kernel_size
        self.register_buffer('eps', torch.as_tensor(1e-08))
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        nn.init.kaiming_normal_(self.weight, a=0.2)
        self.affine = LinearLayer(style_dim, in_channels, bias_init=1.0)
        if upsample:
            factor = 2
            p = len(blur_kernel) - factor - (kernel_size - 1)
            pad0 = (p + 1) // 2 + factor - 1
            pad1 = p // 2 + 1
            self.blur = Blur(blur_kernel, pad=(pad0, pad1), scale_factor=factor)

    def forward(self, input, style):
        batch, in_channels, height, width = input.size()
        weight = self.weight.unsqueeze(0)
        style = self.affine(style).view(batch, 1, -1, 1, 1)
        weight = weight * style
        if self.demodulate:
            dcoefs = (weight.square().sum((2, 3, 4)) + self.eps).rsqrt()
            weight = weight * dcoefs.view(batch, -1, 1, 1, 1)
        weight = weight.view(*self.weight_shape)
        input = input.view(1, -1, height, width)
        if self.upsample:
            weight = weight.view(batch, *self.weight_shape)
            weight = weight.transpose(1, 2).reshape(batch * in_channels, -1, self.kernel_size, self.kernel_size)
            output = F.conv_transpose2d(input, weight, stride=2, padding=0, groups=batch)
            output = output.view(batch, -1, output.size(2), output.size(3))
            output = self.blur(output)
        else:
            output = F.conv2d(input, weight, padding=self.padding, groups=batch)
            output = output.view(batch, -1, height, width)
        return output


class NoiseInjection(nn.Module):

    def __init__(self, in_channels):
        super().__init__()
        scale_factor = torch.zeros(1, in_channels, 1, 1)
        self.scale_factor = nn.Parameter(scale_factor)
        self.noise = None
        self.cache_noise = False

    def forward(self, image):
        if self.noise is None:
            batch, _, height, width = image.size()
            noise = torch.randn(batch, 1, height, width, device=image.device)
            if self.cache_noise:
                self.noise = noise
        else:
            noise = self.noise
        return image + self.scale_factor * noise


class StyleConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, **conv_kwargs):
        super().__init__()
        self.conv = ModulatedConv2d(in_channels, out_channels, kernel_size, **conv_kwargs)
        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.noise = NoiseInjection(out_channels)
        self.register_buffer('one', torch.ones(1))

    def forward(self, input, style_code, mask=None):
        if mask is None:
            x = self._conv(input, style_code)
        else:
            assert input.size(0) * 2 == style_code.size(0)
            x = self._masked_conv(input, style_code, mask)
        x = ops.fused_leaky_relu(x, self.bias)
        return x

    def _conv(self, input, style_code):
        x = self.conv(input, style_code)
        x = self.noise(x)
        return x

    def _masked_conv(self, input, style_codes, mask):
        s0, s1 = torch.chunk(style_codes, chunks=2)
        x0 = self.conv(input, s0)
        x0 = self.noise(x0)
        if mask.size(2) != x0.size(2):
            mask = F.interpolate(mask, size=x0.size(2), mode='nearest')
        x0 = mask * x0
        x1 = self.conv(input, s1)
        x1 = self.noise(x1)
        x1 = (self.one - mask) * x1
        return x0 + x1


class Upsample(Blur):

    def __init__(self, kernel, factor=2):
        p = len(kernel) - factor
        pad0 = (p + 1) // 2 + factor - 1
        pad1 = p // 2
        super().__init__(kernel, (pad0, pad1), factor)
        self.up = factor


class ToRGB(nn.Module):

    def __init__(self, in_channels, style_dim, upsample=True):
        super().__init__()
        if upsample:
            self.upsample = Upsample([1, 3, 3, 1], factor=2)
        self.conv = ModulatedConv2d(in_channels, 3, 1, style_dim, demodulate=False)
        self.bias = nn.Parameter(torch.zeros(1, 3, 1, 1))
        self.register_buffer('one', torch.ones(1))

    def forward(self, input, style_code, skip=None, mask=None):
        if mask is None:
            out = self.conv(input, style_code)
        else:
            assert input.size(0) * 2 == style_code.size(0)
            if mask.size(2) != input.size(2):
                mask = F.interpolate(mask, size=input.size(2), mode='nearest')
            out = self._masked_conv(input, style_code, mask)
        out = out + self.bias
        if skip is not None and hasattr(self, 'upsample'):
            skip = self.upsample(skip)
            out = out + skip
        return out

    def _masked_conv(self, input, style_codes, mask):
        s0, s1 = torch.chunk(style_codes, chunks=2)
        x0 = self.conv(input, s0)
        x0 = mask * x0
        x1 = self.conv(input, s1)
        x1 = (self.one - mask) * x1
        return x0 + x1


class StyleBlock(nn.Module):

    def __init__(self, in_channels, out_channels, style_dim, upsample=False, architecture='skip'):
        super().__init__()
        assert architecture in ('skip',)
        self.architecture = architecture
        self.conv0 = StyleConv(in_channels, out_channels, 3, style_dim=style_dim, upsample=upsample)
        self.conv1 = StyleConv(out_channels, out_channels, 3, style_dim=style_dim, upsample=False)
        self.to_rgb = ToRGB(out_channels, style_dim, in_channels != out_channels)

    def forward(self, x, style_code, skip=None, mask=None):
        x = self.conv0(x, style_code, mask)
        x = self.conv1(x, style_code, mask)
        skip = self.to_rgb(x, style_code, skip, mask=mask)
        return x, skip


@torch.no_grad()
def concat_all_gather(tensor, world_size):
    tensors_gather = [torch.ones_like(tensor) for _ in range(world_size)]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
    output = torch.cat(tensors_gather, dim=0)
    return output


class Queue(nn.Module):

    def __init__(self, queue_size, feature_dim):
        super().__init__()
        self.register_buffer('data', torch.zeros(queue_size, feature_dim))
        self.register_buffer('ptr', torch.zeros(1, dtype=torch.long))
        self.queue_size = queue_size
        self.feature_dim = feature_dim
        self.empty = True
        if torch.distributed.is_initialized():
            self.world_size = torch.distributed.get_world_size()
        else:
            self.world_size = 1

    @torch.no_grad()
    def forward(self, keys):
        if self.world_size > 1:
            keys = concat_all_gather(keys, self.world_size)
        batch_size = keys.size(0)
        ptr = int(self.ptr)
        assert self.queue_size % batch_size == 0
        self.data[ptr:ptr + batch_size, :] = keys
        self.ptr[0] = (ptr + batch_size) % self.queue_size
        self.empty = False

    def reset(self):
        self.ptr[0] = 0
        self.empty = True

    def extra_repr(self):
        return 'queue_size={}, feature_dim={}, ptr={}, empty={}'.format(self.queue_size, self.feature_dim, int(self.ptr), self.empty)


class Prototypes(nn.Linear):

    def __init__(self, in_features, num_prototype, num_queue=0, queue_size=0):
        super().__init__(in_features, num_prototype, bias=False)
        self.num_prototypes = num_prototype
        self.normalize()
        self.queues = nn.ModuleList()
        if queue_size > 0:
            for _ in range(num_queue):
                self.queues.append(Queue(queue_size, in_features))

    def forward(self, *args, command='_forward_impl', **kwargs):
        return getattr(self, command)(*args, **kwargs)

    def _forward_impl(self, input, queue_ids=[None]):
        if self.queues:
            if not isinstance(input, (list, tuple)):
                input = input,
            return self._queue_forward(input, queue_ids)
        elif isinstance(input, (list, tuple)):
            scores = super().forward(torch.cat(input))
            return torch.chunk(scores, chunks=len(input))
        return super().forward(input)

    def _queue_forward(self, input, queue_ids=[None]):
        assert len(input) == len(queue_ids)
        scores = []
        for feat, i in zip(input, queue_ids):
            score = super().forward(feat)
            if i is None or self.queues[i].empty:
                scores.append(score)
            else:
                with torch.no_grad():
                    score_q = super().forward(self.queues[i].data)
                scores.append(torch.cat((score, score_q)))
            if i is not None:
                self.queues[i](feat)
        return scores

    @torch.no_grad()
    def normalize(self):
        w = self.weight.data.clone()
        w = F.normalize(w, p=2, dim=1)
        self.weight.copy_(w)

    @torch.no_grad()
    def interpolate(self, indices, weights=(0.1, 0.3, 0.5, 0.7, 0.9)):
        assert len(indices) == 2
        z1, z2 = self.weight.data[indices].clone().detach()
        zs = []
        for w in weights:
            z_lerp = torch.lerp(z1, z2, w)
            zs.append(z_lerp)
        zs = torch.stack(zs)
        return F.normalize(zs, p=2, dim=1)

    @torch.no_grad()
    def sample(self, batch_size, proto_ids=None, target_ids=None, mode=''):
        assert mode in ('', 'interpolation', 'perturbation')
        if mode == '':
            mode = 'interpolation' if torch.rand(1) < 0.5 else 'perturbation'
        if proto_ids is None:
            proto_ids = torch.randint(0, self.num_prototypes, (batch_size,))
        prototypes = self.weight.data[proto_ids].clone().detach()
        if mode == 'interpolation':
            if target_ids is None:
                target_ids = torch.randint(0, self.num_prototypes, (batch_size,))
            targets = self.weight.data[target_ids].clone().detach()
            weights = torch.rand((batch_size, 1), device=prototypes.device)
            samples = torch.lerp(prototypes, targets, weights)
        else:
            eps = 0.01 * torch.randn_like(prototypes)
            samples = prototypes + eps
        return F.normalize(samples, p=2, dim=1)


class MultiPrototypes(nn.ModuleList):

    def __init__(self, in_features, num_prototypes, num_queue=0, queue_sizes=0):
        assert type(num_prototypes) == type(queue_sizes)
        if isinstance(num_prototypes, int):
            num_prototypes = [num_prototypes]
            queue_sizes = [queue_sizes]
        assert len(num_prototypes) == len(queue_sizes)
        prototypes = []
        for n_proto, q_size in zip(num_prototypes, queue_sizes):
            prototypes.append(Prototypes(in_features, n_proto, num_queue, q_size))
        super().__init__(prototypes)

    def forward(self, *args, command='_forward_impl', **kwargs):
        return getattr(self, command)(*args, **kwargs)

    def _forward_impl(self, *args, **kwargs):
        outputs = []
        for module in self:
            outputs.append(module(*args, **kwargs))
        return outputs

    def normalize(self):
        for module in self:
            module.normalize()

    def interpolate(self, *args, id=0, **kwargs):
        return self[id].interpolate(*args, **kwargs)

    def sample(self, *args, id=0, **kwargs):
        return self[id].sample(*args, **kwargs)


class AdaIN2d(nn.Module):

    def __init__(self, in_features, style_dim):
        super().__init__()
        self.norm = nn.InstanceNorm2d(in_features)
        self.fc = nn.Linear(style_dim, in_features * 2)
        self.register_buffer('one', torch.ones(1))

    def forward(self, input, style_code, mask=None):
        if mask is None:
            return self._forward(input, style_code)
        else:
            assert input.size(0) * 2 == style_code.size(0)
            if mask.size(2) != input.size(2):
                mask = F.interpolate(mask, size=input.size(2), mode='nearest')
            return self._masked_forward(input, style_code, mask)

    def _forward(self, input, style_code):
        h = self.fc(style_code)
        h = h.view(*h.size(), 1, 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        return (self.one + gamma) * self.norm(input) + beta

    def _masked_forward(self, input, style_codes, mask):
        h = self.fc(style_codes)
        h = h.view(*h.size(), 1, 1)
        h0, h1 = torch.chunk(h, chunks=2)
        x = self.norm(input)
        gamma0, b0 = torch.chunk(h0, chunks=2, dim=1)
        output0 = (self.one + gamma0) * x + b0
        output0 = mask * output0
        gamma1, b1 = torch.chunk(h1, chunks=2, dim=1)
        output1 = (self.one + gamma1) * x + b1
        output1 = (self.one - mask) * output1
        return output0 + output1


class FusedLeakyReLUFunctionBackward(Function):

    @staticmethod
    def forward(ctx, grad_output, out, bias, negative_slope, scale):
        ctx.save_for_backward(out)
        ctx.negative_slope = negative_slope
        ctx.scale = scale
        empty = grad_output.new_empty(0)
        grad_input = fused.fused_bias_act(grad_output.contiguous(), empty, out, 3, 1, negative_slope, scale)
        dim = [0]
        if grad_input.ndim > 2:
            dim += list(range(2, grad_input.ndim))
        if bias:
            grad_bias = grad_input.sum(dim).detach()
        else:
            grad_bias = empty
        return grad_input, grad_bias

    @staticmethod
    def backward(ctx, gradgrad_input, gradgrad_bias):
        out, = ctx.saved_tensors
        gradgrad_out = fused.fused_bias_act(gradgrad_input.contiguous(), gradgrad_bias, out, 3, 1, ctx.negative_slope, ctx.scale)
        return gradgrad_out, None, None, None, None


class FusedLeakyReLUFunction(Function):

    @staticmethod
    def forward(ctx, input, bias, negative_slope, scale):
        empty = input.new_empty(0)
        ctx.bias = bias is not None
        if bias is None:
            bias = empty
        out = fused.fused_bias_act(input, bias, empty, 3, 0, negative_slope, scale)
        ctx.save_for_backward(out)
        ctx.negative_slope = negative_slope
        ctx.scale = scale
        return out

    @staticmethod
    def backward(ctx, grad_output):
        out, = ctx.saved_tensors
        grad_input, grad_bias = FusedLeakyReLUFunctionBackward.apply(grad_output, out, ctx.bias, ctx.negative_slope, ctx.scale)
        if not ctx.bias:
            grad_bias = None
        return grad_input, grad_bias, None, None


def fused_leaky_relu(input, bias=None, negative_slope=0.2, scale=2 ** 0.5):
    if input.device.type == 'cpu':
        if bias is not None:
            rest_dim = [1] * (input.ndim - bias.ndim - 1)
            return F.leaky_relu(input + bias.view(1, bias.shape[0], *rest_dim), negative_slope=0.2) * scale
        else:
            return F.leaky_relu(input, negative_slope=0.2) * scale
    else:
        return FusedLeakyReLUFunction.apply(input.contiguous(), bias, negative_slope, scale)


class FusedLeakyReLU(nn.Module):

    def __init__(self, channel, bias=True, negative_slope=0.2, scale=2 ** 0.5):
        super().__init__()
        if bias:
            self.bias = nn.Parameter(torch.zeros(channel))
        else:
            self.bias = None
        self.negative_slope = negative_slope
        self.scale = scale

    def forward(self, input):
        return fused_leaky_relu(input, self.bias, self.negative_slope, self.scale)


def pad_random_affine(img, angle, translate, scale, shear, interpolation):
    """
        img (PIL Image or Tensor): Image to be transformed.

    Returns:
        PIL Image or Tensor: Affine transformed image.
    """
    img_size = VF.get_image_size(img)
    if torch.rand(1) < 0.8 and angle != 0.0:
        angle = torch.empty(1).uniform_(-angle, angle)
        angle = float(angle.item())
    else:
        angle = 0.0
    if torch.rand(1) < 0.8 and translate != (0, 0):
        max_dx = float(translate[0] * img_size[0])
        max_dy = float(translate[1] * img_size[1])
        tx = int(round(torch.empty(1).uniform_(-max_dx, max_dx).item()))
        ty = int(round(torch.empty(1).uniform_(-max_dy, max_dy).item()))
        translations = tx, ty
    else:
        translations = 0, 0
    if torch.rand(1) < 0.8 and not isinstance(scale, float):
        scale = float(torch.empty(1).uniform_(*scale).item())
    else:
        scale = 1.0
    shear_x = shear_y = 0.0
    if torch.rand(1) < 0.8 and shear != (0, 0):
        x = shear[0]
        shear_x = float(torch.empty(1).uniform_(-x, x).item())
        if shear[1] != 0:
            shear_y = float(torch.empty(1).uniform_(-shear[1], shear[1]).item())
        shear = shear_x, shear_y
    pad = round(math.sin(math.radians(abs(angle))) * max(img_size))
    pad += max(abs(translations[0]), abs(translations[1]))
    if scale < 1.0:
        pad += round(max(img_size) * (1.0 - scale) * 0.5)
    if shear != (0.0, 0.0):
        pad += round(math.tan(math.radians(max(shear))) * max(img_size))
    img = VF.pad(img, pad, padding_mode='reflect')
    img = VF.affine(img, angle, translations, scale, shear, interpolation)
    img = VF.crop(img, pad, pad, img_size[1], img_size[0])
    return img


class PadRandomAffine(nn.Module):

    def __init__(self, angle=0, translate=(0, 0), scale=1.0, shear=(0, 0)):
        super().__init__()
        assert isinstance(angle, (int, float))
        assert isinstance(translate, (tuple, list))
        assert isinstance(scale, (int, float, tuple, list))
        assert isinstance(shear, (tuple, list))
        self.angle = float(angle)
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.interpolation = transforms.InterpolationMode.BILINEAR

    def forward(self, img):
        return pad_random_affine(img, self.angle, self.translate, self.scale, self.shear, self.interpolation)


class Resize(nn.ModuleDict):

    def __init__(self, load_size, crop_size, p_crop=0.5):
        interpolation = transforms.InterpolationMode.BICUBIC
        self.p_crop = p_crop
        super().__init__({'load': transforms.Resize(load_size, interpolation), 'centercrop': transforms.CenterCrop((crop_size,) * 2), 'randcrop': transforms.RandomCrop((crop_size,) * 2), 'resize': transforms.Resize((crop_size,) * 2, interpolation)})

    def forward(self, image):
        w, h = image.size
        image = self['load'](image)
        if torch.rand(1) < self.p_crop:
            image = self['randcrop'](image)
        elif w != h:
            image = self['centercrop'](image)
        else:
            image = self['resize'](image)
        return image


class BaseModel(nn.Module):

    def __init__(self, options):
        super().__init__()
        assert hasattr(options, 'run_dir')
        self.opt = options
        self.device = 'cpu'
        if torch.distributed.is_initialized():
            self.rank = torch.distributed.get_rank()
            self.world_size = torch.distributed.get_world_size()
        else:
            self.rank = 0
            self.world_size = 1
        self.loss = {}
        self._create_networks()
        self._create_criterions()
        self._configure_gpu()
        self._create_optimizer()

    def _create_networks(self):
        raise NotImplementedError

    def _create_criterions(self):
        pass

    def _configure_gpu(self):
        self.device = torch.device(self.rank)
        self
        if self.world_size > 1:
            for name, module in self.named_children():
                if torch_utils.count_parameters(module) > 0:
                    module = nn.parallel.DistributedDataParallel(module, device_ids=[self.rank], broadcast_buffers=False, find_unused_parameters=True)
                    setattr(self, name, module)

    def _create_optimizer(self):
        raise NotImplementedError

    def training_step(self, *args, **kwargs):
        raise NotImplementedError

    def get_state(self, ignore=None, **kwargs):
        if ignore is None or isinstance(ignore, str):
            ignore = ignore,
        elif not isinstance(ignore, Iterable):
            raise ValueError('ignore should string or iterable.')
        state = {'options': self.opt}
        state.update(**kwargs)
        for name, net in self.named_children():
            if name in ignore:
                continue
            net = net.module if hasattr(net, 'module') else net
            state[name + '_state_dict'] = net.state_dict()
        if isinstance(self.optimizer, torch.optim.Optimizer):
            state['optimizer'] = self.optimizer.state_dict()
        else:
            for name, optim in self.optimizer.items():
                state[name + '_optimizer'] = optim.state_dict()
        return state

    def load(self, checkpoint=None):
        if checkpoint is None:
            None
            return
        None
        None
        for name, net in self.named_children():
            key = name + '_state_dict'
            if key in checkpoint.keys():
                state_dict = checkpoint[key]
                net = net.module if hasattr(net, 'module') else net
                mis, unex = net.load_state_dict(state_dict, False)
                None
                num_mis, num_unex = len(mis), len(unex)
                if num_mis > 0:
                    None
                    None
                if num_unex > 0:
                    None
                    None
            else:
                None
        for name, opt in self.optimizer.items():
            key = name + '_optimizer'
            if key in checkpoint.keys():
                opt.load_state_dict(checkpoint[key])
                None
            else:
                None
        None


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (AdaIN2d,
     lambda: ([], {'in_features': 4, 'style_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4])], {})),
    (Discriminator,
     lambda: ([], {'in_features': 4, 'hidden_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (DiscriminatorBlock,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (FusedLeakyReLU,
     lambda: ([], {'channel': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (HighPassFilter,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (LinearLayer,
     lambda: ([], {'in_features': 4, 'out_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (MappingNetwork,
     lambda: ([], {'in_features': 4, 'out_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (MultiPrototypes,
     lambda: ([], {'in_features': 4, 'num_prototypes': 4}),
     lambda: ([], {'input': torch.rand([4, 4])})),
    (NoiseInjection,
     lambda: ([], {'in_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (PadRandomAffine,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (PixelNorm,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Projector,
     lambda: ([], {'in_features': 4, 'hidden_features': 4, 'out_features': 4}),
     lambda: ([torch.rand([4, 4])], {})),
    (Prototypes,
     lambda: ([], {'in_features': 4, 'num_prototype': 4}),
     lambda: ([], {'input': torch.rand([4, 4])})),
    (Queue,
     lambda: ([], {'queue_size': 4, 'feature_dim': 4}),
     lambda: ([torch.rand([4, 4])], {})),
    (SwappedPredictionLoss,
     lambda: ([], {'temperature': 4}),
     lambda: ([torch.ones([4, 4], dtype=torch.int64), torch.ones([4, 4], dtype=torch.int64), torch.ones([4, 4], dtype=torch.int64)], {})),
    (ToRGB,
     lambda: ([], {'in_channels': 4, 'style_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4])], {})),
    (VectorNorm,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4])], {})),
]

