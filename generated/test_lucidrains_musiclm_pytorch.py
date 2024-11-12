
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


from torch import nn


from torch.autograd import Function


import torch.distributed as dist


import math


from functools import wraps


from functools import partial


import torch.nn.functional as F


from torch import einsum


import copy


from math import sqrt


from random import choice


from torch.optim import Adam


from torch.utils.data import Dataset


from torch.utils.data import DataLoader


from torch.utils.data import random_split


from torch.nn.utils.rnn import pad_sequence


def all_gather_same_dim(t):
    world_size = dist.get_world_size()
    gathered_tensors = [torch.empty_like(t, device=t.device, dtype=t.dtype) for i in range(world_size)]
    dist.all_gather(gathered_tensors, t)
    return gathered_tensors


def exists(val):
    return val is not None


def all_gather_variable_dim(t, dim=0, sizes=None):
    device, rank, world_size = t.device, dist.get_rank(), dist.get_world_size()
    if not exists(sizes):
        size = torch.tensor(t.shape[dim], device=device, dtype=torch.long)
        sizes = all_gather_same_dim(size)
        sizes = torch.stack(sizes)
    if torch.unique(sizes).numel() == 1:
        gathered_tensors = all_gather_same_dim(t)
        return torch.cat(gathered_tensors, dim=dim), sizes
    max_size = sizes.amax().item()
    padded_t = pad_dim_to(t, max_size, dim=dim)
    gathered_tensors = all_gather_same_dim(padded_t)
    gathered_tensor = torch.cat(gathered_tensors, dim=dim)
    seq = torch.arange(max_size, device=device)
    mask = rearrange(seq, 'j -> 1 j') < rearrange(sizes, 'i -> i 1')
    mask = rearrange(mask, 'i j -> (i j)')
    seq = torch.arange(mask.shape[-1], device=device)
    indices = seq[mask]
    gathered_tensor = gathered_tensor.index_select(dim, indices)
    return gathered_tensor, sizes


class AllGatherFunction(Function):

    @staticmethod
    def forward(ctx, x, dim, sizes, all_reduce_grads):
        x, batch_sizes = all_gather_variable_dim(x, dim=dim, sizes=sizes)
        ctx.dim = dim
        ctx.all_reduce_grads = all_reduce_grads
        ctx.batch_sizes = batch_sizes.tolist()
        return x, batch_sizes

    @staticmethod
    def backward(ctx, grads, _):
        batch_sizes, rank = ctx.batch_sizes, dist.get_rank()
        if ctx.all_reduce_grads:
            dist.all_reduce(grads)
        grads_by_rank = grads.split(batch_sizes, dim=ctx.dim)
        return grads_by_rank[rank], None, None, None


class AllGather(nn.Module):

    def __init__(self, dim, *, all_reduce_grads=False):
        super().__init__()
        self.dim = dim
        self.all_reduce_grads = all_reduce_grads
        self.is_distributed = dist.is_initialized() and dist.get_world_size() > 1

    def forward(self, x, sizes=None):
        if not self.is_distributed:
            return x, None
        return AllGatherFunction.apply(x, self.dim, sizes, self.all_reduce_grads)


def default(*args):
    for arg in args:
        if exists(arg):
            return arg
    return None


class LayerNorm(nn.Module):

    def __init__(self, dim, scale=True):
        super().__init__()
        self.learned_gamma = nn.Parameter(torch.ones(dim)) if scale else None
        self.register_buffer('gamma', torch.ones(dim), persistent=False)
        self.register_buffer('beta', torch.zeros(dim), persistent=False)

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], default(self.learned_gamma, self.gamma), self.beta)


class GEGLU(nn.Module):

    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.gelu(gate) * x


def l2norm(t):
    return F.normalize(t, p=2, dim=-1)


class Attention(nn.Module):

    def __init__(self, dim, causal=False, dim_head=64, heads=8, dropout=0.0, scale=8):
        super().__init__()
        self.heads = heads
        self.scale = scale
        self.causal = causal
        inner_dim = dim_head * heads
        self.norm = LayerNorm(dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.q_scale = nn.Parameter(torch.ones(dim_head))
        self.k_scale = nn.Parameter(torch.ones(dim_head))
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim, bias=False), nn.Dropout(dropout))

    def forward(self, x, rel_pos_bias=None, mask=None):
        b, n, _, device = *x.shape, x.device
        x = self.norm(x)
        q, k, v = self.to_q(x), *self.to_kv(x).chunk(2, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), (q, k, v))
        q, k = map(l2norm, (q, k))
        q = q * self.q_scale
        k = k * self.k_scale
        sim = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        if exists(rel_pos_bias):
            sim = sim + rel_pos_bias
        if exists(mask):
            mask = rearrange(mask, 'b j -> b 1 1 j')
            sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)
        if self.causal:
            i, j = sim.shape[-2:]
            causal_mask = torch.ones((i, j), dtype=torch.bool, device=x.device).triu(j - i + 1)
            sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)
        attn = sim.softmax(dim=-1)
        attn = self.attn_dropout(attn)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


def FeedForward(dim, mult=4, dropout=0.0):
    dim_hidden = int(dim * mult * 2 / 3)
    return nn.Sequential(LayerNorm(dim), nn.Linear(dim, dim_hidden * 2, bias=False), GEGLU(), nn.Dropout(dropout), nn.Linear(dim_hidden, dim, bias=False))


class Transformer(nn.Module):

    def __init__(self, dim, depth, dim_head=64, heads=8, attn_dropout=0.0, ff_mult=4, ff_dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([Attention(dim=dim, dim_head=dim_head, heads=heads, dropout=attn_dropout), FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout)]))

    def forward(self, x, rel_pos_bias=None, mask=None, return_all_layers=False):
        layers = []
        for attn, ff in self.layers:
            x = attn(x, rel_pos_bias=rel_pos_bias, mask=mask) + x
            x = ff(x) + x
            layers.append(x)
        if not return_all_layers:
            return x
        return x, torch.stack(layers[:-1])


def log(t, eps=1e-20):
    return torch.log(t.clamp(min=eps))


def matrix_diag(t):
    device = t.device
    i, j = t.shape[-2:]
    num_diag_el = min(i, j)
    i_range = torch.arange(i, device=device)
    j_range = torch.arange(j, device=device)
    diag_mask = rearrange(i_range, 'i -> i 1') == rearrange(j_range, 'j -> 1 j')
    diag_el = t.masked_select(diag_mask)
    return rearrange(diag_el, '(b d) -> b d', d=num_diag_el)


class SoftmaxContrastiveLearning(nn.Module):

    def __init__(self, *, layers=1, decoupled_contrastive_learning=False, init_temp=10):
        super().__init__()
        self.temperatures = nn.Parameter(torch.ones(layers, 1, 1) * math.log(init_temp))
        self.decoupled_contrastive_learning = decoupled_contrastive_learning
        self.all_gather = AllGather(dim=2)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, audio_latents, text_latents):
        if audio_latents.ndim == 2:
            audio_latents = rearrange(audio_latents, '... -> 1 ...')
        if text_latents.ndim == 2:
            text_latents = rearrange(text_latents, '... -> 1 ...')
        batch = audio_latents.shape[1]
        if self.all_gather.is_distributed:
            latents = torch.stack((audio_latents, text_latents))
            latents, _ = self.all_gather(latents)
            audio_latents, text_latents = latents
        sims = einsum('l i d, l j d -> l i j', audio_latents, text_latents)
        sims = sims * self.temperatures.exp()
        cosine_sims_exp = sims.exp()
        numerator = matrix_diag(cosine_sims_exp)
        if self.decoupled_contrastive_learning:
            eye = torch.eye(batch, device=self.device, dtype=torch.bool)
            cosine_sims_exp = cosine_sims_exp.masked_fill(eye, 0.0)
        denominator_i = reduce(cosine_sims_exp, 'l i j -> l i', 'sum')
        denominator_j = reduce(cosine_sims_exp, 'l i j -> l j', 'sum')
        contrastive_loss = -log(numerator) + 0.5 * (log(denominator_i) + log(denominator_j))
        contrastive_loss = reduce(contrastive_loss, 'l n -> l', 'mean')
        return contrastive_loss.sum()


class SigmoidContrastiveLearning(nn.Module):
    """ https://arxiv.org/abs/2303.15343 """

    def __init__(self, *, layers=1, init_temp=10, init_bias=-10):
        super().__init__()
        self.temperatures = nn.Parameter(torch.ones(layers, 1, 1) * math.log(init_temp))
        self.bias = nn.Parameter(torch.ones(layers, 1, 1) * init_bias)
        self.all_gather = AllGather(dim=1, all_reduce_grads=True)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, audio_latents, text_latents):
        device = self.device
        if audio_latents.ndim == 2:
            audio_latents = rearrange(audio_latents, '... -> 1 ...')
        if text_latents.ndim == 2:
            text_latents = rearrange(text_latents, '... -> 1 ...')
        text_latents, rank_sizes = self.all_gather(text_latents)
        n = text_latents.shape[1]
        sims = einsum('l i d, l j d -> l i j', audio_latents, text_latents)
        sims = sims * self.temperatures.exp() + self.bias
        labels = torch.eye(n, device=device)
        if exists(rank_sizes):
            labels_by_ranks = labels.split(rank_sizes.tolist(), dim=0)
            labels = labels_by_ranks[dist.get_rank()]
        labels = 2 * rearrange(labels, 'i j -> 1 i j') - torch.ones_like(sims)
        return -F.logsigmoid(labels * sims).sum() / n


def Sequential(*modules):
    return nn.Sequential(*filter(exists, modules))


def pair(t):
    return (t, t) if not isinstance(t, tuple) else t


def posemb_sincos_2d(patches, temperature=10000, dtype=torch.float32):
    _, h, w, dim, device, dtype = *patches.shape, patches.device, patches.dtype
    y, x = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing='ij')
    assert dim % 4 == 0, 'feature dimension must be multiple of 4 for sincos emb'
    omega = torch.arange(dim // 4, device=device) / (dim // 4 - 1)
    omega = 1.0 / temperature ** omega
    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    pe = pe.type(dtype)
    return rearrange(pe, '(h w) d -> h w d', h=h, w=w)


def once(fn):
    called = False

    @wraps(fn)
    def inner(x):
        nonlocal called
        if called:
            return
        called = True
        return fn(x)
    return inner


print_once = once(print)


def round_down_nearest_multiple(n, divisor):
    return n // divisor * divisor


class AudioSpectrogramTransformer(nn.Module):

    def __init__(self, dim, depth, patch_size=16, dim_head=64, heads=8, attn_dropout=0.0, ff_mult=4, ff_dropout=0.0, accept_spec=False, accept_spec_time_first=True, spec_n_fft=128, spec_power=2, spec_win_length=24, spec_hop_length=None, spec_pad=0, spec_center=True, spec_pad_mode='reflect', spec_aug_stretch_factor=0.8, spec_aug_freq_mask=80, spec_aug_time_mask=80, patch_dropout_prob=0.25):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.patch_size = pair(patch_size)
        patch_input_dim = self.patch_size[0] * self.patch_size[1]
        self.to_patch_tokens = Sequential(Rearrange('b (h p1) (w p2) -> b h w (p1 p2)', p1=self.patch_size[0], p2=self.patch_size[1]), nn.LayerNorm(patch_input_dim), nn.Linear(patch_input_dim, dim), nn.LayerNorm(dim))
        self.accept_spec = accept_spec
        self.accept_spec_time_first = accept_spec_time_first
        self.spec = Spectrogram(n_fft=spec_n_fft, power=spec_power, win_length=spec_win_length, hop_length=spec_hop_length, pad=spec_pad, center=spec_center, pad_mode=spec_pad_mode)
        self.aug = torch.nn.Sequential(TimeStretch(spec_aug_stretch_factor, fixed_rate=True), FrequencyMasking(freq_mask_param=spec_aug_freq_mask), TimeMasking(time_mask_param=spec_aug_time_mask))
        self.transformer = Transformer(dim=dim, depth=depth, dim_head=dim_head, heads=heads, attn_dropout=attn_dropout, ff_mult=ff_mult, ff_dropout=ff_dropout)
        self.norm = LayerNorm(dim)
        self.patch_dropout_prob = patch_dropout_prob
        mlp_hidden_dim = dim // 4
        self.dynamic_pos_bias_mlp = nn.Sequential(nn.Linear(2, mlp_hidden_dim), nn.SiLU(), nn.Linear(mlp_hidden_dim, mlp_hidden_dim), nn.SiLU(), nn.Linear(mlp_hidden_dim, heads), Rearrange('... i j h -> ... h i j'))

    def forward(self, x, force_no_patch_dropout=False, return_all_layers=False):
        batch, device = x.shape[0], x.device
        assert self.accept_spec and x.ndim == 3 or not self.accept_spec and x.ndim == 2
        if self.accept_spec and self.accept_spec_time_first:
            x = rearrange(x, 'b t f -> b f t')
        if not self.accept_spec:
            x = self.spec(x)
        if self.training:
            x = self.aug(x)
        height, width = x.shape[-2:]
        patch_height, patch_width = self.patch_size
        rounded_height, rounded_width = map(lambda args: round_down_nearest_multiple(*args), ((height, patch_height), (width, patch_width)))
        if (height, width) != (rounded_height, rounded_width):
            print_once(f'spectrogram yielded shape of {height, width}, but had to be cropped to {rounded_height, rounded_width} to be patchified for transformer')
        x = x[..., :rounded_height, :rounded_width]
        x = self.to_patch_tokens(x)
        _, num_patch_height, num_patch_width, _ = x.shape
        grid = torch.stack(torch.meshgrid(torch.arange(num_patch_height, device=device), torch.arange(num_patch_width, device=device), indexing='ij'), dim=-1)
        grid = rearrange(grid, '... c -> (...) c')
        x = x + posemb_sincos_2d(x)
        x = rearrange(x, 'b ... c -> b (...) c')
        if self.training and self.patch_dropout_prob > 0.0 and not force_no_patch_dropout:
            n, device = x.shape[1], x.device
            batch_indices = torch.arange(batch, device=device)
            batch_indices = rearrange(batch_indices, '... -> ... 1')
            num_patches_keep = max(1, int(n * (1 - self.patch_dropout_prob)))
            patch_indices_keep = torch.randn(batch, n, device=device).topk(num_patches_keep, dim=-1).indices
            x = x[batch_indices, patch_indices_keep]
            grid = repeat(grid, '... -> b ...', b=batch)
            grid = grid[batch_indices, patch_indices_keep]
        rel_dist = rearrange(grid, '... i c -> ... i 1 c') - rearrange(grid, '... j c -> ... 1 j c')
        rel_pos_bias = self.dynamic_pos_bias_mlp(rel_dist.float())
        x, all_layers = self.transformer(x, rel_pos_bias=rel_pos_bias, return_all_layers=True)
        x = reduce(x, 'b n d -> b d', 'mean')
        out = self.norm(x)
        if not return_all_layers:
            return out
        return out, all_layers


class MultiLayerContrastiveLoss(nn.Module):

    def __init__(self, *, audio_dim, text_dim, dim_latent, layers, decoupled_contrastive_learning=False, sigmoid_contrastive_loss=False):
        super().__init__()
        self.layers = layers
        self.audio_norm = LayerNorm(audio_dim, scale=False)
        self.audio_gamma = nn.Parameter(torch.ones(layers, 1, audio_dim))
        self.audio_latent_weight = nn.Parameter(torch.randn(layers, audio_dim, dim_latent))
        self.audio_latent_bias = nn.Parameter(torch.randn(layers, 1, dim_latent))
        self.text_norm = LayerNorm(text_dim, scale=False)
        self.text_gamma = nn.Parameter(torch.ones(layers, 1, text_dim))
        self.text_latent_weight = nn.Parameter(torch.randn(layers, text_dim, dim_latent))
        self.text_latent_bias = nn.Parameter(torch.randn(layers, 1, dim_latent))
        klass = SigmoidContrastiveLearning if sigmoid_contrastive_loss else partial(SoftmaxContrastiveLearning, decoupled_contrastive_learning=decoupled_contrastive_learning)
        self.contrast = klass(layers=layers)

    def forward(self, *, audio_layers, text_layers):
        device, batch = audio_layers.device, audio_layers.shape[1]
        audio_gap = reduce(audio_layers, 'l b n d -> l b d', 'mean')
        audio_embeds = self.audio_norm(audio_gap) * self.audio_gamma
        audio_latents = einsum('l b d, l d e -> l b e', audio_embeds, self.audio_latent_weight) + self.audio_latent_bias
        audio_latents = l2norm(audio_latents)
        text_cls_tokens = text_layers[:, :, 0]
        text_embeds = self.text_norm(text_cls_tokens) * self.text_gamma
        text_latents = einsum('l b d, l d e -> l b e', text_embeds, self.text_latent_weight) + self.text_latent_bias
        text_latents = l2norm(text_latents)
        return self.contrast(audio_latents, text_latents)


def interspersed_indices(layers, total_layers):
    assert total_layers >= layers
    step = total_layers / layers
    return (torch.arange(0, layers) * step).floor().long()


def first(it):
    return it[0]


def accum_log(log, new_logs):
    for key, new_value in new_logs.items():
        old_value = log.get(key, 0.0)
        log[key] = old_value + new_value
    return log


def cycle(dl):
    while True:
        for data in dl:
            yield data


def determine_types(data, config):
    output = []
    for el in data:
        for name, data_type in config.items():
            if is_bearable(el, data_type):
                output.append(name)
                break
        else:
            raise TypeError(f'unable to determine type of {data}')
    return tuple(output)


def collate_one_or_multiple_tensors(fn):

    @wraps(fn)
    def inner(data):
        is_one_data = not isinstance(data[0], tuple)
        if is_one_data:
            data = torch.stack(data)
            return data,
        outputs = []
        for datum in zip(*data):
            if is_bearable(datum, Tuple[str, ...]):
                output = list(datum)
            else:
                output = fn(datum)
            outputs.append(output)
        return tuple(outputs)
    return inner


@collate_one_or_multiple_tensors
def curtail_to_shortest_collate(data):
    min_len = min(*[datum.shape[0] for datum in data])
    data = [datum[:min_len] for datum in data]
    return torch.stack(data)


@collate_one_or_multiple_tensors
def pad_to_longest_fn(data):
    return pad_sequence(data, batch_first=True)


def get_dataloader(ds, pad_to_longest=True, **kwargs):
    collate_fn = pad_to_longest_fn if pad_to_longest else curtail_to_shortest_collate
    return DataLoader(ds, collate_fn=collate_fn, **kwargs)


def has_duplicates(tup):
    counts = dict()
    for el in tup:
        if el not in counts:
            counts[el] = 0
        counts[el] += 1
    return any(filter(lambda count: count > 1, counts.values()))


def noop(*args, **kwargs):
    pass


def yes_or_no(question):
    answer = input(f'{question} (y/n) ')
    return answer.lower() in ('yes', 'y')


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (AllGather,
     lambda: ([], {'dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (GEGLU,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (LayerNorm,
     lambda: ([], {'dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
]

