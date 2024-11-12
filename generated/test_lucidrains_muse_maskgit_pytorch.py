
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


from functools import wraps


from collections import namedtuple


import torch


from torch import nn


from torch import einsum


import torch.nn.functional as F


import math


from random import random


from functools import partial


import torchvision.transforms as T


from typing import Callable


from typing import Optional


from typing import List


import logging


from typing import Union


from math import sqrt


from random import choice


from torch.optim import Adam


from torch.utils.data import Dataset


from torch.utils.data import DataLoader


from torch.utils.data import random_split


from torchvision.datasets import ImageFolder


from torchvision.utils import make_grid


from torchvision.utils import save_image


import copy


from torch.autograd import grad as torch_grad


import torchvision


AttentionConfig = namedtuple('AttentionConfig', ['enable_flash', 'enable_math', 'enable_mem_efficient'])


def exists(val):
    return val is not None


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


class Attend(nn.Module):

    def __init__(self, scale=8, dropout=0.0, flash=False):
        super().__init__()
        self.scale = scale
        self.dropout = dropout
        self.attn_dropout = nn.Dropout(dropout)
        self.flash = flash
        assert not (flash and version.parse(torch.__version__) < version.parse('2.0.0')), 'in order to use flash attention, you must be using pytorch 2.0 or above'
        self.cuda_config = None
        self.no_hardware_detected = False
        if not torch.cuda.is_available() or not flash:
            return
        device_properties = torch.cuda.get_device_properties(torch.device('cuda'))
        if device_properties.major == 8 and device_properties.minor == 0:
            print_once('A100 GPU detected, using flash attention if input tensor is on cuda')
            self.cuda_config = AttentionConfig(True, False, False)
        else:
            print_once('Non-A100 GPU detected, using math or mem efficient attention if input tensor is on cuda')
            self.cuda_config = AttentionConfig(False, True, False)

    def flash_attn(self, q, k, v, mask=None):
        default_scale = q.shape[-1] ** -0.5
        is_cuda = q.is_cuda
        q, k, v = map(lambda t: t.contiguous(), (q, k, v))
        rescale = self.scale / default_scale
        q = q * rescale ** 0.5
        k = k * rescale ** 0.5
        use_naive = not is_cuda or not exists(self.cuda_config)
        if not is_cuda or self.no_hardware_detected:
            return FlashAttentionFunction.apply(q, k, v, mask, False, 512, 512)
        try:
            raise Exception()
            with torch.backends.cuda.sdp_kernel(**self.cuda_config._asdict()):
                out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=self.dropout if self.training else 0.0)
        except:
            print_once('no hardware detected, falling back to naive implementation from memory-efficient-attention-pytorch library')
            self.no_hardware_detected = True
            out = FlashAttentionFunction.apply(q, k, v, mask, False, 512, 512)
        return out

    def forward(self, q, k, v, mask=None, force_non_flash=False):
        """
        einstein notation
        b - batch
        h - heads
        n, i, j - sequence length (base sequence length, source, target)
        d - feature dimension
        """
        if self.flash and not force_non_flash:
            return self.flash_attn(q, k, v, mask=mask)
        sim = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        if exists(mask):
            mask_value = -torch.finfo(sim.dtype).max
            sim = sim.masked_fill(~mask, mask_value)
        attn = sim.softmax(dim=-1)
        attn = self.attn_dropout(attn)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        return out


class LayerNorm(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer('beta', torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)


class GEGLU(nn.Module):
    """ https://arxiv.org/abs/2002.05202 """

    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return gate * F.gelu(x)


def l2norm(t):
    return F.normalize(t, dim=-1)


class Attention(nn.Module):

    def __init__(self, dim, dim_head=64, heads=8, cross_attend=False, scale=8, flash=True, dropout=0.0):
        super().__init__()
        self.scale = scale
        self.heads = heads
        inner_dim = dim_head * heads
        self.cross_attend = cross_attend
        self.norm = LayerNorm(dim)
        self.attend = Attend(flash=flash, dropout=dropout, scale=scale)
        self.null_kv = nn.Parameter(torch.randn(2, heads, 1, dim_head))
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.q_scale = nn.Parameter(torch.ones(dim_head))
        self.k_scale = nn.Parameter(torch.ones(dim_head))
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x, context=None, context_mask=None):
        assert not exists(context) ^ self.cross_attend
        n = x.shape[-2]
        h, is_cross_attn = self.heads, exists(context)
        x = self.norm(x)
        kv_input = context if self.cross_attend else x
        q, k, v = self.to_q(x), *self.to_kv(kv_input).chunk(2, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))
        nk, nv = self.null_kv
        nk, nv = map(lambda t: repeat(t, 'h 1 d -> b h 1 d', b=x.shape[0]), (nk, nv))
        k = torch.cat((nk, k), dim=-2)
        v = torch.cat((nv, v), dim=-2)
        q, k = map(l2norm, (q, k))
        q = q * self.q_scale
        k = k * self.k_scale
        if exists(context_mask):
            context_mask = repeat(context_mask, 'b j -> b h i j', h=h, i=n)
            context_mask = F.pad(context_mask, (1, 0), value=True)
        out = self.attend(q, k, v, mask=context_mask)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


def FeedForward(dim, mult=4):
    """ https://arxiv.org/abs/2110.09456 """
    inner_dim = int(dim * mult * 2 / 3)
    return nn.Sequential(LayerNorm(dim), nn.Linear(dim, inner_dim * 2, bias=False), GEGLU(), LayerNorm(inner_dim), nn.Linear(inner_dim, dim, bias=False))


class TransformerBlocks(nn.Module):

    def __init__(self, *, dim, depth, dim_head=64, heads=8, ff_mult=4, flash=True):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([Attention(dim=dim, dim_head=dim_head, heads=heads, flash=flash), Attention(dim=dim, dim_head=dim_head, heads=heads, cross_attend=True, flash=flash), FeedForward(dim=dim, mult=ff_mult)]))
        self.norm = LayerNorm(dim)

    def forward(self, x, context=None, context_mask=None):
        for attn, cross_attn, ff in self.layers:
            x = attn(x) + x
            x = cross_attn(x, context=context, context_mask=context_mask) + x
            x = ff(x) + x
        return self.norm(x)


DEFAULT_T5_NAME = 'google/t5-v1_1-base'


def default(val, d):
    return val if exists(val) else d


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
        assert False
    return config.d_model


def uniform(shape, min=0, max=1, device=None):
    return torch.zeros(shape, device=device).float().uniform_(0, 1)


def prob_mask_like(shape, prob, device=None):
    if prob == 1:
        return torch.ones(shape, device=device, dtype=torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device=device, dtype=torch.bool)
    else:
        return uniform(shape, device=device) < prob


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


class Transformer(nn.Module):

    def __init__(self, *, num_tokens, dim, seq_len, dim_out=None, t5_name=DEFAULT_T5_NAME, self_cond=False, add_mask_id=False, **kwargs):
        super().__init__()
        self.dim = dim
        self.mask_id = num_tokens if add_mask_id else None
        self.num_tokens = num_tokens
        self.token_emb = nn.Embedding(num_tokens + int(add_mask_id), dim)
        self.pos_emb = nn.Embedding(seq_len, dim)
        self.seq_len = seq_len
        self.transformer_blocks = TransformerBlocks(dim=dim, **kwargs)
        self.norm = LayerNorm(dim)
        self.dim_out = default(dim_out, num_tokens)
        self.to_logits = nn.Linear(dim, self.dim_out, bias=False)
        self.encode_text = partial(t5_encode_text, name=t5_name)
        text_embed_dim = get_encoded_dim(t5_name)
        self.text_embed_proj = nn.Linear(text_embed_dim, dim, bias=False) if text_embed_dim != dim else nn.Identity()
        self.self_cond = self_cond
        self.self_cond_to_init_embed = FeedForward(dim)

    def forward_with_cond_scale(self, *args, cond_scale=3.0, return_embed=False, **kwargs):
        if cond_scale == 1:
            return self.forward(*args, return_embed=return_embed, cond_drop_prob=0.0, **kwargs)
        logits, embed = self.forward(*args, return_embed=True, cond_drop_prob=0.0, **kwargs)
        null_logits = self.forward(*args, cond_drop_prob=1.0, **kwargs)
        scaled_logits = null_logits + (logits - null_logits) * cond_scale
        if return_embed:
            return scaled_logits, embed
        return scaled_logits

    def forward_with_neg_prompt(self, text_embed: 'torch.Tensor', neg_text_embed: 'torch.Tensor', cond_scale=3.0, return_embed=False, **kwargs):
        neg_logits = self.forward(*args, neg_text_embed=neg_text_embed, cond_drop_prob=0.0, **kwargs)
        pos_logits, embed = self.forward(*args, return_embed=True, text_embed=text_embed, cond_drop_prob=0.0, **kwargs)
        logits = neg_logits + (pos_logits - neg_logits) * cond_scale
        if return_embed:
            return scaled_logits, embed
        return scaled_logits

    def forward(self, x, return_embed=False, return_logits=False, labels=None, ignore_index=0, self_cond_embed=None, cond_drop_prob=0.0, conditioning_token_ids: 'Optional[torch.Tensor]'=None, texts: 'Optional[List[str]]'=None, text_embeds: 'Optional[torch.Tensor]'=None):
        device, b, n = x.device, *x.shape
        assert n <= self.seq_len
        assert exists(texts) ^ exists(text_embeds)
        if exists(texts):
            text_embeds = self.encode_text(texts)
        context = self.text_embed_proj(text_embeds)
        context_mask = (text_embeds != 0).any(dim=-1)
        if cond_drop_prob > 0.0:
            mask = prob_mask_like((b, 1), 1.0 - cond_drop_prob, device)
            context_mask = context_mask & mask
        if exists(conditioning_token_ids):
            conditioning_token_ids = rearrange(conditioning_token_ids, 'b ... -> b (...)')
            cond_token_emb = self.token_emb(conditioning_token_ids)
            context = torch.cat((context, cond_token_emb), dim=-2)
            context_mask = F.pad(context_mask, (0, conditioning_token_ids.shape[-1]), value=True)
        x = self.token_emb(x)
        x = x + self.pos_emb(torch.arange(n, device=device))
        if self.self_cond:
            if not exists(self_cond_embed):
                self_cond_embed = torch.zeros_like(x)
            x = x + self.self_cond_to_init_embed(self_cond_embed)
        embed = self.transformer_blocks(x, context=context, context_mask=context_mask)
        logits = self.to_logits(embed)
        if return_embed:
            return logits, embed
        if not exists(labels):
            return logits
        if self.dim_out == 1:
            loss = F.binary_cross_entropy_with_logits(rearrange(logits, '... 1 -> ...'), labels)
        else:
            loss = F.cross_entropy(rearrange(logits, 'b n c -> b c n'), labels, ignore_index=ignore_index)
        if not return_logits:
            return loss
        return loss, logits


class SelfCritic(nn.Module):

    def __init__(self, net):
        super().__init__()
        self.net = net
        self.to_pred = nn.Linear(net.dim, 1)

    def forward_with_cond_scale(self, x, *args, **kwargs):
        _, embeds = self.net.forward_with_cond_scale(x, *args, return_embed=True, **kwargs)
        return self.to_pred(embeds)

    def forward_with_neg_prompt(self, x, *args, **kwargs):
        _, embeds = self.net.forward_with_neg_prompt(x, *args, return_embed=True, **kwargs)
        return self.to_pred(embeds)

    def forward(self, x, *args, labels=None, **kwargs):
        _, embeds = self.net(x, *args, return_embed=True, **kwargs)
        logits = self.to_pred(embeds)
        if not exists(labels):
            return logits
        logits = rearrange(logits, '... 1 -> ...')
        return F.binary_cross_entropy_with_logits(logits, labels)


class MaskGitTransformer(Transformer):

    def __init__(self, *args, **kwargs):
        assert 'add_mask_id' not in kwargs
        super().__init__(*args, add_mask_id=True, **kwargs)


class TokenCritic(Transformer):

    def __init__(self, *args, **kwargs):
        assert 'dim_out' not in kwargs
        super().__init__(*args, dim_out=1, **kwargs)


def cosine_schedule(t):
    return torch.cos(t * math.pi * 0.5)


def eval_decorator(fn):

    def inner(model, *args, **kwargs):
        was_training = model.training
        model.eval()
        out = fn(model, *args, **kwargs)
        model.train(was_training)
        return out
    return inner


def get_mask_subset_prob(mask, prob, min_mask=0):
    batch, seq, device = *mask.shape, mask.device
    num_to_mask = (mask.sum(dim=-1, keepdim=True) * prob).clamp(min=min_mask)
    logits = torch.rand((batch, seq), device=device)
    logits = logits.masked_fill(~mask, -1)
    randperm = logits.argsort(dim=-1).argsort(dim=-1).float()
    num_padding = (~mask).sum(dim=-1, keepdim=True)
    randperm -= num_padding
    subset_mask = randperm < num_to_mask
    subset_mask.masked_fill_(~mask, False)
    return subset_mask


def log(t, eps=1e-10):
    return torch.log(t + eps)


def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))


def gumbel_sample(t, temperature=1.0, dim=-1):
    return (t / max(temperature, 1e-10) + gumbel_noise(t)).argmax(dim=dim)


def top_k(logits, thres=0.9):
    k = math.ceil((1 - thres) * logits.shape[-1])
    val, ind = logits.topk(k, dim=-1)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(2, ind, val)
    return probs


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


def accum_log(log, new_logs):
    for key, new_value in new_logs.items():
        old_value = log.get(key, 0.0)
        log[key] = old_value + new_value
    return log


def cycle(dl):
    while True:
        for data in dl:
            yield data


def find_index(arr, cond):
    for ind, el in enumerate(arr):
        if cond(el):
            return ind
    return None


def find_and_pop(arr, cond, default=None):
    ind = find_index(arr, cond)
    if exists(ind):
        return arr.pop(ind)
    if callable(default):
        return default()
    return default


def noop(*args, **kwargs):
    pass


def yes_or_no(question):
    answer = input(f'{question} (y/n) ')
    return answer.lower() in ('yes', 'y')


class LayerNormChan(nn.Module):

    def __init__(self, dim, eps=1e-05):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) * var.clamp(min=self.eps).rsqrt() * self.gamma


MList = nn.ModuleList


def leaky_relu(p=0.1):
    return nn.LeakyReLU(0.1)


class Discriminator(nn.Module):

    def __init__(self, dims, channels=3, groups=16, init_kernel_size=5):
        super().__init__()
        dim_pairs = zip(dims[:-1], dims[1:])
        self.layers = MList([nn.Sequential(nn.Conv2d(channels, dims[0], init_kernel_size, padding=init_kernel_size // 2), leaky_relu())])
        for dim_in, dim_out in dim_pairs:
            self.layers.append(nn.Sequential(nn.Conv2d(dim_in, dim_out, 4, stride=2, padding=1), nn.GroupNorm(groups, dim_out), leaky_relu()))
        dim = dims[-1]
        self.to_logits = nn.Sequential(nn.Conv2d(dim, dim, 1), leaky_relu(), nn.Conv2d(dim, 1, 4))

    def forward(self, x):
        for net in self.layers:
            x = net(x)
        return self.to_logits(x)


class GLUResBlock(nn.Module):

    def __init__(self, chan, groups=16):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(chan, chan * 2, 3, padding=1), nn.GLU(dim=1), nn.GroupNorm(groups, chan), nn.Conv2d(chan, chan * 2, 3, padding=1), nn.GLU(dim=1), nn.GroupNorm(groups, chan), nn.Conv2d(chan, chan, 1))

    def forward(self, x):
        return self.net(x) + x


class ResBlock(nn.Module):

    def __init__(self, chan, groups=16):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(chan, chan, 3, padding=1), nn.GroupNorm(groups, chan), leaky_relu(), nn.Conv2d(chan, chan, 3, padding=1), nn.GroupNorm(groups, chan), leaky_relu(), nn.Conv2d(chan, chan, 1))

    def forward(self, x):
        return self.net(x) + x


class ResnetEncDec(nn.Module):

    def __init__(self, dim, *, channels=3, layers=4, layer_mults=None, num_resnet_blocks=1, resnet_groups=16, first_conv_kernel_size=5):
        super().__init__()
        assert dim % resnet_groups == 0, f'dimension {dim} must be divisible by {resnet_groups} (groups for the groupnorm)'
        self.layers = layers
        self.encoders = MList([])
        self.decoders = MList([])
        layer_mults = default(layer_mults, list(map(lambda t: 2 ** t, range(layers))))
        assert len(layer_mults) == layers, 'layer multipliers must be equal to designated number of layers'
        layer_dims = [(dim * mult) for mult in layer_mults]
        dims = dim, *layer_dims
        self.encoded_dim = dims[-1]
        dim_pairs = zip(dims[:-1], dims[1:])
        append = lambda arr, t: arr.append(t)
        prepend = lambda arr, t: arr.insert(0, t)
        if not isinstance(num_resnet_blocks, tuple):
            num_resnet_blocks = *((0,) * (layers - 1)), num_resnet_blocks
        assert len(num_resnet_blocks) == layers, 'number of resnet blocks config must be equal to number of layers'
        for layer_index, (dim_in, dim_out), layer_num_resnet_blocks in zip(range(layers), dim_pairs, num_resnet_blocks):
            append(self.encoders, nn.Sequential(nn.Conv2d(dim_in, dim_out, 4, stride=2, padding=1), leaky_relu()))
            prepend(self.decoders, nn.Sequential(nn.ConvTranspose2d(dim_out, dim_in, 4, 2, 1), leaky_relu()))
            for _ in range(layer_num_resnet_blocks):
                append(self.encoders, ResBlock(dim_out, groups=resnet_groups))
                prepend(self.decoders, GLUResBlock(dim_out, groups=resnet_groups))
        prepend(self.encoders, nn.Conv2d(channels, dim, first_conv_kernel_size, padding=first_conv_kernel_size // 2))
        append(self.decoders, nn.Conv2d(dim, channels, 1))

    def get_encoded_fmap_size(self, image_size):
        return image_size // 2 ** self.layers

    @property
    def last_dec_layer(self):
        return self.decoders[-1].weight

    def encode(self, x):
        for enc in self.encoders:
            x = enc(x)
        return x

    def decode(self, x):
        for dec in self.decoders:
            x = dec(x)
        return x


def bce_discr_loss(fake, real):
    return (-log(1 - torch.sigmoid(fake)) - log(torch.sigmoid(real))).mean()


def bce_gen_loss(fake):
    return -log(torch.sigmoid(fake)).mean()


def grad_layer_wrt_loss(loss, layer):
    return torch_grad(outputs=loss, inputs=layer, grad_outputs=torch.ones_like(loss), retain_graph=True)[0].detach()


def gradient_penalty(images, output, weight=10):
    batch_size = images.shape[0]
    gradients = torch_grad(outputs=output, inputs=images, grad_outputs=torch.ones(output.size(), device=images.device), create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = rearrange(gradients, 'b ... -> b (...)')
    return weight * ((gradients.norm(2, dim=1) - 1) ** 2).mean()


def group_dict_by_key(cond, d):
    return_val = [dict(), dict()]
    for key in d.keys():
        match = bool(cond(key))
        ind = int(not match)
        return_val[ind][key] = d[key]
    return *return_val,


def string_begins_with(prefix, string_input):
    return string_input.startswith(prefix)


def groupby_prefix_and_trim(prefix, d):
    kwargs_with_prefix, kwargs = group_dict_by_key(partial(string_begins_with, prefix), d)
    kwargs_without_prefix = dict(map(lambda x: (x[0][len(prefix):], x[1]), tuple(kwargs_with_prefix.items())))
    return kwargs_without_prefix, kwargs


def hinge_discr_loss(fake, real):
    return (F.relu(1 + fake) + F.relu(1 - real)).mean()


def hinge_gen_loss(fake):
    return -fake.mean()


def remove_vgg(fn):

    @wraps(fn)
    def inner(self, *args, **kwargs):
        has_vgg = hasattr(self, '_vgg')
        if has_vgg:
            vgg = self._vgg
            delattr(self, '_vgg')
        out = fn(self, *args, **kwargs)
        if has_vgg:
            self._vgg = vgg
        return out
    return inner


def safe_div(numer, denom, eps=1e-08):
    return numer / denom.clamp(min=eps)


class VQGanVAE(nn.Module):

    def __init__(self, *, dim, channels=3, layers=4, l2_recon_loss=False, use_hinge_loss=True, vgg=None, lookup_free_quantization=True, codebook_size=65536, vq_kwargs: dict=dict(codebook_dim=256, decay=0.8, commitment_weight=1.0, kmeans_init=True, use_cosine_sim=True), lfq_kwargs: dict=dict(diversity_gamma=4.0), use_vgg_and_gan=True, discr_layers=4, **kwargs):
        super().__init__()
        vq_kwargs, kwargs = groupby_prefix_and_trim('vq_', kwargs)
        encdec_kwargs, kwargs = groupby_prefix_and_trim('encdec_', kwargs)
        self.channels = channels
        self.codebook_size = codebook_size
        self.dim_divisor = 2 ** layers
        enc_dec_klass = ResnetEncDec
        self.enc_dec = enc_dec_klass(dim=dim, channels=channels, layers=layers, **encdec_kwargs)
        self.lookup_free_quantization = lookup_free_quantization
        if lookup_free_quantization:
            self.quantizer = LFQ(dim=self.enc_dec.encoded_dim, codebook_size=codebook_size, **lfq_kwargs)
        else:
            self.quantizer = VQ(dim=self.enc_dec.encoded_dim, codebook_size=codebook_size, accept_image_fmap=True ** vq_kwargs)
        self.recon_loss_fn = F.mse_loss if l2_recon_loss else F.l1_loss
        self._vgg = None
        self.discr = None
        self.use_vgg_and_gan = use_vgg_and_gan
        if not use_vgg_and_gan:
            return
        if exists(vgg):
            self._vgg = vgg
        layer_mults = list(map(lambda t: 2 ** t, range(discr_layers)))
        layer_dims = [(dim * mult) for mult in layer_mults]
        dims = dim, *layer_dims
        self.discr = Discriminator(dims=dims, channels=channels)
        self.discr_loss = hinge_discr_loss if use_hinge_loss else bce_discr_loss
        self.gen_loss = hinge_gen_loss if use_hinge_loss else bce_gen_loss

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def vgg(self):
        if exists(self._vgg):
            return self._vgg
        vgg = torchvision.models.vgg16(pretrained=True)
        vgg.classifier = nn.Sequential(*vgg.classifier[:-2])
        self._vgg = vgg
        return self._vgg

    @property
    def encoded_dim(self):
        return self.enc_dec.encoded_dim

    def get_encoded_fmap_size(self, image_size):
        return self.enc_dec.get_encoded_fmap_size(image_size)

    def copy_for_eval(self):
        device = next(self.parameters()).device
        vae_copy = copy.deepcopy(self.cpu())
        if vae_copy.use_vgg_and_gan:
            del vae_copy.discr
            del vae_copy._vgg
        vae_copy.eval()
        return vae_copy

    @remove_vgg
    def state_dict(self, *args, **kwargs):
        return super().state_dict(*args, **kwargs)

    @remove_vgg
    def load_state_dict(self, *args, **kwargs):
        return super().load_state_dict(*args, **kwargs)

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        path = Path(path)
        assert path.exists()
        state_dict = torch.load(str(path))
        self.load_state_dict(state_dict)

    def encode(self, fmap):
        fmap = self.enc_dec.encode(fmap)
        fmap, indices, vq_aux_loss = self.quantizer(fmap)
        return fmap, indices, vq_aux_loss

    def decode_from_ids(self, ids):
        if self.lookup_free_quantization:
            ids, ps = pack([ids], 'b *')
            fmap = self.quantizer.indices_to_codes(ids)
            fmap, = unpack(fmap, ps, 'b * c')
        else:
            codes = self.codebook[ids]
            fmap = self.quantizer.project_out(codes)
        fmap = rearrange(fmap, 'b h w c -> b c h w')
        return self.decode(fmap)

    def decode(self, fmap):
        return self.enc_dec.decode(fmap)

    def forward(self, img, return_loss=False, return_discr_loss=False, return_recons=False, add_gradient_penalty=True):
        batch, channels, height, width, device = *img.shape, img.device
        for dim_name, size in (('height', height), ('width', width)):
            assert size % self.dim_divisor == 0, f'{dim_name} must be divisible by {self.dim_divisor}'
        assert channels == self.channels, 'number of channels on image or sketch is not equal to the channels set on this VQGanVAE'
        fmap, indices, commit_loss = self.encode(img)
        fmap = self.decode(fmap)
        if not return_loss and not return_discr_loss:
            return fmap
        assert return_loss ^ return_discr_loss, 'you should either return autoencoder loss or discriminator loss, but not both'
        if return_discr_loss:
            assert exists(self.discr), 'discriminator must exist to train it'
            fmap.detach_()
            img.requires_grad_()
            fmap_discr_logits, img_discr_logits = map(self.discr, (fmap, img))
            discr_loss = self.discr_loss(fmap_discr_logits, img_discr_logits)
            if add_gradient_penalty:
                gp = gradient_penalty(img, img_discr_logits)
                loss = discr_loss + gp
            if return_recons:
                return loss, fmap
            return loss
        recon_loss = self.recon_loss_fn(fmap, img)
        if not self.use_vgg_and_gan:
            if return_recons:
                return recon_loss, fmap
            return recon_loss
        img_vgg_input = img
        fmap_vgg_input = fmap
        if img.shape[1] == 1:
            img_vgg_input, fmap_vgg_input = map(lambda t: repeat(t, 'b 1 ... -> b c ...', c=3), (img_vgg_input, fmap_vgg_input))
        img_vgg_feats = self.vgg(img_vgg_input)
        recon_vgg_feats = self.vgg(fmap_vgg_input)
        perceptual_loss = F.mse_loss(img_vgg_feats, recon_vgg_feats)
        gen_loss = self.gen_loss(self.discr(fmap))
        last_dec_layer = self.enc_dec.last_dec_layer
        norm_grad_wrt_gen_loss = grad_layer_wrt_loss(gen_loss, last_dec_layer).norm(p=2)
        norm_grad_wrt_perceptual_loss = grad_layer_wrt_loss(perceptual_loss, last_dec_layer).norm(p=2)
        adaptive_weight = safe_div(norm_grad_wrt_perceptual_loss, norm_grad_wrt_gen_loss)
        adaptive_weight.clamp_(max=10000.0)
        loss = recon_loss + perceptual_loss + commit_loss + adaptive_weight * gen_loss
        if return_recons:
            return loss, fmap
        return loss


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (Attend,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (GEGLU,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (LayerNorm,
     lambda: ([], {'dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (LayerNormChan,
     lambda: ([], {'dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
]

