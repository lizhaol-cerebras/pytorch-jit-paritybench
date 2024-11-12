
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


from typing import Iterable


from typing import Optional


import torch


import numpy as np


import time


import torch.backends.cudnn as cudnn


from torch.utils.tensorboard import SummaryWriter


import torchvision.transforms as transforms


import torchvision.datasets as datasets


from functools import partial


import torch.nn as nn


from torchvision import transforms


from torchvision.transforms import functional as F


from collections import defaultdict


from collections import deque


import torch.distributed as dist


from itertools import islice


from typing import Callable


from typing import Tuple


import torch.optim as optim


import logging


from collections import OrderedDict


import torchvision.utils


from torch.nn.parallel import DistributedDataParallel as NativeDDP


from collections.abc import Collection


from typing import Any


from typing import List


import torch.optim


from collections import Counter


import functools


import torch.nn.functional as F


import itertools


from torch.nn.parallel import DataParallel


from torch.nn.parallel._functions import Scatter


from torch.nn.parallel.parallel_apply import parallel_apply


from torch import nn


from torch import Tensor


from torch.optim.optimizer import Optimizer


from torch.utils.cpp_extension import BuildExtension


from torch.utils.cpp_extension import CUDAExtension


from torch.cuda import is_available


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000 ** omega
    pos = pos.reshape(-1)
    out = np.einsum('m,d->md', pos, omega)
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    emb = np.concatenate([emb_sin, emb_cos], axis=1)
    return emb


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    emb = np.concatenate([emb_h, emb_w], axis=1)
    return emb


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=1024, depth=24, num_heads=16, decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16, mlp_ratio=4.0, norm_layer=nn.LayerNorm, norm_pix_loss=False):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)
        self.blocks = nn.ModuleList([Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer) for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)
        self.decoder_blocks = nn.ModuleList([Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer) for i in range(decoder_depth)])
        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size ** 2 * in_chans, bias=True)
        self.norm_pix_loss = norm_pix_loss
        self.initialize_weights()

    def initialize_weights(self):
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches ** 0.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches ** 0.5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        torch.nn.init.normal_(self.cls_token, std=0.02)
        torch.nn.init.normal_(self.mask_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0
        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape
        len_keep = int(L * (1 - mask_ratio))
        noise = torch.rand(N, L, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio):
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:, :]
        x, mask, ids_restore = self.random_masking(x, mask_ratio)
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        x = self.decoder_embed(x)
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
        x = torch.cat([x[:, :1, :], x_], dim=1)
        x = x + self.decoder_pos_embed
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        x = self.decoder_pred(x)
        x = x[:, 1:, :]
        return x

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1e-06) ** 0.5
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)
        loss = (loss * mask).sum() / mask.sum()
        return loss

    def forward(self, imgs, mask_ratio=0.75):
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask


class PositionalEmbedding(nn.Module):

    def __init__(self, demb):
        super(PositionalEmbedding, self).__init__()
        self.demb = demb
        inv_freq = 1 / 10000 ** (torch.arange(0.0, demb, 2.0) / demb)
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, pos_seq, bsz=None):
        sinusoid_inp = torch.ger(pos_seq, self.inv_freq)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)
        if bsz is not None:
            return pos_emb[:, None, :].expand(-1, bsz, -1)
        else:
            return pos_emb[:, None, :]


class PositionwiseFF(nn.Module):

    def __init__(self, d_model, d_inner, dropout, pre_lnorm=False):
        super(PositionwiseFF, self).__init__()
        self.d_model = d_model
        self.d_inner = d_inner
        self.dropout = dropout
        self.CoreNet = nn.Sequential(nn.Linear(d_model, d_inner), nn.ReLU(inplace=True), nn.Dropout(dropout), nn.Linear(d_inner, d_model), nn.Dropout(dropout))
        self.layer_norm = nn.LayerNorm(d_model)
        self.pre_lnorm = pre_lnorm

    def forward(self, inp):
        if self.pre_lnorm:
            core_out = self.CoreNet(self.layer_norm(inp))
            output = core_out + inp
        else:
            core_out = self.CoreNet(inp)
            output = self.layer_norm(inp + core_out)
        return output


class MultiHeadAttn(nn.Module):

    def __init__(self, n_head, d_model, d_head, dropout, dropatt=0, pre_lnorm=False):
        super(MultiHeadAttn, self).__init__()
        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.dropout = dropout
        self.q_net = nn.Linear(d_model, n_head * d_head, bias=False)
        self.kv_net = nn.Linear(d_model, 2 * n_head * d_head, bias=False)
        self.drop = nn.Dropout(dropout)
        self.dropatt = nn.Dropout(dropatt)
        self.o_net = nn.Linear(n_head * d_head, d_model, bias=False)
        self.layer_norm = nn.LayerNorm(d_model)
        self.scale = 1 / d_head ** 0.5
        self.pre_lnorm = pre_lnorm

    def forward(self, h, attn_mask=None, mems=None):
        if mems is not None:
            c = torch.cat([mems, h], 0)
        else:
            c = h
        if self.pre_lnorm:
            c = self.layer_norm(c)
        head_q = self.q_net(h)
        head_k, head_v = torch.chunk(self.kv_net(c), 2, -1)
        head_q = head_q.view(h.size(0), h.size(1), self.n_head, self.d_head)
        head_k = head_k.view(c.size(0), c.size(1), self.n_head, self.d_head)
        head_v = head_v.view(c.size(0), c.size(1), self.n_head, self.d_head)
        attn_score = torch.einsum('ibnd,jbnd->ijbn', (head_q, head_k))
        attn_score.mul_(self.scale)
        if attn_mask is not None and attn_mask.any().item():
            if attn_mask.dim() == 2:
                attn_score.masked_fill_(attn_mask[None, :, :, None], -float('inf'))
            elif attn_mask.dim() == 3:
                attn_score.masked_fill_(attn_mask[:, :, :, None], -float('inf'))
        attn_prob = F.softmax(attn_score, dim=1)
        attn_prob = self.dropatt(attn_prob)
        attn_vec = torch.einsum('ijbn,jbnd->ibnd', (attn_prob, head_v))
        attn_vec = attn_vec.contiguous().view(attn_vec.size(0), attn_vec.size(1), self.n_head * self.d_head)
        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)
        if self.pre_lnorm:
            output = h + attn_out
        else:
            output = self.layer_norm(h + attn_out)
        return output


class RelMultiHeadAttn(nn.Module):

    def __init__(self, n_head, d_model, d_head, dropout, dropatt=0, tgt_len=None, ext_len=None, mem_len=None, pre_lnorm=False):
        super(RelMultiHeadAttn, self).__init__()
        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.dropout = dropout
        self.qkv_net = nn.Linear(d_model, 3 * n_head * d_head, bias=False)
        self.drop = nn.Dropout(dropout)
        self.dropatt = nn.Dropout(dropatt)
        self.o_net = nn.Linear(n_head * d_head, d_model, bias=False)
        self.layer_norm = nn.LayerNorm(d_model)
        self.scale = 1 / d_head ** 0.5
        self.pre_lnorm = pre_lnorm

    def _parallelogram_mask(self, h, w, left=False):
        mask = torch.ones((h, w)).byte()
        m = min(h, w)
        mask[:m, :m] = torch.triu(mask[:m, :m])
        mask[-m:, -m:] = torch.tril(mask[-m:, -m:])
        if left:
            return mask
        else:
            return mask.flip(0)

    def _shift(self, x, qlen, klen, mask, left=False):
        if qlen > 1:
            zero_pad = torch.zeros((x.size(0), qlen - 1, x.size(2), x.size(3)), device=x.device, dtype=x.dtype)
        else:
            zero_pad = torch.zeros(0, device=x.device, dtype=x.dtype)
        if left:
            mask = mask.flip(1)
            x_padded = torch.cat([zero_pad, x], dim=1).expand(qlen, -1, -1, -1)
        else:
            x_padded = torch.cat([x, zero_pad], dim=1).expand(qlen, -1, -1, -1)
        x = x_padded.masked_select(mask[:, :, None, None]).view(qlen, klen, x.size(2), x.size(3))
        return x

    def _rel_shift(self, x, zero_triu=False):
        zero_pad = torch.zeros((x.size(0), 1, *x.size()[2:]), device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=1)
        x_padded = x_padded.view(x.size(1) + 1, x.size(0), *x.size()[2:])
        x = x_padded[1:].view_as(x)
        if zero_triu:
            ones = torch.ones((x.size(0), x.size(1)))
            x = x * torch.tril(ones, x.size(1) - x.size(0))[:, :, None, None]
        return x

    def forward(self, w, r, attn_mask=None, mems=None):
        raise NotImplementedError


class RelPartialLearnableMultiHeadAttn(RelMultiHeadAttn):

    def __init__(self, *args, **kwargs):
        super(RelPartialLearnableMultiHeadAttn, self).__init__(*args, **kwargs)
        self.r_net = nn.Linear(self.d_model, self.n_head * self.d_head, bias=False)

    def forward(self, w, r, r_w_bias, r_r_bias, attn_mask=None, mems=None):
        qlen, rlen, bsz = w.size(0), r.size(0), w.size(1)
        if mems is not None:
            cat = torch.cat([mems, w], 0)
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(cat))
            else:
                w_heads = self.qkv_net(cat)
            r_head_k = self.r_net(r)
            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)
            w_head_q = w_head_q[-qlen:]
        else:
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(w))
            else:
                w_heads = self.qkv_net(w)
            r_head_k = self.r_net(r)
            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)
        klen = w_head_k.size(0)
        w_head_q = w_head_q.view(qlen, bsz, self.n_head, self.d_head)
        w_head_k = w_head_k.view(klen, bsz, self.n_head, self.d_head)
        w_head_v = w_head_v.view(klen, bsz, self.n_head, self.d_head)
        r_head_k = r_head_k.view(rlen, self.n_head, self.d_head)
        rw_head_q = w_head_q + r_w_bias
        AC = torch.einsum('ibnd,jbnd->ijbn', (rw_head_q, w_head_k))
        rr_head_q = w_head_q + r_r_bias
        BD = torch.einsum('ibnd,jnd->ijbn', (rr_head_q, r_head_k))
        BD = self._rel_shift(BD)
        attn_score = AC + BD
        attn_score.mul_(self.scale)
        if attn_mask is not None and attn_mask.any().item():
            if attn_mask.dim() == 2:
                attn_score = attn_score.float().masked_fill(attn_mask[None, :, :, None], -float('inf')).type_as(attn_score)
            elif attn_mask.dim() == 3:
                attn_score = attn_score.float().masked_fill(attn_mask[:, :, :, None], -float('inf')).type_as(attn_score)
        attn_prob = F.softmax(attn_score, dim=1)
        attn_prob = self.dropatt(attn_prob)
        attn_vec = torch.einsum('ijbn,jbnd->ibnd', (attn_prob, w_head_v))
        attn_vec = attn_vec.contiguous().view(attn_vec.size(0), attn_vec.size(1), self.n_head * self.d_head)
        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)
        if self.pre_lnorm:
            output = w + attn_out
        else:
            output = self.layer_norm(w + attn_out)
        return output


class RelLearnableMultiHeadAttn(RelMultiHeadAttn):

    def __init__(self, *args, **kwargs):
        super(RelLearnableMultiHeadAttn, self).__init__(*args, **kwargs)

    def forward(self, w, r_emb, r_w_bias, r_bias, attn_mask=None, mems=None):
        qlen, bsz = w.size(0), w.size(1)
        if mems is not None:
            cat = torch.cat([mems, w], 0)
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(cat))
            else:
                w_heads = self.qkv_net(cat)
            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)
            w_head_q = w_head_q[-qlen:]
        else:
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(w))
            else:
                w_heads = self.qkv_net(w)
            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)
        klen = w_head_k.size(0)
        w_head_q = w_head_q.view(qlen, bsz, self.n_head, self.d_head)
        w_head_k = w_head_k.view(klen, bsz, self.n_head, self.d_head)
        w_head_v = w_head_v.view(klen, bsz, self.n_head, self.d_head)
        if klen > r_emb.size(0):
            r_emb_pad = r_emb[0:1].expand(klen - r_emb.size(0), -1, -1)
            r_emb = torch.cat([r_emb_pad, r_emb], 0)
            r_bias_pad = r_bias[0:1].expand(klen - r_bias.size(0), -1)
            r_bias = torch.cat([r_bias_pad, r_bias], 0)
        else:
            r_emb = r_emb[-klen:]
            r_bias = r_bias[-klen:]
        rw_head_q = w_head_q + r_w_bias[None]
        AC = torch.einsum('ibnd,jbnd->ijbn', (rw_head_q, w_head_k))
        B_ = torch.einsum('ibnd,jnd->ijbn', (w_head_q, r_emb))
        D_ = r_bias[None, :, None]
        BD = self._rel_shift(B_ + D_)
        attn_score = AC + BD
        attn_score.mul_(self.scale)
        if attn_mask is not None and attn_mask.any().item():
            if attn_mask.dim() == 2:
                attn_score.masked_fill_(attn_mask[None, :, :, None], -float('inf'))
            elif attn_mask.dim() == 3:
                attn_score.masked_fill_(attn_mask[:, :, :, None], -float('inf'))
        attn_prob = F.softmax(attn_score, dim=1)
        attn_prob = self.dropatt(attn_prob)
        attn_vec = torch.einsum('ijbn,jbnd->ibnd', (attn_prob, w_head_v))
        attn_vec = attn_vec.contiguous().view(attn_vec.size(0), attn_vec.size(1), self.n_head * self.d_head)
        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)
        if self.pre_lnorm:
            output = w + attn_out
        else:
            output = self.layer_norm(w + attn_out)
        return output


class DecoderLayer(nn.Module):

    def __init__(self, n_head, d_model, d_head, d_inner, dropout, **kwargs):
        super(DecoderLayer, self).__init__()
        self.dec_attn = MultiHeadAttn(n_head, d_model, d_head, dropout, **kwargs)
        self.pos_ff = PositionwiseFF(d_model, d_inner, dropout, pre_lnorm=kwargs.get('pre_lnorm'))

    def forward(self, dec_inp, dec_attn_mask=None, mems=None):
        output = self.dec_attn(dec_inp, attn_mask=dec_attn_mask, mems=mems)
        output = self.pos_ff(output)
        return output


class RelLearnableDecoderLayer(nn.Module):

    def __init__(self, n_head, d_model, d_head, d_inner, dropout, **kwargs):
        super(RelLearnableDecoderLayer, self).__init__()
        self.dec_attn = RelLearnableMultiHeadAttn(n_head, d_model, d_head, dropout, **kwargs)
        self.pos_ff = PositionwiseFF(d_model, d_inner, dropout, pre_lnorm=kwargs.get('pre_lnorm'))

    def forward(self, dec_inp, r_emb, r_w_bias, r_bias, dec_attn_mask=None, mems=None):
        output = self.dec_attn(dec_inp, r_emb, r_w_bias, r_bias, attn_mask=dec_attn_mask, mems=mems)
        output = self.pos_ff(output)
        return output


class RelPartialLearnableDecoderLayer(nn.Module):

    def __init__(self, n_head, d_model, d_head, d_inner, dropout, **kwargs):
        super(RelPartialLearnableDecoderLayer, self).__init__()
        self.dec_attn = RelPartialLearnableMultiHeadAttn(n_head, d_model, d_head, dropout, **kwargs)
        self.pos_ff = PositionwiseFF(d_model, d_inner, dropout, pre_lnorm=kwargs.get('pre_lnorm'))

    def forward(self, dec_inp, r, r_w_bias, r_r_bias, dec_attn_mask=None, mems=None):
        output = self.dec_attn(dec_inp, r, r_w_bias, r_r_bias, attn_mask=dec_attn_mask, mems=mems)
        output = self.pos_ff(output)
        return output


class AdaptiveEmbedding(nn.Module):

    def __init__(self, n_token, d_embed, d_proj, cutoffs, div_val=1, sample_softmax=False):
        super(AdaptiveEmbedding, self).__init__()
        self.n_token = n_token
        self.d_embed = d_embed
        self.cutoffs = cutoffs + [n_token]
        self.div_val = div_val
        self.d_proj = d_proj
        self.emb_scale = d_proj ** 0.5
        self.cutoff_ends = [0] + self.cutoffs
        self.emb_layers = nn.ModuleList()
        self.emb_projs = nn.ParameterList()
        if div_val == 1:
            self.emb_layers.append(nn.Embedding(n_token, d_embed, sparse=sample_softmax > 0))
            if d_proj != d_embed:
                self.emb_projs.append(nn.Parameter(torch.Tensor(d_proj, d_embed)))
        else:
            for i in range(len(self.cutoffs)):
                l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i + 1]
                d_emb_i = d_embed // div_val ** i
                self.emb_layers.append(nn.Embedding(r_idx - l_idx, d_emb_i))
                self.emb_projs.append(nn.Parameter(torch.Tensor(d_proj, d_emb_i)))

    def forward(self, inp):
        if self.div_val == 1:
            embed = self.emb_layers[0](inp)
            if self.d_proj != self.d_embed:
                embed = F.linear(embed, self.emb_projs[0])
        else:
            param = next(self.parameters())
            inp_flat = inp.view(-1)
            emb_flat = torch.zeros([inp_flat.size(0), self.d_proj], dtype=param.dtype, device=param.device)
            for i in range(len(self.cutoffs)):
                l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i + 1]
                mask_i = (inp_flat >= l_idx) & (inp_flat < r_idx)
                indices_i = mask_i.nonzero().squeeze()
                if indices_i.numel() == 0:
                    continue
                inp_i = inp_flat.index_select(0, indices_i) - l_idx
                emb_i = self.emb_layers[i](inp_i)
                emb_i = F.linear(emb_i, self.emb_projs[i])
                emb_flat.index_copy_(0, indices_i, emb_i)
            embed = emb_flat.view(*inp.size(), self.d_proj)
        embed.mul_(self.emb_scale)
        return embed


class LogUniformSampler(object):

    def __init__(self, range_max, n_sample):
        """
        Reference : https://github.com/tensorflow/tensorflow/blob/r1.10/tensorflow/python/ops/candidate_sampling_ops.py
            `P(class) = (log(class + 2) - log(class + 1)) / log(range_max + 1)`

        expected count can be approximated by 1 - (1 - p)^n
        and we use a numerically stable version -expm1(num_tries * log1p(-p))

        Our implementation fixes num_tries at 2 * n_sample, and the actual #samples will vary from run to run
        """
        with torch.no_grad():
            self.range_max = range_max
            log_indices = torch.arange(1.0, range_max + 2.0, 1.0).log_()
            self.dist = (log_indices[1:] - log_indices[:-1]) / log_indices[-1]
            self.log_q = (-(-self.dist.double().log1p_() * 2 * n_sample).expm1_()).log_().float()
        self.n_sample = n_sample

    def sample(self, labels):
        """
            labels: [b1, b2]
        Return
            true_log_probs: [b1, b2]
            samp_log_probs: [n_sample]
            neg_samples: [n_sample]
        """
        n_sample = self.n_sample
        n_tries = 2 * n_sample
        with torch.no_grad():
            neg_samples = torch.multinomial(self.dist, n_tries, replacement=True).unique()
            device = labels.device
            neg_samples = neg_samples
            true_log_probs = self.log_q[labels]
            samp_log_probs = self.log_q[neg_samples]
            return true_log_probs, samp_log_probs, neg_samples


class ProjectedAdaptiveLogSoftmax(nn.Module):

    def __init__(self, n_token, d_embed, d_proj, cutoffs, div_val=1, keep_order=False):
        super(ProjectedAdaptiveLogSoftmax, self).__init__()
        self.n_token = n_token
        self.d_embed = d_embed
        self.d_proj = d_proj
        self.cutoffs = cutoffs + [n_token]
        self.cutoff_ends = [0] + self.cutoffs
        self.div_val = div_val
        self.shortlist_size = self.cutoffs[0]
        self.n_clusters = len(self.cutoffs) - 1
        self.head_size = self.shortlist_size + self.n_clusters
        if self.n_clusters > 0:
            self.cluster_weight = nn.Parameter(torch.zeros(self.n_clusters, self.d_embed))
            self.cluster_bias = nn.Parameter(torch.zeros(self.n_clusters))
        self.out_layers = nn.ModuleList()
        self.out_projs = nn.ParameterList()
        if div_val == 1:
            for i in range(len(self.cutoffs)):
                if d_proj != d_embed:
                    self.out_projs.append(nn.Parameter(torch.Tensor(d_proj, d_embed)))
                else:
                    self.out_projs.append(None)
            self.out_layers.append(nn.Linear(d_embed, n_token))
        else:
            for i in range(len(self.cutoffs)):
                l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i + 1]
                d_emb_i = d_embed // div_val ** i
                self.out_projs.append(nn.Parameter(torch.Tensor(d_proj, d_emb_i)))
                self.out_layers.append(nn.Linear(d_emb_i, r_idx - l_idx))
        self.keep_order = keep_order

    def _compute_logit(self, hidden, weight, bias, proj):
        if proj is None:
            logit = F.linear(hidden, weight, bias=bias)
        else:
            proj_hid = F.linear(hidden, proj.t().contiguous())
            logit = F.linear(proj_hid, weight, bias=bias)
        return logit

    def forward(self, hidden, target, keep_order=False):
        """
            hidden :: [len*bsz x d_proj]
            target :: [len*bsz]
        """
        if hidden.size(0) != target.size(0):
            raise RuntimeError('Input and target should have the same size in the batch dimension.')
        if self.n_clusters == 0:
            logit = self._compute_logit(hidden, self.out_layers[0].weight, self.out_layers[0].bias, self.out_projs[0])
            nll = -F.log_softmax(logit, dim=-1).gather(1, target.unsqueeze(1)).squeeze(1)
        else:
            weights, biases = [], []
            for i in range(len(self.cutoffs)):
                if self.div_val == 1:
                    l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i + 1]
                    weight_i = self.out_layers[0].weight[l_idx:r_idx]
                    bias_i = self.out_layers[0].bias[l_idx:r_idx]
                else:
                    weight_i = self.out_layers[i].weight
                    bias_i = self.out_layers[i].bias
                if i == 0:
                    weight_i = torch.cat([weight_i, self.cluster_weight], dim=0)
                    bias_i = torch.cat([bias_i, self.cluster_bias], dim=0)
                weights.append(weight_i)
                biases.append(bias_i)
            head_weight, head_bias, head_proj = weights[0], biases[0], self.out_projs[0]
            head_logit = self._compute_logit(hidden, head_weight, head_bias, head_proj)
            head_logprob = F.log_softmax(head_logit, dim=1)
            nll = torch.zeros_like(target, dtype=hidden.dtype, device=hidden.device)
            offset = 0
            cutoff_values = [0] + self.cutoffs
            for i in range(len(cutoff_values) - 1):
                l_idx, r_idx = cutoff_values[i], cutoff_values[i + 1]
                mask_i = (target >= l_idx) & (target < r_idx)
                indices_i = mask_i.nonzero().squeeze()
                if indices_i.numel() == 0:
                    continue
                target_i = target.index_select(0, indices_i) - l_idx
                head_logprob_i = head_logprob.index_select(0, indices_i)
                if i == 0:
                    logprob_i = head_logprob_i.gather(1, target_i[:, None]).squeeze(1)
                else:
                    weight_i, bias_i, proj_i = weights[i], biases[i], self.out_projs[i]
                    hidden_i = hidden.index_select(0, indices_i)
                    tail_logit_i = self._compute_logit(hidden_i, weight_i, bias_i, proj_i)
                    tail_logprob_i = F.log_softmax(tail_logit_i, dim=1)
                    logprob_i = head_logprob_i[:, -i] + tail_logprob_i.gather(1, target_i[:, None]).squeeze(1)
                if hasattr(self, 'keep_order') and self.keep_order or keep_order:
                    nll.index_copy_(0, indices_i, -logprob_i)
                else:
                    nll[offset:offset + logprob_i.size(0)].copy_(-logprob_i)
                offset += logprob_i.size(0)
        return nll


def sample_logits(embedding, bias, labels, inputs, sampler):
    """
        embedding: an nn.Embedding layer
        bias: [n_vocab]
        labels: [b1, b2]
        inputs: [b1, b2, n_emb]
        sampler: you may use a LogUniformSampler
    Return
        logits: [b1, b2, 1 + n_sample]
    """
    true_log_probs, samp_log_probs, neg_samples = sampler.sample(labels)
    n_sample = neg_samples.size(0)
    b1, b2 = labels.size(0), labels.size(1)
    all_ids = torch.cat([labels.view(-1), neg_samples])
    all_w = embedding(all_ids)
    true_w = all_w[:-n_sample].view(b1, b2, -1)
    sample_w = all_w[-n_sample:].view(n_sample, -1)
    all_b = bias[all_ids]
    true_b = all_b[:-n_sample].view(b1, b2)
    sample_b = all_b[-n_sample:]
    hit = (labels[:, :, None] == neg_samples).detach()
    true_logits = torch.einsum('ijk,ijk->ij', [true_w, inputs]) + true_b - true_log_probs
    sample_logits = torch.einsum('lk,ijk->ijl', [sample_w, inputs]) + sample_b - samp_log_probs
    sample_logits.masked_fill_(hit, -1e+30)
    logits = torch.cat([true_logits[:, :, None], sample_logits], -1)
    return logits


class MemTransformerLM(nn.Module):

    def __init__(self, n_token, n_layer, n_head, d_model, d_head, d_inner, dropout, dropatt, tie_weight=True, d_embed=None, div_val=1, tie_projs=[False], pre_lnorm=False, tgt_len=None, ext_len=None, mem_len=None, cutoffs=[], adapt_inp=False, same_length=False, attn_type=0, clamp_len=-1, sample_softmax=-1):
        super(MemTransformerLM, self).__init__()
        self.n_token = n_token
        d_embed = d_model if d_embed is None else d_embed
        self.d_embed = d_embed
        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_head
        self.word_emb = AdaptiveEmbedding(n_token, d_embed, d_model, cutoffs, div_val=div_val)
        self.drop = nn.Dropout(dropout)
        self.n_layer = n_layer
        self.tgt_len = tgt_len
        self.mem_len = mem_len
        self.ext_len = ext_len
        self.max_klen = tgt_len + ext_len + mem_len
        self.attn_type = attn_type
        self.layers = nn.ModuleList()
        if attn_type == 0:
            for i in range(n_layer):
                self.layers.append(RelPartialLearnableDecoderLayer(n_head, d_model, d_head, d_inner, dropout, tgt_len=tgt_len, ext_len=ext_len, mem_len=mem_len, dropatt=dropatt, pre_lnorm=pre_lnorm))
        elif attn_type == 1:
            for i in range(n_layer):
                self.layers.append(RelLearnableDecoderLayer(n_head, d_model, d_head, d_inner, dropout, tgt_len=tgt_len, ext_len=ext_len, mem_len=mem_len, dropatt=dropatt, pre_lnorm=pre_lnorm))
        elif attn_type in [2, 3]:
            for i in range(n_layer):
                self.layers.append(DecoderLayer(n_head, d_model, d_head, d_inner, dropout, dropatt=dropatt, pre_lnorm=pre_lnorm))
        self.sample_softmax = sample_softmax
        if sample_softmax > 0:
            self.out_layer = nn.Linear(d_model, n_token)
            if tie_weight:
                self.out_layer.weight = self.word_emb.weight
            self.tie_weight = tie_weight
            self.sampler = LogUniformSampler(n_token, sample_softmax)
        else:
            self.crit = ProjectedAdaptiveLogSoftmax(n_token, d_embed, d_model, cutoffs, div_val=div_val)
            if tie_weight:
                for i in range(len(self.crit.out_layers)):
                    self.crit.out_layers[i].weight = self.word_emb.emb_layers[i].weight
            if tie_projs:
                for i, tie_proj in enumerate(tie_projs):
                    if tie_proj and div_val == 1 and d_model != d_embed:
                        self.crit.out_projs[i] = self.word_emb.emb_projs[0]
                    elif tie_proj and div_val != 1:
                        self.crit.out_projs[i] = self.word_emb.emb_projs[i]
        self.same_length = same_length
        self.clamp_len = clamp_len
        self._create_params()

    def backward_compatible(self):
        self.sample_softmax = -1

    def _create_params(self):
        if self.attn_type == 0:
            self.pos_emb = PositionalEmbedding(self.d_model)
            self.r_w_bias = nn.Parameter(torch.Tensor(self.n_head, self.d_head))
            self.r_r_bias = nn.Parameter(torch.Tensor(self.n_head, self.d_head))
        elif self.attn_type == 1:
            self.r_emb = nn.Parameter(torch.Tensor(self.n_layer, self.max_klen, self.n_head, self.d_head))
            self.r_w_bias = nn.Parameter(torch.Tensor(self.n_layer, self.n_head, self.d_head))
            self.r_bias = nn.Parameter(torch.Tensor(self.n_layer, self.max_klen, self.n_head))
        elif self.attn_type == 2:
            self.pos_emb = PositionalEmbedding(self.d_model)
        elif self.attn_type == 3:
            self.r_emb = nn.Parameter(torch.Tensor(self.n_layer, self.max_klen, self.n_head, self.d_head))

    def reset_length(self, tgt_len, ext_len, mem_len):
        self.tgt_len = tgt_len
        self.mem_len = mem_len
        self.ext_len = ext_len

    def init_mems(self):
        if self.mem_len > 0:
            mems = []
            param = next(self.parameters())
            for i in range(self.n_layer + 1):
                empty = torch.empty(0, dtype=param.dtype, device=param.device)
                mems.append(empty)
            return mems
        else:
            return None

    def _update_mems(self, hids, mems, qlen, mlen):
        if mems is None:
            return None
        assert len(hids) == len(mems), 'len(hids) != len(mems)'
        with torch.no_grad():
            new_mems = []
            end_idx = mlen + max(0, qlen - 0 - self.ext_len)
            beg_idx = max(0, end_idx - self.mem_len)
            for i in range(len(hids)):
                cat = torch.cat([mems[i], hids[i]], dim=0)
                new_mems.append(cat[beg_idx:end_idx].detach())
        return new_mems

    def _forward(self, dec_inp, mems=None):
        qlen, bsz = dec_inp.size()
        word_emb = self.word_emb(dec_inp)
        mlen = mems[0].size(0) if mems is not None else 0
        klen = mlen + qlen
        if self.same_length:
            all_ones = word_emb.new_ones(qlen, klen)
            mask_len = klen - self.mem_len
            if mask_len > 0:
                mask_shift_len = qlen - mask_len
            else:
                mask_shift_len = qlen
            dec_attn_mask = (torch.triu(all_ones, 1 + mlen) + torch.tril(all_ones, -mask_shift_len)).byte()[:, :, None]
        else:
            dec_attn_mask = torch.triu(word_emb.new_ones(qlen, klen), diagonal=1 + mlen).byte()[:, :, None]
        hids = []
        if self.attn_type == 0:
            pos_seq = torch.arange(klen - 1, -1, -1.0, device=word_emb.device, dtype=word_emb.dtype)
            if self.clamp_len > 0:
                pos_seq.clamp_(max=self.clamp_len)
            pos_emb = self.pos_emb(pos_seq)
            core_out = self.drop(word_emb)
            pos_emb = self.drop(pos_emb)
            hids.append(core_out)
            for i, layer in enumerate(self.layers):
                mems_i = None if mems is None else mems[i]
                core_out = layer(core_out, pos_emb, self.r_w_bias, self.r_r_bias, dec_attn_mask=dec_attn_mask, mems=mems_i)
                hids.append(core_out)
        elif self.attn_type == 1:
            core_out = self.drop(word_emb)
            hids.append(core_out)
            for i, layer in enumerate(self.layers):
                if self.clamp_len > 0:
                    r_emb = self.r_emb[i][-self.clamp_len:]
                    r_bias = self.r_bias[i][-self.clamp_len:]
                else:
                    r_emb, r_bias = self.r_emb[i], self.r_bias[i]
                mems_i = None if mems is None else mems[i]
                core_out = layer(core_out, r_emb, self.r_w_bias[i], r_bias, dec_attn_mask=dec_attn_mask, mems=mems_i)
                hids.append(core_out)
        elif self.attn_type == 2:
            pos_seq = torch.arange(klen - 1, -1, -1.0, device=word_emb.device, dtype=word_emb.dtype)
            if self.clamp_len > 0:
                pos_seq.clamp_(max=self.clamp_len)
            pos_emb = self.pos_emb(pos_seq)
            core_out = self.drop(word_emb + pos_emb[-qlen:])
            hids.append(core_out)
            for i, layer in enumerate(self.layers):
                mems_i = None if mems is None else mems[i]
                if mems_i is not None and i == 0:
                    mems_i += pos_emb[:mlen]
                core_out = layer(core_out, dec_attn_mask=dec_attn_mask, mems=mems_i)
                hids.append(core_out)
        elif self.attn_type == 3:
            core_out = self.drop(word_emb)
            hids.append(core_out)
            for i, layer in enumerate(self.layers):
                mems_i = None if mems is None else mems[i]
                if mems_i is not None and mlen > 0:
                    cur_emb = self.r_emb[i][:-qlen]
                    cur_size = cur_emb.size(0)
                    if cur_size < mlen:
                        cur_emb_pad = cur_emb[0:1].expand(mlen - cur_size, -1, -1)
                        cur_emb = torch.cat([cur_emb_pad, cur_emb], 0)
                    else:
                        cur_emb = cur_emb[-mlen:]
                    mems_i += cur_emb.view(mlen, 1, -1)
                core_out += self.r_emb[i][-qlen:].view(qlen, 1, -1)
                core_out = layer(core_out, dec_attn_mask=dec_attn_mask, mems=mems_i)
                hids.append(core_out)
        core_out = self.drop(core_out)
        new_mems = self._update_mems(hids, mems, mlen, qlen)
        return core_out, new_mems

    def forward(self, data, target, *mems):
        if not mems:
            mems = self.init_mems()
        tgt_len = target.size(0)
        hidden, new_mems = self._forward(data, mems=mems)
        pred_hid = hidden[-tgt_len:]
        if self.sample_softmax > 0 and self.training:
            assert self.tie_weight
            logit = sample_logits(self.word_emb, self.out_layer.bias, target, pred_hid, self.sampler)
            loss = -F.log_softmax(logit, -1)[:, :, 0]
        else:
            loss = self.crit(pred_hid.view(-1, pred_hid.size(-1)), target.view(-1))
            loss = loss.view(tgt_len, -1)
        if new_mems is None:
            return [loss]
        else:
            return [loss] + new_mems


class AdaptiveLogSoftmax(nn.Module):

    def __init__(self, in_features, n_classes, cutoffs, keep_order=False):
        super(AdaptiveLogSoftmax, self).__init__()
        cutoffs = list(cutoffs)
        if cutoffs != sorted(cutoffs) or min(cutoffs) <= 0 or max(cutoffs) >= n_classes - 1 or len(set(cutoffs)) != len(cutoffs) or any([(int(c) != c) for c in cutoffs]):
            raise ValueError('cutoffs should be a sequence of unique, positive integers sorted in an increasing order, where each value is between 1 and n_classes-1')
        self.in_features = in_features
        self.n_classes = n_classes
        self.cutoffs = cutoffs + [n_classes]
        self.shortlist_size = self.cutoffs[0]
        self.n_clusters = len(self.cutoffs) - 1
        self.head_size = self.shortlist_size + self.n_clusters
        self.cluster_weight = nn.Parameter(torch.zeros(self.n_clusters, self.in_features))
        self.cluster_bias = nn.Parameter(torch.zeros(self.n_clusters))
        self.keep_order = keep_order

    def forward(self, hidden, target, weight, bias, keep_order=False):
        if hidden.size(0) != target.size(0):
            raise RuntimeError('Input and target should have the same size in the batch dimension.')
        head_weight = torch.cat([weight[:self.shortlist_size], self.cluster_weight], dim=0)
        head_bias = torch.cat([bias[:self.shortlist_size], self.cluster_bias], dim=0)
        head_logit = F.linear(hidden, head_weight, bias=head_bias)
        head_logprob = F.log_softmax(head_logit, dim=1)
        nll = torch.zeros_like(target, dtype=hidden.dtype, device=hidden.device)
        offset = 0
        cutoff_values = [0] + self.cutoffs
        for i in range(len(cutoff_values) - 1):
            l_idx, h_idx = cutoff_values[i], cutoff_values[i + 1]
            mask_i = (target >= l_idx) & (target < h_idx)
            indices_i = mask_i.nonzero().squeeze()
            if indices_i.numel() == 0:
                continue
            target_i = target.index_select(0, indices_i) - l_idx
            head_logprob_i = head_logprob.index_select(0, indices_i)
            if i == 0:
                logprob_i = head_logprob_i.gather(1, target_i[:, None]).squeeze(1)
            else:
                weight_i = weight[l_idx:h_idx]
                bias_i = bias[l_idx:h_idx]
                hidden_i = hidden.index_select(0, indices_i)
                tail_logit_i = F.linear(hidden_i, weight_i, bias=bias_i)
                tail_logprob_i = F.log_softmax(tail_logit_i, dim=1)
                logprob_i = head_logprob_i[:, -i] + tail_logprob_i.gather(1, target_i[:, None]).squeeze(1)
            if hasattr(self, 'keep_order') and self.keep_order or keep_order:
                nll.index_copy_(0, indices_i, -logprob_i)
            else:
                nll[offset:offset + logprob_i.size(0)].copy_(-logprob_i)
            offset += logprob_i.size(0)
        return nll


def scatter(inputs, target_gpus, chunk_sizes, dim=0):
    """
    Slices tensors into approximately equal chunks and
    distributes them across given GPUs. Duplicates
    references to objects that are not tensors.
    """

    def scatter_map(obj):
        if isinstance(obj, torch.Tensor):
            try:
                return Scatter.apply(target_gpus, chunk_sizes, dim, obj)
            except:
                None
                None
                None
                quit()
        if isinstance(obj, tuple) and len(obj) > 0:
            return list(zip(*map(scatter_map, obj)))
        if isinstance(obj, list) and len(obj) > 0:
            return list(map(list, zip(*map(scatter_map, obj))))
        if isinstance(obj, dict) and len(obj) > 0:
            return list(map(type(obj), zip(*map(scatter_map, obj.items()))))
        return [obj for targets in target_gpus]
    try:
        return scatter_map(inputs)
    finally:
        scatter_map = None


def scatter_kwargs(inputs, kwargs, target_gpus, chunk_sizes, dim=0):
    """Scatter with support for kwargs dictionary"""
    inputs = scatter(inputs, target_gpus, chunk_sizes, dim) if inputs else []
    kwargs = scatter(kwargs, target_gpus, chunk_sizes, dim) if kwargs else []
    if len(inputs) < len(kwargs):
        inputs.extend([() for _ in range(len(kwargs) - len(inputs))])
    elif len(kwargs) < len(inputs):
        kwargs.extend([{} for _ in range(len(inputs) - len(kwargs))])
    inputs = tuple(inputs)
    kwargs = tuple(kwargs)
    return inputs, kwargs


class BalancedDataParallel(DataParallel):

    def __init__(self, gpu0_bsz, *args, **kwargs):
        self.gpu0_bsz = gpu0_bsz
        super().__init__(*args, **kwargs)

    def forward(self, *inputs, **kwargs):
        if not self.device_ids:
            return self.module(*inputs, **kwargs)
        if self.gpu0_bsz == 0:
            device_ids = self.device_ids[1:]
        else:
            device_ids = self.device_ids
        inputs, kwargs = self.scatter(inputs, kwargs, device_ids)
        if len(self.device_ids) == 1:
            return self.module(*inputs[0], **kwargs[0])
        replicas = self.replicate(self.module, self.device_ids)
        if self.gpu0_bsz == 0:
            replicas = replicas[1:]
        outputs = self.parallel_apply(replicas, device_ids, inputs, kwargs)
        return self.gather(outputs, self.output_device)

    def parallel_apply(self, replicas, device_ids, inputs, kwargs):
        return parallel_apply(replicas, inputs, kwargs, device_ids)

    def scatter(self, inputs, kwargs, device_ids):
        bsz = inputs[0].size(self.dim)
        num_dev = len(self.device_ids)
        gpu0_bsz = self.gpu0_bsz
        bsz_unit = (bsz - gpu0_bsz) // (num_dev - 1)
        if gpu0_bsz < bsz_unit:
            chunk_sizes = [gpu0_bsz] + [bsz_unit] * (num_dev - 1)
            delta = bsz - sum(chunk_sizes)
            for i in range(delta):
                chunk_sizes[i + 1] += 1
            if gpu0_bsz == 0:
                chunk_sizes = chunk_sizes[1:]
        else:
            return super().scatter(inputs, kwargs, device_ids)
        return scatter_kwargs(inputs, kwargs, device_ids, chunk_sizes, dim=self.dim)


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (BalancedDataParallel,
     lambda: ([], {'gpu0_bsz': False, 'module': torch.nn.ReLU()}),
     lambda: ([], {'input': torch.rand([4, 4])})),
    (DecoderLayer,
     lambda: ([], {'n_head': 4, 'd_model': 4, 'd_head': 4, 'd_inner': 4, 'dropout': 0.5}),
     lambda: ([torch.rand([4, 4, 4])], {})),
    (MultiHeadAttn,
     lambda: ([], {'n_head': 4, 'd_model': 4, 'd_head': 4, 'dropout': 0.5}),
     lambda: ([torch.rand([4, 4, 4])], {})),
    (PositionwiseFF,
     lambda: ([], {'d_model': 4, 'd_inner': 4, 'dropout': 0.5}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
]

