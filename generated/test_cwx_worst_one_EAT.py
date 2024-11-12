
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


from functools import partial


import logging


import random


import time


import numpy as np


import torch.nn.functional as F


from sklearn import metrics as sklearn_metrics


from torch.utils.data import Dataset


from torch.utils.data import DataLoader


import torch.nn as nn


from enum import Enum


from enum import auto


from typing import Any


from typing import Optional


import math


import torch.distributed as dist


from typing import Callable


from collections import namedtuple


from typing import Dict


from typing import Tuple


MaskInfo = namedtuple('MaskInfo', ['x_unmasked', 'mask', 'ids_restore', 'ids_keep'])


MaskSeed = namedtuple('MaskSeed', ['seed', 'update', 'ids'])


def _learned_alibi_bias(alibi_bias, batch_size, time_steps, heads, scale, dtype, device):
    assert alibi_bias.size(1) == heads, alibi_bias.shape
    assert alibi_bias.dtype == dtype, alibi_bias.dtype
    assert alibi_bias.device == device, alibi_bias.device
    if alibi_bias.size(-1) < time_steps:
        psz = math.ceil((time_steps - alibi_bias.size(-1)) / 2)
        alibi_bias = F.pad(alibi_bias, (psz, psz, psz, psz), mode='replicate')
    alibi_bias = alibi_bias.expand(batch_size, -1, -1, -1) * scale
    return alibi_bias[..., :time_steps, :time_steps]


def compute_mask_indices(shape: 'Tuple[int, int]', padding_mask: 'Optional[torch.Tensor]', mask_prob: 'float', mask_length: 'int', mask_type: 'str'='static', mask_other: 'float'=0.0, min_masks: 'int'=0, no_overlap: 'bool'=False, min_space: 'int'=0, require_same_masks: 'bool'=True, mask_dropout: 'float'=0.0, add_masks: 'bool'=False, seed: 'Optional[int]'=None, epoch: 'Optional[int]'=None, indices: 'Optional[torch.Tensor]'=None, idc_select_ver: 'int'=1, num_mask_ver: 'int'=2) ->np.ndarray:
    """
    Computes random mask spans for a given shape

    Args:
        shape: the the shape for which to compute masks.
            should be of size 2 where first element is batch size and 2nd is timesteps
        padding_mask: optional padding mask of the same size as shape, which will prevent masking padded elements
        mask_prob: probability for each token to be chosen as start of the span to be masked. this will be multiplied by
            number of timesteps divided by length of mask span to mask approximately this percentage of all elements.
            however due to overlaps, the actual number will be smaller (unless no_overlap is True)
        mask_type: how to compute mask lengths
            static = fixed size
            uniform = sample from uniform distribution [mask_other, mask_length*2]
            normal = sample from normal distribution with mean mask_length and stdev mask_other. mask is min 1 element
            poisson = sample from possion distribution with lambda = mask length
        min_masks: minimum number of masked spans
        no_overlap: if false, will switch to an alternative recursive algorithm that prevents spans from overlapping
        min_space: only used if no_overlap is True, this is how many elements to keep unmasked between spans
        require_same_masks: if true, will randomly drop out masks until same amount of masks remains in each sample
        mask_dropout: randomly dropout this percentage of masks in each example
    """
    bsz, all_sz = shape
    mask = np.full((bsz, all_sz), False)
    if num_mask_ver == 1:
        all_num_mask = int(mask_prob * all_sz / float(mask_length) + np.random.rand())
        all_num_mask = max(min_masks, all_num_mask)
    mask_idcs = []
    for i in range(bsz):
        if seed is not None and epoch is not None and indices is not None:
            seed_i = int(hash((seed, epoch, indices[i].item())) % 1000000.0)
        else:
            seed_i = None
        rng = np.random.default_rng(seed_i)
        if padding_mask is not None:
            sz = all_sz - padding_mask[i].long().sum().item()
            assert sz >= 0, sz
        else:
            sz = all_sz
        if num_mask_ver == 1:
            if padding_mask is not None:
                num_mask = int(mask_prob * sz / float(mask_length) + np.random.rand())
                num_mask = max(min_masks, num_mask)
            else:
                num_mask = all_num_mask
        elif num_mask_ver == 2:
            num_mask = int(mask_prob * sz / float(mask_length) + rng.random())
            num_mask = max(min_masks, num_mask)
        else:
            raise ValueError()
        if mask_type == 'static':
            lengths = np.full(num_mask, mask_length)
        elif mask_type == 'uniform':
            lengths = rng.randint(mask_other, mask_length * 2 + 1, size=num_mask)
        elif mask_type == 'normal':
            lengths = rng.normal(mask_length, mask_other, size=num_mask)
            lengths = [max(1, int(round(x))) for x in lengths]
        elif mask_type == 'poisson':
            lengths = rng.poisson(mask_length, size=num_mask)
            lengths = [int(round(x)) for x in lengths]
        else:
            raise Exception('unknown mask selection ' + mask_type)
        if sum(lengths) == 0:
            if mask_type == 'static':
                raise ValueError(f'this should never happens')
            else:
                lengths = [min(mask_length, sz - 1)]
        if no_overlap:
            mask_idc = []

            def arrange(s, e, length, keep_length):
                span_start = rng.randint(s, e - length)
                mask_idc.extend(span_start + i for i in range(length))
                new_parts = []
                if span_start - s - min_space >= keep_length:
                    new_parts.append((s, span_start - min_space + 1))
                if e - span_start - length - min_space > keep_length:
                    new_parts.append((span_start + length + min_space, e))
                return new_parts
            parts = [(0, sz)]
            min_length = min(lengths)
            for length in sorted(lengths, reverse=True):
                lens = np.fromiter((e - s if e - s >= length + min_space else 0 for s, e in parts), np.int)
                l_sum = np.sum(lens)
                if l_sum == 0:
                    break
                probs = lens / np.sum(lens)
                c = rng.choice(len(parts), p=probs)
                s, e = parts.pop(c)
                parts.extend(arrange(s, e, length, min_length))
            mask_idc = np.asarray(mask_idc)
        else:
            if idc_select_ver == 1:
                min_len = min(lengths)
                if sz - min_len <= num_mask:
                    min_len = sz - num_mask - 1
                mask_idc = rng.choice(sz - min_len, num_mask, replace=False)
            elif idc_select_ver == 2:
                mask_idc = rng.choice(sz, num_mask, replace=False)
            else:
                raise ValueError()
            mask_idc = np.asarray([(mask_idc[j] + offset) for j in range(len(mask_idc)) for offset in range(lengths[j])])
        mask_idc = np.unique(mask_idc[mask_idc < sz])
        if len(mask_idc) >= sz:
            raise ValueError(f'the entire sequence is masked. sz={sz}; mask_idc[mask_idc]; index={indices[i] if indices is not None else None}')
        mask_idcs.append(mask_idc)
    target_len = None
    if require_same_masks:
        if add_masks:
            target_len = max([len(m) for m in mask_idcs])
        else:
            target_len = min([len(m) for m in mask_idcs])
    for i, mask_idc in enumerate(mask_idcs):
        if target_len is not None and len(mask_idc) > target_len:
            mask_idc = rng.choice(mask_idc, target_len, replace=False)
        mask[i, mask_idc] = True
        if target_len is not None and len(mask_idc) < target_len:
            unmasked = np.flatnonzero(~mask[i])
            to_mask = rng.choice(unmasked, target_len - len(mask_idc), replace=False)
            mask[i, to_mask] = True
        if mask_dropout > 0:
            masked = np.flatnonzero(mask[i])
            num_holes = np.rint(len(masked) * mask_dropout).astype(int)
            to_drop = rng.choice(masked, num_holes, replace=False)
            mask[i, to_drop] = False
    return mask


def gather_unmasked(x: 'torch.Tensor', mask_info: 'MaskInfo') ->torch.Tensor:
    return torch.gather(x, dim=1, index=mask_info.ids_keep)


def gather_unmasked_mask(x: 'torch.Tensor', mask_info: 'MaskInfo') ->torch.Tensor:
    return torch.gather(x, dim=1, index=mask_info.ids_keep[..., 0])


def masked_alibi(alibi_bias, mask_info):
    H = alibi_bias.size(1)
    orig_bias = alibi_bias
    index = mask_info.ids_keep.unsqueeze(1)[..., 0].unsqueeze(-1)
    alibi_bias = torch.gather(orig_bias, dim=-2, index=index.expand(-1, H, -1, mask_info.ids_restore.size(1)))
    alibi_bias = torch.gather(alibi_bias, dim=-1, index=index.transpose(-1, -2).expand(-1, H, alibi_bias.size(-2), -1))
    return alibi_bias


def random_masking(x, mask_ratio, mask_seed: 'Optional[MaskSeed]'):
    N, L, D = x.shape
    len_keep = int(L * (1 - mask_ratio))
    generator = None
    if mask_seed is not None:
        seed = int(hash((mask_seed.seed, mask_seed.update, mask_seed.ids.sum().item())) % 1000000.0)
        generator = torch.Generator(device=x.device)
        generator.manual_seed(seed)
    noise = torch.rand(N, L, generator=generator, device=x.device)
    ids_shuffle = noise.argsort(dim=1)
    ids_restore = ids_shuffle.argsort(dim=1)
    ids_keep = ids_shuffle[:, :len_keep]
    ids_keep = ids_keep.unsqueeze(-1).expand(-1, -1, D)
    x_unmasked = torch.gather(x, dim=1, index=ids_keep)
    mask = torch.ones([N, L], dtype=x.dtype, device=x.device)
    mask[:, :len_keep] = 0
    mask = torch.gather(mask, dim=1, index=ids_restore)
    ids_restore = ids_restore.unsqueeze(-1).expand(-1, -1, D)
    return MaskInfo(x_unmasked=x_unmasked, mask=mask, ids_restore=ids_restore, ids_keep=ids_keep)


class ModalitySpecificEncoder(nn.Module):

    def __init__(self, modality_cfg: 'D2vModalityConfig', embed_dim: 'int', local_encoder: 'nn.Module', project_features: 'nn.Module', fixed_positional_encoder: 'Optional[nn.Module]', relative_positional_encoder: 'Optional[nn.Module]', context_encoder: 'nn.Module', decoder: 'nn.Module', get_alibi_bias: 'Optional[Callable[[int, int, str, str], torch.Tensor]]'):
        super().__init__()
        self.modality_cfg = modality_cfg
        self.local_encoder = local_encoder
        self.project_features = project_features
        self.fixed_positional_encoder = fixed_positional_encoder
        self.relative_positional_encoder = relative_positional_encoder
        self.context_encoder = context_encoder
        self.decoder = decoder
        self.get_alibi_bias = get_alibi_bias if modality_cfg.use_alibi_encoder else None
        self.local_grad_mult = self.modality_cfg.local_grad_mult
        self.extra_tokens = None
        if modality_cfg.num_extra_tokens > 0:
            self.extra_tokens = nn.Parameter(torch.zeros(1, modality_cfg.num_extra_tokens, embed_dim))
            if not modality_cfg.init_extra_token_zero:
                nn.init.normal_(self.extra_tokens)
            elif self.extra_tokens.size(1) > 1:
                nn.init.normal_(self.extra_tokens[:, 1:])
        self.alibi_scale = None
        if self.get_alibi_bias is not None:
            self.alibi_scale = nn.Parameter(torch.full((modality_cfg.prenet_depth + modality_cfg.model_depth if modality_cfg.learned_alibi_scale_per_layer else 1, 1, self.modality_cfg.num_alibi_heads if modality_cfg.learned_alibi_scale_per_head else 1, 1, 1), modality_cfg.alibi_scale, dtype=torch.float), requires_grad=modality_cfg.learned_alibi_scale)
        if modality_cfg.learned_alibi and self.get_alibi_bias is not None:
            assert modality_cfg.alibi_max_pos is not None
            alibi_bias = self.get_alibi_bias(batch_size=1, time_steps=modality_cfg.alibi_max_pos, heads=modality_cfg.num_alibi_heads, scale=1.0, dtype=torch.float, device='cpu')
            self.alibi_bias = nn.Parameter(alibi_bias)
            self.get_alibi_bias = partial(_learned_alibi_bias, alibi_bias=self.alibi_bias)

    def upgrade_state_dict_named(self, state_dict, name):
        k = f'{name}.alibi_scale'
        if k in state_dict and state_dict[k].dim() == 4:
            state_dict[k] = state_dict[k].unsqueeze(0)
        return state_dict

    def convert_padding_mask(self, x, padding_mask):
        return padding_mask

    def decoder_input(self, x, mask_info: 'MaskInfo'):
        inp_drop = self.modality_cfg.decoder.input_dropout
        if inp_drop > 0:
            x = F.dropout(x, inp_drop, training=self.training, inplace=True)
        num_extra = self.modality_cfg.num_extra_tokens
        if mask_info is not None:
            num_masked = mask_info.ids_restore.shape[1] - x.shape[1] + num_extra
            mask_tokens = x.new_empty(x.size(0), num_masked, x.size(-1)).normal_(0, self.modality_cfg.mask_noise_std)
            x_ = torch.cat([x[:, num_extra:], mask_tokens], dim=1)
            x = torch.gather(x_, dim=1, index=mask_info.ids_restore)
            if self.modality_cfg.decoder.add_positions_masked:
                assert self.fixed_positional_encoder is not None
                pos = self.fixed_positional_encoder(x, None)
                x = x + pos * mask_info.mask.unsqueeze(-1)
        else:
            x = x[:, num_extra:]
        if self.modality_cfg.decoder.add_positions_all:
            assert self.fixed_positional_encoder is not None
            x = x + self.fixed_positional_encoder(x, None)
        return x, mask_info

    def local_features(self, features):
        if self.local_grad_mult > 0:
            if self.local_grad_mult == 1.0:
                x = self.local_encoder(features)
            else:
                x = GradMultiply.apply(self.local_encoder(features), self.local_grad_mult)
        else:
            with torch.no_grad():
                x = self.local_encoder(features)
        x = self.project_features(x)
        return x

    def contextualized_features(self, x, padding_mask, mask, remove_masked, clone_batch: 'int'=1, mask_seeds: 'Optional[torch.Tensor]'=None, precomputed_mask=None):
        if padding_mask is not None:
            padding_mask = self.convert_padding_mask(x, padding_mask)
        local_features = x
        if mask and clone_batch == 1:
            local_features = local_features.clone()
        orig_B, orig_T, _ = x.shape
        pre_mask_B = orig_B
        mask_info = None
        x_pos = None
        if self.fixed_positional_encoder is not None:
            x = x + self.fixed_positional_encoder(x, padding_mask)[:, :x.size(1), :]
        if mask:
            if clone_batch > 1:
                x = x.repeat_interleave(clone_batch, 0)
                if mask_seeds is not None:
                    clone_hash = [int(hash((mask_seeds.seed, ind)) % 10000000000.0) for ind in range(clone_batch - 1)]
                    clone_hash = torch.tensor([0] + clone_hash).long().view(1, -1)
                    id = mask_seeds.ids
                    id = id.repeat_interleave(clone_batch, 0)
                    id = id.view(-1, clone_batch) + clone_hash
                    id = id.view(-1)
                    mask_seeds = MaskSeed(seed=mask_seeds.seed, update=mask_seeds.update, ids=id)
                if padding_mask is not None:
                    padding_mask = padding_mask.repeat_interleave(clone_batch, 0)
            x, mask_info = self.compute_mask(x, padding_mask, mask_seed=mask_seeds, apply=self.relative_positional_encoder is not None or not remove_masked, precomputed_mask=precomputed_mask)
        if self.relative_positional_encoder is not None:
            x_pos = self.relative_positional_encoder(x)
        masked_padding_mask = padding_mask
        if mask and remove_masked:
            x = mask_info.x_unmasked
            if x_pos is not None:
                x = x + gather_unmasked(x_pos, mask_info)
            if padding_mask is not None and padding_mask.any():
                masked_padding_mask = gather_unmasked_mask(padding_mask, mask_info)
                if not masked_padding_mask.any():
                    masked_padding_mask = None
            else:
                masked_padding_mask = None
        elif x_pos is not None:
            x = x + x_pos
        alibi_bias = None
        alibi_scale = self.alibi_scale
        if self.get_alibi_bias is not None:
            alibi_bias = self.get_alibi_bias(batch_size=pre_mask_B, time_steps=orig_T, heads=self.modality_cfg.num_alibi_heads, dtype=torch.float32, device=x.device)
            if alibi_scale is not None:
                alibi_scale = alibi_scale.clamp_min(0)
                if alibi_scale.size(0) == 1:
                    alibi_bias = alibi_bias * alibi_scale.squeeze(0).type_as(alibi_bias)
                    alibi_scale = None
            if clone_batch > 1:
                alibi_bias = alibi_bias.repeat_interleave(clone_batch, 0)
            if mask_info is not None and remove_masked:
                alibi_bias = masked_alibi(alibi_bias, mask_info)
        if self.extra_tokens is not None:
            num = self.extra_tokens.size(1)
            x = torch.cat([self.extra_tokens.expand(x.size(0), -1, -1), x], dim=1)
            if masked_padding_mask is not None:
                masked_padding_mask = F.pad(masked_padding_mask, (num, 0))
            if alibi_bias is not None:
                alibi_bias = F.pad(alibi_bias, (num, 0, num, 0))
        x = self.context_encoder(x, masked_padding_mask, alibi_bias, alibi_scale[:self.modality_cfg.prenet_depth] if alibi_scale is not None else None)
        return {'x': x, 'local_features': local_features, 'padding_mask': masked_padding_mask, 'alibi_bias': alibi_bias, 'alibi_scale': alibi_scale[self.modality_cfg.prenet_depth:] if alibi_scale is not None and alibi_scale.size(0) > 1 else alibi_scale, 'encoder_mask': mask_info}

    def forward(self, features, padding_mask, mask: 'bool', remove_masked: 'bool', clone_batch: 'int'=1, mask_seeds: 'Optional[torch.Tensor]'=None, precomputed_mask=None):
        x = self.local_features(features)
        return self.contextualized_features(x, padding_mask, mask, remove_masked, clone_batch, mask_seeds, precomputed_mask)

    def reset_parameters(self):
        pass

    def compute_mask(self, x, padding_mask, mask_seed: 'Optional[MaskSeed]', apply, precomputed_mask):
        if precomputed_mask is not None:
            mask = precomputed_mask
            mask_info = self.make_maskinfo(x, mask)
        else:
            B, T, C = x.shape
            cfg = self.modality_cfg
            mask_prob = cfg.mask_prob
            if cfg.mask_prob_min is not None and cfg.mask_prob_min >= 0 and cfg.mask_prob_min < mask_prob:
                mask_prob = np.random.uniform(cfg.mask_prob_min, mask_prob)
            if mask_prob > 0:
                if cfg.mask_length == 1:
                    mask_info = random_masking(x, mask_prob, mask_seed)
                else:
                    if self.modality_cfg.inverse_mask:
                        mask_prob = 1 - mask_prob
                    mask = compute_mask_indices((B, T), padding_mask, mask_prob, cfg.mask_length, min_masks=1, require_same_masks=True, mask_dropout=cfg.mask_dropout, add_masks=cfg.add_masks, seed=mask_seed.seed if mask_seed is not None else None, epoch=mask_seed.update if mask_seed is not None else None, indices=mask_seed.ids if mask_seed is not None else None)
                    mask = torch.from_numpy(mask)
                    if self.modality_cfg.inverse_mask:
                        mask = 1 - mask
                    mask_info = self.make_maskinfo(x, mask)
            else:
                mask_info = None
        if apply:
            x = self.apply_mask(x, mask_info)
        return x, mask_info

    def make_maskinfo(self, x, mask, shape=None):
        if shape is None:
            B, T, D = x.shape
        else:
            B, T, D = shape
        mask = mask
        ids_shuffle = mask.argsort(dim=1)
        ids_restore = ids_shuffle.argsort(dim=1).unsqueeze(-1).expand(-1, -1, D)
        len_keep = T - mask[0].sum()
        if self.modality_cfg.keep_masked_pct > 0:
            len_keep += round((T - int(len_keep)) * self.modality_cfg.keep_masked_pct)
        ids_keep = ids_shuffle[:, :len_keep]
        if shape is not None:
            x_unmasked = None
        else:
            ids_keep = ids_keep.unsqueeze(-1).expand(-1, -1, D)
            x_unmasked = torch.gather(x, dim=1, index=ids_keep)
        mask_info = MaskInfo(x_unmasked=x_unmasked, mask=mask, ids_restore=ids_restore, ids_keep=ids_keep)
        return mask_info

    def apply_mask(self, x, mask_info):
        cfg = self.modality_cfg
        B, T, C = x.shape
        if mask_info is not None:
            mask = mask_info.mask
            if cfg.encoder_zero_mask:
                x = x * (1 - mask.type_as(x).unsqueeze(-1))
            else:
                num_masks = mask.sum().item()
                masks = x.new_empty(num_masks, x.size(-1)).normal_(0, cfg.mask_noise_std)
                x = index_put(x, mask, masks)
        if cfg.mask_channel_prob > 0:
            mask_channel = compute_mask_indices((B, C), None, cfg.mask_channel_prob, cfg.mask_channel_length)
            mask_channel = torch.from_numpy(mask_channel).unsqueeze(1).expand(-1, T, -1)
            x = index_put(x, mask_channel, 0)
        return x

    def remove_pretraining_modules(self, keep_decoder=False):
        if not keep_decoder:
            self.decoder = None


class BlockEncoder(nn.Module):

    def __init__(self, blocks, norm_layer, layer_norm_first, layerdrop, dropout):
        super().__init__()
        self.blocks = blocks
        self.norm = norm_layer
        self.layer_norm_first = layer_norm_first
        self.layerdrop = layerdrop
        self.dropout = nn.Dropout(dropout, inplace=True)

    def forward(self, x, padding_mask, alibi_bias, alibi_scale):
        if self.norm is not None and not self.layer_norm_first:
            x = self.norm(x)
        x = self.dropout(x)
        for i, blk in enumerate(self.blocks):
            if not self.training or self.layerdrop == 0 or np.random.random() > self.layerdrop:
                ab = alibi_bias
                if ab is not None and alibi_scale is not None:
                    scale = alibi_scale[i] if alibi_scale.size(0) > 1 else alibi_scale.squeeze(0)
                    ab = ab * scale.type_as(ab)
                x, _ = blk(x, padding_mask, ab)
        if self.norm is not None and self.layer_norm_first:
            x = self.norm(x)
        return x


class DecoderBase(nn.Module):
    decoder_cfg: 'D2vDecoderConfig'

    def __init__(self, cfg: 'D2vDecoderConfig'):
        super().__init__()
        self.decoder_cfg = cfg

    def reset_parameters(self):
        for mod in self.proj.modules():
            if isinstance(mod, nn.Linear):
                mod.reset_parameters()

    def add_residual(self, x, residual, i, mask_info):
        if residual is None or not self.decoder_cfg.decoder_residual or residual.size(1) != x.size(1):
            return x
        ret = x + residual
        return ret


class Decoder2d(DecoderBase):

    def __init__(self, cfg: 'D2vDecoderConfig', input_dim, h_size, w_size):
        super().__init__(cfg)
        self.h_size = h_size
        self.w_size = w_size

        def make_block(in_dim):
            block = [nn.Conv2d(in_dim, cfg.decoder_dim, kernel_size=cfg.decoder_kernel, padding=cfg.decoder_kernel // 2, groups=cfg.decoder_groups), SamePad2d(cfg.decoder_kernel), TransposeLast(tranpose_dim=-3), LayerNorm(cfg.decoder_dim, elementwise_affine=False), TransposeLast(tranpose_dim=-3), nn.GELU()]
            return nn.Sequential(*block)
        self.blocks = nn.Sequential(*[make_block(input_dim if i == 0 else cfg.decoder_dim) for i in range(cfg.decoder_layers)])
        self.proj = nn.Linear(cfg.decoder_dim, input_dim)

    def forward(self, x, mask_info):
        B, T, C = x.shape
        x = x.transpose(1, 2).reshape(B, C, self.h_size, self.w_size)
        residual = x
        for i, layer in enumerate(self.blocks):
            x = layer(x)
            x = self.add_residual(x, residual, i, mask_info)
            residual = x
        x = x.reshape(B, -1, T).transpose(1, 2)
        x = self.proj(x)
        return x


class EncDecAttention(nn.Module):

    def __init__(self, q_dim, kv_dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0, cosine_attention=False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = q_dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.q_proj = nn.Linear(q_dim, q_dim, bias=qkv_bias)
        self.kv_proj = nn.Linear(kv_dim, 2 * q_dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(q_dim, q_dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.cosine_attention = cosine_attention
        if cosine_attention:
            self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))), requires_grad=True)

    def forward(self, q, kv, padding_mask=None, alibi_bias=None):
        B, N, C = q.shape
        q = self.q_proj(q).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        kv = self.kv_proj(kv).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        dtype = q.dtype
        if self.cosine_attention:
            attn = F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1)
            logit_scale = torch.clamp(self.logit_scale, max=torch.log(torch.tensor(1.0 / 0.01))).exp()
            attn = attn * logit_scale
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
        if alibi_bias is not None:
            attn = attn.type_as(alibi_bias)
            attn[:, :alibi_bias.size(1)] += alibi_bias
        if padding_mask is not None and padding_mask.any():
            attn = attn.masked_fill(padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))
        attn = attn.softmax(dim=-1, dtype=torch.float32)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2)
        x = x.reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class EncDecBlock(nn.Module):

    def __init__(self, q_dim, kv_dim, num_heads, mlp_ratio=4.0, qkv_bias=False, qk_scale=None, drop=0.0, attn_drop=0.0, mlp_drop=0.0, post_mlp_drop=0.0, drop_path=0.0, act_layer=nn.GELU, norm_layer=nn.LayerNorm, layer_norm_first=True, cosine_attention=False, first_residual=True):
        super().__init__()
        self.layer_norm_first = layer_norm_first
        self.norm1 = norm_layer(q_dim)
        self.attn = EncDecAttention(q_dim, kv_dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, cosine_attention=cosine_attention)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(q_dim)
        mlp_hidden_dim = int(q_dim * mlp_ratio)
        self.mlp = Mlp(in_features=q_dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=mlp_drop)
        self.post_mlp_dropout = nn.Dropout(post_mlp_drop, inplace=False)
        self.first_residual = first_residual

    def forward(self, q, kv, padding_mask=None, alibi_bias=None):
        r = q if self.first_residual else 0
        if self.layer_norm_first:
            x = r + self.drop_path(self.attn(self.norm1(q), kv, padding_mask, alibi_bias))
            r = x = self.mlp(self.norm2(x))
            x = r + self.drop_path(self.post_mlp_dropout(x))
        else:
            x = r + self.drop_path(self.attn(q, kv, padding_mask, alibi_bias))
            r = x = self.norm1(x)
            x = self.mlp(x)
            x = self.norm2(r + self.drop_path(self.post_mlp_dropout(x)))
        return x


class EncDecTransformerDecoder(nn.Module):

    def __init__(self, cfg: 'D2vDecoderConfig', input_dim):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, cfg.decoder_dim)
        self.blocks = nn.Sequential(*[EncDecBlock(q_dim=cfg.decoder_dim, kv_dim=input_dim, num_heads=8, mlp_ratio=4.0, qkv_bias=True, qk_scale=None, drop=0.0, attn_drop=0.0, mlp_drop=0.0, post_mlp_drop=0.0, drop_path=0.0, act_layer=nn.GELU, norm_layer=nn.LayerNorm, layer_norm_first=False, cosine_attention=False, first_residual=i > 0) for i in range(cfg.decoder_layers)])
        self.proj = nn.Linear(cfg.decoder_dim, input_dim)

    def reset_parameters(self):
        self.apply(init_bert_params)

    def forward(self, x, kv):
        x = self.input_proj(x)
        for i, layer in enumerate(self.blocks):
            x = layer(x, kv)
        x = self.proj(x)
        return x


class FixedPositionalEncoder(nn.Module):

    def __init__(self, pos_embed):
        super().__init__()
        self.positions = pos_embed

    def forward(self, x, padding_mask):
        return self.positions


class PatchEmbed_new(nn.Module):
    """ Flexible Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, stride=16):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        stride = to_2tuple(stride)
        self.img_size = img_size
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class TransformerDecoder(nn.Module):
    decoder_cfg: 'D2vDecoderConfig'

    def __init__(self, cfg: 'D2vDecoderConfig', input_dim, encoder):
        super().__init__()
        self.decoder_cfg = cfg
        self.input_proj = nn.Linear(input_dim, cfg.decoder_dim)
        self.encoder = encoder
        self.proj = nn.Linear(cfg.decoder_dim, input_dim)

    def reset_parameters(self):
        self.apply(init_bert_params)

    def forward(self, x, mask_info):
        x = self.input_proj(x)
        x = self.encoder(x, None, None, 1)
        x = self.proj(x)
        return x


def compute_block_mask_2d(shape: 'Tuple[int, int]', mask_prob: 'float', mask_length: 'int', mask_prob_adjust: 'float'=0, inverse_mask: 'bool'=False, require_same_masks: 'bool'=True, expand_adjcent: 'bool'=False, mask_dropout: 'float'=0, non_overlapping: 'bool'=False, img_shape: 'tuple'=None, flexible_mask: 'bool'=False) ->torch.Tensor:
    assert mask_length > 1
    B, L = shape
    d = int(L ** 0.5), int(L ** 0.5)
    if img_shape:
        d = img_shape[0], img_shape[1]
    if flexible_mask:
        index = np.random.randint(0, 3)
        block_size_options = np.array([(6, 4), (5, 5), (8, 3)])
        block_size = block_size_options[index]
    if inverse_mask:
        mask_prob = 1 - mask_prob
    if flexible_mask:
        mask = torch.zeros((B, d[0], d[1]))
        mask_inds = torch.randint(0, L, size=(B, int(L * ((mask_prob + mask_prob_adjust) / (block_size[0] * block_size[1])) * (1 + mask_dropout))))
        mask.view(B, -1).scatter_(1, mask_inds, 1)
        centers = mask.nonzero(as_tuple=True)
        inds = [], [], []
        offset = mask_length // 2
        for i in range(block_size[0]):
            for j in range(block_size[1]):
                k1 = i - offset
                k2 = j - offset
                inds[0].append(centers[0])
                inds[1].append(centers[1] + k1)
                inds[2].append(centers[2] + k2)
        i0 = torch.cat(inds[0])
        i1 = torch.cat(inds[1]).clamp_(min=0, max=d[0] - 1)
        i2 = torch.cat(inds[2]).clamp_(min=0, max=d[1] - 1)
        mask[i0, i1, i2] = 1
    elif non_overlapping:
        sz = math.ceil(d[0] / mask_length)
        inp_len = sz * sz
        inp = torch.zeros((B, 1, sz, sz))
        w = torch.ones((1, 1, mask_length, mask_length))
        mask_inds = torch.multinomial(1 - inp.view(B, -1), int(inp_len * (mask_prob + mask_prob_adjust) * (1 + mask_dropout)), replacement=False)
        inp.view(B, -1).scatter_(1, mask_inds, 1)
        mask = torch.nn.functional.conv_transpose2d(inp, w, stride=mask_length).squeeze(1)
        if mask.size(-1) > d[0]:
            mask = mask[..., :d, :d]
    else:
        mask = torch.zeros((B, d[0], d[1]))
        mask_inds = torch.randint(0, L, size=(B, int(L * ((mask_prob + mask_prob_adjust) / mask_length ** 2) * (1 + mask_dropout))))
        mask.view(B, -1).scatter_(1, mask_inds, 1)
        centers = mask.nonzero(as_tuple=True)
        inds = [], [], []
        offset = mask_length // 2
        for i in range(mask_length):
            for j in range(mask_length):
                k1 = i - offset
                k2 = j - offset
                inds[0].append(centers[0])
                inds[1].append(centers[1] + k1)
                inds[2].append(centers[2] + k2)
        i0 = torch.cat(inds[0])
        i1 = torch.cat(inds[1]).clamp_(min=0, max=d[0] - 1)
        i2 = torch.cat(inds[2]).clamp_(min=0, max=d[1] - 1)
        mask[i0, i1, i2] = 1

    def get_nbs(b, m, w):
        all_nbs = torch.nn.functional.conv2d(m.unsqueeze(1), w, padding='same')
        all_nbs = all_nbs.clamp_max_(1).view(b, -1)
        return all_nbs
    if require_same_masks and expand_adjcent:
        w = torch.zeros((1, 1, 3, 3))
        w[..., 0, 1] = 1
        w[..., 2, 1] = 1
        w[..., 1, 0] = 1
        w[..., 1, 2] = 1
        all_nbs = get_nbs(B, mask, w)
    mask = mask.reshape(B, -1)
    if require_same_masks:
        n_masks = mask.sum(dim=-1)
        final_target_len = int(L * mask_prob)
        target_len = int(final_target_len * (1 + mask_dropout))
        for i in range(len(mask)):
            n = n_masks[i]
            m = mask[i]
            r = 0
            while expand_adjcent and n < target_len:
                if r == 0:
                    nbs = all_nbs[i]
                else:
                    nbs = get_nbs(1, m.view(1, d[0], d[1]), w).flatten()
                cands = 1 - m + nbs > 1
                cand_sz = int(cands.sum().item())
                assert cand_sz > 0, f'{nbs} {cand_sz}'
                to_mask = torch.multinomial(cands.float(), min(cand_sz, int(target_len - n)), replacement=False)
                m[to_mask] = 1
                assert to_mask.numel() > 0
                n += to_mask.numel()
                r += 1
            if n > final_target_len:
                to_unmask = torch.multinomial(m, int(n - final_target_len), replacement=False)
                m[to_unmask] = 0
            elif n < final_target_len:
                to_mask = torch.multinomial(1 - m, int(final_target_len - n), replacement=False)
                m[to_mask] = 1
    if inverse_mask:
        mask = 1 - mask
    return mask


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
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


def get_2d_sincos_pos_embed_flexible(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size[0], dtype=np.float32)
    grid_w = np.arange(grid_size[1], dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_size[0], grid_size[1]])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_alibi(max_positions: 'int', attention_heads: 'int', dims: 'int'=1, distance: 'str'='manhattan'):

    def get_slopes(n):

        def get_slopes_power_of_2(n):
            start = 2 ** -2 ** -(math.log2(n) - 3)
            ratio = start
            return [(start * ratio ** i) for i in range(n)]
        if math.log2(n).is_integer():
            return get_slopes_power_of_2(n)
        else:
            closest_power_of_2 = 2 ** math.floor(math.log2(n))
            return get_slopes_power_of_2(closest_power_of_2) + get_slopes(2 * closest_power_of_2)[0::2][:n - closest_power_of_2]
    maxpos = max_positions
    attn_heads = attention_heads
    slopes = torch.Tensor(get_slopes(attn_heads))
    if dims == 1:
        pos_bias = torch.abs(torch.arange(maxpos).unsqueeze(0) - torch.arange(maxpos).unsqueeze(1)) * -1
    elif dims == 2:
        if distance == 'manhattan':
            df = lambda x1, y1, x2, y2: abs(x1 - x2) + abs(y1 - y2)
        elif distance == 'euclidean':
            df = lambda x1, y1, x2, y2: math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        n = math.sqrt(max_positions)
        assert n.is_integer(), n
        n = int(n)
        pos_bias = torch.zeros((max_positions, max_positions))
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    for l in range(n):
                        new_x = i * n + j
                        new_y = k * n + l
                        pos_bias[new_x, new_y] = -df(i, j, k, l)
    else:
        raise Exception(f'unsupported number of alibi dims: {dims}')
    alibi_bias = slopes.unsqueeze(1).unsqueeze(1) * pos_bias.unsqueeze(0).expand(attn_heads, -1, -1)
    return alibi_bias


def get_alibi_bias(alibi_biases, batch_size, time_steps, heads, dtype, device, dims=1, distance='manhattan'):
    cache_key = f'{dims}_{heads}_{distance}'
    buffered = alibi_biases.get(cache_key, None)
    target_size = heads * batch_size
    if buffered is None or buffered.size(0) < target_size or buffered.size(1) < time_steps or buffered.dtype != dtype or buffered.device != device:
        bt = max(time_steps, buffered.size(1) if buffered is not None else 0)
        bn = max(target_size, buffered.size(0) if buffered is not None else 0) // heads
        buffered = get_alibi(bt, heads, dims=dims, distance=distance).repeat(bn, 1, 1)
        alibi_biases[cache_key] = buffered
    b = buffered[:target_size, :time_steps, :time_steps]
    b = b.view(batch_size, heads, time_steps, time_steps)
    return b


class ImageEncoder(ModalitySpecificEncoder):
    modality_cfg: 'D2vImageConfig'

    def __init__(self, modality_cfg: 'D2vImageConfig', embed_dim: 'int', make_block: 'Callable[[float, Optional[int], Optional[int]], nn.ModuleList]', norm_layer: 'Callable[[int], nn.LayerNorm]', layer_norm_first: 'bool', alibi_biases: 'Dict', task: 'Optional[FairseqTask]'):
        if modality_cfg.in_chans == 1:
            img_size = modality_cfg.target_length, 128
        else:
            img_size = to_2tuple(modality_cfg.input_size)
        patch_size = to_2tuple(modality_cfg.patch_size)
        num_patches = img_size[1] // patch_size[1] * (img_size[0] // patch_size[0])
        self.H = img_size[0] // patch_size[0]
        self.W = img_size[1] // patch_size[1]
        self.hw = self.H, self.W
        local_encoder = PatchEmbed_new(img_size, modality_cfg.patch_size, modality_cfg.in_chans, modality_cfg.embed_dim)
        w = local_encoder.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        if modality_cfg.embed_dim != embed_dim:
            local_encoder = nn.Sequential(local_encoder, nn.Linear(modality_cfg.embed_dim, embed_dim))
        project_features = nn.Identity()
        max_length = modality_cfg.max_length
        pos_embed = nn.Parameter(torch.zeros(1, max_length * self.W, embed_dim), requires_grad=False)
        emb = get_2d_sincos_pos_embed_flexible(pos_embed.shape[-1], (max_length, self.W), cls_token=False)
        pos_embed.data.copy_(torch.from_numpy(emb[:max_length * self.W, :]).float().unsqueeze(0))
        fixed_positional_encoder = FixedPositionalEncoder(pos_embed) if modality_cfg.fixed_positions else None
        dpr = np.linspace(modality_cfg.start_drop_path_rate, modality_cfg.end_drop_path_rate, modality_cfg.prenet_depth)
        context_encoder = BlockEncoder(nn.ModuleList(make_block(dpr[i]) for i in range(modality_cfg.prenet_depth)), norm_layer(embed_dim) if not layer_norm_first else None, layer_norm_first, modality_cfg.prenet_layerdrop, modality_cfg.prenet_dropout)
        if modality_cfg.transformer_decoder:
            if modality_cfg.enc_dec_transformer:
                decoder = EncDecTransformerDecoder(modality_cfg.decoder, embed_dim)
            else:
                dec_enc = BlockEncoder(nn.ModuleList(make_block(0, modality_cfg.decoder.decoder_dim, 8) for _ in range(modality_cfg.decoder.decoder_layers)), None, layer_norm_first, 0, 0)
                decoder = TransformerDecoder(modality_cfg.decoder, embed_dim, dec_enc)
        else:
            decoder = Decoder2d(modality_cfg.decoder, embed_dim, self.H, self.W) if modality_cfg.decoder is not None else None
        alibi_bias_fn = partial(get_alibi_bias, alibi_biases=alibi_biases, heads=modality_cfg.num_alibi_heads, dims=modality_cfg.alibi_dims, distance=modality_cfg.alibi_distance)
        super().__init__(modality_cfg=modality_cfg, embed_dim=embed_dim, local_encoder=local_encoder, project_features=project_features, fixed_positional_encoder=fixed_positional_encoder, relative_positional_encoder=None, context_encoder=context_encoder, decoder=decoder, get_alibi_bias=alibi_bias_fn)

    def reset_parameters(self):
        super().reset_parameters()
        if self.decoder is not None:
            self.decoder.reset_parameters()

    @torch.no_grad()
    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)   audio: (N,1,H,W)   1024/16 = 64   128/16 = 8
        x: (N, L, patch_size**2 *3)
        """
        if self.modality_cfg.in_chans == 1:
            p = self.modality_cfg.patch_size
            h = imgs.shape[2] // p
            w = imgs.shape[3] // p
            x = imgs.reshape(shape=(imgs.shape[0], 1, h, p, w, p))
            x = torch.einsum('nchpwq->nhwpqc', x)
            x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * 1))
        else:
            p = self.modality_cfg.patch_size
            h = w = imgs.shape[2] // p
            x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
            x = torch.einsum('nchpwq->nhwpqc', x)
            x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * 3))
        return x

    @torch.no_grad()
    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.modality_cfg.patch_size
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def compute_mask(self, x, padding_mask, mask_seed: 'Optional[MaskSeed]', apply, shape=None, precomputed_mask=None):
        mlen = self.modality_cfg.mask_length
        if mlen <= 1:
            return super().compute_mask(x, padding_mask, mask_seed, apply, precomputed_mask)
        if precomputed_mask is not None:
            mask = precomputed_mask
        else:
            if shape is not None:
                B, L, D = shape
            else:
                B, L, D = x.shape
            mask = compute_block_mask_2d(shape=(B, L), mask_prob=self.modality_cfg.mask_prob, mask_length=self.modality_cfg.mask_length, mask_prob_adjust=self.modality_cfg.mask_prob_adjust, inverse_mask=self.modality_cfg.inverse_mask, require_same_masks=True, mask_dropout=self.modality_cfg.mask_dropout, img_shape=self.hw)
        mask_info = self.make_maskinfo(x, mask, shape)
        if apply:
            x = self.apply_mask(x, mask_info)
        return x, mask_info

    def decoder_input(self, x, mask_info):
        if not self.modality_cfg.transformer_decoder or not self.modality_cfg.enc_dec_transformer:
            return super().decoder_input(x, mask_info)
        inp_drop = self.modality_cfg.decoder.input_dropout
        if inp_drop > 0:
            x = F.dropout(x, inp_drop, training=self.training, inplace=True)
        kv = x[:, self.modality_cfg.num_extra_tokens:]
        assert self.fixed_positional_encoder is not None
        pos = self.fixed_positional_encoder(x, None).expand(x.size(0), -1, -1)
        mask = mask_info.mask.bool()
        if self.modality_cfg.decoder.add_positions_all:
            kv = kv + pos[~mask].view(kv.shape)
        q = pos[mask].view(x.size(0), -1, x.size(-1))
        return q, kv


class AltAttention(nn.Module):

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0, cosine_attention=False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.cosine_attention = cosine_attention
        if cosine_attention:
            self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))), requires_grad=True)

    def forward(self, x, padding_mask=None, alibi_bias=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        dtype = q.dtype
        if self.cosine_attention:
            attn = F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1)
            logit_scale = torch.clamp(self.logit_scale, max=torch.log(torch.tensor(1.0 / 0.01))).exp()
            attn = attn * logit_scale
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
        if alibi_bias is not None:
            attn = attn.type_as(alibi_bias)
            attn[:, :alibi_bias.size(1)] += alibi_bias
        if padding_mask is not None and padding_mask.any():
            attn = attn.masked_fill(padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))
        attn = attn.softmax(dim=-1, dtype=torch.float32)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2)
        x = x.reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class AltBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False, qk_scale=None, drop=0.0, attn_drop=0.0, mlp_drop=0.0, post_mlp_drop=0.0, drop_path=0.0, act_layer=nn.GELU, norm_layer=nn.LayerNorm, layer_norm_first=True, ffn_targets=False, cosine_attention=False):
        super().__init__()
        self.layer_norm_first = layer_norm_first
        self.ffn_targets = ffn_targets
        self.norm1 = norm_layer(dim)
        self.attn = AltAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, cosine_attention=cosine_attention)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=mlp_drop)
        self.post_mlp_dropout = nn.Dropout(post_mlp_drop, inplace=False)

    def forward(self, x, padding_mask=None, alibi_bias=None):
        if self.layer_norm_first:
            x = x + self.drop_path(self.attn(self.norm1(x), padding_mask, alibi_bias))
            r = x = self.mlp(self.norm2(x))
            t = x
            x = r + self.drop_path(self.post_mlp_dropout(x))
            if not self.ffn_targets:
                t = x
        else:
            x = x + self.drop_path(self.attn(x, padding_mask, alibi_bias))
            r = x = self.norm1(x)
            x = self.mlp(x)
            t = x
            x = self.norm2(r + self.drop_path(self.post_mlp_dropout(x)))
            if not self.ffn_targets:
                t = x
        return x, t


class RelativePositionBias(nn.Module):

    def __init__(self, window_size, num_heads):
        super().__init__()
        self.window_size = window_size
        self.num_relative_distance = (2 * window_size[0] - 1) * (2 * window_size[1] - 1) + 3
        self.relative_position_bias_table = nn.Parameter(torch.zeros(self.num_relative_distance, num_heads))
        coords_h = torch.arange(window_size[0])
        coords_w = torch.arange(window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size[0] - 1
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * window_size[1] - 1
        relative_position_index = torch.zeros(size=(window_size[0] * window_size[1] + 1,) * 2, dtype=relative_coords.dtype)
        relative_position_index[1:, 1:] = relative_coords.sum(-1)
        relative_position_index[0, 0:] = self.num_relative_distance - 3
        relative_position_index[0:, 0] = self.num_relative_distance - 2
        relative_position_index[0, 0] = self.num_relative_distance - 1
        self.register_buffer('relative_position_index', relative_position_index)

    def forward(self):
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(self.window_size[0] * self.window_size[1] + 1, self.window_size[0] * self.window_size[1] + 1, -1)
        return relative_position_bias.permute(2, 0, 1).contiguous()


class TextFeatPositionalEncoder(nn.Module):
    """
    Original encoder expects (B, T) long input. This module wraps it to take
    local_encoder output which are (B, T, D) float tensors
    """

    def __init__(self, pos_encoder):
        super().__init__()
        self.pos_encoder = pos_encoder

    def forward(self, x, padding_mask):
        return self.pos_encoder(x[..., 0])


class Decoder1d(DecoderBase):

    def __init__(self, cfg: 'D2vDecoderConfig', input_dim):
        super().__init__(cfg)

        def make_block(in_dim):
            block = [nn.Conv1d(in_dim, cfg.decoder_dim, kernel_size=cfg.decoder_kernel, padding=cfg.decoder_kernel // 2, groups=cfg.decoder_groups), SamePad(cfg.decoder_kernel), TransposeLast(), LayerNorm(cfg.decoder_dim, elementwise_affine=False), TransposeLast(), nn.GELU()]
            return nn.Sequential(*block)
        self.blocks = nn.Sequential(*[make_block(input_dim if i == 0 else cfg.decoder_dim) for i in range(cfg.decoder_layers)])
        projs = []
        curr_dim = cfg.decoder_dim
        for i in range(cfg.projection_layers - 1):
            next_dim = int(curr_dim * cfg.projection_ratio) if i == 0 else curr_dim
            projs.append(nn.Linear(curr_dim, next_dim))
            projs.append(nn.GELU())
            curr_dim = next_dim
        projs.append(nn.Linear(curr_dim, input_dim))
        if len(projs) == 1:
            self.proj = projs[0]
        else:
            self.proj = nn.Sequential(*projs)

    def forward(self, x, mask_info):
        x = x.transpose(1, 2)
        residual = x
        for i, layer in enumerate(self.blocks):
            x = layer(x)
            x = self.add_residual(x, residual, i, mask_info)
            residual = x
        x = x.transpose(1, 2)
        x = self.proj(x)
        return x


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (FixedPositionalEncoder,
     lambda: ([], {'pos_embed': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (RelativePositionBias,
     lambda: ([], {'window_size': [4, 4], 'num_heads': 4}),
     lambda: ([], {})),
    (TextFeatPositionalEncoder,
     lambda: ([], {'pos_encoder': torch.nn.ReLU()}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
]

