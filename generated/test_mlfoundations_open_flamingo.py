
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


import copy


import functools


import warnings


from typing import Any


from typing import cast


from typing import Dict


from typing import Iterable


from typing import Iterator


from typing import List


from typing import NamedTuple


from typing import Optional


from typing import Sequence


from typing import Set


from typing import Tuple


from typing import Union


import torch


import torch.distributed as dist


import torch.distributed.fsdp._traversal_utils as traversal_utils


import torch.nn as nn


from torch.distributed._shard.sharded_tensor import ShardedTensor


from torch.distributed.fsdp._common_utils import _apply_to_modules


from torch.distributed.fsdp._common_utils import _FSDPState


from torch.distributed.fsdp._common_utils import _get_module_fsdp_state_if_fully_sharded_module


from torch.distributed.fsdp._common_utils import _get_param_to_fqns


from torch.distributed.fsdp._common_utils import clean_tensor_name


from torch.distributed.fsdp._fsdp_extensions import _ext_chunk_tensor


from torch.distributed.fsdp._runtime_utils import _lazy_init


from torch.distributed.fsdp.api import ShardingStrategy


from torch.utils.data import Dataset


from torchvision.datasets import ImageFolder


import abc


from torch.nn.parallel import DistributedDataParallel as DDP


import uuid


import random


from collections import defaultdict


import numpy as np


from sklearn.metrics import roc_auc_score


import math


from torch import nn


from torch.distributed.fsdp.wrap import enable_wrap


from torch.distributed.fsdp.wrap import wrap


from torch.distributed.fsdp import FullyShardedDataParallel as FSDP


from torch import einsum


import re


import torchvision


from scipy.optimize import linear_sum_assignment


import logging


from torch.utils.data import DataLoader


from torch.utils.data import IterableDataset


from torch.utils.data import get_worker_info


from torch.utils.data.distributed import DistributedSampler


from torch.distributed.fsdp import CPUOffload


from torch.distributed.fsdp import MixedPrecision


from torch.distributed.fsdp import ShardingStrategy


from torch.distributed.fsdp import BackwardPrefetch


from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper


from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointWrapper


from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointImpl


from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import apply_activation_checkpointing


from torch.distributed.fsdp._init_utils import _init_intra_and_inter_node_groups


from torch.distributed.distributed_c10d import _get_default_group


import time


from torch.distributed.fsdp import FullStateDictConfig


from torch.distributed.fsdp import StateDictType


from torch.distributed.fsdp.api import FullOptimStateDictConfig


def FeedForward(dim, mult=4):
    inner_dim = int(dim * mult)
    return nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, inner_dim, bias=False), nn.GELU(), nn.Linear(inner_dim, dim, bias=False))


class PerceiverAttention(nn.Module):

    def __init__(self, *, dim, dim_head=64, heads=8):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        inner_dim = dim_head * heads
        self.norm_media = nn.LayerNorm(dim)
        self.norm_latents = nn.LayerNorm(dim)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x, latents):
        """
        Args:
            x (torch.Tensor): image features
                shape (b, T, n1, D)
            latent (torch.Tensor): latent features
                shape (b, T, n2, D)
        """
        x = self.norm_media(x)
        latents = self.norm_latents(latents)
        h = self.heads
        q = self.to_q(latents)
        kv_input = torch.cat((x, latents), dim=-2)
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)
        q, k, v = rearrange_many((q, k, v), 'b t n (h d) -> b h t n d', h=h)
        q = q * self.scale
        sim = einsum('... i d, ... j d  -> ... i j', q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)
        out = einsum('... i j, ... j d -> ... i d', attn, v)
        out = rearrange(out, 'b h t n d -> b t n (h d)', h=h)
        return self.to_out(out)


def exists(val):
    return val is not None


class PerceiverResampler(nn.Module):

    def __init__(self, *, dim, depth=6, dim_head=64, heads=8, num_latents=64, max_num_media=None, max_num_frames=None, ff_mult=4):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(num_latents, dim))
        self.frame_embs = nn.Parameter(torch.randn(max_num_frames, dim)) if exists(max_num_frames) else None
        self.media_time_embs = nn.Parameter(torch.randn(max_num_media, 1, dim)) if exists(max_num_media) else None
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads), FeedForward(dim=dim, mult=ff_mult)]))
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): image features
                shape (b, T, F, v, D)
        Returns:
            shape (b, T, n, D) where n is self.num_latents
        """
        b, T, F, v = x.shape[:4]
        if exists(self.frame_embs):
            frame_embs = repeat(self.frame_embs[:F], 'F d -> b T F v d', b=b, T=T, v=v)
            x = x + frame_embs
        x = rearrange(x, 'b T F v d -> b T (F v) d')
        if exists(self.media_time_embs):
            x = x + self.media_time_embs[:T]
        latents = repeat(self.latents, 'n d -> b T n d', b=b, T=T)
        for attn, ff in self.layers:
            latents = attn(x, latents) + latents
            latents = ff(latents) + latents
        return self.norm(latents)


def apply_with_stopping_condition(module, apply_fn, apply_condition=None, stopping_condition=None, **other_args):
    if stopping_condition(module):
        return
    if apply_condition(module):
        apply_fn(module, **other_args)
    for child in module.children():
        apply_with_stopping_condition(child, apply_fn, apply_condition=apply_condition, stopping_condition=stopping_condition, **other_args)


class Flamingo(nn.Module):

    def __init__(self, vision_encoder: 'nn.Module', lang_encoder: 'nn.Module', eoc_token_id: 'int', media_token_id: 'int', vis_dim: 'int', cross_attn_every_n_layers: 'int'=1, gradient_checkpointing: 'bool'=False):
        """
        Args:
            vision_encoder (nn.Module): HF CLIPModel
            lang_encoder (nn.Module): HF causal language model
            eoc_token_id (int): Token id for <|endofchunk|>
            media_token_id (int): Token id for <image>
            vis_dim (int): Dimension of the visual features.
                Visual features are projected to match this shape along the last dimension.
            cross_attn_every_n_layers (int, optional): How often to apply cross attention after transformer layer. Defaults to 1.
        """
        super().__init__()
        self.eoc_token_id = eoc_token_id
        self.media_token_id = media_token_id
        self.vis_dim = vis_dim
        if hasattr(lang_encoder.config, 'd_model'):
            self.lang_dim = lang_encoder.config.d_model
        else:
            self.lang_dim = lang_encoder.config.hidden_size
        self.vision_encoder = vision_encoder.visual
        self.perceiver = PerceiverResampler(dim=self.vis_dim)
        self.lang_encoder = lang_encoder
        self.lang_encoder.init_flamingo(media_token_id=media_token_id, lang_hidden_size=self.lang_dim, vis_hidden_size=self.vis_dim, cross_attn_every_n_layers=cross_attn_every_n_layers, gradient_checkpointing=gradient_checkpointing)
        self._use_gradient_checkpointing = gradient_checkpointing
        self.perceiver._use_gradient_checkpointing = gradient_checkpointing

    def forward(self, vision_x: 'torch.Tensor', lang_x: 'torch.Tensor', attention_mask: 'torch.Tensor'=None, labels: 'torch.Tensor'=None, clear_conditioned_layers: 'bool'=True, past_key_values=None, use_cache: 'bool'=False):
        """
        Forward pass of Flamingo.

        Args:
            vision_x (torch.Tensor): Vision input
                shape (B, T_img, F, C, H, W) with F=1
            lang_x (torch.Tensor): Language input ids
                shape (B, T_txt)
            attention_mask (torch.Tensor, optional): Attention mask. Defaults to None.
            labels (torch.Tensor, optional): Labels. Defaults to None.
            clear_conditioned_layers: if True, clear the conditioned layers
                once the foward pass is completed. Set this to false if the
                same set of images will be reused in another subsequent
                forward pass.
            past_key_values: pre-computed values to pass to language model.
                See past_key_values documentation in Hugging Face
                CausalLM models.
            use_cache: whether to use cached key values. See use_cache
                documentation in Hugging Face CausalLM models.
        """
        assert self.lang_encoder.initialized_flamingo, 'Flamingo layers are not initialized. Please call `init_flamingo` first.'
        assert self.lang_encoder._use_cached_vision_x or vision_x is not None, 'Must provide either vision_x or have precached media using cache_media().'
        if self.lang_encoder._use_cached_vision_x:
            assert vision_x is None, 'Expect vision_x to be None when media has been cached using cache_media(). Try uncache_media() first.'
            assert self.lang_encoder.is_conditioned()
        else:
            self._encode_vision_x(vision_x=vision_x)
            self._condition_media_locations(input_ids=lang_x)
        output = self.lang_encoder(input_ids=lang_x, attention_mask=attention_mask, labels=labels, past_key_values=past_key_values, use_cache=use_cache)
        if clear_conditioned_layers:
            self.lang_encoder.clear_conditioned_layers()
        return output

    def generate(self, vision_x: 'torch.Tensor', lang_x: 'torch.Tensor', attention_mask: 'torch.Tensor'=None, **kwargs):
        """
        Generate text conditioned on vision and language inputs.

        Args:
            vision_x (torch.Tensor): Vision input
                shape (B, T_img, F, C, H, W)
                images in the same chunk are collated along T_img, and frames are collated along F
                currently only F=1 is supported (single-frame videos)
            lang_x (torch.Tensor): Language input
                shape (B, T_txt)
            **kwargs: see generate documentation in Hugging Face CausalLM models. Some notable kwargs:
                max_length (int, optional): Maximum length of the output. Defaults to None.
                attention_mask (torch.Tensor, optional): Attention mask. Defaults to None.
                num_beams (int, optional): Number of beams. Defaults to 1.
                max_new_tokens (int, optional): Maximum new tokens. Defaults to None.
                temperature (float, optional): Temperature. Defaults to 1.0.
                top_k (int, optional): Top k. Defaults to 50.
                top_p (float, optional): Top p. Defaults to 1.0.
                no_repeat_ngram_size (int, optional): No repeat ngram size. Defaults to 0.
                length_penalty (float, optional): Length penalty. Defaults to 1.0.
                num_return_sequences (int, optional): Number of return sequences. Defaults to 1.
                do_sample (bool, optional): Do sample. Defaults to False.
                early_stopping (bool, optional): Early stopping. Defaults to False.
        Returns:
            torch.Tensor: lang_x with generated tokens appended to it
        """
        num_beams = kwargs.pop('num_beams', 1)
        if num_beams > 1:
            vision_x = vision_x.repeat_interleave(num_beams, dim=0)
        self.lang_encoder._use_cached_vision_x = True
        self._encode_vision_x(vision_x=vision_x)
        eos_token_id = kwargs.pop('eos_token_id', self.eoc_token_id)
        output = self.lang_encoder.generate(input_ids=lang_x, attention_mask=attention_mask, eos_token_id=eos_token_id, num_beams=num_beams, **kwargs)
        self.lang_encoder.clear_conditioned_layers()
        self.lang_encoder._use_cached_vision_x = False
        return output

    def _encode_vision_x(self, vision_x: 'torch.Tensor'):
        """
        Compute media tokens from vision input by passing it through vision encoder and conditioning language model.
        Args:
            vision_x (torch.Tensor): Vision input
                shape (B, T_img, F, C, H, W)
                Images in the same chunk are collated along T_img, and frames are collated along F
                Currently only F=1 is supported (single-frame videos)

        rearrange code based on https://github.com/dhansmair/flamingo-mini
        """
        assert vision_x.ndim == 6, 'vision_x should be of shape (b, T_img, F, C, H, W)'
        b, T, F = vision_x.shape[:3]
        assert F == 1, 'Only single frame supported'
        vision_x = rearrange(vision_x, 'b T F c h w -> (b T F) c h w')
        with torch.no_grad():
            vision_x = self.vision_encoder(vision_x)[1]
        vision_x = rearrange(vision_x, '(b T F) v d -> b T F v d', b=b, T=T, F=F)
        vision_x = self.perceiver(vision_x)
        for layer in self.lang_encoder._get_decoder_layers():
            layer.condition_vis_x(vision_x)

    def wrap_fsdp(self, wrapper_kwargs, device_id):
        """
        Manually wraps submodules for FSDP and move other parameters to device_id.

        Why manually wrap?
        - all parameters within the FSDP wrapper must have the same requires_grad.
            We have a mix of frozen and unfrozen parameters.
        - model.vision_encoder.visual needs to be individually wrapped or encode_vision_x errors
            See: https://github.com/pytorch/pytorch/issues/82461#issuecomment-1269136344

        The rough wrapping structure is:
        - FlamingoModel
            - FSDP(FSDP(vision_encoder))
            - FSDP(FSDP(perceiver))
            - lang_encoder
                - FSDP(FSDP(input_embeddings))
                - FlamingoLayers
                    - FSDP(FSDP(gated_cross_attn_layer))
                    - FSDP(FSDP(decoder_layer))
                - FSDP(FSDP(output_embeddings))
                - other parameters

        Known issues:
        - Our FSDP strategy is not compatible with tied embeddings. If the LM embeddings are tied,
            train with DDP or set the --freeze_lm_embeddings flag to true.
        - With FSDP + gradient ckpting, one can increase the batch size with seemingly no upper bound.
            Although the training curves look okay, we found that downstream performance dramatically
            degrades if the batch size is unreasonably large (e.g., 100 MMC4 batch size for OPT-125M).

        FAQs about our FSDP wrapping strategy:
        Why double wrap?
        As of torch==2.0.1, FSDP's _post_forward_hook and _post_backward_hook
        only free gathered parameters if the module is NOT FSDP root.

        Why unfreeze the decoder_layers?
        See https://github.com/pytorch/pytorch/issues/95805
        As of torch==2.0.1, FSDP's _post_backward_hook is only registed if the flat param
        requires_grad=True. We need the postback to fire to avoid OOM.
        To effectively freeze the decoder layers, we exclude them from the optimizer.

        What is assumed to be frozen v. unfrozen?
        We assume that the model is being trained under normal Flamingo settings
        with these lines being called in factory.py:
            ```
            # Freeze all parameters
            model.requires_grad_(False)
            assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 0

            # Unfreeze perceiver, gated_cross_attn_layers, and LM input embeddings
            model.perceiver.requires_grad_(True)
            model.lang_encoder.gated_cross_attn_layers.requires_grad_(True)
            [optional] model.lang_encoder.get_input_embeddings().requires_grad_(True)
            ```
        """
        for block in self.lang_encoder.old_decoder_blocks:
            block.requires_grad_(True)
        with enable_wrap(wrapper_cls=FSDP, **wrapper_kwargs):
            self.perceiver = wrap(wrap(self.perceiver))
            self.lang_encoder.old_decoder_blocks = nn.ModuleList(wrap(wrap(block)) for block in self.lang_encoder.old_decoder_blocks)
            self.lang_encoder.gated_cross_attn_layers = nn.ModuleList(wrap(wrap(layer)) if layer is not None else None for layer in self.lang_encoder.gated_cross_attn_layers)
            self.lang_encoder.init_flamingo_layers(self._use_gradient_checkpointing)
            self.lang_encoder.set_input_embeddings(wrap(wrap(self.lang_encoder.get_input_embeddings())))
            self.lang_encoder.set_output_embeddings(wrap(wrap(self.lang_encoder.get_output_embeddings())))
            self.vision_encoder = wrap(wrap(self.vision_encoder))
        apply_with_stopping_condition(module=self.lang_encoder, apply_fn=lambda m: m, apply_condition=lambda m: len(list(m.children())) == 0, stopping_condition=lambda m: isinstance(m, FSDP))
        for block in self.lang_encoder.old_decoder_blocks:
            for p in block.parameters():
                p.exclude_from_optimizer = True

        def clip_grad_norm_(max_norm):
            self.perceiver.clip_grad_norm_(max_norm)
            for layer in self.lang_encoder.gated_cross_attn_layers:
                if layer is not None:
                    layer.clip_grad_norm_(max_norm)
            self.lang_encoder.get_input_embeddings().clip_grad_norm_(max_norm)
        self.clip_grad_norm_ = clip_grad_norm_

    def _condition_media_locations(self, input_ids: 'torch.Tensor'):
        """
        Compute the media token locations from lang_x and condition the language model on these.
        Args:
            input_ids (torch.Tensor): Language input
                shape (B, T_txt)
        """
        media_locations = input_ids == self.media_token_id
        for layer in self.lang_encoder._get_decoder_layers():
            layer.condition_media_locations(media_locations)

    def cache_media(self, input_ids: 'torch.Tensor', vision_x: 'torch.Tensor'):
        """
        Pre-cache a prompt/sequence of images / text for log-likelihood evaluations.
        All subsequent calls to forward() will generate attending to the LAST
        image in vision_x.
        This is not meant to be used to cache things for generate().
        Args:
            input_ids (torch.Tensor): Language input
                shape (B, T_txt)
            vision_x (torch.Tensor): Vision input
                shape (B, T_img, F, C, H, W)
                Images in the same chunk are collated along T_img, and frames are collated along F
                Currently only F=1 is supported (single-frame videos)
        """
        self._encode_vision_x(vision_x=vision_x)
        self._condition_media_locations(input_ids=input_ids)
        self.lang_encoder._use_cached_vision_x = True

    def uncache_media(self):
        """
        Clear all conditioning.
        """
        self.lang_encoder.clear_conditioned_layers()
        self.lang_encoder._use_cached_vision_x = False


class FlamingoLayer(nn.Module):
    """
    FlamingoLayer is a wrapper around the GatedCrossAttentionBlock and DecoderLayer.
    """

    def __init__(self, gated_cross_attn_layer, decoder_layer, gradient_checkpointing=False):
        super().__init__()
        self.gated_cross_attn_layer = gated_cross_attn_layer
        self.decoder_layer = decoder_layer
        self.vis_x = None
        self.media_locations = None
        if self.gated_cross_attn_layer is not None:
            self.gated_cross_attn_layer._use_gradient_checkpointing = gradient_checkpointing
        self.decoder_layer._use_gradient_checkpointing = gradient_checkpointing

    def is_conditioned(self) ->bool:
        """Check whether the layer is conditioned."""
        return self.vis_x is not None and self.media_locations is not None

    def condition_vis_x(self, vis_x):
        self.vis_x = vis_x

    def condition_media_locations(self, media_locations):
        self.media_locations = media_locations

    def condition_use_cached_media(self, use_cached_media):
        self.use_cached_media = use_cached_media

    def forward(self, lang_x, attention_mask=None, **decoder_layer_kwargs):
        if self.gated_cross_attn_layer is not None:
            if self.vis_x is None:
                raise ValueError('vis_x must be conditioned before forward pass')
            if self.media_locations is None:
                raise ValueError('media_locations must be conditioned before forward pass')
            lang_x = self.gated_cross_attn_layer(lang_x, self.vis_x, media_locations=self.media_locations, use_cached_media=self.use_cached_media)
        lang_x = self.decoder_layer(lang_x, attention_mask=attention_mask, **decoder_layer_kwargs)
        return lang_x


class MaskedCrossAttention(nn.Module):

    def __init__(self, *, dim, dim_visual, dim_head=64, heads=8, only_attend_immediate_media=True):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        inner_dim = dim_head * heads
        self.norm = nn.LayerNorm(dim)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim_visual, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)
        self.only_attend_immediate_media = only_attend_immediate_media

    def forward(self, x, media, media_locations=None, use_cached_media=False):
        """
        Args:
            x (torch.Tensor): text features
                shape (B, T_txt, D_txt)
            media (torch.Tensor): image features
                shape (B, T_img, n, D_img) where n is the dim of the latents
            media_locations: boolean mask identifying the media tokens in x
                shape (B, T_txt)
            use_cached_media: bool
                If true, treat all of x as if they occur after the last media
                registered in media_locations. T_txt does not need to exactly
                equal media_locations.shape[1] in this case
        """
        if not use_cached_media:
            assert media_locations.shape[1] == x.shape[1], f'media_location.shape is {media_locations.shape} but x.shape is {x.shape}'
        T_txt = x.shape[1]
        _, T_img, n = media.shape[:3]
        h = self.heads
        x = self.norm(x)
        q = self.to_q(x)
        media = rearrange(media, 'b t n d -> b (t n) d')
        k, v = self.to_kv(media).chunk(2, dim=-1)
        q, k, v = rearrange_many((q, k, v), 'b n (h d) -> b h n d', h=h)
        q = q * self.scale
        sim = einsum('... i d, ... j d -> ... i j', q, k)
        if exists(media_locations):
            media_time = torch.arange(T_img, device=x.device) + 1
            if use_cached_media:
                text_time = repeat(torch.count_nonzero(media_locations, dim=1), 'b -> b i', i=T_txt)
            else:
                text_time = media_locations.cumsum(dim=-1)
            mask_op = torch.eq if self.only_attend_immediate_media else torch.ge
            text_to_media_mask = mask_op(rearrange(text_time, 'b i -> b 1 i 1'), repeat(media_time, 'j -> 1 1 1 (j n)', n=n))
            sim = sim.masked_fill(~text_to_media_mask, -torch.finfo(sim.dtype).max)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)
        if exists(media_locations) and self.only_attend_immediate_media:
            text_without_media_mask = text_time == 0
            text_without_media_mask = rearrange(text_without_media_mask, 'b i -> b 1 i 1')
            attn = attn.masked_fill(text_without_media_mask, 0.0)
        out = einsum('... i j, ... j d -> ... i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class GatedCrossAttentionBlock(nn.Module):

    def __init__(self, *, dim, dim_visual, dim_head=64, heads=8, ff_mult=4, only_attend_immediate_media=True):
        super().__init__()
        self.attn = MaskedCrossAttention(dim=dim, dim_visual=dim_visual, dim_head=dim_head, heads=heads, only_attend_immediate_media=only_attend_immediate_media)
        self.attn_gate = nn.Parameter(torch.tensor([0.0]))
        self.ff = FeedForward(dim, mult=ff_mult)
        self.ff_gate = nn.Parameter(torch.tensor([0.0]))

    def forward(self, x, media, media_locations=None, use_cached_media=False):
        x = self.attn(x, media, media_locations=media_locations, use_cached_media=use_cached_media) * self.attn_gate.tanh() + x
        x = self.ff(x) * self.ff_gate.tanh() + x
        return x


def getattr_recursive(obj, att):
    """
    Return nested attribute of obj
    Example: getattr_recursive(obj, 'a.b.c') is equivalent to obj.a.b.c
    """
    if att == '':
        return obj
    i = att.find('.')
    if i < 0:
        return getattr(obj, att)
    else:
        return getattr_recursive(getattr(obj, att[:i]), att[i + 1:])


def setattr_recursive(obj, att, val):
    """
    Set nested attribute of obj
    Example: setattr_recursive(obj, 'a.b.c', val) is equivalent to obj.a.b.c = val
    """
    if '.' in att:
        obj = getattr_recursive(obj, '.'.join(att.split('.')[:-1]))
    setattr(obj, att.split('.')[-1], val)


class FlamingoLMMixin(nn.Module):
    """
    Mixin to add cross-attention layers to a language model.
    """

    def set_decoder_layers_attr_name(self, decoder_layers_attr_name):
        self.decoder_layers_attr_name = decoder_layers_attr_name

    def _get_decoder_layers(self):
        return getattr_recursive(self, self.decoder_layers_attr_name)

    def _set_decoder_layers(self, value):
        setattr_recursive(self, self.decoder_layers_attr_name, value)

    def init_flamingo(self, media_token_id, lang_hidden_size, vis_hidden_size, cross_attn_every_n_layers, gradient_checkpointing):
        """
        Initialize Flamingo by adding a new gated cross attn to the decoder. Store the media token id for computing the media locations.
        """
        self.old_decoder_blocks = self._get_decoder_layers()
        self.gated_cross_attn_layers = nn.ModuleList([(GatedCrossAttentionBlock(dim=lang_hidden_size, dim_visual=vis_hidden_size) if (layer_idx + 1) % cross_attn_every_n_layers == 0 else None) for layer_idx, _ in enumerate(self._get_decoder_layers())])
        self.init_flamingo_layers(gradient_checkpointing)
        self.media_token_id = media_token_id
        self.initialized_flamingo = True
        self._use_cached_vision_x = False

    def init_flamingo_layers(self, gradient_checkpointing):
        """
        Re initializes the FlamingoLayers.
        Propagates any changes made to self.gated_corss_attn_layers or self.old_decoder_blocks
        """
        self._set_decoder_layers(nn.ModuleList([FlamingoLayer(gated_cross_attn_layer, decoder_layer, gradient_checkpointing) for gated_cross_attn_layer, decoder_layer in zip(self.gated_cross_attn_layers, self.old_decoder_blocks)]))

    def forward(self, input_ids, attention_mask, **kwargs):
        """Condition the Flamingo layers on the media locations before forward()"""
        if not self.initialized_flamingo:
            raise ValueError('Flamingo layers are not initialized. Please call `init_flamingo` first.')
        media_locations = input_ids == self.media_token_id
        use_cached_media_locations = self._use_cached_vision_x and self.is_conditioned() and not media_locations.any()
        for layer in self._get_decoder_layers():
            if not use_cached_media_locations:
                layer.condition_media_locations(media_locations)
            layer.condition_use_cached_media(use_cached_media_locations)
        kwargs['input_ids'] = input_ids
        kwargs['attention_mask'] = attention_mask
        return super().forward(**kwargs)

    def is_conditioned(self) ->bool:
        """Check whether all decoder layers are already conditioned."""
        return all(l.is_conditioned() for l in self._get_decoder_layers())

    def clear_conditioned_layers(self):
        for layer in self._get_decoder_layers():
            layer.condition_vis_x(None)
            layer.condition_media_locations(None)
            layer.condition_use_cached_media(None)

