import sys
_module = sys.modules[__name__]
del sys
update = _module
api_server = _module
routes = _module
internal = _module
internal_routes = _module
services = _module
file_service = _module
file_operations = _module
app = _module
app_settings = _module
frontend_management = _module
logger = _module
user_manager = _module
checkpoint_pickle = _module
cldm = _module
control_types = _module
mmdit = _module
cli_args = _module
clip_model = _module
clip_vision = _module
comfy_types = _module
conds = _module
controlnet = _module
diffusers_convert = _module
diffusers_load = _module
uni_pc = _module
float = _module
gligen = _module
deis = _module
sampling = _module
utils = _module
latent_formats = _module
autoencoder = _module
dit = _module
embedders = _module
mmdit = _module
common = _module
controlnet = _module
stage_a = _module
stage_b = _module
stage_c = _module
stage_c_coder = _module
common_dit = _module
controlnet = _module
layers = _module
math = _module
model = _module
asymm_models_joint = _module
layers = _module
rope_mixed = _module
temporal_rope = _module
utils = _module
model = _module
attn_layers = _module
controlnet = _module
models = _module
poolers = _module
posemb_layers = _module
autoencoder = _module
attention = _module
diffusionmodules = _module
mmdit = _module
model = _module
openaimodel = _module
upscaling = _module
util = _module
distributions = _module
distributions = _module
ema = _module
encoders = _module
noise_aug_modules = _module
sub_quadratic_attention = _module
temporal_ae = _module
util = _module
lora = _module
model_base = _module
model_detection = _module
model_management = _module
model_patcher = _module
model_sampling = _module
ops = _module
options = _module
sample = _module
sampler_helpers = _module
samplers = _module
sd = _module
sd1_clip = _module
sdxl_clip = _module
supported_models = _module
supported_models_base = _module
adapter = _module
taesd = _module
aura_t5 = _module
bert = _module
flux = _module
genmo = _module
hydit = _module
long_clipl = _module
sa_t5 = _module
sd2_clip = _module
sd3_clip = _module
spiece_tokenizer = _module
t5 = _module
utils = _module
caching = _module
graph = _module
graph_utils = _module
model_loading = _module
nodes_advanced_samplers = _module
nodes_align_your_steps = _module
nodes_attention_multiply = _module
nodes_audio = _module
nodes_canny = _module
nodes_clip_sdxl = _module
nodes_compositing = _module
nodes_cond = _module
nodes_controlnet = _module
nodes_custom_sampler = _module
nodes_differential_diffusion = _module
nodes_flux = _module
nodes_freelunch = _module
nodes_gits = _module
nodes_hunyuan = _module
nodes_hypernetwork = _module
nodes_hypertile = _module
nodes_images = _module
nodes_ip2p = _module
nodes_latent = _module
nodes_lora_extract = _module
nodes_mask = _module
nodes_mochi = _module
nodes_model_advanced = _module
nodes_model_downscale = _module
nodes_model_merging = _module
nodes_model_merging_model_specific = _module
nodes_morphology = _module
nodes_pag = _module
nodes_perpneg = _module
nodes_photomaker = _module
nodes_post_processing = _module
nodes_rebatch = _module
nodes_sag = _module
nodes_sd3 = _module
nodes_sdupscale = _module
nodes_stable3d = _module
nodes_stable_cascade = _module
nodes_tomesd = _module
nodes_torch_compile = _module
nodes_upscale_model = _module
nodes_video_model = _module
nodes_webcam = _module
cuda_malloc = _module
websocket_image_save = _module
execution = _module
fix_torch = _module
folder_paths = _module
latent_preview = _module
main = _module
model_filemanager = _module
download_models = _module
new_updater = _module
node_helpers = _module
nodes = _module
basic_api_example = _module
websockets_api_example = _module
websockets_api_example_ws_images = _module
server = _module
app_test = _module
frontend_manager_test = _module
folder_path_test = _module
folder_paths_test = _module
filter_by_content_types_test = _module
prompt_server_test = _module
download_models_test = _module
user_manager_test = _module
internal_routes_test = _module
file_service_test = _module
file_operations_test = _module
extra_config_test = _module
tests = _module
conftest = _module
test_quality = _module
inference = _module
test_execution = _module
test_inference = _module
pack = _module
conditions = _module
flow_control = _module
specific_tests = _module
stubs = _module
tools = _module
extra_config = _module

from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, matplotlib, numbers, numpy, pandas, queue, random, re, scipy, sklearn, string, tensorflow, time, torch, torchaudio, torchvision, types, typing, uuid, warnings
import operator as op
from dataclasses import dataclass
import numpy as np
from torch import Tensor
__version__ = '1.0.0'
xrange = range
wraps = functools.wraps


import torch


import torch as th


import torch.nn as nn


from collections import OrderedDict


from typing import Dict


from typing import Optional


import enum


import logging


from typing import Callable


from typing import Protocol


from typing import TypedDict


from typing import List


import math


from enum import Enum


import re


import torch.nn.functional as F


from torch import nn


from inspect import isfunction


import numpy as np


from scipy import integrate


import warnings


from torch import optim


from torch.utils import data


from typing import Literal


from typing import Any


import typing as tp


from torch.nn import functional as F


from torch import Tensor


from torch import einsum


from typing import Sequence


from typing import Tuple


from typing import TypeVar


from typing import Union


import torchvision


from torch.autograd import Function


import collections.abc


from itertools import repeat


from functools import partial


from torch.utils import checkpoint


from abc import abstractmethod


from torch.utils.checkpoint import checkpoint


import functools


from typing import Iterable


import copy


import inspect


import uuid


import collections


import scipy.stats


import numpy


import numbers


import itertools


import torchaudio


import random


from torch import randint


import scipy.ndimage


import time


from typing import NamedTuple


from copy import deepcopy


def exists(x):
    return x is not None


def get_attn_precision(attn_precision):
    if args.dont_upcast_attention:
        return None
    if FORCE_UPCAST_ATTENTION_DTYPE is not None:
        return FORCE_UPCAST_ATTENTION_DTYPE
    return attn_precision


def attention_basic(q, k, v, heads, mask=None, attn_precision=None, skip_reshape=False):
    attn_precision = get_attn_precision(attn_precision)
    if skip_reshape:
        b, _, _, dim_head = q.shape
    else:
        b, _, dim_head = q.shape
        dim_head //= heads
    scale = dim_head ** -0.5
    h = heads
    if skip_reshape:
        q, k, v = map(lambda t: t.reshape(b * heads, -1, dim_head), (q, k, v))
    else:
        q, k, v = map(lambda t: t.unsqueeze(3).reshape(b, -1, heads, dim_head).permute(0, 2, 1, 3).reshape(b * heads, -1, dim_head).contiguous(), (q, k, v))
    if attn_precision == torch.float32:
        sim = einsum('b i d, b j d -> b i j', q.float(), k.float()) * scale
    else:
        sim = einsum('b i d, b j d -> b i j', q, k) * scale
    del q, k
    if exists(mask):
        if mask.dtype == torch.bool:
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)
        else:
            if len(mask.shape) == 2:
                bs = 1
            else:
                bs = mask.shape[0]
            mask = mask.reshape(bs, -1, mask.shape[-2], mask.shape[-1]).expand(b, heads, -1, -1).reshape(-1, mask.shape[-2], mask.shape[-1])
            sim.add_(mask)
    sim = sim.softmax(dim=-1)
    out = einsum('b i j, b j d -> b i d', sim, v)
    out = out.unsqueeze(0).reshape(b, heads, -1, dim_head).permute(0, 2, 1, 3).reshape(b, -1, heads * dim_head)
    return out


optimized_attention = attention_basic


class OptimizedAttention(nn.Module):

    def __init__(self, c, nhead, dropout=0.0, dtype=None, device=None, operations=None):
        super().__init__()
        self.heads = nhead
        self.to_q = operations.Linear(c, c, bias=True, dtype=dtype, device=device)
        self.to_k = operations.Linear(c, c, bias=True, dtype=dtype, device=device)
        self.to_v = operations.Linear(c, c, bias=True, dtype=dtype, device=device)
        self.out_proj = operations.Linear(c, c, bias=True, dtype=dtype, device=device)

    def forward(self, q, k, v):
        q = self.to_q(q)
        k = self.to_k(k)
        v = self.to_v(v)
        out = optimized_attention(q, k, v, self.heads)
        return self.out_proj(out)


class QuickGELU(nn.Module):

    def forward(self, x: 'torch.Tensor'):
        return x * torch.sigmoid(1.702 * x)


class ResBlockUnionControlnet(nn.Module):

    def __init__(self, dim, nhead, dtype=None, device=None, operations=None):
        super().__init__()
        self.attn = OptimizedAttention(dim, nhead, dtype=dtype, device=device, operations=operations)
        self.ln_1 = operations.LayerNorm(dim, dtype=dtype, device=device)
        self.mlp = nn.Sequential(OrderedDict([('c_fc', operations.Linear(dim, dim * 4, dtype=dtype, device=device)), ('gelu', QuickGELU()), ('c_proj', operations.Linear(dim * 4, dim, dtype=dtype, device=device))]))
        self.ln_2 = operations.LayerNorm(dim, dtype=dtype, device=device)

    def attention(self, x: 'torch.Tensor'):
        return self.attn(x)

    def forward(self, x: 'torch.Tensor'):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


def LayerNorm2d_op(operations):


    class LayerNorm2d(operations.LayerNorm):

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def forward(self, x):
            return super().forward(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    return LayerNorm2d


class CNetResBlock(nn.Module):

    def __init__(self, c, dtype=None, device=None, operations=None):
        super().__init__()
        self.blocks = nn.Sequential(LayerNorm2d_op(operations)(c, dtype=dtype, device=device), nn.GELU(), operations.Conv2d(c, c, kernel_size=3, padding=1), LayerNorm2d_op(operations)(c, dtype=dtype, device=device), nn.GELU(), operations.Conv2d(c, c, kernel_size=3, padding=1))

    def forward(self, x):
        return x + self.blocks(x)


class ControlNet(nn.Module):

    def __init__(self, c_in=3, c_proj=2048, proj_blocks=None, bottleneck_mode=None, dtype=None, device=None, operations=nn):
        super().__init__()
        if bottleneck_mode is None:
            bottleneck_mode = 'effnet'
        self.proj_blocks = proj_blocks
        if bottleneck_mode == 'effnet':
            embd_channels = 1280
            self.backbone = torchvision.models.efficientnet_v2_s().features.eval()
            if c_in != 3:
                in_weights = self.backbone[0][0].weight.data
                self.backbone[0][0] = operations.Conv2d(c_in, 24, kernel_size=3, stride=2, bias=False, dtype=dtype, device=device)
                if c_in > 3:
                    self.backbone[0][0].weight.data[:, :3] = in_weights[:, :3].clone()
                else:
                    self.backbone[0][0].weight.data = in_weights[:, :c_in].clone()
        elif bottleneck_mode == 'simple':
            embd_channels = c_in
            self.backbone = nn.Sequential(operations.Conv2d(embd_channels, embd_channels * 4, kernel_size=3, padding=1, dtype=dtype, device=device), nn.LeakyReLU(0.2, inplace=True), operations.Conv2d(embd_channels * 4, embd_channels, kernel_size=3, padding=1, dtype=dtype, device=device))
        elif bottleneck_mode == 'large':
            self.backbone = nn.Sequential(operations.Conv2d(c_in, 4096 * 4, kernel_size=1, dtype=dtype, device=device), nn.LeakyReLU(0.2, inplace=True), operations.Conv2d(4096 * 4, 1024, kernel_size=1, dtype=dtype, device=device), *[CNetResBlock(1024, dtype=dtype, device=device, operations=operations) for _ in range(8)], operations.Conv2d(1024, 1280, kernel_size=1, dtype=dtype, device=device))
            embd_channels = 1280
        else:
            raise ValueError(f'Unknown bottleneck mode: {bottleneck_mode}')
        self.projections = nn.ModuleList()
        for _ in range(len(proj_blocks)):
            self.projections.append(nn.Sequential(operations.Conv2d(embd_channels, embd_channels, kernel_size=1, bias=False, dtype=dtype, device=device), nn.LeakyReLU(0.2, inplace=True), operations.Conv2d(embd_channels, c_proj, kernel_size=1, bias=False, dtype=dtype, device=device)))
        self.xl = False
        self.input_channels = c_in
        self.unshuffle_amount = 8

    def forward(self, x):
        x = self.backbone(x)
        proj_outputs = [None for _ in range(max(self.proj_blocks) + 1)]
        for i, idx in enumerate(self.proj_blocks):
            proj_outputs[idx] = self.projections[i](x)
        return {'input': proj_outputs[::-1]}


class CLIPAttention(torch.nn.Module):

    def __init__(self, embed_dim, heads, dtype, device, operations):
        super().__init__()
        self.heads = heads
        self.q_proj = operations.Linear(embed_dim, embed_dim, bias=True, dtype=dtype, device=device)
        self.k_proj = operations.Linear(embed_dim, embed_dim, bias=True, dtype=dtype, device=device)
        self.v_proj = operations.Linear(embed_dim, embed_dim, bias=True, dtype=dtype, device=device)
        self.out_proj = operations.Linear(embed_dim, embed_dim, bias=True, dtype=dtype, device=device)

    def forward(self, x, mask=None, optimized_attention=None):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        out = optimized_attention(q, k, v, self.heads, mask)
        return self.out_proj(out)


ACTIVATIONS = {'quick_gelu': lambda a: a * torch.sigmoid(1.702 * a), 'gelu': torch.nn.functional.gelu}


class CLIPMLP(torch.nn.Module):

    def __init__(self, embed_dim, intermediate_size, activation, dtype, device, operations):
        super().__init__()
        self.fc1 = operations.Linear(embed_dim, intermediate_size, bias=True, dtype=dtype, device=device)
        self.activation = ACTIVATIONS[activation]
        self.fc2 = operations.Linear(intermediate_size, embed_dim, bias=True, dtype=dtype, device=device)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x


class CLIPLayer(torch.nn.Module):

    def __init__(self, embed_dim, heads, intermediate_size, intermediate_activation, dtype, device, operations):
        super().__init__()
        self.layer_norm1 = operations.LayerNorm(embed_dim, dtype=dtype, device=device)
        self.self_attn = CLIPAttention(embed_dim, heads, dtype, device, operations)
        self.layer_norm2 = operations.LayerNorm(embed_dim, dtype=dtype, device=device)
        self.mlp = CLIPMLP(embed_dim, intermediate_size, intermediate_activation, dtype, device, operations)

    def forward(self, x, mask=None, optimized_attention=None):
        x += self.self_attn(self.layer_norm1(x), mask, optimized_attention)
        x += self.mlp(self.layer_norm2(x))
        return x


def attention_pytorch(q, k, v, heads, mask=None, attn_precision=None, skip_reshape=False):
    if skip_reshape:
        b, _, _, dim_head = q.shape
    else:
        b, _, dim_head = q.shape
        dim_head //= heads
        q, k, v = map(lambda t: t.view(b, -1, heads, dim_head).transpose(1, 2), (q, k, v))
    if SDP_BATCH_LIMIT >= q.shape[0]:
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False)
        out = out.transpose(1, 2).reshape(b, -1, heads * dim_head)
    else:
        out = torch.empty((q.shape[0], q.shape[2], heads * dim_head), dtype=q.dtype, layout=q.layout, device=q.device)
        for i in range(0, q.shape[0], SDP_BATCH_LIMIT):
            out[i:i + SDP_BATCH_LIMIT] = torch.nn.functional.scaled_dot_product_attention(q[i:i + SDP_BATCH_LIMIT], k[i:i + SDP_BATCH_LIMIT], v[i:i + SDP_BATCH_LIMIT], attn_mask=mask, dropout_p=0.0, is_causal=False).transpose(1, 2).reshape(-1, q.shape[2], heads * dim_head)
    return out


def _get_attention_scores_no_kv_chunking(query: 'Tensor', key_t: 'Tensor', value: 'Tensor', scale: 'float', upcast_attention: 'bool', mask) ->Tensor:
    if upcast_attention:
        with torch.autocast(enabled=False, device_type='cuda'):
            query = query.float()
            key_t = key_t.float()
            attn_scores = torch.baddbmm(torch.empty(1, 1, 1, device=query.device, dtype=query.dtype), query, key_t, alpha=scale, beta=0)
    else:
        attn_scores = torch.baddbmm(torch.empty(1, 1, 1, device=query.device, dtype=query.dtype), query, key_t, alpha=scale, beta=0)
    if mask is not None:
        attn_scores += mask
    try:
        attn_probs = attn_scores.softmax(dim=-1)
        del attn_scores
    except model_management.OOM_EXCEPTION:
        logging.warning('ran out of memory while running softmax in  _get_attention_scores_no_kv_chunking, trying slower in place softmax instead')
        attn_scores -= attn_scores.max(dim=-1, keepdim=True).values
        torch.exp(attn_scores, out=attn_scores)
        summed = torch.sum(attn_scores, dim=-1, keepdim=True)
        attn_scores /= summed
        attn_probs = attn_scores
    hidden_states_slice = torch.bmm(attn_probs, value)
    return hidden_states_slice


class AttnChunk(NamedTuple):
    exp_values: 'Tensor'
    exp_weights_sum: 'Tensor'
    max_score: 'Tensor'


def dynamic_slice(x: 'Tensor', starts: 'List[int]', sizes: 'List[int]') ->Tensor:
    slicing = [slice(start, start + size) for start, size in zip(starts, sizes)]
    return x[slicing]


def _query_chunk_attention(query: 'Tensor', key_t: 'Tensor', value: 'Tensor', summarize_chunk: 'SummarizeChunk', kv_chunk_size: 'int', mask) ->Tensor:
    batch_x_heads, k_channels_per_head, k_tokens = key_t.shape
    _, _, v_channels_per_head = value.shape

    def chunk_scanner(chunk_idx: 'int', mask) ->AttnChunk:
        key_chunk = dynamic_slice(key_t, (0, 0, chunk_idx), (batch_x_heads, k_channels_per_head, kv_chunk_size))
        value_chunk = dynamic_slice(value, (0, chunk_idx, 0), (batch_x_heads, kv_chunk_size, v_channels_per_head))
        if mask is not None:
            mask = mask[:, :, chunk_idx:chunk_idx + kv_chunk_size]
        return summarize_chunk(query, key_chunk, value_chunk, mask=mask)
    chunks: 'List[AttnChunk]' = [chunk_scanner(chunk, mask) for chunk in torch.arange(0, k_tokens, kv_chunk_size)]
    acc_chunk = AttnChunk(*map(torch.stack, zip(*chunks)))
    chunk_values, chunk_weights, chunk_max = acc_chunk
    global_max, _ = torch.max(chunk_max, 0, keepdim=True)
    max_diffs = torch.exp(chunk_max - global_max)
    chunk_values *= torch.unsqueeze(max_diffs, -1)
    chunk_weights *= max_diffs
    all_values = chunk_values.sum(dim=0)
    all_weights = torch.unsqueeze(chunk_weights, -1).sum(dim=0)
    return all_values / all_weights


def _summarize_chunk(query: 'Tensor', key_t: 'Tensor', value: 'Tensor', scale: 'float', upcast_attention: 'bool', mask) ->AttnChunk:
    if upcast_attention:
        with torch.autocast(enabled=False, device_type='cuda'):
            query = query.float()
            key_t = key_t.float()
            attn_weights = torch.baddbmm(torch.empty(1, 1, 1, device=query.device, dtype=query.dtype), query, key_t, alpha=scale, beta=0)
    else:
        attn_weights = torch.baddbmm(torch.empty(1, 1, 1, device=query.device, dtype=query.dtype), query, key_t, alpha=scale, beta=0)
    max_score, _ = torch.max(attn_weights, -1, keepdim=True)
    max_score = max_score.detach()
    attn_weights -= max_score
    if mask is not None:
        attn_weights += mask
    torch.exp(attn_weights, out=attn_weights)
    exp_weights = attn_weights
    exp_values = torch.bmm(exp_weights, value)
    max_score = max_score.squeeze(-1)
    return AttnChunk(exp_values, exp_weights.sum(dim=-1), max_score)


def efficient_dot_product_attention(query: 'Tensor', key_t: 'Tensor', value: 'Tensor', query_chunk_size=1024, kv_chunk_size: 'Optional[int]'=None, kv_chunk_size_min: 'Optional[int]'=None, use_checkpoint=True, upcast_attention=False, mask=None):
    """Computes efficient dot-product attention given query, transposed key, and value.
      This is efficient version of attention presented in
      https://arxiv.org/abs/2112.05682v2 which comes with O(sqrt(n)) memory requirements.
      Args:
        query: queries for calculating attention with shape of
          `[batch * num_heads, tokens, channels_per_head]`.
        key_t: keys for calculating attention with shape of
          `[batch * num_heads, channels_per_head, tokens]`.
        value: values to be used in attention with shape of
          `[batch * num_heads, tokens, channels_per_head]`.
        query_chunk_size: int: query chunks size
        kv_chunk_size: Optional[int]: key/value chunks size. if None: defaults to sqrt(key_tokens)
        kv_chunk_size_min: Optional[int]: key/value minimum chunk size. only considered when kv_chunk_size is None. changes `sqrt(key_tokens)` into `max(sqrt(key_tokens), kv_chunk_size_min)`, to ensure our chunk sizes don't get too small (smaller chunks = more chunks = less concurrent work done).
        use_checkpoint: bool: whether to use checkpointing (recommended True for training, False for inference)
      Returns:
        Output of shape `[batch * num_heads, query_tokens, channels_per_head]`.
      """
    batch_x_heads, q_tokens, q_channels_per_head = query.shape
    _, _, k_tokens = key_t.shape
    scale = q_channels_per_head ** -0.5
    kv_chunk_size = min(kv_chunk_size or int(math.sqrt(k_tokens)), k_tokens)
    if kv_chunk_size_min is not None:
        kv_chunk_size = max(kv_chunk_size, kv_chunk_size_min)
    if mask is not None and len(mask.shape) == 2:
        mask = mask.unsqueeze(0)

    def get_query_chunk(chunk_idx: 'int') ->Tensor:
        return dynamic_slice(query, (0, chunk_idx, 0), (batch_x_heads, min(query_chunk_size, q_tokens), q_channels_per_head))

    def get_mask_chunk(chunk_idx: 'int') ->Tensor:
        if mask is None:
            return None
        chunk = min(query_chunk_size, q_tokens)
        return mask[:, chunk_idx:chunk_idx + chunk]
    summarize_chunk: 'SummarizeChunk' = partial(_summarize_chunk, scale=scale, upcast_attention=upcast_attention)
    summarize_chunk: 'SummarizeChunk' = partial(checkpoint, summarize_chunk) if use_checkpoint else summarize_chunk
    compute_query_chunk_attn: 'ComputeQueryChunkAttn' = partial(_get_attention_scores_no_kv_chunking, scale=scale, upcast_attention=upcast_attention) if k_tokens <= kv_chunk_size else partial(_query_chunk_attention, kv_chunk_size=kv_chunk_size, summarize_chunk=summarize_chunk)
    if q_tokens <= query_chunk_size:
        return compute_query_chunk_attn(query=query, key_t=key_t, value=value, mask=mask)
    res = torch.cat([compute_query_chunk_attn(query=get_query_chunk(i * query_chunk_size), key_t=key_t, value=value, mask=get_mask_chunk(i * query_chunk_size)) for i in range(math.ceil(q_tokens / query_chunk_size))], dim=1)
    return res


def attention_sub_quad(query, key, value, heads, mask=None, attn_precision=None, skip_reshape=False):
    attn_precision = get_attn_precision(attn_precision)
    if skip_reshape:
        b, _, _, dim_head = query.shape
    else:
        b, _, dim_head = query.shape
        dim_head //= heads
    scale = dim_head ** -0.5
    if skip_reshape:
        query = query.reshape(b * heads, -1, dim_head)
        value = value.reshape(b * heads, -1, dim_head)
        key = key.reshape(b * heads, -1, dim_head).movedim(1, 2)
    else:
        query = query.unsqueeze(3).reshape(b, -1, heads, dim_head).permute(0, 2, 1, 3).reshape(b * heads, -1, dim_head)
        value = value.unsqueeze(3).reshape(b, -1, heads, dim_head).permute(0, 2, 1, 3).reshape(b * heads, -1, dim_head)
        key = key.unsqueeze(3).reshape(b, -1, heads, dim_head).permute(0, 2, 3, 1).reshape(b * heads, dim_head, -1)
    dtype = query.dtype
    upcast_attention = attn_precision == torch.float32 and query.dtype != torch.float32
    if upcast_attention:
        bytes_per_token = torch.finfo(torch.float32).bits // 8
    else:
        bytes_per_token = torch.finfo(query.dtype).bits // 8
    batch_x_heads, q_tokens, _ = query.shape
    _, _, k_tokens = key.shape
    qk_matmul_size_bytes = batch_x_heads * bytes_per_token * q_tokens * k_tokens
    mem_free_total, mem_free_torch = model_management.get_free_memory(query.device, True)
    kv_chunk_size_min = None
    kv_chunk_size = None
    query_chunk_size = None
    for x in [4096, 2048, 1024, 512, 256]:
        count = mem_free_total / (batch_x_heads * bytes_per_token * x * 4.0)
        if count >= k_tokens:
            kv_chunk_size = k_tokens
            query_chunk_size = x
            break
    if query_chunk_size is None:
        query_chunk_size = 512
    if mask is not None:
        if len(mask.shape) == 2:
            bs = 1
        else:
            bs = mask.shape[0]
        mask = mask.reshape(bs, -1, mask.shape[-2], mask.shape[-1]).expand(b, heads, -1, -1).reshape(-1, mask.shape[-2], mask.shape[-1])
    hidden_states = efficient_dot_product_attention(query, key, value, query_chunk_size=query_chunk_size, kv_chunk_size=kv_chunk_size, kv_chunk_size_min=kv_chunk_size_min, use_checkpoint=False, upcast_attention=upcast_attention, mask=mask)
    hidden_states = hidden_states
    hidden_states = hidden_states.unflatten(0, (-1, heads)).transpose(1, 2).flatten(start_dim=2)
    return hidden_states


optimized_attention_masked = optimized_attention


def optimized_attention_for_device(device, mask=False, small_input=False):
    if small_input:
        if model_management.pytorch_attention_enabled():
            return attention_pytorch
        else:
            return attention_basic
    if device == torch.device('cpu'):
        return attention_sub_quad
    if mask:
        return optimized_attention_masked
    return optimized_attention


class CLIPEncoder(torch.nn.Module):

    def __init__(self, num_layers, embed_dim, heads, intermediate_size, intermediate_activation, dtype, device, operations):
        super().__init__()
        self.layers = torch.nn.ModuleList([CLIPLayer(embed_dim, heads, intermediate_size, intermediate_activation, dtype, device, operations) for i in range(num_layers)])

    def forward(self, x, mask=None, intermediate_output=None):
        optimized_attention = optimized_attention_for_device(x.device, mask=mask is not None, small_input=True)
        if intermediate_output is not None:
            if intermediate_output < 0:
                intermediate_output = len(self.layers) + intermediate_output
        intermediate = None
        for i, l in enumerate(self.layers):
            x = l(x, mask, optimized_attention)
            if i == intermediate_output:
                intermediate = x.clone()
        return x, intermediate


class CLIPEmbeddings(torch.nn.Module):

    def __init__(self, embed_dim, vocab_size=49408, num_positions=77, dtype=None, device=None, operations=None):
        super().__init__()
        self.token_embedding = operations.Embedding(vocab_size, embed_dim, dtype=dtype, device=device)
        self.position_embedding = operations.Embedding(num_positions, embed_dim, dtype=dtype, device=device)

    def forward(self, input_tokens, dtype=torch.float32):
        return self.token_embedding(input_tokens, out_dtype=dtype) + comfy.ops.cast_to(self.position_embedding.weight, dtype=dtype, device=input_tokens.device)


class CLIPTextModel_(torch.nn.Module):

    def __init__(self, config_dict, dtype, device, operations):
        num_layers = config_dict['num_hidden_layers']
        embed_dim = config_dict['hidden_size']
        heads = config_dict['num_attention_heads']
        intermediate_size = config_dict['intermediate_size']
        intermediate_activation = config_dict['hidden_act']
        num_positions = config_dict['max_position_embeddings']
        self.eos_token_id = config_dict['eos_token_id']
        super().__init__()
        self.embeddings = CLIPEmbeddings(embed_dim, num_positions=num_positions, dtype=dtype, device=device, operations=operations)
        self.encoder = CLIPEncoder(num_layers, embed_dim, heads, intermediate_size, intermediate_activation, dtype, device, operations)
        self.final_layer_norm = operations.LayerNorm(embed_dim, dtype=dtype, device=device)

    def forward(self, input_tokens, attention_mask=None, intermediate_output=None, final_layer_norm_intermediate=True, dtype=torch.float32):
        x = self.embeddings(input_tokens, dtype=dtype)
        mask = None
        if attention_mask is not None:
            mask = 1.0 - attention_mask.reshape((attention_mask.shape[0], 1, -1, attention_mask.shape[-1])).expand(attention_mask.shape[0], 1, attention_mask.shape[-1], attention_mask.shape[-1])
            mask = mask.masked_fill(mask, float('-inf'))
        causal_mask = torch.empty(x.shape[1], x.shape[1], dtype=x.dtype, device=x.device).fill_(float('-inf')).triu_(1)
        if mask is not None:
            mask += causal_mask
        else:
            mask = causal_mask
        x, i = self.encoder(x, mask=mask, intermediate_output=intermediate_output)
        x = self.final_layer_norm(x)
        if i is not None and final_layer_norm_intermediate:
            i = self.final_layer_norm(i)
        pooled_output = x[torch.arange(x.shape[0], device=x.device), (torch.round(input_tokens) == self.eos_token_id).int().argmax(dim=-1)]
        return x, i, pooled_output


class CLIPTextModel(torch.nn.Module):

    def __init__(self, config_dict, dtype, device, operations):
        super().__init__()
        self.num_layers = config_dict['num_hidden_layers']
        self.text_model = CLIPTextModel_(config_dict, dtype, device, operations)
        embed_dim = config_dict['hidden_size']
        self.text_projection = operations.Linear(embed_dim, embed_dim, bias=False, dtype=dtype, device=device)
        self.dtype = dtype

    def get_input_embeddings(self):
        return self.text_model.embeddings.token_embedding

    def set_input_embeddings(self, embeddings):
        self.text_model.embeddings.token_embedding = embeddings

    def forward(self, *args, **kwargs):
        x = self.text_model(*args, **kwargs)
        out = self.text_projection(x[2])
        return x[0], x[1], out, x[2]


class CLIPVisionEmbeddings(torch.nn.Module):

    def __init__(self, embed_dim, num_channels=3, patch_size=14, image_size=224, dtype=None, device=None, operations=None):
        super().__init__()
        self.class_embedding = torch.nn.Parameter(torch.empty(embed_dim, dtype=dtype, device=device))
        self.patch_embedding = operations.Conv2d(in_channels=num_channels, out_channels=embed_dim, kernel_size=patch_size, stride=patch_size, bias=False, dtype=dtype, device=device)
        num_patches = (image_size // patch_size) ** 2
        num_positions = num_patches + 1
        self.position_embedding = operations.Embedding(num_positions, embed_dim, dtype=dtype, device=device)

    def forward(self, pixel_values):
        embeds = self.patch_embedding(pixel_values).flatten(2).transpose(1, 2)
        return torch.cat([comfy.ops.cast_to_input(self.class_embedding, embeds).expand(pixel_values.shape[0], 1, -1), embeds], dim=1) + comfy.ops.cast_to_input(self.position_embedding.weight, embeds)


class CLIPVision(torch.nn.Module):

    def __init__(self, config_dict, dtype, device, operations):
        super().__init__()
        num_layers = config_dict['num_hidden_layers']
        embed_dim = config_dict['hidden_size']
        heads = config_dict['num_attention_heads']
        intermediate_size = config_dict['intermediate_size']
        intermediate_activation = config_dict['hidden_act']
        self.embeddings = CLIPVisionEmbeddings(embed_dim, config_dict['num_channels'], config_dict['patch_size'], config_dict['image_size'], dtype=dtype, device=device, operations=operations)
        self.pre_layrnorm = operations.LayerNorm(embed_dim)
        self.encoder = CLIPEncoder(num_layers, embed_dim, heads, intermediate_size, intermediate_activation, dtype, device, operations)
        self.post_layernorm = operations.LayerNorm(embed_dim)

    def forward(self, pixel_values, attention_mask=None, intermediate_output=None):
        x = self.embeddings(pixel_values)
        x = self.pre_layrnorm(x)
        x, i = self.encoder(x, mask=None, intermediate_output=intermediate_output)
        pooled_output = self.post_layernorm(x[:, 0, :])
        return x, i, pooled_output


class CLIPVisionModelProjection(torch.nn.Module):

    def __init__(self, config_dict, dtype, device, operations):
        super().__init__()
        self.vision_model = CLIPVision(config_dict, dtype, device, operations)
        self.visual_projection = operations.Linear(config_dict['hidden_size'], config_dict['projection_dim'], bias=False)

    def forward(self, *args, **kwargs):
        x = self.vision_model(*args, **kwargs)
        out = self.visual_projection(x[2])
        return x[0], x[1], out


class StrengthType(Enum):
    CONSTANT = 1
    LINEAR_UP = 2


class ControlBase:

    def __init__(self):
        self.cond_hint_original = None
        self.cond_hint = None
        self.strength = 1.0
        self.timestep_percent_range = 0.0, 1.0
        self.latent_format = None
        self.vae = None
        self.global_average_pooling = False
        self.timestep_range = None
        self.compression_ratio = 8
        self.upscale_algorithm = 'nearest-exact'
        self.extra_args = {}
        self.previous_controlnet = None
        self.extra_conds = []
        self.strength_type = StrengthType.CONSTANT
        self.concat_mask = False
        self.extra_concat_orig = []
        self.extra_concat = None

    def set_cond_hint(self, cond_hint, strength=1.0, timestep_percent_range=(0.0, 1.0), vae=None, extra_concat=[]):
        self.cond_hint_original = cond_hint
        self.strength = strength
        self.timestep_percent_range = timestep_percent_range
        if self.latent_format is not None:
            if vae is None:
                logging.warning('WARNING: no VAE provided to the controlnet apply node when this controlnet requires one.')
            self.vae = vae
        self.extra_concat_orig = extra_concat.copy()
        if self.concat_mask and len(self.extra_concat_orig) == 0:
            self.extra_concat_orig.append(torch.tensor([[[[1.0]]]]))
        return self

    def pre_run(self, model, percent_to_timestep_function):
        self.timestep_range = percent_to_timestep_function(self.timestep_percent_range[0]), percent_to_timestep_function(self.timestep_percent_range[1])
        if self.previous_controlnet is not None:
            self.previous_controlnet.pre_run(model, percent_to_timestep_function)

    def set_previous_controlnet(self, controlnet):
        self.previous_controlnet = controlnet
        return self

    def cleanup(self):
        if self.previous_controlnet is not None:
            self.previous_controlnet.cleanup()
        self.cond_hint = None
        self.extra_concat = None
        self.timestep_range = None

    def get_models(self):
        out = []
        if self.previous_controlnet is not None:
            out += self.previous_controlnet.get_models()
        return out

    def copy_to(self, c):
        c.cond_hint_original = self.cond_hint_original
        c.strength = self.strength
        c.timestep_percent_range = self.timestep_percent_range
        c.global_average_pooling = self.global_average_pooling
        c.compression_ratio = self.compression_ratio
        c.upscale_algorithm = self.upscale_algorithm
        c.latent_format = self.latent_format
        c.extra_args = self.extra_args.copy()
        c.vae = self.vae
        c.extra_conds = self.extra_conds.copy()
        c.strength_type = self.strength_type
        c.concat_mask = self.concat_mask
        c.extra_concat_orig = self.extra_concat_orig.copy()

    def inference_memory_requirements(self, dtype):
        if self.previous_controlnet is not None:
            return self.previous_controlnet.inference_memory_requirements(dtype)
        return 0

    def control_merge(self, control, control_prev, output_dtype):
        out = {'input': [], 'middle': [], 'output': []}
        for key in control:
            control_output = control[key]
            applied_to = set()
            for i in range(len(control_output)):
                x = control_output[i]
                if x is not None:
                    if self.global_average_pooling:
                        x = torch.mean(x, dim=(2, 3), keepdim=True).repeat(1, 1, x.shape[2], x.shape[3])
                    if x not in applied_to:
                        applied_to.add(x)
                        if self.strength_type == StrengthType.CONSTANT:
                            x *= self.strength
                        elif self.strength_type == StrengthType.LINEAR_UP:
                            x *= self.strength ** float(len(control_output) - i)
                    if output_dtype is not None and x.dtype != output_dtype:
                        x = x
                out[key].append(x)
        if control_prev is not None:
            for x in ['input', 'middle', 'output']:
                o = out[x]
                for i in range(len(control_prev[x])):
                    prev_val = control_prev[x][i]
                    if i >= len(o):
                        o.append(prev_val)
                    elif prev_val is not None:
                        if o[i] is None:
                            o[i] = prev_val
                        elif o[i].shape[0] < prev_val.shape[0]:
                            o[i] = prev_val + o[i]
                        else:
                            o[i] = prev_val + o[i]
        return out

    def set_extra_arg(self, argument, value=None):
        self.extra_args[argument] = value


class ControlLora(ControlNet):

    def __init__(self, control_weights, global_average_pooling=False, model_options={}):
        ControlBase.__init__(self)
        self.control_weights = control_weights
        self.global_average_pooling = global_average_pooling
        self.extra_conds += ['y']

    def pre_run(self, model, percent_to_timestep_function):
        super().pre_run(model, percent_to_timestep_function)
        controlnet_config = model.model_config.unet_config.copy()
        controlnet_config.pop('out_channels')
        controlnet_config['hint_channels'] = self.control_weights['input_hint_block.0.weight'].shape[1]
        self.manual_cast_dtype = model.manual_cast_dtype
        dtype = model.get_dtype()
        if self.manual_cast_dtype is None:


            class control_lora_ops(ControlLoraOps, comfy.ops.disable_weight_init):
                pass
        else:


            class control_lora_ops(ControlLoraOps, comfy.ops.manual_cast):
                pass
            dtype = self.manual_cast_dtype
        controlnet_config['operations'] = control_lora_ops
        controlnet_config['dtype'] = dtype
        self.control_model = comfy.cldm.cldm.ControlNet(**controlnet_config)
        self.control_model
        diffusion_model = model.diffusion_model
        sd = diffusion_model.state_dict()
        cm = self.control_model.state_dict()
        for k in sd:
            weight = sd[k]
            try:
                comfy.utils.set_attr_param(self.control_model, k, weight)
            except:
                pass
        for k in self.control_weights:
            if k not in {'lora_controlnet'}:
                comfy.utils.set_attr_param(self.control_model, k, self.control_weights[k].to(dtype))

    def copy(self):
        c = ControlLora(self.control_weights, global_average_pooling=self.global_average_pooling)
        self.copy_to(c)
        return c

    def cleanup(self):
        del self.control_model
        self.control_model = None
        super().cleanup()

    def get_models(self):
        out = ControlBase.get_models(self)
        return out

    def inference_memory_requirements(self, dtype):
        return comfy.utils.calculate_parameters(self.control_weights) * comfy.model_management.dtype_size(dtype) + ControlBase.inference_memory_requirements(self, dtype)


class GEGLU(nn.Module):

    def __init__(self, dim_in, dim_out, dtype=None, device=None, operations=ops):
        super().__init__()
        self.proj = operations.Linear(dim_in, dim_out * 2, dtype=dtype, device=device)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


class FeedForward(nn.Module):

    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.0, dtype=None, device=None, operations=ops):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(operations.Linear(dim, inner_dim, dtype=dtype, device=device), nn.GELU()) if not glu else GEGLU(dim, inner_dim, dtype=dtype, device=device, operations=operations)
        self.net = nn.Sequential(project_in, nn.Dropout(dropout), operations.Linear(inner_dim, dim_out, dtype=dtype, device=device))

    def forward(self, x):
        return self.net(x)


class CrossAttention(nn.Module):

    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0, attn_precision=None, dtype=None, device=None, operations=ops):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)
        self.attn_precision = attn_precision
        self.heads = heads
        self.dim_head = dim_head
        self.to_q = operations.Linear(query_dim, inner_dim, bias=False, dtype=dtype, device=device)
        self.to_k = operations.Linear(context_dim, inner_dim, bias=False, dtype=dtype, device=device)
        self.to_v = operations.Linear(context_dim, inner_dim, bias=False, dtype=dtype, device=device)
        self.to_out = nn.Sequential(operations.Linear(inner_dim, query_dim, dtype=dtype, device=device), nn.Dropout(dropout))

    def forward(self, x, context=None, value=None, mask=None):
        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        if value is not None:
            v = self.to_v(value)
            del value
        else:
            v = self.to_v(context)
        if mask is None:
            out = optimized_attention(q, k, v, self.heads, attn_precision=self.attn_precision)
        else:
            out = optimized_attention_masked(q, k, v, self.heads, mask, attn_precision=self.attn_precision)
        return self.to_out(out)


class GatedCrossAttentionDense(nn.Module):

    def __init__(self, query_dim, context_dim, n_heads, d_head):
        super().__init__()
        self.attn = CrossAttention(query_dim=query_dim, context_dim=context_dim, heads=n_heads, dim_head=d_head, operations=ops)
        self.ff = FeedForward(query_dim, glu=True)
        self.norm1 = ops.LayerNorm(query_dim)
        self.norm2 = ops.LayerNorm(query_dim)
        self.register_parameter('alpha_attn', nn.Parameter(torch.tensor(0.0)))
        self.register_parameter('alpha_dense', nn.Parameter(torch.tensor(0.0)))
        self.scale = 1

    def forward(self, x, objs):
        x = x + self.scale * torch.tanh(self.alpha_attn) * self.attn(self.norm1(x), objs, objs)
        x = x + self.scale * torch.tanh(self.alpha_dense) * self.ff(self.norm2(x))
        return x


class GatedSelfAttentionDense(nn.Module):

    def __init__(self, query_dim, context_dim, n_heads, d_head):
        super().__init__()
        self.linear = ops.Linear(context_dim, query_dim)
        self.attn = CrossAttention(query_dim=query_dim, context_dim=query_dim, heads=n_heads, dim_head=d_head, operations=ops)
        self.ff = FeedForward(query_dim, glu=True)
        self.norm1 = ops.LayerNorm(query_dim)
        self.norm2 = ops.LayerNorm(query_dim)
        self.register_parameter('alpha_attn', nn.Parameter(torch.tensor(0.0)))
        self.register_parameter('alpha_dense', nn.Parameter(torch.tensor(0.0)))
        self.scale = 1

    def forward(self, x, objs):
        N_visual = x.shape[1]
        objs = self.linear(objs)
        x = x + self.scale * torch.tanh(self.alpha_attn) * self.attn(self.norm1(torch.cat([x, objs], dim=1)))[:, 0:N_visual, :]
        x = x + self.scale * torch.tanh(self.alpha_dense) * self.ff(self.norm2(x))
        return x


class GatedSelfAttentionDense2(nn.Module):

    def __init__(self, query_dim, context_dim, n_heads, d_head):
        super().__init__()
        self.linear = ops.Linear(context_dim, query_dim)
        self.attn = CrossAttention(query_dim=query_dim, context_dim=query_dim, dim_head=d_head, operations=ops)
        self.ff = FeedForward(query_dim, glu=True)
        self.norm1 = ops.LayerNorm(query_dim)
        self.norm2 = ops.LayerNorm(query_dim)
        self.register_parameter('alpha_attn', nn.Parameter(torch.tensor(0.0)))
        self.register_parameter('alpha_dense', nn.Parameter(torch.tensor(0.0)))
        self.scale = 1

    def forward(self, x, objs):
        B, N_visual, _ = x.shape
        B, N_ground, _ = objs.shape
        objs = self.linear(objs)
        size_v = math.sqrt(N_visual)
        size_g = math.sqrt(N_ground)
        assert int(size_v) == size_v, 'Visual tokens must be square rootable'
        assert int(size_g) == size_g, 'Grounding tokens must be square rootable'
        size_v = int(size_v)
        size_g = int(size_g)
        out = self.attn(self.norm1(torch.cat([x, objs], dim=1)))[:, N_visual:, :]
        out = out.permute(0, 2, 1).reshape(B, -1, size_g, size_g)
        out = torch.nn.functional.interpolate(out, (size_v, size_v), mode='bicubic')
        residual = out.reshape(B, -1, N_visual).permute(0, 2, 1)
        x = x + self.scale * torch.tanh(self.alpha_attn) * residual
        x = x + self.scale * torch.tanh(self.alpha_dense) * self.ff(self.norm2(x))
        return x


class FourierEmbedder:

    def __init__(self, num_freqs=64, temperature=100):
        self.num_freqs = num_freqs
        self.temperature = temperature
        self.freq_bands = temperature ** (torch.arange(num_freqs) / num_freqs)

    @torch.no_grad()
    def __call__(self, x, cat_dim=-1):
        """x: arbitrary shape of tensor. dim: cat dim"""
        out = []
        for freq in self.freq_bands:
            out.append(torch.sin(freq * x))
            out.append(torch.cos(freq * x))
        return torch.cat(out, cat_dim)


class PositionNet(nn.Module):

    def __init__(self, in_dim, out_dim, fourier_freqs=8):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.fourier_embedder = FourierEmbedder(num_freqs=fourier_freqs)
        self.position_dim = fourier_freqs * 2 * 4
        self.linears = nn.Sequential(ops.Linear(self.in_dim + self.position_dim, 512), nn.SiLU(), ops.Linear(512, 512), nn.SiLU(), ops.Linear(512, out_dim))
        self.null_positive_feature = torch.nn.Parameter(torch.zeros([self.in_dim]))
        self.null_position_feature = torch.nn.Parameter(torch.zeros([self.position_dim]))

    def forward(self, boxes, masks, positive_embeddings):
        B, N, _ = boxes.shape
        masks = masks.unsqueeze(-1)
        positive_embeddings = positive_embeddings
        xyxy_embedding = self.fourier_embedder(boxes)
        positive_null = self.null_positive_feature.view(1, 1, -1)
        xyxy_null = self.null_position_feature.view(1, 1, -1)
        positive_embeddings = positive_embeddings * masks + (1 - masks) * positive_null
        xyxy_embedding = xyxy_embedding * masks + (1 - masks) * xyxy_null
        objs = self.linears(torch.cat([positive_embeddings, xyxy_embedding], dim=-1))
        assert objs.shape == torch.Size([B, N, self.out_dim])
        return objs


class Gligen(nn.Module):

    def __init__(self, modules, position_net, key_dim):
        super().__init__()
        self.module_list = nn.ModuleList(modules)
        self.position_net = position_net
        self.key_dim = key_dim
        self.max_objs = 30
        self.current_device = torch.device('cpu')

    def _set_position(self, boxes, masks, positive_embeddings):
        objs = self.position_net(boxes, masks, positive_embeddings)

        def func(x, extra_options):
            key = extra_options['transformer_index']
            module = self.module_list[key]
            return module(x, objs)
        return func

    def set_position(self, latent_image_shape, position_params, device):
        batch, c, h, w = latent_image_shape
        masks = torch.zeros([self.max_objs], device='cpu')
        boxes = []
        positive_embeddings = []
        for p in position_params:
            x1 = p[4] / w
            y1 = p[3] / h
            x2 = (p[4] + p[2]) / w
            y2 = (p[3] + p[1]) / h
            masks[len(boxes)] = 1.0
            boxes += [torch.tensor((x1, y1, x2, y2)).unsqueeze(0)]
            positive_embeddings += [p[0]]
        append_boxes = []
        append_conds = []
        if len(boxes) < self.max_objs:
            append_boxes = [torch.zeros([self.max_objs - len(boxes), 4], device='cpu')]
            append_conds = [torch.zeros([self.max_objs - len(boxes), self.key_dim], device='cpu')]
        box_out = torch.cat(boxes + append_boxes).unsqueeze(0).repeat(batch, 1, 1)
        masks = masks.unsqueeze(0).repeat(batch, 1)
        conds = torch.cat(positive_embeddings + append_conds).unsqueeze(0).repeat(batch, 1, 1)
        return self._set_position(box_out, masks, conds)

    def set_empty(self, latent_image_shape, device):
        batch, c, h, w = latent_image_shape
        masks = torch.zeros([self.max_objs], device='cpu').repeat(batch, 1)
        box_out = torch.zeros([self.max_objs, 4], device='cpu').repeat(batch, 1, 1)
        conds = torch.zeros([self.max_objs, self.key_dim], device='cpu').repeat(batch, 1, 1)
        return self._set_position(box_out, masks, conds)


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


def vae_sample(mean, scale):
    stdev = nn.functional.softplus(scale) + 0.0001
    var = stdev * stdev
    logvar = torch.log(var)
    latents = torch.randn_like(mean) * stdev + mean
    kl = (mean * mean + var - logvar - 1).sum(1).mean()
    return latents, kl


class VAEBottleneck(nn.Module):

    def __init__(self):
        super().__init__()
        self.is_discrete = False

    def encode(self, x, return_info=False, **kwargs):
        info = {}
        mean, scale = x.chunk(2, dim=1)
        x, kl = vae_sample(mean, scale)
        info['kl'] = kl
        if return_info:
            return x, info
        else:
            return x

    def decode(self, x):
        return x


def snake_beta(x, alpha, beta):
    return x + 1.0 / (beta + 1e-09) * pow(torch.sin(x * alpha), 2)


class SnakeBeta(nn.Module):

    def __init__(self, in_features, alpha=1.0, alpha_trainable=True, alpha_logscale=True):
        super(SnakeBeta, self).__init__()
        self.in_features = in_features
        self.alpha_logscale = alpha_logscale
        if self.alpha_logscale:
            self.alpha = nn.Parameter(torch.zeros(in_features) * alpha)
            self.beta = nn.Parameter(torch.zeros(in_features) * alpha)
        else:
            self.alpha = nn.Parameter(torch.ones(in_features) * alpha)
            self.beta = nn.Parameter(torch.ones(in_features) * alpha)
        self.no_div_by_zero = 1e-09

    def forward(self, x):
        alpha = self.alpha.unsqueeze(0).unsqueeze(-1)
        beta = self.beta.unsqueeze(0).unsqueeze(-1)
        if self.alpha_logscale:
            alpha = torch.exp(alpha)
            beta = torch.exp(beta)
        x = snake_beta(x, alpha, beta)
        return x


def WNConv1d(*args, **kwargs):
    try:
        return torch.nn.utils.parametrizations.weight_norm(ops.Conv1d(*args, **kwargs))
    except:
        return torch.nn.utils.weight_norm(ops.Conv1d(*args, **kwargs))


def get_activation(activation: "Literal['elu', 'snake', 'none']", antialias=False, channels=None) ->nn.Module:
    if activation == 'elu':
        act = torch.nn.ELU()
    elif activation == 'snake':
        act = SnakeBeta(channels)
    elif activation == 'none':
        act = torch.nn.Identity()
    else:
        raise ValueError(f'Unknown activation {activation}')
    if antialias:
        act = Activation1d(act)
    return act


class ResidualUnit(nn.Module):

    def __init__(self, in_channels, out_channels, dilation, use_snake=False, antialias_activation=False):
        super().__init__()
        self.dilation = dilation
        padding = dilation * (7 - 1) // 2
        self.layers = nn.Sequential(get_activation('snake' if use_snake else 'elu', antialias=antialias_activation, channels=out_channels), WNConv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=7, dilation=dilation, padding=padding), get_activation('snake' if use_snake else 'elu', antialias=antialias_activation, channels=out_channels), WNConv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=1))

    def forward(self, x):
        res = x
        x = self.layers(x)
        return x + res


class EncoderBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride, use_snake=False, antialias_activation=False):
        super().__init__()
        self.layers = nn.Sequential(ResidualUnit(in_channels=in_channels, out_channels=in_channels, dilation=1, use_snake=use_snake), ResidualUnit(in_channels=in_channels, out_channels=in_channels, dilation=3, use_snake=use_snake), ResidualUnit(in_channels=in_channels, out_channels=in_channels, dilation=9, use_snake=use_snake), get_activation('snake' if use_snake else 'elu', antialias=antialias_activation, channels=in_channels), WNConv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=2 * stride, stride=stride, padding=math.ceil(stride / 2)))

    def forward(self, x):
        return self.layers(x)


def WNConvTranspose1d(*args, **kwargs):
    try:
        return torch.nn.utils.parametrizations.weight_norm(ops.ConvTranspose1d(*args, **kwargs))
    except:
        return torch.nn.utils.weight_norm(ops.ConvTranspose1d(*args, **kwargs))


class DecoderBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride, use_snake=False, antialias_activation=False, use_nearest_upsample=False):
        super().__init__()
        if use_nearest_upsample:
            upsample_layer = nn.Sequential(nn.Upsample(scale_factor=stride, mode='nearest'), WNConv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=2 * stride, stride=1, bias=False, padding='same'))
        else:
            upsample_layer = WNConvTranspose1d(in_channels=in_channels, out_channels=out_channels, kernel_size=2 * stride, stride=stride, padding=math.ceil(stride / 2))
        self.layers = nn.Sequential(get_activation('snake' if use_snake else 'elu', antialias=antialias_activation, channels=in_channels), upsample_layer, ResidualUnit(in_channels=out_channels, out_channels=out_channels, dilation=1, use_snake=use_snake), ResidualUnit(in_channels=out_channels, out_channels=out_channels, dilation=3, use_snake=use_snake), ResidualUnit(in_channels=out_channels, out_channels=out_channels, dilation=9, use_snake=use_snake))

    def forward(self, x):
        return self.layers(x)


class OobleckEncoder(nn.Module):

    def __init__(self, in_channels=2, channels=128, latent_dim=32, c_mults=[1, 2, 4, 8], strides=[2, 4, 8, 8], use_snake=False, antialias_activation=False):
        super().__init__()
        c_mults = [1] + c_mults
        self.depth = len(c_mults)
        layers = [WNConv1d(in_channels=in_channels, out_channels=c_mults[0] * channels, kernel_size=7, padding=3)]
        for i in range(self.depth - 1):
            layers += [EncoderBlock(in_channels=c_mults[i] * channels, out_channels=c_mults[i + 1] * channels, stride=strides[i], use_snake=use_snake)]
        layers += [get_activation('snake' if use_snake else 'elu', antialias=antialias_activation, channels=c_mults[-1] * channels), WNConv1d(in_channels=c_mults[-1] * channels, out_channels=latent_dim, kernel_size=3, padding=1)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class OobleckDecoder(nn.Module):

    def __init__(self, out_channels=2, channels=128, latent_dim=32, c_mults=[1, 2, 4, 8], strides=[2, 4, 8, 8], use_snake=False, antialias_activation=False, use_nearest_upsample=False, final_tanh=True):
        super().__init__()
        c_mults = [1] + c_mults
        self.depth = len(c_mults)
        layers = [WNConv1d(in_channels=latent_dim, out_channels=c_mults[-1] * channels, kernel_size=7, padding=3)]
        for i in range(self.depth - 1, 0, -1):
            layers += [DecoderBlock(in_channels=c_mults[i] * channels, out_channels=c_mults[i - 1] * channels, stride=strides[i - 1], use_snake=use_snake, antialias_activation=antialias_activation, use_nearest_upsample=use_nearest_upsample)]
        layers += [get_activation('snake' if use_snake else 'elu', antialias=antialias_activation, channels=c_mults[0] * channels), WNConv1d(in_channels=c_mults[0] * channels, out_channels=out_channels, kernel_size=7, padding=3, bias=False), nn.Tanh() if final_tanh else nn.Identity()]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class AudioOobleckVAE(nn.Module):

    def __init__(self, in_channels=2, channels=128, latent_dim=64, c_mults=[1, 2, 4, 8, 16], strides=[2, 4, 4, 8, 8], use_snake=True, antialias_activation=False, use_nearest_upsample=False, final_tanh=False):
        super().__init__()
        self.encoder = OobleckEncoder(in_channels, channels, latent_dim * 2, c_mults, strides, use_snake, antialias_activation)
        self.decoder = OobleckDecoder(in_channels, channels, latent_dim, c_mults, strides, use_snake, antialias_activation, use_nearest_upsample=use_nearest_upsample, final_tanh=final_tanh)
        self.bottleneck = VAEBottleneck()

    def encode(self, x):
        return self.bottleneck.encode(self.encoder(x))

    def decode(self, x):
        return self.decoder(self.bottleneck.decode(x))


def add_fourier_features(inputs: 'torch.Tensor', start=6, stop=8, step=1):
    num_freqs = (stop - start) // step
    assert inputs.ndim == 5
    C = inputs.size(1)
    freqs = torch.arange(start, stop, step, dtype=inputs.dtype, device=inputs.device)
    assert num_freqs == len(freqs)
    w = torch.pow(2.0, freqs) * (2 * torch.pi)
    C = inputs.shape[1]
    w = w.repeat(C)[None, :, None, None, None]
    h = inputs.repeat_interleave(num_freqs, dim=1)
    h = w * h
    return torch.cat([inputs, torch.sin(h), torch.cos(h)], dim=1)


class FourierFeatures(nn.Module):

    def __init__(self, start: 'int'=6, stop: 'int'=8, step: 'int'=1):
        super().__init__()
        self.start = start
        self.stop = stop
        self.step = step

    def forward(self, inputs):
        """Add Fourier features to inputs.

        Args:
            inputs: Input tensor. Shape: [B, C, T, H, W]

        Returns:
            h: Output tensor. Shape: [B, (1 + 2 * num_freqs) * C, T, H, W]
        """
        return add_fourier_features(inputs, self.start, self.stop, self.step)


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: 'torch.Tensor'):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class GLU(nn.Module):

    def __init__(self, dim_in, dim_out, activation, use_conv=False, conv_kernel_size=3, dtype=None, device=None, operations=None):
        super().__init__()
        self.act = activation
        self.proj = operations.Linear(dim_in, dim_out * 2, dtype=dtype, device=device) if not use_conv else operations.Conv1d(dim_in, dim_out * 2, conv_kernel_size, padding=conv_kernel_size // 2, dtype=dtype, device=device)
        self.use_conv = use_conv

    def forward(self, x):
        if self.use_conv:
            x = rearrange(x, 'b n d -> b d n')
            x = self.proj(x)
            x = rearrange(x, 'b d n -> b n d')
        else:
            x = self.proj(x)
        x, gate = x.chunk(2, dim=-1)
        return x * self.act(gate)


class AbsolutePositionalEmbedding(nn.Module):

    def __init__(self, dim, max_seq_len):
        super().__init__()
        self.scale = dim ** -0.5
        self.max_seq_len = max_seq_len
        self.emb = nn.Embedding(max_seq_len, dim)

    def forward(self, x, pos=None, seq_start_pos=None):
        seq_len, device = x.shape[1], x.device
        assert seq_len <= self.max_seq_len, f'you are passing in a sequence length of {seq_len} but your absolute positional embedding has a max sequence length of {self.max_seq_len}'
        if pos is None:
            pos = torch.arange(seq_len, device=device)
        if seq_start_pos is not None:
            pos = (pos - seq_start_pos[..., None]).clamp(min=0)
        pos_emb = self.emb(pos)
        pos_emb = pos_emb * self.scale
        return pos_emb


class ScaledSinusoidalEmbedding(nn.Module):

    def __init__(self, dim, theta=10000):
        super().__init__()
        assert dim % 2 == 0, 'dimension must be divisible by 2'
        self.scale = nn.Parameter(torch.ones(1) * dim ** -0.5)
        half_dim = dim // 2
        freq_seq = torch.arange(half_dim).float() / half_dim
        inv_freq = theta ** -freq_seq
        self.register_buffer('inv_freq', inv_freq, persistent=False)

    def forward(self, x, pos=None, seq_start_pos=None):
        seq_len, device = x.shape[1], x.device
        if pos is None:
            pos = torch.arange(seq_len, device=device)
        if seq_start_pos is not None:
            pos = pos - seq_start_pos[..., None]
        emb = torch.einsum('i, j -> i j', pos, self.inv_freq)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb * self.scale


class RotaryEmbedding(nn.Module):

    def __init__(self, dim, use_xpos=False, scale_base=512, interpolation_factor=1.0, base=10000, base_rescale_factor=1.0, dtype=None, device=None):
        super().__init__()
        base *= base_rescale_factor ** (dim / (dim - 2))
        self.register_buffer('inv_freq', torch.empty((dim // 2,), device=device, dtype=dtype))
        assert interpolation_factor >= 1.0
        self.interpolation_factor = interpolation_factor
        if not use_xpos:
            self.register_buffer('scale', None)
            return
        scale = (torch.arange(0, dim, 2) + 0.4 * dim) / (1.4 * dim)
        self.scale_base = scale_base
        self.register_buffer('scale', scale)

    def forward_from_seq_len(self, seq_len, device, dtype):
        t = torch.arange(seq_len, device=device, dtype=dtype)
        return self.forward(t)

    def forward(self, t):
        device = t.device
        dtype = t.dtype
        t = t / self.interpolation_factor
        freqs = torch.einsum('i , j -> i j', t, comfy.ops.cast_to_input(self.inv_freq, t))
        freqs = torch.cat((freqs, freqs), dim=-1)
        if self.scale is None:
            return freqs, 1.0
        power = (torch.arange(seq_len, device=device) - seq_len // 2) / self.scale_base
        scale = comfy.ops.cast_to_input(self.scale, t) ** rearrange(power, 'n -> n 1')
        scale = torch.cat((scale, scale), dim=-1)
        return freqs, scale


def reshape_for_broadcast(freqs_cis: 'Union[torch.Tensor, Tuple[torch.Tensor]]', x: 'torch.Tensor', head_first=False):
    """
    Reshape frequency tensor for broadcasting it with another tensor.

    This function reshapes the frequency tensor to have the same shape as the target tensor 'x'
    for the purpose of broadcasting the frequency tensor during element-wise operations.

    Args:
        freqs_cis (Union[torch.Tensor, Tuple[torch.Tensor]]): Frequency tensor to be reshaped.
        x (torch.Tensor): Target tensor for broadcasting compatibility.
        head_first (bool): head dimension first (except batch dim) or not.

    Returns:
        torch.Tensor: Reshaped frequency tensor.

    Raises:
        AssertionError: If the frequency tensor doesn't match the expected shape.
        AssertionError: If the target tensor 'x' doesn't have the expected number of dimensions.
    """
    ndim = x.ndim
    assert 0 <= 1 < ndim
    if isinstance(freqs_cis, tuple):
        if head_first:
            assert freqs_cis[0].shape == (x.shape[-2], x.shape[-1]), f'freqs_cis shape {freqs_cis[0].shape} does not match x shape {x.shape}'
            shape = [(d if i == ndim - 2 or i == ndim - 1 else 1) for i, d in enumerate(x.shape)]
        else:
            assert freqs_cis[0].shape == (x.shape[1], x.shape[-1]), f'freqs_cis shape {freqs_cis[0].shape} does not match x shape {x.shape}'
            shape = [(d if i == 1 or i == ndim - 1 else 1) for i, d in enumerate(x.shape)]
        return freqs_cis[0].view(*shape), freqs_cis[1].view(*shape)
    else:
        if head_first:
            assert freqs_cis.shape == (x.shape[-2], x.shape[-1]), f'freqs_cis shape {freqs_cis.shape} does not match x shape {x.shape}'
            shape = [(d if i == ndim - 2 or i == ndim - 1 else 1) for i, d in enumerate(x.shape)]
        else:
            assert freqs_cis.shape == (x.shape[1], x.shape[-1]), f'freqs_cis shape {freqs_cis.shape} does not match x shape {x.shape}'
            shape = [(d if i == 1 or i == ndim - 1 else 1) for i, d in enumerate(x.shape)]
        return freqs_cis.view(*shape)


def rotate_half(x):
    x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)
    return torch.stack([-x_imag, x_real], dim=-1).flatten(3)


def apply_rotary_emb(xq: 'torch.Tensor', xk: 'Optional[torch.Tensor]', freqs_cis: 'Union[torch.Tensor, Tuple[torch.Tensor]]', head_first: 'bool'=False) ->Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor.

    This function applies rotary embeddings to the given query 'xq' and key 'xk' tensors using the provided
    frequency tensor 'freqs_cis'. The input tensors are reshaped as complex numbers, and the frequency tensor
    is reshaped for broadcasting compatibility. The resulting tensors contain rotary embeddings and are
    returned as real tensors.

    Args:
        xq (torch.Tensor): Query tensor to apply rotary embeddings. [B, S, H, D]
        xk (torch.Tensor): Key tensor to apply rotary embeddings.   [B, S, H, D]
        freqs_cis (Union[torch.Tensor, Tuple[torch.Tensor]]): Precomputed frequency tensor for complex exponentials.
        head_first (bool): head dimension first (except batch dim) or not.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.

    """
    xk_out = None
    if isinstance(freqs_cis, tuple):
        cos, sin = reshape_for_broadcast(freqs_cis, xq, head_first)
        xq_out = xq * cos + rotate_half(xq) * sin
        if xk is not None:
            xk_out = xk * cos + rotate_half(xk) * sin
    else:
        xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
        freqs_cis = reshape_for_broadcast(freqs_cis, xq_, head_first)
        xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3).type_as(xq)
        if xk is not None:
            xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
            xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3).type_as(xk)
    return xq_out, xk_out


class Attention(nn.Module):
    """
    We rename some layer names to align with flash attention
    """

    def __init__(self, dim, num_heads, qkv_bias=True, qk_norm=False, attn_drop=0.0, proj_drop=0.0, attn_precision=None, dtype=None, device=None, operations=None):
        super().__init__()
        self.attn_precision = attn_precision
        self.dim = dim
        self.num_heads = num_heads
        assert self.dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.head_dim = self.dim // num_heads
        assert self.head_dim % 8 == 0 and self.head_dim <= 128, 'Only support head_dim <= 128 and divisible by 8'
        self.scale = self.head_dim ** -0.5
        self.Wqkv = operations.Linear(dim, dim * 3, bias=qkv_bias, dtype=dtype, device=device)
        self.q_norm = operations.LayerNorm(self.head_dim, elementwise_affine=True, eps=1e-06, dtype=dtype, device=device) if qk_norm else nn.Identity()
        self.k_norm = operations.LayerNorm(self.head_dim, elementwise_affine=True, eps=1e-06, dtype=dtype, device=device) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.out_proj = operations.Linear(dim, dim, dtype=dtype, device=device)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, freqs_cis_img=None):
        B, N, C = x.shape
        qkv = self.Wqkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q = self.q_norm(q)
        k = self.k_norm(k)
        if freqs_cis_img is not None:
            qq, kk = apply_rotary_emb(q, k, freqs_cis_img, head_first=True)
            assert qq.shape == q.shape and kk.shape == k.shape, f'qq: {qq.shape}, q: {q.shape}, kk: {kk.shape}, k: {k.shape}'
            q, k = qq, kk
        x = optimized_attention(q, k, v, self.num_heads, skip_reshape=True, attn_precision=self.attn_precision)
        x = self.out_proj(x)
        x = self.proj_drop(x)
        out_tuple = x,
        return out_tuple


class ConformerModule(nn.Module):

    def __init__(self, dim, norm_kwargs={}):
        super().__init__()
        self.dim = dim
        self.in_norm = LayerNorm(dim, **norm_kwargs)
        self.pointwise_conv = nn.Conv1d(dim, dim, kernel_size=1, bias=False)
        self.glu = GLU(dim, dim, nn.SiLU())
        self.depthwise_conv = nn.Conv1d(dim, dim, kernel_size=17, groups=dim, padding=8, bias=False)
        self.mid_norm = LayerNorm(dim, **norm_kwargs)
        self.swish = nn.SiLU()
        self.pointwise_conv_2 = nn.Conv1d(dim, dim, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.in_norm(x)
        x = rearrange(x, 'b n d -> b d n')
        x = self.pointwise_conv(x)
        x = rearrange(x, 'b d n -> b n d')
        x = self.glu(x)
        x = rearrange(x, 'b n d -> b d n')
        x = self.depthwise_conv(x)
        x = rearrange(x, 'b d n -> b n d')
        x = self.mid_norm(x)
        x = self.swish(x)
        x = rearrange(x, 'b n d -> b d n')
        x = self.pointwise_conv_2(x)
        x = rearrange(x, 'b d n -> b n d')
        return x


class TransformerBlock(nn.Module):

    def __init__(self, dim, dim_heads=64, cross_attend=False, dim_context=None, global_cond_dim=None, causal=False, zero_init_branch_outputs=True, conformer=False, layer_ix=-1, remove_norms=False, attn_kwargs={}, ff_kwargs={}, norm_kwargs={}, dtype=None, device=None, operations=None):
        super().__init__()
        self.dim = dim
        self.dim_heads = dim_heads
        self.cross_attend = cross_attend
        self.dim_context = dim_context
        self.causal = causal
        self.pre_norm = LayerNorm(dim, dtype=dtype, device=device, **norm_kwargs) if not remove_norms else nn.Identity()
        self.self_attn = Attention(dim, dim_heads=dim_heads, causal=causal, zero_init_output=zero_init_branch_outputs, dtype=dtype, device=device, operations=operations, **attn_kwargs)
        if cross_attend:
            self.cross_attend_norm = LayerNorm(dim, dtype=dtype, device=device, **norm_kwargs) if not remove_norms else nn.Identity()
            self.cross_attn = Attention(dim, dim_heads=dim_heads, dim_context=dim_context, causal=causal, zero_init_output=zero_init_branch_outputs, dtype=dtype, device=device, operations=operations, **attn_kwargs)
        self.ff_norm = LayerNorm(dim, dtype=dtype, device=device, **norm_kwargs) if not remove_norms else nn.Identity()
        self.ff = FeedForward(dim, zero_init_output=zero_init_branch_outputs, dtype=dtype, device=device, operations=operations, **ff_kwargs)
        self.layer_ix = layer_ix
        self.conformer = ConformerModule(dim, norm_kwargs=norm_kwargs) if conformer else None
        self.global_cond_dim = global_cond_dim
        if global_cond_dim is not None:
            self.to_scale_shift_gate = nn.Sequential(nn.SiLU(), nn.Linear(global_cond_dim, dim * 6, bias=False))
            nn.init.zeros_(self.to_scale_shift_gate[1].weight)

    def forward(self, x, context=None, global_cond=None, mask=None, context_mask=None, rotary_pos_emb=None):
        if self.global_cond_dim is not None and self.global_cond_dim > 0 and global_cond is not None:
            scale_self, shift_self, gate_self, scale_ff, shift_ff, gate_ff = self.to_scale_shift_gate(global_cond).unsqueeze(1).chunk(6, dim=-1)
            residual = x
            x = self.pre_norm(x)
            x = x * (1 + scale_self) + shift_self
            x = self.self_attn(x, mask=mask, rotary_pos_emb=rotary_pos_emb)
            x = x * torch.sigmoid(1 - gate_self)
            x = x + residual
            if context is not None:
                x = x + self.cross_attn(self.cross_attend_norm(x), context=context, context_mask=context_mask)
            if self.conformer is not None:
                x = x + self.conformer(x)
            residual = x
            x = self.ff_norm(x)
            x = x * (1 + scale_ff) + shift_ff
            x = self.ff(x)
            x = x * torch.sigmoid(1 - gate_ff)
            x = x + residual
        else:
            x = x + self.self_attn(self.pre_norm(x), mask=mask, rotary_pos_emb=rotary_pos_emb)
            if context is not None:
                x = x + self.cross_attn(self.cross_attend_norm(x), context=context, context_mask=context_mask)
            if self.conformer is not None:
                x = x + self.conformer(x)
            x = x + self.ff(self.ff_norm(x))
        return x


class ContinuousTransformer(nn.Module):

    def __init__(self, dim, depth, *, dim_in=None, dim_out=None, dim_heads=64, cross_attend=False, cond_token_dim=None, global_cond_dim=None, causal=False, rotary_pos_emb=True, zero_init_branch_outputs=True, conformer=False, use_sinusoidal_emb=False, use_abs_pos_emb=False, abs_pos_emb_max_length=10000, dtype=None, device=None, operations=None, **kwargs):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.causal = causal
        self.layers = nn.ModuleList([])
        self.project_in = operations.Linear(dim_in, dim, bias=False, dtype=dtype, device=device) if dim_in is not None else nn.Identity()
        self.project_out = operations.Linear(dim, dim_out, bias=False, dtype=dtype, device=device) if dim_out is not None else nn.Identity()
        if rotary_pos_emb:
            self.rotary_pos_emb = RotaryEmbedding(max(dim_heads // 2, 32), device=device, dtype=dtype)
        else:
            self.rotary_pos_emb = None
        self.use_sinusoidal_emb = use_sinusoidal_emb
        if use_sinusoidal_emb:
            self.pos_emb = ScaledSinusoidalEmbedding(dim)
        self.use_abs_pos_emb = use_abs_pos_emb
        if use_abs_pos_emb:
            self.pos_emb = AbsolutePositionalEmbedding(dim, abs_pos_emb_max_length)
        for i in range(depth):
            self.layers.append(TransformerBlock(dim, dim_heads=dim_heads, cross_attend=cross_attend, dim_context=cond_token_dim, global_cond_dim=global_cond_dim, causal=causal, zero_init_branch_outputs=zero_init_branch_outputs, conformer=conformer, layer_ix=i, dtype=dtype, device=device, operations=operations, **kwargs))

    def forward(self, x, mask=None, prepend_embeds=None, prepend_mask=None, global_cond=None, return_info=False, **kwargs):
        batch, seq, device = *x.shape[:2], x.device
        info = {'hidden_states': []}
        x = self.project_in(x)
        if prepend_embeds is not None:
            prepend_length, prepend_dim = prepend_embeds.shape[1:]
            assert prepend_dim == x.shape[-1], 'prepend dimension must match sequence dimension'
            x = torch.cat((prepend_embeds, x), dim=-2)
            if prepend_mask is not None or mask is not None:
                mask = mask if mask is not None else torch.ones((batch, seq), device=device, dtype=torch.bool)
                prepend_mask = prepend_mask if prepend_mask is not None else torch.ones((batch, prepend_length), device=device, dtype=torch.bool)
                mask = torch.cat((prepend_mask, mask), dim=-1)
        if self.rotary_pos_emb is not None:
            rotary_pos_emb = self.rotary_pos_emb.forward_from_seq_len(x.shape[1], dtype=x.dtype, device=x.device)
        else:
            rotary_pos_emb = None
        if self.use_sinusoidal_emb or self.use_abs_pos_emb:
            x = x + self.pos_emb(x)
        for layer in self.layers:
            x = layer(x, rotary_pos_emb=rotary_pos_emb, global_cond=global_cond, **kwargs)
            if return_info:
                info['hidden_states'].append(x)
        x = self.project_out(x)
        if return_info:
            return x, info
        return x


class AudioDiffusionTransformer(nn.Module):

    def __init__(self, io_channels=64, patch_size=1, embed_dim=1536, cond_token_dim=768, project_cond_tokens=False, global_cond_dim=1536, project_global_cond=True, input_concat_dim=0, prepend_cond_dim=0, depth=24, num_heads=24, transformer_type: "tp.Literal['continuous_transformer']"='continuous_transformer', global_cond_type: "tp.Literal['prepend', 'adaLN']"='prepend', audio_model='', dtype=None, device=None, operations=None, **kwargs):
        super().__init__()
        self.dtype = dtype
        self.cond_token_dim = cond_token_dim
        timestep_features_dim = 256
        self.timestep_features = FourierFeatures(1, timestep_features_dim, dtype=dtype, device=device)
        self.to_timestep_embed = nn.Sequential(operations.Linear(timestep_features_dim, embed_dim, bias=True, dtype=dtype, device=device), nn.SiLU(), operations.Linear(embed_dim, embed_dim, bias=True, dtype=dtype, device=device))
        if cond_token_dim > 0:
            cond_embed_dim = cond_token_dim if not project_cond_tokens else embed_dim
            self.to_cond_embed = nn.Sequential(operations.Linear(cond_token_dim, cond_embed_dim, bias=False, dtype=dtype, device=device), nn.SiLU(), operations.Linear(cond_embed_dim, cond_embed_dim, bias=False, dtype=dtype, device=device))
        else:
            cond_embed_dim = 0
        if global_cond_dim > 0:
            global_embed_dim = global_cond_dim if not project_global_cond else embed_dim
            self.to_global_embed = nn.Sequential(operations.Linear(global_cond_dim, global_embed_dim, bias=False, dtype=dtype, device=device), nn.SiLU(), operations.Linear(global_embed_dim, global_embed_dim, bias=False, dtype=dtype, device=device))
        if prepend_cond_dim > 0:
            self.to_prepend_embed = nn.Sequential(operations.Linear(prepend_cond_dim, embed_dim, bias=False, dtype=dtype, device=device), nn.SiLU(), operations.Linear(embed_dim, embed_dim, bias=False, dtype=dtype, device=device))
        self.input_concat_dim = input_concat_dim
        dim_in = io_channels + self.input_concat_dim
        self.patch_size = patch_size
        self.transformer_type = transformer_type
        self.global_cond_type = global_cond_type
        if self.transformer_type == 'continuous_transformer':
            global_dim = None
            if self.global_cond_type == 'adaLN':
                global_dim = embed_dim
            self.transformer = ContinuousTransformer(dim=embed_dim, depth=depth, dim_heads=embed_dim // num_heads, dim_in=dim_in * patch_size, dim_out=io_channels * patch_size, cross_attend=cond_token_dim > 0, cond_token_dim=cond_embed_dim, global_cond_dim=global_dim, dtype=dtype, device=device, operations=operations, **kwargs)
        else:
            raise ValueError(f'Unknown transformer type: {self.transformer_type}')
        self.preprocess_conv = operations.Conv1d(dim_in, dim_in, 1, bias=False, dtype=dtype, device=device)
        self.postprocess_conv = operations.Conv1d(io_channels, io_channels, 1, bias=False, dtype=dtype, device=device)

    def _forward(self, x, t, mask=None, cross_attn_cond=None, cross_attn_cond_mask=None, input_concat_cond=None, global_embed=None, prepend_cond=None, prepend_cond_mask=None, return_info=False, **kwargs):
        if cross_attn_cond is not None:
            cross_attn_cond = self.to_cond_embed(cross_attn_cond)
        if global_embed is not None:
            global_embed = self.to_global_embed(global_embed)
        prepend_inputs = None
        prepend_mask = None
        prepend_length = 0
        if prepend_cond is not None:
            prepend_cond = self.to_prepend_embed(prepend_cond)
            prepend_inputs = prepend_cond
            if prepend_cond_mask is not None:
                prepend_mask = prepend_cond_mask
        if input_concat_cond is not None:
            if input_concat_cond.shape[2] != x.shape[2]:
                input_concat_cond = F.interpolate(input_concat_cond, (x.shape[2],), mode='nearest')
            x = torch.cat([x, input_concat_cond], dim=1)
        timestep_embed = self.to_timestep_embed(self.timestep_features(t[:, None]))
        if global_embed is not None:
            global_embed = global_embed + timestep_embed
        else:
            global_embed = timestep_embed
        if self.global_cond_type == 'prepend':
            if prepend_inputs is None:
                prepend_inputs = global_embed.unsqueeze(1)
                prepend_mask = torch.ones((x.shape[0], 1), device=x.device, dtype=torch.bool)
            else:
                prepend_inputs = torch.cat([prepend_inputs, global_embed.unsqueeze(1)], dim=1)
                prepend_mask = torch.cat([prepend_mask, torch.ones((x.shape[0], 1), device=x.device, dtype=torch.bool)], dim=1)
            prepend_length = prepend_inputs.shape[1]
        x = self.preprocess_conv(x) + x
        x = rearrange(x, 'b c t -> b t c')
        extra_args = {}
        if self.global_cond_type == 'adaLN':
            extra_args['global_cond'] = global_embed
        if self.patch_size > 1:
            x = rearrange(x, 'b (t p) c -> b t (c p)', p=self.patch_size)
        if self.transformer_type == 'x-transformers':
            output = self.transformer(x, prepend_embeds=prepend_inputs, context=cross_attn_cond, context_mask=cross_attn_cond_mask, mask=mask, prepend_mask=prepend_mask, **extra_args, **kwargs)
        elif self.transformer_type == 'continuous_transformer':
            output = self.transformer(x, prepend_embeds=prepend_inputs, context=cross_attn_cond, context_mask=cross_attn_cond_mask, mask=mask, prepend_mask=prepend_mask, return_info=return_info, **extra_args, **kwargs)
            if return_info:
                output, info = output
        elif self.transformer_type == 'mm_transformer':
            output = self.transformer(x, context=cross_attn_cond, mask=mask, context_mask=cross_attn_cond_mask, **extra_args, **kwargs)
        output = rearrange(output, 'b t c -> b c t')[:, :, prepend_length:]
        if self.patch_size > 1:
            output = rearrange(output, 'b (c p) t -> b c (t p)', p=self.patch_size)
        output = self.postprocess_conv(output) + output
        if return_info:
            return output, info
        return output

    def forward(self, x, timestep, context=None, context_mask=None, input_concat_cond=None, global_embed=None, negative_global_embed=None, prepend_cond=None, prepend_cond_mask=None, mask=None, return_info=False, control=None, transformer_options={}, **kwargs):
        return self._forward(x, timestep, cross_attn_cond=context, cross_attn_cond_mask=context_mask, input_concat_cond=input_concat_cond, global_embed=global_embed, prepend_cond=prepend_cond, prepend_cond_mask=prepend_cond_mask, mask=mask, return_info=return_info, **kwargs)


class LearnedPositionalEmbedding(nn.Module):
    """Used for continuous time"""

    def __init__(self, dim: 'int'):
        super().__init__()
        assert dim % 2 == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.empty(half_dim))

    def forward(self, x: 'Tensor') ->Tensor:
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        fouriered = torch.cat((x, fouriered), dim=-1)
        return fouriered


def TimePositionalEmbedding(dim: 'int', out_features: 'int') ->nn.Module:
    return nn.Sequential(LearnedPositionalEmbedding(dim), comfy.ops.manual_cast.Linear(in_features=dim + 1, out_features=out_features))


class NumberEmbedder(nn.Module):

    def __init__(self, features: 'int', dim: 'int'=256):
        super().__init__()
        self.features = features
        self.embedding = TimePositionalEmbedding(dim=dim, out_features=features)

    def forward(self, x: 'Union[List[float], Tensor]') ->Tensor:
        if not torch.is_tensor(x):
            device = next(self.embedding.parameters()).device
            x = torch.tensor(x, device=device)
        assert isinstance(x, Tensor)
        shape = x.shape
        x = rearrange(x, '... -> (...)')
        embedding = self.embedding(x)
        x = embedding.view(*shape, self.features)
        return x


class Conditioner(nn.Module):

    def __init__(self, dim: 'int', output_dim: 'int', project_out: 'bool'=False):
        super().__init__()
        self.dim = dim
        self.output_dim = output_dim
        self.proj_out = nn.Linear(dim, output_dim) if dim != output_dim or project_out else nn.Identity()

    def forward(self, x):
        raise NotImplementedError()


class NumberConditioner(Conditioner):
    """
        Conditioner that takes a list of floats, normalizes them for a given range, and returns a list of embeddings
    """

    def __init__(self, output_dim: 'int', min_val: 'float'=0, max_val: 'float'=1):
        super().__init__(output_dim, output_dim)
        self.min_val = min_val
        self.max_val = max_val
        self.embedder = NumberEmbedder(features=output_dim)

    def forward(self, floats, device=None):
        floats = [float(x) for x in floats]
        if device is None:
            device = next(self.embedder.parameters()).device
        floats = torch.tensor(floats)
        floats = floats.clamp(self.min_val, self.max_val)
        normalized_floats = (floats - self.min_val) / (self.max_val - self.min_val)
        embedder_dtype = next(self.embedder.parameters()).dtype
        normalized_floats = normalized_floats
        float_embeds = self.embedder(normalized_floats).unsqueeze(1)
        return [float_embeds, torch.ones(float_embeds.shape[0], 1)]


class MultiHeadLayerNorm(nn.Module):

    def __init__(self, hidden_size=None, eps=1e-05, dtype=None, device=None):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(hidden_size, dtype=dtype, device=device))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states
        mean = hidden_states.mean(-1, keepdim=True)
        variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
        hidden_states = (hidden_states - mean) * torch.rsqrt(variance + self.variance_epsilon)
        hidden_states = self.weight * hidden_states
        return hidden_states


class SingleAttention(nn.Module):

    def __init__(self, dim, n_heads, mh_qknorm=False, dtype=None, device=None, operations=None):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.w1q = operations.Linear(dim, dim, bias=False, dtype=dtype, device=device)
        self.w1k = operations.Linear(dim, dim, bias=False, dtype=dtype, device=device)
        self.w1v = operations.Linear(dim, dim, bias=False, dtype=dtype, device=device)
        self.w1o = operations.Linear(dim, dim, bias=False, dtype=dtype, device=device)
        self.q_norm1 = MultiHeadLayerNorm((self.n_heads, self.head_dim), dtype=dtype, device=device) if mh_qknorm else operations.LayerNorm(self.head_dim, elementwise_affine=False, dtype=dtype, device=device)
        self.k_norm1 = MultiHeadLayerNorm((self.n_heads, self.head_dim), dtype=dtype, device=device) if mh_qknorm else operations.LayerNorm(self.head_dim, elementwise_affine=False, dtype=dtype, device=device)

    def forward(self, c):
        bsz, seqlen1, _ = c.shape
        q, k, v = self.w1q(c), self.w1k(c), self.w1v(c)
        q = q.view(bsz, seqlen1, self.n_heads, self.head_dim)
        k = k.view(bsz, seqlen1, self.n_heads, self.head_dim)
        v = v.view(bsz, seqlen1, self.n_heads, self.head_dim)
        q, k = self.q_norm1(q), self.k_norm1(k)
        output = optimized_attention(q.permute(0, 2, 1, 3), k.permute(0, 2, 1, 3), v.permute(0, 2, 1, 3), self.n_heads, skip_reshape=True)
        c = self.w1o(output)
        return c


class DoubleAttention(nn.Module):

    def __init__(self, dim, n_heads, mh_qknorm=False, dtype=None, device=None, operations=None):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.w1q = operations.Linear(dim, dim, bias=False, dtype=dtype, device=device)
        self.w1k = operations.Linear(dim, dim, bias=False, dtype=dtype, device=device)
        self.w1v = operations.Linear(dim, dim, bias=False, dtype=dtype, device=device)
        self.w1o = operations.Linear(dim, dim, bias=False, dtype=dtype, device=device)
        self.w2q = operations.Linear(dim, dim, bias=False, dtype=dtype, device=device)
        self.w2k = operations.Linear(dim, dim, bias=False, dtype=dtype, device=device)
        self.w2v = operations.Linear(dim, dim, bias=False, dtype=dtype, device=device)
        self.w2o = operations.Linear(dim, dim, bias=False, dtype=dtype, device=device)
        self.q_norm1 = MultiHeadLayerNorm((self.n_heads, self.head_dim), dtype=dtype, device=device) if mh_qknorm else operations.LayerNorm(self.head_dim, elementwise_affine=False, dtype=dtype, device=device)
        self.k_norm1 = MultiHeadLayerNorm((self.n_heads, self.head_dim), dtype=dtype, device=device) if mh_qknorm else operations.LayerNorm(self.head_dim, elementwise_affine=False, dtype=dtype, device=device)
        self.q_norm2 = MultiHeadLayerNorm((self.n_heads, self.head_dim), dtype=dtype, device=device) if mh_qknorm else operations.LayerNorm(self.head_dim, elementwise_affine=False, dtype=dtype, device=device)
        self.k_norm2 = MultiHeadLayerNorm((self.n_heads, self.head_dim), dtype=dtype, device=device) if mh_qknorm else operations.LayerNorm(self.head_dim, elementwise_affine=False, dtype=dtype, device=device)

    def forward(self, c, x):
        bsz, seqlen1, _ = c.shape
        bsz, seqlen2, _ = x.shape
        seqlen = seqlen1 + seqlen2
        cq, ck, cv = self.w1q(c), self.w1k(c), self.w1v(c)
        cq = cq.view(bsz, seqlen1, self.n_heads, self.head_dim)
        ck = ck.view(bsz, seqlen1, self.n_heads, self.head_dim)
        cv = cv.view(bsz, seqlen1, self.n_heads, self.head_dim)
        cq, ck = self.q_norm1(cq), self.k_norm1(ck)
        xq, xk, xv = self.w2q(x), self.w2k(x), self.w2v(x)
        xq = xq.view(bsz, seqlen2, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seqlen2, self.n_heads, self.head_dim)
        xv = xv.view(bsz, seqlen2, self.n_heads, self.head_dim)
        xq, xk = self.q_norm2(xq), self.k_norm2(xk)
        q, k, v = torch.cat([cq, xq], dim=1), torch.cat([ck, xk], dim=1), torch.cat([cv, xv], dim=1)
        output = optimized_attention(q.permute(0, 2, 1, 3), k.permute(0, 2, 1, 3), v.permute(0, 2, 1, 3), self.n_heads, skip_reshape=True)
        c, x = output.split([seqlen1, seqlen2], dim=1)
        c = self.w1o(c)
        x = self.w2o(x)
        return c, x


def modulate(x, shift, scale):
    if shift is None:
        shift = torch.zeros_like(scale)
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class MMDiTBlock(nn.Module):

    def __init__(self, dim, heads=8, global_conddim=1024, is_last=False, dtype=None, device=None, operations=None):
        super().__init__()
        self.normC1 = operations.LayerNorm(dim, elementwise_affine=False, dtype=dtype, device=device)
        self.normC2 = operations.LayerNorm(dim, elementwise_affine=False, dtype=dtype, device=device)
        if not is_last:
            self.mlpC = MLP(dim, hidden_dim=dim * 4, dtype=dtype, device=device, operations=operations)
            self.modC = nn.Sequential(nn.SiLU(), operations.Linear(global_conddim, 6 * dim, bias=False, dtype=dtype, device=device))
        else:
            self.modC = nn.Sequential(nn.SiLU(), operations.Linear(global_conddim, 2 * dim, bias=False, dtype=dtype, device=device))
        self.normX1 = operations.LayerNorm(dim, elementwise_affine=False, dtype=dtype, device=device)
        self.normX2 = operations.LayerNorm(dim, elementwise_affine=False, dtype=dtype, device=device)
        self.mlpX = MLP(dim, hidden_dim=dim * 4, dtype=dtype, device=device, operations=operations)
        self.modX = nn.Sequential(nn.SiLU(), operations.Linear(global_conddim, 6 * dim, bias=False, dtype=dtype, device=device))
        self.attn = DoubleAttention(dim, heads, dtype=dtype, device=device, operations=operations)
        self.is_last = is_last

    def forward(self, c, x, global_cond, **kwargs):
        cres, xres = c, x
        cshift_msa, cscale_msa, cgate_msa, cshift_mlp, cscale_mlp, cgate_mlp = self.modC(global_cond).chunk(6, dim=1)
        c = modulate(self.normC1(c), cshift_msa, cscale_msa)
        xshift_msa, xscale_msa, xgate_msa, xshift_mlp, xscale_mlp, xgate_mlp = self.modX(global_cond).chunk(6, dim=1)
        x = modulate(self.normX1(x), xshift_msa, xscale_msa)
        c, x = self.attn(c, x)
        c = self.normC2(cres + cgate_msa.unsqueeze(1) * c)
        c = cgate_mlp.unsqueeze(1) * self.mlpC(modulate(c, cshift_mlp, cscale_mlp))
        c = cres + c
        x = self.normX2(xres + xgate_msa.unsqueeze(1) * x)
        x = xgate_mlp.unsqueeze(1) * self.mlpX(modulate(x, xshift_mlp, xscale_mlp))
        x = xres + x
        return c, x


class DiTBlock(nn.Module):

    def __init__(self, dim, heads=8, global_conddim=1024, dtype=None, device=None, operations=None):
        super().__init__()
        self.norm1 = operations.LayerNorm(dim, elementwise_affine=False, dtype=dtype, device=device)
        self.norm2 = operations.LayerNorm(dim, elementwise_affine=False, dtype=dtype, device=device)
        self.modCX = nn.Sequential(nn.SiLU(), operations.Linear(global_conddim, 6 * dim, bias=False, dtype=dtype, device=device))
        self.attn = SingleAttention(dim, heads, dtype=dtype, device=device, operations=operations)
        self.mlp = MLP(dim, hidden_dim=dim * 4, dtype=dtype, device=device, operations=operations)

    def forward(self, cx, global_cond, **kwargs):
        cxres = cx
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.modCX(global_cond).chunk(6, dim=1)
        cx = modulate(self.norm1(cx), shift_msa, scale_msa)
        cx = self.attn(cx)
        cx = self.norm2(cxres + gate_msa.unsqueeze(1) * cx)
        mlpout = self.mlp(modulate(cx, shift_mlp, scale_mlp))
        cx = gate_mlp.unsqueeze(1) * mlpout
        cx = cxres + cx
        return cx


def timestep_embedding(timesteps, dim, max_period=10000, repeat_only=False):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    if not repeat_only:
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=timesteps.device) / half)
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    else:
        embedding = repeat(timesteps, 'b -> b d', d=dim)
    return embedding


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256, dtype=None, device=None, operations=None):
        super().__init__()
        self.mlp = nn.Sequential(operations.Linear(frequency_embedding_size, hidden_size, bias=True, dtype=dtype, device=device), nn.SiLU(), operations.Linear(hidden_size, hidden_size, bias=True, dtype=dtype, device=device))
        self.frequency_embedding_size = frequency_embedding_size

    def forward(self, t, dtype, **kwargs):
        t_freq = timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, norm_layer=None, bias=True, drop=0.0, use_conv=False, dtype=None, device=None, operations=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        drop_probs = drop
        linear_layer = partial(operations.Conv2d, kernel_size=1) if use_conv else operations.Linear
        self.fc1 = linear_layer(in_features, hidden_features, bias=bias, dtype=dtype, device=device)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs)
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias, dtype=dtype, device=device)
        self.drop2 = nn.Dropout(drop_probs)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


def split_qkv(qkv, head_dim):
    qkv = qkv.reshape(qkv.shape[0], qkv.shape[1], 3, -1, head_dim).movedim(2, 0)
    return qkv[0], qkv[1], qkv[2]


class SelfAttentionContext(nn.Module):

    def __init__(self, dim, heads=8, dim_head=64, dtype=None, device=None, operations=None):
        super().__init__()
        dim_head = dim // heads
        inner_dim = dim
        self.heads = heads
        self.dim_head = dim_head
        self.qkv = operations.Linear(dim, dim * 3, bias=True, dtype=dtype, device=device)
        self.proj = operations.Linear(inner_dim, dim, dtype=dtype, device=device)

    def forward(self, x):
        qkv = self.qkv(x)
        q, k, v = split_qkv(qkv, self.dim_head)
        x = optimized_attention(q.reshape(q.shape[0], q.shape[1], -1), k, v, heads=self.heads)
        return self.proj(x)


class ContextProcessorBlock(nn.Module):

    def __init__(self, context_size, dtype=None, device=None, operations=None):
        super().__init__()
        self.norm1 = operations.LayerNorm(context_size, elementwise_affine=False, eps=1e-06, dtype=dtype, device=device)
        self.attn = SelfAttentionContext(context_size, dtype=dtype, device=device, operations=operations)
        self.norm2 = operations.LayerNorm(context_size, elementwise_affine=False, eps=1e-06, dtype=dtype, device=device)
        self.mlp = Mlp(in_features=context_size, hidden_features=context_size * 4, act_layer=lambda : nn.GELU(approximate='tanh'), drop=0, dtype=dtype, device=device, operations=operations)

    def forward(self, x):
        x += self.attn(self.norm1(x))
        x += self.mlp(self.norm2(x))
        return x


class ContextProcessor(nn.Module):

    def __init__(self, context_size, num_layers, dtype=None, device=None, operations=None):
        super().__init__()
        self.layers = torch.nn.ModuleList([ContextProcessorBlock(context_size, dtype=dtype, device=device, operations=operations) for i in range(num_layers)])
        self.norm = operations.LayerNorm(context_size, elementwise_affine=False, eps=1e-06, dtype=dtype, device=device)

    def forward(self, x):
        for i, l in enumerate(self.layers):
            x = l(x)
        return self.norm(x)


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """

    def __init__(self, hidden_size: 'int', patch_size: 'int', out_channels: 'int', total_out_channels: 'Optional[int]'=None, dtype=None, device=None, operations=None):
        super().__init__()
        self.norm_final = operations.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-06, dtype=dtype, device=device)
        self.linear = operations.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True, dtype=dtype, device=device) if total_out_channels is None else operations.Linear(hidden_size, total_out_channels, bias=True, dtype=dtype, device=device)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), operations.Linear(hidden_size, 2 * hidden_size, bias=True, dtype=dtype, device=device))

    def forward(self, x: 'torch.Tensor', c: 'torch.Tensor') ->torch.Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class RMSNorm(torch.nn.Module):

    def __init__(self, dim: 'int', elementwise_affine: 'bool'=False, eps: 'float'=1e-06, device=None, dtype=None):
        """
        Initialize the RMSNorm normalization layer.
        Args:
            dim (int): The dimension of the input tensor.
            eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.
        Attributes:
            eps (float): A small value added to the denominator for numerical stability.
            weight (nn.Parameter): Learnable scaling parameter.
        """
        super().__init__()
        self.eps = eps
        self.learnable_scale = elementwise_affine
        if self.learnable_scale:
            self.weight = nn.Parameter(torch.empty(dim, device=device, dtype=dtype))
        else:
            self.register_parameter('weight', None)

    def forward(self, x):
        return comfy.ldm.common_dit.rms_norm(x, self.weight, self.eps)


class SelfAttention(nn.Module):
    ATTENTION_MODES = 'xformers', 'torch', 'torch-hb', 'math', 'debug'

    def __init__(self, dim: 'int', num_heads: 'int'=8, qkv_bias: 'bool'=False, qk_scale: 'Optional[float]'=None, proj_drop: 'float'=0.0, attn_mode: 'str'='xformers', pre_only: 'bool'=False, qk_norm: 'Optional[str]'=None, rmsnorm: 'bool'=False, dtype=None, device=None, operations=None):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = operations.Linear(dim, dim * 3, bias=qkv_bias, dtype=dtype, device=device)
        if not pre_only:
            self.proj = operations.Linear(dim, dim, dtype=dtype, device=device)
            self.proj_drop = nn.Dropout(proj_drop)
        assert attn_mode in self.ATTENTION_MODES
        self.attn_mode = attn_mode
        self.pre_only = pre_only
        if qk_norm == 'rms':
            self.ln_q = RMSNorm(self.head_dim, elementwise_affine=True, eps=1e-06, dtype=dtype, device=device)
            self.ln_k = RMSNorm(self.head_dim, elementwise_affine=True, eps=1e-06, dtype=dtype, device=device)
        elif qk_norm == 'ln':
            self.ln_q = operations.LayerNorm(self.head_dim, elementwise_affine=True, eps=1e-06, dtype=dtype, device=device)
            self.ln_k = operations.LayerNorm(self.head_dim, elementwise_affine=True, eps=1e-06, dtype=dtype, device=device)
        elif qk_norm is None:
            self.ln_q = nn.Identity()
            self.ln_k = nn.Identity()
        else:
            raise ValueError(qk_norm)

    def pre_attention(self, x: 'torch.Tensor') ->torch.Tensor:
        B, L, C = x.shape
        qkv = self.qkv(x)
        q, k, v = split_qkv(qkv, self.head_dim)
        q = self.ln_q(q).reshape(q.shape[0], q.shape[1], -1)
        k = self.ln_k(k).reshape(q.shape[0], q.shape[1], -1)
        return q, k, v

    def post_attention(self, x: 'torch.Tensor') ->torch.Tensor:
        assert not self.pre_only
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        q, k, v = self.pre_attention(x)
        x = optimized_attention(q, k, v, heads=self.num_heads)
        x = self.post_attention(x)
        return x


class SwiGLUFeedForward(nn.Module):

    def __init__(self, dim: 'int', hidden_dim: 'int', multiple_of: 'int', ffn_dim_multiplier: 'Optional[float]'=None):
        """
        Initialize the FeedForward module.

        Args:
            dim (int): Input dimension.
            hidden_dim (int): Hidden dimension of the feedforward layer.
            multiple_of (int): Value to ensure hidden dimension is a multiple of this value.
            ffn_dim_multiplier (float, optional): Custom multiplier for hidden dimension. Defaults to None.

        Attributes:
            w1 (ColumnParallelLinear): Linear transformation for the first layer.
            w2 (RowParallelLinear): Linear transformation for the second layer.
            w3 (ColumnParallelLinear): Linear transformation for the third layer.

        """
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(nn.functional.silu(self.w1(x)) * self.w3(x))


class DismantledBlock(nn.Module):
    """
    A DiT block with gated adaptive layer norm (adaLN) conditioning.
    """
    ATTENTION_MODES = 'xformers', 'torch', 'torch-hb', 'math', 'debug'

    def __init__(self, hidden_size: 'int', num_heads: 'int', mlp_ratio: 'float'=4.0, attn_mode: 'str'='xformers', qkv_bias: 'bool'=False, pre_only: 'bool'=False, rmsnorm: 'bool'=False, scale_mod_only: 'bool'=False, swiglu: 'bool'=False, qk_norm: 'Optional[str]'=None, x_block_self_attn: 'bool'=False, dtype=None, device=None, operations=None, **block_kwargs):
        super().__init__()
        assert attn_mode in self.ATTENTION_MODES
        if not rmsnorm:
            self.norm1 = operations.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-06, dtype=dtype, device=device)
        else:
            self.norm1 = RMSNorm(hidden_size, elementwise_affine=False, eps=1e-06)
        self.attn = SelfAttention(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias, attn_mode=attn_mode, pre_only=pre_only, qk_norm=qk_norm, rmsnorm=rmsnorm, dtype=dtype, device=device, operations=operations)
        if x_block_self_attn:
            assert not pre_only
            assert not scale_mod_only
            self.x_block_self_attn = True
            self.attn2 = SelfAttention(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias, attn_mode=attn_mode, pre_only=False, qk_norm=qk_norm, rmsnorm=rmsnorm, dtype=dtype, device=device, operations=operations)
        else:
            self.x_block_self_attn = False
        if not pre_only:
            if not rmsnorm:
                self.norm2 = operations.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-06, dtype=dtype, device=device)
            else:
                self.norm2 = RMSNorm(hidden_size, elementwise_affine=False, eps=1e-06)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        if not pre_only:
            if not swiglu:
                self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=lambda : nn.GELU(approximate='tanh'), drop=0, dtype=dtype, device=device, operations=operations)
            else:
                self.mlp = SwiGLUFeedForward(dim=hidden_size, hidden_dim=mlp_hidden_dim, multiple_of=256)
        self.scale_mod_only = scale_mod_only
        if x_block_self_attn:
            assert not pre_only
            assert not scale_mod_only
            n_mods = 9
        elif not scale_mod_only:
            n_mods = 6 if not pre_only else 2
        else:
            n_mods = 4 if not pre_only else 1
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), operations.Linear(hidden_size, n_mods * hidden_size, bias=True, dtype=dtype, device=device))
        self.pre_only = pre_only

    def pre_attention(self, x: 'torch.Tensor', c: 'torch.Tensor') ->torch.Tensor:
        if not self.pre_only:
            if not self.scale_mod_only:
                shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
            else:
                shift_msa = None
                shift_mlp = None
                scale_msa, gate_msa, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(4, dim=1)
            qkv = self.attn.pre_attention(modulate(self.norm1(x), shift_msa, scale_msa))
            return qkv, (x, gate_msa, shift_mlp, scale_mlp, gate_mlp)
        else:
            if not self.scale_mod_only:
                shift_msa, scale_msa = self.adaLN_modulation(c).chunk(2, dim=1)
            else:
                shift_msa = None
                scale_msa = self.adaLN_modulation(c)
            qkv = self.attn.pre_attention(modulate(self.norm1(x), shift_msa, scale_msa))
            return qkv, None

    def post_attention(self, attn, x, gate_msa, shift_mlp, scale_mlp, gate_mlp):
        assert not self.pre_only
        x = x + gate_msa.unsqueeze(1) * self.attn.post_attention(attn)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x

    def pre_attention_x(self, x: 'torch.Tensor', c: 'torch.Tensor') ->torch.Tensor:
        assert self.x_block_self_attn
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp, shift_msa2, scale_msa2, gate_msa2 = self.adaLN_modulation(c).chunk(9, dim=1)
        x_norm = self.norm1(x)
        qkv = self.attn.pre_attention(modulate(x_norm, shift_msa, scale_msa))
        qkv2 = self.attn2.pre_attention(modulate(x_norm, shift_msa2, scale_msa2))
        return qkv, qkv2, (x, gate_msa, shift_mlp, scale_mlp, gate_mlp, gate_msa2)

    def post_attention_x(self, attn, attn2, x, gate_msa, shift_mlp, scale_mlp, gate_mlp, gate_msa2):
        assert not self.pre_only
        attn1 = self.attn.post_attention(attn)
        attn2 = self.attn2.post_attention(attn2)
        out1 = gate_msa.unsqueeze(1) * attn1
        out2 = gate_msa2.unsqueeze(1) * attn2
        x = x + out1
        x = x + out2
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x

    def forward(self, x: 'torch.Tensor', c: 'torch.Tensor') ->torch.Tensor:
        assert not self.pre_only
        if self.x_block_self_attn:
            qkv, qkv2, intermediates = self.pre_attention_x(x, c)
            attn, _ = optimized_attention(qkv[0], qkv[1], qkv[2], num_heads=self.attn.num_heads)
            attn2, _ = optimized_attention(qkv2[0], qkv2[1], qkv2[2], num_heads=self.attn2.num_heads)
            return self.post_attention_x(attn, attn2, *intermediates)
        else:
            qkv, intermediates = self.pre_attention(x, c)
            attn = optimized_attention(qkv[0], qkv[1], qkv[2], heads=self.attn.num_heads)
            return self.post_attention(attn, *intermediates)


def _block_mixing(context, x, context_block, x_block, c):
    context_qkv, context_intermediates = context_block.pre_attention(context, c)
    if x_block.x_block_self_attn:
        x_qkv, x_qkv2, x_intermediates = x_block.pre_attention_x(x, c)
    else:
        x_qkv, x_intermediates = x_block.pre_attention(x, c)
    o = []
    for t in range(3):
        o.append(torch.cat((context_qkv[t], x_qkv[t]), dim=1))
    qkv = tuple(o)
    attn = optimized_attention(qkv[0], qkv[1], qkv[2], heads=x_block.attn.num_heads)
    context_attn, x_attn = attn[:, :context_qkv[0].shape[1]], attn[:, context_qkv[0].shape[1]:]
    if not context_block.pre_only:
        context = context_block.post_attention(context_attn, *context_intermediates)
    else:
        context = None
    if x_block.x_block_self_attn:
        attn2 = optimized_attention(x_qkv2[0], x_qkv2[1], x_qkv2[2], heads=x_block.attn2.num_heads)
        x = x_block.post_attention_x(x_attn, attn2, *x_intermediates)
    else:
        x = x_block.post_attention(x_attn, *x_intermediates)
    return context, x


def block_mixing(*args, use_checkpoint=True, **kwargs):
    if use_checkpoint:
        return torch.utils.checkpoint.checkpoint(_block_mixing, *args, use_reentrant=False, **kwargs)
    else:
        return _block_mixing(*args, **kwargs)


class JointBlock(nn.Module):
    """just a small wrapper to serve as a fsdp unit"""

    def __init__(self, *args, **kwargs):
        super().__init__()
        pre_only = kwargs.pop('pre_only')
        qk_norm = kwargs.pop('qk_norm', None)
        x_block_self_attn = kwargs.pop('x_block_self_attn', False)
        self.context_block = DismantledBlock(*args, pre_only=pre_only, qk_norm=qk_norm, **kwargs)
        self.x_block = DismantledBlock(*args, pre_only=False, qk_norm=qk_norm, x_block_self_attn=x_block_self_attn, **kwargs)

    def forward(self, *args, **kwargs):
        return block_mixing(*args, context_block=self.context_block, x_block=self.x_block, **kwargs)


class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    dynamic_img_pad: 'torch.jit.Final[bool]'

    def __init__(self, img_size: 'Optional[int]'=224, patch_size: 'int'=16, in_chans: 'int'=3, embed_dim: 'int'=768, norm_layer=None, flatten: 'bool'=True, bias: 'bool'=True, strict_img_size: 'bool'=True, dynamic_img_pad: 'bool'=True, padding_mode='circular', dtype=None, device=None, operations=None):
        super().__init__()
        self.patch_size = patch_size, patch_size
        self.padding_mode = padding_mode
        if img_size is not None:
            self.img_size = img_size, img_size
            self.grid_size = tuple([(s // p) for s, p in zip(self.img_size, self.patch_size)])
            self.num_patches = self.grid_size[0] * self.grid_size[1]
        else:
            self.img_size = None
            self.grid_size = None
            self.num_patches = None
        self.flatten = flatten
        self.strict_img_size = strict_img_size
        self.dynamic_img_pad = dynamic_img_pad
        self.proj = operations.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias, dtype=dtype, device=device)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        if self.dynamic_img_pad:
            x = comfy.ldm.common_dit.pad_to_patch_size(x, self.patch_size, padding_mode=self.padding_mode)
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x


class VectorEmbedder(nn.Module):
    """
    Embeds a flat vector of dimension input_dim
    """

    def __init__(self, input_dim: 'int', hidden_size: 'int', dtype=None, device=None, operations=None):
        super().__init__()
        self.mlp = nn.Sequential(operations.Linear(input_dim, hidden_size, bias=True, dtype=dtype, device=device), nn.SiLU(), operations.Linear(hidden_size, hidden_size, bias=True, dtype=dtype, device=device))

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        emb = self.mlp(x)
        return emb


def get_1d_sincos_pos_embed_from_grid_torch(embed_dim, pos, device=None, dtype=torch.float32):
    omega = torch.arange(embed_dim // 2, device=device, dtype=dtype)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000 ** omega
    pos = pos.reshape(-1)
    out = torch.einsum('m,d->md', pos, omega)
    emb_sin = torch.sin(out)
    emb_cos = torch.cos(out)
    emb = torch.cat([emb_sin, emb_cos], dim=1)
    return emb


def get_2d_sincos_pos_embed_torch(embed_dim, w, h, val_center=7.5, val_magnitude=7.5, device=None, dtype=torch.float32):
    small = min(h, w)
    val_h = h / small * val_magnitude
    val_w = w / small * val_magnitude
    grid_h, grid_w = torch.meshgrid(torch.linspace(-val_h + val_center, val_h + val_center, h, device=device, dtype=dtype), torch.linspace(-val_w + val_center, val_w + val_center, w, device=device, dtype=dtype), indexing='ij')
    emb_h = get_1d_sincos_pos_embed_from_grid_torch(embed_dim // 2, grid_h, device=device, dtype=dtype)
    emb_w = get_1d_sincos_pos_embed_from_grid_torch(embed_dim // 2, grid_w, device=device, dtype=dtype)
    emb = torch.cat([emb_w, emb_h], dim=1)
    return emb


class MMDiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """

    def __init__(self, input_size: 'int'=32, patch_size: 'int'=2, in_channels: 'int'=4, depth: 'int'=28, mlp_ratio: 'float'=4.0, learn_sigma: 'bool'=False, adm_in_channels: 'Optional[int]'=None, context_embedder_config: 'Optional[Dict]'=None, compile_core: 'bool'=False, use_checkpoint: 'bool'=False, register_length: 'int'=0, attn_mode: 'str'='torch', rmsnorm: 'bool'=False, scale_mod_only: 'bool'=False, swiglu: 'bool'=False, out_channels: 'Optional[int]'=None, pos_embed_scaling_factor: 'Optional[float]'=None, pos_embed_offset: 'Optional[float]'=None, pos_embed_max_size: 'Optional[int]'=None, num_patches=None, qk_norm: 'Optional[str]'=None, qkv_bias: 'bool'=True, context_processor_layers=None, x_block_self_attn: 'bool'=False, x_block_self_attn_layers: 'Optional[List[int]]'=[], context_size=4096, num_blocks=None, final_layer=True, skip_blocks=False, dtype=None, device=None, operations=None):
        super().__init__()
        self.dtype = dtype
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        default_out_channels = in_channels * 2 if learn_sigma else in_channels
        self.out_channels = default(out_channels, default_out_channels)
        self.patch_size = patch_size
        self.pos_embed_scaling_factor = pos_embed_scaling_factor
        self.pos_embed_offset = pos_embed_offset
        self.pos_embed_max_size = pos_embed_max_size
        self.x_block_self_attn_layers = x_block_self_attn_layers
        self.hidden_size = 64 * depth
        num_heads = depth
        if num_blocks is None:
            num_blocks = depth
        self.depth = depth
        self.num_heads = num_heads
        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, self.hidden_size, bias=True, strict_img_size=self.pos_embed_max_size is None, dtype=dtype, device=device, operations=operations)
        self.t_embedder = TimestepEmbedder(self.hidden_size, dtype=dtype, device=device, operations=operations)
        self.y_embedder = None
        if adm_in_channels is not None:
            assert isinstance(adm_in_channels, int)
            self.y_embedder = VectorEmbedder(adm_in_channels, self.hidden_size, dtype=dtype, device=device, operations=operations)
        if context_processor_layers is not None:
            self.context_processor = ContextProcessor(context_size, context_processor_layers, dtype=dtype, device=device, operations=operations)
        else:
            self.context_processor = None
        self.context_embedder = nn.Identity()
        if context_embedder_config is not None:
            if context_embedder_config['target'] == 'torch.nn.Linear':
                self.context_embedder = operations.Linear(**context_embedder_config['params'], dtype=dtype, device=device)
        self.register_length = register_length
        if self.register_length > 0:
            self.register = nn.Parameter(torch.randn(1, register_length, self.hidden_size, dtype=dtype, device=device))
        if num_patches is not None:
            self.register_buffer('pos_embed', torch.empty(1, num_patches, self.hidden_size, dtype=dtype, device=device))
        else:
            self.pos_embed = None
        self.use_checkpoint = use_checkpoint
        if not skip_blocks:
            self.joint_blocks = nn.ModuleList([JointBlock(self.hidden_size, num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, attn_mode=attn_mode, pre_only=i == num_blocks - 1 and final_layer, rmsnorm=rmsnorm, scale_mod_only=scale_mod_only, swiglu=swiglu, qk_norm=qk_norm, x_block_self_attn=i in self.x_block_self_attn_layers or x_block_self_attn, dtype=dtype, device=device, operations=operations) for i in range(num_blocks)])
        if final_layer:
            self.final_layer = FinalLayer(self.hidden_size, patch_size, self.out_channels, dtype=dtype, device=device, operations=operations)
        if compile_core:
            assert False
            self.forward_core_with_concat = torch.compile(self.forward_core_with_concat)

    def cropped_pos_embed(self, hw, device=None):
        p = self.x_embedder.patch_size[0]
        h, w = hw
        h = (h + 1) // p
        w = (w + 1) // p
        if self.pos_embed is None:
            return get_2d_sincos_pos_embed_torch(self.hidden_size, w, h, device=device)
        assert self.pos_embed_max_size is not None
        assert h <= self.pos_embed_max_size, (h, self.pos_embed_max_size)
        assert w <= self.pos_embed_max_size, (w, self.pos_embed_max_size)
        top = (self.pos_embed_max_size - h) // 2
        left = (self.pos_embed_max_size - w) // 2
        spatial_pos_embed = rearrange(self.pos_embed, '1 (h w) c -> 1 h w c', h=self.pos_embed_max_size, w=self.pos_embed_max_size)
        spatial_pos_embed = spatial_pos_embed[:, top:top + h, left:left + w, :]
        spatial_pos_embed = rearrange(spatial_pos_embed, '1 h w c -> 1 (h w) c')
        return spatial_pos_embed

    def unpatchify(self, x, hw=None):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        if hw is None:
            h = w = int(x.shape[1] ** 0.5)
        else:
            h, w = hw
            h = (h + 1) // p
            w = (w + 1) // p
        assert h * w == x.shape[1]
        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, w * p))
        return imgs

    def forward_core_with_concat(self, x: 'torch.Tensor', c_mod: 'torch.Tensor', context: 'Optional[torch.Tensor]'=None, control=None, transformer_options={}) ->torch.Tensor:
        patches_replace = transformer_options.get('patches_replace', {})
        if self.register_length > 0:
            context = torch.cat((repeat(self.register, '1 ... -> b ...', b=x.shape[0]), default(context, torch.Tensor([]).type_as(x))), 1)
        blocks_replace = patches_replace.get('dit', {})
        blocks = len(self.joint_blocks)
        for i in range(blocks):
            if ('double_block', i) in blocks_replace:

                def block_wrap(args):
                    out = {}
                    out['txt'], out['img'] = self.joint_blocks[i](args['txt'], args['img'], c=args['vec'])
                    return out
                out = blocks_replace['double_block', i]({'img': x, 'txt': context, 'vec': c_mod}, {'original_block': block_wrap})
                context = out['txt']
                x = out['img']
            else:
                context, x = self.joint_blocks[i](context, x, c=c_mod, use_checkpoint=self.use_checkpoint)
            if control is not None:
                control_o = control.get('output')
                if i < len(control_o):
                    add = control_o[i]
                    if add is not None:
                        x += add
        x = self.final_layer(x, c_mod)
        return x

    def forward(self, x: 'torch.Tensor', t: 'torch.Tensor', y: 'Optional[torch.Tensor]'=None, context: 'Optional[torch.Tensor]'=None, control=None, transformer_options={}) ->torch.Tensor:
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        if self.context_processor is not None:
            context = self.context_processor(context)
        hw = x.shape[-2:]
        x = self.x_embedder(x) + comfy.ops.cast_to_input(self.cropped_pos_embed(hw, device=x.device), x)
        c = self.t_embedder(t, dtype=x.dtype)
        if y is not None and self.y_embedder is not None:
            y = self.y_embedder(y)
            c = c + y
        if context is not None:
            context = self.context_embedder(context)
        x = self.forward_core_with_concat(x, c, context, control, transformer_options)
        x = self.unpatchify(x, hw=hw)
        return x[:, :, :hw[-2], :hw[-1]]


class Attention2D(nn.Module):

    def __init__(self, c, nhead, dropout=0.0, dtype=None, device=None, operations=None):
        super().__init__()
        self.attn = OptimizedAttention(c, nhead, dtype=dtype, device=device, operations=operations)

    def forward(self, x, kv, self_attn=False):
        orig_shape = x.shape
        x = x.view(x.size(0), x.size(1), -1).permute(0, 2, 1)
        if self_attn:
            kv = torch.cat([x, kv], dim=1)
        x = self.attn(x, kv, kv)
        x = x.permute(0, 2, 1).view(*orig_shape)
        return x


class GlobalResponseNorm(nn.Module):
    """from https://github.com/facebookresearch/ConvNeXt-V2/blob/3608f67cc1dae164790c5d0aead7bf2d73d9719b/models/utils.py#L105"""

    def __init__(self, dim, dtype=None, device=None):
        super().__init__()
        self.gamma = nn.Parameter(torch.empty(1, 1, 1, dim, dtype=dtype, device=device))
        self.beta = nn.Parameter(torch.empty(1, 1, 1, dim, dtype=dtype, device=device))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-06)
        return comfy.ops.cast_to_input(self.gamma, x) * (x * Nx) + comfy.ops.cast_to_input(self.beta, x) + x


def avg_pool_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D average pooling module.
    """
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f'unsupported dimensions: {dims}')


def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f'unsupported dimensions: {dims}')


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None, padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(dims, self.channels, self.out_channels, 3, stride=stride, padding=padding)
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if not self.use_conv:
            padding = [x.shape[2] % 2, x.shape[3] % 2]
            self.op.padding = padding
        x = self.op(x)
        return x


class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None, padding=1, dtype=None, device=None, operations=ops):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = operations.conv_nd(dims, self.channels, self.out_channels, 3, padding=padding, dtype=dtype, device=device)

    def forward(self, x, output_shape=None):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            shape = [x.shape[2], x.shape[3] * 2, x.shape[4] * 2]
            if output_shape is not None:
                shape[1] = output_shape[3]
                shape[2] = output_shape[4]
        else:
            shape = [x.shape[2] * 2, x.shape[3] * 2]
            if output_shape is not None:
                shape[0] = output_shape[2]
                shape[1] = output_shape[3]
        x = F.interpolate(x, size=shape, mode='nearest')
        if self.use_conv:
            x = self.conv(x)
        return x


class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(self, channels, emb_channels, dropout, out_channels=None, use_conv=False, use_scale_shift_norm=False, dims=2, use_checkpoint=False, up=False, down=False, kernel_size=3, exchange_temb_dims=False, skip_t_emb=False, dtype=None, device=None, operations=ops):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm
        self.exchange_temb_dims = exchange_temb_dims
        if isinstance(kernel_size, list):
            padding = [(k // 2) for k in kernel_size]
        else:
            padding = kernel_size // 2
        self.in_layers = nn.Sequential(operations.GroupNorm(32, channels, dtype=dtype, device=device), nn.SiLU(), operations.conv_nd(dims, channels, self.out_channels, kernel_size, padding=padding, dtype=dtype, device=device))
        self.updown = up or down
        if up:
            self.h_upd = Upsample(channels, False, dims, dtype=dtype, device=device)
            self.x_upd = Upsample(channels, False, dims, dtype=dtype, device=device)
        elif down:
            self.h_upd = Downsample(channels, False, dims, dtype=dtype, device=device)
            self.x_upd = Downsample(channels, False, dims, dtype=dtype, device=device)
        else:
            self.h_upd = self.x_upd = nn.Identity()
        self.skip_t_emb = skip_t_emb
        if self.skip_t_emb:
            self.emb_layers = None
            self.exchange_temb_dims = False
        else:
            self.emb_layers = nn.Sequential(nn.SiLU(), operations.Linear(emb_channels, 2 * self.out_channels if use_scale_shift_norm else self.out_channels, dtype=dtype, device=device))
        self.out_layers = nn.Sequential(operations.GroupNorm(32, self.out_channels, dtype=dtype, device=device), nn.SiLU(), nn.Dropout(p=dropout), operations.conv_nd(dims, self.out_channels, self.out_channels, kernel_size, padding=padding, dtype=dtype, device=device))
        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = operations.conv_nd(dims, channels, self.out_channels, kernel_size, padding=padding, dtype=dtype, device=device)
        else:
            self.skip_connection = operations.conv_nd(dims, channels, self.out_channels, 1, dtype=dtype, device=device)

    def forward(self, x, emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.
        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return checkpoint(self._forward, (x, emb), self.parameters(), self.use_checkpoint)

    def _forward(self, x, emb):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        emb_out = None
        if not self.skip_t_emb:
            emb_out = self.emb_layers(emb).type(h.dtype)
            while len(emb_out.shape) < len(h.shape):
                emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            h = out_norm(h)
            if emb_out is not None:
                scale, shift = th.chunk(emb_out, 2, dim=1)
                h *= 1 + scale
                h += shift
            h = out_rest(h)
        else:
            if emb_out is not None:
                if self.exchange_temb_dims:
                    emb_out = emb_out.movedim(1, 2)
                h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h


def Normalize(in_channels, num_groups=32):
    return ops.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-06, affine=True)


def slice_attention(q, k, v):
    r1 = torch.zeros_like(k, device=q.device)
    scale = int(q.shape[-1]) ** -0.5
    mem_free_total = model_management.get_free_memory(q.device)
    gb = 1024 ** 3
    tensor_size = q.shape[0] * q.shape[1] * k.shape[2] * q.element_size()
    modifier = 3 if q.element_size() == 2 else 2.5
    mem_required = tensor_size * modifier
    steps = 1
    if mem_required > mem_free_total:
        steps = 2 ** math.ceil(math.log(mem_required / mem_free_total, 2))
    while True:
        try:
            slice_size = q.shape[1] // steps if q.shape[1] % steps == 0 else q.shape[1]
            for i in range(0, q.shape[1], slice_size):
                end = i + slice_size
                s1 = torch.bmm(q[:, i:end], k) * scale
                s2 = torch.nn.functional.softmax(s1, dim=2).permute(0, 2, 1)
                del s1
                r1[:, :, i:end] = torch.bmm(v, s2)
                del s2
            break
        except model_management.OOM_EXCEPTION as e:
            model_management.soft_empty_cache(True)
            steps *= 2
            if steps > 128:
                raise e
            logging.warning('out of memory error, increasing steps and trying again {}'.format(steps))
    return r1


def normal_attention(q, k, v):
    b, c, h, w = q.shape
    q = q.reshape(b, c, h * w)
    q = q.permute(0, 2, 1)
    k = k.reshape(b, c, h * w)
    v = v.reshape(b, c, h * w)
    r1 = slice_attention(q, k, v)
    h_ = r1.reshape(b, c, h, w)
    del r1
    return h_


def pytorch_attention(q, k, v):
    B, C, H, W = q.shape
    q, k, v = map(lambda t: t.view(B, 1, C, -1).transpose(2, 3).contiguous(), (q, k, v))
    try:
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False)
        out = out.transpose(2, 3).reshape(B, C, H, W)
    except model_management.OOM_EXCEPTION as e:
        logging.warning('scaled_dot_product_attention OOMed: switched to slice attention')
        out = slice_attention(q.view(B, -1, C), k.view(B, -1, C).transpose(1, 2), v.view(B, -1, C).transpose(1, 2)).reshape(B, C, H, W)
    return out


def xformers_attention(q, k, v):
    B, C, H, W = q.shape
    q, k, v = map(lambda t: t.view(B, C, -1).transpose(1, 2).contiguous(), (q, k, v))
    try:
        out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=None)
        out = out.transpose(1, 2).reshape(B, C, H, W)
    except NotImplementedError as e:
        out = slice_attention(q.view(B, -1, C), k.view(B, -1, C).transpose(1, 2), v.view(B, -1, C).transpose(1, 2)).reshape(B, C, H, W)
    return out


class AttnBlock(nn.Module):

    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.norm = Normalize(in_channels)
        self.q = ops.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = ops.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = ops.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = ops.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        if model_management.xformers_enabled_vae():
            logging.info('Using xformers attention in VAE')
            self.optimized_attention = xformers_attention
        elif model_management.pytorch_attention_enabled():
            logging.info('Using pytorch attention in VAE')
            self.optimized_attention = pytorch_attention
        else:
            logging.info('Using split attention in VAE')
            self.optimized_attention = normal_attention

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)
        h_ = self.optimized_attention(q, k, v)
        h_ = self.proj_out(h_)
        return x + h_


class FeedForwardBlock(nn.Module):

    def __init__(self, c, dropout=0.0, dtype=None, device=None, operations=None):
        super().__init__()
        self.norm = LayerNorm2d_op(operations)(c, elementwise_affine=False, eps=1e-06, dtype=dtype, device=device)
        self.channelwise = nn.Sequential(operations.Linear(c, c * 4, dtype=dtype, device=device), nn.GELU(), GlobalResponseNorm(c * 4, dtype=dtype, device=device), nn.Dropout(dropout), operations.Linear(c * 4, c, dtype=dtype, device=device))

    def forward(self, x):
        x = x + self.channelwise(self.norm(x).permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return x


class vector_quantize(Function):

    @staticmethod
    def forward(ctx, x, codebook):
        with torch.no_grad():
            codebook_sqr = torch.sum(codebook ** 2, dim=1)
            x_sqr = torch.sum(x ** 2, dim=1, keepdim=True)
            dist = torch.addmm(codebook_sqr + x_sqr, x, codebook.t(), alpha=-2.0, beta=1.0)
            _, indices = dist.min(dim=1)
            ctx.save_for_backward(indices, codebook)
            ctx.mark_non_differentiable(indices)
            nn = torch.index_select(codebook, 0, indices)
            return nn, indices

    @staticmethod
    def backward(ctx, grad_output, grad_indices):
        grad_inputs, grad_codebook = None, None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_output.clone()
        if ctx.needs_input_grad[1]:
            indices, codebook = ctx.saved_tensors
            grad_codebook = torch.zeros_like(codebook)
            grad_codebook.index_add_(0, indices, grad_output)
        return grad_inputs, grad_codebook


class VectorQuantize(nn.Module):

    def __init__(self, embedding_size, k, ema_decay=0.99, ema_loss=False):
        """
        Takes an input of variable size (as long as the last dimension matches the embedding size).
        Returns one tensor containing the nearest neigbour embeddings to each of the inputs,
        with the same size as the input, vq and commitment components for the loss as a touple
        in the second output and the indices of the quantized vectors in the third:
        quantized, (vq_loss, commit_loss), indices
        """
        super(VectorQuantize, self).__init__()
        self.codebook = nn.Embedding(k, embedding_size)
        self.codebook.weight.data.uniform_(-1.0 / k, 1.0 / k)
        self.vq = vector_quantize.apply
        self.ema_decay = ema_decay
        self.ema_loss = ema_loss
        if ema_loss:
            self.register_buffer('ema_element_count', torch.ones(k))
            self.register_buffer('ema_weight_sum', torch.zeros_like(self.codebook.weight))

    def _laplace_smoothing(self, x, epsilon):
        n = torch.sum(x)
        return (x + epsilon) / (n + x.size(0) * epsilon) * n

    def _updateEMA(self, z_e_x, indices):
        mask = nn.functional.one_hot(indices, self.ema_element_count.size(0)).float()
        elem_count = mask.sum(dim=0)
        weight_sum = torch.mm(mask.t(), z_e_x)
        self.ema_element_count = self.ema_decay * self.ema_element_count + (1 - self.ema_decay) * elem_count
        self.ema_element_count = self._laplace_smoothing(self.ema_element_count, 1e-05)
        self.ema_weight_sum = self.ema_decay * self.ema_weight_sum + (1 - self.ema_decay) * weight_sum
        self.codebook.weight.data = self.ema_weight_sum / self.ema_element_count.unsqueeze(-1)

    def idx2vq(self, idx, dim=-1):
        q_idx = self.codebook(idx)
        if dim != -1:
            q_idx = q_idx.movedim(-1, dim)
        return q_idx

    def forward(self, x, get_losses=True, dim=-1):
        if dim != -1:
            x = x.movedim(dim, -1)
        z_e_x = x.contiguous().view(-1, x.size(-1)) if len(x.shape) > 2 else x
        z_q_x, indices = self.vq(z_e_x, self.codebook.weight.detach())
        vq_loss, commit_loss = None, None
        if self.ema_loss and self.training:
            self._updateEMA(z_e_x.detach(), indices.detach())
        z_q_x_grd = torch.index_select(self.codebook.weight, dim=0, index=indices)
        if get_losses:
            vq_loss = (z_q_x_grd - z_e_x.detach()).pow(2).mean()
            commit_loss = (z_e_x - z_q_x_grd.detach()).pow(2).mean()
        z_q_x = z_q_x.view(x.shape)
        if dim != -1:
            z_q_x = z_q_x.movedim(-1, dim)
        return z_q_x, (vq_loss, commit_loss), indices.view(x.shape[:-1])


class StageA(nn.Module):

    def __init__(self, levels=2, bottleneck_blocks=12, c_hidden=384, c_latent=4, codebook_size=8192):
        super().__init__()
        self.c_latent = c_latent
        c_levels = [(c_hidden // 2 ** i) for i in reversed(range(levels))]
        self.in_block = nn.Sequential(nn.PixelUnshuffle(2), nn.Conv2d(3 * 4, c_levels[0], kernel_size=1))
        down_blocks = []
        for i in range(levels):
            if i > 0:
                down_blocks.append(nn.Conv2d(c_levels[i - 1], c_levels[i], kernel_size=4, stride=2, padding=1))
            block = ResBlock(c_levels[i], c_levels[i] * 4)
            down_blocks.append(block)
        down_blocks.append(nn.Sequential(nn.Conv2d(c_levels[-1], c_latent, kernel_size=1, bias=False), nn.BatchNorm2d(c_latent)))
        self.down_blocks = nn.Sequential(*down_blocks)
        self.down_blocks[0]
        self.codebook_size = codebook_size
        self.vquantizer = VectorQuantize(c_latent, k=codebook_size)
        up_blocks = [nn.Sequential(nn.Conv2d(c_latent, c_levels[-1], kernel_size=1))]
        for i in range(levels):
            for j in range(bottleneck_blocks if i == 0 else 1):
                block = ResBlock(c_levels[levels - 1 - i], c_levels[levels - 1 - i] * 4)
                up_blocks.append(block)
            if i < levels - 1:
                up_blocks.append(nn.ConvTranspose2d(c_levels[levels - 1 - i], c_levels[levels - 2 - i], kernel_size=4, stride=2, padding=1))
        self.up_blocks = nn.Sequential(*up_blocks)
        self.out_block = nn.Sequential(nn.Conv2d(c_levels[0], 3 * 4, kernel_size=1), nn.PixelShuffle(2))

    def encode(self, x, quantize=False):
        x = self.in_block(x)
        x = self.down_blocks(x)
        if quantize:
            qe, (vq_loss, commit_loss), indices = self.vquantizer.forward(x, dim=1)
            return qe, x, indices, vq_loss + commit_loss * 0.25
        else:
            return x

    def decode(self, x):
        x = self.up_blocks(x)
        x = self.out_block(x)
        return x

    def forward(self, x, quantize=False):
        qe, x, _, vq_loss = self.encode(x, quantize)
        x = self.decode(qe)
        return x, vq_loss


class Discriminator(nn.Module):

    def __init__(self, c_in=3, c_cond=0, c_hidden=512, depth=6):
        super().__init__()
        d = max(depth - 3, 3)
        layers = [nn.utils.spectral_norm(nn.Conv2d(c_in, c_hidden // 2 ** d, kernel_size=3, stride=2, padding=1)), nn.LeakyReLU(0.2)]
        for i in range(depth - 1):
            c_in = c_hidden // 2 ** max(d - i, 0)
            c_out = c_hidden // 2 ** max(d - 1 - i, 0)
            layers.append(nn.utils.spectral_norm(nn.Conv2d(c_in, c_out, kernel_size=3, stride=2, padding=1)))
            layers.append(nn.InstanceNorm2d(c_out))
            layers.append(nn.LeakyReLU(0.2))
        self.encoder = nn.Sequential(*layers)
        self.shuffle = nn.Conv2d(c_hidden + c_cond if c_cond > 0 else c_hidden, 1, kernel_size=1)
        self.logits = nn.Sigmoid()

    def forward(self, x, cond=None):
        x = self.encoder(x)
        if cond is not None:
            cond = cond.view(cond.size(0), cond.size(1), 1, 1).expand(-1, -1, x.size(-2), x.size(-1))
            x = torch.cat([x, cond], dim=1)
        x = self.shuffle(x)
        x = self.logits(x)
        return x


class StageB(nn.Module):

    def __init__(self, c_in=4, c_out=4, c_r=64, patch_size=2, c_cond=1280, c_hidden=[320, 640, 1280, 1280], nhead=[-1, -1, 20, 20], blocks=[[2, 6, 28, 6], [6, 28, 6, 2]], block_repeat=[[1, 1, 1, 1], [3, 3, 2, 2]], level_config=['CT', 'CT', 'CTA', 'CTA'], c_clip=1280, c_clip_seq=4, c_effnet=16, c_pixels=3, kernel_size=3, dropout=[0, 0, 0.0, 0.0], self_attn=True, t_conds=['sca'], stable_cascade_stage=None, dtype=None, device=None, operations=None):
        super().__init__()
        self.dtype = dtype
        self.c_r = c_r
        self.t_conds = t_conds
        self.c_clip_seq = c_clip_seq
        if not isinstance(dropout, list):
            dropout = [dropout] * len(c_hidden)
        if not isinstance(self_attn, list):
            self_attn = [self_attn] * len(c_hidden)
        self.effnet_mapper = nn.Sequential(operations.Conv2d(c_effnet, c_hidden[0] * 4, kernel_size=1, dtype=dtype, device=device), nn.GELU(), operations.Conv2d(c_hidden[0] * 4, c_hidden[0], kernel_size=1, dtype=dtype, device=device), LayerNorm2d_op(operations)(c_hidden[0], elementwise_affine=False, eps=1e-06, dtype=dtype, device=device))
        self.pixels_mapper = nn.Sequential(operations.Conv2d(c_pixels, c_hidden[0] * 4, kernel_size=1, dtype=dtype, device=device), nn.GELU(), operations.Conv2d(c_hidden[0] * 4, c_hidden[0], kernel_size=1, dtype=dtype, device=device), LayerNorm2d_op(operations)(c_hidden[0], elementwise_affine=False, eps=1e-06, dtype=dtype, device=device))
        self.clip_mapper = operations.Linear(c_clip, c_cond * c_clip_seq, dtype=dtype, device=device)
        self.clip_norm = operations.LayerNorm(c_cond, elementwise_affine=False, eps=1e-06, dtype=dtype, device=device)
        self.embedding = nn.Sequential(nn.PixelUnshuffle(patch_size), operations.Conv2d(c_in * patch_size ** 2, c_hidden[0], kernel_size=1, dtype=dtype, device=device), LayerNorm2d_op(operations)(c_hidden[0], elementwise_affine=False, eps=1e-06, dtype=dtype, device=device))

        def get_block(block_type, c_hidden, nhead, c_skip=0, dropout=0, self_attn=True):
            if block_type == 'C':
                return ResBlock(c_hidden, c_skip, kernel_size=kernel_size, dropout=dropout, dtype=dtype, device=device, operations=operations)
            elif block_type == 'A':
                return AttnBlock(c_hidden, c_cond, nhead, self_attn=self_attn, dropout=dropout, dtype=dtype, device=device, operations=operations)
            elif block_type == 'F':
                return FeedForwardBlock(c_hidden, dropout=dropout, dtype=dtype, device=device, operations=operations)
            elif block_type == 'T':
                return TimestepBlock(c_hidden, c_r, conds=t_conds, dtype=dtype, device=device, operations=operations)
            else:
                raise Exception(f'Block type {block_type} not supported')
        self.down_blocks = nn.ModuleList()
        self.down_downscalers = nn.ModuleList()
        self.down_repeat_mappers = nn.ModuleList()
        for i in range(len(c_hidden)):
            if i > 0:
                self.down_downscalers.append(nn.Sequential(LayerNorm2d_op(operations)(c_hidden[i - 1], elementwise_affine=False, eps=1e-06, dtype=dtype, device=device), operations.Conv2d(c_hidden[i - 1], c_hidden[i], kernel_size=2, stride=2, dtype=dtype, device=device)))
            else:
                self.down_downscalers.append(nn.Identity())
            down_block = nn.ModuleList()
            for _ in range(blocks[0][i]):
                for block_type in level_config[i]:
                    block = get_block(block_type, c_hidden[i], nhead[i], dropout=dropout[i], self_attn=self_attn[i])
                    down_block.append(block)
            self.down_blocks.append(down_block)
            if block_repeat is not None:
                block_repeat_mappers = nn.ModuleList()
                for _ in range(block_repeat[0][i] - 1):
                    block_repeat_mappers.append(operations.Conv2d(c_hidden[i], c_hidden[i], kernel_size=1, dtype=dtype, device=device))
                self.down_repeat_mappers.append(block_repeat_mappers)
        self.up_blocks = nn.ModuleList()
        self.up_upscalers = nn.ModuleList()
        self.up_repeat_mappers = nn.ModuleList()
        for i in reversed(range(len(c_hidden))):
            if i > 0:
                self.up_upscalers.append(nn.Sequential(LayerNorm2d_op(operations)(c_hidden[i], elementwise_affine=False, eps=1e-06, dtype=dtype, device=device), operations.ConvTranspose2d(c_hidden[i], c_hidden[i - 1], kernel_size=2, stride=2, dtype=dtype, device=device)))
            else:
                self.up_upscalers.append(nn.Identity())
            up_block = nn.ModuleList()
            for j in range(blocks[1][::-1][i]):
                for k, block_type in enumerate(level_config[i]):
                    c_skip = c_hidden[i] if i < len(c_hidden) - 1 and j == k == 0 else 0
                    block = get_block(block_type, c_hidden[i], nhead[i], c_skip=c_skip, dropout=dropout[i], self_attn=self_attn[i])
                    up_block.append(block)
            self.up_blocks.append(up_block)
            if block_repeat is not None:
                block_repeat_mappers = nn.ModuleList()
                for _ in range(block_repeat[1][::-1][i] - 1):
                    block_repeat_mappers.append(operations.Conv2d(c_hidden[i], c_hidden[i], kernel_size=1, dtype=dtype, device=device))
                self.up_repeat_mappers.append(block_repeat_mappers)
        self.clf = nn.Sequential(LayerNorm2d_op(operations)(c_hidden[0], elementwise_affine=False, eps=1e-06, dtype=dtype, device=device), operations.Conv2d(c_hidden[0], c_out * patch_size ** 2, kernel_size=1, dtype=dtype, device=device), nn.PixelShuffle(patch_size))

    def gen_r_embedding(self, r, max_positions=10000):
        r = r * max_positions
        half_dim = self.c_r // 2
        emb = math.log(max_positions) / (half_dim - 1)
        emb = torch.arange(half_dim, device=r.device).float().mul(-emb).exp()
        emb = r[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=1)
        if self.c_r % 2 == 1:
            emb = nn.functional.pad(emb, (0, 1), mode='constant')
        return emb

    def gen_c_embeddings(self, clip):
        if len(clip.shape) == 2:
            clip = clip.unsqueeze(1)
        clip = self.clip_mapper(clip).view(clip.size(0), clip.size(1) * self.c_clip_seq, -1)
        clip = self.clip_norm(clip)
        return clip

    def _down_encode(self, x, r_embed, clip):
        level_outputs = []
        block_group = zip(self.down_blocks, self.down_downscalers, self.down_repeat_mappers)
        for down_block, downscaler, repmap in block_group:
            x = downscaler(x)
            for i in range(len(repmap) + 1):
                for block in down_block:
                    if isinstance(block, ResBlock) or hasattr(block, '_fsdp_wrapped_module') and isinstance(block._fsdp_wrapped_module, ResBlock):
                        x = block(x)
                    elif isinstance(block, AttnBlock) or hasattr(block, '_fsdp_wrapped_module') and isinstance(block._fsdp_wrapped_module, AttnBlock):
                        x = block(x, clip)
                    elif isinstance(block, TimestepBlock) or hasattr(block, '_fsdp_wrapped_module') and isinstance(block._fsdp_wrapped_module, TimestepBlock):
                        x = block(x, r_embed)
                    else:
                        x = block(x)
                if i < len(repmap):
                    x = repmap[i](x)
            level_outputs.insert(0, x)
        return level_outputs

    def _up_decode(self, level_outputs, r_embed, clip):
        x = level_outputs[0]
        block_group = zip(self.up_blocks, self.up_upscalers, self.up_repeat_mappers)
        for i, (up_block, upscaler, repmap) in enumerate(block_group):
            for j in range(len(repmap) + 1):
                for k, block in enumerate(up_block):
                    if isinstance(block, ResBlock) or hasattr(block, '_fsdp_wrapped_module') and isinstance(block._fsdp_wrapped_module, ResBlock):
                        skip = level_outputs[i] if k == 0 and i > 0 else None
                        if skip is not None and (x.size(-1) != skip.size(-1) or x.size(-2) != skip.size(-2)):
                            x = torch.nn.functional.interpolate(x, skip.shape[-2:], mode='bilinear', align_corners=True)
                        x = block(x, skip)
                    elif isinstance(block, AttnBlock) or hasattr(block, '_fsdp_wrapped_module') and isinstance(block._fsdp_wrapped_module, AttnBlock):
                        x = block(x, clip)
                    elif isinstance(block, TimestepBlock) or hasattr(block, '_fsdp_wrapped_module') and isinstance(block._fsdp_wrapped_module, TimestepBlock):
                        x = block(x, r_embed)
                    else:
                        x = block(x)
                if j < len(repmap):
                    x = repmap[j](x)
            x = upscaler(x)
        return x

    def forward(self, x, r, effnet, clip, pixels=None, **kwargs):
        if pixels is None:
            pixels = x.new_zeros(x.size(0), 3, 8, 8)
        r_embed = self.gen_r_embedding(r)
        for c in self.t_conds:
            t_cond = kwargs.get(c, torch.zeros_like(r))
            r_embed = torch.cat([r_embed, self.gen_r_embedding(t_cond)], dim=1)
        clip = self.gen_c_embeddings(clip)
        x = self.embedding(x)
        x = x + self.effnet_mapper(nn.functional.interpolate(effnet, size=x.shape[-2:], mode='bilinear', align_corners=True))
        x = x + nn.functional.interpolate(self.pixels_mapper(pixels), size=x.shape[-2:], mode='bilinear', align_corners=True)
        level_outputs = self._down_encode(x, r_embed, clip)
        x = self._up_decode(level_outputs, r_embed, clip)
        return self.clf(x)

    def update_weights_ema(self, src_model, beta=0.999):
        for self_params, src_params in zip(self.parameters(), src_model.parameters()):
            self_params.data = self_params.data * beta + src_params.data.clone() * (1 - beta)
        for self_buffers, src_buffers in zip(self.buffers(), src_model.buffers()):
            self_buffers.data = self_buffers.data * beta + src_buffers.data.clone() * (1 - beta)


class UpDownBlock2d(nn.Module):

    def __init__(self, c_in, c_out, mode, enabled=True, dtype=None, device=None, operations=None):
        super().__init__()
        assert mode in ['up', 'down']
        interpolation = nn.Upsample(scale_factor=2 if mode == 'up' else 0.5, mode='bilinear', align_corners=True) if enabled else nn.Identity()
        mapping = operations.Conv2d(c_in, c_out, kernel_size=1, dtype=dtype, device=device)
        self.blocks = nn.ModuleList([interpolation, mapping] if mode == 'up' else [mapping, interpolation])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class StageC(nn.Module):

    def __init__(self, c_in=16, c_out=16, c_r=64, patch_size=1, c_cond=2048, c_hidden=[2048, 2048], nhead=[32, 32], blocks=[[8, 24], [24, 8]], block_repeat=[[1, 1], [1, 1]], level_config=['CTA', 'CTA'], c_clip_text=1280, c_clip_text_pooled=1280, c_clip_img=768, c_clip_seq=4, kernel_size=3, dropout=[0.0, 0.0], self_attn=True, t_conds=['sca', 'crp'], switch_level=[False], stable_cascade_stage=None, dtype=None, device=None, operations=None):
        super().__init__()
        self.dtype = dtype
        self.c_r = c_r
        self.t_conds = t_conds
        self.c_clip_seq = c_clip_seq
        if not isinstance(dropout, list):
            dropout = [dropout] * len(c_hidden)
        if not isinstance(self_attn, list):
            self_attn = [self_attn] * len(c_hidden)
        self.clip_txt_mapper = operations.Linear(c_clip_text, c_cond, dtype=dtype, device=device)
        self.clip_txt_pooled_mapper = operations.Linear(c_clip_text_pooled, c_cond * c_clip_seq, dtype=dtype, device=device)
        self.clip_img_mapper = operations.Linear(c_clip_img, c_cond * c_clip_seq, dtype=dtype, device=device)
        self.clip_norm = operations.LayerNorm(c_cond, elementwise_affine=False, eps=1e-06, dtype=dtype, device=device)
        self.embedding = nn.Sequential(nn.PixelUnshuffle(patch_size), operations.Conv2d(c_in * patch_size ** 2, c_hidden[0], kernel_size=1, dtype=dtype, device=device), LayerNorm2d_op(operations)(c_hidden[0], elementwise_affine=False, eps=1e-06))

        def get_block(block_type, c_hidden, nhead, c_skip=0, dropout=0, self_attn=True):
            if block_type == 'C':
                return ResBlock(c_hidden, c_skip, kernel_size=kernel_size, dropout=dropout, dtype=dtype, device=device, operations=operations)
            elif block_type == 'A':
                return AttnBlock(c_hidden, c_cond, nhead, self_attn=self_attn, dropout=dropout, dtype=dtype, device=device, operations=operations)
            elif block_type == 'F':
                return FeedForwardBlock(c_hidden, dropout=dropout, dtype=dtype, device=device, operations=operations)
            elif block_type == 'T':
                return TimestepBlock(c_hidden, c_r, conds=t_conds, dtype=dtype, device=device, operations=operations)
            else:
                raise Exception(f'Block type {block_type} not supported')
        self.down_blocks = nn.ModuleList()
        self.down_downscalers = nn.ModuleList()
        self.down_repeat_mappers = nn.ModuleList()
        for i in range(len(c_hidden)):
            if i > 0:
                self.down_downscalers.append(nn.Sequential(LayerNorm2d_op(operations)(c_hidden[i - 1], elementwise_affine=False, eps=1e-06), UpDownBlock2d(c_hidden[i - 1], c_hidden[i], mode='down', enabled=switch_level[i - 1], dtype=dtype, device=device, operations=operations)))
            else:
                self.down_downscalers.append(nn.Identity())
            down_block = nn.ModuleList()
            for _ in range(blocks[0][i]):
                for block_type in level_config[i]:
                    block = get_block(block_type, c_hidden[i], nhead[i], dropout=dropout[i], self_attn=self_attn[i])
                    down_block.append(block)
            self.down_blocks.append(down_block)
            if block_repeat is not None:
                block_repeat_mappers = nn.ModuleList()
                for _ in range(block_repeat[0][i] - 1):
                    block_repeat_mappers.append(operations.Conv2d(c_hidden[i], c_hidden[i], kernel_size=1, dtype=dtype, device=device))
                self.down_repeat_mappers.append(block_repeat_mappers)
        self.up_blocks = nn.ModuleList()
        self.up_upscalers = nn.ModuleList()
        self.up_repeat_mappers = nn.ModuleList()
        for i in reversed(range(len(c_hidden))):
            if i > 0:
                self.up_upscalers.append(nn.Sequential(LayerNorm2d_op(operations)(c_hidden[i], elementwise_affine=False, eps=1e-06), UpDownBlock2d(c_hidden[i], c_hidden[i - 1], mode='up', enabled=switch_level[i - 1], dtype=dtype, device=device, operations=operations)))
            else:
                self.up_upscalers.append(nn.Identity())
            up_block = nn.ModuleList()
            for j in range(blocks[1][::-1][i]):
                for k, block_type in enumerate(level_config[i]):
                    c_skip = c_hidden[i] if i < len(c_hidden) - 1 and j == k == 0 else 0
                    block = get_block(block_type, c_hidden[i], nhead[i], c_skip=c_skip, dropout=dropout[i], self_attn=self_attn[i])
                    up_block.append(block)
            self.up_blocks.append(up_block)
            if block_repeat is not None:
                block_repeat_mappers = nn.ModuleList()
                for _ in range(block_repeat[1][::-1][i] - 1):
                    block_repeat_mappers.append(operations.Conv2d(c_hidden[i], c_hidden[i], kernel_size=1, dtype=dtype, device=device))
                self.up_repeat_mappers.append(block_repeat_mappers)
        self.clf = nn.Sequential(LayerNorm2d_op(operations)(c_hidden[0], elementwise_affine=False, eps=1e-06, dtype=dtype, device=device), operations.Conv2d(c_hidden[0], c_out * patch_size ** 2, kernel_size=1, dtype=dtype, device=device), nn.PixelShuffle(patch_size))

    def gen_r_embedding(self, r, max_positions=10000):
        r = r * max_positions
        half_dim = self.c_r // 2
        emb = math.log(max_positions) / (half_dim - 1)
        emb = torch.arange(half_dim, device=r.device).float().mul(-emb).exp()
        emb = r[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=1)
        if self.c_r % 2 == 1:
            emb = nn.functional.pad(emb, (0, 1), mode='constant')
        return emb

    def gen_c_embeddings(self, clip_txt, clip_txt_pooled, clip_img):
        clip_txt = self.clip_txt_mapper(clip_txt)
        if len(clip_txt_pooled.shape) == 2:
            clip_txt_pooled = clip_txt_pooled.unsqueeze(1)
        if len(clip_img.shape) == 2:
            clip_img = clip_img.unsqueeze(1)
        clip_txt_pool = self.clip_txt_pooled_mapper(clip_txt_pooled).view(clip_txt_pooled.size(0), clip_txt_pooled.size(1) * self.c_clip_seq, -1)
        clip_img = self.clip_img_mapper(clip_img).view(clip_img.size(0), clip_img.size(1) * self.c_clip_seq, -1)
        clip = torch.cat([clip_txt, clip_txt_pool, clip_img], dim=1)
        clip = self.clip_norm(clip)
        return clip

    def _down_encode(self, x, r_embed, clip, cnet=None):
        level_outputs = []
        block_group = zip(self.down_blocks, self.down_downscalers, self.down_repeat_mappers)
        for down_block, downscaler, repmap in block_group:
            x = downscaler(x)
            for i in range(len(repmap) + 1):
                for block in down_block:
                    if isinstance(block, ResBlock) or hasattr(block, '_fsdp_wrapped_module') and isinstance(block._fsdp_wrapped_module, ResBlock):
                        if cnet is not None:
                            next_cnet = cnet.pop()
                            if next_cnet is not None:
                                x = x + nn.functional.interpolate(next_cnet, size=x.shape[-2:], mode='bilinear', align_corners=True)
                        x = block(x)
                    elif isinstance(block, AttnBlock) or hasattr(block, '_fsdp_wrapped_module') and isinstance(block._fsdp_wrapped_module, AttnBlock):
                        x = block(x, clip)
                    elif isinstance(block, TimestepBlock) or hasattr(block, '_fsdp_wrapped_module') and isinstance(block._fsdp_wrapped_module, TimestepBlock):
                        x = block(x, r_embed)
                    else:
                        x = block(x)
                if i < len(repmap):
                    x = repmap[i](x)
            level_outputs.insert(0, x)
        return level_outputs

    def _up_decode(self, level_outputs, r_embed, clip, cnet=None):
        x = level_outputs[0]
        block_group = zip(self.up_blocks, self.up_upscalers, self.up_repeat_mappers)
        for i, (up_block, upscaler, repmap) in enumerate(block_group):
            for j in range(len(repmap) + 1):
                for k, block in enumerate(up_block):
                    if isinstance(block, ResBlock) or hasattr(block, '_fsdp_wrapped_module') and isinstance(block._fsdp_wrapped_module, ResBlock):
                        skip = level_outputs[i] if k == 0 and i > 0 else None
                        if skip is not None and (x.size(-1) != skip.size(-1) or x.size(-2) != skip.size(-2)):
                            x = torch.nn.functional.interpolate(x, skip.shape[-2:], mode='bilinear', align_corners=True)
                        if cnet is not None:
                            next_cnet = cnet.pop()
                            if next_cnet is not None:
                                x = x + nn.functional.interpolate(next_cnet, size=x.shape[-2:], mode='bilinear', align_corners=True)
                        x = block(x, skip)
                    elif isinstance(block, AttnBlock) or hasattr(block, '_fsdp_wrapped_module') and isinstance(block._fsdp_wrapped_module, AttnBlock):
                        x = block(x, clip)
                    elif isinstance(block, TimestepBlock) or hasattr(block, '_fsdp_wrapped_module') and isinstance(block._fsdp_wrapped_module, TimestepBlock):
                        x = block(x, r_embed)
                    else:
                        x = block(x)
                if j < len(repmap):
                    x = repmap[j](x)
            x = upscaler(x)
        return x

    def forward(self, x, r, clip_text, clip_text_pooled, clip_img, control=None, **kwargs):
        r_embed = self.gen_r_embedding(r)
        for c in self.t_conds:
            t_cond = kwargs.get(c, torch.zeros_like(r))
            r_embed = torch.cat([r_embed, self.gen_r_embedding(t_cond)], dim=1)
        clip = self.gen_c_embeddings(clip_text, clip_text_pooled, clip_img)
        if control is not None:
            cnet = control.get('input')
        else:
            cnet = None
        x = self.embedding(x)
        level_outputs = self._down_encode(x, r_embed, clip, cnet)
        x = self._up_decode(level_outputs, r_embed, clip, cnet)
        return self.clf(x)

    def update_weights_ema(self, src_model, beta=0.999):
        for self_params, src_params in zip(self.parameters(), src_model.parameters()):
            self_params.data = self_params.data * beta + src_params.data.clone() * (1 - beta)
        for self_buffers, src_buffers in zip(self.buffers(), src_model.buffers()):
            self_buffers.data = self_buffers.data * beta + src_buffers.data.clone() * (1 - beta)


class EfficientNetEncoder(nn.Module):

    def __init__(self, c_latent=16):
        super().__init__()
        self.backbone = torchvision.models.efficientnet_v2_s().features.eval()
        self.mapper = nn.Sequential(nn.Conv2d(1280, c_latent, kernel_size=1, bias=False), nn.BatchNorm2d(c_latent, affine=False))
        self.mean = nn.Parameter(torch.tensor([0.485, 0.456, 0.406]))
        self.std = nn.Parameter(torch.tensor([0.229, 0.224, 0.225]))

    def forward(self, x):
        x = x * 0.5 + 0.5
        x = (x - self.mean.view([3, 1, 1])) / self.std.view([3, 1, 1])
        o = self.mapper(self.backbone(x))
        return o


class Previewer(nn.Module):

    def __init__(self, c_in=16, c_hidden=512, c_out=3):
        super().__init__()
        self.blocks = nn.Sequential(nn.Conv2d(c_in, c_hidden, kernel_size=1), nn.GELU(), nn.BatchNorm2d(c_hidden), nn.Conv2d(c_hidden, c_hidden, kernel_size=3, padding=1), nn.GELU(), nn.BatchNorm2d(c_hidden), nn.ConvTranspose2d(c_hidden, c_hidden // 2, kernel_size=2, stride=2), nn.GELU(), nn.BatchNorm2d(c_hidden // 2), nn.Conv2d(c_hidden // 2, c_hidden // 2, kernel_size=3, padding=1), nn.GELU(), nn.BatchNorm2d(c_hidden // 2), nn.ConvTranspose2d(c_hidden // 2, c_hidden // 4, kernel_size=2, stride=2), nn.GELU(), nn.BatchNorm2d(c_hidden // 4), nn.Conv2d(c_hidden // 4, c_hidden // 4, kernel_size=3, padding=1), nn.GELU(), nn.BatchNorm2d(c_hidden // 4), nn.ConvTranspose2d(c_hidden // 4, c_hidden // 4, kernel_size=2, stride=2), nn.GELU(), nn.BatchNorm2d(c_hidden // 4), nn.Conv2d(c_hidden // 4, c_hidden // 4, kernel_size=3, padding=1), nn.GELU(), nn.BatchNorm2d(c_hidden // 4), nn.Conv2d(c_hidden // 4, c_out, kernel_size=1))

    def forward(self, x):
        return (self.blocks(x) - 0.5) * 2.0


class StageC_coder(nn.Module):

    def __init__(self):
        super().__init__()
        self.previewer = Previewer()
        self.encoder = EfficientNetEncoder()

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.previewer(x)


class MistolineCondDownsamplBlock(nn.Module):

    def __init__(self, dtype=None, device=None, operations=None):
        super().__init__()
        self.encoder = nn.Sequential(operations.Conv2d(3, 16, 3, padding=1, dtype=dtype, device=device), nn.SiLU(), operations.Conv2d(16, 16, 1, dtype=dtype, device=device), nn.SiLU(), operations.Conv2d(16, 16, 3, padding=1, dtype=dtype, device=device), nn.SiLU(), operations.Conv2d(16, 16, 3, padding=1, stride=2, dtype=dtype, device=device), nn.SiLU(), operations.Conv2d(16, 16, 3, padding=1, dtype=dtype, device=device), nn.SiLU(), operations.Conv2d(16, 16, 3, padding=1, stride=2, dtype=dtype, device=device), nn.SiLU(), operations.Conv2d(16, 16, 3, padding=1, dtype=dtype, device=device), nn.SiLU(), operations.Conv2d(16, 16, 3, padding=1, stride=2, dtype=dtype, device=device), nn.SiLU(), operations.Conv2d(16, 16, 1, dtype=dtype, device=device), nn.SiLU(), operations.Conv2d(16, 16, 3, padding=1, dtype=dtype, device=device))

    def forward(self, x):
        return self.encoder(x)


class MistolineControlnetBlock(nn.Module):

    def __init__(self, hidden_size, dtype=None, device=None, operations=None):
        super().__init__()
        self.linear = operations.Linear(hidden_size, hidden_size, dtype=dtype, device=device)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.linear(x))


def rope(pos: 'Tensor', dim: 'int', theta: 'int') ->Tensor:
    assert dim % 2 == 0
    if comfy.model_management.is_device_mps(pos.device) or comfy.model_management.is_intel_xpu():
        device = torch.device('cpu')
    else:
        device = pos.device
    scale = torch.linspace(0, (dim - 2) / dim, steps=dim // 2, dtype=torch.float64, device=device)
    omega = 1.0 / theta ** scale
    out = torch.einsum('...n,d->...nd', pos, omega)
    out = torch.stack([torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1)
    out = rearrange(out, 'b n d (i j) -> b n d i j', i=2, j=2)
    return out


class EmbedND(nn.Module):

    def __init__(self, dim: 'int', theta: 'int', axes_dim: 'list'):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids: 'Tensor') ->Tensor:
        n_axes = ids.shape[-1]
        emb = torch.cat([rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(n_axes)], dim=-3)
        return emb.unsqueeze(1)


class MLPEmbedder(nn.Module):

    def __init__(self, in_dim: 'int', hidden_dim: 'int', dtype=None, device=None, operations=None):
        super().__init__()
        self.in_layer = operations.Linear(in_dim, hidden_dim, bias=True, dtype=dtype, device=device)
        self.silu = nn.SiLU()
        self.out_layer = operations.Linear(hidden_dim, hidden_dim, bias=True, dtype=dtype, device=device)

    def forward(self, x: 'Tensor') ->Tensor:
        return self.out_layer(self.silu(self.in_layer(x)))


class QKNorm(torch.nn.Module):

    def __init__(self, dim: 'int', dtype=None, device=None, operations=None):
        super().__init__()
        self.query_norm = RMSNorm(dim, dtype=dtype, device=device, operations=operations)
        self.key_norm = RMSNorm(dim, dtype=dtype, device=device, operations=operations)

    def forward(self, q: 'Tensor', k: 'Tensor', v: 'Tensor') ->tuple:
        q = self.query_norm(q)
        k = self.key_norm(k)
        return q, k


class Modulation(nn.Module):

    def __init__(self, dim: 'int', double: 'bool', dtype=None, device=None, operations=None):
        super().__init__()
        self.is_double = double
        self.multiplier = 6 if double else 3
        self.lin = operations.Linear(dim, self.multiplier * dim, bias=True, dtype=dtype, device=device)

    def forward(self, vec: 'Tensor') ->tuple:
        out = self.lin(nn.functional.silu(vec))[:, None, :].chunk(self.multiplier, dim=-1)
        return ModulationOut(*out[:3]), ModulationOut(*out[3:]) if self.is_double else None


def apply_rope(xq: 'Tensor', xk: 'Tensor', freqs_cis: 'Tensor'):
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
    xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
    xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
    return xq_out.reshape(*xq.shape).type_as(xq), xk_out.reshape(*xk.shape).type_as(xk)


def attention(q: 'Tensor', k: 'Tensor', v: 'Tensor', pe: 'Tensor') ->Tensor:
    q, k = apply_rope(q, k, pe)
    heads = q.shape[1]
    x = optimized_attention(q, k, v, heads, skip_reshape=True)
    return x


class DoubleStreamBlock(nn.Module):

    def __init__(self, hidden_size: 'int', num_heads: 'int', mlp_ratio: 'float', qkv_bias: 'bool'=False, dtype=None, device=None, operations=None):
        super().__init__()
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.img_mod = Modulation(hidden_size, double=True, dtype=dtype, device=device, operations=operations)
        self.img_norm1 = operations.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-06, dtype=dtype, device=device)
        self.img_attn = SelfAttention(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias, dtype=dtype, device=device, operations=operations)
        self.img_norm2 = operations.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-06, dtype=dtype, device=device)
        self.img_mlp = nn.Sequential(operations.Linear(hidden_size, mlp_hidden_dim, bias=True, dtype=dtype, device=device), nn.GELU(approximate='tanh'), operations.Linear(mlp_hidden_dim, hidden_size, bias=True, dtype=dtype, device=device))
        self.txt_mod = Modulation(hidden_size, double=True, dtype=dtype, device=device, operations=operations)
        self.txt_norm1 = operations.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-06, dtype=dtype, device=device)
        self.txt_attn = SelfAttention(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias, dtype=dtype, device=device, operations=operations)
        self.txt_norm2 = operations.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-06, dtype=dtype, device=device)
        self.txt_mlp = nn.Sequential(operations.Linear(hidden_size, mlp_hidden_dim, bias=True, dtype=dtype, device=device), nn.GELU(approximate='tanh'), operations.Linear(mlp_hidden_dim, hidden_size, bias=True, dtype=dtype, device=device))

    def forward(self, img: 'Tensor', txt: 'Tensor', vec: 'Tensor', pe: 'Tensor'):
        img_mod1, img_mod2 = self.img_mod(vec)
        txt_mod1, txt_mod2 = self.txt_mod(vec)
        img_modulated = self.img_norm1(img)
        img_modulated = (1 + img_mod1.scale) * img_modulated + img_mod1.shift
        img_qkv = self.img_attn.qkv(img_modulated)
        img_q, img_k, img_v = img_qkv.view(img_qkv.shape[0], img_qkv.shape[1], 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        img_q, img_k = self.img_attn.norm(img_q, img_k, img_v)
        txt_modulated = self.txt_norm1(txt)
        txt_modulated = (1 + txt_mod1.scale) * txt_modulated + txt_mod1.shift
        txt_qkv = self.txt_attn.qkv(txt_modulated)
        txt_q, txt_k, txt_v = txt_qkv.view(txt_qkv.shape[0], txt_qkv.shape[1], 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        txt_q, txt_k = self.txt_attn.norm(txt_q, txt_k, txt_v)
        attn = attention(torch.cat((txt_q, img_q), dim=2), torch.cat((txt_k, img_k), dim=2), torch.cat((txt_v, img_v), dim=2), pe=pe)
        txt_attn, img_attn = attn[:, :txt.shape[1]], attn[:, txt.shape[1]:]
        img = img + img_mod1.gate * self.img_attn.proj(img_attn)
        img = img + img_mod2.gate * self.img_mlp((1 + img_mod2.scale) * self.img_norm2(img) + img_mod2.shift)
        txt += txt_mod1.gate * self.txt_attn.proj(txt_attn)
        txt += txt_mod2.gate * self.txt_mlp((1 + txt_mod2.scale) * self.txt_norm2(txt) + txt_mod2.shift)
        if txt.dtype == torch.float16:
            txt = torch.nan_to_num(txt, nan=0.0, posinf=65504, neginf=-65504)
        return img, txt


class SingleStreamBlock(nn.Module):
    """
    A DiT block with parallel linear layers as described in
    https://arxiv.org/abs/2302.05442 and adapted modulation interface.
    """

    def __init__(self, hidden_size: 'int', num_heads: 'int', mlp_ratio: 'float'=4.0, qk_scale: 'float'=None, dtype=None, device=None, operations=None):
        super().__init__()
        self.hidden_dim = hidden_size
        self.num_heads = num_heads
        head_dim = hidden_size // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.linear1 = operations.Linear(hidden_size, hidden_size * 3 + self.mlp_hidden_dim, dtype=dtype, device=device)
        self.linear2 = operations.Linear(hidden_size + self.mlp_hidden_dim, hidden_size, dtype=dtype, device=device)
        self.norm = QKNorm(head_dim, dtype=dtype, device=device, operations=operations)
        self.hidden_size = hidden_size
        self.pre_norm = operations.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-06, dtype=dtype, device=device)
        self.mlp_act = nn.GELU(approximate='tanh')
        self.modulation = Modulation(hidden_size, double=False, dtype=dtype, device=device, operations=operations)

    def forward(self, x: 'Tensor', vec: 'Tensor', pe: 'Tensor') ->Tensor:
        mod, _ = self.modulation(vec)
        x_mod = (1 + mod.scale) * self.pre_norm(x) + mod.shift
        qkv, mlp = torch.split(self.linear1(x_mod), [3 * self.hidden_size, self.mlp_hidden_dim], dim=-1)
        q, k, v = qkv.view(qkv.shape[0], qkv.shape[1], 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k = self.norm(q, k, v)
        attn = attention(q, k, v, pe=pe)
        output = self.linear2(torch.cat((attn, self.mlp_act(mlp)), 2))
        x += mod.gate * output
        if x.dtype == torch.float16:
            x = torch.nan_to_num(x, nan=0.0, posinf=65504, neginf=-65504)
        return x


class LastLayer(nn.Module):

    def __init__(self, hidden_size: 'int', patch_size: 'int', out_channels: 'int', dtype=None, device=None, operations=None):
        super().__init__()
        self.norm_final = operations.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-06, dtype=dtype, device=device)
        self.linear = operations.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True, dtype=dtype, device=device)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), operations.Linear(hidden_size, 2 * hidden_size, bias=True, dtype=dtype, device=device))

    def forward(self, x: 'Tensor', vec: 'Tensor') ->Tensor:
        shift, scale = self.adaLN_modulation(vec).chunk(2, dim=1)
        x = (1 + scale[:, None, :]) * self.norm_final(x) + shift[:, None, :]
        x = self.linear(x)
        return x


def apply_rotary_emb_qk_real(xqk: 'torch.Tensor', freqs_cos: 'torch.Tensor', freqs_sin: 'torch.Tensor') ->torch.Tensor:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor without complex numbers.

    Args:
        xqk (torch.Tensor): Query and/or Key tensors to apply rotary embeddings. Shape: (B, S, *, num_heads, D)
                            Can be either just query or just key, or both stacked along some batch or * dim.
        freqs_cos (torch.Tensor): Precomputed cosine frequency tensor.
        freqs_sin (torch.Tensor): Precomputed sine frequency tensor.

    Returns:
        torch.Tensor: The input tensor with rotary embeddings applied.
    """
    xqk_even = xqk[..., 0::2]
    xqk_odd = xqk[..., 1::2]
    cos_part = (xqk_even * freqs_cos - xqk_odd * freqs_sin).type_as(xqk)
    sin_part = (xqk_even * freqs_sin + xqk_odd * freqs_cos).type_as(xqk)
    out = torch.stack([cos_part, sin_part], dim=-1).flatten(-2)
    return out


def modulated_rmsnorm(x, scale, eps=1e-06):
    x_normed = comfy.ldm.common_dit.rms_norm(x, eps=eps)
    x_modulated = x_normed * (1 + scale.unsqueeze(1))
    return x_modulated


class AsymmetricAttention(nn.Module):

    def __init__(self, dim_x: 'int', dim_y: 'int', num_heads: 'int'=8, qkv_bias: 'bool'=True, qk_norm: 'bool'=False, attn_drop: 'float'=0.0, update_y: 'bool'=True, out_bias: 'bool'=True, attend_to_padding: 'bool'=False, softmax_scale: 'Optional[float]'=None, device: 'Optional[torch.device]'=None, dtype=None, operations=None):
        super().__init__()
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.num_heads = num_heads
        self.head_dim = dim_x // num_heads
        self.attn_drop = attn_drop
        self.update_y = update_y
        self.attend_to_padding = attend_to_padding
        self.softmax_scale = softmax_scale
        if dim_x % num_heads != 0:
            raise ValueError(f'dim_x={dim_x} should be divisible by num_heads={num_heads}')
        self.qkv_bias = qkv_bias
        self.qkv_x = operations.Linear(dim_x, 3 * dim_x, bias=qkv_bias, device=device, dtype=dtype)
        self.qkv_y = operations.Linear(dim_y, 3 * dim_x, bias=qkv_bias, device=device, dtype=dtype)
        assert qk_norm
        self.q_norm_x = RMSNorm(self.head_dim, device=device, dtype=dtype)
        self.k_norm_x = RMSNorm(self.head_dim, device=device, dtype=dtype)
        self.q_norm_y = RMSNorm(self.head_dim, device=device, dtype=dtype)
        self.k_norm_y = RMSNorm(self.head_dim, device=device, dtype=dtype)
        self.proj_x = operations.Linear(dim_x, dim_x, bias=out_bias, device=device, dtype=dtype)
        self.proj_y = operations.Linear(dim_x, dim_y, bias=out_bias, device=device, dtype=dtype) if update_y else nn.Identity()

    def forward(self, x: 'torch.Tensor', y: 'torch.Tensor', scale_x: 'torch.Tensor', scale_y: 'torch.Tensor', crop_y, **rope_rotation) ->Tuple[torch.Tensor, torch.Tensor]:
        rope_cos = rope_rotation.get('rope_cos')
        rope_sin = rope_rotation.get('rope_sin')
        x = modulated_rmsnorm(x, scale_x)
        y = modulated_rmsnorm(y, scale_y)
        q_y, k_y, v_y = self.qkv_y(y).view(y.shape[0], y.shape[1], 3, self.num_heads, -1).unbind(2)
        q_y = self.q_norm_y(q_y)
        k_y = self.k_norm_y(k_y)
        q_x, k_x, v_x = self.qkv_x(x).view(x.shape[0], x.shape[1], 3, self.num_heads, -1).unbind(2)
        q_x = self.q_norm_x(q_x)
        q_x = apply_rotary_emb_qk_real(q_x, rope_cos, rope_sin)
        k_x = self.k_norm_x(k_x)
        k_x = apply_rotary_emb_qk_real(k_x, rope_cos, rope_sin)
        q = torch.cat([q_x, q_y[:, :crop_y]], dim=1).transpose(1, 2)
        k = torch.cat([k_x, k_y[:, :crop_y]], dim=1).transpose(1, 2)
        v = torch.cat([v_x, v_y[:, :crop_y]], dim=1).transpose(1, 2)
        xy = optimized_attention(q, k, v, self.num_heads, skip_reshape=True)
        x, y = torch.tensor_split(xy, (q_x.shape[1],), dim=1)
        x = self.proj_x(x)
        o = torch.zeros(y.shape[0], q_y.shape[1], y.shape[-1], device=y.device, dtype=y.dtype)
        o[:, :y.shape[1]] = y
        y = self.proj_y(o)
        return x, y


def residual_tanh_gated_rmsnorm(x, x_res, gate, eps=1e-06):
    tanh_gate = torch.tanh(gate).unsqueeze(1)
    x_normed = comfy.ldm.common_dit.rms_norm(x_res, eps=eps) * tanh_gate
    output = x + x_normed
    return output


class AsymmetricJointBlock(nn.Module):

    def __init__(self, hidden_size_x: 'int', hidden_size_y: 'int', num_heads: 'int', *, mlp_ratio_x: float=8.0, mlp_ratio_y: float=4.0, update_y: bool=True, device: Optional[torch.device]=None, dtype=None, operations=None, **block_kwargs):
        super().__init__()
        self.update_y = update_y
        self.hidden_size_x = hidden_size_x
        self.hidden_size_y = hidden_size_y
        self.mod_x = operations.Linear(hidden_size_x, 4 * hidden_size_x, device=device, dtype=dtype)
        if self.update_y:
            self.mod_y = operations.Linear(hidden_size_x, 4 * hidden_size_y, device=device, dtype=dtype)
        else:
            self.mod_y = operations.Linear(hidden_size_x, hidden_size_y, device=device, dtype=dtype)
        self.attn = AsymmetricAttention(hidden_size_x, hidden_size_y, num_heads=num_heads, update_y=update_y, device=device, dtype=dtype, operations=operations, **block_kwargs)
        mlp_hidden_dim_x = int(hidden_size_x * mlp_ratio_x)
        assert mlp_hidden_dim_x == int(1536 * 8)
        self.mlp_x = FeedForward(in_features=hidden_size_x, hidden_size=mlp_hidden_dim_x, multiple_of=256, ffn_dim_multiplier=None, device=device, dtype=dtype, operations=operations)
        if self.update_y:
            mlp_hidden_dim_y = int(hidden_size_y * mlp_ratio_y)
            self.mlp_y = FeedForward(in_features=hidden_size_y, hidden_size=mlp_hidden_dim_y, multiple_of=256, ffn_dim_multiplier=None, device=device, dtype=dtype, operations=operations)

    def forward(self, x: 'torch.Tensor', c: 'torch.Tensor', y: 'torch.Tensor', **attn_kwargs):
        """Forward pass of a block.

        Args:
            x: (B, N, dim) tensor of visual tokens
            c: (B, dim) tensor of conditioned features
            y: (B, L, dim) tensor of text tokens
            num_frames: Number of frames in the video. N = num_frames * num_spatial_tokens

        Returns:
            x: (B, N, dim) tensor of visual tokens after block
            y: (B, L, dim) tensor of text tokens after block
        """
        N = x.size(1)
        c = F.silu(c)
        mod_x = self.mod_x(c)
        scale_msa_x, gate_msa_x, scale_mlp_x, gate_mlp_x = mod_x.chunk(4, dim=1)
        mod_y = self.mod_y(c)
        if self.update_y:
            scale_msa_y, gate_msa_y, scale_mlp_y, gate_mlp_y = mod_y.chunk(4, dim=1)
        else:
            scale_msa_y = mod_y
        x_attn, y_attn = self.attn(x, y, scale_x=scale_msa_x, scale_y=scale_msa_y, **attn_kwargs)
        assert x_attn.size(1) == N
        x = residual_tanh_gated_rmsnorm(x, x_attn, gate_msa_x)
        if self.update_y:
            y = residual_tanh_gated_rmsnorm(y, y_attn, gate_msa_y)
        x = self.ff_block_x(x, scale_mlp_x, gate_mlp_x)
        if self.update_y:
            y = self.ff_block_y(y, scale_mlp_y, gate_mlp_y)
        return x, y

    def ff_block_x(self, x, scale_x, gate_x):
        x_mod = modulated_rmsnorm(x, scale_x)
        x_res = self.mlp_x(x_mod)
        x = residual_tanh_gated_rmsnorm(x, x_res, gate_x)
        return x

    def ff_block_y(self, y, scale_y, gate_y):
        y_mod = modulated_rmsnorm(y, scale_y)
        y_res = self.mlp_y(y_mod)
        y = residual_tanh_gated_rmsnorm(y, y_res, gate_y)
        return y


class AttentionPool(nn.Module):

    def __init__(self, spacial_dim: 'int', embed_dim: 'int', num_heads: 'int', output_dim: 'int'=None, dtype=None, device=None, operations=None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.empty(spacial_dim + 1, embed_dim, dtype=dtype, device=device))
        self.k_proj = operations.Linear(embed_dim, embed_dim, dtype=dtype, device=device)
        self.q_proj = operations.Linear(embed_dim, embed_dim, dtype=dtype, device=device)
        self.v_proj = operations.Linear(embed_dim, embed_dim, dtype=dtype, device=device)
        self.c_proj = operations.Linear(embed_dim, output_dim or embed_dim, dtype=dtype, device=device)
        self.num_heads = num_heads
        self.embed_dim = embed_dim

    def forward(self, x):
        x = x[:, :self.positional_embedding.shape[0] - 1]
        x = x.permute(1, 0, 2)
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)
        x = x + comfy.ops.cast_to_input(self.positional_embedding[:, None, :], x)
        q = self.q_proj(x[:1])
        k = self.k_proj(x)
        v = self.v_proj(x)
        batch_size = q.shape[1]
        head_dim = self.embed_dim // self.num_heads
        q = q.view(1, batch_size * self.num_heads, head_dim).transpose(0, 1).view(batch_size, self.num_heads, -1, head_dim)
        k = k.view(k.shape[0], batch_size * self.num_heads, head_dim).transpose(0, 1).view(batch_size, self.num_heads, -1, head_dim)
        v = v.view(v.shape[0], batch_size * self.num_heads, head_dim).transpose(0, 1).view(batch_size, self.num_heads, -1, head_dim)
        attn_output = optimized_attention(q, k, v, self.num_heads, skip_reshape=True).transpose(0, 1)
        attn_output = self.c_proj(attn_output)
        return attn_output.squeeze(0)


def compute_mixed_rotation(freqs: 'torch.Tensor', pos: 'torch.Tensor'):
    """
    Project each 3-dim position into per-head, per-head-dim 1D frequencies.

    Args:
        freqs: [3, num_heads, num_freqs] - learned rotation frequency (for t, row, col) for each head position
        pos: [N, 3] - position of each token
        num_heads: int

    Returns:
        freqs_cos: [N, num_heads, num_freqs] - cosine components
        freqs_sin: [N, num_heads, num_freqs] - sine components
    """
    assert freqs.ndim == 3
    freqs_sum = torch.einsum('Nd,dhf->Nhf', pos, freqs)
    freqs_cos = torch.cos(freqs_sum)
    freqs_sin = torch.sin(freqs_sum)
    return freqs_cos, freqs_sin


def centers(start: 'float', stop, num, dtype=None, device=None):
    """linspace through bin centers.

    Args:
        start (float): Start of the range.
        stop (float): End of the range.
        num (int): Number of points.
        dtype (torch.dtype): Data type of the points.
        device (torch.device): Device of the points.

    Returns:
        centers (Tensor): Centers of the bins. Shape: (num,).
    """
    edges = torch.linspace(start, stop, num + 1, dtype=dtype, device=device)
    return (edges[:-1] + edges[1:]) / 2


def create_position_matrix(T: 'int', pH: 'int', pW: 'int', device: 'torch.device', dtype: 'torch.dtype', *, target_area: float=36864):
    """
    Args:
        T: int - Temporal dimension
        pH: int - Height dimension after patchify
        pW: int - Width dimension after patchify

    Returns:
        pos: [T * pH * pW, 3] - position matrix
    """
    t = torch.arange(T, dtype=dtype)
    scale = math.sqrt(target_area / (pW * pH))
    w = centers(-pW * scale / 2, pW * scale / 2, pW)
    h = centers(-pH * scale / 2, pH * scale / 2, pH)
    grid_t, grid_h, grid_w = torch.meshgrid(t, h, w, indexing='ij')
    pos = torch.stack([grid_t, grid_h, grid_w], dim=-1)
    pos = pos.view(-1, 3)
    pos = pos
    return pos


class DepthToSpaceTime(nn.Module):

    def __init__(self, temporal_expansion: 'int', spatial_expansion: 'int'):
        super().__init__()
        self.temporal_expansion = temporal_expansion
        self.spatial_expansion = spatial_expansion

    def extra_repr(self):
        return f'texp={self.temporal_expansion}, sexp={self.spatial_expansion}'

    def forward(self, x: 'torch.Tensor'):
        """Forward pass.

        Args:
            x: Input tensor. Shape: [B, C, T, H, W].

        Returns:
            x: Rearranged tensor. Shape: [B, C/(st*s*s), T*st, H*s, W*s].
        """
        x = rearrange(x, 'B (C st sh sw) T H W -> B C (T st) (H sh) (W sw)', st=self.temporal_expansion, sh=self.spatial_expansion, sw=self.spatial_expansion)
        if self.temporal_expansion > 1:
            assert all(x.shape)
            x = x[:, :, self.temporal_expansion - 1:]
            assert all(x.shape)
        return x


def norm_fn(in_channels: 'int', affine: 'bool'=True):
    return GroupNormSpatial(affine=affine, num_groups=32, num_channels=in_channels)


class AttentionBlock(nn.Module):

    def __init__(self, dim: 'int', **attn_kwargs) ->None:
        super().__init__()
        self.norm = norm_fn(dim)
        self.attn = Attention(dim, **attn_kwargs)

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        return x + self.attn(self.norm(x))


def block_fn(channels, *, affine: bool=True, has_attention: bool=False, **block_kwargs):
    attn_block = AttentionBlock(channels) if has_attention else None
    return ResBlock(channels, affine=affine, attn_block=attn_block, **block_kwargs)


class CausalUpsampleBlock(nn.Module):

    def __init__(self, in_channels: 'int', out_channels: 'int', num_res_blocks: 'int', *, temporal_expansion: int=2, spatial_expansion: int=2, **block_kwargs):
        super().__init__()
        blocks = []
        for _ in range(num_res_blocks):
            blocks.append(block_fn(in_channels, **block_kwargs))
        self.blocks = nn.Sequential(*blocks)
        self.temporal_expansion = temporal_expansion
        self.spatial_expansion = spatial_expansion
        self.proj = Conv1x1(in_channels, out_channels * temporal_expansion * spatial_expansion ** 2)
        self.d2st = DepthToSpaceTime(temporal_expansion=temporal_expansion, spatial_expansion=spatial_expansion)

    def forward(self, x):
        x = self.blocks(x)
        x = self.proj(x)
        x = self.d2st(x)
        return x


def cast_tuple(t, length=1):
    return t if isinstance(t, tuple) else (t,) * length


class DownsampleBlock(nn.Module):

    def __init__(self, in_channels: 'int', out_channels: 'int', num_res_blocks, *, temporal_reduction=2, spatial_reduction=2, **block_kwargs):
        """
        Downsample block for the VAE encoder.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            num_res_blocks: Number of residual blocks.
            temporal_reduction: Temporal reduction factor.
            spatial_reduction: Spatial reduction factor.
        """
        super().__init__()
        layers = []
        assert in_channels != out_channels
        layers.append(PConv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=(temporal_reduction, spatial_reduction, spatial_reduction), stride=(temporal_reduction, spatial_reduction, spatial_reduction), padding_mode='replicate', bias=block_kwargs['bias']))
        for _ in range(num_res_blocks):
            layers.append(block_fn(out_channels, **block_kwargs))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


def conv(n_in, n_out, **kwargs):
    return comfy.ops.disable_weight_init.Conv2d(n_in, n_out, 3, padding=1, **kwargs)


class Block(nn.Module):

    def __init__(self, n_in, n_out):
        super().__init__()
        self.conv = nn.Sequential(conv(n_in, n_out), nn.ReLU(), conv(n_out, n_out), nn.ReLU(), conv(n_out, n_out))
        self.skip = comfy.ops.disable_weight_init.Conv2d(n_in, n_out, 1, bias=False) if n_in != n_out else nn.Identity()
        self.fuse = nn.ReLU()

    def forward(self, x):
        return self.fuse(self.conv(x) + self.skip(x))


class Clamp(nn.Module):

    def forward(self, x):
        return torch.tanh(x / 3) * 3


def Decoder(latent_channels=4):
    return nn.Sequential(Clamp(), conv(latent_channels, 64), nn.ReLU(), Block(64, 64), Block(64, 64), Block(64, 64), nn.Upsample(scale_factor=2), conv(64, 64, bias=False), Block(64, 64), Block(64, 64), Block(64, 64), nn.Upsample(scale_factor=2), conv(64, 64, bias=False), Block(64, 64), Block(64, 64), Block(64, 64), nn.Upsample(scale_factor=2), conv(64, 64, bias=False), Block(64, 64), conv(64, 3))


def Encoder(latent_channels=4):
    return nn.Sequential(conv(3, 64), Block(64, 64), conv(64, 64, stride=2, bias=False), Block(64, 64), Block(64, 64), Block(64, 64), conv(64, 64, stride=2, bias=False), Block(64, 64), Block(64, 64), Block(64, 64), conv(64, 64, stride=2, bias=False), Block(64, 64), Block(64, 64), Block(64, 64), conv(64, latent_channels))


class VideoVAE(nn.Module):

    def __init__(self):
        super().__init__()
        self.encoder = Encoder(in_channels=15, base_channels=64, channel_multipliers=[1, 2, 4, 6], num_res_blocks=[3, 3, 4, 6, 3], latent_dim=12, temporal_reductions=[1, 2, 3], spatial_reductions=[2, 2, 2], prune_bottlenecks=[False, False, False, False, False], has_attentions=[False, True, True, True, True], affine=True, bias=True, input_is_conv_1x1=True, padding_mode='replicate')
        self.decoder = Decoder(out_channels=3, base_channels=128, channel_multipliers=[1, 2, 4, 6], temporal_expansions=[1, 2, 3], spatial_expansions=[2, 2, 2], num_res_blocks=[3, 3, 4, 6, 3], latent_dim=12, has_attention=[False, False, False, False, False], padding_mode='replicate', output_norm=False, nonlinearity='silu', output_nonlinearity='silu', causal=True)

    def encode(self, x):
        return self.encoder(x).mode()

    def decode(self, x):
        return self.decoder(x)


class HunYuanDiTBlock(nn.Module):
    """
    A HunYuanDiT block with `add` conditioning.
    """

    def __init__(self, hidden_size, c_emb_size, num_heads, mlp_ratio=4.0, text_states_dim=1024, qk_norm=False, norm_type='layer', skip=False, attn_precision=None, dtype=None, device=None, operations=None):
        super().__init__()
        use_ele_affine = True
        if norm_type == 'layer':
            norm_layer = operations.LayerNorm
        elif norm_type == 'rms':
            norm_layer = RMSNorm
        else:
            raise ValueError(f'Unknown norm_type: {norm_type}')
        self.norm1 = norm_layer(hidden_size, elementwise_affine=use_ele_affine, eps=1e-06, dtype=dtype, device=device)
        self.attn1 = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, qk_norm=qk_norm, attn_precision=attn_precision, dtype=dtype, device=device, operations=operations)
        self.norm2 = norm_layer(hidden_size, elementwise_affine=use_ele_affine, eps=1e-06, dtype=dtype, device=device)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda : nn.GELU(approximate='tanh')
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0, dtype=dtype, device=device, operations=operations)
        self.default_modulation = nn.Sequential(nn.SiLU(), operations.Linear(c_emb_size, hidden_size, bias=True, dtype=dtype, device=device))
        self.attn2 = CrossAttention(hidden_size, text_states_dim, num_heads=num_heads, qkv_bias=True, qk_norm=qk_norm, attn_precision=attn_precision, dtype=dtype, device=device, operations=operations)
        self.norm3 = norm_layer(hidden_size, elementwise_affine=True, eps=1e-06, dtype=dtype, device=device)
        if skip:
            self.skip_norm = norm_layer(2 * hidden_size, elementwise_affine=True, eps=1e-06, dtype=dtype, device=device)
            self.skip_linear = operations.Linear(2 * hidden_size, hidden_size, dtype=dtype, device=device)
        else:
            self.skip_linear = None
        self.gradient_checkpointing = False

    def _forward(self, x, c=None, text_states=None, freq_cis_img=None, skip=None):
        if self.skip_linear is not None:
            cat = torch.cat([x, skip], dim=-1)
            if cat.dtype != x.dtype:
                cat = cat
            cat = self.skip_norm(cat)
            x = self.skip_linear(cat)
        shift_msa = self.default_modulation(c).unsqueeze(dim=1)
        attn_inputs = self.norm1(x) + shift_msa, freq_cis_img
        x = x + self.attn1(*attn_inputs)[0]
        cross_inputs = self.norm3(x), text_states, freq_cis_img
        x = x + self.attn2(*cross_inputs)[0]
        mlp_inputs = self.norm2(x)
        x = x + self.mlp(mlp_inputs)
        return x

    def forward(self, x, c=None, text_states=None, freq_cis_img=None, skip=None):
        if self.gradient_checkpointing and self.training:
            return checkpoint.checkpoint(self._forward, x, c, text_states, freq_cis_img, skip)
        return self._forward(x, c, text_states, freq_cis_img, skip)


def get_1d_rotary_pos_embed(dim: 'int', pos: 'Union[np.ndarray, int]', theta: 'float'=10000.0, use_real=False):
    """
    Precompute the frequency tensor for complex exponentials (cis) with given dimensions.

    This function calculates a frequency tensor with complex exponentials using the given dimension 'dim'
    and the end index 'end'. The 'theta' parameter scales the frequencies.
    The returned tensor contains complex values in complex64 data type.

    Args:
        dim (int): Dimension of the frequency tensor.
        pos (np.ndarray, int): Position indices for the frequency tensor. [S] or scalar
        theta (float, optional): Scaling factor for frequency computation. Defaults to 10000.0.
        use_real (bool, optional): If True, return real part and imaginary part separately.
                                   Otherwise, return complex numbers.

    Returns:
        torch.Tensor: Precomputed frequency tensor with complex exponentials. [S, D/2]

    """
    if isinstance(pos, int):
        pos = np.arange(pos)
    freqs = 1.0 / theta ** (torch.arange(0, dim, 2)[:dim // 2].float() / dim)
    t = torch.from_numpy(pos)
    freqs = torch.outer(t, freqs).float()
    if use_real:
        freqs_cos = freqs.cos().repeat_interleave(2, dim=1)
        freqs_sin = freqs.sin().repeat_interleave(2, dim=1)
        return freqs_cos, freqs_sin
    else:
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
        return freqs_cis


def get_2d_rotary_pos_embed_from_grid(embed_dim, grid, use_real=False):
    assert embed_dim % 4 == 0
    emb_h = get_1d_rotary_pos_embed(embed_dim // 2, grid[0].reshape(-1), use_real=use_real)
    emb_w = get_1d_rotary_pos_embed(embed_dim // 2, grid[1].reshape(-1), use_real=use_real)
    if use_real:
        cos = torch.cat([emb_h[0], emb_w[0]], dim=1)
        sin = torch.cat([emb_h[1], emb_w[1]], dim=1)
        return cos, sin
    else:
        emb = torch.cat([emb_h, emb_w], dim=1)
        return emb


def _to_tuple(x):
    if isinstance(x, int):
        return x, x
    else:
        return x


def get_meshgrid(start, *args):
    if len(args) == 0:
        num = _to_tuple(start)
        start = 0, 0
        stop = num
    elif len(args) == 1:
        start = _to_tuple(start)
        stop = _to_tuple(args[0])
        num = stop[0] - start[0], stop[1] - start[1]
    elif len(args) == 2:
        start = _to_tuple(start)
        stop = _to_tuple(args[0])
        num = _to_tuple(args[1])
    else:
        raise ValueError(f'len(args) should be 0, 1 or 2, but got {len(args)}')
    grid_h = np.linspace(start[0], stop[0], num[0], endpoint=False, dtype=np.float32)
    grid_w = np.linspace(start[1], stop[1], num[1], endpoint=False, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0)
    return grid


def get_2d_rotary_pos_embed(embed_dim, start, *args, use_real=True):
    """
    This is a 2d version of precompute_freqs_cis, which is a RoPE for image tokens with 2d structure.

    Parameters
    ----------
    embed_dim: int
        embedding dimension size
    start: int or tuple of int
        If len(args) == 0, start is num; If len(args) == 1, start is start, args[0] is stop, step is 1;
        If len(args) == 2, start is start, args[0] is stop, args[1] is num.
    use_real: bool
        If True, return real part and imaginary part separately. Otherwise, return complex numbers.

    Returns
    -------
    pos_embed: torch.Tensor
        [HW, D/2]
    """
    grid = get_meshgrid(start, *args)
    grid = grid.reshape([2, 1, *grid.shape[1:]])
    pos_embed = get_2d_rotary_pos_embed_from_grid(embed_dim, grid, use_real=use_real)
    return pos_embed


def get_fill_resize_and_crop(src, tgt):
    th, tw = _to_tuple(tgt)
    h, w = _to_tuple(src)
    tr = th / tw
    r = h / w
    if r > tr:
        resize_height = th
        resize_width = int(round(th / h * w))
    else:
        resize_width = tw
        resize_height = int(round(tw / w * h))
    crop_top = int(round((th - resize_height) / 2.0))
    crop_left = int(round((tw - resize_width) / 2.0))
    return (crop_top, crop_left), (crop_top + resize_height, crop_left + resize_width)


def calc_rope(x, patch_size, head_size):
    th = (x.shape[2] + patch_size // 2) // patch_size
    tw = (x.shape[3] + patch_size // 2) // patch_size
    base_size = 512 // 8 // patch_size
    start, stop = get_fill_resize_and_crop((th, tw), base_size)
    sub_args = [start, stop, (th, tw)]
    rope = get_2d_rotary_pos_embed(head_size, *sub_args)
    rope = rope[0], rope[1]
    return rope


class HunYuanControlNet(nn.Module):
    """
    HunYuanDiT: Diffusion model with a Transformer backbone.

    Inherit ModelMixin and ConfigMixin to be compatible with the sampler StableDiffusionPipeline of diffusers.

    Inherit PeftAdapterMixin to be compatible with the PEFT training pipeline.

    Parameters
    ----------
    args: argparse.Namespace
        The arguments parsed by argparse.
    input_size: tuple
        The size of the input image.
    patch_size: int
        The size of the patch.
    in_channels: int
        The number of input channels.
    hidden_size: int
        The hidden size of the transformer backbone.
    depth: int
        The number of transformer blocks.
    num_heads: int
        The number of attention heads.
    mlp_ratio: float
        The ratio of the hidden size of the MLP in the transformer block.
    log_fn: callable
        The logging function.
    """

    def __init__(self, input_size: 'tuple'=128, patch_size: 'int'=2, in_channels: 'int'=4, hidden_size: 'int'=1408, depth: 'int'=40, num_heads: 'int'=16, mlp_ratio: 'float'=4.3637, text_states_dim=1024, text_states_dim_t5=2048, text_len=77, text_len_t5=256, qk_norm=True, size_cond=False, use_style_cond=False, learn_sigma=True, norm='layer', log_fn: 'callable'=print, attn_precision=None, dtype=None, device=None, operations=None, **kwargs):
        super().__init__()
        self.log_fn = log_fn
        self.depth = depth
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.text_states_dim = text_states_dim
        self.text_states_dim_t5 = text_states_dim_t5
        self.text_len = text_len
        self.text_len_t5 = text_len_t5
        self.size_cond = size_cond
        self.use_style_cond = use_style_cond
        self.norm = norm
        self.dtype = dtype
        self.latent_format = comfy.latent_formats.SDXL
        self.mlp_t5 = nn.Sequential(nn.Linear(self.text_states_dim_t5, self.text_states_dim_t5 * 4, bias=True, dtype=dtype, device=device), nn.SiLU(), nn.Linear(self.text_states_dim_t5 * 4, self.text_states_dim, bias=True, dtype=dtype, device=device))
        self.text_embedding_padding = nn.Parameter(torch.randn(self.text_len + self.text_len_t5, self.text_states_dim, dtype=dtype, device=device))
        pooler_out_dim = 1024
        self.pooler = AttentionPool(self.text_len_t5, self.text_states_dim_t5, num_heads=8, output_dim=pooler_out_dim, dtype=dtype, device=device, operations=operations)
        self.extra_in_dim = pooler_out_dim
        if self.size_cond:
            self.extra_in_dim += 6 * 256
        if self.use_style_cond:
            self.style_embedder = nn.Embedding(1, hidden_size, dtype=dtype, device=device)
            self.extra_in_dim += hidden_size
        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, dtype=dtype, device=device, operations=operations)
        self.t_embedder = TimestepEmbedder(hidden_size, dtype=dtype, device=device, operations=operations)
        self.extra_embedder = nn.Sequential(operations.Linear(self.extra_in_dim, hidden_size * 4, dtype=dtype, device=device), nn.SiLU(), operations.Linear(hidden_size * 4, hidden_size, bias=True, dtype=dtype, device=device))
        num_patches = self.x_embedder.num_patches
        self.blocks = nn.ModuleList([HunYuanDiTBlock(hidden_size=hidden_size, c_emb_size=hidden_size, num_heads=num_heads, mlp_ratio=mlp_ratio, text_states_dim=self.text_states_dim, qk_norm=qk_norm, norm_type=self.norm, skip=False, attn_precision=attn_precision, dtype=dtype, device=device, operations=operations) for _ in range(19)])
        self.before_proj = operations.Linear(self.hidden_size, self.hidden_size, dtype=dtype, device=device)
        self.after_proj_list = nn.ModuleList([operations.Linear(self.hidden_size, self.hidden_size, dtype=dtype, device=device) for _ in range(len(self.blocks))])

    def forward(self, x, hint, timesteps, context, text_embedding_mask=None, encoder_hidden_states_t5=None, text_embedding_mask_t5=None, image_meta_size=None, style=None, return_dict=False, **kwarg):
        """
        Forward pass of the encoder.

        Parameters
        ----------
        x: torch.Tensor
            (B, D, H, W)
        t: torch.Tensor
            (B)
        encoder_hidden_states: torch.Tensor
            CLIP text embedding, (B, L_clip, D)
        text_embedding_mask: torch.Tensor
            CLIP text embedding mask, (B, L_clip)
        encoder_hidden_states_t5: torch.Tensor
            T5 text embedding, (B, L_t5, D)
        text_embedding_mask_t5: torch.Tensor
            T5 text embedding mask, (B, L_t5)
        image_meta_size: torch.Tensor
            (B, 6)
        style: torch.Tensor
            (B)
        cos_cis_img: torch.Tensor
        sin_cis_img: torch.Tensor
        return_dict: bool
            Whether to return a dictionary.
        """
        condition = hint
        if condition.shape[0] == 1:
            condition = torch.repeat_interleave(condition, x.shape[0], dim=0)
        text_states = context
        text_states_t5 = encoder_hidden_states_t5
        text_states_mask = text_embedding_mask.bool()
        text_states_t5_mask = text_embedding_mask_t5.bool()
        b_t5, l_t5, c_t5 = text_states_t5.shape
        text_states_t5 = self.mlp_t5(text_states_t5.view(-1, c_t5)).view(b_t5, l_t5, -1)
        padding = comfy.ops.cast_to_input(self.text_embedding_padding, text_states)
        text_states[:, -self.text_len:] = torch.where(text_states_mask[:, -self.text_len:].unsqueeze(2), text_states[:, -self.text_len:], padding[:self.text_len])
        text_states_t5[:, -self.text_len_t5:] = torch.where(text_states_t5_mask[:, -self.text_len_t5:].unsqueeze(2), text_states_t5[:, -self.text_len_t5:], padding[self.text_len:])
        text_states = torch.cat([text_states, text_states_t5], dim=1)
        freqs_cis_img = calc_rope(x, self.patch_size, self.hidden_size // self.num_heads)
        t = self.t_embedder(timesteps, dtype=self.dtype)
        x = self.x_embedder(x)
        extra_vec = self.pooler(encoder_hidden_states_t5)
        if style is not None:
            style_embedding = self.style_embedder(style)
            extra_vec = torch.cat([extra_vec, style_embedding], dim=1)
        c = t + self.extra_embedder(extra_vec)
        condition = self.x_embedder(condition)
        controls = []
        x = x + self.before_proj(condition)
        for layer, block in enumerate(self.blocks):
            x = block(x, c, text_states, freqs_cis_img)
            controls.append(self.after_proj_list[layer](x))
        return {'output': controls}


class HunYuanDiT(nn.Module):
    """
    HunYuanDiT: Diffusion model with a Transformer backbone.

    Inherit ModelMixin and ConfigMixin to be compatible with the sampler StableDiffusionPipeline of diffusers.

    Inherit PeftAdapterMixin to be compatible with the PEFT training pipeline.

    Parameters
    ----------
    args: argparse.Namespace
        The arguments parsed by argparse.
    input_size: tuple
        The size of the input image.
    patch_size: int
        The size of the patch.
    in_channels: int
        The number of input channels.
    hidden_size: int
        The hidden size of the transformer backbone.
    depth: int
        The number of transformer blocks.
    num_heads: int
        The number of attention heads.
    mlp_ratio: float
        The ratio of the hidden size of the MLP in the transformer block.
    log_fn: callable
        The logging function.
    """

    def __init__(self, input_size: 'tuple'=32, patch_size: 'int'=2, in_channels: 'int'=4, hidden_size: 'int'=1152, depth: 'int'=28, num_heads: 'int'=16, mlp_ratio: 'float'=4.0, text_states_dim=1024, text_states_dim_t5=2048, text_len=77, text_len_t5=256, qk_norm=True, size_cond=False, use_style_cond=False, learn_sigma=True, norm='layer', log_fn: 'callable'=print, attn_precision=None, dtype=None, device=None, operations=None, **kwargs):
        super().__init__()
        self.log_fn = log_fn
        self.depth = depth
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.text_states_dim = text_states_dim
        self.text_states_dim_t5 = text_states_dim_t5
        self.text_len = text_len
        self.text_len_t5 = text_len_t5
        self.size_cond = size_cond
        self.use_style_cond = use_style_cond
        self.norm = norm
        self.dtype = dtype
        self.mlp_t5 = nn.Sequential(operations.Linear(self.text_states_dim_t5, self.text_states_dim_t5 * 4, bias=True, dtype=dtype, device=device), nn.SiLU(), operations.Linear(self.text_states_dim_t5 * 4, self.text_states_dim, bias=True, dtype=dtype, device=device))
        self.text_embedding_padding = nn.Parameter(torch.empty(self.text_len + self.text_len_t5, self.text_states_dim, dtype=dtype, device=device))
        pooler_out_dim = 1024
        self.pooler = AttentionPool(self.text_len_t5, self.text_states_dim_t5, num_heads=8, output_dim=pooler_out_dim, dtype=dtype, device=device, operations=operations)
        self.extra_in_dim = pooler_out_dim
        if self.size_cond:
            self.extra_in_dim += 6 * 256
        if self.use_style_cond:
            self.style_embedder = operations.Embedding(1, hidden_size, dtype=dtype, device=device)
            self.extra_in_dim += hidden_size
        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, dtype=dtype, device=device, operations=operations)
        self.t_embedder = TimestepEmbedder(hidden_size, dtype=dtype, device=device, operations=operations)
        self.extra_embedder = nn.Sequential(operations.Linear(self.extra_in_dim, hidden_size * 4, dtype=dtype, device=device), nn.SiLU(), operations.Linear(hidden_size * 4, hidden_size, bias=True, dtype=dtype, device=device))
        num_patches = self.x_embedder.num_patches
        self.blocks = nn.ModuleList([HunYuanDiTBlock(hidden_size=hidden_size, c_emb_size=hidden_size, num_heads=num_heads, mlp_ratio=mlp_ratio, text_states_dim=self.text_states_dim, qk_norm=qk_norm, norm_type=self.norm, skip=layer > depth // 2, attn_precision=attn_precision, dtype=dtype, device=device, operations=operations) for layer in range(depth)])
        self.final_layer = FinalLayer(hidden_size, hidden_size, patch_size, self.out_channels, dtype=dtype, device=device, operations=operations)
        self.unpatchify_channels = self.out_channels

    def forward(self, x, t, context, text_embedding_mask=None, encoder_hidden_states_t5=None, text_embedding_mask_t5=None, image_meta_size=None, style=None, return_dict=False, control=None, transformer_options=None):
        """
        Forward pass of the encoder.

        Parameters
        ----------
        x: torch.Tensor
            (B, D, H, W)
        t: torch.Tensor
            (B)
        encoder_hidden_states: torch.Tensor
            CLIP text embedding, (B, L_clip, D)
        text_embedding_mask: torch.Tensor
            CLIP text embedding mask, (B, L_clip)
        encoder_hidden_states_t5: torch.Tensor
            T5 text embedding, (B, L_t5, D)
        text_embedding_mask_t5: torch.Tensor
            T5 text embedding mask, (B, L_t5)
        image_meta_size: torch.Tensor
            (B, 6)
        style: torch.Tensor
            (B)
        cos_cis_img: torch.Tensor
        sin_cis_img: torch.Tensor
        return_dict: bool
            Whether to return a dictionary.
        """
        encoder_hidden_states = context
        text_states = encoder_hidden_states
        text_states_t5 = encoder_hidden_states_t5
        text_states_mask = text_embedding_mask.bool()
        text_states_t5_mask = text_embedding_mask_t5.bool()
        b_t5, l_t5, c_t5 = text_states_t5.shape
        text_states_t5 = self.mlp_t5(text_states_t5.view(-1, c_t5)).view(b_t5, l_t5, -1)
        padding = comfy.ops.cast_to_input(self.text_embedding_padding, text_states)
        text_states[:, -self.text_len:] = torch.where(text_states_mask[:, -self.text_len:].unsqueeze(2), text_states[:, -self.text_len:], padding[:self.text_len])
        text_states_t5[:, -self.text_len_t5:] = torch.where(text_states_t5_mask[:, -self.text_len_t5:].unsqueeze(2), text_states_t5[:, -self.text_len_t5:], padding[self.text_len:])
        text_states = torch.cat([text_states, text_states_t5], dim=1)
        _, _, oh, ow = x.shape
        th, tw = (oh + self.patch_size // 2) // self.patch_size, (ow + self.patch_size // 2) // self.patch_size
        freqs_cis_img = calc_rope(x, self.patch_size, self.hidden_size // self.num_heads)
        t = self.t_embedder(t, dtype=x.dtype)
        x = self.x_embedder(x)
        extra_vec = self.pooler(encoder_hidden_states_t5)
        if self.size_cond:
            image_meta_size = timestep_embedding(image_meta_size.view(-1), 256)
            image_meta_size = image_meta_size.view(-1, 6 * 256)
            extra_vec = torch.cat([extra_vec, image_meta_size], dim=1)
        if self.use_style_cond:
            if style is None:
                style = torch.zeros((extra_vec.shape[0],), device=x.device, dtype=torch.int)
            style_embedding = self.style_embedder(style, out_dtype=x.dtype)
            extra_vec = torch.cat([extra_vec, style_embedding], dim=1)
        c = t + self.extra_embedder(extra_vec)
        controls = None
        if control:
            controls = control.get('output', None)
        skips = []
        for layer, block in enumerate(self.blocks):
            if layer > self.depth // 2:
                if controls is not None:
                    skip = skips.pop() + controls.pop()
                else:
                    skip = skips.pop()
                x = block(x, c, text_states, freqs_cis_img, skip)
            else:
                x = block(x, c, text_states, freqs_cis_img)
            if layer < self.depth // 2 - 1:
                skips.append(x)
        if controls is not None and len(controls) != 0:
            raise ValueError('The number of controls is not equal to the number of skip connections.')
        x = self.final_layer(x, c)
        x = self.unpatchify(x, th, tw)
        if return_dict:
            return {'x': x}
        if self.learn_sigma:
            return x[:, :self.out_channels // 2, :oh, :ow]
        return x[:, :, :oh, :ow]

    def unpatchify(self, x, h, w):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.unpatchify_channels
        p = self.x_embedder.patch_size[0]
        assert h * w == x.shape[1]
        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, w * p))
        return imgs


class DiagonalGaussianDistribution(object):

    def __init__(self, parameters, deterministic=False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean)

    def sample(self):
        x = self.mean + self.std * torch.randn(self.mean.shape)
        return x

    def kl(self, other=None):
        if self.deterministic:
            return torch.Tensor([0.0])
        elif other is None:
            return 0.5 * torch.sum(torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar, dim=[1, 2, 3])
        else:
            return 0.5 * torch.sum(torch.pow(self.mean - other.mean, 2) / other.var + self.var / other.var - 1.0 - self.logvar + other.logvar, dim=[1, 2, 3])

    def nll(self, sample, dims=[1, 2, 3]):
        if self.deterministic:
            return torch.Tensor([0.0])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var, dim=dims)

    def mode(self):
        return self.mean


class DiagonalGaussianRegularizer(torch.nn.Module):

    def __init__(self, sample: 'bool'=True):
        super().__init__()
        self.sample = sample

    def get_trainable_parameters(self) ->Any:
        yield from ()

    def forward(self, z: 'torch.Tensor') ->Tuple[torch.Tensor, dict]:
        log = dict()
        posterior = DiagonalGaussianDistribution(z)
        if self.sample:
            z = posterior.sample()
        else:
            z = posterior.mode()
        kl_loss = posterior.kl()
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
        log['kl_loss'] = kl_loss
        return z, log


class LitEma(nn.Module):

    def __init__(self, model, decay=0.9999, use_num_upates=True):
        super().__init__()
        if decay < 0.0 or decay > 1.0:
            raise ValueError('Decay must be between 0 and 1')
        self.m_name2s_name = {}
        self.register_buffer('decay', torch.tensor(decay, dtype=torch.float32))
        self.register_buffer('num_updates', torch.tensor(0, dtype=torch.int) if use_num_upates else torch.tensor(-1, dtype=torch.int))
        for name, p in model.named_parameters():
            if p.requires_grad:
                s_name = name.replace('.', '')
                self.m_name2s_name.update({name: s_name})
                self.register_buffer(s_name, p.clone().detach().data)
        self.collected_params = []

    def reset_num_updates(self):
        del self.num_updates
        self.register_buffer('num_updates', torch.tensor(0, dtype=torch.int))

    def forward(self, model):
        decay = self.decay
        if self.num_updates >= 0:
            self.num_updates += 1
            decay = min(self.decay, (1 + self.num_updates) / (10 + self.num_updates))
        one_minus_decay = 1.0 - decay
        with torch.no_grad():
            m_param = dict(model.named_parameters())
            shadow_params = dict(self.named_buffers())
            for key in m_param:
                if m_param[key].requires_grad:
                    sname = self.m_name2s_name[key]
                    shadow_params[sname] = shadow_params[sname].type_as(m_param[key])
                    shadow_params[sname].sub_(one_minus_decay * (shadow_params[sname] - m_param[key]))
                else:
                    assert not key in self.m_name2s_name

    def copy_to(self, model):
        m_param = dict(model.named_parameters())
        shadow_params = dict(self.named_buffers())
        for key in m_param:
            if m_param[key].requires_grad:
                m_param[key].data.copy_(shadow_params[self.m_name2s_name[key]].data)
            else:
                assert not key in self.m_name2s_name

    def store(self, parameters):
        """
        Save the current parameters for restoring later.
        Args:
          parameters: Iterable of `torch.nn.Parameter`; the parameters to be
            temporarily stored.
        """
        self.collected_params = [param.clone() for param in parameters]

    def restore(self, parameters):
        """
        Restore the parameters stored with the `store` method.
        Useful to validate the model with EMA parameters without affecting the
        original optimization process. Store the parameters before the
        `copy_to` method. After validation (or model saving), use this to
        restore the former parameters.
        Args:
          parameters: Iterable of `torch.nn.Parameter`; the parameters to be
            updated with the stored parameters.
        """
        for c_param, param in zip(self.collected_params, parameters):
            param.data.copy_(c_param.data)


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit('.', 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config):
    if not 'target' in config:
        if config == '__is_first_stage__':
            return None
        elif config == '__is_unconditional__':
            return None
        raise KeyError('Expected key `target` to instantiate.')
    return get_obj_from_str(config['target'])(**config.get('params', dict()))


class BasicTransformerBlock(nn.Module):

    def __init__(self, dim, n_heads, d_head, dropout=0.0, context_dim=None, gated_ff=True, checkpoint=True, ff_in=False, inner_dim=None, disable_self_attn=False, disable_temporal_crossattention=False, switch_temporal_ca_to_sa=False, attn_precision=None, dtype=None, device=None, operations=ops):
        super().__init__()
        self.ff_in = ff_in or inner_dim is not None
        if inner_dim is None:
            inner_dim = dim
        self.is_res = inner_dim == dim
        self.attn_precision = attn_precision
        if self.ff_in:
            self.norm_in = operations.LayerNorm(dim, dtype=dtype, device=device)
            self.ff_in = FeedForward(dim, dim_out=inner_dim, dropout=dropout, glu=gated_ff, dtype=dtype, device=device, operations=operations)
        self.disable_self_attn = disable_self_attn
        self.attn1 = CrossAttention(query_dim=inner_dim, heads=n_heads, dim_head=d_head, dropout=dropout, context_dim=context_dim if self.disable_self_attn else None, attn_precision=self.attn_precision, dtype=dtype, device=device, operations=operations)
        self.ff = FeedForward(inner_dim, dim_out=dim, dropout=dropout, glu=gated_ff, dtype=dtype, device=device, operations=operations)
        if disable_temporal_crossattention:
            if switch_temporal_ca_to_sa:
                raise ValueError
            else:
                self.attn2 = None
        else:
            context_dim_attn2 = None
            if not switch_temporal_ca_to_sa:
                context_dim_attn2 = context_dim
            self.attn2 = CrossAttention(query_dim=inner_dim, context_dim=context_dim_attn2, heads=n_heads, dim_head=d_head, dropout=dropout, attn_precision=self.attn_precision, dtype=dtype, device=device, operations=operations)
            self.norm2 = operations.LayerNorm(inner_dim, dtype=dtype, device=device)
        self.norm1 = operations.LayerNorm(inner_dim, dtype=dtype, device=device)
        self.norm3 = operations.LayerNorm(inner_dim, dtype=dtype, device=device)
        self.n_heads = n_heads
        self.d_head = d_head
        self.switch_temporal_ca_to_sa = switch_temporal_ca_to_sa

    def forward(self, x, context=None, transformer_options={}):
        extra_options = {}
        block = transformer_options.get('block', None)
        block_index = transformer_options.get('block_index', 0)
        transformer_patches = {}
        transformer_patches_replace = {}
        for k in transformer_options:
            if k == 'patches':
                transformer_patches = transformer_options[k]
            elif k == 'patches_replace':
                transformer_patches_replace = transformer_options[k]
            else:
                extra_options[k] = transformer_options[k]
        extra_options['n_heads'] = self.n_heads
        extra_options['dim_head'] = self.d_head
        extra_options['attn_precision'] = self.attn_precision
        if self.ff_in:
            x_skip = x
            x = self.ff_in(self.norm_in(x))
            if self.is_res:
                x += x_skip
        n = self.norm1(x)
        if self.disable_self_attn:
            context_attn1 = context
        else:
            context_attn1 = None
        value_attn1 = None
        if 'attn1_patch' in transformer_patches:
            patch = transformer_patches['attn1_patch']
            if context_attn1 is None:
                context_attn1 = n
            value_attn1 = context_attn1
            for p in patch:
                n, context_attn1, value_attn1 = p(n, context_attn1, value_attn1, extra_options)
        if block is not None:
            transformer_block = block[0], block[1], block_index
        else:
            transformer_block = None
        attn1_replace_patch = transformer_patches_replace.get('attn1', {})
        block_attn1 = transformer_block
        if block_attn1 not in attn1_replace_patch:
            block_attn1 = block
        if block_attn1 in attn1_replace_patch:
            if context_attn1 is None:
                context_attn1 = n
                value_attn1 = n
            n = self.attn1.to_q(n)
            context_attn1 = self.attn1.to_k(context_attn1)
            value_attn1 = self.attn1.to_v(value_attn1)
            n = attn1_replace_patch[block_attn1](n, context_attn1, value_attn1, extra_options)
            n = self.attn1.to_out(n)
        else:
            n = self.attn1(n, context=context_attn1, value=value_attn1)
        if 'attn1_output_patch' in transformer_patches:
            patch = transformer_patches['attn1_output_patch']
            for p in patch:
                n = p(n, extra_options)
        x += n
        if 'middle_patch' in transformer_patches:
            patch = transformer_patches['middle_patch']
            for p in patch:
                x = p(x, extra_options)
        if self.attn2 is not None:
            n = self.norm2(x)
            if self.switch_temporal_ca_to_sa:
                context_attn2 = n
            else:
                context_attn2 = context
            value_attn2 = None
            if 'attn2_patch' in transformer_patches:
                patch = transformer_patches['attn2_patch']
                value_attn2 = context_attn2
                for p in patch:
                    n, context_attn2, value_attn2 = p(n, context_attn2, value_attn2, extra_options)
            attn2_replace_patch = transformer_patches_replace.get('attn2', {})
            block_attn2 = transformer_block
            if block_attn2 not in attn2_replace_patch:
                block_attn2 = block
            if block_attn2 in attn2_replace_patch:
                if value_attn2 is None:
                    value_attn2 = context_attn2
                n = self.attn2.to_q(n)
                context_attn2 = self.attn2.to_k(context_attn2)
                value_attn2 = self.attn2.to_v(value_attn2)
                n = attn2_replace_patch[block_attn2](n, context_attn2, value_attn2, extra_options)
                n = self.attn2.to_out(n)
            else:
                n = self.attn2(n, context=context_attn2, value=value_attn2)
        if 'attn2_output_patch' in transformer_patches:
            patch = transformer_patches['attn2_output_patch']
            for p in patch:
                n = p(n, extra_options)
        x += n
        if self.is_res:
            x_skip = x
        x = self.ff(self.norm3(x))
        if self.is_res:
            x += x_skip
        return x


class SpatialTransformer(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    NEW: use_linear for more efficiency instead of the 1x1 convs
    """

    def __init__(self, in_channels, n_heads, d_head, depth=1, dropout=0.0, context_dim=None, disable_self_attn=False, use_linear=False, use_checkpoint=True, attn_precision=None, dtype=None, device=None, operations=ops):
        super().__init__()
        if exists(context_dim) and not isinstance(context_dim, list):
            context_dim = [context_dim] * depth
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = operations.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-06, affine=True, dtype=dtype, device=device)
        if not use_linear:
            self.proj_in = operations.Conv2d(in_channels, inner_dim, kernel_size=1, stride=1, padding=0, dtype=dtype, device=device)
        else:
            self.proj_in = operations.Linear(in_channels, inner_dim, dtype=dtype, device=device)
        self.transformer_blocks = nn.ModuleList([BasicTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim[d], disable_self_attn=disable_self_attn, checkpoint=use_checkpoint, attn_precision=attn_precision, dtype=dtype, device=device, operations=operations) for d in range(depth)])
        if not use_linear:
            self.proj_out = operations.Conv2d(inner_dim, in_channels, kernel_size=1, stride=1, padding=0, dtype=dtype, device=device)
        else:
            self.proj_out = operations.Linear(in_channels, inner_dim, dtype=dtype, device=device)
        self.use_linear = use_linear

    def forward(self, x, context=None, transformer_options={}):
        if not isinstance(context, list):
            context = [context] * len(self.transformer_blocks)
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        if not self.use_linear:
            x = self.proj_in(x)
        x = x.movedim(1, 3).flatten(1, 2).contiguous()
        if self.use_linear:
            x = self.proj_in(x)
        for i, block in enumerate(self.transformer_blocks):
            transformer_options['block_index'] = i
            x = block(x, context=context[i], transformer_options=transformer_options)
        if self.use_linear:
            x = self.proj_out(x)
        x = x.reshape(x.shape[0], h, w, x.shape[-1]).movedim(3, 1).contiguous()
        if not self.use_linear:
            x = self.proj_out(x)
        return x + x_in


class AlphaBlender(nn.Module):
    strategies = ['learned', 'fixed', 'learned_with_images']

    def __init__(self, alpha: 'float', merge_strategy: 'str'='learned_with_images', rearrange_pattern: 'str'='b t -> (b t) 1 1'):
        super().__init__()
        self.merge_strategy = merge_strategy
        self.rearrange_pattern = rearrange_pattern
        assert merge_strategy in self.strategies, f'merge_strategy needs to be in {self.strategies}'
        if self.merge_strategy == 'fixed':
            self.register_buffer('mix_factor', torch.Tensor([alpha]))
        elif self.merge_strategy == 'learned' or self.merge_strategy == 'learned_with_images':
            self.register_parameter('mix_factor', torch.nn.Parameter(torch.Tensor([alpha])))
        else:
            raise ValueError(f'unknown merge strategy {self.merge_strategy}')

    def get_alpha(self, image_only_indicator: 'torch.Tensor', device) ->torch.Tensor:
        if self.merge_strategy == 'fixed':
            alpha = self.mix_factor
        elif self.merge_strategy == 'learned':
            alpha = torch.sigmoid(self.mix_factor)
        elif self.merge_strategy == 'learned_with_images':
            if image_only_indicator is None:
                alpha = rearrange(torch.sigmoid(self.mix_factor), '... -> ... 1')
            else:
                alpha = torch.where(image_only_indicator.bool(), torch.ones(1, 1, device=image_only_indicator.device), rearrange(torch.sigmoid(self.mix_factor), '... -> ... 1'))
            alpha = rearrange(alpha, self.rearrange_pattern)
        else:
            raise NotImplementedError()
        return alpha

    def forward(self, x_spatial, x_temporal, image_only_indicator=None) ->torch.Tensor:
        alpha = self.get_alpha(image_only_indicator, x_spatial.device)
        x = alpha * x_spatial + (1.0 - alpha) * x_temporal
        return x


class SpatialVideoTransformer(SpatialTransformer):

    def __init__(self, in_channels, n_heads, d_head, depth=1, dropout=0.0, use_linear=False, context_dim=None, use_spatial_context=False, timesteps=None, merge_strategy: 'str'='fixed', merge_factor: 'float'=0.5, time_context_dim=None, ff_in=False, checkpoint=False, time_depth=1, disable_self_attn=False, disable_temporal_crossattention=False, max_time_embed_period: 'int'=10000, attn_precision=None, dtype=None, device=None, operations=ops):
        super().__init__(in_channels, n_heads, d_head, depth=depth, dropout=dropout, use_checkpoint=checkpoint, context_dim=context_dim, use_linear=use_linear, disable_self_attn=disable_self_attn, attn_precision=attn_precision, dtype=dtype, device=device, operations=operations)
        self.time_depth = time_depth
        self.depth = depth
        self.max_time_embed_period = max_time_embed_period
        time_mix_d_head = d_head
        n_time_mix_heads = n_heads
        time_mix_inner_dim = int(time_mix_d_head * n_time_mix_heads)
        inner_dim = n_heads * d_head
        if use_spatial_context:
            time_context_dim = context_dim
        self.time_stack = nn.ModuleList([BasicTransformerBlock(inner_dim, n_time_mix_heads, time_mix_d_head, dropout=dropout, context_dim=time_context_dim, checkpoint=checkpoint, ff_in=ff_in, inner_dim=time_mix_inner_dim, disable_self_attn=disable_self_attn, disable_temporal_crossattention=disable_temporal_crossattention, attn_precision=attn_precision, dtype=dtype, device=device, operations=operations) for _ in range(self.depth)])
        assert len(self.time_stack) == len(self.transformer_blocks)
        self.use_spatial_context = use_spatial_context
        self.in_channels = in_channels
        time_embed_dim = self.in_channels * 4
        self.time_pos_embed = nn.Sequential(operations.Linear(self.in_channels, time_embed_dim, dtype=dtype, device=device), nn.SiLU(), operations.Linear(time_embed_dim, self.in_channels, dtype=dtype, device=device))
        self.time_mixer = AlphaBlender(alpha=merge_factor, merge_strategy=merge_strategy)

    def forward(self, x: 'torch.Tensor', context: 'Optional[torch.Tensor]'=None, time_context: 'Optional[torch.Tensor]'=None, timesteps: 'Optional[int]'=None, image_only_indicator: 'Optional[torch.Tensor]'=None, transformer_options={}) ->torch.Tensor:
        _, _, h, w = x.shape
        x_in = x
        spatial_context = None
        if exists(context):
            spatial_context = context
        if self.use_spatial_context:
            assert context.ndim == 3, f'n dims of spatial context should be 3 but are {context.ndim}'
            if time_context is None:
                time_context = context
            time_context_first_timestep = time_context[::timesteps]
            time_context = repeat(time_context_first_timestep, 'b ... -> (b n) ...', n=h * w)
        elif time_context is not None and not self.use_spatial_context:
            time_context = repeat(time_context, 'b ... -> (b n) ...', n=h * w)
            if time_context.ndim == 2:
                time_context = rearrange(time_context, 'b c -> b 1 c')
        x = self.norm(x)
        if not self.use_linear:
            x = self.proj_in(x)
        x = rearrange(x, 'b c h w -> b (h w) c')
        if self.use_linear:
            x = self.proj_in(x)
        num_frames = torch.arange(timesteps, device=x.device)
        num_frames = repeat(num_frames, 't -> b t', b=x.shape[0] // timesteps)
        num_frames = rearrange(num_frames, 'b t -> (b t)')
        t_emb = timestep_embedding(num_frames, self.in_channels, repeat_only=False, max_period=self.max_time_embed_period)
        emb = self.time_pos_embed(t_emb)
        emb = emb[:, None, :]
        for it_, (block, mix_block) in enumerate(zip(self.transformer_blocks, self.time_stack)):
            transformer_options['block_index'] = it_
            x = block(x, context=spatial_context, transformer_options=transformer_options)
            x_mix = x
            x_mix = x_mix + emb
            B, S, C = x_mix.shape
            x_mix = rearrange(x_mix, '(b t) s c -> (b s) t c', t=timesteps)
            x_mix = mix_block(x_mix, context=time_context)
            x_mix = rearrange(x_mix, '(b s) t c -> (b t) s c', s=S, b=B // timesteps, c=C, t=timesteps)
            x = self.time_mixer(x_spatial=x, x_temporal=x_mix, image_only_indicator=image_only_indicator)
        if self.use_linear:
            x = self.proj_out(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        if not self.use_linear:
            x = self.proj_out(x)
        out = x + x_in
        return out


class OpenAISignatureMMDITWrapper(MMDiT):

    def forward(self, x: 'torch.Tensor', timesteps: 'torch.Tensor', context: 'Optional[torch.Tensor]'=None, y: 'Optional[torch.Tensor]'=None, control=None, transformer_options={}, **kwargs) ->torch.Tensor:
        return super().forward(x, timesteps, context=context, y=y, control=control, transformer_options=transformer_options)


class ResnetBlock(nn.Module):

    def __init__(self, in_c, out_c, down, ksize=3, sk=False, use_conv=True):
        super().__init__()
        ps = ksize // 2
        if in_c != out_c or sk == False:
            self.in_conv = nn.Conv2d(in_c, out_c, ksize, 1, ps)
        else:
            self.in_conv = None
        self.block1 = nn.Conv2d(out_c, out_c, 3, 1, 1)
        self.act = nn.ReLU()
        self.block2 = nn.Conv2d(out_c, out_c, ksize, 1, ps)
        if sk == False:
            self.skep = nn.Conv2d(in_c, out_c, ksize, 1, ps)
        else:
            self.skep = None
        self.down = down
        if self.down == True:
            self.down_opt = Downsample(in_c, use_conv=use_conv)

    def forward(self, x):
        if self.down == True:
            x = self.down_opt(x)
        if self.in_conv is not None:
            x = self.in_conv(x)
        h = self.block1(x)
        h = self.act(h)
        h = self.block2(h)
        if self.skep is not None:
            return h + self.skep(x)
        else:
            return h + x


def get_timestep_embedding(timesteps, embedding_dim):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models:
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    assert len(timesteps.shape) == 1
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


def make_attn(in_channels, attn_type='vanilla', attn_kwargs=None):
    return AttnBlock(in_channels)


def nonlinearity(x):
    return x * torch.sigmoid(x)


class Model(nn.Module):

    def __init__(self, *, ch, out_ch, ch_mult=(1, 2, 4, 8), num_res_blocks, attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels, resolution, use_timestep=True, use_linear_attn=False, attn_type='vanilla'):
        super().__init__()
        if use_linear_attn:
            attn_type = 'linear'
        self.ch = ch
        self.temb_ch = self.ch * 4
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.use_timestep = use_timestep
        if self.use_timestep:
            self.temb = nn.Module()
            self.temb.dense = nn.ModuleList([ops.Linear(self.ch, self.temb_ch), ops.Linear(self.temb_ch, self.temb_ch)])
        self.conv_in = ops.Conv2d(in_channels, self.ch, kernel_size=3, stride=1, padding=1)
        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out, temb_channels=self.temb_ch, dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=dropout)
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=dropout)
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            skip_in = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                if i_block == self.num_res_blocks:
                    skip_in = ch * in_ch_mult[i_level]
                block.append(ResnetBlock(in_channels=block_in + skip_in, out_channels=block_out, temb_channels=self.temb_ch, dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up)
        self.norm_out = Normalize(block_in)
        self.conv_out = ops.Conv2d(block_in, out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, x, t=None, context=None):
        if context is not None:
            x = torch.cat((x, context), dim=1)
        if self.use_timestep:
            assert t is not None
            temb = get_timestep_embedding(t, self.ch)
            temb = self.temb.dense[0](temb)
            temb = nonlinearity(temb)
            temb = self.temb.dense[1](temb)
        else:
            temb = None
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](torch.cat([h, hs.pop()], dim=1), temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h

    def get_last_layer(self):
        return self.conv_out.weight


class VideoResBlock(ResnetBlock):

    def __init__(self, out_channels, *args, dropout=0.0, video_kernel_size=3, alpha=0.0, merge_strategy='learned', **kwargs):
        super().__init__(*args, out_channels=out_channels, dropout=dropout, **kwargs)
        if video_kernel_size is None:
            video_kernel_size = [3, 1, 1]
        self.time_stack = ResBlock(channels=out_channels, emb_channels=0, dropout=dropout, dims=3, use_scale_shift_norm=False, use_conv=False, up=False, down=False, kernel_size=video_kernel_size, use_checkpoint=False, skip_t_emb=True)
        self.merge_strategy = merge_strategy
        if self.merge_strategy == 'fixed':
            self.register_buffer('mix_factor', torch.Tensor([alpha]))
        elif self.merge_strategy == 'learned':
            self.register_parameter('mix_factor', torch.nn.Parameter(torch.Tensor([alpha])))
        else:
            raise ValueError(f'unknown merge strategy {self.merge_strategy}')

    def get_alpha(self, bs):
        if self.merge_strategy == 'fixed':
            return self.mix_factor
        elif self.merge_strategy == 'learned':
            return torch.sigmoid(self.mix_factor)
        else:
            raise NotImplementedError()

    def forward(self, x, temb, skip_video=False, timesteps=None):
        b, c, h, w = x.shape
        if timesteps is None:
            timesteps = b
        x = super().forward(x, temb)
        if not skip_video:
            x_mix = rearrange(x, '(b t) c h w -> b c t h w', t=timesteps)
            x = rearrange(x, '(b t) c h w -> b c t h w', t=timesteps)
            x = self.time_stack(x, temb)
            alpha = self.get_alpha(bs=b // timesteps)
            x = alpha * x + (1.0 - alpha) * x_mix
            x = rearrange(x, 'b c t h w -> (b t) c h w')
        return x


def forward_timestep_embed(ts, x, emb, context=None, transformer_options={}, output_shape=None, time_context=None, num_video_frames=None, image_only_indicator=None):
    for layer in ts:
        if isinstance(layer, VideoResBlock):
            x = layer(x, emb, num_video_frames, image_only_indicator)
        elif isinstance(layer, TimestepBlock):
            x = layer(x, emb)
        elif isinstance(layer, SpatialVideoTransformer):
            x = layer(x, context, time_context, num_video_frames, image_only_indicator, transformer_options)
            if 'transformer_index' in transformer_options:
                transformer_options['transformer_index'] += 1
        elif isinstance(layer, SpatialTransformer):
            x = layer(x, context, transformer_options)
            if 'transformer_index' in transformer_options:
                transformer_options['transformer_index'] += 1
        elif isinstance(layer, Upsample):
            x = layer(x, output_shape=output_shape)
        else:
            x = layer(x)
    return x


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, *args, **kwargs):
        return forward_timestep_embed(self, *args, **kwargs)


class Timestep(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        return timestep_embedding(t, self.dim)


def apply_control(h, control, name):
    if control is not None and name in control and len(control[name]) > 0:
        ctrl = control[name].pop()
        if ctrl is not None:
            try:
                h += ctrl
            except:
                logging.warning('warning control could not be applied {} {}'.format(h.shape, ctrl.shape))
    return h


class UNetModel(nn.Module):
    """
    The full UNet model with attention and timestep embedding.
    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param num_heads_upsample: works with num_heads to set a different number
                               of heads for upsampling. Deprecated.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    :param use_new_attention_order: use a different attention pattern for potentially
                                    increased efficiency.
    """

    def __init__(self, image_size, in_channels, model_channels, out_channels, num_res_blocks, dropout=0, channel_mult=(1, 2, 4, 8), conv_resample=True, dims=2, num_classes=None, use_checkpoint=False, dtype=th.float32, num_heads=-1, num_head_channels=-1, num_heads_upsample=-1, use_scale_shift_norm=False, resblock_updown=False, use_new_attention_order=False, use_spatial_transformer=False, transformer_depth=1, context_dim=None, n_embed=None, legacy=True, disable_self_attentions=None, num_attention_blocks=None, disable_middle_self_attn=False, use_linear_in_transformer=False, adm_in_channels=None, transformer_depth_middle=None, transformer_depth_output=None, use_temporal_resblock=False, use_temporal_attention=False, time_context_dim=None, extra_ff_mix_layer=False, use_spatial_context=False, merge_strategy=None, merge_factor=0.0, video_kernel_size=None, disable_temporal_crossattention=False, max_ddpm_temb_period=10000, attn_precision=None, device=None, operations=ops):
        super().__init__()
        if context_dim is not None:
            assert use_spatial_transformer, 'Fool!! You forgot to use the spatial transformer for your cross-attention conditioning...'
        if num_heads_upsample == -1:
            num_heads_upsample = num_heads
        if num_heads == -1:
            assert num_head_channels != -1, 'Either num_heads or num_head_channels has to be set'
        if num_head_channels == -1:
            assert num_heads != -1, 'Either num_heads or num_head_channels has to be set'
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        if isinstance(num_res_blocks, int):
            self.num_res_blocks = len(channel_mult) * [num_res_blocks]
        else:
            if len(num_res_blocks) != len(channel_mult):
                raise ValueError('provide num_res_blocks either as an int (globally constant) or as a list/tuple (per-level) with the same length as channel_mult')
            self.num_res_blocks = num_res_blocks
        if disable_self_attentions is not None:
            assert len(disable_self_attentions) == len(channel_mult)
        if num_attention_blocks is not None:
            assert len(num_attention_blocks) == len(self.num_res_blocks)
        transformer_depth = transformer_depth[:]
        transformer_depth_output = transformer_depth_output[:]
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.dtype = dtype
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.use_temporal_resblocks = use_temporal_resblock
        self.predict_codebook_ids = n_embed is not None
        self.default_num_video_frames = None
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(operations.Linear(model_channels, time_embed_dim, dtype=self.dtype, device=device), nn.SiLU(), operations.Linear(time_embed_dim, time_embed_dim, dtype=self.dtype, device=device))
        if self.num_classes is not None:
            if isinstance(self.num_classes, int):
                self.label_emb = nn.Embedding(num_classes, time_embed_dim, dtype=self.dtype, device=device)
            elif self.num_classes == 'continuous':
                logging.debug('setting up linear c_adm embedding layer')
                self.label_emb = nn.Linear(1, time_embed_dim)
            elif self.num_classes == 'sequential':
                assert adm_in_channels is not None
                self.label_emb = nn.Sequential(nn.Sequential(operations.Linear(adm_in_channels, time_embed_dim, dtype=self.dtype, device=device), nn.SiLU(), operations.Linear(time_embed_dim, time_embed_dim, dtype=self.dtype, device=device)))
            else:
                raise ValueError()
        self.input_blocks = nn.ModuleList([TimestepEmbedSequential(operations.conv_nd(dims, in_channels, model_channels, 3, padding=1, dtype=self.dtype, device=device))])
        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1

        def get_attention_layer(ch, num_heads, dim_head, depth=1, context_dim=None, use_checkpoint=False, disable_self_attn=False):
            if use_temporal_attention:
                return SpatialVideoTransformer(ch, num_heads, dim_head, depth=depth, context_dim=context_dim, time_context_dim=time_context_dim, dropout=dropout, ff_in=extra_ff_mix_layer, use_spatial_context=use_spatial_context, merge_strategy=merge_strategy, merge_factor=merge_factor, checkpoint=use_checkpoint, use_linear=use_linear_in_transformer, disable_self_attn=disable_self_attn, disable_temporal_crossattention=disable_temporal_crossattention, max_time_embed_period=max_ddpm_temb_period, attn_precision=attn_precision, dtype=self.dtype, device=device, operations=operations)
            else:
                return SpatialTransformer(ch, num_heads, dim_head, depth=depth, context_dim=context_dim, disable_self_attn=disable_self_attn, use_linear=use_linear_in_transformer, use_checkpoint=use_checkpoint, attn_precision=attn_precision, dtype=self.dtype, device=device, operations=operations)

        def get_resblock(merge_factor, merge_strategy, video_kernel_size, ch, time_embed_dim, dropout, out_channels, dims, use_checkpoint, use_scale_shift_norm, down=False, up=False, dtype=None, device=None, operations=ops):
            if self.use_temporal_resblocks:
                return VideoResBlock(merge_factor=merge_factor, merge_strategy=merge_strategy, video_kernel_size=video_kernel_size, channels=ch, emb_channels=time_embed_dim, dropout=dropout, out_channels=out_channels, dims=dims, use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm, down=down, up=up, dtype=dtype, device=device, operations=operations)
            else:
                return ResBlock(channels=ch, emb_channels=time_embed_dim, dropout=dropout, out_channels=out_channels, use_checkpoint=use_checkpoint, dims=dims, use_scale_shift_norm=use_scale_shift_norm, down=down, up=up, dtype=dtype, device=device, operations=operations)
        for level, mult in enumerate(channel_mult):
            for nr in range(self.num_res_blocks[level]):
                layers = [get_resblock(merge_factor=merge_factor, merge_strategy=merge_strategy, video_kernel_size=video_kernel_size, ch=ch, time_embed_dim=time_embed_dim, dropout=dropout, out_channels=mult * model_channels, dims=dims, use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm, dtype=self.dtype, device=device, operations=operations)]
                ch = mult * model_channels
                num_transformers = transformer_depth.pop(0)
                if num_transformers > 0:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    if exists(disable_self_attentions):
                        disabled_sa = disable_self_attentions[level]
                    else:
                        disabled_sa = False
                    if not exists(num_attention_blocks) or nr < num_attention_blocks[level]:
                        layers.append(get_attention_layer(ch, num_heads, dim_head, depth=num_transformers, context_dim=context_dim, disable_self_attn=disabled_sa, use_checkpoint=use_checkpoint))
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(TimestepEmbedSequential(get_resblock(merge_factor=merge_factor, merge_strategy=merge_strategy, video_kernel_size=video_kernel_size, ch=ch, time_embed_dim=time_embed_dim, dropout=dropout, out_channels=out_ch, dims=dims, use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm, down=True, dtype=self.dtype, device=device, operations=operations) if resblock_updown else Downsample(ch, conv_resample, dims=dims, out_channels=out_ch, dtype=self.dtype, device=device, operations=operations)))
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch
        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels
        if legacy:
            dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
        mid_block = [get_resblock(merge_factor=merge_factor, merge_strategy=merge_strategy, video_kernel_size=video_kernel_size, ch=ch, time_embed_dim=time_embed_dim, dropout=dropout, out_channels=None, dims=dims, use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm, dtype=self.dtype, device=device, operations=operations)]
        self.middle_block = None
        if transformer_depth_middle >= -1:
            if transformer_depth_middle >= 0:
                mid_block += [get_attention_layer(ch, num_heads, dim_head, depth=transformer_depth_middle, context_dim=context_dim, disable_self_attn=disable_middle_self_attn, use_checkpoint=use_checkpoint), get_resblock(merge_factor=merge_factor, merge_strategy=merge_strategy, video_kernel_size=video_kernel_size, ch=ch, time_embed_dim=time_embed_dim, dropout=dropout, out_channels=None, dims=dims, use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm, dtype=self.dtype, device=device, operations=operations)]
            self.middle_block = TimestepEmbedSequential(*mid_block)
        self._feature_size += ch
        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(self.num_res_blocks[level] + 1):
                ich = input_block_chans.pop()
                layers = [get_resblock(merge_factor=merge_factor, merge_strategy=merge_strategy, video_kernel_size=video_kernel_size, ch=ch + ich, time_embed_dim=time_embed_dim, dropout=dropout, out_channels=model_channels * mult, dims=dims, use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm, dtype=self.dtype, device=device, operations=operations)]
                ch = model_channels * mult
                num_transformers = transformer_depth_output.pop()
                if num_transformers > 0:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    if exists(disable_self_attentions):
                        disabled_sa = disable_self_attentions[level]
                    else:
                        disabled_sa = False
                    if not exists(num_attention_blocks) or i < num_attention_blocks[level]:
                        layers.append(get_attention_layer(ch, num_heads, dim_head, depth=num_transformers, context_dim=context_dim, disable_self_attn=disabled_sa, use_checkpoint=use_checkpoint))
                if level and i == self.num_res_blocks[level]:
                    out_ch = ch
                    layers.append(get_resblock(merge_factor=merge_factor, merge_strategy=merge_strategy, video_kernel_size=video_kernel_size, ch=ch, time_embed_dim=time_embed_dim, dropout=dropout, out_channels=out_ch, dims=dims, use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm, up=True, dtype=self.dtype, device=device, operations=operations) if resblock_updown else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch, dtype=self.dtype, device=device, operations=operations))
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
        self.out = nn.Sequential(operations.GroupNorm(32, ch, dtype=self.dtype, device=device), nn.SiLU(), operations.conv_nd(dims, model_channels, out_channels, 3, padding=1, dtype=self.dtype, device=device))
        if self.predict_codebook_ids:
            self.id_predictor = nn.Sequential(operations.GroupNorm(32, ch, dtype=self.dtype, device=device), operations.conv_nd(dims, model_channels, n_embed, 1, dtype=self.dtype, device=device))

    def forward(self, x, timesteps=None, context=None, y=None, control=None, transformer_options={}, **kwargs):
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param context: conditioning plugged in via crossattn
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        transformer_options['original_shape'] = list(x.shape)
        transformer_options['transformer_index'] = 0
        transformer_patches = transformer_options.get('patches', {})
        num_video_frames = kwargs.get('num_video_frames', self.default_num_video_frames)
        image_only_indicator = kwargs.get('image_only_indicator', None)
        time_context = kwargs.get('time_context', None)
        assert (y is not None) == (self.num_classes is not None), 'must specify y if and only if the model is class-conditional'
        hs = []
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)
        if 'emb_patch' in transformer_patches:
            patch = transformer_patches['emb_patch']
            for p in patch:
                emb = p(emb, self.model_channels, transformer_options)
        if self.num_classes is not None:
            assert y.shape[0] == x.shape[0]
            emb = emb + self.label_emb(y)
        h = x
        for id, module in enumerate(self.input_blocks):
            transformer_options['block'] = 'input', id
            h = forward_timestep_embed(module, h, emb, context, transformer_options, time_context=time_context, num_video_frames=num_video_frames, image_only_indicator=image_only_indicator)
            h = apply_control(h, control, 'input')
            if 'input_block_patch' in transformer_patches:
                patch = transformer_patches['input_block_patch']
                for p in patch:
                    h = p(h, transformer_options)
            hs.append(h)
            if 'input_block_patch_after_skip' in transformer_patches:
                patch = transformer_patches['input_block_patch_after_skip']
                for p in patch:
                    h = p(h, transformer_options)
        transformer_options['block'] = 'middle', 0
        if self.middle_block is not None:
            h = forward_timestep_embed(self.middle_block, h, emb, context, transformer_options, time_context=time_context, num_video_frames=num_video_frames, image_only_indicator=image_only_indicator)
        h = apply_control(h, control, 'middle')
        for id, module in enumerate(self.output_blocks):
            transformer_options['block'] = 'output', id
            hsp = hs.pop()
            hsp = apply_control(hsp, control, 'output')
            if 'output_block_patch' in transformer_patches:
                patch = transformer_patches['output_block_patch']
                for p in patch:
                    h, hsp = p(h, hsp, transformer_options)
            h = th.cat([h, hsp], dim=1)
            del hsp
            if len(hs) > 0:
                output_shape = hs[-1].shape
            else:
                output_shape = None
            h = forward_timestep_embed(module, h, emb, context, transformer_options, output_shape, time_context=time_context, num_video_frames=num_video_frames, image_only_indicator=image_only_indicator)
        h = h.type(x.dtype)
        if self.predict_codebook_ids:
            return self.id_predictor(h)
        else:
            return self.out(h)


def extract_into_tensor(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].
    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


def make_beta_schedule(schedule, n_timestep, linear_start=0.0001, linear_end=0.02, cosine_s=0.008):
    if schedule == 'linear':
        betas = torch.linspace(linear_start ** 0.5, linear_end ** 0.5, n_timestep, dtype=torch.float64) ** 2
    elif schedule == 'cosine':
        timesteps = torch.arange(n_timestep + 1, dtype=torch.float64) / n_timestep + cosine_s
        alphas = timesteps / (1 + cosine_s) * np.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = torch.clamp(betas, min=0, max=0.999)
    elif schedule == 'squaredcos_cap_v2':
        return betas_for_alpha_bar(n_timestep, lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2)
    elif schedule == 'sqrt_linear':
        betas = torch.linspace(linear_start, linear_end, n_timestep, dtype=torch.float64)
    elif schedule == 'sqrt':
        betas = torch.linspace(linear_start, linear_end, n_timestep, dtype=torch.float64) ** 0.5
    else:
        raise ValueError(f"schedule '{schedule}' unknown.")
    return betas


class AbstractLowScaleModel(nn.Module):

    def __init__(self, noise_schedule_config=None):
        super(AbstractLowScaleModel, self).__init__()
        if noise_schedule_config is not None:
            self.register_schedule(**noise_schedule_config)

    def register_schedule(self, beta_schedule='linear', timesteps=1000, linear_start=0.0001, linear_end=0.02, cosine_s=0.008):
        betas = make_beta_schedule(beta_schedule, timesteps, linear_start=linear_start, linear_end=linear_end, cosine_s=cosine_s)
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])
        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.linear_start = linear_start
        self.linear_end = linear_end
        assert alphas_cumprod.shape[0] == self.num_timesteps, 'alphas have to be defined for each timestep'
        to_torch = partial(torch.tensor, dtype=torch.float32)
        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1.0 - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1.0 - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1.0 / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1.0 / alphas_cumprod - 1)))

    def q_sample(self, x_start, t, noise=None, seed=None):
        if noise is None:
            if seed is None:
                noise = torch.randn_like(x_start)
            else:
                noise = torch.randn(x_start.size(), dtype=x_start.dtype, layout=x_start.layout, generator=torch.manual_seed(seed))
        return extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start + extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise

    def forward(self, x):
        return x, None

    def decode(self, x):
        return x


class SimpleImageConcat(AbstractLowScaleModel):

    def __init__(self):
        super(SimpleImageConcat, self).__init__(noise_schedule_config=None)
        self.max_noise_level = 0

    def forward(self, x):
        return x, torch.zeros(x.shape[0], device=x.device).long()


class ImageConcatWithNoiseAugmentation(AbstractLowScaleModel):

    def __init__(self, noise_schedule_config, max_noise_level=1000, to_cuda=False):
        super().__init__(noise_schedule_config=noise_schedule_config)
        self.max_noise_level = max_noise_level

    def forward(self, x, noise_level=None, seed=None):
        if noise_level is None:
            noise_level = torch.randint(0, self.max_noise_level, (x.shape[0],), device=x.device).long()
        else:
            assert isinstance(noise_level, torch.Tensor)
        z = self.q_sample(x, noise_level, seed=seed)
        return z, noise_level


class HybridConditioner(nn.Module):

    def __init__(self, c_concat_config, c_crossattn_config):
        super().__init__()
        self.concat_conditioner = instantiate_from_config(c_concat_config)
        self.crossattn_conditioner = instantiate_from_config(c_crossattn_config)

    def forward(self, c_concat, c_crossattn):
        c_concat = self.concat_conditioner(c_concat)
        c_crossattn = self.crossattn_conditioner(c_crossattn)
        return {'c_concat': [c_concat], 'c_crossattn': [c_crossattn]}


class CLIPEmbeddingNoiseAugmentation(ImageConcatWithNoiseAugmentation):

    def __init__(self, *args, clip_stats_path=None, timestep_dim=256, **kwargs):
        super().__init__(*args, **kwargs)
        if clip_stats_path is None:
            clip_mean, clip_std = torch.zeros(timestep_dim), torch.ones(timestep_dim)
        else:
            clip_mean, clip_std = torch.load(clip_stats_path, map_location='cpu')
        self.register_buffer('data_mean', clip_mean[None, :], persistent=False)
        self.register_buffer('data_std', clip_std[None, :], persistent=False)
        self.time_embed = Timestep(timestep_dim)

    def scale(self, x):
        x = (x - self.data_mean) * 1.0 / self.data_std
        return x

    def unscale(self, x):
        x = x * self.data_std + self.data_mean
        return x

    def forward(self, x, noise_level=None, seed=None):
        if noise_level is None:
            noise_level = torch.randint(0, self.max_noise_level, (x.shape[0],), device=x.device).long()
        else:
            assert isinstance(noise_level, torch.Tensor)
        x = self.scale(x)
        z = self.q_sample(x, noise_level, seed=seed)
        z = self.unscale(z)
        noise_level = self.time_embed(noise_level)
        return z, noise_level


class AttnVideoBlock(AttnBlock):

    def __init__(self, in_channels: 'int', alpha: 'float'=0, merge_strategy: 'str'='learned'):
        super().__init__(in_channels)
        self.time_mix_block = BasicTransformerBlock(dim=in_channels, n_heads=1, d_head=in_channels, checkpoint=False, ff_in=True)
        time_embed_dim = self.in_channels * 4
        self.video_time_embed = torch.nn.Sequential(ops.Linear(self.in_channels, time_embed_dim), torch.nn.SiLU(), ops.Linear(time_embed_dim, self.in_channels))
        self.merge_strategy = merge_strategy
        if self.merge_strategy == 'fixed':
            self.register_buffer('mix_factor', torch.Tensor([alpha]))
        elif self.merge_strategy == 'learned':
            self.register_parameter('mix_factor', torch.nn.Parameter(torch.Tensor([alpha])))
        else:
            raise ValueError(f'unknown merge strategy {self.merge_strategy}')

    def forward(self, x, timesteps=None, skip_time_block=False):
        if skip_time_block:
            return super().forward(x)
        if timesteps is None:
            timesteps = x.shape[0]
        x_in = x
        x = self.attention(x)
        h, w = x.shape[2:]
        x = rearrange(x, 'b c h w -> b (h w) c')
        x_mix = x
        num_frames = torch.arange(timesteps, device=x.device)
        num_frames = repeat(num_frames, 't -> b t', b=x.shape[0] // timesteps)
        num_frames = rearrange(num_frames, 'b t -> (b t)')
        t_emb = timestep_embedding(num_frames, self.in_channels, repeat_only=False)
        emb = self.video_time_embed(t_emb)
        emb = emb[:, None, :]
        x_mix = x_mix + emb
        alpha = self.get_alpha()
        x_mix = self.time_mix_block(x_mix, timesteps=timesteps)
        x = alpha * x + (1.0 - alpha) * x_mix
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        x = self.proj_out(x)
        return x_in + x

    def get_alpha(self):
        if self.merge_strategy == 'fixed':
            return self.mix_factor
        elif self.merge_strategy == 'learned':
            return torch.sigmoid(self.mix_factor)
        else:
            raise NotImplementedError(f'unknown merge strategy {self.merge_strategy}')


class Conv2DWrapper(torch.nn.Conv2d):

    def forward(self, input: 'torch.Tensor', **kwargs) ->torch.Tensor:
        return super().forward(input)


def partialclass(cls, *args, **kwargs):


    class NewCls(cls):
        __init__ = functools.partialmethod(cls.__init__, *args, **kwargs)
    return NewCls


def make_time_attn(in_channels, attn_type='vanilla', attn_kwargs=None, alpha: 'float'=0, merge_strategy: 'str'='learned'):
    return partialclass(AttnVideoBlock, in_channels, alpha=alpha, merge_strategy=merge_strategy)


class ModelType(Enum):
    EPS = 1
    V_PREDICTION = 2
    V_PREDICTION_EDM = 3
    STABLE_CASCADE = 4
    EDM = 5
    FLOW = 6
    V_PREDICTION_CONTINUOUS = 7
    FLUX = 8


class EPS:

    def calculate_input(self, sigma, noise):
        sigma = sigma.view(sigma.shape[:1] + (1,) * (noise.ndim - 1))
        return noise / (sigma ** 2 + self.sigma_data ** 2) ** 0.5

    def calculate_denoised(self, sigma, model_output, model_input):
        sigma = sigma.view(sigma.shape[:1] + (1,) * (model_output.ndim - 1))
        return model_input - model_output * sigma

    def noise_scaling(self, sigma, noise, latent_image, max_denoise=False):
        if max_denoise:
            noise = noise * torch.sqrt(1.0 + sigma ** 2.0)
        else:
            noise = noise * sigma
        noise += latent_image
        return noise

    def inverse_noise_scaling(self, sigma, latent):
        return latent


class V_PREDICTION(EPS):

    def calculate_denoised(self, sigma, model_output, model_input):
        sigma = sigma.view(sigma.shape[:1] + (1,) * (model_output.ndim - 1))
        return model_input * self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2) - model_output * sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2) ** 0.5


class EDM(V_PREDICTION):

    def calculate_denoised(self, sigma, model_output, model_input):
        sigma = sigma.view(sigma.shape[:1] + (1,) * (model_output.ndim - 1))
        return model_input * self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2) + model_output * sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2) ** 0.5


class ModelSamplingContinuousEDM:

    @classmethod
    def INPUT_TYPES(s):
        return {'required': {'model': ('MODEL',), 'sampling': (['v_prediction', 'edm_playground_v2.5', 'eps'],), 'sigma_max': ('FLOAT', {'default': 120.0, 'min': 0.0, 'max': 1000.0, 'step': 0.001, 'round': False}), 'sigma_min': ('FLOAT', {'default': 0.002, 'min': 0.0, 'max': 1000.0, 'step': 0.001, 'round': False})}}
    RETURN_TYPES = 'MODEL',
    FUNCTION = 'patch'
    CATEGORY = 'advanced/model'

    def patch(self, model, sampling, sigma_max, sigma_min):
        m = model.clone()
        latent_format = None
        sigma_data = 1.0
        if sampling == 'eps':
            sampling_type = comfy.model_sampling.EPS
        elif sampling == 'v_prediction':
            sampling_type = comfy.model_sampling.V_PREDICTION
        elif sampling == 'edm_playground_v2.5':
            sampling_type = comfy.model_sampling.EDM
            sigma_data = 0.5
            latent_format = comfy.latent_formats.SDXL_Playground_2_5()


        class ModelSamplingAdvanced(comfy.model_sampling.ModelSamplingContinuousEDM, sampling_type):
            pass
        model_sampling = ModelSamplingAdvanced(model.model.model_config)
        model_sampling.set_parameters(sigma_min, sigma_max, sigma_data)
        m.add_object_patch('model_sampling', model_sampling)
        if latent_format is not None:
            m.add_object_patch('latent_format', latent_format)
        return m,


class ModelSamplingContinuousV:

    @classmethod
    def INPUT_TYPES(s):
        return {'required': {'model': ('MODEL',), 'sampling': (['v_prediction'],), 'sigma_max': ('FLOAT', {'default': 500.0, 'min': 0.0, 'max': 1000.0, 'step': 0.001, 'round': False}), 'sigma_min': ('FLOAT', {'default': 0.03, 'min': 0.0, 'max': 1000.0, 'step': 0.001, 'round': False})}}
    RETURN_TYPES = 'MODEL',
    FUNCTION = 'patch'
    CATEGORY = 'advanced/model'

    def patch(self, model, sampling, sigma_max, sigma_min):
        m = model.clone()
        latent_format = None
        sigma_data = 1.0
        if sampling == 'v_prediction':
            sampling_type = comfy.model_sampling.V_PREDICTION


        class ModelSamplingAdvanced(comfy.model_sampling.ModelSamplingContinuousV, sampling_type):
            pass
        model_sampling = ModelSamplingAdvanced(model.model.model_config)
        model_sampling.set_parameters(sigma_min, sigma_max, sigma_data)
        m.add_object_patch('model_sampling', model_sampling)
        return m,


def rescale_zero_terminal_snr_sigmas(sigmas):
    alphas_cumprod = 1 / (sigmas * sigmas + 1)
    alphas_bar_sqrt = alphas_cumprod.sqrt()
    alphas_bar_sqrt_0 = alphas_bar_sqrt[0].clone()
    alphas_bar_sqrt_T = alphas_bar_sqrt[-1].clone()
    alphas_bar_sqrt -= alphas_bar_sqrt_T
    alphas_bar_sqrt *= alphas_bar_sqrt_0 / (alphas_bar_sqrt_0 - alphas_bar_sqrt_T)
    alphas_bar = alphas_bar_sqrt ** 2
    alphas_bar[-1] = 4.8973451890853435e-08
    return ((1 - alphas_bar) / alphas_bar) ** 0.5


class ModelSamplingDiscrete:

    @classmethod
    def INPUT_TYPES(s):
        return {'required': {'model': ('MODEL',), 'sampling': (['eps', 'v_prediction', 'lcm', 'x0'],), 'zsnr': ('BOOLEAN', {'default': False})}}
    RETURN_TYPES = 'MODEL',
    FUNCTION = 'patch'
    CATEGORY = 'advanced/model'

    def patch(self, model, sampling, zsnr):
        m = model.clone()
        sampling_base = comfy.model_sampling.ModelSamplingDiscrete
        if sampling == 'eps':
            sampling_type = comfy.model_sampling.EPS
        elif sampling == 'v_prediction':
            sampling_type = comfy.model_sampling.V_PREDICTION
        elif sampling == 'lcm':
            sampling_type = LCM
            sampling_base = ModelSamplingDiscreteDistilled
        elif sampling == 'x0':
            sampling_type = X0


        class ModelSamplingAdvanced(sampling_base, sampling_type):
            pass
        model_sampling = ModelSamplingAdvanced(model.model.model_config)
        if zsnr:
            model_sampling.set_sigmas(rescale_zero_terminal_snr_sigmas(model_sampling.sigmas))
        m.add_object_patch('model_sampling', model_sampling)
        return m,


class StableCascadeSampling(ModelSamplingDiscrete):

    def __init__(self, model_config=None):
        super().__init__()
        if model_config is not None:
            sampling_settings = model_config.sampling_settings
        else:
            sampling_settings = {}
        self.set_parameters(sampling_settings.get('shift', 1.0))

    def set_parameters(self, shift=1.0, cosine_s=0.008):
        self.shift = shift
        self.cosine_s = torch.tensor(cosine_s)
        self._init_alpha_cumprod = torch.cos(self.cosine_s / (1 + self.cosine_s) * torch.pi * 0.5) ** 2
        self.num_timesteps = 10000
        sigmas = torch.empty(self.num_timesteps, dtype=torch.float32)
        for x in range(self.num_timesteps):
            t = (x + 1) / self.num_timesteps
            sigmas[x] = self.sigma(t)
        self.set_sigmas(sigmas)

    def sigma(self, timestep):
        alpha_cumprod = torch.cos((timestep + self.cosine_s) / (1 + self.cosine_s) * torch.pi * 0.5) ** 2 / self._init_alpha_cumprod
        if self.shift != 1.0:
            var = alpha_cumprod
            logSNR = (var / (1 - var)).log()
            logSNR += 2 * torch.log(1.0 / torch.tensor(self.shift))
            alpha_cumprod = logSNR.sigmoid()
        alpha_cumprod = alpha_cumprod.clamp(0.0001, 0.9999)
        return ((1 - alpha_cumprod) / alpha_cumprod) ** 0.5

    def timestep(self, sigma):
        var = 1 / (sigma * sigma + 1)
        var = var.clamp(0, 1.0)
        s, min_var = self.cosine_s, self._init_alpha_cumprod
        t = ((var * min_var) ** 0.5).acos() / (torch.pi * 0.5) * (1 + s) - s
        return t

    def percent_to_sigma(self, percent):
        if percent <= 0.0:
            return 999999999.9
        if percent >= 1.0:
            return 0.0
        percent = 1.0 - percent
        return self.sigma(torch.tensor(percent))


def model_sampling(model_config, model_type):
    s = ModelSamplingDiscrete
    if model_type == ModelType.EPS:
        c = EPS
    elif model_type == ModelType.V_PREDICTION:
        c = V_PREDICTION
    elif model_type == ModelType.V_PREDICTION_EDM:
        c = V_PREDICTION
        s = ModelSamplingContinuousEDM
    elif model_type == ModelType.FLOW:
        c = comfy.model_sampling.CONST
        s = comfy.model_sampling.ModelSamplingDiscreteFlow
    elif model_type == ModelType.STABLE_CASCADE:
        c = EPS
        s = StableCascadeSampling
    elif model_type == ModelType.EDM:
        c = EDM
        s = ModelSamplingContinuousEDM
    elif model_type == ModelType.V_PREDICTION_CONTINUOUS:
        c = V_PREDICTION
        s = ModelSamplingContinuousV
    elif model_type == ModelType.FLUX:
        c = comfy.model_sampling.CONST
        s = comfy.model_sampling.ModelSamplingFlux


    class ModelSampling(s, c):
        pass
    return ModelSampling(model_config)


class BaseModel(torch.nn.Module):

    def __init__(self, model_config, model_type=ModelType.EPS, device=None, unet_model=UNetModel):
        super().__init__()
        unet_config = model_config.unet_config
        self.latent_format = model_config.latent_format
        self.model_config = model_config
        self.manual_cast_dtype = model_config.manual_cast_dtype
        self.device = device
        if not unet_config.get('disable_unet_model_creation', False):
            if model_config.custom_operations is None:
                fp8 = model_config.optimizations.get('fp8', model_config.scaled_fp8 is not None)
                operations = comfy.ops.pick_operations(unet_config.get('dtype', None), self.manual_cast_dtype, fp8_optimizations=fp8, scaled_fp8=model_config.scaled_fp8)
            else:
                operations = model_config.custom_operations
            self.diffusion_model = unet_model(**unet_config, device=device, operations=operations)
            if comfy.model_management.force_channels_last():
                self.diffusion_model
                logging.debug('using channels last mode for diffusion model')
            logging.info('model weight dtype {}, manual cast: {}'.format(self.get_dtype(), self.manual_cast_dtype))
        self.model_type = model_type
        self.model_sampling = model_sampling(model_config, model_type)
        self.adm_channels = unet_config.get('adm_in_channels', None)
        if self.adm_channels is None:
            self.adm_channels = 0
        self.concat_keys = ()
        logging.info('model_type {}'.format(model_type.name))
        logging.debug('adm {}'.format(self.adm_channels))
        self.memory_usage_factor = model_config.memory_usage_factor

    def apply_model(self, x, t, c_concat=None, c_crossattn=None, control=None, transformer_options={}, **kwargs):
        sigma = t
        xc = self.model_sampling.calculate_input(sigma, x)
        if c_concat is not None:
            xc = torch.cat([xc] + [c_concat], dim=1)
        context = c_crossattn
        dtype = self.get_dtype()
        if self.manual_cast_dtype is not None:
            dtype = self.manual_cast_dtype
        xc = xc
        t = self.model_sampling.timestep(t).float()
        context = context
        extra_conds = {}
        for o in kwargs:
            extra = kwargs[o]
            if hasattr(extra, 'dtype'):
                if extra.dtype != torch.int and extra.dtype != torch.long:
                    extra = extra
            extra_conds[o] = extra
        model_output = self.diffusion_model(xc, t, context=context, control=control, transformer_options=transformer_options, **extra_conds).float()
        return self.model_sampling.calculate_denoised(sigma, model_output, x)

    def get_dtype(self):
        return self.diffusion_model.dtype

    def is_adm(self):
        return self.adm_channels > 0

    def encode_adm(self, **kwargs):
        return None

    def extra_conds(self, **kwargs):
        out = {}
        if len(self.concat_keys) > 0:
            cond_concat = []
            denoise_mask = kwargs.get('concat_mask', kwargs.get('denoise_mask', None))
            concat_latent_image = kwargs.get('concat_latent_image', None)
            if concat_latent_image is None:
                concat_latent_image = kwargs.get('latent_image', None)
            else:
                concat_latent_image = self.process_latent_in(concat_latent_image)
            noise = kwargs.get('noise', None)
            device = kwargs['device']
            if concat_latent_image.shape[1:] != noise.shape[1:]:
                concat_latent_image = utils.common_upscale(concat_latent_image, noise.shape[-1], noise.shape[-2], 'bilinear', 'center')
            concat_latent_image = utils.resize_to_batch_size(concat_latent_image, noise.shape[0])
            if denoise_mask is not None:
                if len(denoise_mask.shape) == len(noise.shape):
                    denoise_mask = denoise_mask[:, :1]
                denoise_mask = denoise_mask.reshape((-1, 1, denoise_mask.shape[-2], denoise_mask.shape[-1]))
                if denoise_mask.shape[-2:] != noise.shape[-2:]:
                    denoise_mask = utils.common_upscale(denoise_mask, noise.shape[-1], noise.shape[-2], 'bilinear', 'center')
                denoise_mask = utils.resize_to_batch_size(denoise_mask.round(), noise.shape[0])
            for ck in self.concat_keys:
                if denoise_mask is not None:
                    if ck == 'mask':
                        cond_concat.append(denoise_mask)
                    elif ck == 'masked_image':
                        cond_concat.append(concat_latent_image)
                elif ck == 'mask':
                    cond_concat.append(torch.ones_like(noise)[:, :1])
                elif ck == 'masked_image':
                    cond_concat.append(self.blank_inpaint_image_like(noise))
            data = torch.cat(cond_concat, dim=1)
            out['c_concat'] = comfy.conds.CONDNoiseShape(data)
        adm = self.encode_adm(**kwargs)
        if adm is not None:
            out['y'] = comfy.conds.CONDRegular(adm)
        cross_attn = kwargs.get('cross_attn', None)
        if cross_attn is not None:
            out['c_crossattn'] = comfy.conds.CONDCrossAttn(cross_attn)
        cross_attn_cnet = kwargs.get('cross_attn_controlnet', None)
        if cross_attn_cnet is not None:
            out['crossattn_controlnet'] = comfy.conds.CONDCrossAttn(cross_attn_cnet)
        c_concat = kwargs.get('noise_concat', None)
        if c_concat is not None:
            out['c_concat'] = comfy.conds.CONDNoiseShape(c_concat)
        return out

    def load_model_weights(self, sd, unet_prefix=''):
        to_load = {}
        keys = list(sd.keys())
        for k in keys:
            if k.startswith(unet_prefix):
                to_load[k[len(unet_prefix):]] = sd.pop(k)
        to_load = self.model_config.process_unet_state_dict(to_load)
        m, u = self.diffusion_model.load_state_dict(to_load, strict=False)
        if len(m) > 0:
            logging.warning('unet missing: {}'.format(m))
        if len(u) > 0:
            logging.warning('unet unexpected: {}'.format(u))
        del to_load
        return self

    def process_latent_in(self, latent):
        return self.latent_format.process_in(latent)

    def process_latent_out(self, latent):
        return self.latent_format.process_out(latent)

    def state_dict_for_saving(self, clip_state_dict=None, vae_state_dict=None, clip_vision_state_dict=None):
        extra_sds = []
        if clip_state_dict is not None:
            extra_sds.append(self.model_config.process_clip_state_dict_for_saving(clip_state_dict))
        if vae_state_dict is not None:
            extra_sds.append(self.model_config.process_vae_state_dict_for_saving(vae_state_dict))
        if clip_vision_state_dict is not None:
            extra_sds.append(self.model_config.process_clip_vision_state_dict_for_saving(clip_vision_state_dict))
        unet_state_dict = self.diffusion_model.state_dict()
        if self.model_config.scaled_fp8 is not None:
            unet_state_dict['scaled_fp8'] = torch.tensor([], dtype=self.model_config.scaled_fp8)
        unet_state_dict = self.model_config.process_unet_state_dict_for_saving(unet_state_dict)
        if self.model_type == ModelType.V_PREDICTION:
            unet_state_dict['v_pred'] = torch.tensor([])
        for sd in extra_sds:
            unet_state_dict.update(sd)
        return unet_state_dict

    def set_inpaint(self):
        self.concat_keys = 'mask', 'masked_image'

        def blank_inpaint_image_like(latent_image):
            blank_image = torch.ones_like(latent_image)
            blank_image[:, 0] *= 0.8223
            blank_image[:, 1] *= -0.6876
            blank_image[:, 2] *= 0.6364
            blank_image[:, 3] *= 0.138
            return blank_image
        self.blank_inpaint_image_like = blank_inpaint_image_like

    def memory_required(self, input_shape):
        if comfy.model_management.xformers_enabled() or comfy.model_management.pytorch_attention_flash_attention():
            dtype = self.get_dtype()
            if self.manual_cast_dtype is not None:
                dtype = self.manual_cast_dtype
            area = input_shape[0] * math.prod(input_shape[2:])
            return area * comfy.model_management.dtype_size(dtype) * 0.01 * self.memory_usage_factor * (1024 * 1024)
        else:
            area = input_shape[0] * math.prod(input_shape[2:])
            return area * 0.15 * self.memory_usage_factor * (1024 * 1024)


def unclip_adm(unclip_conditioning, device, noise_augmentor, noise_augment_merge=0.0, seed=None):
    adm_inputs = []
    weights = []
    noise_aug = []
    for unclip_cond in unclip_conditioning:
        for adm_cond in unclip_cond['clip_vision_output'].image_embeds:
            weight = unclip_cond['strength']
            noise_augment = unclip_cond['noise_augmentation']
            noise_level = round((noise_augmentor.max_noise_level - 1) * noise_augment)
            c_adm, noise_level_emb = noise_augmentor(adm_cond, noise_level=torch.tensor([noise_level], device=device), seed=seed)
            adm_out = torch.cat((c_adm, noise_level_emb), 1) * weight
            weights.append(weight)
            noise_aug.append(noise_augment)
            adm_inputs.append(adm_out)
    if len(noise_aug) > 1:
        adm_out = torch.stack(adm_inputs).sum(0)
        noise_augment = noise_augment_merge
        noise_level = round((noise_augmentor.max_noise_level - 1) * noise_augment)
        c_adm, noise_level_emb = noise_augmentor(adm_out[:, :noise_augmentor.time_embed.dim], noise_level=torch.tensor([noise_level], device=device))
        adm_out = torch.cat((c_adm, noise_level_emb), 1)
    return adm_out


class SD21UNCLIP(BaseModel):

    def __init__(self, model_config, noise_aug_config, model_type=ModelType.V_PREDICTION, device=None):
        super().__init__(model_config, model_type, device=device)
        self.noise_augmentor = CLIPEmbeddingNoiseAugmentation(**noise_aug_config)

    def encode_adm(self, **kwargs):
        unclip_conditioning = kwargs.get('unclip_conditioning', None)
        device = kwargs['device']
        if unclip_conditioning is None:
            return torch.zeros((1, self.adm_channels))
        else:
            return unclip_adm(unclip_conditioning, device, self.noise_augmentor, kwargs.get('unclip_noise_augment_merge', 0.05), kwargs.get('seed', 0) - 10)


class StableCascade_C(BaseModel):

    def __init__(self, model_config, model_type=ModelType.STABLE_CASCADE, device=None):
        super().__init__(model_config, model_type, device=device, unet_model=StageC)
        self.diffusion_model.eval().requires_grad_(False)

    def extra_conds(self, **kwargs):
        out = {}
        clip_text_pooled = kwargs['pooled_output']
        if clip_text_pooled is not None:
            out['clip_text_pooled'] = comfy.conds.CONDRegular(clip_text_pooled)
        if 'unclip_conditioning' in kwargs:
            embeds = []
            for unclip_cond in kwargs['unclip_conditioning']:
                weight = unclip_cond['strength']
                embeds.append(unclip_cond['clip_vision_output'].image_embeds.unsqueeze(0) * weight)
            clip_img = torch.cat(embeds, dim=1)
        else:
            clip_img = torch.zeros((1, 1, 768))
        out['clip_img'] = comfy.conds.CONDRegular(clip_img)
        out['sca'] = comfy.conds.CONDRegular(torch.zeros((1,)))
        out['crp'] = comfy.conds.CONDRegular(torch.zeros((1,)))
        cross_attn = kwargs.get('cross_attn', None)
        if cross_attn is not None:
            out['clip_text'] = comfy.conds.CONDCrossAttn(cross_attn)
        return out


class StableCascade_B(BaseModel):

    def __init__(self, model_config, model_type=ModelType.STABLE_CASCADE, device=None):
        super().__init__(model_config, model_type, device=device, unet_model=StageB)
        self.diffusion_model.eval().requires_grad_(False)

    def extra_conds(self, **kwargs):
        out = {}
        noise = kwargs.get('noise', None)
        clip_text_pooled = kwargs['pooled_output']
        if clip_text_pooled is not None:
            out['clip'] = comfy.conds.CONDRegular(clip_text_pooled)
        prior = kwargs.get('stable_cascade_prior', torch.zeros((1, 16, noise.shape[2] * 4 // 42, noise.shape[3] * 4 // 42), dtype=noise.dtype, layout=noise.layout, device=noise.device))
        out['effnet'] = comfy.conds.CONDRegular(prior)
        out['sca'] = comfy.conds.CONDRegular(torch.zeros((1,)))
        return out


class StableAudio1(BaseModel):

    def __init__(self, model_config, seconds_start_embedder_weights, seconds_total_embedder_weights, model_type=ModelType.V_PREDICTION_CONTINUOUS, device=None):
        super().__init__(model_config, model_type, device=device, unet_model=comfy.ldm.audio.dit.AudioDiffusionTransformer)
        self.seconds_start_embedder = comfy.ldm.audio.embedders.NumberConditioner(768, min_val=0, max_val=512)
        self.seconds_total_embedder = comfy.ldm.audio.embedders.NumberConditioner(768, min_val=0, max_val=512)
        self.seconds_start_embedder.load_state_dict(seconds_start_embedder_weights)
        self.seconds_total_embedder.load_state_dict(seconds_total_embedder_weights)

    def extra_conds(self, **kwargs):
        out = {}
        noise = kwargs.get('noise', None)
        device = kwargs['device']
        seconds_start = kwargs.get('seconds_start', 0)
        seconds_total = kwargs.get('seconds_total', int(noise.shape[-1] / 21.53))
        seconds_start_embed = self.seconds_start_embedder([seconds_start])[0]
        seconds_total_embed = self.seconds_total_embedder([seconds_total])[0]
        global_embed = torch.cat([seconds_start_embed, seconds_total_embed], dim=-1).reshape((1, -1))
        out['global_embed'] = comfy.conds.CONDRegular(global_embed)
        cross_attn = kwargs.get('cross_attn', None)
        if cross_attn is not None:
            cross_attn = torch.cat([cross_attn, seconds_start_embed.repeat((cross_attn.shape[0], 1, 1)), seconds_total_embed.repeat((cross_attn.shape[0], 1, 1))], dim=1)
            out['c_crossattn'] = comfy.conds.CONDRegular(cross_attn)
        return out

    def state_dict_for_saving(self, clip_state_dict=None, vae_state_dict=None, clip_vision_state_dict=None):
        sd = super().state_dict_for_saving(clip_state_dict=clip_state_dict, vae_state_dict=vae_state_dict, clip_vision_state_dict=clip_vision_state_dict)
        d = {'conditioner.conditioners.seconds_start.': self.seconds_start_embedder.state_dict(), 'conditioner.conditioners.seconds_total.': self.seconds_total_embedder.state_dict()}
        for k in d:
            s = d[k]
            for l in s:
                sd['{}{}'.format(k, l)] = s[l]
        return sd


class ModelSamplingDiscreteEDM(ModelSamplingDiscrete):

    def timestep(self, sigma):
        return 0.25 * sigma.log()

    def sigma(self, timestep):
        return (timestep / 0.25).exp()


def time_snr_shift(alpha, t):
    if alpha == 1.0:
        return t
    return alpha * t / (1 + (alpha - 1) * t)


class ModelSamplingDiscreteFlow(torch.nn.Module):

    def __init__(self, model_config=None):
        super().__init__()
        if model_config is not None:
            sampling_settings = model_config.sampling_settings
        else:
            sampling_settings = {}
        self.set_parameters(shift=sampling_settings.get('shift', 1.0), multiplier=sampling_settings.get('multiplier', 1000))

    def set_parameters(self, shift=1.0, timesteps=1000, multiplier=1000):
        self.shift = shift
        self.multiplier = multiplier
        ts = self.sigma(torch.arange(1, timesteps + 1, 1) / timesteps * multiplier)
        self.register_buffer('sigmas', ts)

    @property
    def sigma_min(self):
        return self.sigmas[0]

    @property
    def sigma_max(self):
        return self.sigmas[-1]

    def timestep(self, sigma):
        return sigma * self.multiplier

    def sigma(self, timestep):
        return time_snr_shift(self.shift, timestep / self.multiplier)

    def percent_to_sigma(self, percent):
        if percent <= 0.0:
            return 1.0
        if percent >= 1.0:
            return 0.0
        return 1.0 - percent


class ModelSamplingFlux:

    @classmethod
    def INPUT_TYPES(s):
        return {'required': {'model': ('MODEL',), 'max_shift': ('FLOAT', {'default': 1.15, 'min': 0.0, 'max': 100.0, 'step': 0.01}), 'base_shift': ('FLOAT', {'default': 0.5, 'min': 0.0, 'max': 100.0, 'step': 0.01}), 'width': ('INT', {'default': 1024, 'min': 16, 'max': nodes.MAX_RESOLUTION, 'step': 8}), 'height': ('INT', {'default': 1024, 'min': 16, 'max': nodes.MAX_RESOLUTION, 'step': 8})}}
    RETURN_TYPES = 'MODEL',
    FUNCTION = 'patch'
    CATEGORY = 'advanced/model'

    def patch(self, model, max_shift, base_shift, width, height):
        m = model.clone()
        x1 = 256
        x2 = 4096
        mm = (max_shift - base_shift) / (x2 - x1)
        b = base_shift - mm * x1
        shift = width * height / (8 * 8 * 2 * 2) * mm + b
        sampling_base = comfy.model_sampling.ModelSamplingFlux
        sampling_type = comfy.model_sampling.CONST


        class ModelSamplingAdvanced(sampling_base, sampling_type):
            pass
        model_sampling = ModelSamplingAdvanced(model.model.model_config)
        model_sampling.set_parameters(shift=shift)
        m.add_object_patch('model_sampling', model_sampling)
        return m,


def gen_empty_tokens(special_tokens, length):
    start_token = special_tokens.get('start', None)
    end_token = special_tokens.get('end', None)
    pad_token = special_tokens.get('pad')
    output = []
    if start_token is not None:
        output.append(start_token)
    if end_token is not None:
        output.append(end_token)
    output += [pad_token] * (length - len(output))
    return output


class ClipTokenWeightEncoder:

    def encode_token_weights(self, token_weight_pairs):
        to_encode = list()
        max_token_len = 0
        has_weights = False
        for x in token_weight_pairs:
            tokens = list(map(lambda a: a[0], x))
            max_token_len = max(len(tokens), max_token_len)
            has_weights = has_weights or not all(map(lambda a: a[1] == 1.0, x))
            to_encode.append(tokens)
        sections = len(to_encode)
        if has_weights or sections == 0:
            to_encode.append(gen_empty_tokens(self.special_tokens, max_token_len))
        o = self.encode(to_encode)
        out, pooled = o[:2]
        if pooled is not None:
            first_pooled = pooled[0:1]
        else:
            first_pooled = pooled
        output = []
        for k in range(0, sections):
            z = out[k:k + 1]
            if has_weights:
                z_empty = out[-1]
                for i in range(len(z)):
                    for j in range(len(z[i])):
                        weight = token_weight_pairs[k][j][1]
                        if weight != 1.0:
                            z[i][j] = (z[i][j] - z_empty[j]) * weight + z_empty[j]
            output.append(z)
        if len(output) == 0:
            r = out[-1:], first_pooled
        else:
            r = torch.cat(output, dim=-2), first_pooled
        if len(o) > 2:
            extra = {}
            for k in o[2]:
                v = o[2][k]
                if k == 'attention_mask':
                    v = v[:sections].flatten().unsqueeze(dim=0)
                extra[k] = v
            r = r + (extra,)
        return r


class SDXLClipModel(torch.nn.Module):

    def __init__(self, device='cpu', dtype=None, model_options={}):
        super().__init__()
        clip_l_class = model_options.get('clip_l_class', sd1_clip.SDClipModel)
        self.clip_l = clip_l_class(layer='hidden', layer_idx=-2, device=device, dtype=dtype, layer_norm_hidden_state=False, model_options=model_options)
        self.clip_g = SDXLClipG(device=device, dtype=dtype, model_options=model_options)
        self.dtypes = set([dtype])

    def set_clip_options(self, options):
        self.clip_l.set_clip_options(options)
        self.clip_g.set_clip_options(options)

    def reset_clip_options(self):
        self.clip_g.reset_clip_options()
        self.clip_l.reset_clip_options()

    def encode_token_weights(self, token_weight_pairs):
        token_weight_pairs_g = token_weight_pairs['g']
        token_weight_pairs_l = token_weight_pairs['l']
        g_out, g_pooled = self.clip_g.encode_token_weights(token_weight_pairs_g)
        l_out, l_pooled = self.clip_l.encode_token_weights(token_weight_pairs_l)
        cut_to = min(l_out.shape[1], g_out.shape[1])
        return torch.cat([l_out[:, :cut_to], g_out[:, :cut_to]], dim=-1), g_pooled

    def load_sd(self, sd):
        if 'text_model.encoder.layers.30.mlp.fc1.weight' in sd:
            return self.clip_g.load_sd(sd)
        else:
            return self.clip_l.load_sd(sd)


class Adapter(nn.Module):

    def __init__(self, channels=[320, 640, 1280, 1280], nums_rb=3, cin=64, ksize=3, sk=False, use_conv=True, xl=True):
        super(Adapter, self).__init__()
        self.unshuffle_amount = 8
        resblock_no_downsample = []
        resblock_downsample = [3, 2, 1]
        self.xl = xl
        if self.xl:
            self.unshuffle_amount = 16
            resblock_no_downsample = [1]
            resblock_downsample = [2]
        self.input_channels = cin // (self.unshuffle_amount * self.unshuffle_amount)
        self.unshuffle = nn.PixelUnshuffle(self.unshuffle_amount)
        self.channels = channels
        self.nums_rb = nums_rb
        self.body = []
        for i in range(len(channels)):
            for j in range(nums_rb):
                if i in resblock_downsample and j == 0:
                    self.body.append(ResnetBlock(channels[i - 1], channels[i], down=True, ksize=ksize, sk=sk, use_conv=use_conv))
                elif i in resblock_no_downsample and j == 0:
                    self.body.append(ResnetBlock(channels[i - 1], channels[i], down=False, ksize=ksize, sk=sk, use_conv=use_conv))
                else:
                    self.body.append(ResnetBlock(channels[i], channels[i], down=False, ksize=ksize, sk=sk, use_conv=use_conv))
        self.body = nn.ModuleList(self.body)
        self.conv_in = nn.Conv2d(cin, channels[0], 3, 1, 1)

    def forward(self, x):
        x = self.unshuffle(x)
        features = []
        x = self.conv_in(x)
        for i in range(len(self.channels)):
            for j in range(self.nums_rb):
                idx = i * self.nums_rb + j
                x = self.body[idx](x)
            if self.xl:
                features.append(None)
                if i == 0:
                    features.append(None)
                    features.append(None)
                if i == 2:
                    features.append(None)
            else:
                features.append(None)
                features.append(None)
            features.append(x)
        features = features[::-1]
        if self.xl:
            return {'input': features[1:], 'middle': features[:1]}
        else:
            return {'input': features}


class ResidualAttentionBlock(nn.Module):

    def __init__(self, d_model: 'int', n_head: 'int', attn_mask: 'torch.Tensor'=None):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([('c_fc', nn.Linear(d_model, d_model * 4)), ('gelu', QuickGELU()), ('c_proj', nn.Linear(d_model * 4, d_model))]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: 'torch.Tensor'):
        self.attn_mask = self.attn_mask if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: 'torch.Tensor'):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class StyleAdapter(nn.Module):

    def __init__(self, width=1024, context_dim=768, num_head=8, n_layes=3, num_token=4):
        super().__init__()
        scale = width ** -0.5
        self.transformer_layes = nn.Sequential(*[ResidualAttentionBlock(width, num_head) for _ in range(n_layes)])
        self.num_token = num_token
        self.style_embedding = nn.Parameter(torch.randn(1, num_token, width) * scale)
        self.ln_post = LayerNorm(width)
        self.ln_pre = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, context_dim))

    def forward(self, x):
        style_embedding = self.style_embedding + torch.zeros((x.shape[0], self.num_token, self.style_embedding.shape[-1]), device=x.device)
        x = torch.cat([x, style_embedding], dim=1)
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)
        x = self.transformer_layes(x)
        x = x.permute(1, 0, 2)
        x = self.ln_post(x[:, -self.num_token:, :])
        x = x @ self.proj
        return x


class ResnetBlock_light(nn.Module):

    def __init__(self, in_c):
        super().__init__()
        self.block1 = nn.Conv2d(in_c, in_c, 3, 1, 1)
        self.act = nn.ReLU()
        self.block2 = nn.Conv2d(in_c, in_c, 3, 1, 1)

    def forward(self, x):
        h = self.block1(x)
        h = self.act(h)
        h = self.block2(h)
        return h + x


class extractor(nn.Module):

    def __init__(self, in_c, inter_c, out_c, nums_rb, down=False):
        super().__init__()
        self.in_conv = nn.Conv2d(in_c, inter_c, 1, 1, 0)
        self.body = []
        for _ in range(nums_rb):
            self.body.append(ResnetBlock_light(inter_c))
        self.body = nn.Sequential(*self.body)
        self.out_conv = nn.Conv2d(inter_c, out_c, 1, 1, 0)
        self.down = down
        if self.down == True:
            self.down_opt = Downsample(in_c, use_conv=False)

    def forward(self, x):
        if self.down == True:
            x = self.down_opt(x)
        x = self.in_conv(x)
        x = self.body(x)
        x = self.out_conv(x)
        return x


class Adapter_light(nn.Module):

    def __init__(self, channels=[320, 640, 1280, 1280], nums_rb=3, cin=64):
        super(Adapter_light, self).__init__()
        self.unshuffle_amount = 8
        self.unshuffle = nn.PixelUnshuffle(self.unshuffle_amount)
        self.input_channels = cin // (self.unshuffle_amount * self.unshuffle_amount)
        self.channels = channels
        self.nums_rb = nums_rb
        self.body = []
        self.xl = False
        for i in range(len(channels)):
            if i == 0:
                self.body.append(extractor(in_c=cin, inter_c=channels[i] // 4, out_c=channels[i], nums_rb=nums_rb, down=False))
            else:
                self.body.append(extractor(in_c=channels[i - 1], inter_c=channels[i] // 4, out_c=channels[i], nums_rb=nums_rb, down=True))
        self.body = nn.ModuleList(self.body)

    def forward(self, x):
        x = self.unshuffle(x)
        features = []
        for i in range(len(self.channels)):
            x = self.body[i](x)
            features.append(None)
            features.append(None)
            features.append(x)
        return {'input': features[::-1]}


class TAESD(nn.Module):
    latent_magnitude = 3
    latent_shift = 0.5

    def __init__(self, encoder_path=None, decoder_path=None, latent_channels=4):
        """Initialize pretrained TAESD on the given device from the given checkpoints."""
        super().__init__()
        self.taesd_encoder = Encoder(latent_channels=latent_channels)
        self.taesd_decoder = Decoder(latent_channels=latent_channels)
        self.vae_scale = torch.nn.Parameter(torch.tensor(1.0))
        self.vae_shift = torch.nn.Parameter(torch.tensor(0.0))
        if encoder_path is not None:
            self.taesd_encoder.load_state_dict(comfy.utils.load_torch_file(encoder_path, safe_load=True))
        if decoder_path is not None:
            self.taesd_decoder.load_state_dict(comfy.utils.load_torch_file(decoder_path, safe_load=True))

    @staticmethod
    def scale_latents(x):
        """raw latents -> [0, 1]"""
        return x.div(2 * TAESD.latent_magnitude).add(TAESD.latent_shift).clamp(0, 1)

    @staticmethod
    def unscale_latents(x):
        """[0, 1] -> raw latents"""
        return x.sub(TAESD.latent_shift).mul(2 * TAESD.latent_magnitude)

    def decode(self, x):
        x_sample = self.taesd_decoder((x - self.vae_shift) * self.vae_scale)
        x_sample = x_sample.sub(0.5).mul(2)
        return x_sample

    def encode(self, x):
        return self.taesd_encoder(x * 0.5 + 0.5) / self.vae_scale + self.vae_shift


class BertAttention(torch.nn.Module):

    def __init__(self, embed_dim, heads, dtype, device, operations):
        super().__init__()
        self.heads = heads
        self.query = operations.Linear(embed_dim, embed_dim, bias=True, dtype=dtype, device=device)
        self.key = operations.Linear(embed_dim, embed_dim, bias=True, dtype=dtype, device=device)
        self.value = operations.Linear(embed_dim, embed_dim, bias=True, dtype=dtype, device=device)

    def forward(self, x, mask=None, optimized_attention=None):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        out = optimized_attention(q, k, v, self.heads, mask)
        return out


class BertOutput(torch.nn.Module):

    def __init__(self, input_dim, output_dim, layer_norm_eps, dtype, device, operations):
        super().__init__()
        self.dense = operations.Linear(input_dim, output_dim, dtype=dtype, device=device)
        self.LayerNorm = operations.LayerNorm(output_dim, eps=layer_norm_eps, dtype=dtype, device=device)

    def forward(self, x, y):
        x = self.dense(x)
        x = self.LayerNorm(x + y)
        return x


class BertAttentionBlock(torch.nn.Module):

    def __init__(self, embed_dim, heads, layer_norm_eps, dtype, device, operations):
        super().__init__()
        self.self = BertAttention(embed_dim, heads, dtype, device, operations)
        self.output = BertOutput(embed_dim, embed_dim, layer_norm_eps, dtype, device, operations)

    def forward(self, x, mask, optimized_attention):
        y = self.self(x, mask, optimized_attention)
        return self.output(y, x)


class BertIntermediate(torch.nn.Module):

    def __init__(self, embed_dim, intermediate_dim, dtype, device, operations):
        super().__init__()
        self.dense = operations.Linear(embed_dim, intermediate_dim, dtype=dtype, device=device)

    def forward(self, x):
        x = self.dense(x)
        return torch.nn.functional.gelu(x)


class BertBlock(torch.nn.Module):

    def __init__(self, embed_dim, intermediate_dim, heads, layer_norm_eps, dtype, device, operations):
        super().__init__()
        self.attention = BertAttentionBlock(embed_dim, heads, layer_norm_eps, dtype, device, operations)
        self.intermediate = BertIntermediate(embed_dim, intermediate_dim, dtype, device, operations)
        self.output = BertOutput(intermediate_dim, embed_dim, layer_norm_eps, dtype, device, operations)

    def forward(self, x, mask, optimized_attention):
        x = self.attention(x, mask, optimized_attention)
        y = self.intermediate(x)
        return self.output(y, x)


class BertEncoder(torch.nn.Module):

    def __init__(self, num_layers, embed_dim, intermediate_dim, heads, layer_norm_eps, dtype, device, operations):
        super().__init__()
        self.layer = torch.nn.ModuleList([BertBlock(embed_dim, intermediate_dim, heads, layer_norm_eps, dtype, device, operations) for i in range(num_layers)])

    def forward(self, x, mask=None, intermediate_output=None):
        optimized_attention = optimized_attention_for_device(x.device, mask=mask is not None, small_input=True)
        if intermediate_output is not None:
            if intermediate_output < 0:
                intermediate_output = len(self.layer) + intermediate_output
        intermediate = None
        for i, l in enumerate(self.layer):
            x = l(x, mask, optimized_attention)
            if i == intermediate_output:
                intermediate = x.clone()
        return x, intermediate


class BertEmbeddings(torch.nn.Module):

    def __init__(self, vocab_size, max_position_embeddings, type_vocab_size, pad_token_id, embed_dim, layer_norm_eps, dtype, device, operations):
        super().__init__()
        self.word_embeddings = operations.Embedding(vocab_size, embed_dim, padding_idx=pad_token_id, dtype=dtype, device=device)
        self.position_embeddings = operations.Embedding(max_position_embeddings, embed_dim, dtype=dtype, device=device)
        self.token_type_embeddings = operations.Embedding(type_vocab_size, embed_dim, dtype=dtype, device=device)
        self.LayerNorm = operations.LayerNorm(embed_dim, eps=layer_norm_eps, dtype=dtype, device=device)

    def forward(self, input_tokens, token_type_ids=None, dtype=None):
        x = self.word_embeddings(input_tokens, out_dtype=dtype)
        x += comfy.ops.cast_to_input(self.position_embeddings.weight[:x.shape[1]], x)
        if token_type_ids is not None:
            x += self.token_type_embeddings(token_type_ids, out_dtype=x.dtype)
        else:
            x += comfy.ops.cast_to_input(self.token_type_embeddings.weight[0], x)
        x = self.LayerNorm(x)
        return x


class BertModel_(torch.nn.Module):

    def __init__(self, config_dict, dtype, device, operations):
        super().__init__()
        embed_dim = config_dict['hidden_size']
        layer_norm_eps = config_dict['layer_norm_eps']
        self.embeddings = BertEmbeddings(config_dict['vocab_size'], config_dict['max_position_embeddings'], config_dict['type_vocab_size'], config_dict['pad_token_id'], embed_dim, layer_norm_eps, dtype, device, operations)
        self.encoder = BertEncoder(config_dict['num_hidden_layers'], embed_dim, config_dict['intermediate_size'], config_dict['num_attention_heads'], layer_norm_eps, dtype, device, operations)

    def forward(self, input_tokens, attention_mask=None, intermediate_output=None, final_layer_norm_intermediate=True, dtype=None):
        x = self.embeddings(input_tokens, dtype=dtype)
        mask = None
        if attention_mask is not None:
            mask = 1.0 - attention_mask.reshape((attention_mask.shape[0], 1, -1, attention_mask.shape[-1])).expand(attention_mask.shape[0], 1, attention_mask.shape[-1], attention_mask.shape[-1])
            mask = mask.masked_fill(mask, float('-inf'))
        x, i = self.encoder(x, mask, intermediate_output)
        return x, i


class BertModel(torch.nn.Module):

    def __init__(self, config_dict, dtype, device, operations):
        super().__init__()
        self.bert = BertModel_(config_dict, dtype, device, operations)
        self.num_layers = config_dict['num_hidden_layers']

    def get_input_embeddings(self):
        return self.bert.embeddings.word_embeddings

    def set_input_embeddings(self, embeddings):
        self.bert.embeddings.word_embeddings = embeddings

    def forward(self, *args, **kwargs):
        return self.bert(*args, **kwargs)


class FluxClipModel(torch.nn.Module):

    def __init__(self, dtype_t5=None, device='cpu', dtype=None, model_options={}):
        super().__init__()
        dtype_t5 = comfy.model_management.pick_weight_dtype(dtype_t5, dtype, device)
        clip_l_class = model_options.get('clip_l_class', sd1_clip.SDClipModel)
        self.clip_l = clip_l_class(device=device, dtype=dtype, return_projected_pooled=False, model_options=model_options)
        self.t5xxl = comfy.text_encoders.sd3_clip.T5XXLModel(device=device, dtype=dtype_t5, model_options=model_options)
        self.dtypes = set([dtype, dtype_t5])

    def set_clip_options(self, options):
        self.clip_l.set_clip_options(options)
        self.t5xxl.set_clip_options(options)

    def reset_clip_options(self):
        self.clip_l.reset_clip_options()
        self.t5xxl.reset_clip_options()

    def encode_token_weights(self, token_weight_pairs):
        token_weight_pairs_l = token_weight_pairs['l']
        token_weight_pairs_t5 = token_weight_pairs['t5xxl']
        t5_out, t5_pooled = self.t5xxl.encode_token_weights(token_weight_pairs_t5)
        l_out, l_pooled = self.clip_l.encode_token_weights(token_weight_pairs_l)
        return t5_out, l_pooled

    def load_sd(self, sd):
        if 'text_model.encoder.layers.1.mlp.fc1.weight' in sd:
            return self.clip_l.load_sd(sd)
        else:
            return self.t5xxl.load_sd(sd)


class HyditModel(torch.nn.Module):

    def __init__(self, device='cpu', dtype=None, model_options={}):
        super().__init__()
        self.hydit_clip = HyditBertModel(dtype=dtype, model_options=model_options)
        self.mt5xl = MT5XLModel(dtype=dtype, model_options=model_options)
        self.dtypes = set()
        if dtype is not None:
            self.dtypes.add(dtype)

    def encode_token_weights(self, token_weight_pairs):
        hydit_out = self.hydit_clip.encode_token_weights(token_weight_pairs['hydit_clip'])
        mt5_out = self.mt5xl.encode_token_weights(token_weight_pairs['mt5xl'])
        return hydit_out[0], hydit_out[1], {'attention_mask': hydit_out[2]['attention_mask'], 'conditioning_mt5xl': mt5_out[0], 'attention_mask_mt5xl': mt5_out[2]['attention_mask']}

    def load_sd(self, sd):
        if 'bert.encoder.layer.0.attention.self.query.weight' in sd:
            return self.hydit_clip.load_sd(sd)
        else:
            return self.mt5xl.load_sd(sd)

    def set_clip_options(self, options):
        self.hydit_clip.set_clip_options(options)
        self.mt5xl.set_clip_options(options)

    def reset_clip_options(self):
        self.hydit_clip.reset_clip_options()
        self.mt5xl.reset_clip_options()


class SD3ClipModel(torch.nn.Module):

    def __init__(self, clip_l=True, clip_g=True, t5=True, dtype_t5=None, t5_attention_mask=False, device='cpu', dtype=None, model_options={}):
        super().__init__()
        self.dtypes = set()
        if clip_l:
            clip_l_class = model_options.get('clip_l_class', sd1_clip.SDClipModel)
            self.clip_l = clip_l_class(layer='hidden', layer_idx=-2, device=device, dtype=dtype, layer_norm_hidden_state=False, return_projected_pooled=False, model_options=model_options)
            self.dtypes.add(dtype)
        else:
            self.clip_l = None
        if clip_g:
            self.clip_g = sdxl_clip.SDXLClipG(device=device, dtype=dtype, model_options=model_options)
            self.dtypes.add(dtype)
        else:
            self.clip_g = None
        if t5:
            dtype_t5 = comfy.model_management.pick_weight_dtype(dtype_t5, dtype, device)
            self.t5_attention_mask = t5_attention_mask
            self.t5xxl = T5XXLModel(device=device, dtype=dtype_t5, model_options=model_options, attention_mask=self.t5_attention_mask)
            self.dtypes.add(dtype_t5)
        else:
            self.t5xxl = None
        logging.debug('Created SD3 text encoder with: clip_l {}, clip_g {}, t5xxl {}:{}'.format(clip_l, clip_g, t5, dtype_t5))

    def set_clip_options(self, options):
        if self.clip_l is not None:
            self.clip_l.set_clip_options(options)
        if self.clip_g is not None:
            self.clip_g.set_clip_options(options)
        if self.t5xxl is not None:
            self.t5xxl.set_clip_options(options)

    def reset_clip_options(self):
        if self.clip_l is not None:
            self.clip_l.reset_clip_options()
        if self.clip_g is not None:
            self.clip_g.reset_clip_options()
        if self.t5xxl is not None:
            self.t5xxl.reset_clip_options()

    def encode_token_weights(self, token_weight_pairs):
        token_weight_pairs_l = token_weight_pairs['l']
        token_weight_pairs_g = token_weight_pairs['g']
        token_weight_pairs_t5 = token_weight_pairs['t5xxl']
        lg_out = None
        pooled = None
        out = None
        extra = {}
        if len(token_weight_pairs_g) > 0 or len(token_weight_pairs_l) > 0:
            if self.clip_l is not None:
                lg_out, l_pooled = self.clip_l.encode_token_weights(token_weight_pairs_l)
            else:
                l_pooled = torch.zeros((1, 768), device=comfy.model_management.intermediate_device())
            if self.clip_g is not None:
                g_out, g_pooled = self.clip_g.encode_token_weights(token_weight_pairs_g)
                if lg_out is not None:
                    cut_to = min(lg_out.shape[1], g_out.shape[1])
                    lg_out = torch.cat([lg_out[:, :cut_to], g_out[:, :cut_to]], dim=-1)
                else:
                    lg_out = torch.nn.functional.pad(g_out, (768, 0))
            else:
                g_out = None
                g_pooled = torch.zeros((1, 1280), device=comfy.model_management.intermediate_device())
            if lg_out is not None:
                lg_out = torch.nn.functional.pad(lg_out, (0, 4096 - lg_out.shape[-1]))
                out = lg_out
            pooled = torch.cat((l_pooled, g_pooled), dim=-1)
        if self.t5xxl is not None:
            t5_output = self.t5xxl.encode_token_weights(token_weight_pairs_t5)
            t5_out, t5_pooled = t5_output[:2]
            if self.t5_attention_mask:
                extra['attention_mask'] = t5_output[2]['attention_mask']
            if lg_out is not None:
                out = torch.cat([lg_out, t5_out], dim=-2)
            else:
                out = t5_out
        if out is None:
            out = torch.zeros((1, 77, 4096), device=comfy.model_management.intermediate_device())
        if pooled is None:
            pooled = torch.zeros((1, 768 + 1280), device=comfy.model_management.intermediate_device())
        return out, pooled, extra

    def load_sd(self, sd):
        if 'text_model.encoder.layers.30.mlp.fc1.weight' in sd:
            return self.clip_g.load_sd(sd)
        elif 'text_model.encoder.layers.1.mlp.fc1.weight' in sd:
            return self.clip_l.load_sd(sd)
        else:
            return self.t5xxl.load_sd(sd)


class T5LayerNorm(torch.nn.Module):

    def __init__(self, hidden_size, eps=1e-06, dtype=None, device=None, operations=None):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(hidden_size, dtype=dtype, device=device))
        self.variance_epsilon = eps

    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        return comfy.ops.cast_to_input(self.weight, x) * x


activations = {'gelu_pytorch_tanh': lambda a: torch.nn.functional.gelu(a, approximate='tanh'), 'relu': torch.nn.functional.relu}


class T5DenseActDense(torch.nn.Module):

    def __init__(self, model_dim, ff_dim, ff_activation, dtype, device, operations):
        super().__init__()
        self.wi = operations.Linear(model_dim, ff_dim, bias=False, dtype=dtype, device=device)
        self.wo = operations.Linear(ff_dim, model_dim, bias=False, dtype=dtype, device=device)
        self.act = activations[ff_activation]

    def forward(self, x):
        x = self.act(self.wi(x))
        x = self.wo(x)
        return x


class T5DenseGatedActDense(torch.nn.Module):

    def __init__(self, model_dim, ff_dim, ff_activation, dtype, device, operations):
        super().__init__()
        self.wi_0 = operations.Linear(model_dim, ff_dim, bias=False, dtype=dtype, device=device)
        self.wi_1 = operations.Linear(model_dim, ff_dim, bias=False, dtype=dtype, device=device)
        self.wo = operations.Linear(ff_dim, model_dim, bias=False, dtype=dtype, device=device)
        self.act = activations[ff_activation]

    def forward(self, x):
        hidden_gelu = self.act(self.wi_0(x))
        hidden_linear = self.wi_1(x)
        x = hidden_gelu * hidden_linear
        x = self.wo(x)
        return x


class T5LayerFF(torch.nn.Module):

    def __init__(self, model_dim, ff_dim, ff_activation, gated_act, dtype, device, operations):
        super().__init__()
        if gated_act:
            self.DenseReluDense = T5DenseGatedActDense(model_dim, ff_dim, ff_activation, dtype, device, operations)
        else:
            self.DenseReluDense = T5DenseActDense(model_dim, ff_dim, ff_activation, dtype, device, operations)
        self.layer_norm = T5LayerNorm(model_dim, dtype=dtype, device=device, operations=operations)

    def forward(self, x):
        forwarded_states = self.layer_norm(x)
        forwarded_states = self.DenseReluDense(forwarded_states)
        x += forwarded_states
        return x


class T5Attention(torch.nn.Module):

    def __init__(self, model_dim, inner_dim, num_heads, relative_attention_bias, dtype, device, operations):
        super().__init__()
        self.q = operations.Linear(model_dim, inner_dim, bias=False, dtype=dtype, device=device)
        self.k = operations.Linear(model_dim, inner_dim, bias=False, dtype=dtype, device=device)
        self.v = operations.Linear(model_dim, inner_dim, bias=False, dtype=dtype, device=device)
        self.o = operations.Linear(inner_dim, model_dim, bias=False, dtype=dtype, device=device)
        self.num_heads = num_heads
        self.relative_attention_bias = None
        if relative_attention_bias:
            self.relative_attention_num_buckets = 32
            self.relative_attention_max_distance = 128
            self.relative_attention_bias = operations.Embedding(self.relative_attention_num_buckets, self.num_heads, device=device, dtype=dtype)

    @staticmethod
    def _relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
        """
        Adapted from Mesh Tensorflow:
        https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593

        Translate relative position to a bucket number for relative attention. The relative position is defined as
        memory_position - query_position, i.e. the distance in tokens from the attending position to the attended-to
        position. If bidirectional=False, then positive relative positions are invalid. We use smaller buckets for
        small absolute relative_position and larger buckets for larger absolute relative_positions. All relative
        positions >=max_distance map to the same bucket. All relative positions <=-max_distance map to the same bucket.
        This should allow for more graceful generalization to longer sequences than the model has been trained on

        Args:
            relative_position: an int32 Tensor
            bidirectional: a boolean - whether the attention is bidirectional
            num_buckets: an integer
            max_distance: an integer

        Returns:
            a Tensor with the same shape as relative_position, containing int32 values in the range [0, num_buckets)
        """
        relative_buckets = 0
        if bidirectional:
            num_buckets //= 2
            relative_buckets += (relative_position > 0) * num_buckets
            relative_position = torch.abs(relative_position)
        else:
            relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact
        relative_position_if_large = max_exact + torch.log(relative_position.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
        relative_position_if_large = torch.min(relative_position_if_large, torch.full_like(relative_position_if_large, num_buckets - 1))
        relative_buckets += torch.where(is_small, relative_position, relative_position_if_large)
        return relative_buckets

    def compute_bias(self, query_length, key_length, device, dtype):
        """Compute binned relative position bias"""
        context_position = torch.arange(query_length, dtype=torch.long, device=device)[:, None]
        memory_position = torch.arange(key_length, dtype=torch.long, device=device)[None, :]
        relative_position = memory_position - context_position
        relative_position_bucket = self._relative_position_bucket(relative_position, bidirectional=True, num_buckets=self.relative_attention_num_buckets, max_distance=self.relative_attention_max_distance)
        values = self.relative_attention_bias(relative_position_bucket, out_dtype=dtype)
        values = values.permute([2, 0, 1]).unsqueeze(0)
        return values

    def forward(self, x, mask=None, past_bias=None, optimized_attention=None):
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        if self.relative_attention_bias is not None:
            past_bias = self.compute_bias(x.shape[1], x.shape[1], x.device, x.dtype)
        if past_bias is not None:
            if mask is not None:
                mask = mask + past_bias
            else:
                mask = past_bias
        out = optimized_attention(q, k * (k.shape[-1] / self.num_heads) ** 0.5, v, self.num_heads, mask)
        return self.o(out), past_bias


class T5LayerSelfAttention(torch.nn.Module):

    def __init__(self, model_dim, inner_dim, ff_dim, num_heads, relative_attention_bias, dtype, device, operations):
        super().__init__()
        self.SelfAttention = T5Attention(model_dim, inner_dim, num_heads, relative_attention_bias, dtype, device, operations)
        self.layer_norm = T5LayerNorm(model_dim, dtype=dtype, device=device, operations=operations)

    def forward(self, x, mask=None, past_bias=None, optimized_attention=None):
        normed_hidden_states = self.layer_norm(x)
        output, past_bias = self.SelfAttention(self.layer_norm(x), mask=mask, past_bias=past_bias, optimized_attention=optimized_attention)
        x += output
        return x, past_bias


class T5Block(torch.nn.Module):

    def __init__(self, model_dim, inner_dim, ff_dim, ff_activation, gated_act, num_heads, relative_attention_bias, dtype, device, operations):
        super().__init__()
        self.layer = torch.nn.ModuleList()
        self.layer.append(T5LayerSelfAttention(model_dim, inner_dim, ff_dim, num_heads, relative_attention_bias, dtype, device, operations))
        self.layer.append(T5LayerFF(model_dim, ff_dim, ff_activation, gated_act, dtype, device, operations))

    def forward(self, x, mask=None, past_bias=None, optimized_attention=None):
        x, past_bias = self.layer[0](x, mask, past_bias, optimized_attention)
        x = self.layer[-1](x)
        return x, past_bias


class T5Stack(torch.nn.Module):

    def __init__(self, num_layers, model_dim, inner_dim, ff_dim, ff_activation, gated_act, num_heads, relative_attention, dtype, device, operations):
        super().__init__()
        self.block = torch.nn.ModuleList([T5Block(model_dim, inner_dim, ff_dim, ff_activation, gated_act, num_heads, relative_attention_bias=not relative_attention or i == 0, dtype=dtype, device=device, operations=operations) for i in range(num_layers)])
        self.final_layer_norm = T5LayerNorm(model_dim, dtype=dtype, device=device, operations=operations)

    def forward(self, x, attention_mask=None, intermediate_output=None, final_layer_norm_intermediate=True, dtype=None):
        mask = None
        if attention_mask is not None:
            mask = 1.0 - attention_mask.reshape((attention_mask.shape[0], 1, -1, attention_mask.shape[-1])).expand(attention_mask.shape[0], 1, attention_mask.shape[-1], attention_mask.shape[-1])
            mask = mask.masked_fill(mask, float('-inf'))
        intermediate = None
        optimized_attention = optimized_attention_for_device(x.device, mask=attention_mask is not None, small_input=True)
        past_bias = None
        for i, l in enumerate(self.block):
            x, past_bias = l(x, mask, past_bias, optimized_attention)
            if i == intermediate_output:
                intermediate = x.clone()
        x = self.final_layer_norm(x)
        if intermediate is not None and final_layer_norm_intermediate:
            intermediate = self.final_layer_norm(intermediate)
        return x, intermediate


class T5(torch.nn.Module):

    def __init__(self, config_dict, dtype, device, operations):
        super().__init__()
        self.num_layers = config_dict['num_layers']
        model_dim = config_dict['d_model']
        self.encoder = T5Stack(self.num_layers, model_dim, model_dim, config_dict['d_ff'], config_dict['dense_act_fn'], config_dict['is_gated_act'], config_dict['num_heads'], config_dict['model_type'] != 'umt5', dtype, device, operations)
        self.dtype = dtype
        self.shared = operations.Embedding(config_dict['vocab_size'], model_dim, device=device, dtype=dtype)

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, embeddings):
        self.shared = embeddings

    def forward(self, input_ids, *args, **kwargs):
        x = self.shared(input_ids, out_dtype=kwargs.get('dtype', torch.float32))
        if self.dtype not in [torch.float32, torch.float16, torch.bfloat16]:
            x = torch.nan_to_num(x)
        return self.encoder(x, *args, **kwargs)


class FuseModule(nn.Module):

    def __init__(self, embed_dim, operations):
        super().__init__()
        self.mlp1 = MLP(embed_dim * 2, embed_dim, embed_dim, use_residual=False, operations=operations)
        self.mlp2 = MLP(embed_dim, embed_dim, embed_dim, use_residual=True, operations=operations)
        self.layer_norm = operations.LayerNorm(embed_dim)

    def fuse_fn(self, prompt_embeds, id_embeds):
        stacked_id_embeds = torch.cat([prompt_embeds, id_embeds], dim=-1)
        stacked_id_embeds = self.mlp1(stacked_id_embeds) + prompt_embeds
        stacked_id_embeds = self.mlp2(stacked_id_embeds)
        stacked_id_embeds = self.layer_norm(stacked_id_embeds)
        return stacked_id_embeds

    def forward(self, prompt_embeds, id_embeds, class_tokens_mask) ->torch.Tensor:
        id_embeds = id_embeds
        num_inputs = class_tokens_mask.sum().unsqueeze(0)
        batch_size, max_num_inputs = id_embeds.shape[:2]
        seq_length = prompt_embeds.shape[1]
        flat_id_embeds = id_embeds.view(-1, id_embeds.shape[-2], id_embeds.shape[-1])
        valid_id_mask = torch.arange(max_num_inputs, device=flat_id_embeds.device)[None, :] < num_inputs[:, None]
        valid_id_embeds = flat_id_embeds[valid_id_mask.flatten()]
        prompt_embeds = prompt_embeds.view(-1, prompt_embeds.shape[-1])
        class_tokens_mask = class_tokens_mask.view(-1)
        valid_id_embeds = valid_id_embeds.view(-1, valid_id_embeds.shape[-1])
        image_token_embeds = prompt_embeds[class_tokens_mask]
        stacked_id_embeds = self.fuse_fn(image_token_embeds, valid_id_embeds)
        assert class_tokens_mask.sum() == stacked_id_embeds.shape[0], f'{class_tokens_mask.sum()} != {stacked_id_embeds.shape[0]}'
        prompt_embeds.masked_scatter_(class_tokens_mask[:, None], stacked_id_embeds)
        updated_prompt_embeds = prompt_embeds.view(batch_size, seq_length, -1)
        return updated_prompt_embeds


VISION_CONFIG_DICT = {'hidden_size': 1024, 'image_size': 224, 'intermediate_size': 4096, 'num_attention_heads': 16, 'num_channels': 3, 'num_hidden_layers': 24, 'patch_size': 14, 'projection_dim': 768, 'hidden_act': 'quick_gelu'}


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (AbsolutePositionalEmbedding,
     lambda: ([], {'dim': 4, 'max_seq_len': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (AbstractLowScaleModel,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Clamp,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Conv2DWrapper,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (DiagonalGaussianRegularizer,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Downsample,
     lambda: ([], {'channels': 4, 'use_conv': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (EfficientNetEncoder,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 4, 4])], {})),
    (LayerNorm,
     lambda: ([], {'normalized_shape': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (QuickGELU,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ResidualAttentionBlock,
     lambda: ([], {'d_model': 4, 'n_head': 4}),
     lambda: ([torch.rand([4, 4])], {})),
    (ResnetBlock,
     lambda: ([], {'in_c': 4, 'out_c': 4, 'down': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ResnetBlock_light,
     lambda: ([], {'in_c': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ScaledSinusoidalEmbedding,
     lambda: ([], {'dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (SimpleImageConcat,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (SnakeBeta,
     lambda: ([], {'in_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (SwiGLUFeedForward,
     lambda: ([], {'dim': 4, 'hidden_dim': 4, 'multiple_of': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (TimestepBlock,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (TimestepEmbedSequential,
     lambda: ([], {}),
     lambda: ([], {'x': 4, 'emb': 4})),
    (Upsample,
     lambda: ([], {'channels': 4, 'use_conv': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (VectorQuantize,
     lambda: ([], {'embedding_size': 4, 'k': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (extractor,
     lambda: ([], {'in_c': 4, 'inter_c': 4, 'out_c': 4, 'nums_rb': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
]

