
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


from typing import Tuple


import torch


from torch.utils.data import DataLoader


from torch.utils.data import Dataset


from torch.utils.data import random_split


from torchvision.datasets import Caltech101


from torchvision.datasets import CIFAR10


from torchvision.datasets import CIFAR100


from torch.utils.data import ConcatDataset


from torchvision.datasets import DTD


import torchvision


from torchvision.datasets import FGVCAircraft


from torchvision.datasets import Flowers102


from torchvision.datasets import Food101


from torchvision.datasets import OxfordIIITPet


from torchvision.datasets import StanfordCars


from torchvision.datasets import SUN397


import math


import random


import numpy as np


from torch import distributed as dist


from torch.utils.data import DistributedSampler as _DistributedSampler


import logging


import re


from functools import partial


from torch import optim


from torch.cuda.amp import GradScaler


from typing import Optional


from torch import nn


from torch.nn import functional as F


from copy import deepcopy


from typing import Any


from typing import Dict


from typing import Union


import torch.nn as nn


from torch import TensorType


import copy


import torch.nn.functional as F


from torch.utils.checkpoint import checkpoint


from collections import OrderedDict


import warnings


from typing import List


import string


from functools import lru_cache


from typing import Callable


import numbers


from typing import Sequence


import torchvision.transforms.functional as F


from torchvision.transforms import Normalize


from torchvision.transforms import Compose


from torchvision.transforms import RandomResizedCrop


from torchvision.transforms import InterpolationMode


from torchvision.transforms import ToTensor


from torchvision.transforms import Resize


from torchvision.transforms import CenterCrop


from torchvision.transforms import ColorJitter


from torchvision.transforms import Grayscale


from torchvision.transforms import RandomApply


from torchvision.transforms import RandomGrayscale


from torchvision.transforms import RandomHorizontalFlip


from itertools import repeat


import collections.abc


from torch import nn as nn


from torchvision.ops.misc import FrozenBatchNorm2d


from itertools import islice


import pandas as pd


import torchvision.datasets as datasets


from torch.utils.data import SubsetRandomSampler


from torch.utils.data import IterableDataset


from torch.utils.data import get_worker_info


from torch.utils.data.distributed import DistributedSampler


import torch.distributed as dist


import time


from torch.utils.flop_counter import FlopCounterMode


from torchvision import transforms


from torch.utils.data import default_collate


from numpy.typing import NDArray


import types


from torch.nn.parallel.distributed import DistributedDataParallel


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm (with cast back to input dtype)."""

    def forward(self, x: 'torch.Tensor'):
        orig_type = x.dtype
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        return x


class LayerNormFp32(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16 (by casting to float32 and back)."""

    def forward(self, x: 'torch.Tensor'):
        orig_type = x.dtype
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        return x


class LayerScale(nn.Module):

    def __init__(self, dim, init_values=1e-05, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class ResidualAttentionBlock(nn.Module):

    def __init__(self, d_model: 'int', n_head: 'int', mlp_ratio: 'float'=4.0, ls_init_value: 'float'=None, act_layer: 'Callable'=nn.GELU, norm_layer: 'Callable'=LayerNorm, is_cross_attention: 'bool'=False):
        super().__init__()
        self.ln_1 = norm_layer(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ls_1 = LayerScale(d_model, ls_init_value) if ls_init_value is not None else nn.Identity()
        if is_cross_attention:
            self.ln_1_kv = norm_layer(d_model)
        self.ln_2 = norm_layer(d_model)
        mlp_width = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(OrderedDict([('c_fc', nn.Linear(d_model, mlp_width)), ('gelu', act_layer()), ('c_proj', nn.Linear(mlp_width, d_model))]))
        self.ls_2 = LayerScale(d_model, ls_init_value) if ls_init_value is not None else nn.Identity()

    def attention(self, q_x: 'torch.Tensor', k_x: 'Optional[torch.Tensor]'=None, v_x: 'Optional[torch.Tensor]'=None, attn_mask: 'Optional[torch.Tensor]'=None):
        k_x = k_x if k_x is not None else q_x
        v_x = v_x if v_x is not None else q_x
        attn_mask = attn_mask if attn_mask is not None else None
        return self.attn(q_x, k_x, v_x, need_weights=False, attn_mask=attn_mask)[0]

    def forward(self, q_x: 'torch.Tensor', k_x: 'Optional[torch.Tensor]'=None, v_x: 'Optional[torch.Tensor]'=None, attn_mask: 'Optional[torch.Tensor]'=None):
        k_x = self.ln_1_kv(k_x) if hasattr(self, 'ln_1_kv') and k_x is not None else None
        v_x = self.ln_1_kv(v_x) if hasattr(self, 'ln_1_kv') and v_x is not None else None
        x = q_x + self.ls_1(self.attention(q_x=self.ln_1(q_x), k_x=k_x, v_x=v_x, attn_mask=attn_mask))
        x = x + self.ls_2(self.mlp(self.ln_2(x)))
        return x


class Transformer(nn.Module):

    def __init__(self, width: 'int', layers: 'int', heads: 'int', mlp_ratio: 'float'=4.0, ls_init_value: 'float'=None, act_layer: 'Callable'=nn.GELU, norm_layer: 'Callable'=LayerNorm):
        super().__init__()
        self.width = width
        self.layers = layers
        self.grad_checkpointing = False
        self.resblocks = nn.ModuleList([ResidualAttentionBlock(width, heads, mlp_ratio, ls_init_value=ls_init_value, act_layer=act_layer, norm_layer=norm_layer) for _ in range(layers)])

    def get_cast_dtype(self) ->torch.dtype:
        if hasattr(self.resblocks[0].mlp.c_fc, 'int8_original_dtype'):
            return self.resblocks[0].mlp.c_fc.int8_original_dtype
        return self.resblocks[0].mlp.c_fc.weight.dtype

    def forward(self, x: 'torch.Tensor', attn_mask: 'Optional[torch.Tensor]'=None):
        for r in self.resblocks:
            if self.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(r, x, None, None, attn_mask)
            else:
                x = r(x, attn_mask=attn_mask)
        return x


class MultimodalTransformer(Transformer):

    def __init__(self, width: 'int', layers: 'int', heads: 'int', context_length: 'int'=77, mlp_ratio: 'float'=4.0, ls_init_value: 'float'=None, act_layer: 'Callable'=nn.GELU, norm_layer: 'Callable'=LayerNorm, output_dim: 'int'=512):
        super().__init__(width=width, layers=layers, heads=heads, mlp_ratio=mlp_ratio, ls_init_value=ls_init_value, act_layer=act_layer, norm_layer=norm_layer)
        self.context_length = context_length
        self.cross_attn = nn.ModuleList([ResidualAttentionBlock(width, heads, mlp_ratio, ls_init_value=ls_init_value, act_layer=act_layer, norm_layer=norm_layer, is_cross_attention=True) for _ in range(layers)])
        self.register_buffer('attn_mask', self.build_attention_mask(), persistent=False)
        self.ln_final = norm_layer(width)
        self.text_projection = nn.Parameter(torch.empty(width, output_dim))

    def init_parameters(self):
        proj_std = self.transformer.width ** -0.5 * (2 * self.transformer.layers) ** -0.5
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)
        for block in self.transformer.cross_attn:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)
        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float('-inf'))
        mask.triu_(1)
        return mask

    def forward(self, image_embs, text_embs):
        text_embs = text_embs.permute(1, 0, 2)
        image_embs = image_embs.permute(1, 0, 2)
        seq_len = text_embs.shape[0]
        for resblock, cross_attn in zip(self.resblocks, self.cross_attn):
            if self.grad_checkpointing and not torch.jit.is_scripting():
                text_embs = checkpoint(resblock, text_embs, None, None, self.attn_mask[:seq_len, :seq_len])
                text_embs = checkpoint(cross_attn, text_embs, image_embs, image_embs, None)
            else:
                text_embs = resblock(text_embs, attn_mask=self.attn_mask[:seq_len, :seq_len])
                text_embs = cross_attn(text_embs, k_x=image_embs, v_x=image_embs)
        x = text_embs.permute(1, 0, 2)
        x = self.ln_final(x)
        if self.text_projection is not None:
            x = x @ self.text_projection
        return x

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable


class QuickGELU(nn.Module):

    def forward(self, x: 'torch.Tensor'):
        return x * torch.sigmoid(1.702 * x)


def _build_text_decoder_tower(embed_dim, multimodal_cfg, quick_gelu: 'bool'=False, cast_dtype: 'Optional[torch.dtype]'=None):
    multimodal_cfg = MultimodalCfg(**multimodal_cfg) if isinstance(multimodal_cfg, dict) else multimodal_cfg
    act_layer = QuickGELU if quick_gelu else nn.GELU
    norm_layer = LayerNormFp32 if cast_dtype in (torch.float16, torch.bfloat16) else LayerNorm
    decoder = MultimodalTransformer(context_length=multimodal_cfg.context_length, width=multimodal_cfg.width, heads=multimodal_cfg.heads, layers=multimodal_cfg.layers, ls_init_value=multimodal_cfg.ls_init_value, output_dim=embed_dim, act_layer=act_layer, norm_layer=norm_layer)
    return decoder


class ClsPooler(nn.Module):
    """CLS token pooling"""

    def __init__(self, use_pooler_output=True):
        super().__init__()
        self.cls_token_position = 0
        self.use_pooler_output = use_pooler_output

    def forward(self, x: 'BaseModelOutput', attention_mask: 'TensorType'):
        if self.use_pooler_output and isinstance(x, (BaseModelOutputWithPooling, BaseModelOutputWithPoolingAndCrossAttentions)) and x.pooler_output is not None:
            return x.pooler_output
        return x.last_hidden_state[:, self.cls_token_position, :]


_POOLERS = {}


arch_dict = {'roberta': {'config_names': {'context_length': 'max_position_embeddings', 'vocab_size': 'vocab_size', 'width': 'hidden_size', 'heads': 'num_attention_heads', 'layers': 'num_hidden_layers', 'layer_attr': 'layer', 'token_embeddings_attr': 'embeddings'}, 'pooler': 'mean_pooler'}, 'xlm-roberta': {'config_names': {'context_length': 'max_position_embeddings', 'vocab_size': 'vocab_size', 'width': 'hidden_size', 'heads': 'num_attention_heads', 'layers': 'num_hidden_layers', 'layer_attr': 'layer', 'token_embeddings_attr': 'embeddings'}, 'pooler': 'mean_pooler'}, 'mt5': {'config_names': {'context_length': '', 'vocab_size': 'vocab_size', 'width': 'd_model', 'heads': 'num_heads', 'layers': 'num_layers', 'layer_attr': 'block', 'token_embeddings_attr': 'embed_tokens'}, 'pooler': 'mean_pooler'}, 'bert': {'config_names': {'context_length': 'max_position_embeddings', 'vocab_size': 'vocab_size', 'width': 'hidden_size', 'heads': 'num_attention_heads', 'layers': 'num_hidden_layers'}, 'pooler': 'cls_pooler'}, 'm2m_100': {'config_names': {'context_length': 'max_position_embeddings', 'vocab_size': 'vocab_size', 'width': 'd_model', 'heads': 'encoder_attention_heads', 'layers': 'encoder_layers'}, 'pooler': 'cls_pooler'}}


class HFTextEncoder(nn.Module):
    """HuggingFace model adapter"""
    output_tokens: 'torch.jit.Final[bool]'

    def __init__(self, model_name_or_path: 'str', output_dim: 'int', config: 'PretrainedConfig'=None, pooler_type: 'str'=None, proj_type: 'str'=None, pretrained: 'bool'=True, output_tokens: 'bool'=False, cache_dir: 'str'=None):
        super().__init__()
        self.output_tokens = output_tokens
        self.output_dim = output_dim
        if '/' not in model_name_or_path:
            model_name_or_path = os.path.join(cache_dir, model_name_or_path)
        uses_transformer_pooler = pooler_type == 'cls_pooler'
        if transformers is None:
            raise RuntimeError('Please `pip install transformers` to use pre-trained HuggingFace models')
        if config is None:
            self.config = AutoConfig.from_pretrained(model_name_or_path)
            create_func, model_args = (AutoModel.from_pretrained, model_name_or_path) if pretrained else (AutoModel.from_config, self.config)
            if hasattr(self.config, 'is_encoder_decoder') and self.config.is_encoder_decoder:
                self.transformer = create_func(model_args)
                self.transformer = self.transformer.encoder
            else:
                self.transformer = create_func(model_args, add_pooling_layer=uses_transformer_pooler)
        else:
            self.config = config
            self.transformer = AutoModel.from_config(config)
        if pooler_type is None:
            pooler_type = arch_dict[self.config.model_type]['pooler']
        self.vocab_size = getattr(self.config, 'vocab_size', 0)
        self.context_length = getattr(self.config, 'max_position_embeddings', 0)
        self.pooler = _POOLERS[pooler_type]()
        d_model = getattr(self.config, arch_dict[self.config.model_type]['config_names']['width'])
        if d_model == output_dim and proj_type is None:
            self.proj = nn.Identity()
        elif proj_type == 'linear':
            self.proj = nn.Linear(d_model, output_dim, bias=False)
        elif proj_type == 'mlp':
            hidden_size = (d_model + output_dim) // 2
            self.proj = nn.Sequential(nn.Linear(d_model, hidden_size, bias=False), nn.GELU(), nn.Linear(hidden_size, output_dim, bias=False))

    def forward(self, x: 'TensorType'):
        attn_mask = (x != self.config.pad_token_id).long()
        out = self.transformer(input_ids=x, attention_mask=attn_mask)
        pooled_out = self.pooler(out, attn_mask)
        projected = self.proj(pooled_out)
        seq_len = out.last_hidden_state.shape[1]
        tokens = out.last_hidden_state[:, torch.arange(seq_len) != self.pooler.cls_token_position, :] if type(self.pooler) == ClsPooler else out.last_hidden_state
        if self.output_tokens:
            return projected, tokens
        return projected

    def lock(self, unlocked_layers: 'int'=0, freeze_layer_norm: 'bool'=True):
        if not unlocked_layers:
            for n, p in self.transformer.named_parameters():
                p.requires_grad = not freeze_layer_norm if 'LayerNorm' in n.split('.') else False
            return
        encoder = self.transformer.encoder if hasattr(self.transformer, 'encoder') else self.transformer
        layer_list = getattr(encoder, arch_dict[self.config.model_type]['config_names']['layer_attr'])
        None
        embeddings = getattr(self.transformer, arch_dict[self.config.model_type]['config_names']['token_embeddings_attr'])
        modules = [embeddings, *layer_list][:-unlocked_layers]
        for module in modules:
            for n, p in module.named_parameters():
                p.requires_grad = not freeze_layer_norm if 'LayerNorm' in n.split('.') else False

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.transformer.gradient_checkpointing_enable()

    def init_parameters(self):
        pass


def _expand_token(token, batch_size: 'int'):
    return token.view(1, 1, -1).expand(batch_size, -1, -1)


def text_global_pool(x, text: 'Optional[torch.Tensor]'=None, pool_type: 'str'='argmax'):
    if pool_type == 'first':
        pooled, tokens = x[:, 0], x[:, 1:]
    elif pool_type == 'last':
        pooled, tokens = x[:, -1], x[:, :-1]
    elif pool_type == 'argmax':
        assert text is not None
        pooled, tokens = x[torch.arange(x.shape[0]), text.argmax(dim=-1)], x
    else:
        pooled = tokens = x
    return pooled, tokens


class TextTransformer(nn.Module):
    output_tokens: 'torch.jit.Final[bool]'

    def __init__(self, context_length: 'int'=77, vocab_size: 'int'=49408, width: 'int'=512, heads: 'int'=8, layers: 'int'=12, mlp_ratio: 'float'=4.0, ls_init_value: 'float'=None, output_dim: 'int'=512, embed_cls: 'bool'=False, no_causal_mask: 'bool'=False, pad_id: 'int'=0, pool_type: 'str'='argmax', proj_bias: 'bool'=False, act_layer: 'Callable'=nn.GELU, norm_layer: 'Callable'=LayerNorm, output_tokens: 'bool'=False):
        super().__init__()
        assert pool_type in ('first', 'last', 'argmax', 'none')
        self.output_tokens = output_tokens
        self.num_pos = self.context_length = context_length
        self.vocab_size = vocab_size
        self.width = width
        self.output_dim = output_dim
        self.heads = heads
        self.pad_id = pad_id
        self.pool_type = pool_type
        self.token_embedding = nn.Embedding(vocab_size, width)
        if embed_cls:
            self.cls_emb = nn.Parameter(torch.empty(width))
            self.num_pos += 1
        else:
            self.cls_emb = None
        self.positional_embedding = nn.Parameter(torch.empty(self.num_pos, width))
        self.transformer = Transformer(width=width, layers=layers, heads=heads, mlp_ratio=mlp_ratio, ls_init_value=ls_init_value, act_layer=act_layer, norm_layer=norm_layer)
        self.ln_final = norm_layer(width)
        if no_causal_mask:
            self.attn_mask = None
        else:
            self.register_buffer('attn_mask', self.build_causal_mask(), persistent=False)
        if proj_bias:
            self.text_projection = nn.Linear(width, output_dim)
        else:
            self.text_projection = nn.Parameter(torch.empty(width, output_dim))
        self.init_parameters()

    def init_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)
        if self.cls_emb is not None:
            nn.init.normal_(self.cls_emb, std=0.01)
        proj_std = self.transformer.width ** -0.5 * (2 * self.transformer.layers) ** -0.5
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)
        if self.text_projection is not None:
            if isinstance(self.text_projection, nn.Linear):
                nn.init.normal_(self.text_projection.weight, std=self.transformer.width ** -0.5)
                if self.text_projection.bias is not None:
                    nn.init.zeros_(self.text_projection.bias)
            else:
                nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.transformer.grad_checkpointing = enable

    def build_causal_mask(self):
        mask = torch.empty(self.num_pos, self.num_pos)
        mask.fill_(float('-inf'))
        mask.triu_(1)
        return mask

    def build_cls_mask(self, text, cast_dtype: 'torch.dtype'):
        cls_mask = (text != self.pad_id).unsqueeze(1)
        cls_mask = F.pad(cls_mask, (1, 0, cls_mask.shape[2], 0), value=True)
        additive_mask = torch.empty(cls_mask.shape, dtype=cast_dtype, device=cls_mask.device)
        additive_mask.fill_(0)
        additive_mask.masked_fill_(~cls_mask, float('-inf'))
        additive_mask = torch.repeat_interleave(additive_mask, self.heads, 0)
        return additive_mask

    def forward(self, text):
        cast_dtype = self.transformer.get_cast_dtype()
        seq_len = text.shape[1]
        x = self.token_embedding(text)
        attn_mask = self.attn_mask
        if self.cls_emb is not None:
            seq_len += 1
            x = torch.cat([x, _expand_token(self.cls_emb, x.shape[0])], dim=1)
            cls_mask = self.build_cls_mask(text, cast_dtype)
            if attn_mask is not None:
                attn_mask = attn_mask[None, :seq_len, :seq_len] + cls_mask[:, :seq_len, :seq_len]
        x = x + self.positional_embedding[:seq_len]
        x = x.permute(1, 0, 2)
        x = self.transformer(x, attn_mask=attn_mask)
        x = x.permute(1, 0, 2)
        if self.cls_emb is not None:
            pooled, tokens = text_global_pool(x, pool_type='last')
            pooled = self.ln_final(pooled)
        else:
            x = self.ln_final(x)
            pooled, tokens = text_global_pool(x, text, pool_type=self.pool_type)
        if self.text_projection is not None:
            if isinstance(self.text_projection, nn.Linear):
                pooled = self.text_projection(pooled)
            else:
                pooled = pooled @ self.text_projection
        if self.output_tokens:
            return pooled, tokens
        return pooled


def _build_text_tower(embed_dim: 'int', text_cfg: 'CLIPTextCfg', quick_gelu: 'bool'=False, cast_dtype: 'Optional[torch.dtype]'=None, cache_dir: 'str'=None):
    if isinstance(text_cfg, dict):
        text_cfg = CLIPTextCfg(**text_cfg)
    if text_cfg.hf_model_name:
        text = HFTextEncoder(text_cfg.hf_model_name, output_dim=embed_dim, proj_type=text_cfg.hf_proj_type, pooler_type=text_cfg.hf_pooler_type, pretrained=text_cfg.hf_model_pretrained, output_tokens=text_cfg.output_tokens, cache_dir=cache_dir)
    else:
        act_layer = QuickGELU if quick_gelu else nn.GELU
        norm_layer = LayerNormFp32 if cast_dtype in (torch.float16, torch.bfloat16) else LayerNorm
        if text_cfg.norm_kwargs:
            norm_layer = partial(norm_layer, **text_cfg.norm_kwargs)
        if text_cfg.act_kwargs is not None:
            act_layer = partial(act_layer, **text_cfg.act_kwargs)
        text = TextTransformer(context_length=text_cfg.context_length, vocab_size=text_cfg.vocab_size, width=text_cfg.width, heads=text_cfg.heads, layers=text_cfg.layers, mlp_ratio=text_cfg.mlp_ratio, ls_init_value=text_cfg.ls_init_value, output_dim=embed_dim, embed_cls=text_cfg.embed_cls, no_causal_mask=text_cfg.no_causal_mask, pad_id=text_cfg.pad_id, pool_type=text_cfg.pool_type, proj_bias=text_cfg.proj_bias, output_tokens=text_cfg.output_tokens, act_layer=act_layer, norm_layer=norm_layer)
    return text


class AttentionPool2d(nn.Module):

    def __init__(self, spacial_dim: 'int', embed_dim: 'int', num_heads: 'int', output_dim: 'int'=None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)
        x = x + self.positional_embedding[:, None, :]
        x, _ = F.multi_head_attention_forward(query=x, key=x, value=x, embed_dim_to_check=x.shape[-1], num_heads=self.num_heads, q_proj_weight=self.q_proj.weight, k_proj_weight=self.k_proj.weight, v_proj_weight=self.v_proj.weight, in_proj_weight=None, in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]), bias_k=None, bias_v=None, add_zero_attn=False, dropout_p=0.0, out_proj_weight=self.c_proj.weight, out_proj_bias=self.c_proj.bias, use_separate_proj_weight=True, training=self.training, need_weights=False)
        return x[0]


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.act2 = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.act3 = nn.ReLU(inplace=True)
        self.downsample = None
        self.stride = stride
        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            self.downsample = nn.Sequential(OrderedDict([('-1', nn.AvgPool2d(stride)), ('0', nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)), ('1', nn.BatchNorm2d(planes * self.expansion))]))

    def forward(self, x: 'torch.Tensor'):
        identity = x
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.act2(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.act3(out)
        return out


def freeze_batch_norm_2d(module, module_match={}, name=''):
    """
    Converts all `BatchNorm2d` and `SyncBatchNorm` layers of provided module into `FrozenBatchNorm2d`. If `module` is
    itself an instance of either `BatchNorm2d` or `SyncBatchNorm`, it is converted into `FrozenBatchNorm2d` and
    returned. Otherwise, the module is walked recursively and submodules are converted in place.

    Args:
        module (torch.nn.Module): Any PyTorch module.
        module_match (dict): Dictionary of full module names to freeze (all if empty)
        name (str): Full module name (prefix)

    Returns:
        torch.nn.Module: Resulting module

    Inspired by https://github.com/pytorch/pytorch/blob/a5895f85be0f10212791145bfedc0261d364f103/torch/nn/modules/batchnorm.py#L762
    """
    res = module
    is_match = True
    if module_match:
        is_match = name in module_match
    if is_match and isinstance(module, (nn.modules.batchnorm.BatchNorm2d, nn.modules.batchnorm.SyncBatchNorm)):
        res = FrozenBatchNorm2d(module.num_features)
        res.num_features = module.num_features
        res.affine = module.affine
        if module.affine:
            res.weight.data = module.weight.data.clone().detach()
            res.bias.data = module.bias.data.clone().detach()
        res.running_mean.data = module.running_mean.data
        res.running_var.data = module.running_var.data
        res.eps = module.eps
    else:
        for child_name, child in module.named_children():
            full_child_name = '.'.join([name, child_name]) if name else child_name
            new_child = freeze_batch_norm_2d(child, module_match, full_child_name)
            if new_child is not child:
                res.add_module(child_name, new_child)
    return res


class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, image_size=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.image_size = image_size
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.act2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.act3 = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(2)
        self._inplanes = width
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)
        embed_dim = width * 32
        self.attnpool = AttentionPool2d(image_size // 32, embed_dim, heads, output_dim)
        self.init_parameters()

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]
        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))
        return nn.Sequential(*layers)

    def init_parameters(self):
        if self.attnpool is not None:
            std = self.attnpool.c_proj.in_features ** -0.5
            nn.init.normal_(self.attnpool.q_proj.weight, std=std)
            nn.init.normal_(self.attnpool.k_proj.weight, std=std)
            nn.init.normal_(self.attnpool.v_proj.weight, std=std)
            nn.init.normal_(self.attnpool.c_proj.weight, std=std)
        for resnet_block in [self.layer1, self.layer2, self.layer3, self.layer4]:
            for name, param in resnet_block.named_parameters():
                if name.endswith('bn3.weight'):
                    nn.init.zeros_(param)

    def lock(self, unlocked_groups=0, freeze_bn_stats=False):
        assert unlocked_groups == 0, 'partial locking not currently supported for this model'
        for param in self.parameters():
            param.requires_grad = False
        if freeze_bn_stats:
            freeze_batch_norm_2d(self)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        pass

    def stem(self, x):
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.act2(self.bn2(self.conv2(x)))
        x = self.act3(self.bn3(self.conv3(x)))
        x = self.avgpool(x)
        return x

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)
        return x


def _ntuple(n):

    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse


to_2tuple = _ntuple(2)


class AttentionalPooler(nn.Module):

    def __init__(self, d_model: 'int', context_dim: 'int', n_head: 'int'=8, n_queries: 'int'=256, norm_layer: 'Callable'=LayerNorm):
        super().__init__()
        self.query = nn.Parameter(torch.randn(n_queries, d_model))
        self.attn = nn.MultiheadAttention(d_model, n_head, kdim=context_dim, vdim=context_dim)
        self.ln_q = norm_layer(d_model)
        self.ln_k = norm_layer(context_dim)

    def forward(self, x: 'torch.Tensor'):
        x = self.ln_k(x).permute(1, 0, 2)
        N = x.shape[1]
        q = self.ln_q(self.query)
        out = self.attn(q.unsqueeze(1).expand(-1, N, -1), x, x, need_weights=False)[0]
        return out.permute(1, 0, 2)


class PatchDropout(nn.Module):
    """
    https://arxiv.org/abs/2212.00794
    """

    def __init__(self, prob, exclude_first_token=True):
        super().__init__()
        assert 0 <= prob < 1.0
        self.prob = prob
        self.exclude_first_token = exclude_first_token

    def forward(self, x):
        if not self.training or self.prob == 0.0:
            return x
        if self.exclude_first_token:
            cls_tokens, x = x[:, :1], x[:, 1:]
        else:
            cls_tokens = torch.jit.annotate(torch.Tensor, x[:, :1])
        batch = x.size()[0]
        num_tokens = x.size()[1]
        batch_indices = torch.arange(batch)
        batch_indices = batch_indices[..., None]
        keep_prob = 1 - self.prob
        num_patches_keep = max(1, int(num_tokens * keep_prob))
        rand = torch.randn(batch, num_tokens)
        patch_indices_keep = rand.topk(num_patches_keep, dim=-1).indices
        x = x[batch_indices, patch_indices_keep]
        if self.exclude_first_token:
            x = torch.cat((cls_tokens, x), dim=1)
        return x


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
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


class VisionTransformer(nn.Module):
    output_tokens: 'torch.jit.Final[bool]'

    def __init__(self, image_size: 'int', patch_size: 'int', width: 'int', layers: 'int', heads: 'int', mlp_ratio: 'float', ls_init_value: 'float'=None, attentional_pool: 'bool'=False, attn_pooler_queries: 'int'=256, attn_pooler_heads: 'int'=8, output_dim: 'int'=512, patch_dropout: 'float'=0.0, no_ln_pre: 'bool'=False, pos_embed_type: 'str'='learnable', pool_type: 'str'='tok', final_ln_after_pool: 'bool'=False, act_layer: 'Callable'=nn.GELU, norm_layer: 'Callable'=LayerNorm, output_tokens: 'bool'=False):
        super().__init__()
        assert pool_type in ('tok', 'avg', 'none')
        self.output_tokens = True
        image_height, image_width = self.image_size = to_2tuple(image_size)
        patch_height, patch_width = self.patch_size = to_2tuple(patch_size)
        self.grid_size = image_height // patch_height, image_width // patch_width
        self.final_ln_after_pool = final_ln_after_pool
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)
        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        if pos_embed_type == 'learnable':
            self.positional_embedding = nn.Parameter(scale * torch.randn(self.grid_size[0] * self.grid_size[1] + 1, width))
        elif pos_embed_type == 'sin_cos_2d':
            assert self.grid_size[0] == self.grid_size[1], 'currently sin cos 2d pos embedding only supports square input'
            self.positional_embedding = nn.Parameter(torch.zeros(self.grid_size[0] * self.grid_size[1] + 1, width), requires_grad=False)
            pos_embed_type = get_2d_sincos_pos_embed(width, self.grid_size[0], cls_token=True)
            self.positional_embedding.data.copy_(torch.from_numpy(pos_embed_type).float())
        else:
            raise ValueError
        self.patch_dropout = PatchDropout(patch_dropout) if patch_dropout > 0.0 else nn.Identity()
        self.ln_pre = nn.Identity() if no_ln_pre else norm_layer(width)
        self.transformer = Transformer(width, layers, heads, mlp_ratio, ls_init_value=ls_init_value, act_layer=act_layer, norm_layer=norm_layer)
        if attentional_pool:
            if isinstance(attentional_pool, str):
                self.attn_pool_type = attentional_pool
                self.pool_type = 'none'
                if attentional_pool in ('parallel', 'cascade'):
                    self.attn_pool = AttentionalPooler(output_dim, width, n_head=attn_pooler_heads, n_queries=attn_pooler_queries)
                    self.attn_pool_contrastive = AttentionalPooler(output_dim, width, n_head=attn_pooler_heads, n_queries=1)
                else:
                    assert False
            else:
                self.attn_pool_type = ''
                self.pool_type = pool_type
                self.attn_pool = AttentionalPooler(output_dim, width, n_head=attn_pooler_heads, n_queries=attn_pooler_queries)
                self.attn_pool_contrastive = None
            pool_dim = output_dim
        else:
            self.attn_pool = None
            pool_dim = width
            self.pool_type = pool_type
        self.ln_post = norm_layer(pool_dim)
        self.proj = nn.Parameter(scale * torch.randn(pool_dim, output_dim))
        self.init_parameters()

    def lock(self, unlocked_groups=0, freeze_bn_stats=False):
        for param in self.parameters():
            param.requires_grad = False
        if unlocked_groups != 0:
            groups = [[self.conv1, self.class_embedding, self.positional_embedding, self.ln_pre], *self.transformer.resblocks[:-1], [self.transformer.resblocks[-1], self.ln_post], self.proj]

            def _unlock(x):
                if isinstance(x, Sequence):
                    for g in x:
                        _unlock(g)
                elif isinstance(x, torch.nn.Parameter):
                    x.requires_grad = True
                else:
                    for p in x.parameters():
                        p.requires_grad = True
            _unlock(groups[-unlocked_groups:])

    def init_parameters(self):
        pass

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.transformer.grad_checkpointing = enable

    def _global_pool(self, x: 'torch.Tensor') ->Tuple[torch.Tensor, torch.Tensor]:
        if self.pool_type == 'avg':
            pooled, tokens = x[:, 1:].mean(dim=1), x[:, 1:]
        elif self.pool_type == 'tok':
            pooled, tokens = x[:, 0], x[:, 1:]
        else:
            pooled = tokens = x
        return pooled, tokens

    def forward(self, x: 'torch.Tensor'):
        x = self.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        x = torch.cat([_expand_token(self.class_embedding, x.shape[0]), x], dim=1)
        x = x + self.positional_embedding
        x = self.patch_dropout(x)
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        if self.attn_pool is not None:
            if self.attn_pool_contrastive is not None:
                x = self.ln_post(x)
                tokens = self.attn_pool(x)
                if self.attn_pool_type == 'parallel':
                    pooled = self.attn_pool_contrastive(x)
                else:
                    assert self.attn_pool_type == 'cascade'
                    pooled = self.attn_pool_contrastive(tokens)
            else:
                x = self.attn_pool(x)
                x = self.ln_post(x)
                pooled, tokens = self._global_pool(x)
        elif self.final_ln_after_pool:
            pooled, tokens = self._global_pool(x)
            pooled = self.ln_post(pooled)
        else:
            x = self.ln_post(x)
            pooled, tokens = self._global_pool(x)
        if self.proj is not None:
            pooled = pooled @ self.proj
        if self.output_tokens:
            return pooled, tokens
        return pooled


def _build_vision_tower(embed_dim: 'int', vision_cfg: 'CLIPVisionCfg', quick_gelu: 'bool'=False, cast_dtype: 'Optional[torch.dtype]'=None):
    if isinstance(vision_cfg, dict):
        vision_cfg = CLIPVisionCfg(**vision_cfg)
    act_layer = QuickGELU if quick_gelu else nn.GELU
    if vision_cfg.timm_model_name:
        visual = TimmModel(vision_cfg.timm_model_name, pretrained=vision_cfg.timm_model_pretrained, pool=vision_cfg.timm_pool, proj=vision_cfg.timm_proj, proj_bias=vision_cfg.timm_proj_bias, drop=vision_cfg.timm_drop, drop_path=vision_cfg.timm_drop_path, patch_drop=vision_cfg.patch_dropout if vision_cfg.patch_dropout > 0 else None, embed_dim=embed_dim, image_size=vision_cfg.image_size)
    elif isinstance(vision_cfg.layers, (tuple, list)):
        vision_heads = vision_cfg.width * 32 // vision_cfg.head_width
        visual = ModifiedResNet(layers=vision_cfg.layers, output_dim=embed_dim, heads=vision_heads, image_size=vision_cfg.image_size, width=vision_cfg.width)
    else:
        vision_heads = vision_cfg.width // vision_cfg.head_width
        norm_layer = LayerNormFp32 if cast_dtype in (torch.float16, torch.bfloat16) else LayerNorm
        if vision_cfg.norm_kwargs:
            norm_layer = partial(norm_layer, **vision_cfg.norm_kwargs)
        if vision_cfg.act_kwargs is not None:
            act_layer = partial(act_layer, **vision_cfg.act_kwargs)
        visual = VisionTransformer(image_size=vision_cfg.image_size, patch_size=vision_cfg.patch_size, width=vision_cfg.width, layers=vision_cfg.layers, heads=vision_heads, mlp_ratio=vision_cfg.mlp_ratio, ls_init_value=vision_cfg.ls_init_value, patch_dropout=vision_cfg.patch_dropout, attentional_pool=vision_cfg.attentional_pool, attn_pooler_queries=vision_cfg.attn_pooler_queries, attn_pooler_heads=vision_cfg.attn_pooler_heads, pos_embed_type=vision_cfg.pos_embed_type, no_ln_pre=vision_cfg.no_ln_pre, final_ln_after_pool=vision_cfg.final_ln_after_pool, pool_type=vision_cfg.pool_type, output_tokens=True, output_dim=embed_dim, act_layer=act_layer, norm_layer=norm_layer)
    return visual


def prepare_inputs_for_generation(input_ids, image_inputs, past=None, **kwargs):
    if past:
        input_ids = input_ids[:, -1].unsqueeze(-1)
    attention_mask = kwargs.get('attention_mask', None)
    position_ids = kwargs.get('position_ids', None)
    if attention_mask is not None and position_ids is None:
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
    else:
        position_ids = None
    return {'text': input_ids, 'images': image_inputs, 'past_key_values': past, 'position_ids': position_ids, 'attention_mask': attention_mask}


class CoCa(nn.Module):

    def __init__(self, embed_dim, multimodal_cfg: 'MultimodalCfg', text_cfg: 'CLIPTextCfg', vision_cfg: 'CLIPVisionCfg', quick_gelu: 'bool'=False, init_logit_scale: 'float'=np.log(1 / 0.07), init_logit_bias: 'Optional[float]'=None, cast_dtype: 'Optional[torch.dtype]'=None, pad_id: 'int'=0):
        super().__init__()
        multimodal_cfg = MultimodalCfg(**multimodal_cfg) if isinstance(multimodal_cfg, dict) else multimodal_cfg
        text_cfg = CLIPTextCfg(**text_cfg) if isinstance(text_cfg, dict) else text_cfg
        vision_cfg = CLIPVisionCfg(**vision_cfg) if isinstance(vision_cfg, dict) else vision_cfg
        self.text = _build_text_tower(embed_dim=embed_dim, text_cfg=text_cfg, quick_gelu=quick_gelu, cast_dtype=cast_dtype)
        vocab_size = text_cfg.vocab_size if hasattr(text_cfg, 'hf_model_name') and text_cfg.hf_model_name is not None else text_cfg.vocab_size
        self.visual = _build_vision_tower(embed_dim=embed_dim, vision_cfg=vision_cfg, quick_gelu=quick_gelu, cast_dtype=cast_dtype)
        self.text_decoder = _build_text_decoder_tower(vocab_size, multimodal_cfg=multimodal_cfg, quick_gelu=quick_gelu, cast_dtype=cast_dtype)
        self.logit_scale = nn.Parameter(torch.ones([]) * init_logit_scale)
        if init_logit_bias is not None:
            self.logit_bias = nn.Parameter(torch.ones([]) * init_logit_bias)
        else:
            self.logit_bias = None
        self.pad_id = pad_id
        self.context_length = multimodal_cfg.context_length

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable: 'bool'=True):
        self.visual.set_grad_checkpointing(enable)
        self.text.set_grad_checkpointing(enable)
        self.text_decoder.set_grad_checkpointing(enable)

    def _encode_image(self, images, normalize: 'bool'=True):
        image_latent, tokens_embs = self.visual(images)
        image_latent = F.normalize(image_latent, dim=-1) if normalize else image_latent
        return image_latent, tokens_embs

    def _encode_text(self, text, normalize: 'bool'=True):
        text_latent, token_emb = self.text(text)
        text_latent = F.normalize(text_latent, dim=-1) if normalize else text_latent
        return text_latent, token_emb

    def encode_image(self, images, normalize: 'bool'=True):
        image_latent, _ = self._encode_image(images, normalize=normalize)
        return image_latent

    def encode_text(self, text, normalize: 'bool'=True):
        text_latent, _ = self._encode_text(text, normalize=normalize)
        return text_latent

    def forward(self, image, text: 'Optional[torch.Tensor]'=None, image_latent: 'Optional[torch.Tensor]'=None, image_embs: 'Optional[torch.Tensor]'=None):
        if image_latent is None or image_embs is None:
            image_latent, image_embs = self._encode_image(image)
        if text is None:
            return {'image_features': image_latent, 'image_embs': image_embs}
        text_latent, token_embs = self._encode_text(text)
        labels = text[:, -token_embs.shape[1]:]
        logits = self.text_decoder(image_embs, token_embs)
        out_dict = {'image_features': image_latent, 'text_features': text_latent, 'logits': logits, 'labels': labels, 'logit_scale': self.logit_scale.exp()}
        if self.logit_bias is not None:
            out_dict['logit_bias'] = self.logit_bias
        return out_dict

    def generate(self, image, text=None, seq_len=30, max_seq_len=77, temperature=1.0, generation_type='beam_search', top_p=0.1, top_k=1, pad_token_id=None, eos_token_id=None, sot_token_id=None, num_beams=6, num_beam_groups=3, min_seq_len=5, stopping_criteria=None, repetition_penalty=1.0, fixed_output_length=False):
        assert _has_transformers, 'Please install transformers for generate functionality. `pip install transformers`.'
        assert seq_len > min_seq_len, 'seq_len must be larger than min_seq_len'
        with torch.no_grad():
            sot_token_id = 49406 if sot_token_id is None else sot_token_id
            eos_token_id = 49407 if eos_token_id is None else eos_token_id
            pad_token_id = self.pad_id if pad_token_id is None else pad_token_id
            logit_processor = LogitsProcessorList([MinLengthLogitsProcessor(min_seq_len, eos_token_id), RepetitionPenaltyLogitsProcessor(repetition_penalty)])
            if stopping_criteria is None:
                stopping_criteria = [MaxLengthCriteria(max_length=seq_len)]
            stopping_criteria = StoppingCriteriaList(stopping_criteria)
            device = image.device
            if generation_type == 'beam_search':
                output = self._generate_beamsearch(image_inputs=image, pad_token_id=pad_token_id, eos_token_id=eos_token_id, sot_token_id=sot_token_id, num_beams=num_beams, num_beam_groups=num_beam_groups, min_seq_len=min_seq_len, stopping_criteria=stopping_criteria, logit_processor=logit_processor)
                if fixed_output_length and output.shape[1] < seq_len:
                    return torch.cat((output, torch.ones(output.shape[0], seq_len - output.shape[1], device=device, dtype=output.dtype) * self.pad_id), dim=1)
                return output
            elif generation_type == 'top_p':
                logit_warper = GENERATION_TYPES[generation_type](top_p)
            elif generation_type == 'top_k':
                logit_warper = GENERATION_TYPES[generation_type](top_k)
            else:
                raise ValueError(f"generation_type has to be one of {'| ' + ' | '.join(list(GENERATION_TYPES.keys())) + ' |'}.")
            image_latent, image_embs = self._encode_image(image)
            if text is None:
                text = torch.ones((image.shape[0], 1), device=device, dtype=torch.long) * sot_token_id
            was_training = self.training
            num_dims = len(text.shape)
            if num_dims == 1:
                text = text[None, :]
            cur_len = text.shape[1]
            self.eval()
            out = text
            while True:
                x = out[:, -max_seq_len:]
                cur_len = x.shape[1]
                logits = self(image, x, image_latent=image_latent, image_embs=image_embs)['logits'][:, -1]
                mask = (out[:, -1] == eos_token_id) | (out[:, -1] == pad_token_id)
                sample = torch.ones((out.shape[0], 1), device=device, dtype=torch.long) * pad_token_id
                if mask.all():
                    if not fixed_output_length:
                        break
                else:
                    logits = logits[~mask, :]
                    filtered_logits = logit_processor(x[~mask, :], logits)
                    filtered_logits = logit_warper(x[~mask, :], filtered_logits)
                    probs = F.softmax(filtered_logits / temperature, dim=-1)
                    if cur_len + 1 == seq_len:
                        sample[~mask, :] = torch.ones((sum(~mask), 1), device=device, dtype=torch.long) * eos_token_id
                    else:
                        sample[~mask, :] = torch.multinomial(probs, 1)
                out = torch.cat((out, sample), dim=-1)
                cur_len += 1
                if stopping_criteria(out, None):
                    break
            if num_dims == 1:
                out = out.squeeze(0)
            self.train(was_training)
            return out

    def _generate_beamsearch(self, image_inputs, pad_token_id=None, eos_token_id=None, sot_token_id=None, num_beams=6, num_beam_groups=3, min_seq_len=5, stopping_criteria=None, logit_processor=None, logit_warper=None):
        device = image_inputs.device
        batch_size = image_inputs.shape[0]
        image_inputs = torch.repeat_interleave(image_inputs, num_beams, dim=0)
        image_latent, image_embs = self._encode_image(image_inputs)
        input_ids = torch.ones((batch_size * num_beams, 1), device=device, dtype=torch.long)
        input_ids = input_ids * sot_token_id
        beam_scorer = BeamSearchScorer(batch_size=batch_size, num_beams=num_beams, device=device, num_beam_groups=num_beam_groups)
        logits_processor = LogitsProcessorList([MinLengthLogitsProcessor(min_seq_len, eos_token_id=eos_token_id)]) if logit_processor is None else logit_processor
        num_beams = beam_scorer.num_beams
        num_beam_groups = beam_scorer.num_beam_groups
        num_sub_beams = num_beams // num_beam_groups
        batch_size = len(beam_scorer._beam_hyps) // num_beam_groups
        batch_beam_size, cur_len = input_ids.shape
        beam_indices = None
        if num_beams * batch_size != batch_beam_size:
            raise ValueError(f'Batch dimension of `input_ids` should be {num_beams * batch_size}, but is {batch_beam_size}.')
        beam_scores = torch.full((batch_size, num_beams), -1000000000.0, dtype=torch.float, device=device)
        beam_scores[:, ::num_sub_beams] = 0
        beam_scores = beam_scores.view((batch_size * num_beams,))
        while True:
            current_tokens = torch.zeros(batch_size * num_beams, dtype=input_ids.dtype, device=device)
            reordering_indices = torch.zeros(batch_size * num_beams, dtype=torch.long, device=device)
            model_inputs = prepare_inputs_for_generation(input_ids=input_ids, image_inputs=image_inputs)
            outputs = self(model_inputs['images'], model_inputs['text'], image_latent=image_latent, image_embs=image_embs)
            for beam_group_idx in range(num_beam_groups):
                group_start_idx = beam_group_idx * num_sub_beams
                group_end_idx = min(group_start_idx + num_sub_beams, num_beams)
                group_size = group_end_idx - group_start_idx
                batch_group_indices = []
                for batch_idx in range(batch_size):
                    batch_group_indices.extend([(batch_idx * num_beams + idx) for idx in range(group_start_idx, group_end_idx)])
                group_input_ids = input_ids[batch_group_indices]
                next_token_logits = outputs['logits'][batch_group_indices, -1, :]
                vocab_size = next_token_logits.shape[-1]
                next_token_scores_processed = logits_processor(group_input_ids, next_token_logits, current_tokens=current_tokens, beam_group_idx=beam_group_idx)
                next_token_scores = next_token_scores_processed + beam_scores[batch_group_indices].unsqueeze(-1)
                next_token_scores = next_token_scores.expand_as(next_token_scores_processed)
                next_token_scores = next_token_scores.view(batch_size, group_size * vocab_size)
                next_token_scores, next_tokens = torch.topk(next_token_scores, 2 * group_size, dim=1, largest=True, sorted=True)
                next_indices = torch.div(next_tokens, vocab_size, rounding_mode='floor')
                next_tokens = next_tokens % vocab_size
                process_beam_indices = sum(beam_indices, ()) if beam_indices is not None else None
                beam_outputs = beam_scorer.process(group_input_ids, next_token_scores, next_tokens, next_indices, pad_token_id=pad_token_id, eos_token_id=eos_token_id, beam_indices=process_beam_indices, group_index=beam_group_idx)
                beam_scores[batch_group_indices] = beam_outputs['next_beam_scores']
                beam_next_tokens = beam_outputs['next_beam_tokens']
                beam_idx = beam_outputs['next_beam_indices']
                input_ids[batch_group_indices] = group_input_ids[beam_idx]
                group_input_ids = torch.cat([group_input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)
                current_tokens[batch_group_indices] = group_input_ids[:, -1]
                reordering_indices[batch_group_indices] = num_beams * torch.div(beam_idx, group_size, rounding_mode='floor') + group_start_idx + beam_idx % group_size
            input_ids = torch.cat([input_ids, current_tokens.unsqueeze(-1)], dim=-1)
            cur_len = cur_len + 1
            if beam_scorer.is_done or stopping_criteria(input_ids, None):
                break
        final_beam_indices = sum(beam_indices, ()) if beam_indices is not None else None
        sequence_outputs = beam_scorer.finalize(input_ids, beam_scores, next_tokens, next_indices, pad_token_id=pad_token_id, eos_token_id=eos_token_id, max_length=stopping_criteria.max_length, beam_indices=final_beam_indices)
        return sequence_outputs['sequences']


class MeanPooler(nn.Module):
    """Mean pooling"""

    def forward(self, x: 'BaseModelOutput', attention_mask: 'TensorType'):
        masked_output = x.last_hidden_state * attention_mask.unsqueeze(-1)
        return masked_output.sum(dim=1) / attention_mask.sum(-1, keepdim=True)


class MaxPooler(nn.Module):
    """Max pooling"""

    def forward(self, x: 'BaseModelOutput', attention_mask: 'TensorType'):
        masked_output = x.last_hidden_state.masked_fill(attention_mask.unsqueeze(-1), -torch.inf)
        return masked_output.max(1).values


class ClsLastHiddenStatePooler(nn.Module):
    """CLS token pooling
    NOTE: this is equivalent to ClsPooler above with use_pooler_output=False
    """

    def __init__(self):
        super().__init__()
        self.cls_token_position = 0

    def forward(self, x: 'BaseModelOutput', attention_mask: 'TensorType'):
        return x.last_hidden_state[:, self.cls_token_position, :]


class AsymmetricLoss(nn.Module):

    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-08, disable_torch_grad_focal_loss=True):
        super(AsymmetricLoss, self).__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            loss *= one_sided_w
        return -loss.mean()


def gather_features(image_features, text_features, local_loss=False, gather_with_grad=False, rank=0, world_size=1, use_horovod=False):
    assert has_distributed, 'torch.distributed did not import correctly, please use a PyTorch version with support.'
    if use_horovod:
        assert hvd is not None, 'Please install horovod'
        if gather_with_grad:
            all_image_features = hvd.allgather(image_features)
            all_text_features = hvd.allgather(text_features)
        else:
            with torch.no_grad():
                all_image_features = hvd.allgather(image_features)
                all_text_features = hvd.allgather(text_features)
            if not local_loss:
                gathered_image_features = list(all_image_features.chunk(world_size, dim=0))
                gathered_text_features = list(all_text_features.chunk(world_size, dim=0))
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
                all_image_features = torch.cat(gathered_image_features, dim=0)
                all_text_features = torch.cat(gathered_text_features, dim=0)
    elif gather_with_grad:
        all_image_features = torch.cat(torch.distributed.nn.all_gather(image_features), dim=0)
        all_text_features = torch.cat(torch.distributed.nn.all_gather(text_features), dim=0)
    else:
        gathered_image_features = [torch.zeros_like(image_features) for _ in range(world_size)]
        gathered_text_features = [torch.zeros_like(text_features) for _ in range(world_size)]
        dist.all_gather(gathered_image_features, image_features)
        dist.all_gather(gathered_text_features, text_features)
        if not local_loss:
            gathered_image_features[rank] = image_features
            gathered_text_features[rank] = text_features
        all_image_features = torch.cat(gathered_image_features, dim=0)
        all_text_features = torch.cat(gathered_text_features, dim=0)
    return all_image_features, all_text_features


def gather_only_features(features, local_loss=False, gather_with_grad=False, rank=0, world_size=1, use_horovod=False):
    assert has_distributed, 'torch.distributed did not import correctly, please use a PyTorch version with support.'
    if use_horovod:
        assert hvd is not None, 'Please install horovod'
        if gather_with_grad:
            all_features = hvd.allgather(features)
        else:
            with torch.no_grad():
                all_features = hvd.allgather(features)
            if not local_loss:
                gathered_features = list(all_features.chunk(world_size, dim=0))
                gathered_features[rank] = features
                all_features = torch.cat(gathered_features, dim=0)
    elif gather_with_grad:
        all_features = torch.cat(torch.distributed.nn.all_gather(features), dim=0)
    else:
        gathered_features = [torch.zeros_like(features) for _ in range(world_size)]
        dist.all_gather(gathered_features, features)
        if not local_loss:
            gathered_features[rank] = features
        all_features = torch.cat(gathered_features, dim=0)
    return all_features


class proxy_anchor_loss(nn.Module):
    """
    ref: https://arxiv.org/abs/2003.13911
    official pytorch codes: https://github.com/tjddus9597/Proxy-Anchor-CVPR2020
    """

    def __init__(self, scale=16, margin=0.1):
        super(proxy_anchor_loss, self).__init__()
        self.alpha = scale
        self.delta = margin

    def forward(self, dist_mat, target):
        pos_target = target.float()
        neg_target = 1.0 - pos_target
        pos_mat = torch.exp(-self.alpha * (dist_mat - self.delta)) * pos_target
        neg_mat = torch.exp(self.alpha * (dist_mat + self.delta)) * neg_target
        pos_term = 1.0 / torch.unique(target).shape[0] * torch.sum(torch.log(1.0 + torch.sum(pos_mat, axis=0)))
        neg_term = 1.0 / dist_mat.shape[0] * torch.sum(torch.log(1.0 + torch.sum(neg_mat, axis=0)))
        loss = pos_term + neg_term
        return loss


class ClipLoss(nn.Module):

    def __init__(self, local_loss=False, gather_with_grad=False, cache_labels=False, rank=0, world_size=1, use_horovod=False, use_noun_labels=None, tag_mode='mixed', use_longcap=False, use_multipos_loss=False, use_ot_loss=False, use_finegrained_loss=False, use_declip_loss=False, gamma=32, margin=0.25, sgl_delta=0.1, sgl_norm='minmax', clip_lossweight=1.0, sgl_lossweight=1.0):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod
        self.use_noun_labels = use_noun_labels
        self.prev_num_logits = 0
        self.labels = {}
        self.prev_num_logits_multicap = 0
        self.labels_multicap = {}
        self.prev_num_logits_ot = 0
        self.labels_ot = {}
        if self.use_noun_labels:
            self.tagging_loss_function = AsymmetricLoss(gamma_neg=7, gamma_pos=0, clip=0.05)
        self.tag_mode = tag_mode
        self.max_iter = 100
        self.eps = 0.1
        self.use_multipos_loss = use_multipos_loss
        if self.use_multipos_loss:
            self.mpl = proxy_anchor_loss()
        self.use_ot_loss = use_ot_loss
        self.use_finegrained_loss = use_finegrained_loss
        self.use_declip_loss = use_declip_loss
        self.gamma = gamma
        self.margin = margin
        self.sgl_norm = sgl_norm
        self.sgl_delta = sgl_delta
        self.clip_lossweight = clip_lossweight
        self.sgl_lossweight = sgl_lossweight

    def get_ground_truth(self, device, num_logits) ->torch.Tensor:
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.world_size > 1 and self.local_loss:
                labels = labels + num_logits * self.rank
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]
        return labels

    def get_ground_truth_ot(self, device, num_logits, M, N) ->torch.Tensor:
        if device not in self.labels_ot:
            labels = torch.eye(num_logits, device=device, dtype=torch.long).repeat(M, N)
            if self.cache_labels:
                self.labels_ot[device] = labels
                self.prev_num_logits_ot = num_logits
        else:
            labels = self.labels_ot[device]
        return labels

    def get_ground_truth_multicap(self, device, num_logits, M, N) ->torch.Tensor:
        if device not in self.labels_multicap:
            labels = torch.eye(num_logits, device=device, dtype=torch.long).repeat(M, N)
            if self.cache_labels:
                self.labels_multicap[device] = labels
                self.prev_num_logits_multicap = num_logits
        else:
            labels = self.labels_multicap[device]
        return labels

    def get_logits(self, image_features, text_features, logit_scale):
        if self.world_size > 1:
            all_image_features, all_text_features = gather_features(image_features, text_features, self.local_loss, self.gather_with_grad, self.rank, self.world_size, self.use_horovod)
            if len(all_text_features.shape) == 3:
                bs, N_t, c = all_text_features.shape
                all_text_features = all_text_features.permute(1, 0, 2).reshape(N_t * bs, c)
            if len(all_image_features.shape) == 3:
                bs, N_i, c = all_image_features.shape
                all_image_features = all_image_features.permute(1, 0, 2).reshape(N_i * bs, c)
            if self.local_loss:
                logits_per_image = logit_scale * image_features @ all_text_features.T
                logits_per_text = logit_scale * text_features @ all_image_features.T
            else:
                logits_per_image = logit_scale * all_image_features @ all_text_features.T
                logits_per_text = logits_per_image.T
            text_features = all_text_features
            image_features = all_image_features
        else:
            if len(text_features.shape) == 3:
                bs, N_t, c = text_features.shape
                text_features = text_features.permute(1, 0, 2).reshape(N_t * bs, c)
            if len(image_features.shape) == 3:
                bs, N_i, c = image_features.shape
                image_features = image_features.permute(1, 0, 2).reshape(N_i * bs, c)
            logits_per_image = logit_scale * image_features @ text_features.T
            logits_per_text = logit_scale * text_features @ image_features.T
        return logits_per_image, logits_per_text, text_features, image_features

    def Sinkhorn(self, K, u, v):
        r = torch.ones_like(u)
        c = torch.ones_like(v)
        thresh = 0.001
        for i in range(self.max_iter):
            r0 = r
            r = u / torch.matmul(K, c.unsqueeze(-1)).squeeze(-1)
            c = v / torch.matmul(K.permute(0, 2, 1).contiguous(), r.unsqueeze(-1)).squeeze(-1)
            err = (r - r0).abs().mean()
            if err.item() < thresh:
                break
        T = torch.matmul(r.unsqueeze(-1), c.unsqueeze(-2)) * K
        return T

    def pairwise_contrastive_loss_instance(self, a, b, device, logit_scale=1.0):
        batch_size, seq_len, _ = a.shape
        labels = torch.eye(seq_len, device=device, dtype=torch.float).repeat(batch_size, 1)
        logits = torch.einsum('bmd,bnd->bmn', a, b) * logit_scale
        loss = F.cross_entropy(logits.view(-1, seq_len), labels)
        return loss

    def pairwise_contrastive_loss(self, a, b, device, logit_scale=1.0):
        batch_size, seq_len, c = a.shape
        labels = torch.eye(seq_len * batch_size, device=device, dtype=torch.float)
        text_features = a.contiguous().view(seq_len * batch_size, c)
        image_features = b.contiguous().view(seq_len * batch_size, c)
        logits_per_image = logit_scale * image_features @ text_features.T
        logits_per_text = logit_scale * text_features @ image_features.T
        loss = 1 / 2 * (F.cross_entropy(logits_per_text, labels) + F.cross_entropy(logits_per_image, labels))
        return loss

    def pairwise_circleloss(self, dist_mat, is_pos, margin=0.25, gamma=64):
        N = dist_mat.size(0)
        is_neg = 1.0 - is_pos
        s_p = dist_mat * is_pos
        s_n = dist_mat * is_neg
        alpha_p = torch.clamp_min(-s_p.detach() + 1 + margin, min=0.0)
        alpha_n = torch.clamp_min(s_n.detach() + margin, min=0.0)
        delta_p = 1 - margin
        delta_n = margin
        logit_p = -gamma * alpha_p * (s_p - delta_p) + -99999999.0 * (1 - is_pos)
        logit_n = gamma * alpha_n * (s_n - delta_n) + -99999999.0 * (1 - is_neg)
        loss = F.softplus(torch.logsumexp(logit_p, dim=1) + torch.logsumexp(logit_n, dim=1)).mean()
        return loss

    def forward(self, image_features, text_features, logit_scale, logit_scale_local, local_image_features=None, output_dict=False, noun_labels=None, classess_embs=None):
        device = image_features.device
        out_losses = {}
        if text_features is not None and len(text_features.shape) == 3:
            bs, number_text, c = text_features.shape
            if image_features.shape[0] != text_features.shape[0]:
                image_features = image_features.view(bs, -1, c).contiguous()
                number_img = image_features.shape[1]
            else:
                number_img = 1
                image_features = image_features.unsqueeze(1)
            if local_image_features is not None:
                local_image_features = local_image_features.contiguous().view(text_features.shape[0], number_img, -1, c)
            global_losses = []
            all_text_features_list = []
            all_img_features_list = []
            for j in range(number_img):
                if self.world_size > 1:
                    all_img_features_list.append(gather_only_features(image_features[:, j, :], False, self.gather_with_grad, self.rank, self.world_size, self.use_horovod))
                else:
                    all_img_features_list.append(image_features[:, j, :])
            for i in range(number_text):
                if self.world_size > 1:
                    all_text_features_list.append(gather_only_features(text_features[:, i, :], False, self.gather_with_grad, self.rank, self.world_size, self.use_horovod))
                else:
                    all_text_features_list.append(text_features[:, i, :])
            logits_per_image = logit_scale * all_img_features_list[0] @ all_text_features_list[0].T
            logits_per_text = logit_scale * all_text_features_list[0] @ all_img_features_list[0].T
            labels = self.get_ground_truth(device, logits_per_image.shape[0])
            out_losses['CLIP_loss'] = self.clip_lossweight * (F.cross_entropy(logits_per_image, labels) + F.cross_entropy(logits_per_text, labels)) / (2 * number_text)
            if self.use_declip_loss:
                for j in range(number_img):
                    for i in range(number_text):
                        if j == 0 and i == 0:
                            continue
                        logits_per_image = logit_scale * all_img_features_list[j] @ all_text_features_list[i].T
                        logits_per_text = logit_scale * all_text_features_list[i] @ all_img_features_list[j].T
                        global_losses.append((F.cross_entropy(logits_per_image, labels) + F.cross_entropy(logits_per_text, labels)) / (2 * number_text))
                out_losses['(De)CLIP_loss'] = self.clip_lossweight * sum(global_losses)
            all_text_features_list = torch.stack(all_text_features_list, dim=0).view(-1, c)
            if self.use_multipos_loss:
                labels_mul = self.get_ground_truth_multicap(device, all_img_features_list[0].shape[0], number_text, 1)
                all_img_features_list = all_img_features_list[0]
                logits_per_image = all_img_features_list @ all_text_features_list.T
                logits_per_text = logits_per_image.T
                out_losses['I2mT_loss'] = self.pairwise_circleloss(logits_per_image, labels_mul.T, margin=self.margin, gamma=self.gamma) / 10.0
            if self.use_finegrained_loss:
                if self.world_size > 1:
                    local_image_features = gather_only_features(local_image_features, False, self.gather_with_grad, self.rank, self.world_size, self.use_horovod)[:, 0:1, :, :]
                bs, num_i, M, c = local_image_features.shape
                b = bs * num_i
                local_image_features = local_image_features.permute(1, 0, 2, 3).contiguous().view(b, M, c)
                all_text_features = all_text_features_list.contiguous().view(bs, number_text, c)
                local_image_features = F.normalize(local_image_features, p=2, dim=-1)
                all_text_features = F.normalize(all_text_features, p=2, dim=-1)
                similarity = torch.einsum('btd,bpd->btp', all_text_features, local_image_features)
                min_val = torch.min(similarity, dim=-1, keepdim=True).values
                max_val = torch.max(similarity, dim=-1, keepdim=True).values
                epsilon = 1e-10
                if self.sgl_norm == 'minmax':
                    normalized_similarity = (similarity - min_val) / (max_val - min_val + epsilon)
                else:
                    normalized_similarity = F.softmax(similarity, dim=-1)
                if self.sgl_delta < 0:
                    similarity_threshold = 1.0 / local_image_features.shape[1]
                else:
                    similarity_threshold = self.sgl_delta
                normalized_similarity = torch.where(normalized_similarity < similarity_threshold, torch.tensor(0.0, device=normalized_similarity.device), normalized_similarity)
                sum_similarity = torch.sum(normalized_similarity, dim=-1, keepdim=True)
                v_align_weights = normalized_similarity / sum_similarity
                l_grouped_v_patch_embed = torch.einsum('btp,bpd->btd', v_align_weights, local_image_features)
                l_grouped_v_patch_embed = F.normalize(l_grouped_v_patch_embed, p=2, dim=-1)
                out_losses['finegrained_loss'] = self.sgl_lossweight * self.pairwise_contrastive_loss(all_text_features, l_grouped_v_patch_embed, device, logit_scale_local)
            if self.use_ot_loss:
                if self.world_size > 1:
                    local_image_features = gather_only_features(local_image_features, False, self.gather_with_grad, self.rank, self.world_size, self.use_horovod)[:, 0:1, :, :]
                bs, num_i, M, c = local_image_features.shape
                b = bs * num_i
                local_image_features = local_image_features.permute(1, 0, 2, 3).contiguous().view(b, M, c)
                all_text_features = all_text_features.contiguous().view(bs, number_text, c)
                sim = torch.einsum('mbd,ncd->mnbc', local_image_features.permute(1, 0, 2), all_text_features.permute(1, 0, 2)).contiguous()
                sim = sim.view(M, number_text, b * bs)
                sim = sim.permute(2, 0, 1)
                wdist = 1.0 - sim
                xx = torch.zeros(b * bs, M, dtype=sim.dtype, device=sim.device).fill_(1.0 / M)
                yy = torch.zeros(b * bs, number_text, dtype=sim.dtype, device=sim.device).fill_(1.0 / number_text)
                with torch.no_grad():
                    KK = torch.exp(-wdist / self.eps)
                    T = self.Sinkhorn(KK, xx, yy)
                try:
                    torch.isnan(T).any()
                except None:
                    None
                sim_op = torch.sum(T * sim, dim=(1, 2))
                sim_op = logit_scale * sim_op.contiguous().view(b, bs)
                out_losses['ot_local_loss'] = F.cross_entropy(sim_op, self.get_ground_truth_ot(device, b, 1, 1).float())
            return out_losses
        else:
            logits_per_image, logits_per_text, _, _ = self.get_logits(image_features, text_features, logit_scale)
            labels = self.get_ground_truth(device, logits_per_image.shape[0])
            total_loss = (F.cross_entropy(logits_per_image, labels) + F.cross_entropy(logits_per_text, labels)) / 2
        if self.use_noun_labels:
            mixed_tag_loss = self.tagging_loss_function(classess_embs, noun_labels) / 1000.0
            return {'contrastive_loss': total_loss, f'{self.tag_mode}_tag_loss': mixed_tag_loss}
        else:
            return {'contrastive_loss': total_loss} if output_dict else total_loss


class CoCaLoss(ClipLoss):

    def __init__(self, caption_loss_weight, clip_loss_weight, pad_id=0, local_loss=False, gather_with_grad=False, cache_labels=False, rank=0, world_size=1, use_horovod=False):
        super().__init__(local_loss=local_loss, gather_with_grad=gather_with_grad, cache_labels=cache_labels, rank=rank, world_size=world_size, use_horovod=use_horovod)
        self.clip_loss_weight = clip_loss_weight
        self.caption_loss_weight = caption_loss_weight
        self.caption_loss = nn.CrossEntropyLoss(ignore_index=pad_id)

    def forward(self, image_features, text_features, logits, labels, logit_scale, output_dict=False):
        clip_loss = torch.tensor(0)
        if self.clip_loss_weight:
            clip_loss = super().forward(image_features, text_features, logit_scale)
            clip_loss = self.clip_loss_weight * clip_loss
        caption_loss = self.caption_loss(logits.permute(0, 2, 1), labels)
        caption_loss = caption_loss * self.caption_loss_weight
        if output_dict:
            return {'contrastive_loss': clip_loss, 'caption_loss': caption_loss}
        return clip_loss, caption_loss


class DistillClipLoss(ClipLoss):

    def dist_loss(self, teacher_logits, student_logits):
        return -(teacher_logits.softmax(dim=1) * student_logits.log_softmax(dim=1)).sum(dim=1).mean(dim=0)

    def forward(self, image_features, text_features, logit_scale, dist_image_features, dist_text_features, dist_logit_scale, output_dict=False):
        logits_per_image, logits_per_text = self.get_logits(image_features, text_features, logit_scale)
        dist_logits_per_image, dist_logits_per_text = self.get_logits(dist_image_features, dist_text_features, dist_logit_scale)
        labels = self.get_ground_truth(image_features.device, logits_per_image.shape[0])
        contrastive_loss = (F.cross_entropy(logits_per_image, labels) + F.cross_entropy(logits_per_text, labels)) / 2
        distill_loss = (self.dist_loss(dist_logits_per_image, logits_per_image) + self.dist_loss(dist_logits_per_text, logits_per_text)) / 2
        if output_dict:
            return {'contrastive_loss': contrastive_loss, 'distill_loss': distill_loss}
        return contrastive_loss, distill_loss


def neighbour_exchange_bidir(left_rank, right_rank, tensor_to_left, tensor_to_right, group=None):
    tensor_from_left = torch.zeros_like(tensor_to_right)
    tensor_from_right = torch.zeros_like(tensor_to_left)
    send_op_left = torch.distributed.P2POp(torch.distributed.isend, tensor_to_left, left_rank, group=group)
    send_op_right = torch.distributed.P2POp(torch.distributed.isend, tensor_to_right, right_rank, group=group)
    recv_op_left = torch.distributed.P2POp(torch.distributed.irecv, tensor_from_left, left_rank, group=group)
    recv_op_right = torch.distributed.P2POp(torch.distributed.irecv, tensor_from_right, right_rank, group=group)
    reqs = torch.distributed.batch_isend_irecv([send_op_right, send_op_left, recv_op_right, recv_op_left])
    for req in reqs:
        req.wait()
    return tensor_from_right, tensor_from_left


class NeighbourExchangeBidir(torch.autograd.Function):

    @staticmethod
    def forward(ctx, left_rank, right_rank, group, tensor_to_left, tensor_to_right):
        ctx.group = group
        ctx.left_rank = left_rank
        ctx.right_rank = right_rank
        return neighbour_exchange_bidir(left_rank, right_rank, tensor_to_left, tensor_to_right, group=group)

    @staticmethod
    def backward(ctx, *grad_outputs):
        return (None, None, None) + NeighbourExchangeBidir.apply(ctx.right_rank, ctx.left_rank, ctx.group, *grad_outputs)


def neighbour_exchange_bidir_with_grad(left_rank, right_rank, tensor_to_left, tensor_to_right, group=None):
    return NeighbourExchangeBidir.apply(left_rank, right_rank, group, tensor_to_left, tensor_to_right)


def neighbour_exchange(from_rank, to_rank, tensor, group=None):
    tensor_recv = torch.zeros_like(tensor)
    send_op = torch.distributed.P2POp(torch.distributed.isend, tensor, to_rank, group=group)
    recv_op = torch.distributed.P2POp(torch.distributed.irecv, tensor_recv, from_rank, group=group)
    reqs = torch.distributed.batch_isend_irecv([send_op, recv_op])
    for req in reqs:
        req.wait()
    return tensor_recv


class NeighbourExchange(torch.autograd.Function):

    @staticmethod
    def forward(ctx, from_rank, to_rank, group, tensor):
        ctx.group = group
        ctx.from_rank = from_rank
        ctx.to_rank = to_rank
        return neighbour_exchange(from_rank, to_rank, tensor, group=group)

    @staticmethod
    def backward(ctx, grad_output):
        return (None, None, None) + (NeighbourExchange.apply(ctx.to_rank, ctx.from_rank, ctx.group, grad_output),)


def neighbour_exchange_with_grad(from_rank, to_rank, tensor, group=None):
    return NeighbourExchange.apply(from_rank, to_rank, group, tensor)


class SigLipLoss(nn.Module):
    """ Sigmoid Loss for Language Image Pre-Training (SigLIP) - https://arxiv.org/abs/2303.15343

    @article{zhai2023sigmoid,
      title={Sigmoid loss for language image pre-training},
      author={Zhai, Xiaohua and Mustafa, Basil and Kolesnikov, Alexander and Beyer, Lucas},
      journal={arXiv preprint arXiv:2303.15343},
      year={2023}
    }
    """

    def __init__(self, cache_labels=False, rank=0, world_size=1, bidir=True, use_horovod=False):
        super().__init__()
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        assert not use_horovod
        self.use_horovod = use_horovod
        self.bidir = bidir
        self.prev_num_logits = 0
        self.labels = {}

    def get_ground_truth(self, device, dtype, num_logits, negative_only=False) ->torch.Tensor:
        labels = -torch.ones((num_logits, num_logits), device=device, dtype=dtype)
        if not negative_only:
            labels = 2 * torch.eye(num_logits, device=device, dtype=dtype) + labels
        return labels

    def get_logits(self, image_features, text_features, logit_scale, logit_bias=None):
        logits = logit_scale * image_features @ text_features.T
        if logit_bias is not None:
            logits += logit_bias
        return logits

    def _loss(self, image_features, text_features, logit_scale, logit_bias=None, negative_only=False):
        logits = self.get_logits(image_features, text_features, logit_scale, logit_bias)
        labels = self.get_ground_truth(image_features.device, image_features.dtype, image_features.shape[0], negative_only=negative_only)
        loss = -F.logsigmoid(labels * logits).sum() / image_features.shape[0]
        return loss

    def forward(self, image_features, text_features, logit_scale, logit_bias, output_dict=False):
        loss = self._loss(image_features, text_features, logit_scale, logit_bias)
        if self.world_size > 1:
            right_rank = (self.rank + 1) % self.world_size
            left_rank = (self.rank - 1 + self.world_size) % self.world_size
            if self.bidir:
                text_features_to_right = text_features_to_left = text_features
                num_bidir, remainder = divmod(self.world_size - 1, 2)
                for i in range(num_bidir):
                    text_features_recv = neighbour_exchange_bidir_with_grad(left_rank, right_rank, text_features_to_left, text_features_to_right)
                    for f in text_features_recv:
                        loss += self._loss(image_features, f, logit_scale, logit_bias, negative_only=True)
                    text_features_to_left, text_features_to_right = text_features_recv
                if remainder:
                    text_features_recv = neighbour_exchange_with_grad(left_rank, right_rank, text_features_to_right)
                    loss += self._loss(image_features, text_features_recv, logit_scale, logit_bias, negative_only=True)
            else:
                text_features_to_right = text_features
                for i in range(self.world_size - 1):
                    text_features_from_left = neighbour_exchange_with_grad(left_rank, right_rank, text_features_to_right)
                    loss += self._loss(image_features, text_features_from_left, logit_scale, logit_bias, negative_only=True)
                    text_features_to_right = text_features_from_left
        return {'contrastive_loss': loss} if output_dict else loss


class GroupWiseLinear(nn.Module):

    def __init__(self, num_class, hidden_dim, bias=True):
        super().__init__()
        self.num_class = num_class
        self.hidden_dim = hidden_dim
        self.bias = bias
        self.W = nn.Parameter(torch.Tensor(1, num_class, hidden_dim))
        if bias:
            self.b = nn.Parameter(torch.Tensor(1, num_class))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.W.size(2))
        for i in range(self.num_class):
            self.W[0][i].data.uniform_(-stdv, stdv)
        if self.bias:
            for i in range(self.num_class):
                self.b[0][i].data.uniform_(-stdv, stdv)

    def forward(self, x):
        x = torch.einsum('ij,lk->il', x, self.W[0])
        if self.bias:
            x = x + self.b
        return x


class CLIP(nn.Module):
    output_dict: 'torch.jit.Final[bool]'

    def __init__(self, embed_dim: 'int', vision_cfg: 'CLIPVisionCfg', text_cfg: 'CLIPTextCfg', quick_gelu: 'bool'=False, init_logit_scale: 'float'=np.log(1 / 0.07), init_logit_bias: 'Optional[float]'=None, cast_dtype: 'Optional[torch.dtype]'=None, output_dict: 'bool'=False, use_meta_noun: 'bool'=False, tag_mode: 'str'='mixed', cache_dir: 'str'=None):
        super().__init__()
        self.output_dict = output_dict
        self.visual = _build_vision_tower(embed_dim, vision_cfg, quick_gelu, cast_dtype)
        text = _build_text_tower(embed_dim, text_cfg, quick_gelu, cast_dtype)
        self.transformer = text.transformer
        self.context_length = text.context_length
        self.vocab_size = text.vocab_size
        self.output_dim = text.output_dim
        self.token_embedding = text.token_embedding
        self.positional_embedding = text.positional_embedding
        self.ln_final = text.ln_final
        self.text_projection = text.text_projection
        self.text_pool_type = text.pool_type
        self.register_buffer('attn_mask', text.attn_mask, persistent=False)
        self.logit_scale = nn.Parameter(torch.ones([]) * init_logit_scale)
        self.logit_scale_local = None
        if init_logit_bias is not None:
            self.logit_bias = nn.Parameter(torch.ones([]) * init_logit_bias)
        else:
            self.logit_bias = None
        self.use_meta_noun = use_meta_noun
        if use_meta_noun:
            self.num_class = 10000
            self.multi_classifier = GroupWiseLinear(self.num_class, text.output_dim, bias=True)
        else:
            self.multi_classifier = nn.Identity()
        self.tag_mode = tag_mode

    def lock_image_tower(self, unlocked_groups=0, freeze_bn_stats=False):
        self.visual.lock(unlocked_groups=unlocked_groups, freeze_bn_stats=freeze_bn_stats)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.visual.set_grad_checkpointing(enable)
        self.transformer.grad_checkpointing = enable

    def encode_image(self, image, normalize: 'bool'=False):
        features, local_image_features = self.visual(image)
        if normalize:
            return F.normalize(features, dim=-1), F.normalize(local_image_features, dim=-1)
        else:
            return features, local_image_features

    def encode_text(self, text, normalize: 'bool'=False):
        cast_dtype = self.transformer.get_cast_dtype()
        x = self.token_embedding(text)
        x = x + self.positional_embedding
        x = x.permute(1, 0, 2)
        x = self.transformer(x, attn_mask=self.attn_mask)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x)
        x, _ = text_global_pool(x, text, self.text_pool_type)
        if self.text_projection is not None:
            if isinstance(self.text_projection, nn.Linear):
                x = self.text_projection(x)
            else:
                x = x @ self.text_projection
        return F.normalize(x, dim=-1) if normalize else x

    def masked_pairwise_contrastive_loss(self, a, b, inverse_temperature=1.0):
        batch_size, seq_len, _ = a.shape
        labels = torch.eye(seq_len).repeat(batch_size, 1)
        logits = torch.einsum('bmd,bnd->bmn', a, b) * inverse_temperature
        loss = F.cross_entropy(logits.view(-1, seq_len), labels)
        return loss

    def forward(self, image: 'Optional[torch.Tensor]'=None, text: 'Optional[torch.Tensor]'=None):
        if image is not None:
            if len(image.shape) == 5:
                image = image.view(image.shape[0] * image.shape[1], image.shape[2], image.shape[3], image.shape[4])
                image_features, local_image_features = self.encode_image(image, normalize=True)
            else:
                image_features, local_image_features = self.encode_image(image, normalize=True)
        else:
            image_features, local_image_features = None, None
        if text is not None and len(text.shape) == 3:
            text_features = self.encode_text(text.view(-1, 77).contiguous(), normalize=True) if text is not None else None
            text_features = text_features.view(-1, text.shape[1], self.output_dim).contiguous()
        else:
            text_features = self.encode_text(text, normalize=True) if text is not None else None
        if self.use_meta_noun and self.training:
            classess_embs = self.multi_classifier(image_features + text_features) if self.tag_mode == 'mixed' else self.multi_classifier(image_features)
        else:
            classess_embs = None
        if self.logit_scale_local:
            logit_scale_local = self.logit_scale_local.exp()
        else:
            logit_scale_local = None
        if self.output_dict:
            out_dict = {'image_features': image_features, 'local_image_features': local_image_features, 'text_features': text_features, 'logit_scale': self.logit_scale.exp(), 'logit_scale_local': logit_scale_local, 'classess_embs': classess_embs}
            if self.logit_bias is not None:
                out_dict['logit_bias'] = self.logit_bias
            return out_dict
        if self.logit_bias is not None:
            return image_features, text_features, self.logit_scale.exp(), self.logit_bias
        return image_features, text_features, self.logit_scale.exp()


class CustomTextCLIP(nn.Module):
    output_dict: 'torch.jit.Final[bool]'

    def __init__(self, embed_dim: 'int', vision_cfg: 'CLIPVisionCfg', text_cfg: 'CLIPTextCfg', quick_gelu: 'bool'=False, init_logit_scale: 'float'=np.log(1 / 0.07), init_logit_bias: 'Optional[float]'=None, cast_dtype: 'Optional[torch.dtype]'=None, output_dict: 'bool'=False, use_meta_noun: 'bool'=False, tag_mode: 'str'='mixed', cache_dir: 'str'=None):
        super().__init__()
        self.output_dict = output_dict
        self.visual = _build_vision_tower(embed_dim, vision_cfg, quick_gelu, cast_dtype)
        self.text = _build_text_tower(embed_dim, text_cfg, quick_gelu, cast_dtype, cache_dir)
        self.context_length = self.text.context_length
        self.vocab_size = self.text.vocab_size
        self.logit_scale = nn.Parameter(torch.ones([]) * init_logit_scale)
        self.logit_scale_local = None
        if init_logit_bias is not None:
            self.logit_bias = nn.Parameter(torch.ones([]) * init_logit_bias)
        else:
            self.logit_bias = None
        self.use_meta_noun = use_meta_noun
        if use_meta_noun:
            self.num_class = 10000
            self.multi_classifier = GroupWiseLinear(self.num_class, self.text.output_dim, bias=True)
        else:
            self.multi_classifier = nn.Identity()
        self.tag_mode = tag_mode

    def lock_image_tower(self, unlocked_groups=0, freeze_bn_stats=False):
        self.visual.lock(unlocked_groups=unlocked_groups, freeze_bn_stats=freeze_bn_stats)

    def lock_text_tower(self, unlocked_layers: 'int'=0, freeze_layer_norm: 'bool'=True):
        self.text.lock(unlocked_layers, freeze_layer_norm)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.visual.set_grad_checkpointing(enable)
        self.text.set_grad_checkpointing(enable)

    def encode_image(self, image, normalize: 'bool'=False):
        features, local_image_features = self.visual(image)
        if normalize:
            return F.normalize(features, dim=-1), F.normalize(local_image_features, dim=-1)
        else:
            return features, local_image_features

    def encode_text(self, text, normalize: 'bool'=False):
        features = self.text(text)
        return F.normalize(features, dim=-1) if normalize else features

    def forward(self, image: 'Optional[torch.Tensor]'=None, text: 'Optional[torch.Tensor]'=None, image_1: 'Optional[torch.Tensor]'=None, image_2: 'Optional[torch.Tensor]'=None):
        bs = image.shape[0]
        if image is not None:
            image_features, local_image_features = self.encode_image(image, normalize=True)
            if image_1 is not None and image_2 is not None:
                image_features_1, local_image_features = self.encode_image(image_1, normalize=True)
                image_features_2, local_image_features = self.encode_image(image_2, normalize=True)
        else:
            image_features, local_image_features = None, None
        if text is not None and len(text.shape) == 3:
            text_features = self.encode_text(text.view(-1, 77).contiguous(), normalize=True) if text is not None else None
            text_features = text_features.view(bs, -1, self.text.output_dim).contiguous()
        else:
            text_features = self.encode_text(text, normalize=True) if text is not None else None
        if self.use_meta_noun and self.training:
            classess_embs = self.multi_classifier(image_features + text_features) if self.tag_mode == 'mixed' else self.multi_classifier(image_features)
        else:
            classess_embs = None
        if self.logit_scale_local:
            logit_scale_local = self.logit_scale_local.exp()
        else:
            logit_scale_local = None
        if self.output_dict:
            if image_1 is not None and image_2 is not None:
                image_features = [image_features, image_features_1, image_features_2]
            out_dict = {'image_features': image_features, 'local_image_features': local_image_features, 'text_features': text_features, 'logit_scale': self.logit_scale.exp(), 'logit_scale_local': logit_scale_local, 'classess_embs': classess_embs}
            if self.logit_bias is not None:
                out_dict['logit_bias'] = self.logit_bias
            return out_dict
        if self.logit_bias is not None:
            return image_features, text_features, self.logit_scale.exp(), self.logit_bias
        return image_features, text_features, self.logit_scale.exp()


def _setup_size(size, error_msg):
    if isinstance(size, numbers.Number):
        return int(size), int(size)
    if isinstance(size, Sequence) and len(size) == 1:
        return size[0], size[0]
    if len(size) != 2:
        raise ValueError(error_msg)
    return size


def center_crop_or_pad(img: 'torch.Tensor', output_size: 'List[int]', fill=0) ->torch.Tensor:
    """Center crops and/or pads the given image.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions.
    If image size is smaller than output size along any edge, image is padded with 0 and then center cropped.

    Args:
        img (PIL Image or Tensor): Image to be cropped.
        output_size (sequence or int): (height, width) of the crop box. If int or sequence with single int,
            it is used for both directions.
        fill (int, Tuple[int]): Padding color

    Returns:
        PIL Image or Tensor: Cropped image.
    """
    if isinstance(output_size, numbers.Number):
        output_size = int(output_size), int(output_size)
    elif isinstance(output_size, (tuple, list)) and len(output_size) == 1:
        output_size = output_size[0], output_size[0]
    _, image_height, image_width = F.get_dimensions(img)
    crop_height, crop_width = output_size
    if crop_width > image_width or crop_height > image_height:
        padding_ltrb = [(crop_width - image_width) // 2 if crop_width > image_width else 0, (crop_height - image_height) // 2 if crop_height > image_height else 0, (crop_width - image_width + 1) // 2 if crop_width > image_width else 0, (crop_height - image_height + 1) // 2 if crop_height > image_height else 0]
        img = F.pad(img, padding_ltrb, fill=fill)
        _, image_height, image_width = F.get_dimensions(img)
        if crop_width == image_width and crop_height == image_height:
            return img
    crop_top = int(round((image_height - crop_height) / 2.0))
    crop_left = int(round((image_width - crop_width) / 2.0))
    return F.crop(img, crop_top, crop_left, crop_height, crop_width)


class CenterCropOrPad(torch.nn.Module):
    """Crops the given image at the center.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions.
    If image size is smaller than output size along any edge, image is padded with 0 and then center cropped.

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made. If provided a sequence of length 1, it will be interpreted as (size[0], size[0]).
    """

    def __init__(self, size, fill=0):
        super().__init__()
        self.size = _setup_size(size, error_msg='Please provide only two dimensions (h, w) for size.')
        self.fill = fill

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be cropped.

        Returns:
            PIL Image or Tensor: Cropped image.
        """
        return center_crop_or_pad(img, self.size, fill=self.fill)

    def __repr__(self) ->str:
        return f'{self.__class__.__name__}(size={self.size})'


class Attention(nn.Module):

    def __init__(self, dim, num_heads=8, qkv_bias=True, scaled_cosine=False, scale_heads=False, logit_scale_max=math.log(1.0 / 0.01), attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.scaled_cosine = scaled_cosine
        self.scale_heads = scale_heads
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.logit_scale_max = logit_scale_max
        self.in_proj_weight = nn.Parameter(torch.randn((dim * 3, dim)) * self.scale)
        if qkv_bias:
            self.in_proj_bias = nn.Parameter(torch.zeros(dim * 3))
        else:
            self.in_proj_bias = None
        if self.scaled_cosine:
            self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))))
        else:
            self.logit_scale = None
        self.attn_drop = nn.Dropout(attn_drop)
        if self.scale_heads:
            self.head_scale = nn.Parameter(torch.ones((num_heads, 1, 1)))
        else:
            self.head_scale = None
        self.out_proj = nn.Linear(dim, dim)
        self.out_drop = nn.Dropout(proj_drop)

    def forward(self, x, attn_mask: 'Optional[torch.Tensor]'=None):
        L, N, C = x.shape
        q, k, v = F.linear(x, self.in_proj_weight, self.in_proj_bias).chunk(3, dim=-1)
        q = q.contiguous().view(L, N * self.num_heads, -1).transpose(0, 1)
        k = k.contiguous().view(L, N * self.num_heads, -1).transpose(0, 1)
        v = v.contiguous().view(L, N * self.num_heads, -1).transpose(0, 1)
        if self.logit_scale is not None:
            attn = torch.bmm(F.normalize(q, dim=-1), F.normalize(k, dim=-1).transpose(-1, -2))
            logit_scale = torch.clamp(self.logit_scale, max=self.logit_scale_max).exp()
            attn = attn.view(N, self.num_heads, L, L) * logit_scale
            attn = attn.view(-1, L, L)
        else:
            q = q * self.scale
            attn = torch.bmm(q, k.transpose(-1, -2))
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                new_attn_mask = torch.zeros_like(attn_mask, dtype=q.dtype)
                new_attn_mask.masked_fill_(attn_mask, float('-inf'))
                attn_mask = new_attn_mask
            attn += attn_mask
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = torch.bmm(attn, v)
        if self.head_scale is not None:
            x = x.view(N, self.num_heads, L, C) * self.head_scale
            x = x.view(-1, L, C)
        x = x.transpose(0, 1).reshape(L, N, C)
        x = self.out_proj(x)
        x = self.out_drop(x)
        return x


class CustomResidualAttentionBlock(nn.Module):

    def __init__(self, d_model: 'int', n_head: 'int', mlp_ratio: 'float'=4.0, ls_init_value: 'float'=None, act_layer: 'Callable'=nn.GELU, norm_layer: 'Callable'=LayerNorm, scale_cosine_attn: 'bool'=False, scale_heads: 'bool'=False, scale_attn: 'bool'=False, scale_fc: 'bool'=False):
        super().__init__()
        self.ln_1 = norm_layer(d_model)
        self.attn = Attention(d_model, n_head, scaled_cosine=scale_cosine_attn, scale_heads=scale_heads)
        self.ln_attn = norm_layer(d_model) if scale_attn else nn.Identity()
        self.ls_1 = LayerScale(d_model, ls_init_value) if ls_init_value is not None else nn.Identity()
        self.ln_2 = norm_layer(d_model)
        mlp_width = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(OrderedDict([('c_fc', nn.Linear(d_model, mlp_width)), ('gelu', act_layer()), ('ln', norm_layer(mlp_width) if scale_fc else nn.Identity()), ('c_proj', nn.Linear(mlp_width, d_model))]))
        self.ls_2 = LayerScale(d_model, ls_init_value) if ls_init_value is not None else nn.Identity()

    def forward(self, x: 'torch.Tensor', attn_mask: 'Optional[torch.Tensor]'=None):
        x = x + self.ls_1(self.ln_attn(self.attn(self.ln_1(x), attn_mask=attn_mask)))
        x = x + self.ls_2(self.mlp(self.ln_2(x)))
        return x


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (AsymmetricLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (Bottleneck,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (CenterCropOrPad,
     lambda: ([], {'size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (GroupWiseLinear,
     lambda: ([], {'num_class': 4, 'hidden_dim': 4}),
     lambda: ([torch.rand([4, 4])], {})),
    (LayerScale,
     lambda: ([], {'dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (PatchDropout,
     lambda: ([], {'prob': 0}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (QuickGELU,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (proxy_anchor_loss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
]

