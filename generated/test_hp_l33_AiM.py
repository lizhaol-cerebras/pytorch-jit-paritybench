
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


import torch.nn as nn


import torch.nn.functional as F


from typing import List


from typing import Optional


from torch import nn


from torch import Tensor


import time


from collections import namedtuple


from functools import partial


from typing import Callable


from typing import Sequence


from typing import Union


from torch.profiler import ProfilerActivity


from torch.profiler import profile


from torch.profiler import record_function


import math


import copy


import warnings


from torchvision import utils as vutils


import pandas as pd


from torch.utils.data import Dataset


from torch.utils.data import DataLoader


from torchvision.datasets.folder import DatasetFolder


from torchvision.datasets.folder import ImageFolder


from torchvision.datasets.folder import IMG_EXTENSIONS


from torchvision.transforms import InterpolationMode


from torchvision.transforms import transforms


from typing import *


def modify_logit_for_repetition_penalty(logits, prev_output_tokens, repetition_penalty=1.0):
    """Apply repetition penalty. See https://arxiv.org/abs/1909.05858
    logits: (batch_size, vocab_size)
    prev_output_tokens: (batch_size, seq_len)
    """
    if repetition_penalty == 1.0:
        return logits
    score = torch.gather(logits, 1, prev_output_tokens)
    score = torch.where(score < 0, score * repetition_penalty, score / repetition_penalty)
    logits.scatter_(1, prev_output_tokens, score)
    return logits


def sample(logits, top_k=1, top_p=0.0, min_p=0.0, temperature=1.0):
    """Sample from top-k logits.
    Arguments:
        logits: Tensor of shape (batch_size, vocab_size)
    """
    assert top_p <= 1.0 and top_p > 0, 'top-p should be in (0, 1].'
    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits.masked_fill_(indices_to_remove, float('-Inf'))
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=False)
        cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
        sorted_indices_to_remove = cumulative_probs <= 1 - top_p
        sorted_indices_to_remove[..., -1:] = False
        indices_to_remove = sorted_indices_to_remove.scatter(sorted_indices.ndim - 1, sorted_indices, sorted_indices_to_remove)
        logits.masked_fill_(indices_to_remove, float('-inf'))
    return torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1).squeeze(dim=-1)


def capture_graph(model, inference_params, batch_size, max_seqlen, decoding_seqlen=1, mempool=None, n_warmups=2, cond=None):
    device = next(iter(model.parameters())).device
    input_ids = torch.full((batch_size, decoding_seqlen), 0, dtype=torch.long, device=device)
    position_ids = torch.full((batch_size, decoding_seqlen), 0, dtype=torch.long, device=device)
    seqlen_offset_og = inference_params.seqlen_offset
    inference_params.seqlen_offset = max_seqlen - decoding_seqlen
    inference_params.lengths_per_sample[:] = inference_params.seqlen_offset
    s = torch.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(n_warmups):
            logits = model(input_ids, position_ids=position_ids, cond=cond, inference_params=inference_params, num_last_tokens=decoding_seqlen).logits
        s.synchronize()
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
    torch.cuda.current_stream().wait_stream(s)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, pool=mempool):
        logits = model(input_ids, position_ids=position_ids, cond=cond, inference_params=inference_params, num_last_tokens=decoding_seqlen).logits

    def run(new_input_ids, new_position_ids, seqlen, cond=None):
        inference_params.lengths_per_sample[:] = seqlen
        input_ids.copy_(new_input_ids)
        position_ids.copy_(new_position_ids)
        graph.replay()
        return logits.clone()
    inference_params.seqlen_offset = seqlen_offset_og
    return run


@torch.inference_mode()
def update_graph_cache(model, cache, batch_size, seqlen_og, max_seqlen, decoding_seqlens=(1,), dtype=None, n_warmups=2, cond=None):
    if cache is None:
        cache = DecodingCGCache()
    param_example = next(iter(model.parameters()))
    device = param_example.device
    if dtype is None:
        dtype = param_example.dtype
    if (device, dtype) != (cache.device, cache.dtype) or batch_size > cache.max_batch_size or max_seqlen > cache.max_seqlen:
        cache.callables = {}
        cache.mempool = None
        cache.inference_params = None
        gc.collect()
        cache.device, cache.dtype = device, dtype
        cache.max_batch_size, cache.max_seqlen = batch_size, max_seqlen
        assert hasattr(model, 'allocate_inference_cache'), 'CUDA graph decoding requires that the model has a method allocate_inference_cache'
        inf_cache = model.allocate_inference_cache(batch_size, max_seqlen, dtype)
        lengths_per_sample = torch.full((batch_size,), seqlen_og, dtype=torch.int32, device=device)
        cache.inference_params = InferenceParams(max_seqlen=max_seqlen, max_batch_size=batch_size, seqlen_offset=seqlen_og, key_value_memory_dict=inf_cache, lengths_per_sample=lengths_per_sample)
        cache.mempool = torch.cuda.graphs.graph_pool_handle()
    for decoding_seqlen in decoding_seqlens:
        if (batch_size, decoding_seqlen) not in cache.callables:
            cache.callables[batch_size, decoding_seqlen] = capture_graph(model, cache.inference_params, batch_size, max_seqlen, decoding_seqlen=decoding_seqlen, mempool=cache.mempool, n_warmups=n_warmups, cond=cond)

    def dispatch(input_ids, position_ids, seqlen, cond=None):
        batch_size, decoding_seqlen = input_ids.shape[:2]
        return cache.callables[batch_size, decoding_seqlen](input_ids, position_ids, seqlen, cond=cond)
    cache.run = dispatch
    cache.inference_params.seqlen_offset = 0
    return cache


@torch.inference_mode()
def decode(input_ids, model, max_length, top_k=1, top_p=0.0, min_p=0.0, temperature=1.0, repetition_penalty=1.0, eos_token_id=None, teacher_outputs=None, vocab_size=None, cg=False, enable_timing=False, cond=None, streamer: 'Optional[TextStreamer]'=None):
    """Decoding, either greedy or with top-k or top-p sampling.
    If top-k = 0, don't limit the number of candidates (pure sampling).
    Top-k and top-p can be used together. If top_k > 0 and top_p > 0, then top-k is applied first,
    then top-p.
    We assume that all sequences in the same batch have the same length.

    Arguments:
        input_ids: (batch, seq_len)
        max_length: int
        teacher_outputs (optional): (batch, seq_len). If provided, instead of sampling from the
            logits, the next token is taken from the teacher_outputs. Useful for testing.
    Returns: GreedySearchDecoderOnlyOutput or SampleDecoderOnlyOutput, with the following fields:
        sequences: (batch, max_length)
        scores: tuples of (batch, vocab_size)
    """
    if streamer is not None:
        streamer.put(input_ids.cpu())
    batch_size, seqlen_og = input_ids.shape
    teacher_output_len = teacher_outputs.shape[1] if teacher_outputs is not None else 0
    if cg:
        if not hasattr(model, '_decoding_cache'):
            model._decoding_cache = None
        model._decoding_cache = update_graph_cache(model, model._decoding_cache, batch_size, seqlen_og, max_length, cond=cond)
        inference_params = model._decoding_cache.inference_params
        inference_params.reset(max_length, batch_size)
    else:
        inference_params = InferenceParams(max_seqlen=max_length, max_batch_size=batch_size)

    def get_logits(input_ids, inference_params, cond=None):
        decoding = inference_params.seqlen_offset > 0
        if decoding:
            position_ids = torch.full((batch_size, 1), inference_params.seqlen_offset, dtype=torch.long, device=input_ids.device)
        else:
            position_ids = None
        if not cg or not decoding:
            logits = model(input_ids, position_ids=position_ids, cond=cond, inference_params=inference_params, num_last_tokens=1).logits.squeeze(dim=1)
        else:
            logits = model._decoding_cache.run(input_ids, position_ids, inference_params.seqlen_offset, cond=cond).squeeze(dim=1)
        return logits[..., :vocab_size] if vocab_size is not None else logits

    def sample_tokens(logits, inference_params):
        if teacher_outputs is None or teacher_output_len <= inference_params.seqlen_offset:
            token = sample(logits, top_k=top_k, top_p=top_p, min_p=min_p, temperature=temperature)
        else:
            token = teacher_outputs[:, inference_params.seqlen_offset]
        return token.unsqueeze(1)

    def should_stop(current_token, inference_params):
        if inference_params.seqlen_offset == 0:
            return False
        if eos_token_id is not None and (current_token == eos_token_id).all():
            return True
        if inference_params.seqlen_offset >= max_length - 1:
            return True
        return False
    start = torch.Event(enable_timing=enable_timing)
    end = torch.Event(enable_timing=enable_timing)
    if enable_timing:
        start.record()
    scores, sequences = [], [input_ids]
    sequences_cat = input_ids
    while not should_stop(sequences[-1], inference_params):
        scores.append(get_logits(sequences[-1], inference_params, cond=cond))
        inference_params.seqlen_offset += sequences[-1].shape[1]
        if repetition_penalty == 1.0:
            sampled_tokens = sample_tokens(scores[-1], inference_params)
        else:
            logits = modify_logit_for_repetition_penalty(scores[-1].clone(), sequences_cat, repetition_penalty)
            sampled_tokens = sample_tokens(logits, inference_params)
            sequences_cat = torch.cat([sequences_cat, sampled_tokens], dim=1)
        sequences.append(sampled_tokens)
        if streamer is not None:
            streamer.put(sampled_tokens.cpu())
    if streamer is not None:
        streamer.end()
    if enable_timing:
        end.record()
        torch.cuda.synchronize()
        None
    output_cls = GreedySearchDecoderOnlyOutput if top_k == 1 else SampleDecoderOnlyOutput
    return output_cls(sequences=torch.cat(sequences, dim=1), scores=tuple(scores))


class GenerationMixin:

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        raise NotImplementedError

    def generate(self, input_ids, max_length, top_k=1, top_p=0.0, min_p=0.0, temperature=1.0, return_dict_in_generate=False, output_scores=False, cond=None, **kwargs):
        output = decode(input_ids, self, max_length, top_k=top_k, top_p=top_p, min_p=min_p, temperature=temperature, cond=cond, **kwargs)
        if not output_scores:
            output.scores = None
        return output if return_dict_in_generate else output.sequences


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class FinalLayer(nn.Module):

    def __init__(self, hidden_size):
        super().__init__()
        self.norm_final = RMSNorm(hidden_size)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        return x


class GPT2Embeddings(nn.Module):

    def __init__(self, embed_dim, vocab_size, max_position_embeddings, padding_idx=None, word_embed_proj_dim=None, token_drop=0.0, device=None, dtype=None):
        """
        If max_position_embeddings <= 0, there's no position embeddings
        If word_embe_proj_dim is not None (e.g., OPT-350m), we embed to that dimension
            the project up to embed_dim
        """
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        if word_embed_proj_dim is None:
            self.word_embeddings = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx, **factory_kwargs)
            self.project_in = None
        else:
            self.word_embeddings = nn.Embedding(vocab_size, word_embed_proj_dim, padding_idx=padding_idx, **factory_kwargs)
            self.project_in = nn.Linear(word_embed_proj_dim, embed_dim, bias=False, **factory_kwargs)
        self.max_position_embeddings = max_position_embeddings
        if self.max_position_embeddings > 0:
            self.position_embeddings = nn.Embedding(max_position_embeddings, embed_dim, **factory_kwargs)
        self.token_dropout = nn.Dropout(token_drop)

    def forward(self, input_ids, position_ids=None):
        """
        input_ids: (batch, seqlen)
        position_ids: (batch, seqlen)
        """
        batch_size, seqlen = input_ids.shape
        embeddings = self.token_dropout(self.word_embeddings(input_ids))
        if self.project_in is not None:
            embeddings = self.project_in(embeddings)
        if self.max_position_embeddings > 0:
            if position_ids is None:
                position_ids = torch.arange(seqlen, dtype=torch.long, device=input_ids.device)
            position_embeddings = self.position_embeddings(position_ids)
            embeddings = embeddings + position_embeddings
        return embeddings


class GroupAdaLN(nn.Linear):

    def __init__(self, in_features, out_features, num_channels, bias=True):
        super(GroupAdaLN, self).__init__(in_features, out_features, bias)
        self.num_channels = num_channels

    def forward(self, cond):
        channels = self.weight.shape[0] // self.num_channels
        return super().forward(cond).view(-1, self.num_channels, channels)


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """

    def __init__(self, num_classes, hidden_size, dropout_prob=0.1):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if train and use_dropout or force_drop_ids is not None:
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


def _init_weights(module, n_layer, initializer_range=0.02, rescale_prenorm_residual=True, n_residuals_per_layer=1):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, '_no_reinit', False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)
    if rescale_prenorm_residual:
        for name, p in module.named_parameters():
            if name in ['out_proj.weight', 'fc2.weight']:
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


class Block(nn.Module):

    def __init__(self, dim, mixer_cls, mlp_cls, norm_cls=nn.LayerNorm, fused_add_norm=False, residual_in_fp32=False, adaln_group=False, mixer_drop=0.0, mlp_drop=0.0):
        """
        Simple block wrapping a mixer class with LayerNorm/RMSNorm and residual connection"

        This Block has a slightly different structure compared to a regular
        prenorm Transformer block.
        The standard block is: LN -> MHA/MLP -> Add.
        [Ref: https://arxiv.org/abs/2002.04745]
        Here we have: Add -> LN -> Mixer, returning both
        the hidden_states (output of the mixer) and the residual.
        This is purely for performance reasons, as we can fuse add and LayerNorm.
        The residual needs to be provided (except for the very first block).
        """
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.norm = norm_cls(dim)
        self.mixer = mixer_cls(dim)
        self.mixer_dropout = nn.Dropout(mixer_drop)
        self.adaln_group = adaln_group
        self.adaln_factor = 3
        if mlp_cls is not nn.Identity:
            self.norm2 = norm_cls(dim)
            self.mlp = mlp_cls(dim)
            self.adaln_factor += 3
            self.mlp_dropout = nn.Dropout(0.0)
        else:
            self.mlp = None
        if self.fused_add_norm:
            assert RMSNorm is not None, 'RMSNorm import fails'
            assert isinstance(self.norm, (nn.LayerNorm, RMSNorm)), 'Only LayerNorm and RMSNorm are supported for fused_add_norm'
        if adaln_group:
            self.scale_shift_table = nn.Parameter(torch.randn(1, self.adaln_factor, dim) / dim ** 0.5)
        else:
            self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, self.adaln_factor * dim, bias=True))
            nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, hidden_states: 'Tensor', residual: 'Optional[Tensor]'=None, cls_embed=None, inference_params=None, **mixer_kwargs):
        """Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Mixer(LN(residual))
        """
        if not self.fused_add_norm:
            residual = hidden_states + residual if residual is not None else hidden_states
            hidden_states = self.norm(residual)
            if self.residual_in_fp32:
                residual = residual
        else:
            hidden_states, residual = layer_norm_fn(hidden_states, self.norm.weight, self.norm.bias, residual=residual, prenorm=True, residual_in_fp32=self.residual_in_fp32, eps=self.norm.eps, is_rms_norm=isinstance(self.norm, RMSNorm))
        if self.adaln_group:
            scale_shift_params = (self.scale_shift_table + cls_embed).unbind(1)
        else:
            scale_shift_params = self.adaLN_modulation(cls_embed).chunk(self.adaln_factor, dim=1)
        if self.adaln_factor == 3:
            shift_mixer, scale_mixer, gate_mixer = scale_shift_params
        elif self.adaln_factor == 6:
            shift_mixer, shift_mlp, scale_mixer, scale_mlp, gate_mixer, gate_mlp = scale_shift_params
        else:
            raise ValueError('Unsupported adaln_factor value')
        hidden_states = self.mixer_dropout(gate_mixer.unsqueeze(1) * self.mixer(modulate(hidden_states, shift_mixer, scale_mixer), inference_params=inference_params, **mixer_kwargs))
        if self.mlp is not None:
            if not self.fused_add_norm:
                residual = hidden_states + residual
                residual = self.norm2(residual)
                if self.residual_in_fp32:
                    residual = residual
            else:
                hidden_states, residual = layer_norm_fn(hidden_states, self.norm2.weight, self.norm2.bias, residual=residual, prenorm=True, residual_in_fp32=self.residual_in_fp32, eps=self.norm2.eps, is_rms_norm=isinstance(self.norm2, RMSNorm))
            hidden_states = self.mlp_dropout(gate_mlp.unsqueeze(1) * self.mlp(modulate(hidden_states, shift_mlp, scale_mlp)))
        return hidden_states, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)


def create_block(d_model, d_intermediate, ssm_cfg=None, attn_layer_idx=None, attn_cfg=None, norm_epsilon=1e-05, rms_norm=False, residual_in_fp32=False, fused_add_norm=False, layer_idx=None, adaln_group=False, mixer_drop=0.0, mlp_drop=0.0, device=None, dtype=None):
    if ssm_cfg is None:
        ssm_cfg = {}
    if attn_layer_idx is None:
        attn_layer_idx = []
    if attn_cfg is None:
        attn_cfg = {}
    factory_kwargs = {'device': device, 'dtype': dtype}
    if layer_idx not in attn_layer_idx:
        ssm_cfg = copy.deepcopy(ssm_cfg) if ssm_cfg is not None else {}
        ssm_layer = ssm_cfg.pop('layer', 'Mamba1')
        if ssm_layer not in ['Mamba1', 'Mamba2']:
            raise ValueError(f'Invalid ssm_layer: {ssm_layer}, only support Mamba1 and Mamba2')
        mixer_cls = partial(Mamba2 if ssm_layer == 'Mamba2' else Mamba, layer_idx=layer_idx, **ssm_cfg, **factory_kwargs)
    else:
        mixer_cls = partial(MHA, layer_idx=layer_idx, **attn_cfg, **factory_kwargs)
    norm_cls = partial(nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs)
    if d_intermediate == 0:
        mlp_cls = nn.Identity
    else:
        mlp_cls = partial(GatedMLP, hidden_features=d_intermediate, out_features=d_model, **factory_kwargs)
    block = Block(d_model, mixer_cls, mlp_cls, norm_cls=norm_cls, fused_add_norm=fused_add_norm, residual_in_fp32=residual_in_fp32, adaln_group=adaln_group)
    block.layer_idx = layer_idx
    return block


class MixerModel(nn.Module):

    def __init__(self, d_model: 'int', n_layer: 'int', d_intermediate: 'int', vocab_size: 'int', ssm_cfg=None, attn_layer_idx=None, attn_cfg=None, norm_epsilon: 'float'=1e-05, rms_norm: 'bool'=False, initializer_cfg=None, fused_add_norm=False, residual_in_fp32=False, num_classes=1000, num_tokens=256, adaln_group=False, num_groups=4, token_drop=0.0, mixer_drop=0.0, mlp_drop=0.0, device=None, dtype=None) ->None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.embeddings = GPT2Embeddings(d_model, vocab_size, num_tokens + 1, token_drop=token_drop, **factory_kwargs)
        self.cls_embed = LabelEmbedder(num_classes=num_classes, hidden_size=d_model)
        adaln_factor = 3 + (3 if d_intermediate != 0 else 0)
        if adaln_group:
            self.adaln_group = nn.Sequential(nn.SiLU(inplace=False), GroupAdaLN(d_model, num_groups * adaln_factor * d_model, num_channels=num_groups * adaln_factor))
            self.num_groups = num_groups
        else:
            self.adaln_group = nn.Identity()
            self.num_groups = 1
        self.final_layer = FinalLayer(d_model)
        self.fused_add_norm = fused_add_norm
        if self.fused_add_norm:
            if layer_norm_fn is None or rms_norm_fn is None:
                raise ImportError('Failed to import Triton LayerNorm / RMSNorm kernels')
        self.layers = nn.ModuleList([create_block(d_model, d_intermediate=d_intermediate, ssm_cfg=ssm_cfg, attn_layer_idx=attn_layer_idx, attn_cfg=attn_cfg, norm_epsilon=norm_epsilon, rms_norm=rms_norm, residual_in_fp32=residual_in_fp32, fused_add_norm=fused_add_norm, layer_idx=i, adaln_group=adaln_group, mixer_drop=mixer_drop, mlp_drop=mlp_drop, **factory_kwargs) for i in range(n_layer)])
        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(d_model, eps=norm_epsilon, **factory_kwargs)
        self.apply(partial(_init_weights, n_layer=n_layer, **initializer_cfg if initializer_cfg is not None else {}, n_residuals_per_layer=1 if d_intermediate == 0 else 2))

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs) for i, layer in enumerate(self.layers)}

    def forward(self, input_ids, position_ids, cond, inference_params=None, **mixer_kwargs):
        is_train = inference_params is None
        cond_embed = self.cls_embed(cond.squeeze(1), train=is_train)
        if is_train:
            token_embed = self.embeddings(input_ids, position_ids=position_ids)
            hidden_states = torch.cat([cond_embed.unsqueeze(1), token_embed], dim=1)
        elif inference_params.seqlen_offset == 0:
            assert (input_ids == cond).any(), 'first inputs_ids must equal to cond'
            hidden_states = self.cls_embed(cond, train=is_train)
        else:
            hidden_states = self.embeddings(input_ids, position_ids=position_ids)
        ada_cond = self.adaln_group(cond_embed).chunk(self.num_groups, dim=1)
        residual = None
        for i, layer in enumerate(self.layers):
            hidden_states, residual = layer(hidden_states, residual, ada_cond[i % self.num_groups], inference_params=inference_params)
        if not self.fused_add_norm:
            residual = hidden_states + residual if residual is not None else hidden_states
            hidden_states = self.norm_f(residual)
        else:
            hidden_states = layer_norm_fn(hidden_states, self.norm_f.weight, self.norm_f.bias, eps=self.norm_f.eps, residual=residual, prenorm=False, residual_in_fp32=self.residual_in_fp32, is_rms_norm=isinstance(self.norm_f, RMSNorm))
        hidden_states = self.final_layer(hidden_states, cond_embed)
        return hidden_states


class MambaLMHeadModel(nn.Module, GenerationMixin):

    def __init__(self, config: 'MambaConfig', initializer_cfg=None, device=None, dtype=None) ->None:
        super().__init__()
        self.config = config
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.cfg_scale = 1.5
        self.backbone = MixerModel(d_model=config.d_model, n_layer=config.n_layer, d_intermediate=config.d_intermediate, vocab_size=config.vocab_size, ssm_cfg=config.ssm_cfg, attn_layer_idx=config.attn_layer_idx, attn_cfg=config.attn_cfg, rms_norm=config.rms_norm, initializer_cfg=initializer_cfg, fused_add_norm=config.fused_add_norm, residual_in_fp32=config.residual_in_fp32, num_classes=config.num_classes, num_tokens=config.num_tokens, adaln_group=config.adaln_group, num_groups=config.num_groups, token_drop=config.token_drop, mixer_drop=config.mixer_drop, mlp_drop=config.mlp_drop, **factory_kwargs)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False, **factory_kwargs)
        self.apply(partial(_init_weights, n_layer=config.n_layer, **initializer_cfg if initializer_cfg is not None else {}))
        self.tie_weights()

    def tie_weights(self):
        if self.config.tie_embeddings:
            self.lm_head.weight = self.backbone.embeddings.word_embeddings.weight

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.backbone.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)

    def forward(self, input_ids, position_ids=None, cond=None, inference_params=None, num_last_tokens=0, **mixer_kwargs):
        """
        num_last_tokens: if > 0, only return the logits for the last n tokens
        """
        is_train = inference_params is None
        if not is_train and inference_params.seqlen_offset > 0:
            input_ids, _ = torch.split(input_ids, len(input_ids) // 2, dim=0)
            input_ids = torch.cat([input_ids, input_ids])
        hidden_states = self.backbone(input_ids, position_ids, cond, inference_params=inference_params, **mixer_kwargs)
        if num_last_tokens > 0:
            hidden_states = hidden_states[:, -num_last_tokens:]
        lm_logits = self.lm_head(hidden_states)
        if not is_train:
            cond_logits, uncond_logits = torch.split(lm_logits, len(lm_logits) // 2, dim=0)
            lm_logits = uncond_logits + (cond_logits - uncond_logits) * self.cfg_scale
            lm_logits = lm_logits.repeat(2, 1, 1)
        CausalLMOutput = namedtuple('CausalLMOutput', ['logits'])
        return CausalLMOutput(logits=lm_logits)

    @classmethod
    def from_pretrained(cls, pretrained_model_name, device=None, dtype=None, **kwargs):
        config_data = load_config_hf(pretrained_model_name)
        config = MambaConfig(**config_data)
        model = cls(config, device=device, dtype=dtype, **kwargs)
        model.load_state_dict(load_state_dict_hf(pretrained_model_name, device=device, dtype=dtype))
        return model

    def save_pretrained(self, save_directory):
        """
        Minimal implementation of save_pretrained for MambaLMHeadModel.
        Save the model and its configuration file to a directory.
        """
        os.makedirs(save_directory, exist_ok=True)
        model_path = os.path.join(save_directory, 'pytorch_model.bin')
        torch.save(self.state_dict(), model_path)
        config_path = os.path.join(save_directory, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(self.config.__dict__, f, indent=4)


def Normalize(in_channels, norm_type='group'):
    assert norm_type in ['group', 'batch']
    if norm_type == 'group':
        return nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-06, affine=True)
    elif norm_type == 'batch':
        return nn.SyncBatchNorm(in_channels)


class AttnBlock(nn.Module):

    def __init__(self, in_channels, norm_type='group'):
        super().__init__()
        self.norm = Normalize(in_channels, norm_type)
        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)
        b, c, h, w = q.shape
        q = q.reshape(b, c, h * w)
        q = q.permute(0, 2, 1)
        k = k.reshape(b, c, h * w)
        w_ = torch.bmm(q, k)
        w_ = w_ * int(c) ** -0.5
        w_ = F.softmax(w_, dim=2)
        v = v.reshape(b, c, h * w)
        w_ = w_.permute(0, 2, 1)
        h_ = torch.bmm(v, w_)
        h_ = h_.reshape(b, c, h, w)
        h_ = self.proj_out(h_)
        return x + h_


def nonlinearity(x):
    return x * torch.sigmoid(x)


class ResnetBlock(nn.Module):

    def __init__(self, in_channels, out_channels=None, conv_shortcut=False, dropout=0.0, norm_type='group'):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        self.norm1 = Normalize(in_channels, norm_type)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = Normalize(out_channels, norm_type)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            else:
                self.nin_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)
        return x + h


class Upsample(nn.Module):

    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode='nearest')
        if self.with_conv:
            x = self.conv(x)
        return x


class Decoder(nn.Module):

    def __init__(self, z_channels=256, ch=128, ch_mult=(1, 1, 2, 2, 4), num_res_blocks=2, norm_type='group', dropout=0.0, resamp_with_conv=True, out_channels=3):
        super().__init__()
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        block_in = ch * ch_mult[self.num_resolutions - 1]
        self.conv_in = nn.Conv2d(z_channels, block_in, kernel_size=3, stride=1, padding=1)
        self.mid = nn.ModuleList()
        self.mid.append(ResnetBlock(block_in, block_in, dropout=dropout, norm_type=norm_type))
        self.mid.append(AttnBlock(block_in, norm_type=norm_type))
        self.mid.append(ResnetBlock(block_in, block_in, dropout=dropout, norm_type=norm_type))
        self.conv_blocks = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            conv_block = nn.Module()
            res_block = nn.ModuleList()
            attn_block = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks + 1):
                res_block.append(ResnetBlock(block_in, block_out, dropout=dropout, norm_type=norm_type))
                block_in = block_out
                if i_level == self.num_resolutions - 1:
                    attn_block.append(AttnBlock(block_in, norm_type))
            conv_block.res = res_block
            conv_block.attn = attn_block
            if i_level != 0:
                conv_block.upsample = Upsample(block_in, resamp_with_conv)
            self.conv_blocks.append(conv_block)
        self.norm_out = Normalize(block_in, norm_type)
        self.conv_out = nn.Conv2d(block_in, out_channels, kernel_size=3, stride=1, padding=1)

    @property
    def last_layer(self):
        return self.conv_out.weight

    def forward(self, z):
        h = self.conv_in(z)
        for mid_block in self.mid:
            h = mid_block(h)
        for i_level, block in enumerate(self.conv_blocks):
            for i_block in range(self.num_res_blocks + 1):
                h = block.res[i_block](h)
                if len(block.attn) > 0:
                    h = block.attn[i_block](h)
            if i_level != self.num_resolutions - 1:
                h = block.upsample(h)
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class Downsample(nn.Module):

    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = 0, 1, 0, 1
            x = F.pad(x, pad, mode='constant', value=0)
            x = self.conv(x)
        else:
            x = F.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class Encoder(nn.Module):

    def __init__(self, in_channels=3, ch=128, ch_mult=(1, 1, 2, 2, 4), num_res_blocks=2, norm_type='group', dropout=0.0, resamp_with_conv=True, z_channels=256):
        super().__init__()
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.conv_in = nn.Conv2d(in_channels, ch, kernel_size=3, stride=1, padding=1)
        in_ch_mult = (1,) + tuple(ch_mult)
        self.conv_blocks = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            conv_block = nn.Module()
            res_block = nn.ModuleList()
            attn_block = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks):
                res_block.append(ResnetBlock(block_in, block_out, dropout=dropout, norm_type=norm_type))
                block_in = block_out
                if i_level == self.num_resolutions - 1:
                    attn_block.append(AttnBlock(block_in, norm_type))
            conv_block.res = res_block
            conv_block.attn = attn_block
            if i_level != self.num_resolutions - 1:
                conv_block.downsample = Downsample(block_in, resamp_with_conv)
            self.conv_blocks.append(conv_block)
        self.mid = nn.ModuleList()
        self.mid.append(ResnetBlock(block_in, block_in, dropout=dropout, norm_type=norm_type))
        self.mid.append(AttnBlock(block_in, norm_type=norm_type))
        self.mid.append(ResnetBlock(block_in, block_in, dropout=dropout, norm_type=norm_type))
        self.norm_out = Normalize(block_in, norm_type)
        self.conv_out = nn.Conv2d(block_in, z_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        h = self.conv_in(x)
        for i_level, block in enumerate(self.conv_blocks):
            for i_block in range(self.num_res_blocks):
                h = block.res[i_block](h)
                if len(block.attn) > 0:
                    h = block.attn[i_block](h)
            if i_level != self.num_resolutions - 1:
                h = block.downsample(h)
        for mid_block in self.mid:
            h = mid_block(h)
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


def compute_entropy_loss(affinity, loss_type='softmax', temperature=0.01):
    flat_affinity = affinity.reshape(-1, affinity.shape[-1])
    flat_affinity /= temperature
    probs = F.softmax(flat_affinity, dim=-1)
    log_probs = F.log_softmax(flat_affinity + 1e-05, dim=-1)
    if loss_type == 'softmax':
        target_probs = probs
    else:
        raise ValueError('Entropy loss {} not supported'.format(loss_type))
    avg_probs = torch.mean(target_probs, dim=0)
    avg_entropy = -torch.sum(avg_probs * torch.log(avg_probs + 1e-05))
    sample_entropy = -torch.mean(torch.sum(target_probs * log_probs, dim=-1))
    loss = sample_entropy - avg_entropy
    return loss


class VectorQuantizer(nn.Module):

    def __init__(self, n_e, e_dim, beta, entropy_loss_ratio, l2_norm, show_usage):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.entropy_loss_ratio = entropy_loss_ratio
        self.l2_norm = l2_norm
        self.show_usage = show_usage
        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)
        if self.l2_norm:
            self.embedding.weight.data = F.normalize(self.embedding.weight.data, p=2, dim=-1)
        if self.show_usage:
            self.register_buffer('codebook_used', nn.Parameter(torch.zeros(65536)))

    def forward(self, z):
        z = torch.einsum('b c h w -> b h w c', z).contiguous()
        z_flattened = z.view(-1, self.e_dim)
        if self.l2_norm:
            z = F.normalize(z, p=2, dim=-1)
            z_flattened = F.normalize(z_flattened, p=2, dim=-1)
            embedding = F.normalize(self.embedding.weight, p=2, dim=-1)
        else:
            embedding = self.embedding.weight
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + torch.sum(embedding ** 2, dim=1) - 2 * torch.einsum('bd,dn->bn', z_flattened, torch.einsum('n d -> d n', embedding))
        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = embedding[min_encoding_indices].view(z.shape)
        perplexity = None
        min_encodings = None
        vq_loss = None
        commit_loss = None
        entropy_loss = None
        codebook_usage = 0
        if self.show_usage and self.training:
            cur_len = min_encoding_indices.shape[0]
            self.codebook_used[:-cur_len] = self.codebook_used[cur_len:].clone()
            self.codebook_used[-cur_len:] = min_encoding_indices
            codebook_usage = len(torch.unique(self.codebook_used)) / self.n_e
        if self.training:
            vq_loss = torch.mean((z_q - z.detach()) ** 2)
            commit_loss = self.beta * torch.mean((z_q.detach() - z) ** 2)
            entropy_loss = self.entropy_loss_ratio * compute_entropy_loss(-d)
        z_q = z + (z_q - z).detach()
        z_q = torch.einsum('b h w c -> b c h w', z_q)
        return z_q, (vq_loss, commit_loss, entropy_loss, codebook_usage), (perplexity, min_encodings, min_encoding_indices)

    def get_codebook_entry(self, indices, shape=None, channel_first=True):
        if self.l2_norm:
            embedding = F.normalize(self.embedding.weight, p=2, dim=-1)
        else:
            embedding = self.embedding.weight
        z_q = embedding[indices]
        if shape is not None:
            if channel_first:
                z_q = z_q.reshape(shape[0], shape[2], shape[3], shape[1])
                z_q = z_q.permute(0, 3, 1, 2).contiguous()
            else:
                z_q = z_q.view(shape)
        return z_q


class VQModel(nn.Module):

    def __init__(self, config: 'VQConfig'):
        super().__init__()
        self.config = config
        self.encoder = Encoder(ch_mult=config.ch_mult, z_channels=config.z_channels, num_res_blocks=config.num_res_blocks)
        self.decoder = Decoder(ch_mult=config.ch_mult, z_channels=config.z_channels, num_res_blocks=config.num_res_blocks)
        self.quantize = VectorQuantizer(config.n_embed, config.embed_dim, 0.25, 0.0, True, True)
        self.quant_conv = nn.Conv2d(config.z_channels, config.embed_dim, 1)
        self.post_quant_conv = nn.Conv2d(config.embed_dim, config.z_channels, 1)

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h)
        return quant, emb_loss, info

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

    def decode_code(self, code_b, shape=None, channel_first=True):
        quant_b = self.quantize.get_codebook_entry(code_b, shape, channel_first)
        dec = self.decode(quant_b)
        return dec

    def forward(self, input):
        quant, diff, _ = self.encode(input)
        dec = self.decode(quant)
        return dec, diff

    def from_pretrained(self, path):
        sd = torch.load(path, map_location='cpu')
        self.load_state_dict(sd['model'], strict=True)
        None


def VQ_f16(**kwargs):
    return VQModel(VQConfig())


VQ_models = {'VQ-f16': VQ_f16}


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (Downsample,
     lambda: ([], {'in_channels': 4, 'with_conv': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Encoder,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {})),
    (GroupAdaLN,
     lambda: ([], {'in_features': 4, 'out_features': 4, 'num_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Upsample,
     lambda: ([], {'in_channels': 4, 'with_conv': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (VectorQuantizer,
     lambda: ([], {'n_e': 4, 'e_dim': 4, 'beta': 4, 'entropy_loss_ratio': MSELoss(), 'l2_norm': 4, 'show_usage': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
]

