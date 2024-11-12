
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


from torch._guards import active_fake_mode


from torch._subclasses.fake_tensor import FakeTensorMode


from torch.distributed._tools.fsdp2_mem_tracker import FSDPMemTracker


from torch.testing._internal.distributed.fake_pg import FakeStore


import torch.distributed.checkpoint as DCP


import math


from typing import Optional


from typing import Union


from torch import nn


from torch.distributed._tensor import distribute_tensor


from torch.distributed._tensor import init_device_mesh


from torch.distributed._tensor import Replicate


from torch.distributed._tensor import Shard


from torch.distributed.tensor.debug import CommDebugMode


from torch.testing._internal.common_utils import run_tests


from torch.testing._internal.distributed._tensor.common_dtensor import DTensorTestBase


from torch.testing._internal.distributed._tensor.common_dtensor import skip_if_lt_x_gpu


from torch.testing._internal.distributed._tensor.common_dtensor import with_comms


import logging


from collections import defaultdict


from typing import Sequence


import enum


import functools


import re


import time


from typing import Any


from typing import Dict


from typing import List


import torch.distributed as dist


import torch.distributed.checkpoint as dcp


import torch.nn as nn


from torch.distributed.checkpoint.state_dict import get_model_state_dict


from torch.distributed.checkpoint.state_dict import get_optimizer_state_dict


from torch.distributed.checkpoint.state_dict import set_model_state_dict


from torch.distributed.checkpoint.state_dict import set_optimizer_state_dict


from torch.distributed.checkpoint.state_dict import StateDictOptions


from torch.distributed.checkpoint.stateful import Stateful


from torch.utils.data import DataLoader


from typing import Tuple


from torch.utils.data import IterableDataset


from collections import namedtuple


from torch.utils.tensorboard import SummaryWriter


import torch.nn.functional as F


from functools import partial


from torch.distributed._tensor import Partial


from torch.distributed._tensor.experimental import local_map


from torch.optim.lr_scheduler import LambdaLR


from functools import cached_property


from torch.distributed.device_mesh import init_device_mesh


from torch.distributed import DeviceMesh


from torch.distributed._composable.fsdp import CPUOffloadPolicy


from torch.distributed._composable.fsdp import fully_shard


from torch.distributed._composable.fsdp import MixedPrecisionPolicy


from torch.distributed._composable.replicate import replicate


from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper as ptd_checkpoint_wrapper


from torch.distributed.tensor.parallel import ColwiseParallel


from torch.distributed.tensor.parallel import parallelize_module


from torch.distributed.tensor.parallel import PrepareModuleInput


from torch.distributed.tensor.parallel import RowwiseParallel


from torch.distributed.tensor.parallel import SequenceParallel


import copy


from typing import Callable


from torch.distributed.pipelining import PipelineStage


from torch.distributed.pipelining.schedules import get_schedule_class


from torch.distributed.pipelining.schedules import PipelineScheduleMulti


from torch.distributed.pipelining.schedules import PipelineScheduleSingle


from typing import Generator


from typing import Set


import torch.distributed._functional_collectives as funcol


import torch.distributed.distributed_c10d as c10d


from torch.distributed.device_mesh import DeviceMesh


from torch.distributed.elastic.multiprocessing.errors import record


def reshape_for_broadcast(freqs_cis: 'torch.Tensor', x: 'torch.Tensor') ->torch.Tensor:
    """
    Reshape frequency tensor for broadcasting it with another tensor.

    This function reshapes the frequency tensor to have the same shape as the target tensor 'x'
    for the purpose of broadcasting the frequency tensor during element-wise operations.

    The input freqs_cis tensor is assumed to be of shape (max_seqlen, dim),
    and the first seqlen elements will be sliced, but dim must match x.

    Args:
        freqs_cis (torch.Tensor): Frequency tensor to be reshaped.
        x (torch.Tensor): Target tensor for broadcasting compatibility.

    Returns:
        torch.Tensor: Reshaped frequency tensor.
    """
    ndim = x.ndim
    assert 0 <= 1 < ndim
    seqlen = x.shape[1]
    freqs_cis = freqs_cis[0:seqlen]
    assert freqs_cis.shape == (seqlen, x.shape[-1])
    shape = [(d if i == 1 or i == ndim - 1 else 1) for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(xq: 'torch.Tensor', xk: 'torch.Tensor', freqs_cis: 'torch.Tensor') ->Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor.

    This function applies rotary embeddings to the given query 'xq' and key 'xk' tensors using the provided
    frequency tensor 'freqs_cis'. The input tensors are reshaped as complex numbers, and the frequency tensor
    is reshaped for broadcasting compatibility. The resulting tensors contain rotary embeddings and are
    returned as real tensors.

    Args:
        xq (torch.Tensor): Query tensor to apply rotary embeddings.
        xk (torch.Tensor): Key tensor to apply rotary embeddings.
        freqs_cis (torch.Tensor): Precomputed frequency tensor for complex exponentials.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.
    """
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: 'torch.Tensor', num_rep: 'int') ->torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=num_rep)"""
    bsz, seq_len, num_kv_heads, head_dim = x.shape
    if num_rep == 1:
        return x
    return torch.unsqueeze(x, dim=3).expand(bsz, seq_len, num_kv_heads, num_rep, head_dim).reshape(bsz, seq_len, num_kv_heads * num_rep, head_dim)


class Attention(nn.Module):
    """
    Multi-head attention module.

    Args:
        model_args (ModelArgs): Model configuration arguments.

    Attributes:
        num_kv_heads (int): Number of key and value heads.
        num_heads (int): Number of query heads.
        num_rep (int): Number of repetitions for local heads.
        head_dim (int): Dimension size of each attention head.
        wq (Linear): Linear transformation for queries.
        wk (Linear): Linear transformation for keys.
        wv (Linear): Linear transformation for values.
        wo (Linear): Linear transformation for output.

    """

    def __init__(self, model_args: 'ModelArgs'):
        super().__init__()
        self.num_heads = model_args.encoder_num_heads
        self.num_kv_heads = model_args.encoder_num_heads if model_args.encoder_num_kv_heads is None else model_args.encoder_num_kv_heads
        self.num_rep = self.num_heads // self.num_kv_heads
        self.head_dim = model_args.encoder_embed_dim // model_args.encoder_num_heads
        self.wq = nn.Linear(model_args.encoder_embed_dim, model_args.encoder_num_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(model_args.encoder_embed_dim, self.num_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(model_args.encoder_embed_dim, self.num_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(model_args.encoder_num_heads * self.head_dim, model_args.encoder_embed_dim, bias=False)
        self.is_causal = model_args.is_causal

    def init_weights(self, init_std: 'float'):
        for linear in (self.wq, self.wk, self.wv):
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.wo.weight, mean=0.0, std=init_std)

    def forward(self, x: 'torch.Tensor', freqs_cis: 'torch.Tensor'):
        """
        Forward pass of the attention module.

        Args:
            x (torch.Tensor): Input tensor.
            freqs_cis (torch.Tensor): Precomputed frequency tensor.

        Returns:
            torch.Tensor: Output tensor after attention.

        """
        bs, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = xq.view(bs, seqlen, -1, self.head_dim)
        xk = xk.view(bs, seqlen, -1, self.head_dim)
        xv = xv.view(bs, seqlen, -1, self.head_dim)
        if freqs_cis is not None:
            xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)
        keys = repeat_kv(xk, self.num_rep)
        values = repeat_kv(xv, self.num_rep)
        xq = xq.transpose(1, 2)
        xk = keys.transpose(1, 2)
        xv = values.transpose(1, 2)
        output = F.scaled_dot_product_attention(xq, xk, xv, is_causal=self.is_causal)
        output = output.transpose(1, 2).contiguous()
        output = output.view(bs, seqlen, -1)
        return self.wo(output)


class FeedForward(nn.Module):
    """
    FeedForward module

    Args:
        dim (int): Input dimension.
        hidden_dim (int): Hidden dimension of the feedforward layer.
        multiple_of (int): Value to ensure hidden dimension is a multiple of this value.
        ffn_dim_multiplier (Optional[float]): Custom multiplier for hidden dimension. Defaults to None.
        activation: (nn.Module): Activation function to use. Defaults to nn.silu.

    Attributes:
        w1 (Linear): Linear transformation for the first layer, which projects input from input dim to
            hidden dim, and multiplies by the projection from w3 for activation and second layer.
        w2 (Linear): Linear transformation for the second layer.
    """

    def __init__(self, dim: 'int', hidden_dim: 'int', multiple_of: 'int', ffn_dim_multiplier: 'Optional[float]', activation: 'nn.Module'=nn.SiLU()):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.activation = activation
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        return self.w2(self.activation(self.w1(x)))

    def init_weights(self, init_std: 'float'):
        nn.init.trunc_normal_(self.w1.weight, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.w2.weight, mean=0.0, std=init_std)


class TritonFusedRMSNorm(torch.autograd.Function):

    @partial(local_map, out_placements=[Shard(1)], in_placements=(None, [Shard(1)], [Replicate()], None))
    @staticmethod
    def forward(ctx, x, weight, eps):
        x_shape_start = x.shape
        x = x.view(-1, x.shape[-1])
        if x.stride(-1) != 1:
            x = x.contiguous()
        if weight.stride(-1) != 1:
            weight = weight.contiguous()
        M, N = x.shape
        y = torch.empty_like(x)
        rstd = torch.empty((M,), dtype=torch.float32, device=x.device)
        max_size = 65536 // x.element_size()
        block_N = min(max_size, triton.next_power_of_2(N))
        if N > block_N:
            raise ValueError(f'N {N} must be <= block_N={block_N!r}')
        grid = lambda meta: (M,)
        _rms_norm_fwd_kernel[grid](x, x.stride(0), y, y.stride(0), weight, rstd, eps, M, N, block_N)
        ctx.eps = eps
        ctx.save_for_backward(x, weight, rstd)
        ctx.x_shape_start = x_shape_start
        y = y.reshape(x_shape_start)
        return y

    @partial(local_map, out_placements=([Shard(1)], [Partial()], None), in_placements=(None, [Shard(1)]))
    @staticmethod
    def backward(ctx, dy):
        x, weight, rstd = ctx.saved_tensors
        eps = ctx.eps
        x_shape_start = ctx.x_shape_start
        dy = dy.view(-1, dy.shape[-1])
        if dy.stride(-1) != 1:
            dy = dy.contiguous()
        M, N = dy.shape
        dx = torch.empty_like(x)
        sm_count = torch.cuda.get_device_properties(x.device).multi_processor_count
        _dw = torch.empty((sm_count, N), dtype=torch.float32, device=weight.device)
        max_size = 65536 // x.element_size()
        block_N = min(max_size, triton.next_power_of_2(N))
        rows_per_sm = math.ceil(M / sm_count)
        if N > block_N:
            raise ValueError(f'N {N} must be <= block_N={block_N!r}')
        grid = lambda meta: (sm_count,)
        _rms_norm_bwd_kernel_sm[grid](x, x.stride(0), weight, dy, dy.stride(0), dx, dx.stride(0), rstd, _dw, eps, M, N, rows_per_sm, block_N)
        dw = _dw.sum(0)
        dx = dx.view(x_shape_start)
        return dx, dw, None


def fused_rms_norm_fn(x, weight, eps=1e-06):
    return TritonFusedRMSNorm.apply(x, weight, eps)


class FusedRMSNorm(nn.Module):
    """Fused RMS Norm, wraps a fused Triton Kernel"""

    def __init__(self, dim: 'int', eps: 'float'=1e-06):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.fused_rms_norm_fn = fused_rms_norm_fn

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        """leverages Triton Fused RMS Norm kernel"""
        return self.fused_rms_norm_fn(x, self.weight, eps=self.eps)

    def reset_parameters(self):
        torch.nn.init.ones_(self.weight)


class RMSNorm(nn.Module):
    """
    Initialize the RMSNorm normalization layer.

    Args:
        dim (int): The dimension of the input tensor.
        eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.

    Attributes:
        eps (float): A small value added to the denominator for numerical stability.
        weight (nn.Parameter): Learnable scaling parameter.

    """

    def __init__(self, dim: 'int', eps: 'float'=1e-06):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: 'torch.Tensor'):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: 'torch.Tensor'):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

    def reset_parameters(self):
        torch.nn.init.ones_(self.weight)


def build_norm(norm_type: 'str', dim: 'int', eps: 'float'=1e-06):
    """
    Builds the specified normalization layer based on the norm_type.

    Args:
        norm_type (str): The type of normalization layer to build.
            Supported types: layernorm, np_layernorm, rmsnorm, fused_rmsnorm
        dim (int): The dimension of the normalization layer.
        eps (float, optional): The epsilon value for numerical stability. Defaults to 1e-6.

    Returns:
        The built normalization layer.

    Raises:
        NotImplementedError: If an unknown norm_type is provided.
    """
    norm_type = norm_type.lower()
    if norm_type == 'layernorm':
        return nn.LayerNorm(dim, eps=eps, bias=False)
    elif norm_type == 'np_layernorm':
        return nn.LayerNorm(dim, eps=eps, elementwise_affine=False, bias=False)
    elif norm_type == 'rmsnorm':
        return RMSNorm(dim, eps=eps)
    elif norm_type == 'fused_rmsnorm':
        return FusedRMSNorm(dim, eps=eps)
    else:
        raise NotImplementedError(f"Unknown norm_type: '{norm_type}'")


class TransformerBlock(nn.Module):
    """
    TransformerBlock Module

    Args:
        layer_id (int): Identifier for the layer.
        model_args (ModelArgs): Model configuration arguments.

    Attributes:
        n_heads (int): Number of attention heads.
        dim (int): Dimension size of the model.
        head_dim (int): Dimension size of each attention head.
        attention (Attention): Attention module.
        feed_forward (FeedForward): FeedForward module.
        layer_id (int): Identifier for the layer.
        attention_norm (RMSNorm): Layer normalization for attention output.
        ffn_norm (RMSNorm): Layer normalization for feedforward output.

    """

    def __init__(self, layer_id: 'int', model_args: 'ModelArgs'):
        super().__init__()
        self.n_heads = model_args.n_heads
        self.dim = model_args.dim
        self.attention = Attention(model_args)
        self.feed_forward = FeedForward(dim=model_args.dim, hidden_dim=4 * model_args.dim, multiple_of=model_args.multiple_of, ffn_dim_multiplier=model_args.ffn_dim_multiplier)
        self.layer_id = layer_id
        self.num_layers = model_args.n_layers
        self.attention_norm = build_norm(model_args.norm_type, dim=model_args.dim, eps=model_args.norm_eps)
        self.ffn_norm = build_norm(model_args.norm_type, dim=model_args.dim, eps=model_args.norm_eps)
        if model_args.depth_init:
            self.weight_init_std = 0.02 / (2 * (self.layer_id + 1)) ** 0.5
        else:
            self.weight_init_std = 0.02 / (2 * self.num_layers) ** 0.5

    def forward(self, x: 'torch.Tensor', freqs_cis: 'torch.Tensor'):
        """
        Perform a forward pass through the TransformerBlock.

        Args:
            x (torch.Tensor): Input tensor.
            freqs_cis (torch.Tensor): Precomputed cosine and sine frequencies.

        Returns:
            torch.Tensor: Output tensor after applying attention and feedforward layers.

        """
        h = x + self.attention(self.attention_norm(x), freqs_cis)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out

    def init_weights(self):
        for norm in (self.attention_norm, self.ffn_norm):
            norm.reset_parameters()
        self.attention.init_weights(self.weight_init_std)
        self.feed_forward.init_weights(self.weight_init_std)


def precompute_freqs_cis(dim: 'int', end: 'int', theta: 'float'=10000.0) ->torch.Tensor:
    """
    Precompute the frequency tensor for complex exponentials (cis) with given dimensions.

    This function calculates a frequency tensor with complex exponentials using the given dimension 'dim'
    and the end index 'end'. The 'theta' parameter scales the frequencies.
    The returned tensor contains complex values in complex64 data type.

    Args:
        dim (int): Dimension of the frequency tensor.
        end (int): End index for precomputing frequencies.
        theta (float, optional): Scaling factor for frequency computation. Defaults to 10000.0.

    Returns:
        torch.Tensor: Precomputed frequency tensor with complex exponentials.
    """
    freqs = 1.0 / theta ** (torch.arange(0, dim, 2)[:dim // 2].float() / dim)
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


class Transformer(nn.Module):
    """
    Transformer Module

    Args:
        model_args (ModelArgs): Model configuration arguments.

    Attributes:
        model_args (ModelArgs): Model configuration arguments.
        vocab_size (int): Vocabulary size.
        n_layers (int): Number of layers in the model.
        tok_embeddings (ParallelEmbedding): Token embeddings.
        layers (torch.nn.ModuleList): List of Transformer blocks.
        norm (RMSNorm): Layer normalization for the model output.
        output (ColumnParallelLinear): Linear layer for final output.
        freqs_cis (torch.Tensor): Precomputed cosine and sine frequencies.

    """

    def __init__(self, model_args: 'ModelArgs'):
        super().__init__()
        self.model_args = model_args
        self.vocab_size = model_args.vocab_size
        self.n_layers = model_args.n_layers
        self.tok_embeddings = nn.Embedding(model_args.vocab_size, model_args.dim)
        self.register_buffer('freqs_cis', self._precompute_freqs_cis(), persistent=True)
        self.layers = torch.nn.ModuleDict()
        for layer_id in range(model_args.n_layers):
            self.layers[str(layer_id)] = TransformerBlock(layer_id, model_args)
        self.norm = build_norm(model_args.norm_type, dim=model_args.dim, eps=model_args.norm_eps)
        self.output = nn.Linear(model_args.dim, model_args.vocab_size, bias=False)
        self.init_weights()

    def init_weights(self, buffer_device: 'Optional[torch.device]'=None):
        """
        [Note: On ``init_weights`` vs. ``reset_parameters``]
        Modules may define ``reset_parameters`` to initialize parameter values.
        ``reset_parameters`` is meant to only initialize directly owned
        parameters/buffers, not those of their child modules, and it can be
        used to give the initial values for these tensors.
        Separately, users may want custom initialization for their modules,
        different from that in ``reset_parameters``. For this, we define
        ``init_weights``. We only call it in the constructor of this
        ``Transformer`` root module to avoid reinitializing tensors.
        """
        buffer_device = buffer_device or self.freqs_cis.device
        with torch.device(buffer_device):
            self.freqs_cis = self._precompute_freqs_cis()
        if self.tok_embeddings is not None:
            nn.init.normal_(self.tok_embeddings.weight)
        for layer in self.layers.values():
            if layer is not None:
                layer.init_weights()
        if self.norm is not None:
            self.norm.reset_parameters()
        final_out_std = self.model_args.dim ** -0.5
        cutoff_factor = 3
        if self.output is not None:
            nn.init.trunc_normal_(self.output.weight, mean=0.0, std=final_out_std, a=-cutoff_factor * final_out_std, b=cutoff_factor * final_out_std)

    def _precompute_freqs_cis(self) ->torch.Tensor:
        return precompute_freqs_cis(self.model_args.dim // self.model_args.n_heads, self.model_args.max_seq_len, self.model_args.rope_theta)

    def forward(self, tokens: 'torch.Tensor'):
        """
        Perform a forward pass through the Transformer model.

        Args:
            tokens (torch.Tensor): Input token indices.

        Returns:
            torch.Tensor: Output logits after applying the Transformer model.

        """
        h = self.tok_embeddings(tokens) if self.tok_embeddings else tokens
        for layer in self.layers.values():
            h = layer(h, self.freqs_cis)
        h = self.norm(h) if self.norm else h
        output = self.output(h) if self.output else h
        return output

    @classmethod
    def from_model_args(cls, model_args: 'ModelArgs') ->'Transformer':
        """
        Initialize a Transformer model from a ModelArgs object.

        Args:
            model_args (ModelArgs): Model configuration arguments.

        Returns:
            Transformer: Transformer model.

        """
        return cls(model_args)


class Fp32LayerNorm(nn.LayerNorm):
    """
    Wrapper around :class:`~torch.nn.LayerNorm` to support mixed-precision training.
    """

    def __init__(self, *args: Any, **kwargs: Any) ->None:
        super().__init__(*args, **kwargs)

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor.
        Returns:
            torch.Tensor: The normalized output tensor having the same shape as ``x``.
        """
        output = nn.functional.layer_norm(x.float(), self.normalized_shape, self.weight.float() if self.weight is not None else None, self.bias.float() if self.bias is not None else None, self.eps)
        return output.type_as(x)


class TanhGate(nn.Module):
    """Implements a basic learnable gate to scale layer outputs"""

    def __init__(self) ->None:
        super().__init__()
        self.scale = nn.Parameter(torch.zeros(1))

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        """
        Args:
            x (torch.Tensor): input tensor to gate

        Returns:
            torch.Tensor: The output tensor after gating. Has the same shape as ``x``.
        """
        return x * self.scale.tanh()


class TilePositionalEmbedding(nn.Module):
    """
    Positional embedding for tiles, different for every tile, same for every token within a tile.

    For details, please check the documentation of :class:`ViT`.

    Args:
        max_num_tiles (int): The maximum number of tiles an image can be divided into.
        emb_dim (int): The dimensionality of each tile embedding.
    """

    def __init__(self, max_num_tiles: 'int', emb_dim: 'int'):
        super().__init__()
        self.max_num_tiles = max_num_tiles
        self.emb_dim = emb_dim
        self.embedding = nn.Parameter(torch.randn(max_num_tiles, max_num_tiles, 1, emb_dim) / math.sqrt(emb_dim))
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, x: 'torch.Tensor', aspect_ratio: 'torch.Tensor'):
        """
        args:
            x (torch.Tensor): torch.Tensor with shape (bsz * num_imgs, num_tiles, num_tokens, emb_dim).
            aspect_ratio (torch.Tensor): torch.Tensor with shape (bsz * num_imgs, 2),
                representing the aspect ratio of the image before tile-cropping, e.g. (2,1).
        returns:
            torch.Tensor: The input tensor with added positional embeddings.
        """
        bsz_and_num_imgs, num_tiles, num_tokens, emb_dim = x.shape
        for batch_idx, (num_tiles_h, num_tiles_w) in enumerate(aspect_ratio):
            num_non_padded_tiles = int(num_tiles_h * num_tiles_w)
            pos_embed = self.embedding[:num_tiles_h, :num_tiles_w, :, :]
            pos_embed = pos_embed.reshape(num_non_padded_tiles, 1, self.emb_dim)
            x[batch_idx, :num_non_padded_tiles, :, :] += pos_embed * self.gate.tanh()
        return x


class TokenPositionalEmbedding(nn.Module):
    """
    Token positional embedding for images, different for every token in an image.

    Args:
        emb_dim (int): The dimensionality of each token embedding.
        tile_size (int): The size of your image tiles, if the image was tile-cropped in advance. Otherwise,
            the size of the input image. In this case, the function will consider your image as a single tile.
        patch_size (int): The size of each patch. Used to divide the tiles into patches.
            E.g. for ``patch_size=40``, a tile of shape (400, 400) will have 10x10 grid of patches
            with shape (40, 40) each.
    """

    def __init__(self, emb_dim: 'int', tile_size: 'int', patch_size: 'int') ->None:
        super().__init__()
        patch_grid_size = tile_size // patch_size
        scale = emb_dim ** -0.5
        self.positional_embedding = nn.Parameter(scale * torch.randn((patch_grid_size ** 2 + 1, emb_dim)))

    def forward(self, x: 'torch.Tensor', *args: Tuple[Any]) ->torch.Tensor:
        """
        Args:
            x (torch.Tensor): torch.Tensor with shape (..., num_tokens, emb_dim)
            *args (Tuple[Any]): Optional args.

        Returns:
            torch.Tensor: The input tensor with added positional embeddings.
        """
        return x + self.positional_embedding


class TiledTokenPositionalEmbedding(nn.Module):
    """

    Token positional embedding for tiled images. There are two positional embeddings in this module:

    * local_token_positional_embedding: same for every tile, different for every token. Equivalent         to :class:`TokenPositionalEmbedding`, but gated.
    * global_token_positional_embedding: different for every tile, different for every token.

    Notice that tile is different from patch (token). For details, please check the documentation of
    :class:`ViT`.

    Args:
        max_num_tiles (int): The maximum number of tiles an image can be divided into.
        emb_dim (int): The dimensionality of each token embedding.
        tile_size (int): The size of your image tiles, if the image was tile-cropped in advance. Otherwise,
            the size of the input image. In this case, the function will consider your image as a single tile.
        patch_size (int): The size of each patch. Used to divide the tiles into patches.
            E.g. for ``patch_size=40``, a tile of shape (400, 400) will have 10x10 grid of patches
            with shape (40, 40) each.
    """

    def __init__(self, max_num_tiles: 'int', emb_dim: 'int', tile_size: 'int', patch_size: 'int') ->None:
        super().__init__()
        patch_grid_size = tile_size // patch_size
        self.num_tokens_per_tile = patch_grid_size ** 2 + 1
        scale = emb_dim ** -0.5
        self.local_token_positional_embedding = nn.Parameter(scale * torch.randn((patch_grid_size ** 2 + 1, emb_dim)))
        self.global_token_positional_embedding = nn.Parameter(scale * torch.randn(max_num_tiles, max_num_tiles, self.num_tokens_per_tile, emb_dim))
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, x: 'torch.Tensor', aspect_ratio: 'torch.Tensor') ->torch.Tensor:
        """
        Args:
            x (torch.Tensor): torch.Tensor with shape (bsz * num_imgs, num_tiles, num_tokens, emb_dim).
            aspect_ratio (torch.Tensor): torch.Tensor with shape (bsz * num_imgs, 2),
                where aspect_ratio[k] represents the aspect ratio of the k^th image
                of the batch before tile-cropping,  e.g. aspect_ratio[k] = (2,1).
        Returns:
            torch.Tensor: The input tensor with added positional embeddings.
        """
        bsz_and_num_imgs, num_tiles, num_tokens, emb_dim = x.shape
        x = x + self.local_token_positional_embedding * (1 - self.gate.tanh())
        x = x.view(bsz_and_num_imgs, num_tiles, num_tokens, emb_dim)
        for batch_idx, (num_tiles_h, num_tiles_w) in enumerate(aspect_ratio):
            num_non_padded_tiles = int(num_tiles_h * num_tiles_w)
            pos_embed = self.global_token_positional_embedding[:num_tiles_h, :num_tiles_w, :, :]
            pos_embed = pos_embed.reshape(num_non_padded_tiles, self.num_tokens_per_tile, emb_dim)
            pos_embed = pos_embed * self.gate.tanh()
            x[batch_idx, :num_non_padded_tiles, :, :] += pos_embed
        return x


class Conv2dModule(torch.nn.Module):
    """Conv2D Module.
    This is like Conv2D in PyTorch except:

    - PyTorch Conv2D outputs shape (*, out_channels, h_out, w_out), while this module
      outputs (*, h_out * w_out, out_channels).
    - We implement the conv as an unfold -> permute -> linear, where we can column-wise
      shard the linear.

    Arguments:
        in_channels: Input channels.
        out_channels: Output channels.
        kernel_size: Size of convolution kernel. This module also assumes a square kernel.
        stride (default 1): Stride for convolution.
        bias (default False): Use bias in Conv2d.
    """

    def __init__(self, in_channels: 'int', out_channels: 'int', kernel_size: 'int', stride: 'int', bias: 'bool'=False) ->None:
        super().__init__()
        self._unfold = torch.nn.Unfold(kernel_size=(kernel_size, kernel_size), stride=stride)
        self._linear = torch.nn.Linear(in_channels * kernel_size * kernel_size, out_channels, bias=bias)

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        x = self._unfold(x)
        x = x.permute(0, 2, 1)
        return self._linear(x)


class VitTransformerBlock(nn.Module):

    def __init__(self, model_args: 'ModelArgs', attn_scale: 'Optional[nn.Module]'=None, mlp_scale: 'Optional[nn.Module]'=None):
        super().__init__()
        self.attn = Attention(model_args)
        self.ln_attn = Fp32LayerNorm(model_args.encoder_embed_dim, eps=1e-05)
        self.mlp = FeedForward(dim=model_args.encoder_embed_dim, hidden_dim=4 * model_args.encoder_embed_dim, multiple_of=model_args.multiple_of, ffn_dim_multiplier=model_args.ffn_dim_multiplier, activation=model_args.activation)
        self.ln_mlp = Fp32LayerNorm(model_args.encoder_embed_dim, eps=1e-05)
        self.attn_scale = attn_scale or nn.Identity()
        self.mlp_scale = mlp_scale or nn.Identity()

    def forward(self, x: 'torch.Tensor', mask: 'Optional[torch.Tensor]'=None):
        bsz, seq_len, emd_dim = x.shape
        x = x + self.attn_scale(self.attn(x=self.ln_attn(x), freqs_cis=None))
        x = x + self.mlp_scale(self.mlp(self.ln_mlp(x)))
        return x


class CLSEmbedding(nn.Module):
    """
    Adds a CLS token to every tile of an image in the beginning of each token.

    Args:
        emb_dim (int): The dimensionality of the input patch embedding.
    """

    def __init__(self, emb_dim: 'int') ->None:
        super().__init__()
        scale = emb_dim ** -0.5
        self.weight = nn.Parameter(scale * torch.randn(emb_dim))

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        bsz_and_num_imgs, num_tiles, _, emb_dim = x.shape
        cls_emb = self.weight.broadcast_to(bsz_and_num_imgs, num_tiles, 1, emb_dim)
        return torch.cat([cls_emb, x], dim=2)


class Vit(nn.Module):
    """
    Implementation of the ViT architecture (https://arxiv.org/abs/2010.11929),
    with support for tile-cropped images, outputting of hidden layers.

    (credit for the documentation below: `vision_transformer.py

    <https://github.com/pytorch/torchtune/blob/b4fea322189f16629264ee44826f2ac080e922ec/torchtune/modules/vision_transformer.py>`_).

    ViT is a transformer architecture that takes in images and outputs N embedded tokens that
    represent this image. Each image is divided into **patches** by a convolution.
    These patches are flattened and subsequently treated as **tokens** by the transformer.

    To further enhance the performance of ViT and avoid downscaling images, we support tile-cropped images,
    which are images divided into **tiles** during the preprocessing stage. For example, instead of
    downscaling an 800x400 image to fit 400x400, we may crop it into two 400x400 tiles,
    if the ``tile_size=400``.

    Each of these tiles is further broken down into patches by a convolution operation. For example, if
    your ``patch_size=40``, then each (400, 400) tile will become a grid of 10x10 patches, and your whole image will have
    num_tiles * n_tokens -> num_tiles * (10x10 patches + 1 CLS token) -> num_tiles * 101.

    Before the transformer layers, a CLS token is added to each tile as the first token.
    In transformers, a token called CLS is a special token that is added to the beginning of each sequence.
    This token can be used to represent the whole input, instead of using a pooling operation, for example.

    To help the model "see" the whole image, we use positional embeddings. If your image
    was tile-cropped, then you need to use tile positional embeddings:

    - token_pos_embedding (tiled): :class:`TiledTokenPositionalEmbedding`
    - pre_tile_pos_embed: :class:`TilePositionalEmbedding`
    - post_tile_pos_embed: :class:`TilePositionalEmbedding`

    Otherwise, pre and post tile_pos_embed should be None and all you need is a simple
    token positional embedding:

    - token_pos_embedding (not tiled): :class:`TokenPositionalEmbedding`

    All images will be considered as a stack of tiles, even if your image was not tile-cropped. In such cases,
    your image would be composed of a single tile.

    In summary:

    1) An image is broken down into tiles during preprocessing.
    2) In the ViT, the tiles will be broken down into patches.
    3) The patches will be flattened and transformed. We call them tokens, because that's how the transformer sees them.

    Image: shape (8x8)

    .. code-block:: text

        |  1 |  2 |  3 |  4 |  5 |  6 |  7 |  8 |
        |  9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 |
        | 17 | 18 | 19 | 20 | 21 | 22 | 23 | 24 |
        | 25 | 26 | 27 | 28 | 29 | 30 | 31 | 32 |
        | 33 | 34 | 35 | 36 | 37 | 38 | 39 | 40 |
        | 41 | 42 | 43 | 44 | 45 | 46 | 47 | 48 |
        | 49 | 50 | 51 | 52 | 53 | 54 | 55 | 56 |
        | 57 | 58 | 59 | 60 | 61 | 62 | 63 | 64 |

    Tiles: shape (4,4,4) # (num_tiles, tile_size, tile_size)

    .. code-block:: text

        |  1 |  2 |  3 |  4 |    |  5 |  6 |  7 |  8 |
        |  9 | 10 | 11 | 12 |    | 13 | 14 | 15 | 16 |
        | 17 | 18 | 19 | 20 |    | 21 | 22 | 23 | 24 |
        | 25 | 26 | 27 | 28 |    | 29 | 30 | 31 | 32 |

        | 33 | 34 | 35 | 36 |    | 37 | 38 | 39 | 40 |
        | 41 | 42 | 43 | 44 |    | 45 | 46 | 47 | 48 |
        | 49 | 50 | 51 | 52 |    | 53 | 54 | 55 | 56 |
        | 57 | 58 | 59 | 60 |    | 61 | 62 | 63 | 64 |

    Patches: shape (4,4,2,2) # (num_tiles, num_patches_per_tile, patch_size, patch_size)

    .. code-block:: text

        |  1 |  2 |    |  3 |  4 |    |  5 |  6 |    |  7 |  8 |
        |  9 | 10 |    | 11 | 12 |    | 13 | 14 |    | 15 | 16 |

        | 17 | 18 |    | 19 | 20 |    | 21 | 22 |    | 23 | 24 |
        | 25 | 26 |    | 27 | 28 |    | 29 | 30 |    | 31 | 32 |

        | 33 | 34 |    | 35 | 36 |    | 37 | 38 |    | 39 | 40 |
        | 41 | 42 |    | 43 | 44 |    | 45 | 46 |    | 47 | 48 |

        | 49 | 50 |    | 51 | 52 |    | 53 | 54 |    | 55 | 56 |
        | 57 | 58 |    | 59 | 60 |    | 61 | 62 |    | 63 | 64 |

    token: shape (4, 4, 4) # (num_tiles, num_patches_per_tile, emb_dim)

    .. code-block:: text

        |  1 |  2 |  9 |  10 |    |  3 |  4 |  11 |  12 |    |  17 |  18 |  25 |  26 |    | 19 | 20 |  27 |  28 |
        | ... continuation of data ...
        | ... continuation of data ...
        | 37 | 38 | 45 |  46 |    | 39 |  40 | 47 |  48 |    | 53 | 54 |  61 |  62 |    | 55 | 56 |  63 |  64 |

    For the positional embeddings:

    Same for every tile, different for every token.

    - :class:`TokenPositionalEmbedding`

    .. code-block:: text

        |  1 |  2 |  3 |  4 |    |  1 |  2 |  3 |  4 |
        |  9 | 10 | 11 | 12 |    |  9 | 10 | 11 | 12 |
        | 17 | 18 | 19 | 20 |    | 17 | 18 | 19 | 20 |
        | 25 | 26 | 27 | 28 |    | 25 | 26 | 27 | 28 |

        |  1 |  2 |  3 |  4 |    |  1 |  2 |  3 |  4 |
        |  9 | 10 | 11 | 12 |    |  9 | 10 | 11 | 12 |
        | 17 | 18 | 19 | 20 |    | 17 | 18 | 19 | 20 |
        | 25 | 26 | 27 | 28 |    | 25 | 26 | 27 | 28 |

    Different for every tile, different for every token.

    - :class:`TiledTokenPositionalEmbedding`

    .. code-block:: text

        |  1 |  2 |    |  3 |  4 |    |  5 |  6 |    |  7 |  8 |
        |  9 | 10 |    | 11 | 12 |    | 13 | 14 |    | 15 | 16 |

        | 17 | 18 |    | 19 | 20 |    | 21 | 22 |    | 23 | 24 |
        | 25 | 26 |    | 27 | 28 |    | 29 | 30 |    | 31 | 32 |

        | 33 | 34 |    | 35 | 36 |    | 37 | 38 |    | 39 | 40 |
        | 41 | 42 |    | 43 | 44 |    | 45 | 46 |    | 47 | 48 |

        | 49 | 50 |    | 51 | 52 |    | 53 | 54 |    | 55 | 56 |
        | 57 | 58 |    | 59 | 60 |    | 61 | 62 |    | 63 | 64 |

    different for every tile, same for every token within a tile.

    - :class:`TilePositionalEmbedding`

    .. code-block:: text

        |  1 |  1 |  1 |  1 |    |  2 |  2 |  2 |  3 |
        |  1 |  1 |  1 |  1 |    |  2 |  2 |  2 |  3 |
        |  1 |  1 |  1 |  1 |    |  2 |  2 |  2 |  3 |
        |  1 |  1 |  1 |  1 |    |  2 |  2 |  2 |  3 |

        |  3 |  3 |  3 |  3 |    |  4 |  4 |  4 |  4 |
        |  3 |  3 |  3 |  3 |    |  4 |  4 |  4 |  4 |
        |  3 |  3 |  3 |  3 |    |  4 |  4 |  4 |  4 |
        |  3 |  3 |  3 |  3 |    |  4 |  4 |  4 |  4 |

    Args:
        model_args (ModelArgs): The model args.

    Raises:
        ValueError: If `patch_size` is not greater than 0.
        ValueError: If `len(return_intermediates)` is greater than `num_layers`.
    """

    def __init__(self, model_args: 'ModelArgs'):
        super().__init__()
        if model_args.patch_size <= 0:
            raise ValueError(f'kernel size of conv {model_args.patch_size} must be > 0')
        if model_args.return_intermediates and len(model_args.return_intermediates) > model_args.encoder_num_layers:
            raise ValueError(f'len(return_intermediates) must be <= num_layers. Got model_args.return_intermediate={model_args.return_intermediate!r} and model_args.encoder_num_layers={model_args.encoder_num_layers!r}')
        patch_grid_size = model_args.tile_size // model_args.patch_size
        self.patches_per_tile = patch_grid_size ** 2
        self.return_intermediates = model_args.return_intermediates
        self.conv = Conv2dModule(in_channels=model_args.in_channels, out_channels=model_args.encoder_embed_dim, kernel_size=model_args.patch_size, stride=model_args.patch_size, bias=False)
        self.ln_post = Fp32LayerNorm(model_args.encoder_embed_dim)
        self.ln_pre = Fp32LayerNorm(model_args.encoder_embed_dim)
        self.transformer_layers = nn.ModuleList([VitTransformerBlock(model_args) for _ in range(model_args.encoder_num_layers)])
        self.class_embedding = CLSEmbedding(model_args.encoder_embed_dim)
        if model_args.max_num_tiles > 1:
            self.pre_tile_pos_embed = TilePositionalEmbedding(max_num_tiles=model_args.max_num_tiles, emb_dim=model_args.encoder_embed_dim)
            self.post_tile_pos_embed = TilePositionalEmbedding(max_num_tiles=model_args.max_num_tiles, emb_dim=model_args.encoder_embed_dim)
            self.token_pos_embedding = TokenPositionalEmbedding(emb_dim=model_args.encoder_embed_dim, tile_size=model_args.tile_size, patch_size=model_args.patch_size)
        else:
            self.pre_tile_pos_embed = None
            self.post_tile_pos_embed = None
            self.token_pos_embedding = TiledTokenPositionalEmbedding(max_num_tiles=model_args.max_num_tiles, emb_dim=model_args.encoder_embed_dim, tile_size=model_args.tile_size, patch_size=model_args.patch_size)

    def forward(self, images: 'torch.Tensor', aspect_ratio: 'Optional[torch.Tensor]'=None) ->torch.Tensor:
        """
        Processes images and returns the tokens and hidden states.

        Multiple images per sample: we add a dimension num_imgs to the input. This is useful when a single
        sample constains multiple images, for example:

        - sample 1: "<image> what animal is this?"
        - sample 2: "I like <image> more than <image>"

        In this case, sample 1 has one image, and sample 2 has two images. max_n_imgs = max(2,1) = 2.
        So your input should have shape (bsz=2, num_imgs=2, num_tiles, num_channels, tile_size_w, tile_size_h).

        Notice that to batch it, you will have to pad num_imgs to max_num_imgs and max_num_tiles.

        Args:
            images (torch.Tensor): torch.Tensor with shape (bsz, num_imgs, num_tiles, num_channels, tile_size_w, tile_size_h).
            aspect_ratio (Optional[torch.Tensor]): torch.Tensor with shape (bsz, n_imgs, 2). If all
                images have a single tile, i.e. they were not tile-cropped, it should be None.
                Used to calculate the positional embeddings for the tiles.

        Returns:
            Tuple[torch.Tensor, List[torch.Tensor]]: A tuple: (x, hidden_states),
                where x is a torch.tensor of shape (bsz, num_imgs, num_tiles, num_tokens, emb_dim) and
                hidden_states has shape is a list of len(out_indices) torch.tensor with shape
                (bsz, num_imgs, num_tiles, num_tokens, emb_dim).

        Raises:
            ValueError: If aspect_ratio is None, but num_tiles > 1 in the batch.
        """
        bsz, num_imgs, num_tiles, num_channels, width, height = images.shape
        if aspect_ratio is None:
            aspect_ratio = torch.ones((bsz * num_imgs, 2), dtype=torch.int)
            if num_tiles > 1:
                raise ValueError(f'aspect_ratio was not provided, but found num_tiles > 1 for images.shape={images.shape!r}. Please provide aspect_ratio.')
        aspect_ratio = aspect_ratio.reshape(bsz * num_imgs, 2)
        images = images.view(bsz * num_imgs * num_tiles, num_channels, width, height)
        x = self.conv(images)
        _, num_tokens, emb_dim = x.shape
        x = x.reshape(bsz * num_imgs, num_tiles, num_tokens, emb_dim)
        if self.pre_tile_pos_embed:
            x = self.pre_tile_pos_embed(x, aspect_ratio)
        x = self.class_embedding(x)
        num_tokens += 1
        x = self.token_pos_embedding(x, aspect_ratio)
        x = self.ln_pre(x)
        x = x.view(bsz * num_imgs, -1, emb_dim)
        int_x = []
        for layer_idx, transformer_layer in enumerate(self.transformer_layers):
            if layer_idx in self.return_intermediates:
                h = x.view(bsz, num_imgs, num_tiles, num_tokens, emb_dim)
                int_x.append(h)
            x = transformer_layer(x)
        x = self.ln_post(x)
        x = x.view(bsz * num_imgs, num_tiles, num_tokens, emb_dim)
        if self.post_tile_pos_embed:
            x = self.post_tile_pos_embed(x, aspect_ratio)
        x = x.view(bsz, num_imgs, num_tiles, num_tokens, emb_dim)
        return x, int_x


class Projection(nn.Module):
    """Projection transformer to adapt the output of a
    encoder (CLIP) to the decoder model.
    """

    def __init__(self, model_args: 'ModelArgs') ->None:
        super().__init__()
        self.transformer_layers = nn.ModuleList([VitTransformerBlock(model_args, attn_scale=TanhGate(), mlp_scale=TanhGate()) for _ in range(model_args.num_layers_projection)])
        self.num_hidden = len(model_args.return_intermediates or [])
        self.output = nn.Linear(model_args.encoder_embed_dim * (self.num_hidden + 1), model_args.decoder_embed_dim)

    def forward(self, x: 'torch.Tensor', hidden_states: 'Optional[List[torch.Tensor]]'=None) ->torch.Tensor:
        bsz, num_imgs, num_tiles, num_tokens, emb_dim = x.shape
        x = x.view(bsz * num_imgs, num_tiles * num_tokens, emb_dim)
        for layer in self.transformer_layers:
            x = layer(x)
        x = x.view(bsz, num_imgs, num_tiles, num_tokens, emb_dim)
        if self.num_hidden > 0:
            assert hidden_states is not None
            hidden_states = torch.stack(hidden_states, dim=-1)
            hidden_states = hidden_states.view(bsz, num_imgs, num_tiles, num_tokens, -1)
            x = torch.cat([x, hidden_states], dim=-1)
        return self.output(x).reshape(bsz, num_imgs * num_tiles * num_tokens, -1)


class VisionEncoder(nn.Module):
    """Vision encoder model for Llama 3.2 Vision. This combines a vision
    encoder with a projection. We define two different components.

    Args:
        model_args (ModelArgs): configs for the vision encoder.
    """

    def __init__(self, model_args: 'ModelArgs') ->None:
        super().__init__()
        self.vit = Vit(model_args)
        self.proj = Projection(model_args)

    def forward(self, images: 'torch.Tensor', aspect_ratio: 'Optional[torch.Tensor]'=None) ->torch.Tensor:
        """
        Args:
            images (torch.Tensor):
                Image tensor with shape [bsz x num_imgs x num_tiles x num_channels x width x height].
            aspect_ratio (Optional[torch.Tensor]): Tensor with shape [bsz x num_imgs x 2]. If all
                images have a single tile, i.e. they were not tile-cropped, it should be None.
                Used to calculate the positional embeddings for the tiles.
        Returns:
            Tensor: output tensor of a sequence of embedings [bsz x seq_len x decoder_emb_dim]
                where sequence length is num_imgs*num_tiles+num_embeds
        """
        return self.proj(*self.vit(images, aspect_ratio))


class FeedForwardForDecoder(nn.Module):
    """
    FeedForward module for the decoder. It's different from the one in the encoder.
    This is the component which is orignally used in llama3.
    """

    def __init__(self, dim: 'int', hidden_dim: 'int', multiple_of: 'int', ffn_dim_multiplier: 'Optional[float]'):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

    def init_weights(self, init_std: 'float'):
        nn.init.trunc_normal_(self.w1.weight, mean=0.0, std=0.02)
        for linear in (self.w2, self.w3):
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=init_std)


class SelfAttention(nn.Module):
    """
    Multi-head self attention module with rotary position.
    """

    def __init__(self, model_args: 'ModelArgs'):
        super().__init__()
        self.num_heads = model_args.decoder_num_heads
        self.num_kv_heads = model_args.decoder_num_heads if model_args.decoder_num_kv_heads is None else model_args.decoder_num_kv_heads
        self.n_rep = self.num_heads // self.num_kv_heads
        self.head_dim = model_args.decoder_embed_dim // model_args.decoder_num_heads
        self.wq = nn.Linear(model_args.decoder_embed_dim, model_args.decoder_num_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(model_args.decoder_embed_dim, self.num_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(model_args.decoder_embed_dim, self.num_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(model_args.decoder_num_heads * self.head_dim, model_args.decoder_embed_dim, bias=False)

    def init_weights(self, init_std: 'float'):
        for linear in (self.wq, self.wk, self.wv):
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.wo.weight, mean=0.0, std=init_std)

    def forward(self, x: 'torch.Tensor', freqs_cis: 'torch.Tensor'):
        bs, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = xq.view(bs, seqlen, -1, self.head_dim)
        xk = xk.view(bs, seqlen, -1, self.head_dim)
        xv = xv.view(bs, seqlen, -1, self.head_dim)
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)
        keys = repeat_kv(xk, self.n_rep)
        values = repeat_kv(xv, self.n_rep)
        xq = xq.transpose(1, 2)
        xk = keys.transpose(1, 2)
        xv = values.transpose(1, 2)
        output = F.scaled_dot_product_attention(xq, xk, xv, is_causal=True)
        output = output.transpose(1, 2).contiguous()
        output = output.view(bs, seqlen, -1)
        return self.wo(output)


class CrossAttention(nn.Module):
    """
    Multi-head cross attention module.
    """

    def __init__(self, model_args: 'ModelArgs'):
        super().__init__()
        self.num_heads = model_args.decoder_num_heads
        self.num_kv_heads = model_args.decoder_num_heads if model_args.decoder_num_kv_heads is None else model_args.decoder_num_kv_heads
        self.n_rep = self.num_heads // self.num_kv_heads
        self.head_dim = model_args.decoder_embed_dim // model_args.decoder_num_heads
        self.wq = nn.Linear(model_args.decoder_embed_dim, model_args.decoder_num_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(model_args.decoder_embed_dim, self.num_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(model_args.decoder_embed_dim, self.num_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(model_args.decoder_num_heads * self.head_dim, model_args.decoder_embed_dim, bias=False)
        self.q_norm = RMSNorm(dim=self.head_dim, eps=1e-05)
        self.k_norm = RMSNorm(dim=self.head_dim, eps=1e-05)

    def init_weights(self, init_std: 'float'):
        for linear in (self.wq, self.wk, self.wv):
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.wo.weight, mean=0.0, std=init_std)

    def forward(self, x: 'torch.Tensor', encoder_input: 'torch.Tensor', mask: 'Optional[torch.Tensor]'=None):
        bs, seqlen_x, _ = x.shape
        seqlen_y = encoder_input.shape[1]
        xq, xk, xv = self.wq(x), self.wk(encoder_input), self.wv(encoder_input)
        xq = xq.view(bs, seqlen_x, -1, self.head_dim)
        xk = xk.view(bs, seqlen_y, -1, self.head_dim)
        xv = xv.view(bs, seqlen_y, -1, self.head_dim)
        keys = repeat_kv(xk, self.n_rep)
        values = repeat_kv(xv, self.n_rep)
        xq = xq.transpose(1, 2)
        xk = keys.transpose(1, 2)
        xv = values.transpose(1, 2)
        xq = self.q_norm(xq)
        xk = self.k_norm(xk)
        output = F.scaled_dot_product_attention(xq, xk, xv, attn_mask=mask, is_causal=False)
        output = output.transpose(1, 2).contiguous()
        output = output.view(bs, seqlen_x, -1)
        return self.wo(output)


class DecoderTransformerSelfAttnBlock(nn.Module):

    def __init__(self, model_args: 'ModelArgs'):
        super().__init__()
        self.attn = SelfAttention(model_args)
        self.ln_attn = RMSNorm(dim=model_args.decoder_embed_dim, eps=1e-05)
        self.mlp = FeedForwardForDecoder(dim=model_args.decoder_embed_dim, hidden_dim=4 * model_args.decoder_embed_dim, multiple_of=model_args.multiple_of, ffn_dim_multiplier=model_args.ffn_dim_multiplier)
        self.ln_mlp = RMSNorm(dim=model_args.decoder_embed_dim, eps=1e-05)

    def forward(self, x: 'torch.Tensor', freqs_cis: 'torch.Tensor', **kwargs: Dict):
        bsz, seq_len, emd_dim = x.shape
        x = x + self.attn(self.ln_attn(x), freqs_cis)
        x = x + self.mlp(self.ln_mlp(x))
        return x


class DecoderTransformerCrossAttnBlock(nn.Module):

    def __init__(self, model_args: 'ModelArgs'):
        super().__init__()
        self.attn = CrossAttention(model_args)
        self.ln_attn = RMSNorm(dim=model_args.decoder_embed_dim)
        self.mlp = FeedForward(dim=model_args.decoder_embed_dim, hidden_dim=4 * model_args.decoder_embed_dim, multiple_of=model_args.multiple_of, ffn_dim_multiplier=model_args.ffn_dim_multiplier)
        self.ln_mlp = RMSNorm(dim=model_args.decoder_embed_dim)
        self.attn_scale = TanhGate()
        self.mlp_scale = TanhGate()

    def _skip_mask(self, mask: 'Optional[torch.Tensor]') ->Optional[torch.Tensor]:
        """Some tokens in x may not attend to any encoder inputs
        due to the cross attention mask (encoder_mask). This results in
        a full row of the attention matrix being masked out.

        In the example below, the word "the" is masked from every embedding.
        The False value means a token can't attend to an embedding.

        .. code-block:: text

            |emb||emb||emb|
        |The| F    F    F
        |red| T    F    T
        |car| F    T    T

        This results in no inputs into the softmax layer which causes a NaN.
        The skip mask is used to mask the outputs of attention and
        mlp resulting in the token being skipped.

        The above example would result in a skip mask of: [[True], [False], [False]]
        which specifies which tokens to fully mask out.

        """
        if mask is None:
            return None
        if mask.dtype == torch.bool:
            mask = ~mask
        else:
            mask = torch.isneginf(mask)
        mask = torch.all(mask, dim=-1, keepdim=True)
        return mask

    def forward(self, x: 'torch.Tensor', *, encoder_input: Optional[torch.Tensor]=None, encoder_mask: Optional[torch.Tensor]=None, **kwargs: Dict) ->torch.Tensor:
        if encoder_input is None:
            return x
        skip_mask = self._skip_mask(encoder_mask)
        attn_out = self.attn(self.ln_attn(x), encoder_input, mask=encoder_mask)
        if skip_mask is not None:
            attn_out.masked_fill_(skip_mask, 0)
        h = self.attn_scale(attn_out) + x
        mlp_out = self.mlp(self.ln_mlp(h))
        if skip_mask is not None:
            mlp_out.masked_fill_(skip_mask, 0)
        out = h + self.mlp_scale(mlp_out)
        return out


class FusionLayer(nn.Module):
    """
    Deep Fusion model architectures combine pretrained encoder models with pretrained
    language models by infusing the encoder outputs into the middle layers of the LLM.
    This allows the language model to interpret the enocder outputs as text and
    "understand" any modality for which you can train an decoder. To enable the language model
    to adapt to the encoder outputs, the FusionLayer fuses a new learnable layer to an existing
    decoder (language model) layer. This additional layer can take the encoder embeddings and
    learn to combine them with the token embeddings from the decoder.
    """

    def __init__(self, layer: 'nn.Module', fusion_layer: 'nn.Module', fusion_first: 'bool'=True):
        super().__init__()
        self.layer = layer
        self.fusion_layer = fusion_layer

    def forward(self, x: 'torch.Tensor', **kwargs: Dict) ->torch.Tensor:
        x = self.fusion_layer(x, **kwargs)
        x = self.layer(x, **kwargs)
        return x


class FusionEmbedding(nn.Module):
    """
    Fusion embedding supports training additional special tokens while keeping
    the original embedding frozen. When fusing new models with a language model,
    there may be some additional tokens needed to support the fused language model. For
    example, adding a vision encoder might necessitate additional tokens like ``<|image|>``
    to indicate an images position in text and require learning an embedding for this token.
    The FusionEmbedding keeps the original embeddings frozen while learning a much smaller
    second embedding for the additional tokens. During forward this module routes
    the tokens to the appropriate embedding table.
    """

    def __init__(self, vocab_size: 'int', fusion_vocab_size: 'int', embed_dim: 'int') ->None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.fusion_embedding = nn.Embedding(fusion_vocab_size, embed_dim)
        self.dim = embed_dim
        self.num_embeddings = vocab_size + fusion_vocab_size

    def forward(self, input: 'torch.Tensor') ->torch.Tensor:
        bsz, seq_len = input.size()
        vocab_size = self.embedding.num_embeddings
        mask = input < vocab_size
        tokens = torch.masked_select(input, mask)
        fusion_tokens = torch.masked_select(input, ~mask) - vocab_size
        embeds = self.embedding(tokens)
        fusion_embeds = self.fusion_embedding(fusion_tokens)
        out = torch.empty(bsz, seq_len, self.dim, device=self.embedding.weight.device, dtype=self.embedding.weight.dtype)
        mask = mask.unsqueeze(-1).expand(bsz, seq_len, self.dim)
        out.masked_scatter_(mask, embeds)
        out.masked_scatter_(~mask, fusion_embeds)
        return out


class MultimodalDecoder(nn.Module):
    """Decoder multimodal model for Llama 3.2.

    Args:
        model_args (ModelArgs): configs for the vision encoder.
    """

    def __init__(self, model_args: 'ModelArgs'):
        super().__init__()
        self.register_buffer('freqs_cis', self._precompute_freqs_cis(model_args), persistent=True)
        self.layers = []
        for idx in range(1, model_args.decoder_num_layers + 1):
            decoder_layer = DecoderTransformerSelfAttnBlock(model_args)
            if idx % model_args.fusion_interval == 0:
                cross_attn_layer = DecoderTransformerCrossAttnBlock(model_args)
                fusion_layer = FusionLayer(layer=decoder_layer, fusion_layer=cross_attn_layer)
                self.layers.append(fusion_layer)
            else:
                self.layers.append(decoder_layer)
        self.tok_embeddings = FusionEmbedding(model_args.vocab_size, model_args.num_special_tokens, model_args.decoder_embed_dim)
        self.norm = RMSNorm(model_args.decoder_embed_dim, eps=1e-05)
        self.output = nn.Linear(model_args.decoder_embed_dim, model_args.vocab_size, bias=False)

    def _precompute_freqs_cis(self, model_args) ->torch.Tensor:
        return precompute_freqs_cis(model_args.decoder_embed_dim // model_args.decoder_num_heads, model_args.max_seq_len * 2, model_args.rope_theta)

    def forward(self, tokens: 'torch.Tensor', *, encoder_input: Optional[torch.Tensor]=None, encoder_mask: Optional[torch.Tensor]=None) ->torch.Tensor:
        """
        Args:
            tokens (torch.Tensor): input tensor with shape ``[b x s]``
            encoder_input (Optional[torch.Tensor]): Optional input embeds from the encoder. Shape ``[b x s_e x d_e]``
            encoder_mask (Optional[torch.Tensor]):  Boolean tensor defining a relational matrix between
                tokens and encoder embeddings. A True value at position ``i,j`` means token ``i`` can attend
                to embedding ``j`` in the decoder. Mask has shape ``[b x s x s_e]``. Default is None,
                but this is required during inference if the model has been setup with any layers
                which use encoder embeddings and caches have been setup.
        """
        bsz, seq_len = tokens.shape
        h = self.tok_embeddings(tokens)
        for layer in self.layers:
            h = layer(h, freqs_cis=self.freqs_cis, encoder_input=encoder_input, encoder_mask=encoder_mask)
        h = self.norm(h)
        output = self.output(h).float()
        return output


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (CLSEmbedding,
     lambda: ([], {'emb_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Conv2dModule,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4, 'stride': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (CrossAttention,
     lambda: ([], {'model_args': SimpleNamespace(decoder_num_heads=4, decoder_num_kv_heads=4, decoder_embed_dim=4)}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4])], {})),
    (DecoderTransformerCrossAttnBlock,
     lambda: ([], {'model_args': SimpleNamespace(decoder_num_heads=4, decoder_num_kv_heads=4, decoder_embed_dim=4, multiple_of=4, ffn_dim_multiplier=4)}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (FeedForward,
     lambda: ([], {'dim': 4, 'hidden_dim': 4, 'multiple_of': 4, 'ffn_dim_multiplier': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (FeedForwardForDecoder,
     lambda: ([], {'dim': 4, 'hidden_dim': 4, 'multiple_of': 4, 'ffn_dim_multiplier': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Fp32LayerNorm,
     lambda: ([], {'normalized_shape': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (FusionLayer,
     lambda: ([], {'layer': torch.nn.ReLU(), 'fusion_layer': torch.nn.ReLU()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (RMSNorm,
     lambda: ([], {'dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (TanhGate,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (TokenPositionalEmbedding,
     lambda: ([], {'emb_dim': 4, 'tile_size': 4, 'patch_size': 4}),
     lambda: ([torch.rand([4, 4, 2, 4])], {})),
]

