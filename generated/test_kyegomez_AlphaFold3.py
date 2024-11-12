
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


from torch import Tensor


import torch.nn.functional as F


from inspect import isfunction


from typing import Optional


from typing import Tuple


from torch import einsum


from torch.utils.checkpoint import checkpoint_sequential


import torch.nn as nn


class GeneticDiffusionModuleBlock(nn.Module):
    """
    Diffusion Module from AlphaFold 3.

    This module directly predicts raw atom coordinates via a generative diffusion process.
    It leverages a diffusion model trained to denoise 'noised' atomic coordinates back to their
    true state. The diffusion process captures both local and global structural information
    through a series of noise scales.

    Attributes:
        channels (int): The number of channels in the input feature map, corresponding to atomic features.
        num_diffusion_steps (int): The number of diffusion steps or noise levels to use.
    """

    def __init__(self, channels: 'int', num_diffusion_steps: 'int'=1000, training: 'bool'=False, depth: 'int'=30):
        """
        Initializes the DiffusionModule with the specified number of channels and diffusion steps.

        Args:
            channels (int): Number of feature channels for the input.
            num_diffusion_steps (int): Number of diffusion steps (time steps in the diffusion process).
        """
        super(GeneticDiffusionModuleBlock, self).__init__()
        self.channels = channels
        self.num_diffusion_steps = num_diffusion_steps
        self.training = training
        self.depth = depth
        self.noise_scale = nn.Parameter(torch.linspace(1.0, 0.01, num_diffusion_steps))
        self.prediction_network = nn.Sequential(nn.Linear(channels, channels * 2), nn.ReLU(), nn.Linear(channels * 2, channels))

    def forward(self, x: 'Tensor'=None, ground_truth: 'Tensor'=None):
        """
        Forward pass of the DiffusionModule. Applies a sequence of noise and denoising operations to
        the input coordinates to simulate the diffusion process.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_atoms, channels)
                            representing the atomic features including coordinates.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_atoms, channels) with
                        denoised atom coordinates.
        """
        batch_size, num_nodes, num_nodes_two, num_features = x.shape
        noisy_x = x.clone()
        for step in range(self.num_diffusion_steps):
            noise_level = self.noise_scale[step]
            noise = torch.randn_like(x) * noise_level
            noisy_x = x + noise
            noisy_x = self.prediction_network(noisy_x)
        if self.training and ground_truth is not None:
            loss = F.mse_loss(noisy_x, ground_truth)
            return noisy_x, loss
        return noisy_x


class GeneticDiffusion(nn.Module):
    """
    GeneticDiffusion module for performing genetic diffusion.

    Args:
        channels (int): Number of input channels.
        num_diffusion_steps (int): Number of diffusion steps to perform.
        training (bool): Whether the module is in training mode or not.
        depth (int): Number of diffusion module blocks to stack.

    Attributes:
        channels (int): Number of input channels.
        num_diffusion_steps (int): Number of diffusion steps to perform.
        training (bool): Whether the module is in training mode or not.
        depth (int): Number of diffusion module blocks to stack.
        layers (nn.ModuleList): List of GeneticDiffusionModuleBlock instances.

    """

    def __init__(self, channels: 'int', num_diffusion_steps: 'int'=1000, training: 'bool'=False, depth: 'int'=30):
        super(GeneticDiffusion, self).__init__()
        self.channels = channels
        self.num_diffusion_steps = num_diffusion_steps
        self.training = training
        self.depth = depth
        self.layers = nn.ModuleList([GeneticDiffusionModuleBlock(channels, num_diffusion_steps, training, depth) for _ in range(depth)])

    def forward(self, x: 'Tensor'=None, ground_truth: 'Tensor'=None):
        """
        Forward pass of the GeneticDiffusion module.

        Args:
            x (Tensor): Input tensor.
            ground_truth (Tensor): Ground truth tensor.

        Returns:
            Tuple[Tensor, Tensor]: Output tensor and loss tensor.

        """
        if ground_truth is True:
            for layer in self.layers:
                x = layer(x, ground_truth)
            return x
        else:
            for layer in self.layers:
                x = layer(x)
            return x


class GEGLU(nn.Module):

    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)


def exists(val):
    return val is not None


def init_zero_(layer):
    nn.init.constant_(layer.weight, 0.0)
    if exists(layer.bias):
        nn.init.constant_(layer.bias, 0.0)


class FeedForward(nn.Module):

    def __init__(self, dim, mult=4, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.net = nn.Sequential(nn.Linear(dim, dim * mult * 2), GEGLU(), nn.Dropout(dropout), nn.Linear(dim * mult, dim))
        init_zero_(self.net[-1])

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.net(x)


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


class Attention(nn.Module):

    def __init__(self, dim, seq_len=None, heads=8, dim_head=64, dropout=0.0, gating=True):
        super().__init__()
        inner_dim = dim_head * heads
        self.seq_len = seq_len
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)
        self.gating = nn.Linear(dim, inner_dim)
        nn.init.constant_(self.gating.weight, 0.0)
        nn.init.constant_(self.gating.bias, 1.0)
        self.dropout = nn.Dropout(dropout)
        init_zero_(self.to_out)

    def forward(self, x, mask=None, attn_bias=None, context=None, context_mask=None, tie_dim=None):
        device, orig_shape, h, has_context = x.device, x.shape, self.heads, exists(context)
        context = default(context, x)
        q, k, v = self.to_q(x), *self.to_kv(context).chunk(2, dim=-1)
        i, j = q.shape[-2], k.shape[-2]
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))
        q = q * self.scale
        if exists(tie_dim):
            q, k = map(lambda t: rearrange(t, '(b r) ... -> b r ...', r=tie_dim), (q, k))
            q = q.mean(dim=1)
            dots = einsum('b h i d, b r h j d -> b r h i j', q, k)
            dots = rearrange(dots, 'b r ... -> (b r) ...')
        else:
            dots = einsum('b h i d, b h j d -> b h i j', q, k)
        if exists(attn_bias):
            dots = dots + attn_bias
        if exists(mask):
            mask = default(mask, lambda : torch.ones(1, i, device=device).bool())
            context_mask = mask if not has_context else default(context_mask, lambda : torch.ones(1, k.shape[-2], device=device).bool())
            mask_value = -torch.finfo(dots.dtype).max
            mask = mask[:, None, :, None] * context_mask[:, None, None, :]
            dots = dots.masked_fill(~mask, mask_value)
        attn = dots.softmax(dim=-1)
        attn = self.dropout(attn)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        gates = self.gating(x)
        out = out * gates.sigmoid()
        out = self.to_out(out)
        return out


class AxialAttention(nn.Module):

    def __init__(self, dim: 'int', heads: 'int', row_attn: 'bool'=True, col_attn: 'bool'=True, accept_edges: 'bool'=False, global_query_attn: 'bool'=False, **kwargs):
        """
        Axial Attention module.

        Args:
            dim (int): The input dimension.
            heads (int): The number of attention heads.
            row_attn (bool, optional): Whether to perform row attention. Defaults to True.
            col_attn (bool, optional): Whether to perform column attention. Defaults to True.
            accept_edges (bool, optional): Whether to accept edges for attention bias. Defaults to False.
            global_query_attn (bool, optional): Whether to perform global query attention. Defaults to False.
            **kwargs: Additional keyword arguments for the Attention module.
        """
        super().__init__()
        assert not (not row_attn and not col_attn), 'row or column attention must be turned on'
        self.row_attn = row_attn
        self.col_attn = col_attn
        self.global_query_attn = global_query_attn
        self.norm = nn.LayerNorm(dim)
        self.attn = Attention(dim=dim, heads=heads, **kwargs)
        self.edges_to_attn_bias = nn.Sequential(nn.Linear(dim, heads, bias=False), Rearrange('b i j h -> b h i j')) if accept_edges else None

    def forward(self, x: 'torch.Tensor', edges: 'torch.Tensor'=None, mask: 'torch.Tensor'=None) ->torch.Tensor:
        """
        Forward pass of the Axial Attention module.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, height, width, dim).
            edges (torch.Tensor, optional): The edges tensor for attention bias. Defaults to None.
            mask (torch.Tensor, optional): The mask tensor for masking attention. Defaults to None.

        Returns:
            torch.Tensor: The output tensor of shape (batch_size, height, width, dim).
        """
        assert self.row_attn ^ self.col_attn, 'has to be either row or column attention, but not both'
        b, h, w, d = x.shape
        x = self.norm(x)
        if self.col_attn:
            axial_dim = w
            mask_fold_axial_eq = 'b h w -> (b w) h'
            input_fold_eq = 'b h w d -> (b w) h d'
            output_fold_eq = '(b w) h d -> b h w d'
        elif self.row_attn:
            axial_dim = h
            mask_fold_axial_eq = 'b h w -> (b h) w'
            input_fold_eq = 'b h w d -> (b h) w d'
            output_fold_eq = '(b h) w d -> b h w d'
        x = rearrange(x, input_fold_eq)
        if exists(mask):
            mask = rearrange(mask, mask_fold_axial_eq)
        attn_bias = None
        if exists(self.edges_to_attn_bias) and exists(edges):
            attn_bias = self.edges_to_attn_bias(edges)
            attn_bias = repeat(attn_bias, 'b h i j -> (b x) h i j', x=axial_dim)
        tie_dim = axial_dim if self.global_query_attn else None
        out = self.attn(x, mask=mask, attn_bias=attn_bias, tie_dim=tie_dim)
        out = rearrange(out, output_fold_eq, h=h, w=w)
        return out


class MsaAttentionBlock(nn.Module):

    def __init__(self, dim, seq_len, heads, dim_head, dropout=0.0):
        super().__init__()
        self.row_attn = AxialAttention(dim=dim, heads=heads, dim_head=dim_head, row_attn=True, col_attn=False, accept_edges=True)
        self.col_attn = AxialAttention(dim=dim, heads=heads, dim_head=dim_head, row_attn=False, col_attn=True)

    def forward(self, x, mask=None, pairwise_repr=None):
        x = self.row_attn(x, mask=mask, edges=pairwise_repr) + x
        x = self.col_attn(x, mask=mask) + x
        return x


class OuterMean(nn.Module):

    def __init__(self, dim, hidden_dim=None, eps=1e-05):
        super().__init__()
        self.eps = eps
        self.norm = nn.LayerNorm(dim)
        hidden_dim = default(hidden_dim, dim)
        self.left_proj = nn.Linear(dim, hidden_dim)
        self.right_proj = nn.Linear(dim, hidden_dim)
        self.proj_out = nn.Linear(hidden_dim, dim)

    def forward(self, x, mask=None):
        x = self.norm(x)
        left = self.left_proj(x)
        right = self.right_proj(x)
        outer = rearrange(left, 'b m i d -> b m i () d') * rearrange(right, 'b m j d -> b m () j d')
        if exists(mask):
            mask = rearrange(mask, 'b m i -> b m i () ()') * rearrange(mask, 'b m j -> b m () j ()')
            outer = outer.masked_fill(~mask, 0.0)
            outer = outer.mean(dim=1) / (mask.sum(dim=1) + self.eps)
        else:
            outer = outer.mean(dim=1)
        return self.proj_out(outer)


class TriangleMultiplicativeModule(nn.Module):

    def __init__(self, *, dim, hidden_dim=None, mix='ingoing'):
        super().__init__()
        assert mix in {'ingoing', 'outgoing'}, 'mix must be either ingoing or outgoing'
        hidden_dim = default(hidden_dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.left_proj = nn.Linear(dim, hidden_dim)
        self.right_proj = nn.Linear(dim, hidden_dim)
        self.left_gate = nn.Linear(dim, hidden_dim)
        self.right_gate = nn.Linear(dim, hidden_dim)
        self.out_gate = nn.Linear(dim, hidden_dim)
        for gate in (self.left_gate, self.right_gate, self.out_gate):
            nn.init.constant_(gate.weight, 0.0)
            nn.init.constant_(gate.bias, 1.0)
        if mix == 'outgoing':
            self.mix_einsum_eq = '... i k d, ... j k d -> ... i j d'
        elif mix == 'ingoing':
            self.mix_einsum_eq = '... k j d, ... k i d -> ... i j d'
        self.to_out_norm = nn.LayerNorm(hidden_dim)
        self.to_out = nn.Linear(hidden_dim, dim)

    def forward(self, x, mask=None):
        assert x.shape[1] == x.shape[2], 'feature map must be symmetrical'
        if exists(mask):
            mask = rearrange(mask, 'b i j -> b i j ()')
        x = self.norm(x)
        left = self.left_proj(x)
        right = self.right_proj(x)
        if exists(mask):
            left = left * mask
            right = right * mask
        left_gate = self.left_gate(x).sigmoid()
        right_gate = self.right_gate(x).sigmoid()
        out_gate = self.out_gate(x).sigmoid()
        left = left * left_gate
        right = right * right_gate
        out = einsum(self.mix_einsum_eq, left, right)
        out = self.to_out_norm(out)
        out = out * out_gate
        return self.to_out(out)


class PairwiseAttentionBlock(nn.Module):

    def __init__(self, dim, seq_len, heads, dim_head, dropout=0.0, global_column_attn=False):
        super().__init__()
        self.outer_mean = OuterMean(dim)
        self.triangle_attention_outgoing = AxialAttention(dim=dim, heads=heads, dim_head=dim_head, row_attn=True, col_attn=False, accept_edges=True)
        self.triangle_attention_ingoing = AxialAttention(dim=dim, heads=heads, dim_head=dim_head, row_attn=False, col_attn=True, accept_edges=True, global_query_attn=global_column_attn)
        self.triangle_multiply_outgoing = TriangleMultiplicativeModule(dim=dim, mix='outgoing')
        self.triangle_multiply_ingoing = TriangleMultiplicativeModule(dim=dim, mix='ingoing')

    def forward(self, x, mask=None, msa_repr=None, msa_mask=None):
        if exists(msa_repr):
            x = x + self.outer_mean(msa_repr, mask=msa_mask)
        x = self.triangle_multiply_outgoing(x, mask=mask) + x
        x = self.triangle_multiply_ingoing(x, mask=mask) + x
        x = self.triangle_attention_outgoing(x, edges=x, mask=mask) + x
        x = self.triangle_attention_ingoing(x, edges=x, mask=mask) + x
        return x


class PairFormerBlock(nn.Module):

    def __init__(self, *, dim: int, seq_len: int, heads: int, dim_head: int, attn_dropout: float, ff_dropout: float, global_column_attn: bool=False):
        """
        PairFormer Block module.

        Args:
            dim: The input dimension.
            seq_len: The length of the sequence.
            heads: The number of attention heads.
            dim_head: The dimension of each attention head.
            attn_dropout: The dropout rate for attention layers.
            ff_dropout: The dropout rate for feed-forward layers.
            global_column_attn: Whether to use global column attention in pairwise attention block.
        """
        super().__init__()
        self.layer = nn.ModuleList([PairwiseAttentionBlock(dim=dim, seq_len=seq_len, heads=heads, dim_head=dim_head, dropout=attn_dropout, global_column_attn=global_column_attn), FeedForward(dim=dim, dropout=ff_dropout), MsaAttentionBlock(dim=dim, seq_len=seq_len, heads=heads, dim_head=dim_head, dropout=attn_dropout), FeedForward(dim=dim, dropout=ff_dropout)])

    def forward(self, inputs: 'Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]') ->Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Forward pass of the PairFormer Block.

        Args:
            inputs: A tuple containing the input tensors (x, m, mask, msa_mask).

        Returns:
            A tuple containing the output tensors (x, m, mask, msa_mask).
        """
        x, m, mask, msa_mask = inputs
        attn, ff, msa_attn, msa_ff = self.layer
        m = msa_attn(m, mask=msa_mask, pairwise_repr=x)
        m = msa_ff(m) + m
        x = attn(x, mask=mask, msa_repr=m, msa_mask=msa_mask)
        x = ff(x) + x
        return x, m, mask, msa_mask


class PairFormer(nn.Module):

    def __init__(self, *, depth, **kwargs):
        super().__init__()
        self.layers = nn.ModuleList([PairFormerBlock(**kwargs) for _ in range(depth)])

    def forward(self, x, m, mask=None, msa_mask=None):
        inp = x, m, mask, msa_mask
        x, m, *_ = checkpoint_sequential(self.layers, 1, inp)
        return x, m


class AlphaFold3(nn.Module):
    """
    AlphaFold3 model implementation.

    Args:
        dim (int): Dimension of the model.
        seq_len (int): Length of the sequence.
        heads (int): Number of attention heads.
        dim_head (int): Dimension of each attention head.
        attn_dropout (float): Dropout rate for attention layers.
        ff_dropout (float): Dropout rate for feed-forward layers.
        global_column_attn (bool, optional): Whether to use global column attention. Defaults to False.
        pair_former_depth (int, optional): Depth of the PairFormer blocks. Defaults to 48.
        num_diffusion_steps (int, optional): Number of diffusion steps. Defaults to 1000.
        diffusion_depth (int, optional): Depth of the diffusion module. Defaults to 30.
    """

    def __init__(self, dim: 'int', seq_len: 'int', heads: 'int', dim_head: 'int', attn_dropout: 'float', ff_dropout: 'float', global_column_attn: 'bool'=False, pair_former_depth: 'int'=48, num_diffusion_steps: 'int'=1000, diffusion_depth: 'int'=30):
        super().__init__()
        self.dim = dim
        self.seq_len = seq_len
        self.heads = heads
        self.dim_head = dim_head
        self.attn_dropout = attn_dropout
        self.ff_dropout = ff_dropout
        self.global_column_attn = global_column_attn
        self.confidence_projection = nn.Linear(dim, 1)
        self.pairformer = PairFormer(dim=dim, seq_len=seq_len, heads=heads, dim_head=dim_head, attn_dropout=attn_dropout, ff_dropout=ff_dropout, global_column_attn=global_column_attn, depth=pair_former_depth)
        self.diffuser = GeneticDiffusion(channels=dim, num_diffusion_steps=num_diffusion_steps, training=False, depth=diffusion_depth)
        self.norm = nn.LayerNorm(dim)

    def forward(self, pair_representation: 'Tensor', single_representation: 'Tensor', return_loss: 'bool'=False, ground_truth: 'Tensor'=None, return_confidence: 'bool'=False, return_embeddings: 'bool'=True) ->Tensor:
        """
        Forward pass of the AlphaFold3 model.

        Args:
            pair_representation (Tensor): Pair representation tensor.
            single_representation (Tensor): Single representation tensor.
            return_loss (bool, optional): Whether to return the loss. Defaults to False.
            ground_truth (Tensor, optional): Ground truth tensor. Defaults to None.
            return_confidence (bool, optional): Whether to return the confidence. Defaults to False.
            return_embeddings (bool, optional): Whether to return the embeddings. Defaults to False.

        Returns:
            Tensor: Output tensor based on the specified return type.
        """
        b, n, n_two, dim = pair_representation.shape
        b_two, n_two, dim_two = single_representation.shape
        single_representation = single_representation.unsqueeze(2)
        None
        z, j, y, d = single_representation.shape
        single_representation = rearrange(single_representation, 'b n s d -> b n d s')
        single_representation = nn.Linear(y, n)(single_representation)
        None
        single_representation = rearrange(single_representation, 'b n d s -> b n s d')
        None
        pair_representation = self.norm(pair_representation)
        single_representation = self.norm(single_representation)
        x, m = self.pairformer(pair_representation, single_representation)
        None
        None
        x = self.diffuser(x)
        if return_confidence is True:
            x = self.confidence_projection(x)
            return x
        if return_embeddings is True:
            return x


class Always(nn.Module):

    def __init__(self, val):
        super().__init__()
        self.val = val

    def forward(self, x):
        return self.val


class TemplateEmbedder(nn.Module):

    def __init__(self, dim: 'int'=None, depth: 'int'=2, seq_len: 'int'=None, heads: 'int'=64, dim_head: 'int'=64, attn_dropout: 'float'=0.0, ff_dropout: 'float'=0.0, global_column_attn: 'bool'=False, c: 'int'=64, Ntemplates: 'int'=1, *args, **kwargs):
        super(TemplateEmbedder, self).__init__()
        self.layer_norm_z = nn.LayerNorm(c)
        self.layer_norm_v = nn.LayerNorm(c)
        self.linear_no_bias_z = nn.Linear(c, c, bias=False)
        self.linear_no_bias_a = nn.Linear(c, c, bias=False)
        self.pairformer = PairFormer(*args, dim=dim, seq_len=seq_len, heads=heads, dim_head=dim_head, attn_dropout=attn_dropout, ff_dropout=ff_dropout, depth=depth, **kwargs)
        self.relu = nn.ReLU()
        self.final_linear = nn.Linear(c, c, bias=False)

    def forward(self, f, zij, Ntemplates):
        template_backbone_frame_mask = f
        template_pseudo_beta_mask = f
        template_distogram = f
        template_unit_vector = f
        atij = torch.cat([template_distogram, template_backbone_frame_mask, template_unit_vector, template_pseudo_beta_mask], dim=-1)
        asym_id_mask = f == f
        atij = atij * asym_id_mask
        restype = f
        atij = torch.cat([atij, restype, restype], dim=-1)
        uij = torch.zeros_like(atij)
        for _ in range(Ntemplates):
            vij = self.linear_no_bias_z(self.layer_norm_z(zij)) + self.linear_no_bias_a(atij)
            for layer in self.pairformer_stack:
                vij = layer(vij)
            uij += self.layer_norm_v(vij)
        uij /= Ntemplates
        uij = self.final_linear(self.relu(uij))
        return uij


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (Always,
     lambda: ([], {'val': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (FeedForward,
     lambda: ([], {'dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (GEGLU,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (TriangleMultiplicativeModule,
     lambda: ([], {'dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
]

