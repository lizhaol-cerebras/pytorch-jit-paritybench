
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


from typing import Dict


from typing import Optional


from typing import Type


from typing import cast


import torch


from typing import Any


from typing import Generic


from typing import List


from typing import TypeVar


from typing import Iterator


from typing import Tuple


from torch import Tensor


from torch.distributions import Categorical


from abc import ABC


from abc import abstractmethod


from collections import UserList


from typing import Iterable


from typing import TYPE_CHECKING


import math


from enum import Enum


from torch.nn import Module


from typing import Union


import torch.nn.functional as F


from torch.nn import Dropout


from torch.nn import Linear


from typing import Protocol


from torch.nn import Parameter


from torch.nn import Embedding


from torch.nn import Identity


from typing import Mapping


from torch.nn import LayerNorm


from functools import partial


from torch.nn import ModuleList


import warnings


from typing import Set


from typing import Callable


from copy import deepcopy


import functools


from torch.utils.hooks import RemovableHandle


class GELUNew(Module):
    """
    GELU (`Hendrycks et al., 2016`_) approximation, called ``gelu_new`` in many transformer models.

    .. _Hendrycks et al., 2016: https://arxiv.org/abs/1606.08415
    """

    def forward(self, input: 'Tensor') ->Tensor:
        """
        Apply the GELU activation on the input.

        :param input:
            Input tensor.

            *Shape:* ``(batch_size, seq_len, width)``
        """
        return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))


class GELUFast(Module):
    """
    GELU (`Hendrycks et al., 2016`_) approximation used by GPT-NeoX (`Black et al., 2022`_).

    .. _Hendrycks et al., 2016: https://arxiv.org/abs/1606.08415
    .. _Black et al., 2022: https://arxiv.org/abs/2204.06745
    """

    def forward(self, input: 'Tensor') ->Tensor:
        """
        Apply the GELU activation on the input.

        :param input:
            Input tensor.

            *Shape:* ``(batch_size, seq_len, width)``
        """
        alpha = math.sqrt(2.0 / math.pi)
        beta = 0.044715
        return 0.5 * input * (1.0 + torch.tanh(alpha * (input + beta * input.pow(3))))


class AttentionLinearBiases(Module):
    """
    ALiBi: Linear biases for attention (`Press et al., 2022`_).

    .. _Press et al., 2022: https://arxiv.org/abs/2108.12409
    """
    slopes: 'Tensor'

    def __init__(self, *, n_attention_heads: int, is_causal: bool, is_inverted: bool) ->None:
        """
        Construct an ALiBi module.

        :param n_attention_heads:
            Number of attention heads.
        :param is_causal:
            Use causal attention.
        :param invert:
            If ``True``, the biases are inverted, i.e.,
            penalties become rewards.
        """
        super().__init__()
        self.is_causal = is_causal
        self.invert = is_inverted
        slopes = self._calculate_slopes(n_attention_heads)
        self.register_buffer('slopes', slopes, persistent=False)

    def _calculate_slopes(self, n_attention_heads: 'int') ->Tensor:
        """
        Calculate the linear bias slopes for a given number
        of attention heads.

        :param n_attention_heads:
            Number of attention heads.
        :returns:
            Head slope tensor.

            *Shape:* ``(1, heads, 1, 1)``

        :meta private:
        """

        def _slopes_with_step(n_attention_heads, *, step=1):
            ratio = 2.0 ** (-8.0 / n_attention_heads)
            return ratio ** torch.arange(1, 1 + n_attention_heads, step)
        k = 1 << n_attention_heads.bit_length() - 1
        slopes = _slopes_with_step(k)
        if n_attention_heads != k:
            remaining_heads = n_attention_heads - k
            slopes_rest = _slopes_with_step(2 * k, step=2)[:remaining_heads]
            slopes = torch.cat([slopes, slopes_rest])
        return slopes.view(1, -1, 1, 1)

    def calculate_biases(self, seq_len: 'int') ->Tensor:
        """
        Calculate the linear bias tensor upto a given (key) sequence length.

        :param seq_len:
            Maximum number of timesteps to calculate.
        :returns:
            Multi-headed linear bias tensor.

            *Shape:* ``(1, heads, seq_len, seq_len)`` (non-causal) or
            ``(1, heads, 1, seq_len)`` (causal)

        :meta private:
        """
        if self.is_causal:
            distances = torch.arange(1 - seq_len, 1)
        else:
            distances = torch.arange(seq_len) - torch.arange(seq_len).view(-1, 1)
            distances = distances.abs().mul(-1).view(1, 1, seq_len, seq_len)
        if self.invert:
            distances += seq_len - 1
        return distances * self.slopes

    def forward(self, *, attention_scores: Tensor, inplace: bool=True) ->Tensor:
        """
        Apply linear biases to (unmasked) attention scores.

        :param attention_scores:
            Attention scores.

            *Shape:* ``(batch_size, heads, query_len, key_len)``
        :param inplace:
            Update attention scores inplace.
        :returns:
            Attention scores with linear biases.

            *Shape:* ``(batch_size, heads, query_len, key_len)``
        """
        if not inplace:
            attention_scores = attention_scores.clone()
        biases = self.calculate_biases(attention_scores.size(-1))
        return attention_scores + biases


class QkvSplit(ABC):
    """
    Query, key, value splitting strategies.

    After the input projection of the attention layer, we have an array with
    shape ``(batch_size, seq_len, n_heads * head_width)`` where ``n_heads`` is
    the sum of the number of query, key, and value heads. We need to split up
    the array into separate arrays for query, key, and value heads.

    Subclasses of this class implement different splitting strategies.
    """

    @abstractmethod
    def split(self, *, projection: Tensor, head_width: int, n_query_heads: int, n_key_value_heads: int) ->Tuple[Tensor, Tensor, Tensor]:
        """
        Split attention heads in the projection in query, key, and value
        heads.

        :param projection:
            The fused query, key, value projection.

            *Shape:* ``(batch_size, seq_len, (n_query_heads + 2 * n_key_value_heads) * head_width)``
        :param head_width:
            Head width.
        :param n_query_heads:
            Number of query heads.
        :param n_key_value_heads:
            Number of key/value heads.
        :returns:
            Query, key, value tensors.

            *Shapes:*

            - Query: ``(batch_size, n_query_heads, seq_len, head_width)``

            - Key: ``(batch_size, n_key_value_heads, seq_len, head_width)``

            - Value: ``(batch_size, n_key_value_heads, seq_len, head_width)``
        """
        ...


class AttentionHeads:

    def __init__(self, *, n_query_heads: int, n_key_value_heads: int, qkv_split: QkvSplit):
        """
        Construct an attention head configuration. This constructor must
        not be used directly, its signature may change even within a semver
        version. Use the factory methods instead.

        :param n_query_heads:
            Number of query heads.
        :param n_key_value_heads:
            Number of key/value heads.
        :param qkv_split:
            How query, key, and value should be split when using
            :py:class:`~curated_transformers.layers.QkvMode.MERGED_SPLIT_AFTER`.
            Not used for other query, key, value modes.

        :meta private:
        """
        self._n_query_heads = n_query_heads
        self._n_key_value_heads = n_key_value_heads
        self._qkv_split = qkv_split

    @classmethod
    def uniform(cls, n_attention_heads: 'int', qkv_split: 'QkvSplit') ->'AttentionHeads':
        """
        Construct a head configuration where query, key, and value have the
        same number of attention heads.

        :param n_attention_heads:
            Number of attention heads.
        :param qkv_split:
            How query, key, and value should be split when using
            :py:class:`~curated_transformers.layers.QkvMode.MERGED_SPLIT_AFTER`.
            Not used for other query, key, value modes.
        """
        return cls(n_query_heads=n_attention_heads, n_key_value_heads=n_attention_heads, qkv_split=qkv_split)

    @classmethod
    def multi_query(cls, n_query_heads: 'int', qkv_split: 'QkvSplit') ->'AttentionHeads':
        """
        Construct a multi-query attention configuration: key has one head,
        value has one head, query has ``n_query_heads`` heads
        (`Shazeer et al., 2019`_). The key head and the value head are
        broadcast to the shape of the query.

        .. _Shazeer et al., 2019: https://arxiv.org/abs/1911.02150

        :param n_query_heads:
            Number of query heads.
        :param qkv_split:
            How query, key, and value should be split when using
            :py:class:`~curated_transformers.layers.QkvMode.MERGED_SPLIT_AFTER`.
            Not used for other query, key, value modes.
        """
        return cls(n_query_heads=n_query_heads, n_key_value_heads=1, qkv_split=qkv_split)

    @classmethod
    def key_value_broadcast(cls, *, n_query_heads: int, n_key_value_heads: int, qkv_split: QkvSplit) ->'AttentionHeads':
        """
        Construct a head configuration where query has a larger number
        of heads than key and value. Key/value heads are broadcast to
        correspond to the number of query heads.

        :param n_query_heads:
            Number of attention heads. Must be a multiple of
            ``n_key_value_heads``.
        :param n_key_value_heads:
            Number of key and value heads.
        :param qkv_split:
            How query, key, and value should be split when using
            :py:class:`~curated_transformers.layers.QkvMode.MERGED_SPLIT_AFTER`.
            Not used for other query, key, value modes.
        """
        if n_query_heads < n_key_value_heads or n_query_heads % n_key_value_heads != 0:
            raise ValueError(f'Number of query heads ({n_query_heads}) must be a multiple of key/value heads ({n_key_value_heads})')
        return cls(n_query_heads=n_query_heads, n_key_value_heads=n_key_value_heads, qkv_split=qkv_split)


class QkvMode(Enum):
    """
    How the query, key and value projections are handled in
    the self-attention layer.
    """
    SEPARATE = 0
    MERGED_SPLIT_BEFORE = 1
    MERGED_SPLIT_AFTER = 2


class RotaryEmbeddings(Module):
    """
    Rotary embeddings (`Su et al., 2021`_).

    .. _Su et al., 2021: https://arxiv.org/abs/2104.09864
    """
    cos: 'Tensor'
    sin: 'Tensor'
    theta: 'Tensor'

    def __init__(self, width: 'int', *, seq_len: int=512, base: int=10000, device: Optional[torch.device]=None):
        """
        Construct a rotary embedding module. The rotary embedding
        will be precomputed for up to ``seq_len`` positions. The embedding
        will be recomputed when a longer sequence is found in the input.

        :param width:
            Rotary embedding width.
            Must be even.
        :param seq_len:
            Number of positions to initially precompute.
        :param base:
            The base used for :math:`\\theta_i`.
            Determines the cycle length of the embeddings.
        :param device:
            Device on which the module is to be initialized.
        """
        super().__init__()
        if width % 2:
            raise ValueError(f'Width of rotary embeddings must be even, was: {width}')
        if device is not None and device.type == 'meta':
            device = None
        theta = torch.pow(base, -torch.arange(0, width, 2, dtype=torch.float, device=device) / width)
        self.register_buffer('theta', theta, persistent=False)
        self._create_rotary_embed(width=width, length=seq_len)

    def _create_rotary_embed(self, *, width: int, length: int):
        position = torch.arange(length, device=self.theta.device).unsqueeze(1)
        m_theta = position * self.theta.unsqueeze(0)
        m_theta = torch.cat([m_theta, m_theta], dim=-1)
        re_cos = m_theta.cos().view([length, width])
        re_sin = m_theta.sin().view([length, width])
        self.register_buffer('cos', re_cos, persistent=False)
        self.register_buffer('sin', re_sin, persistent=False)

    def _rotate(self, input: 'Tensor'):
        """
        Rotate the input tensor by half of its innermost width.

        :param input:
            Tensor to rotate.

            *Shape:* ``(..., width)``
        :returns:
            Rotated tensor.

            *Shape:* ``(.., width)``

        :meta private:
        """
        half_idx = input.shape[-1] // 2
        input_1 = -input[..., half_idx:]
        input_2 = input[..., :half_idx]
        return torch.cat([input_1, input_2], dim=-1)

    def forward(self, input: 'torch.Tensor', *, positions: Optional[Tensor]=None):
        """
        Apply rotary embeddings to the input.

        :param input:
            Input to apply the rotary embeddings to.

            *Shape:* ``(batch_size, n_heads, seq_len, width_per_head)``
        :param positions:
            Positions of the inputs. If no positions are
            provided, they are assumed to be ``[0, seq_len)``.

            *Shape:* ``(batch_size, seq_len)``
        :returns:
            Input with the rotary embeddings applied.

            *Shape:* ``(batch_size, n_heads, seq_len, width_per_head)``
        """
        batch_size, _, seq_len, width = input.shape
        if positions is None:
            if self.cos.size(-2) < seq_len:
                self._create_rotary_embed(width=width, length=seq_len)
            rot_cos = self.cos[:seq_len, :].view(1, 1, seq_len, width)
            rot_sin = self.sin[:seq_len, :].view(1, 1, seq_len, width)
        else:
            max_len = int(positions.max()) + 1
            if self.cos.size(-2) < max_len:
                self._create_rotary_embed(width=width, length=max_len)
            positions_flat = positions.view(-1)
            rot_cos = self.cos[positions_flat].view(batch_size, 1, seq_len, width)
            rot_sin = self.sin[positions_flat].view(batch_size, 1, seq_len, width)
        return rot_cos * input + rot_sin * self._rotate(input)


def combine_heads(input: 'Tensor') ->Tensor:
    """
    Combine the split attention head representations.

    :param input:
        Tensor split by head.

        *Shape:* ``(batch_size, head, seq_len, width_per_head)``
    :returns:
        Merged tensor.

        *Shape:* ``(batch_size, seq_len, hidden_width)``
    """
    batch_size, head, seq_len, model_width = input.size()
    return input.transpose(1, 2).contiguous().view(batch_size, seq_len, head * model_width)


def split_heads(input: 'Tensor', n_heads: 'int') ->Tensor:
    """
    Split the input by attention head. The caller must validate
    that the innermost dimension is divisable by the number of
    heads.

    :param input:
        Tensor to split by head.

        *Shape:* ``(batch_size, seq_len, hidden_width)``
    :param n_heads:
        Number of attention heads.
    :returns:
        Tensor spilt by head.

        *Shape:* ``(batch_size, head, seq_len, width_per_head)``
    """
    batch_size, seq_len, model_width = input.size()
    assert model_width % n_heads == 0
    head_width = model_width // n_heads
    return input.view(batch_size, seq_len, n_heads, head_width).transpose(1, 2)


class SinusoidalPositionalEmbedding(Module):
    """
    Sinusoidal positional embeddings (`Vaswani et al., 2017`_).

    .. _Vaswani et al., 2017: https://arxiv.org/abs/1706.03762
    """

    def __init__(self, *, width: int, max_len: int, normalize=True, device: Optional[torch.device]=None):
        """
        Construct a sinusoidal positional embedding module.

        :param width:
            Width of the embedding.
        :param max_len:
            Maximum length of the embedding.
        :param normalize:
            Perform L2 normalization of the embedding.
        :param device:
            Device on which the module is to be initialized.
        """
        super().__init__()
        position = torch.arange(max_len, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, width, 2, device=device) * (-math.log(10000.0) / width))
        pe = torch.zeros(max_len, width, device=device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        if normalize == True:
            l2 = torch.linalg.vector_norm(pe, dim=-1)
            pe /= l2.unsqueeze(-1)
        self.pe = pe
        self.pe.requires_grad = False

    def forward(self, input: 'Tensor') ->Tensor:
        """
        Returns the positional embedding for the input.

        :param input:
            Input tensor.

            *Shape:* ``(batch_size, seq_len)``
        :returns:
            Positional embedding for the input.

            *Shape:* ``(seq_len, width)``
        """
        return self.pe[:input.size(1), :]


class PointwiseFeedForward(Module):
    """
    Point-wise feed-forward layer (`Vaswani et al., 2017`_).

    This layer is applied pointwise, meaning that the same
    transformation is applied to each sequence element. This
    transformation is:

    .. math::
        g(xW_1 + b_1)W_2 + b_2

    :math:`W_1` and :math:`b_1` transform the input to an
    intermediate width, :math:`g` is a non-linear activation
    function and :math:`W_2` and :math:`b_2` transform the
    output of the activation back to the input width.

    Gated Linear Units (`Dauphin et al., 2016`_; `Shazeer, 2020`_) are also
    supported. Gating applies the following transformation:

    .. math::
        (g(xW_g + b_g) * (xW_1 + b_1))W_2 + b_2

    :math:`W_g` and :math:`b_g` are the affine transformation for the gate.

    .. _Vaswani et al., 2017: https://arxiv.org/abs/1706.03762
    .. _Dauphin et al., 2016: https://arxiv.org/abs/1612.08083
    .. _Shazeer, 2020: https://arxiv.org/abs/2002.05202
    """
    gate: 'Optional[Linear]'

    def __init__(self, *, activation: Module, hidden_width: int, intermediate_width: int, use_bias: bool, use_gate: bool, device: Optional[torch.device]=None):
        """
        Construct a pointwise feed-forward layer module.

        :param activation:
            Activation used by the pointwise feed-forward layers. The hidden
            input shape must be the same as the output shape (as is typical
            for elementwise activations).
        :param hidden_width:
            The input and output width of the layer.
        :param intermediate_width:
            The width of the projection to which the non-linearity is applied.
        :param use_bias:
            Use biases for linear layers.
        :param use_gate:
            Use Gated Linear Units.
        :param device:
            Device on which the module is to be initialized.
        """
        super().__init__()
        self.intermediate = Linear(hidden_width, intermediate_width, bias=use_bias, device=device)
        if use_gate:
            self.gate = Linear(hidden_width, intermediate_width, bias=use_bias, device=device)
        else:
            self.gate = None
        self.output = Linear(intermediate_width, hidden_width, bias=use_bias, device=device)
        self.activation = activation

    def forward(self, input: 'Tensor') ->Tensor:
        """
        Apply the point-wise feed-forward layer to the input.

        :param input:
            Input.

            *Shape:* ``(batch_size, seq_len, width)``
        :returns:
            Layer output.

            *Shape:* ``(batch_size, seq_len, width)``
        """
        if self.gate is None:
            return self.output(self.activation(self.intermediate(input)))
        else:
            return self.output(self.activation(self.gate(input)) * self.intermediate(input))


class RMSNorm(Module):
    """
    Root Mean Square (RMS) normalization (`Zhang et al., 2019`_).

    .. _Zhang et al., 2019: https://arxiv.org/abs/1910.07467
    """

    def __init__(self, width: 'int', *, eps: float, device: Optional[torch.device]=None):
        """
        Construct a RMS normalization module.

        :param width:
            The (hidden) width of the representations that RMS
            normalization will be applied to.
        :param eps:
            Epsilon to avoid division by zero.
        :param device:
            Device on which the module is to be initialized.
        """
        super().__init__()
        self.eps = eps
        self.weight = Parameter(torch.ones((width,), device=device))

    def forward(self, input: 'Tensor') ->Tensor:
        """
        Apply RMS normalization to a tensor.

        :param input:
            The tensor to apply normalization to.
        :returns:
            Normalized tensor.
        """
        rms = input.square().mean(-1, keepdim=True).add(self.eps).rsqrt()
        return input * rms * self.weight


class QkvSplitGroupedByKVHeads(QkvSplit):
    """
    Split up the projection in key/value-sized chunks.

    First view the array as ``(batch_size, seq_len, n_key_value_heads,
    n_chunks, head_width)``. Then split up the array along the ``n_chunks``
    dimension.
    """

    def split(self, *, projection: Tensor, head_width: int, n_query_heads: int, n_key_value_heads: int) ->Tuple[Tensor, Tensor, Tensor]:
        batch_size, seq_len, _ = projection.shape
        grouped_projection = projection.view(batch_size, seq_len, -1, n_query_heads // n_key_value_heads + 2, head_width)
        query, key, value = grouped_projection.split([n_query_heads // n_key_value_heads, 1, 1], dim=-2)

        def permute_merge_groups(x, n_heads):
            return x.permute(0, 2, 3, 1, 4).reshape(batch_size, n_heads, seq_len, head_width)
        query = permute_merge_groups(query, n_query_heads)
        key = permute_merge_groups(key, n_key_value_heads)
        value = permute_merge_groups(value, n_key_value_heads)
        return query, key, value


class ALBERTLayerGroup(Module):
    """
    ALBERT (`Lan et al., 2022`_) layer group.

    .. _Lan et al., 2022: https://arxiv.org/abs/1909.11942
    """

    def __init__(self, layer_config: 'ALBERTLayerConfig', *, device: Optional[torch.device]=None) ->None:
        super().__init__()
        layer_norm = partial(LayerNorm, layer_config.feedforward.hidden_width, layer_config.layer_norm_eps, device=device)
        attention_config = layer_config.attention
        self.group_layers = ModuleList([EncoderLayer(attention_layer=SelfAttention(attention_heads=AttentionHeads.uniform(attention_config.n_query_heads, QkvSplitGroupedByKVHeads()), attention_scorer=ScaledDotProductAttention(dropout_prob=attention_config.dropout_prob, linear_biases=None), hidden_width=layer_config.feedforward.hidden_width, qkv_mode=QkvMode.SEPARATE, rotary_embeds=None, use_bias=attention_config.use_bias, device=device), feed_forward_layer=PointwiseFeedForward(activation=layer_config.feedforward.activation.module(), hidden_width=layer_config.feedforward.hidden_width, intermediate_width=layer_config.feedforward.intermediate_width, use_bias=layer_config.feedforward.use_bias, use_gate=layer_config.feedforward.use_gate, device=device), dropouts=TransformerDropouts.layer_output_dropouts(layer_config.dropout_prob), layer_norms=TransformerLayerNorms(attn_residual_layer_norm=layer_norm(), ffn_residual_layer_norm=layer_norm()), use_parallel_attention=attention_config.use_parallel_attention) for _ in range(layer_config.n_layers_per_group)])

    def forward(self, input: 'Tensor', attention_mask: 'AttentionMask') ->Tensor:
        """
        Apply the ALBERT layer group to the given piece hidden representations.

        :param input:
            Hidden representations to apply the layer group to.

            *Shape:* ``(batch_size, seq_len, width)``
        :param attention_mask:
            Attention mask. Sequence elements for which the
            corresponding mask element is set to ``False`` are ignored
            during attention calculation.
        :returns:
            Layer output.

            *Shape:* ``(batch_size, seq_len, width)``
        """
        layer_output = input
        for layer in self.group_layers:
            layer_output = layer(layer_output, attention_mask)
        return layer_output


class DecoderWithCache(Module):

    def __init__(self, decoder: 'DecoderModule'):
        super().__init__()
        self.inner = decoder

    def forward(self, piece_ids: 'Tensor', attention_mask: 'AttentionMask', cache: 'List[KeyValueCache]'):
        return self.inner.forward(piece_ids=piece_ids, attention_mask=attention_mask, cache=cache, store_cache=True)


class DecoderWithPositions(Module):

    def __init__(self, decoder: 'DecoderModule'):
        super().__init__()
        self.inner = decoder

    def forward(self, piece_ids: 'Tensor', attention_mask: 'AttentionMask', positions: 'Tensor'):
        return self.inner.forward(piece_ids=piece_ids, attention_mask=attention_mask, positions=positions)


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (GELUFast,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (GELUNew,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (PointwiseFeedForward,
     lambda: ([], {'activation': torch.nn.ReLU(), 'hidden_width': 4, 'intermediate_width': 4, 'use_bias': 4, 'use_gate': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (RMSNorm,
     lambda: ([], {'width': 4, 'eps': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (RotaryEmbeddings,
     lambda: ([], {'width': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (SinusoidalPositionalEmbedding,
     lambda: ([], {'width': 4, 'max_len': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
]

