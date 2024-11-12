
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


import warnings


from torch import Tensor


from typing import Optional


from typing import Tuple


import numpy as np


import torch.nn as nn


import torch.nn.functional as F


import math


class ScaledDotProductAttention(nn.Module):
    """
    Compute the dot products of the query with all keys, divide each by sqrt(key_dim),
    and apply a softmax function to obtain the weights on the values

    Args: key_dim
        key_dim (int): dimension of key

    Inputs: query, key, value
        - **query** (batch, q_len, hidden_dim): tensor containing projection vector for decoder
        - **key** (batch, k_len, hidden_dim): tensor containing projection vector for encoder
        - **value** (batch, v_len, hidden_dim): value and key are the same
        - **mask** (batch, q_len, k_len): tensor containing mask vector for attn_distribution

    Returns: context, attn_distribution
        - **context** (batch, q_len, hidden_dim): tensor containing the context vector from attention mechanism
        - **attn_distribution** (batch, q_len, k_len): tensor containing the attention from the encoder outputs
    """

    def __init__(self, key_dim: 'int') ->None:
        super(ScaledDotProductAttention, self).__init__()
        self.sqrt_key_dim = np.sqrt(key_dim)

    def forward(self, query: 'Tensor', key: 'Tensor', value: 'Tensor', mask: 'Optional[Tensor]'=None) ->Tuple[Tensor, Tensor]:
        key = key.transpose(1, 2)
        attn_distribution = torch.bmm(query, key) / self.sqrt_key_dim
        if mask is not None:
            attn_distribution = attn_distribution.masked_fill(mask, -np.inf)
        attn_distribution = F.softmax(attn_distribution, dim=-1)
        context = torch.bmm(attn_distribution, value)
        return context, attn_distribution


class MultiHeadAttention(nn.Module):
    """
    This technique is proposed in this paper. https://arxiv.org/abs/1706.03762
    Perform the scaled dot-product attention in parallel.

    Args:
        model_dim (int): the number of features in the multi-head attention (default : 512)
        num_heads (int): the number of heads in the multi-head attention (default: 8)

    Inputs: query, key, value, mask
        - **query** (batch, q_len, hidden_dim): tensor containing projection vector for encoder
        - **key** (batch, k_len, hidden_dim): tensor containing projection vector for encoder
        - **value** (batch, v_len, hidden_dim): tensor containing projection vector for encoder
        - **mask** (batch, q_len, k_len): tensor containing mask vector for self attention distribution

    Returns: context, attn_distribution
        - **context** (batch, dec_len, dec_hidden): tensor containing the context vector from attention mechanism
        - **attn_distribution** (batch, dec_len, enc_len): tensor containing the attention from the encoder outputs
    """

    def __init__(self, model_dim: 'int', num_heads: 'int') ->None:
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        self.scaled_dot = ScaledDotProductAttention(self.head_dim)
        self.query_fc = nn.Linear(model_dim, num_heads * self.head_dim)
        self.key_fc = nn.Linear(model_dim, num_heads * self.head_dim)
        self.value_fc = nn.Linear(model_dim, num_heads * self.head_dim)

    def forward(self, query: 'Tensor', key: 'Tensor', value: 'Tensor', mask: 'Optional[Tensor]'=None) ->Tuple[Tensor, Tensor]:
        batch = query.size(0)
        query = self.query_fc(query).view(batch, -1, self.num_heads, self.head_dim)
        key = self.key_fc(key).view(batch, -1, self.num_heads, self.head_dim)
        value = self.value_fc(value).view(batch, -1, self.num_heads, self.head_dim)
        query = query.permute(0, 2, 1, 3).contiguous().view(batch * self.num_heads, -1, self.head_dim)
        key = key.permute(0, 2, 1, 3).contiguous().view(batch * self.num_heads, -1, self.head_dim)
        value = value.permute(0, 2, 1, 3).contiguous().view(batch * self.num_heads, -1, self.head_dim)
        if mask is not None:
            mask = mask.repeat(self.num_heads, 1, 1)
        context, attn_distribution = self.scaled_dot(query, key, value, mask)
        context = context.view(batch, self.num_heads, -1, self.head_dim)
        context = context.permute(0, 2, 1, 3).contiguous().view(batch, -1, self.num_heads * self.head_dim)
        return context, attn_distribution


class PositionWiseFeedForward(nn.Module):
    """
    Implement position-wise feed forward layer.
    FFN(x) = max(0, xW1 + b1)W2 + b2
    """

    def __init__(self, model_dim: 'int'=512, ff_dim: 'int'=2048, dropout: 'float'=0.1) ->None:
        super(PositionWiseFeedForward, self).__init__()
        self.feed_forward = nn.Sequential(nn.Linear(model_dim, ff_dim), nn.ReLU(), nn.Dropout(p=dropout), nn.Linear(ff_dim, model_dim), nn.Dropout(p=dropout))

    def forward(self, inputs: 'Tensor') ->Tensor:
        return self.feed_forward(inputs)


class EncoderLayer(nn.Module):
    """
    Repeated layers common to audio encoders and label encoders

    Args:
        model_dim (int): the number of features in the encoder (default : 512)
        ff_dim (int): the number of features in the feed forward layers (default : 2048)
        num_heads (int): the number of heads in the multi-head attention (default: 8)
        dropout (float): dropout probability of encoder layer (default: 0.1)

    Inputs: inputs, self_attn_mask
        - **inputs**: Audio feature or label feature
        - **self_attn_mask**: Self attention mask to use in multi-head attention

    Returns: outputs, attn_distribution
        - **outputs**: Tensor containing higher (audio, label) feature values
        - **attn_distribution**: Attention distribution in multi-head attention
    """

    def __init__(self, model_dim: 'int'=512, ff_dim: 'int'=2048, num_heads: 'int'=8, dropout: 'float'=0.1) ->None:
        super(EncoderLayer, self).__init__()
        self.layer_norm = nn.LayerNorm(model_dim)
        self.self_attention = MultiHeadAttention(model_dim, num_heads)
        self.encoder_dropout = nn.Dropout(p=dropout)
        self.feed_forward = PositionWiseFeedForward(model_dim, ff_dim, dropout)

    def forward(self, inputs: 'Tensor', self_attn_mask: 'Optional[Tensor]'=None) ->Tuple[Tensor, Tensor]:
        """
        Forward propagate a `inputs` for label encoder.

        Args:
            inputs : A input sequence passed to encoder layer. ``(batch, seq_length, dimension)``
            self_attn_mask : Self attention mask to cover up padding ``(batch, seq_length, seq_length)``

        Returns:
            **outputs** (Tensor): ``(batch, seq_length, dimension)``
            **attn_distribution** (Tensor): ``(batch, seq_length, seq_length)``
        """
        inputs = self.layer_norm(inputs)
        self_attn_output, attn_distribution = self.self_attention(inputs, inputs, inputs, self_attn_mask)
        self_attn_output += inputs
        self_attn_output = self.layer_norm(self_attn_output)
        ff_output = self.feed_forward(self_attn_output)
        output = self.encoder_dropout(ff_output + self_attn_output)
        return output, attn_distribution


class PositionalEncoding(nn.Module):
    """
    Implement the positional encoding (PE) function.

    PE_(pos, 2i)    =  sin(pos / 10000 ** (2i / d_model))
    PE_(pos, 2i+1)  =  cos(pos / 10000 ** (2i / d_model))
    """

    def __init__(self, d_model: 'int'=512, max_len: 'int'=5000) ->None:
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model, requires_grad=False)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, length: 'int') ->Tensor:
        return self.pe[:, :length, :]


def _get_pad_mask(inputs: 'Tensor', inputs_lens: 'Tensor'):
    assert len(inputs.size()) == 3
    batch = inputs.size(0)
    pad_attn_mask = inputs.new_zeros(inputs.size()[:-1])
    for idx in range(batch):
        pad_attn_mask[idx, inputs_lens[idx]:] = 1
    return pad_attn_mask.bool()


def get_attn_pad_mask(inputs: 'Tensor', inputs_lens: 'Tensor', expand_lens):
    pad_attn_mask = _get_pad_mask(inputs, inputs_lens)
    pad_attn_mask = pad_attn_mask.unsqueeze(1).repeat(1, expand_lens, 1)
    return pad_attn_mask


class AudioEncoder(nn.Module):
    """
    Converts the audio signal to higher feature values

    Args:
        device (torch.device): flag indication whether cpu or cuda
        input_size (int): dimension of input vector (default : 80)
        model_dim (int): the number of features in the audio encoder (default : 512)
        ff_dim (int): the number of features in the feed forward layers (default : 2048)
        num_layers (int): the number of audio encoder layers (default: 18)
        num_heads (int): the number of heads in the multi-head attention (default: 8)
        dropout (float): dropout probability of audio encoder (default: 0.1)
        max_len (int): Maximum length to use for positional encoding (default : 5000)

    Inputs: inputs, inputs_lens
        - **inputs**: Parsed audio of batch size number
        - **inputs_lens**: Tensor of sequence lengths

    Returns: outputs
        - **outputs**: Tensor containing higher feature values
    """

    def __init__(self, device: 'torch.device', input_size: 'int'=80, model_dim: 'int'=512, ff_dim: 'int'=2048, num_layers: 'int'=18, num_heads: 'int'=8, dropout: 'float'=0.1, max_len: 'int'=5000) ->None:
        super(AudioEncoder, self).__init__()
        self.device = device
        self.input_size = input_size
        self.model_dim = model_dim
        self.ff_dim = ff_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.input_dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(model_dim)
        self.positional_encoding = PositionalEncoding(model_dim, max_len)
        self.input_fc = nn.Linear(input_size, model_dim)
        self.encoder_layers = nn.ModuleList([EncoderLayer(model_dim, ff_dim, num_heads, dropout) for _ in range(num_layers)])

    def forward(self, inputs: 'Tensor', inputs_lens: 'Tensor') ->Tuple[Tensor, Tensor]:
        """
        Forward propagate a `inputs` for audio encoder.

        Args:
            inputs (torch.FloatTensor): A input sequence passed to audio encoder. Typically inputs will be a padded
                `FloatTensor` of size ``(batch, dimension, seq_length)``.
            inputs_lens (torch.LongTensor): The length of input tensor. ``(batch)``

        Returns:
            **outputs** (Tensor): ``(batch, seq_length, dimension)``
        """
        inputs = inputs.transpose(1, 2)
        seq_len = inputs.size(1)
        self_attn_mask = get_attn_pad_mask(inputs, inputs_lens, seq_len)
        inputs = self.input_fc(inputs) + self.positional_encoding(seq_len)
        outputs = self.input_dropout(inputs)
        for encoder_layer in self.encoder_layers:
            outputs, _ = encoder_layer(outputs, self_attn_mask)
        return outputs


class LabelEncoder(nn.Module):
    """
    Converts the label to higher feature values

    Args:
        device (torch.device): flag indication whether cpu or cuda
        num_vocabs (int): the number of vocabulary
        model_dim (int): the number of features in the label encoder (default : 512)
        ff_dim (int): the number of features in the feed forward layers (default : 2048)
        num_layers (int): the number of label encoder layers (default: 2)
        num_heads (int): the number of heads in the multi-head attention (default: 8)
        dropout (float): dropout probability of label encoder (default: 0.1)
        max_len (int): Maximum length to use for positional encoding (default : 5000)
        pad_id (int): index of padding (default: 0)
        sos_id (int): index of the start of sentence (default: 1)
        eos_id (int): index of the end of sentence (default: 2)

    Inputs: inputs, inputs_lens
        - **inputs**: Ground truth of batch size number
        - **inputs_lens**: Tensor of target lengths

    Returns: outputs
        - **outputs**: Tensor containing higher feature values
    """

    def __init__(self, device: 'torch.device', num_vocabs: 'int', model_dim: 'int'=512, ff_dim: 'int'=2048, num_layers: 'int'=2, num_heads: 'int'=8, dropout: 'float'=0.1, max_len: 'int'=5000, pad_id: 'int'=0, sos_id: 'int'=1, eos_id: 'int'=2) ->None:
        super(LabelEncoder, self).__init__()
        self.device = device
        self.embedding = nn.Embedding(num_vocabs, model_dim)
        self.scale = np.sqrt(model_dim)
        self.positional_encoding = PositionalEncoding(model_dim, max_len)
        self.input_dropout = nn.Dropout(p=dropout)
        self.pad_id = pad_id
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.encoder_layers = nn.ModuleList([EncoderLayer(model_dim, ff_dim, num_heads, dropout) for _ in range(num_layers)])

    def forward(self, inputs: 'Tensor', inputs_lens: 'Tensor') ->Tuple[Tensor, Tensor]:
        """
        Forward propagate a `inputs` for label encoder.

        Args:
            inputs (torch.LongTensor): A input sequence passed to label encoder. Typically inputs will be a padded
                `LongTensor` of size ``(batch, target_length)``
            inputs_lens (torch.LongTensor): The length of input tensor. ``(batch)``

        Returns:
            **outputs** (Tensor): ``(batch, seq_length, dimension)``
        """
        self_attn_mask = None
        batch = inputs.size(0)
        if len(inputs.size()) == 1:
            inputs = inputs.unsqueeze(1)
            target_lens = inputs.size(1)
            embedding_output = self.embedding(inputs) * self.scale
            positional_encoding_output = self.positional_encoding(target_lens)
            inputs = embedding_output + positional_encoding_output
        else:
            inputs = inputs[inputs != self.eos_id].view(batch, -1)
            target_lens = inputs.size(1)
            embedding_output = self.embedding(inputs) * self.scale
            positional_encoding_output = self.positional_encoding(target_lens)
            inputs = embedding_output + positional_encoding_output
            self_attn_mask = get_attn_pad_mask(inputs, inputs_lens, target_lens)
        outputs = self.input_dropout(inputs)
        for encoder_layer in self.encoder_layers:
            outputs, _ = encoder_layer(outputs, self_attn_mask)
        return outputs


class JointNet(nn.Module):
    """
    Combine the audio encoder and label encoders.
    Convert them into log probability values for each word.

    Args:
        num_vocabs (int): the number of vocabulary
        output_size (int): the number of features combined output of audio and label encoders (default : 1024)
        inner_size (int): the number of inner features (default : 512)

    Inputs: audio_encoder, label_encoder
        - **audio_encoder**: Audio encoder output
        - **label_encoder**: Label encoder output

    Returns: output
        - **output**: Tensor expressing the log probability values of each word
    """

    def __init__(self, num_vocabs: 'int', output_size: 'int'=1024, inner_size: 'int'=512) ->None:
        super(JointNet, self).__init__()
        self.fc1 = nn.Linear(output_size, inner_size)
        self.tanh = nn.Tanh()
        self.fc2 = nn.Linear(inner_size, num_vocabs)

    def forward(self, audio_encoder: 'Tensor', label_encoder: 'Tensor') ->Tensor:
        if audio_encoder.dim() == 3 and label_encoder.dim() == 3:
            seq_lens = audio_encoder.size(1)
            target_lens = label_encoder.size(1)
            audio_encoder = audio_encoder.unsqueeze(2)
            label_encoder = label_encoder.unsqueeze(1)
            audio_encoder = audio_encoder.repeat(1, 1, target_lens, 1)
            label_encoder = label_encoder.repeat(1, seq_lens, 1, 1)
        output = torch.cat((audio_encoder, label_encoder), dim=-1)
        output = self.fc1(output)
        output = self.tanh(output)
        output = self.fc2(output)
        output = F.log_softmax(output, dim=-1)
        return output


class TransformerTransducer(nn.Module):
    """
    Transformer-Transducer is that every layer is identical for both audio and label encoders.
    Unlike the basic transformer structure, the audio encoder and label encoder are separate.
    So, the alignment is handled by a separate forward-backward process within the RNN-T architecture.
    And we replace the LSTM encoders in RNN-T architecture with Transformer encoders.

    Args:
        audio_encoder (AudioEncoder): Instance of audio encoder
        label_encoder (LabelEncoder): Instance of label encoder
        num_vocabs (int): the number of vocabulary
        output_size (int): the number of features combined output of audio and label encoders (default : 1024)
        inner_size (int): the number of inner features (default : 512)

    Inputs: inputs, input_lens, targets, targets_lens
        - **inputs** (torch.FloatTensor): A input sequence passed to audio encoder. Typically inputs will be a padded
            `FloatTensor` of size ``(batch, dimension, seq_length)``.
        - **input_lens** (torch.LongTensor): The length of input tensor. ``(batch)``
        - **targets** (torch.LongTensor): A target sequence passed to label encoder. Typically inputs will be a padded
            `LongTensor` of size ``(batch, target_length)``
        - **targets_lens** (torch.LongTensor): The length of target tensor. ``(batch)``

    Returns: output
        - **output** (torch.FloatTensor): Result of model predictions.
    """

    def __init__(self, audio_encoder: 'AudioEncoder', label_encoder: 'LabelEncoder', num_vocabs: 'int', output_size: 'int'=1024, inner_size: 'int'=512) ->None:
        super(TransformerTransducer, self).__init__()
        self.audio_encoder = audio_encoder
        self.label_encoder = label_encoder
        self.joint = JointNet(num_vocabs, output_size, inner_size)

    def forward(self, inputs: 'Tensor', input_lens: 'Tensor', targets: 'Tensor', targets_lens: 'Tensor') ->Tensor:
        """
        Forward propagate a `inputs, targets` for transformer transducer.

        Args:
            inputs (torch.FloatTensor): A input sequence passed to audio encoder. Typically inputs will be a padded
                `FloatTensor` of size ``(batch, dimension, seq_length)``.
            input_lens (torch.LongTensor): The length of input tensor. ``(batch)``
            targets (torch.LongTensor): A target sequence passed to label encoder. Typically inputs will be a padded
                `LongTensor` of size ``(batch, target_length)``
            targets_lens (torch.LongTensor): The length of target tensor. ``(batch)``

        Returns:
            **output** (Tensor): ``(batch, seq_length, num_vocabs)``
        """
        audio_output = self.audio_encoder(inputs, input_lens)
        label_output = self.label_encoder(targets, targets_lens)
        output = self.joint(audio_output, label_output)
        return output

    @torch.no_grad()
    def decode(self, audio_outputs: 'Tensor', max_lens: 'int') ->Tensor:
        batch = audio_outputs.size(0)
        y_hats = list()
        targets = torch.LongTensor([self.label_encoder.sos_id] * batch)
        if torch.cuda.is_available():
            targets = targets
        for i in range(max_lens):
            label_output = self.label_encoder(targets, None)
            label_output = label_output.squeeze(1)
            audio_output = audio_outputs[:, i, :]
            output = self.joint(audio_output, label_output)
            targets = output.max(1)[1]
            y_hats.append(targets)
        y_hats = torch.stack(y_hats, dim=1)
        return y_hats

    @torch.no_grad()
    def recognize(self, inputs: 'Tensor', inputs_lens: 'Tensor') ->Tensor:
        audio_outputs = self.audio_encoder(inputs, inputs_lens)
        max_lens = audio_outputs.size(1)
        return self.decode(audio_outputs, max_lens)


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (MultiHeadAttention,
     lambda: ([], {'model_dim': 4, 'num_heads': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (PositionalEncoding,
     lambda: ([], {}),
     lambda: ([0], {})),
    (ScaledDotProductAttention,
     lambda: ([], {'key_dim': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {})),
]

