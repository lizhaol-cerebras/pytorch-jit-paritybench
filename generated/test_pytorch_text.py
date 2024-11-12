
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


import time


import torch


from collections import Counter


from collections import OrderedDict


from typing import List


from typing import Union


from torch.nn.functional import multi_head_attention_forward as mha_forward


import re


from torch.hub import load_state_dict_from_url


from torch.nn import Module


from torch.utils.data import DataLoader


from functools import partial


from typing import Dict


from typing import Any


import torch.nn as nn


import logging


from torch.utils.data.dataset import random_split


import functools


from torch.optim import AdamW


import random


from itertools import zip_longest


from torch.testing._internal.common_utils import TestCase


from torch.utils.data.datapipes.iter import IterableWrapper


import uuid


from torch.testing import assert_close


from torch.nn import Linear


from torch.utils.data.graph import traverse_dps


from torch.utils.data.graph_settings import get_all_graph_pipes


import copy


from torch.nn import functional as torch_F


from torch.nn import functional as F


from typing import Optional


from torch.hub import _get_torch_home


import inspect


from torch.utils.data import functional_datapipe


from torch.utils.data import IterDataPipe


from torch.utils.data.datapipes.utils.common import StreamWrapper


import collections


import math


from copy import deepcopy


import warnings


from torch import Tensor


from torch.nn.utils.rnn import pad_sequence


from typing import Callable


from torch import nn


from typing import Tuple


import torch.nn.functional as F


from torch.nn.modules.linear import NonDynamicallyQuantizableLinear


from abc import abstractmethod


from typing import Final


from functools import lru_cache


from typing import Mapping


from typing import Sequence


LOCK_TIMEOUT = 600


def _get_response_from_google_drive(url):
    confirm_token = None
    session = requests.Session()
    response = session.get(url, stream=True)
    for k, v in response.cookies.items():
        if k.startswith('download_warning'):
            confirm_token = v
    if confirm_token is None:
        if 'Quota exceeded' in str(response.content):
            raise RuntimeError('Google drive link {} is currently unavailable, because the quota was exceeded.'.format(url))
        else:
            raise RuntimeError('Internal error: confirm_token was not found in Google drive link.')
    url = url + '&confirm=' + confirm_token
    response = session.get(url, stream=True)
    if 'content-disposition' not in response.headers:
        raise RuntimeError("Internal error: headers don't contain content-disposition.")
    filename = re.findall('filename="(.+)"', response.headers['content-disposition'])
    if filename is None:
        raise RuntimeError('Filename could not be autodetected')
    filename = filename[0]
    return response, filename


def _stream_response(r, chunk_size=16 * 1024):
    total_size = int(r.headers.get('Content-length', 0))
    with tqdm(total=total_size, unit='B', unit_scale=1) as t:
        for chunk in r.iter_content(chunk_size):
            if chunk:
                t.update(len(chunk))
                yield chunk


class DownloadManager:

    def get_local_path(self, url, destination):
        if 'drive.google.com' not in url:
            response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, stream=True)
        else:
            response, filename = _get_response_from_google_drive(url)
        with open(destination, 'wb') as f:
            for chunk in _stream_response(response):
                f.write(chunk)


_DATASET_DOWNLOAD_MANAGER = DownloadManager()


logger = logging.getLogger(__name__)


def validate_file(file_obj, hash_value, hash_type='sha256'):
    """Validate a given file object with its hash.
    Args:
        file_obj: File object to read from.
        hash_value (str): Hash for url.
        hash_type (str, optional): Hash type, among "sha256" and "md5" (Default: ``"sha256"``).
    Returns:
        bool: return True if its a valid file, else False.
    """
    if hash_type in ('sha256', 'md5'):
        hash_func = hashlib.new(hash_type, usedforsecurity=False)
    else:
        raise ValueError
    while True:
        chunk = file_obj.read(1024 ** 2)
        if not chunk:
            break
        hash_func.update(chunk)
    return hash_func.hexdigest() == hash_value


def _check_hash(path, hash_value, hash_type):
    logger.info('Validating hash {} matches hash of {}'.format(hash_value, path))
    with open(path, 'rb') as file_obj:
        if not validate_file(file_obj, hash_value, hash_type):
            raise RuntimeError('The hash of {} does not match. Delete the file manually and retry.'.format(os.path.abspath(path)))


def get_lock_dir():
    lock_dir = os.path.join(_CACHE_DIR, 'locks')
    if not os.path.exists(lock_dir):
        os.makedirs(lock_dir, exist_ok=True)
    return lock_dir


def download_from_url(url, path=None, root='.data', overwrite=False, hash_value=None, hash_type='sha256'):
    """Download file, with logic (from tensor2tensor) for Google Drive. Returns
    the path to the downloaded file.
    Args:
        url: the url of the file from URL header. (None)
        path: path where file will be saved
        root: download folder used to store the file in (.data)
        overwrite: overwrite existing files (False)
        hash_value (str, optional): hash for url (Default: ``None``).
        hash_type (str, optional): hash type, among "sha256" and "md5" (Default: ``"sha256"``).
    Examples:
        >>> url = 'http://www.quest.dcs.shef.ac.uk/wmt16_files_mmt/validation.tar.gz'
        >>> torchtext.utils.download_from_url(url)
        >>> url = 'http://www.quest.dcs.shef.ac.uk/wmt16_files_mmt/validation.tar.gz'
        >>> torchtext.utils.download_from_url(url)
        >>> '.data/validation.tar.gz'
    """
    if path is None:
        _, filename = os.path.split(url)
        root = os.path.abspath(root)
        path = os.path.join(root, filename)
    else:
        path = os.path.abspath(path)
        root, filename = os.path.split(os.path.abspath(path))
    lock_dir = get_lock_dir()
    lock = FileLock(os.path.join(lock_dir, filename + '.lock'), timeout=LOCK_TIMEOUT)
    with lock:
        if os.path.exists(path):
            logger.info('File %s already exists.' % path)
            if not overwrite:
                if hash_value:
                    _check_hash(path, hash_value, hash_type)
                return path
        if not os.path.exists(root):
            try:
                os.makedirs(root)
            except OSError:
                raise OSError("Can't create the download directory {}.".format(root))
        _DATASET_DOWNLOAD_MANAGER.get_local_path(url, destination=path)
        logger.info('File {} downloaded.'.format(path))
        if hash_value:
            _check_hash(path, hash_value, hash_type)
        return path


def get_asset_local_path(asset_path: 'str', overwrite=False) ->str:
    """Get local path for assets. Download if path does not exist locally
    Args:
        asset_path: Local path to asset or remote URL
        overwrite: Indicate whether to overwrite the file when downloading from URL (default: False)
    Returns:
        bool: local path of the asset after downloading or reading from cache
    Examples:
        >>> url = 'http://<HOST>/file.txt'
        >>> torchtext.utils.get_asset_local_path(url)
        >>> '.data/file.txt'
        >>> torchtext.utils.get_asset_local_path('/home/user/file.txt')
        >>> '/home/user/file.txt'
    """
    if os.path.exists(asset_path):
        local_path = asset_path
    else:
        local_path = download_from_url(url=asset_path, root=_CACHE_DIR, overwrite=overwrite)
    return local_path


def init_ta_gpt2bpe_encoder():
    encoder_json_path = 'https://download.pytorch.org/models/text/gpt2_bpe_encoder.json'
    vocab_bpe_path = 'https://download.pytorch.org/models/text/gpt2_bpe_vocab.bpe'
    encoder_json_path = get_asset_local_path(encoder_json_path)
    vocab_bpe_path = get_asset_local_path(vocab_bpe_path)
    _seperator = '\x01'
    with open(encoder_json_path, 'r', encoding='utf-8') as f:
        bpe_encoder = json.load(f)
    with open(vocab_bpe_path, 'r', encoding='utf-8') as f:
        bpe_vocab = f.read()
    bpe_merge_ranks = {_seperator.join(merge_pair.split()): i for i, merge_pair in enumerate(bpe_vocab.split('\n')[1:-1])}
    bpe = _ta.GPT2BPEEncoder(bpe_encoder, bpe_merge_ranks, _seperator, T.bytes_to_unicode(), True)
    return bpe


def init_ta_gpt2bpe_vocab():
    vocab_path = 'https://download.pytorch.org/models/text/roberta.vocab.pt'
    vocab_path = get_asset_local_path(vocab_path)
    vocab = torch.load(vocab_path)
    ta_vocab = _ta.Vocab(vocab.get_itos(), vocab.get_default_index())
    return ta_vocab


class RobertaTransformDataPipe(Module):

    def __init__(self) ->None:
        super().__init__()
        encoder_json_path = 'https://download.pytorch.org/models/text/gpt2_bpe_encoder.json'
        vocab_bpe_path = 'https://download.pytorch.org/models/text/gpt2_bpe_vocab.bpe'
        self.tokenizer = T.GPT2BPETokenizer(encoder_json_path, vocab_bpe_path)
        vocab_path = 'https://download.pytorch.org/models/text/roberta.vocab.pt'
        self.vocab = T.VocabTransform(load_state_dict_from_url(vocab_path))
        self.add_bos = T.AddToken(token=0, begin=True)
        self.add_eos = T.AddToken(token=2, begin=False)

    def forward(self, input: 'Dict[str, Any]') ->Dict[str, Any]:
        tokens = self.tokenizer(input['text'])
        tokens = F.truncate(tokens, max_seq_len=254)
        tokens = self.vocab(tokens)
        tokens = self.add_bos(tokens)
        tokens = self.add_eos(tokens)
        input['tokens'] = tokens
        return input


class TextClassificationModel(nn.Module):

    def __init__(self, vocab_size, embed_dim, num_class):
        super(TextClassificationModel, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
        self.fc = nn.Linear(embed_dim, num_class)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        return self.fc(embedded)


def load_sp_model(sp_model):
    """Load a  sentencepiece model for file.

    Args:
        sp_model: the file path or a file object saving the sentencepiece model.

    Outputs:
        output: a SentencePiece model.

    Examples:
        >>> from torchtext.experimental.transforms import load_sp_model
        >>> sp_model = load_sp_model("m_user.model")
        >>> sp_model = load_sp_model(open("m_user.model", 'rb'))

    Note: We also provide several pretrained sentencepiece models. The model was trained with torchtext.datasets.WikiText103,
        torchtext.datasets.EnWik9 and BookCorpus. Both BPE and unigram methods were used to train the model (for more
        details please refer to SentencePiece GitHub https://github.com/google/sentencepiece). We also provide the pretrained model
        with a different size of the vocabulary (i.e. 15000, 25000, 50000).
        The following pretrained sentencepiece models are provided:

            - text_unigram_15000
            - text_unigram_25000
            - text_unigram_50000
            - text_bpe_15000
            - text_bpe_25000
            - text_bpe_50000

    Examples:
        >>> from torchtext.experimental.transforms import PRETRAINED_SP_MODEL
        >>> sp_model_path = torchtext.utils.download_from_url(PRETRAINED_SP_MODEL['text_unigram_25000'])
        >>> sp_model = load_sp_model(sp_model_path)
    """
    if isinstance(sp_model, str):
        with open(sp_model, 'rb') as f:
            return SentencePiecePybind(f.read())
    elif isinstance(sp_model, io.BufferedReader):
        return SentencePiecePybind(sp_model.read())
    else:
        raise TypeError(f'Unsupported type for sp_model argument: {type(sp_model).__name__}. ' + 'Supported types are: ' + ', '.join(['str', 'io.BufferedReader']))


class ScriptableSP(torch.jit.ScriptModule):

    def __init__(self, model_path) ->None:
        super().__init__()
        self.spm = load_sp_model(model_path)

    @torch.jit.script_method
    def encode(self, input: 'str'):
        return self.spm.Encode(input)

    @torch.jit.script_method
    def encode_as_ids(self, input: 'str'):
        return self.spm.EncodeAsIds(input)

    @torch.jit.script_method
    def encode_as_pieces(self, input: 'str'):
        return self.spm.EncodeAsPieces(input)


class PositionalEmbedding(Module):

    def __init__(self, num_embeddings: 'int', embedding_dim: 'int', pad_index: 'int') ->None:
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim, pad_index)
        self.pad_index = pad_index

    def forward(self, input):
        positions = self._make_positions(input, self.pad_index)
        return self.embedding(positions)

    def max_positions(self):
        if self.pad_index is not None:
            return self.num_embeddings - self.pad_index - 1
        else:
            return self.num_embeddings

    def _make_positions(self, tensor, pad_index: 'int'):
        masked = tensor.ne(pad_index).long()
        return torch.cumsum(masked, dim=1) * masked + pad_index


class TransformerEncoder(Module):

    def __init__(self, vocab_size: 'int', embedding_dim: 'int', padding_idx: 'int', max_seq_len: 'int', num_encoder_layers: 'int', num_attention_heads: 'int', ffn_dimension: 'Optional[int]'=None, dropout: 'float'=0.1, normalize_before: 'bool'=False, scaling: 'Optional[float]'=None, return_all_layers: 'bool'=False) ->None:
        super().__init__()
        self.padding_idx = padding_idx
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx)
        ffn_dimension = ffn_dimension or 4 * embedding_dim
        layer = torch.nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=num_attention_heads, dim_feedforward=ffn_dimension, dropout=dropout, activation='gelu', batch_first=True, norm_first=normalize_before)
        self.layers = torch.nn.TransformerEncoder(encoder_layer=layer, num_layers=num_encoder_layers, enable_nested_tensor=True, mask_check=False)
        self.positional_embedding = PositionalEmbedding(max_seq_len, embedding_dim, padding_idx)
        self.embedding_layer_norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.normalize_before = normalize_before
        self.return_all_layers = return_all_layers

    def forward(self, tokens: 'torch.Tensor', attn_mask: 'Optional[torch.Tensor]'=None) ->Union[torch.Tensor, List[torch.Tensor]]:
        if attn_mask is not None:
            torch._assert(attn_mask.is_floating_point() or attn_mask.dtype == torch.bool, f'Only float or bool types are supported for attn_mask not {attn_mask.dtype}')
        padding_mask = tokens.eq(self.padding_idx)
        token_embeddings = self.token_embedding(tokens)
        embedded_positions = self.positional_embedding(tokens)
        embedded = token_embeddings + embedded_positions
        if not hasattr(self, 'normalize_before'):
            self.normalize_before = False
        if not self.normalize_before:
            embedded = self.embedding_layer_norm(embedded)
        embedded = self.dropout(embedded)
        if self.return_all_layers:
            encoded = embedded
            states = [encoded.transpose(1, 0)]
            for layer in self.layers.layers:
                encoded = layer(encoded, src_key_padding_mask=padding_mask, src_mask=attn_mask)
                encoded_t = encoded.transpose(1, 0)
                states.append(encoded_t)
            if self.normalize_before:
                for i, state in enumerate(states):
                    states[i] = self.embedding_layer_norm(state)
            return states
        else:
            encoded = self.layers(embedded, src_key_padding_mask=padding_mask).transpose(1, 0)
            if self.normalize_before:
                encoded = self.embedding_layer_norm(encoded)
            return encoded

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        better_to_old_names = {'self_attn.in_proj_weight': 'attention.input_projection.weight', 'self_attn.in_proj_bias': 'attention.input_projection.bias', 'self_attn.out_proj.weight': 'attention.output_projection.weight', 'self_attn.out_proj.bias': 'attention.output_projection.bias', 'linear1.weight': 'residual_mlp.mlp.0.weight', 'linear1.bias': 'residual_mlp.mlp.0.bias', 'linear2.weight': 'residual_mlp.mlp.3.weight', 'linear2.bias': 'residual_mlp.mlp.3.bias', 'norm1.weight': 'attention_layer_norm.weight', 'norm1.bias': 'attention_layer_norm.bias', 'norm2.weight': 'final_layer_norm.weight', 'norm2.bias': 'final_layer_norm.bias'}
        for i in range(self.layers.num_layers):
            for better, old in better_to_old_names.items():
                better_name = prefix + 'layers.layers.{}.'.format(i) + better
                old_name = prefix + 'layers.{}.'.format(i) + old
                if old_name in state_dict:
                    state_dict[better_name] = state_dict[old_name]
                    state_dict.pop(old_name)
                elif better_name in state_dict:
                    pass
                elif strict:
                    missing_keys.append(better_name)
        super(TransformerEncoder, self)._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)


class RobertaEncoder(Module):

    def __init__(self, vocab_size: 'int', embedding_dim: 'int', ffn_dimension: 'int', padding_idx: 'int', max_seq_len: 'int', num_attention_heads: 'int', num_encoder_layers: 'int', dropout: 'float'=0.1, scaling: 'Optional[float]'=None, normalize_before: 'bool'=False, freeze: 'bool'=False) ->None:
        super().__init__()
        if not scaling:
            head_dim = embedding_dim // num_attention_heads
            scaling = 1.0 / math.sqrt(head_dim)
        self.transformer = TransformerEncoder(vocab_size=vocab_size, embedding_dim=embedding_dim, padding_idx=padding_idx, max_seq_len=max_seq_len, ffn_dimension=ffn_dimension, num_encoder_layers=num_encoder_layers, num_attention_heads=num_attention_heads, dropout=dropout, normalize_before=normalize_before, scaling=scaling, return_all_layers=False)
        if freeze:
            for p in self.parameters():
                p.requires_grad = False

    def forward(self, tokens: 'Tensor', masked_tokens: 'Optional[Tensor]'=None) ->Tensor:
        output = self.transformer(tokens)
        if torch.jit.isinstance(output, List[Tensor]):
            output = output[-1]
        output = output.transpose(1, 0)
        if masked_tokens is not None:
            output = output[masked_tokens, :]
        return output


class RobertaClassificationHead(nn.Module):

    def __init__(self, num_classes, input_dim, inner_dim: 'Optional[int]'=None, dropout: 'float'=0.1, activation=nn.ReLU) ->None:
        super().__init__()
        if not inner_dim:
            inner_dim = input_dim
        self.dense = nn.Linear(input_dim, inner_dim)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(inner_dim, num_classes)
        self.activation_fn = activation()

    def forward(self, features):
        x = features[:, 0, :]
        x = self.dropout(x)
        x = self.dense(x)
        x = self.activation_fn(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class RobertaModel(Module):
    """

    Example - Instatiating model object
        >>> from torchtext.models import RobertaEncoderConf, RobertaModel, RobertaClassificationHead
        >>> roberta_encoder_conf = RobertaEncoderConf(vocab_size=250002)
        >>> encoder = RobertaModel(config=roberta_encoder_conf)
        >>> classifier_head = RobertaClassificationHead(num_classes=2, input_dim=768)
        >>> classifier = RobertaModel(config=roberta_encoder_conf, head=classifier_head)
    """

    def __init__(self, encoder_conf: 'RobertaEncoderConf', head: 'Optional[Module]'=None, freeze_encoder: 'bool'=False) ->None:
        super().__init__()
        assert isinstance(encoder_conf, RobertaEncoderConf)
        self.encoder = RobertaEncoder(**asdict(encoder_conf), freeze=freeze_encoder)
        self.head = head

    def forward(self, tokens: 'Tensor', masked_tokens: 'Optional[Tensor]'=None) ->Tensor:
        features = self.encoder(tokens, masked_tokens)
        if self.head is None:
            return features
        x = self.head(features)
        return x


class TransformerEncoderLayer(Module):

    def __init__(self, embedding_dim: 'int', num_attention_heads: 'int', ffn_dimension: 'Optional[int]'=None, dropout: 'float'=0.1, normalize_before: 'bool'=False, scaling: 'Optional[float]'=None) ->None:
        super().__init__()
        ffn_dimension = ffn_dimension or embedding_dim * 4
        self.better_transformer = torch.nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=num_attention_heads, dim_feedforward=ffn_dimension, dropout=dropout, batch_first=True, activation='gelu', norm_first=normalize_before)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        better_to_old_names = {'better_transformer.self_attn.in_proj_weight': 'attention.input_projection.weight', 'better_transformer.self_attn.in_proj_bias': 'attention.input_projection.bias', 'better_transformer.self_attn.out_proj.weight': 'attention.output_projection.weight', 'better_transformer.self_attn.out_proj.bias': 'attention.output_projection.bias', 'better_transformer.linear1.weight': 'residual_mlp.mlp.0.weight', 'better_transformer.linear1.bias': 'residual_mlp.mlp.0.bias', 'better_transformer.linear2.weight': 'residual_mlp.mlp.3.weight', 'better_transformer.linear2.bias': 'residual_mlp.mlp.3.bias', 'better_transformer.norm1.weight': 'attention_layer_norm.weight', 'better_transformer.norm1.bias': 'attention_layer_norm.bias', 'better_transformer.norm2.weight': 'final_layer_norm.weight', 'better_transformer.norm2.bias': 'final_layer_norm.bias'}
        for better, old in better_to_old_names.items():
            better_name = prefix + better
            old_name = prefix + old
            if old_name in state_dict:
                state_dict[better_name] = state_dict[old_name]
                state_dict.pop(old_name)
            elif better_name in state_dict:
                pass
            elif strict:
                missing_keys.append(better_name)
        super(TransformerEncoderLayer, self)._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

    def forward(self, input: 'torch.Tensor', key_padding_mask: 'torch.Tensor', attn_mask: 'Optional[torch.Tensor]'=None):
        return self.better_transformer(input.transpose(0, 1), attn_mask, key_padding_mask).transpose(0, 1)


PAST_KEY_VALUES_TYPE = Tuple[Tensor, Tensor, Tensor, Tensor]


PAST_KEY_VALUES_UNFILLED_TYPE = Tuple[Tensor, Tensor, Optional[Tensor], Optional[Tensor]]


SEQ_2_SEQ_OUTPUTS_TYPE = Dict[str, Union[Optional[Tensor], List[Tensor], List[Optional[Tensor]], Optional[List[PAST_KEY_VALUES_UNFILLED_TYPE]]]]


PAST_KEY_VALUE_TYPE = Tuple[Tensor, Tensor]


class T5LayerNorm(nn.Module):

    def __init__(self, d_model: 'int', eps: 'float'=1e-06) ->None:
        """Construct a layernorm module in the T5 style. No bias and no subtraction of mean.

        Based on https://github.com/huggingface/transformers/blob/8581a798c0a48fca07b29ce2ca2ef55adcae8c7e/src/transformers/models/t5/modeling_t5.py#L239.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.variance_epsilon = eps

    def forward(self, hidden_states: 'Tensor') ->Tensor:
        """T5 uses a layer_norm which only scales and doesn't shift, which is also known as Root Mean
        Square Layer Normalization https://arxiv.org/abs/1910.07467 thus varience is calculated
        w/o mean and there is no bias. Additionally we want to make sure that the accumulation for
        half-precision inputs is done in fp32.

        Args:
            hidden_states: Tensor to be normalized. Final dimension must be model dimension (i.e. number of expected features in the input).

        Returns:
            Tensor with the same shape as hidden_states after having been normalized.
        """
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states
        return self.weight * hidden_states


class T5MultiheadAttention(nn.MultiheadAttention):

    def __init__(self, embed_dim: 'int', num_heads: 'int', is_decoder: 'bool'=False, dropout: 'float'=0.0, bias: 'bool'=False, qkv_dim: 'int'=64, compute_relative_attention_bias: 'bool'=False, relative_attention_num_buckets: 'int'=32, relative_attention_max_distance: 'int'=128, device: 'Optional[torch.device]'=None, dtype=None) ->None:
        """T5MultiheadAttention based on `nn.MultiheadAttention`.

        Args:
            embed_dim: Total dimension of the model.
            num_heads: Parallel attention heads.
            is_decoder: Whether or not multihead attention is being performed on a decoder layer. Default: `False`
            dropout: Probability of an element to be zeroed. Default: 0.0
            bias: If specified, adds bias to input / output projection layers. Default: `False`.
            qkv_dim: Projection dimension (per head) for query, keys, and values. Defualt: 64.
            compute_relative_attention_bias: Whether or not the relative position embeddings
                need to be computed. Wypically occurs in the first layer of the encoder/decoder
                and the resulting position embeddings are returned to be passed up to higher layers. (defualt: False)
            relative_attention_num_buckets: Number of relative position buckets. Default: `32`
            relative_attention_max_distance: Maximum threshold on the relative distance used to
                allocate buckets. Anything larger gets placed in the same bucket. Default: `128`
        """
        super().__init__(embed_dim, num_heads, dropout, bias, False, False, qkv_dim, qkv_dim, True, device, dtype)
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.is_decoder = is_decoder
        self.inner_dim = qkv_dim * num_heads
        self.q_proj_weight = nn.Parameter(torch.empty((self.inner_dim, embed_dim), **factory_kwargs))
        self.k_proj_weight = nn.Parameter(torch.empty((self.inner_dim, embed_dim), **factory_kwargs))
        self.v_proj_weight = nn.Parameter(torch.empty((self.inner_dim, embed_dim), **factory_kwargs))
        self.out_proj = NonDynamicallyQuantizableLinear(self.inner_dim, embed_dim, bias=bias, **factory_kwargs)
        self.register_parameter('in_proj_weight', None)
        self.compute_relative_attention_bias = compute_relative_attention_bias
        self.relative_attention_num_buckets = relative_attention_num_buckets
        self.relative_attention_max_distance = relative_attention_max_distance
        if compute_relative_attention_bias:
            self.relative_attention_bias = nn.Embedding(relative_attention_num_buckets, num_heads)
        else:
            self.relative_attention_bias = None

    def forward(self, query: 'Tensor', key: 'Tensor', value: 'Tensor', query_length: 'Optional[int]'=None, key_padding_mask: 'Optional[Tensor]'=None, need_weights: 'bool'=True, attn_mask: 'Optional[Tensor]'=None, average_attn_weights: 'bool'=False, position_bias: 'Optional[Tensor]'=None, past_key_value: 'Optional[PAST_KEY_VALUE_TYPE]'=None) ->Tuple[Tensor, Tensor, Optional[Tensor], PAST_KEY_VALUE_TYPE]:
        """Allows the model to jointly attend to information from different representation subspaces
        as described in the paper: `Attention Is All You Need <https://arxiv.org/abs/1706.03762>`.
        Also incorporates relative attention bias when computing attention scores as descripted in the paper:
        `Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer <https://arxiv.org/pdf/1910.10683.pdf>`.

        Args:
            query: Query embeddings of shape :math:`(N, L, E_q)`, where :math:`N` is the batch size, :math:`L` is the target sequence length,
                and :math:`E_q` is the query embedding dimension `embed_dim`.
                Queries are compared against key-value pairs to produce the output.
                See "Attention Is All You Need" for more details.
            key: Key embeddings of shape :math:`(N, S, E_k)`, where :math:`N` is the batch size, :math:`S` is the source sequence length,
                and :math:`E_k` is the key embedding dimension `kdim`.
                See "Attention Is All You Need" for more details.
            value: Value embeddings of shape :math:`(N, S, E_v)`, where :math:`N` is the batch size, :math:`S` is the source
                sequence length, and :math:`E_v` is the value embedding dimension `vdim`.
                See "Attention Is All You Need" for more details.
            key_padding_mask: If specified, a mask of shape :math:`(N, S)` indicating which elements within `key`
                to ignore for the purpose of attention (i.e. treat as "padding").
                Binary masks are supported. For a binary mask, a `True` value indicates that the corresponding `key`
                value will be ignored for the purpose of attention.
            need_weights: If specified, returns `attn_output_weights` in addition to `attn_outputs`.
                Default: `True`.
            attn_mask: If specified, a 2D mask preventing attention to certain positions. Must be of shape
                :math:`(L, S)`, :math:`L` is the target sequence length, and :math:`S` is the source sequence length. A 2D mask will be
                broadcasted across the batch. Binary, and float masks are supported.
                For a binary mask, a `True` value indicates that the corresponding position is not allowed to attend.
                For a float mask, the mask values will be added to the attention weight. Default: `None`
            average_attn_weights: If true, indicates that the returned `attn_weights` should be averaged across
                heads. Otherwise, `attn_weights` are provided separately per head. Note that this flag only has an
                effect when `need_weights=True`. Default: `False` (i.e. average weights across heads)
            position_bias: Position bias tensor used if to add relative attention bias to attention scores. Default: `None`

        Returns:
            attn_output: Attention outputs of shape :math:`(N, L, E)`, where :math:`N` is the batch size,
                :math:`L` is the target sequence length, and :math:`E` is the embedding dimension `embed_dim`.
            attn_output_weights: Only returned when `need_weights=True`. If `average_attn_weights=True`,
              returns attention weights averaged across heads of shape :math:`(L, S)` when input is unbatched or
              :math:`(N, L, S)`, where :math:`N` is the batch size, :math:`L` is the target sequence length, and
              :math:`S` is the source sequence length. If `average_weights=False`, returns attention weights per
              head of shape :math:`(\\text{num\\_heads}, L, S)` when input is unbatched or :math:`(N, \\text{num\\_heads}, L, S)`.
            position_bias: Used in attention scoring. Only computed when `compute_relative_attention_bias=True`
                and `position_bias=None`. Has shape :math:`(1, num_heads, L, S)`.
            key_value: Calculated weights for keys and values. Used for incremental decoding.
        """
        attn_output, position_bias, attn_output_weights, key_value = self._t5_multi_head_attention_forward(query, key, value, query_length=query_length, position_bias=position_bias, key_padding_mask=key_padding_mask, need_weights=need_weights, attn_mask=attn_mask, average_attn_weights=average_attn_weights, past_key_value=past_key_value)
        return attn_output, position_bias, attn_output_weights, key_value

    def _t5_multi_head_attention_forward(self, query: 'Tensor', key: 'Tensor', value: 'Tensor', query_length: 'Optional[int]', position_bias: 'Optional[Tensor]', key_padding_mask: 'Optional[Tensor]'=None, need_weights: 'bool'=True, attn_mask: 'Optional[Tensor]'=None, average_attn_weights: 'bool'=False, past_key_value: 'Optional[PAST_KEY_VALUE_TYPE]'=None) ->Tuple[Tensor, Tensor, Optional[Tensor], PAST_KEY_VALUE_TYPE]:
        """Modified from https://github.com/pytorch/pytorch/blob/5953fd9133c0bdcc0158acf1472fac403bc5f636/torch/nn/functional.py#L4909."""
        is_self_attention = torch.equal(query, key)
        is_batched = F._mha_shape_check(query, key, value, key_padding_mask, attn_mask, self.num_heads)
        if not is_batched:
            query = query.unsqueeze(1)
            key = key.unsqueeze(1)
            value = value.unsqueeze(1)
            if key_padding_mask is not None:
                key_padding_mask = key_padding_mask.unsqueeze(0)
        bsz, tgt_len, embed_dim = query.shape
        real_seq_length = tgt_len
        if past_key_value is not None:
            assert len(past_key_value) == 2, f'past_key_value should have 2 past states: keys and values. Got {len(past_key_value)} past states'
            real_seq_length += past_key_value[0].shape[2] if query_length is None else query_length
        src_len = real_seq_length if is_self_attention else key.shape[1]
        assert embed_dim == self.embed_dim, f'was expecting embedding dimension of {self.embed_dim}, but got {embed_dim}'
        head_dim = self.inner_dim // self.num_heads
        assert key.shape[:2] == value.shape[:2], f"key's sequence and batch dims {key.shape[:2]} do not match value's {value.shape[:2]}"
        assert self.q_proj_weight is not None, 'q_proj_weight is None'
        assert self.k_proj_weight is not None, 'k_proj_weight is None'
        assert self.v_proj_weight is not None, 'v_proj_weight is None'
        if self.in_proj_bias is None:
            b_q = b_k = b_v = None
        else:
            b_q, b_k, b_v = self.in_proj_bias.chunk(3)
        q, k, v = self._t5_in_projection(query, key, value, bsz, head_dim, self.q_proj_weight, self.k_proj_weight, self.v_proj_weight, b_q, b_k, b_v, is_self_attention, past_key_value)
        if attn_mask is None:
            if self.is_decoder:
                if is_self_attention:
                    attn_mask = torch.triu(torch.ones((tgt_len, tgt_len), dtype=torch.bool, device=query.device), diagonal=1)
                else:
                    attn_mask = torch.zeros((tgt_len, src_len), device=query.device)
            else:
                attn_mask = torch.zeros((src_len, src_len), device=query.device, dtype=torch.bool)
        if attn_mask is not None:
            if attn_mask.dtype == torch.uint8:
                warnings.warn('Byte tensor for attn_mask is not supported. Using bool tensor instead.')
                attn_mask = attn_mask
            else:
                assert attn_mask.is_floating_point() or attn_mask.dtype == torch.bool, f'Only float and bool types are supported for attn_mask, not {attn_mask.dtype}'
            if attn_mask.dim() == 2:
                x, y = attn_mask.shape
                attn_mask = attn_mask.view(1, 1, x, y).expand(bsz, self.num_heads, -1, -1)
            else:
                raise RuntimeError(f"attn_mask's dimension {attn_mask.dim()} is not supported")
        if key_padding_mask is not None and key_padding_mask.dtype == torch.uint8:
            warnings.warn('Byte tensor for key_padding_mask is not supported. Using bool tensor instead.')
            key_padding_mask = key_padding_mask
        q = q.view(bsz, -1, self.num_heads, head_dim).transpose(1, 2)
        if past_key_value is None:
            k = k.view(bsz, -1, self.num_heads, head_dim).transpose(1, 2)
            v = v.view(bsz, -1, self.num_heads, head_dim).transpose(1, 2)
        src_len = k.size(2)
        if key_padding_mask is not None:
            if key_padding_mask.shape != (bsz, src_len):
                assert past_key_value is not None, 'Must provide past_key_value if key_padding_mask needs to be expanded.'
                key_padding_mask = key_padding_mask.expand(bsz, src_len)
            assert key_padding_mask.shape == (bsz, src_len), f'expecting key_padding_mask shape of {bsz, src_len}, but got {key_padding_mask.shape}'
            key_padding_mask = key_padding_mask.view(bsz, 1, 1, src_len).expand(-1, self.num_heads, tgt_len, -1)
            if attn_mask is None:
                attn_mask = key_padding_mask
            elif attn_mask.dtype == torch.bool:
                attn_mask = attn_mask.logical_or(key_padding_mask)
            else:
                attn_mask = attn_mask.masked_fill(key_padding_mask, float('-inf'))
        if attn_mask is not None and attn_mask.dtype == torch.bool:
            tmp_attn_mask = torch.zeros_like(attn_mask, dtype=q.dtype)
            attn_mask = tmp_attn_mask.masked_fill(attn_mask, float('-inf'))
        if not self.training:
            dropout_p = 0.0
        else:
            dropout_p = self.dropout
        if position_bias is None:
            if not self.compute_relative_attention_bias:
                position_bias = torch.zeros((1, self.num_heads, real_seq_length, src_len), device=k.device, dtype=k.dtype)
            else:
                position_bias = self._compute_bias(real_seq_length, src_len, bidirectional=not self.is_decoder, device=k.device)
            if past_key_value is not None:
                position_bias = position_bias[:, :, -query.size(1):, :]
        new_key_val = k, v
        attn_output, attn_output_weights = self._t5_dot_product_attention(q, k, v, position_bias, attn_mask, dropout_p)
        attn_output = F.linear(attn_output, self.out_proj.weight, self.out_proj.bias)
        if need_weights:
            if average_attn_weights:
                attn_output_weights = attn_output_weights.sum(dim=1) / self.num_heads
            if not is_batched:
                attn_output = attn_output.squeeze(1)
                attn_output_weights = attn_output_weights.squeeze(0)
            return attn_output, position_bias, attn_output_weights, new_key_val
        else:
            if not is_batched:
                attn_output = attn_output.squeeze(1)
            return attn_output, position_bias, None, new_key_val

    def _t5_in_projection(self, query: 'Tensor', key: 'Tensor', value: 'Tensor', bsz: 'int', head_dim: 'int', w_q: 'Tensor', w_k: 'Tensor', w_v: 'Tensor', b_q: 'Optional[Tensor]'=None, b_k: 'Optional[Tensor]'=None, b_v: 'Optional[Tensor]'=None, is_self_attention: 'bool'=True, past_key_value: 'Optional[PAST_KEY_VALUE_TYPE]'=None) ->Tuple[Tensor, Tensor, Tensor]:
        """Performs the in-projection step of the attention operation. This is simply
        a triple of linear projections, with shape constraints on the weights which
        ensure embedding dimension uniformity in the projected outputs.
        Output is a triple containing projection tensors for query, key and value.

        Modified from https://github.com/pytorch/pytorch/blob/5953fd9133c0bdcc0158acf1472fac403bc5f636/torch/nn/functional.py#L4761.

        Args:
            q, k, v: query, key and value tensors to be projected.
            w_q, w_k, w_v: weights for q, k and v, respectively.
            b_q, b_k, b_v: optional biases for q, k and v, respectively.

        Shape:
            Inputs:
            - q: :math:`(Qdims..., Eq)` where Eq is the query embedding dimension and Qdims are any
                number of leading dimensions.
            - k: :math:`(Kdims..., Ek)` where Ek is the key embedding dimension and Kdims are any
                number of leading dimensions.
            - v: :math:`(Vdims..., Ev)` where Ev is the value embedding dimension and Vdims are any
                number of leading dimensions.
            - w_q: :math:`(Ei, Eq)` where Ei is the dimension to which the query, key, and value
                emebeddings are to be projected
            - w_k: :math:`(Ei, Ek)`
            - w_v: :math:`(Ei, Ev)`
            - b_q: :math:`(Ei)`
            - b_k: :math:`(Ei)`
            - b_v: :math:`(Ei)`
            Output: in output triple :math:`(q', k', v')`,
            - q': :math:`[Qdims..., Ei]`
            - k': :math:`[Kdims..., Ei]`
            - v': :math:`[Vdims..., Ei]`
        """
        Eq, Ek, Ev = query.size(-1), key.size(-1), value.size(-1)
        assert w_q.shape == (self.inner_dim, Eq), f'expecting query weights shape of {self.inner_dim, Eq}, but got {w_q.shape}'
        assert w_k.shape == (self.inner_dim, Ek), f'expecting key weights shape of {self.inner_dim, Ek}, but got {w_k.shape}'
        assert w_v.shape == (self.inner_dim, Ev), f'expecting value weights shape of {self.inner_dim, Ev}, but got {w_v.shape}'
        assert b_q is None or b_q.shape == (self.inner_dim,), f'expecting query bias shape of {self.inner_dim,}, but got {b_q.shape}'
        assert b_k is None or b_k.shape == (self.inner_dim,), f'expecting key bias shape of {self.inner_dim,}, but got {b_k.shape}'
        assert b_v is None or b_v.shape == (self.inner_dim,), f'expecting value bias shape of {self.inner_dim,}, but got {b_v.shape}'
        query_proj = F.linear(query, w_q, b_q)
        if is_self_attention:
            key_proj = F.linear(query, w_k, b_k)
            value_proj = F.linear(query, w_v, b_v)
        elif past_key_value is None:
            key_proj = F.linear(key, w_k, b_k)
            value_proj = F.linear(value, w_v, b_v)
        else:
            key_proj = key
            value_proj = value
        if past_key_value is not None:
            if is_self_attention:
                key_proj = key_proj.view(bsz, -1, self.num_heads, head_dim).transpose(1, 2)
                value_proj = value_proj.view(bsz, -1, self.num_heads, head_dim).transpose(1, 2)
                key_proj = torch.cat([past_key_value[0], key_proj], dim=2)
                value_proj = torch.cat([past_key_value[1], value_proj], dim=2)
            else:
                key_proj = past_key_value[0]
                value_proj = past_key_value[1]
        assert key_proj is not None
        assert value_proj is not None
        return query_proj, key_proj, value_proj

    def _t5_dot_product_attention(self, q: 'Tensor', k: 'Tensor', v: 'Tensor', position_bias: 'Tensor', attn_mask: 'Optional[Tensor]'=None, dropout_p: 'float'=0.0) ->Tuple[Tensor, Tensor]:
        """Computes scaled dot product attention on query, key and value tensors, using
        an optional attention mask if passed, and applying dropout if a probability
        greater than 0.0 is specified.

        Modified from https://github.com/pytorch/pytorch/blob/5953fd9133c0bdcc0158acf1472fac403bc5f636/torch/nn/functional.py#L4814.

        Args:
            q, k, v: Query, key and value tensors. See Shape section for shape details.
            attn_mask: Optional tensor containing mask values to be added to calculated
                attention. May be 2D or 3D; see Shape section for details.
            dropout_p: Dropout probability. If greater than 0.0, dropout is applied.
            position_bias: Position bias used to incorporate realtive attention bias in attention scors

        Shape:
            - q: :math:`(B, H, Nt, E)` where B is the batch size, H is the number of heads, Nt is the target sequence length,
                and E is the head dimension.
            - key: :math:`(B, H, Ns, E)` where B is the batch size, H is the number of heads, Ns is the source sequence length,
                and E is the head dimension.
            - value: :math:`(B, H, Ns, E)` where B is the batch size, H is the number of heads, Ns is the source sequence length,
                and E is the head dimension.
            - attn_mask: a 4D tensor of shape :math:`(B, H, Nt, Ns)`
            - position_bias: :math:`(1, H, Nt, Ns)`
            - Output: attention values have shape :math:`(B, Nt, H*E)`; attention weights
                have shape :math:`(B, H, Nt, Ns)`

        Returns:
            Tensor pair containing attended values and attention weights.
        """
        B, H, _, E = q.shape
        attn = torch.matmul(q, k.transpose(3, 2))
        position_bias = position_bias.repeat(B, 1, 1, 1)
        if attn_mask is not None:
            position_bias += attn_mask
        attn += position_bias
        attn = F.softmax(attn, dim=-1)
        if dropout_p > 0.0:
            attn = F.dropout(attn, p=dropout_p)
        output = torch.matmul(attn, v)
        output = output.transpose(1, 2).contiguous().view(B, -1, H * E)
        return output, attn

    def _compute_bias(self, query_length: 'int', key_length: 'int', bidirectional: 'bool'=True, device: 'Optional[torch.device]'=None) ->Tensor:
        """Compute binned relative position bias.

        Modified from https://github.com/huggingface/transformers/blob/8581a798c0a48fca07b29ce2ca2ef55adcae8c7e/src/transformers/models/t5/modeling_t5.py#L421.
        """
        assert self.relative_attention_bias is not None
        context_position = torch.arange(query_length, dtype=torch.long, device=device)[:, None]
        memory_position = torch.arange(key_length, dtype=torch.long, device=device)[None, :]
        relative_position = memory_position - context_position
        relative_position_bucket = self._relative_position_bucket(relative_position, bidirectional=bidirectional, num_buckets=self.relative_attention_num_buckets, max_distance=self.relative_attention_max_distance)
        values = self.relative_attention_bias(relative_position_bucket)
        values = values.permute([2, 0, 1]).unsqueeze(0)
        return values

    def _relative_position_bucket(self, relative_position: 'Tensor', bidirectional: 'bool'=True, num_buckets: 'int'=32, max_distance: 'int'=128) ->Tensor:
        """Adapted from Mesh Tensorflow:
        https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593
        and https://github.com/huggingface/transformers/blob/8581a798c0a48fca07b29ce2ca2ef55adcae8c7e/src/transformers/models/t5/modeling_t5.py#L374.

        Translates relative position to a bucket number for relative attention. The relative position is defined as
        memory_position - query_position, i.e. the distance in tokens from the attending position to the attended-to
        position. If bidirectional=False, then positive relative positions are invalid. We use smaller buckets for
        small absolute relative_position and larger buckets for larger absolute relative_positions. All relative
        positions >=max_distance map to the same bucket. All relative positions <=-max_distance map to the same bucket.
        This should allow for more graceful generalization to longer sequences than the model has been trained on.

        Args:
            relative_position: Tensor w/ initially constructed relative positions.
            bidirectional: If attention is bidirectional; when in decoder, this should be False.
            num_buckets: Number of buckets to utilize.
            max_distance: Maximum distance between positions.

        Returns:
            Tensor with the same shape as relative_position, containing int32 values in the range [0, num_buckets).
        """
        relative_buckets = torch.zeros(relative_position.shape, dtype=torch.long, device=relative_position.device)
        if bidirectional:
            num_buckets = num_buckets // 2
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


class T5Layer(nn.Module):
    """T5Layer is made up of a self-attn block, optional cross-attn block, and feed-forward network.

    This T5 layer is based on the paper:
    "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer".
    Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena,
    Yanqi Zhou, Wei Li, and Peter J. Liu. 2020. Journal of Machine Learning Research.
    Volume 21 Issue 140 pages 1-67. http://jmlr.org/papers/v21/20-074.html
    Users may modify or implement in a different way during application.

    Args:
        d_model: Number of expected features in the input (required).
        nhead: Number of heads in the multihead attention models (required).
        dim_feedforward: Dimension of the feedforward network model (default: 3072).
        qkv_dim: Projection dimension (per head) for query, keys, and values. (default: 64).
        dropout: Dropout value (default: 0.1).
        activation: Activation function of the intermediate layer, can be a string
            ("relu", "gelu", or "gelu_new") or a unary callable. (default: F.relu)
        is_gated_act: Option to include gated activated as done in FLAN-T5, see
            https://huggingface.co/google/flan-t5-xxl. (default: False)
        layer_norm_eps: The eps value in layer normalization components. (default=1e-6)
        relative_attention_num_buckets: Number of relative position buckets. (default: 32)
        relative_attention_max_distance: Maximum threshold on the relative distance used to
            allocate buckets. Anything larger gets placed in the same bucket. (default: 128)
        compute_relative_attention_bias: Whether or not the relative position embeddings
            need to be computed. Typically occurs in the first layer of the encoder
            and resulting position embeddings are returned to be passed up to higher layers. (default: False)
        is_decoder: Whether the T5Layer will be instantiated as a decoder layer or encoder layer. (default: False)
        device: Device to use any newly constructed Tensors. (optional)
        dtype: Datatype to use on any newly constructed Tensors. (optional)

    Examples::
        >>> single_encoder_layer = T5Layer(d_model=768, nhead=12)
        >>> src = torch.rand(32, 20, 768)
        >>> single_encoder_layer(src)

        >>> single_decoder_layer = T5Layer(d_model=768, nhead=12, is_decoder=True)
        >>> src = torch.rand(32, 20, 768)
        >>> tgt = torch.rand(32, 1, 768)
        >>> single_decoder_layer(tgt, src)
    """

    def __init__(self, d_model: 'int', nhead: 'int', dim_feedforward: 'int'=3072, qkv_dim: 'int'=64, dropout: 'float'=0.1, activation: 'Union[str, Callable[[Tensor], Tensor]]'=F.relu, is_gated_act: 'bool'=False, layer_norm_eps: 'float'=1e-06, relative_attention_num_buckets: 'int'=32, relative_attention_max_distance: 'int'=128, compute_relative_attention_bias: 'bool'=False, is_decoder: 'bool'=False, device: 'Optional[torch.device]'=None, dtype=None) ->None:
        super().__init__()
        self.is_gated_act = is_gated_act
        self.compute_relative_attention_bias = compute_relative_attention_bias
        self.relative_attention_num_buckets = relative_attention_num_buckets
        self.relative_attention_max_distance = relative_attention_max_distance
        self.is_decoder = is_decoder
        self.self_attn = T5MultiheadAttention(d_model, nhead, is_decoder=is_decoder, dropout=dropout, qkv_dim=qkv_dim, compute_relative_attention_bias=compute_relative_attention_bias, relative_attention_num_buckets=relative_attention_num_buckets, relative_attention_max_distance=relative_attention_max_distance, device=device, dtype=dtype)
        if self.is_decoder:
            self.cross_attn = T5MultiheadAttention(d_model, nhead, is_decoder=True, dropout=dropout, qkv_dim=qkv_dim, compute_relative_attention_bias=False, relative_attention_num_buckets=relative_attention_num_buckets, relative_attention_max_distance=relative_attention_max_distance, device=device, dtype=dtype)
            self.norm3 = T5LayerNorm(d_model, eps=layer_norm_eps)
            self.dropout4 = nn.Dropout(dropout)
        else:
            self.cross_attn = None
            self.norm3 = None
            self.dropout4 = None
        if self.is_gated_act:
            self.linear1 = None
            self.linear1_0 = nn.Linear(d_model, dim_feedforward, bias=False)
            self.linear1_1 = nn.Linear(d_model, dim_feedforward, bias=False)
        else:
            self.linear1 = nn.Linear(d_model, dim_feedforward, bias=False)
            self.linear1_0 = None
            self.linear1_1 = None
        self.linear2 = nn.Linear(dim_feedforward, d_model, bias=False)
        self.norm1 = T5LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = T5LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        if isinstance(activation, str):
            assert activation in ('relu', 'gelu', 'gelu_new'), f"Do not support '{activation}' activation. Use 'relu' or 'gelu' or 'gelu_new'"
            if activation == 'relu':
                self.activation = F.relu
            elif activation == 'gelu':
                self.activation = F.gelu
            elif activation == 'gelu_new':
                self.activation = nn.GELU(approximate='tanh')
        else:
            self.activation = activation

    def forward(self, seq: 'Tensor', memory: 'Optional[Tensor]'=None, mask: 'Optional[Tensor]'=None, seq_key_padding_mask: 'Optional[Tensor]'=None, memory_mask: 'Optional[Tensor]'=None, memory_key_padding_mask: 'Optional[Tensor]'=None, position_bias: 'Optional[Tensor]'=None, past_key_values: 'Optional[PAST_KEY_VALUES_TYPE]'=None) ->Tuple[Tensor, Optional[Tensor], Optional[Tensor], Optional[Tensor], PAST_KEY_VALUES_UNFILLED_TYPE]:
        """Pass the inputs (and mask) through the encoder layer.

        Args:
            seq: Input sequence (required).
                Must have shape (B, Ns, E) where B is the batch size, Nt is the sequence length,
                and E is the model dimension. This will be the src sequence if `self.is_decoder = False`
                and tgt sequence if `self.is_decoder = True`.
            memory: Encoder sequence (optional).
                Output from encoder layer, only needs to be included when in decoding context.
            mask: Attention mask for self-attention. (optional).
                Must have shape (Ns, Ns).
            seq_key_padding_mask: Mask for the seq keys per batch (optional).
                Must have shape (B, Ns).
            memory_mask: Attention mask for attention in decoding context. (optional)
                Must have shape (Nm, Nm).
            memory_key_padding_mask: Mask for the memory keys per batch (optional).
                Must have shape (B, Ns).
            position_bias: Relative attention bias to be used when computing self-attention scores (optional)
                Must have shape (B, H, Ns, Ns) where H is the number of heads.
            past_key_values: Past key values used for incremental decoding (optional).
                Tuple with Tensors of shape (B, H, N)>>>>> Check this????

        Returns:
            Tuple of Tensors being hidden states, position bias, self-attention scores, cross-attention scores,
                and key-value pairs.
        """
        if past_key_values is not None:
            self_attn_past_key_value = past_key_values[:2]
            cross_attn_past_key_value = past_key_values[2:]
        else:
            self_attn_past_key_value, cross_attn_past_key_value = None, None
        x = seq
        sa_out, position_bias, sa_scores, sa_kv = self._sa_block(self.norm1(x), mask, seq_key_padding_mask, position_bias, self_attn_past_key_value)
        x = x + sa_out
        if self.is_decoder:
            assert memory is not None, 'Must provide memory (encoder hidden states).'
            assert self.norm3 is not None
            query_length = sa_kv[0].shape[2]
            ca_out, ca_scores, ca_kv = self._ca_block(self.norm3(x), memory, query_length, memory_mask, memory_key_padding_mask, cross_attn_past_key_value)
            x = x + ca_out
        else:
            ca_scores, ca_kv = None, None
        x = x + self._ff_block(self.norm2(x))
        new_key_value = sa_kv + (ca_kv if ca_kv is not None else (None, None))
        assert torch.jit.isinstance(new_key_value, PAST_KEY_VALUES_UNFILLED_TYPE)
        return x, position_bias, sa_scores, ca_scores, new_key_value

    def _sa_block(self, x: 'Tensor', attn_mask: 'Optional[Tensor]', key_padding_mask: 'Optional[Tensor]', position_bias: 'Optional[Tensor]', past_key_value: 'Optional[PAST_KEY_VALUE_TYPE]'=None) ->Tuple[Tensor, Optional[Tensor], Optional[Tensor], PAST_KEY_VALUE_TYPE]:
        """Self-attention block."""
        attn, curr_position_bias, scores, curr_key_value = self.self_attn(x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=True, position_bias=position_bias, past_key_value=past_key_value)
        if self.compute_relative_attention_bias:
            position_bias = curr_position_bias
        return self.dropout1(attn), position_bias, scores, curr_key_value

    def _ca_block(self, x: 'Tensor', mem: 'Tensor', query_length: 'Optional[int]', attn_mask: 'Optional[Tensor]', key_padding_mask: 'Optional[Tensor]', past_key_value: 'Optional[PAST_KEY_VALUE_TYPE]'=None) ->Tuple[Tensor, Optional[Tensor], PAST_KEY_VALUE_TYPE]:
        """Cross-attention block."""
        assert self.cross_attn is not None
        assert self.dropout4 is not None
        attn, _, scores, curr_key_value = self.cross_attn(x, mem, mem, query_length=query_length, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=True, past_key_value=past_key_value)
        return self.dropout4(attn), scores, curr_key_value

    def _ff_block(self, x: 'Tensor') ->Tensor:
        """Feed-forward block."""
        if self.is_gated_act:
            assert self.linear1_0 is not None
            assert self.linear1_1 is not None
            wi_0 = self.activation(self.linear1_0(x))
            wi_1 = self.linear1_1(x)
            hidden_states = wi_0 * wi_1
            hidden_states = self.dropout2(hidden_states)
            hidden_states = self.linear2(hidden_states)
        else:
            assert self.linear1 is not None
            hidden_states = self.linear2(self.dropout2(self.activation(self.linear1(x))))
        return self.dropout3(hidden_states)


class T5Decoder(nn.Module):
    """T5Decoder is a stack of N decoder layers.

    Args:
        d_model: Number of expected features in the input (required).
        nhead: Number of heads in the multihead attention models (required).
        num_layers: Number of decoder layers in the stack (required)
        dim_feedforward: Dimension of the feedforward network model (default=3072).
        qkv_dim: Projection dimension (per head) for query, keys, and values. (defualt=64).
        dropout: Dropout value (default=0.1).
        activation: Activation function of the intermediate layer, can be a string
            ("relu", "gelu", or "gelu_new") or a unary callable. (default: F.relu)
        is_gated_act: Option to include gated activated as done in FLAN-T5, see
            https://huggingface.co/google/flan-t5-xxl. (default: False)
        layer_norm_eps: The eps value in layer normalization components (default=1e-6).
        relative_attention_num_buckets: Number of relative position buckets (default: 32)
        relative_attention_max_distance: Maximum threshold on the relative distance used to
            allocate buckets. Anything larger gets placed in the same bucket (defulat: 128)
        device: Device to use any newly constructed Tensors. (optional)
        dtype: Datatype to use on any newly constructed Tensors. (optional)


    Examples::
        >>> decoder = T5Decoder(d_model=768, nhead=12, num_layers=12)
        >>> memory = torch.rand(32, 10, 512)
        >>> tgt = torch.rand(32, 1, 512)
        >>> decoder(tgt, memory)
    """

    def __init__(self, d_model: 'int', nhead: 'int', num_layers: 'int', dim_feedforward: 'int'=3072, qkv_dim: 'int'=64, dropout: 'float'=0.1, activation: 'Union[str, Callable[[Tensor], Tensor]]'=F.relu, is_gated_act: 'bool'=False, layer_norm_eps: 'float'=1e-06, relative_attention_num_buckets: 'int'=32, relative_attention_max_distance: 'int'=128, device: 'Optional[torch.device]'=None, dtype=None) ->None:
        super().__init__()
        self.layers = nn.ModuleList([T5Layer(d_model, nhead, dim_feedforward=dim_feedforward, qkv_dim=qkv_dim, dropout=dropout, activation=activation, is_gated_act=is_gated_act, layer_norm_eps=layer_norm_eps, relative_attention_num_buckets=relative_attention_num_buckets, relative_attention_max_distance=relative_attention_max_distance, compute_relative_attention_bias=True if i == 0 else False, is_decoder=True, device=device, dtype=dtype) for i in range(num_layers)])
        self.norm = T5LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.num_layers = num_layers

    def forward(self, embedded_tgt: 'Tensor', memory: 'Tensor', tgt_mask: 'Optional[Tensor]'=None, memory_mask: 'Optional[Tensor]'=None, tgt_key_padding_mask: 'Optional[Tensor]'=None, memory_key_padding_mask: 'Optional[Tensor]'=None, past_key_values: 'Optional[List[PAST_KEY_VALUES_TYPE]]'=None, return_past_key_values: 'bool'=False) ->SEQ_2_SEQ_OUTPUTS_TYPE:
        """Pass the inputs (and masks) through the stack of decoder layers.

        Args:
            embedded_tgt: Input sequence to the decoder layer. (required).
                Must have shape (B, Nt, E) where B is the batch size, Nt is the target sequence
                length, and E is the model dimension.
            memory: Sequence from the last layer of the encoder. (required).
                Must have shape (B, Nts, E) where B is the batch size, Ns is the source sequence
                length, and E is the model dimension.
            tgt_mask: Attention mask for self-attention. (optional).
                Must have shape (Nt, Nt).
            memory_mask: Attention mask for cross-attention (optional).
                Must have shape (Nt, Ns).
            tgt_key_padding_mask: Mask for the tgt keys per batch (optional).
                Must have shape (B, Nt).
            memory_key_padding_mask: Mask for the memory keys per batch (optional).
                Must have shape (B, Ns).
            past_key_values: Past key values used for incremental decoding (optional).
                List of Tuple with Tensors of shape (B, H, N)>>>>> Check this????
            return_past_key_values: Boolean stating whether to return past_key_values from model. (default: False)

        Returns:
            Dictionary of last hidden state, all hidden states, position bias, self-attention scores, cross-attention scores
                and past key values (if requested).
        """
        output = self.dropout1(embedded_tgt)
        position_bias = None
        all_outputs = torch.jit.annotate(List[Tensor], [])
        all_sa_scores = torch.jit.annotate(List[Optional[Tensor]], [])
        all_ca_scores = torch.jit.annotate(List[Optional[Tensor]], [])
        all_key_values = torch.jit.annotate(List[Tuple[Tensor, Tensor, Optional[Tensor], Optional[Tensor]]], [])
        for i, mod in enumerate(self.layers):
            all_outputs.append(output)
            output, position_bias, sa_score, ca_score, past_key_value = mod(output, memory, mask=tgt_mask, memory_mask=memory_mask, seq_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask, position_bias=position_bias, past_key_values=past_key_values[i] if past_key_values is not None else None)
            all_sa_scores.append(sa_score)
            all_ca_scores.append(ca_score)
            if past_key_value is not None and return_past_key_values:
                all_key_values.append(past_key_value)
        output = self.norm(output)
        output = self.dropout2(output)
        all_outputs.append(output)
        return {'decoder_output': output, 'decoder_hidden_states': all_outputs, 'decoder_position_bias': position_bias, 'decoder_sa_scores': all_sa_scores, 'decoder_ca_scores': all_ca_scores, 'past_key_values': all_key_values}


class T5Encoder(nn.Module):
    """T5Encoder is a stack of N encoder layers.

    Args:
        d_model: Number of expected features in the input (required).
        nhead: Number of heads in the multihead attention models (required).
        num_layers: Number of encoder layers in the stack (required)
        dim_feedforward: Dimension of the feedforward network model (default=3072).
        qkv_dim: Projection dimension (per head) for query, keys, and values. (defualt=64).
        dropout: Dropout value (default=0.1).
        activation: Activation function of the intermediate layer, can be a string
            ("relu", "gelu", or "gelu_new") or a unary callable. (default: F.relu)
        is_gated_act: Option to include gated activated as done in FLAN-T5, see
            https://huggingface.co/google/flan-t5-xxl. (default: False)
        layer_norm_eps: The eps value in layer normalization components (default=1e-6).
        relative_attention_num_buckets: Number of relative position buckets (default: 32)
        relative_attention_max_distance: Maximum threshold on the relative distance used to
            allocate buckets. Anything larger gets placed in the same bucket (defulat: 128)
        token_embeddings (nn.Module): Embedding layer to be passed in the case that the input to `forward`
            is not already embedded.
        device: Device to use any newly constructed Tensors. (optional)
        dtype: Datatype to use on any newly constructed Tensors. (optional)

    Examples::
        >>> encoder = T5Encoder(d_model=768, nhead=12, num_layers=12)
        >>> tgt = torch.rand(32, 10, 512)
        >>> encoder(tgt)
    """

    def __init__(self, d_model: 'int', nhead: 'int', num_layers: 'int', dim_feedforward: 'int'=3072, qkv_dim: 'int'=64, dropout: 'float'=0.1, activation: 'Union[str, Callable[[Tensor], Tensor]]'=F.relu, is_gated_act: 'bool'=False, layer_norm_eps: 'float'=1e-06, relative_attention_num_buckets: 'int'=32, relative_attention_max_distance: 'int'=128, token_embeddings: 'Optional[nn.Module]'=None, device: 'Optional[torch.device]'=None, dtype=None) ->None:
        super().__init__()
        self.token_embeddings = token_embeddings
        self.layers = nn.ModuleList([T5Layer(d_model, nhead, dim_feedforward=dim_feedforward, qkv_dim=qkv_dim, dropout=dropout, activation=activation, is_gated_act=is_gated_act, layer_norm_eps=layer_norm_eps, relative_attention_num_buckets=relative_attention_num_buckets, relative_attention_max_distance=relative_attention_max_distance, compute_relative_attention_bias=True if i == 0 else False, is_decoder=False, device=device, dtype=dtype) for i in range(num_layers)])
        self.num_layers = num_layers
        self.norm = T5LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src: 'Optional[Tensor]'=None, mask: 'Optional[Tensor]'=None, src_key_padding_mask: 'Optional[Tensor]'=None, embedded_src: 'Optional[Tensor]'=None) ->SEQ_2_SEQ_OUTPUTS_TYPE:
        """Pass the input (and masks) through the stack of encoder layers.

        Args:
            src (Optional[Tensor]): Tokenized input sequence to the encoder.
                Must be batch first with shape (B, Ne) where B is the batch size and Ne is the
                encoder input sequence length.
            mask (Optional[Tensor]): Attention mask for self-attention.
                Must have shape (Nt, Nt).
            src_key_padding_mask (Optional[Tensor]): Mask for the tgt keys per batch.
                Must have shape (B, Nt).
            embedded_src (Optional[Tensor]): Embedded input sequence to the encoder layer.
                Must have shape (B, Nt, E) where B is the batch size, Nt is the target sequence
                length, and E is the model dimension.
                *Note*: If you do not provide this `embedded_tgt`, you must have provided a `token_embedding` layer \\
                    in the initialization of the T5Encoder.

        Returns:
            Dictionary of last hidden layer, all hidden layers, position bias, and self-attention scores.
        """
        if embedded_src is None:
            assert self.token_embeddings is not None and src is not None, 'Must provide `token_embeddings` and `tgt` if not providing already embedded tokens.'
            embedded_src = self.token_embeddings(src)
        output = self.dropout1(embedded_src)
        position_bias = None
        all_outputs = torch.jit.annotate(List[Tensor], [])
        all_sa_scores = torch.jit.annotate(List[Optional[Tensor]], [])
        for mod in self.layers:
            all_outputs.append(output)
            output, position_bias, sa_score, _, _ = mod(output, mask=mask, seq_key_padding_mask=src_key_padding_mask, position_bias=position_bias)
            all_sa_scores.append(sa_score)
        output = self.norm(output)
        output = self.dropout2(output)
        all_outputs.append(output)
        return {'encoder_output': output, 'encoder_hidden_states': all_outputs, 'encoder_position_bias': position_bias, 'encoder_sa_scores': all_sa_scores}


class T5Model(nn.Module):
    """A T5 model. User is able to modify the attributes as needed. The architecture
    is based on the paper "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer".
    Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena,
    Yanqi Zhou, Wei Li, and Peter J. Liu. 2020. Journal of Machine Learning Research.
    Volume 21 Issue 140 pages 1-67. http://jmlr.org/papers/v21/20-074.html

    Args:
        config.encoder_only: Whether or not model should consist of only the encoder as opposed to encoder-decoder (default=False).
        config.linear_head: Whether or not a linear layer should be used to project the output of the decoder's last layer to the vocab (default=False).
        config.embedding_dim: Number of expected features in the encoder/decoder inputs (default=768).
        config.qkv_dim: Projection dimension (per head) for query, keys, and values. (defualt=64).
        config.num_attention_heads: Number of heads in the multiheadattention models (default=12).
        config.num_encoder_layers: Number of encoder layers in the encoder (default=12).
        config.num_decoder_layers: Number of decoder layers in the decoder (default=12).
        config.ffn_dimension: Dimension of the feedforward network model (default=3072).
        config.dropout: Dropout value (default=0.1).
        config.activation: Activation function of encoder/decoder intermediate layer, can be a string
            ("relu" or "gelu") or a unary callable. Default: relu
        config.layer_norm_eps: The eps value in layer normalization components (default=1e-6).
        config.relative_attention_num_buckets: Number of relative position buckets (default: 32)
        config.relative_attention_max_distance: Maximum threshold on the relative distance used to
            allocate buckets. Anything larger gets placed in the same bucket (default: 128)
        config.padding_idx: Index assigned to padding token in vocabulary (default: 0)
        config.max_seq_len: Maximum sequence length (default: 512)
        config.vocab_size: Size of vocabulary (default: 32128)
        config.training: Whether or not to apply dropout (default: False)
        freeze: Indicates whether or not to freeze the model weights. (default: False)

    Examples:
        >>> from torchtext.models import T5Conf, T5Model
        >>> t5_config = T5Conf(encoder_only=False, linear_head=True)
        >>> t5_model = T5Model(t5_config)
        >>> encoder_input = torch.randint(0, t5_config.vocab_size, (32, 512))
        >>> out = t5_model(encoder_input)['decoder_output']
        >>> out.shape
        torch.Size([32, 1, 32128])
    """

    def __init__(self, config: 'T5Conf', freeze: 'bool'=False, device: 'Optional[torch.device]'=None, dtype=None) ->None:
        super().__init__()
        assert isinstance(config, T5Conf)
        self.config = config
        self.embedding_dim = config.embedding_dim
        self.encoder_only = config.encoder_only
        self.linear_head = config.linear_head
        self.padding_idx = config.padding_idx
        self.training = config.training
        self.dropout = config.dropout if config.training else 0.0
        self.dtype = dtype
        self.token_embeddings = nn.Embedding(config.vocab_size, config.embedding_dim, config.padding_idx)
        self.encoder = T5Encoder(d_model=config.embedding_dim, nhead=config.num_attention_heads, num_layers=config.num_encoder_layers, dim_feedforward=config.ffn_dimension, qkv_dim=config.qkv_dim, dropout=self.dropout, activation=config.activation, layer_norm_eps=config.layer_norm_eps, relative_attention_num_buckets=config.relative_attention_num_buckets, relative_attention_max_distance=config.relative_attention_max_distance, token_embeddings=self.token_embeddings, is_gated_act=config.is_gated_act, device=device, dtype=dtype)
        if not config.encoder_only:
            self.decoder = T5Decoder(d_model=config.embedding_dim, nhead=config.num_attention_heads, num_layers=config.num_decoder_layers, dim_feedforward=config.ffn_dimension, qkv_dim=config.qkv_dim, dropout=self.dropout, activation=config.activation, layer_norm_eps=config.layer_norm_eps, relative_attention_num_buckets=config.relative_attention_num_buckets, relative_attention_max_distance=config.relative_attention_max_distance, is_gated_act=config.is_gated_act, device=device, dtype=dtype)
        else:
            self.decoder = None
        if config.linear_head:
            self.lm_head = nn.Linear(config.embedding_dim, config.vocab_size, bias=False)
        else:
            self.lm_head = None
        if freeze:
            for p in self.parameters():
                p.requires_grad = False

    @torch.jit.export
    def _reorder_cache(self, past: 'List[PAST_KEY_VALUES_TYPE]', beam_idx: 'Tensor') ->List[PAST_KEY_VALUES_TYPE]:
        """Reorder past key value pairs in cache. Only relevant in incremental decoding with beam search generation."""
        if past is None:
            return past
        reordered_decoder_past: 'List[PAST_KEY_VALUES_TYPE]' = []
        for layer_past_states in past:
            reordered_layer_past_states = ()
            for layer_past_state in layer_past_states:
                reordered_layer_past_states = reordered_layer_past_states + (layer_past_state.index_select(0, beam_idx),)
            assert len(reordered_layer_past_states) == len(layer_past_states)
            reordered_decoder_past.append(reordered_layer_past_states)
        return reordered_decoder_past

    @torch.jit.export
    def _shift_right(self, input_ids: 'Tensor') ->Tensor:
        """Shift all input sequences to the right"""
        shifted_input_ids = torch.zeros_like(input_ids)
        shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
        shifted_input_ids[:, 0] = self.padding_idx
        return shifted_input_ids

    @torch.jit.export
    def prepare_inputs_for_generation(self, input_ids: 'Tensor', encoder_outputs: 'Optional[SEQ_2_SEQ_OUTPUTS_TYPE]'=None, encoder_padding_mask: 'Optional[Tensor]'=None, past: 'Optional[List[PAST_KEY_VALUES_TYPE]]'=None, return_past_key_values: 'bool'=True, model_kwargs: 'Optional[Dict[str, Union[SEQ_2_SEQ_OUTPUTS_TYPE, Optional[Tensor], Optional[List[PAST_KEY_VALUES_TYPE]], bool]]]'=None) ->Dict[str, Union[Tensor, SEQ_2_SEQ_OUTPUTS_TYPE, Optional[List[PAST_KEY_VALUES_TYPE]], bool]]:
        """Prepare inputs for generation from model_kwargs.

        Args:
            input_ids (Tensor): Seed tokens for generation.
            model_kwargs (Dict): Other specifications for generation

        Returns:
            Dictionary bundling all model inputs for generation.
        """
        if model_kwargs is None:
            assert encoder_outputs is not None
            model_kwargs = {'encoder_outputs': encoder_outputs, 'encoder_padding_mask': encoder_padding_mask, 'past_key_values': past, 'return_past_key_values': return_past_key_values}
        past = model_kwargs.get('past', None)
        if past is not None:
            input_ids = input_ids[:, -1:]
        model_kwargs['decoder_tokens'] = input_ids
        return model_kwargs

    def get_encoder(self) ->T5Encoder:
        return self.encoder

    def get_decoder(self) ->Optional[T5Decoder]:
        if self.decoder is None:
            warnings.warn('Decoder is not set on this model.')
        return self.decoder

    def forward(self, encoder_tokens: 'Optional[Tensor]'=None, decoder_tokens: 'Optional[Tensor]'=None, encoder_mask: 'Optional[Tensor]'=None, decoder_mask: 'Optional[Tensor]'=None, encoder_padding_mask: 'Optional[Tensor]'=None, decoder_padding_mask: 'Optional[Tensor]'=None, encoder_outputs: 'Optional[SEQ_2_SEQ_OUTPUTS_TYPE]'=None, past_key_values: 'Optional[List[PAST_KEY_VALUES_TYPE]]'=None, return_past_key_values: 'bool'=False) ->SEQ_2_SEQ_OUTPUTS_TYPE:
        """Pass the inputs (and mask) through the T5Encoder/T5Decoder in turn.

        Args:
            encoder_tokens: Tokenized input sequence to the encoder.
                Must be batch first with shape (B, Ne) where B is the batch size and Ne is the
                encoder input sequence length. (optional if `encoder_outputs` is provided)
            decoder_tokens: Tokenized input sequence to the decoder.
                Must be batch first with shape (B, Nd) where B is the batch size and Nd is the
                decoder input sequence length. If None and model is encoder-decoder, will initialize decoder
                input sequence to begin with padding index. (optional).
            encoder_mask: Self-attention mask for the encoder input sequence.
                Must have shape (Ne, Ne) (optional).
            decoder_mask: Self-attention mask for the decoder input sequence.
                Must have shape (Nd, Nd) (optional).
            encoder_padding_mask: Padding mask for encoder input sequence.
                Must have shape (B, Ne) (optional).
            decoder_padding_mask: Padding mask for decoder input sequence.
                Must have shape (B, Nd) (optional).
            encoder_outputs: Outputs from previous run of T5Encoder. (optional)
            past_key_values: Previously calculated key values, used in incremental decoding. (optional)
            return_past_key_values: Boolean indicating whether to return key values to user. (default: False)

        Returns:
            encoder_output: Output Tensor from the final layer of the encoder
            encoder_hidden_states: Tuple of output Tensors from each layer of the encoder
            encoder_position_bias: Tensor of relative attention bias computed for input sequence to encoder
            encoder_sa_scores: Tuple of self-attention scores computed at each layer of the encoder
            decoder_output: Output Tensor from the final layer of the decoder
            decoder_hidden_states: Tuple of output Tensors from each layer of the decoder
            decoder_position_bias: Tensor of relative attention bias computed for input sequence to decoder
            encoder_sa_scores: Tuple of self-attention scores computed at each layer of the decoder
            encoder_ca_scores: Tuple of cross-attention scores computed at each layer of the decoder
            past_key_values: List of Tuples of key values calculated during this run, or None.
        """
        seq2seq_model_output: 'SEQ_2_SEQ_OUTPUTS_TYPE' = {'encoder_output': None, 'encoder_hidden_states': None, 'encoder_sa_scores': None, 'encoder_ca_scores': None, 'decoder_output': None, 'decoder_hidden_states': None, 'decoder_sa_scores': None, 'decoder_ca_scores': None, 'past_key_values': None}
        if encoder_outputs is None:
            assert encoder_tokens is not None, 'If `encoder_outputs` is not specified, must provide `encoder_tokens`'
            if encoder_padding_mask is None:
                encoder_padding_mask = encoder_tokens.eq(self.padding_idx)
            encoder_outputs = self.encoder(src=encoder_tokens, mask=encoder_mask, src_key_padding_mask=encoder_padding_mask)
            seq2seq_model_output.update(encoder_outputs)
        if not self.encoder_only:
            assert self.decoder is not None
            assert encoder_outputs is not None
            encoder_output = encoder_outputs.get('encoder_output')
            assert torch.jit.isinstance(encoder_output, Tensor)
            batch_size = encoder_output.size(0)
            encoder_output_device = encoder_output.device
            if decoder_tokens is None:
                decoder_tokens = torch.ones((batch_size, 1), device=encoder_output_device, dtype=torch.long) * self.padding_idx
            if decoder_padding_mask is None:
                decoder_padding_mask = decoder_tokens.eq(self.padding_idx)
                decoder_padding_mask[:, 0] = False
            decoder_embeddings = self.token_embeddings(decoder_tokens)
            decoder_outputs = self.decoder(decoder_embeddings, memory=encoder_output, tgt_mask=decoder_mask, memory_mask=encoder_mask, tgt_key_padding_mask=decoder_padding_mask, memory_key_padding_mask=encoder_padding_mask, past_key_values=past_key_values, return_past_key_values=return_past_key_values)
            decoder_output = decoder_outputs.get('decoder_output')
            assert torch.jit.isinstance(decoder_output, Tensor)
            if self.linear_head:
                assert self.lm_head is not None
                decoder_output = decoder_output * self.embedding_dim ** -0.5
                decoder_output = self.lm_head(decoder_output)
                decoder_outputs['decoder_output'] = decoder_output
            seq2seq_model_output.update(decoder_outputs)
        return seq2seq_model_output


def to_tensor(input: 'Any', padding_value: 'Optional[int]'=None, dtype: 'torch.dtype'=torch.long) ->Tensor:
    """Convert input to torch tensor

    :param padding_value: Pad value to make each input in the batch of length equal to the longest sequence in the batch.
    :type padding_value: Optional[int]
    :param dtype: :class:`torch.dtype` of output tensor
    :type dtype: :class:`torch.dtype`
    :param input: Sequence or batch of token ids
    :type input: Union[List[int], List[List[int]]]
    :rtype: Tensor
    """
    if torch.jit.isinstance(input, List[int]):
        return torch.tensor(input, dtype=torch.long)
    elif torch.jit.isinstance(input, List[List[int]]):
        if padding_value is None:
            output = torch.tensor(input, dtype=dtype)
            return output
        else:
            output = pad_sequence([torch.tensor(ids, dtype=dtype) for ids in input], batch_first=True, padding_value=float(padding_value))
            return output
    else:
        raise TypeError('Input type not supported')


class T5Transform(nn.Module):
    """
    This transform makes use of a pre-trained sentencepiece model to tokenize text input. The resulting output is fed to the T5 model.

    Additional details: https://github.com/google/sentencepiece

    :param sp_model_path: Path to pre-trained sentencepiece model
    :type sp_model_path: str
    :param max_seq_len: Maximum sequence length accepted for inputs to T5 model
    :type max_seq_len: int
    :param eos_idx: End-of-sequence token id
    :type eos_idx: int
    :param padding_idx: Padding token id
    :type padding_idx: int

    Example
        >>> from torchtext.prototype.models import T5Transform
        >>> transform = T5Transform("spm_model", max_seq_len = 10, eos_idx = 1, padding_idx = 0)
        >>> transform(["hello world", "attention is all you need!"])
    """

    def __init__(self, sp_model_path: 'str', max_seq_len: 'int', eos_idx: 'int', padding_idx: 'int'):
        super().__init__()
        self.sp_model = load_sp_model(get_asset_local_path(sp_model_path))
        self.max_seq_len = max_seq_len
        self.eos_idx = eos_idx
        self.padding_idx = padding_idx
        self.pipeline = T.Sequential(T.Truncate(self.max_seq_len - 1), T.AddToken(token=self.eos_idx, begin=False))

    def forward(self, input: 'Union[str, List[str]]') ->torch.Tensor:
        """
        :param input: Input sentence or list of sentences to tokenize.
        :type input: Union[str, List[str]]
        :return: Tokenized text that has been truncated, appended with end-of-sequence token, and padded
        :rtype: torch.Tensor
        """
        tokens = self.encode(input)
        out = to_tensor(self.pipeline(tokens), padding_value=self.padding_idx)
        return out

    @torch.jit.export
    def encode(self, input: 'Union[str, List[str]]') ->Union[List[int], List[List[int]]]:
        """
        :param input: Input sentence or list of sentences to tokenize.
        :type input: Union[str, List[str]]
        :return: Tokenized text that has been translated to token ids
        :rtype: Union[List[int], List[List[int]]]
        """
        if torch.jit.isinstance(input, List[str]):
            tokens: 'List[List[int]]' = []
            for text in input:
                tokens.append(self.sp_model.EncodeAsIds(text))
            return tokens
        elif torch.jit.isinstance(input, str):
            return self.sp_model.EncodeAsIds(input)
        else:
            raise TypeError('Input type not supported')

    @torch.jit.export
    def decode(self, input: 'Union[List[int], List[List[int]]]') ->Union[str, List[str]]:
        """
        :param input: List of token ids or list of lists of token ids (i.e. batched).
        :type input: Union[List[int], List[List[int]]]
        :return: Sentence or list of sentencess that were translated from the input token ids
        :rtype: Union[str, List[str]]
        """
        if torch.jit.isinstance(input, List[List[int]]):
            tokens: 'List[str]' = []
            for ids in input:
                tokens.append(self.sp_model.DecodeIds(ids))
            return tokens
        elif torch.jit.isinstance(input, List[int]):
            return self.sp_model.DecodeIds(input)
        else:
            raise TypeError('Input type not supported')


class MultiheadAttentionContainer(torch.nn.Module):

    def __init__(self, nhead, in_proj_container, attention_layer, out_proj, batch_first=False) ->None:
        """A multi-head attention container

        Args:
            nhead: the number of heads in the multiheadattention model
            in_proj_container: A container of multi-head in-projection linear layers (a.k.a nn.Linear).
            attention_layer: The custom attention layer. The input sent from MHA container to the attention layer
                is in the shape of `(..., L, N * H, E / H)` for query and `(..., S, N * H, E / H)` for key/value
                while the  output shape of the attention layer is expected to be `(..., L, N * H, E / H)`.
                The attention_layer needs to support broadcast if users want the overall MultiheadAttentionContainer
                with broadcast.
            out_proj: The multi-head out-projection layer (a.k.a nn.Linear).
            batch_first: If ``True``, then the input and output tensors are provided
                as `(..., N, L, E)`. Default: ``False``

        Examples::
            >>> import torch
            >>> from torchtext.nn import MultiheadAttentionContainer, InProjContainer, ScaledDotProduct
            >>> embed_dim, num_heads, bsz = 10, 5, 64
            >>> in_proj_container = InProjContainer(torch.nn.Linear(embed_dim, embed_dim),
                                                    torch.nn.Linear(embed_dim, embed_dim),
                                                    torch.nn.Linear(embed_dim, embed_dim))
            >>> MHA = MultiheadAttentionContainer(num_heads,
                                                  in_proj_container,
                                                  ScaledDotProduct(),
                                                  torch.nn.Linear(embed_dim, embed_dim))
            >>> query = torch.rand((21, bsz, embed_dim))
            >>> key = value = torch.rand((16, bsz, embed_dim))
            >>> attn_output, attn_weights = MHA(query, key, value)
            >>> print(attn_output.shape)
            >>> torch.Size([21, 64, 10])
        """
        super(MultiheadAttentionContainer, self).__init__()
        self.nhead = nhead
        self.in_proj_container = in_proj_container
        self.attention_layer = attention_layer
        self.out_proj = out_proj
        self.batch_first = batch_first

    def forward(self, query: 'torch.Tensor', key: 'torch.Tensor', value: 'torch.Tensor', attn_mask: 'Optional[torch.Tensor]'=None, bias_k: 'Optional[torch.Tensor]'=None, bias_v: 'Optional[torch.Tensor]'=None) ->Tuple[torch.Tensor, torch.Tensor]:
        """

        Args:
            query (Tensor): The query of the attention function.
                See "Attention Is All You Need" for more details.
            key (Tensor): The keys of the attention function.
                See "Attention Is All You Need" for more details.
            value (Tensor): The values of the attention function.
                See "Attention Is All You Need" for more details.
            attn_mask (BoolTensor, optional): 3D mask that prevents attention to certain positions.
            bias_k (Tensor, optional): one more key and value sequence to be added to keys at
                sequence dim (dim=-3). Those are used for incremental decoding. Users should provide
                ``bias_v``.
            bias_v (Tensor, optional): one more key and value sequence to be added to values at
                sequence dim (dim=-3). Those are used for incremental decoding. Users should also provide
                ``bias_k``.

        Shape:

            - Inputs:

                - query: :math:`(..., L, N, E)`
                - key: :math:`(..., S, N, E)`
                - value: :math:`(..., S, N, E)`
                - attn_mask, bias_k and bias_v: same with the shape of the corresponding args in attention layer.

            - Outputs:

                - attn_output: :math:`(..., L, N, E)`
                - attn_output_weights: :math:`(N * H, L, S)`

            Note: It's optional to have the query/key/value inputs with more than three dimensions (for broadcast purpose).
            The MultiheadAttentionContainer module will operate on the last three dimensions.

            where where L is the target length, S is the sequence length, H is the number of attention heads,
            N is the batch size, and E is the embedding dimension.
        """
        if self.batch_first:
            query, key, value = query.transpose(-3, -2), key.transpose(-3, -2), value.transpose(-3, -2)
        tgt_len, src_len, bsz, embed_dim = query.size(-3), key.size(-3), query.size(-2), query.size(-1)
        q, k, v = self.in_proj_container(query, key, value)
        assert q.size(-1) % self.nhead == 0, "query's embed_dim must be divisible by the number of heads"
        head_dim = q.size(-1) // self.nhead
        q = q.reshape(tgt_len, bsz * self.nhead, head_dim)
        assert k.size(-1) % self.nhead == 0, "key's embed_dim must be divisible by the number of heads"
        head_dim = k.size(-1) // self.nhead
        k = k.reshape(src_len, bsz * self.nhead, head_dim)
        assert v.size(-1) % self.nhead == 0, "value's embed_dim must be divisible by the number of heads"
        head_dim = v.size(-1) // self.nhead
        v = v.reshape(src_len, bsz * self.nhead, head_dim)
        attn_output, attn_output_weights = self.attention_layer(q, k, v, attn_mask=attn_mask, bias_k=bias_k, bias_v=bias_v)
        attn_output = attn_output.reshape(tgt_len, bsz, embed_dim)
        attn_output = self.out_proj(attn_output)
        if self.batch_first:
            attn_output = attn_output.transpose(-3, -2)
        return attn_output, attn_output_weights


class ScaledDotProduct(torch.nn.Module):

    def __init__(self, dropout=0.0, batch_first=False) ->None:
        """Processes a projected query and key-value pair to apply
        scaled dot product attention.

        Args:
            dropout (float): probability of dropping an attention weight.
            batch_first: If ``True``, then the input and output tensors are provided
                as `(batch, seq, feature)`. Default: ``False``

        Examples::
            >>> import torch, torchtext
            >>> SDP = torchtext.nn.ScaledDotProduct(dropout=0.1)
            >>> q = torch.randn(21, 256, 3)
            >>> k = v = torch.randn(21, 256, 3)
            >>> attn_output, attn_weights = SDP(q, k, v)
            >>> print(attn_output.shape, attn_weights.shape)
            torch.Size([21, 256, 3]) torch.Size([256, 21, 21])
        """
        super(ScaledDotProduct, self).__init__()
        self.dropout = dropout
        self.batch_first = batch_first

    def forward(self, query: 'torch.Tensor', key: 'torch.Tensor', value: 'torch.Tensor', attn_mask: 'Optional[torch.Tensor]'=None, bias_k: 'Optional[torch.Tensor]'=None, bias_v: 'Optional[torch.Tensor]'=None) ->Tuple[torch.Tensor, torch.Tensor]:
        """Uses a scaled dot product with the projected key-value pair to update
        the projected query.

        Args:
            query (Tensor): Projected query
            key (Tensor): Projected key
            value (Tensor): Projected value
            attn_mask (BoolTensor, optional): 3D mask that prevents attention to certain positions.
            attn_mask (BoolTensor, optional): 3D mask that prevents attention to certain positions.
            bias_k (Tensor, optional): one more key and value sequence to be added to keys at
                sequence dim (dim=-3). Those are used for incremental decoding. Users should provide
                ``bias_v``.
            bias_v (Tensor, optional): one more key and value sequence to be added to values at
                sequence dim (dim=-3). Those are used for incremental decoding. Users should also provide
                ``bias_k``.

        Shape:
            - query: :math:`(..., L, N * H, E / H)`
            - key: :math:`(..., S, N * H, E / H)`
            - value: :math:`(..., S, N * H, E / H)`
            - attn_mask: :math:`(N * H, L, S)`, positions with ``True`` are not allowed to attend
                while ``False`` values will be unchanged.
            - bias_k and bias_v:bias: :math:`(1, N * H, E / H)`

            - Output: :math:`(..., L, N * H, E / H)`, :math:`(N * H, L, S)`

            Note: It's optional to have the query/key/value inputs with more than three dimensions (for broadcast purpose).
                The ScaledDotProduct module will operate on the last three dimensions.

            where L is the target length, S is the source length, H is the number
            of attention heads, N is the batch size, and E is the embedding dimension.
        """
        if self.batch_first:
            query, key, value = query.transpose(-3, -2), key.transpose(-3, -2), value.transpose(-3, -2)
        if bias_k is not None and bias_v is not None:
            assert key.size(-1) == bias_k.size(-1) and key.size(-2) == bias_k.size(-2) and bias_k.size(-3) == 1, 'Shape of bias_k is not supported'
            assert value.size(-1) == bias_v.size(-1) and value.size(-2) == bias_v.size(-2) and bias_v.size(-3) == 1, 'Shape of bias_v is not supported'
            key = torch.cat([key, bias_k])
            value = torch.cat([value, bias_v])
            if attn_mask is not None:
                attn_mask = torch.nn.functional.pad(attn_mask, (0, 1))
        tgt_len, head_dim = query.size(-3), query.size(-1)
        assert query.size(-1) == key.size(-1) == value.size(-1), 'The feature dim of query, key, value must be equal.'
        assert key.size() == value.size(), 'Shape of key, value must match'
        src_len = key.size(-3)
        batch_heads = max(query.size(-2), key.size(-2))
        query, key, value = query.transpose(-2, -3), key.transpose(-2, -3), value.transpose(-2, -3)
        query = query * float(head_dim) ** -0.5
        if attn_mask is not None:
            if attn_mask.dim() != 3:
                raise RuntimeError('attn_mask must be a 3D tensor.')
            if attn_mask.size(-1) != src_len or attn_mask.size(-2) != tgt_len or attn_mask.size(-3) != 1 and attn_mask.size(-3) != batch_heads:
                raise RuntimeError('The size of the attn_mask is not correct.')
            if attn_mask.dtype != torch.bool:
                raise RuntimeError('Only bool tensor is supported for attn_mask')
        attn_output_weights = torch.matmul(query, key.transpose(-2, -1))
        if attn_mask is not None:
            attn_output_weights.masked_fill_(attn_mask, -100000000.0)
        attn_output_weights = torch.nn.functional.softmax(attn_output_weights, dim=-1)
        attn_output_weights = torch.nn.functional.dropout(attn_output_weights, p=self.dropout, training=self.training)
        attn_output = torch.matmul(attn_output_weights, value)
        if self.batch_first:
            return attn_output, attn_output_weights
        else:
            return attn_output.transpose(-3, -2), attn_output_weights


class InProjContainer(torch.nn.Module):

    def __init__(self, query_proj, key_proj, value_proj) ->None:
        """A in-proj container to project query/key/value in MultiheadAttention. This module happens before reshaping
        the projected query/key/value into multiple heads. See the linear layers (bottom) of Multi-head Attention in
        Fig 2 of Attention Is All You Need paper. Also check the usage example
        in torchtext.nn.MultiheadAttentionContainer.

        Args:
            query_proj: a proj layer for query. A typical projection layer is torch.nn.Linear.
            key_proj: a proj layer for key. A typical projection layer is torch.nn.Linear.
            value_proj: a proj layer for value. A typical projection layer is torch.nn.Linear.
        """
        super(InProjContainer, self).__init__()
        self.query_proj = query_proj
        self.key_proj = key_proj
        self.value_proj = value_proj

    def forward(self, query: 'torch.Tensor', key: 'torch.Tensor', value: 'torch.Tensor') ->Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Projects the input sequences using in-proj layers. query/key/value are simply passed to
        the forward func of query/key/value_proj, respectively.

        Args:
            query (Tensor): The query to be projected.
            key (Tensor): The keys to be projected.
            value (Tensor): The values to be projected.

        Examples::
            >>> import torch
            >>> from torchtext.nn import InProjContainer
            >>> embed_dim, bsz = 10, 64
            >>> in_proj_container = InProjContainer(torch.nn.Linear(embed_dim, embed_dim),
                                                    torch.nn.Linear(embed_dim, embed_dim),
                                                    torch.nn.Linear(embed_dim, embed_dim))
            >>> q = torch.rand((5, bsz, embed_dim))
            >>> k = v = torch.rand((6, bsz, embed_dim))
            >>> q, k, v = in_proj_container(q, k, v)

        """
        return self.query_proj(query), self.key_proj(key), self.value_proj(value)


MODEL_KWARGS_TYPE = Dict[str, Union[bool, torch.Tensor, Optional[List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]], SEQ_2_SEQ_OUTPUTS_TYPE]]


class BasicEnglishNormalize(nn.Module):
    __jit_unused_properties__ = ['is_jitable']
    """Basic normalization for a string sentence.

    Args:
        regex_tokenizer (torch.classes.torchtext.RegexTokenizer or torchtext._torchtext.RegexTokenizer): a cpp regex tokenizer object.
    """

    def __init__(self, regex_tokenizer) ->None:
        super(BasicEnglishNormalize, self).__init__()
        self.regex_tokenizer = regex_tokenizer

    @property
    def is_jitable(self):
        return not isinstance(self.regex_tokenizer, RegexTokenizerPybind)

    def forward(self, line: 'str') ->List[str]:
        """
        Args:
            lines (str): a text string to tokenize.

        Returns:
            List[str]: a token list after normalizing and splitting on whitespace.
        """
        return self.regex_tokenizer.forward(line)

    def __prepare_scriptable__(self):
        """Return a JITable BasicEnglishNormalize."""
        regex_tokenizer = torch.classes.torchtext.RegexTokenizer(self.regex_tokenizer.patterns_, self.regex_tokenizer.replacements_, True)
        return BasicEnglishNormalize(regex_tokenizer)


class SentencePieceTokenizer(Module):
    """
    Transform for Sentence Piece tokenizer from pre-trained sentencepiece model

    Additional details: https://github.com/google/sentencepiece

    :param sp_model_path: Path to pre-trained sentencepiece model
    :type sp_model_path: str

    Example
        >>> from torchtext.transforms import SentencePieceTokenizer
        >>> transform = SentencePieceTokenizer("spm_model")
        >>> transform(["hello world", "attention is all you need!"])
    """

    def __init__(self, sp_model_path: 'str') ->None:
        super().__init__()
        self.sp_model = load_sp_model(get_asset_local_path(sp_model_path))

    def forward(self, input: 'Any') ->Any:
        """
        :param input: Input sentence or list of sentences on which to apply tokenizer.
        :type input: Union[str, List[str]]
        :return: tokenized text
        :rtype: Union[List[str], List[List[str]]]
        """
        if torch.jit.isinstance(input, List[str]):
            tokens: 'List[List[str]]' = []
            for text in input:
                tokens.append(self.sp_model.EncodeAsPieces(text))
            return tokens
        elif torch.jit.isinstance(input, str):
            return self.sp_model.EncodeAsPieces(input)
        else:
            raise TypeError('Input type not supported')


class SentencePieceProcessor(nn.Module):
    """String to ids transform based on a pretained sentencepiece model

    Args:
       spm_model: the sentencepiece model instance
    """

    def __init__(self, spm_model) ->None:
        super(SentencePieceProcessor, self).__init__()
        self.sp_model = spm_model

    def forward(self, line: 'str') ->List[int]:
        """
        Args:
            line: the input sentence string

        Examples:
            >>> spm_processor('the pretrained sp model names')
            >>> [9, 1546, 18811, 2849, 2759, 2202]
        """
        return self.sp_model.EncodeAsIds(line)

    @torch.jit.export
    def decode(self, ids: 'List[int]') ->str:
        """
        Args:
            ids: the integers list for decoder

        Examples:
            >>> spm_processor.decoder([9, 1546, 18811, 2849, 2759, 2202])
            >>> 'the pretrained sp model names'
        """
        return self.sp_model.DecodeIds(ids)

    def __prepare_scriptable__(self):
        torchbind_spm = torch.classes.torchtext.SentencePiece(self.sp_model._return_content())
        return SentencePieceProcessor(torchbind_spm)


def _log_class_usage(klass):
    identifier = 'torchtext'
    if klass and hasattr(klass, '__name__'):
        identifier += f'.{klass.__name__}'
    torch._C._log_api_usage_once(identifier)


class Vocab(nn.Module):
    __jit_unused_properties__ = ['is_jitable']
    """Creates a vocab object which maps tokens to indices.

    Args:
        vocab (torch.classes.torchtext.Vocab or torchtext._torchtext.Vocab): a cpp vocab object.
    """

    def __init__(self, vocab) ->None:
        super(Vocab, self).__init__()
        self.vocab = vocab
        _log_class_usage(__class__)

    @property
    def is_jitable(self):
        return isinstance(self.vocab, torch._C.ScriptObject)

    @torch.jit.export
    def forward(self, tokens: 'List[str]') ->List[int]:
        """Calls the `lookup_indices` method

        Args:
            tokens: a list of tokens used to lookup their corresponding `indices`.

        Returns:
            The indices associated with a list of `tokens`.
        """
        return self.vocab.lookup_indices(tokens)

    @torch.jit.export
    def __len__(self) ->int:
        """
        Returns:
            The length of the vocab.
        """
        return len(self.vocab)

    @torch.jit.export
    def __contains__(self, token: 'str') ->bool:
        """
        Args:
            token: The token for which to check the membership.

        Returns:
            Whether the token is member of vocab or not.
        """
        return self.vocab.__contains__(token)

    @torch.jit.export
    def __getitem__(self, token: 'str') ->int:
        """
        Args:
            token: The token used to lookup the corresponding index.

        Returns:
            The index corresponding to the associated token.
        """
        return self.vocab[token]

    @torch.jit.export
    def set_default_index(self, index: 'Optional[int]') ->None:
        """
        Args:
            index: Value of default index. This index will be returned when OOV token is queried.
        """
        self.vocab.set_default_index(index)

    @torch.jit.export
    def get_default_index(self) ->Optional[int]:
        """
        Returns:
            Value of default index if it is set.
        """
        return self.vocab.get_default_index()

    @torch.jit.export
    def insert_token(self, token: 'str', index: 'int') ->None:
        """
        Args:
            token: The token used to lookup the corresponding index.
            index: The index corresponding to the associated token.
        Raises:
            RuntimeError: If `index` is not in range [0, Vocab.size()] or if `token` already exists in the vocab.
        """
        self.vocab.insert_token(token, index)

    @torch.jit.export
    def append_token(self, token: 'str') ->None:
        """
        Args:
            token: The token used to lookup the corresponding index.

        Raises:
            RuntimeError: If `token` already exists in the vocab
        """
        self.vocab.append_token(token)

    @torch.jit.export
    def lookup_token(self, index: 'int') ->str:
        """
        Args:
            index: The index corresponding to the associated token.

        Returns:
            token: The token used to lookup the corresponding index.

        Raises:
            RuntimeError: If `index` not in range [0, itos.size()).
        """
        return self.vocab.lookup_token(index)

    @torch.jit.export
    def lookup_tokens(self, indices: 'List[int]') ->List[str]:
        """
        Args:
            indices: The `indices` used to lookup their corresponding`tokens`.

        Returns:
            The `tokens` associated with `indices`.

        Raises:
            RuntimeError: If an index within `indices` is not int range [0, itos.size()).
        """
        return self.vocab.lookup_tokens(indices)

    @torch.jit.export
    def lookup_indices(self, tokens: 'List[str]') ->List[int]:
        """
        Args:
            tokens: the tokens used to lookup their corresponding `indices`.

        Returns:
            The 'indices` associated with `tokens`.
        """
        return self.vocab.lookup_indices(tokens)

    @torch.jit.export
    def get_stoi(self) ->Dict[str, int]:
        """
        Returns:
            Dictionary mapping tokens to indices.
        """
        return self.vocab.get_stoi()

    @torch.jit.export
    def get_itos(self) ->List[str]:
        """
        Returns:
            List mapping indices to tokens.
        """
        return self.vocab.get_itos()

    def __prepare_scriptable__(self):
        """Return a JITable Vocab."""
        if not self.is_jitable:
            cpp_vocab = torch.classes.torchtext.Vocab(self.vocab.itos_, self.vocab.default_index_)
            return Vocab(cpp_vocab)
        return self


class VocabTransform(Module):
    """Vocab transform to convert input batch of tokens into corresponding token ids

    :param vocab: an instance of :class:`torchtext.vocab.Vocab` class.

    Example:
        >>> import torch
        >>> from torchtext.vocab import vocab
        >>> from torchtext.transforms import VocabTransform
        >>> from collections import OrderedDict
        >>> vocab_obj = vocab(OrderedDict([('a', 1), ('b', 1), ('c', 1)]))
        >>> vocab_transform = VocabTransform(vocab_obj)
        >>> output = vocab_transform([['a','b'],['a','b','c']])
        >>> jit_vocab_transform = torch.jit.script(vocab_transform)
    """

    def __init__(self, vocab: 'Vocab') ->None:
        super().__init__()
        assert isinstance(vocab, Vocab)
        self.vocab = vocab

    def forward(self, input: 'Any') ->Any:
        """
        :param input: Input batch of token to convert to correspnding token ids
        :type input: Union[List[str], List[List[str]]]
        :return: Converted input into corresponding token ids
        :rtype: Union[List[int], List[List[int]]]
        """
        if torch.jit.isinstance(input, List[str]):
            return self.vocab.lookup_indices(input)
        elif torch.jit.isinstance(input, List[List[str]]):
            output: 'List[List[int]]' = []
            for tokens in input:
                output.append(self.vocab.lookup_indices(tokens))
            return output
        else:
            raise TypeError('Input type not supported')


class VectorTransform(nn.Module):
    """Vector transform

    Args:
        vector: an instance of torchtext.experimental.vectors.Vectors class.

    Example:
        >>> import torch
        >>> from torchtext.experimental.vectors import FastText
        >>> vector_transform = VectorTransform(FastText())
        >>> jit_vector_transform = torch.jit.script(vector_transform)
    """

    def __init__(self, vector) ->None:
        super(VectorTransform, self).__init__()
        self.vector = vector

    def forward(self, tokens: 'List[str]') ->Tensor:
        """

        Args:
            tokens: a string token list

        Example:
            >>> vector_transform(['here', 'is', 'an', 'example'])

        """
        return self.vector.lookup_vectors(tokens)


def _infer_shape(f):
    num_lines, vector_dim = 0, None
    for line in f:
        if vector_dim is None:
            row = line.rstrip().split(b' ')
            vector = row[1:]
            if len(vector) > 2:
                vector_dim = len(vector)
                num_lines += 1
        else:
            num_lines += 1
    f.seek(0)
    return num_lines, vector_dim


SUPPORTED_DATASETS = {'valid_test': ['dev2010', 'tst2010', 'tst2011', 'tst2012', 'tst2013', 'tst2014'], 'language_pair': {'en': ['ar', 'de', 'fr', 'cs'], 'ar': ['en'], 'fr': ['en'], 'de': ['en'], 'cs': ['en']}, 'year': 16}


SUPPORTED_LANGPAIRS = [(k, e) for k, v in SUPPORTED_DATASETS['language_pair'].items() for e in v]


def reporthook(t):
    """
    https://github.com/tqdm/tqdm.
    """
    last_b = [0]

    def inner(b=1, bsize=1, tsize=None):
        """
        b: int, optional
        Number of blocks just transferred [default: 1].
        bsize: int, optional
        Size of each block (in tqdm units) [default: 1].
        tsize: int, optional
        Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if tsize is not None:
            t.total = tsize
        t.update((b - last_b[0]) * bsize)
        last_b[0] = b
    return inner


class Vectors:

    def __init__(self, name, cache=None, url=None, unk_init=None, max_vectors=None) ->None:
        """
        Args:

            name: name of the file that contains the vectors
            cache: directory for cached vectors
            url: url for download if vectors not found in cache
            unk_init (callback): by default, initialize out-of-vocabulary word vectors
                to zero vectors; can be any function that takes in a Tensor and returns a Tensor of the same size
            max_vectors (int): this can be used to limit the number of
                pre-trained vectors loaded.
                Most pre-trained vector sets are sorted
                in the descending order of word frequency.
                Thus, in situations where the entire set doesn't fit in memory,
                or is not needed for another reason, passing `max_vectors`
                can limit the size of the loaded set.
        """
        cache = '.vector_cache' if cache is None else cache
        self.itos = None
        self.stoi = None
        self.vectors = None
        self.dim = None
        self.unk_init = torch.Tensor.zero_ if unk_init is None else unk_init
        self.cache(name, cache, url=url, max_vectors=max_vectors)

    def __getitem__(self, token):
        if token in self.stoi:
            return self.vectors[self.stoi[token]]
        else:
            return self.unk_init(torch.Tensor(self.dim))

    def __contains__(self, token):
        return token in self.stoi

    def cache(self, name, cache, url=None, max_vectors=None):
        ssl._create_default_https_context = ssl._create_unverified_context
        if os.path.isfile(name):
            path = name
            if max_vectors:
                file_suffix = '_{}.pt'.format(max_vectors)
            else:
                file_suffix = '.pt'
            path_pt = os.path.join(cache, os.path.basename(name)) + file_suffix
        else:
            path = os.path.join(cache, name)
            if max_vectors:
                file_suffix = '_{}.pt'.format(max_vectors)
            else:
                file_suffix = '.pt'
            path_pt = path + file_suffix
        if not os.path.isfile(path_pt):
            if not os.path.isfile(path) and url:
                logger.info('Downloading vectors from {}'.format(url))
                if not os.path.exists(cache):
                    os.makedirs(cache)
                dest = os.path.join(cache, os.path.basename(url))
                if not os.path.isfile(dest):
                    with tqdm(unit='B', unit_scale=True, miniters=1, desc=dest) as t:
                        try:
                            urlretrieve(url, dest, reporthook=reporthook(t))
                        except KeyboardInterrupt as e:
                            os.remove(dest)
                            raise e
                logger.info('Extracting vectors into {}'.format(cache))
                ext = os.path.splitext(dest)[1][1:]
                if ext == 'zip':
                    with zipfile.ZipFile(dest, 'r') as zf:
                        zf.extractall(cache)
                elif ext == 'gz':
                    if dest.endswith('.tar.gz'):
                        with tarfile.open(dest, 'r:gz') as tar:
                            tar.extractall(path=cache)
            if not os.path.isfile(path):
                raise RuntimeError('no vectors found at {}'.format(path))
            logger.info('Loading vectors from {}'.format(path))
            ext = os.path.splitext(path)[1][1:]
            if ext == 'gz':
                open_file = gzip.open
            else:
                open_file = open
            vectors_loaded = 0
            with open_file(path, 'rb') as f:
                num_lines, dim = _infer_shape(f)
                if not max_vectors or max_vectors > num_lines:
                    max_vectors = num_lines
                itos, vectors, dim = [], torch.zeros((max_vectors, dim)), None
                for line in tqdm(f, total=max_vectors):
                    entries = line.rstrip().split(b' ')
                    word, entries = entries[0], entries[1:]
                    if dim is None and len(entries) > 1:
                        dim = len(entries)
                    elif len(entries) == 1:
                        logger.warning('Skipping token {} with 1-dimensional vector {}; likely a header'.format(word, entries))
                        continue
                    elif dim != len(entries):
                        raise RuntimeError('Vector for token {} has {} dimensions, but previously read vectors have {} dimensions. All vectors must have the same number of dimensions.'.format(word, len(entries), dim))
                    try:
                        if isinstance(word, bytes):
                            word = word.decode('utf-8')
                    except UnicodeDecodeError:
                        logger.info('Skipping non-UTF8 token {}'.format(repr(word)))
                        continue
                    vectors[vectors_loaded] = torch.tensor([float(x) for x in entries])
                    vectors_loaded += 1
                    itos.append(word)
                    if vectors_loaded == max_vectors:
                        break
            self.itos = itos
            self.stoi = {word: i for i, word in enumerate(itos)}
            self.vectors = torch.Tensor(vectors).view(-1, dim)
            self.dim = dim
            logger.info('Saving vectors to {}'.format(path_pt))
            if not os.path.exists(cache):
                os.makedirs(cache)
            torch.save((self.itos, self.stoi, self.vectors, self.dim), path_pt)
        else:
            logger.info('Loading vectors from {}'.format(path_pt))
            self.itos, self.stoi, self.vectors, self.dim = torch.load(path_pt)

    def __len__(self):
        return len(self.vectors)

    def get_vecs_by_tokens(self, tokens, lower_case_backup=False):
        """Look up embedding vectors of tokens.

        Args:
            tokens: a token or a list of tokens. if `tokens` is a string,
                returns a 1-D tensor of shape `self.dim`; if `tokens` is a
                list of strings, returns a 2-D tensor of shape=(len(tokens),
                self.dim).
            lower_case_backup : Whether to look up the token in the lower case.
                If False, each token in the original case will be looked up;
                if True, each token in the original case will be looked up first,
                if not found in the keys of the property `stoi`, the token in the
                lower case will be looked up. Default: False.

        Examples:
            >>> examples = ['chip', 'baby', 'Beautiful']
            >>> vec = text.vocab.GloVe(name='6B', dim=50)
            >>> ret = vec.get_vecs_by_tokens(examples, lower_case_backup=True)
        """
        to_reduce = False
        if not isinstance(tokens, list):
            tokens = [tokens]
            to_reduce = True
        if not lower_case_backup:
            indices = [self[token] for token in tokens]
        else:
            indices = [(self[token] if token in self.stoi else self[token.lower()]) for token in tokens]
        vecs = torch.stack(indices)
        return vecs[0] if to_reduce else vecs


class ToTensor(Module):
    """Convert input to torch tensor

    :param padding_value: Pad value to make each input in the batch of length equal to the longest sequence in the batch.
    :type padding_value: Optional[int]
    :param dtype: :class:`torch.dtype` of output tensor
    :type dtype: :class:`torch.dtype`
    """

    def __init__(self, padding_value: 'Optional[int]'=None, dtype: 'torch.dtype'=torch.long) ->None:
        super().__init__()
        self.padding_value = padding_value
        self.dtype = dtype

    def forward(self, input: 'Any') ->Tensor:
        """
        :param input: Sequence or batch of token ids
        :type input: Union[List[int], List[List[int]]]
        :rtype: Tensor
        """
        return F.to_tensor(input, padding_value=self.padding_value, dtype=self.dtype)


class LabelToIndex(Module):
    """
    Transform labels from string names to ids.

    :param label_names: a list of unique label names
    :type label_names: Optional[List[str]]
    :param label_path: a path to file containing unique label names containing 1 label per line. Note that either label_names or label_path should be supplied
                       but not both.
    :type label_path: Optional[str]
    """

    def __init__(self, label_names: 'Optional[List[str]]'=None, label_path: 'Optional[str]'=None, sort_names=False) ->None:
        assert label_names or label_path, 'label_names or label_path is required'
        assert not (label_names and label_path), 'label_names and label_path are mutually exclusive'
        super().__init__()
        if label_path:
            with open(label_path, 'r') as f:
                label_names = [line.strip() for line in f if line.strip()]
        else:
            label_names = label_names
        if sort_names:
            label_names = sorted(label_names)
        self._label_vocab = Vocab(torch.classes.torchtext.Vocab(label_names, None))
        self._label_names = self._label_vocab.get_itos()

    def forward(self, input: 'Any') ->Any:
        """
        :param input: Input labels to convert to corresponding ids
        :type input: Union[str, List[str]]
        :rtype: Union[int, List[int]]
        """
        if torch.jit.isinstance(input, List[str]):
            return self._label_vocab.lookup_indices(input)
        elif torch.jit.isinstance(input, str):
            return self._label_vocab.__getitem__(input)
        else:
            raise TypeError('Input type not supported')

    @property
    def label_names(self) ->List[str]:
        return self._label_names


class Truncate(Module):
    """Truncate input sequence

    :param max_seq_len: The maximum allowable length for input sequence
    :type max_seq_len: int
    """

    def __init__(self, max_seq_len: 'int') ->None:
        super().__init__()
        self.max_seq_len = max_seq_len

    def forward(self, input: 'Any') ->Any:
        """
        :param input: Input sequence or batch of sequence to be truncated
        :type input: Union[List[Union[str, int]], List[List[Union[str, int]]]]
        :return: Truncated sequence
        :rtype: Union[List[Union[str, int]], List[List[Union[str, int]]]]
        """
        return F.truncate(input, self.max_seq_len)


class AddToken(Module):
    """Add token to beginning or end of sequence

    :param token: The token to be added
    :type token: Union[int, str]
    :param begin: Whether to insert token at start or end or sequence, defaults to True
    :type begin: bool, optional
    """

    def __init__(self, token: 'Union[int, str]', begin: 'bool'=True) ->None:
        super().__init__()
        self.token = token
        self.begin = begin

    def forward(self, input: 'Any') ->Any:
        """
        :param input: Input sequence or batch
        :type input: Union[List[Union[str, int]], List[List[Union[str, int]]]]
        """
        return F.add_token(input, self.token, self.begin)


class PadTransform(Module):
    """Pad tensor to a fixed length with given padding value.

    :param max_length: Maximum length to pad to
    :type max_length: int
    :param pad_value: Value to pad the tensor with
    :type pad_value: bool
    """

    def __init__(self, max_length: 'int', pad_value: 'int') ->None:
        super().__init__()
        self.max_length = max_length
        self.pad_value = float(pad_value)

    def forward(self, x: 'Tensor') ->Tensor:
        """
        :param x: The tensor to pad
        :type x: Tensor
        :return: Tensor padded up to max_length with pad_value
        :rtype: Tensor
        """
        max_encoded_length = x.size(-1)
        if max_encoded_length < self.max_length:
            pad_amount = self.max_length - max_encoded_length
            x = torch.nn.functional.pad(x, (0, pad_amount), value=self.pad_value)
        return x


class StrToIntTransform(Module):
    """Convert string tokens to integers (either single sequence or batch)."""

    def __init__(self) ->None:
        super().__init__()

    def forward(self, input: 'Any') ->Any:
        """
        :param input: sequence or batch of string tokens to convert
        :type input: Union[List[str], List[List[str]]]
        :return: sequence or batch converted into corresponding token ids
        :rtype: Union[List[int], List[List[int]]]
        """
        return F.str_to_int(input)


@lru_cache()
def bytes_to_unicode():
    """
    Original Source: https://github.com/openai/gpt-2/blob/master/src/encoder.py#L9

    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a significant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.
    """
    bs = list(range(ord('!'), ord('~') + 1)) + list(range(ord('¡'), ord('¬') + 1)) + list(range(ord('®'), ord('ÿ') + 1))
    cs = bs[:]
    n = 0
    for b in range(2 ** 8):
        if b not in bs:
            bs.append(b)
            cs.append(2 ** 8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


class GPT2BPETokenizer(Module):
    """
    Transform for GPT-2 BPE Tokenizer.

    Reimplements openai GPT-2 BPE in TorchScript. Original openai implementation
    https://github.com/openai/gpt-2/blob/master/src/encoder.py

    :param encoder_json_path: Path to GPT-2 BPE encoder json file.
    :type encoder_json_path: str
    :param vocab_bpe_path: Path to bpe vocab file.
    :type vocab_bpe_path: str
    :param return_tokens: Indicate whether to return split tokens. If False, it will return encoded token IDs as strings (default: False)
    :type return_input: bool
    """
    SPECIAL_TOKENS_ATTRIBUTES = ['bos_token', 'eos_token', 'unk_token', 'sep_token', 'pad_token', 'cls_token', 'mask_token', 'additional_special_tokens']
    __jit_unused_properties__ = ['is_jitable']
    _seperator: 'torch.jit.Final[str]'

    def __init__(self, encoder_json_path: 'str', vocab_bpe_path: 'str', return_tokens: 'bool'=False) ->None:
        super().__init__()
        self._seperator = '\x01'
        with open(get_asset_local_path(encoder_json_path), 'r', encoding='utf-8') as f:
            bpe_encoder = json.load(f)
        with open(get_asset_local_path(vocab_bpe_path), 'r', encoding='utf-8') as f:
            bpe_vocab = f.read()
        bpe_merge_ranks = {self._seperator.join(merge_pair.split()): i for i, merge_pair in enumerate(bpe_vocab.split('\n')[1:-1])}
        self.bpe = GPT2BPEEncoderPyBind(bpe_encoder, bpe_merge_ranks, self._seperator, bytes_to_unicode(), True)
        self._return_tokens = return_tokens

    @property
    def is_jitable(self):
        return isinstance(self.bpe, torch._C.ScriptObject)

    @torch.jit.export
    def _encode(self, text: 'str') ->List[str]:
        """Encode text into a list of tokens IDs

        Args:
            text: An input text string.

        Returns:
            A list of bpe token ids represents each bpe tokens

        For example: "awesome,awe"
            --> bpe --> bpe tokens: ["aw", "esome"], [","], ["aw", e]
            --> bpe encode --> bpe token ids: [707, 5927, 11, 707, 68]
        """
        bpe_token_ids: 'List[int]' = self.bpe.encode(text)
        bpe_tokens: 'List[str]' = []
        for bpe_token_id in bpe_token_ids:
            bpe_tokens.append(str(bpe_token_id))
        return bpe_tokens

    @torch.jit.export
    def _tokenize(self, text: 'str') ->List[str]:
        """Tokenize text into a list of tokens

        Args:
            text: An input text string.

        Returns:
            A list of bpe token ids represents each bpe tokens

        For example: "awesome,awe"
            --> bpe --> bpe tokens: ["aw", "esome"], [","], ["aw", e]
        """
        return self.bpe.tokenize(text)

    def add_special_tokens(self, special_tokens_dict: 'Mapping[str, Union[str, Sequence[str]]]') ->int:
        """Add a dictionary of special tokens (eos, pad, cls…) to the encoder

        :param special_tokens_dict: dict of string. Keys should be in the list of predefined special attributes:
        [bos_token, eos_token, unk_token, sep_token, pad_token, cls_token, mask_token, additional_special_tokens].
        Tokens are only added if they are not already in the vocabulary.
        :type special_tokens_dict: Dict[str, Union[str, List[str]]]
        :return: Number of tokens added to the vocabulary.
        :rtype: int
        """
        for key in special_tokens_dict.keys():
            assert key in self.SPECIAL_TOKENS_ATTRIBUTES, f"Key '{key}' is not in the special token list: {self.SPECIAL_TOKENS_ATTRIBUTES}"
        return self.bpe.add_special_tokens({k: v for k, v in special_tokens_dict.items() if k != 'additional_special_tokens'}, special_tokens_dict.get('additional_special_tokens', []))

    def forward(self, input: 'Any') ->Any:
        """
        :param input: Input sentence or list of sentences on which to apply tokenizer.
        :type input: Union[str, List[str]]
        :return: tokenized text
        :rtype: Union[List[str], List[List(str)]]
        """
        if torch.jit.isinstance(input, List[str]):
            tokens: 'List[List[str]]' = []
            for text in input:
                if self._return_tokens:
                    tokens.append(self._tokenize(text))
                else:
                    tokens.append(self._encode(text))
            return tokens
        elif torch.jit.isinstance(input, str):
            if self._return_tokens:
                return self._tokenize(input)
            else:
                return self._encode(input)
        else:
            raise TypeError('Input type not supported')

    def __prepare_scriptable__(self):
        """Return a JITable tokenizer."""
        if not self.is_jitable:
            tokenizer_copy = deepcopy(self)
            tokenizer_copy.bpe = torch.classes.torchtext.GPT2BPEEncoder(self.bpe.bpe_encoder_, self.bpe.bpe_merge_ranks_, self.bpe.seperator_, self.bpe.byte_encoder_, False)
            return tokenizer_copy
        return self

    @torch.jit.export
    def decode(self, tokens: 'List[str]') ->str:
        """Return a decoded string given a list of string token ids.

        :param input: A list of strings, each string corresponds to token ids.
        :type input: List[str]
        :return: decoded text
        :rtype: str
        """
        return self.bpe.decode([int(token) for token in tokens])


def get_pairs(word):
    """
    Return set of symbol pairs in a word.
    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


class CharBPETokenizer(Module):
    """
    Transform for a Character Byte-Pair-Encoding Tokenizer.

    Args:
        :param bpe_encoder_path: Path to the BPE encoder json file.
        :type bpe_encoder_path: str
        :param bpe_merges_path: Path to the BPE merges text file.
        :type bpe_merges_path: str
        :param return_tokens: Indicate whether to return split tokens. If False, it will return encoded token IDs (default: False).
        :type return_tokens: bool
        :param unk_token: The unknown token. If provided, it must exist in encoder.
        :type unk_token: Optional[str]
        :param suffix: The suffix to be used for every subword that is an end-of-word.
        :type suffix: Optional[str]
        :param special_tokens: Special tokens which should not be split into individual characters. If provided, these must exist in encoder.
        :type special_tokens: Optional[List[str]]
    """

    def __init__(self, bpe_encoder_path: 'str', bpe_merges_path: 'str', return_tokens: 'bool'=False, unk_token: 'Optional[str]'=None, suffix: 'Optional[str]'=None, special_tokens: 'Optional[List[str]]'=None):
        super().__init__()
        with open(get_asset_local_path(bpe_encoder_path), 'r') as f:
            self._encoder = dict(json.load(f))
        with open(get_asset_local_path(bpe_merges_path), 'r', encoding='utf-8') as f:
            bpe_data = f.read()
        merges = [tuple(merge_str.split()) for merge_str in bpe_data.split('\n')[1:-1]]
        self._decoder = {v: k for k, v in self._encoder.items()}
        self._bpe_ranks = dict(zip(merges, range(len(merges))))
        self._return_tokens = return_tokens
        self._cache = {}
        self._pat = '\\S+\\n?'
        if unk_token and unk_token not in self._encoder:
            raise RuntimeError('Unknown token {} not found in encoder. Special tokens must be in encoder.'.format(unk_token))
        self._unk_token = unk_token
        self._suffix = suffix
        if special_tokens:
            for token in special_tokens:
                if token not in self._encoder:
                    raise RuntimeError('Special token {} not found in encoder. Special tokens must be in encoder.'.format(token))
                else:
                    self._cache[token] = token

    @property
    def vocab_size(self):
        return len(self._encoder)

    def _bpe(self, token):
        """Splits the input token into bpe tokens. The output depends on the encoder and merge list specified in the class
        constructor. For example, _bpe("pytorch") may return "p y t o r c h" or "py tor ch" or "pytorch" depending on which
        merges exist.

        Args:
            text: An input text string.

        Returns:
            A string of space separated bpe tokens.
        """
        if token in self._cache:
            return self._cache[token]
        if self._suffix:
            word = tuple(token[:-1]) + (token[-1] + self._suffix,)
        else:
            word = tuple(token)
        pairs = get_pairs(word)
        if not pairs:
            if self._suffix:
                return token + self._suffix
            else:
                return token
        while True:
            bigram = min(pairs, key=lambda pair: self._bpe_ranks.get(pair, float('inf')))
            if bigram not in self._bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                except ValueError:
                    new_word.extend(word[i:])
                    break
                else:
                    new_word.extend(word[i:j])
                    i = j
                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = ' '.join(word)
        self._cache[token] = word
        return word

    def encode(self, text: 'str') ->Union[List[int], List[str]]:
        """Encode text into a list of tokens IDs

        Args:
            text: An input text string.

        Returns:
            A list of bpe token ids represents each bpe token. Return type depends on provided encoder file.
        """
        encoded_tokens = [(self._encoder.get(bpe_token, self._encoder.get(self._unk_token)) if self._unk_token else self._encoder[bpe_token]) for bpe_token in self._tokenize(text)]
        return encoded_tokens

    def _tokenize(self, text: 'str') ->List[str]:
        """Tokenize text into a list of tokens

        Args:
            text: An input text string.

        Returns:
            A list of bpe token strings
        """
        tokens = []
        for token in re.findall(self._pat, text):
            tokens.extend(bpe_token for bpe_token in self._bpe(token).split(' '))
        return tokens

    def decode(self, tokens: 'Union[List[int], List[str]]') ->str:
        """Decode a list of token IDs into a string

        Args:
            token: A list of IDs (either str or int depending on encoder json)

        Returns:
            A decoded string
        """
        decoded_list = [(self._decoder.get(token, self._unk_token) if self._unk_token else self._decoder[token]) for token in tokens]
        if self._suffix:
            return ''.join(decoded_list).replace(self._suffix, ' ')
        else:
            return ' '.join(decoded_list)

    def forward(self, input: 'Union[str, List[str]]') ->Union[List, List[List]]:
        """Forward method of module encodes strings or list of strings into token ids

        Args:
            input: Input sentence or list of sentences on which to apply tokenizer.

        Returns:
            A list or list of lists of token IDs
        """
        if isinstance(input, List):
            tokens: 'List[List[str]]' = []
            for text in input:
                if self._return_tokens:
                    tokens.append(self._tokenize(text))
                else:
                    tokens.append(self.encode(text))
            return tokens
        elif isinstance(input, str):
            if self._return_tokens:
                return self._tokenize(input)
            else:
                return self.encode(input)
        else:
            raise TypeError('Input type not supported')


class CLIPTokenizer(Module):
    """
    Transform for CLIP Tokenizer. Based on Byte-Level BPE.

    Reimplements CLIP Tokenizer in TorchScript. Original implementation:
    https://github.com/mlfoundations/open_clip/blob/main/src/clip/tokenizer.py

    This tokenizer has been trained to treat spaces like parts of the tokens
    (a bit like sentencepiece) so a word will be encoded differently whether it
    is at the beginning of the sentence (without space) or not.

    The below code snippet shows how to use the CLIP tokenizer with encoder and merges file
    taken from the original paper implementation.

    Example
        >>> from torchtext.transforms import CLIPTokenizer
        >>> MERGES_FILE = "http://download.pytorch.org/models/text/clip_merges.bpe"
        >>> ENCODER_FILE = "http://download.pytorch.org/models/text/clip_encoder.json"
        >>> tokenizer = CLIPTokenizer(merges_path=MERGES_FILE, encoder_json_path=ENCODER_FILE)
        >>> tokenizer("the quick brown fox jumped over the lazy dog")

    :param merges_path: Path to bpe merges file.
    :type merges_path: str
    :param encoder_json_path: Optional, path to BPE encoder json file. When specified, this is used
        to infer num_merges.
    :type encoder_json_path: str
    :param num_merges: Optional, number of merges to read from the bpe merges file.
    :type num_merges: int
    :param return_tokens: Indicate whether to return split tokens. If False, it will return encoded token IDs as strings (default: False)
    :type return_input: bool
    """
    __jit_unused_properties__ = ['is_jitable']
    _seperator: 'torch.jit.Final[str]'

    def __init__(self, merges_path: 'str', encoder_json_path: 'Optional[str]'=None, num_merges: 'Optional[int]'=None, return_tokens: 'bool'=False) ->None:
        super().__init__()
        self._seperator = '\x01'
        with open(get_asset_local_path(merges_path), 'r', encoding='utf-8') as f:
            bpe_merges = f.read().split('\n')[1:]
        if encoder_json_path:
            with open(get_asset_local_path(encoder_json_path), 'r', encoding='utf-8') as f:
                bpe_encoder = json.load(f)
            num_merges = len(bpe_encoder) - (256 * 2 + 2)
            bpe_merge_ranks = {self._seperator.join(merge_pair.split()): i for i, merge_pair in enumerate(bpe_merges[:num_merges])}
        else:
            num_merges = num_merges or len(bpe_merges)
            bpe_merge_ranks = {self._seperator.join(merge_pair.split()): i for i, merge_pair in enumerate(bpe_merges[:num_merges])}
            bpe_vocab = list(bytes_to_unicode().values())
            bpe_vocab = bpe_vocab + [(v + '</w>') for v in bpe_vocab]
            bpe_vocab.extend([''.join(merge_pair.split()) for merge_pair in bpe_merges[:num_merges]])
            bpe_vocab.extend(['<|startoftext|>', '<|endoftext|>'])
            bpe_encoder = {v: i for i, v in enumerate(bpe_vocab)}
        self.bpe = CLIPEncoderPyBind(bpe_encoder, bpe_merge_ranks, self._seperator, bytes_to_unicode(), True)
        self._return_tokens = return_tokens

    @property
    def is_jitable(self):
        return isinstance(self.bpe, torch._C.ScriptObject)

    @torch.jit.export
    def _encode(self, text: 'str') ->List[str]:
        """Encode text into a list of tokens IDs

        Args:
            text: An input text string.

        Returns:
            A list of bpe token ids represents each bpe tokens

        For example: "awesome,awe"
            --> bpe --> bpe tokens: ["aw", "esome"], [","], ["aw", "e"]
            --> bpe encode --> bpe token ids: [707, 5927, 11, 707, 68]
        """
        text = text.lower().strip()
        bpe_token_ids: 'List[int]' = self.bpe.encode(text)
        bpe_tokens: 'List[str]' = []
        for bpe_token_id in bpe_token_ids:
            bpe_tokens.append(str(bpe_token_id))
        return bpe_tokens

    @torch.jit.export
    def _tokenize(self, text: 'str') ->List[str]:
        """Tokenize text into a list of tokens

        Args:
            text: An input text string.

        Returns:
            A list of bpe token ids represents each bpe tokens

        For example: "awesome,awe"
            --> bpe --> bpe tokens: ["aw", "esome"], [","], ["aw", "e"]
        """
        text = text.lower().strip()
        return self.bpe.tokenize(text)

    def forward(self, input: 'Any') ->Any:
        """
        :param input: Input sentence or list of sentences on which to apply tokenizer.
        :type input: Union[str, List[str]]
        :return: tokenized text
        :rtype: Union[List[str], List[List(str)]]
        """
        if torch.jit.isinstance(input, List[str]):
            tokens: 'List[List[str]]' = []
            for text in input:
                if self._return_tokens:
                    tokens.append(self._tokenize(text))
                else:
                    tokens.append(self._encode(text))
            return tokens
        elif torch.jit.isinstance(input, str):
            if self._return_tokens:
                return self._tokenize(input)
            else:
                return self._encode(input)
        else:
            raise TypeError('Input type not supported')

    def __prepare_scriptable__(self):
        """Return a JITable tokenizer."""
        if not self.is_jitable:
            tokenizer_copy = deepcopy(self)
            tokenizer_copy.bpe = torch.classes.torchtext.CLIPEncoder(self.bpe.bpe_encoder_, self.bpe.bpe_merge_ranks_, self.bpe.seperator_, self.bpe.byte_encoder_, False)
            return tokenizer_copy
        return self


class BERTTokenizer(Module):
    """
    Transform for BERT Tokenizer.

    Based on WordPiece algorithm introduced in paper:
    https://static.googleusercontent.com/media/research.google.com/ja//pubs/archive/37842.pdf

    The backend kernel implementation is taken and modified from https://github.com/LieluoboAi/radish.

    See PR https://github.com/pytorch/text/pull/1707 summary for more details.

    The below code snippet shows how to use the BERT tokenizer using the pre-trained vocab files.

    Example
        >>> from torchtext.transforms import BERTTokenizer
        >>> VOCAB_FILE = "https://huggingface.co/bert-base-uncased/resolve/main/vocab.txt"
        >>> tokenizer = BERTTokenizer(vocab_path=VOCAB_FILE, do_lower_case=True, return_tokens=True)
        >>> tokenizer("Hello World, How are you!") # single sentence input
        >>> tokenizer(["Hello World","How are you!"]) # batch input

    :param vocab_path: Path to pre-trained vocabulary file. The path can be either local or URL.
    :type vocab_path: str
    :param do_lower_case: Indicate whether to do lower case. (default: True)
    :type do_lower_case: Optional[bool]
    :param strip_accents: Indicate whether to strip accents. (default: None)
    :type strip_accents: Optional[bool]
    :param return_tokens: Indicate whether to return tokens. If false, returns corresponding token IDs as strings (default: False)
    :type return_tokens: bool
    :param never_split: Collection of tokens which will not be split during tokenization. (default: None)
    :type never_split: Optional[List[str]]
    """
    __jit_unused_properties__ = ['is_jitable']

    def __init__(self, vocab_path: 'str', do_lower_case: 'bool'=True, strip_accents: 'Optional[bool]'=None, return_tokens=False, never_split: 'Optional[List[str]]'=None) ->None:
        super().__init__()
        if never_split is None:
            never_split = []
        self.bert_model = BERTEncoderPyBind(get_asset_local_path(vocab_path, overwrite=True), do_lower_case, strip_accents, never_split)
        self._return_tokens = return_tokens
        self._vocab_path = vocab_path
        self._do_lower_case = do_lower_case
        self._strip_accents = strip_accents
        self._never_split = never_split

    @property
    def is_jitable(self):
        return isinstance(self.bert_model, torch._C.ScriptObject)

    @torch.jit.export
    def _encode(self, text: 'str') ->List[str]:
        """Encode text into a list of tokens IDs

        Args:
            text: An input text string.

        Returns:
            A list of token ids represents each sub-word

        For example:
            --> "Hello world!" --> token ids: [707, 5927, 11, 707, 68]
        """
        token_ids: 'List[int]' = self.bert_model.encode(text.strip())
        tokens_ids_str: 'List[str]' = [str(token_id) for token_id in token_ids]
        return tokens_ids_str

    @torch.jit.export
    def _batch_encode(self, text: 'List[str]') ->List[List[str]]:
        """Batch version of _encode i.e operate on list of str"""
        token_ids: 'List[List[int]]' = self.bert_model.batch_encode([t.strip() for t in text])
        tokens_ids_str: 'List[List[str]]' = [[str(t) for t in token_id] for token_id in token_ids]
        return tokens_ids_str

    @torch.jit.export
    def _tokenize(self, text: 'str') ->List[str]:
        """Tokenize text into a list of tokens

        Args:
            text: An input text string.

        Returns:
            A list of tokens (sub-words)

        For example:
            --> "Hello World!": ["Hello", "World", "!"]
        """
        return self.bert_model.tokenize(text.strip())

    @torch.jit.export
    def _batch_tokenize(self, text: 'List[str]') ->List[List[str]]:
        """Batch version of _tokenize i.e operate on list of str"""
        return self.bert_model.batch_tokenize([t.strip() for t in text])

    def forward(self, input: 'Any') ->Any:
        """
        :param input: Input sentence or list of sentences on which to apply tokenizer.
        :type input: Union[str, List[str]]
        :return: tokenized text
        :rtype: Union[List[str], List[List(str)]]
        """
        if torch.jit.isinstance(input, List[str]):
            tokens: 'List[List[str]]' = []
            if self._return_tokens:
                tokens = self._batch_tokenize(input)
            else:
                tokens = self._batch_encode(input)
            return tokens
        elif torch.jit.isinstance(input, str):
            if self._return_tokens:
                return self._tokenize(input)
            else:
                return self._encode(input)
        else:
            raise TypeError('Input type not supported')

    def __prepare_scriptable__(self):
        if not self.is_jitable:
            tokenizer_copy = deepcopy(self)
            tokenizer_copy.bert_model = torch.classes.torchtext.BERTEncoder(self._vocab_path, self._do_lower_case, self._strip_accents, self._never_split)
            return tokenizer_copy
        return self


class RegexTokenizer(Module):
    """
    Regex tokenizer for a string sentence that applies all regex replacements defined in patterns_list. It is backed by the `C++ RE2 regular expression engine <https://github.com/google/re2>`_ from Google.

    Args:
        patterns_list (List[Tuple[str, str]]): a list of tuples (ordered pairs) which contain the regex pattern string
        as the first element and the replacement string as the second element.

    Caveats
        - The RE2 library does not support arbitrary lookahead or lookbehind assertions, nor does it support backreferences. Look at the `docs <https://swtch.com/~rsc/regexp/regexp3.html#caveats>`_ here for more info.
        - The final tokenization step always uses spaces as separators. To split strings based on a specific regex pattern, similar to Python's `re.split <https://docs.python.org/3/library/re.html#re.split>`_, a tuple of ``('<regex_pattern>', ' ')`` can be provided.

    Example
        Regex tokenization based on ``(patterns, replacements)`` list.
            >>> import torch
            >>> from torchtext.transforms import RegexTokenizer
            >>> test_sample = 'Basic Regex Tokenization for a Line of Text'
            >>> patterns_list = [
                (r''', ' '  '),
                (r'"', '')]
            >>> reg_tokenizer = RegexTokenizer(patterns_list)
            >>> jit_reg_tokenizer = torch.jit.script(reg_tokenizer)
            >>> tokens = jit_reg_tokenizer(test_sample)
        Regex tokenization based on ``(single_pattern, ' ')`` list.
            >>> import torch
            >>> from torchtext.transforms import RegexTokenizer
            >>> test_sample = 'Basic.Regex,Tokenization_for+a..Line,,of  Text'
            >>> patterns_list = [
                (r'[,._+ ]+', r' ')]
            >>> reg_tokenizer = RegexTokenizer(patterns_list)
            >>> jit_reg_tokenizer = torch.jit.script(reg_tokenizer)
            >>> tokens = jit_reg_tokenizer(test_sample)
    """
    __jit_unused_properties__ = ['is_jitable']

    def __init__(self, patterns_list) ->None:
        super(RegexTokenizer, self).__init__()
        patterns = [pair[0] for pair in patterns_list]
        replacements = [pair[1] for pair in patterns_list]
        self.regex_tokenizer = RegexTokenizerPybind(patterns, replacements, False)

    @property
    def is_jitable(self):
        return not isinstance(self.regex_tokenizer, RegexTokenizerPybind)

    def forward(self, line: 'str') ->List[str]:
        """
        Args:
            lines (str): a text string to tokenize.

        Returns:
            List[str]: a token list after regex.
        """
        return self.regex_tokenizer.forward(line)

    def __prepare_scriptable__(self):
        """Return a JITable RegexTokenizer."""
        if not self.is_jitable:
            regex_tokenizer_copy = deepcopy(self)
            regex_tokenizer_copy.regex_tokenizer = torch.classes.torchtext.RegexTokenizer(self.regex_tokenizer.patterns_, self.regex_tokenizer.replacements_, False)
            return regex_tokenizer_copy
        return self


class Sequential(torch.nn.Sequential):
    """A container to host a sequence of text transforms."""

    def forward(self, input: 'Any') ->Any:
        """
        :param input: Input sequence or batch. The input type must be supported by the first transform in the sequence.
        :type input: `Any`
        """
        for module in self:
            input = module(input)
        return input


class MaskTransform(torch.nn.Module):
    """
    The transform chooses mask_prob% (example 15%) of the token positions at random for
    prediction.

    If the i-th token is chosen, we replace the i-th token with
    (1) the [MASK] token 80% of the time
    (2) a random token 10% of the time
    (3) the unchanged i-th token 10% of the time.

    Args:
        vocab_len (int): the length of the vocabulary, including special tokens such as [BOS], [PAD], [MASK]
        mask_idx (int): index assigned to mask token in vocabulary
        bos_idx (int): index assigned to beginning-of-sequence token in vocabulary
        pad_idx (int): index assigned to padding token in vocabulary
        mask_bos (bool): indicate whether beginning-of-sequence tokens are eligible for masking (default: False)
        mask_prob (float): probability that a token is chosen for replacement (default: 0.15)

    Example:
        >>> import torch
        >>> from torchtext.transforms import MaskTransform
        >>> sample_tokens = [
                ["[BOS]", "a", "b", "c", "d"],
                ["[BOS]", "a", "b", "[PAD]", "[PAD]"]
            ]
        >>> sample_token_ids = torch.tensor([
                [6, 0, 1, 2, 3], [6, 0, 1, 4, 4]
            ])
        >>> mask_transform = MaskTransform(
                vocab_len = 7,
                mask_idx = 4,
                bos_idx = 6,
                pad_idx = 5,
                mask_bos = False,
                mask_prob = 0.15
            )
        >>> masked_tokens, target_tokens, mask = mask_transform(sample_token_ids)
    """
    mask_mask_prob = 0.8
    rand_mask_prob = 0.1

    def __init__(self, vocab_len: 'int', mask_idx: 'int', bos_idx: 'int', pad_idx: 'int', mask_bos: 'bool'=False, mask_prob: 'float'=0.15):
        super().__init__()
        self.vocab_len = vocab_len
        self.mask_idx = mask_idx
        self.bos_idx = bos_idx
        self.pad_idx = pad_idx
        self.mask_prob = mask_prob
        self.mask_bos = mask_bos

    def forward(self, tokens: 'torch.Tensor') ->Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Applies mask to input tokens.

        Inputs:
            tokens: Tensor with token ids of shape (batch_size x seq_len). Includes token ids for special tokens such as [BOS] and [PAD]

        Outputs:
            masked_tokens: Tensor of tokens after masking has been applied
            target_tokens: Tensor of token values selected for masking
            mask: Tensor with same shape as input tokens (batch_size x seq_len)
                with masked tokens represented by a 1 and everything else as 0.
        """
        mask, mask_mask, rand_mask = self._generate_mask(tokens)
        masked_tokens = self._mask_input(tokens, mask_mask, self.mask_idx)
        masked_tokens = self._mask_input(masked_tokens, rand_mask, torch.randint_like(tokens, high=self.vocab_len))
        target_tokens = torch.masked_select(tokens, mask.bool())
        return masked_tokens, target_tokens, mask

    def _random_masking(self, tokens: 'torch.tensor', mask_prob: 'float') ->torch.Tensor:
        """
        Function to mask tokens randomly.

        Inputs:
            1) tokens: Tensor with token ids of shape (batch_size x seq_len). Includes token ids for special tokens such as [BOS] and [PAD]
            2) mask_prob: Probability of masking a particular token

        Outputs:
            mask: Tensor with same shape as input tokens (batch_size x seq_len)
                with masked tokens represented by a 1 and everything else as 0.
        """
        batch_size, seq_len = tokens.size()
        num_masked_per_seq = int(seq_len * mask_prob)
        mask = torch.zeros((batch_size, seq_len), dtype=torch.int)
        mask[:, :num_masked_per_seq] = 1
        for i in range(batch_size):
            mask[i] = mask[i, torch.randperm(seq_len)]
        return mask

    def _select_tokens_to_mask(self, tokens: 'torch.Tensor', mask_prob: 'float') ->torch.Tensor:
        mask = self._random_masking(tokens, mask_prob)
        if not self.mask_bos:
            mask *= (tokens != self.bos_idx).long()
        return mask

    def _generate_mask(self, tokens: 'torch.Tensor') ->Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mask = self._select_tokens_to_mask(tokens, self.mask_prob)
        mask *= (tokens != self.pad_idx).long()
        mask[0, 0] = 1 if not mask.byte().any() else mask[0, 0]
        probs = torch.rand_like(tokens, dtype=torch.float)
        mask_mask = (probs >= 1 - self.mask_mask_prob).long() * mask
        rand_mask = (probs < self.rand_mask_prob).long() * mask
        return mask, mask_mask, rand_mask

    def _mask_input(self, tokens: 'torch.Tensor', mask: 'torch.Tensor', replacement) ->torch.Tensor:
        return tokens * (1 - mask) + replacement * mask


class GloVe(Vectors):
    url = {'42B': 'http://nlp.stanford.edu/data/glove.42B.300d.zip', '840B': 'http://nlp.stanford.edu/data/glove.840B.300d.zip', 'twitter.27B': 'http://nlp.stanford.edu/data/glove.twitter.27B.zip', '6B': 'http://nlp.stanford.edu/data/glove.6B.zip'}

    def __init__(self, name='840B', dim=300, **kwargs) ->None:
        url = self.url[name]
        name = 'glove.{}.{}d.txt'.format(name, str(dim))
        super(GloVe, self).__init__(name, url=url, **kwargs)


class FastText(Vectors):
    url_base = 'https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.{}.vec'

    def __init__(self, language='en', **kwargs) ->None:
        url = self.url_base.format(language)
        name = os.path.basename(url)
        super(FastText, self).__init__(name, url=url, **kwargs)


class CharNGram(Vectors):
    name = 'charNgram.txt'
    url = 'http://www.logos.t.u-tokyo.ac.jp/~hassy/publications/arxiv2016jmt/jmt_pre-trained_embeddings.tar.gz'

    def __init__(self, **kwargs) ->None:
        super(CharNGram, self).__init__(self.name, url=self.url, **kwargs)

    def __getitem__(self, token):
        vector = torch.Tensor(1, self.dim).zero_()
        if token == '<unk>':
            return self.unk_init(vector)
        chars = ['#BEGIN#'] + list(token) + ['#END#']
        num_vectors = 0
        for n in [2, 3, 4]:
            end = len(chars) - n + 1
            grams = [chars[i:i + n] for i in range(end)]
            for gram in grams:
                gram_key = '{}gram-{}'.format(n, ''.join(gram))
                if gram_key in self.stoi:
                    vector += self.vectors[self.stoi[gram_key]]
                    num_vectors += 1
        if num_vectors > 0:
            vector /= num_vectors
        else:
            vector = self.unk_init(vector)
        return vector


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (InProjContainer,
     lambda: ([], {'query_proj': torch.nn.ReLU(), 'key_proj': torch.nn.ReLU(), 'value_proj': torch.nn.ReLU()}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (MaskTransform,
     lambda: ([], {'vocab_len': 4, 'mask_idx': 4, 'bos_idx': 4, 'pad_idx': 4}),
     lambda: ([torch.rand([4, 4])], {})),
    (PadTransform,
     lambda: ([], {'max_length': 4, 'pad_value': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (RobertaClassificationHead,
     lambda: ([], {'num_classes': 4, 'input_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ScaledDotProduct,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (Sequential,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (T5Layer,
     lambda: ([], {'d_model': 4, 'nhead': 4}),
     lambda: ([torch.rand([4, 4])], {})),
    (T5LayerNorm,
     lambda: ([], {'d_model': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (T5MultiheadAttention,
     lambda: ([], {'embed_dim': 4, 'num_heads': 4}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4]), torch.rand([4, 4])], {})),
    (TransformerEncoderLayer,
     lambda: ([], {'embedding_dim': 4, 'num_attention_heads': 4}),
     lambda: ([torch.rand([4, 4]), torch.rand([4])], {})),
]

