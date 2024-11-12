
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


import torch.distributed as dist


import torch.multiprocessing as mp


import torch.nn as nn


from torch.nn.utils.rnn import pack_padded_sequence


from torch.nn.utils.rnn import pad_packed_sequence


from typing import List


from typing import Tuple


from typing import Dict


from typing import Iterable


from typing import Set


from typing import Union


from typing import TYPE_CHECKING


from typing import Optional


from typing import Callable


from torch.nn.utils.rnn import PackedSequence


import copy


import torch.nn.functional as F


from typing import Any


from torch.cuda.amp import GradScaler


from torch.optim import Adam


from torch.optim import AdamW


from torch.optim import Optimizer


from torch.optim.lr_scheduler import ExponentialLR


from torch.optim.lr_scheduler import _LRScheduler


from torch.distributions.utils import lazy_property


import torch.autograd as autograd


from torch.distributions.distribution import Distribution


from torch.autograd import Function


import itertools


from functools import reduce


import queue


from collections import Counter


from collections import defaultdict


import functools


import re


def extract(path: 'str', reload: 'bool'=False, clean: 'bool'=False) ->str:
    extracted = path
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as f:
            extracted = os.path.join(os.path.dirname(path), f.infolist()[0].filename)
            if reload or not os.path.exists(extracted):
                f.extractall(os.path.dirname(path))
    elif tarfile.is_tarfile(path):
        with tarfile.open(path) as f:
            extracted = os.path.join(os.path.dirname(path), f.getnames()[0])
            if reload or not os.path.exists(extracted):
                f.extractall(os.path.dirname(path))
    elif path.endswith('.gz'):
        extracted = path[:-3]
        with gzip.open(path) as fgz:
            with open(extracted, 'wb') as f:
                shutil.copyfileobj(fgz, f)
    if clean:
        os.remove(path)
    return extracted


def gather(obj: 'Any') ->Iterable[Any]:
    objs = [None] * dist.get_world_size()
    dist.all_gather_object(objs, obj)
    return objs


def is_dist():
    return dist.is_available() and dist.is_initialized()


def is_master():
    return not is_dist() or dist.get_rank() == 0


def wait(fn) ->Any:

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        value = None
        if is_master():
            value = fn(*args, **kwargs)
        if is_dist():
            dist.barrier()
            value = gather(value)[0]
        return value
    return wrapper


@wait
def download(url: 'str', path: 'Optional[str]'=None, reload: 'bool'=False, clean: 'bool'=False) ->str:
    filename = os.path.basename(urllib.parse.urlparse(url).path)
    if path is None:
        path = CACHE
    os.makedirs(path, exist_ok=True)
    path = os.path.join(path, filename)
    if reload and os.path.exists(path):
        os.remove(path)
    if not os.path.exists(path):
        sys.stderr.write(f'Downloading {url} to {path}\n')
        try:
            torch.hub.download_url_to_file(url, path, progress=True)
        except (ValueError, urllib.error.URLError):
            raise RuntimeError(f'File {url} unavailable. Please try other sources.')
    return extract(path, reload, clean)


def pad(tensors: 'List[torch.Tensor]', padding_value: 'int'=0, total_length: 'int'=None, padding_side: 'str'='right') ->torch.Tensor:
    size = [len(tensors)] + [max(tensor.size(i) for tensor in tensors) for i in range(len(tensors[0].size()))]
    if total_length is not None:
        assert total_length >= size[1]
        size[1] = total_length
    out_tensor = tensors[0].data.new(*size).fill_(padding_value)
    for i, tensor in enumerate(tensors):
        out_tensor[i][[(slice(-i, None) if padding_side == 'left' else slice(0, i)) for i in tensor.size()]] = tensor
    return out_tensor


class SinusoidPositionalEmbedding(nn.Module):

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        seq_len, n_model = x[0].shape
        pos = x.new_tensor(range(seq_len)).unsqueeze(-1)
        pos = pos / 10000 ** (x.new_tensor(range(n_model)).div(2, rounding_mode='floor') * 2 / n_model)
        pos[:, 0::2], pos[:, 1::2] = pos[:, 0::2].sin(), pos[:, 1::2].cos()
        return pos


class SinusoidRelativePositionalEmbedding(nn.Module):

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        seq_len, n_model = x[0].shape
        pos = x.new_tensor(range(seq_len))
        pos = (pos - pos.unsqueeze(-1)).unsqueeze(-1)
        pos = pos / 10000 ** (x.new_tensor(range(n_model)).div(2, rounding_mode='floor') * 2 / n_model)
        pos[..., 0::2], pos[..., 1::2] = pos[..., 0::2].sin(), pos[..., 1::2].cos()
        return pos


class Model(nn.Module):

    def __init__(self, n_words, n_tags=None, n_chars=None, n_lemmas=None, encoder='lstm', feat=['char'], n_embed=100, n_pretrained=100, n_feat_embed=100, n_char_embed=50, n_char_hidden=100, char_pad_index=0, char_dropout=0, elmo_bos_eos=(True, True), elmo_dropout=0.5, bert=None, n_bert_layers=4, mix_dropout=0.0, bert_pooling='mean', bert_pad_index=0, finetune=False, n_plm_embed=0, embed_dropout=0.33, encoder_dropout=0.33, pad_index=0, **kwargs):
        super().__init__()
        self.args = Config().update(locals())
        if encoder == 'lstm':
            self.word_embed = nn.Embedding(num_embeddings=self.args.n_words, embedding_dim=self.args.n_embed)
            n_input = self.args.n_embed
            if self.args.n_pretrained != self.args.n_embed:
                n_input += self.args.n_pretrained
            if 'tag' in self.args.feat:
                self.tag_embed = nn.Embedding(num_embeddings=self.args.n_tags, embedding_dim=self.args.n_feat_embed)
                n_input += self.args.n_feat_embed
            if 'char' in self.args.feat:
                self.char_embed = CharLSTM(n_chars=self.args.n_chars, n_embed=self.args.n_char_embed, n_hidden=self.args.n_char_hidden, n_out=self.args.n_feat_embed, pad_index=self.args.char_pad_index, dropout=self.args.char_dropout)
                n_input += self.args.n_feat_embed
            if 'lemma' in self.args.feat:
                self.lemma_embed = nn.Embedding(num_embeddings=self.args.n_lemmas, embedding_dim=self.args.n_feat_embed)
                n_input += self.args.n_feat_embed
            if 'elmo' in self.args.feat:
                self.elmo_embed = ELMoEmbedding(n_out=self.args.n_plm_embed, bos_eos=self.args.elmo_bos_eos, dropout=self.args.elmo_dropout, finetune=self.args.finetune)
                n_input += self.elmo_embed.n_out
            if 'bert' in self.args.feat:
                self.bert_embed = TransformerEmbedding(name=self.args.bert, n_layers=self.args.n_bert_layers, n_out=self.args.n_plm_embed, pooling=self.args.bert_pooling, pad_index=self.args.bert_pad_index, mix_dropout=self.args.mix_dropout, finetune=self.args.finetune)
                n_input += self.bert_embed.n_out
            self.embed_dropout = IndependentDropout(p=self.args.embed_dropout)
            self.encoder = VariationalLSTM(input_size=n_input, hidden_size=self.args.n_encoder_hidden // 2, num_layers=self.args.n_encoder_layers, bidirectional=True, dropout=self.args.encoder_dropout)
            self.encoder_dropout = SharedDropout(p=self.args.encoder_dropout)
        elif encoder == 'transformer':
            self.word_embed = TransformerWordEmbedding(n_vocab=self.args.n_words, n_embed=self.args.n_embed, pos=self.args.pos, pad_index=self.args.pad_index)
            self.embed_dropout = nn.Dropout(p=self.args.embed_dropout)
            self.encoder = TransformerEncoder(layer=TransformerEncoderLayer(n_heads=self.args.n_encoder_heads, n_model=self.args.n_encoder_hidden, n_inner=self.args.n_encoder_inner, attn_dropout=self.args.encoder_attn_dropout, ffn_dropout=self.args.encoder_ffn_dropout, dropout=self.args.encoder_dropout), n_layers=self.args.n_encoder_layers, n_model=self.args.n_encoder_hidden)
            self.encoder_dropout = nn.Dropout(p=self.args.encoder_dropout)
        elif encoder == 'bert':
            self.encoder = TransformerEmbedding(name=self.args.bert, n_layers=self.args.n_bert_layers, pooling=self.args.bert_pooling, pad_index=self.args.pad_index, mix_dropout=self.args.mix_dropout, finetune=True)
            self.encoder_dropout = nn.Dropout(p=self.args.encoder_dropout)
            self.args.n_encoder_hidden = self.encoder.n_out

    def load_pretrained(self, embed=None):
        if embed is not None:
            self.pretrained = nn.Embedding.from_pretrained(embed)
            if embed.shape[1] != self.args.n_pretrained:
                self.embed_proj = nn.Linear(embed.shape[1], self.args.n_pretrained)
            nn.init.zeros_(self.word_embed.weight)
        return self

    def forward(self):
        raise NotImplementedError

    def loss(self):
        raise NotImplementedError

    def embed(self, words, feats=None):
        ext_words = words
        if hasattr(self, 'pretrained'):
            ext_mask = words.ge(self.word_embed.num_embeddings)
            ext_words = words.masked_fill(ext_mask, self.args.unk_index)
        word_embed = self.word_embed(ext_words)
        if hasattr(self, 'pretrained'):
            pretrained = self.pretrained(words)
            if self.args.n_embed == self.args.n_pretrained:
                word_embed += pretrained
            else:
                word_embed = torch.cat((word_embed, self.embed_proj(pretrained)), -1)
        feat_embed = []
        if 'tag' in self.args.feat:
            feat_embed.append(self.tag_embed(feats.pop()))
        if 'char' in self.args.feat:
            feat_embed.append(self.char_embed(feats.pop(0)))
        if 'elmo' in self.args.feat:
            feat_embed.append(self.elmo_embed(feats.pop(0)))
        if 'bert' in self.args.feat:
            feat_embed.append(self.bert_embed(feats.pop(0)))
        if 'lemma' in self.args.feat:
            feat_embed.append(self.lemma_embed(feats.pop(0)))
        if isinstance(self.embed_dropout, IndependentDropout):
            if len(feat_embed) == 0:
                raise RuntimeError(f'`feat` is not allowed to be empty, which is {self.args.feat} now')
            embed = torch.cat(self.embed_dropout(word_embed, torch.cat(feat_embed, -1)), -1)
        else:
            embed = word_embed
            if len(feat_embed) > 0:
                embed = torch.cat((embed, torch.cat(feat_embed, -1)), -1)
            embed = self.embed_dropout(embed)
        return embed

    def encode(self, words, feats=None):
        if self.args.encoder == 'lstm':
            x = pack_padded_sequence(self.embed(words, feats), words.ne(self.args.pad_index).sum(1).tolist(), True, False)
            x, _ = self.encoder(x)
            x, _ = pad_packed_sequence(x, True, total_length=words.shape[1])
        elif self.args.encoder == 'transformer':
            x = self.encoder(self.embed(words, feats), words.ne(self.args.pad_index))
        else:
            x = self.encoder(words)
        return self.encoder_dropout(x)

    def decode(self):
        raise NotImplementedError


NUL = '<nul>'

